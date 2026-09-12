"""Peaky as an MCP server — drive the pipeline from any MCP client (ChatGPT
Developer Mode, Claude Desktop, Cursor, ...).

WHY THIS SHAPE. MCP is the OUTER boundary (client <-> peaky): only small things
cross it — a batch name, a summary, a file path. `io_mascope` stays the INNER
boundary (peaky <-> Mascope): a direct in-process HTTP client whose big peak
tables NEVER touch the model context. Credentials live server-side in .env and
are never sent to the client. So this server MUST run host-side (network + .env
+ mascope-sdk) — it cannot run inside a client's sandbox.

The tool FUNCTIONS below are plain Python (no `mcp` import), so they are unit-
testable and the module imports without the optional `mcp` dependency. Only
`build_server()` / `serve()` import FastMCP (extra: `pip install mascope-peaky[mcp]`).

Long calls (assign a sample ~minutes, a batch ~many minutes) do not block an
MCP request: `assign_sample` / `run_batch` return a job_id immediately and run
on a background thread; poll `job_status`. The job registry is in-memory (not
persistent across a server restart) — fine for interactive use.
"""
from __future__ import annotations

import json
import os
import threading
import traceback
import uuid
from dataclasses import dataclass, field

# the selection budget is the sampling constant, not a retyped number (the CLI's
# --k-max default comes from the same place). A tool signature's default binds at
# def time, so this one import sits at module level; everything else in this
# module is imported lazily inside the tool bodies.
from peaky.batch.sampling import K_MAX as _K_MAX

__all__ = [
    "health", "list_workspaces", "list_datasets", "list_batches", "list_samples",
    "certify_neutrals", "assign_sample", "run_batch", "job_status", "list_jobs",
    "JobManager", "JOBS", "build_server", "serve",
]

# default host-side output root (mirrors the CLI / `peaky setup`)
_OUT_DEFAULT = os.environ.get("PEAKY_OUTPUT_DIR") or os.path.expanduser("~/peaky-output")


# --------------------------------------------------------------------------- #
# background jobs
# --------------------------------------------------------------------------- #
@dataclass
class Job:
    id: str
    kind: str
    params: dict
    status: str = "queued"        # queued | running | done | error
    log: list = field(default_factory=list)
    result: dict | None = None
    error: str | None = None

    def view(self, *, log_tail: int = 25) -> dict:
        return {
            "job_id": self.id, "kind": self.kind, "status": self.status,
            "params": self.params, "log_tail": self.log[-log_tail:],
            "result": self.result, "error": self.error,
        }


class JobManager:
    """Thread-backed in-memory job registry. `submit(kind, fn, params)` runs
    `fn(log)` on a daemon thread, where `log` appends a progress line; the
    function's return value becomes the job result."""

    def __init__(self):
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def submit(self, kind: str, fn, params: dict) -> str:
        jid = uuid.uuid4().hex[:12]
        job = Job(id=jid, kind=kind, params=dict(params))
        with self._lock:
            self._jobs[jid] = job

        def _run():
            job.status = "running"
            try:
                job.result = fn(lambda m: job.log.append(str(m)))
                job.status = "done"
            except Exception as e:  # noqa: BLE001 — surface any pipeline failure
                job.error = _friendly_error(e)
                job.log.append(f"ERROR: {job.error}")
                job.log.append(traceback.format_exc().splitlines()[-1])
                job.status = "error"

        threading.Thread(target=_run, name=f"peaky-job-{jid}", daemon=True).start()
        return jid

    def get(self, jid: str) -> Job | None:
        with self._lock:
            return self._jobs.get(jid)

    def all(self) -> list[Job]:
        with self._lock:
            return list(self._jobs.values())


JOBS = JobManager()


def _friendly_error(e: Exception) -> str:
    """Map a raw SDK/HTTP error to an actionable hint (mirrors cli._friendly_server_error)."""
    s = f"{type(e).__name__}: {e}"
    low = s.lower()
    if "403" in low or "attention required" in low or "cloudflare" in low:
        return (s + " — Mascope Cloudflare WAF rate-limit; wait 15-30 min with NO "
                "traffic, then retry (polling extends the block).")
    if "401" in low or "unauthorized" in low:
        return s + " — MASCOPE_ACCESS_TOKEN likely expired; refresh it in .env."
    if "404" in low or "not found" in low or "no peaks" in low:
        return s + " — not found; IDs go stale when a server copy is renamed, re-list."
    return s


# --------------------------------------------------------------------------- #
# read-only discovery tools (quick, synchronous)
# --------------------------------------------------------------------------- #
def _connect(workspace: str = ""):
    from peaky.io import io_mascope as IO
    return IO.connect(workspace=workspace or None)


def health() -> dict:
    """Check credentials and Mascope reachability. Returns creds presence + a
    live workspace count (no heavy data)."""
    have = bool(os.environ.get("MASCOPE_URL")) and bool(os.environ.get("MASCOPE_ACCESS_TOKEN"))
    from peaky.io import io_mascope as IO
    env = None
    if not have:
        try:
            env = IO._find_env()
            have = os.path.exists(env)
        except Exception:  # noqa: BLE001
            have = False
    out = {"credentials_found": have, "env_path": env, "output_dir": _OUT_DEFAULT}
    if not have:
        out["hint"] = ("No Mascope creds. Set MASCOPE_URL + MASCOPE_ACCESS_TOKEN in "
                       "the server's environment or ~/.mascope/.env.")
        return out
    try:
        ws = IO.list_workspaces()
        out["reachable"] = True
        out["n_workspaces"] = int(len(ws))
    except Exception as e:  # noqa: BLE001
        out["reachable"] = False
        out["error"] = _friendly_error(e)
    return out


def list_workspaces() -> dict:
    """List the Mascope workspaces the token can see."""
    from peaky.io import io_mascope as IO
    ws = IO.list_workspaces()
    col = "workspace_name" if "workspace_name" in ws.columns else ws.columns[0]
    return {"n": int(len(ws)), "workspaces": [str(v) for v in ws[col].tolist()]}


def list_datasets(workspace: str = "") -> dict:
    """List datasets in a workspace (default: the token's default workspace)."""
    from peaky.io import io_mascope as IO
    ds = IO.list_datasets(_connect(workspace))
    col = "dataset_name" if "dataset_name" in ds.columns else ds.columns[0]
    return {"workspace": workspace or "(default)", "n": int(len(ds)),
            "datasets": [str(v) for v in ds[col].tolist()]}


def list_batches(dataset: str, workspace: str = "") -> dict:
    """List sample-batches in a dataset (name / polarity / status)."""
    from peaky.io import io_mascope as IO
    bs = IO.list_batches(_connect(workspace), dataset)
    cols = [c for c in ("sample_batch_name", "polarity", "status") if c in bs.columns]
    return {"dataset": dataset, "n": int(len(bs)),
            "batches": bs[cols].to_dict("records") if cols else bs.to_dict("records")}


def list_samples(dataset: str, batch: str, workspace: str = "", limit: int = 50) -> dict:
    """List samples in a batch. Capped at `limit` rows (batches can hold
    thousands) — the count is always exact; increase `limit` to see more."""
    from peaky.io import io_mascope as IO
    sl = IO.fetch_batch_samples(_connect(workspace), batch, dataset=dataset)
    cols = [c for c in ("sample_item_id", "sample_item_name", "datetime_utc",
                        "tic", "polarity") if c in sl.columns]
    view = sl[cols] if cols else sl
    return {"dataset": dataset, "batch": batch, "n_total": int(len(sl)),
            "n_shown": int(min(limit, len(sl))),
            "samples": view.head(limit).to_dict("records")}


# --------------------------------------------------------------------------- #
# offline analysis tool (fast, synchronous, no network)
# --------------------------------------------------------------------------- #
def certify_neutrals(ledger_csv: str, reagent: str = "auto",
                     ts_parquet: str = "", tol_mda: float = 3.0,
                     min_channels: int = 2) -> dict:
    """Certified-neutral discovery over an existing ledger CSV (OFFLINE — pure
    mass domain, no server). Returns the certificate table: multi-channel /
    cluster-ladder converged cores + their off-grid (P/S/Cl) candidate formulas.
    """
    import pandas as pd
    from peaky.assignment import certified_neutral as CN
    from peaky.chem import contexts as XC
    from peaky.chem import profiles as PR

    led = pd.read_csv(os.path.expanduser(ledger_csv), low_memory=False)
    if "role" in led.columns:
        un = led[led["role"] == "unexplained"]
    else:
        fcol = "neutral_formula" if "neutral_formula" in led.columns else None
        un = led[led[fcol].isna()] if fcol else led
    if "peak_id" not in un.columns:
        un = un.assign(peak_id=[f"p{i}" for i in range(len(un))])
    if "height" not in un.columns:
        un = un.assign(height=1.0)

    rp = PR.resolve(reagent, peaks=un if reagent == "auto" else None)
    profile = XC.get_context(rp.context)
    cluster_reagent = "urea" if str(rp.polarity) in ("+", "positive") else None
    offsets = CN.channel_offsets(list(rp.adducts), cluster_reagent)
    certs = CN.find_certificates(un[["peak_id", "mz", "height"]].reset_index(drop=True),
                                 offsets, tol_mda=tol_mda, min_channels=min_channels)
    ts = pd.read_parquet(os.path.expanduser(ts_parquet)) if ts_parquet else None
    rows = []
    for c in certs:
        forms = [f for f in CN.enumerate_certified(c.core_mass, profile, force=True)
                 if any(el in f for el in ("P", "S", "Cl"))]
        cov = CN.ts_covariation(ts, [h.mz for h in c.hits]) if ts is not None else None
        rows.append({
            "core_mass": round(c.core_mass, 5), "n_channels": c.n_channels,
            "spread_mDa": round(c.spread_mda, 3),
            "member_mzs": [round(h.mz, 4) for h in c.hits],
            "ts_covary_rmin": None if cov is None else round(cov, 2),
            "offgrid_candidates": forms[:8],
        })
    rows.sort(key=lambda r: (-r["n_channels"], r["core_mass"]))
    return {"reagent": rp.label, "n_unexplained": int(len(un)),
            "n_certificates": len(rows), "certificates": rows}


# --------------------------------------------------------------------------- #
# long-running pipeline tools (background jobs)
# --------------------------------------------------------------------------- #
def assign_sample(sample_id: str, reagent: str = "auto", context: str = "",
                  height_cutoff: float | None = None, output_dir: str = "") -> dict:
    """Assign one sample (multi-pass). Returns a job_id immediately; poll
    `job_status`. On completion the result carries the assignment counts, top
    species, and the written ledger CSV path. `height_cutoff` is an ABSOLUTE cps
    override of the height-gated passes; default = a multiple of the sample's own
    noise edge (instrument-independent) — the reagent profile's own multiple when
    it carries one, else the package default (1x)."""
    out_dir = os.path.expanduser(output_dir or os.path.join(_OUT_DEFAULT, "mcp-assign"))

    def work(log):
        from peaky.assignment import assign, passes
        from peaky.chem import profiles
        os.makedirs(out_dir, exist_ok=True)
        rp = profiles.resolve(reagent) if reagent != "auto" else None
        adducts = list(rp.adducts) if rp else None
        ctx = context or (rp.context if rp else "ambient-air")
        cfg = passes.PassConfig(height_cutoff_cps=height_cutoff)
        # relative gate: the profile's own multiple of the sample's noise edge
        # when it carries one, else the package default (logged once).
        profiles.apply_height_cutoff_x_edge(cfg, rp, log=log)
        res = assign.run(sample_id, ctx, cfg=cfg, adducts=adducts, log=log,
                         label_purity=getattr(rp, "purity", None))
        led = res["ledger"]
        path = os.path.join(out_dir, f"{sample_id}_ledger.csv")
        led.to_csv(path, index=False)
        roles = led["role"].value_counts().to_dict() if "role" in led.columns else {}
        top = []
        if "role" in led.columns and "neutral_formula" in led.columns:
            m0 = led[led["role"] == "M0"].sort_values("height", ascending=False)
            top = m0.head(10)[["mz", "neutral_formula", "adduct"]].to_dict("records")
        return {"ledger_csv": path, "context": ctx,
                "roles": {k: int(v) for k, v in roles.items()},
                "top_species": top, "stats": res.get("stats", {})}

    jid = JOBS.submit("assign_sample", work,
                      {"sample_id": sample_id, "reagent": reagent, "output_dir": out_dir})
    return {"job_id": jid, "status": "queued",
            "note": "single-sample assign runs ~minutes; poll job_status(job_id)."}


def run_batch(batch: str, dataset: str = "", reagent: str = "auto",
              k_max: int = _K_MAX, subject: str = "",
              output_dir: str = "") -> dict:
    """Run the whole-batch pipeline (assign the presence-cover subset -> merge ->
    cluster -> Van Krevelen -> PDF). Returns a job_id immediately; poll
    `job_status`. On completion the result carries the versioned run folder + the
    PDF/merged-ledger paths and the batch summary (incl. `selection`: achieved
    coverage + stop reason). `k_max` is the sample budget (default
    `sampling.K_MAX`)."""
    base_out = os.path.expanduser(output_dir or _OUT_DEFAULT)

    def work(log):
        from peaky import pipeline as PL
        res = PL.run_batch(batch=batch, dataset=dataset or None, reagent=reagent,
                           base_out=base_out, k_max=k_max,
                           subject=subject or None, log=log)
        ctx = res.get("ctx")
        run_dir = getattr(ctx, "out_dir", None)
        return {"run_id": getattr(ctx, "run_id", None),
                "run_dir": run_dir,
                "report_pdf": res.get("report_pdf"),
                "merged_ledger": (os.path.join(run_dir, "merged_ledger.csv")
                                  if run_dir else None),
                "assign_summary": {k: v for k, v in (res.get("assign") or {}).items()
                                   if isinstance(v, (int, float, str, bool))}}

    jid = JOBS.submit("run_batch", work,
                      {"batch": batch, "dataset": dataset, "reagent": reagent,
                       "k_max": k_max, "output_dir": base_out})
    return {"job_id": jid, "status": "queued",
            "note": "batch pipeline runs many minutes; poll job_status(job_id)."}


def job_status(job_id: str, log_tail: int = 25) -> dict:
    """Status of a background job (queued/running/done/error) + recent log +
    result/paths when finished."""
    job = JOBS.get(job_id)
    if job is None:
        return {"error": f"no job {job_id!r}"}
    return job.view(log_tail=log_tail)


def list_jobs() -> dict:
    """Recent jobs and their statuses."""
    return {"jobs": [{"job_id": j.id, "kind": j.kind, "status": j.status,
                      "params": j.params} for j in JOBS.all()]}


# --------------------------------------------------------------------------- #
# MCP wiring (imports the optional `mcp` package only here)
# --------------------------------------------------------------------------- #
_TOOLS = [health, list_workspaces, list_datasets, list_batches, list_samples,
          certify_neutrals, assign_sample, run_batch, job_status, list_jobs]


def build_server(name: str = "peaky"):
    """Create a FastMCP server with every tool registered. Requires the optional
    `mcp` dependency (`pip install mascope-peaky[mcp]`)."""
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as e:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "The MCP server needs the `mcp` package. Install it with:\n"
            "    pip install 'mascope-peaky[mcp]'") from e
    server = FastMCP(name)
    for fn in _TOOLS:
        server.tool()(fn)
    return server


def serve(host: str = "127.0.0.1", port: int = 8765,
          transport: str = "streamable-http") -> None:
    """Run the peaky MCP server. `streamable-http` (default) is what ChatGPT
    Developer Mode connectors speak; `sse` and `stdio` are also supported (stdio
    for local clients like Claude Desktop)."""
    server = build_server()
    server.settings.host = host
    server.settings.port = port
    server.run(transport=transport)
