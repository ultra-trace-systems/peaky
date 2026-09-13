"""Offline tests for the MCP server tool functions + job manager.

No network and NO `mcp` package required: the tool functions are plain Python
(FastMCP is only imported by build_server). IO is monkeypatched. Run:
    python3 tests/test_mcp_server.py
"""
import inspect
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from peaky import mcp_server as M  # noqa: E402
from peaky.batch import sampling as SS  # noqa: E402
from peaky.io import io_mascope as IO  # noqa: E402

PASS = FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok  {name}")
    else:
        FAIL += 1
        print(f"FAIL  {name}  {detail}")


# ---------- discovery tools (IO monkeypatched, no network) ----------
_orig = {}


def _patch(name, fn):
    _orig[name] = getattr(IO, name, None)
    setattr(IO, name, fn)


_patch("list_workspaces", lambda: pd.DataFrame({"workspace_name": ["System Workspace", "W2"]}))
_patch("connect", lambda **kw: ("CLIENT", kw.get("workspace")))
_patch("list_datasets", lambda c: pd.DataFrame({"dataset_name": ["A", "B", "C"]}))
_patch("list_batches", lambda c, d: pd.DataFrame(
    {"sample_batch_name": ["b1", "b2"], "polarity": ["+", "+-"], "status": ["ready", "ready"]}))
_patch("fetch_batch_samples", lambda c, b, dataset=None: pd.DataFrame({
    "sample_item_id": [f"s{i}" for i in range(120)],
    "sample_item_name": [f"n{i}" for i in range(120)],
    "datetime_utc": ["2026-06-07"] * 120, "tic": list(range(120)), "polarity": ["+"] * 120}))

check("list_workspaces returns names", M.list_workspaces()["workspaces"] == ["System Workspace", "W2"])
check("list_datasets returns names", M.list_datasets()["datasets"] == ["A", "B", "C"])
lb = M.list_batches("A")
check("list_batches returns rows", lb["n"] == 2 and lb["batches"][0]["sample_batch_name"] == "b1", lb)
ls = M.list_samples("A", "b1", limit=10)
check("list_samples caps rows but reports true total",
      ls["n_total"] == 120 and ls["n_shown"] == 10 and len(ls["samples"]) == 10, ls)

# ---------- certify_neutrals (offline, real CN module) ----------
from peaky.chem import chemistry as CH  # noqa: E402

_mzN0 = CH.ion_mz("C10H15NO2S", "[M+H]+")
_mzN1 = CH.ion_mz("C10H15NO2S", "[M+(CH4N2O)H]+")
_mzN2 = _mzN1 + 60.0323627601
led = pd.DataFrame({
    "peak_id": ["p0", "p1", "p2", "bg"],
    "mz": [_mzN0, _mzN1, _mzN2, 401.3337],
    "height": [50000.0, 400000.0, 3000.0, 900.0],
    "role": ["unexplained"] * 4, "neutral_formula": [None] * 4})
_tmp = Path(M._OUT_DEFAULT).parent / "_mcp_test_ledger.csv"
_tmp.parent.mkdir(parents=True, exist_ok=True)
led.to_csv(_tmp, index=False)
cn = M.certify_neutrals(str(_tmp), reagent="Ur")
check("certify_neutrals finds the NBBS urea ladder",
      cn["n_certificates"] >= 1
      and any(abs(c["core_mass"] - 213.0823) < 1e-3 for c in cn["certificates"]), cn)
check("certify_neutrals surfaces the off-grid C10H15NO2S candidate",
      any("C10H15NO2S" in c["offgrid_candidates"] for c in cn["certificates"]), cn)

def _wait_jobs(manager, jids, timeout: float = 30.0) -> bool:
    """Block until every job in `jids` is done or errored, or `timeout` seconds
    pass -- a wall-clock deadline, not a fixed number of 20 ms polls: a loaded
    CI runner is not this machine. Returns whether they all finished."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if all(manager.get(j).status in ("done", "error") for j in jids):
            return True
        time.sleep(0.02)
    return False


# ---------- JobManager lifecycle ----------
jm = M.JobManager()
jid = jm.submit("test", lambda log: (log("step1"), log("step2"), {"answer": 42})[-1], {"x": 1})
_wait_jobs(jm, [jid])
job = jm.get(jid)
check("job runs to done with result", job.status == "done" and job.result == {"answer": 42}, job.view())
check("job captures log lines", job.log[:2] == ["step1", "step2"], job.log)

jid2 = jm.submit("boom", lambda log: (_ for _ in ()).throw(RuntimeError("HTTP 403 attention required")), {})
_wait_jobs(jm, [jid2])
job2 = jm.get(jid2)
check("job error is captured with a friendly hint",
      job2.status == "error" and "WAF" in (job2.error or ""), job2.view())
check("job_status via manager returns a view", "job_id" in job2.view())

# assign_sample / run_batch return a job_id immediately. The work is NOT idle
# meanwhile: `JobManager.submit` starts it on a daemon thread the moment it
# returns, and both workers look their collaborator up at call time
# (`pipeline.run_batch`, `assign.run`). A job left running here reached the
# fakes the later sections install and overwrote the kwargs those sections
# read -- one check failed in three of three 3.13 CI attempts on one commit.
# So: stub both for this section's duration, write into a temp dir, and wait
# for both jobs before anything is restored.
import tempfile  # noqa: E402

from peaky import pipeline as _PLq  # noqa: E402
from peaky.assignment import assign as _Aq  # noqa: E402

_q_saved = (M.JOBS, _PLq.run_batch, _Aq.run)
_PLq.run_batch = lambda **kw: {"ctx": None}
_Aq.run = lambda *a, **kw: {"ledger": pd.DataFrame(), "stats": {}}
M.JOBS = M.JobManager()
try:
    with tempfile.TemporaryDirectory() as _qd:
        r = M.run_batch("some batch", dataset="A", reagent="Ur", output_dir=_qd)
        check("run_batch returns a queued job_id", "job_id" in r and r["status"] == "queued", r)
        check("run_batch job is registered", M.JOBS.get(r["job_id"]) is not None)
        a = M.assign_sample("sid1", reagent="Ur", output_dir=_qd)
        check("assign_sample returns a queued job_id", "job_id" in a, a)
        lj = M.list_jobs()
        check("list_jobs lists both", len(lj["jobs"]) == 2, lj)
        check("...and both jobs finish before this section hands the registry back",
              _wait_jobs(M.JOBS, [r["job_id"], a["job_id"]]),
              [M.JOBS.get(j["job_id"]).view() for j in (r, a)])
finally:
    M.JOBS, _PLq.run_batch, _Aq.run = _q_saved

# ---------- the batch tool actually FORWARDS its selection budget ----------
# The job runs for real here, with the pipeline stubbed, so a k_max that never
# left the tool signature is caught.
from types import SimpleNamespace  # noqa: E402
from peaky import pipeline as PL  # noqa: E402

_seen_kw = {}
_orig_run_batch = PL.run_batch


def _fake_run_batch(**kw):
    _seen_kw.update(kw)
    return {"ctx": SimpleNamespace(out_dir="/tmp/peaky-mcp-test", run_id="rid"),
            "report_pdf": None, "assign": {"merged_M0": 3}}


PL.run_batch = _fake_run_batch
_saved = M.JOBS
M.JOBS = M.JobManager()
try:
    rj = M.run_batch("some batch", dataset="A", reagent="Ur", k_max=7)
    _wait_jobs(M.JOBS, [rj["job_id"]])
    _job = M.JOBS.get(rj["job_id"])
    check("run_batch job completes with the pipeline stubbed",
          _job.status == "done", _job.view())
    check("run_batch forwards k_max to pipeline.run_batch",
          _seen_kw.get("k_max") == 7, _seen_kw)
    check("run_batch forwards the batch/dataset/reagent it was called with",
          (_seen_kw.get("batch"), _seen_kw.get("dataset"), _seen_kw.get("reagent"))
          == ("some batch", "A", "Ur"), _seen_kw)
    check("run_batch records k_max in the job's params (visible in job_status)",
          M.JOBS.get(rj["job_id"]).view()["params"]["k_max"] == 7,
          M.JOBS.get(rj["job_id"]).view().get("params"))
finally:
    PL.run_batch = _orig_run_batch
    M.JOBS = _saved

# ---------- and its DEFAULT is the sampling constant, not a retyped number ----
# The CLI takes --k-max from SS.K_MAX; the tool must not drift from it when the
# constant moves, and the docstring must name the constant rather than today's
# value (a literal in the docstring is what goes stale silently).
check("run_batch's k_max default tracks sampling.K_MAX",
      inspect.signature(M.run_batch).parameters["k_max"].default == SS.K_MAX,
      inspect.signature(M.run_batch).parameters["k_max"].default)
check("run_batch's docstring names the constant, not a literal default",
      "K_MAX" in (M.run_batch.__doc__ or "")
      and f"default {SS.K_MAX})" not in (M.run_batch.__doc__ or ""),
      M.run_batch.__doc__)

# ---------- the assign tool RESOLVES the profile's own gate multiple ----------
# The MCP path is one of the entry points that resolve the height-gate multiple
# where the profile is (profiles.apply_height_cutoff_x_edge); nothing else in the
# suite covers it, so deleting that call from assign_sample left everything green.
# Mirrors the cmd_assign checks in test_cli.py: register a profile carrying a
# RAISED multiple, stub the assignment run, and read the multiple off the cfg the
# run was handed. A missing resolution leaves the field None -> this FAILS (it
# does not error), which is what a revert should look like.
import tempfile  # noqa: E402

from peaky.assignment import assign as _A  # noqa: E402
from peaky.chem import profiles as _PR  # noqa: E402

_seen_assign = {}
_orig_assign_run = _A.run


def _fake_assign_run(sample_id, context="ambient-air", *, cfg=None, **kw):
    _seen_assign["cfg"] = cfg
    _seen_assign["context"] = context
    _seen_assign["adducts"] = kw.get("adducts")
    return {"ledger": pd.DataFrame({"peak_id": ["p0"], "mz": [100.0076],
                                    "height": [9.0], "role": ["M0"],
                                    "neutral_formula": ["C5H8O"],
                                    "adduct": ["[M-H]-"]}),
            "stats": {"n_peaks": 1}}


_pick = _PR.ReagentProfile(
    name="TofPickMCP", label="tof picker", polarity="-", adducts=["[M-H]-"],
    normaliser="tic", reagent_ion_re=None, ranges="C0-10 H0-20",
    detect_adduct=None, height_cutoff_x_edge=5.0)
_prof_snap = (dict(_PR.PROFILES), dict(_PR._BY_ALIAS))
_PR.register(_pick)
_A.run = _fake_assign_run
_saved = M.JOBS
M.JOBS = M.JobManager()
try:
    with tempfile.TemporaryDirectory() as _d:
        aj = M.assign_sample("sid1", reagent="TofPickMCP", output_dir=_d)
        _wait_jobs(M.JOBS, [aj["job_id"]])
        _ajob = M.JOBS.get(aj["job_id"])
        check("assign_sample job completes with assign.run stubbed",
              _ajob.status == "done", _ajob.view())
        _acfg = _seen_assign.get("cfg")
        check("assign_sample takes the gate multiple from the reagent profile",
              _acfg is not None and _acfg.height_cutoff_x_edge == 5.0,
              vars(_acfg) if _acfg is not None else _ajob.view())
        check("assign_sample's absolute height_cutoff stays the cps override",
              _acfg is not None and _acfg.height_cutoff_cps is None, _acfg)
        _seen_assign.clear()
        aj2 = M.assign_sample("sid1", reagent="TofPickMCP", height_cutoff=250.0,
                              output_dir=_d)
        _wait_jobs(M.JOBS, [aj2["job_id"]])
        _acfg2 = _seen_assign.get("cfg")
        check("assign_sample(height_cutoff=) gates absolutely, multiple inert",
              _acfg2 is not None and _acfg2.height_cutoff_cps == 250.0
              and _acfg2.height_cutoff == 250.0,
              vars(_acfg2) if _acfg2 is not None else None)
finally:
    _A.run = _orig_assign_run
    M.JOBS = _saved
    _PR.PROFILES.clear(); _PR.PROFILES.update(_prof_snap[0])
    _PR._BY_ALIAS.clear(); _PR._BY_ALIAS.update(_prof_snap[1])

# ---------- build_server degrades cleanly without the mcp package ----------
try:
    import mcp.server.fastmcp  # noqa: F401
    _have_mcp = True
except ImportError:
    _have_mcp = False
if _have_mcp:
    srv = M.build_server()
    check("build_server registers all tools (mcp installed)", srv is not None)
else:
    try:
        M.build_server()
        check("build_server raises a helpful ImportError without mcp", False)
    except ImportError as e:
        check("build_server raises a helpful ImportError without mcp",
              "mascope-peaky[mcp]" in str(e))

# restore IO
for k, v in _orig.items():
    if v is not None:
        setattr(IO, k, v)
try:
    _tmp.unlink()
except OSError:
    pass


def test_all():
    assert FAIL == 0, f"{FAIL} checks failed"


if __name__ == "__main__":
    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)
