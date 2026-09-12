"""Offline tests for the MCP server tool functions + job manager.

No network and NO `mcp` package required: the tool functions are plain Python
(FastMCP is only imported by build_server). IO is monkeypatched. Run:
    python3 tests/test_mcp_server.py
"""
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from peaky import mcp_server as M  # noqa: E402
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

# ---------- JobManager lifecycle ----------
jm = M.JobManager()
jid = jm.submit("test", lambda log: (log("step1"), log("step2"), {"answer": 42})[-1], {"x": 1})
for _ in range(50):
    if jm.get(jid).status in ("done", "error"):
        break
    time.sleep(0.02)
job = jm.get(jid)
check("job runs to done with result", job.status == "done" and job.result == {"answer": 42}, job.view())
check("job captures log lines", job.log[:2] == ["step1", "step2"], job.log)

jid2 = jm.submit("boom", lambda log: (_ for _ in ()).throw(RuntimeError("HTTP 403 attention required")), {})
for _ in range(50):
    if jm.get(jid2).status in ("done", "error"):
        break
    time.sleep(0.02)
job2 = jm.get(jid2)
check("job error is captured with a friendly hint",
      job2.status == "error" and "WAF" in (job2.error or ""), job2.view())
check("job_status via manager returns a view", "job_id" in job2.view())

# assign_sample / run_batch return a job_id immediately (work not run here)
_saved = M.JOBS
M.JOBS = M.JobManager()
r = M.run_batch("some batch", dataset="A", reagent="Ur")
check("run_batch returns a queued job_id", "job_id" in r and r["status"] == "queued", r)
check("run_batch job is registered", M.JOBS.get(r["job_id"]) is not None)
a = M.assign_sample("sid1", reagent="Ur")
check("assign_sample returns a queued job_id", "job_id" in a, a)
lj = M.list_jobs()
check("list_jobs lists both", len(lj["jobs"]) == 2, lj)
M.JOBS = _saved

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
    for _ in range(100):
        if M.JOBS.get(rj["job_id"]).status in ("done", "error"):
            break
        time.sleep(0.02)
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
