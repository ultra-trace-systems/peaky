"""Progress-window tests — no network, no display, no Tk.

THE POINT OF THIS FILE is the contract test in section 3. `peaky/progress.py`
reads the run's LOG STREAM (that is its only coupling to the pipeline), so a
reworded log line in assign_batch/assign/pipeline would silently flat-line the
progress bar with nothing failing anywhere. Section 3 pins the literal f-strings
those modules emit against the patterns progress.py parses, so a rewording
breaks a TEST instead of the window.

Run: python3 tests/test_progress.py   (or `pytest tests/test_progress.py`)
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PASS = FAIL = 0
def check(name, cond, detail=""):
    global PASS, FAIL
    if cond: PASS += 1; print(f"  ok  {name}")
    else: FAIL += 1; print(f"FAIL  {name}  {detail}")


from peaky import progress as PG  # noqa: E402

PKG = Path(PG.__file__).resolve().parent


# ---- 1. the parser: log lines -> progress model -----------------------------
st = PG.ProgressState(title="t")
check("starts with no samples known", st.n_samples == 0 and st.sample_frac == 0.0)
check("no ETA before a sample completes", st.eta is None)

check("serial 'assigning' line parses",
      st.feed("[assign_batch] (1/6) assigning aBcD1234 ..."), st.snapshot())
check("  -> n_samples + current sample", st.n_samples == 6 and st.current_sid == "aBcD1234")
check("  -> phase becomes assign", st.phase == "assign")

for i, name in enumerate(("pass0", "pass1", "calibrate"), 1):
    st.feed(f"[run] {name} took 1.4s")
check("stage lines advance the stage bar", st.stage_idx == 3 and st.stage_name == "calibrate")

check("'done' line advances samples", st.feed("[assign_batch] (1/6) done aBcD1234"))
check("  -> samples_done", st.samples_done == 1 and st.sample_frac == 1 / 6)
check("  -> real stage count LEARNED from the first sample", st.n_stages == 3)
check("  -> stage bar resets for the next sample", st.stage_idx == 0)
check("ETA exists once a sample has completed", st.eta is not None)

# parallel: worker logs are replayed AFTER the reduce, so stage lines are not live
par = PG.ProgressState(title="t")
par.feed("[assign_batch] (1/6) assigning x ...")
par.feed("[assign_batch] parallel: 5 worker processes (match-workers/proc=2) over 6 samples")
par.feed("[run] pass0 took 1.0s")
par.feed("[run] pass1 took 1.0s")
check("parallel banner parses", par.parallel == 5)
check("parallel mode does NOT drive the stage bar (replayed logs would lie)",
      par.stage_idx == 0 and par.stage_frac == 0.0)

ph = PG.ProgressState(title="t")
check("phase marker parses", ph.feed("[phase] cluster") and ph.phase == "cluster")
check("run-dir line parses",
      ph.feed("[batch] Sample-run_2026-09-07T101500Z -> /out/Sample-run_2026-09-07T101500Z")
      and ph.out_dir.endswith("Z"))
check("unknown lines are kept as the activity tail, not errors",
      ph.feed("CLUSTERING 412 organic ion-channels") is False
      and "CLUSTERING" in ph.last_line)
check("empty line is a no-op", ph.feed("   ") is False)

check("finished pins the sample bar at 100%",
      (lambda s: (s.__setattr__("finished", True), s.sample_frac)[1])(
          PG.ProgressState(title="t")) == 1.0)


# ---- 2. stats panel: both shapes a run returns ------------------------------
batch_rows = dict(PG.summary_rows(
    {"merged_M0": 812, "merged_tiers": {"Assigned": 700, "Candidate": 112},
     "n_files": 6, "n_in_all_files": 501, "n_single_file": 44,
     "formula_disagreements": 3, "elapsed_s": 754.0}, elapsed=902.0))
check("batch stats: merged M0", batch_rows["merged M0"] == "812")
check("batch stats: tiers rendered", "Assigned 700" in batch_rows["tiers"])
check("batch stats: assignment time from the summary",
      batch_rows["assignment"] == "12:34", batch_rows)
check("batch stats: total runtime is the window's own clock, not the summary's",
      batch_rows["total runtime"] == "15:02", batch_rows)
check("no total-runtime row while a run is still going",
      "total runtime" not in dict(PG.summary_rows({"elapsed_s": 754.0})))

eta_st = PG.ProgressState(title="t")
eta_st.feed("[assign_batch] (3/3) assigning s3 ...")
eta_st.feed("[assign_batch] (3/3) done s3")
check("no ETA once every sample is in (the report tail is not modelled)",
      eta_st.eta is None)
check("sample id survives between samples (it names the last one completed)",
      eta_st.current_sid == "s3")
eta_st.feed("[assign_batch] DONE: 9 merged M0 ({}); 1 in all files, 0 single-file, 0 formula disagreements")
check("sample id cleared once assignment as a whole ends", eta_st.current_sid == "")
sid_st = PG.ProgressState(title="t")
sid_st.feed("[assign_batch] (1/2) assigning s1 ...")
sid_st.feed("[phase] cluster")
check("sample id cleared on a post-assignment phase", sid_st.current_sid == "")

single_rows = dict(PG.summary_rows(
    {"n_peaks": 1000, "by_role": {"M0": 300, "iso_child": 100, "reagent": 50,
                                  "unexplained": 550},
     "signal_by_role": {"M0": 0.5, "iso_child": 0.1, "reagent": 0.3},
     "count_frac_by_role": {"unexplained": 0.55}}, elapsed=61.0))
check("single-sample stats detected via by_role", "assigned M0" in single_rows)
check("single-sample: peaks explained", single_rows["peaks explained"] == "45.0%")
check("single-sample: signal explained", single_rows["signal explained"] == "90.0%")
check("single-sample: runtime", single_rows["total runtime"] == "01:01")

check("empty summary renders placeholders, never raises",
      all(v for _, v in PG.summary_rows(None)))
check("clock formats", (PG._hms(92), PG._hms(3730), PG._hms(None))
      == ("01:32", "1:02:10", "--:--"))


# ---- 3. THE CONTRACT: the pipeline's literal log strings still parse ---------
# Read the emitting source and reconstruct the line the f-string produces. If
# someone rewords a log line, the corresponding check here fails.
def emits(relpath: str, needle: str) -> bool:
    """Is `needle` present verbatim in the module that is supposed to emit it?"""
    return needle in (PKG / relpath).read_text()

check("assign_batch emits the serial 'assigning' line progress.py parses",
      emits("batch/assign_batch.py",
            'log(f"[assign_batch] ({i}/{len(sample_ids)}) assigning {sid} ...")'))
check("assign_batch emits the 'done' line progress.py parses",
      emits("batch/assign_batch.py",
            '''log(f"[assign_batch] ({done}/{len(sample_ids)}) done {out['sid']}")'''))
check("assign_batch emits the parallel banner progress.py parses",
      emits("batch/assign_batch.py", 'log(f"[assign_batch] parallel: {n_jobs} worker processes '))
check("assign_batch emits the DONE line progress.py parses",
      emits("batch/assign_batch.py", 'log(f"[assign_batch] DONE: {summary[\'merged_M0\']} merged M0 '))
check("assign emits the per-stage timing line progress.py parses",
      emits("assignment/assign.py", 'st.log(f"[run] {tag} took {s[\'elapsed_s\']}s")'))
for ph_name in ("cluster", "vankrevelen", "report", "assign", "fetch", "provenance"):
    check(f"pipeline emits [phase] {ph_name}", emits("pipeline.py", f'log("[phase] {ph_name}")'))
    check(f"  -> progress.py knows the phase {ph_name}", ph_name in PG.PHASES)

# and the reconstructed lines really do match the patterns
for line, rx, what in [
    ("[assign_batch] (2/7) assigning kZ9 ...", PG.RE_ASSIGNING, "assigning"),
    ("[assign_batch] (2/7) done kZ9", PG.RE_SAMPLE_DONE, "done"),
    ("[assign_batch] parallel: 5 worker processes (match-workers/proc=2) over 7 samples",
     PG.RE_PARALLEL, "parallel"),
    ("[assign_batch] DONE: 812 merged M0 ({'Assigned': 700}); 501 in all files, "
     "44 single-file, 3 formula disagreements", PG.RE_ASSIGN_DONE, "DONE"),
    ("[run] cleanup took 12.3s", PG.RE_STAGE, "stage"),
    ("[phase] cluster", PG.RE_PHASE, "phase"),
]:
    check(f"pattern matches a real {what} line", bool(rx.match(line)), line)

# batch-level timing must actually reach the summary the window reads
src = (PKG / "batch/assign_batch.py").read_text()
check("assign_batch records elapsed_s in batch_summary", '"elapsed_s": round(time.time() - t_start, 1)' in src)
check("assign_batch records n_jobs beside it (a duration needs its job count)",
      '"n_jobs": n_jobs,' in src)
check("run_batch returns a whole-pipeline elapsed_s",
      '"elapsed_s": elapsed' in (PKG / "pipeline.py").read_text())


# ---- 4. never fatal: the Reporter is a transparent log wrapper ---------------
seen = []
rep = PG.Reporter("t", log=seen.append, ui=None)
rep("[assign_batch] (1/2) assigning s1 ...")
rep("some other line")
check("reporter forwards every line to the real log unchanged",
      seen == ["[assign_batch] (1/2) assigning s1 ...", "some other line"], seen)
check("reporter still advanced the model", rep.state.n_samples == 2)

class _Boom:
    def push(self, snap): raise RuntimeError("ui exploded")
    def close(self, wait): raise RuntimeError("ui exploded")
    def alive(self): return True
    def start(self, timeout=0): return True

boom = PG.Reporter("t", log=seen.append, ui=_Boom())
try:
    boom("[assign_batch] (1/2) done s1")
    boom.finish({"merged_M0": 1})
    boom.close()
    ok = True
except Exception as e:      # noqa: BLE001
    ok = False
check("a UI that raises on every call cannot break the run", ok)

def _bad_log(*a, **k): raise RuntimeError("log exploded")
try:
    PG.Reporter("t", log=_bad_log, ui=None)("x")
    ok = True
except Exception:
    ok = False
check("even a broken underlying log is swallowed", ok)

with PG.Reporter("t", log=seen.append, ui=_Boom()) as r:
    pass
check("context manager exits cleanly", True)

try:
    with PG.Reporter("t", log=seen.append, ui=_Boom()) as r:
        raise ValueError("run blew up")
except ValueError:
    caught = True
else:
    caught = False
check("an exception in the run still propagates (window must not swallow it)", caught)
check("  -> and is recorded for the window to display", "ValueError" in r.state.error)


# ---- 5. enable/disable + headless behaviour ---------------------------------
import os  # noqa: E402

os.environ.pop("PEAKY_PROGRESS", None)
check("disabled by default", PG.enabled(None) is False and PG.enabled(False) is False)
check("--progress enables", PG.enabled(True) is True)
os.environ["PEAKY_PROGRESS"] = "1"
check("PEAKY_PROGRESS=1 enables", PG.enabled(None) is True)
os.environ["PEAKY_PROGRESS"] = "0"
check("PEAKY_PROGRESS=0 does not", PG.enabled(None) is False)
os.environ.pop("PEAKY_PROGRESS", None)

off = PG.open_progress("t", flag=False, log=seen.append)
check("disabled -> a pass-through Reporter with no UI and no hold",
      off.ui is None and off.hold is False)
off("[assign_batch] (1/1) done s"); off.finish({}); off.close()

saved = {k: os.environ.pop(k, None) for k in ("DISPLAY", "WAYLAND_DISPLAY")}
try:
    check("no DISPLAY -> display_available() is False",
          PG.display_available() is False or sys.platform.startswith(("win", "darwin")))
    head = PG.open_progress("t", flag=True, log=seen.append)
    check("headless + --progress falls back to the terminal status, does not crash",
          isinstance(head.ui, PG.TerminalStatus))
    head("[assign_batch] (1/1) assigning s ...")
    head("[assign_batch] (1/1) done s")
    head.finish({"merged_M0": 5})
    head.close()
    check("terminal fallback reports no window to close", head.ui.alive() is False)
finally:
    for k, v in saved.items():
        if v is not None:
            os.environ[k] = v

check("importing progress.py pulls in no GUI toolkit at import time",
      "tkinter" not in sys.modules)


# ---- 6. CLI wiring: the reporter really reaches the pipeline ----------------
# Stubs pipeline.run_batch, so this needs no server. What it pins is that
# cmd_batch passes `log=` (without it the window would never move) and calls
# finish() with the RETURNED summary (not something scraped from the log).
import types  # noqa: E402

from peaky import cli as CLI  # noqa: E402
from peaky import pipeline as PL  # noqa: E402

os.environ.setdefault("MASCOPE_URL", "http://example.invalid")
os.environ.setdefault("MASCOPE_ACCESS_TOKEN", "test-token")

SUMMARY = {"merged_M0": 42, "merged_tiers": {"Assigned": 40}, "n_files": 2,
           "n_in_all_files": 10, "n_single_file": 1, "formula_disagreements": 0,
           "elapsed_s": 12.5}
captured = {}

def _fake_run_batch(**kw):
    captured["log"] = kw.get("log")
    kw["log"]("[phase] assign")
    kw["log"]("[assign_batch] (1/2) assigning s1 ...")
    kw["log"]("[assign_batch] (2/2) done s2")
    return {"ctx": types.SimpleNamespace(out_dir="/tmp/run"), "elapsed_s": 30.0,
            "assign": {"summary": SUMMARY}}

_real = PL.run_batch
PL.run_batch = _fake_run_batch
try:
    args = CLI.build_parser().parse_args(
        ["batch", "--batch", "B", "--dataset", "D", "--out-dir", "/tmp"])
    check("--progress defaults to off", args.progress is False)
    CLI.cmd_batch(args)
    check("cmd_batch passes the reporter as log=", isinstance(captured["log"], PG.Reporter))
    rep = captured["log"]
    check("  -> the reporter tracked the stubbed run", rep.state.samples_done == 2)
    check("  -> finish() got the RETURNED summary, not a parsed one",
          rep.state.stats.get("merged_M0") == 42 and rep.state.finished)
    check("  -> with --progress off there is no UI and no hold",
          rep.ui is None and rep.hold is False)
finally:
    PL.run_batch = _real


def test_all():
    assert FAIL == 0, f"{FAIL} checks failed"


if __name__ == "__main__":
    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)
