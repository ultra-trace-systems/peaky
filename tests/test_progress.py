"""Progress-window tests — no network, no display, no Tk.

THE POINT OF THIS FILE is the contract test in section 3. `peaky/progress.py`
reads the run's LOG STREAM (that is its only coupling to the pipeline), so a
reworded log line in assign_batch/assign/pipeline would silently flat-line the
progress bar with nothing failing anywhere. Section 3 pins the literal f-strings
those modules emit against the patterns progress.py parses, so a rewording
breaks a TEST instead of the window.

Run: python3 tests/test_progress.py   (or `pytest tests/test_progress.py`)
"""
import contextlib
import io
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
par.feed("[phase] assign")
par.feed("[assign_batch] parallel: 5 worker processes (match-workers/proc=2) over 6 samples")
check("parallel banner parses", par.parallel == 5)
check("  -> the banner alone tells the sample count (no 'assigning' line precedes it)",
      par.n_samples == 6 and par.phase == "assign")
_buf = io.StringIO()
with contextlib.redirect_stdout(_buf):
    PG.TerminalStatus().push(par.snapshot())
check("  -> so the terminal status has something to say right after the banner",
      "0/6 samples" in _buf.getvalue(), _buf.getvalue())
par.feed("[run] pass0 took 1.0s")
par.feed("[run] pass1 took 1.0s")
check("parallel mode does NOT drive the stage bar (replayed logs would lie)",
      par.stage_idx == 0 and par.stage_frac == 0.0)
par.feed("[assign_batch] (1/6) done kZ9")
check("parallel 'done' does not name a CURRENT sample (it is the one that just ended)",
      par.samples_done == 1 and par.current_sid == "")

# a banner whose "over N samples" tail was reworded away still has to FREEZE the
# stage bar -- an early samples bar is worth less than not animating a lie
par2 = PG.ProgressState(title="t")
par2.feed("[assign_batch] parallel: 4 worker processes (match-workers/proc=3)")
par2.feed("[run] pass0 took 1.0s")
check("a banner without the sample count still switches to parallel mode",
      par2.parallel == 4 and par2.stage_idx == 0 and par2.n_samples == 0)

# `peaky assign` logs no 'assigning' line at all: the stage lines are the cue
one = PG.ProgressState(title="t", n_samples=1)
one.feed("[run] pass0 took 0.2s")
check("a stage line alone puts the phase at 'assigning' (single-sample runs)",
      one.phase == "assign" and one.stage_idx == 1)

# the ETA clock starts with assignment, not with the window: the TS fetch and
# sample selection before the first 'assigning' line are not per-sample work
import time  # noqa: E402
clk = PG.ProgressState(title="t")
clk.t0 = time.monotonic() - 1000.0           # pretend a 1000 s fetch preceded us
check("no assign clock before assignment starts", clk.t_assign is None)
clk.feed("[assign_batch] (1/3) assigning s1 ...")
clk.feed("[assign_batch] (1/3) done s1")
check("ETA extrapolates from the ASSIGN clock, not the whole-window clock",
      clk.eta is not None and clk.eta < 5.0, clk.eta)
check("  -> while elapsed still counts the whole run", clk.elapsed > 999.0)
clk2 = PG.ProgressState(title="t")
clk2.feed("[assign_batch] parallel: 2 worker processes (match-workers/proc=6) over 4 samples")
check("the parallel banner starts the assign clock too", clk2.t_assign is not None)

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


# ---- 1b. the LITERAL serial stream, as `assign_batch.run` logs it with --jobs 1
# (assigning -> k stage lines -> the offset line -> done -> next assigning). The
# same stream WITHOUT the 'done' lines is what an emitter that forgot them looks
# like; the samples bar must advance on the next 'assigning' regardless.
def _serial_stream(n: int, k: int, with_done: bool):
    for i in range(1, n + 1):
        yield f"[assign_batch] ({i}/{n}) assigning s{i} ..."
        for j in range(k):
            yield f"[run] stage{j} took 0.5s"
        yield f"[assign_batch]   s{i}: offset=0.3"
        if with_done:
            yield f"[assign_batch] ({i}/{n}) done s{i}"
    yield ("[assign_batch] DONE: 9 merged M0 ({}); 1 in all files, 0 single-file, "
           "0 formula disagreements")


for with_done in (True, False):
    tag = "with" if with_done else "WITHOUT"
    ser = PG.ProgressState(title="t")
    ok_done = ok_eta = ok_stages = True
    for ln in _serial_stream(3, 4, with_done):
        ser.feed(ln)
        if (m := PG.RE_ASSIGNING.match(ln)):
            i = int(m.group(1))
            ok_done &= ser.samples_done == i - 1
            ok_eta &= ((ser.eta is not None) if i >= 2 else (ser.eta is None))
            ok_stages &= (ser.n_stages == 4) if i >= 2 else True
    check(f"serial stream {tag} 'done': samples_done == i-1 at each 'assigning'", ok_done)
    check(f"  -> ({tag}) ETA exists from the second sample on, not before", ok_eta)
    check(f"  -> ({tag}) stage count learned from the first sample (k=4)",
          ok_stages and ser.n_stages == 4, ser.n_stages)
    check(f"  -> ({tag}) every sample in at DONE", ser.samples_done == 3)

# parallel: the banner, then one 'done' per finished future, then the workers'
# buffered logs replayed in a burst. Sample granularity only.
pl = PG.ProgressState(title="t")
pl.feed("[phase] assign")
pl.feed("[assign_batch] parallel: 3 worker processes (match-workers/proc=4) over 5 samples")
for i, sid in enumerate(("s4", "s1", "s5", "s2", "s3"), 1):     # completion order
    pl.feed(f"[assign_batch] ({i}/5) done {sid}")
check("parallel: samples bar reaches N/N from the 'done' lines", pl.samples_done == 5)
for ln in _serial_stream(5, 6, with_done=False):
    if ln.startswith("[run]"):
        pl.feed(ln)                                              # the replay burst
check("parallel: replayed stage lines do not move the stage bar or the count",
      pl.stage_idx == 0 and pl.samples_done == 5 and pl.n_stages == PG.NOMINAL_STAGES)


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
eta_st.feed("[assign_batch] (1/3) assigning s1 ...")
eta_st.feed("[assign_batch] (1/3) done s1")
check("sample id survives between samples (it names the last one completed)",
      eta_st.current_sid == "s1" and eta_st.phase == "assign")
eta_st.feed("[assign_batch] (3/3) assigning s3 ...")
eta_st.feed("[assign_batch] (3/3) done s3")
check("no ETA once every sample is in (the report tail is not modelled)",
      eta_st.eta is None)
check("the last 'done' opens the MERGE (align runs next), with no current sample",
      eta_st.phase == "merge" and eta_st.current_sid == "")
eta_st.feed("[assign_batch] DONE: 9 merged M0 ({}); 1 in all files, 0 single-file, 0 formula disagreements")
check("the DONE line (logged after align) means the merge is OVER, not starting",
      eta_st.phase == "merged" and eta_st.current_sid == ""
      and PG.PHASE_LABEL["merged"] != PG.PHASE_LABEL["merge"])
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

# the stage-bar label: a pure snapshot -> text function, so the branch the UI
# used to hide (parallel -> "N workers") is checkable with no display at all
_sn = PG.ProgressState(title="t").snapshot()
check("stage label before any stage line is '--'", PG.stage_text(_sn) == "--")
_sn.update(stage_idx=7, n_stages=36, stage_name="pass2")
check("stage label names the stage out of the count",
      PG.stage_text(_sn) == "pass2 7/36", PG.stage_text(_sn))
_sn.update(parallel=5, stage_idx=0)
check("parallel: the stage slot counts WORKERS (no live stage stream)",
      PG.stage_text(_sn) == "5 workers", PG.stage_text(_sn))
_sn.update(parallel=5, stage_idx=7)
check("  -> even with replayed stage lines behind it", PG.stage_text(_sn) == "5 workers")
check("  -> and it says 'worker', singular, for one",
      PG.stage_text({"parallel": 1}) == "1 worker")
check("stage label survives a snapshot missing every key", PG.stage_text({}) == "--")


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
check("  -> and the banner carries the sample count the parser takes N from",
      emits("batch/assign_batch.py", 'f"over {len(sample_ids)} samples")'))
check("assign_batch emits the DONE line progress.py parses",
      emits("batch/assign_batch.py", 'log(f"[assign_batch] DONE: {summary[\'merged_M0\']} merged M0 '))
check("assign emits the per-stage timing line progress.py parses",
      emits("assignment/assign.py", 'st.log(f"[run] {tag} took {s[\'elapsed_s\']}s")'))
for ph_name in ("cluster", "vankrevelen", "report", "assign", "fetch", "provenance"):
    check(f"pipeline emits [phase] {ph_name}", emits("pipeline.py", f'log("[phase] {ph_name}")'))
    # PHASE_LABEL.get() falls back to the bare name, so a phase missing from it
    # shows as "vankrevelen" rather than "Van Krevelen" and nothing else notices.
    check(f"  -> progress.py has a header label for {ph_name}", ph_name in PG.PHASE_LABEL)

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


# The 'done' line must be logged by the SERIAL branch too: `emits()` is a
# substring test over the whole module and cannot tell that a line lives in the
# parallel branch only (which is exactly how --jobs 1 shipped with a dead bar).
def _serial_branch_src(module_src: str) -> str:
    """Source of the body of `if n_jobs <= 1:` inside assign_batch.run."""
    import ast
    for fn in ast.walk(ast.parse(module_src)):
        if not (isinstance(fn, ast.FunctionDef) and fn.name == "run"):
            continue
        for node in ast.walk(fn):
            t = getattr(node, "test", None)
            if (isinstance(node, ast.If) and isinstance(t, ast.Compare)
                    and isinstance(t.left, ast.Name) and t.left.id == "n_jobs"
                    and len(t.ops) == 1 and isinstance(t.ops[0], ast.LtE)):
                return "\n".join(ast.get_source_segment(module_src, s) for s in node.body)
    return ""


serial_src = _serial_branch_src(src)
check("assign_batch.run has the `if n_jobs <= 1:` serial branch", bool(serial_src))
check("the SERIAL branch itself logs the per-sample 'done' line",
      'log(f"[assign_batch] ({i}/{len(sample_ids)}) done {sid}")' in serial_src)
check("  -> after the sample is applied (inside the per-sample loop)",
      "_apply(" in serial_src
      and serial_src.index("_apply(") < serial_src.index("done {sid}"))

# `peaky assign` calls one sample's assign.run directly: no pipeline runs on that
# path, so nothing logs `[phase] assign` and the header would read "fetching time
# series" for the entire assignment. cmd_assign has to say so itself -- and WHERE
# it says it is the whole point, so read the structure, not the text.
def _progress_with_body(module_src: str, fname: str) -> list:
    """Statements of the `with ... open_progress(...) as prog:` body in `fname`."""
    import ast
    for fn in ast.walk(ast.parse(module_src)):
        if not (isinstance(fn, ast.FunctionDef) and fn.name == fname):
            continue
        for node in ast.walk(fn):
            if (isinstance(node, ast.With) and node.items
                    and "open_progress" in ast.dump(node.items[0].context_expr)):
                return [ast.get_source_segment(module_src, s) for s in node.body]
    return []


ca_body = _progress_with_body((PKG / "cli.py").read_text(), "cmd_assign")
check("cmd_assign wraps the run in open_progress()", bool(ca_body))
check("  -> and marks the assign phase as its FIRST act inside that block",
      bool(ca_body) and ca_body[0].startswith('prog.phase("assign")'), ca_body[:1])
check("  -> i.e. before assign.run, not after it",
      any("assign.run(" in s for s in ca_body[1:]))

check("assign_batch records elapsed_s in batch_summary", '"elapsed_s": round(time.time() - t_start, 1)' in src)
check("assign_batch records n_jobs beside it (a duration needs its job count)",
      '"n_jobs": n_jobs,' in src)
check("run_batch returns a whole-pipeline elapsed_s",
      '"elapsed_s": elapsed' in (PKG / "pipeline.py").read_text())


# ---- 3b. the DOCUMENTED contract -------------------------------------------
# The hold's bounds and the macOS fallback are promises made to whoever types
# --progress; an undocumented bound reads as a hang, and an undocumented macOS
# fallback reads as a broken feature.
ROOT = PKG.parent
CHANGELOG = (ROOT / "CHANGELOG.md").read_text()
for _name, _txt in (("README.md", (ROOT / "README.md").read_text()),
                    ("CHANGELOG.md", CHANGELOG)):
    check(f"{_name}: the hold is bounded by PEAKY_PROGRESS_HOLD_S",
          "PEAKY_PROGRESS_HOLD_S" in _txt)
    check(f"{_name}: the hold needs an interactive terminal",
          "interactive terminal" in _txt or "a tty" in _txt)
    check(f"{_name}: Ctrl-C closes the window at once", "Ctrl-C" in _txt)
    check(f"{_name}: macOS falls back to the terminal status", "macOS" in _txt)
check("the --progress bullets are under [Unreleased], not a released version",
      "PEAKY_PROGRESS_HOLD_S" in CHANGELOG.split("## [Unreleased]")[1].split("\n### [")[0])
check("--progress --help names the hold env var",
      "PEAKY_PROGRESS_HOLD_S seconds" in (PKG / "cli.py").read_text())


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

for bad in (None, "oops", 42, ["x"]):
    try:
        fin = PG.Reporter("t", log=seen.append, ui=_Boom())
        fin.finish(bad)
        fin.phase(None)
        ok = fin.state.finished and fin.state.stats == {}
    except Exception:       # noqa: BLE001
        ok = False
    check(f"finish({bad!r}) / phase(None) never raise (not-a-dict -> no stats)", ok)


class _BadStr:
    """The only thing the outer try/except in finish()/phase() actually buys:
    an argument that raises while being turned into text."""
    def __str__(self): raise RuntimeError("__str__ exploded")


try:
    fin = PG.Reporter("t", log=seen.append, ui=_Boom())
    fin.phase(_BadStr())
    fin.finish({"merged_M0": 1}, error=_BadStr())
    ok = True
except Exception:           # noqa: BLE001
    ok = False
check("an argument whose __str__ raises cannot break the run either", ok)

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


# the hold vs. the ways a run can end -- seen by a UI that records close(wait=)
class _Rec:
    def __init__(self): self.closes = []
    def push(self, snap): pass
    def close(self, wait): self.closes.append(wait)
    def alive(self): return True


def _exit_with(exc):
    rec = _Rec()
    try:
        with PG.Reporter("t", log=seen.append, ui=rec, hold=True) as rr:
            rr("[assign_batch] (1/2) assigning s1 ...")
            if exc is not None:
                raise exc
            rr.finish({"merged_M0": 1})
    except BaseException as e:      # noqa: BLE001
        assert isinstance(e, type(exc)), e
    return rec, rr


rec, rr = _exit_with(KeyboardInterrupt())
check("Ctrl-C: the window is closed WITHOUT waiting", rec.closes == [False], rec.closes)
check("  -> and no stats panel is shown for an interrupted run", rr.state.finished is False)
rec, rr = _exit_with(SystemExit(1))
check("SystemExit: closed without waiting, no stats panel",
      rec.closes == [False] and rr.state.finished is False, rec.closes)
rec, rr = _exit_with(None)
check("a finished run with hold=True waits on the window", rec.closes == [True], rec.closes)
rec, rr = _exit_with(ValueError("run blew up"))
check("a failed run holds too (the error panel is there to be read)",
      rec.closes == [True] and rr.state.finished and "ValueError" in rr.state.error)
rec = _Rec()
with PG.Reporter("t", log=seen.append, ui=rec, hold=False) as rr:
    rr.finish({})
check("hold=False never waits, finished or not", rec.closes == [False], rec.closes)


# ---- 5. enable/disable + headless behaviour ---------------------------------
# Every env var this section touches is put back: the suite runs in ONE process,
# so a var left set here leaks into every later test (and into the rest of the
# file, where `open_progress(flag=False)` would suddenly open a window).
import os  # noqa: E402

_saved_flag = os.environ.get("PEAKY_PROGRESS")
os.environ.pop("PEAKY_PROGRESS", None)
try:
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
finally:
    os.environ.pop("PEAKY_PROGRESS", None)
    if _saved_flag is not None:
        os.environ["PEAKY_PROGRESS"] = _saved_flag

check("importing progress.py pulls in no GUI toolkit at import time",
      "tkinter" not in sys.modules)


# ---- 5b. the three ways the window cannot open, all -> the terminal status ---
# NEVER by unsetting $DISPLAY: on a developer's machine (and on Windows) that
# leaves `display_available()` True, and the test would open a REAL window and
# then block the suite on closing it. Patch the probe, and the toolkit with it.
def _ran_headless(rep) -> bool:
    """Drive a whole miniature run through the fallback and report it survived."""
    rep("[assign_batch] (1/1) assigning s ...")
    rep("[assign_batch] (1/1) done s")
    rep.finish({"merged_M0": 5})
    rep.close()
    return rep.ui.alive() is False      # nothing to close -> the CLI stays quiet


def test_no_display_falls_back_to_terminal_status(monkeypatch):
    """No DISPLAY/WAYLAND_DISPLAY, or macOS: never even try Tk."""
    monkeypatch.setattr(PG, "display_available", lambda: False)
    rep = PG.open_progress("t", flag=True, log=seen.append)
    assert isinstance(rep.ui, PG.TerminalStatus), rep.ui
    assert _ran_headless(rep)


def test_display_announced_but_dead_falls_back(monkeypatch):
    """A display the environment ANNOUNCES but Tk cannot use: a stale $DISPLAY,
    an X server refusing the connection, a broken theme. `Tk()` raises on the UI
    thread, which `start()` must report as False rather than let escape."""
    import types as _types

    monkeypatch.setattr(PG, "display_available", lambda: True)

    class _DeadTk:
        def __init__(self, *a, **k):
            raise RuntimeError('couldn\'t connect to display ":0"')

    fake = _types.ModuleType("tkinter")
    fake.Tk = _DeadTk
    fake.ttk = _types.ModuleType("tkinter.ttk")
    monkeypatch.setitem(sys.modules, "tkinter", fake)
    monkeypatch.setitem(sys.modules, "tkinter.ttk", fake.ttk)

    ui = PG.TkWindow("t")
    assert ui.start() is False                  # reported, not raised
    rep = PG.open_progress("t", flag=True, log=seen.append)
    assert isinstance(rep.ui, PG.TerminalStatus), rep.ui
    assert _ran_headless(rep)


def test_no_tkinter_at_all_falls_back(monkeypatch):
    """The interpreter has no tkinter (a python-tk-less distro build, a slim
    container). The import is inside the UI thread precisely so this is a
    fallback and not an ImportError at `import peaky.progress`."""
    import time as _time

    monkeypatch.setattr(PG, "display_available", lambda: True)
    monkeypatch.setitem(sys.modules, "tkinter", None)   # import -> ImportError
    ui, t = PG.TkWindow("t"), _time.monotonic()
    assert ui.start() is False
    # the ImportError is CAUGHT and the ready event set: an uncaught one kills the
    # UI thread silently and start() burns its whole 5 s timeout before saying so
    assert _time.monotonic() - t < 2.0, _time.monotonic() - t
    rep = PG.open_progress("t", flag=True, log=seen.append)
    assert isinstance(rep.ui, PG.TerminalStatus), rep.ui
    assert _ran_headless(rep)


# ---- 6. CLI wiring: the reporter really reaches the pipeline ----------------
# Stubs pipeline.run_batch AND cli._require_creds, so this needs no server and
# no credentials at all (CI asserts that no ~/.mascope/.env or token exists, and
# the suite runs in one process, so seeding the env here is not reliable). What
# it pins is that cmd_batch passes `log=` (without it the window would never
# move) and calls finish() with the RETURNED summary (not something scraped
# from the log).
import types  # noqa: E402

from peaky import cli as CLI  # noqa: E402
from peaky import pipeline as PL  # noqa: E402

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

_real, _real_creds = PL.run_batch, CLI._require_creds
PL.run_batch = _fake_run_batch
CLI._require_creds = lambda: None      # cmd_batch resolves the name at call time
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
    CLI._require_creds = _real_creds


# ---- 7. the serial branch EXECUTED: one 'done' line per sample --------------
# pytest-style (monkeypatch fixture) so every stub is undone; the __main__ runner
# below drives these with a MonkeyPatch of its own.
def test_serial_branch_logs_done_per_sample(monkeypatch):
    """assign_batch.run(n_jobs=1) over three stubbed samples -- assign.run and
    the Mascope IO replaced, no network -- and the captured log stream read back
    through ProgressState: the bar must reach 3/3 from that stream alone."""
    import tempfile

    import pandas as pd

    from peaky.assignment import assign as A
    from peaky.batch import assign_batch as AB
    from peaky.io import io_mascope as IO

    def _fake_assign(sid, context="ambient-air", *, log=print, **kw):
        for stage in ("pass0", "pass1", "cleanup"):
            log(f"[run] {stage} took 0.1s")
        led = pd.DataFrame({
            "peak_id": [f"{sid}-p1"], "mz": [217.1200 + int(sid[1:]) * 1e-4],
            "height": [1e4], "role": ["M0"], "neutral_formula": ["C10H16O2"],
            "adduct": ["[M+Br]-"], "ion_formula": ["C10H16O2Br-"],
            "tier": ["Assigned"], "ion_score": [0.9], "method": ["pass1"]})
        return {"ledger": led, "plausibility_audit": [], "stats": {"n_peaks": 1}}

    def _offline(*a, **k):
        raise RuntimeError("offline")

    monkeypatch.setattr(A, "run", _fake_assign)
    monkeypatch.setattr(IO, "connect", lambda **kw: "CLIENT")
    monkeypatch.setattr(IO, "fetch_peaks", _offline)   # offsets -> None (guarded in _apply)
    sids = ["s1", "s2", "s3"]
    peaks = pd.DataFrame({"sample_item_id": sids, "polarity": ["-"] * 3})
    lines: list = []
    with tempfile.TemporaryDirectory() as td:
        AB.run(peaks=peaks, reagent="Br", sample_ids=sids, out_dir=td, n_jobs=1,
               log=lines.append)

    done = [PG.RE_SAMPLE_DONE.match(ln) for ln in lines if PG.RE_SAMPLE_DONE.match(ln)]
    assert [m.groups() for m in done] == \
        [("1", "3", "s1"), ("2", "3", "s2"), ("3", "3", "s3")], lines
    # ... and each 'done' lands before the NEXT sample's 'assigning'
    order = [(ln.split(") ")[1].split()[0], int(ln.split("(")[1].split("/")[0]))
             for ln in lines if PG.RE_ASSIGNING.match(ln) or PG.RE_SAMPLE_DONE.match(ln)]
    assert order == [("assigning", 1), ("done", 1), ("assigning", 2), ("done", 2),
                     ("assigning", 3), ("done", 3)], order

    st = PG.ProgressState(title="t")
    for ln in lines:
        st.feed(ln)
    assert (st.samples_done, st.n_samples, st.n_stages) == (3, 3, 3), st.snapshot()


# ---- 8. the hold is for a PERSON at a terminal, and it is bounded ------------
class _Stream(io.StringIO):
    """A stdin/stdout stand-in that answers isatty() the way the test wants."""
    def __init__(self, tty: bool):
        super().__init__()
        self._tty = tty

    def isatty(self):
        return self._tty


def _open_headless(monkeypatch, *, stdin_tty: bool, stdout_tty: bool):
    monkeypatch.setattr(PG, "display_available", lambda: False)    # no Tk anywhere
    monkeypatch.setattr(sys, "stdin", _Stream(stdin_tty))
    monkeypatch.setattr(sys, "stdout", _Stream(stdout_tty))
    return PG.open_progress("t", flag=True, log=seen.append)


def test_hold_only_on_an_interactive_tty(monkeypatch):
    monkeypatch.delenv("PEAKY_PROGRESS_HOLD_S", raising=False)
    assert _open_headless(monkeypatch, stdin_tty=True, stdout_tty=True).hold is True
    assert _open_headless(monkeypatch, stdin_tty=False, stdout_tty=True).hold is False
    assert _open_headless(monkeypatch, stdin_tty=True, stdout_tty=False).hold is False
    monkeypatch.setenv("PEAKY_PROGRESS_HOLD_S", "0")
    assert _open_headless(monkeypatch, stdin_tty=True, stdout_tty=True).hold is False
    monkeypatch.setenv("PEAKY_PROGRESS_HOLD_S", "30")
    assert _open_headless(monkeypatch, stdin_tty=True, stdout_tty=True).hold is True
    # an explicit hold=False from the call site is never overridden
    assert PG.open_progress("t", flag=True, log=seen.append, hold=False).hold is False

    # a closed or absent stream (pythonw, a daemonised run): isatty() RAISES
    # rather than answering, and an unreadable window must not be held either
    class _Closed(io.StringIO):
        def isatty(self): raise ValueError("I/O operation on closed file")

    monkeypatch.setattr(sys, "stdin", _Closed())
    assert PG._interactive() is False
    assert PG.open_progress("t", flag=True, log=seen.append).hold is False


def test_display_available_by_platform(monkeypatch):
    """macOS is NEVER a display for this module: Tk off the main thread aborts
    the process there, so the terminal fallback is the only safe answer."""
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
    monkeypatch.setattr(sys, "platform", "darwin")
    assert PG.display_available() is False
    monkeypatch.setattr(sys, "platform", "win32")
    assert PG.display_available() is True
    monkeypatch.setattr(sys, "platform", "linux")
    assert PG.display_available() is True
    monkeypatch.delenv("DISPLAY")
    assert PG.display_available() is True           # Wayland alone is enough
    monkeypatch.delenv("WAYLAND_DISPLAY")
    assert PG.display_available() is False


def test_hold_seconds_parsing(monkeypatch):
    monkeypatch.delenv("PEAKY_PROGRESS_HOLD_S", raising=False)
    assert PG.hold_seconds() == PG.HOLD_S_DEFAULT == 600.0
    for raw, want in (("0", 0.0), ("30", 30.0), ("2.5", 2.5), ("-5", 0.0),
                      ("  45 ", 45.0), ("ten", 600.0), ("", 600.0)):
        monkeypatch.setenv("PEAKY_PROGRESS_HOLD_S", raw)
        assert PG.hold_seconds() == want, (raw, PG.hold_seconds())


def test_tk_close_is_bounded(monkeypatch):
    """TkWindow.close with a stand-in for the Tk thread (one that exits on the
    queued 'quit', like mainloop does): a hold ends after PEAKY_PROGRESS_HOLD_S
    even if nobody closes the window, and close() never returns with the UI
    thread still alive."""
    import threading
    import time

    def _window():
        w = PG.TkWindow("t")
        w._ok = True

        def _ui():                          # stand-in mainloop
            while True:
                kind, _ = w.q.get()
                if kind == "quit":
                    return
        w._thread = threading.Thread(target=_ui, daemon=True)
        w._thread.start()
        return w

    monkeypatch.setenv("PEAKY_PROGRESS_HOLD_S", "0.3")
    w = _window()
    t = time.monotonic()
    w.close(wait=True)
    dt = time.monotonic() - t
    assert 0.3 <= dt < 3.0, dt
    assert not w.alive()

    w = _window()
    t = time.monotonic()
    w.close(wait=False)
    assert time.monotonic() - t < 1.0 and not w.alive()

    w = _window()
    w.close(wait=True, timeout=0.05)        # explicit bound wins over the env
    assert not w.alive()

    w = PG.TkWindow("t")                    # never started: nothing to close
    w.close(wait=True)
    assert not w.alive()


def test_all():
    assert FAIL == 0, f"{FAIL} checks failed"


if __name__ == "__main__":
    import pytest
    for _fn in [v for k, v in list(globals().items())
                if k.startswith("test_") and k != "test_all" and callable(v)]:
        _mp = pytest.MonkeyPatch()
        try:
            _fn(_mp)
            PASS += 1
            print(f"  ok  {_fn.__name__}")
        except Exception as e:      # noqa: BLE001
            FAIL += 1
            print(f"FAIL  {_fn.__name__}  {e!r}")
        finally:
            _mp.undo()
    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)
