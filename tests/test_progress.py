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
import io  # noqa: E402


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
