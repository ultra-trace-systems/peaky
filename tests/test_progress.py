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
import tomllib
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
# ...and it logs no 'done' line either, so nothing in that stream ever completes
# a sample: the samples bar would read 0/1 for the whole run and the stage bar
# would stay against the NOMINAL count instead of this run's real one. The
# command says it itself (cmd_assign -> prog.sample_done()).
for _n in ("pass1", "cleanup"):
    one.feed(f"[run] {_n} took 0.2s")
check("  -> a single-sample stream alone never advances the samples bar",
      one.samples_done == 0 and one.sample_frac == 0.0)
check("  -> and its stage bar is against the nominal count, so it cannot fill",
      one.n_stages == PG.NOMINAL_STAGES and one.stage_frac < 1.0)
one.mark_sample_done()
check("mark_sample_done() completes the sample with no log line to do it",
      one.samples_done == 1 and one.n_samples == 1 and one.sample_frac == 1.0)
check("  -> and learns the stage count from the stages that just ran (k=3)",
      one.n_stages == 3 and one.stage_frac == 1.0)
check("  -> without resetting the stage bar: this was the LAST sample",
      one.stage_idx == 3 and one.stage_name == "cleanup")
_buf = io.StringIO()
with contextlib.redirect_stdout(_buf):
    PG.TerminalStatus().push(one.snapshot())
check("  -> so the terminal fallback reports 1/1 samples, not 0/1",
      "1/1 samples" in _buf.getvalue(), _buf.getvalue())

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
# the two per-file summary lines the LIVE stats panel sums (section 10 below)
check("assign emits the per-file tiers line progress.py sums into the live panel",
      emits("assignment/assign.py", 'log(f"[run] tiers {tc}")'))
check("assign emits the per-file stats line progress.py sums into the live panel",
      emits("assignment/assign.py", 'log(f"[run] stats {json.dumps(st)}")'))
check("  -> and stamps the admission counts on it (the 'admitted by occurrence' row)",
      emits("assignment/assign.py",
            'st["admitted"] = {"height": adm["height"], "occurrence": adm["occurrence"],'))
check("assign_batch's DONE line carries the merge numbers the panel reads provisionally",
      emits("batch/assign_batch.py",
            '''f"({summary['merged_tiers']}); {summary['n_in_all_files']} in all files, "''')
      and emits("batch/assign_batch.py", '''f"{summary['n_single_file']} single-file, "''')
      and emits("batch/assign_batch.py",
                '''f"{summary['formula_disagreements']} formula disagreements")'''))
# The keys the totals are summed from come from ledger.stats ITSELF, dumped the
# way assign.run dumps them: a renamed key would otherwise zero a row of the panel
# with nothing failing anywhere.
import json as _json  # noqa: E402

import pandas as _pd  # noqa: E402

from peaky.assignment import ledger as _L  # noqa: E402

_led = _pd.DataFrame({"role": ["M0", "M0", "iso_child", "unexplained"],
                      "height": [10.0, 5.0, 1.0, 2.0],
                      "tier": ["Assigned", "Candidate", None, None]})
_st = _L.stats(_led)
check("ledger.stats names n_peaks / by_role / by_tier the way the live panel reads them",
      _st.get("n_peaks") == 4 and (_st.get("by_role") or {}).get("M0") == 2
      and _st["by_role"].get("unexplained") == 1
      and _st.get("by_tier") == {"Assigned": 1, "Candidate": 1}, _st)
_lt = PG.LiveTotals()
check("  -> and its json.dumps round-trips into one file's totals",
      _lt.read_stats(_json.dumps(_st)) and (_lt.files, _lt.m0, _lt.peaks, _lt.unexplained)
      == (1, 2, 4, 1) and _lt.tiers == {"Assigned": 1, "Candidate": 1}, _lt)
# That line comes from `_safe`, so ONLY a `safe=True` stage is countable -- and
# NOMINAL_STAGES is the denominator of the stage bar until a completed sample
# replaces it, which on the `peaky assign` path happens only at the very end.
# A nominal counting all the table's rows put the bar at 20/36 for a whole run.
from peaky.assignment import assign as _A  # noqa: E402

_timed = [_s.name for _s in _A._STAGES if _s.safe]
check("NOMINAL_STAGES counts the stages of assign.run that TIME themselves",
      PG.NOMINAL_STAGES == len(_timed), f"{PG.NOMINAL_STAGES} != {len(_timed)}")
check("  -> i.e. fewer than the stage table's rows (the untimed ones log nothing)",
      len(_timed) < len(_A._STAGES), f"{len(_timed)} of {len(_A._STAGES)}")
for ph_name in ("cluster", "vankrevelen", "report", "assign", "fetch", "provenance",
                "select"):
    check(f"pipeline emits [phase] {ph_name}", emits("pipeline.py", f'log("[phase] {ph_name}")'))
    # PHASE_LABEL.get() falls back to the bare name, so a phase missing from it
    # shows as "vankrevelen" rather than "Van Krevelen" and nothing else notices.
    check(f"  -> progress.py has a header label for {ph_name}", ph_name in PG.PHASE_LABEL)


# `emits()` is a substring test over the WHOLE module, so it cannot tell that a
# marker lives in `run_batch` only -- which is exactly how the pooled pipeline
# shipped: it announced the assign phase and nothing else, leaving the phase
# line on a stale label through the pooled fetch and the provenance write. Both
# entry points run the same shape of run, so both emit the same markers.
def _func_src(module_src: str, fname: str) -> str:
    """Source of the body of the function `fname` (not its nested defs)."""
    import ast
    for fn in ast.walk(ast.parse(module_src)):
        if isinstance(fn, ast.FunctionDef) and fn.name == fname:
            return "\n".join(ast.get_source_segment(module_src, s) for s in fn.body)
    return ""


pipe_src = (PKG / "pipeline.py").read_text()
for _fn in ("run_batch", "run_pooled_batches"):
    _body = _func_src(pipe_src, _fn)
    check(f"pipeline.{_fn} is a top-level function", bool(_body))
    for ph_name in ("fetch", "assign", "provenance"):
        check(f"  -> {_fn} ITSELF emits [phase] {ph_name}",
              f'log("[phase] {ph_name}")' in _body)
_gen = _func_src(pipe_src, "generate_report")
for ph_name in ("cluster", "vankrevelen", "report"):
    check(f"generate_report ITSELF emits [phase] {ph_name} (both paths share it)",
          f'log("[phase] {ph_name}")' in _gen)

# the pooled path picks its own cover, before the run dir even exists
_pool_body = _func_src(pipe_src, "run_pooled_batches")
check("the pooled pipeline marks the select phase around its own set-cover",
      '[phase] select' in _pool_body
      and _pool_body.index('[phase] select') < _pool_body.index("SS.select_cover_samples("))

# the residual stage: assign_batch marks it, the window has a label for it, and
# its (i/N) lines count ON from the cover's -- the samples bar grows, not resets
check("assign_batch emits [phase] residual around the residual stage",
      emits("batch/assign_batch.py", 'log("[phase] residual")')
      and "residual" in PG.PHASE_LABEL)
res_st = PG.ProgressState(title="t")
for ln in ("[assign_batch] (1/2) assigning a0 ...", "[assign_batch] (1/2) done a0",
           "[assign_batch] (2/2) assigning a1 ...", "[assign_batch] (2/2) done a1"):
    res_st.feed(ln)
check("residual stream: the cover's last 'done' reads as the merge",
      res_st.phase == "merge" and res_st.samples_done == 2 and res_st.n_samples == 2)
res_st.feed("[phase] residual")
check("residual stream: the phase marker reads as 'targeting the residual'",
      res_st.phase == "residual"
      and res_st.snapshot()["phase_label"] == "targeting the residual",
      res_st.snapshot()["phase_label"])
res_st.feed("[phase] assign")
res_st.feed("[assign_batch] (3/4) assigning z1 ...")
check("residual stream: the first residual 'assigning' grows N and keeps the 2 done",
      res_st.phase == "assign" and res_st.n_samples == 4 and res_st.samples_done == 2
      and res_st.current_sid == "z1" and res_st.sample_frac == 0.5, res_st.snapshot())
res_st.feed("[assign_batch] (3/4) done z1")
res_st.feed("[assign_batch] (4/4) assigning z2 ...")
res_st.feed("[assign_batch] (4/4) done z2")
check("residual stream: ...and the stage's last 'done' is the (final) merge again",
      res_st.phase == "merge" and res_st.samples_done == 4 and res_st.sample_frac == 1.0)

# and the label that marker looks up is part of the interface, not decoration
sel_st = PG.ProgressState(title="t")
sel_st.feed("[phase] select")
check("a select marker reads as 'selecting samples'",
      sel_st.phase == "select"
      and sel_st.snapshot()["phase_label"] == "selecting samples",
      sel_st.snapshot()["phase_label"])

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
# `find` not `index`: a deleted marker must FAIL this check, not raise out of the
# module and take the whole contract section down as a collection error.
_i_apply = serial_src.find("_apply(")
_i_done = serial_src.find("done {sid}")
check("  -> after the sample is applied (inside the per-sample loop)",
      0 <= _i_apply < _i_done, f"_apply at {_i_apply}, 'done' at {_i_done}")


# On the single-batch path the pipeline delegates selection to assign_batch, so
# THAT is where the select phase begins and ends. The branch brackets itself:
# without the closing marker the window would go on reading "selecting samples"
# through the admission table and the reference lists that follow it.
def _if_branch_src(module_src: str, fname: str, test_src: str) -> str:
    """Body source of the first `if <test_src>:` inside the function `fname`."""
    import ast
    for fn in ast.walk(ast.parse(module_src)):
        if not (isinstance(fn, ast.FunctionDef) and fn.name == fname):
            continue
        for node in ast.walk(fn):
            if (isinstance(node, ast.If)
                    and ast.get_source_segment(module_src, node.test) == test_src):
                return "\n".join(ast.get_source_segment(module_src, s) for s in node.body)
    return ""


sel_src = _if_branch_src(src, "run", "sample_ids is None")
check("assign_batch.run has the `if sample_ids is None:` selection branch", bool(sel_src))
check("the SELECTION branch opens with the select phase marker",
      '[phase] select' in sel_src
      and sel_src.index('[phase] select') < sel_src.index("SS.select_cover_samples("))
check("  -> and hands the phase back to assign once the cover is picked",
      'log("[phase] assign")' in sel_src
      and sel_src.rindex('[phase] assign') > sel_src.index("SS.select_cover_samples("))

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
# and it has to COMPLETE that sample once assign.run returns -- no `(i/N) done`
# line reaches this path, so without it the samples bar ends at 0/1 and the
# stage bar never learns how many stages the run really had.
_i_run = next((i for i, s in enumerate(ca_body) if "assign.run(" in s), -1)
_i_done = next((i for i, s in enumerate(ca_body) if s.startswith("prog.sample_done(")), -1)
_i_rep = next((i for i, s in enumerate(ca_body) if s.startswith('prog.phase("report")')), -1)
check("  -> and marks its ONE sample done between assign.run and the report phase",
      -1 < _i_run < _i_done < _i_rep, ca_body)

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
    # the window now opens unasked, so the way to turn it OFF is part of the
    # contract -- undocumented, a window nobody wanted has no visible off switch
    check(f"{_name}: names --no-progress, the way to turn the window off",
          "--no-progress" in _txt)
    # `peaky assign` is the one-sample case: its bars DO fill, but an ETA over
    # completed samples cannot exist for it. Undocumented, a missing ETA reads
    # as the same bug the filled bars just stopped being.
    check(f"{_name}: says why a single-sample run shows no ETA",
          "completes when the run does" in _txt)
# The CURRENT release train: the [Unreleased] section plus the section for the
# version pyproject names right now. The --progress bullets must be in one of
# those -- still unreleased, or in the release actually shipping them -- and
# never orphaned under an older version.
#
# Self-locating on purpose. Pinned to [Unreleased] alone, this check FAILED the
# moment a release was cut and the bullets moved into their release section,
# which is the one time nobody wants a spurious test failure. Sliced on
# LINE-ANCHORED `## [` headings too: a plain `split("## [Unreleased]")` also
# matches the string inside a bullet's prose -- the 0.5.0-retitle note quotes a
# heading -- which silently truncated the span it searched.
_VERSION = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
_SECTIONS = re.split(r"^## \[", CHANGELOG, flags=re.M)[1:]
# `Unreleased]\n` matches the BARE heading only: the legacy `## [Unreleased] —
# 0.4.0` section (0.4.0 was never cut, so its heading stands) is an old release,
# not the current train, and must not satisfy this check.
_TRAIN = [s for s in _SECTIONS
          if s.startswith("Unreleased]\n") or s.startswith(_VERSION + "]")]
check("the changelog has an [Unreleased] section and one for the current version",
      len(_TRAIN) == 2, [s[:30] for s in _TRAIN])
check("the --progress bullets are under [Unreleased] or the version shipping them",
      any("PEAKY_PROGRESS_HOLD_S" in s for s in _TRAIN),
      [s[:30] for s in _TRAIN])
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
    check("--progress enables, --no-progress disables",
          PG.enabled(True) is True and PG.enabled(False) is False)
    os.environ["PEAKY_PROGRESS"] = "1"
    check("PEAKY_PROGRESS=1 enables", PG.enabled(None) is True)
    check("... and an explicit --no-progress still beats it", PG.enabled(False) is False)
    os.environ["PEAKY_PROGRESS"] = "0"
    check("PEAKY_PROGRESS=0 does not", PG.enabled(None) is False)
    check("... and an explicit --progress still beats it", PG.enabled(True) is True)
    os.environ.pop("PEAKY_PROGRESS", None)

    # Neither flag nor env: the DEFAULT, which asks whether a person is there.
    # The window is ON for a terminal and must stay OFF everywhere else -- a CI
    # job or a skill-driven run would otherwise get [progress] lines in output it
    # is capturing, from a window nobody asked for and nobody can see.
    _saved_tty = PG._interactive
    try:
        PG._interactive = lambda: True
        check("on by default at an interactive terminal", PG.enabled(None) is True)
        check("but that default is not an explicit REQUEST (so no display warning)",
              PG.explicitly_requested(None) is False)
        PG._interactive = lambda: False
        check("off by default for a pipe / CI job / skill-driven run",
              PG.enabled(None) is False)
        os.environ["PEAKY_PROGRESS"] = "1"
        check("PEAKY_PROGRESS=1 is an explicit request even with no terminal",
              PG.enabled(None) is True and PG.explicitly_requested(None) is True)
        os.environ.pop("PEAKY_PROGRESS", None)
        check("--progress is an explicit request too",
              PG.explicitly_requested(True) is True)
    finally:
        PG._interactive = _saved_tty
        os.environ.pop("PEAKY_PROGRESS", None)

    # The CLI flag must be TRI-state -- --progress / --no-progress / unset. A
    # `store_true` default of False would read as an explicit opt-out on every
    # ordinary run and the interactive default could never fire.
    from peaky import cli as _CLI  # noqa: E402
    _ap = _CLI.build_parser()
    check("an unset --progress parses as None, so the default decides",
          _ap.parse_args(["batch", "--batch", "B"]).progress is None)
    check("--progress / --no-progress force it on / off",
          _ap.parse_args(["batch", "--batch", "B", "--progress"]).progress is True
          and _ap.parse_args(["batch", "--batch", "B", "--no-progress"]).progress is False)

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
    check("--progress is unset here, so progress.enabled decides",
          args.progress is None)
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


# ---- 7b. the SINGLE-SAMPLE command EXECUTED: both bars reach the end --------
def test_single_sample_command_fills_both_bars(monkeypatch):
    """`peaky assign` driven for real with assign.run stubbed (no network, no
    credentials): the window must end at 1/1 samples with a FULL stage bar.

    A source check cannot see this -- the bars are numbers that exist only once
    the command has run -- and this is the command whose samples bar sat dead at
    0/1, with the stage bar frozen part-way, for the whole of every run."""
    import tempfile

    from peaky.assignment import assign as A

    # The stages that log a time, MINUS a few: a real run emits fewer than the
    # nominal (stages are gated by reagent, context and the --no-passN flags),
    # which is why the count is learned at all -- and it is what makes the
    # stage-bar assertion below load-bearing, since a denominator left at
    # NOMINAL_STAGES cannot reach 100% however many stages ran.
    stages = [st_.name for st_ in A._STAGES if st_.safe][:-4]
    assert 0 < len(stages) < PG.NOMINAL_STAGES, stages

    def _fake_assign(sid, context="ambient-air", *, log=print, **kw):
        for name in stages:
            log(f"[run] {name} took 0.1s")
        return {"ledger": None, "stats": {"n_peaks": 7}}

    made = {}
    _real_open = PG.open_progress

    def _spy_open(*a, **kw):
        made["rep"] = _real_open(*a, **kw)
        return made["rep"]

    monkeypatch.setattr(A, "run", _fake_assign)
    monkeypatch.setattr(PG, "open_progress", _spy_open)
    monkeypatch.setattr(CLI, "_require_creds", lambda: None)
    monkeypatch.setattr(CLI, "_write_assign_outputs", lambda args, out, base: None)

    with tempfile.TemporaryDirectory() as td:
        args = CLI.build_parser().parse_args(
            ["assign", "--sample-id", "s1", "--reagent", "Br", "--output-dir", td])
        with contextlib.redirect_stdout(io.StringIO()):
            CLI.cmd_assign(args)

    snap = made["rep"].state.snapshot()
    assert (snap["samples_done"], snap["n_samples"]) == (1, 1), snap
    assert snap["sample_frac"] == 1.0, snap
    # the load-bearing one: `finished` alone pins the SAMPLES bar at 100%, so
    # only the stage bar can tell whether the stage count was really learned
    assert snap["stage_frac"] == 1.0, snap
    assert (snap["stage_idx"], snap["n_stages"]) == (len(stages), len(stages)), snap
    assert snap["finished"] and snap["stats"] == {"n_peaks": 7}, snap


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


# ---- 9. the clock KEEPS TICKING between log lines ---------------------------
# `snapshot()` freezes elapsed/eta at the instant it is taken, and Reporter only
# takes one when a log line arrives. A --jobs>1 run logs NOTHING between the
# worker banner and the first completed future (workers buffer, the parent
# replays after the reduce), so on a real batch the window went perfectly still
# for minutes -- clock, ETA and both bars -- which reads as a hung run. Observed
# live on a 5-worker batch against a real server: two screenshots 18 s apart
# both read "elapsed 00:34". These pin the re-tick that fixes it.

class _FakeRoot:
    """Just enough root for _drain: it only ever calls .after()."""
    def after(self, *a, **kw):
        return None


def _stale_snapshot(monkeypatch, *, age_s, done=0, n=11, assign_age_s=None, finished=False):
    """A snapshot taken `age_s` ago, as the run thread would have pushed it."""
    clock = {"t": 1000.0}
    monkeypatch.setattr(PG.time, "monotonic", lambda: clock["t"])
    st = PG.ProgressState(title="t", n_samples=n)
    # t0's default_factory captured the REAL time.monotonic at class-definition
    # time, so the patch above cannot reach it -- pin the origin explicitly.
    st.t0 = clock["t"]
    st.parallel, st.samples_done, st.finished = 5, done, finished
    if assign_age_s is not None:
        st.t_assign = clock["t"] - assign_age_s
    snap = st.snapshot()                      # frozen here
    clock["t"] += age_s                       # ... and now time passes, silently
    return snap, clock


def test_retick_advances_the_clock_between_log_lines(monkeypatch):
    snap, _ = _stale_snapshot(monkeypatch, age_s=18.0)
    check("snapshot freezes elapsed at the moment it is taken", snap["elapsed"] == 0.0)
    fresh = PG.retick(snap)
    check("retick moves elapsed forward by the silence (18 s)",
          abs(fresh["elapsed"] - 18.0) < 1e-6, fresh["elapsed"])
    check("retick leaves the run's FACTS alone (counts/phase/bars)",
          (fresh["samples_done"], fresh["n_samples"], fresh["phase"], fresh["sample_frac"])
          == (snap["samples_done"], snap["n_samples"], snap["phase"], snap["sample_frac"]))


def test_retick_advances_the_eta_too(monkeypatch):
    # 2 of 10 done, assignment started 20 s before the snapshot -> 10 s/sample,
    # eta 80 s. 10 s of silence later the same 2 samples took 30 s -> eta 120 s.
    snap, _ = _stale_snapshot(monkeypatch, age_s=10.0, done=2, n=10, assign_age_s=20.0)
    check("frozen eta extrapolates over the assign clock", abs(snap["eta"] - 80.0) < 1e-6,
          snap["eta"])
    fresh = PG.retick(snap)
    check("retick re-extrapolates the eta over the silence", abs(fresh["eta"] - 120.0) < 1e-6,
          fresh["eta"])


def test_retick_freezes_a_finished_run(monkeypatch):
    snap, _ = _stale_snapshot(monkeypatch, age_s=30.0, done=11, n=11, finished=True)
    check("a finished run's clock is a RESULT and must not drift",
          PG.retick(snap)["elapsed"] == snap["elapsed"])
    check("a snapshot with no time base is returned untouched",
          PG.retick({"finished": False})["finished"] is False)


def test_drain_hands_the_ui_a_reticked_snapshot(monkeypatch):
    """The bug was here, not in retick: _drain re-applied the FROZEN dict every
    120 ms, so the repaint was a no-op. Drive _drain with no Tk at all."""
    snap, _ = _stale_snapshot(monkeypatch, age_s=18.0)
    w = PG.TkWindow("t")                       # __init__ touches no Tk
    w.root, w._last = _FakeRoot(), snap
    seen = []
    w._apply = seen.append
    w._drain()
    check("_drain applied exactly one snapshot", len(seen) == 1, seen)
    check("_drain's snapshot has the ADVANCED clock, not the frozen one",
          seen and abs(seen[0]["elapsed"] - 18.0) < 1e-6,
          seen[0]["elapsed"] if seen else None)


def test_drain_still_drains_the_queue_and_reschedules(monkeypatch):
    snap, _ = _stale_snapshot(monkeypatch, age_s=1.0)
    w = PG.TkWindow("t")
    w.root = _FakeRoot()
    seen = []
    w._apply = seen.append
    w.push(snap)                               # newest state arrives via the queue
    w._drain()
    check("_drain takes the newest queued state", len(seen) == 1 and seen[0]["n_samples"] == 11)
    quits = []
    w._quit = lambda: quits.append(1)
    w.q.put(("quit", None))
    w._drain()
    check("_drain still honours a quit message", quits == [1])


# ---- 10. the stats panel is LIVE --------------------------------------------
# A 74-minute `peaky batch ... --jobs 1 --progress` run (15 files, 2026-09-13,
# launched under `setsid nohup` with DISPLAY set) opened a real window whose
# stats rows read '--' from start to finish: they were filled only by finish(),
# from the returned summary, and the window closed the instant the run ended.
# The fixture is that run's own log, cut to the lines the parser reads (the
# head and files 1-2 verbatim -- SDK noise, a validation-problems line and all
# -- then the marker + summary lines of files 3-15 and the tail), with the
# operator's home directory in the paths replaced by a neutral prefix.
FIXTURE_LOG = Path(__file__).resolve().parent / "fixtures" / "progress_run_batch_excerpt.txt"


def _fixture_lines() -> list:
    return FIXTURE_LOG.read_text(encoding="utf-8").splitlines()


def _fixture_stats() -> list:
    """The per-file stats dicts, parsed INDEPENDENTLY of progress.py."""
    import json
    return [json.loads(ln[len("[run] stats "):]) for ln in _fixture_lines()
            if ln.startswith("[run] stats ")]


def _replay(lines, until_done: int | None = None) -> "PG.ProgressState":
    """Feed `lines` into a fresh state; stop right after sample `until_done` is done."""
    st = PG.ProgressState(title="t")
    for ln in lines:
        st.feed(ln)
        if (until_done is not None and PG.RE_SAMPLE_DONE.match(ln)
                and st.samples_done >= until_done):
            break
    return st


def _sum(stats, get) -> int:
    return sum(get(s) for s in stats)


def test_live_totals_from_the_captured_batch_log(monkeypatch):
    """The whole run replayed: the totals equal an independent parse of the
    fixture's own stats lines, the first file's numbers read by eye off the log,
    and the DONE line's merge numbers land provisionally."""
    lines, stats = _fixture_lines(), _fixture_stats()
    assert len(stats) == 15
    st = _replay(lines)
    lv = st.snapshot()["live"]
    assert st.is_batch and lv["is_batch"]
    assert (st.samples_done, st.n_samples, lv["files"]) == (15, 15, 15)
    assert st.n_stages == 19            # this run's real stage count, learned from file 1
    assert lv["m0"] == _sum(stats, lambda s: s["by_role"]["M0"])
    assert lv["peaks"] == _sum(stats, lambda s: s["n_peaks"])
    assert lv["unexplained"] == _sum(stats, lambda s: s["by_role"]["unexplained"])
    want_tiers: dict = {}
    for s in stats:
        for k, v in s["by_tier"].items():
            want_tiers[k] = want_tiers.get(k, 0) + v
    assert lv["tiers"] == want_tiers and set(want_tiers) == {"Assigned", "Candidate"}
    assert lv["admitted"] == {k: _sum(stats, lambda s, k=k: s["admitted"][k])
                              for k in ("height", "occurrence", "rejected")}
    assert lv["last"] == stats[-1]
    # the first file, read by eye off the log
    first = _replay(lines, until_done=1).snapshot()["live"]
    assert (first["files"], first["m0"], first["peaks"], first["unexplained"]) == (1, 1869, 2715, 276)
    assert first["tiers"] == {"Assigned": 1451, "Candidate": 418}
    assert first["admitted"] == {"height": 2687, "occurrence": 9, "rejected": 19}
    # the DONE line: provisional merge numbers, and the phase moves as before
    assert lv["merged"] == {"merged_M0": 2636,
                            "merged_tiers": {"Assigned": 1692, "Candidate": 944},
                            "n_in_all_files": 846, "n_single_file": 375,
                            "formula_disagreements": 73}
    assert st.phase == "provenance"
    assert st.out_dir.endswith("BATCHxxxxxxxxxxx_2026-09-13T120802Z")


def test_batch_panel_reads_pending_merge_then_the_exact_summary(monkeypatch):
    lines, stats = _fixture_lines(), _fixture_stats()
    # before the first file is in: the block says so rather than showing zeros
    early = dict(PG.panel_rows(_replay(lines[:25]).snapshot()))
    assert early["samples"] == "0/15 done" and early["per-file M0 (sum)"] == "--", early
    assert early["merged M0"] == "pending merge", early
    # at 3/15 -- minute ~15 of the run: the per-file block is live, the merge block waits
    rows = dict(PG.panel_rows(_replay(lines, until_done=3).snapshot()))
    top = stats[:3]
    assert rows["samples"] == "3/15 done", rows
    assert rows["per-file M0 (sum)"] == str(_sum(top, lambda s: s["by_role"]["M0"]))
    assert rows["per-file tiers (sum)"] == (
        f"Assigned {_sum(top, lambda s: s['by_tier']['Assigned'])}  "
        f"Candidate {_sum(top, lambda s: s['by_tier']['Candidate'])}"), rows
    un, pk = _sum(top, lambda s: s["by_role"]["unexplained"]), _sum(top, lambda s: s["n_peaks"])
    assert rows["unexplained peaks"] == f"{100 * un / pk:.1f}%", rows
    assert rows["admitted by occurrence"] == str(_sum(top, lambda s: s["admitted"]["occurrence"]))
    for k in ("merged M0", "tiers", "in all files", "single-file", "formula disagreements"):
        assert rows[k] == "pending merge", (k, rows[k])
    assert rows["assignment"] == "--" and "total runtime" not in rows
    # after the DONE line, through the report tail: the merge numbers are readable
    # (provisionally, off the log) before finish() ever runs
    st = _replay(lines)
    snap = st.snapshot()
    rows = dict(PG.panel_rows(snap))
    assert not snap["finished"] and "total runtime" not in rows
    assert (rows["merged M0"], rows["in all files"], rows["single-file"],
            rows["formula disagreements"]) == ("2636", "846", "375", "73"), rows
    assert rows["tiers"] == "Assigned 1692  Candidate 944" and rows["samples"] == "15/15 done"
    # finish(): the exact summary replaces them -- numbers deliberately DIFFERENT
    # from the log's, to prove which source the final panel reads -- and the
    # per-file block stays beside it
    rep = PG.Reporter("t", log=seen.append, ui=None)
    rep.state = st
    rep.finish({"merged_M0": 2600, "merged_tiers": {"Assigned": 1700, "Candidate": 900},
                "n_files": 15, "n_in_all_files": 800, "n_single_file": 300,
                "formula_disagreements": 70, "elapsed_s": 4205.6})
    rows = dict(PG.panel_rows(st.snapshot()))
    assert (rows["merged M0"], rows["in all files"], rows["samples"]) == ("2600", "800", "15")
    assert rows["tiers"] == "Assigned 1700  Candidate 900"
    assert rows["assignment"] == "1:10:05" and rows["total runtime"] != "--"
    assert rows["per-file M0 (sum)"] == str(_sum(stats, lambda s: s["by_role"]["M0"]))
    # a run that FAILED before any summary has nothing pending: its gaps read '--'
    fail = _replay(lines, until_done=3)
    rep = PG.Reporter("t", log=seen.append, ui=None)
    rep.state = fail
    rep.finish({}, error="RuntimeError: boom")
    rows = dict(PG.panel_rows(fail.snapshot()))
    assert rows["merged M0"] == "--" and rows["per-file M0 (sum)"] != "--", rows


_STATS_LINE = ('[run] stats {"n_peaks": 100, "by_role": {"M0": 60, "iso_child": 20, '
               '"reagent": 0, "artifact": 0, "unexplained": 20}, '
               '"by_tier": {"Assigned": 50, "Candidate": 10}, '
               '"admitted": {"height": 95, "occurrence": 5, "rejected": 3}}')


def test_per_file_summary_lines_count_a_file_once(monkeypatch):
    st = PG.ProgressState(title="t")
    st.feed("[assign_batch] (1/4) assigning s1 ...")
    assert st.feed("[run] tiers {'Assigned': 55, 'Candidate': 5}")
    lv = st.snapshot()["live"]
    assert lv["files"] == 0 and lv["tiers"] == {}       # a tiers line alone counts nothing yet
    assert st.feed(_STATS_LINE)
    lv = st.snapshot()["live"]
    # the tiers line is taken over the stats line's own by_tier, and counted ONCE
    assert lv["files"] == 1 and lv["tiers"] == {"Assigned": 55, "Candidate": 5}, lv
    assert (lv["m0"], lv["peaks"], lv["unexplained"]) == (60, 100, 20)
    assert lv["admitted"] == {"height": 95, "occurrence": 5, "rejected": 3}
    # no tiers line ahead of the stats line: its by_tier is the fallback
    st.feed("[assign_batch] (1/4) done s1")
    st.feed("[assign_batch] (2/4) assigning s2 ...")
    st.feed(_STATS_LINE)
    lv = st.snapshot()["live"]
    assert lv["files"] == 2 and lv["tiers"] == {"Assigned": 105, "Candidate": 15}, lv
    # a file that logs its tiers and then fails before its stats line is not
    # half-counted, and its tiers are not charged to the next file either
    st.feed("[assign_batch] (3/4) assigning s3 ...")
    st.feed("[run] tiers {'Assigned': 999}")
    st.feed("[assign_batch] (4/4) assigning s4 ...")
    st.feed(_STATS_LINE)
    lv = st.snapshot()["live"]
    assert lv["files"] == 3 and lv["tiers"] == {"Assigned": 155, "Candidate": 25}, lv
    # parallel mode: the per-file lines arrive in the replay burst after the
    # reduce, and unlike the stage lines they ARE read there
    par = PG.ProgressState(title="t")
    par.feed("[assign_batch] parallel: 3 worker processes (match-workers/proc=4) over 2 samples")
    par.feed("[run] pass0 took 1.0s")
    par.feed(_STATS_LINE)
    par.feed(_STATS_LINE)
    assert par.stage_idx == 0 and par.snapshot()["live"]["m0"] == 120


def test_garbled_summary_lines_are_not_counted(monkeypatch):
    st = PG.ProgressState(title="t")
    st.feed("[assign_batch] (1/2) assigning s1 ...")
    st.feed(_STATS_LINE)
    for bad in ('[run] stats {"n_peaks": ', "[run] stats []", "[run] stats {\"n_peaks\": \"many\"}",
                "[run] tiers {'Assigned': ", "[run] tiers 7", "[run] tiers [1, 2]",
                "[run] tiers {'Assigned': 'lots'}"):
        assert st.feed(bad) is False, bad
    lv = st.snapshot()["live"]
    assert lv["files"] == 1 and lv["m0"] == 60 and lv["tiers"] == {"Assigned": 50, "Candidate": 10}
    # a DONE line whose tail was reworded still moves the phase; the rows just wait
    assert st.feed("[assign_batch] DONE: 7 merged M0 and then something else")
    assert st.phase == "merged" and lv["merged"] == {} and st.snapshot()["live"]["merged"] == {}
    # ... and the Reporter wrapper stays never-fatal around all of it
    rep = PG.Reporter("t", log=seen.append, ui=_Boom())
    rep('[run] stats {"n_peaks": 1, "by_role": {"M0": "x"}}')
    rep("[run] tiers {'A': 1}")
    assert rep.state.live.files == 0


def test_single_sample_panel_fills_from_its_own_stats_line(monkeypatch):
    """`peaky assign` logs no batch marker at all, so its panel must never say
    'pending merge' -- and its one stats line IS the final panel, so it shows as
    soon as the line lands, before the report writes."""
    one = PG.ProgressState(title="t", n_samples=1)
    one.feed("[run] pass0 took 0.2s")
    rows = dict(PG.panel_rows(one.snapshot()))
    assert "pending merge" not in rows.values() and rows["merged M0"] == "--", rows
    one.feed("[run] tiers {'Assigned': 250, 'Candidate': 50}")
    one.feed('[run] stats {"n_peaks": 1000, "by_role": {"M0": 300, "iso_child": 100, '
             '"reagent": 50, "unexplained": 550}, "signal_by_role": {"M0": 0.5, '
             '"iso_child": 0.1, "reagent": 0.3}, "count_frac_by_role": {"unexplained": 0.55}}')
    snap = one.snapshot()
    assert not snap["is_batch"]
    rows = dict(PG.panel_rows(snap))
    assert rows["assigned M0"] == "300" and rows["peaks explained"] == "45.0%", rows
    assert rows["signal explained"] == "90.0%" and rows["total runtime"] == "--", rows
    one.mark_sample_done()
    one.stats, one.finished = dict(snap["live"]["last"]), True
    rows = dict(PG.panel_rows(one.snapshot()))
    assert rows["assigned M0"] == "300" and rows["total runtime"] != "--", rows


def test_window_grid_is_live_before_finish(monkeypatch):
    """The Tk path itself, with tkinter faked: `_apply` must put the running
    totals into the stats grid on an UNFINISHED snapshot -- this is the grid
    that read '--' for 74 minutes -- and swap in the summary at finish()."""
    import types as _types

    class _Var:
        def __init__(self, value=""): self.value = value
        def set(self, v): self.value = v
        def get(self): return self.value

    class _Widget:
        def __init__(self, *a, **k): pass
        def grid(self, *a, **k): pass

    class _Frame:
        def winfo_children(self): return []
        def grid_columnconfigure(self, *a, **k): pass

    class _Btn:
        def __init__(self): self.states = []
        def state(self, s): self.states.append(tuple(s))
        def focus_set(self): pass

    fake = _types.ModuleType("tkinter")
    fake.StringVar, fake.Label = _Var, _Widget
    monkeypatch.setitem(sys.modules, "tkinter", fake)
    w = PG.TkWindow("t")                                   # __init__ touches no Tk
    for attr in ("v_head", "v_sample", "v_stage", "v_phase", "v_clock", "v_tail"):
        setattr(w, attr, _Var())
    w.pb_sample, w.pb_stage, w.stats_frame, w.btn = {}, {}, _Frame(), _Btn()

    lines, stats = _fixture_lines(), _fixture_stats()
    st = _replay(lines, until_done=3)
    w._apply(st.snapshot())
    grid = {k: v.get() for k, v in w._stats_labels.items()}
    assert grid["samples"] == "3/15 done", grid
    assert grid["per-file M0 (sum)"] == str(_sum(stats[:3], lambda s: s["by_role"]["M0"])), grid
    assert grid["merged M0"] == "pending merge", grid
    assert w.v_sample.get() == "3/15" and w.btn.states == []    # unfinished: Close stays disabled
    # finish(): the exact numbers, the runtime row, Close enabled
    rep = PG.Reporter("t", log=seen.append, ui=None)
    rep.state = st
    rep.finish({"merged_M0": 2636, "merged_tiers": {"Assigned": 1692, "Candidate": 944},
                "n_files": 15, "n_in_all_files": 846, "n_single_file": 375,
                "formula_disagreements": 73, "elapsed_s": 4205.6})
    w._apply(st.snapshot())
    grid = {k: v.get() for k, v in w._stats_labels.items()}
    assert grid["merged M0"] == "2636" and grid["samples"] == "15", grid
    assert grid["per-file M0 (sum)"] == str(_sum(stats[:3], lambda s: s["by_role"]["M0"]))
    assert "total runtime" in grid and w.v_phase.get() == "done"
    assert w.btn.states == [("!disabled",)]


# ---- 11. the hold: a person at a terminal, OR a window that was asked for -----
# The run above was launched under `setsid nohup` with DISPLAY set: `--progress`
# was passed, a real window came up, and the hold was refused for want of a tty,
# so the window vanished the instant the run finished.
def test_hold_decision_table(monkeypatch):
    monkeypatch.delenv("PEAKY_PROGRESS_HOLD_S", raising=False)
    table = [
        # flag,  env,  tty,   window, want
        (True,  None, False, True,  True),    # --progress under nohup with DISPLAY: the reported case
        (None,  "1",  False, True,  True),    # PEAKY_PROGRESS=1 is the same request
        (None,  "yes", False, True, True),
        (None,  None, False, True,  False),   # not asked for: never wait off a terminal
        (None,  "0",  False, True,  False),   # an explicit OFF is not a request either
        (True,  None, False, False, False),   # asked for, but only the terminal fallback came up
        (None,  "1",  False, False, False),
        (None,  None, True,  True,  True),    # a person at a terminal: as before
        (None,  None, True,  False, True),    # (moot for the terminal fallback, which ignores it)
        (True,  None, True,  True,  True),
        (False, "1",  False, True,  False),   # --no-progress is not a request
    ]
    for flag, env, tty, window, want in table:
        if env is None:
            monkeypatch.delenv("PEAKY_PROGRESS", raising=False)
        else:
            monkeypatch.setenv("PEAKY_PROGRESS", env)
        monkeypatch.setattr(PG, "_interactive", lambda tty=tty: tty)
        assert PG.hold_wanted(flag, window_up=window) is want, (flag, env, tty, window)
    # the bounds win over everything: a zero hold, or the call site saying no
    monkeypatch.delenv("PEAKY_PROGRESS", raising=False)
    monkeypatch.setattr(PG, "_interactive", lambda: True)
    assert PG.hold_wanted(True, window_up=True, hold=False) is False
    monkeypatch.setenv("PEAKY_PROGRESS_HOLD_S", "0")
    assert PG.hold_wanted(True, window_up=True) is False
    monkeypatch.setenv("PEAKY_PROGRESS_HOLD_S", "30")
    assert PG.hold_wanted(True, window_up=True) is True


class _UpWindow:
    """A TkWindow stand-in that comes up whenever the display probe says one
    can (so the headless branch is still reachable), and records how it was
    closed. The probe is PATCHED by every test that uses this -- never left to
    the machine: CI has no display, a developer's box does."""
    def __init__(self, title): self.title, self.closed_with = title, None
    def start(self, timeout=5.0): return PG.display_available()
    def alive(self): return True
    def push(self, snap): pass
    def close(self, wait, timeout=None): self.closed_with = wait


def test_open_progress_holds_an_explicit_window_off_a_tty(monkeypatch):
    monkeypatch.setattr(PG, "TkWindow", _UpWindow)
    monkeypatch.setattr(PG, "display_available", lambda: True)    # a window CAN come up
    monkeypatch.setattr(PG, "_interactive", lambda: False)
    monkeypatch.delenv("PEAKY_PROGRESS_HOLD_S", raising=False)
    monkeypatch.delenv("PEAKY_PROGRESS", raising=False)
    rep = PG.open_progress("t", flag=True, log=seen.append)
    assert isinstance(rep.ui, _UpWindow) and rep.hold is True
    with rep:
        rep("[assign_batch] (1/1) assigning s ...")
        rep("[assign_batch] (1/1) done s")
        rep.finish({"merged_M0": 1})
    assert rep.ui.closed_with is True          # __exit__ waited (bounded by hold_seconds)
    # ... and the CLI says so, since the process is now visibly still alive
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        CLI._progress_hold_note(rep)
    assert "close the progress window" in buf.getvalue(), buf.getvalue()
    monkeypatch.setenv("PEAKY_PROGRESS", "1")
    assert PG.open_progress("t", flag=None, log=seen.append).hold is True
    monkeypatch.delenv("PEAKY_PROGRESS")
    # a Ctrl-C mid-run still closes WITHOUT waiting, hold or no hold
    rep = PG.open_progress("t", flag=True, log=seen.append)
    try:
        with rep:
            raise KeyboardInterrupt()
    except KeyboardInterrupt:
        pass
    assert rep.ui.closed_with is False
    # the terminal fallback under the same explicit request never waits (the
    # existing rule): nothing came up that could be read
    monkeypatch.setattr(PG, "display_available", lambda: False)
    monkeypatch.setenv("PEAKY_PROGRESS", "1")
    rep = PG.open_progress("t", flag=True, log=seen.append)
    assert isinstance(rep.ui, PG.TerminalStatus) and rep.hold is False
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        CLI._progress_hold_note(rep)
    assert buf.getvalue() == ""


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
