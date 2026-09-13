"""Live progress window for a run — ON by default at an interactive terminal;
`--no-progress` (or `PEAKY_PROGRESS=0`) turns it off, and it never defaults on for
a pipe, a CI job or a skill-driven run (see `enabled`).

Peaky already threads a `log=print` callable through every level of a run
(`pipeline.run_batch` -> `assign_batch.run` -> `assign.run` -> each of its ~36
stages), and the lines that callable receives already carry the run's shape:
`(i/N) assigning <sid>`, `(i/N) done <sid>`, `[run] <stage> took Xs`, and at the
end of every file the two summary lines `[run] tiers {...}` / `[run] stats {...}`
that the stats panel sums into running per-file totals. So this module is
nothing more than a **`log` wrapper**: it forwards every line untouched to the
real log and, on the side, reads those lines into a progress model that drives
a small Tk window.

THE DEPENDENCY IS ONE-WAY ON PURPOSE. Nothing in the pipeline imports this module
or knows a window exists -- the LOG STREAM is the entire interface, and `Reporter`
is a drop-in for `print`. That keeps the science modules free of UI concerns, and
it means the only thing that can break the window is a change to the log TEXT --
which `tests/test_progress.py` pins by asserting the real emitted strings still
match the patterns parsed here, so such a change fails a test rather than
silently flat-lining the bar.

NEVER FATAL. No display, no tkinter, or any UI exception whatsoever degrades to a
one-line terminal status and then to silence. A progress window must not be able
to kill a two-hour assignment run, so every UI call site is guarded and the run's
own output is never altered by whether the window is there.

Parallel runs (`--jobs > 1`) report at SAMPLE granularity only: workers buffer
their logs and the parent replays them after the reduce (`assign_batch.run`), so
per-stage lines arrive in a burst at the end and would make a stage bar lie. The
parser detects the parallel banner and stops driving the stage bar from then on.
"""
from __future__ import annotations

import ast
import json
import os
import queue
import re
import sys
import threading
import time
from dataclasses import dataclass, field

__version__ = "0.1.0"

# Stage count for the within-sample bar before a first sample has finished. The
# real number is LEARNED from the first completed sample (stages are gated by
# reagent/context, so it legitimately varies), which keeps this module from
# importing `assign` just to count its stage table.
#
# What it counts is the stages that TIME THEMSELVES: only an `assign._STAGES`
# entry with `safe=True` emits `[run] <tag> took Xs`, and that line is the
# parser's only cue. Counting all 36 table rows (20 of which are timed) capped
# the bar at 56% for any run that never got to learn the real number -- i.e.
# every `peaky assign` run. tests/test_progress.py pins this against the real
# stage table, so adding a timed stage fails a test rather than skewing the bar.
NOMINAL_STAGES = 20

# Every phase the window can show, in run order (`assign` is the long pole; the
# rest are the tail). A phase with no entry here falls back to its bare name, so
# the contract test checks each `[phase]` marker the pipeline emits is present.
PHASE_LABEL = {
    "fetch": "fetching time series", "select": "selecting samples",
    "assign": "assigning", "merge": "merging ledgers",
    "merged": "ledgers merged", "cluster": "clustering",
    "vankrevelen": "Van Krevelen", "report": "building report",
    "provenance": "recording provenance", "done": "done",
}

# --------------------------------------------------------------------------- #
# 1. the log grammar -- the ONLY coupling to the rest of peaky.
#    Every pattern here is pinned by tests/test_progress.py against the literal
#    f-string that emits it, so a reworded log line fails a test.
# --------------------------------------------------------------------------- #
RE_ASSIGNING = re.compile(r"^\[assign_batch\] \((\d+)/(\d+)\) assigning (\S+)")
RE_SAMPLE_DONE = re.compile(r"^\[assign_batch\] \((\d+)/(\d+)\) done (\S+)")
# the sample count is OPTIONAL on purpose: losing it costs an early samples bar,
# but failing to match the banner at all would let replayed worker stage lines
# drive the stage bar, which is the one thing parallel mode must never do.
RE_PARALLEL = re.compile(r"^\[assign_batch\] parallel: (\d+) worker processes"
                         r"(?:.*?\bover (\d+) samples)?")
# The prefix alone moves the phase (it is all the parser needed before the panel
# went live); the tail is OPTIONAL and gives the merge numbers a provisional
# reading through the report tail, until `finish()` brings the exact summary.
RE_ASSIGN_DONE = re.compile(
    r"^\[assign_batch\] DONE: (\d+) merged M0"
    r"(?: \((\{.*\})\); (\d+) in all files, (\d+) single-file, "
    r"(\d+) formula disagreements)?")
RE_STAGE = re.compile(r"^\[run\] (\S+) took ([\d.]+)s")
# The two lines `assign.run` logs as a file ends: the M0 tier counts as a dict
# repr, then `ledger.stats` (+ the admission counts) as JSON. Each is read whole
# and summed into `LiveTotals`; a line that fails to parse is simply not counted.
RE_FILE_TIERS = re.compile(r"^\[run\] tiers (\{.*\})$")
RE_FILE_STATS = re.compile(r"^\[run\] stats (\{.*\})$")
RE_PHASE = re.compile(r"^\[phase\] (\w+)")
RE_RUNDIR = re.compile(r"^\[(?:batch|pool)\] (\S+) -> (\S+)$")


def _hms(sec: float | None) -> str:
    """Compact clock: 92 -> '01:32', 3730 -> '1:02:10', None -> '--:--'."""
    if sec is None or sec < 0:
        return "--:--"
    sec = int(sec)
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


@dataclass
class LiveTotals:
    """Running per-file totals for the stats panel, summed from the two summary
    lines `assign.run` logs at the end of every file: `[run] tiers {...}` (the
    M0 tier counts, a dict repr) and `[run] stats {...}` (the JSON of
    `ledger.stats` plus the admission counts). They are PARSED, unlike the
    numbers `finish()` brings, and shown for what they are -- sums over the
    files done so far -- so a batch window has something to say during the
    hour its files take instead of a column of '--' (which is all a panel fed
    by the final summary alone could show, and on a run that closed on
    finishing all anyone ever saw).

    A file counts ONCE, when its stats line lands: the tiers line just ahead of
    it is kept aside and folded in with the rest, so the totals never move in
    two half-steps and a file that never reaches its stats line (a failure in
    between) is not half-counted. The `by_tier` inside the stats line is the
    fallback for a tiers line that never came. The merge-level numbers (merged
    M0, in all files, single-file, formula disagreements) do not exist until
    `align()` has run; the DONE line then gives them a provisional reading
    (`merged`), replaced by the exact summary at `finish()`."""

    files: int = 0                  # files whose stats line has been read
    m0: int = 0                     # sum of by_role.M0
    peaks: int = 0                  # sum of n_peaks
    unexplained: int = 0            # sum of by_role.unexplained
    tiers: dict = field(default_factory=dict)      # per-file M0 tier counts, summed
    admitted: dict = field(default_factory=dict)   # admitted.{height,occurrence,rejected}, summed
    last: dict = field(default_factory=dict)       # the last file's stats, verbatim
    merged: dict = field(default_factory=dict)     # provisional, from the DONE line
    pending_tiers: dict | None = None              # this file's tiers line, awaiting its stats line

    def new_file(self) -> None:
        """A file is starting: a tiers line left over from one that never
        reached its stats line must not be charged to this one."""
        self.pending_tiers = None

    def read_tiers(self, payload: str) -> bool:
        try:
            d = ast.literal_eval(payload)
            if not isinstance(d, dict):
                return False
            self.pending_tiers = {str(k): int(v) for k, v in d.items()}
        except Exception:
            return False
        return True

    def read_stats(self, payload: str) -> bool:
        try:
            st = json.loads(payload)
            if not isinstance(st, dict):
                return False
            role = st.get("by_role") or {}
            tiers = (self.pending_tiers if self.pending_tiers is not None
                     else (st.get("by_tier") or {}))
            # read EVERY field before touching the sums: a half-parsed file
            # must not skew them
            m0 = int(role.get("M0", 0))
            peaks = int(st.get("n_peaks", 0))
            unexplained = int(role.get("unexplained", 0))
            tiers = {str(k): int(v) for k, v in dict(tiers).items()}
            admitted = {str(k): int(v) for k, v in dict(st.get("admitted") or {}).items()}
        except Exception:
            return False
        self.files += 1
        self.m0 += m0
        self.peaks += peaks
        self.unexplained += unexplained
        for k, v in tiers.items():
            self.tiers[k] = self.tiers.get(k, 0) + v
        for k, v in admitted.items():
            self.admitted[k] = self.admitted.get(k, 0) + v
        self.last = st
        self.pending_tiers = None
        return True

    def read_done(self, m: re.Match) -> None:
        """The DONE line's merge numbers -- provisional until `finish()`. A
        reworded tail (no groups) still moves the phase; the rows then wait."""
        if m.group(3) is None:
            return
        try:
            tiers = ast.literal_eval(m.group(2))
        except Exception:
            tiers = {}
        self.merged = {"merged_M0": int(m.group(1)),
                       "merged_tiers": tiers if isinstance(tiers, dict) else {},
                       "n_in_all_files": int(m.group(3)),
                       "n_single_file": int(m.group(4)),
                       "formula_disagreements": int(m.group(5))}

    def as_dict(self) -> dict:
        return {"files": self.files, "m0": self.m0, "peaks": self.peaks,
                "unexplained": self.unexplained, "tiers": dict(self.tiers),
                "admitted": dict(self.admitted), "last": dict(self.last),
                "merged": dict(self.merged)}


@dataclass
class ProgressState:
    """What the window draws. Advanced ONLY by `feed()` reading log lines, plus
    `finish()` taking the run's own returned summary dict (exact, not parsed)."""

    title: str = "peaky"
    phase: str = "fetch"
    n_samples: int = 0
    samples_done: int = 0
    current_sid: str = ""
    parallel: int = 0             # worker count; 0 = serial (stage bar is live)
    stage_name: str = ""
    stage_idx: int = 0
    n_stages: int = NOMINAL_STAGES
    t0: float = field(default_factory=time.monotonic)
    t_assign: float | None = None  # assign-phase clock (first assigning line / parallel banner)
    out_dir: str = ""
    last_line: str = ""
    stats: dict = field(default_factory=dict)
    finished: bool = False
    error: str = ""
    live: LiveTotals = field(default_factory=LiveTotals)
    # Only a batch/pool run logs `[assign_batch]`, `[phase]`, `[batch]`/`[pool]`
    # lines; `peaky assign` never does. The panel needs to know which it is
    # looking at: a merge is only "pending" where there is going to be one.
    is_batch: bool = False

    # -- derived ----------------------------------------------------------- #
    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.t0

    @property
    def assign_elapsed(self) -> float:
        """Seconds since assignment proper began -- the per-sample rate must not
        be diluted by the time-series fetch and sample selection before it."""
        return time.monotonic() - (self.t0 if self.t_assign is None else self.t_assign)

    @property
    def eta(self) -> float | None:
        """Linear extrapolation over COMPLETED samples. None until one lands --
        an ETA from zero completions is a guess dressed as a number."""
        if self.finished or not self.samples_done or not self.n_samples:
            return None
        if self.samples_done >= self.n_samples:
            return None            # samples all in; the report tail is not modelled
        per = self.assign_elapsed / self.samples_done
        return max(0.0, per * (self.n_samples - self.samples_done))

    @property
    def sample_frac(self) -> float:
        if self.finished:
            return 1.0
        return self.samples_done / self.n_samples if self.n_samples else 0.0

    @property
    def stage_frac(self) -> float:
        if self.parallel or not self.n_stages:
            return 0.0
        return min(1.0, self.stage_idx / self.n_stages)

    def _start_assign_clock(self) -> None:
        if self.t_assign is None:
            self.t_assign = time.monotonic()

    def _learn_stages(self) -> None:
        """Learn the real stage count from the sample that just completed -- the
        stages are gated by reagent/context, so the count is a property of the
        run, not a constant. Only a serial run has live stage lines to count."""
        if not self.parallel and self.stage_idx:
            self.n_stages = self.stage_idx

    def mark_sample_done(self, count: int = 1) -> None:
        """Record completed samples that no `(i/N) done` line announced.

        `peaky assign` runs ONE sample through `assign.run` directly -- there is
        no assign_batch on that path, so the log stream is stage lines and
        nothing else, and the samples bar would sit at 0/1 for the whole run
        while the stage bar never learned how many stages this run really has.
        The same two effects the `done` line has, minus the stage-bar reset:
        this is the last sample, so the stage bar stays full rather than
        dropping to zero at the very end."""
        self.samples_done = max(self.samples_done, int(count))
        self.n_samples = max(self.n_samples, self.samples_done)
        self._learn_stages()

    def feed(self, line: str) -> bool:
        """Read one log line into the model. Returns True if anything changed.
        Unrecognised lines only refresh `last_line` (the window's activity tail)."""
        line = str(line).strip()
        if not line:
            return False
        self.last_line = line[:160]
        if line.startswith(("[assign_batch]", "[phase]", "[batch]", "[pool]")):
            self.is_batch = True

        if (m := RE_ASSIGNING.match(line)):
            i = int(m.group(1))
            self._start_assign_clock()
            self.phase = "assign"
            self.n_samples = int(m.group(2))
            self.current_sid = m.group(3)
            self.live.new_file()
            if i > 1:
                # The i-th 'assigning' means i-1 samples are complete, whether or
                # not a per-sample 'done' line was logged in between (an emitter
                # without one must not leave the samples bar dead), and the stage
                # lines seen since the last reset are the previous sample's count.
                self.samples_done = max(self.samples_done, i - 1)
                self._learn_stages()
            self.stage_idx, self.stage_name = 0, ""   # new sample -> restart stage bar
            return True
        if (m := RE_SAMPLE_DONE.match(line)):
            self._start_assign_clock()
            self.samples_done, self.n_samples = int(m.group(1)), int(m.group(2))
            # In parallel mode this names the sample that just FINISHED while
            # others are still running; under an "assigning" header it would
            # read as the current one. Only a serial run has a current sample.
            self.current_sid = "" if self.parallel else m.group(3)
            self._learn_stages()
            self.stage_idx, self.stage_name = 0, ""
            if self.samples_done >= self.n_samples:
                # last sample in -> what runs next is the merge (align + guards
                # + TS stamp), up to the DONE line
                self.phase, self.current_sid = "merge", ""
            else:
                self.phase = "assign"
            return True
        if (m := RE_PARALLEL.match(line)):
            # Workers replay their logs only after the reduce, so stage lines stop
            # being live here. Freeze the stage bar rather than animate a lie.
            self._start_assign_clock()
            self.parallel = int(m.group(1))
            if m.group(2):          # "... over N samples": nothing else says N
                self.n_samples = int(m.group(2))
            self.phase = "assign"
            self.current_sid = ""
            self.stage_idx, self.stage_name = 0, ""
            return True
        if (m := RE_ASSIGN_DONE.match(line)):
            # logged AFTER align(): the merge is over, the report tail is next
            self.samples_done = self.n_samples or self.samples_done
            self.phase = "merged"
            self.current_sid = ""          # no longer inside any one sample
            self.live.read_done(m)         # provisional; finish() brings the exact numbers
            return True
        # The per-file summary lines are exact per-file numbers whenever they
        # arrive -- in a parallel run that is the replay burst after the reduce,
        # so unlike the stage lines they are read in parallel mode too.
        if (m := RE_FILE_TIERS.match(line)):
            return self.live.read_tiers(m.group(1))
        if (m := RE_FILE_STATS.match(line)):
            return self.live.read_stats(m.group(1))
        if (m := RE_STAGE.match(line)) and not self.parallel:
            self.phase = "assign"          # `peaky assign` has no 'assigning' line
            self.stage_name = m.group(1)
            self.stage_idx += 1
            self.n_stages = max(self.n_stages, self.stage_idx)
            return True
        if (m := RE_PHASE.match(line)):
            self.phase = m.group(1)
            if self.phase != "assign":
                self.current_sid = ""
            return True
        if (m := RE_RUNDIR.match(line)):
            self.out_dir = m.group(2)
            return True
        return False

    def snapshot(self) -> dict:
        """A plain dict handed across the thread boundary -- the UI thread never
        touches this object, so there is nothing to race on."""
        return {"title": self.title, "phase": self.phase,
                "phase_label": PHASE_LABEL.get(self.phase, self.phase),
                "n_samples": self.n_samples, "samples_done": self.samples_done,
                "current_sid": self.current_sid, "parallel": self.parallel,
                "stage_name": self.stage_name, "stage_idx": self.stage_idx,
                "n_stages": self.n_stages, "sample_frac": self.sample_frac,
                "stage_frac": self.stage_frac, "elapsed": self.elapsed,
                "eta": self.eta, "out_dir": self.out_dir,
                "last_line": self.last_line, "stats": dict(self.stats),
                "finished": self.finished, "error": self.error,
                "is_batch": self.is_batch,
                # the running totals, plus the run shape they are read against
                # (`summary_rows` takes this dict alone)
                "live": {**self.live.as_dict(), "is_batch": self.is_batch,
                         "samples_done": self.samples_done,
                         "n_samples": self.n_samples},
                # The TIME BASE, not just the derived clock: `elapsed`/`eta` above
                # are frozen at the instant this snapshot is taken, and a snapshot
                # is only taken when a log line arrives. `retick` needs the origin
                # to recompute them against the current time between log lines.
                "t0": self.t0, "t_assign": self.t_assign}


def retick(s: dict) -> dict:
    """A copy of snapshot `s` with `elapsed`/`eta` recomputed against the CURRENT
    time; `s` itself when it is finished or carries no time base.

    `snapshot()` freezes the clock at the moment it is taken, and `Reporter`
    only takes one when a log line arrives. A PARALLEL run logs nothing at all
    between the worker banner and the first completed future -- workers buffer
    their output and the parent replays it after the reduce -- so on a real
    batch that silence runs to minutes. Without this the window's clock, its
    ETA and both bars sit perfectly still for that whole stretch, which is
    indistinguishable from a hung run: precisely the stretch the window exists
    to reassure the watcher through, failing in precisely the way that matters.

    Only the wall-clock fields move here. Counts, phase and bars are facts about
    the run and may only change when the run says so."""
    if not isinstance(s, dict) or s.get("finished") or s.get("t0") is None:
        return s
    out = dict(s)
    now = time.monotonic()
    out["elapsed"] = now - s["t0"]
    done, n = s.get("samples_done") or 0, s.get("n_samples") or 0
    if done and n and done < n:
        # same extrapolation as ProgressState.eta, over the ASSIGN clock
        base = s.get("t_assign") or s["t0"]
        out["eta"] = max(0.0, (now - base) / done * (n - done))
    return out


# --------------------------------------------------------------------------- #
# 2. summary -> the stats panel. The FINAL numbers come from the dict
#    `assign_batch.run` returns, so they are EXACT (never parsed out of the
#    log); while the run is going the panel shows the running per-file totals
#    (`LiveTotals`, parsed) and says which numbers are still to come.
# --------------------------------------------------------------------------- #
def summary_rows(summary: dict | None, elapsed: float | None = None,
                 live: dict | None = None) -> list:
    """[(label, value)] for the stats panel. Handles BOTH shapes a run returns:
    a batch summary (`assign_batch.run`) and a single sample's ledger stats
    (`ledger.stats`, keyed by role) -- `peaky assign` and `peaky batch` count
    different things, and showing a column of '--' for the other one is worse
    than showing the numbers that sample actually has.

    `live` is `snapshot()["live"]`: the running per-file totals plus the run
    shape they are read against. With no summary yet, a batch panel is built
    from them, and a single-sample run's one stats line IS its final panel, so
    it shows as soon as the line lands rather than after the report writes.
    `elapsed` is given only once the run is over (it is the window's own
    clock), so `elapsed is None` is also how a row knows it may say "pending"."""
    s = summary or {}
    lv = live or {}
    if "by_role" in s:
        return _single_sample_rows(s, elapsed)
    if not s and lv.get("files") and not lv.get("is_batch"):
        return _single_sample_rows(lv.get("last") or {}, elapsed)
    return _batch_rows(s, elapsed, lv)


def _tier_text(tiers: dict) -> str:
    return "  ".join(f"{k} {v}" for k, v in (tiers or {}).items())


def _batch_rows(s: dict, elapsed: float | None, lv: dict) -> list:
    """The batch panel: the per-file block first (what is knowable while the
    files are still being assigned -- sums over the files done so far), then
    the merge block, then the two clocks.

    The merge rows read "pending merge" only while the run is going, only for a
    run that IS a batch (a single-sample run has no merge to wait for) and only
    until some merge numbers exist: the DONE line's provisional ones through
    the report tail, then the exact summary. A finished run with no summary
    (a failure) has nothing pending, so its gaps read '--'."""
    running = elapsed is None
    files = int(lv.get("files") or 0)
    done, n = int(lv.get("samples_done") or 0), int(lv.get("n_samples") or 0)
    peaks, unexpl = int(lv.get("peaks") or 0), int(lv.get("unexplained") or 0)
    admitted = lv.get("admitted") or {}
    # the exact summary once it exists, else the DONE line's provisional reading
    merged = s or lv.get("merged") or {}
    gap = "pending merge" if (running and lv.get("is_batch") and not merged) else "--"
    if s.get("n_files") is not None:
        samples = s["n_files"]
    else:
        samples = f"{done}/{n} done" if n else "--"
    rows = [
        ("samples", samples),
        ("per-file M0 (sum)", lv.get("m0") if files else "--"),
        ("per-file tiers (sum)", (_tier_text(lv.get("tiers")) or "--") if files else "--"),
        ("unexplained peaks", f"{100 * unexpl / peaks:.1f}%" if peaks else "--"),
        ("admitted by occurrence", admitted.get("occurrence", "--") if files else "--"),
        ("merged M0", merged.get("merged_M0", gap)),
        ("tiers", _tier_text(merged.get("merged_tiers")) or gap),
        ("in all files", merged.get("n_in_all_files", gap)),
        ("single-file", merged.get("n_single_file", gap)),
        ("formula disagreements", merged.get("formula_disagreements", gap)),
    ]
    # TWO different durations, both worth seeing: the assignment itself (measured
    # by assign_batch, in the summary) and the whole command (measured by the
    # window). The gap between them is the fetch + cluster + VK + report tail.
    assign_s = s.get("elapsed_s")
    rows.append(("assignment", _hms(assign_s) if assign_s is not None else "--"))
    if elapsed is not None:
        rows.append(("total runtime", _hms(elapsed)))
    return [(k, str(v)) for k, v in rows]


def panel_rows(s: dict) -> list:
    """The stats panel for a snapshot: the running totals while the run is
    going, the exact summary (with the totals kept beside it) once `finish()`
    has landed. Pure -- no Tk -- so the live panel is testable headless."""
    if s.get("finished"):
        return summary_rows(s.get("stats"), s.get("elapsed"), live=s.get("live"))
    return summary_rows(None, None, live=s.get("live"))


def _single_sample_rows(s: dict, elapsed: float | None) -> list:
    """Stats panel for `peaky assign` (one sample): the coverage numbers the CLI
    already prints, so the window and the terminal agree."""
    role = s.get("by_role") or {}
    sig = s.get("signal_by_role") or {}
    cf = s.get("count_frac_by_role") or {}
    explained_sig = 100 * (sig.get("M0", 0) + sig.get("iso_child", 0)
                           + sig.get("reagent", 0))
    rows = [
        ("peaks", s.get("n_peaks", "--")),
        ("assigned M0", role.get("M0", "--")),
        ("isotope children", role.get("iso_child", "--")),
        ("reagent", role.get("reagent", "--")),
        ("unexplained", role.get("unexplained", "--")),
        ("peaks explained", f"{100 * (1 - cf['unexplained']):.1f}%"
         if "unexplained" in cf else "--"),
        ("signal explained", f"{explained_sig:.1f}%" if sig else "--"),
        ("total runtime", _hms(elapsed) if elapsed is not None else "--"),
    ]
    return [(k, str(v)) for k, v in rows]


def stage_text(s: dict) -> str:
    """The label beside the stage bar, from a snapshot alone -- no Tk, so it is
    testable on a headless machine.

    A parallel run has no live stage stream (workers buffer their logs and the
    parent replays them after the reduce), so the slot says what IS knowable --
    how many workers are on it -- rather than animating a lie. A serial run
    names the running stage out of the learned count, and says nothing at all
    until the first stage of a sample lands."""
    n = int(s.get("parallel") or 0)
    if n:
        return f"{n} worker" + ("s" if n != 1 else "")
    if not s.get("stage_idx"):
        return "--"
    return f"{s.get('stage_name') or '--'} {s['stage_idx']}/{s.get('n_stages') or '?'}"


# --------------------------------------------------------------------------- #
# 3. the Tk window. Owns its OWN thread and its own `Tk()`; the run thread only
#    ever puts snapshots on a queue, which is the only thread-safe way to drive
#    Tk from elsewhere.
# --------------------------------------------------------------------------- #
def display_available() -> bool:
    """Is there a display we can SAFELY open a Tk window on?

    macOS: no. The window runs on a daemon thread (`TkWindow.start`), and on
    macOS Tk/Cocoa must be driven from the process's main thread -- off it, Tk
    aborts the whole process (a hard crash, not a Python exception any guard
    here could catch), i.e. the progress window would kill the run it reports
    on. macOS gets the terminal status line instead.
    Windows: assumed (no env var announces a display). Linux: X11 or Wayland."""
    if sys.platform == "darwin":
        return False
    if sys.platform.startswith("win"):
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


class TkWindow:
    """The window. `start()` returns False if Tk can't come up, which is the
    caller's cue to fall back -- it never raises into the run."""

    POLL_MS = 120

    def __init__(self, title: str):
        self.title = title
        self.q: queue.Queue = queue.Queue()
        self._ready = threading.Event()
        self._ok = False
        self._thread: threading.Thread | None = None
        self._last: dict = {}
        self.root = None
        self._stats_labels: dict = {}

    # -- run-thread side --------------------------------------------------- #
    def start(self, timeout: float = 5.0) -> bool:
        if not display_available():
            return False
        self._thread = threading.Thread(target=self._main, name="peaky-progress",
                                        daemon=True)
        self._thread.start()
        self._ready.wait(timeout)          # wait for Tk to actually come up
        return self._ok

    def push(self, snap: dict) -> None:
        self.q.put(("state", snap))

    def close(self, wait: bool, timeout: float | None = None) -> None:
        """Take the window down. `wait=True` keeps it up until the user closes it
        or `timeout` seconds pass (default `hold_seconds()`, i.e. env
        `PEAKY_PROGRESS_HOLD_S`) -- the point of a finished run's stats panel is
        that it can be READ, but never at the price of a process that cannot end
        on its own. Ctrl-C during the hold, or `wait=False`, closes it now.

        Always ends by joining the UI thread (bounded): the Tcl interpreter must
        be destroyed on ITS thread before the process exits (see `_teardown`),
        and a `quit` merely queued for the next 120 ms poll is a race with
        interpreter shutdown."""
        if not self._ok or not self._thread:
            return
        if wait:
            try:
                self._thread.join(hold_seconds() if timeout is None else timeout)
            except KeyboardInterrupt:
                pass
        self.q.put(("quit", None))
        try:
            self._thread.join(1.0)
        except KeyboardInterrupt:
            pass

    def alive(self) -> bool:
        return bool(self._ok and self._thread and self._thread.is_alive())

    # -- UI-thread side ---------------------------------------------------- #
    def _main(self) -> None:
        try:
            import tkinter as tk
            from tkinter import ttk
        except Exception:                       # no tkinter in this interpreter
            self._ready.set()
            return
        try:
            self.root = tk.Tk()
            self._build(tk, ttk)
            self._ok = True
        except Exception:                       # no display, broken theme, ...
            self._ready.set()
            return
        finally:
            self._ready.set()
        try:
            self.root.after(self.POLL_MS, self._drain)
            self.root.mainloop()
        except Exception:
            pass
        finally:
            self._teardown()

    def _teardown(self) -> None:
        """Destroy the Tcl interpreter and drop EVERY reference to it -- from this
        thread, the one that created it. If a Tk object (root, a widget, or a
        StringVar) outlives this thread, the main thread finalizes it at
        interpreter shutdown and Tcl aborts the process with
        'Tcl_AsyncDelete: async handler deleted by the wrong thread' -- i.e. the
        progress window would kill the run it was reporting on, at the very end."""
        try:
            if self.root is not None:
                self.root.destroy()
        except Exception:
            pass
        self.root = None
        self._stats_labels.clear()
        for attr in ("pb_sample", "pb_stage", "v_sample", "v_stage", "v_head",
                     "v_phase", "v_clock", "v_tail", "btn", "stats_frame"):
            setattr(self, attr, None)

    def _build(self, tk, ttk) -> None:
        r = self.root
        r.title(self.title)
        r.minsize(520, 300)
        r.protocol("WM_DELETE_WINDOW", self._quit)
        pad = {"padx": 12, "pady": 3}
        mono = ("TkFixedFont", 10)

        self.v_head = tk.StringVar(value=self.title)
        tk.Label(r, textvariable=self.v_head, anchor="w",
                 font=("TkDefaultFont", 11, "bold")).pack(fill="x", padx=12, pady=(12, 2))
        self.v_phase = tk.StringVar(value="starting ...")
        tk.Label(r, textvariable=self.v_phase, anchor="w",
                 fg="#555").pack(fill="x", **pad)

        def bar(label):
            row = tk.Frame(r); row.pack(fill="x", **pad)
            tk.Label(row, text=label, width=8, anchor="w").pack(side="left")
            pb = ttk.Progressbar(row, mode="determinate", maximum=1000, length=260)
            pb.pack(side="left", fill="x", expand=True)
            var = tk.StringVar(value="--")
            tk.Label(row, textvariable=var, width=18, anchor="w",
                     font=mono).pack(side="left", padx=(8, 0))
            return pb, var

        self.pb_sample, self.v_sample = bar("samples")
        self.pb_stage, self.v_stage = bar("stage")

        self.v_clock = tk.StringVar(value="elapsed --:--")
        tk.Label(r, textvariable=self.v_clock, anchor="w",
                 font=mono).pack(fill="x", **pad)

        ttk.Separator(r, orient="horizontal").pack(fill="x", padx=12, pady=8)
        self.stats_frame = tk.Frame(r)
        self.stats_frame.pack(fill="x", padx=12)
        self._render_stats(summary_rows(None), tk)

        ttk.Separator(r, orient="horizontal").pack(fill="x", padx=12, pady=8)
        self.v_tail = tk.StringVar(value="")
        tk.Label(r, textvariable=self.v_tail, anchor="w", fg="#777",
                 font=("TkFixedFont", 8), wraplength=560,
                 justify="left").pack(fill="x", padx=12)

        self.btn = ttk.Button(r, text="Close", command=self._quit)
        self.btn.pack(anchor="e", padx=12, pady=10)
        self.btn.state(["disabled"])

    def _render_stats(self, rows, tk) -> None:
        """Two-column label/value grid. Updates values in place; REBUILDS when the
        label set itself changes -- `peaky assign` and `peaky batch` report
        different quantities, and leaving the placeholder set behind would show
        two half-filled tables at once."""
        keys = [k for k, _ in rows]
        if keys != getattr(self, "_stats_keys", None):
            for w in self.stats_frame.winfo_children():
                w.destroy()
            self._stats_labels = {}
            self._stats_keys = keys
            for i, (k, _) in enumerate(rows):
                tk.Label(self.stats_frame, text=k, anchor="w", fg="#555").grid(
                    row=i, column=0, sticky="w", pady=1)
                var = tk.StringVar()
                tk.Label(self.stats_frame, textvariable=var, anchor="w",
                         font=("TkFixedFont", 10)).grid(row=i, column=1,
                                                        sticky="w", padx=(16, 0))
                self._stats_labels[k] = var
            self.stats_frame.grid_columnconfigure(1, weight=1)
        for k, v in rows:
            self._stats_labels[k].set(v)

    def _drain(self) -> None:
        """Poll the queue AND re-tick the clock, so elapsed/ETA keep moving during
        a long silent stage. Guarded: a UI error must not take down the run."""
        try:
            while True:
                kind, payload = self.q.get_nowait()
                if kind == "quit":
                    self._quit()
                    return
                self._last = payload
        except queue.Empty:
            pass
        if self.root is None:
            return                      # torn down; nothing left to draw on
        try:
            if self._last:
                # RE-TICKED, not replayed: `self._last` is frozen at the last log
                # line, so re-applying it verbatim repaints identical numbers and
                # the clock stands still through every silent stretch.
                self._apply(retick(self._last))
        except Exception:
            pass
        try:
            self.root.after(self.POLL_MS, self._drain)
        except Exception:
            pass

    def _apply(self, s: dict) -> None:
        import tkinter as tk

        self.v_head.set(s["title"])
        done, n = s["samples_done"], s["n_samples"]
        self.pb_sample["value"] = 1000 * s["sample_frac"]
        self.v_sample.set(f"{done}/{n}" if n else "--")

        # stage_frac is already 0 in parallel mode (no live stage stream)
        self.pb_stage["value"] = 1000 * s["stage_frac"]
        self.v_stage.set(stage_text(s))

        sid = f" · {s['current_sid']}" if s["current_sid"] else ""
        self.v_phase.set(("error: " + s["error"]) if s["error"]
                         else f"{s['phase_label']}{sid}")
        eta = s["eta"]
        elapsed = f"elapsed {_hms(s['elapsed'])}"
        self.v_clock.set(elapsed if s["finished"] or eta is None
                         else f"{elapsed}   eta ~{_hms(eta)}")
        self.v_tail.set(s["out_dir"] or s["last_line"])

        # LIVE, not only at the end: the per-file totals fill in as files finish
        # (the label set is stable through a run, so this is a value update, not
        # a rebuild); finish() then swaps in the exact summary.
        self._render_stats(panel_rows(s), tk)
        if s["finished"]:
            self.v_phase.set(("FAILED: " + s["error"]) if s["error"] else "done")
            try:
                self.btn.state(["!disabled"])
                self.btn.focus_set()
            except Exception:
                pass

    def _quit(self) -> None:
        """Break the mainloop. The actual destroy happens in `_teardown` once
        `mainloop()` returns, so it always runs on the UI thread."""
        try:
            if self.root is not None:
                self.root.quit()
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# 4. terminal fallback -- no display, no tkinter, or Tk refused to start.
#    Deliberately NOT a redrawing bar: peaky's own log is streaming to the same
#    terminal, so a `\r` bar would fight it. One compact line per sample instead.
# --------------------------------------------------------------------------- #
class TerminalStatus:
    # No start(): `open_progress` only ever starts a TkWindow, and reaches for
    # this one exactly when that failed -- there is nothing here to bring up.

    def push(self, s: dict) -> None:
        key = (s["samples_done"], s["n_samples"], s["phase"])
        if key == getattr(self, "_key", None) or not s["n_samples"]:
            return
        self._key = key
        eta = f" · eta ~{_hms(s['eta'])}" if s["eta"] else ""
        print(f"[progress] {s['samples_done']}/{s['n_samples']} samples · "
              f"{s['phase_label']} · elapsed {_hms(s['elapsed'])}{eta}", flush=True)

    def close(self, wait: bool) -> None:
        pass

    def alive(self) -> bool:
        return False        # nothing to close -> the CLI's hold note stays quiet


# --------------------------------------------------------------------------- #
# 5. the Reporter -- a drop-in for `print`, which is what makes it `log=`-able
#    everywhere in the pipeline without any of it knowing.
# --------------------------------------------------------------------------- #
class Reporter:
    """Callable: `reporter(line)` logs the line AND advances the window.

    Use as a context manager so the window is always closed, including on an
    exception -- which it then displays rather than vanishing on."""

    def __init__(self, title: str, *, log=print, ui=None, hold: bool = True,
                 n_samples: int = 0):
        self.log = log
        self.state = ProgressState(title=title, n_samples=n_samples)
        self.ui = ui
        self.hold = hold

    # -- log interface (this is the whole coupling to the pipeline) ---------- #
    def __call__(self, *args, **kw) -> None:
        line = " ".join(str(a) for a in args)
        try:
            self.log(*args, **kw)                # the run's own output, untouched
        except Exception:
            pass
        try:
            self.state.feed(line)
            self._push()
        except Exception:
            pass                                 # UI is never allowed to be fatal

    def _push(self) -> None:
        if self.ui:
            try:
                self.ui.push(self.state.snapshot())
            except Exception:
                pass

    # -- lifecycle ---------------------------------------------------------- #
    def phase(self, name: str) -> None:
        """Set the phase directly (for callers outside the logged pipeline).
        Guarded like __call__: a progress call can never raise into the run."""
        try:
            self.state.phase = str(name)
            self._push()
        except Exception:
            pass

    def sample_done(self, count: int = 1) -> None:
        """Mark samples complete from OUTSIDE the log stream, for a caller that
        assigns a sample itself instead of going through `assign_batch` (the
        only emitter of the `(i/N) done` line). Guarded like `phase()`: a
        progress call can never raise into the run."""
        try:
            self.state.mark_sample_done(count)
            self._push()
        except Exception:
            pass

    def finish(self, summary: dict | None = None, error: str = "") -> None:
        """Freeze the window on the run's real numbers. `summary` is the dict the
        pipeline RETURNS (batch_summary), so nothing here depends on parsing.
        Anything that is not a dict is shown as no stats rather than raised."""
        try:
            self.state.stats = dict(summary) if isinstance(summary, dict) else {}
            self.state.error = str(error or "")
            self.state.phase = "done"
            self.state.finished = True
            self._push()
        except Exception:
            pass

    def close(self, wait: bool | None = None) -> None:
        """Close the UI. `wait` defaults to "hold, if the run finished" -- the
        hold exists so a finished run's numbers can be read, never for an
        unfinished one. Guarded like every other UI call site: teardown runs
        from __exit__, and an exception here would REPLACE a real exception
        from the run itself."""
        if self.ui:
            try:
                if wait is None:
                    wait = self.hold and self.state.finished
                self.ui.close(wait=bool(wait))
            except Exception:
                pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is not None and issubclass(exc_type, (KeyboardInterrupt, SystemExit)):
            # The user (or the process) wants OUT: no stats panel, no hold --
            # a Ctrl-C that then waits on a window is a hang with extra steps.
            self.close(wait=False)
            return False
        if exc_type is not None and not self.state.finished:
            self.finish(self.state.stats, error=f"{exc_type.__name__}: {exc}")
        self.close()
        return False


def _interactive() -> bool:
    """A person at a terminal? Only then is anyone there to read a window."""
    try:
        return bool(sys.stdin.isatty() and sys.stdout.isatty())
    except Exception:           # stdin/stdout replaced or closed (pythonw, pipes)
        return False


_TRUTHY = {"1", "true", "yes", "on"}


def enabled(flag: bool | None = None) -> bool:
    """Is the window on for this run?

    `--progress` / `--no-progress` wins; else `PEAKY_PROGRESS` (truthy is on,
    anything else set -- `0` included -- is an explicit off); else the DEFAULT:
    on for a person at a terminal, off for everything else.

    Interactive-only by design. A pipe, a CI job, an MCP or skill-driven run has
    nobody to read a window, and the terminal fallback would write `[progress]`
    lines into output somebody is capturing -- so "on by default" must never
    mean "on in a script". `_interactive()` already draws exactly that line for
    the hold, and this is the same question one step earlier.
    """
    if flag is not None:
        return bool(flag)
    env = os.environ.get("PEAKY_PROGRESS", "").strip().lower()
    if env:
        return env in _TRUTHY
    return _interactive()


def explicitly_requested(flag: bool | None = None) -> bool:
    """Did someone ASK for the window, or did the default hand it to them?

    Only an explicit request earns the "no usable display" note: unasked-for, it
    is a complaint about something nobody requested, printed on every run."""
    if flag is not None:
        return bool(flag)
    return os.environ.get("PEAKY_PROGRESS", "").strip().lower() in _TRUTHY


HOLD_S_DEFAULT = 600.0


def hold_wanted(flag: bool | None, *, window_up: bool, hold: bool = True) -> bool:
    """Should the finished window be held open to be read? The decision table,
    in order:

    - the call site said no (`hold=False`), or `PEAKY_PROGRESS_HOLD_S` is 0:
      never, whatever else is true;
    - a person at a terminal (`_interactive()`): yes -- the case the hold was
      built for;
    - the window was ASKED for (`--progress`, or `PEAKY_PROGRESS` truthy) AND a
      Tk window actually came up: yes, tty or not. A `setsid nohup peaky batch
      ... --progress` run with DISPLAY set opens a real window on the desktop;
      with the hold gated on a tty alone that window closed the instant the run
      finished, so its stats panel -- the thing it was opened for -- was never
      seen at all;
    - anything else: no. That is the implicit default off a terminal and the
      terminal fallback (which has nothing to hold), so a script, a CI job or a
      skill-driven run never waits on a window it did not ask for or could not
      show.

    Bounded either way: the hold ends at `hold_seconds()`, on Close, or on
    Ctrl-C (`TkWindow.close`)."""
    if not hold or hold_seconds() <= 0:
        return False
    if _interactive():
        return True
    return bool(window_up and explicitly_requested(flag))


def hold_seconds() -> float:
    """How long a finished run keeps its window up to be read: env
    `PEAKY_PROGRESS_HOLD_S` in seconds (default 600; 0 = no hold at all).
    Unparseable -> the default, so a typo can neither hang nor skip the hold."""
    raw = os.environ.get("PEAKY_PROGRESS_HOLD_S", "").strip()
    if not raw:
        return HOLD_S_DEFAULT
    try:
        return max(0.0, float(raw))
    except ValueError:
        return HOLD_S_DEFAULT


def open_progress(title: str, *, flag: bool | None = None, log=print,
                  hold: bool = True, n_samples: int = 0) -> Reporter:
    """The one entry point the CLI uses. Returns a `log`-compatible Reporter --
    disabled, it is a transparent pass-through to `log`, so call sites need no
    branch of their own.

    `hold` (keep the finished window up to be read) is decided by
    `hold_wanted`: on an interactive terminal, or wherever the window was asked
    for explicitly and really came up, and only while `PEAKY_PROGRESS_HOLD_S`
    > 0. The terminal fallback never waits, so a pipe, a CI job or a
    skill-driven run cannot hang on a window it cannot show."""
    if not enabled(flag):
        return Reporter(title, log=log, ui=None, hold=False, n_samples=n_samples)
    ui = TkWindow(title)
    if not ui.start():
        # Asked for and not delivered -> say why. Defaulted on, stay quiet: the
        # one-line status speaks for itself, and a warning about a window nobody
        # requested would print on every run on a headless box.
        if explicitly_requested(flag):
            print("[progress] no usable display for a window; "
                  "falling back to terminal status", flush=True)
        ui = TerminalStatus()
    hold = hold_wanted(flag, window_up=ui.alive(), hold=hold)
    rep = Reporter(title, log=log, ui=ui, hold=hold, n_samples=n_samples)
    rep._push()
    return rep
