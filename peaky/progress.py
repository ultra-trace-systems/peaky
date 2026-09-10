"""Live progress window for a run — opt-in via `--progress` (or `PEAKY_PROGRESS=1`).

Peaky already threads a `log=print` callable through every level of a run
(`pipeline.run_batch` -> `assign_batch.run` -> `assign.run` -> each of its ~36
stages), and the lines that callable receives already carry the run's shape:
`(i/N) assigning <sid>`, `(i/N) done <sid>`, `[run] <stage> took Xs`. So this
module is nothing more than a **`log` wrapper**: it forwards every line untouched
to the real log and, on the side, reads those lines into a progress model that
drives a small Tk window.

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
# importing `assign` just to count `_STAGES`.
NOMINAL_STAGES = 36

# Phase order for the header. `assign` is the long pole; the rest are the tail.
PHASES = ("fetch", "select", "assign", "merge", "cluster", "vankrevelen",
          "report", "provenance", "done")
PHASE_LABEL = {
    "fetch": "fetching time series", "select": "selecting samples",
    "assign": "assigning", "merge": "merging ledgers", "cluster": "clustering",
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
RE_PARALLEL = re.compile(r"^\[assign_batch\] parallel: (\d+) worker processes")
RE_ASSIGN_DONE = re.compile(r"^\[assign_batch\] DONE: (\d+) merged M0")
RE_STAGE = re.compile(r"^\[run\] (\S+) took ([\d.]+)s")
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
    out_dir: str = ""
    last_line: str = ""
    stats: dict = field(default_factory=dict)
    finished: bool = False
    error: str = ""

    # -- derived ----------------------------------------------------------- #
    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.t0

    @property
    def eta(self) -> float | None:
        """Linear extrapolation over COMPLETED samples. None until one lands --
        an ETA from zero completions is a guess dressed as a number."""
        if self.finished or not self.samples_done or not self.n_samples:
            return None
        if self.samples_done >= self.n_samples:
            return None            # samples all in; the report tail is not modelled
        per = self.elapsed / self.samples_done
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

    def feed(self, line: str) -> bool:
        """Read one log line into the model. Returns True if anything changed.
        Unrecognised lines only refresh `last_line` (the window's activity tail)."""
        line = str(line).strip()
        if not line:
            return False
        self.last_line = line[:160]

        if (m := RE_ASSIGNING.match(line)):
            self.phase = "assign"
            self.n_samples = int(m.group(2))
            self.current_sid = m.group(3)
            self.stage_idx, self.stage_name = 0, ""   # new sample -> restart stage bar
            return True
        if (m := RE_SAMPLE_DONE.match(line)):
            self.phase = "assign"
            self.samples_done, self.n_samples = int(m.group(1)), int(m.group(2))
            self.current_sid = m.group(3)
            if not self.parallel and self.stage_idx:
                self.n_stages = self.stage_idx          # learn the real stage count
            self.stage_idx, self.stage_name = 0, ""
            return True
        if (m := RE_PARALLEL.match(line)):
            # Workers replay their logs only after the reduce, so stage lines stop
            # being live here. Freeze the stage bar rather than animate a lie.
            self.parallel = int(m.group(1))
            self.phase = "assign"
            return True
        if RE_ASSIGN_DONE.match(line):
            self.samples_done = self.n_samples or self.samples_done
            self.phase = "merge"
            self.current_sid = ""          # no longer inside any one sample
            return True
        if (m := RE_STAGE.match(line)) and not self.parallel:
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
                "finished": self.finished, "error": self.error}


# --------------------------------------------------------------------------- #
# 2. summary -> the stats panel. Reads the dict `assign_batch.run` returns, so
#    the final numbers are EXACT (never parsed out of the log).
# --------------------------------------------------------------------------- #
def summary_rows(summary: dict | None, elapsed: float | None = None) -> list:
    """[(label, value)] for the stats panel. Handles BOTH shapes a run returns:
    a batch summary (`assign_batch.run`) and a single sample's ledger stats
    (`ledger.stats`, keyed by role) -- `peaky assign` and `peaky batch` count
    different things, and showing a column of '--' for the other one is worse
    than showing the numbers that sample actually has."""
    s = summary or {}
    if "by_role" in s:
        return _single_sample_rows(s, elapsed)
    tiers = s.get("merged_tiers") or {}
    tier_txt = "  ".join(f"{k} {v}" for k, v in tiers.items()) or "--"
    rows = [
        ("merged M0", s.get("merged_M0", "--")),
        ("tiers", tier_txt),
        ("samples", s.get("n_files", "--")),
        ("in all files", s.get("n_in_all_files", "--")),
        ("single-file", s.get("n_single_file", "--")),
        ("formula disagreements", s.get("formula_disagreements", "--")),
    ]
    # TWO different durations, both worth seeing: the assignment itself (measured
    # by assign_batch, in the summary) and the whole command (measured by the
    # window). The gap between them is the fetch + cluster + VK + report tail.
    assign_s = s.get("elapsed_s")
    rows.append(("assignment", _hms(assign_s) if assign_s is not None else "--"))
    if elapsed is not None:
        rows.append(("total runtime", _hms(elapsed)))
    return [(k, str(v)) for k, v in rows]


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


# --------------------------------------------------------------------------- #
# 3. the Tk window. Owns its OWN thread and its own `Tk()`; the run thread only
#    ever puts snapshots on a queue, which is the only thread-safe way to drive
#    Tk from elsewhere.
# --------------------------------------------------------------------------- #
def display_available() -> bool:
    """Is there a display to open a window on? X11/Wayland on Linux; assumed on
    macOS/Windows (no env var announces it there)."""
    if sys.platform.startswith(("win", "darwin")):
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
        self._widgets: list = []      # every Tk object we hold, for _teardown
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

    def close(self, wait: bool) -> None:
        """`wait=True` keeps the process alive until the user closes the window --
        the point of a finished run's stats panel is that it can be READ. Ctrl-C
        during the wait just stops waiting; the run is already complete."""
        if not self._ok or not self._thread:
            return
        if not wait:
            self.q.put(("quit", None))
        try:
            self._thread.join()
        except KeyboardInterrupt:
            self.q.put(("quit", None))

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
        self._widgets.clear()
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

        self._widgets = [r]           # everything Tk-owned, dropped in _teardown
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
                self._apply(self._last)
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

        if s["parallel"]:
            # honest about what is knowable: no live stage stream in parallel mode
            self.pb_stage["value"] = 0
            self.v_stage.set(f"{s['parallel']} workers")
        else:
            self.pb_stage["value"] = 1000 * s["stage_frac"]
            self.v_stage.set(f"{s['stage_name'] or '--'} {s['stage_idx']}/{s['n_stages']}"
                             if s["stage_idx"] else "--")

        sid = f" · {s['current_sid']}" if s["current_sid"] else ""
        self.v_phase.set(("error: " + s["error"]) if s["error"]
                         else f"{s['phase_label']}{sid}")
        eta = s["eta"]
        elapsed = f"elapsed {_hms(s['elapsed'])}"
        self.v_clock.set(elapsed if s["finished"] or eta is None
                         else f"{elapsed}   eta ~{_hms(eta)}")
        self.v_tail.set(s["out_dir"] or s["last_line"])

        if s["finished"]:
            self._render_stats(summary_rows(s["stats"], s["elapsed"]), tk)
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
    def start(self, timeout: float = 0.0) -> bool:
        return True

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
        """Set the phase directly (for callers outside the logged pipeline)."""
        self.state.phase = name
        self._push()

    def finish(self, summary: dict | None = None, error: str = "") -> None:
        """Freeze the window on the run's real numbers. `summary` is the dict the
        pipeline RETURNS (batch_summary), so nothing here depends on parsing."""
        self.state.stats = dict(summary or {})
        self.state.error = error
        self.state.phase = "done"
        self.state.finished = True
        self._push()

    def close(self) -> None:
        # Guarded like every other UI call site: teardown runs from __exit__, and
        # an exception here would REPLACE a real exception from the run itself.
        if self.ui:
            try:
                self.ui.close(wait=self.hold and self.state.finished)
            except Exception:
                pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is not None and not self.state.finished:
            self.finish(self.state.stats, error=f"{exc_type.__name__}: {exc}")
        self.close()
        return False


def enabled(flag: bool | None = None) -> bool:
    """`--progress` wins; else `PEAKY_PROGRESS` (1/true/yes/on)."""
    if flag:
        return True
    return os.environ.get("PEAKY_PROGRESS", "").strip().lower() in {"1", "true", "yes", "on"}


def open_progress(title: str, *, flag: bool | None = None, log=print,
                  hold: bool = True, n_samples: int = 0) -> Reporter:
    """The one entry point the CLI uses. Returns a `log`-compatible Reporter --
    disabled, it is a transparent pass-through to `log`, so call sites need no
    branch of their own."""
    if not enabled(flag):
        return Reporter(title, log=log, ui=None, hold=False, n_samples=n_samples)
    ui = TkWindow(title)
    if not ui.start():
        print("[progress] no usable display for a window; "
              "falling back to terminal status", flush=True)
        ui = TerminalStatus()
    rep = Reporter(title, log=log, ui=ui, hold=hold, n_samples=n_samples)
    rep._push()
    return rep
