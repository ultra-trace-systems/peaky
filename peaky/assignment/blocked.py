"""Why an unexplained peak is unexplained: the best blocked reading and its gate.

Every exit between enumeration and commit used to drop its candidate without a
trace. The context filter discarded its reason, `commit_winners` skipped a
sub-threshold winner together with its arbitration frame, the labelled rescue's
gates and the pass-0 ppm gate only logged, and a peak the grid box could not
reach had no candidate at all. A bright line could therefore sit in a ledger as
a bare 'unexplained' row, and only an offline replay of every gate said why.

This module keeps those exits. `assign.run` starts one `Recorder` per run (a
C42 or isotope-response re-run starts a fresh one) and the exits `note` what
they stopped: the reading (neutral, adduct, ion formula, score, ppm) and the
gate with a short reason, keyed by peak. Nothing here is read back by a pass:
the record changes no assignment, no tier, no level and no existing column.
Outside a run (a pass called on its own, a test) no recorder is active and
every `note` is a no-op.

At the end of the run `stamp` writes two columns on every `unexplained` row:

  blocked_reading  ``<neutral> <adduct> (<ion>), score <s>, ppm <+p>``: the best
                   reading a gate stopped on the peak (``score n/a`` when no
                   scorer read it on the peak);
  blocked_by       the gate that stopped it, with a short reason, e.g.
                   ``context:O/C 1.60>1.50``, ``score<min_raw 0.62<0.70 (pass 1)``,
                   ``labeled:sibling``, ``pass0:ppm 3.0>2.0``,
                   ``cleared:mass-gate``, ``no candidate in box (C0-16 ...)``,
                   ``not admitted (below the height gate, not persistent)``.

Both are NA on every other row. The best reading is the one the scorer read
highest, at or above the suspect floor (`tau_suspect`); below it, a released
isotope line's note comes first, then a reading no scorer read (nearest the
run's mass centre first), then the weaker scored readings. A reading stopped
more than once keeps its first exit (pass 0's mass gate, not the later passes
that met it again), unless it got through in between and was cleared: then
the clear. A context-blocked reading is never scored by the
pass that blocked it; `stamp` scores the SCORE_TOP_K nearest of each
unexplained peak in one call to the run's own local scorer, so they compete on
score like the rest (under PEAKY_LOCAL_SCORING=0 they stay unscored: the
server's matcher is never called for a diagnostic).

After the run the ledger API keeps the two columns honest where a ledger
carries them: `ledger.clear_assignment` writes the cleared reading onto the row
(the batch's per-file rewrite), and a commit, an isotope attach or a reagent /
artifact mark empties them.

`batch_table` pools a batch's per-file ledgers into `tables/unexplained_blocked.csv`:
one row per m/z cluster of unexplained per-file rows, with the file count, the
heights, the most frequent blocked reading and the gate counts.
"""
from __future__ import annotations

import bisect
import functools
import re
import threading
from dataclasses import dataclass

import numpy as np
import pandas as pd

from peaky.chem import chemistry as C

__version__ = "0.1.0"

READING = "blocked_reading"
GATE = "blocked_by"
COLUMNS = (READING, GATE)

#: below this score a scored reading is weaker evidence than a released
#: isotope line's note or an unscored reading (the commit floor's 'Reject' line;
#: the run's `tau_suspect` when the stamp has the cfg)
SUSPECT_FLOOR = 0.50
#: the unscored readings of one peak `stamp` scores (nearest the centre first)
SCORE_TOP_K = 5
#: the longest free-text reason kept in `blocked_by`
REASON_MAX = 120

NOT_ADMITTED = "not admitted (below the height gate, not persistent)"
NO_SCORED = "no candidate scored on the peak (pass 1)"
NO_RECORD = "no recorded exit (not a pass-1 target)"

_UNEXPLAINED = "unexplained"
#: the exits that replace a reading's earlier one: it was committed, then cleared
_REPLACING = ("cleared:", "released:")


@dataclass
class _Block:
    gate: str
    neutral: str | None = None
    adduct: str | None = None
    ion: str | None = None
    score: float | None = None
    ppm: float | None = None
    seq: int = 0
    to_score: bool = False


class Recorder:
    """The exits of one run: {peak_id: {reading key: _Block}}, the pass-1
    targets with grid candidates and those without (`box`, the box text)."""

    def __init__(self) -> None:
        self.blocks: dict = {}
        self.targets: set = set()
        self.box: dict = {}
        self._seq = 0
        self.errors = 0               # notes that raised (never the run's problem)
        self.first_error: str | None = None

    def add(self, peak_id, gate: str, **kw) -> None:
        """Note one exit. A reading's FIRST exit stands (pass 0's mass gate,
        not the later passes that met the same reading again); a later exit
        only fills a score, ion or ppm the first lacked -- unless the reading
        got through in between and was cleared: the clear replaces it."""
        self._seq += 1
        b = _Block(gate=gate, seq=self._seq, **kw)
        key = (b.neutral, b.adduct or "") if b.neutral else ("", gate)
        mine = self.blocks.setdefault(peak_id, {})
        old = mine.get(key)
        if old is not None and not gate.startswith(_REPLACING):
            for f in ("ion", "score", "ppm"):
                if getattr(old, f) is None and getattr(b, f) is not None:
                    setattr(old, f, getattr(b, f))
            return
        mine[key] = b


#: the recorder of the run on THIS thread (a run is one thread's work; two runs on
#: two threads -- the MCP server's jobs -- never note into, or stop, each other's)
_STATE = threading.local()


def _rec() -> Recorder | None:
    return getattr(_STATE, "rec", None)


def start() -> Recorder:
    """A fresh recorder for this run (replacing any earlier one)."""
    _STATE.rec = Recorder()
    return _STATE.rec


def stop() -> None:
    _STATE.rec = None


def active() -> bool:
    return _rec() is not None


def _guarded(fn):
    """An exit's note never changes the run: whatever it raises is counted on
    the recorder (and logged by the stamp), never passed to the pass."""
    @functools.wraps(fn)
    def wrapper(*a, **kw):
        try:
            return fn(*a, **kw)
        except Exception as e:      # noqa: BLE001 -- a diagnostic must not fail a pass
            rec = _rec()
            if rec is not None:
                rec.errors += 1
                if rec.first_error is None:
                    rec.first_error = f"{fn.__name__}: {type(e).__name__}: {e}"
            return None
    return wrapper


# ---------------------------------------------------------------------------
# small text helpers
# ---------------------------------------------------------------------------
def _txt(v) -> str | None:
    try:
        if v is None or pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    s = str(v).strip()
    return s or None


def _num(v) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


def short(text, n: int = REASON_MAX) -> str:
    """One line, at most `n` characters ('...' marks a cut)."""
    t = " ".join(str(text or "").split())
    return t if len(t) <= n else t[: n - 3].rstrip() + "..."


_VK = re.compile(r"^(?P<name>.+?)=(?P<val>-?[\d.]+) out of \((?P<lo>[^,]+), (?P<hi>[^)]+)\)$")
_CAP = re.compile(r"^(?P<el>\^?[A-Z][a-z]?)=(?P<n>\d+) > (?P<cap>\d+)$")
_MIN_C = re.compile(r"^(?P<el>\^?[A-Z][a-z]?) in neutral needs C>=(?P<min>\d+) \(got C=(?P<c>\d+)\)")


def compact_reason(why) -> str:
    """The context filter's reason in a few characters: 'O/C=1.60 out of
    (0.0, 1.5)' -> 'O/C 1.60>1.50', 'N=3 > 2' -> 'N 3>2'; anything else as
    it is (one line, `short`)."""
    t = " ".join(str(why or "").split())
    m = _VK.match(t)
    if m:
        val = float(m["val"])
        lo, hi = _num(m["lo"]), _num(m["hi"])
        if hi is not None and val > hi:
            return f"{m['name']} {val:.2f}>{hi:.2f}"
        if lo is not None and val < lo:
            return f"{m['name']} {val:.2f}<{lo:.2f}"
        return f"{m['name']} {val:.2f}"
    m = _CAP.match(t)
    if m:
        return f"{m['el']} {m['n']}>{m['cap']}"
    m = _MIN_C.match(t)
    if m:
        return f"{m['el']} on C {m['c']}<{m['min']}"
    return short(t) or "rejected"


#: where a gate text's detail starts: a parenthesis or a ': ' explanation
_DETAIL = re.compile(r" \(|: ")
#: a number or a comparison of two ending a word, alone or after '=', '<', '>'
#: ('0.62<0.70', 'z=2.6>2.0', '|ppm|=1.20') -- never inside a word ('O<3x15N')
_NUMBERS = re.compile(r"(?:(?<=[=<>])|^)[-+]?\d+(?:\.\d+)?(?:[<>][-+]?\d+(?:\.\d+)?)?$")


def gate_class(text) -> str:
    """A `blocked_by` text without its numbers, formulas and detail -- what a
    count groups by: 'score<min_raw 0.62<0.70 (pass 1)' -> 'score<min_raw',
    'mass-gate z=2.6>2.0 without pattern (pass 1)' -> 'mass-gate z without
    pattern', 'cleared:mass-gate: z=2.2 ...' -> 'cleared:mass-gate',
    'cleared:displaced by certified-neutral C19H37ClN4O3' -> 'cleared:displaced
    by certified-neutral'."""
    t = _txt(text)
    if t is None:
        return "(none)"
    m = _DETAIL.search(t)
    keep = []
    for k, w in enumerate((t[: m.start()] if m else t).split(" ")):
        w = _NUMBERS.sub("", w).rstrip("=<>")
        # a later word still carrying a digit is a formula or a count: dropped
        if w and not (k and any(ch.isdigit() for ch in w)):
            keep.append(w)
    return " ".join(keep).rstrip(" =<>:+-") or "(none)"


def ion_formula(neutral, adduct) -> str | None:
    """The ion a (neutral, adduct) reading names, signed ('C5H7O5-'); None when
    the adduct does not parse."""
    if not neutral or not adduct:
        return None
    from peaky.assignment import tiers as _T     # lazy: tiers imports the ledger, which imports this
    try:
        cnt = _T._ion_counts(neutral, adduct, labelled=True)
    except Exception:           # noqa: BLE001 -- an odd adduct string has no ion text
        return None
    if not cnt or any(v < 0 for v in cnt.values()):
        return None
    sign = "+" if "]+" in adduct else "-" if "]-" in adduct else ""
    return C.format_formula(cnt) + sign


def reading_text(b: _Block) -> str | None:
    """'<neutral> <adduct> (<ion>), score <s>, ppm <+p>' (None for a gate-only note)."""
    if not b.neutral:
        return None
    s = f"{b.neutral} {b.adduct}" if b.adduct else b.neutral
    ion = b.ion or ion_formula(b.neutral, b.adduct)
    if ion:
        s += f" ({ion})"
    s += f", score {b.score:.2f}" if b.score is not None else ", score n/a"
    if b.ppm is not None:
        s += f", ppm {b.ppm:+.2f}"
    return s


def reading_key(text) -> str:
    """A `blocked_reading` without its numbers: '<neutral> <adduct> (<ion>)'."""
    t = _txt(text)
    return "" if t is None else t.split(", score")[0]


_SCORE = re.compile(r", score ([\d.]+)")


def reading_score(text) -> float | None:
    m = _SCORE.search(str(text or ""))
    return float(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# the exits
# ---------------------------------------------------------------------------
@_guarded
def note(peak_id, gate: str, *, neutral=None, adduct=None, ion=None, score=None, ppm=None,
         to_score: bool = False) -> None:
    """Record that `gate` stopped a reading (or, without `neutral`, the peak)."""
    rec = _rec()
    if rec is None or peak_id is None:
        return
    rec.add(peak_id, str(gate), neutral=_txt(neutral), adduct=_txt(adduct), ion=_txt(ion),
                score=_num(score), ppm=_num(ppm), to_score=bool(to_score))


@_guarded
def note_at(peak_id, gate: str, mz, *, neutral, adduct, **kw) -> None:
    """`note` with the ppm of the reading's ion against the peak's m/z `mz`."""
    if _rec() is None:
        return
    ion = C.ion_mz(neutral, adduct)
    note(peak_id, gate, neutral=neutral, adduct=adduct, ppm=(float(mz) - ion) / ion * 1e6, **kw)


@_guarded
def note_winner(w, gate: str) -> None:
    """An arbitration winner (`core.arbitrate`'s row) a commit gate refused."""
    if _rec() is None:
        return
    g = w.get if hasattr(w, "get") else (lambda k, d=None: d)
    note(g("peak_id"), gate, neutral=g("neutral"), adduct=g("adduct"), ion=g("ion_formula"),
         score=g("raw_score"), ppm=g("ppm_error"))


def attribute(targets: pd.DataFrame, formulas, adducts, tol_ppm: float) -> list[tuple]:
    """Which target peaks each formula reaches: [(peak_id, formula, adduct, ppm)]
    for every adduct whose ion lies within `tol_ppm` of a target's m/z (the
    window the grid enumerated with). Arithmetic on formulas a pass already
    enumerated -- no enumeration of its own."""
    if targets is None or not len(targets) or not formulas:
        return []
    mz = pd.to_numeric(targets["mz"], errors="coerce").to_numpy(dtype=float)
    pid = targets["peak_id"].to_numpy()
    ok = np.isfinite(mz)
    order = np.argsort(mz[ok], kind="mergesort")
    ms, ps = mz[ok][order].tolist(), pid[ok][order]
    shifts = [(a, C.ADDUCT_SHIFTS[a]) for a in dict.fromkeys(adducts or ()) if a in C.ADDUCT_SHIFTS]
    out = []
    for f in sorted(formulas):
        m = C.neutral_mass(f)
        for a, s in shifts:
            ion = m + s
            if ion <= 0:
                continue
            tol = ion * float(tol_ppm) * 1e-6
            for j in range(bisect.bisect_left(ms, ion - tol), bisect.bisect_right(ms, ion + tol)):
                out.append((ps[j], f, a, (ms[j] - ion) / ion * 1e6))
    return out


@_guarded
def note_context(targets: pd.DataFrame, rejected: dict, adducts, tol_ppm: float, *,
                 prefix: str = "context:") -> None:
    """The readings a context filter rejected ({formula: its reason}), each on
    the target peaks it reaches; scored later, at the stamp."""
    if _rec() is None or not rejected:
        return
    for p, f, a, ppm in attribute(targets, rejected, adducts, tol_ppm):
        note(p, prefix + compact_reason(rejected[f]), neutral=f, adduct=a, ppm=ppm, to_score=True)


@_guarded
def note_grid(targets: pd.DataFrame, formulas, adducts, tol_ppm: float, box: str) -> None:
    """Pass 1's grid: which targets it gave a candidate (before the context
    filter) and which it could not reach (`box`: the grid box in words)."""
    rec = _rec()
    if rec is None or targets is None or not len(targets):
        return
    hit = {p for p, *_ in attribute(targets, formulas, adducts, tol_ppm)}
    for p in targets["peak_id"]:
        if p in hit:
            rec.targets.add(p)
        else:
            rec.box[p] = box


def _row_reading(ledger: pd.DataFrame, i) -> dict:
    def g(c):
        return ledger.at[i, c] if c in ledger.columns else None
    s_ion, s_cmp = _num(g("ion_score")), _num(g("compound_score"))
    score = min(s for s in (s_ion, s_cmp) if s is not None) if (s_ion is not None or s_cmp is not None) else None
    return dict(neutral=_txt(g("neutral_formula")), adduct=_txt(g("adduct")), ion=_txt(g("ion_formula")),
                score=score, ppm=_num(g("ppm_error")))


def carries_columns(ledger: pd.DataFrame) -> bool:
    return all(c in ledger.columns for c in COLUMNS)


def _write(ledger: pd.DataFrame, i, reading: str | None, gate: str | None) -> None:
    for c in COLUMNS:
        col = ledger[c]
        if not (col.dtype == object or pd.api.types.is_string_dtype(col.dtype)):
            ledger[c] = col.astype(object)
    ledger.at[i, READING] = pd.NA if reading is None else reading
    ledger.at[i, GATE] = pd.NA if gate is None else gate


@_guarded
def note_cleared(ledger: pd.DataFrame, i, reason, children=()) -> None:
    """`ledger.clear_assignment` is about to clear row `i` (and release
    `children`, its isotope lines): during a run the recorder keeps the
    cleared reading; on a ledger that already carries the two columns (the
    batch's per-file rewrite) they are written on the rows directly."""
    direct, rec = carries_columns(ledger), _rec()
    if not direct and rec is None:
        return
    pid = ledger.at[i, "peak_id"]
    rd = _row_reading(ledger, i)
    gate = "cleared:" + short(reason)
    own = " ".join(x for x in (rd["neutral"], rd["adduct"]) if x)
    rel = "released: a line of the cleared reading" + (f" {own}" if own else "")
    if direct:
        _write(ledger, i, reading_text(_Block(gate=gate, **rd)), gate)
        for ci in children:
            _write(ledger, ci, None, rel)
        return
    rec.add(pid, gate, **rd)
    for ci in children:
        rec.add(ledger.at[ci, "peak_id"], rel)


@_guarded
def forget(ledger: pd.DataFrame, i) -> None:
    """Row `i` is no longer unexplained (committed, attached, marked): empty
    its two columns where the ledger carries them."""
    if carries_columns(ledger) and (pd.notna(ledger.at[i, READING]) or pd.notna(ledger.at[i, GATE])):
        _write(ledger, i, None, None)


# ---------------------------------------------------------------------------
# the stamp
# ---------------------------------------------------------------------------
def _centre(cfg) -> float:
    mu = _num(getattr(cfg, "cal_mu", None)) if cfg is not None else None
    if mu is None and cfg is not None:
        mu = _num(getattr(cfg, "prior_offset", None))
    return mu or 0.0


def _rank(b: _Block, floor: float, centre: float) -> tuple:
    """Larger is better. Readings of equal standing are ordered by their own
    text, never by when they were noted (a pass may note in set order)."""
    near = -abs((b.ppm if b.ppm is not None else 0.0) - centre)
    if b.neutral and b.score is not None:
        return (3 if b.score >= floor else 0, b.score, near, b.neutral, b.adduct or "")
    if not b.neutral:
        return (2, b.seq, 0.0, b.gate, "")
    return (1, near, 0.0, b.neutral, b.adduct or "")


def best(blocks: dict | None, *, floor: float = SUSPECT_FLOOR, centre: float = 0.0) -> _Block | None:
    if not blocks:
        return None
    return max(blocks.values(), key=lambda b: _rank(b, floor, centre))


def _score_unscored(rec: Recorder, pids, *, client, sample_id, cfg, centre: float, log) -> int:
    """Score the nearest SCORE_TOP_K never-scored readings of each unexplained
    peak in ONE call to the run's scorer; a reading takes the raw score
    (min of ion and compound score) of its own ion on its own peak. Local
    scoring only (never the server's matcher); any failure leaves them
    unscored. Returns how many it scored."""
    from peaky.io import io_mascope as IO
    if sample_id is None or not IO._local_scoring_enabled():
        return 0
    want: dict = {}
    for p in pids:
        bl = [b for b in rec.blocks.get(p, {}).values() if b.to_score and b.score is None and b.neutral]
        if not bl:
            continue
        bl.sort(key=lambda b: (abs((b.ppm if b.ppm is not None else 0.0) - centre), b.neutral, b.adduct or ""))
        want[p] = bl[:SCORE_TOP_K]
    formulas = sorted({b.neutral for bl in want.values() for b in bl})
    if not formulas:
        return 0
    try:
        scored = IO.score_candidates(client, sample_id, formulas,
                                     mechanism_ids=getattr(cfg, "mechanism_ids", None))
    except Exception as e:      # noqa: BLE001 -- a diagnostic never fails the run
        log(f"[blocked] scoring the blocked readings failed ({type(e).__name__}: {e}); they stay unscored")
        return 0
    if scored is None or not len(scored) or not {"is_base", "sample_peak_id", "ion_score"} <= set(scored.columns):
        return 0
    base = scored[scored["is_base"].astype(bool) & scored["sample_peak_id"].notna() & scored["ion_score"].notna()]
    if not len(base):
        return 0
    from peaky.assignment.passes.core import _mech_to_adduct     # lazy: passes import the ledger
    got: dict = {}
    for _, r in base.iterrows():
        k = (r["sample_peak_id"], str(r["compound_formula"]), _mech_to_adduct(r))
        s = [x for x in (_num(r.get("ion_score")), _num(r.get("compound_score"))) if x is not None]
        raw = min(s) if s else None
        if raw is not None and (k not in got or raw > got[k][0]):
            got[k] = (raw, _txt(r.get("ion_formula")), _num(r.get("ppm_error")))
    n = 0
    for p, bl in want.items():
        for b in bl:
            hit = got.get((p, b.neutral, b.adduct))
            if hit is None:
                continue
            b.score, b.ion = hit[0], hit[1] or b.ion
            b.ppm = hit[2] if hit[2] is not None else b.ppm
            n += 1
    return n


def stamp(ledger: pd.DataFrame, rec: Recorder | None = None, *, client=None, sample_id=None, cfg=None,
          log=print) -> dict:
    """Write `blocked_reading` / `blocked_by` on every unexplained row (NA on
    every other row), from `rec` (default: the active recorder). Returns the
    rows per gate class."""
    rec = rec if rec is not None else _rec()
    for c in COLUMNS:
        ledger[c] = pd.Series([pd.NA] * len(ledger), index=ledger.index, dtype=object)
    if rec is None or "role" not in ledger.columns or "peak_id" not in ledger.columns:
        return {}
    un = ledger.index[ledger["role"].astype(str) == _UNEXPLAINED]
    if not len(un):
        return {}
    floor = _num(getattr(cfg, "tau_suspect", None)) if cfg is not None else None
    floor = SUSPECT_FLOOR if floor is None else floor
    centre = _centre(cfg)
    n_scored = _score_unscored(rec, list(dict.fromkeys(ledger.loc[un, "peak_id"])), client=client,
                               sample_id=sample_id, cfg=cfg, centre=centre, log=log)
    adm = ledger["admitted_by"] if "admitted_by" in ledger.columns else None
    counts: dict = {}
    for i in un:
        pid = ledger.at[i, "peak_id"]
        b = best(rec.blocks.get(pid), floor=floor, centre=centre)
        reading = None
        if b is not None:
            reading, gate = reading_text(b), b.gate
        elif pid in rec.box:
            gate = f"no candidate in box ({rec.box[pid]})"
        elif pid in rec.targets:
            gate = NO_SCORED
        elif adm is not None and _txt(adm.at[i]) is None:
            gate = NOT_ADMITTED
        else:
            gate = NO_RECORD
        ledger.at[i, READING] = pd.NA if reading is None else reading
        ledger.at[i, GATE] = gate
        k = gate_class(gate)
        counts[k] = counts.get(k, 0) + 1
    top = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    log(f"[blocked] {len(un)} unexplained rows ({n_scored} blocked readings scored at the stamp): "
        + ", ".join(f"{k} {v}" for k, v in top[:8]) + (" ..." if len(top) > 8 else ""))
    if rec.errors:
        log(f"[blocked] WARNING {rec.errors} note(s) failed and were dropped (first: {rec.first_error})")
    return counts


# ---------------------------------------------------------------------------
# the batch table
# ---------------------------------------------------------------------------
TABLE_COLUMNS = ("mz", "mz_min", "mz_max", "n_files", "n_rows", "median_height", "max_height",
                 "reading", "n_files_reading", "reading_score_median", "gate", "gates")


def _median_score(readings) -> float:
    s = [x for x in map(reading_score, readings) if x is not None]
    return float(np.median(s)) if s else np.nan


def batch_table(ledgers: dict, *, tol_ppm: float, offsets: dict | None = None) -> pd.DataFrame:
    """One row per m/z cluster of the batch's unexplained per-file rows
    (single-linkage over the pooled rows at `tol_ppm`, the merge's window, on
    the m/z corrected by each file's offset in `offsets` -- {sid: ppm}, as the
    merge aligns): the cluster's median / min / max m/z, the files and rows in it, the
    median and maximum height, the blocked reading most files show (without
    its numbers: '<neutral> <adduct> (<ion>)'), how many files show it and its
    median score there, the most frequent gate class and every gate class's
    row count ('score<min_raw 5; context:O/C 3'). Brightest first."""
    offsets = offsets or {}
    parts = []
    for sid in sorted(ledgers, key=str):
        led = ledgers[sid]
        if led is None or not len(led) or not {"role", "mz"} <= set(led.columns):
            continue
        u = led[led["role"].astype(str) == _UNEXPLAINED]
        if not len(u):
            continue
        off = _num(offsets.get(sid)) or 0.0
        parts.append(pd.DataFrame({
            "mz": pd.to_numeric(u["mz"], errors="coerce").to_numpy(dtype=float) * (1.0 - off / 1e6),
            "sid": str(sid),
            "height": (pd.to_numeric(u["height"], errors="coerce").to_numpy(dtype=float)
                       if "height" in u.columns else np.nan),
            "reading": u[READING].to_numpy(dtype=object) if READING in u.columns else None,
            "gate": u[GATE].to_numpy(dtype=object) if GATE in u.columns else None,
        }))
    if not parts:
        return pd.DataFrame({c: pd.Series(dtype=object) for c in TABLE_COLUMNS})
    df = pd.concat(parts, ignore_index=True)
    df = df[np.isfinite(df["mz"])].sort_values(["mz", "sid"], kind="mergesort").reset_index(drop=True)
    if not len(df):
        return pd.DataFrame({c: pd.Series(dtype=object) for c in TABLE_COLUMNS})
    ms = df["mz"].to_numpy(dtype=float)
    cid = np.zeros(len(ms), dtype=np.int64)
    if len(ms) > 1:
        cid[1:] = np.cumsum(np.diff(ms) / ms[:-1] * 1e6 > float(tol_ppm))
    df["key"] = df["reading"].map(reading_key)
    df["klass"] = df["gate"].map(gate_class)
    out = []
    for _, g in df.groupby(cid, sort=True):
        read, n_read, s_med = "", 0, np.nan
        kk = g[g["key"] != ""]
        if len(kk):
            per = kk.groupby("key")["sid"].nunique()
            med = kk.groupby("key")["reading"].agg(_median_score)
            read = sorted(per.index, key=lambda k: (-per[k], -(med[k] if np.isfinite(med[k]) else -1.0), k))[0]
            n_read, s_med = int(per[read]), float(med[read])
        gc = g["klass"].value_counts()
        order = sorted(gc.items(), key=lambda kv: (-kv[1], kv[0]))
        h = g["height"].to_numpy(dtype=float)
        out.append({
            "mz": round(float(np.median(g["mz"])), 5), "mz_min": round(float(g["mz"].min()), 5),
            "mz_max": round(float(g["mz"].max()), 5), "n_files": int(g["sid"].nunique()), "n_rows": int(len(g)),
            "median_height": float(np.nanmedian(h)) if np.isfinite(h).any() else np.nan,
            "max_height": float(np.nanmax(h)) if np.isfinite(h).any() else np.nan,
            "reading": read, "n_files_reading": n_read,
            "reading_score_median": round(s_med, 3) if np.isfinite(s_med) else np.nan,
            "gate": order[0][0], "gates": "; ".join(f"{k} {v}" for k, v in order),
        })
    t = pd.DataFrame(out, columns=list(TABLE_COLUMNS))
    return t.sort_values(["median_height", "mz"], ascending=[False, True], kind="mergesort",
                         na_position="last").reset_index(drop=True)
