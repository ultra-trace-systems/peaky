"""The batch's m/z axis from its own peaks: calibration-free LOCKS and a piecewise model.

No reference list and no calibrant. Every input is either first principles or the
batch itself:

  * exact atomic masses and the electron's; even-electron ions of closed-shell
    neutrals ((H + halogens + N) odd, a ring/double-bond count >= 0) and an
    oxygen ceiling;
  * natural isotope abundances: a 13C/12C ratio that must match the carbon
    number, and the 34S / 37Cl / 81Br line an S / Cl / Br formula must show;
  * a real ion recurs in most of the batch's spectra, noise does not;
  * the axis error varies smoothly with m/z between steps, so (a) one unit step
    (CH2, O, CO2 ...) barely changes it -- a mass DIFFERENCE between two peaks is
    calibration-free -- and (b) correct formulas line up on one curve while wrong
    ones scatter off it.

The procedure (`fit`):

  1. LOCKS: a peak in >= PASS1_FRAC of the spectra -- not an isotopologue of a
     brighter one (13C, 18O, 37Cl ... above it, within ISO_CHILD_PPM so a step
     between them does not hide it: the space holds monoisotopic formulas only,
     so a wrong one can be unique there) -- whose +-LOCK_PPM window holds exactly
     one formula once its own isotope lines have struck the rivals (an S / Cl /
     Br rival only when its line would clear the detection floor and is absent),
     and that formula is CONFIRMED (no monoisotopic F or non-reagent I; S / Cl /
     Br only with their line recurring at the ratio the formula predicts). Then
     locks by exact mass difference: a peak one unit away from >= 2 locked ions
     that all propose the same (structurally valid) formula, nothing proposing
     another.
  2. The SMOOTH axis: a smoothing spline (GCV) through the MAIN BODY of the
     locks -- the longest run without a MAX_GAP_DA gap or a step, and cut where
     the share of the recurring peaks that lock collapses (`_yield_span`: past an
     undetected step the true ions no longer lock and only coincidences remain,
     which would bridge it); bright ions (lock masses read 0 by construction) are
     kept out. Locks more than OFF_CURVE_PPM off it are dropped (a formula outside
     the space sits off by its own error), then a peak whose +-NARROW_PPM window
     around its CORRECTED mass holds one confirmed formula is locked too.
  3. STEPS: an Orbitrap's axis can step (measured: -2.6 ppm between two adjacent
     peaks, then +2 ppm 30 Da higher, in the instrument's own centroids), which
     no smooth curve follows and which breaks the difference links. Above the
     curve's last lock the model walks up segment by segment: locked ions +- one
     unit propose formulas for the next peaks, each proposal implies that peak's
     own m/z error, and the straight line most of them agree on (RANSAC) is the
     segment's axis; a proposal on it that is also the only formula of the space
     within +-NARROW_PPM of the corrected mass locks; a segment is cut at a step
     inside it; when nothing more locks, the next segment starts above.
     A segment needs SEG_MIN_LOCKS locks on its line (off it: evicted), a slope
     no steeper than a smooth stretch and a jump from the one below no larger
     than STEP_MAX_PPM; a stretch inside a segment where recurring peaks sit but
     none fits its model is a HOLE, left uncorrected.
  4. GAPS: two segments whose lines meet across their gap (a JOIN: the walk only
     resumed past a sparse stretch) are bridged. Across a STEP, it lies somewhere in
     the gap. A recurring
     peak in it that holds a unique formula under exactly one side's model, one
     exact unit step from a lock on that side, extends that side; the rest of the gap -- and everything past the first and
     last lock -- is left uncorrected rather than corrected by the wrong side.

`AxisModel` is the result: a spline segment and line segments, each over its own
m/z domain, nothing corrected outside them. It duck-types `wave.WaveFit` where
the batch applies a correction (`predict`, `mz_range`, `as_dict`) and carries its
own `in_scope` and `invert` (back to the server's axis, segment by segment).

Cross-validation (`_validate`) is 5-fold and interleaved: it measures how well the
model predicts a lock it did not see, not whether the locks are right -- a
self-consistent set of wrong locks validates as well as a right one, which is what
the yield cut, the off-curve drop and the gap rule are for.
"""
from __future__ import annotations

import collections
import math
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

__all__ = ["AxisModel", "FormulaSpace", "fit", "MASS", "UNITS"]

ME = 0.00054857990946
MASS = {"C": 12.0, "H": 1.00782503207, "N": 14.0030740048, "O": 15.99491461956,
        "S": 31.97207100, "F": 18.99840322, "Cl": 34.96885268, "Br": 78.9183376,
        "I": 126.904473}
HALOGENS = ("F", "Cl", "Br", "I")
D13C, R13 = 1.003354835, 0.010816                  # 13C - 12C; 13C/12C per carbon
#: the line that confirms an element, and its height per atom relative to the parent
HEAVY_LINES = {"S": (1.995796, 0.0447), "Cl": (1.997050, 0.320), "Br": (1.997954, 0.973)}
#: a heavy line counts as SEEN when it recurs beside the parent in this share of the
#: spectra that hold the parent, at SEEN_RATIO x the ratio the formula predicts
SEEN_FRAC = 0.5
SEEN_RATIO = (0.5, 1.6)
#: monoisotopic elements no isotope line can confirm: never locked (the reagent's own
#: element excepted -- an iodide-adduct batch puts I in every analyte ion)
UNCONFIRMABLE = ("F", "I")
#: heavy-isotope shifts and the parent/child height ratio at or above which a peak that
#: far above a recurring peak is that peak's isotopologue (13C ... 34S: >= 3x; 37Cl
#: (0.32 per atom): >= 2x; 81Br (0.97): >= 0.7x)
ISO_SHIFTS = ((1.003354835, 3.0), (0.997035, 3.0), (1.004217, 3.0), (2.004246, 3.0),
              (2.006710, 3.0), (1.995796, 3.0), (1.997050, 2.0), (1.997954, 0.7))
ISO_CHILD_PPM = 5.0    # wide enough that a step between parent and isotopologue cannot hide it
#: one-unit steps between homologues / oxidation states (neutral, closed-shell)
UNITS = {"CH2": {"C": 1, "H": 2}, "O": {"O": 1}, "O2": {"O": 2}, "H2": {"H": 2},
         "H2O": {"H": 2, "O": 1}, "CO": {"C": 1, "O": 1}, "CH2O": {"C": 1, "H": 2, "O": 1},
         "CO2": {"C": 1, "O": 2}, "C2H2O": {"C": 2, "H": 2, "O": 1},
         "C2H4O": {"C": 2, "H": 4, "O": 1}}
UNIT_MASS = {u: sum(MASS[el] * k for el, k in d.items()) for u, d in UNITS.items()}

BIN_PPM = 3.0          # peaks of the spectra chained into one bin across this gap
LOCK_PPM = 5.0         # the calibration-free window: the raw axis must sit inside it somewhere
NARROW_PPM = 1.0       # the window around a CORRECTED mass (~6x the locks' own scatter)
PASS1_FRAC = 0.8       # a calibration-free lock recurs in this share of the spectra ...
EXT_FRAC = 0.6         # ... a lock found with the axis known, in this share
BRIGHT_X = 100.0       # bins this many times the median height stay out of the fit
FLOOR_X = 3.0          # detection floor = this x the spectra's 1st-percentile height
WIDE_13C_X = 2.0       # an expected 13C line under this x the floor reads noisily ...
WIN_13C = (0.75, 1.25)  # ... the 13C/12C AREA ratio vs expected: 0.9-1.0x measured
WIN_13C_WIDE = (0.5, 1.5)
WIN_13C_HEIGHT = (0.65, 1.25)  # heights read low on bright ions (~0.8x)
LINK_MDA = 0.5         # a difference link: measured step vs the unit's exact mass
OFF_CURVE_PPM = 1.5    # a lock this far off the curve is dropped (~10x the locks' scatter) --
LOCAL_K = 6            # ... judged first against the median of its LOCAL_K nearest locks
SPLINE_MAX_SLOPE = 40.0  # |d ppm / d m/z| x m/z: the smooth curve never bends faster. Scale-free:
                         # a real axis read ~22 at both ends (0.37 ppm/Da at m/z 60, 0.07 at 300);
                         # a curve bent to a wrong lock read 580 and more
MAX_GAP_DA = 15.0      # the smooth curve spans the longest run of locks without a gap this wide ...
BREAK_K = 4            # ... or a STEP: the median error of BREAK_K locks on either side differs
BREAK_PPM = 1.5        # ... by more than this, faster than a smooth axis bends (the steepest
BREAK_SLOPE = 0.15     # ... smooth stretch measured is ~0.07 ppm/Da)
YIELD_WIN_DA = 20.0    # ... and it ends where, in a window this wide, the share of the recurring
YIELD_DROP = 0.3       # ... peaks that lock falls under this x the body's median share
MAX_ERR_PPM = 10.0     # a proposed formula further off than this is not considered
SEG_TOL_PPM = 0.5      # a proposal agrees with a segment's line within this
SEG_SPAN_DA = 30.0     # the walk looks this far past its current top
ANCHOR_DA = 60.0       # anchors within this of a target propose formulas for it
SEG_MIN_LOCKS = 5      # a segment needs at least this many locks (a family of radicals put 3-4
                       # self-consistent wrong locks on one line)
STEP_MAX_PPM = 4.0     # a jump between segments larger than this is no Orbitrap step (measured:
                       # 2-3 ppm) but a family of wrong formulas -- that segment is not kept
HOLE_DA = 6.0          # inside a segment, a stretch this long between two locks that holds
HOLE_MIN_PEAKS = 3     # this many recurring peaks, none with a unique formula under the
                       # segment's model while >= 2 fit it shifted by one of HOLE_SHIFTS (ppm),
                       # is an excursion of the axis: left uncorrected
HOLE_SHIFTS = tuple(sorted({sg * d for d in (1.5, 2.0, 2.5, 3.0, 3.5, 4.0) for sg in (-1, 1)}))
SEG_BREAK_K = 3        # a step inside a segment: BREAK_K for its (fewer) locks
SEG_CV_MAX = 0.35      # a line segment whose own held-out locks scatter more than this (ppm,
                       # ~2x a real segment's) is not an axis: dropped, with the ones above it
TRIM_K = 3             # the curve's last (first) 1..TRIM_K locks are cut when they all sit
TRIM_REF = 8           # > BREAK_PPM, on one side, off the line through the TRIM_REF before them:
                       # the first locks past a step, too few for the BREAK_K test
STEP_MIN_PPM = 1.0     # two segments' lines this far apart across their gap make a STEP;
                       # closer, the walk only resumed past a sparse stretch (a join)
MIN_LOCKS = 30         # fewer locks than this: no model
SCOPE_SLACK_PPM = 3.0  # a segment's domain: its extreme locks' MEASURED m/z, +- the bin width
                       # (20 ppm carried a correction across a step to a peak 17 ppm past the edge)
RANSAC_DRAWS = 2000
RANSAC_MAX_SLOPE = 0.1  # ppm per Da; the steepest smooth stretch measured is ~0.07

_TOK = re.compile(r"([A-Z][a-z]?)(\d*)")


def parse(ion: str) -> collections.Counter:
    return collections.Counter({el: int(k) if k else 1 for el, k in _TOK.findall(str(ion).rstrip("+-"))})


def name(d, polarity: str = "-") -> str:
    order = ["C", "H", "N", "O", "S", "F", "Cl", "Br", "I"]
    return "".join(el + ("" if d[el] == 1 else str(d[el])) for el in order if d.get(el)) + polarity


def exact_mz(ion: str, polarity: str = "-") -> float:
    return sum(MASS[el] * k for el, k in parse(ion).items()) + (ME if polarity == "-" else -ME)


# --------------------------------------------------------------------------- #
# the formula space: every closed-shell even-electron ion, no chemistry rules
# --------------------------------------------------------------------------- #
class FormulaSpace:
    """C0-40 H0-84 N0-4 O0-30 S0-1 F0-13 Cl0-1 (+ the reagent's Br / I, 0-2), as ions of
    `polarity`. Only structure constrains it: (H + halogens + N) odd (an
    even-electron ion of a closed-shell neutral), a neutral ring/double-bond
    count >= 0, O <= 2C + 3N + 4S + 2, F only on carbon, no Cl beside F."""

    RANGES = {"C": 40, "H": 84, "N": 4, "O": 30, "S": 1, "F": 13, "Cl": 1}

    def __init__(self, polarity: str = "-", reagent_elements=()):
        self.polarity = polarity
        self.reagent_elements = tuple(e for e in reagent_elements if e in ("Br", "I"))
        self.ranges = dict(self.RANGES, **{e: 2 for e in self.reagent_elements})
        els = [e for e in self.ranges if e != "H"]
        grids = np.meshgrid(*[np.arange(self.ranges[e] + 1) for e in els], indexing="ij")
        g = {e: a.ravel() for e, a in zip(els, grids)}
        keep = ((g["O"] <= 2 * g["C"] + 3 * g["N"] + 4 * g["S"] + 2)
                & ~((g["F"] > 0) & (g["C"] == 0)) & ~((g["Cl"] > 0) & (g["F"] > 0)))
        self.els = els
        self.g = {e: a[keep].astype(np.int16) for e, a in g.items()}
        self.halo = sum(self.g[e] for e in els if e in HALOGENS)
        self.heavy = sum(self.g[e] * MASS[e] for e in els) + (ME if polarity == "-" else -ME)
        # the neutral's ring/double-bond count from the ion's: [M-H]- / [M+NO3]- read
        # DBE_ion + 0.5, [M+H]+ DBE_ion + 1.5 -- the looser one keeps more rivals
        self.dbe_off = 0.5 if polarity == "-" else 1.5

    def valid(self, d) -> bool:
        """Is this composition in the space (the structural rules and the ranges)?"""
        if any(d.get(e, 0) < 0 for e in d) or any(d.get(e, 0) > self.ranges.get(e, 0) for e in d if e != "_"):
            return False
        X = d.get("H", 0) + sum(d.get(e, 0) for e in HALOGENS)
        n, c = d.get("N", 0), d.get("C", 0)
        return bool((X + n) % 2 == 1 and c - X / 2 + n / 2 + self.dbe_off >= 0
                    and d.get("O", 0) <= 2 * c + 3 * n + 4 * d.get("S", 0) + 2
                    and not (d.get("F", 0) and not c) and not (d.get("Cl", 0) and d.get("F", 0)))

    def near(self, mz_true: float, tol_ppm: float) -> pd.DataFrame:
        """Every ion within +-tol_ppm of `mz_true`: columns ion, mz, ppm (mz_true vs
        it) and one count column per element."""
        rows = []
        h0 = np.rint((mz_true - self.heavy) / MASS["H"]).astype(np.int64)
        for dh in (-1, 0, 1):
            h = h0 + dh
            mz = self.heavy + h * MASS["H"]
            ppm = (mz_true - mz) / mz * 1e6
            X = h + self.halo
            ok = ((h >= 0) & (h <= self.ranges["H"]) & (np.abs(ppm) <= tol_ppm)
                  & ((X + self.g["N"]) % 2 == 1)
                  & (self.g["C"] - X / 2 + self.g["N"] / 2 + self.dbe_off >= 0))
            for i in np.flatnonzero(ok):
                d = collections.Counter({e: int(self.g[e][i]) for e in self.els})
                d["H"] = int(h[i])
                rows.append({"ion": name(d, self.polarity), "mz": float(mz[i]),
                             "ppm": float(ppm[i]), **{e: d[e] for e in self.els + ["H"]}})
        cols = ["ion", "mz", "ppm"] + self.els + ["H"]
        out = pd.DataFrame(rows, columns=cols).drop_duplicates("ion")
        return out.reset_index(drop=True)

    def confirmable(self, d) -> bool:
        """No element the batch could never confirm (F, a non-reagent I)."""
        return not any(d.get(e, 0) and e not in self.reagent_elements for e in UNCONFIRMABLE)


# --------------------------------------------------------------------------- #
# the batch's recurring peaks and their isotope lines
# --------------------------------------------------------------------------- #
@dataclass
class Bins:
    table: pd.DataFrame        # bin, mz, h, area, n_files, frac, r13, r_<el>, f_<el>, iso_child
    n_spectra: int
    floor: float               # detection floor (height units)
    intensity: str             # 'area' or 'height' -- what the isotope ratios are read on


def _nearest(sorted_mz, targets, tol_ppm):
    """Index of the nearest of `sorted_mz` to each target and whether it is within tol."""
    if len(sorted_mz) == 0:
        return np.zeros(len(targets), dtype=int), np.zeros(len(targets), dtype=bool)
    j = np.clip(np.searchsorted(sorted_mz, targets), 1, max(len(sorted_mz) - 1, 1))
    j0 = np.clip(j - 1, 0, len(sorted_mz) - 1)
    j1 = np.clip(j, 0, len(sorted_mz) - 1)
    jj = np.where(np.abs(sorted_mz[j0] - targets) <= np.abs(sorted_mz[j1] - targets), j0, j1)
    return jj, np.abs(sorted_mz[jj] - targets) / targets * 1e6 < tol_ppm


def bins_from_ts(ts: pd.DataFrame, sample_col: str = "sample_item_id") -> Bins:
    """One row per m/z bin of the batch's peaks (chained across BIN_PPM gaps, the
    tallest peak per spectrum): the share of spectra it is in; per spectrum then the
    median, its 13C/12C ratio and its 34S / 37Cl / 81Br line ratio, with the share of
    the spectra holding it that hold that line too; and whether it is the
    isotopologue of a brighter recurring bin."""
    use_area = "area" in ts.columns and pd.to_numeric(ts["area"], errors="coerce").gt(0).mean() > 0.9
    cols = [sample_col, "mz", "height"] + (["area"] if use_area else [])
    t = ts[cols].copy()
    t["mz"] = pd.to_numeric(t["mz"], errors="coerce")
    t["height"] = pd.to_numeric(t["height"], errors="coerce")
    t = t[np.isfinite(t["mz"]) & np.isfinite(t["height"]) & (t["height"] > 0)]
    inten = "area" if use_area else "height"
    if use_area:
        t["area"] = pd.to_numeric(t["area"], errors="coerce")
        t = t[t["area"] > 0]
    t = t.sort_values("mz").reset_index(drop=True)
    n = int(t[sample_col].nunique())
    edge = t.groupby(sample_col)["height"].quantile(0.01)
    floor = FLOOR_X * float(edge.median()) if len(edge) else 0.0
    m = t["mz"].to_numpy()
    gap = np.diff(m) / m[1:] * 1e6 if len(m) > 1 else np.array([])
    t["bin"] = np.concatenate([[0], np.cumsum(gap > BIN_PPM)]) if len(m) else []
    allpk = t                                      # every peak: the partner lines are read here
    t = t.sort_values("height", ascending=False).drop_duplicates(["bin", sample_col]).reset_index(drop=True)
    lines = {"13C": D13C, **{el: d for el, (d, _) in HEAVY_LINES.items()}}
    ratio = {k: np.full(len(t), np.nan) for k in lines}
    full = {sid: g.sort_values("mz") for sid, g in allpk.groupby(sample_col)}
    for sid, g in t.groupby(sample_col):
        # against ALL of the spectrum's peaks, not the tallest per bin: in a bromide batch
        # an ion's 13C line and a neighbour's 81Br line sit 0.65 mDa apart, chain into one
        # bin, and the taller one read as the 13C line struck the true formula
        gs = full[sid]
        gm, gi = gs["mz"].to_numpy(), gs[inten].to_numpy()
        for k, d in lines.items():
            tgt = g["mz"].to_numpy() + d
            jj, ok = _nearest(gm, tgt, 2.0)
            ratio[k][g.index.to_numpy()] = np.where(ok, gi[jj] / g[inten].to_numpy(), np.nan)
    for k in lines:
        t[f"r_{k}"] = ratio[k]
    for el in HEAVY_LINES:
        t[f"has_{el}"] = t[f"r_{el}"].notna().astype(float)
    agg = t.groupby("bin").agg(
        mz=("mz", "median"), h=("height", "median"), area=(inten, "median"),
        n_files=(sample_col, "nunique"), r13=("r_13C", "median"),
        **{f"r_{el}": (f"r_{el}", "median") for el in HEAVY_LINES},
        **{f"f_{el}": (f"has_{el}", "mean") for el in HEAVY_LINES})
    agg["frac"] = agg["n_files"] / max(n, 1)
    agg = agg.reset_index().sort_values("mz").reset_index(drop=True)
    bm, bh = agg["mz"].to_numpy(), agg["h"].to_numpy()
    child = np.zeros(len(agg), dtype=bool)
    for d, x in ISO_SHIFTS:
        jj, ok = _nearest(bm, bm - d, ISO_CHILD_PPM)
        child |= ok & (bh[jj] >= x * bh) & (jj != np.arange(len(bm)))
    agg["iso_child"] = child
    return Bins(table=agg, n_spectra=n, floor=floor, intensity=inten)


# --------------------------------------------------------------------------- #
# isotope tests
# --------------------------------------------------------------------------- #
def _c13_ok(b, nC: int, bins: Bins) -> bool:
    """The bin's 13C/12C ratio fits nC carbons (untestable -> no veto)."""
    if not np.isfinite(b.r13) or nC <= 0:
        return True
    q = b.r13 / (nC * R13)
    if bins.intensity == "height":
        lo, hi = WIN_13C_HEIGHT
    elif b.h * nC * R13 < WIDE_13C_X * bins.floor:
        lo, hi = WIN_13C_WIDE
    else:
        lo, hi = WIN_13C
    return lo <= q <= hi


def _line_seen(b, el: str, k: int) -> bool:
    """The element's heavy line recurs beside the bin at the ratio k atoms predict."""
    rel = HEAVY_LINES[el][1] * max(k, 1)
    r = getattr(b, f"r_{el}")
    return bool(getattr(b, f"f_{el}") >= SEEN_FRAC and np.isfinite(r)
                and SEEN_RATIO[0] * rel <= r <= SEEN_RATIO[1] * rel)


def _strike(cands: pd.DataFrame, b, bins: Bins) -> pd.DataFrame:
    """Rivals the bin's own isotope lines rule out: a carbon number its 13C ratio
    contradicts (no 13C line where even one carbon would show -> carbon-free only);
    an S / Cl / Br formula whose line would clear the floor and is absent."""
    if cands.empty:
        return cands
    keep = np.ones(len(cands), dtype=bool)
    if np.isfinite(b.r13):
        keep &= np.array([_c13_ok(b, int(c), bins) and c > 0 for c in cands["C"]])
    elif b.h * R13 > bins.floor and b.frac >= PASS1_FRAC:
        keep &= cands["C"].to_numpy() == 0
    for el, (_, rel) in HEAVY_LINES.items():
        if el in cands.columns:
            k = cands[el].to_numpy()
            vis = b.h * rel * np.maximum(k, 1) > bins.floor
            absent = getattr(b, f"f_{el}") < SEEN_FRAC
            keep &= ~((k > 0) & vis & absent)
    return cands[keep]


def _confirmed(d, b, space: FormulaSpace) -> bool:
    if not space.confirmable(d):
        return False
    return all(not d.get(el) or _line_seen(b, el, d[el]) for el in HEAVY_LINES)


def _candidates(b, space: FormulaSpace, bins: Bins, mz_true: float, tol: float) -> pd.DataFrame:
    return _strike(space.near(mz_true, tol), b, bins)


def _unique_lock(b, space: FormulaSpace, bins: Bins, mz_true: float, tol: float) -> str | None:
    c = _candidates(b, space, bins, mz_true, tol)
    if len(c) != 1:
        return None
    row = c.iloc[0]
    d = {e: int(row[e]) for e in space.els + ["H"]}
    return row["ion"] if _confirmed(d, b, space) else None


# --------------------------------------------------------------------------- #
# the model
# --------------------------------------------------------------------------- #
@dataclass
class AxisModel:
    """ppm error of the measured m/z as a function of m/z, one segment per stretch of
    the axis -- a spline (sampled knots, linear interpolation) or a line -- each over
    its own domain [lo, hi]; nothing is corrected outside the domains.
    m/z_corr = m/z_meas * (1 - predict * 1e-6), as WaveFit."""
    segments: list                       # dicts: kind, lo, hi, n, x/y (spline) or a/b (line)
    n_locks: int = 0
    stats: dict = field(default_factory=dict)
    K: int = -1                          # not a Chebyshev wave (massqc dispatches on in_scope)

    @property
    def mz_range(self) -> tuple:
        return (float(self.segments[0]["lo"]), float(self.segments[-1]["hi"]))

    @staticmethod
    def _eval(seg, m):
        m = np.asarray(m, dtype=float)
        if seg["kind"] == "line":
            return seg["a"] + seg["b"] * (m - 400.0)
        return np.interp(m, seg["x"], seg["y"])

    def _domain(self, seg, m):
        dom = (m >= seg["lo"] * (1 - SCOPE_SLACK_PPM * 1e-6)) & (m <= seg["hi"] * (1 + SCOPE_SLACK_PPM * 1e-6))
        for a, b in seg.get("holes") or ():
            dom &= ~((m > a * (1 + SCOPE_SLACK_PPM * 1e-6)) & (m < b * (1 - SCOPE_SLACK_PPM * 1e-6)))
        return dom

    def in_scope(self, mz) -> np.ndarray:
        mz = np.asarray(mz, dtype=float)
        out = np.zeros(mz.shape, dtype=bool)
        for seg in self.segments:
            out |= self._domain(seg, mz)
        return out

    def predict(self, mz, extrapolate: bool = False) -> np.ndarray:
        """The ppm error at `mz`: NaN outside every segment's domain, unless
        `extrapolate` (then the nearest segment's model)."""
        mz = np.asarray(mz, dtype=float)
        out = np.full(mz.shape, np.nan)
        for seg in self.segments:
            sel = self._domain(seg, mz) & np.isnan(out)
            if sel.any():
                out[sel] = self._eval(seg, mz[sel])
        if extrapolate:
            rest = np.isnan(out) & np.isfinite(mz)
            if rest.any():
                lo = np.array([s["lo"] for s in self.segments])
                hi = np.array([s["hi"] for s in self.segments])
                m = mz[rest][:, None]
                dist = np.where(m < lo, lo - m, np.where(m > hi, m - hi, 0.0))
                k = np.argmin(dist, axis=1)
                vals = np.empty(rest.sum())
                for i, seg in enumerate(self.segments):
                    s = k == i
                    if s.any():
                        vals[s] = self._eval(seg, mz[rest][s])
                out[rest] = vals
        return out

    def invert(self, corrected) -> np.ndarray:
        """The measured m/z each corrected one came from: per segment, the fixed point
        of m = c / (1 - ppm(m) * 1e-6) that lies in that segment's domain; a value no
        segment's correction produces came from outside them all and is returned
        unchanged. Where two segments' images overlap (a few tenths of a mDa beside a
        step) the lower segment's reading is taken."""
        c = np.asarray(corrected, dtype=float)
        out = c.copy()
        done = np.zeros(c.shape, dtype=bool)
        for seg in self.segments:
            r = c.copy()
            for _ in range(8):
                r = c / (1 - self._eval(seg, r) * 1e-6)
            ok = self._domain(seg, r) & ~done & np.isfinite(c)
            out[ok] = r[ok]
            done |= ok
        return out

    def max_abs_ppm(self) -> float:
        """The largest correction the model applies anywhere in its domains."""
        vals = [np.abs(self._eval(s, np.linspace(s["lo"], s["hi"], 400))).max() for s in self.segments]
        return float(max(vals))

    def as_dict(self) -> dict:
        segs = []
        for s in self.segments:
            d = {}
            for k, v in s.items():
                if k in ("x", "y"):
                    d[k] = [round(float(q), 5) for q in v]
                elif k == "holes":
                    d[k] = [[float(a), float(b)] for a, b in v]
                elif isinstance(v, (np.integer,)):
                    d[k] = int(v)
                elif isinstance(v, (np.floating, float)):
                    d[k] = float(v)
                else:
                    d[k] = v
            segs.append(d)
        return {"model": "locks", "segments": segs, "n_locks": int(self.n_locks), "stats": self.stats}

    @classmethod
    def from_dict(cls, d: dict) -> "AxisModel":
        segs = []
        for s in d["segments"]:
            s = dict(s)
            if s.get("kind") == "spline":
                s["x"] = np.asarray(s["x"], dtype=float)
                s["y"] = np.asarray(s["y"], dtype=float)
            segs.append(s)
        return cls(segments=segs, n_locks=int(d.get("n_locks", 0)), stats=dict(d.get("stats") or {}))


def main_body(x, y, k: int = BREAK_K) -> tuple[float, float]:
    """The m/z span of the longest run of locks (x sorted) that holds no gap wider
    than MAX_GAP_DA and no step: k locks either side whose median errors differ by
    more than BREAK_PPM at more than BREAK_SLOPE ppm/Da."""
    runs = _runs(x, y, k)
    a, b = max(runs, key=lambda r: r[1] - r[0])
    return float(np.asarray(x, float)[a]), float(np.asarray(x, float)[b - 1])


def _runs(x, y, k):
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    cut = set((np.flatnonzero(np.diff(x) > MAX_GAP_DA) + 1).tolist())
    score = np.zeros(n)
    for i in range(k, n - k + 1):
        jump = abs(np.median(y[i:i + k]) - np.median(y[i - k:i]))
        span = np.median(x[i:i + k]) - np.median(x[i - k:i])
        if jump > BREAK_PPM and span > 0 and jump / span > BREAK_SLOPE:
            score[i] = jump
    for i in np.flatnonzero(score):
        if score[i] == score[max(0, i - k):i + k].max():
            cut.add(int(i))
    edges = [0] + sorted(cut) + [n]
    return [(edges[j], edges[j + 1]) for j in range(len(edges) - 1) if edges[j + 1] > edges[j]]


def _trim_ends(x, y) -> tuple[float, float]:
    """(lo, hi) of the sorted locks (x, y) once the 1..TRIM_K locks at either end that
    all sit more than BREAK_PPM off -- on one side of -- the line through the TRIM_REF
    locks next to them are cut: the first locks past a step."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    lo_i, hi_i = 0, n
    for side in ("hi", "lo"):
        for j in range(TRIM_K, 0, -1):
            if side == "hi":
                tail, ref = slice(hi_i - j, hi_i), slice(max(lo_i, hi_i - j - TRIM_REF), hi_i - j)
            else:
                tail, ref = slice(lo_i, lo_i + j), slice(lo_i + j, min(hi_i, lo_i + j + TRIM_REF))
            if ref.stop - ref.start < 4 or np.ptp(x[ref]) <= 0:
                continue
            b, a = np.polyfit(x[ref], y[ref], 1)
            dev = y[tail] - (a + b * x[tail])
            if np.all(np.abs(dev) > BREAK_PPM) and (np.all(dev > 0) or np.all(dev < 0)):
                if side == "hi":
                    hi_i -= j
                else:
                    lo_i += j
                break
    return float(x[lo_i]), float(x[hi_i - 1])


class _Curve:
    """A smoothing spline (GCV) through the locks, held flat past its end knots."""

    def __init__(self, x, y):
        from scipy.interpolate import make_smoothing_spline
        order = np.argsort(x)
        x, y = np.asarray(x, float)[order], np.asarray(y, float)[order]
        keep = np.concatenate([[True], np.diff(x) > 1e-6])
        self.x, self.y = x[keep], y[keep]
        self.lo, self.hi = float(self.x[0]), float(self.x[-1])
        self.sp = make_smoothing_spline(self.x, self.y) if len(self.x) >= 5 else None
        self.c = float(np.median(self.y)) if len(self.y) else 0.0

    def __call__(self, q):
        q = np.clip(np.asarray(q, float), self.lo, self.hi)
        return self.sp(q) if self.sp is not None else np.full(np.shape(q), self.c)

    def max_slope(self) -> tuple[float, float]:
        """(largest |d ppm / d m/z| x m/z over the knots' span, where)."""
        if self.sp is None or self.hi <= self.lo:
            return 0.0, self.lo
        g = np.linspace(self.lo, self.hi, max(200, int((self.hi - self.lo) * 4)))
        d = np.abs(self.sp.derivative()(g)) * g
        k = int(np.argmax(d))
        return float(d[k]), float(g[k])


def _spline(x, y):
    return _Curve(x, y)


def _local_outliers(x, y, k: int = LOCAL_K) -> np.ndarray:
    """Locks (x sorted) more than OFF_CURVE_PPM off a robust (Theil-Sen) line through
    their k nearest neighbours in m/z: judged without the curve they would bend, and
    with the local slope (the axis rises 0.37 ppm/Da at m/z 60 -- a median of the
    neighbours dropped true locks there)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    out = np.zeros(n, dtype=bool)
    if n <= k:
        return out
    for i in range(n):
        lo, hi = max(0, i - k), min(n, i + k + 1)
        idx = np.r_[lo:i, i + 1:hi]
        nb = idx[np.argsort(np.abs(x[idx] - x[i]))[:k]]
        xa, ya = x[nb], y[nb]
        dx = xa[:, None] - xa[None, :]
        dy = ya[:, None] - ya[None, :]
        ok = dx > 0.5
        slope = float(np.median(dy[ok] / dx[ok])) if ok.any() else 0.0
        icpt = float(np.median(ya - slope * xa))
        out[i] = abs(y[i] - (icpt + slope * x[i])) > OFF_CURVE_PPM
    return out


def _ransac(m, e, w, *, tol=SEG_TOL_PPM, seed=0):
    """The line e = a + b (m - 400) the most target peaks (each counted once, by its
    best-supported proposal) agree with within `tol`; refitted on its inliers."""
    n = len(m)
    if n < 2:
        return None
    keys, inv = np.unique(m, return_inverse=True)
    rng = np.random.default_rng(seed)
    pairs = rng.integers(0, n, size=(RANSAC_DRAWS, 2))
    best, arg = -1.0, None
    for i, j in pairs:
        if abs(m[i] - m[j]) < 5:
            continue
        b = (e[j] - e[i]) / (m[j] - m[i])
        if abs(b) > RANSAC_MAX_SLOPE:
            continue
        a = e[i] - b * (m[i] - 400.0)
        inl = np.abs(e - (a + b * (m - 400.0))) <= tol
        acc = np.zeros(len(keys))
        np.maximum.at(acc, inv[inl], w[inl])
        score = float(acc.sum())
        if score > best:
            best, arg = score, (a, b)
    if arg is None:
        return None
    a, b = arg
    inl = np.abs(e - (a + b * (m - 400.0))) <= tol
    if inl.sum() >= 2 and np.ptp(m[inl]) > 0:
        b2, a2 = np.polyfit(m[inl] - 400.0, e[inl], 1)
        if abs(b2) <= RANSAC_MAX_SLOPE:
            a, b = a2, b2
    return float(a), float(b)


#: the rivals a walk proposal must have none of: 'lockable' -- formulas that could ever be
#: confirmed (no F, no non-reagent I). 'all' (F too) left a real batch's axis above its first
#: step unmodelled -- F-bearing formulas crowd every +-1 ppm window there -- and, its walk
#: starting later, let a gap peak extend the curve across the step.
WALK_RIVALS = "lockable"


def _walk_rivals(cands, space):
    """The rivals a walk proposal must have none of besides itself: 'all' (every
    formula of the space), 'lockable' (those that could ever be confirmed: no F, no
    non-reagent I), 'none'."""
    if WALK_RIVALS == "none":
        return None
    if WALK_RIVALS == "lockable" and not cands.empty:
        bad = np.zeros(len(cands), dtype=bool)
        for e in UNCONFIRMABLE:
            if e in cands.columns and e not in space.reagent_elements:
                bad |= cands[e].to_numpy() > 0
        return cands[~bad]
    return cands


def _shift(ion: str, unit: str, sign: int, polarity: str) -> str | None:
    d = parse(ion)
    for el, k in UNITS[unit].items():
        d[el] += sign * k
    if min(d.values()) < 0:
        return None
    return name(d, polarity)


def _proposals(t_mz, a_mz, a_ion, *, polarity, by="difference"):
    """Formulas the anchors propose for the targets, one unit away. by='difference':
    the MEASURED step matches the unit within LINK_MDA (calibration-free); by='theory':
    the target sits within MAX_ERR_PPM of anchor formula + unit (its implied error is
    what the walk fits). Returns rows (target index, formula, implied ppm, anchor index)."""
    t_mz = np.asarray(t_mz, float)
    a_mz = np.asarray(a_mz, float)
    if not len(t_mz) or not len(a_mz):
        return []
    a_th = np.array([exact_mz(i, polarity) for i in a_ion])
    base = a_mz if by == "difference" else a_th
    rows = []
    for u, um in UNIT_MASS.items():
        for sgn in (1, -1):
            sh = base + sgn * um
            order = np.argsort(sh)
            s = sh[order]
            if by == "difference":
                lo = np.searchsorted(s, t_mz - LINK_MDA * 1e-3)
                hi = np.searchsorted(s, t_mz + LINK_MDA * 1e-3, side="right")
            else:
                lo = np.searchsorted(s, t_mz / (1 + MAX_ERR_PPM * 1e-6))
                hi = np.searchsorted(s, t_mz / (1 - MAX_ERR_PPM * 1e-6), side="right")
            for ti in np.flatnonzero(hi > lo):
                for k in order[lo[ti]:hi[ti]]:
                    f = _shift(a_ion[k], u, sgn, polarity)
                    if f is None:
                        continue
                    th = exact_mz(f, polarity)
                    rows.append((int(ti), f, (t_mz[ti] - th) / th * 1e6, int(k)))
    return rows


def _yield_span(rec_mz, lock_mz, lo, hi) -> tuple[float, float]:
    """Narrow [lo, hi] to where the share of the recurring peaks that lock holds up: walk
    out from the window with the most locks and stop where, in two YIELD_WIN_DA windows
    in a row (each with >= 3 recurring peaks), it falls under YIELD_DROP x the median."""
    rec_mz = np.sort(np.asarray(rec_mz, float))
    lock_mz = np.sort(np.asarray(lock_mz, float))
    starts = np.arange(lo, hi, YIELD_WIN_DA / 2)
    if len(starts) < 3:
        return lo, hi
    nrec = np.array([((rec_mz >= s) & (rec_mz < s + YIELD_WIN_DA)).sum() for s in starts])
    nlock = np.array([((lock_mz >= s) & (lock_mz < s + YIELD_WIN_DA)).sum() for s in starts])
    y = np.where(nrec >= 3, nlock / np.maximum(nrec, 1), np.nan)
    if np.isnan(y).all():
        return lo, hi
    med = float(np.nanmedian(y))
    # a window with too few recurring peaks to judge (NaN) is no evidence either way --
    # a hole is MAX_GAP_DA's business; the walk starts from the window holding the most
    # locks (the highest SHARE can be a 3-peak window at an edge)
    low = np.where(np.isnan(y), False, y < YIELD_DROP * med)
    i0 = int(np.argmax(nlock))
    top = len(starts) - 1
    for i in range(i0, len(starts) - 1):
        if low[i] and low[i + 1]:
            top = i - 1
            break
    bot = 0
    for i in range(i0, 0, -1):
        if low[i] and low[i - 1]:
            bot = i + 1
            break
    new_hi = hi if top == len(starts) - 1 else starts[top] + YIELD_WIN_DA
    new_lo = lo if bot == 0 else starts[bot]
    return float(max(lo, new_lo)), float(min(hi, new_hi))


def fit(ts: pd.DataFrame, *, polarity: str = "-", reagent_elements=(), sample_col="sample_item_id",
        log=print) -> tuple[AxisModel | None, dict]:
    """The batch's axis from its own peaks (module note). Returns (model or None,
    info); info['locks'] lists every lock (m/z, ion, ppm, how it was found)."""
    space = FormulaSpace(polarity, reagent_elements)
    bins = bins_from_ts(ts, sample_col)
    B = bins.table
    info: dict = {"n_spectra": bins.n_spectra, "n_bins": int(len(B)), "intensity": bins.intensity,
                  "n_iso_children": int(B["iso_child"].sum()) if len(B) else 0}
    if bins.n_spectra < 2 or not len(B):
        info["why"] = "fewer than 2 spectra: recurrence cannot tell ions from noise"
        return None, info
    need1 = max(2, math.ceil(PASS1_FRAC * bins.n_spectra - 1e-9))
    need2 = max(2, math.ceil(EXT_FRAC * bins.n_spectra - 1e-9))
    rec = B[(B.n_files >= need2) & ~B.iso_child]
    strong = B[(B.n_files >= need1) & ~B.iso_child]
    med_h = float(strong["h"].median()) if len(strong) else 0.0
    locks: dict[int, dict] = {}            # bin index -> {mz, ion, ppm, how, bright}

    def add(i, ion, how):
        mz = float(B.at[i, "mz"])
        th = exact_mz(ion, polarity)
        locks[i] = {"mz": mz, "ion": ion, "ppm": (mz - th) / th * 1e6, "how": how,
                    "bright": bool(med_h > 0 and float(B.at[i, "h"]) > BRIGHT_X * med_h)}

    def accept(b, f) -> bool:
        d = parse(f)
        return space.valid(d) and _confirmed(d, b, space) and _c13_ok(b, d["C"], bins)

    # 1a. calibration-free uniqueness
    for i, b in strong.iterrows():
        ion = _unique_lock(b, space, bins, float(b.mz), LOCK_PPM)
        if ion:
            add(i, ion, "unique +-5 ppm")
    # 1b. exact mass differences, repeated
    for _ in range(10):
        targets = strong[~strong.index.isin(list(locks))]
        if targets.empty or not locks:
            break
        aidx = list(locks)
        rows = _proposals(targets["mz"].to_numpy(), [locks[i]["mz"] for i in aidx],
                          [locks[i]["ion"] for i in aidx], polarity=polarity, by="difference")
        props = collections.defaultdict(lambda: collections.defaultdict(set))
        for ti, f, e, k in rows:
            if abs(e) <= MAX_ERR_PPM:
                props[ti][f].add(aidx[k])
        new = {}
        for ti, fs in props.items():
            if len(fs) != 1:
                continue
            f, anchors = next(iter(fs.items()))
            i = targets.index[ti]
            if len(anchors) >= 2 and accept(B.loc[i], f):
                new[i] = f
        if not new:
            break
        for i, f in new.items():
            add(i, f, "mass difference")
    info["n_pass1"] = int(sum(v["how"] == "unique +-5 ppm" for v in locks.values()))
    info["n_links"] = int(sum(v["how"] == "mass difference" for v in locks.values()))

    def drop_outside(lo, hi):
        out = [i for i, v in locks.items() if not lo <= v["mz"] <= hi]
        for i in out:
            locks.pop(i)
        return len(out)

    def fit_curve():
        """The smooth curve through the body's locks. A lock is judged off the curve
        against the median of its neighbours first (a spline fitted through a wrong
        lock bends to it: one radical locked as C19H43N4O2- at +4.8 ppm pulled the
        curve to 1.4 ppm/Da), then against the spline; a spline still steeper than
        SPLINE_MAX_SLOPE anywhere loses the lock driving it."""
        for _ in range(40):
            items = sorted(((i, v) for i, v in locks.items() if not v["bright"]), key=lambda kv: kv[1]["mz"])
            if len(items) < 5:
                return None
            x = np.array([v["mz"] for _, v in items])
            y = np.array([v["ppm"] for _, v in items])
            bad = _local_outliers(x, y)
            if bad.any():
                for (i, _), b in zip(items, bad):
                    if b:
                        locks.pop(i)
                continue
            f = _Curve(x, y)
            off = [i for i, v in locks.items() if abs(v["ppm"] - float(f(v["mz"]))) > OFF_CURVE_PPM]
            if off:
                for i in off:
                    locks.pop(i)
                continue
            slope, at = f.max_slope()
            if slope <= SPLINE_MAX_SLOPE:
                return f
            near = [(abs(y[j] - float(np.median(y[max(0, j - LOCAL_K):j + LOCAL_K + 1]))), i)
                    for j, (i, v) in enumerate(items) if abs(v["mz"] - at) <= 5.0]
            if not near:
                return None
            locks.pop(max(near)[1])
        return None

    def body_pts():
        return sorted((v["mz"], v["ppm"]) for v in locks.values() if not v["bright"])

    # 2. the smooth axis over the main body of the locks
    def cut_body():
        n = 0
        pts = body_pts()
        if pts:
            n += drop_outside(*main_body(*np.array(pts).T))
        pts = body_pts()
        if len(pts) > TRIM_REF:
            n += drop_outside(*_trim_ends(*np.array(pts).T))
        return n

    n_out = cut_body()
    n_before = len(locks)
    curve = fit_curve()
    n_off = n_before - len(locks)
    if curve is not None:
        lo_x, hi_x = body_pts()[0][0], body_pts()[-1][0]
        for i, b in rec[(rec.mz >= lo_x) & (rec.mz <= hi_x)].iterrows():
            if i in locks:
                continue
            ion = _unique_lock(b, space, bins, float(b.mz) / (1 + float(curve(b.mz)) * 1e-6), NARROW_PPM)
            if ion and accept(b, ion):
                add(i, ion, "unique +-1 ppm on the curve")
        # where the locks thin out (an undetected step leaves only coincidences)
        lo_y, hi_y = _yield_span(rec["mz"], [v["mz"] for v in locks.values()], lo_x, hi_x + 1e-6)
        n_out += drop_outside(lo_y, hi_y)
        n_out += cut_body()
        n_before = len(locks)
        curve = fit_curve()
        n_off += n_before - len(locks)
    info["n_outside_body"] = int(n_out)
    info["n_off_curve"] = int(n_off)
    pts = body_pts()
    info["n_narrow"] = int(sum(v["how"] == "unique +-1 ppm on the curve" for v in locks.values()))
    if curve is None or len(pts) < MIN_LOCKS:
        info["why"] = (f"{len(pts)} locks, fewer than {MIN_LOCKS}" if len(pts) < MIN_LOCKS else
                       "no smooth curve through the locks within the slope bound")
        info["locks"] = _lock_table(locks)
        return None, info
    lo_x, hi_x = pts[0][0], pts[-1][0]
    grid = np.unique(np.concatenate([np.arange(lo_x, hi_x, 0.5), [hi_x]]))
    segs = [{"kind": "spline", "lo": float(lo_x), "hi": float(hi_x), "n": len(pts),
             "x": grid, "y": curve(grid)}]

    # 3. the walk above the curve's last lock
    top = hi_x
    for _k in range(8):
        sl: dict[int, str] = {}
        line = None
        last_ab = None
        for _rnd in range(15):
            anchors = {**{i: v["ion"] for i, v in locks.items()}, **sl}
            span_hi = (max(B.at[i, "mz"] for i in sl) if sl else top) + SEG_SPAN_DA
            tgt = rec[(rec.mz > top) & (rec.mz <= span_hi) & ~rec.index.isin(list(sl))]
            if len(tgt) < 3:
                # a hole in the recurring ions: reach the next ones, within an anchor's reach
                nxt = rec[(rec.mz > span_hi) & (rec.mz <= top + ANCHOR_DA)].head(10)
                if len(nxt):
                    span_hi = float(nxt["mz"].max())
                    tgt = rec[(rec.mz > top) & (rec.mz <= span_hi) & ~rec.index.isin(list(sl))]
            near_a = [j for j in anchors if B.at[j, "mz"] >= top - ANCHOR_DA]
            rows = _proposals(tgt["mz"].to_numpy(), [B.at[j, "mz"] for j in near_a],
                              [anchors[j] for j in near_a], polarity=polarity, by="theory")
            H = pd.DataFrame([(tgt.index[ti], float(tgt["mz"].iloc[ti]), f, e, near_a[k])
                              for ti, f, e, k in rows],
                             columns=["i", "mz", "ion", "ppm", "anchor"])
            if not H.empty:
                H = H.groupby(["i", "mz", "ion"], as_index=False).agg(ppm=("ppm", "first"),
                                                                       w=("anchor", "nunique"))
            own = pd.DataFrame([(i, float(B.at[i, "mz"]), f,
                                 (B.at[i, "mz"] - exact_mz(f, polarity)) / exact_mz(f, polarity) * 1e6, 3)
                                for i, f in sl.items()], columns=["i", "mz", "ion", "ppm", "w"])
            P = pd.concat([H, own]) if not H.empty else own
            if P.empty or P["i"].nunique() < 3:
                break
            ab = _ransac(P["mz"].to_numpy(float), P["ppm"].to_numpy(float), P["w"].to_numpy(float))
            if ab is None:
                break
            a_, b_ = ab
            last_ab = ab
            line = (lambda q, a_=a_, b_=b_: a_ + b_ * (np.asarray(q, float) - 400.0))
            added = 0
            for i, g in (H.groupby("i") if not H.empty else []):
                ok = g[(g["ppm"] - line(g["mz"].iloc[0])).abs() <= SEG_TOL_PPM]
                if len(ok) != 1 or i in sl:
                    continue
                b = B.loc[i]
                f = ok["ion"].iloc[0]
                # the proposal must also be the only formula there on this line that could
                # ever lock (WALK_RIVALS): see the module constant
                cands = _candidates(b, space, bins, float(b.mz) / (1 + float(line(b.mz)) * 1e-6), NARROW_PPM)
                cands = _walk_rivals(cands, space)
                if (cands is None or set(cands["ion"]) <= {f}) and accept(b, f):
                    sl[i] = f
                    added += 1
            if not added:
                break
        if len(sl) < SEG_MIN_LOCKS:
            break
        # a step or a hole inside the segment: keep its first run, walk on from there
        order = sorted(sl, key=lambda i: B.at[i, "mz"])
        m = np.array([B.at[i, "mz"] for i in order])
        e = np.array([(B.at[i, "mz"] - exact_mz(sl[i], polarity)) / exact_mz(sl[i], polarity) * 1e6
                      for i in order])
        # (the first run long enough to be a segment: a lone lock before a hole is dropped)
        runs = [r for r in _runs(m, e, SEG_BREAK_K) if r[1] - r[0] >= SEG_MIN_LOCKS]
        if not runs:
            break
        a0, b0 = runs[0]
        keep = order[a0:b0]
        m, e = m[a0:b0], e[a0:b0]
        # the segment's line: locks off it evicted and the line refitted (a lock from below
        # the step heading the segment bent its slope); steeper than any smooth stretch,
        # too few locks left, or a jump no Orbitrap step takes -> no segment
        ok = np.ones(len(m), dtype=bool)
        b_, a_ = 0.0, float(np.mean(e))
        for _ in range(6):
            if ok.sum() < 2:
                break
            b_, a_ = np.polyfit(m[ok] - 400.0, e[ok], 1) if np.ptp(m[ok]) > 0 else (0.0, float(np.mean(e[ok])))
            now = np.abs(e - (a_ + b_ * (m - 400.0))) <= SEG_TOL_PPM
            if (now == ok).all():
                break
            ok = now
        keep = [k for k, o in zip(keep, ok) if o]
        m, e = m[ok], e[ok]
        if len(keep) < SEG_MIN_LOCKS or abs(b_) > RANSAC_MAX_SLOPE:
            break
        prev = segs[-1]
        mid = (prev["hi"] + float(m.min())) / 2
        jump = float(a_ + b_ * (mid - 400.0) - AxisModel._eval(prev, mid))
        if abs(jump) > STEP_MAX_PPM:
            info.setdefault("rejected_segments", []).append(
                {"lo": round(float(m.min()), 3), "hi": round(float(m.max()), 3), "jump_ppm": round(jump, 2)})
            break
        sd = float(np.std(e - (a_ + b_ * (m - 400.0))))
        for i in keep:
            add(i, sl[i], f"segment {len(segs) + 1}")
        segs.append({"kind": "line", "lo": float(m.min()), "hi": float(m.max()), "n": int(len(keep)),
                     "a": float(a_), "b": float(b_), "sd": sd})
        top = float(m.max())
    # a line segment whose own locks do not predict each other is no axis: drop it, and
    # the segments walked from it
    info["dropped_segments"] = []
    for s_i in range(1, len(segs)):
        cv = _segment_cv(locks, segs[s_i], f"segment {s_i + 1}")
        segs[s_i]["cv"] = cv
        if cv is not None and cv > SEG_CV_MAX:
            for s_j in range(s_i, len(segs)):
                info["dropped_segments"].append([round(segs[s_j]["lo"], 3), round(segs[s_j]["hi"], 3)])
                for i in [i for i, v in locks.items() if v["how"] == f"segment {s_j + 1}"]:
                    locks.pop(i)
            segs = segs[:s_i]
            break

    # holes: a stretch inside a segment where recurring peaks sit but none holds a unique
    # formula under the segment's model -- an axis excursion no lock reached (a line
    # once ran straight across a 10 Da dip) -- is left uncorrected
    info["holes"] = []
    for seg in segs:
        xs = sorted(v["mz"] for v in locks.values() if not v["bright"] and seg["lo"] <= v["mz"] <= seg["hi"])
        holes = []
        for x1, x2 in zip(xs, xs[1:]):
            if x2 - x1 < HOLE_DA:
                continue
            inner = rec[(rec.mz > x1 + 0.5) & (rec.mz < x2 - 0.5)]
            if len(inner) < HOLE_MIN_PEAKS:
                continue
            def n_fit(shift):
                return sum(bool(_unique_lock(b, space, bins, float(b.mz) / (
                    1 + (float(AxisModel._eval(seg, b.mz)) + shift) * 1e-6), NARROW_PPM))
                    for _, b in inner.iterrows())
            if n_fit(0.0):
                continue
            # no peak fits the model there -- but a crowded formula space alone does that
            # too: a hole needs positive evidence of another axis level, >= 2 of its peaks
            # fitting the model shifted by one same step
            if any(n_fit(d) >= 2 for d in HOLE_SHIFTS):
                holes.append([float(x1), float(x2)])
        seg["holes"] = holes
        info["holes"] += [[round(a, 3), round(b, 3)] for a, b in holes]

    # 4. the gaps between segments: a recurring peak unique under one side only extends it
    model = AxisModel(segments=segs)
    steps, joins = [], []
    def linked(i, f, seg):
        """f is one exact unit step (measured difference) from a lock in `seg`."""
        side = [j for j, v in locks.items() if seg["lo"] <= v["mz"] <= seg["hi"]
                and abs(v["mz"] - B.at[i, "mz"]) <= ANCHOR_DA]
        rows = _proposals([B.at[i, "mz"]], [locks[j]["mz"] for j in side],
                          [locks[j]["ion"] for j in side], polarity=polarity, by="difference")
        return any(g == f for _, g, _, _ in rows)

    gaps = []
    for s in range(len(segs) - 1):
        L, R = segs[s], segs[s + 1]
        mid = (L["hi"] + R["lo"]) / 2
        jump = float(AxisModel._eval(R, mid) - AxisModel._eval(L, mid))
        if abs(jump) < STEP_MIN_PPM:
            # a JOIN: no step between them, nothing ambiguous -- bridge the gap
            joins.append({"between": [round(L["hi"], 3), round(R["lo"], 3)], "ppm": round(jump, 2)})
            L["hi"] = R["lo"] = mid
            continue
        votes = []
        for i, b in rec[(rec.mz > L["hi"]) & (rec.mz < R["lo"])].iterrows():
            fl = _unique_lock(b, space, bins, float(b.mz) / (1 + float(AxisModel._eval(L, b.mz)) * 1e-6), NARROW_PPM)
            fr = _unique_lock(b, space, bins, float(b.mz) / (1 + float(AxisModel._eval(R, b.mz)) * 1e-6), NARROW_PPM)
            # a vote needs both: unique under that side's model only, AND one exact unit
            # step from a lock on that side (a coincidence under the wrong side's model
            # extended it across the step)
            if fl and not fr and linked(i, fl, L):
                votes.append((float(b.mz), "L", i, fl))
            elif fr and not fl and linked(i, fr, R):
                votes.append((float(b.mz), "R", i, fr))
        left = [v for v in votes if v[1] == "L"]
        right = [v for v in votes if v[1] == "R"]
        hiL = max([v[0] for v in left], default=L["hi"])
        loR = min([v[0] for v in right], default=R["lo"])
        if hiL < loR:                      # consistent: extend both sides
            for mz_, side, i, f in left + right:
                if (side == "L" and mz_ <= hiL) or (side == "R" and mz_ >= loR):
                    add(i, f, "gap vote")
            L["hi"], R["lo"] = hiL, loR
            if L["kind"] == "spline" and hiL > L["x"][-1]:
                L["x"] = np.append(L["x"], hiL)
                L["y"] = np.append(L["y"], L["y"][-1])
        mid = (L["hi"] + R["lo"]) / 2
        jump = float(AxisModel._eval(R, mid) - AxisModel._eval(L, mid))
        steps.append({"between": [round(L["hi"], 3), round(R["lo"], 3)], "ppm": round(jump, 2)})
        gaps.append([round(L["hi"], 3), round(R["lo"], 3)])
    model.n_locks = len(locks)
    info["steps"] = steps
    info["joins"] = joins
    info["uncorrected_gaps"] = gaps
    info["locks"] = _lock_table(locks)
    info.update(_validate(model, locks))
    info["max_abs_ppm"] = round(model.max_abs_ppm(), 3)
    info["cv_by_segment"] = [None] + [(round(s_["cv"], 3) if s_.get("cv") is not None else None) for s_ in segs[1:]]
    model.stats = {k: info[k] for k in ("n_pass1", "n_links", "n_narrow", "n_off_curve", "steps", "joins",
                                        "uncorrected_gaps", "dropped_segments", "rejected_segments",
                                        "holes", "cv_by_segment",
                                        "cv_ppm", "raw_ppm", "max_abs_ppm") if k in info}
    return model, info


def _segment_cv(locks, seg, how) -> float | None:
    """Leave-one-out error of a line segment on its own locks (median |residual|, ppm)."""
    pts = sorted((v["mz"], v["ppm"]) for v in locks.values() if v["how"] == how)
    if len(pts) < 4:
        return None
    x, y = np.array(pts).T
    r = []
    for k in range(len(x)):
        m = np.arange(len(x)) != k
        b, a = np.polyfit(x[m] - 400.0, y[m], 1) if np.ptp(x[m]) > 0 else (0.0, y[m].mean())
        r.append(y[k] - (a + b * (x[k] - 400.0)))
    return float(np.median(np.abs(r)))


def _lock_table(locks) -> list:
    return [{"mz": round(v["mz"], 5), "ion": v["ion"], "ppm": round(v["ppm"], 3), "how": v["how"],
             "bright": v["bright"]} for _, v in sorted(locks.items(), key=lambda kv: kv[1]["mz"])]


def _validate(model: AxisModel, locks) -> dict:
    """5-fold (interleaved) cross-validation on the model's own non-bright locks, the
    segments held fixed: how well it predicts a lock it did not see, against leaving
    the axis alone. See the module note for what it cannot see."""
    pts = sorted((v["mz"], v["ppm"]) for v in locks.values() if not v["bright"])
    if len(pts) < 10:
        return {}
    x, y = np.array(pts).T
    seg_of = np.full(len(x), -1)
    for s, seg in enumerate(model.segments):
        seg_of[(x >= seg["lo"]) & (x <= seg["hi"]) & (seg_of < 0)] = s
    folds = np.arange(len(x)) % 5
    pred = np.full(len(x), np.nan)
    for k in range(5):
        tr, te = folds != k, folds == k
        for s in np.unique(seg_of[te & (seg_of >= 0)]):
            mtr, mte = tr & (seg_of == s), te & (seg_of == s)
            if mtr.sum() < 2:
                continue
            if model.segments[s]["kind"] == "spline":
                pred[mte] = _spline(x[mtr], y[mtr])(x[mte])
            else:
                b_, a_ = (np.polyfit(x[mtr] - 400.0, y[mtr], 1) if np.ptp(x[mtr]) > 0
                          else (0.0, y[mtr].mean()))
                pred[mte] = a_ + b_ * (x[mte] - 400.0)
    r = y - pred
    ok = np.isfinite(r)
    return {"cv_ppm": round(float(np.median(np.abs(r[ok]))), 3) if ok.any() else None,
            "raw_ppm": round(float(np.median(np.abs(y))), 3)}
