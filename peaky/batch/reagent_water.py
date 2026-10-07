"""The reagent-ion water ladder of a batch (C15).

A soft-interface CIMS carries its reagent ions hydrated: Br-.(H2O)n, NO3-.(H2O)n,
HNO3.NO3-.(H2O)n ... The per-file reagent library declares only the first rung of the
halide cores (`reagents.build_library`, k = 1) and nothing for the nitrate cores, whose
bare ions are the HNO3 / HNO2 analyte readings. How far a ladder reaches is not a
constant of the chemistry: it moves with the source and the humidity. On a five-day
bromide/nitrate TOF batch the ladders end at n <= 8 (Br-) and n <= 5 (NO3-) before an
instrument restart and run past n = 15 after it, where the higher rungs were committed
as C1-C37 organics (C13H12O8 [M-H]- at Br-.(H2O)12, C13H22N2O4 [M+NO3]- at NO3-.(H2O)15).

So the ladder is MEASURED on the batch's own time series, per acquisition segment:

  * segments  -- the spectra in time order, cut where two consecutive spectra are more
                 than max(GAP_MIN_MINUTES, GAP_X_MEDIAN x the median spacing) apart; a
                 segment of fewer than MIN_SEGMENT_SPECTRA spectra joins its neighbour.
                 One segment when the frame carries no time.
  * a rung    -- core.(H2O)n, n = 1 .. N_MAX, for every core the profile declares
                 (`ReagentProfile.water_cores`; halogen isotopologues enumerated). In
                 a segment it PASSES when the core and every rung 1..n are present (a
                 peak within the window) in >= MIN_PRESENCE of the segment's spectra,
                 and the rung itself is present >= DECOY_X x the highest presence of
                 its decoy offsets (DECOY_OFFSETS_DA; floor DECOY_FLOOR, which never
                 binds while MIN_PRESENCE > DECOY_X x DECOY_FLOOR). A rung that
                 passes in any segment is a reagent ion of the batch -- stamped and
                 stripped over every segment (a known limit: a peak that is the rung
                 after a restart and something else before it goes with the rung;
                 the TOF test below strips by segment instead).
  * the window -- the batch's stamping half-window (`MassScale.stamp_ppm`).

That fixed-offset test is the ORBITRAP's (or any batch without a width model). On a TOF
it fails where the ladder matters most: at m/z 450-700 the line is 0.05-0.07 Da wide, so
offsets of 0.02-0.05 Da sit inside the line, where a TOF picker reports the line's own
weak satellites (about 10 % of its height, ~1-1.5 FWHM below and ~0.5-1 FWHM above it), and
a crowded humid spectrum leaves one weak rung that ends the contiguous ladder. On a
bromide/nitrate TOF batch eight Br-.(H2O)n rungs between n = 21 and 35,
NO3-.(H2O)29..34 and HNO3.NO3-.(H2O)27/31 of one humid stretch stayed Assigned as
C18-C36 organics that way (a known "C30H58Cl4 [M+Br]-" was Br-.(H2O)31: no 13C line
where C30 needs 0.33x). So when the batch's width model is TOF-class
(`evidence.instrument`: R < 50 000 at m/z 200) each rung takes the TOF RUNG TEST
(`_detect_tof`) instead, per segment:

  * present   -- in >= MIN_PRESENCE of the segment's spectra, as above;
  * decoys    -- the local chance of a peak in the window: the MEAN presence at
                 +-k x FWHM(m), k in TOF_DECOY_FWHM (2..5: outside the line's own
                 satellites), skipping a decoy within TOF_DECOY_CLEAR_FWHM FWHM (+ the
                 window) of any declared core or rung -- another ladder is not chance.
                 The rung passes at >= DECOY_X x that (floor DECOY_FLOOR). The mean,
                 not the highest: one real ion among a dozen decoys in a crowded TOF
                 spectrum is not evidence that the rung's position is random;
  * a ladder  -- gaps allowed: the ladder runs from the core through every rung present
                 in >= TOF_LINK_SHARE x MIN_PRESENCE of the spectra (a weak rung carries
                 it past, an absent one ends it), and a rung beyond the contiguous part
                 must CO-VARY with its ladder: median Pearson r of log height >=
                 TOF_COVARY_R with the present rungs within +-TOF_COVARY_SPAN, over the
                 >= TOF_COVARY_MIN_SPECTRA spectra holding both;
  * no carbon -- a cluster has no 13C. Where the rung is bright enough that a line with
                 TOF_M1_CARBONS carbons would show its 13C line above the picker's local
                 floor (the TOF_M1_FLOOR_QUANTILE height within +-TOF_M1_FLOOR_SPAN_DA),
                 the observed M+1/M0 must stay below the cluster's own (2H/17O/15N) plus
                 TOF_M1_CARBONS x 1.07 %; a line holding more carbon is not the rung. An
                 M+1 position within TOF_M1_BLEND_FWHM FWHM (+ the window) of a rung of
                 another declared core present in the segment is not tested: a TOF line
                 there blends with that reagent rung, whose height is not 13C.

A passing rung becomes a reagent row of the batch stamp (`stamp_rows`), and a merged
ANALYTE row whose m/z sits within the window of a passing rung is taken out of the
merged ledger (`strip_rung_rows`) and listed in tables/reagent_water.csv: its reading
is the water cluster. After the TOF test the strip follows the segments: a row leaves
only when a file carrying its winning reading lies in a segment where the rung passed;
a reading carried only by files of segments where the rung did not pass is another line
at the rung's m/z there, and stays (with a note on its `tier_reason`). The per-file
ledgers are untouched. Nothing here tiers or levels a row; a profile without water
cores, or a batch where no core is present, changes nothing.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd

from peaky.chem import chemistry as C

WATER = C.neutral_mass("H2O")
#: decoy offsets (Da) a rung's presence is compared with, in the same segment
DECOY_OFFSETS_DA = (-0.05, -0.035, -0.02, 0.02, 0.035, 0.05)
MIN_PRESENCE = 0.5
DECOY_X = 3.0
DECOY_FLOOR = 0.02
N_MAX = 45
GAP_MIN_MINUTES = 60.0
GAP_X_MEDIAN = 5.0
MIN_SEGMENT_SPECTRA = 10

# The TOF rung test (a TOF-class width model; see the module docstring). Measured on a
# bromide/nitrate TOF batch: the line's own satellites sit at -1.0..-1.75 and mostly
# +0.5..+0.75 FWHM, so decoys start at 2 FWHM; the weakest rung inside a humid ladder is
# present in 0.33 of its segment and the first absent one in <= 0.21 (link at 0.25);
# off-rung lines at the decoy positions co-vary with the ladder at median r 0.25-0.33
# (share >= 0.8: 0.15-0.24; every present off-ladder line at m/z 300-900 against the two
# nearest present rungs: share 0.23 in the humid segment, 0.40 in the dry one), its rungs
# at 0.96-0.99 (0.85-0.99); a bright rung's M+1 excess over its own reaches 4.1
# carbon-equivalents in that crowded spectrum (veto at 5), the readings it displaced
# need 13-36.
#: decoy offsets of the TOF test, in FWHM at the rung's m/z (both signs)
TOF_DECOY_FWHM = (2.0, 2.5, 3.0, 3.5, 4.0, 5.0)
#: a TOF decoy this close (FWHM, plus the window) to a declared core or rung is skipped
TOF_DECOY_CLEAR_FWHM = 1.5
#: a rung present in >= this x MIN_PRESENCE of a segment carries its ladder past it
TOF_LINK_SHARE = 0.5
#: co-variation of a rung beyond the contiguous ladder with its present neighbours
TOF_COVARY_R = 0.8
TOF_COVARY_SPAN = 2
TOF_COVARY_MIN_SPECTRA = 8
#: the M+1 (13C) test of a bright rung: the carbon count it answers for, and the
#: picker's local floor it is judged against (a low quantile of the peak heights near it)
TOF_M1_CARBONS = 5
TOF_M1_FLOOR_QUANTILE = 0.05
TOF_M1_FLOOR_SPAN_DA = 25.0
#: an M+1 position this close (FWHM, plus the window) to a rung of another declared core
#: present in the segment is not tested: the TOF picker reports lines >= ~0.5 FWHM apart
#: as two peaks (the satellites above), closer ones as one blended line
TOF_M1_BLEND_FWHM = 0.5

#: halogen isotopologues a core is enumerated over (mass, tag) -- the reagent
#: library's own tags, so a rung row joins the per-file `[Br1+1xH2O]- (79Br)` rows
_ISOTOPES = {"Br": ((78.9183371, "79Br"), (80.9162906, "81Br")),
             "Cl": ((34.96885268, "35Cl"), (36.96590259, "37Cl"))}
TABLE_COLUMNS = ("core", "iso_tag", "n", "ion_formula", "mz", "mz_obs", "segments",
                 "presence", "decoy_presence", "displaced")


@dataclass(frozen=True)
class Core:
    """One reagent-side core ion: `composition` is its neutral atom count (the anion
    adds an electron), `tag` the halogen isotopologue ('' for none)."""
    composition: str
    tag: str
    mz: float

    @property
    def label(self) -> str:
        return f"{self.composition}-" + (f" ({self.tag})" if self.tag else "")


def cores(water_cores, *, polarity: str = "-") -> list[Core]:
    """The core ions of the declared compositions, halogen isotopologues enumerated
    (a composition with k Br atoms gives k + 1 cores: 79Br..81Br combinations)."""
    out: list[Core] = []
    sign = -1.0 if polarity == "+" else 1.0
    for comp in water_cores or ():
        cnt = C.parse_formula(str(comp))
        hal = [el for el in _ISOTOPES if cnt.get(el, 0)]
        rest = {el: v for el, v in cnt.items() if el not in hal}
        base = C.neutral_mass(C.format_formula(rest)) if rest else 0.0
        if not hal:
            out.append(Core(str(comp), "", base + sign * C.M_E))
            continue
        el = hal[0]
        for combo in itertools.combinations_with_replacement(_ISOTOPES[el], cnt[el]):
            mass = base + sum(m for m, _ in combo) + sign * C.M_E
            out.append(Core(str(comp), "+".join(t for _, t in combo), mass))
    return out


def rung_formula(core: Core, n: int, charge: str = "-") -> str:
    """The ion formula of core.(H2O)n, formatted like the reagent library's."""
    cnt = dict(C.parse_formula(core.composition))
    cnt["H"] = cnt.get("H", 0) + 2 * n
    cnt["O"] = cnt.get("O", 0) + n
    return C.format_formula(cnt) + charge


def segments(ts: pd.DataFrame, *, gap_min: float = GAP_MIN_MINUTES, gap_x: float = GAP_X_MEDIAN,
             min_spectra: int = MIN_SEGMENT_SPECTRA) -> pd.Series:
    """{spectrum id: segment index} over the frame's spectra (`sample_item_id`),
    cut at acquisition gaps; one segment when the frame carries no time."""
    ids = pd.unique(ts["sample_item_id"])
    if "datetime_utc" not in ts.columns or len(ids) < 2:
        return pd.Series(0, index=pd.Index(ids, name="sample_item_id"))
    t = pd.to_datetime(ts.groupby("sample_item_id")["datetime_utc"].first(), utc=True, errors="coerce")
    if t.isna().all():
        return pd.Series(0, index=t.index)
    t = t.sort_values()
    gaps = t.diff().dt.total_seconds().div(60.0)
    thr = max(float(gap_min), float(gap_x) * float(gaps.median()))
    seg = (gaps > thr).cumsum().fillna(0).astype(int)
    # a segment too short to measure a presence on joins its neighbour
    sizes = seg.value_counts().sort_index()
    order = list(sizes.index)
    label = {}
    for i, s in enumerate(order):
        if sizes[s] >= min_spectra or len(order) == 1:
            label[s] = s
        elif i > 0:
            label[s] = label[order[i - 1]]
        else:
            label[s] = None                      # the first one: joins the next kept segment
    nxt = next((label[s] for s in order if label[s] is not None), 0)
    label = {s: (v if v is not None else nxt) for s, v in label.items()}
    # renumber 0.. in time order
    uniq = {v: i for i, v in enumerate(dict.fromkeys(label[s] for s in order))}
    return seg.map(lambda s: uniq[label[s]])


def _presence(sorted_mz: dict, sids, targets: np.ndarray, tol_ppm: float) -> tuple[np.ndarray, list]:
    """Share of `sids` holding a peak within tol of each target, and the matched m/z."""
    hits = np.zeros(len(targets), dtype=float)
    mzs: list[list[float]] = [[] for _ in targets]
    for sid in sids:
        mz = sorted_mz.get(sid)
        if mz is None or not len(mz):
            continue
        j = np.clip(np.searchsorted(mz, targets), 1, len(mz) - 1)
        left, right = mz[j - 1], mz[j]
        best = np.where(np.abs(right - targets) < np.abs(left - targets), right, left)
        ok = np.abs(best - targets) <= targets * tol_ppm * 1e-6
        hits += ok
        for k in np.nonzero(ok)[0]:
            mzs[k].append(float(best[k]))
    n = max(len(sids), 1)
    return hits / n, mzs


def tof_fwhm(resolution):
    """The FWHM(m/z) callable of a TOF-class width model (`evidence.instrument`: R < 50 000
    at m/z 200), else None -- an Orbitrap-class model, or none, keeps the fixed-offset
    test. `resolution` is a chem.resolution.Resolution, its `as_dict`, or a number."""
    if resolution is None:
        return None
    from peaky.assignment import evidence as EV
    klass, fwhm = EV.instrument(resolution)
    return fwhm if klass == "tof" else None


def detect(ts: pd.DataFrame, water_cores, *, tol_ppm: float, polarity: str = "-",
           n_max: int = N_MAX, min_presence: float = MIN_PRESENCE, decoy_x: float = DECOY_X,
           decoy_floor: float = DECOY_FLOOR, decoy_offsets=DECOY_OFFSETS_DA,
           resolution=None) -> pd.DataFrame:
    """One row per PASSING rung (columns TABLE_COLUMNS minus `displaced`): the core,
    n, its ion formula, exact and observed m/z (median of the matched peaks in the
    segments it passed in), those segments, the presence and decoy presence there.

    `resolution` (the batch's width model) picks the rung test: a TOF-class model runs
    the TOF rung test (`_detect_tof`; `decoy_offsets` is then not read), anything else
    the fixed-offset test below."""
    empty = pd.DataFrame(columns=[c for c in TABLE_COLUMNS if c != "displaced"])
    cs = cores(water_cores, polarity=polarity)
    if not cs or ts is None or not len(ts):
        return empty
    fwhm = tof_fwhm(resolution)
    if fwhm is not None:
        return _detect_tof(ts, cs, tol_ppm=tol_ppm, fwhm=fwhm, charge="+" if polarity == "+" else "-",
                           n_max=n_max, min_presence=min_presence, decoy_x=decoy_x,
                           decoy_floor=decoy_floor, columns=empty.columns)
    seg = segments(ts)
    frame = ts[["sample_item_id", "mz"]].dropna()
    sorted_mz = {sid: np.sort(g["mz"].to_numpy(dtype=float)) for sid, g in frame.groupby("sample_item_id")}
    by_seg = {s: list(seg.index[seg == s]) for s in sorted(seg.unique())}
    charge = "+" if polarity == "+" else "-"
    rows = []
    for core in cs:
        ns = np.arange(1, n_max + 1)
        rung_mz = core.mz + ns * WATER
        passed: dict[int, list] = {}
        for s, sids in by_seg.items():
            core_p, _ = _presence(sorted_mz, sids, np.array([core.mz]), tol_ppm)
            if core_p[0] < min_presence:
                continue
            pres, mzs = _presence(sorted_mz, sids, rung_mz, tol_ppm)
            dec = np.zeros(len(ns))
            for off in decoy_offsets:
                d, _ = _presence(sorted_mz, sids, rung_mz + off, tol_ppm)
                dec = np.maximum(dec, d)
            contiguous = np.cumprod(pres >= min_presence).astype(bool)
            ok = contiguous & (pres >= decoy_x * np.maximum(dec, decoy_floor))
            for k in np.nonzero(ok)[0]:
                passed.setdefault(int(ns[k]), []).append((s, float(pres[k]), float(dec[k]), mzs[k]))
        rows.extend(_rung_rows(core, passed, charge))
    return pd.DataFrame(rows, columns=empty.columns) if rows else empty


def _rung_rows(core: Core, passed: dict, charge: str) -> list[dict]:
    """The output rows of one core's passing rungs: `passed` = {n: [(segment, presence,
    decoy presence, matched m/z list), ...]}."""
    rows = []
    for n, hits in sorted(passed.items()):
        obs = [m for _, _, _, ms in hits for m in ms]
        rows.append(dict(core=core.composition, iso_tag=core.tag, n=n,
                         ion_formula=rung_formula(core, n, charge),
                         mz=float(core.mz + n * WATER), mz_obs=float(np.median(obs)) if obs else np.nan,
                         segments="|".join(str(s) for s, _, _, _ in hits),
                         presence="|".join(f"{p:.2f}" for _, p, _, _ in hits),
                         decoy_presence="|".join(f"{d:.2f}" for _, _, d, _ in hits)))
    return rows


# --- the TOF rung test ----------------------------------------------------------------------------------
_PLUS1_RATIO = {"H": "2H", "N": "15N", "O": "17O", "C": "13C", "S": "33S", "Si": "29Si"}


def cluster_m1_ratio(core: Core, n: int) -> float:
    """The cluster's own first-order M+1/M0 (2H, 15N, 17O, 13C, 33S, 29Si per atom; a
    labelled '^N' adds nothing): what the rung's M+1 line holds with no carbon in it."""
    from peaky.chem import isotopes as I
    cnt = dict(C.parse_formula(core.composition))
    cnt["H"] = cnt.get("H", 0) + 2 * n
    cnt["O"] = cnt.get("O", 0) + n
    return float(sum(I.ISOTOPE_RATIO[_PLUS1_RATIO[el]] * v for el, v in cnt.items() if el in _PLUS1_RATIO))


def _match(peaks: dict, sids, targets: np.ndarray, tol_ppm: float):
    """(hit [spectra x targets] bool, height, m/z) of the nearest peak within tol of each
    target in each spectrum (height 0 / m/z NaN where none); `peaks` = {sid: (m/z
    sorted, heights)}."""
    targets = np.asarray(targets, dtype=float)
    hit = np.zeros((len(sids), len(targets)), dtype=bool)
    hgt = np.zeros((len(sids), len(targets)))
    mzs = np.full((len(sids), len(targets)), np.nan)
    for i, sid in enumerate(sids):
        mz, h = peaks.get(sid, (np.empty(0), np.empty(0)))
        if not len(mz):
            continue
        j = np.clip(np.searchsorted(mz, targets), 1, max(len(mz) - 1, 1))
        lo = np.minimum(j - 1, len(mz) - 1)
        hi = np.minimum(j, len(mz) - 1)
        pick = np.where(np.abs(mz[hi] - targets) < np.abs(mz[lo] - targets), hi, lo)
        ok = np.abs(mz[pick] - targets) <= targets * tol_ppm * 1e-6
        hit[i] = ok
        hgt[i] = np.where(ok, h[pick], 0.0)
        mzs[i] = np.where(ok, mz[pick], np.nan)
    return hit, hgt, mzs


def _nearest(sorted_pos: np.ndarray, pos) -> np.ndarray:
    """Distance from each of `pos` to the nearest of `sorted_pos` (inf when there is none)."""
    pos = np.asarray(pos, dtype=float)
    if not len(sorted_pos):
        return np.full(pos.shape, np.inf)
    j = np.searchsorted(sorted_pos, pos)
    return np.minimum(np.abs(sorted_pos[np.minimum(j, len(sorted_pos) - 1)] - pos),
                      np.abs(sorted_pos[np.maximum(j - 1, 0)] - pos))


def _covary(hit: np.ndarray, hgt: np.ndarray, present: np.ndarray, *, span: int = TOF_COVARY_SPAN,
            min_spectra: int = TOF_COVARY_MIN_SPECTRA) -> np.ndarray:
    """Per rung, the median Pearson r of its log height with each PRESENT rung within
    +-span over the spectra holding both (>= min_spectra of them); NaN with none."""
    with np.errstate(divide="ignore", invalid="ignore"):
        lh = np.log10(np.where(hit & (hgt > 0), hgt, np.nan))
    out = np.full(hit.shape[1], np.nan)
    for i in range(hit.shape[1]):
        rs = []
        for j in range(max(i - span, 0), min(i + span, hit.shape[1] - 1) + 1):
            if j == i or not present[j]:
                continue
            both = np.isfinite(lh[:, i]) & np.isfinite(lh[:, j])
            if both.sum() < min_spectra:
                continue
            a, b = lh[both, i], lh[both, j]
            if np.std(a) > 0 and np.std(b) > 0:
                rs.append(float(np.corrcoef(a, b)[0, 1]))
        if rs:
            out[i] = float(np.median(rs))
    return out


def _detect_tof(ts: pd.DataFrame, cs: list, *, tol_ppm: float, fwhm, charge: str, n_max: int,
                min_presence: float, decoy_x: float, decoy_floor: float, columns) -> pd.DataFrame:
    """`detect` on a TOF: the rung test of the module docstring, per segment. A frame
    without heights cannot co-vary or weigh an M+1 line: its ladder is then the
    contiguous one and no rung is tested for carbon."""
    from peaky.chem import isotopes as I
    seg = segments(ts)
    has_h = "height" in ts.columns
    frame = ts[["sample_item_id", "mz"] + (["height"] if has_h else [])].dropna(subset=["mz"])
    peaks = {}
    for sid, g in frame.groupby("sample_item_id"):
        mz = g["mz"].to_numpy(dtype=float)
        h = (pd.to_numeric(g["height"], errors="coerce").fillna(0.0).to_numpy(dtype=float) if has_h
             else np.ones(len(mz)))
        o = np.argsort(mz, kind="mergesort")
        peaks[sid] = (mz[o], h[o])
    by_seg = {s: list(seg.index[seg == s]) for s in sorted(seg.unique())}
    # every segment's peak heights in m/z order: the picker's local floor for the M+1 test
    floors = {}
    for s, sids in by_seg.items():
        mz = np.concatenate([peaks[x][0] for x in sids if x in peaks] or [np.empty(0)])
        h = np.concatenate([peaks[x][1] for x in sids if x in peaks] or [np.empty(0)])
        o = np.argsort(mz, kind="mergesort")
        floors[s] = (mz[o], h[o])
    ns = np.arange(1, n_max + 1)
    ladders = [c.mz + np.arange(0, n_max + 1) * WATER for c in cs]
    # every core's presence per segment: a ladder exists where its core is present
    core_pres = {s: _match(peaks, sids, np.array([c.mz for c in cs]), tol_ppm)[0].mean(axis=0)
                 for s, sids in by_seg.items()}
    r13 = I.ISOTOPE_RATIO["13C"]
    rows = []
    for ci, core in enumerate(cs):
        rung_mz = core.mz + ns * WATER
        fw = np.array([float(fwhm(m)) for m in rung_mz])
        # every OTHER declared core and rung (the core's own sit a water mass away)
        others = np.sort(np.concatenate([x for cj, x in enumerate(ladders) if cj != ci] or [np.empty(0)]))
        decoys = []          # (positions, keep mask) per decoy offset
        for k in TOF_DECOY_FWHM:
            for sign in (-1.0, 1.0):
                pos = rung_mz + sign * k * fw
                keep = _nearest(others, pos) > TOF_DECOY_CLEAR_FWHM * fw + pos * tol_ppm * 1e-6
                decoys.append((pos, keep))
        q_own = np.array([cluster_m1_ratio(core, int(n)) for n in ns])
        m1_exact = rung_mz + I.D_13C
        passed: dict[int, list] = {}
        for s, sids in by_seg.items():
            if core_pres[s][ci] < min_presence:
                continue
            hit, hgt, mzs = _match(peaks, sids, rung_mz, tol_ppm)
            pres = hit.mean(axis=0)
            # the local chance presence: the mean over the decoys left standing
            tot = np.zeros(len(ns))
            cnt = np.zeros(len(ns))
            for pos, keep in decoys:
                p = _match(peaks, sids, pos, tol_ppm)[0].mean(axis=0)
                tot += np.where(keep, p, 0.0)
                cnt += keep
            dec = np.where(cnt > 0, tot / np.maximum(cnt, 1), 0.0)
            present = pres >= min_presence
            linked = np.cumprod(pres >= TOF_LINK_SHARE * min_presence).astype(bool)
            contiguous = np.cumprod(present).astype(bool)
            covaries = (_covary(hit, hgt, present) >= TOF_COVARY_R) if has_h else np.zeros(len(ns), bool)
            ladder = linked & (contiguous | covaries)
            # the M+1 test of a bright rung
            carbon = np.zeros(len(ns), dtype=bool)
            test = np.nonzero(present & ladder)[0]
            if has_h and len(test):
                # the rungs of the OTHER cores present here: an M+1 line that close blends with one
                busy = [ladders[cj] for cj in range(len(cs)) if cj != ci and core_pres[s][cj] >= min_presence]
                busy = np.sort(np.concatenate(busy)) if busy else np.empty(0)
                blended = _nearest(busy, m1_exact[test]) <= (TOF_M1_BLEND_FWHM * fw[test]
                                                             + m1_exact[test] * tol_ppm * 1e-6)
                h1 = _match(peaks, sids, m1_exact[test], tol_ppm)[1]
                fmz, fh = floors[s]
                for k, i in enumerate(test):
                    if blended[k]:
                        continue
                    on = hit[:, i]
                    h0 = float(np.median(hgt[on, i]))
                    a, b = np.searchsorted(fmz, [m1_exact[i] - TOF_M1_FLOOR_SPAN_DA, m1_exact[i] + TOF_M1_FLOOR_SPAN_DA])
                    if b <= a or not I.satellite_observable(
                            "C", TOF_M1_CARBONS, h0, float(np.quantile(fh[a:b], TOF_M1_FLOOR_QUANTILE))):
                        continue
                    q = float(h1[on, k].sum() / max(hgt[on, i].sum(), 1e-300))
                    carbon[i] = q - q_own[i] >= TOF_M1_CARBONS * r13
            ok = present & ladder & (pres >= decoy_x * np.maximum(dec, decoy_floor)) & ~carbon
            for k in np.nonzero(ok)[0]:
                passed.setdefault(int(ns[k]), []).append(
                    (s, float(pres[k]), float(dec[k]), [float(m) for m in mzs[hit[:, k], k]]))
        rows.extend(_rung_rows(core, passed, charge))
    return pd.DataFrame(rows, columns=columns) if rows else pd.DataFrame(columns=columns)


def _txt(v) -> str:
    """A reading's label component as text: '' for any NA."""
    try:
        if v is None or pd.isna(v):
            return ""
    except (TypeError, ValueError):
        pass
    return str(v)


def _segment_set(v) -> set:
    """The segments a rung passed in, from its `segments` field ('0|1'); empty when unreadable."""
    try:
        return {int(float(p)) for p in str(v).split("|") if p.strip()}
    except ValueError:
        return set()


def _reading_files(merged: pd.DataFrame, rows, jitter: pd.DataFrame) -> dict:
    """{row position: [src ...]}: the files carrying each merged row's WINNING reading (its
    neutral_formula + adduct) in its own align cluster -- the jitter cluster whose per-file
    m/z mean is the row's m/z (`assign_batch.align`: the merged m/z is that mean) and, when
    the row lists its files (`srcs`), whose files they are. A row with no such cluster is
    left out (its files are unknown)."""
    need = {"cluster", "src", "mz", "neutral_formula", "adduct"}
    if jitter is None or not len(jitter) or not need <= set(jitter.columns):
        return {}
    mean = jitter.groupby("cluster")["mz"].mean()
    o = np.argsort(mean.to_numpy(dtype=float), kind="mergesort")
    cmz, cid = mean.to_numpy(dtype=float)[o], mean.index.to_numpy()[o]
    out = {}
    for i in rows:
        m = float(pd.to_numeric(merged["mz"].iloc[i], errors="coerce"))
        j = int(np.searchsorted(cmz, m))
        near = [k for k in (j - 1, j) if 0 <= k < len(cmz)]
        if not np.isfinite(m) or not near:
            continue
        k = min(near, key=lambda x: abs(cmz[x] - m))
        if abs(cmz[k] - m) > abs(m) * 1e-9:
            continue
        g = jitter[jitter["cluster"] == cid[k]]
        if "srcs" in merged.columns and _txt(merged["srcs"].iloc[i]):
            if set(_txt(merged["srcs"].iloc[i]).split(",")) != set(g["src"].astype(str)):
                continue
        nf = _txt(merged["neutral_formula"].iloc[i]) if "neutral_formula" in merged.columns else ""
        ad = _txt(merged["adduct"].iloc[i]) if "adduct" in merged.columns else ""
        w = g[(g["neutral_formula"].map(_txt) == nf) & (g["adduct"].map(_txt) == ad)]
        out[i] = [str(x) for x in w["src"]]
    return out


def strip_rung_rows(merged: pd.DataFrame, rungs: pd.DataFrame, *, tol_ppm: float, log=print,
                    jitter: pd.DataFrame | None = None, segment_of=None):
    """(kept, stripped): the merged analyte rows whose m/z sits within tol of a passing
    rung's observed m/z are the water cluster, not the reading; they leave the merged
    ledger. `stripped` carries the rung each one sat on (`rung`).

    With `jitter` (align's per-file readings) and `segment_of` ({spectrum: segment};
    `measure` returns it when the TOF test ran) the strip follows the segments: a row
    leaves only when a file carrying its WINNING reading lies in a segment where its rung
    passed. A reading carried only by files of segments where the rung did not pass reads
    another line at the rung's m/z there (a ladder that reaches the rung in a humid
    stretch can be absent in a dry one): it stays, with a note on its `tier_reason`. A row
    whose reading's files have no known segment leaves as before. Without them (the
    fixed-offset test) a passing rung strips over the whole batch."""
    if merged is None or not len(merged) or rungs is None or not len(rungs):
        return merged, (merged.iloc[0:0].copy() if merged is not None else pd.DataFrame())
    ref = pd.to_numeric(rungs["mz_obs"], errors="coerce").fillna(pd.to_numeric(rungs["mz"])).to_numpy()
    order = np.argsort(ref)
    ref = ref[order]
    labels = [f"{rungs['core'].iloc[i]}{('(' + rungs['iso_tag'].iloc[i] + ')') if rungs['iso_tag'].iloc[i] else ''}"
              f".(H2O){int(rungs['n'].iloc[i])}" for i in order]
    rung_segs = ([_segment_set(rungs["segments"].iloc[i]) for i in order] if "segments" in rungs.columns
                 else [set() for _ in order])
    mz = pd.to_numeric(merged["mz"], errors="coerce").to_numpy()
    j = np.clip(np.searchsorted(ref, mz), 1, max(len(ref) - 1, 1))
    cand = np.stack([j - 1, np.minimum(j, len(ref) - 1)], axis=1) if len(ref) > 1 else np.zeros((len(mz), 2), int)
    rung_of = [None] * len(mz)
    rung_k = [None] * len(mz)
    for i, m in enumerate(mz):
        if not np.isfinite(m):
            continue
        for k in cand[i]:
            if abs(ref[k] - m) <= m * tol_ppm * 1e-6:
                rung_of[i], rung_k[i] = labels[k], k
                break
    hit = np.array([r is not None for r in rung_of])
    if jitter is not None and segment_of is not None and hit.any():
        seg_map = {str(k): int(v) for k, v in pd.Series(segment_of).items()}
        files = _reading_files(merged, np.nonzero(hit)[0], jitter)
        spared = []
        for i in np.nonzero(hit)[0]:
            fseg = {seg_map[s] for s in files.get(i, ()) if s in seg_map}
            rseg = rung_segs[rung_k[i]]
            if fseg and rseg and not (fseg & rseg):
                hit[i] = False
                spared.append((i, rung_of[i], rseg, fseg))
        if spared:
            from peaky.assignment.cleanup import _note
            merged = merged.copy()
            for i, lab, rseg, fseg in spared:
                _note(merged, merged.index[i],
                      f"on the water rung {lab} of segment(s) {'|'.join(map(str, sorted(rseg)))}; its reading's "
                      f"files are in segment(s) {'|'.join(map(str, sorted(fseg)))}, where that rung did not pass: "
                      f"kept")
            log(f"[reagent-water] {len(spared)} merged row(s) on a passing rung kept: their reading's files lie "
                f"only in segments where the rung did not pass: "
                + ", ".join(f"{_txt(merged['neutral_formula'].iloc[i])} {_txt(merged['adduct'].iloc[i])} ({lab})"
                            for i, lab, _, _ in spared[:4]) + (" ..." if len(spared) > 4 else ""))
    stripped = merged.loc[hit].copy()
    stripped["rung"] = [r for r, h in zip(rung_of, hit) if h]
    kept = merged.loc[~hit].reset_index(drop=True)
    if len(stripped):
        log(f"[reagent-water] {len(stripped)} merged row(s) sit on a passing water rung and leave the "
            f"merged ledger: " + ", ".join(f"{a} {b} = {r}" for a, b, r in
                                           zip(stripped["neutral_formula"].astype(str).head(4),
                                               stripped["adduct"].astype(str).head(4),
                                               stripped["rung"].head(4))) + (" ..." if len(stripped) > 4 else ""))
    return kept, stripped


def stamp_rows(rungs: pd.DataFrame) -> pd.DataFrame:
    """The passing rungs as reagent rows for the batch stamp (`timeseries.stamping_frame`
    aux rows): m/z = the observed median, the isotopologue tag as `iso_label` so the
    79Br and 81Br rungs of one formula stay two tracks."""
    cols = ["mz", "role", "ion_formula", "iso_label", "neutral_formula", "adduct"]
    if rungs is None or not len(rungs):
        return pd.DataFrame(columns=cols)
    mz = pd.to_numeric(rungs["mz_obs"], errors="coerce").fillna(pd.to_numeric(rungs["mz"]))
    return pd.DataFrame({"mz": mz.to_numpy(), "role": "reagent", "ion_formula": rungs["ion_formula"].to_numpy(),
                         "iso_label": [t if t else None for t in rungs["iso_tag"]],
                         "neutral_formula": None, "adduct": None}, columns=cols)


def table(rungs: pd.DataFrame, stripped: pd.DataFrame) -> pd.DataFrame:
    """tables/reagent_water.csv: the passing rungs, each with the merged readings it
    displaced (`neutral adduct`, '; '-joined)."""
    if rungs is None or not len(rungs):
        return pd.DataFrame(columns=list(TABLE_COLUMNS))
    t = rungs.copy()
    lab = [f"{c}{('(' + g + ')') if g else ''}.(H2O){int(n)}" for c, g, n in zip(t["core"], t["iso_tag"], t["n"])]
    disp: dict[str, list] = {}
    if stripped is not None and len(stripped):
        for r, nf, ad in zip(stripped["rung"], stripped["neutral_formula"].astype(str), stripped["adduct"].astype(str)):
            disp.setdefault(r, []).append(f"{nf} {ad}")
    t["displaced"] = ["; ".join(disp.get(x, [])) for x in lab]
    return t[list(TABLE_COLUMNS)]


def rung_test(resolution) -> dict | None:
    """The TOF rung test as batch_summary records it -- its constants and the width
    model's FWHM at m/z 200 and 600 -- or None when the fixed-offset test runs (an
    Orbitrap-class width model, or none)."""
    fwhm = tof_fwhm(resolution)
    if fwhm is None:
        return None
    return {"test": "tof", "fwhm_at_200": round(float(fwhm(200.0)), 5), "fwhm_at_600": round(float(fwhm(600.0)), 5),
            "decoy_fwhm": list(TOF_DECOY_FWHM), "decoy_clear_fwhm": TOF_DECOY_CLEAR_FWHM,
            "decoy_x": DECOY_X, "min_presence": MIN_PRESENCE,
            "link_presence": TOF_LINK_SHARE * MIN_PRESENCE, "covary_r": TOF_COVARY_R,
            "covary_span": TOF_COVARY_SPAN, "covary_min_spectra": TOF_COVARY_MIN_SPECTRA,
            "m1_carbons": TOF_M1_CARBONS, "m1_blend_fwhm": TOF_M1_BLEND_FWHM, "strip_by_segment": True}


def measure(ts: pd.DataFrame | None, profile, *, tol_ppm: float, log=print, resolution=None) -> dict:
    """The batch's ladder for a reagent profile: {rungs, n_cores, segment_sizes,
    tol_ppm, rung_test, segment_of}. Empty rungs when the profile declares no water cores
    or there is no time series. `resolution` is the batch's width model: TOF-class runs
    the TOF rung test (`rung_test` records it; `segment_of` = {spectrum: segment}, which
    makes `strip_rung_rows` strip by segment), anything else the fixed-offset one
    (`rung_test` and `segment_of` None: the strip runs over the whole batch)."""
    wc = tuple(getattr(profile, "water_cores", None) or ())
    pol = getattr(profile, "polarity", "-") or "-"
    out = {"rungs": detect(None, (), tol_ppm=tol_ppm), "n_cores": len(cores(wc, polarity=pol)), "tol_ppm": float(tol_ppm),
           "segment_sizes": {}, "rung_test": None, "segment_of": None}
    if not wc or ts is None or not len(ts):
        return out
    seg = segments(ts)
    out["segment_sizes"] = {str(k): int(v) for k, v in seg.value_counts().sort_index().items()}
    out["rung_test"] = rung_test(resolution)
    if out["rung_test"]:
        out["segment_of"] = seg
    out["rungs"] = detect(ts, wc, tol_ppm=tol_ppm, polarity=pol, resolution=resolution)
    r = out["rungs"]
    log(f"[reagent-water] {len(r)} passing rung(s) over {out['n_cores']} core(s) in "
        f"{len(out['segment_sizes'])} segment(s) {out['segment_sizes']} at +-{tol_ppm:.2f} ppm"
        + (f" (TOF rung test: decoys at {min(TOF_DECOY_FWHM):g}-{max(TOF_DECOY_FWHM):g} FWHM, gaps "
           f"allowed, M+1 carbon test, strip by segment)" if out["rung_test"] else "")
        + (": " + ", ".join(f"{c}{('(' + g + ')') if g else ''} n<={int(g_['n'].max())}"
                            for (c, g), g_ in r.groupby(["core", "iso_tag"], sort=False)) if len(r) else ""))
    return out


def summary(rungs: pd.DataFrame, stripped: pd.DataFrame, *, n_cores: int, tol_ppm: float,
            segment_sizes: dict, rung_test: dict | None = None) -> dict:
    """batch_summary['merge_gates']['reagent_water']; `rung_test` (the TOF test's
    record) is added only when that test ran."""
    by_core: dict[str, list] = {}
    if rungs is not None and len(rungs):
        for c, g, n in zip(rungs["core"], rungs["iso_tag"], rungs["n"]):
            by_core.setdefault(f"{c}-" + (f" ({g})" if g else ""), []).append(int(n))
    out = {"n_cores": int(n_cores), "tol_ppm": float(tol_ppm), "segments": segment_sizes,
           "n_rungs": int(len(rungs)) if rungs is not None else 0,
           "rungs_by_core": by_core,
           "n_stripped": int(len(stripped)) if stripped is not None else 0,
           "stripped": ([f"{a} {b} ({r})" for a, b, r in zip(stripped["neutral_formula"].astype(str),
                                                           stripped["adduct"].astype(str), stripped["rung"])]
                        if stripped is not None and len(stripped) else [])}
    if rung_test:
        out["rung_test"] = dict(rung_test)
    return out
