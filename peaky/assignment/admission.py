"""Admission: which peaks are ELIGIBLE for formula search -- persistence OR brightness.

Every pass that hunts for new formulas (pass-1 grid enumeration, the ladder /
siloxane / residual explainers) draws its candidate peaks through one gate. It
used to be brightness alone (`height >= cfg.height_cutoff`), and on a
low-sensitivity TOF that threw away almost all of the real chemistry: measured on
a 230-spectrum mixed-reagent TOF batch, an absolute 100 cps cutoff kept 32 of
4025 m/z bins, while ~500 bins recur in >= 80 % of the spectra at a median 3-4 cps.
Noise does not recur at a fixed m/z across hundreds of spectra; ions do. Probing
that batch at the exact masses of 21 highly oxygenated molecules an Orbitrap saw
on the same air hit 21/21 (present in a median 93 % of spectra, ~2 cps) while
mass-shifted decoys hit ~0 -- so the recurrent weak population is real, and it
sits 50x below the old cutoff.

Occurrence and brightness do not trade off (no bright bin is transient), so the
persistence path is purely ADDITIVE: it admits peaks for consideration and does
not lower the bar for confirming them. At 2-3 cps the isotopologues are
sub-count, so an occurrence-admitted peak normally lands as a Candidate -- that
is correct, and the isotope rules are untouched.

  occurrence(peak) = fraction of the batch's spectra holding a peak within
                     +-tol of that peak's m/z (one peak per spectrum counts
                     once; tol = `sampling.BATCH_TOL_PPM`, the one tolerance
                     the selection, this table and the merge share)
  admitted         = height >= cfg.height_cutoff  OR  occurrence >= threshold
                     (threshold = `resolve_threshold`: the number given as
                     `cfg.occurrence_min`, or the batch's Otsu split for "auto";
                     stored on `cfg.occurrence_threshold`)
  admitted_by      = 'height' | 'occurrence' (persistence only) | '' (not
                     eligible at the gated sites below)

ONE RULE, NOT A BINNER. The table is per PEAK (`batch.traces.PeakIndex.
occurrence`, one O(n) sweep over the m/z-sorted batch), and a ledger peak is
looked up by exactly the rule the table was built with -- distinct spectra within
+-tol of ITS m/z -- so construction and lookup cannot disagree. The first version
binned the batch by single-linkage gap clustering and looked a peak up within
tol of its bin's centre: on a TOF 91 % of the "persistent" bins were wider than
the tolerance (median 51 peaks, one 134 ppm), so much of their persistence was
the chaining, and a peak that WAS binned could fail to find its own bin (3149
rows denied the path at one operating point). Making the lookup match the span
handed a chained bin's persistence to adjacent noise; capping the bin width split
one ion's jitter cloud across bins (four of six recurrent test ions dropped
below threshold). Per-peak windows have neither failure: a noise peak 12 ppm
from a recurrent ion sees only itself, and a recurrent ion's own peaks each see
the cloud around them.

THE THRESHOLD is derived from the batch, not a constant: the occurrence
distribution is cleanly bimodal (transient below 0.1, persistent above 0.9, a
thin middle) on every instrument measured, and Otsu's split of it lands at
0.40-0.55 on all of them. The split is taken over TRACES, not peaks: a peak's
occurrence is weighted by 1 / (its distinct-spectrum count), so an ion present in
90 % of 500 spectra (450 peaks) and a noise spike seen twice each contribute one
unit -- the unweighted per-peak histogram is dominated by persistent ions and
splits ~0.1 too high. (Measured: the weighted split reproduces the former
bin-table split to +-0.02 on four Orbitrap modes.) A fixed 0.8 was tried first
and discarded two-thirds of the recurring weak ions the path exists to recover;
so `occurrence_min="auto"` = the weighted Otsu split clamped to
[AUTO_MIN, AUTO_MAX]; a number overrides it; `0` disables the path. Fewer than
MIN_SPECTRA spectra give too coarse an occurrence to trust, and the path is off.

THE BRIGHTNESS FLOOR can be derived from the same table
(`derive_height_cutoff_x_edge`). The height gate is a multiple of each sample's
noise edge, and the right multiple is a property of the PEAK PICKER: one that
stops at the noise edge wants 1.0 (rare real ions sit at 1-3x it), one that picks
INTO the noise admits almost everything it found at 1.0. Two statistics of the
batch read that behaviour off the data, and the floor is raised only when BOTH
say so:

  * the TRANSIENT SHARE of the peaks a floor of x edges would admit (the table
    labels every batch peak persistent or transient). Measured on five batches
    of two instruments: three clean Orbitrap modes 0.08-0.16 at 1x; a TOF 0.34 at
    1x, 0.17 at 5x, 0.13 at 8x; an Orbitrap mode with the reagent ion in range
    0.50 at 1x (the reagent ion's noise skirt plus a scan-edge pile-up), 0.14 at
    2x. The rule takes the smallest multiple on a fixed grid whose admitted
    population is at most MAX_TRANSIENT_SHARE transient.
  * the PICKER TAIL: the share of the batch's peaks that sit below
    PICKER_TAIL_X (0.75) x their own sample's 1st-percentile edge. A picker with
    a hard threshold leaves essentially nothing under it -- at most 1 in 10 000
    peaks on every Orbitrap mode, the reagent-in-range one included -- while a
    TOF picker that keeps picking into the noise leaves a thin soft tail (31 in
    10 000). Only a share above PICKER_INTO_NOISE_FRACTION qualifies a batch for
    a raised floor. (A proportion, not an order statistic: the interior quantiles
    do NOT separate the instruments -- p1/p5 is 0.83-0.85 on all five batches --
    and extreme order statistics move with the number of peaks per spectrum.)

Why both: on the reagent-in-range Orbitrap mode the transient share alone would
have raised the floor to 2x, and that removed 45 scan-edge pile-up rows and 12
reagent-skirt rows -- but also 14 multi-file Assigned ions at 1-2x the edge (a
C6H12O [M+H]+ seen in 9 of 27 files among them): there the "noise edge" is the
picker's relative threshold under a huge reagent ion, not noise, and the transient
peaks are an m/z-localised artefact, not picker noise. The tail ratio tells the two
apart. Result: every Orbitrap mode keeps 1.0 exactly, the TOF lands at 5x --
inside the recall plateau of a live sweep (17 of 18 findable target ions from 5x
up) though below its 8-12x precision optimum, which a site pins in its profile --
and the persistence path keeps the recurring weak ions the raised floor drops.

WHICH PASSES ARE GATED -- and how to re-derive it. `admissible()` has four call
sites, but one of them is `passes/directors._target_peaks`, the shared
target-peak helper of FIVE pass functions, so the gate reaches EIGHT pass-level
sites. Count the callers of `_target_peaks`, never the calls to `admissible`:

  via `_target_peaks`   the pass-1 grid enumeration (`run_pass1`), pass-2 series
                        growth (`run_pass2`), pass-3 contaminant families
                        (`run_pass3`) and pass-3's two cluster resolvers
                        (`_resolve_hx_clusters`, `_resolve_acid_i2_clusters`)
  direct                the pass-6 ladder gap-fill (ladders), residual stage B
                        (residual), the siloxane ladder (siloxane -- both the
                        work set and the seed test)

`admitted_by` describes eligibility at THOSE sites. Still brightness-only
(`height >= cfg.height_cutoff`), a documented follow-up and not a promise this
module makes: residual stage A (the ~2-Da isotope-doublet scan), the reflist
rescue, pass-3's series DETECTION statistics (`series_detect`, which counts a
series' members rather than proposing formulas for a peak) and the
isotope-satellite tests in `passes/postprocess`.

Single-sample runs have no batch context: without an occurrence table the gate
is brightness alone, unchanged.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from peaky.batch import sampling as SS

__version__ = "0.5.1"  # per-PEAK occurrence (traces.PeakIndex): one rule for the
                       # table and the lookup; trace-weighted Otsu; the brightness
                       # floor derived from the batch (derive_height_cutoff_x_edge)

DEFAULT_OCCURRENCE_MIN = "auto"   # Otsu split of the batch's occurrence distribution
AUTO_MIN, AUTO_MAX = 0.25, 0.75   # clamp for the derived threshold
MIN_SPECTRA = 10                  # below this, occurrence is too coarse -> path off
ADMIT_HEIGHT = "height"
ADMIT_OCCURRENCE = "occurrence"
ADMIT_NONE = ""

# The batch-derived brightness floor (see the module note, THE BRIGHTNESS FLOOR):
# the smallest multiple on this grid at which the peaks a height gate of that
# many edges would admit are at most MAX_TRANSIENT_SHARE transient (occurrence
# below the persistence threshold). The grid is the sweep that was run live; a
# continuous optimum would chase noise in the share estimate.
X_EDGE_GRID = (1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0)
MAX_TRANSIENT_SHARE = 0.20
# ... and only for a picker that PICKS INTO THE NOISE: more than
# PICKER_INTO_NOISE_FRACTION of the batch's peaks sit below PICKER_TAIL_X x their
# sample's 1st-percentile edge. A hard-threshold picker leaves essentially nothing
# under its edge (<= 0.0001 on every Orbitrap mode measured); a TOF's soft tail
# reads 0.0031. The constant sits at the geometric middle of that gap.
PICKER_TAIL_X = 0.75
PICKER_INTO_NOISE_FRACTION = 0.0005


def bin_occurrence(ts_peaks: pd.DataFrame, *, tol_ppm: float = SS.BATCH_TOL_PPM,
                   sample_col: str = "sample_item_id", mz_col: str = "mz",
                   height_col: str = "height", index=None) -> pd.DataFrame:
    """The batch's per-PEAK occurrence table, one row per physical peak sorted by
    m/z: `mz`, `sample` (integer code), `occurrence` (fraction of spectra with a
    peak within +-tol of it), `n_samples` (that count), `height` and `h_edge`
    (height as a multiple of its own sample's noise edge). Built on
    `traces.PeakIndex` at `tol_ppm`, which defaults to `sampling.BATCH_TOL_PPM`
    -- the same tolerance the selection and the merge use, so `batch` and
    `assign --ts-batch` see one occurrence for one peak. `.attrs` carries
    `tol_ppm` (required by `lookup_occurrence`), the batch's `n_samples`, the
    trace-weighted `auto_threshold`, and `kind='per-peak'`. Pass an existing
    `index` to reuse one built by the caller (the same tolerance is required).

    The name is historical: the rows are peaks, not bins. Every consumer only ever
    read `mz` / `occurrence` / the attrs, which are unchanged."""
    from peaky.batch import traces as TR

    tol = float(tol_ppm)
    cols = [c for c in (sample_col, mz_col, height_col) if c in ts_peaks.columns]
    if len(cols) < 3:
        raise ValueError(f"bin_occurrence needs {sample_col!r}, {mz_col!r}, {height_col!r} "
                         f"columns (got {list(ts_peaks.columns)[:8]})")
    if index is None:
        index = TR.PeakIndex(ts_peaks, tol_ppm=tol, mz_col=mz_col,
                             sample_col=sample_col, height_col=height_col)
    elif float(index.tol_ppm) != tol:
        raise ValueError(f"the PeakIndex was built at {index.tol_ppm} ppm, not {tol} ppm")
    if len(index) == 0 or index.n_samples == 0:
        out = pd.DataFrame({"mz": pd.Series(dtype=float), "sample": pd.Series(dtype=np.int64),
                            "occurrence": pd.Series(dtype=float),
                            "n_samples": pd.Series(dtype=int),
                            "height": pd.Series(dtype=float), "h_edge": pd.Series(dtype=float)})
        out.attrs.update(tol_ppm=tol, n_samples=int(index.n_samples), kind="per-peak",
                         auto_threshold=None)
        return out
    occ = index.occurrence()
    out = pd.DataFrame({"mz": index.mz, "sample": index.sample, "occurrence": occ,
                        "n_samples": np.rint(occ * index.n_samples).astype(int),
                        "height": index.height, "h_edge": index.height_in_edges()})
    out.attrs.update(tol_ppm=tol, n_samples=int(index.n_samples), kind="per-peak",
                     auto_threshold=auto_threshold(occ, weights=_trace_weights(out)))
    return out


def _trace_weights(table: pd.DataFrame) -> np.ndarray | None:
    """Per-row weights that make each TRACE count once in the occurrence
    distribution: 1 / (the row's distinct-spectrum count). None for a table
    without counts (each row then counts once, the pre-per-peak behaviour)."""
    if table is None or "n_samples" not in table.columns:
        return None
    n = pd.to_numeric(table["n_samples"], errors="coerce").to_numpy(dtype=float)
    return 1.0 / np.maximum(np.nan_to_num(n, nan=1.0), 1.0)


def otsu_threshold(x, *, weights=None, n_bins: int = 50) -> float | None:
    """Otsu's split of a distribution on [0, 1]: the cut that maximises the
    between-class variance of the two sides. `weights` (same length as `x`)
    weight the histogram -- 1 / count makes the split a split between TRACES
    rather than between peaks. None for fewer than 2 finite values."""
    x = np.asarray(x, dtype=float)
    w = None if weights is None else np.asarray(weights, dtype=float)
    ok = np.isfinite(x) if w is None else np.isfinite(x) & np.isfinite(w)
    x = x[ok]
    if w is not None:
        w = w[ok]
    if x.size < 2:
        return None
    h, e = np.histogram(x, bins=n_bins, range=(0.0, 1.0), weights=w)
    tot = h.sum()
    if not tot > 0:
        return None
    wn = h / tot
    c = (e[:-1] + e[1:]) / 2
    var = np.full(n_bins + 1, -1.0)
    for i in range(1, n_bins):
        w0, w1 = wn[:i].sum(), wn[i:].sum()
        if w0 <= 0 or w1 <= 0:
            continue
        m0 = (wn[:i] * c[:i]).sum() / w0
        m1 = (wn[i:] * c[i:]).sum() / w1
        var[i] = w0 * w1 * (m0 - m1) ** 2
    best = var.max()
    if best < 0:
        return None
    # every cut through an EMPTY stretch between the two modes is equally
    # optimal; take the middle of that stretch rather than its first edge
    opt = np.where(var >= best - 1e-12 * max(best, 1e-300))[0]
    return float(e[int(round((opt[0] + opt[-1]) / 2))])


def auto_threshold(occurrence, *, weights=None) -> float | None:
    """The batch-derived persistence threshold: Otsu's split of the occurrence
    distribution (trace-weighted when `weights` are given), clamped to
    [AUTO_MIN, AUTO_MAX]. None when undefined."""
    t = otsu_threshold(occurrence, weights=weights)
    if t is None:
        return None
    return float(min(max(t, AUTO_MIN), AUTO_MAX))


def resolve_threshold(cfg, table: pd.DataFrame | None) -> float | None:
    """The numeric persistence threshold for this run: `cfg.occurrence_min` when
    it is a number > 0, the table's auto threshold when it is 'auto'; None
    (path off) for 0 / no table / fewer than MIN_SPECTRA spectra."""
    om = getattr(cfg, "occurrence_min", DEFAULT_OCCURRENCE_MIN)
    if table is None or len(table) == 0:
        return None
    if int(table.attrs.get("n_samples", 0) or 0) < MIN_SPECTRA:
        return None
    if isinstance(om, str):
        if om.strip().lower() != "auto":
            raise ValueError(f"occurrence_min must be a number or 'auto', got {om!r}")
        t = table.attrs.get("auto_threshold")
        if t is None:
            t = auto_threshold(table["occurrence"].to_numpy(), weights=_trace_weights(table))
        return float(t) if t is not None else None
    om = float(om)
    return om if om > 0 else None


def why_off(cfg, table: pd.DataFrame | None) -> str:
    """Why `resolve_threshold` gave no threshold, as one human phrase (`''` when
    it gave one). There are FOUR distinct reasons and a log line that says
    "path off" without saying which is unactionable -- a batch that is one
    spectrum short of MIN_SPECTRA and a batch whose occurrences carry no split
    need opposite responses. Every caller that reports the path being off should
    use this so `assign` and `batch` give the same account of the same run."""
    om = getattr(cfg, "occurrence_min", DEFAULT_OCCURRENCE_MIN)
    # the knob first: 0 disables the path whatever the table says
    if not isinstance(om, str) and (om is None or float(om) <= 0):
        return f"occurrence_min={om!r} -- the knob switches it off"
    if table is None or len(table) == 0:
        return ("no batch occurrence table (a single-sample run without "
                "--ts-batch has no batch context)")
    n = int(table.attrs.get("n_samples", 0) or 0)
    if n < MIN_SPECTRA:
        return f"only {n} spectra, fewer than the {MIN_SPECTRA} an occurrence needs to mean anything"
    if isinstance(om, str) and resolve_threshold(cfg, table) is None:
        return (f"{len(table)} peaks over {n} spectra, but their occurrences carry no "
                "split for 'auto' to derive a threshold from")
    return ""


def lookup_occurrence(mz, table: pd.DataFrame | None) -> np.ndarray:
    """Occurrence at each probe m/z, by the SAME rule the table was built with:
    the fraction of spectra holding a peak within the table's own tolerance
    (`table.attrs['tol_ppm']`, stamped by `bin_occurrence`) of the probe, one
    peak per spectrum. NaN where no peak lies within tolerance (nothing there),
    or without a table. A table without the tolerance is refused rather than
    guessed at.

    A table that carries pre-aggregated rows instead of peaks (no `sample`
    column -- a hand-built or legacy table) is read the only way it can be: the
    nearest row within tolerance supplies its stored occurrence."""
    mz = np.asarray(mz, dtype=float)
    out = np.full(mz.shape, np.nan)
    if table is None or len(table) == 0:
        return out
    if table.attrs.get("tol_ppm") is None:
        raise ValueError("occurrence table lacks attrs['tol_ppm'] (build it with "
                         "admission.bin_occurrence, which stamps the binning tolerance)")
    tol = float(table.attrs["tol_ppm"])
    ok = np.isfinite(mz)
    if not ok.any():
        return out
    if "sample" in table.columns:
        # per-peak table: distinct spectra within +-tol of the probe -- exactly
        # how every row's own occurrence was computed
        from peaky.batch import traces as TR
        n_spec = int(table.attrs.get("n_samples", 0) or 0)
        bm = table["mz"].to_numpy(dtype=float)
        smp = table["sample"].to_numpy(dtype=np.int64)
        order = np.argsort(bm, kind="mergesort")
        bm, smp = bm[order], smp[order]
        q = mz[ok]
        hw = TR._halfwin(q, tol)
        lo = np.searchsorted(bm, q - hw, side="left")
        hi = np.searchsorted(bm, q + hw, side="right")
        vals = np.full(q.shape, np.nan)
        if n_spec > 0:
            for k in np.flatnonzero(hi > lo):
                vals[k] = np.unique(smp[lo[k]:hi[k]]).size / n_spec
        out[ok] = vals
        return out
    bm = table["mz"].to_numpy(dtype=float)
    occ = table["occurrence"].to_numpy(dtype=float)
    order = np.argsort(bm)
    bs, os_ = bm[order], occ[order]
    if len(bs) == 1:
        pick = np.zeros(int(ok.sum()), dtype=int)
    else:
        j = np.clip(np.searchsorted(bs, mz[ok]), 1, len(bs) - 1)
        left, right = j - 1, j
        pick = np.where(np.abs(bs[left] - mz[ok]) <= np.abs(bs[right] - mz[ok]), left, right)
    near = np.abs(bs[pick] - mz[ok]) / mz[ok] * 1e6 <= tol
    out[ok] = np.where(near, os_[pick], np.nan)
    return out


def persistent_trace_count(table: pd.DataFrame | None, threshold: float | None) -> int:
    """How many distinct persistent IONS the table holds: the trace-weighted count
    of rows at or above `threshold` (each row weighs 1 / its spectrum count, so
    an ion present in k spectra contributes k x 1/k = 1). 0 without a threshold."""
    if table is None or threshold is None or not len(table):
        return 0
    occ = pd.to_numeric(table["occurrence"], errors="coerce").to_numpy(dtype=float)
    w = _trace_weights(table)
    if w is None:
        return int(np.nansum(occ >= threshold))
    return int(round(float(np.nansum(np.where(occ >= threshold, w, 0.0)))))


def picker_tail_fraction(table: pd.DataFrame | None, *, x: float = PICKER_TAIL_X) -> float:
    """The share of the batch's peaks sitting below `x` times their own sample's
    1st-percentile noise edge (`h_edge` is height / that edge). ~0 for a picker
    with a hard threshold (nothing is picked under it), a small but non-zero share
    for one that picks into the noise. A proportion over ALL peaks, so it does not
    move with the number of peaks per spectrum. NaN without a per-peak table."""
    if table is None or not len(table) or "h_edge" not in table.columns:
        return float("nan")
    he = pd.to_numeric(table["h_edge"], errors="coerce").to_numpy(dtype=float)
    he = he[np.isfinite(he)]
    if not len(he):
        return float("nan")
    return float(np.mean(he < x))


def derive_height_cutoff_x_edge(table: pd.DataFrame | None, threshold: float | None,
                                *, grid=X_EDGE_GRID,
                                max_transient_share: float = MAX_TRANSIENT_SHARE,
                                picker_fraction_min: float = PICKER_INTO_NOISE_FRACTION) -> dict | None:
    """The brightness floor this batch's own peaks argue for, as a multiple of
    the noise edge (module note, THE BRIGHTNESS FLOOR): the smallest `x` on
    `grid` such that, of the batch peaks with height >= x edges, at most
    `max_transient_share` are transient (occurrence below `threshold`) -- but
    only for a picker that picks into the noise (`picker_tail_fraction` above
    `picker_fraction_min`); a hard-threshold picker keeps 1.0 whatever its
    transient share (the reagent-in-range Orbitrap lesson: its transient peaks
    are a localised artefact, and a raised floor cut real multi-file ions).
    Needs a per-peak table with `h_edge` and a resolved threshold; None
    otherwise (no batch, path off) -- the caller then keeps the package default.

    Returns {x_edge, transient_share (grid value -> share), share_at_1,
    share_at_x, max_transient_share, picker_tail_fraction, picker_tail_x,
    picker_fraction_min, picker_into_noise, n_peaks, bound} where `bound` is
    True when even the largest grid value left more than the allowed share (the
    floor is then that value and the batch is worth a look)."""
    if table is None or threshold is None or not len(table) or "h_edge" not in table.columns:
        return None
    occ = pd.to_numeric(table["occurrence"], errors="coerce").to_numpy(dtype=float)
    he = pd.to_numeric(table["h_edge"], errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(occ) & np.isfinite(he)
    if not ok.any():
        return None
    occ, he = occ[ok], he[ok]
    transient = occ < float(threshold)
    shares: dict = {}
    chosen, chosen_share, bound = None, None, False
    for x in grid:
        m = he >= float(x)
        if not m.any():
            break
        s = float(transient[m].mean())
        shares[float(x)] = round(s, 4)
        if chosen is None and s <= max_transient_share:
            chosen, chosen_share = float(x), s
    if not shares:
        return None
    if chosen is None:
        chosen = max(shares)
        chosen_share = shares[chosen]
        bound = True
    frac = picker_tail_fraction(table)
    into_noise = bool(np.isfinite(frac) and frac > picker_fraction_min)
    if not into_noise:
        # a hard-threshold picker: the transient share is not picker noise, and
        # raising the floor would cut real ions at 1-3x the edge
        chosen, chosen_share, bound = float(grid[0]), shares[float(grid[0])], False
    return {"x_edge": chosen, "transient_share": shares,
            "share_at_1": shares.get(1.0), "share_at_x": round(float(chosen_share), 4),
            "max_transient_share": float(max_transient_share),
            "picker_tail_fraction": None if not np.isfinite(frac) else round(float(frac), 5),
            "picker_tail_x": float(PICKER_TAIL_X),
            "picker_fraction_min": float(picker_fraction_min),
            "picker_into_noise": into_noise,
            "n_peaks": int(ok.sum()), "bound": bound}


def admissible(ledger: pd.DataFrame, cfg) -> pd.Series:
    """Boolean mask: `height >= cfg.height_cutoff` OR `occurrence >= threshold`.

    The threshold is whatever `stamp_admission` RESOLVED for this run
    (`cfg.occurrence_threshold`), including its decision to switch the path off
    (`None`) -- that is what makes `admitted_by == ''` mean "not admissible".
    The raw `cfg.occurrence_min` knob is consulted only on a cfg that was never
    stamped (`occurrence_resolved` False), where there is no resolved decision
    to honour; otherwise a batch that resolved OFF (too few spectra, or an
    occurrence distribution with no split) would be silently re-admitted at
    every gated site by the knob the resolver had already rejected.

    The persistence path needs an `occurrence` column (stamped by
    `stamp_admission`) and a threshold; otherwise this is exactly the brightness
    gate."""
    hcut = float(getattr(cfg, "height_cutoff", 0.0) or 0.0)
    h = ledger["height"].fillna(0).astype(float) >= hcut if "height" in ledger.columns \
        else pd.Series(False, index=ledger.index)
    thr = getattr(cfg, "occurrence_threshold", None)
    if thr is None and not getattr(cfg, "occurrence_resolved", False):
        om = getattr(cfg, "occurrence_min", 0.0)   # unstamped cfg: the knob still counts
        thr = float(om) if not isinstance(om, str) and om and float(om) > 0 else None
    if thr is not None and "occurrence" in ledger.columns:
        occ = pd.to_numeric(ledger["occurrence"], errors="coerce").fillna(-1.0)
        h = h | (occ >= float(thr))
    return h


def stamp_admission(ledger: pd.DataFrame, cfg, table: pd.DataFrame | None = None) -> dict:
    """Annotate the ledger IN PLACE with `occurrence` (the fraction of the batch's
    spectra holding a peak within tolerance of the peak's m/z, NaN without batch
    context) and `admitted_by` ('height' | 'occurrence' | ''), using the cfg's
    RESOLVED `height_cutoff` (call after the noise edge is stamped) and the
    persistence threshold resolved from `cfg.occurrence_min` + the table (stored
    on `cfg.occurrence_threshold`, with `cfg.occurrence_resolved = True` to mark
    the decision final -- `admissible` then honours it, threshold or no
    threshold, so `admitted_by == ''` is exactly "not admissible"). Returns the
    counts {height, occurrence, rejected, n_peaks, occurrence_min,
    occurrence_threshold, height_gate_cps} (the last = the resolved brightness
    gate in cps; `n_peaks` = rows of the batch table, 0 without one)."""
    ledger["occurrence"] = lookup_occurrence(ledger["mz"].to_numpy(dtype=float), table) \
        if "mz" in ledger.columns else np.nan
    hcut = float(getattr(cfg, "height_cutoff", 0.0) or 0.0)
    thr = resolve_threshold(cfg, table)
    try:
        cfg.occurrence_threshold = thr
        cfg.occurrence_resolved = True   # `thr is None` now means OFF, not "unstamped"
    except AttributeError:
        pass
    by_h = ledger["height"].fillna(0).astype(float) >= hcut
    by_o = (pd.to_numeric(ledger["occurrence"], errors="coerce").fillna(-1.0) >= thr) \
        if thr is not None else pd.Series(False, index=ledger.index)
    ledger["admitted_by"] = np.where(by_h, ADMIT_HEIGHT, np.where(by_o, ADMIT_OCCURRENCE, ADMIT_NONE))
    return {"height": int(by_h.sum()), "occurrence": int((~by_h & by_o).sum()),
            "rejected": int((~by_h & ~by_o).sum()),
            "n_peaks": int(len(table)) if table is not None else 0,
            "occurrence_min": getattr(cfg, "occurrence_min", None),
            "occurrence_threshold": thr, "height_gate_cps": hcut}
