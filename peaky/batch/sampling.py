"""Which real samples get assigned for a batch: greedy PRESENCE SET-COVER.

`assign.run` scores ONE sample at a time (match_compounds is per-sample; a
synthetic union spectrum cannot be scored), so a whole-batch ledger is the merge
of a small subset of per-sample assignments. The merge has no prevalence filter:
a compound is in the ledger iff some ASSIGNED sample contained it. Selection is
therefore the whole recall story -- an analyte present in samples that were never
assigned is simply missing.

THE RULE (2026-09-12, replacing the time-grid+max-TIC and brightest-arg-max
selectors):

  * Universe = the batch's m/z bins (`timeseries.build_matrix`) that are PRESENT
    in at least `min_prevalence` (2) samples. No height floor anywhere: the
    peak picker's own detection edge varies ~1000x between instruments and
    modes (TOF 0.8 cps vs a reagent-in-range Orbitrap mode 800 cps), so any
    absolute cps floor is a no-op on one mode and blinds another. The prevalence
    gate is the noise filter and has no units.
  * Greedy: each pick is the sample that holds the most NOT-YET-COVERED universe
    bins. Presence cover is submodular, so plain greedy is near-optimal and can
    trade redundancy (two near-identical rich samples are not both taken). A
    time grid / max-TIC add nothing: the clock is uncorrelated with the air and
    the first greedy pick is already the sample carrying the most distinct bins
    (the objective is that COUNT, never brightness -- 'richest' below means
    total ion current and ranks the pads only).
  * Stop on MARGINAL GAIN: once `k_min` (6) samples are taken, stop when the
    next sample would add fewer than `min_gain` (0.5 %) of the universe. `k_max`
    (30) is a wall-clock budget only; a run that hits it is flagged
    (`stop_reason == 'k_max'`) because it means the batch was still gaining.
    A coverage-target stop is broken both ways (trivially met with a floor,
    never met without one), so none exists.

Measured on a pooled 5036-sample field-campaign table (the numbers are
docs/SAMPLING.md section 7; keep the two in step): this lands at k=15 and holds
94 % of the rare (<5 % prevalence) ledger ions of a dedicated sub-batch run, vs
54 % for the old 6-sample time grid and 91 % for the 12-sample arg-max cap
(which never reached its coverage target on any real batch). A single greedy
over a pooled multi-batch table also does NOT starve quiet groups (per-group
coverage 87-93 %), so pooling uses the same selector; per-group achieved
coverage is recorded when `group_col` is given.

Pure pandas/numpy; no network. The selected `sample_item_id`s feed `assign.run`
one at a time (in pick order), then the ledgers are merged (assign_batch.py).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__version__ = "0.2.0"  # presence set-cover replaces representative + brightest

K_MIN = 6              # take at least this many samples before the gain stop applies
K_MAX = 30             # wall-clock budget (a run that hits it is flagged)
MIN_GAIN = 0.005       # stop when the next pick adds < this fraction of the universe
MIN_PREVALENCE = 2     # a bin enters the universe if present in >= this many samples

# The ONE m/z binning tolerance for every batch-level operation: sample selection
# (the presence cover's bins), the admission table (`admission.bin_occurrence`)
# and the merge (`assign_batch.DEFAULT_TOL_PPM` is this constant). All three bin
# identically or the run is incoherent: a bin the selector covered must be the
# bin the merge sees, and a peak's `occurrence` must be read off its own bin.
BATCH_TOL_PPM = 6.0

ROLE_COVER = "cover"   # a greedy pick (adds bins_new uncovered bins)
ROLE_PAD = "pad"       # richest-TIC pad up to k_min after the greedy exhausted the bins
UNGROUPED = "(ungrouped)"   # `group_col` label for a sample whose group value is missing
STOP_GAIN = "gain-floor"
STOP_KMAX = "k_max"
STOP_EXHAUSTED = "exhausted"


def sample_table(peaks: pd.DataFrame, *, sample_col: str = "sample_item_id",
                 time_col: str = "datetime_utc", name_col: str = "sample_item_name",
                 height_col: str = "height", tic_col: str = "tic") -> pd.DataFrame:
    """One row per sample: id, time, name, `tic`, `n_peaks`. Accepts EITHER

      * a per-peak batch table (has `height`) -> tic = sum of peak heights, OR
      * a per-sample table such as `samples.list` (already has a `tic` column,
        one row per sample) -> tic taken directly, no peak load needed.

    Sorted by time when a clock is present. Only `sample_col` is required."""
    if peaks is None or sample_col not in getattr(peaks, "columns", []):
        raise ValueError(f"peaks needs a {sample_col!r} column")
    cols = peaks.columns
    if tic_col in cols and height_col not in cols:
        # already per-sample (e.g. samples.list)
        keep = [c for c in (sample_col, time_col, name_col, tic_col) if c in cols]
        tab = peaks[keep].drop_duplicates(subset=sample_col).reset_index(drop=True)
        tab = tab.rename(columns={tic_col: "tic", time_col: "datetime_utc",
                                  name_col: "sample_item_name"})
    else:
        parts: dict[str, tuple] = {"n_peaks": (sample_col, "size")}
        if height_col in cols:
            parts["tic"] = (height_col, "sum")
        if time_col in cols:
            parts["datetime_utc"] = (time_col, "first")
        if name_col in cols:
            parts["sample_item_name"] = (name_col, "first")
        tab = peaks.groupby(sample_col).agg(**parts).reset_index()
    sort_key = "datetime_utc" if "datetime_utc" in tab.columns else sample_col
    return tab.sort_values(sort_key).reset_index(drop=True)


def is_per_peak(peaks, *, mz_col: str = "mz", height_col: str = "height") -> bool:
    """True when `peaks` is a per-PEAK table (m/z + height per row) -- what the
    cover needs. A per-sample table (`samples.list`) has neither."""
    cols = set(getattr(peaks, "columns", []))
    return mz_col in cols and height_col in cols


def _empty(tab: pd.DataFrame, meta: dict) -> pd.DataFrame:
    out = tab.iloc[:0].assign(pick=pd.Series(dtype=int), role=pd.Series(dtype=str),
                              bins_new=pd.Series(dtype=int),
                              coverage=pd.Series(dtype=float))
    out.attrs["selection"] = meta
    return out


def select_cover_samples(peaks: pd.DataFrame, *, k_min: int = K_MIN, k_max: int = K_MAX,
                         min_gain: float = MIN_GAIN, min_prevalence: int = MIN_PREVALENCE,
                         group_col: str | None = None, tol_ppm: float = BATCH_TOL_PPM,
                         sample_col: str = "sample_item_id", **table_kw) -> pd.DataFrame:
    """Greedy presence set-cover over the batch's m/z bins (THE RULE; module note).

    `peaks` must be the per-PEAK batch table (`sample_item_id`, `mz`, `height`;
    optionally `datetime_utc`, `sample_item_name`). Returns the selected
    `sample_table()` rows in PICK order with

      pick       1-based greedy order
      role       'cover' (a greedy pick) | 'pad' (richest-TIC pad up to k_min)
      bins_new   universe bins this pick covered for the first time (marginal gain)
      coverage   cumulative fraction of the universe covered after this pick
      <group_col>  the sample's group as a string, when `group_col` is given
                   (a missing group value becomes `UNGROUPED`)

    and `.attrs['selection']` = {method, k, n_samples, n_bins, n_bins_total,
    n_bins_gated, min_prevalence, tol_ppm, achieved_coverage, stop_reason,
    next_gain, k_min, k_max, min_gain[, coverage_by_group, picks_by_group]}.
    `tol_ppm` is the m/z binning tolerance (default `BATCH_TOL_PPM`, the merge's
    tolerance too -- pass the same value to both).

    Stop reasons: 'gain-floor' (next pick < min_gain of the universe, k >= k_min),
    'k_max' (budget hit while still gaining -- WARN, see `k_max_warning`),
    'exhausted' (every universe bin covered, or every sample taken).
    Fewer than k_min samples -> all are taken. Deterministic for a given table.
    """
    from peaky.batch import timeseries as TS

    if not is_per_peak(peaks):
        raise ValueError("presence-cover selection needs the per-peak batch table "
                         "(a `mz` and a `height` column per peak); a per-sample "
                         "table cannot be binned")
    k_max = max(int(k_max), 1)
    k_min = max(min(int(k_min), k_max), 1)
    min_prevalence = max(int(min_prevalence), 1)
    tab = sample_table(peaks, sample_col=sample_col, **table_kw)
    n = len(tab)
    meta: dict = {"method": "presence-cover", "k": 0, "n_samples": int(n),
                  "n_bins": 0, "n_bins_total": 0, "n_bins_gated": 0,
                  "min_prevalence": min_prevalence, "tol_ppm": float(tol_ppm),
                  "achieved_coverage": 0.0, "stop_reason": STOP_EXHAUSTED,
                  "next_gain": 0.0, "k_min": k_min, "k_max": k_max,
                  "min_gain": float(min_gain)}
    if n == 0:
        return _empty(tab, meta)

    mat, _bin_mz = TS.build_matrix(peaks, sample_col=sample_col,
                                   tol_ppm=float(tol_ppm))   # samples x bins
    if mat.shape[1] == 0:
        return _empty(tab, meta)
    A_all = (mat > 0).to_numpy()                       # presence; NaN -> False
    samples = np.asarray(mat.index)
    prev = A_all.sum(axis=0)
    gate = prev >= min_prevalence
    if not gate.any():
        # NO bin reaches min_prevalence -- a 1-sample batch, but also any batch
        # whose samples share no m/z at this tolerance. An empty universe would
        # cover trivially and pick nothing, so fall back to every bin.
        gate = prev >= 1
    A = A_all[:, gate]
    n_bins = int(A.shape[1])
    if n_bins == 0:
        # EVERY bin gated out, which the >=1 fallback above leaves possible only
        # when no peak anywhere has a positive height. An empty universe has no
        # coverage to report: `covered.mean()` is NaN (plus numpy warnings), and
        # that NaN would reach `coverage`, `achieved_coverage` and a bare `NaN`
        # token in batch_summary.json. Nothing to cover -> nothing to select,
        # the same answer as the empty matrix above.
        meta.update(n_bins_total=int(A_all.shape[1]), n_bins_gated=int((~gate).sum()))
        return _empty(tab, meta)
    gain_floor = float(min_gain) * n_bins

    # gain_floor is a fraction of the GATED universe (n_bins), not of all bins:
    # the singletons the prevalence gate dropped are not coverable, so counting
    # them would scale the floor by an irrelevant, instrument-dependent number.
    covered = np.zeros(n_bins, dtype=bool)
    picked: list[int] = []
    gains: list[int] = []
    covs: list[float] = []
    stop, next_gain = STOP_EXHAUSTED, 0
    while len(picked) < len(samples):
        g = (A & ~covered[None, :]).sum(axis=1)
        if picked:
            g[picked] = -1
        # TIE-BREAK: argmax returns the FIRST maximum, and build_matrix pivots on
        # the sample id, so equal-gain samples resolve to the lexicographically
        # smallest sample_item_id. Deterministic and input-order independent.
        j = int(g.argmax())
        gj = int(g[j])
        # stop-check precedence: exhausted, then gain-floor, then k_max (a pick
        # that adds nothing is never taken, even below k_min; the budget is the
        # last word only while the batch is still gaining above the floor).
        if gj <= 0:
            stop, next_gain = STOP_EXHAUSTED, 0
            break
        if len(picked) >= k_min and gj < gain_floor:
            stop, next_gain = STOP_GAIN, gj
            break
        if len(picked) >= k_max:
            stop, next_gain = STOP_KMAX, gj
            break
        picked.append(j)
        covered |= A[j]
        gains.append(gj)
        covs.append(float(covered.mean()))
    roles = [ROLE_COVER] * len(picked)

    # pad to k_min with the richest remaining samples (tiny batch, or the bins
    # ran out early): cheap cross-file corroboration, never fewer than k_min.
    if len(picked) < min(k_min, len(samples)):
        tic = (tab.set_index(sample_col)["tic"].reindex(samples).fillna(0.0).to_numpy()
               if "tic" in tab.columns else np.zeros(len(samples)))
        # TIE-BREAK: a stable sort on descending tic, so equal-TIC samples keep
        # the matrix's (sorted-id) row order -- same determinism as the cover.
        for j in np.argsort(-tic, kind="stable"):
            j = int(j)
            if j in picked:
                continue
            picked.append(j); gains.append(0); roles.append(ROLE_PAD)
            covs.append(float(covered.mean()))
            if len(picked) >= min(k_min, len(samples)):
                break

    sel = tab.set_index(sample_col).loc[samples[picked]].reset_index()
    sel["pick"] = np.arange(1, len(picked) + 1)
    sel["role"] = roles
    sel["bins_new"] = np.asarray(gains, dtype=int)
    sel["coverage"] = np.round(np.asarray(covs, dtype=float), 4)
    meta.update(k=int(len(picked)), n_bins=n_bins, n_bins_total=int(A_all.shape[1]),
                n_bins_gated=int((~gate).sum()),
                achieved_coverage=round(float(covered.mean()), 4), stop_reason=stop,
                next_gain=round(next_gain / n_bins, 4) if n_bins else 0.0)

    if group_col is not None:
        if group_col not in peaks.columns:
            raise KeyError(f"group_col {group_col!r} not in peaks columns "
                           f"(got {list(peaks.columns)[:8]})")
        # Normalise the labels to str ONCE, before the set and the sort: a missing
        # group value is a real state (the pool warns about ungrouped peak rows
        # and leaves them in the table it hands us), and on pandas >= 3
        # `astype(str)` no longer turns NaN into "nan", so the labels would stay
        # a str/float mix and `sorted(set(...))` would raise. Ungrouped samples
        # still carry bins and are still selectable; they just group under
        # UNGROUPED, which no per-group report matches.
        gser = peaks.groupby(sample_col)[group_col].first().reindex(samples)
        grp = gser.astype(str).where(gser.notna(), UNGROUPED).to_numpy()
        sel[group_col] = grp[picked]
        cov_by, picks_by = {}, {}
        pick_mask = np.zeros(len(samples), dtype=bool)
        pick_mask[picked] = True
        for gname in sorted(set(grp)):
            rows = grp == gname
            gbins = A[rows].any(axis=0)
            nb = int(gbins.sum())
            cov_by[gname] = round(float((covered & gbins).sum() / nb), 4) if nb else 0.0
            picks_by[gname] = int((rows & pick_mask).sum())
        meta["coverage_by_group"] = cov_by
        meta["picks_by_group"] = picks_by
    sel.attrs["selection"] = meta
    return sel


def select_cover_sample_ids(peaks: pd.DataFrame, *, sample_col: str = "sample_item_id",
                            **kw) -> list:
    """Convenience: just the selected `sample_item_id`s (pick order)."""
    return select_cover_samples(peaks, sample_col=sample_col, **kw)[sample_col].tolist()


def k_max_warning(meta: dict | None) -> str | None:
    """The warning text for a selection that hit its `k_max` budget while the
    batch was still gaining (the next sample would have added `next_gain` of the
    universe); None otherwise."""
    if not meta or meta.get("stop_reason") != STOP_KMAX:
        return None
    return (f"selection hit k_max={meta.get('k_max')} while still gaining "
            f"({meta.get('next_gain', 0) * 100:.2f}% of {meta.get('n_bins')} bins per "
            f"extra sample; achieved coverage {meta.get('achieved_coverage', 0):.1%}) "
            f"-- raise --k-max to cover more of this batch")


def describe(meta: dict | None) -> str:
    """One log line for a selection meta dict."""
    m = meta or {}
    s = (f"presence-cover: {m.get('k', 0)} samples of {m.get('n_samples', '?')} cover "
         f"{m.get('achieved_coverage', 0):.1%} of {m.get('n_bins', 0)} m/z bins "
         f"(present in >={m.get('min_prevalence', MIN_PREVALENCE)} samples; "
         f"{m.get('n_bins_gated', 0)} singleton bins gated out); "
         f"stop={m.get('stop_reason')}")
    if m.get("stop_reason") == STOP_GAIN:
        s += f" (next pick would add {m.get('next_gain', 0):.2%})"
    return s
