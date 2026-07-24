"""Representative-sample selection for batch assignment — THE RULE.

A single averaged / single-file spectrum misses peaks that are only present at
certain times: `assign.run` scores ONE sample, so an analyte that spikes during
an event window is never in the candidate list when the chosen file predates the
spike. This bit us on the uronium 24-h run — the 08:20 snapshot sat at hour 11.3,
BEFORE the hour 18-22 event, so the event-only analytes (C10H19NO2, C9H14O2, ...)
were invisible until event files were assigned and merged.

THE RULE (agent-peaky, set 2026-06-19): to get a peak list representative of the
WHOLE batch, assign a small fixed subset and merge it, where the subset is

  * N_TIME (=5) samples evenly spaced across the batch's TIME range — catches
    peaks that appear / disappear over the run; the time endpoints are always
    included, so the start and end of the experiment are covered; PLUS
  * the single MAX-TIC sample — the richest spectrum, the most peaks above the
    detection floor in one shot.

Selecting in TIME (not by row index) matters: an irregularly-sampled run (dense
early, a lone late file, or a gap) must still place a pick on the sparse late
region. The downstream merge (by m/z) of the per-sample assignments is the
representative peak matrix.

Pure pandas/numpy; no network. The selected `sample_item_id`s feed `assign.run`
one at a time (match_compounds is per-sample), then the ledgers are merged.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__version__ = "0.1.0"

N_TIME = 5  # evenly-time-spaced samples per batch (the rule)


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


def select_representative_samples(peaks: pd.DataFrame, *, n_time: int = N_TIME,
                                  include_max_tic: bool = True,
                                  sample_col: str = "sample_item_id",
                                  **table_kw) -> pd.DataFrame:
    """Pick the representative sample subset for assignment (THE RULE).

    Returns the selected rows of `sample_table()`, time-ordered, with an added
    `role` column: 'time-grid', 'max-TIC', or 'time-grid+max-TIC' (when the
    max-TIC sample is also a time-grid pick). Assign each id, then merge by m/z.

    - `n_time` samples are chosen evenly across the batch TIME range: the nearest
      DISTINCT sample to each of `n_time` equally-spaced target times (so the two
      endpoints are always included and dense regions are not over-sampled).
    - The single max-TIC sample is added (union; deduped against the grid).
    - Fewer than `n_time` samples -> all are returned (max-TIC still flagged).
    """
    tab = sample_table(peaks, sample_col=sample_col, **table_kw)
    n = len(tab)
    if n == 0:
        return tab.assign(role=pd.Series(dtype=str))
    has_time = "datetime_utc" in tab.columns
    has_tic = "tic" in tab.columns

    # --- n_time evenly TIME-spaced picks (nearest distinct sample per target) ---
    if not has_time or n <= n_time:
        grid = list(range(n))                       # too few to thin, or no clock
    else:
        t = tab["datetime_utc"].astype("int64").to_numpy()   # ns since epoch
        targets = np.linspace(t[0], t[-1], n_time)
        grid, used = [], set()
        for tg in targets:
            for j in np.argsort(np.abs(t - tg)):     # nearest sample not yet taken
                j = int(j)
                if j not in used:
                    used.add(j)
                    grid.append(j)
                    break
    roles = {i: "time-grid" for i in grid}

    # --- max-TIC (the richest single spectrum) ---
    if include_max_tic and has_tic:
        mi = int(np.asarray(tab["tic"]).argmax())
        roles[mi] = "time-grid+max-TIC" if mi in roles else "max-TIC"

    keep = sorted(roles)
    sel = tab.loc[keep].copy()
    sel["role"] = [roles[i] for i in keep]
    sort_key = "datetime_utc" if has_time else sample_col
    return sel.sort_values(sort_key).reset_index(drop=True)


def select_representative_sample_ids(peaks: pd.DataFrame, *,
                                     sample_col: str = "sample_item_id",
                                     **kw) -> list:
    """Convenience: just the selected `sample_item_id`s (time order)."""
    sel = select_representative_samples(peaks, sample_col=sample_col, **kw)
    return sel[sample_col].tolist()


# ---------------------------------------------------------------------------
# Strategy 2: BRIGHTEST-COVERAGE (the "bin-then-assign" realization).
#
# THE RULE picks samples by TIME; but match_compounds can only commit a peak in a
# sample where that peak's isotope envelope is actually present, i.e. where it is
# bright. On a reagent-CIMS run the max-TIC pick is dominated by the (huge) reagent
# ion, so it is the brightest sample for only a small fraction of ANALYTE peaks
# (measured: 13% on a representative Br- batch) and event-only analytes stay unexplained.
#
# Brightest-coverage instead bins ALL batch peaks by m/z (timeseries.build_matrix),
# and for each significant bin takes the sample where it is BRIGHTEST. Because each
# bin has exactly one brightest (arg-max) sample, the winner sets are DISJOINT, so
# the optimal cover is simply: take winner samples in descending bins-won order
# until `coverage_target` of significant bins is covered (capped at k_max, floored
# at k_min). The selected ids feed the SAME assign.run + merge as THE RULE — only
# WHICH real samples get assigned changes, never the scoring or the merge. It is a
# COVERAGE play (catches more analyte peaks), not a speed play (~k_max assigns).
# ---------------------------------------------------------------------------
def select_brightest_coverage_samples(
        peaks: pd.DataFrame, *, coverage_target: float = 0.85, k_max: int = 10,
        k_min: int = N_TIME + 1, height_floor: float = 1000.0,
        include_time_grid: bool = True, sample_col: str = "sample_item_id",
        **table_kw) -> pd.DataFrame:
    """Pick the winner-sample subset that COVERS the brightest occurrence of as many
    significant m/z bins as possible (brightest-coverage strategy; see module note).

    Returns the selected `sample_table()` rows, time-ordered, with a `role` column
    ('coverage-winner' / 'time-grid' / 'coverage+time-grid') and a `bins_won` int
    (significant bins this sample is the brightest for). Feed `[sample_item_id]`
    straight into `assign_batch.run(sample_ids=...)`.

    - `height_floor` (cps): a bin is "significant" if its max height across samples
      is >= this. Reagent-relative — lower it for a quieter dataset.
    - greedy cover by bins-won until `coverage_target` of significant bins is covered,
      bounded `k_min` <= n <= `k_max`. Padded to `k_min` with the richest (max-TIC)
      remaining samples so a too-high floor / quiet dataset never under-selects.
    - `include_time_grid` unions the two TIME endpoints (cheap insurance the run
      start/end are represented even if they win no bins).
    """
    from peaky.batch import timeseries as TS

    k_min = min(k_min, k_max)            # k_max is the hard cap, even below the default floor
    tab = sample_table(peaks, sample_col=sample_col, **table_kw)
    n = len(tab)
    if n == 0:
        return tab.assign(role=pd.Series(dtype=str), bins_won=pd.Series(dtype=int))
    if n <= k_min:                       # too few samples to be selective: take all
        return tab.assign(role="coverage-winner", bins_won=0)

    mat, _bin_mz = TS.build_matrix(peaks, sample_col=sample_col)   # samples x bins
    maxh = mat.max(axis=0)                                         # per-bin max (skips NaN)
    sig = maxh.index[maxh >= height_floor]                         # significant bins
    win_counts = (mat[sig].idxmax(axis=0).value_counts()          # sample -> #bins won (desc)
                  if len(sig) else pd.Series(dtype=int))
    total = max(int(len(sig)), 1)

    selected: list = []
    bins_won: dict = {}
    covered = 0
    for sid, cnt in win_counts.items():
        if len(selected) >= k_max:
            break
        selected.append(sid); bins_won[sid] = int(cnt); covered += int(cnt)
        if covered / total >= coverage_target and len(selected) >= k_min:
            break
    # pad to k_min with the richest remaining samples (robustness for a high floor)
    if len(selected) < k_min:
        order = (tab.sort_values("tic", ascending=False)[sample_col].tolist()
                 if "tic" in tab.columns else tab[sample_col].tolist())
        for sid in order:
            if sid not in bins_won:
                selected.append(sid); bins_won[sid] = 0
                if len(selected) >= k_min:
                    break

    roles = {sid: "coverage-winner" for sid in selected}
    if include_time_grid and "datetime_utc" in tab.columns and n >= 2:
        for sid in (tab.iloc[0][sample_col], tab.iloc[-1][sample_col]):
            roles[sid] = "coverage+time-grid" if sid in roles else "time-grid"
            bins_won.setdefault(sid, 0)

    sel = tab[tab[sample_col].isin(roles)].copy()
    sel["role"] = sel[sample_col].map(roles)
    sel["bins_won"] = sel[sample_col].map(bins_won).fillna(0).astype(int)
    sort_key = "datetime_utc" if "datetime_utc" in sel.columns else sample_col
    return sel.sort_values(sort_key).reset_index(drop=True)


def select_brightest_coverage_sample_ids(peaks: pd.DataFrame, *,
                                         sample_col: str = "sample_item_id",
                                         **kw) -> list:
    """Convenience: just the brightest-coverage `sample_item_id`s (time order)."""
    sel = select_brightest_coverage_samples(peaks, sample_col=sample_col, **kw)
    return sel[sample_col].tolist()


def select_pooled_union(peaks: pd.DataFrame, *,
                        group_col: str = "sample_batch_name",
                        sample_col: str = "sample_item_id",
                        coverage_target: float = 0.90, k_max: int = 6,
                        height_floor: float = 1000.0, **kw) -> tuple[list, pd.DataFrame]:
    """Per-GROUP brightest-coverage UNION for a pooled multi-batch peak table.

    Pooling many batches (e.g. all per-zone batches of one mode x range) into ONE
    unified ledger needs a sample subset that represents EVERY group. A single naive
    brightest-coverage pass over the pool is biased: the group with the highest
    absolute intensity wins most m/z bins and hogs the winner slots, starving the
    quieter groups (measured on a pooled field campaign: naive pooling gave one
    group only 56% bright-bin coverage). Selecting brightest-coverage WITHIN each
    group and unioning the picks guarantees each group contributes its own analyte-
    representative samples, at the cost of ~`k_max` x n_groups assignments.

    Returns `(union_ids, provenance)` where `union_ids` is the de-duplicated sample
    list (group order, then each group's time order) and `provenance` is the
    concatenated `select_brightest_coverage_samples` table with a `group_col` column.
    """
    if group_col not in peaks.columns:
        raise KeyError(f"group_col {group_col!r} not in peaks columns "
                       f"(got {list(peaks.columns)[:8]})")
    frames, ids = [], []
    for g, dfg in peaks.groupby(group_col, sort=True):
        sel = select_brightest_coverage_samples(
            dfg, coverage_target=coverage_target, k_max=k_max,
            height_floor=height_floor, sample_col=sample_col, **kw)
        sel = sel.assign(**{group_col: g})
        frames.append(sel)
        ids.extend(sel[sample_col].tolist())
    prov = (pd.concat(frames, ignore_index=True) if frames
            else peaks.iloc[:0].assign(role=None, bins_won=0))
    union = list(dict.fromkeys(ids))            # de-dup, preserve first-seen order
    return union, prov
