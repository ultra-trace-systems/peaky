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

THE RESIDUAL STAGE (2026-09-13, `residual_universe` + `select_residual_cover`).
The cover stops on marginal gain, so a tail of universe bins is never in any
assigned file -- on a 6154-sample field campaign 18 % of 4702 bins, almost all
of it the noise-edge tail (median campaign maximum ~2x the picker edge), but 36
of those bins reach 1-3.4 kcps somewhere, and nothing in that tail can ever
enter the ledger because no assigned file contains it. After the cover's merge
and stamp, a second, TARGETED selection runs over the bins that are (a) absent
from every assigned file, (b) unexplained by the whole-batch stamp (so an
isotope satellite of an assigned ion is not re-targeted) and (c) bright
SOMEWHERE: at least `RESIDUAL_MIN_X_EDGE` (5) x that sample's own noise edge (an
absolute cps floor can override). Each such bin must be assigned where it is
bright, so a sample counts for a bin only where the bin stands at
>= `RESIDUAL_FRAC_OF_MAX` (50 %) of its own campaign maximum (and above that
sample's admission gate); the greedy cover over that relation stops at
`RESIDUAL_K_MAX` (10) picks or when the next sample adds no bin. Measured on the
campaign above: the 36 bright bins needed 21 samples at 50 % of their maximum,
and assigning them gave 6 Assigned + 10 Candidate + 8 isotope satellites of
already-assigned parents + 1 sidelobe + 11 unexplained -- the bright end of the
residual, at a cost bounded by `RESIDUAL_K_MAX` files.

Pure pandas/numpy; no network. The selected `sample_item_id`s feed `assign.run`
one at a time (in pick order), then the ledgers are merged (assign_batch.py).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__version__ = "0.3.0"  # + the targeted residual stage (residual_universe / select_residual_cover)

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

# The residual stage (module note). RESIDUAL_DEFAULT is the `residual` switch of
# assign_batch.run / pipeline.run_batch (`--residual` / `--no-residual`): ON, so a
# batch run targets the bright end of what the cover left behind; off reproduces
# the cover-only run exactly (no stage column, no residual block, no extra file).
RESIDUAL_DEFAULT = True
RESIDUAL_MIN_X_EDGE = 5.0    # a residual bin must reach this x its sample's noise edge somewhere
RESIDUAL_FRAC_OF_MAX = 0.5   # a sample counts for a bin only at >= this share of the bin's maximum
RESIDUAL_K_MAX = 10          # budget of the residual stage (files)
EDGE_Q = 0.01                # the noise edge: the 1st percentile of a sample's picked heights
                             # (the same statistic as passes.noise_edge / traces.PeakIndex.sample_edge)

ROLE_COVER = "cover"   # a greedy pick (adds bins_new uncovered bins)
ROLE_PAD = "pad"       # richest-TIC pad up to k_min after the greedy exhausted the bins
ROLE_RESIDUAL = "residual"   # a residual-stage pick (adds bins_new residual bins, at >= frac_of_max)
UNGROUPED = "(ungrouped)"   # `group_col` label for a sample whose group value is missing
STOP_GAIN = "gain-floor"
STOP_KMAX = "k_max"
STOP_EXHAUSTED = "exhausted"
STOP_EMPTY = "empty"         # residual stage: no bin to target


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


def _prevalence_gate(prev: np.ndarray, min_prevalence: int) -> np.ndarray:
    """The universe: bins present in >= `min_prevalence` samples. When NO bin
    reaches it -- a 1-sample batch, but also any batch whose samples share no
    m/z at this tolerance -- an empty universe would cover trivially and pick
    nothing, so the gate falls back to every bin present at all. One rule for
    the cover and the residual stage, so they agree on what a universe bin is."""
    prev = np.asarray(prev)
    gate = prev >= min_prevalence
    if not gate.any():
        gate = prev >= 1
    return gate


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
    gate = _prevalence_gate(prev, min_prevalence)
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


# ---------------------------------------------------------------------------
# the residual stage: what the cover left behind, and where to assign it
# ---------------------------------------------------------------------------
def residual_universe(peaks: pd.DataFrame, *, assigned, stamped=None,
                      min_x_edge: float | None = RESIDUAL_MIN_X_EDGE,
                      min_cps: float | None = None,
                      min_prevalence: int = MIN_PREVALENCE, tol_ppm: float = BATCH_TOL_PPM,
                      sample_col: str = "sample_item_id", mz_col: str = "mz",
                      height_col: str = "height") -> pd.DataFrame:
    """The bins the residual stage targets (module note, THE RESIDUAL STAGE).

    Bins are the cover's own (`timeseries.bin_ids` at `tol_ppm`: the rule
    `build_matrix` pivots on, read off the long table so no dense matrix is
    built); the universe is the cover's own (`_prevalence_gate`). A universe bin
    is RESIDUAL when it is

      (a) absent from every file in `assigned` (the cover's sample ids),
      (b) unexplained by the whole-batch stamp: `stamped`, when given, is a
          boolean per ROW of `peaks` (True = the peak carries an ion stamp, i.e.
          `annotate_peaks`' `ion_formula` is set), and a bin holding any stamped
          peak is explained -- an isotope satellite or reagent line the stamp
          already names is not re-targeted; and
      (c) bright somewhere: its maximum height over the batch, as a multiple of
          THAT sample's own noise edge (the `EDGE_Q` quantile of the sample's
          heights), is >= `min_x_edge`, and/or its maximum in cps is >=
          `min_cps`. Either bound may be None (not applied); the caller passes
          an absolute `min_cps` to override the edge-relative rule.

    Returns one row per residual bin, ascending m/z: `bin` (the bin number, =
    the column of `build_matrix` for the same table), `bin_mz` (height-weighted
    mean, like `build_matrix`), `prevalence` (samples present), `max_cps`,
    `max_x_edge` (the maximum as a multiple of its sample's edge),
    `sample_at_max` (where it peaks; ties -> the smallest sample id). The frame's
    `.attrs['residual']` records the funnel -- n_universe, n_uncovered,
    n_explained, n_below_floor, n_residual, the floor and the median edge -- and
    `.attrs['edge_cps']` the per-sample edge (a Series by sample id) so the
    caller can size a per-sample admission gate without recomputing it."""
    from peaky.batch import timeseries as TS

    cols = ["bin", "bin_mz", "prevalence", "max_cps", "max_x_edge", "sample_at_max"]
    meta: dict = {"n_universe": 0, "n_uncovered": 0, "n_explained": 0, "n_below_floor": 0,
                  "n_residual": 0, "min_x_edge": min_x_edge, "min_cps": min_cps,
                  "min_prevalence": int(min_prevalence), "tol_ppm": float(tol_ppm),
                  "edge_median_cps": None}

    def _out(rows: pd.DataFrame, edge: pd.Series) -> pd.DataFrame:
        rows.attrs["residual"] = meta
        rows.attrs["edge_cps"] = edge
        return rows

    if peaks is None or not len(peaks) or not is_per_peak(peaks, mz_col=mz_col, height_col=height_col):
        return _out(pd.DataFrame(columns=cols), pd.Series(dtype=float))
    b_all = TS.bin_ids(peaks, tol_ppm=tol_ppm, mz_col=mz_col, height_col=height_col,
                       sample_col=sample_col)
    keep = b_all >= 0
    if not keep.any():
        return _out(pd.DataFrame(columns=cols), pd.Series(dtype=float))
    b = b_all[keep]
    s_ = peaks[sample_col].to_numpy()[keep]
    h = pd.to_numeric(peaks[height_col], errors="coerce").to_numpy(dtype=float)[keep]
    mz = pd.to_numeric(peaks[mz_col], errors="coerce").to_numpy(dtype=float)[keep]
    n_bins = int(b.max()) + 1
    pres = h > 0                                       # presence, as build_matrix's `> 0`
    # prevalence: distinct samples present per bin
    pairs = pd.DataFrame({"b": b[pres], "s": s_[pres]}).drop_duplicates()
    prev = np.bincount(pairs["b"].to_numpy(), minlength=n_bins)
    gate = _prevalence_gate(prev, min_prevalence)
    # (a) covered: present in an assigned file
    in_pick = np.isin(s_, np.asarray(list(assigned or []), dtype=object))
    covered = np.bincount(b[pres & in_pick], minlength=n_bins) > 0
    # (b) explained: any stamped peak in the bin
    if stamped is not None:
        st = np.asarray(stamped, dtype=bool)
        if len(st) != len(b_all):
            raise ValueError(f"stamped has {len(st)} rows, peaks {len(b_all)}")
        explained = np.bincount(b[st[keep]], minlength=n_bins) > 0
    else:
        explained = np.zeros(n_bins, dtype=bool)
    # (c) brightness, as a multiple of the sample's own edge and in cps
    edge = pd.Series(h).groupby(s_).quantile(EDGE_Q)      # by sample id
    e_row = edge.reindex(s_).to_numpy(dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        x = np.where(e_row > 0, h / e_row, np.nan)
    hmax = pd.Series(h).groupby(b).max().reindex(range(n_bins)).to_numpy(dtype=float)
    xmax = pd.Series(x).groupby(b).max().reindex(range(n_bins)).to_numpy(dtype=float)
    ok = np.ones(n_bins, dtype=bool)
    if min_x_edge is not None:
        ok &= np.nan_to_num(xmax, nan=-np.inf) >= float(min_x_edge)
    if min_cps is not None:
        ok &= np.nan_to_num(hmax, nan=-np.inf) >= float(min_cps)
    uncovered = gate & ~covered
    residual = uncovered & ~explained & ok
    meta.update(n_universe=int(gate.sum()), n_uncovered=int(uncovered.sum()),
                n_explained=int((uncovered & explained).sum()),
                n_below_floor=int((uncovered & ~explained & ~ok).sum()),
                n_residual=int(residual.sum()),
                edge_median_cps=(round(float(edge.median()), 3) if len(edge) else None))
    if not residual.any():
        return _out(pd.DataFrame(columns=cols), edge)
    rid = np.flatnonzero(residual)
    wsum = np.bincount(b, weights=mz * h, minlength=n_bins)
    hsum = np.bincount(b, weights=h, minlength=n_bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        bin_mz = np.where(hsum > 0, wsum / hsum, np.nan)
    # where each residual bin peaks: brightest row, ties to the smallest sample id
    sub = np.isin(b, rid)
    at = (pd.DataFrame({"bin": b[sub], "h": h[sub], "s": s_[sub]})
            .sort_values(["bin", "h", "s"], ascending=[True, False, True], kind="mergesort")
            .drop_duplicates("bin").set_index("bin")["s"])
    rows = pd.DataFrame({"bin": rid, "bin_mz": bin_mz[rid], "prevalence": prev[rid],
                         "max_cps": hmax[rid], "max_x_edge": np.round(xmax[rid], 3),
                         "sample_at_max": at.reindex(rid).to_numpy()})
    rows = rows.sort_values("bin_mz", kind="mergesort").reset_index(drop=True)
    return _out(rows, edge)


def select_residual_cover(peaks: pd.DataFrame, bins: pd.DataFrame, *,
                          frac_of_max: float = RESIDUAL_FRAC_OF_MAX,
                          k_max: int = RESIDUAL_K_MAX, min_height=None,
                          tol_ppm: float = BATCH_TOL_PPM, sample_col: str = "sample_item_id",
                          height_col: str = "height", **table_kw) -> pd.DataFrame:
    """Greedy cover of the residual bins by the samples in which they are BRIGHT.

    `bins` is `residual_universe`'s table (its `bin` column). A sample counts
    for a bin only where the bin's height there (summed per sample and bin, as
    `build_matrix` does) is >= `frac_of_max` x the bin's maximum over the batch
    -- the bin is assigned where it stands tall, with its isotopes visible --
    and, when `min_height` (a mapping sample id -> cps: the per-file admission
    gate) is given, at or above that sample's own value (a missing/NaN entry
    binds nothing). Each pick is the sample counting for the most not-yet-
    covered bins; stop at `k_max` picks ('k_max') or when the next sample would
    add no bin ('exhausted'); no bins -> nothing ('empty'). Tie-break as the
    cover's: the lexicographically smallest sample id.

    Returns the same table shape as `select_cover_samples` -- the
    `sample_table` rows in pick order with `pick`, `role` (`ROLE_RESIDUAL`),
    `bins_new` (residual bins this pick covered first, at >= frac_of_max) and
    `coverage` (cumulative fraction of the RESIDUAL universe) -- with
    `.attrs['selection']` = {method, k, n_samples, n_bins, frac_of_max, k_max,
    achieved_coverage, stop_reason, next_gain} and `.attrs['covered_by']` =
    {bin -> the sample id of the pick that first covered it}."""
    from peaky.batch import timeseries as TS

    k_max = max(int(k_max), 1)
    tab = sample_table(peaks, sample_col=sample_col, **table_kw) if peaks is not None and len(peaks) \
        else pd.DataFrame(columns=[sample_col])
    meta: dict = {"method": "residual-cover", "k": 0, "n_samples": int(len(tab)),
                  "n_bins": int(len(bins)) if bins is not None else 0,
                  "frac_of_max": float(frac_of_max), "k_max": k_max,
                  "achieved_coverage": 0.0, "stop_reason": STOP_EMPTY, "next_gain": 0.0}
    if bins is None or not len(bins) or not len(tab):
        out = _empty(tab, meta)
        out.attrs["covered_by"] = {}
        return out
    target = np.asarray(bins["bin"], dtype=np.int64)
    b_all = TS.bin_ids(peaks, tol_ppm=tol_ppm, height_col=height_col, sample_col=sample_col)
    rows = np.isin(b_all, target)
    sub = pd.DataFrame({sample_col: peaks[sample_col].to_numpy()[rows],
                        "_h": pd.to_numeric(peaks[height_col], errors="coerce").to_numpy(dtype=float)[rows],
                        "_bin": b_all[rows]})
    H = (sub.pivot_table(index=sample_col, columns="_bin", values="_h", aggfunc="sum")
            .reindex(columns=target).fillna(0.0))
    samples = np.asarray(H.index)                   # sorted ids (the tie-break order)
    M = H.to_numpy(dtype=float)
    hmax = M.max(axis=0)
    good = (M > 0) & (M >= float(frac_of_max) * hmax[None, :])
    if min_height is not None:
        mh = pd.Series(min_height, dtype=float).reindex(samples).to_numpy(dtype=float)
        bound = np.isfinite(mh)
        good[bound] &= M[bound] >= mh[bound][:, None]
    n_t = len(target)
    covered = np.zeros(n_t, dtype=bool)
    covered_by: dict = {}
    picked: list[int] = []
    gains: list[int] = []
    covs: list[float] = []
    stop, next_gain = STOP_EXHAUSTED, 0
    while len(picked) < len(samples):
        g = (good & ~covered[None, :]).sum(axis=1)
        if picked:
            g[picked] = -1
        j = int(g.argmax())              # first maximum = smallest sample id
        gj = int(g[j])
        if gj <= 0:
            stop, next_gain = STOP_EXHAUSTED, 0
            break
        if len(picked) >= k_max:
            stop, next_gain = STOP_KMAX, gj
            break
        new = good[j] & ~covered
        for t in np.flatnonzero(new):
            covered_by[int(target[t])] = samples[j]
        picked.append(j)
        covered |= good[j]
        gains.append(gj)
        covs.append(float(covered.mean()))
    if not picked:
        meta.update(stop_reason=stop, next_gain=0.0)
        out = _empty(tab, meta)
        out.attrs["covered_by"] = {}
        return out
    sel = tab.set_index(sample_col).loc[samples[picked]].reset_index()
    sel["pick"] = np.arange(1, len(picked) + 1)
    sel["role"] = ROLE_RESIDUAL
    sel["bins_new"] = np.asarray(gains, dtype=int)
    sel["coverage"] = np.round(np.asarray(covs, dtype=float), 4)
    meta.update(k=int(len(picked)), achieved_coverage=round(float(covered.mean()), 4),
                stop_reason=stop, next_gain=round(next_gain / n_t, 4) if n_t else 0.0)
    sel.attrs["selection"] = meta
    sel.attrs["covered_by"] = covered_by
    return sel


def describe_residual(meta: dict | None, universe: dict | None = None) -> str:
    """One log line for a residual selection: the funnel (`residual_universe`'s
    meta) and the cover (`select_residual_cover`'s)."""
    u, m = universe or {}, meta or {}
    floor = []
    if u.get("min_x_edge") is not None:
        floor.append(f"{u['min_x_edge']:g}x the sample's noise edge"
                     + (f" (median edge {u['edge_median_cps']:g} cps)"
                        if u.get("edge_median_cps") is not None else ""))
    if u.get("min_cps") is not None:
        floor.append(f"{u['min_cps']:g} cps")
    s = (f"residual: {u.get('n_uncovered', 0)} of {u.get('n_universe', 0)} universe bins are in "
         f"no assigned file; {u.get('n_explained', 0)} explained by the stamp, "
         f"{u.get('n_below_floor', 0)} below the floor ({' and '.join(floor) or 'none'}), "
         f"{u.get('n_residual', 0)} targeted")
    if m.get("n_bins", 0):
        s += (f"; {m.get('k', 0)} sample(s) at >= {m.get('frac_of_max', 0):.0%} of each bin's "
              f"maximum cover {m.get('achieved_coverage', 0):.0%} of them; stop={m.get('stop_reason')}")
        if m.get("stop_reason") == STOP_KMAX:
            s += f" (next pick would add {m.get('next_gain', 0):.1%})"
    return s
