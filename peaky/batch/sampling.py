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
`RESIDUAL_K_MAX` (10) picks or when the next sample adds no bin. A candidate that
is the FT ringing sidelobe of a saturating neighbour is not worth a file: the
per-file cleanup would only label it an artifact. Its tier is read the way
`timeseries.flag_sidelobe_channels` reads a channel -- a parent within
SIDELOBE_DMZ (12 mDa) at >= 100x, and the candidate/parent height ratio LOCKED
across >= 20 shared samples (cv < 0.08) -- and a confirmed sidelobe is dropped;
one that only meets cleanup's static rule in the sample where it peaks, with too
few shared samples to test the ratio, is kept as a SUSPECT and covered last (a
clean bin outranks any number of suspects), since a real ion that merely sits
near a bright peak does get assigned. Measured on the
campaign above: the 36 bright bins needed 21 samples at 50 % of their maximum,
and assigning them gave 6 Assigned + 10 Candidate + 8 isotope satellites of
already-assigned parents + 1 sidelobe + 11 unexplained -- the bright end of the
residual, at a cost bounded by `RESIDUAL_K_MAX` files.

THE EPISODE STAGE (`episode_universe` + `select_episode_samples`). A short
bright episode -- a plume over a few spectra -- holds bins the cover counts once:
one file covers them, and the marginal-gain rule never asks for a second. Its
evidence (the 13C line, the adduct partner, a second file agreeing) then rests on
whichever single plume spectrum the cover happened to take. Right after the
cover, on the same bins (`timeseries.bin_ids` at the batch tolerance):

  * EPISODIC bin: present in `EPISODE_MIN_N` (3) <= n <= max(`EPISODE_MAX_N`
    (19), `EPISODE_MAX_FRAC` (1 %) of the samples) samples, with a LIFT >=
    `EPISODE_MIN_LIFT` (5) -- its maximum over the bin's baseline, the median
    over EVERY sample of the bin's height there or, where it is absent, that
    sample's noise edge. An episodic bin is in at most max(19, 1 %) samples, so
    in a batch of more than 38 samples this median is the batch's edge level and
    the lift is the maximum over the typical edge (the bin's own heights where
    present never set it there; they do only in a small batch). This one rule
    stands for the two of the episode spec ("the median where present, or the
    batch baseline where absent elsewhere"): read literally, that gives 235
    episodic bins in 66 episodes on 36 field batches against this rule's 736 in
    140 (a plume recurring at a similar height counts here, not there) -- and
    a peak height >= `EPISODE_BRIGHT_X_EDGE` (3) x that sample's noise edge
    (`EDGE_Q`, as the residual stage reads it).
  * EPISODE: time overlap. Each episodic bin's RUN is the stretch of
    time-ordered samples around its peak sample in which it is present, one
    missing sample allowed; runs that overlap or touch (one gap allowed) are one
    episode. Recorded: start, end, peak sample (the sample holding the most of
    the episode's bins at >= 3x edge, then the largest summed multiple), n_bins,
    max_x_edge.
  * THE COVERAGE RULE (selection): every episodic bin whose peak reaches
    `EPISODE_MIN_X_EDGE` (5) x that sample's noise edge (`--episode-min-x-edge`)
    QUALIFIES and has its own peak spectrum selected; an episode holding one also
    gets its peak sample and up to `EPISODE_NEIGHBOURS` (2) neighbours -- samples
    inside the episode in which >= `EPISODE_NEIGHBOUR_FRAC` (50 %) of its bins
    still stand at >= 3x edge, nearest the peak first. There is NO budget: the
    stage guarantees the coverage of every qualifying bin, not a cost (wall time
    is spent as the batch needs; bins below 5x may be lost). The cover's files are
    free. `k_episode_max` (`K_EPISODE_MAX`, default None) is an optional safety
    cap: the order is then every episode's peak sample, every qualifying bin's own
    peak sample, then the neighbours, episodes by descending max_x_edge, and a cap
    that leaves a wanted file out is flagged as `k_max` is (`stop_reason ==
    'k_max'`, `episode_budget_warning`) -- the rule then does not hold. The picks
    are outside the cover's `k_max` / `min_gain` and are assigned with the cover's
    files (selected_samples.csv role 'episode'); the residual stage that follows
    reads them as assigned.
  * ROW FLAG: in every file an episode wanted that is selected (its own pick or a
    cover file), the per-file ledger rows on an episodic bin carry `episode =
    True` (`episode_row_flags`); per-file only, nothing reads it yet.

Pure pandas/numpy; no network. The selected `sample_item_id`s feed `assign.run`
one at a time (in pick order), then the ledgers are merged (assign_batch.py).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__version__ = "0.4.0"  # + the episode stage (episode_universe / select_episode_samples / episode_row_flags)
                       # 0.3.0: + the targeted residual stage (residual_universe / select_residual_cover)

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

# The episode stage (module note). EPISODE_DEFAULT is the `episodes` switch of
# assign_batch.run / pipeline.run_batch (`--episodes` / `--no-episodes`); off
# reproduces the run without the stage exactly.
EPISODE_DEFAULT = True
K_EPISODE_MAX = None         # optional safety cap on the files the stage adds (None = unlimited)
EPISODE_MIN_N = 3            # an episodic bin is present in at least this many samples ...
EPISODE_MAX_N = 19           # ... and at most max(this, EPISODE_MAX_FRAC x the samples)
EPISODE_MAX_FRAC = 0.01
EPISODE_MIN_LIFT = 5.0       # its maximum over its baseline (module note)
EPISODE_MIN_X_EDGE = 5.0     # THE COVERAGE RULE: an episodic bin peaking at >= this x its sample's
                             # noise edge has its peak spectrum (and its episode's) selected
EPISODE_BRIGHT_X_EDGE = 3.0  # an episodic bin's peak, and a neighbour's bins, at >= this x the edge
EPISODE_NEIGHBOURS = 2       # neighbours per episode beside its peak sample
EPISODE_NEIGHBOUR_FRAC = 0.5  # a neighbour holds >= this share of the episode's bins at >= 3x edge
EPISODE_GAP = 1              # missing samples a run (and two touching runs) may have between them

ROLE_COVER = "cover"   # a greedy pick (adds bins_new uncovered bins)
ROLE_PAD = "pad"       # richest-TIC pad up to k_min after the greedy exhausted the bins
ROLE_RESIDUAL = "residual"   # a residual-stage pick (adds bins_new residual bins, at >= frac_of_max)
ROLE_EPISODE = "episode"     # an episode-stage pick (bins_new = the episode's bins it holds at >= 3x edge)
UNGROUPED = "(ungrouped)"   # `group_col` label for a sample whose group value is missing
STOP_GAIN = "gain-floor"
STOP_KMAX = "k_max"
STOP_EXHAUSTED = "exhausted"
STOP_EMPTY = "empty"         # residual stage: no bin to target
# residual-bin tiers (residual_universe): a clean target, a static-rule sidelobe
# SUSPECT (kept, covered last), a ratio-locked CONFIRMED sidelobe (dropped)
TIER_CLEAN = "clean"
TIER_SUSPECT = "suspect"
TIER_SIDELOBE = "sidelobe"


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

    A bin that passes all three is then TIERED against any saturating neighbour
    (a bin within `timeseries.SIDELOBE_DMZ` that reaches `SIDELOBE_MIN_PARENT`
    somewhere -- its MAXIMUM, not the batch median the channel flagger uses: a
    plume ion saturates only in the samples where its sidelobe appears), the way
    `flag_sidelobe_channels` tiers a channel:

      * `TIER_SIDELOBE` -- over the samples holding both, the neighbour's median
        clears SIDELOBE_MIN_PARENT and >= SIDELOBE_FACTOR x the bin's, there are
        >= SIDELOBE_MIN_PAIRS of them, and the height ratio is locked
        (cv < SIDELOBE_CV): a confirmed ringing sidelobe.
        DROPPED (a file would only label it an artifact); returned in
        `.attrs['sidelobes']`, counted as `n_sidelobe`.
      * `TIER_SUSPECT` -- too few shared samples to test the ratio, but cleanup's
        static rule holds in the sample where the bin peaks (a neighbour there at
        >= SIDELOBE_MIN_PARENT and >= SIDELOBE_FACTOR x the bin). KEPT: a real
        ion near a bright peak does get assigned; `select_residual_cover` covers
        suspects last. Counted as `n_suspect`.
      * `TIER_CLEAN` -- no such neighbour, or the ratio was measured and varies
        (an independent ion; its `ratio_cv` is still recorded).

    Returns one row per residual bin, ascending m/z: `bin` (the bin number, =
    the column of `build_matrix` for the same table), `bin_mz` (height-weighted
    mean, like `build_matrix`), `prevalence` (samples present), `max_cps`,
    `max_x_edge` (the maximum as a multiple of its sample's edge),
    `sample_at_max` (where it peaks; ties -> the smallest sample id), `tier`,
    `sidelobe_of` (the neighbour's m/z that decided the tier, NaN if none),
    `n_pairs` (samples holding both) and `ratio_cv` (NaN when not measurable).
    The frame's `.attrs['residual']` records the funnel -- n_universe,
    n_uncovered, n_explained, n_below_floor, n_sidelobe, n_suspect, n_residual
    (= the rows, clean + suspect), the floor and the median edge --
    `.attrs['sidelobes']` the dropped rows (a list of records, same columns,
    tier 'sidelobe') and `.attrs['edge_cps']` the per-sample edge ({sample id
    -> cps}) so the caller can size a per-sample admission gate without
    recomputing it. Plain containers on purpose: pandas compares `.attrs`
    when it concatenates or rounds a frame, and a Series there raises."""
    from peaky.batch import timeseries as TS

    cols = ["bin", "bin_mz", "prevalence", "max_cps", "max_x_edge", "sample_at_max",
            "tier", "sidelobe_of", "n_pairs", "ratio_cv"]
    meta: dict = {"n_universe": 0, "n_uncovered": 0, "n_explained": 0, "n_below_floor": 0,
                  "n_sidelobe": 0, "n_suspect": 0,
                  "n_residual": 0, "min_x_edge": min_x_edge, "min_cps": min_cps,
                  "min_prevalence": int(min_prevalence), "tol_ppm": float(tol_ppm),
                  "edge_median_cps": None}

    def _out(rows: pd.DataFrame, edge: pd.Series,
             sidelobes: pd.DataFrame | None = None) -> pd.DataFrame:
        rows.attrs["residual"] = meta
        rows.attrs["edge_cps"] = {k: float(v) for k, v in edge.items()}
        rows.attrs["sidelobes"] = (sidelobes.to_dict("records")
                                   if sidelobes is not None else [])
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
    # sidelobe tiers against any saturating neighbour (docstring)
    tier, sl_of, n_pairs, r_cv = _sidelobe_tiers(
        rid, bin_mz, hmax, b, s_, h, at,
        dmz=TS.SIDELOBE_DMZ, factor=TS.SIDELOBE_FACTOR, min_parent=TS.SIDELOBE_MIN_PARENT,
        cv_max=TS.SIDELOBE_CV, min_pairs=TS.SIDELOBE_MIN_PAIRS)
    rows = pd.DataFrame({"bin": rid, "bin_mz": bin_mz[rid], "prevalence": prev[rid],
                         "max_cps": hmax[rid], "max_x_edge": np.round(xmax[rid], 3),
                         "sample_at_max": at.reindex(rid).to_numpy(),
                         "tier": tier, "sidelobe_of": sl_of, "n_pairs": n_pairs,
                         "ratio_cv": r_cv})
    rows = rows.sort_values("bin_mz", kind="mergesort").reset_index(drop=True)
    drop = (rows["tier"] == TIER_SIDELOBE).to_numpy()
    sidelobes = rows[drop].reset_index(drop=True)
    rows = rows[~drop].reset_index(drop=True)
    meta.update(n_sidelobe=int(drop.sum()),
                n_suspect=int((rows["tier"] == TIER_SUSPECT).sum()),
                n_residual=int(len(rows)))
    return _out(rows, edge, sidelobes)


def _sidelobe_tiers(rid, bin_mz, hmax, b, s_, h, at_max, *, dmz: float, factor: float,
                    min_parent: float, cv_max: float, min_pairs: int):
    """Tier each candidate bin in `rid` against the bins within `dmz` that reach
    `min_parent` somewhere (`hmax`, per bin) (module note / `residual_universe`).
    Returns four arrays aligned with `rid`: tier (TIER_*), the deciding
    neighbour's m/z (NaN if none), the number of samples holding both, and the
    candidate/neighbour height-ratio cv (NaN when not measurable). Per neighbour
    the outcome is: ratio measured over >= `min_pairs` shared samples in which
    the neighbour's median clears `min_parent` and >= `factor` x the candidate's
    -> 'sidelobe' if locked (cv < `cv_max`) else 'independent'; otherwise
    cleanup's static rule in the sample where the candidate peaks (the neighbour
    there >= `min_parent` and >= `factor` x the candidate) -> 'suspect'; else
    nothing. The bin's tier is SIDELOBE if any neighbour confirms one, else
    SUSPECT if any neighbour is a static suspect, else CLEAN (a measured
    independent neighbour is still recorded, for the reader)."""
    n = len(rid)
    tier = np.array([TIER_CLEAN] * n, dtype=object)
    parent = np.full(n, np.nan)
    npairs = np.zeros(n, dtype=int)
    rcv = np.full(n, np.nan)
    if n == 0:
        return tier, parent, npairs, rcv
    bright = np.flatnonzero(np.nan_to_num(hmax, nan=0.0) >= min_parent)
    if not len(bright):
        return tier, parent, npairs, rcv
    pmz = bin_mz[bright]                                 # ascending (bins are)
    cands: dict[int, list[int]] = {}
    for k, c in enumerate(rid):
        lo, hi = np.searchsorted(pmz, [bin_mz[c] - dmz, bin_mz[c] + dmz])
        # every saturating bin within dmz except the candidate's own: bins are
        # distinct by the gap rule, and a ringing sidelobe at m/z 100 sits only
        # ~1-1.5 mDa from its parent (the channel flagger's 1.5 mDa exclusion
        # skips a merged ION's own track, which has no analogue here)
        ps = [int(bright[j]) for j in range(lo, hi) if int(bright[j]) != int(c)]
        if ps:
            cands[k] = ps
    if not cands:
        return tier, parent, npairs, rcv
    need = sorted({int(rid[k]) for k in cands} | {p for ps in cands.values() for p in ps})
    sub = np.isin(b, need)
    H = (pd.DataFrame({"s": s_[sub], "b": b[sub], "h": h[sub]})
           .pivot_table(index="s", columns="b", values="h", aggfunc="sum"))
    for k, ps in cands.items():
        c = int(rid[k])
        if c not in H.columns:
            continue
        hc = H[c].fillna(0.0)
        s_max = at_max.get(c)
        verdicts = []                       # (rank, parent_mz, n_pairs, cv); rank 2 sidelobe, 1 suspect, 0 independent
        for p in ps:
            if p not in H.columns:
                continue
            hp = H[p].fillna(0.0)
            both = (hc > 0) & (hp > 0)
            npair = int(both.sum())
            if (npair >= min_pairs and float(hp[both].median()) >= min_parent
                    and float(hp[both].median()) >= factor * float(hc[both].median())):
                ratio = (hc[both] / hp[both]).replace([np.inf, -np.inf], np.nan).dropna()
                if len(ratio) >= min_pairs and ratio.mean() > 0:
                    cv = float(ratio.std() / ratio.mean())
                    verdicts.append((2 if cv < cv_max else 0, float(bin_mz[p]), npair, cv))
                    continue
            if s_max is not None and s_max in H.index:
                hp_s, hc_s = float(hp.get(s_max, 0.0)), float(hc.get(s_max, 0.0))
                if hp_s >= min_parent and hc_s > 0 and hp_s >= factor * hc_s:
                    verdicts.append((1, float(bin_mz[p]), npair, np.nan))
        if not verdicts:
            continue
        rank, pm, npair, cv = max(verdicts, key=lambda v: (v[0], -abs(v[1] - bin_mz[c])))
        tier[k] = {2: TIER_SIDELOBE, 1: TIER_SUSPECT, 0: TIER_CLEAN}[rank]
        parent[k], npairs[k], rcv[k] = pm, npair, cv
    return tier, parent, npairs, rcv


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
    covered bins, CLEAN bins first: a `tier` column marks the sidelobe
    SUSPECTS (`residual_universe`), and the gain is lexicographic -- one clean
    bin outranks any number of suspects -- so budget goes to suspects only once
    no sample adds a clean bin. Stop at `k_max` picks ('k_max') or when the
    next sample would add no bin ('exhausted'); no bins -> nothing ('empty').
    Tie-break as the cover's: the lexicographically smallest sample id.

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
    # lexicographic weights: a clean bin is worth more than every suspect together
    susp = ((bins["tier"].astype(str) == TIER_SUSPECT).to_numpy()
            if "tier" in bins.columns else np.zeros(n_t, dtype=bool))
    weight = np.where(susp, 1, n_t + 1)
    covered = np.zeros(n_t, dtype=bool)
    covered_by: dict = {}
    picked: list[int] = []
    gains: list[int] = []
    covs: list[float] = []
    stop, next_gain = STOP_EXHAUSTED, 0
    while len(picked) < len(samples):
        fresh = good & ~covered[None, :]
        g = (fresh * weight[None, :]).sum(axis=1)
        if picked:
            g[picked] = -1
        j = int(g.argmax())              # first maximum = smallest sample id
        gj = int(fresh[j].sum())         # bins this pick would add (any tier)
        if g[j] <= 0:
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
         f"{u.get('n_sidelobe', 0)} dropped as confirmed sidelobes of a saturating neighbour, "
         f"{u.get('n_residual', 0)} targeted"
         + (f" ({u['n_suspect']} static-rule sidelobe suspect(s), covered last)"
            if u.get("n_suspect") else ""))
    if m.get("n_bins", 0):
        s += (f"; {m.get('k', 0)} sample(s) at >= {m.get('frac_of_max', 0):.0%} of each bin's "
              f"maximum cover {m.get('achieved_coverage', 0):.0%} of them; stop={m.get('stop_reason')}")
        if m.get("stop_reason") == STOP_KMAX:
            s += f" (next pick would add {m.get('next_gain', 0):.1%})"
    return s


# ---------------------------------------------------------------------------
# the episode stage: short bright episodes get their peak file and a neighbour
# ---------------------------------------------------------------------------
EPISODE_BIN_COLS = ["bin", "bin_mz", "n_present", "max_cps", "max_x_edge", "lift", "baseline_cps",
                    "peak_sample", "run_start", "run_end", "episode", "qualifies"]
EPISODE_TABLE_COLS = ["episode", "start", "end", "start_time", "end_time", "n_spectra", "peak_sample",
                      "n_qualifying", "bin_peaks",
                      "n_bins", "max_x_edge", "bins_mz", "wanted", "files", "picked", "over_budget"]


def _runs(positions: np.ndarray, gap: int) -> list[tuple[int, int]]:
    """Contiguous runs of sorted sample positions, `gap` missing samples allowed."""
    if not len(positions):
        return []
    cut = np.flatnonzero(np.diff(positions) > gap + 1)
    starts = np.r_[0, cut + 1]
    ends = np.r_[cut, len(positions) - 1]
    return [(int(positions[a]), int(positions[e])) for a, e in zip(starts, ends)]


def episode_universe(peaks: pd.DataFrame, *, tol_ppm: float = BATCH_TOL_PPM,
                     min_n: int = EPISODE_MIN_N, max_n: int = EPISODE_MAX_N,
                     max_frac: float = EPISODE_MAX_FRAC, min_lift: float = EPISODE_MIN_LIFT,
                     bright_x_edge: float = EPISODE_BRIGHT_X_EDGE,
                     min_x_edge: float = EPISODE_MIN_X_EDGE, gap: int = EPISODE_GAP,
                     n_neighbours: int = EPISODE_NEIGHBOURS,
                     neighbour_frac: float = EPISODE_NEIGHBOUR_FRAC,
                     sample_col: str = "sample_item_id", mz_col: str = "mz",
                     height_col: str = "height", peak_col: str = "peak_id",
                     **table_kw) -> pd.DataFrame:
    """The episodic bins and the episodes they form (module note, THE EPISODE STAGE).

    Bins are the cover's (`timeseries.bin_ids` at `tol_ppm`, read off the long
    table); a sample's noise edge is the residual stage's (`EDGE_Q` of its
    heights); time order is `sample_table`'s (by `datetime_utc`, else by id).
    A bin is EPISODIC when it is present (height > 0) in `min_n` <= n <=
    max(`max_n`, floor(`max_frac` x samples)) samples, its maximum height is
    >= `min_lift` x its baseline (the median over every sample of its height
    there, or that sample's edge where it is absent) and that maximum is
    >= `bright_x_edge` x the edge of the sample holding it (ties: the earliest).
    It QUALIFIES (the coverage rule) when that maximum is >= `min_x_edge` x
    the edge.

    Returns one row per episodic bin (EPISODE_BIN_COLS, ascending m/z): its
    presence count, maximum (cps and x edge), lift and baseline, the sample at
    its maximum, its run (first / last time position of the stretch of samples
    around that maximum where it is present, `gap` missing samples allowed) and
    the episode it belongs to. `.attrs`:

      episodes        list of {episode, start, end (sample ids), start_time,
                      end_time, n_spectra, peak_sample, n_bins, max_x_edge,
                      bins (bin numbers), bins_mz, wanted} in time order. Runs
                      that overlap or touch (`gap` allowed) are one episode;
                      the peak sample holds the most of its bins at
                      >= `bright_x_edge` x edge (then the largest summed multiple,
                      then the earliest); `wanted` = the peak sample, then up to
                      `n_neighbours` samples inside the episode holding
                      >= `neighbour_frac` of its bins at >= `bright_x_edge` x edge,
                      nearest the peak first (then more bins, then brighter);
                      `n_bright` = {sample id -> its bins at >= `bright_x_edge` x
                      edge} over the episode's samples.
      episode_meta    the funnel: n_samples, n_max, n_bins_in_range (presence
                      within [min_n, n_max]), n_episodic, n_episodes and the
                      thresholds.
      episodic_peaks  {sample id -> [peak ids on an episodic bin]} (when the
                      table has `peak_col`), for the row flag.
      episodic_mz     {sample id -> [bin m/z of the episodic bins present]}.
    Plain containers on purpose (pandas compares `.attrs`)."""
    from peaky.batch import timeseries as TS

    meta: dict = {"n_samples": 0, "n_max": int(max_n), "n_bins_in_range": 0, "n_episodic": 0,
                  "n_episodes": 0, "min_n": int(min_n), "max_n": int(max_n),
                  "max_frac": float(max_frac), "min_lift": float(min_lift),
                  "bright_x_edge": float(bright_x_edge), "min_x_edge": float(min_x_edge),
                  "n_qualifying": 0, "gap": int(gap),
                  "n_neighbours": int(n_neighbours), "neighbour_frac": float(neighbour_frac),
                  "tol_ppm": float(tol_ppm)}

    def _out(rows: pd.DataFrame, episodes=(), ep_peaks=None, ep_mz=None) -> pd.DataFrame:
        rows.attrs["episodes"] = list(episodes)
        rows.attrs["episode_meta"] = meta
        rows.attrs["episodic_peaks"] = dict(ep_peaks or {})
        rows.attrs["episodic_mz"] = dict(ep_mz or {})
        return rows

    if peaks is None or not len(peaks) or not is_per_peak(peaks, mz_col=mz_col, height_col=height_col):
        return _out(pd.DataFrame(columns=EPISODE_BIN_COLS))
    tab = sample_table(peaks, sample_col=sample_col, height_col=height_col, **table_kw)
    order = tab[sample_col].to_numpy()
    n_s = len(order)
    pos_of = {s: i for i, s in enumerate(order)}
    n_max = max(int(max_n), int(np.floor(float(max_frac) * n_s)))
    meta.update(n_samples=int(n_s), n_max=int(n_max))
    b_all = TS.bin_ids(peaks, tol_ppm=tol_ppm, mz_col=mz_col, height_col=height_col,
                       sample_col=sample_col)
    keep = b_all >= 0
    if not keep.any():
        return _out(pd.DataFrame(columns=EPISODE_BIN_COLS))
    b = b_all[keep]
    s_ = peaks[sample_col].to_numpy()[keep]
    h = pd.to_numeric(peaks[height_col], errors="coerce").to_numpy(dtype=float)[keep]
    mz = pd.to_numeric(peaks[mz_col], errors="coerce").to_numpy(dtype=float)[keep]
    edge = pd.Series(h).groupby(s_).quantile(EDGE_Q)                   # by sample id
    e_pos = edge.reindex(order).to_numpy(dtype=float)
    e_pos = np.where(np.isfinite(e_pos), e_pos, np.nanmedian(e_pos) if np.isfinite(e_pos).any() else 0.0)
    pres = h > 0
    long = (pd.DataFrame({"b": b[pres], "s": s_[pres], "h": h[pres]})
              .groupby(["b", "s"], sort=True)["h"].sum().reset_index())  # as build_matrix sums
    n_present = long.groupby("b").size()
    in_range = n_present.index[(n_present >= int(min_n)) & (n_present <= n_max)].to_numpy()
    meta["n_bins_in_range"] = int(len(in_range))
    if not len(in_range):
        return _out(pd.DataFrame(columns=EPISODE_BIN_COLS))
    sub = long[long["b"].isin(in_range)].copy()
    sub["pos"] = sub["s"].map(pos_of).astype(int)
    sub["x"] = sub["h"].to_numpy() / e_pos[sub["pos"].to_numpy()]
    rec = []
    for bin_, g in sub.groupby("b", sort=True):
        p = g["pos"].to_numpy()
        hh = g["h"].to_numpy()
        v = e_pos.copy()
        v[p] = hh
        base = float(np.median(v))
        k = int(np.lexsort((p, -hh))[0])               # the maximum; ties -> the earliest sample
        hmax, xmax = float(hh[k]), float(g["x"].to_numpy()[k])
        lift = hmax / base if base > 0 else np.inf
        if not (lift >= float(min_lift) and xmax >= float(bright_x_edge)):
            continue
        ps = np.sort(p)
        run = next(r for r in _runs(ps, int(gap)) if r[0] <= p[k] <= r[1])
        rec.append(dict(bin=int(bin_), n_present=int(len(p)), max_cps=hmax, max_x_edge=round(xmax, 3),
                        lift=round(float(lift), 3), baseline_cps=round(base, 3),
                        peak_sample=order[p[k]], peak_pos=int(p[k]), run_start=run[0], run_end=run[1]))
    meta["n_episodic"] = len(rec)
    if not rec:
        return _out(pd.DataFrame(columns=EPISODE_BIN_COLS))
    rows = pd.DataFrame(rec)
    wsum = np.bincount(b, weights=mz * h, minlength=int(b.max()) + 1)
    hsum = np.bincount(b, weights=h, minlength=int(b.max()) + 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        rows["bin_mz"] = (wsum / hsum)[rows["bin"].to_numpy()]
    # episodes: overlapping / touching runs, swept in time order
    rows = rows.sort_values(["run_start", "run_end", "bin"], kind="mergesort").reset_index(drop=True)
    ep = np.zeros(len(rows), dtype=int)
    cur, cur_end = 1, int(rows.at[0, "run_end"])
    for i in range(1, len(rows)):
        if int(rows.at[i, "run_start"]) <= cur_end + int(gap) + 1:
            cur_end = max(cur_end, int(rows.at[i, "run_end"]))
        else:
            cur += 1
            cur_end = int(rows.at[i, "run_end"])
        ep[i] = cur
    ep[0] = 1
    rows["episode"] = ep
    xs = sub.set_index(["b", "pos"])["x"]
    episodes = []
    for e, g in rows.groupby("episode", sort=True):
        lo, hi = int(g["run_start"].min()), int(g["run_end"].max())
        nb = len(g)
        win = np.arange(lo, hi + 1)
        X = np.zeros((nb, len(win)))
        for r, bin_ in enumerate(g["bin"].to_numpy()):
            xb = xs.loc[int(bin_)]
            sel = xb.index.to_numpy()
            ok = (sel >= lo) & (sel <= hi)
            X[r, sel[ok] - lo] = xb.to_numpy()[ok]
        bright = X >= float(bright_x_edge)
        cnt = bright.sum(axis=0)
        sumx = np.where(bright, X, 0.0).sum(axis=0)
        pk = int(np.lexsort((win, -sumx, -cnt))[0])     # most bins bright, then brightest, then earliest
        cand = [j for j in range(len(win)) if j != pk and cnt[j] > 0
                and cnt[j] >= float(neighbour_frac) * nb]
        cand.sort(key=lambda j: (abs(j - pk), -cnt[j], -sumx[j], j))
        neigh = [str(order[win[j]]) for j in cand[:max(int(n_neighbours), 0)]]
        peak_s = str(order[win[pk]])
        # the coverage rule: the bins of this episode peaking at >= min_x_edge x edge
        # each want their own peak spectrum (brightest first), beside the episode's
        q = g[g["max_x_edge"].astype(float) >= float(min_x_edge)].sort_values(
            ["max_x_edge", "peak_pos"], ascending=[False, True], kind="mergesort")
        bin_peaks = []
        for x in q["peak_sample"].astype(str):
            if x != peak_s and x not in bin_peaks:
                bin_peaks.append(x)
        if len(q):
            wanted = [peak_s] + bin_peaks + [x for x in neigh if x not in bin_peaks]
        else:
            wanted, bin_peaks, neigh = [], [], []
        t0 = tab["datetime_utc"].iloc[lo] if "datetime_utc" in tab.columns else None
        t1 = tab["datetime_utc"].iloc[hi] if "datetime_utc" in tab.columns else None
        episodes.append({"episode": int(e), "start": order[lo], "end": order[hi],
                         "start_time": None if t0 is None or pd.isna(t0) else str(t0),
                         "end_time": None if t1 is None or pd.isna(t1) else str(t1),
                         "n_spectra": int(hi - lo + 1), "peak_sample": order[win[pk]],
                         "n_bins": int(nb), "max_x_edge": round(float(g["max_x_edge"].max()), 3),
                         "bins": [int(x) for x in g["bin"]],
                         "bins_mz": [round(float(x), 5) for x in sorted(g["bin_mz"])],
                         "n_qualifying": int(len(q)),
                         "bin_peaks": bin_peaks,
                         "neighbours": list(neigh),
                         "wanted": [str(x) for x in wanted],
                         "n_bright": {str(order[win[j]]): int(cnt[j]) for j in range(len(win)) if cnt[j]}})
    meta["n_episodes"] = len(episodes)
    rows["qualifies"] = rows["max_x_edge"].astype(float) >= float(min_x_edge)
    meta["n_qualifying"] = int(rows["qualifies"].sum())
    epi_bins = set(rows["bin"].tolist())
    inb = np.isin(b, list(epi_bins)) & pres
    ep_mz: dict = {}
    bmz = dict(zip(rows["bin"], rows["bin_mz"]))
    for s, bb in pd.DataFrame({"s": s_[inb], "b": b[inb]}).drop_duplicates().groupby("s")["b"]:
        ep_mz[str(s)] = sorted(round(float(bmz[int(x)]), 6) for x in bb)
    ep_peaks: dict = {}
    if peak_col in peaks.columns:
        pid = peaks[peak_col].to_numpy()[keep]
        for s, ids in pd.DataFrame({"s": s_[inb], "p": pid[inb]}).groupby("s")["p"]:
            ep_peaks[str(s)] = sorted(str(x) for x in ids if pd.notna(x) and str(x) != "")
    rows = rows.drop(columns=["peak_pos"]).sort_values("bin_mz", kind="mergesort").reset_index(drop=True)
    return _out(rows[EPISODE_BIN_COLS], episodes, ep_peaks, ep_mz)


def select_episode_samples(peaks: pd.DataFrame, universe: pd.DataFrame, *, assigned,
                           k_max: int | None = K_EPISODE_MAX, sample_col: str = "sample_item_id",
                           **table_kw) -> pd.DataFrame:
    """The episode stage's picks (module note, THE COVERAGE RULE). `universe` is
    `episode_universe`'s frame; `assigned` the cover's sample ids (a wanted file
    already selected is free). Every episode holding a QUALIFYING bin (peak >=
    the universe's `min_x_edge` x edge) wants its peak sample, the peak sample of
    each qualifying bin, and its neighbours; an episode with none wants nothing.
    All are taken (no budget) unless `k_max` (an optional safety cap on the NEW
    files) binds; then the order is every episode's peak sample, then every
    qualifying bin's own peak sample, then the neighbours, episodes by descending
    max_x_edge; a wanted file the cap leaves out is `over_budget` and the stage
    stops on 'k_max' (`episode_budget_warning`). No episode -> 'empty'; all
    served -> 'exhausted'.

    Returns the `sample_table` rows of the new files in pick order with `pick`
    (1-based within the stage), `role` (ROLE_EPISODE), `bins_new` (the
    episode's bins the file holds at >= the bright x edge), `coverage` (the
    fraction of the qualifying episodes with at least one selected file after
    this pick) and `episode` (the episode it was taken for);
    `.attrs['selection']` = {method 'episode', min_x_edge, n_qualifying_bins, k,
    k_max, n_episodes, n_episodes_qualifying, n_wanted, n_free, n_over_budget,
    stop_reason, sample_ids, episodes: the universe's records with `files`
    (wanted files in the selection, free or picked), `picked` (taken by this
    stage for it) and `over_budget`} and `.attrs['flag_files']` = every file an
    episode wanted that is selected (the files whose episodic rows are flagged)."""
    cap = None if k_max is None else max(int(k_max), 0)
    tab = (sample_table(peaks, sample_col=sample_col, **table_kw) if peaks is not None and len(peaks)
           else pd.DataFrame(columns=[sample_col]))
    eps = [dict(e) for e in (universe.attrs.get("episodes") or [])] if universe is not None else []
    emeta = dict(universe.attrs.get("episode_meta") or {}) if universe is not None else {}
    meta: dict = {"method": "episode", "min_x_edge": emeta.get("min_x_edge", EPISODE_MIN_X_EDGE),
                  "n_qualifying_bins": int(emeta.get("n_qualifying", 0)), "k": 0, "k_max": cap,
                  "n_episodes": len(eps),
                  "n_episodes_qualifying": sum(1 for e in eps if e.get("wanted")),
                  "n_episodic_bins": int(emeta.get("n_episodic", 0)), "n_wanted": 0, "n_free": 0,
                  "n_over_budget": 0, "stop_reason": STOP_EMPTY, "sample_ids": [], "universe": emeta}
    for e in eps:
        e.update(files=[], picked=[], over_budget=[])
    if not any(e.get("wanted") for e in eps):
        meta["episodes"] = eps
        out = _empty(tab, meta)
        out.attrs["flag_files"] = []
        return out
    order = sorted(range(len(eps)), key=lambda i: (-float(eps[i]["max_x_edge"]), i))
    rounds = [[(i, eps[i]["wanted"][0]) for i in order if eps[i].get("wanted")],
              [(i, s) for i in order for s in eps[i].get("bin_peaks", [])],
              [(i, s) for i in order for s in eps[i].get("neighbours", [])
               if s not in eps[i].get("bin_peaks", [])]]
    have = {str(x) for x in (assigned or [])}
    picked: list = []
    pick_ep: list = []
    for rnd in rounds:
        for i, s in rnd:
            e = eps[i]
            meta["n_wanted"] += 1
            if s in have:
                meta["n_free"] += 1
                e["files"].append(s)
            elif cap is None or len(picked) < cap:
                have.add(s)
                picked.append(s)
                pick_ep.append(e["episode"])
                e["files"].append(s)
                e["picked"].append(s)
            else:
                e["over_budget"].append(s)
                meta["n_over_budget"] += 1
    stop = STOP_KMAX if meta["n_over_budget"] else STOP_EXHAUSTED
    flag_files = sorted({s for e in eps for s in e["files"]})
    meta.update(k=len(picked), stop_reason=stop, sample_ids=list(picked), episodes=eps)
    if not picked:
        out = _empty(tab, meta)
        out.attrs["flag_files"] = flag_files
        return out
    sel = tab.set_index(sample_col).loc[picked].reset_index()
    sel["pick"] = np.arange(1, len(picked) + 1)
    sel["role"] = ROLE_EPISODE
    free = {str(x) for x in (assigned or [])}
    served = {e["episode"]: any(s in free for s in e["files"]) for e in eps if e.get("wanted")}
    by_ep = {e["episode"]: e for e in eps}
    nbright, cov = [], []
    for s, eid in zip(picked, pick_ep):
        nbright.append(int(by_ep[eid]["n_bright"].get(s, 0)))
        served[eid] = True
        cov.append(round(sum(served.values()) / len(served), 4))
    sel["bins_new"] = np.asarray(nbright, dtype=int)
    sel["coverage"] = cov
    sel["episode"] = np.asarray(pick_ep, dtype=int)
    sel.attrs["selection"] = meta
    sel.attrs["flag_files"] = flag_files
    return sel


def episode_row_flags(ledger: pd.DataFrame, sample_id, universe: pd.DataFrame | None,
                      flag_files, *, tol_ppm: float = BATCH_TOL_PPM) -> pd.Series:
    """The `episode` column of one per-file ledger (module note, ROW FLAG): True on
    a row of a file an episode wanted (`flag_files`) whose peak is on an episodic
    bin -- by its `peak_id` among the batch table's peaks of that sample on such
    a bin (the per-file peaks and the batch series carry the server's peak ids),
    or, for a row without a peak id (a synthetic row), by its m/z within
    `tol_ppm` of an episodic bin present in that sample. False everywhere else. Aligned to `ledger`'s index."""
    out = pd.Series(False, index=ledger.index, dtype=bool)
    sid = str(sample_id)
    if universe is None or sid not in {str(x) for x in (flag_files or ())} or not len(ledger):
        return out
    pids = set((universe.attrs.get("episodic_peaks") or {}).get(sid, ()))
    mzs = np.asarray((universe.attrs.get("episodic_mz") or {}).get(sid, ()), dtype=float)
    if "peak_id" in ledger.columns:
        pid = ledger["peak_id"].astype(object)
        by_id = pid.map(lambda x: pd.notna(x) and str(x) in pids)
        out |= by_id.astype(bool)
        no_id = pid.map(lambda x: not (pd.notna(x) and str(x) != ""))
    else:
        no_id = pd.Series(True, index=ledger.index)
    if len(mzs) and "mz" in ledger.columns and no_id.any():
        m = pd.to_numeric(ledger["mz"], errors="coerce").to_numpy(dtype=float)
        j = np.clip(np.searchsorted(mzs, m), 0, len(mzs) - 1)
        jl = np.clip(j - 1, 0, len(mzs) - 1)
        d = np.minimum(np.abs(mzs[j] - m), np.abs(mzs[jl] - m))
        near = np.isfinite(m) & (d <= m * float(tol_ppm) * 1e-6)
        out |= pd.Series(near, index=ledger.index) & no_id.astype(bool)
    return out


def episodes_table(selection_meta: dict | None) -> pd.DataFrame:
    """tables/episodes.csv: one row per episode of the stage's record
    (EPISODE_TABLE_COLS; list columns joined by ';')."""
    eps = (selection_meta or {}).get("episodes") or []
    rows = []
    for e in eps:
        r = {c: e.get(c) for c in EPISODE_TABLE_COLS}
        for c in ("bins_mz", "wanted", "files", "picked", "over_budget"):
            r[c] = ";".join(str(x) for x in (e.get(c) or []))
        rows.append(r)
    return pd.DataFrame(rows, columns=EPISODE_TABLE_COLS)


def episode_budget_warning(meta: dict | None) -> str | None:
    """The warning for an episode stage whose optional cap (`k_episode_max`) left a
    wanted file out; None otherwise (the stage has no budget by default)."""
    if not meta or meta.get("stop_reason") != STOP_KMAX:
        return None
    left = sorted({e["episode"] for e in meta.get("episodes") or [] if e.get("over_budget")})
    unserved = [e["episode"] for e in meta.get("episodes") or [] if e.get("wanted") and not e.get("files")]
    return (f"episode stage hit its cap k_episode_max={meta.get('k_max')}: {meta.get('n_over_budget')} "
            f"wanted file(s) of {len(left)} episode(s) left out ({len(unserved)} episode(s) with no file "
            f"at all); the coverage rule (every episodic bin at >= {meta.get('min_x_edge')}x edge has its "
            f"peak spectrum) does not hold -- drop --k-episode-max to restore it")


def describe_episodes(meta: dict | None) -> str:
    """One log line for the episode stage."""
    m = meta or {}
    u = m.get("universe") or {}
    s = (f"episodes: {u.get('n_episodic', 0)} episodic bin(s) (present in "
         f"{u.get('min_n', EPISODE_MIN_N)}-{u.get('n_max', EPISODE_MAX_N)} samples, lift >= "
         f"{u.get('min_lift', EPISODE_MIN_LIFT):g}, peak >= {u.get('bright_x_edge', EPISODE_BRIGHT_X_EDGE):g}x "
         f"the sample's edge) in {m.get('n_episodes', 0)} episode(s); {m.get('n_qualifying_bins', 0)} "
         f"peak at >= {m.get('min_x_edge', EPISODE_MIN_X_EDGE):g}x edge (the coverage rule)")
    if m.get("n_episodes"):
        s += (f"; {m.get('n_wanted', 0)} wanted file(s), {m.get('n_free', 0)} already selected, "
              f"{m.get('k', 0)} added ("
              + ("no cap" if m.get("k_max") is None else f"cap {m.get('k_max')}")
              + f"), stop={m.get('stop_reason')}")
    return s
