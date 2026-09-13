# Peaky — Sampling (which real samples get assigned)

This document explains **how a many-sample batch is reduced to a small subset to
assign**: one selector, a greedy **presence set-cover** over the batch's m/z bins
with a marginal-gain stop, and the exact arithmetic it uses — followed by the
**residual stage** (§3b), a second, targeted selection over the bins the cover
left in no assigned file. It is a module
deep-dive companion to [`ARCHITECTURE.md`](ARCHITECTURE.md) (the whole pipeline,
the *Whole batch* flow), [`MERGE.md`](MERGE.md) (which combines the per-sample
ledgers this selects), and [`TIMESERIES.md`](TIMESERIES.md) (whose `build_matrix`
the selector bins on).

**Code:** `peaky/batch/sampling.py`. Pure pandas/numpy, no network. The selected
`sample_item_id`s feed `assign.run` one at a time — `match_compounds` is
per-sample, so a synthetic union spectrum can't be scored.

> Keep this in sync with the code. Every threshold below is a named constant in
> `sampling.py`; if you change one there, change it here. `BATCH_TOL_PPM` serves
> every batch-level operation — sample selection, the admission table and the
> merge (`assign_batch.DEFAULT_TOL_PPM` *is* it) — and all three must bin
> identically, so change it in one place only.

---

## 1. What this stage does, and why it is the recall story

`assign.run` scores **one** sample, so an analyte present only during an event
window is never a candidate when the chosen file predates the spike. To get a peak
list representative of the *whole* batch, assign a small subset and merge it.

The merge ([`MERGE.md`](MERGE.md)) has **no prevalence filter**: a compound is in
the ledger iff some *assigned* sample contained it. Selection is therefore the
whole recall story — whatever is only in samples that were never assigned is
simply missing. That is what the selector optimises: **how many of the batch's
distinct m/z bins are present in at least one assigned sample.**

```
per-peak batch table (sample_item_id, mz, height[, datetime_utc, name])
        │  build_matrix → samples × m/z bins (presence = height > 0)
        ▼
 ┌──────────────── presence set-cover ──────────────────────────────────┐
 │ universe = bins present in ≥ MIN_PREVALENCE (2) samples  (no floor)  │
 │ repeat: pick the sample holding the most NOT-YET-COVERED bins        │
 │   stop when  k ≥ K_MIN (6)  and  next gain < MIN_GAIN (0.5 %)        │
 │   or   k = K_MAX (30)  → flagged  (the batch was still gaining)      │
 │   or   nothing left to cover  → pad to K_MIN with the richest TIC    │
 └──────────────────────────────────────────────────────────────────────┘
        ▼
 selected rows in pick order: pick, role, bins_new, coverage
 + .attrs['selection'] = {k, n_bins, achieved_coverage, stop_reason, …}
```

---

## 2. Inputs

- The **per-peak batch table**: `sample_item_id`, `mz`, `height` per peak
  (`sample_item_name`, `datetime_utc`, and a group column optional). The batch
  pipeline passes the full-batch time series it fetches anyway (`ts_peaks`); the
  pool passes the pooled table. A per-*sample* table (`samples.list`) cannot be
  binned and is refused (`is_per_peak` is the check).

---

## 3. The transformation, stage by stage

1. **`sample_table`.** Collapse to one row per sample: `sample_item_id`,
   `datetime_utc`, `sample_item_name`, `tic` (= Σ heights), `n_peaks`.

2. **Bin.** `timeseries.build_matrix` gap-clusters every peak by m/z at
   `BATCH_TOL_PPM` (6 ppm — passed explicitly; it is also the merge's tolerance,
   so a bin the selector covered is the bin the merge sees) and pivots to a
   samples × bins height matrix. Presence = `height > 0`.

3. **Universe (the prevalence gate).** A bin enters the universe if it is present
   in **≥ `MIN_PREVALENCE` (2) samples**. There is **no height floor**: the
   picker's own detection edge varies ~1000× between instruments and modes (TOF
   0.8 cps; Orbitrap EasyIC ~10; urea 122-600 ~150; a mode with the reagent ion in
   range ~800), so any absolute cps floor is a no-op on one mode and blinds the
   selector on another. The prevalence gate is the noise filter and has no units
   — a bin seen in exactly one sample is a singleton, and singletons are 6–14 % of
   Orbitrap bins and ~47 % of TOF bins. (If **no** bin reaches the prevalence — a
   1-sample batch, but also any batch whose samples share no m/z at this
   tolerance — the gate relaxes to 1 rather than leaving an empty universe that
   covers trivially and picks nothing.)

4. **Greedy cover.** Each pick is the sample with the most universe bins not yet
   covered (`bins_new`, the marginal gain); `coverage` is the cumulative fraction
   of the universe covered. Presence cover is *submodular*, so plain greedy is
   near-optimal and — unlike an arg-max-by-brightness ranking, whose winner sets
   are disjoint — it can trade redundancy: two near-identical rich samples are
   not both taken. **Tie-break:** equal-gain samples resolve to the first row of
   `build_matrix`'s matrix, which is pivoted on the sample id — i.e. the
   **lexicographically smallest `sample_item_id`**. The pick is therefore
   deterministic and independent of the input row order.

5. **Stop rule = marginal gain.** After `K_MIN` (6) picks, stop when the next
   sample would add **< `MIN_GAIN` (0.5 %) of the universe** → `stop_reason =
   'gain-floor'`, `next_gain` records the rejected fraction. The floor is
   `MIN_GAIN × n_bins` over the **gated** universe: the singletons the prevalence
   gate dropped are not coverable, so counting them would scale the floor by an
   irrelevant, instrument-dependent number. `K_MAX` (30) is a wall-clock
   **budget only**: a run that reaches it while the next gain is still ≥ the
   floor stops with `'k_max'` and is **warned** in the log, the report and
   `batch_summary.json` — it means the batch was still gaining and `--k-max`
   should be raised. When every universe bin is covered (or every sample is
   taken) the reason is `'exhausted'`. A *coverage-target* stop does not exist:
   measured on real batches it is trivially met behind a height floor and never
   met without one.

   **Stop-check precedence** (the order the loop tests them): *exhausted* →
   *gain-floor* → *k_max*. A pick that adds nothing is never taken, even below
   `K_MIN`; the budget is the last word, and only while the batch is still
   gaining above the floor.

6. **Pad.** If the greedy ended before `K_MIN` (a tiny batch, or the bins ran
   out), the richest remaining samples by `tic` are added with `role = 'pad'`
   and `bins_new = 0` — cheap cross-file corroboration, never fewer than
   `min(K_MIN, n)`. **Tie-break:** a *stable* sort on descending `tic`, so
   equal-TIC samples keep the matrix's (sorted-id) row order.

7. **Groups (pooling).** With `group_col`, the same single greedy runs over the
   whole pool and the meta additionally records `coverage_by_group` (the fraction
   of each group's own universe bins covered by the picks, whichever group they
   came from) and `picks_by_group`. Because the objective is presence, not
   brightness, a loud group cannot hog the picks; measured on a pooled 5-zone
   campaign every zone reached 87–93 % of its own bins, the smallest zone with
   zero explicit picks.

---

## 3b. The residual stage (targeted second selection)

The gain stop leaves a tail: bins present in ≥ 2 samples that no assigned file
holds. Measured on a 6154-sample Texas Ur+ campaign run (15 cover files at the gain-floor stop): **18 % of 4702 universe bins** (846). Almost
all of it is the noise-edge tail — 0.02 % of the universe's signal, median campaign
maximum 299 cps against a ~154 cps median picker edge, 96 % never above 1 kcps —
but **36 of those bins reach 1–3.4 kcps somewhere**, and nothing in the tail can
ever enter the ledger, because the merge only holds what an assigned file
contained. A whole-batch stamp cannot help either: it names ions the ledger
already has.

So `assign_batch.run` runs a **second, targeted selection after the cover's merge
and stamp** (`residual_universe` + `select_residual_cover`; switch `residual`,
CLI `--residual` / `--no-residual`, **on by default**):

1. **Residual universe** (`residual_universe`). The cover's own bins
   (`timeseries.bin_ids` at `BATCH_TOL_PPM` — the rule `build_matrix` pivots on,
   read off the long table so no dense matrix is built) and the cover's own
   prevalence gate. A universe bin is *residual* when it is
   - **(a) absent from every assigned file** (no cover file holds a peak in it),
   - **(b) unexplained by the whole-batch stamp**: no peak of the bin carries an
     `ion_formula` in the annotated time series — so an isotope satellite, a
     reagent line or an artifact the stamp already names is not re-targeted
     (once predicted satellites are stamped, those drop out here too), and
   - **(c) bright somewhere**: its maximum over the batch, as a multiple of *that
     sample's* own noise edge (`EDGE_Q` = the 1st percentile of the sample's
     heights, the statistic every height gate is a multiple of), is
     ≥ `RESIDUAL_MIN_X_EDGE` (5) — `--residual-min-x-edge`; `--residual-min-cps`
     is an absolute floor instead. The floor is **never below the run's own
     admission-gate multiple** (`height_cutoff_x_edge`, [`ASSIGNMENT.md`](ASSIGNMENT.md)):
     a bin no file would admit is not worth a file; the absolute gate
     (`--height-cutoff`) is a floor as well. On the campaign above the 36 bright
     bins sit ≥ 6.5× the median edge.
   Each survivor is then **tiered against a saturating neighbour** — a bin within
   `timeseries.SIDELOBE_DMZ` (12 mDa) that reaches `SIDELOBE_MIN_PARENT`
   (50 kcps) somewhere (its maximum, not the batch median the channel flagger
   uses: a plume ion saturates only in the samples where its sidelobe appears,
   and can sit a single mDa away at m/z 100) — as `flag_sidelobe_channels` tiers
   a merged channel ([`TIMESERIES.md`](TIMESERIES.md)): with ≥
   `SIDELOBE_MIN_PAIRS` (20) samples holding both, the neighbour's median over
   them ≥ 50 kcps and ≥ `SIDELOBE_FACTOR` (100×) the bin's, a height ratio
   **locked** across those samples (cv <
   `SIDELOBE_CV`, 0.08) makes it a **confirmed sidelobe** (`tier = 'sidelobe'`):
   dropped, since a file would only label it an artifact (`n_sidelobe`; the rows
   are still written to `residual_bins.csv`). A varying ratio means an
   independent ion → `'clean'`, `ratio_cv` recorded. With too few shared samples
   to measure, cleanup's static rule in the sample where the bin peaks (a
   neighbour there ≥ 50 kcps and ≥ 100× the bin) makes it a **suspect**
   (`'suspect'`, `n_suspect`): *kept*, because a real ion that merely sits near
   a bright peak does get assigned, but covered last (below).
2. **Residual cover** (`select_residual_cover`). A sample *counts* for a bin only
   where the bin's height (summed per sample and bin, as the matrix does) is
   ≥ `RESIDUAL_FRAC_OF_MAX` (50 %) of the bin's maximum — the bin is assigned
   where it stands tall, with its isotopes visible — **and** at or above that
   sample's own gate. Greedy over that relation: each pick is the sample counting
   for the most not-yet-covered residual bins, **clean bins first** — the gain is
   lexicographic, one clean bin outranks any number of suspects, so budget goes
   to suspects only once no sample adds a clean bin; stop at `RESIDUAL_K_MAX`
   (10) picks (`'k_max'`, `--residual-k-max`) or when the next sample adds no bin
   (`'exhausted'`); no bins → `'empty'`. Tie-break as the cover's (smallest
   `sample_item_id`). `bins_new` and `coverage` are of the **residual** universe.
3. **Assign + one merge.** The picks go through the same per-file path (serial
   or the process pool, the same cfg and offsets), then `align()` runs **once**
   over every per-file ledger — cover and residual together — so the merged
   ledger, the trace reconciliation and the time-series stamp include them.
   Merged rows carry **`stage`** (`cover` if any cover file holds the ion, else
   `residual`: the stage that first put it in the ledger); `n_files` still counts
   only files that contain the ion.

Measured on the campaign above (the 21-sample manual experiment this stage
replaces; `RESIDUAL_K_MAX` bounds the automatic run at 10): the 36 bright bins
needed 21 samples at 50 % of their maximum; assigning them gave **6 Assigned +
10 Candidate** (mostly "reagent-N isobar unresolved"), 8 isotope satellites of
parents the cover already assigns (the stamp explains those once it carries
predicted satellites), 1 sidelobe artifact and 11 unexplained — the bright end
of the residual, at ~4 min per extra file.

```
cover merge + stamp
        │  residual_universe: universe bins  ∖ present-in-an-assigned-file
        │                                     ∖ stamped  ∩  max ≥ floor × edge
        ▼
 ┌────────────── residual cover ────────────────────────────────────────┐
 │ sample counts for a bin iff height ≥ 50 % of the bin's max ∧ ≥ gate  │
 │ repeat: pick the sample counting for the most uncovered residual bins│
 │   stop at RESIDUAL_K_MAX (10) → 'k_max'  |  next adds none → 'exhausted'
 └──────────────────────────────────────────────────────────────────────┘
        ▼
 assign the picks (same per-file path) → ONE align over cover + residual files
 selected_samples.csv += picks (role 'residual'); residual_bins.csv; stage column
```

---

## 4. Constants reference

All in `peaky/batch/sampling.py`; the CLI (`--k-min`, `--k-max`, `--min-gain`,
`--residual`, `--residual-min-x-edge`, `--residual-min-cps`, `--residual-k-max`)
mirrors them for `batch` and `pool`.

| constant | value | role |
| --- | --- | --- |
| `MIN_PREVALENCE` | 2 | a bin enters the universe if present in ≥ this many samples |
| `K_MIN` | 6 | picks taken before the marginal-gain stop applies; pad target |
| `MIN_GAIN` | 0.005 | stop when the next pick adds < this fraction of the universe |
| `K_MAX` | 30 | budget; hitting it while still gaining → `stop_reason='k_max'` + warning |
| `BATCH_TOL_PPM` | 6.0 | m/z gap-clustering tolerance for the bins — the one tolerance for every batch-level binning (selection and merge: `assign_batch.DEFAULT_TOL_PPM = BATCH_TOL_PPM`); recorded as `selection.tol_ppm` |
| `RESIDUAL_DEFAULT` | True | the residual stage runs unless `--no-residual`; off reproduces the cover-only run exactly |
| `RESIDUAL_MIN_X_EDGE` | 5.0 | a residual bin must reach this × its sample's noise edge somewhere (never below the run's gate multiple) |
| `RESIDUAL_FRAC_OF_MAX` | 0.5 | a sample counts for a residual bin only at ≥ this share of the bin's maximum |
| `RESIDUAL_K_MAX` | 10 | budget of the residual stage; hitting it while still gaining → `stop_reason='k_max'` |
| `EDGE_Q` | 0.01 | the noise edge: the 1st percentile of a sample's heights (= `passes.noise_edge`) |

---

## 5. Metrics, defined

- **`bins_new`** — universe bins this pick covered for the first time (the
  marginal gain that ranked it).
- **`coverage`** — cumulative fraction of the universe covered after this pick;
  the last row's value is `achieved_coverage`.
- **`next_gain`** — the fraction the *rejected* next pick would have added
  (0 for `exhausted`).
- **`n_bins` / `n_bins_total` / `n_bins_gated`** — universe size, all bins, and
  the singletons the prevalence gate removed.
- **`tic`** — Σ peak heights per sample; only used to rank pads.
- **`n_uncovered` / `n_explained` / `n_below_floor` / `n_residual`** (residual
  stage) — the funnel: universe bins in no assigned file; of those, the ones the
  stamp explains; the ones never reaching the floor; the ones targeted.
- **`max_x_edge`** — a residual bin's maximum height as a multiple of the edge
  of the sample it peaks in (`residual_bins.csv`).
- **`coverage_of_residual`** — the fraction of the residual universe the extra
  picks cover (at ≥ `frac_of_max` of each bin's maximum).

---

## 6. Outputs

| artifact | content |
| --- | --- |
| `select_cover_samples` | selected `sample_table` rows in pick order: `pick`, `role ∈ {cover, pad}`, `bins_new`, `coverage` (+ the group column); `.attrs['selection']` = the meta dict |
| `select_cover_sample_ids` | just the `sample_item_id`s (pick order) |
| `tables/selected_samples.csv` | the chosen subset written by `assign_batch.run` (the pool writes it from its own selection table, plus `selection_provenance.csv` at the run root) |
| `batch_summary.json['selection']` | `method, k, n_samples, n_bins, n_bins_total, n_bins_gated, min_prevalence, tol_ppm, achieved_coverage, stop_reason, next_gain, k_min, k_max, min_gain[, coverage_by_group, picks_by_group]` — also copied into `run_manifest.json['output']['counts']` |
| `k_max_warning(meta)` / `describe(meta)` | the warning text / one-line log summary the callers print |
| `residual_universe` | the residual bins (`bin`, `bin_mz`, `prevalence`, `max_cps`, `max_x_edge`, `sample_at_max`, `tier` ∈ clean / suspect, `sidelobe_of`, `n_pairs`, `ratio_cv`); `.attrs['residual']` = the funnel (incl. `n_sidelobe`, `n_suspect`) + floor, `.attrs['sidelobes']` = the dropped rows (records), `.attrs['edge_cps']` = the per-sample edge ({id → cps}) |
| `select_residual_cover` | the extra picks, same shape as `select_cover_samples` with `role = 'residual'`; `.attrs['selection']` = {method 'residual-cover', k, n_bins, frac_of_max, k_max, achieved_coverage, stop_reason, next_gain}, `.attrs['covered_by']` = {bin → pick} |
| `tables/residual_bins.csv` | the targeted bins + `covered_by` (written by `assign_batch.run` when the stage is on) |
| `batch_summary.json['selection']['residual']` | `n_bins_residual, n_universe, n_uncovered, n_explained, n_below_floor, n_sidelobe, n_suspect, floor{min_x_edge, min_cps, edge_median_cps, source}, frac_of_max, k, k_max, coverage_of_residual, stop_reason, next_gain, sample_ids` (`skipped` when there was no time series) |

---

## 7. Evidence (2026-09-12, real batch tables)

Rare (< 5 % prevalence) ions of a dedicated zone run's ledger that are present
in the samples selected from the pooled 5-zone table (5036 samples):

| selector | k | rare ions present |
| --- | --- | --- |
| 5 time-spaced + max-TIC (the former default) | 6 | 54 % |
| brightest arg-max, `k_max` 10 + 2 endpoints (the former `--select brightest`) | 12 | 91 % — and its 0.85 coverage target was never reached on any real batch; `k_max` always bound |
| presence cover with a 1000 cps floor | 12 | 64 % |
| **presence cover, no floor, prevalence ≥ 2, gain stop** | **15** | **94 %** (99 % of all ledger ions) |

Where the gain stop lands: 9–18 on 265–995-sample field-campaign batches, 11 on
a 66-sample zero-air batch, 11 on a 265-sample chamber day, 15 on the 5036-sample
pool (in 10 s), 18 on a 230-sample TOF batch, 10–30 on the six parallel Orbitrap
modes of one instrument. Selection is deterministic (identical picks on re-run).

---

## 8. Properties, invariants & gotchas

- **Selection is the recall story.** The merge keeps whatever any assigned sample
  contained and nothing else; assigning more, better-chosen samples is the only
  way to find more.
- **No absolute cps anywhere in selection.** The universe is prevalence-gated,
  not height-gated — the same code covers a TOF (edge 0.8 cps) and an Orbitrap
  mode with the reagent ion in range (edge 800 cps).
- **Selection has no height gate; ASSIGNMENT does.** Once the selected samples
  are assigned, the height-gated passes gate each one on a multiple of *its own*
  noise edge (`height_cutoff_x_edge`; 1.0 = the bottom 1 % of picked peaks). That
  multiple is a property of the **peak picker**: a picker that stops at the edge
  wants 1.0 (rare real ions sit at 1–3× it), while a picker that picks *into* the
  noise admits almost everything it found at 1.0 — on one 230-spectrum TOF batch
  1.0 merged 4346 ions, 3307 of them single-file, where 5.0 kept 57 % of the
  picked peaks and 74 % of the assigned ones, i.e. a tighter candidate list. By
  default (`"auto"`) a batch run **derives** the multiple from the batch's own
  peaks — the smallest grid multiple whose admitted peaks are at most 20 %
  transient, which keeps a picker that stops at the edge at exactly 1.0 and
  raises one that picks into the noise ([`ASSIGNMENT.md`](ASSIGNMENT.md)); a site
  pins a number for its own instrument via a `--reagent-config` profile's
  `height_cutoff_x_edge` ([`REAGENTS.md`](REAGENTS.md) §3a); no bundled profile
  sets one.
- **A time grid / max-TIC add nothing.** The clock is uncorrelated with the air,
  and the first greedy pick is already the sample carrying the most distinct
  bins. That count is the objective, never brightness — *richest* (§3.6) means
  total ion current and ranks the **pads** only.
- **`k_max` is a budget, not a target.** If it binds, coverage is incomplete and
  the run says so; raise it rather than trusting the ledger to be complete.
- **Few-samples shortcut.** `n == 0` → empty; `n ≤ K_MIN` → take all.
- **An empty universe selects nothing.** No m/z bins at all, or no peak with a
  positive height anywhere (so not even the `≥ 1` prevalence fallback finds a
  bin) → an empty selection with `k = 0` and `achieved_coverage = 0.0`, never a
  padded subset whose coverage is NaN.
- **Both tie-breaks are deterministic.** Equal-gain cover picks go to the
  lexicographically smallest `sample_item_id`; equal-TIC pads keep that same
  order (stable sort). Shuffling the input rows cannot change the result.
- **Pick order is assignment order.** `assign_batch.run` assigns the ids in the
  order returned (and `align()` has order-sensitive tie-breaks), so the CSV's
  `pick` column is also the merge order — the residual picks continue the
  numbering after the cover's.
- **A confirmed sidelobe never costs a file; a mere suspect never loses one to a
  clean bin.** The ratio-lock test is the same one the channel flagger trusts;
  the static rule alone only orders the greedy.
- **The residual stage cannot re-target what the ledger explains.** Its universe
  is read off the annotated time series *after* the cover's stamp, so an ion the
  ledger names anywhere — analyte, satellite, reagent line, artifact — is not a
  target, and the stage only ever adds files for bins no assigned file holds.
- **A residual file is a full assignment.** The targeted sample is assigned like
  any cover file (every peak, the same gate), so its background ions corroborate
  the cover's rows (`n_files` grows) while the ions only it holds carry
  `stage = 'residual'`.
- **`--no-residual` is today's run.** Off, nothing changes: no `stage` column,
  no `residual` block, no `residual_bins.csv`, byte-identical material outputs.

---

## 9. Code map

| function | role |
| --- | --- |
| `sample_table` | batch table → one row/sample (id, time, name, tic, n_peaks) |
| `is_per_peak` | does the table carry `mz` + `height` per row (what the cover needs)? |
| `select_cover_samples` | THE RULE: greedy presence set-cover with the marginal-gain stop (+ per-group coverage) |
| `select_cover_sample_ids` | id-list convenience wrapper |
| `residual_universe` | THE RESIDUAL STAGE, part 1: the universe bins in no assigned file, unexplained by the stamp, bright somewhere |
| `select_residual_cover` | THE RESIDUAL STAGE, part 2: greedy cover of those bins by the samples in which each stands at ≥ 50 % of its maximum |
| `describe_residual` | the one-line log summary of the funnel + the residual cover |
| `timeseries.bin_ids` (reused, at `BATCH_TOL_PPM`) | the row-aligned bin rule `build_matrix` pivots on, read off the long table |
| `k_max_warning` / `describe` | the warning text / log line for a selection meta dict |
| `timeseries.build_matrix` (reused, at `BATCH_TOL_PPM`) | the samples × m/z-bin matrix the cover bins on |
