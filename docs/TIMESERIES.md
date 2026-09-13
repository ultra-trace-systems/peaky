# Peaky — Time series (the m/z-bin intensity matrix + disposition)

This document explains **how a batch's per-sample peaks become a samples × m/z
intensity matrix**, how that matrix is reagent-normalised into per-bin
variability, and how a *time-series disposition* is stamped back onto the ledger.
It is a module deep-dive companion to [`ARCHITECTURE.md`](ARCHITECTURE.md) (the
whole pipeline), [`SAMPLING.md`](SAMPLING.md) (which bins on `build_matrix`), and
[`CLUSTERING.md`](CLUSTERING.md) (which clusters traces built on the same cadence).

**Code:** `peaky/batch/timeseries.py`. All pure pandas/numpy, no network. It never
changes a formula — only the tier/role annotation, with commentary.

> Keep this in sync with the code. Every threshold below is a named constant or a
> literal in `timeseries.py`; if you change one there, change it here.

---

## 1. What this stage does

A sum spectrum can't tell a bright stable inlet contaminant from a real ambient
analyte; a *time series* can. In a halide-CIMS the physically meaningful quantity
is the analyte **normalised to the reagent ion** (removes instrument-sensitivity +
reagent-flow common-mode drift). Then a **flat** normalised trace (low cv, no
diel) is background, and a **variable** trace that co-varies with a known family is
real ambient chemistry. This module builds the matrix, measures each bin's
variability + family correlation, and applies **conservative** auto-actions
(demote a TS-confirmed flat background commit; flag inlet contaminants).

```
batch peaks (sample_item_id, mz, height)
   │  build_matrix: gap-cluster mz into bins (tol_ppm 5), pivot
   ▼
 matrix  (samples × m/z bins)   +   bin_mz (intensity-weighted bin centre)
   │  reagent_total(reagent_mzs) → per-sample reagent signal
   ▼  normalize: matrix ÷ reagent_total   (or pass-through if none)
 normalised matrix
   │  bin_metrics: presence, median, cv_norm
   │  family_trace + correlate: r_mono / r_formic (z-scored log-traces)
   ▼  _disposition(formula, cv, r_mono, r_formic)
 stamp ledger M0 rows: ts_cv_norm / ts_r_mono / ts_r_formic / ts_disposition
   └─ conservative demote: flat di-bromide / CO3 Assigned → Candidate
```

---

## 2. Inputs

- A batch per-**peak** time series (`sample_item_id`, `mz`, `height`,
  `datetime_utc`) — fetched by `io_mascope.fetch_batch_peaks`.
- The **ledger** (M0 rows to annotate). The reagent normaliser m/z, mono-anchor
  m/z, and formic m/z default off the ledger when not supplied.

---

## 3. The transformation, stage by stage

1. **Build the matrix** (`build_matrix`, `tol_ppm` = `DEFAULT_TOL_PPM` 5.0). Sort
   peaks by m/z, single-linkage gap-cluster into **m/z bins**
   (`cumsum(diff/mz·1e6 > tol_ppm)`), set each bin's centre to the
   **intensity-weighted mean** `Σ(mz·h)/Σh`, and pivot to a **samples × bins**
   matrix (`aggfunc="sum"`). Note: this matrix is indexed by *sample*, not
   time-binned — one row per sample.

2. **Reagent total** (`reagent_total`, `tol_ppm` 8.0). Sum the bins matching the
   reagent ion m/z (e.g. the Br₃⁻ isotopologues) → a per-sample reagent signal, the
   normaliser.

3. **Normalise** (`normalize`). Divide every bin by the per-sample reagent total (a
   concentration proxy; zeros → NaN). With no reagent series it passes through
   (TIC-normalised reagents — uronium / ¹⁵N-nitrate — handle this upstream via the
   profile `normaliser`).

4. **Per-bin metrics** (`bin_metrics`). `presence` = fraction of samples the bin is
   detected in; `median`; **`cv_norm` = std/mean** of the normalised bin.

5. **Family traces** (`family_trace`, `correlate`). A reference family's
   **z-scored mean log10 trace** (clipped to the smallest positive value); each
   bin's log-trace is Pearson-correlated to it → `r_mono` (monoterpene-SOA anchors
   `C10H16O3..O6`) and `r_formic` (formic/oxygenate pool, default bin near
   **m/z 124.9243**).

6. **Classify** (`_disposition`). Per M0 row, from formula + behaviour:
   - `CO3` adduct or `Br[23]` ion **and** `cv < FLAT_CV (0.25)` →
     `background:…(TS-flat)`;
   - else `cv < 0.25`: `Si`/`F` → inlet/instrument contaminant, otherwise flat
     background;
   - `r_mono ≥ COVARY_R (0.70)` → ambient biogenic-SOA; `r_formic ≥ 0.9` → ambient
     acid/oxygenate pool; `cv ≥ 0.45` → ambient variable; else intermediate.

7. **Stamp + conservatively demote** (`apply_timeseries`). Write the four `ts_*`
   columns onto each M0 row. If `demote` and a row is `Assigned` **and** its
   disposition is a flat **di-bromide** or **CO3-channel** background, cap it at
   `Candidate` with a tier-reason note. Nothing else is changed — never a formula.

8. **Cadence helper** (`auto_bin_minutes`). The time-bin width for the
   correlation/cluster/VK layer: the **native median inter-sample cadence**,
   **ceil**-rounded (never down), floored at 1 min. A bin narrower than the real
   spacing aliases into a spurious regular comb of empty bins; ceil guarantees
   ≥ 1 sample per bin. Falls back to `span/target_bins (50)` below 3 samples.

9. **Single-compound trace** (`trace`). From a finished run, pull one compound's
   temporal trace reproducibly: resolve a neutral formula → its m/z (highest
   `ion_score` adduct) via `merged_ledger.csv`, or take a float m/z directly; sum
   the `*_ts.parquet` peaks within `tol_ppm` per sample time.

---

## 4. Constants reference

All in `peaky/batch/timeseries.py`.

| constant | value | role |
| --- | --- | --- |
| `DEFAULT_TOL_PPM` | 5.0 | m/z-bin gap tolerance (matrix) + ledger↔bin matching |
| `reagent_total` `tol_ppm` | 8.0 | window matching a reagent ion m/z to a bin |
| `FLAT_CV` | 0.25 | `cv_norm` below this → flat / background |
| `COVARY_R` | 0.70 | correlation above this → co-varies with the family |
| `r_formic` co-vary | 0.9 | acid/oxygenate-pool disposition threshold |
| variable cv | 0.45 | `cv_norm` ≥ this (without family r) → "ambient:variable" |
| `auto_bin_minutes` `target_bins` | 50 | fallback bin count (only when < 3 samples) |
| mono anchors | `C10H16O3/O4/O5/O6` | default biogenic-SOA family for `r_mono` |
| formic default m/z | 124.9243 | default `r_formic` reference bin |

---

## 5. Metrics, defined

- **`cv_norm`** — std/mean of a bin's *reagent-normalised* intensity across
  samples; the flat-vs-variable axis (< 0.25 flat, ≥ 0.45 variable).
- **`presence`** — fraction of samples the bin is detected in.
- **bin m/z** — intensity-weighted mean `Σ(mz·h)/Σh` over a gap cluster (so a bright
  peak pulls the bin centre, not a faint neighbour).
- **`r_mono` / `r_formic`** — Pearson r of a bin's z-scored **log10** trace against
  a family's z-scored mean log-trace; the co-variation evidence.
- **`ts_disposition`** — the categorical verdict (`background:…` / `ambient:…` /
  `intermediate`) that drives the conservative demote.

---

## 6. Outputs

| artifact | content |
| --- | --- |
| `build_matrix` | `(matrix, bin_mz)` — samples × m/z-bin intensities + bin centres |
| ledger `ts_cv_norm` / `ts_r_mono` / `ts_r_formic` / `ts_disposition` | per-M0 time-series annotation (in place) |
| `apply_timeseries` summary | `{annotated, demoted, ambient, background}` |
| `trace` | tidy `[datetime_utc, <value>]` for one compound; `attrs`: mz, assignment, n_peak_ids, tol_ppm |

---

## 7. Properties, invariants & gotchas

- **The matrix is per-sample, not time-binned.** `build_matrix` rows are samples;
  time-binning for the cluster/VK layer is a separate concern handled at the
  native cadence (`auto_bin_minutes`).
- **Native cadence avoids the aliasing comb.** A fixed grid finer than the real
  (slightly irregular) sample spacing periodically catches zero samples, rendering
  a spurious regular set of drop-to-floor teeth. `ceil` to the cadence guarantees
  ≥ 1 sample per bin.
- **Normalisation removes common-mode drift.** Dividing by the reagent total
  cancels instrument-sensitivity and reagent-flow swings, so `cv_norm` reflects
  chemistry, not the source.
- **Conservative by design.** It never edits a formula and only demotes a flat
  **di-bromide / CO3** Assigned commit (TS-confirmed background) — every other
  disposition is annotation + commentary.
- **Reagent-less profiles pass through.** Uronium / ¹⁵N-nitrate normalise on TIC
  (their reagent ions sit below the acquisition window — a positive urea-CIMS
  spectrum starts at ~m/z 122, excluding the 61/121 uronium reagent ions); the
  profile's `normaliser` decides, so `normalize` here is a no-op for them.
- **A flat reagent can't cancel a varying common-mode.** Measured on one ambient urea-CIMS
  batch (2026-07-07): the uronium reagent ions are essentially flat over the diurnal
  cycle (m/z 61 ~5% / m/z 121 ~2% amplitude; the dimer even weakly anti-correlates
  with the flat-panel ~15:00 afternoon wave). So even where a reagent divisor applied,
  reagent-normalisation would **not** remove that afternoon wave — it is real ambient/
  environmental signal, not a reagent/detection artifact. (One-batch measurement.)
- **`trace` is reproducible** — it reads the run's own `*_ts.parquet` +
  `merged_ledger.csv`, so the answer is fixed by the run, not re-derived.

---

## 9. Trace reconciliation — between the merge and the stamp

`recentre_ledger`, `collapse_trace_labels` and `stamp_tolerance` run in
`assign_batch.run` after the merge (and the sidelobe flag) and before
`annotate_peaks`; all three read the batch through `batch/traces.PeakIndex`, the
same m/z-sorted index the admission table is built on.

**Why.** The merged m/z is an *anchor* minted from the few assigned samples. On
a TOF the assignment snaps it to theory — a formula is only committed where a
sample's draw lands near it — so real anchors sit 1.8 ppm (sd) from the
theoretical mass while a same-size anchor drawn at random from the ion's own
trace sits 5.7 ppm away: the trace genuinely lives ~5.8 ppm off theory, and a
±6 ppm window centred on the anchor misses most of it (measured on a
230-spectrum TOF batch: 22.6 % of anchors outside their own trace's window;
C10H16O9 `[M+NO3]-` 40 % of spectra from the anchor, 81 % from the trace centre;
mean coverage 61 → 74 % over 867 ions, gains > 5 pp on 41 %, losses on 3.5 %).
Two assigned samples whose draws differ by more than the merge tolerance also
mint two competing rows for one ion (26.8 % of well-populated rows shared a
trace). An Orbitrap ledger moved 0.11 ppm and changed nothing — the no-op that
validates the diagnosis.

1. **`recentre_ledger`** — for each row, `PeakIndex.mean_shift` from the anchor
   (the robust median of the one-peak-per-spectrum window, iterated to the local
   mode), never more than `RECENTRE_MAX_DRIFT_PPM` (10) away. The row moves only
   where the re-centred window covers *more* spectra. Guard: an anchor covering
   < `RECENTRE_GUARD_COV` (10 %) of the spectra that wants to move
   > `RECENTRE_GUARD_PPM` (6 ppm) must be corroborated (≥ 2 assigned files, or
   Assigned) — an almost-empty anchor plus a long jump is how a label lands on a
   neighbour's trace. Adds `mz_anchor`, `mz_trace`, `trace_offset_ppm`,
   `trace_cov_anchor`, `trace_cov`, `trace_moved`, `trace_guarded`.
2. **`collapse_trace_labels`** — gap-cluster `mz_trace` at the merge tolerance
   (`trace_id`); a trace with several rows keeps one `winner` by the merge's
   own ordering — most assigned files → tier → `ion_score` — with proximity to
   the trace centre only as a deterministic tie-break; the rest become
   `collapsed` (kept, flagged, never stamped; `single` = alone). Never
   `ion_score` first: per-file, it is blind to reproducibility and on that batch
   it lost 2 of 14 known channels to one-file competitors. And never proximity
   before the score: the anchors were snapped to theory, so on a live TOF batch
   "nearest" handed the trace of the known monomer C10H16O9 to a one-file
   C15H21NO3 whose anchor happened to sit 0.25 ppm from the centre (score 0.846
   vs 0.960); the score ordering picked the reference-list label in 4 of 4
   same-file, same-tier ties where exactly one label was on the list.
3. **`stamp_tolerance`** — the stamping half-window = max(tol, min(2·tol,
   2.5·σ)), σ = the **third quartile** of the per-trace robust scatter
   (`traces.batch_scatter_ppm`; the dim traces scatter more — 3.6 vs 1.2 ppm on
   the TOF — and are the ones a window sized from the bright ones loses; the top
   decile on one Orbitrap mode is a scan-edge artefact 40× the median). An
   Orbitrap (0.24–0.34 ppm) keeps 6 ppm; a TOF (3.8–4.2 ppm) gets ~10 ppm.
   `annotate_peaks` then stamps from `stamping_frame`, which uses `mz_trace` and
   skips collapsed rows; the one-to-one contest is unchanged.

Measured on the real runs (scripts/eval_trace_stamping.py): TOF, all ions, mean
stamped coverage 0.422 → 0.488 (38 % of ions gaining > 5 pp, 3.7 % losing),
C10H16O9 0.135 → 0.857; two Orbitrap batches 0.754 → 0.754 and 0.542 → 0.542.
`batch_summary.json['traces']` records the counts, σ and the window.

| constant | value | role |
| --- | --- | --- |
| `RECENTRE_MAX_DRIFT_PPM` | 10.0 | an anchor may move at most this far (median move 2.7 ppm; ±10 delivers +11.6 of the +12.7 pp total) |
| `RECENTRE_GUARD_COV` / `RECENTRE_GUARD_PPM` | 0.10 / 6.0 | the low-evidence guard: an anchor under 10 % coverage may not move > 6 ppm uncorroborated |
| `STAMP_TOL_SIGMA` / `STAMP_TOL_MAX_X` | 2.5 / 2.0 | window = this many σ, never wider than this × the merge tolerance |
| `traces.SCATTER_Q` | 0.75 | the per-trace scatter quantile that sizes the window |
| `traces.MZ_FLOOR_DA` | 1.5 mDa | every window is ±max(tol ppm, this) — the stamp's own floor, shared by occurrence and trace |

---

## 8. Code map

| function | role |
| --- | --- |
| `build_matrix` | gap-cluster peaks into m/z bins → samples × bins matrix + bin centres |
| `auto_bin_minutes` | native-cadence time-bin width (ceil, anti-aliasing) for cluster/VK |
| `reagent_total` / `normalize` | per-sample reagent signal; reagent-normalised matrix |
| `bin_metrics` | per-bin presence / median / `cv_norm` |
| `family_trace` / `correlate` | z-scored log family trace; per-bin Pearson r |
| `_disposition` | formula + (cv, r_mono, r_formic) → background/ambient/intermediate |
| `apply_timeseries` | stamp `ts_*` columns; conservative flat-background demote |
| `find_ts_parquet` / `trace` | locate the run's TS parquet; one-compound reproducible trace |
| `recentre_ledger` / `collapse_trace_labels` / `stamp_tolerance` | §9: re-centre merged anchors on their traces, collapse competing labels, size the stamp window |
| `collapse_peak_matches` | one row per physical peak (Mascope's match-expanded rows folded back) |
