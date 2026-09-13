# Peaky — Merge (cross-sample alignment + m/z jitter)

This document explains **how per-sample assignments become one merged peak list**
— the offset-aware m/z alignment, the consensus-row selection, and the file-to-file
**jitter** accounting. It is a module deep-dive companion to
[`ARCHITECTURE.md`](ARCHITECTURE.md) (the whole pipeline, the *Whole batch* flow),
[`SAMPLING.md`](SAMPLING.md) (which picks the files merged here), and
[`ASSIGNMENT.md`](ASSIGNMENT.md) (what each per-file ledger contains).

**Code:** `peaky/batch/assign_batch.py`. `align()` / `merge_union()` /
`jitter_report()` are **pure** (offline-tested); `run()` does the network
assignment loop and writes the run artifacts.

> Keep this in sync with the code. Every threshold below is a named constant or a
> literal in `assign_batch.py`; if you change one there, change it here.

---

## 1. What this stage does

`match_compounds` is per-sample — a synthetic union spectrum can't be scored — so
the batch path assigns each cover-selected file **separately** and then **combines
the real per-file ledgers by m/z**. The combine is **offset-aware**: each file
carries a median mass offset, alignment happens on offset-corrected m/z so a
genuine same-peak isn't split by per-file calibration drift, and the report
separates the *raw* mass spread from the *calibration-removed* (residual) spread —
that residual is the genuine peak-position noise the maintainer wanted to
investigate.

```
selected sample_ids (SAMPLING.md)
   │  for each: assign.run → per-file ledger (kept on disk) + estimate_offset
   ▼
 per_file = {sid → M0 rows [mz, neutral_formula, adduct, tier, ion_score]}
   │  align(offsets): _mz_adj = mz·(1 − offset_ppm/1e6)
   ▼  single-linkage gap-cluster _mz_adj at tol_ppm (6)
 one row per m/z cluster:
   consensus mz = mean(raw mz);  best = Assigned>Candidate, then ion_score
   n_files, srcs, formula_agree, mz_jitter_ppm_raw, mz_jitter_ppm_caldj
   │  (positive urea: prefer_amine_over_ammonium at the merged level)
   ▼
 trace reconciliation (TIMESERIES.md §9): each row's anchor re-centred on its
 own trace (mz_trace), competing labels on one trace collapsed (trace_role),
 the stamp window sized to the batch's per-ion scatter
   ▼
 merged_ledger.csv  +  jitter.csv  +  batch_summary.json  →  _batch_ts.parquet stamp
```

---

## 2. Inputs

- `per_file` — `{src → DataFrame}` of each file's **M0 (assigned-compound) rows**
  in the `_M0_COLS` schema (`mz`, `neutral_formula`, `adduct`, `tier`,
  `ion_score`), extracted by `_m0`.
- `offsets` — `{src → median ppm}` from `io_mascope.estimate_offset`
  ([`DATA_IO.md`](DATA_IO.md)); missing → treated as 0.

---

## 3. The transformation, stage by stage

1. **Assign each selected file** (`run`). Loop the selected `sample_ids`,
   `A.run(sid, …)` each, write `per_file/<sid>_ledger.csv`, keep the M0 rows, and
   record the file's `estimate_offset`. The reagent's analyte channels are forced
   at batch level (`assign_kw.setdefault("adducts", prof.adducts)`) so a per-sample
   match gap can't flip polarity.

2. **Offset-correct for alignment** (`align`). For each file,
   `_mz_adj = mz · (1 − offset_ppm/1e6)`. This is used **only** to cluster; the
   reported masses stay raw.

3. **Gap-cluster** (`_cluster_mz`). Sort by `_mz_adj`; consecutive-gap
   single-linkage: `gaps = diff(mz)/mz · 1e6`, `cluster_id = cumsum(gaps >
   tol_ppm)` with **`tol_ppm` = `DEFAULT_TOL_PPM` (6.0 = `sampling.BATCH_TOL_PPM`,
   the tolerance the selector binned on — see [`SAMPLING.md`](SAMPLING.md))**.
   One cluster ≈ one physical peak across files.

4. **Pick the consensus row.** Within a cluster, rank by tier
   (`TIER_RANK = {Assigned:2, Candidate:1}`, else 0) then `ion_score`, both
   descending; the top row supplies the merged `neutral_formula` / `adduct` /
   `tier` / `ion_score`. The merged **`mz` is the mean of the cluster's raw m/z**.
   Also recorded: `n_files` (distinct srcs), `srcs`, `formula_agree`
   (`≤ 1` distinct formula), and the two jitter spreads (§5).

5. **Positive urea amine re-read.** When `prof.polarity == "+"`,
   `cleanup.prefer_amine_over_ammonium(merged, r_min = amine_r_min (0.7))` re-reads
   uncorroborated `[M+NH4]⁺` as `[M+H]⁺` of the `+NH3` amine (mass/isotope-
   identical, simpler in an N-rich source) — done at the **merged** level where
   cross-channel corroboration is complete.

6. **Pool the plausibility audit + write artifacts.** Per-file plausibility
   demotes are pooled and written; `merged_ledger.csv` (root), `jitter.csv`
   (tables/), `selected_samples.csv`, and `batch_summary.json` are emitted.

7. **Jitter report** (`jitter_report`, standalone analysis). Per-file offset =
   median observed-vs-theoretical ppm of its assignments (`_theo_ppm`). Then:
   - **`by_formula`** — same `(neutral_formula, adduct)` in ≥ 2 files:
     `mz_jitter_raw` (raw ppm spread) vs `mz_jitter_resid` (spread after removing
     each file's offset) + `tier_stable`.
   - **`by_mz`** — offset-corrected m/z clusters exposing **formula
     disagreements** (same peak, different formula across files).

---

## 4. Constants reference

All in `peaky/batch/assign_batch.py`.

| constant | value | role |
| --- | --- | --- |
| `DEFAULT_TOL_PPM` | 6.0 (`= sampling.BATCH_TOL_PPM`) | single-linkage gap tolerance for cross-file m/z clustering — the same constant the selector bins on |
| `TIER_RANK` | `{Assigned:2, Candidate:1}` | consensus-row preference (then `ion_score`) |
| `_M0_COLS` | `[mz, neutral_formula, adduct, tier, ion_score, admitted_by, occurrence]` | the per-file M0 schema aligned (the last two = admission provenance, carried for the winning row; absent columns are tolerated) |
| `run` `amine_r_min` | 0.7 | min trace correlation for the positive amine re-read |
| `run` `k_min` / `k_max` / `min_gain` / `min_prevalence` | 6 / 30 / 0.005 / 2 | passed through to `sampling.select_cover_samples` (see [`SAMPLING.md`](SAMPLING.md)) |

---

## 5. Metrics, defined

- **`mz_jitter_ppm_raw`** — `(max − min)/mean · 1e6` over a cluster's **raw**
  m/z; the total file-to-file mass spread.
- **`mz_jitter_ppm_caldj`** — the same spread over the **offset-corrected** m/z;
  what remains after per-file calibration is removed (the genuine noise).
- **`formula_agree`** — `True` iff the cluster carries ≤ 1 distinct neutral formula.
- **per-file offset** — median observed-vs-theoretical ppm of a file's assignments
  (`jitter_report`); `offset_spread_ppm` = max − min across files.
- **`mz_jitter_resid`** (by_formula) — ppm spread of one assignment across files
  **after** subtracting each file's offset; the residual peak-position noise.
- **`n_files` / `n_in_all_files` / `n_single_file`** — how widely a merged peak was
  seen; corroboration breadth.

---

## 6. Outputs

| artifact | content |
| --- | --- |
| `merged_ledger.csv` (run root) | one row per m/z cluster: consensus mz, best assignment, `n_files`, `srcs`, `formula_agree`, `mz_jitter_ppm_raw/caldj`, plus the trace reconciliation columns (`mz_anchor`, `mz_trace`, `trace_offset_ppm`, `trace_cov_anchor`, `trace_cov`, `trace_moved`, `trace_guarded`, `trace_id`, `trace_role`; [`TIMESERIES.md`](TIMESERIES.md) §9) — **the result** |
| `tables/jitter.csv` | long form, one row per (cluster, file): `cluster`, `src`, `mz`, formula, adduct, tier, `ion_score` |
| `per_file/<sid>_ledger.csv` | each assigned file's full single-sample ledger (audit / re-merge) |
| `tables/selected_samples.csv` | the selected subset in pick order (`pick`, `role`, `bins_new`, `coverage`) |
| `batch_summary.json` (run root) | reagent/context, the `selection` block (k, achieved coverage, stop reason), the resolved height gate (`height_cutoff_x_edge` + its source, and the `gate` derivation block), the `admission` block, the `traces` block (re-centred / collapsed counts, per-ion scatter, stamp window), per-file offsets + noise edges, merged tier counts, agreement counts |
| `jitter_report()` dict | `{offsets, by_formula, by_mz, summary}` — the standalone jitter analysis |

---

## 7. Properties, invariants & gotchas

- **Assign reals, then merge.** A synthetic union spectrum can't be scored
  (`match_compounds` is per-sample), so combining real per-file ledgers is the only
  principled path.
- **Offset correction aligns; raw masses report.** `_mz_adj` is used only to avoid
  splitting a peak by calibration drift — the merged `mz` and `mz_jitter_ppm_raw`
  are computed on raw masses, so the two jitter columns are an honest before/after.
- **Consensus = best tier, then best score.** The merged formula is the
  highest-confidence one seen, not a vote — but `formula_agree`/`by_mz` surface
  disagreements rather than hiding them.
- **The amine re-read is positive-only and merged-level** — it needs the full
  cross-channel picture, so it can't run per-file.
- **Fetch the batch by name for fresh ids.** `run(batch=…)` re-fetches the
  per-sample list live so the selected ids are valid for `get_peaks` (cached ids go
  stale / 404 when a server copy is renamed).
- **`align`/`merge_union`/`jitter_report` are pure** and offline-tested; only
  `run` touches the network.

---

## 8. Code map

| function | role |
| --- | --- |
| `run` | assign the selected subset, record offsets, align, write run artifacts |
| `_m0` | extract a ledger's M0 rows in the `_M0_COLS` schema |
| `_cluster_mz` | single-linkage gap clustering of an ascending m/z array |
| `align` | offset-aware cluster → merged consensus rows + long jitter frame |
| `merge_union` | just the merged frame from `align` |
| `jitter_report` | by-formula raw-vs-residual spread + by-m/z formula disagreements |
| `_theo_ppm` | observed-vs-theoretical ppm for an assigned (neutral, adduct) |
