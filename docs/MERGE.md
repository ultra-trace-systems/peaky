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
   consensus mz = mean(raw mz);  the files VOTE in two stages:
     which ION  -- most files wins (Assigned-file count, ion_score break ties;
                   a curated identity, Assigned somewhere, is never outvoted)
     which LABEL of it -- the reading Assigned in the most files (a same-ion
                   label split is the reagent-N isobar: Candidate = undecided)
   n_files, n_files_ion, n_files_winner, alternatives, tier_reason, srcs,
   ion_agree, formula_agree, mz_jitter_ppm_raw, mz_jitter_ppm_caldj
   │  (positive urea, ONCE on the merged ledger: relabel_reagent_n_adducts, then
   │   prefer_amine_over_ammonium -- each writes what it did to tier_reason)
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

4. **The vote** (`_vote`), in two stages. A *reading* is a
   `(neutral_formula, adduct)` pair; its *ion* is the element composition of
   neutral + adduct (`_ion_key`, the tier engine's `_ion_counts`), so
   `C13H14O4 [M+NH4]+` and `C13H17NO4 [M+H]+` are one ion, `C13H18NO4+`.
   - **Which ion** sits at the m/z is what files can genuinely disagree on, and
     the count decides it: the ion carried by the most **files** wins, then the
     number of files carrying it at **Assigned** tier (`TIER_RANK = {Assigned:2,
     Candidate:1}`, else 0), then the best `ion_score`, and last the ion's own
     text — so a full tie resolves identically whatever order the files arrived
     in (serial and parallel runs stay byte-identical). This is the order
     `collapse_trace_labels` already applies to competing labels on one trace.
     **Curated** exemption: an ion one of whose labels is a neutral in the
     `curated` set — a reference-list rescue or the pass-0 known-species list,
     gathered by `_curated_neutrals` from the per-file `method` — ranks first when
     that label reached Assigned in at least one file and in no fewer files than
     any grid ion did: exempt from the file count, not from corroboration, so a
     list identity is not outvoted by grid *guesses* while a grid ion Assigned in
     more files is a real contest the count decides. When it decides a cluster the
     merged `tier_reason` says so: `curated identity kept over the 9-file
     C27H30O14 [M+H]+ reading (vote 1 of 10 files)`. A `certified:` neutral is the
     file's own multi-channel evidence for a grid formula, already credited by its
     tier, and gets no exemption.
   - **Which label** of the winning ion is decided by corroboration, not by count:
     on a same-ion pair the tier engine marks a reading Assigned only when a
     discriminating channel was present in that file (an N-free sibling, the joint
     NH4+urea pair, a series anchor) and Candidate when it had nothing to decide
     with, so counting Candidate files would be counting silence. The label
     Assigned in the most files wins; file count and score break ties. When that
     overrides a bigger count the row says so: `same ion C13H18NO4+ read two
     ways: kept C13H14O4 [M+NH4]+ (Assigned in 5 of its 5 files) over the 6-file
     C13H17NO4 [M+H]+ (Assigned in 0)`.

   The winning reading's best per-file row (tier, then `ion_score`, then `src`)
   supplies the merged `neutral_formula` / `adduct` / `tier` / `ion_score` /
   `admitted_by` / `occurrence`. The merged **`mz` is the mean of the cluster's
   raw m/z**. Also recorded: `n_files` (distinct srcs), `n_files_ion` (files
   carrying the winning ion), `n_files_winner` (files carrying the winning
   reading), `alternatives` (every losing reading, best first, e.g.
   `C15H25N [M+H]+ x1 Candidate 0.97`; empty when unanimous), `srcs`,
   `ion_agree` (one ion in the cluster), `formula_agree` (`≤ 1` distinct
   neutral), and the two jitter spreads (§5).

   *Why a vote.* The previous rule ranked the Assigned-file count first, so one
   file's Assigned reading outvoted many files' Candidate reading of a different
   ion: on a 15-file Texas Ur⁺ run, 12 of 73 split clusters were decided by a
   minority, and nothing on the merged row said so. *Why two stages.* Of those 73
   split clusters 43 were two labels of one ion and 30 were different ions; in 16
   of the 43 a pure count would hand the ion to a label nobody had corroborated
   over one some file had. The curated exception keeps a list identity from
   losing to a grid majority of guesses: on the same run the D7 cyclosiloxane
   urea adduct at m/z 579.171 and tricresyl phosphate at 429.157, each locked in
   one file by the known-species list (mass + own-twin gate), faced an O14 /
   N4O10 grid formula the per-file engine itself flags as implausible (Candidate
   in every file). Sulfolane at 181.065, from the same list in one file, met
   fluorenone `C13H8O [M+H]+` Assigned in nine, and the count decided that one.

5. **Positive urea re-reads, once per batch.** When `prof.polarity == "+"` two
   gates run on the merged ledger, and each writes its note to the merged row's
   `tier_reason` (the column is always present, `NA` where no gate spoke):
   - `cleanup.relabel_reagent_n_adducts(merged)` — a pure hydrocarbon read via
     `[M+NH4]⁺` / uronium becomes `[M+H]⁺` of the N-heterocycle unless the
     hydrocarbon shows its own `[M+H]⁺` **anywhere in the batch**. The per-file
     stage is switched off for batch runs (`assign_kw["reagent_n_relabel"] =
     False`, honoured by `assign.run`): its skip key is a presence test that flips
     with each file's S/N, which split one ion into two readings across the files
     (C15H22 `[M+NH4]⁺` in 14 files, C15H25N `[M+H]⁺` in the 15th) — a phantom
     disagreement, and a phantom minority for the vote.
   - `cleanup.prefer_amine_over_ammonium(merged, ts_peaks, r_min = amine_r_min
     (0.6))` re-reads uncorroborated `[M+NH4]⁺` as `[M+H]⁺` of the `+NH3` amine
     (mass/isotope-identical, simpler in an N-rich source) unless the adduct's
     trace tracks its parent — done at the **merged** level where cross-channel
     corroboration is complete.
   `batch_summary.json["merge_gates"]` records both gates' counts.

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
| `TIER_RANK` | `{Assigned:2, Candidate:1}` | the vote's Assigned-file count (a tie-break after file count) and the best-row pick within the winning reading (then `ion_score`) |
| `align` `curated` | `_curated_neutrals` of the per-file ledgers (`method` starts with `reflist-rescue` or `known:`; `_CURATED_METHODS`) | a reading of one of these, Assigned in ≥ 1 file and in no fewer files than any grid reading, wins the vote outright |
| `_M0_COLS` | `[mz, neutral_formula, adduct, tier, ion_score, admitted_by, occurrence]` | the per-file M0 schema aligned (the last two = admission provenance, carried for the winning row; absent columns are tolerated) |
| `run` `amine_r_min` | 0.6 | min trace correlation for the positive amine re-read |
| `assign_kw` `reagent_n_relabel` | `False` (set by `run`) | the per-file hydrocarbon-on-N-cluster re-read stands down; `run` applies it once to the merged ledger |
| `run` `k_min` / `k_max` / `min_gain` / `min_prevalence` | 6 / 30 / 0.005 / 2 | passed through to `sampling.select_cover_samples` (see [`SAMPLING.md`](SAMPLING.md)) |

---

## 5. Metrics, defined

- **`mz_jitter_ppm_raw`** — `(max − min)/mean · 1e6` over a cluster's **raw**
  m/z; the total file-to-file mass spread.
- **`mz_jitter_ppm_caldj`** — the same spread over the **offset-corrected** m/z;
  what remains after per-file calibration is removed (the genuine noise).
- **`formula_agree`** — `True` iff the cluster carries ≤ 1 distinct neutral formula.
- **`n_files_ion`** / **`n_files_winner`** — files carrying the winning ion / the
  winning reading; `n_files` is the whole cluster. `n_files_ion < n_files` marks
  a contest between different ions, `n_files_winner < n_files_ion` a label
  chosen by corroboration over a bigger count.
- **`ion_agree`** — `True` iff the cluster carries one ion (element composition of
  neutral + adduct). `ion_agree` and not `formula_agree` = a same-ion label split
  (the reagent-N isobar), not a spectral disagreement.
- **`alternatives`** — the losing readings, best first, each as
  `formula adduct xN tier score` (the best tier / score any file gave it);
  `''` when unanimous. The per-file detail is `tables/jitter.csv`.
- **`tier_reason`** (merged) — what the vote and the batch-level gates did to the
  row: the curated exemption or a corroboration-over-count label choice when one
  decided the vote; the reagent-N re-read,
  naming the reading it replaced; the amine gate, naming the `[M+NH4]⁺` neutral it
  did not confirm and why. `NA` when nothing needed saying.
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
| `merged_ledger.csv` (run root) | one row per m/z cluster: consensus mz, the winning reading, the vote (`n_files`, `n_files_ion`, `n_files_winner`, `alternatives`), `srcs`, `ion_agree`, `formula_agree`, `mz_jitter_ppm_raw/caldj`, the batch-level gates' `tier_reason`, plus the trace reconciliation columns (`mz_anchor`, `mz_trace`, `trace_offset_ppm`, `trace_cov_anchor`, `trace_cov`, `trace_moved`, `trace_guarded`, `trace_id`, `trace_role`; [`TIMESERIES.md`](TIMESERIES.md) §9) — **the result** |
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
- **Consensus = a vote, in two stages.** The count picks the ION (tier and score
  only break ties; a curated identity is exempt from the count), corroboration
  picks its LABEL (the reading Assigned in the most files). The losers stay on
  the row (`alternatives`) and in `jitter.csv`; `formula_agree` stays `False` on
  a split, and `ion_agree` tells a label split from a real contest.
- **The two positive-mode re-reads are merged-level** — the amine gate needs the
  full cross-channel picture, and the hydrocarbon-on-N-cluster re-read needs every
  file's `[M+H]⁺` rows at once; per file, the latter split one ion into two
  readings across the batch. Both say what they did in the merged `tier_reason`.
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
| `align` | offset-aware cluster → the vote per cluster → merged rows + long jitter frame |
| `_ion_key` | a reading's ion (neutral + adduct composition, Hill order, charge sign): the key two labels of one ion share |
| `_vote` | rank one cluster's ions (curated, file count, Assigned-file count, best score, text) and, within each, its labels (Assigned-file count, file count, best score, text) |
| `_describe` | one losing reading as `formula adduct xN tier score` for `alternatives` |
| `_curated_neutrals` | the reflist-rescue / known-species neutrals of a per-file ledger (the vote's exemption) |
| `_protected_neutrals` | those plus the certified neutrals (the amine gate's exemption) |
| `merge_union` | just the merged frame from `align` |
| `jitter_report` | by-formula raw-vs-residual spread + by-m/z formula disagreements |
| `_theo_ppm` | observed-vs-theoretical ppm for an assigned (neutral, adduct) |
