# Peaky — Outputs reference

Where Peaky writes things, and what each artifact is. The batch run-folder layout
is the single source of truth in [`peaky/paths.py`](../peaky/paths.py) (`RunPaths`),
shared by the writers and the report reader so the filenames can't drift.

---

## Batch run — one versioned folder per run

`peaky batch` creates **one timestamped folder** under `--out-dir`
(default `~/peaky-output`), so a re-run never overwrites a previous one:

```
<batch-slug>_<YYYY-MM-DDTHHMMSSZ>/     ← folder name == Report ID (UTC stamp)
```

### Run root (flat — read by several modules + the cross-run registry)

| Artifact | What it is / what it's for |
|---|---|
| `merged_ledger.csv` | **The result.** Every merged peak (one row each): role, neutral formula + adduct, scores, ppm, confidence, tier, provenance, the winning row's `admitted_by` / `occurrence` and, with a time series, its `ts_disposition` / `ts_cv_norm` (a background label: flatness never changes the tier), the files' vote on the reading (`n_files_ion` / `n_files_winner` of `n_files`; `ion_agree`; the losing readings in `alternatives`), what the vote, the batch-level known-species decision ([MERGE.md](MERGE.md) §3 step 4b) and the re-reads did to the row (`tier_reason`), `stage` (`cover` = some cover file holds the ion; `residual` = only the residual stage's files do — present whenever that stage is on, i.e. unless `--no-residual`), and the trace reconciliation (`mz_anchor` = the merge's m/z, `mz_trace` = the ion's trace centre the stamp uses, `trace_offset_ppm`, `trace_cov_anchor` / `trace_cov` = share of spectra with a peak within tolerance of each, `trace_moved`, `trace_guarded`, `trace_id`, `trace_role` ∈ `single` / `winner` / `collapsed` — a collapsed row is a competing label for a trace another row won; it is kept, flagged, and never stamped). The provenance anchor. Plus the **evidence level** of every row (`evidence_level` 2b…5b, `evidence_axes`, `level_reason`, `n_plausible_structures` — [EVIDENCE_LEVELS.md](EVIDENCE_LEVELS.md)): recomputed on the pooled per-file ledgers and stamped by `(neutral_formula, adduct)`; `NA` on a merged row whose reading no per-file ledger holds (a batch-level re-read). And the **claim** that level supports on every row (`claim`: `identified` = 1–4a, the neutral established; `ion` = 4b–4d, the ion composition pinned; `tentative` = 5a, 5b or no level, so a batch-level re-read reads `tentative` — [EVIDENCE_LEVELS.md](EVIDENCE_LEVELS.md) §1.1): a verdict beside `tier`, not a replacement for it; the two can disagree and neither changes the other. And `ion_only_of` (the winner file's `[M-H]-` parent peak) on the **ion-only** rows — the `[M]-.` electron-attachment line beside a committed acid, a Candidate carrying the acid's composition (stage `ion_only`; levelled 4d / 5a on its own ¹³C; `NA` elsewhere). And the winner file's peak separability, `resolvability` (`isolated` / `resolved` / `blended` / `unresolvable`) and `sep_hwhm`, when the run had a width model. |
| `run_manifest.json` | **Reproducibility manifest.** Pins the run to its exact code (package + per-module version + content hash + git commit), input-data hash (`ts_sha1`), resolved config (the user knobs incl. `height_cutoff_x_edge` and `occurrence_min`; run-derived fields such as `occurrence_threshold` are dropped), the `selection` and `admission` blocks under `counts` (so the resolved persistence threshold is recorded next to the config fingerprint) beside `merged_tiers` and `merged_claims` (the merged rows per claim), and output hash (`merged_ledger_sha1`). |
| `batch_summary.json` | Run counts + per-file calibration offsets (M0/tier counts, n_files, offsets; per-file `noise_edge_cps` + the resolved gate `height_gate_cps` + the `height_cutoff_x_edge` it came from + `admitted` counts by path), the batch-level `height_cutoff_x_edge` + `height_cutoff_x_edge_source`, the **`admission`** block (`occurrence_min` = the knob, `'auto'` or a number; `occurrence_threshold` = the resolved fraction, `null` when the path is off; `n_peaks` = rows of the per-peak occurrence table; `n_persistent_peaks` = those at or above the threshold; `n_persistent_traces` = the trace-weighted count of those, ≈ the number of persistent ions; `n_spectra`; `tol_ppm`), the **`gate`** block (empty when the multiple was pinned by a flag / cfg / profile or could not be derived; else the batch-derived brightness floor: `x_edge`, `transient_share` per grid multiple, `share_at_1`, `share_at_x`, `max_transient_share`, `n_peaks`, `bound`, `source`), the **`traces`** block (empty without a time series: `n_rows`, `n_recentred`, `n_guarded`, `median_abs_move_ppm`, `mean_cov_anchor` / `mean_cov_trace`, `n_traces`, `n_collapsed`, `n_multi_label_traces`, `sigma_ppm` = the third-quartile per-ion scatter, `stamp_tol_ppm` = the window the stamp used) and the **`selection`** block: `k`, `n_bins`, `achieved_coverage`, `stop_reason` (`gain-floor` / `k_max` / `exhausted`), `next_gain`, `tol_ppm` (the binning tolerance, = the admission table's and the merge's), `k_min`/`k_max`/`min_gain` (+ `coverage_by_group` for a pool), and — when the residual stage is on — its **`residual`** sub-block (`n_bins_residual`, `n_universe`, `n_uncovered`, `n_explained`, `n_below_floor`, `n_sidelobe`, `n_suspect`, `floor` = {`min_x_edge`, `min_cps`, `edge_median_cps`, `source`}, `frac_of_max`, `k`, `k_max`, `coverage_of_residual`, `stop_reason` ∈ `exhausted` / `k_max` / `empty`, `next_gain`, `sample_ids`; `skipped` when there was no time series), plus `n_files_by_stage` and `merged_by_stage` at the top level and a `stage` on every `per_file` row. Plus the **`evidence_levels`** block: `pooled` (one count per neutral/adduct pair over the pooled files), `merged` (per merged row), `per_stage` (merged rows by `cover` / `residual`), `n_pairs`, `n_unstamped`, `n_corroborate`, `cross_source` (the `--corroborate` sources), `neutral_pairs` (rule U's funnel: `pair`, `neutrals`, `committed_both`, `cho`, `present`, `covary`, `clean`, `upair`; `{pair: [], neutrals: 0, upair: 0}` on a profile without a pair), `label_twins` (rule K's funnel on a 15N-labelled nitrate channel: `in_scope`, `f` (the twin fraction), `lines`, `codetected`, `references`, `cluster_k`, `committed_lines` and `lines_<verdict>` per 14N line verdict (`tracks`, `consistent`, `excess`, `absent`, `untestable`), `alien`, `untie`, `readings`, `testable`, `passes`, `refuted`, `lines_refuted`; `{in_scope: false, lines: 0, untie: 0, readings: 0, refuted: 0}` out of scope). Plus the **`claims`** block right after it (each count a `{identified, ion, tentative: n}` dict, zeros kept): `merged` (per merged row), `pooled` (per pair), `per_stage`, `by_tier` (merged rows per tier, Assigned first; the ion-only rows under their own `ion-only` key, not their tier) and `n_unlevelled` (merged rows with no level, read tentative). Plus the **`ion_only`** block: `channels` (the ion-only adducts the profile opened, `[]` when none), `merged` (merged rows carrying an `ion_only_of` link), `per_file_rows`, `n_files_with`, `merged_levels` (their levels, 4d / 5a). Plus **`merge_gates`**: `known` (the batch-level known-species decision — `pooled` / `locked` / `confirmed_kept` / `conflict` / `lead_only` / `mass_only_outvoted` / `no_cluster`), `reagent_water` (the reagent-water ladder, [MERGE.md](MERGE.md) §3 step 4a: `n_cores`, `tol_ppm`, `segments` = spectra per acquisition segment, `n_rungs`, `rungs_by_core` = the passing n per core, `n_stripped` and `stripped` = the merged readings that sat on a passing rung) and, on a positive channel, `reagent_n` and `amine`. Plus **`resolution`** (the peak-width model the per-file resolvability stamp used — `coef`, `exponent`, `offset`, `r_at_200`, `r_at_600`, `n_peaks`, `r_spread`, `source` ∈ `measured` / `declared`; `null` when not stamped) and **`resolvability`** (the per-file M0 class counts summed over the files). |
| `per_file/<sid>_ledger.csv` | The full single-sample ledger for **each** assigned sample, kept for audit / re-merge. Carries the admission provenance per peak: `occurrence` (float in [0, 1] — the fraction of the batch's spectra holding a peak within tolerance of this peak's m/z, one per spectrum; NaN without batch context or when no batch peak lies within tolerance) and `admitted_by` (`height` / `occurrence` = persistence only / `''` = not eligible at the eight gated sites — the pass-1 grid, pass-2 series growth, pass-3 contaminant families and pass-3's two cluster resolvers, the pass-6 ladder gap-fill, residual stage B and the siloxane ladder; residual stage A, the reflist rescue, pass-3's series *detection* statistics and the postprocess satellite tests are still brightness-only, so an empty stamp says nothing about them). Each also carries its own per-file evidence level (`evidence_level` / `evidence_axes` / `level_reason` / `n_plausible_structures`, computed on that file alone; the batch level on the merged ledger is the pooled one) and the `claim` it supports, on the M0 rows only (`NA` on isotope children, reagent ions and unexplained peaks) and `ion_only_of` (the parent peak of an ion-only `[M]-.` row; `NA` elsewhere). And `known_lead`: the known-species claim pass 0 anchored on-cal in this file but did not commit, as JSON (formula, family, label, adduct, ion, verdict `deferred` / `refuted`, `why` with the file's numbers, `summary` without them); `NA` elsewhere. The batch pools it ([MERGE.md](MERGE.md) §3 step 4b). And the separability stamp on every M0 row: `resolvability`, `sep_hwhm` (distance to the nearest picked peak, in HWHM at that mass), `d_crit_hwhm` (the separation at which two peaks of that height ratio become bimodal); `NA` without a width model. And the two assignability flags, bool: `below_assignability` (the assignment argues with itself) and `tentative_lead` (the proposal is unsupported, not contradicted; C19(c) split it off `below_assignability`), both read as 5b by the level ([EVIDENCE_LEVELS.md](EVIDENCE_LEVELS.md) §2); a ledger written before the split has no `tentative_lead` column. |
| `index.jsonl` | **At the `--out-dir` base, not inside the run folder.** Cross-run registry — one compact row per run, loadable with `pandas.read_json(lines=True)` to find or diff runs. |

### `figures/` — all `.png`

| Artifact | What it is |
|---|---|
| `van_krevelen_<tag>.png` | Van Krevelen of the assigned analytes (Si excluded — the clean atmospheric view). |
| `van_krevelen_full_<tag>.png` | Full Van Krevelen — every assigned peak by CHO/CHON/CHOS backbone (Si/F/halogen folded in). |
| `clusters_changing_<tag>_p*.png` | Correlation-cluster panels (A4 portrait, paginated) of the dynamic, co-varying analyte families. |
| `clusters_flat_<tag>_p1.png` | The uncorrelated/flat remainder + Si contamination, bunched into one overview. |
| `clusters_changers_<tag>_p*.png` | Big standalone changers — single channels that move ≥~5–10× on their own. |
| `clusters_unassigned_<tag>_p*.png` | The same clustering applied to the **unexplained** residual. |
| `gka_<tag>.png` | The static GKA findings page embedded in the report. |

### `tables/` — all `.csv` / `.xlsx`

| Artifact | What it is |
|---|---|
| `selected_samples.csv` | Which samples were assigned + why, in pick order: `pick`, `role` (`cover` = a greedy pick, `pad` = richest-TIC pad up to `k_min`, `residual` = a residual-stage pick, numbered on after the cover's), `bins_new` (m/z bins this pick covered first — its marginal gain; for a residual pick, residual bins covered at ≥ 50 % of their maximum) and `coverage` (cumulative fraction of the batch's bins covered; of the residual universe for a residual pick). |
| `residual_bins.csv` | The residual stage's candidates (written whenever the stage is on): one row per universe bin in no assigned file, unexplained by the stamp and above the floor — `bin`, `bin_mz`, `prevalence`, `max_cps`, `max_x_edge` (the maximum as a multiple of its sample's edge), `sample_at_max`, `tier` (`clean` / `suspect` = targeted, suspects last; `sidelobe` = a confirmed ringing sidelobe of a saturating neighbour, dropped), `sidelobe_of` (that neighbour's m/z), `n_pairs` (samples holding both), `ratio_cv` (the height-ratio cv when measurable), `covered_by` (the residual pick that carries it; empty if none did). |
| `evidence_levels.csv` | One row per `(neutral_formula, adduct)` pair of the pooled per-file ledgers with its `evidence_level` and every fact behind it (`iso`, `multiline`, `multiline_elements` (the elements whose in-band lines made it, C17), `carbon_ev`, `chan2`, `anchor`, `branch`, `reagent_only_iso`, `ion_only`, `upair` (rule U), `label_untie` / `label_veto` / `label_note` (rule K), `iso_labels`, `tied`, `below`, `lead` (any row a tentative lead, C19(c)), `lowconf`, `degeneracy`, `saturated`, `res_ok`, `resolvability`, `n_files`, `known_fam`, `corroborated`, `n_axes`, `n_plausible_structures`, `evidence_axes`, `level_reason`) and the pair's `claim` — the audit trail of the merged ledger's level, the same table `scripts/level_ledger.py` writes post hoc. |
| `neutral_pairs.csv` | Rule U (docs/EVIDENCE_LEVELS.md §3 `upair`, §4 row 9′): one row per neutral committed under an adduct of the profile's neutral pair (the uronium profile: `[M+H]+` and `[M+(CH4N2O)H]+`), every clause measured on the stamped batch time series — `committed_both`, `cho` (C/H/O only), `det_bare` / `ppm_bare` / `det_cluster` / `ppm_cluster` (share of spectra with a peak within 2 ppm, median ppm), `n_both` / `r_log` (the co-variation), `clean`, `nC_bare` / `nC_cluster` / `nC_obs_*` / `c13_scale` / `c13_contradicts` (the area-¹³C carbon check), `present`, `covary` — and the verdict `upair`. Written on every run; empty (header only) on a profile without a pair or a run without a time series. |
| `label_twins.csv` | Rule K (docs/EVIDENCE_LEVELS.md §3 `label_untie` / `label_veto`): on a 15N-labelled nitrate channel, one row per 14N line of every neutral committed on either nitrate adduct (`adduct` = `[M+NO3]-`: `committed` (on `[M+NO3]-`), `mz` / `mz_partner`, `n14` / `n_codetected` / `partner_share` (spectra with the 14N line, with both lines, their ratio), `h15_median`, `reference`, the cluster-k statistics `k_ratio` / `k_sd` / `k_r` / `k_n`, `cluster_k`, `line_verdict` (tracks / consistent / excess / absent / untestable), `alias_only_tie`, `untie` (the line would clear the reading's tie; the pooled level's `label_untie` records where it did), `alien`, `veto` and its `note`) and one per committed `[Y+^NO3]-` reading (`f`, `E` (expected twin sightings), `obs`, `twin_verdict` ∈ passes / unclear / refuted / untestable, `veto` and its `note`), measured on the stamped batch time series. Written on every run; empty (header only) out of scope or without a time series. |
| `reagent_water.csv` | The batch's reagent-water ladder ([MERGE.md](MERGE.md) §3 step 4a), one row per passing rung core.(H2O)n: `core`, `iso_tag` (the halogen isotopologue, empty for none), `n`, `ion_formula`, `mz` (exact) and `mz_obs` (median of the matched peaks), `segments` / `presence` / `decoy_presence` (the segments it passed in, `|`-joined), and `displaced` (the merged readings that sat on it and left the merged ledger). Header only when the profile declares no water cores or no rung passes. |
| `jitter.csv` | Per-(cluster, file) table: every file's own reading (formula, adduct, tier, `ion_score`, `evidence_level`) of each merged peak — the per-file detail behind the merged row's vote, whose ion stage ranks by the evidence class before the file count — and its m/z for the raw vs calibration-adjusted jitter. |
| `van_krevelen_full_<tag>.csv` | The full-VK data behind the figure (one row per assigned neutral). |
| `clusters_changing_<tag>.csv` / `.xlsx` | Cluster membership; the XLSX has one tab per cluster (formula / channel / m/z / match_score / tier). |
| `clusters_flat_<tag>.csv`, `clusters_changers_<tag>.csv`, `clusters_unassigned_<tag>.csv` | Membership for the flat / changers / unassigned figures. |
| `channel_agreement_<tag>.csv` | QC: how often a multi-channel neutral's ion channels agree in time. |
| `plausibility_audit_<tag>.csv` | One row per peak the hardened plausibility layer touched (demoted or relabelled): `before_tier`, `after_tier_or_role`, the `reason`, the supporting `evidence` (O/C or DBE/C or series r), the `degeneracy_note`, and `n_iso`. Always written (header-only when nothing was touched). |
| `predicted_satellites.csv` | One row per **predicted** isotope-satellite line of the batch stamp that had a candidate peak (`timeseries.annotate_peaks`): `ion_formula` (the parent's), `iso_label`, `ion_mz` (the predicted m/z), `iso_rel` (predicted height relative to the parent), `n_candidates` (peaks in the window), `n_eval` / `n_pass` (samples in which the parent was stamped and a candidate sat on the line / passed the height gate), `pass_share`, `judged`, `kept` (the track-coherence verdict) and `n_stamped`. Always written (header-only when no line had a candidate). |

### `report/` — the PDF

| Artifact | What it is |
|---|---|
| `report_<run-id>.pdf` | **The standard iterable A4 report** (cover · claims · findings · coverage · evidence levels · composition · scrutiny · reference lists · GKA · mass-defect QC · families · changers · clusters · methods · assignments appendix). The cover shows the Report ID + a date+time "generated" line. On a run with levels its summary opens on a claims line (merged rows per claim and, with the per-file ledgers on disk, each claim's share of the committed-peak signal), and the **Claims** page follows the cover: merged rows and committed signal per claim — every per-file M0 height credited with the merged claim of its `(neutral_formula, adduct)`, the readings no merged row carries an explicit `no merged row` bucket, never folded into tentative — the tier × claim crosstab (Assigned / Candidate / ion-only) and the brightest rows where tier and claim part. The findings' top species and the appendix carry a claim column; a run without levels gets no Claims page. |
| `report_<run-id>_compressed.pdf` | Optional size-reduced companion for emailing (needs `pip install mascope-peaky[compress]`). The full report is left byte-for-byte untouched. |

### `data/` — bulky inputs kept with the run

| Artifact | What it is |
|---|---|
| `<tag>_ts.parquet` | The full-batch per-sample peak time series — written here **only** when fetched live (no on-disk source). A parquet passed by `--ts` is *referenced*, never copied. |

---

## The time-series parquet — schema

Two files carry the batch time series. Both are **one row per (sample × peak)** —
a long/tidy table, not a matrix — so a 995-sample batch with ~2400 peaks each is
~2.4 M rows.

| File | Content |
|---|---|
| `per_file/_batch_ts.parquet` | The **annotated** series: raw peaks **+ the assignment columns** below. This is the one to hand to downstream software. |
| `data/<tag>_ts.parquet` | The **raw** series only (no assignment columns), kept when the TS was fetched live. |

**Raw columns** (from Mascope; exactly which are present depends on how the TS was
fetched — `sample_item_id`, `mz` and `height` are always there):

| Column | Arrow type | Meaning |
|---|---|---|
| `sample_batch_name` | `large_string` | Batch the sample belongs to. |
| `sample_item_id` | `large_string` | **Sample key** — one acquisition file. |
| `sample_item_name` | `large_string` | Human-readable sample name. |
| `datetime_utc` | `timestamp[us, tz=UTC]` | Acquisition time — the x-axis of every trace. |
| `peak_id` | `large_string` | Mascope's per-sample peak id. Unique *within* a sample; **not** stable across samples, so it cannot be used to join a peak to the same peak in another file. |
| `mz` | `double` | The peak's **raw fitted** m/z **in that sample** — it jitters sample to sample and is *not* the calibrated mass. |
| `height` | `double` | Peak height (cps) — the quantity to plot. |
| `area` | `double` | Integrated peak area. |
| `sparsity` | `double` | Mascope peak-shape/quality metric. |

**Assignment columns** (added by `timeseries.annotate_peaks`, `_batch_ts.parquet`
only). All are `<NA>`/`NaN` on a peak that matched no known ion:

| Column | Arrow type | Meaning |
|---|---|---|
| `neutral_formula` | `large_string` | Assigned neutral formula, e.g. `C8H4O3`. Analyte M0s only. |
| `adduct` | `large_string` | Ionisation channel, e.g. `[M+H]+`, `[M+I]-`. |
| `tier` | `large_string` | `Assigned` (trust it) or `Candidate` (tentative). |
| `ion_mz` | `double` | The ledger m/z this peak was matched to — the ion's **trace centre** (`mz_trace`, the merge anchor re-centred on the batch's own trace; the anchor itself is the ledger's `mz_anchor`). Join key: all rows sharing an `ion_mz` are the same ion. |
| `role` | `large_string` | What kind of known ion: `M0` (analyte), `reagent` (reagent-cluster ladder), `iso_child` (heavy-isotope satellite), `artifact` (FT ringing ghost — not an ion at all). `<NA>` = unknown track. |
| `ion_formula` | `large_string` | **The detected ION's formula for EVERY identified ion, analyte or not** — `CH3IO2-` (analyte), `I3-` (reagent), the parent's ion formula on an isotope satellite. `ion_formula.notna()` = identified; `neutral_formula.notna()` = analyte with a molecular reading. In a reagent-dominated spectrum this is the column that shows the file is ~97 % signal-characterised, not ~20 %. |
| `iso_label` | `large_string` | Isotopologue qualifier: `13C`/`81Br`/… on satellites, the reagent line's tag (`79Br+81Br`, `127I+127I`) on multi-isotopologue reagent formulas — so one `ion_formula` can carry several distinct heavy lines without colliding. |
| `stamp_source` | `large_string` | Where the stamped identity came from: `M0` (a merged analyte), `observed` (an ion a per-file ledger identified: the reagent ladder, a claimed satellite, an artifact) or `predicted` — a diagnostic isotope satellite (13C / 81Br / 37Cl / 15N / 34S / 29Si / 30Si / 18O) predicted from its parent's ion formula that no per-file ledger claimed, stamped only where the parent was stamped in the **same sample** and this peak's height / (parent height × predicted relative abundance) lies in 0.3–3.5, the per-file passes' own window (see below). `<NA>` on an unstamped row. |
| `dup_candidate` | `bool` | `True` for a peak that fell inside an ion's mass window but **lost** the one-to-one contest. Its identity columns stay `<NA>`. An audit trail — the row is never dropped. |
| `intensity_suspect` | `bool` | **Trust the formula, do not quantify this channel.** The ion's m/z lands on the ringing sidelobe of a saturating neighbour, so the height here is the neighbour's, not the analyte's. Carried from the merged ledger's own column. |

### Predicted satellites

A per-file ledger claims a satellite only where that file's picker picked it, and
the faint diagnostic lines — 15N (0.36 % per N), 18O (0.20 % per O), a single
34S / 29Si / 30Si — sit below the picker's edge in most files. So a parent
Assigned in every assigned file could still leave its 15N / 18O tracks
unexplained wherever a plume lifted them into view. `timeseries.stamping_frame`
therefore adds one **predicted** `iso_child` row per (parent, label) for the
diagnostic lines of every merged M0 with a known `ion_formula`, at the parent's
stamped m/z (its trace centre, so the instrument offset carries over) plus the
line's exact shift. Precedence is fixed: a satellite a per-file ledger observed
supersedes the predicted one, and a predicted line is never minted on a track an
M0 / reagent / observed satellite / artifact already holds — an assigned analyte
at a satellite offset is an analyte. `annotate_peaks` then matches every known
row **first** (a peak inside any known row's window is never offered to a
predicted line) and stamps a predicted line only under the intensity gate above,
plus a **track-coherence** rule the per-sample gate cannot express: a true
satellite's ratio is a constant of nature and passes the window in nearly every
judged sample, whereas an independent compound on the line fails in most and
passes in the few where its height happens to fit. Once a line has been judged in
at least 10 samples (parent present, a candidate on the line) it keeps its stamps
only if at least half of them passed; otherwise the whole track stays unexplained.
`tables/predicted_satellites.csv` holds one audit row per predicted line that had
a candidate (samples judged / passed, pass share, judged, kept, peaks stamped).
`batch_summary.json['traces']['stamp']` counts the stamped peaks by source
(`n_iso_observed` apart from `n_iso_predicted`), the predicted tracks stamped /
judged / rejected, and the predicted rows minted and superseded. The per-file
ledgers and their coverage figures are untouched.

### The one-to-one guarantee

Within one sample, a given ion — `(neutral_formula, adduct)` for analytes,
`(role, ion_formula, iso_label)` in general — is stamped on **at most one
peak**. So

```python
df = pd.read_parquet("per_file/_batch_ts.parquet")
# analyte quantification (M0 channels only — satellites deliberately carry no
# neutral_formula, so per-neutral sums cannot double-count them):
trace = (df[df.neutral_formula.notna() & ~df.dup_candidate]
           .groupby(["neutral_formula", "adduct", "datetime_utc"])["height"].sum())
# every known ion, reagent ladder included:
ions = (df[df.ion_formula.notna() & ~df.dup_candidate]
          .groupby(["role", "ion_formula", "iso_label", "datetime_utc"],
                   dropna=False)["height"].sum())
```

yields exactly one point per ion per sample — no double counting.

This has to be enforced because the stamp is a **mass match**, not a peak-identity
join: the merged ledger holds one row per ion with no peak ids, and only ~6 of a
batch's samples are ever assigned, so the other ~989 have no per-peak decision to
carry over. Left unconstrained the match is many-to-one — a shoulder or split peak
inside the same window gets stamped with the same formula as the real peak
(measured: 2385 duplicated (sample, ion) pairs, 61 ions, on a 2.4 M-row uronium
batch). The **assignment itself never does this** (verified: 9784 per-file M0 keys,
zero owned by more than one peak — the shoulder is left `unexplained`), so the
duplication was purely an artifact of the re-match. Two rules restore it:

1. **One-to-one** — per `(sample, ion)` keep the single best peak; the rest get
   `dup_candidate = True`.
2. **Consensus** — "best" means nearest the ion's *consensus* m/z, not the bare
   ledger mass. Without this the winner flips between two raw tracks sample by
   sample — whichever happens to be present — splicing two different peaks into one
   trace (measured: 232 and 378 flips for two ions). The consensus is built by
   splitting an ion's candidates into tracks (a gap wider than `halfwin` starts a
   new one) and picking one by two rules:
   - **A track is scored by its BRIGHTEST member, not its summed height.** Summing
     conflates brightness with prevalence, and an **FT ringing sidelobe** of a bright
     neighbour is ubiquitous-but-dim — it recurs beside its parent in *every* sample.
     Summed height handed `C12H19NO6 [M+H]+` to its sidelobe track (1576 cps × 559
     samples) over the real peak (2390 cps × 70).
   - **The ledger mass is anchored.** Offset 0 is where the *assignment* committed
     the formula, so the track holding it is displaced only by one at least
     `ANCHOR_MARGIN` (2×) brighter. `C19H34O6Si [M+NH4]+` clears that bar (1205 vs
     473 cps) and correctly moves; `C14H28O3Si [M+H]+` at 1.86× does not.

Both default on; `annotate_peaks(..., one_to_one=False)` / `consensus=False` restore
the raw behaviour.

> **Known residual — a stamp is a mass match, not proof of identity.** Where a
> sample's real peak is **absent**, a neighbour inside the tolerance still collects
> the stamp, and that neighbour is sometimes an FT ringing sidelobe. Scale on the
> On a 1.73 M-row uronium batch: **0.28 %** of stamped rows (4777 of 1.73 M) sit >1 mDa from
> their ledger mass, and **10 ions of 2127** span more than 0.5 mDa across the
> batch. The worst of these — where the channel's whole intensity is a
> neighbour's sidelobe — are now detected and marked `intensity_suspect`
> (see below); the rest are visible as a wide `(mz - ion_mz)` spread.

### Sidelobe-contaminated channels (`intensity_suspect`)

A saturating peak **rings**: FT/Gibbs sidelobes sit a few mDa either side of it at
a roughly fixed fraction of its height. When an assigned ion's m/z lands on one,
the *formula* can still be right while the *height* is the neighbour's.
`C18H30O6` is the worked example — clean on `[M+H]+` at m/z 343.211, but its urea
adduct at 403.244 rides 11.5 mDa from a 520 000-cps `C20H34O8` peak at a locked
0.71 % of it. A trace built from that channel tracks `C20H34O8`, not `C18H30O6`.

**Static features cannot detect this.** Over 25 498 raw tracks across 30 campaign
runs, contaminated channels are *indistinguishable* from real ions that merely sit
near a bright peak:

| | contaminated | real, near a bright peak |
|---|---|---|
| satellite fraction of parent | 0.69 % | 0.23 % *(smaller!)* |
| \|Δm/z\| to parent | 11.5 mDa | 10.1 mDa |
| **ratio-to-parent cv (time series)** | **0.033 – 0.051** | **0.21 – 1.09** |

Only the time series separates them: a sidelobe holds a near-constant ratio to its
parent; an independent ion varies on its own. So `timeseries.flag_sidelobe_channels`
runs at **merge level**, where the batch TS exists — not in per-file cleanup — and
sets `intensity_suspect` plus `sidelobe_parent_mz` on the merged ledger.
`SIDELOBE_CV = 0.08` sits in the empty gap, biased to under-flag. Scored against
that labelled set: **6/6 contaminated channels caught, 0 false positives of 72**.

The assignment is **never** altered — no retraction, no tier change — because the
neutral is usually real and corroborated on another channel. Only quantification is
in question:

```python
df = pd.read_parquet("per_file/_batch_ts.parquet")
quant = df[df.neutral_formula.notna() & ~df.intensity_suspect]   # safe to integrate
```

---

## Single-sample run — `peaky assign`

Writes into `--output-dir` with the prefix `<sample-id>_<YYYYMMDD-HHMM>`:

| Artifact | What it is |
|---|---|
| `<prefix>_ledger.csv` | Every peak: role (`M0` / `iso_child` / `reagent` / `artifact` / `unexplained`), formula, adduct, all scores (incl. arbitration `eff_score`/`eff_margin`/`tied`), ppm, confidence, `tier` + `tier_reason`, candidate/degeneracy density, provenance, commentary, alternatives, isotopologues. Plus the evidence level (`evidence_level` 2b…5b, `evidence_axes`, `level_reason`, `n_plausible_structures` — [EVIDENCE_LEVELS.md](EVIDENCE_LEVELS.md)) and the `claim` it supports (`identified` / `ion` / `tentative`, on M0 rows; `NA` elsewhere) and `ion_only_of` (the `[M-H]-` parent peak of an ion-only `[M]-.` row; `NA` elsewhere), and `known_lead` (a known-species claim pass 0 anchored on-cal but did not commit, as JSON with its verdict `deferred` / `refuted` and why; `NA` elsewhere), and the separability stamp (`resolvability` / `sep_hwhm` / `d_crit_hwhm` on M0 rows, from the run's width model; `NA` without one), and the two assignability flags `below_assignability` / `tentative_lead` (bool; [EVIDENCE_LEVELS.md](EVIDENCE_LEVELS.md) §2). |
| `<prefix>_assignments.xlsx` | The styled multi-sheet workbook: **By claim** · Summary · Read-me legend · **Assigned** · **Candidates** · By evidence level · Below assignability (the M0 rows either assignability flag marks, with a `tentative_lead` column saying which; a ledger with neither flag column has it empty) · Unassigned (evidence-characterized) · By class · Unique formulas · Isotopologues · Peak ownership · Target list · Reagent ions. Frozen headers, autofilters, tier/confidence/evidence-level/claim color chips; `evidence_level` on the Assigned, Candidates, Target list and Peak ownership sheets, with `claim` directly before it (the Candidates sheet's rank-1 row only: an alternative makes no claim) and `best_claim` beside `best_tier` on Unique formulas. **By claim**, the sheet the workbook opens on, holds one `summary` row per claim (meaning, levels, count and share of the M0 rows, summed height and its share, the Assigned / Candidate / ion-only split, the level histogram), then the `tier disagrees` rows (Assigned but tentative, Candidate but identified; brightest first) and the twenty brightest rows of each claim. The Summary gains a **Claims** section between Coverage and Tiers (count, share of assignments and of assigned signal per claim) and the Read me opens on the claim classes — including that the By class sheet's `n_identified` counts tier-Assigned rows, a column name older than the claim. A ledger without levels gets none of this and renders as before; one with levels but no `claim` column gets the claim read off the level. |
| `<prefix>_summary.md` | Narrative + top assignments + coverage; with levels, a `Claims:` line before the tiers and each top assignment tagged with its tier and claim. |
| `<prefix>_manifest.json` | Module versions, prescan fingerprint, series-evidence table, per-pass timing, and the stage summaries (the `evidence` stage's `claims` tally among them). |
| `<prefix>_gka.html` | Interactive rotating-GKA widget (self-contained, no server). |
| `<prefix>_gka_unexplained.html` | The same widget over the **unexplained residual only** — the place to hunt for missed homologous structure. |
| `checkpoints/` | Per-pass ledger checkpoints (an audit trail of what each pass committed). |

---

## Reproducibility note

Every figure, table, and ledger above is a **pure function of the input data** —
byte-identical whenever you re-run the same data. The **only** thing the run
timestamp changes is the PDF cover's "generated" line + the Report ID + the
run-folder name + `run_manifest.json`. See
[ARCHITECTURE.md §7](ARCHITECTURE.md#7-reproducibility--provenance).

> `peaky report --run-dir <folder> ...` regenerates the `figures/` + `report/`
> artifacts of an existing run **offline** (no assignment, no network) from the
> ledgers already on disk + the TS parquet.
