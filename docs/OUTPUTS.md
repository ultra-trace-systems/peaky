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
| `merged_ledger.csv` | **The result.** Every merged peak (one row each): role, neutral formula + adduct, scores, ppm, confidence, tier, provenance, the winning row's `admitted_by` / `occurrence` and, with a time series, its `ts_disposition` / `ts_cv_norm` (a background label: flatness never changes the tier), the files' vote on the reading (`n_files_ion` / `n_files_winner` of `n_files`; `ion_agree`; the losing readings in `alternatives`), what the vote, the batch-level known-species decision ([MERGE.md](MERGE.md) §3 step 4b), the re-reads and the isotopologue gate (step 4c: a parent's note naming the line it was given, a mixed or exempt line's note) did to the row (`tier_reason`), `isotopologue_lines` (on a parent row: the lines the isotopologue gate gave it, each as `13C 283.0501 (area x1.00 of predicted, 9 spectra; was C14H10O5 [M-H]-)` -- the label, the line's m/z, its median area ratio to the predicted line, the spectra read and the reading the merged ledger had there -- `; `-joined in ascending m/z; empty on every other row, and on every row when the gate stripped nothing or did not run: one row per parent ion stays the contract, and the batch publish still sends the M0 rows, the server modelling the isotope family from the parent's formula), `stage` (`cover` = some cover file holds the ion; `residual` = only the residual stage's files do — present whenever that stage is on, i.e. unless `--no-residual`), and the trace reconciliation (`mz_anchor` = the merge's m/z, `mz_trace` = the ion's trace centre the stamp uses, `trace_offset_ppm`, `trace_cov_anchor` / `trace_cov` = share of spectra with a peak within tolerance of each, `trace_moved`, `trace_guarded`, `trace_id`, `trace_role` ∈ `single` / `winner` / `collapsed` — a collapsed row is a competing label for a trace another row won; it is kept, flagged, and never stamped). The provenance anchor. Plus the **evidence level** of every row on the evidence scale (`evidence_level` 3c / 4a / 4b / 5a / 5b, or the buckets `reagent` / `NA`, with `evidence`, `would_lift`, `competitors_left`, `tags`, `context`, `context_source` — [EVIDENCE_LEVELS.md](EVIDENCE_LEVELS.md)): levelled on the pooled per-file ledgers and stamped by `(neutral_formula, adduct)`; a merged row whose reading no pooled pair holds (a batch-level re-read) has no level and the evidence `no pooled pair: a batch-level re-read`; on a TOF-class or class-less run every row reads `NA` (not assessed). And the **claim** that level supports on every row (`claim`: `identified` = 3c, the compound named; `neutral` = 4a, the neutral established among the run's declared reagent channels — a side channel the run keeps locked, e.g. formate or acetate on a nitrate source, could re-read an `[M-H]-` ion as a cluster of a smaller neutral, and where one would the row's `evidence` says `side channels locked`; `ion` = 4b, the ion composition established; `tentative` = 5a, 5b or no level, so a batch-level re-read reads `tentative`; beside them the buckets `reagent` and `not assessed` — [EVIDENCE_LEVELS.md](EVIDENCE_LEVELS.md) §1.1): a verdict beside `tier`, not a replacement for it; the two can disagree and neither changes the other. **`NA` is a literal token**: pandas reads it as NaN by default, so read the `claim` column or pass `keep_default_na=False` to tell "not assessed" from "no level" ([EVIDENCE_LEVELS.md](EVIDENCE_LEVELS.md) §2.2). And `ion_only_of` (the winner file's `[M-H]-` parent peak) on the **ion-only** rows — the `[M]-.` electron-attachment line beside a committed acid, a Candidate carrying the acid's composition (stage `ion_only`; levelled 4b at best — an ion-only channel reads the ion only — or 5a / 5b; empty elsewhere). And the winner file's peak separability, `resolvability` (`isolated` / `resolved` / `blended` / `unresolvable`) and `sep_hwhm`, when the run had a width model. And the **TOF mass-only flag** (`mass_only`, bool, with `mass_only_reason`; [below](#readings-that-rest-on-mass-alone-on-a-tof-mass_only)): on a TOF-class run `True` on an Assigned row that no attached isotope line speaking for the neutral supports in any of its Assigned files (the flag's definition below: the neutral's own elements, never a reagent line), `False` on the other Assigned rows (at or above the threshold `False` is not support either); empty on every other row and on every row of an Orbitrap-class run. It never changes a tier or a level. |
| `run_manifest.json` | **Reproducibility manifest.** Pins the run to its exact code (package + per-module version + content hash + git commit), input-data hash (`ts_sha1`), resolved config (the user knobs incl. `height_cutoff_x_edge` and `occurrence_min`; run-derived fields such as `occurrence_threshold` are dropped), the `selection` and `admission` blocks under `counts` (so the resolved persistence threshold is recorded next to the config fingerprint) beside `merged_tiers` and `merged_claims` (the merged rows per claim and bucket) and `tof_flag` (the TOF mass-only flag's tallies; its threshold `tof_flag_mz` is in the config), `context_flags` (the run-level switches of the batch's context profile, as in `batch_summary.json`) and `isotopologue_gate` (`{ran, skipped}`: whether the isotopologue gate ran, or why not -- `--no-isotopologue-gate` is a run argument, not a config field, so the fingerprint alone does not show it), and output hash (`merged_ledger_sha1`). |
| `batch_summary.json` | Run counts + per-file calibration offsets (M0/tier counts, n_files, offsets; per-file `noise_edge_cps` + the resolved gate `height_gate_cps` + the `height_cutoff_x_edge` it came from + `admitted` counts by path), the batch-level `height_cutoff_x_edge` + `height_cutoff_x_edge_source`, the **`admission`** block (`occurrence_min` = the knob, `'auto'` or a number; `occurrence_threshold` = the resolved fraction, `null` when the path is off; `n_peaks` = rows of the per-peak occurrence table; `n_persistent_peaks` = those at or above the threshold; `n_persistent_traces` = the trace-weighted count of those, ≈ the number of persistent ions; `n_spectra`; `tol_ppm`), the **`gate`** block (empty when the multiple was pinned by a flag / cfg / profile or could not be derived; else the batch-derived brightness floor: `x_edge`, `transient_share` per grid multiple, `share_at_1`, `share_at_x`, `max_transient_share`, `n_peaks`, `bound`, `source`), the **`traces`** block (empty without a time series: `n_rows`, `n_recentred`, `n_guarded`, `median_abs_move_ppm`, `mean_cov_anchor` / `mean_cov_trace`, `n_traces`, `n_collapsed`, `n_multi_label_traces`, `sigma_ppm` = the third-quartile per-ion scatter, `stamp_tol_ppm` = the window the stamp used) and the **`selection`** block: `k`, `n_bins`, `achieved_coverage`, `stop_reason` (`gain-floor` / `k_max` / `exhausted`), `next_gain`, `tol_ppm` (the binning tolerance, = the admission table's and the merge's), `k_min`/`k_max`/`min_gain` (+ `coverage_by_group` for a pool), and — when the residual stage is on — its **`residual`** sub-block (`n_bins_residual`, `n_universe`, `n_uncovered`, `n_explained`, `n_below_floor`, `n_sidelobe`, `n_suspect`, `floor` = {`min_x_edge`, `min_cps`, `edge_median_cps`, `source`}, `frac_of_max`, `k`, `k_max`, `coverage_of_residual`, `stop_reason` ∈ `exhausted` / `k_max` / `empty`, `next_gain`, `sample_ids`; `skipped` when there was no time series), plus `n_files_by_stage` and `merged_by_stage` at the top level and a `stage` on every `per_file` row. Plus the **`evidence_levels`** block ([EVIDENCE_LEVELS.md](EVIDENCE_LEVELS.md) §10.2): `scale` (the scale's release), `instrument` (`{class, r200}`: `orbitrap` / `tof` / `null` from the width model; a non-Orbitrap class reads `NA` everywhere), `pooled` (one count per neutral/adduct pair over the pooled files, per level and bucket), `merged` (per merged row), `per_stage` (merged rows by `cover` / `residual`), `n_pairs`, `n_unstamped` (merged rows no pooled pair holds), `side_channels_locked` and `unlocked` (the side-channel lock and the channels unlocked from it), `n_corroborate` and `cross_source` (the `--corroborate` sources and the size of the merge vote's cross set), `partners` (`{source: n neutrals}`: the other-source partners each Orbitrap-class `--corroborate` run dir gave), `amine_r_min`, `neutral_pairs` (rule U's funnel: `pair`, `neutrals`, `committed_both`, `cho`, `present`, `covary`, `clean`, `upair`; `{pair: [], neutrals: 0, upair: 0}` on a profile without a pair), `label_twins` (rule K's funnel on a 15N-labelled nitrate channel: `in_scope`, `f` (the twin fraction), `lines`, `codetected`, `references`, `cluster_k`, `committed_lines` and `lines_<verdict>` per 14N line verdict (`tracks`, `consistent`, `excess`, `absent`, `untestable`), `alien`, `untie`, `readings`, `testable`, `passes`, `refuted`, `lines_refuted`; `{in_scope: false, lines: 0, untie: 0, readings: 0, refuted: 0}` out of scope), `iso_checks` (the isotope checks' funnel: `instrument` (`orbitrap` / `tof`: the batch's resolved instrument class, a TOF roster winning over a width model that reads Orbitrap-class; the width model's when none was resolved), `tested` (every row, rule H's included, the isotopologue gate's `SAT` rows not), `vetoed_pairs`, `locked_pairs` (rule H), and per check `C` / `REQ` / `HIGH` its `tested`, `vetoed` and one count per verdict, `H` its `tested`, `locked` and one count per verdict (`lock`, `no_lock`, `si_rich`, `heavy`, `reagent`, `untestable`), and `SAT` (`{vetoed}`, only when the isotopologue gate stripped a row); `{instrument, tested: 0, vetoed_pairs: 0, locked_pairs: 0}` without a time series). Plus the **`claims`** block right after it (each count a `{identified, neutral, ion, tentative, reagent, not assessed: n}` dict, zeros kept): `merged` (per merged row), `pooled` (per pair), `per_stage`, `by_tier` (merged rows per tier, Assigned first; the ion-only rows under their own `ion-only` key, not their tier) and `n_unlevelled` (merged rows with no level, read tentative), then **`levels_not_assessed_reason`** (`null`, or the one sentence saying the evidence scale assessed nothing because no file calibrated its degeneracy window — too few isotope-backed core rows; the console WARNING and the PDF cover and Evidence levels page say the same once, as does a single-sample workbook's Summary sheet). Plus **`reflists_context`** (how the run's reference lists were activated: `tags`, `matched` = `{tag: [[field, keyword], ...]}`, `active` = `[[id, version, how], ...]`; the evidence scale's `context_source` reads it), **`amine_r_min`** and **`reagent_halogen`** (the halogen of the reagent profile's declared channels the pooled pair facts read, `null` for a halogen-free set; each `per_file` row records its own run's, which the merge vote's class read; [EVIDENCE_LEVELS.md](EVIDENCE_LEVELS.md) §3.4) at the top level. Plus the **side channels** at the top level ([REAGENTS.md](REAGENTS.md) §3b): `side_channels` (the union of the side channels the files opened, `[]` when none), `side_channels_files` (`{adduct: n files that opened it}`), `side_channels_requested` (what the run asked for: the profile's declaration or the `--side-channels` choice) and `side_channels_source` (`profile <name>`; `profile <name> (declares none)` for a profile that declares none, `--side-channels none` on it included; `explicit config`; `no profile`); each `per_file` row records its own `side_channels`. Plus the **`ion_only`** block: `channels` (the ion-only adducts the profile opened, `[]` when none), `merged` (merged rows carrying an `ion_only_of` link), `per_file_rows`, `n_files_with`, `merged_levels` (their evidence levels). Plus **`context_flags`** at the top level: the run-level switches of the batch's context profile (`{nox_skeleton: true, small_acid_band: true}` on a nitrate-reagent Orbitrap-class run in a context that opens organonitrates, `{}` = the named context as is; [ASSIGNMENT_DETAIL.md](ASSIGNMENT_DETAIL.md) §8.3), derived by the helper every file's run calls (`assign.run_context_profile`); the evidence levels' space and the PDF's scrutiny page read it, and each `per_file` row records its own run's `context_flags`. Plus **`warnings`**: notes the run wants read beside its counts; a trace-first batch whose residual files ran with other context switches than the batch records (they run without the trace flag) is named there, not only in the log. Plus **`merge_gates`**: `known` (the batch-level known-species decision — `pooled` / `locked` / `confirmed_kept` / `conflict` / `lead_only` / `mass_only_outvoted` / `no_cluster`), `reagent_water` (the reagent-water ladder, [MERGE.md](MERGE.md) §3 step 4a: `n_cores`, `tol_ppm`, `segments` = spectra per acquisition segment, `n_rungs`, `rungs_by_core` = the passing n per core (n = 0 is a reagent-only core itself, the (HNO3)2.NO3- dimer core), `n_stripped` and `stripped` = the merged readings that sat on a passing rung; `rung_test` only when the batch's width model is TOF-class and the TOF rung test ran: its constants, the FWHM at m/z 200 and 600, and `strip_by_segment` — the strip then follows the segments of the reading's files), `isotopologue` (the merged-ledger isotopologue gate, [MERGE.md](MERGE.md) §3 step 4c: `ran`, `n_stripped`, `n_mixed`, `n_exempt`, `n_parent_removed` (stripped rows whose parent reading the element-signature removal of step 6a then took out of the merged ledger: verdict `parent removed` in `tables/isotopologue_rows.csv`, no `SAT` veto), `restamped` (the per-file rewrite below re-stamped the batch series), and when it ran `instrument`, `window_ppm`, `constants`, `n_parents`, `n_matched`, `n_judged`, `n_stripped_assigned`, `stripped` (`neutral adduct = label of parent ion`), `per_file` (what the per-file ledgers now record, `iso_checks.reconcile_per_file`: `iso_child` / `reagent` / `released` rows rewritten, `released_parent_removed` among the released, `files` per sample id, `problems` only when a rewritten ledger fails the ledger invariants); `skipped` = why it did not run: `--no-isotopologue-gate`, an instrument class other than `orbitrap`, no width model, no time series, a series without peak areas, no merged rows), `element_signature` (the merge-time element-signature removal, step 6a: `removed`, `pairs` (`neutral_formula`, `adduct`, `mz`, `tier`, `note`, `reading_left` per removed row: `false` when another merged row of the same reading stays, e.g. the one a known-species decision marks, and then nothing downstream treats the reading as gone), `restamped` when the series was re-stamped without them, `per_file` when a row was removed (what the per-file ledgers now record, `iso_checks.release_signature_removed`: `released` M0 rows, `children` released with them, `files` per sample id, `problems` only when a rewritten ledger fails the ledger invariants); `skipped` on a batch whose resolved class is not `orbitrap`), `tof_m2` (the TOF ion-M+2 gates after the stamp, on a batch whose resolved class is `tof`, [MERGE.md](MERGE.md) §3 step 6b: `ran`, `req_demoted`, `known_demoted` (known-species decisions overruled), `doublet_demoted`, `doublet_exempt`; `skipped` = `not a TOF-class batch` / `no width model` when it did not run) and, on a positive channel, `reagent_n` and `amine`. Plus **`scorer`** (`local` = candidates scored in-process, the default; `server` = by match_compounds, under `PEAKY_LOCAL_SCORING=0`; absent on a run made before the key existed). Plus **`resolution`** (the peak-width model the per-file resolvability stamp used — `coef`, `exponent`, `offset`, `r_at_200`, `r_at_600`, `n_peaks`, `r_spread`, `source` ∈ `measured` / `declared`; `null` when not stamped) and **`resolvability`** (the per-file M0 class counts summed over the files). Plus **`tof_flag`**, right after `ion_only` (the TOF mass-only flag, [below](#readings-that-rest-on-mass-alone-on-a-tof-mass_only)): `instrument` (the class it keyed on — the width model's, else the files' scoring class), `applied` (`true` on a TOF-class run), `threshold_mz` (`--tof-flag-mz`, default 350), `n_assigned`, `n_flagged`, `n_known` (known species, exempt), `n_positive` (Assigned rows with an own isotope line), `below` and `at_or_above` (`{assigned, flagged}` either side of the threshold) and `definition` (the line test in one sentence). |
| `per_file/<sid>_ledger.csv` | The full single-sample ledger for **each** assigned sample, kept for audit / re-merge. Carries the admission provenance per peak: `occurrence` (float in [0, 1] — the fraction of the batch's spectra holding a peak within tolerance of this peak's m/z, one per spectrum; NaN without batch context or when no batch peak lies within tolerance) and `admitted_by` (`height` / `occurrence` = persistence only / `''` = not eligible at the eight gated sites — the pass-1 grid, pass-2 series growth, pass-3 contaminant families and pass-3's two cluster resolvers, the pass-6 ladder gap-fill, residual stage B and the siloxane ladder; residual stage A, the reflist rescue, pass-3's series *detection* statistics and the postprocess satellite tests are still brightness-only, so an empty stamp says nothing about them). Each also carries its own per-file evidence level (the eight evidence columns `evidence_level` … `claim`, computed on that file alone with every file-count minimum 1 and no time series; the batch level on the merged ledger is the pooled one, [EVIDENCE_LEVELS.md](EVIDENCE_LEVELS.md) §10) on the M0 rows only (empty on isotope children, reagent ions and unexplained peaks) and `ion_only_of` (the parent peak of an ion-only `[M]-.` row; `NA` elsewhere). **The merged-ledger isotopologue gate rewrites a file's ledger after the merge** ([MERGE.md](MERGE.md) §3 step 4c): where the file committed a stripped reading as an M0 on the gate's line, the row is the parent's `iso_child` under the gate's label, its `iso_match_score` the gate's own agreement min(rho, 1/rho) of the batch area ratio, when the file commits the parent (a reagent parent's: a `reagent` row `reagent isotopologue: <label> of <ion>`), else -- and wherever the parent reading then left the merged ledger (`parent removed`) -- `unexplained`; its commentary carries the `batch isotopologue gate` mark (a released row's, inside the ledger's `CLEARED (` prefix; search for the mark, not a prefix) and names the parent ion, the label and what the row was (`was <neutral> <adduct>, <tier>`), and its per-reading columns (the eight evidence columns, resolvability, degeneracy, `ppm_error_cal`) are emptied. The file's role, signal, confidence and tier counts, `n_M0` and resolvability class counts in `batch_summary.json` `per_file` are recounted from the rewritten ledger, with an `isotopologue_gate` entry counting what was rewritten there; every other stat stays the file's own run's. **The element-signature removal rewrites it too** ([MERGE.md](MERGE.md) §3 step 6a): every M0 row that commits a reading the batch removed (and that left the merged ledger) is released to `unexplained`, its own isotope lines with it (each `CLEARED (batch element-signature removal: a line of the refuted reading ...)`), its commentary carrying the `batch element-signature removal` mark inside the `CLEARED (` prefix, the refuted line (the REQ note) and what the row was (`was <neutral> <adduct>, <tier>`), its per-reading columns emptied; the file's counts are recounted the same way, with an `element_signature_gate` entry (`released`, `children`). And `known_lead`: the known-species claim pass 0 anchored on-cal in this file but did not commit, as JSON (formula, family, label, adduct, ion, verdict `deferred` / `refuted`, `why` with the file's numbers, `summary` without them); `NA` elsewhere. The batch pools it ([MERGE.md](MERGE.md) §3 step 4b). And the separability stamp on every M0 row: `resolvability`, `sep_hwhm` (distance to the nearest picked peak, in HWHM at that mass), `d_crit_hwhm` (the separation at which two peaks of that height ratio become bimodal); `NA` without a width model. And the two assignability flags, bool: `below_assignability` (the assignment argues with itself) and `tentative_lead` (the proposal is unsupported, not contradicted; split off `below_assignability`), read by the evidence level as a rejection (`below_assignability` by an implausible-chemistry setter: 5b) and as a tag with no level effect (`tentative_lead`: the `lead` tag) ([EVIDENCE_LEVELS.md](EVIDENCE_LEVELS.md) §3.2, §4); a ledger written before the split has no `tentative_lead` column. Beside them `lead_by`: the setter(s) that made the row a lead, pipe-joined codes (`reflist_dim`, `off_budget`, `spec_n3` / `spec_gapfill` / `spec_minor`, `radical_anion`, `reagent_n`), empty elsewhere; the evidence level prints it in the `lead` tag (a ledger written before it has none). |
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
| `evidence_levels.csv` | One row per `(neutral_formula, adduct)` pair of the pooled per-file ledgers, levelled on the evidence scale ([EVIDENCE_LEVELS.md](EVIDENCE_LEVELS.md) §2.6): the eight evidence columns (`evidence_level`, `evidence`, `would_lift`, `competitors_left`, `tags`, `context`, `context_source`, `claim`) first, then the step facts (`tag_kinds`, `split_pinned`, `split_how`, `positive_fact`, `named_list`, `named_mode_flag`, `window_only`, `window_isobar`, `chloride_open`, the amine gate's `nh4_gate` / `nh4_admissible` / `nh4_inadmissible` / `nh4_gate_detail`, `side_aliases`, `route_alias`, `anchor_kind` / `anchor_why` (what anchored the pair in the internal pass: `two routes`, `ladder`, `listed` or `none`, and why), `n_left_inpass`, `n_series_excl`, `iterations`), step 1's facts (`n_competitors`, `n_excl_iso`, the committed reading's `committed_*` facts, `no_comp_info`, the space notes) and the pooled pair facts (`iso_veto` / `iso_note`, `label_veto` / `label_untie` / `label_note`, `lowconf`, `below`, `lead`, `tied`, `ion_only`, `upair`, `lead_lift` / `lock_note`, `iso_labels`, `n_files`, ...; the last few are recorded and read by no step of the scale). The audit trail of the merged ledger's level. `scripts/level_ledger.py --out` writes the same levels post hoc (the level, `claim`, `would_lift`, `competitors_left` and its own decision facts, one row per pair), not this table: no `evidence` / `tags` / `context` columns and no pair facts. `NA` is a literal token (read with `keep_default_na=False`, or read `claim`); on a TOF-class or class-less batch (every pair `NA`) the table keeps the pooled pair facts, without the step facts. |
| `neutral_pairs.csv` | Rule U (docs/EVIDENCE_LEVELS.md §3.2 `upair`: a pooled fact the evidence scale records and does not read): one row per neutral committed under an adduct of the profile's neutral pair (the uronium profile: `[M+H]+` and `[M+(CH4N2O)H]+`), every clause measured on the stamped batch time series — `committed_both`, `cho` (C/H/O only), `det_bare` / `ppm_bare` / `det_cluster` / `ppm_cluster` (share of spectra with a peak within 2 ppm, median ppm), `n_both` / `r_log` (the co-variation), `clean`, `nC_bare` / `nC_cluster` / `nC_obs_*` / `c13_scale` / `c13_contradicts` (the area-¹³C carbon check), `present`, `covary` — and the verdict `upair`. Written on every run; empty (header only) on a profile without a pair or a run without a time series. |
| `label_twins.csv` | Rule K (docs/EVIDENCE_LEVELS.md §3.2 `label_veto`: a vetoed pair is rejected, 5b; `label_untie` is recorded only): on a 15N-labelled nitrate channel, one row per 14N line of every neutral committed on either nitrate adduct (`adduct` = `[M+NO3]-`: `committed` (on `[M+NO3]-`), `mz` / `mz_partner`, `n14` / `n_codetected` / `partner_share` (spectra with the 14N line, with both lines, their ratio), `h15_median`, `reference`, the cluster-k statistics `k_ratio` / `k_sd` / `k_r` / `k_n`, `cluster_k`, `line_verdict` (tracks / consistent / excess / absent / untestable), `alias_only_tie`, `untie` (the line would clear the reading's tie; the pooled level's `label_untie` records where it did), `alien`, `veto` and its `note`) and one per committed `[Y+^NO3]-` reading (`f`, `E` (expected twin sightings), `obs`, `twin_verdict` ∈ passes / unclear / refuted / untestable, `veto` and its `note`), measured on the stamped batch time series. Written on every run; empty (header only) out of scope or without a time series. |
| `iso_checks.csv` | The isotope checks (docs/EVIDENCE_LEVELS.md §3.2 `iso_veto`): one row per tested pooled pair and `check` — `C` (the 13C carbon count, Orbitrap-class only: `n_carbon`, `c_area` / `c_height` (bias-corrected), `se_area` / `se_height`, `bias_area` / `bias_height`, `occupied`), `REQ` (a required heavy line: the decisive `line`, its `expected` ratio to the stamped line, `line_eff` the height this batch shows the element's lines at (× theory, ≤ 1), `n_used` detectable spectra, `n_present` / `det_frac`, `window_ppm`; an `artifact` stamp (a side lobe or ringing a file marked) is no line, and a spectrum where only an artifact holds the position counts neither in `n_used` nor as present — the per-file element-evidence test and the evidence levels read an Orbitrap file's artifact rows the same way; on a TOF-class batch REQ reads the ion's own M+2 line by the tier pass's primitive (`satellites.heavy_line_verdict`): `line` = `M+2 (81Br)` / `(37Cl)` / `(81Br/37Cl)`, `expected` the ion's whole M+2 cluster over the stamped line, `n_used` the testable spectra (the line predicted at ≥ 3 × the batch's detection edge and not blended), `n_present` / `det_frac` the spectra / share it is seen in (≥ 0.6× within `window_ppm` = max(15, 3 sigma)), `occupied` the share of the pair's spectra whose M+2 position another line holds or a split line could carry; absent (a veto) over ≥ 10 testable spectra, at least as many as the blended, seen in < 30 %; `line_eff`, `n_present_wide`, `det_frac_wide` empty) or `HIGH` (a heavy line too high: `offset`, `expected` M+2, `ratio_area` / `ratio_height`, `r`, `presence`, `offset_mda`, `other_m0` / `other_13c` (the slot guards)) — with `instrument`, `ion`, `mz`, `stamped`, `n_spectra`, the `verdict` (C: agree / ambiguous / contradict / untestable / scan_edge / exempt_14N; REQ: present / absent / untestable; HIGH: consistent / guarded / too_high; H: lock / no_lock / si_rich / heavy / reagent / untestable), `veto` and a `note` with the numbers. A vetoed pair is rejected by the evidence level (5b, `iso_veto`), and rule C's carbon count is test (c)'s observed count where it is testable. On an Orbitrap-class batch a `REQ` veto whose `line` is an element-signature line (81Br / 37Cl / 34S / 29Si / 30Si) also removes the reading from the merged ledger ([MERGE.md](MERGE.md) §3 step 6a). The isotopologue gate adds one `SAT` row (`verdict` `isotopologue`, `veto` True, `line` the isotope label, `expected`, `ratio_area` / `ratio_height` the median ratios, `n_spectra` / `n_used` the spectra read, `note`) per merged row it stripped, so the pooled pair reads the same refutation as the merged ledger. Rule H adds one `H` row per pooled pair whose ion carries Cl or Br: the line at the exact halogen spacing (`offset` 37Cl / 81Br, `ratio_area` / `ratio_height`, `r`, `presence`, `offset_mda`, `other_m0`, `expected` = n × per atom), `lock` (never a veto), `element`, `n_halogen`, the count window `ratio_lo` / `ratio_hi`, the heavy check `heavy_cl` / `heavy_br`, and `budget_ok` / `budget_why` (the neutral against the batch context's element budget with the locked halogen lifted / its first violation), and the silicon test of a Cl line from m/z ~206 (the 2026-09-28 decision: `si_rich` where the M+1 region shows the 29Si line a Si-rich ion making the partner from 30Si must carry): `si_n` (the silicons the partner implies read as 30Si), `si29_expected`, `si29_seen` (the 29Si line's area, or on a blended +1 line its excess over the reading's own +1 line) and `si29_mode` (`resolved`; `unparted` where the width model parts 29Si from 13C but no 29Si line is present, so the +1 region decides as on a blend; `blended`; empty where not tested) — on a blend the +1 region's position decides (a reading that itself carries Si is also refused on its excess; the 2026-09-29 decision), and a `si_rich` note gives that position and the half-way mark, or says a Si reading was refused on its excess; a locked pair's lead the lock answers is recorded as lifted (`lead_lift`; the evidence scale gives a lead no level effect, so the lift moves no level). Written on every run; empty (header only) without a time series. |
| `reagent_water.csv` | The batch's reagent-water ladder ([MERGE.md](MERGE.md) §3 step 4a), one row per passing rung core.(H2O)n, n ≥ 1, plus an n = 0 row for a declared reagent-only core that passes the same presence test (the (HNO3)2.NO3- dimer core and its 15N twin: no analyte reading, so the core line is itself a reagent row): `core`, `iso_tag` (the halogen isotopologue, empty for none), `n`, `ion_formula`, `mz` (exact) and `mz_obs` (median of the matched peaks), `segments` / `presence` / `decoy_presence` (the segments it passed in, `|`-joined; on a TOF `decoy_presence` is the mean presence of the rung's width-scaled decoys), and `displaced` (the merged readings that sat on it and left the merged ledger; after the TOF rung test only those whose winning reading a file of a segment where the rung passed carries — a row kept on a rung says so in its `tier_reason`). Header only when the profile declares no water cores or no rung passes. |
| `isotopologue_rows.csv` | The merged-ledger isotopologue gate ([MERGE.md](MERGE.md) §3 step 4c; Orbitrap-class batches with a width model and peak areas in the time series): one row per merged row found on another merged ion's (or a reagent ion's) 13C / 18O / 15N / 34S / 37Cl / 81Br / Si line within max(1 ppm, 4 sigma), judged over ≥ 3 spectra and read as one of the verdicts below (a stripped row's line is recorded where it now belongs: the parent's `isotopologue_lines` in the merged ledger and the per-file ledgers' `iso_child` rows) (a row whose median ratio falls outside 0.4-4× is not listed): the row's `mz`, `neutral_formula`, `adduct`, `ion`, `tier`, `ion_score`; `verdict` (`isotopologue` = median area ratio 0.4-1.4× its expected share and fewer than max(1, 10 %) of the spectra above 2×; `mixed` = a median of 1.4-4×, or in band but failing that tail test; `exempt` = in band but a known-species decision, a curated reading (the pass-0 registry for the batch's polarity and context, and the active reference lists), an ion-only row, an isotope-labelled reading, or a reading on a labelled pair's channel whose line is the label's element; `parent removed` = a row stripped as an `isotopologue` whose parent reading the element-signature removal (step 6a) then took out of the merged ledger: the row stays out of the merged ledger, its line unexplained (in the per-file ledgers too), and its pooled pair carries no `SAT` veto) and `action` (`stripped` = left the merged ledger, its pooled pair a `SAT` veto in `iso_checks.csv`; `stripped; parent removed` = the `parent removed` rows; `note` = a note on the row's `tier_reason` only); `label` (the line, e.g. `13C`, `29Si`, `13C2`), `expected` (the line's share of the parent), the main parent (`parent_mz`, `parent_ion`, `parent_neutral`, `parent_adduct`, `parent_tier`; empty neutral and `reagent` tier for a reagent ion) and every parent read (`parents`), `offset_ppm`, `rho_area` / `rho_height` (median observed / expected), `n_both` (spectra holding the line and a parent), `n_tail`, `share` (the parent's share of the line), the per-file role votes `files_m0` / `files_iso_parent` / `files_other` (for the record), `alternatives` and `note`. A stripped row whose parent the element-signature removal (step 6a) later takes out of the merged ledger stays listed with verdict `parent removed`, action `stripped; parent removed` and a note naming the parent (`merge_gates.isotopologue.n_parent_removed` counts them). Always written; header only when nothing was found or the gate did not run. |
| `jitter.csv` | Per-(cluster, file) table: every file's own reading (formula, adduct, tier, `ion_score`, `vote_class`) of each merged peak — the per-file detail behind the merged row's vote, whose ion stage ranks by the reading's vote class (0 unconfirmed / 1 formula confirmed / 2 neutral backed; not an evidence level, [EVIDENCE_LEVELS.md](EVIDENCE_LEVELS.md) §13) before the file count — and its m/z for the raw vs calibration-adjusted jitter. |
| `van_krevelen_full_<tag>.csv` | The full-VK data behind the figure (one row per assigned neutral). |
| `clusters_changing_<tag>.csv` / `.xlsx` | Cluster membership; the XLSX has one tab per cluster (formula / channel / m/z / match_score / tier). |
| `clusters_flat_<tag>.csv`, `clusters_changers_<tag>.csv`, `clusters_unassigned_<tag>.csv` | Membership for the flat / changers / unassigned figures. |
| `channel_agreement_<tag>.csv` | QC: how often a multi-channel neutral's ion channels agree in time. |
| `plausibility_audit_<tag>.csv` | One row per peak the hardened plausibility layer touched (demoted, relabelled, or cleared by the element-evidence gate: `after_tier_or_role` = `unexplained`, the `reason` starting `element_evidence:`): `before_tier`, `after_tier_or_role`, the `reason`, the supporting `evidence` (O/C or DBE/C or series r), the `degeneracy_note`, and `n_iso`. Always written (header-only when nothing was touched). |
| `predicted_satellites.csv` | One row per **predicted** isotope-satellite line of the batch stamp that had a candidate peak (`timeseries.annotate_peaks`): `ion_formula` (the parent's), `iso_label`, `ion_mz` (the predicted m/z), `iso_rel` (predicted height relative to the parent), `n_candidates` (peaks in the window), `n_eval` / `n_pass` (samples in which the parent was stamped and a candidate sat on the line / passed the height gate), `pass_share`, `judged`, `kept` (the track-coherence verdict) and `n_stamped`. Always written (header-only when no line had a candidate). |

### `report/` — the PDF

| Artifact | What it is |
|---|---|
| `report_<run-id>.pdf` | **The standard iterable A4 report** (cover · claims · findings · coverage · evidence levels · composition · scrutiny · reference lists · GKA · mass-defect QC · families · changers · clusters · methods · assignments appendix). The cover shows the Report ID + a date+time "generated" line. On a run with levels its summary opens on a claims line (merged rows per claim and, with the per-file ledgers on disk, each claim's share of the committed-peak signal), and the **Claims** page follows the cover: merged rows and committed signal per claim and bucket (the four claims, `reagent`, `not assessed`) — every per-file M0 height credited with the merged claim of its `(neutral_formula, adduct)`, the readings no merged row carries an explicit `no merged row` bucket, never folded into tentative — the tier × claim crosstab (Assigned / Candidate / ion-only) and the brightest rows where tier and claim part. The findings' top species and the appendix carry a claim column; the **Evidence levels** page counts the levels and buckets of the evidence scale. On a TOF-class or class-less run every level reads `NA` and the report says the scale was not assessed on this instrument class. A run without levels gets no Claims page. |
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

### Readings that rest on mass alone on a TOF (`mass_only`)

On a TOF the formula grid saturates with m/z. A shifted-mass decoy (the same
spectra moved off every real ion, assigned by the same engine) measures it: on a
bromide/nitrate TOF batch (R ≈ 9 700 at m/z 200), pooled over seven files, the
decoy spectra had 21 Assigned readings below m/z 350 against 292 for the real
spectra, but 234 against 122 at or above it — the CHNOS mass-defect gap has
closed there and an accurate mass picks one of dozens of formulas. A
lower-resolution nitrate TOF (R ≈ 4 100) showed the decoy at the real spectrum's
level from m/z 350 up in one decoy arm and from about m/z 450 up in another.
Below that, too, most TOF readings rest on mass: their own isotope lines are not
seen.

Such readings are **flagged, never demoted** — what does make it at high masses
stays visible, with what it rests on beside it. On a TOF-class run
(`assignment/mass_only.py`) every Assigned merged row gets `mass_only`:

- **`False`** when, in at least one file that holds the reading at tier
  Assigned, an isotope child of its M0 (an `iso_child` row pointing at it) sits
  at its label's exact spacing from the committed parent line (the evidence
  scale's placement test), stands at **0.5–2×** its count-aware expected height,
  and measures an element the **neutral** supplies most of in the ion: the 13C
  line of an organic neutral; a 37Cl / 81Br / 34S / 29Si / 30Si line of a neutral
  that carries the element. The reagent's own twin never counts (the 81Br line
  of a Br-free neutral's `[M+Br]-` adduct says the ion holds the reagent, not
  what the neutral is), nor does a listed line another reading owns. This is the
  evidence scale's per-file isotope fact (`multiline_elements`) with one rule on
  top: on a run whose reagent channels carry a halogen (`reagent_halogen`), a
  line of that halogen counts only when the **ion** holds more of it than one
  reagent channel puts on an ion (two, by `[M+HBr+Br]-`). Below that the line
  does not depend on the label: `C8H6BrNO4 [M+NO3]-` and `C8H6N2O7 [M+Br]-` are
  one ion, whose 81Br line is the reagent's twin under the second label, so it
  keeps neither unflagged; a 13C line still counts beside it. Only **attached**
  children count: a peak at the right spacing that the scorer did not attach to
  the reading (one no reading owns, often near the noise edge) is not seen.
- **`False`** for a pass-0 known species (a `known:` commit at Assigned in one of
  its files, or the batch's known-species decision on the row): its identity is
  the curated list's.
- **`True`** otherwise, with `mass_only_reason`: below `--tof-flag-mz` (default
  **350**) *the reading rests on mass alone*; at or above it *a TOF's formula
  space is saturated here (decoy-measured on two TOFs)*. The threshold only picks
  the reason; it is the decoy's constant, unmeasured on iodide TOFs.

**`False` is not support at or above the threshold.** A shifted spectrum keeps
real isotope spacings, and a 0.5–2× 13C band bounds the carbon count only to a
factor of two, so there even a present line is weak: on the bromide/nitrate TOF's
seven-file decoy arms the line test left 58 shifted-decoy readings at or above
m/z 350 unflagged against 12 real ones. Below m/z 350 it flagged all 21 decoy
readings, so there the flag does separate a decoy from a real reading.

Candidate rows, ion-only rows and every row of an Orbitrap-class run keep the
columns empty. On the two TOF batches above the flag marks 261 of 355 Assigned
readings (164 of 222 below m/z 350, 97 of 133 at or above it) and 250 of 282
(176 of 193, 74 of 89). The flag is information: it never changes `tier`,
`evidence_level` or `claim`, and it does not catch a reading its own isotope
lines refute (that is the isotope checks' job) — it says where no isotope line
was there to check. The counts are in `batch_summary.json` (`tof_flag`), the PDF
Findings page (one sentence; flagged species carry † in the species tables), a
single-sample workbook (the Assigned sheet's two columns and the Summary's
*TOF mass-only flag* section) and the scorecard's element census.

```python
m = pd.read_csv("merged_ledger.csv")
mass_only = m[(m.tier == "Assigned") & (m.mass_only == True)]   # empty off a TOF
```

---

## Single-sample run — `peaky assign`

Writes into `--output-dir` with the prefix `<sample-id>_<YYYYMMDD-HHMM>`:

| Artifact | What it is |
|---|---|
| `<prefix>_ledger.csv` | Every peak: role (`M0` / `iso_child` / `reagent` / `artifact` / `unexplained`), formula, adduct, all scores (incl. arbitration `eff_score`/`eff_margin`/`tied`), ppm, confidence, `tier` + `tier_reason`, candidate/degeneracy density, provenance, commentary, alternatives, isotopologues. Plus the evidence level on the evidence scale, the file levelled alone (the eight columns `evidence_level`, `evidence`, `would_lift`, `competitors_left`, `tags`, `context`, `context_source`, `claim` on M0 rows, empty elsewhere — [EVIDENCE_LEVELS.md](EVIDENCE_LEVELS.md); `NA`, not assessed, without an Orbitrap-class width model) and `ion_only_of` (the `[M-H]-` parent peak of an ion-only `[M]-.` row; `NA` elsewhere), and `known_lead` (a known-species claim pass 0 anchored on-cal but did not commit, as JSON with its verdict `deferred` / `refuted` and why; `NA` elsewhere), and the separability stamp (`resolvability` / `sep_hwhm` / `d_crit_hwhm` on M0 rows, from the run's width model; `NA` without one), and the two assignability flags `below_assignability` / `tentative_lead` (bool; [EVIDENCE_LEVELS.md](EVIDENCE_LEVELS.md) §3.2) with `lead_by`, the setter that made a row a lead. On a TOF-class sample, `mass_only` / `mass_only_reason` on the Assigned M0 rows (the file's own isotope lines; [above](#readings-that-rest-on-mass-alone-on-a-tof-mass_only)), empty elsewhere. |
| `<prefix>_assignments.xlsx` | The styled multi-sheet workbook: **By claim** · Summary · Read-me legend · **Assigned** · **Candidates** · By evidence level · Below assignability (the M0 rows either assignability flag marks, with a `tentative_lead` column saying which and `lead_by` which setter made it a lead; a ledger with neither flag column has it empty) · Unassigned (evidence-characterized) · By class · Unique formulas · Isotopologues · Peak ownership · Target list · Reagent ions. Frozen headers, autofilters, tier/confidence/evidence-level/claim color chips; the evidence columns ([EVIDENCE_LEVELS.md](EVIDENCE_LEVELS.md) §2.1) on the Assigned, Candidates, Target list and Peak ownership sheets, with `claim` directly before `evidence_level` (the Candidates sheet's rank-1 row only: an alternative makes no claim); the Assigned sheet keeps the header `tier_reason` (it no longer renames it), and the Unassigned sheet's residual characterization is headed `residual_evidence` and `best_claim` beside `best_tier` on Unique formulas. **By claim**, the sheet the workbook opens on, holds one `summary` row per claim and bucket (the four claims, `reagent`, `not assessed`: meaning, levels, count and share of the M0 rows, summed height and its share, the Assigned / Candidate / ion-only split, the level histogram), then the `tier disagrees` rows (Assigned but tentative, Candidate but identified; brightest first) and the twenty brightest rows of each claim. **By evidence level** holds one row per level and bucket of the evidence scale (3c, 4a, 4b, 5a, 5b, `reagent`, `NA`) with its count, share, tier split, a histogram of the tag kinds and the brightest rows. The Summary gains a **Claims** section between Coverage and Tiers (count, share of assignments and of assigned signal per claim) and the Read me opens on the claim classes — including that the By class sheet's `n_identified` counts tier-Assigned rows, a column name older than the claim. A ledger without levels gets none of this and renders as before; one with levels but no `claim` column gets the claim read off the level. A TOF-class sample's ledger also carries the mass-only flag: the Assigned sheet shows `mass_only` (a blue chip, information not a verdict) and `mass_only_reason` right after `tier_reason`, the Summary a **TOF mass-only flag** section (flagged of Assigned, and the split below / at or above `--tof-flag-mz`) and the Read me two rows on the columns; any other ledger renders as before. |
| `<prefix>_summary.md` | Narrative + top assignments + coverage; with levels, a `Claims:` line before the tiers and each top assignment tagged with its tier and claim. |
| `<prefix>_manifest.json` | Module versions, prescan fingerprint, series-evidence table, per-pass timing, and the stage summaries (the `evidence` stage's `claims` tally among them). Its `stats.sidelobe_guard` is the pre-pass Orbitrap side-lobe guard's record: `flagged` (rows marked `artifact` and locked), `skipped` (`null`, or why it did not run: `off (sidelobe_guard=False)` (`--no-sidelobe-guard`), not an Orbitrap (or an unknown instrument class), trace sample, no width model / area, a declared model the widths cannot calibrate, a width ratio outside 0.5-2), `by_signature` (`narrow` / `neg-lobe` / `mirror` counts), `n_mirror`, `n_no_signature` (in-band peaks spared), `n_exempt` (lines spared as isotope lines, by kind: the fine-structure spacings, and a candidate at an isotopologue offset of a brighter line of the list), `signatures` (those tested: no `narrow` when the calibration is not measured) and `calibration` (`ratio`, `n`, `measured`, `slope`). A batch's `batch_summary.json` carries the same block on each `per_file` row. `stats.context_flags` is the run-level switches of the context profile the file was judged on (`{}` = the named context as is). The stage summaries hold `element_evidence` (the element-evidence gate: `tested`, `cleared`, `cleared_isotope`, `cleared_mono` (P / I), `si_ladder_kept`, the verdict counts `confirmed` / `contradicted` / `unobservable`, `instrument_class`, and `calibration` = the file's exact-offset calibration in words, Orbitrap class only). |
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
> ledgers already on disk + the TS parquet. The batch name and the dataset name
> come from the run's `run_manifest.json` (override with `--batch` / `--dataset`),
> so the regenerated report unlocks the same reference lists the run did.
