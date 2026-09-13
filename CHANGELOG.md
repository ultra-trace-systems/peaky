# Changelog

All notable changes to Peaky are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims to
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed


- **The admission table is per PEAK, and the lookup is the table's own rule.**
  `admission.bin_occurrence` no longer gap-clusters the batch into bins: it is built on
  the new `batch/traces.PeakIndex`, and every batch peak's `occurrence` is the fraction
  of spectra holding a peak within ±tolerance of *its* m/z (one O(n) sweep over the
  m/z-sorted batch, one peak per spectrum; the window is ±max(tol ppm, 1.5 mDa), the
  stamp's own floor). `lookup_occurrence` answers a ledger peak by exactly that rule, so
  construction and lookup cannot disagree. The persistence threshold (`auto`) is Otsu's
  split of that distribution weighted so each TRACE counts once (each peak by one over
  its spectrum count): it reproduces the former bin-table split to ±0.02 on four
  Orbitrap modes and reads 0.38 instead of 0.44 on a TOF batch; unweighted, the
  per-peak histogram is dominated by persistent ions and splits ~0.1 too high.
  `batch_summary.json['admission']` reports `n_peaks`, `n_persistent_peaks` and
  `n_persistent_traces` (the trace-weighted count, ≈ the number of persistent ions)
  in place of `n_bins` / `n_persistent_bins`, which counted bins under a rule the
  admission decision did not use.
- **The brightness floor is derived from the batch by default.** An unset
  `height_cutoff_x_edge` now resolves to the policy token `"auto"`
  (`passes.config.AUTO_HEIGHT_CUTOFF_X_EDGE`); `--height-cutoff-x-edge auto` asks for
  it explicitly and a reagent profile may carry it. On a batch, `assign_batch.run`
  resolves it to a number from the batch's own peaks
  (`admission.derive_height_cutoff_x_edge`), on two conditions: the picker must pick
  INTO the noise — more than 5 in 10 000 of the batch's peaks sit below 0.75× their
  sample's 1st-percentile edge; a hard-threshold picker leaves essentially nothing under
  its edge — and then the floor is the smallest of 1 / 1.5 / 2 / 3 / 5 / 8 / 12 / 20 at
  which the peaks it would admit are at most 20 % transient (occurrence below the
  persistence threshold). The number is stamped on the cfg every per-file run copies
  and recorded with its source, both statistics and the whole transient-share table
  (`batch_summary.json` `height_cutoff_x_edge`, `height_cutoff_x_edge_source`, `gate`).
  A single sample with no batch reads `"auto"` as 1.0 (`DEFAULT_HEIGHT_CUTOFF_X_EDGE`),
  and so does a batch too short for a persistence table (with a source saying why).
  Measured live on five batches of two instruments: every Orbitrap mode keeps exactly
  1.0 (tail share ≤ 1 in 10 000; three modes 8–16 % transient at 1×); the TOF (tail
  share 31 in 10 000, 34 % transient at 1×) lands at 5× (17 %; inside the recall
  plateau of the live sweep
  — 17 of 18 findable target ions from 5× up — though below its 8–12× precision
  optimum, which a site pins in its profile). The tail condition exists because the
  reagent-in-range Orbitrap mode is 50 % transient at 1× (the reagent ion's noise skirt
  plus a scan-edge pile-up) and the share alone raised it to 2×, which removed 45
  pile-up rows and 12 skirt rows but also 14 multi-file Assigned ions at 1–2× the edge
  (a C6H12O `[M+H]+` seen in 9 of 27 files among them) — there the edge is the picker's
  relative threshold under a huge reagent ion, not noise. The reproducibility manifest
  fingerprints the *token* for an unset multiple, never the derived number (which is
  run-derived, like the persistence threshold, and lives in `counts`).
- **The batch time series is stamped from each ion's TRACE, not from the merge
  anchor.** Between the merge and `annotate_peaks`, `timeseries.recentre_ledger`
  re-centres every merged row on its own trace (`PeakIndex.mean_shift` from the
  anchor, at most 10 ppm; an anchor covering < 10 % of the spectra that wants to move
  > 6 ppm must be seen in ≥ 2 files or be Assigned), `collapse_trace_labels` collapses
  rows whose trace centres fall within the merge tolerance onto one winner (the
  merge's own ordering: most assigned files, then tier, then `ion_score`, with
  proximity to the trace centre only as a deterministic tie-break — never
  `ion_score` first, which is per-file and blind to reproducibility, and never
  proximity before the score, which on theory-snapped TOF anchors carries no
  information and lost the known monomer C10H16O9 to a one-file competitor), and
  `stamp_tolerance` sizes the stamping window to the batch's own per-ion scatter (2.5 ×
  the third quartile of the per-trace robust scatter, between 1× and 2× the merge
  tolerance). The merged ledger gains `mz_anchor`, `mz_trace`, `trace_offset_ppm`,
  `trace_cov_anchor`, `trace_cov`, `trace_moved`, `trace_guarded`, `trace_id` and
  `trace_role` (`single` / `winner` / `collapsed`; nothing is dropped);
  `_batch_ts.parquet`'s `ion_mz` is now the trace centre; `batch_summary.json['traces']`
  records the counts, the scatter and the window. Why: on a TOF the assignment snaps
  the merge anchor to theory (a formula is only committed where a sample's draw lands
  near it) while the ion's trace sits several ppm away — 22.6 % of one 230-spectrum TOF
  batch's anchors were outside their own trace's window and 26.8 % of its well-populated
  rows shared a trace with a competing label. Measured on that batch's real runs
  (`scripts/eval_trace_stamping.py`): mean stamped coverage over all ions 0.422 → 0.488
  with 38 % of ions gaining more than 5 points and 3.7 % losing, the highly oxygenated
  monomer C10H16O9 0.135 → 0.857, window ±9.5 ppm; two Orbitrap batches moved by
  0.16–0.34 ppm and changed nothing (0.754 → 0.754, 0.542 → 0.542; window ±6 ppm) —
  the no-op that validates the diagnosis.
- **Reference-list context tags are resolved from the batch name, the DATASET name and
  the reagent label** (`assign_batch.run`, the pooled pipeline and the report all pass
  the dataset), and the phrase "AP oxidation" (also `ap-oxidation` / `ap_oxidation`)
  unlocks the alpha-pinene and monoterpene contexts. On a campaign whose batch names
  describe instrument and reagent the chemistry lives only in the dataset name, and the
  alpha-pinene HOM reference list had been silently inactive on alpha-pinene data.
- **The positive pass-0 `cyclosiloxane` and `indoor_sulfur` families apply to EVERY
  positive context**, not only to the labelled-ammonium runs they were seeded from:
  `directors._known_species("positive", …)` returns them regardless of context, so
  uronium and EasyIC runs now also get `known:` locks for D3–D7 / L2–L5 siloxanes and
  for the benzothiazole / DMSO / thiophene / sulfolane / NBBS set. The commit gates are
  unchanged (≥2 channels, or a confirmed ²⁹Si/³⁰Si envelope + the Si-count M+1 check,
  or a confirmed ³⁴S envelope), so this adds locks only where the evidence is already
  there — but a previously Candidate D4 or benzothiazole can now come out Assigned.
- **The pass-4 halogen cap applies in every context, not just positive ones.**
  `residual.stage_a_iso_pairs` drops a ~1.998-Da doublet whenever the context caps that
  halogen at zero (`max_Br` = `max_Cl` = 0) — the rule is written against the context
  profile, so any halogen-free context gets it.
- **The labelled-nitrogen ¹⁴N envelope line is emitted for every `^N`-bearing ion**, so
  it reaches the ¹⁵N-NITRATE profile as well as the ammonium one it was built for: a
  `[M+^NO3]⁻` cluster now predicts a −0.997 Da satellite at `(1 − purity)/purity` of M0
  and `complete_isotope_envelopes` will claim it. **Caveat, to be validated on a
  labelled-nitrate batch:** a labelled-nitrate source can carry a *real* `[X+¹⁴NO₃]⁻`
  analyte channel from the reagent's unlabelled fraction, and it sits at exactly that
  mass and at a comparable 0.6–7 % of the labelled cluster. Such a channel would now be
  attached as an isotope child of the labelled reading rather than standing as its own
  M0. Gating the line per profile is deliberately NOT done here (see the open items).
- `ReagentProfile.purity` is no longer inert. `assign.run` publishes it
  (`isotopes.set_label_purity`) and it now drives BOTH the local scorer's
  `predict_isotopes` call and peaky's own envelope predictor; `isotopes.LABEL_PURITY_15N`
  remains the default. Behaviour-neutral at 0.98, which is also mascope_tools' default.

- **Batch sample selection is one greedy presence set-cover** (`sampling.
  select_cover_samples`, `docs/SAMPLING.md`), replacing the 5-time-spaced+max-TIC
  rule and the brightest arg-max cover. Universe = the batch's m/z bins present
  in ≥ 2 samples — **no height floor**, because the peak picker's detection edge
  spans ~1000× between instruments and modes (0.8 cps on a TOF, ~800 cps on an
  Orbitrap mode with the reagent ion in range), so any absolute floor is a no-op
  on one and blinds the selector on another. Each pick is the sample holding the
  most not-yet-covered bins; the run stops when the next pick would add
  < `--min-gain` (0.5 %) after `--k-min` (6) picks, and `--k-max` (30) is a
  flagged budget rather than a target. The merge keeps whatever any assigned
  sample contained and nothing else, so this selection is what decides recall:
  on a pooled 5036-sample field-campaign table the cover lands at k = 15 and
  holds 94 % of the rare (< 5 % prevalence) ions of a dedicated sub-batch run,
  vs 54 % for the old default and 91 % for the old 12-sample arg-max cap.
  `pool` runs the same single cover over the pooled table (a loud group cannot
  hog a presence objective; per-group achieved coverage is recorded) instead of
  a per-group union. `batch_summary.json` / `run_manifest.json` gain a
  `selection` block (`k`, `n_bins`, `achieved_coverage`, `stop_reason`,
  `next_gain`, `tol_ppm`, `coverage_by_group`); `tables/selected_samples.csv` is
  now in pick order with `pick`, `role` (`cover` / `pad`), `bins_new`,
  `coverage`. `sampling.BATCH_TOL_PPM` (6 ppm) is the one m/z binning tolerance
  for every batch-level operation — the selector bins on it and
  `assign_batch.DEFAULT_TOL_PPM` *is* it, so selection and merge bin identically.
- **Height thresholds in the passes are multiples of the sample's own noise
  edge**, not absolute cps. `assign.run` computes `noise_edge_cps` (the 1st
  percentile of the sample's picked heights) once per sample; `PassConfig.
  height_cutoff` is now a read-only property = `height_cutoff_x_edge` (`None` =
  unset, resolving to 1.0) × that edge, with `height_cutoff_cps` as an override
  for offline callers. `peaky assign --height-cutoff` becomes that override
  (default none) and `--height-cutoff-x-edge` sets the multiple; the MCP
  `assign_sample(height_cutoff=)` likewise. Per-file `noise_edge_cps` and the
  resolved gate `height_gate_cps` are recorded in `batch_summary.json`
  (`height_cutoff_cps` in `run_manifest.json['config']` stays the knob).
- **Eight formula-hunting sites draw their candidates from the admission gate
  instead of brightness alone**: the pass-1 grid, pass-2 series growth, pass-3
  contaminant families and pass-3's two cluster resolvers (HX, acid·I₂) — those
  five share `directors._target_peaks`, so one gate call covers them all — plus
  the pass-6 ladder gap-fill, residual stage B and the siloxane ladder (its work
  set *and* its seed test — the lowest rung of a weak ladder is exactly a
  persistent sub-gate peak, and a brightness-only seed test left it a 2-member
  run). Still brightness-only, a documented follow-up: residual stage A, the
  reflist rescue, pass-3's series *detection* statistics (`series_detect`) and
  the isotope-satellite tests in `passes/postprocess`. Re-derive the gated set
  from the callers of `_target_peaks`, not from a grep for `admissible(`.
  The admission table joins selection and the merge on `sampling.BATCH_TOL_PPM`
  (above), so `batch` and `assign --ts-batch` bin a peak the same way.

### Removed

- `peaky batch --select / --coverage-target / --height-floor` and
  `peaky pool --coverage-target / --height-floor` (`--k-max` stays, default 30;
  `--k-min` / `--min-gain` added to both). `sampling.select_representative_samples`,
  `select_brightest_coverage_samples`, `select_pooled_union` and the `*_sample_ids`
  wrappers; `assign_batch.run(select=, coverage_target=, height_floor=, n_time=,
  include_max_tic=)`, `pipeline.run_batch(select=, …)`, `run_pooled_batches(
  coverage_target=, height_floor=)`; `PassConfig(height_cutoff=)` (use
  `height_cutoff_cps=`); the MCP `run_batch(select=)` (now `k_max=`).

### Fixed

- **`run_manifest.json` stamped the wrong package version on every run.**
  `peaky.__version__` restated `"0.5.0"` as a literal while `pyproject.toml` had moved
  to `0.7.0`, so `code.package_version` — the field that exists to tell one run folder's
  code from another's — read 0.5.0 on runs produced by 0.7.0 code and could not
  distinguish them at all. `__version__` is no longer declared in `peaky/__init__.py`:
  it is resolved on first access from pyproject's `[project].version` (a source
  checkout or editable install, name-checked so a vendored `peaky/` cannot inherit a
  host tree's version), falling back to `importlib.metadata` for an installed wheel and
  only then to a literal that `tests/test_shim.py` pins to pyproject. Resolution is lazy
  and cached, so `import peaky` stays ~0.1 ms. `git.commit` / `branch` / `dirty` were
  always correct and remain the way to trace an existing run folder.
- **`code.module_versions` is derived from the package, and now names every module.**
  The registry was a hand-written dict in `assignment/assign.py` listing only the
  modules that file imports at module level, so it had gone 17 modules stale — among
  them `traces`, `pipeline`, `sampling` and `assign_batch`, i.e. exactly the per-peak
  admission and presence set-cover selection behaviour, all reporting `None`.
  `assign.module_versions()` now walks the package and reads each module's
  `__version__` from its AST (parsed, never imported: importing 38 modules to read a
  string would drag matplotlib and the Mascope SDK into every run, and `assign.py`
  cannot import the `pipeline` that calls it). All 38 modules are registered, the
  result is cached per process, and `tests/test_provenance.py` asserts the registry
  equals what is on disk — so a new module is fingerprinted without anyone
  remembering to list it. `MODULE_VERSIONS` (the dict) is gone; call
  `module_versions()`. `code.module_hashes` already pinned every file by sha1, so
  past runs stayed reproducible — only the human-readable naming was missing.
- **The repo's version strings are pinned to each other.** A new
  `tests/test_versioning.py` asserts the whole set agrees: `peaky.__version__` and
  `_FALLBACK_VERSION` against pyproject, `uv.lock`'s pin against pyproject (so a bump
  that forgot `uv lock` fails offline, not just under CI's `--frozen`), and
  `CITATION.cff` against the CHANGELOG. `CITATION.cff` deliberately LAGS pyproject —
  it names the last release actually tagged, published and archived, which a citation
  has to resolve to, and as of 0.7.0 in pyproject that is still 0.6.0 — so it is
  checked against a dated `## [x.y.z]` CHANGELOG heading (version and date both) and
  for never running ahead of pyproject, rather than for equality with it. The comment
  in the file now says so outright instead of leaving the gap to read as drift.
  The 0.5.0 section heading, which still read "Unreleased", is retitled to name the
  release and its date (2026-06-30): 0.5.0 was tagged and released, and a released
  section labelled unreleased made this file's own history unreadable.
- **The persistence path was inert at the shipped default, and the TOF ledger was a
  flood.** With `height_cutoff_x_edge = 1.0` the brightness path admitted ~99 % of a
  TOF's picked peaks (0.12 % of admissions came from persistence), and the merged
  ledger of a 230-spectrum TOF batch was 64 % single-file with 70 % formula
  disagreement among multi-file rows. The batch-derived floor (above) raises that
  batch to 5× — the live sweep measured 1988 merged rows instead of 5331, 55 %
  single-file, recall of the known monomers unchanged or better — while leaving the
  Orbitrap modes whose picker already stops at the noise edge at exactly 1.0.
- **A peak could fail to find its own occurrence bin.** The table was built by
  single-linkage chaining (a bin breaks only where consecutive peaks are more than
  the tolerance apart) and read within ±tolerance of the bin's height-weighted centre:
  on a TOF batch 91 % of the persistent bins were wider than the tolerance (median 51
  peaks, one 134 ppm), so a large share of the persistence signal was the chaining and
  3149 rows sitting inside persistent bins were denied the path at one operating point.
  Matching the lookup to the bin span handed a chained bin's persistence to adjacent
  noise; capping the bin width split one ion's jitter cloud across bins. The per-peak
  rule (above) has neither failure.
- **The report cover under-reported the persistence path** by an order of magnitude:
  it counted MERGED rows admitted by persistence (0–3 on real batches), but the merge
  keeps the highest tier then score so those rows almost always lose to a
  height-admitted row in the same bin; per-file admissions were 14–63, and on one batch
  the line was suppressed entirely. The cover now reports the per-file total alongside
  the merged count, and states the batch-derived floor and the trace reconciliation.

- **The batch time series carried a peak once per MATCH, not once per peak.** Mascope's
  peak loaders expand a peak into one row per target isotope it matches
  (`load_peaks` / `samples.get_peaks`, `matches=True` by default), and nothing in the
  batch path folded that back down — so `per_file/_batch_ts.parquet` held two rows for
  every peak two targets claimed: same `sample_item_id`, same `peak_id`, byte-identical
  `mz` / `area` / `height`, only the advisory `target_*` columns differing. Targets
  collide exactly when they imply the SAME ion (a neutral read as `[M+NO3]-` and a
  neutral one HNO3 heavier read as `[M-H]-` are one ion formula), so the pair sits at
  exactly 0.00 ppm and no mass filter separates it. Everything downstream counted the
  peak twice: `build_matrix` sums heights per (sample, m/z bin), so that bin's intensity
  doubled, and a per-trace peak count read **2.0 peaks/sample for a single ion** — which
  reads as two merged ions rather than one peak listed twice. Measured on two field
  batches: 0.29 % and 0.42 % of rows, always pairs. New
  `timeseries.collapse_peak_matches` keys on `(sample_item_id, peak_id)` — or
  `(sample_item_id, mz)` for a frame trimmed to the TS columns — and keeps the
  best-scoring match's `target_*` columns, ranking on the match scores with the target
  ids as a final tie-break so the winner is fixed by content rather than by the order
  the server returned rows in. Row order is preserved and a clean frame comes back
  untouched, so it is a no-op on collapsed input and idempotent; it runs wherever a time
  series enters (`pipeline.load` / `run_batch` / `run_pool` / `generate_report` and
  `assign_batch.run`), which also protects sample selection, the amine gate and sidelobe
  flagging. Existing parquets are repaired on read.

- **A missing `--group-by` value no longer crashes pooled selection.**
  `sampling.select_cover_samples(group_col=…)` now normalises the group labels to
  strings before it sorts them: on pandas ≥ 3 `astype(str)` leaves NaN alone, so a
  group column holding any null value raised `TypeError: '<' not supported between
  instances of 'str' and 'float'`. Ungrouped samples stay selectable — their bins
  are real — and group under `sampling.UNGROUPED` (`"(ungrouped)"`), which no
  per-group report matches.
- **A batch whose peaks all have zero height no longer produces a NaN coverage.**
  Every bin failed both prevalence gates, the universe was empty, and the mean of
  that empty array became the `coverage` column, `selection.achieved_coverage` and
  a bare `NaN` token in `batch_summary.json` (rejected by strict JSON parsers),
  with numpy warnings on the way. An empty universe now returns an empty selection,
  like a batch with no bins at all.
- **The brightest-coverage selector silently never reached its coverage
  target**: every run on disk had assigned exactly `k_max` + 2 samples at
  0.39–0.61 achieved coverage while `batch_summary.json` recorded the *requested*
  0.85. The new selector records the achieved coverage and the stop reason, and
  warns (log, report, summary) when the `k_max` budget binds while the batch is
  still gaining.
- **The absolute 100 cps `height_cutoff` blinded the height-gated passes
  (ladders, siloxane, residual, reflist rescue, isotope-satellite checks) on
  low-edge modes**: it excluded 97 % of picked TOF peaks and 87 % of an EasyIC
  mode's, while being a no-op on modes whose picker edge sits above it. The
  edge-relative gate keeps every picked peak but the bottom 1 % eligible on any
  instrument. The gate also bounds which unassigned peaks pass 1 enumerates, so
  on a TOF batch the primary pass had been committing ~2 grid assignments per
  file; with the edge gate it commits ~370 (self-calibration backbone 220–460
  peaks, median |ppm| 0.87 vs 1.13 before), most of them Candidate-tier leads
  at 1–5× the edge — a TOF ledger now carries that Candidate tail by design.
  `peaky assign --height-cutoff-x-edge` (the single-sample command; `batch` and
  `pool` gate at the 1.0× default) raises the bar when a tighter list is wanted.
  An EasyIC batch went from 63 to 119 merged M0 with no formula disagreements.

- **A reference list's radicals are read off the formula, not the hydrogen count.**
  `reflists.load_catalog` sorted each species into the closed-shell pool (matched by
  default) or the radical pool (skipped unless `include_radicals=True`) by a
  hand-set `radical` flag, and the monoterpene HOM list had set that flag from an
  odd hydrogen count, which is wrong once nitrogen is present. Its 118 organic
  nitrates (C10H15NO8, C10H13NO10, ...), the main HOM class under NOx, sat in the
  radical pool, where the selection prior, the rescue and the report's
  corroboration never looked, and three nitrogen-bearing radicals were matched
  instead. The loader now reads radical status off DBE parity (a half-integer DBE
  is an odd-electron neutral) and treats the flag as a claim, warning when a list
  disagrees. The list's flags are corrected: 573 closed-shell and 257 radicals,
  where they said 458 and 372 (`data_version` 2024.2).

- **The Keller contaminant list holds molecules only.** Nine entries were ions or
  salts: the CN fragment of acetonitrile, the tetrabutylammonium, trityl and
  monomethoxytrityl cations, and five quaternary-ammonium chlorides, four of them
  at a negative DBE. They are dropped. NMP, entered as its protonated ion, is the
  neutral C5H9NO — which puts its [M+H]+ back on the m/z 100.07569 the source
  measured, so the list is 48 of 50 verified against the source masses rather
  than 47 — and acetic and propionic acid are named for the acids rather than the
  iron complexes the source saw. The list goes from 59 species to 50
  (`data_version` 2008.2), the split Mascope's reference seed uses.

- **`peaky assign` unlocks its reference lists.** Only the batch path ever called
  `active_lists`, so a single-sample run passed `reflists_active=None`: the
  selection prior was empty, the rescue-verify pass never fired, and the manifest
  key above — new in this same release, so it never shipped empty — would have
  landed `[]` on the single-sample path. `reflists.activate` is now the single
  unlock step (`resolve_context_tags` → `active_lists`, tags returned so the
  caller can log them), and `cli.cmd_assign`, the MCP `assign_sample` tool,
  `assign_batch.run` and the report path all go through it. A lone sample has no
  batch name — its metadata is the context plus the reagent profile's label, the
  same pair on both single-sample paths — so it activates the always-active
  lab-contaminant list and nothing chemistry-specific unless one of those names a
  chemistry; `peaky batch` is still the way to unlock a HOM list.

- **A formula is validated before its parity is trusted.** `chemistry.dbe` scores
  an element outside the mass table as divalent — sodium acetate comes out at
  DBE 1.5 and would have pooled as a radical with no message — and
  `parse_formula` ignores charge and bracket notation, so an anion written
  `[C10H14NO8]-` would have loaded as a closed-shell neutral. `load_catalog` now
  warns and skips (counted, on `ReferenceList.skipped`) any species whose formula
  uses an unknown element, does not round-trip as a Hill-notation neutral, or has
  a negative DBE. The bundled lists skip nothing.

- **One parity test for the whole package.** `chemistry.odd_electron` is now the
  single DBE-parity helper the closed-shell grid gate (`dbe_ok`), the
  plausibility radical exemption and the reference-list loader all call, instead
  of three copies of the same half-integer test; `dbe_ok` calls a half-integer
  DBE "odd-electron (radical) — blocked on the closed-shell grid" rather than
  "not a valid neutral", which is what the reference lists had always called it.

- `tests/test_peaklists.py` loads every bundled list and fails on a species with
  a negative DBE or a radical claim its parity contradicts, and holds the
  corrected pool to its consumers: an organic nitrate's ion mass is matched by
  `match_by_mass` / `match_assigned` and carried into the selection prior, a real
  nitrogen-bearing radical's is not, and both are only with
  `include_radicals=True`. Mascope keeps the same test on its own copy of the
  lists.

### Added

- **`peaky/batch/traces.py`** — the one trace primitive over the m/z-sorted batch peak
  list: `PeakIndex` (exact-duplicate rows dropped, deterministic sample codes;
  `occurrence` — the per-peak distinct-spectrum sweep; `occurrence_at`; `coverage_at`;
  `members` — the nearest peak per spectrum within the window; `mean_shift` — the
  local mode of the peak density, capped in drift; `scatter_ppm` — the robust per-ion
  mass scatter, re-measured in a wider window when the first estimate says the cloud
  fills it; `sample_edge` / `height_in_edges`) and `batch_scatter_ppm`. Admission, the
  trace reconciliation of the merged ledger and the stamp all read the batch through
  it, so a peak's occurrence, its trace and its stamp are one object at one rule.
- `admission.derive_height_cutoff_x_edge` / `persistent_trace_count`,
  `timeseries.recentre_ledger` / `collapse_trace_labels` / `stamp_tolerance`, and
  `scripts/eval_trace_stamping.py` (score the trace stamp against the anchor stamp on a
  finished run, per ion and against a list of ions known to be present — the benchmark
  rule for any stamping change is coverage of KNOWN ions, never how many rows were
  admitted).
- `tests/test_traces.py`: a TOF-like batch whose ledger anchor is snapped to theory
  while its trace lives 6.5 ppm away (coverage rises from a minority to > 80 % of
  spectra end to end), a phantom competitor on the same trace losing to the multi-file
  label despite its higher `ion_score`, the low-evidence guard, an Orbitrap-like batch
  where nothing moves, and the derived floor staying at 1× where the picker stops at
  the noise edge and rising where it picks into the noise.

- **`peaky publish-batch <run_dir>`** - publish a `peaky batch` run's merged ledger
  onto Mascope's batch ledger as a batch run of its own (Mascope's
  `POST /api/batch-peaks/batch/{id}/runs/import`). Each merged m/z lands on the
  nearest batch peak and the server measures peaky's formula against every sample
  that holds the peak, so the batch ledger shows Mascope's fit under the `peaky`
  name; adducts are resolved to mechanism ids by default (a row without one lands
  nothing), ion formulas come from the per-file ledgers or are derived, and
  `--dry-run` shows the payload. `docs/PUBLISH.md` gained a section.
- **¹⁵N-labelled ammonium reagent profile `NH4_15N`** (`^NH4+` ionisation mode, server mechanism
  `+^NH4+`; aliases `15nh4`, `^nh4+`, `ammonium-15n`, …) with the `[M+^NH4]+` adduct
  (+19.0309, `chemistry.ADDUCT_SHIFTS`), both mechanism maps, the `ammonium-15n` context
  (every context alias names the label — there is no bare `ammonium` / `nh4` / `nh4-cims`
  alias that would hand an unlabelled-ammonium user the ¹⁵N channels; an unknown context
  raises), an `ammonium15N` reagent-cluster library (`[(^NH3)n+H]+` + hydrates + the ¹⁴N
  monomer), `[M+^NH4]+` on the siloxane/PDMS/phthalate/PEG families, and the relabel-only
  dehydration aliases `[M+^NH4-H2O]+` / `[M+H-H2O]+` in `_DIFF_TO_ADDUCT`. Built from the
  2026-09-10 exploratory file: ¹⁴N/¹⁵N adduct pairs 0.018–0.021 → purity 0.98, reagent
  ions below the 40 Da window → TIC normaliser, `label_isotope=None`. `assign.run` drops
  `[M+NH4]+` from the opportunistic channels on a labelled-ammonium run (it would only
  re-claim the 2 % ¹⁴N satellites). New post-tier stage `nh4_dehydration`
  (`cleanup.relabel_ammonium_dehydration`): the MS2-proven declustering cascade
  `[M+^NH4]+ → [M+H]+ → [M+H-H2O]+` / `[M+^NH4-H2O]+` re-reads the alkene/enone
  readings of X−H2O onto the corroborated hydrate X (own-adduct-weakness gate — the
  parent-relative fallback applies only to an alkene with no protonated form of its own —
  second loss annotated only). `docs/REAGENTS.md` callout, `docs/ASSIGNMENT_DETAIL.md` §3.7b,
  `tests/test_nh4_15n.py`.
- **Labelled-reagent ¹⁴N line in the envelope predictor** (`isotopes._per_atom("^N")`, at
  the active reagent purity, default `LABEL_PURITY_15N` 0.98): a `^N` ion now predicts its
  −0.997 Da `14N` satellite, so
  `complete_isotope_envelopes` claims it — and displaces a pass-1 CHON `[M+H]+` mass-fit
  sitting on it when it matches the predicted 2 % of a ≥10× brighter labelled parent
  (pass-0 locks kept). Before, only 7 of the ~340 `[M+^NH4]+` satellites were attached.
- **Pass-4 iso-pairs respect the context halogen caps** (`residual.stage_a_iso_pairs`): a
  ~1.998-Da doublet in a halogen-free positive run (max_Br = max_Cl = 0) is no longer
  read as a Br/Cl pair (7 `C5H7BrO3 [M+^NH4]+` phantoms on the ammonium file).
- **Positive pass-0 known species**: `cyclosiloxane` (D3–D7, L2–L5; gate ≥2 channels OR a
  confirmed ²⁹Si/³⁰Si envelope + the Si-count M+1 check) and `indoor_sulfur`
  (benzothiazoles, dithiocarbamate ester, thiazoles, DMSO/DMSO₂, DMDS/DMTS, thiophenes,
  sulfolane, NBBS; gate ≥2 channels OR a confirmed ³⁴S envelope). Passes 1/2 are CHO(N)-only
  and no positive pass-3 family opens S, so benzothiazole `[M+H]+` (68 kcps, MS2: −HCN →
  C6H5S⁺) was unexplained and D4 only a low siloxane-ladder Candidate.
- Labelled-ammonium runs score NO opportunistic channel: `[M+Na]+` sits 0.2 mDa from
  `[(X−O2+C2H4)+^NH4]+` (Na − ^NH4 = 3.9584 Da vs C2H4 − O2 = 3.9585 Da), so every ^NH4
  adduct of an O≥2 neutral had a hydrocarbon·Na twin the complexity prior preferred
  (114 Na fits; palmitic acid read as C18H36·Na⁺). Pass-3 family adduct lists drop
  `[M+NH4]+`/`[M+Na]+` on such runs too. The `ammonium-15n` context admits H/C up to 3.0
  (C3–C4 amines). The `ammonium15N` library adds the urea crossover and the reagent-derived
  CO/CO₂ clusters ((¹⁵NH₃)₂H⁺·CO at 65.049 = the doubly-labelled "formamide" ion) and the
  reagent-made ¹⁵N-acetamide ions (61.041 protonated, 79.065 as its ¹⁵NH₄⁺ adduct: the
  79/61 ratio 0.18 equals the ambient ¹⁴N acetamide's 78.068/60.044 = 0.21, so 61.041 is
  the amide, not ambient ketene·¹⁵NH₄⁺ of the same composition).
- `relabel_ammonium_dehydration` brightness ceiling: a dehydration product may not exceed
  1.5× its parent's strongest form (adduct or protonated) — a 2 kcps C3H8O2 parent no
  longer claims 9 kcps acetone `[M+H]+` as its water loss (ambiguity note instead).
- Pass-3 family `organosulfur` (S 1–2 on `[M+H]+`/`[M+^NH4]+`/`[M+NH4]+`/urea; opened by the
  `ammonium-15n` context, ³⁴S-gated by the tier engine): reaches the C5H12N2S / C9H18N2S /
  C11H22N2S `[M+H]+` thioureas (each with a 4 % ³⁴S line) that no other positive pass can;
  tetramethylthiourea (C5H12N2S) added to the `indoor_sulfur` known list.
- **Mass-dependent calibration centre** (`peaky/assignment/masscal.py`, new): the pass-1
  self-calibration and the tier gate now also fit `ppm = a + b·1000/mz` (b = the constant
  absolute offset in mDa) on the backbone; `z_of(ppm, cfg, mz=)` and `tiers._cal_z` judge a
  peak against the centre at ITS m/z, clamped to the backbone's own m/z coverage
  (`cal_mz_lo` / `cal_mz_hi`), so the trend is never extrapolated onto masses the backbone
  never saw. On the 2026-09-10 file the Orbitrap's low-mass residual was −0.12 mDa (−2 ppm
  at m/z 61 vs −0.2 ppm above 160, MAD 0.13–0.22 ppm), so a constant centre rejects every
  bright sub-80 ion (ketene·¹⁵NH₄⁺, urea·H⁺, acetamide, acetic acid, the amines) at z = 6.
  Callers without an m/z keep the constant model unchanged; a flat source keeps it exactly.
  **The fit is accepted only when** `|b| > 3·SE(b)` (`SLOPE_MIN_SE`) **and** the trimmed
  residual RMS is ≤ 0.8× the constant model's on the same points (`TREND_SIGMA_RATIO_MAX`)
  **and** `|b| ≤ 0.5 mDa` (`MAX_ABS_OFFSET_MDA`, a physical cap on the calibration curve's
  residual absolute offset) **and** each half of the fitted `1000/mz` range holds ≥ 5 kept
  points (`MIN_SIDE_N`, the lever guard): a 2-SE rule would be a 5 % two-sided test, i.e. a
  phantom trend on a FLAT source in ~6–7 % of samples at any n (Monte Carlo, 21–300 rows),
  and a backbone whose only low-`1000/mz` point is one corroborated outlier is a lever, not
  a trend. `masscal.centre` / `sigma_at` are the single implementation behind
  `passes.core.cal_center` / `cal_sigma_at` / `z_of` and `tiers._cal_z`.
  The fitted trend is run-derived, so `provenance`'s reproducible config fingerprint
  excludes `cal_a` / `cal_b` / `cal_sigma_trend` / `cal_mz_lo` / `cal_mz_hi`: `calibrate`
  writes them onto the shared `cfg`, so without that the last sample's numbers land in
  `run_manifest.json` and two identical re-runs of a batch fingerprint differently.
  `cal_mu` / `cal_sigma` are excluded for the same reason, now that the pipeline hands the
  provenance manifest the same `cfg` the per-sample loop fits: leaving them in would keep
  the fingerprint dependent on whichever sample finished last. This is a manifest schema
  change — manifests written before it carry the two fields as a record of the run's
  calibration centre, and `batch_summary.json` still reports the per-file offsets.
- `cal_abs_floor_mda` (0.03 mDa, `PassConfig`, default `masscal.ABS_FLOOR_MDA`): an absolute
  floor on the trend sigma, active only below ~m/z 120, where the Orbitrap's residual
  curves faster than 1/mz (dimethylamine `[M+H]+` at 46.065 sat 0.9 ppm = 0.04 mDa off the
  fitted trend). `confidence_label(..., mz=)` / `cal_center` grade against the trend centre
  too, so an on-trend −2 ppm sub-80 ion is Good, not Low. `PassConfig.cal_abs_floor_mda` is
  the single owner of the floor: `apply_tiers` / `compute_tiers` take the run's `cfg` and
  carry the value onto `tiers._Cal`, so overriding it moves the commit gates and the tier
  gate together.
- Reference peaklist `isoprene_ox_wennberg2018` (27 closed-shell isoprene oxidation products, Wennberg et al. 2018) added to `peaky/data/peaklists/`, gated by the new `isoprene_ox` context (batch keywords isoprene/ISOPN/IEPOX/ISOPOOH/methacrolein) and `biogenic_soa`/`ambient_summer`; rescues the isoprene dihydroxy-dinitrate C5H10N2O8 as an isotope-confirmed Assigned in the 2026 field-campaign ¹⁵NO₃⁻ data.
- **A run records which reference lists it had active** — `reflists.active_versions`
  puts `reflists_active: [(id, data_version), ...]` in the single-sample manifest
  (beside the selection prior it builds from the same lists) and in
  `batch_summary.json`. A list's closed-shell/radical split changes with its
  `data_version`, so a ledger now says which revision shaped its prior and its
  rescue instead of silently depending on the installed copy.

- **A reagent profile can carry its own height-gate multiple**
  (`ReagentProfile.height_cutoff_x_edge`, `docs/REAGENTS.md` §3a). The gate is a
  multiple of the sample's own noise edge, and the right multiple belongs to the
  **peak picker**, not to the reagent: a picker that stops at the noise edge
  wants 1.0 (rare real ions sit at 1–3× the edge there, so raising it discards
  them), while a picker that picks *into* the noise admits nearly everything it
  found at 1.0 — on one 230-spectrum time-of-flight batch, 1.0 merged 4346 ions
  of which 3307 were seen in a single file only, where 5.0 kept 57 % of the
  picked peaks and 74 % of the assigned ones, i.e. a tighter candidate list for
  the height-gated passes. So the global default stays **1.0** (now the single
  constant `passes.config.DEFAULT_HEIGHT_CUTOFF_X_EDGE`, read by both
  `PassConfig.height_cutoff_x_edge_resolved` and the fallback), **no bundled
  profile sets the field** (pinned by a test), and behaviour out of the box is
  unchanged. A site
  raises it for its own instrument from a `--reagent-config` file —
  `height_cutoff_x_edge` is now a loadable reagent-config field — with no code
  change. `profiles.resolve_height_cutoff_x_edge` implements the order once —
  an explicit `--height-cutoff-x-edge` / cfg value > the profile's value > the
  package default — and every entry point that resolves a profile and builds a
  `PassConfig` applies it (`peaky assign`, `assign.main`, `assign_batch.run`,
  `pipeline.run_batch` / `run_pooled_batches`, the MCP `assign_sample`,
  `scripts/certify_neutrals.py`). `pipeline.run` builds no `PassConfig` of its
  own; it resolves the profile and *reports* the multiple the assign stage will
  use as `height_cutoff_x_edge` in its return value.
  `peaky assign --height-cutoff-x-edge` and `PassConfig.height_cutoff_x_edge`
  both default to *unset* (`None`) rather than to 1.0, so explicitness is read
  off the value instead of guessed by comparing it to the default: an explicit
  multiple that happens to **equal** the global default still outranks a profile
  that carries a higher one. The resolved multiple is
  recorded: `batch_summary.json` gains `height_cutoff_x_edge` +
  `height_cutoff_x_edge_source` (and each file's `height_cutoff_x_edge` beside
  its `height_gate_cps`), and it stays in the `run_manifest.json['config']`
  fingerprint, which one log line per run mirrors.
- **Admission gate: persistence OR brightness** (`assignment/admission.py`).
  A peak is eligible for formula search at the eight gated sites (pass-1 grid,
  pass-2/3 target peaks, ladder gap-fill, residual stage B, siloxane ladder —
  the full list and its derivation are under *Changed*) if `height >=
  height_cutoff OR occurrence >= threshold`, where occurrence is the fraction
  of the batch's spectra whose m/z bin holds a peak (bins = `timeseries.
  build_matrix` at `sampling.BATCH_TOL_PPM`; computed once per batch from the
  time series) and the threshold is **derived from the batch**
  (`--occurrence-min auto`: Otsu's split of the bimodal bin-occurrence
  distribution, clamped to 0.25–0.75, 0.40–0.55 on every instrument measured;
  a fixed 0.8 was tried first and discarded two-thirds of the recurring weak
  ions; a number overrides, `0` disables, < 10 spectra switch it off). The
  resolver's decision is FINAL: when it switches the path off, the gate stays
  off, so a numeric `--occurrence-min` cannot re-open a path the spectrum count
  (or an occurrence distribution with no split) closed. That is what makes an
  empty `admitted_by` mean exactly "not eligible at the gated sites".
  Noise does not recur at a fixed m/z; ions do: on a 230-spectrum mixed-reagent
  TOF batch the old absolute cutoff kept 32 of 4025 bins while ~500 recur in
  > 80 % of spectra at a median 3–4 cps, and 21 highly oxygenated molecules an
  Orbitrap saw on the same air all sit in that recurrent population at ~2 cps
  (mass-shifted decoys: none). The path is additive and admits peaks for
  consideration only — confirmation rules are unchanged, and **persistence
  gates entry while only corroboration gates the tier**: an occurrence-admitted
  M0 with no isotopologue / cross-channel / series corroboration is capped at
  Candidate (`tier_reason` `persistent-weak`), because at that intensity
  nothing constrains which formula the real ion got. Per-file ledgers record
  `occurrence` (float in [0, 1], NaN without batch context) and `admitted_by`
  (`height` / `occurrence` / `''` = not eligible at the gated sites), the
  merged ledger carries the winner's, `batch_summary.json` gains an `admission`
  block (`occurrence_min` = the knob, `occurrence_threshold` = the resolved
  fraction, `n_bins`, `n_persistent_bins`, `n_spectra`, `tol_ppm`) and per-file
  `admitted` counts, `run_manifest.json` records that block under `counts` and
  the module version under `code.module_versions`, a run whose persistence path
  is off logs which of the four reasons applies (`admission.why_off`),
  and the report cover states how many merged peaks were eligible by
  persistence only (with the resolved threshold and the knob).
  `PassConfig.occurrence_min`; `assign.run(occurrence=)`; `peaky assign
  --ts-batch` computes the table for a single sample. `batch` / `pool` also
  gained the absolute `--height-cutoff` override and `--height-cutoff-x-edge`,
  which only `assign` had (the three flags are defined once for all three).
- **`--progress`: a live progress window for a run** (`peaky/progress.py`, new).
  `peaky assign|batch|pool --progress` opens a small Tk window with a samples bar,
  a within-sample stage bar, elapsed + ETA, and — when the run ends — the run's
  own stats (merged M0, tiers, in-all-files, single-file, formula disagreements)
  next to how long it took. Opt-in, so scripted and skill-driven runs are
  untouched; `PEAKY_PROGRESS=1` also enables it. Everything in the window is also
  on stdout, so nothing is lost by never seeing it.

  **`peaky assign` is a one-sample run and reports as one.** Its samples bar
  reaches 1/1 and its stage bar fills when `assign.run` returns: the `(i/N) done`
  line that drives both comes from `assign_batch`, which that path never goes
  through, so the command marks its own sample complete. It shows **no ETA** —
  an ETA is extrapolated over completed samples, and the only sample here
  completes when the run does; the window omits the field rather than printing a
  clock that cannot move.

  **The hold is for a person at a terminal, and it is bounded.** A finished run
  keeps its window up so the stats panel can be READ — but only when stdin and
  stdout are both a tty, and only until you close the window, press Ctrl-C, or
  `PEAKY_PROGRESS_HOLD_S` seconds pass (default 600; `0` disables the hold
  entirely). A pipe, a CI job or a skill-driven run therefore never waits on a
  window, whatever `PEAKY_PROGRESS` says, and Ctrl-C during a run closes the
  window immediately, with no stats panel and no wait.

  It is a **`log` wrapper, not a pipeline change**: peaky already threads
  `log=print` from `run_batch` down to each assignment stage, so `Reporter` is a
  drop-in for `print` that forwards every line untouched and reads the lines it
  recognises into a progress model. Nothing in the pipeline imports `progress.py`
  or knows a window exists — the log stream is the whole interface, and
  `tests/test_progress.py` pins the literal log strings the pipeline emits
  against the patterns parsed here so a rewording fails a test instead of
  silently flat-lining the bar. Never fatal: no display, no tkinter, or any UI
  exception degrades to a one-line terminal status and then to silence.
  **macOS always takes that fallback**: the window runs on a daemon thread, and
  Tk/Cocoa driven off the process's main thread aborts the process outright —
  not an exception any guard could catch — so on Darwin the progress window
  would kill the run it reports on. There is no window there, by design.

  Parallel runs (`--jobs > 1`) report at **sample granularity only** and say so
  ("N workers" in place of the stage bar): workers buffer their logs and the
  parent replays them after the reduce, so a stage bar driven from them would
  animate a lie.
- **Runs are timed.** `batch_summary.json` gains `elapsed_s` (+ the `n_jobs` that
  produced it — a duration is meaningless without it), and `run_batch` /
  `run_pooled_batches` return a whole-pipeline `elapsed_s` and log it. Run-time
  metadata only: the reproducibility fingerprint hashes `merged_ledger.csv` and
  the input TS, not the summary, so determinism is unaffected.
- **`[phase] <name>` log markers** for the pipeline steps with no per-item
  progress of their own (fetch / select / assign / cluster / vankrevelen /
  report / provenance) — readable in a plain log, and what the window's phase
  line reads. `run_batch` and `run_pooled_batches` emit the same set, so the
  phase line is as truthful on a pooled run as on a single-batch one; `select`
  comes from whichever side runs the set-cover (the pooled pipeline itself, or
  `assign_batch.run` when it picks the cover for one batch, which brackets it
  back to `assign` as soon as the picks are in).

## [0.7.0] - 2026-09-03 (publish a peaky run into Mascope)

### Added

- **Contributor License Agreement** — external contributors now accept the Ultra
  Trace Systems Individual Contributor License Agreement once, on their first
  pull request, by replying to the CLA assistant's comment
  (`.github/workflows/cla.yml`); the `CLA Assistant` check blocks merging until
  every commit author has. The agreement and the signature register are shared
  with Mascope in [ultra-trace-systems/cla](https://github.com/ultra-trace-systems/cla),
  so one acceptance covers both projects. New `CONTRIBUTING.md` explains it.
- **`peaky publish` — upload a finished ledger into Mascope's peak-assignment run
  ledger** (`io/publish.py`, `docs/PUBLISH.md`). This closes the loop peaky was
  missing: a run stops being a local file and becomes a run in the same store the
  in-app engine writes to, visible in the run selector, the peak inspector, the
  batch Assignments overview and the verification loop — which is what makes the
  two engines comparable on one sample. Implements the client half of Mascope's
  run-import contract: chunked assembly sized by serialized bytes as well as row
  count, a row-offset `chunk.index` and an `import_id` so a retried chunk is a
  replay rather than a duplicate, and resume via `--import-id`.

  Three parts of the translation are **not** field-for-field, and each is a place
  a naive mapping is silently wrong:
  - **Two tiers, and neither is a copy of the other.** peaky tiers mechanically
    (window uniqueness, corroboration, degeneracy, O-count); Mascope tiers by
    thresholding a row's _evidence_ — fit weighted by the formula's chemical
    plausibility — against the run's declared bands. The published row carries
    both: `engine_tier` is peaky's verdict, on committed M0 rows only (null
    elsewhere, and absence is not agreement), while `tier` is **not sent at
    all** — it is a pure function of inputs the server already holds, so the
    server derives it and the two implementations cannot drift apart at a band
    edge. On a real ledger this preserved 195 disagreements that were
    previously flattened away. `--dry-run` reports both distributions and the
    disagreement count.
  - **The intensity is instrument-determined**: heights for Orbitrap, areas for
    TOF. It scales the sample's consensus vote in the batch view while the unit
    label is derived separately, so the wrong one mis-weights the sample with
    nothing able to detect it. `--intensity auto` reads the instrument and
    refuses to guess when it cannot classify it.
  - **Vocabularies differ where the shared lineage suggests they would not**:
    peaky's `unexplained` role is Mascope's `unassigned`, and peaky's capitalized
    tiers are not the server's spellings.

  Synthetic de-blending sub-peaks are excluded (they exist in no Mascope peak
  file), an `iso_child` publishes its owner's formula and `parent_peak_id` as the
  owner reference, non-finite floats become nulls, and the reserved provenance
  keys the server strips are dropped locally so the summary says so. Verified
  end to end against a Mascope dev server: 1319 rows over two chunked requests,
  every owner link resolved, the server's derived evidence written on exactly the
  scored rows, no reserved key stored, and the batch fold-in run.

  A **live contract tripwire** (`tests/test_publish_contract.py`, opt-in behind
  `MASCOPE_LIVE=1` like the existing live smoke, which CI asserts is unset) runs
  the real protocol against a real server and reads every contract point back:
  the server-derived tier, the engine tier, the resolved mechanism, the owner
  link, the alternatives shape, the server's own evidence and plausibility, and
  the absence of the reserved keys. The offline suite pins what peaky sends;
  only this catches the server's contract moving underneath a client that owns
  a private copy of it.

  Cosmetic gap, reported by the command itself: `--dry-run`'s preview of the
  tiers Mascope will derive needs
  `mascope_tools.composition.heuristic_filter.formula_plausibility`, which is not
  in the 2026.6.25 release the dependency resolves to. Without it the preview
  assumes a plausibility of 1.0 and reads high for a formula the server weighs
  down. Nothing is at risk — the tier is derived server-side and the preview is
  not published. Upgrade `mascope-tools` once a release exports it.

## [0.6.0] — 2026-08-12 (report refactor + modern-server-only I/O)

### Removed (legacy workspace-server support)

- **The legacy (workspace-based) server code paths are gone** (`io/io_mascope.py`):
  `_patch_datasets_list_for_legacy_servers` (no callers), the `_legacy_*` raw-endpoint
  helpers, `resolve_batch_id`, `_legacy_load_batch_peaks`, and the silent fallback arms
  in `list_datasets` / `list_batches` / `fetch_batch_samples` / `fetch_batch_peaks`.
  Every current server is datasets-based; the fallbacks actively harmed debugging —
  each modern-path failure was swallowed by a bare `except Exception` and re-surfaced
  as the legacy path's own unrelated error (a bare `/api/sample/batches` GET that
  modern servers reject with 422), masking the real cause. The helpers now make one
  modern call, raise a clear error on an empty result, and let SDK errors propagate
  unmasked (regression-tested). Committed run outputs (root-level report PDFs, figure
  PNGs, class time-series CSVs) were also removed from the repository — runs belong in
  the git-ignored `output/`.

### Changed

- **Brand**: Karsa Oy → Ultra Trace Systems Oy (NOTICE, CITATION.cff, repository URLs).
- **A real-SDK contract tripwire** in `tests/test_io_mascope.py` runs peaky's filter
  helpers through the installed SDK's actual matching code offline, so an SDK contract
  change fails CI the day it lands. The suite now also passes on Windows (tempdir-cwd
  teardown, hardcoded `/tmp` paths).

### Fixed (batch-name resolution vs the current mascope-sdk)

- **`peaky batch` matched no batch on current SDKs and died on a legacy-endpoint
  422** (`io/io_mascope.py`). SDKs through 2026.7.7 resolved their two batch
  filters with OPPOSITE conventions — `load_peaks(batches=)` escaped a plain
  string itself (literal substring; only a compiled pattern was a regex), while
  `samples.list(batch=)` used a plain string as a RAW regex and crashed on
  compiled patterns — so the previous one-size `re.escape()`d string was escaped
  twice by `load_peaks`, silently matched nothing, and every batch run fell
  through to the legacy `/api/sample/batches` fallback, which modern servers
  reject (dataset_id required). **mascope-sdk 2026.8.12 unified the contract**
  (a plain string is a case-insensitive literal substring on every name filter;
  only a compiled `re.Pattern` is a regex), so batch names are now passed RAW —
  the interim `escape_batch()` / `literal_batch_pattern()` split introduced on
  this branch is gone again; `fetch_pooled_peaks` still compiles its user regex
  un-escaped. The contract is pinned by a real-SDK offline tripwire (which also
  fails on the SDK's deprecation shim), and `mascope-sdk>=2026.8.12` is the
  floor — keep the SDK at the latest release (SKILL.md gotcha).

### Added

- **`_batch_ts.parquet` stamps every KNOWN ion, analyte or not** (`batch/timeseries.py`
  `identified_rows`/`stamping_frame`, `batch/assign_batch.py`). Three new columns:
  `role` (M0 / reagent / iso_child / artifact), `ion_formula` (the detected ion's
  formula — from the ledger for reagent clusters, the PARENT's for isotope
  satellites) and `iso_label` (13C/81Br satellite labels; the reagent line's
  isotopologue tag, e.g. `79Br+81Br`, so heavy lines of one formula stay distinct
  traces). Motivation: on the 2026-07-21 iodide batch 465 of 695 m/z tracks looked
  "unassigned" when only 206 were actually unknown — the 10 reagent-ladder tracks
  alone are 76.7 % of batch signal and fully identified in the ledger. The file
  goes from ~20 % to ~96.5 % signal-labelled for external consumers; satellites
  deliberately carry NO `neutral_formula` so per-neutral sums cannot double-count.
  The one-to-one/consensus contest now also dedups the non-analyte tracks
  (`dup_candidate` semantics unchanged). Consumer contract:
  `ion_formula.notna()` = identified, `neutral_formula.notna()` = analyte.
- **`[M-H+I2]-` — the deprotonated-acid · I₂ cluster channel (iodide CIMS)**
  (`chem/chemistry.py`, `chem/profiles.py`, `assignment/passes/{core,directors}.py`,
  `assignment/series_gka.py`). After the off-grid iodine fix (below), the I₂
  clusters of ledger acids sat in the unexplained residual: 298.8073
  (`[HCOOH-H+I2]-`), 312.8229 (acetic), 314.8020 (carbonic), 328.8179 (glycolic),
  331.7921 (HNO₄) — each exactly degenerate with the covalent organo-iodine
  `[M+I]-` reading the series passes used to invent (`CHIO2` et al.), and
  unreachable as `[M+I2]-` because the deprotonated neutral is an open-shell
  radical. Follows the `[M+HBr+Br]-` pattern: a relabel-only decomposition alias
  (registered in `ADDUCT_SHIFTS` + `_DIFF_TO_ADDUCT`, deliberately NOT in
  `ADDUCT_TO_MECH`), claimed by a pass-3 resolver (`_resolve_acid_i2_clusters`)
  that scores the covalent alias `(A-H+I) [M+I]-` — the identical ion — and
  commits `neutral = A, adduct = [M-H+I2]-` onto UNEXPLAINED peaks only, for
  acid anchors (O≥1, C/N/S≥1, H≥1, no I) and their ±CH₂ homologs.
  `_prefer_adduct_reading` gains the matching iodide branch (covalent mono-I
  `[M+I]-` winner → acid `[M-H+I2]-`; `[M-H]-` winner → the generic HI
  subtraction, `CIO2- == CO2·I-`), guarded so the pass-0 `reactive_iodine`
  registry species (HOI, INO₂, INO₃, CINO, ICl…) are NEVER re-read — commit
  order (pass 0 locks first) plus the registry guard keep the contested
  `HOI2-`/`I2NO2-` lines with the time-behaviour ruling: ambient iodine
  analytes, not acid clusters. The inverse of the Br organic-acid lesson: the
  new cluster channel must not bury real analytes, and the real iodine species
  must not be dissolved into clusters.

### Fixed (found by the first end-to-end iodide batch)

- **A profile channel missing from `_DIFF_TO_ADDUCT` was silently relabelled
  `[M-H]-`** (`assignment/passes/core.py`). The map turns an ion-vs-neutral element
  difference back into an adduct label and falls through `.get(diff, "[M-H]-")`, so
  a registered-but-unmapped channel does not error — it writes a DEPROTONATION to
  the ledger and inflates the `[M-H]-` census. `[M+I2]-` commits were reported as
  `[M-H]-` (2 merged ions, ~51 TS rows). Added `[M+I2]-`/`[M+I3]-` **and the five
  Br cluster channels that had the same latent gap** (`[M+Br2]-`, `[M+Br3]-`,
  `[M+HBr+Br]-`, `[M+HBr+Br2]-`, `[M+HBr+CO3]-`), plus a test that round-trips
  EVERY channel of EVERY registered profile so the class of bug cannot recur.
- **Covalent iodine could reach a neutral through the series passes**
  (`chem/contexts.py`, `assignment/residual.py`). `ambient-air` had `max_I=1` while
  `max_F`/`max_P` were already 0, and `min_C_for` (the reagent-alias guard that
  protects Br and Cl) had no `I` entry — so pass-2/pass-4 extrapolated `+CO`,
  `+C2H2O`, `+O` off the pass-0 iodine anchors into `CHIO2`, `CHIO3`, `C2H3IO2`,
  `C2H3IO3`, `INO4`. Each is really the I₂ cluster of an acid already in the ledger
  (`CHIO2 [M+I]-` ≡ `[HCOOH−H+I₂]-`) and is un-confirmable, ¹²⁷I being
  monoisotopic. Now `max_I=0`, consistent with the F/P policy; `run_pass0_known`
  bypasses the context filter so the real reactive-iodine chemistry is untouched,
  and the `water` context keeps `max_I=2` (iodinated DBPs are the analyte there).
  `residual.stage_b_series` additionally applies `filter_by_context` — it had only
  the STRUCTURAL gates (DBE + oxygen cap), which is why the pass-0 species became
  springboards; a test pins that DBE/oxygen alone do not reject `CHIO2`.
  Batch verification: iodine-bearing neutrals 20 → 12 (all pass-0), **leaks 8 → 0**.

### Added (sidelobe-contaminated ion channels — trust the formula, not the height)

- **`timeseries.flag_sidelobe_channels`** + two new merged-ledger columns,
  **`intensity_suspect`** (bool) and **`sidelobe_parent_mz`** (float), carried onto
  every stamped row of `_batch_ts.parquet`. A saturating peak rings, and when an
  assigned ion's m/z lands on a sidelobe the FORMULA can be right while the HEIGHT
  is the neighbour's: `C18H30O6` is clean on `[M+H]+` (343.211) but its urea adduct
  (403.244) rides 11.5 mDa from a 520k-cps `C20H34O8` at a locked 0.71 %, so that
  trace tracks the wrong compound. The assignment is **never** altered — no
  retraction, no tier change — only quantification is flagged.
- **Why it lives at merge level, not in per-file cleanup**: static features cannot
  detect it. Over a labelled set of **25 498 raw tracks / 30 runs**, contaminated
  channels look _identical_ to real ions near a bright peak — satellite fraction
  0.69 % vs 0.23 % (the artifact is the BIGGER one), |Δm/z| 11.5 vs 10.1 mDa. Only
  the time series separates them (ratio-to-parent cv 0.033–0.051 vs 0.21–1.09, an
  empty gap between). `SIDELOBE_CV = 0.08` sits in that gap. Scored on the labelled
  set: **6/6 caught, 0 false positives of 72.**
- The rule evaluates **every** raw track carrying ≥`SIDELOBE_MIN_FRAC` of an ion's
  samples, not just the most-sampled one — `C18H30O6`'s dominant track (n=475,
  cv 0.106) hides the locked one (n=288, cv 0.033), and the exported trace mixes
  both. (Found by scoring an earlier draft against the test set.)
- **Uncorroborated sidelobe assignments are now demoted** (`demote_uncorroborated`,
  default on): when a flagged channel's neutral has no OTHER ion channel, nothing
  but the sidelobe supports the compound, so Assigned -> Candidate (demoted, never
  deleted — the ledger's no-drop rule). Campaign-wide: 81 channels checked, 7
  flagged, 1 demoted; `C18H30O6` keeps Assigned in all 4 runs because `[M+H]+` at
  343.211 corroborates it, while `C8H19NO9`/`C31H30` — single-channel — do not.
- **`cleanup.flag_ringing_artifacts` documented as unexplained-peaks-ONLY, on
  purpose.** It runs post-pass-6, so a pass that already claimed a sidelobe hides
  it — which looks like an obvious bug to fix by also displacing committed M0s.
  Measured: that rule selects 74 commits across 30 runs and scores **0/53 correct,
  51 false positives** against the TS oracle (C12H16O6, C13H29NO9, C4H8N2P2S, the
  siloxanes — all real). The docstring now carries that number so the "fix" is not
  reintroduced; the TS-gated merge-level pass is the correct home for it.

### Fixed (time-series parquet: one ion, one peak, one trace)

- **`_batch_ts.parquet` no longer stamps one formula onto two peaks**
  (`batch/timeseries.py` `annotate_peaks`). The stamp is a mass match, not a
  peak-identity join, and it was many-to-one: neighbouring raw peaks each grabbed
  their nearest ledger ion, so a shoulder/split peak inside the same window got the
  SAME `neutral_formula`+`adduct` as the real peak and a downstream
  `groupby(formula, adduct)` saw two traces for one ion in one sample. Measured on
  a 2.4 M-row uronium batch: **2385 duplicated (sample, ion) pairs across 61 ions →
  0**. The assignment itself never did this (9784 per-file M0 keys, zero owned by
  > 1 peak — the shoulder is left `unexplained`), so this only ever affected the
  > parquet. Two rules, both default-on and individually switchable
  > (`one_to_one=` / `consensus=`):
  - **one-to-one** — per `(sample, ion)` keep the single best peak; ties break to
    the brighter, then lowest row index (deterministic → byte-reproducible).
  - **consensus** (`_consensus_offsets`) — "best" is nearest the ion's consensus
    m/z, not the bare ledger mass, which otherwise makes the winner flip between two
    raw tracks sample by sample, splicing two different peaks into one trace (232
    and 378 flips for two ions). Candidates are split into tracks (gap > `halfwin`),
    then one is chosen by **brightest member** (not summed height) with the **ledger
    mass anchored** (`ANCHOR_MARGIN` 2×). Both rules come from real failures found
    in adversarial review: scoring by summed height handed `C12H19NO6 [M+H]+` to an
    **FT ringing sidelobe** — ubiquitous-but-dim (1576 cps × 559 samples) and
    already flagged `role=artifact` by the assignment's own cleanup — over the real
    peak (2390 cps × 70), i.e. worse than the bug being fixed. The anchor encodes
    that offset 0 is the assignment's own answer, not an estimate; a decisively
    brighter track (`C19H34O6Si`, 1205 vs 473 cps) still wins, a marginal one
    (`C14H28O3Si`, 1.86×) does not.
- **New `dup_candidate` column** (`bool`) — `True` for a peak that fell inside an
  ion's window but lost. Its assignment columns stay `<NA>`; **no row is ever
  dropped** and heights are untouched, so the drops are auditable. Adds ~0 bytes
  after compression. Full parquet schema now documented in
  [`docs/OUTPUTS.md`](docs/OUTPUTS.md). (`tests/test_timeseries.py`, +25 checks.)
- **Known residual, now documented honestly** — a stamp is a mass match, not proof
  of identity: where a sample's real peak is absent, a neighbour inside tolerance
  (sometimes a ringing sidelobe) still collects it, because the assignment's
  `role=artifact` verdict exists only for the ~6 assigned samples and cannot be
  carried to the other ~989. Measured: 0.28 % of stamped rows sit >1 mDa from their
  ledger mass; 10 ions of 2127 span >0.5 mDa. (An earlier draft of this note
  claimed the residual was a benign "bistable peak-fit"; the per-file ledgers'
  own FT-sidelobe commentary disproves that, and the note is corrected.)

### Added (iodide reagent profile — I⁻ CIMS as a built-in)

- **`IODIDE` `ReagentProfile`** (`chem/profiles.py`, name `I`, aliases
  `iodide`/`iodine`/`i-`/`i-cims`): negative mode, analyte channels
  `[M+I]-` / `[M-H]-` / `[M+I2]-`, `detect_adduct` `[M+I]-`, normalise on the
  in-window reagent ion (`reagent_ion_re` `I\d*-$`). Chemistry learned from the
  `2026-07-21 Iodide negative m/z 40-600 acquisition` batch
  (server-confirmed channels: HNO₃ as both `HINO3-` and `NO3-`, formic/acetic as
  `[M-H]-`, formic also as `CH2IO2-`). **Covalent iodine is OFF the neutral grid**
  (`ranges` has no I): ¹²⁷I is iodine's only isotope, so an in-neutral I can never
  be isotope-confirmed — same policy as F/P. The Br-specific isotope machinery
  (doublet clear-both, halocarbon relabel, `_prefer_adduct_reading`) is `Br`-gated
  and stays inert under `reagent_element='I'`.
- **`[M+I2]-` / `[M+I3]-` adduct shifts** (`chem/chemistry.py`) — the poly-iodide
  cluster channels; their server mechanisms (`+I2-`/`+I3-`) were already in
  `ADDUCT_TO_MECH`.
- **`_IODINE_BACKGROUND` pure-iodine-oxide source clusters** (`chem/reagents.py`,
  `build_library("I")` only): `I2O-` (269.80, ~2M cps) and `I3O-` (396.71) —
  bright, time-STABLE source ions, on top of the generic In⁻ ladder (I⁻/I₂⁻·/I₃⁻
  = #4/#1/#2 by height), IOₓ⁻ oxides and I·H₂O. Reagent-acid clusters (`HINO3-`,
  `IH2O2-`, `CH2IO2-`) are **deliberately NOT** in the library — they are the
  `[M+I]-` analyte reading of HNO₃/H₂O₂/HCOOH (the Br organic-acid ruling,
  applied to iodide).
- **Pass-0 `reactive_iodine` known-species family** (`assignment/passes/
directors.py`): HOI, HIO₂, HIO₃, INO₂, INO₃, ICN (`CNI`), INCO (`CINO`), ICl,
  IBr — the canonical iodide-CIMS analytes, detected as `[M+I]-`. Covalent iodine
  is monoisotopic + off-grid, so they must be supplied as known formulas (the
  PFCA precedent: at defect −0.19..−0.27 no grid-reachable organic exists, so
  exact-mass commit is safe). Reagent-vs-analyte for poly-iodide ions decided by
  TIME behavior: HOI₂⁻ swings 55× (photochemical HOI), I₂NO₂⁻ 2.3× → analytes;
  the bare ladder and I₂O⁻/I₃O⁻ are stable → source background. Validated live
  on a representative sample: 8 species committed at <0.6 ppm (ICl with its
  0.31-ratio ³⁷Cl twin, IBr with its 0.96-ratio ⁸¹Br twin, ICN cross-channel via
  `[M+I]-` + `[M]-.`); unexplained signal 6.2% → 3.8%.
- **`label_bromide_clusters` is now Br-gated** (`assignment/cleanup.py`
  `run_cleanup`): the defect+1.998-twin heuristic reads ANY heavy-halogen cluster
  region as "bromide" — under iodide it grabbed the I₂NO₂⁻/IBr·I⁻ neighborhood
  with a false bromide note — so it only runs when `cfg.reagent_element == "Br"`.
- **The shed hydrogen halide in the cluster library follows the reagent**
  (`chem/reagents.py` `build_library`): `HBr` was a fixed `_CLUSTER_NEUTRALS`
  entry, so `build_library("I")` emitted phantom `[In+HBr]-` ions (and a Cl
  library would have too). Now the hydride is `H<reagent>` (HBr / HCl / HI):
  the I library carries `[I+HI]-` (254.817) and zero Br-bearing formulas.
- **IOₓ⁻ oxide anions are NOT reagent labels under iodide** (adversarial-review
  catch): `build_library` skips the generic RO⁻/RO₂⁻/RO₃⁻ entries for
  `reagent == "I"` — IO⁻/IO₂⁻/IO₃⁻ are ion-identical to the `[M-H]-` ions of the
  iodine oxyacids, and **iodate IO₃⁻ is iodic acid's dominant channel** (the
  new-particle-formation tracer); labelling it reagent locked THE key
  iodide-CIMS analyte away from assignment (the HNO₃/NO₃⁻ ruling, applied to
  iodine oxides). Br/Cl oxide entries unchanged.
- **OIO added to `reactive_iodine`** (`IO2`, via `[OIO+I]-` = I₂O₂⁻ 285.799);
  the IO radical is deliberately NOT listed — `[IO+I]-` is composition-identical
  to the locked I₂O⁻ source cluster, a blind spot now documented in
  `docs/REAGENTS.md` beside the I₃⁻/ambient-I₂ one (check I₂O⁻/I₂⁻ ratio drift).
- **Coverage hardening from the review**: Cl-library tests (`[Cl+HCl]-` + ³⁷Cl
  twin, no-Br guard) + Br/Cl/I library-size snapshots; a consistency pin that
  every built-in profile adduct resolves in BOTH `ADDUCT_TO_MECH` and
  `ADDUCT_SHIFTS` (a dropped mapping silently disables a channel); the
  `run_cleanup` Br-gate pinned in all three directions (Br runs / I skipped /
  None skipped); the full six-alias iodide loop.
  (`tests/test_profiles.py`, `tests/test_reagents.py`, `tests/test_chemistry.py`,
  `tests/test_passes.py`, `tests/test_cleanup.py`; docs in `docs/REAGENTS.md`.)

### Added (time-series parquet now carries the assignment)

- **`per_file/_batch_ts.parquet` peaks are stamped with their assigned formula/channel**
  (`batch/timeseries.py::annotate_peaks`, wired into `batch/assign_batch.py`). Each ts
  peak gains four columns — `neutral_formula`, `adduct` (the ionisation channel),
  `tier`, and `ion_mz` (the matched assigned m/z) — by nearest-m/z match to the final
  merged ledger within `max(mz·tol_ppm, 1.5 mDa)`; unmatched peaks keep `<NA>`.
  Vectorised (searchsorted), ~2 s on a 4 M-row batch. The file is now written in **both**
  the serial and parallel paths (previously only an internal, un-annotated worker-transfer
  artifact in parallel mode), so downstream time-series analysis has the formula per peak,
  not just `m/z`. (`tests/test_timeseries.py`.)

### Fixed (clustering — weak diel analytes buried in the flat panel)

- **Diel-structure gate lowered `DIURNAL_ETA2` 0.50 → 0.30** (`batch/cluster.py`).
  The 0.50 bar was set where diel analytes score 0.57–0.72, but weak ones (a real
  low-amplitude daily wave, diurnal η² 0.30–0.50 — e.g. TPPO C18H15OP, C17–20 O2
  oxidation products, organonitrates) fell under it and were bunched into the
  "flat background" panel, where their shared wave leaked into the flat median.
  0.30 still clears the ~0.1 structureless background, so both regimes hold; the
  weak analytes now surface into the structured-background / family pages. The
  residual gentle wave in the flat median is pervasive common-mode (boundary-layer
  breathing shared by all ambient channels), which a per-channel gate cannot
  remove, so the panel is retitled **"Low-amplitude / common-mode background"** and
  its subtitle notes the median wave is the shared boundary layer, not a hidden
  analyte.

### Fixed (phantom heteroatom assignments — Si / P)

- **Silicon isotope gate** (`passes/config.py` `het_iso_penalty_Si`,
  `passes/core.py` `_DIAG`). Si now sits in the isotope-evidence gate alongside
  S/Cl/Br: an unconfirmed Si formula (no matched ²⁹Si/³⁰Si satellite) pays a
  het-iso penalty and loses arbitration to a CHO/CHON rival instead of winning on
  accurate mass alone; a real siloxane whose ²⁹Si/³⁰Si envelope is confirmed pays
  nothing. The gate diagnostic accepts either the M+1 (²⁹Si) or M+2 (³⁰Si) line.
- **Tier demotion of uncorroborated Si and mono-isotopic P/I** (`tiers.py`). A Si
  formula with no confirmed satellite and no cross-channel/series support, or a
  mono-isotopic P/I (no possible isotope twin) seen only on a single clustering
  channel, is demoted Assigned → Candidate; `known:` / cross-channel / isotope-
  confirmed species are spared. Kills phantom PDMS/silanol fits and orphan-adduct
  organophosphates (e.g. a "C13H29O4P" that was really a neighbouring CHON's ¹⁵N
  satellite).

### Fixed (exhaustive isotopologue claiming — no leaked satellites)

- **Faint diagnostic satellites are now claimed** (`chem/isotopes.py`
  `diag_min_rel` + `D_15N`/`D_30SI`; `passes/postprocess.py`; `assignment/cleanup.py`
  `reclaim_satellites`). The M+1/M+2 lines of ¹⁵N (0.36 %/N), ¹⁸O, and a single
  ³⁴S/²⁹Si/³⁰Si sit below the envelope plausibility floor and were left unclaimed,
  so they floated free as base peaks that mass-coincidence phantoms grabbed. They
  are now predicted below the floor and swept by `reclaim_satellites` (extended from
  ¹³C/⁸¹Br/³⁷Cl to the full diagnostic set with atom-count-aware ratio gates).
- **Strong-scoring phantoms displaced onto their true parent**
  (`complete_isotope_envelopes`). A peak on a faint parent-satellite line whose
  intensity matches the predicted satellite is moved into the iso-child role even
  when its own accurate-mass score is high — mass identity + intensity consistency
  outweigh the score. Excess-intensity and High-confidence victims are spared.
  (`tests/test_phantom_guards.py`, 27 checks.)

### Added (batch performance — sample-level parallelism)

- **`peaky batch --jobs/-j N`** (`batch/assign_batch.py`, `cli.py`, `pipeline.py`;
  env `PEAKY_JOBS`). Assigns the selected samples across a spawn process pool —
  ~3.5× faster on multicore. Byte-identical to a serial run: results are reduced in
  `sample_ids` order into the deterministic merge, and each worker gets its own
  pickled `PassConfig`. Per-worker `match_compounds` concurrency is scaled down
  (`PEAKY_MATCH_WORKERS`, `io_mascope.py`) so total server load stays bounded.
  `--jobs 1` keeps the exact serial path; default is physical-core count capped at
  the sample count. (`tests/test_batch_parallel.py`, 16 checks.)

### Changed (clustering — residual-space / common-mode redesign)

- **Cluster figures now cluster on de-glued residual correlation**
  (`batch/cluster.py`, `batch/clustering.py`, `reporting/pdf_report.py`). Raw
  pairwise correlation was dominated by a shared diel common-mode wave, collapsing
  distinct chemistries into one blob. The CHANGING set now runs assigned channels +
  gated unassigned bins through ONE unified space clustered on
  log → per-channel diel-anomaly → common-mode-removed residuals (`corr_space='raw'`
  restores the legacy behaviour); families are labeled by their assigned members
  ("co-varies with X"), anchor-free families flagged NOVEL. BACKGROUND is split into
  common-mode diel carriers / low-amplitude diel-structured / genuinely flat; short
  batches fall back to raw correlation. (`tests/test_cluster.py`,
  `tests/test_clustering.py`.)

### Added (off-grid discovery: certified-neutral + organothiophosphates)

- **MCP server** (`peaky/mcp_server.py`, `peaky mcp`; extra `pip install
'mascope-peaky[mcp]'`; see `docs/MCP.md`). Drives the pipeline from any MCP
  client (ChatGPT Developer Mode, Claude Desktop, Cursor) without a shell —
  tools: `health`, `list_workspaces/datasets/batches/samples`, `certify_neutrals`
  (offline), `assign_sample`/`run_batch` (background jobs → `job_status`).
  `io_mascope` stays a direct in-process HTTP client (peak tables never cross
  the MCP boundary); credentials stay server-side. Tool functions are plain
  Python (FastMCP imported lazily), so the offline suite covers them without the
  optional dependency.
- **Certified-neutral discovery** (pass 7 — `peaky/assignment/certified_neutral.py`,
  `run_pass_certified`, `scripts/certify_neutrals.py`; see `docs/CERTIFIED_NEUTRAL.md`).
  When ≥2 distinct ion channels in one spectrum converge on the same neutral core mass
  (different adducts, or reagent-cluster ladder rungs `[M+nUrea+H]+`, urea step 60.0324),
  those are N independent mass constraints on one unknown — a _certificate_ that licenses
  enumerating the expanded element box (P/S/Cl, past the per-peak caps) for that mass
  only, oracle-scored, isotope-gated (³⁴S/³⁷Cl/⁸¹Br; ¹³C never), committed onto every
  member peak under its own channel label. The pass-5 inverse: cross-channel evidence
  _licenses_ new formula space instead of _completing_ known formulas — so off-grid
  families (organophosphate pesticides, sulfonamide plasticizers) are discoverable
  generically, with no whitelist. Also interrogates weak M0 incumbents: a strong
  certificate (iso-confirmed or ≥3 channels) displaces a bogus single-channel fit (e.g.
  an unsupported `[M+Na]+`) via `clear_assignment`, audit-trailed. Reagent-free primary
  path; optional `ts_peaks` co-variation corroboration. Validated on the NBBS urea ladder
  (→ C₁₀H₁₅NO₂S) and cross-channel malathion (C₁₀H₁₉O₆PS₂); first real-ledger run
  blind-rediscovered benzothiazole (C₇H₅NS).
- **Organothiophosphate pesticide family** in positive pass-0 (`_known_species`): malathion
  - homologs + des-ethyl TP + ~14 common OP-thioate insecticides. P is off the grid and S
    above `max_S`, so these were structurally invisible; committed under a ≥2-channel **or**
    diagnostic-isotope gate (the fast-path/naming layer; certified-neutral is the generic path).

### Changed (corroboration + I/O robustness)

- **Generalized the pass-0 P-corroboration gate**: any confirmed diagnostic heavy-isotope
  envelope (³⁴S/³⁷Cl/⁸¹Br) substitutes for the 2nd ion channel — not a hard-coded
  `organothiophosphate`+³⁴S special case. ¹³C is explicitly excluded (every C formula has a
  ¹³C line, so it can't refute an off-grid P). A ³⁷Cl-confirmed single-channel chlorinated
  thiophosphate now commits; a ¹³C-only one still refuses.
- **WAF-retry the bulk batch loader** (`io_mascope.fetch_batch_peaks`): bounded exponential
  backoff on Cloudflare/origin transients (403/429/5xx/521/522, read timeouts); non-transient
  errors (legacy 404) re-raise immediately so the per-sample fallback still fires. Prevents a
  burst 521 from dropping whole-batch TS loads onto the per-sample loader (which hangs).

### Fixed (docs reconciliation)

- Pass-0 docs now list the organothiophosphate family + the isotope waiver; the "flat
  background" cluster panels are documented as amplitude-only (a coherent low-amplitude
  diurnal wave can be mislabeled flat); reagent-is-flat caveat added (reagent normalisation
  cannot remove the common-mode wave — it is real ambient signal); ~45 stale `passes.py:NNN`
  citations re-anchored by function name to `passes/{directors,core,postprocess,config}.py`.

### Added (¹⁵N-labelled nitrate CIMS)

- **Labelled-reagent covalent-product rescue** (`peaky/assignment/labeled.py`, pipeline
  stage `labeled_15n`). In a ¹⁵N-nitrate run a covalent ¹⁵N-organonitrate product sits
  *j·*0.997 Da off any grid formula, so it is left unexplained or absorbed by a
  partially-fluorinated fit. The pass re-enumerates the CHON grid at the shifted mass,
  substitutes ¹⁵N (`^N`), and commits only under a four-gate discipline (on-calibration
  mass, organonitrate plausibility `O≥3·n(¹⁵N)`, matched isotopologue, non-degenerate).
  No-op unless `profile.label_isotope` is set. `NO3_15N` now declares
  `label_isotope='^N'`, `label_max=2`.
- **¹⁵N-nitrate ¹⁴NO₃-cluster re-read** (`cleanup.relabel_nitrate_clusters`, post-tier
  stage `relabel_nitrate_clusters`). In a NOx-oxidation run the free chamber ¹⁴NO₃⁻
  clusters with oxygenated analytes to give `[X+¹⁴NO₃]⁻`, the exact isobar of the
  covalent organonitrate `[Y−H]⁻` (Y = X + HNO₃). ¹⁴NO₃ is kept **off** the scoring
  grid (an uncontrolled isobar competitor would flip genuine organonitrates arbitrarily);
  instead `[Y−H]⁻` is re-read as `[X+NO₃]⁻` only when the parent X is independently
  detected via its own `[X−H]⁻` and/or its ¹⁵N cluster `[X+¹⁵NO₃]⁻` (lenient bar). Tier
  preserved (exact isobar → same ion/mass/score). Gated on the labelled-nitrate profile.

### Fixed (¹⁵N over-reach + clustering)

- **Fluorine F/H-coherence cap** (`tiers.F_H_COHERENCE`). A partially-fluorinated M0
  (`F≥1 & F<2·H`, H-rich, sub-PFAS F) is the classic absorber of a mass shift the grid
  cannot express (¹⁵N-organonitrates in a ¹⁵N run); ¹⁹F is monoisotopic, so the fluorine
  count is a mass-only claim → demote Assigned→Candidate unless a ¹³C child pins the
  carbon count. PFCA/TFA (`H=1`) and true polyfluoro (`F≥2H`) untouched. One of three
  fluorine-exemption closures (with the plausibility carbon-cluster F-free-clause drop
  and the cleanup `(H+F)/C` carbon-rich floor).
- **¹⁵N-rescue calibration gate.** The covalent-product rescue now accepts a ¹⁵N reading
  only inside the run's own calibrated mass window (`|z| ≤ 2.6` on the corroborated ¹⁴N
  core) instead of a blind ±2 ppm window, so it never proposes a fill the tier engine
  would demote as an off-calibration coincidence.
- **Equilibration-settling family demote** (`cluster.py`). A family that is flat once the
  leading `SETTLE_FRAC` (0.18) window is dropped **and** starts high
  (`SETTLING_START_MIN` 0.8) is demoted as instrument/reagent settling; the `_starts_high`
  guard spares real early events. **Bright modest movers**: a bright channel
  (`≥1000 cps`) surfaces as a big changer at the lower `BIG_CHANGE_FOLD_BRIGHT` (2.0) fold.
- **Column-less empty match frame guard** (`cleanup` halogen recovery): a no-match
  `score_candidates` response can be a bare empty DataFrame with no columns; filtering
  `sample_peak_id` then raised `KeyError`. Now tolerated.

### Changed (BREAKING — output schema)

- **Report tier `Identified` renamed to `Assigned`.** The top assignment tier is now
  labelled **Assigned** everywhere it surfaces: the `tier` column values in
  `merged_ledger.csv` (and every per-file ledger), the workbook **Assigned** sheet
  (was "Identified"), the PDF report tier counts/labels, the GKA-widget legend, and
  the Summary "Tiers" rows. **This is a schema break**: downstream consumers that
  filter on `tier == "Identified"` must switch to `tier == "Assigned"`. The
  `Candidate` and below-assignability tiers are unchanged.
- The Summary M0 role label is now **"M0 (has formula)"** (was "assigned (M0)") to
  avoid colliding with the renamed tier; the role word "assigned" is otherwise
  unchanged.

### Added (plausibility hardening — Stage 3, demote-only)

- **One shared plausibility oracle** (`peaky/plausibility.py`): `is_oxygen_monster`
  (`O/C > 1.3`) and `is_carbon_cluster` (`DBE/C >= 1.0`, F-free, C≥2, half-integer-DBE
  radicals EXEMPT) now back BOTH the scrutiny `implausible()`/`scan()` flags and the
  new tier demotes, so a flagged formula and a demoted formula can never disagree. The
  carbon-cluster cutoff is `DBE/C >= 1.0` (NOT the earlier 0.75 proposal, which wrongly
  caught real aromatics — pyridine/coumarin/umbelliferone/furfural/phthalic anhydride
  all sit below 1.0 and are spared).
- **Per-file demotes** (`demote_oxygen_monsters`, `demote_carbon_clusters`, wired into
  `assign.run` after tiering): an oxygen-lattice monster (`O/C > 1.3` AND degeneracy
  mass-saturated — _not_ niso-gated, since a ¹³C confirms carbon count, not the O count)
  or a carbon cluster is demoted Assigned→Candidate + `below_assignability`. Never
  deletes a row.
- **New artifact `tables/plausibility_audit_<tag>.csv`** — one row per touched peak
  (`mz, neutral_formula, before_tier, after_tier_or_role, reason, evidence, degeneracy_note,
n_iso`); always written (header-only when nothing was touched) so the artifact set is
  stable.

### Fixed (off-calibration degenerate-winner displacement)

- **The winner-selection / cross-file merge could pick a mass-degenerate competitor
  that the pipeline's own tiering step then flags as off-calibration with no
  corroboration — displacing a better, corroborated assignment entirely.** Two
  layers were hardened so the calibration-sigma + isotope/cross-channel/series
  corroboration gate the tier engine computes is applied AT WINNER-SELECTION, not
  only at report time:
  - **Per-file re-arbitration** (`passes.rearbitrate_offcal_degenerate`, new
    `rearbitrate` stage after `siloxane`, before `degeneracy`/`tiers`). Pass 1
    commits the highest-`eff_score` candidate _before_ the mass calibration is
    fitted, so the off-cal arbitration penalty never sees it; with the local
    in-process scorer (`PEAKY_LOCAL_SCORING`, default in 0.5.0) a sub-ppm-coincident
    off-cal high-DBE/heteroatom "monster" can out-score the real on-trend molecule.
    The new stage re-uses `tiers._calibrate` (the isotopologue-backed CHO/CHON core)
    and displaces a winner that is off-calibration (>|2.6|σ), uncorroborated, AND in
    the aromatic-monster corner (`DBE/C ≥ 0.70`) with a stored alternative that is
    on-calibration, chemically plausible (`plausibility.implausible`), and strictly
    less unsaturated (lower DBE). The plausibility + lower-DBE guards are essential:
    a blunt "swap to the best on-cal alternative" reverses many correct calls.
    Corroborated, on-cal, locked, known-species, and series/anchor winners are never
    touched. No-op when uncalibrated.
  - **Cross-file consensus merge** (`assign_batch.align`). Per-file mass-calibration
    jitter can flip a degenerate pair in a single file (a competitor reads on-cal /
    Assigned there with a marginally higher local score) while the other files agree
    on the real formula. The winner is now the formula with the broadest
    **Assigned-tier cross-file support** — ranked by (Assigned-file count, file
    count, best tier, best score) — instead of the single highest per-file
    `ion_score`. A no-op when a cluster carries one formula.
  - Validated on a Ur⁺ batch: m/z 424.218 returns to **C18H30O10
    [M+NH4]+** (the α-pinene HOM oligomer, Assigned across 5 files, on the bundled
    Kang reflist) and m/z 464.143 returns to **C22H23N3O7 [M+Na]+** (on-cal CHON)
    instead of the off-cal C17H31N5O6 (5 N, no N source) and C36H17N (DBE-29
    azabenzo-PAH) the local scorer had selected — matching what the server-scored
    path already produced. 4 per-file swaps across the 11-file batch, every one an
    aromatic monster (DBE 21–35) → on-cal oxygenated molecule, zero false positives.

### Deferred

- **In-source fragment auto-detection.** A batch-level heuristic that relabelled an
  adduct-less protonated M0 as an `in-source fragment` of a heavier co-varying parent
  (full adduct-ratio + facile-loss + time-series triangulation), plus a companion
  series-coherence check that dissolved time-incoherent homolog ladders, was prototyped
  and **removed before release**: on real merged multi-sample data the triangulation
  over-fired (co-incidental facile-loss mass matches between unrelated co-varying
  analytes), so the `role=fragment` label, the report "Fragment ions" sheet, the grey
  Van Krevelen fragment marker, and the `ledger.mark_fragment` API were dropped. The
  retained O-monster + carbon-cluster demotes and the `plausibility_audit` CSV are
  unaffected. Fragment detection may return once a more discriminating gate is found.

## [0.5.0] — 2026-06-30 (reference peaklists + chemical-plausibility hardening)

Adds a context-gated literature/contaminant peaklist layer and closes a set of
chemical-plausibility gaps surfaced by manual review and a cross-pipeline
(Orbitool) comparison — the pipeline now assigns by mass **and** checks that the
isotope evidence + ionization chemistry actually support each Assigned formula.

### Added

- **Reference peaklists** (`peaky/reflists.py` + `peaky/data/peaklists/`): a curated,
  self-describing catalog (metadata + version + references + provenance) of known
  molecules per chemical system — seeded with α-pinene OH-oxidation HOM (Kang, FZJ
  E&U 557; 830 neutrals) and the Keller 2008 MS contaminant list (59 neutrals).
  Used three ways, all soft + provenance-tagged (never overrides an isotope-scored
  Assigned): (1) **selection prior** — a candidate on an active list wins a near-tie
  in arbitration; (2) **rescue-verify** — unexplained peaks matched by mass are scored
  with the server and committed if confirmed (or kept as a tentative low-quality
  Candidate when too dim to confirm); (3) **report** corroboration/rescue section
  - `tables/reflist_matches_*.csv`. Lists are context-gated by run metadata
    (contaminants always active).
- `docs/ASSIGNMENT_DETAIL.md` — exhaustive per-pass / per-gate pipeline reference.

### Changed (chemical-plausibility hardening)

- **Reagent-halocarbon relabel** — bromomethane reagent fragments mis-read as a bare
  element + reagent-cluster (e.g. CHBr₂⁻ as "C" via `[M+HBr+Br]-`) are reclassified
  on the invariant ion composition (CH₂Br₂→reagent, dibromoacetic acid→named).
- **Confirmed-isotope F-demote exemption** — a high-F formula is exempted only when a
  Cl/Br/S anchor's diagnostic isotope (³⁴S/³⁷Cl/⁸¹Br) is _confirmed_, not merely in
  the formula (a reagent-Br adduct's ⁸¹Br does not count).
- **Si-count intensity gate** (siloxane ladder **and** pass-0 silanediol) — a Si-rich
  commit requires its ²⁹Si M+1 to _match_ the Si count, not just be matched; stops a
  high-O HOM (e.g. C₁₀H₁₈O₁₁) being claimed as a siloxane on a too-weak envelope.
- **New tier demotes** (post-tiering, never deletes): carbon-cluster (F-free H/C<0.35),
  implausible-ionization (heteroatom-free hydrocarbon via an anion channel that needs
  an acidic/H-bond site), and speculative-residual (residual:\* commits resting on
  off-cal z, uncorroborated multi-N, 0-anchor series, or a sole minor channel).
- **Scrutiny page** — F-flag wording corrected (¹⁹F is monoisotopic — the F _count_ is
  unconfirmable; any ¹³C/⁸¹Br satellites confirm only carbon/the adduct), per-row
  evidence (score · ppm · isotopes · sane-alternative), and pagination.

### Fixed

- Report cover now states the **actual** sample-selection method (single-sample /
  brightest-coverage / representative) and a peak census (total / assigned /
  unexplained) from the ledger; reference-list section paginated (no clipping);
  single-sample reports include the Van Krevelen figure.

## [Unreleased] — 0.4.0 (public-release refactor)

A refactor pass preparing Peaky for the public `karsa-oy/peaky` repo: cleaner
install, content-stable reproducibility, organized outputs, a brightest-coverage
batch mode, and a full design-doc set.

### Added

- **`peaky setup`** — one-command workspace bootstrap: creates `.env` from the
  template, points outputs at the workspace's `output/` folder (`PEAKY_OUTPUT_DIR`),
  creates it, verifies the install (+ the Mascope connection if creds are set), and
  prints the layout + next steps. Re-runnable. Makes "clone → install → know what to
  do" a two-command path. Batch `--out-dir` now defaults to `$PEAKY_OUTPUT_DIR` (the
  workspace `output/`) else `~/peaky-output`.
- `docs/ARCHITECTURE.md` — the canonical design doc (ledger model, pass sequence,
  end-to-end data flow with diagram, reproducibility model, module map).
  Companion docs `docs/ASSIGNMENT.md` (what assignment produces, for a scientist)
  and `docs/OUTPUTS.md` (every artifact, where + what).
- `CHANGELOG.md` (this file).
- **Brightest-coverage batch selection** (`--select brightest`, the "bin-then-assign"
  mode). Bins all batch peaks by m/z and assigns each significant bin's _brightest_
  sample (greedy set-cover, `--coverage-target`/`--k-max`/`--height-floor`). Better
  analyte coverage than the time-grid+max-TIC default (which a reagent-CIMS run's
  reagent ion dominates); feeds the same assign → merge → report chain, so outputs
  are unchanged. A coverage play, not a speed play. (`sampling.select_brightest_coverage_samples`.)
- Legacy workspace-based Mascope server support (`io_mascope`): connects to older
  deployments where `/api/datasets` 404s, resolving workspaces/batches via the raw
  endpoints. Additive and gated — modern servers are unaffected.

### Changed

- **Import package renamed `mascope_assign` → `peaky`.** A `mascope_assign`
  back-compat shim aliases the old import path — including submodules — to the same
  `peaky` objects, so existing `import mascope_assign` code keeps working unchanged.
  Version bumped to 0.4.0.
- **PyPI distribution name is `mascope-peaky`** (`peaky` was already registered).
  The import package and the `peaky` CLI are unchanged — `pip install mascope-peaky`
  then `import peaky` / run `peaky` (dist ≠ import, like scikit-learn/sklearn).
- **Single canonical lockfile.** Removed the hand-maintained `requirements.txt`
  (which had drifted from the real pins); `uv.lock` is now the only pinned source.
  `pip install -e .` uses the pyproject ranges; `uv sync` uses the exact pins. CI
  gains a `locked` job that enforces `uv.lock` with `uv sync --frozen`.
- Moved `ROADMAP.md` → `docs/ROADMAP.md` (kept as development history); README now
  points at `docs/ARCHITECTURE.md` as the entry point for how Peaky works.
- Repository URL → `github.com/karsa-oy/peaky` (the public home).

### Fixed

- **Reproducibility: content is a pure function of inputs; only the report timestamp
  varies.** `pipeline.stamp_source_date_epoch()` pins `SOURCE_DATE_EPOCH` to a FIXED
  content epoch (`CONTENT_EPOCH`, 1980-01-01Z), so matplotlib PNG/PDF metadata and the
  openpyxl xlsx timestamps are constant — every figure's pixels, `merged_ledger.csv`,
  the per-file/cluster csv, and the xlsx tables are byte-identical for identical input
  data, **regardless of when the run happens**. Run time reaches output ONLY as visible
  PDF-cover text (the "generated" line + Report ID), the run-folder name, and
  `run_manifest.json`. The assignment xlsx's run-time "generated" cell was removed (it
  was the only run-time leak into a data file), and `write_excel` is now post-processed
  for byte-stability too. `test_determinism.py` asserts the contract: two runs at
  different times over the same inputs → identical figure/xlsx/csv bytes, with the PDF
  differing only by its visible cover timestamp.
- **`run_batch` now runs the FULL pipeline.** `peaky.run_batch` pointed at the
  assign-only `assign_batch.run` (no figures/report); it now maps to
  `pipeline.run_batch` (assign → cluster → Van Krevelen → report). `run_assign_batch`
  exposes the assign+merge half; `run_pipeline` aliases `run_batch`.
- `run_manifest.json` stores the input time-series path relative to the run dir (or
  absolute when referenced externally) instead of a bare basename, so it stays
  reproducible when the input TS is referenced rather than copied.
- Documented `cleanup.reclaim_envelope_tails` as a known no-op on real data (the leak it
  targets is absorbed upstream); kept but no longer implicitly trusted.

### Changed (outputs)

- **Run folders are organized into subdirectories.** A new `paths.RunPaths` is the single
  source of truth for the layout, shared by the writers and the report reader so their
  filename contract can't drift: `.png` → `figures/`, `.csv`/`.xlsx` → `tables/`, the PDF
  → `report/`. `merged_ledger.csv`, `run_manifest.json`, `batch_summary.json`, and
  `per_file/` stay at the run root (read by several modules + the cross-run registry).
- **The input time-series is no longer copied into every run.** A parquet passed by path
  is referenced in place; only a live-fetched series is persisted once, to `data/`. This
  removes a ~40 MB duplicate per run.
