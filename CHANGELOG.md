# Changelog

All notable changes to Peaky are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims to
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] — report refactor

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
  those are N independent mass constraints on one unknown — a *certificate* that licenses
  enumerating the expanded element box (P/S/Cl, past the per-peak caps) for that mass
  only, oracle-scored, isotope-gated (³⁴S/³⁷Cl/⁸¹Br; ¹³C never), committed onto every
  member peak under its own channel label. The pass-5 inverse: cross-channel evidence
  *licenses* new formula space instead of *completing* known formulas — so off-grid
  families (organophosphate pesticides, sulfonamide plasticizers) are discoverable
  generically, with no whitelist. Also interrogates weak M0 incumbents: a strong
  certificate (iso-confirmed or ≥3 channels) displaces a bogus single-channel fit (e.g.
  an unsupported `[M+Na]+`) via `clear_assignment`, audit-trailed. Reagent-free primary
  path; optional `ts_peaks` co-variation corroboration. Validated on the NBBS urea ladder
  (→ C₁₀H₁₅NO₂S) and cross-channel malathion (C₁₀H₁₉O₆PS₂); first real-ledger run
  blind-rediscovered benzothiazole (C₇H₅NS).
- **Organothiophosphate pesticide family** in positive pass-0 (`_known_species`): malathion
  + homologs + des-ethyl TP + ~14 common OP-thioate insecticides. P is off the grid and S
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
  mass-saturated — *not* niso-gated, since a ¹³C confirms carbon count, not the O count)
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
    commits the highest-`eff_score` candidate *before* the mass calibration is
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

## [Unreleased] — 0.5.0 (reference peaklists + chemical-plausibility hardening)

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
  + `tables/reflist_matches_*.csv`. Lists are context-gated by run metadata
  (contaminants always active).
- `docs/ASSIGNMENT_DETAIL.md` — exhaustive per-pass / per-gate pipeline reference.

### Changed (chemical-plausibility hardening)
- **Reagent-halocarbon relabel** — bromomethane reagent fragments mis-read as a bare
  element + reagent-cluster (e.g. CHBr₂⁻ as "C" via `[M+HBr+Br]-`) are reclassified
  on the invariant ion composition (CH₂Br₂→reagent, dibromoacetic acid→named).
- **Confirmed-isotope F-demote exemption** — a high-F formula is exempted only when a
  Cl/Br/S anchor's diagnostic isotope (³⁴S/³⁷Cl/⁸¹Br) is *confirmed*, not merely in
  the formula (a reagent-Br adduct's ⁸¹Br does not count).
- **Si-count intensity gate** (siloxane ladder **and** pass-0 silanediol) — a Si-rich
  commit requires its ²⁹Si M+1 to *match* the Si count, not just be matched; stops a
  high-O HOM (e.g. C₁₀H₁₈O₁₁) being claimed as a siloxane on a too-weak envelope.
- **New tier demotes** (post-tiering, never deletes): carbon-cluster (F-free H/C<0.35),
  implausible-ionization (heteroatom-free hydrocarbon via an anion channel that needs
  an acidic/H-bond site), and speculative-residual (residual:* commits resting on
  off-cal z, uncorroborated multi-N, 0-anchor series, or a sole minor channel).
- **Scrutiny page** — F-flag wording corrected (¹⁹F is monoisotopic — the F *count* is
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
  mode). Bins all batch peaks by m/z and assigns each significant bin's *brightest*
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