# Changelog

All notable changes to Peaky are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims to
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] — report refactor

### Fixed (batch-name resolution vs the current mascope-sdk)
- **`peaky batch` matched no batch on current SDKs and died on a legacy-endpoint
  422** (`io/io_mascope.py`). The SDK resolves its two batch filters with OPPOSITE
  conventions — `load_peaks(batches=)` escapes a plain string itself (literal
  substring; only a compiled pattern is a regex), `samples.list(batch=)` uses a
  plain string as a RAW regex and rejects compiled patterns — so the previous
  one-size `re.escape()`d string was escaped twice by `load_peaks`, silently
  matched nothing, and every batch run fell through to the legacy
  `/api/sample/batches` fallback, which modern servers reject (dataset_id
  required). `escape_batch()` (escaped string) now feeds `samples.list`;
  the new `literal_batch_pattern()` (compiled, IGNORECASE) feeds `load_peaks`;
  `fetch_pooled_peaks` compiles its user regex un-escaped. Both contracts are
  pinned by offline regression tests, and `mascope-sdk>=2026.7.7` is the floor
  they are written against — keep the SDK at the latest release (SKILL.md gotcha).

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
  channels look *identical* to real ions near a bright peak — satellite fraction
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
  >1 peak — the shoulder is left `unexplained`), so this only ever affected the
  parquet. Two rules, both default-on and individually switchable
  (`one_to_one=` / `consensus=`):
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