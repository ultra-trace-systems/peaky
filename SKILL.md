---
name: mascope-peak-assign
description: >-
  Multi-pass chemical-formula assignment for high-resolution mass-spec peaks
  stored in Mascope. Use when a user asks to assign formulas, annotate a
  spectrum, identify compounds, build a target list, or explain unassigned /
  contaminant / homolog peaks for a Mascope sample_id. SDK-native, runs locally
  via the Bash tool / `peaky` CLI; defers all mass + isotope scoring to Mascope's
  match_compounds; produces an 11-sheet tiered Excel (Assigned / Candidates /
  below-assignability) with commentary, close alternatives, per-isotopologue
  scores, a peak-ownership audit, and an interactive rotating-GKA widget. Also runs
  a representative-sample BATCH pipeline (5 time-spaced + max-TIC samples assigned
  and merged), time-series correlation clustering, a full Van Krevelen, and a
  standard iterable PDF assignment report. Triggers: "assign formulas", "peak
  assignment", "what's in sample X", "annotate spectrum", "unassigned peaks",
  "Kendrick / GKA", "homologous series", "CIMS", "HOM", "PFAS / contaminants",
  "assign a batch", "Van Krevelen", "cluster figures", "PDF report".
---

# Mascope multi-pass peak assignment

> This skill is **Peaky** (`import peaky`, CLI `peaky`); the skill id stays
> `mascope-peak-assign` for back-compat. See `README.md` / `docs/ARCHITECTURE.md`.

A reproducible, test-driven, SDK-native pipeline. **All heavy work runs on the
host Python** (which has `mascope-sdk`) via the **Bash tool / `peaky` CLI** (a
shell MCP only if your Claude runs sandboxed); the `mascope__*` MCP is never used
to transport peak tables through context. (Outside Claude, just run the `peaky`
CLI in a terminal.)

Canonical home and iteration repo: `~/.claude/skills/mascope-peak-assign/`.

## Operating principle

Division of labor is the core design decision:

- **Mascope is the scoring oracle.** Its isotope-scored maths runs **in-process by
  default** via `mascope_tools` (`local_scoring.py`; IsoSpec + `score_pattern`),
  with the network `match_compounds` as an opt-in fallback (`PEAKY_LOCAL_SCORING=0`).
  Either way we get, per candidate ion, a per-isotopologue verdict (each
  isotopologue's `match_score` + the attributed `sample_peak_id`). We never invent a
  match score; we hand it candidate neutral formulas and read its verdict.
- **We own candidate generation, chemistry plausibility, series logic, and
  arbitration** — i.e. _which_ formulas are worth asking about and _which_
  answer wins each peak.

State lives in one mutable **ledger DataFrame** (one row per physical peak).
Passes are functions ledger→ledger; they fill and annotate, never drop rows. The
ledger's commit API enforces structural invariants so no pass can corrupt it.

## Pre-flight

1. Run on the host Python via the Bash tool / `peaky` CLI (host has `mascope-sdk`),
   not a sandboxed shell. **First-time setup from a fresh clone:** `pip install -e .`
   then **`peaky setup`** — that one command creates the workspace `.env` + `output/`,
   verifies the install (and the connection if creds are set), and prints the layout
   + next steps. Re-run `peaky setup` any time to re-check.
2. Creds: edit the repo-root `.env` (or `~/.mascope/.env`) with `MASCOPE_URL` +
   `MASCOPE_ACCESS_TOKEN` (auto-loaded; or `--env` / `$MASCOPE_ENV`). Batch outputs
   default to the workspace `output/` (`$PEAKY_OUTPUT_DIR`, set by `peaky setup`)
   else `~/peaky-output`; override per run with `peaky batch --out-dir ...`.
3. Pick `--reagent` (forces analyte channels) and/or `--context` for the sample.

## Running

Discover data, then assign — one sample, or a whole batch (the `peaky` console
command; `mascope-assign` is a back-compat alias, `python3 -m peaky …` equivalent):

```bash
peaky list datasets
peaky list samples --batch "<batch>" --dataset "<workspace>"

# one sample
peaky assign --sample-id <ID> --reagent <Br|Ur|NO3|NO3_15N|auto> \
    --height-cutoff 100 --output-dir ~/peaky-output/<name>

# a whole batch (assign subset -> merge -> cluster -> Van Krevelen -> PDF report)
peaky batch --batch "<batch>" --dataset "<workspace>" --reagent <Br|Ur|...> \
    [--select representative|brightest] --out-dir ~/peaky-output

# MANY same-chemistry batches -> ONE unified ledger + whole-pool + per-group reports
peaky pool --batches "<regex over batch names>" --dataset "<workspace>" \
    --reagent <Br|Ur|NO3_15N|...> [--k-max 6] [--group-by sample_batch_name]

# regenerate figures + PDF of an existing run, offline (no assignment, no network)
peaky report --run-dir <run-folder> --reagent <Br|Ur|...> --ts <ts.parquet>
```

`peaky pool` pools every batch matching `--batches` (a REGEX over batch names, e.g.
`'HR-CIMS 100-500.*zone'` = the per-zone batches of one mode x range) into ONE
unified ledger, so an analyte present in ANY group is discovered once, then each
group's own time series is read against that shared peak list. Selection is a
per-GROUP brightest-coverage UNION (`sampling.select_pooled_union`): brightest within
each group, then unioned — a single naive pooled pass lets the loudest group hog the
winners and starves quieter ones (measured: one group fell to 56% bright-bin
coverage under naive pooling vs balanced ≥82% with the union). Emits the whole-pool
run + one report per `--group-by` value (default `sample_batch_name`, i.e. per batch),
each sharing the unified ledger + re-clustering on its own TS slice. Validated on a
real multi-batch field campaign: the unified ledger captured 99.86% of a dedicated
per-group run's compounds.

`--select brightest` bins ALL batch peaks and assigns each significant m/z bin's
brightest sample (better analyte coverage than the default 5-time-spaced+max-TIC
rule; `--coverage-target`/`--k-max`/`--height-floor` tune it). Single-sample
`assign` writes `<ID>_<UTC>_{ledger.csv, assignments.xlsx, summary.md,
manifest.json, gka.html}` + per-pass checkpoints (~5 min on a ~1000-peak Br-CIMS
sample). Batch writes one versioned run folder — see **Outputs** below and
`docs/OUTPUTS.md`.

### MCP server (drive peaky from ChatGPT / Claude Desktop / Cursor)

`peaky mcp` (extra: `pip install 'mascope-peaky[mcp]'`) serves the pipeline as MCP
tools — `list_*`, `certify_neutrals` (offline), `assign_sample`/`run_batch`
(background jobs → `job_status`). It runs HOST-SIDE: `io_mascope` stays a direct
in-process HTTP client so peak tables never cross the MCP boundary, and creds stay
in `.env`. ChatGPT Developer Mode connectors need a URL (SSE/streamable-HTTP) via a
tunnel, not localhost. See `docs/MCP.md`.

### Contexts

`ambient-air` (= atmospheric), `chamber`, `indoor-air`, `object-headspace`,
`combustion`, `water`, `food`, **`uronium`** (= positive urea-CIMS), `none`.
Context sets plausibility bounds + which Pass-3 contaminant families are eligible,
and carries `polarity` + grid-box width. Reagent adducts are NOT set by context —
they are **detected from the sample** (`ionization_mechanism` column).

### Polarity (negative Br-CIMS vs positive urea-CIMS)

The pipeline was built negative-mode Br-specialized; positive-mode support
(2026-06-16) is reagent/polarity-driven, detected from the adducts:

- **`uronium` context** (positive urea-CIMS): N-heavy VK priors, wider C46/O32
  grid, channels `[M+H]+` / `[M+(CH4N2O)H]+` (urea adduct, +61.0396) + opportunistic
  `[M+Na]+`/`[M+NH4]+`; urea `[urea_n+H]+` reagent-cluster library.
- **Br-specific passes are guarded in positive mode**: `detect_composites` is
  gated on a halogen adduct (its M+1 test misfires without one); the carbon-clamp
  skips Si (²⁹Si dominates the M+1, not ¹³C); di-bromide / iso-pair / `reagent_element`
  logic goes inert when no halogen is in the adduct.
- **`NO3_15N` context (¹⁵N-labelled nitrate⁻):** server mechanism `+^NO3-`; adduct
  `[M+^NO3]-` adds ¹⁵NO₃ (+62.985, not the ¹⁴N +61.988). The server models the
  reagent ¹⁵N as natural-abundance, so it tags the real 100%-¹⁵N peak as a non-base
  `[15N]` isotopologue and a phantom ¹⁴N line as M0 — `io_mascope.flatten_match_tree`
  **re-anchors** `is_base` onto the ¹⁵N line (else the whole `[M+^NO3]-` channel is
  dropped; same ¹⁵N-modeling also depresses the aggregate score of ¹⁵N poly-halogens).

### Halogen / heteroatom policy (two-sided)

**Isotope-confirmable (Cl/Br/S) → open + tier on the envelope; monoisotopic (F/P) →
off the grid except specific known families.** Negative pass-0 known-species now
include: **PFCAs** `CnHF(2n-1)O2` (TFA…PFOA, exact-mass committed — F is off the
grid so the clean low-F acids would otherwise be missed); **chlorinated paraffins**
`CnH(2n+2-x)Clx` (SCCP/MCCP/LCCP, committed+locked **only** with a confirmed ³⁷Cl
envelope ≥2 satellites → Assigned, bypassing the ¹⁵N-depressed compound_score);
positive pass-0 adds **organophosphates** (TEP/TBP/TPPO…, cross-channel-gated since
P is monoisotopic) **and organothiophosphate/-dithioate insecticides** (~19 OP-thioates:
malathion `C10H19O6PS2` ±CH₂ homologs + des-ethyl product, chlorpyrifos, diazinon,
parathion, phorate, dimethoate, phosmet…; P off the grid, S₂/S₃ above `max_S`). The
P-bearing corroboration gate: a known P species commits with **≥2 ion channels OR** a
confirmed **diagnostic** heavy-isotope envelope (³⁴S / ³⁷Cl / ⁸¹Br, **not ¹³C** — ¹³C
can't refute an off-grid P) standing in for the 2nd channel.
`cleanup.demote_unconfirmed_fluorine` (run **after**
`apply_tiers`) demotes F≥4 M0 that are not a PFCA and lack a Cl/Br/S anchor —
¹⁹F mass-coincidence "monsters" → Candidate + `below_assignability`.

- **Mass offset ("be aware of the −X ppm"):** the reported `ppm_error` stays RAW,
  but every quality gate is offset-aware. `io_mascope.estimate_offset` seeds a
  rough offset (`cfg.prior_offset`) for the pre-calibration pass-0 gate; pass-1
  self-`calibrate` (score-based backbone) refines it; `confidence_label`,
  `tiers._calibrate`, arbitration (`CAL_ARB_WEIGHT` off-trend penalty), and
  `relabel_confidence` all judge ppm vs the calibrated center, not 0. Without
  these a large uniform offset collapses everything to Candidate and lets
  off-trend mass-coincidences win then z-reject (peak left unexplained).

### Key flags

`--ppm` (m/z trust, default 1.0) · `--search-ppm` (enumeration tol, 3.0) ·
`--height-cutoff` (cps, 100) · `--no-pass2/3/4` · `--no-cache`.

## Representative-sample batch pipeline (assign a whole batch, not one file)

A single averaged file misses analytes present only part of a run. The batch
pipeline assigns a **representative subset and merges by m/z**:

- **`sampling.select_representative_samples(peaks)`** — THE RULE: 5 samples evenly
  spaced in TIME (nearest distinct sample to each of 5 equally-spaced target times;
  endpoints always in) **+ the max-TIC sample**. Selecting in time (not row index)
  means a lone late file in an irregular run still gets a pick.
- **`assign_batch.run(batch=NAME | peaks=, reagent='auto', out_dir=, ts_peaks=, amine_r_min=0.6)`**
  — resolves the `profiles.ReagentProfile`, runs `assign.run` per selected file
  (keeps each `per_file/<sid>_ledger.csv`), then an **offset-aware merge** (`align`)
  - a file-to-file **jitter** table. Pass `ts_peaks` (the full-batch per-peak time
    series) to enable the positive-mode NH4→amine gate below. Writes `merged_ledger.csv`
    + `batch_summary.json` at the **run root**, per-file ledgers in `per_file/`, and
    `selected_samples.csv` / `jitter.csv` in `tables/`. The run folder is organized by
    `paths.RunPaths`: `.png`→`figures/`, `.csv`/`.xlsx`→`tables/`, the PDF→`report/`,
    a live-fetched TS→`data/` (a TS passed by path is referenced, not copied).
- IDs must be fetched FRESH from the live server (`io_mascope.fetch_batch_samples`,
  regex-escape the batch name) — cached `sample_item_id`s 404 when a server copy is
  renamed.

### Reagent-aware chemistry

`assign.run(..., adducts=)` forces a known reagent's channels (a sparse-match positive
file otherwise falls back to `[M-H]-` = wrong polarity). For positive urea-CIMS,
**`cleanup.prefer_amine_over_ammonium(ledger, ts_peaks=, r_min=0.6, protected=)`** resolves
the exact-mass degeneracy `[M+NH4]+` of X ≡ `[M+H]+` of the amine X+NH3 by TIME: a real
ammonium adduct must TRACK X's own `[M+H]+`/urea parent (2 h-binned log-corr). THREE-WAY
per neutral — **keep** the adduct (Assigned) when it tracks a shaped parent (r≥`r_min`);
**re-read** to the amine when it fails to track a shaped parent (r≤0.2) OR the parent
channels are absent from the TS (their absence refutes the ammonium reading); **cap at
Candidate** (keep the adduct but demote) when the pairing is weak (0.2<r<`r_min`) or the
present parent is FLAT (no shape to correlate — the NBBS blind spot). Overrides that keep
the adduct at tier: Si-bearing X (siloxane adducts are real), a `protected` neutral
(reflist-rescue / known-species / certified provenance, established independent of the NH4
channel — this holds NBBS, whose weak isobar-contaminated NH4 trace fails the tracking
test), and a valence-impossible amine (forced). Without a TS it degrades to the binary
presence test. (Run via `assign_batch` at the merged level, where corroboration is complete;
`assign_batch` gathers the `protected` set from the per-file `method` provenance.)

### Time-series clustering + figures + report

- **`cluster.py`** — TIC/reagent-normalised log-correlation, COMPLETE linkage at r>0.6
  (signed distance keeps anti-phase apart) → `render_a4` A4-portrait paginated panel
  figures (all clusters + a "remaining peaks" overview, panel legend = `formula
(ion-channels / isotope-peaks / match-score)`). **Flatness gate:** `split_varying(traces,
cols, cv_min=FLAT_CV)` pulls flat traces (cv<0.30, no reliable shape) OUT of clustering and
  `render_flat_panel` bunches them into ONE overview (so flat noise doesn't shatter into spurious
  n3-4 clusters). `FLAT_CV` knob — now serves only the UNASSIGNED path. **⚠ "flat" means
  LOW-AMPLITUDE, not necessarily structureless.** Both flat gates are amplitude-family —
  `split_varying` (cv gate) AND `split_flat_clusters` (smoothed max/median gate) — neither tests
  for time-of-day *structure*, so a coherent low-amplitude diurnal wave can be mislabeled flat.
  Measured on one ambient positive urea-CIMS batch (2026-07-07): of ~1010 channels binned "flat
  background", the MEDIAN diurnal η² (log-variance explained by time-of-day) was ~0.52 and ~70% had
  η²≥0.3 — most carry a shared low-amplitude ~15:00 afternoon wave; after removing that common-mode
  wave, ~42% still retain independent structure (real weak analytes). Numbers are one batch, not universal. **Per-ion:** assigned analytes
  cluster PER ION CHANNEL (formula+adduct), not the per-neutral sum (`analyte_viz.ion_traces`), because
  channels often diverge in time (`analyte_viz.channel_agreement` QC: Ur 44% / Br 22% of multi-channel
  neutrals' brightest pair disagree). Legend = `formula+adduct (score)`. `cluster.write_cluster_workbook`
  emits a per-cluster XLSX (one tab per cluster: formula / channel / m/z / match_score / tier).
  **Shape-cluster (assigned):** cluster ALL bright organic channels on RAW log-correlation (NOT
  reagent-norm — that makes the flat background spuriously 'rise' together), then `cluster.merge_similar`
  (COMPLETE-linkage centroid merge, `MERGE_R`) folds near-identical-shape families, then
  `cluster.split_flat_clusters` (via `cluster_flatness` = smoothed max/median of the member-MEAN trace,
  `FLAT_CLUSTER_RANGE`) DEMOTES clusters that co-vary but whose family mean is flat into the flat panel.
  The non-clustering remainder + flat clusters + Si = the flat panel; only dynamic families are shown.
  **Big standalone changers:** `cluster.big_changers` (per-channel smoothed max/median ≥ `BIG_CHANGE_FOLD`,
  ~≥5-10× raw, no family needed) + `render_changers` (A4-PORTRAIT small-multiples, paginated, decade-
  snapped log y) pull dramatically-changing single channels into their own report section (the `changers`
  SECTION), out of the flat panel.
- **`analyte_viz.render_van_krevelen_full`** — every assigned peak by CHO/CHON/CHOS
  backbone (Si/F/halogen folded into the backbone, not split out).
- **`pdf_report.build(out_dir, tag=, label=, ts_path=, batch_name=, run_id=, generated=)`** — the
  STANDARD iterable PDF report (uniform A4 portrait). Structure = `SECTIONS = [cover, findings, coverage,
composition, scrutiny, gka, families, changers, clusters, methods]`, each a `section(ctx, pdf)` fn over
  a context loaded once by `load_context`. To change the report, edit/reorder a section — nothing else
  couples. Section highlights:
  - `findings` (page 2): event-TIC of the FULL batch (total signal vs wall-clock, rep samples ticked) +
    data-driven takeaways — event rise×, SIGNAL-weighted composition, top species, oligomer/HOM line.
  - `composition`: distinct-neutral COUNT by backbone AND signal-weighted, + the ammonium/amine
    degeneracy two-way (`composition.py`: [M+NH4]+(CHO) is mass-identical to [M+H]+(amine X+NH3) so many
    CHON are double-counted — disclosed + collapsed count). Positive-only messaging gated on `ctx['positive']`.
  - coverage `Signal & peaks by role`: splits explained signal into analyte(M0+iso)/reagent/unexplained
    (a reagent-dominated Br- spectrum isn't "fully characterised"); the cover echoes the split.
  - `scrutiny` (`plausibility.py`): flags Candidate-tier mass-coincidence formulas (high heteroatom /
    very low H/C / wrong-mode halogen); Assigned never flagged; renders only if something is flagged.
  - The `gka` section renders `gka_figure.render_gka` on demand from the merged ledger.
    The cover stamps `run_id` (Report ID) + a date+TIME `generated` line; the PDF FILENAME embeds the
    Report ID (`report_<run_id>.pdf`). Cover formula-disagreements read from the merged ledger's own
    `formula_agree` (authoritative). **Reproducible content:** figures/PDF-figures/CSV/xlsx are a
    deterministic fn of the INPUT DATA ONLY — `stamp_source_date_epoch()` pins `SOURCE_DATE_EPOCH` to a
    FIXED content epoch, so they're byte-identical whenever you re-run the same data. The run time appears
    ONLY as the PDF cover's "generated" text + the Report ID + the run-folder name (so a re-analysis next
    week reproduces the same numbers/pixels; only the Report ID differs). The live `match_compounds` step
    is the only server-side non-determinism.
- **Run versioning** — `pipeline.make_run_dir(base, batch_name, when)` / `run_id` / `run_stamp` /
  `slugify`: every set of outputs goes in its own timestamped folder `<batch-slug>_<date>_<time>/`
  (folder name == Report ID). Pass ONE `datetime.now()` per run so folder, id and cover agree.
- **`gka_figure.render_gka(ledger, png, …)`** — STATIC GKA findings page (print
  counterpart of the rotating-GKA widget): a small-multiple grid of Kendrick mass-defect
  plots, one per repeat-unit FAMILY (alkyl CH2 / oxidation O,CO,CO2,H2O / alkoxylate
  C2H4O,C3H6O / siloxane / fluorinated CF2), each rotated to its own base so that family's
  homologous series flatten into horizontal ladders, over a grey cloud of every assigned
  neutral. A family is shown ONLY if it forms a SERIES (`present_families`): organic needs a
  ≥min_len ladder; CONTAMINANT families (siloxane→Si, fluorinated→F) need only a short
  ≥`contam_min_len`(2) ladder under their base, then highlight all element-bearing peaks. A
  scattered element set with NO series (e.g. assorted F mass-fits that never step by CF2) is NOT
  plotted. Pure fns: `detect_series` / `element_members` / `present_families` / `family_summary` / `kmd`.

The whole assign → merge → cluster → Van Krevelen → PDF chain is now in-package:
run **`peaky batch --batch ... --dataset ... --reagent ...`** (one versioned run
folder), and **`peaky report --run-dir ...`** to regenerate figures + the PDF
offline from an existing run. (The old `run_assign.py` / `run_clusters.py` /
`run_vankrevelen.py` / `run_report.py` scratch drivers are superseded by these.)
**A server may sit behind a Cloudflare WAF** — a burst of live runs can trip a 403
("Attention Required") that clears after 15-30 min of no traffic (polling extends it).

## The pipeline

| pass          | what it does                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| ------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Pre           | detect reagent adducts from the sample; prescan isotope fingerprint; **label reagent-ion clusters** (Brₙ, Brₙ·neutral, BrO/BrO₂/BrO₃ with **both ⁷⁹/⁸¹Br isotopologues**) so they are never assignment candidates — each is **recorded with its `ion_formula`** (known formula = assigned, whatever the class)                                                                                                                                                                    |
| 0             | **known species** (committed + locked, runs first): families the generic grid/gates would miss — atmospheric acids/radicals, nitroaromatics, **PFCAs**, **chlorinated paraffins** (only with a confirmed ³⁷Cl envelope; a low-server-score CP may be re-anchored to a real peak + ³⁷Cl envelope, never fabricated), silanediol contaminants (the ²⁹Si M+1 must **match** the Si count, not merely be present), and (positive mode) **organophosphates** AND **organothiophosphate/-dithioate insecticides** (malathion family, chlorpyrifos, diazinon…; P off the grid, S above `max_S`) — a P-bearing known species commits with **≥2 ion channels OR** a confirmed diagnostic ³⁴S/³⁷Cl/⁸¹Br isotope envelope (**not ¹³C**) substituting for the 2nd channel                                                                                                          |
| 1             | lock the high-confidence **CHO/CHON backbone**: grid-enumerate, score with `match_compounds`, arbitrate (complexity-penalised, isotopologue-gated, **reference-list near-tie prior**), commit M0 owners + attach Mascope's isotopologue children, lock High peaks                                                                                                                                                                                                                  |
| 2             | **iterative GKA series** expansion from locked anchors (CH₂/O/H₂O/CO/CO₂/C₂H₂O + siloxane/CF₂), chaining confirmed members as new anchors                                                                                                                                                                                                                                                                                                                                         |
| 3             | **automatic series detection** (the machine "rotating plot") opens contaminant families on decoy-controlled evidence; HBr-cluster ladder; organosulfate/nitrate/siloxane/amine + iso-gated bromo/chloro-organics                                                                                                                                                                                                                                                                  |
| 4             | **residual explainer**: isotope-pair resolution of ~1.998-Da doublets + deep 2-step series; DBE-only plausibility; ppm-disciplined acceptance                                                                                                                                                                                                                                                                                                                                     |
| 5             | **known-neutral completion**: cross-channel partners + series gaps of passes 1–4 (no new formula space)                                                                                                                                                                                                                                                                                                                                                                           |
| 6             | **anchored ladder gap-fill** (`ladders.py`): walk homolog/oxidation diagonals (+O/+CH₂/+CO₂/−H₂O, constant-DBE for carbon growth) out from Assigned anchors, satellite-guarded, Candidate tier                                                                                                                                                                                                                                                                                  |
| 7             | **certified-neutral discovery** (`certified_neutral.py` + `run_pass_certified`): groups of UNEXPLAINED peaks whose back-calculated neutral masses CONVERGE across >=2 distinct ion channels (adducts / reagent-cluster ladder rungs, urea step 60.0324) certify ONE neutral core -> only for certified cores is the expanded element box (P/S/Cl) enumerated, oracle-scored, and committed onto EVERY member peak (cross-channel tier evidence). Interrogates the unexplained residual PLUS weak M0s (Low/Suspect/near-tie, unlocked) -- a strong certificate (iso-confirmed or >=3 channels) DISPLACES a bogus single-channel incumbent (e.g. an unsupported [M+Na]+ fit), audit-trailed. The pass-5 INVERSE: cross-channel evidence LICENSES new formula space instead of completing known formulas. Optional ts_peaks co-variation corroboration (guarded; reagent-free primary path); ladder-shift ambiguity resolved by parsimony (fewest assumed cluster units). Ground truth: NBBS urea ladder (214/274/334 -> C10H15NO2S) + malathion cross-channel. See docs/CERTIFIED_NEUTRAL.md |
| iso-env       | **isotope-envelope completion** (`complete_isotope_envelopes`, before pass 4 + post-audit): claim every committed peak's full predicted M+2/M+4 envelope (Si/Br/Cl combos), attaching unexplained satellites and **displacing weak M0s that are really a parent's satellite** — kills the ~44% of "residual" peaks that are isotope lines (the silanediol M+2 mis-read as a Cl-F-S organic)                                                                                       |
| composite     | **composite detection** (`detect_composites`, post-audit, **halogen-adduct-gated**): the M+1 region (¹³C/²⁹Si) is halogen-free so it scales only with the assigned compound; if observed M0 exceeds the M+1-implied intensity, an unresolved co-eluting compound shares the m/z. Flags (does not demote) + reads the co-component's halogen content off the even-shift residual. Skipped in positive/no-halogen mode (the even-shift residual is then ordinary isotope structure) |
| siloxane      | **PDMS/siloxane-ladder assignment** (`siloxane.py`, late + locked): the +C₂H₆OSi (74.0188) silicone oligomer ladder is mass-degenerate per peak (CHON O-monsters out-score the true Si formula at the offset), so a dedicated pass claims it on the **ladder spacing + the ²⁹Si/³⁰Si isotope envelope**, displacing UNLOCKED monsters, Candidate tier — bypassing the CHON-centric heuristics. Inert where the context forbids Si                                                 |
| iso-env (3rd) | **post-pass-6 envelope sweep**: the di-bromide `[M+HBr+Br]⁻` SOA cores commit in pass 6, too late for the earlier two sweeps — this claims their M+2/M+4 satellites                                                                                                                                                                                                                                                                                                               |
| cleanup       | **residual cleanup** (`cleanup.py`, post-pass-6): (1) **isotope-confirmed recovery** — commit CHO + isotope-confirmed covalent-halogen molecules the score gate dropped; (2) **bromide-cluster labelling** with a covalent-fit oracle check ("reagent-adduct preferred over degenerate di-bromo organic"); (3) **ringing/sidelobe artifact flagging** — weak peaks within ~10 mDa of a ≥50k-cps, ≥100× ion → `ROLE_ARTIFACT` (not unexplained); (4) **plausibility demotes** (post-tier, never delete a row) — carbon-cluster (F-free H/C<0.35), implausible-ionization (heteroatom-free hydrocarbon detected through an anion channel that needs an acidic / H-bond site), and speculative-residual (a `residual:*` commit resting on off-cal z, uncorroborated multi-N, a 0-anchor series, or a sole minor channel) each drop a tier; (5) **reagent-halocarbon relabel** (Br-reagent runs only) — bromomethane fragments mis-read as a bare element + Br-cluster (e.g. CH₂Br₂ as "C" via `[M+HBr+Br]-`) are reclassified on their invariant ion composition (→ reagent / named) |
| reflist       | **reference-peaklist rescue** (`reflists.py`, context-gated by run metadata; the contaminant list is always on): **unexplained** peaks matched by mass to an active curated list (α-pinene OH-oxidation HOM · Keller-2008 MS contaminants) are **re-scored by the server** and committed if confirmed, else kept as a tentative low-quality Candidate. Soft + provenance-tagged (`reflist-rescue:*`) — it never overrides an isotope-scored Assigned (the near-tie selection prior lives in pass-1 arbitration)                                                                                                          |
| degeneracy    | **honest cross-family degeneracy** (`degeneracy.apply_degeneracy`, before tiers): re-count distinct plausible IONS in the calibrated window across ALL families; stamps `degeneracy_density` + note (unique / mass-degenerate+tie-set / MASS-SATURATED). The tier engine consumes this — an uncorroborated mass-degenerate commit is capped at Candidate                                                                                                                          |
| audits        | 13C carbon-clamp (pre-pass-4 + post), Br-doublet repair, calibrated mass gate                                                                                                                                                                                                                                                                                                                                                                                                     |

## Chemistry rules (enforced + regression-tested)

- **DBE on the neutral**: non-negative INTEGER + Senior's rule. Half-integer is
  an ion-only artifact (deprotonation), so organic nitrates pass as neutrals.
- **Halogens count as hydrogens** everywhere: DBE, and the Van Krevelen ratio
  uses `(H+F+Cl+Br+I)/(C+Si)` (Si is a C-equivalent backbone atom).
- **Structural oxygen cap**: `O ≤ 2·(C+N+S+P)+4` (valence, not a Van Krevelen
  prior — kills `C3H5ClO17`-type mass-fits; real HOMs pass).
- **Reagent-halogen policy**: the reagent halogen's ION isotope can't prove it
  sits in the NEUTRAL (covalent `X(Br)[M-H]⁻` ≡ `Y·HBr·Br⁻`). The complexity
  prior on the reagent element is never waived by isotope confirmation, and
  anchor·HX clusters are resolved as clusters of the bare analyte.
- **Isotopologue-gated heteroatoms**: a neutral S/Cl/Br needs its Mascope-
  confirmed ³⁴S/³⁷Cl/⁸¹Br or its complexity skepticism is not waived.

## Outputs

**Batch runs** write one versioned folder (`<slug>_<UTC>Z/`) with subdirs
`figures/` `tables/` `report/` `data/` + `merged_ledger.csv` / `run_manifest.json`
/ `batch_summary.json` / `per_file/` at the root — full reference in
**[`docs/OUTPUTS.md`](docs/OUTPUTS.md)**. The **single-sample** `assign` files:

| file                | contents                                                                                                                                                                                                                                                                                                                                                      |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `_ledger.csv`       | every peak: **role** (M0 / iso_child / reagent / **artifact** / unexplained), formula, adduct, scores (incl. arbitration `eff_score`/`eff_margin`/`tied`), ppm, confidence, **tier + tier_reason + candidate_density + degeneracy_density/degeneracy_note**, provenance, commentary, alternatives, isotopologues (reagent rows now carry their `ion_formula`) |
| `_assignments.xlsx` | Summary · Read me (legend) · **Assigned** · **Candidates** (one row per candidate formula) · Unassigned (evidence-characterized) · By class · Unique formulas · Isotopologues · Peak ownership (all peaks) · Target list · Reagent ions — styled: frozen headers, autofilters, number formats, tier/confidence color chips                                  |
| `_summary.md`       | narrative + top assignments + coverage                                                                                                                                                                                                                                                                                                                        |
| `_manifest.json`    | module versions, prescan, series evidence table, per-pass timing                                                                                                                                                                                                                                                                                              |
| `_gka.html`         | interactive rotating-GKA widget (see below)                                                                                                                                                                                                                                                                                                                   |

## Interactive rotating-GKA widget

`python3 scripts/gka_widget.py LEDGER.csv [-o out.html] [--ppm 2]` → a
self-contained HTML (no server). Slider rotates the scaling factor X so
homologous series flatten into horizontal rows; peaks colored by status
(backbone / low / unassigned). Band detector uses the mass-accuracy-derived
tolerance `δGKA ≈ (X/mass(R))·δm`, `δm = ppm·(m/z)·1e-6`. Use it to spot
structure (CF₂ contaminant ladders, oxidation series) the auto-detector did not
already open. `peaky assign` emits one per run (plus a second over the unexplained residual).

## Diel (time-of-day) composition figures

`python3 scripts/diel_classes.py --run-dir <RUN_FOLDER> --label 'NO3 CIMS' --tz-offset -5`
→ `figures/story_<tag>_diel_{composite,individual}.png`. Per-ion HOUR-OF-DAY
profile (median within each local hour on log intensity, normalised to the ion's
own mean) grouped by backbone class (`cluster.formula_class`): the **composite**
overlays one median line per class; the **individual** small-multiples show every
ion's profile (thin) + class median (bold), one panel per class. `--tz-offset H`
sets local time (UTC+H; CDT = −5), `--tier {assigned,all}`, `--classes` to
restrict panels. Reveals the oxidation-state→time-of-day clock (daytime
photochemistry builds oxygen ~13–15 h; nocturnal reduced fatty acids ~03–05 h;
semivolatile chloroparaffins lag to ~17 h). Pure helpers `load_ion_diel` /
`diel_profile` / `peak_hour` are importable. Also accepts `--ledger/--ts-parquet`
directly instead of `--run-dir`.

## Module map (`peaky/`)

| module                | role                                                                                                                                                                                                                                                                                                                                                                                                                     |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `chemistry.py`        | masses (incl. positive adducts: `[M+H]+`/`[M+Na]+`/`[M+NH4]+`/urea), formula algebra, grid (integer-DBE / Senior / O-cap), complexity penalty, grid cache                                                                                                                                                                                                                                                                |
| `contexts.py`         | context profiles (incl. positive `uronium` + `polarity`/grid-width fields) + plausibility filter + contaminant families (incl. long-PDMS `pdms`)                                                                                                                                                                                                                                                                         |
| `ledger.py`           | the peak DataFrame + invariants + commit API                                                                                                                                                                                                                                                                                                                                                                             |
| `io_mascope.py`       | the ONLY Mascope I/O: peaks, cheminfo, parallel `match_compounds` + per-isotopologue parser, adduct detection, `estimate_offset` (rough pre-calibration)                                                                                                                                                                                                                                                                 |
| `isotopes.py`         | prescan fingerprint → grid constraints; **`isotope_pattern()`** envelope predictor (per-element convolution)                                                                                                                                                                                                                                                                                                             |
| `series_gka.py`       | GKA/Kendrick math, repeat units, propagation                                                                                                                                                                                                                                                                                                                                                                             |
| `ladders.py`          | pass-6 anchored homolog/oxidation ladder gap-fill                                                                                                                                                                                                                                                                                                                                                                        |
| `series_detect.py`    | automatic decoy-controlled series detection + chain extraction                                                                                                                                                                                                                                                                                                                                                           |
| `reagents.py`         | reagent-cluster library + labeler (Br isotopologues; positive `[urea_n+H]+`; records `ion_formula`)                                                                                                                                                                                                                                                                                                                      |
| `passes/` (package)   | arbitration (complexity + **calibration-aware off-trend penalty**) + the pass director + offset-tolerant `calibrate`/`confidence_label`/`relabel_confidence` + polarity-aware pass-0. Split: `directors.py` (pass drivers + known-species registry), `core.py` (`arbitrate`/`calibrate`/`commit_winners`/`confidence_label`/`relabel_confidence`/`z_of`), `postprocess.py` (iso-envelopes, audits, composites, demotions), `config.py` (`PassConfig`)                                                                                                                                                                                                                                     |
| `residual.py`         | Pass 4 residual explainer                                                                                                                                                                                                                                                                                                                                                                                                |
| `siloxane.py`         | dedicated PDMS/siloxane-ladder assignment (+C₂H₆OSi spacing + ²⁹Si/³⁰Si envelope, late + locked)                                                                                                                                                                                                                                                                                                                         |
| **`certified_neutral.py`** | PURE certified-neutral core: `channel_offsets` (adduct + cluster-ladder offsets, alias de-dup), `find_certificates` (>=2-channel convergence -> `Certificate`, parsimony selection), `enumerate_certified` (expanded P/S/Cl box for a certified mass ONLY), `ts_covariation` (optional batch-TS corroboration), `corroboration_count`. Pipeline entry `run_pass_certified` (passes/directors, pass 7); standalone `scripts/certify_neutrals.py` (offline certificate table / full oracle mode) |
| `analyte_viz.py`      | **consistent** Van Krevelen + raw time-series from a ledger + batch TS (one row per neutral, RAW intensity, changing-cv threshold). **`render_van_krevelen_full`** = EVERY assigned peak by CHO/CHON/CHOS backbone (Si/F/halogen folded in). `attach_dynamics(bin_minutes=)` for short batches. CLIs: `scripts/analyte_plots.py`, `scripts/analyte_widgets.py` (interactive HTML)                                        |
| `degeneracy.py`       | honest cross-family mass-degeneracy measurement (`degeneracy_density`/note)                                                                                                                                                                                                                                                                                                                                              |
| `cleanup.py`          | residual cleanup: isotope-confirmed recovery, bromide-cluster labelling, ringing-artifact flagging, satellite reclaim, **`prefer_amine_over_ammonium`** (positive: THREE-WAY time-tracking gate — keep `[M+NH4]+` adduct that tracks a shaped parent, re-read to the `[M+H]+` amine when it fails to track / the parent is absent, cap Candidate when weak-or-flat; Si + `protected`-provenance + valence overrides); **plausibility demotes** `demote_implausible_carbon` / `demote_implausible_ionization` / `demote_speculative_residual` + `relabel_reagent_halocarbons` (Br-reagent-gated)                                                                                                                                                                               |
| **`reflists.py`**     | curated, self-describing **reference-peaklist** catalog (`peaky/data/peaklists/`: metadata + version + references + provenance) — `load_catalog`/`active_lists` (context-gated; contaminants always on), `match_assigned` (selection-prior corroboration), `rescue_unexplained_by_reflist` (mass-match → server re-score → commit-if-confirmed, else tentative Candidate). Soft + provenance-tagged; never overrides an isotope-scored Assigned |
| **`sampling.py`**     | THE RULE — `select_representative_samples` (5 evenly-time-spaced + max-TIC) for batch assignment                                                                                                                                                                                                                                                                                                                         |
| **`assign_batch.py`** | `run(batch\|peaks, ts_peaks=, amine_r_min=)` — assign the reps, keep per-file ledgers, offset-aware merge (`align`) + jitter table; applies the positive amine gate at merge level (three-way time-tracking)                                                                                                                                                                                                                                       |
| **`cluster.py`**      | correlation clustering (log-corr, COMPLETE linkage r>0.6, signed distance) → `render_a4` A4-portrait paginated panels + remaining-peaks overview. **Flatness gate** `split_varying`/`render_flat_panel` (cv<`FLAT_CV` bunched, not clustered). `render_changers` = A4-portrait big-standalone-changers page. `write_cluster_workbook(when=)` — byte-reproducible per-cluster XLSX (timestamps pinned to a FIXED content epoch, not the run time) |
| **`composition.py`**  | report composition accounting (pure): `signal_by_backbone` (intensity-weighted CHO/CHON/CHOS), `amine_shadow_stats`/`collapsed_composition` (the [M+NH4]+/[M+H]+-amine degeneracy two-way), `top_species_by_signal`, `oligomer_flag` (high-C high-O HOM-dimer candidates)                                                                                                                                                |
| **`plausibility.py`** | chemical-plausibility QC: `scan(merged, polarity)` flags Candidate-only mass-coincidence formulas (high heteroatom / very low H/C / wrong-mode halogen); Assigned never flagged (powers the `scrutiny` report section)                                                                                                                                                                                                 |
| **`pdf_report.py`**   | STANDARD iterable PDF report (uniform A4) — `build()` over `SECTIONS=[cover, findings, coverage, composition, scrutiny, gka, families, changers, clusters, methods]`, ctx loaded once. PDF filename = `report_<run_id>.pdf`                                                                                                                                                                                              |
| **`gka_figure.py`**   | STATIC GKA findings page: per-family small-multiple Kendrick mass-defect plots (`render_gka`), each rotated to flatten its homologous series into horizontal ladders. a family shows ONLY if it forms a series (`present_families`); contaminants (siloxane/fluorinated) need a short ≥2 ladder, else not plotted. Print counterpart of `scripts/gka_widget.py`                                                          |
| `profiles.py`         | `ReagentProfile` (Br/Ur/NO3: polarity/adducts/normaliser/context) + `resolve('auto', config=)`; `register()` / `load_config()` add reagents from a JSON/TOML file (`--reagent-config`) without forking the package                                                                                                                                                                                                       |
| `timeseries.py`       | **time-resolved disposition** (optional, `--ts-batch`): reagent-normalise a batch's per-sample peaks, cv_norm + family co-variation -> classify each M0 inlet-flat-background vs ambient analyte, demote flat di-bromide/CO3 background                                                                                                                                                                                  |
| `tiers.py`            | Assigned/Candidate tiering (margin, density, lattice/BrCl, **mass-error gate, CO₃-channel gate, degeneracy-aware**)                                                                                                                                                                                                                                                                                                    |
| `report.py`           | Excel / markdown / sheets                                                                                                                                                                                                                                                                                                                                                                                                |
| `assign.py`           | orchestrator + `PassConfig` + module manifest                                                                                                                                                                                                                                                                                                                                                                            |

## Testing & iteration

`pytest tests/` (or `for t in tests/test_*.py; do python3 "$t"; done`) — **the
offline suite must stay green**, no network (io_mascope live smoke gated behind
`MASCOPE_LIVE=1`). `python3 tests/test_smoke.py` is a 2-second no-creds install check.
Every module has a matching `tests/test*<module>.py`; CI (`.github/workflows/test.yml`)
runs the suite on 3.12–3.13 with no credentials. Add a test with each change; keep it green.
See `docs/ARCHITECTURE.md` for the design (ledger model, pass sequence, data
flow), `README.md` for the dev loop, and `docs/ROADMAP.md` for current state +
the open quality work + lessons.

## Gotchas

- `match_compounds` (plural); integer `mz_tolerance`; batched at 200, now scored
  concurrently (5 workers).
- `cheminfo` is flaky/slow and OFF by default (`cfg.use_cheminfo`) — the local
  grid is the primary, complete enumerator. It only adds compound names.
- Extra channels (e.g. `+CO3-`) must be passed as explicit `mechanism_ids`;
  the server's auto-select only covers the sample's own channels.
- `% signal explained` is a coverage metric, not a quality metric — see ROADMAP.
- **Time-series reagent-normalisation needs the reagent IN the measured mass
  range.** A positive urea-CIMS spectrum starting at 122 m/z excludes the main
  `[urea_n+H]+` ions (61/121); `timeseries.reagent_total` then normalises to a
  weak high-n cluster (or falls back to TIC, which is analyte-dominated → closure
  artifact). Check the reagent is actually measured before trusting cv_norm
  magnitudes / disposition timing; use RAW when no good normaliser exists.
- **Per-file assignment ≠ experiment assignment.** `assign.run` assigns ONE
  sample; a time-series event peak weak in that file is missed. For an experiment,
  assign representative files (background + event-extreme) and merge by m/z.
- **Cross-CIMS comparison is ionisation-selective**: Br⁻ and urea⁺ detect
  different compound sets; a matching formula need not be the same molecule.
