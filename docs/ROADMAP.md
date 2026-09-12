# PEAKY — DEV LOG / RESUME HERE (updated 2026-06-21, session 5)

**Project:** consolidate the Mascope pipeline into ONE scalable, shareable Claude Code
skill ("peaky"): representative-sample assignment → merge → time-series clustering →
figures → PDF report. Memory: `agent-peaky` (+ `mascope-sdk-knowledge`, `mascope-assign-package`).

## SESSION 5 (2026-06-21) — ¹⁵N nitrate, native clustering, two-sided halogen handling
**Server is now `<mascope-server>`** (<server> retired); creds in the repo-root `.env`.
Ran the full pipeline on two June-3 batches in dataset **"<dataset>"**:

```bash
# uronium (positive, urea-CIMS):
mascope-assign batch --reagent Ur \
  --batch "<uronium-batch>" \
  --dataset "<dataset>" --out-dir <output>/peaky
# ¹⁵N nitrate (negative):
mascope-assign batch --reagent NO3_15N \
  --batch "<nitrate-batch>" \
  --dataset "<dataset>" --out-dir <output>/peaky
# (add --ts <run>/<tag>_ts.parquet to reuse a cached batch time series and skip the re-fetch)
```
Latest outputs: uronium `<output>/<uronium-run>/`
(1738 M0, OPEs Identified); ¹⁵NO₃⁻ `<nitrate-run>/` (623 M0, 13
chlorinated paraffins Identified). **NEW this session (committed):** `NO3_15N` ¹⁵N
profile + `flatten_match_tree` ¹⁵N re-anchor; `auto_bin_minutes` now bins at the NATIVE
sample cadence (was ~29-min/24h, smeared zero-air events); two-sided halogen handling
(PFCA + chlorinated-paraffin + OPE known-species; `demote_unconfirmed_fluorine` after
apply_tiers). See SKILL.md "Halogen / heteroatom policy" + memory `agent-peaky` SESSION 5.
**OPEN:** nitrate reagent-ion library (label flat ¹⁵NO₃ clusters → lift explained-%);
MS2 for the degenerate halogen families; fold the zero-event ambient discriminant into
clustering. ¹⁵NO₃⁻ ~57% signal explained (rest = mass-degenerate + reagent background).

### CP RECOVERY — known species the server scores too low to anchor (2026-06-21, follow-up)
User asked: chlorinated paraffins that fail on a low server score should still assign "with
user override." Root-caused live: for ¹⁵N-labelled poly-Cl the server's aggregate
`compound_score` collapses (¹⁴N phantom lines + wide Cl envelope), so under `possible_match_
threshold` (0.4) `match_compounds` returns the base ion **UNANCHORED** (no `sample_peak_id`,
ppm NaN). `run_pass0_known`'s main loop iterates only server-anchored bases, so those
congeners never reach the ³⁷Cl check → left unexplained. (Checked: most truly-missing
formulas have NO real peak within 5 ppm — the server was right to score them ~0; blind
formula-forcing would fabricate.) **Fix = evidence-gated AUTO-recovery** (chosen over a blind
force-list): `passes._recover_isotope_locked_known` runs after the main pass-0 loop, re-anchors
each `_RECOVERABLE_KNOWN_FAMS` (={`chlorinated_paraffin`}) base to a real, still-unexplained
ledger peak by **exact mass** (offset-aware, `theo_mz`), and commits ONLY when ≥2 ³⁷Cl
satellites (M0+k·`_D37CL`) are present in the **ledger** (not the depressed server `iso_score`).
Cannot fabricate (no peak / no envelope → no commit); monoisotopic F/P families excluded (no
isotope twin). Commits Identified (method `known:chlorinated_paraffin`), confidence `Good
(chlorinated-paraffin, recovered)`, commentary records the depressed score + satellite count.
Live-verified on <sample>: +5 recovered (C14H26Cl4, C12H21Cl5, C10H16Cl6, C13H23Cl5, C10H15Cl7;
±0.6 ppm, 2–5 sats); C14H26Cl4 & C10H15Cl7 were absent from the prior merged ledger.
test_passes +6 (recover / no-peak-no-fabricate / single-satellite-refuse / F-family-excluded);
suite green. A manual analyst force-list was deferred (auto-recovery covers the real cases).
COMMITTED in **ed2001a** (with the two audit fixes + cutoff below).

### AUDIT OVER-CLEARING — identified peaks wrongly dumped to unexplained (2026-06-21, follow-up 2)
User: "isotopologues that weren't assigned even though they were identified ended in unexplained."
Quantified on the ¹⁵NO₃⁻ batch: ~4.3% of unexplained SIGNAL was server-`[15N]`-tagged peaks our
pipeline left unexplained. Per-pass checkpoint trace (assign.run `checkpoint_dir=`) on <sample> found
TWO over-aggressive audit clears (both in `passes.audit_isotopes` / `demote_carbon_inconsistent`):
  1. **Br-doublet "clear-both"** (`audit_isotopes` step 1): two committed M0s ~1.998 Da apart at
     ~1:1 height with neither formula carrying Br → BOTH cleared, on the premise "a 1.998 doublet
     proves Br." Valid ONLY in Br-CIMS. In ¹⁵N-nitrate (no Br reagent, `cfg.reagent_element=None`)
     unrelated CHON compounds routinely fall ~1.998 apart → it destroyed **27 pairs / 54 real
     M0s** (e.g. C5H6O6+C8H4O4). FIX: gate the clear-both branch on `cfg.reagent_element=="Br"`
     (the lighter-has-Br→⁸¹Br-child branch is valid regardless, kept).
  2. **¹³C carbon-clamp** (`audit_isotopes` step 3 + `demote_carbon_inconsistent`): cleared an M0
     when its measured ¹³C ratio implied a carbon count far from the formula. Fired on genuine
     ~2k cps `[M+¹⁵NO₃]⁻` M0s (C8H11NO6, C10H18N2O8, C9H13NO6) whose ~90–150 cps ¹³C sits near the
     noise floor → ratio reads ~half the carbons → false clear. FIX: clamp ONLY when the ¹³C
     satellite is `>= cfg.height_cutoff` (reliably measured); the over-claim O-monster always has a
     BRIGHT ¹³C, so it still fires (C19-vs-C11 test preserved).
**Residual** `[15N]` (~8k signal, led by C3H4N2O3 @5005cps) is NOT a clearing bug: it's
`_context_filter` plausibility (high-N-density small formulas) + sub-`height_cutoff` peaks (e.g.
C15H26O4 @180cps). User reviewed: C3H4N2O3 is plausible, LEFT AS-IS (no plausibility-filter change).

**`height_cutoff` default 500→100** (PassConfig + assign.py CLI): the `mascope-assign` batch CLI
already defaulted to 100, so per-sample `assign.run()` defaults (used in tests/dev) now match the
batch. Clean per-sample fix impact (<sample>, cutoff=100, same path): M0 **1009→1119**, unexplained
**521→391**, unexplained signal **23%→18%** — consistent with the bug fixes, NOT over-commit.

test_passes +8 total (CP recover/no-fabricate/single-sat/F-excluded; non-Br doublet kept; sub-floor
clamp skipped); suite 31 files green. **COMMITTED ed2001a.** NOT pushed.

**REPORT RE-RUN (batch, ed2001a, --ts reused):** `<output>/<nitrate-run>/`. Merged batch: assigned M0 **623→1375** (Identified 372→**645**, Candidate
251→730), unassigned peaks **1461→684 (−53%)**. Big jump = the Br-doublet bug fired on all 6 files
(no Br reagent) so recovery compounds across files+merge. NB the large Candidate share is honestly
low-confidence (incl. ~314 F-containing mass-coincidences flagged `below_assignability`); the
Identified gain (+273) is the meaningful-confidence improvement. The `reclaim_envelope_tails`
session-5 "no-op on real data" bug is now moot for this batch (the leak was the two audit clears,
not deep halogen tails). Diagnostic scratch was in <output>/<ckpt>/ (cleaned).

### REPRODUCIBILITY INDEX + time-series query (2026-06-22, follow-up 3)
User: the per-batch `<reagent>_ts.parquet` holds the full peak×time table (all peaks, assigned +
unassigned) — could we index the skill so runs are reproducible? Built (new `provenance.py`):
- **`provenance.record_run`** wired into `pipeline.run_batch` (runs last, best-effort, never fatal):
  writes **`run_manifest.json`** into every run dir — code identity (git commit+dirty, package
  version, per-module sha1 hashes, python+dep versions), input (dataset/batch/sample_ids + **ts
  parquet sha1**), the resolved `PassConfig` fingerprint (runtime fields dropped), and output
  (**merged_ledger.csv sha1** + counts) — and appends a compact row to a cross-run registry
  **`<out-dir>/index.jsonl`** (load via `pd.read_json(lines=True)`) for query/diff across runs.
  Same commit + same ts_sha1 → same merged_ledger_sha1 (determinism-tested).
- **`timeseries.trace(run_dir, formula_or_mz)`** — reproducible single-compound trace from a run's
  cached parquet+merged_ledger: pass a neutral formula (→ its highest-score adduct m/z) or a float
  m/z (any peak, assigned or not); returns tidy [datetime_utc, height] summed over the ppm window,
  `df.attrs` carries the m/z + pipeline assignment ('<formula> <adduct> (<tier>)' or 'unassigned').
NB the batch path does NOT pass cfg → it uses `assign.run`'s `PassConfig()` default, so the
cutoff=100 default (above) is what the batch actually uses; the manifest records it. test_provenance
(12) + trace tests (3); suite 32 files green. Live-verified on the 190107Z run (trace C10H14O6 →
Identified 309 pts; trace 179.0076 → unassigned, peak 5004). COMMIT: <this commit>.

### OFF-GRID DISCOVERY — organothiophosphate family is a WHITELIST PATCH; two generic replacements NOT yet built (2026-07-07)
The positive pass-0 `organothiophosphate` family (~19 named OP-thioate/-dithioate insecticides — malathion
`C10H19O6PS2` + homologs, chlorpyrifos, diazinon, parathion, phorate, dimethoate, phosmet…) is a **hand-curated
whitelist**: it only finds species someone thought to list, and P/high-S stay off the generic grid. It closed the
bright unexplained nocturnal cluster in an ambient urea-CIMS batch, but the principled, knowledge-free replacements are **NOT built yet**:

- **Certified-neutral rescue pass** *(BUILT 2026-07 as pass 7 — `certified_neutral.py` + `run_pass_certified` + `scripts/certify_neutrals.py`; docs/CERTIFIED_NEUTRAL.md. First real-ledger offline run blind-rediscovered benzothiazole (C7H5NS, core 135.0143) from the unexplained residual.)* Pair unexplained
  peaks that differ by a known channel offset (H⁺↔urea, H⁺↔NH₄) **AND** co-vary in time → each surviving pair
  *certifies a neutral mass* (two independent mass constraints per neutral, so no mass-degeneracy blow-up). Enumerate
  **THAT one mass only** with the exotic elements opened (P, higher S, Cl) → server-score the survivors. Generic
  off-grid discovery with **no pesticide knowledge** and no combinatorial explosion. This is the intended replacement
  for the whitelist.
- **Fragment-cation modeling** *(NOT buildable without a ledger schema change — honesty caveat)*. In-source fragment
  ions such as `(CH3O)2P=S⁺` (`C2H6O2PS⁺`, m/z 124.98 — the dimethyl-OP marker) have **no intact "neutral M"**, so the
  grid (which enumerates neutrals + adducts) literally cannot represent them. Needs a **ledger schema extension** — a
  fragment/charge role for rows that are a bare cation, not `neutral + adduct`. Until that schema exists, this class of
  evidence can't be captured at all.
**SESSION 4 — SHAREABILITY REFACTOR (2026-06-20). Goal: a small group `pip install`s + validates on
THEIR machines.** A 7-dimension review (48 findings) produced a 5-phase plan. DECISIONS: ship as BOTH a
pip CLI + a portable Claude skill; **mascope-sdk is on PUBLIC PyPI (MIT) → now a CORE dependency** (a plain
`pip install` pulls it, no private index); validators need desktop Claude Code + creds, **NO custom shell
MCP** (built-in Bash, or the CLI in a terminal, is the host-exec path; the cowork `mcp__shell__run_command`
is an author-sandbox artifact); NO3⁻ reagent to be added before ship → reagent profiles must become
config-expandable; the demo (Ur+Br) shareable as the offline demo. Data path LIVE-VERIFIED via host Python
(`mascope-assign list datasets` → 35; "<workspace>" → "<dataset>" → <sample-id>,
961 peaks).
- **Phase 0 DONE — installable.** `pyproject.toml` (PEP 621 / hatchling; deps pinned, mascope-sdk core,
  `[dev]`=pytest), `requirements.txt` lock, public LAZY `__init__.py` v0.3.0 (run / run_batch / run_pipeline
  / PassConfig / get_context / resolve_reagent / ReagentProfile / build_report + __version__), `.env.example`
  (+ `.gitignore` `!.env.example`). Creds docs unified to `~/.mascope/.env` (io_mascope docstring + README +
  SKILL) + `$MASCOPE_ENV` override + an actionable `connect()` error. `pip install -e .` verified from /tmp.
- **Phase 1 DONE — one CLI + safety.** `peaky/cli.py` + `__main__.py`; `[project.scripts]
  mascope-assign`. Subcommands: `list datasets|batches|samples` (discovery — `io_mascope.list_datasets`
  / `list_batches` added), `assign` (+ **`--reagent` / `--adducts` / `--context` / `--env`** — closes the
  silent positive-mode→[M-H]- mis-assignment blocker by forcing the profile's channels), `gka` (offline).
  `gka_widget` moved into the package; `scripts/run_assignment.py` + `scripts/gka_widget.py` are now thin
  back-compat shims. Fail-fast creds preflight + WAF-403 / 401 / stale-404 friendly errors. `test_cli.py`
  (19 checks). **Suite 768 assertions / 27 files green.** README + SKILL run sections rewritten to the CLI;
  SKILL execution language made tool-agnostic (built-in Bash, not the named MCP).
- **Phase 2 DONE — scratch drivers folded into the package (byte-verified).** `peaky/clustering.py`
  `cluster_batch(out_dir, ts, profile)` (lift of run_clusters.py 244 LOC) + `analyte_viz.van_krevelen_batch`
  + `timeseries.auto_bin_minutes` (shared bin heuristic). `pipeline.py` gained `RunContext` / `make_run_context`
  / `generate_report` (offline cluster+VK+report) / `run_batch` (full assign→report) — replacing run_peaky.py's
  `PEAKY_*` env-var + subprocess threading with ONE in-process object. CLI gained `batch` (full, live) +
  `report` (offline regen). run_clusters.py / run_vankrevelen.py → thin shims; run_peaky.py rewritten over
  generate_report (no env/subprocess). VERIFICATION (A/B old driver vs new fn, fixed SOURCE_DATE_EPOCH, Br):
  cluster CSVs+PNGs+xlsx byte-identical (9 figs); VK CSV+2 PNGs byte-identical; **generate_report reproduces the
  whole scratch chain byte-for-byte — 18 artifacts incl. the 6.95 MB PDF**. The PACKAGE now has NO {Ur,Br} maps
  (functions read profile.label); the demo maps remain only in the demo-specific shims. test_clustering (8)
  + RunContext (4). **Suite 780/28 green.** REMAINING: `mascope-assign batch` (full live path) is wired but not
  yet LIVE-validated end-to-end (needs a ~40-min live assign run; WAF-prone) — once it is, DELETE the
  demo-assign scratch shims. Deferred: thread assign_batch's computed maps through its return (the file-based
  inter-stage contract is deterministic + verified, so low priority).
- **Phase 3 DONE — pytest-compat + smoke + determinism + CI.** Each tests/test_*.py now exposes a validating
  `test_all()` guarded by `if __name__=='__main__'`, so `pytest tests/` collects + passes (was INTERNALERROR on
  the module-level sys.exit) while `python3 tests/x.py` still works. NEW `test_smoke.py` (no-network install
  check — public API + every submodule imports, deps resolve, flatten_match_tree, xlsx round-trip; ~2s) +
  `test_determinism.py` (locks the SOURCE_DATE_EPOCH contract: byte-identical PDF/PNG at a fixed epoch, different
  bytes when it changes). `.github/workflows/test.yml` = matrix 3.11–3.13, install from public PyPI, assert NO
  Mascope creds, run smoke + pytest. requires-python>=3.11 (matches the pandas 3.0 lock). **Suite 833/30 green;
  pytest 30/30.** Phases 0–2 committed (4dd8233); Phase 3 committed.
- **Phase 4 IN PROGRESS — onboarding + reuse.** DONE: (4a) NO3⁻ `ReagentProfile` (provisional) + config-driven
  reagents — `profiles.register()` / `load_config()` (JSON/TOML) + `resolve(config=)` + CLI `--reagent-config`
  on assign/batch (add a reagent WITHOUT forking); test_profiles.py 14. (4d) QUICKSTART.md (clone→install→creds→
  list→assign→batch→report + reagent-config example + troubleshooting); README links it, SKILL notes NO3/config.
  Suite 847/31 green. REMAINING: (4c) `--demo` offline mode from a bundled demo parquet — the Br demo is ~5.8 MB
  (per_file 2.1M + ts 3.6M); decide bundle-direct vs Git LFS vs trimmed (.gitignore `*.parquet` needs a carve-out
  + hatchling `artifacts`). (4b) polarity/profile-aware `cleanup.py` (drive RECOVERY_ADDUCTS / reagent_element
  from the profile) — DEFERRED to when NO3⁻ is live-tested, because it changes a verified assignment module that
  can't be re-validated offline (needs a ~40-min Br/Ur re-run to prove no regression). (4e) `ClusterConfig` for
  cluster.py constants. PLUS: LIVE-validate `mascope-assign batch` end-to-end, then DELETE the demo-assign
  shims. **Private GitHub remote DONE — `alekseishcherbinin/mascope-assign` (private); `origin` tracks
  main; CI green on 3.11–3.13.** Creds: `.env` can now live in the repo root (project-local, git-ignored,
  found regardless of cwd via `io_mascope._find_env`) as well as `~/.mascope/.env`.

**REPORT CRITICAL-REVIEW FIXES (session 3, 2026-06-20) — committed, reports regenerated.**
A 6-dimension adversarial review of the Ur/Br PDFs surfaced two genuinely misleading
headline numbers + a missing science narrative; all fixed in-package (presentation only,
no live re-run — regenerated from the canonical demo-assign/<tag>/ ledgers via run_peaky.py):
- **NEW `peaky/composition.py`** (pure; test_composition.py 24): `signal_by_backbone`
  (intensity-weighted CHO/CHON/CHOS), `amine_shadow_stats`/`collapsed_composition` (the
  [M+NH4]+(CHO) vs [M+H]+(amine X+NH3) degeneracy — 243 of 414 Ur CHON share an exact NH3-shifted
  CHO twin counted twice → 639 distinct collapsed), `top_species_by_signal`, `oligomer_flag`
  (C18-40 & O≥7 HOM-dimer candidates).
- **`pdf_report.py`**: (1) NEW `findings` SECTION (page 2, after cover) — event-TIC of the full
  batch (total signal vs wall-clock, rep samples ticked) + data-driven takeaways: event rise×,
  signal-weighted composition (Ur **94% CHO by signal vs 47% CHON by count** — the key correction),
  top species (C10H16O2=16%, the limonene-oxidation ladder), oligomer/HOM line (signal-sorted,
  wrapped). (2) coverage page "Signal & peaks by ROLE" splits explained signal into
  analyte(M0+iso)/reagent/unexplained — Br "98% explained" cover now annotated "analyte 35% /
  reagent 61% / unexplained 4%" (the reagent-dominated-signal fix). (3) NH4→amine caveat GATED to
  positive mode (was wrongly printed on Br negative report). (4) cover formula-disagreements now
  read from the merged ledger's own `formula_agree` (68, authoritative) with a denominator/rate —
  fixes the 65-vs-68 jitter-vs-ledger mismatch. (5) composition page: count + signal-weighted +
  two-way shadow disclosure; text now BEFORE the VK figure. (6) methods page documents the amine
  re-read + a parameters line (tol 6.0ppm · cluster r>0.6 · amine r≥0.7) + reagent-signal caveat.
- ROUND 2 (commit 086ac54): **NEW `plausibility.py`** (test 18) — `scan()` flags Candidate-only
  neutrals that look like mass coincidences (N≥3 & O/C≥1, N≥4 & O≥8, H/C<0.35, or a halogen in a
  positive-mode neutral); Identified (isotope-scored) never flagged. New report SECTION `scrutiny`
  lists them (Ur 5 incl. dibromo C10H18Br2O12; Br 10 incl. C35H4/C8N2O14). **REGRESSION FIXED:** the
  amine shadow note + findings "amine re-reads" line are now GATED to positive mode (`ctx['positive']`
  = any adduct ends '+') — they had leaked onto the Br negative report (which runs no re-read).
  Background-family tagging deliberately NOT added (a formula signature can't separate lab glycols
  from real small acids — it mislabels CH2O2 formic acid, the #1 Br analyte; that's a temporal call).
- ROUND 3 — **DETERMINISM (commit ac08a33).** Ran the full OFFLINE generation pipeline (run_clusters
  + run_vankrevelen + run_report, all read the saved `{tag}_ts.parquet`, NO live call) TWICE with
  different PYTHONHASHSEED + fixed run-id/SOURCE_DATE_EPOCH, diffed every output. Result: every
  PDF/PNG/CSV byte-identical; the ONLY non-deterministic file was `clusters_changing_*.xlsx` — and
  only its embedded timestamps (openpyxl stamps datetime.now() into docProps/core.xml + every zip
  member mtime; sheet data identical). FIX (commits ac08a33 + 1e873ca): `cluster._make_xlsx_
  deterministic(when=)` replaces those stamps with the RUN timestamp (resolved from `when` ->
  SOURCE_DATE_EPOCH env -> now); run_peaky now sets SOURCE_DATE_EPOCH = the run's chosen time so every
  artifact of one run (xlsx + matplotlib PDF/PNG metadata) shares it. So the bytes are deterministic
  GIVEN the inputs+time — same-time re-run = byte-identical (0 diffs across hash seeds, verified), a
  later run carries a later stamp (NOT frozen: folder id / cover / xlsx / PDF all agree, e.g. all
  ...123822Z). User correction: timestamps & report ids SHOULD differ across runs at different times;
  only the scientific CONTENT must be reproducible. test_cluster 38. Harness `/tmp/determinism_test.py`.
  **The live `match_compounds` assignment is the ONLY non-deterministic part of a from-scratch run**
  (server-side; ±2 amine re-reads between two live runs) — all local code is deterministic given inputs.
- SECTIONS now `[cover, findings, coverage, composition, scrutiny, gka, families, changers, clusters,
  methods]`. Full suite green (26 test files). **STILL NOT folded:** enrichment-colored VK
  (scratch); the offsets_ppm=null in batch_summary (IO.estimate_offset returns None — jitter_summary
  has the real per-file offsets; live-path only, low priority).
- ROUND 5 (commits 96773c4 + f423de6, user-spotted from screenshots). (a) `render_changers` (the
  "Large standalone changes" section) rebuilt for FEW channels: was a native-sized strip with empty
  columns + colliding log y-tick labels; now **A4-PORTRAIT page(s)** like the other cluster figures
  (out_prefix + paginated, returns list; pdf_report.changers embeds fit-to-A4 not native; driver passes
  the prefix). Decade-snapped log y (major ticks only), title sized to fit. **Every report page is now
  595x842pt (A4)** — verified. (b) default PDF filename embeds the Report ID → `report_<run_id>.pdf`.
  test_cluster 41 (A4-portrait assertion). Final reports: `<output>/<batch-slug>-{Ur,
  Br}-CIMS_2026-06-20T131{003,052}Z/` (git f423de6). NB a PARALLEL session added scratch
  `<output>/accretion-prototype.py` + an accretion-VK revive note in the memory — left
  untouched (separate workstream).

**SESSION WRAP (2026-06-20, git be91636 + docs).** All report improvements committed; SKILL.md +
README.md refreshed (SECTIONS list, composition.py/plausibility.py modules, A4 changers, PDF-name=Report
ID, determinism note, test count **749 across 26 files**). Tree clean. **NEXT FOCUS = REFACTORING.**
Prime candidates: (1) fold the scratch drivers `<output>/run_{demo,clusters,
vankrevelen,report,peaky}.py` into ONE parameterised package CLI (`peaky/cli.py` or extend
`pipeline.py`) — they're the last copy-paste layer outside the repo and the label/batch/reagent maps are
duplicated across them; (2) the report drivers re-read per-file ledgers to rebuild the explained-mz /
ion-label / meta maps that `assign_batch` already computed — thread those through `out['…']` instead of
re-deriving; (3) consider a single `RunContext` (batch_name/reagent/tag/run_id/when/paths) passed to all
stages instead of env vars (PEAKY_OUT/PEAKY_RUN_ID/PEAKY_GENERATED/SOURCE_DATE_EPOCH). Keep the 749-test
suite green through the refactor; nothing about the report CONTENT should change (byte-diff the outputs).

**Data path:** SDK-over-shell, creds at `~/.mascope/.env`; `io_mascope.connect()` reads the
live file (the `mcp__mascope__*` tools hold a stale token). **<server> is behind a Cloudflare
WAF** — a burst of live runs trips a 403 HTML block ("Attention Required", NOT a token error)
that clears after 15-30 min of NO traffic (polling EXTENDS it). For a blocked live re-run use
`<output>/deferred-rerun.py` (waits it out, then runs).

**PIPELINE — all built + tested this session (`peaky/`):**
- `profiles.py` — ReagentProfile (Br/Ur: polarity/adducts/normaliser/`context`) + `resolve('auto')`.
- `sampling.py` — THE RULE (as built then; **superseded 2026-09-12 by the greedy presence
  set-cover, see `docs/SAMPLING.md`**): assign 5 evenly-TIME-spaced samples + the max-TIC
  sample, then merge (a single averaged file misses part-of-run analytes).
- `assign_batch.py` — `run(batch|peaks, reagent='auto', out_dir, ts_peaks=, amine_r_min=0.7)`:
  select reps → `assign.run` per file (keeps `per_file/<sid>_ledger.csv`) → OFFSET-AWARE merge
  (`align`) + JITTER table; positive reagents get the NH4→amine gate (below). `assign.run` gained
  an `adducts=` override (forces the reagent channels — a sparse-match positive file otherwise
  fell back to [M-H]- = wrong polarity).
- `cluster.py` — TIC/reagent-norm log-correlation, COMPLETE linkage r>0.6, signed distance
  (anti-phase stays apart) → `render_a4` A4-portrait paginated panels (ALL clusters + a
  "remaining peaks" overview); `cluster_rows(maxc=None)`. **FLATNESS GATE (NEW, user 2026-06-20):**
  `split_varying(traces, cols, cv_min=FLAT_CV)` pulls flat traces (cv<`FLAT_CV`=0.30) OUT before
  correlation clustering — flat traces have no reliable shape so their pairwise r is noise and they
  shattered into many spurious n3-4 clusters. The flat set is bunched into ONE `render_flat_panel`
  overview (overlaid traces + median; lists members if ≤72 else count + CSV). Wired in `run_clusters.py`
  for the FLAT set (was 42 intensity-band clusters → 1 panel) and the UNASSIGNED set (had NO cv gate →
  Ur 40 clusters → 5 real clusters r̄0.81-0.96 + 1 flat panel; 365/408 bins were flat noise). The
  CHANGING set is already cv≥0.30 so it IS the varying side (unchanged). `FLAT_CV` is the tunable knob
  (unassigned median cv 0.16, so 0.30 is aggressive — keeps only clear movers; lower it to keep weak
  coherent families). Driver also nukes stale `clusters_*_p*.png` first (shorter run can't orphan pages).
  **SHAPE-CLUSTER REDESIGN (user 2026-06-20 — "varying clusters in the flat panel"):** the per-trace
  cv/transient gate CAN'T see coherence (a synchronized burst barely moves a single trace's cv), so it
  left real co-varying burst families in the flat bucket. NEW assigned path (run_clusters CHANGING):
  cluster ALL bright organic ion-channels (no cv gate) on **RAW** log-correlation — NOT reagent-norm,
  which made the raw-flat background all 'rise' (reagent decays) and collapse into ONE spurious 566-member
  family — then `cluster.merge_similar(Lg, lab, big, merge_r=MERGE_R=0.85, link=MERGE_LINK='complete')`
  folds near-identical-shape clusters (COMPLETE linkage on centroids; average/single CHAIN distinct
  shapes into a blob). The non-clustering remainder + Si = the flat panel. Ur: 75 raw→60 merged families
  covering 673, flat panel 924→567 and now GENUINELY flat (bursts at h0.7-1.5 are their own families;
  big clusters are the h0.2 decay). Cost: ~18 cluster pages (user wanted all signal dissected). The
  per-trace `trace_varies`/`trace_dynamic_range`/`split_varying`/`FLAT_CV`/`PEAK_RANGE` now serve only
  the UNASSIGNED path. `merge_similar` test added (cluster 29).
  **FLAT-CLUSTER DEMOTION (user 2026-06-20 — listed ~33 'flat' clusters):** some clusters PASS the
  correlation cut (members co-vary) but the FAMILY MEAN is flat (correlated background riding the same
  tiny jitter). General rule = a CLUSTER-LEVEL flatness test: `cluster.cluster_flatness(members, traces)`
  = smoothed max/median of the member-MEAN raw trace; `split_flat_clusters(rows, traces,
  range_min=FLAT_CLUSTER_RANGE=1.4)` demotes clusters whose mean doesn't move into the flat panel.
  Threshold from data: user-flagged flat clusters sat at mean-DR 1.04-1.35, real families ≥1.48 (gap
  ~1.4); the metric also caught flat clusters the user missed (general, not their exact list). Ur: 60
  merged → **16 dynamic families** (decay + bursts) + 44 flat clusters (234 ch) demoted; flat panel
  567→801 (still flat); all 33 user IDs demoted. Driver renders only dynamic rows; CSV/workbook = dynamic
  only; flat panel = remainder + flat-cluster members + Si. `cluster_flatness`/`split_flat_clusters`
  tested (cluster 32).
  **BIG STANDALONE CHANGERS (user 2026-06-20 — "some traces increase by a lot, x10, I'm interested in
  those, not others"):** the user does NOT want the weak common-mode event tail (rejected TIC-norm:
  "small changes get exaggerated"); they want the channels that change DRAMATICALLY on their own.
  `cluster.big_changers(traces, cols, grid, fold_min=BIG_CHANGE_FOLD=3.0)` = single channels whose
  smoothed(w2) max/median ≥ fold (~≥5-10× raw), no family needed; `render_changers` = small-multiples,
  one mini-plot per channel (raw cps log y, full-timeline x, titled `formula+adduct  N× · peak h`).
  Driver pulls them from the flat candidates → `clusters_changers_<tag>.png/.csv`, OUT of the flat
  panel. New report SECTION `changers` (after families). Br: 4 (C19H28O6−H 7× spike@h0.1, C20H32O7 4×,
  C3H4O4 3×, C9H16O8 3× — the early-event over-responders); Ur: 1. Threshold tunable (BIG_CHANGE_FOLD).
  cluster tests 36. NOTE: these are the weak-event-tail over-responders — too noisy to cluster, but
  individually a big fold, which is what the user cares about.
- **PER-ION CLUSTERING + CHANNEL-AGREEMENT QC (NEW, user 2026-06-20):** assigned analytes now cluster
  PER ION CHANNEL (formula+adduct), NOT the per-neutral SUM, because a neutral's channels often diverge
  in time — `analyte_viz.channel_agreement` showed **Ur 44% / Br 22%** of multi-channel neutrals have
  their TWO BRIGHTEST channels disagree (r<0.4); reagent-cluster ions (urea/Br-cluster) track
  analyte×reagent so they part from the direct [M+H]+/[M-H]- channel when reagent drifts (many anti-
  correlate, e.g. Ur C10H22O4 r=-0.77). `analyte_viz.ion_traces` (one trace per ion, no summing) +
  `ion_label`/`ADDUCT_SUFFIX` (C6H14O4+Ur⁺). Driver clusters per-ion → Ur 51 neutrals now split their
  channels across different clusters (the divergence, correctly separated). Legend = formula+adduct
  (match-score), ion-count DROPPED. `cluster.write_cluster_workbook` → per-cluster XLSX (summary sheet +
  ONE TAB PER CLUSTER: neutral_formula / channel / m/z / match_score / tier / median_cps / cv).
  `channel_agreement_<tag>.csv` QC output. The families report page notes members=ion-channels.
- **RUN VERSIONING (NEW, user 2026-06-20):** every set of outputs goes in its OWN timestamped folder
  so a re-run never overwrites a prior one. `pipeline.slugify/run_stamp/run_id/make_run_dir` →
  `<output>/<batch-slug>_<YYYY-MM-DDTHHMMSSZ>/` (folder name == run id; pass ONE
  `datetime.now(timezone.utc)` per run so folder/id/cover all agree). **Timestamps are UTC** (folder
  `…Z`, cover `… UTC`) — user 2026-06-20 caught that the run env clock was UTC+3 (EEST) while theirs was
  UTC+1, so a local-time stamp came out 2h off the file dates; UTC is unambiguous. `run_stamp` converts a
  tz-aware `when` to UTC and assumes naive = UTC. `pdf_report.build(run_id=)` stamps the cover (title
  page) with the Report ID and a date+TIME-UTC `generated` line. Orchestrator `run_peaky.py <tag>`
  (scratch) picks the timestamp once, copies the assignment inputs from the canonical `demo-assign/
  <tag>/` into the run folder, then runs clusters+VK+report into it via `PEAKY_OUT/PEAKY_RUN_ID/
  PEAKY_GENERATED` env (run_clusters/run_vankrevelen/run_report honour PEAKY_OUT; default = canonical
  dir). Assignment (live, WAF-gated) still lands in the canonical dir; downstream generation versions.
- `analyte_viz.py` — Van Krevelen (organic + `render_van_krevelen_full` = every assigned peak by
  CHO/CHON/CHOS backbone, Si/F/halogen folded in) + `attach_dynamics(bin_minutes=)` (short-batch).
- `pdf_report.py` — STANDARD ITERABLE report: `SECTIONS=[cover, coverage, composition, gka,
  families, changers, clusters, methods]` (the `changers` section = big standalone changers), ctx
  loaded once by `load_context`. Cover stamped UTC (Report ID + `… UTC` generated). "Assignment quality" page =
  match-score-by-tier + mass-accuracy box (Id/Cand/isotopologue) + assigned-vs-unassigned +
  per-ADDUCT channels with signal%. Cover has batch name + skill version (git sha). Cluster legend
  = `formula (ion-channels / isotope-peaks / match-score)`. The `gka` section renders
  `gka_figure.render_gka` on demand from the merged ledger (also drops a standalone `gka_<tag>.png`).
- `gka_figure.py` (NEW, session 3 2026-06-20) — STATIC GKA findings page, the print counterpart of
  the rotating-GKA widget (`scripts/gka_widget.py`). Small-multiple grid of Kendrick mass-defect
  plots, ONE PER repeat-unit FAMILY (alkyl CH2 / oxidation O,CO,CO2,H2O,C2H2O / alkoxylate
  C2H4O,C3H6O / siloxane C2H6OSi / fluorinated CF2), each rotated to its own base so that family's
  homologous series FLATTEN into horizontal ladders, over a grey cloud of every assigned neutral +
  a family-rollup bar. Format chosen w/ user: grey cloud (not DBE/backbone-coloured), per-family
  small-multiples (not 1 CH2 panel, not coloured-by-family ladders), full y-range. Pure
  `detect_series` / `family_summary` / `kmd`; needs only `neutral_formula` (merged OR single-file
  ledger). CH2-over-detection handled by highlighting only the longest `top_chains=10` series
  ≥`highlight_min_len=5` per panel. Test `tests/test_gka_figure.py` (13 checks). Built into the demo
  Ur+Br reports (`<output>/` has the iteration PNGs v1→v5). **v5 (user review):**
  each panel now highlights ONLY its own base-unit series (every ladder horizontal by construction —
  a TILTED line had meant a folded-in non-base unit, e.g. C3H6O under the C2H4O base); EMPTY families
  (siloxane/CF2 on the demo) are DROPPED via a dynamic ceil((n+1)/2)×2 grid; the rollup bar is now
  base-unit-consistent with the panels (oxidation = 72 O-series, not the 251 O+CO+CO2+H2O family count).
  **v6 (user 2026-06-20 — "siloxanes weren't in GKA"):** CONTAMINANT families (siloxane→Si, fluorinated→F)
  now have an `element` field and are ELEMENT-based, not ladder-based — the 16 Ur siloxanes were assigned
  but never formed a ≥4-rung C2H6OSi ladder (longest run = 3 siloxanols / D3-D4 pair), so the empty panel
  was dropped. Now `present_families` keeps a contaminant panel whenever ≥`MIN_ELEMENT`(3) element-bearing
  peaks exist; `_panel` highlights EVERY Si/F-bearing peak + connects whatever short ladders exist
  (min_len 2). `element_members`/`present_families` exported + tested. Ur now shows siloxane (16, 3 ladders);
  Br shows siloxane + fluorinated (12 F). Organic families unchanged (still need a ≥min_len ladder).
  **v7 (user 2026-06-20 — "if there is no series shouldn't plot in GKA"):** REVERSED the pure
  element rule — a contaminant family is now shown ONLY if it forms a series (`present_families`:
  contaminant needs a ≥`contam_min_len`=2 ladder under its base, not just ≥MIN_ELEMENT element-bearing
  peaks; MIN_ELEMENT removed). So Br's 12 F-bearing (assorted mass-fits, 0 CF2 ladders) NO LONGER plot
  (they scattered, confusing); siloxane stays (it has short C2H6OSi ladders: Ur 3, Br 2). Context: F
  has NO isotope (19F 100%), so high-F formulas are pure mass-coincidence-prone fits — scattered F with
  no CF2 series shouldn't masquerade as a fluorinated family. The F that IS credible gets confidence
  from Br/Cl/S in the ion, not F. test_gka_figure 18.
  [M+NH4]+ as [M+H]+ of the +NH3 amine (SAME ion) UNLESS the NH4 trace co-varies (r>=0.7) with the
  [M+H]+/urea parent OR the amine is valence-impossible (forced). Wired in assign_batch (merged level).
- `residual.py` pass4.A FIX: F enabled only for carbon-CLAMPED pairs (F×wide-C grid was a CPU
  blow-up that HUNG on high-iso-pair Br samples). **CAVEAT: re-verify Br ambient-ref flagships.**

**DEMO RESULTS** (live "the demo (Ur+ CIMS)" / "(Br- CIMS)", <server>, 81/80 samples).
Outputs in `<output>/{Ur,Br}/` (report_*.pdf + clusters/VK pngs + merged_
ledger.csv + per_file/). **Ur 1319 M0 (1065 Id); channels [M+H]+ 670/urea 454/NH4 155/Na 40**
(after the amine gate; from-scratch re-run reproduced it). **Br 502 M0 (402 Id).** Scratch drivers
(NOT in repo): run_demo / run_clusters / run_vankrevelen / run_report / run_triple_traces /
deferred_rerun, all in `<output>/`.

**NEXT (priority):**
1. **Na+ gate (proposed, NOT done):** the 40 uronium [M+Na]+ are the WEAKEST channel (mean score
   0.845, only 20% corroborated, median 3 files, 12 single-file). Apply the NH4-style gate — demote
   uncorroborated/non-co-varying Na+ to Candidate (or drop the opportunistic Na+ channel for uronium).
   The **[M+H2O+H]+ water-adduct channel was REJECTED**: 92 unexplained peaks sit at +18.0106 but only
   1 co-varies with its parent (vs 28% for NH4) — coincidental spacing, not adducts.
2. **Br re-run with the TS wired** (negative; the NH4 gate is a no-op there, just refreshes per-file).
3. **Fold the scratch drivers into the package** (one CLI: assign+cluster+report a batch) + de-hardcode
   → create the GitHub remote + push (none yet; .env is outside the repo, .gitignore covers parquet/log/npy).
4. Binning max-width split guard in `timeseries.build_matrix` (single-linkage chains on dense/drifting data).

**Tests: 24 files green** (sampling 19, assign_batch 12, cluster 36, pdf_report 14, pipeline 9,
cleanup 29, analyte_viz 15, io_mascope 21, gka_figure 18, …). Run `python3 tests/test_*.py`. WAF:
don't run scoring CONCURRENTLY.

---

# Roadmap — state after v48 (2026-06-16) and what comes next

## ✅ DONE (2026-06-16): URONIUM run through the FULL pipeline (positive mode)
The positive-mode **uronium MODE** is built and the full `assign.run` ran on
`<sample-id>` (2025-10-02 08:20, batch "<batch>", dataset
"<dataset>", <server>). Outputs: `<output>/`
(ledger.csv + 11-sheet assignments.xlsx + summary.md + gka.html + FINDINGS.md).
**Result: 701 M0 (606 Identified / 95 Candidate), 165 iso_child, 13 artifact,
398 unexplained; 0 ledger problems; 540 offline tests green.** Channels 474
[M+H]+ / 227 [M+(CH4N2O)H]+ (bright analytes appear in BOTH -> cross-channel
corroboration). What was built:
- **`contexts.uronium`** (alias `urea-cims`/`urea`): `polarity="positive"`,
  N-heavy VK priors (h/c 0.4-2.6, o/c 0-1.5, n/c 0-0.6, dbe/c 0-1.1), grid
  **C46/O32** (new `grid_c_max`/`grid_o_max` ContextProfile fields, read by
  `build_ranges`), max_Si 4 / max_Br=Cl=F 0, pass3 amine/siloxane/glycol/phthalate.
- **urea reagent library** (`reagents._build_positive_library`): [urea_n+H]+ @
  61/121/181/241...; `reagent_for_adducts` returns "urea"; `label_reagents` works
  for it. (No-reagent sample -> ~0 clusters present, as expected.)
- **`assign.run` wiring**: polarity detection from adducts; positive opportunistic
  channels [M+Na]+/[M+NH4]+; `reagent_element` set ONLY for halogens (urea puts no
  halogen in the neutral -> di-bromide/iso-pair/_prefer_adduct inert); urea-cluster
  TS normaliser. `passes._mech_to_adduct`: added the urea diff -> the channel was
  being MISLABELED [M-H]- (it derives the adduct from the ion-vs-neutral element
  diff and fell through to the [M-H]- default). `detect_composites` GATED on a
  halogen adduct (misfires without one). pass-0 known-species is a no-op in
  positive mode.
- **-2 ppm "be aware" (user)**: the source sits at a uniform **-2.45 ppm (sigma
  0.27)**. Reported `ppm_error` stays RAW (you SEE -2.4); the pipeline re-centers
  every quality gate on -2.45. THREE offset bugs fixed (a 0-centered ppm gate
  broke calibration/tiers at a large offset): (1) `passes.calibrate` selects the
  backbone by SCORE not the |ppm|<=2 confidence label; (2) `confidence_label`
  judges ppm vs `cal_mu`; (3) new `passes.relabel_confidence` re-grades pass-1's
  pre-calibration labels post-calibrate; (4) `tiers._calibrate` outlier guard is
  median-relative not 0-centered. Without these the offset collapsed everything to
  Candidate (75/690) and let off-trend monsters survive (e.g. 462=C15H27NO15 at
  +0.32 ppm); fixed -> 606/95 and the monster is correctly cleared.
- **isotopologue attachment** now works (165 iso_child incl. 5 15N / 6 18O / 138
  13C) -> the shortcut's "79/127 unassigned = satellites" artefact is gone.
- **time-series** re-derived on the full ledger (urea-cluster normalised, 1.52 M
  batch peaks): **580 flat-background / 91 intermediate / 30 ambient** (~83% flat
  -- the no-reagent background character the ROADMAP predicted).
- **Br-reference regression: PASSED (live, 2026-06-16).** The v48 sample
  `<sample-id>` is 404 on <server>, but the user pointed out the SAME physical
  sample lives in dataset "<dataset>" / batch "<batch>" at **08:21:59 = `<sample-id>`** (just a different server copy).
  Full run on it: **22/22 flagships present + 0 junk** (offset-aware
  check_flagships). This sample sits at **-1.99 ppm** (vs the old -0.6 ppm copy) --
  a great stress test of the offset-tolerance work, which it passed. It exposed
  TWO more 0-centered gaps, now fixed: (a) **pass-0's |ppm|<=2 known-species gate
  was offset-blind** -> at -1.9 ppm it dropped the silanediol ladder and pass 1
  grabbed 244.97 with the off-trend C5H10O6 junk; fixed with a rough pre-cal
  (`io_mascope.estimate_offset` from the sample's own matches -> `cfg.prior_offset`
  seeds the pass-0 gate); recovered all 5 silanediols + killed C5H10O6. (b)
  **`relabel_confidence` was re-grading LOCKED pass-0 commits** -> the known
  composite silanediol n=4 (35% Mascope score) became "Reject"; now skips locked
  rows (deliberate grades preserved). `check_flagships` is now **offset-aware**
  (bounds judged vs the median-ppm backbone center, so a different-server copy
  passes). +3 tests (io_mascope estimate_offset, passes pass-0 offset gate).

### ✅ DONE (2026-06-16, user-requested): siloxane ladder ASSIGNED + analyte cluster
Outputs now `<output>/`. Two follow-ups the user asked for:
**(A) Assign the characterized PDMS/siloxane ladder.** The 36% unexplained was
dominated by a PDMS ladder spaced by **+74.0186 = C2H6OSi** (the server even fit
684 as a Si10 monster). It's mass-DEGENERATE per peak: at the -2.45 ppm offset a
CHON O-monster out-scores the true Si formula (O is free in the complexity prior),
wins arbitration, then a CHON-centric audit clears it -> stuck unexplained. Three
fixes: (1) new uronium-scoped **`pdms` Pass-3 family** + max_Si 4->12 (shared
siloxane family untouched); (2) **calibration-aware arbitration** (`CAL_ARB_WEIGHT`
in passes.arbitrate: penalise off-trend |z|>accept so an off-trend coincidence no
longer wins-then-z-rejects); (3) **carbon clamp skips Si** (29Si dominates the M+1,
not 13C, so it was clearing real siloxanes). The decisive piece is the new
**`siloxane.py` (assign_siloxane_ladder)**: a LATE, LOCKED pass that finds the
+C2H6OSi ladder and commits each member on the **29Si/30Si isotope envelope**
(oracle-confirmed) as a PDMS oligomer, displacing UNLOCKED monsters, Candidate
tier -- bypassing the CHON heuristics that run earlier. Result: **22 ladder
members committed (+44 Si satellites), Si-bearing M0 ~10->29, unexplained
398->342, signal explained 64%->80.7%** (M0 69.6 + iso 11.1). Tier 604 Id / 115
Cand. 560+ offline tests green (new test_siloxane.py).
**(B) Analyte cluster (the peaks changing over the experiment).** The batch is a
**24-h run** (1188 samples, Oct 1 21:01 -> Oct 2 20:59). Urea-cluster-normalised,
cv_norm + hierarchical clustering: **138 changing peaks (cv>=0.3) vs 1084 flat**.
The analyte cluster (`cluster 12`, ~97 members) is **flat for ~17 h then spikes
~4x at hours 18-22**, cleanly separated from the constant flat background. Led by
monoterpene oxidation products **C10H14O (cv 1.36) / C10H16O2 (cv 1.28)** + a large
oxygenated-CHON/amine pool (C12-15 NOx). Tooling `assign-dev/ts_uro/
analyte_cluster.py` -> plot `uronium_analyte_cluster.png` + `uronium_changing_
peaks.csv`. (The siloxane background is correctly NOT in the spiking analyte cluster.)
Residual gaps (minor): 610/653/684 ladder rungs still unassigned (weak 29Si
confirmation); 684 keeps a C44 amine monster. Characterisation in <run>/
FINDINGS.md; superseded outputs <run>/v3/v4/.

## ⭐ LESSONS from the uronium/positive-mode session (2026-06-16)

1. **A large systematic mass offset exposes EVERY 0-centered ppm gate.** The
   pipeline was built for a ~0-offset instrument; the uronium source sits at
   -2.45 ppm and the Br low-temp copy at -1.99 ppm. SIX independent gates were
   silently hard-wired to 0 ppm and each broke differently: (a) `passes.calibrate`
   selected its backbone by the confidence LABEL (which embeds |ppm|<=2) ->
   biased mu toward 0; (b) `confidence_label` judged |ppm| vs 0 -> whole backbone
   read 'Low' -> tier collapsed to Candidate; (c) `tiers._calibrate` had a
   0-centered |ppm|<=2 outlier cut -> tier engine never calibrated; (d) the
   pass-0 known-species gate |ppm|<=2 dropped on-trend contaminants -> the
   silanediol-vs-C5H10O6 collision; (e) **arbitration** scores |ppm| vs theoretical
   (offset-BLIND) so an off-trend mass-coincidence out-scores the true formula,
   WINS, then is z-rejected -> peak left unexplained; (f) `check_flagships` fixed
   bound. RULE: every ppm judgement is relative to the calibrated/rough center,
   never 0. Reported ppm stays RAW (the user still SEES the offset).
2. **Self-calibration runs too late for the early passes.** pass-0/pass-1 commit
   BEFORE the pass-1 self-cal. Fix = a rough pre-calibration from the sample's OWN
   server matches (`io_mascope.estimate_offset` -> `cfg.prior_offset`) to seed the
   early gates; the real fit refines it. `relabel_confidence` then re-grades pass-1
   post-cal (but must SKIP locked commits -- they carry deliberate grades).
3. **Br-specific machinery misfires in positive/no-halogen mode** -- each needs
   gating on the actual reagent/polarity: composite test (M+1), carbon-clamp
   (reads 29Si as 13C), di-bromide/iso-pair, reagent_element complexity prior.
4. **A mass-degenerate contaminant ladder is NOT assignable by the general
   per-peak machinery** -- CHON O-monsters (O is free in the complexity prior)
   out-score the true Si formula, and CHON-centric audits then clear it. It needs
   a DEDICATED, evidence-decisive pass (series spacing + the actual isotope
   envelope) that runs LATE and LOCKS (`siloxane.py`). Generalisable pattern for
   any homologous contaminant family.
5. **Per-file assignment misses the experiment.** `assign.run` assigns ONE sample;
   a peak weak in that snapshot (e.g. an event analyte while the snapshot is
   pre-event) is never assigned. Assign representative files + MERGE by m/z (one
   event-peak file recovered +146 peaks / +21 changing analytes here).
6. **Reagent-normalise the time series ONLY if the reagent is in the mass range.**
   The uronium spectrum starts at 122 m/z, above the main `[urea_n+H]+` ions
   (61/121); normalising to the weak in-range cluster (or analyte-dominated TIC)
   gives closure artifacts. cv-based SELECTION survived (the divisor was flat) but
   the timing did not. Check first; use RAW when no good normaliser exists.
7. **Cross-CIMS comparison is ionisation-selective** -- Br⁻ and urea⁺ detect
   different compound sets (Br⁻ won't ionise low-O monoterpene products); a
   matching formula need not be the same molecule. Don't assume two reagents on
   the same day = the same air without checking shared compounds co-vary.
8. **A reference sample can move servers / change calibration.** The Br v48
   reference (<sample-id>, -0.6 ppm) 404'd on <server>; the same physical sample
   exists in "<dataset>"/08:21 (<sample-id>) at -1.99
   ppm. Regression tooling MUST be offset-aware to survive this.

## ⭐ NEXT TIME (improvements, priority order)
1. **Finish the siloxane ladder** -- 610/653/684 still unassigned (weak 29Si
   confirmation) and 684 keeps a C44 monster. Relax `siloxane.py`'s 29Si gate for
   INTERIOR ladder rungs (membership between two confirmed rungs is itself strong
   evidence) and let the series spacing carry them.
2. **"Experiment-assignment" mode** -- a flag that auto-picks representative files
   (background + TS event-extremes), runs `assign.run` on each, and emits ONE
   merged ledger. Formalise `assign-dev/ts_uro/merge_experiment.py`. (match_compounds
   is per-sample, so a synthetic union can't be scored -- merge real files.)
3. **Time-series-DRIVEN assignment** -- use the TS to find the changing peaks
   FIRST, then target-assign them across files, so experiment analytes are
   prioritised over flat background.
4. **Robust no-reagent normaliser** -- detect when the reagent ion is absent from
   the mass range and fall back to a median-of-flat-bins normaliser (not TIC).
5. **Positive-mode flagship set** -- establish + pin a validated uronium flagship
   list (like the 22 Br flagships) so positive runs have a regression guard.
6. **Generalise offset-awareness** -- audit for any remaining 0-centered ppm
   reference; consider a single `calibrated_ppm()` helper used everywhere.
7. **ladders.py positive mode** -- the pass-6 gap-fill is Br-anchor-specific; make
   the oxidation/homolog diagonals fire on positive adducts too.

## Where the pipeline stands

**LATEST GOOD: v48** (Br-CIMS) on `<sample-id>` (ambient air, <instrument>
2025.10.02). v48 = v46 + time-series findings converted to standing pipeline code
(new timeseries.py, [BrHF]- inorganic cluster, reclaim_satellites, below-
assignability flag+sheet) + the terminal steps applied:
**270 M0 (164 Identified / 106 Candidate — 7 flat di-bromide/CO3 demoted by TS),
iso_child 320, reagent 32, artifact 17, unexplained 330, 26 below-assignability,
22/22 flagships, 0 junk, 540 offline tests green** (was 497; +43 for the positive
uronium mode + offset-tolerant calibration). NB the Br sample is now 404 on <server>
(only the local cache remains) — re-derive flagships from the cached v48 ledger. Outputs `assign-dev/v48/`
(v48_ledger.csv + v48_assignments.xlsx, built by make_v48.py). Run the full
pipeline time-resolved with `run_assignment.py --ts-batch '<batch>'
--ts-dataset '<dataset>'`. (Earlier good: v46 = 471 tests,
21/21 flagships; v47 = consolidation/Orbitool audit.) Run `python3
scripts/check_flagships.py <ledger.csv>` after ANY change — it asserts the 22
validated Br identifications (Br-specific; N/A to uronium) and the banned junk.

### v45 -> v46: Orbitool cross-check fixes (chemistry-gate gaps)
Triggered by comparing our v45 to an Orbitool peak list of the SAME sample (the
Orbitool author flagged that DBE filtering can reject real species). Two real,
narrow gaps fixed (we were NOT making the core error — we threshold DBE on the
NEUTRAL, already assign glycerol, keep isoprene polyols):
- **h_to_c CEILING 2.6 -> 2.75** (ambient/chamber/indoor, contexts.py): admits C3
  saturated polyols (glycerol C3H8O3 H/C 2.67, propylene glycol) in candidate
  generation. They previously slipped in ONLY via the pass-4 iso-pair bypass;
  DBE>=0 already caps H/C at 2+2/Ceff so the ceiling only clipped C3 glycols.
  Effect: glycerol now a pass-1 **High** commit (was "Good (iso-pair)").
- **Nitroaromatics added to pass-0** (`_known_species` in passes/directors.py, "nitroaromatic"
  family: C6H4N2O5 dinitrophenol, C6H5NO3 nitrophenol, C6H5NO4 nitrocatechol,
  C7H6N2O5 dinitrocresol): H-poor / high-DBE BrC tracers the ambient VK floor +
  DBE/C ceiling block from the grid. Only those present + |ppm|<=2 commit. Effect:
  **dinitrophenol C6H4N2O5 [M-H]- @183.0047 now Identified** (the only M0 add).
- Result: +1 M0 (dinitrophenol), 0 lost, 21/21 flagships, +7 tests (contexts
  31->36, passes 110->112). Comparison + finds reports in `assign-dev/v45/`
  (ORBITOOL_COMPARISON.md, ORBITOOL_GENUINE_FINDS.md, TOP10_DBE_ISOTOPE.md).
- **Orbitool genuine-finds verdict (oracle-confirmed):** Orbitool has NO bright,
  well-scoring organic peak we wrongly drop. The strong SOA (succinic, C6H10O3,
  C8H10O4, C9H14O4) we already assign; the genuinely-missed leads (C5H8O4,
  C10H16O8/O18O8, C9H14O5, C13H20O5, C5H6O3) score only 0.43-0.47 on the oracle
  (mass-only, no 81Br-twin corroboration) -> real compounds but unconfirmed here;
  time-series (#3) is the principled way to claim them, not a looser bar.

### v44 → v45: reagent-formula / `[81BrO]-` fix MATERIALISED (was uncommitted)
The post-v44 reagent fix is now in an output. `reagents.build_library` returns
`(label, mass, ion_formula)` and enumerates BOTH 79/81Br isotopologues for the
BrO/BrO2/BrO3 oxide anions; `ledger.mark_reagent(ion_formula=)` records every
reagent ion's formula. PRINCIPLE: known formula -> assigned, regardless of class
(analyte/contaminant/ion-source). Verified in v45: the `[81BrO]-` twin at 96.91
(2350 cps) is now `reagent`/`BrO-` (was unexplained); **17/30 reagent rows carry
an `ion_formula`** (Br3-, Br2-, BrO-, C2H2Br3O2-, CH2Br3O2-; was 0/29) — the 13
without one are the generic multi-Br clusters where no confident formula scores
above threshold. **Zero regression**: the M0 set is byte-identical to v44 (0
lost, 0 new, 0 tier changes); the only delta is reagent 29→30 / unexplained
330→329 (the one [81BrO]- peak). Outputs in `assign-dev/v44/` retained for the
diff.

### v45 step 0(b): top-10 unexplained <400 redone with DBE + isotope enforced
Done (analysis only, no locked IDs added — see `assign-dev/v45/TOP10_DBE_ISOTOPE.md`
and `assign-dev/dbe_iso_top10_v2.py`). Method validated PRINCIPLE 2 live: the
isotope-envelope check (predicted `isotope_pattern` vs observed M+2/M+4) decisively
re-ranks candidates. KEY RESULTS: (1) **386.98 = C17H12N2O2S [M+Br]- (0.73,
+0.20 ppm on-cal, DBE 13), isotope-confirmed single Br** — a benzothiazole/azo-dye
N2S aromatic *contaminant*; the higher-scoring Br-free C9H12N2O11S2 (0.79) is
ruled out by the 81Br twin (exactly the lesson). Tentative Candidate-grade, NOT a
pass-0 lock (it's a one-off contaminant at 0.73). (2) **365.10**: isotope FLIPS
the old guess — C20H17N2O3S [M-H]- (0.75, on-cal) is Br-free and contradicted by
the 367 twin; the iso-consistent C15H26O5 [M+Br]- (0.77) is -1.75 ppm off-cal, so
1-Br is confirmed but the formula is marginal. (3) **HALF the top-12 are isotope
SATELLITES** of brighter neighbors (388.98=81Br twin of 386.98; 335.07=81Br twin
of the recovered 333.07 C14H22O4 = the deferred M1; 300.75=37Cl twin of a reagent;
398.01=13C twin of an iso_child; the 138.94/140.94/142.93 Br lattice). CONCLUSION
unchanged: sub-400 residual is reagent-bromide lattice + satellites of assigned
peaks, not missed organics; DBE+isotope narrows a couple of "mass-saturated"
peaks to a best guess but adds no locked IDs.

### v43->v44: adversarial-review fixes (workflow wf_<id>, 12 confirmed-real)
- **H3** (tiers.py v0.3.0, in pipeline): tier engine consumes `degeneracy_density`;
  uncorroborated mass-degenerate commits capped at Candidate.
- **H1/H2** (cleanup.recover_isotope_gated): recovery restricted to CHO +
  isotope-confirmed covalent halogen (RECOVERY_BOX dropped N/S/P). C14H22O4
  [M+Br]- KEPT (->Candidate), C18H14N2O3S REJECTED (N2+S uncorroborated). A
  broad-competitor-score veto was tried first but over-rejected C14H22O4 (an
  impossible DBE<0 fluorinated fit out-scores it) -> the CHO-only +
  heteroatom-corroboration rule is the right discriminator.
- **H4** (cleanup.flag_ringing_artifacts): MIN_RING_PARENT=50000 cps + 100x ratio
  floor -> the physically-impossible 273.05 flag (parent 6385 cps) dropped; 17
  real artifacts kept (FT sidelobe amplitude scales UP with parent intensity).
- **M4** (cleanup.label_bromide_clusters): covalent-fit oracle check before
  commentary -> 5/18 clusters get "reagent-adduct reading preferred over
  degenerate di-bromo organic (ion C2H2Br3O2- 0.96 / CH2Br3O2- 0.83)".
- DEFERRED: M1 (wire recovery 81Br twins as iso_child -- would make recovery
  read "corroborated" and H3 would then SPARE it from demotion; the adduct-Br
  twin is non-discriminating, so leaving it unattached is correct for tiering);
  M3 (mostly resolved by H4); M5 (report molecular vs non-molecular separately).

### KEY METHOD LESSON (user, 2026-06-13): always enforce DBE + isotope count
The server `cheminfo` enumerator does NOT enforce DBE -- a raw cheminfo re-score
returns valence-impossible "fits" (C5H12F13O DBE=-6.5, F14H11N3O no-carbon) that
look high-scoring but are nonsense. Use the LOCAL grid (DBE/Senior/O-cap enforced)
or filter with `chemistry.dbe_ok`. AND constrain by the isotope-confirmed halogen
count: 386.98/388.98 are a 1-Br pair (M+2/M0=1.09, no M+4) -> only 2 DBE-valid
1-Br candidates, best guess **C17H12N2O2S [M+Br]- (0.73, +0.2 ppm, DBE 13** --
likely a benzothiazole/azo-dye N2S aromatic contaminant). The Br-free 0.79 fit
(C9H12N2O11S2) is RULED OUT by the 81Br twin. So several "mass-saturated" verdicts
were partly an artifact of skipping DBE -- some heavy peaks DO narrow to a real
best guess. TODO next session: redo the top-10 unexplained <400 with DBE+isotope
enforced (top-3 DBE-valid isotope-consistent formulas each).

Pipeline shape: pass 0 (known species, locked, twin-gated) → pass 1 (CHO/CHON
backbone + self-calibration) → pass 2 (GKA series) → pass 3 (evidence-opened
contaminant families) → **iso-envelope completion** → **pre-pass-4 carbon
clamp** → pass 4 (residual iso-pairs/deep series) → pass 5 (known-neutral
completion) → isotope-physics audit → calibrated mass-gate audit →
**iso-envelope completion (2nd sweep)** → **composite detection + de-blending**
→ pass 6 (anchored ladder gap-fill) → **iso-envelope completion (3rd sweep,
post-pass-6: claims the di-bromide [M+HBr+Br]- cores' M+2/M+4 satellites that
pass 6 commits too late for the earlier sweeps)** → **residual cleanup
(cleanup.py: isotope-confirmed low-complexity recovery + bromide-cluster
labelling + ringing/shoulder artifact flagging; new ROLE_ARTIFACT)** →
**degeneracy audit (degeneracy.apply_degeneracy)** → tiers.apply_tiers
**(degeneracy-aware: caps uncorroborated mass-degenerate commits at Candidate)**.

Quality work added 2026-06-13 (single-file, all tested, 21/21 flagships, 0 junk):
mass-error-distribution test + background-CO3-channel gate (tiers.py v0.2.0,
de-risk 179->170 Identified); honest degeneracy measurement (degeneracy.py,
note: unique / mass-degenerate+tie-set / mass-saturated) now CONSUMED by the
tier engine (tiers.py v0.3.0): an uncorroborated commit whose cross-family
degeneracy_density>2 (or MASS-SATURATED) is capped at Candidate -- degeneracy
runs before tiers, corroborated backbones (iso/cross-channel/series) are spared
(v43: 2 self-contradictory "unique"+MASS-SATURATED recovered ions demoted,
172->170 Identified, all 108 corroborated high-degeneracy rows kept);
3rd envelope sweep (+13 di-bromide isotopologues, 60.6->61.4% peaks explained).
INVESTIGATED + EXHAUSTED for sub-400 unexplained: the residual there is the
reagent-bromide lattice, not missed organics. The CH2-ladder "no-fit" peaks =
real homologous families: Family A (di-bromide SOA M+2/M+4) now attached;
Family B (heavy-halogen CH2 ladder, defect -0.10..-0.16) is mass-degenerate AND
its isotope envelopes are owned by neighbors (halogen-counting fails) -> NOT
crackable from one averaged spectrum, needs time-series. NEXT real unlock =
time-series co-variation.

## Built this session (v36–v41) — the isotope machinery

- **Isotope-envelope completion** (`isotopes.isotope_pattern` per-element
  convolution + `passes.complete_isotope_envelopes`, before pass 4 + post-audit).
  ~44% of the bright "residual"/Candidate peaks were ISOTOPE SATELLITES (M+2/13C)
  of brighter peaks, not new compounds. Attaches unexplained satellites +
  DISPLACES weak (non-High, weak-score) M0s onto their true parent. Fixed the
  393/395 bug (silanediol Si4+Br M+2 mis-read as a phantom Cl-F-S). iso_child
  276→304; peaks-explained 58.6→60.6%.
- **Composite-peak detection** (`passes.detect_composites`). The M+1 region
  (13C/29Si) is halogen-FREE so it scales only with the assigned compound; if
  observed M0 exceeds the M+1-implied intensity, an unresolved CO-ELUTING
  compound shares the m/z. Reads the co-component's halogen off the even-shift
  residual (M+2/M0, M+4/M+2 ~ Br/BrCl/Br2). The silanediol n>=3 rungs are
  ~30–45% co-eluting BrCl/Br — formula (Si4, proven by +74.0188 rung spacing)
  AND prediction (binomial, Mascope agrees) are BOTH correct; the PEAK is mixed.
- **Composite de-blending** (`passes.split_composites`, user-designed). The
  owner keeps `assigned_fraction` of its measured height; the co-eluting
  compound becomes a SYNTHETIC `<id>.2` sub-peak at the same m/z (synthetic=True,
  host_peak_id->host, carries co_height + co_halogen). Signal conserved:
  silanediol 393 → host eff 10994 + sub 9092 = 20086 measured. stats() uses
  effective height + excludes synthetic from n_peaks. Signal 91.0→90.2% (the
  honest drop = co-component fractions no longer mis-credited).
- 6 adversarial-review findings fixed in the isotope code (81Br mass constant
  was 0.16 mDa off + inconsistent; dead tier-guard; max_shift truncation; flat
  ppm window mislabeling 29Si/13C). See commit c882594.

## What the residual IS (eliminated — do not re-chase)

- **Sulfate / nitrate**: dead (0.06x / 0.0–0.1x decoys; the sample's N lives in
  the small acids + dinitrophenol, see below).
- **The C/H lattice is biogenic SOA** (v30-v32): bright n_Br=2 peaks are
  mono-/sesquiterpene oxidation products as `[M+HBr+Br]-` reagent clusters
  (covalent-Br2 and a Br2 reagent adduct are the SAME ion). 12 di-bromide CHO
  cores assigned (C9–C15 H_xO_4 ladders, e.g. 409=C15H22O3). All CHO-only.
- **The diagonals are MOSTLY NOT new compounds.** Repeatedly characterized
  (workflows + live scoring): the rotating-GKA diagonals are dominated by
  (a) 81Br/13C ISOTOPE SATELLITES of brighter peaks marching as parallel CH2
  ladders (an isotope envelope of a CH2 series IS a CH2 series), and
  (b) fluorinated-contaminant CH2/C2H4O ladders. A non-integer GKA rotation
  makes contaminant ladders + satellites look like SOA — adversarial
  verification (constant-DBE, same-adduct, satellite guard, dedicated isotope
  envelope, live score) is MANDATORY before encoding any diagonal as chemistry.
- **Di-bromide naming campaign on the remaining nBr=2 residual: ZERO new
  names** (workflow wf_<id>). All 14 candidates rejected on adversarial
  verification — including 424.99=C15H22O4, whose M+2/M+4 are OWNED by a
  confirmed single-Br fluorinated contaminant (C14H12F8O [M+Br]-), so its +HBr
  pairing only proves a SINGLE-Br adduct. The remaining nBr=2 residual is:
  tribromide/3-Br reagent clusters (194.95/208.96/280.76/282.76/199.85/238.76),
  multi-Br mass-domain-ambiguous (124.92/140.92/192.95/194.94), one
  multiply-charged (239.03), and uncorroborated N2/S mass-fits with FABRICATED
  isotope envelopes (their apparent M+2/M+4 are crowding from adjacent assigned
  di-bromides). The clean di-bromide SOA already got assigned in v41; what
  remains is not nameable from the sum spectrum.
- **Reagent = dibromomethane (CH2Br2)** → Br-/Br2-/Br3- clusters (Br3- DOMINANT
  ~129k cps) + trace HBr. Server has +Br-/+Br2-/+Br3- mechanisms. [M+Br3]- must
  NOT be a blanket scoring channel (timed out batches, lost 40 base M0s incl.
  TFA). OPEN: the [M+HBr+Br]- vs covalent-monobromo [M+Br]- LABEL choice for
  di-bromide cores is the user's call (same ion; 480-cps HBr.Br- argues for
  covalent-monobromo). Left as-is pending decision.

## Next steps, in priority order

0. **DONE (v45, 2026-06-15):** (a) re-ran to v45 — reagent-formula / `[81BrO]-`
   fix materialised, zero regression, 21/21 flagships; (b) redid the top-10
   unexplained <400 with DBE + isotope enforced — 386.98 -> C17H12N2O2S [M+Br]-
   confirmed, no locked IDs added (see the v45 block at the top + `assign-dev/
   v45/TOP10_DBE_ISOTOPE.md`). **Still open from 0(c):** **M1** — wire the
   recovered-ion 81Br twins (e.g. 335.07 = the twin of 333.07 = C14H22O4 [M+Br]-)
   as `iso_child`. STILL DELIBERATELY DEFERRED: attaching corroborates the
   recovery, so H3 would spare it from the Candidate cap, contradicting the
   intended Candidate outcome. Only do it if the user wants those twins off the
   residual AND accepts the tier consequence. **M3** (symmetric artifact split)
   is largely resolved by H4. Nothing here is blocking; the real remaining unlock
   is time-series (step 3).
1. **Name the composite co-components** (the open de-blending step). The
   `<id>.2` synthetic sub-peaks now sit in the ledger with their m/z + measured
   halogen constraint (the silanediol rungs host ~9k-cps BrCl sub-peaks). Build
   a constrained match on them (halogen pinned, mass fixed) to NAME the
   co-component → this formally allows TWO M0 owners per peak (the data model
   supports it: distinct peak_id, host link, fractional signal). DESIGN CALL the
   user flagged: do we commit the `.2` peak as a second M0?
2. **Add dinitrophenol C6H4N2O5 [M-H]- (183.0047, 808 cps, -0.24 ppm, DBE 6)**
   to the pass-0 known-atmospheric list. A genuine missed nitroaromatic tracer
   (verified credible during the v41 row analysis; the ONE solid new name from
   the diagonal hunting — it is CHON, immune to the di-bromide envelope trap).
   Candidate (tentative): 293.0579 = C15H15ClO4 [M-H]- (respects measured Cl,
   1215 cps, but -2.4 ppm — marginal).
3. **Time-resolved correlation confirmer — DONE (2026-06-16, THE unlock realised).**
   Data arrived on the **<server>** server: dataset "<dataset>",
   batch **"<batch>"** (= Oct-2, the SAME day as the v46 reference
   sample), **1199 samples ~1/min over 24 h**. Pipeline: `assign-dev/ts_*.py`
   (load -> bin 5 ppm -> sample×bin matrix -> **reagent-normalize** by Br3- to
   kill sensitivity drift -> cv_norm + Pearson corr + hierarchical clusters ->
   interpret/verify workflow). FULL FINDINGS: `assign-dev/ts/TIMESERIES_FINDINGS.md`
   + plot `family_timeseries.png`. KEY RESULTS (adversarially verified, wf agents):
   - **Inlet vs ambient cleanly separated**: median |diel_ratio-1| = 0.013 (flat
     background) vs 0.246 (biogenic). Fluorinated (61/64 flat) + silanediol (9)
     "Identified" are inlet/instrument background -> re-tier to contaminant.
   - **Monoterpene biogenic SOA CONFIRMED** (cluster 6, 25 CHO members C10H16/18Ox,
     r_mono 0.94, true intra-family co-variation not shared-diel). v46's biogenic
     assignments validated. BrO-/HO2 are HOx hitchhikers on the morning envelope,
     not SOA members.
   - **C15H24O3 [M+Br]- (331.09): promote Candidate->Identified-Good** (r_formic
     0.96, presence 0.998, unique formula). Cap at Good (mass -2..-3 ppm).
   - **DI-BROMIDE "SOA CORES" OVERTURNED**: every di-Br core (398.98/409.00/440.99/
     342.92/370.95/304.90/358.95) is BRIGHT + FLAT (cv_norm 0.06-0.22, r_mono~0,
     diel~1.0) = stable inlet/instrument background, NOT biogenic SOA. (Flatness
     is not low-S/N: the flattest are the brightest, 33-51k cps.) Keep the mass/
     isotope tier, REMOVE the biogenic-SOA-core label. This corrects the v30-v41
     "C/H lattice = di-bromide biogenic SOA" theme below.
   - The "anti-correlated nighttime" cluster is mostly a **Br3-normalization
     closure artifact** (anti-PHASE, diel~1.0), not night chemistry. Its 11
     unassigned co-varying peaks are single-Br (1.9979 Da) inorganic/halogen
     condition ions, NOT missed organics; 169.954/171.952 are 13C/13C+81Br
     satellites of the Identified C3H6BrO3-.
   - 13 concrete pipeline actions in TIMESERIES_FINDINGS.md §5 (re-tier, promote,
     downgrade sparse, name/decline). NEXT: ingest time-series into the pipeline
     (a `timeseries.py` module: cv_norm + family-correlation -> auto re-tier
     contaminants, promote co-varying Candidates, gate di-bromide labels). Needs
     oracle confirmation on the flagged items (server was offline during the wf).
4. **Below-assignability certificates.** For each bright has-constraints
   residual peak, run the clamped frame×element enumeration and stamp the result
   into the Unassigned sheet ("searched N frames, zero fits → composite/exotic").
5. **Robustness niceties.** Batch-level re-retry before fail-loud in
   score_candidates; the [M+Br]- vs [M+HBr+Br]- label decision (#OPEN above).

## Standing lessons (encode-don't-remember)

- **TIME-SERIES: always reagent-normalize (analyte/Br3-) BEFORE correlating** —
  raw heights carry an instrument-sensitivity/reagent common-mode that inflates
  every correlation (a 234-bin "cluster" of everything). Normalization isolates
  real chemistry. BUT it introduces a CLOSURE artifact: anything that doesn't
  co-vary with the (SOA-tracking) Br3- denominator shows up ANTI-correlated —
  report anti-correlation, not a night/RH mechanism, unless the ABSOLUTE diel
  confirms it. **cv_norm < ~0.22 + flat diel = inlet/instrument background**
  (bright+flat is the textbook stable-source signature); **high cv_norm + high
  intra-family r = real ambient**. This is the inlet-vs-ambient discriminator.
- **Di-bromide "SOA cores" are flat inlet background, not biogenic SOA** (time-
  series, 2026-06-16): they fail the ambient benchmark (cv_norm 0.06-0.22 vs
  0.36-0.64, r_mono~0 vs ~0.95). The v30-v41 "C/H lattice = di-bromide biogenic
  SOA" reading is OVERTURNED for the flat cores — keep their mass/isotope tier
  but drop the biogenic-SOA label. (The mono-Br monoterpene SOA C10H16/18Ox IS
  confirmed ambient.)
- A di-bromide (or any multi-isotope) name needs its OWN DEDICATED isotope
  envelope — M+2/M+4 at the predicted intensity that isn't already owned by an
  adjacent assigned species. A good ppm + an apparent M+2/M+4 is NOT enough; the
  envelope is routinely fabricated by crowding from neighbors (the v41 campaign:
  all 14 di-bromide candidates failed exactly here).
- The diagonals' "well-aligning rows" are mostly ISOTOPE SATELLITES (an isotope
  envelope of a CH2 series is itself a CH2 ladder) + the multi-halogen lattice.
  Good ppm ≠ real; high-DBE / O-monster / het-ignoring mass-fits are fictions.
- The M+1 region (13C/29Si) is halogen-free → it gives a compound's true
  intensity independent of any coincident halogen co-eluter (the composite test).
- Every evidence GATE needs a complementary RECOVERY path.
- Negative evidence (absent satellite) never overrides independent positive
  evidence (agreeing channel, twin satellite).
- Server scoring is authoritative; never trust hand ion-mass arithmetic.
- Coverage metrics reward fiction; flagships + count-based coverage are honest.
- GKA: trust integer-A(R) rows; verify non-integer rotations; half-integer X is
  a feature (C/H lattice / fixed-heteroatom view).
