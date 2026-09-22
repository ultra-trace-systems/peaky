# Evidence levels — how much a committed formula is worth

**Status: implemented (B2).** `peaky/assignment/evidence.py` is the `evidence`
stage of `assign.run` and the pooled recompute of `assign_batch.run`;
`tests/test_evidence.py` pins the contract. The executable reference for every
predicate below is still `scripts/level_ledger.py`; the in-core stage reproduces
it row for row on the fixtures in `tests/fixtures/levels/` and hits the three
golden count vectors exactly. Where this document, the script and the code
disagree, the script is right and the other two are bugs (one known reading
difference is listed in §10).

## 1. What a level is

`tier` says whether the engine is willing to *print* a formula (Assigned) or
only to *offer* it (Candidate). It is a verdict on the assignment. A level
says what the evidence behind a committed formula is worth on the scale a
reader of an identification paper already knows — Schymanski et al. (2014),
*Environ. Sci. Technol.* 48, 2097 — numbered **downward, 1 = best**, adapted
to chemical ionization, where there is no chromatography, no fragment
spectrum, no library, and the reagent is part of the ion.

One level per **committed M0 row**. Isotope children, reagent ions,
artifacts and unexplained peaks carry no level (`NA`). The level is computed
per `(neutral_formula, adduct)` over the rows that share it inside one
*source* (a file, or a batch's pooled files — §6), and stamped on every M0
row of that pair.

Levels 1 and 2a are defined and never fire: 1 needs an authentic standard
measured in the same source and reagent chemistry; 2a needs a library
spectrum, and CIMS has none.

## 2. The columns a level reads

All from the ledger `peaky.assignment.ledger` already writes. Every read is
null-safe: NaN, `pd.NA`, `None`, `''` and `'false'` are False;
`bool(numpy.nan)` is True and once inflated 5b by 127 rows.

| column | rows | read as |
|---|---|---|
| `role` | all | `M0` are levelled; `iso_child` supply satellites; everything else is ignored |
| `peak_id`, `parent_peak_id` | M0, iso | a child hangs off the M0 with `peak_id == parent_peak_id` **in the same file** |
| `iso_label` | iso | the satellite's tag before any `+` (`13C+1` → `13C`; `81Br`, `34S`, `37Cl`, `29Si`, `30Si`, `18O`, `15N`) |
| `height` | M0, iso | the ratio child/parent, tested against natural abundance |
| `ion_formula` | M0 | carbon count for the 13C expectation |
| `neutral_formula`, `adduct` | M0 | the pair the level is about |
| `method` | M0 | `known:<family>` marks a pass-0 curated identity |
| `confidence` | M0 | its first word: `High`, `Good`, `Low`, `Suspect` |
| `tied` | M0 | the arbiter broke a near-tie |
| `below_assignability` | M0 | `tiers.flag_below_assignability` (O ≥ 11 and mass-saturated) |
| `degeneracy_density` | M0 | plausible ions sharing the mass in the calibrated window (`degeneracy.apply_degeneracy`) |
| `degeneracy_note` | M0 | contains `MASS-SATURATED` when the audit capped |
| `resolvability` | M0 | `resolved` / `isolated` / `blended` / `unresolvable`; **trace-first only** — absent on the cover path |
| `series_unit`, `anchor_peak_id` | M0 | a homologous-series or anchor tie |
| `isotopologues` | M0 | the server-attributed satellite list (a repr'd list; non-empty counts as isotope support) |
| `tier` | M0 | reported beside the level; never read by a predicate |

Optional, reported only: `mz`, `ppm_error_cal`, `occurrence`, `n_files`.

## 3. The evidence record — one per (neutral, adduct) per source

Computed exactly as `level_ledger.measure_source` does.

| fact | definition |
|---|---|
| `iso` | some child of some row of the pair has a height ratio within **0.5–2.0×** of natural abundance (13C: 0.0107 per carbon of `ion_formula`; 34S 0.0443; 37Cl 0.3196; 81Br 0.9728; 29Si 0.0508; 30Si 0.0335) **or** any row's `isotopologues` list is non-empty |
| `multiline` | ≥ 2 distinct satellite tags across the pair's children (excluding `M0`) |
| `carbon_ev` | a `13C…` tag is among them |
| `chan2` | the **neutral** is committed under ≥ 2 distinct adducts in this source |
| `anchor` | any row has `anchor_peak_id` or `series_unit` |
| `branch` | the neutral's adduct set meets both `{[M-H]-}` and `{[M+NO3]-, [M+15NO3]-, [M+^NO3]-, [M+Br]-, [M+HBr+Br]-, [M+CO3]-}` — deprotonated **and** clustered: the gas-phase-acidity branch |
| `reagent_only_iso` | the channel has a reagent halogen (§3.1), the pair has satellites, none is 13C, and every tag starts with that halogen's heavy isotope (`81Br` / `37Cl`; iodine is monoisotopic, so an iodide channel never sets it) |
| `tied` | **all** rows of the pair are tied |
| `below` | **any** row is below assignability |
| `lowconf` | **all** rows are `Low` or `Suspect` |
| `degeneracy` | median `degeneracy_density` over rows that have one; NaN when none has |
| `saturated` | any `degeneracy_note` contains `MASS-SATURATED` |
| `res_ok` | no row carries a `resolvability` value, **or** at least one is `resolved` / `isolated` — a source that never measured it is not penalised |
| `corroborated` | the neutral is in the **cross set** (§6): the other reagent channel, the other instrument on the same air, or a `--corroborate` source |
| `known_fam` | the family of the first `known:` method among the rows, else `''` |

Derived:

```
n_axes         = iso + chan2 + anchor + corroborated            # 0..4
cross          = corroborated or multiline or known_fam != ''    # an axis outside this channel's ionization chemistry
neutral_backed = corroborated or chan2 or anchor or known_fam != '' or carbon_ev
degenerate     = saturated or degeneracy >= 3
unique         = degeneracy <= 1                                  # NaN is neither
hard           = tied or below or lowconf
```

`n_files` (files of the source that carry the pair) is **recorded** in
`evidence_axes` but is **not an axis**: the golden vectors were measured
without it, and whether persistence across files should count is a
calibration question for B3, decided on roster hit rate and decoy rate.

### 3.1 The reagent halogen

The halogen of the source's **commonest cluster adduct** (`[M+Br]-` → Br;
`[M+NO3]-` → none), never of "any halogen adduct present": a nitrate channel
with two stray `[M+Br]-` rows against 346 `[M+NO3]-` is a nitrate channel.

## 4. The decision table — first predicate that holds wins

| order | level | predicate | meaning |
|---:|---|---|---|
| 1 | **5b** | `hard` | the assignment argues with itself: a near-tie the arbiter broke, a row below assignability, or a score the engine calls Low/Suspect |
| 2 | **5b** | `degenerate and n_axes == 0` | mass-degenerate with nothing to break the tie |
| 3 | **2b** | `known_fam != ''` and `scope(known_fam) == "compound"` and `n_plausible_structures == 1` | a curated **identity** on a formula that admits one structure |
| 4 | **3a** | `known_fam != ''` (any other curated commit) | a named class, isomers open |
| 5 | **3b** | `branch` | a substituent only: the same neutral deprotonated and clustered means an acidic hydrogen, nothing more |
| 6 | **4c** | `n_axes == 0 and unique and res_ok` | formula unopposed — one plausible ion in the calibrated window on a separable peak — but nothing corroborates it |
| 7 | **5a** | `n_axes == 0` | exact mass only; no discriminating test was possible |
| 8 | **4d** | `not neutral_backed and reagent_only_iso` | **ion** formula only: the sole isotope support is the reagent halogen, which pins the ion and says nothing about the neutral (CIMS-specific; no Schymanski analogue) |
| 9 | **4a** | `n_axes >= 2 and cross` | formula confirmed **and the neutral established**: two orthogonal axes, at least one from outside this channel's ionization chemistry |
| 10 | **4b** | else | formula confirmed, one corroboration |

Levels **1** and **2a** are in `LEVEL_ORDER` and never assigned.

### 4.1 Level 2b needs both an identity claim and a unique structure

`scope(family)` is a property of the pass-0 registry, not of the level: a
family whose entries are hand-listed compounds (`atmospheric`,
`reactive_iodine`, `ambient_inorganic`, `nitroaromatic`, `cyclosiloxane`,
`indoor_sulfur`, `organophosphate`, `organothiophosphate`, `easyic_hydride`)
asserts a **compound**; a family generated from a formula loop
(`perfluoroacid` CₙHF₂ₙ₋₁O₂, `chlorinated_paraffin` CₙH₂ₙ₊₂₋ₓClₓ,
`contaminant:silanediol`) asserts a **class**. `n_plausible_structures` is
read from `peaky/data/isomer_space.csv` (§5); a formula absent from it has
`NA` and can never be 2b.

The boundary case is trifluoroacetic acid: CF₃COOH is the only structure of
C₂HF₃O₂, so the isomer space says 1 — and it stays **3a**, because the
perfluoroacid family's commit says "a PFCA of this formula", not "TFA". A
per-compound entry (or a standard, level 1) would lift it. This is why 2b
is "from isomer space, never a whitelist": the level rule reads a structure
count and a registry scope, and no list of formulas lives in `evidence.py`.

Nitrophenol (C₆H₅NO₃) is the other direction: a compound-scope family, but
the isomer space says 3 (2-, 3-, 4-nitrophenol), so 3a.

### 4.2 Settled decisions

- **One rating channel.** A level rates a `(neutral, adduct)` pair as seen
  through one channel; a second channel is an *axis* (`chan2`), not a second
  rating. The batch stamps the same level on every merged row of the pair.
- **Cross-reagent (and cross-instrument) agreement is formula evidence, never
  more.** It can carry a row to 4a; it cannot make 3b, 3a or 2b, which need
  a structural claim the mass cannot give.
- **Reagent ions are excluded by role.** `role == reagent` rows never carry
  a level; a reagent cluster that a pass mis-committed as an analyte is the
  reagent stage's bug, not a level.
- **The isomer ceiling is stated per row.** `n_plausible_structures` is
  written on every M0 row whose formula is in the isomer space (`NA`
  otherwise). It caps 2b and is reported so a reader sees that a 4a
  C₁₀H₁₆O₃ is one formula and, by a chemist's count, dozens of structures.
- **Resolvability binds only where it was measured.** The cover path does not
  produce it; a cover-path 4c is not denied for lacking it.
- **Isotope evidence is judged on physics.** A satellite whose ratio is
  outside 0.5–2× natural abundance is not evidence; a satellite the server
  attributed (`isotopologues`) is.
- **Predicate order is the design.** `hard` outranks a curated identity: a
  known species the arbiter had to tie-break is 5b, and that is a defect in
  the pass-0 lock to be fixed there, not hidden here.
- **The tier is not an input.** Levels and tiers are computed from the same
  columns and may disagree; B3 measures where.

## 5. `peaky/data/isomer_space.csv`

`formula, n_plausible_structures, family, name, rationale`. One row per
formula a pass-0 family can commit, with the count of structures a chemist
would accept for that formula in this chemistry and the reason, in words.
Seeded in B1 with the 2b and 3a formulas of the three golden sets and the
positive-mode families; B2 loads it through `peaky.paths.pkg_data`. Adding a
row needs a rationale; a row without one is a whitelist entry and is refused
by `tests/test_evidence.py::test_isomer_space_rows_carry_a_rationale`.

## 6. Where it runs

### 6.1 Single sample — a stage in `assign.run`

A new `_Stage("evidence", …)` in `peaky/assignment/assign.py::_STAGES`,
placed **after `iso_env_final` and before `timeseries`**: after every tier
and demote stage (the last of which is `plausibility`), after
`reflist_rescue` (which commits M0 rows post-tier), and after the final
isotope sweep (which adds the satellites the `iso` axis reads).
`timeseries` writes only `ts_*` dispositions and is not an input.
`safe=False` (a level that cannot be computed is a bug, not a lost stage),
`store=True` (the summary goes to `batch_summary`).

The stage calls `evidence.apply_levels(ledger, cfg=cfg, cross=cross)` which
writes the four columns of §7 in place and returns the summary. `cross` is
the set of neutral formulas from `--corroborate` sources (repeatable CLI
option on `peaky assign` and `peaky batch`; a run dir, an out-dir holding one
run, or a ledger CSV; resolved as `level_ledger.resolve_source` does); empty
for a bare single-sample run.

### 6.2 Batch — recomputed on the pooled files, stamped on the merged ledger

The merged ledger carries none of the predicate columns, so the batch level
is **not** read off merged rows. `assign_batch.run` pools the per-file
ledgers of the assigned files (cover **and** residual stages) as one source
and calls `evidence.level_pooled(per_file, cross=cross)`: one evidence record
per `(neutral, adduct)` over all files, so `chan2` sees a second adduct in
*any* file, `iso` any file's satellite, `tied`/`lowconf` require *all* rows
(across files) and `below` any. The result is joined onto the merged ledger
by `(neutral_formula, adduct)` — each merged row is one ion, so the join is
one-to-one — and the four columns are written there too. The per-file
ledgers keep their own per-file levels (computed at 6.1).

Two batch runs named together (`peaky batch … --corroborate <other run>`)
corroborate this run by the other's M0 neutrals. The trace-first path is a
one-file batch and follows the same rule.

`batch_summary["evidence_levels"]` = `{"pooled": {level: n}` (one count per
pair), `"merged": {level: n}` (per merged row), `"per_stage": {cover: {…},
residual: {…}}`, `"n_pairs"`, `"n_unstamped"` (merged rows whose reading no
per-file ledger holds), `"n_corroborate"`, `"cross_source": [...]}`; the pair
table with every fact of §3 is written to `tables/evidence_levels.csv`.

### 6.3 The post-hoc script

`scripts/level_ledger.py` stays as the reference and the tool for runs made
before the column existed. `scripts/scorecard.py` already prefers an in-core
`evidence_level` column when present.

## 7. The column contract

| column | type | on | value |
|---|---|---|---|
| `evidence_level` | str | M0 rows; `NA` elsewhere | one of `2b 3a 3b 4a 4b 4c 4d 5a 5b` (never `1` / `2a` today) |
| `evidence_axes` | str | M0 rows | `|`-joined, in this order, of the axes that hold: `iso`, `chan2`, `anchor`, `corroborated`, then the modifiers `multiline`, `carbon`, `branch`, `reagent_only_iso`, `known:<family>`, `files:<n>` (batch only); `''` when none |
| `level_reason` | str | M0 rows | one sentence naming the predicate that fired, in the words of §4, with the numbers (`"5b: near-tie broken by the arbiter"`, `"4c: 1 plausible ion in the window, resolved, no axis"`, `"4a: iso + chan2, corroborated by the other source"`) |
| `n_plausible_structures` | Int64 | M0 rows whose formula is in the isomer space; `NA` otherwise | from `isomer_space.csv` |

Written to: the per-file `<prefix>_ledger.csv`, `merged_ledger.csv`,
`tables/evidence_levels.csv` (batch: one row per pair with every fact of §3), the
Excel workbook (a column on the ledger sheets and a new **"By evidence
level"** sheet: one row per level with count, share, tier split, the axes
histogram and the twenty brightest rows), one PDF table after the tier
table, the publish comment (`io/publish.py` adds `evidence_level` to the
published ledger fields), `docs/OUTPUTS.md`, README, SKILL.md. Every
consumer must render a ledger **without** the columns unchanged (older
runs, `peaky report` on them).

## 8. Golden fixtures and what the tests pin

`tests/fixtures/levels/` (see its README): 42 gzipped per-file ledgers from
three real runs, trimmed to the rows and columns of §2, and
`expected_levels.csv` with the level and every fact of §3 per
`(source, neutral, adduct)`, written by `level_ledger.py` at the commit that
added them. `tests/test_evidence.py`:

- one test per level (2b, 3a, 3b, 4a, 4b, 4c, 4d, 5a, 5b) on a synthetic
  ledger where exactly that level fires, **and a mutant** that flips one
  input and must change the level;
- the three golden count vectors, exact:
  `tv` 1373 → 21/15/107/38/162/9/9/33/979,
  `tof` 3364 → 6/16/182/38/260/79/91/135/2557,
  `orbi` 1707 → 0/12/217/44/215/119/0/30/1070;
- row-for-row equality with `expected_levels.csv`;
- non-M0 rows carry `NA`; a reagent-ion row never carries a level;
- missing columns and all-null columns do not raise;
- the pooled batch computation equals the script's pooling on the same files;
- every row of the isomer space carries a rationale; every compound-scope
  2b formula in the fixtures is in the isomer space;
- `evidence_axes` order and `level_reason` are stable strings.

All of it passes since B2; `tests/test_evidence_outputs.py` pins the wiring
(stage order, the batch recompute, every output, `--corroborate`).

## 9. Worked rows (real rows of `expected_levels.csv`)

| source | neutral | adduct | facts | level |
|---|---|---|---|---|
| tv_bromide | HO2 | [M+Br]- | known:atmospheric, one 81Br satellite | **2b** — identity outranks the reagent-only isotope reading |
| tv_nitrate | C2HF3O2 | [M-H]- | known:perfluoroacid, chan2, corroborated by the bromide channel | **3a** — class scope: TFA stays 3a |
| orbi | C9H8O3 | [M-H]- and [M+^NO3]- | branch (bare and clustered), iso, chan2, anchor | **3b** |
| tv_bromide | C5H8O3 | [M-H]- | iso, corroborated by the nitrate channel | **4a** |
| orbi | C7H8O6 | [M-H]- | anchor only, 12 of 12 files | **4b** |
| orbi | C4H8O6 | [M-H]- | no axis, one plausible ion in the window, resolvability not measured | **4c** |
| tof | H2O2 | [M+Br]- | one 81Br satellite, no 13C, no other axis, 23 files | **4d** — the ion HBr·O₂H⁻ is pinned, the neutral is not |
| orbi | C8H6O5 | [M-H]- | no axis, two plausible ions | **5a** |
| tof | HO2 | [M-H]- | known:atmospheric **but** Low confidence | **5b** — `hard` outranks the identity (a defect for the pass-0 lock) |
| orbi | HBr | [M+^NO3]- | iso, anchor, below assignability | **5b** |

Across the three sets 4,606 of 6,444 pairs are 5b; the top two levels hold
70. That distribution is the point of the scale, not a problem with it.

## 10. Open for B2/B3 (not decided here)

- Whether `n_files ≥ 2` becomes an axis (B3, on calibration).
- Whether a level above 350 m/z with one axis should be capped at 4b (the A3
  decoy split: above m/z 350 the mass-defect gap no longer protects the
  Orbitrap) — B3 measures, the predicate table changes only if the
  calibration order breaks.
- One reading difference between the code and the reference script, to be
  adjudicated in B3 on real runs: the ledger writes `isotopologues` with
  `json.dumps`, so a satellite without a per-line score (a ³⁷Cl envelope
  confirmed against the ledger) carries `null`; the script's `ast.literal_eval`
  cannot read that cell and counts the list as empty, the in-core `as_list`
  reads JSON first. No fixture row moves (the goldens and the row-for-row test
  hold), but a real run's chlorinated-paraffin rows may differ on the `iso`
  fact between the two.
- A detectability term for heteroatom satellites (a 34S/81Br line that would
  sit below the noise edge cannot count as *missing*) — C3's topic; the level
  today only counts satellites that were seen.
