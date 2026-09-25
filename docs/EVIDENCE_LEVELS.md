# Evidence levels — how much a committed formula is worth

**Status: implemented (B2), validated on real runs (B3), verified (B4).**
`peaky/assignment/evidence.py` is the `evidence`
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
| `resolvability` | M0 | `resolved` / `isolated` / `blended` / `unresolvable`; stamped by the `resolvability` stage on every run that has a width model (a batch measures one from the raw profile of a middling spectrum; `--resolving-power` declares or declines it); NA on an offline run without one |
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
| `corroborated` | the neutral is in the **cross set** (§6.4): a source — the other reagent channel, the other instrument on the same air, a `--corroborate` run — that holds it at **4b or better by its own evidence**; never for an ion-only pair |
| `known_fam` | the family of the first `known:` method among the rows, else `''` |
| `ion_only` | the pair was written by the **ion-only stage** (`ion_only`, C7): adduct `[M]-.` with method `ion_only:*` (a merged ledger: an `ion_only_of` link) — the +1.0078 Da electron-attachment line beside a committed `[M-H]-` acid, carrying the acid's composition. An ion-only pair is levelled on its **own** satellite alone (§4 row 2b′) and is kept **out of the per-neutral pools in both directions**: `chan2`, `branch`, `anchor` and `reagent_only_iso` are computed over the regular rows only, so the row never gives its parent a second channel and never takes an axis from it; it is never `corroborated` and its neutral never enters a cross set (`corroborating_neutrals` skips it) |

Derived:

```
n_axes         = iso + chan2 + anchor + corroborated            # 0..4
cross          = corroborated or multiline or known_fam != ''    # an axis outside this channel's ionization chemistry
neutral_backed = corroborated or chan2 or anchor or known_fam != '' or carbon_ev
degenerate     = saturated or degeneracy >= 3
unique         = degeneracy <= 1                                  # NaN is neither
hard           = tied or below or lowconf
```

Two of these terms can never decide a level today and are kept because the
reference script states both (verified by mutation, B4): `carbon_ev` inside
`neutral_backed`, and the `not carbon_ev` guard of `reagent_only_iso` —
`reagent_only_iso` already requires every satellite tag to be the reagent
halogen's, so a pair with a ¹³C line never reaches the 4d test at all.

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
| 1′ | **4d** / **5a** | `ion_only` | an ion-only row: **4d** when its own satellite passes the band (`iso`) — the composition is pinned by exact mass and ¹³C, the ionization process and the neutral are open — else **5a**, exact mass only. The same rung the reagent-halogen case (row 8) reaches by the other route: 4d = *ion pinned, neutral not*, by either route |
| 2 | **5b** | `degenerate and n_axes == 0` | mass-degenerate with nothing to break the tie |
| 3 | **2b** | `known_fam != ''` and `scope(known_fam) == "compound"` and `n_plausible_structures == 1` | a curated **identity** on a formula that admits one structure |
| 4 | **3a** | `known_fam != ''` (any other curated commit) | a named class, isomers open |
| 5 | **3b** | `branch` | a substituent only: the same neutral deprotonated and clustered means an acidic hydrogen, nothing more |
| 6 | **4c** | `n_axes == 0 and unique and res_ok` | formula unopposed — one plausible ion in the calibrated window on a separable peak — but nothing corroborates it |
| 7 | **5a** | `n_axes == 0` | exact mass only; no discriminating test was possible |
| 8 | **4d** | `not neutral_backed and reagent_only_iso` | **ion** formula only: the sole isotope support is the reagent halogen, which pins the ion and says nothing about the neutral (CIMS-specific; no Schymanski analogue). With row 1′, 4d reads *ion pinned, neutral not* whichever route reached it |
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
- **Resolvability binds only where it was measured.** Every run with a width
  model (measured from the raw profile, or declared) produces it; a run without
  one (offline, no `--resolving-power`) is not denied 4c for lacking it.
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
`reflist_rescue` (which commits M0 rows post-tier), after `ion_only` (C7:
the electron-attachment rows, post-tier like the rescue, placed before the
final sweep so that sweep claims each new row's own ¹³C — the satellite row
1′ reads), and after the final isotope sweep (which adds the satellites the
`iso` axis reads).
`timeseries` writes only `ts_*` dispositions and is not an input.
`safe=False` (a level that cannot be computed is a bug, not a lost stage),
`store=True` (the stage summary is kept in the run's `summaries` and written
to the single-sample `<prefix>_manifest.json`; a batch reports the pooled
recompute of §6.2 in `batch_summary.json` instead).

The stage calls `evidence.apply_levels(ledger, cfg=cfg, cross=cross)` which
writes the four columns of §7 in place and returns the summary. `cross` is
the cross set of §6.4: the neutral formulas the `--corroborate` sources hold
at 4b or better by their own evidence (repeatable CLI option on `peaky
assign` and `peaky batch`; a run dir, an out-dir holding one run, or a ledger
CSV; resolved as `level_ledger.resolve_source` does); empty for a bare
single-sample run.

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
corroborate this run by the neutrals the other holds at 4b or better on its
own evidence (§6.4). The trace-first path is a one-file batch and follows the
same rule.

`batch_summary["evidence_levels"]` = `{"pooled": {level: n}` (one count per
pair), `"merged": {level: n}` (per merged row), `"per_stage": {cover: {…},
residual: {…}}`, `"n_pairs"`, `"n_unstamped"` (merged rows whose reading no
per-file ledger holds), `"n_corroborate"`, `"cross_source": [...]}`; the pair
table with every fact of §3 is written to `tables/evidence_levels.csv`.

### 6.3 The post-hoc script

`scripts/level_ledger.py` stays as the reference and the tool for runs made
before the column existed. `scripts/scorecard.py` already prefers an in-core
`evidence_level` column when present.

### 6.4 The cross set — what a source's sighting is worth (C8)

A source corroborates a neutral only when it **pins it on its own**: at level
**4b or better** when the source is levelled with **no cross set at all**
(`evidence.source_neutrals`; `CORROBORATE_MAX_LEVEL = "4b"`). Below that the
source's grid merely enumerated the same formula at a peak — 4c unopposed but
unconfirmed, 4d the ion only, 5a exact mass alone, 5b arguing with itself —
and two grids agreeing is not a second sighting of the neutral.

- A source with per-file ledgers (a run dir, an out-dir holding one run, a
  per-file or single-sample ledger CSV) is **re-levelled**: its ledgers pooled
  as ONE source, `level_pooled(…, cross=None)` — the batch's own merged-row
  level, §6.2 — and a neutral counts when any of its pairs reaches 4b.
  Stored levels are never read here, so a source that was itself run with
  `--corroborate` cannot hand a run back the agreement it got from it, and
  two sources named together (`level_ledger.py A B`) are each corroborated by
  what the other pins on its own — never by their mutual agreement.
- A merged ledger (no `role` column) carries none of the predicate columns:
  its stored `evidence_level` is read, **without its own `corroborated`
  axis** — a row counts at 4b or better when `corroborated` is not among its
  axes, or when it still holds `iso` / `chan2` / `anchor` once that axis is
  taken away (and is not an `iso`-only `reagent_only_iso` row, which would
  fall to 4d). A level the axis alone produced — a 4b of one corroboration, a
  curated row with no axis of its own — does not count. A source carrying
  neither the predicate columns nor `evidence_level` raises: there is nothing
  to judge the sighting by.
- Ion-only pairs never count, in either direction (§3).

Measured on the same-air pair before the rule (three regression runs, the
per-file levels, the merge vote and the merged levels replayed with the new
cross set; each replay's baseline reproduced its run row for row): the
labelled-nitrate Orbitrap offered the TOF 1521 neutrals, 379 of them pinned on
its own; the TOF offered the Orbitrap 3728, 435 pinned (3437 of its pooled
pairs are 5b — a 10k-resolution TOF can seldom pin a formula); of the TOF's 49
vote winners lifted by the axis alone, 28 rested on an Orbitrap 5a / 5b. The
merge vote (C2b) then changes 36 winners on the TOF (every test-set row and
every roster compound keeps its reading), 12 on the Orbitrap (all between
implausible grid formulas) and none on the uronium channel, where the 4a rungs
built on the TOF's 5b agreement fall to 4b. The curated perfluoroheptanoic
acid, mass-saturated on both instruments with no axis on either, falls from
3a to 5b on both: its one "axis" was the mutual agreement.

## 7. The column contract

| column | type | on | value |
|---|---|---|---|
| `evidence_level` | str | M0 rows; `NA` elsewhere | one of `2b 3a 3b 4a 4b 4c 4d 5a 5b` (never `1` / `2a` today) |
| `evidence_axes` | str | M0 rows | `|`-joined, in this order, of the axes that hold: `iso`, `chan2`, `anchor`, `corroborated`, then the modifiers `multiline`, `carbon`, `branch`, `reagent_only_iso`, `ion_only`, `known:<family>`, `files:<n>` (batch only); `''` when none |
| `level_reason` | str | M0 rows | one sentence naming the predicate that fired, in the words of §4, with the numbers (`"5b: near-tie broken by the arbiter"`, `"4c: 1 plausible ion in the window, resolved, no axis"`, `"4a: iso + chan2, corroborated by the other source"`) |
| `n_plausible_structures` | Int64 | M0 rows whose formula is in the isomer space; `NA` otherwise | from `isomer_space.csv` |

Written to: the per-file `<prefix>_ledger.csv`, `merged_ledger.csv`,
`tables/evidence_levels.csv` (batch: one row per pair with every fact of §3, the
`ion_only` flag included), the
Excel workbook (a column on the ledger sheets and a new **"By evidence
level"** sheet: one row per level with count, share, tier split, the axes
histogram and the twenty brightest rows), an **Evidence levels** page in the
PDF report after the assignment-quality page and a line on its cover, the
published engine provenance (`io/publish.py` carries the four columns in
`engine_provenance`, dropping `NA` values), `docs/OUTPUTS.md`, README, SKILL.md. Every
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
- the three golden count vectors, exact (each source corroborated by what
  the other pins on its own, §6.4):
  `tv` 1373 → 21/15/107/16/143/15/10/37/1009,
  `tof` 3364 → 6/15/182/22/247/82/99/138/2573,
  `orbi` 1707 → 0/11/217/9/203/139/0/35/1093
  (before C8, with any-level membership: 21/15/107/38/162/9/9/33/979,
  6/16/182/38/260/79/91/135/2557 and 0/12/217/44/215/119/0/30/1070 —
  183 of the 6,444 pairs moved, every one to a lower level);
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

Across the three sets 4,675 of 6,444 pairs are 5b; the top two levels hold
68 (4,606 and 70 before §6.4). That distribution is the point of the scale,
not a problem with it.

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
  fact between the two. **B3 adjudicated (2026-09-22): no row of the three
  working-set channels moved on it, plain or corroborated — the commits that
  carry a `null` score are `known:` rows, levelled 3a before the `iso` fact is
  read.** The script keeps its reader; aligning it would move no golden.
- Whether `neutral_backed` should count `anchor` and `carbon_ev` — B3's
  calibration (the wrong-adduct decoy) shows both are facts about the **ion**:
  a ¹³C line and a series tie survive a wrong adduct set, so 64 wrong-adduct
  rows sit at 4b and 30 at 4c on the labelled-nitrate channel and 339 at 4c on
  the uronium channel. Option (a): drop them from `neutral_backed` and send rows
  whose only axes are ion-level to a generalised 4d ("ion pinned, neutral not";
  the goldens move, `level_ledger.py` and `evidence.py` change together, R1–R3
  re-run and re-validated row for row). Option (b): keep, and document that
  4b / 4c rate the ion formula. The user's decision; not changed here.
- A detectability term for heteroatom satellites (a 34S/81Br line that would
  sit below the noise edge cannot count as *missing*) — C3's topic; the level
  today only counts satellites that were seen.
