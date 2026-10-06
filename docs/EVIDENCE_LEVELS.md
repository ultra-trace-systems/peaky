# Evidence levels — how much a committed formula is worth

This is the contract of **the evidence scale of peaky 0.10.0** (the release is
the constant `evidence.SCALE_RELEASE`; this phrase is the only place the docs
name it). The vocabulary lives in `peaky/assignment/levels/scale.py`, the
machinery in the `peaky/assignment/levels/` package, and the public entry
points in `peaky/assignment/evidence.py`. Where this document and the code
disagree, the code is the contract and the document has a bug.

**The rule behind the scale.** Every level-3 unlock must be evidence that has
been measured to carry formula-specific information. Everything else —
co-detection in two channels, a homologous ladder, a class list, another
source's sighting — is printed as a **tag**, with its measured base rate where
one exists, and is never itself a level unlock (§9). One designed exception to
"a tag moves nothing": two routes, a ladder or another source's partner can
ANCHOR the step-1 series exclusion (§5.4), which removes competitors and so can
lift a pair out of 5a.

## 1. What a level is

`tier` says whether the engine is willing to *print* a formula (Assigned) or
only to *offer* it (Candidate). It is a verdict on the assignment. A level
says what the evidence behind a committed formula is worth, on the scale a
reader of an identification paper already knows — Schymanski et al. (2014),
*Environ. Sci. Technol.* 48, 2097 — numbered **downward, 1 = best**, adapted
to chemical ionization, where there is no chromatography, no fragment
spectrum, no library, and the reagent is part of the ion.

**The unit** is a pooled pair `(neutral_formula, adduct)` of one *source* (a
batch's pooled files, or one file alone, §10). The level is stamped on every
committed M0 row of that pair. Each pair takes the FIRST outcome that applies,
in this order: **reagent, 5b, 5a, 4b (ion-only), 3c, 4a, 4b**.

| `evidence_level` | meaning (`LEVEL_MEANING`) |
|---|---|
| `1` | confirmed by an authentic standard in the same source and chemistry — defined, never assigned |
| `2` | matched to a library spectrum or MS2 standard — defined, never assigned |
| `3c` | ion established (§5), neutral / adduct split pinned (§6), and a **named** context-list entry names the neutral (§7) |
| `4a` | ion established, split pinned, and a **positive fact**: an own in-band isotope line of an element the neutral contains, the 15N label, or an NH4 adduct tracking its parent (§8) |
| `4b` | ion established; the split is open, or pinned without a positive fact, or the channel reads the ion only |
| `5a` | a competitor ion is left in the calibrated window (§5) |
| `5b` | rejected by a check (§4), or nothing could be enumerated or tested |
| `reagent` | a bucket, not a level: a reagent ion or reagent cluster (§4) |
| `NA` | a bucket, not a level: not assessed on this instrument class (§1.2) |

`LEVEL_ORDER = ["1", "2", "3c", "4a", "4b", "5a", "5b"]`; `LEVELS = ["3c",
"4a", "4b", "5a", "5b"]` (the ones that are assigned); `BUCKETS =
["reagent", "NA"]`. Level 3 has one rung, 3c: two routes, ladders and class
lists, which earlier drafts read as level-3 evidence, are tags here.

### 1.1 Claims — what a level lets a reader say

A reader of a result table asks something coarser than the level: may this
formula be reported as a compound? `evidence.claim_class(level)` reads the
level as one of four **claims** (`evidence.CLAIMS`), and two non-claim
buckets are reported beside them, never folded into them:

| claim | levels | what a reader may say (`CLAIM_MEANING`) |
|---|---|---|
| `identified` | 1, 2, 3c (`CLAIM_IDENTIFIED`) | the compound is named: ion established, split pinned and a named context-list entry |
| `neutral` | 4a (`CLAIM_NEUTRAL`) | the neutral is established, with no named identity |
| `ion` | 4b (`CLAIM_ION`) | the ion composition is established; the neutral / adduct split or the process stays open |
| `tentative` | 5a, 5b, no level | a competitor is left, the reading is rejected, or the row has no level |
| `reagent` (bucket) | `reagent` | a reagent ion or reagent cluster, not levelled |
| `not assessed` (bucket) | `NA` | not assessed on this instrument class |

`CLAIM_KEYS` = the four claims + `reagent` + `not assessed`, in that order;
`summarize_claims` returns all six keys, zeros kept.

- **A pure function of the level.** `claim_class` reads `evidence_level` and
  nothing else. NaN, `pd.NA`, `None`, `''`, 5a, 5b and any string it does not
  know read `tentative`; the literal `NA` reads `not assessed`.
- **Only a committed formula makes a claim.** Every committed M0 row carries
  one (per file from the file's own level, §10.1; on the merged ledger from
  the pooled level, §10.2). Isotope children, reagent ions, artifacts and
  unexplained peaks carry nothing (empty). A committed row with no level — a
  merged row whose reading no pooled pair holds, or a file the per-file stage
  could not level — reads `tentative`.
- **Tier and claim are separate verdicts.** The tier says whether the engine
  prints a formula; the claim says what the evidence lets a reader say of it.
  They are not nested, and the outputs show the two side by side. Neither is
  read off, corrected from or routed into the other.
- **Nothing upstream reads it.** Not the tiers, not the merge vote (whose
  class is its own, §13), not a check: the level and the claim change no ion,
  no tier and no assignment.

### 1.2 The instrument class: `NA` — not assessed

The scale is built for Orbitrap-class data. It is assessed only when the
source's width model resolves **≥ 50 000 at m/z 200** (`evidence.ORBITRAP_R200`,
`evidence.instrument`). A TOF-class source and a class-less one (no width
model: a ledger CSV, an old run, a single-sample run with
`--resolving-power none` or through the MCP `assign_sample` tool) read `NA` on
every committed M0 row, with the claim `not assessed`, the evidence string

```
NA · not assessed on this instrument class (width model R(200) = 9 652 < 50 000)
NA · not assessed on this instrument class (no width model: the instrument class is unknown)
```

and every other column empty. The check runs before any fact work: no gate,
no enumeration, no isotope probe runs on a TOF. A batch records the class as
`batch_summary.evidence_levels.instrument = {class, r200}`.

## 2. The columns and the evidence string

### 2.1 The columns

Every committed M0 row of a per-file ledger and every merged row carries the
eight columns of `evidence.COLUMNS`; every other per-file row carries them
empty.

| column | value |
|---|---|
| `evidence_level` | `3c` `4a` `4b` `5a` `5b`, or a bucket `reagent` / `NA` (§1); empty when the row has no level |
| `evidence` | the evidence string (§2.3): the level, the decisive facts and the tags, in one line |
| `would_lift` | what the pair lacks for the next level up (§2.4) |
| `competitors_left` | the competitor ions left in the calibrated window after the isotope and series exclusions, `"; "`-joined, full list; an isotope-line competitor reads `<label> line of <neutral> <adduct>`; empty when none |
| `tags` | the tag texts (§9), `" \| "`-joined; empty when none |
| `context` | the context-list entries that match the neutral (§7), named and class, `"; "`-joined: `reflist:<id> = <name>`, `registry:<family> = <label>` (named) or `reflist:<id>`, `registry:<family>` (class); empty when none |
| `context_source` | how the lists that matched were activated; when nothing matched, how the run's lists were (§2.5) |
| `claim` | `claim_class(evidence_level)` (§1.1) |

The columns of earlier development builds — `evidence_axes`, `level_reason`,
`n_plausible_structures` — are not written by any output.

### 2.2 The `NA` token

`NA` is written literally. **pandas reads the string `NA` as NaN by default**,
so a reader that must tell "not assessed" from "no level" either reads the
`claim` column (`not assessed` vs `tentative`) or reads the CSV with
`keep_default_na=False`; reading `evidence_level` alone with the default
parser folds `NA` into "no level".

A ledger levelled on a scale older than this one (it carries `evidence_axes` /
`level_reason`, or a letter this scale does not define) is read with **every**
letter as no level -- the letters both scales share (4a, 4b, 5a, 5b) too: an
old 4a is not a 4a of this scale. The report, the PDF, `publish` and
`scripts/ab_compare.py` show such rows as "no level (pre-<release> scale)",
re-read their claims (tentative) and say so in one sentence.

### 2.3 The evidence string

Segments are separated by `" · "` (U+00B7 with a space on each side).

- **3c / 4a / 4b:**
  `<level text> · ion: <ion text> · split: <PINNED|open> -- <how> · positive fact: <fact|none> · named list: <entries|none> · context source: <context_source> · tags: <tags|none>`
  - level text: `3c (split pinned + named context-list entry)`,
    `4a (split pinned + positive fact)` or `4b`. A 3c whose every named entry
    is mode-flagged (§7) adds
    ` [FLAG: every named entry's source mode contradicts this run (MODE CONTRADICTS); flag only, not a block]`.
- **5a / 5b / reagent:**
  `<5a|5b|reagent bucket> · <decision> · ion: <ion text> · split: <how> · context source: <...> · tags: <...>`
  (the ion segment is left out when there is no ion text). The decision is
  `reagent identity (<molecules>)`, `rejected: <reasons>`, `untestable` or
  `competitors left (<n>)`.
- **NA:** §1.2.
- **No level:** a per-file row the stage could not level reads
  `no level · the run's reagent profile is unknown (...)` or
  `no level · the file is uncalibrated (...) ... no calibrated window to enumerate competitors in`
  (§10.1); a pooled pair of a batch none of whose files is calibrated reads
  `no level · no file of the source is calibrated (...): no run window to enumerate competitors in`
  (§10.2); a merged row no pooled pair holds reads
  `no pooled pair: a batch-level re-read` (§10.2).

The **ion text** is `unique in the calibrated window`, or
`<n> competitor(s); isotopes exclude <k>; series excludes <s>; <m> left` (or
`none left`), then `; own lines in band: <lines>` when the committed reading
has matched lines, then ` [committed contradicted]` when its own lines refute
it; or `untestable (no ion composition / outside the enumerable space)`.

The **split text** (`how`) is one of the §6 outcomes: `label (the ion carries
the reagent's ^N)`, `only decomposition`, `halogen count`, `<k>
decompositions: <readings>`, `committed neutral not plausible (<why>); <k>
plausible: <readings>`, `ion-only channel (process open)`, `no ion
composition`, `label points to the cluster reading: <readings>`, `the ion
carries ^N but no plausible cluster reading`, or an amine-gate outcome
(§6.5). Bracketed suffixes disclose what decided it:
`[X+reagent isobar admitted over the context window: ...]`,
`[pinned only by the context window: ...]`,
`[NH4 admissibility rule: Y's uronium adduct ion is present under another reading: ...]`,
`[NH4 reading not kept by the amine gate: ...]`; a pinned split that a locked
side channel would open ends
`; side channels locked (<channels> would open it: <channel>: <neutral> <adduct>; ...)`.

### 2.4 `would_lift`

| level | text |
|---|---|
| 3c | `level 2 (MS2 / standards) is not automatic` |
| 4a | `3c needs a NAMED context-list entry naming the neutral` (+ ` (the class-list match is a tag only)` when a class list matched); 3c withheld (§7): `3c withheld: the named entry is on the list that rescued this reading -- an independent named list, or MS2 / standards` |
| 4b, pinned | `4a needs a positive fact (an own in-band isotope line of the neutral's elements, the 15N label, or NH4 tracking)` (+ `; 3c needs a named context-list entry` when none matched) |
| 4b, open | `split not pinned: <split text>` |
| 4b, ion-only | `ion-only channel: the neutral and the process stay open` |
| 5a | `competitors left: <first six>` (+ ` (+<n> more)`); the lowconf ceiling (§4): `engine confidence Low/Suspect in every file: not established (5a ceiling; a file at Good or High lifts it), not refuted` |
| 5b | `refuted: <reasons>`, or `nothing could be enumerated or tested` |
| reagent | `reagent ion / reagent cluster (not levelled)` |

### 2.5 `context_source`

One entry per list, `"; "`-joined:

- `reflist:<id>: always active`
- `reflist:<id>: keyword '<kw>' in the <batch|dataset|reagent label> name` (a
  list activated by a context tag: which keyword in which name unlocked it)
- `registry:<family>: pass-0 registry (<polarity>)`
- `reflist:<id>: activation not recorded` (a run made before the activation
  record existed)

When nothing matched the neutral, the field lists how the run's active lists
were activated, then `registry: pass-0 registry (<polarity>)`. The run's
record is `batch_summary["reflists_context"] = {tags, matched: {tag: [[field,
keyword], ...]}, active: [[id, version, how], ...]}`. Which lists activate is
unchanged by the record (`reflists.activate`).

### 2.6 `tables/evidence_levels.csv`

One row per pooled pair of a batch: `neutral_formula`, `adduct`, the eight
columns, then the step facts and the pair facts:

- the step facts: `tag_kinds` (the tag kinds, sorted, `|`-joined, §9),
  `split_pinned`, `split_how`, `positive_fact`, `named_list`,
  `named_mode_flag`, `window_only`, `window_isobar`, `chloride_open`,
  `nh4_gate`, `nh4_admissible`, `nh4_inadmissible`, `nh4_gate_detail`,
  `side_aliases`, `route_alias`, `anchor_kind` / `anchor_why` (what anchored
  the pair in the internal pass of §5.4, in the words an other-source partner
  prints: `two routes`, `ladder`, `listed` or `none`; the why adds `+ listed`
  when a context list also names a route / ladder anchor, and is the pass's
  own reason -- `split open`, `competitors left (2)`, ... -- when it is no
  anchor. The pass's raw labels are internal and never written),
  `n_left_inpass`, `n_series_excl`, `iterations`;
- step 1's facts: `n_competitors`, `n_excl_iso`, the committed reading's
  `committed_*` facts (contradicted, matched elements and lines, plausible as
  its own decomposition), `no_comp_info`, the space notes;
- the pair facts of §3.2 (`iso_veto`, `label_veto`, `lowconf`, `below`,
  `lead`, `tied`, `ion_only`, `n_files`, ...).

On a batch the scale does not assess (a TOF-class or class-less width model,
every pair `NA`) the table keeps the pair facts of §3.2 after the eight
columns; the step facts and step 1's facts are absent (nothing is enumerated,
gated or tested on such a run). A batch that committed nothing writes the
header alone.

`scripts/level_ledger.py --out` writes the same levels post hoc (§10.4), not
this table: one row per pair with the level, `claim`, `would_lift`,
`competitors_left` and its own decision facts (`n_left`, `rejected_by`,
`split_pinned`, `split_rule`, `positive_fact`, `named_entry`, `anchor_kind`,
`n_series_excl`, ...). It equals the table on those columns; it writes no
`evidence`, `tags`, `context` / `context_source` and none of the pair facts.

## 3. The facts the scale reads

### 3.1 The rows

The isotope probes need **every** per-file ledger row with height > 0
(unexplained, artifact and reagent rows included): a line is "present" when a
peak is there, whatever the engine called it. A batch reads its per-file
ledgers from the `per_file/<sid>_ledger.csv` files it has just written, with
pandas' default CSV parser, in sorted sample-id order, so the batch and a
post-hoc re-level of its run directory see byte-identical inputs. Text cells
treat `pd.NA`, NaN, `"nan"` and `"<NA>"` as empty.

### 3.2 The pair facts

Pooled over the source's files (a batch: `tied` and `lowconf` need ALL rows,
`below` and `lead` ANY):

| fact | from | what the scale does with it |
|---|---|---|
| `iso_veto` | the isotope checks, `tables/iso_checks.csv` (batch only; rules C / REQ / HIGH on the stamped time series) | step 0: rejected (5b) |
| `label_veto` | the 15N-twin facts of a labelled-nitrate batch, `tables/label_twins.csv` (batch only) | step 0: rejected (5b); its `alien` set (a 14N line the label shows is another ion) also stops a label pin |
| `lowconf` | every row's confidence `Low` / `Suspect` | step 0: the 5a ceiling (§4) |
| `below` | `below_assignability` (the assignment argues with itself) on any row | step 0: rejected (5b) by an implausible-chemistry setter; O≥11 on a mass-saturated window alone is a tag (§4) |
| `lead` / `lead_by` | `tentative_lead` (the proposal is unsupported, not contradicted) and its setter | **no level effect**: printed as the `lead` tag on every level |
| `tied` | the arbiter's near-tie | a `tie` tag on 5a; no level effect |
| `ion_only` | an ion-only channel (`[M]-.`) | 4b at best (§8) |
| `label_untie`, `upair`, `lead_lift` | rule K's untie, rule U's neutral pair (`tables/neutral_pairs.csv`), rule H's halogen lock | recorded in the facts table; read by no step of the scale |

The facts table also records, pooled, the facts the merge vote's private
class is computed from (§13; the vote reads them per file): `iso`,
`multiline` / `multiline_elements`, `carbon_ev`, `chan2`, `anchor`, `branch`,
`reagent_only_iso`, `res_ok` / `resolvability`, `saturated`, `known_fam`,
`corroborated`, `neutral_backed`. No step of the scale reads them.

On a single file there are no batch checks: `iso_veto` and `label_veto` are
False per file.

### 3.3 The run inputs

What a source needs beyond its ledgers (`RunInputs`): the reagent profile
name, the context, the polarity, the active reference lists and their
activation record, the width model, the per-file stats (height gates and the
degeneracy stage's calibration), and for a batch the stamped time series, the
merged ledger, the protected neutrals, the batch-check tables and the run's
`amine_r_min`.

### 3.4 The reagent halogen

The pair facts (§3.2: `reagent_halogen`, `reagent_only_iso`, rule H's lock
lift) and the merge vote's class (§13) read the reagent's halogen. It is the
halogen the run's **declared analyte channels** name (`evidence.channel_halogen`):
Br for a channel set holding `[M+Br]-` or `[M+HBr+Br]-` (a bromide or mixed
nitrate/bromide reagent), likewise Cl / I, none for a halogen-free set.
Declared, not opened: a side channel the server opens on a nitrate run
(`[M+Br2]-`) makes it no bromide reagent.

- **Per file:** `assign.run` takes it from the channels before the
  opportunistic ones join, hands it to the file's pair facts and records it in
  the file's stats (`per_file[i].reagent_halogen` in `batch_summary.json`); a
  batch's parent computes the file's vote class with that same halogen.
- **Pooled:** the batch's reagent profile's channels, recorded as
  `batch_summary["reagent_halogen"]`.
- **Post-hoc** (`evidence.source_from_run_dir`, `scripts/level_ledger.py`, a
  decoy arm in its main run's context): the run's recorded `reagent_halogen`,
  else the declared channels of the run's reagent profile (`reagent`, a run
  made before the record).

Where no channels are known it falls back to the halogen of the source's
**commonest cluster adduct** (`evidence.detect_reagent_halogen`,
`evidence.DETECT_HALOGEN`): a ledger CSV levelled alone without `--reagent`
(with it, the profile's channels, §10.4), a `--corroborate`
source re-levelled for the vote's cross set (`vote_cross_neutrals`), a
per-file stats record without the key. `[M+Br]-` → Br, `[M+NO3]-` → none: a
nitrate channel with two stray `[M+Br]-` rows against 346 `[M+NO3]-` is a
nitrate channel. The count is a guess on a mixed reagent: on the mixed
Br-/NO3- TOF it flipped to none when per-file `[M+NO3]-` M0 rows came to
outnumber `[M+Br]-` 3907 : 3847, and the reagent-81Br rule went off for the
whole run.

## 4. Step 0 — sort out

- **The reagent bucket.** The neutral is made only of the reagent's own
  molecules (at least one that is not water; water alone only on a cluster
  adduct) and is read on a reagent adduct:
  - nitrate (negative runs): HNO3, H^NO3, HBr, H2O on `[M-H]-`, `[M+NO3]-`,
    `[M+^NO3]-`, `[M+Br]-`;
  - uronium (positive runs): urea, NH3, H2O on `[M+H]+`,
    `[M+(CH4N2O)H]+`, `[M+NH4]+`.

  Evidence: `reagent bucket · reagent identity (urea + H2O) · ...`.
- **5b.** Any one of these rejects the pair:
  - `iso_veto` or `label_veto`;
  - the committed reading is contradicted by its own isotope lines (§5.3);
  - a `below_assignability` setter of implausible chemistry: the O-monster,
    the carbon cluster, an implausibly carbon-rich skeleton, an implausible
    ionization, an off-calibration residual fit, unconfirmed fluorine. The
    setter is read from the row's `tier_reason` / `commentary` text; a below
    flag no setter text explains is rejected too (`below: setter not found`).
  - "O≥11 on a mass-saturated window" alone is not a rejection: it is a tag
    (`O>=11 on a mass-saturated window (degeneracy statement, not a rejection)`).
- **Untestable (5b).** An ion with no composition, or a committed neutral
  outside the enumerable space that no widening admits: nothing could be
  enumerated or tested.
- **The lowconf ceiling (5a).** `lowconf` alone (every row Low / Suspect and no
  other rejection) does not reject (C47, 2026-10-05): the pair is levelled on
  its facts — untestable stays 5b, competitors left stay 5a — and reads at most
  **5a**, `engine confidence Low/Suspect in every file: not established (5a
  ceiling; a file at Good or High lifts it), not refuted` (tag kind `lowconf 5a
  ceiling`). It still anchors nothing: no member of the series exclusion, no
  route, no ladder. The flag tracked the calibration centre more than the
  chemistry (the C42 recentring alone moved 40 pairs out of it and 13 in); on
  the two Orbitrap runs the scale was validated on it was the only reason of
  86 (labelled nitrate) and 174 (uronium) 5b pairs.

## 5. Step 1 — the ion formula → else 5a

### 5.1 The calibrated window

Each file's window is **mu ± 3 sigma** (`K_SIGMA`) of its mass calibration.
The calibration is the degeneracy audit's own (`stats["degeneracy_cal"] =
{mu, sigma}`, persisted per file since the scale exists). A run made before
that refits it: the (mu, sigma) that reproduces the stored
`degeneracy_density` values on a grid around the recomputed calibration
(kept when it reproduces ≥ 99 % of them). An uncalibrated file in a batch
takes (median of `ppm_error - ppm_error_cal`, the run sigma), the run sigma
being the median of the per-file sigmas. The run window is the median of the
per-file windows. A source none of whose files is calibrated has no run
sigma and no window: its pairs get no level (§2.3, "No level"), and the
enumeration refuses a non-finite window.

### 5.2 The space and the competitors

The **space** is the run's element space exactly as the degeneracy audit
builds it: the context profile plus the contaminant families at least
`FAMILY_MIN_FILES` (2) of its files opened (`context.family_union`; every family
on a one-file source; a family one file alone opened is kept on
`RunContext.families_dropped`, C47 — before, one Candidate row in one file
opened `fluorinated` for every pair of a uronium run), plus the curated formulas (the pass-0 registry of
the polarity and the active reference lists), over the reagent profile's
adducts and the pair's own adduct. A committed neutral the space drops is
enumerated in the space widened to admit its class, so same-class competitors
are still counted instead of the pair reading "unique" vacuously.

The **competitors** of a pair are every ion of that space inside the window,
probed at the pooled m/z in the run window and at each file's m/z in that
file's window (the union), plus the engine's own alternatives (a tied
alternative always; an untied one inside a window). The committed ion and any
reading with the same element counts (the same ion read another way) are not
competitors. A committed reading's isotope line that predicts ≥ 1/2 of the
observed height in ≥ 1 file is a competitor too (an isotope-line competitor,
`<label> line of <neutral> <adduct>`).

### 5.3 The isotope line model and the tests

Each candidate's observable lines are predicted count-aware (fine structure
merged within the width model's FWHM) and probed in every file the pair was
seen in:

- a line is **testable** at ≥ 3× the file's height gate (`DET_X`); ≥ 5× for
  the light 15N / 18O lines and composite lines (`DET_X_MINOR`);
- it is **present (in band)** at 0.5–2× its predicted height (`BAND`); a line
  within 0.5 Da of the file's first / last peak is untestable (`SCAN_MARGIN`);
  a 37Cl-bearing line is probed at its exact position, the resolution taken
  into account (`CL37_NEIGHBOUR_FWHM`);
- a reading whose ion carries the label (`^N`) predicts the labelled
  reagent's 14N impurity line, n(15N) × (1 − purity) / purity of M0 at
  −0.997 Da (`14N (reagent impurity)`): only the engine's own twin or child
  (or the M0 of a reading the levels already refuted) there at ≥ 0.5× matches
  it; any other peak at ≥ 0.5× — an unrefuted reading's M0 or an ion the
  engine left unexplained (the same neutral's 14N cluster sits at exactly that
  m/z and can sit far above the impurity level) — leaves the line present
  (tested, neither bad nor matched); below 0.5× it is too low (there is no
  upper bound);
- the line efficiency of each element is measured on the run (the median
  seen/theory height of the committed readings' single-element lines,
  clipped to [0.25, 1]).

The tests, per candidate:

- **(a)** a line absent, too high or too low in ≥ 80 % (`ABSENT_FRAC`) of ≥ 3
  testable files (`NMIN_FILES`; 1 on a single-file source);
- **(b)** another candidate's observed in-band line this one predicts nothing
  comparable at (> 4× what it predicts there, `B_FACTOR`);
- **(c)** the observed carbon count (the batch's rule-C reading where it is
  testable, else the per-file 13C line) differs from the candidate's by more
  than max(1.5, 0.25 nC, 3 se) (`C_TOL_*`);
- **(k)** on a labelled-nitrate run, a reading that can only be a 14N
  `[M+NO3]-` cluster implies its 15N twin at +0.99703 Da (`D15N_TWIN`) at the
  file's twin ratio; absent or too low in ≥ 80 % of the testable files.

A candidate is **contradicted** when (a), (b), (c) or (k) holds. A
contradicted competitor is excluded. A contradicted committed reading is a
step-0 rejection. A line **matched in band** in ≥ 50 % (`MATCH_FRAC`) of ≥ 3
testable files gives its element as a matched element — the positive fact of
§8 when the neutral contains it.

### 5.4 Series exclusion and the outcome

**Series exclusion** (CH2 and CF2 only). When the pair sits in an ANCHORED
homologous series — a chain of committed pairs on the same adduct, neutral ± k
units, with ≥ 2 members at the right spacing that an internal pass reads as
route-anchored (seen through two of the reagent's own routes) — a competitor
is excluded when the competitor shifted by the same k units, on its own
adduct, is a committed ion at none of the anchors' spacings. An
other-source partner (§10.3) counts as a route for the anchoring. Isotope-line
competitors are never series-testable. It is iterated with the ladder to a
fixed point (≤ 20 passes). It stays in step 1
although its anchors are route tags: it is not a level-3 unlock, and without
it established ions fall to 5a (measured when the scale was built: 30 pairs,
0.41 % of the committed height, on a labelled-nitrate Orbitrap run; 8 pairs,
0.04 %, on a uronium Orbitrap run).

**Any competitor left: 5a.**

## 6. Step 2 — the split → pinned or open

An established ion can still be several neutrals: X on adduct a is the same
ion as X' on adduct a'. Step 2 asks whether the committed neutral / adduct
split is the only plausible one.

### 6.1 The grid and the side-channel lock

The **grid** is the run's adducts: the reagent profile's, every non-ion-only
adduct the run committed, `[M+NH4]+` beside `[M+H]+`, and the pair's own
adduct — minus the **locked side channels**.

- **Side channels are locked.** `evidence.SIDE_CHANNELS_LOCKED = True` is the
  single switch: formate, acetate, CO3-, O2-, O3-, NH4+ (except the uronium
  run's own channel, below), Na+ and chloride never enter the grid or the route
  classes (the uronium run's `[M+NH4]+` is in the grid but is not a route
  class). `evidence.UNLOCKED` (empty) is the hook for a side-channel sweep: a
  channel name there (e.g. `"formate"`) is unlocked alone. The batch records
  both (`batch_summary.evidence_levels.side_channels_locked`, `.unlocked`).
- **Chloride is locked too.** No run scored it, and the data do not show the
  channel (X − HCl partner presence LR 0.86 against formula neighbours).
- **Not side channels, so they stay in the grid:** unlabelled `[M+NO3]-` on a
  labelled-nitrate run; `[M+NH4]+` on a positive run, which is the uronium
  profile's own channel (the amine gate decides it, §6.5).
- **The locked-channel note.** Where a locked channel WOULD open a pinned
  split, the split text says so — e.g. `side channels locked (acetate, formate
  would open it: formate: C9H14O [M+CHO2]-; ...)`, or for chloride (the split
  re-run with `[M+Cl]-` added) `side channels locked (chloride would open it:
  C10H16O4 [M+Cl]-)` — with the tag `side channels locked` and the fact column
  `side_aliases` / `chloride_open`.

### 6.2 A plausible decomposition

The neutral of a decomposition must pass all of these: counts ≥ 0, not
empty; integer DBE ≥ 0 under Senior's rule and the O cap (a curated formula
is not exempt); inside the run's element space; no labelled reagent atom.

### 6.3 The context window and the reagent-isobar window rule

A decomposition can also fail ONLY the run's context window (the reagent
profile's H/C, O/C, N/C ... ranges, `context filter: ...`) while it passes
everything in §6.2. One rule decides such a window-only reading:

- **The context window does not exclude the X+reagent isobar.** When the
  committed ion is a reagent cluster of X (its adduct carries a reagent
  molecule R) and the window-only reading is X + R on the adduct without R,
  i.e. `adduct_delta(a) − adduct_delta(d) = R`, the reading stays live and the
  split is OPEN:
  - uronium: urea CH4N2O (X·urea·H+ = `[M+(CH4N2O)H]+` of X vs `[M+H]+` of
    X+urea);
  - nitrate: HNO3 (X·NO3- vs `[M-H]-` of X+HNO3), H^NO3, HBr.

  Split text `[X+reagent isobar admitted over the context window: ...]`, tag
  `X+reagent isobar admitted over the window`, column `window_isobar`.
- **The reverse direction is not this isobar.** When the window-only reading
  is X − R on the cluster (e.g. nitrophenol C6H5NO3 `[M-H]-` against
  C6H4·NO3-, a benzyne-type neutral), the window keeps pinning the split and
  the pin is disclosed: `[pinned only by the context window: ...]`, tag
  `pinned only by the context window`, column `window_only`.
- An NH4 reading the amine gate does not keep, or that the NH4 admissibility
  rule removes, is not a window-only reading: the gate or the rule removes it,
  not the window.

### 6.4 Pinned by

The rules in order; the first that decides wins.

1. **No ion composition** → open. **An ion-only channel** → open (`ion-only
   channel (process open)`).
2. **The 15N label** (a labelled-nitrate run): the ion carries the reagent's
   ^N on a ^N adduct, with no `label_veto` and not marked alien by the 15N-twin
   facts → PINNED (`label (...)`), and the label is a positive fact. An ^N ion on an
   unlabelled adduct → open (`label points to the cluster reading: ...`).
3. **The amine gate** decides on a positive run with `[M+NH4]+` in the grid
   (§6.5).
4. **Only decomposition**: exactly one plausible decomposition, and it is the
   committed one (after the amine gate, the NH4 admissibility rule and the
   window rule have decided which readings stay) → PINNED.
5. **Halogen count**: a Br / Cl count no grid adduct can supply, whose line is
   matched in band → PINNED. As coded this rule is subsumed by rule 4 (every
   decomposition then carries the element, so "only decomposition" has
   already decided); it is kept for the record.
6. **Committed neutral not plausible** → open.
7. **k decompositions** → open (`<k> decompositions: <readings>`).

### 6.5 The amine gate and the NH4 admissibility rule

On a uronium run `[M+NH4]+` is the reagent's own channel: X `[M+H]+` and Y =
X − NH3 `[M+NH4]+` are the same ion. **The engine's rule**: NH4+ is assigned
only when there is a uronium parent (`cleanup.prefer_amine_over_ammonium`,
applied by the batch at merge). The scale calls the gate's own decision on
every `[M+NH4]+` reading in the grid, in the engine's order:

1. **Si in Y**: kept.
2. **Y protected** (a reflist-rescue, `known:` or `certified:` method in any
   per-file ledger): kept.
3. **The verdict.** The `[M+NH4]+` trace tracks the best of Y's own `[M+H]+` /
   urea-cluster parent traces with r ≥ 0.6 (`amine_r_min`; the run's own value
   when it set one) over ≥ 12 two-hour bins: kept, and **NH4 adduct tracks its
   parent** is a positive fact. (A confound: a protonated amine that loses NH3
   in the source also "tracks".)
4. **Otherwise** the engine re-reads the ion as the amine Y + NH3 `[M+H]+`,
   unless that amine is valence-impossible: the NH4 reading is then kept,
   unconfirmed.

Without a time series (a single file, a decoy arm) the gate takes its
no-time-series path: steps 1, 2 and the valence exception only.

**The NH4 admissibility rule.** For a committed `[M+H]+` (an amine reading X),
an NH4 reading Y `[M+NH4]+` is admissible only if Y has a uronium adduct: the
**ion** of Y `[M+H]+` or of Y's urea cluster `[Y+(CH4N2O)H]+` is present in the
run — by ion composition, whatever neutral name it was committed under, on an
M0 row of any per-file ledger or a row of the merged ledger. Only an ion inside
the run's scan (the first / last peak over the per-file ledgers) counts as
missing; when both uronium ions of Y lie outside the scan, the reading stays
admissible. An inadmissible reading leaves the decomposition set before the
gate (columns `nh4_inadmissible`, tag `NH4 reading inadmissible`). An
uncommitted peak at Y's ion m/z, or a committed ion of another composition a
few ppm away, is not presence. When Y is admissible only through its ion under
another reading (typically Y·urea committed as `[M+H]+` of Y+urea), the split
text and the tag `NH4 reading admissible via Y's ion under another reading`
say where the ion was found; the gate's own parent lookup is by name, so its
verdict there can still be "parent channels absent", and the split is open.

How the gate's result sets the split:

| committed reading | NH4 reading | split |
|---|---|---|
| X `[M+NH4]+` | kept: tracks its parent | PINNED, `NH4 adduct tracks its parent (r, bins)`; a 4a positive fact |
| X `[M+NH4]+` | kept by an exception (Si, protected, amine impossible) | PINNED, the exception named; no positive fact |
| X `[M+NH4]+` | not kept | open: `amine default (NH4 reading unconfirmed: <why>; ...)` |
| X `[M+H]+` | Y inadmissible | Y leaves the set; the usual rules decide (typically `NH4 reading inadmissible: no uronium adduct of Y (...); only decomposition`) |
| X `[M+H]+` | Y admissible, not kept | open: `amine default (NH4 reading unconfirmed: Y -- <why>)` |
| X `[M+H]+` | Y admissible, kept | open: `NH4 reading kept by the amine gate: Y ... -- the committed [M+H]+ (amine) reading is contested` |
| any other ion | not kept | the NH4 reading leaves the set (noted in the split text); the usual rules decide |
| any other ion | kept | the NH4 reading stays a live decomposition |

The amine default stays OPEN: the gate's fall-back to the amine is a default,
not a confirmation (the engine caps it at Candidate).

## 7. Step 3 — level 3 = 3c only

**3c** = ion established (step 1) AND split pinned (step 2) AND a **named**
context-list entry matches the neutral (by its element counts, on any adduct;
a labelled neutral never matches). 3c does not need a positive fact.

**The list that rescued the reading cannot certify it** (C47): a pair that is a
tentative lead with setter `reflist_dim` in any file — the reference list
rescued the dim reading — does not reach 3c through that list's named entry.
It takes the level its other facts give (4a / 4b), carries the tag `3c
withheld: the named entry is on the list that rescued this reading (lead:
reflist_dim) ...` (tag kind `3c withheld (list rescued the lead)`) and, at 4a,
`would_lift` = `3c withheld: ... an independent named list, or MS2 /
standards`. The named entry is still printed in `named_list` and `context`.

A run's context lists are its active reference lists and the pass-0 registry
of its polarity and context.

- **Named** (can make 3c): a reference-list entry with a `name` (on a list
  that has names at all, its `origin` where the name is empty, e.g.
  "Trifluoroacetic acid, TFA (anion)"); a pass-0 registry entry of a
  compound-scope family (e.g. nitroaromatic: nitrophenol; organophosphate:
  TEP; cyclosiloxane: D3).
- **Class** (a tag only): a formula-only list (no names), and a class-scope
  registry family (a formula loop, e.g. perfluoroacid, chlorinated paraffin).

**The mode / ion-form flag — printed, never blocking.** A named entry whose
recorded source mode or ion form contradicts or differs from the pair is
flagged (`named_mode_flag`); a mode contradiction is a flag, and 3c stands:

- `MODE CONTRADICTS: entry recorded in ESI±, this run is ESI∓` when the entry's
  `modes` lack the run's ESI polarity;
- `MODE CONTRADICTS: list is <polarity> (...)` only when the entry has no
  `modes` and the list's polarity differs;
- `ion form: the entry is an anion` on a positive run, `... a cation` on a
  negative run;
- `ion form differs: entry = the anion [M-H]-, this pair = <adduct>` when an
  anion entry is read on any other adduct on a negative run (it fires on
  TFA·NO3-, which is correct chemistry);
- pass-0 registry hits are never flagged (the registry is read per polarity).

A 3c with a mode contradiction carries the tag `FLAG, named entry's source
mode contradicts this run: <name> (...) -- flag only, does not block 3c` and
the tag kind `3c name: mode contradicts`; when EVERY named entry is so
flagged, the level text carries the FLAG suffix of §2.3.

## 8. Step 4 — 4a / 4b

- **4a:** split pinned AND a positive fact:
  - an own in-band isotope line of an element the NEUTRAL contains (matched in
    ≥ 50 % of ≥ 3 testable files, §5.3) — a line of an element only the
    reagent brings (the 81Br of a Br-free neutral's bromide cluster) does not
    count;
  - the 15N label;
  - "NH4 adduct tracks its parent" (§6.5).
- **4b:** everything else that passed step 1: the split open, pinned without
  a positive fact, or an ion-only channel.

The `positive_fact` column names it, e.g. `own in-band isotope line(s)
13C|18O (C,O of the neutral)`.

## 9. Tags — printed, never a level unlock

Each tag is a tag because its measured information content is at or near
chance. No tag unlocks a level; the route, ladder and partner tags can still
move one indirectly, as anchors of the step-1 series exclusion (§5.4: an
excluded competitor no longer keeps the pair at 5a). The base rates below were measured when the scale was built on a
labelled-nitrate Orbitrap run and a uronium Orbitrap run (CHO neutrals X on
the base route; the null is formula neighbours X ± Δ, Δ = ±CH2, ±O, ±H2,
±C2H4, ±CO, ±H2O, ±O2, +CH2O). The route and partner tags print the run's own
base rate, recomputed per run: `base rate <x> % vs <y> % for formula neighbours
X±delta (LR <lr>, <n> CHO neutrals)`.

| tag (tag kind) | as computed | measured base rate | what would make it a level |
|---|---|---|---|
| `two routes` (routes) | the reagent's own route pair (`[M-H]-` + nitrate cluster; `[M+H]+` + urea cluster) seen in the same file in ≥ 2 files; not rejected, not alias-vetoed, not another ion's isotope line; printed with the route co-variation r | 16.1 % vs 15.2 % (LR 1.06, 409 X) labelled nitrate; 73.8 % vs 57.5 % (LR 1.28, 367 X) uronium | a formula-specific test at LR ≥ ~10 against this null, replicated on an independent batch |
| `other-source partner` (other source) | the same neutral levelled through a route class this run lacks, in another Orbitrap-class run (§10.3); counted only when this ion's split is pinned, else `other-source route (...) not counted: this ion's own split is open` | 28.6 % vs 24.8 % (LR 1.15, 455 X); 22.9 % vs 20.6 % (LR 1.11, 345 X) | as for two routes, and the partner must tell X from the receiving run's side-channel alias |
| `ladder` (ladder) | a CH2 / CF2 chain on the same adduct with ≥ 2 route-tagged anchors, pinned split, ≥ 2 files; else `chain: ... (not anchored / split open / seen in < 2 files)` | its anchors are route tags, so it inherits LR ~1 | anchors that are themselves validated levels (3c), plus a measured ladder null |
| `class list` (class list) | a formula-only or class-scope list match | not measured: a formula-only entry restates the formula | an enrichment test of list members vs list-neighbour formulas, or MS2 / standards |
| `named list (not used: ...)` (named list, not 3c) | a named entry on a pair that is not 3c (split open / ion not established) | — | — |
| `side channels locked` (side channels locked (split)) | a locked channel would have opened this pinned split (§6.1) | n/a: a statement about the lock | a side-channel sweep that unlocks a channel the data show |
| (chloride would open the split (locked)) | the chloride subset of the above | chloride partner presence LR 0.86 | as for side channels |
| `pinned only by the context window` | only a context-window exclusion keeps the split pinned (§6.3) | n/a: a window, not a measurement | — |
| `X+reagent isobar admitted over the window` | the reagent-isobar window rule kept the X+R reading live (§6.3) | n/a: a rule | — |
| `FLAG, named entry's source mode contradicts this run` (3c name: mode contradicts) | §7 | n/a: a flag | — |
| `NH4 reading inadmissible` | the NH4 admissibility rule removed Y (§6.5) | n/a: a rule | — |
| `NH4 reading admissible (NH4 admissibility rule): ... under another reading` | §6.5 | n/a: a rule | — |
| `amine default` / `NH4 reading kept by the amine gate` (and the kind `NH4 reading dropped by the gate`) | the gate's decision on the admissible NH4 readings | gate tracking: 73.8 % of 191 Y track their parent vs 59.0 % for neighbours' parents (LR 1.25) | a brightness-matched tracking null at LR ≥ ~8, or a reagent-NH3 modulation test |
| `two routes alias-vetoed` (routes alias-vetoed) | positive runs: the NH4+ alias Y = X − NH3 is a committed, plausible neutral and the gate keeps its NH4 reading | — | — |
| formate alias text (route alias locked) | negative runs: `formate alias Y (X - CH2O2) would explain both route ions (side channel locked: not applied)` | — | — |
| `lead: <setter> (a tentative lead: not answered by an own isotope line)` | the pair carries `tentative_lead` in any file; printed on every level | — | a `reflist_dim` lead withholds 3c from that list (§7) |
| `3c withheld: ...` (3c withheld (list rescued the lead)) | a pinned, established pair with a named entry on the list that rescued it (§7) | n/a: a rule | an independent named list, or MS2 / standards |
| `engine confidence Low/Suspect in every file: 5a ceiling ...` (lowconf 5a ceiling) | `lowconf` alone (§4) | n/a: a rule | a file at Good or High |
| `tie`, `suspect match: <reading> [<list>]` | on 5a: the arbiter's near-tie; exactly one candidate in the window is on a list | — | — |
| `series exclusion xN`, `O>=11 on a mass-saturated window (...)`, `list favours <X>` | step-1 notes; `list favours` on an open split where exactly one plausible decomposition is listed | — | — |

Co-variation everywhere else (route r, possible in-source fragments) is
printed only where the tag says so; it never moves a level.

## 10. Where it runs

One pipeline, three callers (`evidence.level_source` on a `Source`):

### 10.1 Per file — the `evidence` stage of `assign.run`

`_Stage("evidence", …)` runs after every tier and demote stage, the reflist
rescue, the ion-only stage and the final envelope sweep, before `timeseries`.
`evidence.apply_levels` levels the file **alone**, in "adapted" mode: every
file-count minimum 1, no time series (the amine gate's no-time-series path),
no merged ledger (the file's own ledger is the NH4 rule's ion index and the
protected set), no batch checks, no partners. The window is the degeneracy
stage's calibration of that file (else the refit of §5.1); the height gate is
the run's resolved gate (else the noise edge).

- The instrument class comes from the run's width model: `peaky assign`
  measures one from the sample's raw profile by default (`--resolving-power
  auto`) or takes a declared R; a batch passes its own. With no width model
  (`--resolving-power none`, a failed measurement, the MCP `assign_sample`
  tool) the file reads `NA`.
- A file whose reagent profile is unknown (its adducts match no registered
  profile) gets no level: evidence `no level · the run's reagent profile is
  unknown ...`.
- An uncalibrated file levelled alone has no window (no run sigma to borrow):
  no level, evidence `no level · the file is uncalibrated ...`, claim
  tentative.
- The stage summary is `{levels, claims, n_levelled, n_pairs, instrument}`;
  the run's log line and `peaky assign` print the claim tally.
- `--corroborate` on single-sample `peaky assign` is accepted, logged and
  ignored: a single sample has no merge vote and no partners.

### 10.2 Batch — the pooled levels, stamped on the merged ledger

After the batch checks (`neutral_pairs`, `label_twins`, `iso_checks`),
`assign_batch.run` levels the per-file ledgers of the assigned files (cover and
residual stages) as ONE source with every file-count minimum 3
(`evidence.level_batch`), with the stamped time series, the merged ledger, the
protected neutrals, the per-file gates and calibrations, the batch checks and
the `--corroborate` partners (§10.3). `evidence.stamp_merged` joins the result
onto the merged ledger by `(neutral_formula, adduct)` (one-to-one); a merged
row no pooled pair holds (a batch-level re-read) gets no level, claim
tentative and the evidence `no pooled pair: a batch-level re-read`. The
per-file ledgers keep their own per-file levels (§10.1), so a merged row can
read 4a where its per-file rows read 4b, and the other way round. A batch
none of whose Orbitrap-class files is calibrated (a one-file batch of an
uncalibrated file included) has no run window: every pooled pair gets no
level, claim tentative and the evidence `no level · no file of the source is
calibrated ...`; one calibrated file is enough to lend the others its sigma.

`batch_summary.json` records:

- `evidence_levels`: `scale` (`"peaky <release>"`), `instrument` {class,
  r200}, `pooled` (one count per pair over LEVELS + BUCKETS), `merged` (per
  merged row), `per_stage` (merged rows by `cover` / `residual`), `n_pairs`,
  `n_unstamped`, `side_channels_locked`, `unlocked`, `n_corroborate` and
  `cross_source` (the merge vote's cross set, §13), `partners` ({source: n
  neutrals}), `amine_r_min`, and the batch checks' funnels `neutral_pairs`,
  `label_twins`, `iso_checks`;
- `claims`: `merged`, `pooled`, `per_stage`, `by_tier` (ion-only rows under
  their own `ion-only` key) — each over the six `CLAIM_KEYS`, zeros kept — and
  `n_unlevelled`;
- `reflists_context` (§2.5), `amine_r_min` and `reagent_halogen` (§3.4) at
  the top level; `ion_only.merged_levels`.

`tables/evidence_levels.csv` is §2.6.

### 10.3 Other-source partners and `--corroborate`

`peaky batch ... --corroborate <source>` (repeatable) feeds two things:

- **the merge vote's evidence class** (§13), exactly as before the scale;
- **the other-source partner tag**, only from a source that is an
  Orbitrap-class batch run directory: it is levelled once by the scale with no
  partners of its own, and `evidence.partners_from` takes its pairs whose
  internal pass reads them through two routes, a ladder or a listed pin (not
  ion-only, on a route class, minus the locked side-channel classes). The
  partner text names the source by its run directory name.

A TOF-class or class-less run directory, a ledger CSV or a merged-only source
gives no partners (logged), and neither does an Orbitrap-class run directory
none of whose files is calibrated: it has no run window, so its pairs carry no
level (§10.2) and none can anchor a partner (logged with that reason, by the
batch and by `scripts/level_ledger.py` alike). Nor does a run directory whose
batch summary names a reagent profile or context this process does not know
(a run made under a `--reagent-config` profile, say), or none: it cannot be
levelled here, so it is logged and skipped, never a stop of the batch. A partner
is a tag and never unlocks a level, but in the internal pass it counts as a
route: it can anchor a series exclusion (§5.4), and so lift a pair out of 5a
(one partner pair can anchor a whole CF2 ladder of an acid series).

### 10.4 The post-hoc script and decoy arms

`scripts/level_ledger.py <run dir>... [--corroborate <run dir>...] [--mode
run|adapted|strict] [--main <run dir>] [--resolving-power R --reagent NAME
[--context NAME] [--window MU,SIGMA]] [--out levels.csv] [--vector]`
re-levels any run directory after the fact — old runs included (the window
refit of §5.1) — with the shared fact layer (`evidence.source_from_run_dir`)
and its own, independently written decision layer (step 0 order, step 1
outcome, the split outcome table, the level, the `would_lift` texts). The
in-core decision must equal it row for row. `--vector` prints the pair count
per level and bucket and each level's share of the committed M0 height.

A single ledger CSV is a one-file "adapted" source with no width model (`NA`)
unless `--resolving-power` (R at m/z 200) and `--reagent` (the profile) are
given. It is then levelled as the per-file stage of a single-sample `peaky
assign` would level it: the profile's context (`--context` overrides it), the
reference lists that context and the profile label activate (the
always-active lists included), the halogen of the profile's declared
channels, and the window `--window` gives (the file's `degeneracy_cal`), else
refitted from its degeneracy counts. What a lone CSV cannot know stays at the
stage's own fallback: the height gate (with no gate or noise edge recorded,
the 1st-percentile height of the file's peaks) and any list a batch or
dataset NAME activated by keyword. A file of a batch whose lists were
activated by such a name can therefore differ from its stored per-file level
by those lists' entries.
`--main <run dir>` levels the CSV in that run's context instead, as a decoy
arm (below) — the pooled run's window, gate and models, not the per-file
stage's.

A **decoy arm** — one ledger levelled inside a main run's context (its window,
gate, line efficiency, carbon-count model, 15N twin ratio, grid, lists and
scan) — is `source_from_run_dir(path, main=<run>, mode="adapted"|"strict")`:
"adapted" sets every file-count minimum to 1, "strict" keeps the main run's.
Arms have no time series and no partners.

## 11. Constants

All named, one place each (`peaky/assignment/levels/*`; the switch and the
release in `evidence.py`).

| constant | value | where |
|---|---|---|
| `ORBITRAP_R200` | 50 000 (R at m/z 200) | the instrument class (§1.2) |
| `K_SIGMA` | 3 | the calibrated half-window, in sigma |
| `NMIN_FILES` / `NMIN_FILES_ARM` | 3 / 1 | testable files for an isotope test (batch / one file) |
| `DET_X` / `DET_X_MINOR` | 3 / 5 × the height gate | a line is testable |
| `BAND` | 0.5–2 × predicted | a line is present |
| `ABSENT_FRAC` | 0.8 | tests (a), (b), (k) |
| `MATCH_FRAC` | 0.5 | a line matched in band |
| `C_TOL_ABS`, `C_TOL_REL`, `C_TOL_SE` | 1.5, 0.25, 3 | test (c) |
| `B_FACTOR`, `B_MIN_RATIO` | 4, 0.04 | test (b) |
| `D15N_TWIN` | 0.99703 Da | test (k) |
| `SCAN_MARGIN` | 0.5 Da | a line at the scan edge is untestable |
| `CL37_NEIGHBOUR_FWHM` | 1 FWHM | the exact 37Cl probe |
| `ISOLINE_LIST_X` | 2 | an isotope-line competitor (≥ 1/2 of the observed height) |
| `ORBI_MIN_PPM`, `ORBI_K` | 1 ppm, 4 | the isotope position window without a fit |
| `TWIN_Q`, `TWIN_MIN_PAIRS`, `TWIN_BRIGHT_X` | 0.10, 3, 10 | a file's 15N twin ratio |
| `ROUTE_COFILES` | 2 (1 on one file) | two routes: co-seen in ≥ 2 files |
| `LADDER_MIN_FILES`, `CROSS_MIN_FILES`, `HOMO_MIN_ANCHORS` | 2, 2, 2 | ladder, partner, anchored chain |
| `FAMILY_MIN_FILES` | 2 (1 on one file) | a contaminant family opens the run's space (§5.2) |
| `LOWCONF_REASON`, `LOWCONF_CEILING_WHY`, `LEAD_RESCUE_SETTER`, `LIFT_3C_WITHHELD` | §4, §7 | the lowconf ceiling; 3c withheld |
| `LADDER_UNITS` | CH2, CF2 | ladders and series exclusion |
| `MAX_PASSES` | 20 | series exclusion / ladder iteration |
| `AMINE_R_MIN`, `AMINE_R_REJECT`, `AMINE_MIN_OVERLAP` | 0.6 (the run's `amine_r_min` when set), 0.2, 12 two-hour bins | the amine gate |
| `SIDE_CHANNELS_LOCKED`, `UNLOCKED` | True, empty | the side-channel lock and its sweep hook |
| `REAGENT_MOL`, `REAGENT_ADDUCTS`, `CLUSTER_ADDUCTS` | §4 | the reagent bucket |
| `REAGENT_MOLS` | urea; HNO3, H^NO3, HBr | the reagent-isobar window rule |
| `OWN_REAGENT_CHANNEL` | `[M+NH4]+` on positive runs | §6.1 |
| `BELOW_SETTERS`, `O11_SETTER` | §4 | step 0 |

## 12. Known limits

- **The lock holds most of the 4a height on a labelled-nitrate run.** When the
  scale was built, 4a held 81 % of a labelled-nitrate Orbitrap run's committed
  M0 height, and 76 % of the height is 4a only because the side channels are
  locked: a formate, acetate or chloride reading would open the split. This
  includes the C11 `[M-H]-` family, which is known to track formate/acetate.
  The locked-channel note says so on every such row. A side-channel sweep
  (unlock one channel the data show, through `UNLOCKED`) decides how much of
  it is real.
- **4b on a uronium run is the urea isobar.** For most urea-cluster ions, the
  cluster reading X and the protonated X+urea reading are the same ion, and
  nothing in the scale can choose between them (the reagent-isobar window rule
  keeps the split open uniformly for every X). About 70 % of a uronium
  Orbitrap run's committed height sat at 4b for this reason. The engine's own
  "reagent-N isobar unresolved" note says the same. Two routes could split it
  once validated; today they are a tag (LR 1.28).
- **The lead / list circularity, half closed.** A reference list that rescued
  a dim reading (`reflist_dim`) no longer certifies it at 3c (§7, C47); a
  `known:` registry commit still reaches 3c through its own registry family.
  A tentative lead has no level effect; the `lead` tag is printed on every
  level so the reader sees it.
- **The run-level family union needs two files** (§5.2, C47). A family one
  file alone opened is recorded (`families_dropped`) and admits no competitor;
  a batch of one file keeps every family it opened.
- **NH4 tracking as a positive fact** has a measured LR of 1.25 (two-hour
  tracking is mostly family-wide co-variation). It fired on no pair in the runs
  the scale was built on, which committed no `[M+NH4]+` row.
- **No informative decoy test of 3c / 4a on a uronium run.** Its wrong-adduct
  decoys are `[M+NH4]+` readings, which take the gate's no-time-series path
  and so cannot reach 4a or 3c by construction. On a labelled-nitrate run no
  shift or wrong-adduct decoy reached 3c or 4a.
- **Keyed by polarity, not by the reagent profile.** The reagent bucket, the
  reagent-isobar molecules, the own `[M+NH4]+` channel and the amine gate are
  written for nitrate (negative) and uronium (positive) runs. A run with
  another reagent of the same polarity is levelled with those rules.
- **Below setters are parsed from free text** (`tier_reason` / `commentary`).
  A tier-text edit can turn a setter into `below: setter not found`, which
  rejects.
- **Smaller as-built choices,** kept for parity with the reference the scale
  was validated against: the halogen-count pin is subsumed by "only
  decomposition" (§6.4); the chloride lock blocks only an addition (a grid that
  already holds `[M+Cl]-` keeps it); a committed `[M+NH4]+` is decided by the
  gate alone (the admissibility rule applies to `[M+H]+` readings); the level
  text's mode FLAG needs EVERY named entry flagged while the tag needs ANY; the
  mode flags are substring tests on name and origin.
- **One file alone** cannot be levelled without a calibration, and a
  single-sample run without a width model reads `NA` (§10.1).

## 13. The merge vote's class

The batch merge ranks a cluster's ions by the **class** of their per-file
readings before the file count (`assign_batch._vote`, [MERGE.md](MERGE.md)
§3 step 4). That class is NOT the evidence level and is never printed as one.

- It is computed privately in `evidence.py`, by the decision peaky used before
  the scale (`evidence.vote_classes` over `EV.trim(ledger)`, the
  `--corroborate` cross set, the width model and the reagent halogen the
  file's own run read, §3.4), from per-file facts. The vote
  is the only path from evidence to assignments, so the scale moves no
  assignment.
- **2 = neutral backed** (the formula is confirmed and the neutral backed by
  two axes or a corroborating source), **1 = formula confirmed**, **0 =
  unconfirmed** (`VOTE_CLASS_TEXT`, `VOTE_CLASS_MEANING`).
- The vote note in the merged `tier_reason` prints the class:
  `evidence outranks the count: kept X (formula confirmed in 2 of 11 files)
  over the 5-file Y (unconfirmed)`. `tables/jitter.csv` carries each file's
  reading's `vote_class` (0 / 1 / 2).
- `--corroborate` feeds the class's cross set as it always did: the neutrals a
  source pins on its own (`vote_cross_neutrals`; a source with per-file ledgers
  is re-levelled by the vote's decision with no cross set). A merged-only
  source written on the scale (it has `evidence_level` but no per-file facts)
  is refused: name its run directory.
- Measured when the scale was wired in: the class equals the class the
  pre-scale code stored, on every M0 row of three regression runs (26 278
  rows, 0 mismatches).
