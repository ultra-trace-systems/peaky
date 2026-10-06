# The scorecard — what a run assigned, how good it is, what it missed

`scripts/scorecard.py` is the per-session overview of a `peaky batch` run. It
answers the same three questions every time, so two sessions can be compared
on the same page: **what was assigned**, **how good is it**, **what was
missed**. It reads a finished run dir only — it never contacts a server — and
appends one row per run to a scoreboard so the delta between sessions is
visible at once.

```bash
python scripts/scorecard.py <run_dir>... [--levels levels.csv] [--other <run_dir>]
    [--other-instrument <run_dir> --overlap 'start,end' --mask 'start,end' --floor-cps 10 --floor-share 0.8]
    [--decoy none|shift|ppm|adducts|both --decoy-offset 0.35 --decoy-ppm=9,-9 --decoy-files 1] [--decoy-ledgers DIR]
    [--rosters DIR] [--out ~/peaky-output/scoreboard]
```

A run dir is the timestamped folder that holds `merged_ledger.csv`, or the
`--out-dir` that holds exactly one. Everything the card needs is in the run:
the merged ledger, `per_file/*_ledger.csv`, `per_file/_batch_ts.parquet`,
`batch_summary.json` and `run_manifest.json`.

## Outputs

| file | what |
|---|---|
| `<out>/<run name>/SCORECARD.md` | the card: §0 the claim, then six numbered sections (below) |
| `<out>/<run name>/scorecard.json` | the same, as data — every table as records |
| `<out>/<run name>/decoy/<file>__<arm>.csv.gz` | each decoy arm's engine ledger (`control`, `shift`, one per ppm shift -- `ppmp9`, `ppmm9` -- and `adducts`), kept for a re-count |
| `<out>/<run name>/decoy/manifest.json` | what those ledgers were made with: mode, offset, ppm shifts (`ppm_k`) and the ones skipped per file with why (`ppm_skipped`), files, adduct sets, per file the scoring its arms were judged at (`inherited` / `class-fallback`) and the inherited width, offset, window, floor and anchors (`scoring_detail`), per file the mass calibration its arms ran at (`calibration`: `control` / `own`), engine code |
| `<out>/scoreboard.jsonl` | one row per run, appended; the board's memory |
| `<out>/SCOREBOARD.md` | the claims table first, then every channel's latest row with its delta to the row before (`## All metrics`) |
| `<out>/scoreboard.html` | the page source of "Peaky Scoreboard": the claims table, the channel table and one tabbed panel per run |

A **channel** is `<batch slug>|<reagent>|<path>` (`cover` or `trace-first`),
so a path change is a different channel and its comparison belongs to
`scripts/ab_compare.py`; the delta on the board is always the same channel
against its previous row.

## 0. The claim

Every committed reading carries, beside its tier, the **claim** its evidence
level supports on the evidence scale (`evidence.claim_class`,
docs/EVIDENCE_LEVELS.md §1.1), and two buckets reported beside the claims:

| claim | levels | what a reader may say |
|---|---|---|
| identified | 1, 2, 3c | the compound is named: ion established, split pinned and a named context-list entry |
| neutral | 4a | the neutral is established among the run's declared reagent channels, with no named identity ([EVIDENCE_LEVELS.md](EVIDENCE_LEVELS.md) §1.1) |
| ion | 4b | the ion composition is established; the neutral / adduct split or the process stays open |
| tentative | 5a, 5b, no level | a competitor is left, the reading is rejected, or the row has no level |
| reagent (bucket) | `reagent` | a reagent ion or reagent cluster, not levelled |
| not assessed (bucket) | `NA` | not assessed on this instrument class (a TOF-class or class-less run) |

The card's claim block is schema 2 (`CLAIMS_SCHEMA = 2`: the six keys above,
zeros kept). Board rows written under schema 1 (three claims on the pre-release
level scale) still render beside new ones; their deltas against a schema-2 row
read None.

The card always derives the claim from `evidence_level` (a run made before the
`claim` column existed carries none; one made before the in-core level takes
it from the in-process levels join). Where a run did write a `claim` column,
the rows whose stored claim differs from the derived one are counted as
`stamp_mismatch` — a stamping defect, expected 0. Only a committed M0 reading
makes a claim; every merged row has one (a row with no level reads tentative).

- **Merged rows per claim**, and the **tier × claim crosstab** with three tier
  rows: Assigned, Candidate and ion-only (the `ion_only_of` rows, 4b at best,
  kept out of the Candidate row as the headline keeps them), by rows and by
  signal.
- **Signal per claim.** The committed signal is every per-file M0 reading
  (`per_file/*_ledger.csv`, `role == 'M0'`, `height`) over all assigned files.
  Each reading is joined on (neutral_formula, adduct) to the merged ledger and
  takes the MERGED claim and tier; the denominator is ALL per-file M0 height.
  A reading no merged row carries (the merge vote kept another reading of that
  ion) is its own **`unmatched`** bucket — never folded into tentative. The
  Assigned-only shares sit beside the all-tier ones.
- **Disagreeing rows.** Tier and claim are separate verdicts and are not
  nested: a Candidate can be identified (the vote was split, the evidence is
  good) and an Assigned row can be tentative. The card counts both kinds and
  lists them brightest first with the evidence string and the tier reason; it
  never changes one from the other.
- **No corroboration split.** The evidence scale never reads the
  `--corroborate` source (another source's sighting is a tag), so a run's
  levels ARE its own evidence and the card no longer splits the identified
  rows by corroboration.

## Acceptance

Today's acceptance criteria are read on the **identified** class; each keeps
the metric it was read on before, beside it on the card (`acceptance`), the
board and the page:

| criterion | read on (identified class) | the old metric, kept |
|---|---|---|
| roster recall not lower | `roster_identified` — roster formulas read as themselves whose reading is identified | `roster_assigned` |
| roster recall not lower (neutral or better) | `roster_neutral_or_better` — the same, identified or neutral (3c + 4a): a class list names no single compound, so its formulas never reach 3c | `roster_assigned` |
| roster misreads not higher | `roster_misread_identified` — a roster line read as another neutral that is identified | `roster_misread` |
| decoy rate not higher, per arm | `decoy_shift_identified_rate`, `decoy_shift_identified_lt_350_rate` (the Da shift arm; on an Orbitrap row the below-350 criterion is marked blind there), `decoy_ppm_identified_rate`, `decoy_adducts_identified_rate` | `decoy_shift_rate`, `decoy_ppm_rate`, `decoy_adducts_rate` (each arm's own Assigned rate) |
| decoy rate not higher below m/z 350 (the headline shift arm: the ppm arms when they ran) | `decoy_headline_lt_350_rate` (Assigned) | none: the headline is read on Assigned already |
| bright M0 not worse | `bright_m0_not_identified` (ion-only rows excluded, as for the old count) | `bright_m0_not_assigned` |
| M1 families not worse | `m1_families` (unchanged) | `m1_families` |
| cross-instrument agreement not lower | `m3_own_missing_identified` — the other instrument's identified rows (3c) that this run lacks | `m3_other_instrument_own_missing` |

The board rows written before the claim lack these keys: the delta reads None
and the tables show a dash.

A delta is taken only between two rows **measured alike** (`comparable`):
a metric that reads the evidence scale needs the same scale (`scale` on the
row; a row written before the key reads its `claims_schema`); a roster count
needs the same M2 presence test (`roster_test`; 1 before the key); a decoy
rate needs the same arm calibration (`decoy_calibration`; `own` before the
key) and scoring (`decoy_scoring`); the headline below m/z 350 needs the
same headline arm (`decoy_headline_arm`: the ppm arms, or the Da arm, which
is blind there on an Orbitrap), and a ppm-arm rate (and a headline that
quotes the ppm arms) the same shifts (`decoy_ppm_k`; none before the key).
Otherwise the previous value is withheld
and, where both rows hold a number, the card's §6 and the page's delta table
say `not comparable: <why> (previous <value>)`, and the board's claims table
says so in its last column -- 20 roster formulas identified on the
pre-0.10.0 scale and 2 on 0.10.0 is a change of scale, not a regression.

## The six sections

1. **What was assigned.** Spectra, peaks, the stamped share of peaks and of
   signal (M0 alone as well), merged rows by tier, stage and adduct channel,
   the file vote, and **stamp coverage** — merged rows with no stamped peak in
   the time series. A merged row the series never carries is a stamping defect
   (the batch-level reagent-N re-read drops the stamp, for one), not a
   chemistry question, and the rows are listed with their `tier_reason`.
2. **The brightest 50 ions** by median height over the batch, with role,
   reading, tier, evidence level, claim and tag kinds — and the count of bright M0
   rows that are not Assigned (and not identified), plus how many of the
   batch's 50 brightest tracks overall are unstamped (those are named in M1).
   Every bright ion is either explained or a named miss.
3. **The best-evidence 50** (the best level first, then the brightest), the level
   vector `3c/4a/4b/5a/5b/reagent/NA`, the rows per claim and the tag-kind
   histogram over every levelled row.
4. **What was missed.**
   - *M1 unexplained*: the brightest unstamped **tracks** (the unstamped
     peaks binned by the batch tolerance) present in ≥ 50 % of spectra. Each
     is joined to the nearest stamped ion (Δ mDa / ppm), to
     `characterize_residual`'s tag from the per-file `unexplained` rows
     (iso-partner / has-constraints / isolated, with the 13C carbon bracket)
     and to the engine's own reason for clearing it (`CLEARED (mass-gate …)
     Was: …` becomes `mass-gate: z=8.7 > 4.0; tried C9H17NO [M+^NO3]-`).
     Tracks are then **grouped into families** by their shift from the
     nearest parent, so a comb is one row: a merged-but-unstamped row, `+H`
     (the M⁻• electron-attachment family), isotope lines, a cluster-channel
     line of a bare parent, a neutral loss, a reagent-ladder line, an isotope
     line of another unstamped track, a shoulder of a brighter stamped ion,
     or `unknown`. In a ±H tie between two acids that differ by H₂ the +H
     reading wins. **Not** built from `tables/residual_bins.csv`: that table
     lists only bins absent from every cover file, and every headline miss
     of the first cut sat inside the cover files.
   - *M2 expected but not assigned*: every neutral on the rosters
     (`peaky/data/rosters/*.csv`), the reagent's reference ions
     (`chem.reference_ions`) and the pass-0 known species of the run's
     polarity, looked for on the run's own channels in the time series:
     `assigned` as itself, `candidate`, `same ion` (the line is read as
     another neutral / adduct split of the same ion composition --
     C10H15NO7 [M-H]- and C10H14O4 [M+NO3]- are one composition, one exact
     mass, and nothing in that line's mass tells them apart: not a misread),
     `read as` something else (the reading is shown), `unstamped` (present in
     the series, with the engine's reason), `isotope/reagent line` (the only
     stamped lines there are isotope satellites of another ion or reagent
     lines of another composition: no sighting) or `absent`. A reagent line
     of the expected ion's own composition is a sighting: the reagent's
     reference ions (the reagent ion, its water clusters) are `assigned`
     there, read as the reagent, and any other source's formula is `same ion`.
     The **presence test** (`ROSTER_TEST` 2, `roster_test` on the
     row): a line counts within `roster_window_ppm` -- 4 x the run's measured
     mass sigma (`mass_scale.sigma_ppm`), at least 1 ppm, at most the run's
     tolerance (a flat 6 ppm on an Orbitrap at ~0.2 ppm sigma reached
     neighbours 25 sigma away) -- and in >= 20 % of the spectra, stamped or
     not; of the stamped lines that pass, the best-read one stands for the
     formula (its own reading, then another split of its ion, then another
     reading, then an isotope / reagent line; the nearest among equals).
     `present` counts every status but `isotope/reagent line` and
     `absent`. Roster recall is reported per roster and per class, and every
     source's line on the card counts each status; each
     row carries the claim of the reading on its line, and `roster_claim`
     counts per roster the formulas read as themselves per claim,
     `neutral_or_better` (identified + neutral, side by side with identified)
     and the misreads whose other reading is identified -- per line read: the
     claim of the one line M2 picked for a formula, not its best claim over
     every channel it is read on; the M2 block names
     the level scale its claims read (`scale`), the test and its window.
   - *M3 found elsewhere*: neutrals the other path (`--other`, ≥ 2 files or
     Assigned) holds and this run lacks; and neutrals the other instrument
     (`--other-instrument`) establishes — level 3c, 4a or 4b on the evidence
     scale — above the detection floor inside the overlap window (with the
     masked gaps removed), that this run lacks, counted twice on two level
     sets that differ: `m3_other_instrument_missing` reads the other
     instrument's in-core level (for a run levelled before the scale, its
     post-hoc level with this run as its `--corroborate` partner), which keeps
     its ion-only readings and can owe a series anchor to an other-source
     partner (its `--corroborate` sources, which may be this run);
     `m3_other_instrument_own_missing` reads its own evidence
     (`own_levels_for`: its files pooled with no other-source partner), with
     the ion-only readings left out. Where the other instrument
     is TOF-class or class-less its rows read `NA` (not assessed), so M3
     counts its **Assigned** rows instead and its metric label says so. The
     counts are split by claim (`n_good_by_claim`, `n_missing_by_claim`:
     identified = 3c, neutral = 4a, ion = 4b), before the listed rows are cut
     to 40. The other instrument's levels never feed this run's levels.
5. **Is it right.** (a) roster recall (see M2; the rosters are unreviewed
   until the user signs them off); (b) the element census of Assigned
   neutrals — F, Si, P, Cl, Br, S, N and C > 20, with examples — and, on a
   TOF-class run, the Assigned rows the **mass-only flag** marks (no attached
   isotope line of the neutral's own elements in any Assigned file;
   [OUTPUTS.md](OUTPUTS.md)), split at the run's `tof_flag.threshold_mz`
   (board keys `census_mass_only` and `census_mass_only_split`, appended;
   `null` off a TOF and on a run made before the flag); (c) the
   **decoy false-discovery bound**: the engine run offline (`assign.run(peaks=)`)
   on the brightest cover file(s) as they are (the control), with every m/z
   shifted (two kinds of arm, below), and with the adduct set of the
   wrong polarity chemistry (`[M+Cl]-`/`[M+I]-` on a negative run,
   `[M+Na]+`/`[M+NH4]+` on a positive one) less every channel the run
   itself reads (`own_adducts`: its profile's adducts, every adduct its
   ledgers commit, a side channel it opened -- an iodide run's `[M+I]-` is
   never "wrong"; with nothing left the arm does not run and
   `decoy.adducts_skipped` says so); what is still Assigned, per tier
   and level, is the error bound, reported on either side of m/z 350
   (Assigned and its rate against the control's Assigned on the same side)
   and per 50-Da m/z bin (`decoy.bins`: control, the 0.35 Da arm and the ppm
   arms, each with its rate). The **0.35 Da arm** (`shift`,
   `--decoy-offset`) is kept for continuity: below ~m/z 350 it lands in the
   mass-defect gap where no CHNOS formula sits, so on an Orbitrap the engine
   proposes almost nothing there and no tier gate is tested; the card labels
   it "blind below ~m/z 350 on an Orbitrap". The **ppm arms** (`--decoy-ppm`,
   default +9 and -9 ppm; `none` for none) scale every m/z by
   (1 + k x 1e-6): outside the file's match window -- the true formula is out
   of reach -- but inside the populated mass-defect band, where the formula
   grid is dense, so the engine proposes wrong formulas and the mass,
   degeneracy and pattern gates are exercised (a shift keeps every isotope
   and label spacing, so no shift decoy tests the label / isotope vetoes).
   A shift inside the window (the wider of `PassConfig.search_ppm`
   and the match window the file is scored at; 5 ppm on an Orbitrap, 15 ppm
   on a TOF or without a snapshot) is no decoy: it is skipped, and
   `decoy.ppm_skipped` says why. Each ppm arm (`decoy.ppm_arms`) is rated
   against its files' control; pooled (`decoy.ppm`) they are rated against
   the control counted once per arm that ran (`ppm.control`). The
   wrong-adducts arm is also counted at the **ion level**
   (`ion_level_counts`): an Assigned row whose ion composition equals the
   control's M0 reading on the same peak (any tier) only re-splits an ion the
   run already reads -- `X [M+NH4]+` is the ion of `X+NH3 [M+H]+` -- and is
   wrong only about the neutral, by construction of the arm; a different
   composition, or a peak the control left unexplained, is a NEW ion.
   `decoy.adducts` carries `assigned_same_ion`, `assigned_new_ion`,
   `new_ion_rate` (new ions against the control's Assigned, below m/z 350
   too) and `same_ion_share`, beside the reading-level `assigned_rate`; the
   board row `decoy_adducts_new_ion_rate` and `decoy_adducts_same_ion_share`.
   The card's
   **headline below m/z 350** (`decoy.headline`; board
   `decoy_headline_lt_350_rate`, `decoy_headline_arm`) is the pooled ppm
   arms whenever they ran, else the 0.35 Da arm; the line names the
   calibration the arms actually ran at (`calibration_summary`: the
   control's, their own, unrecorded on a re-count of ledgers kept before the
   field, or mixed) and the M0 rows of any tier the arm committed below 350
   against the control's -- on a TOF the ±9 ppm arms sit inside the 15 ppm
   window and are skipped, and the 0.35 Da arm that the headline then quotes
   commits few rows below 350 there too. The board's lead claims table
   reads its decoy column from the same arm (`decoy_ppm_identified_rate` and
   its below-350 rate, named with the shifts, when the ppm arms ran), and an
   Orbitrap row's Da-arm number is marked blind below 350 (`decoy_orbitrap`
   on the row). `--decoy shift` runs both
   kinds of shift arm, `--decoy ppm` the ppm arms only, `--decoy both` every
   arm. Each arm is
   scored at the measurement the run judged its file at -- the width, offset,
   window and abundance floor of the file's `pattern_scoring` snapshot in the
   batch summary (a 0.9.0 run records one per sample) -- with the per-peak
   signal-to-noise its per-file ledger carries; a run from before 0.9.0
   records no snapshot, and a snapshot that fails the registration's check
   (`io_mascope._check_offline_scoring`: a width, offset, window or floor
   that is not a finite number -- text and booleans included; a snapshot may
   omit its floor, the library default -- a width or window not above zero,
   an abundance floor outside [0, 1)) counts as none: those arms take the
   offline class fallback, a TOF's width at zero offset.
   Every arm but the control also runs at its file's **control calibration**
   (`inherited_calibration`): the pass-stage mass fit (`passes.calibrate`:
   the mass gate's mu, sigma and 1/mz trend) and the tier engine's
   (`tiers._calibrate`: the tiers, the degeneracy audit, the winner selection)
   are taken on the control arm's ledger, not on the arm's own commits. An
   arm's own backbone is made of wrong readings; when it is too small to
   calibrate (a shifted backbone, or a wrong-adducts arm that commits few
   rows), the
   arm runs with the mass z-test and the degeneracy audit off and keeps every
   mass fit -- it would bound an engine no real file runs, since a real file
   always calibrates. `decoy.calibration` says per file what its arms ran at
   (`control`; `own` where the control arm errored and left nothing to
   inherit; `unrecorded` on a re-count of ledgers kept before the field), the
   manifest keeps it, and the board row records one word
   (`decoy_calibration`). The arms are levelled at the file's run window, as
   before.
   An arm on the run's own channels (control, shift, ppm) opens the side
   channels the run recorded opening (`batch_summary.side_channels`; none for
   a run that recorded none); the wrong-adducts arm reads its wrong set alone.
   What an arm does NOT inherit from the run: its batch-derived height
   cutoff, its pre-calibration prior offset, its batch occurrence table, its
   batch time series (pass 7's time-series corroboration and the time-series
   stage), its active reference lists (the
   reflist prior and the pass-8 rescue) and its corroboration -- so the
   control is the engine on the file under the arm's own settings, not the
   run's per-file ledger. They differ most in M0 and Assigned rows on a TOF
   batch, and (on the pre-release level scale, which read the corroborating
   source) in the identified class on a corroborated batch (R2, 0.9.0:
   the control holds 11 identified pairs where the run's file holds 50, 43 of
   them identified only through corroboration); card C39. On a TOF the
   brightest file -- the decoy file -- can be the batch's worst-fitted one
   (R3: 8.75 ppm on 29 anchors, where the batch's median fitted width is
   3.0), and its arms are scored at that; the card prints each file's
   inherited width, offset, window, floor and anchors. Per file,
   `decoy.scoring` says what the arms were scored at (`inherited` /
   `class-fallback`, or `unrecorded` on a re-count of ledgers kept before the
   field existed) and `decoy.scoring_detail` the inherited numbers, both kept
   in the decoy manifest for a re-count; the card prints them under the decoy
   heading and the board row records one word for the card (`decoy_scoring`:
   one of those, or `mixed`). A delta of a decoy rate across a change of
   scoring or of arm calibration is marked not comparable (see
   Acceptance). Two units sit
   side by side: the tier counts are M0 **rows**; the level vector and the
   **per-claim** counts are distinct (neutral_formula, adduct) M0 **pairs** of
   any tier, each pair's m/z and tier taken from its brightest M0 row (the
   Assigned-only variants beside them). Each arm reports its pairs per claim and
   bucket below and above m/z 350, and its rate against the control's pairs
   of the same claim; where the control holds no pair of that claim the rate
   is undefined (`null`, a dash on the card), not 0 %. Each arm's engine
   ledger is kept as `<out>/<run>/decoy/<file>__<arm>.csv.gz`, with
   `manifest.json` beside it; `--decoy-ledgers DIR` (a scoreboard out dir, or
   the decoy dir itself) counts those ledgers instead of running the engine
   again — a re-score at no engine cost, the same counts. The card then
   reports the manifest's offset, ppm shifts, files, adduct sets and scoring, not the command
   line's, and `ledgers.code` names the engine that made them (ledgers kept
   before the ppm arm carry none). It implies
   `--decoy both` unless `--decoy` is given; a missing ledger is that arm's
   error, and a DIR with no kept ledger for a run stops the scorecard before
   any card or board row is written. An arm that crashes the engine is
   recorded as that arm's error — exactly `{"error": ...}` — and the card
   goes on;
   (d) **falsification survival**: the
   13C-implied carbon count against the formula on every Assigned row with a
   measured satellite, the 34S / 37Cl / 81Br / 29Si line where the formula
   demands one, the time covariance of satellites and adduct pairs with
   their parent, and the **separability** of the Assigned peaks (the
   `resolvability` stamp's class counts over the per-file rows) with the
   per-file rows the separability / satellite tier rules capped at Candidate.
6. **Delta** against the previous row of the same channel, for the metrics
   the board tracks — the claim metrics first; a metric measured differently
   on the two rows has no delta and a `note` column says why (see
   Acceptance).

## Levels

The merged ledger's in-core `evidence_level` (the evidence scale,
docs/EVIDENCE_LEVELS.md) is preferred. A run made before the scale carries no
such column (or the pre-release one): the level then comes from
`scripts/level_ledger.py`, run in-process on the run dir, and `--levels` takes
that script's `--out` CSV instead. The scorecard reads `NA` with
`keep_default_na=False` (or from `claim`), so "not assessed" never folds into
"no level".

## Rosters

`peaky/data/rosters/alpha_pinene.csv` and `contaminants.csv` are cited
expectation lists (name, neutral formula, class, reference, note); nothing in
the assignment engine reads them, so a hit is evidence about the engine. Two
rows may share a formula (isomers); a formula is counted once. Add a row with
its citation; `--rosters DIR` swaps in another directory.

## Tests

`tests/test_scorecard.py` builds a four-spectrum run dir with three assigned
acids, one 13C satellite, a merged row the stamp never carried, three +H lines,
a shoulder, an unknown and a non-persistent bright track, and pins every
panel's numbers, the M1 family grouping, the M2 statuses, the in-process and
in-core level paths, the offline decoy engine run, M3 with a window, the
board round trip with a delta, and the CLI.

`tests/test_scorecard_decoy_arms.py` pins the decoy arms as a real file runs
them: the control calibration taken and given back, arm by arm; the ppm arm's
m/z scaling, keys and kept-ledger names, a shift inside the match window
skipped, the ppm arms pooled against the control counted once per arm, the
headline (ppm arm first, else the 0.35 Da arm, marked blind on an Orbitrap;
the calibration it names read from the arms, never assumed),
the 50-Da bins, a re-count of kept ppm-arm ledgers; a run's own channel never
in the wrong set (and no arm when none is left), the wrong-adducts arm's
same-ion re-splits told from its new ions (on the same peak only), and the
board's lead decoy cell and acceptance reading the headline arm.

`tests/test_scorecard_roster.py` pins M2's presence test (the sigma window,
the share of spectra, an isotope / reagent line no sighting, a same-ion split
no misread, the best-read line in the window, a reagent line of the
expected ion's composition a sighting), the roster claims side by side with
their scale, and a delta across a scale, a presence test, a decoy
calibration, a headline arm or a set of ppm shifts marked not comparable on
the card, the board (both tables) and the page.

`tests/test_scorecard_claims.py` writes in-core levels onto that run and adds
a Candidate the acid branch identifies, an ion-only line and a per-file
reading no merged row carries, and pins the claim block (rows, the crosstab
with its ion-only row, the signal shares with the unmatched bucket, the
disagreeing rows, the stamp mismatch), the claim on the brightest / best-evidence / M2 / M3 panels, the
decoy pairs per claim on the in-core and post-hoc level paths, the per-claim
rates (undefined against a control with none of the claim), an errored arm
left exactly as its error, the kept arm ledgers re-counted by `--decoy-ledgers`
to the same numbers with the manifest's offset and files, an explicit
`--decoy none` kept, a DIR with no kept ledger stopping before the board, the
board-row keys (every old key first, unchanged), the old metrics' delta-chip
directions unchanged, §0 before §1, the claims table leading SCOREBOARD.md,
the §6 delta table's alignment beside a row from before the claim, the Claims
table on the page, and a card from before the claim rendering beside a new
one.
