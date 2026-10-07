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
    [--decoy none|shift|adducts|both --decoy-offset 0.35 --decoy-files 1] [--decoy-ledgers DIR]
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
| `<out>/<run name>/decoy/<file>__<arm>.csv.gz` | each decoy arm's engine ledger (`control`, `shift`, `adducts`), kept for a re-count |
| `<out>/<run name>/decoy/manifest.json` | what those ledgers were made with: mode, offset, files, adduct sets, engine code |
| `<out>/scoreboard.jsonl` | one row per run, appended; the board's memory |
| `<out>/SCOREBOARD.md` | the claims table first, then every channel's latest row with its delta to the row before (`## All metrics`) |
| `<out>/scoreboard.html` | the page source of "Peaky Scoreboard": the claims table, the channel table and one tabbed panel per run |

A **channel** is `<batch slug>|<reagent>|<path>` (`cover` or `trace-first`),
so a path change is a different channel and its comparison belongs to
`scripts/ab_compare.py`; the delta on the board is always the same channel
against its previous row.

## 0. The claim

Every committed reading carries, beside its tier, the **claim** its evidence
level supports (`evidence.claim_class`, docs/EVIDENCE_LEVELS.md):

| claim | levels | what a reader may say |
|---|---|---|
| identified | 1, 2a, 2b, 3a, 3b, 4a | the neutral is established: the formula can be reported as a compound or class |
| ion | 4b, 4c, 4d | the ion composition is pinned; the neutral / adduct split is open |
| tentative | 5a, 5b, no level | exact mass only, or the assignment argues with itself |

The card always derives the claim from `evidence_level` (a run made before the
`claim` column existed carries none; one made before the in-core level takes
it from the in-process levels join). Where a run did write a `claim` column,
the rows whose stored claim differs from the derived one are counted as
`stamp_mismatch` — a stamping defect, expected 0. Only a committed M0 reading
makes a claim; every merged row has one (a row with no level reads tentative).

- **Merged rows per claim**, and the **tier × claim crosstab** with three tier
  rows: Assigned, Candidate and ion-only (the `ion_only_of` rows, levels 4d /
  5a, kept out of the Candidate row as the headline keeps them), by rows and by
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
  lists them brightest first with the level reason and the tier reason; it
  never changes one from the other.
- **Corroboration.** The identified rows are split by the channel's OWN
  evidence (`own_levels_for`: its per-file ledgers levelled with no cross
  set): identified alone, or identified only with the corroborating source,
  with the own-evidence levels of the latter. An identified row that is 5b on
  its own evidence is flagged by name.

## Acceptance

Today's acceptance criteria are read on the **identified** class; each keeps
the metric it was read on before, beside it on the card (`acceptance`), the
board and the page:

| criterion | read on (identified class) | the old metric, kept |
|---|---|---|
| roster recall not lower | `roster_identified` — roster formulas read as themselves whose reading is identified | `roster_assigned` |
| roster misreads not higher | `roster_misread_identified` — a roster line read as another neutral that is identified | `roster_misread` |
| decoy rate not higher, per arm | `decoy_shift_identified_rate`, `decoy_shift_identified_lt_350_rate`, `decoy_adducts_identified_rate` | `decoy_shift_rate`, `decoy_adducts_rate` |
| bright M0 not worse | `bright_m0_not_identified` (ion-only rows excluded, as for the old count) | `bright_m0_not_assigned` |
| M1 families not worse | `m1_families` (unchanged) | `m1_families` |
| cross-instrument agreement not lower | `m3_own_missing_identified` — the other instrument's rows at level <= 4a by its own evidence that this run lacks | `m3_other_instrument_own_missing` |

The board rows written before the claim lack these keys: the delta reads None
and the tables show a dash.

## The six sections

1. **What was assigned.** Spectra, peaks, the stamped share of peaks and of
   signal (M0 alone as well), merged rows by tier, stage and adduct channel,
   the file vote, and **stamp coverage** — merged rows with no stamped peak in
   the time series. A merged row the series never carries is a stamping defect
   (the batch-level reagent-N re-read drops the stamp, for one), not a
   chemistry question, and the rows are listed with their `tier_reason`.
2. **The brightest 50 ions** by median height over the batch, with role,
   reading, tier, evidence level, claim and axes — and the count of bright M0
   rows that are not Assigned (and not identified), plus how many of the
   batch's 50 brightest tracks overall are unstamped (those are named in M1).
   Every bright ion is either explained or a named miss.
3. **The best-evidence 50** (level ≤ 4a first, then most axes, then
   brightest), the level vector `2b/3a/3b/4a/4b/4c/4d/5a/5b`, the rows per
   claim and the axes histogram over every levelled row.
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
     `assigned` as itself, `candidate`, `read as` something else (the reading
     is shown), `unstamped` (present in the series, with the engine's reason)
     or `absent`. Roster recall is reported per roster and per class; each
     row carries the claim of the reading on its line, and `roster_claim`
     counts per roster the formulas read as themselves per claim and the
     misreads whose other reading is identified.
   - *M3 found elsewhere*: neutrals the other path (`--other`, ≥ 2 files or
     Assigned) holds and this run lacks; and neutrals the other instrument
     (`--other-instrument`) holds at level ≤ 4b, above the detection floor
     inside the overlap window (with the masked gaps removed), that this run
     lacks — counted twice: by the other instrument's merged in-core level
     (`m3_other_instrument_missing`, the original metric) and by its **own**
     evidence, its per-file ledgers levelled with no cross set
     (`m3_other_instrument_own_missing`, `own_levels_for`). The two differ where
     the other instrument's in-core level owes a rung to its own
     `--corroborate` source — on a same-air pair that source is the run being
     scored, so the first count can hold this run's own agreement against it
     (EVIDENCE_LEVELS.md §6.4). Both counts are split by claim
     (`n_good_by_claim`, `n_missing_by_claim`: identified = the other's level
     <= 4a, ion = 4b), before the listed rows are cut to 40. The other
     instrument also corroborates this run's levels.
5. **Is it right.** (a) roster recall (see M2; the rosters are unreviewed
   until the user signs them off); (b) the element census of Assigned
   neutrals — F, Si, P, Cl, Br, S, N and C > 20, with examples; (c) the
   **decoy false-discovery bound**: the engine run offline (`assign.run(peaks=)`)
   on the brightest cover file(s) as they are (the control), with every m/z
   shifted by `--decoy-offset` (0.35 Da lands in the mass-defect gap below
   ~m/z 350, where no CHNOS formula sits), and with the adduct set of the
   wrong polarity chemistry (`[M+Cl]-`/`[M+I]-` on a negative run,
   `[M+Na]+`/`[M+NH4]+` on a positive one); what is still Assigned, per tier
   and level, is the error bound, reported on either side of m/z 350 because
   the mass-defect gap a 0.35 Da shift lands in closes above it. Two units sit
   side by side: the tier counts are M0 **rows**; the level vector and the
   **per-claim** counts are distinct (neutral_formula, adduct) M0 **pairs** of
   any tier, each pair's m/z and tier taken from its brightest M0 row (the
   Assigned-only variants beside them). Each arm reports identified / ion /
   tentative pairs below and above m/z 350, and its rate against the
   control's pairs of the same claim (`identified_rate` equals the old
   `good_level_rate`); where the control holds no pair of that claim the rate
   is undefined (`null`, a dash on the card), not 0 %. Each arm's engine
   ledger is kept as `<out>/<run>/decoy/<file>__<arm>.csv.gz`, with
   `manifest.json` beside it; `--decoy-ledgers DIR` (a scoreboard out dir, or
   the decoy dir itself) counts those ledgers instead of running the engine
   again — a re-score at no engine cost, the same counts. The card then
   reports the manifest's offset, files and adduct sets, not the command
   line's, and `ledgers.code` names the engine that made them. It implies
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
   the board tracks — the claim metrics first.

## Levels

Until the in-core `evidence_level` column exists (Phase B of the plan), the
level column comes from `scripts/level_ledger.py`, run in-process on the run
dir (corroborated by `--other-instrument` when given). `--levels` takes that
script's `--out` CSV instead. When the merged ledger carries
`evidence_level` (and `evidence_axes`), it is preferred and the script is not
run.

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

`tests/test_scorecard_claims.py` writes in-core levels onto that run and adds
a Candidate the acid branch identifies, an ion-only line and a per-file
reading no merged row carries, and pins the claim block (rows, the crosstab
with its ion-only row, the signal shares with the unmatched bucket, the
disagreeing rows, the corroboration split through `own_levels_for`, the stamp
mismatch), the claim on the brightest / best-evidence / M2 / M3 panels, the
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
