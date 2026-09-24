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
    [--decoy none|shift|adducts|both --decoy-offset 0.35 --decoy-files 1]
    [--rosters DIR] [--out ~/peaky-output/scoreboard]
```

A run dir is the timestamped folder that holds `merged_ledger.csv`, or the
`--out-dir` that holds exactly one. Everything the card needs is in the run:
the merged ledger, `per_file/*_ledger.csv`, `per_file/_batch_ts.parquet`,
`batch_summary.json` and `run_manifest.json`.

## Outputs

| file | what |
|---|---|
| `<out>/<run name>/SCORECARD.md` | the card: six numbered sections (below) |
| `<out>/<run name>/scorecard.json` | the same, as data — every table as records |
| `<out>/scoreboard.jsonl` | one row per run, appended; the board's memory |
| `<out>/SCOREBOARD.md` | every channel's latest row with its delta to the row before |
| `<out>/scoreboard.html` | the page source of "Peaky Scoreboard": the channel table and one tabbed panel per run |

A **channel** is `<batch slug>|<reagent>|<path>` (`cover` or `trace-first`),
so a path change is a different channel and its comparison belongs to
`scripts/ab_compare.py`; the delta on the board is always the same channel
against its previous row.

## The six sections

1. **What was assigned.** Spectra, peaks, the stamped share of peaks and of
   signal (M0 alone as well), merged rows by tier, stage and adduct channel,
   the file vote, and **stamp coverage** — merged rows with no stamped peak in
   the time series. A merged row the series never carries is a stamping defect
   (the batch-level reagent-N re-read drops the stamp, for one), not a
   chemistry question, and the rows are listed with their `tier_reason`.
2. **The brightest 50 ions** by median height over the batch, with role,
   reading, tier, evidence level and axes — and the count of bright M0 rows
   that are not Assigned, plus how many of the batch's 50 brightest tracks
   overall are unstamped (those are named in M1). Every bright ion is either
   explained or a named miss.
3. **The best-evidence 50** (level ≤ 4a first, then most axes, then
   brightest), the level vector `2b/3a/3b/4a/4b/4c/4d/5a/5b` and the axes
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
     `assigned` as itself, `candidate`, `read as` something else (the reading
     is shown), `unstamped` (present in the series, with the engine's reason)
     or `absent`. Roster recall is reported per roster and per class.
   - *M3 found elsewhere*: neutrals the other path (`--other`, ≥ 2 files or
     Assigned) holds and this run lacks; and neutrals the other instrument
     (`--other-instrument`) holds at level ≤ 4b, above the detection floor
     inside the overlap window (with the masked gaps removed), that this run
     lacks. The other instrument also corroborates this run's levels.
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
   the mass-defect gap a 0.35 Da shift lands in closes above it. An arm that
   crashes the engine is recorded as that arm's error and the card goes on;
   (d) **falsification survival**: the
   13C-implied carbon count against the formula on every Assigned row with a
   measured satellite, the 34S / 37Cl / 81Br / 29Si line where the formula
   demands one, the time covariance of satellites and adduct pairs with
   their parent, and the **separability** of the Assigned peaks (the
   `resolvability` stamp's class counts over the per-file rows) with the
   per-file rows the separability / satellite tier rules capped at Candidate.
6. **Delta** against the previous row of the same channel, for the metrics
   the board tracks.

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
