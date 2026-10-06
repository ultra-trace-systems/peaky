# Peaky — Merge (cross-sample alignment + m/z jitter)

This document explains **how per-sample assignments become one merged peak list**
— the offset-aware m/z alignment, the consensus-row selection, and the file-to-file
**jitter** accounting. It is a module deep-dive companion to
[`ARCHITECTURE.md`](ARCHITECTURE.md) (the whole pipeline, the *Whole batch* flow),
[`SAMPLING.md`](SAMPLING.md) (which picks the files merged here), and
[`ASSIGNMENT.md`](ASSIGNMENT.md) (what each per-file ledger contains).

**Code:** `peaky/batch/assign_batch.py`. `align()` / `merge_union()` /
`jitter_report()` are **pure** (offline-tested); `run()` does the network
assignment loop and writes the run artifacts.

> Keep this in sync with the code. Every threshold below is a named constant or a
> literal in `assign_batch.py`; if you change one there, change it here.

---

## 1. What this stage does

`match_compounds` is per-sample — a synthetic union spectrum can't be scored — so
the batch path assigns each cover-selected file **separately** and then **combines
the real per-file ledgers by m/z**. The combine is **offset-aware**: each file
carries a median mass offset, alignment happens on offset-corrected m/z so a
genuine same-peak isn't split by per-file calibration drift, and the report
separates the *raw* mass spread from the *calibration-removed* (residual) spread —
that residual is the genuine peak-position noise the maintainer wanted to
investigate.

```
selected sample_ids (SAMPLING.md)
   │  for each: assign.run → per-file ledger (kept on disk) + estimate_offset
   ▼
 per_file = {sid → M0 rows [mz, neutral_formula, adduct, tier, ion_score]}
   │  align(offsets): _mz_adj = mz·(1 − offset_ppm/1e6)
   ▼  single-linkage gap-cluster _mz_adj at tol_ppm (6)
 one row per m/z cluster:
   consensus mz = mean(raw mz);  the files VOTE in two stages:
     which ION  -- the best per-file VOTE CLASS wins (neutral backed >
                   formula confirmed > unconfirmed; not the evidence level),
                   then most files (Assigned-file count, ion_score
                   break ties); a known species is decided after the vote
     which LABEL of it -- the reading Assigned in the most files (a same-ion
                   label split is the reagent-N isobar: Candidate = undecided)
   n_files, n_files_ion, n_files_winner, alternatives, tier_reason, srcs,
   ion_agree, formula_agree, mz_jitter_ppm_raw, mz_jitter_ppm_caldj
   │  reagent-water ladder (batch/reagent_water.py): rungs core.(H2O)n measured on
   │   the batch TS; a merged row on a passing rung leaves, the rung is stamped
   │  (positive urea, ONCE on the merged ledger: relabel_reagent_n_adducts, then
   │   prefer_amine_over_ammonium -- each writes what it did to tier_reason)
   ▼
 trace reconciliation (TIMESERIES.md §9): each row's anchor re-centred on its
 own trace (mz_trace), competing labels on one trace collapsed (trace_role),
 the stamp window sized to the batch's per-ion scatter
   ▼
 merged_ledger.csv  +  jitter.csv  +  batch_summary.json  →  _batch_ts.parquet stamp
   ▼  (TOF-class only) the ion-M+2 gates: a REQ-refuted winner and an 81Br doublet
      partner are Candidate, no re-vote (step 6b)
```

---

## 2. Inputs

- `per_file` — `{src → DataFrame}` of each file's **M0 (assigned-compound) rows**
  in the `_M0_COLS` schema (`mz`, `neutral_formula`, `adduct`, `tier`,
  `ion_score`, the admission / ion-only / resolvability provenance, and the
  reading's `vote_class` — computed in the parent from the file's own facts,
  read by the vote), extracted by `_m0`.
- `offsets` — `{src → median ppm}` from `io_mascope.estimate_offset`
  ([`DATA_IO.md`](DATA_IO.md)); missing → treated as 0.

---

## 3. The transformation, stage by stage

1. **Assign each selected file** (`run`). Loop the selected `sample_ids`,
   `A.run(sid, …)` each, write `per_file/<sid>_ledger.csv`, keep the M0 rows, and
   record the file's `estimate_offset`. The reagent's analyte channels are forced
   at batch level (`assign_kw.setdefault("adducts", prof.adducts)`) so a per-sample
   match gap can't flip polarity.

2. **Offset-correct for alignment** (`align`). For each file,
   `_mz_adj = mz · (1 − offset_ppm/1e6)`. This is used **only** to cluster; the
   reported masses stay raw.

3. **Gap-cluster** (`_cluster_mz`). Sort by `_mz_adj`; consecutive-gap
   single-linkage: `gaps = diff(mz)/mz · 1e6`, `cluster_id = cumsum(gaps >
   tol_ppm)`. One cluster ≈ one physical peak across files. **The window is the
   batch's measured merge window**, `traces.MassScale.merge_ppm`: `run` measures
   ONE mass scale before its first merge (`traces.measure_mass_scale` — the
   per-file anchors, offset-corrected, walked onto their trace by mean shift, one
   centre per trace, the third quartile of the per-trace scatter σ; the stamp's
   own estimator, measured once for both) and sizes the window as
   `max(tol, min(2·tol, 2.5·√2·σ))` with `tol` = `DEFAULT_TOL_PPM` (6.0 =
   `sampling.BATCH_TOL_PPM`, the tolerance the selector binned on — see
   [`SAMPLING.md`](SAMPLING.md)). The gap between two anchors is the difference
   of two draws, √2 wider than one draw about a centre, hence √2 × the stamp's
   2.5 σ. Floored at the binning tolerance and capped at twice it: a run with no
   time series (nothing to measure) clusters at the flat 6 ppm exactly; an
   Orbitrap (σ 0.2–0.3 ppm) sits on the floor and clusters at 6 ppm (measured on
   two Orbitrap channels: the merge is inert from 3 to 9.5 ppm — no 6 ppm cluster
   holds two picked peaks of one file, and a tighter window only cuts one ion's
   per-file cloud in two); a TOF (σ 3.7 ppm) clusters at 12 ppm (measured on a
   28-file batch: at 6 ppm the merge minted two rows for one ion 135 times —
   adjacent merged rows closer than the stamping window, 125 of them collapsed by
   the trace stage afterwards, and the vote in each split row never saw its rival;
   at 12 ppm none is left, and no cluster holds two picked peaks of one file —
   that starts at 15 ppm). `batch_summary['mass_scale']` records σ, the trace
   count and both windows; the log line is `[scale] …`. A pure `align()` caller
   passes the window it means (`tol_ppm`, default `DEFAULT_TOL_PPM`).

4. **The vote** (`_vote`), in two stages. A *reading* is a
   `(neutral_formula, adduct)` pair; its *ion* is the element composition of
   neutral + adduct (`_ion_key`, the tier engine's `_ion_counts`), so
   `C13H14O4 [M+NH4]+` and `C13H17NO4 [M+H]+` are one ion, `C13H18NO4+`.
   - **Which ion** sits at the m/z is what files can genuinely disagree on, and
     the **vote class, then the count**, decides it. Each ion takes the best
     **vote class** of its per-file readings (the `vote_class` column, computed
     in the parent by `evidence.vote_classes` over the file's facts, the
     `--corroborate` cross set and the batch's width model; [`EVIDENCE_LEVELS.md`](EVIDENCE_LEVELS.md) §13):
     **2 = neutral backed** — the formula confirmed and the neutral backed by
     two axes (one outside the channel), a curated identity or class, the acid
     branch, or the `--corroborate` source pinning the neutral on its own
     evidence (this reading itself at whatever strength); **1 = formula
     confirmed** — the formula or the ion pinned, the neutral not; **0 =
     unconfirmed** — exact mass alone or an assignment that argues with
     itself, or no class at all (a pure `align()` caller without the column:
     every reading is 0 and the vote is the pure count). **The vote class is
     not the evidence level and not the claim.** It is the decision peaky used
     before the evidence scale, kept private to the vote so that the scale —
     a reader's grade of the committed formula — never moves a reading; the
     vote reads `vote_class`, never `evidence_level` or `claim`, and the class
     is never printed as a level. The ion of the best class wins whatever the
     file count; among ions of one class the ion carried by the most **files**
     wins, then the number of files carrying it at **Assigned** tier
     (`TIER_RANK = {Assigned:2, Candidate:1}`, else 0), then the best
     `ion_score`, and last the ion's own text — so a full tie resolves
     identically whatever order the files arrived in (serial and parallel runs
     stay byte-identical). When the class decided against an ion carried by at
     least as many files, the row says so, naming the class: `evidence
     outranks the count: kept C9H16O6 [M+NO3]- (neutral backed in 2 of 12
     files) over the 9-file C14H21N [M+Br]- (unconfirmed)`. The count-first
     order (files, tier, score) is the
     one `collapse_trace_labels` applies to competing labels on one trace.
     Nothing is exempt from the count. A **known-species** identity used to be
     (the "curated exemption": an ion carrying a pass-0 or reference-list label
     ranked first once that label had reached Assigned somewhere); it is now
     decided once for the batch, by pooled evidence, after the vote — step 4b.
   - **Which label** of the winning ion is decided by corroboration, not by count:
     on a same-ion pair the tier engine marks a reading Assigned only when a
     discriminating channel was present in that file (an N-free sibling, the joint
     NH4+urea pair, a series anchor) and Candidate when it had nothing to decide
     with, so counting Candidate files would be counting silence. The label
     Assigned in the most files wins; file count and score break ties. When that
     overrides a bigger count the row says so: `same ion C13H18NO4+ read two
     ways: kept C13H14O4 [M+NH4]+ (Assigned in 5 of its 5 files) over the 6-file
     C13H17NO4 [M+H]+ (Assigned in 0)`.

   The winning reading's best per-file row (tier, then `ion_score`, then `src`)
   supplies the merged `neutral_formula` / `adduct` / `tier` / `ion_score` /
   `admitted_by` / `occurrence` / `resolvability` / `sep_hwhm` (the winner
   file's peak separability, when the run had a width model) / `ts_disposition` /
   `ts_cv_norm` (the winner file's time-series label, when the run had a time series:
   flatness labels a row and never tiers it, so the label has to survive the merge).
   The merged **`mz` is the mean of the cluster's
   raw m/z**. Also recorded: `n_files` (distinct srcs), `n_files_ion` (files
   carrying the winning ion), `n_files_winner` (files carrying the winning
   reading), `alternatives` (every losing reading, best first, e.g.
   `C15H25N [M+H]+ x1 Candidate 0.97`; empty when unanimous), `srcs`,
   `ion_agree` (one ion in the cluster), `formula_agree` (`≤ 1` distinct
   neutral), and the two jitter spreads (§5).

   *Why a vote.* The previous rule ranked the Assigned-file count first, so one
   file's Assigned reading outvoted many files' Candidate reading of a different
   ion: on a 15-file uronium run, 12 of 73 split clusters were decided by a
   minority, and nothing on the merged row said so. *Why two stages.* Of those 73
   split clusters 43 were two labels of one ion and 30 were different ions; in 16
   of the 43 a pure count would hand the ion to a label nobody had corroborated
   over one some file had. *Why no exemption.* On the same run the D7
   cyclosiloxane urea adduct at m/z 579.171 and tricresyl phosphate at 429.157,
   each locked in one file by the known-species list (mass + own-twin gate),
   faced an O14 / N4O10 grid formula the per-file engine itself flags as
   implausible (Candidate in every file); sulfolane at 181.065, from the same
   list in one file, met fluorenone `C13H8O [M+H]+` Assigned in nine. A rank
   exemption got the first two right and the third right by the count alone;
   step 4b gets all three right by the evidence the other files hold — they
   could not test the siloxane's twin or the phosphate's second channel (that
   is silence), and they could and did test sulfolane's ³⁴S (a refutation).
   *Why the evidence before the count.* Once the merge window put one TOF
   ion's readings in one row (under the flat window they sat in rows of their
   own, 8–12 ppm apart, each looking unanimous), the count handed the peak to
   the reading fitted in the most files whatever the files' evidence for it:
   on a 28-file TOF batch the Orbitrap-confirmed `C9H16O6 [M+NO3]-` (Assigned,
   neutral backed, 2 files) lost to `C14H21N [M+Br]-` (unconfirmed, 9 files), the HOMs `C10H16O9`
   and `C10H18O9` to bromide adducts of N-compounds read in 3–4 files, and the
   roster's pinic acid `C9H14O4` a 1-vs-1 tie on `ion_score` to an
   organosilicon formula — none a reading a negative-mode CIMS should carry
   over the one the other instrument confirms. A per-file vote class measures THAT
   file's evidence for the reading; the file count measures persistence. The
   first says which reading is right, the second how often it was seen, and a
   reading no file could establish does not become right by being fitted in
   more of them. Replayed on the three regression channels' per-file ledgers
   before the rule was coded: 160 of the TOF's 885 multi-ion clusters change
   their winner (every one a lower-class many-file reading yielding to a
   higher-class one; the four cases above come back), 56 of the ¹⁵N-nitrate
   Orbitrap's 268 and 5 of the uronium Orbitrap's 10.

4a. **The reagent-water ladder** (`batch/reagent_water.py`; right after the vote
   and the reagent-cluster guard, once per batch). The profile's `water_cores`
   (Br- / Br2-. / Br3- / HNO3.Br- on the bromide profile; NO3- / HNO3.NO3- /
   (HNO3)2.NO3- / NO2- on the nitrate one; the labelled cores on ¹⁵N-nitrate;
   none on the positive profiles) are looked for in the batch's OWN time series
   as core.(H2O)n, n = 1…45, per acquisition segment (the spectra cut at gaps
   longer than max(60 min, 5× the median spacing); a segment under 10 spectra
   joins its neighbour). A rung passes when, in some segment, the core and every
   rung 1…n are present in ≥ 50 % of the spectra within the stamping window and
   the rung is ≥ 3× the presence of its decoy offsets (±0.02 / 0.035 / 0.05 Da;
   floor 0.02, inactive while the presence bar is 50 %). How far a ladder
   reaches is measured, not declared: on the five-day bromide/nitrate TOF batch
   it ends at n ≤ 8 (Br-) and n ≤ 5 (NO3-) before an instrument restart and runs
   past n = 15 after it. A merged analyte row within the window of a passing rung
   is the water cluster and leaves the merged ledger (on that batch 70 rows,
   C13H12O8 [M-H]- at Br-.(H2O)12 and C13H22N2O4 [M+NO3]- at NO3-.(H2O)15 among
   them); every passing rung becomes a
   reagent row of the stamp (with its isotopologue tag, so the ⁷⁹Br and ⁸¹Br
   rungs stay two tracks). `tables/reagent_water.csv` lists the rungs and the
   readings each displaced; `batch_summary.json["merge_gates"]["reagent_water"]`
   the counts. The per-file ledgers are untouched. Two limits stand: a rung is
   measured per segment but stamped over the whole batch (and, on the fixed-offset
   test, stripped over it), and on the fixed-offset test a halogen rung can pass
   for one isotopologue while its twin fails the decoy gate on a neighbouring peak.

   **On a TOF the rung test changes** (the batch's width model is TOF-class: R <
   50 000 at m/z 200; an Orbitrap-class model, or none, keeps the test above
   exactly). Fixed Da offsets do not hold there: at m/z 450-700 the line is
   0.05-0.07 Da wide, the offsets sit inside it, and a TOF picker reports the
   line's own weak satellites (about a tenth of its height, ~1-1.5 FWHM below and
   ~0.5-1 FWHM above), so a bright rung fails its own decoys; and in a humid
   stretch one weak rung ends the contiguous ladder. On the bromide/nitrate TOF
   batch that left eight Br-.(H2O)n rungs between n = 21 and 35, NO3-.(H2O)29-34
   and HNO3.NO3-.(H2O)27/31 Assigned as C18-C36 organics (a "known" C30H58Cl4
   [M+Br]- was Br-.(H2O)31, its M+1 line at 0.02x where C30 needs 0.33x). Per
   segment, the TOF test passes a rung that is (a) present in ≥ 50 % of the
   spectra; (b) ≥ 3× the MEAN presence of up to twelve decoys at ±2, 2.5, 3,
   3.5, 4 and 5 FWHM(m) -- the local chance of a peak in the window -- skipping
   any decoy within 1.5 FWHM (+ the window) of another declared core or its
   rungs, which is another ladder rather than chance; (c) on its ladder: the
   ladder runs from the core through every rung present in ≥ 25 % of the spectra
   (a weak rung carries it past, an absent one ends it), and a rung beyond the
   contiguous part must co-vary with it -- median Pearson r of log height ≥ 0.8
   with the present rungs within ±2, over ≥ 8 shared spectra; and (d) carries no
   carbon: where the rung is bright enough that a C5 line's 13C would clear the
   picker's local floor (the 5th-percentile height within ±25 Da), its M+1/M0
   must stay under the cluster's own (2H/17O/15N) plus 5 × 1.07 % -- except where
   its M+1 position sits within 0.5 FWHM (+ the window) of a rung of another
   declared core present in the segment, where a TOF line is the blend of both and
   the test is not asked (the M+1 of Br-.(H2O)n lies 6 mDa from
   (HNO3)2.NO3-.(H2O)n-6). After the TOF test **the strip follows the segments**:
   a merged row on a passing rung leaves only when a file carrying its winning
   reading (`tables/jitter.csv`) lies in a segment where the rung passed; a
   reading carried only by files of segments where the rung did not pass reads
   another line at the rung's m/z, and stays, with a note on its `tier_reason`
   (the stamp still takes every passing rung over the whole batch). On the
   bromide/nitrate TOF batch C14H15N3O5 [M+NO3]- (m/z 367.091, Assigned) sits on
   Br-.(H2O)16, which passes only in the humid segment; its reading comes from
   three files of the dry segment, where the Br- ladder ends at n = 8 and the line
   at that m/z has no ⁸¹Br twin -- not the water cluster, and it stays.
   `merge_gates.reagent_water.rung_test` records the constants and the FWHM at
   m/z 200 and 600 (present only when the TOF test ran; `decoy_presence` in
   `tables/reagent_water.csv` is then the mean local decoy presence). Measured
   offline on the bromide/nitrate TOF batch's own time series and per-file
   readings: 121 → 181 passing rungs, every rung the fixed-offset test passed
   among them; the 72 rows the fixed-offset test stripped are stripped still, and
   54 more merged rows leave, 19 of them Assigned -- 18 of the batch's 19 Assigned
   organic readings at a water-cluster m/z (the rungs above, Br-.(H2O)8 =
   C4H7NO4Si [M+NO3]- and Br-.(H2O)13 = C13H18N2O2 [M+Br]-; the 19th, C13H18O11
   [M+Br]- at HBr.Br-.(H2O)15, has no declared core and no ladder in the data) and
   C4H11NO3S [M+Br]- on HNO3.Br-.(H2O)5, an ambiguous gap rung of weak lines (in
   the dry segment, where that ladder ends at n = 4, a ⁷⁹Br/⁸¹Br pair of about the
   same height sits 16-18 ppm below the rung).
   Two rows stay on a rung by the segment rule: C14H15N3O5 [M+NO3]- above and a
   one-file Candidate on HNO3.Br-.(H2O)7. No truth reading of the batch is among
   the strips; the nearest truth row sits 24 ppm (observed m/z) from a passing
   rung, against a 9.2 ppm window. The bright displaced rungs' M+1/M0 is
   0.00-0.05 (at most 3.3 carbon-equivalents) where the C13-C36 readings need
   0.15-0.41. Two low-resolution nitrate TOF batches pass the same rungs as
   before (a three-rung island 25 rungs above an absent ladder is not taken), and
   both Orbitrap batches are unchanged. Measured, the constants have room: the
   link presence at 0.2 or 0.3, and the carbon bar at 4 or 6, give the same rungs,
   and neither the decoy skip nor the carbon test changes a rung of that batch
   (they guard a crowded or a gap rung); co-variation at 0.9 passes three fewer,
   NO3-.(H2O)29 (r 0.87) among them. Lines at the decoy positions co-vary with a
   ladder at median r 0.25-0.33 (15-24 % reach 0.8, on 34 / 48 lines), its rungs
   at 0.96-0.99; over every present off-ladder line at m/z 300-900 against the two
   nearest present rungs the share reaching 0.8 is 0.23 in the humid segment and
   0.40 in the dry one, so the co-variation clause is weaker in a dry segment
   (every gap rung that passes on that batch is in the humid one). A halogen rung
   does not need its other isotopologue: HNO3.⁷⁹Br-.(H2O)6 and 8 pass as gap rungs
   while HNO3.⁸¹Br-.(H2O)6 and 8 are absent in that segment (a ⁷⁹Br/⁸¹Br partner
   test is an open item).

4b. **Known species, decided once** (`lock_known_species`; after the vote,
   before the polarity re-reads). Every file's known-species evidence is pooled
   (`known_evidence`): its `known:` commits with the route read off the row
   (`_known_route`: the commentary's "corroborated by" — ≥ 2 ion channels or a
   diagnostic ²⁹Si/³⁰Si / ³⁴S envelope; else the recorded satellites of an
   element the neutral contains — a chlorinated paraffin's ³⁷Cl envelope; else
   exact mass alone, the monoisotopic families' own rule), and the
   `known_lead` records pass 0 leaves on the claims it refuses
   ([ASSIGNMENT_DETAIL.md](ASSIGNMENT_DETAIL.md) §3.0) — `deferred` when that
   file could not test the claim, or tested it and found the diagnostic lines
   present and consistent though the scorer did not credit them; `refuted` when
   a line the file could show (predicted at 2× the gate) is absent or too small,
   or the own-⁸¹Br-twin ratio failed — judged on the ledger, never on the
   scorer's silence. Per (neutral, adduct), on the merged row within
   `tol_ppm` of the pooled m/z:
   - confirmed in ≥ 1 file and refuted in none → the row takes the known
     reading whatever the count (the confirmed files' best tier / `ion_score` /
     `admitted_by` / `occurrence`; the vote's winner moves to the head of
     `alternatives`; `n_files_winner` / `n_files_ion` = the confirmed files in
     the cluster), and `tier_reason` says so: `known species decided once for
     the batch: tetradecamethylcycloheptasiloxane (D7) (C14H42O7Si7
     [M+(CH4N2O)H]+) -- confirmed in 1 file (corroborated by a confirmed
     29Si/30Si envelope (single channel)); could not test it in 9 files (single
     channel; the Si twin is predicted below the detection floor); kept over the
     9-file C27H30O14 [M+H]+ reading (vote 1 of 10 files)`. A row the vote
     already gave to the species gains the note only. **Two lines for
     Assigned**: a species of a family that demands corroboration keeps the
     confirming files' Assigned tier only when some confirming file holds two
     independent lines of evidence beyond the exact mass — a second ion channel
     of the neutral in that ledger, or two own-element diagnostic satellites
     (`n_channels ≥ 2 or n_satellites ≥ 2`, counted by `known_evidence` from
     the ledger; a reagent's ⁸¹Br twin on a `[M+Br]-` reading is not the
     neutral's); one channel and one satellite line in every confirming file is
     capped at Candidate, locked or kept (`…; capped Candidate (one ion channel
     and one satellite line in every confirming file: two independent lines are
     needed for Assigned)`) — the D7 cyclosiloxane, J12: two files, the urea
     adduct alone, the ²⁹Si line alone, the ³⁰Si line never testable at that
     intensity. Mass-only families are outside this rule (exact mass is their
     own standard).
   - confirmed somewhere and refuted somewhere → left to the vote; the row
     records the conflict (`known species sulfolane (…) confirmed in 1 file but
     refuted in 9 (…); left to the vote`), and where the vote's own winner is the
     conflicted species and it was refuted in more files than confirmed, the
     merged tier is capped at Candidate (`…; capped Candidate (refuted in more
     files than confirmed)`) — one file's Assigned cannot stand for the batch.
     Each group's reasons are the leads' `summary` (the reason without the
     file's numbers), with the file count per reason when there are several.
   - never confirmed → a `known-species lead: …; not locked` note on the row,
     nothing moves.
   - a species whose family's rule is exact mass alone (the perfluoroacids, the
     nitroaromatics, the C0 atmospheric acids, reactive iodine: no twin and no
     second channel demanded) carries nothing to pool beyond the count of files
     it fitted in, so it never overrides the vote — a PFCA `[M-H]-` on-cal in 2
     files of a ~4k TOF cannot displace an 11-file ⁸¹Br-corroborated CHOS
     `[M+Br]-` reading 6 ppm away; the row notes it (`mass-only known species …;
     the vote's 11-file … reading stands`) and the vote stands.
   The merged row a pooled reading belongs to is found by membership — the row
   whose own reading or whose `alternatives` lists it — and by the m/z window
   only for a reading no file committed: a minority reading's own m/z sits
   outside the merge window of a cluster whose mean the majority ion pulls
   6–8 ppm away, which is exactly where a lock matters.
   A reference-list rescue gets no such lock — its corroboration is a ¹³C line,
   which every carbon formula has — and a `certified:` neutral never did.
   `batch_summary.json["merge_gates"]["known"]` records the counts (`pooled`,
   `locked`, `confirmed_kept`, `conflict`, `lead_only`, `mass_only_outvoted`,
   `no_cluster`).

5. **Positive urea re-reads, once per batch.** When `prof.polarity == "+"` two
   gates run on the merged ledger, and each writes its note to the merged row's
   `tier_reason` (the column is always present, `NA` where no gate spoke):
   - `cleanup.relabel_reagent_n_adducts(merged)` — a pure hydrocarbon read via
     `[M+NH4]⁺` / uronium becomes `[M+H]⁺` of the N-heterocycle unless the
     hydrocarbon shows its own `[M+H]⁺` **anywhere in the batch**. The per-file
     stage is switched off for batch runs (`assign_kw["reagent_n_relabel"] =
     False`, honoured by `assign.run`): its skip key is a presence test that flips
     with each file's S/N, which split one ion into two readings across the files
     (C15H22 `[M+NH4]⁺` in 14 files, C15H25N `[M+H]⁺` in the 15th) — a phantom
     disagreement, and a phantom minority for the vote.
   - `cleanup.prefer_amine_over_ammonium(merged, ts_peaks, r_min = amine_r_min
     (0.6))` re-reads uncorroborated `[M+NH4]⁺` as `[M+H]⁺` of the `+NH3` amine
     (mass/isotope-identical, simpler in an N-rich source) unless the adduct's
     trace tracks its parent — done at the **merged** level where cross-channel
     corroboration is complete.
   `batch_summary.json["merge_gates"]` records both gates' counts beside the
   known-species decision's.

6. **The residual stage** (`residual=True`, the default; [`SAMPLING.md`](SAMPLING.md)
   §3b). After the cover's merge and stamp, the universe bins in **no assigned
   file**, unexplained by the stamp and bright somewhere (≥ 5× that sample's
   noise edge, never below the run's gate) are covered by at most 10 more
   samples — each counting for a bin only where the bin stands at ≥ 50 % of its
   maximum — which go through the same per-file path. `align()` then runs
   **once** over every per-file ledger (cover + residual), and the guards, the
   trace reconciliation and the stamp are redone over that. Merged rows carry
   `stage` (`cover` if any cover file holds the ion, else `residual`);
   `selected_samples.csv` gains the picks (role `residual`),
   `tables/residual_bins.csv` the targeted bins, and
   `batch_summary.json['selection']['residual']` the record. Off, nothing changes.

6b. **The TOF ion-M+2 gates** (`iso_checks.tof_m2_gates`; after the stamp and
   the batch isotope checks, TOF-class batches only -- a width model under the
   Orbitrap class's resolving power; without one nothing runs). On a TOF an
   `[M+Br]-` reading's own ⁸¹Br line is the one line its composition
   guarantees, and the per-file twin test never asks for it (the reagent's
   halogen masks the neutral's window). Two demotions, Assigned → Candidate,
   with **no re-vote** (the winner and its reading stay; the row says why in
   `tier_reason`):
   - **REQ** -- the merged winner's pair is refuted by the TOF branch of the
     batch REQ check (the ion's own M+2 line, the reagent's halogen included,
     absent over the batch: `tables/iso_checks.csv`, [EVIDENCE_LEVELS.md](EVIDENCE_LEVELS.md)
     `iso_veto`). This includes a species the known-species lock (step 4b)
     decided: a "known" reading whose own envelope the batch refutes is not
     Assigned (`…; this overrules the known-species decision`). The lock runs
     before the stamp and cannot read REQ, so a lock that displaced the vote's
     winner is **demoted, not undone**: the row keeps the known reading, now
     Candidate, and the vote's reading the lock moved to the head of
     `alternatives` stays there (the note names it). The per-file test usually
     gets there first: a C30 chlorinated paraffin `[M+Br]-` on the
     bromide/nitrate TOF batch (one Br's M+2 line where BrCl4 predicts 2.3×, no
     ¹³C line: the reagent's water cluster at the same nominal mass) was
     Candidate in every file that carried it, so the vote already made it
     Candidate and the lock only noted it.
   - **the ⁸¹Br doublet** (`doublets`) -- the row's line stands at 0.58-1.56×
     (0.6-1.6 × the ⁸¹Br/⁷⁹Br ratio) the line one ⁸¹Br spacing (1.99795 Da)
     below it in ≥ 50 % of the spectra showing it (each line the tallest within
     the merge window): it is that line's ⁸¹Br partner, not an M0 -- unless the
     reading's ion carries Br or Cl whose own M+2 line REQ sees in ≥ 50 % of its
     testable spectra (REQ's seen share `det_frac` over the pair's stamped
     spectra, read whatever REQ's verdict -- a pair stamped in one spectrum can
     be exempt on it). A known-species lock is left to REQ.
   The per-file half of the package is the tier pass's (`tiers.apply_tof_m2`,
   [ASSIGNMENT_DETAIL.md](ASSIGNMENT_DETAIL.md)), so the vote already counts a
   file whose own peak list refutes the reading as a Candidate file. Measured
   offline on the bromide/nitrate TOF batch (the built functions on the run's
   stored per-file ledgers and stamped series, the vote re-run on every touched
   cluster, the known-species lock re-run on it): merged Assigned 355 → about
   287 (the per-file test alone about 306; then REQ about 6 more and the doublet
   13, 11 halogen readings exempt; the two gates alone, without the per-file
   test, would take 46 and 15: 294), the batch's three known-false
   Assigned readings out, all 34 known-true readings kept (one loses two of its
   14 per-file Assigned files), the roster unchanged; on a nitrate-only
   low-resolution TOF nothing moves. `batch_summary.json["merge_gates"]["tof_m2"]`
   records `ran`, `req_demoted`, `known_demoted`, `doublet_demoted`,
   `doublet_exempt` (and `skipped` when it did not run).

7. **Pool the plausibility audit + write artifacts.** Per-file plausibility
   demotes are pooled and written; `merged_ledger.csv` (root), `jitter.csv`
   (tables/), `selected_samples.csv`, and `batch_summary.json` are emitted.
   Just before the merged ledger is written, after the evidence levels are
   stamped, a TOF-class run's Assigned rows take the **mass-only flag**
   (`assignment/mass_only.flag_merged`): `mass_only` / `mass_only_reason` from
   the same per-file ledgers the levels pooled — `True` where no file holding
   the reading at Assigned shows an isotope line of the neutral's own elements
   at 0.5-2x its expected height, known species exempt. An annotation only: no
   tier, level, claim or reading moves ([OUTPUTS.md](OUTPUTS.md),
   `batch_summary.json['tof_flag']`).

8. **Jitter report** (`jitter_report`, standalone analysis). Per-file offset =
   median observed-vs-theoretical ppm of its assignments (`_theo_ppm`). Then:
   - **`by_formula`** — same `(neutral_formula, adduct)` in ≥ 2 files:
     `mz_jitter_raw` (raw ppm spread) vs `mz_jitter_resid` (spread after removing
     each file's offset) + `tier_stable`.
   - **`by_mz`** — offset-corrected m/z clusters exposing **formula
     disagreements** (same peak, different formula across files).

---

## 4. Constants reference

All in `peaky/batch/assign_batch.py`.

| constant | value | role |
| --- | --- | --- |
| `DEFAULT_TOL_PPM` | 6.0 (`= sampling.BATCH_TOL_PPM`) | the BINNING tolerance (the selector's bins, the admission table, the trace index) and the default + floor of the merge window: what a pure `align()` call clusters at, and what `run` clusters at when nothing could be measured |
| `traces.MassScale.merge_ppm` | `max(tol, min(2·tol, 2.5·√2·σ))` — 6 ppm on an Orbitrap, 12 on a TOF | the window `run` clusters at, from the batch's measured per-ion scatter σ (`traces.measure_mass_scale`, once per batch, before the first merge); `traces.MERGE_GAP_SIGMA` = 2.5·√2 = 3.54, `traces.WINDOW_MAX_X` = 2.0 |
| `evidence.VOTE_CLASS_TEXT` | `{2: neutral backed, 1: formula confirmed, 0: unconfirmed}` | the vote class of a per-file reading (`evidence.vote_classes`, carried as `vote_class`) — the ion key before the file count — and the words a vote note prints for it. Private to the vote: never an evidence level, never read through `evidence.claim_class` |
| `TIER_RANK` | `{Assigned:2, Candidate:1}` | the vote's Assigned-file count (a tie-break after the evidence class and the file count) and the best-row pick within the winning reading (then `ion_score`) |
| `lock_known_species` `tol_ppm` / `mz_floor_da` | the merge window / 1.5 mDa | the window a pooled known-species ion is matched to its merged cluster with (`run` passes `MassScale.merge_ppm`; the stamp's mDa floor); a species confirmed in ≥ 1 file and refuted in none takes that cluster |
| `_M0_COLS` | `[mz, neutral_formula, adduct, tier, ion_score, admitted_by, occurrence, ion_only_of, resolvability, sep_hwhm, ts_disposition, ts_cv_norm, vote_class]` | the per-file M0 schema aligned (admission / ion-only / separability provenance, carried for the winning row; the reading's vote class, read by `_vote` and never written to a ledger; absent columns are tolerated) |
| `run` `amine_r_min` | 0.6 | min trace correlation for the positive amine re-read |
| `assign_kw` `reagent_n_relabel` | `False` (set by `run`) | the per-file hydrocarbon-on-N-cluster re-read stands down; `run` applies it once to the merged ledger |
| `run` `k_min` / `k_max` / `min_gain` / `min_prevalence` | 6 / 30 / 0.005 / 2 | passed through to `sampling.select_cover_samples` (see [`SAMPLING.md`](SAMPLING.md)) |
| `run` `residual` / `residual_min_x_edge` / `residual_min_cps` / `residual_k_max` / `residual_frac_of_max` | True / 5.0 / None / 10 / 0.5 | the residual stage (`sampling.residual_universe` / `select_residual_cover`; [`SAMPLING.md`](SAMPLING.md) §3b) |
| `STAGE_COVER` / `STAGE_RESIDUAL` | `cover` / `residual` | the `stage` a merged row carries when the residual stage is on (`align(stages=)`) |

---

## 5. Metrics, defined

- **`mz_jitter_ppm_raw`** — `(max − min)/mean · 1e6` over a cluster's **raw**
  m/z; the total file-to-file mass spread.
- **`mz_jitter_ppm_caldj`** — the same spread over the **offset-corrected** m/z;
  what remains after per-file calibration is removed (the genuine noise).
- **`formula_agree`** — `True` iff the cluster carries ≤ 1 distinct neutral formula.
- **`n_files_ion`** / **`n_files_winner`** — files carrying the winning ion / the
  winning reading; `n_files` is the whole cluster. `n_files_ion < n_files` marks
  a contest between different ions, `n_files_winner < n_files_ion` a label
  chosen by corroboration over a bigger count.
- **`ion_agree`** — `True` iff the cluster carries one ion (element composition of
  neutral + adduct). `ion_agree` and not `formula_agree` = a same-ion label split
  (the reagent-N isobar), not a spectral disagreement.
- **`alternatives`** — the losing readings, best first, each as
  `formula adduct xN tier score` (the best tier / score any file gave it);
  `''` when unanimous. The per-file detail is `tables/jitter.csv`.
- **`tier_reason`** (merged) — what the vote and the batch-level gates did to the
  row: an evidence-over-count ion choice (`evidence outranks the count: kept …
  (neutral backed in 2 of 12 files) over the 9-file … (unconfirmed)`) or a
  corroboration-over-count label choice when one decided the vote; the
  known-species decision; the reagent-N re-read,
  naming the reading it replaced; the amine gate, naming the `[M+NH4]⁺` neutral it
  did not confirm and why. `NA` when nothing needed saying.
- **per-file offset** — median observed-vs-theoretical ppm of a file's assignments
  (`jitter_report`); `offset_spread_ppm` = max − min across files.
- **`mz_jitter_resid`** (by_formula) — ppm spread of one assignment across files
  **after** subtracting each file's offset; the residual peak-position noise.
- **`n_files` / `n_in_all_files` / `n_single_file`** — how widely a merged peak was
  seen; corroboration breadth.

---

## 6. Outputs

| artifact | content |
| --- | --- |
| `merged_ledger.csv` (run root) | one row per m/z cluster: consensus mz, the winning reading, the vote (`n_files`, `n_files_ion`, `n_files_winner`, `alternatives`), `srcs`, `ion_agree`, `formula_agree`, `mz_jitter_ppm_raw/caldj`, the batch-level gates' `tier_reason`, `stage` (`cover` / `residual`; only when the residual stage is on), the pooled evidence level on the evidence scale — `evidence_level`, `evidence`, `would_lift`, `competitors_left`, `tags`, `context`, `context_source`, `claim` (stamped after the vote, never read by it; [`EVIDENCE_LEVELS.md`](EVIDENCE_LEVELS.md) §10.2), plus the trace reconciliation columns (`mz_anchor`, `mz_trace`, `trace_offset_ppm`, `trace_cov_anchor`, `trace_cov`, `trace_moved`, `trace_guarded`, `trace_id`, `trace_role`; [`TIMESERIES.md`](TIMESERIES.md) §9) — **the result** |
| `tables/jitter.csv` | long form, one row per (cluster, file): `cluster`, `src`, `mz`, formula, adduct, tier, `ion_score`, `vote_class` (0 / 1 / 2: the vote class of that file's reading, the key the vote ranked ions by; not an evidence level) |
| `per_file/<sid>_ledger.csv` | each assigned file's full single-sample ledger (audit / re-merge) |
| `tables/selected_samples.csv` | the selected subset in pick order (`pick`, `role` ∈ `cover` / `pad` / `residual`, `bins_new`, `coverage`) |
| `tables/reagent_water.csv` | the reagent-water ladder (step 4a): one row per passing rung — `core`, `iso_tag`, `n`, `ion_formula`, exact `mz` and observed `mz_obs`, the `segments` it passes in with their `presence` / `decoy_presence`, and the merged reading(s) it `displaced`; header only when nothing passes |
| `tables/residual_bins.csv` | the residual stage's targeted bins (`bin_mz`, `prevalence`, `max_cps`, `max_x_edge`, `sample_at_max`, `covered_by`); written when the stage is on |
| `batch_summary.json` (run root) | reagent/context, the `selection` block (k, achieved coverage, stop reason; `residual` sub-block: the funnel, the floor, k, coverage_of_residual, stop_reason, sample_ids), `n_files_by_stage` / `merged_by_stage` when the stage is on, the resolved height gate (`height_cutoff_x_edge` + its source, and the `gate` derivation block), the `admission` block, the `traces` block (re-centred / collapsed counts, per-ion scatter, stamp window), per-file offsets + noise edges, merged tier counts and the claim tallies beside them (`claims`), agreement counts, and `merge_gates` (the known-species decision, the positive-mode gates and `reagent_water`: cores, window, segments, rungs by core, the stripped readings) |
| `jitter_report()` dict | `{offsets, by_formula, by_mz, summary}` — the standalone jitter analysis |

---

## 7. Properties, invariants & gotchas

- **Assign reals, then merge.** A synthetic union spectrum can't be scored
  (`match_compounds` is per-sample), so combining real per-file ledgers is the only
  principled path.
- **Offset correction aligns; raw masses report.** `_mz_adj` is used only to avoid
  splitting a peak by calibration drift — the merged `mz` and `mz_jitter_ppm_raw`
  are computed on raw masses, so the two jitter columns are an honest before/after.
- **Consensus = a vote, in two stages.** The per-file evidence class, then the
  count, picks the ION (tier and score only break ties; nothing is exempt — a
  known species is decided after the vote by pooled evidence), corroboration
  picks its LABEL (the reading Assigned in the most files). The losers stay on
  the row (`alternatives`) and in `jitter.csv` (with each reading's vote class);
  `formula_agree` stays `False` on a split, and `ion_agree` tells a label split
  from a real contest. Frames without a vote class vote by the count exactly as before.
- **The two positive-mode re-reads are merged-level** — the amine gate needs the
  full cross-channel picture, and the hydrocarbon-on-N-cluster re-read needs every
  file's `[M+H]⁺` rows at once; per file, the latter split one ion into two
  readings across the batch. Both say what they did in the merged `tier_reason`.
- **Fetch the batch by name for fresh ids.** `run(batch=…)` re-fetches the
  per-sample list live so the selected ids are valid for `get_peaks` (cached ids go
  stale / 404 when a server copy is renamed).
- **`align`/`merge_union`/`jitter_report` are pure** and offline-tested; only
  `run` touches the network.

---

## 8. Code map

| function | role |
| --- | --- |
| `run` | assign the selected subset, record offsets, align, write run artifacts |
| `_m0` | extract a ledger's M0 rows in the `_M0_COLS` schema |
| `_cluster_mz` | single-linkage gap clustering of an ascending m/z array |
| `align` | offset-aware cluster → the vote per cluster → merged rows + long jitter frame |
| `_ion_key` | a reading's ion (neutral + adduct composition, Hill order, charge sign): the key two labels of one ion share |
| `_evidence_class` | a thin wrapper of `evidence._vote_class_of`: the vote class (2 / 1 / 0) of one reading from the private decision's verdict |
| `_vote` | rank one cluster's ions (regular before ion-only, evidence class, file count, Assigned-file count, best score, text) and, within each, its labels (Assigned-file count, file count, best score, text) |
| `_describe` | one losing reading as `formula adduct xN tier score` for `alternatives` |
| `_curated_neutrals` | the reflist-rescue / known-species neutrals of a per-file ledger (the vote's exemption) |
| `_protected_neutrals` | those plus the certified neutrals (the amine gate's exemption) |
| `merge_union` | just the merged frame from `align` |
| `jitter_report` | by-formula raw-vs-residual spread + by-m/z formula disagreements |
| `_theo_ppm` | observed-vs-theoretical ppm for an assigned (neutral, adduct) |
