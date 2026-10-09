# Changelog

All notable changes to Peaky are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims to
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Precision gates keyed on the run's instrument class: Orbitrap side lobes, merged
  isotopologue rows, element evidence for widened heteroatoms, and the carbon skeleton of
  organonitrates (`--no-sidelobe-guard` and `--no-isotopologue-gate` turn the first two
  off).** Each gate keeps a reading off a line that is not the reading's own, or lets a real
  organonitrate through windows written for CHO skeletons. They read two new runtime fields
  of `PassConfig` (kept out of the config fingerprint): `instrument_class` (`orbitrap` /
  `tof` / `None` = unknown, every class-gated stage off) -- a batch resolves it once from
  its axis class, a TOF roster winning over a width model that reads Orbitrap-class, and
  hands it to every file and worker (the batch isotope checks, the merge-time removal and
  the TOF M+2 gates read the same class, no longer the width model's alone); a single-sample
  run takes the scoring snapshot's type, else a measured width model's class (a declared
  scalar R is a guess: `None`) -- and `trace_sample` (trace-first's synthetic sample, whose
  heights are batch means: no stage that compares lines of one spectrum runs there).
  - **Side lobes, per file before pass 0 (Orbitrap class only).** An Orbitrap peak list can
    hold weak entries 1-5 line widths beside a line at least 50x brighter that the raw
    profile does not have; their offset scales with the parent's width, not with mass (a
    negative lobe at -2.35 +/- 0.1 FWHM), and they are narrower than a line.
    `chem/sidelobes.py` finds them within ONE peak list, in units of the parent's own
    observed width (area / 1.0645 height) or the width model scaled by the file's median
    observed/model ratio; a candidate also needs an artifact signature (narrower than 0.7x a
    line, at the negative-lobe offset -2.6...-2.1 FWHM, or mirrored on the other side).
    `assignment/sidelobe_guard.py` marks those rows `artifact` and locks them. Exempt:
    isotope fine-structure lines (2H / 17O / 15N beside 13C, 13C15N <-> 18O) and a candidate
    at an isotopologue offset of any brighter line of the list (a neighbouring ion's 13C
    line is not a lobe), matched within the isotope tests' exact-offset window at its
    pre-calibration floor (1 ppm; the guard runs before the file is calibrated). A wider,
    width-scaled tolerance (0.3 FWHM, ~2 ppm at m/z 170-230) was measured on a positive-mode
    Orbitrap batch: the 8 peaks it spared over 1 ppm were lobes (a -2.7 mDa lobe population of
    a bright line whose real 13C line is a separate peak inside 1 ppm, a "13C" line at 3.4x its
    prediction, a "29Si" line of a Si-free ion, a lobe the batch's A/B judged a correct
    removal); a nitrate batch flagged the same peaks either way. Skipped on a TOF, an unknown class, the trace sample, without a
    width model or peak areas, and on a file whose widths cannot calibrate the model; a file
    with fewer than 10 bright peaks is tested on the geometric signatures only. Knobs
    `PassConfig.sidelobe_guard / sidelobe_ratio / sidelobe_band_fwhm / sidelobe_narrow`;
    per-file stats `sidelobe_guard`.
  - **Merged isotopologue rows (batch, Orbitrap class).** With a width model and peak areas
    in the time series, `iso_checks.satellite_rows` reads every merged row's line against
    the 13C / 18O / 15N / 34S / 37Cl / 81Br / Si lines of every merged ion below it (and of
    the reagent ions), spectrum by spectrum, within max(1 ppm, 4 sigma). A line whose area
    ratio to its expected share has a median of 0.4-1.4x over >= 3 spectra, with fewer than
    max(1, 10 %) of them above 2x, is the parent's isotopologue: the row leaves the merged
    ledger before the stamp, the stamp gives the line to the parent, and the pooled pair
    reads an isotope-check veto (check `SAT` in `tables/iso_checks.csv`, level 5b). A median
    of 1.4-4x, or an in-band one that fails the tail test, is a mixed line: a note only.
    Known-species decisions, curated readings (the pass-0 registry and the active reference
    lists), ion-only rows and isotope-labelled readings are exempt (a note). A stripped row
    whose parent the element-signature removal below takes out later is kept in the table,
    annotated `parent removed`. New table `tables/isotopologue_rows.csv`;
    `batch_summary.json['merge_gates']['isotopologue']`. The stripped line is a valid
    assignment of the parent's isotopologue, so it is recorded, not dropped: in every
    per-file ledger that committed the stripped reading as an M0 on the line, the row
    becomes the parent's `iso_child` under the gate's label where that file commits the
    parent (a reagent parent's: a reagent isotopologue), and is released to `unexplained`,
    with a note naming the parent and the line, where it does not or where the parent
    later left the merged ledger (`iso_checks.reconcile_per_file`). The rewritten ledgers
    are what `per_file/<sid>_ledger.csv`, the evidence levels and the file's role counts
    in `batch_summary.json` read, so per-file readers and a per-file publish agree with the
    merged ledger, and the rows on the line no longer feed the stripped pair's level. Each parent's merged row
    lists the lines it was given in the new column `isotopologue_lines` (one row per parent
    ion stays the merged ledger's contract; the batch publish is unchanged).
  - **Element evidence: a heteroatom from a widened search needs its own isotope line.** The
    per-peak grid proposes only C / H / N / O, so every S, Cl, Br, Si, P or I formula comes
    from a widened proposer, which proposes a neutral mass and never asks for the element's
    line. `satellites.element_evidence` looks for 81Br / 37Cl / 34S / 29Si / 30Si at the
    exact offset within max(1 ppm, 4 sigma), self-calibrated per file on the 13C / 18O lines
    of its committed CHON rows (a logistic detection curve and the file's own ratio
    percentiles; a file with fewer than 30 calibration lines contradicts nothing). On an
    Orbitrap, a peak that a per-file stage marked `artifact` (a side lobe, or cleanup's
    ringing) is no confirming line and sets no floor. It still leaves its position
    untestable, which is how REQ reads an artifact stamp in the series; the TOF M+2 test
    reads every peak. The new per-file stage
    `element_evidence` (after re-arbitration, before resolvability, degeneracy and tiers)
    CLEARS a non-curated, unlocked M0 whose Br / Cl / Si, S >= 2 or over-cap element is
    contradicted, and a monoisotopic P / I that a widened proposer other than pass 7 put in
    a profile that budgets it at 0 (every class); the cleared peak is locked as unexplained,
    so no later stage gives it a formula. A picked line inside the window under the share a
    sighting needs is reported `low (0.xx of prediction)`, not `absent`. On a TOF, Br / Cl
    are judged by the ion's own M+2 line (with a width model; none: no verdict), with no S /
    Si verdict; an unknown class and the trace sample keep the P / I rule only. Pass 7 on an
    Orbitrap reads the predicate instead of a scorer label, and on every class a two-channel
    `[M-H]-` / `[M+X]-` pair that differs by the reagent acid commits no off-budget P / I
    (nor, on an Orbitrap, an S / Cl / Br / Si winner the peak list does not confirm). The
    siloxane / pdms families take only the run channels siloxanes show (`run_adducts`),
    never an anion cluster such as `[M+NO3]-` or `[M+Br]-`, on which they fitted Si1
    "clusters" onto reagent-cluster lines. Once a file is calibrated, an Orbitrap-class run
    searches the twin test (was 15 ppm) and the residual iso-pair finder (was 8 ppm) within
    max(1 ppm, 4 sigma); before it (pass 0's known-species test) and in a file that never
    calibrates they keep 15 / 8 ppm, since a line can sit more than 1 ppm off there. At the
    merge, on a batch whose resolved class is Orbitrap, a batch REQ veto on an
    element-signature line removes the reading (curated and known-species readings stay):
    `batch_summary.json['merge_gates']['element_signature']`, and the series is re-stamped
    without it. Every per-file ledger that committed a reading that left the merged ledger
    releases the row, and its isotope lines, to `unexplained`, with a note naming the refuted
    line (`iso_checks.release_signature_removed`); a reading another merged row still holds
    (the one a known-species decision marks) reads `reading_left: false` and stays. Per-file
    readers and a per-file publish then agree with the merged ledger, and no pooled pair holds
    the removed reading. The re-stamp reads the rewritten per-file ledgers (after either merged
    gate rewrote one), so no spectrum names a released reading's ion on its isotope lines.
  - **Nitrate Orbitrap runs read the carbon skeleton of organonitrates.** An -ONO2 / -NO2
    group adds N, 2 O and one DBE, so dinitrates, trinitrates and nitroaromatics failed Van
    Krevelen windows written for CHO skeletons. With `ContextProfile.nox_skeleton` on, the
    context filter, the plausibility demotes (O-monster, heteroatom coincidence, carbon
    cluster), re-arbitration, the PDF scrutiny page and the evidence level's space also read
    up to three groups' skeletons (Ceff >= 3), and a formula passes when any reading fits;
    with `small_acid_band` on, C3-C4 polycarbonyl acids get wider windows.
    `contexts.run_profile` (through `assign.run_context_profile`, which the batch's own
    record calls too) switches both on only for the NO3 / NO3_15N reagents on an
    Orbitrap-class run in a context that opens organonitrates; never on a TOF, an unknown
    class or the trace sample. A skeleton-only `[M-H]-` reading yields to its same-ion
    `[M+NO3]-` twin. A row whose outcome the skeleton reading changed is Candidate, never
    Assigned: the reading presumes organonitrate chemistry that MS1 cannot confirm for one
    formula. On NOx-free deuterium-labelled flow-reactor files it had made Assigned N3
    formulas of deuterated products (C2 + D + O is N3 within 0.2 mDa) on peaks left
    unexplained without it. "Changed": a proposer that runs the context filter admitted it
    only on a skeleton reading, or the skeleton spared it the carbon-cluster demote or the
    oxygen-monster demote on a mass-degenerate row, or re-arbitration took it only on its
    skeleton, or it stands on such a row (a ladder fill's anchor, a completion row's
    sibling; pass-2 and residual series children are not followed). Rows from
    proposers that never read the windows (pass 6, completion, pass-7 certificates, a pass-3
    family a GKA series opened, which now commits as `contaminant:<family>:gka`) are judged
    by the demotes and their anchors alone. Curated readings keep their tier. `tier_reason` says how, per file and on the merged row;
    `stats['nox_skeleton_gate']` and `batch_summary.json['nox_skeleton_gate']` count it. On
    the NOx-chemistry nitrate batch below, 59 skeleton-reliant merged rows, all judged right,
    are Candidate instead of Assigned (same ions: recall and the expert comparison do not
    move; one merged row now reads its ion as the nitrate cluster its 1-microscan files hold,
    not as a deprotonated molecule); on the deuterium-labelled files, Assigned formulas with
    N >= 3 fall from 47 back to 31, the count before the skeleton readings. The switches
    are recorded in `batch_summary.json['context_flags']` and each file's stats
    (`context_flags`); a trace-first batch whose residual files ran with other switches says
    so under `batch_summary.json['warnings']`.
  - **The (HNO3)2.NO3- dimer core is a reagent row** (rung n = 0 of the reagent-water
    ladder) when it passes the rung presence test, on any instrument class; it has no
    analyte reading, so a merged reading on it leaves the merged ledger as on any rung.

  The run manifest's counts carry `context_flags` and the isotopologue gate's state (a run
  argument, not a config field). On a TOF the side-lobe guard, the isotopologue gate, the
  narrowed windows, the merge-time removal and the skeleton readings are off, but four
  pieces ARE active: the element gate's M+2 Br / Cl test, pass 7's two-channel P / I rule,
  the siloxane adduct restriction and the (HNO3)2.NO3- reagent core (and, as on every class,
  the element gate's off-budget P / I rule). End to end on a private nitrate Orbitrap batch
  checked against its raw profiles, with all four gates together: Assigned precision 97.5 ->
  99.3 % (539 rows; 598 before the skeleton tier cap), Candidate precision 80.4 -> 98.3 %
  (97.7 % before the cap), and the recall of ions the raw profiles judged
  real 82.3 -> 90.2 %; against an independent expert assignment of 935 ions the same formula
  rose from 693 to 759 and a different formula fell from 10 to 1, and on the batch's
  1-microscan files from 421 to 463 and from 8 to 0. In A/B runs on a positive-mode Orbitrap
  batch 12 merged rows left, each judged correct; on a TOF NO3/Br batch with a truth set the
  truth recall went from 32/34 to 33/34 with false readings staying at 0, and 166 merged
  rows left (163 of them Candidates), most of them exotic P, PS, Cl, Si or Br-bearing
  readings. These numbers were measured before the last review fixes above (the class source
  of the merge-time removal, the calibrated twin window, the lobe exemption at isotopologue
  offsets, side-lobe artifacts no longer confirming an element, the locked clear, the
  curated exemption of the isotopologue gate, and the artifact rule of the evidence scale
  and of the element-evidence test). Only the positive-mode batch was re-run with all of
  them; replayed with the 1 ppm exemption window it keeps all 12 removals, and the artifact
  rule moves no level there. The nitrate batch, re-run end to end with every fix and the
  skeleton tier cap, reproduces the numbers above exactly (both its 100- and 1-microscan files).
- **The batch's m/z axis is modelled from its own peaks (`--mass-axis auto`; `locks`,
  `reference` and `off` force a path).** On an Orbitrap the correction below no longer rests on
  the reagent's ~25 reference ions: `batch/axislock.py` finds calibration-free LOCKS -- recurring
  peaks whose +-5 ppm window holds one closed-shell formula once the peak's own 13C / 34S /
  37Cl / 81Br lines strike the rivals, then peaks one exact unit step (CH2, O, CO2 ...) from two
  locks; isotopologues of a brighter peak and elements no isotope line confirms (F, a
  non-reagent I) never lock -- fits a smoothing spline through their main body, drops locks off
  it, adds +-1 ppm locks on the corrected axis, and walks up the range segment by segment to
  model the STEPS an Orbitrap's axis can take (measured in the instrument's own centroids:
  -2.6 ppm between two adjacent peaks, +2 ppm 30 Da higher). Between two segments across a
  step, and past the first and last lock, nothing is corrected: there the step's position is
  unknown. It is applied when it predicts held-out locks (5-fold) within 0.3 ppm or half the raw
  error and some lock reads 1 ppm off, up to 7 ppm; the reference-ion wave stays the fallback
  when the locks build no model, and a TOF is still only measured. On the nitrate Orbitrap batch below, ~480
  locks span m/z 57-423 and held-out locks sit 0.11 ppm from the model (1.6 ppm raw); against
  the independent assignment the same formula rose from 573 to 693 (m/z 365-423: from 0
  to 62), a different formula fell from 11 to 10 of 935, and the ions peaky alone
  holds fell from 40 to 26; on the batch's 1-microscan files, 387 -> 421 and
  9 -> 8. `tables/mass_axis_locks.csv` lists every lock and how it was found.
- **`peaky batch` / `pool` measure the batch's m/z axis and correct an Orbitrap's axis error
  before anything is assigned (`--mass-axis auto`, the default; `off` skips it).** The batch time
  series is probed against the reagent's formula-certain reference ions (the `peaky mass-qc`
  probe, judged by the instrument's own rules); on an Orbitrap whose verdict is an axis error
  the time series and every file's peak table are corrected, in the parent and in every spawned
  worker: a trend inside the reference ions' m/z range (left as is outside it, where its shape is
  unmeasured), a flat offset at every m/z. Pass 1's per-file self-calibration models only a
  constant plus a 1/(m/z) term, so an axis that rose and fell across the range left correct
  formulas several scoring widths off-centre: on a nitrate Orbitrap batch whose axis read 0 ppm
  at m/z 62, +2.4 ppm near m/z 200 and +0.2 ppm at m/z 308 (identically in every file), the
  uncorrected run held 307 ions and put P/S/Cl formulas on CHON(N) ions; with the correction it
  holds 641, and against an independent, hand-curated assignment of the same ions the same
  formula rose from 171 to 573 and a different formula fell from 57 to 11. Measured but not
  corrected: a TOF (its reference ions span m/z 62-220 and its offsets are mostly
  ion-specific; a batch the server lists as TOF stays one whatever resolving power it is
  given), a pool spanning several batches (one wave would correct each batch by the others'
  axis), server-side scoring (`PEAKY_LOCAL_SCORING=0` scores the server's own peaks) and a
  correction above 4 ppm anywhere (a calibration to fix at the instrument). `peaky publish
  --batch` puts a corrected run's merged m/z back on the server's axis before matching it to
  the batch peaks (within 5 ppm). A reagent without a reference-ion
  table (nitrate, labelled nitrate and bromide have one) is not measured, and the measurement
  never stops a batch. `batch_summary.json['mass_axis']` and the run manifest record the
  verdict, the wave, its scope, how far the peaks moved or why nothing was applied;
  `tables/mass_axis.csv` the reference ions. A corrected series carries the shift it removed
  (`mz_axis_ppm`), so a run's `per_file/_batch_ts.parquet` fed back with `--ts` is restored to
  the server's axis first (`peaky mass-qc --ts` restores it too). The cluster and Van Krevelen
  figures read the corrected series; the PDF's coverage tables and a pool's per-group reports
  read the run's recorded input (the series it was given) within their 8 ppm windows. The
  correction is held per sample: runs that share samples in one process (the MCP server's
  threads) share it. Trace-first keeps its own wave.

### Fixed

- **The evidence scale reads no isotope line off an artifact row.** The line probe used to
  take the tallest peak in a line's position window whatever its role, and it read an
  artifact there as an occupant. Its 25 ppm reach for a displaced line could also pick an
  artifact. So a side lobe beside a reading's 13C line, taller than the line, made the line
  "occupied" (present but never matched), and the reading lost its 13C positive fact. The
  probe now picks the tallest other peak, and the reach skips artifacts. A window only
  artifacts hold is untestable (`shadowed`), never absent. The exact 37Cl probe and the
  isotope-line competitors' position follow the same rule. It is the rule the
  element-evidence predicate and REQ follow: an artifact confirms no line and refutes none.
  On the two level fixtures no level moves; the evidence text of 4 pairs does (the expected
  table is re-pinned). A reading whose 13C line is an unresolved doublet with a lobe now
  has that file untestable rather than matched off the lobe, so it can fall short of the
  three testable files a positive fact needs.
- **`peaky mass-qc` measures batches of few spectra.** A reference ion needed 10 spectra to
  count, so every batch of fewer (a time series of one aggregated spectrum per file) read
  NO_REFERENCE although the ions sat in every file. The bar is now what half the batch is,
  at least 3 and at most 10: unchanged from 20 spectra up.
- **`peaky mass-qc --orbitrap` judges an Orbitrap by its own rules.** The calibrant ladder
  and thresholds were sized on TOF batches: the anchor tier (an ion clean at TOF resolution)
  stops at m/z 220, the reagent / lock-mass ions sit off the analyte axis, and a 4 ppm swing
  or 2 ppm offset is ~10 Orbitrap scoring widths. On the batch above it reported a flat
  +2.1 ppm offset. On an Orbitrap every usable reference ion now calibrates except those
  over 100x the median height, and a trend needs a 1 ppm swing that predicts held-out ions
  with half the constant's error (an offset: 1 ppm). A TOF is judged as before.

## [0.10.0b1] — 2026-10-07 (pre-release: the evidence scale, the isotope checks, the m/z-dependent mass centre, declared side channels, the TOF M+2 and mass-only gates)

### Changed

- **Side channels are declared per reagent and closed by default; `--side-channels` opens them.**
  `assign.run` opened [M+CO3]- and [M+Br2]- on every negative run and [M+Na]+ / [M+NH4]+ on every
  positive run whenever the server listed the mechanism (Mascope 1.10 lists them all), and nothing
  closed them: no flag, profile field or config knob, and a spawned worker re-resolved them on its
  own client. On the batches measured the polarity-wide set mostly re-read ions the run already
  held under another reading, or had no support of their own: on a labelled-nitrate Orbitrap batch
  72 of 77 Assigned [M+CO3]- readings were the ion the closed run holds as a radical-anion
  Candidate (the carbonate cluster of the C(n-1) neutral is the radical anion of the Cn neutral),
  and in the 45 of the 77 where the readings' neutrals had lines to test, the carbonate
  reading's neutral co-varied better than both the radical-anion and the superoxide reading's
  in 16, never at r >= 0.85, and less well than one of them in 29; on a bromide/nitrate TOF
  batch the CO3- ion was present in 2 of 230 spectra and 20 of the 22 [M+Br2]- readings with a
  testable M+2 line failed it; on a uronium batch none of 8 [M+Na]+ readings tracked a partner,
  while 54 of 78 [M+NH4]+ readings tracked their own [M+H]+ / urea parent at r >= 0.7.
  `ReagentProfile.side_channels` now declares them: the uronium profile [M+NH4]+, which the batch's
  amine gate keeps (it tracks its parent) or re-reads as the protonated amine at Candidate, every
  other built-in profile none; a `--reagent-config` profile declares its own and `compose` unions
  them. `--side-channels ADDUCT ...` on `peaky assign`, `batch` and `pool`
  (`PassConfig.side_channels`, `pipeline.run_batch(side_channels=)`) opens exactly the channels
  named instead, `--side-channels none` closes them all; a channel of the other polarity or
  without a server mechanism is skipped and logged, and a labelled-ammonium run keeps [M+NH4]+ and
  [M+Na]+ closed. The choice is a `PassConfig` field, so it reaches every spawned worker and the
  run manifest. Each file's stats record the channels it opened (`side_channels`), and
  `batch_summary.json` the union, the files per channel, the requested set and its source; an
  offline sample registers the side channels its run opens, so the scorecard's arms on the run's
  own channels open what the run recorded (the wrong-adducts arm reads its wrong set alone).
  Runs of every profile but uronium now run in the configuration the closed measurements used.
  Offline on a uronium Orbitrap batch, with a stand-in server listing every mechanism, the default
  opened [M+NH4]+ in all 10 files and nothing else; merged Assigned went 723 -> 648 (79 of them
  [M+NH4]+, the amine gate keeping 78 readings it confirmed by co-variation with their parent,
  2-h bins at r >= 0.6, and re-reading 67) and the batch's truth
  rows read 35 of 42 TRUE and 0 of 8 FALSE against 29 and 4 with `--side-channels none`, which
  reproduces the measured closed run reading for reading. A tier rule that would also follow the
  NH4 admissibility rule was measured and not built: it changes no Assigned reading on that batch
  and would cap 3 of 239 Assigned ammonium readings on a second uronium batch, among them the
  known-species adduct of a cyclic siloxane and a reading the amine gate found tracking its
  parent.
  docs/REAGENTS.md section 3b, README, SKILL.md, docs/OUTPUTS.md, docs/SCORECARD.md.
- **`--reagent auto` stops with an error instead of guessing a profile.** When none of the peak
  table's server matches is a reagent's diagnostic adduct (no matches at all, or only generic
  ones such as [M+H]+ / [M-H]-), `profiles.resolve` used to return the first registered profile
  of the guessed polarity -- bromide for every negative guess, uronium for every positive one.
  It now raises, naming the mechanisms it saw, their polarity, the signatures auto-detect reads
  and the known reagents, and asks for `--reagent`; `peaky batch`, `peaky pool` and
  `peaky assign` print that reason and exit 1 before any assignment runs, and an MCP job ends in
  an error carrying it. Detection reads the server's matches without `detect_adducts`' [M-H]-
  default (new `io_mascope.recognised_adducts`), so an unmatched table cannot select a profile
  that declares that channel. A table with a diagnostic adduct resolves exactly as before:
  offline on the time-series inputs of a labelled-nitrate Orbitrap batch, a uronium Orbitrap
  batch and a bromide/nitrate TOF batch, auto still gives NO3+NO3_15N, Ur and Br+NO3. From
  Python, `assign_batch.run(batch=NAME)` with its default `reagent='auto'` no longer resolves on
  the batch's sample roster, which carries a polarity but no server match (a positive roster
  gave Ur and a negative one Br, whatever the reagent): it reads the matches of `peaks=`, else
  of `ts_peaks=`, and with neither raises asking for `reagent=NAME` or the per-peak time series.
  `peaky batch` / `peaky pool` already pass the resolved name.
- **`peaky publish-batch` sends the merged rows peaky holds Assigned, not every row.** A batch
  import row carries the m/z, the neutral and ion formulas and the mechanism id but no verdict,
  so every merged Candidate landed in Mascope's batch ledger indistinguishable from an Assigned
  reading (a bromide/nitrate TOF batch: 2536 rows sent for 355 Assigned; two Orbitrap batches:
  1392 for 514 and 1083 for 723). `publish.build_batch_rows` now leaves the other rows out
  (counted by peaky's tier in `held_back`, and printed by the command) unless
  `--include-candidates` (`include_candidates=True`), which sends them with a warning; the run's
  config records the sent rows by peaky's tier (`published_tiers`), since the rows cannot say
  it. A merged ledger with no `tier` column is refused unless every row is asked for. When rows
  are held back from a ledger levelled before the evidence scale, the command says that its
  "older" count and the config's claims tally cover the whole merged ledger, not only the rows
  sent. The ledger, the tiers and the per-sample payload are unchanged. README and
  docs/PUBLISH.md.
- **README: the Validation section says how a release is validated, not an older engine's
  counts.** The merged M0 and tier counts it quoted came from two batch runs of an engine that
  has since changed its tiers and gates and gained the evidence scale. It now describes the
  yardsticks without numbers: a frozen per-run truth set of TRUE / FALSE readings checked
  against the raw spectra with evidence computed outside the engine; mass-shift and wrong-adduct
  decoy arms re-run offline on the run's brightest cover file(s), the shift arm estimating how
  often a mass with no true formula is still Assigned, reported below and above m/z 350 because
  a 0.35 Da shift reads near zero below ~350 by construction; and the scorecard's isotope checks
  (13C carbon count, heteroatom lines, satellite and partner co-variation), with a pointer to
  `scripts/scorecard.py`. This release's measured numbers follow its validation run.
- **The `neutral` claim is defined among the run's declared reagent channels.** A 4a reading is
  pinned with the side channels (formate, acetate, chloride, ...) locked, and on a
  labelled-nitrate batch a family of ten C11/C12 `[M-H]-` readings carrying a quarter of the
  committed M0 height is claimed neutral while the own 15NO3- cluster of its three brightest
  members is seen in 0, 0 and 6 of 305 spectra. `CLAIM_MEANING["neutral"]` (the workbook's By
  claim, Summary and Read me), the PDF Claims page legend and notes, the README,
  `docs/OUTPUTS.md`, `docs/ASSIGNMENT.md`, `docs/SCORECARD.md` and the claim table of
  `docs/EVIDENCE_LEVELS.md` now say that the neutral is established among the declared channels
  and that a side channel the run keeps locked (e.g. formate or acetate on a nitrate source)
  could re-read an `[M-H]-` ion as a cluster of a smaller neutral, which the row's evidence then
  names ("side channels locked"). No level changes.
- **The 15N-label stage no longer spends a minute per file copying one list.**
  `chemistry.candidates_for_peaks` rebuilt the memoised grid's mass column on every call; on the
  label stage's box (213,758 formulas) that cost 20-40 ms a call, ~3800 times on one
  labelled-nitrate Orbitrap file, so the stage took 73-113 s per file (31 % of the per-file
  stage time on that batch) while filling nothing. The grid cache now keeps the mass column with
  the grid (`_grid_and_masses`; `_grid_cached` still returns the grid). Measured offline with a
  stub scorer, the label stage's candidate enumeration drops from 63-79 s to 0.06 s per file
  with an identical scorer input (the stage's one scoring call remains and is not in that
  figure). No caller's candidate set changes (0 of 3816 label-stage calls, and 0 of 3603 calls
  over the residual, cleanup, siloxane and chain-head boxes on a labelled-nitrate, a uronium and
  a bromide/nitrate TOF batch). (`tests/test_chemistry_grid.py`.)
- **`scripts/scorecard.py`: a delta is taken only between two rows measured alike.** A metric
  that reads the evidence scale needs the same scale, a roster count the same presence test, a
  decoy rate the same arm calibration and scoring, the headline below m/z 350 the same headline
  arm (ppm or Da) and a ppm-arm rate the same shifts; otherwise the previous value is withheld
  and the card's delta table, the page's and the board's claims table say
  `not comparable: <why> (previous <value>)` -- 20 roster formulas identified on the pre-0.10.0
  scale and 2 on 0.10.0 is a change of scale, not a regression.
- **Three rule changes of the evidence scale (decided after
  the scale landed).** (1) `lowconf` alone -- every row of a pair Low / Suspect and no other
  rejection -- is a **5a ceiling, not a 5b rejection**: the pair is levelled on its facts
  (untestable stays 5b, competitors left stay 5a) and reads at most 5a, `engine confidence
  Low/Suspect in every file: not established (5a ceiling; a file at Good or High lifts it), not
  refuted` (tag kind `lowconf 5a ceiling`); it still anchors nothing (no member of the series
  exclusion, no route, no ladder). The flag tracked the calibration centre more than the
  chemistry (the mass-dependent recentring alone moved 40 pairs out of it and 13 in). (2) **A contaminant
  family opens the run's space only from >= 2 files** (`levels.context.FAMILY_MIN_FILES`,
  `family_union`; every family on a one-file source; the families one file alone opened are
  kept on `RunContext.families_dropped`), the scale's other 2-file minima: one Candidate row in
  one file of the uronium run had opened `fluorinated` for all of its 1196 pairs. (3) **A
  reference list that rescued a dim reading cannot also certify it at 3c**: a tentative lead
  whose setter is `reflist_dim` takes the level its other facts give (tag kind `3c withheld
  (list rescued the lead)`; at 4a `would_lift` says so). `scripts/level_ledger.py` carries the
  same three rules (parity with the core, row for row). Tiers and the merge vote are untouched,
  so no tier, claim-by-tier or truth-set number moves. Measured post hoc on two Orbitrap
  validation runs against their levels before the change: labelled nitrate 86 pairs 5b -> 5a and the two
  reflist-rescued 3c pairs (BHA C11H16O2 and dimethyl phthalate C10H10O4 [M-H]-) 3c -> 4b,
  nothing else (vector 1906 10/161/256/395/1084 -> 8/161/258/481/998); uronium 174 pairs 5b -> 5a, 128
  5a -> 4b, 2 5a -> 4a and C6H15O4P [M+H]+ 5a -> 3c (1196 16/55/701/239/185 -> 17/57/829/282/11:
  the 5b bucket is 11 pairs). The level fixtures move with it (`tests/fixtures/levels/
  expected_levels_v1.csv.gz`, re-cut: nitrate 1850 9/186/290/500/864/1/0, uronium 1161
  17/75/913/140/16/0/0) and so do the golden vectors kept outside the repository.
  docs/EVIDENCE_LEVELS.md sections 2.4, 3.2, 4, 5.2, 7, 9, 11 and 12.

### Fixed

- **A `--reagent-config` profile reaches the batch's parallel workers.** `peaky batch` / `pool`
  with `--reagent NAME --reagent-config FILE --jobs N` (N > 1) registered the profile in the
  parent only; the assign pool's workers are spawned interpreters whose registry holds the
  built-ins, so the per-file evidence stage's lookup of the run's reagent raised
  `KeyError: unknown reagent 'NAME'` in every worker (`--jobs 1` ran). The pool now hands each
  worker the profiles and aliases the parent added or replaced (`profiles.registry_extras`,
  registered by `assign_batch._worker_init`), however they were registered.

- **The tier pass survives nullable string columns.** `tiers` read its string columns as
  `str(v or "")`, which raises "boolean value of NA is ambiguous" on `pd.NA`, and a ledger
  carries `pd.NA` whenever rows were appended without a column or it was read back with
  nullable string dtypes; `compute_tiers` then stopped in its calibration step on the first
  row whose neutral formula was missing (0.9.0 died the same way on a `--reagent NO3_15N`
  batch, where the labelled-reagent rescue leaves `admitted_by` unset). `tiers._txt` is an
  NA/NaN/None-safe `str()`, and the module's ledger string reads go through it
  (`admitted_by` was already read null-safely).

- **The bromide machinery reads the declared channels, not an opted-in halogen side channel.**
  `has_halogen_adduct`, the switch of the halide reagents' even-shift composite stage, was
  computed after the side channels joined the run's adducts, so an [M+Br2]- side channel on a
  nitrate run turned the de-blend on: on a labelled-nitrate Orbitrap batch run with the
  polarity-wide channels open, 162 composite splits over its 12 files (none in the closed run)
  and 8 synthetic co-component rows Assigned in the per-file ledgers, while the merged list did
  not change. The reagent-cluster library key (and with it the reagent element) and pass 3's
  cluster key read the joined list too, so the same side channel loaded the Br library (in an
  offline probe its Br-, 81Br- and Br3- peaks became reagent rows), gave the arbitration and the
  cleanup's bromide-cluster stage a Br reagent element, and handed pass 3 the Br key, which runs
  its HBr-cluster resolution and opens the bromo-organic family; an opted-in [M+Cl]-, [M+I2]- or
  [M+I3]- did the same for its halide. All of them now read the declared analyte channels,
  the footing the reagent halogen already had; a bromide or iodide run, and every run with its
  side channels at the default, is unchanged.
- **A certified neutral whose ion score is under the engine's Suspect band edge is Candidate.**
  A pass-7 certified neutral earns its tier from the channels' convergent neutral mass, and pass 7
  labelled it 'Good (certified)' with no score floor. The scorecard's ppm decoy found wrong
  certified P formulas Assigned on spectra shifted a few ppm off their true formulas, at ion
  scores far under those of the certified commits the unshifted controls Assigned (all >= 0.74).
  A `certified:` commit whose `ion_score` is under `PassConfig.tau_suspect` (0.50, the engine's
  existing Suspect band edge) is now Candidate, with a reason naming the score and the edge
  (`tiers.LOCKED_SCORE_METHODS`, `tiers.lock_score_floor`). A certificate is multi-channel by
  construction, so its other channels spare no member; a ladder rung carries its weakest anchored
  member's score; a row with no recorded score is not judged. Pass-0 known species are not
  floored: the decoy also Assigned known-species locks at low scores, but real known species score
  in the same range (NO2- and HSO4- on an iodide batch, ethanol on an NO+ batch, cyclosiloxanes
  with their own 29Si/30Si lines: 0.03-0.48, against 0.001-0.49 for the decoy locks), so a floor
  there removes real readings as fast as decoys. A known species the batch locks or keeps while
  no confirming file holds it at Assigned now says so on the merged row. Offline (per-file
  ledgers re-tiered, decoy arms re-tiered): no real per-file Assigned reading of the checked
  batches is demoted; uronium +-6 ppm decoy Assigned 41 -> 35 (2.61 % -> 2.23 % of 2 x 784
  matched control readings), the uncalibrated uronium -6/+6 arms 47 -> 29; the labelled-nitrate
  decoy is unchanged (its low-score decoys are known-species locks). docs/ASSIGNMENT_DETAIL.md
  §1.6 and docs/CERTIFIED_NEUTRAL.md.
- **On a TOF the reagent-water ladder reads its high rungs as water clusters, not organics.** The
  ladder gate judged each rung against decoys at fixed Da offsets (+-0.02 / 0.035 / 0.05) and ended a
  ladder at its first weak rung. At m/z 450-700 a TOF line is 0.05-0.07 Da wide: the offsets sit
  inside it, on the weak satellites a TOF picker reports beside a bright line (about a tenth of its
  height, ~1-1.5 FWHM below and ~0.5-1 FWHM above), and in a humid stretch one weak rung ends the
  contiguous ladder. On a bromide/nitrate TOF batch eight Br-.(H2O)n rungs between n = 21 and 35,
  NO3-.(H2O)29-34 and HNO3.NO3-.(H2O)27/31 stayed Assigned as C18-C36 organics -- a "known"
  C30H58Cl4 [M+Br]- was Br-.(H2O)31, its M+1 line at 0.02x where C30 needs 0.33x. When the batch's
  width model is TOF-class (R < 50 000 at m/z 200) each rung now takes a TOF rung test per segment:
  present in half the spectra; at least 3x the mean presence of up to twelve decoys at +-2..5
  FWHM(m), skipping a decoy that sits on another declared core's ladder; on its ladder, which runs
  through every rung present in a quarter of the spectra and whose rungs beyond the contiguous part
  must co-vary with their present neighbours (median r of log height >= 0.8 over >= 8 shared
  spectra); and, where the rung is bright enough to show a C5 line's 13C, an M+1/M0 within the
  cluster's own plus five carbons (not asked where the M+1 line would blend with a rung of another
  declared core present in that segment). After the TOF test a merged row on a passing rung leaves
  the merged ledger only when a file carrying its winning reading lies in a segment where the rung
  passed; a reading carried only by files of other segments is another line at that m/z and stays,
  with a note on its `tier_reason`. An Orbitrap-class width model, or none, keeps the fixed-offset
  test and the batch-wide strip exactly. `batch_summary.json` records the TOF test under
  `merge_gates.reagent_water.rung_test`. Measured offline on the batch's own time series and
  per-file readings: 121 -> 181 passing rungs, every earlier rung kept; the 72 rows stripped before
  are stripped still and 54 more leave, 19 of them Assigned -- 18 of the batch's 19 Assigned organic
  readings at a water-cluster m/z (the 19th, at HBr.Br-.(H2O)15, has no declared core and no ladder
  in the data) and C4H11NO3S [M+Br]- on HNO3.Br-.(H2O)5, an ambiguous gap rung of weak lines.
  C14H15N3O5 [M+NO3]- stays: it sits on Br-.(H2O)16, a rung of the humid segment only, but its
  reading comes from dry-segment files, where the line has no 81Br twin. No truth reading of the
  batch is stripped; the nearest truth row sits 24 ppm (observed m/z) from a passing rung, against a
  9.2 ppm window. Two low-resolution nitrate TOF batches and both Orbitrap batches pass the same
  rungs as before.
  docs/MERGE.md step 4a, `tests/test_reagent_water.py`.
- **On a TOF an Assigned reading must show its ion's own M+2 line.** A `[M+Br]-` reading was
  never asked for its own 81Br line: the per-file twin test asks for the neutral's Br / Cl / S
  line, and on a bromide adduct the reagent's halogen masks that window, so the test stood down
  exactly where the ion's composition guarantees a line at ~0.97x the parent. A new primitive,
  `satellites.heavy_line_verdict`, predicts the ion's whole nominal M+2 cluster
  (`isotopes.nominal_cluster`: 81Br, 37Cl, 34S, 18O, 13C2 ..., one line at R ~10 000; the reagent
  adduct's halogen included), tests it where it reaches the counting-detector floor (3x the
  batch's detection edge), sees it at >= 0.6x within max(15 ppm, 3x the file's fitted sigma),
  and reads it untestable where another line within one FWHM is at least as tall or the lines
  within half a FWHM could be a split of it. The new assign stage `tof_m2`
  (`tiers.apply_tof_m2`), the last tier word on a TOF, makes an Assigned M0 whose line is absent
  Candidate and says which line, its predicted height and what the row was otherwise Assigned
  on; known species are tested like any row. It only demotes and sets no flag, so the merge
  vote's class is untouched; the run's stats record its counts (`tof_m2`). Offline on a
  bromide/nitrate TOF batch: 153 per-file Assigned readings demoted, merged Assigned 355 -> about
  306 on its own. Off a TOF, without a width model or without a detection edge it does nothing (two
  Orbitrap batches: no row moves).
- **The batch REQ check on a TOF reads the ion's own M+2 line with the same primitive.** Its TOF
  branch asked for halogen lines only where they were predicted at 3x each spectrum's height
  gate (15x its edge on the bromide/nitrate TOF batch: 2848 of 3273 Br / Cl pairs untestable),
  took any peak inside the stamp window or 20 ppm as the line, and had no blend guard. It now
  judges the ion's own M+2 line in every spectrum showing the pair with
  `satellites.heavy_line_verdict` -- floor 3x the batch's detection edge (`iso_checks.measure`
  takes it as `edge_cps`), window max(15 ppm, 3x the batch's per-ion scatter), both blend
  guards -- and refutes a pair over >= 10 testable spectra, at least as many as the blended
  ones, where the line is seen in < 30 % of them; `occupied` is the share of its spectra whose
  M+2 position is blended. On that batch REQ now refutes 993 of 3273 pairs (117 before); the
  Orbitrap branch and the Orbitrap batches' tables are unchanged byte for byte.
- **On a TOF a merged winner its own M+2 line refutes is Candidate -- a known species
  included -- and an 81Br doublet partner is no M0.** The batch REQ veto never reached a TOF's
  tier, `lock_known_species` (which runs before the batch's isotope checks) could keep a
  "known" reading Assigned that they contradict, and a merged line standing at ~1x the line one
  81Br spacing below it was Assigned as an M0 of its own. `iso_checks.tof_m2_gates` runs after
  the stamp on TOF-class batches only and demotes, with no re-vote: a winner whose pair REQ
  refutes -- a known-species decision included, and the row says so; a lock that displaced the
  vote's winner is demoted, not undone (the row keeps the known reading, now Candidate, and
  names the vote's reading at the head of `alternatives`) -- and a line at 0.58-1.56x the line
  1.99795 Da below it in >= 50 % of the spectra showing it (`iso_checks.doublets`), unless its
  reading's ion carries Br / Cl whose own M+2 REQ sees in >= 50 % of its stamped spectra.
  `batch_summary.json["merge_gates"]["tof_m2"]` records the counts. With the per-file test,
  offline on the bromide/nitrate TOF batch: merged Assigned 355 -> about 287 (133 -> 85 at m/z
  >= 350; on top of the per-file test REQ about 6 and the doublet 13, 11 halogen readings
  exempt); its three known-false Assigned readings out; a "known" C30 chlorinated paraffin
  `[M+Br]-` (one Br's M+2 line where BrCl4 predicts 2.3x, no 13C line: the reagent's water
  cluster at the same nominal mass) Candidate too, the per-file test already refuting it in
  every file that carried it; all 34 known-true readings kept; the roster's recall unchanged.
  A nitrate-only low-resolution TOF does not move (282 -> 282). docs/MERGE.md section 3 step
  6b, docs/ASSIGNMENT_DETAIL.md, docs/OUTPUTS.md.
- **`--reagent auto` no longer reads the polarity from the batch or sample name.** The polarity
  guess looked for a bare '+' or '-' in a `polarity` column, then in the batch name, and only
  then in the mechanisms, so a dated batch name ("<date> - <date>") read as negative: a positive
  uronium batch whose matches were blank, or only [M+H]+ / [M+Na]+, resolved to the bromide
  profile with no warning, and a name carrying '+' sent a negative batch to the uronium profile.
  Polarity now comes from the charge each server mechanism carries in the standard notation (the
  legacy '-H+' reads negative), then from a `polarity` column's words; names are never read. On
  the three real inputs above with only their (de)protonation matches kept, all three -- the
  positive uronium batch included -- resolved to Br before and now stop, naming [M+H]+ /
  positive and [M-H]- / negative.
- **`peaky pool` with its default `--reagent auto` works.** The pooled table was trimmed to its
  four time-series columns before the reagent was resolved, dropping the `ionization_mechanism`
  column auto-detect reads, so every default pool run raised "could not auto-detect reagent".
  The reagent is now resolved on the full table, as `peaky batch` does; the workers still get
  the trimmed table. On the three real inputs above the pool path now hands the assignment
  NO3+NO3_15N, Ur and Br+NO3 (before: an error on all three).
- **`peaky assign` and the MCP `assign_sample` tool no longer fall back to [M-H]- when
  auto-detect fails.** `peaky assign` caught the failure, printed a note and let `assign.run`
  read the channels per sample, which defaults to [M-H]- in negative mode when nothing is
  recognised; the MCP tool skipped auto-detect altogether and always took that per-sample path
  (bare matched channels, ambient-air context). Both now resolve the profile from the sample's
  own matches -- the MCP tool thereby takes the detected profile's channels, context, label and
  purity, as `peaky assign` does -- and stop with the resolver's reason when that fails. An
  explicit `--reagent` / `--adducts` is unchanged; `assign.run` called from Python without
  adducts keeps its per-sample default but now logs a WARNING when that default is a fallback.
- **README and SKILL.md: `--trace-first` is experimental, not "the TOF path".** The README
  presented it as the TOF path with a per-file scatter figure from one low-resolution TOF, and
  the skill offered it as an equal alternative to rolling the centres; both now use the CLI's
  own wording: EXPERIMENTAL, on its one A/B it recovered about half the ions a file cover found
  in two or more files and Assigned fewer of them (the isotope evidence that earns Assigned
  lives inside a spectrum, and a trace sample averages it away), for batch-level centred masses
  and not a replacement for the cover path.
- **The PDF calls the total signal a transient event only when the data show one.** The Findings
  page printed "peaks at hour X -- N x the late-run baseline -- then decays (a transient event)"
  on every run, whatever N was: on a labelled-nitrate Orbitrap batch, a uronium batch and a
  bromide/nitrate TOF batch the maximum was 1.10x, 1.31x and 1.08x the late-run baseline, all
  under the 1.7x burst bar the same report holds a varying trace to, and the TOF's maximum sat
  at sample 224 of 230. The sentence now says "a transient event" only when the maximum reaches
  `cluster.PEAK_RANGE` and does not sit in the last 10 % of the samples (a later one is said to
  sit too close to the end of the run to tell a transient from a rise); below the bar it says
  there is none and gives the band 90 % of the samples lie in relative to the baseline
  (0.66-1.09x on the nitrate batch, which dips well below its maximum).
- **The reference-list page measures its chance level instead of asserting it.** A hard-coded
  sentence said the near-0-ppm matches "far exceed chance (~single-digit random hits)". The page
  now re-runs the rescue's own mass match on the unexplained m/z set shifted by -40..+40 ppm
  (six shifts, never inside the match window) and prints the range of matches within 1 ppm
  beside the observed count, saying whether the observed count exceeds it (with the ratio to its
  median), falls within it, or falls below it (then a match is no evidence by itself). Measured:
  the labelled-nitrate batch 261 vs 44-103, the uronium batch 96 vs 1-22, the bromide/nitrate
  TOF 110 vs 238-296 (below chance). About 0.1 s per report.
- **Composition no longer counts reagent ions or inorganic carbon as organic chemistry, and
  weights Assigned readings only.** Every neutral was classed CHO / CHON / CHOS by its N and S
  and weighted over every tier, so on the bromide/nitrate TOF the nitrate reagent ion itself
  (HNO3 read as NO3-, HNO3.NO3- and HNO3.Br-: 41 % of all M0 height) made the batch "41% CHO /
  54% CHON", and "(a few bright CHO species carry most of the signal)" followed on every
  negative-mode run. An inorganic neutral is now its own class (`composition.composition_class`;
  `backbone()` is unchanged): a carbon-free one, or one whose single carbon is a carbon oxide /
  sulfide (CO, CO2, OCS, CS2) or a cyanide / cyanate with at most one H and one halogen (HCN,
  HNCO, ICN, INCO, ClCN; `composition.is_inorganic_carbon`); formic acid, urea, fluoroform and
  chloropicrin stay organic. The composition by signal covers the (neutral, adduct) readings
  held at tier Assigned only (`composition.assigned_composition`) and says so: the organic
  classes as shares of the Assigned organic signal, the inorganic ions as a share of all
  Assigned signal. The bright-CHO clause appears only when the five brightest CHO neutrals carry
  at least half of the Assigned organic signal, and says so with the number, before the
  inorganic line; the ammonium/amine count note only on a run with an NH4+ or urea channel (it
  printed on an NO+ batch with neither). "Top species" lists organic neutrals, with the
  inorganic ones on a separate "reagent and inorganic ions" line; the Composition page counts
  them in an `inorg.` row. The TOF now reads 94 / 3 / 3 % CHO / CHON / CHOS with inorganic ions
  at 67 % of the Assigned signal and its five brightest CHO neutrals at 78 % of the organic
  signal; an iodide Orbitrap batch 89 / 4 / 7 % with 89 % inorganic (it printed "15% CHO / 85%
  CHON ... a few bright CHO species carry most of the signal", that CHON being INO2, INCO, HNO3
  and ICN: 54, 16, 7 and 5 % of all M0 height); the labelled-nitrate batch loses the bright-CHO
  clause (its five brightest CHO neutrals carry 33 % of the organic signal). The ledgers are
  unchanged: a reagent-identity reading stays a correct reading.
- **The oligomer line lists Assigned neutrals only.** "Accretion / oligomer products" took every
  high-carbon, high-oxygen neutral of the merged ledger: 27 of 27 were Candidate readings on the
  labelled-nitrate batch, 12 of 13 on the uronium batch, 352 of 380 on the TOF.
  `composition.oligomer_flag` takes `tiers` (default Assigned; never an ion-only row), and the
  page says how many it shows of how many and how many more hold no Assigned reading (0, 1 and
  28 Assigned on those batches).
- **The PDF cover names the package version, and a git commit only when there is one.** It
  printed "peaky (assign v0.6.1) · git <sha>", the version of one module rather than of the
  package, and "git ?" on any install that is not a git checkout. It now prints "peaky
  <version>" and "· git <short sha>" (with "+modified" for uncommitted changes) only from a
  checkout; a run folder's `run_manifest.json` names the code that assigned the run, and a
  report regenerated by other code names both.
- **The Methods text reads the run's own settings.** "amine co-variation r>=0.7" now prints the
  run's `amine_r_min` (the code default is 0.6; a run that did not record it says it shows the
  default); "server isotope-scored matching" now names the scorer the run recorded in the new
  `batch_summary.json` key `scorer` (`local`, the in-process default, or `server` under
  `PEAKY_LOCAL_SCORING=0`), and a run without the key says scoring is in-process by default; the
  cover's trace line prints the stamping window to two decimals (it printed "±9.22777806538331
  ppm") and the median move and per-ion scatter to three significant figures.
- **`peaky report` regenerates a run's report with the run's own batch and dataset names.** It
  titled the report with the reagent label and passed no dataset name, so a report regenerated
  offline lost every reference list the dataset name had unlocked (on the TOF run folder: the
  terpene-oxidation list and its page section). The batch and dataset names now come from the
  run folder's `run_manifest.json`; `--batch` keeps precedence and a new `--dataset` flag
  overrides the dataset.
- **A run whose files never calibrated says once that its evidence levels were not assessed.**
  On sparse Orbitrap tables (an iodide and an NO+ batch: 2-9 isotope-backed core rows per file
  against the calibration's 20) no file calibrates the degeneracy window, so every pair reads
  "no level · no file of the source is calibrated" (190 of 190 and 169 of 169 pooled pairs) and
  every claim tentative, with no word of it above the row level.
  `evidence.levels_not_assessed_reason` recognises the state; `assign_batch` records it as
  `batch_summary.json['levels_not_assessed_reason']` (null otherwise) and logs one console
  WARNING; the PDF cover and Evidence levels page print the sentence (read off the merged rows
  for a run made before the key), and a single-sample workbook prints it on its Summary sheet.
  The Claims and Evidence levels pages now give each no-level row the reason its evidence names:
  on those batches they said all 188 and 165 rows were "a batch-level re-read" (a merged reading
  no pooled pair holds), which none was. No level or tier changes.
- **A foreign peak on a labelled reading's 14N impurity line is present, not 'too low'.** A
  reading whose ion carries the 15N label (`^N`) predicts the labelled reagent's 14N impurity
  line at -0.997 Da (~2 % of M0 per label). Only the engine's own twin or child (or the M0 of a
  reading the levels already refuted) matches it, and an occupied peak at >= half the impurity
  level counted as present, but a free peak the engine had left unexplained -- at that exact m/z
  sits the same neutral's 14N nitrate cluster, which can sit far above the impurity level -- was
  read as the line 'too low' whatever its height, so a correct 15N-nitrate cluster was refuted
  by its own isotope lines (5b). Any other peak there at >= half the impurity level now leaves
  the line present (tested, neither bad nor matched; no positive credit); below half it is still
  too low (`levels.lines.eval_candidate`; the reference `scripts/level_ledger.py` shares it).
  Re-levelled offline, a labelled-nitrate Orbitrap batch: at the pooled batch level
  (`tables/evidence_levels.csv`) exactly one pair moves, C10H18O4 [M+^NO3]- 5b -> 4a (tentative
  -> neutral; 1902 pairs 8/162/258/476/998 -> 8/163/258/476/997); levelled per file (the
  per-file ledgers, and the levels of a single-file `peaky assign`), 20 readings of five
  C10H18Ox [M+^NO3]- pairs (x = 2, 4-7) move 5b -> 5a in 11 of 12 files, their claim stays
  tentative, and 581 more readings change only their evidence/tag text. A uronium batch and the
  level fixtures do not move, and no tier moves. docs/EVIDENCE_LEVELS.md 5.3.
- **The time-series clustering floors follow a low batch noise edge down, so a TOF batch gets
  its families.** The cluster figures gated channels on absolute median floors -- 200 cps for an
  assigned ion channel, 50 cps for an unassigned bin -- and started every panel's log axis at 50
  cps. On a counting TOF (edge ~0.5-1 cps) few channels cleared them: on a bromide/nitrate TOF
  batch only 10 assigned channels and 16 unassigned bins, giving 3 families (2 of them reagent
  ions with their ringing satellites) and no unassigned cluster. The floors are now 3.33x
  (assigned) and 0.83x (unassigned) the batch's typical detection edge (`batch_summary.json`
  `noise_edge_batch_cps`), capped at the old 200 / 50 cps: the edge only lowers them, so an
  Orbitrap batch whose edge sits at or above ~60 cps keeps its floors and its figures, and a run
  directory without the key keeps 200 / 50 cps. With floors at a TOF's scale a fixed 50 cps axis
  bottom would draw few-cps families as empty panels, so the panel axes bottom out at the
  unassigned floor. Because an edge can sit far below the practical detection level (e.g. a
  batch exported in sub-unit heights), at most `top_n` (400) unassigned bins join the unified
  clustering and at most 400 varying leftover bins are clustered, the brightest by median;
  `clusters_summary.json` counts the qualifying bins moved to the leftover path
  (`n_union_over_cap`) and the varying bins not drawn (`n_varying_over_cap`, flagged `over_cap`
  in `tables/clusters_unassigned_<tag>.csv`) -- the two counts overlap. The summary's gates
  record both floors, the edge and their source (`floor_source`); the PDF prints sub-cps floors
  to 3 significant figures (they printed as "0 cps") and lists the bins the cap left out, so its
  unexplained funnel adds up; the cluster CSVs write `median_cps` to 3 significant figures below
  100 cps. New `cluster_batch` parameters: `unassigned_floor`, `noise_edge_batch_cps`,
  `floor_x_edge`, `unassigned_floor_x_edge`, `top_n` (`floor` still pins the assigned floor).
  Measured by replaying the clustering stage on copies of finished runs: five Orbitrap batches
  with edges of 61-758 cps give the same families, figures and tables as before (apart from the
  new `over_cap` column and the family names below); an EasyIC Orbitrap batch (edge 35 cps,
  floors 117 / 29 cps) gates 426 unassigned bins instead of 232, with the same 1 family and 5
  pages; the bromide/nitrate TOF 3 -> 64 families (16 -> 497 channels), 0 -> 9 unassigned
  clusters, 4 -> 33 pages at the edge the batch run now records (70 families, 546 channels, 35
  pages at the run's older recorded edge); a low-resolution nitrate TOF 0 -> 44 families, 0 ->
  14 unassigned clusters, 2 -> 33 pages (uncapped: 137 clusters, 109 pages).
- **A co-varying family is named by an Assigned member when it has one.** The `co-varies with X`
  label took the brightest member with a formula whatever its tier, so a family holding Assigned
  readings was often named by a Candidate one (labelled-nitrate Orbitrap batch: 16 of 39 labels
  named a non-Assigned member, 8 of them with an Assigned member in the family). Members now
  rank Assigned first, then by median (`clustering.family_label`); families and their members do
  not change. Labels on a non-Assigned member: labelled nitrate 16 -> 8 of 39, uronium 5 -> 1 of
  14, iodide 2 -> 0 of 13, bromide/nitrate TOF 31 -> 16 of 64 (41 -> 20 of 70 at the run's older
  recorded edge), a low-resolution nitrate TOF 27 -> 21 of 44 -- each one left in a family with
  no Assigned member.
- **`scripts/scorecard.py`: every decoy arm runs at its file's control calibration.** An arm
  calibrated on its own commits, which are wrong readings by construction; where that backbone
  was too small to calibrate (every wrong-adducts arm on a TOF batch, a -6 ppm shifted arm on
  the uronium Orbitrap batch: 16 rows, 20 needed) the arm ran with the mass z-test and the
  degeneracy audit off and kept every mass fit, bounding an engine no real file runs (that arm
  Assigned 32 rows, 21 of them certified formulas; at the control's calibration 13; the
  wrong-adducts arm of the bromide/nitrate TOF batch's brightest file Assigned 87 rows against
  the control's 58, at the control's calibration 42 -- still 72 % of the control, 39 of them new
  ions: an open engine finding on the TOF, not a resolved artefact). Now `passes.calibrate` and
  `tiers._calibrate` take the control arm's ledger for the length of each arm's engine run
  (`inherited_calibration`); `decoy.calibration`, the decoy manifest and the board row
  (`decoy_calibration`) say what the arms ran at.
- **`scripts/scorecard.py`: the wrong-adducts arm counts new ions beside readings, and never
  calls a run's own channel wrong.** Most of the arm's Assigned rows only re-split an ion the
  run already reads (X [M+NH4]+ is the ion of X+NH3 [M+H]+): on the uronium Orbitrap batch's
  brightest file the arm Assigns 343 rows, 85.1 % of the control's 403 Assigned; 334 of them
  carry the ion composition of the control's M0 reading on the same peak (any tier: for 119 of
  them the control holds that reading only below Assigned, and 218 of the 343 sit on a peak the
  control Assigned); 9 are new ions (2.2 %). On the labelled-nitrate batch's brightest file the
  arm Assigns 22 (6.6 %), 4 of them re-splits: 18 new ions = 5.4 %. `decoy.adducts` now carries
  `assigned_same_ion`, `assigned_new_ion`, `new_ion_rate` and `same_ion_share` beside the
  reading-level rate (board `decoy_adducts_new_ion_rate`). The wrong set leaves out every
  channel the run itself reads (its profile's adducts, every adduct its ledgers commit, a side
  channel it opened) -- an iodide run's `[M+I]-` was declared wrong -- and with nothing left the
  arm does not run.
- **`scripts/scorecard.py`: the roster block names its scale, counts neutral-or-better, and its
  presence test sees only the formula's own line.** M2 looked for a roster line within a flat 6
  ppm (25 sigma on an Orbitrap), took the nearest stamped line in any share of the spectra,
  counted another ion's isotope satellite or a reagent line as a sighting, and called another
  split of the same ion composition a misread. Now the window is 4 x the run's measured mass
  sigma (1 ppm at least, the tolerance at most), a line counts in >= 20 % of the spectra stamped
  or not, and of the lines that pass the best-read one stands for the formula; an isotope
  satellite of another ion, or a reagent line of another composition, is no sighting, while a
  reagent line of the expected ion's own composition is (the reagent's reference ions -- its
  ion, its water clusters -- are read as themselves there); a same-composition split is
  `same ion`. Measured on the labelled-nitrate and uronium Orbitrap batches and the
  bromide/nitrate TOF batch (59 roster formulas): present 33 / 34 / 35 -> 28 / 30 / 26, misread
  9 / 2 / 11 -> 1 / 0 / 2. The roster claims carry `neutral_or_better` (3c + 4a, per line read:
  the claim of the line M2 picked) beside `identified` (a class list never reaches 3c; board
  `roster_neutral_or_better`, a tracked metric and an acceptance criterion), the card counts
  every M2 status for every source, and the M2 block and the board row name their level scale
  (`scale`) and presence test (`roster_test`).
- **The signal-to-noise the score reads is judged before it is believed.** Three
  terms of the v2 fit are set by a peak's `signal_to_noise`: whether an ABSENT predicted
  line is charged (rel x SNR_base >= k_detect), the intensity tolerance of a matched line
  and the centroiding term of its mass width. The column the server sends was taken at its
  word, and on the bromide/nitrate TOF it is not a signal-to-noise at all: over a file's
  ~1500-2600 picked peaks it does not track height (Spearman -0.07..0.17 in 28/28 files; a
  478-count peak carries 1.1, a 2-count peak 12) where every Orbitrap file gives 0.99-1.00
  with an implied noise that is flat within a file (~19 cps on the uronium files). Read as an
  SNR it excused every missing line of a bright ion (no 81Br line of a bromide cluster was
  ever charged: 89 Assigned [M+Br]- rows had no 81Br child hung under them) and charged dim
  ions for lines they could never show. Now `scoring_for_sample` judges the
  column once per sample (`local_scoring.assess_snr`: Spearman of height vs the column over
  the file's peaks, >= 0.5 is a signal-to-noise; fewer than 30 peaks are not judged) and
  records the verdict in the scoring snapshot (`snr_source` server / none /
  poisson_fallback, `snr_spearman`, `snr_n`, `snr_edge`; a stand-in judges its own table);
  where the column fails, the local scorer reads the counting-statistics SNR of an
  ion-counting detector, h / sqrt(h + edge^2) (the peak's Poisson noise in quadrature with
  the picker's detection edge, the file's 1st-percentile height). The heights of a TOF file
  are per-file averages, so the true ion counts are higher and this SNR is a conservative
  lower bound: a matched line's ratio tolerance is at least its Poisson scatter, and an absent
  line is charged only where even this bound says it was within reach (an absent 81Br line
  from ~10 cps at a 0.74 cps edge; a 13C line only on bright ions). The run log and each
  file's stats say so (`snr_source`). Offline replica on a bromide/nitrate TOF validation run (28 files, 10 309
  readings; the server column reproduces the ledger scores: median and 95th-percentile
  difference 0, 50 rows differ by more than 0.01): 155 of 1622 per-file Assigned rows score
  under the Good bar with the fallback, 107 of them from above it (58 [M+Br]-, 30
  [M+HBr+Br]-, 16 [M+NO3]-: the 81Br line absent or off); the per-row median change is
  +0.0006; every TRUE reading of the run's frozen truth set keeps at least half of its files (one
  pair exactly half). The Orbitrap runs are untouched (their column passes). The reference-
  list rescue (pass 8), which stamps tiers after the tier pass, honours the floor below; the
  network scorer (`PEAKY_LOCAL_SCORING=0`) still reads the server column, whatever the
  snapshot says.
- **A counting-detector floor for the TOF tier.** On a TOF an M0 under 3x (the scorer's
  own `k_detect`) the batch's typical detection edge -- the median of its files' own
  1st-percentile heights, which `assign_batch` now measures once per batch
  (`PassConfig.noise_edge_batch_cps`, `batch_summary.noise_edge_batch_cps`) -- is Candidate
  whatever hangs under it (`tiers.tof_assign_floor`, `TOF_ASSIGN_FLOOR_X_EDGE`; the file's
  own edge on a single-sample run; `PassConfig.instrument_type` from the scoring snapshot
  keys it, so an Orbitrap never sees it). A handful-of-ions centroid has no testable mass and
  no testable isotope line, and the kid or series step that corroborated it is itself
  sub-edge. A file's own edge follows its total ion count: two files of the bromide/nitrate
  TOF batch with a 5x lower count had edges of 0.10-0.13 against the batch's 0.47-0.96; the
  batch carried 466 of its 1622 Assigned M0 rows under 3 counts, 136 of them in those two
  files (of their 188 Assigned rows), among them both silicon FALSE readings of the frozen
  truth set (C10H24N2Si and C11H11N3OSi [M+Br]- at 1.5 and 0.65 counts, 'Good' on a series
  anchor and a sub-count kid). At the batch's floor (2.22 cps on the cached peaks of all 28
  files; 1.70 on the live run's 18 cover files) 331 per-file Assigned rows are under it; every
  TRUE reading of the truth set keeps at least half of its files. The per-file stats carry the
  floor in force (`tof_assign_floor_cps`, None off a TOF) and the class (`instrument_type`);
  a decoy arm of the scorecard takes the run's batch edge (`batch_summary.
  noise_edge_batch_cps`), so it is tiered at the floor the run was. The edge is measured on
  the first per-file stage's files (the cover; on a trace-first run the residual picks); the
  trace sample itself (averaged traces, no class) never sees the floor.
- **The fit scores a mass at the sample's own m/z-dependent centre.** The v2 score
  judged every line against ONE offset per sample (its server matches' median), and the
  pass-1 calibration that could fit the instrument's 1/mz mass trend selected its backbone
  by that same score -- so it saw only rows already near the constant offset, and on the
  15N-nitrate Orbitrap rejected the trend in 12/12 files (all 12 pooled too), leaving the
  bright low-mass acids (C5H8O4 / C6H6O4 / C7H8O3 [M-H]-, +0.7-0.9 ppm at m/z 131-141
  where the instrument centre is +0.7-0.8) at v2 0.50-0.68: Candidate.
  Now (a) the local scorer also reports `ion_score_massfree` (the isotope pattern alone;
  a new ledger column) and takes a per-line `centre`; (b) `calibrate()` fits the trend on
  CHO-CHON rows whose pattern-only score is Good AND that carry an observed isotope line
  (an untested pattern -- a dim O-rich coincidence with every satellite below detection --
  scores ~1 and sat ~0.4 ppm off the trend above m/z 400); (c) once a trend is accepted
  `assign.run` re-runs the file from pass 0 with every line judged at the trend's centre
  at its own m/z (`PassConfig.score_at_trend`, default on) -- again while the re-run's
  own calibration moves the trend by more than 0.05 ppm anywhere (at most twice), so the
  file ends scored at the trend its gates use -- and records the trend in the sample's
  `pattern_scoring` snapshot (`mu_source: "trend"`; a batch carries it back from each
  worker), which a decoy arm inherits. The network scorer (`PEAKY_LOCAL_SCORING=0`)
  judges one offset: there nothing re-runs. Single-file check (passes 0-1, three
  labelled-nitrate files): the first fit (b 0.16-0.21 mDa, coverage from m/z 139-157)
  is refined by the re-run (b 0.207-0.218, coverage from m/z 131), within the pre-0.9.0
  engine's per-file fits (0.16-0.22).
- **"unique formula in the calibrated window" needs the degeneracy audit.** On an
  uncalibrated file the audit is skipped and stamps nothing, yet a row with no stored
  rival still read the unique-window text (the mixed TOF: 59 of 67 such per-file gains).
  It now reads "no rival formula in the search window (degeneracy not measured: file
  uncalibrated)". The tier is unchanged; the evidence levels never read the text.
- **The reagent halogen comes from the declared channels.** The pair facts and the merge
  vote's class named it from the commonest committed cluster adduct; on the mixed Br-/NO3- TOF
  that count flipped (3907 [M+NO3]- : 3847 [M+Br]- per-file M0) and switched the
  reagent-81Br rule (`reagent_only_iso`) off for the whole run. `evidence.channel_halogen` reads the run's
  declared channels (`assign.run`, before opportunistic ones) and the batch profile's
  (`assign_batch`); the count remains the fallback where no channels are known.

### Added

- **A TOF-class batch flags the readings that rest on mass alone (`mass_only`), and never demotes
  them.** On a TOF the formula grid saturates with m/z: a shifted-mass decoy of a bromide/nitrate
  TOF batch, pooled over seven files, was Assigned 21 times below m/z 350 against 292 for the real
  spectra, but 234 times at or above it against 122, and a lower-resolution nitrate TOF showed the
  decoy at the real level from m/z 350 up in one decoy arm and from about m/z 450 up in another;
  below that, too, most TOF readings show no isotope line of their own. The merged ledger now
  carries `mass_only` (bool) and `mass_only_reason` on every Assigned row of a TOF-class run
  (`assignment/mass_only.py`): `True` when no file that holds the reading at tier Assigned shows an
  attached isotope child of its M0 at its exact spacing and 0.5-2x its expected height that
  measures an element the neutral itself supplies (the 13C line of an organic neutral; 37Cl / 81Br
  / 34S / 29Si / 30Si of a neutral carrying the element -- the reagent's own 81Br twin never counts;
  the evidence scale's per-file line test), pass-0 known species exempt. On a run whose reagent
  channels carry a halogen, a line of that halogen counts only when the ion holds more of it than
  one reagent channel puts on an ion (two, by `[M+HBr+Br]-`): `C8H6BrNO4 [M+NO3]-` and
  `C8H6N2O7 [M+Br]-` are one ion, so their 81Br line keeps neither unflagged, whichever label the
  engine picked. Only attached lines count: a peak at the right spacing that the scorer did not
  attach is not seen. The reason says the reading rests on mass alone below `PassConfig.tof_flag_mz`
  (`--tof-flag-mz` on `peaky batch`, `pool` and `assign`, default 350) and that a TOF's formula
  space is saturated at or above it. `False` is not support at or above the threshold: there even a
  present line is weak (a shifted spectrum keeps real isotope spacings), and on the bromide/nitrate
  TOF's seven-file decoy arms the line test left 58 shifted-decoy readings at or above m/z 350
  unflagged against 12 real ones; below m/z 350 it flagged all 21 decoy readings. Candidate rows
  and every row of an Orbitrap-class run keep the columns empty. `batch_summary.json['tof_flag']`
  records the class, the threshold and the Assigned / flagged counts either side of it (also under
  the manifest's counts); the PDF Findings page says them in one sentence, adds that an unmarked
  reading at or above the threshold is not supported either, and marks flagged species with a
  dagger in its species lists and the appendix; a single-sample workbook shows the two columns on
  its Assigned sheet and a *TOF mass-only flag* Summary section; the scorecard's element census
  counts them (board keys `census_mass_only`, `census_mass_only_split`, appended). Tier, evidence
  level and claim are untouched: re-stamped offline on copies of a bromide/nitrate TOF batch and a
  low-resolution nitrate TOF batch, every original merged column stayed byte-identical while the
  flag marked 261 of 355 Assigned readings (164 of 222 below m/z 350, 97 of 133 at or above; 31
  known species exempt) and 250 of 282 (176 of 193, 74 of 89); two Orbitrap batches got empty
  columns. Of the bromide/nitrate TOF batch's 34 true readings 8 carry the flag (every true reading
  of that batch sits below m/z 350): four small bromide adducts whose only attached line is the
  reagent's 81Br (two of them show an in-band 13C peak the scorer did not attach), and four nitrate
  clusters of oxidation products without an attached in-band 13C line. The flag says what a reading
  rests on, not that it is wrong. README, docs/OUTPUTS.md, docs/MERGE.md, docs/QC_AND_REPORT.md and
  docs/SCORECARD.md.
- **`peaky publish` leads its summary with the rows peaky holds Candidate that Mascope will show
  as `assigned`.** Mascope's tier column, tier strip, tier filter and roll-ups read the tier it
  derives from fit x plausibility, not peaky's; the dry run printed only the total disagreement.
  `publish.build_rows` now counts `candidate_shown_assigned` (a Candidate that Mascope's banding
  calls `assigned`, not every disagreement), and the summary's first line gives that count and
  says where peaky's verdict is read (the `engine tier` column, the `tier_disagrees` filter).
  Summed over files: 4246, 2129 and 9166 such rows (against 5081, 2155 and 9560 disagreements)
  on a labelled-nitrate Orbitrap batch, a uronium Orbitrap batch and a bromide/nitrate TOF
  batch.
- **README: a TOF caveat.** On TOF data the evidence levels of 0.10.0 are not assessed (`NA`)
  and the tier rests on mass and isotope evidence. Where the formula space is crowded (at higher
  m/z, and at any m/z on a low-resolution TOF) mass alone cannot separate formulas, so an
  Assigned reading without isotope support is to be read as mass-only. Isotope support is the
  line the ion's formula demands (reagent atoms included) at the predicted ratio; an
  isotopologue row in the ledger is not support by itself, and at higher m/z on a TOF even a
  present line is weak evidence.
- **README: a known limit for very bright Orbitrap ions.** On a high-intensity Orbitrap the
  brightest ions can sit about +0.6 to +1.0 ppm off their formula's mass while their own isotope
  lines sit on centre; their pattern score then falls low enough that 0.10.0 commits no reading
  for them.
- **`scripts/scorecard.py`: a populated-defect ppm-shift decoy arm.** The 0.35 Da shift arm
  moves every line below ~m/z 350 into the empty mass-defect gap where no CHNOS formula sits, so
  on an Orbitrap the engine proposes nothing there, no tier gate is tested and the arm reads 0 %
  whatever the tiers do (on the uronium Orbitrap batch's brightest file: 2 M0 rows, none
  Assigned, against the control's 700 / 403 Assigned). The new arms scale every m/z by (1 + k x
  1e-6), default k = +9 and -9 ppm (`--decoy-ppm`; `none` for none): outside the file's match
  window, inside the populated band where the formula grid is dense, so wrong formulas are
  proposed and the mass, degeneracy and pattern gates are exercised (a shift keeps every isotope
  and label spacing, so no shift decoy tests the label / isotope vetoes). A shift inside the
  window (the wider of the grid's 3 ppm and the window the file is scored at: 5 ppm on an
  Orbitrap, 15 ppm on a TOF) is skipped and says why. Each arm is rated against its files'
  control; pooled, against the control counted once per arm. Every arm now reports its Assigned
  rate below and at or above m/z 350 and per 50-Da bin (`decoy.bins`); the card leads the decoy
  section with the headline below m/z 350 (`decoy.headline`: the ppm arms when they ran, else
  the 0.35 Da arm, labelled blind below ~350 on an Orbitrap and kept for continuity), naming the
  calibration the arms actually ran at and the M0 rows the arm committed below 350 against the
  control's; new board-row keys (`decoy_ppm_*`, `decoy_headline_lt_350_rate`,
  `decoy_headline_arm`, `decoy_orbitrap`) and two acceptance criteria (the ppm arms' identified
  rate beside their own Assigned rate; the headline, Assigned already, beside no older metric).
  The board's lead claims table reads its decoy column from the headline arm (the ppm arms named
  with their shifts), and an Orbitrap row's 0.35 Da number below 350 is marked blind there, on
  the board and in the acceptance block. `--decoy shift` runs both kinds of shift arm,
  `--decoy ppm` the ppm arms only, `--decoy both` every arm. On that file the control reproduces
  the run's per-file ledger (696 of 696 common M0 rows, same reading and tier) and the ppm arms
  Assign 6 (+9 ppm) and 9 (-9 ppm): 15 of 806 = 1.9 %, below m/z 350 12 of 790 = 1.5 %; over the
  batch's two brightest files 33 of 1568 = 2.1 % (below 350 1.7 %, at or above 7 of 34). On the
  labelled-nitrate Orbitrap batch's brightest file (control 331 Assigned, 790 of 809 common M0
  rows as the run read them) the ppm arms Assign 21 and 30: 51 of 662 = 7.7 %, below m/z 350 48
  of 660 = 7.3 %, where the 0.35 Da arm Assigns none below 350.
- **The evidence scale of peaky 0.10.0.** Every committed formula carries, beside its tier, an
  evidence level that says what the evidence behind it is worth (`docs/EVIDENCE_LEVELS.md`, the
  contract; `peaky/assignment/levels/`, the machinery; `peaky.assignment.evidence`, the entry
  points). A CIMS-adapted Schymanski scale built on one rule: a level-3 unlock must be evidence
  measured to carry formula-specific information, and everything else is a tag printed with its
  measured base rate.
  - **Levels:** `3c` (ion established, the neutral / adduct split pinned, and a NAMED context-list
    entry names the neutral), `4a` (pinned + a positive fact: an own in-band isotope line of an
    element the neutral contains, the 15N label, or an NH4 adduct tracking its parent), `4b` (the
    ion only: the split open, pinned without a positive fact, or an ion-only channel), `5a` (a
    competitor ion left in the calibrated window), `5b` (rejected, or nothing could be tested);
    the buckets `reagent` (reagent ions and clusters) and `NA` (not assessed); 1 and 2 defined and
    never assigned. Each pooled `(neutral_formula, adduct)` pair takes the first outcome that
    applies: reagent, 5b, 5a, 4b (ion-only), 3c, 4a, 4b.
  - **Claims:** `identified` (3c), `neutral` (4a), `ion` (4b), `tentative` (5a, 5b, no level), with
    the buckets `reagent` and `not assessed` reported beside them (`evidence.CLAIM_KEYS`).
  - **The steps:** step 0 sorts out reagent ions and rejections (`iso_veto`, `label_veto`,
    `lowconf`, an implausible-chemistry below-assignability setter, the reading's own isotope
    lines); step 1 enumerates every ion of the run's space in the mu ± 3 sigma window and excludes
    competitors by isotope-line tests (a), (b), (c), (k) and CH2 / CF2 series exclusion; step 2
    asks whether the split is pinned over the run's adducts with the side channels locked
    (`evidence.SIDE_CHANNELS_LOCKED`, the single switch; `evidence.UNLOCKED`, the sweep hook), by
    the 15N label or by being the only plausible decomposition, with the engine's amine gate and
    the NH4 admissibility rule deciding `[M+NH4]+` readings on a uronium run and the reagent-isobar
    window rule keeping the X+reagent isobar live; steps 3-4 read 3c / 4a / 4b off the facts.
  - **Tags, never levels:** two routes, other-source partners, CH2 / CF2 ladders, class lists, the
    locked side channels that would open a split, window-only pins, the amine gate's decisions, a
    tentative lead (on every level), ties and series exclusions.
  - **Instrument class:** assessed only with a width model resolving >= 50 000 at m/z 200; a
    TOF-class or class-less source reads `NA` before any fact work.
  - **Outputs:** eight columns on every committed M0 row of the per-file and merged ledgers --
    `evidence_level`, `evidence` (one line: the level, the decisive facts, the tags),
    `would_lift`, `competitors_left`, `tags`, `context`, `context_source`, `claim`.
    `tables/evidence_levels.csv` (one row per pooled pair, every step fact);
    `batch_summary.json` `evidence_levels` (`scale`, `instrument`, `pooled` / `merged` /
    `per_stage` over levels and buckets, `n_pairs` (pooled pairs), `n_unstamped` (merged rows
    without a pooled pair), `side_channels_locked`, `unlocked`, `n_corroborate` (the vote's
    cross-set size), `cross_source` (the `--corroborate` paths), `partners`, `amine_r_min`, the
    batch checks' funnels), `claims` over the six keys, `reflists_context`
    (which keyword in which name activated each reference list; `reflists.activate` records it)
    and `amine_r_min`. `NA` is a literal token: read `claim`, or pass `keep_default_na=False`.
    The single-sample workbook's **By evidence level** sheet and the PDF's **Evidence levels**
    page show the levels, and `publish` carries the eight columns in the engine provenance.
  - **Where it runs:** per file in the `evidence` stage of `assign.run` (the file alone, every
    file-count minimum 1, no time series; the degeneracy stage now persists its calibration as
    `stats["degeneracy_cal"]` for the window); pooled in `assign_batch.run` over the per-file
    ledgers re-read from disk in sorted order (every minimum 3, with the stamped time series, the
    merged ledger and the batch checks) and stamped on the merged ledger by pair; post hoc by
    `scripts/level_ledger.py` with its own independently written decision layer. A merged row no
    pooled pair holds reads `no pooled pair: a batch-level re-read`; an uncalibrated file
    levelled alone gets no level and says why, and so does every pooled pair of a batch none of
    whose files is calibrated (no run window). A lone ledger CSV levelled by the script with
    `--resolving-power` and `--reagent` is levelled as the per-file stage of a single-sample
    `peaky assign` (the profile's context and the reference lists it activates).
  - **`--corroborate SOURCE`** (new in this release, on `peaky assign` and `peaky batch`,
    repeatable): a run directory, an out directory holding one run, or a ledger CSV of the other
    reagent channel or the other instrument on the same air. It feeds the merge vote's class and,
    from an Orbitrap-class run directory, the other-source partner tag (a TOF-class or class-less
    run directory, or a ledger CSV, feeds the vote only, logged). A partner never unlocks a
    level, but as a route it can anchor the step-1 series exclusion, which can lift a pair out
    of 5a. A run directory none of whose files is calibrated, or whose reagent profile or
    context this process does not know (a run made under a `--reagent-config` profile), gives
    no partners (logged with the reason; the batch does not stop on it). On single-sample
    `peaky assign` it is accepted, logged and ignored.
  - **The merge vote keeps its own class.** The vote ranks a cluster's ions by a private class
    (`evidence.vote_classes`: neutral backed / formula confirmed / unconfirmed), computed by the
    decision the vote has always read (the pre-scale decision, kept frozen in `evidence.py` and in
    the new `scripts/level_ledger.py`, with the new `peaky/data/isomer_space.csv` and the
    decision's golden sets in the new `tests/fixtures/levels/`), so the scale moves no
    assignment. The vote note in `tier_reason` and `tables/jitter.csv` (`vote_class`, 0 / 1 /
    2) print the class, never a level.
    The class is computed in the batch's parent with the reagent halogen the file's own run read
    from its declared channels (carried back in the file's stats, `per_file[i].reagent_halogen`);
    the pooled pair facts read the batch profile's (`batch_summary["reagent_halogen"]`), which a
    post-hoc re-level reads back.
  - **Never written:** the development builds' `evidence_axes`, `level_reason` and
    `n_plausible_structures` columns; a ledger that carries them is read as levelled on an older
    scale (every letter no level, its claim tentative) and the outputs say so.
  - **Validated:** the in-core scale reproduces a reference implementation level for level and on
    every text field on a labelled-nitrate Orbitrap run (1 850 pairs) and a uronium Orbitrap run
    (1 161 pairs) and all twelve of their decoy tables; the in-run pooled levels equal a post-hoc
    re-level of the same run directory; the vote class equals the stored class on every M0 row of
    three regression runs (26 278 rows). Known limits (the lock holds most 4a height on a
    labelled-nitrate run, the urea isobar holds most 4b height on a uronium run, the lead / list
    circularity, the run-level family union) are listed in the spec's section 12. The per-file
    stage costs about 5-20 s per Orbitrap file.

- **A halogen lock answers a tentative lead (rule H).** A lead is unsupported,
  not contradicted -- a speculative residual fit, a reference-list match too dim to
  confirm, a commit outside the element budget. A fourth batch check,
  `H` in `batch/iso_checks.py`, now writes a POSITIVE fact per pooled pair whose ion
  carries Cl or Br: a `lock` where the stamped series shows the line at the exact
  37Cl - 35Cl / 81Br - 79Br spacing (1 ppm) in >= 60 % of >= 20 M0 spectra, co-varying
  (r(log area) >= 0.8), at 0.65-1.45 x the ion's count x 0.3198 / 0.9728, and the M0
  not itself the heavy line of a lighter one; from m/z ~206, where a 30Si line (0.206
  mDa below 37Cl) can sit inside the 1 ppm window, a Cl lock is refused (`si_rich`)
  where the ion's M+1 region shows the 29Si line a Si-rich ion making the partner from
  30Si must carry (the 2026-09-28 decision, "the 29Si line decides": n = ratio / 0.0335
  silicons, 29Si n x 0.0508 at +0.99957 Da; where the width model parts it from 13C the
  line itself at >= half its expected area, present and co-varying like the partner
  (where no such line is present, the +1 region decides: the peak picker may not have
  parted the two, `unparted`); else the +1 region -- every peak from 29Si to 13C --
  present and co-varying with the M0 like the partner (r(log region area, log M0 area)
  >= 0.8) and sitting, area-weighted and the median over the spectra, at least half-way
  from the reading's own +1 position toward the blend a Si reading implies -- the
  position decides, no height criterion (the 2026-09-29 decision) -- and a reading that
  itself carries 1-2 Si is also refused on reading >= half that 29Si area above its own
  +1 line, since its own 29Si pulls the half-way mark toward 29Si (the Si-reading
  guard); 81Br needs no test; the 30Si spacing is AME2020's 1.9968436, so the test
  starts at m/z 206.3). The position and not the height: a peak picker reporting a
  blend at its apex keeps its position toward 29Si while its area comes out short
  (72-84 % on the regression batches' siloxanes), and rule C, which a height test leans
  on for carbon-rich readings, reads no ion too dim for its 13C line -- the excess test
  built first (excess AND position) let dim siloxane misreads lock unrefuted at any m/z
  from 206.3 (48 of 52 dim readings of the uronium batch's real D8 / D9 urea clusters
  with the lock gates forced, up to 173 of 263 synthetic Si7-Si12 readings; the
  position test none, bar one Si2 reading the guard refuses). The price: ~1.0-1.4
  refusals per 1000 real Cl ions placed on real +1 regions (0.5-0.9 for the excess
  test), where another ion's line near 29Si merges into a dim ion's +1 region; on a
  TOF-class batch, whose peak list reads a +1 region ~1.3 mDa below the ion's own +1
  position, the position is no silicon signature -- harmless while no TOF Cl lock forms
  above m/z 206.3 (the batch's one TOF lock, C2H3ClO2 [M+NO3]- at m/z 156, sits below
  the silicon test's threshold). The Si-reading guard is an OR (a Si reading is refused
  on the position or its excess; as an AND it would reopen the Si2 corner: 393 of 852
  synthetic blended cases, the dim Si1 / Si2 readings among them, no longer refused),
  and it asks no position, so it can refuse a real Si1-2 Cl reading on any excess in its
  +1 region, a neighbour's line at the 13C position included (0 / 0 / 1 of the 30 / 12
  / 37 real Si1-2 ions of the regression batches, forced blended).
  Never on an ion with Si >= 3 or both halogens, and
  never on a halogen the batch's reagent could have put there: a line counts only where
  the ion carries more of it than one reagent ion supplies (the batch's reagent
  decides, not the adduct label: HBr [M+^NO3]- locks on the labelled-nitrate batch,
  HBr [M+NO3]- would not on a bromide one). On the pooled batch a locked pair whose
  lead rows were all set by a setter the lock answers -- the three speculative-residual
  reasons, the dim reference-list rescue, and the element budget only where the locked
  halogen is the neutral's sole violation (`budget_ok`, the CF2 exemption's
  construction) -- and none is below assignability is lifted: the pooled pair facts
  read its lead False and record the lift (`lead_lift`, `lock_note` in
  `tables/evidence_levels.csv`), so the evidence scale prints no `lead` tag on it; no
  step of the scale reads the lift, the isotope-check vetoes and rule K outrank it, and the
  per-file ledgers, the tier and the vote never see it. `tables/iso_checks.csv` gains
  the `H` rows and their columns
  (`lock`, `element`, `n_halogen`, `ratio_lo` / `ratio_hi`, `heavy_cl` / `heavy_br`,
  `budget_ok` / `budget_why`, and the silicon test's `si_n`, `si29_expected`,
  `si29_seen`, `si29_mode`), `batch_summary.json` the H funnel and `locked_pairs`. On
  the regression batches (one variable against the previous trunk run; the runs equal
  the offline replay row for row, their tables' C / REQ / HIGH rows byte for byte):
  the labelled-nitrate Orbitrap locks 16 pairs and lifts 4 -- three
  chlorinated acids (C6H9ClO3, C7H11ClO5, C6H10Cl2O4 [M-H]-, series gap-fills whose
  37Cl line and its count are the only evidence against their 15N-nitrate twins with
  one Cl fewer, 35Cl + 4 H - 2 C being 15N within 0.044 mDa) and HBr [M+^NO3]-; the
  uronium Orbitrap locks 1 and lifts it (C7H11ClO2 [M+(CH4N2O)H]+, Cl its only budget
  violation); the bromide / nitrate TOF locks 1 and lifts none
  (its bromide reagent could have put the Br in 659 of its ions; at 1 ppm rule H is
  Orbitrap-only in practice). The silicon test reads the four Cl locks above m/z
  206.3 (C7H11ClO5, C6H10Cl2O4, C10H19ClO3 [M-H]-, the uronium urea cluster) and finds no
  29Si line on any: their +1 region sits at their own 13C position, 1.5-1.8 mDa short of
  the half-way mark, and the tables are identical under the excess and the position
  test. No uronium siloxane can form a Cl lock at all (none passes the one-Cl gates at 1
  ppm). Handed their M+2 lines as Cl partners (taken within 3 ppm where 1 ppm finds them
  in < 60 % of the spectra: Si10 24 %, Si11 0 % at the 30Si spacing) and read as every
  CHNOS(+Si <= 2)+Cl formula within 1 ppm, the position test refuses 115 of 224 cases,
  rule C refutes 48 more, and the other 61 have a +1 region the gates cannot read (the
  excess test: 2 refused, 150 left to rule C, 72 through). Over the full series rule C
  refutes every one of the 37 Cl1 / Cl2 readings (23 one-Cl) of the D5, Si10 and Si11
  urea clusters within 1 ppm, and Si12's 19 (11); in the dim half it is untestable on
  the D5 urea cluster's 3, 2 of Si10's and all 19 of Si12's. The D7 urea cluster's
  readings are rule-C-untestable, and only the lock's 1 ppm gate stops that cluster. On
  the Orbitrap batches' real Si-free ions, forced blended with a one-Cl partner, the
  position test fires on 0 and 1 as the excess test did -- the co-variation gate's work
  (37 -> 1 from m/z 206.3 for the excess test, 23 -> 0 and 14 -> 1; the first count's
  38th fire left with the threshold move) -- and on the TOF on 24 (the excess test 17;
  the seven more Si-free ions whose +1 region sits 1.4-2.6 mDa below their own +1
  position, 1.2-2.1 mDa above 29Si). `tests/test_halogen_lock_*.py`.

- **Isotope checks test what a formula claims about its own isotope lines.**
  The per-file isotope test only asks whether a child it found sits in the band; three
  batch checks (`batch/iso_checks.py`, on the stamped batch series) ask what the
  formula claims and refute it -- the pair reads 5b (`iso_veto`) -- where the series
  says otherwise: **rule C**
  (Orbitrap-class batches: the 13C line's area and height both read a carbon count the
  formula misses by more than max(1.5, 0.25 C, 3 se); not read where 13C makes under
  half the +1 line, as on a Si-rich ion), **REQ** (a heavy line the formula requires
  -- 81Br, 37Cl, their Br2 / Cl2 patterns, 34S, 29Si / 30Si -- absent in >= 80 % of >=
  3 spectra where it would clear 3x the height gate, judged at the height this batch
  shows that element's lines; a TOF only when absent at the stamp window AND 20 ppm)
  and **HIGH** (a co-varying heavy line >= 3x the formula's count-aware M+2 and >=
  0.2x its M0, guarded against another pair's line, a plain 13C line, a non-isotopic
  +2 alias, a height no envelope reaches, and an element no ion at that mass can carry
  in the number the line implies). The instrument class is read off the batch's width
  model. `tables/iso_checks.csv` holds every test with its numbers and note,
  `batch_summary.json` the funnel (`evidence_levels.iso_checks`), and
  `scripts/level_ledger.py` reads the table from the run directory; rule C's carbon
  count is also the scale's carbon test (c) where it is testable. Batch only, never per
  file; a curated row is not exempt. On the regression batches (one variable against
  the previous trunk run; the runs equal the offline replay row for row; no ion or tier
  moves): the labelled-nitrate Orbitrap refutes 69 merged rows (REQ 63, rule C 7, HIGH
  5, six by both REQ and C): brominated and
  chlorinated readings whose 81Br / 37Cl line is absent in every spectrum where it
  must show, readings whose 13C line counts far fewer carbons than they carry (3.7 for
  10, 5.8 for 22), and aromatic labelled-nitrate readings whose 37Cl line says they
  are chloride adducts of monoterpene products; the uronium batch 3 (a
  C12H14N2O4Si [M+H]+ reading with no 29Si line in any of the 282 spectra
  where it would show, where C13H15N2O3S+ fits the mass at 0.1 ppm against 0.9 and the
  M+2 line sits at the 34S spacing, its urea cluster, and one urea cluster rule C
  refutes); the bromide / nitrate TOF 16 (REQ 11, HIGH 5). No roster species is
  refuted by rule C. `tests/test_iso_checks*.py`.

- **On a labelled-nitrate channel the reagent's two isotopologues arbitrate each
  other's reading (rule K).** A 15N-labelled nitrate reagent sees one neutral X's
  cluster twice, 0.997 Da apart: big as [X+^NO3]- and small as [X+NO3]- (the reagent's
  14N impurity plus ambient 14N nitrate). The batch now tests each line against the
  other on its own stamped time series (`batch/label_twins.py`, profiles labelled `^N`
  that cluster on `[M+^NO3]-`). A committed 14N line TRACKS X's cluster when its
  14N/15N ratio follows the batch's cluster ratio through the run (the per-spectrum
  median over the bright CHO acid clusters committed on `[M+^NO3]-`, the tested pair
  left out: ratio 0.5-2, sd(log) <= 0.25, r >= 0.8 over >= 50 of >= 100 co-detected
  spectra). A tracking line unties the arbiter's tie with the same-ion organonitrate
  [X'-H]- where every file's tie is with that alias alone (`label_untie`, recorded; it
  clears the pooled pair's `tied` fact, so it can remove the `tie` tag, never a level); a
  line tested but noisy,
  or below the cluster ratio (an organonitrate can only add 14N: trifluoroacetic
  acid's cluster runs at the reagent's own impurity, 0.12x), still counts as X's
  cluster; a line above twice the cluster ratio, or whose 15N partner is absent in
  >= 80 % of its spectra, is cluster or organonitrate undecided and reads 5b
  (`label_veto`). In the other direction a
  labelled [Y+^NO3]- reading is refuted (5b) when the 14N twin its reagent must carry,
  (1 - purity) / purity of its height (`ReagentProfile.purity`), is absent from the
  spectra where the batch's own 13C detection curve says it would be seen and the scan
  reached it. Facts of the pooled batch only, never per file (the per-file level, the
  tier and the vote keep the arbiter's tie).
  `tables/label_twins.csv` holds every line's verdict and every reading's test,
  `batch_summary.json` the funnel (`evidence_levels.label_twins`), and
  `scripts/level_ledger.py` reads the table from the run directory. On the labelled-nitrate
  regression batch (one variable against the previous trunk run; the run equals the
  offline replay row for row): of 291 committed 14N lines 14 track, 18 are consistent,
  13 excess, 220 have no 15N partner and 26 are untestable; 8 tied rows untie; 2
  labelled readings are refuted (0 of 42 / 53 expected twin sightings); no merged row
  changes its ion or tier. The uronium and TOF batches are out of scope and do not
  move. `tests/test_label_twins*.py`. The measured 37Cl lock on Cl-free labelled
  readings is left to the isotope checks (a too-high heavy line), not to this rule.

- **A uronium neutral pair, measured on the batch's own time series (rule U).** A
  uronium channel has no acid branch -- the pattern that lets the anion channels say
  "the same neutral, deprotonated and clustered". Its equivalent is the protonated
  and the urea-clustered ion of one neutral, 60.0324 Da apart. A reagent profile can
  now declare such a pair (`ReagentProfile.neutral_pair`; the bundled uronium profile
  declares `([M+H]+, [M+(CH4N2O)H]+)`, every other profile none), and the batch
  measures it on its own stamped time series (`batch/neutral_pairs.py`): the neutral is
  committed under both adducts, carries only C, H and O, both ions sit at exact mass
  (within 2 ppm in at least half the spectra, signed median within 1 ppm), their heights
  co-vary (r >= 0.5 on the log scale), the stamp gives both to this neutral, and
  neither ion's 13C line contradicts the carbon count. The fact (`upair`) is recorded
  per pooled pair; no step of the evidence scale reads it. Batch only, and
  profile-scoped: only a profile that declares a pair is measured.
  `tables/neutral_pairs.csv` records every clause per neutral, and
  `batch_summary.json` the funnel (`evidence_levels.neutral_pairs`). The round-3 design
  also held pairs whose two lines drift apart through the batch's steps, and a
  fragment of a brighter parent; both tests read hand-dated steady states of one
  batch, no batch-generic form reproduced them, and they are not built.
  `tests/test_neutral_pairs.py`. On the uronium regression batch 167 of the 344
  neutrals committed on both adducts hold the pair.

- **The reagent-ion water ladder, measured on the batch's own time series.** A soft-interface
  CIMS carries its reagent ions hydrated -- Br-.(H2O)n, NO3-.(H2O)n, HNO3.NO3-.(H2O)n --
  and how far a ladder reaches moves with the source and the humidity: on the five-day
  bromide/nitrate TOF regression batch it ends at n <= 8 (Br-) and n <= 5 (NO3-) before an
  instrument restart and runs past n = 15 after it, where the rungs were committed as
  C1-C37 organics (C13H12O8 [M-H]- = Br-.(H2O)12, C13H22N2O4 [M+NO3]- = NO3-.(H2O)15). The
  reagent library declared only the first halide rung and no nitrate rung at all. A new
  batch step (`peaky/batch/reagent_water.py`, run once after the vote) looks for
  core.(H2O)n, n = 1..45, of every core the profile declares (`ReagentProfile.water_cores`:
  Br- / Br2-. / Br3- / HNO3.Br- on the bromide profile, NO3- / HNO3.NO3- / (HNO3)2.NO3- /
  NO2- on the nitrate one, the labelled cores on 15N-nitrate, none on the positive
  profiles; `compose()` unions them; a config profile may declare its own) per
  acquisition segment (a gap longer than max(60 min, 5x the median spacing) cuts; a
  segment under 10 spectra joins its neighbour). A rung passes when, in some segment, the core and every
  lower rung are present in >= 50 % of the spectra within the stamping window and the
  rung is >= 3x the presence of its decoy offsets (its 0.02 floor never binds while the
  presence bar is 50 %). A merged analyte row on a passing rung
  leaves the merged ledger as the water cluster it is; the rung becomes a reagent row of
  the batch stamp. `tables/reagent_water.csv` lists the rungs and the readings each
  displaced, `batch_summary.json["merge_gates"]["reagent_water"]` the counts. The
  per-file ledgers are untouched; nothing is tiered or levelled by it. Measured offline
  before the build and reproduced by the run: 114 passing rungs and 70 displaced rows on
  the TOF batch (42 Assigned); the unexplained share of its time-series signal falls from
  16.95 % to 5.75 %, and nothing else in the merged ledger changes. 0 rungs on both
  Orbitrap batches, whose outputs are identical (the labelled-nitrate window starts at m/z 130, above five of its seven cores, and the two
  trimer cores inside it are absent; the uronium profile declares none). Known limits:
  the rungs are measured per segment but stamped and stripped batch-wide, so one Assigned
  row that is a real species before the restart and the water rung after it left with
  the rung; and a halogen rung passes for one isotopologue while its twin fails the decoy
  gate on neighbouring peaks, leaving about 60 merged rows on unstamped rungs.
  `tests/test_reagent_water.py`.

- **Claims in every output -- what a committed formula lets a reader say.** The evidence
  levels say how good the evidence is; a reader of a result table asks something coarser -- may
  this formula be reported as a compound? -- and the tier (print or offer) does not answer it.
  `evidence.claim_class(level)` reads a level as a claim (the scale's entry above), with
  `CLAIMS`, `CLAIM_IDENTIFIED`, `CLAIM_NEUTRAL`, `CLAIM_ION`, `CLAIM_MEANING` and
  `summarize_claims` beside it. `claim` is a column the `evidence` stage writes: on every
  committed M0 row per file (from the file's own level; empty on isotope children, reagent ions
  and unexplained peaks -- only a committed formula makes a claim), on every merged row (from the
  pooled level; a batch-level re-read with no level reads tentative) and in
  `tables/evidence_levels.csv`. The stage summary and its log line carry the tally (`claims`);
  `batch_summary.json['claims']` holds it per merged row, per pooled pair, per stage and per tier
  (an ion-only row under its own `ion-only` key) with `n_unlevelled`; `run_manifest.json` counts
  `merged_claims` beside `merged_tiers`; `peaky assign` and `peaky batch` print the tally; the MCP
  tools return it (`assign_sample` with each top species' claim); and `publish` carries `claim` in
  the engine provenance and `claims` in a batch run's config -- never in a tier field, where the
  `engine_tier` map reads the legacy spelling `Identified` as Assigned. The single-sample workbook
  opens on a **By claim** sheet (per claim: count, share, signal, the Assigned / Candidate /
  ion-only split and the level histogram; then the rows where tier and claim part, brightest
  first; then the twenty brightest rows of each claim), with `claim` directly before
  `evidence_level` on every sheet that shows the level, `best_claim` on Unique formulas, a Summary
  **Claims** section and a Read me that opens on the classes; the summary markdown gains a
  `Claims:` line; the PDF report leads its cover summary with the claims and follows the cover
  with a **Claims** page (merged rows and committed signal per claim, the tier x claim crosstab,
  the brightest rows where the two part), and the findings and the appendix carry the claim. A
  ledger levelled on an older scale shows its rows as no level (tentative) and says so; a run
  without levels renders unchanged. `scripts/scorecard.py` leads the card with the claim (§0:
  rows, the tier x claim crosstab with an ion-only row, signal per claim, the disagreeing rows),
  counts each decoy arm's pairs per claim on either side of m/z 350, reads the acceptance criteria
  on the scale beside the metric each was read on before, keeps each arm's engine ledger for a
  re-count at no engine cost (`--decoy-ledgers`), and puts a claims table first on the board and
  the page; `scripts/ab_compare.py` counts the tier, level and claim changes per shared ion and
  the claim histogram. Additive: nothing upstream reads the claim -- not the tiers, not the merge
  vote, not the cross set -- so no ion, tier or level moves. Tier and claim are separate verdicts
  and are not nested; the outputs show both side by side (the workbook, the PDF and the scorecard
  list where they part), and neither is corrected from the other. `tests/test_claims.py`,
  `tests/test_claims_outputs.py`, `tests/test_scorecard_claims.py` and the claim tests in
  `tests/test_ab_compare.py`.

- **Resolvability for every run, and two tier rules that read the spectrum's physics.** The
  nearest-neighbour separability flag was trace-first only, so every cover run -- every
  baseline -- rated `Assigned` without knowing whether the picked centroid was the ion's
  own. Now a batch measures ONE peak-width model from the raw profile of a middling spectrum
  (`batch.tracefirst.measure_resolution`; `--resolving-power R` declares one, `none` declines; the
  `assign` command takes the flag too and measures its own sample by default) and hands it to every
  per-file run, and the new **`resolvability`** stage (`assignment/resolvability.py`, before
  `degeneracy`) stamps `resolvability` / `sep_hwhm` / `d_crit_hwhm` on every M0 row from the
  sample's own picked peaks (synthetic sub-peaks excluded; skipped without a model, the columns
  stay NA). The width model itself moved to **`chem.resolution`** (`Resolution`, `classify_pair`,
  `nearest_neighbour_classes`; `batch.tracefirst` re-exports the names it used to own), the
  merged row carries the winner file's class, `batch_summary` records the model (`resolution`)
  and the summed class counts (`resolvability`), and the scorecard hands the recorded model to its
  offline decoy arms so they are rated by the same rule as the run they bound. **Tier rule 1,
  separability:** a `blended` / `unresolvable` M0 with no isotope / second-channel / series
  corroboration is Candidate -- the nearest picked peak sits inside the bimodality separation for
  the pair's height ratio, so the centroid is displaced and the mass the formula was fitted to is
  not the ion's own. **Tier rule 2, the satellite verdict:** the ledger-based twin test pass 0
  applies to a refused known-species claim -- now `assignment/satellites.py`, one implementation
  for both callers -- is applied to every committed row whose neutral carries Br, Cl or S: a
  diagnostic line predicted at >= 4x the file's noise edge that is absent within 15 ppm, or present
  under 0.6x its prediction, REFUTES the count whatever else corroborates the row (a series step
  or a second channel cannot put back a line that is not there); a line predicted under the
  multiple leaves the count untested, and an untested count with no other corroboration does not
  earn Assigned ("for want of evidence, not against it"); the ion's atom count predicts the line,
  the neutral's element is the one tested, and a reagent adduct's own Br / Cl sits in the same
  M+2 window and masks the test (untestable, never refuted). Si keeps its own rule (on a TOF its
  M+1 is unresolved from 13C). Measured before the change on two same-air batches (per-file
  Assigned M0 rows): a ~10k TOF has 51 % of its M0 peaks blended -- 126 Assigned rows blended with
  nothing else, 56 with an untestable heteroatom, 8 refuted (one organophosphate read as Assigned
  in every spectrum with no 37Cl line where 40x the floor was predicted); an Orbitrap (R ~120k at
  m/z 200) has 100 Br / Cl series and completion commits whose 81Br / 37Cl line is predicted at
  4x the floor and absent, and 45 blended uncorroborated rows.
  A predicted line height under 10 cps is written with one decimal in the verdict's reason (a TOF's
  34S line of a 5-cps parent is 0.2 cps, not "0 cps"); text only.

- **Known species decided once per batch, by pooled evidence — the vote's curated exemption is
  retired.** Pass 0 locks a known species where THAT file shows the corroboration its family
  demands (two ion channels, or a diagnostic 29Si/30Si / 34S / 37Cl / 81Br twin; exact mass alone
  for the monoisotopic families) and refuses it elsewhere, and on a batch the twin clears the
  picker's floor in one file of ten: the D7 cyclosiloxane urea adduct was `known:` in one file and
  grid-fit as an O14 formula the engine itself flags implausible in the nine others, and the merge
  vote kept it only through a rank exemption for "curated" labels. Now every refused on-cal claim
  leaves a **`known_lead`** on its peak (`passes.directors._record_known_lead`, a new ledger column:
  formula, family, label, adduct, ion, m/z, ppm, score, channel count, and a verdict judged on the
  LEDGER, never on the scorer's silence — `refuted` when a diagnostic line the file could show
  (predicted at 2x the resolved gate) is absent or under 0.6x its predicted height, or the
  own-81Br-twin ratio failed; `deferred` when nothing could be tested or every testable line is
  present and consistent though the scorer did not credit it; for Si the 29Si line ALONE and the
  30Si line, because an Orbitrap resolves 29Si from 13C above ~m/z 300 and a blended M+1
  prediction over-demands by the 13C share — the D5 urea adduct shows both lines at their predicted
  ratios in every file of the uronium reference batch and the scorer credited two; an absence, or a
  picked ratio, counts only for a line predicted at 4x the per-file noise edge — measured: the
  instrument labels a centroid above S/N 1.8 and the per-scan noise is 1.6x the per-file edge, so a
  weaker line reaches the per-file list in a fraction of the scans, censored low or not at all — two
  of the three "refutations" of the D7 cyclosiloxane were lines present in 3-6 of 23 scans), and the batch
  pools those with the `known:` commits
  (`assign_batch.known_evidence`; a commit's evidence is read off its own row — the commentary's
  "corroborated by", else the recorded satellites of an element the neutral contains, a paraffin's
  37Cl envelope, else exact mass) and decides each known ion ONCE on the merged ledger after the
  vote (`assign_batch.lock_known_species`): confirmed in at least one file and refuted in none →
  the cluster takes the known reading whatever the count (the confirmed files' tier / score /
  admission provenance; the vote's winner to the head of `alternatives`; the evidence and what it
  overrode in `tier_reason`); confirmed and refuted → left to the vote, the conflict on the row,
  and the merged tier capped at Candidate where the row's own reading is the conflicted species
  and the refutations outnumber the confirmations; never confirmed → a `known-species lead` note;
  a species of a corroborated family is Assigned only when some confirming file holds two independent
  lines beyond the exact mass (a second ion channel of the neutral, or two own-element diagnostic
  satellites, counted from the ledger), else capped at Candidate, locked or kept — the D7
  cyclosiloxane on the uronium reference batch (two files, one channel, the 29Si line alone; its 30Si
  line sits below the label threshold in every scan) reads D7 at Candidate, as the per-scan test of
  J12 supports;
  a family whose own rule is exact mass alone (the PFCAs, the nitroaromatics, the C0 acids) has
  nothing to pool beyond the count, so it never overrides the vote (a PFCA on-cal in 2 files of a
  ~4k TOF does not displace an 11-file 81Br-corroborated reading 6 ppm away; the row says so). The
  merged row a pooled reading belongs to is found by membership (the vote lists every losing reading
  in `alternatives`), the m/z window only for a lead no file committed — a minority reading's own
  m/z falls outside the merge window of a cluster the majority ion pulls 6-8 ppm away.
  The note counts files per reason (each lead carries the reason with the file's numbers, `why`,
  and without them, `summary`). Silence never votes against a species, a
  refutation does — sulfolane, 34S-confirmed in one file against fluorenone `[M+H]+` in nine bright
  files that show no 34S, stays fluorenone by evidence where it used to by count. `align(curated=)`,
  `_curated_neutrals` and `_CURATED_METHODS` are gone; `batch_summary["merge_gates"]["known"]`
  records the counts. `docs/MERGE.md` §3 step 4b, `docs/ASSIGNMENT_DETAIL.md` §3.0, `docs/OUTPUTS.md`.

- **Ion-only electron-attachment rows — the `ion_only` stage.** On a nitrate CIMS the bright O-rich
  acids show a second line +1.0078 Da (one H) above their `[M-H]-`: the acid's own composition as a
  radical anion, exact to 0.05 mDa and pinned by its own 13C, yet anti-correlated with the acid
  and switching 25x within an hour with the source — a source-state effect (a primary ion attaching
  to a high-electron-affinity compound), not chemistry of the air. A new post-tier stage
  (`cleanup.commit_ion_only_electron_attachment`, after `reflist_rescue`, before `iso_env_final`,
  not `safe`) commits, for every committed `[M-H]-` acid (any tier, not below assignability, at
  least one carbon), the UNEXPLAINED peak at the acid neutral's M-. mass inside the calibrated gate
  (|z| <= 2.6 via `z_of`; +-3 ppm uncalibrated) as a **Candidate `[M]-.`** carrying the acid's
  composition: `method ion_only:electron_attachment`, pass 9, `confidence "Good (ion only)"`, a new
  ledger column `ion_only_of` (the parent's peak id), the shift in mDa and the height ratio in the
  commentary; never an anchor or series tie, never locked; the parent row is not touched. Two
  separability guards keep a 13C line or an unresolved 13C/+H blend (4.47 mDa apart) out of the
  bucket: the gate's half-width at that mass must be under half the gap, and the file's own picked
  peaks must show at least three adjacent pairs at <= 1.25x the gap within +-50 Da of the parent
  (an Orbitrap below ~m/z 350 picks hundreds per file; a ~4k TOF none anywhere, so every parent is
  skipped and the log says so; an Orbitrap above ~m/z 400 skips too). Never a grid channel, and
  nothing existing moves: the final envelope sweep attaches only UNEXPLAINED satellites to an
  ion-only parent (never displacing a committed M0 -- the row's 13C sits 0.3 mDa from where a weak
  cluster reading of another neutral can be), and at the batch merge an ion-only reading ranks below
  every regular reading in its cluster however many files carry it (the radical anion of a C_n acid
  is 0.44 mDa from the labelled-nitrate cluster of the C_{n-1} organonitrate), going to
  `alternatives` with a `tier_reason` note. Opened
  by the reagent profile: `ReagentProfile.ion_only_channels` (the nitrate profiles `NO3` / `NO3_15N`
  declare `("[M]-.",)`; `compose` unions it; a `--reagent-config` entry may list it), copied onto
  `PassConfig.ion_only_channels` by `profiles.apply_ion_only_channels` at every entry point with the
  height-gate rule (a cfg that already carries a tuple, `()` included, outranks the profile; None =
  unset). The final envelope sweep then claims the new row's own 13C; the evidence scale reads an
  ion-only pair 4b at best (the channel reads the ion only: the neutral and the process stay open),
  and the merge vote's class gives the parent no second channel from it and never counts it in the
  `--corroborate` cross set; `tables/evidence_levels.csv` carries an `ion_only` flag. A batch carries
  `ion_only_of` onto the merged row (the winner file's parent) and writes
  `batch_summary.json['ion_only']` (`channels`, `merged`, `per_file_rows`, `n_files_with`,
  `merged_levels`). `scripts/scorecard.py` counts the bucket on its own (`ion_only`, a board column
  and a tile) and keeps it OUT of the Candidate count. `publish` sends a null mechanism for `[M]-.`
  (no server mechanism is mapped). `tests/test_ion_only.py`.

- **`--trace-first` (batch): assign the batch's persistent ions ONCE, from their centred
  traces.** Opt-in and EXPERIMENTAL — see the measured result at the end of this entry.
  `--resolving-power` defaults to measuring the width from the raw profile. `peaky.batch.tracefirst` builds
  the traces (`PeakIndex` seeds recurring in >= 5 % of spectra, brightest first, each
  consuming its 0.4-HWHM dedup cell — the Cubison & Jimenez fit floor, below which two
  positions are one observable), centres each one adaptively (`batch.centre`), measures
  the axis against the reference ions and applies the fitted wave inside its calibrant
  range (`batch.massqc` / `batch.wave`), sizes membership from the reference ions'
  per-spectrum noise (4 sigma, never below the validated 12 ppm, never above half the
  cell), rejects a seed whose members scatter as widely as a uniform fill of that window
  would (their robust sd against the 0.74 W a fill gives; a fill recurring in half the
  spectra is kept and flagged `fills_window`),
  probes the isotopologue positions of every persistent trace (20 members and 60 %
  co-occurrence with the parent, no width test — a dim satellite reads as a fill by
  nature), stamps each trace's separability from its nearest neighbour (`resolvability`:
  unresolvable / blended / resolved, a flag never a filter), and hands the engine ONE
  synthetic sample — one peak per trace, batch-mean height (absent = 0), the co-registered
  estimator the isotope ratios rest on. That sample goes through `assign.run` and then the
  same merge, trace reconciliation, stamp and residual stages as a cover file (`n_files`
  is 1; the residual stage still picks real files for what the trace stamp left
  unexplained). `tables/traces.csv` carries every trace's measurements, `per_file/`
  the trace ledger with them merged in, and `batch_summary.json['trace_first']` the
  build's numbers and the mass-qc verdict.

  **What it measures against a file cover, on a 4-day mixed-reagent TOF batch (230
  spectra), same reagent and same commit.** Trace-first: 1,443 merged rows, 258 Assigned,
  53 ion disagreements, 11 files. The cover: 2,645 rows, 618 Assigned, 802 disagreements,
  28 files. Of the 1,141 neutrals the cover found in >= 2 files, trace-first recovers 547
  (47.9 %). Split by peak brightness, the two meet on the brightest ions (57 % vs 61 %
  Assigned above 50x the noise edge) and diverge in the working range (12 % vs 29 % at
  5-10x). The cause is structural, not a gate: the isotope satellite that earns Assigned
  sits in the same SPECTRUM as its parent, and a trace sample averages the batch into one
  peak per ion — 22 % of Assigned traces carry a confirmed isotopologue against 10 % of
  Candidates, and the rate collapses below 20 % occurrence. Trace centres are genuinely
  precise (standard error 0.20-0.85 ppm against 0.83 ppm per file), so use this for
  batch-level centred masses and for mass-qc, not to replace the cover. Untested end to
  end: the 0.4-HWHM dedup cell (worth 4-9 points of offered positions on that batch) and
  the hard-coded 5 ppm engine commit tolerance in `TRACE_DEFAULTS`, which is ~14x the
  centre's own standard error here but is not derived from it.

- **`--trace-episodes` (batch, with `--trace-first`): seed a trace BELOW the occurrence
  floor for a short plume.** Off by default. The floor asks an ion to recur across the
  batch, which an episode never does — 41 of those 1,141 cover neutrals sit under a 5 %
  occurrence floor and never seeded at all. A second pass seeds when a candidate's
  detections are packed into <= 5 % of the campaign (a contiguous run of k spectra reads
  ~k/2n; scattered detections read 0.37-0.43) and it reaches 2x the batch noise edge.
  Contiguity does the discriminating: of the 24 reachable ions only 10 reach 3x the edge.
  It runs after the seeds, so an episode never takes a persistent ion's dedup cell. Off by
  default because it offers 322 more positions for 20 the cover confirms, and no
  end-to-end run has yet said what the other ~300 are.

- **`assign.run(..., peaks=frame)` runs the engine OFFLINE.** `io_mascope` serves a
  registered in-memory table as the sample (`register_offline_sample`), the mechanism
  lookups resolve to the names themselves for the channels the sample declares (the
  opportunistic extra channels stay closed, as on a server that does not list them), and
  the local scorer does the maths — no connection, no cache. The trace-first path and
  the tests use it; `tests/test_tracefirst.py` runs the whole engine on a synthetic trace
  sample and finds its acids and their 13C.

- **`--rolling-centre` (batch and pool): the adaptive centre in the trace reconciliation, a
  stamping window per trace, and a stamp that follows a moving centre.** Off by default;
  the default path is bit-identical to before (`tests/test_rolling_stamp.py` checks it). With
  the flag, `timeseries.recentre_ledger` hands every merged ion's trace to `batch.centre`:
  the per-spectrum noise `sigma_ppm`, the random-walk step `gamma_ppm` and the members'
  scatter about their own centre `resid_ppm` land on the merged ledger with
  `centre_scheme` / `centre_window` / `track_span_ppm`, and an ion whose drift is
  resolvable and worth following gets a rolling centre (its `mz_trace` becomes the track's
  median, the track itself goes to the stamp). `timeseries.stamp_tolerances` then sizes the
  stamping half-window PER TRACE from that residual — the same
  `max(tol, min(2 tol, 2.5 sigma))` rule as the batch window, applied row by row — instead of
  one batch quantile for every ion (`stamp_tol_ppm` on the merged ledger; the batch window
  stays the fallback). `annotate_peaks` reads `stamp_tol_ppm` per row and, given the tracks,
  collects a rolling ion's candidates within its window plus half its track span and keeps
  only those within the window of the centre interpolated at the peak's own timestamp — so
  the window MOVES with the ion, and the one-to-one contest measures distance from the
  moving centre. Without timestamps on the time series the rolling path is skipped and says
  so. `batch_summary.json['traces']` records `rolling` (how many rows rolled, median window
  and span) and `stamp_tol_per_trace`. Measured on synthetic batches: an ion wandering
  ±12 ppm over 400 spectra is stamped in all of them with the track and in under 70 %
  from its median; a static ion is untouched.

- **`peaky mass-qc` — the batch's mass axis measured against an EXTERNAL reference.**
  Pass 1 self-calibrates on the Assigned backbone, which is circular: an axis 30 ppm wrong
  yields a self-consistent calibration and a batch of confident wrong formulas, and nothing in
  the run log says the axis moved. The new command probes formula-certain reference ions
  (`peaky.chem.reference_ions`: the 30-ion nitrate-CIMS core shipped as
  `data/reference_ions/nitrate.csv` with grade / anchor / blend flags, its 15N-reagent variant
  computed through `chem.ion_mz`, and a provisional bromide ladder whose every Br adduct
  carries its 81Br twin) in a batch time series — live (`--batch`/`--dataset`) or offline
  (`--ts parquet`, no credentials) — and reports per ion the occurrence, offset from theory,
  per-spectrum noise, random-walk step and optimal window (`batch.centre`), then a verdict
  with the remedy it implies: `clean`, `axis_offset` (flat bias → a constant), `axis_trend`
  (a smooth wave → apply it), `blended` (ion-to-ion jumps, no smooth part → centre only,
  widen the tolerance, cap the tier), with `drifting` appended when the centres roll.
  `peaky.batch.wave` is the written model behind `axis_trend`: Chebyshev in `(m/z)^+1/2`
  (TOF, flight time) or `(m/z)^-1/2` (Orbitrap, frequency), the degree chosen by leave-one-out
  cross-validation over 0..5 — **0 included**, so a flat offset is expressible — under a
  one-standard-error rule with an L1 score (a MAD-based score let noise buy a degree on 25
  points), iteratively 3-sigma clipped about the median residual, and refusing to predict
  outside the calibrant range. Two procedural rules from a calibration session that got them
  wrong: every gate is absolute (a "below the batch median" gate deletes every persistent ion
  of a TOF batch whose median jitter is 0), and a calibrant rule that relaxes when the strict
  set is too small REPORTS WHICH TIER IT USED (`calibrant_tier`, `calibrant_tiers_tried`).
  The 81Br twin is a free internal check: a Br adduct whose light and heavy lines disagree in
  offset, height ratio or occurrence is withdrawn as a calibrant whatever the formula says.
  Writes `mass_qc.csv` / `mass_qc.json`. No assignment behaviour changes.

- **`peaky.batch.centre` — the adaptive trace-centre estimator.** A trace's per-spectrum
  positions carry white noise `sigma` and a slow random walk `gamma`; both come out of the
  trace's own structure function `S(k) = 0.5 · robust_var(x[i+k] − x[i]) = sigma² + 0.5 gamma² k`,
  with no user parameter and no assumption about batch length. A rolling median over `W`
  spectra has error `sigma²/W + gamma² W/24`, minimised at `W* = sqrt(24) · sigma / gamma`;
  `trace_centre` rolls only when the series is long enough (`min_n`), a walk is resolvable
  (`S(kmax)/S(1) >= min_rise`) and the predicted gain beats the batch median by `min_gain` —
  otherwise the centre is the batch median, bit for bit, so short batches behave exactly as
  before. A gap guard keeps a window from spanning a hole in the batch. `rolling_members`
  re-collects a drifting trace along its own track and `batch_centres` runs the estimator
  over a `PeakIndex`. Nothing calls it yet; `tests/test_centre.py` pins sigma/gamma recovery
  within 15 %, `W*` within 1.5× of the empirical optimum, never-roll on pure noise,
  always-roll on a walk, and the bit-identical median below `min_n`.

- **`PassConfig.audit_floor_cps`** — the detection floor the post-run isotope audit
  judges 13C satellites against (`postprocess.audit_isotopes`: "would the satellite be
  comfortably visible?" and "is this measured satellite reliable?"). Default `None`
  keeps the resolved height gate, which is right when the sample is a spectrum. Set it
  when the sample is a derived table on a different footing: a batch of trace-averaged
  heights (absent = 0) compresses every dim ion far below the per-spectrum floor its
  satellite must clear to be picked, so the gate — a multiple of the table's own 1st
  percentile — predicted "visible" satellites no spectrum could show and the audit
  cleared 178 of 258 otherwise-silent formulas on a 766-spectrum TOF trace sample
  (succinic acid among them).

- **`PassConfig.audit_sat_ppm`** — how far from parent + 1.00335 the isotope audit
  looks for the 13C satellite (sweeper, completeness check, halogen-twin fallback).
  Default 5.0 is the Orbitrap ruling. A ~4k-resolution TOF blends 13C with the +H
  isobar of a neighbouring homolog into one M+1 peak whose apex sits up to 4.5 mDa
  (25 ppm at m/z 180) from the 13C position: on the same TOF trace sample 108 of 131
  remaining missing-13C clears had a clean satellite trace 5–25 ppm away (succinic acid's
  at 16.9 ppm). Set it to the instrument's M+1 blend.

- **`PassConfig.pass0_ppm`** — the mass gate of the pass-0 known-species commit
  (`directors.run_pass0_known`), previously hard-coded at 2 ppm. Default 2.0 is
  unchanged. On a TOF whose weakly bound clusters sit 7–11 ppm high, the iodine acids
  (`reactive_iodine` family: HOI, HIO2, HIO3 …), H2SO4·NO3⁻ and MSA could never commit;
  raise it to the instrument's displacement. The nitrate profile now documents that
  iodine reaches it through this family on the [M−H]⁻ / [M+NO3]⁻ channels, and that
  iodine stays off the neutral grid on purpose.

- **`scripts/ab_compare.py` — an A/B between two `peaky batch` runs over one batch.**
  `scripts/diff_runs.py` asks "same peaks, same formula?" and keys on
  `(sample_item_id, peak_id)`; it cannot see a path change, because a cover of
  files and `--trace-first` (or a fixed centre and `--rolling-centre`) rebuild the
  peak set from scratch and share no peak ids at all. The new script keys on
  chemistry instead and reports, from `merged_ledger.csv`,
  `per_file/_batch_ts.parquet` and `batch_summary.json`: the tier and neutral
  headline; the merged rows that pair across the runs within a ppm window yet read
  a different `(neutral_formula, adduct)`; the per-ion share of samples carrying a
  series, with every move above a threshold; and the headline number for a path
  change — the share of the reference run's multi-file neutrals the challenger
  keeps, split by occurrence and median m/z. It reads an `evidence_level`
  histogram when the column is there and says so when it is not. A run dir may be
  named directly or by the `--out-dir` that holds it. The `--rolling-centre`
  block is read from both shapes it is written in — `traces.rolling` on the cover
  path and a flat `n_rolling` under `trace_first` — because reading only one of
  them described a batch that rolled 965 of its ions as a fixed-centre run. The
  headline also breaks the merged rows down by selection stage and refuses to
  compare two runs that did not run the same ones: a run with the residual stage
  on carries rows a run without it never looked for, so crediting the difference
  to whatever flag was under test reads a stage as an effect. A run dir that
  predates the stage field reports `n/a`, not zero.

  First use settled three open questions. `--rolling-centre` is inert on both
  instrument classes: same commit, same batch, flag on against flag off gives
  0 ion disagreements in 1148 rows on an Orbitrap and 0 in 7346 on a TOF, even
  though 434 and 965 ions roll and the TOF centres move by a median 1.4 ppm.
  `origin/main` against the five-PR stack on the same Orbitrap batch is
  likewise identical. And `--trace-first`, run on an Orbitrap for the first
  time, recovers 96.3 % of the cover path's neutrals against 47.9 % on a TOF,
  so the recorded "the deficit is structural" reads as instrument-dependent.

- **`scripts/scorecard.py`: M3 also counts by the other instrument's own evidence.** "Found by
  the other instrument, missing here" read the other run's merged in-core level, and on a same-air
  pair that level can owe a series anchor to the other run's own `--corroborate` partners -- the
  run being scored -- so the metric could hold this run's agreement against it. The card now counts
  M3 a second time with the other instrument's per-file ledgers levelled with no partners
  (`own_levels_for`): `m3_other_instrument_own_missing` on the board row beside the unchanged
  `m3_other_instrument_missing`, a line and a table in both card renderings.

- **`scripts/scorecard.py` — what a `peaky batch` run assigned, how good it is, what
  it missed, the same way every session.** One card per run (`SCORECARD.md` +
  `scorecard.json`), one row appended to a scoreboard (`scoreboard.jsonl`), and the
  board (`SCOREBOARD.md` + the page source `scoreboard.html`) regenerated over every
  channel's latest row with its delta to the row before. It reads a finished run dir
  only — the merged ledger, the per-file ledgers, `_batch_ts.parquet`,
  `batch_summary.json`, `run_manifest.json` — and never a server. Six sections: the
  headline with **stamp coverage** (merged rows the time series never carries — a
  stamping defect, not a chemistry question); the brightest 50 ions with reading,
  tier and evidence level, and the count of bright M0 rows that are not
  Assigned; the best-evidence 50 with the level vector;
  what was missed in three senses — **M1** the brightest unstamped tracks in >= 50 %
  of spectra, each joined to the nearest stamped ion, to `characterize_residual`'s
  tag and to the engine's own clearing reason, then **grouped into families by
  their shift from the nearest parent** so a comb is one row (+H, isotope lines,
  cluster lines, neutral losses, reagent-ladder lines, shoulders; a ±H tie between
  two acids that differ by H2 reads as +H), **M2** every roster / reference-ion /
  known-species neutral looked for on the run's own channels (assigned, candidate,
  read as something else, present but unstamped with the reason, absent), **M3**
  what the other path or the other instrument on the same air found at level <= 4b
  above the detection floor inside the overlap window and this run lacks; is it
  right — roster recall against cited rosters (`peaky/data/rosters/`, alpha-pinene
  products and the usual contaminants; unreviewed until signed off), the element
  census of Assigned neutrals, the **decoy false-discovery bound** (the engine run
  offline on the brightest cover file as it is, with every m/z shifted by 0.35 Da,
  and with the wrong polarity's adduct set — what is still Assigned is the error
  bound, split at m/z 350 where the mass-defect gap closes; an arm that crashes
  the engine is recorded, not fatal) and **falsification survival** (the 13C-implied carbon count against the
  formula, the 34S / 37Cl / 81Br / 29Si line where the formula demands one, the time
  covariance of satellites and adduct pairs with their parent); and the delta. The
  level column is the merged ledger's in-core `evidence_level` when the run wrote the
  scale's columns, else a `scripts/level_ledger.py --out` CSV given with `--levels`, else
  the run levelled post hoc by `scripts/level_ledger.py` in-process. Not built from
  `tables/residual_bins.csv`: it lists only bins absent from every cover file, and
  every headline miss of the first cut sat inside the cover files.

  `scripts/level_ledger.py`'s null-safe truthiness now also covers `pd.NA`, which an
  in-memory engine ledger carries where a CSV round-trip has NaN.

  First use, on the existing run dirs (Scoreboard v0): four channels of one same-air
  campaign, six run dirs, one card each, ~8-15 min per card with the decoy. A 15N-nitrate
  Orbitrap batch (305 spectra): 633 Assigned / 681 Candidate / 1,138 neutrals, 93.3 % of
  the signal stamped, 0 merged rows unstamped; the +4.5 mDa comb beside its bright acids is
  one M1 family row -- `+H` of the `[M-H]-` parent, the M-. electron-attachment family, 66
  tracks and 2.6 % of the signal -- and 535 of the 746 Assigned rows that demand a
  halogen or sulfur isotope line have none; the +0.35 Da decoy keeps 19 % of the
  brightest file's Assigned rows, every one an H-rich CHOS formula above m/z 450 where
  the shift is absorbable. A mixed-reagent TOF batch (230 spectra): 618 / 2,027 / 2,320,
  83.2 % stamped, the reagent's Br-.(H2O)n lines named; the decoy keeps 108 % of the
  control's Assigned rows on the shifted axis and 183 % on the wrong adduct set -- the
  TOF's Assigned tier is not mass-discriminating on that file -- and only 133 of 436
  13C carbon counts land within one carbon. A uronium Orbitrap batch (319 spectra):
  857 / 291 / 801, 97.7 % stamped, 7 merged rows unstamped and they are exactly the 7 rows
  the merge's reagent-N re-read relabelled; the shifted axis keeps 0.3 % of Assigned rows
  but the wrong adduct set keeps 74 %, because the adduct mass difference is absorbed into
  the neutral formula (C22H42O6 on the urea channel reads as C23H43NO7 on ammonium): the
  ion is pinned by mass, the neutral/adduct split is not. The same batch under
  `origin/main` and under the stack: zero delta on every metric. Its `--trace-first`
  run: 792 / 400 / 866, 35 of the cover's 744 multi-file neutrals absent, 11 of its own
  503 absent from the cover. One engine defect surfaced and is left for its own fix:
  `tiers.compute_tiers` reads `admitted_by` with a bare `or`, which raises on `pd.NA`
  when the wrong-adducts arm runs on a negative file; the card records the arm's error.

### Changed

- **An isotope child is evidence at its exact spacing from the committed line, in its
  count-aware band.** The scorer commits an ion's most ABUNDANT isotopologue (a Br2
  ion on 79Br81Br, a Cl4 ion on a 37Cl line) and labels its lines counted from the MONO line
  (`81Br2`, `M0`, `13C+81Br`); peaky's own passes label theirs from the PARENT line
  (`2x81Br`, `81Br(pair)`, `81Br+13C`, `M+5`). The pair facts read a child by the part of its
  label before the first `+`, against a single-atom expectation relative to the mono line,
  wherever the line sat; a non-empty isotopologues list alone gave `iso`. Now
  (`chem/isotopes.judge_source`, a standalone twin in `scripts/level_ledger.py` pinned
  text for text; the pair facts the merge vote's class reads, docs/EVIDENCE_LEVELS.md §3.2
  and §13): the committed configuration is read off the
  parent's m/z (the most probable Br / Cl / S / Si one within 5 ppm on an Orbitrap-class width
  model, 20 ppm on a TOF-class one or without one; none -> the mono line); each label is read
  both ways and whole ('+' parts summed, the joint probability; `13C2` its own C(n, 2) x
  0.0107^2; `M+n` the ion's lines within max(12 mDa, FWHM/2)); the expectation is the line's
  configuration's probability over the committed one's, multinomial over the ion's atoms; a
  line credits the elements it adds to the committed line (a scorer `13C+81Br` on a heavy
  parent is its 13C line); a child counts only within max(1 ppm, 4 sigma(h)) of its exact
  spacing, sigma(h)^2 = a^2 + b^2/h self-fitted on the source's own `13C` children (binned MAD,
  weighted least squares, a >= 0.02 ppm, >= 40 children or no test), rescued by the calibrated
  parent position or a >= 3x brighter neighbour pulling it, `M+n` exempt, and per file a
  TOF-class file whose fit sits on the intercept floor untested; a dropped child is dropped
  for every fact; `carbon_ev` is a placed line that adds 13C; `reagent_only_iso` needs every
  line naming a heavy atom to name only the reagent halogen's (`81Br`, `81Br2`, `2x81Br`; a pure
  `M0` line names what it differs in from the committed line) and no kept, in-band line that only
  the ion's full halogen count makes (D4's full-count line, a POSITION rule: its heavy index
  counted from the line the parent is committed on, j = k - k_P, outside [-k_c(s), s - k_c(s)],
  s the adduct's atoms of the halogen, k_c(m) the heavy atoms of an m-atom ion's most probable
  line -- such a line is the neutral's halogen, whatever the pair's other lines; a line where an
  s-atom ion puts one stays the reagent's whatever its height, a line that changes no atom of the
  halogen never counts, nor does any line of a neutral without the halogen; counted from k_c(n)
  instead, as first built, every index of a parent stamped off its most probable line shifted --
  a 13C line of a Br2 ion stamped on 79Br2 read as the neutral's Br -- moving no fact on
  the three batches or the fixtures; the Br-free reading keeps the reagent's flag on
  one pair of the full replay, a dim reagent Br2 triplet read as a ladder rung; a
  parent committed on a HEAVIER line than its most probable one would read one 2-Da step of a
  single Br as the neutral's halogen, where a one-Br reagent ion puts a line too -- latent, every
  off-line parent of the three batches is committed on a lighter one, and an open question); the
  isotopologues list holds `iso` only through an entry that passes the same test;
  `iso_labels` lists the kept lines' whole labels. The Br-free `reagent_only_iso` hold of
  the isotope checks is released with it (its line expects 0: no isotope support). The width model
  reaches every path that reads the facts (a `--corroborate` run dir and a script source read
  their own `batch_summary.json`), and the script reads isotopologues lists JSON first like the
  engine.
  LANDING numbers, measured offline (an online run was blocked then: Mascope had renamed its
  ionization mechanisms), one variable against the trunk's final runs, whose per-file A.run
  ledgers, merged ledger and tables the offline tools reproduce byte for byte (every time-series
  stamp too). On the three regression batches' STORED per-file ledgers the test reads
  dibromoacetic acid's 81Br2 and 79Br2 lines around its 79Br81Br-committed M0 and no longer
  credits the only "halogen" line of C18H15BrO3S and C13H15ClO8 `[M-H]-`, which sits at the
  F-for-OH / 34S spacing (the chlorinated paraffin C10H18Cl4 `[M+^NO3]-`, whose recovery labels
  every ladder line '37Cl', is a separate card); the full-count rule moves `reagent_only_iso` on
  five pooled TOF pairs (IBr, C13HBrO4 `[M+Br]-` and C27H27BrN2O2 `[M+HBr+Br]-` lose it on an
  in-band 79Br2 / 81Br3 line; the Br-free C10H12O4 and C8H14O5 `[M+HBr+Br]-` take it on their `M0`
  line), 18 per-file rows and six fixture rows. The landing itself, the FULL offline replay of the
  final engine (every file's A.run on the stored scorer responses and the batch half, with the
  engine items below): the labelled-nitrate Orbitrap pooled pairs 1848 -> 1850 (four dim pairs
  appearing or disappearing: pass 4 at the exact Br spacing, one pass-7 re-read), merged rows
  1419 -> 1420, Assigned 647 -> 647, 632 time-series peaks restamped (0.018 % of the signal;
  restamped: the stamped reading changed -- neutral, adduct, role, ion formula, isotope label,
  stamp source or duplicate flag --, an m/z-only shift not counted); the uronium Orbitrap pooled
  pairs 1161 -> 1161, merged rows 1148 -> 1148, 373 peaks restamped (0.013 %), and one tier:
  merged Assigned 836 -> 835, hexamethylcyclotrisiloxane (D3) C6H18O3Si3 `[M+H]+` capped
  Candidate once I7 frees the 29Si line the trunk hung under it in four files (its urea adduct's
  line, 61 Da up; that adduct is committed in one file of ten, so every confirming file has one
  ion channel); the bromide/nitrate TOF pooled pairs 4317 -> 4305, merged rows 2239 -> 2241,
  Assigned 449 -> 450, 10320 time-series peaks restamped (1.41 % of the signal; among them m/z
  331.024, the 81Br line of a Br doublet at 329.025, now stamped C11H11NO5S `[M+NO3]-` in 99
  spectra, where three files hung it as an 81Br line under a two-channel C11H22O2S2 `[M+Br]-`
  certificate member and now leave both peaks unexplained), and chloroacetic acid keeping its
  `81Br+37Cl` M+4 line (the 13 mislinked 37Cl children gone). Of the TOF pooled pairs that move
  beyond the stored-ledger reading above (appear, disappear or read differently; counted on the
  development builds' levelling), 58 come from the per-file items -- 21 on pairs held only by the
  one file whose mass-calibration core fell to 19, under the 20 it needs, once the pass-0
  mislinks left it (C4H6N2O6S `[M-H]-` no longer committed -- its bromide cluster now its own
  pair, its two-channel certificate Low under option B, too weak to displace the pass-3
  C4H5NO3S `[M+NO3]-` reading of the `[M-H]-` peak --; two dim pairs appearing or
  disappearing), 23 other dim pairs and two pairs appearing or disappearing, and 12 others --, 4
  from pass 7's gate keyed on the committed ions (one dim pair appearing, three disappearing) and
  25 net from its reagent line counted only on >= 3 channels (option B; see the I7 entry): 26
  pairs move that did not -- among them C16H19NO4S `[M-H]-` and C10H21ClN2O6S `[M+Br]-` no
  longer committed and 19 dim pairs appearing or disappearing -- and one dim pair no longer
  appears (C3H5NO10 `[M+Br]-`).
  `tests/test_isotope_children.py`, `tests/test_isotope_levels.py`; every synthetic child of the
  level tests now sits at its label's exact spacing from its parent.

- **The ledger records which setter made a row a tentative lead (`lead_by`).**
  Per-file ledgers gain a column beside `tentative_lead`: the setter's code
  (`reflist_dim`, `off_budget`, `spec_n3` / `spec_gapfill` / `spec_minor` for the three
  speculative-residual reasons, `radical_anion`, `reagent_n`; pipe-joined where two
  mark a row), empty on every other row, created and reset with the flags. The Below
  assignability sheet and the published engine provenance carry it. Rule H reads it on
  the pooled batch, where the commentary that also names the setter is not kept; a
  ledger written before it reads "any setter" (a lock lifts a lead only where the
  element budget holds too). The evidence scale's `lead` tag names the setter; no level
  moves with the column.

- **A reagent-halogen isotope line is the reagent's only where the reagent put the
  halogen there.** `reagent_only_iso` (the pair fact "the sole isotope support is
  the reagent's own halogen", which the merge vote's class reads) now clears where the
  ion's halogen is all the neutral's own: a brominated neutral's `[M-H]-` or `[M+NO3]-`
  carries its own 81Br line (hypobromous acid's nitrate cluster at m/z 157.909). An ion carrying none of
  the reagent halogen keeps the flag: a 1:1 +2 Da line on a Br-free ion says it
  carries a Br its formula lacks, and is no support for that formula (the flag held
  until a count-aware isotope band judges it per file; the batch's HIGH check
  refutes such a reading). On the bromide / nitrate TOF regression batch 23 per-file
  M0 rows (15 pairs; pooled 13, 8 merged) lose the flag, all brominated neutrals'
  nitrate clusters; the Orbitrap batches do not move.

- **A tentative lead is its own flag (`tentative_lead`), apart from `below_assignability`.**
  `below_assignability` said two things: the assignment argues with itself (O >= 11 on
  a saturated formula, an O-monster, a carbon cluster, an implausible ionization, an
  off-calibration residual, unconfirmed F >= 4), or the proposal is only unsupported.
  The second half is now a per-row column of its own, written by the reference-list
  dim rescue, the element-budget demote, the speculative-residual demote (N >= 3 with no
  isotope, a series gap-fill with no anchors, a sole minor channel), an uncorroborated
  radical anion and the reagent-N re-read; every other setter keeps
  `below_assignability`, and a row both flags mark stays hard. The evidence scale pools
  the lead over any row and prints it as the `lead` tag on every level, with no level
  effect; the ion-only parent choice and the Below assignability sheet read either flag;
  the tier engine reads neither. `scripts/level_ledger.py` reads it in lockstep (a
  ledger without the column reads False). A commit, a clear or a displacement now resets
  both flags (`below_assignability` was never reset before; on the regression runs that clears it
  on six displaced isotope-child rows and moves nothing). Publish carries the column.
  This is the split the isotope checks and the halogen lock build on: a lock
  can answer a lead; nothing answers a contradiction. On the three
  regression batches (one variable against the previous trunk run): 0 of 1419 / 1148 /
  2239 merged rows change ion or tier; the lead moves from
  `below_assignability` to `tentative_lead` on 1288 / 37 / 1319 per-file M0 rows (128 /
  13 / 645 pooled pairs). `tests/test_tentative_lead*.py`.

- **`multiline` counts two elements the neutral supplies, each by an in-band line.**
  `multiline`, a pair fact the merge vote's class reads, is the only term inside one
  channel that backs the neutral. It counted raw satellite tags, so a 13C line
  with its 13C2 line (one element twice), the urea's own 15N on a urea adduct, the
  bromide's own 81Br on a bromide adduct and a generic `M+n` child all made "two lines"
  (2026-09-26 output audit). It now needs lines of two distinct elements, each within
  0.5-2x of its natural abundance, each supplied mostly by the neutral: more than half
  of the ion's atoms of the element. The 18O line of an oxygen-rich neutral's urea
  adduct counts (4 of 5 O); the 18O line of formic acid's nitrate cluster does not (2 of
  5). 15N and 18O join the ratio band, per atom of the ion like 13C per carbon, so they
  also count toward the `iso` fact. The evidence table records the elements
  (`multiline_elements`). An ion carrying Br or Cl does not have its 18O line measured:
  the halogen's 81Br / 37Cl line owns the M+2 region (one peak with it on a TOF). On the
  regression runs it no longer counts two chlorine lines (labelled nitrate), the urea's
  15N, 13C with 13C2 or an out-of-band 18O line (uronium), or the bromide's 81Br or
  nitrate's 18O on formic acid (TOF). `tests/test_multiline_elements.py`.

- **The ionization check covers N-only neutrals.** `cleanup.demote_implausible_ionization`
  demoted only a pure hydrocarbon read through an anion channel (`[M-H]-` or a cluster;
  electron attachment exempt): any N or O skipped the test, although an aliphatic amine has
  no acidic proton either. It now also demotes an **N-only** neutral (`cleanup.anion_implausible`):
  C >= 1, N >= 1, no S / P / halogen / Si, and O = 0, or O = 1 with DBE <= 1, or O = 2 with
  DBE <= 0 -- formulas no carboxylic acid or phenol fits. Same channels, same demotion
  (Assigned -> Candidate + `below_assignability`, so the pair levels 5b), its own note, and a
  second count in the stage summary (`ionization_demoted_n_only`). A heavy-isotope label folds
  into its element, so a `^N`-only neutral is judged as an N-compound instead of passing as a
  hydrocarbon. Why: a uniform +0.35 Da shift keeps every inter-peak spacing, so the shifted
  labelled-nitrate spectrum rebuilt H-saturated N2 formulas (C21H46N2, C22H46N2O, C25H54N2O2 as
  `[M-H]-` and `[M+^NO3]-`) on two channels; the census of this rule (seven definitions on
  the three regression runs and all decoy arms) costs 0 Assigned rows on the
  labelled-nitrate run and 7 on the TOF run -- the aryl-amine C13H12N2 / C13H14N2 family,
  plausible chemistry a formula-only rule cannot exempt. `tests/test_ionization_plausibility.py`.
  Run on both anion regression batches (one variable against the previous trunk runs, same
  inputs): the labelled-nitrate run changes one Candidate row (C5H13NO2 `[M+NO3]-`, now
  below assignability: an amino-diol the formula rule cannot tell from junk); the TOF run
  demotes the predicted seven Assigned readings, Assigned 497 -> 491. Two TOF peaks are
  re-read once their reading drops: m/z 332.146 goes to a Candidate nitrate cluster, and m/z 511.175 to
  another Assigned reading, C35H28O2S `[M-H]-`, which the peak's own
  1:1 M+2 doublet contradicts -- an isotope check that never refutes a too-high line, left to
  the isotope-check card. On the labelled-nitrate shift decoy the Assigned rows at
  m/z >= 350 go 87 -> 44; the control and adduct arms keep their
  Assigned counts there, and no decoy arm gains a row.

- **Flatness labels a row; it no longer tiers it.** The general flat-background
  demote added above capped every flat `Assigned` commit at `Candidate` whenever the
  run varies. It was written for a certified-mixture run, where the EasyIC
  calibrant's PAH ladder and the air-plasma C/N/O family sat at `Assigned` in a
  cylinder that contains none of them. But a flat trace says where an ion comes from
  (a steady inlet, the source, the calibrant), not what it is. On a varying chamber
  batch the general demote capped rows that their isotope pattern and a second ion
  channel confirm, nitric acid `[M-H]-` among them, and it moved no decoy rate. The
  verdict now lives only in `ts_disposition` (`background:flat`,
  `background:inlet/instrument ...`), and the tier stays with the identity evidence
  for every row, mass-only rows included. The certified mixture's calibrant
  background is labelled `background`, which is what a reader has to see. The
  summary reports `varying_frac` and `flat_informative` (formerly
  `flat_demote_armed`), so a background label can be read against whether the run
  moves at all. The merged batch ledger now carries the donor row's `ts_disposition`
  and `ts_cv_norm`, so the label reaches a batch reader as well. The reagent-cluster
  channels follow the same rule: the older channel demote, which capped a flat
  di-bromide or `[M+CO3]-` commit at `Candidate` although its label
  (`background:di-bromide cluster`, `background:CO3-channel`) already said
  background, is removed too. On a bromide TOF that cap took IBr and rows confirmed
  by their isotope pattern or a second ion channel. So flatness lowers no tier any
  more: `apply_timeseries` loses its `demote` switch and its summary the `demoted`
  count. Unchanged: the reagent-normaliser guard (`MAX_NORMALISER_CV`), every
  disposition, and the checks that compare a trace with another trace to test
  identity (a sidelobe locked to a bright neighbour, an adduct tracking its parent).
  On the same-air working set (one variable per run): the general demote had cost
  merged Assigned 647 -> 566 (labelled nitrate), 836 -> 737 (uronium) and 493 -> 357
  (~10k TOF), roster winners 23 -> 22, 20 -> 15, 12 -> 11 and bright M0 not Assigned
  1 -> 17, 2 -> 33, 5 -> 7, with no decoy rate moved. Without it the two Orbitrap
  channels return to their earlier results exactly (647, 836). Without the channel
  rule as well, the TOF reads 497: 18 di-bromide rows go back to `Assigned`, 4 more
  than before the general demote, because the channel rule held 6 rows there too. An offline replay of the merge predicted every row. Roster
  winners, bright M0, both decoys and the cross-instrument metrics are unchanged.

- **The degeneracy audit counts what the run could have committed.** `degeneracy.py` feeds the
  tier's degeneracy cap, `below_assignability` (O >= 11 on a saturated window), the O-monster demote and
  the merge vote's class (the evidence scale enumerates its competitors in the same space), and it
  enumerated ONE space for every run: the Br-CIMS adducts (`[M+Br]-`, `[M-H]-`, `[M+CO3]-`, `[M+HBr+Br]-`, `[M+HBr+CO3]-` -- the stage passed
  none), a fixed box (C <= 20, O <= 12, H <= 36, F <= 17, Si <= 3, Cl / Br <= 2, S <= 1) and the
  Br-CIMS contaminant caps. So a uronium (+) channel tried only anions and 83 % of its M0 rows read
  density 0, "unique"; a nitrate channel never tried `[M+NO3]-` / `[M+^NO3]-`; 70-80 % of all
  competitor slots on three channels were cross-family mixtures no pass can commit (F+Si+N,
  F+Si+Cl, ...), on the uronium channel F / Cl / Br formulas its context rules out; and the
  committed formula itself lay outside the box on 6-45 % of the rows (C > 20, O > 12, P, S2 ...),
  where density 0 then read "unique" -- the quantity an earlier "degeneracy is the
  wrong discriminator" finding rested on. The count is now every ion the run could have committed inside
  the calibrated window: **its channels** (the adducts it scored -- the detected reagent channels
  and the opportunistic ones the server resolved; a row on any other adduct adds its own), **its
  element space** (the context's element budget and filter; a cap the context sets to zero raised
  to a contaminant family's ceiling only where the file opened that family -- the context's
  declared pass-3 families, the reagent's organohalogen family, the GKA evidence pass 3 carried --
  plus the curated formulas, the pass-0 registry and the active reference lists:
  `degeneracy.opened_families`, `space_profiles`), and **no box**: C to the context's
  `grid_c_max`, O to `grid_o_max`, H from an integer DBE under Senior's cap and the structural
  oxygen cap. The enumeration is analytic and per peak (for each heteroatom combination the
  neutral mass is linear in C and the DBE) and proposes exactly the grid enumerator's formulas,
  window by window (tested), so the 9.7 M-formula relaxed grid and its ~40 s build per worker are
  gone: the audit takes 0.3-1.7 s per file instead of ~37 s. A commit outside the space makes the
  count a **lower bound** (the others plus itself): at >= 3 it still decides "degenerate"; below,
  the density is NaN, "not measured" -- never degenerate, no tier cap. The stage
  summary records the `channels`, the opened `families`, `not_measured` and `lower_bound`.
  `DEFAULT_ADDUCTS`, `RELAXED_BOX`, `relaxed_profile` and `GRID_MASS_MAX` are gone;
  `measure_degeneracy` / `apply_degeneracy` take `adducts`, `families` and `curated`. Sized before
  coding on the same-air working set (the C9 runs replayed: the audit re-measured per file -- the
  old default set reproduced the stored density on 26,820 of 26,835 rows -- then the tier cap, the
  below flag, the O-monster demote, the per-file levels, the merge vote and the merged levels; the
  build reproduces the census on 25,834 of 25,835 rows, the other being a relabelled `[M]-.` row
  that now counts its own channel), and the regression runs equal that replay: the uronium
  Orbitrap's merged Assigned 838 -> 836; the labelled-nitrate Orbitrap's Assigned 631 -> 646;
  the ~10k TOF's Assigned 517 -> 492 -- heavy C24-C40 readings at m/z 490-960 whose windows are saturated once C > 20 and O > 12 are
  counted; the six test-set rows, every roster winner, the cross-instrument agreement and the shift
  decoy unchanged or better. The wrong-adduct decoy rises (labelled nitrate 20.8 -> 23.9 %, uronium
  73.1 -> 74.9 %): every extra decoy row had been capped by competitors no pass can commit (F+Si
  mixtures; anion readings of a cation peak), so the old rate was lowered by the defect itself.

- **A commit outside the run's element budget is Candidate unless a curated list names it.** The
  per-peak grid filters every candidate through the run context (`ambient-air` keeps P, F and I
  at zero -- monoisotopic, never isotope-confirmable -- and S at one), but the passes that widen
  the search on evidence of their own do not: the certified multi-channel core opens P / S / Cl,
  the ladder gap-fill, the residual series / isotope pairs and completion extend what they are
  given, and a CF2 chain opens the fluorinated family on `max_F=0`. The evidence that proposed
  such a formula -- two channels converging on one neutral mass, a series step -- was then read
  back as the second channel, the acid branch or the anchor that the tier counts as
  confirmation, so a certified `C40H66N3O2PS3` stood at Assigned on its own proposal.
  The `plausibility` stage now holds every commit to the context's **element budget**
  (`contexts.element_budget`: the structural gate, the carbon-free allowlist and the heteroatom
  caps -- `filter_by_profile`'s first three steps, factored out, its answers unchanged on 9,434
  formulas x 28 contexts): outside it, and on no curated list (`passes.known_formulas`, the pass-0
  registry for the run's polarity and context, plus the active reference lists), the row is
  Assigned -> Candidate + `below_assignability` with a note and an audit row
  (`plausibility.demote_off_budget`). **A CF2 series keeps its fluorine**:
  19F is monoisotopic, so no satellite can pin it, but two committed rows on one adduct whose
  neutrals differ by exactly CF2 carry two more fluorines between them; a formula whose only
  violation is fluorine is kept when the ledger commits its CF2 neighbour (`budget_cf2_kept` in the
  stage summary) -- peak pairs one CF2 apart were 20x chance on the labelled-nitrate Orbitrap and
  4x on the ~10k TOF. The budget is not the context's
  minimum-carbon rule for a halogen (a reagent-alias guard: bromoacetic acid on a nitrate channel
  is no alias) nor its Van Krevelen windows (grid priors: nitrobenzoic acid is DBE/C 0.86). Sized
  on the same-air working set before coding (the C8 runs replayed: per-file levels, the merge
  vote, the merged levels): the ~10k TOF's Assigned 556 -> 518, and every Assigned phosphorus and
  multi-sulfur neutral goes (P 15 -> 0, S>=2 13 -> 0, F 15 -> 9, Cl 22 -> 13); the labelled-nitrate
  Orbitrap 633 -> 631 (its two unlisted fluorinated rows); the uronium Orbitrap 842 -> 840 (two
  chlorinated rows in positive mode); the six test-set rows, every roster compound and the
  cross-instrument agreement did not move.

- **The `--corroborate` cross set counts only what the other source pins on its own.**
  `--corroborate` backed a neutral in the merge vote's class whenever the other source's ledgers
  committed the same formula on ANY evidence -- so two instruments whose grids both fit a
  mass-degenerate formula backed each other on nothing but their agreement, and since that
  agreement puts a reading in the vote's top class it also decided merge winners. The cross set
  (`evidence.vote_cross_neutrals`, docs/EVIDENCE_LEVELS.md §13) is now the neutrals each source
  pins on its own: its per-file ledgers are pooled as one source and decided with NO cross set, so
  a source that was itself run with `--corroborate` cannot hand a run back the agreement it got
  from it; ion-only pairs never count. A merged-only source written before 0.10.0 is read by its
  stored private level without its own `corroborated` axis: a neutral counts where that level
  passes the same cut a run directory's pooled ledgers are held to and an axis of its own
  remains (`evidence._source_neutrals`, `_stored_own_good`). A merged-only source written on
  the 0.10.0 scale carries no per-file facts and is refused, and so is one carrying no level:
  name its run directory.
  Measured on the same-air pair before the change: the labelled-nitrate Orbitrap offered the TOF
  1521 neutrals, 379 of them pinned on its own; the TOF offered the Orbitrap 3728, 435 pinned.
  Replaying the vote on the three regression runs: 36 winners change on the TOF (every test-set
  row and roster compound keeps its reading), 12 on the nitrate Orbitrap (all between implausible
  grid formulas), none on the uronium channel. The three runs then matched the replay row for row
  (but for the post-align gates): 36 / 12 / 0 ions changed, merged Assigned 538 → 555 / 630 → 633
  / 840 = 840, and no scorecard acceptance metric worse on any channel. Named cost: a curated
  perfluoroheptanoic acid on a mass-saturated window, which neither instrument pins on its own,
  backs neither, and its TOF peak returns to the count's reading.

- **The merge vote reads the per-file evidence.** With the merge window finally putting one TOF ion's
  readings in one row, the vote's keys — files, Assigned-file count, `ion_score`, text — handed the
  peak to the reading fitted in the most files whatever the files' evidence for it: on a 28-file TOF
  batch the Orbitrap-confirmed `C9H16O6 [M+NO3]-` (Assigned, neutral backed, 2 files) lost to
  `C14H21N [M+Br]-` (unconfirmed, 9 files), the HOMs `C10H16O9 [M+NO3]-` and `C10H18O9 [M+NO3]-` to
  bromide adducts of N-compounds read in 3–4 files, and the roster's pinic acid `C9H14O4` a 1-vs-1
  tie on `ion_score` to an organosilicon formula — and the batch's board lost 14 cross-instrument
  agreements and two roster compounds to duplicate rows leaving. `assign_batch.align` now carries
  each file's reading's class (`vote_class`, `_M0_COLS`), and `_vote` ranks a cluster's ions by the
  best evidence CLASS of their readings before the file count: neutral backed (the neutral
  established, or held by the `--corroborate` source) > formula confirmed (the formula or the ion
  pinned) > unconfirmed (exact mass alone, or self-contradicting) — then files, Assigned count,
  score, text; the count decides among equals; the ion-only-last rule and the label stage are
  unchanged; frames without a class vote by the count exactly as before. A per-file class measures
  that file's evidence for the reading, the file count measures persistence, and a reading no file
  could establish does not become right by being fitted in more of them. The row says when the
  class overrode a count (`tier_reason`: `evidence outranks the count: kept … (neutral backed in 2
  of 12 files) over the 9-file … (unconfirmed)`) and `tables/jitter.csv` carries each file's
  `vote_class`. Measured on the three regression channels' per-file ledgers before coding (the
  pure merge replayed): on the TOF 160 of 885 multi-ion clusters change their winner — every one a
  lower-class many-file reading yielding to a higher-class one, none the other way; the four cases
  above come back, the other instrument's neutrals missing here fall 193 → 170 by the scorecard's
  own rule, and the pinic acid and `C10H16O9` rows are no longer misread — on the ¹⁵N-nitrate
  Orbitrap 56 of 268 (unconfirmed fluorinated and organosilicon grid formulas in 5–10 files
  yielding to better-backed 1–4-file readings), on the uronium Orbitrap 5 of 10 (the D7
  cyclosiloxane urea adduct, neutral backed in 2 files, now won by the vote itself rather than by
  the known-species lock alone).

- **The batch merge window is sized from the batch's own mass scatter.** `assign_batch.align`
  clustered the per-file anchors at the flat `sampling.BATCH_TOL_PPM` (6 ppm) whatever the
  instrument, while one ion's anchors scatter 3–4 ppm across the files of a TOF batch: measured on
  a 28-file TOF batch, the 6 ppm window minted two rows for one ion 135 times (adjacent merged rows
  closer than the stamping window), the trace stage collapsed 125 of them afterwards, and the vote
  in each split row never saw its rival — a 1-file `C14H17NO4S [M-H]-` kept a row of its own
  6.5 ppm from the 7-file `C10H16O6 [M+NO3]-`. A batch run now measures ONE `traces.MassScale`
  before its first merge (`traces.measure_mass_scale`: the per-file anchors, offset-corrected,
  walked onto their trace by mean shift, one centre per trace, the third-quartile per-trace scatter
  σ — the estimator the stamp already used, measured once for both) and sizes both windows from it
  by the one rule `traces.window_ppm` = `max(tol, min(2·tol, k·σ))`: the stamping window at
  k = 2.5 as before, the merge window at k = 2.5·√2 (`MERGE_GAP_SIGMA`), because the gap between
  two anchors is the difference of two draws. Both are floored at the binning tolerance and capped
  at twice it, so a run without a time series is the flat-window run exactly; an Orbitrap
  (0.2–0.3 ppm) stays at 6 ppm on both (measured on two Orbitrap channels: the merge is inert from
  3 to 9.5 ppm, no 6 ppm cluster holds two picked peaks of one file, and a tighter window only cuts
  one ion's per-file cloud in two — 54 extra rows at 0.5 ppm); a TOF (3.7 ppm) merges at 12 ppm,
  where no adjacent merged rows remain inside the stamping window and no cluster holds two picked
  peaks of one file (that starts at 15 ppm). The same window decides the known-species match
  (`lock_known_species`) and the trace-label collapse (`collapse_trace_labels`); selection,
  admission and the trace index still bin at `BATCH_TOL_PPM`, which stays the one binning
  tolerance. `batch_summary['mass_scale']` records σ, the trace count and both windows (`tol_ppm`
  stays the binning tolerance; `traces.stamp_tol_ppm` / `sigma_ppm` are the same numbers as
  before), `run_manifest` carries it under `output.counts.mass_scale`, and the log says
  `[scale] …`. `timeseries.stamp_tolerance` is unchanged for a caller with only a merged ledger.

- **The reagent-N isobar flag fires from both sides of the pair.** `tiers._reagent_n_isobar`
  flagged a winner only when it sat on an N-donating reagent adduct (`[M+NH4]+` / uronium) with a
  same-ion N-richer alternative; the protonated N-richer neutral, with its same-ion N-poorer alias on
  the donor adduct, reached Assigned as "unique formula in the calibrated window" once the alias was
  dropped (C5H12N2S `[M+H]+` Assigned in two files against C5H9NS `[M+NH4]+` in ten) — and the
  batch vote's label stage, which trusts an Assigned label as a corroborated one, is only as honest
  as the flag. It now returns `("donor" | "amine", alias formula, alias adduct)`; on the amine side
  isotopes do not count either, and the winner is capped at Candidate unless a second channel of its
  own or a series anchor fixes the nitrogen count (`reagent-N isobar unresolved: … is the same ion
  as C5H9NS [M+NH4]+ (the N-poorer neutral on an N-donating reagent adduct) …`; resolved: `nitrogen
  count fixed by a second ionization channel of the protonated neutral`). ROADMAP session-6 item 3.

- **A same-ion tie in the reference-list rescue is decided by chemistry, and recorded.** Two
  entries of one list can name the same ion under two of the run's adducts — an acid's reagent
  cluster and the deprotonated organonitrate one HNO3 heavier (C5H6O6 `[M+NO3]-` ≡ C5H7NO9
  `[M-H]-`) — and neither the mass nor the isotopes can separate them; the winner was whichever
  entry the formula frozenset iterated first, i.e. hash order, and three TOF peaks flipped reading
  between byte-identical runs. `reflists._target_table` is now fully ordered (mass, then a cluster
  channel before a bare one, then the run's channel order, formula and list text), `match_by_mass`
  treats targets within 1e-6 ppm as one ion and keeps the cluster reading (the list's native
  detection, the decomposition-alias policy of the tier engine), returning the others as `aliases`,
  and the rescue writes the alias into the row's `alternatives` and commentary ("Same ion as C5H7NO9
  [M-H]- (…): the reagent-cluster reading is kept …"). Same answer whatever the channel order.

### Fixed

- **A decoy arm is scored at what its file was scored at, not at a TOF's width.** Under
  the v2 fit a sample's width, offset and window come from its server record and its own
  server matches (`io_mascope.scoring_for_sample`); an offline sample has neither, so it is
  scored at the more forgiving class -- a TOF's -- at zero offset. The scorecard's decoy arms
  are offline samples, so an Orbitrap run's arms were judged at a TOF's width while the run
  was judged at its own fitted one. `register_offline_sample(..., scoring=)` (and
  `assign.run(peaks=, scoring=)`, refused without `peaks=`) now takes a measured sample's
  `pattern_scoring` snapshot, a `PatternScoring` or an instrument class ('orbi' / 'tof');
  `scoring_for_sample` judges the sample at it and the snapshot says `inherited`. A class
  other than those, a width, offset, window or floor that is not a finite number (a
  `PatternScoring` may leave its width or offset unset: the library default / zero; a
  snapshot may omit its floor), a width or window not above zero, an abundance floor
  outside [0, 1), or any other type is refused at registration. Each decoy arm inherits its
  file's snapshot from the run's batch summary (`scorecard.decoy_scoring`, by the same
  check; a run from before 0.9.0, or a snapshot that fails it, leaves the arm at the class
  fallback) and keeps the per-peak `signal_to_noise` of its per-file ledger (it was dropped,
  so an arm was judged in the no-SNR mode its file was not). The card prints per file what
  the arms were scored at, with the inherited width, offset, window, floor and anchors
  (`decoy.scoring` / `decoy.scoring_detail`, kept in the decoy manifest for a re-count), and
  the board row appends `decoy_scoring`; nothing renders it yet, so a decoy delta across a
  change of scoring -- on the board, the page's run panels or the card's delta section -- is
  not marked. What an arm still does not
  inherit -- the run's opportunistic channels, height cutoff, prior offset, occurrence table,
  time series, reference lists and corroboration -- is listed on the card and in
  docs/SCORECARD.md. The trace-first synthetic sample stays at
  the class fallback (scored at an Orbitrap's width it lost a third of a batch's
  assignments, mostly for want of a signal-to-noise). Re-registering an offline
  sample forgets the scoring computed for its old table.

- **A labelled adduct keeps its 15N when the ion is read from neutral + adduct.** The
  isotope line test reads every line against the ion's mono m/z, from the stored ion string
  when it is signed, else neutral + adduct -- and that fallback (`tiers._ion_counts`) dropped a
  labelled reagent's caret, so an unsigned `[M+^NO3]-` / `[M+^NH4]+` row read 14N: its mass sat
  0.997 Da low and the committed line of a multi-halogen ion was lost (a Cl4 `[M+^NO3]-`
  committed on its 37Cl1 line read as mono, its 37Cl2 line out of band); a pass-7 ladder rung
  the oracle wrote no string for stored `C5H3F6N2O6-` for a labelled nitrate rung. The
  reference script mirrored the drop. `tiers._ion_counts(..., labelled=True)` keeps the label
  as `^N`, as `parse_formula` reads a signed labelled ion (the tier gates' default reading is
  unchanged); `evidence.ion_composition`, the rung's ion and the script read it so -- and with
  `evidence.ion_composition` the batch isotope checks' ion counts (`batch/iso_checks.ion_counts`):
  HIGH's expected M+2 on such a row now counts the ion's N without the labelled atom (its
  13C x 15N term), physically right, as the label is already 15N. Latent on
  the stored data: no M0 row on a caret adduct stores an unsigned ion string in the three
  regression runs or the fixtures, and every labelled rung of the replays carries the oracle's
  string -- the full offline replay of the three batches reproduces the replay before it in every
  per-file ledger column but the evidence columns the full-count rule moves (18 rows), and every
  batch table and time-series stamp. `tests/test_isotope_levels.py`,
  `tests/test_isotope_children.py`, `tests/test_known_ion_kids.py`.

- **A known or certified ion takes its own isotope lines.** The scorer returns
  the lines of every ion of a compound; pass 0 (known species) and pass 7 (certified neutral)
  hung all of them under whichever ion they committed -- HNO4's nitrate cluster carried the
  81Br line of HNO4.Br- 18.93 Da up, the PFCAs' nitrate clusters their bromide clusters'
  lines, TPPO's urea adduct its `[M+H]+` ion's 13C 59 Da below, chloroacetic acid's bromide
  cluster the 37Cl lines of its nitrate cluster 14.93 Da below. The lines are now keyed on
  (compound, ion): pass 0's attach loop, recorded isotopologues list, chlorinated-paraffin
  gate and confidence count the ion's own lines (the single-channel P / S / Si gates still read
  the compound), and each pass-7 member takes its committed ion's. A pass-7 ladder rung is
  committed with its real ion formula (the oracle's own string for its channel, else neutral +
  reagent units + adduct) where it stored the bare neutral, and takes its own lines. One
  commit path still stores an unsigned ion string: a re-arbitration commit
  (`rearb<-cheminfo+grid`: one M0 row of the 28,092 in the three batches' final replays, on the
  bromide/nitrate TOF); the levels
  read its composition from neutral + adduct, so no level or fact depends on it, and the
  reference script, which prints the composed string, shows another `ion` for that pair.
  The certificate's diagnostic-isotope gate (a 34S / 37Cl / 81Br line earns Good (certified) and
  the displacement strength) now reads the lines of the ions the certificate COMMITS -- each
  member's: an anchored member's oracle string, a rung's real ion (the user, 2026-10-01: "key it
  on the ion") -- not every line the scorer found for the compound. Measured before the decision
  on the three batches' offline replays, with the reagent's line counted on two channels as then
  built: 0 / 0 / 11 of 379 / 43 / 835 gate decisions flip, all two-channel `[M-H]-` + `[M+NO3]-`
  certificates of Br-free winners whose only line sat under their `[M+Br]-` ion, which they do
  not commit (the doublet already held by another reading): 8 Good -> Low, 3 tied ones lose the
  displacement strength (under the rule below those 11 read Low either way). A line whose only
  diagnostic isotope is the reagent halogen's heavy one (`81Br`, `13C+81Br`, a two-Br ion's
  `81Br2`; a 13C, 18O, 15N or 33S beside it adds none) under a committed ion of a winner that
  carries no Br counts only on a certificate of >= 3 channels (`cert.n_channels`), where two
  other ions confirm the mass: on two -- one file, two ions and the reagent's own line -- it is
  not enough, and such a certificate reads Low (certified) with no displacement
  strength (this supersedes an earlier rule that counted the line on two channels as on
  three). Lines of the winner's own elements (34S, 37Cl,
  alone or combined -- `13C+37Cl`, `81Br+37Cl` --; the 81Br of a winner that carries Br, which
  pass 7's P / S / Cl box never certifies on the
  pipeline's own runs) count on two channels as on three. The label is not without consequence,
  as the 2026-10-01 entry had it ("the merged ledger's vote decides"): a pair seen in one file
  has no vote, so that file's label sets the pair's merged tier. Measured on the
  full offline replay at the final engine against the same replay with the line counted on two
  channels: the two Orbitrap batches unchanged (every per-file A.run ledger, table and stamp
  byte-identical; neither has a bromide channel); on the TOF 65 of the 835 gate decisions flip
  (45 of them past the time-series check: 39 untied, 6 tied), every one a two-channel certificate
  of a Br-free winner on the reagent's line alone -- 49 Good
  -> Low (certified), 16 tied ones (Low either way) lose the displacement strength; 120 per-file
  rows in 22 of the 28 files change (34 certificate members Good -> Low (certified) and two
  ladder rows Good -> Low (ladder) on the same reading, 75 rows re-read or changing role, five
  isotope lines relabelled, four M0 rows only in their evidence columns); merged rows 2242 ->
  2241, Assigned 454 -> 450 -- C16H19NO4S `[M+Br]-` (m/z 400.022) Assigned -> Candidate and its
  `[M-H]-` pair gone (its one file's certificate no longer displaces the weak C12H23N3O2
  `[M+Br]-` series reading of m/z 320.097, which the end mass gate then clears), C25H36N2O8S
  `[M+NO3]-` Assigned -> Candidate (its `[M+Br]-` peak at m/z 603.134 now read as C23H28N2O17
  `[M-H]-`), 8 merged pairs gone (the two named above -- C16H19NO4S `[M-H]-`, C25H36N2O8S
  `[M+Br]-` -- and 6 dim ones) and 7 new, all dim; 1198 time-series stamp rows change their
  neutral or adduct (0.10 % of the signal; 2226, 0.25 %, by the landing count of the
  isotope-child entry under Changed, which also counts a changed role, ion formula, isotope
  label, stamp source or duplicate flag) and 3513 keep theirs at a shifted consensus m/z.
  Chloroacetic acid is untouched: m/z 172.901 (`[M+Br]-`) and 155.970 (`[M+NO3]-`) stay
  Assigned, read so in 13 of 17 and 14 of 15 files. On the TOF 68 of the 89 passing gate
  decisions rest on a reagent line alone, all on >= 3 channels: 54 get past the time-series check
  (50 at Good (certified), 20 winners) and 14 are skipped as anti-correlated in time (the check
  vetoes 21 % of these passes, 1 of the other 21, 5 %). One of them reads a dim bromide-reagent
  triplet at m/z 422.83 / 424.83 / 426.83, `reagent` at trunk, as a C10H4N2O5S `[M+HBr+Br]-`
  ladder rung at Good (certified), a Candidate in the batch. Pass 7 does not revisit the label:
  in 21 of the 74 committing certificates every member whose ion carried the counted line is
  later cleared or re-read (19 by the isotope audit's carbon count -- that triplet's `[M+Br]-`
  member among them --, 1 by the end mass gate, 1 by the ladder gap-fill), and four members of
  three of them keep Good (certified); three of the four (the anchored
  members) carry "diagnostic isotope envelope confirmed", the fourth, that ladder rung, does not
  (a rung's commentary never does). "Commits" means the certificate's intended members (each
  member's ion is fixed before the commit loop): every passing decision that gets past the
  time-series check commits all of them (74 of 89; the other 15 commit nothing). The pair facts
  keep the line apart from the neutral's own isotope support (`reagent_only_iso`, `chan2`). Pass 0's
  single-channel gates stay on the compound. `tests/test_known_ion_kids.py` (its pass-7 spectra
  built from the scorer's own envelope and scored by the local scorer: the reagent's line on two
  channels and on three, a '13C+81Br' line alone, a winner wanting no envelope, a two-Br ion's
  `81Br` / `81Br2` lines, the winner's own 34S / 37Cl -- alone or as `13C+37Cl` / `81Br+37Cl` --
  and a Br winner's 81Br (on its two-Br bromide cluster too) on two channels, a 34S or 37Cl line
  under an uncommitted ion, a tied certificate on two and on three channels and a tied one of a
  winner wanting no envelope, an order-1 rung's own line; a two-Br ion based on its all-light
  line, as the server bases it).
  ATTRIBUTION on the offline replays (each fix against the replay without it, before option B;
  the landing numbers are the isotope-child entry's under Changed): per-file ledger rows whose
  role, reading, parent, label, list or confidence change (attributed exactly by a replay without
  the residual fix below): the labelled-nitrate Orbitrap 23 (pass 0 18, pass 7 1, and 4 knock-on
  rows: two ions committed later in two files, with a line each), the uronium Orbitrap 30 (all
  pass 0: TPPO's urea adduct and D3's lines), the bromide/nitrate TOF 723 (pass 0 620 -- the
  PFCAs, HNO3 / HNO4 / HO2 and nitrophenol lose their bromide clusters' lines, 28 dim pass-0 rows
  turn Low, one file's calibration core falls under 20 --, pass 7 59, and 44 from the rungs' real
  ions: 32 rungs carry their ion instead of the bare neutral, 12 rows are later commits and lines
  that follow); the ladder rung C7H6N2OP2 `[M+Br]-` keeps its own 81Br line and its commit. The
  gate keyed on the committed ions, against the same replay with the compound-keyed gate: the two
  Orbitrap batches unchanged (every file, table and stamp byte-identical); the TOF 8 of 28 files,
  428 rows -- 8 certificate members Good -> Low (certified); 8 rows re-read, five members and
  three lines that follow them (a Low certificate's members are weak commits that later stages
  re-read: one C8H15ClN3PS3 `[M+NO3]-` member is now a C10H10BrN3O4 reading, one C5H15ClN2P2S2
  `[M-H]-` member a C5H12O7 `[M+Br]-` one, the others an isotope line or unexplained); 9 rows of
  tetrafluoropropionic acid C3H2F4O2 (five M0 rows and four of their lines) committed by pass 3's
  late fluorinated sweep in two files; 12 rows whose other columns alone change -- the commentary
  of six unexplained peaks (members the isotope audit clears anyway) and of the tied C9H9ClP2
  certificate's two members (Low (certified) before and after, now without "diagnostic isotope
  envelope confirmed"), the isotope match score of two lines, the evidence columns of two members
  --; and 391 rows whose displayed `ppm_error_cal` alone shifts by 0.01 ppm in one of those files
  (the offset re-fits on the changed commits; no decision reads the column); per file
  C6H9ClO3S `[M+NO3]-` takes a Low engine confidence in one file; pooled, one dim pair
  appearing (C10H10BrN3O4 `[M+NO3]-`) and three disappearing (C5H15ClN2P2S2 `[M-H]-`, C8H15ClN3PS3 `[M-H]-` / `[M+NO3]-`), 0 merged tier moves;
  1556 time-series stamp rows (0.28 % of the signal) change their stamped ion or its m/z. Pooled:
  see the entry above.

- **The Br doublet spacing of the residual pass is the exact 81Br - 79Br difference.**
  `residual.D_PAIR_BR` was 1.997795, 0.16 mDa below the exact 1.9979535 its +-8 ppm pair
  finder and +-6 ppm residual characterisation centre on; now exact, with a value pin. Only dim
  doublets at the window's rim appear or disappear (per file); the residual tags shift from
  'iso-partner 81Br' toward '37Cl' where a partner sat between the two spacings.
  `tests/test_residual.py`. ATTRIBUTION on the offline replays (this fix against the replay
  without it, before option B; the landing numbers are the isotope-child entry's under Changed):
  per-file rows change in pass 4's dim edge doublets and what re-reads the freed peaks (the
  labelled-nitrate Orbitrap 9, the bromide/nitrate TOF 36; the uronium batch's context caps every
  doublet, so nothing moves there); pooled, the labelled-nitrate Orbitrap 3 dim pairs appear
  or disappear, the TOF 11 (4 merged): ten dim pairs appearing or disappearing -- the ten the
  engine lane measured for this fix alone -- and C15H12N2O4S `[M+Br]-`, the one interaction
  with the line test above: the pair's new row (pass 4, one more file) carries an `81Br(pair)`
  child at 0.78x, which the earlier label reading could not read (expected 0) and the
  whole-label reading puts in band (0.9728 expected).

- **The isotope checks measure a batch whose committed pairs are all carbon-free.**
  Rule C (`batch/iso_checks.py`) raised `KeyError: 'mz'` when no stamped pair
  carried carbon, which stopped the batch run at its evidence stage; it now tests nothing
  there, and REQ, HIGH and rule H run as on any batch. Real batches always carry carbon
  pairs; rule H gives a carbon-free pair (HBr) a line to read.
  `tests/test_halogen_lock_check.py`.

- **The two isotopologues of a bromide water cluster stamped as one line between them.**
  `reagents.build_library` labelled `[Br1+1xH2O]-` (and every halide-core cluster) without
  the core's isotopologue tag, so its 79Br and 81Br lines shared (ion formula, no tag) in
  the batch stamp and were stamped once at their median m/z (97.93 for Br-.H2O), which
  matches neither peak: the TOF's largest water family (8 % of its signal) stayed
  unstamped. The label now carries the tag like the bare core's (`[Br1+1xH2O]- (81Br)`),
  and `timeseries.identified_rows` already reads it. `tests/test_reagent_water.py`.

- **The two O-rich rules read the window the way their reasons say.** `below_assignability`
  for O >= 11 (`tiers.flag_below_assignability`, reason "O>=11, mass-saturated") fired on any
  degenerate window (>= 3 ions), and the O-monster demote's second leg (`plausibility`) matched
  any `MASS-DEGENERATE` note -- two ions included. Once the degeneracy audit counts the commit
  itself, a small high-O/C acid with one competitor reads `MASS-DEGENERATE: 2`: malonic acid,
  C2H4O4 and C3H6O4 on the labelled-nitrate Orbitrap (`[M+^NO3]-`, beside a fluorinated `[M-H]-`)
  fell to Candidate + below assignability, and malonic acid on a TOF -- two channels in
  17-23 files -- stood below assignability because two of its files read density 2. Now the O >= 11 flag reads the audit's `MASS-SATURATED` flag only
  (`tiers._saturated`: more than `degeneracy.SATURATION_DENSITY` plausible formulas, a lower bound
  included), and the O-monster leg reads the tier engine's own "degenerate" window
  (`plausibility._mass_degenerate` = `tiers._degeneracy`: >= 3 plausible ions or saturated) -- a
  unique or two-ion window spares the small polyacids, a crowded one still demotes an O-rich fit
  whatever corroborates it. Reading "saturated" in the O-monster leg too was sized and set aside:
  it kept the same small-acid fixes but also released five O/C >= 1.4 formulas in 5-24-ion windows
  to Assigned (C6H14O9 `[M+Br]-` on 22 TOF files, C5H10ClNO7 `[M+NO3]-` on 10 Orbitrap files).
  Sized on the same-air working set before the change (the census replay of the audit's
  consumers) and the regression runs equal it row for row: labelled nitrate Assigned 646 -> 647,
  ~10k TOF 492 -> 493, uronium unchanged; the test set, every roster winner, the
  cross-instrument metrics and the shift decoy unchanged. The wrong-adduct decoy on the labelled-nitrate channel admits two O-rich
  `[M+Cl]-` fits in two-ion windows (110 -> 112 rows, 23.9 -> 24.3 %): sparing a two-ion window is
  the small-acid fix itself.

- **A batch-level re-read left its time-series stamp without an ion.** The stamping frame borrows
  each merged analyte's `ion_formula` from the per-file ledgers by its (neutral, adduct) key; a
  reading no per-file ledger holds — the reagent-N re-read on the merged frame (C8H14
  `[M+(CH4N2O)H]+` → C9H18N2O `[M+H]+`, the same ion), the amine re-read, now the known-species
  lock — found no key, stamped its peaks with a neutral and no ion, and everything that reads
  `ion_formula` as "identified" (the residual universe, the scorecard's stamp coverage, the
  predicted satellites) took the track for unexplained: every one of the 7 reagent-N re-read rows
  of a 10-file uronium batch, in all 319 spectra, and none of the other 1141. `stamping_frame` now
  derives the ion from the reading itself (`publish.ion_formula_for`) wherever the modal lookup
  finds nothing; a per-file ion still wins where one exists.

- **The tier engine's persistence read was not null-safe** (`tiers.py`, `persist_only`).
  `str(r.get("admitted_by") or "")` gave `"nan"` on a float-NaN cell (NaN is truthy) — False by
  accident on real runs, where the admission gate leaves 4 % of M0 rows unstamped — and raised
  `boolean value of NA is ambiguous` on a `pd.NA` cell, which is why the scorecard's wrong-adducts
  decoy arm had never completed on a negative channel. Now an explicit string test; result-identical
  on real runs (0 of 1314 / 1148 / 2645 ions moved on the three reference channels).

- `scripts/scorecard.py`: the decoy arms are rated by the engine's own `evidence_level` when the
  engine wrote it, so an arm is levelled by the same leveller as the run it bounds; an older
  engine's ledger is still levelled post hoc.

- **A mixed inlet got no calibrants at all.** `profiles.compose` names a
  two-reagent module `Br+NO3`; `reference_ions.get` matched only single-reagent aliases,
  raised, and the trace builder read that as "no reference list", skipping mass-qc and the
  wave entirely — so the mixed-reagent TOF, the case the wave exists for, was the one that
  silently got no mass-axis measurement. `get` now unions the parts: 53 ions and 20 anchors
  on `Br+NO3` against 30 and 11 from nitrate alone, the twelve 79/81Br twin pairs intact,
  an ion certain in both kept once at its stronger grading.

- **The mass wave was fitted in the wrong variable on an Orbitrap.** The mass-qc
  call inside the trace builder passed `tof=True` as a literal, so a trace-first Orbitrap
  run fitted its wave against flight time for an analyser that disperses in frequency.
  `Resolution.is_tof` now reads the basis off the measured width exponent.

- **A wave fit could stand on calibrants it had already discarded.** The clip loop
  recorded each fit against the mask the NEXT clip proposed, so the `min_n` guard the
  docstring promised never held: a real batch returned a degree-2 wave standing on 4
  surviving calibrants out of 7. And `share` divided the survivors' residual by EVERY
  calibrant's spread, two different sets, so any clip flattered it and an outlier in the
  denominator could make noise read as an explained wave — a degree-ZERO fit reported
  "explains 85 %". Both sides now come from the set the fit was judged on.

- **The trace fill gate had no power to do its job.** `ks_uniform` compared a
  trace's members against a uniform ON THEIR OWN RANGE, which divides out the width — the
  only thing separating an ion from a fill. The statistic stopped depending on sigma at all
  (a 1.35 ppm ion and a true fill both read ~0.21-0.25), so `KS * sqrt(n)` was a disguised
  member count: a real ion was called a fill 100 % of the time at 10 members and 94 % at 40,
  silently making the rule "keep an ion only if it occurs in ~37 % of spectra". It also read
  "cannot reject uniform" as "is a fill". The test is now the width, and on that TOF batch
  it takes the positions the trace layer offers from 65 % to 75 % of the cover's ions.

- **The formula search applied its ppm tolerance to the neutral mass, not the ion.**
  `chemistry.candidates_for_peaks` sized its window as `neutral_mass x search_ppm`,
  but every caller passes ion m/z and means the ion's window (the scorer, the z-gate
  and the commit tolerance are all on the ion). On a cluster adduct the neutral is
  lighter than the ion, so the enumeration net was narrower than the search tolerance
  by neutral/ion: 1.8x for H2SO4 . Br- (`search_ppm=12` reached 6.6 ppm on the ion),
  1.6x for H2SO4 . NO3-, 2.3x for acetic acid . Br-; on [M-H]- / [M+H]+ the ratio is ~1
  and nothing changes. Candidates the scorer would have accepted were never
  enumerated, and the peak sat unexplained. The window is now `ion_mz x search_ppm`
  for every adduct, and `tests/test_chemistry.py` probes a neutral at +9.5 / +10.5 ppm
  on the ion through four adducts. A/B measurements are in the pull request.

- **The heteroatom isotopologue gate charged peaks that could never have shown their
  satellite.** `passes.core`'s `_evidence_penalty` subtracts a gate penalty (0.30
  halogen / 0.12 S / 0.12 Si) from `eff_score` whenever a candidate's diagnostic
  satellite — 37Cl, 81Br, 34S, 29Si/30Si — is unconfirmed, and it had no
  observability test: a peak too dim for its satellite to clear the height gate was
  charged exactly like a bright peak whose satellite is genuinely missing. That turns
  "we could not have looked" into evidence against the formula, and because the
  penalty is subtracted before arbitration it hands the peak to a rival. The gate is
  now waived when the predicted satellite height (per-atom abundance × atom count ×
  parent height) falls below `cfg.height_cutoff`; the plain complexity prior still
  applies, because unobservable is not confirmation either — the same reading the
  reference-list rescue already took for a dim 13C ("tentative lead, not confirmed").
  That rescue now asks the shared predicate (`isotopes.satellite_observable`) instead
  of its own inline `0.011 × nC × height`: 13C is 1.07 % per carbon there, 2.7 % lower,
  so a rescue sitting exactly on the floor now lands tentative, and a missing floor
  reads as observable instead of raising.
  The reagent-element branch is deliberately NOT waived: it is about neutral-vs-ion
  ownership of a halogen the reagent also supplies, and the ion's twin is as bright as
  the ion, so brightness never made that question answerable. Measured on two
  trace-first ledgers of a 1057-spectrum TOF sample (nitrate and bromide channels):
  the gate was being charged against an unobservable satellite for 46 % / 34 % of the
  S-bearing winners and 57 % / 42 % of the Si-bearing ones — and for none of the Cl
  and almost none of the Br, whose twins are 32 % and 97 % of the parent and so are
  visible wherever the parent is. 10 % / 8 % of all M0 winners changed effective
  score (by up to 0.12).

- **The Si tier demote stated a refutation that could not have happened.** The rule in
  `tiers.compute_tiers` demotes an uncorroborated silicon formula because "silicon has
  a strong M+1/M+2 twin that must appear if real" — true only when the twin was within
  reach. The tier is unchanged (an uncorroborated formula has no evidence for its
  silicon either way and must not read as Assigned), but on a peak whose twin is
  predicted below the detection floor the reason now says the claim was untestable
  rather than refuted: 39 rows across the two ledgers above. The mono-isotopic P/I and
  partially-fluorinated rules need no such guard — those elements have no minor isotope
  at all, so no brightness could ever have made their count testable.

## [0.9.0] — 2026-09-30 (the v2 fit at the sample's own width, the standard adduct notation, the abstraction and solvent-cluster channels, the privacy scan)

### Changed

- **The local scorer judges a candidate at the sample's own measurement, not at a
  fixed Orbitrap's.** It scored with `score_pattern`, which scaled its mass term
  by a flat 5 ppm and averaged its terms over the lines a candidate *matched* - so
  an envelope that predicted three lines and found one cost nothing, and on a TOF,
  whose ordinary mass error is a whole Orbitrap window, every candidate scored near
  zero and a run committed almost nothing. It now scores with Mascope's
  `score_pattern_v2` and a `PatternScoring` built per sample
  (`io_mascope.scoring_for_sample`): the width and offset fitted from the sample's
  own targeted matches - the library's `fit_mass_accuracy`, falling back to the
  instrument class below eight anchors - and the line-matching window of that class
  (5 ppm Orbitrap, 15 TOF). The peaks frame carries each peak's `signal_to_noise`
  into the score, so a predicted line that is missing is charged where the noise
  says it should have been visible and excused where it says it could not.

  This is the same function, the same fit and the same widths Mascope's own
  assignment engine uses, so a peaky run and an in-app run of one sample can be
  compared as two sets of assignments rather than as two scorers.

  **The library it is built on is the one Mascope 1.10 publishes.** The v2
  scorer, the fit and the widths entered `mascope-tools` after 2026.6.25 and
  reached PyPI with 2026.9.30, which is now the dependency's floor; until that
  release this work lived on a branch pinned to the library by revision, the one
  the reference runs were scored with. The per-peak `signal_to_noise` the score
  reads is sent by a server of the same release; against an older server the
  score runs in its no-SNR mode and charges an absent line on predicted
  abundance alone.

- **A published run says what scored it.** Its config carries `score_version` and
  a `pattern_scoring` block - width, offset, whether the width was fitted or the
  instrument class's, the anchors behind it, the matching window, the envelope
  floor - in the same keys Mascope's engine stamps on its own runs. Two runs of
  one sample disagreeing is a different fact when they were judged at different
  widths, and neither said so before.

- **The predicted envelope is anchored on the ion's own line.** IsoSpec returns
  the most abundant configuration first, which for a dibromide is the mixed 79/81
  line two mass units above the peak the candidate was proposed for - and every
  index-0 reading downstream (the intensity the envelope is normalised to, the
  score's anchor, `is_base`) then described the wrong line. `anchor_on_monoisotopic`
  puts M0 first, which is also how the network path derives `is_base`, so the two
  backends now agree about which row is the ion's.

- **A run's manifest records the `mascope-tools` version.** The library is the
  scorer - the fit, the isotope prediction and the class widths all come from it -
  so a run's numbers are not reproducible without naming it.

### Added
- `PEAKY_MATCH_PPM` widened the local scorer's line-matching window per run, so a TOF
  reference run (5-15 ppm accuracy) was not emptied by the Orbitrap-sized 5 ppm default.
  **Removed again in the same release**: the window is now the instrument class's
  (`resolve_match_tolerance_ppm`), which is what an operator was saying by setting it.

- **A source-solvent cluster channel for positive-mode sources, kept off the covalent
  grid on purpose.** A low-pressure positive source running on solvent vapour does not
  only protonate single molecules — it builds proton- and hydride-bound CLUSTERS of the
  solvent, and on one EasyIC⁺ acquisition the ethanol dimer `(C2H5OH)2H+` at m/z 93.0911
  was **6.5 % of total signal and about half of everything the run left unexplained**,
  with its ¹³C satellite and the mixed `EtOH·H3O+` (65.0597) unexplained beside it. Peaky
  could not reach them, and must not be taught to: `[(C2H6O)2+H]+` implies the neutral
  C4H12O2, whose DBE is −1, and the grid's chemistry gate is right to refuse it. So the
  clusters get their own enumerated family — `assignment/solvent_clusters.py`, committed
  and locked in pass 0's slot, offline (no scorer is asked; none could be). It enumerates
  the `[Sₙ+H]⁺` and `[Sₙ−H]⁺` ladders and their `−H₂O` condensation rung over the source
  solvents a context declares (`ContextProfile.source_solvents`: water / methanol /
  ethanol / acetone for `easyic`, `uronium` and `ammonium-15n`; the hydride ladder only
  where the context actually has an `[M-H]+` channel). A rung commits only with its
  solvent's **own monomer ion present and ≥0.2 % of the spectrum**, the **rung below
  observed** at exact solvent-mass spacing, and the pass-0 mass gate; a rung that would
  displace a legal covalent reading must additionally be a major ion — and may never
  be a **water** rung at any intensity: every other unit is anchored by its own
  monomer ion and steps by a rare mass difference, while water is anchored by nothing
  and +18.0106 is the commonest difference in a spectrum (26 % of peaks on an
  orange-peel headspace run had such a partner, against 7.7 % for +C2H6O). A rung that
  does displace is tiered **Candidate** with the covalent reading stored in
  `alternatives`. Which readings
  compete is the SOURCE's property and it grows with the profile: `C4H11O2+` is
  protonated butanediol, and `C4H9O+` is protonated C4H8O, hydride-abstracted C4H10O
  *and* — since EasyIC gained `[M-CH3]+` — the methyl loss of pentanol, so the
  alternatives are worked out against the context's own `reagent_adducts` rather
  than a fixed pair. The neutral reported is the SOLVENT on a cluster adduct
  (`C2H6O [M+(C2H6O)H]+`), so the ¹³C satellite is picked up by the ordinary envelope
  sweep. On the acquisition above it explains 10.1 % of total signal and re-reads three
  ions that had been committed as covalent molecules at High confidence. Swept over 232
  other EasyIC and urea per-file ledgers it commits 9 rungs in total; on every urea
  batch measured, and on the orange-peel headspace batches, it commits nothing, because
  those windows start above ethanol's monomer ions and the family refuses a ladder whose
  monomer it cannot see.

- **A cluster-vs-covalent dual note in the EasyIC ambiguity layer.** A proton/hydride-bound
  cluster and its covalent isomer are the same ion with the same isotope pattern, so MS1
  cannot split them — exactly the carbonyl-vs-alcohol situation one ionisation step
  further out. `cleanup.annotate_easyic_ambiguity` gains a fourth case for it, built from
  the same library the commit layer uses so the two cannot describe the chemistry
  differently: an `C4H11O2+` or `C4H9O+` commit in a spectrum carrying a strong solvent
  monomer keeps its covalent reading and records the cluster reading beside it. Unlike the
  other cases it does not skip locked rows — a lock says no pass may *change* a reading,
  and the pass that locked a High grid winner never weighed a cluster reading at all.

- **The EasyIC source-ion library, and the reason its background was being read as
  sample.** On the certified-cylinder run the calibrant's own PAH ladder (C13H8,
  C14H10, C14H12, C15H8, C15H10, C15H12) and most of an air-plasma C/N/O family sat
  in the ledger as **Assigned analytes** in a cylinder that contains none of them.
  The first guess — that the library was simply short of entries — was wrong twice
  over, and the run says so:
  - **The time-series layer already classifies flat background, and it was getting
    the answer backwards.** Those bins are flat to cv_norm 0.06-0.08 against
    1.1-3.8 for every certified analyte, yet the batch run labelled them
    `ambient:variable` at cv_norm 0.53. Cause: the only EasyIC library ions inside
    the 50-200 window were the urea CROSSOVER masses from the other source module —
    0.013 % of TIC with an own cv of 0.93 — and dividing by a trace that moves does
    not remove a common-mode swing, it injects one. **A normaliser now has to clear
    the same bar it is used to judge** (`MAX_NORMALISER_CV` = `FLAT_CV`): if it
    moves more than a flat channel it is rejected, with its own cv reported, and the
    traces stay un-normalised. A real reagent beam clears it easily — fluoranthene
    holds ±5 % through a 12× load swing. With the guard in place those six bins read
    `background:flat` and the certified components keep `ambient:variable`.
  - **The flat-background demote is no longer specific to two channels.** It fired
    only for di-bromide and CO3 commits, so everything else stayed `Assigned` —
    which is where a reader looks. Any flat `Assigned` commit is now capped at
    `Candidate` with the cv in its `tier_reason`. It is armed only when the run
    itself varies (`MIN_VARYING_FRAC`): a steady-state batch is flat everywhere,
    and there flatness distinguishes nothing, so nothing is demoted. On the
    dilution series 54 % of bins vary, and the separation between the two
    populations is 9-65×, nowhere near the threshold.
  - **What is NOT in the library, deliberately.** Naming those PAH and C/N/O
    compositions as reagent ions was the obvious fix and is the wrong one: a
    reagent label is permanent, and anthracene/phenanthrene C14H10 is a primary
    target of any combustion or urban-air run. That is the HIO3 ruling — labelling
    the iodine oxides reagent would have locked away iodic acid. Behaviour
    identifies them, so behaviour tiers them, and in a run where anthracene really
    varies it keeps its tier.
  - Added instead are the ions that can **never** be an analyte: the calibrant's
    acetylene-loss fragments C14H8 / C12H6 (a fragment cannot gain hydrogen, which
    is what separates them from the H-richer ladder above; C14H8 is present and
    flat at cv_norm 0.13 and is now labelled `reagent` rather than committed),
    carbon-free N2+·/N4+·, and hydronium with its water-cluster series — in range
    now that acquisitions start at m/z 17.
  - Net on the 50-200 sample: `Assigned` 140 -> 96, nitrogen-bearing `Assigned`
    16 -> 8, rows carrying a background disposition 9 -> 87, and every certified
    component still `Assigned`. This also absorbs most of the nitrogen-phantom cost
    the abstraction channels added below.

- **The positive-mode abstraction channels `[M-H]+` and `[M-CH3]+` are reachable,
  and `[M-CH3]+` exists at all.** Audited against a certified 18-component
  calibration cylinder, EasyIC's charge-transfer channel was exact (8 of 8
  compounds, |ppm| <= 0.5) while every one of its other channels named a molecule
  that is not in the bottle. Cause: the mechanism plumbing is keyed on deployment
  mechanism ids, and no deployment registers one for a positive-mode abstraction —
  so `ADDUCT_TO_MECH` filtered `[M-H]+` out before the scorer saw it, even though
  the local scoring backend has computed `-H+` all along. The channel was
  *unreachable*, not unscorable. An abstraction ion is also mass- and
  envelope-identical to the protonated form of the neutral one H2 (or one CH4)
  lighter, so with only the `[M+H]+` reading offered, that reading won every time:
  acetone was read as propenal, isoprene as cyclopentadiene, hexanal as C6H10O, and
  the two brightest peaks of the window — the benzyl and methylbenzyl cations — as
  protonated C7H6 and C8H8.
  - Abstraction channels now travel inside `cfg.mechanism_ids` as tagged tokens
    (`io_mascope.LOCAL_MECH_PREFIX`), translated to a local mechanism name in the
    one place ids reach the scorer and stripped in the two places they reach the
    server. Fourteen call sites thread `mechanism_ids` already; a second parameter
    would have had to be added to each, and the one that got missed would lose the
    channel again in silence. They stay out of `ADDUCT_TO_MECH` on purpose: the
    server spells negative deprotonation `-H+` too, so keying `[M-H]+` there would
    make `MECH_TO_ADDUCT` ambiguous.
  - `[M-CH3]+` is new — in `ADDUCT_SHIFTS`, in `_DIFF_TO_ADDUCT` (a channel missing
    there is silently written to the ledger as a deprotonation), and in the EasyIC
    profile and context. It is the cyclic methylsiloxanes' quantifier channel: D4
    281.0511 and D5 355.0699 are 27 kcps and 107 kcps on the certified run, the
    brightest peak of the 210-500 window among them, against `[M+H]+` lines
    carrying 0.5 % of the same compound. Si deliberately stays out of the
    enumeration grid — the siloxanes reach the channel through the pass-0
    `cyclosiloxane` known list, where the commit is already gated on >= 2 channels
    or a confirmed 29Si/30Si envelope.
  - Both channels are MINOR channels, and are never enumerated from. The alias is
    exact, so a good score cannot choose between the readings: on the certified run
    isoprene's `[M-H]+` scores 0.900 against protonated cyclopentadiene's 0.902 on
    the same peak, and hexanal's 0.982 against C6H10O's 0.982. Three consequences,
    each measured on that run rather than argued:
    - Un-penalised, the winner is whatever sorts first. Toluene's protonated line
      93.0699 flipped from the correct `C7H8 [M+H]+` to `C7H10 [M-H]+`. The
      minor-channel ranking penalty sends an exact tie to the protonation reading
      and the flip is gone; a genuinely better abstraction fit can still win by more
      than the penalty.
    - Enumerating candidates FROM these channels turned 59 honestly-unexplained
      peaks into commits, 40 of them nitrogen-bearing neutrals in a
      nitrogen-free cylinder, and gained nothing. The channels are now scored for
      neutrals the run already has (pass 0's known species, pass 5's cross-channel
      search, a reference-list rescue) but never proposed from — the `[M+Br3]-`
      ruling.
    - The minor-channel commit gate's score and series exemptions do not apply to
      them, so cross-channel standing for the neutral is the only way in. A high
      score says the ion composition is right, not which molecule shed which
      radical.
  - Channel COUNT was tried as a tie-break and rejected on measurement: in a
    fragmenting source the spectrum has a peak at nearly every (nominal mass,
    defect) a small CHO formula needs, so C5H6 matches all four channels exactly
    like C5H8 and C6H10O like C6H12O. The count separates nothing and a prior keyed
    on it fires on both sides of every tie.
  - Net on the certified mixture, same two samples, same peaks: 9 of 15 components
    named correctly before, 11 of 15 after (D4 and D5 gained), no regression on any
    quantifier or secondary channel, and unexplained peaks down 291 -> 247 and
    108 -> 59. D4 and D5 also reclaim their own ²⁹Si/³⁰Si satellites, evicting five
    sodium-adduct commits that had been sitting on them, two of them `Assigned`.
    The three hydride components (acetone, isoprene, hexanal) are unchanged and
    still read as the neutral two hydrogens lighter: that degeneracy is not
    resolvable from one spectrum, and what stands between the reader and the wrong
    molecule is `cleanup.annotate_easyic_ambiguity`'s dual-reading note. **Known
    cost:** nitrogen-bearing neutrals rose from 62 to 89 in the 50-200 sample and 58
    to 72 in 210-500 — every one of them wrong in a nitrogen-free cylinder. They are
    the air-plasma CxNyOz+ source ions that `_EASYIC_SOURCE_IONS` does not label, so
    the peaks are open to any grid formula; the new channels widen the opening
    rather than create it.
  - With the network scorer forced on (`PEAKY_LOCAL_SCORING=0`) these channels
    cannot be scored at all; the run now says so once per sample instead of
    dropping a declared channel without a word.

- **A privacy scan that refuses internal identifiers, run by the suite and by CI.**
  Peaky is public; the servers, workspaces, datasets and batches a run touches are
  not. The rule was applied by hand three times and missed once, when a stacked pull
  request raced the scrub and put a server name, a workspace name and a batch id
  into a test comment. `scripts/privacy_scan.py` now matches the SHAPE of an
  internal identifier — a service hostname, a private network address, a
  16-character Mascope id, a date-ranged batch name, a workspace named after a
  person — and `tests/test_privacy.py` runs it over every tracked file, so it is
  part of the required suite, and it sees files that are only staged as well as
  tracked ones. A pull request's own title and body are not files, so no file check
  can see them — the same scanner reads them from stdin (`--stdin`), and wiring that
  into CI follows separately. The rules match shapes and never a list of real names — a denylist would have to live in
  this public repository to work, publishing exactly what it hides — so a
  deliberate stand-in that must keep the shape carries a `privacy-ok: <reason>`
  comment. The one finding the new scan turned up in the existing tree, a
  date-ranged batch name in `tests/test_reflists.py`, is neutralized; the assertions
  are unchanged.

### Changed

- **Mechanisms are spelled in the standard adduct notation, and `mascope-tools`
  moves to the release that reads it.** Mascope 1.10 stores and returns every
  ionization mechanism as chemists write the ion - `[M+H]+`, `[M-H]-`, `[M+Br]-`,
  `[M]+.` - and its `mascope-tools` (2026.9.30) reads the legacy
  `<operation><moiety><moiety charge>` spelling by its grammar, under which the
  strings peaky used to hand the scorer for a subtraction meant the opposite ion:
  `-H-` for `[M-H]-` is a hydride removed, a cation, and `-H+` for `[M-H]+` a
  proton removed, an anion. So peaky now speaks the standard notation wherever a
  mechanism is named. `adduct_to_mech` is the label's one spelling through the
  library's `standard_notation` (terms in the library's order: `[M+(CH4N2O)H]+`
  as `[M+CH4N2O+H]+`, `[M+HBr+Br]-` as `[M+Br+HBr]-`), `ADDUCT_TO_MECH` maps each
  label to that spelling, and the abstraction channels ride their tokens as
  `[M-H]+` / `[M-CH3]+`. What comes back from a server is read by the mechanism
  it names, whichever notation the row stores: `resolve_mechanism_ids`,
  `detect_adducts`, the mass-error anchors and `_mechanism_names` all go through
  the library's `mechanism_key`, so a server before 1.10 (`-H+`) and one from
  1.10 on (`[M-H]-`) both work, and the sign rewrite `_mechanism_names` used to
  do by the row's polarity is gone - it would now turn a deprotonation into a
  hydride abstraction. A row whose polarity column contradicts its spelling is
  scored as it reads, the way the server reads it, and logged. Nothing about
  what is scored changes: the same ions, the same envelopes. The ceiling that
  held `mascope-tools` below 2026.9.24 comes off with it; the floor is the
  library Mascope 1.10 publishes.

### Fixed

- **`peaky batch --help` and `peaky pool --help` crashed** with `TypeError: %o format:
  an integer is required, not dict`. argparse %-expands every help string, and the
  `--residual` help (shared by both subcommands) wrote its "50%" with an f-string
  `{...:.0%}`, so a bare `%` reached argparse's formatter; every other percent in
  `cli.py` was already `%%`. The literal is now escaped like its neighbours, and
  `tests/test_cli.py` renders the help of every parser — the top level, each subcommand
  and each nested `curate` verb — so a bare `%` in a help string fails the suite instead
  of the user's `--help`.

- **`tests/test_privacy.py` failed on Windows**, with 177 `mascope-id` findings in
  `tests/fixtures/match_tree.json`, the one file the privacy scan allowlists.
  `scan_file` compared `str()` of the repo-relative path with `ALLOWED_PATHS`; on
  Windows that string is `\`-separated, while the allowlist, like `git ls-files`,
  spells paths with `/`, so the fixture was scanned there and nowhere else, and CI,
  which runs on Linux, stayed green. The path is now compared, and reported, in its
  `/` form on every OS, whether the scan walks the tree or is handed a file, so a
  finding names a file the same way everywhere and in the spelling `ALLOWED_PATHS`
  takes. The suite pins it with a pure Windows path, which behaves the same on every
  OS, so a regression fails the Linux run too.

## [0.8.0] — 2026-09-13 (per-peak admission, batch-derived floor, trace-stamped time series, the merge vote, the residual stage)

### Added

- **The residual stage: a targeted second selection after the cover's merge.**
  The presence cover stops on marginal gain, so a tail of universe bins is never in
  any assigned file — 18 % of 4702 bins on a 6154-sample uronium campaign, almost all of it the noise-edge tail, but 36 of them reaching 1–3.4 kcps
  somewhere — and nothing in that tail can ever enter the ledger. `peaky batch` /
  `pool` now run a second, targeted selection once the cover is assigned, merged and
  stamped (`sampling.residual_universe` + `select_residual_cover`,
  `assign_batch.run(residual=...)`): a universe bin is a target when it is absent
  from every assigned file, unexplained by the whole-batch stamp (an isotope
  satellite or reagent line the stamp names is not re-targeted) and bright somewhere
  — at least `--residual-min-x-edge` (5) × *that* sample's own noise edge, never below
  the run's admission-gate multiple, or `--residual-min-cps` as an absolute floor. A
  sample counts for a bin only where the bin stands at ≥ 50 % of its maximum (and
  above that file's gate), the greedy cover over that relation stops at
  `--residual-k-max` (10) picks or when the next sample adds no bin, and the picks go
  through the same per-file path (serial or the pool, same cfg and offsets). `align()`
  then runs ONCE over every per-file ledger, cover and residual together, so the
  merged ledger, the trace reconciliation and the time-series stamp include them.
  Provenance: merged rows carry `stage` (`cover` if any cover file holds the ion,
  else `residual`), `tables/selected_samples.csv` gains the picks with role
  `residual` (numbered on), `tables/residual_bins.csv` lists the targets with the pick
  that carries each, `batch_summary.json['selection']['residual']` records the funnel
  (`n_uncovered` / `n_explained` / `n_below_floor` / `n_bins_residual`), the floor,
  `k`, `coverage_of_residual`, `stop_reason` and `sample_ids` (plus
  `n_files_by_stage`, `merged_by_stage`, a `stage` per `per_file` row), `n_files` and
  `sample_ids` count both stages, the report cover and Methods page describe the
  stage, and the progress window counts the extra files on (`[phase] residual`). ON
  by default; `--no-residual` reproduces the cover-only run exactly (no `stage`
  column, no residual block, no extra file), and serial vs pool output stays
  byte-identical either way. Measured on the campaign above (manual, 21 files at 50 %
  of the maximum): 6 Assigned + 10 Candidate + 8 isotope satellites of already-assigned
  parents + 1 sidelobe + 11 unexplained out of the 36 bright bins, ~4 min per file.
  `timeseries.bin_ids` (new) is the row-aligned bin rule `build_matrix` now pivots on,
  so the residual universe is read off the long table, bin by bin, at a fraction of
  the dense matrix's memory and can never bin differently from the cover.
  **Sidelobe guard.** A live run on a 1397-sample Ur+ batch of the same campaign spent one of
  its five residual files on a bin the per-file cleanup then labelled an FT ringing
  sidelobe (1 mDa from a 584 kcps peak). Residual candidates are now tiered against a
  saturating neighbour as `flag_sidelobe_channels` tiers a channel — except that a
  neighbour qualifies by its maximum, not the batch median: the live parent (700 kcps
  at most) has a 15.6 kcps median over the 1397 samples and rings only in the plume,
  1.5 mDa from its sidelobe: a candidate/neighbour height ratio locked across ≥ 20
  shared samples (cv < 0.08, the neighbour ≥ 50 kcps and ≥ 100× over them) is a
  confirmed sidelobe and is dropped
  (`n_sidelobe`, rows kept in `residual_bins.csv` with `tier = 'sidelobe'`); a
  candidate that only meets cleanup's static rule in the sample where it peaks, with
  too few shared samples to test the ratio, is a *suspect* (`n_suspect`) — kept,
  because a real ion near a bright peak does get assigned, but covered last: the
  residual cover's gain is lexicographic, one clean bin outranks any number of
  suspects. `residual_bins.csv` gains `tier`, `sidelobe_of`, `n_pairs`, `ratio_cv`.
  The report cover now says "15 files (10 cover + 5 residual)" and previews both
  stages (5 cover rows and 3 residual rows, each with an "and N more" line) instead of
  the first 8 picks, which never reached the residual ones; without a residual stage
  it reads as before.

### Changed

- **The progress window opens by default at an interactive terminal.** `--progress`
  is no longer an opt-in switch: `peaky assign` / `batch` / `pool` open the window
  whenever a person is at a tty, and the flag became `--progress` / `--no-progress`
  (tri-state, defaulting to unset) so a run can force it either way. `PEAKY_PROGRESS=0`
  is the env off switch, `PEAKY_PROGRESS=1` the env on switch, and an explicit flag
  beats the env. The default is deliberately interactive-only — a pipe, a CI job, an
  MCP or skill-driven run has nobody to read a window and its terminal fallback would
  write `[progress]` lines into captured output — so scripted runs are unchanged.
  Relatedly, the "no usable display, falling back to terminal status" note now prints
  only when the window was actually ASKED for; defaulted on, the one-line status
  speaks for itself instead of complaining on every run. The hold stays bounded by
  `PEAKY_PROGRESS_HOLD_S` seconds (default 600; `0` disables it) and Ctrl-C still
  closes it at once; when it applies is widened under *Fixed* below. macOS still goes
  straight to the terminal status.
- **The progress window's stats panel is live.** It was filled only by the returned
  summary at the very end — every row read `--` for the whole run — so a window that
  closed on finishing never showed a number at all. `progress.py` now also reads the
  two summary lines `assign.run` logs as each file finishes (`[run] tiers {...}` and
  `[run] stats {...}`, the M0 tier counts and `ledger.stats` + the admission counts)
  and shows running per-file totals while the files are assigned: samples done, M0
  (sum), tiers (sum), the unexplained-peak share, peaks admitted by persistence. The
  merge-level rows (merged M0, tiers, in all files, single-file, formula
  disagreements) read `pending merge` until the merge, take a provisional reading from
  the `DONE:` line through the cluster / Van Krevelen / report tail, and are replaced
  by the exact summary when the run finishes — the final numbers are still never
  parsed. A file counts once, when its stats line lands; a line that fails to parse is
  not counted. `peaky assign` gets its coverage rows as soon as its one stats line
  lands, ahead of the report writes. The terminal fallback is unchanged (one line per
  sample). `tests/test_progress.py` replays the log of a 15-file uronium run whose
  window sat on `--` for 74 minutes and checks the totals against an independent parse.

- **The admission table is per PEAK, and the lookup is the table's own rule.**
  `admission.bin_occurrence` no longer gap-clusters the batch into bins: it is built on
  the new `batch/traces.PeakIndex`, and every batch peak's `occurrence` is the fraction
  of spectra holding a peak within ±tolerance of *its* m/z (one O(n) sweep over the
  m/z-sorted batch, one peak per spectrum; the window is ±max(tol ppm, 1.5 mDa), the
  stamp's own floor). `lookup_occurrence` answers a ledger peak by exactly that rule, so
  construction and lookup cannot disagree. The persistence threshold (`auto`) is Otsu's
  split of that distribution weighted so each TRACE counts once (each peak by one over
  its spectrum count): it reproduces the former bin-table split to ±0.02 on four
  Orbitrap modes and reads 0.38 instead of 0.44 on a TOF batch; unweighted, the
  per-peak histogram is dominated by persistent ions and splits ~0.1 too high.
  `batch_summary.json['admission']` reports `n_peaks`, `n_persistent_peaks` and
  `n_persistent_traces` (the trace-weighted count, ≈ the number of persistent ions)
  in place of `n_bins` / `n_persistent_bins`, which counted bins under a rule the
  admission decision did not use.
- **The brightness floor is derived from the batch by default.** An unset
  `height_cutoff_x_edge` now resolves to the policy token `"auto"`
  (`passes.config.AUTO_HEIGHT_CUTOFF_X_EDGE`); `--height-cutoff-x-edge auto` asks for
  it explicitly and a reagent profile may carry it. On a batch, `assign_batch.run`
  resolves it to a number from the batch's own peaks
  (`admission.derive_height_cutoff_x_edge`), on two conditions: the picker must pick
  INTO the noise — more than 5 in 10 000 of the batch's peaks sit below 0.75× their
  sample's 1st-percentile edge; a hard-threshold picker leaves essentially nothing under
  its edge — and then the floor is the smallest of 1 / 1.5 / 2 / 3 / 5 / 8 / 12 / 20 at
  which the peaks it would admit are at most 20 % transient (occurrence below the
  persistence threshold). The number is stamped on the cfg every per-file run copies
  and recorded with its source, both statistics and the whole transient-share table
  (`batch_summary.json` `height_cutoff_x_edge`, `height_cutoff_x_edge_source`, `gate`).
  A single sample with no batch reads `"auto"` as 1.0 (`DEFAULT_HEIGHT_CUTOFF_X_EDGE`),
  and so does a batch too short for a persistence table (with a source saying why).
  Measured live on five batches of two instruments: every Orbitrap mode keeps exactly
  1.0 (tail share ≤ 1 in 10 000; three modes 8–16 % transient at 1×); the TOF (tail
  share 31 in 10 000, 34 % transient at 1×) lands at 5× (17 %; inside the recall
  plateau of the live sweep
  — 17 of 18 findable target ions from 5× up — though below its 8–12× precision
  optimum, which a site pins in its profile). The tail condition exists because the
  reagent-in-range Orbitrap mode is 50 % transient at 1× (the reagent ion's noise skirt
  plus a scan-edge pile-up) and the share alone raised it to 2×, which removed 45
  pile-up rows and 12 skirt rows but also 14 multi-file Assigned ions at 1–2× the edge
  (a C6H12O `[M+H]+` seen in 9 of 27 files among them) — there the edge is the picker's
  relative threshold under a huge reagent ion, not noise. The reproducibility manifest
  fingerprints the *token* for an unset multiple, never the derived number (which is
  run-derived, like the persistence threshold, and lives in `counts`).
- **The batch time series is stamped from each ion's TRACE, not from the merge
  anchor.** Between the merge and `annotate_peaks`, `timeseries.recentre_ledger`
  re-centres every merged row on its own trace (`PeakIndex.mean_shift` from the
  anchor, at most 10 ppm; an anchor covering < 10 % of the spectra that wants to move
  > 6 ppm must be seen in ≥ 2 files or be Assigned), `collapse_trace_labels` collapses
  rows whose trace centres fall within the merge tolerance onto one winner (the
  merge's own ordering: most assigned files, then tier, then `ion_score`, with
  proximity to the trace centre only as a deterministic tie-break — never
  `ion_score` first, which is per-file and blind to reproducibility, and never
  proximity before the score, which on theory-snapped TOF anchors carries no
  information and lost the known monomer C10H16O9 to a one-file competitor), and
  `stamp_tolerance` sizes the stamping window to the batch's own per-ion scatter (2.5 ×
  the third quartile of the per-trace robust scatter, between 1× and 2× the merge
  tolerance). The merged ledger gains `mz_anchor`, `mz_trace`, `trace_offset_ppm`,
  `trace_cov_anchor`, `trace_cov`, `trace_moved`, `trace_guarded`, `trace_id` and
  `trace_role` (`single` / `winner` / `collapsed`; nothing is dropped);
  `_batch_ts.parquet`'s `ion_mz` is now the trace centre; `batch_summary.json['traces']`
  records the counts, the scatter and the window. Why: on a TOF the assignment snaps
  the merge anchor to theory (a formula is only committed where a sample's draw lands
  near it) while the ion's trace sits several ppm away — 22.6 % of one 230-spectrum TOF
  batch's anchors were outside their own trace's window and 26.8 % of its well-populated
  rows shared a trace with a competing label. Measured on that batch's real runs
  (`scripts/eval_trace_stamping.py`): mean stamped coverage over all ions 0.422 → 0.488
  with 38 % of ions gaining more than 5 points and 3.7 % losing, the highly oxygenated
  monomer C10H16O9 0.135 → 0.857, window ±9.5 ppm; two Orbitrap batches moved by
  0.16–0.34 ppm and changed nothing (0.754 → 0.754, 0.542 → 0.542; window ±6 ppm) —
  the no-op that validates the diagnosis.
- **Reference-list context tags are resolved from the batch name, the DATASET name and
  the reagent label** (`assign_batch.run`, the pooled pipeline and the report all pass
  the dataset), and the phrase "AP oxidation" (also `ap-oxidation` / `ap_oxidation`)
  unlocks the alpha-pinene and monoterpene contexts. On a campaign whose batch names
  describe instrument and reagent the chemistry lives only in the dataset name, and the
  alpha-pinene HOM reference list had been silently inactive on alpha-pinene data.
- **The positive pass-0 `cyclosiloxane` and `indoor_sulfur` families apply to EVERY
  positive context**, not only to the labelled-ammonium runs they were seeded from:
  `directors._known_species("positive", …)` returns them regardless of context, so
  uronium and EasyIC runs now also get `known:` locks for D3–D7 / L2–L5 siloxanes and
  for the benzothiazole / DMSO / thiophene / sulfolane / NBBS set. The commit gates are
  unchanged (≥2 channels, or a confirmed ²⁹Si/³⁰Si envelope + the Si-count M+1 check,
  or a confirmed ³⁴S envelope), so this adds locks only where the evidence is already
  there — but a previously Candidate D4 or benzothiazole can now come out Assigned.
- **The pass-4 halogen cap applies in every context, not just positive ones.**
  `residual.stage_a_iso_pairs` drops a ~1.998-Da doublet whenever the context caps that
  halogen at zero (`max_Br` = `max_Cl` = 0) — the rule is written against the context
  profile, so any halogen-free context gets it.
- **The labelled-nitrogen ¹⁴N envelope line is emitted for every `^N`-bearing ion**, so
  it reaches the ¹⁵N-NITRATE profile as well as the ammonium one it was built for: a
  `[M+^NO3]⁻` cluster now predicts a −0.997 Da satellite at `(1 − purity)/purity` of M0
  and `complete_isotope_envelopes` will claim it. **Caveat, to be validated on a
  labelled-nitrate batch:** a labelled-nitrate source can carry a *real* `[X+¹⁴NO₃]⁻`
  analyte channel from the reagent's unlabelled fraction, and it sits at exactly that
  mass and at a comparable 0.6–7 % of the labelled cluster. Such a channel would now be
  attached as an isotope child of the labelled reading rather than standing as its own
  M0. Gating the line per profile is deliberately NOT done here (see the open items).
- `ReagentProfile.purity` is no longer inert. `assign.run` publishes it
  (`isotopes.set_label_purity`) and it now drives BOTH the local scorer's
  `predict_isotopes` call and peaky's own envelope predictor; `isotopes.LABEL_PURITY_15N`
  remains the default. Behaviour-neutral at 0.98, which is also mascope_tools' default.

- **Batch sample selection is one greedy presence set-cover** (`sampling.
  select_cover_samples`, `docs/SAMPLING.md`), replacing the 5-time-spaced+max-TIC
  rule and the brightest arg-max cover. Universe = the batch's m/z bins present
  in ≥ 2 samples — **no height floor**, because the peak picker's detection edge
  spans ~1000× between instruments and modes (0.8 cps on a TOF, ~800 cps on an
  Orbitrap mode with the reagent ion in range), so any absolute floor is a no-op
  on one and blinds the selector on another. Each pick is the sample holding the
  most not-yet-covered bins; the run stops when the next pick would add
  < `--min-gain` (0.5 %) after `--k-min` (6) picks, and `--k-max` (30) is a
  flagged budget rather than a target. The merge keeps whatever any assigned
  sample contained and nothing else, so this selection is what decides recall:
  on a pooled 5036-sample field-campaign table the cover lands at k = 15 and
  holds 94 % of the rare (< 5 % prevalence) ions of a dedicated sub-batch run,
  vs 54 % for the old default and 91 % for the old 12-sample arg-max cap.
  `pool` runs the same single cover over the pooled table (a loud group cannot
  hog a presence objective; per-group achieved coverage is recorded) instead of
  a per-group union. `batch_summary.json` / `run_manifest.json` gain a
  `selection` block (`k`, `n_bins`, `achieved_coverage`, `stop_reason`,
  `next_gain`, `tol_ppm`, `coverage_by_group`); `tables/selected_samples.csv` is
  now in pick order with `pick`, `role` (`cover` / `pad`), `bins_new`,
  `coverage`. `sampling.BATCH_TOL_PPM` (6 ppm) is the one m/z binning tolerance
  for every batch-level operation — the selector bins on it and
  `assign_batch.DEFAULT_TOL_PPM` *is* it, so selection and merge bin identically.
- **Height thresholds in the passes are multiples of the sample's own noise
  edge**, not absolute cps. `assign.run` computes `noise_edge_cps` (the 1st
  percentile of the sample's picked heights) once per sample; `PassConfig.
  height_cutoff` is now a read-only property = `height_cutoff_x_edge` (`None` =
  unset, resolving to 1.0) × that edge, with `height_cutoff_cps` as an override
  for offline callers. `peaky assign --height-cutoff` becomes that override
  (default none) and `--height-cutoff-x-edge` sets the multiple; the MCP
  `assign_sample(height_cutoff=)` likewise. Per-file `noise_edge_cps` and the
  resolved gate `height_gate_cps` are recorded in `batch_summary.json`
  (`height_cutoff_cps` in `run_manifest.json['config']` stays the knob).
- **Eight formula-hunting sites draw their candidates from the admission gate
  instead of brightness alone**: the pass-1 grid, pass-2 series growth, pass-3
  contaminant families and pass-3's two cluster resolvers (HX, acid·I₂) — those
  five share `directors._target_peaks`, so one gate call covers them all — plus
  the pass-6 ladder gap-fill, residual stage B and the siloxane ladder (its work
  set *and* its seed test — the lowest rung of a weak ladder is exactly a
  persistent sub-gate peak, and a brightness-only seed test left it a 2-member
  run). Still brightness-only, a documented follow-up: residual stage A, the
  reflist rescue, pass-3's series *detection* statistics (`series_detect`) and
  the isotope-satellite tests in `passes/postprocess`. Re-derive the gated set
  from the callers of `_target_peaks`, not from a grep for `admissible(`.
  The admission table joins selection and the merge on `sampling.BATCH_TOL_PPM`
  (above), so `batch` and `assign --ts-batch` bin a peak the same way.

### Removed

- `peaky batch --select / --coverage-target / --height-floor` and
  `peaky pool --coverage-target / --height-floor` (`--k-max` stays, default 30;
  `--k-min` / `--min-gain` added to both). `sampling.select_representative_samples`,
  `select_brightest_coverage_samples`, `select_pooled_union` and the `*_sample_ids`
  wrappers; `assign_batch.run(select=, coverage_target=, height_floor=, n_time=,
  include_max_tic=)`, `pipeline.run_batch(select=, …)`, `run_pooled_batches(
  coverage_target=, height_floor=)`; `PassConfig(height_cutoff=)` (use
  `height_cutoff_cps=`); the MCP `run_batch(select=)` (now `k_max=`).

### Fixed

- **The deep-series tie-break no longer depends on `PYTHONHASHSEED`** (#8). Residual
  stage B collected its anchors into a set and `series_gka.propose_for_peak` walked that
  set, so when two anchors reached the same candidate with identical support and mass
  error (C4H6O4 + CH2 and C6H10O4 − CH2 both give C5H8O4) the anchor named in the Pass-4
  commentary was whichever the set yielded first — a different one per process. The
  formula, score and tier never varied; the ledger bytes did, against the determinism
  contract in `ARCHITECTURE.md`. `propose_for_peak` now walks the anchors in sorted order
  and breaks a support / ppm tie by the fewest steps, then the anchor, unit and adduct
  text, so its result is a pure function of its arguments and the nearest anchor is the
  one recorded; pass 5 walks its anchors sorted too. `tests/test_determinism_assign.py`
  drives stage B in subprocesses under six different hash seeds and requires byte-identical
  ledgers (on the previous code the same six seeds named three different anchors).
- **`peaky batch --batch` settles the batch ONCE, and the time series can no longer
  pool sibling batches.** The SDK matches a plain string as a case-insensitive literal
  SUBSTRING and applied that rule differently on the two calls a batch run makes:
  `load_peaks` (the time series) kept *every* batch the string occurred in, while
  `samples.list` (the roster) demanded a unique match. On a dataset where one batch's
  name is a prefix of its siblings' (`Site A Ur 122-600` beside `Site A Ur 122-600 11`
  and `... wind zone 1`), `--batch "Site A Ur 122-600"` silently pooled all three
  batches' peaks into the time series and then died at the roster with "Multiple
  batchs matching"; addressed by id, the roster resolved but the time series found
  nothing and the run died with "no peaks for batch". `io_mascope.resolve_batch` now
  settles `--batch` up front — an exact batch id, else an exact (case-insensitive)
  name, else a unique literal substring; an ambiguous string raises with the
  candidates, before any run folder exists — and both fetchers go through it: the
  roster is asked by id and the time series by the resolved name with `exact=True`.
  A name several batches in the dataset share is refused for the time series (the
  SDK loader addresses batches by name only, so it cannot be isolated) rather than
  pooled. The resolved DISPLAY name now titles the run folder, the report cover,
  `batch_summary.json` and the manifest even when the run was addressed by id (the
  id is recorded beside it as `input.batch_id` / `RunContext.batch_id`), so the
  `--batch <id> --ts <parquet>` workaround no longer produces a folder named after
  an opaque id. `peaky list samples --batch` and the MCP `list_samples` / `run_batch`
  tools take a name or id the same way; `peaky list batches` and the regex-driven
  `peaky pool` are unchanged.
- **The merge is a vote, and the merged row shows it.** `assign_batch.align` now
  decides each m/z cluster in two stages. *Which ion*: the ion (neutral + adduct
  composition, `_ion_key`) carried by the most FILES wins, with the Assigned-file
  count and `ion_score` only as tie-breaks (the order `collapse_trace_labels`
  already used for competing labels on one trace) and the ion's own text as the
  last key, so serial and parallel runs stay byte-identical. *Which label of it*:
  the reading Assigned in the most files — on a same-ion pair such as
  `C13H14O4 [M+NH4]+` vs `C13H17NO4 [M+H]+` (the reagent-N isobar) the tier engine
  marks a file Assigned only when a discriminating channel was present and
  Candidate when it had nothing to decide with, so counting Candidate files would
  be counting silence. The previous rule ranked the Assigned-file count first for
  everything, so one file's Assigned reading outvoted many files' Candidate
  reading of a *different* ion: on a 15-file uronium run 12 of the 73 split
  clusters were decided by a minority, and nothing on the merged row said so; of
  those 73, 43 were two labels of one ion and 30 different ions, and in 16 of the
  43 a pure count would have handed the ion to a label nobody had corroborated.
  A curated identity — a reference-list rescue or the pass-0 known-species list
  (`_curated_neutrals`) — is exempt from the file count, not from corroboration:
  Assigned in at least one file and in no fewer files than any grid ion, it is
  not outvoted by grid guesses. On that run the D7 cyclosiloxane urea adduct at m/z
  579.171 and tricresyl phosphate at 429.157, each locked in one file, would
  otherwise have lost to an O14 / N4O10 grid formula the per-file engine itself
  flags as implausible (Candidate in every file); sulfolane at 181.065, from the
  same list in one file, met fluorenone `C13H8O [M+H]+` Assigned in nine, and the
  count decided that one. (`certified:` neutrals get no exemption: a multi-channel
  certification is the file's own evidence for a grid formula, already credited by
  its tier.) The merged row records the vote — `n_files_ion` and `n_files_winner`
  (files carrying the winning ion / reading; `n_files` is the cluster),
  `alternatives` (the losing readings, best first: `C15H25N [M+H]+ x1 Candidate
  0.97`), `ion_agree` beside `formula_agree` (a same-ion label split is not a
  spectral disagreement), and a `tier_reason` note whenever the exemption or a
  corroboration-over-count choice decided it — and `batch_summary.json` adds
  `ion_disagreements` next to `formula_disagreements`.
- **The hydrocarbon-on-N-cluster re-read is decided once per batch.**
  `cleanup.relabel_reagent_n_adducts` re-reads a pure hydrocarbon seen via
  `[M+NH4]+` / uronium as `[M+H]+` of the N-heterocycle unless the hydrocarbon also
  shows its own `[M+H]+` — a presence test that, per file, flips with S/N: on the
  same run `C15H22 [M+NH4]+` was kept in the fourteen files holding
  `C15H22 [M+H]+` and re-read to `C15H25N [M+H]+` in the one that did not (the stage
  fired 31–52 times per file), a disagreement the spectra never had and, for the
  vote, a phantom minority. A batch now runs the per-file stage off
  (`assign.run(reagent_n_relabel=False)`, set by `assign_batch.run`) and applies the
  same rule once to the merged ledger, where every file's `[M+H]+` rows are pooled.
  Single-sample runs are unchanged.
- **The merged ledger explains its own re-reads.** The merged row carries
  `tier_reason`, so the batch-level gates' notes land on the row a reader sees — a
  merged formula could differ from every per-file reading with nothing on the row
  to say why (`cleanup._note` found no such column and wrote nothing). The
  reagent-N note names the reading it replaced
  (`re-read C11H20 [M+(CH4N2O)H]+ as [M+H]+ of C12H24N2O: …`), and
  `batch_summary.json["merge_gates"]` records both gates' counts.
- **An explicitly requested progress window closed unread when the run was launched
  off a terminal.** The post-run hold was gated on a tty alone, so `setsid nohup peaky
  batch ... --progress` with `DISPLAY` set opened a real window on the desktop and tore
  it down the instant the run finished — 74 minutes of bars, and the stats panel it was
  opened for never visible. `progress.hold_wanted` now holds a finished window at an
  interactive terminal OR wherever the window was asked for explicitly (`--progress`,
  or `PEAKY_PROGRESS` truthy) and a Tk window actually came up — still for at most
  `PEAKY_PROGRESS_HOLD_S` seconds (default 600; `0` disables it), still ended at once
  by Close or Ctrl-C. The implicit default and the terminal fallback never hold, so a
  script or a CI job cannot wait on a window it did not ask for or could not show. The
  `--progress` help text and the README say so.
- **`run_manifest.json` stamped the wrong package version on every run.**
  `peaky.__version__` restated `"0.5.0"` as a literal while `pyproject.toml` had moved
  to `0.7.0`, so `code.package_version` — the field that exists to tell one run folder's
  code from another's — read 0.5.0 on runs produced by 0.7.0 code and could not
  distinguish them at all. `__version__` is no longer declared in `peaky/__init__.py`:
  it is resolved on first access from pyproject's `[project].version` (a source
  checkout or editable install, name-checked so a vendored `peaky/` cannot inherit a
  host tree's version), falling back to `importlib.metadata` for an installed wheel and
  only then to a literal that `tests/test_shim.py` pins to pyproject. Resolution is lazy
  and cached, so `import peaky` stays ~0.1 ms. `git.commit` / `branch` / `dirty` were
  always correct and remain the way to trace an existing run folder.
- **`code.module_versions` is derived from the package, and now names every module.**
  The registry was a hand-written dict in `assignment/assign.py` listing only the
  modules that file imports at module level, so it had gone 17 modules stale — among
  them `traces`, `pipeline`, `sampling` and `assign_batch`, i.e. exactly the per-peak
  admission and presence set-cover selection behaviour, all reporting `None`.
  `assign.module_versions()` now walks the package and reads each module's
  `__version__` from its AST (parsed, never imported: importing 38 modules to read a
  string would drag matplotlib and the Mascope SDK into every run, and `assign.py`
  cannot import the `pipeline` that calls it). All 38 modules are registered, the
  result is cached per process, and `tests/test_provenance.py` asserts the registry
  equals what is on disk — so a new module is fingerprinted without anyone
  remembering to list it. `MODULE_VERSIONS` (the dict) is gone; call
  `module_versions()`. `code.module_hashes` already pinned every file by sha1, so
  past runs stayed reproducible — only the human-readable naming was missing.
- **The repo's version strings are pinned to each other.** A new
  `tests/test_versioning.py` asserts the whole set agrees: `peaky.__version__` and
  `_FALLBACK_VERSION` against pyproject, `uv.lock`'s pin against pyproject (so a bump
  that forgot `uv lock` fails offline, not just under CI's `--frozen`), and
  `CITATION.cff` against the CHANGELOG. `CITATION.cff` deliberately LAGS pyproject —
  it names the last release actually tagged, published and archived, which a citation
  has to resolve to, and as of 0.7.0 in pyproject that is still 0.6.0 — so it is
  checked against a dated `## [x.y.z]` CHANGELOG heading (version and date both) and
  for never running ahead of pyproject, rather than for equality with it. The comment
  in the file now says so outright instead of leaving the gap to read as drift.
  The 0.5.0 section heading, which still read "Unreleased", is retitled to name the
  release and its date (2026-06-30): 0.5.0 was tagged and released, and a released
  section labelled unreleased made this file's own history unreadable.
- **The isotope bookkeeping lost the parent↔satellite link, which made the P/S
  corroboration gate unauditable from a run's own output.** Two halves of one defect,
  measured on a urea-reagent 122–600 Da batch: (a) all 453 `iso_child` rows in one file
  recorded `parent_peak_id` but nothing about WHO that parent is, so reading a satellite meant
  joining it back to the M0 row; and (b) known-species (pass-0) commits wrote
  `isotopologues: []` even when a matched satellite was the evidence that *licensed*
  them. `C8H13O5PS2` (malathion transformation product, −C2H6O) commits at m/z
  285.00145 on a SINGLE ion channel, so it can only have passed the
  organophosphate/-thiophosphate/indoor-sulfur gate — ≥2 ion channels OR a confirmed
  diagnostic ³⁴S/³⁷Cl/⁸¹Br envelope — via the isotope route, and both its satellites
  were picked and attached (¹³C at 286.0048, ³⁴S at 286.9973, expected 286.9972); the
  ledger recorded none of it. Confirming the commit was legitimate took reading
  `directors.py` and computing the +1.99580 ³⁴S offset by hand. Now:
  `attach_isotopologue` stamps the owner's identity onto the satellite
  (`parent_neutral_formula` / `parent_adduct`, new columns beside `parent_peak_id`),
  so every attach path — `complete_isotope_envelopes`, the siloxane/residual/cleanup
  attachers, `displace_to_isotopologue`'s re-parented grandchildren — yields a row
  that reads "³⁴S satellite of C8H13O5PS2 [M+H]+" on its own, and `validate` treats a
  stamp disagreeing with its parent as an I2 violation; the pass-0 commit path records
  the Mascope-scored satellites it rested on, and the isotope-locked chlorinated-paraffin
  recovery records the ³⁷Cl envelope it was locked on; and `tier_reason` now names the
  route, `"; corroborated by 1 ion channel + a confirmed ³⁴S satellite"` versus
  `"; corroborated by 2 ion channels"`, read off the ledger's own columns (channels
  from the M0 rows sharing the neutral, satellites from `isotopologues` **and** from
  the `iso_child` rows pointing back), so it is right on ledgers written before this
  fix too. Re-tiering that batch's untouched ledger reports all three of its
  single-channel isotope-licensed commits (the malathion TP, `C2H6OS`, `C8H7NS2`)
  correctly. The gate
  had not leaked — it was simply impossible to tell from the output.
- **The persistence path was inert at the shipped default, and the TOF ledger was a
  flood.** With `height_cutoff_x_edge = 1.0` the brightness path admitted ~99 % of a
  TOF's picked peaks (0.12 % of admissions came from persistence), and the merged
  ledger of a 230-spectrum TOF batch was 64 % single-file with 70 % formula
  disagreement among multi-file rows. The batch-derived floor (above) raises that
  batch to 5× — the live sweep measured 1988 merged rows instead of 5331, 55 %
  single-file, recall of the known monomers unchanged or better — while leaving the
  Orbitrap modes whose picker already stops at the noise edge at exactly 1.0.
- **A peak could fail to find its own occurrence bin.** The table was built by
  single-linkage chaining (a bin breaks only where consecutive peaks are more than
  the tolerance apart) and read within ±tolerance of the bin's height-weighted centre:
  on a TOF batch 91 % of the persistent bins were wider than the tolerance (median 51
  peaks, one 134 ppm), so a large share of the persistence signal was the chaining and
  3149 rows sitting inside persistent bins were denied the path at one operating point.
  Matching the lookup to the bin span handed a chained bin's persistence to adjacent
  noise; capping the bin width split one ion's jitter cloud across bins. The per-peak
  rule (above) has neither failure.
- **The report cover under-reported the persistence path** by an order of magnitude:
  it counted MERGED rows admitted by persistence (0–3 on real batches), but the merge
  keeps the highest tier then score so those rows almost always lose to a
  height-admitted row in the same bin; per-file admissions were 14–63, and on one batch
  the line was suppressed entirely. The cover now reports the per-file total alongside
  the merged count, and states the batch-derived floor and the trace reconciliation.

- **The batch time series carried a peak once per MATCH, not once per peak.** Mascope's
  peak loaders expand a peak into one row per target isotope it matches
  (`load_peaks` / `samples.get_peaks`, `matches=True` by default), and nothing in the
  batch path folded that back down — so `per_file/_batch_ts.parquet` held two rows for
  every peak two targets claimed: same `sample_item_id`, same `peak_id`, byte-identical
  `mz` / `area` / `height`, only the advisory `target_*` columns differing. Targets
  collide exactly when they imply the SAME ion (a neutral read as `[M+NO3]-` and a
  neutral one HNO3 heavier read as `[M-H]-` are one ion formula), so the pair sits at
  exactly 0.00 ppm and no mass filter separates it. Everything downstream counted the
  peak twice: `build_matrix` sums heights per (sample, m/z bin), so that bin's intensity
  doubled, and a per-trace peak count read **2.0 peaks/sample for a single ion** — which
  reads as two merged ions rather than one peak listed twice. Measured on two field
  batches: 0.29 % and 0.42 % of rows, always pairs. New
  `timeseries.collapse_peak_matches` keys on `(sample_item_id, peak_id)` — or
  `(sample_item_id, mz)` for a frame trimmed to the TS columns — and keeps the
  best-scoring match's `target_*` columns, ranking on the match scores with the target
  ids as a final tie-break so the winner is fixed by content rather than by the order
  the server returned rows in. Row order is preserved and a clean frame comes back
  untouched, so it is a no-op on collapsed input and idempotent; it runs wherever a time
  series enters (`pipeline.load` / `run_batch` / `run_pool` / `generate_report` and
  `assign_batch.run`), which also protects sample selection, the amine gate and sidelobe
  flagging. Existing parquets are repaired on read.

- **A missing `--group-by` value no longer crashes pooled selection.**
  `sampling.select_cover_samples(group_col=…)` now normalises the group labels to
  strings before it sorts them: on pandas ≥ 3 `astype(str)` leaves NaN alone, so a
  group column holding any null value raised `TypeError: '<' not supported between
  instances of 'str' and 'float'`. Ungrouped samples stay selectable — their bins
  are real — and group under `sampling.UNGROUPED` (`"(ungrouped)"`), which no
  per-group report matches.
- **A batch whose peaks all have zero height no longer produces a NaN coverage.**
  Every bin failed both prevalence gates, the universe was empty, and the mean of
  that empty array became the `coverage` column, `selection.achieved_coverage` and
  a bare `NaN` token in `batch_summary.json` (rejected by strict JSON parsers),
  with numpy warnings on the way. An empty universe now returns an empty selection,
  like a batch with no bins at all.
- **The brightest-coverage selector silently never reached its coverage
  target**: every run on disk had assigned exactly `k_max` + 2 samples at
  0.39–0.61 achieved coverage while `batch_summary.json` recorded the *requested*
  0.85. The new selector records the achieved coverage and the stop reason, and
  warns (log, report, summary) when the `k_max` budget binds while the batch is
  still gaining.
- **The absolute 100 cps `height_cutoff` blinded the height-gated passes
  (ladders, siloxane, residual, reflist rescue, isotope-satellite checks) on
  low-edge modes**: it excluded 97 % of picked TOF peaks and 87 % of an EasyIC
  mode's, while being a no-op on modes whose picker edge sits above it. The
  edge-relative gate keeps every picked peak but the bottom 1 % eligible on any
  instrument. The gate also bounds which unassigned peaks pass 1 enumerates, so
  on a TOF batch the primary pass had been committing ~2 grid assignments per
  file; with the edge gate it commits ~370 (self-calibration backbone 220–460
  peaks, median |ppm| 0.87 vs 1.13 before), most of them Candidate-tier leads
  at 1–5× the edge — a TOF ledger now carries that Candidate tail by design.
  `peaky assign --height-cutoff-x-edge` (the single-sample command; `batch` and
  `pool` gate at the 1.0× default) raises the bar when a tighter list is wanted.
  An EasyIC batch went from 63 to 119 merged M0 with no formula disagreements.

- **A reference list's radicals are read off the formula, not the hydrogen count.**
  `reflists.load_catalog` sorted each species into the closed-shell pool (matched by
  default) or the radical pool (skipped unless `include_radicals=True`) by a
  hand-set `radical` flag, and the monoterpene HOM list had set that flag from an
  odd hydrogen count, which is wrong once nitrogen is present. Its 118 organic
  nitrates (C10H15NO8, C10H13NO10, ...), the main HOM class under NOx, sat in the
  radical pool, where the selection prior, the rescue and the report's
  corroboration never looked, and three nitrogen-bearing radicals were matched
  instead. The loader now reads radical status off DBE parity (a half-integer DBE
  is an odd-electron neutral) and treats the flag as a claim, warning when a list
  disagrees. The list's flags are corrected: 573 closed-shell and 257 radicals,
  where they said 458 and 372 (`data_version` 2024.2).

- **The Keller contaminant list holds molecules only.** Nine entries were ions or
  salts: the CN fragment of acetonitrile, the tetrabutylammonium, trityl and
  monomethoxytrityl cations, and five quaternary-ammonium chlorides, four of them
  at a negative DBE. They are dropped. NMP, entered as its protonated ion, is the
  neutral C5H9NO — which puts its [M+H]+ back on the m/z 100.07569 the source
  measured, so the list is 48 of 50 verified against the source masses rather
  than 47 — and acetic and propionic acid are named for the acids rather than the
  iron complexes the source saw. The list goes from 59 species to 50
  (`data_version` 2008.2), the split Mascope's reference seed uses.

- **`peaky assign` unlocks its reference lists.** Only the batch path ever called
  `active_lists`, so a single-sample run passed `reflists_active=None`: the
  selection prior was empty, the rescue-verify pass never fired, and the manifest
  key above — new in this same release, so it never shipped empty — would have
  landed `[]` on the single-sample path. `reflists.activate` is now the single
  unlock step (`resolve_context_tags` → `active_lists`, tags returned so the
  caller can log them), and `cli.cmd_assign`, the MCP `assign_sample` tool,
  `assign_batch.run` and the report path all go through it. A lone sample has no
  batch name — its metadata is the context plus the reagent profile's label, the
  same pair on both single-sample paths — so it activates the always-active
  lab-contaminant list and nothing chemistry-specific unless one of those names a
  chemistry; `peaky batch` is still the way to unlock a HOM list.

- **A formula is validated before its parity is trusted.** `chemistry.dbe` scores
  an element outside the mass table as divalent — sodium acetate comes out at
  DBE 1.5 and would have pooled as a radical with no message — and
  `parse_formula` ignores charge and bracket notation, so an anion written
  `[C10H14NO8]-` would have loaded as a closed-shell neutral. `load_catalog` now
  warns and skips (counted, on `ReferenceList.skipped`) any species whose formula
  uses an unknown element, does not round-trip as a Hill-notation neutral, or has
  a negative DBE. The bundled lists skip nothing.

- **One parity test for the whole package.** `chemistry.odd_electron` is now the
  single DBE-parity helper the closed-shell grid gate (`dbe_ok`), the
  plausibility radical exemption and the reference-list loader all call, instead
  of three copies of the same half-integer test; `dbe_ok` calls a half-integer
  DBE "odd-electron (radical) — blocked on the closed-shell grid" rather than
  "not a valid neutral", which is what the reference lists had always called it.

- `tests/test_peaklists.py` loads every bundled list and fails on a species with
  a negative DBE or a radical claim its parity contradicts, and holds the
  corrected pool to its consumers: an organic nitrate's ion mass is matched by
  `match_by_mass` / `match_assigned` and carried into the selection prior, a real
  nitrogen-bearing radical's is not, and both are only with
  `include_radicals=True`. Mascope keeps the same test on its own copy of the
  lists.

### Added

- **The batch stamp predicts the diagnostic isotope satellites of every merged analyte**
  (`timeseries.predicted_satellite_rows`, called by `stamping_frame`; stamped by
  `annotate_peaks`). A per-file ledger claims a satellite only where that file's picker
  picked it, and the faint diagnostic lines — 15N (0.36 % per N), 18O (0.20 % per O), a
  single 34S / 29Si / 30Si — sit below the picker's ~150–220 cps edge in most files, so a
  parent Assigned in every assigned file still left its 15N / 18O tracks unexplained
  wherever a plume lifted them into view (a 6154-spectrum uronium batch, 15 assigned:
  C12H27O4P [M+(CH4N2O)H]+ and C12H14O [M+NH4]+ Assigned in every ledger with only their
  13C claimed; the 15N line at m/z 328.2013 stood in 109 spectra up to 1.1 kcps and the
  18O line at 194.1425 in 4, picked in none of the 15 files — while a targeted
  single-sample assign of a plume file claimed both, so the per-file logic was right and
  only the batch stamp was blind). The stamp now adds one `iso_child` row per (parent,
  label) for the 13C / 81Br / 37Cl / 15N / 34S / 29Si / 30Si / 18O lines
  (`cleanup.reclaim_satellites`' diagnostic set) of every M0 with a known ion formula, at
  the parent's stamped m/z (its trace centre, so the instrument offset carries over) plus
  the line's exact shift, with the pattern's atom-count-aware relative abundance
  (`chem.isotopes.isotope_pattern`, merged at 0.5 mDa rather than its 6 mDa default,
  which folds 18O into the 13C2 centroid 13 ppm away at m/z 194). Precedence: a satellite
  a per-file ledger observed supersedes the predicted one of the same (ion, label), and no
  predicted row is minted within the stamping window of any known row — an assigned
  analyte at a satellite offset is an analyte. `annotate_peaks` matches every known row
  first (a peak inside any known row's window, winner or one-to-one loser, is never
  offered to a predicted line) and stamps a predicted line only under the per-file
  passes' intensity gate: the parent must be stamped in the **same sample** and
  height / (parent height × rel) must lie in 0.3–3.5; the one-to-one contest then runs
  among the gated candidates, so a shoulder that fails the gate cannot take the label from
  the real satellite. On top of the per-sample gate sits a **track-coherence rule** it
  cannot express: a true satellite's ratio is a constant of nature and passes the window
  in nearly every judged sample, whereas an independent compound on the line fails in
  most and passes in the few where its height happens to fit — measured on two live
  runs, 7 and 8 such tracks at 1–26 % pass share, 24 and 268 mislabelled peaks that the
  per-sample gate alone handed out. Once a line has been judged in ≥ 10 samples (parent
  present, a candidate on the line; `PRED_TRACK_MIN_N`) it keeps its stamps only if ≥ 50 %
  of them passed (`PRED_TRACK_MIN_SHARE`); otherwise the whole track stays unexplained,
  unstamped and unflagged. Lines judged in fewer samples stand on the per-sample gate.
  `_batch_ts.parquet` gains `stamp_source` (`M0` / `observed` / `predicted`);
  `tables/predicted_satellites.csv` audits every predicted line that had a candidate
  (samples judged / passed, pass share, judged, kept, peaks stamped);
  `batch_summary.json['traces']['stamp']` counts the stamped peaks by source
  (`n_iso_observed` apart from `n_iso_predicted`), the predicted tracks stamped / judged /
  rejected and the predicted rows minted and superseded;
  `stamping_frame(..., predict_satellites=False)` restores the observed-only stamp. A track
  explained as a predicted satellite carries an `ion_formula`, so the residual stage (which
  reads the cover's stamp) no longer targets it as an unexplained bin. The per-file ledgers and their coverage figures are untouched. Re-stamped
  offline on three finished batches: every previously stamped peak kept its stamp and its
  `dup_candidate` flag; 6 766 of 969 135 peaks (10-file uronium batch, 60 tracks),
  7 423 of 3.29 M (12-file uronium, 229 tracks) and 2 309 of 1.96 M (12-file labelled
  nitrate, 118 tracks) were newly explained, with a median measured / predicted height
  ratio of 0.79–0.91 — the 34S lines of the two brightest thio-compounds in every one of
  1 397 spectra among them; the coherence rule then removed 0, 24 and 268 of those stamps,
  every one on a track whose median ratio also sat outside 0.5–2.
- **`peaky/batch/traces.py`** — the one trace primitive over the m/z-sorted batch peak
  list: `PeakIndex` (exact-duplicate rows dropped, deterministic sample codes;
  `occurrence` — the per-peak distinct-spectrum sweep; `occurrence_at`; `coverage_at`;
  `members` — the nearest peak per spectrum within the window; `mean_shift` — the
  local mode of the peak density, capped in drift; `scatter_ppm` — the robust per-ion
  mass scatter, re-measured in a wider window when the first estimate says the cloud
  fills it; `sample_edge` / `height_in_edges`) and `batch_scatter_ppm`. Admission, the
  trace reconciliation of the merged ledger and the stamp all read the batch through
  it, so a peak's occurrence, its trace and its stamp are one object at one rule.
- `admission.derive_height_cutoff_x_edge` / `persistent_trace_count`,
  `timeseries.recentre_ledger` / `collapse_trace_labels` / `stamp_tolerance`, and
  `scripts/eval_trace_stamping.py` (score the trace stamp against the anchor stamp on a
  finished run, per ion and against a list of ions known to be present — the benchmark
  rule for any stamping change is coverage of KNOWN ions, never how many rows were
  admitted).
- `tests/test_traces.py`: a TOF-like batch whose ledger anchor is snapped to theory
  while its trace lives 6.5 ppm away (coverage rises from a minority to > 80 % of
  spectra end to end), a phantom competitor on the same trace losing to the multi-file
  label despite its higher `ion_score`, the low-evidence guard, an Orbitrap-like batch
  where nothing moves, and the derived floor staying at 1× where the picker stops at
  the noise edge and rising where it picks into the noise.

- **`peaky publish-batch <run_dir>`** - publish a `peaky batch` run's merged ledger
  onto Mascope's batch ledger as a batch run of its own (Mascope's
  `POST /api/batch-peaks/batch/{id}/runs/import`). Each merged m/z lands on the
  nearest batch peak and the server measures peaky's formula against every sample
  that holds the peak, so the batch ledger shows Mascope's fit under the `peaky`
  name; adducts are resolved to mechanism ids by default (a row without one lands
  nothing), ion formulas come from the per-file ledgers or are derived, and
  `--dry-run` shows the payload. `docs/PUBLISH.md` gained a section.
- **¹⁵N-labelled ammonium reagent profile `NH4_15N`** (`^NH4+` ionisation mode, server mechanism
  `+^NH4+`; aliases `15nh4`, `^nh4+`, `ammonium-15n`, …) with the `[M+^NH4]+` adduct
  (+19.0309, `chemistry.ADDUCT_SHIFTS`), both mechanism maps, the `ammonium-15n` context
  (every context alias names the label — there is no bare `ammonium` / `nh4` / `nh4-cims`
  alias that would hand an unlabelled-ammonium user the ¹⁵N channels; an unknown context
  raises), an `ammonium15N` reagent-cluster library (`[(^NH3)n+H]+` + hydrates + the ¹⁴N
  monomer), `[M+^NH4]+` on the siloxane/PDMS/phthalate/PEG families, and the relabel-only
  dehydration aliases `[M+^NH4-H2O]+` / `[M+H-H2O]+` in `_DIFF_TO_ADDUCT`. Built from the
  2026-09-10 exploratory file: ¹⁴N/¹⁵N adduct pairs 0.018–0.021 → purity 0.98, reagent
  ions below the 40 Da window → TIC normaliser, `label_isotope=None`. `assign.run` drops
  `[M+NH4]+` from the opportunistic channels on a labelled-ammonium run (it would only
  re-claim the 2 % ¹⁴N satellites). New post-tier stage `nh4_dehydration`
  (`cleanup.relabel_ammonium_dehydration`): the MS2-proven declustering cascade
  `[M+^NH4]+ → [M+H]+ → [M+H-H2O]+` / `[M+^NH4-H2O]+` re-reads the alkene/enone
  readings of X−H2O onto the corroborated hydrate X (own-adduct-weakness gate — the
  parent-relative fallback applies only to an alkene with no protonated form of its own —
  second loss annotated only). `docs/REAGENTS.md` callout, `docs/ASSIGNMENT_DETAIL.md` §3.7b,
  `tests/test_nh4_15n.py`.
- **Labelled-reagent ¹⁴N line in the envelope predictor** (`isotopes._per_atom("^N")`, at
  the active reagent purity, default `LABEL_PURITY_15N` 0.98): a `^N` ion now predicts its
  −0.997 Da `14N` satellite, so
  `complete_isotope_envelopes` claims it — and displaces a pass-1 CHON `[M+H]+` mass-fit
  sitting on it when it matches the predicted 2 % of a ≥10× brighter labelled parent
  (pass-0 locks kept). Before, only 7 of the ~340 `[M+^NH4]+` satellites were attached.
- **Pass-4 iso-pairs respect the context halogen caps** (`residual.stage_a_iso_pairs`): a
  ~1.998-Da doublet in a halogen-free positive run (max_Br = max_Cl = 0) is no longer
  read as a Br/Cl pair (7 `C5H7BrO3 [M+^NH4]+` phantoms on the ammonium file).
- **Positive pass-0 known species**: `cyclosiloxane` (D3–D7, L2–L5; gate ≥2 channels OR a
  confirmed ²⁹Si/³⁰Si envelope + the Si-count M+1 check) and `indoor_sulfur`
  (benzothiazoles, dithiocarbamate ester, thiazoles, DMSO/DMSO₂, DMDS/DMTS, thiophenes,
  sulfolane, NBBS; gate ≥2 channels OR a confirmed ³⁴S envelope). Passes 1/2 are CHO(N)-only
  and no positive pass-3 family opens S, so benzothiazole `[M+H]+` (68 kcps, MS2: −HCN →
  C6H5S⁺) was unexplained and D4 only a low siloxane-ladder Candidate.
- Labelled-ammonium runs score NO opportunistic channel: `[M+Na]+` sits 0.2 mDa from
  `[(X−O2+C2H4)+^NH4]+` (Na − ^NH4 = 3.9584 Da vs C2H4 − O2 = 3.9585 Da), so every ^NH4
  adduct of an O≥2 neutral had a hydrocarbon·Na twin the complexity prior preferred
  (114 Na fits; palmitic acid read as C18H36·Na⁺). Pass-3 family adduct lists drop
  `[M+NH4]+`/`[M+Na]+` on such runs too. The `ammonium-15n` context admits H/C up to 3.0
  (C3–C4 amines). The `ammonium15N` library adds the urea crossover and the reagent-derived
  CO/CO₂ clusters ((¹⁵NH₃)₂H⁺·CO at 65.049 = the doubly-labelled "formamide" ion) and the
  reagent-made ¹⁵N-acetamide ions (61.041 protonated, 79.065 as its ¹⁵NH₄⁺ adduct: the
  79/61 ratio 0.18 equals the ambient ¹⁴N acetamide's 78.068/60.044 = 0.21, so 61.041 is
  the amide, not ambient ketene·¹⁵NH₄⁺ of the same composition).
- `relabel_ammonium_dehydration` brightness ceiling: a dehydration product may not exceed
  1.5× its parent's strongest form (adduct or protonated) — a 2 kcps C3H8O2 parent no
  longer claims 9 kcps acetone `[M+H]+` as its water loss (ambiguity note instead).
- Pass-3 family `organosulfur` (S 1–2 on `[M+H]+`/`[M+^NH4]+`/`[M+NH4]+`/urea; opened by the
  `ammonium-15n` context, ³⁴S-gated by the tier engine): reaches the C5H12N2S / C9H18N2S /
  C11H22N2S `[M+H]+` thioureas (each with a 4 % ³⁴S line) that no other positive pass can;
  tetramethylthiourea (C5H12N2S) added to the `indoor_sulfur` known list.
- **Mass-dependent calibration centre** (`peaky/assignment/masscal.py`, new): the pass-1
  self-calibration and the tier gate now also fit `ppm = a + b·1000/mz` (b = the constant
  absolute offset in mDa) on the backbone; `z_of(ppm, cfg, mz=)` and `tiers._cal_z` judge a
  peak against the centre at ITS m/z, clamped to the backbone's own m/z coverage
  (`cal_mz_lo` / `cal_mz_hi`), so the trend is never extrapolated onto masses the backbone
  never saw. On the 2026-09-10 file the Orbitrap's low-mass residual was −0.12 mDa (−2 ppm
  at m/z 61 vs −0.2 ppm above 160, MAD 0.13–0.22 ppm), so a constant centre rejects every
  bright sub-80 ion (ketene·¹⁵NH₄⁺, urea·H⁺, acetamide, acetic acid, the amines) at z = 6.
  Callers without an m/z keep the constant model unchanged; a flat source keeps it exactly.
  **The fit is accepted only when** `|b| > 3·SE(b)` (`SLOPE_MIN_SE`) **and** the trimmed
  residual RMS is ≤ 0.8× the constant model's on the same points (`TREND_SIGMA_RATIO_MAX`)
  **and** `|b| ≤ 0.5 mDa` (`MAX_ABS_OFFSET_MDA`, a physical cap on the calibration curve's
  residual absolute offset) **and** each half of the fitted `1000/mz` range holds ≥ 5 kept
  points (`MIN_SIDE_N`, the lever guard): a 2-SE rule would be a 5 % two-sided test, i.e. a
  phantom trend on a FLAT source in ~6–7 % of samples at any n (Monte Carlo, 21–300 rows),
  and a backbone whose only low-`1000/mz` point is one corroborated outlier is a lever, not
  a trend. `masscal.centre` / `sigma_at` are the single implementation behind
  `passes.core.cal_center` / `cal_sigma_at` / `z_of` and `tiers._cal_z`.
  The fitted trend is run-derived, so `provenance`'s reproducible config fingerprint
  excludes `cal_a` / `cal_b` / `cal_sigma_trend` / `cal_mz_lo` / `cal_mz_hi`: `calibrate`
  writes them onto the shared `cfg`, so without that the last sample's numbers land in
  `run_manifest.json` and two identical re-runs of a batch fingerprint differently.
  `cal_mu` / `cal_sigma` are excluded for the same reason, now that the pipeline hands the
  provenance manifest the same `cfg` the per-sample loop fits: leaving them in would keep
  the fingerprint dependent on whichever sample finished last. This is a manifest schema
  change — manifests written before it carry the two fields as a record of the run's
  calibration centre, and `batch_summary.json` still reports the per-file offsets.
- `cal_abs_floor_mda` (0.03 mDa, `PassConfig`, default `masscal.ABS_FLOOR_MDA`): an absolute
  floor on the trend sigma, active only below ~m/z 120, where the Orbitrap's residual
  curves faster than 1/mz (dimethylamine `[M+H]+` at 46.065 sat 0.9 ppm = 0.04 mDa off the
  fitted trend). `confidence_label(..., mz=)` / `cal_center` grade against the trend centre
  too, so an on-trend −2 ppm sub-80 ion is Good, not Low. `PassConfig.cal_abs_floor_mda` is
  the single owner of the floor: `apply_tiers` / `compute_tiers` take the run's `cfg` and
  carry the value onto `tiers._Cal`, so overriding it moves the commit gates and the tier
  gate together.
- Reference peaklist `isoprene_ox_wennberg2018` (27 closed-shell isoprene oxidation products, Wennberg et al. 2018) added to `peaky/data/peaklists/`, gated by the new `isoprene_ox` context (batch keywords isoprene/ISOPN/IEPOX/ISOPOOH/methacrolein) and `biogenic_soa`/`ambient_summer`; rescues the isoprene dihydroxy-dinitrate C5H10N2O8 as an isotope-confirmed Assigned in the 2026 field-campaign ¹⁵NO₃⁻ data.
- **A run records which reference lists it had active** — `reflists.active_versions`
  puts `reflists_active: [(id, data_version), ...]` in the single-sample manifest
  (beside the selection prior it builds from the same lists) and in
  `batch_summary.json`. A list's closed-shell/radical split changes with its
  `data_version`, so a ledger now says which revision shaped its prior and its
  rescue instead of silently depending on the installed copy.

- **A reagent profile can carry its own height-gate multiple**
  (`ReagentProfile.height_cutoff_x_edge`, `docs/REAGENTS.md` §3a). The gate is a
  multiple of the sample's own noise edge, and the right multiple belongs to the
  **peak picker**, not to the reagent: a picker that stops at the noise edge
  wants 1.0 (rare real ions sit at 1–3× the edge there, so raising it discards
  them), while a picker that picks *into* the noise admits nearly everything it
  found at 1.0 — on one 230-spectrum time-of-flight batch, 1.0 merged 4346 ions
  of which 3307 were seen in a single file only, where 5.0 kept 57 % of the
  picked peaks and 74 % of the assigned ones, i.e. a tighter candidate list for
  the height-gated passes. So the global default stays **1.0** (now the single
  constant `passes.config.DEFAULT_HEIGHT_CUTOFF_X_EDGE`, read by both
  `PassConfig.height_cutoff_x_edge_resolved` and the fallback), **no bundled
  profile sets the field** (pinned by a test), and behaviour out of the box is
  unchanged. A site
  raises it for its own instrument from a `--reagent-config` file —
  `height_cutoff_x_edge` is now a loadable reagent-config field — with no code
  change. `profiles.resolve_height_cutoff_x_edge` implements the order once —
  an explicit `--height-cutoff-x-edge` / cfg value > the profile's value > the
  package default — and every entry point that resolves a profile and builds a
  `PassConfig` applies it (`peaky assign`, `assign.main`, `assign_batch.run`,
  `pipeline.run_batch` / `run_pooled_batches`, the MCP `assign_sample`,
  `scripts/certify_neutrals.py`). `pipeline.run` builds no `PassConfig` of its
  own; it resolves the profile and *reports* the multiple the assign stage will
  use as `height_cutoff_x_edge` in its return value.
  `peaky assign --height-cutoff-x-edge` and `PassConfig.height_cutoff_x_edge`
  both default to *unset* (`None`) rather than to 1.0, so explicitness is read
  off the value instead of guessed by comparing it to the default: an explicit
  multiple that happens to **equal** the global default still outranks a profile
  that carries a higher one. The resolved multiple is
  recorded: `batch_summary.json` gains `height_cutoff_x_edge` +
  `height_cutoff_x_edge_source` (and each file's `height_cutoff_x_edge` beside
  its `height_gate_cps`), and it stays in the `run_manifest.json['config']`
  fingerprint, which one log line per run mirrors.
- **Admission gate: persistence OR brightness** (`assignment/admission.py`).
  A peak is eligible for formula search at the eight gated sites (pass-1 grid,
  pass-2/3 target peaks, ladder gap-fill, residual stage B, siloxane ladder —
  the full list and its derivation are under *Changed*) if `height >=
  height_cutoff OR occurrence >= threshold`, where occurrence is the fraction
  of the batch's spectra whose m/z bin holds a peak (bins = `timeseries.
  build_matrix` at `sampling.BATCH_TOL_PPM`; computed once per batch from the
  time series) and the threshold is **derived from the batch**
  (`--occurrence-min auto`: Otsu's split of the bimodal bin-occurrence
  distribution, clamped to 0.25–0.75, 0.40–0.55 on every instrument measured;
  a fixed 0.8 was tried first and discarded two-thirds of the recurring weak
  ions; a number overrides, `0` disables, < 10 spectra switch it off). The
  resolver's decision is FINAL: when it switches the path off, the gate stays
  off, so a numeric `--occurrence-min` cannot re-open a path the spectrum count
  (or an occurrence distribution with no split) closed. That is what makes an
  empty `admitted_by` mean exactly "not eligible at the gated sites".
  Noise does not recur at a fixed m/z; ions do: on a 230-spectrum mixed-reagent
  TOF batch the old absolute cutoff kept 32 of 4025 bins while ~500 recur in
  > 80 % of spectra at a median 3–4 cps, and 21 highly oxygenated molecules an
  Orbitrap saw on the same air all sit in that recurrent population at ~2 cps
  (mass-shifted decoys: none). The path is additive and admits peaks for
  consideration only — confirmation rules are unchanged, and **persistence
  gates entry while only corroboration gates the tier**: an occurrence-admitted
  M0 with no isotopologue / cross-channel / series corroboration is capped at
  Candidate (`tier_reason` `persistent-weak`), because at that intensity
  nothing constrains which formula the real ion got. Per-file ledgers record
  `occurrence` (float in [0, 1], NaN without batch context) and `admitted_by`
  (`height` / `occurrence` / `''` = not eligible at the gated sites), the
  merged ledger carries the winner's, `batch_summary.json` gains an `admission`
  block (`occurrence_min` = the knob, `occurrence_threshold` = the resolved
  fraction, `n_bins`, `n_persistent_bins`, `n_spectra`, `tol_ppm`) and per-file
  `admitted` counts, `run_manifest.json` records that block under `counts` and
  the module version under `code.module_versions`, a run whose persistence path
  is off logs which of the four reasons applies (`admission.why_off`),
  and the report cover states how many merged peaks were eligible by
  persistence only (with the resolved threshold and the knob).
  `PassConfig.occurrence_min`; `assign.run(occurrence=)`; `peaky assign
  --ts-batch` computes the table for a single sample. `batch` / `pool` also
  gained the absolute `--height-cutoff` override and `--height-cutoff-x-edge`,
  which only `assign` had (the three flags are defined once for all three).
- **`--progress`: a live progress window for a run** (`peaky/progress.py`, new).
  `peaky assign|batch|pool --progress` opens a small Tk window with a samples bar,
  a within-sample stage bar, elapsed + ETA, and — when the run ends — the run's
  own stats (merged M0, tiers, in-all-files, single-file, formula disagreements)
  next to how long it took. Opt-in, so scripted and skill-driven runs are
  untouched; `PEAKY_PROGRESS=1` also enables it. Everything in the window is also
  on stdout, so nothing is lost by never seeing it.

  **`peaky assign` is a one-sample run and reports as one.** Its samples bar
  reaches 1/1 and its stage bar fills when `assign.run` returns: the `(i/N) done`
  line that drives both comes from `assign_batch`, which that path never goes
  through, so the command marks its own sample complete. It shows **no ETA** —
  an ETA is extrapolated over completed samples, and the only sample here
  completes when the run does; the window omits the field rather than printing a
  clock that cannot move.

  **The hold is for a person at a terminal, and it is bounded.** A finished run
  keeps its window up so the stats panel can be READ — but only when stdin and
  stdout are both a tty, and only until you close the window, press Ctrl-C, or
  `PEAKY_PROGRESS_HOLD_S` seconds pass (default 600; `0` disables the hold
  entirely). A pipe, a CI job or a skill-driven run therefore never waits on a
  window, whatever `PEAKY_PROGRESS` says, and Ctrl-C during a run closes the
  window immediately, with no stats panel and no wait.

  It is a **`log` wrapper, not a pipeline change**: peaky already threads
  `log=print` from `run_batch` down to each assignment stage, so `Reporter` is a
  drop-in for `print` that forwards every line untouched and reads the lines it
  recognises into a progress model. Nothing in the pipeline imports `progress.py`
  or knows a window exists — the log stream is the whole interface, and
  `tests/test_progress.py` pins the literal log strings the pipeline emits
  against the patterns parsed here so a rewording fails a test instead of
  silently flat-lining the bar. Never fatal: no display, no tkinter, or any UI
  exception degrades to a one-line terminal status and then to silence.
  **macOS always takes that fallback**: the window runs on a daemon thread, and
  Tk/Cocoa driven off the process's main thread aborts the process outright —
  not an exception any guard could catch — so on Darwin the progress window
  would kill the run it reports on. There is no window there, by design.

  Parallel runs (`--jobs > 1`) report at **sample granularity only** and say so
  ("N workers" in place of the stage bar): workers buffer their logs and the
  parent replays them after the reduce, so a stage bar driven from them would
  animate a lie.
- **Runs are timed.** `batch_summary.json` gains `elapsed_s` (+ the `n_jobs` that
  produced it — a duration is meaningless without it), and `run_batch` /
  `run_pooled_batches` return a whole-pipeline `elapsed_s` and log it. Run-time
  metadata only: the reproducibility fingerprint hashes `merged_ledger.csv` and
  the input TS, not the summary, so determinism is unaffected.
- **`[phase] <name>` log markers** for the pipeline steps with no per-item
  progress of their own (fetch / select / assign / cluster / vankrevelen /
  report / provenance) — readable in a plain log, and what the window's phase
  line reads. `run_batch` and `run_pooled_batches` emit the same set, so the
  phase line is as truthful on a pooled run as on a single-batch one; `select`
  comes from whichever side runs the set-cover (the pooled pipeline itself, or
  `assign_batch.run` when it picks the cover for one batch, which brackets it
  back to `assign` as soon as the picks are in).

## [0.7.0] - 2026-09-03 (publish a peaky run into Mascope)

### Added

- **Contributor License Agreement** — external contributors now accept the Ultra
  Trace Systems Individual Contributor License Agreement once, on their first
  pull request, by replying to the CLA assistant's comment
  (`.github/workflows/cla.yml`); the `CLA Assistant` check blocks merging until
  every commit author has. The agreement and the signature register are shared
  with Mascope in [ultra-trace-systems/cla](https://github.com/ultra-trace-systems/cla),
  so one acceptance covers both projects. New `CONTRIBUTING.md` explains it.
- **`peaky publish` — upload a finished ledger into Mascope's peak-assignment run
  ledger** (`io/publish.py`, `docs/PUBLISH.md`). This closes the loop peaky was
  missing: a run stops being a local file and becomes a run in the same store the
  in-app engine writes to, visible in the run selector, the peak inspector, the
  batch Assignments overview and the verification loop — which is what makes the
  two engines comparable on one sample. Implements the client half of Mascope's
  run-import contract: chunked assembly sized by serialized bytes as well as row
  count, a row-offset `chunk.index` and an `import_id` so a retried chunk is a
  replay rather than a duplicate, and resume via `--import-id`.

  Three parts of the translation are **not** field-for-field, and each is a place
  a naive mapping is silently wrong:
  - **Two tiers, and neither is a copy of the other.** peaky tiers mechanically
    (window uniqueness, corroboration, degeneracy, O-count); Mascope tiers by
    thresholding a row's _evidence_ — fit weighted by the formula's chemical
    plausibility — against the run's declared bands. The published row carries
    both: `engine_tier` is peaky's verdict, on committed M0 rows only (null
    elsewhere, and absence is not agreement), while `tier` is **not sent at
    all** — it is a pure function of inputs the server already holds, so the
    server derives it and the two implementations cannot drift apart at a band
    edge. On a real ledger this preserved 195 disagreements that were
    previously flattened away. `--dry-run` reports both distributions and the
    disagreement count.
  - **The intensity is instrument-determined**: heights for Orbitrap, areas for
    TOF. It scales the sample's consensus vote in the batch view while the unit
    label is derived separately, so the wrong one mis-weights the sample with
    nothing able to detect it. `--intensity auto` reads the instrument and
    refuses to guess when it cannot classify it.
  - **Vocabularies differ where the shared lineage suggests they would not**:
    peaky's `unexplained` role is Mascope's `unassigned`, and peaky's capitalized
    tiers are not the server's spellings.

  Synthetic de-blending sub-peaks are excluded (they exist in no Mascope peak
  file), an `iso_child` publishes its owner's formula and `parent_peak_id` as the
  owner reference, non-finite floats become nulls, and the reserved provenance
  keys the server strips are dropped locally so the summary says so. Verified
  end to end against a Mascope dev server: 1319 rows over two chunked requests,
  every owner link resolved, the server's derived evidence written on exactly the
  scored rows, no reserved key stored, and the batch fold-in run.

  A **live contract tripwire** (`tests/test_publish_contract.py`, opt-in behind
  `MASCOPE_LIVE=1` like the existing live smoke, which CI asserts is unset) runs
  the real protocol against a real server and reads every contract point back:
  the server-derived tier, the engine tier, the resolved mechanism, the owner
  link, the alternatives shape, the server's own evidence and plausibility, and
  the absence of the reserved keys. The offline suite pins what peaky sends;
  only this catches the server's contract moving underneath a client that owns
  a private copy of it.

  Cosmetic gap, reported by the command itself: `--dry-run`'s preview of the
  tiers Mascope will derive needs
  `mascope_tools.composition.heuristic_filter.formula_plausibility`, which is not
  in the 2026.6.25 release the dependency resolves to. Without it the preview
  assumes a plausibility of 1.0 and reads high for a formula the server weighs
  down. Nothing is at risk — the tier is derived server-side and the preview is
  not published. Upgrade `mascope-tools` once a release exports it.

## [0.6.0] — 2026-08-12 (report refactor + modern-server-only I/O)

### Removed (legacy workspace-server support)

- **The legacy (workspace-based) server code paths are gone** (`io/io_mascope.py`):
  `_patch_datasets_list_for_legacy_servers` (no callers), the `_legacy_*` raw-endpoint
  helpers, `resolve_batch_id`, `_legacy_load_batch_peaks`, and the silent fallback arms
  in `list_datasets` / `list_batches` / `fetch_batch_samples` / `fetch_batch_peaks`.
  Every current server is datasets-based; the fallbacks actively harmed debugging —
  each modern-path failure was swallowed by a bare `except Exception` and re-surfaced
  as the legacy path's own unrelated error (a bare `/api/sample/batches` GET that
  modern servers reject with 422), masking the real cause. The helpers now make one
  modern call, raise a clear error on an empty result, and let SDK errors propagate
  unmasked (regression-tested). Committed run outputs (root-level report PDFs, figure
  PNGs, class time-series CSVs) were also removed from the repository — runs belong in
  the git-ignored `output/`.

### Changed

- **Brand**: Karsa Oy → Ultra Trace Systems Oy (NOTICE, CITATION.cff, repository URLs).
- **A real-SDK contract tripwire** in `tests/test_io_mascope.py` runs peaky's filter
  helpers through the installed SDK's actual matching code offline, so an SDK contract
  change fails CI the day it lands. The suite now also passes on Windows (tempdir-cwd
  teardown, hardcoded `/tmp` paths).

### Fixed (batch-name resolution vs the current mascope-sdk)

- **`peaky batch` matched no batch on current SDKs and died on a legacy-endpoint
  422** (`io/io_mascope.py`). SDKs through 2026.7.7 resolved their two batch
  filters with OPPOSITE conventions — `load_peaks(batches=)` escaped a plain
  string itself (literal substring; only a compiled pattern was a regex), while
  `samples.list(batch=)` used a plain string as a RAW regex and crashed on
  compiled patterns — so the previous one-size `re.escape()`d string was escaped
  twice by `load_peaks`, silently matched nothing, and every batch run fell
  through to the legacy `/api/sample/batches` fallback, which modern servers
  reject (dataset_id required). **mascope-sdk 2026.8.12 unified the contract**
  (a plain string is a case-insensitive literal substring on every name filter;
  only a compiled `re.Pattern` is a regex), so batch names are now passed RAW —
  the interim `escape_batch()` / `literal_batch_pattern()` split introduced on
  this branch is gone again; `fetch_pooled_peaks` still compiles its user regex
  un-escaped. The contract is pinned by a real-SDK offline tripwire (which also
  fails on the SDK's deprecation shim), and `mascope-sdk>=2026.8.12` is the
  floor — keep the SDK at the latest release (SKILL.md gotcha).

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
  channels look _identical_ to real ions near a bright peak — satellite fraction
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
  > 1 peak — the shoulder is left `unexplained`), so this only ever affected the
  > parquet. Two rules, both default-on and individually switchable
  > (`one_to_one=` / `consensus=`):
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
  those are N independent mass constraints on one unknown — a _certificate_ that licenses
  enumerating the expanded element box (P/S/Cl, past the per-peak caps) for that mass
  only, oracle-scored, isotope-gated (³⁴S/³⁷Cl/⁸¹Br; ¹³C never), committed onto every
  member peak under its own channel label. The pass-5 inverse: cross-channel evidence
  _licenses_ new formula space instead of _completing_ known formulas — so off-grid
  families (organophosphate pesticides, sulfonamide plasticizers) are discoverable
  generically, with no whitelist. Also interrogates weak M0 incumbents: a strong
  certificate (iso-confirmed or ≥3 channels) displaces a bogus single-channel fit (e.g.
  an unsupported `[M+Na]+`) via `clear_assignment`, audit-trailed. Reagent-free primary
  path; optional `ts_peaks` co-variation corroboration. Validated on the NBBS urea ladder
  (→ C₁₀H₁₅NO₂S) and cross-channel malathion (C₁₀H₁₉O₆PS₂); first real-ledger run
  blind-rediscovered benzothiazole (C₇H₅NS).
- **Organothiophosphate pesticide family** in positive pass-0 (`_known_species`): malathion
  - homologs + des-ethyl TP + ~14 common OP-thioate insecticides. P is off the grid and S
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
  mass-saturated — _not_ niso-gated, since a ¹³C confirms carbon count, not the O count)
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
    commits the highest-`eff_score` candidate _before_ the mass calibration is
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

## [0.5.0] — 2026-06-30 (reference peaklists + chemical-plausibility hardening)

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
  - `tables/reflist_matches_*.csv`. Lists are context-gated by run metadata
    (contaminants always active).
- `docs/ASSIGNMENT_DETAIL.md` — exhaustive per-pass / per-gate pipeline reference.

### Changed (chemical-plausibility hardening)

- **Reagent-halocarbon relabel** — bromomethane reagent fragments mis-read as a bare
  element + reagent-cluster (e.g. CHBr₂⁻ as "C" via `[M+HBr+Br]-`) are reclassified
  on the invariant ion composition (CH₂Br₂→reagent, dibromoacetic acid→named).
- **Confirmed-isotope F-demote exemption** — a high-F formula is exempted only when a
  Cl/Br/S anchor's diagnostic isotope (³⁴S/³⁷Cl/⁸¹Br) is _confirmed_, not merely in
  the formula (a reagent-Br adduct's ⁸¹Br does not count).
- **Si-count intensity gate** (siloxane ladder **and** pass-0 silanediol) — a Si-rich
  commit requires its ²⁹Si M+1 to _match_ the Si count, not just be matched; stops a
  high-O HOM (e.g. C₁₀H₁₈O₁₁) being claimed as a siloxane on a too-weak envelope.
- **New tier demotes** (post-tiering, never deletes): carbon-cluster (F-free H/C<0.35),
  implausible-ionization (heteroatom-free hydrocarbon via an anion channel that needs
  an acidic/H-bond site), and speculative-residual (residual:\* commits resting on
  off-cal z, uncorroborated multi-N, 0-anchor series, or a sole minor channel).
- **Scrutiny page** — F-flag wording corrected (¹⁹F is monoisotopic — the F _count_ is
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
  mode). Bins all batch peaks by m/z and assigns each significant bin's _brightest_
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
