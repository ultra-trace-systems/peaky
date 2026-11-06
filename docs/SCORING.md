# Peaky — Scoring (in-process IsoSpec + `score_pattern_v2`)

This document explains **how a candidate neutral formula gets a match score** —
the in-process backend that replaced the network `match_compounds` round-trip.
It is a module deep-dive companion to [`ARCHITECTURE.md`](ARCHITECTURE.md) (the
whole pipeline), [`DATA_IO.md`](DATA_IO.md) (which dispatches here from
`score_candidates`), and [`ASSIGNMENT.md`](ASSIGNMENT.md) (which consumes the
scores). The cardinal rule still holds: **Mascope is the scorer.** This module
runs Mascope's *own* released maths (`mascope_tools.composition`) locally; Peaky
never invents a mass or isotope score.

**Code:** `peaky/io/local_scoring.py` (`score_candidates_local`, the engine);
the dispatch + the `PEAKY_LOCAL_SCORING` switch live in
`peaky/io/io_mascope.py` (`score_candidates` / `_score_candidates_local`).

> Keep this in sync with the code. Every threshold below is a named constant in
> `local_scoring.py`; if you change one there, change it here.

---

## 1. What this stage does

Given a sample's peaks and a list of candidate **neutral** formulas, score each
`(neutral × adduct)` ion against the spectrum and return one row per predicted
isotopologue — the **same schema** `io_mascope.flatten_match_tree` produces, so
the passes downstream don't know or care which backend ran.

The motivation is structural: the backend `match_compounds` is a *deep-annotation*
primitive — per `(formula × adduct)` it computes the full theoretical envelope and
returns the whole matched-**and-unmatched** tree, `O(candidates × adducts ×
envelope)` work and tens of thousands of rows for an `O(matches)` signal. That
drove the timeouts and OOM. The local path computes the identical envelope with
**IsoSpec** (`predict_isotopes`), scores it with Mascope's own
**`score_pattern_v2`** — a Gaussian mass likelihood at the sample's fitted width
times an intensity likelihood at the peak's own signal-to-noise, charging a
predicted line that is absent where the noise says it should have been visible —
and emits **only matched isotopologues** — no network, no 30k-row trees.

What a sample is judged at arrives as a **`PatternScoring`**
(`io_mascope.scoring_for_sample`): the width and offset fitted from the sample's
own anchors, and the line-matching window of its instrument class. Its
predecessor `score_pattern` scaled the mass term by a fixed 5 ppm and averaged
its terms over the lines a candidate *matched*, so an envelope that predicted
three lines and found one cost nothing — and on a TOF, whose ordinary error is a
whole Orbitrap window, every candidate scored near zero and a reference run
committed almost nothing.

```
sample peaks (mz, height, peak_id)        candidate neutral formulas
        │  dedup on peak_id, sort by mz            │
        └──────────────┬───────────────────────────┘
                       ▼   for each (neutral × adduct)
        combine_formula_and_ionization → ion formula
                       ▼
        predict_isotopes (IsoSpec) → (pred_mz, pred_int, labels)
                       ▼   anchor_on_monoisotopic: the ion's own line first
                       ▼   pred_rel = pred_int / pred_int[0]
        match M0 in ±scoring.mz_tolerance_ppm  ── not found ──► drop candidate
                       │ found
                       ▼   match each isotope; keep if intensity-error ≤ tol
        score_pattern_v2(obs_ppm − mu, obs_int, obs_snr, pred_rel, sigma_ppm)
                       ▼   one score per ion, copied onto every iso row
        flat per-isotopologue rows  (matched: peak id + ppm; else None)
```

---

## 2. Inputs

- **`peaks`** — the sample's raw peaks. `mz` / `height` / `peak_id` are read, and
  `signal_to_noise` where the server sends it; rows are **deduped on `peak_id`**
  (raw server peaks repeat per match) and sorted by m/z so the window search can
  use `np.searchsorted`. Without the noise column the score runs in its no-SNR
  mode and charges an absent line on predicted abundance alone, which is a
  different reading of every faint line — so `fetch_peaks` caches to a versioned
  file rather than reusing a frame from before the column existed. **The column is
  judged before it is believed** (`local_scoring.assess_snr`): a signal-to-noise
  tracks height, and on one TOF the server's column does not (Spearman −0.07..0.17 over
  a file's peaks against 0.99–1.00 on every Orbitrap file; a 478-count peak carried 1.1, a
  2-count peak 12), which read as an SNR excuses every missing line of a bright ion and
  charges dim ions for lines they could never show. `io_mascope.scoring_for_sample`
  records the verdict in the scoring snapshot (`snr_source`: `server` — the column,
  `none` — no column, the no-SNR mode, `poisson_fallback` — Spearman under 0.5 over at
  least 30 peaks; with `snr_spearman`, `snr_n`, `snr_edge`), and `peaks_for_scoring`
  then hands the scorer a copy whose column is the counting-statistics SNR of an
  ion-counting detector, `h / sqrt(h + edge²)` (`poisson_snr`; `edge` = the file's
  1st-percentile picked height, `passes.config.noise_edge`). A TOF file's heights are
  per-file averages, so the true ion counts are higher and this is a conservative lower
  bound on the SNR: a matched line's ratio tolerance is at least its Poisson scatter, and an
  absent line is charged only where even the bound says it was within reach (an absent 81Br
  line from ~10 cps at a 0.74 cps edge; a 13C line only on bright ions). A stand-in sample
  judges its own table. The network scorer (`PEAKY_LOCAL_SCORING=0`) reads the server's
  column regardless.
- **`scoring`** — a `mascope_tools.composition.PatternScoring`, built per sample
  by `io_mascope.scoring_for_sample`: `sigma_ppm` and `mu_ppm` from
  `fit_mass_accuracy` over the sample's own targeted matches (falling back to the
  instrument class's `resolve_fallback_sigma_ppm` below eight anchors), and
  `mz_tolerance_ppm` from `resolve_match_tolerance_ppm` — 5 ppm Orbitrap, 15 TOF.
  Passing nothing scores at the library's defaults, which are an Orbitrap's.

  **The width and the offset have different bars.** The library reports both or
  neither, and below its minimum it returns `(0.0, None)` — a zero offset that
  reads exactly like a measured one. A robust spread needs a distribution; a
  median needs a few points that agree. So the offset is taken from the anchors
  whenever there are at least `MIN_OFFSET_ANCHORS` (3) of them, even where the
  width falls back to the class, and `pattern_scoring` records `sigma_source` and
  `mu_source` separately. This is not cosmetic: on a nitrate source sitting
  1.2 ppm low with six anchors, scoring at `mu = 0` charged that error to every
  candidate and handed the peak to whichever reading's own error cancelled it —
  the reference committed a third of what it had, and disagreed where it did.

  The minimum is five rather than three because an anchor is a peak the server
  matched within the instrument's *matching* window — 15 ppm on a TOF, five times
  its accuracy — so a mis-matched anchor sits in the set looking like a
  measurement. On a bromide TOF sample with three anchors, two mis-matched to the
  same wrong species, the median landed at −10.4 ppm for a source the engine
  measures within 0.3 ppm of calibration.

  **Below five anchors the centre has two more sources before zero.** `mu_source`
  reads, in order: `fitted` (the sample's own anchors); `axis` (none, but the
  batch's m/z-axis correction reached the sample — `peaky batch --mass-axis`, its
  peaks moved onto the model's axis, so zero is right where the model reaches; the
  snapshot adds `axis_scope_share`, the share of the sample's peaks inside the
  model's scope); `self` (none, and no axis correction: an Orbitrap-class file
  scored locally measured its own centre before pass 0, `assignment/selfcentre.py`;
  the measurement is under `self_centre`); else `assumed_zero`. A run that then
  accepts a mass trend says `trend`; a stand-in says `inherited`. The self-centre
  reads a backbone chosen without the mass term: the file's bright unexplained
  peaks (≥ 10× its noise edge, under 100× its median height, the 150 brightest),
  enumerated as pass 1 enumerates them (the CHON grid of the run's context and its
  channels, ±`search_ppm` around the raw m/z) and judged by the pattern-only score;
  a peak counts where exactly one ion formula's pattern holds there
  (`ion_score_massfree` ≥ `tau_suspect`), and that reading is a CHO/CHON ion with
  C ≥ 1, pattern-Good (≥ `tau_good`) and 13C-confirmed (its 13C line matched in
  ratio within 1.5 ppm of the M0's error). The backbone's median raw error is the
  centre when at least 8 points agree (scaled MAD ≤ 0.6 ppm), their bulk (centre ±
  2 spreads) lies inside the search window and it sits ≥ 0.1 ppm from zero; the
  file is then scored at it from pass 0 on, the isotope-response re-run included,
  until a mass trend its calibration accepts takes over (`trend`), and the pass-0
  known-species gate keeps its own centre (`prior_offset`). The batch's merge does
  not read it: the files are aligned on their pre-calibration offset
  (`per_file.offset_ppm`, `estimate_offset`), null below eight anchors. Zero was
  the only honest reading only while nothing else had been measured: scored there, an
  anchorless file 0.6 ppm off commits almost nothing at the class width, and the
  pass-1 calibration cannot see why — its backbone is what the zero-centred score
  let through. `PassConfig.self_centre = False` scores such a file at zero.

  **Outside the axis model's scope no m/z is moved; near its ends the centre
  carries the model's edge.** The model corrects only where its locks reach;
  below or above them, and in a gap or hole between its segments, every sample's
  peaks keep the server's m/z, an anchored sample's included, because nothing
  measured the axis there. A sample's centres are fitted on the mostly corrected
  peaks -- the scorer's `mu` (the anchors' median, or zero for an `axis` file) and
  calibrate's `cal_mu` -- so a peak the model did not reach was judged against a
  centre read off the other axis (an NO+ batch whose locks stop near m/z 129: its
  C9-C12 hydrocarbons above it read 0.7 ppm off and fell out of Assigned). A
  model the batch applies through the sub-ppm branch (`assign_batch._lock_model`:
  its largest correction and the locks' raw median both under 1 ppm) therefore
  CARRIES the axis at its first lock and at its last into the scoring centre of
  every line up to `assign_batch.EDGE_CARRY_DA` (30 Da) below / above it
  (`AxisModel.edge_carry_record`, `masscal.edge_carry_at`; the line's m/z is read
  on the served axis, as the model's own inverse reads it). Each end's value is
  read off the END LOCKS: the model there plus the median residual of the end
  segment's non-bright locks within `axislock.EDGE_LOCK_DA` (5 Da) of it, at least
  `EDGE_LOCKS_MIN` (3) of them, else the model's own end value -- a spline's end
  knot can leave its own locks (a uronium axis: -0.41 ppm at its first lock, its
  locks there -0.15..-0.25, an independent axis -0.20; read -0.28), and a median of
  three ignores one stray lock (2.5 % of the locks of a field campaign's lock models
  read more than 0.5 ppm off their model):
  * the local scorer judges each MATCHED line at the file's centre plus the carry
    at that line's peak (`score_candidates_local(centre_offset=)`, per line: an M0
    in scope whose 13C line lies outside is judged on both axes as they are);
  * every calibrated centre at a line's m/z adds it: `cal_center` / `z_of` (the
    commit gate, the mass-gate audit, the residual and siloxane gates), the direct
    `cal_mu` readers (the isotope-gated recovery, the speculative-residual demote,
    the labelled rescue, the reference-list rescue), the tier calibration
    (`tiers._calibrate` / `_cal_z`, and `ppm_error_cal`; a batch-pooled calibration
    lent to an uncalibrated file, `calpool.apply_pool`, takes the file's carry too),
    the off-calibration re-arbitration (which reads the tier calibration) and the
    degeneracy window;
  * calibrate (its backbone and its mass-trend rows) and the tier calibration
    read a backbone row outside the scope on the carried scale (its ppm less the
    carry), so the out-of-scope rows they recover measure the same centre as the
    corrected ones; so do the centres fitted on a file's own rows elsewhere: the
    scorer's `mu` of a `fitted` file (an anchor past the edge, `sample_mass_errors
    (carry=)`; `pattern_scoring.axis_edge_carry.anchors` counts them), an
    uncalibrated file's display centre (`stamp_calibrated_ppm`: the Assigned rows'
    median), and the batch-pooled calibration: its core (`calpool.pool_calibration
    (carry=)`, the batch's record) and the file's own centre it lends a window about
    (the Assigned rows' median, `apply_pool`).

  The ledger's m/z and every line in scope are unchanged. Nothing is carried past
  the reach (measured against an independent
  axis: within 30 Da the edge's value is the axis to ~0.1 ppm on the flat NO+
  axes, while a curved labelled-nitrate axis bends 0.2-0.4 ppm away from it
  further out), in a gap or hole (a step there is unseen), or by a model applied
  at 1 ppm or more: those batches, which the reference bar corrected before the
  sub-ppm branch existed, run exactly as before, and their out-of-scope lines are
  still judged against the in-scope centre. A line past the reach of a carried
  model is judged at the file's centre as it stands, and that centre is now fitted
  on the carried scale: it moves by the carried rows' weight in the fit (a few
  hundredths of a ppm on a field campaign's batches), enough to take a line that
  sat on a gate across it (z 1.99 -> 2.01). The record rides on the model
  (`mass_axis.wave.stats.edge_carry`, also `mass_axis.edge_carry`) to every
  worker (each end `{mz, ppm, model_ppm, n_locks}`: the value carried, the model's
  own end value, the locks it was read off), on each file's snapshot
  (`pattern_scoring.axis_edge_carry`, with `share`, the share of the file's peaks it
  reaches, and `anchors`, the anchors read on the carried scale) and, as
  `PassConfig.axis_edge_carry`,
  on the file's run (`per_file.axis_edge_carry`: `reach_da`, `m0_carried`, the
  committed M0 rows it judged); a stand-in registered with that snapshot (a decoy
  arm) inherits it. The network scorer (`PEAKY_LOCAL_SCORING=0`) judges one
  constant offset, so there the carry reaches the gates only; the trace-first
  sample is not under the batch's correction and carries nothing; the evidence
  levels' per-file window (`degeneracy_cal`) stays one constant. Moving the
  out-of-scope peaks instead, by the model's in-scope correction at the file's
  anchors (an earlier build), fixed the NO+ batch and broke a curved one (a
  labelled-nitrate batch whose correction swings from -2.5 ppm at m/z 45 through
  +0.9 at m/z 100 to +0.2 at its edge, m/z 173: the peaks above it moved
  0.25-0.36 ppm off their file's centre), and it moved the gaps too.

  **A sub-ppm model is applied without the locks that disagree with their
  neighbours.** The branch's verdict is the as-built model's (`_lock_model`: the
  apply bar, the predictive bar and the half-the-correction cv test read the model
  `axislock.fit` built); the model then APPLIED is its spline segments refitted
  (`AxisModel.refit`, `assign_batch.LOCK_REFIT_PPM` = 0.5): each segment's
  non-bright locks lose, one at a time and the worst first, the lock furthest
  off a Theil-Sen line through its `axislock.LOCAL_K` (6) nearest locks while it
  sits more than 0.5 ppm off, and the smoothing spline (GCV) is redrawn through the
  rest on the segment's own grid. A smoothing spline drawn through a few
  co-located locks that read 0.6-0.9 ppm off their neighbours -- a lock on a line
  blended with a brighter ion's 13C line, its labelled alias -- bends towards them:
  on a labelled-nitrate batch (cv 0.094 ppm, largest correction 0.90) it rose
  +0.10..+0.15 ppm over m/z 261-269 and corrected a true cluster line by +0.23 ppm,
  which then fell out of Good in all 7 files it was Good in before the branch
  existed; the refit drops 12 of its 297 locks and corrects that line by +0.06.
  Worst first, because judged all at once a cluster of high locks drags its good
  neighbours' lines up with it (two good locks, +0.006 and -0.159 ppm, went that
  way). Line (walk) segments, every domain, the grid and the verdict stay as
  built; a segment that drops nothing, or whose refit would bend faster than
  `SPLINE_MAX_SLOPE`, is the as-built segment, so a model that drops nothing is
  applied byte for byte as built (on a field campaign's five sub-ppm batches: 0,
  0, 0, 1 and 47 locks dropped). The edge carry is read off the KEPT locks. 0.4 ppm
  starves the GCV fit (its smoothing parameter fell 297 -> 47 on that batch);
  0.75 leaves the bump. `mass_axis.refit` (also `mass_axis.wave.stats.refit`)
  records `off_ppm`, `k`, `n_dropped`, `dropped` (`mz`, `ion`, `ppm` and `off_ppm`,
  how far off the refitted model each sat), `max_abs_ppm_built` and `max_abs_ppm`
  (the refitted model's); `mass_axis.max_abs_ppm` stays the verdict's, and
  `tables/mass_axis_locks.csv` keeps every lock the fit found. A model at 1 ppm or
  more is never refitted.

  **An offline sample has neither a record nor anchors.** A table served from
  memory (`register_offline_sample`: a scorecard decoy arm, the trace-first
  synthetic sample, a test) has no server record to name its class and usually
  no server matches, so on its own it is scored at the more forgiving class — a
  TOF's width and window — at an assumed-zero offset. Registered with a
  `scoring` (`register_offline_sample(..., scoring=)`, `assign.run(peaks=,
  scoring=)`) it is judged at that instead: a measured sample's
  `pattern_scoring` snapshot or a `PatternScoring` as it is (`sigma_source` /
  `mu_source` `inherited`, the originals under `inherited`), or an instrument
  class at that class's width. The scorecard's decoy arms inherit their file's
  snapshot; the trace-first sample stays at the class fallback: scored at an
  Orbitrap's width with an assumed-zero offset it lost a third of a batch's
  assignments, mostly because its traces carry no signal-to-noise (adding it
  recovers 135 of the uronium batch's 209 lost) and partly because their offsets are ones the
  class width cannot absorb -- uncorrected outside the wave's range and
  over-corrected inside it on the labelled-nitrate batch.
- **`formulas`** — candidate neutral formulas (the grid + cheminfo union).
- **channels** — either peaky `adducts` (labels like `[M+Br]-`) or already-resolved
  mascope **`mechanisms`** strings (`[M+Br]-`); the dispatcher passes the latter, which
  skips `adduct_to_mech`. Each is parsed once via `utils.parse_ionization`.

---

## 3. The transformation, stage by stage

All thresholds are the named constants from `local_scoring.py` (see §4).

1. **Prepare peaks.** Keep `[mz, height, peak_id]`, drop null m/z,
   `drop_duplicates(peak_id)`, sort by m/z → parallel `mzs` / `ints` / `pids`
   arrays.

2. **Build the ion** (`utils.combine_formula_and_ionization`). neutral + parsed
   ionization → the ion formula string (e.g. `C6H12BrO6-`). `adduct_to_mech`
   (when adducts, not mechanisms, are passed) gives the label its one spelling
   in the standard adduct notation, terms in the library's order:
   `[M+HBr+Br]-` → `[M+Br+HBr]-`, `[M+(CH4N2O)H]+` → `[M+CH4N2O+H]+`.

3. **Predict the envelope** (`predict_isotopes`, IsoSpec). → `pred_mz`,
   `pred_int`, `labels` for the charged ion, then **`anchor_on_monoisotopic`**:
   the ion's own line is moved first, whatever order IsoSpec returned it in.
   Everything downstream reads index 0 as the ion — the intensity the envelope is
   normalized to, the anchor every score term is measured against, `is_base` —
   and for a dibromide the most abundant configuration is the mixed 79/81 line
   two mass units up, which is not the peak the candidate was proposed for.
   Normalize to that line: **`pred_rel = pred_int / pred_int[0]`**, so a brighter
   satellite runs above 1. Empty envelope → skip.

4. **Match the monoisotopic base (i = 0).** Window half-width
   `d = pred_mz · scoring.mz_tolerance_ppm · 1e-6`; `searchsorted` for
   `[mz−d, mz+d]`, take the **closest** peak. **If M0 is not detected, the
   candidate is dropped entirely** (`base_int is None` → not a candidate at all)
   — there is no scoring an isotope envelope with no anchor.

5. **Match each heavier isotope (i > 0).** Same window + closest peak, but a peak
   is **accepted only if its intensity matches**: relative observed
   `rel_obs = ints[k]/base_int`, intensity error
   `ierr = |pred_rel[i] − rel_obs| / pred_rel[i]`, kept iff
   **`ierr ≤ INTENSITY_TOLERANCE` (0.4)** (= mascope_tools
   `ISOTOPE_MATCHING_INTENSITY_TOLERANCE`). A peak in the mass window with the
   wrong abundance is **not** attributed — it stays unmatched.

6. **Score the ion** (`score_pattern_v2`). One score per ion, on
   `(obs_ppm − centre, obs_int, obs_snr, pred_rel)` -- `centre` the sample's
   constant `scoring.mu_ppm`, or, once the sample's own calibration
   accepted a 1/mz mass trend, that trend's centre at each line's own m/z
   (`centre=` a `masscal.MassTrend`, clamped to its m/z coverage) -- at
   `sigma_ppm=scoring.sigma_ppm`: each line contributes a Gaussian mass
   likelihood times an intensity likelihood whose tolerance is set by that
   peak's signal-to-noise, an **absent** line contributes a miss penalty iff the
   noise says it should have been visible (`pred_rel[i]·SNR_base ≥ 3`) and is
   excluded otherwise, and the per-line likelihoods combine as a
   predicted-abundance-weighted geometric mean. Reported `ppm_error` stays the
   raw measured error: the offset belongs to the calibration, not to the row.
   The same `compound_score` / `ion_score` is copied onto every isotopologue row
   of that ion, and so is `ion_score_massfree`: the same score with every
   matched line's error at the centre -- the isotope pattern alone, which the
   calibration picks its trend backbone by.

7. **Categorize** (`_category`). `score ≥ PROBABLE_THRESHOLD (0.8)` → `probable`;
   `≥ POSSIBLE_THRESHOLD (0.4)` → `possible`; else `unlikely`. Bands on the fit's
   scale: v1 put a correct assignment near 0.95 whatever its envelope did, while
   the fit charges a missing line and a mass off the sample's own width, so a
   correct ion with a lone unremarkable peak lands in the 0.5–0.8 range these
   bands split.

8. **Emit rows.** One row per predicted isotopologue. Matched rows carry the
   attributed `sample_peak_id`, `sample_peak_mz/intensity`, and a real
   `ppm_error`; unmatched isotopologues carry `None` for peak id, score, and ppm
   (so the envelope is described but the gaps are honest).

---

## 4. Constants reference

All in `peaky/io/local_scoring.py`.

| constant | value | role |
| --- | --- | --- |
| `PROBABLE_THRESHOLD` | 0.8 | score ≥ → `probable`, on the fit's scale |
| `POSSIBLE_THRESHOLD` | 0.4 | score ≥ → `possible`; below → `unlikely` |
| `INTENSITY_TOLERANCE` | 0.4 | max relative abundance error to attribute an isotope (= `ISOTOPE_MATCHING_INTENSITY_TOLERANCE`) |
| `scoring.mz_tolerance_ppm` | 5 / 15 | half-window for matching a predicted line to a peak, from the instrument class (`resolve_match_tolerance_ppm`) |
| `scoring.sigma_ppm` | fitted | the mass term's width: the sample's own, else its class's `resolve_fallback_sigma_ppm` (0.3 Orbitrap, 3.0 TOF), widened by `PRED_SIGMA_PPM` |
| `MIN_OFFSET_ANCHORS` | 5 | anchors below which no offset is claimed either; above it the median stands even when the width cannot be fitted. Five because an anchor is a match within the instrument's *matching* window, so a mis-match can sit in the set - and two of them cannot carry a median of five |
| `k_detect` / `miss_penalty` | 3.0 / 0.3 | when an absent line is charged, and what it costs (in `mascope_tools`) |

---

## 5. Metrics, defined

- **ion score** — `score_pattern_v2(obs_ppm − mu, obs_int, obs_snr, pred_rel,
  sigma_ppm=…)`: a single 0–1 number, the fit quality of the envelope against
  this sample's own measurement. The whole arbitration downstream ranks on this.
  It is the same number Mascope's assignment engine ranks and tiers on, from the
  same library function, so a comparison between the two engines is a comparison
  of their assignments rather than of their scorers.
- **`pred_rel`** — predicted isotope intensities normalized to the base
  (`pred_int / pred_int[0]`); the reference the observed pattern is judged against.
- **`ierr` (intensity error)** — `|pred_rel − rel_obs| / pred_rel`; the gate that
  decides whether a mass-window peak is *really* this isotope (≤ 0.4) or a
  coincidence.
- **`ppm_error`** — `(obs_mz − pred_mz)/pred_mz · 1e6`, populated only on matched
  rows.

---

## 6. Outputs

One DataFrame, **identical columns to `flatten_match_tree`** (so it is a true
drop-in):

| column group | content |
| --- | --- |
| compound | `compound_formula`, `compound_score`, `compound_category` |
| ion | `ion_formula`, `ion_score`, `ion_category`, `mechanism_id` (`im.mascope_notation`) |
| isotopologue | `isotope_formula`, `iso_label` (`M0` / `13C` / …), `is_base` (`i == 0`), `theo_mz`, `rel_abundance` |
| match | `iso_score`/`iso_category` (None if unmatched), `sample_peak_id`, `sample_peak_mz`, `sample_peak_intensity`, `ppm_error`, `abundance_error` (None for the base) |

`io_mascope._score_candidates_local` stamps `frame.attrs` with
`match_batches = 0`, `match_batch_failures = []`, `match_formulas = len(formulas)`
so callers see a uniform shape across both backends.

---

## 7. Properties, invariants & gotchas

- **Same maths, same schema — only the *source* moves.** The scores are Mascope's
  authored code (`mascope_tools`, public PyPI, same authors), executed locally and
  pinned to a library version. This keeps the "Mascope is the scorer" invariant.
- **No M0, no candidate.** An envelope whose monoisotopic line isn't detected is
  dropped before scoring — there is nothing to anchor relative abundances to.
- **Intensity-gated isotope attribution.** Being inside the ±5 ppm window is *not*
  enough; the peak's abundance must match the prediction within 40 %, or it stays
  unmatched. This is what stops a dense spectrum from "confirming" every isotope
  by mass coincidence.
- **Only matched isotopologues are scored data.** Unmatched envelope lines are
  still emitted (so the predicted pattern is visible) but with `None` scores/ppm —
  never a fabricated zero.
- **Deterministic.** No network and no RNG, so a re-run over the same inputs is
  byte-identical — this is what lets the default scorer satisfy
  `test_determinism.py` (the `match_compounds` fallback re-introduces server-side
  variation). See [`ARCHITECTURE.md §7`](ARCHITECTURE.md#7-reproducibility--provenance).
- **Mixed +/- adducts are unsupported** by `adduct_to_mech` (it raises); a
  multi-part adduct must be all-add or all-subtract.
- **The switch is opt-out, not opt-in.** Local scoring is the default;
  `PEAKY_LOCAL_SCORING=0/false/no/off` falls back to the network path (see
  [`DATA_IO.md §3`](DATA_IO.md#3-the-transformation-stage-by-stage) and
  [`MASCOPE_TOOLS_INTEGRATION.md`](MASCOPE_TOOLS_INTEGRATION.md)).

---

## 8. Code map

| function (`local_scoring.py` unless noted) | role |
| --- | --- |
| `score_candidates_local` | the engine: peaks + formulas → flat per-isotopologue table |
| `adduct_to_mech` | peaky adduct label → its one spelling in the standard adduct notation |
| `_category` | score → `probable` / `possible` / `unlikely` |
| `utils.parse_ionization` / `combine_formula_and_ionization` (mascope_tools) | parse channel; build the ion formula |
| `predict_isotopes` (mascope_tools, IsoSpec) | theoretical isotope envelope of the ion |
| `score_pattern_v2` (mascope_tools) | the fit score: mass and intensity likelihoods per line, a detectability-gated charge for an absent one, combined as an abundance-weighted geometric mean |
| `io_mascope.score_candidates` | backend dispatcher (local default ↔ `match_compounds`) |
| `io_mascope._score_candidates_local` / `_local_scoring_enabled` | local bridge (peaks from cache, mechanism names) + `PEAKY_LOCAL_SCORING` switch |
