# Peaky — The batch's m/z axis (measurement, lock model, band-step frame)

> `peaky batch` / `peaky pool`, before anything is assigned: `--mass-axis {auto,locks,reference,off}`,
> `--axis-fingerprints {auto,refresh,off}`, `--axis-fingerprint-dir DIR`.
> Code: `batch/assign_batch.py` (`measure_axis`, `run`), `batch/axislock.py` (the lock model),
> `batch/axisframe.py` (the band-step frame), `batch/massqc.py` (the reference-ion probe and the
> apply / invert helpers), `io/io_mascope.py` (`fetch_axis_fingerprint`, `set_axis_correction`).

## 1. What this stage does

An Orbitrap's served m/z axis can sit off the true one by several ppm, and not uniformly:
a smooth curve across the range plus **steps** at fixed m/z. Pass 1's per-file self-calibration
models only a constant and a 1/(m/z) term, so a curved or stepped axis leaves correct formulas
several scoring widths off-centre. This stage measures the batch's axis once, from its own
peaks, and corrects the time series and every served peak table before assignment; the
correction is recorded, reversible (`mz_axis_ppm`, `restore_axis`) and undone for publishing.

Three sources of correction, in the order they are tried on an Orbitrap-class batch
(`--mass-axis auto`):

1. **The band-step frame** (`--axis-fingerprints auto`, the default; `refresh` reads the
   documents from the server again): the calibration's band-wise steps, read from each
   sample's served **axis fingerprint**, taken out first.
2. **The lock model** (`batch/axislock.py`): a smooth spline plus explicit steps fitted to
   calibration-free locks found in the batch's own peaks; on the step-free (W) series when a
   frame was built, on the served series otherwise.
3. **The reference-ion wave** (`batch/massqc.py`, `--mass-axis reference` or the fallback when
   the locks build no model): the reagent's formula-certain ions (nitrate, labelled nitrate
   and bromide have a table).

A TOF is measured and never corrected; a pool of several batches and server-side scoring
(`PEAKY_LOCAL_SCORING=0`) are measured and held. `--mass-axis off` skips the stage.

## 2. The lock model (what the frame sits in front of)

`axislock.fit(ts, polarity=...)` finds **locks**: recurring peaks whose +-5 ppm window
(`LOCK_PPM`) holds one closed-shell formula once the peak's own 13C / 34S / 37Cl / 81Br lines
strike the rivals, then peaks one exact unit step (CH2, O, CO2 ...) from two locks; it fits a
smoothing spline through their main body, drops locks off the curve (`OFF_CURVE_PPM` 1.5),
adds +-1 ppm locks on the corrected axis (`NARROW_PPM`) and walks up the range segment by
segment to model the steps an Orbitrap's axis takes (`STEP_MIN_PPM` 1.0 ... `STEP_MAX_PPM`
4.0). Between two segments across a step, and past the first and last lock, nothing is
corrected. The model is applied when it predicts held-out locks (5-fold) within 0.3 ppm
(`LOCK_CV_OK_PPM`) or half the raw error and some lock reads 1 ppm off (verdicts `axis_trend`
/ `axis_steps`; `axis_ok` applies nothing; `locks_unpredictive` / `locks_unvalidated` apply
nothing and fall back to the reference wave in `auto`), up to 7 ppm (`LOCK_MAX_CORRECTION_PPM`;
a walk segment's line past the cap is cut and left uncorrected, a curve past it holds the
model as a broken calibration). `tables/mass_axis_locks.csv` lists every lock.

What the lock model cannot do is the reason for the frame: a step it finds is left as a gap
(the step's position between two locks is unknown), and a step it cannot cross with a unit
link ends the walk. On a nitrate Orbitrap batch the walk left two steps at band edges as
gaps; on a second batch of the same calibration it ended below a band edge that the walk on
the step-free series crossed.

## 3. The band-step frame (`batch/axisframe.py`)

### 3.1 What Mascope serves

A Thermo Orbitrap profile is stored as chunks of points on the FFT frequency bins plus one
additive m/z correction per chunk, written by the mass calibration; on the Orbitraps that
write it, that correction is one constant per frequency band of ~16.28 law-kHz, so the m/z
error is a smooth trend across bands plus a band-wise part that shows up as **steps at the
band edges** (0.3 ppm to tens of ppm). A lock mass rescales a whole scan and cannot remove a
step. Mascope measures this **axis fingerprint** of every raw Orbitrap file when it converts
it and serves it per sample item (`GET /api/samples/{sample_item_id}/axis_fingerprint`; the
SDK's `client.samples.get_axis_fingerprint(id)`, `mascope-sdk >= 2026.10.28`) as a document of
schema `mascope.axis_fingerprint` version 1: a `status` (`ok`, `no_fingerprint`, `missing`,
`unavailable`, `error`), per acquisition mode the band grid (`grid.period_khz`,
`grid.phase_khz`, `grid.source` `measured` or `seed`, `grid.edge_uncertainty_khz`,
`grid.n_hits`), the calibration law's `B_median` / `C_median`, the `segments` table `[k,
f_lo_khz, f_hi_khz, mz_lo, mz_hi, s_ppm, n_chunks, n_level, level_share, n_scans, scan_sd_ppm,
interp]` and the mode's `flags`, and an `item` block for the sample (its `mode_id`, `streams`,
`lock`). `s_ppm` is the relative spacing anomaly of the band's points off the frequency bins
once the stored correction is undone; the m/z correction the band carries follows exactly:
`r = ((1 + s e-6)^(-2/3) - 1) e6` (`axisframe.r_from_s`, the inverse of Mascope's
`s_from_fudge_ratio`). The law `f(m) = sqrt((B + sqrt(B^2 + 4 m C)) / (2 m))` (law-kHz) and
`k = floor((f(m) - phase) / period)` put a peak in its band; a higher k is a lower m/z.

### 3.2 What peaky takes from it, and why only that

The **band steps**. Over the measured bands (`interp` False, `n_level >= MIN_N_LEVEL` 3,
`level_share >= MIN_LEVEL_SHARE` 0.5), `r_k` is split into a smooth trend and a band-wise
residual `d_k = r_k - trend(k)`: first in each document's own table (which decides the bands
it contributes, 3.3), then in the calibration group's pooled table, whose `d` is applied. The
trend is a **robust local quadratic** (LOESS: at each k a weighted quadratic over the measured
bands within `H_BANDS` 6, tricube weights of span 6.5, times bisquare robustness weights from
the previous pass, `LOESS_ITERATIONS` 3). The robust scale is `max(1.4826 x median |residual|,
SIGMA_FLOOR_PPM 0.1)` and the cut-off `4.685 sigma`, 0.47 ppm at the floor: a band within half
a ppm of its local trend is trend, a band further off is a step and carries weight 0. A served
table is smooth to a few thousandths of a ppm, so without the floor the scale would be the
table's rounding and a step's neighbours would be discarded with it. A window's **support** is
its bands carrying weight: a quadratic through at least `LOESS_MIN_QUADRATIC` 5 of them, a
line through at least `LOESS_MIN_BANDS` 3, no trend with fewer (the band is then unmeasured);
a band without a trend in one pass keeps its weight for the next. The trend must read a band
from its support: `d` is taken where the band carries weight itself, where trusted bands
(carrying weight and a trend of their own in the final pass) lie on both sides of it in its
window, or where one lies within `LOESS_REACH_BANDS` 1 of it. A band without a trend of its
own (a lone band beside a staircase or a gap) keeps the weight an earlier pass set, is fitted
through but is no support, else the bands past a staircase would look two-sided. A band whose
trusted support lies on one side only, two or more bands away, is read by an extrapolation: it
keeps its `d` when that is within the cut-off (it is trend) and is otherwise **extrapolated**
-- no `d`, unmeasured, counted (`n_extrapolated`; `extrapolated_k` per document,
`extrapolated` per band of a group). Unmeasured bands (no trend, extrapolated, interpolated or
missing) get `d = 0` and are counted. Only `d` is applied: `m_W = m (1 - d_k(m) e-6)`. The
smooth part of the axis error (the trend across bands, the law term, Mascope's one-point
calibration factor, the instrument's common curve) is deliberately left to the lock model,
which already handles it and validates it on held-out locks; the fingerprint contributes what
chemistry cannot see, the position and size of the steps. A document whose largest `|d|`
passes `H_MAX_PPM` 60 is refused (a broken calibration). The trend values and robustness
weights are recorded per group (a band with weight 0 is a step the smoother refused to
follow).

### 3.3 One calibration group per batch

Documents of ONE calibration leave different bands unmeasured and fit slightly different
trends, so per-item tables would put one peak in different bands and give it different steps.
Items whose documents share polarity, resolution and scan range, agree on `C / B^2` within
`C_OVER_B2_TOL` 1e-11 and on every band both measure within `S_AGREE_PPM` 0.3 form a
**calibration group**: one table (the per-band median `s` over the members that measure it; a
member contributes only the bands that carry a trend in its own table, so a band without one
or extrapolated there is not pooled from it), one grid (the `n_hits`-weighted median period
and phase, the phases first brought within half a period of the best-measured member's), one B
and C (medians). Every item of the group uses the group's table, grid and law, so `k(m)` is
the same for all its items. v1 requires **exactly one group in the batch**: two or more (a
batch mixing calibrations) hold the frame (`held = "N calibration groups in one batch"`).

The **edge zone** is in frequency: `max(members' edge_uncertainty_khz (null -> 0.1 kHz), half
the spread of the members' phases)`. A peak whose law frequency lies within it gets the mean
of the two neighbouring bands' `d` and is counted (`n_edge_ambiguous`); the inverse is the
exact inverse of that rule: inside a zone it accepts the mean-`d` preimage only, outside every
zone a band's own preimage.

### 3.4 A usable document

`status` `ok`; schema `mascope.axis_fingerprint`, version <= 1; the schema's shape (a field
the schema gives as a list -- `modes`, `flags`, `item.streams`, `item.scans`, a mode's
`flags`, `segments.columns` / `rows` -- holding a count, a string or an object is refused as
`malformed document: <field> is <type>, not a list`, `axisframe.document_shape_why`); the mode
is `item.mode_id` (else `dominant_mode`) among the document's modes, with the batch's
polarity; the sample not stitched (`item.streams` null or one stream, no composite stream
key); none of the refusing flags `grid_unmeasured`, `not_stepwise`,
`fft_step_not_power_of_two`, `calibration_changes_within_mode` (the advisory `grid_weak`,
`sparse_segments`, `lock_alternates`, `lock_missing`, `lock_ambiguous` are recorded under
`notes`); `grid.source` `measured`; band indices and a span within `MAX_TABLE_BANDS` 10 000
(else `not a band table`); at least `MIN_MEASURED_SEGMENTS` 8 measured bands in the document's
own table, with a trend to read `d` from, and in the pooled table; `|d| <= 60` ppm in both.
Every refusal names its reason in the record; a malformed document is a refusal like any
other, never a stop.

### 3.5 All or nothing, and where the frame is tried

The frame is tried only on an Orbitrap-class batch, in the lock-model path (`--mass-axis auto`
or `locks`), without a hold (a one-batch pool is a batch; a pool of several batches,
server-side scoring and an explicit hold read no fingerprint), with `--axis-fingerprints auto`
or `refresh`. **Every** item of the batch needs a usable document of the batch's polarity and
the documents must form one group; otherwise no frame is built, today's path runs unchanged
and `batch_summary['mass_axis']['frame']` carries the build record with `held` saying why
(per-document verdicts under `documents`) when at least one document was served (section 6).

### 3.6 Order in `measure_axis`, and the gate

1. The **raw lock fit** first: today's model on the served series, its cap and clip, kept as
   the fallback whatever happens next.
2. The series into the W frame (`FrameSet.to_frame`: every item's band steps taken out).
3. The lock model fitted on the W series **unchanged** (`_lock_model`), then today's cap and
   clip on the W model.
4. The **gate**, truth-free; any rule failing holds the frame and today's path stands on the
   raw fit already made (`frame.held` names the rule):
   - (a) the W fit reports a step whose bracket lies within one band width of a grid edge
     (the frame left a step);
   - (b) at a grid edge inside the W model's domain (under `axis_ok`, between the lowest and
     the highest lock) with >= 2 non-bright locks within one band on each side (locks inside
     the edge zone, widened by half the step the frame took out there, left out: their side
     cannot be told), the medians of the locks' residuals differ by more than
     `EDGE_STEP_GATE_PPM` 0.5 across it;
   - (c) the W verdict is not one of `axis_ok`, `axis_trend`, `axis_steps` (nothing is applied
     under a model that does not predict its held-out locks);
   - (d) **reach**: the W model's applied scope (under `axis_ok`, the W fit's lock range) is
     shorter than the raw model's by more than one band width at either end (the frame must
     not shorten what today corrects);
   - (e) **quality**: the W fit's held-out error exceeds `FRAME_QUALITY_CV_RATIO` 1.5 x the raw
     fit's (both finite).
   A caller's hold holds the frame; the raw model's own cap hold does not (the frame may be
   what makes the smooth part correctable), and when the raw walk builds no model the W fit is
   still tried with (d) and (e) vacuous. A frame path that raises holds the frame too
   (`the frame path failed (<type>: <message>)`): the record goes back to the raw fit's, which
   stands.
5. On pass, the **composite** is applied per item: `m_corr = m_W (1 - c_W(m_W) e-6)` inside the
   W model's scope; outside it the policy `OUTSIDE_POLICY = "raw"` applies nothing (the value
   `"frame"`, d alone outside, exists for comparison runs). Under a W verdict of `axis_ok`
   (nothing smooth left to correct) the inner model is the W fit's own lock model with every
   segment a zero line, its holes and the gaps between its segments kept
   (`axisframe.scope_model`; one zero line over the fit's scope when it records no lock model,
   the frame held when it records neither): the band steps alone, applied only where the W fit
   saw locks, and `scope_mz` is that lock range, never the table's reach.
   `predict` returns one total ppm `D = d + c_W - d c_W e-6` with `m_corr = m (1 - D e-6)`
   exactly, so `mz_axis_ppm` restores the series exactly. The inverse (`FrameItem.invert`)
   undoes the inner model (`AxisModel.invert`, the lower segment beside a step) and then
   searches the preimage over the bands k-1, k, k+1 (and the edge zones' mean-d preimages),
   takes the lowest served m/z where two images overlap and leaves a value in a gap unchanged
   (counted).

### 3.7 Reading the documents

`io_mascope.fetch_axis_fingerprint(client, sample_id, ...)`: the SDK's
`samples.get_axis_fingerprint` when the installed SDK has it, else the SDK's generic GET of
`samples/{id}/axis_fingerprint` (an SDK older than the pinned minimum, installed by hand, still
reads the document from a current server, and the run's log says so); a 404 (an older server, a
file that is not an Orbitrap's), a permission or network error -- any exception -- is None with
a logged reason; a fingerprint never stops a batch. Documents whose status is final for a file
(`ok`, `no_fingerprint`) are cached at
`~/.mascope-assign-cache/<sample item id>/axis_fingerprint.v1.json`; `missing`, `unavailable`
and `error` are fetched again every run (a backfill can change them).
`--axis-fingerprints refresh` reads every item's document from the server again, past its
cached copy, and caches the final, readable documents it gets (a file fingerprinted anew by a
newer reader or a backfill; a document of another status is used for that run only and leaves
an earlier cached copy as it was). A document whose fields do not have the schema's shape (a
list field holding a count, a string or an object: `axisframe.document_shape_why`) is handed to
the reader, which refuses it with that reason, and is never cached; a cached copy of the wrong
shape is fetched again. The items are asked in turn, and the first one served no document, for
whatever reason, rules the frame out: the rest are not asked (`fetch`:
`not asked: item <id> had no document`), since a server that is down or rate-limiting would
cost every further item the SDK's retries. A failed request's reason is the server's message
without its address (`io_mascope.server_error_text`). `--axis-fingerprint-dir DIR` reads
`DIR/<sample item id>.json` instead and never asks the server (documents computed locally,
offline runs, tests). How the documents were read is recorded (`source`: `sdk`, `route`,
`directory`, `none`; `refreshed` under `--axis-fingerprints refresh`).

## 4. Constants reference

| constant | value | role |
| --- | --- | --- |
| `axisframe.MIN_MEASURED_SEGMENTS` | 8 | measured bands a group's table needs |
| `axisframe.H_MAX_PPM` | 60 | largest band step admitted; above it the calibration is broken |
| `axisframe.EDGE_STEP_GATE_PPM` | 0.5 | gate (b): residual lock step across a grid edge |
| `axisframe.S_AGREE_PPM` | 0.3 | two documents of one calibration agree on every common band within this |
| `axisframe.MIN_N_LEVEL` / `MIN_LEVEL_SHARE` | 3 / 0.5 | a band counts as measured |
| `axisframe.C_OVER_B2_TOL` | 1e-11 | two documents of one calibration agree on the law's `C / B^2` within this |
| `axisframe.MAX_TABLE_BANDS` | 10 000 | a band index or span past this is not a band table (refused) |
| `axisframe.DEFAULT_EDGE_UNCERTAINTY_KHZ` | 0.1 | the edge zone when the document carries none |
| `axisframe.H_BANDS` / `TRICUBE_SPAN_BANDS` / `LOESS_ITERATIONS` | 6 / 6.5 / 3 | the robust local quadratic trend |
| `axisframe.MAD_TO_SIGMA` / `BISQUARE_C` | 1.4826 / 4.685 | its robust scale and cut-off |
| `axisframe.SIGMA_FLOOR_PPM` | 0.1 | the robust scale's floor (cut-off 0.47 ppm): a band within half a ppm of its trend is trend |
| `axisframe.LOESS_MIN_QUADRATIC` / `LOESS_MIN_BANDS` | 5 / 3 | bands carrying weight for a local quadratic / a line; fewer, no trend |
| `axisframe.LOESS_REACH_BANDS` | 1 | a trend from one side only reads a step this many bands past its support; further out, an extrapolated band |
| `axisframe.OUTSIDE_POLICY` | `"raw"` | outside the W model's scope nothing is applied |
| `assign_batch.FRAME_QUALITY_CV_RATIO` | 1.5 | gate (e) |
| `assign_batch.FRAME_INVERSE_SPREAD_PPM` | 0.05 | `to_server_axis`: a merged row's source inverses counted as disagreeing in its log line |
| `assign_batch.LOCK_MAX_CORRECTION_PPM` | 7 | the lock model's cap, applied to the W model too |
| `pipeline.REPORT_FRAME_CORRECTED_TS_PPM` | 4 | a frame reaching past this hands the PDF's coverage tables the corrected series |
| `io_mascope.AXIS_FINGERPRINT_CACHE` | `axis_fingerprint.v1.json` | the per-sample cache file |
| `io_mascope.AXIS_FINGERPRINT_CACHED_STATUSES` | `ok`, `no_fingerprint` | the statuses cached |

## 5. Outputs and records

| artifact | content |
| --- | --- |
| `batch_summary.json['mass_axis']` | `model` `frame` when applied (`locks` / `reference` otherwise), the verdict, `wave` (the `FrameSet` record: `model` `frame`, the inner lock model, per item its law B / C, grid, edge zone, `k0`, `d_ppm`, `measured`), `inner` (the W lock model's verdict, scope, cap, lock summary), `scope_mz`, `max_abs_ppm`, `shift_ppm`, and `frame`: `policy`, `n_items`, `n_groups`, `source`, `refreshed` (a refresh run), `documents` (per item: `usable`, `status`, `mode_id`, `sha1`, `notes`, `why`, `fetch`; for a usable document its `n_bands`, `n_measured`, `n_extrapolated`, `extrapolated_k`, `h_max_ppm`, `k_range`, grid and lock), `groups` (grid, zone, `n_measured`, `n_unmeasured`, `n_extrapolated`, `h_max_ppm`, the trend's `trend_sigma_ppm` / `trend_scatter_ppm`, and per band `s_ppm`, `r_ppm`, `trend_ppm`, `weight`, `d_ppm`, `measured`, `extrapolated`), `fits` (`raw` and `W`: verdict, `cv_ppm`, `raw_ppm`, `max_abs_ppm`, `scope_mz`, lock counts, steps, uncorrected gaps, clipped), `gate` (`a`..`e`, `failed`, `passed`, `why`), `held` (None when applied, else the rule or the build failure) and `items` (per item: `sample_file_id`, `mode_id`, `document_sha1`, grid, `n_bands`, `n_measured`, `n_interp`, `d_min_ppm`, `d_max_ppm`, flags, `n_peaks`, `n_moved`, `n_unmeasured_band`, `n_edge_ambiguous`). A run that was served a document and built no frame carries `frame` as the build record (`held`, `documents`, `n_items`, `n_usable`, `n_groups`, `groups`, `policy`, `source`, `refreshed`); a frame built and then held carries the record above without `items`. |
| `tables/mass_axis_locks.csv` | every lock of the applied fit; the column `frame` = `W` when the frame was applied (the W fit's locks) |
| `tables/mass_axis_frame.csv` | when applied: item x band -- `sample_item_id`, `k`, `mz_lo`, `mz_hi`, `d_ppm`, `measured` |
| `tables/mass_axis.csv` | the reference-ion probe table (reference path) |
| `data/<tag>_ts_corrected.parquet` | the corrected series the PDF's coverage / batch-max tables read when the frame's correction passes 4 ppm anywhere (`pipeline._report_ts_path`; a pool's per-group report folders get their group's slice, `pipeline._group_report_ts_path`); the run's recorded input `data/<tag>_ts.parquet` stays the raw one |
| `run_manifest.json['input']['axis_fingerprints']` | per item `{sha1, status, mode_id}` of the served document, present only when documents were read (a sha1 exists); no key otherwise |
| `run_manifest.json['output']['counts']['mass_axis']`, and so every `index.jsonl` row | the summary's record as recorded, except a frame record, which grows by kilobytes per item: `frame` summarised to `held`, `gate` (`passed`, `failed`), `n_items`, `n_usable`, `n_groups`, per group its size, reach, edge zone and largest step, `source`, `refreshed`; an applied frame's `wave` to `model`, `version`, `policy`, `inner` and `n_items` (`pipeline._manifest_mass_axis`). The full record stays in `batch_summary.json`. |
| `peaky publish-batch` | a frame run's merged row is put back on the server's axis by every source item named in its `srcs` column (median; `to_server_axis(mz, record, srcs)`), and a row whose sources disagree by more than half the `--tolerance-ppm` is warned about before anything is sent (`load_batch_run(...)['axis_inverse']`) |

The run log says what happened at every step (`[mass-axis] fingerprint: ...` for each document
not read cleanly, `frame: one calibration group over N spectra ...`, `frame NOT applied: ...`,
`APPLIED ... band steps`).

## 6. Byte identity

Two cases, by what the run was served.

- **Nothing served, nothing recorded.** With `--axis-fingerprints off`, on a TOF, in
  `--mass-axis reference` / `off`, under a hold (a pool of several batches, server-side
  scoring), and whenever the first item asked is served no document (a 404 from an older
  server, an SDK and server without the route, no connection and no cached copy, a permission
  or network error, no file in `--axis-fingerprint-dir`; the fetch stops there, so no later
  item is asked), `batch_summary.json` (its wall clock aside), the merged and per-file ledgers
  and every table are byte for byte the run's without the feature: `mass_axis` has no `frame`
  key, `run_manifest.json` has no `input.axis_fingerprints`, and the manifest's counts and the
  `index.jsonl` row carry the same `mass_axis` record. Only the log says what was looked for.
- **A document served, no frame applied.** When at least one item was served a document,
  whatever its status (`ok`, `no_fingerprint`, `missing`, `unavailable`, `error`, or a
  malformed one), and no frame was applied (an item without a usable document, a later item
  served none, several calibration groups, a pooled table past its limits, a hold by the W
  fit or the gate, a reader or frame path that raised), the run adds three records to the
  run without the feature and changes nothing else: `batch_summary['mass_axis']['frame']`
  (the build record, or the held frame's record, section 5),
  `run_manifest.json['input']['axis_fingerprints']` (sha1, status and mode of each document
  served) and, in the manifest's `output.counts.mass_axis`, that `frame` record summarised
  (`pipeline._manifest_mass_axis`), which every `index.jsonl` row carries too. The model, the
  verdict, the correction, the ledgers and the tables are the run's without the feature.

`off` is compared with a checkout of main (the code without the feature) by an opt-in test
(set `PEAKY_MAIN_WORKTREE` to that checkout with a synced `.venv`;
`tests/test_axisframe_wiring.py`), which CI skips. CI compares the off, no-document, 404, TOF
and hold arms with the run without the feature byte for byte, and the runs with a held frame
or a malformed document with it, `mass_axis.frame` aside.

## 7. Properties, invariants & gotchas (v1 limits)

- **The frame removes steps, not the curve.** Everything smooth stays with the lock model; a
  calibration whose smooth error passes the 7-ppm cap is still held as broken, with or without
  a frame, and nothing is applied beyond the W model's scope (policy `raw`).
- **Stitched samples are refused** (several streams, several factors on one axis).
- **Interpolated and otherwise unmeasured bands get no step** (`d = 0`); the peaks on them are
  corrected by the lock model alone and counted (`n_unmeasured_band`).
- **One calibration group per batch.** A batch mixing calibrations holds the frame; per-group
  models are a later version.
- **A peak is moved by its band's step, nothing is joined.** A peak split at a band edge
  (one ion picked as two centroids on either side of an edge, as positive-mode spectra show)
  stays two peaks, each moved by its own band's step; peaks inside the edge zone get the mean
  of the two bands' steps and are counted.
- **Per item, by one map.** Under one calibration group every item's map is the same table on
  the same grid, so a merged row's source items agree exactly when put back on the server's
  axis; the spread is still measured and said.
- **The report's 8-ppm match.** The PDF's coverage and batch-max tables match the ledger to the
  series within 8 ppm; a lock model (<= 7 ppm) and a reference wave (<= 4 ppm) stay inside, a
  frame need not, so past 4 ppm the report reads the corrected series (section 5). A pool's
  per-group reports read their group's slice of the corrected series the same way
  (`pipeline._group_report_ts_path`).
- **Never a stop.** A document that cannot be read, a server without the route, a refused
  document: a logged reason and today's path. A document of an unexpected shape is a refusal
  (`malformed document: <field> is <type>, not a list`), and a reader or a frame path that
  raises all the same holds the frame (`documents unreadable (<exception type>)`, `the frame
  path failed (...)`) with the raw fit standing. A server that answers no document (down,
  rate-limiting, an older one) costs one item's requests: the fetch stops at the first item
  served none, and its reason is the server's message without the server's address.
- **The trend's known limits** (measured on synthetic tables; the `axisframe` module note has
  the detail). (1) On a measured run's first or last band (the table's ends, a band beside a
  run of `H_BANDS` or more bands without a trend, such as a document's own trend-less staircase
  leaves, a lone band beside a short gap, and in the pooled table a band beside the bands its
  members extrapolated or left without a trend), a small step is read as trend and its 1-3
  neighbours take about -0.6 x the step (up to -1.3 x on a short run at a table end); they can
  lie inside the lock model's scope, at the table's low-m/z end. At the floored scale a step of
  ~3 ppm or more on a contiguous run's end band is seen; a lone band beside a gap of one to
  three bands hides one of up to ~8 ppm, a two-band run at a table end beside a gap of two or
  three bands one of up to ~5 ppm on its outer band (its other band taking about -0.8 x it) or
  of ~2 ppm on its inner band (the outer one taking about -1.3 x it, more than the step), and
  the threshold rises with the first pass's scale, which gross steps elsewhere in the table
  inflate (at a first-pass scale of 3.3 ppm a 36-ppm end step was read as trend and put up to
  23 ppm on the clean bands beside it, under the 60-ppm limit). (2) A corner in the trend (a
  ramp meeting a plateau) gives its bands a spurious d of ~1-3 ppm. (3) A real step under the
  0.47-ppm cut-off is partly smoothed into the trend. (4) A gross step (~8-10 ppm or more) on
  the first band of a short run after a gap leaves the rest of that run without a trend: no
  correction there, and no wrong one. (5) A stretch of a broken calibration's staircase smooth
  enough to carry a trend of its own (its bands on one line) is trusted like any trend: the
  first band of the clean run beside it is read through it and can take a spurious d of a few
  ppm, up to tens, under the 60-ppm limit. (6) A band at a staircase's edge that had no trend
  in the pass before the last keeps the weight an earlier pass set; when the final pass gives
  it a trend again it is trusted, and the first clean bands beside it can take a spurious d of
  ~15 ppm typically, up to ~40, in the document's own table and, where that band is pooled, in
  the applied one (rare: about 0.1 % of random synthetic tables with staircases and gaps). The
  0.1-ppm floor is a choice, not an optimum: on one calibration d is stable for floors
  0.03-0.1, on other tables it moves by up to several ppm between 0.05 and 0.2.
- **What the reach rule costs.** A real step on a band two or more bands past its one-sided
  support is not corrected: the band is extrapolated and left to the lock model, as without
  the frame. The reading one band beside the support is kept, so a bend directly beside a
  clean run is still read as a step there. Of two adjacent real steps on a run's last two
  bands, the outer one is extrapolated and lost.
- **`axis_ok` is bounded by the locks.** With nothing smooth left to correct the band steps
  are applied alone, and only inside the W fit's lock domains: the bands beyond the outermost
  locks, where no lock tests the frame, are left as served; an end band that holds locks gets
  its d, limit (1) included.

## 8. Code map

| function | role |
| --- | --- |
| `assign_batch.run` | `axis_fingerprints` / `axis_fingerprint_dir`; `_batch_frame` fetches (stopping at the first item served no document) and builds, `_unread_record` when the reader raised; `measure_axis(frame=, frame_record=)` |
| `assign_batch.measure_axis` / `_frame_path` | the raw fit first, the W fit, the gate (a)-(e), the composite applied per item, the records |
| `assign_batch.to_server_axis` | a corrected m/z back on the server's axis; a frame needs the rows' `srcs` |
| `axisframe.load_document` / `document_shape_why` / `Fingerprint` | a served document read or refused with a reason; its shape checked before anything is read or cached |
| `axisframe.step_table` / `loess_trend` | a band table's trend (the floored robust scale, the support and reach rules), `d`, the measured and extrapolated bands |
| `axisframe.group_fingerprints` / `CalibrationGroup` | the calibration groups, the pooled table, grid and trend |
| `axisframe.FrameItem` / `FrameSet` | one item's map (`to_W`, `predict`, `invert`, `in_scope`); the batch's maps around one lock model |
| `axisframe.gate` | rules (a)-(c) |
| `axisframe.scope_model` | the inner model under the W verdict `axis_ok`: the W fit's lock domains, zero correction |
| `io_mascope.fetch_axis_fingerprint` / `axis_fingerprint_source` | the document per sample, cached; how it was read |
| `io_mascope.set_axis_correction` | a frame record registers each sample's own map for `fetch_peaks` |
| `massqc.fit_from_record` | a `wave` record back into its model (`locks`, `frame`, or a wave) |
| `pipeline._report_ts_path` / `_group_report_ts_path` / `_axis_fingerprint_inputs` / `_manifest_mass_axis` | the report's corrected series past 4 ppm, and a pool's per-group slice of it; the manifest's documents read; the manifest's summary of the frame record |
| `publish.load_batch_run` | the merged ledger back on the server's axis, per source item for a frame run |
