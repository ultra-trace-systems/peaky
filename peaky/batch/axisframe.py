"""The band-step frame: Mascope's Orbitrap axis fingerprint turned into a per-item map
that takes the band-wise STEPS out of the m/z axis before the lock model sees it.

WHAT THE FINGERPRINT IS. On some Orbitrap models the data system divides the frequency
axis into bands about 16.28 kHz wide, and each mass calibration stores ONE additive
m/z correction per band on top of a smooth trend. A raw file keeps a scan's profile as
chunks of points on the FFT frequency bins plus that per-chunk correction; converting
the corrected m/z back to frequency with the scan's own B, C spaces the points off the
bins by a relative amount `s` (ppm), one value per band. The lock mass rescales a
whole scan and cannot remove a step between bands, and no centroid flag marks one.
Mascope measures `s` per band when it converts a file (schema
`mascope.axis_fingerprint`, version 1) and serves the document per sample file and per
sample item; it alters no m/z with it.
Per acquisition mode the document gives the band grid (`grid.period_khz`,
`grid.phase_khz` in law-kHz; `grid.source` 'measured' or 'seed'; `grid.n_hits` and
`grid.edge_uncertainty_khz` say how well the edges were found), the mode's
`calibration.B_median` / `C_median` / `C_over_B2`, and a `segments` table with one row
per band k: `[k, f_lo_khz, f_hi_khz, mz_lo, mz_hi, s_ppm, n_chunks, n_level,
level_share, n_scans, scan_sd_ppm, interp]`.

THE LAW AND THE BANDS. `f_law(m) = sqrt((B + sqrt(B^2 + 4 m C)) / (2 m))` [law-kHz],
inverse `m = B/f^2 + C/f^4`. Band `k = floor((f_law(m) - phase) / period)` holds the
frequencies `[phase + k period, phase + (k + 1) period)`, i.e. the m/z `(mz_lo, mz_hi]`
with `mz_lo = m(f_hi)`, `mz_hi = m(f_lo)`: a higher k is a LOWER m/z. The edge between
bands k and k + 1 sits at `mz_lo(k) == mz_hi(k + 1)`; the edge with index j (frequency
`phase + j period`) separates band j (below it in m/z) from band j - 1 (above it).

s <-> r. With `r = fudge / law_mz` the stored correction as ppm of the m/z,
`s = ((1 + r e-6)^-1.5 - 1) e6` (Mascope's `reference.s_from_fudge_ratio`, checked
bit-exact against the stored per-chunk corrections), so, exactly,
`r = ((1 + s e-6)^(-2/3) - 1) e6`, about -(2/3) s (`r_from_s`). `r_k` is the relative
m/z error the calibration wrote into band k.

ONE TABLE PER CALIBRATION GROUP. The documents of one calibration agree on every band's
s to a few hundredths of a ppm, yet a short file leaves bands unmeasured that a long
one measures, every file's grid is fitted anew (phases scatter by up to 0.3 kHz) and a
trend fitted per document differs at the ends; per-item tables then put the same peak
in different bands and give it different d (the batch's spectra then disagree and the
lock bins average two axis states). So the items of a batch are grouped into
CALIBRATION GROUPS (`group_fingerprints`): two documents belong together when their
modes have the same polarity, resolution and scan range, their `calibration.C_over_B2`
agree within C_OVER_B2_TOL and, on every band measured in both, |s_a - s_b| <=
S_AGREE_PPM (different calibrations differ by 2 ppm or more). A document joins the
first group whose EVERY member it agrees with, else starts one (the stricter reading: a
chain of near-agreements never merges two calibrations). A group has ONE table (per
band the median s over the members that measure it -- a member contributes only the
bands its own table reads a d on; a band no member measures stays unmeasured, d = 0,
counted), ONE grid (the n_hits-weighted median period and phase of the members, the
phases first brought within half a period of the member with most hits, the result
modulo the period; every member's bands are aligned onto it by their centre frequency,
so a grid whose phase wrapped past 0 shifts its labels by one and still pools right)
and B, C = the medians of the members' `B_median` / `C_median`. Every item of the group
uses the group's table, grid, B and C: k(m) and d are identical for all of them.
Version 1 requires exactly ONE group per batch; several (one batch mixing calibrations)
HOLD the frame with `held = "N calibration groups in one batch"` (the groups are
recorded).

THE TREND IS LOCAL (the LOESS convention, pinned). The error of the served axis is a
smooth part (the trend of r across bands, the law's own curvature, Mascope's one
calibration factor, the instrument's common curve) PLUS a band-wise residual `d_k` that
shows up as STEPS at the band edges, 0.3 ppm to tens of ppm. The lock model
(`batch.axislock`) measures the smooth part from the batch's own chemistry and handles
it today; what chemistry cannot see is a step that no unit link crosses. So the frame
takes out `d_k` only. A global cubic in k misfit the ends of the band range (ramps of a
few ppm in d at both ends of a served table that were trend, not steps), so `trend(k)`
is a ROBUST LOCAL QUADRATIC (`loess_trend`) over the group's measured bands (x = k, y =
r_k): at each band a weighted least-squares quadratic over the measured bands with |k' -
k| <= H_BANDS = 6 (one-sided at the ends), weights = tricube (1 - (|k' - k| / 6.5)^3)^3
x the bisquare robustness weight of the previous iteration; LOESS_ITERATIONS = 3
robustness iterations after the unweighted first fit; the bisquare weight is w = (1 -
(res / (BISQUARE_C sigma))^2)^2 for |res| < BISQUARE_C sigma, 0 beyond, with BISQUARE_C
= 4.685 and the robust scale sigma = max(MAD_TO_SIGMA x median|residual|,
SIGMA_FLOOR_PPM): MAD_TO_SIGMA = 1.4826 (the normal-consistent scale; NEVER the raw
median absolute residual alone: the two conventions moved d by several ppm on one served
document), SIGMA_FLOOR_PPM = 0.1 ppm. THE FLOOR IS WHAT MAKES THE FIT ROBUST ON A REAL
TABLE. A served table carries s to four decimals and is smooth in k to a few thousandths
of a ppm (to the 1e-4 rounding at best), so the scale of its residuals collapses towards
the rounding and, without a floor, every band off the local quadratic by more than ~0.02
ppm (~5e-4 ppm on the smoothest tables) is discarded: an isolated 3-ppm step zeroed its
whole 13-band neighbourhood in the first pass, the bands left without a trend were reset
to weight 1 -- the step itself among them -- and the trend then FOLLOWED the step: d = 0
on the step band and -0.86 x the step on its neighbours (measured on synthetic tables at
a band scatter <= 0.002 ppm and on the served documents). With the floor the cut-off is
BISQUARE_C x 0.1 = 0.47 ppm: a band within half a ppm of the local trend is trend
(weight 0.35-1), a band further off is a step (weight 0), whatever the rounding. The
SUPPORT of a window is the bands CARRYING a positive weight: a quadratic through >=
LOESS_MIN_QUADRATIC = 5 of them, a line through >= LOESS_MIN_BANDS = 3, no trend with
fewer -- that band is UNMEASURED (d = 0) -- and a band without a trend in a pass KEEPS
its weight for the next one (never reset to 1: a gross step reset would come back and
become the trend; a first version, which reset, left a 30-ppm step unmeasured in every
trial). THE TREND MUST READ ITS BAND FROM ITS SUPPORT: d is taken where the band carries
weight itself, where TRUSTED bands lie on BOTH sides of it within its window (an
interpolation), or where one lies within LOESS_REACH_BANDS = 1 of it (the reading by
which the robust fit refused that band: its residual from the fit beside it). A trusted
band carries weight AND a trend of its own in the final pass; a band without a trend (a
lone band beside a staircase or a gap) keeps the weight an earlier pass set (from a fit
through it while its neighbours still carried weight, its own leverage hiding most of
its offset; 1 when no pass gave it a trend), is fitted through, but is no support:
counted as support it made the bands past a staircase look two-sided (on synthetic
tables one band 30 ppm off beside a two-band gap put up to 17 ppm on the first band of
the run past the gap in the document's own table; ~2 ppm without it, KNOWN LIMIT (5)). A
band whose window's trusted bands all lie on one side, the nearest two or more bands
away, is read by an extrapolation, which may confirm it as trend (its residual within
the cut-off: it keeps that d, as any band within half a ppm of its trend) but never make
it a step: beyond the cut-off the band is EXTRAPOLATED -- no d, unmeasured, counted
(`n_extrapolated`, `extrapolated_k` per document, `extrapolated` per band of a group)
and left to the lock model as without the frame; a real step on such a band is not
corrected (of two adjacent real steps on a run's last two bands, the outer one is
extrapolated and lost). Such bands come from a gross step on the first bands of a run
(beside a table end or a gap): it zeroes its neighbours' weights in the first pass, and
their trend, a line or quadratic extrapolated from the far side of the run over two
bands or more, gave them a spurious d of several ppm with every gate passing (a broken
calibration's served document: its own trend-less staircase, then such bands, then a
real band step beside a clean run; the step keeps its d). A run of adjacent steps of
tens of ppm (a broken calibration's staircase) zeroes itself and its neighbours and is
left unmeasured (or measured as steps where a clean trend lies on both sides), left to
the lock model as today, unless a stretch of it is smooth enough to carry a trend of its
own (KNOWN LIMIT (5)) or a band at its edge is trusted on a weight an earlier pass set
(KNOWN LIMIT (6)); an isolated step on an interior band whose windows hold measured
bands on both sides is measured as itself to the table's scatter (up to a scatter of
0.05 ppm; served tables scatter by 1e-4 to 0.05). KNOWN LIMITS, measured on synthetic
tables: (1) on a measured run's FIRST or LAST band -- the table's ends, a band beside a
gap of H_BANDS or more bands without a trend and a lone band beside a short gap; in the
pooled table also a band beside the bands its members extrapolated or left without a
trend (a document's own trend-less staircase makes such a gap: no member contributes
those bands to the pooled table) -- the window is one-sided and the band's own leverage
is high, so a small step there is NOT seen: the step band keeps a weight near 1 and d
near 0 and its neighbours take weight 0 and about -0.6 x the step, up to -1.3 x on a
short run at a table end (the failure the floor removes in the interior); at a scatter
of 0.1 ppm or more the same happens beside the second band for steps of a few ppm. The
wrong d lands on the 1-3 bands next to the run's end, and these can lie inside the lock
model's scope (at the low-m/z end of a table, where a batch's lightest locks sit). The
reach rule does not remove it: those bands' trend is an interpolation between the end
band and the run. At the floored scale (cut-off 0.47 ppm) a step of ~3 ppm or more on
the end band of a contiguous run is seen (weight 0) and measured as itself (its
neighbour carries weight: within the reach). The threshold is higher in three cases. A
lone band beside a gap of one to three bands has more leverage still and hides a step of
up to ~8 ppm; the run past the gap takes up to about half of it (3.8 ppm from 8 ppm
beside a one-band gap), in the pooled table too. A run of two bands at a table end
beside a gap of two or three bands reads a step of up to ~5 ppm on its outer band as
trend, its other band taking about -0.8 x it, and a ~2-ppm step on its inner band beside
a three-band gap as trend, the outer band taking about -1.3 x it (larger steps leave
both bands unmeasured). And the threshold rises with the FIRST pass's sigma, which gross
steps anywhere in the table inflate: four bands 40-49 ppm off (each measured as its own
step) raised it from the floor to ~0.3 ppm and hid a 5-ppm step on the first band past a
gap, ~3 ppm landing on the bands beside it; a pooled table whose first-pass sigma was
3.3 ppm read a 36-ppm step on a run's end band as trend and put up to 23 ppm on the
clean bands beside it, under H_MAX_PPM. A leave-one-out residual would measure the small
end step too but extrapolates the curved top end of real tables by up to 7 ppm, so it is
not used for the weights. (2) A corner in the trend (a ramp meeting a plateau, seen at
the top end of some tables) gives its bands a spurious d of ~1-3 ppm at any floor: a
local quadratic cannot follow a corner. (3) A real step under the 0.47-ppm cut-off is
partly taken into the trend (at a lock-mass band of a served table the frame read one
half to three quarters of the step the lock model measured there). (4) A gross step
(~8-10 ppm or more) on the first band of a short run after a gap zeroes the rest of that
run, which is then left without a trend (d = 0, unmeasured): the correction is lost
there, no wrong one is applied. (5) A stretch of a staircase smooth enough to carry a
trend of its own (three or more of its bands on one line: a ramp of steps of one size)
is trusted like any trend, and the first band of the clean run beside it (rarely the
next few) is read by an interpolation through it: a spurious d, a few ppm typically and
up to tens, under H_MAX_PPM, so no limit refuses it (staircases of 2-8 bands with steps
of 5-40 ppm, part of them unmeasured, before a clean run: about 6 % of the pooled tables
kept |d| > 0.5 there, 11 % when a band without a trend counted as support). Requiring
two trusted bands on each side removed almost none of them and moved d on served
documents, so it is not used. A band without a trend of its own, fitted through, still
pulls the trend of the run's bands within its window by about 7 % of its offset from ~30
ppm up (at most ~2 ppm below that) in the document's own table (~2 ppm at 30 ppm off, ~7
at 100, up to ~14 at 200; its h_max and record); the pooled table leaves that band out
and does not carry it. (6) A band at a staircase's edge that had no trend in the pass
before the last keeps the weight an earlier pass set; when the final pass gives it a
trend again it is trusted all the same, and the first clean bands beside it are read by
an interpolation through it: a spurious d of ~15 ppm typically and up to ~40, under
H_MAX_PPM, in the document's own table and, where the member contributes that band, in
the pooled one (about 0.1 % of random tables with staircases and gaps). Requiring a
trend in the pass that set the weight as well removed most of these but lost more real
steps elsewhere, so it is not used without an A/B run. The 0.1-ppm floor keeps one
calibration's documents in agreement (d stable to 0.1 ppm for floors 0.03-0.1 on one
calibration; on other tables d moves by up to several ppm between floors 0.05 and 0.2,
so the value is a choice, not an optimum). `d_k = r_k - trend(k)` on measured bands,
`d = 0` (and 'unmeasured') on interpolated, missing, trend-less or extrapolated ones; a
band whose final weight is 0 lies more than 0.47 ppm off the local trend fitted without
it: a step the smoother refused to follow (or it had no trend in the pass before the
last and kept the weight 0 it had). The trend values, the weights, the scale behind the
weights (`sigma_ppm`, floored) and the table's own scatter about the trend
(`scatter_ppm`: MAD_TO_SIGMA x median|residual|, unfloored) are recorded per group.
Lock-mass bookkeeping (`m_lock_eff`, `s_lock_eff_ppm`) cancels in the trend subtraction
and is only recorded.

THE EDGE ZONE IS IN FREQUENCY. The grid is measured per file with an edge
uncertainty of ~0.1-0.4 kHz, and the grids of one calibration scatter by up to
0.3 kHz in phase; a 0.06-kHz shift of an edge moves it by ~0.12 Da at m/z 400
(dm/m = 2 df/f). So a peak is EDGE-AMBIGUOUS when its law frequency lies within
`zone_khz` of a band edge, with `zone_khz` = max(members' `grid.edge_uncertainty_khz`
(null -> DEFAULT_EDGE_UNCERTAINTY_KHZ = 0.1), half the spread of the members' phases).
Such a peak gets the MEAN of the two neighbouring bands' d (d = 0 for an unmeasured
or missing neighbour: it halves the worst error) and is counted (`n_edge_ambiguous`);
the inverse accepts only the mean-d preimage there (the exact inverse of the forward
rule; `FrameItem.preimage`). The
map is thus continuous ACROSS an edge and jumps by half the step at each zone
boundary. (A fixed 3-ppm ambiguity in m/z was the earlier rule; it is gone.)

THE FRAME, THE LOCK MODEL AND THE COMPOSITE. `m_W = m (1 - d(m) e-6)` on the served
m/z as peaky reads it. The lock model is fitted on the W-frame series UNCHANGED and
composed afterwards: `m_corr = m_W (1 - c_W(m_W) e-6)`, reported as ONE total ppm `D`
with `m_corr = m (1 - D e-6)` exactly (`D = d + c_W - d c_W e-6`), so a corrected
series restores exactly (assign_batch.AXIS_COL / restore_axis). The composite reaches
where the W model does (policy 'raw'): no d is applied where no lock was seen. Under the
W verdict `axis_ok` (nothing smooth to correct) the inner model is the W fit's own lock
model with every segment set to zero (`scope_model`): the band steps alone, inside the
W fit's lock domains; the composite's scope is the lock range, never the table's reach.

WHAT IS REFUSED (`load_document`, a reason for each): a document whose status is not
`ok`; a schema version this reader does not know; a mode of the wrong polarity; no
`segments` table; a grid that was not measured (`grid.source` 'seed': the band edges
are a guess); a mode flagged `grid_unmeasured`, `not_stepwise`,
`fft_step_not_power_of_two` or `calibration_changes_within_mode`; a stitched sample
(its peaks sit on several streams' factors); fewer than MIN_MEASURED_SEGMENTS
measured bands; a step table whose largest |d| passes H_MAX_PPM (a broken
calibration, left to the instrument); a document whose fields do not have the
schema's shape (`document_shape_why`: a list field holding a count, a string or an
object) or whose band index passes MAX_TABLE_BANDS. `grid_weak` is advisory: allowed
and recorded.
The group's pooled table is held by the same two limits.

CONSTANTS, a priori (their provenance; a change needs an A/B run):
  MIN_MEASURED_SEGMENTS = 8   a local quadratic over 13 bands needs a few bands on
                              each side of a step to see it as one
  MIN_N_LEVEL = 3             Mascope's `min_chunks_per_segment`: a band is measured
                              when >= 3 chunks sit at its level
  MIN_LEVEL_SHARE = 0.5       the share under which Mascope calls a table
                              `not_stepwise`: one level must own half a band's chunks
  S_AGREE_PPM = 0.3           same-calibration documents agree on s to a few hundredths
                              of a ppm, different calibrations differ by 2 ppm or more:
                              0.3 separates them
  C_OVER_B2_TOL = 1e-11       the law's shape term; a lock-mass rescale changes it by
                              ~1e-13, a recalibration by far more
  DEFAULT_EDGE_UNCERTAINTY_KHZ = 0.1  the zone when a grid reports none (served values
                              about 0.08-0.4 kHz)
  H_BANDS = 6, TRICUBE_SPAN_BANDS = 6.5, LOESS_ITERATIONS = 3, MAD_TO_SIGMA = 1.4826,
  BISQUARE_C = 4.685, LOESS_MIN_QUADRATIC = 5, LOESS_MIN_BANDS = 3  the trend (above)
  LOESS_REACH_BANDS = 1       a trend from one side only reads the band beside its
                              support (the reading that refused it) and, further out,
                              confirms trend but makes no step: the largest reach that
                              removes the spurious d beside a broken calibration's
                              staircase; it changed no d on any other served document
  SIGMA_FLOOR_PPM = 0.1       the robust scale's floor: a served table is smooth to a
                              few thousandths of a ppm, so the measured scale is near
                              the rounding, not scatter; 0.1 (cut-off 0.47 ppm) keeps
                              a band within half a ppm of the local trend and discards
                              a step; the value is a choice, not an optimum (KNOWN
                              LIMITS)
  H_MAX_PPM = 60.0            above the largest band step measured on a calibration
                              still worth correcting (tens of ppm after a bad one)
  EDGE_STEP_GATE_PPM = 0.5    the lock model's own line tolerance (axislock.SEG_TOL_PPM):
                              a residual step across an edge larger than this means
                              the frame left, or made, a step
  OUTSIDE_POLICY = "raw"      outside the W model's scope nothing is applied (the A/B
                              runs kept it: d alone beyond the scope lost), under the
                              verdict axis_ok too (the scope is then the W fit's lock
                              domains, `scope_model`); the alternative "frame" applies d
                              alone on every measured band and stays for comparison runs

Pure numpy / pandas (batch.axislock's AxisModel imported where a lock model is built):
no server access and no import of the io layer. Every object
duck-types `batch.axislock.AxisModel` for `batch.massqc` (`in_scope`, `predict`,
`invert` / `preimage`, `mz_range`, `max_abs_ppm`, `as_dict`), is JSON-safe through
`as_dict`/`from_dict` and picklable for spawned workers.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

SCHEMA = "mascope.axis_fingerprint"
VERSION = 1                   # the schema version this reader knows (a higher one is refused)

MIN_MEASURED_SEGMENTS = 8
H_MAX_PPM = 60.0
EDGE_STEP_GATE_PPM = 0.5
OUTSIDE_POLICY = "raw"
MIN_LEVEL_SHARE = 0.5
MIN_N_LEVEL = 3
S_AGREE_PPM = 0.3
C_OVER_B2_TOL = 1e-11
DEFAULT_EDGE_UNCERTAINTY_KHZ = 0.1

# the LOESS trend (module note THE TREND IS LOCAL)
H_BANDS = 6
TRICUBE_SPAN_BANDS = 6.5
LOESS_ITERATIONS = 3
MAD_TO_SIGMA = 1.4826         # sigma = 1.4826 x median |residual|, never the raw median alone
BISQUARE_C = 4.685
LOESS_MIN_QUADRATIC = 5       # bands carrying weight for a local quadratic ...
LOESS_MIN_BANDS = 3           # ... and for a local line; fewer: no trend, the band is unmeasured
LOESS_REACH_BANDS = 1         # a trend read from one side only reaches this many bands past its support;
                              # further out a band off it by the cut-off is extrapolated: no d, unmeasured
SIGMA_FLOOR_PPM = 0.1         # the robust scale never falls under this: a served table is smooth to
                              # a few thousandths of a ppm, and a scale that small discards every band

POLICIES = ("raw", "frame")
REFUSING_FLAGS = ("grid_unmeasured", "not_stepwise", "fft_step_not_power_of_two",
                  "calibration_changes_within_mode")
ADVISORY_FLAGS = ("grid_weak", "sparse_segments", "lock_alternates", "lock_missing", "lock_ambiguous")
#: the lock-model verdicts (assign_batch._lock_model) under which d may be applied: the W
#: model predicts its held-out locks, or there is nothing smooth to correct
ACCEPTED_VERDICTS = ("axis_ok", "axis_trend", "axis_steps")

SEGMENT_COLUMNS = ["k", "f_lo_khz", "f_hi_khz", "mz_lo", "mz_hi", "s_ppm", "n_chunks", "n_level",
                   "level_share", "n_scans", "scan_sd_ppm", "interp"]


# --------------------------------------------------------------------------- #
# the law, the bands, s <-> r
# --------------------------------------------------------------------------- #
def s_from_r(r_ppm):
    """Mascope's `reference.s_from_fudge_ratio`: the spacing anomaly s (ppm) of a chunk
    whose m/z carry the correction r (ppm of the law's m/z): s = ((1 + r e-6)^-1.5 - 1) e6."""
    return ((1.0 + np.asarray(r_ppm, dtype=float) * 1e-6) ** -1.5 - 1.0) * 1e6


def r_from_s(s_ppm):
    """The exact inverse of `s_from_r`: the stored relative m/z correction r (ppm) of a
    band whose spacing anomaly is s: r = ((1 + s e-6)^(-2/3) - 1) e6, about -(2/3) s."""
    return ((1.0 + np.asarray(s_ppm, dtype=float) * 1e-6) ** (-2.0 / 3.0) - 1.0) * 1e6


def law_khz(mz, B: float, C: float):
    """f_law(m) = sqrt((B + sqrt(B^2 + 4 m C)) / (2 m)) [law-kHz]; NaN for m <= 0 or NaN."""
    m = np.asarray(mz, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        f = np.sqrt((B + np.sqrt(B * B + 4.0 * m * C)) / (2.0 * m))
        return np.where(m > 0, f, np.nan)


def mz_of_f(f_khz, B: float, C: float):
    """The inverse law m = B/f^2 + C/f^4."""
    f = np.asarray(f_khz, dtype=float)
    return B / f ** 2 + C / f ** 4


def k_of_mz(mz, B: float, C: float, period: float, phase: float):
    """Band index k = floor((f_law(m) - phase) / period) as a FLOAT array (NaN where the
    law is undefined); `int` it only where finite."""
    return np.floor((law_khz(mz, B, C) - phase) / period)


def band_edges_mz(k, B: float, C: float, period: float, phase: float) -> tuple:
    """(mz_lo, mz_hi) of band k: m/z of its upper and lower frequency edge."""
    k = np.asarray(k, dtype=float)
    return mz_of_f(phase + period * (k + 1.0), B, C), mz_of_f(phase + period * k, B, C)


# --------------------------------------------------------------------------- #
# the robust local trend and the step table
# --------------------------------------------------------------------------- #
def _tricube(x):
    x = np.abs(np.asarray(x, dtype=float))
    return np.where(x < 1.0, (1.0 - x ** 3) ** 3, 0.0)


def _local_fit(k, r, rob, at, *, h: float, span: float, trusted=None) -> tuple:
    """The local polynomial through (k, r) at each `at`, weights tricube(|k' - k| / span)
    x `rob` over the bands within |k' - k| <= h. Its SUPPORT is the bands carrying a
    positive weight: a quadratic through >= LOESS_MIN_QUADRATIC of them, a line through
    >= LOESS_MIN_BANDS, NaN (no value this pass) with fewer -- a gross step zeroes its
    neighbours' weights in the first pass and a window left with one or two bands would
    otherwise fit a constant or a line through them, the step itself included.
    Returns (value, n bands in the window, n bands carrying weight, degree, two_sided:
    bands carrying weight, and `trusted` where given, lie below AND above `at`, reach: the
    distance in bands from `at` to the nearest such band, 0 for one itself, inf with
    none) per `at`."""
    at = np.asarray(at, dtype=float)
    trusted = np.ones(len(k), dtype=bool) if trusted is None else np.asarray(trusted, dtype=bool)
    out = np.full(at.shape, np.nan)
    n_win = np.zeros(at.shape, dtype=int)
    n_sup = np.zeros(at.shape, dtype=int)
    deg = np.zeros(at.shape, dtype=int)
    two_sided = np.zeros(at.shape, dtype=bool)
    reach = np.full(at.shape, np.inf)
    for i, a in enumerate(at):
        if not np.isfinite(a):
            continue
        dx = k - a
        sel = np.abs(dx) <= h
        n_win[i] = int(sel.sum())
        w = _tricube(dx[sel] / span) * rob[sel]
        pos = w > 0
        n_pos = int(pos.sum())
        n_sup[i] = n_pos
        held = pos & trusted[sel]
        if held.any():
            dpos = dx[sel][held]
            two_sided[i] = bool((dpos < 0).any() and (dpos > 0).any())
            reach[i] = float(np.abs(dpos).min())
        if n_pos < LOESS_MIN_BANDS:
            continue
        degree = 2 if n_pos >= LOESS_MIN_QUADRATIC else 1
        x, y, sw = dx[sel][pos], r[sel][pos], np.sqrt(w[pos])
        X = np.vander(x, degree + 1, increasing=True)
        coef = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)[0]
        out[i] = coef[0]
        deg[i] = degree
    return out, n_win, n_sup, deg, two_sided, reach


def loess_trend(k, r, *, k_eval=None, h: float = H_BANDS, span: float = TRICUBE_SPAN_BANDS,
                iterations: int = LOESS_ITERATIONS, c: float = BISQUARE_C,
                scale: float = MAD_TO_SIGMA, floor: float = SIGMA_FLOOR_PPM) -> dict:
    """The robust local quadratic (LOESS) through (k, r), the module's pinned
    convention (module note THE TREND IS LOCAL): at each k a weighted least-squares
    quadratic over |k' - k| <= h with tricube weights (1 - (|k' - k| / span)^3)^3 times
    bisquare robustness weights; `iterations` robustness iterations after the
    unweighted first fit, each with sigma = max(`scale` x median|residual| over every
    fitted band, `floor`) (scale 1.4826: the normal-consistent MAD, the raw median alone
    is NOT this convention; the floor, 0.1 ppm, keeps the fit robust on a table smooth
    to its rounding) and w = (1 - (res / (c sigma))^2)^2 for |res| < c sigma else 0; a
    band without a trend in a pass keeps its weight. A window's support is the bands
    carrying a positive weight: a quadratic through >= LOESS_MIN_QUADRATIC, a line
    through >= LOESS_MIN_BANDS, NaN with fewer (`_local_fit`). Returns dict(trend at
    `k_eval` (default k), weights (the final robustness weights per k), n_window (bands
    within h), n_support (bands carrying weight in the final pass), degree, two_sided
    (TRUSTED bands -- carrying weight and a trend of their own in the final pass; a band
    without one keeps the weight an earlier pass set -- lie on both sides of each
    `k_eval` within h), reach (the distance in bands to the nearest of them within h: 0
    on a trusted band itself, inf with none), sigma_ppm (the floored scale behind the
    final weights), scatter_ppm (scale x median|residual| unfloored: the table's own
    scatter about the trend; 0 when every residual is 0))."""
    k = np.asarray(k, dtype=float)
    r = np.asarray(r, dtype=float)
    if k.ndim != 1 or k.shape != r.shape or not len(k):
        raise ValueError("k and r must be one non-empty band table")
    if not (np.isfinite(k).all() and np.isfinite(r).all()):
        raise ValueError("k and r must be finite (pass the measured bands only)")
    rob = np.ones(len(k))
    sigma = scatter = 0.0
    for it in range(int(iterations) + 1):
        fitted = _local_fit(k, r, rob, k, h=h, span=span)[0]
        if it == iterations:
            break
        res = r - fitted
        ok = np.isfinite(res)
        scatter = float(scale * np.median(np.abs(res[ok]))) if ok.any() else 0.0
        sigma = max(scatter, float(floor))
        u = np.where(ok, res / (c * sigma), 0.0)
        new = np.where(np.abs(u) < 1.0, (1.0 - u ** 2) ** 2, 0.0)
        new[~ok] = rob[~ok]                     # a band without a trend of its own keeps its weight
        rob = new
    at = k if k_eval is None else np.asarray(k_eval, dtype=float)
    # a band without a trend of its own in the final pass keeps the weight an earlier pass set:
    # it is fitted through, but it is no band the trend reads its neighbours from (a band whose
    # trend returns in the final pass is trusted on such a weight too: KNOWN LIMIT (6))
    trusted = (rob > 0) & np.isfinite(fitted)
    trend, n_win, n_sup, deg, two_sided, reach = _local_fit(k, r, rob, at, h=h, span=span, trusted=trusted)
    return {"trend": trend, "weights": rob, "n_window": n_win, "n_support": n_sup, "degree": deg,
            "two_sided": two_sided, "reach": reach, "sigma_ppm": sigma, "scatter_ppm": scatter}


def step_table(k, s_ppm, measured) -> dict:
    """r_k, the LOESS trend over the measured bands (evaluated at every band given) and
    d_k = r_k - trend(k) on the measured bands whose trend reads them from its support,
    0 on the others (module note THE TREND IS LOCAL): the band carries weight itself,
    trusted bands (carrying weight and a trend of their own, `loess_trend`) lie on both
    sides of it in its window, or one lies within LOESS_REACH_BANDS of it. A measured
    band whose trend comes from one side only and from further away keeps d only within
    the robust cut-off (BISQUARE_C x sigma: it is trend); beyond it the band is
    EXTRAPOLATED: no d, unmeasured here. Returns dict(r, trend, d, weights, measured (the
    bands d applies to: a measured band whose window supports no trend, or that is
    extrapolated, is unmeasured here), extrapolated, n_window, n_support, degree,
    sigma_ppm, scatter_ppm)."""
    k = np.asarray(k, dtype=float)
    s = np.asarray(s_ppm, dtype=float)
    measured = np.asarray(measured, dtype=bool)
    if k.ndim != 1 or k.shape != s.shape or k.shape != measured.shape:
        raise ValueError("k, s and measured must be one band table")
    r = np.where(np.isfinite(s), r_from_s(np.where(np.isfinite(s), s, 0.0)), np.nan)
    meas = measured & np.isfinite(r)
    if not meas.any():
        raise ValueError("no measured band")
    lo = loess_trend(k[meas], r[meas], k_eval=k)
    trend = lo["trend"]
    weights = np.zeros(len(k))
    weights[meas] = lo["weights"]
    with_trend = meas & np.isfinite(trend)
    one_sided = with_trend & ~(lo["two_sided"] | (lo["reach"] <= LOESS_REACH_BANDS))
    # an extrapolated trend may confirm a band as trend (within the cut-off), never make it a step
    with np.errstate(invalid="ignore"):
        extrapolated = one_sided & ~(np.abs(r - trend) < BISQUARE_C * lo["sigma_ppm"])
    meas_out = with_trend & ~extrapolated
    d = np.where(meas_out, r - trend, 0.0)
    d = np.where(np.isfinite(d), d, 0.0)
    weights = np.where(meas_out, weights, 0.0)
    return {"r": r, "trend": trend, "d": d, "weights": weights, "measured": meas_out,
            "extrapolated": extrapolated, "n_window": lo["n_window"], "n_support": lo["n_support"],
            "degree": lo["degree"], "sigma_ppm": lo["sigma_ppm"], "scatter_ppm": lo["scatter_ppm"]}


def weighted_median(values, weights) -> float:
    """The weighted median: the smallest value whose cumulative weight reaches half the
    total (ties -> the lower value). Non-finite or non-positive weights are left out."""
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    ok = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if not ok.any():
        raise ValueError("weighted_median needs a finite value with a positive weight")
    v, w = v[ok], w[ok]
    order = np.argsort(v, kind="stable")
    v, w = v[order], w[order]
    cum = np.cumsum(w)
    i = int(np.searchsorted(cum, 0.5 * cum[-1], side="left"))
    return float(v[min(i, len(v) - 1)])


# --------------------------------------------------------------------------- #
# reading a served document
# --------------------------------------------------------------------------- #
def document_sha1(doc) -> str:
    """sha1 of the document's canonical JSON (sorted keys, no spaces)."""
    text = json.dumps(doc, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _num(x):
    try:
        v = float(x)
    except (TypeError, ValueError, OverflowError):
        return np.nan
    return v


@dataclass
class Fingerprint:
    """One usable document, read for one acquisition mode: the grid, the law's B and C,
    the per-band s and which bands are measured, plus the document's OWN step table
    (what the frame would be from this document alone; a group pools s first)."""
    mode_id: str
    polarity: str
    resolution: object
    scan_range: object
    B: float
    C: float
    C_over_B2: float
    period: float
    phase: float
    n_hits: float
    edge_uncertainty_khz: float
    k0: int
    s: np.ndarray                    # s_ppm per band (NaN where the table has none)
    measured: np.ndarray             # bool: interp False, n_level >= 3, level_share >= 0.5, s finite, and
                                     # a trend that reads the band (step_table): the bands d applies to
    interp: np.ndarray               # bool: the table's own interpolation flag (or no row)
    r: np.ndarray
    trend: np.ndarray
    d: np.ndarray                    # 0 on unmeasured bands
    weights: np.ndarray
    meta: dict = field(default_factory=dict)

    @property
    def k1(self) -> int:
        return self.k0 + len(self.s) - 1

    @property
    def ks(self) -> np.ndarray:
        return np.arange(self.k0, self.k1 + 1)

    @property
    def n_measured(self) -> int:
        return int(self.measured.sum())

    @property
    def h_max(self) -> float:
        """The largest |d| of the document's own table over its measured bands (0 without any)."""
        return float(np.abs(self.d[self.measured]).max()) if self.measured.any() else 0.0

    def band_centre_khz(self) -> np.ndarray:
        """The centre frequency of every band on the document's own grid."""
        return self.phase + self.period * (self.ks + 0.5)

    def frame_item(self, sample_item_id=None, inner=None, policy: str = OUTSIDE_POLICY) -> "FrameItem":
        """The frame of this document alone: a one-member calibration group."""
        sid = "item" if sample_item_id is None else str(sample_item_id)
        group = CalibrationGroup.from_fingerprints({sid: self})
        it = group.frame_item(sid, inner=inner, policy=policy)
        if sample_item_id is None:
            it.sample_item_id = None
        return it


@dataclass
class Loaded:
    """What `load_document` found: the fingerprint, or why the document was refused;
    `notes` carries the advisory flags; `sha1` the served document's digest."""
    fingerprint: "Fingerprint | None"
    why: "str | None"
    notes: list = field(default_factory=list)
    status: "str | None" = None
    mode_id: "str | None" = None
    sha1: "str | None" = None

    @property
    def ok(self) -> bool:
        return self.fingerprint is not None and self.why is None

    def __iter__(self):
        yield self.fingerprint
        yield self.why

    def as_dict(self) -> dict:
        fp = self.fingerprint
        out = {"usable": self.ok, "status": self.status, "mode_id": self.mode_id, "sha1": self.sha1,
               "notes": list(self.notes)}
        if self.why is not None:
            out["why"] = self.why
        if fp is not None:
            out.update(n_bands=int(len(fp.s)), n_measured=fp.n_measured,
                       n_extrapolated=fp.meta.get("n_extrapolated"), extrapolated_k=fp.meta.get("extrapolated_k"),
                       h_max_ppm=round(fp.h_max, 4),
                       k_range=[int(fp.k0), int(fp.k1)], resolution=fp.resolution,
                       scan_range=fp.scan_range, C_over_B2=fp.C_over_B2,
                       grid={"period_khz": fp.period, "phase_khz": fp.phase,
                             "source": fp.meta.get("grid_source"), "n_hits": fp.n_hits,
                             "edge_uncertainty_khz": fp.edge_uncertainty_khz},
                       lock=fp.meta.get("lock"))
        return _jsonable(out)


#: the fields the schema gives as JSON arrays, where the document has them: (where, key)
_LIST_FIELDS = (("document", "modes"), ("document", "flags"), ("item", "streams"), ("item", "scans"),
                ("mode", "flags"), ("segments", "columns"), ("segments", "rows"))
#: a segment table's band index and span: a served table holds a few dozen bands, k in the
#: tens; a k or a span past this is not a band table (and would size the arrays it is read into)
MAX_TABLE_BANDS = 10_000


def document_shape_why(doc) -> "str | None":
    """Why `doc` does not have the served schema's shape, or None: not an object, or a
    field the schema gives as a list (`modes`, `flags`, `item.streams`, `item.scans`, a
    mode's `flags`, its `segments.columns` / `segments.rows`) holding something else (a
    count, a string, an object). `load_document` refuses such a document with this
    reason instead of reading it; io_mascope.fetch_axis_fingerprint does not cache it.
    Reads no values: a document of the right shape may still be refused for its content."""
    if not isinstance(doc, dict):
        return "not a document object"
    item = doc.get("item") if isinstance(doc.get("item"), dict) else {}
    modes = doc.get("modes")
    blocks = {"document": [doc], "item": [item],
              "mode": [m for m in (modes if isinstance(modes, list) else []) if isinstance(m, dict)]}
    blocks["segments"] = [m["segments"] for m in blocks["mode"] if isinstance(m.get("segments"), dict)]
    for where, key in _LIST_FIELDS:
        for block in blocks[where]:
            value = block.get(key)
            if value is not None and not isinstance(value, list):
                name = key if where == "document" else f"{where}.{key}"
                return f"malformed document: {name} is {type(value).__name__}, not a list"
    return None


def load_document(doc, *, polarity: str) -> Loaded:
    """Read a served fingerprint document for a batch of `polarity` ('+' or '-') into a
    `Fingerprint`, or refuse it with a reason (module note: WHAT IS REFUSED). The mode
    is `item.mode_id` when the per-item route served the document, else the file's
    `dominant_mode`. Never raises on a document's content: a field of an unexpected
    type (`document_shape_why`) and a band index no table has are refusals too."""
    notes: list = []
    status = doc.get("status") if isinstance(doc, dict) else None
    sha1 = document_sha1(doc) if isinstance(doc, dict) else None

    def refuse(why, mode_id=None):
        return Loaded(None, why, notes, status, mode_id, sha1)

    if not isinstance(doc, dict):
        return refuse("no document")
    if doc.get("schema") != SCHEMA:
        return refuse(f"schema {doc.get('schema')!r} is not {SCHEMA!r}")
    version = doc.get("version")
    if not isinstance(version, int) or isinstance(version, bool) or version > VERSION or version < 1:
        return refuse(f"schema version {version!r}: this reader knows version {VERSION}")
    if status != "ok":
        return refuse(f"status {status!r}" + (f" ({doc.get('error')})" if status == "error" else ""))
    shape = document_shape_why(doc)
    if shape is not None:
        return refuse(shape)
    item = doc.get("item") if isinstance(doc.get("item"), dict) else {}
    mode_id = item.get("mode_id") or doc.get("dominant_mode")
    if not mode_id:
        return refuse("no mode: the item's scans belong to no mode and the file has no dominant mode")
    modes = [m for m in (doc.get("modes") or []) if isinstance(m, dict) and m.get("mode_id") == mode_id]
    if not modes:
        return refuse(f"mode {mode_id!r} is not among the document's modes", mode_id)
    mode = modes[0]
    if polarity not in ("+", "-"):
        return refuse(f"batch polarity {polarity!r} is neither '+' nor '-'", mode_id)
    if mode.get("polarity") != polarity:
        return refuse(f"mode polarity {mode.get('polarity')!r} is not the batch's {polarity!r}", mode_id)
    if item and item.get("polarity") not in (None, polarity):
        return refuse(f"item polarity {item.get('polarity')!r} is not the batch's {polarity!r}", mode_id)
    streams = item.get("streams") if item else None
    stream_key = str(item.get("stream_key") or "") if item else ""
    if (streams is not None and len(streams) != 1) or stream_key.startswith("composite"):
        n = len(streams) if streams is not None else "several"
        return refuse(f"a stitched sample ({n} streams): its peaks sit on several factors", mode_id)
    flags = [str(f) for f in (mode.get("flags") or [])]
    bad = [f for f in REFUSING_FLAGS if f in flags]
    if bad:
        return refuse("mode flagged " + ", ".join(bad), mode_id)
    notes += [f for f in ADVISORY_FLAGS if f in flags]
    notes += [f"file:{f}" for f in (doc.get("flags") or []) if isinstance(f, str)]
    grid = mode.get("grid")
    if not isinstance(grid, dict):
        return refuse("no band grid", mode_id)
    if grid.get("source") != "measured":
        return refuse(f"grid source {grid.get('source')!r}: the band edges are a guess", mode_id)
    period, phase = _num(grid.get("period_khz")), _num(grid.get("phase_khz"))
    if not (np.isfinite(period) and period > 0 and np.isfinite(phase)):
        return refuse("grid period / phase unreadable", mode_id)
    n_hits = _num(grid.get("n_hits"))
    n_hits = n_hits if np.isfinite(n_hits) and n_hits > 0 else 1.0
    unc = _num(grid.get("edge_uncertainty_khz"))
    unc = unc if np.isfinite(unc) and unc >= 0 else DEFAULT_EDGE_UNCERTAINTY_KHZ
    cal = mode.get("calibration") if isinstance(mode.get("calibration"), dict) else {}
    B, C = _num(cal.get("B_median")), _num(cal.get("C_median"))
    if not (np.isfinite(B) and B > 0 and np.isfinite(C)):
        return refuse("calibration B_median / C_median unreadable", mode_id)
    c_over_b2 = _num(cal.get("C_over_B2"))
    if not np.isfinite(c_over_b2):
        c_over_b2 = C / B ** 2
    seg = mode.get("segments")
    if not isinstance(seg, dict) or not seg.get("rows"):
        return refuse("no segment table", mode_id)
    columns = list(seg.get("columns") or [])
    missing = [c for c in SEGMENT_COLUMNS if c not in columns]
    if missing:
        return refuse(f"segment table lacks columns {missing}", mode_id)
    at = {c: columns.index(c) for c in SEGMENT_COLUMNS}
    rows = [r for r in seg["rows"] if isinstance(r, (list, tuple)) and len(r) >= len(columns)]
    if not rows:
        return refuse("segment table rows unreadable", mode_id)
    try:
        ks = np.array([int(r[at["k"]]) for r in rows])
    except (TypeError, ValueError, OverflowError):
        return refuse("segment table k unreadable", mode_id)
    k0, k1 = int(ks.min()), int(ks.max())
    if max(abs(k0), abs(k1), k1 - k0) > MAX_TABLE_BANDS:
        return refuse(f"segment table k past +-{MAX_TABLE_BANDS} or spanning more bands: not a band table", mode_id)
    n = k1 - k0 + 1
    s = np.full(n, np.nan)
    n_level = np.zeros(n)
    share = np.zeros(n)
    interp = np.ones(n, dtype=bool)          # a band the table does not list counts as interpolated
    present = np.zeros(n, dtype=bool)
    for r in rows:
        i = int(r[at["k"]]) - k0
        present[i] = True
        s[i] = _num(r[at["s_ppm"]])
        n_level[i] = _num(r[at["n_level"]])
        share[i] = _num(r[at["level_share"]])
        interp[i] = bool(r[at["interp"]])
    measured = present & ~interp & (n_level >= MIN_N_LEVEL) & (share >= MIN_LEVEL_SHARE) & np.isfinite(s)
    if int(measured.sum()) < MIN_MEASURED_SEGMENTS:
        return refuse(f"{int(measured.sum())} measured bands, fewer than {MIN_MEASURED_SEGMENTS}", mode_id)
    table = step_table(np.arange(k0, k1 + 1), s, measured)
    measured = table["measured"]
    if int(measured.sum()) < MIN_MEASURED_SEGMENTS:
        return refuse(f"{int(measured.sum())} bands with a trend to read d from, fewer than "
                      f"{MIN_MEASURED_SEGMENTS}", mode_id)
    h_max = float(np.abs(table["d"][measured]).max())
    if h_max > H_MAX_PPM:
        return refuse(f"the largest band step is {h_max:.1f} ppm, above {H_MAX_PPM:g}: a broken "
                      "calibration to fix at the instrument", mode_id)
    lock = item.get("lock") if isinstance(item.get("lock"), dict) else None
    if lock is None and isinstance(mode.get("lock"), dict):
        lock = mode["lock"]
    scan_range = mode.get("scan_range")
    if isinstance(scan_range, (list, tuple)):
        scan_range = [_num(x) for x in scan_range]
    meta = {
        "grid_source": grid.get("source"),
        "grid_weak": "grid_weak" in flags,
        "mode_flags": flags,
        "lock": {"m_lock_eff": (lock or {}).get("m_lock_eff"),
                 "s_lock_eff_ppm": (lock or {}).get("s_lock_eff_ppm")},
        "item": {k: item.get(k) for k in ("sample_item_id", "sample_file_id", "stream_key",
                                          "mz_calibration_factor", "file_mz_calibration_factor")
                 if item.get(k) is not None},
        "n_item_scans": len(item.get("scans") or []) if item else None,
        "sigma_ppm": table["sigma_ppm"],
        "scatter_ppm": table["scatter_ppm"],
        "n_extrapolated": int(table["extrapolated"].sum()),
        "extrapolated_k": [int(k) for k in np.arange(k0, k1 + 1)[table["extrapolated"]]],
        "document_sha1": sha1,
    }
    fp = Fingerprint(mode_id=str(mode_id), polarity=polarity, resolution=mode.get("resolution"),
                     scan_range=scan_range, B=B, C=C, C_over_B2=float(c_over_b2), period=period, phase=phase,
                     n_hits=float(n_hits), edge_uncertainty_khz=float(unc), k0=k0, s=s, measured=measured,
                     interp=interp | ~present, r=table["r"], trend=table["trend"], d=table["d"],
                     weights=table["weights"], meta=meta)
    return Loaded(fp, None, notes, status, str(mode_id), sha1)


# --------------------------------------------------------------------------- #
# calibration groups: one grid, one table
# --------------------------------------------------------------------------- #
def align_k(fp: Fingerprint, period: float, phase: float) -> np.ndarray:
    """Each band of `fp` on the grid (period, phase): the band index its centre frequency
    falls in. The labels shift by one when one grid's phase wrapped past 0."""
    return np.floor((fp.band_centre_khz() - phase) / period).astype(int)


def same_calibration(a: Fingerprint, b: Fingerprint) -> tuple:
    """(True, how) when two documents describe one calibration: same mode polarity,
    resolution and scan range, C_over_B2 within C_OVER_B2_TOL, and |s_a - s_b| <=
    S_AGREE_PPM on every band measured in both (aligned by frequency; at least one such
    band); else (False, why)."""
    if a.polarity != b.polarity:
        return False, f"polarity {a.polarity!r} vs {b.polarity!r}"
    if a.resolution != b.resolution:
        return False, f"resolution {a.resolution!r} vs {b.resolution!r}"
    ra, rb = a.scan_range, b.scan_range
    same_range = (ra is None and rb is None) or (
        isinstance(ra, (list, tuple)) and isinstance(rb, (list, tuple)) and len(ra) == len(rb)
        and all(abs(_num(x) - _num(y)) <= 1e-6 for x, y in zip(ra, rb)))
    if not same_range:
        return False, f"scan range {ra!r} vs {rb!r}"
    if not (np.isfinite(a.C_over_B2) and np.isfinite(b.C_over_B2)) or abs(a.C_over_B2 - b.C_over_B2) > C_OVER_B2_TOL:
        return False, f"C/B^2 differ by {abs(a.C_over_B2 - b.C_over_B2):.2e} (> {C_OVER_B2_TOL:g})"
    Kb = align_k(b, a.period, a.phase)
    diffs, ks = [], []
    for i, K in enumerate(Kb):
        j = int(K) - a.k0
        if b.measured[i] and 0 <= j < len(a.s) and a.measured[j]:
            diffs.append(abs(float(a.s[j]) - float(b.s[i])))
            ks.append(int(K))
    if not diffs:
        return False, "no band measured in both"
    worst = int(np.argmax(diffs))
    if diffs[worst] > S_AGREE_PPM:
        return False, f"s differs by {diffs[worst]:.3f} ppm on band k={ks[worst]} (> {S_AGREE_PPM:g})"
    return True, f"{len(diffs)} common bands, s within {max(diffs):.4f} ppm"


def pool_grid(fps: list) -> dict:
    """One grid for a calibration group: the n_hits-weighted median period, the phases
    first brought within half a period of the member with most hits, their n_hits-
    weighted median taken modulo the period; `zone_khz` = max(members'
    edge_uncertainty_khz, half the spread of the wrapped phases)."""
    hits = np.array([fp.n_hits for fp in fps], dtype=float)
    periods = np.array([fp.period for fp in fps], dtype=float)
    phases = np.array([fp.phase for fp in fps], dtype=float)
    ref = int(np.argmax(hits))
    period = weighted_median(periods, hits)
    wrapped = phases - period * np.round((phases - phases[ref]) / period)
    phase_raw = weighted_median(wrapped, hits)
    phase = float(phase_raw % period)
    unc = np.array([fp.edge_uncertainty_khz for fp in fps], dtype=float)
    zone = float(max(unc.max(), 0.5 * (wrapped.max() - wrapped.min())))
    return {"period_khz": period, "phase_khz": phase, "zone_khz": zone, "reference": ref,
            "periods_khz": periods.tolist(), "phases_khz": phases.tolist(), "phases_wrapped_khz": wrapped.tolist(),
            "n_hits": hits.tolist(), "edge_uncertainty_khz": unc.tolist(),
            "phase_spread_khz": float(wrapped.max() - wrapped.min())}


@dataclass
class CalibrationGroup:
    """The items of one calibration: one grid, one pooled band table (per band the
    median s over the members measuring it; a member contributes the bands its own
    table reads a d on), one B and C, one edge zone, and the LOESS trend and d over the
    pooled table (module note ONE TABLE PER CALIBRATION GROUP). `frame_item(sid)` is
    the FrameItem every member shares the arrays of."""
    members: list
    mode_id: str
    polarity: str
    resolution: object
    scan_range: object
    B: float
    C: float
    C_over_B2: float
    period: float
    phase: float
    zone_khz: float
    k0: int
    s: np.ndarray                    # pooled s per band (NaN where no member measures it)
    n_measuring: np.ndarray          # members measuring each band
    s_spread: np.ndarray             # max - min of the members' s per band (0 with one member)
    r: np.ndarray
    trend: np.ndarray
    d: np.ndarray
    weights: np.ndarray
    measured: np.ndarray
    extrapolated: np.ndarray         # bool: a step read only by a trend from one side, too far past its
                                     # support: no d
    meta: dict = field(default_factory=dict)
    index: int = 0

    @classmethod
    def from_fingerprints(cls, fps: dict, *, index: int = 0) -> "CalibrationGroup":
        """Pool the documents {sample id: Fingerprint} of ONE calibration (checked by the
        caller: `same_calibration`). Raises ValueError when a member's bands do not map
        one-to-one onto the pooled grid or no band is measured."""
        if not fps:
            raise ValueError("a calibration group needs at least one document")
        sids = [str(s) for s in fps]
        members = [fps[s] for s in fps]
        grid = pool_grid(members)
        period, phase = grid["period_khz"], grid["phase_khz"]
        B = float(np.median([fp.B for fp in members]))
        C = float(np.median([fp.C for fp in members]))
        Ks = []
        for sid, fp in zip(sids, members):
            K = align_k(fp, period, phase)
            if len(K) > 1 and not np.all(np.diff(K) == 1):
                raise ValueError(f"item {sid}: its bands do not map one-to-one onto the group's grid")
            Ks.append(K)
        k0 = int(min(K.min() for K in Ks))
        k1 = int(max(K.max() for K in Ks))
        n = k1 - k0 + 1
        stack = np.full((len(members), n), np.nan)
        for i, (fp, K) in enumerate(zip(members, Ks)):
            idx = K - k0
            stack[i, idx[fp.measured]] = fp.s[fp.measured]
        have = np.isfinite(stack)
        n_meas = have.sum(axis=0)
        s_pool = np.where(n_meas > 0, np.nanmedian(np.where(n_meas > 0, stack, 0.0), axis=0), np.nan)
        hi = np.max(np.where(have, stack, -np.inf), axis=0)
        lo = np.min(np.where(have, stack, np.inf), axis=0)
        spread = np.where(n_meas > 1, hi - lo, 0.0)
        table = step_table(np.arange(k0, k1 + 1), s_pool, n_meas > 0)
        first = members[0]
        meta = {"grid": dict(grid, members=[{"sample_item_id": sid, "period_khz": fp.period, "phase_khz": fp.phase,
                                             "phase_wrapped_khz": grid["phases_wrapped_khz"][i],
                                             "n_hits": fp.n_hits, "edge_uncertainty_khz": fp.edge_uncertainty_khz,
                                             "k_shift": int(K[0] - fp.k0), "n_measured": fp.n_measured}
                                            for i, (sid, fp, K) in enumerate(zip(sids, members, Ks))]),
                "sigma_ppm": table["sigma_ppm"], "scatter_ppm": table["scatter_ppm"],
                "B_members": [fp.B for fp in members], "C_members": [fp.C for fp in members]}
        return cls(members=sids, mode_id=first.mode_id, polarity=first.polarity, resolution=first.resolution,
                   scan_range=first.scan_range, B=B, C=C,
                   C_over_B2=float(np.median([fp.C_over_B2 for fp in members])), period=period, phase=phase,
                   zone_khz=grid["zone_khz"], k0=k0, s=s_pool, n_measuring=n_meas, s_spread=spread,
                   r=table["r"], trend=table["trend"], d=table["d"], weights=table["weights"],
                   measured=table["measured"], extrapolated=table["extrapolated"], meta=meta, index=index)

    @property
    def k1(self) -> int:
        return self.k0 + len(self.d) - 1

    @property
    def n_measured(self) -> int:
        return int(self.measured.sum())

    @property
    def n_unmeasured(self) -> int:
        return int(len(self.d) - self.measured.sum())

    @property
    def n_extrapolated(self) -> int:
        return int(self.extrapolated.sum())

    @property
    def h_max(self) -> float:
        return float(np.abs(self.d[self.measured]).max()) if self.measured.any() else 0.0

    @property
    def s_spread_max(self) -> float:
        return float(self.s_spread.max()) if len(self.s_spread) else 0.0

    def frame_item(self, sample_item_id, inner=None, policy: str = OUTSIDE_POLICY, *, meta=None) -> "FrameItem":
        """The member's FrameItem: the group's grid, B, C, zone and table (shared arrays)."""
        m = {"mode_id": self.mode_id, "polarity": self.polarity, "group": int(self.index),
             "n_members": len(self.members), "n_bands": int(len(self.d)), "n_measured": self.n_measured,
             "n_extrapolated": self.n_extrapolated, "h_max_ppm": round(self.h_max, 4),
             "zone_khz": float(self.zone_khz)}
        if meta:
            m.update(meta)
        return FrameItem(sample_item_id=str(sample_item_id), B=self.B, C=self.C, period=self.period,
                         phase=self.phase, k0=self.k0, d=self.d, measured=self.measured, zone_khz=self.zone_khz,
                         inner=inner, policy=policy, meta=m)

    def bands(self) -> list:
        """Per band: k, mz_lo, mz_hi, s, r, trend, weight, d, measured, extrapolated (a
        step read only by a trend from one side, too far past its support: no d),
        n_measuring, s_spread."""
        ks = np.arange(self.k0, self.k1 + 1)
        lo, hi = band_edges_mz(ks, self.B, self.C, self.period, self.phase)
        return [{"k": int(k), "mz_lo": round(float(a), 4), "mz_hi": round(float(b), 4),
                 "s_ppm": None if not np.isfinite(sv) else round(float(sv), 4),
                 "r_ppm": None if not np.isfinite(rv) else round(float(rv), 4),
                 "trend_ppm": None if not np.isfinite(tv) else round(float(tv), 4),
                 "weight": round(float(wv), 4), "d_ppm": round(float(dv), 4), "measured": bool(mv),
                 "extrapolated": bool(xv), "n_measuring": int(nm), "s_spread_ppm": round(float(sp), 4)}
                for k, a, b, sv, rv, tv, wv, dv, mv, xv, nm, sp in zip(
                    ks, lo, hi, self.s, self.r, self.trend, self.weights, self.d, self.measured,
                    self.extrapolated, self.n_measuring, self.s_spread)]

    def as_dict(self) -> dict:
        """JSON-safe record: members, grid (period, phase, zone, the members' grids), B, C,
        bands measured / unmeasured / extrapolated, s spread, and the per-band table."""
        g = self.meta.get("grid", {})
        return _jsonable({
            "index": int(self.index), "members": list(self.members), "n_members": len(self.members),
            "mode_id": self.mode_id, "polarity": self.polarity, "resolution": self.resolution,
            "scan_range": self.scan_range, "B": self.B, "C": self.C, "C_over_B2": self.C_over_B2,
            "grid": {"period_khz": self.period, "phase_khz": self.phase, "zone_khz": self.zone_khz,
                     "phase_spread_khz": g.get("phase_spread_khz"), "reference_member": g.get("reference"),
                     "members": g.get("members")},
            "k_range": [int(self.k0), int(self.k1)], "n_bands": int(len(self.d)),
            "n_measured": self.n_measured, "n_unmeasured": self.n_unmeasured,
            "n_extrapolated": self.n_extrapolated,
            "h_max_ppm": round(self.h_max, 4), "s_spread_max_ppm": round(self.s_spread_max, 4),
            "trend_sigma_ppm": self.meta.get("sigma_ppm"), "trend_scatter_ppm": self.meta.get("scatter_ppm"),
            "reach_mz": [round(float(x), 3) for x in band_edges_mz(self.k1, self.B, self.C, self.period, self.phase)[:1]]
                        + [round(float(band_edges_mz(self.k0, self.B, self.C, self.period, self.phase)[1]), 3)],
            "bands": self.bands(),
            "separated_by": self.meta.get("separated_by")})


def group_fingerprints(fps: dict) -> list:
    """The calibration groups of {sample id: Fingerprint}, in the items' order: a
    document joins the first group whose every member it agrees with
    (`same_calibration`), else starts a new one (its `meta['separated_by']` says why it
    did not join the first group)."""
    groups: list = []                         # [(sids, fingerprints dict, why_separate)]
    for sid, fp in fps.items():
        placed = False
        why_first = None
        for g in groups:
            verdicts = [same_calibration(other, fp) for other in g[1].values()]
            if all(v[0] for v in verdicts):
                g[1][str(sid)] = fp
                placed = True
                break
            if why_first is None:
                bad = next(v for v in verdicts if not v[0])
                why_first = f"item {sid} vs item {next(iter(g[1]))}: {bad[1]}"
        if not placed:
            groups.append(([], {str(sid): fp}, why_first))
    out = []
    for i, (_, members, why) in enumerate(groups):
        cg = CalibrationGroup.from_fingerprints(members, index=i)
        if why is not None:
            cg.meta["separated_by"] = why
        out.append(cg)
    return out


# --------------------------------------------------------------------------- #
# one item's frame
# --------------------------------------------------------------------------- #
@dataclass
class FrameItem:
    """One sample item's map: the band steps `d` over bands k0..k1 (0 on unmeasured ones)
    on the group's grid, the edge zone in frequency, an inner lock model fitted in the
    W frame (`batch.axislock.AxisModel`, or None when there is nothing smooth to
    correct) and the OUTSIDE_POLICY. Duck-types AxisModel: m/z_corr = m/z (1 -
    predict(m/z) e-6), nothing outside `in_scope`."""
    sample_item_id: "str | None"
    B: float
    C: float
    period: float
    phase: float
    k0: int
    d: np.ndarray
    measured: np.ndarray
    zone_khz: float = DEFAULT_EDGE_UNCERTAINTY_KHZ
    inner: object = None
    policy: str = OUTSIDE_POLICY
    meta: dict = field(default_factory=dict)
    stats: dict = field(default_factory=dict)
    K: int = -1                          # not a Chebyshev wave (massqc dispatches on in_scope)

    def __post_init__(self):
        self.d = np.asarray(self.d, dtype=float)
        self.measured = np.asarray(self.measured, dtype=bool)
        if self.d.ndim != 1 or self.d.shape != self.measured.shape or not len(self.d):
            raise ValueError("d and measured must be one non-empty band table")
        if self.policy not in POLICIES:
            raise ValueError(f"policy {self.policy!r} is not one of {POLICIES}")
        self.k0 = int(self.k0)
        self.zone_khz = float(self.zone_khz)
        if not (np.isfinite(self.zone_khz) and self.zone_khz >= 0):
            raise ValueError(f"zone_khz {self.zone_khz!r} must be a finite non-negative frequency")

    # --- bands --------------------------------------------------------------------
    @property
    def k1(self) -> int:
        return self.k0 + len(self.d) - 1

    @property
    def h_max(self) -> float:
        return float(np.abs(self.d[self.measured]).max()) if self.measured.any() else 0.0

    @property
    def reach(self) -> tuple:
        """The m/z range the band table covers (its unmeasured bands included)."""
        lo, _ = band_edges_mz(self.k1, self.B, self.C, self.period, self.phase)
        _, hi = band_edges_mz(self.k0, self.B, self.C, self.period, self.phase)
        return float(lo), float(hi)

    @property
    def edge_indices(self) -> np.ndarray:
        """The edge indices j of the table, k0 .. k1 + 1 (edge j at frequency phase + j period)."""
        return np.arange(self.k0, self.k1 + 2)

    @property
    def edges(self) -> np.ndarray:
        """Every band edge of the table in m/z, ascending (k1 + 1 edges ... k0's upper edge)."""
        return np.sort(mz_of_f(self.phase + self.period * self.edge_indices.astype(float), self.B, self.C))

    @property
    def zone_bounds(self) -> np.ndarray:
        """The m/z where the edge zones begin and end (every edge's frequency +- zone_khz),
        ascending: where the frame's map jumps by half a step."""
        f = self.phase + self.period * self.edge_indices.astype(float)
        return np.sort(np.concatenate([mz_of_f(f - self.zone_khz, self.B, self.C),
                                       mz_of_f(f + self.zone_khz, self.B, self.C)]))

    def zone_ppm_at(self, edge_mz) -> np.ndarray:
        """The edge zone as ppm of m/z at an edge: 2 zone_khz / f(edge) e6 (dm/m = 2 df/f)."""
        return 2.0 * self.zone_khz / law_khz(edge_mz, self.B, self.C) * 1e6

    def band_width_at(self, edge_mz) -> np.ndarray:
        """One band width in m/z at an edge: the wider of the two bands it separates."""
        e = np.asarray(edge_mz, dtype=float)
        k_above = np.floor((law_khz(e * (1 + 1e-9), self.B, self.C) - self.phase) / self.period)
        lo_a, hi_a = band_edges_mz(k_above, self.B, self.C, self.period, self.phase)
        lo_b, hi_b = band_edges_mz(k_above + 1, self.B, self.C, self.period, self.phase)
        return np.maximum(hi_a - lo_a, hi_b - lo_b)

    def edge_steps(self) -> list:
        """The step the frame predicts at every band edge of the table: per edge (ascending
        m/z) the band index and d on each side and `step_ppm` = d(higher m/z side) -
        d(lower m/z side), the jump of the served axis's error going up in m/z; 'measured'
        when both sides are measured bands (a side off the table or unmeasured has d = 0)."""
        j = self.edge_indices[::-1].astype(float)   # ascending m/z = descending edge index
        E = mz_of_f(self.phase + self.period * j, self.B, self.C)
        d_lo, m_lo, _ = self._d_at_k(j)             # band j lies below the edge in m/z
        d_hi, m_hi, _ = self._d_at_k(j - 1.0)       # band j - 1 above it
        return [{"edge_mz": float(e), "k_below": int(k), "k_above": int(k) - 1, "d_below": float(a),
                 "d_above": float(b), "step_ppm": float(b - a), "measured": bool(ma and mb)}
                for e, k, a, b, ma, mb in zip(E, j, d_lo, d_hi, m_lo, m_hi)]

    def k_of(self, mz) -> np.ndarray:
        """Band index as a float array (NaN where the law is undefined)."""
        return k_of_mz(mz, self.B, self.C, self.period, self.phase)

    def _d_at_k(self, kf) -> tuple:
        """(d, measured, in_table) at float band indices: d = 0 and unmeasured off the table."""
        kf = np.asarray(kf, dtype=float)
        ok = np.isfinite(kf)
        idx = np.zeros(kf.shape, dtype=int)
        idx[ok] = (kf[ok] - self.k0).astype(int)
        inside = ok & (idx >= 0) & (idx < len(self.d))
        d = np.zeros(kf.shape)
        meas = np.zeros(kf.shape, dtype=bool)
        d[inside] = self.d[idx[inside]]
        meas[inside] = self.measured[idx[inside]]
        return d, meas, inside

    def _locate(self, mz) -> dict:
        """Where each m/z sits: its band k, whether its law frequency lies within zone_khz
        of a table edge (`amb`; then `nb` is the band across that edge and `j` the edge
        index), and the effective d / measured / in-reach under the mean-of-neighbours
        rule (module note THE EDGE ZONE IS IN FREQUENCY)."""
        m = np.asarray(mz, dtype=float)
        with np.errstate(invalid="ignore", divide="ignore"):
            u = (law_khz(m, self.B, self.C) - self.phase) / self.period
        ok = np.isfinite(u)
        k = np.where(ok, np.floor(np.where(ok, u, 0.0)), np.nan)
        frac = np.where(ok, u - k, 0.5)
        near_lo = frac < 0.5                        # the nearer edge is the band's low-frequency one (index k)
        j = np.where(near_lo, k, k + 1.0)
        dist = np.where(near_lo, frac, 1.0 - frac) * self.period
        nb = np.where(near_lo, k - 1.0, k + 1.0)
        in_table_edge = ok & (j >= self.k0) & (j <= self.k1 + 1)
        amb = in_table_edge & (dist <= self.zone_khz)
        d_k, meas_k, in_k = self._d_at_k(k)
        d_n, meas_n, in_n = self._d_at_k(nb)
        return {"m": m, "k": k, "j": j, "nb": nb, "amb": amb,
                "d": np.where(amb, 0.5 * (d_k + d_n), d_k),
                "measured": np.where(amb, meas_k | meas_n, meas_k),
                "in_reach": np.where(amb, in_k | in_n, in_k)}

    def d_of(self, mz) -> np.ndarray:
        """The step the frame takes out at each m/z: its band's d, or the mean of the two
        neighbouring bands' d within the edge zone; 0 off the table."""
        return self._locate(mz)["d"]

    def in_reach(self, mz) -> np.ndarray:
        return self._locate(mz)["in_reach"]

    def measured_at(self, mz) -> np.ndarray:
        return self._locate(mz)["measured"]

    def edge_ambiguous(self, mz) -> np.ndarray:
        """Within zone_khz (in law frequency) of a band edge of the table, outer edges included."""
        return self._locate(mz)["amb"]

    def to_W(self, mz) -> np.ndarray:
        """m_W = m (1 - d(m) e-6): the served m/z with its band's step taken out
        (unchanged off the table and on unmeasured bands)."""
        loc = self._locate(mz)
        return loc["m"] * (1.0 - loc["d"] * 1e-6)

    # --- the AxisModel face ---------------------------------------------------------
    def _parts(self, mz):
        loc = self._locate(mz)
        m, d = loc["m"], loc["d"]
        m_W = m * (1.0 - d * 1e-6)
        frame_scope = loc["in_reach"] & loc["measured"]
        if self.inner is None:
            inner_scope = np.zeros(m.shape, dtype=bool)
        else:
            inner_scope = np.asarray(self.inner.in_scope(m_W), dtype=bool)
        return m, d, m_W, frame_scope, inner_scope

    def in_scope(self, mz) -> np.ndarray:
        """Where the composite reaches: the inner model's scope (read at m_W) under
        policy 'raw'; that or a measured band under 'frame'; the measured bands alone
        without an inner model (a frame not composed yet: assign_batch never applies
        one, under axis_ok its inner model is `scope_model`'s)."""
        _, _, _, frame_scope, inner_scope = self._parts(mz)
        if self.inner is None:
            return frame_scope
        if self.policy == "frame":
            return frame_scope | inner_scope
        return inner_scope

    def _inner_ppm(self, m_W, inner_scope, frame_scope, extrapolate: bool) -> np.ndarray:
        """c_W at m_W: the inner model inside its scope; 0 where the frame alone applies
        (policy 'frame', or no inner model); its extrapolation elsewhere when asked, NaN
        otherwise."""
        if self.inner is None:
            return np.zeros(np.shape(m_W))
        c = np.asarray(self.inner.predict(m_W, extrapolate=False), dtype=float)
        alone = ~inner_scope & frame_scope if self.policy == "frame" else np.zeros(np.shape(m_W), dtype=bool)
        c = np.where(alone, 0.0, c)
        if extrapolate:
            rest = np.isnan(c) & np.isfinite(m_W)
            if rest.any():
                c[rest] = np.asarray(self.inner.predict(m_W[rest], extrapolate=True), dtype=float)
        return c

    def predict(self, mz, extrapolate: bool = False) -> np.ndarray:
        """The TOTAL ppm D with m_corr = m (1 - D e-6) exactly: D = d + c_W - d c_W e-6,
        i.e. m (1 - D e-6) == m (1 - d e-6) (1 - c_W e-6). NaN outside `in_scope` unless
        `extrapolate` (then the inner model's nearest segment, composed with d)."""
        m, d, m_W, frame_scope, inner_scope = self._parts(mz)
        c = self._inner_ppm(m_W, inner_scope, frame_scope, extrapolate)
        D = d + c - d * c * 1e-6
        if not extrapolate:
            D = np.where(self.in_scope(m), D, np.nan)
        return D

    def invert(self, corrected) -> np.ndarray:
        """The served m/z each corrected one came from (`preimage`, the values alone);
        a value without a preimage is returned unchanged and counted in
        stats['invert_no_preimage'], one with several in stats['invert_several_preimages']."""
        out, found, several = self.preimage(corrected)
        self.stats["invert_no_preimage"] = int(self.stats.get("invert_no_preimage", 0) + (~found).sum())
        self.stats["invert_several_preimages"] = int(self.stats.get("invert_several_preimages", 0) + several.sum())
        return out

    def preimage(self, corrected) -> tuple:
        """The served m/z each corrected one came from, with two masks: m_W by the inner
        model's own inverse (AxisModel.invert: the lower segment wins beside a step),
        then the preimage of m_W under the frame: the fixed point m = m_W / (1 - d e-6)
        with d of band k-1, k or k+1 of m_W, accepted when m lies in that band and
        outside every edge zone, and with d the mean of the two bands at each nearby
        edge, accepted when m lies in that edge's zone (inside a zone only the mean-d
        preimage is accepted: the exact inverse of the forward rule). Several distinct
        preimages (where the map's images overlap: half a step at a zone boundary): the
        lowest m/z, `several` True. None (where they leave a gap): the value unchanged,
        `found` False (it is then off the server's axis by up to half the band step).
        Where nothing was applied (policy 'raw' outside the inner model) the value is
        returned unchanged and found -- except within |D| ppm beyond an end of the inner
        model's scope, where its inverse can carry a value the forward map left as it
        was into the scope and move it by up to |D| (AxisModel.invert's own behaviour at
        a segment end, widened by d); such a value is reported found, not flagged.
        Returns (served m/z, found, several)."""
        c = np.asarray(corrected, dtype=float)
        finite = np.isfinite(c)
        if self.inner is not None:
            m_W = np.asarray(self.inner.invert(c), dtype=float)
            need = finite & np.asarray(self.inner.in_scope(m_W), dtype=bool) if self.policy == "raw" else finite
        else:
            m_W = c.copy()
            need = finite
        out = m_W.copy()
        found = np.ones(c.shape, dtype=bool)
        several = np.zeros(c.shape, dtype=bool)
        idx = np.flatnonzero(need)
        if not len(idx):
            return out, found, several
        mw = m_W[idx]
        kW = self.k_of(mw)
        best = np.full(mw.shape, np.nan)
        n_distinct = np.zeros(mw.shape, dtype=int)

        def take(cand, ok):
            nonlocal best
            good = ok & np.isfinite(cand)
            # a distinct preimage: not the one already held (two routes to one value are one preimage)
            new = good & (np.isnan(best) | (np.abs(cand - best) > 1e-9 * np.abs(cand)))
            n_distinct[new] += 1
            sel = good & (np.isnan(best) | (cand < best))
            best[sel] = cand[sel]

        for dk in (-1.0, 0.0, 1.0):                 # a band's own d, outside the zones
            kk = kW + dk
            d_kk, _, _ = self._d_at_k(kk)
            cand = mw / (1.0 - d_kk * 1e-6)
            loc = self._locate(cand)
            take(cand, (loc["k"] == kk) & ~loc["amb"])
        for dj in (-1.0, 0.0, 1.0, 2.0):            # the mean d of the two bands at an edge, inside its zone
            j = kW + dj
            d_mean = 0.5 * (self._d_at_k(j)[0] + self._d_at_k(j - 1.0)[0])
            cand = mw / (1.0 - d_mean * 1e-6)
            loc = self._locate(cand)                # the forward map's own zone test: no boundary disagreement
            take(cand, loc["amb"] & (loc["j"] == j))
        miss = np.isnan(best)
        out[idx[~miss]] = best[~miss]
        found[idx[miss]] = False
        several[idx] = n_distinct > 1
        return out, found, several

    @property
    def mz_range(self) -> tuple:
        """The hull of where the composite can reach: the inner model's range (in the W
        frame, a few ppm off the served axis) under 'raw', joined with the table's reach
        under 'frame'; the reach alone without an inner model."""
        lo, hi = self.reach
        if self.inner is None:
            return (lo, hi)
        ilo, ihi = (float(x) for x in self.inner.mz_range)
        if self.policy == "frame":
            return (min(lo, ilo), max(hi, ihi))
        return (ilo, ihi)

    def _probe_grid(self) -> np.ndarray:
        lo, hi = self.mz_range
        zb = self.zone_bounds
        pts = [np.linspace(lo, hi, 4001), self.edges, zb * (1 - 1e-9), zb * (1 + 1e-9)]
        if self.inner is not None:
            for s in getattr(self.inner, "segments", []):
                pts.append(np.array([float(s["lo"]), float(s["hi"])]))
                if s.get("kind") == "spline":
                    pts.append(np.asarray(s["x"], dtype=float))
        g = np.unique(np.concatenate(pts))
        return g[(g >= lo * (1 - 5e-6)) & (g <= hi * (1 + 5e-6))]

    def max_abs_ppm(self) -> float:
        """The largest total correction |D| the composite applies anywhere in scope."""
        g = self._probe_grid()
        D = self.predict(g, extrapolate=False)
        return float(np.nanmax(np.abs(D))) if np.isfinite(D).any() else 0.0

    # --- records -------------------------------------------------------------------
    def with_inner(self, inner, policy=None) -> "FrameItem":
        return replace(self, inner=inner, policy=self.policy if policy is None else policy,
                       d=self.d.copy(), measured=self.measured.copy(), meta=dict(self.meta), stats={})

    def as_dict(self, *, shared_inner: bool = False) -> dict:
        """JSON-safe. `shared_inner` leaves the inner model and the policy to the FrameSet's
        record that holds this item."""
        out = {"model": "frame_item", "version": 1, "sample_item_id": self.sample_item_id,
               "B": float(self.B), "C": float(self.C), "period_khz": float(self.period),
               "phase_khz": float(self.phase), "zone_khz": float(self.zone_khz), "k0": int(self.k0),
               "d_ppm": [float(x) for x in self.d], "measured": [bool(x) for x in self.measured],
               "meta": _jsonable(self.meta), "stats": _jsonable(self.stats)}
        if not shared_inner:
            out["policy"] = self.policy
            out["inner"] = None if self.inner is None else self.inner.as_dict()
        return out

    @classmethod
    def from_dict(cls, d: dict, *, inner=None, policy=None) -> "FrameItem":
        inn = inner
        if inn is None and d.get("inner") is not None:
            inn = _inner_from_dict(d["inner"])
        return cls(sample_item_id=d.get("sample_item_id"), B=float(d["B"]), C=float(d["C"]),
                   period=float(d["period_khz"]), phase=float(d["phase_khz"]),
                   zone_khz=float(d.get("zone_khz", DEFAULT_EDGE_UNCERTAINTY_KHZ)), k0=int(d["k0"]),
                   d=np.asarray(d["d_ppm"], dtype=float), measured=np.asarray(d["measured"], dtype=bool),
                   inner=inn, policy=policy or d.get("policy") or OUTSIDE_POLICY,
                   meta=dict(d.get("meta") or {}), stats=dict(d.get("stats") or {}))


def _inner_from_dict(rec):
    from peaky.batch.axislock import AxisModel
    if rec is None:
        return None
    if isinstance(rec, dict) and rec.get("model") == "locks":
        return AxisModel.from_dict(rec)
    raise ValueError(f"a frame's inner model must be a lock model record, got {type(rec).__name__}")


def scope_model(wave, scope_mz=None):
    """The inner model of a composite under the W verdict `axis_ok` (nothing smooth to
    correct): the W fit's own lock model (its record `wave`, which assign_batch keeps
    under every verdict) with every segment a zero line over the same domain, its holes
    kept, so that under policy 'raw' the composite applies the band steps alone and only
    where the W fit saw locks, and its scope is the lock range. Without a lock-model
    record, one zero line over `scope_mz` ([lo, hi]); None when neither is usable."""
    from peaky.batch.axislock import AxisModel
    segs = []
    n_locks = 0
    if isinstance(wave, dict) and wave.get("model") == "locks" and wave.get("segments"):
        for s in wave["segments"]:
            seg = {"kind": "line", "lo": float(s["lo"]), "hi": float(s["hi"]), "n": int(s.get("n") or 0),
                   "a": 0.0, "b": 0.0}
            if s.get("holes"):
                seg["holes"] = [[float(a), float(b)] for a, b in s["holes"]]
            segs.append(seg)
        n_locks = int(wave.get("n_locks") or 0)
    elif scope_mz is not None and len(scope_mz) == 2:
        lo, hi = (_num(x) for x in scope_mz)
        if np.isfinite(lo) and np.isfinite(hi) and lo < hi:
            segs = [{"kind": "line", "lo": float(lo), "hi": float(hi), "n": 0, "a": 0.0, "b": 0.0}]
    if not segs:
        return None
    return AxisModel(segments=segs, n_locks=n_locks,
                     stats={"zero": "axis_ok: the W fit's lock domains, no correction"})


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return [_jsonable(v) for v in x.tolist()]
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, (np.integer, int)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        v = float(x)
        return v if math.isfinite(v) else None
    return x


# --------------------------------------------------------------------------- #
# the batch's frames
# --------------------------------------------------------------------------- #
@dataclass
class FrameSet:
    """Every item's frame of one batch, keyed by sample id, with ONE inner lock model
    (fitted on the W-frame series of the whole batch), one policy and the calibration
    group records the items were built from. A FrameSet is not a map: `for_item(sid)`
    is; `predict` / `in_scope` / `invert` on the set raise, so no item's map is ever
    applied to the batch."""
    items: dict
    inner: object = None
    policy: str = OUTSIDE_POLICY
    stats: dict = field(default_factory=dict)
    groups: list = field(default_factory=list)
    K: int = -1

    def __post_init__(self):
        if not self.items:
            raise ValueError("a FrameSet needs at least one item")
        if self.policy not in POLICIES:
            raise ValueError(f"policy {self.policy!r} is not one of {POLICIES}")
        self.items = {str(k): v for k, v in self.items.items()}
        if self.inner is None:
            # items that already share one model (by identity) hand it to the set
            shared = {id(it.inner) for it in self.items.values()}
            if len(shared) == 1:
                self.inner = next(iter(self.items.values())).inner
        for sid, it in self.items.items():
            it.inner = self.inner
            it.policy = self.policy
            if it.sample_item_id is None:
                it.sample_item_id = sid

    @property
    def sample_ids(self) -> list:
        return list(self.items)

    def for_item(self, sample_item_id) -> FrameItem:
        sid = str(sample_item_id)
        if sid not in self.items:
            raise KeyError(f"no frame for sample item {sid!r}: the frame is all-or-nothing per batch")
        return self.items[sid]

    def to_frame(self, ts: pd.DataFrame, *, sample_col: str = "sample_item_id", mz_col: str = "mz") -> pd.DataFrame:
        """The time series in the W frame: each item's `mz` with its band step taken out
        (`FrameItem.to_W`), everything else as given. Counts per item what moved, what
        sat in an edge zone (`n_edge_ambiguous`) and what fell on an unmeasured band
        (stats['frame']). Raises on an item without a frame."""
        if ts is None or sample_col not in ts.columns or mz_col not in ts.columns:
            raise ValueError(f"the series needs columns {sample_col!r} and {mz_col!r}")
        out = ts.copy()
        mz = out[mz_col].to_numpy(dtype=float)
        new = mz.copy()
        per: dict = {}
        sids = out[sample_col].astype(str).to_numpy()
        for sid in pd.unique(sids):
            sel = np.flatnonzero(sids == sid)
            it = self.for_item(sid)
            loc = it._locate(mz[sel])
            d, meas, inside, amb = loc["d"], loc["measured"], loc["in_reach"], loc["amb"]
            new[sel] = loc["m"] * (1.0 - d * 1e-6)
            per[sid] = {"n_peaks": int(len(sel)), "n_in_reach": int(inside.sum()),
                        "n_moved": int((inside & (d != 0)).sum()),
                        "n_unmeasured_band": int((inside & ~meas).sum()),
                        "n_edge_ambiguous": int((amb & inside).sum()),
                        "d_ppm": {"min": round(float(d[inside].min()), 4) if inside.any() else None,
                                  "max": round(float(d[inside].max()), 4) if inside.any() else None}}
        out[mz_col] = new
        self.stats["frame"] = per
        return out

    def with_inner(self, inner, policy=None) -> "FrameSet":
        pol = self.policy if policy is None else policy
        return FrameSet(items={sid: it.with_inner(inner, pol) for sid, it in self.items.items()},
                        inner=inner, policy=pol, stats=dict(self.stats), groups=list(self.groups))

    def _not_a_map(self, what):
        raise TypeError(f"a FrameSet is per item: call for_item(sample_item_id).{what}; one item's "
                        "band steps must never be applied to the batch")

    def predict(self, *a, **k):
        self._not_a_map("predict")

    def in_scope(self, *a, **k):
        self._not_a_map("in_scope")

    def invert(self, *a, **k):
        self._not_a_map("invert")

    def to_W(self, *a, **k):
        self._not_a_map("to_W")

    @property
    def h_max(self) -> float:
        return max(it.h_max for it in self.items.values())

    def max_abs_ppm(self) -> float:
        return max(it.max_abs_ppm() for it in self.items.values())

    @property
    def mz_range(self) -> tuple:
        los, his = zip(*(it.mz_range for it in self.items.values()))
        return (float(min(los)), float(max(his)))

    def describe(self) -> dict:
        """A JSON-safe summary for batch_summary['mass_axis']['frame']: the policy, the
        inner model's range, the calibration groups (grid, zone, B, C, the per-band
        table) and per item its group, mode, document and lock."""
        return _jsonable({
            "policy": self.policy, "n_items": len(self.items), "n_groups": len(self.groups),
            "h_max_ppm": round(self.h_max, 4),
            "inner": None if self.inner is None else {"model": "locks", "mz_range": list(self.inner.mz_range)},
            "groups": list(self.groups),
            "items": {sid: {"group": it.meta.get("group"), "mode_id": it.meta.get("mode_id"),
                            "n_bands": int(len(it.d)), "n_measured": int(it.measured.sum()),
                            "h_max_ppm": round(it.h_max, 4), "k_range": [it.k0, it.k1],
                            "reach_mz": [round(x, 3) for x in it.reach], "zone_khz": it.zone_khz,
                            "grid": {"period_khz": it.period, "phase_khz": it.phase},
                            "document_grid": it.meta.get("document_grid"),
                            "grid_weak": it.meta.get("grid_weak"), "lock": it.meta.get("lock"),
                            "document_sha1": it.meta.get("document_sha1")}
                      for sid, it in self.items.items()},
            "stats": self.stats})

    def as_dict(self) -> dict:
        return {"model": "frame", "version": 1, "policy": self.policy,
                "inner": None if self.inner is None else self.inner.as_dict(),
                "items": {sid: it.as_dict(shared_inner=True) for sid, it in self.items.items()},
                "groups": _jsonable(self.groups), "stats": _jsonable(self.stats)}

    @classmethod
    def from_dict(cls, d: dict) -> "FrameSet":
        if d.get("model") != "frame":
            raise ValueError(f"not a frame record: model {d.get('model')!r}")
        if int(d.get("version", 1)) > 1:
            raise ValueError(f"frame record version {d.get('version')} is newer than this reader")
        inner = _inner_from_dict(d.get("inner"))
        policy = d.get("policy") or OUTSIDE_POLICY
        items = {sid: FrameItem.from_dict(rec, inner=inner, policy=policy) for sid, rec in d["items"].items()}
        return cls(items=items, inner=inner, policy=policy, stats=dict(d.get("stats") or {}),
                   groups=list(d.get("groups") or []))

    @classmethod
    def from_documents(cls, documents: dict, *, polarity: str, policy: str = OUTSIDE_POLICY) -> tuple:
        """The batch's frames from its items' served documents ({sample id: document or
        None}), all or nothing: (FrameSet, record) when every item has a usable document
        of the batch's polarity and the documents form exactly ONE calibration group
        whose pooled table passes the limits, else (None, record) with record['held']
        saying why. The record (JSON-safe) carries every item's `Loaded.as_dict()` and
        every group's record (`CalibrationGroup.as_dict`), held or not."""
        per = {}
        fps = {}
        held = None
        for sid, doc in documents.items():
            loaded = load_document(doc, polarity=polarity) if doc is not None else \
                Loaded(None, "no document served", [], None, None, None)
            per[str(sid)] = loaded.as_dict()
            if loaded.ok:
                fps[str(sid)] = loaded.fingerprint
            elif held is None:
                held = f"item {sid}: {loaded.why}"
        if not documents:
            held = "no items"
        groups: list = []
        if fps:
            try:
                groups = group_fingerprints(fps)
            except ValueError as e:
                held = held or f"calibration groups: {e}"
        group_recs = [g.as_dict() for g in groups]
        record = {"items": per, "n_usable": len(fps), "n_items": len(documents), "policy": policy,
                  "n_groups": len(groups), "groups": group_recs}
        if held is None and len(groups) != 1:
            held = f"{len(groups)} calibration groups in one batch"
        if held is None:
            held = group_hold(groups[0])
        if held is not None:
            record["held"] = held
            return None, record
        g = groups[0]
        items = {}
        for sid in g.members:
            fp = fps[sid]
            items[sid] = g.frame_item(sid, policy=policy, meta={
                "document_sha1": fp.meta.get("document_sha1"), "lock": fp.meta.get("lock"),
                "grid_source": fp.meta.get("grid_source"), "grid_weak": fp.meta.get("grid_weak"),
                "mode_flags": fp.meta.get("mode_flags"), "item": fp.meta.get("item"),
                "document_grid": {"period_khz": fp.period, "phase_khz": fp.phase, "n_hits": fp.n_hits,
                                  "edge_uncertainty_khz": fp.edge_uncertainty_khz},
                "document_n_measured": fp.n_measured})
        return cls(items=items, policy=policy, groups=group_recs), record


def group_hold(group: CalibrationGroup) -> "str | None":
    """Why a calibration group's POOLED table cannot be applied, or None: fewer than
    MIN_MEASURED_SEGMENTS bands with a trend, or a band step above H_MAX_PPM. The same
    two limits `load_document` holds every member to; pooling can leave fewer bands
    with a trend than a member had (a run of gross steps zeroes its neighbours) or a
    larger step than any member's (the trend of the pooled table is its own)."""
    if group.n_measured < MIN_MEASURED_SEGMENTS:
        return f"the pooled table has {group.n_measured} measured bands, fewer than {MIN_MEASURED_SEGMENTS}"
    if group.h_max > H_MAX_PPM:
        return (f"the pooled table's largest band step is {group.h_max:.1f} ppm, above {H_MAX_PPM:g}: "
                "a broken calibration to fix at the instrument")
    return None


def from_record(rec):
    """A frame record ('model' 'frame' or 'frame_item') or object back into its class; None
    for anything else (massqc.fit_from_record dispatches here)."""
    if isinstance(rec, (FrameSet, FrameItem)):
        return rec
    if isinstance(rec, dict):
        if rec.get("model") == "frame":
            return FrameSet.from_dict(rec)
        if rec.get("model") == "frame_item":
            return FrameItem.from_dict(rec)
    return None


# --------------------------------------------------------------------------- #
# the frame gate
# --------------------------------------------------------------------------- #
def _edge_list(frame) -> tuple:
    """Every item's band edges (m/z) with one band width, the frame's own |step| and the
    edge zone in ppm each, merged where two items' edges fall within a tenth of a band
    width of each other (the larger |step| and zone kept)."""
    items = list(frame.items.values()) if isinstance(frame, FrameSet) else [frame]
    rows = []
    for it in items:
        E = np.array([s["edge_mz"] for s in it.edge_steps()])
        w = it.band_width_at(E)
        z = it.zone_ppm_at(E)
        rows += [(s["edge_mz"], float(wk), abs(s["step_ppm"]), float(zk)) for s, wk, zk in zip(it.edge_steps(), w, z)]
    rows.sort()
    edges, widths, spans, zones = [], [], [], []
    for e, w, s, z in rows:
        if edges and abs(e - edges[-1]) <= 0.1 * min(w, widths[-1]):
            spans[-1] = max(spans[-1], s)
            zones[-1] = max(zones[-1], z)
            continue
        edges.append(e)
        widths.append(w)
        spans.append(s)
        zones.append(z)
    return np.array(edges), np.array(widths), np.array(spans), np.array(zones)


def gate(model, info, locks_table, frame) -> dict:
    """The truth-free frame gate on the lock fit made in the W frame (module note). Fails:
      (a) the W fit reports a STEP whose bracket lies within one band width of a grid
          edge (the frame left a step there, or made one);
      (b) at a grid edge inside the W model's domain with >= 2 non-bright locks within
          one band on each side (locks in the edge zone, widened by half the step the
          frame took out there, left out: their side cannot be told), the medians of
          the locks' residuals differ by more than EDGE_STEP_GATE_PPM;
      (c) the verdict is not one under which d may be applied (`locks_unpredictive`,
          `locks_unvalidated`, none at all), or there is no W model while the verdict
          is not `axis_ok`.
    `model`: the W AxisModel (None under axis_ok); `info`: assign_batch's mass-axis info
    (its verdict, info['locks']['steps']) or axislock.fit's own info; `locks_table`:
    its lock rows (mz in the W frame, ppm, bright). Returns a JSON-safe record with
    `passed`, `failed` (the checks) and `why`. The reach and quality rules (d), (e)
    compare the W fit with the raw fit and live where both fits are made."""
    info = info or {}
    verdict = info.get("verdict")
    steps = info.get("steps")
    if steps is None and isinstance(info.get("locks"), dict):
        steps = info["locks"].get("steps")
    steps = steps or []
    locks = locks_table
    if locks is None:
        locks = pd.DataFrame(columns=["mz", "ppm", "bright"])
    elif not isinstance(locks, pd.DataFrame):
        locks = pd.DataFrame(list(locks))
    edges, widths, spans, zones = _edge_list(frame)
    out = {"passed": True, "failed": [], "why": None}

    # (c) the verdict
    c_ok = verdict in ACCEPTED_VERDICTS and (model is not None or verdict == "axis_ok")
    out["c"] = {"verdict": verdict, "ok": bool(c_ok)}
    if not c_ok:
        out["failed"].append("c")
        out["why"] = (f"verdict {verdict!r}: the W model does not predict its held-out locks"
                      if verdict in ("locks_unpredictive", "locks_unvalidated") else
                      f"verdict {verdict!r}: no W model to validate the frame against")

    # (a) a step beside an edge
    near = []
    for st in steps:
        try:
            lo, hi = (float(x) for x in st["between"])
        except (KeyError, TypeError, ValueError):
            continue
        dist = np.where((edges >= lo) & (edges <= hi), 0.0, np.minimum(np.abs(edges - lo), np.abs(edges - hi)))
        hit = np.flatnonzero(dist <= widths)
        if len(hit):
            j = int(hit[np.argmin(dist[hit])])
            near.append({"between": [lo, hi], "ppm": st.get("ppm"), "edge_mz": round(float(edges[j]), 4),
                         "distance_mz": round(float(dist[j]), 4)})
    out["a"] = {"steps": len(steps), "near_edge": near}
    if near:
        out["failed"].append("a")
        out["why"] = out["why"] or (f"the W fit reports a step of {near[0]['ppm']} ppm between m/z "
                                    f"{near[0]['between'][0]:.2f} and {near[0]['between'][1]:.2f}, within one "
                                    f"band of the grid edge at {near[0]['edge_mz']:.2f}")

    # (b) a residual step across an edge
    tested, failing, worst = 0, [], 0.0
    if len(locks) and {"mz", "ppm"} <= set(locks.columns):
        bright = locks["bright"].astype(bool).to_numpy() if "bright" in locks.columns else np.zeros(len(locks), bool)
        mz = pd.to_numeric(locks["mz"], errors="coerce").to_numpy(dtype=float)
        ppm = pd.to_numeric(locks["ppm"], errors="coerce").to_numpy(dtype=float)
        keep = ~bright & np.isfinite(mz) & np.isfinite(ppm)
        mz, ppm = mz[keep], ppm[keep]
        if model is not None:
            pred = np.asarray(model.predict(mz, extrapolate=False), dtype=float)
        else:
            pred = np.zeros(len(mz))
        resid = ppm - pred
        good = np.isfinite(resid)
        for e, w, span, zone in zip(edges, widths, spans, zones):
            if model is not None:
                if not bool(np.asarray(model.in_scope(np.array([e])))[0]):
                    continue
            elif not (len(mz) and mz.min() <= e <= mz.max()):
                continue
            # a lock in the edge zone (its band is ambiguous), or within half the step the frame
            # took out there (the two bands' images overlap or part by that in W), may sit on
            # either side of the edge in the served frame
            amb = np.abs(mz - e) / e * 1e6 <= zone + 0.5 * span
            left = good & ~amb & (mz >= e - w) & (mz < e)
            right = good & ~amb & (mz > e) & (mz <= e + w)
            if left.sum() < 2 or right.sum() < 2:
                continue
            tested += 1
            jump = float(np.median(resid[right]) - np.median(resid[left]))
            worst = max(worst, abs(jump))
            if abs(jump) > EDGE_STEP_GATE_PPM:
                failing.append({"edge_mz": round(float(e), 4), "n_left": int(left.sum()),
                                "n_right": int(right.sum()), "jump_ppm": round(jump, 3)})
    out["b"] = {"edges_tested": tested, "worst_jump_ppm": round(worst, 3), "failing": failing}
    if failing:
        out["failed"].append("b")
        f0 = failing[0]
        out["why"] = out["why"] or (f"the locks' residuals step by {f0['jump_ppm']:+.2f} ppm across the grid "
                                    f"edge at m/z {f0['edge_mz']:.2f} ({f0['n_left']} locks below, "
                                    f"{f0['n_right']} above), more than {EDGE_STEP_GATE_PPM:g}")
    out["passed"] = not out["failed"]
    out["failed"] = sorted(out["failed"])
    return _jsonable(out)
