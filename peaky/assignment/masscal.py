"""Mass-dependent mass-error centre for the calibrated z gate.

The pass-1 self-calibration used to model the instrument's mass error as ONE
constant ppm offset (median) with a ppm sigma (MAD). On an Orbitrap the error
is not constant in ppm: below ~m/z 120 the residual of the calibration curve is
an (approximately) constant ABSOLUTE offset, so in ppm it grows as 1/mz. On the
2026-09-10 ^NH4+ file the Assigned backbone read -0.76 ppm at m/z 80-120,
-0.29 at 120-160 and -0.18 above 160 (MAD 0.13-0.22 ppm throughout), and the
bright sub-80 ions (ketene/urea/acetamide/acetic acid on their reagent channels)
all sat at -2.0 ppm = -0.12 mDa -- rejected at z=6 by a centre of -0.3 ppm and
a 0.33 ppm sigma. That is 0.015 mDa of absolute tolerance at m/z 61, far below
what any centroiding delivers.

`fit_mass_trend` fits  ppm = a + b * (1000 / mz)  robustly (3 rounds of MAD
trimming); with x = 1000/mz the slope b IS the constant absolute offset in mDa,
so the log line reads physically. The fit is accepted only when ALL of
  * the slope is significant, |b| > SLOPE_MIN_SE * SE(b)  (3 SE: a 2-SE rule is
    a 5 % two-sided test, so a flat source grew a phantom trend in ~6 % of
    samples at any n);
  * the trend explains variance the constant model cannot: the RMS of the
    trimmed residuals is <= TREND_SIGMA_RATIO_MAX x the constant model's RMS on
    the same points (RMS, not MAD, on purpose -- the low-mass tail the model
    exists for is a MINORITY of the backbone, which a robust MAD ignores);
  * |b| <= MAX_ABS_OFFSET_MDA -- a larger constant offset is a broken
    calibration, not a residual to model;
  * both halves of the fitted 1000/mz range hold >= MIN_SIDE_N kept points: a
    single corroborated outlier at one end of the mass range is a LEVER the
    least-squares line passes through (its residual is ~0, so trimming never
    removes it and the variance ratio even improves) -- not a trend.
A flat source keeps the constant model exactly. The fit also records the
backbone's m/z coverage (`mz_lo`, `mz_hi`); `centre` holds the centre at the
nearest edge outside it, so a 150-480 backbone never extrapolates +2 ppm onto
an m/z 61 it never saw. `centre` and `sigma_at` are the ONE implementation of
the mass-dependent centre and sigma: passes.core (`cal_center`, `z_of`) and
tiers (`_cal_z`) both call them. Callers without an m/z keep the old constant
centre and sigma, so nothing changes for them."""
from __future__ import annotations

from typing import NamedTuple

import numpy as np

__version__ = "0.2.0"  # + 3-SE / variance-ratio / |b| cap acceptance; range clamp

# absolute floor (mDa) on the trend sigma, active only where it exceeds the ppm
# sigma (below ~m/z 120 at sigma 0.25 ppm): the Orbitrap's residual at the
# low-mass edge curves faster than 1/mz (46.065 sat 0.9 ppm = 0.04 mDa off the
# fitted trend), and 0.03 mDa is the absolute-accuracy floor the backbone itself
# shows (mDa MAD 0.01-0.05 across the range). The runtime owner is
# PassConfig.cal_abs_floor_mda; this constant is only its default.
ABS_FLOOR_MDA = 0.03
# acceptance rule of fit_mass_trend (see the module note)
SLOPE_MIN_SE = 3.0            # |b| must exceed this many standard errors
TREND_SIGMA_RATIO_MAX = 0.8   # RMS(trend residuals) / RMS(constant residuals)
# |b| cap. b is the residual ABSOLUTE offset of the instrument's calibration
# curve in mDa (x = 1000/mz, so ppm = b*1000/mz means dm = b mDa at every mass).
# 0.5 mDa is already 1.25 ppm at m/z 400 -- a working calibration has no constant
# term that large, so beyond it the source is mis-calibrated and the honest answer
# is the constant model, not a 1/mz correction worth 8 ppm at m/z 61. The measured
# 2026-09-10 labelled-ammonium file sat at -0.12 mDa.
MAX_ABS_OFFSET_MDA = 0.5
MIN_SIDE_N = 5                # kept points required in EACH half of the x range


class MassTrend(NamedTuple):
    """ppm = a + b*1000/mz fitted on the calibration backbone."""
    a: float        # ppm intercept (the high-mass asymptote)
    b: float        # constant absolute offset, mDa (slope on x = 1000/mz)
    sigma: float    # robust residual sigma, ppm (floored at sigma_floor)
    n: int          # backbone points kept after trimming
    mz_lo: float    # backbone m/z coverage: the centre is held constant outside
    mz_hi: float


def fit_mass_trend(mz, ppm, *, min_n: int = 20, sigma_floor: float = 0.25,
                   rounds: int = 3, trim_k: float = 3.0) -> MassTrend | None:
    """Robust fit of ppm = a + b*(1000/mz). Returns a MassTrend, or None when
    the backbone is too small, spans too little mass, or the trend is not
    accepted (slope not significant at SLOPE_MIN_SE, residual RMS not below
    TREND_SIGMA_RATIO_MAX x the constant model's, |b| > MAX_ABS_OFFSET_MDA, or
    fewer than MIN_SIDE_N kept points in either half of the 1000/mz range)."""
    x = 1000.0 / np.asarray(mz, dtype=float)
    y = np.asarray(ppm, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < min_n or (x.max() - x.min()) < 1.5:
        return None
    keep = np.ones(len(x), dtype=bool)
    a = b = 0.0
    for _ in range(rounds):
        if keep.sum() < min_n:
            return None
        A = np.column_stack([np.ones(keep.sum()), x[keep]])
        coef, *_ = np.linalg.lstsq(A, y[keep], rcond=None)
        a, b = float(coef[0]), float(coef[1])
        res = y - (a + b * x)
        mad = 1.4826 * np.median(np.abs(res[keep] - np.median(res[keep])))
        if mad <= 0:
            break
        keep = np.abs(res) <= trim_k * max(mad, sigma_floor)
    if keep.sum() < min_n:
        return None
    res = y - (a + b * x)
    sigma = max(float(1.4826 * np.median(np.abs(res[keep]))), sigma_floor)
    xk, yk, rk = x[keep], y[keep], res[keep]
    # (0) coverage: no one-sided lever (see the module note)
    x_mid = 0.5 * (xk.min() + xk.max())
    if (xk < x_mid).sum() < MIN_SIDE_N or (xk >= x_mid).sum() < MIN_SIDE_N:
        return None
    # (1) slope significance (OLS standard error on the kept points)
    sxx = float(((xk - xk.mean()) ** 2).sum())
    if sxx <= 0:
        return None
    se_b = float(np.sqrt((rk ** 2).sum() / max(len(rk) - 2, 1) / sxx))
    if abs(b) <= SLOPE_MIN_SE * se_b:
        return None
    # (2) the trend must beat the constant model on the same points
    rms_t = float(np.sqrt(np.mean(rk ** 2)))
    rms_c = float(np.sqrt(np.mean((yk - np.median(yk)) ** 2)))
    if rms_c <= 0 or rms_t > TREND_SIGMA_RATIO_MAX * rms_c:
        return None
    # (3) physical cap on the constant absolute offset
    if abs(b) > MAX_ABS_OFFSET_MDA:
        return None
    mzk = 1000.0 / xk
    return MassTrend(a, b, sigma, int(keep.sum()), float(mzk.min()), float(mzk.max()))


def centre(a: float, b: float, mz: float,
           mz_lo: float | None = None, mz_hi: float | None = None) -> float:
    """Calibrated ppm centre at m/z. With the backbone coverage given, `mz` is
    clamped into [mz_lo, mz_hi] first: outside the fitted range the centre is
    the constant model at the nearest edge, never a 1/mz extrapolation."""
    m = float(mz)
    if mz_lo is not None and m < float(mz_lo):
        m = float(mz_lo)
    if mz_hi is not None and m > float(mz_hi):
        m = float(mz_hi)
    return a + b * 1000.0 / m


def sigma_at(sigma_ppm: float, mz: float, abs_floor_mda: float = ABS_FLOOR_MDA) -> float:
    """The trend sigma at m/z: the ppm sigma, floored by the absolute floor
    expressed in ppm at that mass (active only where the floor exceeds it)."""
    return max(float(sigma_ppm), float(abs_floor_mda) * 1000.0 / float(mz))
