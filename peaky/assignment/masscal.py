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
so the log line reads physically. The fit is accepted only when the slope is
significant (|b| > 2 SE) -- a flat source keeps the constant model exactly.
Consumers pass the peak's m/z to `z_of` / the tier gate; callers without an m/z
keep the old constant centre and sigma, so nothing changes for them."""
from __future__ import annotations

import numpy as np

__version__ = "0.1.0"


def fit_mass_trend(mz, ppm, *, min_n: int = 20, sigma_floor: float = 0.25,
                   rounds: int = 3, trim_k: float = 3.0):
    """Robust fit of ppm = a + b*(1000/mz). Returns (a, b, sigma, n_used) or
    None when the backbone is too small, spans too little mass, or the slope is
    not significant (|b| <= 2 SE)."""
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
    res = y - (a + b * x)
    sigma = max(float(1.4826 * np.median(np.abs(res[keep]))), sigma_floor)
    # slope significance (OLS standard error on the kept points)
    xk = x[keep]
    sxx = float(((xk - xk.mean()) ** 2).sum())
    if sxx <= 0:
        return None
    se_b = float(np.sqrt((res[keep] ** 2).sum() / max(keep.sum() - 2, 1) / sxx))
    if abs(b) <= 2.0 * se_b:
        return None
    return a, b, sigma, int(keep.sum())


def centre(a: float, b: float, mz: float) -> float:
    """Calibrated ppm centre at m/z."""
    return a + b * 1000.0 / float(mz)
