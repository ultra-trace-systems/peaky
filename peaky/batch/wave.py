"""The mass-offset wave: a batch's smooth axis error as a written model.

    delta_ppm(m/z) = sum_k c_k T_k(u),   u = linear map of v = (m/z)^p onto [-1, 1]
    m/z_corr = m/z_meas * (1 - delta_ppm * 1e-6)

with p = +1/2 for a TOF (flight time) and p = -1/2 for an Orbitrap (frequency).
T_k are Chebyshev polynomials of the first kind; K is picked by leave-one-out
cross-validation rather than by eye, and the fit is iteratively 3-sigma clipped.

Two rules learned the hard way:

  * K MUST be allowed to be 0. A search over 1..5 cannot express a flat offset,
    which is exactly the `axis_offset` verdict: one channel read as 0.45 ppm
    unfittable until K = 0 was permitted, then resolved to 0.10 ppm as a
    constant +0.90 ppm bias.
  * The LOO score must be an L1 mean, not a MAD. A MAD-based summary is far too
    jumpy at these sample sizes: on a genuinely flat 25-point series it read
    K = 2 as 33 % better than K = 0 on noise alone, and a constant bias was
    reported as a trend. A one-standard-error rule then takes the SIMPLEST K
    whose CV error is within one SE of the best (~0.75/sqrt(n) of it).

Scope: a degree-4 Chebyshev swings freely in an empty stretch (one channel's
curve reached +0.67 ppm above the last calibrant on the strength of a single
point). `WaveFit.domain` records the populated range and `predict` refuses
outside it unless told to extrapolate.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__version__ = "0.1.0"
__all__ = ["WaveFit", "fit_wave"]


def rsd(x) -> float:
    x = np.asarray(x, dtype=float)
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


@dataclass
class WaveFit:
    coef: list                       # Chebyshev coefficients, len = K + 1
    p: float                         # exponent on m/z
    domain: tuple                    # (v_lo, v_hi) in v = (m/z)^p
    mz_range: tuple                  # populated m/z range of the calibrants
    K: int
    n: int                           # calibrants kept after clipping
    n_clipped: int
    resid_ppm: float                 # robust sd of the residual
    raw_ppm: float                   # robust sd before the fit
    loo_ppm: float                   # leave-one-out CV error (L1)
    span_ppm: float                  # predicted swing across the populated range
    share: float                     # fraction of the variance the model explains

    def _u(self, mz):
        v = np.asarray(mz, dtype=float) ** self.p
        lo, hi = self.domain
        return v, (2 * (v - lo) / (hi - lo) - 1 if hi > lo else np.zeros_like(v))

    def predict(self, mz, extrapolate: bool = False):
        """delta_ppm at `mz`; NaN outside the calibrant range unless `extrapolate`."""
        v, u = self._u(mz)
        out = np.polynomial.chebyshev.chebval(u, self.coef)
        if not extrapolate:
            lo, hi = min(self.domain), max(self.domain)
            out = np.where((v >= lo) & (v <= hi), out, np.nan)
        return out

    def correct(self, mz, extrapolate: bool = False):
        """m/z with the wave removed; unchanged outside the range unless `extrapolate`."""
        d = self.predict(mz, extrapolate)
        return np.asarray(mz, dtype=float) * (1 - np.nan_to_num(d) * 1e-6)

    def as_dict(self) -> dict:
        return {k: (list(map(float, v)) if isinstance(v, (list, tuple, np.ndarray)) else
                    (int(v) if isinstance(v, (int, np.integer)) else float(v)))
                for k, v in self.__dict__.items()}


def _cheb_fit(u, y, K):
    A = np.polynomial.chebyshev.chebvander(u, K)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return coef


def _loo(u, y, K) -> float:
    """Leave-one-out CV error as the MEAN ABSOLUTE held-out residual."""
    n = len(u)
    if n <= K + 2:
        return float("inf")
    err = np.empty(n)
    for i in range(n):
        m = np.ones(n, dtype=bool)
        m[i] = False
        c = _cheb_fit(u[m], y[m], K)
        err[i] = y[i] - np.polynomial.chebyshev.chebval(u[i], c)
    return float(np.mean(np.abs(err)))


def _one_se(n: int, k_tol: float | None) -> float:
    """The one-standard-error margin for the L1 CV score: |e| of a Gaussian has
    mean 0.8 sigma and sd 0.6 sigma, so the relative SE of its mean over n points
    is ~0.75 / sqrt(n). A higher K must beat the simplest by more than that."""
    return float(k_tol) if k_tol is not None else max(0.75 / np.sqrt(max(n, 1)), 0.05)


def fit_wave(mz, ppm, *, tof: bool = True, k_max: int = 5, clip: float = 3.0,
             n_clip_iter: int = 4, min_n: int = 6, k_tol: float | None = None) -> WaveFit | None:
    """Fit delta_ppm against (m/z)^p. None with fewer than `min_n` calibrants;
    a clip that would leave fewer than that is refused, so the returned fit
    always stands on at least `min_n` of them and `n` says how many. K is picked by LOO over 0..k_max, including 0:
    the SIMPLEST K whose CV error is within one standard error of the best
    (`k_tol` overrides that margin with a fixed fraction)."""
    mz = np.asarray(mz, dtype=float)
    ppm = np.asarray(ppm, dtype=float)
    ok = np.isfinite(mz) & np.isfinite(ppm)
    mz, ppm = mz[ok], ppm[ok]
    if len(mz) < min_n:
        return None
    p = 0.5 if tof else -0.5
    v = mz ** p
    lo, hi = float(v.min()), float(v.max())
    u_all = 2 * (v - lo) / (hi - lo) - 1 if hi > lo else np.zeros_like(v)
    keep = np.ones(len(mz), dtype=bool)
    best = None
    for _ in range(n_clip_iter):
        u, y = u_all[keep], ppm[keep]
        scores = {K: _loo(u, y, K) for K in range(0, k_max + 1) if len(y) > K + 2}
        if not scores:
            return None
        best_loo = min(scores.values())
        tol = _one_se(len(y), k_tol)
        K = min(k for k in sorted(scores)
                if scores[k] <= best_loo * (1 + tol) or scores[k] - best_loo <= 1e-3)
        coef = _cheb_fit(u, y, K)
        # the fit as it stands on THIS set, which holds at least `min_n` points
        # by construction. Recording it here is the guard: the old code recorded
        # the fit against the mask the NEXT clip proposed, so a clip that cut the
        # calibrants below min_n still returned -- a degree-2 wave standing on 4
        # surviving points, reported as though it stood on all of them.
        best = (K, coef, scores[K], keep.copy())
        res_all = ppm - np.polynomial.chebyshev.chebval(u_all, coef)
        # clip about the MEDIAN residual: an outlier pulls the least-squares
        # constant toward itself, so the good points' residuals share an offset
        # and a clip about zero would throw them out and keep the outlier
        dev = res_all - np.median(res_all[keep])
        s = max(rsd(res_all[keep]), 1e-9)
        new = np.abs(dev) <= clip * s
        if new.sum() < min_n or np.array_equal(new, keep):
            break
        keep = new
    K, coef, loo, keep = best
    # judge the fit on the points it was JUDGED on. `raw` used to be the spread
    # of EVERY calibrant while `resid` was the spread of the survivors, so
    # `share` compared two different sets and any clip at all flattered it --
    # a 4-point degree-2 fit read "explains 100%". The outliers' own spread is
    # not hidden: `verdict` reports it as ion_to_ion_spread_ppm.
    raw = rsd(ppm[keep])
    res_all = ppm - np.polynomial.chebyshev.chebval(u_all, coef)
    resid = rsd(res_all[keep]) if keep.sum() > 1 else 0.0
    grid = np.linspace(mz.min(), mz.max(), 200)
    ug = 2 * (grid ** p - lo) / (hi - lo) - 1 if hi > lo else np.zeros_like(grid)
    pred = np.polynomial.chebyshev.chebval(ug, coef)
    share = float(np.clip(1 - (resid / max(raw, 1e-12)) ** 2, 0, 1))
    return WaveFit(coef=list(map(float, np.atleast_1d(coef))), p=p, domain=(lo, hi),
                   mz_range=(float(mz.min()), float(mz.max())), K=int(K),
                   n=int(keep.sum()), n_clipped=int((~keep).sum()),
                   resid_ppm=float(resid), raw_ppm=float(raw), loo_ppm=float(loo),
                   span_ppm=float(pred.max() - pred.min()), share=share)
