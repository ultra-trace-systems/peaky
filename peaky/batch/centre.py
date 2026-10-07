"""The adaptive trace-centre estimator: where an ion's m/z sits in each spectrum
of a batch, from its own position series.

A trace's per-spectrum positions carry white noise `sigma` (the picker's
centroid jitter, per spectrum) and a slow random walk `gamma` (the step, per
spectrum, of whatever moves the position: an unresolved neighbour's changing
abundance, a drifting axis). One centre for the whole batch averages over
something that has already moved; a centre per spectrum has only the noise of
its window. A rolling median over W spectra has error

    e(W)^2 = sigma^2 / W + gamma^2 * W / 24

which is minimised at W* = sqrt(24) * sigma / gamma. Both terms come from the
trace's own structure function

    S(k) = 0.5 * robust_var(x[i+k] - x[i])  =  sigma^2 + 0.5 * gamma^2 * k

so nothing here needs a user parameter or an assumption about batch length.
The constant sqrt(24) ~ 4.9 is the one the cross-instrument validation settled
on; the rolling-median optimum by the median's pi/2 efficiency argument is
sqrt(6 pi) ~ 4.3, and the error curve is flat between the two (the measured
centre error at either is within 3 % of the minimum).

The module is pure numpy, deterministic, and never worse than today: when the
series is short, the walk is unresolvable, or the predicted gain is small, the
centre is the plain batch median, bit for bit.

    noise_and_drift(x)          sigma, gamma, rise
    optimal_window(sigma, gamma, n)
    trace_centre(mz, t=None)    centre per spectrum + an info dict
    rolling_members(...)        the members of a drift-following trace
    batch_centres(index, seeds, times)   one row per trace over a PeakIndex

What it will not fix: accuracy against theory. A static per-ion offset (an
unresolved isobar displacing the centroid) survives any amount of centring --
centring buys precision, only resolution buys accuracy.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__version__ = "0.1.0"
__all__ = ["noise_and_drift", "optimal_window", "trace_centre", "rolling_members",
           "batch_centres", "rsd", "WINDOW_K"]

WINDOW_K = float(np.sqrt(24.0))   # W* = WINDOW_K * sigma / gamma
MAD_TO_SIGMA = 1.4826


def rsd(x) -> float:
    """Robust sd: 1.4826 x the median absolute deviation."""
    x = np.asarray(x, dtype=float)
    return float(MAD_TO_SIGMA * np.median(np.abs(x - np.median(x))))


def noise_and_drift(x, kmax: int = 32) -> tuple[float, float, float]:
    """(sigma, gamma, rise) of a position series from its structure function.

    sigma  white noise per point (the same units as `x`)
    gamma  random-walk step per point; 0 when the series carries no walk
    rise   S(kmax) / S(1) -- the evidence that anything drifts at all (1 for
           pure noise, growing with the walk)

    NaN throughout when the series is too short for at least four lags with
    more than eight differences each."""
    x = np.asarray(x, dtype=float)
    S: dict[int, float] = {}
    for k in range(1, min(kmax, max(len(x) // 4, 1)) + 1):
        d = x[k:] - x[:-k]
        if len(d) > 8:
            S[k] = 0.5 * rsd(d) ** 2
    if len(S) < 4:
        return float("nan"), float("nan"), float("nan")
    ks = np.array(sorted(S), dtype=float)
    sv = np.array([S[k] for k in ks])
    s1 = max(float(sv[0]), 1e-12)
    den = float(np.sum((ks - 1) ** 2))
    slope = max(float(np.sum((ks - 1) * (sv - s1)) / den), 0.0) if den > 0 else 0.0
    # S(k) = sigma^2 + 0.5 gamma^2 k  ->  slope = 0.5 gamma^2 and S(1) = sigma^2 + slope
    gamma2 = 2.0 * slope
    sigma2 = max(s1 - slope, 1e-12)
    return float(np.sqrt(sigma2)), float(np.sqrt(gamma2)), float(sv[-1] / s1)


def optimal_window(sigma: float, gamma: float, n: int) -> float:
    """W* = WINDOW_K * sigma / gamma, clipped to [5, n]; n when there is no walk."""
    if not np.isfinite(gamma) or gamma <= 0 or not np.isfinite(sigma):
        return float(n)
    return float(np.clip(WINDOW_K * sigma / gamma, 5, n))


def _predicted_error(sigma: float, gamma: float, W: float) -> float:
    return float(np.hypot(sigma / np.sqrt(W), gamma * np.sqrt(W / 24.0)))


def _rolling_median(x: np.ndarray, W: int, t=None, max_span=None) -> np.ndarray:
    """Centred rolling median over W points; with `t` and `max_span` a window
    is shrunk (from the far end) so it never spans more than `max_span` in
    time -- a gap in the batch must not stretch it across the gap."""
    n = len(x)
    h = W // 2
    out = np.empty(n)
    for i in range(n):
        a = max(0, min(i - h, n - W))
        b = min(n, a + W)
        if t is not None and max_span is not None:
            lo, hi = a, b
            while hi - lo > 5 and (t[hi - 1] - t[lo]) > max_span:
                if (t[i] - t[lo]) > (t[hi - 1] - t[i]):
                    lo += 1
                else:
                    hi -= 1
            a, b = lo, hi
        out[i] = np.median(x[a:b])
    return out


def trace_centre(mz, t=None, *, min_n: int = 40, min_gain: float = 1.15,
                 min_rise: float = 1.3, max_span_factor: float = 3.0,
                 window: int | None = None) -> tuple[np.ndarray, dict]:
    """The centre of one trace, per spectrum.

    mz      the trace's positions (Th), one per spectrum, NaN allowed; in
            spectrum-time order
    t       optional times (datetime64 or float hours), same length; used only
            for the gap guard
    window  force a rolling window of this many spectra (scheme 'forced')

    Returns (centre, info): `centre` is an array like `mz` (NaN where `mz` is);
    `info` carries n, scheme ('global' | 'rolling' | 'forced'), window, sigma,
    gamma, rise, e_global, e_roll (predicted errors), resid_ppm (robust sd of
    the members about their own centre -- what should size this trace's
    stamping window) and se_ppm.

    The centre ROLLS only when all of: n >= min_n, a walk is resolvable
    (gamma > 0 and S(kmax)/S(1) >= min_rise), and the predicted error of the
    rolling centre beats the batch median by >= min_gain. Otherwise the centre
    is the batch median of the positions, exactly what a batch had before."""
    x = np.asarray(mz, dtype=float)
    good = np.isfinite(x)
    n = int(good.sum())
    info = dict(n=n, scheme="global", window=n, sigma=np.nan, gamma=np.nan, rise=np.nan,
                e_global=np.nan, e_roll=np.nan, resid_ppm=np.nan, se_ppm=np.nan)
    if n == 0:
        return np.full(len(x), np.nan), info
    ref = float(np.median(x[good]))
    ppm = ref > 1.0                       # work in ppm when the series is in Th
    y = (x[good] - ref) / ref * 1e6 if ppm else x[good] - ref
    tt = None
    if t is not None:
        tt = np.asarray(t)[good]
        if np.issubdtype(tt.dtype, np.datetime64):
            tt = (tt - tt[0]) / np.timedelta64(1, "h")
        tt = tt.astype(float)
    sigma, gamma, rise = noise_and_drift(y)
    info.update(sigma=sigma, gamma=gamma, rise=rise)
    W = n
    if window is not None:
        W = int(np.clip(window, 5, n))
        info["scheme"] = "forced"
    elif (n >= min_n and np.isfinite(gamma) and gamma > 0
          and np.isfinite(rise) and rise >= min_rise):
        Wo = optimal_window(sigma, gamma, n)
        e_glob = _predicted_error(sigma, gamma, n)
        e_roll = _predicted_error(sigma, gamma, Wo)
        info.update(e_global=e_glob, e_roll=e_roll)
        if e_glob / max(e_roll, 1e-9) >= min_gain:
            W = int(np.clip(round(Wo), 5, max(5, n // 3))) | 1     # odd, <= n/3
            info["scheme"] = "rolling"
    info["window"] = W
    span = None
    if tt is not None and len(tt) > 2:
        span = max_span_factor * float(np.median(np.diff(tt))) * W
    c = np.full(n, np.median(y)) if W >= n else _rolling_median(y, W, tt, span)
    cy = np.full(len(x), np.nan)
    cy[good] = c
    out = (cy / 1e6 * ref + ref) if ppm else cy + ref
    if n >= 5:
        res = (x[good] - out[good]) / out[good] * 1e6 if ppm else x[good] - out[good]
        info["resid_ppm"] = rsd(res)
        info["se_ppm"] = float(info["resid_ppm"] / np.sqrt(max(W, 1)))
    return out, info


def rolling_members(index, centre_t, centre_c, times_by_code, tol_ppm: float,
                    c_ref: float, widen: float = 2.0) -> np.ndarray:
    """Members of a DRIFT-FOLLOWING trace: for every spectrum, the nearest peak
    to that spectrum's own interpolated centre, within +-tol_ppm.

    index          a batch.traces.PeakIndex
    centre_t/_c    times / centres the rolling estimate is known at
    times_by_code  float time per PeakIndex sample code (same units as centre_t)
    c_ref          the trace's reference centre, sizing the search window
    widen          how much wider than tol the search window is (the track may
                   wander beyond +-tol of the reference centre)"""
    hw = c_ref * tol_ppm * widen * 1e-6
    i0, i1 = index.window(c_ref - hw, c_ref + hw)
    if i1 <= i0:
        return np.empty(0, dtype=np.int64)
    mz = index.mz[i0:i1]
    sc = index.sample[i0:i1]
    ct = np.asarray(centre_t, dtype=float)
    cc = np.asarray(centre_c, dtype=float)
    o = np.argsort(ct)
    ct, cc = ct[o], cc[o]
    cen = np.interp(times_by_code[sc], ct, cc)         # flat outside the range
    d = np.abs(mz - cen) / cen * 1e6
    keep = d <= tol_ppm
    if not keep.any():
        return np.empty(0, dtype=np.int64)
    idxs = np.nonzero(keep)[0]
    order = np.lexsort((d[idxs], sc[idxs]))            # per sample, nearest first
    ss = sc[idxs][order]
    first = np.ones(len(order), dtype=bool)
    first[1:] = ss[1:] != ss[:-1]
    return (i0 + idxs[order][first]).astype(np.int64)


def batch_centres(index, seeds, times_by_code, *, tol_ppm: float, refine: bool = True,
                  **kw) -> pd.DataFrame:
    """Run the estimator across a whole PeakIndex: one row per seed centre with
    seed_mz, centre_mz (median of the track), n, occurrence, scheme, window,
    sigma_ppm, gamma_ppm, rise, resid_ppm, se_ppm, and the arrays members /
    centres / times. With `refine`, a rolling trace re-collects its members
    along its own track (`rolling_members`) and is centred again. Seeds with
    no members are skipped. Deterministic for a given index and seed order."""
    rows = []
    for c0 in np.asarray(seeds, dtype=float):
        mem = index.members(c0, tol_ppm)
        if len(mem) == 0:
            continue
        t = times_by_code[index.sample[mem]]
        o = np.argsort(t, kind="mergesort")
        mem, t = mem[o], t[o]
        cen, info = trace_centre(index.mz[mem], t, **kw)
        if refine and info["scheme"] == "rolling":
            mem2 = rolling_members(index, t, cen, times_by_code, tol_ppm,
                                   float(np.median(cen)))
            if len(mem2) >= 5:
                t2 = times_by_code[index.sample[mem2]]
                o2 = np.argsort(t2, kind="mergesort")
                mem, t = mem2[o2], t2[o2]
                cen, info = trace_centre(index.mz[mem], t, **kw)
        rows.append(dict(seed_mz=float(c0), centre_mz=float(np.median(cen)),
                         n=int(len(mem)), occurrence=len(mem) / index.n_samples,
                         scheme=info["scheme"], window=int(info["window"]),
                         sigma_ppm=info["sigma"], gamma_ppm=info["gamma"],
                         rise=info["rise"], resid_ppm=info["resid_ppm"],
                         se_ppm=info["se_ppm"], members=mem, centres=cen, times=t))
    cols = ["seed_mz", "centre_mz", "n", "occurrence", "scheme", "window", "sigma_ppm",
            "gamma_ppm", "rise", "resid_ppm", "se_ppm", "members", "centres", "times"]
    return pd.DataFrame(rows, columns=cols)
