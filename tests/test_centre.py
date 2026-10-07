"""batch/centre.py -- the adaptive trace-centre estimator, on synthetic series
with known noise and drift. Run: pytest tests/test_centre.py"""
import numpy as np
import pandas as pd
import pytest

from peaky.batch import centre as CE
from peaky.batch import traces as TR


def _series(rng, n, sigma, gamma):
    walk = np.cumsum(rng.normal(0.0, gamma, n)) if gamma > 0 else np.zeros(n)
    return rng.normal(0.0, sigma, n) + walk, walk


@pytest.mark.parametrize("sigma,gamma", [(3.0, 0.5), (2.0, 0.2), (4.0, 1.0)])
def test_noise_and_drift_recover_sigma_and_gamma_within_15_percent(sigma, gamma):
    rng = np.random.default_rng(1)
    est = np.array([CE.noise_and_drift(_series(rng, 1500, sigma, gamma)[0])[:2]
                    for _ in range(12)])
    s, g = np.median(est, axis=0)
    assert abs(s - sigma) / sigma < 0.15, (s, sigma)
    assert abs(g - gamma) / gamma < 0.15, (g, gamma)


def test_pure_noise_reads_zero_drift_and_unit_rise():
    rng = np.random.default_rng(2)
    x, _ = _series(rng, 1500, 2.0, 0.0)
    s, g, rise = CE.noise_and_drift(x)
    assert g == 0.0
    assert abs(rise - 1.0) < 0.15


def test_optimal_window_is_within_1p5x_of_the_empirical_optimum():
    """The rolling-median centre error, measured against the walk that made the
    series, is minimised near WINDOW_K * sigma / gamma."""
    rng = np.random.default_rng(3)
    sigma, gamma, n = 3.0, 0.5, 1200
    errs = {}
    for W in (5, 9, 15, 21, 29, 41, 61, 91):
        e = []
        for _ in range(12):
            x, walk = _series(rng, n, sigma, gamma)
            c = np.array([np.median(x[max(0, i - W // 2):i + W // 2 + 1]) for i in range(n)])
            e.append(np.sqrt(np.mean((c - walk) ** 2)))
        errs[W] = float(np.mean(e))
    w_emp = min(errs, key=errs.get)
    w_star = CE.optimal_window(sigma, gamma, n)
    assert w_star / w_emp < 1.5 and w_emp / w_star < 1.5, (w_star, w_emp, errs)
    # and rolling at W* really beats the batch median
    assert errs[min(errs, key=lambda w: abs(w - w_star))] < np.hypot(sigma / np.sqrt(n), gamma * np.sqrt(n / 24))


def test_pure_noise_never_rolls_and_a_walk_always_does():
    rng = np.random.default_rng(4)
    ref = 200.0
    x, _ = _series(rng, 800, 3.0, 0.0)
    c, info = CE.trace_centre(ref * (1 + x * 1e-6))
    assert info["scheme"] == "global"
    assert np.allclose(c, np.median(ref * (1 + x * 1e-6)))
    x, _ = _series(rng, 800, 3.0, 0.8)
    c, info = CE.trace_centre(ref * (1 + x * 1e-6))
    assert info["scheme"] == "rolling"
    assert 5 <= info["window"] <= 800 // 3
    assert info["window"] % 2 == 1
    assert info["e_roll"] < info["e_global"] / 1.15


def test_below_min_n_the_centre_is_the_batch_median_bit_for_bit():
    rng = np.random.default_rng(5)
    x, _ = _series(rng, 30, 3.0, 1.0)            # a strong walk, but too short
    mz = 300.0 * (1 + x * 1e-6)
    c, info = CE.trace_centre(mz, min_n=40)
    assert info["scheme"] == "global"
    assert np.all(c == np.median(mz))


def test_nan_positions_are_skipped_and_returned_as_nan():
    rng = np.random.default_rng(6)
    x, _ = _series(rng, 200, 2.0, 0.0)
    mz = 150.0 * (1 + x * 1e-6)
    mz[[3, 40, 199]] = np.nan
    c, info = CE.trace_centre(mz)
    assert info["n"] == 197
    assert np.isnan(c[[3, 40, 199]]).all() and np.isfinite(c[0])


def test_the_gap_guard_keeps_a_window_from_spanning_a_gap():
    rng = np.random.default_rng(7)
    n = 600
    x, _ = _series(rng, n, 2.0, 0.6)
    t = np.arange(n, dtype=float)
    t[300:] += 500.0                              # a 500-h hole in the batch
    x[300:] += 40.0                               # ... across which the ion sits elsewhere
    mz = 250.0 * (1 + x * 1e-6)
    c_guard, info = CE.trace_centre(mz, t, window=41)
    c_free, _ = CE.trace_centre(mz, None, window=41)
    # with the guard, the centres just before the gap ignore the far side
    edge = slice(290, 300)
    assert np.all(np.abs(c_guard[edge] - np.median(mz[250:300])) < np.abs(c_free[edge] - np.median(mz[250:300])).max() + 1e-9)
    assert (c_guard[edge] < c_free[edge]).all()


def _synthetic_index(rng, n_spectra=300, sigma_ppm=1.0, gamma_ppm=0.5, ion=200.0):
    """A PeakIndex with one drifting ion plus uniform noise peaks."""
    x, walk = _series(rng, n_spectra, sigma_ppm, gamma_ppm)
    rows = [dict(sample_item_id=f"s{i:04d}", mz=ion * (1 + x[i] * 1e-6), height=10.0)
            for i in range(n_spectra)]
    for i in range(n_spectra):                    # 20 noise peaks per spectrum
        for m in rng.uniform(190.0, 210.0, 20):
            rows.append(dict(sample_item_id=f"s{i:04d}", mz=m, height=1.0))
    df = pd.DataFrame(rows)
    idx = TR.PeakIndex(df, tol_ppm=6.0)
    times = np.arange(idx.n_samples, dtype=float)
    return idx, times, ion * (1 + walk * 1e-6)


def test_rolling_members_follow_the_track_not_the_reference_centre():
    rng = np.random.default_rng(8)
    idx, times, track = _synthetic_index(rng, gamma_ppm=1.0)
    t = np.arange(len(track), dtype=float)
    mem = CE.rolling_members(idx, t, track, times, tol_ppm=6.0, c_ref=float(np.median(track)))
    got = idx.mz[mem]
    # (almost) every spectrum is found -- sigma is 1 ppm, so a 6 ppm window
    # holds all but the rare >6-sigma draw -- and the members hug the track
    assert len(mem) >= 0.99 * idx.n_samples
    d = np.abs(got - track[idx.sample[mem]]) / track[idx.sample[mem]] * 1e6
    assert d.max() <= 6.0
    # whereas a FIXED window at the reference centre loses the far end of a
    # 300-spectrum walk at 1 ppm/spectrum
    fixed = idx.members(float(np.median(track)), 6.0)
    assert len(fixed) < len(mem)


def test_batch_centres_one_row_per_seed_and_deterministic():
    rng = np.random.default_rng(9)
    idx, times, track = _synthetic_index(rng)
    seeds = [float(np.median(track)), 205.0, 999.0]  # the ion, a noise region, nothing
    a = CE.batch_centres(idx, seeds, times, tol_ppm=6.0)
    b = CE.batch_centres(idx, seeds, times, tol_ppm=6.0)
    assert list(a.columns) == ["seed_mz", "centre_mz", "n", "occurrence", "scheme", "window",
                               "sigma_ppm", "gamma_ppm", "rise", "resid_ppm", "se_ppm",
                               "members", "centres", "times"]
    assert len(a) == 2                             # the 999 seed has no members
    assert a.iloc[0]["occurrence"] >= 0.99
    assert np.array_equal(a.iloc[0]["members"], b.iloc[0]["members"])
    assert a.iloc[0]["centre_mz"] == b.iloc[0]["centre_mz"]
    assert abs(a.iloc[0]["centre_mz"] - np.median(track)) / 200.0 * 1e6 < 1.0


def test_batch_centres_empty_input():
    rng = np.random.default_rng(10)
    idx, times, _ = _synthetic_index(rng, n_spectra=50)
    out = CE.batch_centres(idx, [], times, tol_ppm=6.0)
    assert len(out) == 0 and "centre_mz" in out.columns
