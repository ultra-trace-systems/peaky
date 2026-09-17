"""batch/wave.py -- the mass-offset wave."""
import numpy as np

from peaky.batch import wave as W


def _mz(n=25, lo=60.0, hi=320.0, seed=0):
    rng = np.random.default_rng(seed)
    return np.sort(rng.uniform(lo, hi, n)), rng


def test_a_flat_offset_is_fitted_as_k_zero_nearly_always():
    """K = 0 must be reachable AND chosen on a flat series: a fixed 5 % CV margin
    let noise buy a degree on 25 points; the one-SE margin does not (a chance
    slope beyond 2 SE still can, so the claim is 'nearly always', not 'always')."""
    ks, c0 = [], []
    for seed in range(20):
        mz, rng = _mz(seed=seed)
        w = W.fit_wave(mz, 3.0 + rng.normal(0, 0.2, len(mz)))
        ks.append(w.K)
        c0.append(w.coef[0])
    assert sum(k == 0 for k in ks) >= 17, ks
    assert all(abs(c - 3.0) < 0.25 for c in c0)
    assert all(W.fit_wave(*(_mz(seed=s)[0], 3.0 + np.zeros(25))).span_ppm < 1e-9 for s in range(3))


def test_a_smooth_curve_in_sqrt_mz_is_fitted_and_removed():
    mz, rng = _mz(n=30)
    u = (np.sqrt(mz) - np.sqrt(60)) / (np.sqrt(320) - np.sqrt(60))
    ppm = 1.0 + 8.0 * u - 3.0 * u ** 2 + rng.normal(0, 0.2, len(mz))
    w = W.fit_wave(mz, ppm, tof=True)
    assert w is not None and 1 <= w.K <= 3
    assert w.resid_ppm < 0.4 and w.raw_ppm > 0.8
    assert w.share > 0.9 and w.span_ppm > 4.0
    corr = W.rsd((ppm - w.predict(mz)))
    assert corr < 0.4
    # correct() moves each m/z by -delta ppm
    assert np.allclose((w.correct(mz) / mz - 1) * 1e6, -w.predict(mz), atol=1e-6)


def test_predict_refuses_outside_the_calibrant_range_unless_told():
    mz, rng = _mz()
    w = W.fit_wave(mz, 2.0 + rng.normal(0, 0.1, len(mz)))
    assert np.isnan(w.predict([30.0, 900.0])).all()
    assert np.isfinite(w.predict([30.0, 900.0], extrapolate=True)).all()
    assert np.allclose(w.correct([30.0]), [30.0])            # untouched outside


def test_an_outlier_is_clipped_and_counted():
    mz, rng = _mz()
    ppm = 1.0 + rng.normal(0, 0.1, len(mz))
    ppm[7] += 12.0
    w = W.fit_wave(mz, ppm)
    assert w.n_clipped == 1 and w.n == len(mz) - 1
    assert abs(w.coef[0] - 1.0) < 0.1
    # the clip is about the MEDIAN residual: a clip about zero would have kept
    # the outlier (which pulls the constant toward itself) and dropped the rest
    assert abs(np.median(w.predict(mz) - ppm) ) < 0.1


def test_too_few_calibrants_returns_none():
    assert W.fit_wave([100.0, 200.0, 300.0], [1.0, 1.0, 1.0]) is None


def test_orbitrap_exponent_is_minus_one_half():
    mz, rng = _mz()
    w = W.fit_wave(mz, rng.normal(0, 0.1, len(mz)), tof=False)
    assert w.p == -0.5
    v = mz ** -0.5
    assert w.domain == (float(v.min()), float(v.max()))      # in v, which falls with m/z
    assert np.isfinite(w.predict(mz)).all()


def test_a_clip_that_would_leave_too_few_calibrants_is_refused():
    """The returned fit must stand on at least `min_n` calibrants.

    The old loop recorded the fit against the mask the NEXT clip proposed, so a
    clip cutting the set below min_n still returned -- on a real Orbitrap batch,
    a degree-2 wave standing on 4 surviving points out of 7."""
    rng = np.random.default_rng(5)
    mz = np.linspace(170.0, 221.0, 8)
    ppm = rng.normal(0.0, 0.05, len(mz))
    ppm[:4] += np.array([6.0, -6.0, 5.0, -5.0])        # four gross outliers
    w = W.fit_wave(mz, ppm, tof=False, min_n=6)
    assert w is None or w.n >= 6
    # and the fit never reports more surviving points than it has calibrants
    if w is not None:
        assert w.n + w.n_clipped == len(mz)


def test_the_explained_share_is_measured_on_the_points_the_fit_was_judged_on():
    """`share` compared a clipped residual against the UNclipped spread, so any
    clip at all flattered it -- a 4-point degree-2 fit on a real Orbitrap batch
    read 'explains 100%'."""
    rng = np.random.default_rng(6)
    mz = np.linspace(100.0, 400.0, 16)
    u = 2 * (mz ** 0.5 - (mz ** 0.5).min()) / ((mz ** 0.5).max() - (mz ** 0.5).min()) - 1
    ppm = 2.0 * u + rng.normal(0.0, 0.05, len(mz))     # a real degree-1 wave
    ppm[3] += 15.0                                     # one gross outlier
    w = W.fit_wave(mz, ppm, tof=True, min_n=6, k_max=2)
    assert w is not None and w.n_clipped >= 1          # the outlier is dropped
    assert w.n >= 6 and w.n + w.n_clipped == len(mz)
    # raw is the SURVIVORS' spread, so it no longer carries the outlier's 15 ppm
    assert w.raw_ppm < 3.0
    # and the identity behind `share` holds on one consistent set
    assert abs(w.share - max(0.0, 1 - (w.resid_ppm / w.raw_ppm) ** 2)) < 1e-9
    assert w.share > 0.9                               # the wave really is there


def test_noise_with_one_outlier_does_not_read_as_an_explained_wave():
    """The same denominator bug let pure noise plus an outlier report a large
    explained share, because the outlier inflated the raw it was measured against."""
    rng = np.random.default_rng(7)
    mz = np.linspace(100.0, 400.0, 16)
    ppm = rng.normal(0.0, 0.10, len(mz))
    ppm[3] += 15.0
    w = W.fit_wave(mz, ppm, tof=True, min_n=6, k_max=2)
    assert w is not None
    assert w.share < 0.5                               # there is no wave to explain
