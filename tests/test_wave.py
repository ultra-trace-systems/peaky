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
