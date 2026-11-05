"""The per-spectrum satellite test (assignment/satellites.py: satellite_fit, own_13c_series,
satellite_verdict): a line beside a brighter one is an artefact when the partner explains
its height across the spectra (a fit through the origin, centred R^2 >= SAT_FIT_R2 on >=
SAT_FIT_MIN_N spectra) and it has no 13C line of its own. Every parameter is pinned."""
import numpy as np

from peaky.assignment import satellites as SAT
from peaky.chem import isotopes as ISO


def _ramp(n=12, slope=0.4, noise=0.03, seed=3):
    rng = np.random.default_rng(seed)
    p = 10 ** np.linspace(4, 6, n)
    return p, slope * p * (1 + rng.normal(0, noise, n))


def test_the_fit_goes_through_the_origin():
    p = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
    f = SAT.satellite_fit(p, 0.25 * p)
    assert f["n"] == 5 and abs(f["slope"] - 0.25) < 1e-12 and abs(f["r2"] - 1) < 1e-12
    # an offset line (s = p + 10) is not proportional: the through-origin fit leaves a residual
    g = SAT.satellite_fit(p, p + 10.0)
    assert g["r2"] < 0.9


def test_only_the_spectra_holding_both_lines_count():
    p = np.array([1e5, 2e5, np.nan, 0.0, 4e5, 5e5, 6e5])
    s = np.array([4e4, 8e4, 1e4, 1e4, 0.0, 2e5, 2.4e5])
    assert SAT.satellite_fit(p, s)["n"] == 4


def test_the_centred_r2_needs_the_parent_to_explain_the_variation():
    # a flat parent (+-5 %) and a satellite scattering 10 % about 0.4x: uncentred R^2 near 1,
    # centred far under SAT_FIT_R2 -> not explained
    rng = np.random.default_rng(1)
    p = 1e6 * (1 + 0.05 * np.sin(np.arange(20)))
    s = 0.4 * p * (1 + rng.normal(0, 0.1, 20))
    f = SAT.satellite_fit(p, s)
    assert f["r2_uncentred"] > 0.98 and f["r2"] < 0.5
    assert not SAT.satellite_verdict(p, s)["explained"]


def _lin(seed, noise=0.12, n=12):
    rng = np.random.default_rng(seed)
    p = np.linspace(1e5, 3e5, n)
    return p, 0.4 * p * (1 + rng.normal(0, noise, n))


def test_the_r2_threshold():                                                       # SAT_FIT_R2
    assert SAT.SAT_FIT_R2 == 0.9
    v = SAT.satellite_verdict(*_lin(7))                  # R2 0.94
    assert 0.92 < v["r2"] < 0.95 and v["artefact"]
    w = SAT.satellite_verdict(*_lin(4))                  # R2 0.88
    assert 0.85 < w["r2"] < 0.9 and not w["explained"] and not w["artefact"]


def test_the_fewest_spectra():                                                     # SAT_FIT_MIN_N
    assert SAT.SAT_FIT_MIN_N == 5
    p, s = _ramp(n=5, noise=0.0)
    assert SAT.satellite_verdict(p, s)["artefact"]
    p4, s4 = _ramp(n=4, noise=0.0)
    v = SAT.satellite_verdict(p4, s4)
    assert not v["explained"] and "not testable" in v["why"]


def test_an_own_13c_line_makes_it_an_ion_and_an_untestable_one_does_not():
    p, s = _ramp()
    assert not SAT.satellite_verdict(p, s, own_13c=True)["artefact"]
    assert SAT.satellite_verdict(p, s, own_13c=False)["artefact"]
    v = SAT.satellite_verdict(p, s, own_13c=None)
    assert v["artefact"] and "untestable" in v["why"]


def test_the_own_13c_band():                                                       # SAT_OWN_13C_BAND
    assert SAT.SAT_OWN_13C_BAND == (0.5, 2.0)
    s = np.array([1e4, 2e4, 3e4, 4e4, 5e4])
    share = ISO.R_13C_PER_C * 10
    assert SAT.own_13c_series(s, s * share * 1.0, 10) is True
    assert SAT.own_13c_series(s, s * share * 0.45, 10) is False
    assert SAT.own_13c_series(s, s * share * 0.55, 10) is True
    assert SAT.own_13c_series(s, s * share * 2.1, 10) is False
    assert SAT.own_13c_series(s, np.zeros(5), 10) is False
    assert SAT.own_13c_series(s, s * share, None) is None                          # no carbon count
    assert SAT.own_13c_series(s, s * share, 10, testable=np.zeros(5, bool)) is None
    # the testable mask decides which spectra speak: a line seen only where testable
    t = np.array([True, True, False, False, False])
    c = np.where(t, s * share, 0.0)
    assert SAT.own_13c_series(s, c, 10, testable=t) is True


def test_a_sub_linear_follower_is_not_explained_and_its_exponent_is_reported():
    # s = 0.4 p^0.42 (the measured exponent of the uronium-mode partners): it follows its parent, but not
    # proportionally -- the through-origin test does not explain it; the log-log fit reads the exponent
    p = 10 ** np.linspace(4, 6, 30)
    s = 0.4 * p ** 0.42
    v = SAT.satellite_verdict(p, s)
    assert not v["explained"] and abs(v["loglog_slope"] - 0.42) < 1e-6 and v["loglog_r2"] > 0.999
    assert "does not follow its parent proportionally" in v["why"] and "log-log exponent 0.42" in v["why"]
    q = SAT.satellite_verdict(p, 0.25 * p)
    assert q["explained"] and abs(q["loglog_slope"] - 1.0) < 1e-9 and "log-log exponent 1.00" in q["why"]
