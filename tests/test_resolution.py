"""chem.resolution: the width model (moved out of batch.tracefirst, which
re-exports it) and the nearest-neighbour separability classes."""
import numpy as np
import pytest

from peaky.batch import tracefirst as TFT
from peaky.chem import resolution as RES


def test_the_width_model_is_one_object_shared_by_tracefirst_and_the_stage():
    assert TFT.Resolution is RES.Resolution
    assert TFT.classify_pair is RES.classify_pair and TFT.d_crit_hwhm is RES.d_crit_hwhm
    assert TFT.hwhm is RES.hwhm and TFT.dedup_ppm is RES.dedup_ppm
    assert TFT.FIT_FLOOR_HWHM == RES.FIT_FLOOR_HWHM == 0.4


def test_a_model_round_trips_through_its_summary_record():
    m = RES.Resolution(coef=3.7e-7, exponent=1.59, n_peaks=10, r_spread=(87000.0, 122000.0), source="measured")
    back = RES.Resolution.from_dict(m.as_dict())
    assert back == m
    assert RES.Resolution.from_dict({"coef": 1.0 / 6500}).r_at(200.0) == pytest.approx(6500.0)


def test_nearest_neighbour_classes_flag_isolated_resolved_blended_and_unresolvable():
    R = 10000.0
    hw = RES.hwhm(302.0, R)                       # ~0.0151 Th
    # name -> (m/z, height): A isolated; C/D resolved (5 HWHM apart); E/F blended
    # (1.5 HWHM: a fit could separate them, a picker cannot); G/H unresolvable (0.2)
    mz = {"A": 300.0, "C": 301.0, "D": 301.0 + 5.0 * hw, "E": 302.0, "F": 302.0 + 1.5 * hw,
          "G": 303.0, "H": 303.0 + 0.2 * hw}
    names = ["F", "A", "H", "C", "E", "D", "G"]          # deliberately unsorted
    out = RES.nearest_neighbour_classes([mz[n] for n in names], [1e4] * len(names), R)
    cls = dict(zip(names, out["resolvability"]))
    assert cls == {"A": "isolated", "C": "resolved", "D": "resolved", "E": "blended",
                   "F": "blended", "G": "unresolvable", "H": "unresolvable"}
    sep = dict(zip(names, out["sep_hwhm"]))
    assert np.isnan(sep["A"])
    assert sep["D"] == pytest.approx(5.0, rel=0.02) and sep["F"] == pytest.approx(1.5, rel=0.02)
    assert sep["H"] == pytest.approx(0.2, rel=0.05)
    # equal heights become bimodal at 2 sigma = 1.70 HWHM, and that threshold is on the row
    assert dict(zip(names, out["d_crit_hwhm"]))["F"] == pytest.approx(2.0 / 1.1774396, rel=1e-3)
    # the neighbour index points back at the partner, in INPUT positions
    nb = dict(zip(names, out["neighbour"]))
    assert names[nb["E"]] == "F" and names[nb["F"]] == "E" and nb["A"] == -1


def test_the_class_follows_the_height_ratio_and_the_model_scales_with_mass():
    # a dim neighbour needs MORE separation to be bimodal: the same 3-HWHM gap
    # is resolved for equal heights and blended against a 1:100 partner
    R = 10000.0
    hw = RES.hwhm(400.0, R)
    equal = RES.classify_pair(400.0, 1e4, 400.0 + 3.0 * hw, 1e4, R)
    dim = RES.classify_pair(400.0, 1e4, 400.0 + 3.0 * hw, 1e2, R)
    assert equal["resolvability"] == "resolved" and dim["resolvability"] == "blended"
    assert dim["d_crit_hwhm"] > equal["d_crit_hwhm"]
    # an Orbitrap model widens as m^1.5: the same Th gap is more HWHM at low mass
    orbi = RES.Resolution(coef=3.7e-7, exponent=1.59)
    assert orbi.hwhm(150.0) < orbi.hwhm(450.0) / 4


def test_fewer_than_two_peaks_are_isolated_and_a_nan_height_reads_as_zero():
    one = RES.nearest_neighbour_classes([300.0], [1.0], 6500.0)
    assert one["resolvability"] == ["isolated"] and one["neighbour"][0] == -1
    hw = RES.hwhm(300.0, 6500.0)
    out = RES.nearest_neighbour_classes([300.0, 300.0 + 1.0 * hw], [1e4, float("nan")], 6500.0)
    # a zero-height partner is the dimmest possible ratio: still classified, never NaN
    assert out["resolvability"][0] in ("blended", "resolved") and np.isfinite(out["sep_hwhm"][0])


def _n_maxima(d_hwhm: float, ratio: float) -> int:
    """Local maxima of G(x) + ratio * G(x - d), counted on a fine grid. x is in
    HWHM: exp(-ln2 x^2) is at half height at x = 1."""
    x = np.linspace(-6.0, 14.0, 200001)
    y = np.exp(-np.log(2.0) * x ** 2) + ratio * np.exp(-np.log(2.0) * (x - d_hwhm) ** 2)
    dy = np.diff(y)
    return int(np.sum((dy[:-1] > 0) & (dy[1:] <= 0)))


@pytest.mark.parametrize("ratio", [1.0, 0.5, 0.1, 1.0 / 50, 0.01, 0.001])
def test_d_crit_is_where_two_gaussians_of_that_height_ratio_turn_bimodal(ratio):
    # brute force, in HWHM directly (a Gaussian at half height 1 HWHM from its
    # centre): one maximum just inside the threshold, two just outside. The
    # threshold was once 1.39x too wide (sigma -> HWHM multiplied where it
    # divides), which stamped resolvable peaks `blended`.
    dc = RES.d_crit_hwhm(ratio)
    assert _n_maxima(0.98 * dc, ratio) == 1
    assert _n_maxima(1.02 * dc, ratio) == 2


def test_d_crit_matches_the_textbook_values():
    assert RES.d_crit_hwhm(1.0) == pytest.approx(1.699, abs=2e-3)       # 2 sigma
    assert RES.d_crit_hwhm(1.0 / 50) / 2.0 == pytest.approx(1.65, abs=5e-3)  # in FWHM
