"""The edge carry on the batch-pooled tier calibration (assignment/calpool.py).

A file whose own core cannot calibrate the tier engine is re-tiered in a window the
batch's pooled core lends it. Under a sub-ppm batch axis model's edge carry
(PassConfig.axis_edge_carry; tests/test_axis_edge_carry.py) that lent window judges a
line just outside the model's scope at its centre plus the model's edge too, as the
file's own calibration would (tiers.compute_tiers attaches the carry to a calibration
from elsewhere; calpool.apply_pool lends it to the degeneracy re-run as well).

Synthetic ledgers only.
"""

from __future__ import annotations

import pytest

from peaky.assignment import calpool as CP
from peaky.assignment import tiers as T
from peaky.assignment.passes import PassConfig
from peaky.batch import axislock as AL
from tests.test_calpool import AD, NEUTRALS, _batch, _file

#: the axis model's edge carry: lines up to 30 Da below its first lock (m/z 105) read +3 ppm
REC = {"reach_da": 30.0, "slack_ppm": AL.SCOPE_SLACK_PPM, "below": {"mz": 105.0, "ppm": 3.0},
       "above": {"mz": 400.0, "ppm": 0.0}}


def _pooled(carry):
    pool = CP.pool_calibration(_batch())
    rows = [(nf, AD, 0.0, 1e5, False, "Good") for nf in NEUTRALS[:6]]
    rows[0] = (NEUTRALS[0], AD, 3.0, 1e5, False, "Good")     # m/z 101: below the first lock, 3 ppm off
    at = _file("u", rows, tier=False)
    led = at.copy(deep=True)
    cfg = PassConfig()
    cfg.axis_edge_carry = carry
    T.apply_tiers(led, cfg=cfg)
    inputs = CP.PoolInputs(at_tiers=at, cfg=cfg, profile=None, adducts=[AD], families=(),
                           curated=frozenset(), level_inputs=None)
    CP.apply_pool(led, pool, inputs, audit=[], log=lambda *a: None)
    return led.set_index("peak_id")


def test_the_lent_window_judges_a_carried_line_at_the_edge():
    plain, carried = _pooled(None), _pooled(REC)
    assert plain.at["u-0", "tier"] == "Candidate"            # 3 ppm off the lent centre, uncorroborated
    assert carried.at["u-0", "tier"] == "Assigned"           # on the carried centre
    for pid in ("u-1", "u-2", "u-3", "u-4", "u-5"):          # in scope: the same verdicts
        assert carried.at[pid, "tier"] == plain.at[pid, "tier"]
        assert carried.at[pid, "tier_reason"] == plain.at[pid, "tier_reason"]


def test_a_calibration_from_elsewhere_takes_the_files_carry():
    cal = T._Cal(0.1, 0.3, 0.2, -0.05, 0.25, 100.0, 300.0, abs_floor_mda=0.02, source=T.CAL_SOURCE_POOLED)
    got = T._with_carry(cal, REC)
    assert tuple(got) == (0.1, 0.3) and got.carry == REC
    assert (got.a, got.b, got.sigma_t, got.mz_lo, got.mz_hi) == (0.2, -0.05, 0.25, 100.0, 300.0)
    assert got.abs_floor_mda == 0.02 and T._cal_source(got) == T.CAL_SOURCE_POOLED
    # at a carried line the z is the plain z of its error less the carry
    assert T._cal_z(got, 3.0, 101.0) == pytest.approx(T._cal_z(cal, 0.0, 101.0))
    assert T._cal_z(got, 3.0, 150.0) == T._cal_z(cal, 3.0, 150.0)          # in scope: as before
