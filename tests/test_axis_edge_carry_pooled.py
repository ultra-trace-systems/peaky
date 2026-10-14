"""The edge carry on the batch-pooled tier calibration (assignment/calpool.py).

A file whose own core cannot calibrate the tier engine is re-tiered in a window the
batch's pooled core lends it. Under a sub-ppm batch axis model's edge carry
(PassConfig.axis_edge_carry; tests/test_axis_edge_carry.py) that lent window judges a
line just outside the model's scope at its centre plus the model's edge too, as the
file's own calibration would (tiers.compute_tiers attaches the carry to a calibration
from elsewhere; calpool.apply_pool lends it to the degeneracy re-run as well), and the
pool reads a row past the edge on the carried scale: its core (pool_calibration(carry=),
the batch's record) and the file's own centre it lends the window about.

Synthetic ledgers only.
"""

from __future__ import annotations

import pytest

from peaky.assignment import calpool as CP
from peaky.assignment import tiers as T
from peaky.assignment.passes import PassConfig
from peaky.batch import axislock as AL
from peaky.chem import chemistry as C
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


# --------------------------------------------------------------------------- the pooled core and the lent centre
#: a record carrying +0.6 ppm into every row above m/z 160 (its reach widened to the core's span here)
REC_UP = {"reach_da": 200.0, "slack_ppm": AL.SCOPE_SLACK_PPM, "below": {"mz": 50.0, "ppm": 0.0},
          "above": {"mz": 160.0, "ppm": 0.6}}


def _up(nf) -> float:
    """The measured axis under REC_UP: +0.6 ppm above m/z 160, as measured below."""
    return 0.6 if C.ion_mz(nf, AD) > 160.0 else 0.0


def test_the_pooled_core_reads_rows_past_the_edge_on_the_carried_scale():
    # every file's rows above the edge read +0.6 ppm as measured (9 of the 16 ions)
    base = lambda f, k: 0.05 * ((k % 5) - 2) + 0.01 * f
    batch = _batch(ppm=lambda f, k: base(f, k) + _up(NEUTRALS[k]))
    flat = CP.pool_calibration(_batch(ppm=base))
    plain, carried = CP.pool_calibration(batch), CP.pool_calibration(batch, carry=REC_UP)
    assert carried.cal[0] == pytest.approx(flat.cal[0], abs=1e-9)
    assert plain.cal[0] > flat.cal[0] + 0.3
    assert (carried.core["ppm"].abs() < 0.2).all()


def test_the_files_own_lent_centre_is_read_on_the_carried_scale():
    pool = CP.pool_calibration(_batch())
    at = _file("u", [(nf, AD, _up(nf), 1e5, False, "Good") for nf in NEUTRALS[:16]], tier=False)
    got = {}
    for name, carry in (("plain", None), ("carried", REC_UP)):
        led = at.copy(deep=True)
        cfg = PassConfig()
        cfg.axis_edge_carry = carry
        T.apply_tiers(led, cfg=cfg)
        inputs = CP.PoolInputs(at_tiers=at, cfg=cfg, profile=None, adducts=[AD], families=(),
                               curated=frozenset(), level_inputs=None)
        got[name] = CP.apply_pool(led, pool, inputs, audit=[], log=lambda *a: None)["cal"]
    assert got["carried"]["centre"] == got["plain"]["centre"] == "file"
    assert got["carried"]["mu"] == pytest.approx(0.0, abs=1e-9)
    assert got["plain"]["mu"] == pytest.approx(0.6, abs=1e-9)


def test_the_lent_windows_degeneracy_count_centres_a_carried_line_at_the_edge(monkeypatch):
    from peaky.assignment import degeneracy as DG
    real, seen = DG.enumerate_window, {}

    def spy(mz, lo, hi, chans, space, centre=None):
        seen[round(float(mz), 4)] = (lo, hi, centre)
        return real(mz, lo, hi, chans, space, centre=centre)
    monkeypatch.setattr(DG, "enumerate_window", spy)
    pool = CP.pool_calibration(_batch())
    rows = [(nf, AD, 0.0, 1e5, False, "Good") for nf in NEUTRALS[:6]]
    rows[0] = (NEUTRALS[0], AD, 3.0, 1e5, False, "Good")     # m/z 101: below the first lock, 3 ppm off
    at = _file("u", rows, tier=False)
    led = at.copy(deep=True)
    cfg = PassConfig()
    cfg.axis_edge_carry = REC
    T.apply_tiers(led, cfg=cfg)
    inputs = CP.PoolInputs(at_tiers=at, cfg=cfg, profile=None, adducts=[AD], families=(),
                           curated=frozenset(), level_inputs=None)
    res = CP.apply_pool(led, pool, inputs, audit=[], log=lambda *a: None)
    mu = res["cal"]["mu"]
    m0 = at.set_index("peak_id")["mz"]
    assert seen[round(float(m0["u-0"]), 4)][2] == pytest.approx(mu + 3.0)    # carried: at the edge
    assert seen[round(float(m0["u-1"]), 4)][2] == pytest.approx(mu)          # in scope: the lent centre


def test_a_calibration_handed_to_the_tier_engine_judges_a_carried_line_at_the_edge():
    rows = [(nf, AD, 0.0, 1e5, False, "Good") for nf in NEUTRALS[:6]]
    rows[0] = (NEUTRALS[0], AD, 3.0, 1e5, False, "Good")
    at = _file("u", rows, tier=False)
    cal = T._Cal(0.0, 0.3, source=T.CAL_SOURCE_POOLED)
    carried = PassConfig()
    carried.axis_edge_carry = REC
    got = T.compute_tiers(at, cfg=carried, cal=cal).set_index("peak_id")
    plain = T.compute_tiers(at, cfg=PassConfig(), cal=cal).set_index("peak_id")
    assert got.at["u-0", "tier"] == "Assigned"
    assert plain.at["u-0", "tier"] == "Candidate"


def test_the_batch_builds_its_pool_under_the_axis_models_carry():
    # the pool is built in the batch run (assign_batch.run's pooled step): the call hands it the
    # batch's record
    import inspect
    from peaky.batch import assign_batch as AB
    src = inspect.getsource(AB.run)
    call = src[src.index("_CP.pool_calibration("):]
    call = call[:call.index(")\n") + 1]
    assert 'carry=axis_info.get("edge_carry")' in call
