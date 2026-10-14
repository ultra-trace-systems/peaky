"""The scoring centre of a line just outside the batch's axis model: the model's edge, carried.

The lock model corrects peak m/z only inside its scope (its segments' domains, first lock
to last). A file's centres -- the local scorer's mu (the anchors' median, or zero on the
corrected axis) and calibrate()'s cal_mu -- are fitted on the mostly corrected peaks, so a
peak just outside the scope, still on the measured axis, was judged against a centre read
off the other axis: on an EasyIC batch whose sub-ppm axis the lock model newly corrects,
the C9-C12 hydrocarbons above the last lock (raw -0.7 ppm) fell out of Assigned.

A model the batch applies through the sub-ppm branch (`assign_batch._lock_model`: its
largest correction and the locks' raw median both under 1 ppm -- a batch the reference-ion
bar left uncorrected) CARRIES its correction at its first and its last lock into the
scoring centre of every line within EDGE_CARRY_DA below / above it. No m/z moves; in scope,
in the model's gaps and holes and beyond the reach nothing changes, and a model applied at
1 ppm or more carries nothing.

Pinned here:
  * masscal.edge_carry_at / carried_centre: below / above within reach -> the edge's value;
    in scope, a gap, beyond the reach, NaN, no record -> 0; the boundary on the served axis;
  * AxisModel.edge_carry_record / edge_carry, through as_dict / from_dict;
  * _lock_model grants it to a sub-ppm model only and records it;
  * io_mascope: the sample's record, the snapshot's keys, a stand-in inherits it;
  * the local scorer's per-line `centre_offset` (none or a zero one: the same frame);
  * cal_center / z_of / calibrate, the tier calibration and the degeneracy window carry it;
  * a run: the lines within reach are judged at the edge, the rest exactly as without it.

Offline: synthetic spectra, no network.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mascope_tools.composition import PatternScoring  # noqa: E402
from mascope_tools.composition.heuristic_filter import (  # noqa: E402
    anchor_on_monoisotopic,
    predict_isotopes,
)

from peaky.assignment import assign as A  # noqa: E402
from peaky.assignment import degeneracy as DG  # noqa: E402
from peaky.assignment import masscal as MC  # noqa: E402
from peaky.assignment import tiers as T  # noqa: E402
from peaky.assignment.passes import config as PCfg  # noqa: E402
from peaky.assignment.passes import core as PC  # noqa: E402
from peaky.batch import assign_batch as AB  # noqa: E402
from peaky.batch import axislock as AL  # noqa: E402
from peaky.batch import massqc as MQ  # noqa: E402
from peaky.batch.wave import WaveFit  # noqa: E402
from peaky.chem import chemistry as C  # noqa: E402
from peaky.io import io_mascope as IO  # noqa: E402
from peaky.io import local_scoring as LS  # noqa: E402

#: CHO acids [M-H]-, m/z 131-497 (C10 at 201.1, C11 215.1, C12 229.1, C13 243.1 ...)
ACIDS = [f"C{n}H{2 * n - 2}O4" for n in range(5, 25)] + [f"C{n}H{2 * n - 4}O6" for n in range(18, 30)]
SHIFT = -0.6        # the measured axis (ppm), flat: in scope the model removes it, outside it stays
LO, HI = 120.0, 200.0
#: CHO neutrals at odd [M-H]- nominal masses (no line on another's 13C): 11 in scope (m/z 131-191),
#: 8 just below the model's first lock (m/z 101-117), 13 just above its last (m/z 201-225), and two
#: in scope and two above too weak to show a 13C line. Lines past BOTH ends: read as measured, the
#: core's two ends then sit 0.6 ppm off its middle -- a step no 1/mz mass trend absorbs
RUN_STRONG_LOW = ["C4H6O3", "C4H8O3", "C7H8O", "C6H6O2", "C6H8O2", "C6H10O2", "C5H8O3", "C5H10O3"]
RUN_STRONG_IN = ["C5H8O4", "C8H10O2", "C7H12O3", "C5H10O5", "C8H12O3", "C6H10O5", "C10H16O2", "C8H14O4",
                 "C11H16O2", "C9H14O4", "C7H12O6"]
RUN_STRONG_OUT = ["C9H14O5", "C9H16O5", "C8H14O6", "C13H20O2", "C12H18O3", "C11H16O4", "C11H18O4", "C10H16O5",
                  "C10H18O5", "C9H16O6", "C14H22O2", "C13H20O3", "C12H18O4"]
RUN_WEAK_IN, RUN_WEAK_OUT = ["C7H10O2", "C11H18O3"], ["C11H16O5", "C11H18O5"]


def _quiet(*_a, **_k):
    pass


def _ion(nf: str) -> str:
    cnt = dict(C.parse_formula(nf))
    cnt["H"] -= 1                                            # [M-H]-
    return C.format_formula({k: v for k, v in cnt.items() if v > 0})


def _peaks(neutrals=ACIDS, shift=SHIFT) -> pd.DataFrame:
    """Every predicted line of each neutral's [M-H]- ion, `shift` ppm off; heights the
    predicted abundances (only the mass term moves a score)."""
    rows = []
    for k, nf in enumerate(neutrals):
        mzs, ints, _ = anchor_on_monoisotopic(*predict_isotopes(_ion(nf), -1))
        for i, (m, h) in enumerate(zip(mzs, ints)):
            height = h / ints[0] * 2e5
            if height < 300:
                continue
            rows.append({"peak_id": f"p{k}_{i}", "mz": m * (1 + shift * 1e-6), "height": height,
                         "area": height, "signal_to_noise": height / 30.0})
    return pd.DataFrame(rows).sort_values("mz").reset_index(drop=True)


def _rec(lo=LO, hi=HI, v_lo=SHIFT, v_hi=SHIFT, reach=30.0) -> dict:
    """An edge-carry record (AxisModel.edge_carry_record's shape)."""
    return {"reach_da": reach, "slack_ppm": AL.SCOPE_SLACK_PPM,
            "below": {"mz": lo, "ppm": v_lo}, "above": {"mz": hi, "ppm": v_hi}}


def _model(carry: bool = True, y=SHIFT) -> AL.AxisModel:
    """A flat lock model over m/z 120-200 (the axis reads `y` there), carried or not."""
    m = AL.AxisModel(segments=[{"kind": "spline", "lo": LO, "hi": HI, "n": 12,
                                "x": np.array([LO, HI]), "y": np.array([y, y])}], n_locks=12)
    if carry:
        m.stats["edge_carry"] = _rec(v_lo=y, v_hi=y)
    return m


def _run(sid, peaks=None, **cfg_kw):
    cfg = PCfg.PassConfig(height_cutoff_cps=100, **cfg_kw)
    return A.run(sid, context="ambient-air", cfg=cfg, peaks=_peaks() if peaks is None else peaks,
                 adducts=["[M-H]-"], do_pass2=False, do_pass3=False, do_pass4=False, do_pass5=False,
                 do_pass_certified=False, use_cache=False, log=_quiet, scoring="orbi")


def _m0(res, lo=0.0, hi=1e9) -> pd.DataFrame:
    led = res["ledger"]
    return led[(led["role"] == "M0") & (led["mz"] > lo) & (led["mz"] <= hi)]


@pytest.fixture(autouse=True)
def _forget():
    yield
    IO.clear_axis_correction()


# --------------------------------------------------------------------------- the carry itself
class TestTheCarry:
    def test_within_reach_below_and_above_the_edge_value_else_nothing(self):
        rec = _rec(v_lo=-0.2, v_hi=-0.7)
        mz = [85.0, 95.0, 119.9, 150.0, 200.1, 229.9, 230.5, np.nan]
        got = MC.edge_carry_at(rec, mz)
        assert got.tolist() == [0.0, -0.2, -0.2, 0.0, -0.7, -0.7, 0.0, 0.0]

    def test_no_record_or_no_mz_carries_nothing(self):
        assert MC.edge_carry_at(None, [150.0, 210.0]).tolist() == [0.0, 0.0]
        assert float(MC.edge_carry_at(_rec(), 210.0)) == SHIFT
        assert float(MC.edge_carry_at(_rec(), None)) == 0.0
        c = 0.25
        assert MC.carried_centre(None, c, 210.0) is c
        assert MC.carried_centre(_rec(), c, 150.0) is c          # in scope: the centre itself
        assert MC.carried_centre(_rec(), c, None) is c
        assert MC.carried_centre(_rec(), c, 210.0) == pytest.approx(c + SHIFT)
        assert np.allclose(MC.carried_centre(_rec(), c, np.array([150.0, 210.0])), [c, c + SHIFT])

    def test_the_boundary_is_read_on_the_served_axis(self):
        # a peak measured just inside the scope's slack is corrected (moved 0.6 ppm down
        # here): not carried; one measured past it stays where it was: carried. The record
        # is the model's own, with the domains' slack: a record without it would carry a
        # peak the model corrects (inside the slack) a second time
        model = _model(carry=False, y=+0.6)
        rec = model.edge_carry_record(30.0)
        assert rec["slack_ppm"] == AL.SCOPE_SLACK_PPM == 3.0
        assert (rec["below"]["ppm"], rec["above"]["ppm"]) == pytest.approx((0.6, 0.6))
        raw = np.array([LO * (1 - 2.9e-6), LO * (1 - 5e-6), HI * (1 + 2.9e-6), HI * (1 + 3.1e-6)])
        served = MQ.apply_correction(model, raw)
        assert MQ.in_scope(model, raw).tolist() == [True, False, True, False]
        assert MC.edge_carry_at(rec, served).tolist() == [0.0, 0.6, 0.0, 0.6]

    @pytest.mark.parametrize("y", [+0.6, -0.6])
    def test_outside_is_what_the_models_own_inverse_reads(self, y):
        # every served m/z near either edge: carried exactly where the model's inverse
        # (publish's reading of the served axis) finds no segment whose correction
        # produced it, i.e. hands it back as measured
        model, rec = _model(carry=False, y=y), _rec(v_lo=y, v_hi=y)
        served = np.concatenate([LO * (1 + np.linspace(-8e-6, 2e-6, 401)),
                                 HI * (1 + np.linspace(-2e-6, 8e-6, 401))])
        outside = MQ.invert_correction(model, served) == served
        assert outside.any() and (~outside).any()
        assert np.array_equal(MC.edge_carry_at(rec, served) != 0, outside)
        # a measured peak is carried exactly when it lies outside the scope, but within
        # |y| ppm of the slack's edge, where an in-scope peak's corrected m/z lands too
        raw = served
        got = MC.edge_carry_at(rec, MQ.apply_correction(model, raw)) != 0
        band = np.minimum(np.abs(raw / (LO * (1 - 3e-6)) - 1), np.abs(raw / (HI * (1 + 3e-6)) - 1)) <= 0.61e-6
        assert np.array_equal(got[~band], ~MQ.in_scope(model, raw)[~band])

    def test_gaps_and_holes_between_segments_carry_nothing(self):
        m = AL.AxisModel(segments=[
            {"kind": "spline", "lo": 120.0, "hi": 200.0, "n": 9, "x": np.array([120.0, 200.0]),
             "y": np.array([0.2, -0.4]), "holes": [[150.0, 160.0]]},
            {"kind": "line", "lo": 260.0, "hi": 300.0, "n": 6, "a": -0.5, "b": 0.0}])
        rec = m.edge_carry_record(30.0)
        assert rec["below"] == {"mz": 120.0, "ppm": pytest.approx(0.2), "model_ppm": pytest.approx(0.2),
                                "n_locks": 0}
        assert rec["above"] == {"mz": 300.0, "ppm": pytest.approx(-0.5), "model_ppm": pytest.approx(-0.5),
                                "n_locks": 0}
        got = MC.edge_carry_at(rec, [100.0, 155.0, 210.0, 250.0, 320.0])
        assert got.tolist() == pytest.approx([0.2, 0.0, 0.0, 0.0, -0.5])

    def test_the_record_travels_with_the_model(self):
        m = _model()
        assert m.edge_carry == _rec()
        back = MQ.fit_from_record(m.as_dict())
        assert back.edge_carry == _rec()
        assert _model(carry=False).edge_carry is None
        assert MQ.fit_from_record(_model(carry=False).as_dict()).edge_carry is None


# --------------------------------------------------------------------------- the edge value
def _locks(mz, resid, model, bright=()):
    """A lock table (axislock's lock rows: mz / ppm / bright) whose readings sit `resid` off `model`."""
    mz = np.asarray(mz, dtype=float)
    ppm = model.predict(mz, extrapolate=True) + np.asarray(resid, dtype=float)
    return pd.DataFrame({"mz": mz, "ion": "x", "ppm": ppm, "how": "unique +-5 ppm",
                         "bright": [i in bright for i in range(len(mz))]})


class TestTheValueAtEachEnd:
    """Each end's value is the model there put back on its own end locks: the model at the
    edge plus the median residual of the end segment's non-bright locks within EDGE_LOCK_DA
    (5 Da) of it, at least EDGE_LOCKS_MIN (3) of them -- a spline end knot can leave its locks
    (a uronium axis read -0.41 ppm at its first lock, the locks there -0.15..-0.25)."""

    def _spline(self):
        # a sloped spline over 120-200: -0.40 at the first lock, -0.20 at the last
        return AL.AxisModel(segments=[{"kind": "spline", "lo": LO, "hi": HI, "n": 20,
                                       "x": np.array([LO, HI]), "y": np.array([-0.40, -0.20])}], n_locks=20)

    def test_the_value_is_the_model_plus_the_median_residual_of_the_locks_near_the_edge(self):
        model = self._spline()
        # within 5 Da of the first lock: five locks +0.2 off the model (one +3: the median holds);
        # further in they read -0.5 off and are not read. At the last: three -0.1 off, one +2 further in
        mz = [120.0, 121.0, 122.5, 124.0, 124.9, 130.0, 140.0, 150.0, 190.0, 196.0, 198.0, 200.0]
        resid = [0.2, 3.0, 0.2, 0.2, 0.2, -0.5, -0.5, -0.5, 2.0, -0.1, -0.1, -0.1]
        rec = model.edge_carry_record(30.0, locks=_locks(mz, resid, model))
        assert rec["below"] == {"mz": LO, "ppm": pytest.approx(-0.40 + 0.2), "model_ppm": pytest.approx(-0.40),
                                "n_locks": 5}
        assert rec["above"] == {"mz": HI, "ppm": pytest.approx(-0.20 - 0.1), "model_ppm": pytest.approx(-0.20),
                                "n_locks": 3}
        assert (AL.EDGE_LOCK_DA, AL.EDGE_LOCKS_MIN) == (5.0, 3)
        # the carry is that value
        assert float(MC.edge_carry_at(rec, 100.0)) == pytest.approx(-0.2)
        assert float(MC.edge_carry_at(rec, 210.0)) == pytest.approx(-0.3)

    def test_bright_locks_and_other_segments_locks_are_not_read(self):
        two = AL.AxisModel(segments=[
            {"kind": "spline", "lo": LO, "hi": HI, "n": 9, "x": np.array([LO, HI]), "y": np.array([-0.4, -0.2])},
            {"kind": "line", "lo": 202.0, "hi": 205.0, "n": 3, "a": -0.5, "b": 0.0}])
        # the first lock is bright (+5); three more within 5 Da read +0.1 off. The last segment's
        # three locks read -0.3 off; the first segment's top locks, within 5 Da of the last lock
        # too, read +1 off and belong to the other segment
        mz = [120.0, 121.0, 122.0, 123.0, 198.0, 199.0, 200.0, 202.0, 203.5, 205.0]
        resid = [5.0, 0.1, 0.1, 0.1, 1.0, 1.0, 1.0, -0.3, -0.3, -0.3]
        rec = two.edge_carry_record(30.0, locks=_locks(mz, resid, two, bright={0}))
        assert rec["below"]["ppm"] == pytest.approx(-0.4 + 0.1) and rec["below"]["n_locks"] == 3
        assert rec["above"]["ppm"] == pytest.approx(-0.5 - 0.3) and rec["above"]["n_locks"] == 3

    @pytest.mark.parametrize("locks", [None, "empty", "two near, many far"])
    def test_with_fewer_than_three_locks_near_the_edge_the_models_own_value_stands(self, locks):
        model = self._spline()
        table = {None: None, "empty": pd.DataFrame(columns=["mz", "ppm", "bright"]),
                 "two near, many far": _locks([LO, LO + 1.0, 130.0, 140.0, 150.0, 160.0], [0.3] * 6, model)}[locks]
        rec = model.edge_carry_record(30.0, locks=table)
        assert rec["below"]["ppm"] == pytest.approx(-0.40) and rec["below"]["n_locks"] == 0

    def test_the_batch_reads_the_value_off_the_fits_lock_table(self, monkeypatch):
        model = self._spline()
        mz = np.r_[np.linspace(LO, LO + 4.0, 5), np.linspace(140.0, 180.0, 9), np.linspace(HI - 4.0, HI, 5)]
        table = _locks(mz, np.where(mz < 160.0, 0.2, -0.1), model).to_dict("records")
        info = {"n_pass1": 19, "n_links": 0, "n_narrow": 19, "n_off_curve": 0, "locks": table}
        monkeypatch.setattr(AL, "fit", lambda *a, **k: (AL.AxisModel.from_dict(model.as_dict()),
                                                        dict(info, cv_ppm=0.06, raw_ppm=0.3, max_abs_ppm=0.4)))
        logs = []
        _, got, _ = AB.measure_axis(_tiny_ts(), "NO3", "orbitrap", log=logs.append)
        rec = got["edge_carry"]
        assert rec["below"]["ppm"] == pytest.approx(-0.2) and rec["below"]["model_ppm"] == pytest.approx(-0.4)
        assert rec["above"]["ppm"] == pytest.approx(-0.3) and rec["below"]["n_locks"] == 5
        assert any("(the model -0.40, 5 locks)" in str(m) for m in logs)


# --------------------------------------------------------------------------- the grant
def _tiny_ts():
    return pd.DataFrame({"sample_item_id": ["s0"] * 4 + ["s1"] * 4,
                         "mz": [100.0, 150.0, 210.0, 260.0] * 2, "height": [1e4] * 8})


@pytest.mark.parametrize("cv,raw,peak,carried", [
    (0.06, 0.3, 0.6, True),          # a sub-ppm model the card's bar applies: carried
    (0.1, 0.5, 0.9, True),           # still under 1 ppm
    (0.1, 0.5, 1.2, False),          # a 1.2 ppm model (applied at the reference bar too): never
    (0.1, 0.5, 1.0, False),          # exactly 1 ppm: the reference bar applies it, so never
    (0.1, 1.1, 0.9, False),          # the locks' raw median at 1 ppm or more: never
    (0.1, 1.0, 0.9, False),          # ... exactly 1 ppm too
    (0.1, 0.2, 0.35, False),         # not applied at all
])
def test_the_carry_is_granted_to_a_sub_ppm_model_only(monkeypatch, cv, raw, peak, carried):
    model = _model(carry=False)
    info = {"n_pass1": 12, "n_links": 0, "n_narrow": 12, "n_off_curve": 0}
    monkeypatch.setattr(AL, "fit", lambda *a, **k: (AL.AxisModel.from_dict(model.as_dict()),
                                                    dict(info, cv_ppm=cv, raw_ppm=raw, max_abs_ppm=peak)))
    logs = []
    _, got, _ = AB.measure_axis(_tiny_ts(), "NO3", "orbitrap", log=logs.append)
    rec = (got.get("wave") or {}).get("stats", {}).get("edge_carry")
    if carried:
        assert got["applied"]
        assert rec == got["edge_carry"] and rec["reach_da"] == AB.EDGE_CARRY_DA
        assert rec["below"]["mz"] == LO and rec["above"]["mz"] == HI
        assert any("[mass-axis] edge carry" in str(m) for m in logs)
        assert MQ.fit_from_record(got["wave"]).edge_carry == rec
    else:
        assert rec is None and "edge_carry" not in got


def test_the_reach_is_30_da():
    # the design's measured plateau (the edge value is the axis to ~0.1 ppm within 30 Da on
    # the flat axes; a curved labelled-nitrate axis bends away past it); 40 Da is open
    assert AB.EDGE_CARRY_DA == 30.0


# --------------------------------------------------------------------------- the sample
class TestTheSample:
    SID = "ec-sample"

    def setup_method(self):
        IO.register_offline_sample(self.SID, _peaks(), ["[M-H]-"], scoring="orbi")

    def teardown_method(self):
        IO.unregister_offline_sample(self.SID)

    def test_the_sample_reads_its_models_record_and_the_snapshot_says_so(self):
        IO.set_axis_correction([self.SID], _model())
        assert IO.axis_edge_carry(self.SID) == _rec()
        snap = IO.scoring_snapshot(None, self.SID)
        carry = snap["axis_edge_carry"]
        assert {k: v for k, v in carry.items() if k not in ("share", "anchors")} == _rec()
        assert 0 < carry["share"] < 1 and snap["mu_source"] == "axis" and carry["anchors"] == 0
        IO.clear_axis_correction([self.SID])
        assert IO.axis_edge_carry(self.SID) is None
        assert "axis_edge_carry" not in IO.scoring_snapshot(None, self.SID)

    @pytest.mark.parametrize("fit", [_model(carry=False),
                                     WaveFit(coef=[SHIFT, 0.0], p=-0.5, domain=(HI ** -0.5, LO ** -0.5),
                                             mz_range=(LO, HI), K=1, n=10, n_clipped=0, resid_ppm=0.1,
                                             raw_ppm=0.1, loo_ppm=0.1, span_ppm=0.0, share=0.9)])
    def test_a_model_without_a_record_carries_nothing(self, fit):
        IO.set_axis_correction([self.SID], fit)
        assert IO.axis_edge_carry(self.SID) is None
        assert "axis_edge_carry" not in IO.scoring_snapshot(None, self.SID)

    def test_anchors_past_the_edge_are_read_on_the_carried_scale(self):
        # a file whose server anchors all sit just above the model's last lock (on the measured
        # axis, -0.6 ppm): its fitted centre is read on the carried scale (0), so the lines
        # there are judged at 0 + the carry, and the in-scope ones at 0 -- not at -0.6 + the
        # carry (the carry counted twice) and -0.6
        P = _peaks(RUN_STRONG_IN + RUN_STRONG_OUT)
        mech = IO.ADDUCT_TO_MECH["[M-H]-"]
        P["target_compound_formula"] = None
        P["ionization_mechanism"] = None
        for nf in RUN_STRONG_OUT[:6]:
            k = (P["mz"] - C.ion_mz(nf, "[M-H]-")).abs().idxmin()
            P.loc[k, ["target_compound_formula", "ionization_mechanism"]] = [nf, mech]
        got = {}
        for carry in (True, False):
            sid = f"ec-anchors-{carry}"
            IO.register_offline_sample(sid, P, ["[M-H]-"], scoring="orbi")
            IO.set_axis_correction([sid], _model(carry=carry))
            try:
                snap = IO.scoring_snapshot(None, sid)
                o = IO.score_candidates(None, sid, [RUN_STRONG_IN[0]] + RUN_STRONG_OUT[:3],
                                        mechanism_ids=["[M-H]-"])
                got[carry] = (snap, o[o["iso_label"] == "M0"].set_index("compound_formula")["ion_score"])
            finally:
                IO.clear_axis_correction([sid])
                IO.unregister_offline_sample(sid)
        snap, score = got[True]
        assert snap["mu_source"] == "fitted" and snap["fitted_anchors"] == 6
        assert snap["mu_ppm"] == pytest.approx(0.0, abs=1e-6) and snap["axis_edge_carry"]["anchors"] == 6
        assert (score >= 0.95).all()
        plain_snap, plain = got[False]
        assert plain_snap["mu_ppm"] == pytest.approx(SHIFT)            # without a record: as measured
        assert plain[RUN_STRONG_IN[0]] < 0.8
        assert IO.sample_mass_errors(P, max_abs_ppm=5.0) == pytest.approx(
            IO.sample_mass_errors(P, max_abs_ppm=5.0, carry=None))

    def test_a_stand_in_inherits_the_carry_with_the_snapshot(self):
        IO.set_axis_correction([self.SID], _model())
        snap = IO.scoring_snapshot(None, self.SID)
        IO.register_offline_sample("ec-standin", _peaks(), ["[M-H]-"], scoring=snap)
        try:
            assert IO.axis_edge_carry("ec-standin") == _rec()
            assert IO.scoring_snapshot(None, "ec-standin")["axis_edge_carry"]["reach_da"] == 30.0
        finally:
            IO.unregister_offline_sample("ec-standin")


# --------------------------------------------------------------------------- the local scorer
class TestTheLocalScorer:
    SC = PatternScoring(sigma_ppm=0.58, mu_ppm=0.0, mz_tolerance_ppm=5.0)

    def _score(self, peaks, **kw):
        out = LS.score_candidates_local(peaks, ["C11H20O4"], adducts=["[M-H]-"], scoring=kw.pop("scoring", self.SC),
                                        **kw)
        return out[out["iso_label"] == "M0"]

    def test_each_matched_line_is_judged_at_its_own_carried_centre(self):
        peaks = _peaks(["C11H20O4"])
        at_mu = self._score(peaks, scoring=PatternScoring(sigma_ppm=0.58, mu_ppm=SHIFT, mz_tolerance_ppm=5.0))
        carried = self._score(peaks, centre_offset=lambda m: np.full(np.shape(m), SHIFT))
        plain = self._score(peaks)
        assert carried["ion_score"].iloc[0] == pytest.approx(at_mu["ion_score"].iloc[0])
        assert carried["ion_score"].iloc[0] > 0.9 > plain["ion_score"].iloc[0]
        # per line: an offset on the M0 only judges its 13C line at the plain centre
        m0 = float(peaks["mz"].min())
        part = self._score(peaks, centre_offset=lambda m: np.where(np.asarray(m) < m0 + 0.5, SHIFT, 0.0))
        assert plain["ion_score"].iloc[0] < part["ion_score"].iloc[0] < carried["ion_score"].iloc[0]
        assert part["ion_score_massfree"].iloc[0] == plain["ion_score_massfree"].iloc[0]

    def test_the_carry_rides_on_a_mass_trend_too(self):
        # a file re-scored at its C42 trend: a line past the edge is judged at the trend's
        # centre at its m/z plus the carry
        peaks = _peaks(["C11H20O4"], shift=SHIFT + 0.1)          # the trend reads +0.1 there
        trend = MC.MassTrend(a=0.1, b=0.0, sigma=0.3, n=30, mz_lo=100.0, mz_hi=400.0)
        carried = self._score(peaks, centre=trend, centre_offset=lambda m: np.full(np.shape(m), SHIFT))
        plain = self._score(peaks, centre=trend)
        at_mu = self._score(peaks, scoring=PatternScoring(sigma_ppm=0.58, mu_ppm=SHIFT + 0.1,
                                                          mz_tolerance_ppm=5.0))
        assert carried["ion_score"].iloc[0] == pytest.approx(at_mu["ion_score"].iloc[0])
        assert carried["ion_score"].iloc[0] > 0.9 > plain["ion_score"].iloc[0]

    def test_no_offset_or_a_zero_one_gives_the_same_frame(self):
        peaks = _peaks(ACIDS[:6])
        base = LS.score_candidates_local(peaks, ACIDS[:6], adducts=["[M-H]-"], scoring=self.SC)
        for off in (None, lambda m: np.zeros(np.shape(m))):
            got = LS.score_candidates_local(peaks, ACIDS[:6], adducts=["[M-H]-"], scoring=self.SC,
                                            centre_offset=off)
            pd.testing.assert_frame_equal(got, base)


# --------------------------------------------------------------------------- the mass gates
def _ledger(n_in=10, n_out=12, out_ppm=SHIFT) -> pd.DataFrame:
    """A backbone of isotope-backed CHO M0 rows: `n_in` in scope at 0 ppm, `n_out` above the
    model's last lock (within reach) at the measured `out_ppm`; one 13C child each."""
    mz = [130.0 + 6.0 * k if k < n_in else 201.0 + 2.0 * (k - n_in) for k in range(n_in + n_out)]
    ppm = [(0.02 if k % 2 else -0.02) + (0.0 if k < n_in else out_ppm) for k in range(n_in + n_out)]
    return _backbone(mz, ppm)


def _backbone(mzs, ppms) -> pd.DataFrame:
    """Isotope-backed CHO M0 rows at `mzs` reading `ppms`; one 13C child each."""
    rows = []
    for k, (mz, ppm) in enumerate(zip(mzs, ppms)):
        mz, ppm = float(mz), float(ppm)
        rows.append({"peak_id": f"m{k}", "role": "M0", "mz": mz, "ppm_error": ppm, "ion_score": 0.95,
                     "compound_score": 0.95, "ion_score_massfree": 0.97, "neutral_formula": "C8H14O4",
                     "adduct": "[M-H]-", "confidence": "High", "isotopologues": "[13C]", "parent_peak_id": None,
                     "iso_label": None, "height": 1e5, "locked": False, "tied": False})
        rows.append({"peak_id": f"m{k}c", "role": "iso_child", "mz": mz + 1.00336, "ppm_error": ppm,
                     "ion_score": None, "compound_score": None, "ion_score_massfree": None,
                     "neutral_formula": None, "adduct": None, "confidence": None, "isotopologues": None,
                     "parent_peak_id": f"m{k}", "iso_label": "13C", "height": 1e4, "locked": False,
                     "tied": False})
    return pd.DataFrame(rows)


class TestTheCalibratedGates:
    def _cfg(self, carry=True, **kw):
        cfg = PCfg.PassConfig(**kw)
        cfg.axis_edge_carry = _rec() if carry else None
        return cfg

    def test_cal_center_and_z_of_carry_the_edge_within_reach(self):
        cfg = self._cfg()
        cfg.cal_mu, cfg.cal_sigma = 0.05, 0.3
        assert PC.cal_center(cfg, 150.0) == 0.05 and PC.cal_center(cfg, None) == 0.05
        assert PC.cal_center(cfg, 210.0) == pytest.approx(0.05 + SHIFT)
        assert PC.cal_center(cfg, 250.0) == 0.05                       # beyond the reach
        assert PC.z_of(0.05 + SHIFT, cfg, 210.0) == pytest.approx(0.0)
        assert PC.z_of(0.05 + SHIFT, cfg, 250.0) == pytest.approx(2.0)
        # on top of a mass trend too
        cfg.cal_a, cfg.cal_b, cfg.cal_mz_lo, cfg.cal_mz_hi = 0.1, -0.02, 100.0, 400.0
        assert PC.cal_center(cfg, 210.0) == pytest.approx(MC.centre(0.1, -0.02, 210.0, 100.0, 400.0) + SHIFT)

    def test_no_record_leaves_every_centre_as_it_was(self):
        cfg = self._cfg(carry=False)
        cfg.cal_mu, cfg.cal_sigma = 0.05, 0.3
        assert PC.cal_center(cfg, 210.0) == 0.05
        assert PCfg.PassConfig().axis_edge_carry is None
        assert "axis_edge_carry" in PCfg.PassConfig.RUNTIME_FIELDS

    def test_calibrate_fits_the_backbone_on_the_carried_scale(self):
        led = _ledger()
        plain, carried = self._cfg(carry=False, cal_min_n=20), self._cfg(cal_min_n=20)
        PC.calibrate(led, plain, log=_quiet)
        PC.calibrate(led, carried, log=_quiet)
        assert plain.cal_mu == pytest.approx(SHIFT, abs=0.03)          # the rows above the edge outvote
        assert carried.cal_mu == pytest.approx(0.0, abs=0.03) and carried.cal_sigma == 0.25

    def test_the_tier_calibration_and_its_z_carry_the_edge(self):
        led = _ledger()
        m0 = led[led["role"] == "M0"]
        kids = led.loc[led["role"] == "iso_child", "parent_peak_id"].value_counts()
        plain = T._calibrate(m0, kids)
        cal = T._calibrate(m0, kids, carry=_rec())
        assert plain[0] == pytest.approx(SHIFT, abs=0.03) and cal[0] == pytest.approx(0.0, abs=0.03)
        assert T._cal_z(cal, SHIFT, 210.0) == pytest.approx(T._cal_z(cal, 0.0, 150.0), abs=1e-9)
        assert T._cal_z(cal, SHIFT, None) == pytest.approx((SHIFT - cal[0]) / cal[1])
        T.stamp_calibrated_ppm(led, carry=_rec())
        out = led[led["mz"] > 200.5]
        assert np.allclose(out["ppm_error_cal"], out["ppm_error"] - cal[0] - SHIFT)
        before = _ledger()
        T.stamp_calibrated_ppm(before)
        assert np.allclose(before["ppm_error_cal"], before["ppm_error"] - plain[0])

    def test_calibrate_fits_the_mass_trend_on_the_carried_scale(self):
        # the corrected backbone follows ppm = a + b*1000/mz (b = -0.1 mDa); the isotope-tested
        # rows just above the model's last lock read that trend plus the axis there (+0.3)
        a0, b0 = 0.05, -0.1
        mz_in = np.r_[np.linspace(60.0, 95.0, 10), np.linspace(100.0, 199.0, 14)]
        mz_out = np.linspace(201.0, 229.0, 12)
        jitter = np.resize([0.02, -0.02, 0.0], len(mz_in) + len(mz_out))
        mz = np.r_[mz_in, mz_out]
        led = _backbone(mz, a0 + b0 * 1000.0 / mz + np.where(mz > HI, 0.3, 0.0) + jitter)
        carried, plain = self._cfg(carry=False, cal_min_n=20), self._cfg(carry=False, cal_min_n=20)
        carried.axis_edge_carry = _rec(v_lo=0.3, v_hi=0.3)
        PC.calibrate(led, carried, log=_quiet)
        PC.calibrate(led, plain, log=_quiet)
        assert (carried.cal_a, carried.cal_b) == pytest.approx((a0, b0), abs=0.01)
        assert carried.cal_mz_hi == pytest.approx(229.0)
        # read as measured, the rows past the edge pull the trend off the corrected axis
        assert plain.cal_b is None or abs(plain.cal_a - a0) > 0.05
        # the gate: a line past the edge is judged at the trend there plus the carry
        assert PC.z_of(a0 + b0 * 1000.0 / 215.0 + 0.3, carried, 215.0) == pytest.approx(0.0, abs=0.1)

    def test_an_uncalibrated_files_stamp_centre_is_read_on_the_carried_scale(self):
        # a core too small to calibrate: the displayed errors are centred on the Assigned rows'
        # median -- read on the carried scale, so the rows past the edge do not outvote it
        led = _ledger(n_in=3, n_out=6)
        led["tier"] = pd.Series(np.where(led["role"] == "M0", T.TIER_ASSIGNED, None), dtype=object)
        plain, carried = led.copy(), led.copy()
        assert T._calibrate(led[led["role"] == "M0"], pd.Series(dtype=int)) is None
        mu_p, _ = T.stamp_calibrated_ppm(plain)
        mu_c, _ = T.stamp_calibrated_ppm(carried, carry=_rec())
        assert mu_p == pytest.approx(SHIFT, abs=0.03) and mu_c == pytest.approx(0.0, abs=0.03)
        out = carried[(carried["role"] == "M0") & (carried["mz"] > HI)]
        assert np.allclose(out["ppm_error_cal"], out["ppm_error"] - mu_c - SHIFT)
        assert np.allclose(out["ppm_error_cal"], 0.0, atol=0.05)

    def test_the_tier_engines_own_calibration_reads_the_carried_scale(self):
        # compute_tiers calibrating on its own (no calibration handed in): 10 isotope-backed rows
        # in scope at 0 ppm, 14 just past either edge at the measured -0.6; an uncorroborated row
        # in scope (V, 0 ppm) and one past the edge (U, -0.6). On the carried scale the core's
        # centre is 0 and neither is off it; read as measured the rows past the edges outvote the rest
        led = _committed([(f"i{k}", 130.0 + 6.0 * k, (0.02 if k % 2 else -0.02), RUN_STRONG_IN[k], True)
                          for k in range(10)]
                         + [(f"l{k}", 95.0 + 4.0 * k, SHIFT + (0.02 if k % 2 else -0.02), RUN_STRONG_LOW[k], True)
                            for k in range(6)]
                         + [(f"o{k}", 201.0 + 2.0 * k, SHIFT + (0.02 if k % 2 else -0.02), RUN_STRONG_OUT[k], True)
                            for k in range(8)]
                         + [("V", 150.5, 0.0, "C7H10O2", False), ("U", 225.5, SHIFT, "C11H18O5", False)])
        carried, plain = self._cfg(), self._cfg(carry=False)
        got = T.compute_tiers(led, cfg=carried).set_index("peak_id")
        assert got.at["V", "tier"] == got.at["U", "tier"] == T.TIER_ASSIGNED
        was = T.compute_tiers(led, cfg=plain).set_index("peak_id")
        assert was.at["V", "tier"] == T.TIER_CANDIDATE and "instrument accuracy" in was.at["V", "tier_reason"]

    def test_the_degeneracy_alternatives_are_ranked_about_the_carried_centre(self, monkeypatch):
        # of two rivals the one nearer the row's own centre is named first: mu (0.1) in scope,
        # mu + the carry (-0.5) past the edge
        def window(mz, lo, hi, chans, space, centre=None):
            return {"r-near-mu": ("C9H18O3", "[M-H]-", 0.15), "r-near-carried": ("C7H10O5", "[M-H]-", -0.45)}
        monkeypatch.setattr(DG, "enumerate_window", window)
        monkeypatch.setattr(DG, "enumeration_space", lambda *a, **k: type(
            "S", (), {"profiles": [type("P", (), {"label": "x"})()], "curated": ()})())
        monkeypatch.setattr(DG, "_caps", lambda p: {e: 0 for e in DG.ELEMENT_CEILING})
        monkeypatch.setattr(DG, "_space_reason", lambda *a, **k: None)
        led = _ledger(n_in=1, n_out=1)
        got = DG.measure_degeneracy(led, cal=T._Cal(0.1, 0.3, carry=_rec()), adducts=["[M-H]-"])
        assert got["m0"]["alts"][0].startswith("C9H18O3")
        assert got["m1"]["alts"][0].startswith("C7H10O5")

    def test_the_degeneracy_window_follows_the_carried_centre(self, monkeypatch):
        seen = []

        def spy(mz, lo, hi, chans, space, centre=None):
            seen.append((round(mz, 1), lo, hi, centre))
            return {}
        monkeypatch.setattr(DG, "enumerate_window", spy)
        monkeypatch.setattr(DG, "enumeration_space", lambda *a, **k: type(
            "S", (), {"profiles": [type("P", (), {"label": "x"})()], "curated": ()})())
        monkeypatch.setattr(DG, "_caps", lambda p: {e: 0 for e in DG.ELEMENT_CEILING})
        monkeypatch.setattr(DG, "_space_reason", lambda *a, **k: None)
        led = _ledger(n_in=1, n_out=1)
        DG.measure_degeneracy(led, cal=T._Cal(0.1, 0.3, carry=_rec()), adducts=["[M-H]-"])
        (m_in, lo_in, hi_in, c_in), (m_out, lo_out, hi_out, c_out) = sorted(seen)
        assert c_in == pytest.approx(0.1) and c_out == pytest.approx(0.1 + SHIFT)
        assert (lo_out - lo_in, hi_out - hi_in) == pytest.approx((SHIFT, SHIFT))


def _committed(rows) -> pd.DataFrame:
    """A ledger of committed [M-H]- M0 rows (peak_id, mz, ppm, neutral, isotope-backed): High, one 13C
    child each when isotope-backed."""
    from peaky.assignment import ledger as L
    peaks = []
    for pid, mz, _p, _n, iso in rows:
        peaks.append((pid, mz, 1e5))
        if iso:
            peaks.append((pid + "c", mz + 1.00336, 1e4))
    led = L.new_ledger(pd.DataFrame(peaks, columns=["peak_id", "mz", "height"]))
    for pid, mz, ppm, nf, iso in rows:
        L.commit_assignment(led, pid, neutral_formula=nf, adduct="[M-H]-", ion_formula=_ion(nf) + "-",
                            ion_score=0.95, compound_score=0.95, eff_score=0.93, eff_margin=0.2, tied=False,
                            ppm_error=ppm, pass_no=1, method="cheminfo+grid", confidence="High" if iso else "Good",
                            commentary="backbone", isotopologues=[{"label": "13C", "peak_id": pid + "c"}] if iso else [])
        if iso:
            L.attach_isotopologue(led, pid + "c", pid, iso_label="13C", iso_match_score=0.9)
    return led


def _rearb_ledger(win_ppm=-1.0, alt_ppm=0.15):
    """20 isotope-backed CHO backbone rows near 0 ppm (m/z 150-169) and one aromatic monster
    M0 at m/z 464.143, `win_ppm`, with a plausible lower-DBE alternative at `alt_ppm`."""
    from peaky.assignment import ledger as L
    rows = []
    for i in range(20):
        rows += [(f"bb{i}", 150.0 + i, 1e5), (f"bb{i}c", 151.0 + i, 1e3)]
    rows.append(("P", 464.143, 5e3))
    led = L.new_ledger(pd.DataFrame(rows, columns=["peak_id", "mz", "height"]))
    for i in range(20):
        nf = f"C{8 + i}H{14 + i}O4"
        L.commit_assignment(led, f"bb{i}", neutral_formula=nf, adduct="[M-H]-", ion_formula=nf, ion_score=0.95,
                            compound_score=0.95, eff_score=0.93, eff_margin=0.2, tied=False,
                            ppm_error=[-0.10, 0.0, 0.10, 0.05, -0.05][i % 5], pass_no=1, method="cheminfo+grid",
                            confidence="High", commentary="backbone",
                            isotopologues=[{"label": "13C", "peak_id": f"bb{i}c"}])
        L.attach_isotopologue(led, f"bb{i}c", f"bb{i}", iso_label="13C", iso_match_score=0.9)
    L.commit_assignment(led, "P", neutral_formula="C36H17N", adduct="[M+H]+", ion_formula="C36H18N+",
                        ion_score=0.88, compound_score=0.88, eff_score=0.85, eff_margin=0.24, tied=False,
                        ppm_error=win_ppm, pass_no=1, method="cheminfo+grid", confidence="Good", commentary="Pass 1",
                        alternatives=[{"formula": "C22H23N3O7", "adduct": "[M+Na]+", "ion_score": 0.70,
                                       "raw_score": 0.70, "eff_score": 0.61, "ppm": alt_ppm}])
    return led


def test_the_off_cal_re_arbitration_judges_a_carried_winner_at_the_edge():
    # the winner sits 1 ppm off where the axis model, applied up to m/z 440 and reading
    # -1 ppm there, left it uncorrected: on the carried centre it is on calibration and stands
    from peaky.assignment import passes
    rec = _rec(lo=100.0, hi=440.0, v_lo=0.0, v_hi=-1.0)
    led = _rearb_ledger()
    cfg = PCfg.PassConfig()
    cfg.axis_edge_carry = rec
    assert passes.rearbitrate_offcal_degenerate(led, cfg, log=_quiet)["swapped"] == 0
    assert led.loc[led["peak_id"] == "P", "neutral_formula"].iloc[0] == "C36H17N"
    led = _rearb_ledger()
    assert passes.rearbitrate_offcal_degenerate(led, PCfg.PassConfig(), log=_quiet)["swapped"] == 1


@pytest.mark.parametrize("win_ppm, alt_ppm, carried, swapped", [
    # the winner on the carried centre stands, though its alternative is on it too (the
    # winner's own z is read at its m/z)
    (-1.0, -0.9, True, 0),
    # a winner off even the carried centre gives way to an alternative ON it (the
    # alternative's z is read at the same m/z) -- and, without the record, to none
    (+0.6, -0.95, True, 1),
    (+0.6, -0.95, False, 0),
])
def test_the_off_cal_re_arbitration_reads_each_z_at_the_peaks_mz(win_ppm, alt_ppm, carried, swapped):
    from peaky.assignment import passes
    cfg = PCfg.PassConfig()
    cfg.axis_edge_carry = _rec(lo=100.0, hi=440.0, v_lo=0.0, v_hi=-1.0) if carried else None
    led = _rearb_ledger(win_ppm, alt_ppm)
    assert passes.rearbitrate_offcal_degenerate(led, cfg, log=_quiet)["swapped"] == swapped
    assert led.loc[led["peak_id"] == "P", "neutral_formula"].iloc[0] == ("C22H23N3O7" if swapped else "C36H17N")


# --------------------------------------------------------------------------- the direct cal_mu readers
class TestTheDirectReaders:
    """The stages that read calibrate()'s cal_mu themselves judge a line past the model's
    edge at cal_mu plus the carry: each one's line reads the measured axis there (-1.5 ppm
    off a 0.0 centre, sigma 0.3), within reach of the record's last lock: accepted with the
    record, refused without it."""
    V = -1.5

    def _rec_for(self, mz):
        return _rec(lo=mz - 80.0, hi=mz - 10.0, v_lo=0.0, v_hi=self.V)

    def _cfg(self, mz, carried, *, height=10.0, sigma=0.3):
        cfg = PCfg.PassConfig(height_cutoff_cps=height)
        cfg.cal_mu, cfg.cal_sigma, cfg.mechanism_ids = 0.0, sigma, None
        cfg.axis_edge_carry = self._rec_for(mz) if carried else None
        return cfg

    @pytest.mark.parametrize("carried", [True, False])
    def test_the_isotope_gated_recovery(self, carried):
        from peaky.assignment import cleanup as CU
        from peaky.assignment import ledger as L
        from peaky.chem import contexts as X
        mz = C.neutral_mass("C5H10O3") + C.ADDUCT_SHIFTS["[M+Br]-"]
        led = L.new_ledger(pd.DataFrame([("p", mz, 1.0e4), ("twin", mz + CU.BR, 9.6e3), ("noise", 150.0, 5e3)],
                                        columns=["peak_id", "mz", "height"]))

        def score(client, sid, formulas, *, allow_partial=True, mechanism_ids=None):
            return pd.DataFrame([{"ion_formula": "C5H10BrO3-", "ion_score": 0.74, "ppm_error": self.V,
                                  "sample_peak_id": "p", "sample_peak_mz": mz, "sample_peak_intensity": 1.0e4}]
                                if "C5H10O3" in formulas else [])
        out = CU.recover_isotope_gated(None, "S", led, X.get_context("ambient-air"), self._cfg(mz, carried),
                                       score_fn=score, log=_quiet)
        assert out["recovered"] == (1 if carried else 0)
        if carried:
            row = led.loc[led["peak_id"] == "p"].iloc[0]
            assert row["neutral_formula"] == "C5H10O3" and "z=+0.0 within calibrated accuracy" in row["commentary"]

    @pytest.mark.parametrize("carried", [True, False])
    def test_the_speculative_residual_demote(self, carried):
        from peaky.assignment import cleanup as CU
        mz = 199.1
        led = pd.DataFrame([dict(role="M0", neutral_formula="C10H16O4", adduct="[M-H]-", tier="Assigned",
                                 method="residual:iso-pair", ppm_error=self.V, mz=mz, isotopologues="[]",
                                 commentary="", below_assignability=False)])
        out = CU.demote_speculative_residual(led, self._cfg(mz, carried, sigma=0.4), log=_quiet)
        assert out == {"residual_demoted": 0 if carried else 1}
        assert led.at[0, "tier"] == ("Assigned" if carried else "Candidate")

    @pytest.mark.parametrize("carried", [True, False])
    def test_the_labelled_rescue(self, carried):
        from peaky.assignment import labeled as LB
        from peaky.assignment import ledger as L
        from peaky.chem import contexts as X
        neu = "C9H15^NO5"
        mz = C.ion_mz(neu, "[M-H]-")
        led = L.new_ledger(pd.DataFrame({"peak_id": ["p1"], "mz": [mz], "height": [5e4]}))

        def score(client, sid, formulas, allow_partial=True, mechanism_ids=None):
            return pd.DataFrame([dict(compound_formula=neu, ion_formula="C9H14^NO5-", mechanism_id="[M-H]-",
                                      is_base=is_base, ion_score=0.95, compound_score=0.95, sample_peak_mz=mz,
                                      sample_peak_id=pid, ppm_error=self.V)
                                 for is_base, pid in ((True, "p1"), (False, "p1c")) if neu in formulas])
        out = LB.rescue_labeled(None, "s", led, X.get_context("ambient-air"), self._cfg(mz, carried),
                                adducts=["[M-H]-"], label_isotope="^N", label_max=1, score_fn=score, log=_quiet)
        assert out["rescued"] == (1 if carried else 0)

    @pytest.mark.parametrize("carried", [True, False])
    def test_the_reference_list_rescue(self, carried):
        from peaky.assignment import ledger as L
        from peaky.assignment import reflists as RL
        f = "C10H16O5"
        mz = C.ion_mz(f, "[M-H]-")
        rl = RL.ReferenceList(id="t", system="t", label="T", data_version="1", polarity="negative",
                              native_detection="[M-H]-", applies_to_contexts=("x",),
                              references=({"authors": "X", "title": "T"},), formulas=frozenset({f}),
                              radicals=frozenset(), conditions_of={f: ()}, source_file="t.json",
                              always_active=False, meta_of={})
        led = L.new_ledger(pd.DataFrame([("conf", mz, 1.0e5)], columns=["peak_id", "mz", "height"]))

        def oracle(client, sid, formulas, *, allow_partial=True, mechanism_ids=None):
            base = dict(compound_formula=f, compound_score=0.9, ion_formula="C10H15O5-", ion_score=0.9,
                        iso_label="M0", is_base=True, iso_score=0.9, sample_peak_id="conf", sample_peak_mz=mz,
                        sample_peak_intensity=1e4, ppm_error=self.V)
            return pd.DataFrame([base, dict(base, is_base=False, iso_label="13C", sample_peak_mz=mz + 1.00336,
                                            iso_score=0.95)] if f in formulas else [])
        out = RL.rescue_unexplained_by_reflist(None, "S", led, None, self._cfg(mz, carried, height=100.0), [rl],
                                               ["[M-H]-"], score_fn=oracle, log=_quiet)
        assert out["rescued"] == (1 if carried else 0)


# --------------------------------------------------------------------------- a run
class TestARun:
    """An anchorless file on a flat -0.6 ppm axis, corrected over m/z 120-200 by a lock model:
    in scope it is scored at zero on the corrected axis; the acids just above the model's
    last lock (C10-C12, m/z 201-229) sit at the measured -0.6 ppm."""

    def _pair(self):
        runs = {}
        for sid, carry in (("ec-run-carried", True), ("ec-run-plain", False)):
            IO.set_axis_correction([sid], _model(carry=carry))
            runs[carry] = _run(sid)
        return runs[True], runs[False]

    @staticmethod
    def _rows(m0):
        return m0[["peak_id", "neutral_formula", "mz", "ion_score", "ppm_error"]].reset_index(drop=True)

    def test_the_lines_within_reach_are_judged_at_the_edge(self):
        carried, plain = self._pair()
        near = _m0(carried, HI, HI + 30.0)
        assert sorted(near["neutral_formula"]) == ["C10H18O4", "C11H20O4", "C12H22O4"]
        assert (near["ion_score"] >= 0.9).all()
        assert (_m0(plain, HI, HI + 30.0)["ion_score"] < 0.8).all()
        assert carried["pattern_scoring"]["axis_edge_carry"]["reach_da"] == 30.0
        assert carried["stats"]["axis_edge_carry"]["m0_carried"] == 3

    def test_in_scope_and_beyond_the_reach_nothing_changes(self):
        carried, plain = self._pair()
        for lo, hi in ((0.0, HI), (HI + 30.0, 1e9)):
            pd.testing.assert_frame_equal(self._rows(_m0(carried, lo, hi)), self._rows(_m0(plain, lo, hi)))
        assert "axis_edge_carry" not in plain["pattern_scoring"] and "axis_edge_carry" not in plain["stats"]
        # no m/z moved: the carried lines keep their measured m/z
        near = _m0(carried, HI, HI + 30.0).set_index("peak_id")["mz"]
        raw = _peaks().set_index("peak_id")["mz"]
        assert np.array_equal(near.to_numpy(), raw.loc[near.index].to_numpy())

    def test_a_reused_cfg_carries_no_stale_record(self):
        # the record is the sample's own, read per run: a PassConfig that still holds an
        # earlier file's record judges a file without one at its plain centres
        res = _run("ec-run-stale", axis_edge_carry=_rec())
        assert "axis_edge_carry" not in res["stats"] and "axis_edge_carry" not in res["pattern_scoring"]


class TestARunWithItsTiers:
    """A file that calibrates the tier engine: 11 isotope-backed CHO lines in scope (corrected
    to 0 ppm), 8 just below the model's first lock and 13 just above its last (on the measured
    axis, -0.6 ppm), with two lines in scope and two above too weak for a 13C line
    (uncorroborated: the tier engine's z-tail reads them). The degeneracy and tier stages calibrate on the carried scale -- the core's
    centre is 0, so no line reads off-calibration -- and the rows past the edge display their
    error on the corrected rows' footing."""

    @staticmethod
    def _peaks():
        strong = _peaks(RUN_STRONG_LOW + RUN_STRONG_IN + RUN_STRONG_OUT)
        weak = _peaks(RUN_WEAK_IN + RUN_WEAK_OUT)
        weak = weak.assign(height=weak["height"] * 400 / 2e5)
        weak = weak[weak["height"] >= 300].assign(signal_to_noise=lambda d: d["height"] / 30.0,
                                                  area=lambda d: d["height"], peak_id=lambda d: "w" + d["peak_id"])
        return pd.concat([strong, weak]).sort_values("mz").reset_index(drop=True)

    def test_the_tier_calibration_and_the_degeneracy_window_read_the_carried_scale(self):
        sid = "ec-run-tiers"
        IO.set_axis_correction([sid], _model())
        res = _run(sid, peaks=self._peaks())
        led = res["ledger"]
        m0 = led[led["role"] == "M0"].set_index("neutral_formula")
        names = RUN_STRONG_LOW + RUN_STRONG_IN + RUN_STRONG_OUT + RUN_WEAK_IN + RUN_WEAK_OUT
        assert set(names) <= set(m0.index)
        assert res["stats"]["axis_edge_carry"]["m0_carried"] == len(RUN_STRONG_LOW + RUN_STRONG_OUT + RUN_WEAK_OUT)
        # the degeneracy stage's window (stats `degeneracy_cal`) is centred on the carried core
        assert res["stats"]["degeneracy_cal"]["mu"] == pytest.approx(0.0, abs=0.01)
        assert pd.to_numeric(m0.loc[names, "degeneracy_density"]).notna().all()
        # the tier calibration: no uncorroborated line, in scope or past the edge, is off it
        weak = m0.loc[RUN_WEAK_IN + RUN_WEAK_OUT]
        assert (weak["isotopologues"].astype(str) == "[]").all()
        assert (m0.loc[names, "tier"] == T.TIER_ASSIGNED).all()
        assert not m0["tier_reason"].astype(str).str.contains("instrument accuracy").any()
        # the stamp: the rows past the edge read -0.6 ppm as measured, 0 off the carried centre,
        # and the centre is the corrected rows' (0), not the measured axis the most rows read
        past = m0.loc[RUN_STRONG_LOW + RUN_STRONG_OUT + RUN_WEAK_OUT]
        assert np.allclose(past["ppm_error"], SHIFT, atol=0.01)
        assert np.allclose(past["ppm_error_cal"], 0.0, atol=0.01)
        assert np.allclose(m0.loc[RUN_STRONG_IN + RUN_WEAK_IN, "ppm_error_cal"], 0.0, atol=0.01)
