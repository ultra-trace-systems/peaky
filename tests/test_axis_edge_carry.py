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
        # here): not carried; one measured past it stays where it was: carried
        model = _model(carry=False, y=+0.6)
        rec = _rec(v_lo=+0.6, v_hi=+0.6)
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
        assert rec["below"] == {"mz": 120.0, "ppm": pytest.approx(0.2)}
        assert rec["above"] == {"mz": 300.0, "ppm": pytest.approx(-0.5)}
        got = MC.edge_carry_at(rec, [100.0, 155.0, 210.0, 250.0, 320.0])
        assert got.tolist() == pytest.approx([0.2, 0.0, 0.0, 0.0, -0.5])

    def test_the_record_travels_with_the_model(self):
        m = _model()
        assert m.edge_carry == _rec()
        back = MQ.fit_from_record(m.as_dict())
        assert back.edge_carry == _rec()
        assert _model(carry=False).edge_carry is None
        assert MQ.fit_from_record(_model(carry=False).as_dict()).edge_carry is None


# --------------------------------------------------------------------------- the grant
def _tiny_ts():
    return pd.DataFrame({"sample_item_id": ["s0"] * 4 + ["s1"] * 4,
                         "mz": [100.0, 150.0, 210.0, 260.0] * 2, "height": [1e4] * 8})


@pytest.mark.parametrize("cv,raw,peak,carried", [
    (0.06, 0.3, 0.6, True),          # a sub-ppm model the card's bar applies: carried
    (0.1, 0.5, 0.9, True),           # still under 1 ppm
    (0.1, 0.5, 1.2, False),          # a 1.2 ppm model (applied at the reference bar too): never
    (0.1, 1.1, 0.9, False),          # the locks' raw median at 1 ppm or more: never
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
        assert {k: v for k, v in carry.items() if k != "share"} == _rec()
        assert 0 < carry["share"] < 1 and snap["mu_source"] == "axis"
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
    rows = []
    for k in range(n_in + n_out):
        inside = k < n_in
        mz = 130.0 + 6.0 * k if inside else 201.0 + 2.0 * (k - n_in)
        ppm = (0.02 if k % 2 else -0.02) + (0.0 if inside else out_ppm)
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


def _rearb_ledger():
    """20 isotope-backed CHO backbone rows near 0 ppm (m/z 150-169) and one off-cal aromatic
    monster M0 at m/z 464.143, -1.0 ppm, with an on-cal plausible lower-DBE alternative."""
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
                        ppm_error=-1.0, pass_no=1, method="cheminfo+grid", confidence="Good", commentary="Pass 1",
                        alternatives=[{"formula": "C22H23N3O7", "adduct": "[M+Na]+", "ion_score": 0.70,
                                       "raw_score": 0.70, "eff_score": 0.61, "ppm": 0.15}])
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
