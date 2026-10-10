"""The scoring centre of a file with no anchors: the batch's axis, else its own backbone.

A sample's candidates are judged against its mass offset. Mascope's anchors give one
when at least five matched; otherwise the run scored at an ASSUMED zero, wherever the
file sat. At the Orbitrap class width (0.58 ppm) a correct ion 0.6 ppm off misses the
pass-1 floor, so an anchorless file 0.6 ppm off commits almost nothing, and the pass-1
calibration cannot see why: its backbone is the rows the zero-centred score let through.

Pinned here:
  * the ladder in `io_mascope.scoring_for_sample`: own anchors ("fitted") > the batch's
    axis correction ("axis") > the file's self-centre ("self") > "assumed_zero";
  * `selfcentre`: the backbone is chosen WITHOUT the mass term (unique, pattern-Good,
    13C-confirmed CHO / CHON ions on bright peaks), and the centre is stated only on
    enough, tight, in-window points;
  * `assign.run` measures it once, before pass 0, only on an anchorless Orbitrap-class
    file the axis did not reach; an anchored file runs byte-identically; a stand-in
    judged at an inherited snapshot (a decoy arm) keeps the inherited centre;
  * the batch records each file's `mu_source` and counts them.

Offline: synthetic spectra, no network.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mascope_tools.composition.heuristic_filter import (  # noqa: E402
    anchor_on_monoisotopic,
    predict_isotopes,
)

from peaky.assignment import assign as A  # noqa: E402
from peaky.assignment import selfcentre as SC  # noqa: E402
from peaky.assignment.passes import config as PCfg  # noqa: E402
from peaky.batch import assign_batch as AB  # noqa: E402
from peaky.batch.wave import WaveFit  # noqa: E402
from peaky.chem import chemistry as C  # noqa: E402
from peaky.io import io_mascope as IO  # noqa: E402

#: CHO acids [M-H]-, m/z 131-497, every line present
ACIDS = [f"C{n}H{2 * n - 2}O4" for n in range(5, 25)] + [f"C{n}H{2 * n - 4}O6" for n in range(18, 30)]
SHIFT = -0.6        # the file's offset (ppm): a pattern-perfect line there misses the pass-1 floor at 0


def _ion(nf: str) -> str:
    cnt = dict(C.parse_formula(nf))
    cnt["H"] -= 1                                            # [M-H]-
    return C.format_formula({k: v for k, v in cnt.items() if v > 0})


def _peaks(neutrals=ACIDS, shift=SHIFT, *, anchors: int = 0) -> pd.DataFrame:
    """Every predicted line of each neutral's [M-H]- ion, `shift` ppm off; heights the
    predicted abundances (only the mass term moves a score). The first `anchors`
    ions carry a server match (an anchor) on their M0 line."""
    rows = []
    for k, nf in enumerate(neutrals):
        mzs, ints, _ = anchor_on_monoisotopic(*predict_isotopes(_ion(nf), -1))
        for i, (m, h) in enumerate(zip(mzs, ints)):
            height = h / ints[0] * 2e5
            if height < 300:
                continue
            row = {"peak_id": f"p{k}_{i}", "mz": m * (1 + shift * 1e-6), "height": height,
                   "area": height, "signal_to_noise": height / 30.0}
            if anchors:
                match = i == 0 and k < anchors
                row.update(target_compound_formula=nf if match else None,
                           target_isotope_formula=None,
                           ionization_mechanism="[M-H]-" if match else None)
            rows.append(row)
    return pd.DataFrame(rows).sort_values("mz").reset_index(drop=True)


def _const(ppm: float, lo: float = 100.0, hi: float = 600.0) -> WaveFit:
    """A constant axis correction (reaches every m/z)."""
    return WaveFit(coef=[ppm], p=-0.5, domain=(hi ** -0.5, lo ** -0.5), mz_range=(lo, hi), K=0,
                   n=10, n_clipped=0, resid_ppm=0.1, raw_ppm=0.1, loo_ppm=0.1, span_ppm=0.0, share=0.0)


def _scoped(ppm: float, lo: float, hi: float) -> WaveFit:
    """A flat correction that reaches only its calibrants' range lo-hi (a trend, K = 1)."""
    return WaveFit(coef=[ppm, 0.0], p=-0.5, domain=(hi ** -0.5, lo ** -0.5), mz_range=(lo, hi), K=1,
                   n=10, n_clipped=0, resid_ppm=0.1, raw_ppm=0.1, loo_ppm=0.1, span_ppm=0.0, share=0.9)


def _run(sid, peaks=None, *, scoring="orbi", **cfg_kw):
    log = []
    cfg = PCfg.PassConfig(height_cutoff_cps=100, **cfg_kw)
    res = A.run(sid, context="ambient-air", cfg=cfg, peaks=_peaks() if peaks is None else peaks,
                adducts=["[M-H]-"], do_pass2=False, do_pass3=False, do_pass4=False, do_pass5=False,
                do_pass_certified=False, use_cache=False, log=log.append, scoring=scoring)
    return res, [str(m) for m in log]


def _m0(res) -> pd.DataFrame:
    led = res["ledger"]
    return led[led["role"] == "M0"]


@pytest.fixture(autouse=True)
def _forget():
    yield
    for sid in list(IO._SELF_CENTRE):
        IO.reset_self_centre(sid)
    IO.clear_axis_correction()


# --------------------------------------------------------------------------- the ladder
class TestTheCentreLadder:
    SID = "sc-ladder"

    def _snap(self, peaks):
        IO.register_offline_sample(self.SID, peaks, ["[M-H]-"], scoring="orbi")
        return IO.scoring_for_sample(None, self.SID), IO.scoring_snapshot(None, self.SID)

    def teardown_method(self):
        IO.unregister_offline_sample(self.SID)

    def test_no_anchors_and_nothing_else_is_an_assumed_zero(self):
        sc, snap = self._snap(_peaks())
        assert sc.mu_ppm == 0.0 and snap["mu_source"] == "assumed_zero"
        assert "self_centre" not in snap and "axis_scope_share" not in snap

    def test_a_self_centre_is_what_an_anchorless_file_is_scored_at(self):
        self._snap(_peaks())
        IO.set_self_centre(self.SID, {"mu_ppm": -0.6, "n": 30, "spread_ppm": 0.05})
        sc, snap = IO.scoring_for_sample(None, self.SID), IO.scoring_snapshot(None, self.SID)
        assert sc.mu_ppm == pytest.approx(-0.6) and snap["mu_source"] == "self"
        assert snap["self_centre"] == {"n": 30, "spread_ppm": 0.05}
        IO.reset_self_centre(self.SID)
        assert IO.scoring_for_sample(None, self.SID).mu_ppm == 0.0

    def test_the_axis_correction_outranks_the_self_centre(self):
        self._snap(_peaks())
        IO.set_self_centre(self.SID, {"mu_ppm": -0.6})
        IO.set_axis_correction([self.SID], _const(-0.6, 140.0, 600.0))
        sc, snap = IO.scoring_for_sample(None, self.SID), IO.scoring_snapshot(None, self.SID)
        assert sc.mu_ppm == 0.0 and snap["mu_source"] == "axis"
        assert snap["axis_scope_share"] == 1.0                   # a constant reaches every m/z
        # the axis correction's clear forgets the self-centre measured on the old axis
        IO.clear_axis_correction([self.SID])
        assert IO.self_centre(self.SID) is None

    def test_own_anchors_outrank_both_and_the_snapshot_is_unchanged(self):
        peaks = _peaks(anchors=6)
        _, before = self._snap(peaks)
        assert before["mu_source"] == "fitted" and before["mu_ppm"] == pytest.approx(SHIFT, abs=1e-3)
        IO.set_self_centre(self.SID, {"mu_ppm": 0.9})
        assert IO.scoring_snapshot(None, self.SID) == before
        IO.set_axis_correction([self.SID], _const(0.0))
        IO.reset_self_centre(self.SID)
        assert IO.scoring_snapshot(None, self.SID)["mu_source"] == "fitted"

    def test_a_re_registration_forgets_it(self):
        self._snap(_peaks())
        IO.set_self_centre(self.SID, {"mu_ppm": -0.6})
        self._snap(_peaks())
        assert IO.self_centre(self.SID) is None


class TestOutsideTheAxisModelsScope:
    """The batch's axis model corrects only where its locks reach. An anchored file's
    fitted centre is measured on the corrected peaks, so the peaks the model left
    alone would be judged at a centre that is no longer theirs: they take the
    correction the model applies where the file's anchors sit."""

    @staticmethod
    def _mzs(res, lo, hi):
        led = res["ledger"]
        m0 = led[(led["role"] == "M0") & (led["mz"] >= lo) & (led["mz"] <= hi)]
        return m0

    def test_an_anchored_files_peaks_outside_the_scope_follow_its_anchors(self):
        sid = "sc-out-anch"
        IO.set_axis_correction([sid], _scoped(SHIFT, 120.0, 300.0))     # anchors (m/z 131-243) inside
        res, _ = _run(sid, _peaks(anchors=8))
        snap = res["pattern_scoring"]
        assert snap["mu_source"] == "fitted" and snap["mu_ppm"] == pytest.approx(0.0, abs=0.02)
        assert snap["axis_outside_shift_ppm"] == pytest.approx(SHIFT, abs=1e-3)
        assert 0 < snap["axis_scope_share"] < 1
        # every acid commits on mass, the ones above m/z 300 (outside the model) included
        assert len(_m0(res)) == len(ACIDS) and (_m0(res)["ion_score"] > 0.9).all()
        out = self._mzs(res, 300.5, 600.0)
        assert len(out) and (out["ppm_error"].abs() < 0.05).all()

    def test_without_the_shift_they_would_sit_off_the_fitted_centre(self, monkeypatch):
        # the same file with the shift switched off: the peaks above m/z 300 stay
        # on the raw axis while the anchors' centre reads the corrected one
        sid = "sc-out-noshift"
        monkeypatch.setattr(IO, "_outside_shift", lambda *a, **k: None)
        IO.set_axis_correction([sid], _scoped(SHIFT, 120.0, 300.0))
        res, _ = _run(sid, _peaks(anchors=8))
        assert self._mzs(res, 300.5, 600.0).empty

    def test_an_anchorless_files_peaks_outside_the_scope_stay_uncorrected(self):
        sid = "sc-out-bare"
        IO.set_axis_correction([sid], _scoped(SHIFT, 120.0, 300.0))
        res, log = _run(sid)
        snap = res["pattern_scoring"]
        assert snap["mu_source"] == "axis" and "axis_outside_shift_ppm" not in snap
        assert not any("[self-centre]" in m for m in log)
        assert len(self._mzs(res, 120.0, 300.0)) and self._mzs(res, 300.5, 600.0).empty

    def test_too_few_anchors_inside_the_scope_shift_nothing(self):
        sid = "sc-out-few"
        IO.register_offline_sample(sid, _peaks(anchors=8), ["[M-H]-"], scoring="orbi")
        IO.set_axis_correction([sid], _scoped(SHIFT, 120.0, 140.0))     # 1 anchor inside
        try:
            got = IO.fetch_peaks(None, sid)
            assert IO._AXIS_OUTSIDE[sid] is None
            raw = _peaks(anchors=8).set_index("peak_id")["mz"]
            far = got[got["mz"] > 141.0]
            assert np.allclose(far["mz"].to_numpy(), raw.loc[far["peak_id"]].to_numpy(), rtol=0, atol=1e-12)
        finally:
            IO.unregister_offline_sample(sid)


# --------------------------------------------------------------------------- the backbone
def _flat(rows) -> pd.DataFrame:
    """A local-scorer flat table: (peak, neutral, ion, massfree, M0 ppm, 13C ppm or None)."""
    out = []
    for pid, nf, ion, mf, ppm, c13 in rows:
        common = {"compound_formula": nf, "mechanism_id": "[M-H]-", "ion_formula": ion,
                  "ion_score_massfree": mf}
        out.append({**common, "is_base": True, "iso_label": "M0", "sample_peak_id": pid, "ppm_error": ppm})
        out.append({**common, "is_base": False, "iso_label": "13C",
                    "sample_peak_id": None if c13 is None else pid + "c", "ppm_error": c13})
    return pd.DataFrame(out)


class TestTheBackbone:
    cfg = PCfg.PassConfig()

    def test_a_unique_pattern_good_13c_confirmed_cho_ion_is_a_point(self):
        pts = SC.backbone(_flat([("a", "C5H8O4", "C5H7O4-", 0.97, -0.55, -0.4)]), ["a"], self.cfg)
        assert pts["peak_id"].tolist() == ["a"] and pts["ppm"].tolist() == [-0.55]

    @pytest.mark.parametrize("rows,why", [
        ([("a", "C5H8O4", "C5H7O4-", 0.97, -0.5, -0.4), ("a", "C4H6N2O3", "C4H5N2O3-", 0.6, 1.9, 2.0)],
         "a rival's pattern holds there too"),
        ([("a", "C5H8O4", "C5H7O4-", 0.75, -0.5, -0.4)], "pattern under Good"),
        ([("a", "C5H8O4", "C5H7O4-", 0.97, -0.5, None)], "no 13C line"),
        ([("a", "C5H8O4", "C5H7O4-", 0.97, -0.5, 1.6)], "its 13C line sits 2.1 ppm away"),
        ([("a", "C2H6OS", "C2H5OS-", 0.97, -0.5, -0.4)], "not a CHO / CHON reading"),
        ([("a", "HNO3", "NO3-", 0.97, -0.5, -0.5)], "no carbon"),
    ])
    def test_anything_less_is_not(self, rows, why):
        assert SC.backbone(_flat(rows), ["a"], self.cfg).empty, why

    def test_a_weak_rival_struck_by_its_isotopes_leaves_the_point(self):
        rows = [("a", "C5H8O4", "C5H7O4-", 0.97, -0.5, -0.4), ("a", "C4H6N2O3", "C4H5N2O3-", 0.2, 1.9, None)]
        assert len(SC.backbone(_flat(rows), ["a"], self.cfg)) == 1

    def test_two_readings_of_one_ion_are_one_ion(self):
        rows = [("a", "C5H8O4", "C5H7O4-", 0.97, -0.5, -0.4), ("a", "C5H9O4", "C5H7O4-", 0.97, -0.5, -0.4)]
        assert len(SC.backbone(_flat(rows), ["a"], self.cfg)) == 1


class TestTheCentre:
    cfg = PCfg.PassConfig()

    @staticmethod
    def _pts(ppms):
        return pd.DataFrame({"ppm": ppms}), np.linspace(100, 300, len(ppms))

    def test_enough_tight_points_state_and_apply_it(self):
        rec = SC.centre_of(*self._pts([-0.6, -0.55, -0.62, -0.58, -0.65, -0.5, -0.6, -0.61]), self.cfg)
        assert rec["applied"] and rec["mu_ppm"] == pytest.approx(-0.6, abs=0.01) and rec["n"] == 8

    def test_too_few_points_state_nothing(self):
        rec = SC.centre_of(*self._pts([-0.6] * (SC.MIN_N - 1)), self.cfg)
        assert not rec["applied"] and rec["mu_ppm"] is None and "needs" in rec["why"]

    def test_a_wide_scatter_states_nothing(self):
        rec = SC.centre_of(*self._pts(list(np.linspace(-2.0, 1.0, 12))), self.cfg)
        assert not rec["applied"] and rec["mu_ppm"] is None and "scatters" in rec["why"]

    def test_a_backbone_reaching_the_window_edge_is_truncated_and_not_stated(self):
        # the +-3 ppm enumeration cannot hold the tail of a backbone centred 2.5 ppm off
        # with a 0.4 ppm spread: its median is biased, so it is not stated
        pts = [-2.9, -2.1, -2.5, -2.95, -2.05, -2.6, -2.4, -2.8, -2.2]
        rec = SC.centre_of(*self._pts(pts), self.cfg)
        assert not rec["applied"] and rec["mu_ppm"] is None and "window" in rec["why"]

    def test_a_tight_backbone_far_off_is_stated(self):
        # a source sitting -2.6 ppm off: inside the window, so measured
        rec = SC.centre_of(*self._pts([-2.6, -2.7, -2.5, -2.65, -2.55, -2.6, -2.62, -2.58]), self.cfg)
        assert rec["applied"] and rec["mu_ppm"] == pytest.approx(-2.6, abs=0.02)

    def test_a_centre_near_zero_is_measured_but_not_applied(self):
        rec = SC.centre_of(*self._pts([0.05, 0.02, -0.03, 0.04, 0.06, 0.0, 0.03, 0.05]), self.cfg)
        assert not rec["applied"] and rec["mu_ppm"] is not None and "zero" in rec["why"]


class TestEligibility:
    def _cfg(self, **kw):
        cfg = PCfg.PassConfig(**kw)
        cfg.instrument_class = kw.pop("instrument_class", "orbitrap")
        return cfg

    SNAP = {"mu_source": "assumed_zero", "instrument_type": "orbi"}

    def test_an_anchorless_orbitrap_file_scored_locally_is_eligible(self):
        assert SC.eligibility(self.SNAP, self._cfg(), inherited=False, local=True) is None

    @pytest.mark.parametrize("snap,cfg_kw,inherited,local,word", [
        ({"mu_source": "fitted", "instrument_type": "orbi"}, {}, False, True, "fitted"),
        ({"mu_source": "axis", "instrument_type": "orbi"}, {}, False, True, "axis"),
        ({"mu_source": "inherited", "instrument_type": "orbi"}, {}, False, True, "inherited"),
        ({"mu_source": "assumed_zero", "instrument_type": "tof"}, {}, False, True, "Orbitrap"),
        ({"mu_source": "assumed_zero", "instrument_type": None}, {}, False, True, "Orbitrap"),
        ({"mu_source": "assumed_zero", "instrument_type": "orbi"}, {"self_centre": False}, False, True, "off"),
        ({"mu_source": "assumed_zero", "instrument_type": "orbi"}, {"trace_sample": True}, False, True, "trace"),
        ({"mu_source": "assumed_zero", "instrument_type": "orbi"}, {}, True, True, "trend"),
        ({"mu_source": "assumed_zero", "instrument_type": "orbi"}, {}, False, False, "network"),
    ])
    def test_everything_else_keeps_its_centre(self, snap, cfg_kw, inherited, local, word):
        why = SC.eligibility(snap, self._cfg(**cfg_kw), inherited=inherited, local=local)
        assert why is not None and word in why

    def test_a_tof_class_run_is_not_eligible(self):
        assert "Orbitrap" in SC.eligibility(self.SNAP, self._cfg(instrument_class="tof"),
                                            inherited=False, local=True)


# --------------------------------------------------------------------------- the run
class TestTheRun:
    def test_an_anchorless_file_off_zero_is_scored_at_its_own_centre(self):
        off, _ = _run("sc-run-off", self_centre=False)
        on, log = _run("sc-run-on")
        assert any("[self-centre]" in m and "from pass 0" in m for m in log), log
        snap = on["pattern_scoring"]
        assert snap["mu_source"] == "self" and snap["mu_ppm"] == pytest.approx(SHIFT, abs=0.05)
        assert snap["self_centre"]["n"] >= SC.MIN_N
        assert on["stats"]["mu_source"] == "self" and on["stats"]["self_centre"]["applied"]
        # at zero the -0.6 ppm acids miss the pass-1 floor; at the centre they commit Good
        assert len(_m0(on)) == len(ACIDS) and (_m0(on)["ion_score"] > 0.9).all()
        assert len(_m0(off)) < len(ACIDS) // 2
        assert off["pattern_scoring"]["mu_source"] == "assumed_zero"
        assert off["stats"]["self_centre"] == {"applied": False, "why": "off (PassConfig.self_centre=False)"}

    def test_the_centre_holds_through_every_re_run(self, monkeypatch):
        seen = []
        real = A.passes.run_pass1

        def spy(client, sid, led, profile, pre, cfg, adducts, log=print):
            seen.append(IO.scoring_for_sample(client, sid).mu_ppm)
            return real(client, sid, led, profile, pre, cfg, adducts, log=log)

        monkeypatch.setattr(A.passes, "run_pass1", spy)
        _run("sc-run-reruns")
        assert seen and all(v == pytest.approx(SHIFT, abs=0.05) for v in seen), seen

    def test_a_file_at_zero_runs_as_before(self):
        on, log = _run("sc-zero-on", _peaks(shift=0.0))
        off, _ = _run("sc-zero-off", _peaks(shift=0.0), self_centre=False)
        assert on["stats"]["self_centre"]["applied"] is False and "zero" in on["stats"]["self_centre"]["why"]
        assert on["pattern_scoring"]["mu_source"] == "assumed_zero"
        pd.testing.assert_frame_equal(on["ledger"], off["ledger"])

    def test_an_anchored_file_is_byte_identical(self):
        peaks = _peaks(anchors=6)
        on, log = _run("sc-anch-on", peaks)
        off, _ = _run("sc-anch-off", peaks, self_centre=False)
        assert not any("[self-centre]" in m for m in log)
        assert on["pattern_scoring"]["mu_source"] == "fitted" and on["stats"]["self_centre"] is None
        pd.testing.assert_frame_equal(on["ledger"], off["ledger"])
        assert on["pattern_scoring"] == off["pattern_scoring"]

    def test_a_file_the_axis_reached_is_scored_at_zero_on_the_corrected_axis(self):
        sid = "sc-axis"
        IO.set_axis_correction([sid], _const(SHIFT))
        res, log = _run(sid)
        assert not any("[self-centre]" in m for m in log)
        assert res["pattern_scoring"]["mu_source"] == "axis" and res["pattern_scoring"]["mu_ppm"] == 0.0
        assert res["stats"]["self_centre"] is None
        assert len(_m0(res)) == len(ACIDS)

    def test_a_stand_in_keeps_the_centre_it_inherits(self):
        on, _ = _run("sc-measured")
        arm, log = _run("sc-arm", scoring=dict(on["pattern_scoring"]))
        assert not any("[self-centre]" in m for m in log)
        snap = arm["pattern_scoring"]
        assert snap["mu_source"] == "inherited" and snap["inherited"]["mu_source"] == "self"
        assert snap["mu_ppm"] == pytest.approx(on["pattern_scoring"]["mu_ppm"])

    def test_a_ppm_decoy_stand_in_is_not_re_centred_onto_its_shift(self):
        # the decoy arm's lines sit 3 ppm further off: judged at the measured centre
        # they stay off it (a self-centre of the decoy would have moved onto them)
        on, _ = _run("sc-decoy-src")
        arm, _ = _run("sc-decoy", _peaks(shift=SHIFT + 3.0), scoring=dict(on["pattern_scoring"]))
        assert arm["pattern_scoring"]["mu_ppm"] == pytest.approx(on["pattern_scoring"]["mu_ppm"])
        assert (_m0(arm)["ion_score"] < 0.5).all() if len(_m0(arm)) else True

    def test_a_batch_worker_returns_the_self_centred_snapshot(self):
        kw = dict(cfg=PCfg.PassConfig(height_cutoff_cps=100), peaks=_peaks(), adducts=["[M-H]-"],
                  do_pass2=False, do_pass3=False, do_pass4=False, do_pass5=False,
                  do_pass_certified=False, use_cache=False, scoring="orbi")
        AB._worker_init("ambient-air", None, kw, None)
        out = AB._assign_one("sc-worker")
        assert out["pattern_scoring"]["mu_source"] == "self"
        assert out["stats"]["mu_source"] == "self"


# --------------------------------------------------------------------------- the batch record
def test_the_batch_counts_each_files_centre_source():
    got = AB.scoring_centres({
        "a": {"mu_source": "fitted", "mu_ppm": -0.5},
        "b": {"mu_source": "self", "mu_ppm": -0.55, "self_centre": {"n": 20}},
        "c": {"mu_source": "trend", "mu_ppm": -0.4, "self_centre": {"n": 12}},
        "d": {"mu_source": "axis", "mu_ppm": 0.0},
        "e": {"mu_source": "assumed_zero", "mu_ppm": 0.0},
        "f": None,
    })
    assert got["by_mu_source"] == {"assumed_zero": 1, "axis": 1, "fitted": 1, "self": 1, "trend": 1,
                                   "unrecorded": 1}
    assert got["self"] == {"b": -0.55, "c": -0.4}
