"""C42: a candidate's mass is judged at the sample's own mass-dependent centre.

The v2 score judged every line against ONE offset per sample (the server
matches' median). An Orbitrap whose error runs as 1/mz -- the labelled-nitrate
run sits at +0.8 ppm at m/z 131 and -0.2 ppm at 400 -- then has every low-mass
ion judged a sigma off a centre it does not sit at. The calibration that could
fit the trend selected its backbone by that same score, so it only ever saw
rows already near the constant offset and rejected the trend (12/12 files, and
all 12 pooled). Three pieces break the loop and are pinned here:

  * the scorer takes a per-line centre (`centre=`, a masscal.MassTrend) and
    also reports `ion_score_massfree`, the isotope pattern's score alone;
  * calibrate() fits the trend on rows chosen by the pattern-only score;
  * assign.run, once a trend is accepted, re-runs the file from pass 0 at the
    per-line centre, and records the trend in the sample's scoring snapshot,
    which a stand-in (a decoy arm) inherits.

Also C42(c) -- "unique formula in the calibrated window" needs the degeneracy
audit to have run -- and C43 -- the reagent halogen comes from the declared
channels, not a count of committed adducts.

Offline: no network.
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
from peaky.assignment import evidence as EV  # noqa: E402
from peaky.assignment import ledger as L  # noqa: E402
from peaky.assignment import masscal as MC  # noqa: E402
from peaky.assignment import tiers as T  # noqa: E402
from peaky.assignment.passes import config as PCfg  # noqa: E402
from peaky.assignment.passes import core as PC  # noqa: E402
from peaky.chem import chemistry as C  # noqa: E402
from peaky.io import io_mascope as IO  # noqa: E402
from peaky.io.local_scoring import score_candidates_local  # noqa: E402

# an R1-like instrument: the server matches' constant offset, and the 1/mz trend
# the labelled-nitrate Orbitrap's own backbone shows
ORBI = PatternScoring(sigma_ppm=0.58, mu_ppm=0.175, mz_tolerance_ppm=5.0)
A_PPM, B_MDA = -0.6, 0.18
TREND = MC.MassTrend(a=A_PPM, b=B_MDA, sigma=0.25, n=300, mz_lo=130.0, mz_hi=700.0)


def _centre(mz: float) -> float:
    return A_PPM + B_MDA * 1000.0 / mz


def _peaks(neutrals, adduct="[M-H]-", *, ppm=None, start=0) -> pd.DataFrame:
    """Every predicted line of each neutral's ion, all shifted by `ppm(mz)`
    (default: the trend's centre at the ion's own m/z); heights are the
    predicted abundances, so only the mass term moves a score."""
    ppm = ppm or (lambda mz: _centre(mz))
    rows = []
    for k, nf in enumerate(neutrals):
        cnt = dict(C.parse_formula(nf))
        cnt["H"] = cnt.get("H", 0) - 1                       # [M-H]-
        ion = C.format_formula({k2: v for k2, v in cnt.items() if v > 0})
        mzs, ints, _ = anchor_on_monoisotopic(*predict_isotopes(ion, -1))
        shift = ppm(float(mzs[0]))
        for i, (m, h) in enumerate(zip(mzs, ints)):
            height = h / ints[0] * 2e5
            if height < 300:
                continue
            rows.append({"peak_id": f"p{start + k}_{i}", "mz": m * (1 + shift * 1e-6),
                         "height": height, "area": height, "signal_to_noise": height / 30.0})
    return pd.DataFrame(rows).sort_values("mz").reset_index(drop=True)


def _base(flat: pd.DataFrame) -> pd.Series:
    return flat[flat["is_base"]].iloc[0]


# --------------------------------------------------------------------------- the scorer
class TestThePerLineCentre:
    def test_a_low_mass_ion_at_the_trend_centre_scores_at_the_trend_and_not_at_the_offset(self):
        # C5H8O4 [M-H]- at m/z 131: the trend puts it +0.77 ppm, the constant
        # offset +0.175 -- a sigma away. Judged at its own centre it is on mass.
        peaks = _peaks(["C5H8O4"])
        at_offset = _base(score_candidates_local(peaks, ["C5H8O4"], ["[M-H]-"], scoring=ORBI))
        at_trend = _base(score_candidates_local(peaks, ["C5H8O4"], ["[M-H]-"], scoring=ORBI,
                                                centre=TREND))
        assert at_trend["ion_score"] > 0.9 > 0.8 > at_offset["ion_score"]

    def test_a_heavy_ion_is_judged_at_the_centre_of_its_own_mass(self):
        # at m/z 277 the trend centre is +0.05 ppm: the constant offset is
        # nearly right there, and the trend keeps it right
        peaks = _peaks(["C12H22O7"])
        at_trend = _base(score_candidates_local(peaks, ["C12H22O7"], ["[M-H]-"], scoring=ORBI,
                                                centre=TREND))
        assert at_trend["ion_score"] > 0.9

    def test_outside_the_trend_s_coverage_the_centre_is_held_at_the_edge(self):
        # coverage starts at m/z 200: an m/z 131 ion is judged at the centre AT
        # 200 (+0.30 ppm), never a 1/mz extrapolation down to it
        edge = MC.MassTrend(A_PPM, B_MDA, 0.25, 300, 200.0, 700.0)
        at_edge_centre = _peaks(["C5H8O4"], ppm=lambda mz: _centre(200.0))
        at_extrapolated = _peaks(["C5H8O4"])
        s_edge = _base(score_candidates_local(at_edge_centre, ["C5H8O4"], ["[M-H]-"],
                                              scoring=ORBI, centre=edge))["ion_score"]
        s_extra = _base(score_candidates_local(at_extrapolated, ["C5H8O4"], ["[M-H]-"],
                                               scoring=ORBI, centre=edge))["ion_score"]
        assert s_edge > 0.9 > s_extra

    def test_the_reported_error_stays_the_raw_one(self):
        peaks = _peaks(["C5H8O4"])
        row = _base(score_candidates_local(peaks, ["C5H8O4"], ["[M-H]-"], scoring=ORBI,
                                           centre=TREND))
        assert row["ppm_error"] == pytest.approx(_centre(row["theo_mz"]), abs=0.02)


class TestThePatternOnlyScore:
    def test_it_ignores_the_mass_error(self):
        # 1.5 ppm off at a 0.58 ppm width: the full score is spent, the
        # isotope pattern is not
        off = _base(score_candidates_local(_peaks(["C9H14O4"], ppm=lambda mz: 1.5),
                                           ["C9H14O4"], ["[M-H]-"], scoring=ORBI))
        on = _base(score_candidates_local(_peaks(["C9H14O4"], ppm=lambda mz: 0.175),
                                          ["C9H14O4"], ["[M-H]-"], scoring=ORBI))
        assert off["ion_score"] < 0.2
        assert off["ion_score_massfree"] > 0.9
        assert off["ion_score_massfree"] == pytest.approx(on["ion_score_massfree"], abs=1e-9)

    def test_it_reaches_the_ledger_through_the_commit(self):
        led = L.new_ledger(pd.DataFrame({"peak_id": ["a"], "mz": [131.035], "height": [1e5]}))
        L.commit_assignment(led, "a", neutral_formula="C5H8O4", adduct="[M-H]-",
                            ion_formula="C5H7O4-", ion_score=0.6, ion_score_massfree=0.97,
                            ppm_error=0.8, pass_no=1, method="cheminfo+grid",
                            confidence="Low", commentary="c")
        assert led.loc[0, "ion_score_massfree"] == pytest.approx(0.97)


# --------------------------------------------------------------------------- the calibration
_ACIDS = [f"C{n}H{2 * n - 2}O4" for n in range(5, 25)] + [f"C{n}H{2 * n - 4}O6" for n in range(18, 30)]


def _backbone_ledger(*, with_pattern: bool, seed: int = 7) -> pd.DataFrame:
    """CHO [M-H]- rows (m/z 131-560) on the R1-like trend, each with the full
    score the constant-offset scorer would give it and a pattern-only score."""
    rng = np.random.default_rng(seed)
    rows, commits = [], []
    for i, nf in enumerate(_ACIDS):
        mz = C.ion_mz(nf, "[M-H]-")
        ppm = _centre(mz) + rng.normal(0, 0.05)
        full = float(np.exp(-0.5 * ((ppm - ORBI.mu_ppm) / ORBI.sigma_ppm) ** 2))
        rows += [(f"b{i}", mz, 1e5), (f"b{i}c", mz + 1.003355, 1e4)]
        commits.append((f"b{i}", nf, ppm, full))
    # dim O-rich coincidences: an untested pattern (no satellite), off the trend
    for j, nf in enumerate(["C12H18O17", "C11H14O16", "C12H20O18", "C14H16O17", "C13H16O18"]):
        mz = C.ion_mz(nf, "[M-H]-")
        rows.append((f"j{j}", mz, 3e2))
        commits.append((f"j{j}", nf, 0.45, 0.95))
    led = L.new_ledger(pd.DataFrame(rows, columns=["peak_id", "mz", "height"]))
    for pid, nf, ppm, full in commits:
        cnt = dict(C.parse_formula(nf))
        cnt["H"] -= 1
        L.commit_assignment(led, pid, neutral_formula=nf, adduct="[M-H]-",
                            ion_formula=C.format_formula(cnt) + "-", ion_score=full,
                            ion_score_massfree=(0.97 if with_pattern else None),
                            ppm_error=ppm, pass_no=1, method="cheminfo+grid",
                            confidence="Good", commentary="backbone")
        if pid.startswith("b"):
            L.attach_isotopologue(led, pid + "c", pid, iso_label="13C", iso_match_score=0.9)
    return led


class TestTheCalibrationBackbone:
    def test_an_untested_pattern_does_not_enter_the_trend(self):
        # the five dim O-rich rows score ~1 on the pattern (no satellite to
        # contradict them) and sit off the trend: they are not in its backbone
        cfg = PCfg.PassConfig()
        logs = []
        PC.calibrate(_backbone_ledger(with_pattern=True), cfg, log=logs.append)
        line = next(m for m in logs if "mass trend" in m)
        assert f"of {len(_ACIDS)} isotope-tested" in line, line

    def test_a_full_score_backbone_loses_the_low_masses_the_trend_is_for(self):
        # the loop the card names: only rows near the constant offset reach
        # Good by the full score, so the low-mass end -- where the trend departs
        # from the offset -- is cut out of the very points it is fitted on. On
        # the real labelled-nitrate files that left no acceptable trend at all;
        # on this clean synthetic set a trend survives, but its coverage starts
        # above m/z 150 and the centre at m/z 131 is held at that edge.
        cfg = PCfg.PassConfig()
        assert PC.calibrate(_backbone_ledger(with_pattern=False), cfg, log=lambda *a: None)
        if cfg.cal_b is not None:
            assert cfg.cal_mz_lo > 150
            assert PC.cal_center(cfg, 131.035) < _centre(131.035) - 0.1

    def test_the_pattern_only_backbone_fits_it(self):
        cfg = PCfg.PassConfig()
        assert PC.calibrate(_backbone_ledger(with_pattern=True), cfg, log=lambda *a: None)
        assert cfg.cal_b == pytest.approx(B_MDA, abs=0.03)
        assert cfg.cal_trend_n and cfg.cal_trend_n >= 20
        assert cfg.cal_mz_lo < 132
        assert PC.cal_center(cfg, 131.035) == pytest.approx(_centre(131.035), abs=0.1)


# --------------------------------------------------------------------------- the run
class TestTheRunRescoresAtTheTrend:
    """Offline assign.run on a spectrum whose masses follow the trend."""

    @staticmethod
    def _run(sid, **cfg_kw):
        log = []
        cfg = PCfg.PassConfig(height_cutoff_cps=100, **cfg_kw)
        res = A.run(sid, context="ambient-air", cfg=cfg, peaks=_peaks(_ACIDS),
                    adducts=["[M-H]-"], do_pass2=False, do_pass3=False, do_pass4=False,
                    do_pass5=False, do_pass_certified=False, use_cache=False,
                    log=log.append, scoring=ORBI)
        return res, log

    def test_an_accepted_trend_re_runs_the_file_at_the_per_line_centre(self):
        res, log = self._run("c42-run-trend")
        assert any("scoring centre -> the mass trend" in str(m) for m in log), log
        assert IO.scoring_trend("c42-run-trend") is not None
        assert res["pattern_scoring"]["mu_source"] == "trend"
        assert res["pattern_scoring"]["trend"]["b"] == pytest.approx(B_MDA, abs=0.03)
        led = res["ledger"]
        low = led[(led["role"] == "M0") & (led["mz"] < 160)]
        assert len(low) and (low["ion_score"] > 0.9).all(), low[["mz", "ion_score"]]

    def test_switched_off_the_file_is_scored_at_the_constant_offset(self):
        res, log = self._run("c42-run-flat", score_at_trend=False)
        assert not any("scoring centre -> the mass trend" in str(m) for m in log)
        assert IO.scoring_trend("c42-run-flat") is None
        assert res["pattern_scoring"]["mu_source"] != "trend"


# --------------------------------------------------------------------------- the snapshot
def _snapshot(**extra) -> dict:
    return {"sigma_ppm": 0.58, "mu_ppm": 0.175, "mz_tolerance_ppm": 5.0,
            "sigma_source": "fitted", "mu_source": "fitted", **extra}


class TestTheScoringTrendRecord:
    def setup_method(self):
        self.peaks = _peaks(["C5H8O4"])

    def test_a_set_trend_is_what_the_snapshot_records(self):
        IO.register_offline_sample("c42-snap", self.peaks, ["[M-H]-"], scoring=_snapshot())
        IO.scoring_for_sample(None, "c42-snap", self.peaks)
        IO._SCORING_TREND_INHERITED.discard("c42-snap")
        IO.set_scoring_trend("c42-snap", TREND)
        snap = IO.scoring_snapshot(None, "c42-snap", self.peaks)
        assert snap["mu_source"] == "trend"
        assert snap["trend"]["b"] == pytest.approx(B_MDA)

    def test_a_stand_in_inherits_the_measured_sample_s_trend_and_keeps_it(self):
        rec = IO._trend_record(TREND)
        IO.register_offline_sample("c42-arm", self.peaks, ["[M-H]-"], scoring=_snapshot(trend=rec))
        IO.scoring_for_sample(None, "c42-arm", self.peaks)
        assert IO.scoring_trend("c42-arm") == pytest.approx(TREND)
        IO.reset_scoring_trend("c42-arm")             # a new run of the stand-in
        assert IO.scoring_trend("c42-arm") is not None
        assert IO.scoring_snapshot(None, "c42-arm", self.peaks)["trend"]["b"] == pytest.approx(B_MDA)

    def test_re_registering_without_a_trend_forgets_it(self):
        rec = IO._trend_record(TREND)
        IO.register_offline_sample("c42-re", self.peaks, ["[M-H]-"], scoring=_snapshot(trend=rec))
        IO.scoring_for_sample(None, "c42-re", self.peaks)
        IO.register_offline_sample("c42-re", self.peaks, ["[M-H]-"], scoring=_snapshot())
        IO.scoring_for_sample(None, "c42-re", self.peaks)
        assert IO.scoring_trend("c42-re") is None

    def test_a_run_s_own_trend_is_forgotten_by_the_next_run(self):
        IO.register_offline_sample("c42-own", self.peaks, ["[M-H]-"], scoring=_snapshot())
        IO.scoring_for_sample(None, "c42-own", self.peaks)
        IO.set_scoring_trend("c42-own", TREND)
        IO.reset_scoring_trend("c42-own")
        assert IO.scoring_trend("c42-own") is None
        assert "trend" not in IO.scoring_snapshot(None, "c42-own", self.peaks)


# --------------------------------------------------------------------------- C42(c)
def _one_row(*, stamp_density):
    led = L.new_ledger(pd.DataFrame({"peak_id": ["u"], "mz": [200.1], "height": [1e4]}))
    L.commit_assignment(led, "u", neutral_formula="C8H12O3", adduct="[M+Br]-",
                        ion_formula="C8H12O3.Br-", ion_score=0.9, compound_score=0.9,
                        ppm_error=0.1, pass_no=1, method="cheminfo+grid",
                        confidence="Good", commentary="clean")
    if stamp_density is not None:
        led["degeneracy_density"] = pd.Series([stamp_density], index=led.index, dtype="object")
        led["degeneracy_note"] = pd.Series(["unique within ±3σ calibrated mass window"],
                                           index=led.index, dtype="object")
    return T.compute_tiers(led).set_index("peak_id")


def test_unique_in_the_calibrated_window_needs_the_audit_to_have_run():
    audited = _one_row(stamp_density=1)
    assert "unique formula in the calibrated window" in audited.at["u", "tier_reason"]
    skipped = _one_row(stamp_density=None)
    reason = skipped.at["u", "tier_reason"]
    assert "unique formula in the calibrated window" not in reason
    assert "degeneracy not measured" in reason


# --------------------------------------------------------------------------- C43
@pytest.mark.parametrize("adducts,halogen", [
    (["[M+NO3]-", "[M-H]-", "[M+Br]-"], "Br"),         # the mixed Br-/NO3- TOF
    (["[M+Br]-", "[M+HBr+Br]-"], "Br"),
    (["[M+NO3]-", "[M+^NO3]-", "[M-H]-"], None),         # 15N nitrate
    (["[M+I]-", "[M-H]-"], "I"),
    (["[M+Cl]-"], "Cl"),
    ([], None),
])
def test_the_reagent_halogen_is_named_by_the_declared_channels(adducts, halogen):
    assert EV.channel_halogen(adducts) == halogen


def test_the_levels_read_the_given_halogen_instead_of_counting(monkeypatch):
    led = _backbone_ledger(with_pattern=True)
    calls = []
    real = EV.detect_reagent_halogen
    monkeypatch.setattr(EV, "detect_reagent_halogen", lambda m0: calls.append(1) or real(m0))
    # the evidence scale of peaky 0.10.0: the per-file reading the halogen
    # feeds is the merge vote's private class (evidence.vote_classes)
    EV.vote_classes(led, halogen="Br")
    assert not calls
    EV.vote_classes(led)
    assert calls


# --------------------------------------------------------------------------- refute round 1 (0300a05)
class TestTheTrendStepRule:
    T1 = MC.MassTrend(A_PPM, B_MDA, 0.25, 90, 145.0, 497.0)        # misses the lowest masses
    T2 = MC.MassTrend(A_PPM, B_MDA, 0.25, 96, 131.0, 497.0)        # the re-run's refit
    T3 = MC.MassTrend(A_PPM + 0.01, B_MDA, 0.25, 96, 131.0, 497.0)  # agrees with T2

    def test_no_trend_fitted_and_none_scored_changes_nothing(self):
        assert A._trend_step(None, None, local=True, reruns=0) == "done"

    def test_a_first_accepted_trend_re_runs(self):
        assert A._trend_step(None, self.T1, local=True, reruns=0) == "rerun"

    def test_a_refit_that_moved_re_runs_again(self):
        # +0.64 vs +0.77 ppm at m/z 131: the first fit held the centre at its edge
        assert MC.trend_shift(self.T1, self.T2) > A.TREND_AGREE_PPM
        assert A._trend_step(self.T1, self.T2, local=True, reruns=1) == "rerun"

    def test_an_agreeing_refit_stops(self):
        assert A._trend_step(self.T2, self.T3, local=True, reruns=2) == "done"
        assert A._trend_step(self.T2, self.T3, local=True, reruns=1) == "done"

    def test_the_re_run_budget_stops_it(self):
        assert A._trend_step(self.T1, self.T2, local=True,
                             reruns=A.MAX_TREND_RERUNS) == "done"

    def test_a_rejected_refit_leaves_the_gates_at_the_scoring_trend(self):
        assert A._trend_step(self.T2, None, local=True, reruns=1) == "keep_gates"

    def test_the_network_scorer_never_re_runs(self):
        assert A._trend_step(None, self.T1, local=False, reruns=0) == "network"


def test_the_vectorised_centre_is_the_scalar_one():
    mz = np.array([61.0, 131.035, 250.0, 900.0])
    got = MC.centre_array(A_PPM, B_MDA, mz, 130.0, 700.0)
    want = [MC.centre(A_PPM, B_MDA, m, 130.0, 700.0) for m in mz]
    assert got == pytest.approx(want)


class TestTheRunAfterTheRefute:
    @staticmethod
    def _run(sid, adducts=("[M-H]-",), **cfg_kw):
        log = []
        cfg = PCfg.PassConfig(height_cutoff_cps=100, **cfg_kw)
        res = A.run(sid, context="ambient-air", cfg=cfg, peaks=_peaks(_ACIDS),
                    adducts=list(adducts), do_pass2=False, do_pass3=False, do_pass4=False,
                    do_pass5=False, do_pass_certified=False, use_cache=False,
                    log=log.append, scoring=ORBI)
        return res, [str(m) for m in log]

    def test_the_pattern_only_score_reaches_the_committed_rows(self):
        res, _ = self._run("c42r-carry")
        m0 = res["ledger"][res["ledger"]["role"] == "M0"]
        grid = m0[m0["method"].astype(str).str.startswith("cheminfo+grid")]
        assert len(grid) and grid["ion_score_massfree"].notna().all()

    def test_the_scoring_trend_converges_on_the_calibration_s_own(self):
        res, log = self._run("c42r-converge")
        assert sum("scoring centre -> the mass trend" in m for m in log) >= 1
        agree = [m for m in log if "scoring trend and calibration agree within" in m]
        assert agree and "re-run limit" not in agree[-1], log
        assert float(agree[-1].split("within ")[1].split(" ppm")[0]) <= A.TREND_AGREE_PPM
        assert res["pattern_scoring"]["trend"]["mz_lo"] < 132

    def test_a_run_forgets_its_sample_s_previous_trend_before_scoring(self, monkeypatch):
        # an offline sample is re-registered (which clears a trend) on every run;
        # a server sample is not, so the run itself must reset it first
        calls = []
        real_reset, real_resolve = IO.reset_scoring_trend, IO.scoring_for_sample
        monkeypatch.setattr(IO, "reset_scoring_trend",
                            lambda sid: calls.append(("reset", sid)) or real_reset(sid))
        monkeypatch.setattr(IO, "scoring_for_sample",
                            lambda *a, **k: calls.append(("resolve", a[1])) or real_resolve(*a, **k))
        self._run("c42r-twice")
        assert calls and calls[0] == ("reset", "c42r-twice"), calls[:3]

    def test_every_re_run_starts_from_the_uncalibrated_state(self, monkeypatch):
        seen = []
        real = A.passes.run_pass1

        def spy(client, sid, led, profile, pre, cfg, adducts, log=print):
            seen.append(cfg.cal_mu)
            return real(client, sid, led, profile, pre, cfg, adducts, log=log)

        monkeypatch.setattr(A.passes, "run_pass1", spy)
        self._run("c42r-fresh")
        assert len(seen) >= 2 and all(v is None for v in seen), seen

    def test_the_levels_read_the_declared_halogen(self, monkeypatch):
        got = []
        real = A.evidence.apply_levels

        def spy(*a, **kw):
            # the per-file stage hands the halogen to the pair facts in its run inputs
            got.append(kw["run_inputs"].summary.get("reagent_halogen"))
            return real(*a, **kw)

        monkeypatch.setattr(A.evidence, "apply_levels", spy)
        br, _ = self._run("c42r-br", adducts=("[M-H]-", "[M+Br]-"))
        nohal, _ = self._run("c42r-nohal", adducts=("[M-H]-",))
        assert got == ["Br", None]
        # ... and the run's stats carry it: a batch's parent computes the file's
        # merge-vote class with the halogen this run read
        assert br["stats"]["reagent_halogen"] == "Br" and nohal["stats"]["reagent_halogen"] is None


def test_a_batch_worker_returns_what_it_scored_at():
    from peaky.batch import assign_batch as AB
    kw = dict(cfg=PCfg.PassConfig(height_cutoff_cps=100), peaks=_peaks(_ACIDS),
              adducts=["[M-H]-"], do_pass2=False, do_pass3=False, do_pass4=False,
              do_pass5=False, do_pass_certified=False, use_cache=False, scoring=ORBI)
    AB._worker_init("ambient-air", None, kw, None)
    out = AB._assign_one("c42r-worker")
    assert out["pattern_scoring"]["mu_source"] == "trend"
    assert out["pattern_scoring"]["trend"]["b"] == pytest.approx(B_MDA, abs=0.03)


def test_an_uncalibrated_file_says_so():
    reason = _one_row(stamp_density=None).at["u", "tier_reason"]
    assert "degeneracy not measured: file uncalibrated" in reason


def test_a_ledger_without_the_pattern_column_falls_back_to_the_full_score():
    led = _backbone_ledger(with_pattern=True).drop(columns=["ion_score_massfree"])
    cfg = PCfg.PassConfig()
    assert PC.calibrate(led, cfg, log=lambda *a: None) is not None


def test_the_pooled_levels_forward_the_given_halogen(monkeypatch):
    led = _backbone_ledger(with_pattern=True)
    calls = []
    real = EV.detect_reagent_halogen
    monkeypatch.setattr(EV, "detect_reagent_halogen", lambda m0: calls.append(1) or real(m0))
    from peaky.assignment.levels import source as SRC

    def pooled(summary):
        # the pooled pair facts of the evidence scale (the batch's source)
        src = EV.source_from_frames({"f1": led, "f2": led}, run_inputs=EV.RunInputs(summary=summary))
        return SRC.pair_facts(src)

    pooled({"reagent_halogen": None})
    assert not calls
    pooled({})
    assert calls
