"""scripts/scorecard.py -- the decoy arms bound the engine as a real file runs it.

A decoy arm's own commits are wrong readings, so its own backbone is no measure
of the instrument: every arm takes its file's CONTROL calibration
(`inherited_calibration`). The ledgers here are synthetic and small enough to
count on paper: a control backbone of 24 CHO rows at +1.0 ppm (calibrates), an
arm backbone of 5 rows (too few to calibrate on its own).

Run: pytest tests/test_scorecard_decoy_arms.py -q
"""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import scorecard as SC  # noqa: E402
from peaky.assignment import passes as PA  # noqa: E402
from peaky.assignment import tiers as TI  # noqa: E402
from test_scorecard import write_run  # noqa: E402


def backbone(n: int, ppm: float, start: int = 0) -> pd.DataFrame:
    """`n` committed CHO M0 rows at `ppm` (+-0.05 ppm alternating), each with an observed 13C line."""
    rows = []
    for i in range(start, start + n):
        c = 6 + i % 12
        rows.append(dict(peak_id=f"p{i}", role="M0", mz=100.0 + 7.0 * i, height=1000.0 - i,
                         neutral_formula=f"C{c}H{2 * c - 2}O{2 + i % 5}", adduct="[M-H]-", tier="Assigned",
                         confidence="High", ion_score=0.95, ion_score_massfree=0.95,
                         ppm_error=ppm + (0.05 if i % 2 else -0.05),
                         isotopologues=json.dumps([{"label": "13C", "score": 0.9, "peak_id": f"i{i}"}]),
                         parent_peak_id=None))
        rows.append(dict(peak_id=f"i{i}", role="iso_child", mz=101.0 + 7.0 * i, height=60.0, neutral_formula=None,
                         adduct=None, tier=None, confidence=None, ion_score=None, ion_score_massfree=None,
                         ppm_error=None, isotopologues=None, parent_peak_id=f"p{i}"))
    return pd.DataFrame(rows)


CONTROL = backbone(24, 1.0)
ARM = backbone(5, -2.0, start=100)


def test_an_arm_takes_the_control_calibration_and_gives_it_back():
    own_pass, own_tier = PA.calibrate, TI._calibrate
    cfg = PA.PassConfig()
    m0 = ARM[ARM.role == "M0"]
    kids = ARM.loc[ARM.role == "iso_child", "parent_peak_id"].value_counts()
    # on its own the arm is too small: both stages leave it uncalibrated
    assert PA.calibrate(ARM, PA.PassConfig(), log=lambda *a: None) is None and TI._calibrate(m0, kids) is None
    with SC.inherited_calibration(CONTROL) as inherited:
        assert inherited is True
        mu, sigma, n = PA.calibrate(ARM, cfg, log=lambda *a: None)
        cal = TI._calibrate(m0, kids)
        cal_floor = TI._calibrate(m0, kids, abs_floor_mda=0.5)
    assert n == 24 and mu == pytest.approx(1.0, abs=0.06) and cfg.cal_mu == pytest.approx(mu)
    assert cal is not None and cal[0] == pytest.approx(1.0, abs=0.06) and cal_floor.abs_floor_mda == 0.5
    # restored on exit, and on an exception inside the block
    assert PA.calibrate is own_pass and TI._calibrate is own_tier
    with pytest.raises(RuntimeError):
        with SC.inherited_calibration(CONTROL):
            raise RuntimeError("engine failure")
    assert PA.calibrate is own_pass and TI._calibrate is own_tier
    # no control ledger: nothing to inherit, the arm calibrates on its own
    with SC.inherited_calibration(None) as inherited:
        assert inherited is False and PA.calibrate is own_pass


def _fake_engine(monkeypatch, fail_control=False):
    """A fake offline engine that records, per arm, whether the calibration it would read is the control's."""
    seen = {}

    def fake(run_, peaks, sample_id, adducts, log=lambda *a: None, scoring=None):
        arm = sample_id.rsplit("-", 1)[1]
        seen[arm] = PA.calibrate is not SC_OWN[0]
        if arm == "control" and fail_control:
            raise ValueError("control failed")
        return CONTROL.copy() if arm == "control" else ARM.copy()

    SC_OWN = [PA.calibrate]
    monkeypatch.setattr(SC, "run_engine_offline", fake)
    monkeypatch.setattr(SC, "level_arm", lambda *a, **k: pd.DataFrame())
    return seen


def test_every_decoy_arm_runs_at_its_files_control_calibration(tmp_path, monkeypatch):
    run = SC.load_run(str(write_run(tmp_path / "out")))
    seen = _fake_engine(monkeypatch)
    save = tmp_path / "kept"
    dc = SC.decoy(run, "both", 0.35, 1, save_dir=str(save))
    assert seen == {"control": False, "shift": True, "adducts": True}
    assert dc["calibration"] == {"s1": "control"} and SC.calibration_summary(dc) == "control"
    assert json.loads((save / "manifest.json").read_text())["calibration"] == {"s1": "control"}
    card = SC.build_card(run, rosters=SC.load_rosters(), board=[], log=lambda *a: None)
    card["decoy"] = dc
    assert SC.board_row(card)["decoy_calibration"] == "control"
    assert "`s1` its file's control arm (inherited)" in SC.render_md(card)


def test_an_arm_without_a_control_ledger_says_it_calibrated_on_its_own(tmp_path, monkeypatch):
    run = SC.load_run(str(write_run(tmp_path / "out")))
    seen = _fake_engine(monkeypatch, fail_control=True)
    dc = SC.decoy(run, "shift", 0.35, 1)
    assert dc["control"] == {"error": "ValueError: control failed"} and seen["shift"] is False
    assert dc["calibration"] == {"s1": "own"}
