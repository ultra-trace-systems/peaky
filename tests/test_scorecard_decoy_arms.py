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


# --------------------------------------------------------------------------- the populated-defect ppm-shift arm
SNAP = {"score_version": 2, "sigma_ppm": 0.5, "mu_ppm": 0.1, "sigma_source": "fitted", "mu_source": "fitted",
        "fitted_anchors": 60, "mz_tolerance_ppm": 5.0, "abundance_floor": 0.01, "instrument_type": "orbi",
        "has_signal_to_noise": True}


def orbi_run(tmp_path, window=5.0):
    """The fixture run with an Orbitrap scoring snapshot (a `window` ppm match window) for each file."""
    rd = write_run(tmp_path / "out")
    summ = json.loads((rd / "batch_summary.json").read_text())
    summ["pattern_scoring"] = {sid: dict(SNAP, mz_tolerance_ppm=window) for sid in ("s1", "s2")}
    (rd / "batch_summary.json").write_text(json.dumps(summ))
    return SC.load_run(str(rd))


def committed(mzs, prefix="a") -> pd.DataFrame:
    """Assigned M0 rows at the given m/z (one CHO reading each)."""
    cols = ["peak_id", "role", "tier", "mz", "height", "neutral_formula", "adduct", "method", "ion_formula", "confidence"]
    return pd.DataFrame([dict(peak_id=f"{prefix}{i}", role="M0", tier="Assigned", mz=mz, height=100.0 - i,
                              neutral_formula=f"C{6 + i}H{8 + 2 * i}O4", adduct="[M-H]-", method="cheminfo+grid",
                              ion_formula=f"C{6 + i}H{7 + 2 * i}O4-", confidence="High")
                         for i, mz in enumerate(mzs)], columns=cols)


ARMS = {"control": [120.0, 180.0, 260.0, 420.0], "shift": [], "ppmp9": [180.0], "ppmm9": [260.0, 420.0],
        "adducts": []}


def fake_arms(monkeypatch, arms=ARMS):
    seen = {}

    def fake(run_, peaks, sample_id, adducts, log=lambda *a: None, scoring=None):
        arm = sample_id.rsplit("-", 1)[1]
        seen[arm] = {"mz": peaks["mz"].to_numpy().copy(), "inherited": PA.calibrate is not own[0],
                     "window": (scoring or {}).get("mz_tolerance_ppm")}
        return committed(arms[arm], prefix=arm)

    own = [PA.calibrate]
    monkeypatch.setattr(SC, "run_engine_offline", fake)
    monkeypatch.setattr(SC, "level_arm", lambda *a, **k: pd.DataFrame())
    return seen


def test_the_ppm_arm_scales_every_mz_and_names_itself():
    peaks = pd.DataFrame({"peak_id": ["a", "b"], "mz": [100.0, 400.0], "height": [1.0, 2.0]})
    up = SC.decoy_ppm_peaks(peaks, 9.0)
    assert up["mz"].tolist() == pytest.approx([100.0009, 400.0036]) and up["peak_id"].tolist() == ["a_decoy", "b_decoy"]
    assert peaks["mz"].tolist() == [100.0, 400.0]                                    # the input is left alone
    assert SC.ppm_arm(9) == "ppmp9" and SC.ppm_arm(-9.0) == "ppmm9" and SC.ppm_arm(4.5) == "ppmp4d5"
    assert SC.parse_ppm_list(None) == list(SC.DECOY_PPM) == [9.0, -9.0]
    assert SC.parse_ppm_list("6,-6, 12") == [6.0, -6.0, 12.0] and SC.parse_ppm_list("none") == [] == SC.parse_ppm_list("0")
    # kept-ledger names must stay readable by the level layer's arm-file pattern
    from peaky.assignment.levels import source as SRC
    assert SRC.ARM_FILE_RE.match(f"s1__{SC.ppm_arm(-4.5)}.csv.gz").group("arm") == "ppmm4d5"


def test_a_ppm_shift_runs_only_outside_the_files_match_window(tmp_path, monkeypatch):
    run = orbi_run(tmp_path)
    assert SC.arm_window_ppm(run, "s1") == 5.0 and SC.decoy_instrument(run, "s1") == "orbi"
    seen = fake_arms(monkeypatch)
    dc = SC.decoy(run, "ppm", 0.35, 1, ppm_k=[9.0, -9.0, 4.0])
    assert set(seen) == {"control", "ppmp9", "ppmm9"} and dc["shift"] is None and dc["adducts"] is None
    assert dc["ppm_skipped"] == {"ppmp4": {"s1": "+4 ppm is inside the 5 ppm match window: the true formula stays "
                                                 "in reach, so the arm is no decoy"}}
    ctrl_mz = seen["control"]["mz"]
    assert seen["ppmp9"]["mz"] == pytest.approx(ctrl_mz * (1 + 9e-6)) and seen["ppmm9"]["mz"] == pytest.approx(ctrl_mz * (1 - 9e-6))
    assert seen["ppmp9"]["inherited"] and seen["ppmm9"]["inherited"] and not seen["control"]["inherited"]
    # a file without a snapshot is scored at the class fallback's 15 ppm window: +-9 ppm is no decoy there
    plain = SC.load_run(str(write_run(tmp_path / "plain")))
    assert SC.arm_window_ppm(plain, "s1") == 15.0
    fake_arms(monkeypatch)
    dc2 = SC.decoy(plain, "ppm", 0.35, 1, ppm_k=[9.0, -9.0])
    assert dc2["ppm"] is None and dc2["ppm_arms"] == {} and set(dc2["ppm_skipped"]) == {"ppmp9", "ppmm9"}
    assert dc2["headline"] is None


def test_the_ppm_arms_pool_against_the_control_once_per_arm_and_lead_the_headline(tmp_path, monkeypatch):
    run = orbi_run(tmp_path)
    fake_arms(monkeypatch)
    dc = SC.decoy(run, "both", 0.35, 1, ppm_k=[9.0, -9.0])
    ctrl, pp = dc["control"], dc["ppm"]
    assert ctrl["assigned"] == 4 and ctrl["assigned_lt_350"] == 3 and ctrl["assigned_ge_350"] == 1
    p9, m9 = dc["ppm_arms"]["ppmp9"], dc["ppm_arms"]["ppmm9"]
    assert (p9["k"], p9["assigned"], p9["assigned_rate"]) == (9.0, 1, 25.0)
    assert (m9["assigned_lt_350"], m9["assigned_ge_350"], m9["assigned_ge_350_rate"]) == (1, 1, 100.0)
    # pooled: 3 decoy Assigned against the control counted twice (8; 6 below 350, 2 at or above)
    assert pp["assigned"] == 3 and pp["n_arms"] == 2 and pp["control"]["assigned"] == 8
    assert pp["assigned_rate"] == pytest.approx(37.5) and pp["assigned_lt_350_rate"] == pytest.approx(100 * 2 / 6)
    assert pp["assigned_ge_350_rate"] == pytest.approx(50.0)
    # the 0.35 Da arm proposes nothing here; the headline quotes the ppm arms
    assert dc["shift"]["assigned"] == 0 and dc["shift"]["assigned_lt_350_rate"] == 0.0
    hl = dc["headline"]
    assert hl["arm"] == "ppm" and hl["rate_lt_350"] == pytest.approx(100 * 2 / 6) and "+9/-9 ppm" in hl["label"]
    assert (hl["assigned_lt_350"], hl["control_assigned_lt_350"]) == (2, 6)
    # per 50-Da bin: the control, the Da arm, and the ppm arms against the control counted twice
    assert dc["bins"] == [
        {"bin": "100-150", "control": 1, "shift": 0, "shift_rate": 0.0, "ppm": 0, "ppm_control": 2, "ppm_rate": 0.0},
        {"bin": "150-200", "control": 1, "shift": 0, "shift_rate": 0.0, "ppm": 1, "ppm_control": 2, "ppm_rate": 50.0},
        {"bin": "250-300", "control": 1, "shift": 0, "shift_rate": 0.0, "ppm": 1, "ppm_control": 2, "ppm_rate": 50.0},
        {"bin": "400-450", "control": 1, "shift": 0, "shift_rate": 0.0, "ppm": 1, "ppm_control": 2, "ppm_rate": 50.0}]
    # the card: the headline line, the Da arm marked blind on an Orbitrap, a row per arm, the bins
    card = SC.build_card(run, rosters=SC.load_rosters(), board=[], log=lambda *a: None)
    card["decoy"] = dc
    row = SC.board_row(card)
    assert row["decoy_headline_arm"] == "ppm" and row["decoy_headline_lt_350_rate"] == pytest.approx(100 * 2 / 6)
    assert row["decoy_ppm_rate"] == pytest.approx(37.5) and row["decoy_ppm_k"] == [9.0, -9.0]
    assert row["decoy_ppm_control_assigned"] == 8 and row["decoy_shift_lt_350_rate"] == 0.0
    md = SC.render_md(card)
    assert "**Shift decoy below m/z 350** (ppm shift +9/-9 ppm, pooled" in md and "**33.3 %**" in md
    assert "| shift +0.35 Da (blind below ~m/z 350 on an Orbitrap) |" in md
    assert "| shift +9 ppm |" in md and "| shift -9 ppm |" in md and "| ppm shifts pooled |" in md
    assert "| 150-200 | 1 | 0 | 0.0 | 1 | 2 | 50.0 |" in md
    page = SC.render_html([dict(card, row=row)], [row])
    assert "ppm shifts pooled" in page and "150-200" in page


def test_without_a_ppm_arm_the_headline_is_the_da_arm(tmp_path, monkeypatch):
    run = orbi_run(tmp_path)
    fake_arms(monkeypatch, dict(ARMS, shift=[300.0]))
    dc = SC.decoy(run, "shift", 0.35, 1, ppm_k=[])
    hl = dc["headline"]
    assert hl["arm"] == "shift" and hl["label"] == "shift +0.35 Da (blind below ~m/z 350 on an Orbitrap)"
    assert hl["rate_lt_350"] == pytest.approx(100 / 3) and dc["ppm"] is None and "ppm" not in {b for r in dc["bins"] for b in r}


def test_kept_ppm_arm_ledgers_recount_to_the_same_numbers(tmp_path, monkeypatch):
    run = orbi_run(tmp_path)
    fake_arms(monkeypatch)
    kept = tmp_path / "kept"
    first = SC.decoy(run, "both", 0.35, 1, save_dir=str(kept), ppm_k=[9.0, -9.0, 4.0])
    names = sorted(p.name for p in kept.iterdir())
    assert names == ["manifest.json", "s1__adducts.csv.gz", "s1__control.csv.gz", "s1__ppmm9.csv.gz",
                     "s1__ppmp9.csv.gz", "s1__shift.csv.gz"]
    man = json.loads((kept / "manifest.json").read_text())
    assert man["ppm_k"] == [9.0, -9.0, 4.0] and list(man["ppm_skipped"]) == ["ppmp4"]
    monkeypatch.setattr(SC, "run_engine_offline", lambda *a, **k: (_ for _ in ()).throw(AssertionError("kept ledgers")))
    again = SC.decoy(run, "both", 0.5, 1, ledgers_dir=str(kept), ppm_k=[6.0])     # the manifest wins
    assert again["ppm_k"] == [9.0, -9.0, 4.0] and again["ppm_skipped"] == first["ppm_skipped"]
    for key in ("ppm", "headline", "bins"):
        assert again[key] == first[key], key
    assert again["ppm_arms"] == first["ppm_arms"]


# --------------------------------------------------------------------------- the wrong-adducts arm at the ion level
def reading(pid, neutral, adduct, mz, tier="Assigned", height=100.0):
    return dict(peak_id=pid, role="M0", tier=tier, mz=mz, height=height, neutral_formula=neutral, adduct=adduct,
                method="cheminfo+grid", ion_formula=None, confidence="High")


def test_a_runs_own_channel_is_never_a_wrong_adduct(tmp_path):
    run = SC.load_run(str(write_run(tmp_path / "out")))
    assert SC.wrong_adducts_for(run) == ["[M+Cl]-", "[M+I]-"]                      # a nitrate run reads neither
    # an iodide run reads [M+I]-: only [M+Cl]- is wrong for it
    iodide = SC.load_run(str(write_run(tmp_path / "iod")))
    iodide.summary["reagent"] = "I"
    iodide.profile = SC.profile_for("I")
    assert "[M+I]-" in SC.own_adducts(iodide) and SC.wrong_adducts_for(iodide) == ["[M+Cl]-"]
    # a side channel the run opened and committed on is its own, wherever it shows (a ledger, a summary list)
    pf = run.per_file.copy()
    pf.loc[pf.index[0], "adduct"] = "[M+Cl]-"
    opened = SC.Run(run.path, run.ledger, dict(run.summary, side_channels=["[M+I]-"]), run.manifest, run.ts, pf,
                    profile=run.profile)
    assert SC.wrong_adducts_for(opened) == []


def test_the_adducts_arm_is_not_run_when_every_wrong_adduct_is_the_runs_own(tmp_path, monkeypatch):
    run = SC.load_run(str(write_run(tmp_path / "out")))
    run.summary["side_channels"] = ["[M+Cl]-", "[M+I]-"]
    seen = fake_arms(monkeypatch)
    dc = SC.decoy(run, "adducts", 0.35, 1)
    assert "adducts" not in seen and dc["adducts"] is None and dc["wrong_adducts"] == []
    assert dc["adducts_skipped"] == "every wrong-adduct candidate ['[M+Cl]-', '[M+I]-'] is one of the run's own channels"


def test_ion_level_counts_tell_a_resplit_of_the_controls_ion_from_a_new_ion():
    control = pd.DataFrame([
        reading("p1", "C6H11ClO4", "[M-H]-", 181.03),          # ion C6H10ClO4-
        reading("p2", "C7H10O5", "[M-H]-", 173.05),
        reading("p4", "C5H8O4", "[M-H]-", 131.03, tier="Candidate"),
    ])
    arm = pd.DataFrame([
        reading("p1", "C6H10O4", "[M+Cl]-", 181.03, height=500.0),   # the same ion, split as a chloride cluster
        reading("p2", "C4H10O3", "[M+Cl]-", 173.05, height=400.0),   # another ion on a read peak: new
        reading("p3", "C9H8O2", "[M+I]-", 274.96, height=300.0),     # a peak the control left unexplained: new
        reading("p4", "C5H8O4", "[M+Cl]-", 167.0, height=200.0),      # same peak id, different ion: new
        reading("p5", "C3H4O2", "[M+Cl]-", 107.0, tier="Candidate"),  # not Assigned: not counted
    ])
    assert SC.ion_key("C6H10O4", "[M+Cl]-") == SC.ion_key("C6H11ClO4", "[M-H]-")
    c = SC.ion_level_counts(arm, control)
    assert (c["assigned_same_ion"], c["assigned_new_ion"], c["assigned_new_ion_lt_350"]) == (1, 3, 3)
    assert c["new_ion_examples"][0] == "C4H10O3 [M+Cl]- @ 173.0500"
    # no control ledger: every Assigned row is a new ion
    assert SC.ion_level_counts(arm, None)["assigned_new_ion"] == 4


def test_the_adducts_arm_reports_its_ion_level_rate_beside_the_reading_level(tmp_path, monkeypatch):
    run = SC.load_run(str(write_run(tmp_path / "out")))
    ledgers = {
        "control": [reading("p1", "C6H11ClO4", "[M-H]-", 181.03), reading("p2", "C7H10O5", "[M-H]-", 173.05),
                    reading("p3", "C8H12O4", "[M-H]-", 171.07), reading("p4", "C10H16O3", "[M-H]-", 183.1)],
        "adducts": [reading("p1", "C6H10O4", "[M+Cl]-", 181.03), reading("p2", "C4H10O3", "[M+Cl]-", 173.05)],
    }
    monkeypatch.setattr(SC, "run_engine_offline", lambda run_, peaks, sample_id, adducts, log=None, scoring=None:
                        pd.DataFrame(ledgers[sample_id.rsplit("-", 1)[1]]))
    monkeypatch.setattr(SC, "level_arm", lambda *a, **k: pd.DataFrame())
    dc = SC.decoy(run, "adducts", 0.35, 1)
    ad = dc["adducts"]
    assert ad["assigned"] == 2 and ad["assigned_rate"] == pytest.approx(50.0)          # reading level: 2 of 4
    assert ad["assigned_same_ion"] == 1 and ad["assigned_new_ion"] == 1
    assert ad["new_ion_rate"] == pytest.approx(25.0) and ad["same_ion_share"] == pytest.approx(50.0)
    card = SC.build_card(run, rosters=SC.load_rosters(), board=[], log=lambda *a: None)
    card["decoy"] = dc
    row = SC.board_row(card)
    assert row["decoy_adducts_rate"] == pytest.approx(50.0) and row["decoy_adducts_new_ion_rate"] == pytest.approx(25.0)
    assert row["decoy_wrong_adducts"] == ["[M+Cl]-", "[M+I]-"]
    md = SC.render_md(card)
    assert "of 2 Assigned, 1 (50.0 %) carry the ion composition of the control's reading" in md
    assert "**1 new ions = 25.0 % of the control's Assigned**" in md and "reading-level 50.0 %" in md
