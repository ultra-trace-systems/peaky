"""assignment.satellites: the diagnostic-line verdict on the ledger, and its two
callers' readings of it (pass 0's known lead, the tier engine)."""
import pandas as pd

from peaky.assignment import ledger as L
from peaky.assignment import satellites as SAT
from peaky.assignment.passes import directors as D
from peaky.assignment.passes.config import PassConfig
from peaky.chem import isotopes as ISO

BR = {"C": 6, "H": 5, "Br": 1, "O": 2}          # a bromo-organic: one 81Br line at +1.998


def _ledger(parent_h=1000.0, twin_h=None, extra=()):
    rows = [("P", 300.0, parent_h)]
    if twin_h is not None:
        rows.append(("T", 300.0 + ISO.D_81BR, twin_h))
    rows += list(extra)
    return L.new_ledger(pd.DataFrame(rows, columns=["peak_id", "mz", "height"]))


def test_a_line_the_file_could_show_and_did_not_refutes():
    v = SAT.twin_verdict(_ledger(), "P", BR, floor=10.0)
    assert v["kind"] == "refuted" and v["verdict"] == "refuted" and v["twin"] == "Br"
    assert "no 81Br line" in v["why"] and v["lines"][0]["status"] == "absent"
    assert v["lines"][0]["testable"] and v["lines"][0]["pred_h"] == 1000.0 * ISO.R_81BR_PER_BR


def test_a_line_predicted_under_four_times_the_floor_is_untestable_not_refuted():
    v = SAT.twin_verdict(_ledger(parent_h=100.0), "P", BR, floor=50.0)     # predicted 97 < 200
    assert v["kind"] == "untestable" and v["verdict"] == "deferred"
    assert "under 4x the 50-cps floor" in v["why"] and v["lines"][0]["status"] == "under"


def test_a_present_consistent_line_supports_and_a_present_low_line_refutes():
    ok = SAT.twin_verdict(_ledger(twin_h=950.0), "P", BR, floor=10.0)
    assert ok["kind"] == "supported" and ok["verdict"] == "deferred"
    assert ok["lines"][0]["status"] == "present" and abs(ok["lines"][0]["obs_ratio"] - 0.95) < 1e-9
    low = SAT.twin_verdict(_ledger(twin_h=300.0), "P", BR, floor=10.0)     # 0.30 < 0.6 x 0.97
    assert low["kind"] == "refuted" and "under 0.6x" in low["why"] and low["lines"][0]["status"] == "low"
    # present under the multiple: support with the ratio censored, never a refutation
    cens = SAT.twin_verdict(_ledger(parent_h=100.0, twin_h=30.0), "P", BR, floor=50.0)
    assert cens["kind"] == "supported" and cens["lines"][0]["status"] == "present-censored"


def test_the_ion_count_predicts_the_line_and_the_neutral_picks_the_element():
    # a Cl-neutral's 37Cl is tested even when the composition also carries S
    v = SAT.twin_verdict(_ledger(), "P", {"C": 4, "H": 3, "Cl": 2, "S": 1, "O": 2}, floor=10.0)
    assert v["twin"] == "Cl" and v["lines"][0]["pred_ratio"] == 2 * ISO.R_37CL_PER_CL
    s = SAT.twin_verdict(_ledger(), "P", {"C": 4, "H": 3, "Cl": 2, "S": 1, "O": 2}, floor=10.0, element="S")
    assert s["twin"] == "S" and s["lines"][0]["label"] == "34S"
    assert SAT.twin_element({"C": 5, "H": 9, "P": 1, "O": 4}) is None
    assert SAT.twin_element({"Si": 2, "S": 1}, elements=SAT.TIER_ELEMENTS) == "S"


def test_no_floor_no_height_no_twin_and_a_masked_window_are_all_untestable():
    assert SAT.twin_verdict(_ledger(), "P", BR, floor=None)["kind"] == "untestable"
    assert SAT.twin_verdict(_ledger(parent_h=float("nan")), "P", BR, floor=10.0)["summary"] == "no parent height"
    assert SAT.twin_verdict(_ledger(), "P", {"C": 3, "F": 7, "O": 2}, floor=10.0)["twin"] is None
    m = SAT.twin_verdict(_ledger(), "P", BR, floor=10.0, masked_by="the reagent adduct's 81Br line ...")
    assert m["kind"] == "untestable" and m["why"].startswith("the reagent adduct's")


def test_the_reagent_halogen_masks_the_neutrals_m_plus_2_window():
    neutral = {"C": 5, "H": 10, "S": 1, "O": 3}
    assert SAT.reagent_masks("S", neutral, dict(neutral, Br=1)) is not None     # [M+Br]-
    assert SAT.reagent_masks("S", neutral, dict(neutral)) is None               # [M+NO3]-
    assert SAT.reagent_masks("Br", {"C": 6, "H": 5, "Br": 1}, {"C": 6, "H": 5, "Br": 2}) is not None
    assert SAT.reagent_masks("Si", {"Si": 2}, {"Si": 2, "Br": 1}) is None       # not a tier element


def test_peak_near_finds_the_closest_peak_within_the_window_or_none():
    mz = pd.Series([300.0, 300.004, 310.0], index=["a", "b", "c"])
    assert SAT.peak_near(mz, 300.0041, ppm=15.0) == "b"
    assert SAT.peak_near(mz, 305.0, ppm=15.0) is None
    assert SAT.peak_near(pd.Series([], dtype=float), 300.0) is None


def test_pass_zeros_wrapper_returns_the_four_lead_fields_with_its_prefix():
    led = _ledger()
    cfg = PassConfig(height_cutoff_cps=10.0)
    v = D._twin_verdict(led, "P", BR, cfg)
    assert set(v) == {"verdict", "twin", "why", "summary"}
    assert v["verdict"] == "refuted" and v["why"].startswith("single channel; no 81Br line")
    # the constants pass 0 documents still resolve where they always did
    assert D.TWIN_REFUTE_X_FLOOR == SAT.TWIN_REFUTE_X_FLOOR == 4.0
    assert D.TWIN_MIN_FRAC == SAT.TWIN_MIN_FRAC == 0.6 and D._TWIN_LINES is SAT.TWIN_LINES
    # an unresolved gate is 'deferred', worded as before
    bare = D._twin_verdict(led, "P", BR, PassConfig())
    assert bare["verdict"] == "deferred" and bare["summary"] == "no resolved detection floor"
