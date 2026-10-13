"""Fluorine by evidence (assignment/fluorine.py): the CF2 ladder lifts the two
fluorine refusals, the 14N twin of the 15N alias and the 13C line refute it,
the fluorinated family reaches F25, and the family's sub-mDa aliases answer to
their 13C line in arbitration.

Every ledger here is synthetic: a partially fluorinated acid series
C(n)H5F(2n-5)O2 read on [M-H]- (F < 2H, F >= 4, so both refusals apply),
committed or left as picked lines.

Run: pytest tests/test_fluorine.py -q
"""
from __future__ import annotations

import pandas as pd
import pytest

from peaky.assignment import cleanup as CL
from peaky.assignment import degeneracy as DG
from peaky.assignment import fluorine as FL
from peaky.assignment import ledger as L
from peaky.assignment import passes as P
from peaky.assignment import plausibility as PL
from peaky.assignment import tiers as T
from peaky.assignment.passes import directors as DIR
from peaky.chem import chemistry as C
from peaky.chem import contexts as X
from peaky.chem import isotopes as ISO

ADD = "[M-H]-"
#: the series: neutral -> height
SERIES = {"C5H5F5O2": 1000.0, "C6H5F7O2": 800.0, "C7H5F9O2": 600.0, "C8H5F11O2": 400.0}


def _ion(nf: str, adduct: str = ADD) -> str:
    return C.format_formula(T._ion_counts(nf, adduct, labelled=True)) + "-"


def _mz(nf: str, adduct: str = ADD) -> float:
    return C.ion_mz(nf, adduct)


def _cfg(**kw) -> P.PassConfig:
    cfg = P.PassConfig(height_cutoff_cps=kw.pop("floor", 50.0), instrument_class=kw.pop("cls", "orbitrap"))
    cfg.cal_mu, cfg.cal_sigma = 0.0, kw.pop("sigma", 0.2)
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


def _ledger(lines: dict, committed: set, *, extra: list[tuple] = ()) -> pd.DataFrame:
    """A ledger of the series' lines (`lines`: neutral -> height) plus `extra`
    (peak_id, mz, height) lines; the neutrals in `committed` committed on [M-H]-."""
    rows = [dict(peak_id=nf, mz=_mz(nf), height=h) for nf, h in lines.items()]
    rows += [dict(peak_id=p, mz=m, height=h) for p, m, h in extra]
    led = L.new_ledger(pd.DataFrame(rows))
    for nf in committed:
        L.commit_assignment(led, nf, neutral_formula=nf, adduct=ADD, ion_formula=_ion(nf),
                            ion_score=0.95, compound_score=0.95, eff_score=0.75, eff_margin=None, tied=False,
                            ppm_error=0.1, pass_no=3, method="contaminant:fluorinated:gka",
                            confidence="High (fluorinated)", commentary="Pass 3 (contaminant:fluorinated:gka)")
    return led


def _row(led, pid):
    return led.loc[led["peak_id"] == pid].iloc[0]


# =========================================================================== the ladder
def test_a_ladder_counts_its_lines_and_its_committed_members():
    led = _ledger(SERIES, {"C6H5F7O2", "C7H5F9O2"})
    v = FL.verdicts(led, cfg=_cfg())
    lad = v["C6H5F7O2"]
    assert (lad.lines, lad.committed) == (4, 2)
    assert dict(lad.steps) == {-1: False, 1: True, 2: False}
    assert lad.lift and not lad.full
    # three committed members: the normal tier rules apply
    led = _ledger(SERIES, {"C6H5F7O2", "C7H5F9O2", "C8H5F11O2"})
    lad = FL.verdicts(led, cfg=_cfg())["C6H5F7O2"]
    assert lad.committed == 3 and lad.full


def test_a_committed_member_must_carry_the_ladders_own_composition():
    """A line at a CF2 step committed as some OTHER reading is a ladder line, not
    a committed member: the +1 and +2 lines read as a CHO ion leave the row with
    one committed member (itself), lifted but not full, at Candidate."""
    led = _ledger(SERIES, {"C6H5F7O2"})
    for nf in ("C7H5F9O2", "C8H5F11O2"):
        L.commit_assignment(led, nf, neutral_formula="C30H40O2", adduct=ADD, ion_formula="C30H39O2-",
                            ion_score=0.9, compound_score=0.9, ppm_error=0.3, pass_no=1, method="cheminfo+grid",
                            confidence="Good", commentary="Pass 1")
    lad = FL.verdicts(led, cfg=_cfg())["C6H5F7O2"]
    assert dict(lad.steps) == {-1: False, 1: False, 2: False}
    assert lad.committed == 1 and lad.lift and not lad.full
    tier, reason = _tier(led, "C6H5F7O2")
    assert tier == "Candidate" and "Candidate until 3" in reason


def test_a_lone_reading_and_a_one_neighbour_reading_are_not_lifted():
    led = _ledger({"C6H5F7O2": 800.0}, {"C6H5F7O2"})
    assert not FL.verdicts(led, cfg=_cfg())["C6H5F7O2"].lift
    led = _ledger({"C6H5F7O2": 800.0, "C7H5F9O2": 600.0}, {"C6H5F7O2"})
    lad = FL.verdicts(led, cfg=_cfg())["C6H5F7O2"]
    assert lad.lines == 2 and not lad.lift


def test_a_ladder_line_must_sit_at_the_step_and_at_a_consistent_height():
    lines = dict(SERIES)
    led = _ledger(lines, {"C6H5F7O2"})
    # the +1 line 3 ppm off its step (the exact-offset window is 1 ppm here) and the +2 line 100x brighter
    led.loc[led["peak_id"] == "C7H5F9O2", "mz"] += _mz("C7H5F9O2") * 3e-6
    led.loc[led["peak_id"] == "C8H5F11O2", "height"] = 8.0e4
    lad = FL.verdicts(led, cfg=_cfg())["C6H5F7O2"]
    assert dict(lad.steps) == {-1: False}
    assert not lad.lift
    # 0.5 ppm off is inside the window
    led2 = _ledger(lines, {"C6H5F7O2"})
    led2.loc[led2["peak_id"] == "C7H5F9O2", "mz"] += _mz("C7H5F9O2") * 0.5e-6
    assert 1 in dict(FL.verdicts(led2, cfg=_cfg())["C6H5F7O2"].steps)


def test_a_ladder_needs_an_adjacent_member():
    """Lines at +2 and +3 CF2 only (the +1 member missing) do not lift the
    refusal: one of the ladder lines must be a +/-1 CF2 neighbour."""
    lines = {"C6H5F7O2": 800.0, "C8H5F11O2": 400.0, "C9H5F13O2": 300.0}
    led = _ledger(lines, {"C6H5F7O2", "C8H5F11O2", "C9H5F13O2"})
    lad = FL.verdicts(led, cfg=_cfg())["C6H5F7O2"]
    assert dict(lad.steps) == {2: True, 3: True} and lad.lines == 3
    assert not lad.lift and not lad.full
    # the +1 member back: lifted
    led = _ledger(dict(lines, C7H5F9O2=600.0), {"C6H5F7O2", "C8H5F11O2", "C9H5F13O2"})
    assert FL.verdicts(led, cfg=_cfg())["C6H5F7O2"].full


def test_the_window_is_the_exact_offset_one_on_a_calibrated_orbitrap_only():
    assert FL.window_ppm(_cfg(sigma=0.2)) == pytest.approx(1.0)
    assert FL.window_ppm(_cfg(sigma=0.5)) == pytest.approx(2.0)
    assert FL.window_ppm(_cfg(cls="tof")) == pytest.approx(FL.LADDER_PPM)
    assert FL.window_ppm(None) == pytest.approx(FL.LADDER_PPM)


def test_every_ladder_member_carries_fluorine():
    # C3H6F2O2 - CF2 is C2H6O2, no fluorine left: a line there is no ladder member
    led = _ledger({"C3H6F2O2": 900.0, "C4H6F4O2": 700.0}, {"C3H6F2O2"},
                  extra=[("cho", _mz("C3H6F2O2") - FL.D_CF2, 800.0)])
    lad = FL.verdicts(led, cfg=_cfg())["C3H6F2O2"]
    assert dict(lad.steps) == {1: False}


def test_a_ledger_without_lines_has_no_ladder():
    led = pd.DataFrame([dict(role="M0", neutral_formula="C6H5F7O2", tier="Assigned")])
    assert FL.verdicts(led) == {}


# =========================================================================== the tier engine
def _tier(led, pid, cfg=None):
    t = T.compute_tiers(led, cfg=cfg or _cfg()).set_index("peak_id")
    return t.at[pid, "tier"], t.at[pid, "tier_reason"]


def test_the_partial_fluorine_cap_stands_without_a_ladder():
    led = _ledger({"C6H5F7O2": 800.0}, {"C6H5F7O2"})
    tier, reason = _tier(led, "C6H5F7O2")
    assert tier == "Candidate" and "below PFAS-like F/H coherence" in reason


def test_a_ladder_lifts_the_cap_to_a_candidate_with_its_own_reason():
    led = _ledger(SERIES, {"C6H5F7O2"})
    tier, reason = _tier(led, "C6H5F7O2")
    assert tier == "Candidate"
    assert "PFAS-like" not in reason and "CF2 ladder of 4 lines" in reason and "Candidate until 3" in reason


def test_three_committed_members_let_the_normal_tier_rules_apply():
    led = _ledger(SERIES, {"C6H5F7O2", "C7H5F9O2", "C8H5F11O2"})
    tier, reason = _tier(led, "C6H5F7O2")
    assert tier == "Assigned", reason
    assert "PFAS-like" not in reason


# =========================================================================== the F >= 4 demote
def test_the_demote_keeps_a_full_ladder_member_at_its_tier_without_a_flag():
    led = _ledger(SERIES, {"C6H5F7O2", "C7H5F9O2", "C8H5F11O2"})
    T.apply_tiers(led, cfg=_cfg())
    before = _row(led, "C6H5F7O2")["tier"]
    out = CL.demote_unconfirmed_fluorine(led, cfg=_cfg(), log=lambda *a: None)
    r = _row(led, "C6H5F7O2")
    assert r["tier"] == before == "Assigned"
    assert not bool(r["below_assignability"])
    assert "CF2 ladder" in r["commentary"] and "unconfirmed fluorine" not in r["commentary"]
    assert out == {"f_demoted": 0, "f_ladder": 3}


def test_the_demote_holds_a_lifted_member_at_candidate_without_the_flag():
    led = _ledger(SERIES, {"C6H5F7O2"})
    T.apply_tiers(led, cfg=_cfg())
    led.loc[led["peak_id"] == "C6H5F7O2", "tier"] = "Assigned"     # as if another rule had passed it
    CL.demote_unconfirmed_fluorine(led, cfg=_cfg(), log=lambda *a: None)
    r = _row(led, "C6H5F7O2")
    assert r["tier"] == "Candidate" and not bool(r["below_assignability"])
    assert "Candidate until 3" in str(r["tier_reason"])


def test_the_demote_refuses_a_reading_with_no_ladder_as_before():
    led = _ledger({"C6H5F7O2": 800.0}, {"C6H5F7O2"})
    T.apply_tiers(led, cfg=_cfg())
    out = CL.demote_unconfirmed_fluorine(led, cfg=_cfg(), log=lambda *a: None)
    r = _row(led, "C6H5F7O2")
    assert r["tier"] == "Candidate" and bool(r["below_assignability"])
    assert "unconfirmed fluorine" in r["commentary"]
    assert out == {"f_demoted": 1}


# =========================================================================== what refutes a ladder
def _labelled(led: pd.DataFrame) -> pd.DataFrame:
    """Add a committed [M+^NO3]- cluster: the ledger is a 15N-nitrate one."""
    nf = "C10H16O4"
    extra = L.new_ledger(pd.DataFrame([dict(peak_id="cl", mz=_mz(nf, "[M+^NO3]-"), height=5e4)]))
    led = pd.concat([led, extra], ignore_index=True)
    L.commit_assignment(led, "cl", neutral_formula=nf, adduct="[M+^NO3]-", ion_formula=_ion(nf, "[M+^NO3]-"),
                        ion_score=0.97, compound_score=0.97, ppm_error=0.0, pass_no=1, method="cheminfo+grid",
                        confidence="High", commentary="Pass 1")
    return led


@pytest.mark.parametrize("ratio, refuted", [(0.02, True), (0.012, True), (0.08, True), (0.2, False), (0.003, False)])
def test_the_14n_twin_of_the_15n_alias_refutes_an_n_free_f_reading(ratio, refuted):
    # floor 5: the alias's twin at the band's lower edge (0.5 f x 800 ~ 8 cps) is testable
    twin = ("twin", _mz("C6H5F7O2") - FL.D_15N, 800.0 * ratio)
    led = _labelled(_ledger(SERIES, {"C6H5F7O2", "C7H5F9O2", "C8H5F11O2"}, extra=[twin]))
    lad = FL.verdicts(led, cfg=_cfg(floor=5.0))["C6H5F7O2"]
    assert (lad.veto is not None) == refuted
    assert lad.alias_open is None                  # tested: refuted, or no twin in the band
    assert lad.full != refuted
    if refuted:
        assert "14N twin" in lad.veto


def test_an_untested_15n_alias_keeps_a_full_ladder_at_candidate():
    """On a 15N-nitrate ledger an N-free C>=2, F>=3 reading whose alias's 14N twin
    would read under the floor has its alias UNTESTED: lifted, never full."""
    led = _labelled(_ledger(SERIES, {"C6H5F7O2", "C7H5F9O2", "C8H5F11O2"}))
    lad = FL.verdicts(led, cfg=_cfg(floor=50.0))["C6H5F7O2"]     # 0.5 f x 800 ~ 8 cps < 50
    assert lad.veto is None and lad.lift and lad.committed == 3
    assert not lad.full and "under the floor" in lad.alias_open
    tier, reason = _tier(led, "C6H5F7O2", _cfg(floor=50.0))
    assert tier == "Candidate" and "15N alias" in reason and "untested" in reason
    T.apply_tiers(led, cfg=_cfg(floor=50.0))
    led.loc[led["peak_id"] == "C6H5F7O2", "tier"] = "Assigned"     # as if another rule had passed it
    CL.demote_unconfirmed_fluorine(led, cfg=_cfg(floor=50.0), log=lambda *a: None)
    r = _row(led, "C6H5F7O2")
    assert r["tier"] == "Candidate" and not bool(r["below_assignability"])
    # no cfg: no floor, nothing testable -- the alias stays open
    assert FL.verdicts(led)["C6H5F7O2"].alias_open
    # the same bright reading with its alias tested and absent takes the normal tier rules
    assert FL.verdicts(led, cfg=_cfg(floor=5.0))["C6H5F7O2"].full
    # an alias needs C >= 2 and F >= 3: C1/F2 readings have none to test
    assert FL._alias_twin({"C": 1, "H": 1, "F": 3}, 800.0, FL._Lines(led), 300.0, 1.0, 0.02, 50.0) == (None, None)
    assert FL._alias_twin({"C": 4, "H": 5, "F": 2}, 800.0, FL._Lines(led), 300.0, 1.0, 0.02, 50.0) == (None, None)


def test_the_hold_on_an_untested_alias_is_one_switch(monkeypatch):
    """ALIAS_UNTESTED_HOLDS off: the ladder alone decides (the alias stays named
    as open, but a full ladder takes the normal tier rules)."""
    led = _labelled(_ledger(SERIES, {"C6H5F7O2", "C7H5F9O2", "C8H5F11O2"}))
    assert not FL.verdicts(led, cfg=_cfg(floor=50.0))["C6H5F7O2"].full
    monkeypatch.setattr(FL, "ALIAS_UNTESTED_HOLDS", False)
    lad = FL.verdicts(led, cfg=_cfg(floor=50.0))["C6H5F7O2"]
    assert lad.alias_open and lad.full
    assert _tier(led, "C6H5F7O2", _cfg(floor=50.0))[0] == "Assigned"


def test_a_saturated_reading_has_no_15n_alias_to_test():
    """The C2F3 -> H2^NO4 swap lowers the DBE by one: a DBE-0 reading's alias has
    DBE -1 and cannot exist, so there is nothing to test -- a full ladder takes
    the normal tier rules on a 15N-nitrate ledger, and a line where the alias's
    14N twin would sit refutes nothing."""
    sat = {"C5H7F5O": 1000.0, "C6H7F7O": 800.0, "C7H7F9O": 600.0, "C8H7F11O": 400.0}
    assert C.dbe("C6H7F7O") == 0 and not FL.alias_admissible(C.parse_formula("C6H7F7O"))
    assert FL.alias_admissible(C.parse_formula("C6H5F7O2"))       # DBE 1: the alias (DBE 0) can exist
    assert FL.alias_admissible(None) and FL.alias_admissible({"C": 1, "H": 1, "F": 3})
    twin = ("twin", _mz("C6H7F7O") - FL.D_15N, 16.0)                # 0.02 of the line: in the band
    led = _labelled(_ledger(sat, {"C6H7F7O", "C7H7F9O", "C8H7F11O"}, extra=[twin]))
    lad = FL.verdicts(led, cfg=_cfg(floor=50.0))["C6H7F7O"]
    assert lad.veto is None and lad.alias_open is None and lad.full
    tier, _reason = _tier(led, "C6H7F7O", _cfg(floor=50.0))
    assert tier == "Assigned"
    # the DBE-1 series on the same ledger keeps its alias open
    led = _labelled(_ledger(SERIES, {"C6H5F7O2", "C7H5F9O2", "C8H5F11O2"}))
    assert FL.verdicts(led, cfg=_cfg(floor=50.0))["C6H5F7O2"].alias_open


def test_the_twin_is_read_only_on_a_15n_nitrate_ledger():
    twin = ("twin", _mz("C6H5F7O2") - FL.D_15N, 16.0)
    led = _ledger(SERIES, {"C6H5F7O2", "C7H5F9O2", "C8H5F11O2"}, extra=[twin])
    lad = FL.verdicts(led, cfg=_cfg())["C6H5F7O2"]
    assert lad.full and lad.alias_open is None
    # a 15N-AMMONIUM ledger ([M+^NH4]+) carries ^N but makes no C2F3 / H2^NO4 alias
    amm = L.new_ledger(pd.DataFrame([dict(peak_id="am", mz=_mz("C8H18O5", "[M+^NH4]+"), height=5e4)]))
    L.commit_assignment(amm, "am", neutral_formula="C8H18O5", adduct="[M+^NH4]+",
                        ion_formula=_ion("C8H18O5", "[M+^NH4]+"), ion_score=0.97, compound_score=0.97,
                        ppm_error=0.0, pass_no=1, method="cheminfo+grid", confidence="High", commentary="Pass 1")
    assert not FL.labelled_nitrate(amm) and FL.labelled_nitrate(_labelled(led))
    led = pd.concat([led, amm], ignore_index=True)
    lad = FL.verdicts(led, cfg=_cfg())["C6H5F7O2"]
    assert lad.full and lad.veto is None and lad.alias_open is None


def test_an_f_reading_that_carries_15n_has_its_alias_left_open():
    """[M+^NO3]- of an F neutral carries one ^N (its own twin ~ f), its alias two
    (~ 2f); the 14N share is not known well enough to tell f from 2f (the air's
    14N nitrate adds to the reagent's impurity), so no twin refutes it -- and
    nothing tests its alias: lifted, never full."""
    f = (1 - ISO.label_purity()) / ISO.label_purity()
    series = {nf: h for nf, h in SERIES.items()}
    rows = [dict(peak_id=nf, mz=_mz(nf, "[M+^NO3]-"), height=h) for nf, h in series.items()]
    for share in (1.0, 2.0, 4.0):
        led = L.new_ledger(pd.DataFrame(rows + [dict(peak_id="twin", height=800.0 * f * share,
                                                     mz=_mz("C6H5F7O2", "[M+^NO3]-") - FL.D_15N)]))
        for nf in ("C6H5F7O2", "C7H5F9O2", "C8H5F11O2"):
            L.commit_assignment(led, nf, neutral_formula=nf, adduct="[M+^NO3]-",
                                ion_formula=_ion(nf, "[M+^NO3]-"), ion_score=0.95, compound_score=0.95,
                                ppm_error=0.1, pass_no=3, method="contaminant:fluorinated:gka",
                                confidence="High (fluorinated)", commentary="Pass 3")
        lad = FL.verdicts(led, cfg=_cfg(floor=5.0))["C6H5F7O2"]
        assert lad.veto is None and lad.lift and not lad.full, (share, lad)
        assert "carries ^N1" in lad.alias_open
        tier, reason = _tier(led, "C6H5F7O2", _cfg(floor=5.0))
        assert tier == "Candidate" and "untested" in reason


@pytest.mark.parametrize("n", [3, 5, 8, 10, 14, 20])
def test_the_13c_line_cannot_split_the_15n_alias(n):
    """The C2F3 -> H2^NO4 swap moves two carbons: a perfect 13C line of the
    reading never refutes its alias (contradicts' tolerance is >= 2.5 C)."""
    h0 = 1e5
    for n_true, n_other in ((n, n - 2), (n - 2, n)):
        if n_true < 1 or n_other < 1:
            continue
        bad, _c = FL.IR.contradicts(None, n_other, h0=h0, h_sat=h0 * FL.IR.R13C_RAW * n_true)
        assert not bad


def test_a_13c_line_that_contradicts_the_carbon_count_refutes_the_ladder():
    # C8: its 13C line reads C3 (above the floor) -> refuted; without a floor nothing is testable
    h0 = 400.0
    led = _ledger(SERIES, {"C6H5F7O2", "C7H5F9O2", "C8H5F11O2"},
                  extra=[("c13", _mz("C8H5F11O2") + FL.IR.D13C, h0 * 0.0107 * 3)])
    L.attach_isotopologue(led, "c13", "C8H5F11O2", iso_label="13C")
    lad = FL.verdicts(led, cfg=_cfg(floor=5.0))["C8H5F11O2"]
    assert lad.veto is not None and "13C line reads ~C3 for C8" in lad.veto
    assert not lad.lift
    assert FL.verdicts(led, cfg=None)["C8H5F11O2"].veto is None
    # a line that agrees (C8) leaves the ladder standing
    led.loc[led["peak_id"] == "c13", "height"] = h0 * 0.0107 * 8
    assert FL.verdicts(led, cfg=_cfg(floor=5.0))["C8H5F11O2"].full


def test_the_13c_veto_reads_no_other_rows_m0():
    """The ladder veto is a refusal: it reads only the row's own 13C child or an
    unexplained line. A line at +1.00336 that another reading commits as its M0
    (a CHON fit here, reading ~C3 for the C8 row) refutes nothing."""
    h0 = 400.0
    led = _ledger(SERIES, {"C6H5F7O2", "C7H5F9O2", "C8H5F11O2"},
                  extra=[("c13", _mz("C8H5F11O2") + FL.IR.D13C, h0 * 0.0107 * 3)])
    assert FL.verdicts(led, cfg=_cfg(floor=5.0))["C8H5F11O2"].veto is not None     # unexplained: it refutes
    L.commit_assignment(led, "c13", neutral_formula="C9H7N3O6", adduct=ADD, ion_formula="C9H6N3O6-",
                        ion_score=0.8, compound_score=0.8, ppm_error=0.5, pass_no=1, method="cheminfo+grid",
                        confidence="Good", commentary="Pass 1")
    lad = FL.verdicts(led, cfg=_cfg(floor=5.0))["C8H5F11O2"]
    assert lad.veto is None and lad.full


def test_a_refuted_ladder_says_so_on_the_refusal():
    twin = ("twin", _mz("C6H5F7O2") - FL.D_15N, 16.0)
    led = _labelled(_ledger(SERIES, {"C6H5F7O2", "C7H5F9O2", "C8H5F11O2"}, extra=[twin]))
    T.apply_tiers(led, cfg=_cfg())
    CL.demote_unconfirmed_fluorine(led, cfg=_cfg(), log=lambda *a: None)
    r = _row(led, "C6H5F7O2")
    assert r["tier"] == "Candidate" and bool(r["below_assignability"])
    assert "does not lift the refusal" in r["commentary"] and "14N twin" in r["commentary"]
    assert "PFAS-like" in r["tier_reason"] and "14N twin" in r["tier_reason"]


# =========================================================================== the element budget
def test_the_budget_keeps_a_full_ladder_member_whose_steps_skip_a_member():
    """C6 with its +2 / +3 members committed and +1 only a picked line: no committed
    CF2 NEIGHBOUR (the step rule), but a full ladder."""
    series = dict(SERIES, C9H5F13O2=300.0)
    led = L.ensure_flags(_ledger(series, {"C6H5F7O2", "C8H5F11O2", "C9H5F13O2"}))
    out = PL.demote_off_budget(led, context="ambient-air", log=lambda *a: None, cfg=_cfg())
    r = _row(led, "C6H5F7O2")
    assert not bool(r["tentative_lead"])
    assert out["budget_demoted"] == 0 and out["budget_cf2_kept"] == 3
    # without its ladder the same row is a lead
    led = L.ensure_flags(_ledger({"C6H5F7O2": 800.0, "C8H5F11O2": 400.0}, {"C6H5F7O2", "C8H5F11O2"}))
    out = PL.demote_off_budget(led, context="ambient-air", log=lambda *a: None, cfg=_cfg())
    assert bool(_row(led, "C6H5F7O2")["tentative_lead"]) and out["budget_demoted"] == 2


@pytest.mark.parametrize("case, committed, labelled, floor, lead, demoted", [
    ("two committed members", {"C6H5F7O2", "C8H5F11O2"}, False, 50.0, True, 2),
    ("15N alias open", {"C6H5F7O2", "C8H5F11O2", "C9H5F13O2"}, True, 50.0, True, 1),
    ("15N alias tested", {"C6H5F7O2", "C8H5F11O2", "C9H5F13O2"}, True, 5.0, False, 0),
])
def test_the_budget_keeps_only_a_full_ladder_not_a_lifted_one(case, committed, labelled, floor, lead, demoted):
    """The budget's keep reads `full`, as the tiers and the demote do: a ladder
    that only lifts the refusal -- two committed members, or three on a
    15N-nitrate ledger whose alias is untested (its twin would read under the
    floor) -- leaves the off-budget row a lead; tested, the same ladder is full
    and kept. (The row's +1 neighbour is only a picked line, so the
    committed-step rule does not keep it.)"""
    led = _ledger(dict(SERIES, C9H5F13O2=300.0), committed)
    led = L.ensure_flags(_labelled(led) if labelled else led)
    lad = FL.verdicts(led, cfg=_cfg(floor=floor))["C6H5F7O2"]
    assert lad.lift and lad.full != lead, case
    out = PL.demote_off_budget(led, context="ambient-air", log=lambda *a: None, cfg=_cfg(floor=floor))
    assert bool(_row(led, "C6H5F7O2")["tentative_lead"]) == lead, case
    assert out["budget_demoted"] == demoted, case


# =========================================================================== the F cap
def test_the_fluorinated_family_reaches_f25_and_the_space_follows():
    add = X.CONTAMINANT_FAMILIES["fluorinated"]["add"]
    assert add["F"] == (1, 25)
    ranges = DIR.build_ranges(X.get_context("ambient-air"), None, include_N=True, extra_elements=add)
    assert ranges["F"] == (1, 25)
    assert DIR._family_ok("C15H12F18", ranges) and DIR._family_ok("C14H10F18", ranges)
    assert not DIR._family_ok("C16H2F26", ranges)
    # the degeneracy space reaches every count a family can commit
    for fam in X.CONTAMINANT_FAMILIES.values():
        for el, (_lo, hi) in fam["add"].items():
            if el in DG.ELEMENT_CEILING:
                assert DG.ELEMENT_CEILING[el] >= hi, (el, hi)


# =========================================================================== the 13C alias term
def _iso_row(**kw):
    base = dict(compound_formula=None, compound_score=None, compound_category=2, ion_formula=None,
                ion_score=None, ion_category=2, mechanism_id="m", isotope_formula=None, iso_label="M0",
                is_base=True, theo_mz=533.0, rel_abundance=1.0, iso_score=None, iso_category=2,
                sample_peak_id="P", sample_peak_mz=533.0218, sample_peak_intensity=2000.0,
                ppm_error=0.2, abundance_error=0.0)
    base.update(kw)
    return base


def _alias_scored():
    """C20H9F10O2 vs C12H10F15O2 (the C8 / H1F5 alias, 0.16 mDa) on one peak; the
    C20 reading scores higher."""
    return pd.DataFrame([
        _iso_row(compound_formula="C20H9F10O2", ion_formula="C20H8F10O2-", ion_score=0.73,
                 compound_score=0.73, ppm_error=0.1),
        _iso_row(compound_formula="C12H10F15O2", ion_formula="C12H9F15O2-", ion_score=0.62,
                 compound_score=0.62, ppm_error=0.4),
    ])


def test_the_13c_line_decides_a_sub_mda_alias_when_testable():
    cfg = _cfg(floor=60.0)
    base = P.arbitrate(_alias_scored(), cfg)["winners"].iloc[0]
    assert base["neutral"] == "C20H9F10O2"
    line = {"P": (2000.0, 2000.0 * 0.0107 * 11.5, None, None, None)}      # 13C reads C11.5
    w = P.arbitrate(_alias_scored(), cfg, c13_alias=line)["winners"].iloc[0]
    assert w["neutral"] == "C12H10F15O2"
    assert "13C reads ~C" in w["_c13_note"] and "C20H9F10O2" in w["_c13_note"]
    alt = {a["formula"]: a["eff_score"] for a in w["alternatives"]}
    assert alt["C20H9F10O2"] == pytest.approx(0.73 - 0.20 - FL.C13_ALIAS_PENALTY)
    assert "13C reads" in P.core._commentary(w, 3, "contaminant:fluorinated:gka")


@pytest.mark.parametrize("line, cfg", [
    ((200.0, 200.0 * 0.0107 * 11.5, None, None, None), dict(floor=60.0)),   # untestable: 13C under the floor
    ((2000.0, 2000.0 * 0.0107 * 30, None, None, None), dict(floor=60.0)),    # contradicts both
    ((2000.0, 2000.0 * 0.0107 * 16, None, None, None), dict(floor=60.0)),    # contradicts neither
])
def test_the_13c_term_stays_out_where_it_cannot_decide(line, cfg):
    w = P.arbitrate(_alias_scored(), _cfg(**cfg), c13_alias={"P": line})["winners"].iloc[0]
    assert w["neutral"] == "C20H9F10O2"
    assert not isinstance(w.get("_c13_note"), str)


@pytest.mark.parametrize("mda, aliased", [(0.9, True), (1.5, False)])
def test_the_13c_term_needs_an_alias_within_the_window(mda, aliased):
    """Only readings within ALIAS_MDA (1 mDa) of the top one are aliases. Both
    readings sit inside the calibration acceptance (centre 1.5 ppm, sigma 1), so
    the mass term does not decide: at 0.9 mDa the line (C11.5) charges C20 and
    C12 wins, at 1.5 mDa the term stays out and C20 keeps the peak."""
    s = _alias_scored()
    s.loc[1, "ppm_error"] = 0.1 + mda / 533.0218 * 1e3
    cfg = _cfg(floor=60.0, sigma=1.0)
    cfg.cal_mu = 1.5
    line = (2000.0, 2000.0 * 0.0107 * 11.5, None, None, None)
    pen, _note = FL.c13_alias_penalties(s, line, cfg)
    assert pen == ({0: FL.C13_ALIAS_PENALTY} if aliased else {})
    w = P.arbitrate(s, cfg, c13_alias={"P": line})["winners"].iloc[0]
    assert w["neutral"] == ("C12H10F15O2" if aliased else "C20H9F10O2")


def test_the_13c_term_is_testable_only_where_the_carbon_poor_alias_clears_the_floor():
    """Testable means the 13C line of the alias with FEWEST carbons is predicted
    above the floor: at h0 = 450 the C12 reading's line (~58 cps) is under a 60 cps
    floor while the C20 reading's (~97) is above it -- a line that is missing or
    weak could be either, so the term stays out, whatever the line reads."""
    s = _alias_scored()
    cfg = _cfg(floor=60.0)
    assert 450.0 * FL.IR.R13C * 12 < 60.0 < 450.0 * FL.IR.R13C * 20
    assert FL.c13_alias_penalties(s, (450.0, 450.0 * 0.0107 * 12, None, None, None), cfg) == ({}, None)
    # brighter, the same reading is testable and charges C20
    assert FL.c13_alias_penalties(s, (900.0, 900.0 * 0.0107 * 12, None, None, None), cfg)[0] == {
        0: FL.C13_ALIAS_PENALTY}


@pytest.mark.parametrize("n, read, h0, snr0, why", [
    # the margin K x sigma: a read 2.5 C above C15 is 3 C nearer it than C23, under 2 x 2.15 C
    (15, 2.5, 1e4, 200.0, "margin"),
    # the line's own S/N: S/N ~3.4 on the 13C line (snr0 20) puts sigma at ~5 C, so 6 C is not enough
    (15, 1.0, 1200.0, 20.0, "line S/N"),
])
def test_the_13c_term_charges_only_beyond_its_margin(n, read, h0, snr0, why):
    s = _pair_scored(n).sort_values("ion_score", ascending=False)
    assert FL.c13_alias_penalties(s, _line(n + read, h0=h0, snr0=snr0), _cfg(floor=60.0)) == ({}, None), why


def _nitrate_alias_scored() -> pd.DataFrame:
    """The 15N alias pair (C2F3 vs H2^NO4, 0.21 mDa): C6H5F7O2 [M-H]- against
    C4H6F4O3 [M+^NO3]- on one peak, two carbons apart."""
    return pd.DataFrame([
        _iso_row(compound_formula="C6H5F7O2", ion_formula="C6H4F7O2-", ion_score=0.73, compound_score=0.73,
                 ppm_error=0.1, sample_peak_mz=240.9999),
        _iso_row(compound_formula="C4H6F4O3", ion_formula="C4H6F4^NO6-", ion_score=0.62, compound_score=0.62,
                 ppm_error=0.97, sample_peak_mz=240.9999),
    ])


def test_the_13c_term_never_charges_under_two_carbons():
    """C13_MIN_GAP: at C4 a bright line's 2 sigma is ~1 carbon, but an alias is
    charged only >= 2 carbons farther off than the nearest -- a line read 0.3 C
    above C4 (1.4 C nearer it than C6) charges neither reading of the 15N alias."""
    s = _nitrate_alias_scored()
    line = (1e4, 1e4 * FL.IR.R13C_RAW * 4.3, None, None, None)
    assert FL.C13_K * FL.c13_sigma(4.3, line) < 1.4          # the sigma margin alone would charge C6
    assert FL.c13_alias_penalties(s, line, _cfg(floor=60.0)) == ({}, None)


def test_the_13c_note_is_dropped_when_the_charged_top_stays_top():
    """The term charges C20 but C20 still outscores C12: the term did not decide
    the peak, so the winner carries no 13C note."""
    s = _alias_scored()
    s.loc[0, ["ion_score", "compound_score"]] = 0.95
    line = {"P": (2000.0, 2000.0 * 0.0107 * 11.5, None, None, None)}
    assert FL.c13_alias_penalties(s, line["P"], _cfg(floor=60.0))[0] == {0: FL.C13_ALIAS_PENALTY}
    w = P.arbitrate(s, _cfg(floor=60.0), c13_alias=line)["winners"].iloc[0]
    assert w["neutral"] == "C20H9F10O2"
    assert not isinstance(w.get("_c13_note"), str)
    assert "13C reads" not in P.core._commentary(w, 3, "contaminant:fluorinated:gka")


def test_c13_lines_reads_the_line_above_each_unexplained_peak():
    led = _ledger({"C6H5F7O2": 800.0}, set(), extra=[("c13", _mz("C6H5F7O2") + FL.IR.D13C, 50.0)])
    got = FL.c13_lines(led, _cfg())
    assert got["C6H5F7O2"][:2] == (800.0, 50.0)
    assert FL.c13_lines(led, None) == {}


def _pair_scored(n: int) -> pd.DataFrame:
    """C(n+8)H11F13O3 vs Cn H10F18O3 [M-H]- (the C8 / H1F5 alias) on one peak; the
    carbon-rich reading scores higher."""
    hi, lo = f"C{n + 8}H11F13O3", f"C{n}H10F18O3"
    return pd.DataFrame([
        _iso_row(compound_formula=hi, ion_formula=f"C{n + 8}H10F13O3-", ion_score=0.73, compound_score=0.73,
                 ppm_error=0.1, sample_peak_mz=595.035),
        _iso_row(compound_formula=lo, ion_formula=f"C{n}H9F18O3-", ion_score=0.62, compound_score=0.62,
                 ppm_error=0.37, sample_peak_mz=595.035),
    ])


def _line(c_read: float, h0: float = 1e4, snr0: float = 200.0):
    return (h0, h0 * FL.IR.R13C_RAW * c_read, None, None, snr0)


@pytest.mark.parametrize("n", [14, 15])
@pytest.mark.parametrize("read, winner", [
    (0.0, "lo"),     # a precise line at the carbon-poor count: the C+8 alias pays
    (1.5, "lo"),     # ~10 % high (noisy): still clearly nearer the carbon-poor count
    (4.0, "hi"),     # half-way: neither alias is clearly farther -- the term stays out
    (8.0, "hi"),     # the line reads the carbon-rich count: the carbon-poor alias pays
])
def test_the_13c_term_is_comparative_at_high_carbon(n, read, winner):
    """C14/C22 and C15/C23: contradicts' tolerance (0.35 n) exceeds the 8-carbon
    gap at C23, so an absolute test never charged the C+8 alias there; the
    comparative term does, and never charges the alias the line reads."""
    s = _pair_scored(n)
    w = P.arbitrate(s, _cfg(floor=60.0), c13_alias={"P": _line(n + read)})["winners"].iloc[0]
    want = f"C{n}H10F18O3" if winner == "lo" else f"C{n + 8}H11F13O3"
    assert w["neutral"] == want, (n, read, w["neutral"])
    pen, _note = FL.c13_alias_penalties(s.sort_values("ion_score", ascending=False), _line(n + read), _cfg(floor=60.0))
    assert bool(pen) == (read != 4.0)


def test_the_13c_term_reads_both_counts_where_the_run_reads_by_area():
    """By area (or with a response) the raw height reading AND the corrected one
    must both charge an alias: a line that reads the carbon-poor count by height
    but the carbon-rich one by area charges neither."""
    s = _pair_scored(15).sort_values("ion_score", ascending=False)
    h0 = 1e4
    agree = (h0, h0 * 0.0107 * 15, 100.0, 100.0 * FL.IR.R13C * 15, 200.0)
    split = (h0, h0 * 0.0107 * 15, 100.0, 100.0 * FL.IR.R13C * 23, 200.0)
    cfg = _cfg(floor=60.0, iso_area=True)
    assert FL.c13_alias_penalties(s, agree, cfg)[0] == {0: FL.C13_ALIAS_PENALTY}
    assert FL.c13_alias_penalties(s, split, cfg)[0] == {}
    assert len(FL.c13_reads(split, None, True)) == 2 and len(FL.c13_reads(split)) == 1


def test_c13_lines_reads_an_m0_line_but_not_another_rows_isotope():
    """An F line's own 13C line is often committed as a CHON(F) fit before pass 3
    reaches its parent: that line is read. A line another row claims as its
    isotope is not."""
    nf = "C6H5F7O2"
    led = _ledger({nf: 800.0}, set(), extra=[("c13", _mz(nf) + FL.IR.D13C, 50.0)])
    L.commit_assignment(led, "c13", neutral_formula="C9H7N3O6", adduct=ADD, ion_formula="C9H6N3O6-",
                        ion_score=0.8, compound_score=0.8, ppm_error=0.5, pass_no=1, method="cheminfo+grid",
                        confidence="Good", commentary="Pass 1")
    assert FL.c13_lines(led, _cfg())[nf][:2] == (800.0, 50.0)
    led2 = _ledger({nf: 800.0, "C8H5F11O2": 900.0}, {"C8H5F11O2"},
                   extra=[("c13", _mz(nf) + FL.IR.D13C, 50.0)])
    L.attach_isotopologue(led2, "c13", "C8H5F11O2", iso_label="34S")
    assert nf not in FL.c13_lines(led2, _cfg())


# =========================================================================== the wiring
def test_pass3_hands_the_13c_lines_to_the_fluorinated_family_only(monkeypatch):
    import dataclasses
    from collections import defaultdict
    from peaky.io import io_mascope as IO
    seen = []
    monkeypatch.setattr(DIR, "_enumerate", lambda *a, **k: {"C6H5F7O2"})
    monkeypatch.setattr(DIR, "_mech_ids_for", lambda client, adducts: [])
    monkeypatch.setattr(DIR, "_context_filter", lambda formulas, profile, rejected=None: set(formulas))
    monkeypatch.setattr(IO, "score_candidates", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(DIR, "arbitrate", lambda scored, cfg, profile=None, *, c13_alias=None:
                        seen.append(c13_alias) or {"winners": pd.DataFrame()})
    methods = []
    monkeypatch.setattr(DIR, "commit_winners",
                        lambda ledger, arb, **kw: methods.append(kw["method"]) or defaultdict(int))
    prof = dataclasses.replace(X.get_context("ambient-air"), pass3_families=("fluorinated", "siloxane"))
    led = _ledger({"C6H5F7O2": 800.0}, set(), extra=[("c13", _mz("C6H5F7O2") + FL.IR.D13C, 60.0)])
    DIR.run_pass3(None, "s", led, prof, None, _cfg(), [ADD], reagent=None, log=lambda *a: None)
    got = dict(zip(methods, seen))
    assert set(got) == {"contaminant:fluorinated", "contaminant:siloxane"}, got
    assert got["contaminant:fluorinated"] == FL.c13_lines(led, _cfg()) and got["contaminant:fluorinated"]
    assert got["contaminant:siloxane"] is None


def test_the_stages_thread_the_files_cfg_into_the_fluorine_readers(monkeypatch):
    """Without cfg the ladder window widens to 2 ppm and the 13C veto and the
    alias test are off: the demote and the element budget must get the file's."""
    import types
    from peaky.assignment import assign as A
    got = {}
    monkeypatch.setattr(A.cleanup, "demote_unconfirmed_fluorine",
                        lambda led, **kw: got.setdefault("demote", kw) and {})
    monkeypatch.setattr(A.plausibility, "demote_implausible",
                        lambda led, **kw: got.setdefault("plaus", kw) and {})
    monkeypatch.setattr(A, "_curated", lambda st: frozenset())
    cfg = _cfg()
    st = types.SimpleNamespace(led=pd.DataFrame(), cfg=cfg, log=lambda *a: None, plaus_audit=[],
                               profile=X.get_context("ambient-air"))
    stage = next(s for s in A._STAGES if s.name == "demote_fluorine")
    stage.fn(st)
    A._stage_plausibility(st)
    assert got["demote"]["cfg"] is cfg and got["plaus"]["cfg"] is cfg


def test_demote_implausible_hands_the_cfg_on_to_the_budget(monkeypatch):
    """The second hop: demote_implausible passes the file's cfg to
    demote_off_budget (whose CF2 ladder reads its windows and floor)."""
    got = {}
    monkeypatch.setattr(PL, "demote_off_budget", lambda led, **kw: got.update(kw) or {})
    cfg = _cfg()
    PL.demote_implausible(_ledger(SERIES, {"C6H5F7O2"}), log=lambda *a: None, context="ambient-air", cfg=cfg)
    assert got["cfg"] is cfg and got["context"] == "ambient-air"


# =========================================================================== the uncapped contexts
def test_an_uncapped_context_keeps_f17_unless_the_family_opens():
    """A context that leaves max_F at its default is enumerated to F17 as before;
    the fluorinated family's F25 adds its own profile only where it opened."""
    for name in ("food", "none"):
        assert [DG._caps(p)["F"] for p in DG.space_profiles(name, ())] == [17]
        assert [DG._caps(p)["F"] for p in DG.space_profiles(name, ("siloxane",))] == [17]
        assert [DG._caps(p)["F"] for p in DG.space_profiles(name, ("fluorinated",))] == [17, 25]
    assert [DG._caps(p)["F"] for p in DG.space_profiles("ambient-air", ("fluorinated",))] == [0, 25]
