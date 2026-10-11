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
    twin = ("twin", _mz("C6H5F7O2") - FL.D_15N, 800.0 * ratio)
    led = _labelled(_ledger(SERIES, {"C6H5F7O2", "C7H5F9O2", "C8H5F11O2"}, extra=[twin]))
    lad = FL.verdicts(led, cfg=_cfg())["C6H5F7O2"]
    assert (lad.veto is not None) == refuted
    assert lad.full != refuted
    if refuted:
        assert "14N twin" in lad.veto


def test_the_twin_is_read_only_on_a_15n_nitrate_ledger():
    twin = ("twin", _mz("C6H5F7O2") - FL.D_15N, 16.0)
    led = _ledger(SERIES, {"C6H5F7O2", "C7H5F9O2", "C8H5F11O2"}, extra=[twin])
    assert FL.verdicts(led, cfg=_cfg())["C6H5F7O2"].full


def test_an_f_reading_that_carries_15n_is_not_refuted_by_a_twin():
    """[M+^NO3]- of an F neutral carries one ^N (its own twin ~ f), its alias two
    (~ 2f); the 14N share is not known well enough to tell f from 2f (the air's
    14N nitrate adds to the reagent's impurity), so no twin refutes it."""
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
        lad = FL.verdicts(led, cfg=_cfg())["C6H5F7O2"]
        assert lad.veto is None and lad.full, (share, lad)


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


def test_the_13c_term_needs_an_alias_within_the_window():
    s = _alias_scored()
    s.loc[1, "ppm_error"] = 3.0                      # 1.5 mDa from the top: not an alias
    w = P.arbitrate(s, _cfg(floor=60.0),
                    c13_alias={"P": (2000.0, 2000.0 * 0.0107 * 11.5, None, None, None)})["winners"].iloc[0]
    assert w["neutral"] == "C20H9F10O2"


def test_c13_lines_reads_the_line_above_each_unexplained_peak():
    led = _ledger({"C6H5F7O2": 800.0}, set(), extra=[("c13", _mz("C6H5F7O2") + FL.IR.D13C, 50.0)])
    got = FL.c13_lines(led, _cfg())
    assert got["C6H5F7O2"][:2] == (800.0, 50.0)
    assert FL.c13_lines(led, None) == {}
