"""The element-budget demote (plausibility.demote_off_budget): a commit whose
neutral lies outside the run context's element budget and that no curated list
names is Candidate + below_assignability, and the evidence level reads 5b.

The budget is contexts.element_budget -- steps 1-3 of filter_by_profile (the
structural gate, the carbon-free allowlist, the heteroatom caps), never the
halogen minimum-carbon scaffold or the Van Krevelen windows."""
from types import SimpleNamespace

import pandas as pd
import pytest

from peaky.assignment import assign as A
from peaky.assignment import evidence as EV
from peaky.assignment import passes
from peaky.assignment import plausibility as PL
from peaky.chem import contexts as X


# --------------------------------------------------------------------------- the budget
@pytest.mark.parametrize("formula,reason", [
    ("C40H66N3O2PS3", "S=3 > 1"),       # the TOF's certified P/S3 commit
    ("C5H6ClN2OP", "P=1 > 0"),          # P is monoisotopic: never isotope-confirmable
    ("C5HF3O2", "F=3 > 0"),             # so is F
    ("C11H20O5S2", "S=2 > 1"),
    ("C8H24O4Si4", "Si=4 > 1"),
    ("C9H12N4O3", "N=4 > 3"),
    ("C10H17Cl3", "Cl=3 > 2"),
])
def test_ambient_budget_rejects_on_the_caps(formula, reason):
    assert X.element_budget(formula, "ambient-air") == (False, reason)


@pytest.mark.parametrize("formula", [
    "C10H16O5",       # a monoterpene oxidation product
    "C7H5NO4",        # nitrobenzoic acid: DBE/C 0.86, outside the VK window, inside the budget
    "C8H4O4",         # (H+X)/C 0.50: a VK violation, not a budget one
    "C2H3BrO2",       # bromoacetic acid: the minimum-carbon rule is a reagent-alias guard, not budget
    "C4H7ClO4",
    "C12H15NO3Si",    # one Si is inside the ambient budget
    "HNO3",           # an allowlisted carbon-free analyte
    "C40H66O2",       # carbon count is not a budget term
])
def test_ambient_budget_keeps_what_only_the_grid_priors_reject(formula):
    assert X.element_budget(formula, "ambient-air") == (True, None)


def test_budget_follows_the_context():
    assert X.element_budget("C11H20O5S2", "uronium") == (True, None)          # S <= 2 there
    assert X.element_budget("C7H11ClO2", "uronium") == (False, "Cl=1 > 0")   # no halogen in + mode
    assert X.element_budget("C8H24O4Si4", "uronium") == (True, None)          # the PDMS ladder
    assert X.element_budget("HNO3", "uronium") == (False, "no carbon")        # allowlist is ambient-only
    assert X.element_budget("C10H16O5", X.get_context("ambient-air")) == (True, None)   # a profile works


def test_structural_gate_is_part_of_the_budget():
    ok, why = X.element_budget("C10H15O2", "ambient-air")                     # odd-electron
    assert not ok and why


@pytest.mark.parametrize("formula", [
    "C40H66N3O2PS3", "C7H5NO4", "C8H4O4", "C2H3BrO2", "C4H7ClO4", "HNO3", "C10H16O5", "C5HF3O2",
    "C10H15O2", "CH3NO5", "C13H9N3", "C9H18O9Si", "C3H8N2O", "H2SO4", "C24H40Cl10", "NO3",
])
@pytest.mark.parametrize("ctx", ["ambient-air", "uronium", "indoor-air", "water", "chamber"])
def test_filter_by_profile_is_unchanged_by_the_refactor(formula, ctx):
    """filter_by_profile = the budget, then the minimum-carbon rule, then the VK
    windows -- the budget answer is its answer whenever the budget rejects."""
    ok_b, why_b = X.element_budget(formula, ctx)
    ok_f, why_f = X.filter_by_context(formula, ctx)
    if not ok_b:
        assert (ok_f, why_f) == (ok_b, why_b)
    else:
        assert ok_f or "C>=" in why_f or "out of" in why_f or "implausible for C=" in why_f


# --------------------------------------------------------------------------- the demote
def _ledger(rows):
    base = dict(role="M0", tier="Assigned", below_assignability=False, isotopologues="[]",
                degeneracy_note="", commentary="", method="cheminfo+grid", adduct="[M-H]-",
                confidence="High (0.9)", tied=False, degeneracy_density=0, series_unit=None,
                anchor_peak_id=None, parent_peak_id=None, iso_label=None, height=1000.0)
    out = []
    for k, r in enumerate(rows):
        d = dict(base, peak_id=f"p{k}", mz=100.0 + k, ion_formula=r.get("neutral_formula", ""))
        d.update(r)
        out.append(d)
    return pd.DataFrame(out)


def test_off_budget_commit_is_candidate_below_assignability_with_a_note_and_an_audit_row():
    led = _ledger([dict(neutral_formula="C5H6ClN2OP", method="certified:multi-channel"),
                   dict(neutral_formula="C10H16O5")])
    audit = []
    out = PL.demote_off_budget(led, context="ambient-air", audit=audit, log=lambda *a: None)
    assert out == {"budget_demoted": 1}
    assert led.loc[0, "tier"] == "Candidate" and bool(led.loc[0, "below_assignability"])
    assert "outside the ambient-air element budget (P=1 > 0)" in led.loc[0, "commentary"]
    assert led.loc[1, "tier"] == "Assigned" and not bool(led.loc[1, "below_assignability"])
    assert len(audit) == 1 and audit[0]["evidence"] == "P=1 > 0" and audit[0]["before_tier"] == "Assigned"


def test_the_level_reads_the_demote_as_5b():
    led = _ledger([dict(neutral_formula="C4H8N2S3", adduct="[M-H]-", method="certified:multi-channel"),
                   dict(neutral_formula="C4H8N2S3", adduct="[M+NO3]-", method="certified:multi-channel",
                        peak_id="q", mz=241.97)])
    before = EV.compute_levels(led)
    assert set(before["evidence_level"]) == {"3b"}            # the certificate read back as the acid branch
    PL.demote_off_budget(led, context="ambient-air", log=lambda *a: None)
    after = EV.compute_levels(led)
    assert set(after["evidence_level"]) == {"5b"}
    assert all("below assignability" in r for r in after["level_reason"])


def test_a_curated_formula_is_exempt_whichever_pass_committed_it():
    led = _ledger([dict(neutral_formula="C6HF11O2", method="completion:known-neutral"),
                   dict(neutral_formula="C18H15OP", method="certified:multi-channel")])
    curated = passes.known_formulas("negative", "ambient-air") | {"C18H15OP"}
    assert "C6HF11O2" in curated                               # the PFCA family of pass 0
    out = PL.demote_off_budget(led, context="ambient-air", curated=curated, log=lambda *a: None)
    assert out == {"budget_demoted": 0}
    assert list(led["tier"]) == ["Assigned", "Assigned"]


def test_grid_priors_and_the_reagent_alias_guard_are_not_the_budget():
    led = _ledger([dict(neutral_formula="C7H5NO4"), dict(neutral_formula="C2H3BrO2"),
                   dict(neutral_formula="C8H4O4"), dict(neutral_formula="C30H43NO")])
    out = PL.demote_off_budget(led, context="ambient-air", log=lambda *a: None)
    assert out == {"budget_demoted": 0}
    assert (led["tier"] == "Assigned").all()


def test_candidates_are_flagged_and_other_roles_are_untouched():
    led = _ledger([dict(neutral_formula="C5HF3O2", tier="Candidate"),
                   dict(neutral_formula="C5HF3O2", role="iso_child", tier=None),
                   dict(neutral_formula="C5HF3O2", method="ion_only:electron_attachment", adduct="[M]-.")])
    audit = []
    out = PL.demote_off_budget(led, context="ambient-air", audit=audit, log=lambda *a: None)
    assert out == {"budget_demoted": 1}
    assert led.loc[0, "tier"] == "Candidate" and bool(led.loc[0, "below_assignability"])
    assert not bool(led.loc[1, "below_assignability"]) and not bool(led.loc[2, "below_assignability"])
    assert audit[0]["before_tier"] == "Candidate"


def test_no_context_is_a_no_op_and_demote_implausible_keeps_its_old_keys():
    led = _ledger([dict(neutral_formula="C5H6ClN2OP")])
    assert PL.demote_off_budget(led, context=None, log=lambda *a: None) == {"budget_demoted": 0}
    out = PL.demote_implausible(led, audit=[], log=lambda *a: None)
    assert set(out) == {"o_demoted", "c_cluster_demoted"} and led.loc[0, "tier"] == "Assigned"
    out = PL.demote_implausible(led, audit=[], log=lambda *a: None, context="ambient-air")
    assert out["budget_demoted"] == 1 and led.loc[0, "tier"] == "Candidate"


def test_the_stage_passes_the_context_and_the_curated_lists():
    led = _ledger([dict(neutral_formula="C5H6ClN2OP"),     # off budget, not curated
                   dict(neutral_formula="C6HF11O2"),       # off budget, pass-0 registry (PFCA)
                   dict(neutral_formula="C18H15OP"),       # off budget, on the reference list
                   dict(neutral_formula="C10H16O5")])
    st = SimpleNamespace(led=led, plaus_audit=[], log=lambda *a: None,
                         profile=X.get_context("ambient-air"),
                         cfg=SimpleNamespace(reflist_formulas=frozenset({"C18H15OP"})))
    out = A._stage_plausibility(st)
    assert out["budget_demoted"] == 1
    assert list(led["tier"]) == ["Candidate", "Assigned", "Assigned", "Assigned"]
    assert len(st.plaus_audit) == 1


def test_the_stage_is_the_pipelines_plausibility_stage():
    stage = next(s for s in A._STAGES if s.name == "plausibility")
    assert stage.fn is A._stage_plausibility


def test_known_formulas_follow_polarity():
    neg = passes.known_formulas("negative", "ambient-air")
    pos = passes.known_formulas("positive", "uronium")
    assert "C2HF3O2" in neg and "C18H15OP" not in neg          # PFCAs are a negative-mode family
    assert "C18H15OP" in pos and "C8H24O4Si4" in pos           # organophosphates, cyclosiloxanes
