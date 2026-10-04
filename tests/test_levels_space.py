"""The evidence scale's enumeration space (peaky.assignment.levels.space): the
plausible-decomposition rule and its reasons, adduct deltas, decompositions, the
full (untruncated) enumeration and the D5 widening for a committed neutral the
context filter drops. Offline, no grid build."""
import pytest

from peaky.assignment import degeneracy as DG
from peaky.assignment.levels import space as SP
from peaky.chem import chemistry as C

NITRATE = "NO3+NO3_15N"
AMBIENT_FAMS = ("organosulfate", "nitrate", "siloxane", "amine", "fluorinated")


@pytest.fixture(scope="module")
def space():
    return SP.Space(NITRATE, "ambient-air", [], AMBIENT_FAMS)


def _nz(d):
    return {k: v for k, v in (d or {}).items() if v}


# --------------------------------------------------------------------------- adduct composition
def test_adduct_delta_reads_the_adduct_composition_with_the_label_apart():
    assert _nz(SP.adduct_delta("[M+^NO3]-")) == {"^N": 1, "O": 3}
    assert _nz(SP.adduct_delta("[M+NO3]-")) == {"N": 1, "O": 3}
    assert _nz(SP.adduct_delta("[M-H]-")) == {"H": -1}
    assert _nz(SP.adduct_delta("[M+(CH4N2O)H]+")) == {"C": 1, "H": 5, "N": 2, "O": 1}
    assert _nz(SP.adduct_delta("[M+H]+")) == {"H": 1}


def test_adduct_delta_is_none_for_an_adduct_without_composition():
    assert SP.adduct_delta("not an adduct") is None


def test_reagent_supply_is_the_most_atoms_any_adduct_adds():
    assert SP.reagent_supply(["[M+NO3]-", "[M-H]-"], "N") == 1
    assert SP.reagent_supply(["[M+NO3]-", "[M-H]-"], "Br") == 0
    assert SP.reagent_supply(["[M+(CH4N2O)H]+", "[M+H]+"], "N") == 2


def test_ion_counts_of_prefers_a_charged_ion_formula_and_drops_zeros():
    assert SP.ion_counts_of("C6H10O5", "[M+NO3]-") == {"C": 6, "H": 10, "N": 1, "O": 8}
    assert SP.ion_counts_of("C6H10O5", "[M+^NO3]-") == {"C": 6, "H": 10, "^N": 1, "O": 8}
    assert SP.ion_counts_of("ignored", "ignored", "C2H3O2-") == {"C": 2, "H": 3, "O": 2}
    assert SP.ion_counts_of(None, None, None) == {}


# --------------------------------------------------------------------------- plausible_neutral
def test_plausible_neutral_reasons(space):
    assert space.plausible_neutral({"C": 2, "H": -1}) == (False, "negative count")
    assert space.plausible_neutral({"C": 0, "H": 0}) == (False, "empty")
    assert space.plausible_neutral({"C": 2, "H": 4, "^N": 1, "O": 2}) == (False, "carries the labelled reagent atom")
    odd = {"C": 1, "H": 5}
    ok, why = space.plausible_neutral(odd)
    assert not ok and why == C.dbe_ok(odd)[1] and why
    assert space.plausible_neutral({"C": 6, "H": 10, "O": 5}) == (True, "")


def test_plausible_neutral_outside_the_space_names_the_first_profiles_reason(space):
    ok, why = space.plausible_neutral({"C": 10, "H": 10, "Cl": 1, "N": 2, "O": 1, "P": 1})
    assert not ok
    assert why == "P1 over the ambient-air element budget (P<=0)"
    ok, why = space.plausible_neutral({"C": 10, "H": 9, "Cl": 1, "N": 1, "O": 1, "P": 1, "S": 1})
    assert not ok and why == "4 heteroatom types > 3"


def test_a_curated_formula_bypasses_the_space_but_not_the_dbe_rule(space):
    for f in sorted(space.curated):
        if DG._space_reason(f, space.profiles, frozenset()) is not None and C.dbe_ok(C.parse_formula(f))[0]:
            assert space.space_reason(f) is None
            assert space.plausible_neutral(C.parse_formula(f)) == (True, "")
            break
    else:
        pytest.skip("no curated formula outside the base space")


# --------------------------------------------------------------------------- decompositions
def test_decompositions_one_per_adduct_ion_only_skipped(space):
    counts = SP.ion_counts_of("C6H10O5", "[M+NO3]-")
    out = SP.decompositions(space, counts, "C6H10O5", "[M+NO3]-", ["[M+NO3]-", "[M-H]-", "[M]-."])
    assert [d["adduct"] for d in out] == ["[M+NO3]-", "[M-H]-"]
    first = out[0]
    assert first["neutral"] == "C6H10O5" and first["ok"] and first["why"] == ""
    assert out[1]["neutral"] == "C6H11NO8"


def test_decompositions_negative_count_has_no_neutral(space):
    counts = {"C": 2, "H": 3, "O": 2}          # acetate: no nitrate inside
    out = SP.decompositions(space, counts, "C2H4O2", "[M-H]-", ["[M+NO3]-"])
    by = {d["adduct"]: d for d in out}
    assert by["[M+NO3]-"]["neutral"] is None and not by["[M+NO3]-"]["ok"]
    assert by["[M+NO3]-"]["why"] == "negative count"
    assert by["[M-H]-"]["neutral"] == "C2H4O2" and by["[M-H]-"]["ok"]


# --------------------------------------------------------------------------- enumeration
def test_enumerate_is_degeneracy_enumerate_window_and_not_truncated(space):
    mz = C.ion_mz("C10H16O8", "[M+NO3]-")
    found = space.enumerate(mz, -6.0, 6.0, space.adducts)
    assert found == DG.enumerate_window(mz, -6.0, 6.0, space.adducts, space)
    assert len(found) > DG.MAX_ALTS
    assert space.canon("C10H16O8", "[M+NO3]-") in found
    for ion, (n, a, ppm) in found.items():
        assert -6.0 <= ppm <= 6.0 and space.canon(n, a) == ion


def test_enumerate_skips_channels_without_a_shift_and_canon_never_raises(space):
    mz = C.ion_mz("C6H10O5", "[M+NO3]-")
    assert space.enumerate(mz, -1, 1, ["[M+nonsense]-"]) == {}
    assert space.canon("C6H10O5", "not an adduct") is None


def test_filter_kind_blanks_the_numbers():
    assert DG.filter_kind("O=7 implausible for C=2") == "O=# implausible for C=#"
    assert DG.filter_kind("H/C 3.10 outside [0.70, 2.75]") == "H/C # outside [#, #]"


# --------------------------------------------------------------------------- D5 widening
def test_relaxed_space_admits_the_committed_neutral_and_says_why(space):
    n = "C10H10ClN2OP"
    assert space.space_reason(n) == "P1 over the ambient-air element budget (P<=0)"
    rs = space.relaxed(n)
    assert rs is not None and rs is not space
    assert rs.space_reason(n) is None
    assert rs.relaxed_note == ("space widened to admit the committed neutral "
                               "(P1 over the ambient-air element budget (P<=0))")
    assert space.relaxed(n) is rs                  # memoised
    assert space.relaxed_note == ""                # the run space itself is untouched
    mz = C.ion_mz(n, "[M+^NO3]-")
    assert space.canon(n, "[M+^NO3]-") in rs.enumerate(mz, -1.0, 1.0, space.adducts + ["[M+^NO3]-"])


def test_relaxed_is_none_when_no_widening_admits_the_neutral(space):
    assert space.relaxed("C10H9ClNOPS") is None      # four heteroatom types
    assert space.relaxed("C10H9ClNOPS") is None


def test_relaxed_admits_a_context_filter_failure_of_its_own_kind():
    sp = SP.Space(NITRATE, "ambient-air", [], ())
    n = "C2H2O7"                                     # far over the context's O/C window and the C1/C2 rule
    why = sp.space_reason(n)
    assert why is not None and why.startswith("context filter: ")
    rs = sp.relaxed(n)
    assert rs is not None
    assert DG.filter_kind(why[len("context filter: "):]) in rs.accept_kinds
    assert sp.accept_kinds == frozenset()            # the run space keeps its filter
    mz = C.ion_mz(n, "[M-H]-")
    assert sp.canon(n, "[M-H]-") not in sp.enumerate(mz, -1.0, 1.0, sp.adducts)
    assert sp.canon(n, "[M-H]-") in rs.enumerate(mz, -1.0, 1.0, sp.adducts)
