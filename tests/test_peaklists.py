"""The bundled reference peaklists: every species a molecule, every radical flag
agreeing with its formula, and the loader sorting by the formula.

Radical status follows DBE parity, not the hydrogen count. The monoterpene HOM
list set its `radical` flag from an odd H count, which is wrong once nitrogen is
present: an organic nitrate like C10H15NO8 has odd H and an integer DBE, so it is
closed-shell, and 118 of them sat in the radical pool default matching skips.
The loader now reads parity off the formula and treats the flag as a claim;
these tests hold every bundled list to its claims. Mascope keeps the same
integrity test on its own copy of the lists until its step 2.7 settles which
copy peaky reads.
"""
import json
import re
import warnings
from pathlib import Path

import pytest

from peaky import chemistry as C
from peaky import reflists as RL
from peaky.paths import pkg_data

LIST_FILES = sorted(Path(pkg_data("peaklists")).glob("*.json"))

# DBE = 1 + sum n_i (v_i - 2) / 2 over the neutral's elements. Spelled out here
# rather than taken from chemistry.dbe, so the check stands apart from the loader
# it checks; an element missing from the table fails loudly instead of counting
# as divalent. '^N' (15N, from labelled-reagent chemistry) is a species of its
# own with nitrogen's valence, so a labelled formula is scored, not rejected.
VALENCE = {"C": 4, "Si": 4, "N": 3, "^N": 3, "P": 3, "O": 2, "S": 2,
           "H": 1, "F": 1, "Cl": 1, "Br": 1, "I": 1}
_NEUTRAL = re.compile(r"(?:\^?[A-Z][a-z]?\d*)+")
_ELEMENT = re.compile(r"(\^?[A-Z][a-z]?)(\d*)")


def dbe(formula: str) -> float:
    assert _NEUTRAL.fullmatch(formula), f"{formula!r} is not a neutral formula"
    total = 1.0
    for el, n in _ELEMENT.findall(formula):
        assert el in VALENCE, f"{formula}: no valence for {el}"
        total += (int(n) if n else 1) * (VALENCE[el] - 2) / 2
    return total


def odd_electron(formula: str) -> bool:
    return dbe(formula) % 1 == 0.5


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_the_lists_are_found():
    assert len(LIST_FILES) >= 2, LIST_FILES


@pytest.mark.parametrize(("formula", "radical"), [
    ("C10H16O7", False),    # a closed-shell HOM
    ("C10H15O8", True),     # its peroxy radical
    ("C10H15NO8", False),   # odd H, but an organic nitrate is closed-shell
    ("C10H16NO9", True),    # even H, and a nitrogen-bearing radical
    ("HO2", True),
    ("H3N", False),
    ("C2H8O2Si", False),    # dimethylsilanediol: silicon counts with carbon
    ("C6H15O4P", False),    # triethyl phosphate: phosphorus counts with nitrogen
    ("IO2", True),          # iodine dioxide: iodine counts with hydrogen
    ("C10H15^NO8", False),  # the 15N-labelled nitrate: labelled N counts as N
    ("C10H16^NO9", True),   # and its labelled radical
])
def test_radical_status_is_read_off_the_formula(formula, radical):
    assert RL.is_radical(formula) is radical
    assert odd_electron(formula) is radical


@pytest.mark.parametrize(("formula", "radical"), [
    ("C10H16O7", False),
    ("HO2", True),
    ("C10H15NO8", False),   # odd H, integer DBE: the case the H count gets wrong
    ("C10H16NO9", True),    # even H, half-integer DBE: the other way round
    ("H3N", False),
    ("C6H15O4P", False),    # P counts with N
    ("IO2", True),          # I counts with H
])
def test_odd_electron_is_the_one_parity_helper(formula, radical):
    # chemistry.odd_electron is what the grid gate (dbe_ok), the plausibility
    # radical exemption and the list loader share; the valence table above is
    # its independent oracle. Both call forms, a formula and its counts.
    assert C.odd_electron(formula) is radical
    assert C.odd_electron(C.parse_formula(formula)) is radical
    assert odd_electron(formula) is radical
    assert C.dbe_ok(formula)[0] is (not radical)


def test_an_ion_or_a_salt_has_a_negative_dbe():
    assert dbe("C25H54ClN") == -1      # behentrimonium chloride
    assert dbe("C16H36N") == -0.5      # the tetrabutylammonium cation
    assert dbe("C10H16O7") == 3


@pytest.mark.parametrize("path", LIST_FILES, ids=lambda p: p.stem)
def test_the_declared_size_is_the_real_one(path):
    # `n_species` is what the docs quote (ASSIGNMENT.md, ASSIGNMENT_DETAIL.md,
    # peaklists/README.md); a correction that drops entries must move it too
    d = read(path)
    assert d["n_species"] == len(d["species"])


@pytest.mark.parametrize("path", LIST_FILES, ids=lambda p: p.stem)
def test_every_species_is_a_molecule(path):
    ions = [s["formula"] for s in read(path)["species"] if dbe(s["formula"]) < 0]
    assert ions == []


@pytest.mark.parametrize("path", LIST_FILES, ids=lambda p: p.stem)
def test_every_radical_flag_matches_its_parity(path):
    # an absent flag claims closed-shell, as the loader reads it
    wrong = [s["formula"] for s in read(path)["species"]
             if bool(s.get("radical", False)) != odd_electron(s["formula"])]
    assert wrong == []


def test_the_bundled_lists_load_into_their_parity_pools():
    with warnings.catch_warnings():
        warnings.simplefilter("error")          # a disagreeing flag would warn
        catalog = RL.load_catalog()
    for path in LIST_FILES:
        d = read(path)
        L = catalog[d["id"]]
        formulas = {s["formula"] for s in d["species"]}
        assert L.radicals == {f for f in formulas if odd_electron(f)}, path.name
        assert L.formulas == formulas - L.radicals, path.name


def test_the_bundled_lists_carry_the_corrected_split():
    catalog = RL.load_catalog()
    hom = catalog["monoterpene_hom_kang2024"]
    # 455 CHO + 118 organic nitrates closed-shell, 254 CHO + 3 N-bearing radicals;
    # the hydrogen-count flag had them 458 and 372
    assert (len(hom.formulas), len(hom.radicals)) == (573, 257)
    assert {"C10H15NO8", "C10H13NO10"} <= hom.formulas
    keller = catalog["contaminants_keller2008"]
    assert (len(keller.formulas), len(keller.radicals)) == (50, 0)
    assert "C5H9NO" in keller.formulas          # NMP, no longer its [M+H]+ ion


def test_the_demoted_nitrogen_radicals_are_named_by_the_list_and_its_caveat():
    # the correction moved 118 nitrates INTO default matching and exactly these
    # three N-bearing radicals out of it -- the only loss, so name them here and
    # in the list's own caveat rather than trusting the counts above
    hom = RL.load_catalog()["monoterpene_hom_kang2024"]
    demoted = {"C4H6NO7", "C4H8NO6", "C5H8NO7"}
    assert {f for f in hom.radicals if "N" in f} == demoted
    assert demoted.isdisjoint(hom.formulas)
    caveat = read(next(p for p in LIST_FILES
                       if p.stem == "monoterpene_hom_kang2024"))["provenance"]["caveat"]
    assert [f for f in sorted(demoted) if f not in caveat] == []


def test_parity_not_the_flag_decides_and_a_wrong_flag_warns(tmp_path):
    (tmp_path / "t.json").write_text(json.dumps({"id": "t", "species": [
        {"formula": "C10H16O7", "radical": False},
        {"formula": "C10H15NO8", "radical": True},   # the old odd-H flag
        {"formula": "C10H16NO9"},                    # absent: claims closed-shell
        {"formula": "C10H15O8", "radical": True},
    ]}), encoding="utf-8")
    with pytest.warns(UserWarning, match=r"t\.json: .* 2 species .*C10H15NO8, C10H16NO9"):
        L = RL.load_catalog(str(tmp_path))["t"]
    assert L.formulas == {"C10H16O7", "C10H15NO8"}
    assert L.radicals == {"C10H16NO9", "C10H15O8"}
    assert L.pool() == L.formulas


def test_a_salt_or_an_ion_is_skipped_before_the_parity_test(tmp_path):
    # chemistry.dbe scores an unknown element as divalent (sodium acetate would
    # come out DBE 1.5, a "radical"), and parse_formula drops charge notation
    # (the ion would load as a closed-shell neutral): neither may reach a pool
    (tmp_path / "t.json").write_text(json.dumps({"id": "t", "species": [
        {"formula": "C2H3NaO2"},                    # sodium acetate
        {"formula": "[C10H14NO8]-"},                # a bracketed ion
        {"formula": "C16H36N"},                     # tetrabutylammonium: DBE -0.5
        {"formula": "C10H16O7"},
        {"formula": "C10H15O8", "radical": True},
    ]}), encoding="utf-8")
    with pytest.warns(UserWarning, match=(r"t\.json: 3 species skipped.*C2H3NaO2: element Na"
                                          r".*\[C10H14NO8\]-: does not read back as C10H14NO8"
                                          r".*C16H36N: DBE -0\.5")) as rec:
        L = RL.load_catalog(str(tmp_path))["t"]
    assert len(rec) == 1                            # no radical-flag warning on top
    assert L.skipped == ("C2H3NaO2", "[C10H14NO8]-", "C16H36N")
    assert L.formulas == {"C10H16O7"}
    assert L.radicals == {"C10H15O8"}


def test_six_wrong_flags_name_five_and_truncate(tmp_path):
    six = ["C10H16O7", "C10H15NO8", "C10H14O8", "C9H14O7", "C8H12O6", "C10H16O9"]
    (tmp_path / "t.json").write_text(json.dumps({"id": "t", "species": [
        {"formula": f, "radical": True} for f in six]}), encoding="utf-8")   # all closed-shell
    with pytest.warns(UserWarning) as rec:
        L = RL.load_catalog(str(tmp_path))["t"]
    msg = str(rec[0].message)
    assert "6 species disagrees" in msg
    assert ", ".join(six[:5]) + ", ...)" in msg and six[5] not in msg
    # and it says where the full list is, rather than leaving the reader at five
    assert msg.endswith("parity decides -- run tests/test_peaklists.py for the full list")
    assert (L.formulas, L.radicals) == (frozenset(six), frozenset())


def test_the_bundled_lists_skip_nothing():
    assert all(L.skipped == () for L in RL.load_catalog().values())


def test_the_corrected_pool_reaches_the_matchers_and_the_prior():
    # the correction only matters if the consumers see it: an organic nitrate is
    # matchable by default, a real N-bearing radical is not, and the H-count flag
    # had it exactly the other way round. Both are on the bundled HOM list.
    nitrate, radical = "C10H15NO8", "C5H8NO7"
    lists = RL.active_lists(RL.load_catalog(), context_tags={"monoterpene_ox"})
    mzs = [C.ion_mz(f, "[M-H]-") for f in (nitrate, radical)]

    hits = RL.match_by_mass(mzs, lists, ["[M-H]-"], tol_ppm=2.0)
    assert [h["formula"] for h in hits] == [nitrate]        # the radical mass is unmatched
    both = RL.match_by_mass(mzs, lists, ["[M-H]-"], tol_ppm=2.0, include_radicals=True)
    assert [h["formula"] for h in both] == [nitrate, radical]

    assert set(RL.match_assigned([nitrate, radical], lists)) == {nitrate}
    assert set(RL.match_assigned([nitrate, radical], lists,
                                 include_radicals=True)) == {nitrate, radical}

    # the selection prior assign.run hands to arbitration, built from the same lists
    prior = RL.prior_formulas(lists)
    assert nitrate in prior and radical not in prior
    assert RL.prior_formulas([]) == frozenset()             # a run with no lists


def test_a_run_records_which_list_versions_were_active():
    # assign.run puts this next to the selection prior it builds from the same
    # lists, and the CLI writes it into the sample's manifest as `reflists_active`
    act = RL.active_lists(RL.load_catalog(), context_tags={"monoterpene_ox"})
    assert RL.active_versions(act) == [("contaminants_keller2008", "2008.2"),
                                       ("monoterpene_hom_kang2024", "2024.2")]
    assert RL.active_versions(None) == []           # a run with no lists
