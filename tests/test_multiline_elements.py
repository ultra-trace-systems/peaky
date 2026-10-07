"""`multiline` counts ELEMENTS the neutral supplies, each by an in-band line (C17).

`multiline` was the only intra-channel outside term of the pre-0.10.0 level 4a:
two isotope lines of the ion that speak for the NEUTRAL. That decision lives on,
private, as the merge vote's class (`multiline` with another axis backs the
neutral: class 2); the fact is a column of tables/evidence_levels.csv. (The
evidence scale's own positive fact -- an own in-band line of an element the
neutral contains -- is computed by its line model, tests/test_levels_lines.py.) Until C17 it was
"two or more raw satellite tags", so a 13C line plus the 13C2 line (one element
twice), a urea adduct's own 15N, a bromide adduct's own 81Br or a generic 'M+n'
child all made it (2026-09-26 output audit, K01). Now: two distinct elements,
each with a line whose height is inside the ratio band of its expected
abundance, each supplied mostly by the neutral (more than half of the ion's atoms
of it). 15N and 18O join the band, per atom of the ion like 13C per carbon.

Run: pytest tests/test_multiline_elements.py -q
"""

from __future__ import annotations

import pandas as pd

from peaky.assignment import evidence as EV
from tests.test_evidence import child, ledger, m0, vote_of

N15 = EV.PER_ATOM_ABUNDANCE["15N"][1]
O18 = EV.PER_ATOM_ABUNDANCE["18O"][1]


def facts(rows) -> dict:
    """{(neutral, adduct): the pooled evidence record} of a synthetic ledger."""
    out = EV._series_pooled({"f": ledger(rows)})
    return {(r.neutral_formula, r.adduct): r for r in out.itertuples(index=False)}


def c13(peak, parent, ion_c, height=1000.0):
    return child(peak, parent, "13C+1", height * EV.C13_PER_CARBON * ion_c)


# --------------------------------------------------------------------------- the pieces
def test_tag_element_folds_every_satellite_spelling_into_its_element():
    for tag, el in [("13C", "C"), ("13C2", "C"), ("81Br", "Br"), ("2x81Br", "Br"), ("81Br2", "Br"),
                    ("81Br(pair)", "Br"), ("37Cl(pair)", "Cl"), ("2x37Cl", "Cl"), ("37Cl3", "Cl"),
                    ("15N", "N"), ("14N", "^N"), ("18O", "O"), ("34S", "S"), ("29Si", "Si"), ("30Si", "Si")]:
        assert EV.tag_element(tag) == el, tag
    for tag in ("M", "M0", "", "nan", "M+2"):
        assert EV.tag_element(tag) is None, tag


def test_an_element_speaks_for_the_neutral_when_the_neutral_supplies_most_of_it():
    # urea adduct of an N-free neutral: its N is all the reagent's; its O mostly the neutral's
    assert EV.neutral_elements("C10H16O4", "C11H21N2O5") == {"C", "H", "O"}
    assert "N" not in EV.neutral_elements("C10H16O4", "C11H21N2O5")
    # formic acid on nitrate: 2 of 5 O are the neutral's -> the 18O line measures the reagent
    assert "O" not in EV.neutral_elements("CH2O2", "CHNO5")
    # a C1 neutral on a C1 adduct: half is not most
    assert "C" not in EV.neutral_elements("CH2O2", "C2H7N2O3")
    # a bromide adduct of a Br-free neutral: the 81Br line is the reagent's
    assert "Br" not in EV.neutral_elements("C8H14O2", "C8H14BrO2")
    # an N2 neutral on nitrate supplies 2 of 3 N
    assert "N" in EV.neutral_elements("C10H16N2O8", "C10H16N3O11")
    # on a 15N-labelled nitrate adduct the ion's 14N atoms are all the neutral's: a
    # natural 15N line measures the neutral; the label's own atoms ('^N', seen by a
    # '14N' impurity line) are the reagent's
    assert "N" in EV.neutral_elements("C10H17NO7", "C10H17N^NO10")
    assert "^N" not in EV.neutral_elements("C10H17NO7", "C10H17N^NO10")
    assert "^N" in EV.neutral_elements("C10H15^NO7", "C10H14^NO7")    # a labelled product, deprotonated


def test_15N_and_18O_are_expected_per_atom_of_the_ion():
    assert abs(EV.expected_ratio("18O", "C11H21N2O5") - 5 * O18) < 1e-12
    assert abs(EV.expected_ratio("15N", "C11H21N2O5") - 2 * N15) < 1e-12
    # a '^N' atom is already 15N: only the 14N atoms give a 15N satellite
    assert abs(EV.expected_ratio("15N", "C10H17N^NO10") - 1 * N15) < 1e-12
    # unchanged: 13C per carbon, the tabulated halogen / S / Si satellites, nothing for 'M'
    assert abs(EV.expected_ratio("13C", "C10H15O4") - 10 * EV.C13_PER_CARBON) < 1e-12
    assert EV.expected_ratio("81Br", "C8H14BrO2") == EV.ISOTOPE_ABUNDANCE["81Br"]
    assert EV.expected_ratio("M", "C8H14BrO2") == 0.0


# --------------------------------------------------------------------------- the rule on a pair
def test_13C_and_13C2_are_one_element():
    rows = [m0("p", "C22H42O6", adduct="[M+H]+", ion="C22H43O6", height=1000.0, series_unit="CH2"),
            c13("c", "p", 22), child("d", "p", "13C2+2", 1000.0 * (0.0107 * 22) ** 2 / 2)]
    r = facts(rows)[("C22H42O6", "[M+H]+")]
    assert r.iso and not r.multiline and r.multiline_elements == "C"
    assert vote_of(rows)[("C22H42O6", "[M+H]+")] == 1           # iso + anchor, nothing outside


def test_a_urea_adducts_own_15N_is_not_the_neutrals():
    ion = "C11H19N2O3"                                            # C10H14O2 + urea + H
    rows = [m0("p", "C10H14O2", adduct="[M+(CH4N2O)H]+", ion=ion, height=1000.0, series_unit="CH2"),
            c13("c", "p", 11), child("n", "p", "15N+1", 1000.0 * 2 * N15)]
    r = facts(rows)[("C10H14O2", "[M+(CH4N2O)H]+")]
    assert not r.multiline and r.multiline_elements == "C"


def test_the_18O_line_of_an_O_rich_neutral_on_urea_is_the_neutrals():
    ion = "C11H21N2O5"                                            # C10H16O4 + urea + H: 4 of 5 O
    rows = [m0("p", "C10H16O4", adduct="[M+(CH4N2O)H]+", ion=ion, height=1000.0, series_unit="CH2"),
            c13("c", "p", 11), child("o", "p", "18O+2", 1000.0 * 5 * O18)]
    r = facts(rows)[("C10H16O4", "[M+(CH4N2O)H]+")]
    assert r.multiline and r.multiline_elements == "C|O"
    assert vote_of(rows)[("C10H16O4", "[M+(CH4N2O)H]+")] == 2    # two elements: the neutral backed


def test_a_line_outside_its_ratio_band_counts_for_nothing():
    """The size check: an '18O' line five times its expected height is another peak."""
    ion = "C11H21N2O5"
    rows = [m0("p", "C10H16O4", adduct="[M+(CH4N2O)H]+", ion=ion, height=1000.0, series_unit="CH2"),
            c13("c", "p", 11), child("o", "p", "18O+2", 1000.0 * 5 * O18 * 5)]
    r = facts(rows)[("C10H16O4", "[M+(CH4N2O)H]+")]
    assert not r.multiline and r.multiline_elements == "C"


def test_formic_acids_nitrate_18O_measures_the_nitrate():
    rows = [m0("p", "CH2O2", adduct="[M+NO3]-", ion="CHNO5", height=1000.0, series_unit="CH2"),
            c13("c", "p", 1), child("o", "p", "18O+2", 1000.0 * 5 * O18)]
    r = facts(rows)[("CH2O2", "[M+NO3]-")]
    assert not r.multiline and r.multiline_elements == "C"       # its one C counts; 2 of 5 O do not


def test_an_M_plus_4_child_on_a_bromide_adduct_is_no_line():
    """The audit's golden: a bromide adduct with the reagent's 81Br line and a
    generic 'M+4' child held 'two lines' (4a); neither speaks for the neutral."""
    rows = [m0("p", "C8H14O2", adduct="[M+Br]-", ion="C8H14BrO2", height=1000.0, series_unit="CH2"),
            c13("c", "p", 8), child("b", "p", "81Br+2", 972.8), child("m", "p", "M+4", 40.0)]
    r = facts(rows)[("C8H14O2", "[M+Br]-")]
    assert r.iso and not r.multiline and r.multiline_elements == "C"
    assert vote_of(rows)[("C8H14O2", "[M+Br]-")] == 1


def test_an_18O_line_of_a_bromide_or_chloride_ion_is_not_measured():
    """The halogen owns the ion's M+2 region: an '18O' line there is the 81Br /
    37Cl line's shoulder, or -- with no halogen line at all -- evidence against
    the formula. It gives neither the isotope axis nor an element."""
    for adduct, ion in (("[M+Br]-", "C16H12BrO9"), ("[M+Cl]-", "C16H12ClO9")):
        rows = [m0("p", "C16H12O9", adduct=adduct, ion=ion, height=1000.0, series_unit="CH2"),
                c13("c", "p", 16), child("o", "p", "18O+2", 1000.0 * 9 * O18)]
        r = facts(rows)[("C16H12O9", adduct)]
        assert r.multiline_elements == "C" and not r.multiline, adduct
        assert EV.expected_ratio("18O", ion) == 0.0
        rows = [m0("p", "C16H12O9", adduct=adduct, ion=ion, height=1000.0),
                child("o", "p", "18O+2", 1000.0 * 9 * O18)]
        assert not facts(rows)[("C16H12O9", adduct)].iso, adduct
    assert EV.expected_ratio("18O", "C16H11O9") > 0                 # the same neutral deprotonated: measured


def test_an_18O_line_alone_now_gives_the_isotope_axis():
    rows = [m0("p", "C10H16O8", adduct="[M-H]-", ion="C10H15O8", height=1000.0),
            child("o", "p", "18O+2", 1000.0 * 8 * O18)]
    r = facts(rows)[("C10H16O8", "[M-H]-")]
    assert r.iso and not r.multiline
    rows[1] = child("o", "p", "18O+2", 1000.0 * 8 * O18 * 3)       # out of band
    assert not facts(rows)[("C10H16O8", "[M-H]-")].iso


def test_the_two_elements_are_recorded():
    """The facts string lists `multiline` and the fact names the two elements."""
    ion = "C11H21N2O5"
    rows = [m0("p", "C10H16O4", adduct="[M+(CH4N2O)H]+", ion=ion, height=1000.0, series_unit="CH2"),
            c13("c", "p", 11), child("o", "p", "18O+2", 1000.0 * 5 * O18)]
    out = EV._series_pooled({"f": ledger(rows)})
    assert out.multiline_elements.iloc[0] == "C|O" and "multiline" in out.evidence_axes.iloc[0].split("|")


def test_the_reference_script_agrees_on_each_case():
    """scripts/level_ledger.py implements the same rule on its own code."""
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location(
        "level_ledger", Path(__file__).resolve().parents[1] / "scripts" / "level_ledger.py")
    LL = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(LL)
    cases = [
        ("C10H16O4", "C11H21N2O5", "18O", 5 * O18, True),
        ("C10H16O4", "C11H21N2O5", "18O", 25 * O18, False),
        ("C10H14O2", "C11H19N2O3", "15N", 2 * N15, False),
        ("CH2O2", "CHNO5", "18O", 5 * O18, False),
    ]
    for neutral, ion, tag, ratio, want in cases:
        adduct = "[M+NO3]-" if ion == "CHNO5" else "[M+(CH4N2O)H]+"
        n_c = EV.count_element(ion, "C")
        rows = [m0("p", neutral, adduct=adduct, ion=ion, height=1000.0, series_unit="CH2"),
                c13("c", "p", n_c), child("x", "p", f"{tag}+2", 1000.0 * ratio)]
        frame = ledger(rows).assign(__file="f")
        got = LL.measure_source("s", frame, None).set_index(["neutral", "adduct"]).loc[(neutral, adduct)]
        core = facts(rows)[(neutral, adduct)]
        assert bool(got.multiline) == bool(core.multiline) == want, (neutral, tag)
        assert got.multiline_elements == core.multiline_elements
