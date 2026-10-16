"""Offline tests for the MS2 fragment-assignment core (peaky/ms2/subformula.py):
ion parsing, electron parity, the neutral-loss decomposition, the per-parent
sub-formula candidates and their ranking, the constant-ppm recalibration, and
the long-format window table.

Measured cases are our own lab's: (A) the 2026-09-10 exploratory 15N-ammonium
file, energy-averaged MS2 of seven precursors, re-read here from the readings
the exploratory script of that day committed; (B) two monoterpene standards
measured in 15N-ammonium mode on 2026-09-11. Every number is inlined (m/z on
the precursor-recalibrated axis, rounded to 0.1 mDa as recorded). (C) is
synthetic exact-mass chemistry."""
import functools
import itertools
import time

import numpy as np
import pandas as pd
import pytest

from peaky.chem import chemistry as C
from peaky.ms2 import subformula as S
from peaky.ms2 import (assign_fragments, best_readings, decompose_loss,
                       fragment_candidates, mass_offset, apply_offset, parse_ion,
                       format_ion, ion_parity, fragment_ok)


def _mz(ion: str) -> float:
    counts, charge = parse_ion(ion)
    return C.ion_mass(counts, charge)


def _top(df, peak, parent):
    rows = df[(df["peak"] == peak) & (df["parent"] == parent)]
    return rows.iloc[0] if len(rows) else None


# ------------------------------------------------------------------ (A) ----
# 2026-09-10 exploratory file, positive mode, ^NH4+ reagent, energy-averaged
# MS2 of seven precursors. Per precursor: (m/z, top formula, loss formula) as
# that day's script read it, or (m/z, None, None) where it read nothing from
# that precursor. Its loss NAMES were trivial names ('water + 15N-ammonia');
# the core writes formula decompositions instead, so only formulas are
# compared, plus the decompositions spelled out in test_a_loss_decompositions.
FILE_0910 = {
    "C7H6NS+": [
        (136.0215, "C7H6NS+", ""), (88.0215, None, None), (109.0106, "C6H5S+", "CHN"),
        (136.0249, None, None), (95.0491, None, None), (118.0579, None, None),
        (65.0385, "C5H5+", "C2HNS"), (105.0447, None, None), (136.0816, None, None),
        (136.0757, None, None), (100.0474, None, None),
    ],
    "C4H10^NO4+": [
        (137.0575, "C4H10^NO4+", ""), (137.0597, None, None), (137.0249, None, None),
        (119.0702, None, None), (119.0338, "C4H7O4+", "H3^N"), (137.0939, None, None),
        (137.1302, None, None), (137.0727, None, None), (119.1066, None, None),
        (119.0468, "C4H8^NO3+", "H2O"), (119.0855, None, None), (101.0596, None, None),
        (137.1325, None, None), (137.0961, None, None), (137.0209, None, None),
    ],
    "C11H13O+": [
        (161.0961, "C11H13O+", ""), (43.0177, "C2H3O+", "C9H10"),
        (133.0647, "C9H9O+", "C2H4"), (143.0702, None, None), (143.1066, None, None),
        (161.0941, None, None), (161.1302, None, None), (161.1325, None, None),
        (119.0855, "C9H11+", "C2H2O"), (125.0960, None, None), (161.0808, None, None),
        (125.0596, None, None), (161.1006, None, None), (161.0916, None, None),
        (143.0855, "C11H11+", "H2O"),
    ],
    "C11H15O2+": [
        (179.1067, "C11H15O2+", ""), (161.0960, "C11H13O+", "H2O"),
        (59.0490, "C3H7O+", "C8H8O"), (119.0854, "C9H11+", "C2H4O2"),
        (121.0647, "C8H9O+", "C3H6O"), (179.0636, None, None),
        (43.0177, "C2H3O+", "C9H12O"), (133.0647, "C9H9O+", "C2H6O"),
        (91.0541, "C7H7+", "C4H8O2"), (161.0808, None, None), (179.1340, None, None),
        (179.1014, None, None), (179.1195, None, None), (143.0702, None, None),
        (179.0703, None, None), (179.1119, None, None), (179.1000, None, None),
        (179.1087, None, None), (179.1317, None, None), (179.1429, None, None),
        (179.0388, None, None), (161.0917, None, None), (43.0171, None, None),
        (179.0855, None, None),
    ],
    "C11H18^NO2+": [
        (179.1066, "C11H15O2+", "H3^N"), (197.1302, "C11H18^NO2+", ""),
        (161.0961, "C11H13O+", "H5^NO"), (179.1196, "C11H16^NO+", "H2O"),
        (197.0961, None, None), (197.1536, None, None), (197.1173, None, None),
        (197.0809, None, None), (197.1900, None, None),
    ],
    "C12H22^NO2+": [
        (177.1274, "C12H17O+", "H5^NO"), (213.1615, "C12H22^NO2+", ""),
        (195.1510, "C12H20^NO+", "H2O"), (177.1404, "C12H18^N+", "H4O2"),
        (213.1849, None, None), (213.1486, None, None), (213.1122, None, None),
        (159.1168, "C12H15+", "H7^NO2"),
    ],
    "C8H28^NO4Si4+": [
        (297.0825, "C8H25O4Si4+", "H3^N"), (315.1060, "C8H28^NO4Si4+", ""),
        # 4.8 ppm low on the recalibrated axis: inside 5 ppm, as it was inside
        # the 6 ppm of that day's script
        (281.0498, "C7H21O4Si4+", "CH7^N"),
    ],
}


@pytest.mark.parametrize("precursor", list(FILE_0910))
def test_a_every_reading_of_the_0910_file_is_reproduced(precursor):
    rows = FILE_0910[precursor]
    df = assign_fragments([mz for mz, _, _ in rows], {"target": precursor})
    assert sorted(df["peak"].unique()) == list(range(len(rows)))
    for i, (mz, formula, loss) in enumerate(rows):
        top = _top(df, i, "target")
        if formula is None:
            # read nothing then -> no reading from this precursor now either
            assert top is None, (mz, top.to_dict())
            continue
        assert top is not None, (mz, formula)
        assert (top["formula"], top["loss"]) == (formula, loss), (mz, top.to_dict())
        assert top["rank"] == 1 and top["parity"] == "EE"
        assert top["is_precursor"] == (loss == "")
        if loss == "":
            # the precursor reading: no loss, no pieces
            assert (top["loss_name"], top["loss_pieces"], top["unnamed_pieces"]) == ("", 0, 0)
        else:
            assert top["loss_pieces"] >= 1
        assert top["parent_ion"] == precursor
        assert abs(top["ppm"]) < 5.0


def test_a_loss_decompositions_are_formulas_in_library_order():
    want = {
        (197.1302, "C11H18^NO2+", 161.0961): ("H2O + H3^N", 2, 0),
        (213.1615, "C12H22^NO2+", 159.1168): ("H2O + H2O + H3^N", 3, 0),
        (213.1615, "C12H22^NO2+", 177.1404): ("H2O + H2O", 2, 0),
        (315.1060, "C8H28^NO4Si4+", 281.0498): ("H3^N + CH4", 2, 0),
        (136.0215, "C7H6NS+", 109.0106): ("CHN", 1, 0),
        # one unnamed piece beats two named ones: fewest pieces first
        (179.1067, "C11H15O2+", 43.0177): ("C9H12O", 1, 1),
        (179.1067, "C11H15O2+", 91.0541): ("C4H8O2", 1, 1),
        (136.0215, "C7H6NS+", 65.0385): ("C2HNS", 1, 1),
    }
    for (_, parent, mz), (name, k, unnamed) in want.items():
        top = fragment_candidates(parent, mz)[0]
        assert (top.loss_name, top.loss_pieces, top.unnamed_pieces) == (name, k, unnamed), \
            (parent, mz, top)


def test_a_bare_carbon_loss_is_not_a_reading_the_co_isolate_is():
    """88.0215 = C3H6NS+ fits C7H6NS+ minus C4 at -0.5 ppm, but no neutral
    C4 leaves an ion: the 0.4 Da window also held C4H10NS2+ (136.0248, 13 % of
    the window signal), and that ion minus CH4S IS C3H6NS+."""
    window = {"target": "C7H6NS+", "co1": "C4H10NS2+", "co2": "C8H10NO+"}
    rows = FILE_0910["C7H6NS+"]
    df = assign_fragments([mz for mz, _, _ in rows], window)
    p88 = df[df["peak"] == 1]
    assert list(p88["parent"]) == ["co1"]
    r = p88.iloc[0]
    assert (r["formula"], r["loss"], r["loss_name"], r["n_parents"]) == ("C3H6NS+", "CH4S", "CH4S", 1)
    assert decompose_loss("C4") is None
    # the window ions are their own precursor readings
    assert _top(df, 3, "co1")["is_precursor"] and _top(df, 9, "co2")["is_precursor"]
    # 65.0385 = C5H5+ is a sub-formula of the target AND of C8H10NO+: both kept
    p65 = df[df["peak"] == 6]
    assert list(p65["parent"]) == ["target", "co2"] and set(p65["n_parents"]) == {2}
    assert list(p65["loss"]) == ["C2HNS", "C3H5NO"]


def test_a_an_ammonium_cluster_of_a_saturated_molecule_is_its_own_precursor():
    """C6H14O2 + ^NH4+ = C6H18^NO2+ has ion DBE -1.5, below the fragment gate,
    yet it was a co-isolate of the 137.057 window (137.1302, 10 % of the
    signal). A parent is an input: it is never gated, and its -^NH3 product
    (C6H15O2+, DBE -0.5) is an ordinary fragment."""
    assert not fragment_ok("C6H18^NO2+")
    window = {"target": "C4H10^NO4+", "C6": "C6H18^NO2+"}
    df = assign_fragments([137.1302, 119.1066], window)
    assert _top(df, 0, "C6")["is_precursor"]
    r = _top(df, 1, "C6")
    assert (r["formula"], r["loss"]) == ("C6H15O2+", "H3^N")
    assert _top(df, 1, "target") is None


# ------------------------------------------------------------------ (B) ----
# 2026-09-11, two monoterpene standards in 15N-ammonium mode. [M+H]+ C10H17+
# energy-averaged MS2 (first standard), and the [M+^NH4]+ window at m/z 155
# (C10H20^N+ plus the co-isolated C10H19O+ at 155.143).
MT_137 = [  # (m/z, fragment, loss name) -- None: a window ion, never a fragment
    (137.1325, "C10H17+", ""), (81.0698, "C6H9+", "C4H8"), (95.0854, "C7H11+", "C3H6"),
    (137.0586, None, None), (137.0246, None, None), (119.0701, None, None),
    (119.0338, None, None), (67.0541, "C5H7+", "C5H10"), (101.0596, None, None),
    (119.1066, None, None), (57.0698, "C4H9+", "C6H8"), (119.0468, None, None),
    (119.0854, "C9H11+", "H2 + CH4"), (101.0232, None, None), (69.0697, "C5H9+", "C5H8"),
]


def test_b_monoterpene_protonated_fragments():
    df = assign_fragments(pd.DataFrame({"mz": [m for m, _, _ in MT_137],
                                        "intensity": np.arange(len(MT_137), dtype=float)}),
                          "C10H17+")
    assert df["parent"].isin(["C10H17+", ""]).all()
    for i, (mz, formula, name) in enumerate(MT_137):
        top = _top(df, i, "C10H17+")
        if formula is None:
            assert top is None, (mz, top.to_dict())
            row = df[df["peak"] == i].iloc[0]
            assert (row["n_candidates"], row["n_parents"], row["formula"]) == (0, 0, "")
            continue
        assert (top["formula"], top["loss_name"]) == (formula, name), (mz, top.to_dict())
        assert abs(top["ppm"]) < 3.0 and not top["radical_loss"]
        assert top["intensity"] == float(i)
    # C4H8 / C3H6 are library pieces; C5H10 / C6H8 are single unnamed pieces;
    # CH6 (DBE -1) needs two molecules
    r = _top(df, 12, "C10H17+")
    assert (r["loss"], r["loss_pieces"], r["unnamed_pieces"]) == ("CH6", 2, 0)
    for i, unnamed in ((1, 0), (2, 0), (7, 1), (10, 1)):
        assert _top(df, i, "C10H17+")["unnamed_pieces"] == unnamed, MT_137[i]


def test_b_monoterpene_ammonium_adduct_window():
    window = {"adduct": "C10H20^N+", "water cluster": "C10H19O+"}
    peaks = [155.15606, 137.13246, 155.14299, 81.06979, 95.08546, 109.10112,
             155.07026, 155.10663]
    df = assign_fragments(peaks, window)
    r = _top(df, 1, "adduct")
    assert (r["formula"], r["loss"], r["loss_name"]) == ("C10H17+", "H3^N", "H3^N")
    assert _top(df, 1, "water cluster")["loss"] == "H2O"
    # 137 / 81 / 95 / 109 are sub-formulas of both window ions: composition
    # cannot attribute them, and both readings are kept
    for i in (1, 3, 4, 5):
        assert set(df.loc[df["peak"] == i, "n_parents"]) == {2}
    assert _top(df, 0, "adduct")["is_precursor"] and _top(df, 2, "water cluster")["is_precursor"]
    for i in (6, 7):                                  # unread window ions
        assert (df["peak"] == i).sum() == 1 and _top(df, i, "") is not None


# ------------------------------------------------------------------ (C) ----
def test_ion_parsing_and_formatting():
    assert parse_ion("C10H17O+") == ({"C": 10, "H": 17, "O": 1}, 1)
    assert parse_ion("C10H17O5-") == ({"C": 10, "H": 17, "O": 5}, -1)
    assert parse_ion("C4H10^NO4+") == ({"C": 4, "H": 10, "^N": 1, "O": 4}, 1)
    assert parse_ion("C7H8+.") == parse_ion("C7H8+•") == ({"C": 7, "H": 8}, 1)
    assert format_ion({"C": 6, "H": 9}, 1) == "C6H9+"
    assert format_ion({"^N": 1, "O": 3}, -1) == "^NO3-"
    for bad in ("C10H17O", "C10H17O++", "C10H17O2+2", "2+", "+", "", "C9Xx+",
                "C7H7+.",           # radical mark on an even-electron ion
                "c10h17o+"):
        with pytest.raises(ValueError):
            parse_ion(bad)
    with pytest.raises(ValueError):
        format_ion({"C": 6, "H": 9}, 2)


def test_ion_mass_and_parity():
    assert C.ion_mass("^NO3", -1) == pytest.approx(62.98540, abs=1e-5)
    assert C.ion_mass("CH5N2O", 1) == pytest.approx(61.03964, abs=1e-5)
    assert C.ion_mass({"C": 2, "H": 3, "O": 1}, 1) == pytest.approx(43.01784, abs=1e-5)
    with pytest.raises(ValueError):
        C.ion_mass("C2H3O", 2)
    assert ion_parity("C10H17O+") == "EE"          # DBE 2.5
    assert ion_parity("C10H17O5-") == "EE"         # 2.5
    assert ion_parity("C7H8+.") == "OE"            # 4.0
    # the NEUTRAL test reads the same half-integer DBE the other way round
    assert C.odd_electron("C10H17O") and ion_parity("C10H17O+") == "EE"
    assert fragment_ok("H3O+") and fragment_ok("H4N+")        # DBE -0.5
    assert not fragment_ok("H3+") and not fragment_ok("C10H40+")
    assert not fragment_ok("P2+")                  # DBE 2 > Senior cap 1 + 1/2 (P adds no cap)
    assert not fragment_ok("H6^NO+")               # ^NH4+.H2O, DBE -1.5
    assert fragment_ok("C2H7Si2+")                # Si counts in the cap
    with pytest.raises(ValueError):
        fragment_ok({"C": 2, "O": -1})


def _all_subformulas(ion: str) -> set[str]:
    """Every element-wise sub-formula of an ion, by plain iteration (no numpy)."""
    counts, charge = parse_ion(ion)
    els = list(counts)
    out = set()
    for sub in itertools.product(*[range(counts[e] + 1) for e in els]):
        c = {e: n for e, n in zip(els, sub) if n}
        if c:
            out.add(format_ion(c, charge))
    return out


def test_fragment_gate_is_chemistry_and_the_grid_uses_it():
    """fragment_ok, read straight off chemistry.dbe / chemistry.seniors_cap on
    random formulas; and every parent's sub-formula grid is exactly the
    sub-formulas fragment_ok accepts, plus the parent itself."""
    rng = np.random.default_rng(7)
    els = list(C.M)
    for _ in range(3000):
        counts = {e: int(rng.integers(0, 4)) for e in els if rng.random() < 0.35}
        counts["H"] = int(rng.integers(0, 24))
        counts = {e: n for e, n in counts.items() if n}
        if not counts:
            continue
        d = C.dbe(counts)
        ee = d != int(d)
        want = (sum(n for e, n in counts.items() if e != "H") > 0
                and d >= (-0.5 if ee else 0.0) and d <= C.seniors_cap(counts) + 0.5)
        assert fragment_ok(counts) == want, counts
        assert ion_parity(counts) == ("EE" if ee else "OE")
    for parent in ("C6H18^NO2+", "C8H28^NO4Si4+", "C7H8+.", "C4H10NS2+", "C4H11O4P2+",
                   "C3H9ClOSi2+", "C5H9^NO3-", "C10H18^NO7-", "C4H6BrF2IO+", "C10H16NO7-"):
        counts, charge = parse_ion(parent)
        grid = S._parent_grid(S._ion_key(counts), charge)
        got = {format_ion({e: int(n) for e, n in zip(grid.els, row) if n}, charge)
               for row in grid.sub}
        want = {f for f in _all_subformulas(parent) if fragment_ok(f)} | {format_ion(counts, charge)}
        assert got == want, parent
        assert len(got) == len(grid.sub)                       # no duplicate rows
        assert np.array_equal(grid.t2, [round(2 * C.dbe(f[:-1])) for f in
                                        (format_ion({e: int(n) for e, n in zip(grid.els, row) if n},
                                                    charge) for row in grid.sub)])
    # the gate removes fragments, never the parent: the cluster's own
    # ammonium-water fragment (DBE -1.5) is not a reading
    assert fragment_candidates("C6H18^NO2+", _mz("H6^NO+")) == []
    assert [c.formula for c in fragment_candidates("C6H18^NO2+", _mz("C6H18^NO2+"))] == ["C6H18^NO2+"]


def test_loss_libraries_are_hill_formulas_of_the_right_parity():
    for f in S.CLOSED_SHELL_LOSSES:
        assert C.format_formula(C.parse_formula(f)) == f
        assert C.dbe(f) >= 0 and C.dbe(f) == int(C.dbe(f)), f
    for f in S.RADICAL_LOSSES:
        assert C.format_formula(C.parse_formula(f)) == f
        assert C.dbe(f) >= 0.5 and C.dbe(f) != int(C.dbe(f)), f
    assert len(set(S.CLOSED_SHELL_LOSSES + S.RADICAL_LOSSES)) == \
        len(S.CLOSED_SHELL_LOSSES) + len(S.RADICAL_LOSSES)


def test_loss_libraries_are_the_documented_ones_and_each_entry_is_named():
    """The lists docs/MS2.md section 5 prints, in priority order. Dropping an
    entry would not change a loss FORMULA -- most entries also pass as one
    unnamed piece -- but it changes `unnamed_pieces`, which ranks."""
    assert S.CLOSED_SHELL_LOSSES == (
        "H2O", "H3N", "H3^N", "CO", "CO2", "H2", "CH4", "C2H4", "CH2O", "CH4O",
        "CH2O2", "C2H2O", "C2H4O2", "CHN", "CHNO", "CH4N2O", "H2O2", "HNO2", "H^NO2",
        "HNO3", "H^NO3", "H2S", "CH4S", "O2S", "O3S", "HF", "HCl", "HBr", "HI",
        "N2", "O2", "C2H2", "C3H6", "C4H8", "C3H6O", "C2H4O", "C5H8",
        "C2H6O", "H2O4S", "C2H6OSi", "CS2", "COS", "N2O")
    assert S.RADICAL_LOSSES == (
        "H", "CH3", "HO", "HO2", "NO", "^NO", "NO2", "^NO2", "NO3", "^NO3",
        "CHO", "CH3O", "C2H5", "Cl", "Br", "I")
    for f in S.CLOSED_SHELL_LOSSES + S.RADICAL_LOSSES:
        d = decompose_loss(f, k_max=1)
        assert (d.pieces, d.unnamed, d.radical) == ((f,), 0, f in S.RADICAL_LOSSES), f
    # priority breaks size ties: H4O2 is two waters, not H2 + H2O2
    assert decompose_loss("H4O2").pieces == ("H2O", "H2O")


def test_h_free_library_losses_read():
    """CS2, COS and N2O have no H or halogen, so only the library lets them
    read: a protonated methyl dimethyldithiocarbamate (C4H10NS2+, the
    2026-09-10 file's 136.025 co-isolate) can lose CS2 to C3H10N+, its
    thiocarbamate analogue COS, and a C2H7N2O2+ ion N2O."""
    for loss in ("CS2", "COS", "N2O"):
        d = decompose_loss(loss)
        assert (d.pieces, d.unnamed) == ((loss,), 0), loss
    for parent, frag, loss in (("C4H10NS2+", "C3H10N+", "CS2"),
                               ("C4H10NOS+", "C3H10N+", "COS"),
                               ("C2H7N2O2+", "C2H7O+", "N2O")):
        top = fragment_candidates(parent, _mz(frag))[0]
        assert (top.formula, top.loss, top.loss_name, top.unnamed_pieces) == \
            (frag, loss, loss, 0), parent
    # still no bare-skeleton pieces: CS (one C, one S) is neither named nor H-bearing
    assert decompose_loss("CS") is None


def test_sequential_loss_is_accepted_as_two_molecules():
    # protonated C10H16O losing water and then H2: H4O has DBE -1, so k >= 2
    top = fragment_candidates("C10H17O+", _mz("C10H13+"))[0]
    assert (top.formula, top.loss, top.loss_name, top.loss_pieces) == \
        ("C10H13+", "H4O", "H2O + H2", 2)
    assert (top.pieces, top.unnamed_pieces, top.is_precursor) == (("H2O", "H2"), 0, False)
    # the precursor reading carries no loss and no pieces at all
    pre = fragment_candidates("C10H17O+", _mz("C10H17O+"))
    assert [(c.is_precursor, c.loss, c.loss_name, c.loss_pieces, c.unnamed_pieces, c.pieces,
             c.radical_loss) for c in pre] == [(True, "", "", 0, 0, (), False)]


def test_minimal_k_bound_and_k_max():
    assert decompose_loss("H8") is None                       # DBE -3 -> k >= 4
    d = decompose_loss("H8", k_max=4)
    assert d.pieces == ("H2", "H2", "H2", "H2") and d.unnamed == 0
    assert fragment_candidates("C9H17O+", _mz("C9H9O+")) == []
    four = fragment_candidates("C9H17O+", _mz("C9H9O+"), k_max=4)
    assert [c.loss_name for c in four] == ["H2 + H2 + H2 + H2"]
    assert decompose_loss("H5^NO").name == "H2O + H3^N"
    assert decompose_loss("H7^NO2").k == 3
    for bad in (0, S.K_MAX_LIMIT + 1, 2.0, True):
        with pytest.raises(ValueError):
            decompose_loss("H2O", k_max=bad)


def test_a_long_dehydration_chain_reads_when_k_max_allows_it():
    """A hexose's [M+^NH4]+ (C6H16^NO6+) declusters and dehydrates step by
    step. -^NH3 -4 H2O (C6H5O2+, loss H11^NO4, DBE -4) needs five molecules
    and -^NH3 -5 H2O (C6H3O+, H13^NO5, DBE -5) six: no library can do it in
    fewer, so the default k_max 3 refuses both and an explicit k_max reads them."""
    p = "C6H16^NO6+"
    for frag, k in (("C6H13O6+", 1), ("C6H11O5+", 2), ("C6H9O4+", 3), ("C6H7O3+", 4),
                    ("C6H5O2+", 5), ("C6H3O+", 6)):
        for k_max in range(1, S.K_MAX_LIMIT + 1):
            got = fragment_candidates(p, _mz(frag), k_max=k_max)
            if k_max < k:
                assert got == [], (frag, k_max)
            else:
                assert [(c.formula, c.loss_name) for c in got] == \
                    [(frag, " + ".join(["H2O"] * (k - 1) + ["H3^N"]))], (frag, k_max)
    assert decompose_loss("H11^NO4", k_max=5).pieces == ("H2O",) * 4 + ("H3^N",)
    # the cap itself, by value: docs/MS2.md promises k_max=6 reads the whole
    # -^NH3 -5 H2O chain, and 7 is refused
    assert S.K_MAX_LIMIT == 6
    six = fragment_candidates(p, _mz("C6H3O+"), k_max=6)
    assert [(c.formula, c.loss, c.loss_name, c.loss_pieces, c.unnamed_pieces) for c in six] == \
        [("C6H3O+", "H13^NO5", "H2O + H2O + H2O + H2O + H2O + H3^N", 6, 0)]
    with pytest.raises(ValueError):
        fragment_candidates(p, _mz("C6H3O+"), k_max=7)


def test_bare_skeleton_remainders_are_rejected_named_ones_are_not():
    for loss in ("C4", "C", "O", "C2O", "C3", "O3"):
        assert decompose_loss(loss) is None, loss
    assert decompose_loss("NH") is None                       # Senior rule 2: 4 < 6
    # CO / CO2 are library pieces, so a bare-skeleton loss stays possible
    assert decompose_loss("CO").pieces == ("CO",)
    assert decompose_loss("CO2").pieces == ("CO2",)
    assert fragment_candidates("C10H19O+", _mz("C10H19+")) == []        # -O
    co = fragment_candidates("C10H19O+", _mz("C9H19+"))[0]
    assert (co.loss, co.loss_name, co.unnamed_pieces) == ("CO", "CO", 0)
    # one unnamed piece on its own
    d = decompose_loss("C11H20O3")
    assert d.pieces == ("C11H20O3",) and d.unnamed == 1
    # library pieces PLUS one unnamed remainder: C5H14O (DBE -1) needs two
    # molecules, and H2O + C5H12 (k 2) beats the all-named H2O + CH4 + C4H8 (k 3)
    for loss, pieces in (("C5H14O", ("H2O", "C5H12")), ("C6H16O", ("H2O", "C6H14"))):
        d = decompose_loss(loss)
        assert (d.pieces, d.unnamed, d.radical) == (pieces, 1, False), loss
    # a remainder that fails its rules is not taken: H6O (DBE -2) is not
    # H2O + H4 (H4 has DBE -1) but the all-named H2O + H2 + H2
    d = decompose_loss("H6O")
    assert (d.pieces, d.unnamed) == (("H2O", "H2", "H2"), 0)


def test_halogen_only_losses_are_one_unnamed_molecule():
    """The H-OR-HALOGEN clause, halogen side: I2 and Br2 are no library pieces
    (HI, HBr, I, Br are), so they read only as one unnamed closed-shell piece.
    That is the iodide reagent's declustering: I3- -> I- and an [M-H+I2]-
    adduct -> [M-H]-, both by losing I2."""
    for loss in ("I2", "Br2", "BrI"):
        d = decompose_loss(loss)
        assert (d.pieces, d.unnamed, d.radical) == ((loss,), 1, False), loss
    for parent, frag, loss in (("I3-", "I-", "I2"), ("Br3-", "Br-", "Br2"),
                               ("C5H9I2O2-", "C5H9O2-", "I2")):     # [M-H+I2]- of C5H10O2
        got = fragment_candidates(parent, _mz(frag))
        assert [(c.formula, c.loss, c.loss_name, c.unnamed_pieces, c.radical_loss)
                for c in got] == [(frag, loss, loss, 1, False)], parent


def test_anion_losses_and_the_15n_nitrate_cluster():
    p = "C10H17O5-"                                           # an [M-H]- ion
    w = fragment_candidates(p, _mz("C10H15O4-"))[0]
    assert (w.loss, w.loss_name, w.parity) == ("H2O", "H2O", "EE")
    c = fragment_candidates(p, _mz("C9H17O3-"))[0]
    assert (c.loss, c.loss_name) == ("CO2", "CO2")
    # C10H18O4 . 15NO3- (M = C10H18O4): the cluster releases the reagent ion ...
    cl = "C10H18^NO7-"
    r = fragment_candidates(cl, _mz("^NO3-"))[0]
    assert (r.formula, r.loss, r.loss_pieces, r.unnamed_pieces) == ("^NO3-", "C10H18O4", 1, 1)
    assert r.mz_theo == pytest.approx(62.98540, abs=1e-5)
    # ... or the analyte's [M-H]- (C10H17O4-) by losing the labelled nitric
    # acid H^NO3 ...
    mh = fragment_candidates(cl, _mz("C10H17O4-"))
    assert [(c.formula, c.loss, c.loss_name, c.loss_pieces, c.unnamed_pieces) for c in mh] == \
        [("C10H17O4-", "H^NO3", "H^NO3", 1, 0)]
    # ... and C10H17O5- = [M+O-H]- (one O more than the analyte: the
    # composition of an O-atom transfer from the nitrate) by losing the
    # labelled nitrous acid H^NO2
    a = fragment_candidates(cl, _mz("C10H17O5-"))[0]
    assert (a.formula, a.loss, a.loss_name, a.unnamed_pieces) == ("C10H17O5-", "H^NO2", "H^NO2", 0)


def test_radical_losses_are_flagged_ranked_last_and_optional():
    # EE precursor -> OE fragment by CH3 loss
    r = fragment_candidates("C10H17O5-", _mz("C9H14O5-"))
    assert len(r) == 1
    assert (r[0].loss, r[0].parity, r[0].radical_loss) == ("CH3", "OE", True)
    assert fragment_candidates("C10H17O5-", _mz("C9H14O5-"), allow_radical=False) == []
    # OE precursor -> EE fragment by H loss
    h = fragment_candidates("C7H8+.", _mz("C7H7+"))
    assert [(c.formula, c.loss, c.parity, c.radical_loss) for c in h] == \
        [("C7H7+", "H", "EE", True)]
    assert fragment_candidates("C7H8+.", _mz("C7H7+"), allow_radical=False) == []
    # ranking (a): a closed-shell reading comes before a radical one even when
    # the radical one is closer. Two readings of opposite parity at one nominal
    # mass differ by an odd number of 14N; in this 14N-only parent the cheapest
    # trade (N for CH2) is 12.6 mDa (^N for NH, 10.8 mDa, needs a parent holding
    # both N and ^N: test_parity_flipping_trades_at_one_nominal_mass), so this
    # needs a wide window: an organonitrate anion losing NO2 (radical) vs CH2O2
    # (closed).
    p, oe = "C10H16NO7-", "C10H16O5-"
    cands = fragment_candidates(p, _mz(oe), tol_ppm=0.0, tol_mda=15.0)
    assert [(c.formula, c.radical_loss) for c in cands] == \
        [("C9H14NO5-", False), (oe, True)]
    assert abs(cands[1].ppm) < abs(cands[0].ppm)
    df = assign_fragments([_mz(oe)], p, tol_ppm=0.0, tol_mda=15.0)
    assert list(df["rank"]) == [1, 2] and list(df["radical_loss"]) == [False, True]
    assert list(df["parity"]) == ["EE", "OE"] and list(df["dbe"]) == [3.5, 3.0]
    # the OE precursor's reading in the table: an EE fragment by a radical loss
    df = assign_fragments([_mz("C7H7+")], "C7H8+.")
    assert (list(df["parity"]), list(df["loss"]), list(df["radical_loss"])) == \
        (["EE"], ["H"], [True])
    df = assign_fragments([_mz(oe)], p, tol_ppm=0.0, tol_mda=15.0, allow_radical=False)
    assert list(df["formula"]) == ["C9H14NO5-"]


def test_parity_flipping_trades_at_one_nominal_mass():
    """docs/MS2.md section 8, by value: at one nominal mass, opposite-parity
    readings differ by an odd number of 14N atoms. The cheapest trades are ^N
    for NH (10.8 mDa, only in a parent holding both N and ^N) and N for CH2
    (12.6 mDa); the closer ones are multi-atom (^N2 for NO, 2.2 mDa, also
    mixed-label only; CH4N3 for C3H6O, 1.3 mDa)."""
    M = C.M
    assert (M["N"] + M["H"] - M["^N"]) * 1e3 == pytest.approx(10.79, abs=0.01)
    assert (M["C"] + 2 * M["H"] - M["N"]) * 1e3 == pytest.approx(12.58, abs=0.01)
    assert (M["N"] + M["O"] - 2 * M["^N"]) * 1e3 == pytest.approx(-2.23, abs=0.01)
    assert (2 * M["C"] + 2 * M["H"] + M["O"] - 3 * M["N"]) * 1e3 == pytest.approx(1.34, abs=0.01)
    # the ^NH4+ adduct of a C4H9NO amide holds both labels: its OE fragment
    # C3H7^N+ (radical loss) has an EE twin C3H8N+ 10.8 mDa away, and rule (a)
    # ranks the closed-shell twin first
    got = fragment_candidates("C4H13N^NO+", _mz("C3H7^N+"), tol_ppm=0.0, tol_mda=11.0)
    assert [(c.formula, c.parity, c.loss, c.loss_name, c.radical_loss) for c in got] == \
        [("C3H8N+", "EE", "CH5^NO", "CH5^NO", False),
         ("C3H7^N+", "OE", "CH6NO", "H3N + CH3O", True)]
    assert (_mz("C3H7^N+") - got[0].mz_theo) * 1e3 == pytest.approx(-10.79, abs=0.01)
    # the 14N-only adduct (NH4+): no twin inside 11 mDa, two at N for CH2 (12.6)
    p14 = "C4H13N2O+"
    assert [c.formula for c in fragment_candidates(p14, _mz("C3H7N+"), tol_ppm=0.0,
                                                   tol_mda=11.0)] == ["C3H7N+"]
    assert [(c.formula, c.parity) for c in fragment_candidates(p14, _mz("C3H7N+"), tol_ppm=0.0,
                                                               tol_mda=13.0)] == \
        [("C4H9+", "EE"), ("C2H5N2+", "EE"), ("C3H7N+", "OE")]


def test_unnamed_radical_remainders():
    """A radical loss that is no library radical: one unnamed radical piece
    (half-integer DBE >= 1/2, sum of valences >= 2 x largest - 1, H or
    halogen), or a library radical plus one unnamed closed-shell remainder."""
    def dec(f):
        d = decompose_loss(f)
        return d and (d.pieces, d.unnamed, d.radical)
    assert dec("C3H7") == (("C3H7",), 1, True)                # one unnamed radical
    assert dec("H2N") == (("H2N",), 1, True)                  # valences 5 >= 2*3 - 1
    assert dec("CH") is None                                  # valences 5 < 2*4 - 1
    assert dec("C2H7") == (("H2", "C2H5"), 0, True)           # DBE -1/2: two pieces, named
    assert dec("C3H10N") == (("H3N", "C3H7"), 1, True)        # named + unnamed radical
    assert dec("C3H10^NO") == (("H2O", "C3H8^N"), 1, True)
    # no closed library piece fits CClF4 (DBE -1/2): the library radical Cl
    # plus an unnamed CLOSED-shell remainder
    assert dec("CClF4") == (("Cl", "CF4"), 1, True)
    # ... and at k_max 2, where it is the LAST size: once Cl is held, the rest
    # CF4 is closed-shell and needs one piece, so the search must not prune it
    # with the radical bound (which would ask for two)
    d = decompose_loss("CClF4", k_max=2)
    assert (d.pieces, d.unnamed, d.radical) == (("Cl", "CF4"), 1, True)
    assert decompose_loss("CClF4", k_max=1) is None
    # the remainder rules themselves, clause by clause (the search's k-bound
    # pruning already keeps most negative-DBE rests away from them)
    rules = {  # formula: acceptable as the one unnamed piece?
        "C5H12": True, "C4": False, "H4": False, "NH": False, "CF4": True,   # closed
        "C3H7": True, "CH": False, "CH7": False, "C2H": True, "CH5": False,  # radical
        "C2O3": False, "Cl": True,
        "NO2": False,          # no H or halogen: it reads only because it is a library radical
        "I2": True, "Br2": True, "BrI": True, "CBr4": True,    # halogen, no H: closed
    }
    rows = np.array([S._vec(C.parse_formula(f)) for f in rules])
    radical = np.array([C.odd_electron(f) for f in rules])
    assert list(S._remainder_ok(rows, radical)) == list(rules.values())
    # allow_radical=False: a radical loss is no decomposition at all
    assert decompose_loss("CH3", allow_radical=False) is None
    assert decompose_loss("C3H10N", allow_radical=False) is None
    assert decompose_loss("H2O", allow_radical=False).pieces == ("H2O",)


def test_ranking_rules_b_to_e():
    """Within one (peak, parent): (a) closed-shell loss first -- see
    test_radical_losses_are_flagged_ranked_last_and_optional -- then (b) fewer
    loss pieces, (c) fewer unnamed pieces, (d) smaller |ppm|, (e) formula.
    Each case below is decided by exactly one rule against the others."""
    def read(parent, mz, **kw):
        return [(c.formula, c.loss_pieces, c.unnamed_pieces)
                for c in fragment_candidates(parent, mz, **kw)]
    # (b) at the default window: 2 pieces at -4.7 ppm before 3 pieces at 0 ppm
    assert read("C8H28^NO4Si4+", _mz("C3H3O3+")) == [("C2H7Si2+", 2, 1), ("C3H3O3+", 3, 1)]
    # (b) before (c): 1 piece with 1 unnamed before 3 named pieces at 0 ppm
    assert read("C11H18^NO2+", _mz("C7H3+"), tol_mda=10.0) == \
        [("C3H4^NO2+", 1, 1), ("C7H3+", 3, 0)]
    # (c) before (d): no unnamed piece (C2H4O2) at 362 ppm before one unnamed
    # (C2H4S) at 0 ppm
    assert read("C3H9O2S+", _mz("CH5O2+"), tol_mda=20.0) == [("CH5S+", 1, 0), ("CH5O2+", 1, 1)]
    # (d) is |ppm|, not signed: +7.8 ppm before -36 ppm
    mz = _mz("H4^NO2-") + 0.4e-3
    got = fragment_candidates("C10H18^NO7-", mz, tol_ppm=0.0, tol_mda=3.0)
    assert [(c.formula, round(c.ppm)) for c in got] == [("H4^NO2-", 8), ("C4H3-", -36)]
    # (d) at a sub-ppm gap, against the formula order: in a heteroatom-rich
    # (synthetic) anion, two one-unnamed-piece closed-shell readings 0.3 ppm
    # apart; +4.68 ppm comes first although 'C5H9NO9S-' sorts before
    # 'C9H9NO4S2-' (a |ppm| key rounded to whole ppm would flip them)
    got = fragment_candidates("C15H25N3O9S2-", 258.99906)
    assert [(c.formula, c.loss, c.loss_pieces, c.unnamed_pieces, c.radical_loss,
             round(c.ppm, 2)) for c in got] == \
        [("C9H9NO4S2-", "C6H16N2O5", 1, 1, False, 4.68),
         ("C5H9NO9S-", "C10H16N2S", 1, 1, False, -4.98)]
    # (e) the formula string, when all else ties (no real pair of formulas
    # ties on |ppm|, so the key itself)
    def cand(formula, ppm):
        return S.FragmentCandidate(formula, "EE", 0.5, 100.0, ppm, "H2O", 1, "H2O", 0,
                                   False, False, ("H2O",))
    pool = [cand("C2+", 1.0), cand("C1+", -1.0), cand("C3+", -2.0), cand("C0+", 0.5)]
    assert [c.formula for c in sorted(pool, key=S.FragmentCandidate.sort_key)] == \
        ["C0+", "C1+", "C2+", "C3+"]
    # (d) on the key itself, at a 0.1 ppm gap against the formula order
    pool = [cand("C0+", 0.35), cand("C1+", -0.25)]
    assert [c.formula for c in sorted(pool, key=S.FragmentCandidate.sort_key)] == ["C1+", "C0+"]


def test_composition_ambiguity_keeps_both_parents():
    window = {"MH": "C10H17+", "adduct": "C10H20^N+"}
    df = assign_fragments([_mz("C6H9+"), _mz("C10H17+")], window)
    p0 = df[df["peak"] == 0]
    assert list(p0["parent"]) == ["MH", "adduct"] and set(p0["n_parents"]) == {2}
    assert list(p0["loss"]) == ["C4H8", "C4H11^N"]
    p1 = df[df["peak"] == 1]
    assert list(p1["is_precursor"]) == [True, False] and list(p1["loss"]) == ["", "H3^N"]
    assert list(p1["loss_pieces"]) == [0, 1] and list(p1["unnamed_pieces"]) == [0, 0]
    assert list(p1["loss_name"]) == ["", "H3^N"]
    # a sequence of formulas works like a mapping keyed by the formulas
    df2 = assign_fragments([_mz("C6H9+")], ["C10H17+", "C10H20^N+"])
    assert list(df2["parent"]) == ["C10H17+", "C10H20^N+"]


def test_n_parents_counts_parents_and_n_candidates_counts_readings():
    """Two readings from ONE parent is not composition ambiguity: n_parents
    counts parents, n_candidates counts readings within (peak, parent)."""
    window = {"a": "C8H28^NO4Si4+", "b": "C10H17+"}
    df = assign_fragments([_mz("C3H3O3+")], window)
    assert list(df["parent"]) == ["a", "a"] and list(df["rank"]) == [1, 2]
    assert list(df["n_candidates"]) == [2, 2] and set(df["n_parents"]) == {1}
    assert list(df["formula"]) == ["C2H7Si2+", "C3H3O3+"]
    # one reading from each of two parents: n_parents 2, n_candidates 1 each
    df = assign_fragments([_mz("C6H9+")], {"MH": "C10H17+", "adduct": "C10H20^N+"})
    assert list(df["n_candidates"]) == [1, 1] and list(df["n_parents"]) == [2, 2]
    # best_readings keeps rank 1 per (peak, parent)
    best = best_readings(assign_fragments([_mz("C3H3O3+")], window))
    assert list(best["formula"]) == ["C2H7Si2+"]


def test_dbe_and_ppm_columns_are_the_ion_dbe_and_signed_error():
    # C6H9+ read 0.2 mDa LOW: ppm negative, dbe the ion's 2.5 (not the
    # neutral-style 2.0 of C6H8)
    mz = _mz("C6H9+") - 0.2e-3
    c = fragment_candidates("C10H17+", mz)[0]
    assert (c.formula, c.parity, c.dbe) == ("C6H9+", "EE", 2.5)
    assert c.ppm == pytest.approx(-0.2e-3 / _mz("C6H9+") * 1e6) and c.ppm < 0
    df = assign_fragments([mz, _mz("C10H17+") + 0.3e-3], "C10H17+")
    assert list(df["dbe"]) == [2.5, 2.5]
    assert df["ppm"].iloc[0] < 0 < df["ppm"].iloc[1]
    # an odd-electron fragment of an odd-electron precursor: integer ion DBE
    oe = fragment_candidates("C7H8+.", _mz("C5H6+") + 0.1e-3)[0]
    assert (oe.formula, oe.parity, oe.dbe, oe.loss, oe.radical_loss) == \
        ("C5H6+", "OE", 3.0, "C2H2", False)
    assert oe.ppm > 0


# The schema docs/MS2.md section 7 documents, written out here rather than read
# from S.OUTPUT_COLUMNS, so a changed declaration fails. 'str' is whatever
# pandas makes of dtype='str' (its string dtype on pandas 3, object before).
DOCUMENTED_DTYPES = {
    "peak": "int64", "mz": "float64", "mz_cal": "float64", "intensity": "float64",
    "parent": "str", "parent_ion": "str", "formula": "str", "parity": "str",
    "dbe": "float64", "ppm": "float64", "loss": "str", "loss_pieces": "int64",
    "loss_name": "str", "unnamed_pieces": "int64", "radical_loss": "bool",
    "is_precursor": "bool", "rank": "int64", "n_candidates": "int64",
    "n_parents": "int64",
}


# What a peak no parent explains carries, besides its own peak / mz / mz_cal /
# intensity and NaN dbe / ppm.
UNEXPLAINED_ROW = {
    "parent": "", "parent_ion": "", "formula": "", "parity": "", "loss": "",
    "loss_pieces": 0, "loss_name": "", "unnamed_pieces": 0, "radical_loss": False,
    "is_precursor": False, "rank": 0, "n_candidates": 0, "n_parents": 0,
}


def _assert_documented_schema(df):
    assert list(df.columns) == list(DOCUMENTED_DTYPES)
    for col, want in DOCUMENTED_DTYPES.items():
        assert df[col].dtype == pd.Series([], dtype=want).dtype, (col, df[col].dtype, want)


def test_every_peak_appears_with_stable_dtypes_and_order():
    peaks = pd.DataFrame({"mz": [_mz("C6H9+"), 250.0, _mz("C10H17+"), 40.5],
                          "intensity": [1.0, 2.0, 3.0, 4.0]},
                         index=[10, 11, 12, 13])
    df = assign_fragments(peaks, {"a": "C10H17+", "b": "C10H20^N+"})
    assert list(df.columns) == list(S.OUTPUT_COLUMNS)
    assert sorted(df["peak"].unique()) == [0, 1, 2, 3]      # positions, not labels
    assert df["peak"].is_monotonic_increasing
    un = df[df["n_candidates"] == 0]
    assert list(un["peak"]) == [1, 3]
    # the whole unexplained row, column by column (docs/MS2.md section 3.8):
    # empty text, zero counts, both flags False, NaN dbe / ppm -- so that
    # df[df.is_precursor] or df[~df.radical_loss] never picks one up as a reading
    for _, row in un.iterrows():
        assert {c: row[c] for c in UNEXPLAINED_ROW} == UNEXPLAINED_ROW, row.to_dict()
        assert np.isnan(row["dbe"]) and np.isnan(row["ppm"])
        assert row["mz_cal"] == row["mz"]                 # offset 0
    assert not df.loc[df["is_precursor"], "formula"].eq("").any()
    assert list(df["intensity"].unique()) == [1.0, 2.0, 3.0, 4.0]
    empty = assign_fragments([], "C10H17+")
    all_unread = assign_fragments([250.0], "C10H17+")
    for d in (df, empty, all_unread):
        _assert_documented_schema(d)
    best = best_readings(df)
    assert list(best["peak"]) == [0, 0, 1, 2, 2, 3]
    assert set(best["rank"]) == {0, 1}


def test_output_carries_the_input_mz_and_the_parent_ion_formula():
    """`mz` is the input as given, `mz_cal` the axis the reading is judged on
    (on unexplained rows too), `parent` the label and `parent_ion` the parent's
    normalised ion formula."""
    raw = [_mz("C6H9+") * (1 + 5e-6), 250.0]
    df = assign_fragments(pd.DataFrame({"mz": raw}), {"MH": "C10H17+"}, offset_ppm=5.0)
    assert list(df["mz"]) == raw
    assert df["mz_cal"].tolist() == pytest.approx(list(apply_offset(np.array(raw), 5.0)),
                                                  rel=0, abs=1e-9)
    assert (df["mz"] != df["mz_cal"]).all()
    # no 'intensity' column: NaN, never a measured 0 (docs/MS2.md section 7)
    assert df["intensity"].isna().all()
    assert (list(df["parent"]), list(df["parent_ion"])) == (["MH", ""], ["C10H17+", ""])
    assert df["formula"].iloc[0] == "C6H9+" and abs(df["ppm"].iloc[0]) < 1e-6
    # a radical-marked parent is reported by its formula, without the mark
    for parents, label in (({"tol": "C7H8+."}, "tol"), (["C7H8+•"], "C7H8+•"),
                           (" C7H8+. ", " C7H8+. ")):
        df = assign_fragments([_mz("C7H7+")], parents)
        assert (list(df["parent"]), list(df["parent_ion"])) == ([label], ["C7H8+"]), parents


def test_determinism_across_runs_and_cold_caches():
    peaks = [_mz(i) for i in ("C6H9+", "C7H11+", "C5H7+", "C10H17+", "C9H11+")] + [100.0]
    window = {"a": "C10H17+", "b": "C10H20^N+", "c": "C10H19O+"}
    first = assign_fragments(peaks, window)
    S._decompose.cache_clear()
    S._parent_grid.cache_clear()
    second = assign_fragments(peaks, window)
    pd.testing.assert_frame_equal(first, second)


def test_offset_recovery_on_a_synthetic_shift():
    ions = ["C10H20^N+", "H4^N+", "C10H17+"]
    shifted = {i: _mz(i) * (1 + 12.0e-6) for i in ions}
    off = mass_offset([(shifted["C10H20^N+"], "C10H20^N+"), (shifted["H4^N+"], "H4^N+")])
    assert off.offset_ppm == pytest.approx(12.0, abs=1e-6)
    assert all(abs(a.residual_ppm) < 1e-6 for a in off.anchors)
    assert list(off.to_frame().columns) == ["ion", "measured_mz", "theo_mz",
                                            "error_ppm", "residual_ppm"]
    # 12 ppm is outside the window uncorrected, and exact once applied
    assert fragment_candidates("C10H20^N+", shifted["C10H17+"]) == []
    df = assign_fragments([shifted["C10H17+"]], "C10H20^N+", offset_ppm=off.offset_ppm)
    assert df["formula"].iloc[0] == "C10H17+" and abs(df["ppm"].iloc[0]) < 1e-6
    assert df["mz_cal"].iloc[0] == pytest.approx(apply_offset(shifted["C10H17+"], 12.0))
    # median, not mean: one bad anchor of three does not move it
    med = mass_offset([(_mz("C10H17+") * (1 + 2e-6), "C10H17+"),
                       (_mz("H4^N+") * (1 + 3e-6), "H4^N+"),
                       (_mz("C10H20^N+") * (1 + 40e-6), "C10H20^N+")])
    assert med.offset_ppm == pytest.approx(3.0, abs=1e-6)
    # signs: an anchor reading HIGH has a positive error, and its residual is
    # the error re-read on the offset axis, (error - offset) / (1 + offset*1e-6):
    # the +40 ppm outlier still reads ~37 ppm high, the +2 ppm anchor ~1 ppm low
    assert [a.error_ppm for a in med.anchors] == pytest.approx([2.0, 3.0, 40.0], abs=1e-6)
    assert [a.residual_ppm for a in med.anchors] == \
        pytest.approx([(e - 3.0) / (1 + 3e-6) for e in (2.0, 3.0, 40.0)], abs=1e-6)
    assert med.anchors[0].residual_ppm < 0 < med.anchors[2].residual_ppm
    assert [a.theo_mz for a in med.anchors] == \
        pytest.approx([_mz("C10H17+"), _mz("H4^N+"), _mz("C10H20^N+")], abs=1e-9)
    # anchors are stored by their normalised ion formula (no radical mark,
    # no whitespace), in the order given
    oe = mass_offset([(_mz("C7H8+"), "C7H8+."), (_mz("C10H17+"), " C10H17+ ")])
    assert [a.ion for a in oe.anchors] == ["C7H8+", "C10H17+"]
    assert list(oe.to_frame()["ion"]) == ["C7H8+", "C10H17+"]
    with pytest.raises(ValueError):
        mass_offset([])


def test_offset_sign_two_anchor_median_and_anion_anchors():
    # a LOW-reading axis: a negative offset, which moves the m/z back UP
    lo_mz = _mz("C10H17+") * (1 - 12.0e-6)
    lo = mass_offset([(lo_mz, "C10H17+")])
    assert lo.offset_ppm == pytest.approx(-12.0, abs=1e-6)
    assert lo.anchors[0].error_ppm == pytest.approx(-12.0, abs=1e-6)
    assert abs(lo.anchors[0].residual_ppm) < 1e-6
    assert apply_offset(lo_mz, lo.offset_ppm) == pytest.approx(_mz("C10H17+"), rel=0, abs=1e-9)
    assert apply_offset(lo_mz, lo.offset_ppm) > lo_mz
    # two anchors (the precursor plus the reagent ion): the median of two is
    # their mean, and the residuals split evenly around it
    two = mass_offset([(_mz("C10H17+") * (1 + 2e-6), "C10H17+"),
                       (_mz("H4^N+") * (1 + 6e-6), "H4^N+")])
    assert two.offset_ppm == pytest.approx(4.0, abs=1e-6)
    assert [a.residual_ppm for a in two.anchors] == \
        pytest.approx([(e - 4.0) / (1 + 4e-6) for e in (2.0, 6.0)], abs=1e-6)
    mixed = mass_offset([(_mz("C10H17+") * (1 - 3e-6), "C10H17+"),
                         (_mz("H4^N+") * (1 + 5e-6), "H4^N+")])
    assert mixed.offset_ppm == pytest.approx(1.0, abs=1e-6)
    # ANION anchors (negative mode, the 15N-nitrate reagent ion and an [M-H]-):
    # the theoretical m/z is the anion's own, two electron masses above the
    # cation of the same formula (a cation mass here would read +17 ppm on ^NO3-)
    neg_ions = ("C10H17O5-", "^NO3-")
    neg = mass_offset([(_mz(i) * (1 + 12.0e-6), i) for i in neg_ions])
    assert neg.offset_ppm == pytest.approx(12.0, abs=1e-6)
    assert [a.theo_mz for a in neg.anchors] == \
        pytest.approx([C.ion_mass("C10H17O5", -1), C.ion_mass("^NO3", -1)], rel=0, abs=1e-9)
    assert neg.anchors[1].theo_mz == pytest.approx(62.98540, abs=1e-5)
    assert all(abs(a.residual_ppm) < 1e-6 for a in neg.anchors)
    assert [a.ion for a in neg.anchors] == list(neg_ions)
    exact = mass_offset([(C.ion_mass("^NO3", -1), "^NO3-")])
    assert exact.offset_ppm == pytest.approx(0.0, abs=1e-6)
    # and the corrected anion axis reads the cluster's fragments exactly
    shifted = _mz("C10H17O4-") * (1 + 12.0e-6)
    assert fragment_candidates("C10H18^NO7-", shifted) == []
    df = assign_fragments([shifted], "C10H18^NO7-", offset_ppm=neg.offset_ppm)
    assert df["formula"].iloc[0] == "C10H17O4-" and abs(df["ppm"].iloc[0]) < 1e-6


def test_one_composition_as_a_cation_and_an_anion_parent():
    """The sub-formula grid is cached per (composition, charge): the same
    C10H17O5 as a cation and as an anion parent, in one process and either
    order, reads its own fragment exactly, and the two grids differ by two
    electron masses on every row."""
    for order in (("+", "-", "+"), ("-", "+", "-")):
        S._parent_grid.cache_clear()
        theo = {}
        for sign in order:
            got = fragment_candidates("C10H17O5" + sign, _mz("C10H15O4" + sign))
            assert [(c.formula, c.loss) for c in got] == [("C10H15O4" + sign, "H2O")], order
            assert abs(got[0].ppm) < 1e-6
            theo.setdefault(sign, got[0].mz_theo)
            assert got[0].mz_theo == theo[sign]
        assert theo["-"] - theo["+"] == pytest.approx(2 * C.M_E, rel=0, abs=1e-9)
    key = S._ion_key(parse_ion("C10H17O5+")[0])
    pos, neg = S._parent_grid(key, 1), S._parent_grid(key, -1)
    assert np.array_equal(pos.sub, neg.sub) and np.array_equal(pos.t2, neg.t2)
    assert np.allclose(neg.mass - pos.mass, 2 * C.M_E, rtol=0, atol=1e-9)


def test_tolerance_floor_applies_at_low_mass_only():
    # C2H3O+ 43.0178 read 0.25 mDa low (-5.8 ppm): inside the 0.3 mDa floor,
    # outside a pure 5 ppm window (0.215 mDa at m/z 43)
    low = _mz("C2H3O+") - 0.25e-3
    assert [c.formula for c in fragment_candidates("C11H13O+", low)] == ["C2H3O+"]
    assert fragment_candidates("C11H13O+", low, tol_mda=0.0) == []
    # the default floor is 0.3 mDa, pinned from both sides and both signs
    for off in (-0.29e-3, 0.29e-3):
        assert [c.formula for c in fragment_candidates("C11H13O+", _mz("C2H3O+") + off)] == \
            ["C2H3O+"], off
    for off in (-0.31e-3, 0.31e-3):
        assert fragment_candidates("C11H13O+", _mz("C2H3O+") + off) == [], off
    # at m/z 281 the ppm term (1.4 mDa) is the window and the floor is inert
    hi = _mz("C7H21O4Si4+")
    p = "C8H28^NO4Si4+"
    assert fragment_candidates(p, hi - 1.3e-3)[0].formula == "C7H21O4Si4+"   # -4.6 ppm
    assert fragment_candidates(p, hi - 1.6e-3) == []                          # -5.7 ppm
    assert fragment_candidates(p, hi - 1.6e-3, tol_ppm=6.0)[0].formula == "C7H21O4Si4+"


def test_assign_fragments_defaults_and_arguments_reach_the_reading():
    """assign_fragments' own defaults are the documented 5 ppm / 0.3 mDa / k 3,
    and an explicit tol_ppm / tol_mda / k_max is what each reading uses."""
    def read(peaks, parent, **kw):
        return list(assign_fragments(peaks, parent, **kw)["formula"])
    # tol_ppm: -4.6 ppm reads, -5.7 ppm does not by default, does at 6 ppm
    hi = _mz("C7H21O4Si4+")
    si = "C8H28^NO4Si4+"
    assert read([hi - 1.3e-3, hi - 1.6e-3], si) == ["C7H21O4Si4+", ""]
    assert read([hi - 1.6e-3], si, tol_ppm=6.0) == ["C7H21O4Si4+"]
    assert read([hi - 1.3e-3], si, tol_ppm=4.0) == [""]
    # tol_mda: the 0.3 mDa floor, both sides, and an explicit one
    c2 = _mz("C2H3O+")
    assert read([c2 - 0.29e-3, c2 + 0.29e-3, c2 - 0.31e-3, c2 + 0.31e-3], "C11H13O+") == \
        ["C2H3O+", "C2H3O+", "", ""]
    assert read([c2 - 0.25e-3], "C11H13O+", tol_mda=0.0) == [""]
    assert read([c2 - 0.31e-3], "C11H13O+", tol_mda=0.35) == ["C2H3O+"]
    # k_max: the hexose adduct's -^NH3 -3/-4 H2O need 4 / 5 molecules
    hexose = "C6H16^NO6+"
    peaks = [_mz("C6H9O4+"), _mz("C6H7O3+"), _mz("C6H5O2+")]
    assert read(peaks, hexose) == ["C6H9O4+", "", ""]
    assert read(peaks, hexose, k_max=4) == ["C6H9O4+", "C6H7O3+", ""]
    df = assign_fragments(peaks, hexose, k_max=5)
    assert list(df["formula"]) == ["C6H9O4+", "C6H7O3+", "C6H5O2+"]
    assert list(df["loss_pieces"]) == [3, 4, 5]
    assert read(peaks, hexose, k_max=2) == ["", "", ""]


def test_bad_inputs_raise():
    with pytest.raises(ValueError):
        assign_fragments(pd.DataFrame({"m": [1.0]}), "C10H17+")
    with pytest.raises(ValueError):
        assign_fragments([float("nan")], "C10H17+")
    with pytest.raises(ValueError):
        assign_fragments([81.07], {"a": "C10H17+", "b": "C10H17O5-"})   # mixed polarity
    with pytest.raises(ValueError):
        assign_fragments([81.07], {})
    with pytest.raises(ValueError):
        fragment_candidates("C10H17+", 81.07, tol_ppm=-1)
    with pytest.raises(ValueError):
        decompose_loss("")
    nan, inf = float("nan"), float("inf")
    # tolerances: finite and >= 0, at both entry points (an infinite window
    # would read every sub-formula, a NaN one none)
    for bad in (-1.0, nan, inf, "5"):
        for kw in ({"tol_ppm": bad}, {"tol_mda": bad}):
            with pytest.raises(ValueError):
                fragment_candidates("C10H17+", 81.07, **kw)
            with pytest.raises(ValueError):
                assign_fragments([81.07], "C10H17+", **kw)
    # m/z: finite and positive, never silently 'unexplained'
    for bad in (-81.07, 0.0, nan, inf):
        with pytest.raises(ValueError):
            fragment_candidates("C10H17+", bad)
        with pytest.raises(ValueError):
            assign_fragments([81.07, bad], "C10H17+")
        with pytest.raises(ValueError):
            assign_fragments(pd.DataFrame({"mz": [bad]}), "C10H17+")
    # the axis offset: finite
    for bad in (nan, inf, -inf):
        with pytest.raises(ValueError):
            assign_fragments([81.07], "C10H17+", offset_ppm=bad)
    # empty formulas: no atoms is no ion and no loss
    for bad in ("C0+", "C0H0-"):
        with pytest.raises(ValueError, match="empty"):
            parse_ion(bad)
    with pytest.raises(ValueError):
        format_ion({}, 1)
    for bad in ({}, {"C": 0}, "C0"):
        with pytest.raises(ValueError, match="empty"):
            decompose_loss(bad)
    # one radical mark at most; surrounding whitespace is not part of the formula
    for bad in ("C7H8+..", "C7H8+.•", "C7H8+. .", "C7 H8+"):
        with pytest.raises(ValueError):
            parse_ion(bad)
    assert parse_ion(" C10H17O+ ") == parse_ion("C10H17O+") == ({"C": 10, "H": 17, "O": 1}, 1)
    assert parse_ion("\tC7H8+.\n") == ({"C": 7, "H": 8}, 1)


def test_peaks_must_be_one_dimensional():
    """(mz, intensity) pairs flattened would read every intensity as an m/z."""
    for bad in ([(81.0698, 132.0), (137.1325, 95.0854)],
                np.array([[81.0698, 132.0], [95.0854, 500.0]]),
                list(zip([81.0698, 95.0854], [57.07, 69.07]))):
        with pytest.raises(ValueError, match="1-D"):
            assign_fragments(bad, "C10H17+")
    # 1-D inputs: list, array, Series, a scalar
    for ok in ([81.0698, 95.0854], np.array([81.0698, 95.0854]),
               pd.Series([81.0698, 95.0854], index=[7, 9])):
        df = assign_fragments(ok, "C10H17+")
        assert list(df["peak"]) == [0, 1] and list(df["formula"]) == ["C6H9+", "C7H11+"]
        # an m/z array carries no intensities: NaN, never a measured 0
        assert df["intensity"].isna().all() and df["intensity"].dtype == np.float64
    scalar = assign_fragments(81.0698, "C10H17+")
    assert list(scalar["formula"]) == ["C6H9+"] and scalar["intensity"].isna().all()


def test_formula_strings_are_parsed_strictly():
    # chemistry.parse_formula skips what it cannot read; a loss string must not
    for bad in ("C2H4o2", "H2O-H2", "H2O+", "2H2O", "H2 O", "H2O,CO", "h2o", "C2H4O2x", "Xx2"):
        with pytest.raises(ValueError):
            decompose_loss(bad)
    assert decompose_loss(" C2H4O2 ").pieces == ("C2H4O2",)
    with pytest.raises(ValueError):
        decompose_loss({"C": 2, "H": -2})                   # negative count
    with pytest.raises(ValueError):
        decompose_loss({"Xx": 1})
    # the middle dot is a radical mark too, and is checked like the others
    assert parse_ion("C7H8+·") == ({"C": 7, "H": 8}, 1)
    with pytest.raises(ValueError):
        parse_ion("C7H7+·")
    # charge: exactly +1 / -1, never a bool
    for bad in (True, False, 2, 0, -2):
        with pytest.raises(ValueError):
            format_ion({"C": 6, "H": 9}, bad)


def test_parents_are_checked_ion_formula_strings():
    # a (counts, charge) tuple, None or a number is not a parent: only an ion
    # formula string goes through parse_ion's checks
    for bad in (({"C": 10, "H": 17}, 1), ({"C": 10, "H": 17, "Xx": 1}, 1),
                ({"C": 10, "H": 17}, 2), None, 137.13):
        with pytest.raises(ValueError):
            fragment_candidates(bad, 81.0698)
        with pytest.raises(ValueError):
            assign_fragments([81.0698], {"a": bad})
    with pytest.raises(ValueError):
        assign_fragments([81.0698], [({"C": 10, "H": 17}, 1)])
    # duplicate labels, also after str(): a sequence, and a mapping {1, '1'}
    with pytest.raises(ValueError, match="duplicate"):
        assign_fragments([81.0698], ["C10H17+", "C10H17+"])
    with pytest.raises(ValueError, match="duplicate"):
        assign_fragments([81.0698], {1: "C10H17+", "1": "C10H20^N+"})


def test_oversize_parent_grid_is_refused_up_front():
    big = "C100H201O50N4S2+"                       # 15.6M sub-formulas > 5M
    with pytest.raises(ValueError, match="sub-formulas"):
        fragment_candidates(big, 300.0)
    for peaks in ([], [300.0]):                   # refused with or without peaks
        with pytest.raises(ValueError, match="sub-formulas"):
            assign_fragments(peaks, {"ok": "C10H17+", "big": big})
    assert len(assign_fragments([], "C40H70O20N2S+")) == 0


def test_mass_offset_refuses_bad_anchors():
    for bad in ([(float("nan"), "C10H17+")], [(-137.1, "C10H17+")], [(0.0, "C10H17+")],
                [(137.1325, "C10H17")], [(137.1325, "C10H17+"), (81.07, "C6H9++")]):
        with pytest.raises(ValueError):
            mass_offset(bad)


def test_performance_smoke_large_parent():
    parent = "C30H51O15+"                       # a protonated C30H50O15
    fragment_candidates(parent, 300.0)          # first call builds the grid
    grid = S._parent_grid(S._ion_key(parse_ion(parent)[0]), 1)
    assert 1000 < len(grid.mass) < 31 * 52 * 16
    # every grid row is a sub-formula of the parent that passes the gate
    assert (grid.sub <= np.array([30, 51, 15])).all()
    assert np.all(np.diff(grid.mass) >= 0)
    # probe EXACT sub-formula masses, so every call reaches the loss search:
    # 40 random rows plus 20 whose loss needs >= 2 molecules (loss 2*DBE <= -1),
    # with the loss cache cold
    t2_parent = 2 + int(S._VM2[grid.col] @ np.array([30, 51, 15]))
    multi = np.flatnonzero(t2_parent - grid.t2 + 2 <= -1)
    rng = np.random.default_rng(0)
    rows = np.concatenate([rng.choice(len(grid.mass), 40, replace=False),
                           rng.choice(multi, 20, replace=False)])
    for k_max in (3, S.K_MAX_LIMIT):
        S._decompose.cache_clear()
        t0 = time.perf_counter()
        cands = [fragment_candidates(parent, float(grid.mass[r]), k_max=k_max) for r in rows]
        assert time.perf_counter() - t0 < 1.0, k_max
        assert S._decompose.cache_info().misses >= len(rows)        # the search ran
        assert sum(1 for c in cands if c) >= 15
        assert any(x.loss_pieces >= 2 for c in cands for x in c)


@functools.lru_cache(maxsize=None)
def _lib_multisets(n_lib: int, k: int):
    """Every multiset of k of the first n_lib library pieces, in lexicographic
    index order -> (indices, element sums, radical-piece count)."""
    lib = S._LIB_VEC[:n_lib].astype(np.int64)
    is_rad = (np.arange(n_lib) >= S._N_CLOSED).astype(np.int64)
    combos = list(itertools.combinations_with_replacement(range(n_lib), k))
    cs = np.array(combos, dtype=np.int64).reshape(len(combos), k)
    return cs, lib[cs].sum(axis=1), is_rad[cs].sum(axis=1)


def _brute_decompose(counts: dict, k_max: int):
    """decompose_loss by exhaustive enumeration with NO pruning: every library
    multiset of each size in lexicographic index order, all-named first, then
    named + one unnamed remainder judged by the module's own remainder rule
    (pinned clause by clause in test_unnamed_radical_remainders). The pruned
    search must give exactly this."""
    L = S._vec(counts)
    radical = bool((2 + int(L @ S._VM2)) % 2)
    n_lib = len(S._LIB_NAMES) if radical else S._N_CLOSED
    for k in range(1, k_max + 1):
        cs, tot, nrad = _lib_multisets(n_lib, k)
        hit = np.flatnonzero((tot == L).all(axis=1) & (nrad <= 1))
        if len(hit):
            return tuple(S._LIB_NAMES[i] for i in cs[hit[0]]), 0
        cs, tot, nrad = _lib_multisets(n_lib, k - 1)
        rest = L[None, :] - tot
        ok = ((rest >= 0).all(axis=1) & rest.any(axis=1) & (nrad <= 1)
              & S._remainder_ok(rest, radical & (nrad == 0)))
        hit = np.flatnonzero(ok)
        if len(hit):
            i = hit[0]
            return tuple(S._LIB_NAMES[j] for j in cs[i]) + (S._formula_of(rest[i]),), 1
    return None


def test_pruned_search_equals_exhaustive_enumeration():
    """The search drops frontier multisets whose rest needs more pieces than
    are left. That bound must never drop a valid split: compare with the
    unpruned enumeration on H-poor, halogenated, radical and closed losses
    (where the rest after a library radical is closed-shell, and the radical
    bound would over-prune it), at k_max 2 and 3."""
    rng = np.random.default_rng(11)
    hi = {"C": 4, "H": 7, "O": 4, "N": 2, "^N": 2, "S": 2, "F": 6, "Cl": 3, "Br": 2, "I": 2}
    losses = ["CClF4", "BrF2", "C2BrF6HI", "CBrF3", "C2Cl2F4", "CHF3Br", "C3H7", "C2H7",
              "H2N", "C3H10N", "CClF3O", "C2F5I"]
    while len(losses) < 400:
        c = {e: int(rng.integers(0, n)) for e, n in hi.items() if rng.random() < 0.45}
        c = {e: n for e, n in c.items() if n}
        if c:
            losses.append(C.format_formula(c))
    for k_max in (2, 3):
        S._decompose.cache_clear()
        n_read = 0
        for loss in losses:
            counts = C.parse_formula(loss)
            got = decompose_loss(loss, k_max=k_max)
            want = _brute_decompose(counts, k_max)
            assert (got and (got.pieces, got.unnamed)) == (want or None), (loss, k_max)
            n_read += got is not None
        assert n_read > 100, k_max          # the comparison is not all-None


def _radical_sums(n: int) -> list[str]:
    """Every distinct formula that is the sum of n RADICAL_LOSSES entries."""
    out = set()
    for combo in itertools.combinations_with_replacement(S.RADICAL_LOSSES, n):
        tot: dict[str, int] = {}
        for piece in combo:
            for e, k in C.parse_formula(piece).items():
                tot[e] = tot.get(e, 0) + k
        out.add(C.format_formula({e: k for e, k in tot.items() if k}))
    return sorted(out)


def test_a_loss_holds_at_most_one_radical_piece():
    """A radical loss is exactly ONE radical piece plus closed-shell ones; a
    closed-shell loss has none. In 15N the library has no closed-shell
    nitrogen oxide, so ^NO + ^NO2 + ^NO3 (^N3O6) is no reading at k_max 3 or
    4, where every split would need two or three radicals, and two radicals
    never make a closed-shell loss (^N2O3 = ^NO + ^NO2). The 14N N3O6 reads,
    as closed-shell pieces around one radical, only because N2O is named."""
    for loss, k_max in (("^N3O6", 3), ("^N3O6", 4), ("N^N2O3", 3), ("Br2I3", 4), ("^N2O3", 3),
                        ("^N2O3", S.K_MAX_LIMIT)):
        assert decompose_loss(loss, k_max=k_max) is None, (loss, k_max)
    d = decompose_loss("N3O6")
    assert (d.pieces, d.unnamed, d.radical) == (("O2", "N2O", "NO3"), 0, True)
    # every sum of two (closed-shell losses) or three (radical losses) library
    # radicals against the unpruned enumeration, which counts radical pieces
    for n, k_max in itertools.product((2, 3), (3, 4)):
        S._decompose.cache_clear()
        n_none = 0
        for loss in _radical_sums(n):
            got = decompose_loss(loss, k_max=k_max)
            want = _brute_decompose(C.parse_formula(loss), k_max)
            assert (got and (got.pieces, got.unnamed)) == (want or None), (loss, k_max)
            assert got is None or (got.radical == (n == 3)
                                   and sum(p in S.RADICAL_LOSSES for p in got.pieces)
                                   <= int(got.radical)), (loss, got)
            n_none += got is None
        assert n_none >= 10, (n, k_max)     # the rule rejects some of them


# Deep losses (k 5-6) where the frontier would grow combinatorially without
# the k-bound pruning: synthetic, found by a random search over large
# halogenated / S / N / Si losses; each splits into six pieces. Pruned: 5-35 ms
# each; unpruned: 0.1-1.5 s.
_DEEP_LOSSES = ("C6H18BrF7N3O7S2", "C6H23Cl2N2O7", "C7H25BrNO6", "C11H25Cl3F7IN^NO4S",
                "C6H21F7IN3O5SSi", "C6H16BrCl2F4O4", "C3H18BrCl2I^NO5Si2", "C6H22BrO6")


def test_deep_losses_decompose_fast_at_the_k_max_cap():
    S._decompose.cache_clear()
    t0 = time.perf_counter()
    decs = [decompose_loss(f, k_max=S.K_MAX_LIMIT) for f in _DEEP_LOSSES]
    elapsed = time.perf_counter() - t0
    assert all(d is not None and d.k == 6 for d in decs), [d and d.name for d in decs]
    # about 0.1 s together with pruning, ~3 s without it
    assert elapsed < 0.6, elapsed
