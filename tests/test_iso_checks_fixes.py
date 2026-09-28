"""The C11+a verification fixes (2026-09-27, the user's decisions after the skeptics):
rule C reads no carbon count where 13C does not dominate the +1 line (a Si-rich
ion's is mostly 29Si); REQ judges "would I have seen it" at the height this batch
shows the element's lines (the uronium batch's Si lines run 0.4-0.6x theory);
HIGH names an element only where an ion carrying the atoms the line implies fits
the mass; and a line several heavy spacings reach is named by the nearest one.

Run: pytest tests/test_iso_checks_fixes.py -q
"""

from __future__ import annotations

import numpy as np
import pytest

from peaky.batch import iso_checks as IC
from peaky.chem import chemistry as C
from tests.test_iso_checks import FILL, H, N, _c13, _get, _has, _m0, _measure, _row, _series

D81 = 1.9979521


# --------------------------------------------------------------------------- rule C: 13C's share of the +1 line
def test_c13_share_counts_every_light_contributor():
    share, top = IC.c13_share(C.parse_formula("C8H24O4Si4"))          # D4: 29Si outweighs 13C
    assert share < IC.C_MIN_13C_SHARE and top == "Si"
    share, top = IC.c13_share(C.parse_formula("C12H14O4Si"))          # one Si beside twelve C
    assert share > IC.C_MIN_13C_SHARE and top == "Si"
    assert IC.c13_share(C.parse_formula("C10H16O4"))[0] > 0.9
    assert IC.c13_share({}) == (0.0, "")


def test_rule_c_reads_no_carbon_count_on_a_si_rich_ion():
    """A +1 line read at 0 C refutes a 12-carbon Si1 formula, but not a Si4 one:
    there the line 13C would make is mostly 29Si, 3.8 mDa lower."""
    si4, si1 = "C8H24O4Si4", "C12H14O4Si"
    t = _measure(_series(lambda i: _c13(i, si4, H, c=0.0) + _c13(i, si1, H, c=0.0, phase=1.0)), [(si4, H), (si1, H)])
    r = _get(t, "C", si4)
    assert r["verdict"] == "untestable" and not r["veto"]
    assert "the +1 line is mostly 29Si" in r["note"] and "no carbon count" in r["note"]
    r = _get(t, "C", si1)
    assert r["verdict"] == "contradict" and r["veto"]


# --------------------------------------------------------------------------- REQ: the batch's own line height
REF_BR = ["C6H9BrO3", "C7H11BrO3", "C8H13BrO3", "C9H15BrO3"]
DIM = "C10H17BrO3"


def _br_line(i, n, h0, share):
    """A bromine ion's M0 at h0 and its 81Br line at `share` x the theory height."""
    mz = C.ion_mz(n, H)
    return [_m0(i, n, H, h0), _row(i, mz + D81, share * 0.9728 * h0, role="iso_child", label="81Br")]


def _dim(i, h0):
    return [_m0(i, DIM, H, h0)]                                       # its 81Br line is missing


def test_line_efficiency_is_the_median_seen_height_within_floor_and_one():
    for share, eff in ((0.5, 0.5), (1.3, 1.0), (0.1, IC.REQ_EFF_FLOOR)):
        t = _measure(_series(lambda i, s=share: sum((_br_line(i, n, 1e4, s) for n in REF_BR[:3]), [])),
                     [(n, H) for n in REF_BR[:3]])
        assert _get(t, "REQ", REF_BR[0])["line_eff"] == pytest.approx(eff, abs=0.02), share
    # fewer than REQ_EFF_PAIRS seen lines: no measurement, the theory height
    t = _measure(_series(lambda i: sum((_br_line(i, n, 1e4, 0.5) for n in REF_BR[:2]), [])),
                 [(n, H) for n in REF_BR[:2]])
    assert _get(t, "REQ", REF_BR[0])["line_eff"] == 1.0


def test_req_spares_a_line_this_batch_would_show_below_its_floor():
    """At theory height the dim ion's 81Br line would clear 3x the floor (absent:
    refuted); at the half height this batch shows bromine lines it would not."""
    h0 = 4.0 * FILL / 0.9728                                          # theory line 4x the floor, seen 2x
    pairs = [(n, H) for n in REF_BR[:3]] + [(DIM, H)]
    t = _measure(_series(lambda i: sum((_br_line(i, n, 1e4, 1.0) for n in REF_BR[:3]), []) + _dim(i, h0)), pairs)
    r = _get(t, "REQ", DIM)
    assert r["verdict"] == "absent" and r["veto"] and r["line_eff"] == pytest.approx(1.0, abs=0.02)
    t = _measure(_series(lambda i: sum((_br_line(i, n, 1e4, 0.5) for n in REF_BR[:3]), []) + _dim(i, h0)), pairs)
    r = _get(t, "REQ", DIM)
    assert r["verdict"] == "untestable" and not r["veto"] and r["line_eff"] == pytest.approx(0.5, abs=0.02)
    assert "this batch shows the element's lines at 0.50x" in r["note"]


def test_a_mixed_halogen_line_reads_the_smaller_efficiency():
    eff = {"Br": (0.6, 5), "Cl": (0.8, 4)}
    assert IC._eff_of(eff, "BrCl") == 0.6 and IC._eff_of(eff, "Si") == 1.0
    assert IC._eff_of({"BrCl": (0.9, 3), **eff}, "BrCl") == 0.9


# --------------------------------------------------------------------------- HIGH: the element must fit the mass
def test_an_element_fits_only_where_a_chnos_rest_makes_up_the_mass():
    assert not IC.element_fits(131.0815, "Br", 1, 5.0, 1.0)           # a Br leaves 52 Da at +0.163 defect
    assert IC.element_fits(547.195229, "Br", 1, 20.0, -1.0)           # C28H38NO5Br- at 2.4 ppm
    assert IC.element_fits(157.909401, "Br", 1, 5.0, -1.0)            # HBrO.NO3-
    assert IC.element_fits(78.918337 + C.M_E, "Br", 1, 5.0, -1.0)     # Br- itself: no rest
    assert not IC.element_fits(241.0506, "Br", 3, 5.0, -1.0)          # three Br do not fit m/z 241
    assert not IC.element_fits(241.0506, "S", 22, 5.0, -1.0)          # a 1.0x 34S line: 22 S
    assert not IC.element_fits(50.0, "Br", 1, 5.0, -1.0)              # lighter than the atom
    assert IC.element_fits(float("nan"), "Br", 1, 5.0, -1.0)          # no mass: no verdict to withhold


def test_high_guards_a_line_whose_element_no_ion_at_that_mass_carries():
    n = "C4H6O"                                                       # [M-H]- at m/z 69.03: no Br fits
    mz = C.ion_mz(n, H)
    ts = _series(lambda i: [_m0(i, n, H, 1e4 * (1.5 + np.sin(i / 2))),
                            _row(i, mz + IC.HIGH_OFFSETS["81Br"], 1e4 * (1.5 + np.sin(i / 2)))])
    r = _get(_measure(ts, [(n, H)]), "HIGH", n)
    assert r["verdict"] == "guarded" and not r["veto"]
    assert "no ion carrying the Br this line needs fits its mass" in r["note"]


# --------------------------------------------------------------------------- HIGH: which spacing names the line
def _per(**labs):
    return {k: {"ra": np.array([ra]), "off": np.array([off])} for k, (ra, off) in labs.items()}


def test_one_line_reached_by_two_spacings_is_named_by_the_nearer():
    # a line 0.017 mDa off 37Cl is 0.227 mDa off 30Si: one peak; 30Si read it a touch taller
    per = _per(**{"37Cl": (0.43, 0.017), "30Si": (0.46, 0.227)})
    assert IC._label(["37Cl", "30Si"], per, 0, 0.25e-3) == "37Cl"


def test_two_lines_report_the_taller():
    per = _per(**{"81Br": (0.5, 0.0), "18O": (1.0, 0.0)})              # 6.3 mDa apart: two peaks
    assert IC._label(["81Br", "18O"], per, 0, 0.25e-3) == "18O"


# --------------------------------------------------------------------------- pinned by the mutation pass
def test_line_efficiency_is_the_median_not_the_best_line():
    shares = dict(zip(REF_BR[:3], (0.4, 0.5, 0.9)))
    t = _measure(_series(lambda i: sum((_br_line(i, n, 1e4, s) for n, s in shares.items()), [])),
                 [(n, H) for n in shares])
    assert _get(t, "REQ", REF_BR[0])["line_eff"] == pytest.approx(0.5, abs=0.02)


def test_line_efficiency_reads_no_lower_than_a_quarter():
    assert IC.REQ_EFF_FLOOR == 0.25 and IC.REQ_EFF_PAIRS == 3 and IC.REQ_EFF_SEEN == 0.5
    t = _measure(_series(lambda i: sum((_br_line(i, n, 1e4, 0.1) for n in REF_BR[:3]), [])),
                 [(n, H) for n in REF_BR[:3]])
    assert _get(t, "REQ", REF_BR[0])["line_eff"] == 0.25


def test_line_efficiency_counts_only_the_spectra_where_theory_says_the_line_shows():
    """Three bromine ions dim in 60 % of the spectra (their line below the floor,
    not picked) and bright elsewhere (seen at half height): judged over the
    spectra where it could show they are seen lines and set 0.5; over every
    spectrum they are not, and the theory height (1) stands."""
    def f(i):
        h0 = 1e4 if i % 5 < 2 else 2.0 * FILL
        return sum((_br_line(i, n, h0, 0.5) if h0 > 100 else [_m0(i, n, H, h0)] for n in REF_BR[:3]), [])
    t = _measure(_series(f), [(n, H) for n in REF_BR[:3]])
    assert _get(t, "REQ", REF_BR[0])["line_eff"] == pytest.approx(0.5, abs=0.02)


def test_the_atoms_a_line_implies_read_off_its_smaller_ratio():
    """A 81Br line at 1.0x on area and 2.0x on height: one Br by the smaller ratio
    (fits m/z 241 -- refuted), two by the larger (no Br2 ion at 241 -- guarded)."""
    from tests.test_iso_checks import Z, _hi
    r = _get(_measure(_series(lambda i: _hi(i, ratio=1.0, ratio_h=2.0)), [(Z, H)]), "HIGH", Z)
    assert r["verdict"] == "too_high" and r["offset"] == "81Br"


def test_the_fit_keeps_hydrogen_within_2c_n_4_and_the_orbitrap_window_at_5_ppm():
    # a Br beside C2H10 (the only CHNOS rest of 34.078 Da): ten H on two C is no ion
    assert not IC.element_fits(78.9183371 + 2 * 12.0 + 10 * 1.00782503207 + C.M_E, "Br", 1, 5.0, -1.0)
    assert IC.element_fits(78.9183371 + 2 * 12.0 + 8 * 1.00782503207 + C.M_E, "Br", 1, 5.0, -1.0)
    assert IC.HIGH_FIT_PPM == {"orbitrap": 5.0, "tof": 20.0}
