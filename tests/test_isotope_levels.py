"""C11+c on the levels: an isotope child is evidence where it sits at its label's
exact spacing from the COMMITTED parent line and is as tall as the ion's
composition makes it (count-aware, relative to that line); the isotopologues
list answers the same question; a dropped child is dropped for every fact.

Every source here is physically possible: each ion at its exact m/z (its
committed isotopologue's where that is not the mono line), each child at its
configuration's exact position plus a stated offset, and a background of CHO
[M-H]- acids whose '13C' lines scatter as sigma(h)^2 = a^2 + b^2 / h, so the
source's own fit sets the window. The reference script levels every source
alike, fact for fact (`_both`).

Run: pytest tests/test_isotope_levels.py -q
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from peaky.assignment import evidence as EV
from peaky.chem import chemistry as C
from peaky.chem import isotopes as I
from tests.test_evidence import LEDGER_COLUMNS, m0

_spec = importlib.util.spec_from_file_location(
    "level_ledger_lv", Path(__file__).resolve().parents[1] / "scripts" / "level_ledger.py")
LL = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(LL)

RES_ORBI = {"coef": 5.0134221e-07, "exponent": 1.5344223, "offset": 0.0}     # R(200) ~ 118 000
RES_TOF = {"coef": 0.00010564869, "exponent": 0.99632248, "offset": 0.0}      # R(200) ~ 9 650
SP = I.ISOTOPE_SPACING
FACTS = ["iso", "multiline", "multiline_elements", "carbon_ev", "reagent_only_iso", "iso_labels"]


# --------------------------------------------------------------------------- builders
def ion_of(neutral: str, adduct: str) -> tuple[dict, str]:
    counts = {e: v for e, v in EV.ion_composition(neutral, adduct, None).items() if v}
    return counts, C.format_formula(counts) + adduct.rstrip(".")[-1]          # '[M]-.' is an anion too


def parent(pid, neutral, adduct="[M-H]-", *, height=1e6, heavy=None, off_ppm=0.0, pcal=0.0, **kw):
    """An M0 row at the exact m/z of the ion's committed line (`heavy`, {} = mono), `off_ppm` from it
    (pcal = the calibration's own reading of that offset, ppm_error_cal)."""
    counts, ion = ion_of(neutral, adduct)
    true = I.mono_mz(counts, adduct.rstrip(".")[-1]) + I.heavy_shift(heavy or {})
    row = m0(pid, neutral, adduct=adduct, ion=ion, mz=true * (1 + off_ppm * 1e-6), height=height, **kw)
    row["ppm_error_cal"] = pcal
    row["_true"] = true
    return row


def kid(pid, par, label, height, shift, off_ppm=0.0):
    """An isotope child `shift` Da above the parent's TRUE position, `off_ppm` (of the parent m/z) off it."""
    row = m0(pid, None, adduct=None, height=height, mz=par["_true"] + shift + off_ppm * 1e-6 * par["_true"])
    row.update(role="iso_child", parent_peak_id=par["peak_id"], iso_label=label, tier="Assigned",
               degeneracy_density=None, ion_formula=None)
    return row


def background(n=400, a=0.2, b=6.0, seed=11, exact=False):
    """n CHO [M-H]- acids, each with a '13C' line at its exact spacing plus a residual of scatter
    sigma(h)^2 = a^2 + b^2 / h (a seeded draw, or `exact`: ten residuals per height whose MAD is exact)."""
    rng = np.random.default_rng(seed)
    if exact:
        pat = np.array([-3, -1, -1, -1, 0, 0, 1, 1, 1, 3], float) / I.MAD_TO_SIGMA
        hs = np.repeat([30, 100, 300, 1000, 3000, 10000, 30000, 100000], 10)
        hs = np.resize(hs, n)
        res = np.array([pat[i % 10] * np.sqrt(a * a + b * b / h) for i, h in enumerate(hs)])
    else:
        hs = 10 ** rng.uniform(1.5, 5.0, n)
        res = rng.normal(0.0, np.sqrt(a * a + b * b / hs))
    rows = []
    for i in range(n):
        nc, no = 5 + i % 20, 9 + (i // 20) % 6                   # O9-O14: no test neutral is among them
        neutral = f"C{nc}H{2 * nc - 4 + 2 * (i // 120)}O{no}"
        p = parent(f"bg{i}", neutral, height=1e7)
        rows += [p, kid(f"bgc{i}", p, "13C", float(hs[i]), SP["13C"], float(res[i]))]
    return rows


def frame(rows) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=LEDGER_COLUMNS)


def _both(rows, resolution=RES_ORBI, *, per_file=False, cross=None) -> dict:
    """{(neutral, adduct): (level, facts)} from the engine -- pooled, or per file -- and the SAME from the
    reference script (asserted)."""
    led = frame(rows)
    if per_file:
        out = EV.compute_levels(led, resolution=resolution, cross=cross)
        j = led[led.role == "M0"][["peak_id", "neutral_formula", "adduct"]].merge(out, on="peak_id")
        per = {(n, a): lv for n, a, lv in zip(j.neutral_formula, j.adduct, j.evidence_level)}
        pooled = EV.level_pooled({"f": led}, resolution=resolution, cross=cross)
        facts = {(r.neutral_formula, r.adduct): r for r in pooled.itertuples(index=False)}
        eng = {k: (per[k], facts[k]) for k in per}
    else:
        pooled = EV.level_pooled({"f": led}, resolution=resolution, cross=cross)
        eng = {(r.neutral_formula, r.adduct): (r.evidence_level, r) for r in pooled.itertuples(index=False)}
    halogen = LL.detect_reagent_halogen(led[led.role == "M0"])
    ref = LL.assign_levels(LL.measure_source("f", led.assign(__file="f"), halogen, resolution, per_file),
                           set(cross or ()))
    for r in ref.itertuples(index=False):
        lv, f = eng[(r.neutral, r.adduct)]
        assert r.level == lv, (r.neutral, r.adduct, r.level, lv)
        if not per_file:
            for c in FACTS:
                assert str(getattr(r, c)) == str(getattr(f, c)), (r.neutral, r.adduct, c)
    return eng


def _level(got, neutral, adduct="[M-H]-"):
    return got[(neutral, adduct)][0]


def _fact(got, neutral, adduct="[M-H]-"):
    return got[(neutral, adduct)][1]


# --------------------------------------------------------------------------- the position (I4)
def test_a_line_off_its_exact_position_counts_only_where_its_height_allows_the_offset():
    """1.5 ppm off a C10 line: outside the 1-ppm window of a 1e5-cps line,
    inside the ~2-ppm window of a 100-cps one (the source's own fit)."""
    bright = parent("a", "C10H16O4", height=1e6)
    dim = parent("b", "C11H18O4", height=935.0)
    rows = background() + [bright, kid("ac", bright, "13C", 1.07e5, SP["13C"], 1.5),
                           dim, kid("bc", dim, "13C", 110.0, SP["13C"], 1.5)]
    got = _both(rows)
    led = frame(rows)
    t = led[led.iso_label == "13C"].merge(led[led.role == "M0"][["peak_id", "mz"]], left_on="parent_peak_id",
                                          right_on="peak_id", suffixes=("", "_p"))
    fit = I.fit_position_sigma((t.mz - t.mz_p - SP["13C"]) / t.mz_p * 1e6, t.height)
    assert I.position_window_ppm(1.07e5, fit) < 1.5 < I.position_window_ppm(110.0, fit)
    assert _level(got, "C10H16O4") == "4c" and not _fact(got, "C10H16O4").iso
    assert _level(got, "C11H18O4") == "4b" and _fact(got, "C11H18O4").iso
    # at its exact position the bright line counts
    rows[-3] = kid("ac", bright, "13C", 1.07e5, SP["13C"], 0.0)
    assert _level(_both(rows), "C10H16O4") == "4b"


def test_a_source_with_fewer_than_40_13c_children_is_not_tested():
    bright = parent("a", "C10H16O4", height=1e6)
    rows = background(n=38) + [bright, kid("ac", bright, "13C", 1.07e5, SP["13C"], 1.5)]
    assert _level(_both(rows), "C10H16O4") == "4b"


def test_n1_a_brighter_neighbour_pulls_the_line_toward_itself():
    """C10H16O5 [M]-.'s pattern on R1: its 13C line displaced +3 ppm toward the
    25x brighter [M-H]- line of the neutral with two more H, 4.47 mDa above (at
    its own exact m/z, `_n1`). Not 3x the child, or the child displaced away from
    it, and the line stays dropped."""
    assert _level(_both(_n1(nb_height=2.7e6)[0]), "C10H16O5", "[M]-.") == "4b"
    assert _level(_both(_n1(nb_height=1.0e4)[0]), "C10H16O5", "[M]-.") == "4c"          # not 3x the child
    assert _level(_both(_n1(nb_height=2.7e6, off_ppm=-3.0)[0]), "C10H16O5", "[M]-.") == "4c"   # the other side


def test_pcal_a_displaced_parent_line_places_its_line_through_the_calibration():
    """The parent line sits 3 ppm high and the file's calibration says so
    (ppm_error_cal +3): its 13C child, at the TRUE parent + 1.00335, counts."""
    p = parent("a", "C12H20O4", height=1e6, off_ppm=3.0, pcal=3.0)
    rows = background() + [p, kid("ac", p, "13C", 1.28e5, SP["13C"], 0.0)]
    assert _level(_both(rows), "C12H20O4") == "4b"
    rows[-2] = dict(p, ppm_error_cal=0.0)                            # uncalibrated: 3 ppm off, dropped
    assert _level(_both(rows), "C12H20O4") == "4c"


def test_a_tof_line_30_ppm_off_is_dropped_and_5_ppm_off_counts():
    """On a TOF-like source (a 2.5, b 1.8 ppm: a ~10-ppm window) an 81Br line
    of a brominated acid 30 ppm off its exact spacing is another peak."""
    p = parent("a", "C7H11BrO4", height=1e5)
    rows = background(a=2.5, b=1.8) + [p, kid("ab", p, "81Br", 0.97e5, SP["81Br"], 30.0)]
    got = _both(rows, RES_TOF)
    assert _level(got, "C7H11BrO4") == "4c" and _fact(got, "C7H11BrO4").iso_labels == ""
    rows[-1] = kid("ab", p, "81Br", 0.97e5, SP["81Br"], 5.0)
    got = _both(rows, RES_TOF)
    assert _level(got, "C7H11BrO4") == "4b" and _fact(got, "C7H11BrO4").iso_labels == "81Br"


def test_the_per_file_tof_guard():
    """A TOF-class file whose '13C' fit sits on the intercept floor (its scatter
    collapsed: no height-independent term at all) is not position-tested per
    file; pooled, or on an Orbitrap-class file, the same line drops."""
    p = parent("a", "C7H11BrO4", height=1e5)
    rows = background(a=0.0, b=6.0, exact=True) + [p, kid("ab", p, "81Br", 0.97e5, SP["81Br"], 6.0)]
    assert _level(_both(rows, RES_TOF, per_file=True), "C7H11BrO4") == "4b"      # guarded: no test
    assert _level(_both(rows, RES_ORBI, per_file=True), "C7H11BrO4") == "4c"     # Orbitrap-class: tested
    assert _level(_both(rows, RES_TOF), "C7H11BrO4") == "4c"                     # pooled: no guard
    assert _level(_both(rows, None, per_file=True), "C7H11BrO4") == "4c"         # class-less: no guard


def test_a_dropped_line_is_dropped_for_every_fact():
    """A bromide cluster of a Br-free neutral: its reagent 81Br line kept and a
    '13C' line 3 ppm off at 1e5 cps. The dropped 13C line gives no carbon, so
    the sole satellite is the reagent's: 4d, iso_labels '81Br'."""
    rows = background()
    p = parent("a", "C8H14O2", "[M+Br]-", height=1e6)
    rows += [p, kid("ab", p, "81Br", 0.97e6, SP["81Br"]), kid("ac", p, "13C", 0.86e5, SP["13C"], 3.0),
             parent("q", "C9H16O2", "[M+Br]-"), parent("r", "C9H18O2", "[M+Br]-")]
    got = _both(rows)
    f = _fact(got, "C8H14O2", "[M+Br]-")
    assert _level(got, "C8H14O2", "[M+Br]-") == "4d"
    assert f.iso_labels == "81Br" and not f.carbon_ev and f.reagent_only_iso
    rows[-4] = kid("ac", p, "13C", 0.86e5, SP["13C"], 0.0)                     # placed: carbon pins it
    got = _both(rows)
    assert _level(got, "C8H14O2", "[M+Br]-") == "4b" and _fact(got, "C8H14O2", "[M+Br]-").carbon_ev


# --------------------------------------------------------------------------- the committed line (I1, D4)
def test_dibromoacetic_acid_committed_on_79br81br_reads_both_its_neighbour_lines():
    """R1's C2H2Br2O2 [M-H]-: committed on 79Br81Br; its '81Br2' (scorer) and
    'M0' (the 79Br2 line) children 1.998 Da either side, 0.486 and 0.514 of it
    -- a textbook 1:2:1. Before C11+c both read expected 0 (5a); now iso (4b)."""
    p = parent("a", "C2H2Br2O2", height=1e5, heavy={"81Br": 1})
    rows = background() + [p, kid("u", p, "81Br2", 0.49e5, SP["81Br"]), kid("d", p, "M0", 0.50e5, -SP["81Br"])]
    got = _both(rows)
    f = _fact(got, "C2H2Br2O2")
    assert _level(got, "C2H2Br2O2") == "4b" and f.iso and f.iso_labels == "81Br2"
    # read against the mono line (the old reading) the 81Br2 line expects 0.9728^2 and M0 nothing: still in
    # band for 81Br2 alone -- but put the parent at its mono position and the same lines sit 2 Da off
    q = parent("a", "C2H2Br2O2", height=1e5)
    rows2 = background() + [q, kid("u", q, "81Br2", 0.49e5, SP["81Br"]), kid("d", q, "M0", 0.50e5, -SP["81Br"])]
    assert not _fact(_both(rows2), "C2H2Br2O2").iso


def test_count_labels_count_and_a_line_the_ion_cannot_make_never_does():
    rows = background()
    br2 = parent("a", "C8H14Br2O4", height=1e5)                             # committed on its mono line
    brfree = parent("b", "C31H34O10", height=1e5)                           # a Br-free [M-H]- with a 1:1 +2 line
    si7 = parent("c", "C14H42O7Si7", "[M+H]+", height=1e5)
    s2 = parent("d", "C8H14O4S2", height=1e5)
    rows += [br2, kid("a2", br2, "2x81Br", 0.93e5, 2 * SP["81Br"]),
             brfree, kid("b1", brfree, "81Br", 0.95e5, SP["81Br"]),
             si7, kid("c1", si7, "29Si", 0.36e5, SP["29Si"]), kid("c2", si7, "30Si", 0.23e5, SP["30Si"]),
             s2, kid("d1", s2, "34S", 0.0435e5, SP["34S"])]
    got = _both(rows)
    assert _fact(got, "C8H14Br2O4").iso                     # peaky's '2x81Br', parent-relative: the 81Br2 line, 0.946x
    assert not _fact(got, "C31H34O10").iso                  # an ion with no Br makes no 81Br line
    assert _fact(got, "C14H42O7Si7", "[M+H]+").iso          # 7 x 0.0508 and 7 x 0.0335: its own lines
    assert not _fact(got, "C8H14O4S2").iso                  # one 34S line of S2 is 0.0886: 0.0435 is out of band


def test_13c2_has_its_own_expectation():
    p = parent("a", "C22H42O6", "[M+H]+", height=1e5, series_unit="CH2")
    rows = background() + [p, kid("c2", p, "13C2", 1e5 * 231 * 0.0107 ** 2, 2 * SP["13C"])]
    assert _fact(_both(rows), "C22H42O6", "[M+H]+").iso
    rows[-1] = kid("c2", p, "13C2", 1e5 * 22 * 0.0107, 2 * SP["13C"])      # the per-carbon M+1 value: 9x too tall
    assert not _fact(_both(rows), "C22H42O6", "[M+H]+").iso


# --------------------------------------------------------------------------- whole labels (I3, D2)
def _bromide(*rows):
    return [*rows, parent("q", "C9H16O2", "[M+Br]-"), parent("r", "C9H18O2", "[M+Br]-")]


def test_chloroacetic_acids_m4_line_proves_its_chlorine():
    """C2H3ClO2 [M+Br]- (Br1 Cl1) on the TOF: its +2 line is the 81Br / 37Cl
    blend; its '81Br+37Cl' M+4 line (0.9728 x 0.3196) proves the Cl -- with its
    13C line and its nitrate cluster (chan2) it is 4a, C and Cl."""
    rows = background(a=2.5, b=1.8)
    p = parent("a", "C2H3ClO2", "[M+Br]-", height=1e5)
    n = parent("n", "C2H3ClO2", "[M+NO3]-", height=1e4)
    rows += _bromide(p, kid("c", p, "13C", 1e5 * 2 * 0.0107, SP["13C"]),
                     kid("b", p, "81Br", 1.30e5, SP["81Br"]),
                     kid("m4", p, "81Br+37Cl", 1e5 * 0.9728 * 0.3196, SP["81Br"] + SP["37Cl"]), n)
    got = _both(rows, RES_TOF)
    f = _fact(got, "C2H3ClO2", "[M+Br]-")
    assert f.multiline and f.multiline_elements == "C|Cl"
    assert _level(got, "C2H3ClO2", "[M+Br]-") == "4a"
    # without the M+4 line the Cl rests on nothing: 4b
    got = _both([r for r in rows if r["peak_id"] != "m4"], RES_TOF)
    assert _level(got, "C2H3ClO2", "[M+Br]-") == "4b"


def test_an_m3_81br_13c_line_is_a_carbon_line():
    """C7H6N2O6S [M+Br]-: its reagent 81Br line and an '81Br+13C' M+3 line --
    the 13C line of its 81Br isotopologue. Carbon pins the neutral: 4b, not 4d."""
    p = parent("a", "C7H6N2O6S", "[M+Br]-", height=1e5)
    rows = background() + _bromide(p, kid("b", p, "81Br", 0.97e5, SP["81Br"]),
                                   kid("m3", p, "81Br+13C", 1e5 * 0.9728 * 7 * 0.0107, SP["81Br"] + SP["13C"]))
    got = _both(rows)
    f = _fact(got, "C7H6N2O6S", "[M+Br]-")
    assert f.carbon_ev and not f.reagent_only_iso and _level(got, "C7H6N2O6S", "[M+Br]-") == "4b"
    assert f.iso_labels == "81Br|81Br+13C"


def test_a_scorer_13c_81br_label_on_a_heavy_parent_credits_carbon_only():
    """A Br3 ion committed on 79Br2 81Br: the scorer's '13C+81Br' (mono-counted)
    is the parent's own 13C line -- it adds carbon, not bromine."""
    p = parent("a", "C19H27BrN2O", "[M+HBr+Br]-", height=1e5, heavy={"81Br": 1})
    rows = background() + _bromide(p, kid("c", p, "13C+81Br", 1e5 * 19 * 0.0107, SP["13C"]))
    f = _fact(_both(rows), "C19H27BrN2O", "[M+HBr+Br]-")
    assert f.iso and f.carbon_ev and f.iso_labels == "13C+81Br" and not f.reagent_only_iso


def test_a_generic_m_plus_n_line_is_exempt_and_credits_nothing():
    p = parent("a", "C10H16O4", height=1e5)
    rows = background() + [p, kid("m", p, "M+2", 1e5 * 0.0056, 2.0055, 25.0)]
    f = _fact(_both(rows), "C10H16O4")
    assert f.iso_labels == "M+2" and not f.carbon_ev and f.multiline_elements == ""


# --------------------------------------------------------------------------- the list (D5)
def test_the_isotopologues_list_answers_the_same_question():
    rows = background()
    p = parent("a", "C10H16O4", height=1e6)
    other = orphan("o", p["_true"] + SP["13C"], 1.07e5)                     # an iso row at a's 13C position
    far = orphan("f", (p["_true"] + SP["13C"]) * (1 + 3e-6), 1.07e5)        # 3 ppm off that position
    rows += [p, other, far]
    for peak, want in (("o", "4b"), ("f", "4c"), ("nowhere", "4c")):
        p["isotopologues"] = json.dumps([{"label": "13C", "score": None, "peak_id": peak}])
        assert _level(_both(rows), "C10H16O4") == want, peak
    # a list naming a line out of band (2x the expectation of 10 carbons ... 5x) does not count
    other["height"] = 5.35e5
    p["isotopologues"] = json.dumps([{"label": "13C", "score": 0.9, "peak_id": "o"}])
    assert _level(_both(rows), "C10H16O4") == "4c"


def test_the_committed_lines_tolerance_follows_the_instrument_class():
    """Dibromoacetic acid's 79Br81Br parent 8 ppm off its exact position: within
    the TOF-class / class-less 20 ppm it is still read as committed on that line
    (its 81Br2 / 79Br2 neighbours count), beyond the Orbitrap-class 5 ppm it is
    read as the mono line and its neighbours sit 2 Da off every reading."""
    def rows_at(res):
        p = parent("a", "C2H2Br2O2", height=1e5, heavy={"81Br": 1}, off_ppm=8.0, pcal=8.0)
        return background(a=2.5, b=1.8) + [p, kid("u", p, "81Br2", 0.49e5, SP["81Br"]),
                                           kid("d", p, "M0", 0.50e5, -SP["81Br"])]
    assert _fact(_both(rows_at(RES_TOF), RES_TOF), "C2H2Br2O2").iso
    assert _fact(_both(rows_at(None), None), "C2H2Br2O2").iso
    assert not _fact(_both(rows_at(RES_ORBI), RES_ORBI), "C2H2Br2O2").iso


# --------------------------------------------------------------------------- D4's full-count line (a position rule)
# A pair whose ion carries n atoms of the reagent halogen, s of them the adduct's: a kept, in-band line whose heavy
# index relative to the committed line, j = k - k_c(n), lies outside [-k_c(s), s - k_c(s)] -- a line an s-atom ion
# committed on its own most probable line cannot make -- is the NEUTRAL's halogen, so the pair is not reagent-only,
# whatever other lines it carries (k_c(1) = 0, k_c(2) = k_c(3) = 1 for Br). Height alone is no such evidence.
def test_a_line_only_the_full_halogen_count_makes_is_the_neutrals_halogen():
    """C9H20BrN3O6 [M+Br]- (R3): a Br2 ion, one Br the neutral's and one the reagent's, committed on 79Br81Br;
    a Br1 ion puts lines at j = 0 and +1 only. Its 'M0' line -- the 79Br2 line 1.998 Da below, j = -1 -- is
    the neutral's second Br: 4b, alone or beside its '81Br2' line. The '81Br2' line alone (j = +1: where a Br1
    ion committed on its 79Br line puts its 81Br line) is the reagent's: 4d. Both lines in band (0.514 /
    0.486 expected)."""
    p = parent("a", "C9H20BrN3O6", "[M+Br]-", height=1e5, heavy={"81Br": 1})
    low = kid("d", p, "M0", 0.51e5, -SP["81Br"])
    up = kid("u", p, "81Br2", 0.49e5, SP["81Br"])
    for lines, flag, level in (([low], False, "4b"), ([low, up], False, "4b"), ([up], True, "4d")):
        got = _both(background() + _bromide(p, *lines))
        f = _fact(got, "C9H20BrN3O6", "[M+Br]-")
        assert f.iso and f.reagent_only_iso == flag, [r["peak_id"] for r in lines]
        assert _level(got, "C9H20BrN3O6", "[M+Br]-") == level, [r["peak_id"] for r in lines]


def test_a_full_count_line_the_band_or_the_position_test_refuses_is_no_evidence():
    """The same Br2 [M+Br]- with its '81Br2' line: a 79Br2 line out of band (0.10x against 0.514) is height
    evidence only -- still a line of the halogen's pattern, the flag holds (4d); a 79Br2 line in band but 3 ppm
    off its exact position on the Orbitrap (window 1 ppm: dropped) is no line at all (4d)."""
    p = parent("a", "C9H20BrN3O6", "[M+Br]-", height=1e5, heavy={"81Br": 1})
    up = kid("u", p, "81Br2", 0.49e5, SP["81Br"])
    for low in (kid("d", p, "M0", 0.10e5, -SP["81Br"]), kid("d", p, "M0", 0.51e5, -SP["81Br"], 3.0)):
        got = _both(background() + _bromide(p, low, up))
        f = _fact(got, "C9H20BrN3O6", "[M+Br]-")
        assert f.iso and f.reagent_only_iso and _level(got, "C9H20BrN3O6", "[M+Br]-") == "4d"


def test_a_br3_clusters_lines_where_a_br2_ion_puts_one_are_the_reagents():
    """C12H9BrN2 [M+HBr+Br]- (R3: C12H10Br3N2-, one Br the neutral's, two the reagent's) committed on
    79Br2 81Br; a Br2 ion committed on 79Br81Br puts lines at j = -1, 0, +1. The scorer's '81Br2' line 1.998 Da
    up at 0.97x (the 79Br 81Br2 line, j = +1) and the 'M0' line (79Br3, j = -1, 0.343 expected) are the
    reagent's: 4d. Peaky's '2x81Br' line 3.996 Da up is the 81Br3 line (j = +2), which only Br3 makes: in band
    (0.32x of 0.3155) it is the neutral's -- 4b; at R3's 1.23x (3.9x its expectation) it is out of band, and
    the pair stays 4d on its '81Br2' line, as built."""
    p = parent("a", "C12H9BrN2", "[M+HBr+Br]-", height=1e5, heavy={"81Br": 1})
    up = kid("u", p, "81Br2", 0.97e5, SP["81Br"])
    for lines, flag, level in (
            ([up], True, "4d"),
            ([kid("d", p, "M0", 0.34e5, -SP["81Br"])], True, "4d"),
            ([up, kid("t", p, "2x81Br", 0.32e5, 2 * SP["81Br"])], False, "4b"),
            ([up, kid("t", p, "2x81Br", 1.23e5, 2 * SP["81Br"])], True, "4d")):
        got = _both(background() + _bromide(p, *lines))
        f = _fact(got, "C12H9BrN2", "[M+HBr+Br]-")
        assert f.iso and f.reagent_only_iso == flag, [(r["iso_label"], r["height"]) for r in lines]
        assert _level(got, "C12H9BrN2", "[M+HBr+Br]-") == level, [(r["iso_label"], r["height"]) for r in lines]


def test_every_line_of_a_br_free_neutrals_hbr_br_cluster_is_the_reagents():
    """A Br-free neutral's [M+HBr+Br]-: the reagent supplies both Br (s = n = 2), so every line the ion makes is
    the reagent's. Its 'M0' line (79Br2) -- which, naming no part, left the pair without the flag as built,
    though a line of the reagent's own pattern -- takes the flag like its '81Br2' line (4d)."""
    p = parent("a", "C10H12O4", "[M+HBr+Br]-", height=1e5, heavy={"81Br": 1})
    low = kid("d", p, "M0", 0.51e5, -SP["81Br"])
    up = kid("u", p, "81Br2", 0.49e5, SP["81Br"])
    for lines in ([low], [up], [low, up]):
        got = _both(background() + _bromide(p, *lines))
        f = _fact(got, "C10H12O4", "[M+HBr+Br]-")
        assert f.iso and f.reagent_only_iso and _level(got, "C10H12O4", "[M+HBr+Br]-") == "4d"


def test_a_scorer_label_on_a_heavy_parent_credits_only_what_the_line_adds():
    """A Br2 neutral's [M-H]- committed on 79Br81Br: the scorer's '13C+81Br' is
    the parent's own 13C line (it adds a 13C, not an 81Br). It credits carbon
    alone -- read as crediting Br too it would make a second element of the
    neutral's (multiline) and, with a series tie, a 4a."""
    p = parent("a", "C8H14Br2O4", height=1e5, heavy={"81Br": 1}, series_unit="CH2")
    rows = background() + [p, kid("c", p, "13C+81Br", 1e5 * 8 * 0.0107, SP["13C"])]
    got = _both(rows)
    f = _fact(got, "C8H14Br2O4")
    assert f.iso and f.multiline_elements == "C" and not f.multiline and _level(got, "C8H14Br2O4") == "4b"


def test_carbon_is_a_placed_13c_line_whatever_its_height():
    """D10: carbon_ev is the presence of a placed line that adds 13C, not an
    in-band one (the TOF's 13C lines run tall as a class: blends). A bromide
    cluster with its reagent 81Br line and a placed 13C line 3x its
    expectation: carbon pins the neutral (4b), no reagent-only 4d."""
    p = parent("a", "C8H14O2", "[M+Br]-", height=1e6)
    rows = background() + _bromide(p, kid("b", p, "81Br", 0.97e6, SP["81Br"]),
                                   kid("c", p, "13C", 3 * 8 * 0.0107 * 1e6, SP["13C"]))
    got = _both(rows)
    f = _fact(got, "C8H14O2", "[M+Br]-")
    assert f.carbon_ev and not f.reagent_only_iso and _level(got, "C8H14O2", "[M+Br]-") == "4b"


def test_an_m0_child_of_a_mono_parent_is_no_line_of_the_ion():
    """D4: an 'M0' child 2 Da BELOW a bromide cluster committed on its MONO line
    (R3: fifteen such lines) names the mono line of an ion whose mono line is the
    parent itself -- it expects nothing. In a source too thin to be
    position-tested it still never makes the isotope axis."""
    p = parent("a", "C9H16O3", "[M+Br]-", height=1e5)
    rows = background(n=20) + _bromide(p, kid("m", p, "M0", 0.6e5, -SP["81Br"]))
    got = _both(rows)
    assert not _fact(got, "C9H16O3", "[M+Br]-").iso and _level(got, "C9H16O3", "[M+Br]-") == "4c"


# --------------------------------------------------------------------------- the hold released (I8, D1)
def test_a_br_free_ion_with_a_one_to_one_plus_2_line_takes_no_reagent_flag():
    """D1: R3's C31H34O10 [M-H]- on the bromide channel -- a 0.95:1 line 1.998 Da
    up that the batch stamps as the reagent's water cluster. No Br on the ion:
    the line is none of its lines (expected 0), so no isotope axis, and it is not
    the reagent's line on THIS ion either (the hold is released). Mass-degenerate
    with no axis: 5b, per file and pooled, no reagent_only_iso."""
    p = parent("a", "C31H34O10", height=1e4, degeneracy=3.0)
    rows = background() + _bromide(p, kid("b", p, "81Br", 0.95e4, SP["81Br"]))
    for per_file in (False, True):
        got = _both(rows, per_file=per_file)
        assert _level(got, "C31H34O10") == "5b"
    f = _fact(_both(rows), "C31H34O10")
    assert not f.iso and not f.reagent_only_iso and f.iso_labels == "81Br"
    assert "reagent_only_iso" not in f.evidence_axes and f.level_reason.startswith("5b: mass-degenerate")
    # its bromide cluster (the reagent put the Br there) keeps the flag
    q = parent("c", "C9H14O3", "[M+Br]-", height=1e4)
    rows += [q, kid("qb", q, "81Br", 0.97e4, SP["81Br"])]
    f = _fact(_both(rows), "C9H14O3", "[M+Br]-")
    assert f.reagent_only_iso and _level(_both(rows), "C9H14O3", "[M+Br]-") == "4d"


# --------------------------------------------------------------------------- the decided edges, pinned (fix round 1)
# Each case sits on one side of a decided threshold with its twin on the other side, so moving the threshold
# (or dropping the condition) moves a level; the C11+c verify lenses found every one of them unpinned.
def orphan(pid, mz, height, label="13C"):
    """An isotope row of the file whose line sits at `mz` but that the ledger hangs under no parent (a
    line the scorer listed under one M0 while the ledger could not attach it there): it is no child of
    any pair, yet an M0 / iso row the levelling reads -- N1's neighbour or a list entry."""
    row = m0(pid, None, adduct=None, height=height, mz=mz)
    row.update(role="iso_child", parent_peak_id=None, iso_label=label, tier="Assigned", degeneracy_density=None,
               ion_formula=None)
    return row


def test_a_tof_file_whose_fit_has_an_intercept_is_position_tested_per_file():
    """D3's per-file TOF guard skips only a fit that sits ON the 0.02-ppm floor. A TOF-class file whose '13C'
    scatter has a height-independent term -- a realistic TOF (a 2.5, b 1.8 ppm) or one just above the floor
    (a 0.03) -- is tested per file: an 81Br line 30 ppm (resp. 6 ppm) off its exact spacing drops."""
    p = parent("a", "C7H11BrO4", height=1e5)
    rows = background(a=2.5, b=1.8) + [p, kid("ab", p, "81Br", 0.97e5, SP["81Br"], 30.0)]
    assert _level(_both(rows, RES_TOF, per_file=True), "C7H11BrO4") == "4c"
    rows = background(a=0.03, b=6.0, exact=True) + [p, kid("ab", p, "81Br", 0.97e5, SP["81Br"], 6.0)]
    assert _level(_both(rows, RES_TOF, per_file=True), "C7H11BrO4") == "4c"
    rows = background(a=0.0, b=6.0, exact=True) + [p, kid("ab", p, "81Br", 0.97e5, SP["81Br"], 6.0)]
    assert _level(_both(rows, RES_TOF, per_file=True), "C7H11BrO4") == "4b"    # on the floor: guarded


def test_exactly_40_13c_children_make_a_fit():
    """Fewer than 40 '13C' children -> no test; 40 is a fit. 39 background lines + the test line (itself a
    '13C' child) = 40: the bright line 1.5 ppm off drops; 38 + 1 = 39 is not tested and it counts."""
    bright = parent("a", "C10H16O4", height=1e6)
    rows = background(n=39) + [bright, kid("ac", bright, "13C", 1.07e5, SP["13C"], 1.5)]
    assert _level(_both(rows), "C10H16O4") == "4c"
    rows = background(n=38) + [bright, kid("ac", bright, "13C", 1.07e5, SP["13C"], 1.5)]
    assert _level(_both(rows), "C10H16O4") == "4b"


def test_a_thin_source_keeps_its_height_term():
    """46 '13C' children, 8 height bins of 5-6: every bin holds >= 5 (SIGMA_MIN_PER_BIN), so the fit keeps
    its b / h term and the ~2-ppm window of a 110-cps line keeps it 1.5 ppm off."""
    dim = parent("b", "C11H18O4", height=935.0)
    rows = background(n=45) + [dim, kid("bc", dim, "13C", 110.0, SP["13C"], 1.5)]
    assert _level(_both(rows), "C11H18O4") == "4b"


def _n1(neutral="C10H16O5", *, nb_height=3.2e5, off_ppm=3.0, role="M0"):
    """R1's C10H16O5 [M]-. geometry: the ion's 13C line `off_ppm` off its exact position, and the [M-H]- line
    of the neutral with two more H at ITS exact m/z -- 4.47 mDa (the 13C / H doublet) above that position
    (20.6 ppm at m/z 217). As `role` 'unexplained' the same line is a row the ledger left unassigned."""
    p = parent("a", neutral, "[M]-.", height=1e5)
    counts = C.parse_formula(neutral)
    nb = parent("n", C.format_formula(dict(counts, H=counts["H"] + 2)), height=nb_height)
    if role != "M0":
        nb.update(role=role, neutral_formula=None, adduct=None, ion_formula=None, tier=None, method=None)
    nc = counts["C"]
    return background() + [p, kid("ac", p, "13C", 1e5 * nc * 0.0107, SP["13C"], off_ppm), nb], nb, p


def test_n1_pulls_only_from_three_times_the_childs_height():
    """N1: a neighbour >= 3x the child's height. 3.0e4 (2.8x the 1.07e4 child) does not pull, 3.3e4 (3.08x)
    does."""
    assert _level(_both(_n1(nb_height=3.0e4)[0]), "C10H16O5", "[M]-.") == "4c"
    assert _level(_both(_n1(nb_height=3.3e4)[0]), "C10H16O5", "[M]-.") == "4b"


def test_n1_reaches_25_ppm_of_the_exact_position():
    """The 4.47-mDa doublet is 20.6 ppm at m/z 217 (C10H16O5 [M]-.: within reach, pulled) but 28.1 ppm at
    m/z 159 (C7H10O4 [M]-.: beyond the 25-ppm reach, a line 3 ppm off stays dropped)."""
    rows, nb, p = _n1("C10H16O5")
    assert (nb["mz"] - (p["mz"] + SP["13C"])) / nb["mz"] * 1e6 < 25.0
    assert _level(_both(rows), "C10H16O5", "[M]-.") == "4b"
    rows, nb, p = _n1("C7H10O4")
    assert (nb["mz"] - (p["mz"] + SP["13C"])) / nb["mz"] * 1e6 > 25.0
    assert _level(_both(rows), "C7H10O4", "[M]-.") == "4c"


def test_n1_rescues_a_line_pulled_at_most_half_way():
    """|residual| <= 0.5 x the distance to the neighbour (4.47 mDa): a line 0.45 of the way (2.01 mDa,
    9.3 ppm) is rescued, one 0.55 of the way (2.46 mDa, 11.3 ppm) is not."""
    x = parent("a", "C10H16O5", "[M]-.")["_true"] + SP["13C"]
    near, far = 0.45 * 0.00447 / x * 1e6, 0.55 * 0.00447 / x * 1e6
    assert _level(_both(_n1(off_ppm=near)[0]), "C10H16O5", "[M]-.") == "4b"
    assert _level(_both(_n1(off_ppm=far)[0]), "C10H16O5", "[M]-.") == "4c"


def test_n1_and_the_list_read_only_the_rows_the_levelling_reads():
    """N1's neighbour and a list entry must be an M0 / iso row of the file. Per file the whole ledger reaches
    the levelling: the same bright line as an UNEXPLAINED row pulls nothing, and a list entry naming an
    unexplained row at the exact 13C position answers nothing -- in the engine and the script alike (pooled,
    the same frame reaches both too)."""
    for per_file in (True, False):
        assert _level(_both(_n1(role="unexplained")[0], per_file=per_file), "C10H16O5", "[M]-.") == "4c"
        assert _level(_both(_n1()[0], per_file=per_file), "C10H16O5", "[M]-.") == "4b"     # an M0 row: pulls
    p, ent = _listed(1e6, 1.07e5)
    ent.update(role="unexplained", iso_label=None)
    for per_file in (True, False):
        assert _level(_both(background() + [p, ent], per_file=per_file), "C10H16O4") == "4c"
    ent.update(role="iso_child", iso_label="13C")
    assert _level(_both(background() + [p, ent]), "C10H16O4") == "4b"                   # an iso row: it answers


# --------------------------------------------------------------------------- the list's own rules (D5)
def _listed(parent_height, entry_height, *, parent_off_ppm=0.0, pcal=0.0, entry_off_ppm=0.0, label="13C",
            shift=None):
    """A C10H16O4 [M-H]- M0 whose isotopologues list names ONE line, an iso row of the file at the parent's
    TRUE position + `shift` (its 13C spacing by default), `entry_off_ppm` off, under no parent (`orphan`):
    the list is the pair's only isotope evidence."""
    p = parent("a", "C10H16O4", height=parent_height, off_ppm=parent_off_ppm, pcal=pcal)
    at = (p["_true"] + (SP["13C"] if shift is None else shift)) * (1 + entry_off_ppm * 1e-6)
    ent = orphan("e", at, entry_height, label)
    p["isotopologues"] = json.dumps([{"label": label, "score": 0.9, "peak_id": "e"}])
    return p, ent


def test_a_list_entry_is_placed_through_the_parents_calibration():
    """The parent line sits 3 ppm high and its calibration says so (ppm_error_cal +3): the entry its list
    names sits at the TRUE parent + 1.00335 -- placed through pcal, as a child would be (4b); uncalibrated,
    or with the calibration's sign the other way, it is 3 / 6 ppm off (4c)."""
    p, ent = _listed(1e6, 1.07e5, parent_off_ppm=3.0, pcal=3.0)
    assert _level(_both(background() + [p, ent]), "C10H16O4") == "4b"
    for pcal in (0.0, -3.0):
        p["ppm_error_cal"] = pcal
        assert _level(_both(background() + [p, ent]), "C10H16O4") == "4c", pcal


def test_a_list_entry_is_placed_at_its_own_height():
    """A dim parent (1 000 cps) whose list entry (107 cps, in band) sits 1.5 ppm off: the entry's own
    ~2.4-ppm window places it (4b); the parent's height would give ~1.1 ppm. A bright entry the same 1.5 ppm
    off drops (4c)."""
    p, ent = _listed(1e3, 107.0, entry_off_ppm=1.5)
    assert _level(_both(background() + [p, ent]), "C10H16O4") == "4b"
    p, ent = _listed(1e6, 1.07e5, entry_off_ppm=1.5)
    assert _level(_both(background() + [p, ent]), "C10H16O4") == "4c"


def test_a_list_entry_is_not_rescued_by_a_neighbour():
    """The list test has no N1: an entry 3 ppm off the 13C position of C10H16O5 [M]-., with the 30x brighter
    [M-H]- line of C10H18O5 4.47 mDa above it (N1 rescues a CHILD there), answers nothing (4c)."""
    rows, nb, p = _n1()
    child = rows[-2]
    ent = orphan("e", child["mz"], child["height"])
    q = dict(p, isotopologues=json.dumps([{"label": "13C", "score": 0.9, "peak_id": "e"}]))
    assert _level(_both(background() + [q, ent, nb]), "C10H16O5", "[M]-.") == "4c"
    assert _level(_both(rows), "C10H16O5", "[M]-.") == "4b"                         # as its child: rescued


def test_a_list_entry_names_a_row_of_its_own_file():
    """Two files reuse peak ids (spec section 2): file f2's M0 names 'e' in its list; f2 has no row 'e',
    f1's 'e' sits exactly at f2's parent's 13C position. The entry names no row of ITS file: 4c, in the
    engine and the script; with 'e' in f2 it answers (4b)."""
    p, ent = _listed(1e6, 1.07e5)
    for f2_rows, want in (([p], "4c"), ([p, ent], "4b")):
        f1, f2 = frame(background() + ([ent] if want == "4c" else [])), frame(f2_rows)
        out = EV.level_pooled({"f1": f1, "f2": f2}, resolution=RES_ORBI)
        assert dict(zip(zip(out.neutral_formula, out.adduct), out.evidence_level))[("C10H16O4", "[M-H]-")] == want
        both = pd.concat([f1.assign(__file="f1"), f2.assign(__file="f2")], ignore_index=True)
        ref = LL.assign_levels(LL.measure_source("s", both, LL.detect_reagent_halogen(both[both.role == "M0"]),
                                                 RES_ORBI, False), set())
        assert dict(zip(zip(ref.neutral, ref.adduct), ref.level))[("C10H16O4", "[M-H]-")] == want


def test_a_generic_list_entry_is_exempt_from_the_position_test():
    """An 'M+n' entry names no exact shift: like an 'M+n' child it is judged by its band alone. The list names
    an 'M+2' line 5 mDa above the 13C2 position at the height the ion's lines within 12 mDa give: 4b."""
    shift = 2 * SP["13C"] + 0.005
    e, _s = I.generic_expectation({"C": 10, "H": 15, "O": 4}, {}, shift, 0.012)
    p, ent = _listed(1e6, 1e6 * e, label="M+2", shift=shift)
    assert e > 0 and _level(_both(background() + [p, ent]), "C10H16O4") == "4b"


def test_a_guarded_tof_file_does_not_position_test_its_list():
    """Per file, a TOF-class file whose 13C fit sits on the intercept floor is not position-tested (the
    guard): neither its children nor its list entries. A list entry 6 ppm off, in band, holds iso there (4b);
    pooled (no guard) or on an Orbitrap-class file it drops (4c)."""
    p, ent = _listed(1e6, 1.07e5, entry_off_ppm=6.0)
    rows = background(a=0.0, b=6.0, exact=True) + [p, ent]
    assert _level(_both(rows, RES_TOF, per_file=True), "C10H16O4") == "4b"
    assert _level(_both(rows, RES_TOF), "C10H16O4") == "4c"
    assert _level(_both(rows, RES_ORBI, per_file=True), "C10H16O4") == "4c"


# --------------------------------------------------------------------------- the 'M+n' window (D2)
def test_the_m_plus_n_window_is_12_mda_or_half_the_fwhm():
    """An 'M+2' child reads the ion's lines within max(12 mDa, FWHM/2) of its measured shift. Orbitrap
    (FWHM/2 ~1 mDa): a line 8 mDa above the 13C2 position reads the lines within 12 mDa, in band -> iso.
    TOF at m/z 201 (FWHM/2 ~10.4 mDa): a line 14 mDa from every M+2 line expects 0 (a FWHM-wide window would
    sum them). TOF at m/z 401 (FWHM/2 ~20.7 mDa): a line 16 mDa above 13C2 reads them -> iso."""
    fw = EV.instrument(RES_TOF)[1]
    p = parent("a", "C10H16O4", height=1e5)
    c10 = {"C": 10, "H": 15, "O": 4}
    e, _s = I.generic_expectation(c10, {}, 2 * SP["13C"] + 0.008, 0.012)
    rows = background() + [p, kid("m", p, "M+2", 1e5 * e, 2 * SP["13C"] + 0.008)]
    assert e > 0 and _fact(_both(rows, RES_ORBI), "C10H16O4").iso
    at = 2 * SP["13C"] + 0.014
    wide, _s = I.generic_expectation(c10, {}, at, float(fw(p["_true"] + at)))
    rows = background(a=2.5, b=1.8) + [p, kid("m", p, "M+2", 1e5 * wide, at)]
    assert wide > 0 and 0.012 > float(fw(p["_true"] + at)) / 2
    assert not _fact(_both(rows, RES_TOF), "C10H16O4").iso
    q = parent("b", "C20H32O8", height=1e5)
    at = 2 * SP["13C"] + 0.016
    half = float(fw(q["_true"] + at)) / 2
    e20, _s = I.generic_expectation({"C": 20, "H": 31, "O": 8}, {}, at, half)
    rows = background(a=2.5, b=1.8) + [q, kid("m", q, "M+2", 1e5 * e20, at)]
    assert half > 0.016 > 0.012 and e20 > 0
    assert _fact(_both(rows, RES_TOF), "C20H32O8").iso


# --------------------------------------------------------------------------- 13C2 is a carbon line (D10, I2)
def test_a_13c2_line_is_a_carbon_line():
    """'13C2' adds 13C: a bromide cluster with its reagent 81Br line and a placed, in-band 13C2 line (no 13C
    line) has carbon -- 4b, not the reagent-only 4d."""
    p = parent("a", "C22H42O6", "[M+Br]-", height=1e5)
    rows = background() + _bromide(p, kid("b", p, "81Br", 0.97e5, SP["81Br"]),
                                   kid("c2", p, "13C2", 1e5 * 231 * 0.0107 ** 2, 2 * SP["13C"]))
    got = _both(rows)
    f = _fact(got, "C22H42O6", "[M+Br]-")
    assert f.carbon_ev and not f.reagent_only_iso and _level(got, "C22H42O6", "[M+Br]-") == "4b"


# --------------------------------------------------------------------------- a labelled adduct read from neutral + adduct
def test_an_unsigned_labelled_row_is_read_with_its_15n():
    """D4: the ion's counts come from evidence.ion_composition -- neutral + adduct when the stored ion string
    carries no sign (a ledger row may hold the bare neutral). A labelled reagent's '^N' must stay 15N there:
    read as 14N the ion's mono m/z sits 0.997 Da low, the committed line of a Cl4 [M+^NO3]- (its 37Cl1
    line) is lost and its 37Cl2 line reads against the mono line (1.28 expected for 0.48 seen: out of band).
    Signed ('C10H18Cl4O3^N-', the oracle's spelling) or unsigned, the pair is levelled alike -- in the
    engine and the script."""
    true = C.ion_mz("C10H18Cl4", "[M+^NO3]-") + SP["37Cl"]              # the 37Cl1 line, committed
    for stored in ("C10H18Cl4O3^N-", "C10H18Cl4"):
        p = m0("a", "C10H18Cl4", adduct="[M+^NO3]-", ion=stored, mz=true, height=1e5)
        p.update(ppm_error_cal=0.0, _true=true)
        rows = background() + [p, kid("c", p, "37Cl", 0.479e5, SP["37Cl"])]
        f = _fact(_both(rows), "C10H18Cl4", "[M+^NO3]-")
        assert f.iso and f.iso_labels == "37Cl", stored
    counts, _ion = ion_of("C10H18Cl4", "[M+^NO3]-")
    assert counts.get("^N") == 1 and "N" not in counts
