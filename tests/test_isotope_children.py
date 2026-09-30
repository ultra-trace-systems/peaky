"""The isotope child judged against its committed parent (C11+c) --
peaky/chem/isotopes.py's shared helper and its standalone twin in
scripts/level_ledger.py.

A child line counts as isotope evidence where it sits at its label's exact
spacing from the COMMITTED parent line (the scorer commits an ion's most
abundant isotopologue: a Br2 ion on 79Br81Br) and is as tall as the ion's
composition makes it relative to that line. The label is read under both
conventions in use (scorer labels count from the mono line, peaky's own from the
parent line) and the reading nearer the measured shift wins. The position window
is max(1 ppm, 4 sigma(h)), sigma(h) fitted on the source's own '13C' children.

Every synthetic line here sits at the exact m/z its configuration has.

Run: pytest tests/test_isotope_children.py -q
"""

from __future__ import annotations

import importlib.util
import inspect
import math
from pathlib import Path

import numpy as np
import pytest

from peaky.assignment import evidence as EV
from peaky.chem import chemistry as C
from peaky.chem import isotopes as I

_spec = importlib.util.spec_from_file_location(
    "level_ledger_iso", Path(__file__).resolve().parents[1] / "scripts" / "level_ledger.py")
LL = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(LL)

#: every name of the helper section; the script's copy must be the same text
SHARED = ["split_label", "parse_label_part", "heavy_key", "heavy_shift", "_heavy_nominal", "_heavy_add",
          "heavy_probability", "ion_sign", "mono_mz", "_committed_candidates", "committed_tolerance_ppm",
          "committed_configuration", "_ion_lines", "generic_expectation", "_label_elements", "_expand",
          "resolve_child", "reagent_part", "most_probable_heavy", "full_count_line", "fit_position_sigma",
          "position_window_ppm", "_in_band", "_placed", "_pulled", "judge_source", "_reagent_line", "line_facts"]
CONSTANTS = ["HEAVY_ISOTOPES", "ELEMENT_MASS", "ELECTRON_MASS", "ISOTOPE_SPACING", "ISOTOPE_ELEMENT", "ISOTOPE_RATIO",
             "M2_OWNERS", "COMMITTED_ISOTOPES", "COMMITTED_MIN_P", "COMMITTED_TOL_PPM", "COMMITTED_TOL_CLASSLESS_PPM",
             "GENERIC_HALF_WIDTH_DA", "POSITION_MIN_PPM", "POSITION_K", "SIGMA_MIN_CHILDREN", "SIGMA_BINS",
             "SIGMA_MIN_PER_BIN", "SIGMA_A_FLOOR_PPM", "SIGMA_CLIP_PPM", "SIGMA_CLIP_K", "MAD_TO_SIGMA",
             "NEIGHBOUR_RATIO", "NEIGHBOUR_REACH_PPM", "NEIGHBOUR_FRACTION", "RATIO_BAND", "_ENUM_ISOTOPES"]


def ion_mz(counts: dict, sign: str = "-") -> float:
    return I.mono_mz(counts, sign)


# --------------------------------------------------------------------------- tables
def test_spacings_are_the_exact_isotope_mass_differences():
    sp = I.ISOTOPE_SPACING
    assert sp["13C"] == pytest.approx(1.0033548378, abs=1e-10)
    assert sp["81Br"] == pytest.approx(80.9162906 - C.M["Br"], abs=1e-10) == pytest.approx(1.9979535, abs=1e-9)
    assert sp["37Cl"] == pytest.approx(1.9970499, abs=1e-7)
    assert sp["34S"] == pytest.approx(1.9957963, abs=1e-7)
    assert sp["29Si"] == pytest.approx(0.9995681, abs=1e-7) and sp["30Si"] == pytest.approx(1.9968436, abs=1e-7)
    assert sp["18O"] == pytest.approx(2.004245, abs=1e-7) and sp["15N"] == pytest.approx(C.M["^N"] - C.M["N"])
    assert sp["14N"] == -sp["15N"]
    # the light masses are chemistry's, so an ion's mono m/z matches ion_mz
    assert I.mono_mz({"C": 10, "H": 15, "O": 4}, "-") == pytest.approx(C.ion_mz("C10H16O4", "[M-H]-"), abs=1e-9)


def test_the_per_atom_ratios_are_the_evidence_bands():
    for tag, r in EV.ISOTOPE_ABUNDANCE.items():
        assert I.ISOTOPE_RATIO[tag] == r
    assert I.ISOTOPE_RATIO["13C"] == EV.C13_PER_CARBON
    for tag, (_el, r) in EV.PER_ATOM_ABUNDANCE.items():
        assert I.ISOTOPE_RATIO[tag] == r
    assert I.RATIO_BAND == (EV.RATIO_LO, EV.RATIO_HI)


# --------------------------------------------------------------------------- grammar
@pytest.mark.parametrize("label, parts", [
    ("13C", ["13C"]), ("13C+81Br", ["13C", "81Br"]), ("81Br+13C", ["81Br", "13C"]),
    ("13C+1", ["13C"]), ("81Br+2", ["81Br"]), ("13C2+2", ["13C2"]),        # the tests' legacy nominal note
    ("M+5", ["M+5"]), ("13C+M+4", ["13C", "M+4"]), ("2x81Br+M0", ["2x81Br", "M0"]),
    ("81Br/37Cl(pair)", ["81Br/37Cl(pair)"]), ("13C+81Br+37Cl3", ["13C", "81Br", "37Cl3"]),
])
def test_split_label(label, parts):
    assert I.split_label(label) == parts


@pytest.mark.parametrize("part, parsed", [
    ("13C", ("set", {"13C": 1})), ("13C2", ("set", {"13C": 2})), ("81Br2", ("set", {"81Br": 2})),
    ("2x81Br", ("set", {"81Br": 2})), ("3x37Cl", ("set", {"37Cl": 3})), ("37Cl3", ("set", {"37Cl": 3})),
    ("29Si2", ("set", {"29Si": 2})), ("81Br(pair)", ("set", {"81Br": 1})), ("37Cl(pair)", ("set", {"37Cl": 1})),
    ("81Br37Cl(pair)", ("set", {"81Br": 1, "37Cl": 1})),
    ("81Br/37Cl(pair)", ("alt", [{"81Br": 1}, {"37Cl": 1}])),
    ("M0", ("mono", {})), ("M+4", ("gen", 4)), ("14N", ("set", {"14N": 1})),
    ("M", ("bad", None)), ("12C", ("bad", None)), ("Br81", ("bad", None)), ("", ("bad", None)),
])
def test_parse_label_part(part, parsed):
    assert I.parse_label_part(part) == parsed


# --------------------------------------------------------------------------- the expectation
def test_heavy_probability_is_the_multinomial_over_the_ions_atoms():
    br2 = {"C": 2, "H": 1, "Br": 2, "O": 2}
    assert I.heavy_probability({"81Br": 1}, br2) == pytest.approx(2 * 0.9728)
    assert I.heavy_probability({"81Br": 2}, br2) == pytest.approx(0.9728 ** 2)
    assert I.heavy_probability({"81Br": 3}, br2) == 0.0                     # more heavy atoms than the ion has
    assert I.heavy_probability({"37Cl": 1}, br2) == 0.0                     # an element it lacks
    assert I.heavy_probability({"13C": 2}, {"C": 10}) == pytest.approx(45 * 0.0107 ** 2)   # I2: C(n, 2) r^2
    assert I.heavy_probability({"13C": 1, "81Br": 1}, br2) == pytest.approx(2 * 0.0107 * 2 * 0.9728)
    assert I.heavy_probability({"29Si": 1, "30Si": 1}, {"Si": 7}) == pytest.approx(42 * 0.0508 * 0.0335)


def test_the_committed_line_is_the_most_probable_configuration_at_the_parents_mass():
    dba = {"C": 2, "H": 1, "Br": 2, "O": 2}                       # dibromoacetic acid [M-H]-
    mono = ion_mz(dba)
    br = I.ISOTOPE_SPACING["81Br"]
    assert I.committed_configuration(dba, br, mono + br, 5.0) == {"81Br": 1}
    assert I.committed_configuration(dba, 0.0, mono, 5.0) == {}
    assert I.committed_configuration(dba, 2 * br, mono + 2 * br, 5.0) == {"81Br": 2}
    # 3 ppm off still names it on an Orbitrap; 8 ppm off only within the TOF tolerance; none -> mono
    off = (mono + br) * 8e-6
    assert I.committed_configuration(dba, br + off, mono + br, 5.0) == {}
    assert I.committed_configuration(dba, br + off, mono + br, 20.0) == {"81Br": 1}
    # Cl4 is committed on a 37Cl line (4 x 0.3196 > 1), Cl3 on its mono line
    cl4 = {"C": 10, "H": 18, "Cl": 4, "N": 1, "O": 3}
    assert I.committed_configuration(cl4, I.ISOTOPE_SPACING["37Cl"], 400.0, 5.0) == {"37Cl": 1}
    # a Br1Cl1 ion: 81Br35Cl and 79Br37Cl are 0.9 mDa apart; within the window the more probable wins
    brcl = {"C": 2, "H": 2, "Br": 1, "Cl": 1, "O": 2}
    mid = (I.ISOTOPE_SPACING["81Br"] + I.ISOTOPE_SPACING["37Cl"]) / 2
    assert I.committed_configuration(brcl, mid, 200.0, 5.0) == {"81Br": 1}
    assert I.committed_configuration(brcl, float("nan"), 200.0, 5.0) == {}
    assert I.committed_tolerance_ppm("orbitrap") == 5.0 and I.committed_tolerance_ppm("tof") == 20.0
    assert I.committed_tolerance_ppm(None) == 20.0


def _resolve(label, counts, hp, delta):
    return I.resolve_child(label, counts, hp, delta)


def test_a_br2_ion_committed_on_79br81br_reads_both_neighbour_lines():
    """R1's dibromoacetic acid: '81Br2' (scorer, mono-counted) 1.998 Da above
    and 'M0' (the 79Br2 line) 1.998 Da below the 79Br81Br parent -- 0.486 and
    0.514 of it."""
    dba = {"C": 2, "H": 1, "Br": 2, "O": 2}
    br = I.ISOTOPE_SPACING["81Br"]
    up = _resolve("81Br2", dba, {"81Br": 1}, br)
    assert up["expected"] == pytest.approx(0.9728 / 2) and up["shift"] == pytest.approx(br)
    assert up["elements"] == ["Br"] and up["kind"] == "set"
    down = _resolve("M0", dba, {"81Br": 1}, -br)
    assert down["expected"] == pytest.approx(1 / (2 * 0.9728)) and down["readings"] == pytest.approx((-br,))
    assert down["kind"] == "mono" and down["elements"] == ["Br"] and down["parts"] == []
    # peaky's parent-relative '81Br' names the same upper line
    rel = _resolve("81Br", dba, {"81Br": 1}, br)
    assert rel["expected"] == pytest.approx(0.9728 / 2)
    # an 'M0' child of a MONO-committed parent is no line of the ion (2 Da below its lightest line)
    assert _resolve("M0", dba, {}, -br)["expected"] == 0.0


def test_count_labels_count():
    br2 = {"C": 8, "H": 13, "Br": 2, "O": 4}
    br = I.ISOTOPE_SPACING["81Br"]
    assert _resolve("81Br2", br2, {}, 2 * br)["expected"] == pytest.approx(0.9728 ** 2)
    assert _resolve("2x81Br", br2, {}, 2 * br)["expected"] == pytest.approx(0.9728 ** 2)
    assert _resolve("81Br", br2, {}, br)["expected"] == pytest.approx(2 * 0.9728)
    cl3 = {"C": 6, "H": 7, "Cl": 3, "O": 2}
    assert _resolve("37Cl3", cl3, {}, 3 * I.ISOTOPE_SPACING["37Cl"])["expected"] == pytest.approx(0.3196 ** 3)
    si7 = {"C": 14, "H": 42, "O": 7, "Si": 7}
    assert _resolve("29Si", si7, {}, I.ISOTOPE_SPACING["29Si"])["expected"] == pytest.approx(7 * 0.0508)
    assert _resolve("29Si2", si7, {}, 2 * I.ISOTOPE_SPACING["29Si"])["expected"] == pytest.approx(21 * 0.0508 ** 2)
    assert _resolve("30Si", si7, {}, I.ISOTOPE_SPACING["30Si"])["expected"] == pytest.approx(7 * 0.0335)
    # 13C2 has its own expectation (I2); a '13C' label still per carbon
    c10 = {"C": 10, "H": 15, "O": 4}
    assert _resolve("13C2", c10, {}, 2 * I.ISOTOPE_SPACING["13C"])["expected"] == pytest.approx(45 * 0.0107 ** 2)
    assert _resolve("13C", c10, {}, I.ISOTOPE_SPACING["13C"])["expected"] == pytest.approx(0.107)
    assert _resolve("13C+1", c10, {}, I.ISOTOPE_SPACING["13C"])["expected"] == pytest.approx(0.107)


def test_a_line_the_ion_cannot_make_expects_nothing():
    brfree = {"C": 31, "H": 33, "O": 10}
    assert _resolve("81Br", brfree, {}, I.ISOTOPE_SPACING["81Br"])["expected"] == 0.0
    # S2: one 34S line is 2 x 0.0443 -- a 0.0435 line (the D4 / S2 isobar's) is out of band
    s2 = {"C": 8, "H": 13, "O": 4, "S": 2}
    e = _resolve("34S", s2, {}, I.ISOTOPE_SPACING["34S"])["expected"]
    assert e == pytest.approx(2 * 0.0443) and not I._in_band(0.0435, e)
    # 18O on a Br / Cl ion is not measured; 14N is rule K's
    assert _resolve("18O", {"C": 5, "H": 8, "Br": 1, "O": 4}, {}, I.ISOTOPE_SPACING["18O"])["expected"] == 0.0
    assert _resolve("18O", {"C": 5, "H": 8, "O": 4}, {}, I.ISOTOPE_SPACING["18O"])["expected"] == \
        pytest.approx(4 * 0.00205 / 0.99757)
    assert _resolve("14N", {"C": 5, "H": 8, "^N": 1, "O": 7}, {}, I.ISOTOPE_SPACING["14N"])["expected"] == 0.0
    assert _resolve("banana", {"C": 5}, {}, 1.0)["kind"] == "bad"


def test_a_joint_line_is_the_joint_probability_and_credits_what_it_adds():
    """Chloroacetic acid on bromide: C2H3ClO2 [M+Br]- (Br1 Cl1). Its M+4
    '81Br+37Cl' line is 0.9728 x 0.3196 of the mono line and names both
    elements; a Cl-free bromide cluster makes no such line."""
    ion = {"C": 2, "H": 3, "Cl": 1, "O": 2, "Br": 1}
    m4 = I.ISOTOPE_SPACING["81Br"] + I.ISOTOPE_SPACING["37Cl"]
    r = _resolve("81Br+37Cl", ion, {}, m4)
    assert r["expected"] == pytest.approx(0.9728 * 0.3196) and r["elements"] == ["Br", "Cl"]
    assert r["parts"] == ["81Br", "37Cl"] and r["shift"] == pytest.approx(m4)
    assert _resolve("81Br+37Cl", {"C": 2, "H": 3, "O": 2, "Br": 1}, {}, m4)["expected"] == 0.0
    # the M+3 '81Br+13C' line of a bromide cluster adds carbon (and Br)
    r = _resolve("81Br+13C", {"C": 7, "H": 6, "N": 2, "O": 6, "S": 1, "Br": 1}, {},
                 I.ISOTOPE_SPACING["81Br"] + I.ISOTOPE_SPACING["13C"])
    assert set(r["elements"]) == {"Br", "C"} and r["expected"] == pytest.approx(0.9728 * 7 * 0.0107)
    # a scorer '13C+81Br' on a Br3 ion committed on 79Br2 81Br is the parent's own 13C line: carbon only
    br3 = {"C": 19, "H": 28, "Br": 3, "N": 2, "O": 1}
    r = _resolve("13C+81Br", br3, {"81Br": 1}, I.ISOTOPE_SPACING["13C"])
    assert r["elements"] == ["C"] and r["expected"] == pytest.approx(19 * 0.0107)
    # either line of a BrCl doublet ('81Br/37Cl(pair)'): the probability-weighted blend of the two
    brcl = {"C": 2, "H": 2, "Br": 1, "Cl": 1, "O": 2}
    r = _resolve("81Br/37Cl(pair)", brcl, {}, 1.9977)
    assert r["expected"] == pytest.approx(0.9728 + 0.3196)
    w = (0.9728 * I.ISOTOPE_SPACING["81Br"] + 0.3196 * I.ISOTOPE_SPACING["37Cl"]) / (0.9728 + 0.3196)
    assert r["shift"] == pytest.approx(w)
    r = _resolve("81Br37Cl(pair)", brcl, {}, m4)
    assert r["expected"] == pytest.approx(0.9728 * 0.3196)


def test_the_nearer_reading_wins():
    """A 79Br81Br-committed Br2 ion's '81Br' child: parent-relative it is the
    81Br2 line (+1.998), mono-counted the parent's own position (0): the child's
    measured shift decides; both readings are the position test's."""
    br2 = {"C": 2, "H": 1, "Br": 2, "O": 2}
    br = I.ISOTOPE_SPACING["81Br"]
    r = _resolve("81Br", br2, {"81Br": 1}, br)
    assert r["readings"] == pytest.approx((br, 0.0)) and r["shift"] == pytest.approx(br)
    r = _resolve("81Br", br2, {"81Br": 1}, 0.0005)
    assert r["shift"] == pytest.approx(0.0)


def test_a_generic_line_reads_the_ions_lines_in_its_window():
    c10 = {"C": 10, "H": 15, "O": 4}
    e, s = I.generic_expectation(c10, {}, 2.005, 0.012)
    lines = [p for _k, sh, p in I._ion_lines(I.heavy_key(c10)) if abs(sh - 2.005) <= 0.012]
    assert e == pytest.approx(sum(lines)) and 2.0 < s < 2.01
    r = _resolve("M+2", c10, {}, 2.005)
    assert r["kind"] == "gen" and r["readings"] == () and r["elements"] == [] and r["expected"] == pytest.approx(e)
    assert I.generic_expectation(c10, {}, 7.5, 0.012)[0] == 0.0


def test_a_reagent_part_names_the_reagent_halogen_alone():
    for p in ("81Br", "81Br2", "2x81Br", "81Br(pair)"):
        assert I.reagent_part(p, "81Br"), p
    for p in ("13C", "81Br37Cl(pair)", "81Br/37Cl(pair)", "M+4", "37Cl", "banana"):
        assert not I.reagent_part(p, "81Br"), p
    assert not I.reagent_part("81Br", None)


# --------------------------------------------------------------------------- sigma(h)
def _synthetic_13c(a, b, n=400, seed=7):
    rng = np.random.default_rng(seed)
    h = 10 ** rng.uniform(1.5, 5.0, n)
    r = rng.normal(0.0, np.sqrt(a ** 2 + b ** 2 / h))
    return r, h


def _exact_13c(a, b, heights=(30, 100, 300, 1000, 3000, 10000, 30000, 100000)):
    """Ten '13C' residuals per height whose 1.4826 x MAD is exactly
    sqrt(a^2 + b^2 / h): the estimator must return (a, b)."""
    pat = np.array([-3, -1, -1, -1, 0, 0, 1, 1, 1, 3], float)
    r, h = [], []
    for hi in heights:
        r += list(pat * math.sqrt(a * a + b * b / hi) / I.MAD_TO_SIGMA)
        h += [hi] * len(pat)
    return np.array(r), np.array(h)


def test_sigma_h_is_fitted_on_the_sources_own_13c_children():
    fit = I.fit_position_sigma(*_exact_13c(0.3, 6.0))
    assert (fit.a, fit.b, fit.floored, fit.n) == (pytest.approx(0.3), pytest.approx(6.0), False, 80)
    fit = I.fit_position_sigma(*_exact_13c(2.5, 1.8))                          # a TOF-like scatter
    assert (fit.a, fit.b) == (pytest.approx(2.5), pytest.approx(1.8))
    assert I.position_window_ppm(1e6, fit) == pytest.approx(4 * math.hypot(2.5, 1.8 / 1e3))
    fit = I.fit_position_sigma(*_exact_13c(0.2, 6.0))                          # an Orbitrap-like scatter
    assert I.position_window_ppm(1e6, fit) == 1.0 and I.position_window_ppm(100.0, fit) == pytest.approx(
        4 * math.sqrt(0.04 + 0.36))
    r, h = _exact_13c(0.3, 6.0)
    assert I.fit_position_sigma(r[:39], h[:39]) is None                        # < 40 children: no test
    assert I.fit_position_sigma(np.r_[r[:39], np.nan], np.r_[h[:39], 10.0]) is None
    # no height-independent scatter: the intercept sits on the floor, and says so
    fit = I.fit_position_sigma(*_exact_13c(0.0, 6.0))
    assert fit.a == pytest.approx(0.02) and fit.b == pytest.approx(6.0) and fit.floored
    assert np.isnan(I.position_window_ppm(float("nan"), fit)) and np.isnan(I.position_window_ppm(0.0, fit))
    # the pre-clip: a 40-ppm mislinked line does not widen the window
    r2, h2 = np.r_[r, 40.0, -40.0], np.r_[h, 1000.0, 1000.0]
    assert I.fit_position_sigma(r2, h2).b == pytest.approx(6.0, rel=0.02)


def test_the_estimator_is_pinned():
    """Binned-MAD WLS, not a Gaussian ML (R1's window median 1.07 vs 1.41 ppm):
    the numbers of one fixed sample."""
    r, h = _synthetic_13c(0.4, 5.0, n=240, seed=3)
    fit = I.fit_position_sigma(r, h)
    assert (round(fit.a, 4), round(fit.b, 4), fit.n) == (FIT_PIN["a"], FIT_PIN["b"], FIT_PIN["n"])


FIT_PIN = {"a": 0.3888, "b": 3.7752, "n": 240}


def test_the_decided_constants_hold_their_values():
    """The binding C11+c numbers (BUILD_SPEC 2026-09-30, D2-D4): the text-identity test above compares the
    two copies with each other; this pins them to the decision."""
    assert I.COMMITTED_TOL_PPM == {"orbitrap": 5.0, "tof": 20.0} and I.COMMITTED_TOL_CLASSLESS_PPM == 20.0
    assert I.COMMITTED_MIN_P == 1e-4 and I.GENERIC_HALF_WIDTH_DA == 0.012
    assert (I.POSITION_MIN_PPM, I.POSITION_K) == (1.0, 4.0)
    assert (I.SIGMA_MIN_CHILDREN, I.SIGMA_BINS, I.SIGMA_MIN_PER_BIN) == (40, 8, 5)
    assert (I.SIGMA_A_FLOOR_PPM, I.SIGMA_CLIP_PPM, I.SIGMA_CLIP_K, I.MAD_TO_SIGMA) == (0.02, 5.0, 6.0, 1.4826)
    assert (I.NEIGHBOUR_RATIO, I.NEIGHBOUR_REACH_PPM, I.NEIGHBOUR_FRACTION) == (3.0, 25.0, 0.5)
    assert I.RATIO_BAND == (0.5, 2.0) and EV.ORBITRAP_R200 == 50_000.0


def _mislinked(seed, a, b, out, n=240, n_out=24):
    """A source's '13C' residuals: n lines scattering as sigma(h)^2 = a^2 + b^2 / h, plus n_out BRIGHT
    children +-`out` ppm off (mislinked lines of another species) -- what the pre-clip is for."""
    rng = np.random.default_rng(seed)
    h = 10 ** rng.uniform(1.5, 5.0, n)
    r = rng.normal(0.0, np.sqrt(a * a + b * b / h))
    ho = 10 ** rng.uniform(3.0, 5.0, n_out)
    ro = rng.choice([-1, 1], n_out) * out
    return np.r_[r, ro], np.r_[h, ho]


def test_the_estimator_at_its_edges():
    """Samples that sit where each piece of the pinned estimator matters (the equal-bin sample above sees
    none of them):
    - 40 children is a fit (39 is none);
    - the pre-clip floor is 5 ppm: Orbitrap-like (a 0.4, b 5) with 10 % bright lines 4 ppm off -- 6 x MAD
      (2.9 ppm) is below the floor, so the 4-ppm lines stay in the fit (a 3-ppm floor would clip them);
    - the pre-clip is 6 x MAD: TOF-like (a 2.5, b 1.8; MAD sigma 2.66 ppm) with 10 % bright lines 12 ppm off
      -- 16 ppm keeps them (4 x MAD would clip them); the WLS slope comes out negative there and b clamps at 0;
    - the bins are weighted by their counts: a flat-topped source (40 lines at 30 cps, 40 at 1e5 cps, 10 at
      each height between) gives quantile bins of unequal counts (unweighted: a 0.307);
    - a thin source (46 children, bins of 5-6) keeps every bin (a minimum of 6 per bin drops two)."""
    r, h = _exact_13c(0.3, 6.0)
    assert I.fit_position_sigma(r[:40], h[:40]) is not None and I.fit_position_sigma(r[:39], h[:39]) is None
    fit = I.fit_position_sigma(*_mislinked(5, 0.4, 5.0, 4.0))
    assert (round(fit.a, 3), round(fit.b, 3), fit.n) == (0.454, 5.157, 264)
    fit = I.fit_position_sigma(*_mislinked(5, 2.5, 1.8, 12.0))
    assert (round(fit.a, 3), fit.b, fit.n, fit.floored) == (2.899, 0.0, 264, False)
    pat = np.array([-3, -1, -1, -1, 0, 0, 1, 1, 1, 3], float) / I.MAD_TO_SIGMA
    r, h = [], []
    for hi, n in [(30, 40), (100, 10), (300, 10), (1000, 10), (3000, 10), (1e4, 10), (3e4, 10), (1e5, 40)]:
        r += list(np.resize(pat, n) * math.sqrt(0.09 + 36.0 / hi))
        h += [hi] * n
    fit = I.fit_position_sigma(np.array(r), np.array(h))
    assert (round(fit.a, 3), round(fit.b, 3)) == (0.304, 5.994)
    rng = np.random.default_rng(3)
    h = 10 ** rng.uniform(1.5, 5.0, 46)
    fit = I.fit_position_sigma(rng.normal(0.0, np.sqrt(0.16 + 25.0 / h)), h)
    assert (round(fit.a, 4), round(fit.b, 4), fit.n) == (0.3962, 9.6246, 46)


# --------------------------------------------------------------------------- the twin
def test_the_script_carries_the_same_functions_text_for_text():
    for name in SHARED:
        a, b = getattr(I, name), getattr(LL, name)
        a, b = getattr(a, "__wrapped__", a), getattr(b, "__wrapped__", b)
        assert inspect.getsource(a) == inspect.getsource(b), name
    assert I.PositionSigma._fields == LL.PositionSigma._fields == ("a", "b", "floored", "n")
    for name in CONSTANTS:
        assert getattr(I, name) == getattr(LL, name), name


GRID_LABELS = ["13C", "13C2", "13C+81Br", "81Br+13C", "81Br", "81Br2", "2x81Br", "81Br(pair)", "37Cl", "37Cl(pair)",
               "81Br/37Cl(pair)", "81Br37Cl(pair)", "81Br+37Cl", "34S", "29Si", "30Si", "29Si2", "18O", "15N", "14N",
               "M0", "M+2", "M+4", "2x81Br+M0", "13C+1", "garbage"]
GRID_IONS = [({"C": 2, "H": 1, "Br": 2, "O": 2}, "-"), ({"C": 2, "H": 3, "Cl": 1, "O": 2, "Br": 1}, "-"),
             ({"C": 10, "H": 18, "Cl": 4, "N": 1, "O": 3}, "-"), ({"C": 14, "H": 43, "O": 7, "Si": 7}, "+"),
             ({"C": 10, "H": 15, "O": 5}, "-"), ({"C": 8, "H": 13, "O": 4, "S": 2}, "-"),
             ({"C": 10, "H": 16, "^N": 1, "O": 8}, "-"), ({"C": 19, "H": 28, "Br": 3, "N": 2, "O": 1}, "-")]


def _same(a, b) -> bool:
    """Equal, NaN equal to NaN (a dict / sequence compared item by item)."""
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return a == b


def test_the_script_and_the_engine_read_every_label_alike():
    for counts, sign in GRID_IONS:
        mono = I.mono_mz(counts, sign)
        assert LL.mono_mz(counts, sign) == mono
        for cfg in [{}, *[dict(k) for k, _s, _p in I._committed_candidates(I.heavy_key(
                {e: n for e, n in counts.items() if e in ("Br", "Cl", "S", "Si")}))]]:
            pmz = mono + I.heavy_shift(cfg)
            for tol in (5.0, 20.0):
                assert I.committed_configuration(counts, pmz - mono, pmz, tol) == \
                    LL.committed_configuration(counts, pmz - mono, pmz, tol)
            for label in GRID_LABELS:
                for delta in (-1.998, 0.0, 1.0034, 1.998, 2.0067, 3.0013, 3.996):
                    assert _same(I.resolve_child(label, counts, cfg, delta),
                                 LL.resolve_child(label, counts, cfg, delta)), (label, counts, cfg, delta)
    for label in GRID_LABELS:
        for p in I.split_label(label):
            assert I.parse_label_part(p) == LL.parse_label_part(p)
            assert I.reagent_part(p, "81Br") == LL.reagent_part(p, "81Br")
    r, h = _synthetic_13c(0.5, 4.0)
    assert tuple(I.fit_position_sigma(r, h)) == tuple(LL.fit_position_sigma(r, h))


# --------------------------------------------------------------------------- one source, judged
def _cho(nc: int) -> dict:
    """A C{nc}H{2nc-5}O4 [M-H]- ion (a plausible CHO acid)."""
    return {"C": nc, "H": 2 * nc - 5, "O": 4}


def _source(extra_children=(), extra_parents=None, n_fit=400, a=0.2, b=6.0, extra_rows=()):
    """A synthetic Orbitrap-like source: n_fit CHO parents at their exact m/z,
    each with a '13C' child at the exact spacing plus a residual drawn so the
    source's fit is (a, b) (`_exact_13c`); then the test's own rows
    (`extra_rows`: (peak_id, mz, height) iso rows of the file under no parent)."""
    r, h = _exact_13c(a, b)
    parents, children = {}, []
    for i in range(n_fit):
        counts = _cho(5 + i % 16)
        pmz = I.mono_mz(counts, "-") + 0.0 * i
        pid = f"p{i}"
        parents[("f", pid)] = dict(mz=pmz, height=1e7, pcal=0.0, counts=counts, sign="-")
        cm = pmz + I.ISOTOPE_SPACING["13C"] + r[i % len(r)] * pmz / 1e6
        children.append(dict(file="f", peak_id=f"c{i}", parent=pid, label="13C", mz=cm, height=float(h[i % len(h)])))
    parents.update(extra_parents or {})
    children.extend(extra_children)
    fm = np.array([p["mz"] for p in parents.values()] + [c["mz"] for c in children] + [r[1] for r in extra_rows],
                  float)
    fh = np.array([p["height"] for p in parents.values()] + [c["height"] for c in children]
                  + [r[2] for r in extra_rows], float)
    fp = np.array([k[1] for k in parents] + [c["peak_id"] for c in children] + [r[0] for r in extra_rows], object)
    return parents, children, {"f": (fm, fh, fp)}


def _verdicts(parents, children, rows, lists=(), **kw):
    got = I.judge_source(children, parents, list(lists), rows, **kw)
    twin = LL.judge_source(children, parents, list(lists), rows, **kw)
    assert _same(got, twin) or (_same(got["children"], twin["children"]) and got["lists"] == twin["lists"]
                                and tuple(got["fit"] or ()) == tuple(twin["fit"] or ()))
    return got, {c["peak_id"]: v for c, v in zip(children, got["children"])}


def test_a_child_off_its_exact_position_drops_where_its_height_says_it_can_be_measured():
    """On an Orbitrap-like fit (a 0.2 ppm, b 6 ppm): 0.4 mDa off a C10 line at
    m/z 200 is 2 ppm -- outside the 1-ppm window of a 10 000-cps line, inside
    the ~2.5-ppm window of a 100-cps one."""
    counts = _cho(10)
    pmz = I.mono_mz(counts, "-")
    x = pmz + I.ISOTOPE_SPACING["13C"]
    extra_p = {("f", "q"): dict(mz=pmz, height=1e6, pcal=0.0, counts=counts, sign="-"),
               ("f", "s"): dict(mz=pmz, height=935.0, pcal=0.0, counts=counts, sign="-")}
    kids = [dict(file="f", peak_id="bright", parent="q", label="13C", mz=x + 0.0004, height=1.07e5),
            dict(file="f", peak_id="dim", parent="s", label="13C", mz=x + 0.0004, height=100.0),
            dict(file="f", peak_id="exact", parent="q", label="13C", mz=x, height=1.07e5)]
    got, v = _verdicts(*_source(kids, extra_p), klass="orbitrap")
    # the three test lines join the fit (the fit is (0.2, 6.0) without them): the windows stay ~1 and > 2 ppm
    fit = got["fit"]
    assert got["tested"] and fit.a < 0.3 and fit.b > 5.0
    assert I.position_window_ppm(1.07e5, fit) < 1.5 < 2.2 < I.position_window_ppm(100.0, fit)
    assert not v["bright"]["keep"] and v["dim"]["keep"] and v["exact"]["keep"]
    assert v["exact"]["ok"] and v["dim"]["ok"]                        # 0.107 of the parent: in band
    # the same source with fewer than 40 '13C' children is not tested at all
    got, v = _verdicts(*_source(kids, extra_p, n_fit=0), klass="orbitrap")
    assert not got["tested"] and got["fit"] is None and v["bright"]["keep"]


def test_n1_a_brighter_neighbour_pulls_a_line_toward_itself():
    """An [M]-. line's 13C child pulled +3 ppm toward the 25x brighter [M-H]-
    line of the neutral with two more H, at its own exact m/z 4.47 mDa above
    (the 13C / H doublet; on a TOF / Orbitrap centroid the blend shifts the dim
    line): kept; with the child displaced the other way, the neighbour too dim,
    beyond 25 ppm (the same doublet at m/z 159 is 28 ppm) or the residual more
    than half the distance, dropped."""
    def case(neutral, off_ppm, n_h):
        counts = C.parse_formula(neutral)                          # the [M]-. ion
        nb = dict(counts, H=counts["H"] + 1)                       # [M-H]- of the neutral with two more H
        pmz = I.mono_mz(counts, "-")
        x = pmz + I.ISOTOPE_SPACING["13C"]
        n_mz = I.mono_mz(nb, "-")
        assert abs(n_mz - x - 0.00447) < 1e-5
        p = {("f", "q"): dict(mz=pmz, height=1e5, pcal=0.0, counts=counts, sign="-"),
             ("f", "nb"): dict(mz=n_mz, height=n_h, pcal=0.0, counts=nb, sign="-")}
        kid = [dict(file="f", peak_id="k", parent="q", label="13C", mz=x + off_ppm * 1e-6 * pmz,
                    height=1e5 * counts["C"] * 0.0107)]
        return _verdicts(*_source(kid, p), klass="orbitrap")[1]["k"]["keep"]
    assert case("C10H16O5", 3.0, 2.7e6)                  # >= 3x, same side, 3 ppm <= half of 20.6 ppm
    assert not case("C10H16O5", -3.0, 2.7e6)             # displaced the other way
    assert not case("C10H16O5", 3.0, 2.0e4)              # not 3x the child
    assert not case("C7H10O4", 3.0, 2.7e6)               # the doublet is 28 ppm at m/z 159: beyond 25 ppm
    assert not case("C10H16O5", 11.3, 2.7e6)             # 2.44 mDa off: more than half of 4.47 mDa


def test_pcal_the_calibrated_parent_position_places_a_line_too():
    """A parent line measured 2 ppm high (ppm_error_cal +2): its 13C child sits
    at the TRUE parent + 1.00335, 2 ppm below the measured parent + 1.00335."""
    counts = _cho(12)
    true = I.mono_mz(counts, "-")
    meas = true * (1 + 2e-6)
    p = {("f", "q"): dict(mz=meas, height=1e7, pcal=2.0, counts=counts, sign="-"),
         ("f", "u"): dict(mz=meas, height=1e7, pcal=float("nan"), counts=counts, sign="-")}
    kids = [dict(file="f", peak_id="k", parent="q", label="13C", mz=true + I.ISOTOPE_SPACING["13C"], height=1.3e6),
            dict(file="f", peak_id="k2", parent="u", label="13C", mz=true + I.ISOTOPE_SPACING["13C"], height=1.3e6)]
    _got, v = _verdicts(*_source(kids, p), klass="orbitrap")
    assert v["k"]["keep"] and not v["k2"]["keep"]


def test_generic_lines_are_exempt_and_unreadable_labels_are_not():
    counts = _cho(10)
    pmz = I.mono_mz(counts, "-")
    p = {("f", "q"): dict(mz=pmz, height=1e6, pcal=0.0, counts=counts, sign="-")}
    kids = [dict(file="f", peak_id="g", parent="q", label="M+2", mz=pmz + 2.0071, height=6000.0),
            dict(file="f", peak_id="b", parent="q", label="banana", mz=pmz + 1.0034, height=1e5)]
    got, v = _verdicts(*_source(kids, p), klass="orbitrap")
    assert v["g"]["keep"] and v["g"]["kind"] == "gen" and not v["b"]["keep"]


def test_the_tof_guard_skips_a_source_whose_fit_collapsed_onto_the_floor():
    counts = _cho(10)
    pmz = I.mono_mz(counts, "-")
    p = {("f", "q"): dict(mz=pmz, height=1e6, pcal=0.0, counts=counts, sign="-")}
    kids = [dict(file="f", peak_id="k", parent="q", label="13C", mz=pmz + 1.0034 + 0.004, height=1e5)]
    src = _source(kids, p, a=0.0, b=6.0)
    assert not _verdicts(*src, klass="tof", guard=True)[0]["tested"]
    assert _verdicts(*src, klass="tof", guard=False)[0]["tested"]
    assert _verdicts(*src, klass="orbitrap", guard=True)[0]["tested"]
    assert _verdicts(*src, klass=None, guard=True)[0]["tested"]       # class-less: no guard
    assert not _verdicts(*src, klass="tof", guard=False)[1]["k"]["keep"]


def test_a_list_entry_answers_the_childrens_question():
    """The isotopologues list counts through an entry that is not the parent's
    dropped child, names an M0 / iso row of the file, is placed at its own
    height (pcal allowed, no N1) and is in band."""
    counts = _cho(10)
    pmz = I.mono_mz(counts, "-")
    x = pmz + I.ISOTOPE_SPACING["13C"]
    q2 = _cho(11)                                                     # another acid, at its own m/z (213.1)
    p = {("f", "q"): dict(mz=pmz, height=1e6, pcal=0.0, counts=counts, sign="-"),
         ("f", "q2"): dict(mz=I.mono_mz(q2, "-"), height=1e6, pcal=0.0, counts=q2, sign="-")}
    kids = [dict(file="f", peak_id="off", parent="q", label="13C", mz=x + 0.001, height=1.07e5),
            # a line the ledger labelled '81Br' (dropped: nowhere near +1.998) that the list calls '13C'
            dict(file="f", peak_id="mislabelled", parent="q", label="81Br", mz=x, height=1.07e5),
            # another parent's child (mislinked: 13 Da below it) 1 mDa off this parent's 13C position, in band
            dict(file="f", peak_id="stranger", parent="q2", label="13C", mz=x + 0.001, height=1.07e5)]
    # iso rows of the file at this parent's 13C position under no parent: in band, and 4.7x too tall
    src = _source(kids, p, extra_rows=[("other", x, 1.07e5), ("tall", x, 5e5)])
    lists = [("f", "q", [{"label": "13C", "peak_id": "off"}]),                  # its own dropped child
             ("f", "q", [{"label": "13C", "peak_id": "other"}]),                # another M0 at the exact place
             ("f", "q", [{"label": "13C", "peak_id": "nowhere"}]),              # names no row of the file
             ("f", "q", [{"label": "81Br", "peak_id": "other"}]),               # a line the ion cannot make
             ("f", "q", [{"label": "13C", "score": None, "peak_id": "other"}]),  # a null score reads
             ("f", "q", ["not a dict"]),
             ("f", "q", [{"label": "13C", "peak_id": "mislabelled"}]),          # its dropped child, relabelled
             ("f", "q", [{"label": "13C", "peak_id": "stranger"}]),             # off position at its height
             ("f", "q", [{"label": "13C", "peak_id": "tall"}]),                 # placed, 4.7x the expectation
             ("f", "q", [{"label": "13C", "peak_id": "tall"}, {"label": "13C", "peak_id": "other"}])]
    got, v = _verdicts(*src, lists=lists, klass="orbitrap")
    assert not v["off"]["keep"] and not v["mislabelled"]["keep"]
    assert got["lists"] == [False, True, False, False, True, False, False, False, False, True]


def test_n1_never_counts_the_childs_own_parent():
    """A Br1Cl1 ion committed on 81Br35Cl: a scorer '37Cl' child is the 79Br37Cl
    line, 0.9 mDa BELOW the parent. Displaced toward the parent it is not
    rescued by it -- the parent is no neighbour of its own line."""
    counts = {"C": 2, "H": 2, "Br": 1, "Cl": 1, "O": 2}
    mono = I.mono_mz(counts, "-")
    pmz = mono + I.ISOTOPE_SPACING["81Br"]
    x = mono + I.ISOTOPE_SPACING["37Cl"]                     # 0.9 mDa below the parent
    p = {("f", "q"): dict(mz=pmz, height=1e6, pcal=0.0, counts=counts, sign="-")}
    kid = [dict(file="f", peak_id="k", parent="q", label="37Cl", mz=x + 0.0003, height=3e5)]
    got, v = _verdicts(*_source(kid, p), klass="orbitrap")
    assert I.position_window_ppm(3e5, got["fit"]) * pmz / 1e6 < 0.0003
    assert not v["k"]["keep"]


def test_line_facts_read_the_kept_lines():
    def line(parts, elements, ok, kind="set"):
        return dict(parts=parts, elements=elements, ok=ok, kind=kind)
    f = I.line_facts([line(["81Br"], ["Br"], True)], "81Br")
    assert f == dict(iso=True, lined={"Br"}, carbon=False, labels={"81Br"}, reagent_only=True)
    f = I.line_facts([line(["81Br"], ["Br"], True), line(["81Br", "13C"], ["Br", "C"], False)], "81Br")
    assert f["carbon"] and not f["reagent_only"] and f["lined"] == {"Br"}
    # a pure 'M0' line names what it differs in from the committed line: the lighter line of a
    # heavy-committed Br pattern names Br (the reagent's halogen); of a mono-committed parent, nothing
    f = I.line_facts([line([], ["Br"], True, "mono")], "81Br")
    assert f["iso"] and f["reagent_only"] and f["labels"] == set()
    assert not I.line_facts([line([], [], False, "mono")], "81Br")["reagent_only"]
    assert not I.line_facts([line([], ["Br", "S"], True, "mono")], "81Br")["reagent_only"]
    f = I.line_facts([line(["2x81Br"], ["Br"], False), line(["81Br2"], ["Br"], False)], "81Br")
    assert f["reagent_only"] and not f["iso"]
    f = I.line_facts([line(["M+4"], [], False, "gen")], "81Br")
    assert not f["reagent_only"] and f["labels"] == {"M+4"}
    assert not I.line_facts([line(["81Br"], ["Br"], True)], None)["reagent_only"]
    assert I.line_facts([], "81Br") == dict(iso=False, lined=set(), carbon=False, labels=set(), reagent_only=False)


def test_the_most_probable_line_of_an_n_atom_halogen_ion():
    """k_c(n), the heavy atoms on the line a scorer commits: Br 0, 1, 1, 2, 2 (Br1-Br5: 0.9728 per atom); Cl
    0, 0, 0, 1, 1, 1 (Cl1-Cl6: 0.3196) -- Cl4 is committed on a 37Cl line, Cl3 on its mono line."""
    assert [I.most_probable_heavy(n, "81Br") for n in range(6)] == [0, 0, 1, 1, 2, 2]
    assert [I.most_probable_heavy(n, "37Cl") for n in range(7)] == [0, 0, 0, 0, 1, 1, 1]
    for n in range(8):
        for iso in ("81Br", "37Cl"):
            assert I.most_probable_heavy(n, iso) == LL.most_probable_heavy(n, iso)


def test_a_line_only_the_full_count_makes_on_either_side_of_the_reagents_range():
    """D4's full-count line (the position rule): j = k - k_c(n) outside [-k_c(s), s - k_c(s)], kept lines only
    (the caller passes those), in band only.

    Br2 [M+Br]- (n 2, the neutral 1, s 1: j in [0, 1] is the reagent's): 79Br2 (k 0, j -1) the neutral's;
    79Br81Br (k 1) and 81Br2 (k 2, j +1) the reagent's. Br3 [M+HBr+Br]- (n 3, s 2: j in [-1, 1]): 79Br3 (k 0)
    and 79Br 81Br2 (k 2) the reagent's, 81Br3 (k 3, j +2) the neutral's. Out of band: never. A Br-free
    neutral (s = n): nothing. An adduct supplying none (s 0): nothing. Cl2 [M+Cl]- (s 1, k_c(2) = k_c(1) = 0:
    j in [0, 1]): 37Cl2 (k 2) the neutral's. No satellite, an 'M+n' line (no configuration): nothing."""
    br2, br3 = {"C": 9, "H": 20, "Br": 2, "N": 3, "O": 6}, {"C": 12, "H": 10, "Br": 3, "N": 2}

    def v(counts, k, ok=True, heavy=True, iso="81Br"):
        return dict(ok=ok, counts=counts, heavy=({iso: k} if k else {}) if heavy else None)
    for mod in (I, LL):
        f = mod.full_count_line
        assert f(v(br2, 0), "81Br", 1) and not f(v(br2, 1), "81Br", 1) and not f(v(br2, 2), "81Br", 1)
        assert not f(v(br2, 0, ok=False), "81Br", 1)
        assert not f(v(br3, 0), "81Br", 1) and not f(v(br3, 2), "81Br", 1) and f(v(br3, 3), "81Br", 1)
        assert not f(v(br3, 3, ok=False), "81Br", 1)
        assert not any(f(v(br2, k), "81Br", 0) for k in range(3))              # the neutral carries no Br
        assert not any(f(v(br2, k), "81Br", 2) for k in range(3))              # the adduct supplies none
        cl2 = {"C": 6, "H": 9, "Cl": 2, "O": 3}
        assert f(v(cl2, 2, iso="37Cl"), "37Cl", 1) and not f(v(cl2, 1, iso="37Cl"), "37Cl", 1)
        assert not f(v(br2, 0), None, 1) and not f(v(br2, 0, heavy=False), "81Br", 1)
    # through line_facts: the full-count line clears the flag whatever the pair's other lines
    lines = [dict(parts=["81Br2"], elements=["Br"], ok=True, kind="set", counts=br2, heavy={"81Br": 2}),
             dict(parts=[], elements=["Br"], ok=True, kind="mono", counts=br2, heavy={})]
    assert I.line_facts(lines[:1], "81Br", 1)["reagent_only"]
    assert not I.line_facts(lines, "81Br", 1)["reagent_only"] and not I.line_facts(lines[1:], "81Br", 1)["reagent_only"]
    assert I.line_facts(lines, "81Br", 0)["reagent_only"]


# --------------------------------------------------------------------------- the width model reaches every path (c3)
RES_TOF = {"coef": 0.00010564869, "exponent": 0.99632248, "offset": 0.0, "r_at_200": 9651.6}
RES_ORBI = {"coef": 5.0134221e-07, "exponent": 1.5344223, "offset": 0.0, "r_at_200": 117528.7}


def test_the_instrument_class_is_read_off_the_width_model():
    from peaky.chem.resolution import Resolution
    assert EV.instrument(None) == (None, None) and EV.instrument({"source": "none"}) == (None, None)
    assert EV.instrument(Resolution.from_r(9650))[0] == "tof" and EV.instrument(120_000)[0] == "orbitrap"
    for res, klass in ((RES_TOF, "tof"), (RES_ORBI, "orbitrap")):
        k, fwhm = EV.instrument(res)
        k2, fwhm2 = LL.instrument(res)
        assert k == k2 == klass and fwhm(300.0) == pytest.approx(fwhm2(300.0), rel=1e-12)
    assert LL.instrument(None) == (None, None)
    # the class is read at m/z 200: an Orbitrap resolving 60 000 there (33 000 at m/z 600) is Orbitrap-class
    mid = {"coef": 200 ** (1 - 1.5344223) / 60_000.0, "exponent": 1.5344223, "offset": 0.0}
    assert EV.instrument(mid)[0] == LL.instrument(mid)[0] == "orbitrap"
    assert EV.instrument(Resolution.from_r(49_000))[0] == LL.instrument(Resolution.from_r(49_000).as_dict())[0] == "tof"
    from peaky.batch import iso_checks as IC
    assert EV.ORBITRAP_R200 == LL.ORBITRAP_R200 == IC.ORBITRAP_R200


def _run_dir(tmp_path, name="RUN_1", resolution=RES_TOF, rows=None):
    import json

    import pandas as pd
    from tests.test_evidence import ledger, m0
    run = tmp_path / name
    (run / "per_file").mkdir(parents=True)
    ledger(rows or [m0("p", "C10H16O4", ion="C10H15O4-", mz=C.ion_mz("C10H16O4", "[M-H]-"), series_unit="CH2")]) \
        .to_csv(run / "per_file" / "s1_ledger.csv", index=False)
    summary = {"reagent": "NO3"}
    if resolution is not None:
        summary["resolution"] = resolution
    json.dump(summary, open(run / "batch_summary.json", "w"))
    assert isinstance(pd, object)
    return run


def test_a_run_dirs_width_model_is_read_from_its_batch_summary(tmp_path):
    run = _run_dir(tmp_path)
    for f in (EV.source_resolution, LL.source_resolution):
        assert f(str(run)) == RES_TOF
        assert f(str(tmp_path)) == RES_TOF                        # an out-dir holding one run
        assert f(str(run / "per_file" / "s1_ledger.csv")) is None  # a ledger CSV is class-less
    bare = _run_dir(tmp_path / "x", resolution=None)
    assert EV.source_resolution(str(bare)) is None and LL.source_resolution(str(bare)) is None


def test_every_level_path_takes_the_width_model(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from peaky.assignment import assign as A
    from peaky.chem.resolution import Resolution
    from tests.test_evidence import ledger, m0
    led = ledger([m0("p", "C10H16O4", ion="C10H15O4-", mz=C.ion_mz("C10H16O4", "[M-H]-"), series_unit="CH2")])
    rp = Resolution.from_dict(RES_TOF)
    assert EV.compute_levels(led, resolution=rp).equals(EV.compute_levels(led))
    assert EV.level_pooled({"f": led}, resolution=RES_ORBI).equals(EV.level_pooled({"f": led}))
    assert EV.source_neutrals({"f": led}, resolution=rp) == EV.source_neutrals({"f": led}) == {"C10H16O4"}
    # the per-file stage hands the run's width model to the levels
    seen = {}
    real = EV.apply_levels

    def spy(ledger_, **kw):
        seen.update(kw)
        return real(ledger_, **kw)
    monkeypatch.setattr(A.evidence, "apply_levels", spy)
    inner = []
    real_lp = EV._level_pairs

    def spy_lp(frames, **kw):
        inner.append((kw.get("resolution"), kw.get("per_file")))
        return real_lp(frames, **kw)
    monkeypatch.setattr(EV, "_level_pairs", spy_lp)
    st = SimpleNamespace(led=led.copy(), cfg=None, corroborate=set(), resolving_power=rp, log=lambda *a: None)
    A._stage_evidence(st)
    assert seen["resolution"] is rp and inner == [(rp, True)]          # per file: the guard's flag set
    EV.level_pooled({"f": led}, resolution=RES_ORBI)
    assert inner[-1][0] == RES_ORBI and not inner[-1][1]
    # a --corroborate run dir is levelled with its own width model
    got = {}
    real_sn = EV.source_neutrals

    def spy_sn(per_file, **kw):
        got.setdefault("res", []).append(kw.get("resolution"))
        return real_sn(per_file, **kw)
    monkeypatch.setattr(EV, "source_neutrals", spy_sn)
    run = _run_dir(tmp_path)
    assert EV.corroborating_neutrals([str(run), str(run / "per_file" / "s1_ledger.csv")]) == {"C10H16O4"}
    assert got["res"] == [RES_TOF, None]
    # the script levels a run dir with its batch_summary's model, a CSV without one
    rec = []
    real_ms = LL.measure_source

    def spy_ms(label, frame, halogen, resolution=None, per_file=False):
        rec.append(resolution)
        return real_ms(label, frame, halogen, resolution, per_file)
    monkeypatch.setattr(LL, "measure_source", spy_ms)
    LL.run([str(run), str(run / "per_file" / "s1_ledger.csv")], [])
    assert rec == [RES_TOF, None]


def test_the_batch_and_the_scorecard_hand_over_their_width_model(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace

    from tests.test_iso_checks import _run_batch
    seen = {}
    real = EV.level_pooled

    def spy(frames, **kw):
        seen.update(kw)
        return real(frames, **kw)
    monkeypatch.setattr(EV, "level_pooled", spy)
    _run_batch(tmp_path, monkeypatch, resolving_power=100_000)
    assert seen["resolution"].r_at(200.0) == pytest.approx(100_000)
    spec = importlib.util.spec_from_file_location(
        "scorecard", Path(__file__).resolve().parents[1] / "scripts" / "scorecard.py")
    SC = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "scorecard", SC)
    spec.loader.exec_module(SC)
    got = []
    real_sc = SC.EV.level_pooled

    def spy_sc(frames, **kw):
        got.append(kw.get("resolution"))
        return real_sc(frames, **kw)
    monkeypatch.setattr(SC.EV, "level_pooled", spy_sc)
    from tests.test_evidence import ledger, m0
    pf = ledger([m0("p", "C10H16O4", ion="C10H15O4-", series_unit="CH2")]).assign(__file="s1")
    SC.own_levels_for(SimpleNamespace(path=str(tmp_path / "none"), per_file=pf, summary={"resolution": RES_ORBI}))
    SC.own_levels_for(SimpleNamespace(path=str(tmp_path / "none"), per_file=pf))
    assert got == [RES_ORBI, None]


# --------------------------------------------------------------------------- a labelled adduct's 15N (fix round 1)
def test_a_labelled_adduct_keeps_its_15n_when_the_ion_is_read_from_neutral_plus_adduct():
    """The ion's counts from neutral + adduct keep a labelled reagent atom as '^N' -- as parse_formula reads a
    signed labelled ion string -- in the engine (evidence.ion_composition) and the script (ion_counts); the
    tier gates' own reading (tiers._ion_counts by default) still folds it into N."""
    from peaky.assignment import tiers as TI
    cases = [("C10H18Cl4", "[M+^NO3]-", {"C": 10, "H": 18, "Cl": 4, "^N": 1, "O": 3}),
             ("C10H15NO2S", "[M+^NH4]+", {"C": 10, "H": 19, "N": 1, "^N": 1, "O": 2, "S": 1}),
             ("C10H15NO2S", "[M+^NH4-H2O]+", {"C": 10, "H": 17, "N": 1, "^N": 1, "O": 1, "S": 1}),
             ("C10H16O4", "[M+NO3]-", {"C": 10, "H": 16, "N": 1, "O": 7}),
             ("C9H12O2", "[M+HBr+Br]-", {"C": 9, "H": 13, "Br": 2, "O": 2}),
             ("C10H15NO2S", "[M+(CH4N2O)H]+", {"C": 11, "H": 20, "N": 3, "O": 3, "S": 1})]
    for neutral, adduct, want in cases:
        got = {k: v for k, v in EV.ion_composition(neutral, adduct, None).items() if v}
        assert got == want == LL.ion_counts(neutral, adduct, None), (neutral, adduct, got)
        assert got == {k: v for k, v in EV.ion_composition(neutral, adduct, neutral).items() if v}  # unsigned
        sign = adduct[-1]
        assert I.mono_mz(got, sign) == pytest.approx(C.ion_mz(neutral, adduct), abs=1e-9), (neutral, adduct)
    assert TI._ion_counts("C10H18Cl4", "[M+^NO3]-") == {"C": 10, "H": 18, "Cl": 4, "N": 1, "O": 3}
