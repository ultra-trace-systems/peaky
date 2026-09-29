"""C11+b rule H pins from the mutation hunt (peaky/batch/iso_checks.py rule H
and its silicon test, peaky/assignment/evidence.py the lift,
peaky/assignment/ledger.py `lead_by`, scripts/level_ledger.py `lift_leads`;
docs/EVIDENCE_LEVELS.md §3 `lead_lift`, §4.2).

Each test pins one rule a surviving mutant broke, on the synthetic series of
tests/test_halogen_lock_check.py (exact isotope ratios, the ion's own 13C line
included) or the small ledgers of tests/test_halogen_lock_lift.py. Where the
reference script re-implements a rule it is held in lockstep with the engine
(`_lockstep`: every level input the lift touches, row for row). The silicon
test (the 2026-09-28 decision, "the 29Si line decides") is pinned at each of
its edges in both regimes: the line a Si reading implies at half its area,
present and co-varying (resolved); the region's area-weighted median position
half-way toward 29Si, which alone refuses a Si-free reading, and a Si
reading's excess (the Si-reading guard) -- both the 2026-09-29 decision --,
the region's bounds, presence and co-variation with the M0 (blended), the
regime read at the ion's +1 m/z. A line at the 29Si position beside a real
chloro acid is another ion's (what the test would read); in the blended regime
it merges with the 13C line into one peak -- a full-area centroid here (the
apex-reporting picker is pinned with scenario 2, tests/test_halogen_lock_check.py).

Equivalent survivors (no input tells them apart):
- iso_checks: `supply[el] and` dropped from the reagent test (an ion the check
  reads carries n >= 1, so `n <= 0` is False exactly where the short-circuit
  was); the heavy check's three-Cl window (it lies inside the union of
  the two- and four-Cl windows); the partner search's reach (-2 .. 1) cut to
  (-1, 0) (it differs only on bit-identical m/z in one spectrum); the silicon
  test's co-variation gate letting a NaN r pass (`not r < LOCK_RMIN`): r is
  NaN only where the region's (or the M0's) area is exactly the same in every
  one of the >= 12 spectra it is present in (a refusal needs >= 60 % of >= 20)
  -- no measured peak list gives that (0 of the 346 real Si-free pairs from
  m/z 206.3 with a present +1 region on the three regression batches), and a
  fixture giving it would need lines summing to one exact area in every
  spectrum around the ion's own 13C line, which varies with the M0.
- evidence: `clean` built without a lift (it is read only for a lifted pair); the
  iso_veto filter on the lift set (a vetoed pair is alien, so outside the lift).
- ledger / plausibility: the NaN branch of `lead_setters` (str(nan) is "nan",
  which the string filter drops); `_demote_row` testing `lead is not None` (every
  caller passes None or a code).
- level_ledger: `anchor_clean`'s ion-only term and lift_leads' iso_veto test (a
  lifted pair is never ion-only, a vetoed one is alien); "H" dropped from
  ISO_CHECKS (it only orders veto notes, and rule H never vetoes).

Run: pytest tests/test_halogen_lock_hunt.py -q
"""
from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest

from peaky.assignment import evidence as EV
from peaky.assignment import ledger as L
from peaky.batch import iso_checks as IC
from peaky.chem import chemistry as C
from tests.test_evidence import ledger, m0
from tests.test_halogen_lock_check import (
    BIG, BIG_CL, BR1, CL1, MIXED, _h, _h0, _hal, _inorganic, _noise_for, _verdict)
from tests.test_halogen_lock_lift import (
    LL, NO3L, NOTE, _f, _h_row, _h_table, _lockstep, _one, _r, _run_dir, lock)
from tests.test_iso_checks import FILL, H, ORBI, _get, _ion, _measure, _row, _series, _wave

NO3 = "[M+NO3]-"
BRCL = "C2H2BrClO2"                                    # bromochloroacetic acid: a mixed-halogen ion


def _pair(i, n, a, h, *, area_x=1.0, mz=None, ion=None):
    """An M0 stamp of (n, a) at height `h` (area `area_x` x h) and its 13C line
    (area on the same basis), as test_iso_checks._c13 builds them."""
    ion = ion or _ion(n, a)
    mz = C.ion_mz(n, a) if mz is None else mz
    ic = C.parse_formula(ion.rstrip("+-"))
    h1 = h * (ic.get("C", 0) * IC.R13C + ic.get("O", 0) * IC.R17O)
    return [_row(i, mz, h, area_x * h, role="M0", nf=n, ad=a, ion=ion),
            _row(i, mz + IC.D13C, h1, area_x * h1 * ((mz + 1) / mz) ** 1.5, role="iso_child", label="13C")]


def _37cl(i, mz, h, *, a=None, off=0.0, role="iso_child", nf="", ad=""):
    """The line one 37Cl - 35Cl spacing (+ `off` Da) above an M0 at `mz`."""
    return _row(i, mz + IC.LOCK_D["Cl"] + off, h, a, role=role, nf=nf, ad=ad, label="" if role == "M0" else "37Cl")


# =========================================================================== the constants
def test_the_lock_spacings_are_the_isotope_spacings():
    """LOCK_D is the exact 37Cl - 35Cl / 81Br - 79Br spacing, the one the M+2
    isotope table carries (not HIGH's 81Br offset, 1.4 uDa away)."""
    assert IC.LOCK_D == {"Cl": 1.9970499, "Br": 1.9979521}
    assert IC.LOCK_D == {el: IC._ISO[el][1][0] for el in ("Cl", "Br")}


# =========================================================================== the stamp guard
@pytest.mark.parametrize("ppm", [0.5, -0.5])
def test_a_stamp_off_by_its_mass_error_still_locks(ppm):
    """A calibrated stamp sits a fraction of a ppm off the ion's all-light m/z:
    that is the ion, not a heavy isotopologue."""
    shift = ppm * 1e-6 * C.ion_mz(CL1, H)
    r = _verdict(lambda i: _hal(i, stamp_shift=shift))
    assert r["verdict"] == "lock" and r["mz"] == pytest.approx(C.ion_mz(CL1, H) + shift)


def test_a_stamp_far_below_the_all_light_ion_is_untestable():
    """The guard reads the stamp's distance either way: 0.6 Da BELOW the ion's
    all-light m/z is no line of the ion either."""
    r = _verdict(lambda i: _hal(i, stamp_shift=-0.6))
    assert r["verdict"] == "untestable" and "-0.600 Da" in r["note"] and not bool(r["lock"])


def test_an_adduct_with_no_ion_mz_is_untestable():
    """A propanoate cluster the chemistry table cannot price: the stamp's
    distance from the all-light ion is unknown, so no lock (the ion column still
    names its Cl)."""
    a, ion = "[M+C3H5O2]-", "C9H14ClO5-"               # C6H9ClO3 + CH3CH2COO-
    with pytest.raises(KeyError):
        C.ion_mz(CL1, a)
    mz = C.ion_mz("C9H14ClO5", "[M]-.")

    def build(i):
        h = 1e5 * _wave(i)
        return _pair(i, CL1, a, h, mz=mz, ion=ion) + [_37cl(i, mz, 0.3198 * h)]
    frames = {"f1": EV.trim(ledger([m0("p0", CL1, adduct=a, ion=ion, mz=mz)]))}
    r = _get(_measure(_series(build), [], frames=frames), "H", CL1, a)
    assert r["verdict"] == "untestable" and "no ion m/z" in r["note"] and r["element"] == "Cl"


# =========================================================================== the heavy check's windows
@pytest.mark.parametrize("lighter, why", [
    (IC.LOCK_PER_ATOM["Cl"], "one Cl: a BrCl ion's 79Br37Cl line, 0.32x its 79Br35Cl line"),
    (0.55, "two Cl only (0.42-0.93): a BrCl2 ion's 79Br35Cl37Cl line, read 14 % low"),
])
def test_the_one_and_two_chlorine_heavy_windows(lighter, why):
    """The M0 read as a Br1 ion is itself a heavy Cl line: a line one Cl spacing
    below it, which it is `lighter` x -- its own 81Br partner (0.97x) locks, the
    Cl heavy check refuses it. Each ratio sits in exactly one Cl count window."""
    assert [k for k in IC.LOCK_HEAVY_N["Cl"] if IC.lock_gates(1.0, 1.0, lighter, *IC.count_window("Cl", k))] == \
        [1 if lighter < 0.4 else 2], why

    def build(i):
        rows = _hal(i, BR1, lighter=lighter, ld=IC.LOCK_D["Cl"])
        # the 81Br35Cl line of the same ion, 0.9 mDa above the M0 (6.6 ppm: outside every 1 ppm search)
        mz = rows[0]["mz"]
        return rows + [_row(i, mz - IC.LOCK_D["Cl"] + IC.LOCK_D["Br"], rows[0]["height"] / lighter * 0.9728)]
    r = _verdict(build, BR1)
    assert r["verdict"] == "heavy" and bool(r["heavy_cl"]) and not bool(r["heavy_br"]) and not bool(r["lock"])
    assert r["ratio_area"] == pytest.approx(0.9728)                   # the line itself would lock


# =========================================================================== area and height
def test_r_reads_the_areas_not_the_heights():
    """The line's heights co-vary with the M0's exactly; its areas scatter (r
    0.5 on log area): r is the areas', so no lock."""
    h0 = _h0()
    noise = _noise_for(0.5, h0)
    mz = C.ion_mz(CL1, H)
    r = _verdict(lambda i: _pair(i, CL1, H, h0[i])
                 + [_37cl(i, mz, 0.3198 * h0[i], a=0.3198 * h0[i] * np.exp(noise[i]))])
    assert r["r"] == pytest.approx(0.5, abs=1e-6) and r["verdict"] == "no_lock"
    assert r["ratio_height"] == pytest.approx(0.3198)


def test_the_height_ratio_reads_the_heights():
    """Areas twice the heights (a wide peak): the area ratio line / M0 and the
    height ratio each on its own basis."""
    mz = C.ion_mz(CL1, H)

    def build(i):
        h = 1e5 * _wave(i)
        return _pair(i, CL1, H, h, area_x=2.0) + [_37cl(i, mz, 0.3198 * h, a=0.6396 * h)]
    r = _verdict(build)
    assert r["ratio_height"] == pytest.approx(0.3198) and r["ratio_area"] == pytest.approx(0.3198)
    assert r["verdict"] == "lock"


# =========================================================================== the silicon test (2026-09-28)
E29 = 0.3198 / IC.LOCK_SI_PER_ATOM["30Si"] * IC.LOCK_SI_PER_ATOM["29Si"]     # the 29Si a 0.32x "30Si" line implies


def _si_line(i, frac, *, d=IC.D29SI, present=lambda i: True, blend=False, n=BIG, off_ppm=0.0, **kw):
    """A C16H29ClO6 [M-H]- series (m/z 351, a 37Cl line at its count) and a
    line at `d` above the M0 of `frac` x the 29Si height a Si reading of that
    line implies (another ion's line at the 29Si position: what the silicon test
    reads); `blend`: merged with the ion's 13C line into one peak -- a full-area
    centroid, at their area-weighted position carrying both areas (the
    apex-reporting picker, one peak at the blend's apex with part of its area,
    is scenario 2's other model: tests/test_halogen_lock_check.py)."""
    rows = _hal(i, n, **kw)
    if not present(i) or not frac:
        return rows
    m0 = rows[0]
    mz, x = m0["mz"] + d + off_ppm * 1e-6 * m0["mz"], frac * E29 * m0["height"]
    if not blend:
        return rows + [_row(i, mz, x)]
    c13 = next(r for r in rows if r["iso_label"] == "13C")
    a = c13["area"] + x
    c13.update(mz=(c13["area"] * c13["mz"] + x * mz) / a, height=c13["height"] + x, area=a, iso_label="")
    return rows


def _si(build, rp, n=BIG, veto=""):
    """The H row of the series under the width model `rp`; asserts the pair's
    veto state in the same table -- `veto` names the checks that refute it ("":
    none). The blended fixtures merge another ion's line into the chloro acid's
    13C line: rule C, reading that +1 peak as the formula's 13C line, refutes
    them where it pulls the peak off the 13C position (no line within 3 ppm:
    ~0 C) or reads too many carbons -- the veto outranks any lock (a refuted
    pair never lifts), so each test reads the silicon test's own verdict and
    says whether rule C also refutes the reading (refute A5)."""
    t = _h(build, [(n, H)], resolution=rp)
    r = _get(t, "H", n, H).copy()
    v = t[t["veto"] & (t["neutral_formula"] == n) & (t["adduct"] == H)]
    r["vetoed_by"] = "|".join(sorted(v["check"]))
    assert r["vetoed_by"] == veto, (r["vetoed_by"], veto)
    return r


def test_the_silicon_constants_are_the_isotope_tables():
    """AME2020: 29Si - 28Si 0.9995683, 30Si - 28Si 1.9968436 (29.973770136 -
    27.976926535). The isotope table's 30Si entry keeps 1.9968442 (0.6 uDa high):
    re-rounding it moves HIGH rows of the regression batches (2026-09-29)."""
    assert (IC.D29SI, IC.D30SI) == (0.9995683, 1.9968436)
    assert IC._ISO["Si"][2][0] == IC.HIGH_OFFSETS["30Si"] == 1.9968442
    assert IC.LOCK_SI_PER_ATOM == pytest.approx({"29Si": 0.04685 / 0.92223, "30Si": 0.03092 / 0.92223}, rel=1e-12)
    assert IC.LOCK_SI_FRAC == 0.5


def test_the_ions_own_plus_one_line():
    """m1_line: the +1 isotopes per atom (13C, 2H, 15N, 17O, 33S, 29Si), none
    from Cl / Br (their heavy line is at +2), at their height-weighted spacing."""
    ion = {"C": 6, "H": 8, "Cl": 1, "O": 3, "N": 1, "S": 1, "Br": 1}
    parts = [(6 * 0.0107 / 0.9893, 1.0033548378), (8 * 0.000115 / 0.999885, 1.0062767),
             (1 * 0.00364 / 0.99636, 0.9970349), (3 * 0.00038 / 0.99757, 1.0042169),
             (1 * 0.0075 / 0.9499, 0.9993878)]
    h = sum(x for x, _d in parts)
    got = IC.m1_line(ion)
    assert got[0] == pytest.approx(h, rel=1e-12)
    assert got[1] == pytest.approx(sum(x * d for x, d in parts) / h, rel=1e-12)
    assert IC.m1_line({"Si": 2})[0] == pytest.approx(2 * IC.LOCK_SI_PER_ATOM["29Si"])
    assert IC.m1_line({"Cl": 2, "Br": 1}) == (0.0, IC.D13C) and IC.m1_line({}) == (0.0, IC.D13C)


@pytest.mark.parametrize("frac, verdict", [(0.51, "si_rich"), (0.49, "lock")])
def test_the_resolved_29si_line_at_half_its_expected_area(frac, verdict):
    r = _si(lambda i: _si_line(i, frac), ORBI)
    assert r["si29_mode"] == "resolved" and r["verdict"] == verdict
    assert r["si29_seen"] == pytest.approx(frac * E29) and r["si29_expected"] == pytest.approx(E29)


def test_the_resolved_29si_line_reads_its_area_not_its_height():
    """A wide 29Si line (Si lines run wide): its height 0.3x, its area 0.6x the
    expected 29Si -- the area decides."""
    def build(i):
        rows = _si_line(i, 0.6)
        rows[-1]["height"] = rows[-1]["area"] / 2
        return rows
    r = _si(build, ORBI)
    assert r["verdict"] == "si_rich" and r["si29_seen"] == pytest.approx(0.6 * E29)


def test_the_resolved_29si_line_needs_the_partners_presence_and_covariation():
    """Present in 24 of 40 stamps the line decides (resolved); in 23 it is not
    present, and the +1 region decides instead (unparted: the peak picker may
    not have parted it from 13C) -- there, a line in 23 of 40 spectra does not
    follow the M0 and refuses nothing."""
    r = _si(lambda i: _si_line(i, 1.0, present=lambda i: i < 24), ORBI)                  # 24 / 40
    assert (r["verdict"], r["si29_mode"]) == ("si_rich", "resolved")
    r = _si(lambda i: _si_line(i, 1.0, present=lambda i: i < 23), ORBI)
    assert (r["verdict"], r["si29_mode"]) == ("lock", "unparted")
    h0 = _h0()
    for r_target, verdict in ((0.805, "si_rich"), (0.795, "lock")):
        noise = _noise_for(r_target, h0)

        def build(i):
            rows = _si_line(i, 1.0)
            rows[-1]["area"] *= np.exp(noise[i])
            rows[-1]["height"] = rows[-1]["area"]
            return rows
        assert _si(build, ORBI)["verdict"] == verdict, r_target


@pytest.mark.parametrize("off, verdict, mode", [(0.9, "si_rich", "resolved"), (-0.9, "si_rich", "resolved"),
                                               (1.1, "si_rich", "unparted"), (-1.1, "lock", "unparted")])
def test_the_resolved_29si_line_is_looked_for_within_one_ppm(off, verdict, mode):
    """Within 1 ppm of 29Si the line itself decides; 1.1 ppm off it is not found
    there and the +1 region decides (unparted): 1.1 ppm above 29Si is inside the
    region (it reads the line), 1.1 ppm below is outside it."""
    r = _si(lambda i: _si_line(i, 1.0, off_ppm=off), ORBI)
    assert (r["verdict"], r["si29_mode"]) == (verdict, mode)


@pytest.mark.parametrize("off, verdict, mode", [(0.8, "si_rich", "resolved"), (-0.8, "si_rich", "resolved"),
                                               (1.2, "si_rich", "unparted"), (-1.2, "lock", "unparted")])
def test_the_29si_line_is_looked_for_at_the_literal_spacing(off, verdict, mode):
    """The line placed at the LITERAL 29Si - 28Si spacing (0.9995683, AME2020),
    not the module's constant, 0.8 / 1.2 ppm off: found within 1 ppm by the
    resolved search, or not (then the +1 region decides: above 29Si it reads the
    line, below it the line is outside) -- a wrong D29SI moves the verdict or
    the mode (the refute's mutant S06, D29SI at the 33S spacing, 0.18 mDa low).
    A line AT the 33S spacing cannot be told from 29Si by position wherever the
    test runs: 0.18 mDa is inside 1 ppm from m/z 180 on, and the test starts at
    206.3 -- the half-area gate does that (a real ion's 33S line is 0.008 per S)."""
    r = _si(lambda i: _si_line(i, 1.0, d=0.9995683, off_ppm=off), ORBI)
    assert (r["verdict"], r["si29_mode"]) == (verdict, mode)


def test_the_regime_is_read_at_the_ions_plus_one_mz():
    """A width model that parts 29Si from 13C at the M0's m/z but not at its +1
    m/z blends them: the test is where the lines are."""
    mz = C.ion_mz(BIG, H)
    rp = {"coef": (IC.D13C - IC.D29SI) / (mz + 0.5), "exponent": 1.0}
    res = IC._resolution(rp)
    assert res.fwhm(mz) < IC.D13C - IC.D29SI < res.fwhm(mz + IC.D13C)
    assert _si(lambda i: _si_line(i, 0), rp)["si29_mode"] == "blended"
    # no width model: blended
    assert _si(lambda i: _si_line(i, 0), None)["si29_mode"] == "blended"


def _own13(ion) -> float:
    """The area of the fixture's own 13C (+17O) line per M0 (_hal: its height x the
    width factor ((m+1)/m)^1.5)."""
    mz = C.ion_mz(BIG, H)
    return (ion["C"] * IC.R13C + ion["O"] * IC.R17O) * ((mz + 1) / mz) ** 1.5


def _lo():
    from tests.test_halogen_lock_check import LO_RES
    return LO_RES


SI2CL, SI1CL = "C9H21ClO6Si2", "C11H21ClO7Si"    # chloro acids carrying Si2 / Si1 (m/z 315 / 327)


def _si_reading(i, n, frac, *, d=IC.D13C, area_x=1.0, h=1e5):
    """A Cl reading `n` [M-H]- that itself carries silicon, as the peak picker
    reports it where 29Si and 13C blend (LO_RES from a +1 m/z of ~306): the M0;
    its own +1 line (`m1_line`: 13C, 2H, 17O and its own 29Si, at their
    height-weighted spacing) with `frac` x the 29Si a Si reading of its M+2
    implies from another line at `d` merged into it -- a full-area centroid;
    and its M+2 -- 37Cl, the Si count x 30Si and the 29Si2 line -- as one line
    at their area-weighted position (0.02 mDa below the 37Cl spacing). Every
    area `area_x` x its height."""
    ion = C.parse_formula(_ion(n, H).rstrip("-"))
    k = ion["Si"]
    a29, a30 = IC.LOCK_SI_PER_ATOM["29Si"], IC.LOCK_SI_PER_ATOM["30Si"]
    parts = [(IC.LOCK_D["Cl"], IC.LOCK_PER_ATOM["Cl"]), (IC.D30SI, k * a30), (2 * IC.D29SI, k * (k - 1) / 2 * a29 ** 2)]
    x2 = sum(x for _d, x in parts)
    d2 = sum(dd * x for dd, x in parts) / x2
    own, c1 = IC.m1_line(ion)
    x = frac * x2 / a30 * a29
    mz, h0 = C.ion_mz(n, H), h * _wave(i)
    return [_row(i, mz, h0, area_x * h0, role="M0", nf=n, ad=H, ion=_ion(n, H)),
            _row(i, mz + (own * c1 + x * d) / (own + x), (own + x) * h0, area_x * (own + x) * h0),
            _row(i, mz + d2, x2 * h0, area_x * x2 * h0, role="iso_child", label="37Cl")]


@pytest.mark.parametrize("n, veto", [(SI2CL, ""), (SI1CL, "C")])
@pytest.mark.parametrize("frac, verdict", [(0.51, "si_rich"), (0.49, "lock")])
def test_a_si_reading_is_also_refused_on_half_the_29si_excess(n, veto, frac, verdict):
    """The Si-reading guard (the 2026-09-29 decision): a Cl reading carrying
    1-2 Si has its own 29Si in its +1 line, which pulls the half-way mark toward
    29Si; its region is refused on the position OR on reading >= half the 29Si
    a Si reading of its M+2 implies above its own +1 line. The excess here sits
    at the 13C position (another ion's line there moves the region away from
    29Si): the excess alone decides, at its edge, for a Si2 and a Si1 reading.
    (Rule C cannot read SI2CL, whose +1 line is 48 % 13C, and refutes SI1CL,
    whose +1 peak sits 0.4 mDa off 13C reading ~40 C.)"""
    r = _si(lambda i: _si_reading(i, n, frac), _lo(), n=n, veto=veto)
    assert (r["verdict"], r["si29_mode"]) == (verdict, "blended")
    assert r["si29_seen"] == pytest.approx(frac * r["si29_expected"], abs=1e-9)


def test_a_si_free_reading_is_refused_on_the_position_alone():
    """0.3x the 29Si at its position, merged with the ion's 13C line (a
    full-area centroid): the region reads only 0.3x the 29Si above the ion's own
    +1 line -- less than half -- but sits at least half-way toward 29Si, and a
    Si-free reading is refused on the position alone (the 2026-09-29 decision:
    the excess test, as built before, locked the dim and apex-reported
    siloxane misreads)."""
    r = _si(lambda i: _si_line(i, 0.3, blend=True), _lo(), veto="C")
    assert r["verdict"] == "si_rich" and 0 < r["si29_seen"] < 0.5 * E29
    ion = C.parse_formula(_ion(BIG, H).rstrip("-"))
    assert r["si29_seen"] == pytest.approx(_own13(ion) + 0.3 * E29 - IC.m1_line(ion)[0])


def test_the_blended_excess_at_the_13c_position_is_no_silicon():
    """A +1 line 0.6x the 29Si too tall but sitting where 13C sits (more carbon,
    or another ion there): a Si-free reading has no excess criterion -- no
    refusal (excess OR position would refuse it)."""
    r = _si(lambda i: _si_line(i, 0.6, d=IC.D13C, blend=True), _lo(), veto="C")      # rule C: ~43 C for 16
    assert r["verdict"] == "lock" and r["si29_seen"] > 0.5 * E29


def test_a_dim_si_free_ion_with_a_neighbour_at_13c_is_not_refused():
    """The negative control's shape as a fixture: the chloro acid dim (12 x
    the noise edge x the wave: its own 13C line above the edge in every
    spectrum, too dim for rule C, so nothing else stands between the lock and
    its lift). Its own 13C line alone is the +1 region -- co-varying, at its
    own position: no refusal. Another ion's line at the 13C spacing merged into
    it, co-varying, 1.0x the 29Si a Si reading implies: the region reads twice
    the half 29Si above the ion's own +1 line but does not move toward 29Si --
    no refusal either (excess OR position would refuse it)."""
    for frac in (0.0, 1.0):
        t = _h(lambda i: _si_line(i, frac, d=IC.D13C, blend=True, h=12 * FILL), [(BIG, H)], resolution=_lo())
        r = _get(t, "H", BIG, H)
        assert (r["verdict"], r["si29_mode"]) == ("lock", "blended") and IC.veto(t) == {}, frac
        assert _get(t, "C", BIG, H)["verdict"] == "untestable" and r["presence"] == 1.0
        assert bool(r["si29_seen"] > 0.5 * E29) == (frac > 0)


def test_the_blended_line_must_shift_half_way_toward_29si():
    """The same excess placed so the blend's median position lands just past /
    just short of half the shift a Si reading implies (LOCK_SI_FRAC)."""
    ion = C.parse_formula(_ion(BIG, H).rstrip("-"))
    own, c1 = IC.m1_line(ion)
    blend = (own * c1 + E29 * IC.D29SI) / (own + E29)
    at_max = c1 - 0.5 * (c1 - blend)
    ion13 = _own13(ion)
    x = 0.8 * E29
    for eps, verdict in ((-2e-5, "si_rich"), (2e-5, "lock")):
        # the extra line's spacing that puts the area-weighted +1 position at at_max + eps
        d = ((at_max + eps) * (ion13 + x) - ion13 * IC.D13C) / x
        assert _si(lambda i: _si_line(i, 0.8, d=d, blend=True), _lo(), veto="C")["verdict"] == verdict, eps


def test_the_blended_region_is_29si_to_13c():
    """The region runs from 1 ppm below 29Si to 1 ppm above 13C: a line at the
    15N spacing (2.5 mDa below 29Si) is not in it; one 0.9 ppm below 29Si is."""
    lo = _lo()
    assert _si(lambda i: _si_line(i, 1.0, d=0.9970349), lo)["verdict"] == "lock"
    assert _si(lambda i: _si_line(i, 1.0, off_ppm=-0.9), lo)["verdict"] == "si_rich"
    assert _si(lambda i: _si_line(i, 1.0, off_ppm=-1.2), lo)["verdict"] == "lock"


def test_the_blended_excess_is_over_the_m0s_area():
    """Wide peaks (every area 2x its height) on a Si reading, whose excess
    counts: the region's excess reads per M0 area -- 0.3x the 29Si, no refusal
    -- not per M0 height (which would read the +1 line twice over and refuse)."""
    r = _si(lambda i: _si_reading(i, SI2CL, 0.3, area_x=2.0), _lo(), n=SI2CL)
    assert r["verdict"] == "lock" and r["si29_seen"] == pytest.approx(0.3 * r["si29_expected"], abs=1e-9)


def test_the_blended_position_is_the_median_over_the_spectra():
    """An extra line (2x the 29Si's area) in every spectrum, co-varying with the
    M0: in 21 of 40 spectra it sits where it puts the +1 region 0.2 mDa inside
    the half-way mark toward 29Si, in the other 19 at 13C (two sources
    alternating). The median spectrum is on the Si side and the lock is
    refused; the mean of all would sit past the mark."""
    ion = C.parse_formula(_ion(BIG, H).rstrip("-"))
    own, c1 = IC.m1_line(ion)
    at_max = c1 - 0.5 * (c1 - (own * c1 + E29 * IC.D29SI) / (own + E29))
    a13, x = _own13(ion), 2.0 * E29
    target = at_max - 2e-4
    d = (target * (a13 + x) - a13 * IC.D13C) / x              # the extra line's spacing that puts the region there
    assert (21 * target + 19 * IC.D13C) / 40 > at_max          # the mean would not refuse
    r = _si(lambda i: _si_line(i, 2.0, d=d if i < 21 else IC.D13C, blend=True), _lo())
    assert r["verdict"] == "si_rich"                          # (rule C: area 43 C, height at 13C, not both)
    # 20 of 40 on the Si side: the median sits between the two groups, past the mark
    assert _si(lambda i: _si_line(i, 2.0, d=d if i < 20 else IC.D13C, blend=True), _lo(),
               veto="C")["verdict"] == "lock"


def _region(rows):
    """The +1 region's peak of a blended _si_line series (the 13C line merged with the extra one)."""
    return next(r for r in rows if 0.99 < r["mz"] - rows[0]["mz"] < 1.01)


def test_the_blended_region_must_co_vary_with_the_m0():
    """The 2026-09-29 decision (U1): the +1 region co-varies with the M0 --
    r(log region area, log M0 area) >= LOCK_RMIN over the spectra it is present
    in -- as the lock partner and the resolved 29Si line must: a Si-rich ion's
    29Si line is its own. The same region scattered to r 0.795 is not the
    ion's: no refusal."""
    h0 = _h0()
    for r_target, verdict in ((0.805, "si_rich"), (0.795, "lock")):
        noise = _noise_for(r_target, h0)

        def build(i):
            rows = _si_line(i, 1.0, blend=True)
            reg = _region(rows)
            reg.update(area=reg["area"] * np.exp(noise[i]), height=reg["height"] * np.exp(noise[i]))
            return rows
        r = _si(build, _lo(), veto="C")
        assert r["si29_mode"] == "blended" and r["verdict"] == verdict, r_target
        assert r["si29_seen"] > 0.5 * E29                    # the excess alone would refuse


def test_an_anti_correlated_region_does_not_co_vary():
    """R2A-3: the gate reads r itself, not its size. Another ion's line at the
    29Si position that FALLS as the chloro acid rises (its precursor, say),
    merged with the 13C line into one peak (a full-area centroid): the region
    sits past the half-way mark toward 29Si in every spectrum (the position
    alone refuses a co-varying region) and reads far more than half the 29Si
    above the ion's own +1 line, but at r -0.95 it is not the ion's own line --
    no refusal (|r| would refuse it)."""
    ion = C.parse_formula(_ion(BIG, H).rstrip("-"))
    own, c1 = IC.m1_line(ion)
    at_max = c1 - 0.5 * (c1 - (own * c1 + E29 * IC.D29SI) / (own + E29))

    def build(i):
        rows = _hal(i, BIG)
        c13 = next(r for r in rows if r["iso_label"] == "13C")
        x = E29 * 1.5e5 / _wave(i)                     # 0.89 x the 29Si pooled, falling as the M0 rises
        a = c13["area"] + x
        c13.update(mz=(c13["area"] * c13["mz"] + x * (rows[0]["mz"] + IC.D29SI)) / a, height=c13["height"] + x,
                   area=a, iso_label="")
        return rows
    series = [build(i) for i in range(40)]
    pa = np.array([s[0]["area"] for s in series])
    reg = [_region(s) for s in series]
    ra = np.array([g["area"] for g in reg])
    assert np.corrcoef(np.log(pa), np.log(ra))[0, 1] == pytest.approx(-0.95, abs=0.005)
    assert max(g["mz"] - s[0]["mz"] for g, s in zip(reg, series)) < at_max          # past the mark everywhere
    assert ra.sum() / pa.sum() - own > 0.5 * E29                                     # and the excess
    r = _si(build, _lo(), veto="C")
    assert (r["verdict"], r["si29_mode"]) == ("lock", "blended") and r["si29_seen"] > 0.5 * E29


def test_the_blended_region_co_varies_by_area():
    """The M0's heights scatter against its areas (r 0.5 in log: a peak whose
    width changes spectrum to spectrum); the +1 region's area follows the M0's
    area exactly. r reads the areas, as the lock partner's does: the region is
    the ion's and the lock is refused."""
    h0 = _h0()
    noise = _noise_for(0.5, h0)

    def build(i):
        rows = _si_line(i, 1.0, blend=True)
        rows[0]["height"] = rows[0]["area"] * np.exp(noise[i])
        return rows
    r = _si(build, _lo(), veto="C")
    assert r["verdict"] == "si_rich" and r["r"] == pytest.approx(1.0)   # the partner's r reads areas too


def test_the_blended_region_reads_its_peaks_area_weighted():
    """The peak picker split the region: a small line at the 29Si position
    (0.6x the 29Si) beside the ion's 13C line and a larger excess at 13C (more
    carbon, or another ion). Area-weighted the region sits on the 13C side: no
    Si blend, though its first peak is the 29Si-position one."""
    def build(i):
        rows = _si_line(i, 0.6)
        c13 = next(r for r in rows if r["iso_label"] == "13C")
        c13.update(height=c13["height"] + 0.5 * rows[0]["height"], area=c13["area"] + 0.5 * rows[0]["height"])
        return rows
    r = _si(build, _lo(), veto="C")                           # rule C: ~62 C for 16
    assert r["verdict"] == "lock" and r["si29_seen"] > 0.5 * E29


def test_the_blended_region_needs_the_partners_presence():
    """The +1 region (the ion's 13C line and the extra one) below the floor in 17
    of 40 spectra, while the taller 37Cl line is in all: 23 / 40 < 60 %."""
    def build(n_on):
        def b(i):
            rows = _si_line(i, 1.0, blend=True)
            return rows if i < n_on else [r for r in rows if not (0.99 < r["mz"] - rows[0]["mz"] < 1.01)]
        return b
    assert _si(build(24), _lo(), veto="C")["verdict"] == "si_rich"
    assert _si(build(23), _lo(), veto="C")["verdict"] == "lock"


def test_the_silicon_test_runs_only_on_a_line_that_passes_the_gates():
    r = _si(lambda i: _si_line(i, 1.0, ratio=0.1), ORBI)
    assert r["verdict"] == "no_lock" and r["si29_mode"] == "" and np.isnan(r["si_n"])


def test_the_note_names_a_missing_29si_line():
    r = _si(lambda i: _si_line(i, 0), ORBI)
    assert r["verdict"] == "lock" and r["note"].endswith("no 29Si line of a Si9.5 reading of the line (-0.00x of "
                                                         "0.48x, unparted)")
    r = _si(lambda i: _si_line(i, 0.2), ORBI)                   # present, too small: the line decides
    assert r["verdict"] == "lock" and r["note"].endswith("(0.10x of 0.48x, resolved)")


def test_the_silicon_columns_are_rule_hs_own():
    t = _h(lambda i: _si_line(i, 1.0), [(BIG, H)], resolution=ORBI)
    assert {"si_n", "si29_expected", "si29_seen", "si29_mode"} <= set(IC.TABLE_COLUMNS)
    other = t[t["check"] != "H"]
    assert len(other) and other["si_n"].isna().all() and other["si29_mode"].isna().all()


def test_the_recorded_offset_is_the_median():
    """19 of 40 lines 0.23 mDa low (inside 1 ppm at m/z 235), 21 on the spacing:
    `offset_mda` records the median (0), not the mean (-0.109)."""
    mz = C.ion_mz(BIG_CL, H)

    def build(i):
        h = 1e5 * _wave(i)
        return _pair(i, BIG_CL, H, h) + [_37cl(i, mz, 0.3198 * h, off=-0.23e-3 if i < 19 else 0.0)]
    r = _verdict(build, BIG_CL)
    assert r["offset_mda"] == pytest.approx(0.0, abs=1e-6) and r["verdict"] == "lock"


# =========================================================================== the recorded columns
def test_a_partner_committed_as_another_pairs_m0_is_recorded_not_gated():
    """The 37Cl line committed as an isobaric CHNOSiF formula (0.45 ppm off it):
    `other_m0` records the share of such lines; it gates nothing (D5)."""
    other = "C3H7N2O3FSi"
    mz = C.ion_mz(CL1, H)
    assert abs(C.ion_mz(other, H) - (mz + IC.LOCK_D["Cl"])) / mz * 1e6 < 1.0

    def build(i):
        h = 1e5 * _wave(i)
        return _pair(i, CL1, H, h) + [_37cl(i, mz, 0.3198 * h, role="M0", nf=other, ad=H)]
    r = _verdict(build)
    assert r["other_m0"] == 1.0 and r["verdict"] == "lock"
    assert _verdict(lambda i: _hal(i))["other_m0"] == 0.0


def test_a_mixed_ion_records_its_chlorine():
    """Bromochloroacetic acid with both of its M+2 lines (81Br35Cl 0.97x, 79Br37Cl
    0.32x, 0.9 mDa apart): untestable, and the row names the ion's Cl -- the
    first of LOCK_D's elements -- and its count."""
    mz = C.ion_mz(BRCL, H)

    def build(i):
        h = 1e5 * _wave(i)
        return _pair(i, BRCL, H, h) + [_37cl(i, mz, 0.3198 * h),
                                       _row(i, mz + IC.LOCK_D["Br"], 0.9728 * h, role="iso_child", label="81Br")]
    r = _verdict(build, BRCL)
    assert r["verdict"] == "untestable" and "both Cl and Br" in r["note"]
    assert (r["element"], int(r["n_halogen"]), r["offset"]) == ("Cl", 1, "37Cl")


def test_the_no_lock_note_names_the_gate_that_failed():
    """A line at a third of the one-Cl height, present in exactly 60 % of the
    stamps: the note names the window, not the presence (60 % passes)."""
    r = _verdict(lambda i: _hal(i, ratio=0.1, present=lambda i: i < 24))
    assert r["verdict"] == "no_lock" and r["presence"] == 0.6
    assert r["note"].endswith("; outside the 1 Cl window") and "present in <" not in r["note"]


def test_a_line_in_two_spectra_records_no_correlation():
    """Two points make no correlation: r is NaN (a line seen in 2 of 40 spectra)."""
    assert np.isnan(IC._pearson(np.array([1.0, 2.0]), np.array([3.0, 5.0])))
    r = _verdict(lambda i: _hal(i, present=lambda i: i < 2))
    assert r["n_used"] == 2 and np.isnan(r["r"]) and r["verdict"] == "no_lock"


def test_a_flat_m0_gives_no_correlation():
    """Degenerate input (the same area in every spectrum): no variance, no r, no lock."""
    mz = C.ion_mz(CL1, H)
    r = _verdict(lambda i: _pair(i, CL1, H, 1e5) + [_37cl(i, mz, 0.3198e5)])
    assert np.isnan(r["r"]) and r["verdict"] == "no_lock"


def test_the_ions_halogen_count_is_an_integer_column(tmp_path):
    t = _h(lambda i: _hal(i), [(CL1, H)])
    assert str(t["n_halogen"].dtype) == "Int64"
    p = tmp_path / "iso_checks.csv"
    t.to_csv(p, index=False)
    back = pd.read_csv(p, dtype=str, keep_default_na=False)
    assert list(back.loc[back["check"] == "H", "n_halogen"]) == ["1"]


# =========================================================================== the reagent (D2)
def test_a_reagent_owned_ion_without_a_passing_line_reads_reagent():
    """The verdict order: the reagent's ownership is decided before the gates.
    BrHNO3- on the bromide + nitrate batch with a noisy 81Br line (r 0.5): the
    reagent could have put the Br there, so `reagent`, not `no_lock`."""
    r = _inorganic("HBr", NO3, MIXED, noise=_noise_for(0.5, _h0()))
    assert r["verdict"] == "reagent" and r["r"] == pytest.approx(0.5, abs=1e-6)


# =========================================================================== evidence: the lift
def test_an_ion_only_sibling_gives_the_lifted_pair_no_channel(tmp_path):
    """The ion-only stage's [M]-. row is the parent's composition on another
    adduct, no second channel -- lifted or not; the reference script alike."""
    io_row = _r("io", adduct="[M]-.", mz=C.ion_mz(CL1, H) + 1.00728)
    io_row["method"] = "ion_only:ea"
    m = _lockstep(tmp_path, "R", _f(_r("p", lead=True), io_row), _h_table([_h_row(CL1, H)]))
    r = m.loc[(CL1, H)]
    assert bool(r["lead_lift"]) and not bool(r["chan2"]) and r["evidence_level"] == "4b"


def test_a_lock_fact_without_budget_ok_is_conservative():
    """A lock fact handed in without `budget_ok` lifts no element-budget lead
    (and no lead of unknown setter); a speculative one it still lifts."""
    bare = {"veto": {}, "lock": {(CL1, H): {"element": "Cl", "n": 1, "note": NOTE}}}
    for code, lifts in (("off_budget", False), ("", False), ("spec_gapfill", True)):
        r = _one({"f1": _f(_r("p", lead=True, lead_by=code))}, iso=bare)
        assert bool(r["lead_lift"]) is lifts, code


# =========================================================================== ledger: lead_by
@pytest.mark.parametrize("value", ["nan", "NaN", pd.NA, float("nan"), np.float64("nan"), None, "", " "])
def test_nan_like_cells_name_no_setter(value):
    assert L.lead_setters(value) == frozenset()
    assert LL.lead_setters(value) == set()


def test_the_script_reads_the_setters_like_the_ledger():
    for v in ("spec_gapfill", "off_budget|spec_minor", "nan|spec_n3", "spec_n3|", "radical_anion"):
        assert LL.lead_setters(v) == set(L.lead_setters(v)), v


def test_the_converted_column_reads_empty_on_the_other_rows():
    """A `lead_by` column read back all-empty from a CSV is float NaN; the first
    code written converts it to text, "" on every other row (not NaN)."""
    led = pd.DataFrame({"peak_id": ["p", "q", "r"], "role": ["M0"] * 3, "below_assignability": [False] * 3,
                        "tentative_lead": [False] * 3, "lead_by": [np.nan] * 3})
    back = pd.read_csv(io.StringIO(led.to_csv(index=False)))
    assert back["lead_by"].dtype.kind == "f"
    L.mark_lead(back, 1, "spec_gapfill")
    assert list(back["lead_by"]) == ["", "spec_gapfill", ""]


# =========================================================================== the reference script
def test_the_scripts_setter_list_is_the_engines():
    assert LL.LIFTABLE_LEADS == set(EV.LIFTABLE_LEADS)
    assert set(EV.LIFTABLE_LEADS) < set(L.LEAD_SETTERS)


@pytest.mark.parametrize("code", L.LEAD_SETTERS)
@pytest.mark.parametrize("budget_ok", [True, False])
def test_each_setter_lifts_in_lockstep(tmp_path, code, budget_ok):
    m = _lockstep(tmp_path, "R", _f(_r("p", lead=True, lead_by=code)), _h_table([_h_row(CL1, H, budget_ok=budget_ok)]))
    assert bool(m.loc[(CL1, H), "lead_lift"]) is EV.lead_liftable({code}, budget_ok)


def test_a_lead_only_sibling_gives_no_channel_in_the_script(tmp_path):
    m = _lockstep(tmp_path, "R", _f(_r("p", lead=True), _r("s", adduct=NO3, lead=True)), _h_table([_h_row(CL1, H)]))
    assert bool(m.loc[(CL1, H), "lead_lift"]) and not bool(m.loc[(CL1, H), "chan2"])


def test_a_stray_code_on_an_unflagged_row_blocks_nothing_in_the_script(tmp_path):
    m = _lockstep(tmp_path, "R", _f(_r("p", lead=True), _r("q", lead_by="radical_anion")), _h_table([_h_row(CL1, H)]))
    assert bool(m.loc[(CL1, H), "lead_lift"])


def test_one_uncoded_lead_row_needs_the_budget_in_the_script(tmp_path):
    frame = _f(_r("p", lead=True, lead_by="spec_gapfill"), _r("q", lead=True, lead_by=""))
    m = _lockstep(tmp_path, "R", frame, _h_table([_h_row(CL1, H, budget_ok=False)]))
    assert not bool(m.loc[(CL1, H), "lead_lift"])
    m = _lockstep(tmp_path, "R2", frame, _h_table([_h_row(CL1, H, budget_ok=True)]))
    assert bool(m.loc[(CL1, H), "lead_lift"])


def test_every_lead_rows_code_counts_in_the_script(tmp_path):
    frame = _f(_r("p", lead=True, lead_by="spec_gapfill"), _r("q", lead=True, lead_by="radical_anion"))
    m = _lockstep(tmp_path, "R", frame, _h_table([_h_row(CL1, H)]))
    assert not bool(m.loc[(CL1, H), "lead_lift"])


def test_the_script_carries_the_lock_note(tmp_path):
    run = _run_dir(tmp_path, "R", _f(_r("p", lead=True)), _h_table([_h_row(CL1, H)]))
    ref = LL.run([str(run)], [], None, None, "auto").set_index(["neutral", "adduct"])
    core = _one({"s1": _f(_r("p", lead=True))}, iso=lock((CL1, H)))
    assert ref.at[(CL1, H), "lock_note"] == core["lock_note"] == NOTE


def test_without_a_label_table_the_script_does_not_fold(tmp_path):
    """With no rule K table the 14N and 15N nitrate clusters are two channels
    for a lifted [M+^NO3]- (the fold is rule K's)."""
    m = _lockstep(tmp_path, "R", _f(_r("p", adduct=NO3L, lead=True), _r("s", adduct=NO3)),
                  _h_table([_h_row(CL1, NO3L)]))
    assert bool(m.loc[(CL1, NO3L), "lead_lift"]) and bool(m.loc[(CL1, NO3L), "chan2"])


def test_the_script_reads_a_table_written_before_rule_h(tmp_path):
    t = _h_table([dict(neutral_formula="C7H12O4", adduct=H, check="REQ", verdict="absent", veto=True, note="r")])
    old = t.drop(columns=["lock", "element", "n_halogen", "ratio_lo", "ratio_hi", "heavy_cl", "heavy_br",
                          "budget_ok", "budget_why"])
    p = tmp_path / "iso_checks.csv"
    old.to_csv(p, index=False)
    assert LL.iso_check_facts(str(p)) == IC.facts(old) == {"veto": {("C7H12O4", H): "REQ: r"}, "lock": {}}


def test_a_measured_frame_without_lead_unknown_is_conservative(tmp_path):
    """assign_levels on a measured frame that carries no `lead_unknown` (not
    measure_source's) takes every lead as of unknown setter: it lifts only where
    `budget_ok`, like the engine on a ledger without `lead_by`."""
    frame = _f(_r("p", lead=True))
    run = _run_dir(tmp_path, "R", frame, None)
    label, led = LL.load_source(str(run))
    measured = LL.measure_source(label, led, None).drop(columns=["lead_unknown"])
    for ok in (False, True):
        out = LL.assign_levels(measured, set(), iso=IC.facts(_h_table([_h_row(CL1, H, budget_ok=ok)])))
        assert bool(out.set_index(["neutral", "adduct"]).at[(CL1, H), "lead_lift"]) is ok
