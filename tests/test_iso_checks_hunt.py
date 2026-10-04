"""C11+a isotope-check pins from the mutation hunt (peaky/batch/iso_checks.py,
peaky/assignment/evidence.py, scripts/level_ledger.py, peaky/batch/assign_batch.py;
docs/EVIDENCE_LEVELS.md §3 `iso_veto`, §4.2).

Each test pins one rule a surviving mutant broke, on small synthetic series with
exact isotope ratios (the builders of tests/test_iso_checks.py); where a rule is
a boundary, the series sits ON it (float-exact) so `>=` and `>` part.

Run: pytest tests/test_iso_checks_hunt.py -q
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from peaky.assignment import evidence as EV
from peaky.batch import iso_checks as IC
from peaky.chem import chemistry as C
from peaky.chem.resolution import Resolution
from tests import test_iso_checks as TI
from tests.test_evidence import child, ledger, m0
from tests.test_iso_checks import (D81, FILL, H, N, NO3, ORBI, REFS, SCALE_O, SCALE_T, TOF, Y, Z, _br, _c13, _filler,
                                   _get, _has, _hi, _ion, _iso_table, _ll, _measure, _refs, _row, _series, _wave, quiet)

X = "C12H20O4"
XS = "C12H20O4S"          # a heteroatom pair: never in rule C's bias population


def _rows(build, *, filler=_filler, low=True) -> pd.DataFrame:
    rows = []
    for i in range(N):
        rows += filler(i, low=low) if filler is _filler else filler(i)
        rows += build(i)
    return pd.DataFrame(rows)


def _m0row(i, n, a, h, *, mz=None, area=None, role="M0"):
    mz = C.ion_mz(n, a) if mz is None else mz
    return _row(i, mz, h, area, role=role, nf=n if role else "", ad=a if role else "", ion=_ion(n, a) if role else "")


# =========================================================================== the batch
def test_an_infinite_resolving_power_is_tof_class():
    """A degenerate width model (FWHM underflows to R = inf at m/z 200) is no
    Orbitrap: the class needs a finite R >= 50 000."""
    rp = Resolution(coef=1e-320, exponent=1.0)
    assert not np.isfinite(rp.r_at(200.0))
    assert IC.instrument_class(rp) == "tof"
    assert IC.instrument_class({"coef": 1e-320, "exponent": 1.0}) == "tof"


def test_the_mass_scale_defaults():
    """No mass scale: the stamp window is 6 ppm and sigma unknown (the Orbitrap
    window stays 1 ppm); a scale without stamp_ppm stamps at its tol_ppm."""
    ts = _series(lambda i: _br(i, present=lambda i: False))
    assert _get(_measure(ts, [(Y, H)], resolution=TOF, scale=None), "REQ", Y)["window_ppm"] == 6.0
    assert _get(_measure(ts, [(Y, H)], resolution=TOF, scale={"sigma_ppm": 3.6, "tol_ppm": 12.0}),
                "REQ", Y)["window_ppm"] == 12.0
    assert _get(_measure(ts, [(Y, H)], resolution=TOF, scale={"sigma_ppm": 3.6, "stamp_ppm": float("nan"),
                                                               "tol_ppm": 11.0}), "REQ", Y)["window_ppm"] == 11.0
    assert _get(_measure(ts, [(Y, H)], scale={"stamp_ppm": 6.0}), "REQ", Y)["window_ppm"] == 1.0
    # ... and a 1.5 ppm line is absent there (a sigma read as 0.5 ppm would widen it to 2)
    off = _series(lambda i: _br(i, ppm=1.5))
    assert _get(_measure(off, [(Y, H)], scale={"stamp_ppm": 6.0}), "REQ", Y)["verdict"] == "absent"


def test_the_noise_edge_is_the_first_percentile_of_the_positive_heights():
    """The spectrum's floor is the 1st-percentile height of its peaks with a
    positive height: zero-height rows do not pull it to zero, and a graded
    noise floor reads its 1st percentile, not its 5th."""
    zeros = lambda i: [_row(i, 800.0 + 0.37 * j, 0.0, 1.0) for j in range(40)]      # noqa: E731
    ts = _series(lambda i: _br(i, h=12.0 / _wave(i), present=lambda i: False) + zeros(i))
    r = _get(_measure(ts, [(Y, H)]), "REQ", Y)
    assert r["verdict"] == "untestable" and r["n_used"] == 0

    def graded(i):
        # 60 noise peaks at 10, 11, ... 69 cps and one at the scan start: 1st percentile 10, 5th ~12
        return [_row(i, 700.0 + 0.37 * j, 10.0 + j) for j in range(60)] + [_row(i, 60.0, 10.0)]
    # the 81Br line would be 33 cps: detectable over 3 x 10, not over 3 x 12
    ts = _rows(lambda i: _br(i, h=33.0 / 0.9728 / _wave(i), present=lambda i: False), filler=graded)
    r = _get(_measure(ts, [(Y, H)]), "REQ", Y)
    assert r["verdict"] == "absent" and r["n_used"] == N


def test_the_brightest_m0_stamp_of_a_spectrum_is_read():
    """Two M0 stamps of one pair in a spectrum: rule C reads the brighter one."""
    pairs = [(n, H) for n in REFS] + [(X, H)]
    mz = C.ion_mz(X, H)
    ts = _series(lambda i: _refs(i) + _c13(i, X, H) + [_m0row(i, X, H, 3e4, mz=mz + 0.5)])
    r = _get(_measure(ts, pairs), "C", X)
    assert r["verdict"] == "agree" and r["mz"] == pytest.approx(mz) and r["n_used"] == N


def test_the_13c_line_is_found_below_its_exact_position():
    """The M+1 search looks both ways: a 13C line 1 ppm BELOW m0 + 1.0033548 is read."""
    pairs = [(n, H) for n in REFS] + [(X, H)]
    mz = C.ion_mz(X, H)

    def build(i):
        rows = _c13(i, X, H)
        rows[1]["mz"] = (mz + IC.D13C) * (1 - 1e-6)
        return _refs(i) + rows
    r = _get(_measure(_series(build), pairs), "C", X)
    assert r["verdict"] == "agree" and r["c_area"] == pytest.approx(12.0, abs=1e-6)


def test_a_13c_line_is_read_in_its_own_spectrum_only():
    """Rule C's M+1 is the nearest peak IN THE SAME SPECTRUM: a line at m0 +
    1.0033548 that opens the NEXT spectrum is not this spectrum's 13C line."""
    mz = C.ion_mz(XS, H)
    ion = _ion(XS, H)
    h0 = 1e5
    h1 = h0 * (12 * IC.R13C + C.parse_formula(ion)["O"] * IC.R17O)

    def spectrum(i):
        if i % 2 == 0:       # the M0 and noise below it, nothing above
            return ([_row(i, 60.0 + 0.37 * j, FILL) for j in range(60)]
                    + [_row(i, mz, h0, role="M0", nf=XS, ad=H, ion=ion)])
        # the next spectrum: its lowest peak sits exactly where the 13C line would be
        return [_row(i, mz + IC.D13C, h1, h1 * ((mz + 1) / mz) ** 1.5)] + [_row(i, 700.0 + 0.37 * j, FILL)
                                                                           for j in range(60)]
    ts = pd.DataFrame(sum((spectrum(i) for i in range(N)), []))
    r = _get(_measure(ts, [(XS, H)]), "C", XS)
    assert r["n_used"] == N // 2 and r["verdict"] == "contradict" and r["c_area"] < 0


def test_the_nearest_window_is_inclusive():
    """_Series.nearest keeps a peak exactly `ppm` away (a float-exact case)."""
    ts = pd.DataFrame(dict(sample_item_id=["a", "a"], mz=[100.0, 512.0 + 2.0 ** -9], height=[1.0, 1.0]))
    S = IC._Series(ts)
    ppm = 2.0 ** -18 * 1e6              # exactly 2**-9 / 512 * 1e6
    assert S.nearest(np.array([0]), np.array([512.0]), ppm=ppm)[0] == 1
    assert S.nearest(np.array([0]), np.array([512.0]), ppm=ppm * (1 - 1e-9))[0] == -1


# =========================================================================== pooled pairs
def test_a_pair_ion_only_in_any_file_is_ion_only_when_pooled():
    """A pair the ion-only stage wrote in ONE file is ion-only pooled: rule C skips it."""
    x = "C10H16O5"
    a = "[M]-."
    mz = C.ion_mz(x, H) + 1.00728
    frames = {"f1": EV.trim(ledger([m0("io", x, adduct=a, ion="C10H16O5-", mz=mz, method="ion_only:ea")])),
              "f2": EV.trim(ledger([m0("io", x, adduct=a, ion="C10H16O5-", mz=mz, method="pass2")]))}
    assert IC._pooled(frames)["ion_only"].tolist() == [True]
    ts = _series(lambda i: _refs(i) + [_row(i, mz, 1e5, role="M0", nf=x, ad=a, ion="C10H16O5-"),
                                       _row(i, mz + IC.D13C, 1e5 * 3 * IC.R13C, role="iso_child", label="13C")])
    t = IC.measure(ts, frames, NO3, resolution=ORBI, mass_scale=SCALE_O, log=quiet)
    assert not _has(t, "C", x, a)


def test_an_unstamped_pair_is_read_at_its_median_pooled_mass():
    """The pooled m/z is the MEDIAN over the files (evidence._measure's):
    an outlying file does not move where an unstamped pair is read."""
    mz = C.ion_mz(Y, H)
    frames = {f"f{k}": EV.trim(ledger([m0("p", Y, adduct=H, ion=_ion(Y, H), mz=v)]))
              for k, v in enumerate([mz, mz, mz * (1 + 10e-6)])}
    ts = _series(lambda i: _br(i, stamped=False, present=lambda i: False))
    r = _get(_measure(ts, [], frames=frames), "REQ", Y)
    assert not r["stamped"] and r["verdict"] == "absent" and r["n_spectra"] == N


# =========================================================================== rule C
def test_rule_c_carbon_free_ions_are_not_tested():
    n, a = "HNO3", "[M+NO3]-"
    pairs = [(x, H) for x in REFS] + [(n, a)]
    t = _measure(_series(lambda i: _refs(i) + [_m0row(i, n, a, 1e5 * _wave(i))]), pairs)
    assert not _has(t, "C", n, a) and _has(t, "C", REFS[0])


def test_rule_c_13c_window_is_three_ppm():
    pairs = [(n, H) for n in REFS] + [(X, H)]
    mz = C.ion_mz(X, H)
    for ppm, verdict in ((2.0, "agree"), (4.5, "contradict")):
        def build(i, ppm=ppm):
            rows = _c13(i, X, H)
            rows[1]["mz"] = (mz + IC.D13C) * (1 + ppm * 1e-6)
            return _refs(i) + rows
        assert _get(_measure(_series(build), pairs), "C", X)["verdict"] == verdict, ppm


def test_rule_c_occupancy_names_another_pair_only():
    """The 'too many' slot guard counts another pair's M0 or named iso_child,
    never the pair's own line, never an unnamed child."""
    y = "C8H14O4"
    pairs = [(n, H) for n in REFS] + [(y, H)]

    def build(stamp, k=N):
        def f(i):
            rows = _c13(i, y, H, c=14.0)
            if i < k:
                rows[1].update(stamp)
            return _refs(i) + rows
        return f
    own_child = dict(role="iso_child", neutral_formula=y, adduct=H, iso_label="13C")
    own_m0 = dict(role="M0", neutral_formula=y, adduct=H, iso_label="", ion_formula=_ion(y, H))
    other_child = dict(role="iso_child", neutral_formula="C9H9NO", adduct=H, iso_label="13C")
    r = _get(_measure(_series(build(own_child)), pairs), "C", y)
    assert r["occupied"] == 0.0 and r["verdict"] == "contradict"
    r = _get(_measure(_series(build(own_m0)), pairs), "C", y)
    assert r["occupied"] == 0.0 and r["verdict"] == "contradict"
    r = _get(_measure(_series(build(other_child, k=12)), pairs), "C", y)
    assert r["occupied"] == pytest.approx(12 / N) and r["verdict"] == "untestable"
    # 9 of 40 (22.5 %) is > 20 %
    r = _get(_measure(_series(lambda i: _refs(i) + _c13(i, y, H, c=14.0, occupied=i < 9)), pairs), "C", y)
    assert r["verdict"] == "untestable"


def test_rule_c_occupancy_is_over_the_used_spectra():
    """An occupied slot in spectra too dim to use does not excuse the reading."""
    y = "C8H14O4"
    pairs = [(n, H) for n in REFS] + [(y, H)]
    ts = _series(lambda i: _refs(i) + _c13(i, y, H, c=14.0, occupied=i < 10, h=20.0 / _wave(i) if i < 10 else 1e5))
    r = _get(_measure(ts, pairs), "C", y)
    assert r["n_used"] == N - 10 and r["occupied"] == 0.0 and r["verdict"] == "contradict"


def test_rule_c_brightness_gate_is_five_times_the_edge():
    """kDL 5: a spectrum is used when h0 x C x R13C >= 5 x its noise edge
    (5.5x used, 4.5x not), and ON the edge it is used (float-exact)."""
    def build(ratio):
        def f(i):
            rr = ratio(i)
            h0 = rr * FILL / (12 * IC.R13C)
            return [_m0row(i, X, H, h0), _row(i, C.ion_mz(X, H) + IC.D13C, h0 * 12 * IC.R13C, role="iso_child",
                                              label="13C")]
        return f
    r = _get(_measure(_series(build(lambda i: 5.5 if i < 10 else (4.5 if i < 20 else 0.5))), [(X, H)]), "C", X)
    assert r["n_used"] == 10
    h_edge = 385.2317554240631                      # h * 12 * R13C == 50.0 exactly
    assert (np.array([h_edge]) * 12 * IC.R13C)[0] == 5.0 * FILL
    ts = _series(lambda i: [_m0row(i, X, H, h_edge), _row(i, C.ion_mz(X, H) + IC.D13C, 50.0, role="iso_child",
                                                          label="13C")])
    assert _get(_measure(ts, [(X, H)]), "C", X)["n_used"] == N


def _noisy(i, c, dlt, n=X):
    """_c13 whose 13C AREA scatters +-dlt spectrum to spectrum (height exact)."""
    rows = _c13(i, n, H, c=c)
    rows[1]["area"] *= 1 + (dlt if i % 2 == 0 else -dlt)
    return rows


def test_rule_c_standard_error_is_the_delta_method():
    """se = sqrt(n/(n-1) sum(res^2)) / sum(x0), res = x1 - r x0, in carbons
    (/ the width factor / R13C) -- pinned on a scattered area series."""
    pairs = [(n, H) for n in REFS] + [(X, H)]
    ts = _series(lambda i: _refs(i) + _noisy(i, 16.0, 0.4))
    r = _get(_measure(ts, pairs), "C", X)
    m = ts[(ts.neutral_formula == X) & (ts.role == "M0")].sort_values("sample_item_id")
    c1 = ts[(ts.iso_label == "13C") & np.isclose(ts.mz, C.ion_mz(X, H) + IC.D13C)].sort_values("sample_item_id")
    x0, x1 = m["area"].to_numpy(), c1["area"].to_numpy()
    rr = x1.sum() / x0.sum()
    se = np.sqrt(N / (N - 1) * ((x1 - rr * x0) ** 2).sum()) / x0.sum()
    mzm = C.ion_mz(X, H)
    assert r["se_area"] == pytest.approx(se / ((mzm + 1) / mzm) ** 1.5 / IC.R13C, rel=1e-9)


def test_rule_c_tolerance_widens_to_three_standard_errors():
    """A 16-C reading for 12 misses by 4 > 0.25 x 12: with the area's 3 se
    above 4 it agrees (the height is exact and says 16), with 3 se below 4 and
    4 se above it it contradicts -- the area's own se, not the height's."""
    pairs = [(n, H) for n in REFS] + [(X, H)]
    r = _get(_measure(_series(lambda i: _refs(i) + _noisy(i, 16.0, 0.56)), pairs), "C", X)
    assert 2 * r["se_area"] < 4.0 - 0.05 < 4.0 + 0.05 < 3 * r["se_area"] and r["se_height"] < 1e-9
    assert r["verdict"] == "agree"
    r = _get(_measure(_series(lambda i: _refs(i) + _noisy(i, 16.0, 0.40)), pairs), "C", X)
    assert 3 * r["se_area"] < 4.0 - 0.05 < 4.0 + 0.05 < 4 * r["se_area"]
    assert r["verdict"] == "contradict"


def test_rule_c_area_on_its_tolerance_agrees():
    """|c_area - C| equal to the tolerance (3.0 for 12 C) is no miss (float-exact)."""
    mz = C.ion_mz(XS, H)
    ion = _ion(XS, H)
    a1 = 9944.092340701434                          # reads exactly 9.0 carbons on the area
    h1 = 1e5 * (6.0 * IC.R13C + C.parse_formula(ion)["O"] * IC.R17O)
    ts = _series(lambda i: [_row(i, mz, 1e5, role="M0", nf=XS, ad=H, ion=ion),
                            _row(i, mz + IC.D13C, h1, a1, role="iso_child", label="13C")])
    r = _get(_measure(ts, [(XS, H)]), "C", XS)
    assert r["c_area"] == 9.0 and r["bias_area"] == 0.0 and r["se_area"] < 1e-9
    assert r["verdict"] == "agree"


def test_rule_c_scan_edge_is_one_dalton():
    """Below scan start + 1 Da is not tested: 0.8 Da above the lowest m/z is
    the edge, 1.2 Da above it is tested."""
    x = "C6H8O4"                                   # 6 C reading 3: a miss of 3 > 1.5
    mz = C.ion_mz(x, H)
    pairs = [(n, H) for n in REFS] + [(x, H)]
    for gap, verdict in ((0.8, "scan_edge"), (1.2, "contradict")):
        ts = _series(lambda i, gap=gap: _refs(i) + _c13(i, x, H, c=3.0) + [_row(i, mz - gap, FILL)], low=False)
        assert _get(_measure(ts, pairs), "C", x)["verdict"] == verdict, gap


EXTRA = ["C11H18O4", "C11H18O5", "C12H20O5", "C13H22O4", "C10H14O4", "C9H12O4", "C8H10O5"]
EDGE = ["C5H2O6", "C6H6O5", "C7H10O4", "C8H14O3", "C9H18O2", "C10H22O", "C11H10O"]


def test_rule_c_bias_population_is_testable_and_above_the_edge():
    """The level-free bias is the median over TESTABLE (>= 8 used spectra)
    heteroatom-free pairs above the scan edge: seven half-reading pairs that
    are too dim, or sit on the scan edge, move nothing."""
    pairs = [(n, H) for n in REFS + [X] + EXTRA]
    ts = _series(lambda i: _refs(i) + _c13(i, X, H, c=6.0) + sum(
        (_c13(i, n, H, c=0.5 * C.parse_formula(n)["C"], phase=0.4 * k) if i < 5 else []   # 5 spectra: untestable
         for k, n in enumerate(EXTRA)), []))
    r = _get(_measure(ts, pairs), "C", X)
    assert r["bias_area"] == pytest.approx(0.0, abs=1e-9) and r["verdict"] == "contradict"
    pairs = [(n, H) for n in REFS + [X] + EDGE]
    ts = _series(lambda i: _refs(i) + _c13(i, X, H, c=6.0) + sum(
        (_c13(i, n, H, c=0.5 * C.parse_formula(n)["C"], phase=0.4 * k) for k, n in enumerate(EDGE)), []), low=False)
    t = _measure(ts, pairs)
    assert set(t.loc[(t["check"] == "C") & t["neutral_formula"].isin(EDGE), "verdict"]) == {"scan_edge"}
    r = _get(t, "C", X)
    assert r["bias_area"] == pytest.approx(0.0, abs=1e-9) and r["verdict"] == "contradict"


def test_rule_c_bias_is_zero_without_a_population():
    """A batch whose every pair carries a heteroatom has no bias population: bias 0."""
    ts = _series(lambda i: _c13(i, XS, H, c=9.1))
    r = _get(_measure(ts, [(XS, H)]), "C", XS)
    assert r["bias_area"] == 0.0 and r["bias_height"] == 0.0 and r["verdict"] == "agree"


def test_rule_c_height_reading_takes_its_own_bias():
    """The references read double on the height only: the height bias divides
    it out, so an area miss with an (unbiased) height hit is ambiguous."""
    pairs = [(n, H) for n in REFS] + [(X, H)]
    ts = _series(lambda i: sum((_c13(i, n, H, c_h=2.0 * C.parse_formula(n)["C"], phase=0.7 * k)
                                for k, n in enumerate(REFS)), []) + _c13(i, X, H, c=6.0, c_h=24.0))
    r = _get(_measure(ts, pairs), "C", X)
    assert r["bias_height"] == pytest.approx(1.0, abs=1e-6) and r["c_height"] == pytest.approx(12.0, abs=1e-6)
    assert r["verdict"] == "ambiguous" and not r["veto"]


# =========================================================================== REQ
def test_req_reads_the_37cl_line_of_a_chlorine_ion():
    """37Cl: 0.3200x per Cl at +1.9970499 Da."""
    n = "C6H9ClO3"
    mz = C.ion_mz(n, H)

    def build(present):
        def f(i):
            h0 = 1e4 * _wave(i)
            rows = [_m0row(i, n, H, h0)]
            if present:
                rows.append(_row(i, mz + 1.9970499, 0.32 * h0, role="iso_child", label="37Cl"))
            return rows
        return f
    r = _get(_measure(_series(build(True)), [(n, H)]), "REQ", n)
    assert r["verdict"] == "present" and r["line"] == "M+2 (37Cl)"
    assert r["expected"] == pytest.approx(0.2424 / 0.7576, abs=2e-3)
    assert _get(_measure(_series(build(False)), [(n, H)]), "REQ", n)["verdict"] == "absent"


def test_req_reads_the_silicon_lines():
    """29Si (+0.9995683, 0.0508x) and 30Si (+1.9968442, 0.0335x) of an Si ion."""
    n = "C6H16O3Si"
    mz = C.ion_mz(n, H)
    r29, r30 = 0.04685 / 0.92223, 0.03092 / 0.92223

    def build(l29=True, l30=True):
        def f(i):
            h0 = 1e5 * _wave(i)
            rows = [_m0row(i, n, H, h0)]
            if l29:
                rows.append(_row(i, mz + 0.9995683, r29 * h0))
            if l30:
                rows.append(_row(i, mz + 1.9968442, r30 * h0))
            return rows
        return f
    r = _get(_measure(_series(build()), [(n, H)]), "REQ", n)
    assert r["verdict"] == "present" and r["line"] == "29Si" and r["expected"] == pytest.approx(r29, abs=2e-3)
    r = _get(_measure(_series(build(l30=False)), [(n, H)]), "REQ", n)
    assert r["verdict"] == "absent" and r["line"] == "30Si" and r["expected"] == pytest.approx(r30, abs=2e-3)


def test_req_orbitrap_window_is_one_ppm():
    assert _get(_measure(_series(lambda i: _br(i, ppm=0.9)), [(Y, H)]), "REQ", Y)["verdict"] == "present"
    assert _get(_measure(_series(lambda i: _br(i, ppm=1.1)), [(Y, H)]), "REQ", Y)["verdict"] == "absent"
    # 4 sigma: sigma 0.4 ppm -> 1.6 ppm, a line at 1.8 ppm is outside
    r = _get(_measure(_series(lambda i: _br(i, ppm=1.8)), [(Y, H)], scale={"sigma_ppm": 0.4, "stamp_ppm": 6.0}),
             "REQ", Y)
    assert r["verdict"] == "absent" and r["window_ppm"] == pytest.approx(1.6)


def test_req_unstamped_pair_is_read_within_the_stamp_window():
    """An unstamped pair takes the tallest peak within the STAMP window (6 ppm) of its pooled m/z."""
    def build(i):
        rows = _br(i, stamped=False, present=lambda i: False)
        rows[0]["mz"] *= 1 + 3e-6
        return rows
    r = _get(_measure(_series(build), [(Y, H)]), "REQ", Y)
    assert not r["stamped"] and r["verdict"] == "absent" and r["n_spectra"] == N


def test_req_detectable_is_three_times_the_floor():
    """81Br expected at 32 cps (> 3 x 10, < 3.5 x 10) is detectable; at 28 cps
    (< 3 x 10, > 2.5 x 10) it is not; ON 30.0 it is (float-exact)."""
    ratio = IC.required_lines(IC.ion_counts(Y, H, _ion(Y, H)), ORBI_R.fwhm(C.ion_mz(Y, H)), 0.0, False)[1][0]["ratio"]

    def at(level):
        return lambda i: [_m0row(i, Y, H, level(i) / ratio)]         # the M0 alone: nothing below the noise
    r = _get(_measure(_series(at(lambda i: 32.0 if i < 10 else (28.0 if i < 20 else 5.0))), [(Y, H)]), "REQ", Y)
    assert r["n_used"] == 10 and r["verdict"] == "absent"
    h = 30.0 / ratio
    for _ in range(64):
        if h * ratio == 30.0:
            break
        h = np.nextafter(h, np.inf if h * ratio < 30.0 else -np.inf)
    assert h * ratio == 30.0
    ts = _series(lambda i: [_m0row(i, Y, H, h)])
    assert _get(_measure(ts, [(Y, H)]), "REQ", Y)["n_used"] == N


ORBI_R = Resolution.from_dict(ORBI)


def test_req_absent_fraction_boundary():
    """A quarter of 12 detectable spectra holding the line is present (> 20 %)."""
    ts = _series(lambda i: _br(i, h=1e4 if i < 12 else 12.0 / _wave(i), present=lambda i: i < 3))
    r = _get(_measure(ts, [(Y, H)]), "REQ", Y)
    assert r["n_used"] == 12 and r["det_frac"] == 0.25 and r["verdict"] == "present"


def test_req_counts_the_line_in_detectable_spectra_only():
    """The line seen only where it could not be detected does not count as present."""
    ts = _series(lambda i: _br(i, h=1e4 if i < 10 else 12.0 / _wave(i), present=lambda i: i >= 10))
    r = _get(_measure(ts, [(Y, H)]), "REQ", Y)
    assert r["n_used"] == 10 and r["n_present"] == 0 and r["verdict"] == "absent"


def test_req_tof_wide_window_is_twenty_ppm():
    for ppm, verdict in ((18.0, "present"), (22.0, "absent")):
        r = _get(_measure(_series(lambda i, ppm=ppm: _br(i, ppm=ppm)), [(Y, H)], resolution=TOF, scale=SCALE_T),
                 "REQ", Y)
        assert r["verdict"] == verdict, ppm
    # a stamp window wider than 20 ppm: the wide count is the union of both windows
    r = _get(_measure(_series(lambda i: _br(i, ppm=22.0)), [(Y, H)], resolution=TOF,
                      scale={"sigma_ppm": 3.6, "stamp_ppm": 25.0}), "REQ", Y)
    assert r["verdict"] == "present" and r["n_present"] == N and r["n_present_wide"] == N


def test_req_a_line_counts_at_the_blend_centroid_or_the_pure_component():
    """On a low-R Orbitrap-class batch the 81Br line of a 30-carbon ion blends
    with its 18O and 13C2 components: a peak within the window of the blend
    centroid OR of the pure 81Br component is the line."""
    n = "C30H45BrO5"
    rp = {"coef": (1 / 60_000) / np.sqrt(200.0), "exponent": 1.5}          # R = 60 000 at m/z 200
    assert IC.instrument_class(rp) == "orbitrap"
    mz = C.ion_mz(n, H)
    counts = IC.ion_counts(n, H, _ion(n, H))
    sc, req = IC.required_lines(counts, Resolution.from_dict(rp).fwhm(mz), 0.0, False)
    q = req[0]
    cen, pure = mz + q["centroid"] - sc, mz + q["pure"] - sc
    assert (cen - pure) / mz * 1e6 > 0.6                                  # the blend sits above the pure line
    for pos in (pure + (cen - pure) + 0.7e-6 * mz,                        # 0.7 ppm above the centroid only
                pure - 0.7e-6 * mz):                                      # 0.7 ppm below the pure line only
        near_c, near_p = abs(pos - cen) / mz * 1e6 <= 1.0, abs(pos - pure) / mz * 1e6 <= 1.0
        assert near_c != near_p
        ts = _series(lambda i, pos=pos: [_m0row(i, n, H, 1e4 * _wave(i)), _row(i, pos, q["ratio"] * 1e4 * _wave(i))])
        r = _get(_measure(ts, [(n, H)], resolution=rp), "REQ", n)
        assert r["verdict"] == "present" and r["n_present"] == N, pos


def test_req_reports_the_most_absent_line():
    """Two absent lines of a Br2 ion: the row reports the emptier one."""
    z = "C2H2Br2O2"
    mz = C.ion_mz(z, H)

    def f(i):
        h = 1e4 * _wave(i)
        rows = [_row(i, mz + D81, h, role="M0", nf=z, ad=H, ion=_ion(z, H))]
        if i < 5:
            rows.append(_row(i, mz + 2 * D81, 0.486 * h))
        return rows
    r = _get(_measure(_series(f), [(z, H)]), "REQ", z)
    assert r["verdict"] == "absent" and r["line"] == "M0 (all-light)" and r["det_frac"] == 0.0


def test_required_lines_take_a_halogen_group_from_a_quarter_of_the_stamped_line():
    """REQ_FRAC 0.25: BrCl committed on its M+2 line leaves the M+4 group at
    0.24x (not required); BrCl3 committed on M+4 requires the M+6 group at 0.27x."""
    fw = 0.002
    sc, req = IC.required_lines({"C": 6, "H": 8, "O": 2, "Br": 1, "Cl": 1}, fw, 1.997, False)
    assert [q["label"] for q in req] == ["M0 (all-light)"]
    sc, req = IC.required_lines({"C": 6, "H": 8, "O": 2, "Br": 1, "Cl": 3}, fw, 3.994, False)
    m6 = [q for q in req if q["label"].startswith("M+6")]
    assert len(m6) == 1 and 0.25 < m6[0]["ratio"] < 0.3


def test_required_lines_take_sulfur_and_silicon_from_half_their_line():
    """REQ_SHARE 0.5: a 29Si component blended with the 13C line is required
    when it is >= half of that line (4 C: 0.53) and not below (6 C: 0.44)."""
    fw = 0.0075                                                            # wide enough to blend 29Si with 13C
    shares = {}
    for nc in (4, 6):
        counts = {"C": nc, "H": 2 * nc, "O": 2, "Si": 1}
        fs, cen, rel = IC._lines(IC.fine_structure(counts), fw)
        m = fs[[t == {"29Si": 1} for t in fs["tags"]]].iloc[0]
        shares[nc] = float(m["rel"] / rel[int(m["line"])])
        sc, req = IC.required_lines(counts, fw, 0.0, False)
        shares[nc, "req"] = "29Si" in [q["label"] for q in req]
    assert 0.5 < shares[4] < 0.6 and shares[4, "req"]
    assert 0.4 < shares[6] < 0.5 and not shares[6, "req"]


def test_required_lines_on_a_tof_take_no_sulfur_or_silicon():
    """required_lines(tof=True): 34S / 29Si / 30Si sit unresolved inside a TOF's
    M+1 / M+2 clusters, so a halogen-free S / Si ion requires nothing there --
    the flag decides, not the width (at the same narrow width the Orbitrap
    reading requires all three). Unreachable through measure(), whose TOF scan
    takes only Br / Cl ions; pinned on the public function."""
    counts = {"C": 6, "H": 9, "O": 3, "S": 1, "Si": 1}
    assert IC.required_lines(counts, 0.001, 0.0, True)[1] == []
    assert {q["label"] for q in IC.required_lines(counts, 0.001, 0.0, False)[1]} == {"34S", "29Si", "30Si"}


def test_observable_lines_merge_closer_than_one_fwhm():
    fs = pd.DataFrame({"shift": [0.0, 0.6, 2.0, 3.0], "rel": [1.0, 1.0, 1.0, 1.0], "tags": [{}, {}, {}, {}]})
    out, cen, rel = IC._lines(fs, 1.0)
    assert out["line"].tolist() == [0, 0, 1, 2]                            # 0.6 < 1 merges; 1.4 and exactly 1.0 do not
    assert cen.tolist() == pytest.approx([0.3, 2.0, 3.0]) and rel.tolist() == [2.0, 1.0, 1.0]


def test_fine_structure_is_multinomial():
    """Count-aware: the 33S34S component of an S2 ion is 2 x a33 x a34."""
    fs = IC.fine_structure({"S": 2})
    a33, a34 = 0.0075 / 0.9499, 0.0425 / 0.9499
    got = fs[[t == {"33S": 1, "34S": 1} for t in fs["tags"]]]["rel"]
    assert len(got) == 1 and got.iloc[0] == pytest.approx(2 * a33 * a34, rel=1e-9)
    assert fs[[t == {"34S": 2} for t in fs["tags"]]]["rel"].iloc[0] == pytest.approx(a34 ** 2, rel=1e-9)


# =========================================================================== HIGH
def _hl(i, *, n=Z, pos=None, ratio=1.0, ratio_h=None, m0_ppm=0.0, m0_role="M0", eps=0.0, flat_h=False, keep=True):
    """An M0 (optionally unstamped / off by m0_ppm) and a partner line at `pos`
    Da above it: area ratio x M0 x exp(eps_i); `flat_h` gives the partner a
    constant height."""
    mz0 = C.ion_mz(n, H) * (1 + m0_ppm * 1e-6)
    h0 = 1e4 * _wave(i)
    rows = [_m0row(i, n, H, h0, mz=mz0, role=m0_role)]
    if keep:
        e = np.exp(eps if i % 2 == 0 else -eps)
        a = ratio * h0 * e
        hh = (ratio if ratio_h is None else ratio_h) * (1e4 * 1.5 if flat_h else h0 * e)
        rows.append(_row(i, mz0 + (1.9979535 if pos is None else pos), hh, a))
    return rows


def _high(ts, klass="orbitrap", n=Z):
    res, scale = (ORBI, SCALE_O) if klass == "orbitrap" else (TOF, SCALE_T)
    return _get(_measure(ts, [(n, H)], resolution=res, scale=scale), "HIGH", n)


def test_high_partner_window():
    """The heavy partner is the nearest peak within 1 ppm (Orbitrap) / 10 ppm (TOF) of M0 + offset."""
    mz = C.ion_mz(Z, H)
    at = lambda ppm: _series(lambda i: _hl(i, pos=1.9979535 + ppm * 1e-6 * mz))   # noqa: E731
    assert _high(at(0.8))["verdict"] == "too_high"
    assert _high(at(1.5))["verdict"] == "consistent"
    assert _high(at(7.0), "tof")["verdict"] == "too_high"
    assert _high(at(15.0), "tof")["verdict"] == "consistent"


def test_high_reads_an_unstamped_pair_within_the_parent_window():
    """An unstamped pair's M0 is the tallest peak within 2 ppm (Orbitrap) / 12
    ppm (TOF) of its pooled m/z; nothing there, no row."""
    for klass, ppm, has in (("orbitrap", 1.5, True), ("orbitrap", 3.0, False),
                            ("tof", 9.0, True), ("tof", 18.0, False)):
        ts = _series(lambda i, ppm=ppm: _hl(i, m0_ppm=ppm, m0_role=""))
        res, scale = (ORBI, SCALE_O) if klass == "orbitrap" else (TOF, SCALE_T)
        t = _measure(ts, [(Z, H)], resolution=res, scale=scale)
        assert _has(t, "HIGH", Z) == has, (klass, ppm)
        if has:
            r = _get(t, "HIGH", Z)
            assert not r["stamped"] and r["verdict"] == "too_high"


def test_high_correlation_needs_eight_co_spectra():
    """Seven of ten spectra hold the line (70 % >= 60 %) but r needs >= 8: consistent."""
    ts = _series(lambda i: _hi(i, n_m0=10, seen=lambda i: i < 7))
    r = _high(ts)
    assert r["n_spectra"] == 10 and r["presence"] == 0.7 and not np.isfinite(r["r"]) and r["verdict"] == "consistent"


def test_high_correlation_threshold_is_0_8():
    r = _high(_series(lambda i: _hl(i, eps=0.35)))
    assert 0.8 < r["r"] < 0.9 and r["verdict"] == "too_high"
    r = _high(_series(lambda i: _hl(i, eps=0.45)))
    assert 0.7 < r["r"] < 0.8 and r["verdict"] == "consistent"


def test_high_correlation_is_on_the_log_areas():
    """The partner's AREA co-varies with the M0's; its height is flat: r reads the areas."""
    r = _high(_series(lambda i: _hl(i, flat_h=True)))
    assert r["r"] == pytest.approx(1.0) and r["verdict"] == "too_high"


def test_high_floor_and_cap_hold_on_both_area_and_height():
    for ra, rh in ((0.5, 0.1), (0.1, 0.5)):                              # the 0.2 floor
        assert _high(_series(lambda i: _hl(i, ratio=ra, ratio_h=rh)))["verdict"] == "consistent", (ra, rh)
    for ra, rh in ((2.9, 3.5), (3.5, 2.9)):                              # expected + 3.0
        r = _high(_series(lambda i: _hl(i, ratio=ra, ratio_h=rh)))
        assert r["verdict"] == "guarded" and "no isotope envelope" in r["note"], (ra, rh)
    assert _high(_series(lambda i: _hl(i, ratio=3.2)))["verdict"] == "guarded"


def test_high_slot_guards_are_fractions_of_the_line_spectra():
    """The slot guards count over the spectra where the LINE was found: 6 of 25 (24 %) guard."""
    for stamp in (dict(role="M0", neutral_formula="C9H9NO", adduct=H), dict(role="iso_child", iso_label="13C")):
        def f(i, stamp=stamp):
            rows = _hl(i, keep=i < 25)
            if i < 6:
                rows[1].update(stamp)
            return rows
        r = _high(_series(f))
        assert r["presence"] == 25 / N and r["verdict"] == "guarded", stamp
    # 9 of 40 (22.5 %) guards
    r = _high(_series(lambda i: _hi(i, stamp=lambda i: ("M0", "C9H9NO", "") if i < 9 else ("", "", ""))))
    assert r["other_m0"] == pytest.approx(9 / N) and r["verdict"] == "guarded"
    r = _high(_series(lambda i: _hi(i, stamp=lambda i: ("iso_child", "", "13C") if i < 9 else ("", "", ""))))
    assert r["other_13c"] == pytest.approx(9 / N) and r["verdict"] == "guarded"


def test_high_every_alias_is_guarded_on_an_orbitrap():
    """A line at the C3<->F2 or the N2<->CH2O spacing sits nearer the alias
    than the heavy spacing (30Si, 18O); one nearer the heavy spacing than the
    alias is a heavy line."""
    for alias, off in (("C3<->F2", "30Si"), ("N2<->CH2O", "18O")):
        r = _high(_series(lambda i, a=alias: _hl(i, pos=IC.HIGH_ALIASES[a])))
        assert r["verdict"] == "guarded" and r["offset"] == off and "alias" in r["note"], alias
    # 0.05 mDa below 34S, 0.086 mDa above F<->OH: nearer 34S -- not alias-guarded (a 1.0x 34S line
    # would need ~22 S, which no ion at this mass carries: the element fit guards it instead)
    r = _high(_series(lambda i: _hl(i, pos=IC.HIGH_OFFSETS["34S"] - 5e-5)))
    assert r["offset"] == "34S" and "alias" not in r["note"]
    assert r["verdict"] == "guarded" and "no ion carrying the S this line needs fits its mass" in r["note"]


def test_high_reports_the_tallest_refuting_line():
    """Two heavy lines too high: the row reports the one with the larger ratio."""
    def f(i):
        rows = _hl(i, ratio=0.5)
        rows.append(_row(i, C.ion_mz(Z, H) + IC.HIGH_OFFSETS["18O"], 1e4 * _wave(i)))
        return rows
    r = _high(_series(f))
    assert r["verdict"] == "too_high" and r["offset"] == "18O" and r["ratio_area"] == pytest.approx(1.0)


def test_high_an_exactly_duplicated_partner_reads_as_its_first_row():
    """Tie-break pin: two rows at the SAME (spectrum, m/z) -- the stamped series
    can carry a duplicate -- read as the first row whichever side of the heavy
    target they sit (here 0.3 ppm below it; the first row is another pair's M0,
    so the slot guard holds)."""
    mz = C.ion_mz(Z, H)

    def f(i):
        rows = _hl(i, keep=False)
        pos = (mz + 1.9979535) * (1 - 0.3e-6)
        h = 1e4 * _wave(i)
        return rows + [_row(i, pos, h, role="M0", nf="C9H9NO", ad=H), _row(i, pos, h)]
    r = _high(_series(f))
    assert r["other_m0"] == 1.0 and r["verdict"] == "guarded"


def test_expected_m2_is_count_aware():
    ion = {"C": 10, "H": 15, "N": 2, "O": 4, "S": 1, "Cl": 1, "Br": 1, "Si": 1}
    a13, a15, a18 = 0.0107 / 0.9893, 0.00364 / 0.99636, 0.00205 / 0.99757
    a34, a37, a81, a30 = 0.0425 / 0.9499, 0.2424 / 0.7576, 0.4931 / 0.5069, 0.03092 / 0.92223
    want = a81 + a37 + a34 + a30 + 4 * a18 + 45 * a13 ** 2 + 20 * a13 * a15
    assert IC.expected_m2(ion) == pytest.approx(want, rel=1e-12)
    assert IC.expected_m2({"C": 10, "H": 16, "O": 4}) == pytest.approx(4 * a18 + 45 * a13 ** 2, rel=1e-12)


# =========================================================================== measure, the facts
def test_a_non_positive_x_edge_reads_as_the_edge_itself():
    ts = _series(lambda i: _br(i, h=12.0 / _wave(i), present=lambda i: False))
    base = _get(_measure(ts, [(Y, H)], x_edge=1.0), "REQ", Y)
    assert base["verdict"] == "untestable"
    for x in (0.0, -2.0, float("nan"), None):
        assert _get(_measure(ts, [(Y, H)], x_edge=x), "REQ", Y)["verdict"] == "untestable", x


def test_veto_reads_the_flag_and_the_note_strictly():
    t = _iso_table([dict(neutral_formula="A", adduct=H, check="REQ", veto=True, note=""),
                    dict(neutral_formula="B", adduct=H, check="HIGH", veto="False", note="b"),
                    dict(neutral_formula="C", adduct=H, check="HIGH", veto="True", note="c"),
                    dict(neutral_formula="D", adduct=H, check="C", veto=float("nan"), note="d"),
                    dict(neutral_formula="E", adduct=H, check="C", veto=None, note="e")])
    assert IC.veto(t) == {("A", H): "REQ", ("C", H): "HIGH: c"}
    assert IC.facts(t) == {"veto": {("A", H): "REQ", ("C", H): "HIGH: c"}, "lock": {}}
    assert IC.facts(t.iloc[:0]) is None and IC.facts(_iso_table([dict(neutral_formula="B", adduct=H, check="C",
                                                                       veto=False)])) == {"veto": {}, "lock": {}}


# =========================================================================== evidence / level_ledger
def test_ion_composition_prefers_a_signed_stored_ion_and_falls_back_to_it():
    """A stored ion formula with a charge sign is the ion; unsigned, the ion is
    neutral + adduct; an unparseable adduct falls back to the stored formula --
    in the engine and the reference script alike."""
    LL = _ll()
    for f in (EV.ion_composition, LL.ion_composition):
        assert f("C8H14O2", "[M+Br]-", "C7H11O4-") == {"C": 7, "H": 11, "O": 4}
        assert f("C8H14O2", "reagent", "C8H14O2Br") == {"C": 8, "H": 14, "O": 2, "Br": 1}
        assert f("C8H14O2", "", "C8H14O2Br") == {"C": 8, "H": 14, "O": 2, "Br": 1}
        assert f("C8H14O4", "[M-H]-", float("nan")) == {"C": 8, "H": 13, "O": 4}
    assert IC.ion_counts("C8H12O4", "[M+^NO3]-", "C8H12^NO7-") == C.parse_formula("C8H12^NO7-")
    assert not LL.carries_reagent("C8H14O2", "[M+Br]-", "C8H14O2Br-", None)
    assert not LL.carries_reagent("C8H14O2", "[M+Br]-", "C8H14O2Br-", "")


def test_the_veto_is_never_an_axis_nor_cross():
    """A vetoed pair's axis count is its own evidence's (the veto adds none, the
    pool exclusion takes chan2), and the veto never makes a pair `cross`."""
    x = "C10H18O4"
    base = TI.EV._series_pooled({"f": TI._branch_rows(x)})
    out = TI.EV._series_pooled({"f": TI._branch_rows(x)}, iso={"veto": {(x, "[M+NO3]-"): "n"}})
    b, o = base.set_index("adduct"), out.set_index("adduct")
    assert o.loc["[M+NO3]-", "iso_veto"] and not b.loc["[M+NO3]-", "iso_veto"]
    assert int(o.loc["[M+NO3]-", "n_axes"]) == int(b.loc["[M+NO3]-", "n_axes"]) - int(b.loc["[M+NO3]-", "chan2"])
    assert not b["cross"].any() and not o["cross"].any()


def test_the_reference_script_vetoes_an_ion_only_pair_like_the_engine():
    LL = _ll()
    x = "C10H16O5"
    rows = ledger([m0("h", x, adduct=H, ion=_ion(x, H), mz=C.ion_mz(x, H)),
                   m0("io", x, adduct="[M]-.", ion="C10H16O5-", mz=C.ion_mz(x, H) + 1.00728, method="ion_only:ea"),
                   child("ic", "io", "13C+1", 1000.0 * EV.C13_PER_CARBON * 10)])
    iso = {"veto": {(x, "[M]-."): "HIGH: h"}}
    core = EV._series_pooled({"s1": rows}, iso=iso)
    ref = LL.assign_levels(LL.measure_source("s1", rows.assign(__file="s1"), None), set(), None, None, iso)
    m = core.merge(ref, left_on=["neutral_formula", "adduct"], right_on=["neutral", "adduct"])
    assert len(m) == 2 and (m["evidence_level"] == m["level"]).all()        # the private decision, alike
    io = m[m["adduct"] == "[M]-."].iloc[0]
    assert io["iso_veto_x"] and io["iso_veto_y"] and io["iso_note_y"] == "HIGH: h"


def test_the_reference_script_iso_table_lookup(tmp_path, capsys):
    """--iso-checks: an out dir holding TWO runs names no table; a vetoed row
    with an empty note reads as the check's name; a named table applies to a
    source without its own table."""
    LL = _ll()
    x = "C10H18O4"
    rows = TI._branch_rows(x)
    table = _iso_table([dict(neutral_formula=x, adduct="[M+NO3]-", check="REQ", veto=True, note="")])
    out = tmp_path / "out"
    for run in ("RUN_1", "RUN_2"):
        (out / run / "per_file").mkdir(parents=True)
        (out / run / "tables").mkdir()
        table.to_csv(out / run / "tables" / "iso_checks.csv", index=False)
    assert LL.iso_check_facts(str(out)) is None and "no iso_checks.csv" in capsys.readouterr().err
    csv = out / "RUN_1" / "tables" / "iso_checks.csv"
    assert LL.iso_check_facts(str(csv)) == {"veto": {(x, "[M+NO3]-"): "REQ"}, "lock": {}} \
        == IC.facts(pd.read_csv(csv))
    bare = tmp_path / "bare"
    (bare / "per_file").mkdir(parents=True)
    rows.to_csv(bare / "per_file" / "s1_ledger.csv", index=False)
    got = LL.run([str(bare)], [], None, None, str(csv)).set_index("adduct")
    assert got.loc["[M+NO3]-", "iso_veto"] and got.loc["[M+NO3]-", "iso_note"] == "REQ"


# =========================================================================== the batch wiring
def test_the_batch_hands_the_checks_its_own_width_model_scale_gate_and_profile(tmp_path, monkeypatch):
    """assign_batch.run calls iso_checks.measure with the batch's width model,
    its MassScale, its height_cutoff_x_edge and its reagent profile."""
    from peaky.assignment import passes as PA
    from peaky.batch import assign_batch as AB
    from peaky.batch import traces as TR
    seen = {}
    real = AB._IC.measure

    def spy(ts, frames, prof=None, **kw):
        seen.update(kw, prof=prof)
        return real(ts, frames, prof, **kw)
    monkeypatch.setattr(AB._IC, "measure", spy)
    real_run = AB.run
    monkeypatch.setattr(AB, "run", lambda *a, **kw: real_run(*a, cfg=PA.PassConfig(height_cutoff_x_edge=4.0), **kw))
    TI._run_batch(tmp_path, monkeypatch)
    summ = json.load(open(tmp_path / "batch_summary.json"))
    assert IC.instrument_class(seen["resolution"]) == "orbitrap"
    assert isinstance(seen["mass_scale"], TR.MassScale)
    assert seen["x_edge"] == 4.0 == summ["height_cutoff_x_edge"]
    assert seen["prof"] is not None and "[M+NO3]-" in seen["prof"].adducts
