"""C11+b rule H, the check: the exact-spacing halogen lock (EVIDENCE_LEVELS §3
`lead_lift`; batch/iso_checks.py).

A fourth check of the batch time series writes one row per pooled pair whose ION
carries Cl or Br, with a POSITIVE fact -- `lock` -- and never a veto. It locks
when the ion's stamped M0 has a partner at the exact 37Cl - 35Cl / 81Br - 79Br
spacing (1 ppm) in >= 60 % of >= 20 M0 spectra, co-varying with it (r(log
area) >= 0.8), at a pooled area ratio of 0.65-1.45 x n x 0.3198 (Cl) / 0.9728
(Br) for the ion's count n; unless the M0 is itself the heavy line of a lighter
one (either spacing), the line sits nearer another +2 spacing (30Si, 34S, 18O,
13C2, C3<->F2), the ion carries Si >= 3 or both halogens, or the batch's reagent
could have put the halogen there (it supplies as many as the ion carries). The
synthetic series carry exact ratios, so each gate is tested at its edge.

Run: pytest tests/test_halogen_lock_check.py -q
"""

from __future__ import annotations

import dataclasses
import io

import numpy as np
import pandas as pd
import pytest

from peaky.batch import iso_checks as IC
from peaky.chem import chemistry as C
from peaky.chem import profiles as P
from tests.test_iso_checks import (
    H, LABEL, N, NO3, ORBI, SCALE_O, SCALE_T, TOF, _frames, _get, _ion, _ll, _measure, _row, _series, _wave, quiet)

CL1, BR1, CL2 = "C6H9ClO3", "C2H3BrO2", "C4H6Cl2O2"      # a chloro acid, bromoacetic, a dichloro acid
BIG_CL = "C10H17ClO4"                                     # a chloro acid above m/z 206
BROMIDE = P.resolve("Br")                                 # supplies up to 2 Br per ion ([M+HBr+Br]-)
MIXED = P.resolve("Br+NO3")


def _el(n, a) -> str:
    ion = C.parse_formula(_ion(n, a).rstrip("+-"))
    return "Cl" if ion.get("Cl") else "Br"


def _orth(x: np.ndarray, seed_phase: float = 0.3) -> np.ndarray:
    """A pattern with zero mean and no correlation with `x`."""
    u = np.cos(2 * np.pi * np.arange(len(x)) / 5.0 + seed_phase)
    u = u - u.mean()
    xc = x - x.mean()
    u = u - (u @ xc) / (xc @ xc) * xc
    return u / np.sqrt(u @ u / len(u))


def _noise_for(r: float, h0: np.ndarray) -> np.ndarray:
    """log-area noise that gives a line r(log area) == r with its M0 over all spectra."""
    x = np.log(h0)
    vx = x.var()
    return _orth(x) * np.sqrt(vx * (1.0 / r ** 2 - 1.0))


def _h0(h=1e5, phase=0.0):
    return np.array([h * _wave(i, phase) for i in range(N)])


def _hal(i, n=CL1, a=H, *, ratio=None, d=None, off_ppm=0.0, present=lambda i: True, h=1e5, phase=0.0,
         noise=None, stamp=True, stamp_shift=0.0, n_m0=N, lighter=None, ld=None, lpresent=lambda i: True,
         lrole=""):
    """An M0 stamp of (n, a) and its partner `d` Da above (default the ion's own
    halogen spacing, `off_ppm` off it) at `ratio` x its area (default the
    ion's count x per atom); `lighter`: a line `ld` Da below at M0 / `lighter`."""
    if i >= n_m0:
        return []
    el = _el(n, a)
    ion = C.parse_formula(_ion(n, a).rstrip("+-"))
    ratio = ion[el] * IC.LOCK_PER_ATOM[el] if ratio is None else ratio
    d = IC.LOCK_D[el] if d is None else d
    mz = C.ion_mz(n, a) + stamp_shift
    h0 = h * _wave(i, phase)
    rows = [_row(i, mz, h0, role="M0", nf=n, ad=a, ion=_ion(n, a)) if stamp else _row(i, mz, h0)]
    if present(i):
        a1 = ratio * h0 * (np.exp(noise[i]) if noise is not None else 1.0)
        rows.append(_row(i, mz + d + off_ppm * 1e-6 * mz, a1, role="iso_child",
                         label=IC.LOCK_OFFSET[el]))
    if lighter is not None and lpresent(i):
        rows.append(_row(i, mz - ld, h0 / lighter, role=lrole))
    return rows


def _h(build, pairs, **kw) -> pd.DataFrame:
    return _measure(_series(build), pairs, **kw)


def _verdict(build, n=CL1, a=H, **kw) -> pd.Series:
    return _get(_h(build, [(n, a)], **kw), "H", n, a)


def _inorganic(n, a, prof, **kw) -> pd.Series:
    """The verdict on a carbon-free ion (HBr, HNO3), beside a carbon-bearing pair
    (rule C, which runs first, reads a batch with a carbon line)."""
    ref = "C10H16O4"
    t = _h(lambda i: _hal(i, n, a, **kw) + [_row(i, C.ion_mz(ref, H), 1e5 * _wave(i, 2.0), role="M0", nf=ref,
                                                  ad=H, ion=_ion(ref, H))], [(n, a), (ref, H)], prof=prof)
    return _get(t, "H", n, a)


# =========================================================================== the lock
@pytest.mark.parametrize("n, ratio, verdict", [
    (CL1, 0.3198, "lock"), (BR1, 0.9728, "lock"), (CL2, 2 * 0.3198, "lock"),
    (CL2, 0.3198, "no_lock"),                       # a one-Cl line on a two-Cl ion: the n multiplier
    (CL1, 2 * 0.3198, "no_lock"),                   # ... and the other way
])
def test_a_line_at_the_ions_count_locks(n, ratio, verdict):
    r = _verdict(lambda i: _hal(i, n, ratio=ratio), n)
    assert r["verdict"] == verdict and bool(r["lock"]) == (verdict == "lock")
    assert not bool(r["veto"]) and r["check"] == "H" and r["instrument"] == "orbitrap"
    el = _el(n, H)
    k = C.parse_formula(_ion(n, H).rstrip("-"))[el]
    assert (r["element"], int(r["n_halogen"]), r["offset"]) == (el, k, IC.LOCK_OFFSET[el])
    assert (r["ratio_lo"], r["ratio_hi"]) == pytest.approx(IC.count_window(el, k))
    assert r["ratio_area"] == pytest.approx(ratio) and r["presence"] == 1.0 and r["n_spectra"] == N


def test_the_count_window_and_the_gates_at_their_edges():
    assert IC.count_window("Cl", 1) == pytest.approx((0.65 * 0.3198, 1.45 * 0.3198))
    assert IC.count_window("Br", 2) == pytest.approx((0.65 * 2 * 0.9728, 1.45 * 2 * 0.9728))
    lo, hi = IC.count_window("Cl", 2)
    ok = dict(presence=0.6, r=0.8, ratio=lo, lo=lo, hi=hi)
    assert IC.lock_gates(**ok) and IC.lock_gates(**dict(ok, ratio=hi))
    for bad in (dict(presence=np.nextafter(0.6, 0)), dict(r=np.nextafter(0.8, 0)),
                dict(ratio=np.nextafter(lo, 0)), dict(ratio=np.nextafter(hi, 9)), dict(r=np.nan),
                dict(ratio=np.nan)):
        assert not IC.lock_gates(**dict(ok, **bad)), bad
    # on the series: just inside and just outside each end of the one- and two-Cl windows
    for n in (CL1, CL2):
        k = C.parse_formula(_ion(n, H).rstrip("-"))["Cl"]
        for f, verdict in ((0.649, "no_lock"), (0.651, "lock"), (1.449, "lock"), (1.451, "no_lock")):
            assert _verdict(lambda i: _hal(i, n, ratio=f * k * 0.3198), n)["verdict"] == verdict, (n, f)


@pytest.mark.parametrize("n, off, verdict", [(CL1, 0.99, "lock"), (CL1, 1.01, "no_lock"), (BR1, -0.99, "lock"),
                                             (BR1, -1.01, "no_lock"), (BR1, 0.99, "lock"), (BR1, 1.01, "no_lock")])
def test_the_partner_window_is_one_ppm(n, off, verdict):
    # (a 37Cl line 1 ppm LOW at m/z 163 sits nearer the 30Si spacing: test_a_partner_nearer_...)
    r = _verdict(lambda i: _hal(i, n, off_ppm=off), n)
    assert r["verdict"] == verdict
    if verdict == "no_lock":
        assert r["n_used"] == 0 and f"no line at the {IC.LOCK_OFFSET[_el(n, H)]} offset" in r["note"]


def test_presence_is_counted_over_the_m0s_spectra():
    assert _verdict(lambda i: _hal(i, present=lambda i: i < 24))["verdict"] == "lock"          # 24 / 40 = 0.6
    r = _verdict(lambda i: _hal(i, present=lambda i: i < 23))
    assert r["verdict"] == "no_lock" and "present in < 60%" in r["note"]
    # 12 of 20 stamps locks, 11 of 20 does not: the denominator is the M0's own spectra
    assert _verdict(lambda i: _hal(i, n_m0=20, present=lambda i: i < 12))["verdict"] == "lock"
    assert _verdict(lambda i: _hal(i, n_m0=20, present=lambda i: i < 11))["verdict"] == "no_lock"
    r = _verdict(lambda i: _hal(i, n_m0=19))
    assert r["verdict"] == "untestable" and "stamped in 19 spectra (needs 20)" in r["note"]


def test_the_line_must_co_vary_with_the_m0():
    h0 = _h0()
    for r_target, verdict in ((0.805, "lock"), (0.795, "no_lock")):
        noise = _noise_for(r_target, h0)
        r = _verdict(lambda i: _hal(i, noise=noise, ratio=0.30))
        assert r["r"] == pytest.approx(r_target, abs=1e-9)
        assert r["verdict"] == verdict
    # a line running against the M0 never locks, however tall it reads
    anti = -2.0 * (np.log(h0) - np.log(h0).mean())
    assert _verdict(lambda i: _hal(i, noise=anti, ratio=0.30))["verdict"] == "no_lock"


# =========================================================================== the heavy check
@pytest.mark.parametrize("ld_el, lighter, heavy", [
    ("Cl", 1.0, True),        # the M0 is 1.0x a line one 37Cl spacing below: the Cl3 / Cl4 window
    ("Cl", 1.8, True),        # only the Cl4 window holds 1.8 (the heavy check tries Cl up to 4)
    ("Cl", 0.15, False),      # below one Cl (0.208)
    ("Cl", 2.0, False),       # above four
    ("Br", 1.0, True),        # one Br
    ("Br", 1.9, True),        # only the Br2 window holds 1.9
    ("Br", 3.0, False),       # above two
])
def test_the_heavy_check_at_both_spacings(ld_el, lighter, heavy):
    r = _verdict(lambda i: _hal(i, lighter=lighter, ld=IC.LOCK_D[ld_el]))
    assert r["verdict"] == ("heavy" if heavy else "lock") and bool(r["lock"]) != heavy
    assert bool(r[f"heavy_{ld_el.lower()}"]) == heavy and not bool(r[f"heavy_{'br' if ld_el == 'Cl' else 'cl'}"])
    if heavy:
        assert f"one {IC.LOCK_OFFSET[ld_el]} spacing below it" in r["note"]


def test_the_lighter_line_needs_its_own_presence_and_correlation():
    assert _verdict(lambda i: _hal(i, lighter=1.0, ld=IC.LOCK_D["Cl"], lpresent=lambda i: i < 24))["verdict"] == \
        "heavy"
    assert _verdict(lambda i: _hal(i, lighter=1.0, ld=IC.LOCK_D["Cl"], lpresent=lambda i: i < 23))["verdict"] == \
        "lock"
    h0 = _h0()
    anti = np.exp(2.0 * (np.log(h0) - np.log(h0).mean()))           # a lighter line running against the M0
    r = _verdict(lambda i: _hal(i, lighter=1.0 * anti[i], ld=IC.LOCK_D["Cl"]))
    assert r["verdict"] == "lock" and not bool(r["heavy_cl"])


# =========================================================================== untestable
def test_si_3_is_never_locked_si_2_is():
    si2, si3 = "C5H15ClOSi2", "C7H21ClO2Si3"          # chloro-pentamethyldisiloxane, -heptamethyltrisiloxane
    a = "[M+NO3]-"
    assert _verdict(lambda i: _hal(i, si2, a), si2, a)["verdict"] == "lock"
    r = _verdict(lambda i: _hal(i, si3, a), si3, a)
    assert r["verdict"] == "untestable" and "Si3" in r["note"] and not bool(r["lock"])


def test_a_mixed_halogen_ion_is_untestable():
    n = "C2H2BrClO2"                                  # bromochloroacetic acid
    r = _verdict(lambda i: _hal(i, n, d=IC.LOCK_D["Cl"], ratio=0.3198), n)
    assert r["verdict"] == "untestable" and "both Cl and Br" in r["note"]


def test_an_unstamped_pair_is_untestable():
    r = _verdict(lambda i: _hal(i, stamp=False))
    assert r["verdict"] == "untestable" and "no M0 stamp" in r["note"] and not bool(r["stamped"])


def test_a_pair_stamped_on_a_heavy_isotopologue_is_untestable():
    n = "C2H2Br2O2"                                   # dibromoacetic acid, committed on its 79Br81Br line
    r = _verdict(lambda i: _hal(i, n, stamp_shift=IC.LOCK_D["Br"], ratio=0.5), n)
    assert r["verdict"] == "untestable" and "heavy isotopologue" in r["note"]
    # on its all-light line, the 2 x 0.97 line of two bromines locks
    assert _verdict(lambda i: _hal(i, n), n)["verdict"] == "lock"


# =========================================================================== the reagent (D2)
def test_the_reagent_supply_reads_the_batchs_adducts():
    assert IC.reagent_supply(BROMIDE.adducts, "Br") == 2 and IC.reagent_supply(MIXED.adducts, "Br") == 2
    assert IC.reagent_supply(BROMIDE.adducts, "Cl") == 0
    assert IC.reagent_supply(NO3.adducts, "Br") == 0 and IC.reagent_supply(LABEL.adducts, "Br") == 0
    assert IC.reagent_supply(["[M+Br]-", "[M-H]-"], "Br") == 1
    assert IC.reagent_supply([], "Cl") == IC.reagent_supply(None, "Cl") == 0


def test_a_bromide_batch_never_locks_a_line_its_reagent_could_make():
    """The same ion BrHNO3- read either way: the bromide batch's reagent put the Br
    there (the R3 shape: a perfect 1:1 line) -- whatever the adduct label says."""
    for n, a in (("HNO3", "[M+Br]-"), ("HBr", "[M+NO3]-")):
        r = _inorganic(n, a, MIXED)
        assert r["verdict"] == "reagent" and not bool(r["lock"]), (n, a)
        assert "reagent puts up to 2 Br" in r["note"] and r["ratio_area"] == pytest.approx(0.9728)
    r = _verdict(lambda i: _hal(i, BR1, "[M+Br]-"), BR1, "[M+Br]-", prof=BROMIDE)
    assert r["verdict"] == "reagent"                                 # Br2 ion, the reagent supplies 2


def test_an_ion_carrying_more_than_the_reagent_supplies_locks():
    one = dataclasses.replace(BROMIDE, adducts=["[M+Br]-", "[M-H]-"])   # a bromide batch without [M+HBr+Br]-
    r = _verdict(lambda i: _hal(i, BR1, "[M+Br]-"), BR1, "[M+Br]-", prof=one)
    assert r["verdict"] == "lock" and int(r["n_halogen"]) == 2
    # a chlorine ion on a bromide batch: the reagent supplies no Cl
    assert _verdict(lambda i: _hal(i), prof=BROMIDE)["verdict"] == "lock"


def test_a_nitrate_batch_locks_the_samples_bromine_whatever_the_label():
    for n, a, prof in (("HBr", "[M+^NO3]-", LABEL), ("HBr", "[M+NO3]-", NO3), ("HNO3", "[M+Br]-", NO3)):
        assert _inorganic(n, a, prof)["verdict"] == "lock", (n, a)
    # no profile: the batch's committed adducts stand in for its reagent
    assert _inorganic("HNO3", "[M+Br]-", None)["verdict"] == "reagent"


def test_a_batch_of_carbon_free_pairs_only_is_measured():
    """Rule C finds no 13C line to read where no committed pair carries carbon:
    it tests nothing (it once raised on such a batch) and rule H still locks the
    sample's HBr."""
    n, a = "HBr", "[M+^NO3]-"
    t = _h(lambda i: _hal(i, n, a), [(n, a)], prof=LABEL)
    assert not (t["check"] == "C").any() and _get(t, "H", n, a)["verdict"] == "lock"
    assert IC.summary(t, ORBI)["C"]["tested"] == 0 and IC.summary(t, ORBI)["locked_pairs"] == 1


# =========================================================================== the line position (D4)
def test_a_partner_nearer_the_30si_spacing_does_not_lock():
    d30 = IC.HIGH_OFFSETS["30Si"]
    assert C.ion_mz(BIG_CL, H) > 206          # the 30Si line sits inside the 37Cl line's 1 ppm here
    r = _verdict(lambda i: _hal(i, BIG_CL, d=d30), BIG_CL)
    assert r["verdict"] == "other_spacing" and "nearer the 30Si spacing" in r["note"] and not bool(r["lock"])
    assert r["offset_mda"] == pytest.approx((d30 - IC.LOCK_D["Cl"]) * 1e3)
    assert _verdict(lambda i: _hal(i, BIG_CL), BIG_CL)["verdict"] == "lock"
    # below m/z 206 the 30Si position is outside 1 ppm of the 37Cl one: no line, no lock
    r = _verdict(lambda i: _hal(i, CL1, d=d30))
    assert r["verdict"] == "no_lock" and r["n_used"] == 0


def test_the_nearest_spacing():
    mid = (IC.HIGH_OFFSETS["30Si"] - IC.LOCK_D["Cl"]) / 2 * 1e3            # -0.103 mDa
    assert IC.nearest_spacing("Cl", 0.0) == "37Cl" and IC.nearest_spacing("Cl", mid + 1e-6) == "37Cl"
    assert IC.nearest_spacing("Cl", mid - 1e-6) == "30Si"
    assert IC.nearest_spacing("Cl", (IC.LOCK_OTHER["34S"] - IC.LOCK_D["Cl"]) * 1e3) == "34S"
    assert IC.nearest_spacing("Br", 0.4) == "81Br" and IC.nearest_spacing("Br", 6.0) == "18O"
    assert IC.nearest_spacing("Cl", (IC.LOCK_OTHER["C3<->F2"] - IC.LOCK_D["Cl"]) * 1e3) == "C3<->F2"
    assert IC.nearest_spacing("Cl", 9.6) == "13C2"
    assert set(IC.LOCK_OTHER) == {"30Si", "34S", "18O", "13C2", "C3<->F2"}


@pytest.mark.parametrize("mda", [-8.0, -4.0, 4.0, 8.0])
def test_a_decoy_spacing_never_locks(mda):
    for n in (CL1, BR1):
        d = IC.LOCK_D[_el(n, H)] + mda / 1e3
        assert _verdict(lambda i: _hal(i, n, d=d), n)["verdict"] == "no_lock", (n, mda)


# =========================================================================== the budget (the CF2 analogue)
def test_budget_ok_is_the_budget_with_the_locked_halogen_lifted():
    assert IC.budget_verdict(CL1, "Cl", "uronium") == (True, "Cl=1 > 0")          # Cl its only violation
    assert IC.budget_verdict("C4H7ClO2S2", "Cl", "ambient-air") == (False, "S=2 > 1")
    assert IC.budget_verdict("C4H7ClO2S2", "Cl", "uronium") == (True, "Cl=1 > 0")
    assert IC.budget_verdict(CL1, "Cl", "ambient-air") == (True, "")
    assert IC.budget_verdict("C4H7ClO2S2", "Cl", None) == (True, "")
    assert IC.budget_verdict(CL1, "Cl", "no-such-context") == (True, "")
    t = IC.measure(_series(lambda i: _hal(i) + _hal(i, "C4H7ClO2S2", phase=1.0)),
                   _frames([(CL1, H), ("C4H7ClO2S2", H)]), NO3, resolution=ORBI, mass_scale=SCALE_O,
                   context="ambient-air", log=quiet)
    a, b = _get(t, "H", CL1), _get(t, "H", "C4H7ClO2S2")
    assert (bool(a["budget_ok"]), a["budget_why"]) == (True, "")
    assert (bool(b["budget_ok"]), b["budget_why"]) == (False, "S=2 > 1")
    assert bool(a["lock"]) and bool(b["lock"])        # the lock is the line's; the budget gate is the lift's


# =========================================================================== the table and its facts
def _table(**kw):
    """A lock (CL1), a no-lock (BR1 without its line), a vetoed pair (Y: no 81Br
    line, REQ) and a halogen-free pair (no H row)."""
    y = "C6H9BrO3"
    pairs = [(CL1, H), (BR1, H), (y, H), ("C10H16O4", H)]
    ts = _series(lambda i: _hal(i) + _hal(i, BR1, present=lambda i: False, phase=0.5)
                 + _hal(i, y, present=lambda i: False, phase=1.0, h=1e4)
                 + [_row(i, C.ion_mz("C10H16O4", H), 1e5 * _wave(i, 2.0), role="M0", nf="C10H16O4", ad=H,
                         ion=_ion("C10H16O4", H))])
    return _measure(ts, pairs, **kw)


def test_the_h_rows_never_veto_and_the_facts_carry_the_locks():
    t = _table()
    assert list(t.columns) == list(IC.TABLE_COLUMNS)
    h = t[t["check"] == "H"]
    assert set(h["neutral_formula"]) == {CL1, BR1, "C6H9BrO3"}              # only ions carrying Cl / Br
    assert not h["veto"].any() and t["lock"].dtype == bool
    assert not t.loc[t["check"] != "H", "lock"].any()
    assert IC.veto(t) == IC.veto(t[t["check"] != "H"]) and ("C6H9BrO3", H) in IC.veto(t)
    f = IC.facts(t)
    assert set(f) == {"veto", "lock"} and f["veto"] == IC.veto(t)
    assert set(f["lock"]) == {(CL1, H)}
    lk = f["lock"][(CL1, H)]
    assert set(lk) == {"element", "n", "budget_ok", "note"}
    assert (lk["element"], lk["n"], lk["budget_ok"]) == ("Cl", 1, True) and lk["note"] == _get(t, "H", CL1)["note"]
    s = IC.summary(t, ORBI)
    assert s["locked_pairs"] == 1 and s["vetoed_pairs"] == len(IC.veto(t)) and s["tested"] == len(t)
    assert s["H"] == {"tested": 3, "locked": 1, "lock": 1, "no_lock": 2, "other_spacing": 0, "heavy": 0,
                      "reagent": 0, "untestable": 0}
    assert "vetoed" in s["REQ"] and "locked" not in s["REQ"]


def test_a_table_read_back_from_csv_gives_the_same_facts_in_both_readers(tmp_path):
    t = _table(x_edge=1.0)
    p = tmp_path / "iso_checks.csv"
    t.to_csv(p, index=False)
    back = pd.read_csv(p)
    assert IC.facts(back) == IC.facts(t)
    assert _ll().iso_check_facts(str(p)) == IC.facts(t)
    # only a rule H row locks: a `lock` cell on another check's row is no lock (a hand-edited table)
    odd = back.copy()
    odd.loc[odd["check"] == "REQ", "lock"] = True
    odd.to_csv(p, index=False)
    assert IC.lock(odd) == IC.lock(t) and _ll().iso_check_facts(str(p))["lock"] == IC.lock(t)


def test_a_table_written_before_rule_h_has_no_locks():
    t = _table()
    old = t[t["check"] != "H"].drop(columns=["lock", "element", "n_halogen", "ratio_lo", "ratio_hi", "heavy_cl",
                                             "heavy_br", "budget_ok", "budget_why"])
    assert IC.lock(old) == {} and IC.facts(old) == {"veto": IC.veto(t), "lock": {}}
    buf = io.StringIO()
    old.to_csv(buf, index=False)
    assert IC.facts(pd.read_csv(io.StringIO(buf.getvalue())))["lock"] == {}
    assert IC.summary(old, ORBI)["H"]["tested"] == 0 and IC.summary(old, ORBI)["locked_pairs"] == 0


def test_the_tof_class_runs_rule_h_too():
    r = _verdict(lambda i: _hal(i), resolution=TOF, scale=SCALE_T)
    assert r["instrument"] == "tof" and r["verdict"] == "lock"
    # the TOF's stamp window is no excuse: the partner window stays 1 ppm
    assert _verdict(lambda i: _hal(i, off_ppm=3.0), resolution=TOF, scale=SCALE_T)["verdict"] == "no_lock"


def test_the_log_counts_the_locks():
    seen = []
    IC.measure(_series(lambda i: _hal(i)), _frames([(CL1, H)]), NO3, resolution=ORBI, mass_scale=SCALE_O,
               log=seen.append)
    line = [x for x in seen if x.startswith("[iso_checks]") and "rule H" in x]
    assert line and "rule H 1 tested, 1 locked" in line[0] and "1 locked" in line[0].split("->")[1]



# =========================================================================== the batch
def test_a_batch_writes_the_h_rows_with_its_context_and_their_funnel(tmp_path, monkeypatch):
    """assign_batch.run hands the check its own context (the element budget each
    file demoted against) and writes the H rows and the funnel."""
    import json

    from tests.test_iso_checks import Y, _run_batch
    seen = {}
    real = IC.measure

    def spy(*a, **kw):
        seen.update(kw)
        return real(*a, **kw)
    monkeypatch.setattr(IC, "measure", spy)
    _run_batch(tmp_path, monkeypatch)
    assert seen.get("context") == "ambient-air"
    table = pd.read_csv(tmp_path / "tables" / "iso_checks.csv")
    h = table[table["check"] == "H"]
    assert list(zip(h["neutral_formula"], h["adduct"])) == [(Y, H)]           # the one ion carrying Br
    assert h["verdict"].iloc[0] == "no_lock" and not h["veto"].any() and bool(h["budget_ok"].iloc[0])
    summ = json.load(open(tmp_path / "batch_summary.json"))["evidence_levels"]["iso_checks"]
    assert summ["locked_pairs"] == 0 and summ["H"]["tested"] == 1 and summ["H"]["no_lock"] == 1
