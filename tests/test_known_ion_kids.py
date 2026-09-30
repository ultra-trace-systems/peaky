"""C11+c I7 / I7b: a committed ion takes its OWN scored isotope lines.

The scorer returns the lines of every ion of a compound. Pass 0 (known species)
and pass 7 (certified neutral) used to hang every line of the compound under
whichever of its ions they committed -- HNO4's nitrate cluster carried the 81Br
line of HNO4.Br- 18.93 Da up, chloroacetic acid's bromide cluster the 37Cl lines
of its nitrate cluster 14.93 Da below. Now the lines are keyed on (compound,
ion): the attach loop, the recorded isotopologues list, the chlorinated-paraffin
gate and the confidence count the ion's own lines; the single-channel P / S / Si
gates still read the compound. A pass-7 ladder rung is committed with its real
ion formula (it stored the bare neutral) and takes its own lines.

Run: pytest tests/test_known_ion_kids.py -q
"""

from __future__ import annotations

import json

import pandas as pd

from peaky import ledger as L
from peaky import passes as P
from peaky.chem import chemistry as CH
from peaky.chem import contexts as XC

ACFG = P.PassConfig(height_cutoff_cps=100.0)
AIR = XC.get_context("ambient-air")
URO = XC.get_context("uronium")
BR81 = 1.9979535
C13 = 1.0033548


def _ledger(rows):
    return L.new_ledger(pd.DataFrame({"peak_id": [r[0] for r in rows], "mz": [r[1] for r in rows],
                                      "height": [r[2] for r in rows]}))


def _row(compound, ion, label, pid, mz, *, base, score=0.9, mech="m", ppm=0.3):
    return dict(compound_formula=compound, compound_score=score, compound_category=2, ion_formula=ion,
                ion_score=score, ion_category=2, mechanism_id=mech, isotope_formula=ion, iso_label=label,
                is_base=base, theo_mz=mz, rel_abundance=1.0, iso_score=score, iso_category=2,
                sample_peak_id=pid, sample_peak_mz=mz, sample_peak_intensity=1e4, ppm_error=ppm,
                abundance_error=0.0)


# --------------------------------------------------------------------------- pass 0 (I7)
TFA = "C2HF3O2"
MZ_H = CH.ion_mz(TFA, "[M-H]-")                 # C2F3O2-  112.9856
MZ_BR = CH.ion_mz(TFA, "[M+Br]-")               # C2HBrF3O2-  192.9118


def _tfa_scored(h_score=0.9):
    return pd.DataFrame([
        _row(TFA, "C2F3O2-", "M0", "a", MZ_H, base=True, score=h_score, mech="mH"),
        _row(TFA, "C2F3O2-", "13C", "a13", MZ_H + C13, base=False, mech="mH"),
        _row(TFA, "C2HBrF3O2-", "M0", "b", MZ_BR, base=True, mech="mBr"),
        _row(TFA, "C2HBrF3O2-", "81Br", "b81", MZ_BR + BR81, base=False, mech="mBr"),
        _row(TFA, "C2HBrF3O2-", "13C+81Br", "b8113", MZ_BR + BR81 + C13, base=False, mech="mBr")])


def _tfa_ledger():
    return _ledger([("a", MZ_H, 2e4), ("a13", MZ_H + C13, 430.0), ("b", MZ_BR, 1e4),
                    ("b81", MZ_BR + BR81, 9700.0), ("b8113", MZ_BR + BR81 + C13, 210.0)])


def _parent(led, pid):
    return led.loc[led.peak_id == pid, "parent_peak_id"].iloc[0]


def _list(led, pid):
    return {e["peak_id"] for e in json.loads(led.loc[led.peak_id == pid, "isotopologues"].iloc[0] or "[]")}


def test_each_ion_of_a_known_compound_takes_its_own_lines():
    led = _tfa_ledger()
    P.run_pass0_known(None, "SID", led, AIR, ACFG, ["[M-H]-", "[M+Br]-"],
                      score_fn=lambda *a, **k: _tfa_scored(), log=lambda *a: None)
    assert L.role_of(led, "a") == L.role_of(led, "b") == L.ROLE_M0
    assert _parent(led, "a13") == "a"
    assert _parent(led, "b81") == "b" and _parent(led, "b8113") == "b"
    assert _list(led, "a") == {"a13"} and _list(led, "b") == {"b81", "b8113"}


def test_the_confidence_counts_the_ions_own_lines():
    """`Good if ion_score >= 0.7 OR n_kids >= 2`: a 0.6-scored [M-H]- with one own
    line is Low -- the bromide cluster's two lines are not its evidence."""
    led = _tfa_ledger()
    P.run_pass0_known(None, "SID", led, AIR, ACFG, ["[M-H]-", "[M+Br]-"],
                      score_fn=lambda *a, **k: _tfa_scored(h_score=0.6), log=lambda *a: None)
    assert str(led.loc[led.peak_id == "a", "confidence"].iloc[0]).startswith("Low")
    assert str(led.loc[led.peak_id == "b", "confidence"].iloc[0]).startswith("Good")


def test_the_paraffin_gate_counts_the_ions_own_37cl_lines():
    """A chlorinated paraffin's [M-H]- whose two 37Cl lines are its bromide
    cluster's is refused (a known_lead), not committed on the other ion's lines."""
    cp = "C11H18Cl6"
    mz = CH.ion_mz(cp, "[M-H]-")
    mzb = CH.ion_mz(cp, "[M+Br]-")
    scored = pd.DataFrame([
        _row(cp, "C11H17Cl6-", "M0", "cp0", mz, base=True, score=0.7),
        _row(cp, "C11H18BrCl6-", "37Cl", "x1", mzb + 1.99705, base=False),
        _row(cp, "C11H18BrCl6-", "37Cl2", "x2", mzb + 3.9941, base=False)])
    led = _ledger([("cp0", mz, 5000.0), ("x1", mzb + 1.99705, 4000.0), ("x2", mzb + 3.9941, 2500.0)])
    P.run_pass0_known(None, "SID", led, AIR, ACFG, ["[M-H]-", "[M+Br]-"],
                      score_fn=lambda *a, **k: scored, log=lambda *a: None)
    assert L.role_of(led, "cp0") == L.ROLE_UNEXPLAINED
    assert json.loads(led.loc[led.peak_id == "cp0", "known_lead"].iloc[0])["formula"] == cp
    # its OWN two 37Cl lines commit it
    own = scored.assign(ion_formula="C11H17Cl6-", isotope_formula="C11H17Cl6-")
    own.loc[1:, ["theo_mz", "sample_peak_mz"]] = [[mz + 1.99705] * 2, [mz + 3.9941] * 2]
    led2 = _ledger([("cp0", mz, 5000.0), ("x1", mz + 1.99705, 7000.0), ("x2", mz + 3.9941, 5500.0)])
    P.run_pass0_known(None, "SID", led2, AIR, ACFG, ["[M-H]-", "[M+Br]-"],
                      score_fn=lambda *a, **k: own, log=lambda *a: None)
    assert L.role_of(led2, "cp0") == L.ROLE_M0 and _parent(led2, "x1") == "cp0"


# --------------------------------------------------------------------------- pass 7 (I7b) + the rung ion
NBBS = "C10H15NO2S"
MZ_N0 = CH.ion_mz(NBBS, "[M+H]+")
MZ_N1 = CH.ion_mz(NBBS, "[M+(CH4N2O)H]+")
MZ_N2 = MZ_N1 + 60.0323627601
MZ_N3 = CH.ion_mz(NBBS, "[M+NH4]+")
S34 = 1.99580


def _nbbs(client, sid, formulas, *, mechanism_ids=None, **kw):
    if NBBS not in formulas:
        return pd.DataFrame([])
    return pd.DataFrame([
        _row(NBBS, "C10H16NO2S+", "M0", "n0", MZ_N0, base=True, score=0.93, mech="mH", ppm=0.05),
        _row(NBBS, "C10H16NO2S+", "34S", "n0s", MZ_N0 + S34, base=False, mech="mH", ppm=0.05),
        _row(NBBS, "C11H20N3O3S+", "M0", "n1", MZ_N1, base=True, score=0.93, mech="mU", ppm=0.05),
        # the ammonium channel is scored but not anchored on the member peak (the oracle's
        # base line matched no peak): the member commits as a ladder rung of that channel
        _row(NBBS, "C10H19N2O2S+", "M0", None, MZ_N3, base=True, score=0.93, mech="mN", ppm=0.05),
        _row(NBBS, "C10H19N2O2S+", "34S", "n3s", MZ_N3 + S34, base=False, mech="mN", ppm=0.05)])


def _nbbs_ledger():
    return _ledger([("n0", MZ_N0, 50000.0), ("n1", MZ_N1, 400000.0), ("n2", MZ_N2, 3000.0),
                    ("n0s", MZ_N0 + S34, 2200.0), ("n3", MZ_N3, 20000.0), ("n3s", MZ_N3 + S34, 900.0)])


def test_a_certified_members_lines_hang_under_their_own_ion():
    led = _nbbs_ledger()
    s = P.run_pass_certified(None, "SID", led, URO, ACFG, ["[M+H]+", "[M+(CH4N2O)H]+", "[M+NH4]+"],
                             reagent="urea", score_fn=_nbbs, log=lambda *a: None)
    assert s["committed"] == 1 and s["rungs_committed"] == 2, s
    assert _parent(led, "n0s") == "n0"                      # the [M+H]+ ion's 34S line
    assert _parent(led, "n3s") == "n3"                      # the ammonium rung's own 34S line


def test_a_ladder_rung_is_committed_with_its_real_ion_formula():
    led = _nbbs_ledger()
    P.run_pass_certified(None, "SID", led, URO, ACFG, ["[M+H]+", "[M+(CH4N2O)H]+", "[M+NH4]+"],
                         reagent="urea", score_fn=_nbbs, log=lambda *a: None)
    by = led.set_index("peak_id")
    assert by.loc["n2", "method"] == "certified:ladder-rung" and by.loc["n3", "method"] == "certified:ladder-rung"
    assert by.loc["n3", "ion_formula"] == "C10H19N2O2S+"   # the oracle's own string for its channel
    assert by.loc["n2", "ion_formula"] == "C12H24N5O4S+"   # neutral + urea + (CH4N2O)H: one rung above
    assert by.loc["n0", "ion_formula"] == "C10H16NO2S+"


def test_the_rung_ion_reads_the_oracle_then_the_composition():
    from types import SimpleNamespace
    from peaky.assignment.passes import directors as D
    hit = SimpleNamespace(adduct="[M+Br]-", cluster_order=0)
    assert D._rung_ion("C7H6N2OP2", hit, None, {"[M+Br]-": "C7H6BrN2OP2-"}) == "C7H6BrN2OP2-"
    assert D._rung_ion("C7H6N2OP2", hit, None, {}) == "C7H6BrN2OP2-"
    # the oracle writes a labelled reagent's '^N' apart ('...NO6^N-'): its own string wins
    lab = SimpleNamespace(adduct="[M+^NO3]-", cluster_order=0)
    assert D._rung_ion("C5H3F6NO3", lab, None, {"[M+^NO3]-": "C5H3F6NO6^N-"}) == "C5H3F6NO6^N-"
    assert D._rung_ion("C5H3F6NO3", lab, None, {}) == "C5H3F6N2O6-"
    rung = SimpleNamespace(adduct="[M+(CH4N2O)H]+", cluster_order=2)
    assert D._rung_ion(NBBS, rung, "urea", {}) == "C13H28N7O5S+"
    assert D._rung_ion(NBBS, rung, None, {}) is None      # a cluster rung without a molecular reagent
    assert D._rung_ion(NBBS, SimpleNamespace(adduct="bogus", cluster_order=0), None, {}) is None
