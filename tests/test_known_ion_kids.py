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
    # without it the composition keeps the label as '^N' (C11+c fix round 1: it folded 15N into N, 0.997 Da low)
    assert D._rung_ion("C5H3F6NO3", lab, None, {}) == "C5H3F6N^NO6-"
    assert CH.parse_formula(D._rung_ion("C5H3F6NO3", lab, None, {})) == CH.parse_formula("C5H3F6NO6^N-")
    nh4 = SimpleNamespace(adduct="[M+^NH4]+", cluster_order=0)
    assert D._rung_ion(NBBS, nh4, None, {}) == "C10H19N^NO2S+"
    rung = SimpleNamespace(adduct="[M+(CH4N2O)H]+", cluster_order=2)
    assert D._rung_ion(NBBS, rung, "urea", {}) == "C13H28N7O5S+"
    assert D._rung_ion(NBBS, rung, None, {}) is None      # a cluster rung without a molecular reagent
    assert D._rung_ion(NBBS, SimpleNamespace(adduct="bogus", cluster_order=0), None, {}) is None


# --------------------------------------------------------------------------- the gates stay on the compound (D6 (c))
def test_the_single_channel_s_gate_reads_the_compounds_envelope():
    """D6 (c): iso_confirmed stays keyed on the compound -- the gate licenses the commit, the ion's own lines
    are its evidence. MSA seen on one channel (CH3O3S-, on-cal) whose 34S line the scorer matched under its
    nitrate cluster (scored 9 ppm off: not a second channel): the 34S envelope of ANY of its ions licenses
    the single-channel [M-H]- commit, whose own lines stay its own (the nitrate cluster's 34S line does not
    hang under it). Without that line the single channel is refused."""
    msa = "CH4O3S"
    mz_h, mz_n = CH.ion_mz(msa, "[M-H]-"), CH.ion_mz(msa, "[M+NO3]-")
    s34 = 1.9957963
    scored = pd.DataFrame([
        _row(msa, "CH3O3S-", "M0", "h", mz_h, base=True, mech="mH", ppm=0.2),
        _row(msa, "CH4NO6S-", "M0", "n", mz_n, base=True, mech="mN", ppm=9.0),
        _row(msa, "CH4NO6S-", "34S", "n34", mz_n + s34, base=False, mech="mN", ppm=9.0)])
    rows = [("h", mz_h, 5e4), ("n", mz_n * (1 + 9e-6), 2e4), ("n34", (mz_n + s34) * (1 + 9e-6), 900.0)]
    led = _ledger(rows)
    P.run_pass0_known(None, "SID", led, AIR, ACFG, ["[M-H]-", "[M+NO3]-"],
                      score_fn=lambda *a, **k: scored, log=lambda *a: None)
    assert L.role_of(led, "h") == L.ROLE_M0 and L.role_of(led, "n34") != L.ROLE_ISO
    led2 = _ledger(rows)
    P.run_pass0_known(None, "SID", led2, AIR, ACFG, ["[M-H]-", "[M+NO3]-"],
                      score_fn=lambda *a, **k: scored.iloc[:2], log=lambda *a: None)
    assert L.role_of(led2, "h") != L.ROLE_M0


def test_the_single_channel_si_gate_reads_the_compounds_envelope():
    """D6 (c) for iso_confirmed_si: a D5 siloxane (C10H30O5Si5) seen on one channel ([M+H]+, on-cal, its M+1
    blend present) whose 29Si / 30Si lines the scorer matched under its ammonium adduct (scored 9 ppm off):
    the envelope of any of its ions licenses the commit; without those lines it is refused."""
    d5 = "C10H30O5Si5"
    mz_h, mz_n = CH.ion_mz(d5, "[M+H]+"), CH.ion_mz(d5, "[M+NH4]+")
    s29, s30 = 0.9995681, 1.9968436
    scored = pd.DataFrame([
        _row(d5, "C10H31O5Si5+", "M0", "h", mz_h, base=True, mech="mH", ppm=0.2),
        _row(d5, "C10H34NO5Si5+", "M0", "n", mz_n, base=True, mech="mN", ppm=9.0),
        _row(d5, "C10H34NO5Si5+", "29Si", "n29", mz_n + s29, base=False, mech="mN", ppm=9.0),
        _row(d5, "C10H34NO5Si5+", "30Si", "n30", mz_n + s30, base=False, mech="mN", ppm=9.0)])
    m1 = 5 * 0.0508 + 10 * 0.0107
    rows = [("h", mz_h, 5e4), ("h1", mz_h + s29, 5e4 * m1), ("n", mz_n * (1 + 9e-6), 2e4),
            ("n29", (mz_n + s29) * (1 + 9e-6), 2e4 * 5 * 0.0508), ("n30", (mz_n + s30) * (1 + 9e-6), 2e4 * 5 * 0.0335)]
    led = _ledger(rows)
    P.run_pass0_known(None, "SID", led, URO, ACFG, ["[M+H]+", "[M+NH4]+"],
                      score_fn=lambda *a, **k: scored, log=lambda *a: None)
    assert L.role_of(led, "h") == L.ROLE_M0
    led2 = _ledger(rows)
    P.run_pass0_known(None, "SID", led2, URO, ACFG, ["[M+H]+", "[M+NH4]+"],
                      score_fn=lambda *a, **k: scored.iloc[:2], log=lambda *a: None)
    assert L.role_of(led2, "h") != L.ROLE_M0


# --------------------------------------------------------------------------- a rung on a scored, unanchored channel
MZ_N3L = CH.ion_mz(NBBS, "[M+^NH4]+")


def _nbbs_labelled(client, sid, formulas, *, mechanism_ids=None, **kw):
    """The NBBS certificate with a 15N-ammonium channel the oracle scored (its own string, the label written
    apart: 'C10H19NO2S^N+') but did not anchor on the member peak."""
    if NBBS not in formulas:
        return pd.DataFrame([])
    return pd.DataFrame([
        _row(NBBS, "C10H16NO2S+", "M0", "n0", MZ_N0, base=True, score=0.93, mech="mH", ppm=0.05),
        _row(NBBS, "C10H16NO2S+", "34S", "n0s", MZ_N0 + S34, base=False, mech="mH", ppm=0.05),
        _row(NBBS, "C11H20N3O3S+", "M0", "n1", MZ_N1, base=True, score=0.93, mech="mU", ppm=0.05),
        _row(NBBS, "C10H19NO2S^N+", "M0", None, MZ_N3L, base=True, score=0.93, mech="mN", ppm=0.05),
        _row(NBBS, "C10H19NO2S^N+", "34S", "n3s", MZ_N3L + S34, base=False, mech="mN", ppm=0.05)])


def test_a_rung_on_a_scored_unanchored_channel_takes_the_oracles_string_and_its_lines():
    """The oracle's ion string per channel comes from EVERY base row of the winner, anchored or not: the
    15N-ammonium rung is committed under 'C10H19NO2S^N+' (not a composition string written another way) and
    its own 34S line, keyed on that string, hangs under it."""
    led = _ledger([("n0", MZ_N0, 50000.0), ("n1", MZ_N1, 400000.0), ("n2", MZ_N2, 3000.0),
                   ("n0s", MZ_N0 + S34, 2200.0), ("n3", MZ_N3L, 20000.0), ("n3s", MZ_N3L + S34, 900.0)])
    s = P.run_pass_certified(None, "SID", led, URO, ACFG, ["[M+H]+", "[M+(CH4N2O)H]+", "[M+^NH4]+"],
                             reagent="urea", score_fn=_nbbs_labelled, log=lambda *a: None)
    by = led.set_index("peak_id")
    assert s["rungs_committed"] == 2 and by.loc["n3", "method"] == "certified:ladder-rung", s
    assert by.loc["n3", "ion_formula"] == "C10H19NO2S^N+" and _parent(led, "n3s") == "n3"
