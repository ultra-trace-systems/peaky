"""C11+c I7 / I7b: a committed ion takes its OWN scored isotope lines.

The scorer returns the lines of every ion of a compound. Pass 0 (known species)
and pass 7 (certified neutral) used to hang every line of the compound under
whichever of its ions they committed -- HNO4's nitrate cluster carried the 81Br
line of HNO4.Br- 18.93 Da up, chloroacetic acid's bromide cluster the 37Cl lines
of its nitrate cluster 14.93 Da below. Now the lines are keyed on (compound,
ion): the attach loop, the recorded isotopologues list, the chlorinated-paraffin
gate and the confidence count the ion's own lines; the single-channel P / S / Si
gates still read the compound. A pass-7 ladder rung is committed with its real
ion formula (it stored the bare neutral) and takes its own lines. Pass 7's diagnostic-isotope gate reads
the lines of the ions the certificate commits; its tests run on physically built spectra scored by the
local scorer.

Run: pytest tests/test_known_ion_kids.py -q
"""

from __future__ import annotations

import json

import pandas as pd
from mascope_tools.composition.heuristic_filter import predict_isotopes

from peaky import ledger as L
from peaky import passes as P
from peaky.chem import chemistry as CH
from peaky.chem import contexts as XC
from peaky.io import local_scoring as LS

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


# --------------------------------------------------------------------------- pass 7's gate reads the committed ions
# The certificate's diagnostic-isotope gate (34S / 37Cl / 81Br) reads the lines of the ions the certificate COMMITS
# -- each member's: an anchored member's oracle string, a ladder rung's `_rung_ion` -- not every line the scorer
# found for the compound (the user, 2026-10-01: "key it on the ion"). The reagent halogen's line of a committed ion
# counts, on a two-channel certificate as on a three-channel one: a Br-free winner's committed [M+Br]- ion's 81Br
# line proves that channel is a real bromide cluster of the certified neutral (the user, 2026-10-01: "Why would it
# not be evidence if there is two other ions confirming?"; the gate stays as built, the same evening).
#
# Every spectrum here is physically possible: an ion's lines are the scorer's own envelope (`predict_isotopes`) at
# the ion's height, lines closer than the Orbitrap FWHM (R ~ 118 000 at m/z 200, R ~ m^-1/2) are one peak at their
# sum and intensity-weighted centroid, and a peak under the 100-cps floor is not picked (one spectrum is the TOF's,
# as measured). The oracle is the local scorer on that peak list (`score_candidates_local`: each line matched to the
# nearest peak within 5 ppm and 40 % of its predicted height), asked for the certificate's compound only. The
# certificates' members are dim, so their own 37Cl / 34S lines stay under the floor: only then is a line of one ion
# the certificate's only diagnostic line, which each test asserts of its spectrum before running the pass.
R200 = 118_000.0
FLOOR = 100.0
DIAG = ("34S", "37Cl", "81Br")


def _spectrum(ions) -> pd.DataFrame:
    """The peak list `ions` make -- (prefix, ion formula without its sign, charge, height[, ppm off]) each: every
    isotope line of the ion at its predicted height, merged with its neighbours within the FWHM, kept at or above
    FLOOR. A peak's id is its tallest line's: the prefix for an M0, '<prefix>:<label>' for another line."""
    lines = []
    for prefix, body, z, h, *off in ions:
        mz, it, lab = predict_isotopes(body, z, None)
        lines += [(float(m) * (1 + (off[0] if off else 0.0) * 1e-6), float(r) * h, prefix, str(lb))
                  for m, r, lb in zip(mz, it / it[0], lab)]
    groups: list[list] = []
    for ln in sorted(lines):
        if groups and ln[0] - groups[-1][-1][0] < ln[0] / (R200 * (200.0 / ln[0]) ** 0.5):
            groups[-1].append(ln)
        else:
            groups.append([ln])
    rows = []
    for g in groups:
        h = sum(x[1] for x in g)
        if h >= FLOOR:
            top = max(g, key=lambda x: x[1])
            rows.append((top[2] if top[3] == "M0" else f"{top[2]}:{top[3]}", sum(x[0] * x[1] for x in g) / h, h))
    return pd.DataFrame(rows, columns=["peak_id", "mz", "height"])


def _local(peaks, compound, adducts=None, mechanisms=None):
    """The oracle: the local scorer on `peaks`, for `compound` alone."""
    def score(client, sid, formulas, *, mechanism_ids=None, **kw):
        return LS.score_candidates_local(peaks, [f for f in formulas if f == compound], adducts,
                                         mechanisms=mechanisms)
    return score


def _diag_lines(peaks, compound, adducts=None, mechanisms=None) -> set:
    """(ion, label, peak) of every diagnostic line the scorer matches for `compound` (the gate's candidates)."""
    s = LS.score_candidates_local(peaks, [compound], adducts, mechanisms=mechanisms)
    s = s[s["sample_peak_id"].notna() & ~s["is_base"].astype(bool)
          & (pd.to_numeric(s["iso_score"], errors="coerce") > 0.4)]
    return {(i, lb, p) for i, lb, p in zip(s["ion_formula"], s["iso_label"], s["sample_peak_id"])
            if any(d in lb for d in DIAG)}


def _hold(led, pid, neutral, adduct, ion, lines=()):
    """Another reading already holds the doublet at `pid` at Good (pass 4's iso-pair), with its lines."""
    L.commit_assignment(led, pid, neutral_formula=neutral, adduct=adduct, ion_formula=ion, ion_score=0.9, pass_no=4,
                        method="residual:iso-pair", confidence="Good (iso-pair)", commentary="a Br doublet")
    for k in lines:
        L.attach_isotopologue(led, k, pid, iso_label=k.split(":", 1)[1])


def _certify(led, peaks, compound, adducts=None, *, mechanisms=None, profile=AIR, channels=None, reagent=None):
    s = P.run_pass_certified(None, "SID", led, profile, ACFG, channels or adducts, reagent=reagent,
                             score_fn=_local(peaks, compound, adducts, mechanisms), log=lambda *a: None)
    return s, led.set_index("peak_id")


CS = "C6H9ClO3S"                                 # Br-free; its S and Cl want the diagnostic envelope
BR_NO3 = ["[M-H]-", "[M+NO3]-", "[M+Br]-"]
CS_H = ("h", "C6H8ClO3S", -1, 250.0)             # [M-H]-   194.9888; M+2 (37Cl + 34S) 0.365x = 91 cps: unpicked
CS_N = ("n", "C6H9ClNO6S", -1, 200.0)            # [M+NO3]- 257.9845; M+2 0.377x = 75 cps: unpicked
CS_B = ("b", "C6H9BrClO3S", -1, 240.0)           # [M+Br]-  274.9150; M+2 81Br + 37Cl + 34S unresolved, 1.338x =
#                                                  321 cps (the scorer's '81Br' line: 0.973x, inside its 40 %);
#                                                  M+4 0.369x = 89 cps: unpicked


def test_a_line_under_an_ion_the_certificate_does_not_commit_confirms_nothing():
    """The two-channel [M-H]- + [M+NO3]- certificate (the shape of the bromide/nitrate TOF's 11 flips): the doublet
    at CS's [M+Br]- position is another neutral's bromide cluster (C9H2BrF2O3-, 4 ppm above CS's, its M+2 0.97x),
    held at Good by its own reading, so CS's [M+Br]- ion is no member. The scorer matches CS's [M+Br]- ion and its
    '81Br' line onto that doublet -- a line under no ion the certificate commits: no diagnostic envelope, Low."""
    peaks = _spectrum([CS_H, CS_N, ("x", "C9H2BrF2O3", -1, 1.5e4)])
    assert _diag_lines(peaks, CS, BR_NO3) == {("C6H9BrClO3S-", "81Br", "x:81Br")}
    led = L.new_ledger(peaks.copy())
    _hold(led, "x", "C9H2F2O3", "[M+Br]-", "C9H2BrF2O3-", [p for p in peaks.peak_id if p.startswith("x:")])
    s, by = _certify(led, peaks, CS, BR_NO3)
    assert s["committed"] == 1 and by.loc["h", "neutral_formula"] == CS and by.loc["n", "neutral_formula"] == CS, s
    assert by.loc["h", "confidence"] == "Low (certified)" and by.loc["n", "confidence"] == "Low (certified)"
    assert "diagnostic isotope envelope" not in by.loc["h", "commentary"]
    assert by.loc["x", "neutral_formula"] == "C9H2F2O3" and _parent(led, "x:81Br") == "x"


def test_a_34s_line_under_an_ion_the_certificate_does_not_commit_confirms_nothing():
    """The same shape with a 34S line: methanesulfonic acid's [M-H]- + [M+NO3]- certificate, its bromide cluster
    (CH4BrO3S-, 4000 cps) held at Good by the isomeric reading CH5BrO3S [M-H]- (the same ion), so no member. At
    m/z 177 the cluster's 34S line, 2.15 mDa below its 81Br line, is resolved (FWHM 1.4 mDa) and the scorer matches
    it under the uncommitted bromide ion -- a 34S line is no more a line of the certificate than an 81Br one: Low.
    The members' own 34S lines (4.5 % of 1500 / 1200 cps) are under the floor."""
    msa = "CH4O3S"
    peaks = _spectrum([("h", "CH3O3S", -1, 1500.0), ("n", "CH4NO6S", -1, 1200.0), ("b", "CH4BrO3S", -1, 4000.0)])
    assert _diag_lines(peaks, msa, BR_NO3) == {("CH4BrO3S-", "81Br", "b:81Br"), ("CH4BrO3S-", "34S", "b:34S"),
                                               ("CH4BrO3S-", "81Br+34S", "b:81Br+34S")}
    led = L.new_ledger(peaks.copy())
    _hold(led, "b", "CH5BrO3S", "[M-H]-", "CH4BrO3S-", ["b:81Br"])
    s, by = _certify(led, peaks, msa, BR_NO3)
    assert s["committed"] == 1 and by.loc["h", "neutral_formula"] == msa and by.loc["n", "neutral_formula"] == msa, s
    assert by.loc["h", "confidence"] == "Low (certified)" and by.loc["n", "confidence"] == "Low (certified)"
    assert "diagnostic isotope envelope" not in by.loc["n", "commentary"]
    assert by.loc["b", "neutral_formula"] == "CH5BrO3S" and L.role_of(led, "b:34S") == L.ROLE_UNEXPLAINED


def test_the_reagent_line_of_a_committed_bromide_cluster_confirms_the_certificate():
    """The three-channel certificate that commits its [M+Br]- ion: that ion's M+2 -- 81Br + 37Cl + 34S, unresolved,
    1.34x, which the scorer reads as its '81Br' line -- is the reagent's bromine on a Br-free winner and confirms the
    channel is a real bromide cluster of the certified neutral: Good (certified), whether the oracle anchored the
    [M+Br]- member or, the cluster read 3 ppm high (off the 2-ppm anchor gate, inside the scorer's 5 ppm), it is
    committed as a rung of that channel under the oracle's own string. The line hangs under the [M+Br]- member. It
    is the certificate's only diagnostic line: the members' own lines and the cluster's M+4 are under the floor."""
    for ppm, rungs in ((0.0, 0), (3.0, 1)):
        peaks = _spectrum([CS_H, CS_N, CS_B + (ppm,)])
        assert abs(peaks.set_index("peak_id").loc["b:81Br", "height"] / CS_B[3] - 1.338) < 0.005
        assert _diag_lines(peaks, CS, BR_NO3) == {("C6H9BrClO3S-", "81Br", "b:81Br")}
        led = L.new_ledger(peaks.copy())
        s, by = _certify(led, peaks, CS, BR_NO3)
        assert s["committed"] == 1 and s["rungs_committed"] == rungs, (ppm, s)
        assert by.loc["b", "ion_formula"] == "C6H9BrClO3S-" and _parent(led, "b:81Br") == "b", ppm
        for pid in ("h", "n", "b"):
            assert by.loc[pid, "neutral_formula"] == CS and by.loc[pid, "confidence"] == "Good (certified)", (ppm, pid)
        assert "diagnostic isotope envelope confirmed" in by.loc["h", "commentary"], ppm


def test_the_reagent_line_confirms_a_two_channel_certificate_that_commits_its_bromide_cluster():
    """The same line on TWO channels -- an [M+NO3]- + [M+Br]- or an [M-H]- + [M+Br]- certificate (the third ion
    under the floor) that commits its [M+Br]- ion, whose 1.34x '81Br' M+2 is the only diagnostic line: Good
    (certified). One other ion confirms the mass here, and the gate counts the line as on three channels."""
    for other in (CS_N, CS_H):
        peaks = _spectrum([other, CS_B])
        assert _diag_lines(peaks, CS, BR_NO3) == {("C6H9BrClO3S-", "81Br", "b:81Br")}
        led = L.new_ledger(peaks.copy())
        s, by = _certify(led, peaks, CS, BR_NO3)
        assert s["committed"] == 1 and s["peaks_claimed"] == 2, (other[0], s)
        for pid in (other[0], "b"):
            assert by.loc[pid, "neutral_formula"] == CS and by.loc[pid, "confidence"] == "Good (certified)", pid
        assert "diagnostic isotope envelope confirmed" in by.loc[other[0], "commentary"], other[0]
        assert _parent(led, "b:81Br") == "b", other[0]


def test_a_13c_81br_line_alone_under_a_committed_ion_confirms_the_certificate():
    """A certificate passing on a '13C+81Br' line alone: the bromide/nitrate TOF batch's own two-channel [M+NO3]- +
    [M+Br]- certificate of this shape (C10H23ClN2O3P2), at the m/z and heights its file measured. The [M+Br]-
    cluster's M+2 read 0.55x of its 48.7-cps M0 (the scorer's '81Br' line expects 0.97x: out of its 40 %), its M+1
    0.19x (a blend: '13C' expects 0.11x), its M+4 lies split 25 mDa below and 16 mDa above (outside the 5 ppm), and
    the nitrate cluster's lines (2.4 cps) are under the TOF's ~1.5-cps edge -- so the scorer's one diagnostic line
    is the cluster's M+3, '13C+81Br' (0.112x of 0.105x): a line of the committed [M+Br]- ion naming the reagent's
    81Br. Good (certified), the line under that member."""
    cp = "C10H23ClN2O3P2"
    peaks = pd.DataFrame([("n", 378.075402, 2.41), ("b", 395.006538, 48.72), ("b1", 396.009008, 9.45),
                          ("b2", 397.004448, 26.97), ("b3", 398.007904, 5.45), ("b4", 398.976160, 2.59),
                          ("b4'", 399.017571, 4.64)], columns=["peak_id", "mz", "height"])
    assert _diag_lines(peaks, cp, BR_NO3) == {
        ("C10H23BrClN2O3P2-", "13C+81Br", "b3")}                     # privacy-ok: a molecular formula, not an id
    led = L.new_ledger(peaks.copy())
    s, by = _certify(led, peaks, cp, BR_NO3)
    assert s["committed"] == 1 and s["peaks_claimed"] == 2, s
    for pid in ("n", "b"):
        assert by.loc[pid, "neutral_formula"] == cp and by.loc[pid, "confidence"] == "Good (certified)", pid
    assert _parent(led, "b3") == "b" and by.loc["b3", "iso_label"] == "13C+81Br"


def test_a_committed_rungs_own_line_confirms_the_certificate():
    """A ladder rung of order 1 confirms with its own line: NBBS's urea ladder with the urea-dimer cluster
    [M+(CH4N2O)2H]+ the strongest ion (5000 cps) and the only one whose 34S line (4.5 %) clears the floor. The
    oracle here also scores that channel (its string C12H24N5O4S+) and reads the cluster 3 ppm high, off the anchor
    gate, so the member is committed as the order-1 rung of the urea channel under `_rung_ion`'s composition -- the
    same string -- and its 34S line, the certificate's only diagnostic line, confirms it: Good (certified). (The
    pipeline scores exactly the channels it passes to this pass, all of order 0, so on its own runs no rung of
    order >= 1 has a scored line: this pins the rule for a caller whose scorer reads more channels.)"""
    peaks = _spectrum([("n0", "C10H16NO2S", 1, 2000.0), ("n1", "C11H20N3O3S", 1, 2000.0),
                       ("n2", "C12H24N5O4S", 1, 5000.0, 3.0)])
    mech = ["+H+", "+(CH4N2O)H+", "+(CH4N2O)2H+"]
    assert _diag_lines(peaks, NBBS, mechanisms=mech) == {("C12H24N5O4S+", "34S", "n2:34S")}
    led = L.new_ledger(peaks.copy())
    s, by = _certify(led, peaks, NBBS, mechanisms=mech, profile=URO, channels=["[M+H]+", "[M+(CH4N2O)H]+"],
                     reagent="urea")
    assert s["committed"] == 1 and s["rungs_committed"] == 1, s
    assert by.loc["n2", "method"] == "certified:ladder-rung" and by.loc["n2", "ion_formula"] == "C12H24N5O4S+"
    for pid in ("n0", "n1", "n2"):
        assert by.loc[pid, "neutral_formula"] == NBBS and by.loc[pid, "confidence"] == "Good (certified)", pid
    assert _parent(led, "n2:34S") == "n2"
