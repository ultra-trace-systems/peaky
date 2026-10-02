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
the lines of the ions the certificate commits, the reagent's 81Br line of a Br-free winner only on a
certificate of >= 3 channels; its tests run on physically built spectra scored by the local scorer.

Run: pytest tests/test_known_ion_kids.py -q
"""

from __future__ import annotations

import json

import numpy as np
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
# found for the compound (the user, 2026-10-01: "key it on the ion"). A line naming only the reagent halogen's heavy
# isotope ('81Br', '13C+81Br') under a committed ion of a winner that carries no Br counts only on a certificate of
# >= 3 channels, where two other ions confirm (the user, 2026-10-02): on two, one file, two ions and the reagent's
# own line are not enough for 'identified'. Lines of the winner's own elements (34S, 37Cl; the 81Br of a winner
# that carries Br) count on two channels as on three.
#
# Every spectrum here is physically possible: an ion's lines are the scorer's own envelope (`predict_isotopes`) at
# the ion's height, each a Gaussian of the Orbitrap FWHM (R ~ 118 000 at m/z 200, R ~ m^-1/2); lines whose summed
# profile shows no minimum between them are one peak at their sum and intensity-weighted centroid (a weak line on a
# strong one's flank is a shoulder, not a peak, however far beyond one FWHM it sits), and a peak under the 100-cps
# floor is not picked (one spectrum is the TOF's, as measured). The oracle is the local scorer on that peak list
# (`score_candidates_local`: each line matched to the nearest peak within 5 ppm and 40 % of its predicted height),
# asked for the certificate's compound only. The certificates' members are dim, so their own 37Cl / 34S lines stay
# under the floor: only then is a line of one ion the certificate's only diagnostic line, which each test asserts of
# its spectrum before running the pass.
R200 = 118_000.0
FLOOR = 100.0
DIAG = ("34S", "37Cl", "81Br")


def _sigma(mz):
    """The Gaussian width of a line at `mz`: FWHM = mz / R, R = R200 (200 / mz)^1/2."""
    return mz / (R200 * (200.0 / mz) ** 0.5) / 2.354820045


def _spectrum(ions) -> pd.DataFrame:
    """The peak list `ions` make -- (prefix, ion formula without its sign, charge, height[, ppm off]) each: every
    isotope line of the ion at its predicted height, a Gaussian of the instrument's width; lines with no minimum of
    the summed profile between them are one peak (their sum, at their intensity-weighted centroid), kept at or above
    FLOOR. A peak's id is its tallest line's: the prefix for an M0, '<prefix>:<label>' for another line."""
    lines = []
    for prefix, body, z, h, *off in ions:
        mz, it, lab = predict_isotopes(body, z, None)
        lines += [(float(m) * (1 + (off[0] if off else 0.0) * 1e-6), float(r) * h, prefix, str(lb))
                  for m, r, lb in zip(mz, it / it[0], lab)]
    regions: list[list] = []                      # lines that can touch: within 10 sigma of the previous one
    for ln in sorted(lines):
        if regions and ln[0] - regions[-1][-1][0] < 10 * _sigma(ln[0]):
            regions[-1].append(ln)
        else:
            regions.append([ln])
    rows = []
    for reg in regions:
        m, h = np.array([x[0] for x in reg]), np.array([x[1] for x in reg])
        s = _sigma(m)
        x = np.arange(m[0] - 4 * s[0], m[-1] + 4 * s[-1], s.min() / 100)
        y = (h[:, None] * np.exp(-0.5 * ((x[None, :] - m[:, None]) / s[:, None]) ** 2)).sum(0)
        cuts = x[1:-1][(y[1:-1] < y[:-2]) & (y[1:-1] < y[2:])]          # the summed profile's minima
        side = np.searchsorted(cuts, m)
        for j in range(len(cuts) + 1):
            g = [ln for ln, k in zip(reg, side) if k == j]
            tot = sum(q[1] for q in g)
            if g and tot >= FLOOR:
                top = max(g, key=lambda q: q[1])
                rows.append((top[2] if top[3] == "M0" else f"{top[2]}:{top[3]}", sum(q[0] * q[1] for q in g) / tot,
                             tot))
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


def _hold(led, pid, neutral, adduct, ion, lines=(), note="a Br doublet"):
    """Another reading already holds the doublet at `pid` at Good (pass 4's iso-pair), with its lines."""
    L.commit_assignment(led, pid, neutral_formula=neutral, adduct=adduct, ion_formula=ion, ion_score=0.9, pass_no=4,
                        method="residual:iso-pair", confidence="Good (iso-pair)", commentary=note)
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
    """The same shape with a 34S line: thioacetic acid's (C2H4OS, DBE 1) [M-H]- + [M+NO3]- certificate, its bromide
    cluster (C2H4BrOS-, 4000 cps) held at Good by the isomeric reading C2H5BrOS [M-H]- (DBE 0; the same ion), so no
    member. At m/z 156.9 the cluster's 34S line (179 cps), 2.16 mDa below its 81Br line, is a peak of its own (1.83
    FWHM: the profile dips to 118 cps between them) and the scorer matches it under the uncommitted bromide ion -- a
    34S line is no more a line of the certificate than an 81Br one: Low. (Methanesulfonic acid's would not do: at
    m/z 176.9 the same line is a shoulder of the 81Br peak, and its isomeric reading has DBE -1.) The members' own
    34S lines (4.5 % of 1500 / 1200 cps) are under the floor."""
    taa, hold = "C2H4OS", "C2H5BrOS"
    assert CH.dbe_ok(taa)[0] and CH.dbe_ok(hold)[0] and not CH.dbe_ok("CH5BrO3S")[0]
    assert set(_spectrum([("b", "CH4BrO3S", -1, 4000.0)]).peak_id) == {"b", "b:81Br", "b:81Br+34S"}   # the shoulder
    peaks = _spectrum([("h", "C2H3OS", -1, 1500.0), ("n", "C2H4NO4S", -1, 1200.0), ("b", "C2H4BrOS", -1, 4000.0)])
    assert _diag_lines(peaks, taa, BR_NO3) == {("C2H4BrOS-", "81Br", "b:81Br"), ("C2H4BrOS-", "34S", "b:34S"),
                                               ("C2H4BrOS-", "81Br+34S", "b:81Br+34S")}
    led = L.new_ledger(peaks.copy())
    _hold(led, "b", hold, "[M-H]-", "C2H4BrOS-", ["b:81Br"])
    s, by = _certify(led, peaks, taa, BR_NO3)
    assert s["committed"] == 1 and by.loc["h", "neutral_formula"] == taa and by.loc["n", "neutral_formula"] == taa, s
    assert by.loc["h", "confidence"] == "Low (certified)" and by.loc["n", "confidence"] == "Low (certified)"
    assert "diagnostic isotope envelope" not in by.loc["n", "commentary"]
    assert by.loc["b", "neutral_formula"] == hold and L.role_of(led, "b:34S") == L.ROLE_UNEXPLAINED


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


def test_the_reagent_line_alone_does_not_confirm_a_two_channel_certificate():
    """The same line on TWO channels -- an [M+NO3]- + [M+Br]- or an [M-H]- + [M+Br]- certificate (the third ion
    under the floor) that commits its [M+Br]- ion, whose 1.34x '81Br' M+2 is the only diagnostic line: the line is
    the reagent's bromine on a Br-free winner, and one file, two ions and the reagent's own line are not enough
    (the user, 2026-10-02: the reagent line counts only on a certificate of >= 3 channels). Low (certified), no
    "diagnostic isotope envelope confirmed"; the line still hangs under the [M+Br]- member."""
    for other in (CS_N, CS_H):
        peaks = _spectrum([other, CS_B])
        assert _diag_lines(peaks, CS, BR_NO3) == {("C6H9BrClO3S-", "81Br", "b:81Br")}
        led = L.new_ledger(peaks.copy())
        s, by = _certify(led, peaks, CS, BR_NO3)
        assert s["committed"] == 1 and s["peaks_claimed"] == 2, (other[0], s)
        for pid in (other[0], "b"):
            assert by.loc[pid, "neutral_formula"] == CS and by.loc[pid, "confidence"] == "Low (certified)", pid
            assert not by.loc[pid, "tied"], pid
            assert "diagnostic isotope envelope" not in by.loc[pid, "commentary"], pid
        assert _parent(led, "b:81Br") == "b", other[0]


TOF_CP = "C10H23ClN2O3P2"
TOF_PEAKS = [("n", 378.075402, 2.41), ("b", 395.006538, 48.72), ("b1", 396.009008, 9.45), ("b2", 397.004448, 26.97),
             ("b3", 398.007904, 5.45), ("b4", 398.976160, 2.59), ("b4'", 399.017571, 4.64)]


def test_a_13c_81br_line_alone_confirms_only_a_certificate_of_three_channels():
    """A certificate whose only line is a '13C+81Br' line: the bromide/nitrate TOF batch's own two-channel [M+NO3]- +
    [M+Br]- certificate of this shape (C10H23ClN2O3P2), at the m/z and heights its file measured. The [M+Br]-
    cluster's M+2 read 0.55x of its 48.7-cps M0 (the scorer's '81Br' line expects 0.97x: out of its 40 %), its M+1
    0.19x (a blend: '13C' expects 0.11x), its M+4 lies split 25 mDa below and 16 mDa above (outside the 5 ppm), and
    the nitrate cluster's lines (2.4 cps) are under the TOF's ~1.5-cps edge -- so the scorer's one diagnostic line
    is the cluster's M+3, '13C+81Br' (0.112x of 0.105x): a line of the committed [M+Br]- ion naming only the
    reagent's 81Br beside its carbon. On two channels it confirms nothing: Low (certified). With a third committed
    ion -- the neutral's [M-H]- at 3 cps, its own lines under the edge -- the reagent line counts: Good (certified).
    The line hangs under the [M+Br]- member either way."""
    for h, conf in ((None, "Low (certified)"), (3.0, "Good (certified)")):
        rows = TOF_PEAKS + ([("h", CH.ion_mz(TOF_CP, "[M-H]-"), h)] if h else [])
        peaks = pd.DataFrame(rows, columns=["peak_id", "mz", "height"])
        assert _diag_lines(peaks, TOF_CP, BR_NO3) == {
            ("C10H23BrClN2O3P2-", "13C+81Br", "b3")}                 # privacy-ok: a molecular formula, not an id
        led = L.new_ledger(peaks.copy())
        s, by = _certify(led, peaks, TOF_CP, BR_NO3)
        members = ("n", "b") + (("h",) if h else ())
        assert s["committed"] == 1 and s["peaks_claimed"] == len(members), (h, s)
        for pid in members:
            assert by.loc[pid, "neutral_formula"] == TOF_CP and by.loc[pid, "confidence"] == conf, (h, pid)
        assert _parent(led, "b3") == "b" and by.loc["b3", "iso_label"] == "13C+81Br", h


def test_the_winners_own_34s_or_37cl_line_confirms_a_two_channel_certificate():
    """Lines of the winner's OWN elements count on two channels: an [M-H]- + [M+NO3]- certificate whose members are
    bright enough to show them -- CS's M+2 (37Cl 0.32x + 34S 0.045x, one peak; the scorer's '37Cl' line) on both
    ions, thioacetic acid's 34S line (4.5 %, its own peak at m/z 77 / 140) on both -- and no [M+Br]- ion at all:
    Good (certified), "diagnostic isotope envelope confirmed", each line under its own member."""
    for compound, h, n, label in ((CS, ("h", "C6H8ClO3S", -1, 2000.0), ("n", "C6H9ClNO6S", -1, 1500.0), "37Cl"),
                                  ("C2H4OS", ("h", "C2H3OS", -1, 4000.0), ("n", "C2H4NO4S", -1, 3000.0), "34S")):
        peaks = _spectrum([h, n])
        assert _diag_lines(peaks, compound, BR_NO3) == {(h[1] + "-", label, f"h:{label}"),
                                                        (n[1] + "-", label, f"n:{label}")}, compound
        led = L.new_ledger(peaks.copy())
        s, by = _certify(led, peaks, compound, BR_NO3)
        assert s["committed"] == 1 and s["peaks_claimed"] == 2, (compound, s)
        for pid in ("h", "n"):
            assert by.loc[pid, "neutral_formula"] == compound, (compound, pid)
            assert by.loc[pid, "confidence"] == "Good (certified)", (compound, pid)
            assert "diagnostic isotope envelope confirmed" in by.loc[pid, "commentary"], (compound, pid)
            assert _parent(led, f"{pid}:{label}") == pid, (compound, pid)


def test_a_bromine_bearing_winners_own_81br_line_confirms_a_two_channel_certificate(monkeypatch):
    """A brominated winner's 81Br line is its own bromine, not only the reagent's: it confirms a two-channel
    certificate. 2-bromoethanesulfonic acid (C2H5BrO3S) seen as its [M-H]- and [M+NO3]- ions, each with
    its 1:1 81Br line (the 34S line, 4.5 %, under the floor), no bromide cluster: Good (certified). Pass 7's
    expanded box opens P / S / Cl only, so on the pipeline's own runs no certified winner carries Br; the box is
    opened to Br here to pin the rule for a caller whose box does."""
    from peaky.assignment import certified_neutral as CN
    monkeypatch.setitem(CN._CERT_EXTRA, "Br", (0, 2))
    bes = "C2H5BrO3S"
    peaks = _spectrum([("h", "C2H4BrO3S", -1, 1500.0), ("n", "C2H5BrNO6S", -1, 1200.0)])
    assert _diag_lines(peaks, bes, BR_NO3) == {("C2H4BrO3S-", "81Br", "h:81Br"), ("C2H5BrNO6S-", "81Br", "n:81Br")}
    led = L.new_ledger(peaks.copy())
    s, by = _certify(led, peaks, bes, BR_NO3)
    assert s["committed"] == 1 and s["peaks_claimed"] == 2, s
    for pid in ("h", "n"):
        assert by.loc[pid, "neutral_formula"] == bes and by.loc[pid, "confidence"] == "Good (certified)", pid
        assert "diagnostic isotope envelope confirmed" in by.loc[pid, "commentary"], pid
        assert _parent(led, f"{pid}:81Br") == pid


def test_a_37cl_line_under_an_ion_the_certificate_does_not_commit_confirms_nothing():
    """A 37Cl line is no more a line of the certificate than a 34S or an 81Br one when its ion is not committed:
    dichloroacetic acid (C2H2Cl2O2) certified on its dim [M-H]- (120 cps) and [M+Br]- (130 cps) ions, its [M+NO3]-
    ion (800 cps; its 37Cl line 0.64x, 512 cps) held at Good by the isomeric reading C2H3Cl2NO5 [M-H]- (the same
    ion), so no member. The bromide cluster's M+2 -- 81Br 0.97x + 37Cl 0.64x, one peak at 1.61x -- matches neither
    of the scorer's lines (out of their 40 %), its M+4 and the [M-H]- ion's M+2 are under the floor: the scorer's
    one diagnostic line is the nitrate cluster's '37Cl', under an ion the certificate does not commit. Low."""
    dca, hold = "C2H2Cl2O2", "C2H3Cl2NO5"
    assert CH.dbe_ok(dca)[0] and CH.dbe_ok(hold)[0]
    peaks = _spectrum([("h", "C2HCl2O2", -1, 120.0), ("n", "C2H2Cl2NO5", -1, 800.0), ("b", "C2H2BrCl2O2", -1, 130.0)])
    assert _diag_lines(peaks, dca, BR_NO3) == {("C2H2Cl2NO5-", "37Cl", "n:37Cl")}
    led = L.new_ledger(peaks.copy())
    _hold(led, "n", hold, "[M-H]-", "C2H2Cl2NO5-", ["n:37Cl"], note="a Cl2 doublet")
    s, by = _certify(led, peaks, dca, BR_NO3)
    assert s["committed"] == 1 and by.loc["h", "neutral_formula"] == dca and by.loc["b", "neutral_formula"] == dca, s
    for pid in ("h", "b"):
        assert by.loc[pid, "confidence"] == "Low (certified)", pid
        assert "diagnostic isotope envelope" not in by.loc[pid, "commentary"], pid
    assert by.loc["n", "neutral_formula"] == hold and _parent(led, "n:37Cl") == "n"


# CS and C5H9O4PS (Cl + C for P + O) are 0.18 mDa apart: with every member read 0.22 ppm under CS's ions -- between
# the two -- the oracle scores them within 0.02 of each other and the certificate is TIED.
TIE = ("C6H9ClO3S", "C5H9O4PS")


def _certify_tied(led, peaks):
    def score(client, sid, formulas, *, mechanism_ids=None, **kw):
        return LS.score_candidates_local(peaks, [f for f in formulas if f in TIE], BR_NO3)
    s = P.run_pass_certified(None, "SID", led, AIR, ACFG, BR_NO3, score_fn=score, log=lambda *a: None)
    return s, led.set_index("peak_id")


def _tied(three):
    """The members between CS and C5H9O4PS (`TIE`); the nitrate cluster held by a weak single-channel reading
    (glyceric acid dinitrate C3H4N2O8 [M+NO3]-, Low, 2.8 ppm off)."""
    peaks = _spectrum(([CS_H + (-0.22,)] if three else []) + [CS_N + (-0.22,), CS_B + (-0.22,)])
    led = L.new_ledger(peaks.copy())
    L.commit_assignment(led, "n", neutral_formula="C3H4N2O8", adduct="[M+NO3]-", ion_formula="C3H4N3O11-",
                        ion_score=0.6, ppm_error=-2.83, pass_no=1, method="grid", confidence="Low",
                        commentary="a single-channel fit")
    return _certify_tied(led, peaks)


def test_a_tied_certificate_reads_low_even_with_its_diagnostic_line():
    """A tied certificate reads Low (certified) even where its diagnostic line counts: the three-channel certificate
    of `_tied`, its [M+Br]- ion's 81Br line counted ("diagnostic isotope envelope confirmed"), strong on its three
    channels -- it displaces the weak incumbent of the nitrate cluster -- and every member Low (certified), tied."""
    s, by = _tied(True)
    assert s["committed"] == 1 and s["displaced"] == 1, s
    for pid in ("h", "n", "b"):
        assert by.loc[pid, "neutral_formula"] in TIE and bool(by.loc[pid, "tied"]), pid
        assert by.loc[pid, "confidence"] == "Low (certified)", pid
        assert "diagnostic isotope envelope confirmed" in by.loc[pid, "commentary"], pid


def test_a_tied_two_channel_certificate_on_the_reagent_line_alone_displaces_nothing():
    """The two-channel ([M+NO3]- + [M+Br]-) certificate of `_tied`: its only diagnostic line is the [M+Br]- ion's
    reagent 81Br, which counts for nothing on two channels, so the tied certificate has no displacement strength
    (as built before, the line made it strong): the weak incumbent keeps the nitrate cluster and only the bromide
    cluster is committed, Low (certified), tied, without "diagnostic isotope envelope confirmed"."""
    s, by = _tied(False)
    assert s["committed"] == 1 and s["displaced"] == 0 and s["peaks_claimed"] == 1, s
    assert by.loc["b", "neutral_formula"] in TIE and bool(by.loc["b", "tied"])
    assert by.loc["b", "confidence"] == "Low (certified)"
    assert "diagnostic isotope envelope" not in by.loc["b", "commentary"]
    assert by.loc["n", "neutral_formula"] == "C3H4N2O8" and by.loc["n", "confidence"] == "Low"


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
