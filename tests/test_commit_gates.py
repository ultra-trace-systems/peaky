"""One commit contract for the heteroatom routes (wrong-assignment guards).

The per-peak grid proposes C / H / N / O only; an S, Cl, Br or Si reading comes
from a widened route -- a pass-7 certificate, a contaminant family, a residual
series, the isotope-gated recovery, a pass-0 known species -- and those routes
committed with weaker evidence than pass 1. The pieces tested here:

  * satellites.own_line_verdict -- the contract: a reading whose NEUTRAL carries
    S / Cl / Br / Si shows its own line where the line is predicted at >= 2x the
    resolved height gate, or the commit is refused; dim, masked, blended or held
    positions are untestable (today's outcome, Candidate at most for a route that
    declares the strict contract); Orbitrap-class runs only;
  * the routes: commit_winners (passes 1-3, 5), pass 3's families at tau_low,
    pass 7 (pooled over the certificate's members), pass 0 (its lead), the
    residual series, the recovery's own-offset M+2, the envelope sweep's own-line
    position and ratio;
  * the batch's known-species cap, now monotone: own 13C / 34S lines count toward
    the two lines, and an exact-mass-only pool of a species whose rule demands
    corroboration is capped too.

Every spectrum is synthetic.

Run: pytest tests/test_commit_gates.py -q
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from peaky import ledger as L
from peaky import passes as P
from peaky.assignment import cleanup as CU
from peaky.assignment import ladders as LAD
from peaky.assignment import reflists as RL
from peaky.assignment import residual as RD
from peaky.assignment import satellites as SAT
from peaky.assignment.passes import directors as D
from peaky.batch import assign_batch as AB
from peaky.chem import chemistry as CH
from peaky.chem import contexts as XC
from peaky.chem import isotopes as ISO
from tests.test_element_evidence import AIR, HN, NOLOG, TAA, _cfg, _file, _p7, _scorer, _srow

S34 = ISO.D_34S
OS = "C5H10O5S"                                   # an organosulfate reading, as the family proposes it
OS_MZ = CH.ion_mz(OS, "[M-H]-")
OS_ION = "C5H9O5S-"


def _v(lines, h0=3000.0, neutral=None, ion=None, floor=10.0, ppm=1.0, mz0=OS_MZ):
    mz = np.array(sorted([m for m, _ in lines] + [mz0]))
    hh = np.array([dict(lines + [(mz0, h0)])[m] for m in mz])
    return SAT.own_line_verdict(mz, hh, mz0, h0, CH.parse_formula(neutral or OS), ion, floor, ppm=ppm)


# --------------------------------------------------------------------------- the contract
def test_an_absent_line_the_file_would_show_refuses_the_reading():
    """S1 at a 3000-cps parent: its 34S line predicted at 133 cps, 13x the 10-cps gate. Absent ->
    refused, with the numbers; present at its ratio -> shown; a P / F / CHNO reading has nothing
    to test."""
    v = _v([])
    assert v["verdict"] == SAT.LG_REFUSED and v["elements"] == {"S": SAT.LG_REFUSED}, v
    assert v["why"].startswith("34S predicted 133 cps (13.3x the 10-cps gate), no line within 1.00 ppm")
    assert _v([(OS_MZ + S34, 120.0)])["verdict"] == SAT.LG_SHOWN
    for neutral in ("C5H10O5", "C5H10O5P", "C5H9F3O2"):
        assert _v([], neutral=neutral)["verdict"] == SAT.LG_NONE


def test_the_multiple_is_two_times_the_gate():
    """Predicted at 1.9x the gate the absence is untestable (today's outcome); at 2.1x it refuses."""
    for x, want in ((1.9, SAT.LG_UNTESTABLE), (2.1, SAT.LG_REFUSED)):
        h0 = x * 10.0 / ISO.R_34S_PER_S
        assert _v([], h0=h0)["verdict"] == want, x


def test_a_low_line_refuses_only_where_its_height_is_not_censored():
    """At >= 4x the gate a line under 0.6 of its prediction refutes (the tier's twin rule); between
    2x and 4x the per-file height is censored low, so only a line under 0.25 of it does."""
    pred = 3000.0 * ISO.R_34S_PER_S                  # 13x
    assert _v([(OS_MZ + S34, 0.5 * pred)])["verdict"] == SAT.LG_REFUSED
    h0 = 3.0 * 10.0 / ISO.R_34S_PER_S                # 3x
    p3 = h0 * ISO.R_34S_PER_S
    assert _v([(OS_MZ + S34, 0.5 * p3)], h0=h0)["verdict"] == SAT.LG_SHOWN
    assert _v([(OS_MZ + S34, 0.1 * p3)], h0=h0)["verdict"] == SAT.LG_REFUSED


def test_a_taller_line_a_blend_and_the_reagents_halogen_are_untestable():
    """A line 3x the prediction is another ion's (held); a neighbour inside one FWHM could hold the
    line (blended); the reagent's own 81Br sits in a bromide adduct's 34S window (masked): none of
    them refuses."""
    pred = 3000.0 * ISO.R_34S_PER_S
    assert _v([(OS_MZ + S34, 3.0 * pred)])["verdict"] == SAT.LG_UNTESTABLE
    assert _v([(OS_MZ + S34 + 0.0008, 2.0 * pred)])["verdict"] == SAT.LG_UNTESTABLE
    mzb = CH.ion_mz(OS, "[M+Br]-")
    v = _v([], mz0=mzb, ion=CH.parse_formula("C5H10BrO5S"))
    assert v["verdict"] == SAT.LG_UNTESTABLE and "reagent adduct" in v["why"], v


def test_silicon_is_shown_by_either_of_its_lines():
    si = "C6H18O3Si3"
    mz0 = CH.ion_mz(si, "[M+H]+")
    v = _v([(mz0 + ISO.D_29SI, 3000.0 * 3 * ISO.R_29SI_PER_SI)], neutral=si, mz0=mz0)
    assert v["verdict"] == SAT.LG_SHOWN and v["elements"] == {"Si": SAT.LG_SHOWN}
    assert _v([], neutral=si, mz0=mz0)["verdict"] == SAT.LG_REFUSED


def test_a_certificate_pools_its_members():
    """An element any member shows stands; one no member shows while a member refuses it is refused."""
    shown = {"verdict": SAT.LG_SHOWN, "elements": {"S": SAT.LG_SHOWN}, "why": ""}
    ref = {"verdict": SAT.LG_REFUSED, "elements": {"S": SAT.LG_REFUSED}, "why": "no 34S"}
    unt = {"verdict": SAT.LG_UNTESTABLE, "elements": {"S": SAT.LG_UNTESTABLE}, "why": "masked"}
    assert SAT.pool_line_verdicts([shown, ref])["verdict"] == SAT.LG_SHOWN
    assert SAT.pool_line_verdicts([unt, ref]) == {"verdict": SAT.LG_REFUSED, "elements": {"S": SAT.LG_REFUSED},
                                                  "why": "no 34S"}
    assert SAT.pool_line_verdicts([unt, unt])["verdict"] == SAT.LG_UNTESTABLE
    assert SAT.pool_line_verdicts([])["verdict"] == SAT.LG_NONE


def test_the_gate_is_orbitrap_class_only():
    led = _file([("p", OS_MZ, 3000.0)])
    for cfg in (_cfg(None), _cfg("tof"), _cfg("orbitrap", trace_sample=True)):
        assert SAT.LineGate(led, cfg).row_verdict(led, "p", OS, OS_ION)["verdict"] == SAT.LG_UNTESTABLE
    assert SAT.LineGate(led, _cfg()).row_verdict(led, "p", OS, OS_ION)["verdict"] == SAT.LG_REFUSED
    assert SAT.LineGate(led, _cfg()).row_verdict(led, "p", "C5H10O5", "C5H9O5-")["verdict"] == SAT.LG_NONE


# --------------------------------------------------------------------------- commit_winners and pass 3
def _arb(pid, mz, neutral=OS, ion=OS_ION, adduct="[M-H]-", score=0.85):
    w = pd.DataFrame([dict(peak_id=pid, raw_score=score, ppm_error=0.1, n_iso=0, tied=False, adduct=adduct,
                           neutral=neutral, ion_formula=ion, ion_score=score, compound_score=score,
                           ion_score_massfree=score, eff_score=score, eff_margin=0.2, alternatives=[],
                           sample_peak_mz=mz)])
    return {"winners": w, "iso_children": pd.DataFrame()}


def _cw(led, cfg, line_gate="refuse"):
    return P.commit_winners(led, _arb("p", OS_MZ), pass_no=3, method="contaminant:organosulfate",
                            context="ambient-air", cfg=cfg, lock=False, min_raw_score=0.7,
                            confidence_suffix="organosulfate", claim_unexplained_only=True, line_gate=line_gate)


def test_commit_winners_refuses_and_writes_why():
    led = _file([("p", OS_MZ, 3000.0)])
    s = _cw(led, _cfg())
    assert s["committed"] == 0 and s["rejected"]["own_line"] == 1, s
    assert L.role_of(led, "p") == L.ROLE_UNEXPLAINED
    note = led.loc[led.peak_id == "p", "commentary"].iloc[0]
    assert note.startswith("REFUSED (own_line: C5H10O5S [M-H]- [pass 3 contaminant:organosulfate] -- 34S ")
    # the same refusal again writes no second note
    _cw(led, _cfg())
    assert led.loc[led.peak_id == "p", "commentary"].iloc[0] == note
    # its line in place: committed, and a summary the contract never touched reads as before
    led = _file([("p", OS_MZ, 3000.0), ("p34", OS_MZ + S34, 133.0)])
    s = _cw(led, _cfg())
    assert s["committed"] == 1 and "own_line" not in s["rejected"], s
    # off an Orbitrap, and with the contract off, the reading commits as before
    for cfg, gate in ((_cfg(None), "refuse"), (_cfg(), None)):
        led = _file([("p", OS_MZ, 3000.0)])
        assert _cw(led, cfg, line_gate=gate)["committed"] == 1


def test_the_strict_contract_caps_an_untestable_reading_at_low():
    """A dim parent: the 34S line is predicted under 2x the gate (untestable). 'refuse' commits it as
    scored (Good); 'strict' commits it Low -- Candidate in the tier engine."""
    h0 = 1.5 * 10.0 / ISO.R_34S_PER_S
    led = _file([("p", OS_MZ, h0)])
    _cw(led, _cfg())
    assert led.loc[led.peak_id == "p", "confidence"].iloc[0] == "Good (organosulfate)"
    led = _file([("p", OS_MZ, h0)])
    s = _cw(led, _cfg(), line_gate="strict")
    assert led.loc[led.peak_id == "p", "confidence"].iloc[0] == "Low (organosulfate)"
    assert s["rejected"]["own_line_untestable_low"] == 1


def _pass3(monkeypatch, led, score, cfg=None):
    rows = [_srow(OS, OS_ION, "M0", "p", OS_MZ, base=True, score=score)]
    monkeypatch.setattr(D, "_enumerate", lambda client, mzs, mech, ranges, cfg, adducts, **k:
                        {OS} if ranges.get("S", (0, 0))[1] > 0 else set())
    monkeypatch.setattr(D, "_mech_ids_for", lambda client, adducts: [])
    monkeypatch.setattr(D.IO, "score_candidates", lambda client, sid, formulas, **k:
                        pd.DataFrame(rows) if OS in formulas else pd.DataFrame([]))
    return D.run_pass3(None, "s", led, AIR, None, cfg or _cfg(), ["[M-H]-", "[M+NO3]-"], reagent=None, **NOLOG)


def test_a_contaminant_family_commits_at_the_pass1_bar_under_the_contract(monkeypatch):
    """Score 0.6 (between tau_suspect and tau_low): no longer committed. Score 0.85: committed with its
    34S line in place, refused without it (the reason on the row)."""
    lines = [("p", OS_MZ, 3000.0), ("p34", OS_MZ + S34, 133.0)]
    led = _file(lines)
    _pass3(monkeypatch, led, 0.6)
    assert L.role_of(led, "p") == L.ROLE_UNEXPLAINED
    led = _file(lines)
    _pass3(monkeypatch, led, 0.85)
    assert L.role_of(led, "p") == L.ROLE_M0 and led.set_index("peak_id").at["p", "method"] == "contaminant:organosulfate"
    led = _file(lines[:1])
    _pass3(monkeypatch, led, 0.85)
    assert L.role_of(led, "p") == L.ROLE_UNEXPLAINED
    assert "REFUSED (own_line: C5H10O5S [M-H]-" in led.set_index("peak_id").at["p", "commentary"]


def test_a_family_declaring_the_strict_contract_commits_an_untestable_reading_low(monkeypatch):
    monkeypatch.setitem(XC.CONTAMINANT_FAMILIES, "organosulfate",
                        dict(XC.CONTAMINANT_FAMILIES["organosulfate"], line_gate="strict"))
    led = _file([("p", OS_MZ, 1.5 * 10.0 / ISO.R_34S_PER_S)])
    _pass3(monkeypatch, led, 0.85)
    assert led.set_index("peak_id").at["p", "confidence"] == "Low (organosulfate)"


# --------------------------------------------------------------------------- pass 7
def test_a_certificate_whose_s_line_no_member_shows_is_refused_and_one_member_suffices():
    """Thioacetic acid's [M+NO3]- / [M+Br]- certificate (no reagent-acid pair): the nitrate member's
    34S line is predicted at 13x the gate and absent, the bromide member's is masked by the reagent's
    81Br -- refused, nothing committed, the reason on both rows. The nitrate member's line in place:
    one member shows the element, and the certificate commits. Off an Orbitrap there is no
    exact-offset test: committed as before."""
    mn, mb = CH.ion_mz(TAA, "[M+NO3]-"), CH.ion_mz(TAA, "[M+Br]-")
    members = [("n", "C2H4NO4S-", mn), ("b", "C2H4BrOS-", mb)]
    ads = HN + ["[M+Br]-"]
    led = _file([("n", mn, 3000.0), ("b", mb, 3000.0)])
    s, by = _p7(led, _scorer(TAA, members), ads)
    assert s["committed"] == 0 and s["line_refused"] == 1, s
    assert all(by.loc[p, "commentary"].startswith("REFUSED (own_line: C2H4OS") for p in ("n", "b"))
    led = _file([("n", mn, 3000.0), ("b", mb, 3000.0), ("n34", mn + S34, 3000.0 * ISO.R_34S_PER_S)])
    s, by = _p7(led, _scorer(TAA, members), ads)
    assert s["committed"] == 1 and "line_refused" not in s, s
    led = _file([("n", mn, 3000.0), ("b", mb, 3000.0)])
    s, by = _p7(led, _scorer(TAA, members), ads, cfg=_cfg(None))
    assert s["committed"] == 1, s


# --------------------------------------------------------------------------- pass 0
MSA = "CH4O3S"
MSA_H, MSA_N = CH.ion_mz(MSA, "[M-H]-"), CH.ion_mz(MSA, "[M+NO3]-")


def _msa_scored():
    return pd.DataFrame([_srow(MSA, "CH3O3S-", "M0", "h", MSA_H, base=True),
                         _srow(MSA, "CH4NO6S-", "M0", "n", MSA_N, base=True)]).assign(
        mechanism_id=["mH", "mN"])


def _p0(led, cfg):
    s = P.run_pass0_known(None, "SID", led, AIR, cfg, ["[M-H]-", "[M+NO3]-"],
                          score_fn=lambda *a, **k: _msa_scored(), **NOLOG)
    return s, led.set_index("peak_id")


def test_pass0_does_not_lock_a_known_s_species_whose_line_is_absent():
    """MSA on two channels (its corroboration rule met), both bright, no 34S line anywhere: neither ion
    is locked; the lead votes 'refuted' where the line was predicted at >= 4x the gate (the twin
    test's own multiple), 'deferred' where under it (a censored absence)."""
    led = _file([("h", MSA_H, 4000.0), ("n", MSA_N, 4000.0)])
    s, by = _p0(led, _cfg())
    assert s["committed"] == 0 and s["line_refused"] == 2, s
    lead = json.loads(by.at["h", "known_lead"])
    assert lead["verdict"] == "refuted" and lead["twin"] == "S", lead
    h3 = 3.0 * 10.0 / ISO.R_34S_PER_S
    led = _file([("h", MSA_H, h3), ("n", MSA_N, h3)])
    s, by = _p0(led, _cfg())
    assert s["committed"] == 0 and json.loads(by.at["h", "known_lead"])["verdict"] == "deferred"
    # its lines in place -> locked as before; off an Orbitrap -> locked as before
    led = _file([("h", MSA_H, 4000.0), ("n", MSA_N, 4000.0), ("h34", MSA_H + S34, 4000.0 * ISO.R_34S_PER_S),
                 ("n34", MSA_N + S34, 4000.0 * ISO.R_34S_PER_S)])
    s, by = _p0(led, _cfg())
    assert s["committed"] == 2 and "line_refused" not in s, s
    led = _file([("h", MSA_H, 4000.0), ("n", MSA_N, 4000.0)])
    s, by = _p0(led, _cfg(None))
    assert s["committed"] == 2, s


def test_a_known_species_with_no_heteroatom_of_its_own_passes():
    """HO2 on a bromide adduct: the ion's Br is the reagent's, the neutral carries no S / Cl / Br / Si
    -- nothing to test (the bromide-cluster radicals stay as they were)."""
    led = _file([("p", CH.ion_mz("HO2", "[M+Br]-"), 3000.0)])
    assert SAT.LineGate(led, _cfg()).row_verdict(led, "p", "HO2", "HBrO2-")["verdict"] == SAT.LG_NONE


# --------------------------------------------------------------------------- the residual series
def test_a_residual_series_reading_refused_when_its_81br_line_is_absent():
    br = "C12H16BrO6"
    mz = CH.ion_mz("C12H17BrO6", "[M-H]-")
    led = _file([("p", mz, 3000.0)])
    w = pd.Series(dict(neutral="C12H17BrO6", adduct="[M-H]-", ion_formula=br + "-"))
    assert RD._own_line_refused(SAT.LineGate(led, _cfg()), led, "p", w, "pass 4 residual:series")
    assert "81Br predicted" in led.set_index("peak_id").at["p", "commentary"]
    led = _file([("p", mz, 3000.0), ("p81", mz + ISO.D_81BR, 2900.0)])
    assert not RD._own_line_refused(SAT.LineGate(led, _cfg()), led, "p", w, "pass 4 residual:series")


# --------------------------------------------------------------------------- the recovery's own M+2
CL = "C5H9ClO3"
CL_MZ = CH.ion_mz(CL, "[M-H]-")


def _recover(lines, cfg):
    led = L.new_ledger(pd.DataFrame(lines, columns=["peak_id", "mz", "height"]))

    def score(client, sid, formulas, *, allow_partial=True, mechanism_ids=None):
        return pd.DataFrame([{"ion_formula": "C5H8ClO3-", "ion_score": 0.74, "ppm_error": 0.1,
                              "sample_peak_id": "p", "sample_peak_mz": CL_MZ,
                              "sample_peak_intensity": 1e4}]) if CL in formulas else pd.DataFrame([])
    cfg.cal_mu, cfg.cal_sigma, cfg.mechanism_ids = 0.0, 0.3, None
    out = CU.recover_isotope_gated(None, "S", led, AIR, cfg, score_fn=score, **NOLOG)
    return out, led


def test_the_recovery_reads_the_candidates_own_37cl_line():
    """A foreign peak 5.6 mDa above the 37Cl position at the Cl ratio: read at the 81Br offset in a
    6-mDa window it was the candidate's '37Cl' line. On an Orbitrap-class run the M+2 is read at the
    candidate's own offset within max(2 ppm, 3 sigma): no recovery. The line in place: recovered.
    Off an Orbitrap the reading is as before."""
    foreign = [("p", CL_MZ, 1e4), ("x", CL_MZ + ISO.D_37CL + 0.0056, 3300.0), ("bg", 180.0, 50.0)]
    out, led = _recover(foreign, _cfg())
    assert out["recovered"] == 0 and L.role_of(led, "p") == L.ROLE_UNEXPLAINED, out
    out, led = _recover([("p", CL_MZ, 1e4), ("c", CL_MZ + ISO.D_37CL, 3300.0), ("bg", 180.0, 50.0)], _cfg())
    assert out["recovered"] == 1 and led.set_index("peak_id").at["p", "neutral_formula"] == CL, out
    out, led = _recover(foreign, _cfg(None))
    assert out["recovered"] == 1, out


# --------------------------------------------------------------------------- the envelope sweep
def _env(lines, cfg, neutral=OS, adduct="[M-H]-", ion=OS_ION):
    led = L.new_ledger(pd.DataFrame(lines, columns=["peak_id", "mz", "height"]))
    L.commit_assignment(led, "p", neutral_formula=neutral, adduct=adduct, ion_formula=ion, ion_score=0.9,
                        ppm_error=0.1, pass_no=3, method="contaminant:under-test", confidence="Good",
                        commentary="under test")
    cfg.cal_sigma = 0.3
    P.complete_isotope_envelopes(led, cfg, **NOLOG)
    return led


def test_the_sweep_attaches_a_heteroatom_line_only_at_its_position_and_ratio():
    """The organosulfate's 34S line: 3 ppm off its position (a neighbour) or at 2.9x its ratio -> stays
    its own peak on an Orbitrap-class run; in place at its ratio -> attached. Its 13C line keeps the
    sweep's 5-ppm / 0.3-3.5x reading (here 1.5 ppm off the 13C position, 4 ppm off the 13C + 33S
    line the pattern merges it into: outside the strict window, attached); off an Orbitrap
    everything reads as before."""
    pred = 3000.0 * ISO.R_34S_PER_S
    c13 = OS_MZ + ISO.D_13C_EXACT
    base = [("p", OS_MZ, 3000.0), ("c", c13 * (1 + 1.5e-6), 3000.0 * 5 * ISO.R_13C_PER_C)]
    off = base + [("s", (OS_MZ + S34) * (1 + 3e-6), pred)]
    high = base + [("s", OS_MZ + S34, 2.9 * pred)]
    good = base + [("s", OS_MZ + S34, pred)]
    for lines, want in ((off, L.ROLE_UNEXPLAINED), (high, L.ROLE_UNEXPLAINED), (good, L.ROLE_ISO)):
        led = _env(lines, _cfg())
        assert L.role_of(led, "s") == want and L.role_of(led, "c") == L.ROLE_ISO, lines
    for lines in (off, high):
        assert L.role_of(_env(lines, _cfg(None)), "s") == L.ROLE_ISO


def test_a_satellite_pulled_toward_a_taller_fine_line_is_still_the_parents():
    """A cyclosiloxane [M+H]+: its 29Si line, resolved from the 13C line 3.8 mDa above it, is picked
    pulled toward it (+1.5 ppm, as real 29Si lines sit on calibrated Orbitrap files) -- inside the
    2-ppm floor, attached although 3 sigma of the bright backbone is 0.9 ppm; a line 3 ppm off at the
    same height is another ion's."""
    si = "C6H18O3Si3"
    ion = "C6H19O3Si3+"
    mz0 = CH.ion_mz(si, "[M+H]+")
    si29 = mz0 + ISO.D_29SI
    h29 = 3000.0 * 3 * ISO.R_29SI_PER_SI
    for shift, want in ((1.5e-6, L.ROLE_ISO), (3e-6, L.ROLE_UNEXPLAINED)):
        led = _env([("p", mz0, 3000.0), ("si", si29 * (1 + shift), h29)], _cfg(), neutral=si,
                   adduct="[M+H]+", ion=ion)
        assert L.role_of(led, "si") == want, shift
    assert SAT.iso_position_ppm(_cfg(), 0.3, 5.0) == SAT.ISO_POS_MIN_PPM == 2.0
    assert SAT.iso_position_ppm(_cfg(), 1.0, 5.0) == 3.0 and SAT.iso_position_ppm(_cfg(), 2.0, 5.0) == 5.0
    assert SAT.iso_position_ppm(_cfg("tof"), 0.3, 5.0) == 5.0


# --------------------------------------------------------------------------- the later routes
def test_the_gate_refuse_helper_writes_why_once():
    led = _file([("p", OS_MZ, 3000.0)])
    gate = SAT.LineGate(led, _cfg())
    assert gate.refuse(led, "p", OS, OS_ION, reading=f"{OS} [M-H]-", route="unit")
    assert not gate.refuse(led, "p", "C5H10O5", "C5H9O5-", reading="x", route="unit")
    assert led.set_index("peak_id").at["p", "commentary"].count("REFUSED (own_line:") == 1


def test_a_ladder_gapfill_of_an_s_anchor_shows_its_own_line():
    """Pass 6 walks CHONS anchors, so a gap-fill can carry S: the C6 homolog of an organosulfate
    anchor is refused where its 34S line is absent, committed where it is in place, and committed as
    before off an Orbitrap."""
    gap = "C6H12O5S"
    gmz = CH.ion_mz(gap, "[M-H]-")

    def run(lines, cfg):
        led = L.new_ledger(pd.DataFrame(lines, columns=["peak_id", "mz", "height"]))
        L.commit_assignment(led, "a", neutral_formula=OS, adduct="[M-H]-", ion_formula=OS_ION, ion_score=0.95,
                            compound_score=0.95, ppm_error=0.1, pass_no=1, method="cheminfo+grid",
                            confidence="Good", commentary="anchor")
        cfg.cal_mu, cfg.cal_sigma, cfg.mechanism_ids = 0.0, 0.5, None

        def score(client, sid, formulas, mechanism_ids=None):
            return pd.DataFrame([dict(is_base=True, sample_peak_id="g", sample_peak_mz=gmz, compound_formula=gap,
                                      ion_formula="C6H11O5S-", ion_score=0.9, compound_score=0.9, ppm_error=0.1,
                                      iso_label="M0", iso_score=None)]) if gap in formulas else pd.DataFrame()
        out = LAD.run_ladder_gapfill(None, "S", led, AIR, cfg, ["[M-H]-"], score_fn=score, log=lambda *a: None)
        return out, led
    base = [("a", OS_MZ, 5000.0), ("g", gmz, 3000.0)]
    out, led = run(base, _cfg())
    assert out["committed"] == 0 and out["own_line_refused"] == 1, out
    assert "REFUSED (own_line: C6H12O5S [M-H]- [pass 6 ladder:gapfill]" in led.set_index("peak_id").at["g", "commentary"]
    out, led = run(base + [("g34", gmz + S34, 3000.0 * ISO.R_34S_PER_S)], _cfg())
    assert out["committed"] == 1 and "own_line_refused" not in out, out
    out, led = run(base, _cfg(None))
    assert out["committed"] == 1, out


def test_a_reference_list_rescue_shows_its_own_line():
    """An isotope-confirmed (13C) list formula carrying S: refused where its 34S line is absent at 13x
    the gate (the row says why, the summary counts it), rescued with the line in place."""
    lst = RL.ReferenceList(id="unit", system="t", label="unit list", data_version="1", polarity="negative",
                           native_detection="[M-H]-", applies_to_contexts=("x",), references=(),
                           formulas=frozenset({OS}), radicals=frozenset(), conditions_of={OS: ()},
                           source_file="u.json", always_active=False, meta_of={})

    def score(client, sid, formulas, *, allow_partial=True, mechanism_ids=None):
        row = dict(compound_formula=OS, compound_score=0.9, ion_formula=OS_ION, ion_score=0.9, iso_label="M0",
                   is_base=True, iso_score=0.9, sample_peak_id="p", sample_peak_mz=OS_MZ,
                   sample_peak_intensity=3000.0, ppm_error=0.1)
        kid = dict(row, is_base=False, iso_label="13C", sample_peak_mz=OS_MZ + ISO.D_13C_EXACT, iso_score=0.95)
        return pd.DataFrame([row, kid])

    def run(lines, cfg):
        led = L.new_ledger(pd.DataFrame(lines, columns=["peak_id", "mz", "height"]))
        cfg.cal_mu, cfg.cal_sigma, cfg.mechanism_ids = 0.0, 0.3, None
        return RL.rescue_unexplained_by_reflist(None, "S", led, None, cfg, [lst], ["[M-H]-"], score_fn=score,
                                                log=lambda *a: None), led
    out, led = run([("p", OS_MZ, 3000.0)], _cfg())
    assert out == {"rescued": 0, "tentative": 0, "own_line_refused": 1}, out
    assert L.role_of(led, "p") == L.ROLE_UNEXPLAINED
    assert "[reflist-rescue:unit] -- 34S predicted" in led.set_index("peak_id").at["p", "commentary"]
    out, led = run([("p", OS_MZ, 3000.0), ("p34", OS_MZ + S34, 3000.0 * ISO.R_34S_PER_S)], _cfg())
    assert out == {"rescued": 1, "tentative": 0}, out
    out, led = run([("p", OS_MZ, 3000.0)], _cfg(None))
    assert out == {"rescued": 1, "tentative": 0}, out


def test_the_offcal_rearbitration_never_swaps_to_a_refused_alternative():
    """The off-cal aromatic monster of the re-arbitration test, its on-cal alternative now a sulfur
    reading: with no 34S line where the file would show it the monster is kept (the tier engine
    demotes it); off an Orbitrap the swap is made as before."""
    from tests.test_rearbitrate import build_ledger

    def run(cfg):
        led = build_ledger()
        i = led.index[led["peak_id"] == "P_swap"][0]
        led.at[i, "alternatives"] = json.dumps([{"formula": "C21H23N3O6S", "adduct": "[M+Na]+", "ion_score": 0.70,
                                                 "raw_score": 0.70, "eff_score": 0.61, "ppm": 0.15}])
        return P.rearbitrate_offcal_degenerate(led, cfg, log=lambda *a: None), led.set_index("peak_id")
    res, by = run(_cfg())
    assert res["swapped"] == 0 and res["own_line_refused"] == 1 and by.at["P_swap", "neutral_formula"] == "C36H17N", res
    res, by = run(_cfg(None))
    assert res["swapped"] == 1 and by.at["P_swap", "neutral_formula"] == "C21H23N3O6S", res


# --------------------------------------------------------------------------- the known-species cap
def _merged(nf, ad, mz, tier="Assigned"):
    return pd.DataFrame([dict(mz=mz, neutral_formula=nf, adduct=ad, tier=tier, ion_score=0.9, n_files=3,
                              n_files_ion=3, n_files_winner=3, alternatives="", tier_reason=pd.NA,
                              srcs="a,b,c", admitted_by="height", occurrence=1.0)])


def _rec(nf, ad, mz, family, src, why, **kw):
    return dict(dict(neutral=nf, adduct=ad, mz=mz, family=family, label=nf, src=src, verdict="confirmed",
                     why=why, summary=why, ion_score=0.9, tier="Assigned", admitted_by="height",
                     occurrence=1.0, n_channels=1, n_satellites=0), **kw)


def _lock(nf, ad, mz, family, **kw):
    m = _merged(nf, ad, mz)
    why = kw.pop("why", AB.MASS_ONLY_ROUTE)
    AB.lock_known_species(m, [_rec(nf, ad, mz, family, s, why, **kw) for s in "abc"], tol_ppm=6.0,
                          log=lambda *a: None)
    return m.iloc[0]


def test_the_known_species_cap_is_monotone():
    """H2SO4 (its rule demands a second channel or the 34S line): exact mass alone in every file was
    Assigned while one 34S line made it Candidate -- now both are Candidate; MTBT's one channel + its
    34S line + its own 13C line is two lines -> Assigned. A PFCA (exact mass is its rule) stays
    Assigned at any line count, and a reactive-iodine ICl with its 37Cl line is no longer capped
    below its exact-mass-only reading."""
    h2so4 = _lock("H2SO4", "[M-H]-", 96.9601, "atmospheric")
    assert h2so4["tier"] == "Candidate" and "exact mass alone in every confirming file" in h2so4["tier_reason"]
    h2so4 = _lock("H2SO4", "[M-H]-", 96.9601, "atmospheric", n_satellites=1,
                  why="corroborated by a confirmed 34S envelope (1 satellite)")
    assert h2so4["tier"] == "Candidate" and "one ion channel and one satellite line" in h2so4["tier_reason"]
    mtbt = "C8H7NS2"
    one = _lock(mtbt, "[M+H]+", 182.0093, "indoor_sulfur", n_satellites=1,
                why="corroborated by a confirmed 34S envelope (1 satellite)")
    assert one["tier"] == "Candidate"
    two = _lock(mtbt, "[M+H]+", 182.0093, "indoor_sulfur", n_satellites=1, n_c13=1,
                why="corroborated by a confirmed 34S envelope (1 satellite)")
    assert two["tier"] == "Assigned" and "capped" not in two["tier_reason"]
    assert _lock("C5HF9O2", "[M-H]-", 262.9760, "perfluoroacid")["tier"] == "Assigned"
    icl = _lock("ICl", "[M+^NO3]-", 224.8587, "reactive_iodine", n_satellites=1,
                why="corroborated by a confirmed 37Cl envelope (1 satellite)")
    assert icl["tier"] == "Assigned" and "capped" not in icl["tier_reason"]
    assert D.known_demands_corroboration("atmospheric", "H2SO4") and D.known_demands_corroboration("x", "H2O4S")
    assert not D.known_demands_corroboration("atmospheric", "HNO2")
    assert D.known_demands_corroboration("cyclosiloxane", "C14H42O7Si7")


def test_known_evidence_records_the_rows_own_13c_line():
    led = pd.DataFrame([
        dict(role="M0", peak_id="p1", mz=182.0093, neutral_formula="C8H7NS2", adduct="[M+H]+",
             method="known:indoor_sulfur", tier="Assigned", ion_score=0.9, admitted_by="height", occurrence=1.0,
             commentary="Pass 0 (known indoor-sulfur): C8H7NS2 [M+H]+ = MTBT, ppm 0.1, ion score 0.9",
             isotopologues=json.dumps([{"label": "13C", "score": 0.9, "peak_id": "c"},
                                       {"label": "34S", "score": 0.9, "peak_id": "s"}]),
             parent_peak_id=pd.NA, iso_label=pd.NA),
        dict(role="M0", peak_id="p2", mz=183.0781, neutral_formula="C6H15O4P", adduct="[M+H]+",
             method="known:organophosphate", tier="Assigned", ion_score=0.9, admitted_by="height",
             occurrence=1.0, commentary="Pass 0 (known organophosphate): C6H15O4P [M+H]+ = TEP, ppm 0.1",
             isotopologues=json.dumps([{"label": "13C+34S", "score": 0.9, "peak_id": "x"}]),
             parent_peak_id=pd.NA, iso_label=pd.NA)])
    recs = {r["neutral"]: r for r in AB.known_evidence(led, src="f")}
    assert recs["C8H7NS2"]["n_c13"] == 1 and recs["C8H7NS2"]["n_satellites"] == 1
    assert recs["C6H15O4P"]["n_c13"] == 0
