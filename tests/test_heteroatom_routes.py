"""Heteroatom routes and exemptions (card p2c): an element enters by its own evidence.

* the element-budget demote keeps a commit whose every over-budget Cl / Br / S
  shows its own 37Cl / 81Br / 34S line (satellites.own_line), on an
  Orbitrap-class run;
* the positive sources (uronium, EasyIC) open two isotope-gated families, both
  claimed last (pass 3's late phase): organochlorine reads only peaks with a
  37Cl partner and commits only on the ion's own 37Cl line; organosulfur_34s
  refuses a reading whose 34S line was testable and is not seen, and commits
  one whose line is untestable -- which the tier engine then holds at
  Candidate, whatever else corroborates it;
* DMMP and TCPP join the positive organophosphate list, behind its unchanged
  corroboration gate;
* the carbon-cluster rule spares a formula whose O / N carry its unsaturation
  (oxalic acid, nitrooxyacetic acid, a pyrimidinone), never an F / Si
  decorated or H-poor bare-carbon fit;
* a reagent profile's `ranges` string is documentation: enumeration never reads it.

Every fixture is synthetic.

Run: pytest tests/test_heteroatom_routes.py -q
"""
from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from peaky import ledger as L
from peaky import passes as P
from peaky.assignment import degeneracy as DG
from peaky.assignment import plausibility as PL
from peaky.assignment import satellites as SAT
from peaky.assignment.passes import directors as D
from peaky.chem import chemistry as C
from peaky.chem import contexts as X
from peaky.chem import isotopes as ISO
from peaky.chem import profiles as PR

ROOT = Path(__file__).resolve().parents[1]
QUIET = dict(log=lambda *a: None)
URO = X.get_context("uronium")
EIC = X.get_context("easyic")
AIR = X.get_context("ambient-air")
ORBI = SimpleNamespace(instrument_class="orbitrap", trace_sample=False, cal_sigma=None)
TOF = SimpleNamespace(instrument_class="tof", trace_sample=False, cal_sigma=None)
R37 = ISO.R_37CL_PER_CL          # ~0.32 per Cl
R34 = ISO.R_34S_PER_S            # ~0.045 per S
R81 = ISO.R_81BR_PER_BR          # ~0.97 per Br


def cf(f):
    return C.parse_formula(f)


def _lines(*pairs):
    mz = np.array([p[0] for p in pairs], dtype=float)
    h = np.array([p[1] for p in pairs], dtype=float)
    o = np.argsort(mz)
    return mz[o], h[o]


# --------------------------------------------------------------------------- the own-line primitive
def test_the_own_line_window_is_a_flat_ppm_on_an_orbitrap_only():
    """The line and its parent shift together, so the window does not widen with the file's
    calibration sigma (the spec's ~1 ppm)."""
    assert SAT.own_line_window_ppm(None) is None
    assert SAT.own_line_window_ppm(TOF) is None
    assert SAT.own_line_window_ppm(SimpleNamespace(instrument_class="orbitrap", trace_sample=True)) is None
    assert SAT.own_line_window_ppm(ORBI) == SAT.OWN_LINE_PPM == 1.0
    for sigma in (0.2, 0.6, 1.5):
        cal = SimpleNamespace(instrument_class="orbitrap", trace_sample=False, cal_sigma=sigma)
        assert SAT.own_line_window_ppm(cal) == SAT.OWN_LINE_PPM


@pytest.mark.parametrize("ratio,ppm,n_ion,ok", [
    (R37, 0.1, 1, True),               # Cl1 at its prediction
    (0.6 * R37, -0.3, 1, True),        # low but in the band
    (2 * R37, 0.0, 1, False),          # a Cl2 line under a Cl1 reading: fits Cl2 as well
    (0.3 * R37, 0.0, 1, False),        # too low
    (R37, 1.6, 1, False),              # outside the 1-ppm window
    (R37, 1.2, 1, False),              # ... however the file is calibrated
    (2 * R37, 0.0, 2, True),           # Cl2 seen
    (R37, 0.0, 2, False),              # a Cl1 line under a Cl2 reading: 0.5x, in the band, fits Cl1
    (0.65 * 2 * R37, 0.0, 2, False),   # ... and a little over it still reads Cl1 better
    (2 * R37, 0.0, 3, False),          # a Cl3 reading under a Cl2 line (one atom fewer)
    (3 * R37, 0.0, 3, True),           # ... and here Cl3
])
def test_own_line_reads_the_count(ratio, ppm, n_ion, ok):
    m0, h0 = 250.0, 1e4
    tgt = m0 + ISO.D_37CL
    mz, h = _lines((m0, h0), (tgt * (1 + ppm * 1e-6), ratio * h0))
    v = SAT.own_line(mz, h, m0, h0, {"C": 9, "Cl": n_ion}, "Cl", window_ppm=SAT.OWN_LINE_PPM)
    assert v["confirmed"] is ok, v["why"]


def test_a_line_is_testable_from_twice_the_floor():
    assert SAT.OWN_LINE_TESTABLE_X == 2.0
    assert SAT.own_line_testable(2 / R34 * 100.0 + 1, {"S": 1}, "S", 100.0)
    assert not SAT.own_line_testable(2 / R34 * 100.0 - 1, {"S": 1}, "S", 100.0)
    assert SAT.own_line_testable(2 / R34 * 100.0 / 2 + 1, {"S": 2}, "S", 100.0)     # two atoms, twice the line
    assert not SAT.own_line_testable(1e9, {"S": 1}, "S", None)                       # no floor: never testable
    assert not SAT.own_line_testable(1e9, {"C": 5}, "S", 100.0)                      # no atom of it


def test_own_line_needs_a_line_and_a_window():
    mz, h = _lines((250.0, 1e4))
    assert not SAT.own_line(mz, h, 250.0, 1e4, {"Cl": 1}, "Cl", window_ppm=1.0)["confirmed"]
    mz, h = _lines((250.0, 1e4), (250.0 + ISO.D_37CL, R37 * 1e4))
    assert not SAT.own_line(mz, h, 250.0, 1e4, {"Cl": 1}, "Cl", window_ppm=None)["confirmed"]
    assert not SAT.own_line(mz, h, 250.0, 1e4, {"P": 1}, "P", window_ppm=1.0)["confirmed"]   # no line exists


# --------------------------------------------------------------------------- the off-budget exemption
def _budget_ledger(neutral, adduct, *, iso=(), mz_shift=0.0, tier="Assigned"):
    """One committed M0 (`neutral` on `adduct`, at its exact m/z + `mz_shift` Da) and its
    picked heavy lines `iso`: [(shift Da, ratio to the parent, ppm off)]."""
    m0 = C.ion_mz(neutral, adduct) + mz_shift
    rows = [dict(peak_id="p0", mz=m0, height=2e4, role="M0", tier=tier, neutral_formula=neutral,
                 adduct=adduct, ion_formula="", method="certified:multi-channel",
                 below_assignability=False, isotopologues="[]", degeneracy_note="", commentary="",
                 confidence="Good (certified)")]
    for k, (d, r, ppm) in enumerate(iso):
        t = m0 + d
        rows.append(dict(peak_id=f"i{k}", mz=t * (1 + ppm * 1e-6), height=r * 2e4, role="unexplained",
                         tier=None, neutral_formula=None, adduct=None, ion_formula=None, method=None,
                         below_assignability=False, isotopologues=None, degeneracy_note=None,
                         commentary=None, confidence=None))
    return pd.DataFrame(rows)


def test_a_chlorine_its_own_37cl_line_shows_keeps_its_tier_on_a_positive_source():
    """The pass-7 shape of a chlorinated thiazolinone-like urea-source ion: Cl1 outside the
    uronium budget (Cl <= 0), its 37Cl line at the predicted 0.32x within 0.1 ppm."""
    led = _budget_ledger("C4H4ClNOS", "[M+H]+", iso=[(ISO.D_37CL, 1.05 * R37, 0.1)])
    audit = []
    out = PL.demote_off_budget(led, context=URO, audit=audit, cfg=ORBI, **QUIET)
    assert out == {"budget_demoted": 0, "budget_cf2_kept": 0, "budget_iso_kept": 1}
    assert led.loc[0, "tier"] == "Assigned" and not audit
    assert "kept outside the uronium element budget (Cl=1 > 0) on its own isotope line" in led.loc[0, "commentary"]
    assert "37Cl line at 1.05x its prediction" in led.loc[0, "commentary"]


@pytest.mark.parametrize("iso", [
    [],                                          # no line
    [(ISO.D_37CL, 0.1, 0.1)],                    # a line far under the prediction
    [(ISO.D_37CL, 2 * R37, 0.1)],                # the line of a Cl2 ion under a Cl1 reading
    [(ISO.D_37CL, R37, 2.5)],                    # a neighbour 2.5 ppm off
    [(ISO.D_34S, R37, 0.0)],                     # a line at the 34S offset is no 37Cl line
])
def test_without_its_own_line_the_chlorine_is_a_lead(iso):
    led = _budget_ledger("C4H4ClNOS", "[M+H]+", iso=iso)
    out = PL.demote_off_budget(led, context=URO, cfg=ORBI, **QUIET)
    assert out["budget_demoted"] == 1 and out["budget_iso_kept"] == 0
    assert led.loc[0, "tier"] == "Candidate" and led.loc[0, "lead_by"] == "off_budget"


def test_no_line_is_read_off_an_orbitrap_class_run():
    for cfg in (None, TOF):
        led = _budget_ledger("C4H4ClNOS", "[M+H]+", iso=[(ISO.D_37CL, R37, 0.0)])
        out = PL.demote_off_budget(led, context=URO, cfg=cfg, **QUIET)
        assert out == {"budget_demoted": 1, "budget_cf2_kept": 0}       # no budget_iso_kept count either
        assert led.loc[0, "tier"] == "Candidate"


def test_every_over_budget_element_needs_its_line():
    # P has no line: a chlorinated phosphate outside the ambient budget stays a lead
    led = _budget_ledger("C6H11ClO4P", "[M-H]-", iso=[(ISO.D_37CL, R37, 0.0)])
    assert PL.demote_off_budget(led, context=AIR, cfg=ORBI, **QUIET)["budget_demoted"] == 1
    # two sulfurs over the ambient cap of one: kept on a 34S line that shows two ...
    led = _budget_ledger("C10H18O6S2", "[M-H]-", iso=[(ISO.D_34S, 2 * R34, 0.2)])
    assert PL.demote_off_budget(led, context=AIR, cfg=ORBI, **QUIET)["budget_iso_kept"] == 1
    # ... not on one that shows the one the budget already admits
    led = _budget_ledger("C10H18O6S2", "[M-H]-", iso=[(ISO.D_34S, R34, 0.2)])
    assert PL.demote_off_budget(led, context=AIR, cfg=ORBI, **QUIET)["budget_demoted"] == 1
    # bromine on a positive source, on its 81Br line
    led = _budget_ledger("C6H11BrO", "[M+H]+", iso=[(ISO.D_81BR, R81, -0.2)])
    assert PL.demote_off_budget(led, context=URO, cfg=ORBI, **QUIET)["budget_iso_kept"] == 1


def test_a_commit_on_a_heavy_isotopologue_is_not_read():
    led = _budget_ledger("C4H4ClNOS", "[M+H]+", iso=[(ISO.D_37CL, R37, 0.0)], mz_shift=ISO.D_37CL)
    assert PL.demote_off_budget(led, context=URO, cfg=ORBI, **QUIET)["budget_demoted"] == 1


def test_the_stage_hands_the_runs_cfg_to_the_demote():
    from peaky.assignment import assign as A
    led = _budget_ledger("C4H4ClNOS", "[M+H]+", iso=[(ISO.D_37CL, R37, 0.0)])
    st = SimpleNamespace(led=led, plaus_audit=[], log=lambda *a: None, profile=URO,
                         cfg=SimpleNamespace(reflist_formulas=frozenset(), instrument_class="orbitrap",
                                             trace_sample=False, cal_sigma=None))
    out = A._stage_plausibility(st)
    assert out["budget_iso_kept"] == 1 and led.loc[0, "tier"] == "Assigned"


# --------------------------------------------------------------------------- the positive families
def test_the_positive_sources_open_the_two_gated_families_last():
    for prof in (URO, EIC):
        assert prof.pass3_families[-2:] == ("organosulfur_34s", "organochlorine")   # after every other
        assert "organosulfur" not in prof.pass3_families
        assert prof.max_Cl == 0 and prof.max_S == 2            # the context budget is unchanged
    fam = X.CONTAMINANT_FAMILIES["organochlorine"]
    assert fam["add"] == {"Cl": (1, 3)} and X.family_iso_gate("organochlorine") == "Cl"
    assert not X.family_iso_gate_untestable("organochlorine")             # its line or nothing
    s34 = X.CONTAMINANT_FAMILIES["organosulfur_34s"]
    assert s34["add"] == {"S": (1, 2)} and s34["adducts"] == ("[M+H]+",)
    assert X.family_iso_gate("organosulfur_34s") == "S" and X.family_iso_gate_untestable("organosulfur_34s")
    # the labelled-ammonium source keeps its ungated family, in its old place
    nh4 = X.get_context("ammonium-15n")
    assert nh4.pass3_families == ("amine", "organosulfur", "siloxane", "pdms", "glycol_peg", "phthalate")
    assert X.family_iso_gate("organosulfur") is None and X.family_iso_gate("nope") is None
    assert not set(AIR.pass3_families) & {"organochlorine", "organosulfur_34s"}


def test_the_gated_families_open_no_element_space():
    """degeneracy / the evidence level: an iso-gated family raises no zero cap; the S family
    adds nothing on a source that already budgets S <= 2."""
    for prof in (URO, EIC):
        assert DG.space_profiles(prof, ("organochlorine", "organosulfur_34s")) == [prof]
    assert len(DG.space_profiles(AIR, ("fluorinated",))) == 2              # an ordinary family still does


def _gate_ledger(rows):
    return L.new_ledger(pd.DataFrame(rows, columns=["peak_id", "mz", "height"]))


CL1 = "C9H11ClO2"                              # a chlorinated aromatic ether-alcohol
M_CL = C.ion_mz(CL1, "[M+H]+")
M_X = 300.0                                    # a peak whose 37Cl-offset partner reads Cl2


def test_the_gate_reads_only_peaks_with_a_37cl_partner():
    led = _gate_ledger([("a", M_CL, 1e4), ("a37", M_CL + ISO.D_37CL, R37 * 1e4),
                        ("b", 250.0, 1e4), ("x", M_X, 1e4), ("x37", M_X + ISO.D_37CL, 2 * R37 * 1e4)])
    tgt = led[led["peak_id"].isin(["a", "b", "x"])]
    g = D._iso_gate_targets(led, tgt, "organochlorine", "Cl", URO, ORBI)
    assert g["pids"] == {"a", "x"} and g["label"] == "37Cl" and not g["untestable_ok"]
    assert g["profile"].max_Cl == 3 and URO.max_Cl == 0
    assert X.filter_by_profile(CL1, g["profile"])[0] and not X.filter_by_profile(CL1, URO)[0]
    assert D._iso_gate_targets(led, tgt, "organochlorine", "Cl", URO, TOF) is None


def _winners(rows):
    base = dict(ion_score=0.9, ion_score_massfree=None, compound_score=0.9, raw_score=0.9, eff_score=0.9,
                eff_margin=0.2, ppm_error=0.1, n_iso=1, tied=False, alternatives=[])
    return pd.DataFrame([dict(base, peak_id=p, neutral=n, adduct=a, ion_formula=i) for p, n, a, i in rows])


def test_pass3_commits_a_gated_winner_only_on_its_own_37cl_line(monkeypatch):
    led = _gate_ledger([("a", M_CL, 1e4), ("a37", M_CL + ISO.D_37CL, R37 * 1e4),
                        ("b", 250.0, 1e4), ("x", M_X, 1e4), ("x37", M_X + ISO.D_37CL, 2 * R37 * 1e4)])
    seen = {}

    def fake_enum(client, mzs, mech_ids, ranges, cfg, adducts, **kw):
        if ranges["Cl"][1] > 0:
            seen["mzs"] = sorted(mzs)
            return {CL1, "C11H17ClO4"}
        return set()

    arb = {"winners": _winners([("a", CL1, "[M+H]+", "C9H12ClO2+"),
                                ("x", "C11H17ClO4", "[M+H]+", "C11H18ClO4+")]),   # Cl1 under a Cl2 line
           "iso_children": pd.DataFrame([{"peak_id": "a37", "parent_peak_id": "a", "iso_label": "37Cl",
                                          "iso_score": 0.9},
                                         {"peak_id": "x37", "parent_peak_id": "x", "iso_label": "37Cl",
                                          "iso_score": 0.9}])}
    monkeypatch.setattr(D, "_enumerate", fake_enum)
    monkeypatch.setattr(D, "_mech_ids_for", lambda client, adducts: [])
    monkeypatch.setattr(D.IO, "score_candidates", lambda *a, **k: pd.DataFrame({"x": [1]}))
    monkeypatch.setattr(D, "arbitrate", lambda scored, cfg, profile, **kw: arb)
    cfg = P.PassConfig(height_cutoff_cps=1.0)
    cfg.instrument_class = "orbitrap"
    D.run_pass3(None, "s", led, URO, None, cfg, ["[M+H]+", "[M+(CH4N2O)H]+", "[M+Na]+", "[M+NH4]+"],
                reagent=None, **QUIET)
    assert seen["mzs"] == sorted([M_CL, M_X])                       # never the partner-less peak
    by = led.set_index("peak_id")
    assert by.at["a", "neutral_formula"] == CL1 and by.at["a", "method"] == "contaminant:organochlorine"
    assert by.at["a37", "role"] == L.ROLE_ISO and by.at["a37", "parent_peak_id"] == "a"
    assert by.at["x", "role"] == L.ROLE_UNEXPLAINED and by.at["x37", "role"] == L.ROLE_UNEXPLAINED


def test_pass3_skips_the_gated_family_off_an_orbitrap(monkeypatch):
    led = _gate_ledger([("a", M_CL, 1e4), ("a37", M_CL + ISO.D_37CL, R37 * 1e4)])
    calls = []
    monkeypatch.setattr(D, "_enumerate", lambda client, mzs, mech_ids, ranges, *a, **k:
                        calls.append(ranges["Cl"]) or set())
    monkeypatch.setattr(D, "_mech_ids_for", lambda client, adducts: [])
    for klass in ("tof", None):
        cfg = P.PassConfig(height_cutoff_cps=1.0)
        cfg.instrument_class = klass
        D.run_pass3(None, "s", led, URO, None, cfg, ["[M+H]+"], reagent=None, **QUIET)
    assert all(hi == 0 for _lo, hi in calls)                          # the other families ran, never Cl


# --------------------------------------------------------------------------- the 34S gate
S1 = "C9H10N2S"                                # a benzothiazole-amine-like thio-heterocycle
S1B = "C7H11NS"
S1C = "C10H12N2S"
FLOOR = 100.0
DIM = 1000.0                                    # 34S predicted at ~44 cps: under 2x the floor
BRIGHT = 2e5                                    # 34S predicted at ~8860 cps: testable (its own line too)


def _s_ledger(with_line: bool):
    md, mb, mc = (C.ion_mz(f, "[M+H]+") for f in (S1, S1B, S1C))
    rows = [("d", md, DIM), ("b", mb, BRIGHT), ("c", mc, BRIGHT)]
    if with_line:
        rows.append(("c34", mc + ISO.D_34S, R34 * BRIGHT))
    return _gate_ledger(rows), (md, mb, mc)


def _s_run(monkeypatch, led, prof=URO, *, phase="all", carried=None, klass="orbitrap"):
    seen = {"S": [], "Cl": []}

    def fake_enum(client, mzs, mech_ids, ranges, cfg, adducts, **kw):
        for el in ("S", "Cl"):
            if ranges[el][1] > 0:
                seen[el].append(sorted(mzs))
        return {S1, S1B, S1C} if ranges["S"][1] > 0 else set()

    arb = {"winners": _winners([("d", S1, "[M+H]+", "C9H11N2S+"), ("b", S1B, "[M+H]+", "C7H12NS+"),
                                ("c", S1C, "[M+H]+", "C10H13N2S+")]),
           "iso_children": pd.DataFrame(columns=["peak_id", "parent_peak_id", "iso_label", "iso_score"])}
    monkeypatch.setattr(D, "_enumerate", fake_enum)
    monkeypatch.setattr(D, "_mech_ids_for", lambda client, adducts: [])
    monkeypatch.setattr(D.IO, "score_candidates", lambda *a, **k: pd.DataFrame({"x": [1]}))
    monkeypatch.setattr(D, "arbitrate", lambda scored, cfg, profile, **kw: arb)
    cfg = P.PassConfig(height_cutoff_cps=FLOOR)
    cfg.instrument_class = klass
    out = D.run_pass3(None, "s", led, prof, None, cfg, ["[M+H]+", "[M+(CH4N2O)H]+", "[M+Na]+", "[M+NH4]+"],
                      reagent=None, phase=phase, carried=carried, **QUIET)
    return seen, out


def test_the_34s_gate_refuses_a_testable_line_that_is_not_there(monkeypatch):
    """A dim S1 reading (its 34S line predicted under 2x the floor) commits; a bright one whose
    34S line is testable and absent is never even enumerated (no partner, no untestable line); a
    bright one with its 34S line at the prediction commits."""
    led, (md, mb, mc) = _s_ledger(with_line=True)
    seen, _ = _s_run(monkeypatch, led)
    assert seen["S"] == [sorted([md, mc])]
    by = led.set_index("peak_id")
    assert by.at["d", "neutral_formula"] == S1 and by.at["d", "method"] == "contaminant:organosulfur_34s"
    assert by.at["c", "neutral_formula"] == S1C
    assert by.at["b", "role"] == L.ROLE_UNEXPLAINED


def test_the_34s_gate_refuses_a_winner_whose_testable_line_is_missing(monkeypatch):
    """The winner check on its own: a bright peak admitted on a partner, whose winner's line
    then does not fit its count, is refused."""
    led, (md, mb, mc) = _s_ledger(with_line=False)
    led = pd.concat([led, _gate_ledger([("c34", mc + ISO.D_34S, 0.3 * R34 * BRIGHT)])], ignore_index=True)
    g = D._iso_gate_targets(led, led[led["peak_id"].isin(["d", "b", "c"])], "organosulfur_34s", "S", URO,
                            SimpleNamespace(instrument_class="orbitrap", trace_sample=False,
                                            height_cutoff=FLOOR))
    assert g["pids"] == {"d"} and g["untestable_ok"] and g["floor"] == FLOOR
    assert g["profile"] is URO                                    # S <= 2 already: nothing lifted
    g["pids"] |= {"b", "c"}                                       # as if each had shown a partner
    arb = {"winners": _winners([("d", S1, "[M+H]+", ""), ("b", S1B, "[M+H]+", ""), ("c", S1C, "[M+H]+", "")]),
           "iso_children": pd.DataFrame({"peak_id": ["c34"], "parent_peak_id": ["c"]})}
    out = D._iso_gate_winners(led, arb, g, log=lambda *a: None)
    assert list(out["winners"]["peak_id"]) == ["d"] and not len(out["iso_children"])


def test_a_winner_the_gate_refuses_is_the_peaks_blocked_reading():
    """A gated family's refusal is an exit: during a run the refused winner is the
    peak's blocked reading, its gate the family's own-line verdict."""
    from peaky.assignment import blocked as BL
    led, (md, mb, mc) = _s_ledger(with_line=False)
    g = D._iso_gate_targets(led, led[led["peak_id"].isin(["d", "b", "c"])], "organosulfur_34s", "S", URO,
                            SimpleNamespace(instrument_class="orbitrap", trace_sample=False,
                                            height_cutoff=FLOOR))
    g["pids"] |= {"b"}
    arb = {"winners": _winners([("d", S1, "[M+H]+", ""), ("b", S1B, "[M+H]+", "")]),
           "iso_children": pd.DataFrame(columns=["peak_id", "parent_peak_id"])}
    BL.start()
    try:
        out = D._iso_gate_winners(led, arb, g, log=lambda *a: None, fam_key="organosulfur_34s")
        BL.stamp(led, log=lambda *a: None)
    finally:
        BL.stop()
    assert list(out["winners"]["peak_id"]) == ["d"]
    by = led.set_index("peak_id")
    assert by.at["b", "blocked_by"].startswith("iso_gate:organosulfur_34s: no 34S line within")
    assert BL.gate_class(by.at["b", "blocked_by"]) == "iso_gate:organosulfur_34s"
    assert by.at["b", "blocked_reading"].startswith(f"{S1B} [M+H]+")


def test_the_gated_families_claim_in_the_late_phase(monkeypatch):
    """Curated phase: no S / Cl reading is enumerated, the families are held; series phase: they
    run with no series evidence at all, after whatever passes 4/5/7 left."""
    led, (md, mb, mc) = _s_ledger(with_line=True)
    seen, out = _s_run(monkeypatch, led, phase="curated")
    assert seen == {"S": [], "Cl": []} and "_carry" in out
    assert (led["role"] == L.ROLE_UNEXPLAINED).all()
    carry = out["_carry"]
    carry["evidence"] = carry["evidence"].iloc[0:0]               # no GKA structure at all
    seen, out = _s_run(monkeypatch, led, phase="series", carried=carry)
    assert seen["S"] == [sorted([md, mc])]
    assert led.set_index("peak_id").at["d", "method"] == "contaminant:organosulfur_34s"


def test_the_ungated_organosulfur_keeps_its_place_and_its_rule(monkeypatch):
    """The labelled-ammonium source: organosulfur claims in the curated phase, every peak, no gate.
    It is still a commit route under the commit contract (satellites.own_line_verdict): the dim
    reading, whose 34S line is untestable, commits as before; the bright one, whose 34S line is
    testable and absent, is refused at the commit."""
    led, (md, mb, mc) = _s_ledger(with_line=False)
    seen, _ = _s_run(monkeypatch, led, X.get_context("ammonium-15n"), phase="curated")
    assert seen["S"] == [sorted([md, mb, mc])]
    by = led.set_index("peak_id")
    assert by.at["d", "method"] == "contaminant:organosulfur"     # no 34S line to test, committed as before
    assert by.at["b", "role"] == L.ROLE_UNEXPLAINED
    assert "REFUSED (own_line: " + S1B in str(by.at["b", "commentary"])


def test_the_gated_families_are_skipped_off_an_orbitrap(monkeypatch):
    led, _ = _s_ledger(with_line=True)
    seen, _ = _s_run(monkeypatch, led, klass="tof")
    assert seen == {"S": [], "Cl": []} and (led["role"] == L.ROLE_UNEXPLAINED).all()


# --------------------------------------------------------------------------- the tier hold
def _tier_led(method, h, *, kid=True, line=False, neutral=S1):
    """The reviewer's shape: one pass-3 [M+H]+ commit at height `h` (floor 1000 cps), with a
    13C child (`kid`) and its 34S line at the prediction (`line`)."""
    m0 = C.ion_mz(neutral, "[M+H]+")
    peaks = [("A", m0, h)]
    if kid:
        peaks.append(("A13", m0 + ISO.D_13C_EXACT, 0.099 * h))
    if line:
        peaks.append(("A34", m0 + ISO.D_34S, R34 * h))
    led = L.new_ledger(pd.DataFrame(peaks, columns=["peak_id", "mz", "height"]))
    L.commit_assignment(led, "A", neutral_formula=neutral, adduct="[M+H]+",
                        ion_formula="",
                        ion_score=0.93, compound_score=0.93, eff_score=0.9, eff_margin=0.3, tied=False,
                        ppm_error=0.2, pass_no=3, method=method, confidence="Good (organosulfur)",
                        commentary="Pass 3",
                        isotopologues=([{"label": "13C", "score": 0.9, "peak_id": "A13"}] if kid else []))
    if kid:
        L.attach_isotopologue(led, "A13", "A", iso_label="13C", iso_match_score=0.9)
    return led


def _tiers(led, klass="orbitrap"):
    from peaky.assignment import tiers as T
    cfg = P.PassConfig(height_cutoff_cps=1000.0, instrument_type="orbi")
    cfg.instrument_class = klass
    return T.compute_tiers(led, cfg=cfg, profile=URO).set_index("peak_id").loc["A"]


def test_an_untestable_s_from_the_gated_route_stays_candidate_whatever_corroborates_it():
    """34S predicted at 221 cps under a 1000-cps floor, a 13C child: before, Assigned
    ('S count untested'); the 13C child pins the carbons, not the sulfur."""
    from peaky.assignment import tiers as T
    t = _tiers(_tier_led("contaminant:organosulfur_34s", 5000.0))
    assert t["tier"] == T.TIER_CANDIDATE and t["tier_reason"].startswith(T.GATED_HOLD), t["tier_reason"]
    assert "S1 proposed by the isotope-gated organosulfur_34s family" in t["tier_reason"]
    # a re-arbitrated row of the route is held the same way ...
    t = _tiers(_tier_led("rearb<-contaminant:organosulfur_34s", 5000.0))
    assert t["tier"] == T.TIER_CANDIDATE
    # ... but not once it carries no sulfur at all
    t = _tiers(_tier_led("rearb<-contaminant:organosulfur_34s", 5000.0, neutral="C10H14N2O"))
    assert not str(t["tier_reason"]).startswith(T.GATED_HOLD)
    # the ungated family (the labelled-ammonium source) is read as before
    t = _tiers(_tier_led("contaminant:organosulfur", 5000.0))
    assert t["tier"] == T.TIER_ASSIGNED and "S count untested" in t["tier_reason"]


def test_a_gated_s_on_its_own_line_is_assigned_and_says_so():
    from peaky.assignment import tiers as T
    t = _tiers(_tier_led("contaminant:organosulfur_34s", 1e5, line=True))
    assert t["tier"] == T.TIER_ASSIGNED, t["tier_reason"]
    assert "S1 on its own line (isotope-gated organosulfur_34s: 34S line at" in t["tier_reason"]
    # off an Orbitrap-class run nothing is seen
    t = _tiers(_tier_led("contaminant:organosulfur_34s", 1e5, line=True), klass="tof")
    assert t["tier"] == T.TIER_CANDIDATE and t["tier_reason"].startswith(T.GATED_HOLD)


# --------------------------------------------------------------------------- DMMP / TCPP
def _row(compound, ion, label, pid, mz, *, base, mech, score=0.9):
    return dict(compound_formula=compound, compound_score=score, compound_category=2, ion_formula=ion,
                ion_score=score, ion_category=2, mechanism_id=mech, isotope_formula=ion, iso_label=label,
                is_base=base, theo_mz=mz, rel_abundance=1.0, iso_score=score, iso_category=2,
                sample_peak_id=pid, sample_peak_mz=mz, sample_peak_intensity=1e4, ppm_error=0.3,
                abundance_error=0.0)


DMMP, TCPP = "C3H9O3P", "C9H18Cl3O4P"


def test_dmmp_and_tcpp_are_positive_organophosphates():
    ops = D._known_species("positive", "uronium")["organophosphate"]
    assert DMMP in ops and TCPP in ops
    assert {DMMP, TCPP} <= P.known_formulas("positive", "uronium")
    assert DMMP not in P.known_formulas("negative", "ambient-air")


def _pass0(rows, scored):
    led = L.new_ledger(pd.DataFrame(rows, columns=["peak_id", "mz", "height"]))
    P.run_pass0_known(None, "SID", led, URO, P.PassConfig(height_cutoff_cps=100.0),
                      ["[M+H]+", "[M+(CH4N2O)H]+"], score_fn=lambda *a, **k: scored, **QUIET)
    return led


def test_tcpp_commits_on_one_channel_with_its_37cl_line():
    mh = C.ion_mz(TCPP, "[M+H]+")
    scored = pd.DataFrame([_row(TCPP, "C9H19Cl3O4P+", "M0", "t", mh, base=True, mech="mH"),
                           _row(TCPP, "C9H19Cl3O4P+", "37Cl", "t37", mh + ISO.D_37CL, base=False, mech="mH")])
    led = _pass0([("t", mh, 3e4), ("t37", mh + ISO.D_37CL, 2.9e4)], scored)
    assert L.role_of(led, "t") == L.ROLE_M0
    assert led.loc[led.peak_id == "t", "method"].iloc[0] == "known:organophosphate"
    assert L.role_of(led, "t37") == L.ROLE_ISO


def test_dmmp_needs_its_second_channel():
    mh, mu = C.ion_mz(DMMP, "[M+H]+"), C.ion_mz(DMMP, "[M+(CH4N2O)H]+")
    one = pd.DataFrame([_row(DMMP, "C3H10O3P+", "M0", "d", mh, base=True, mech="mH")])
    led = _pass0([("d", mh, 1e4), ("u", mu, 8e3)], one)
    assert L.role_of(led, "d") != L.ROLE_M0                         # monoisotopic P: one channel is no evidence
    two = pd.concat([one, pd.DataFrame([_row(DMMP, "C4H14N2O4P+", "M0", "u", mu, base=True, mech="mU")])])
    led = _pass0([("d", mh, 1e4), ("u", mu, 8e3)], two)
    assert L.role_of(led, "d") == L.ROLE_M0 and L.role_of(led, "u") == L.ROLE_M0


# --------------------------------------------------------------------------- the carbon-cluster rule
@pytest.mark.parametrize("f", [
    "C2H2O4",        # oxalic acid
    "C2H2O2",        # glyoxal
    "C2H3NO5",       # nitrooxyacetic acid
    "C5H2O5",        # croconic acid
    "C4H3BrN2O2",    # a bromouracil
    "C4H4N2O",       # a pyrimidinone
    "C6H4N2O5",      # dinitrophenol
    "C4H3NO2",       # maleimide
])
def test_heteroatom_unsaturation_is_no_carbon_cluster(f):
    assert C.dbe(cf(f)) / cf(f)["C"] >= PL.DBE_PER_C_MONSTER       # DBE/C >= 1: the old rule's target
    assert PL.hetero_unsaturated(cf(f)) and not PL.is_carbon_cluster(cf(f))
    assert "carbon cluster" not in str(PL.implausible(f, tier="Candidate"))


@pytest.mark.parametrize("f", [
    "C24H2", "C5H2", "C2H2",       # bare carbon
    "C12HF", "C15HFO2", "C14HFO4", "C10HF4N3O3",   # F-decorated fits stay caught, O / N or not
    "C7H13N5O2Si3",               # a Si-N fit (DBE reads Si as a carbon)
    "C4H2O", "C5H2O",             # (O+N)/C under 0.3
    "C2O", "C2N2O",               # no hydrogen skeleton
    "C10H3NO2",                   # H/C 0.3: an H-poor nitro fit
    "C4H4N2", "C2H3N3",           # ring N is not credited (O-free azines read as before)
])
def test_bare_carbon_fits_stay_carbon_clusters(f):
    assert PL.is_carbon_cluster(cf(f)) and not PL.hetero_unsaturated(cf(f))


def test_skeleton_dbe_credits_the_carbonyls_and_nitro_groups_its_oxygen_can_form():
    assert PL.skeleton_dbe(cf("C2H2O4")) == 0          # two C=O
    assert PL.skeleton_dbe(cf("C2H3NO5")) == -1        # N=O + C=O (+ one more the O allows): a bound
    assert PL.skeleton_dbe(cf("C6H4N2O5")) == 1        # five carbonyl-or-nitro O on six carbons
    assert PL.skeleton_dbe(cf("C2O")) == 2             # one C=O on C2O leaves DBE/C 1.0
    assert PL.skeleton_dbe(cf("C2H3^NO5")) == PL.skeleton_dbe(cf("C2H3NO5"))   # labels fold


def test_the_demote_spares_oxalic_acid_and_still_takes_c24h2():
    led = pd.DataFrame([dict(role="M0", mz=m, neutral_formula=f, tier="Assigned", commentary="",
                             below_assignability=False, isotopologues="[]")
                        for f, m in (("C2H2O4", 152.98), ("C2H3NO5", 183.99), ("C24H2", 290.0))])
    assert PL.demote_carbon_clusters(led, **QUIET) == {"c_cluster_demoted": 1}
    assert list(led["tier"]) == ["Assigned", "Assigned", "Candidate"]


# --------------------------------------------------------------------------- the profile `ranges` string
def test_enumeration_never_reads_a_reagent_profiles_ranges():
    """`ReagentProfile.ranges` (e.g. '... S0-2 Cl0-2 Br0-2') is documentation: the grid box is
    `build_ranges` from the context profile, CHO(N) only, and the heteroatoms enter through the
    pass-3 families and the curated lists. Pinned so the field cannot quietly start (or seem to
    start) driving the search."""
    for name in ("Br", "NO3", "NO3_15N", "I", "Ur", "EasyIC", "NH4_15N"):
        try:
            prof = PR.resolve(name)
        except Exception:      # noqa: BLE001 -- a profile this build does not register
            continue
        box = D.build_ranges(X.get_context(prof.context), None, include_N=True)
        assert all(box[el] == (0, 0) for el in ("S", "P", "Si", "F", "Cl", "Br", "I")), (name, box)
    # the only reader of the field is the profile module itself (composing a merged profile, and
    # its config loader's field list): an attribute read, a getattr(..., "ranges") or a
    # ["ranges"] / .get("ranges") read anywhere else would be a consumer
    readers = set()

    def _is_ranges(node):
        return isinstance(node, ast.Constant) and node.value == "ranges"

    for py in (ROOT / "peaky").rglob("*.py"):
        for node in ast.walk(ast.parse(py.read_text())):
            hit = (isinstance(node, ast.Attribute) and node.attr == "ranges" and isinstance(node.ctx, ast.Load)
                   and not (isinstance(node.value, ast.Name) and node.value.id == "self"))
            hit |= (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id in ("getattr", "hasattr") and len(node.args) >= 2 and _is_ranges(node.args[1]))
            hit |= isinstance(node, ast.Subscript) and _is_ranges(node.slice)
            hit |= (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr in ("get", "pop") and bool(node.args) and _is_ranges(node.args[0]))
            if hit:
                readers.add(py.relative_to(ROOT).as_posix())
    assert readers == {"peaky/chem/profiles.py"}, readers
