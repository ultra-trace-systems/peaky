"""Offline tests for passes.py arbitration + commit. Run: python3 tests/test_passes.py"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from peaky import passes as P  # noqa: E402
from peaky import ledger as L  # noqa: E402

PASS = FAIL = 0
CFG = P.PassConfig()


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok  {name}")
    else:
        FAIL += 1
        print(f"FAIL  {name}  {detail}")


def iso_row(**kw):
    base = dict(compound_formula=None, compound_score=None, compound_category=2,
                ion_formula=None, ion_score=None, ion_category=2, mechanism_id="m",
                isotope_formula=None, iso_label="M0", is_base=True, theo_mz=100.0,
                rel_abundance=1.0, iso_score=None, iso_category=2,
                sample_peak_id=None, sample_peak_mz=100.0, sample_peak_intensity=1e4,
                ppm_error=0.2, abundance_error=0.0)
    base.update(kw)
    return base


# ---------- confidence_label ----------
check("High needs score>=0.90 + iso + untied",
      P.confidence_label(0.97, -0.3, 1, False, CFG) == "High")
check("no iso -> not High (Good)",
      P.confidence_label(0.97, -0.3, 0, False, CFG) == "Good")
check("tied -> not High",
      P.confidence_label(0.97, -0.3, 1, True, CFG) == "Good")
check("0.82 -> Good", P.confidence_label(0.82, 0.5, 0, False, CFG) == "Good")
check("0.74 -> Low", P.confidence_label(0.74, 0.5, 0, False, CFG) == "Low")
check("0.55 -> Suspect", P.confidence_label(0.55, 0.5, 0, False, CFG) == "Suspect")
check("0.40 -> Reject", P.confidence_label(0.40, 0.5, 0, False, CFG) == "Reject")
check("suffix applied", P.confidence_label(0.75, 0.5, 0, False, CFG, "series") == "Low (series)")

# ---------- _mech_to_adduct: 15N nitrate label from the server '^N' marker ----
check("_mech_to_adduct 14N nitrate -> [M+NO3]-",
      P._mech_to_adduct({"ion_formula": "C6H5N2O6-", "compound_formula": "C6H5NO3"})
      == "[M+NO3]-")
# EVERY channel of EVERY registered profile must round-trip through
# _mech_to_adduct. A channel missing from _DIFF_TO_ADDUCT does not raise -- it hits
# the `.get(diff, "[M-H]-")` default and is silently written to the ledger as a
# DEPROTONATION (found on the first iodide batch: [M+I2]- commits reported as
# [M-H]-, which also inflated the [M-H]- census). This loop is the guard.
from peaky import profiles as _PRF          # noqa: E402
from peaky import chemistry as _CH          # noqa: E402
_ADDUCT_DELTA = {          # ion composition minus neutral composition
    "[M-H]-": {"H": -1}, "[M+Br]-": {"Br": 1}, "[M+Cl]-": {"Cl": 1},
    "[M+I]-": {"I": 1}, "[M+I2]-": {"I": 2}, "[M+I3]-": {"I": 3},
    "[M+NO3]-": {"N": 1, "O": 3}, "[M+Br2]-": {"Br": 2}, "[M+Br3]-": {"Br": 3},
    "[M+HBr+Br]-": {"H": 1, "Br": 2}, "[M+HBr+Br2]-": {"H": 1, "Br": 3},
    "[M+HBr+CO3]-": {"H": 1, "Br": 1, "C": 1, "O": 3},
    "[M-H+I2]-": {"H": -1, "I": 2},   # deprotonated-acid . I2 (mixed-sign diff)
    "[M+CO3]-": {"C": 1, "O": 3}, "[M+H]+": {"H": 1}, "[M+NH4]+": {"N": 1, "H": 4},
    "[M+Na]+": {"Na": 1}, "[M+(CH4N2O)H]+": {"C": 1, "H": 5, "N": 2, "O": 1},
}
for _p in _PRF.PROFILES.values():
    for _a in _p.adducts:
        if _a == "[M+^NO3]-":
            continue                        # heavy-isotope label, covered above
        _d = _ADDUCT_DELTA.get(_a)
        check(f"{_p.name}: channel {_a} has a known element delta in the test map",
              _d is not None, _a)
        if _d is None:
            continue
        _base = {"C": 6, "H": 8, "O": 4}    # an arbitrary neutral to build an ion on
        _ion = dict(_base)
        for _el, _n in _d.items():
            _ion[_el] = _ion.get(_el, 0) + _n
        _got = P._mech_to_adduct({
            "ion_formula": _CH.format_formula(_ion),
            "compound_formula": _CH.format_formula(_base)})
        check(f"{_p.name}: {_a} round-trips through _mech_to_adduct (not silently [M-H]-)",
              _got == _a, f"got {_got}")

# OFF-GRID ELEMENT INVARIANT: covalent iodine must not reach a neutral through the
# generic passes. Enforced by the CONTEXT cap (ambient-air max_I=0, exactly like
# max_F/max_P=0) -- 127I is monoisotopic so it can never be isotope-confirmed, and in
# an iodide source every such formula is degenerate with an [acid-H+I2]- cluster.
# Real regression: the first iodide batch produced CHIO2, CHIO3, C2H3IO2, C2H3IO3,
# INO4 from series extrapolation off the pass-0 iodine anchors.
for _f in ("CHIO2", "CHIO3", "C2H3IO2", "C2H3IO3", "INO4"):
    check(f"off-grid: {_f} rejected by ambient-air (max_I=0)",
          P._context_filter([_f], "ambient-air") == [], _f)
check("off-grid: ordinary CHO/CHON neutrals still pass ambient-air",
      set(P._context_filter(["C6H8O4", "C10H16O3", "C5H7NO4"], "ambient-air"))
      == {"C6H8O4", "C10H16O3", "C5H7NO4"})
check("off-grid: the water context still ALLOWS iodine (iodinated DBPs are the "
      "analyte there)", P._context_filter(["C2H3IO2"], "water") == ["C2H3IO2"])

check("_mech_to_adduct 15N nitrate (^N) -> [M+^NO3]-",
      P._mech_to_adduct({"ion_formula": "C6H5NO6^N-", "compound_formula": "C6H5NO3"})
      == "[M+^NO3]-")
check("_mech_to_adduct deprotonation unaffected",
      P._mech_to_adduct({"ion_formula": "C5H7O5-", "compound_formula": "C5H8O5"})
      == "[M-H]-")
check("_mech_to_adduct does not add ^ when no marker",
      P._mech_to_adduct({"ion_formula": "C6H5N2O6-", "compound_formula": "C6H5NO3"})
      != "[M+^NO3]-")

# ---------- arbitration: CHO beats CHON on a peak ----------
scored = pd.DataFrame([
    # peak A: CHO strong vs CHON weaker
    iso_row(sample_peak_id="A", compound_formula="C10H16O4", compound_score=0.96,
            ion_formula="C10H15O4-", ion_score=0.96, ppm_error=-0.3),
    iso_row(sample_peak_id="A", compound_formula="C9H13NO5", compound_score=0.90,
            ion_formula="C9H12NO5-", ion_score=0.90, ppm_error=0.8),
    # peak A 13C child attributed to peak B (non-base)
    iso_row(sample_peak_id="B", compound_formula="C10H16O4", compound_score=0.96,
            ion_formula="C10H15O4-", ion_score=0.96, isotope_formula="[13C]C9H15O4-",
            iso_label="13C", is_base=False, iso_score=0.93, ppm_error=0.1),
])
arb = P.arbitrate(scored, CFG)
win = arb["winners"]
wa = win[win.peak_id == "A"].iloc[0]
check("peak A winner is CHO C10H16O4", wa["neutral"] == "C10H16O4", wa["neutral"])
check("peak A n_iso counts the 13C child", wa["n_iso"] == 1, wa["n_iso"])
check("peak A has CHON as alternative",
      any(a["formula"] == "C9H13NO5" for a in wa["alternatives"]), wa["alternatives"])
kids = arb["iso_children"]
check("13C child B attributed to parent A",
      len(kids) == 1 and kids.iloc[0]["peak_id"] == "B"
      and kids.iloc[0]["parent_peak_id"] == "A", kids.to_dict("records"))

# ---------- complexity penalty: CHO wins over slightly-higher neutral-Br ----------
scored2 = pd.DataFrame([
    iso_row(sample_peak_id="C", compound_formula="C7H12BrO4", compound_score=0.93,
            ion_formula="C7H12BrO4-", ion_score=0.93, ppm_error=0.4),   # neutral Br
    iso_row(sample_peak_id="C", compound_formula="C7H12O4", compound_score=0.88,
            ion_formula="C7H11O4-", ion_score=0.88, ppm_error=0.6),     # CHO, [M-H]-
])
arb2 = P.arbitrate(scored2, CFG)
wc = arb2["winners"].iloc[0]
check("neutral Br loses to CHO after complexity penalty (0.93-0.20 < 0.88)",
      wc["neutral"] == "C7H12O4", (wc["neutral"], wc["eff_score"]))

# ---------- isotopologue-gated heteroatoms ----------
# peak D: an organosulfate (S) slightly outscores a CHON alt, but has NO 34S
# evidence -> should lose after the het-iso penalty.
scored3 = pd.DataFrame([
    iso_row(sample_peak_id="D", compound_formula="C8H12O6S", compound_score=0.90,
            ion_formula="C8H11O6S-", ion_score=0.90, ppm_error=0.3),
    iso_row(sample_peak_id="D", compound_formula="C9H15NO5", compound_score=0.86,
            ion_formula="C9H14NO5-", ion_score=0.86, ppm_error=0.5),
])
arb3 = P.arbitrate(scored3, CFG)
wd = arb3["winners"].iloc[0]
check("S candidate without 34S loses to CHON (iso gate)",
      wd["neutral"] == "C9H15NO5", (wd["neutral"], wd["eff_score"]))

# peak E: same S candidate but WITH a confirmed 34S satellite -> should win.
scored4 = pd.DataFrame([
    iso_row(sample_peak_id="E", compound_formula="C8H12O6S", compound_score=0.90,
            ion_formula="C8H11O6S-", ion_score=0.90, ppm_error=0.3),
    iso_row(sample_peak_id="F", compound_formula="C8H12O6S", compound_score=0.90,
            ion_formula="C8H11O6S-", ion_score=0.90, isotope_formula="[34S]C8H11O6-",
            iso_label="34S", is_base=False, iso_score=0.88, ppm_error=0.2),
    iso_row(sample_peak_id="E", compound_formula="C9H15NO5", compound_score=0.86,
            ion_formula="C9H14NO5-", ion_score=0.86, ppm_error=0.5),
])
arb4 = P.arbitrate(scored4, CFG)
we = arb4["winners"][arb4["winners"].peak_id == "E"].iloc[0]
check("S candidate WITH 34S confirmed wins",
      we["neutral"] == "C8H12O6S", (we["neutral"], we["eff_score"]))

# ---------- reagent-element alias: adduct reading beats covalent ----------
# In Br-CIMS the ion C6H10BrO3- is equally covalent C6H11BrO3 [M-H]- or
# C6H10O3 [M+Br]-. Both carry 81Br confirmation; with reagent_element="Br" the
# covalent reading keeps its complexity prior and must lose the tie.
cfg_br = P.PassConfig()
cfg_br.reagent_element = "Br"
scored5 = pd.DataFrame([
    iso_row(sample_peak_id="G", compound_formula="C6H11BrO3", compound_score=0.90,
            ion_formula="C6H10BrO3-", ion_score=0.90, ppm_error=0.3),
    iso_row(sample_peak_id="G", compound_formula="C6H10O3", compound_score=0.90,
            ion_formula="C6H10BrO3-", ion_score=0.90, ppm_error=0.3),
    # 81Br satellite confirmed for BOTH (same ion) — attach to the covalent one
    iso_row(sample_peak_id="H", compound_formula="C6H11BrO3", compound_score=0.90,
            ion_formula="C6H10BrO3-", ion_score=0.90, isotope_formula="[81Br]C6H10O3-",
            iso_label="81Br", is_base=False, iso_score=0.95, ppm_error=0.1),
    iso_row(sample_peak_id="H", compound_formula="C6H10O3", compound_score=0.90,
            ion_formula="C6H10BrO3-", ion_score=0.90, isotope_formula="[81Br]C6H10O3-",
            iso_label="81Br", is_base=False, iso_score=0.95, ppm_error=0.1),
])
arb5 = P.arbitrate(scored5, cfg_br)
wg = arb5["winners"][arb5["winners"].peak_id == "G"].iloc[0]
check("adduct reading C6H10O3 [M+Br]- beats covalent C6H11BrO3 (reagent prior)",
      wg["neutral"] == "C6H10O3", (wg["neutral"], wg["eff_score"]))

# HBr cluster unit arithmetic
from peaky import series_gka as G2  # noqa: E402
check("Y + HBr = covalent alias composition",
      G2.formula_add("C6H10O3", "HBr", 1) == "C6H11BrO3",
      G2.formula_add("C6H10O3", "HBr", 1))
check("alias - HBr recovers anchor",
      G2.formula_add("C6H11BrO3", "HBr", -1) == "C6H10O3")

# ---------- alias guard extends to the CO3 background channel (426.976 case) ----------
w_co3 = {"neutral": "C11H14BrNO8", "adduct": "[M+CO3]-",
         "ion_formula": "C12H14BrNO11-"}
r_co3 = P._prefer_adduct_reading(w_co3, cfg_br)
check("covalent bromo + CO3 relabeled to HBr-cluster of a Br-free neutral",
      r_co3["neutral"] == "C11H13NO8" and r_co3["adduct"] == "[M+HBr+CO3]-",
      (r_co3["neutral"], r_co3["adduct"]))
check("CO3 relabel carries the reagent-rule note", "_relabel_note" in r_co3)
# but only when the cluster channel mass is registered: O2 has no HBr-cluster shift
w_o2 = {"neutral": "C11H14BrNO8", "adduct": "[M+O2]-", "ion_formula": "x"}
r_o2 = P._prefer_adduct_reading(w_o2, cfg_br)
check("unmodelled cluster channel keeps the covalent reading",
      r_o2["neutral"] == "C11H14BrNO8" and r_o2["adduct"] == "[M+O2]-",
      (r_o2["neutral"], r_o2["adduct"]))
# no reagent element set -> never relabel
check("no relabel without a reagent element",
      P._prefer_adduct_reading(dict(w_co3), P.PassConfig())["adduct"] == "[M+CO3]-")

# ---------- iodide: covalent-I winner decomposes to deprotonated-acid + I2 ----------
# [A-H+I2]- is the SAME ion as covalent (A-H+I) [M+I]- -- the reading the series
# passes used to invent (CHIO2 @298.807 et al.). The relabel reports the acid.
cfg_i = P.PassConfig(reagent_element="I")
for cov, acid in [("CHIO2", "CH2O2"), ("INO4", "HNO4"), ("C2H3IO3", "C2H4O3")]:
    r_i = P._prefer_adduct_reading({"neutral": cov, "adduct": "[M+I]-",
                                    "ion_formula": "x"}, cfg_i)
    check(f"iodide acid rule: {cov} [M+I]- -> {acid} [M-H+I2]-",
          r_i["neutral"] == acid and r_i["adduct"] == "[M-H+I2]-",
          (r_i["neutral"], r_i["adduct"]))
# ion mass identity of the decomposition (exactly degenerate readings)
check("[HCOOH-H+I2]- is the covalent CHIO2 [M+I]- ion, exactly",
      abs(_CH.ion_mz("CH2O2", "[M-H+I2]-") - _CH.ion_mz("CHIO2", "[M+I]-")) < 1e-9)
# the pass-0 reactive-iodine species must NEVER be re-read (no C/N/S oxy-acid
# behind them: HOI -> "H2O", HIO3 -> "H2O3"; ICl/IBr/ICN have no O) -- their
# I2X- lines were ruled ambient analytes on time behaviour (HOI2-/I2NO2-).
for keep in ("HOI", "HIO2", "HIO3", "ICl", "IBr", "CNI", "INO2", "INO3",
             "CINO", "CH3I"):
    r_k = P._prefer_adduct_reading({"neutral": keep, "adduct": "[M+I]-",
                                    "ion_formula": "x"}, cfg_i)
    check(f"iodide relabel keeps {keep} covalent (not an oxy-acid cluster)",
          r_k["neutral"] == keep and r_k["adduct"] == "[M+I]-",
          (r_k["neutral"], r_k["adduct"]))
# [M-H]- winners fall through to the generic HI-subtraction (CIO2- == CO2.I-)
r_dep = P._prefer_adduct_reading({"neutral": "CHIO2", "adduct": "[M-H]-",
                                  "ion_formula": "x"}, cfg_i)
check("iodide [M-H]- covalent falls through to the HI-subtraction rule",
      r_dep["neutral"] == "CO2" and r_dep["adduct"] == "[M+I]-",
      (r_dep["neutral"], r_dep["adduct"]))

# ---------- commit into ledger end-to-end ----------
peaks = pd.DataFrame({"peak_id": ["A", "B", "C"], "mz": [200.1, 201.1, 191.0],
                      "height": [1e5, 1e4, 8e4]})
led = L.new_ledger(peaks)
summary = P.commit_winners(led, arb, pass_no=1, method="cheminfo",
                           context="ambient-air", cfg=CFG, lock=True,
                           min_raw_score=0.70)
check("commit reports >=1 committed", summary["committed"] >= 1, summary)
check("peak A committed as M0 CHO",
      L.role_of(led, "A") == L.ROLE_M0
      and led.loc[led.peak_id == "A", "neutral_formula"].iloc[0] == "C10H16O4")
check("peak A is High and locked",
      led.loc[led.peak_id == "A", "confidence"].iloc[0] == "High"
      and L.is_locked(led, "A"))
check("peak B attached as iso_child of A",
      L.role_of(led, "B") == L.ROLE_ISO
      and led.loc[led.peak_id == "B", "parent_peak_id"].iloc[0] == "A")
check("commentary written", "C10H16O4" in led.loc[led.peak_id == "A", "commentary"].iloc[0])
check("ledger validates after commit", L.validate(led) == [], L.validate(led))

# ---------- claim_unexplained_only: families cannot displace assignments ----------
peaks_c = pd.DataFrame({"peak_id": ["X1"], "mz": [200.1], "height": [1e4]})
led_c = L.new_ledger(peaks_c)
L.commit_assignment(led_c, "X1", neutral_formula="C10H16O4", adduct="[M-H]-",
                    ion_score=0.80, pass_no=1, method="cheminfo", confidence="Good",
                    commentary="backbone, unlocked")
scored_c = pd.DataFrame([
    iso_row(sample_peak_id="X1", compound_formula="C7H8F4O2", compound_score=0.95,
            ion_formula="C7H7F4O2-", ion_score=0.95, ppm_error=0.2)])
arb_c = P.arbitrate(scored_c, CFG)
s_c = P.commit_winners(led_c, arb_c, pass_no=3, method="contaminant:fluorinated",
                       context="ambient-air", cfg=CFG, lock=False,
                       min_raw_score=0.5, confidence_suffix="fluorinated",
                       claim_unexplained_only=True)
check("family cannot displace existing M0 (claim_unexplained_only)",
      s_c["committed"] == 0
      and led_c.loc[led_c.peak_id == "X1", "neutral_formula"].iloc[0] == "C10H16O4",
      (s_c, led_c.neutral_formula.iloc[0]))

# without the guard, the higher score would displace (documents the contrast)
s_c2 = P.commit_winners(led_c, arb_c, pass_no=3, method="contaminant:fluorinated",
                        context="ambient-air", cfg=CFG, lock=False,
                        min_raw_score=0.5, confidence_suffix="fluorinated")
check("without guard the F formula would displace (contrast)",
      s_c2["committed"] == 1, s_c2)

# ---------- only_peaks restriction ----------
led_o = L.new_ledger(pd.DataFrame({"peak_id": ["P1", "P2"], "mz": [200.1, 300.2],
                                   "height": [1e4, 1e4]}))
scored_o = pd.DataFrame([
    iso_row(sample_peak_id="P1", compound_formula="C9H8F10O", compound_score=0.9,
            ion_formula="C9H8BrF10O-", ion_score=0.9, ppm_error=0.3),
    iso_row(sample_peak_id="P2", compound_formula="C10H8F12O", compound_score=0.9,
            ion_formula="C10H8BrF12O-", ion_score=0.9, ppm_error=0.3)])
arb_o = P.arbitrate(scored_o, CFG)
s_o = P.commit_winners(led_o, arb_o, pass_no=3, method="contaminant:fluorinated",
                       context="ambient-air", cfg=CFG, lock=False,
                       min_raw_score=0.5, confidence_suffix="fluorinated",
                       claim_unexplained_only=True, only_peaks={"P1"})
check("only_peaks: P1 committed, P2 skipped",
      s_o["committed"] == 1 and L.role_of(led_o, "P2") == L.ROLE_UNEXPLAINED, s_o)

# ---------- _family_ok ----------
fr = {"C": (0, 20), "H": (0, 44), "F": (1, 17), "O": (0, 6), "N": (0, 0),
      "S": (0, 0), "P": (0, 0), "Si": (0, 0), "Cl": (0, 0), "Br": (0, 0), "I": (0, 0)}
check("_family_ok accepts in-range fluorinated", P._family_ok("C10H8F14", fr))
check("_family_ok rejects element beyond family range", not P._family_ok("C5H5F3N2O2", fr))
check("_family_ok rejects half-integer DBE", not P._family_ok("C8H14NO12", fr))

# ---------- ranges helper ----------
from peaky import contexts as X  # noqa: E402
from peaky import isotopes as ISO  # noqa: E402
pre = ISO.PrescanResult(has_Br=True, estimated_max_C=12)
# Pass1/2: CHO(N) only -- heteroatoms NEVER auto-added from prescan
r = P.build_ranges(X.get_context("ambient-air"), pre, include_N=True)
s = P.ranges_to_string(r)
check("CHON ranges include N", "N" in s, s)
check("CHON ranges exclude Br/Cl/S/Si (no auto-add)",
      all(x not in s for x in ("Br", "Cl", "Si")) and " S" not in (" " + s), s)
check("ranges cap C near estimate", r["C"][1] <= 16, r["C"])
# Pass3: heteroatoms enter only via extra_elements
r3 = P.build_ranges(X.get_context("ambient-air"), pre, include_N=True,
                    extra_elements={"S": (1, 1), "O": (3, 6)})
check("extra_elements adds S in pass3", r3["S"] == (1, 1), r3["S"])

# ---------- _mech_to_adduct: label derived from exact ion-neutral diff ----------
def adduct_of(comp, ion):
    return P._mech_to_adduct({"compound_formula": comp, "ion_formula": ion})

check("adduct: deprotonation", adduct_of("C5H10O", "C5H9O-") == "[M-H]-")
check("adduct: Br attachment", adduct_of("C10H16O4", "C10H16BrO4-") == "[M+Br]-")
check("adduct: carbonate (v13 mislabel regression)",
      adduct_of("C5H10O", "C6H10O4-") == "[M+CO3]-",
      adduct_of("C5H10O", "C6H10O4-"))
check("adduct: carbonate on N/Br neutral",
      adduct_of("C11H14BrNO8", "C12H14BrNO11-") == "[M+CO3]-",
      adduct_of("C11H14BrNO8", "C12H14BrNO11-"))
check("adduct: superoxide", adduct_of("C5H10O", "C5H10O3-") == "[M+O2]-")
check("adduct: nitrate", adduct_of("C5H10O", "C5H10NO4-") == "[M+NO3]-")
check("adduct: electron attachment", adduct_of("C5H10O", "C5H10O-") == "[M]-.")
check("adduct: unknown diff falls back to [M-H]-",
      adduct_of("C5H10O", "C5H12NaO5-") == "[M-H]-")

# ---------- calibrate + z_of ----------
cal_led = L.new_ledger(pd.DataFrame({
    "peak_id": [f"c{i}" for i in range(30)] + ["x1"],
    "mz": [100.0 + i for i in range(30)] + [400.0],
    "height": [1e4] * 31}))
for i in range(30):
    L.commit_assignment(cal_led, f"c{i}", neutral_formula="C5H10O3",
                        adduct="[M-H]-", ion_score=0.92,
                        ppm_error=0.2 + (0.3 if i % 2 else -0.3),
                        pass_no=1, method="cheminfo+grid", confidence="Good",
                        commentary="backbone")
cfg_cal = P.PassConfig()
res = P.calibrate(cal_led, cfg_cal, log=lambda *a: None)
check("calibrate fits backbone mu", res is not None and abs(cfg_cal.cal_mu - 0.2) < 0.11,
      (cfg_cal.cal_mu, cfg_cal.cal_sigma))
check("z_of judges against calibration",
      P.z_of(0.2, cfg_cal) is not None and P.z_of(0.2, cfg_cal) < 0.1
      and P.z_of(4.0, cfg_cal) > cfg_cal.cal_z_pattern, P.z_of(4.0, cfg_cal))
check("z_of None when uncalibrated", P.z_of(1.0, P.PassConfig()) is None)
check("z_of None on NaN ppm", P.z_of(float("nan"), cfg_cal) is None)
small = L.new_ledger(pd.DataFrame({"peak_id": ["a"], "mz": [100.0], "height": [1e4]}))
cfg_small = P.PassConfig()
check("calibrate refuses a tiny backbone",
      P.calibrate(small, cfg_small, log=lambda *a: None) is None
      and cfg_small.cal_mu is None)

# ---------- offset-tolerance: a -2.4 ppm instrument (uronium) ----------
# the backbone is high-SCORE but mislabeled 'Low' (the pre-calibration confidence
# gate is centered on 0 ppm). calibrate must still find the -2.4 ppm center by
# selecting on SCORE, not the label.
off_led = L.new_ledger(pd.DataFrame({
    "peak_id": [f"o{i}" for i in range(30)],
    "mz": [150.0 + i for i in range(30)], "height": [1e4] * 30}))
for i in range(30):
    L.commit_assignment(off_led, f"o{i}", neutral_formula="C8H18O3",
                        adduct="[M+H]+", ion_score=0.95, compound_score=0.95,
                        ppm_error=-2.4 + (0.2 if i % 2 else -0.2),
                        pass_no=1, method="cheminfo+grid",
                        confidence="Low", commentary="backbone")
cfg_off = P.PassConfig()
P.calibrate(off_led, cfg_off, log=lambda *a: None)
check("calibrate finds the -2.4 ppm offset despite 'Low' labels",
      cfg_off.cal_mu is not None and abs(cfg_off.cal_mu + 2.4) < 0.15
      and cfg_off.cal_sigma < 0.6, (cfg_off.cal_mu, cfg_off.cal_sigma))

# confidence_label judges ppm against the calibrated center
cfg_u = P.PassConfig()
check("confidence_label: -2.4 ppm reads Low when uncalibrated (centered 0)",
      P.confidence_label(0.97, -2.4, 1, False, cfg_u) == "Low")
cfg_u.cal_mu, cfg_u.cal_sigma = -2.4, 0.3
check("confidence_label: -2.4 ppm reads High at the calibrated center",
      P.confidence_label(0.97, -2.4, 1, False, cfg_u) == "High")
check("confidence_label: an off-trend +0.3 ppm fit reads Low even calibrated",
      P.confidence_label(0.85, 0.3, 0, False, cfg_u) == "Low")

# relabel_confidence re-grades pass-1's commits against the fitted center:
# upgrade the on-cal backbone, demote the off-trend monster, keep the suffix.
rel_led = L.new_ledger(pd.DataFrame({
    "peak_id": ["good", "mon"], "mz": [200.0, 462.1], "height": [1e5, 1e5]}))
L.commit_assignment(rel_led, "good", neutral_formula="C8H18O3", adduct="[M+H]+",
                    ion_score=0.96, compound_score=0.96, ppm_error=-2.45,
                    pass_no=1, method="cheminfo+grid", confidence="Low",
                    commentary="x")
L.commit_assignment(rel_led, "mon", neutral_formula="C15H27NO15", adduct="[M+H]+",
                    ion_score=0.81, compound_score=0.81, ppm_error=0.32,
                    pass_no=2, method="gka-series", confidence="Good (series)",
                    commentary="x")
cfg_rel = P.PassConfig(); cfg_rel.cal_mu, cfg_rel.cal_sigma = -2.45, 0.3
P.relabel_confidence(rel_led, cfg_rel, log=lambda *a: None)
gconf = str(rel_led.loc[rel_led.peak_id == "good", "confidence"].iloc[0])
mconf = str(rel_led.loc[rel_led.peak_id == "mon", "confidence"].iloc[0])
check("relabel upgrades the on-cal backbone (Low -> Good/High)",
      gconf.startswith(("Good", "High")), gconf)
check("relabel demotes the off-trend monster + keeps the suffix",
      mconf.startswith(("Low", "Suspect")) and "series" in mconf, mconf)
check("relabel is a no-op when uncalibrated",
      P.relabel_confidence(rel_led, P.PassConfig(), log=lambda *a: None) == 0)

# arbitration is calibration-aware: an off-trend mass-coincidence with a HIGHER
# raw score loses to the on-trend formula once calibrated (else it wins then is
# z-rejected at commit, leaving the peak unexplained -- the high-Si PDMS failure).
cfg_arb = P.PassConfig(); cfg_arb.cal_mu, cfg_arb.cal_sigma = -2.45, 0.27
sc_off = pd.DataFrame([
    iso_row(sample_peak_id="P", compound_formula="C12H39NO6Si6", compound_score=0.85,
            ion_formula="C12H40NO6Si6", ion_score=0.85, ppm_error=-2.40),   # on-trend
    iso_row(sample_peak_id="P", compound_formula="C19H51NO4Si9", compound_score=0.90,
            ion_formula="C19H52NO4Si9", ion_score=0.90, ppm_error=1.60),    # off-trend, higher
])
w_cal = P.arbitrate(sc_off, cfg_arb)["winners"].iloc[0]
check("arbitration: on-trend PDMS beats higher-scoring off-trend coincidence (calibrated)",
      w_cal["neutral"] == "C12H39NO6Si6", (w_cal["neutral"], w_cal["eff_score"]))
w_unc = P.arbitrate(sc_off, P.PassConfig())["winners"].iloc[0]
check("arbitration uncalibrated: higher raw score still wins (no cal penalty)",
      w_unc["neutral"] == "C19H51NO4Si9", w_unc["neutral"])

# ---------- commit gates: NaN ppm, z-score, minor channel ----------
def gate_ledger(pids):
    return L.new_ledger(pd.DataFrame({
        "peak_id": pids, "mz": [300.0 + i for i in range(len(pids))],
        "height": [1e4] * len(pids)}))

def one_winner(pid, formula, ion, ppm, score=0.85, n_iso=0):
    return {"winners": pd.DataFrame([{
        "peak_id": pid, "neutral": formula, "ion_formula": ion,
        "adduct": P._mech_to_adduct({"compound_formula": formula, "ion_formula": ion}),
        "ion_score": score, "compound_score": score, "raw_score": score,
        "eff_score": score, "ppm_error": ppm, "n_iso": n_iso, "tied": False,
        "alternatives": []}]), "iso_children": pd.DataFrame()}

led_g = gate_ledger(["n1"])
s = P.commit_winners(led_g, one_winner("n1", "C9H10BrNO8", "C9H9BrNO8-", None),
                     pass_no=3, method="contaminant:bromo_organic", context="ambient-air",
                     cfg=cfg_cal, lock=False, min_raw_score=0.5)
check("NaN-ppm winner rejected", s["committed"] == 0 and s["rejected"]["nan_ppm"] == 1, s)

led_g = gate_ledger(["z1"])
s = P.commit_winners(led_g, one_winner("z1", "C9H10O4", "C9H9O4-", 3.9),
                     pass_no=3, method="contaminant:bromo_organic", context="ambient-air",
                     cfg=cfg_cal, lock=False, min_raw_score=0.5)
check("z>accept without evidence rejected",
      s["committed"] == 0 and s["rejected"]["mass_gate"] == 1, s)

led_g = gate_ledger(["z2"])
s = P.commit_winners(led_g, one_winner("z2", "C9H10O4", "C9H9O4-", 0.9, n_iso=1),
                     pass_no=3, method="contaminant:bromo_organic", context="ambient-air",
                     cfg=cfg_cal, lock=False, min_raw_score=0.5)
check("2<z<=4 with isotope evidence commits", s["committed"] == 1, (s, P.z_of(0.9, cfg_cal)))

led_g = gate_ledger(["z3"])
s = P.commit_winners(led_g, one_winner("z3", "C9H10O4", "C9H9O4-", 3.9, n_iso=3),
                     pass_no=3, method="contaminant:bromo_organic", context="ambient-air",
                     cfg=cfg_cal, lock=False, min_raw_score=0.5)
check("z>pattern band rejected even with isotopes",
      s["committed"] == 0 and s["rejected"]["mass_gate"] == 1, s)

# minor-channel commit gate: a lone Low CO3 winner is refused...
led_g = gate_ledger(["m1"])
s = P.commit_winners(led_g, one_winner("m1", "C7H9NO8", "C8H9NO11-", 0.3, score=0.75),
                     pass_no=3, method="contaminant:bromo_organic", context="ambient-air",
                     cfg=cfg_cal, lock=False, min_raw_score=0.5)
check("lone Low minor-channel winner rejected",
      s["committed"] == 0 and s["rejected"]["minor_channel"] == 1, s)
# ...a Good+ CO3 winner commits...
led_g = gate_ledger(["m2"])
s = P.commit_winners(led_g, one_winner("m2", "C5H10O", "C6H10O4-", 0.3, score=0.92),
                     pass_no=1, method="cheminfo+grid", context="ambient-air",
                     cfg=cfg_cal, lock=False, min_raw_score=0.5)
check("Good minor-channel winner commits", s["committed"] == 1, s)
# ...and a Low CO3 winner with the same neutral assigned via a primary channel commits.
led_g = gate_ledger(["m3", "m4"])
P.commit_winners(led_g, one_winner("m3", "C7H9NO8", "C7H8NO8-", 0.2, score=0.9),
                 pass_no=1, method="cheminfo+grid", context="ambient-air",
                 cfg=cfg_cal, lock=False, min_raw_score=0.5)
s = P.commit_winners(led_g, one_winner("m4", "C7H9NO8", "C8H9NO11-", 0.3, score=0.75),
                     pass_no=3, method="contaminant:bromo_organic", context="ambient-air",
                     cfg=cfg_cal, lock=False, min_raw_score=0.5)
check("cross-channel-corroborated minor winner commits", s["committed"] == 1, s)

# ---------- channel prior in arbitrate: near-tie goes to the primary channel ----------
scored_ch = pd.DataFrame([
    iso_row(sample_peak_id="P", compound_formula="C7H9NO8", compound_score=0.95,
            ion_formula="C8H9NO11-", ion_score=0.95, ppm_error=1.1),     # [M+CO3]-
    iso_row(sample_peak_id="P", compound_formula="C10H16O5", compound_score=0.87,
            ion_formula="C10H16BrO5-", ion_score=0.87, ppm_error=-0.7),  # [M+Br]-
])
arb_ch = P.arbitrate(scored_ch, CFG)
wp = arb_ch["winners"].iloc[0]
check("near-tie: primary-channel [M+Br]- beats minor [M+CO3]- (295.018 case)",
      wp["neutral"] == "C10H16O5" and wp["adduct"] == "[M+Br]-",
      (wp["neutral"], wp["adduct"], wp["eff_score"]))

# ---------- M0-vs-iso-child displacement through commit_winners ----------
led_d = gate_ledger(["A1", "A2"])
L.commit_assignment(led_d, "A2", neutral_formula="C6H11NO7Si", adduct="[M+CO3]-",
                    ion_score=0.60, ppm_error=1.5, pass_no=2, method="gka-series",
                    confidence="Low (series)", commentary="weak own M0")
arb_d = {"winners": pd.DataFrame([{
            "peak_id": "A1", "neutral": "C10H16O5", "ion_formula": "C10H16BrO5-",
            "adduct": "[M+Br]-", "ion_score": 0.87, "compound_score": 0.87,
            "raw_score": 0.87, "eff_score": 0.87, "ppm_error": -0.7, "n_iso": 1,
            "tied": False, "alternatives": []}]),
         "iso_children": pd.DataFrame([{
            "peak_id": "A2", "parent_peak_id": "A1", "iso_label": "81Br",
            "iso_score": 0.95}])}
s = P.commit_winners(led_d, arb_d, pass_no=1, method="cheminfo+grid",
                     context="ambient-air", cfg=cfg_cal, lock=False,
                     min_raw_score=0.5)
check("weak M0 displaced into parent's 81Br child (297.016 case)",
      s["iso_displaced"] == 1 and L.role_of(led_d, "A2") == L.ROLE_ISO
      and led_d.loc[led_d.peak_id == "A2", "parent_peak_id"].iloc[0] == "A1", s)
# a strong existing M0 is NOT displaced
led_d2 = gate_ledger(["B1", "B2"])
L.commit_assignment(led_d2, "B2", neutral_formula="C9H14O4", adduct="[M-H]-",
                    ion_score=0.95, ppm_error=0.1, pass_no=1, method="cheminfo+grid",
                    confidence="Good", commentary="strong own M0")
arb_d2 = {"winners": arb_d["winners"].assign(peak_id="B1"),
          "iso_children": arb_d["iso_children"].assign(peak_id="B2",
                                                       parent_peak_id="B1")}
s = P.commit_winners(led_d2, arb_d2, pass_no=3, method="contaminant:bromo_organic",
                     context="ambient-air", cfg=cfg_cal, lock=False,
                     min_raw_score=0.5)
check("Good existing M0 not displaced",
      s["iso_displaced"] == 0 and L.role_of(led_d2, "B2") == L.ROLE_M0, s)

# ---------- audit_mass_gate ----------
led_a = gate_ledger(["q1", "q2", "q3", "q4", "q5"])
L.commit_assignment(led_a, "q1", neutral_formula="C11H12N2O16", adduct="[M-H]-",
                    ion_score=0.81, ppm_error=3.83, pass_no=1, method="cheminfo+grid",
                    confidence="Low", commentary="pass1 junk tail")
L.commit_assignment(led_a, "q2", neutral_formula="C10H16O4", adduct="[M-H]-",
                    ion_score=0.95, ppm_error=-0.3, pass_no=1, method="cheminfo+grid",
                    confidence="Good", commentary="backbone")
L.commit_assignment(led_a, "q3", neutral_formula="C9H13NO5", adduct="[M-H]-",
                    ion_score=0.72, ppm_error=1.6, pass_no=3, method="x",
                    confidence="Low", commentary="2-4 sigma, no evidence")
L.commit_assignment(led_a, "q4", neutral_formula="C8H12O6S", adduct="[M-H]-",
                    ion_score=0.72, ppm_error=1.6, pass_no=3, method="x",
                    confidence="Low", commentary="2-4 sigma WITH child")
L.attach_isotopologue(led_a, "q5", "q4", iso_label="34S", iso_match_score=0.8)
out = P.audit_mass_gate(led_a, cfg_cal, log=lambda *a: None)
check("audit clears >4-sigma pass1 Low", L.role_of(led_a, "q1") == L.ROLE_UNEXPLAINED, out)
check("audit clears 2-4 sigma Low without evidence",
      L.role_of(led_a, "q3") == L.ROLE_UNEXPLAINED, out)
check("audit keeps 2-4 sigma Low WITH iso child", L.role_of(led_a, "q4") == L.ROLE_M0, out)
check("audit ledger validates clean", L.validate(led_a) == [], L.validate(led_a))

# ---------- audit_isotopes: post-run isotope-physics audit ----------
def mk_ledger(rows):
    led = L.new_ledger(pd.DataFrame(
        {"peak_id": [r[0] for r in rows], "mz": [r[1] for r in rows],
         "height": [r[2] for r in rows]}))
    return led

def commit(led, pid, neutral, ion, conf="Good", score=0.9):
    L.commit_assignment(led, pid, neutral_formula=neutral, adduct="[M+Br]-",
                        ion_formula=ion, ion_score=score, ppm_error=0.1,
                        pass_no=3, method="test", confidence=conf,
                        commentary="t")

ACFG = P.PassConfig(height_cutoff=100.0)

# v16 case: 462.99/464.99 — two Good M0s 1.99795 apart, ~1:1; light ion has Br
led = mk_ledger([("La", 462.9933, 26882.0), ("Hb", 464.9913, 25609.0),
                 ("X", 463.9966, 3450.0)])   # X = 13C of La
commit(led, "La", "C12H12F12", "C12H12BrF12-")
commit(led, "Hb", "C11H9F7O12", "C11H8F7O12-")
s = P.audit_isotopes(led, ACFG, log=lambda *a: None)
check("audit: Br-doublet heavy M0 demoted to 81Br child",
      L.role_of(led, "Hb") == L.ROLE_ISO and s["doublet_child"] == 1, s)
check("audit: 13C satellite swept up as evidence",
      L.role_of(led, "X") == L.ROLE_ISO and s["c13_attached"] == 1, s)

# doublet where NEITHER formula carries Br -> both cleared, but ONLY in Br-CIMS
# (where a ~1:1 1.998 doublet is strong evidence of an unassigned bromine).
ACFG_BR = P.PassConfig(height_cutoff=100.0, reagent_element="Br")
led = mk_ledger([("Lc", 284.0501, 641.0), ("Hd", 286.0480, 543.0)])
commit(led, "Lc", "C9H20NO4", "C9H20NO4-")    # no Br anywhere
commit(led, "Hd", "C14H10O3", "C15H10O6-")
s = P.audit_isotopes(led, ACFG_BR, log=lambda *a: None)
check("audit: no-Br doublet clears both formulas (Br-CIMS)",
      L.role_of(led, "Lc") == L.ROLE_UNEXPLAINED
      and L.role_of(led, "Hd") == L.ROLE_UNEXPLAINED
      and s["doublet_cleared"] == 2, s)

# SAME doublet under a NON-bromine reagent (e.g. 15N-nitrate, reagent_element
# None): the 1.998 spacing is NOT halogen evidence -- two unrelated CHON
# compounds routinely fall ~1.998 apart -- so neither M0 may be cleared.
led = mk_ledger([("Lc", 284.0501, 641.0), ("Hd", 286.0480, 543.0)])
commit(led, "Lc", "C9H20NO4", "C9H20NO4-")
commit(led, "Hd", "C14H10O3", "C15H10O6-")
s = P.audit_isotopes(led, ACFG, log=lambda *a: None)   # reagent_element=None
check("audit: non-Br reagent does NOT clear a 1.998 doublet of two non-Br M0s",
      L.role_of(led, "Lc") == L.ROLE_M0
      and L.role_of(led, "Hd") == L.ROLE_M0
      and s["doublet_cleared"] == 0, s)

# 13C clamp: formula claims C19, satellite measures ~C11 (bright satellite -> fires)
led = mk_ledger([("M", 444.9861, 4761.0), ("K", 445.9895, 564.0)])
commit(led, "M", "C19H15N2O4S", "C19H14BrN2O4S-")
L.attach_isotopologue(led, "K", "M", iso_label="13C")
s = P.audit_isotopes(led, ACFG, log=lambda *a: None)
check("audit: 13C carbon clamp clears C19-vs-C11",
      L.role_of(led, "M") == L.ROLE_UNEXPLAINED and s["c13_clamp"] == 1, s)

# 13C clamp guard: a low-intensity M0 whose 13C satellite is BELOW the detection
# floor must NOT be clamped -- the sub-floor ratio is noise (under-reads carbon),
# which falsely cleared genuine ~2k cps [M+15NO3]- M0s. Same C8-vs-~C4 ratio as
# the false positive, but the satellite (90 cps) is < height_cutoff (100).
led = mk_ledger([("Lm", 280.0440, 2000.0), ("Lk", 281.0474, 90.0)])
commit(led, "Lm", "C8H11NO6", "C8H11NO9^N-", conf="Good", score=0.87)
L.attach_isotopologue(led, "Lk", "Lm", iso_label="13C")
s = P.audit_isotopes(led, ACFG, log=lambda *a: None)
check("audit: 13C clamp SKIPS a sub-floor (noisy) 13C satellite",
      L.role_of(led, "Lm") == L.ROLE_M0 and s["c13_clamp"] == 0, s)

# 13C missing: big peak, formula predicts a visible satellite, none exists
led = mk_ledger([("N", 168.9505, 10300.0)])
commit(led, "N", "C3H6O3", "C3H6BrO3-")
s = P.audit_isotopes(led, ACFG, log=lambda *a: None)
check("audit: predicted-13C-absent clears the formula",
      L.role_of(led, "N") == L.ROLE_UNEXPLAINED and s["c13_missing"] == 1, s)

# sanity: a CORRECT assignment survives all four checks
led = mk_ledger([("P", 295.0184, 2965.0), ("Q", 297.0164, 2924.0),
                 ("R", 296.0218, 320.0)])
commit(led, "P", "C10H16O5", "C10H16BrO5-")
L.attach_isotopologue(led, "Q", "P", iso_label="81Br")
s = P.audit_isotopes(led, ACFG, log=lambda *a: None)
check("audit: correct assignment untouched (13C attached, nothing cleared)",
      L.role_of(led, "P") == L.ROLE_M0 and s["c13_clamp"] == 0
      and s["c13_missing"] == 0 and s["c13_attached"] == 1, s)

# ---------- complete_isotope_envelopes (the 393/395 silanediol bug) ----------
from peaky import chemistry as CHEM  # noqa: E402
# silanediol Si4+Br at 393 (M0), its M+2 at 395 wrongly committed as a Cl-F-S
# organic, M+4 at 397 attached as 395's child. Heights = real Si4+Br envelope.
led = mk_ledger([("Si", 393.0045, 20086.0), ("Mp2", 395.0028, 24523.0),
                 ("Mp4", 397.0029, 6344.0), ("far", 600.0, 5000.0)])
commit(led, "Si", "C8H26O5Si4", "C8H26BrO5Si4-")        # the silanediol M0
L.lock_peaks(led, ["Si"])                                 # pass-0 locks it
L.commit_assignment(led, "Mp2", neutral_formula="C8H12ClF6NO2S", adduct="[M+CO3]-",
                    ion_formula="C9H12ClF6NO5S-", ion_score=0.63, ppm_error=-1.6,
                    pass_no=4, method="residual:iso-pair", confidence="Low (iso-pair)",
                    commentary="phantom Cl doublet")
L.attach_isotopologue(led, "Mp4", "Mp2", iso_label="37Cl(pair)")
from peaky.assignment import tiers as _T  # noqa: E402
_T.apply_tiers(led)
out = P.complete_isotope_envelopes(led, P.PassConfig(), log=lambda *a: None)
check("envelope: silanediol M+2 (395) displaced off its phantom formula",
      L.role_of(led, "Mp2") == L.ROLE_ISO and out["displaced"] >= 1, out)
check("envelope: 395 re-parented to the silanediol 393",
      led.loc[led.peak_id == "Mp2", "parent_peak_id"].iloc[0] == "Si")
check("envelope: 397 (M+4) re-parented to the silanediol too",
      led.loc[led.peak_id == "Mp4", "parent_peak_id"].iloc[0] == "Si")
check("envelope: ledger still valid after displacement", L.validate(led) == [], L.validate(led))

# di-bromide [M+HBr+Br]- core (2 Br in the ion) commits in pass 6 -> the new 3rd
# envelope sweep must claim its M+2 (~1.95x) and M+4 (~0.95x) satellites, which
# were sitting in the residual (the Family-A GKA ladder, 2026-06-13).
ledd = mk_ledger([("core", 356.9348, 742.0), ("m2", 358.9328, 1395.0),
                  ("m4", 360.9308, 700.0)])
L.commit_assignment(ledd, "core", neutral_formula="C10H14O4", adduct="[M+HBr+Br]-",
                    ion_formula="C10H15Br2O4-", ion_score=0.80, ppm_error=-0.4,
                    pass_no=6, method="ladder:gapfill", confidence="Good (ladder)",
                    commentary="di-bromide SOA core")
_T.apply_tiers(ledd)
outd = P.complete_isotope_envelopes(ledd, P.PassConfig(), log=lambda *a: None)
check("envelope: di-bromide M+2 (358.93) attached to the core",
      L.role_of(ledd, "m2") == L.ROLE_ISO
      and ledd.loc[ledd.peak_id == "m2", "parent_peak_id"].iloc[0] == "core", outd)
check("envelope: di-bromide M+4 (360.93) attached to the core",
      L.role_of(ledd, "m4") == L.ROLE_ISO
      and ledd.loc[ledd.peak_id == "m4", "parent_peak_id"].iloc[0] == "core", outd)

# guard: a CHO-only M0 must NOT claim a coincidental peak ~2 Da above it (its
# pattern has no M+2 driver), and must NOT displace a real neighbour
led2 = mk_ledger([("a", 200.0, 1e5), ("b", 202.0, 5e4)])
commit(led2, "a", "C10H16O4", "C10H15O4-")   # CHO ion, no Br/Cl/Si
commit(led2, "b", "C9H12O5", "C9H11O5-")      # independent neighbour
_T.apply_tiers(led2)
out2 = P.complete_isotope_envelopes(led2, P.PassConfig(), log=lambda *a: None)
check("envelope: CHO ion does not claim a coincidental +2 neighbour",
      L.role_of(led2, "b") == L.ROLE_M0 and out2["displaced"] == 0, out2)

# guard (review #3): a STRONG victim (High conf OR near-High score) sitting at a
# parent's M+2 with matching intensity must NOT be displaced -- the tier column
# is NA during this pass, so protection rides on confidence + score, not tier.
led3 = mk_ledger([("p", 300.0, 1e5), ("q", 301.9979, 9.7e4)])
commit(led3, "p", "C9H17BrO5", "C9H17BrO5-")          # 1-Br parent, predicts M+2
L.commit_assignment(led3, "q", neutral_formula="C12H20O8", adduct="[M-H]-",
                    ion_formula="C12H19O8-", ion_score=0.95, ppm_error=0.2,
                    pass_no=1, method="cheminfo+grid", confidence="High",
                    commentary="strong standalone fit")
outg = P.complete_isotope_envelopes(led3, P.PassConfig(), log=lambda *a: None)
check("envelope: a High/strong-score victim is NOT displaced (tier-NA safe)",
      L.role_of(led3, "q") == L.ROLE_M0 and outg["displaced"] == 0, outg)

# ---------- detect_composites (silanediol-on-BrCl even/odd test) ----------
# build a silanediol n=4 envelope: M0 inflated ~45% by a coincident BrCl
# compound. Odd shifts (M+1) = pure silanediol; even (M0/M+2/M+4) carry the
# extra halogen. Heights from real Br-CIMS data.
from peaky.chem import chemistry as _CH  # noqa: E402
mz0 = _CH.ion_mz("C8H26O5Si4", "[M+Br]-")   # 393.0046
comp = mk_ledger([("M0", mz0, 20086.0), ("M1", mz0 + 1.0008, 2698.0),
                  ("M1b", mz0 + 1.0034, 531.0), ("M2", mz0 + 1.9979, 24523.0),
                  ("M3", mz0 + 2.9986, 2748.0), ("M4", mz0 + 3.9957, 6344.0),
                  ("clean", 244.9670, 10712.0), ("cleanM1", 244.9670 + 1.0, 1461.0),
                  ("cleanM2", 244.9670 + 1.9979, 7000.0)])
commit(comp, "M0", "C8H26O5Si4", "C8H26BrO5Si4-")
commit(comp, "clean", "C4H14O3Si2", "C4H14BrO3Si2-")   # n=2: ~clean, small extra
oc = P.detect_composites(comp, P.PassConfig(), log=lambda *a: None)
note4 = comp.loc[comp.peak_id == "M0", "composite_note"].iloc[0]
check("composite: silanediol n=4 flagged as composite", pd.notna(note4) and oc["flagged"] >= 1, note4)
check("composite: identifies the BrCl co-component", "BrCl" in str(note4), note4)
check("composite: estimates ~45% co-component", "45%" in str(note4) or "44%" in str(note4), note4)
check("composite: clean n=2 NOT flagged",
      pd.isna(comp.loc[comp.peak_id == "clean", "composite_note"].iloc[0]),
      comp.loc[comp.peak_id == "clean", "composite_note"].iloc[0])
# a CHO compound whose M0 matches its M+1 is NOT flagged (no inflation)
pure = mk_ledger([("p", 200.0, 1e5), ("pm1", 201.0034, 1e5 * 10 * 0.0107)])
commit(pure, "p", "C10H16O4", "C10H15O4-")
P.detect_composites(pure, P.PassConfig(), log=lambda *a: None)
check("composite: a self-consistent CHO M0 is NOT flagged",
      pd.isna(pure.loc[pure.peak_id == "p", "composite_note"].iloc[0]))

# ---------- split_composites: de-blend into fractional sub-peaks ----------
P.detect_composites(comp, P.PassConfig(), log=lambda *a: None)
comp2 = P.split_composites(comp, P.PassConfig(), log=lambda *a: None)
host = comp2[comp2.peak_id == "M0"].iloc[0]
sub = comp2[comp2.peak_id == "M0.2"]
check("split: a synthetic sub-peak M0.2 was created", len(sub) == 1, list(comp2.peak_id))
check("split: host keeps assigned_fraction < 1 (the co-component is removed)",
      float(host["assigned_fraction"]) < 0.95, host["assigned_fraction"])
if len(sub):
    s = sub.iloc[0]
    check("split: sub-peak is synthetic, at the host m/z, linked to the host",
          bool(s["synthetic"]) and abs(float(s["mz"]) - float(host["mz"])) < 1e-6
          and s["host_peak_id"] == "M0", (s["synthetic"], s["host_peak_id"]))
    check("split: signal is conserved (host_eff + sub == measured host height)",
          abs(float(host["height"]) * float(host["assigned_fraction"])
              + float(s["height"]) - float(host["height"])) < 2.0,
          (host["height"], host["assigned_fraction"], s["height"]))
    check("split: sub-peak commentary names the co-component (BrCl)",
          "BrCl" in str(s["commentary"]), s["commentary"])
# stats: synthetic excluded from the real-peak count, signal not double-counted
st_c = L.stats(comp2)
check("split: stats excludes synthetic from n_peaks", st_c["n_synthetic"] >= 1
      and st_c["n_peaks"] == int((~comp2["synthetic"].fillna(False).astype(bool)).sum()),
      (st_c["n_peaks"], st_c["n_synthetic"]))
check("split: ledger still validates with synthetic rows", L.validate(comp2) == [],
      L.validate(comp2))

# ---------- demote_carbon_inconsistent (pre-pass-4 O15-monster clear) ----------
# the 409.0015 case: pass 1 grabbed C11H10N2O15 (ion C11) but the 13C satellite
# at +1.0034 measures ~C16 -> must clear BEFORE pass 4 so the di-bromide SOA
# core (C15) can be re-claimed. Satellite ratio 16*1.07% = 0.171 of the parent.
led = mk_ledger([("mon", 409.0015, 4720.0), ("sat", 410.0049, 807.0)])
commit(led, "mon", "C11H10N2O15", "C11H10N2BrO15-")   # ion carbon = 11
n = P.demote_carbon_inconsistent(led, ACFG, log=lambda *a: None)
check("pre-pass4: C11 monster with ~C16 satellite demoted",
      n == 1 and L.role_of(led, "mon") == L.ROLE_UNEXPLAINED, (n, L.role_of(led, "mon")))
check("pre-pass4: freed peak is re-claimable (unexplained, no formula)",
      pd.isna(led.loc[led.peak_id == "mon", "neutral_formula"].iloc[0]))
# a carbon-CONSISTENT assignment is left alone (C15 ion, ~C15 satellite)
led2 = mk_ledger([("ok", 409.0015, 4720.0), ("s2", 410.0049, 758.0)])
commit(led2, "ok", "C15H23BrO3", "C15H23Br2O3-")      # ion carbon = 15
n2 = P.demote_carbon_inconsistent(led2, ACFG, log=lambda *a: None)
check("pre-pass4: carbon-consistent C15 assignment survives",
      n2 == 0 and L.role_of(led2, "ok") == L.ROLE_M0, (n2, L.role_of(led2, "ok")))
# no satellite -> cannot measure -> never demotes (avoids false clears)
led3 = mk_ledger([("nosat", 409.0015, 4720.0)])
commit(led3, "nosat", "C11H10N2O15", "C11H10N2BrO15-")
check("pre-pass4: no satellite -> no demotion",
      P.demote_carbon_inconsistent(led3, ACFG, log=lambda *a: None) == 0
      and L.role_of(led3, "nosat") == L.ROLE_M0)

# ---------- audit: twin-satellite fallback for missing 13C ----------
# v20 false-clear: C3H6O3.Br- at 10.3k cps -- own 13C absent from the peak
# list, but the 81Br twin's 13C satellite exists and proves the carbon.
from peaky import chemistry as CH  # noqa: E402
mz_l = CH.ion_mz("C3H6O3", "[M+Br]-")
led = mk_ledger([("La", mz_l, 10300.0),
                 ("Tw", mz_l + 1.9979535, 10000.0),
                 ("Sat", mz_l + 1.9979535 + 1.0033548, 350.0)])
commit(led, "La", "C3H6O3", "C3H6BrO3-")
L.attach_isotopologue(led, "Tw", "La", iso_label="81Br")
s = P.audit_isotopes(led, ACFG, log=lambda *a: None)
check("audit: twin 13C satellite blocks the missing-13C clear",
      L.role_of(led, "La") == L.ROLE_M0 and s["c13_missing"] == 0, s)
# control: same peak with NO twin satellite still clears
led = mk_ledger([("Lb", mz_l, 10300.0), ("Tw2", mz_l + 1.9979535, 10000.0)])
commit(led, "Lb", "C3H6O3", "C3H6BrO3-")
L.attach_isotopologue(led, "Tw2", "Lb", iso_label="81Br")
s = P.audit_isotopes(led, ACFG, log=lambda *a: None)
check("audit: no satellite anywhere still clears",
      L.role_of(led, "Lb") == L.ROLE_UNEXPLAINED and s["c13_missing"] == 1, s)
# cross-channel fallback: missing 13C but the SAME neutral assigned Good on
# another peak (other adduct) -> positive evidence wins, no clear (v21 case)
mh = CH.ion_mz("C10H16O6", "[M-H]-")
mbr = CH.ion_mz("C10H16O6", "[M+Br]-")
led = mk_ledger([("MH", mh, 1060.0), ("MB", mbr, 2426.0)])
L.commit_assignment(led, "MH", neutral_formula="C10H16O6", adduct="[M-H]-",
                    ion_formula="C10H15O6-", ion_score=0.87, ppm_error=-0.9,
                    pass_no=1, method="t", confidence="Good", commentary="t")
L.commit_assignment(led, "MB", neutral_formula="C10H16O6", adduct="[M+Br]-",
                    ion_formula="C10H16BrO6-", ion_score=0.86, ppm_error=-0.5,
                    pass_no=1, method="t", confidence="Good", commentary="t")
s = P.audit_isotopes(led, ACFG, log=lambda *a: None)
check("audit: cross-channel agreement blocks the missing-13C clear",
      L.role_of(led, "MB") == L.ROLE_M0 and s["c13_missing"] == 0, s)

# ---------- pass 5: known-neutral completion ----------
from peaky import contexts as XC  # noqa: E402
PROF5 = XC.get_context("ambient-air")
ADD5 = ["[M-H]-", "[M+Br]-"]
mz_c2 = CH.ion_mz("C2H4O3", "[M+Br]-")
mz_c5 = CH.ion_mz("C5H10O3", "[M+Br]-")
mz_c3 = CH.ion_mz("C3H6O3", "[M+Br]-")          # bracketed gap (C2..C5, j=1)
mz_x_mh = CH.ion_mz("C10H16O3", "[M-H]-")
mz_x_br = CH.ion_mz("C10H16O3", "[M+Br]-")      # cross-channel partner
led5 = mk_ledger([("A2", mz_c2, 700.0), ("A5", mz_c5, 1000.0),
                  ("G3", mz_c3, 10300.0), ("AX", mz_x_mh, 400.0),
                  ("PX", mz_x_br, 1500.0), ("far", 555.555, 300.0)])
commit(led5, "A2", "C2H4O3", "C2H4BrO3-", conf="High")
commit(led5, "A5", "C5H10O3", "C5H10BrO3-", conf="High")
L.commit_assignment(led5, "AX", neutral_formula="C10H16O3", adduct="[M-H]-",
                    ion_formula="C10H15O3-", ion_score=0.9, ppm_error=0.1,
                    pass_no=1, method="t", confidence="Good", commentary="t")

def fake5(client, sample_id, formulas, *, mechanism_ids=None, **kw):
    rows = []
    if "C3H6O3" in formulas:
        rows.append(dict(compound_formula="C3H6O3", compound_score=0.9,
                         ion_formula="C3H6BrO3-", ion_score=0.9, iso_label="M0",
                         is_base=True, theo_mz=mz_c3, rel_abundance=1.0,
                         iso_score=0.9, sample_peak_id="G3", sample_peak_mz=mz_c3,
                         sample_peak_intensity=10300.0, ppm_error=0.3,
                         abundance_error=0.0))
    if "C10H16O3" in formulas:
        rows.append(dict(compound_formula="C10H16O3", compound_score=0.9,
                         ion_formula="C10H16BrO3-", ion_score=0.9, iso_label="M0",
                         is_base=True, theo_mz=mz_x_br, rel_abundance=1.0,
                         iso_score=0.9, sample_peak_id="PX", sample_peak_mz=mz_x_br,
                         sample_peak_intensity=1500.0, ppm_error=0.2,
                         abundance_error=0.0))
    return pd.DataFrame(rows)

s5 = P.run_pass5_completion(None, "SID", led5, PROF5, ACFG, ADD5,
                            score_fn=fake5, log=lambda *a: None)
check("pass5 commits the bracketed series gap (C3H6O3)",
      led5.loc[led5.peak_id == "G3", "neutral_formula"].iloc[0] == "C3H6O3", s5)
check("pass5 commits the cross-channel partner (C10H16O3 [M+Br]-)",
      led5.loc[led5.peak_id == "PX", "neutral_formula"].iloc[0] == "C10H16O3"
      and led5.loc[led5.peak_id == "PX", "adduct"].iloc[0] == "[M+Br]-", s5)
check("pass5 commits exactly the two targets", s5["committed"] == 2, s5)
check("pass5 method recorded", "completion" in
      led5.loc[led5.peak_id == "G3", "method"].iloc[0])
check("untargeted peak untouched",
      L.role_of(led5, "far") == L.ROLE_UNEXPLAINED)
check("ledger valid after pass5", L.validate(led5) == [], L.validate(led5))

# ---------- pass 0: known-contaminant pre-pass (silanediol ladder) ----------
check("silanediol series composition", P._silanediol_series(3) ==
      ["C2H8O2Si1", "C4H14O3Si2", "C6H20O4Si3"], P._silanediol_series(3))
mz_n2 = CH.ion_mz("C4H14O3Si2", "[M+Br]-")
# Si2 needs a 29Si M+1 ~ 2*4.68% + 4*1.07% = 13.7% of M0 (the Si-count intensity gate)
led0 = mk_ledger([("D2", mz_n2, 10712.0), ("D2si", mz_n2 + 0.99957, 1500.0),
                  ("D2br", mz_n2 + 1.99795, 10100.0)])

def fake0(client, sample_id, formulas, *, mechanism_ids=None, **kw):
    rows = []
    if "C4H14O3Si2" in formulas:
        rows.append(dict(compound_formula="C4H14O3Si2", compound_score=0.92,
                         ion_formula="C4H14BrO3Si2-", ion_score=0.92,
                         iso_label="M0", is_base=True, theo_mz=mz_n2,
                         rel_abundance=1.0, iso_score=0.92,
                         sample_peak_id="D2", sample_peak_mz=mz_n2,
                         sample_peak_intensity=10712.0, ppm_error=-0.87,
                         abundance_error=0.0))
        rows.append(dict(compound_formula="C4H14O3Si2", compound_score=0.92,
                         ion_formula="C4H14BrO3Si2-", ion_score=0.92,
                         iso_label="81Br", is_base=False,
                         theo_mz=mz_n2 + 1.99795, rel_abundance=0.97,
                         iso_score=0.94, sample_peak_id="D2br",
                         sample_peak_mz=mz_n2 + 1.99795,
                         sample_peak_intensity=10100.0, ppm_error=-0.8,
                         abundance_error=0.02))
    return pd.DataFrame(rows)

s0 = P.run_pass0_known(None, "SID", led0, PROF5, ACFG, ADD5,
                              score_fn=fake0, log=lambda *a: None)
check("pass0 commits the silanediol oligomer",
      led0.loc[led0.peak_id == "D2", "neutral_formula"].iloc[0] == "C4H14O3Si2"
      and s0["committed"] == 1, s0)
check("pass0 locks the peak", L.is_locked(led0, "D2"))
check("pass0 attaches the 81Br child",
      L.role_of(led0, "D2br") == L.ROLE_ISO, s0)
# the v24 failure mode: pass-1 must NOT be able to displace it with a CHO fit
try:
    L.commit_assignment(led0, "D2", neutral_formula="C5H10O6",
                        adduct="[M+Br]-", ion_formula="C5H10BrO6-",
                        ion_score=0.95, ppm_error=0.8, pass_no=1, method="grid",
                        confidence="High", commentary="bogus")
    stolen = True
except L.LedgerError:
    stolen = False
check("locked contaminant refuses the CHO grid fit", not stolen
      and led0.loc[led0.peak_id == "D2", "neutral_formula"].iloc[0] == "C4H14O3Si2")

# pass0 twin gate: a contaminant claim on a peak whose own 81Br twin is
# missing must be refused (the v25 lactic-acid collision)
mz_n1 = CH.ion_mz("C2H8O2Si1", "[M+Br]-")
ledg = mk_ledger([("X1", mz_n1, 11950.0), ("Xw", mz_n1 + 1.99795, 427.0)])

def fake_n1(client, sample_id, formulas, *, mechanism_ids=None, **kw):
    return pd.DataFrame([dict(
        compound_formula="C2H8O2Si1", compound_score=0.9,
        ion_formula="C2H8BrO2Si-", ion_score=0.9, iso_label="M0",
        is_base=True, theo_mz=mz_n1, rel_abundance=1.0, iso_score=0.9,
        sample_peak_id="X1", sample_peak_mz=mz_n1,
        sample_peak_intensity=11950.0, ppm_error=1.0, abundance_error=0.0)])

sg = P.run_pass0_known(None, "SID", ledg, PROF5, ACFG, ADD5,
                              score_fn=fake_n1, log=lambda *a: None)
check("pass0 refuses claim with inconsistent own twin (ratio 0.04)",
      sg["committed"] == 0 and L.role_of(ledg, "X1") == L.ROLE_UNEXPLAINED, sg)

# pass0 nitroaromatic: dinitrophenol C6H4N2O5 [M-H]- is H-poor (VK-floor blocked)
# so the grid can't reach it; pass-0 supplies it. No Br -> no twin gate, just the
# |ppm|<=2 + score gate. (v45->v46 fix; confirmed present by Orbitool.)
check("dinitrophenol in the known-species nitroaromatic family",
      P._known_species().get("nitroaromatic", {}).get("C6H4N2O5") is not None)
mz_dnp = CH.ion_mz("C6H4N2O5", "[M-H]-")
ledn = mk_ledger([("DNP", mz_dnp, 808.0)])

def fake_dnp(client, sample_id, formulas, *, mechanism_ids=None, **kw):
    if "C6H4N2O5" not in formulas:
        return pd.DataFrame([])
    return pd.DataFrame([dict(
        compound_formula="C6H4N2O5", compound_score=0.82,
        ion_formula="C6H3N2O5-", ion_score=0.82, iso_label="M0",
        is_base=True, theo_mz=mz_dnp, rel_abundance=1.0, iso_score=0.82,
        sample_peak_id="DNP", sample_peak_mz=mz_dnp,
        sample_peak_intensity=808.0, ppm_error=-0.24, abundance_error=0.0)])

sn = P.run_pass0_known(None, "SID", ledn, PROF5, ACFG, ADD5,
                       score_fn=fake_dnp, log=lambda *a: None)
check("pass0 commits dinitrophenol [M-H]- (no Br twin gate)",
      sn["committed"] == 1
      and ledn.loc[ledn.peak_id == "DNP", "neutral_formula"].iloc[0] == "C6H4N2O5"
      and ledn.loc[ledn.peak_id == "DNP", "method"].iloc[0] == "known:nitroaromatic", sn)

# pass0 reactive iodine (iodide-CIMS): covalent I is monoisotopic + off-grid, so
# INO2 (via [M+I]- = I2NO2-, the 299.802 line) must be supplied as a
# known species. No Br in the ion -> no twin gate; exact-mass + score commit.
check("INO2 in the known-species reactive_iodine family",
      P._known_species().get("reactive_iodine", {}).get("INO2") is not None)
check("HOI in the known-species reactive_iodine family",
      P._known_species().get("reactive_iodine", {}).get("HOI") is not None)
check("OIO (IO2 radical) in the reactive_iodine family",
      P._known_species().get("reactive_iodine", {}).get("IO2") is not None)
check("iodic acid (HIO3) in the reactive_iodine family (owns IO3- via [M-H]-)",
      P._known_species().get("reactive_iodine", {}).get("HIO3") is not None)
mz_ino2 = CH.ion_mz("INO2", "[M+I]-")
# non-circular anchor: the server-observed I2NO2- line (299.8025)
check("INO2 [M+I]- lands on the observed 299.8025 line",
      abs(mz_ino2 - 299.8025) < 0.001, mz_ino2)
ledi = mk_ledger([("INO2", mz_ino2, 2.8e6)])

def fake_ino2(client, sample_id, formulas, *, mechanism_ids=None, **kw):
    if "INO2" not in formulas:
        return pd.DataFrame([])
    return pd.DataFrame([dict(
        compound_formula="INO2", compound_score=0.99,
        ion_formula="I2NO2-", ion_score=0.99, iso_label="M0",
        is_base=True, theo_mz=mz_ino2, rel_abundance=1.0, iso_score=0.99,
        sample_peak_id="INO2", sample_peak_mz=mz_ino2,
        sample_peak_intensity=2.8e6, ppm_error=0.34, abundance_error=0.0)])

si = P.run_pass0_known(None, "SID", ledi, PROF5, ACFG, ["[M+I]-", "[M-H]-"],
                       score_fn=fake_ino2, log=lambda *a: None)
check("pass0 commits INO2 [M+I]- (reactive iodine, off-grid covalent I)",
      si["committed"] == 1
      and ledi.loc[ledi.peak_id == "INO2", "neutral_formula"].iloc[0] == "INO2"
      and ledi.loc[ledi.peak_id == "INO2", "method"].iloc[0] == "known:reactive_iodine", si)
check("pass0 locks the reactive-iodine commit", L.is_locked(ledi, "INO2"))

# IBr via [M+I]- = I2Br-: 'Br' in the ion formula -> the pass-0 self-twin gate
# applies; the 0.96-ratio 81Br twin (the 332.728/334.726 pair) passes.
mz_ibr = CH.ion_mz("IBr", "[M+I]-")
# non-circular anchor: the server-observed I2Br- line (332.7278)
check("IBr [M+I]- lands on the observed 332.7278 line",
      abs(mz_ibr - 332.7278) < 0.001, mz_ibr)
ledb = mk_ledger([("IBR", mz_ibr, 5716.0), ("IBRtw", mz_ibr + 1.99795, 5503.0)])

def fake_ibr(client, sample_id, formulas, *, mechanism_ids=None, **kw):
    if "IBr" not in formulas:
        return pd.DataFrame([])
    return pd.DataFrame([
        dict(compound_formula="IBr", compound_score=0.99, ion_formula="BrI2-",
             ion_score=0.99, iso_label="M0", is_base=True, theo_mz=mz_ibr,
             rel_abundance=1.0, iso_score=0.99, sample_peak_id="IBR",
             sample_peak_mz=mz_ibr, sample_peak_intensity=5716.0,
             ppm_error=-0.1, abundance_error=0.0),
        dict(compound_formula="IBr", compound_score=0.99, ion_formula="BrI2-",
             ion_score=0.99, iso_label="81Br", is_base=False,
             theo_mz=mz_ibr + 1.99795, rel_abundance=0.97, iso_score=0.95,
             sample_peak_id="IBRtw", sample_peak_mz=mz_ibr + 1.99795,
             sample_peak_intensity=5503.0, ppm_error=0.04, abundance_error=0.02)])

sb = P.run_pass0_known(None, "SID", ledb, PROF5, ACFG, ["[M+I]-", "[M-H]-"],
                       score_fn=fake_ibr, log=lambda *a: None)
check("pass0 commits IBr [M+I]- (81Br self-twin ratio 0.96 passes the gate)",
      sb["committed"] == 1
      and ledb.loc[ledb.peak_id == "IBR", "neutral_formula"].iloc[0] == "IBr", sb)
check("pass0 attaches the IBr 81Br child", L.role_of(ledb, "IBRtw") == L.ROLE_ISO, sb)

# ---------- pass0 known-species gate is offset-aware (prior_offset) ----------
# a silanediol at -2.3 ppm: rejected when the instrument offset is unknown
# (|2.3|>2.0), committed once the rough offset (-1.9) is seeded -- the
# silanediol-vs-C5H10O6 collision at a -1.9 ppm source.
mz_off = CH.ion_mz("C4H14O3Si2", "[M+Br]-")


def fake_off(client, sample_id, formulas, *, mechanism_ids=None, **kw):
    if "C4H14O3Si2" not in formulas:
        return pd.DataFrame([])
    return pd.DataFrame([
        dict(compound_formula="C4H14O3Si2", compound_score=0.92,
             ion_formula="C4H14BrO3Si2-", ion_score=0.92, iso_label="M0",
             is_base=True, theo_mz=mz_off, rel_abundance=1.0, iso_score=0.92,
             sample_peak_id="O2", sample_peak_mz=mz_off,
             sample_peak_intensity=10712.0, ppm_error=-2.3, abundance_error=0.0),
        dict(compound_formula="C4H14O3Si2", compound_score=0.92,
             ion_formula="C4H14BrO3Si2-", ion_score=0.92, iso_label="81Br",
             is_base=False, theo_mz=mz_off + 1.99795, rel_abundance=0.97,
             iso_score=0.94, sample_peak_id="O2br",
             sample_peak_mz=mz_off + 1.99795, sample_peak_intensity=10100.0,
             ppm_error=-2.3, abundance_error=0.02)])


_si_m1 = [("O2si", mz_off + 0.99957, 1500.0)]   # Si2 29Si M+1 (the Si-count intensity gate)
led_b = mk_ledger([("O2", mz_off, 10712.0), *_si_m1, ("O2br", mz_off + 1.99795, 10100.0)])
P.run_pass0_known(None, "SID", led_b, PROF5, P.PassConfig(), ADD5,
                  score_fn=fake_off, log=lambda *a: None)
check("pass0 rejects a -2.3 ppm known species when offset-blind",
      L.role_of(led_b, "O2") == L.ROLE_UNEXPLAINED)
led_o = mk_ledger([("O2", mz_off, 10712.0), *_si_m1, ("O2br", mz_off + 1.99795, 10100.0)])
cfg_seed = P.PassConfig(); cfg_seed.prior_offset = -1.9
P.run_pass0_known(None, "SID", led_o, PROF5, cfg_seed, ADD5,
                  score_fn=fake_off, log=lambda *a: None)
check("pass0 commits the -2.3 ppm known species once offset seeded (-1.9)",
      L.role_of(led_o, "O2") == L.ROLE_M0
      and led_o.loc[led_o.peak_id == "O2", "neutral_formula"].iloc[0] == "C4H14O3Si2")

# ---------- build_ranges reads the context grid-box width ----------
PROF_URO = XC.get_context("uronium")
r_amb = P.build_ranges(PROF5, None, include_N=True)               # ambient-air
r_uro = P.build_ranges(PROF_URO, None, include_N=True)
check("ambient grid box C<=40 / O<=30",
      r_amb["C"][1] == 40 and r_amb["O"][1] == 30, (r_amb["C"], r_amb["O"]))
check("uronium grid box widened to C<=46 / O<=32",
      r_uro["C"][1] == 46 and r_uro["O"][1] == 32, (r_uro["C"], r_uro["O"]))
check("uronium grid admits N up to the context cap",
      r_uro["N"][1] == PROF_URO.max_N, r_uro["N"])
r_ovr = P.build_ranges(PROF_URO, None, include_N=True, c_max=20, o_max=10)
check("explicit c_max/o_max overrides the profile",
      r_ovr["C"][1] == 20 and r_ovr["O"][1] == 10, (r_ovr["C"], r_ovr["O"]))

# ---------- positive polarity: pass 0 knows the organophosphate contaminants ----
check("_known_species(positive) carries the organophosphate family",
      "organophosphate" in P._known_species("positive")
      and "C6H15O4P" in P._known_species("positive")["organophosphate"])
check("_known_species(positive) registers ambient ammonia (H3N) as a known species",
      "H3N" in P._known_species("positive").get("ambient_inorganic", {}))
check("_known_species(negative) has NO ammonia entry (urea-adduct channel is +mode)",
      "ambient_inorganic" not in P._known_species("negative"))
check("_known_species(negative) keeps the atmospheric list",
      "atmospheric" in P._known_species("negative"))
check("_known_species(negative) carries the PFCA (perfluoroacid) series incl TFA",
      "perfluoroacid" in P._known_species("negative")
      and "C2HF3O2" in P._known_species("negative")["perfluoroacid"])
check("_known_species(negative) carries chlorinated_paraffin incl C11H18Cl6",
      "chlorinated_paraffin" in P._known_species("negative")
      and "C11H18Cl6" in P._known_species("negative")["chlorinated_paraffin"])

# pass0: commit a 37Cl-confirmed chlorinated paraffin, refuse an unconfirmed one
_cpmz = 358.947
def _cp_rows(with_kids):
    rows = [iso_row(compound_formula="C11H18Cl6", compound_score=0.5, ion_formula="C11H17Cl6-",
                    isotope_formula="C11H17Cl6-", iso_label="M0", is_base=True, theo_mz=_cpmz,
                    sample_peak_id="cp0", sample_peak_mz=_cpmz, sample_peak_intensity=5000.0,
                    ppm_error=0.4, ion_score=0.7)]
    if with_kids:
        rows += [iso_row(compound_formula="C11H18Cl6", ion_formula="C11H17Cl6-",
                         isotope_formula="[37Cl]C11H17Cl5-", iso_label="37Cl", is_base=False,
                         theo_mz=_cpmz + 1.997, sample_peak_id="cp1", sample_peak_mz=_cpmz + 1.997,
                         sample_peak_intensity=4000.0, ppm_error=0.4, iso_score=0.95),
                 iso_row(compound_formula="C11H18Cl6", ion_formula="C11H17Cl6-",
                         isotope_formula="[37Cl2]C11H17Cl4-", iso_label="37Cl2", is_base=False,
                         theo_mz=_cpmz + 3.994, sample_peak_id="cp2", sample_peak_mz=_cpmz + 3.994,
                         sample_peak_intensity=2500.0, ppm_error=0.4, iso_score=0.9)]
    return pd.DataFrame(rows)
def _fake_cp(c, s, forms, *, mechanism_ids=None, **kw):
    return _cp_rows(True) if "C11H18Cl6" in forms else pd.DataFrame([])
def _fake_cp0(c, s, forms, *, mechanism_ids=None, **kw):
    return _cp_rows(False) if "C11H18Cl6" in forms else pd.DataFrame([])
_ledcp = mk_ledger([("cp0", _cpmz, 5000.0), ("cp1", _cpmz + 1.997, 4000.0), ("cp2", _cpmz + 3.994, 2500.0)])
P.run_pass0_known(None, "SID", _ledcp, PROF5, ACFG, ADD5, score_fn=_fake_cp, log=lambda *a: None)
check("pass0 commits a 37Cl-confirmed chlorinated paraffin",
      _ledcp.loc[_ledcp.peak_id == "cp0", "neutral_formula"].iloc[0] == "C11H18Cl6"
      and _ledcp.loc[_ledcp.peak_id == "cp0", "method"].iloc[0] == "known:chlorinated_paraffin")
_ledcp0 = mk_ledger([("cp0", _cpmz, 5000.0)])
P.run_pass0_known(None, "SID", _ledcp0, PROF5, ACFG, ADD5, score_fn=_fake_cp0, log=lambda *a: None)
check("pass0 refuses a 37Cl-unconfirmed chlorinated paraffin (n_kids<2)",
      L.role_of(_ledcp0, "cp0") == L.ROLE_UNEXPLAINED)

# pass0 RECOVERY: a chlorinated paraffin the server scored too low to ANCHOR
# (base unanchored: no sample_peak_id, ppm NaN -- the "too low score on the
# server" miss) is recovered from the ledger by exact mass + a real 37Cl envelope.
_rmz = 422.939
def _cp_unanchored(c, s, forms, *, mechanism_ids=None, **kw):
    if "C11H18Cl6" not in forms:
        return pd.DataFrame([])
    rows = [iso_row(compound_formula="C11H18Cl6", compound_score=0.12,
                    ion_formula="C11H18Cl6O3^N-", isotope_formula="C11H18Cl6O3^N-",
                    iso_label="M0", is_base=True, theo_mz=_rmz, sample_peak_id=None,
                    sample_peak_mz=_rmz, sample_peak_intensity=0.0,
                    ppm_error=None, ion_score=0.12)]
    for k, lab in ((1, "37Cl+15N"), (2, "37Cl2+15N")):   # server can't confirm them either
        rows.append(iso_row(compound_formula="C11H18Cl6", ion_formula="C11H18Cl6O3^N-",
                            iso_label=lab, is_base=False, theo_mz=_rmz + k * P._D37CL,
                            sample_peak_id=None, sample_peak_mz=None,
                            sample_peak_intensity=0.0, ppm_error=None, iso_score=0.1))
    return pd.DataFrame(rows)
_ledr = mk_ledger([("r0", _rmz, 5000.0), ("r1", _rmz + P._D37CL, 6000.0),
                   ("r2", _rmz + 2 * P._D37CL, 4000.0)])
sr = P.run_pass0_known(None, "SID", _ledr, PROF5, ACFG, ADD5,
                       score_fn=_cp_unanchored, log=lambda *a: None)
check("pass0 RECOVERS a low-score chlorinated paraffin via the ledger 37Cl envelope",
      L.role_of(_ledr, "r0") == L.ROLE_M0
      and _ledr.loc[_ledr.peak_id == "r0", "neutral_formula"].iloc[0] == "C11H18Cl6"
      and _ledr.loc[_ledr.peak_id == "r0", "method"].iloc[0] == "known:chlorinated_paraffin"
      and sr.get("recovered", 0) == 1, sr)
check("pass0 recovery locks the M0 and attaches its 37Cl satellites",
      L.is_locked(_ledr, "r0")
      and L.role_of(_ledr, "r1") == L.ROLE_ISO
      and L.role_of(_ledr, "r2") == L.ROLE_ISO)
check("pass0 recovery flags the depressed-score recovery in the confidence label",
      "recovered" in str(_ledr.loc[_ledr.peak_id == "r0", "confidence"].iloc[0]))

# no real peak at the mass -> recovery must NOT fabricate
_ledr_none = mk_ledger([("x", 200.0, 5000.0)])
sr0 = P.run_pass0_known(None, "SID", _ledr_none, PROF5, ACFG, ADD5,
                        score_fn=_cp_unanchored, log=lambda *a: None)
check("pass0 recovery does NOT fabricate when no real peak exists at the mass",
      sr0.get("recovered", 0) == 0 and L.role_of(_ledr_none, "x") == L.ROLE_UNEXPLAINED)

# real M0 peak but only ONE 37Cl satellite -> envelope unconfirmed, no commit
_ledr1 = mk_ledger([("r0", _rmz, 5000.0), ("r1", _rmz + P._D37CL, 6000.0)])
sr1 = P.run_pass0_known(None, "SID", _ledr1, PROF5, ACFG, ADD5,
                        score_fn=_cp_unanchored, log=lambda *a: None)
check("pass0 recovery refuses a single-satellite (37Cl envelope unconfirmed)",
      sr1.get("recovered", 0) == 0 and L.role_of(_ledr1, "r0") == L.ROLE_UNEXPLAINED)

# safety: a monoisotopic-F family (perfluoroacid) is NOT recoverable even if the
# ledger happens to carry peaks at the right masses -- no isotope twin corroborates
_pmz = 412.9664
def _pfca_unanchored(c, s, forms, *, mechanism_ids=None, **kw):
    if "C8HF15O2" not in forms:
        return pd.DataFrame([])
    rows = [iso_row(compound_formula="C8HF15O2", compound_score=0.1,
                    ion_formula="C8F15O2-", isotope_formula="C8F15O2-", iso_label="M0",
                    is_base=True, theo_mz=_pmz, sample_peak_id=None, ppm_error=None,
                    ion_score=0.1)]
    rows += [iso_row(compound_formula="C8HF15O2", ion_formula="C8F15O2-",
                     iso_label="37Cl", is_base=False, theo_mz=_pmz + k * P._D37CL,
                     sample_peak_id=None, ppm_error=None, iso_score=0.1) for k in (1, 2)]
    return pd.DataFrame(rows)
_ledp = mk_ledger([("p0", _pmz, 5000.0), ("p1", _pmz + P._D37CL, 6000.0),
                   ("p2", _pmz + 2 * P._D37CL, 4000.0)])
spf = P.run_pass0_known(None, "SID", _ledp, PROF5, ACFG, ADD5,
                        score_fn=_pfca_unanchored, log=lambda *a: None)
check("pass0 recovery does NOT apply to monoisotopic-F families (perfluoroacid)",
      spf.get("recovered", 0) == 0 and L.role_of(_ledp, "p0") == L.ROLE_UNEXPLAINED)

# TEP (C6H15O4P) seen in BOTH [M+H]+ and [M+(urea)H]+ -> cross-channel corroborated
_mzH = CH.ion_mz("C6H15O4P", "[M+H]+")
_mzU = CH.ion_mz("C6H15O4P", "[M+(CH4N2O)H]+")
def _ope_row(ion, mz, pid, mech):
    return dict(compound_formula="C6H15O4P", compound_score=0.92, ion_formula=ion,
                ion_score=0.92, iso_label="M0", is_base=True, theo_mz=mz,
                rel_abundance=1.0, iso_score=0.92, sample_peak_id=pid,
                sample_peak_mz=mz, sample_peak_intensity=5000.0, ppm_error=0.0,
                abundance_error=0.0, mechanism_id=mech)
def fake_ope2(client, sid, formulas, *, mechanism_ids=None, **kw):
    if "C6H15O4P" not in formulas:
        return pd.DataFrame([])
    return pd.DataFrame([_ope_row("C6H16O4P+", _mzH, "tepH", "mH"),
                         _ope_row("C7H20N2O5P+", _mzU, "tepU", "mU")])
led_ope = mk_ledger([("tepH", _mzH, 5000.0), ("tepU", _mzU, 4000.0)])
s_ope = P.run_pass0_known(None, "SID", led_ope, PROF_URO, ACFG,
                          ["[M+H]+", "[M+(CH4N2O)H]+"], score_fn=fake_ope2,
                          log=lambda *a: None)
check("pass0 commits TEP across both ion channels (cross-channel corroborated)",
      s_ope["committed"] == 2
      and led_ope.loc[led_ope.peak_id == "tepH", "neutral_formula"].iloc[0] == "C6H15O4P"
      and led_ope.loc[led_ope.peak_id == "tepH", "method"].iloc[0] == "known:organophosphate", s_ope)
check("pass0 labels the TEP urea channel adduct correctly",
      led_ope.loc[led_ope.peak_id == "tepU", "adduct"].iloc[0] == "[M+(CH4N2O)H]+")

# single-channel TEP -> NOT committed (monoisotopic P needs >=2 channels)
def fake_ope1(client, sid, formulas, *, mechanism_ids=None, **kw):
    if "C6H15O4P" not in formulas:
        return pd.DataFrame([])
    return pd.DataFrame([_ope_row("C6H16O4P+", _mzH, "tepH", "mH")])
led_ope1 = mk_ledger([("tepH", _mzH, 5000.0)])
s_ope1 = P.run_pass0_known(None, "SID", led_ope1, PROF_URO, ACFG,
                           ["[M+H]+", "[M+(CH4N2O)H]+"], score_fn=fake_ope1,
                           log=lambda *a: None)
check("pass0 refuses a single-channel organophosphate (no cross-channel support)",
      s_ope1["committed"] == 0, s_ope1)

# ---------- positive polarity: organothiophosphate pesticide family (malathion) ----
check("_known_species(positive) carries the organothiophosphate family with malathion",
      "organothiophosphate" in P._known_species("positive")
      and "C10H19O6PS2" in P._known_species("positive")["organothiophosphate"])
# malathion (C10H19O6PS2) seen in BOTH [M+H]+ and [M+(urea)H]+ -> committed
_mzMH = CH.ion_mz("C10H19O6PS2", "[M+H]+")
_mzMU = CH.ion_mz("C10H19O6PS2", "[M+(CH4N2O)H]+")
def _mal_row(ion, mz, pid, mech):
    return dict(compound_formula="C10H19O6PS2", compound_score=0.9, ion_formula=ion,
                ion_score=0.9, iso_label="M0", is_base=True, theo_mz=mz,
                rel_abundance=1.0, iso_score=0.9, sample_peak_id=pid,
                sample_peak_mz=mz, sample_peak_intensity=5000.0, ppm_error=0.0,
                abundance_error=0.0, mechanism_id=mech)
def fake_mal2(client, sid, formulas, *, mechanism_ids=None, **kw):
    if "C10H19O6PS2" not in formulas:
        return pd.DataFrame([])
    return pd.DataFrame([_mal_row("C10H20O6PS2+", _mzMH, "malH", "mH"),
                         _mal_row("C11H24N2O7PS2+", _mzMU, "malU", "mU")])
led_mal = mk_ledger([("malH", _mzMH, 5000.0), ("malU", _mzMU, 4000.0)])
s_mal = P.run_pass0_known(None, "SID", led_mal, PROF_URO, ACFG,
                          ["[M+H]+", "[M+(CH4N2O)H]+"], score_fn=fake_mal2,
                          log=lambda *a: None)
check("pass0 commits malathion across both ion channels (organothiophosphate)",
      s_mal["committed"] == 2
      and led_mal.loc[led_mal.peak_id == "malH", "neutral_formula"].iloc[0] == "C10H19O6PS2"
      and led_mal.loc[led_mal.peak_id == "malH", "method"].iloc[0] == "known:organothiophosphate", s_mal)
# single-channel malathion -> NOT committed (P needs >=2 channels)
def fake_mal1(client, sid, formulas, *, mechanism_ids=None, **kw):
    if "C10H19O6PS2" not in formulas:
        return pd.DataFrame([])
    return pd.DataFrame([_mal_row("C10H20O6PS2+", _mzMH, "malH", "mH")])
led_mal1 = mk_ledger([("malH", _mzMH, 5000.0)])
s_mal1 = P.run_pass0_known(None, "SID", led_mal1, PROF_URO, ACFG,
                           ["[M+H]+", "[M+(CH4N2O)H]+"], score_fn=fake_mal1,
                           log=lambda *a: None)
check("pass0 refuses a single-channel organothiophosphate (no cross-channel support)",
      s_mal1["committed"] == 0, s_mal1)
# single channel BUT a matched 34S envelope -> commits (isotope stands in for 2nd channel)
_mzDE = CH.ion_mz("C8H13O5PS2", "[M+H]+")
def fake_de_s34(client, sid, formulas, *, mechanism_ids=None, **kw):
    if "C8H13O5PS2" not in formulas:
        return pd.DataFrame([])
    base = _mal_row("C8H14O5PS2+", _mzDE, "deH", "mH")
    base["compound_formula"] = "C8H13O5PS2"
    s34 = dict(base); s34.update(iso_label="34S", is_base=False,
                                 theo_mz=_mzDE + 1.99579, sample_peak_id="deS34",
                                 sample_peak_mz=_mzDE + 1.99579, iso_score=0.9)
    return pd.DataFrame([base, s34])
led_de = mk_ledger([("deH", _mzDE, 9000.0), ("deS34", _mzDE + 1.99579, 800.0)])
s_de = P.run_pass0_known(None, "SID", led_de, PROF_URO, ACFG,
                         ["[M+H]+", "[M+(CH4N2O)H]+"], score_fn=fake_de_s34,
                         log=lambda *a: None)
check("pass0 commits a single-channel organothiophosphate with a confirmed 34S envelope",
      L.role_of(led_de, "deH") == L.ROLE_M0
      and led_de.loc[led_de.peak_id == "deH", "neutral_formula"].iloc[0] == "C8H13O5PS2", s_de)
# same single channel but NO 34S match -> still refused (guards the phosphate esters)
def fake_de_nos34(client, sid, formulas, *, mechanism_ids=None, **kw):
    if "C8H13O5PS2" not in formulas:
        return pd.DataFrame([])
    base = _mal_row("C8H14O5PS2+", _mzDE, "deH", "mH")
    base["compound_formula"] = "C8H13O5PS2"
    return pd.DataFrame([base])
led_de2 = mk_ledger([("deH", _mzDE, 9000.0)])
s_de2 = P.run_pass0_known(None, "SID", led_de2, PROF_URO, ACFG,
                          ["[M+H]+", "[M+(CH4N2O)H]+"], score_fn=fake_de_nos34,
                          log=lambda *a: None)
check("pass0 still refuses a single-channel thiophosphate with no 34S envelope",
      s_de2["committed"] == 0, s_de2)
# GENERALIZED: single channel + a 37Cl envelope (not 34S) also corroborates
_mzCP = CH.ion_mz("C9H11Cl3NO3PS", "[M+H]+")   # chlorpyrifos (in the OTP list)
def fake_cp_cl37(client, sid, formulas, *, mechanism_ids=None, **kw):
    if "C9H11Cl3NO3PS" not in formulas:
        return pd.DataFrame([])
    base = _mal_row("C9H12Cl3NO3PS+", _mzCP, "cpH", "mH")
    base["compound_formula"] = "C9H11Cl3NO3PS"
    cl37 = dict(base); cl37.update(iso_label="37Cl", is_base=False,
                                   theo_mz=_mzCP + 1.99705, sample_peak_id="cpCl37",
                                   sample_peak_mz=_mzCP + 1.99705, iso_score=0.9)
    return pd.DataFrame([base, cl37])
led_cp = mk_ledger([("cpH", _mzCP, 9000.0), ("cpCl37", _mzCP + 1.99705, 3000.0)])
s_cp = P.run_pass0_known(None, "SID", led_cp, PROF_URO, ACFG,
                         ["[M+H]+", "[M+(CH4N2O)H]+"], score_fn=fake_cp_cl37,
                         log=lambda *a: None)
check("pass0 commits a single-channel thiophosphate via a 37Cl envelope (generalized)",
      L.role_of(led_cp, "cpH") == L.ROLE_M0, s_cp)
# 13C GUARD: a 13C line is NOT a diagnostic heteroatom isotope -> must NOT license off-grid P
def fake_de_13c_only(client, sid, formulas, *, mechanism_ids=None, **kw):
    if "C8H13O5PS2" not in formulas:
        return pd.DataFrame([])
    base = _mal_row("C8H14O5PS2+", _mzDE, "deH", "mH")
    base["compound_formula"] = "C8H13O5PS2"
    c13 = dict(base); c13.update(iso_label="13C", is_base=False,
                                 theo_mz=_mzDE + 1.00336, sample_peak_id="deC13",
                                 sample_peak_mz=_mzDE + 1.00336, iso_score=0.9)
    return pd.DataFrame([base, c13])
led_13c = mk_ledger([("deH", _mzDE, 9000.0), ("deC13", _mzDE + 1.00336, 800.0)])
s_13c = P.run_pass0_known(None, "SID", led_13c, PROF_URO, ACFG,
                          ["[M+H]+", "[M+(CH4N2O)H]+"], score_fn=fake_de_13c_only,
                          log=lambda *a: None)
check("pass0 refuses a single-channel thiophosphate corroborated only by 13C (13C guard)",
      s_13c["committed"] == 0, s_13c)

# ---------- selection prior: a reference-list neutral wins a near-tie ----------
sp = pd.DataFrame([
    iso_row(sample_peak_id="Z", compound_formula="C8H12O4", compound_score=0.88,
            ion_formula="C8H11O4-", ion_score=0.88, ppm_error=0.2),       # not listed
    iso_row(sample_peak_id="Z", compound_formula="C10H16O5", compound_score=0.86,
            ion_formula="C10H15O5-", ion_score=0.86, ppm_error=0.2),      # listed HOM
])
check("no prior: higher raw wins (C8H12O4)",
      P.arbitrate(sp, P.PassConfig())["winners"].iloc[0]["neutral"] == "C8H12O4")
cfg_sp = P.PassConfig(); cfg_sp.reflist_formulas = frozenset({"C10H16O5"})
check("reflist prior breaks the near-tie toward the listed HOM",
      P.arbitrate(sp, cfg_sp)["winners"].iloc[0]["neutral"] == "C10H16O5")
# but the prior must NOT override a clear winner (gap 0.12 > prior 0.04)
sp2 = pd.DataFrame([
    iso_row(sample_peak_id="Y", compound_formula="C8H12O4", compound_score=0.92,
            ion_formula="C8H11O4-", ion_score=0.92, ppm_error=0.2),
    iso_row(sample_peak_id="Y", compound_formula="C10H16O5", compound_score=0.80,
            ion_formula="C10H15O5-", ion_score=0.80, ppm_error=0.2),
])
check("reflist prior does NOT override a clear winner (gap 0.12 > 0.04)",
      P.arbitrate(sp2, cfg_sp)["winners"].iloc[0]["neutral"] == "C8H12O4")


# ---------- pass 7: certified-neutral discovery (NBBS urea ladder) ----------
# 3 unexplained peaks: [M+H]+, [M+urea+H]+, [M+2urea+H]+ of NBBS C10H15NO2S.
# The oracle anchors the two REGISTERED channels; the 2-urea rung commits on
# certificate evidence. The 34S kid on the [M+H]+ makes it Good (certified).
_mzN0 = CH.ion_mz("C10H15NO2S", "[M+H]+")            # 214.0896
_mzN1 = CH.ion_mz("C10H15NO2S", "[M+(CH4N2O)H]+")    # 274.1220
_mzN2 = _mzN1 + 60.0323627601                         # 334.1544 (no registered channel)
def _nbbs_row(ion, mz, pid, mech, formula="C10H15NO2S"):
    return dict(compound_formula=formula, compound_score=0.9, ion_formula=ion,
                ion_score=0.93, iso_label="M0", is_base=True, theo_mz=mz,
                rel_abundance=1.0, iso_score=0.93, sample_peak_id=pid,
                sample_peak_mz=mz, sample_peak_intensity=50000.0, ppm_error=0.05,
                abundance_error=0.0, mechanism_id=mech)
def fake_cert_nbbs(client, sid, formulas, *, mechanism_ids=None, **kw):
    if "C10H15NO2S" not in formulas:
        return pd.DataFrame([])
    b0 = _nbbs_row("C10H16NO2S+", _mzN0, "n0", "mH")
    b1 = _nbbs_row("C11H20N3O3S+", _mzN1, "n1", "mU")
    s34 = dict(b0); s34.update(iso_label="34S", is_base=False,
                               theo_mz=_mzN0 + 1.99580, sample_peak_id="n0s",
                               sample_peak_mz=_mzN0 + 1.99580, iso_score=0.9)
    return pd.DataFrame([b0, b1, s34])
led_cn = mk_ledger([("n0", _mzN0, 50000.0), ("n1", _mzN1, 400000.0),
                    ("n2", _mzN2, 3000.0), ("n0s", _mzN0 + 1.99580, 2200.0),
                    ("bg", 401.3337, 900.0)])
s_cn = P.run_pass_certified(None, "SID", led_cn, PROF_URO, ACFG,
                            ["[M+H]+", "[M+(CH4N2O)H]+", "[M+NH4]+"],
                            reagent="urea", score_fn=fake_cert_nbbs,
                            log=lambda *a: None)
check("pass7 certifies + commits the NBBS ladder", s_cn["committed"] == 1, s_cn)
check("pass7 claims all three rungs (2 anchored + 1 ladder rung)",
      s_cn["peaks_claimed"] == 3 and s_cn["rungs_committed"] == 1, s_cn)
check("pass7 commits the SAME neutral on every member peak",
      set(led_cn.loc[led_cn.peak_id.isin(["n0", "n1", "n2"]), "neutral_formula"])
      == {"C10H15NO2S"})
check("pass7 cross-channel: >=2 distinct adduct labels on the neutral",
      led_cn.loc[(led_cn.neutral_formula == "C10H15NO2S")
                 & (led_cn.role == L.ROLE_M0), "adduct"].nunique() >= 2)
check("pass7 method labels anchored vs ladder-rung commits",
      led_cn.loc[led_cn.peak_id == "n0", "method"].iloc[0] == "certified:multi-channel"
      and led_cn.loc[led_cn.peak_id == "n2", "method"].iloc[0] == "certified:ladder-rung")
check("pass7 34S kid attached to the anchored M0",
      L.role_of(led_cn, "n0s") == L.ROLE_ISO)
check("pass7 iso-confirmed -> Good (certified)",
      str(led_cn.loc[led_cn.peak_id == "n0", "confidence"].iloc[0]).startswith("Good"))
check("pass7 leaves unrelated peaks unexplained",
      L.role_of(led_cn, "bg") == L.ROLE_UNEXPLAINED)

# oracle anchors only ONE member channel -> no commit (certificate unproven)
def fake_cert_one(client, sid, formulas, *, mechanism_ids=None, **kw):
    if "C10H15NO2S" not in formulas:
        return pd.DataFrame([])
    return pd.DataFrame([_nbbs_row("C10H16NO2S+", _mzN0, "n0", "mH")])
led_cn1 = mk_ledger([("n0", _mzN0, 50000.0), ("n1", _mzN1, 400000.0),
                     ("n2", _mzN2, 3000.0)])
s_cn1 = P.run_pass_certified(None, "SID", led_cn1, PROF_URO, ACFG,
                             ["[M+H]+", "[M+(CH4N2O)H]+", "[M+NH4]+"],
                             reagent="urea", score_fn=fake_cert_one,
                             log=lambda *a: None)
check("pass7 refuses a certificate the oracle anchors on only 1 channel",
      s_cn1["committed"] == 0, s_cn1)

# ts_peaks=None path (default) already covered above; ts co-variation gate:
_ts_rows = []
for t in range(12):
    for mz, base_h in ((_mzN0, 50000.0), (_mzN1, 400000.0), (_mzN2, 3000.0)):
        _ts_rows.append(dict(datetime_utc=f"2026-06-0{(t % 7) + 1}T0{t % 10}:00:00Z",
                             mz=mz, height=base_h * (1.0 + 0.5 * ((t * 7) % 5))))
_ts = pd.DataFrame(_ts_rows)
led_cn2 = mk_ledger([("n0", _mzN0, 50000.0), ("n1", _mzN1, 400000.0),
                     ("n2", _mzN2, 3000.0), ("n0s", _mzN0 + 1.99580, 2200.0)])
s_cn2 = P.run_pass_certified(None, "SID", led_cn2, PROF_URO, ACFG,
                             ["[M+H]+", "[M+(CH4N2O)H]+", "[M+NH4]+"],
                             reagent="urea", ts_peaks=_ts, score_fn=fake_cert_nbbs,
                             log=lambda *a: None)
check("pass7 with co-varying ts_peaks still commits (extra corroboration)",
      s_cn2["committed"] == 1, s_cn2)
check("pass7 ts co-variation lands in the commentary",
      "co-vary in time" in str(led_cn2.loc[led_cn2.peak_id == "n0", "commentary"].iloc[0]))


# ---------- pass 7: weak-M0 interrogation + displacement ----------
# The n1 peak (274.122) carries a BOGUS single-channel [M+Na]+ Candidate-grade
# fit (instrument makes no Na). A 3-channel iso-confirmed certificate must
# displace it and re-read the peak as [M+urea+H]+ of the certified neutral.
led_dp = mk_ledger([("n0", _mzN0, 50000.0), ("n1", _mzN1, 400000.0),
                    ("n2", _mzN2, 3000.0), ("n0s", _mzN0 + 1.99580, 2200.0)])
L.commit_assignment(led_dp, "n1", neutral_formula="C12H21NO4", adduct="[M+Na]+",
                    ion_formula="C12H21NNaO4+", ion_score=0.86, ppm_error=0.4,
                    pass_no=1, method="grid", confidence="Low",
                    commentary="single-channel Na mass fit")
s_dp = P.run_pass_certified(None, "SID", led_dp, PROF_URO, ACFG,
                            ["[M+H]+", "[M+(CH4N2O)H]+", "[M+NH4]+"],
                            reagent="urea", score_fn=fake_cert_nbbs,
                            log=lambda *a: None)
check("pass7 displaces a weak [M+Na]+ Candidate under a strong certificate",
      s_dp["displaced"] == 1, s_dp)
check("pass7 re-reads the displaced peak as the certified neutral",
      led_dp.loc[led_dp.peak_id == "n1", "neutral_formula"].iloc[0] == "C10H15NO2S"
      and "CH4N2O" in str(led_dp.loc[led_dp.peak_id == "n1", "adduct"].iloc[0]))
check("pass7 displacement is audited in the commentary",
      "displaced by certified-neutral" in str(
          led_dp.loc[led_dp.peak_id == "n1", "commentary"].iloc[0])
      or "DISPLACED" in str(led_dp.loc[led_dp.peak_id == "n1", "commentary"].iloc[0])
      or "CLEARED" in str(led_dp.loc[led_dp.peak_id == "n1", "commentary"].iloc[0]))

# a Good-confidence incumbent is NOT interrogated: cert forms from the rest
led_gd = mk_ledger([("n0", _mzN0, 50000.0), ("n1", _mzN1, 400000.0),
                    ("n2", _mzN2, 3000.0), ("n0s", _mzN0 + 1.99580, 2200.0)])
L.commit_assignment(led_gd, "n1", neutral_formula="C12H21NO4", adduct="[M+Na]+",
                    ion_formula="C12H21NNaO4+", ion_score=0.95, ppm_error=0.1,
                    pass_no=1, method="grid", confidence="Good",
                    commentary="well-scored fit")
s_gd = P.run_pass_certified(None, "SID", led_gd, PROF_URO, ACFG,
                            ["[M+H]+", "[M+(CH4N2O)H]+", "[M+NH4]+"],
                            reagent="urea", score_fn=fake_cert_nbbs,
                            log=lambda *a: None)
check("pass7 leaves a Good-confidence incumbent alone",
      s_gd["displaced"] == 0
      and led_gd.loc[led_gd.peak_id == "n1", "neutral_formula"].iloc[0] == "C12H21NO4",
      s_gd)

# incumbent already IS the certified neutral -> corroborated, not displaced
led_cb = mk_ledger([("n0", _mzN0, 50000.0), ("n1", _mzN1, 400000.0),
                    ("n2", _mzN2, 3000.0), ("n0s", _mzN0 + 1.99580, 2200.0)])
L.commit_assignment(led_cb, "n1", neutral_formula="C10H15NO2S",
                    adduct="[M+(CH4N2O)H]+", ion_formula="C11H20N3O3S+",
                    ion_score=0.8, ppm_error=0.3, pass_no=1, method="grid",
                    confidence="Low", commentary="weak but right")
s_cb = P.run_pass_certified(None, "SID", led_cb, PROF_URO, ACFG,
                            ["[M+H]+", "[M+(CH4N2O)H]+", "[M+NH4]+"],
                            reagent="urea", score_fn=fake_cert_nbbs,
                            log=lambda *a: None)
check("pass7 counts a same-formula weak incumbent as corroborated, not displaced",
      s_cb["corroborated_existing"] == 1 and s_cb["displaced"] == 0
      and led_cb.loc[led_cb.peak_id == "n1", "neutral_formula"].iloc[0] == "C10H15NO2S",
      s_cb)


# ---------- pass 3: acid.I2 cluster resolver (iodide) ----------
# Anchored acids reappear as conjugate-base . I2 at A - H + 2*126.9045; the
# resolver scores the covalent alias (A-H+I) [M+I]- and commits the acid.
_mz_fa = _CH.ion_mz("CH2O2", "[M+I]-")        # formic anchor, 172.9123
_mz_fa_i2 = _CH.ion_mz("CH2O2", "[M-H+I2]-")  # its I2 rung, 298.8071
_mz_ac_i2 = _CH.ion_mz("C2H4O2", "[M-H+I2]-")  # acetic rung (CH2 homolog bridge)
_mz_hno2 = _CH.ion_mz("HNO2", "[M+I]-")        # HNO2 anchor
_mz_ino2 = _CH.ion_mz("INO2", "[M+I]-")        # == [HNO2-H+I2]- exactly (contested)


def _i2_row(x, ion, mz, pid):
    return dict(compound_formula=x, compound_score=0.85, ion_formula=ion,
                ion_score=0.88, iso_label="M0", is_base=True, theo_mz=mz,
                rel_abundance=1.0, iso_score=0.88, sample_peak_id=pid,
                sample_peak_mz=mz, sample_peak_intensity=30000.0, ppm_error=0.10,
                abundance_error=0.0, mechanism_id="+I-")


def fake_i2_score(client, sid, formulas, *, mechanism_ids=None, **kw):
    rows = []
    if "CHIO2" in formulas:      # covalent alias of formic's I2 rung
        rows.append(_i2_row("CHIO2", "CHI2O2-", _mz_fa_i2, "t1"))
    if "C2H3IO2" in formulas:    # covalent alias of the acetic homolog rung
        rows.append(_i2_row("C2H3IO2", "C2H3I2O2-", _mz_ac_i2, "t2"))
    return pd.DataFrame(rows)


led_i2 = mk_ledger([("fa", _mz_fa, 80000.0), ("t1", _mz_fa_i2, 30000.0),
                    ("t2", _mz_ac_i2, 12000.0), ("hn", _mz_hno2, 60000.0),
                    ("ino2", _mz_ino2, 250000.0), ("bg", 401.4012, 5000.0)])
for pid, nf in [("fa", "CH2O2"), ("hn", "HNO2")]:
    L.commit_assignment(led_i2, pid, neutral_formula=nf, adduct="[M+I]-",
                        ion_formula="x-", ion_score=0.9, ppm_error=0.1,
                        pass_no=1, method="grid", confidence="Good",
                        commentary="anchor")
# the contested I2NO2- line: pass-0 reactive-iodine INO2, committed + LOCKED
# first (ruled an ambient analyte on TIME behaviour) -- must never be re-read
# as [HNO2-H+I2]- even though the ion is identical.
L.commit_assignment(led_i2, "ino2", neutral_formula="INO2", adduct="[M+I]-",
                    ion_formula="I2NO2-", ion_score=0.95, ppm_error=0.1,
                    pass_no=0, method="known:reactive-iodine",
                    confidence="High", commentary="pass 0")
L.lock_peaks(led_i2, ["ino2"])
ICFG = P.PassConfig(height_cutoff=100.0, reagent_element="I")
s_i2 = P._resolve_acid_i2_clusters(None, "S", led_i2, PROF5, ICFG,
                                   score_fn=fake_i2_score, log=lambda *a: None)
check("acid.I2 resolver commits both rungs", s_i2["committed"] == 2, s_i2)
check("formic I2 rung: neutral is the ACID on [M-H+I2]-",
      led_i2.loc[led_i2.peak_id == "t1", "neutral_formula"].iloc[0] == "CH2O2"
      and led_i2.loc[led_i2.peak_id == "t1", "adduct"].iloc[0] == "[M-H+I2]-")
check("acetic rung reached via the CH2 homolog bridge",
      led_i2.loc[led_i2.peak_id == "t2", "neutral_formula"].iloc[0] == "C2H4O2"
      and led_i2.loc[led_i2.peak_id == "t2", "adduct"].iloc[0] == "[M-H+I2]-")
check("commit method is cluster:I2",
      led_i2.loc[led_i2.peak_id == "t1", "method"].iloc[0] == "cluster:I2")
check("commentary names the covalent alias",
      "CHIO2" in str(led_i2.loc[led_i2.peak_id == "t1", "commentary"].iloc[0]))
check("locked pass-0 INO2 is NOT re-read as [HNO2-H+I2]-",
      led_i2.loc[led_i2.peak_id == "ino2", "neutral_formula"].iloc[0] == "INO2"
      and led_i2.loc[led_i2.peak_id == "ino2", "adduct"].iloc[0] == "[M+I]-")
check("unrelated peak stays unexplained", L.role_of(led_i2, "bg") == L.ROLE_UNEXPLAINED)
check("resolver ledger validates clean", L.validate(led_i2) == [], L.validate(led_i2))

# anchors that are NOT acids never propose: iodine-bearing / O-free anchors
led_i2b = mk_ledger([("a1", _CH.ion_mz("HIO3", "[M+I]-"), 50000.0),
                     ("t3", _CH.ion_mz("HIO3", "[M-H+I2]-"), 8000.0)])
L.commit_assignment(led_i2b, "a1", neutral_formula="HIO3", adduct="[M+I]-",
                    ion_formula="HI2O3-", ion_score=0.9, ppm_error=0.1,
                    pass_no=0, method="known:reactive-iodine",
                    confidence="High", commentary="pass 0")
s_i2b = P._resolve_acid_i2_clusters(None, "S", led_i2b, PROF5, ICFG,
                                    score_fn=fake_i2_score, log=lambda *a: None)
check("iodine-bearing anchor (HIO3) never seeds an I2-cluster proposal",
      s_i2b["committed"] == 0
      and L.role_of(led_i2b, "t3") == L.ROLE_UNEXPLAINED, s_i2b)


def test_all():
    assert FAIL == 0, f"{FAIL} checks failed"


if __name__ == "__main__":
    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)
