"""Offline tests for reagents.py. Run: python3 tests/test_reagents.py"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from peaky import reagents as RG  # noqa: E402
from peaky import ledger as L  # noqa: E402

PASS = FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok  {name}")
    else:
        FAIL += 1
        print(f"FAIL  {name}  {detail}")


def near(lib, mz, ppm=10):
    return [lbl for lbl, m, _f in lib if abs(m - mz) / mz * 1e6 <= ppm]


def near_f(lib, mz, ppm=10):
    return [f for _lbl, m, f in lib if abs(m - mz) / mz * 1e6 <= ppm]


# --- library contains Br-, [Br3]- and isotopologues at the right masses ---
lib = RG.build_library("Br")
check("Br- present ~78.9189", bool(near(lib, 78.9189)), near(lib, 78.9189))
# tribromide [Br3]- monoisotopic 3*78.9183 + e = 236.7555
check("[Br3]- present ~236.7555", bool(near(lib, 236.7555)), near(lib, 236.7555))
# isotopologue 79Br2 81Br at ~238.7535
check("[Br3]- 79,79,81 isotopologue ~238.7535", bool(near(lib, 238.7535)), near(lib, 238.7535))
# Br . H2O cluster ~ 78.9189 + 18.0106 = 96.929 (HNO3/HNO2 were removed from
# the cluster library 2026-06-12 -- they are ambient analytes assigned in pass 0)
check("[Br+H2O]- present ~96.929", bool(near(lib, 96.929)), near(lib, 96.929))
check("HNO3 NOT in reagent library (now an analyte)", not near(lib, 141.914))
# di-bromide radical anion Br2-. = 2*78.9183 + e = 157.8372 (user registered it
# on the server 2026-06-12); the labeler must catch bare even-n clusters too
check("[Br2]-. present ~157.8372", bool(near(lib, 157.8372)), near(lib, 157.8372))
# ambient ORGANIC ACIDS were removed from the cluster library: [Br+HCOOH]- =
# [CH2O2+Br]- = the analyte channel (formic acid's 124.92/126.92 giants), so it
# must NOT be a reagent label anymore
check("[Br+HCOOH]- (124.924) NOT in reagent library (it is the [M+Br]- analyte)",
      not near(lib, 124.924), near(lib, 124.924))
check("[Br+pinic]- (267.006) NOT in reagent library", not near(lib, 267.006))
# HBr cluster on the di-bromide core stays reagent (pure halogen, no analyte):
# [Br2+HBr]- = 157.8372 + 79.926 = 237.763
check("[Br+HBr]- (HBr2- ~160.843) still reagent", bool(near(lib, 160.843)))
# [Br+HF]- = BrHF- = 98.9251 -- HF background halogen cluster (v47 time-series ID;
# the only clean fit among the variable-unassigned residual). Both isotopologues.
check("[Br+HF]- (BrHF- ~98.9251) present", bool(near(lib, 98.9251)), near(lib, 98.9251))
check("[Br+HF]- 81Br twin (~100.9231) present", bool(near(lib, 100.9231)))
check("[Br+HF]- carries ion_formula HBrF-", "HBrF-" in near_f(lib, 98.9251))

# --- reagent_for_adducts ---
check("Br reagent from [M+Br]-", RG.reagent_for_adducts(["[M-H]-", "[M+Br]-"]) == "Br")
check("I reagent from [M+I]-", RG.reagent_for_adducts(["[M+I]-"]) == "I")
check("None when no halide reagent", RG.reagent_for_adducts(["[M-H]-", "[M+NO3]-"]) is None)

# --- labeler marks the bright Br3 cluster peaks ---
peaks = pd.DataFrame({
    "peak_id": ["b1", "b3", "b3b", "org"],
    "mz": [78.9189, 236.7555, 238.7535, 257.0181],
    "height": [2e5, 1e5, 9e4, 8e4],
})
led = L.new_ledger(peaks)
n = RG.label_reagents(led, "Br", ppm=15)
check("labels >=3 reagent peaks", n >= 3, n)
check("Br3 peak labeled reagent",
      L.role_of(led, "b3") == L.ROLE_REAGENT, L.role_of(led, "b3"))
check("organic peak NOT labeled reagent",
      L.role_of(led, "org") == L.ROLE_UNEXPLAINED, L.role_of(led, "org"))
check("reagent commentary written",
      "reagent ion" in str(led.loc[led.peak_id == "b3", "commentary"].iloc[0]))
# known formula -> assigned: the reagent row must carry its ion_formula
check("Br3 reagent row records ion_formula Br3-",
      str(led.loc[led.peak_id == "b3", "ion_formula"].iloc[0]) == "Br3-",
      led.loc[led.peak_id == "b3", "ion_formula"].iloc[0])

# --- BOTH BrO- isotopologues present (the 81Br twin at 96.91 was being dropped) ---
check("79BrO- present ~94.9138", bool(near(lib, 94.9138)), near(lib, 94.9138))
check("81BrO- present ~96.9118 (the missed twin)", bool(near(lib, 96.9118)), near(lib, 96.9118))
check("BrO- ion formula recorded as BrO-", "BrO-" in near_f(lib, 96.9118), near_f(lib, 96.9118))
# and the labeler assigns it out of the residual
led2 = L.new_ledger(pd.DataFrame({"peak_id": ["bo"], "mz": [96.9117], "height": [2350.0]}))
RG.label_reagents(led2, "Br", ppm=15)
check("81BrO- peak (96.91) now labeled reagent, not unexplained",
      L.role_of(led2, "bo") == L.ROLE_REAGENT, L.role_of(led2, "bo"))
check("81BrO- peak carries ion_formula BrO-",
      str(led2.loc[led2.peak_id == "bo", "ion_formula"].iloc[0]) == "BrO-")

# --- IODIDE (I⁻ CIMS) reagent library ---------------------------------------
# learned from the 2026-07-21 batch: the In⁻ ladder + IOₓ⁻ + poly-iodide
# source-background clusters. I is monoisotopic (127I only), so no isotope branching.
ilib = RG.build_library("I")
check("I- present ~126.9050", bool(near(ilib, 126.9050)), near(ilib, 126.9050))
# I2-. radical anion = 2*126.9045 + e = 253.8095 (the BRIGHTEST ion in the source)
check("[I2]-. present ~253.8095", bool(near(ilib, 253.8095)), near(ilib, 253.8095))
check("[I3]- present ~380.7140", bool(near(ilib, 380.7140)), near(ilib, 380.7140))
# iodine oxides IOx- are NOT reagent entries: IO-/IO2-/IO3- are ion-identical to
# the [M-H]- deprotonation of HOI/HIO2/HIO3 (iodate = iodic acid's DOMINANT
# channel, the NPF tracer) -- left for pass-0 reactive_iodine, the HNO3 ruling.
check("IO- (142.90) NOT in I library (it is [HOI-H]-)",
      not near(ilib, 142.8999), near(ilib, 142.8999))
check("IO2- (158.89) NOT in I library (it is [HIO2-H]-)",
      not near(ilib, 158.8948), near(ilib, 158.8948))
check("IO3- (174.89) NOT in I library (it is [HIO3-H]-, iodic acid's main line)",
      not near(ilib, 174.8897), near(ilib, 174.8897))
# ... but the Br oxides are untouched by the iodine gate
check("BrO- still in Br library after the IOx ruling", bool(near(lib, 94.9138)))
# I . H2O cluster ~ 126.9050 + 18.0106 = 144.9156
check("[I+H2O]- present ~144.9156", bool(near(ilib, 144.9156)), near(ilib, 144.9156))
# poly-iodide source-background clusters: only the time-STABLE pure iodine oxides
check("[I2O]- present ~269.8044", bool(near(ilib, 269.8044)), near(ilib, 269.8044))
check("[I3O]- present ~396.7089", bool(near(ilib, 396.7089)), near(ilib, 396.7089))
check("[I2O]- carries ion_formula I2O-", "I2O-" in near_f(ilib, 269.8044), near_f(ilib, 269.8044))
# reagent-acid clusters are the [M+I]- ANALYTE channel, NOT reagent (like the Br
# organic-acid ruling): I.HNO3 (189.90), I.H2O2 (160.91), I.HCOOH (172.91) stay out.
check("[I+HNO3]- (189.90) NOT in reagent library (it is the [M+I]- HNO3 analyte)",
      not near(ilib, 189.9007), near(ilib, 189.9007))
check("[I+H2O2]- (160.91) NOT in reagent library (it is the [M+I]- H2O2 analyte)",
      not near(ilib, 160.9105), near(ilib, 160.9105))
check("[I+HCOOH]- (172.91) NOT in reagent library (it is the [M+I]- formic analyte)",
      not near(ilib, 172.9106), near(ilib, 172.9106))
# reactive-iodine AMBIENT species are pass-0 known species, NOT reagent background:
# HOI (via HOI2- 270.81, 55x time-varying) and INO2 (via I2NO2- 299.80) stay out.
check("[HOI2]- (270.81) NOT in reagent library (it is the [M+I]- HOI analyte)",
      not near(ilib, 270.8122), near(ilib, 270.8122))
check("[I2NO2]- (299.80) NOT in reagent library (it is the [M+I]- INO2 analyte)",
      not near(ilib, 299.8024), near(ilib, 299.8024))
# the poly-iodide background is iodine-specific: it must NOT leak into the Br library
check("iodine background NOT in Br library ([I2O]- absent)", not near(lib, 269.8044))
# the shed hydrogen halide is the REAGENT'S OWN: the I library carries [I+HI]-
# (254.8173), NOT the Br-CIMS [I+HBr]- (206.832) phantom
check("[I+HI]- present ~254.8173 (reagent's own hydride)",
      bool(near(ilib, 254.8173)), near(ilib, 254.8173))
check("NO [I+HBr]- phantom (~206.832) in the I library",
      not near(ilib, 206.8318), near(ilib, 206.8318))
check("no Br in any I-library ion formula",
      not any("Br" in f for _l, _m, f in ilib),
      [f for _l, _m, f in ilib if "Br" in f])
check("Br library still carries [Br+HBr]- (160.843)", bool(near(lib, 160.843)))
check("no I in any Br-library ion formula",
      not any("I" in f for _l, _m, f in lib), [f for _l, _m, f in lib if "I" in f])

# --- Cl library: the shed hydride follows the reagent there too ---------------
clib = RG.build_library("Cl")
check("[Cl+HCl]- present ~70.9461", bool(near(clib, 70.9461)), near(clib, 70.9461))
check("[Cl+HCl]- 37Cl isotopologue ~72.9431", bool(near(clib, 72.9431)), near(clib, 72.9431))
check("no Br in any Cl-library ion formula",
      not any("Br" in f for _l, _m, f in clib), [f for _l, _m, f in clib if "Br" in f])
# stability snapshots: pin the library sizes so a future _CLUSTER_NEUTRALS /
# oxide-rule change is a CONSCIOUS edit here, not a silent behavior shift
check("Br library size snapshot (62)", len(lib) == 62, len(lib))
check("Cl library size snapshot (62)", len(clib) == 62, len(clib))
check("I library size snapshot (18: no isotope branching, no IOx oxides)",
      len(ilib) == 18, len(ilib))

# the labeler pulls the bright iodide clusters out of the residual, leaves analytes
iled = L.new_ledger(pd.DataFrame({
    "peak_id": ["i1", "i2", "i3", "i2o", "i2no2", "hno3", "org"],
    "mz": [126.9050, 253.8095, 380.7140, 269.8044, 299.8024, 189.9007, 89.0244],
    "height": [3e6, 1.9e7, 9e6, 2.2e6, 2.8e6, 1.7e4, 2e5],
}))
ni = RG.label_reagents(iled, "I", ppm=15)
check("labels >=4 iodide reagent peaks", ni >= 4, ni)
check("I2-. peak labeled reagent", L.role_of(iled, "i2") == L.ROLE_REAGENT, L.role_of(iled, "i2"))
check("I2O- background peak labeled reagent", L.role_of(iled, "i2o") == L.ROLE_REAGENT,
      L.role_of(iled, "i2o"))
check("I2NO2- (INO2 analyte) NOT labeled reagent -- left for pass-0 reactive iodine",
      L.role_of(iled, "i2no2") == L.ROLE_UNEXPLAINED, L.role_of(iled, "i2no2"))
check("I.HNO3 (HNO3 analyte) NOT labeled reagent -- left for [M+I]- assignment",
      L.role_of(iled, "hno3") == L.ROLE_UNEXPLAINED, L.role_of(iled, "hno3"))
check("organic acid peak NOT labeled reagent",
      L.role_of(iled, "org") == L.ROLE_UNEXPLAINED, L.role_of(iled, "org"))
check("I2-. reagent row records ion_formula I2-",
      str(iled.loc[iled.peak_id == "i2", "ion_formula"].iloc[0]) == "I2-",
      iled.loc[iled.peak_id == "i2", "ion_formula"].iloc[0])

# --- does not touch assigned peaks ---
led2 = L.new_ledger(peaks)
L.commit_assignment(led2, "b3", neutral_formula="C5H8O2", adduct="[M-H]-",
                    ion_score=0.9, pass_no=1, method="x", confidence="High",
                    commentary="real assignment")
RG.label_reagents(led2, "Br", ppm=15)
check("assigned peak not overwritten by reagent labeler",
      L.role_of(led2, "b3") == L.ROLE_M0)

# --- POSITIVE molecular reagent: the urea (uronium) cluster library ----------
ulib = RG.build_library("urea")
# [urea_n + H]+ at 61.0396 / 121.0720 / 181.1044 / 241.1368
check("[urea+H]+ present ~61.0396", bool(near(ulib, 61.0396)), near(ulib, 61.0396))
check("[urea2+H]+ present ~121.0720", bool(near(ulib, 121.0720)), near(ulib, 121.0720))
check("[urea3+H]+ present ~181.1044", bool(near(ulib, 181.1044)), near(ulib, 181.1044))
check("[urea4+H]+ present ~241.1368", bool(near(ulib, 241.1368)), near(ulib, 241.1368))
# spacing of the [urea_n+H]+ series is exactly one urea (60.0324)
uHmasses = sorted(m for _l, m, _f in ulib if _l.endswith("+H]+"))
check("urea [R_n+H]+ cluster spacing ~60.0324",
      abs((uHmasses[1] - uHmasses[0]) - 60.0324) < 1e-3, uHmasses[1] - uHmasses[0])
# ion formulae are CATIONS with the known elemental composition
check("[urea+H]+ ion_formula CH5N2O+", "CH5N2O+" in near_f(ulib, 61.0396), near_f(ulib, 61.0396))
check("[urea2+H]+ ion_formula C2H9N4O2+", "C2H9N4O2+" in near_f(ulib, 121.0720), near_f(ulib, 121.0720))
# ammonia-charged clusters [urea_n + NH4]+: n>=2 are urea-multimer ion-source
# clusters (reagent). n=1 (78.0662) is NOT here -- it is ambient NH3 measured via its
# single urea adduct [NH3+(urea)H]+ (same ion), registered as a pass-0 known species.
check("[urea+NH4]+ (78.0662) NOT in reagent library (it is the ambient-NH3 analyte)",
      not near(ulib, 78.0662), near(ulib, 78.0662))
check("[urea2+NH4]+ present ~138.0985 (urea-multimer NH3 cluster, reagent)",
      bool(near(ulib, 138.0985)), near(ulib, 138.0985))
check("[urea2+NH4]+ ion_formula C2H12N5O2+",
      "C2H12N5O2+" in near_f(ulib, 138.0985), near_f(ulib, 138.0985))

# reagent_for_adducts maps the urea adduct -> 'urea', halogens unchanged
check("reagent_for_adducts urea", RG.reagent_for_adducts(["[M+(CH4N2O)H]+", "[M+H]+"]) == "urea")
check("reagent_for_adducts Br unchanged", RG.reagent_for_adducts(["[M+Br]-", "[M-H]-"]) == "Br")
check("reagent_for_adducts bare positive -> None",
      RG.reagent_for_adducts(["[M+H]+", "[M+Na]+"]) is None,
      RG.reagent_for_adducts(["[M+H]+", "[M+Na]+"]))

# the labeler pulls urea clusters out of the positive residual + records formula
uled = L.new_ledger(pd.DataFrame({
    "peak_id": ["u1", "u2", "u3", "org"],
    "mz": [61.0396, 121.0720, 181.1044, 158.1536],   # org = a real [M+H]+ analyte
    "height": [5e4, 9e4, 4e4, 2.5e5],
}))
nu = RG.label_reagents(uled, "urea", ppm=15)
check("labels >=3 urea reagent peaks", nu >= 3, nu)
check("urea2 peak labeled reagent", L.role_of(uled, "u2") == L.ROLE_REAGENT, L.role_of(uled, "u2"))
check("urea2 reagent row records ion_formula C2H9N4O2+",
      str(uled.loc[uled.peak_id == "u2", "ion_formula"].iloc[0]) == "C2H9N4O2+",
      uled.loc[uled.peak_id == "u2", "ion_formula"].iloc[0])
check("positive analyte peak NOT labeled reagent",
      L.role_of(uled, "org") == L.ROLE_UNEXPLAINED, L.role_of(uled, "org"))
check("labeled urea reagent peaks are LOCKED (a later pass can't overwrite them)",
      bool(uled.loc[uled.peak_id == "u2", "locked"].iloc[0]), "u2 not locked")

# --- reclaim_reagent_clusters: displace an analyte M0 a pass put on a reagent mass
# (the CHNO@61 / CH4N2O@121 == urea [R_n+H]+ degeneracy). The BRIGHT reagent peak
# gets forced to reagent even though pass 1 committed CHNO/CH4N2O onto it.
rled = L.new_ledger(pd.DataFrame({
    "peak_id": ["bright61", "dim61", "dimer121", "realorg"],
    "mz": [61.039624, 61.039077, 121.071931, 158.1536],
    "height": [3.3e6, 1.5e4, 1.6e7, 2.5e5],
}))
# simulate pass-1 committing the reagent-degenerate analyte onto the BRIGHT peaks
L.commit_assignment(rled, "bright61", neutral_formula="CHNO", adduct="[M+NH4]+",
                    ion_formula="CH5N2O+", ion_score=0.93, compound_score=0.93,
                    ppm_error=0.1, pass_no=1, method="cheminfo+grid", confidence="Good", commentary="test")
L.commit_assignment(rled, "dimer121", neutral_formula="CH4N2O", adduct="[M+(CH4N2O)H]+",
                    ion_formula="C2H9N4O2+", ion_score=0.99, compound_score=0.99,
                    ppm_error=0.1, pass_no=1, method="cheminfo+grid", confidence="Good", commentary="test")
out_rc = RG.reclaim_reagent_clusters(rled, "urea", ppm=12, log=lambda *a, **k: None)
check("reclaim: bright 61 M0 phantom displaced to reagent",
      L.role_of(rled, "bright61") == L.ROLE_REAGENT, L.role_of(rled, "bright61"))
check("reclaim: bright 121 dimer M0 phantom displaced to reagent",
      L.role_of(rled, "dimer121") == L.ROLE_REAGENT, L.role_of(rled, "dimer121"))
check("reclaim: reclaimed reagent ions are locked",
      bool(rled.loc[rled.peak_id == "bright61", "locked"].iloc[0]))
check("reclaim: real off-mass analyte untouched",
      L.role_of(rled, "realorg") == L.ROLE_UNEXPLAINED)
check("reclaim: counts (>=2 reagent, >=2 displaced M0)",
      out_rc["reagent"] >= 2 and out_rc["displaced_m0"] >= 2, out_rc)

# --- strip_reagent_cluster_rows: merge-level guard drops reagent ions mislabelled
# as analyte from a merged ledger (no role column; matches by exact ion mass).
merged = pd.DataFrame({
    "mz": [61.039624, 78.066209, 121.071931, 158.1536, 214.0896],
    # 78.066 = ammonia's urea adduct (analyte, NOT a reagent mass) -> kept
    "neutral_formula": ["CHNO", "H3N", "CH4N2O", "C9H18O", "C10H15NO2S"],
    "adduct": ["[M+NH4]+", "[M+(CH4N2O)H]+", "[M+(CH4N2O)H]+", "[M+NH4]+", "[M+H]+"],
    "tier": ["Candidate", "Assigned", "Assigned", "Candidate", "Assigned"],
})
kept, stripped = RG.strip_reagent_cluster_rows(merged, "urea", log=lambda *a, **k: None)
check("strip: reagent monomer(61)/dimer(121) removed; NH3 analyte(78) NOT stripped",
      set(stripped["mz"].round(3)) == {61.040, 121.072}, stripped["mz"].tolist())
check("strip: NH3 analyte + real analytes (158, NBBS@214) kept",
      set(kept["neutral_formula"]) == {"H3N", "C9H18O", "C10H15NO2S"},
      kept["neutral_formula"].tolist())

# --- label_reagent_isotopologues: claim the bright ¹³C/¹⁵N satellites of the reagent
# ions (the urea-dimer ¹³C/¹⁵N at 122.075/122.069 were the 2 biggest 'unexplained'
# peaks). Gated on intensity: a peak far brighter than the satellite (analyte on top)
# is left. Dimer ion C2H9N4O2+ @121.072: ¹³C +1.0034=122.0753, ¹⁵N +0.9970=122.0690.
iled = L.new_ledger(pd.DataFrame({
    "peak_id": ["dimer", "d13c", "d15n", "d18o_analyte"],
    "mz": [121.071931, 122.075286, 122.068966, 123.076241],
    # d18o_analyte: the dimer's ¹⁸O mass, but 5M cps >> the predicted ~64k satellite
    # -> a co-eluting analyte sits on top, so the intensity gate must LEAVE it.
    "height": [1.6e7, 3.3e5, 2.3e5, 5.0e6],
}))
L.mark_reagent(iled, "dimer", "reagent ion: [(CH4N2O)2+H]+", ion_formula="C2H9N4O2+")
out_iso = RG.label_reagent_isotopologues(iled, ppm=12, min_rel=0.004,
                                         log=lambda *a, **k: None)
check("iso-reagent: dimer ¹³C satellite (122.075) claimed as reagent",
      L.role_of(iled, "d13c") == L.ROLE_REAGENT, L.role_of(iled, "d13c"))
check("iso-reagent: dimer ¹⁵N satellite (122.069) claimed as reagent",
      L.role_of(iled, "d15n") == L.ROLE_REAGENT, L.role_of(iled, "d15n"))
check("iso-reagent: claimed satellites are locked",
      bool(iled.loc[iled.peak_id == "d13c", "locked"].iloc[0]))
check("iso-reagent: satellite carries a descriptive commentary",
      "isotopologue" in str(iled.loc[iled.peak_id == "d13c", "commentary"].iloc[0]))
check("iso-reagent: bright analyte-on-top (5M cps at the ¹⁸O mass) NOT stolen",
      L.role_of(iled, "d18o_analyte") == L.ROLE_UNEXPLAINED,
      L.role_of(iled, "d18o_analyte"))
check("iso-reagent: count", out_iso["iso_reagent"] == 2, out_iso)

def test_all():
    assert FAIL == 0, f"{FAIL} checks failed"


if __name__ == "__main__":
    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)
