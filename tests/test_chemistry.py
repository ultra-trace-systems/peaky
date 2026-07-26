"""Offline unit tests for chemistry.py. Run: python3 tests/test_chemistry.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from peaky import chemistry as C  # noqa: E402

PASS = 0
FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok  {name}")
    else:
        FAIL += 1
        print(f"FAIL  {name}  {detail}")


def approx(a, b, tol=1e-4):
    return abs(a - b) <= tol


# --- parse / format round trip ---
check("parse C10H16O4", C.parse_formula("C10H16O4") == {"C": 10, "H": 16, "O": 4})
check("parse with single counts", C.parse_formula("CHN") == {"C": 1, "H": 1, "N": 1})
check("format canonical order",
      C.format_formula({"O": 4, "C": 10, "H": 16}) == "C10H16O4",
      C.format_formula({"O": 4, "C": 10, "H": 16}))
check("parse empty -> {}", C.parse_formula("") == {})

# --- exact masses (literature monoisotopic) ---
check("mass C10H16O4 ~ 200.1049", approx(C.neutral_mass("C10H16O4"), 200.10486, 1e-3),
      C.neutral_mass("C10H16O4"))
check("mass H2O ~ 18.0106", approx(C.neutral_mass("H2O"), 18.0106, 1e-3),
      C.neutral_mass("H2O"))

# --- adduct ion m/z ---
# pinic acid C9H14O4 (186.0892) as [M-H]- -> 185.0819
check("[M-H]- of C9H14O4 ~ 185.0819",
      approx(C.ion_mz("C9H14O4", "[M-H]-"), 185.0819, 1e-3),
      C.ion_mz("C9H14O4", "[M-H]-"))
# nitrate adduct adds NO3 (61.98818) minus electron
check("[M+NO3]- shift ~ +61.98818", approx(C.ADDUCT_SHIFTS["[M+NO3]-"], 61.98818, 1e-3),
      C.ADDUCT_SHIFTS["[M+NO3]-"])
# 15N-labelled nitrate reagent: the cluster adds 15N, so +62.98540, exactly
# 0.997035 Da (the 15N-14N delta) heavier than the 14N adduct above.
check("[M+^NO3]- shift ~ +62.98540",
      approx(C.ADDUCT_SHIFTS["[M+^NO3]-"], 62.98540, 1e-3), C.ADDUCT_SHIFTS["[M+^NO3]-"])
check("15N nitrate is +0.997035 vs 14N",
      approx(C.ADDUCT_SHIFTS["[M+^NO3]-"] - C.ADDUCT_SHIFTS["[M+NO3]-"], 0.997035, 1e-5),
      C.ADDUCT_SHIFTS["[M+^NO3]-"] - C.ADDUCT_SHIFTS["[M+NO3]-"])
check("nitrophenol [M+^NO3]- ion m/z ~ 202.0123 (server-observed)",
      approx(C.ion_mz("C6H5NO3", "[M+^NO3]-"), 202.0123, 2e-3), C.ion_mz("C6H5NO3", "[M+^NO3]-"))
# iodide adducts: [M+I]- adds I (126.9045), and the poly-iodide analyte channels
# [M+I2]-/[M+I3]- add 2/3 iodines (the reagent I2-./I3- clustering onto a neutral).
check("[M+I]- shift ~ +126.9050", approx(C.ADDUCT_SHIFTS["[M+I]-"], 126.9050, 1e-3),
      C.ADDUCT_SHIFTS["[M+I]-"])
check("[M+I2]- shift ~ +253.8095", approx(C.ADDUCT_SHIFTS["[M+I2]-"], 253.8095, 1e-3),
      C.ADDUCT_SHIFTS["[M+I2]-"])
check("[M+I3]- shift ~ +380.7140", approx(C.ADDUCT_SHIFTS["[M+I3]-"], 380.7140, 1e-3),
      C.ADDUCT_SHIFTS["[M+I3]-"])
check("[M+I2]- is one iodine (126.9045) heavier than [M+I]-",
      approx(C.ADDUCT_SHIFTS["[M+I2]-"] - C.ADDUCT_SHIFTS["[M+I]-"], 126.90447, 1e-4),
      C.ADDUCT_SHIFTS["[M+I2]-"] - C.ADDUCT_SHIFTS["[M+I]-"])
# HNO3 seen as the [M+I]- cluster == the server-observed HINO3- at 189.90
check("HNO3 [M+I]- ion m/z ~ 189.9004 (server-observed)",
      approx(C.ion_mz("HNO3", "[M+I]-"), 189.9004, 2e-3), C.ion_mz("HNO3", "[M+I]-"))
# [M+H]+ of H2O -> 19.018
check("[M+H]+ of H2O ~ 19.0178", approx(C.ion_mz("H2O", "[M+H]+"), 19.0178, 1e-3),
      C.ion_mz("H2O", "[M+H]+"))
# di-bromide analyte cluster frames (2026-06-12). [M+Br2]- = M + 2*78.9183 + e;
# [M+HBr+Br]- = M + H + 2*Br + e (one H heavier). The SOA core C15H22O3 seen as
# [M+HBr+Br]- must land on the 409.0015 lattice peak.
check("[M+Br2]- shift ~ +157.8372",
      approx(C.ADDUCT_SHIFTS["[M+Br2]-"], 157.8372, 1e-3), C.ADDUCT_SHIFTS["[M+Br2]-"])
check("[M+HBr+Br]- of C15H22O3 ~ 409.0019 (the lattice peak)",
      approx(C.ion_mz("C15H22O3", "[M+HBr+Br]-"), 409.0019, 2e-3),
      C.ion_mz("C15H22O3", "[M+HBr+Br]-"))
check("[M+HBr+Br]- is one H heavier than [M+Br2]-",
      approx(C.ADDUCT_SHIFTS["[M+HBr+Br]-"] - C.ADDUCT_SHIFTS["[M+Br2]-"], 1.00783, 1e-4))
# deprotonated-acid + I2 cluster frames (2026-07-26). [M-H+I2]- = M - H + 2*I + e
# (one H LIGHTER than [M+I2]-). The unexplained-residual masses of the 2026-07-21
# iodide batch are I2 clusters of ledger acids: formic 298.807, acetic 312.823,
# carbonic 314.802, glycolic 328.818, HNO4 331.792.
check("[M-H+I2]- is one H lighter than [M+I2]-",
      approx(C.ADDUCT_SHIFTS["[M+I2]-"] - C.ADDUCT_SHIFTS["[M-H+I2]-"], 1.00783, 1e-4))
check("[HCOOH-H+I2]- ~ 298.8071 (the residual peak)",
      approx(C.ion_mz("CH2O2", "[M-H+I2]-"), 298.8071, 2e-3),
      C.ion_mz("CH2O2", "[M-H+I2]-"))
check("[HNO4-H+I2]- ~ 331.7922 (the residual peak)",
      approx(C.ion_mz("HNO4", "[M-H+I2]-"), 331.7922, 2e-3),
      C.ion_mz("HNO4", "[M-H+I2]-"))
check("[M-H+I2]- of the acid == [M+I]- of its covalent-I alias (exact degeneracy)",
      approx(C.ion_mz("C2H4O2", "[M-H+I2]-"), C.ion_mz("C2H3IO2", "[M+I]-"), 1e-9))

# --- DBE on the neutral ---
check("DBE C10H16O4 == 3", C.dbe("C10H16O4") == 3, C.dbe("C10H16O4"))   # pinonic-ish
check("DBE benzene C6H6 == 4", C.dbe("C6H6") == 4, C.dbe("C6H6"))
check("DBE CO2 (O=C=O) == 2", C.dbe("CO2") == 2, C.dbe("CO2"))

# --- integer-DBE-on-neutral / nitrate handling ---
# Neutral organic nitrate: C8H15NO12 -> integer DBE, must PASS dbe_ok
ok_neutral, why_n = C.dbe_ok("C8H15NO12")
check("neutral nitrate C8H15NO12 has integer DBE -> ok", ok_neutral, f"{C.dbe('C8H15NO12')} {why_n}")
# The corresponding ION formula C8H14NO12 is half-integer -> must be REJECTED as a neutral
ok_ion, why_i = C.dbe_ok("C8H14NO12")
check("ion-form nitrate C8H14NO12 half-integer -> rejected as neutral",
      not ok_ion, f"{C.dbe('C8H14NO12')} {why_i}")

# --- negative DBE rejected ---
ok_neg, _ = C.dbe_ok("CH6")   # DBE = 1+1-3 = -1
check("negative DBE CH6 rejected", not ok_neg, C.dbe("CH6"))

# --- Senior's rule ---
ok_sen, why_s = C.dbe_ok("C2H2O20")  # absurd O load fine for DBE, but check senior on unsaturation
check("normal molecule passes senior", C.dbe_ok("C10H16O4")[0])

# --- grid: every emitted formula obeys the gates ---
ranges = C.parse_ranges("C0-12 H0-24 O0-8 N0-2")
grid = C.enumerate_grid(ranges, mass_min=30, mass_max=300)
check("grid is non-empty", len(grid) > 100, len(grid))
bad = [f for _, f in grid if not C.dbe_ok(f)[0]]
check("grid: ALL formulas pass dbe_ok (integer DBE>=0, senior)", not bad, bad[:5])
# grid masses are sorted-able and match neutral_mass
mismatch = [(m, f) for m, f in grid if not approx(m, C.neutral_mass(f), 1e-6)]
check("grid: stored mass == neutral_mass(formula)", not mismatch, mismatch[:3])
# a known formula should appear
forms = {f for _, f in grid}
check("grid contains C6H6 (benzene)", "C6H6" in forms)
check("grid contains C9H14O4-sized formulas", any(f.startswith("C9H14O4") for f in forms))

# --- candidates_for_peaks: pinic acid as [M-H]- ---
mz = C.ion_mz("C9H14O4", "[M-H]-")
cands = C.candidates_for_peaks([mz], C.parse_ranges("C0-12 H0-24 O0-8"),
                               ["[M-H]-"], ppm_tolerance=2.0, mass_min=30, mass_max=300)
check("candidates_for_peaks recovers C9H14O4", "C9H14O4" in cands, sorted(cands)[:8])

# --- halogens count as hydrogens in DBE ---
check("DBE CHBr3 == 0 (halogens as H)", C.dbe("CHBr3") == 0, C.dbe("CHBr3"))
check("DBE CCl4 == 0", C.dbe("CCl4") == 0, C.dbe("CCl4"))
check("DBE C6H5Br == 4 (PhBr keeps ring count)", C.dbe("C6H5Br") == 4, C.dbe("C6H5Br"))
check("DBE C6F6 == 4 (perfluorobenzene)", C.dbe("C6F6") == 4, C.dbe("C6F6"))

# --- structural oxygen cap ---
check("O-cap kills C3H5ClO17", not C.oxygen_ok("C3H5ClO17")[0], C.oxygen_ok("C3H5ClO17"))
check("O-cap kills HClO13S", not C.oxygen_ok("HClO13S")[0])
check("O-cap kills CH3ClO13S", not C.oxygen_ok("CH3ClO13S")[0])
check("HOM C10H18O7 passes O-cap", C.oxygen_ok("C10H18O7")[0])
check("HOM dimer C19H28O18 passes O-cap", C.oxygen_ok("C19H28O18")[0])
check("H2SO4 passes O-cap (S skeleton)", C.oxygen_ok("H2SO4")[0])
check("HNO3 passes O-cap", C.oxygen_ok("HNO3")[0])
check("O3 passes O-cap (cap=4 for no skeleton)", C.oxygen_ok("O3")[0])
# grid never emits O-cap violations
grid_o = C.enumerate_grid(C.parse_ranges("C0-4 H0-12 O0-25 N0-1"), 30, 400)
bad_o = [f for _, f in grid_o if not C.oxygen_ok(f)[0]]
check("grid respects structural O-cap", not bad_o, bad_o[:5])

# --- complexity penalty ---
check("CHO formula has zero penalty", C.complexity_penalty("C10H16O4") == 0.0)
check("one N small penalty", abs(C.complexity_penalty("C10H15NO4") - 0.03) < 1e-9,
      C.complexity_penalty("C10H15NO4"))
check("neutral Br hits cap 0.20", C.complexity_penalty("C10H15BrO4") == 0.20,
      C.complexity_penalty("C10H15BrO4"))
check("penalty monotone: CHON < CHOS",
      C.complexity_penalty("C5H10NO3") < C.complexity_penalty("C5H10O3S"))

def test_all():
    assert FAIL == 0, f"{FAIL} checks failed"


if __name__ == "__main__":
    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)
