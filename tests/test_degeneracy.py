"""Offline tests for degeneracy.py. Run: python3 tests/test_degeneracy.py
Uses a TINY element box (fast grid) + an injected calibration so no network and
no 1-2 min full-range build."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from peaky import chemistry as C          # noqa: E402
from peaky import ledger as L             # noqa: E402
from peaky import degeneracy as D         # noqa: E402

PASS = FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok  {name}")
    else:
        FAIL += 1
        print(f"FAIL  {name}  {detail}")


TINY = "C0-14 H0-28 N0-2 O0-7 S0-1 F0-8"

# --- deterministic unit checks --------------------------------------------
check("uncalibrated -> empty", D.measure_degeneracy(
    L.new_ledger(pd.DataFrame({"peak_id": ["a"], "mz": [200.0], "height": [1.0]})),
    cal=None) == {})

# same-ion decomposition alias collapses to one canonical ion
a1 = D._canonical_ion("C2H3BrO2", "[M+Br]-")     # ion C2H3Br2O2
a2 = D._canonical_ion("C2H2O2", "[M+HBr+Br]-")    # +H +Br +Br -> C2H3Br2O2
check("covalent/cluster aliases share one canonical ion", a1 == a2 and a1 is not None, (a1, a2))
# iodide analog with a MIXED-sign adduct: acid.I2 cluster vs covalent [M+I]-
a3 = D._canonical_ion("CH2O2", "[M-H+I2]-")       # -H +I +I -> CHI2O2
a4 = D._canonical_ion("CHIO2", "[M+I]-")          # +I        -> CHI2O2
check("acid.I2 / covalent-I aliases share one canonical ion",
      a3 == a4 and a3 is not None, (a3, a4))

prof = D.relaxed_profile(__import__("peaky.chem.contexts", fromlist=["get_context"]).get_context("ambient-air"))
check("relaxed profile lifts the F cap", prof.max_F >= 10 and prof.max_Si >= 6)

# --- integration: brute-find a near-isobar [M-H]- pair, expect density>=2 ---
from peaky.chem import contexts as X  # noqa: E402
grid = sorted(C._grid_cached(C.parse_ranges(TINY), 30.0, 300.0))
shift = C.ADDUCT_SHIFTS["[M-H]-"]
prof_r = D.relaxed_profile(X.get_context("ambient-air"))
def plausible(fm):
    return X.filter_by_profile(fm, prof_r)[0]
pair = None
for i in range(len(grid) - 1):
    m, f = grid[i]
    if not (150 <= m <= 260) or not plausible(f):
        continue
    for j in range(i + 1, len(grid)):
        m2, f2 = grid[j]
        if m2 - m > 0.003:
            break
        # both plausible, distinct, and < 6 ppm apart (each within 3 ppm of mid)
        if f != f2 and plausible(f2) and abs(m2 - m) / m * 1e6 < 6:
            pair = (m, f, m2, f2)
            break
    if pair:
        break
check("found a near-isobar pair in the tiny grid", pair is not None, "no plausible near-isobar < 6 ppm")

if pair:
    m, f, m2, f2 = pair
    mz = (m + m2) / 2 + shift          # peak between the two ion masses
    led = L.new_ledger(pd.DataFrame({"peak_id": ["P", "Q"], "mz": [mz, 999.0],
                                     "height": [1e4, 1e4]}))
    L.commit_assignment(led, "P", neutral_formula=f, adduct="[M-H]-",
                        ion_formula=f"{f}-", ion_score=0.9, compound_score=0.9,
                        ppm_error=0.0, pass_no=1, method="cheminfo+grid",
                        confidence="Good", commentary="x")
    # window sigma 6 ppm, k 3 -> +/-18 ppm so the <6 ppm pair both fall in
    res = D.measure_degeneracy(led, cal=(0.0, 6.0), box=TINY,
                               adducts=("[M-H]-",), k_sigma=3.0)
    dP = res.get("P", {})
    check("degenerate peak: density >= 2", dP.get("density", 0) >= 2, dP)
    check("degenerate peak: note flags MASS-DEGENERATE",
          "MASS-DEGENERATE" in dP.get("note", ""), dP.get("note"))
    check("note names the competitor formula", f2 in dP.get("note", "") or
          any(f2 in a for a in dP.get("alts", [])), dP)

# a clean low-mass peak (its own [M-H]- mass) -> density 1, 'unique' note
clean_mz = C.neutral_mass("C3H6O2") + shift
led2 = L.new_ledger(pd.DataFrame({"peak_id": ["U"], "mz": [clean_mz], "height": [1e4]}))
L.commit_assignment(led2, "U", neutral_formula="C3H6O2", adduct="[M-H]-",
                    ion_formula="C3H5O2-", ion_score=0.9, compound_score=0.9,
                    ppm_error=0.0, pass_no=1, method="cheminfo+grid",
                    confidence="Good", commentary="x")
resU = D.measure_degeneracy(led2, cal=(0.0, 1.0), box=TINY, adducts=("[M-H]-",))
check("low-mass clean peak: density == 1", resU.get("U", {}).get("density") == 1, resU.get("U"))

# apply_degeneracy stamps the columns
D.apply_degeneracy(led2, cal=(0.0, 1.0), box=TINY, adducts=("[M-H]-",))
check("apply_degeneracy stamps degeneracy_note on M0",
      led2.loc[led2.peak_id == "U", "degeneracy_note"].notna().all())

def test_all():
    assert FAIL == 0, f"{FAIL} checks failed"


if __name__ == "__main__":
    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)
