"""Offline tests for local_scoring.py — the pure helpers that need no mascope_tools
or network (adduct-label -> mechanism conversion, category thresholds). The scoring
itself is exercised against real data by scripts/eval_local_scoring.py.
Run: python tests/test_local_scoring.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from peaky import local_scoring as LS  # noqa: E402

PASS = FAIL = 0
def check(name, cond, detail=""):
    global PASS, FAIL
    if cond: PASS += 1; print(f"  ok  {name}")
    else: FAIL += 1; print(f"FAIL  {name}  {detail}")


# ---- adduct label -> mascope_tools mechanism string -------------------------
cases = {
    "[M+Br]-": "+Br-", "[M-H]-": "-H-", "[M+CO3]-": "+CO3-",
    "[M+H]+": "+H+", "[M+NH4]+": "+NH4+", "[M+(CH4N2O)H]+": "+(CH4N2O)H+",
    "[M+^NO3]-": "+^NO3-",                         # 15N-labelled nitrate
}
for adduct, mech in cases.items():
    check(f"adduct_to_mech({adduct}) == {mech}", LS.adduct_to_mech(adduct) == mech,
          LS.adduct_to_mech(adduct))

# multi-part adducts collapse to a single signed group
check("multi-add [M+HBr+Br]- -> +HBrBr-", LS.adduct_to_mech("[M+HBr+Br]-") == "+HBrBr-",
      LS.adduct_to_mech("[M+HBr+Br]-"))
check("multi-add [M+HBr+CO3]- -> +HBrCO3-", LS.adduct_to_mech("[M+HBr+CO3]-") == "+HBrCO3-",
      LS.adduct_to_mech("[M+HBr+CO3]-"))

# mixed +/- decomposition aliases have NO mechanism string BY DESIGN: the
# iodide [M-H+I2]- channel is scored through its covalent alias (A-H+I) [M+I]-
# (pass 3), never sent to a scorer directly. The raise is the contract.
try:
    LS.adduct_to_mech("[M-H+I2]-")
    check("adduct_to_mech([M-H+I2]-) raises (relabel-only channel)", False)
except ValueError:
    check("adduct_to_mech([M-H+I2]-) raises (relabel-only channel)", True)

# unrecognised input raises
try:
    LS.adduct_to_mech("not-an-adduct")
    check("bad adduct raises", False)
except ValueError:
    check("bad adduct raises", True)

# ---- category thresholds (match peaky DEFAULT_MATCH_PARAMS) ------------------
check("_category(0.9) == probable", LS._category(0.9) == "probable")
check("_category(0.5) == possible", LS._category(0.5) == "possible")
check("_category(0.2) == unlikely", LS._category(0.2) == "unlikely")
check("threshold boundary 0.8 -> probable", LS._category(0.8) == "probable")
check("threshold boundary 0.4 -> possible", LS._category(0.4) == "possible")

def test_local_scoring():
    assert FAIL == 0, f"{FAIL} checks failed"


if __name__ == "__main__":
    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)
