"""Offline tests for local_scoring.py — the helpers that need no network (the
adduct label -> mechanism spelling, through the library's notation module, and
the category thresholds). The scoring itself is exercised against real data by
scripts/eval_local_scoring.py.
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


# ---- adduct label -> the mechanism string the library scores it as ---------
# The standard adduct notation, which the labels are written in: a label is its
# own spelling, a grouped term is split, terms are in the library's order, and
# a subtraction keeps the charge the label says ('[M-H]-' is the anion; the
# legacy '-H-' the scorer used to be handed reads as the hydride cation now).
cases = {
    "[M+Br]-": "[M+Br]-", "[M-H]-": "[M-H]-", "[M+CO3]-": "[M+CO3]-",
    "[M+H]+": "[M+H]+", "[M+NH4]+": "[M+NH4]+", "[M+(CH4N2O)H]+": "[M+CH4N2O+H]+",
    "[M+^NO3]-": "[M+^NO3]-",                      # 15N-labelled nitrate
    "[M-H]+": "[M-H]+", "[M-CH3]+": "[M-CH3]+",    # the abstraction channels
    "[M]+.": "[M]+.",                              # electron transfer
}
for adduct, mech in cases.items():
    check(f"adduct_to_mech({adduct}) == {mech}", LS.adduct_to_mech(adduct) == mech,
          LS.adduct_to_mech(adduct))

# multi-part adducts keep their terms, in the one order the library writes them
check("multi-add [M+HBr+Br]- -> [M+Br+HBr]-", LS.adduct_to_mech("[M+HBr+Br]-") == "[M+Br+HBr]-",
      LS.adduct_to_mech("[M+HBr+Br]-"))
check("multi-add [M+HBr+CO3]- -> [M+CO3+HBr]-", LS.adduct_to_mech("[M+HBr+CO3]-") == "[M+CO3+HBr]-",
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
