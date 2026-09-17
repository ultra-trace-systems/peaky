"""Offline tests for isotopes.py prescan. Run: python3 tests/test_isotopes.py"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from peaky import isotopes as ISO  # noqa: E402

PASS = FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok  {name}")
    else:
        FAIL += 1
        print(f"FAIL  {name}  {detail}")


def peaks(rows):
    return pd.DataFrame(rows, columns=["mz", "height"])


# --- brominated species: parent + 81Br satellite at ~1:1 ---
br = peaks([(200.0, 1.0e5), (200.0 + ISO.D_81BR, 0.98e5),  # Br pair
            (201.00336, 0.10e5)])                          # a 13C on the parent (~9 C)
r = ISO.prescan(br)
check("detects Br (1:1 pair)", r.has_Br, r.as_dict())
check("Br not flagged multi without triplet", not r.has_multi_Br)
check("13C carbon estimate ~9", 7 <= r.estimated_max_C <= 11, r.estimated_max_C)

# --- Br2 triplet -> multi-Br ---
br2 = peaks([(300.0, 1.0e5), (300.0 + ISO.D_81BR, 1.9e5), (300.0 + 2 * ISO.D_81BR, 0.95e5)])
r2 = ISO.prescan(br2)
check("detects multi-Br triplet", r2.has_Br and r2.has_multi_Br, r2.as_dict())

# --- chlorinated: +1.997 at ~0.32 ---
cl = peaks([(150.0, 1.0e5), (150.0 + ISO.D_37CL, 0.32e5)])
rc = ISO.prescan(cl)
check("detects Cl (~0.32 ratio)", rc.has_Cl and not rc.has_Br, rc.as_dict())

# --- sulfur: +1.9958 at ~0.045 ---
s = peaks([(180.0, 1.0e5), (180.0 + ISO.D_34S, 0.045e5)])
rs = ISO.prescan(s)
check("detects S (~0.045 ratio)", rs.has_S, rs.as_dict())

# --- silicon: +0.99957 at ~0.05 ---
si = peaks([(220.0, 1.0e5), (220.0 + ISO.D_29SI, 0.05e5)])
rsi = ISO.prescan(si)
check("detects Si (~0.05 ratio)", rsi.has_Si, rsi.as_dict())

# --- clean CHO: no heteroatoms flagged (only a 13C satellite) ---
cho = peaks([(185.0, 1.0e5), (185.0 + ISO.D_13C, 0.10e5), (171.0, 5e3)])
rcho = ISO.prescan(cho)
check("clean CHO: no Br/Cl/S/Si", not (rcho.has_Br or rcho.has_Cl or rcho.has_S or rcho.has_Si), rcho.as_dict())

# --- reagent stripping: a bare Br pair declared reagent is ignored ---
rr = ISO.prescan(br, reagent_mzs=[200.0, 200.0 + ISO.D_81BR])
check("reagent peaks stripped -> Br not detected", not rr.has_Br, rr.as_dict())

# --- constrain_ranges: zero out unseen elements, cap C ---
base = {"C": (0, 40), "H": (0, 80), "O": (0, 30), "N": (0, 5),
        "S": (0, 1), "Br": (0, 2), "Cl": (0, 2), "Si": (0, 1)}
caps = {"S": 1, "Br": 2, "Cl": 2, "Si": 1}
constrained = ISO.constrain_ranges(base, r, caps)   # r had Br + C~9
check("Br kept (evidence)", constrained["Br"] == (0, 2), constrained["Br"])
check("Cl zeroed (no evidence)", constrained["Cl"] == (0, 0), constrained["Cl"])
check("S zeroed (no evidence)", constrained["S"] == (0, 0), constrained["S"])
check("Si zeroed (no evidence)", constrained["Si"] == (0, 0), constrained["Si"])
check("C capped near estimate+headroom", constrained["C"][1] <= 15, constrained["C"])

# --- context with Br cap 0 forces Br off even with evidence ---
caps_nobr = {"S": 1, "Br": 0, "Cl": 0, "Si": 0}
c2 = ISO.constrain_ranges(base, r, caps_nobr)
check("context Br cap 0 -> Br zeroed despite evidence", c2["Br"] == (0, 0), c2["Br"])

# --- isotope_pattern: predict the envelope of an ion formula ---
def line(pat, dm, tol=0.01):
    for d, rel, lab in pat:
        if abs(d - dm) <= tol:
            return (rel, lab)
    return None

# 1 Br: M+2 ~0.97, plus 13C; no M+4
p = ISO.isotope_pattern("C15H23BrO3", min_rel=0.05)
check("1-Br has M+2 (81Br) ~0.97", line(p, 1.9978) and abs(line(p, 1.9978)[0] - 0.97) < 0.1,
      line(p, 1.9978))
check("1-Br M+2 labelled 81Br", line(p, 1.9978) and line(p, 1.9978)[1] == "81Br")
check("1-Br has 13C M+1", line(p, 1.0034) is not None)

# 2 Br: M+2 ~1.95, M+4 ~0.95
p2 = ISO.isotope_pattern("C15H23Br2O3", min_rel=0.05)
check("2-Br M+2 ~1.95", line(p2, 1.9978) and abs(line(p2, 1.9978)[0] - 1.95) < 0.2, line(p2, 1.9978))
check("2-Br has M+4 ~0.95", line(p2, 3.9956) and abs(line(p2, 3.9956)[0] - 0.95) < 0.2, line(p2, 3.9956))

# silanediol Si4+Br: merged M+2 ~1.1 (81Br + 30Si), and an M+4
ps = ISO.isotope_pattern("C8H26BrO5Si4", min_rel=0.05)
check("silanediol merged M+2 ~1.1 (Br+Si)", line(ps, 1.9978) and 0.95 <= line(ps, 1.9978)[0] <= 1.3,
      line(ps, 1.9978))
check("silanediol has an M+4 line", line(ps, 3.9946) is not None, ps)

# CHO-only ion: NO M+2 driver -> only 13C lines, so it can never claim a +2 neighbour
pc = ISO.isotope_pattern("C10H15O4", min_rel=0.03)
check("CHO-only has no M+2 (Br/Cl/Si) line",
      line(pc, 1.9978, tol=0.004) is None and line(pc, 1.997, tol=0.004) is None, pc)
check("CHO-only does have 13C", line(pc, 1.0034) is not None)

# --- 81Br mass constant is the true AME2020 value, consistent across modules ---
check("D_81BR is the true 81Br-79Br delta (1.99795, not 1.99780)",
      abs(ISO.D_81BR - 1.9979521) < 1e-6, ISO.D_81BR)
from peaky.assignment import passes as _P  # noqa: E402
check("isotopes D_81BR ~ passes._DBR (same physical line)",
      abs(ISO.D_81BR - _P._DBR) < 5e-4, (ISO.D_81BR, _P._DBR))
check("2x81Br label mass is 2*81Br (no rounding drift)",
      any(lab == "2x81Br" and abs(m - 2 * ISO.D_81BR) < 1e-4
          for m, lab, _ in ISO._LABEL_TABLE),
      [(m, lab) for m, lab, _ in ISO._LABEL_TABLE if "81Br" in lab])

# --- max_shift large enough that a 4-halogen M+8 is not truncated ---
p4 = ISO.isotope_pattern("CBr4", min_rel=0.05, max_shift=12.0)
check("CBr4 keeps its M+8 (~8 Da) line under max_shift=12",
      max(d for d, _, _ in p4) > 7.5, max(d for d, _, _ in p4))

# --- observability: could the diagnostic satellite have been SEEN? -----------
# One predicate behind every "the twin must appear if real" rule (the het-iso
# gate in passes.core, the Si demote in tiers, the reflist rescue's dim branch).
check("34S on one S is ~4.4% of the parent",
      abs(ISO.satellite_height("S", 1, 1.0e4) - 443.0) < 1.0,
      ISO.satellite_height("S", 1, 1.0e4))
check("the line scales with the atom count",
      ISO.satellite_height("S", 3, 1.0e4) == 3 * ISO.satellite_height("S", 1, 1.0e4))
check("a 1e4 cps S parent clears a 100 cps floor (443 >= 100)",
      ISO.satellite_observable("S", 1, 1.0e4, 100.0))
check("a 1e3 cps S parent does NOT (44 < 100) -- absence proves nothing there",
      not ISO.satellite_observable("S", 1, 1.0e3, 100.0))
check("81Br is nearly 1:1, so a Br twin is visible wherever the parent is",
      ISO.satellite_observable("Br", 1, 110.0, 100.0))
check("Si is judged on its BRIGHTER line (29Si 5.1%, not 30Si 3.1%)",
      ISO.DIAG_SATELLITE_RATIO["Si"] == max(ISO.R_29SI_PER_SI, ISO.R_30SI_PER_SI))
check("an element with no minor isotope has no diagnostic line at all",
      not any(e in ISO.DIAG_SATELLITE_RATIO for e in ("N", "P", "I", "F"))
      and ISO.satellite_height("P", 4, 1.0e9) == 0.0)
check("no resolved floor -> nothing can be ruled out",
      ISO.satellite_observable("S", 1, 1.0, None))
check("a NaN / zero / negative height is no line (NaN-safe, never a comparison)",
      ISO.satellite_height("C", 10, float("nan")) == 0.0
      and ISO.satellite_height("C", 10, 0.0) == 0.0
      and ISO.satellite_height("C", 0, 1.0e4) == 0.0)


def test_all():
    assert FAIL == 0, f"{FAIL} checks failed"


if __name__ == "__main__":
    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)
