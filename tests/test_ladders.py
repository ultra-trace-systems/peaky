"""Offline tests for ladders.py (anchored gap-fill). Run: python3 tests/test_ladders.py"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from peaky import chemistry as C  # noqa: E402
from peaky import ladders as LAD  # noqa: E402
from peaky import ledger as L  # noqa: E402
from peaky import passes as P  # noqa: E402

PASS = FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok  {name}")
    else:
        FAIL += 1
        print(f"FAIL  {name}  {detail}")


# ---- unit helpers ----
check("_apply +O adds an oxygen", LAD._apply({"C": 15, "H": 24, "O": 2}, {"O": 1}, 1)
      == {"C": 15, "H": 24, "O": 3})
check("_apply +C2H4 x2", LAD._apply({"C": 9, "H": 12, "O": 4}, {"C": 2, "H": 4}, 2)
      == {"C": 13, "H": 20, "O": 4})
check("_apply refuses negative", LAD._apply({"C": 1, "H": 2}, {"C": -1, "H": -2}, 2) is None)
check("_scoreable HBr+Br -> covalent [M-H]-",
      LAD._scoreable("C15H22O4", "[M+HBr+Br]-") == ("C15H24Br2O4", "[M-H]-"))
check("_scoreable [M+Br]- stays native",
      LAD._scoreable("C15H24O3", "[M+Br]-") == ("C15H24O3", "[M+Br]-"))

# ---- isotope-satellite guard ----
import numpy as np  # noqa: E402
mzs = np.array([300.0000, 301.99795, 301.0034])  # base, 81Br line, 13C line
hs = np.array([1.0e4, 9.0e3, 2.0e2])
check("guard flags an 81Br satellite (ratio ~0.9)",
      LAD._is_iso_satellite(mzs, hs, 301.99795, 9.0e3))
check("guard flags a 13C satellite (ratio ~0.02)",
      LAD._is_iso_satellite(mzs, hs, 301.0034, 2.0e2))
check("guard passes a genuine isolated peak",
      not LAD._is_iso_satellite(mzs, hs, 305.0000, 5.0e3))

# ---- end-to-end: the worked C15H22O3 -> O4 di-bromide oxidation gap ----
# the +O rung scores only 0.69 (weak 2-Br envelope), so it commits ONLY via the
# +HBr pairing: its bromine-free neutral C15H22O4 is independently seen at
# [M+Br]- (the 1-Br partner peak).
cfg = P.PassConfig(height_cutoff_cps=100.0)
cfg.cal_mu, cfg.cal_sigma = 0.0, 0.5
prof = type("Prof", (), {"label": "ambient-air"})()
anchor_mz = C.ion_mz("C15H22O3", "[M+HBr+Br]-")    # 409.0019
gap_mz = C.ion_mz("C15H22O4", "[M+HBr+Br]-")        # 424.9969 (the +O rung)
pair_mz = C.ion_mz("C15H22O4", "[M+Br]-")           # 345.07  (the 1-Br partner)
far = 600.0
peaks = pd.DataFrame({"peak_id": ["anc", "gap", "pair", "far"],
                      "mz": [anchor_mz, gap_mz, pair_mz, far],
                      "height": [4719.0, 1184.0, 844.0, 500.0]})


def fresh():
    led = L.new_ledger(peaks)
    L.commit_assignment(led, "anc", neutral_formula="C15H22O3", adduct="[M+HBr+Br]-",
                        ion_formula="C15H23Br2O3-", ion_score=0.93, compound_score=0.93,
                        ppm_error=-1.0, pass_no=4, method="residual:iso-pair",
                        confidence="Good (iso-pair)", commentary="anchor")
    L.commit_assignment(led, "pair", neutral_formula="C15H22O4", adduct="[M+Br]-",
                        ion_formula="C15H22BrO4-", ion_score=0.85, compound_score=0.85,
                        ppm_error=0.2, pass_no=1, method="cheminfo+grid",
                        confidence="Good", commentary="1-Br partner")
    return led


def mock_score(client, sample_id, formulas, mechanism_ids=None):
    cov = "C15H24Br2O4"   # = C15H22O4 [M+HBr+Br]- covalent equiv
    if cov not in formulas:
        return pd.DataFrame()
    return pd.DataFrame([{
        "is_base": True, "sample_peak_id": "gap", "sample_peak_mz": gap_mz,
        "compound_formula": cov, "ion_formula": "C15H23Br2O4-",
        "ion_score": 0.69, "compound_score": 0.69, "ppm_error": -0.6,
        "iso_label": "M0", "iso_score": None}])


led = fresh()
out = LAD.run_ladder_gapfill(None, "S", led, prof, cfg, ["[M+Br]-", "[M+HBr+Br]-"],
                             score_fn=mock_score, log=lambda *a: None)
check("di-bromide +O rung commits via +HBr pairing", out["committed"] == 1, out)
gr = led[led.peak_id == "gap"].iloc[0]
check("gap relabelled to bromine-free C15H22O4 [M+HBr+Br]-",
      gr["neutral_formula"] == "C15H22O4" and gr["adduct"] == "[M+HBr+Br]-")
check("gap method is ladder:gapfill", gr["method"] == "ladder:gapfill")
check("paired di-bromide gap is weak-grade (Low/Suspect)",
      str(gr["confidence"]).startswith(("Low", "Suspect")), gr["confidence"])
check("far peak untouched", L.role_of(led, "far") == "unexplained")
check("ledger validates", L.validate(led) == [], L.validate(led))

# WITHOUT the 1-Br partner, the 0.69 di-bromide gap is rejected (below tau_good)
led_np = L.new_ledger(peaks)
L.commit_assignment(led_np, "anc", neutral_formula="C15H22O3", adduct="[M+HBr+Br]-",
                    ion_formula="C15H23Br2O3-", ion_score=0.93, compound_score=0.93,
                    ppm_error=-1.0, pass_no=4, method="x", confidence="Good", commentary="a")
out_np = LAD.run_ladder_gapfill(None, "S", led_np, prof, cfg, ["[M+HBr+Br]-"],
                                score_fn=mock_score, log=lambda *a: None)
check("unpaired weak di-bromide gap rejected (no overfit)",
      out_np["committed"] == 0 and out_np.get("rejected_score", 0) >= 1, out_np)

# never overwrites an existing M0 (the TFA-overwrite regression guard)
led2 = fresh()
L.commit_assignment(led2, "gap", neutral_formula="C9H4F12", adduct="[M-H]-",
                    ion_score=0.97, compound_score=0.97, ppm_error=0.1, pass_no=1,
                    method="x", confidence="High", commentary="strong fluorinated M0")
out2 = LAD.run_ladder_gapfill(None, "S", led2, prof, cfg, ["[M+Br]-", "[M+HBr+Br]-"],
                              score_fn=mock_score, log=lambda *a: None)
check("existing M0 never overwritten (unexplained-only)",
      led2[led2.peak_id == "gap"].iloc[0]["neutral_formula"] == "C9H4F12"
      and out2["committed"] == 0, out2)

# fluorinated anchor is never a ladder seed (no proposals off F12 contaminants)
ledf = L.new_ledger(pd.DataFrame({"peak_id": ["f", "t"], "mz": [462.9933, 477.009],
                                  "height": [26882.0, 900.0]}))
L.commit_assignment(ledf, "f", neutral_formula="C12H12F12", adduct="[M+Br]-",
                    ion_formula="C12H12BrF12-", ion_score=0.97, compound_score=0.97,
                    ppm_error=-0.8, pass_no=3, method="contaminant:fluorinated",
                    confidence="High (fluorinated)", commentary="PFAS contaminant")
outf = LAD.run_ladder_gapfill(None, "S", ledf, prof, cfg, ["[M+Br]-"],
                              score_fn=mock_score, log=lambda *a: None)
check("fluorinated anchor seeds no ladder", outf["committed"] == 0, outf)

# O>9 monster neighbour is gated out before scoring
ledo = L.new_ledger(pd.DataFrame({"peak_id": ["a", "m"], "mz": [300.0, 700.0],
                                  "height": [5000.0, 800.0]}))
L.commit_assignment(ledo, "a", neutral_formula="C10H16O7", adduct="[M+Br]-",
                    ion_formula="C10H16BrO7-", ion_score=0.9, compound_score=0.9,
                    ppm_error=0.1, pass_no=1, method="x", confidence="Good", commentary="a")
# +CO2 x4 would reach C14H16O15 (O15) -> must be gated; even if a peak existed
check("O>9 neighbour gated (no proposal)",
      "C14H16O15" not in str(LAD.run_ladder_gapfill(
          None, "S", ledo, prof, cfg, ["[M+Br]-"],
          score_fn=lambda *a, **k: pd.DataFrame(), log=lambda *a: None)))

def test_all():
    assert FAIL == 0, f"{FAIL} checks failed"


if __name__ == "__main__":
    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)
