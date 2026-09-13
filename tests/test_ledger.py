"""Offline unit tests for ledger.py. Run: python3 tests/test_ledger.py"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
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


def raises(f, exc=L.LedgerError):
    try:
        f(); return False
    except exc:
        return True
    except Exception as e:
        print("    (wrong exc)", type(e).__name__, e)
        return False


def fresh():
    peaks = pd.DataFrame({
        "peak_id": ["p1", "p2", "p3", "p4", "p2"],   # p2 duplicated on purpose
        "mz": [185.0819, 186.0852, 187.0, 200.0, 186.0852],
        "height": [1e5, 1.1e3, 5e2, 9e4, 9e2],       # keep the bright p2 row
    })
    return L.new_ledger(peaks)


# --- construction + dedup ---
led = fresh()
check("dedup by peak_id", len(led) == 4, len(led))
check("kept bright p2 (height 1100)", led.loc[led.peak_id == "p2", "height"].iloc[0] == 1100)
check("all roles start unexplained", (led["role"] == L.ROLE_UNEXPLAINED).all())
check("assignment columns present", "ion_score" in led.columns and "alternatives" in led.columns)

# --- commit an M0 assignment ---
L.commit_assignment(
    led, "p1", neutral_formula="C9H14O4", adduct="[M-H]-", ion_score=0.97,
    compound_score=0.95, ppm_error=-0.3, pass_no=1, method="cheminfo",
    confidence="High", commentary="locked pass 1",
    alternatives=[{"formula": "C8H10N O5", "score": 0.81, "reason": "complexity penalty"}],
    isotopologues=[{"label": "13C1", "score": 0.94}, {"label": "13C2", "score": 0.88}])
check("p1 is M0", L.role_of(led, "p1") == L.ROLE_M0)
check("p1 dbe computed", led.loc[led.peak_id == "p1", "dbe"].iloc[0] == 3)
check("p1 alternatives stored as json", "C8H10N" in led.loc[led.peak_id == "p1", "alternatives"].iloc[0])

# --- provenance required (I5) ---
check("commit without commentary raises",
      raises(lambda: L.commit_assignment(
          led, "p4", neutral_formula="C10H16O3", adduct="[M-H]-", ion_score=0.9,
          pass_no=1, method="cheminfo", confidence="High", commentary="")))

# --- attach isotopologue (I2) ---
L.attach_isotopologue(led, "p2", "p1", iso_label="13C1", iso_match_score=0.94)
check("p2 is iso_child of p1", L.role_of(led, "p2") == L.ROLE_ISO
      and led.loc[led.peak_id == "p2", "parent_peak_id"].iloc[0] == "p1")
check("attach to non-M0 parent raises",
      raises(lambda: L.attach_isotopologue(led, "p3", "p4", iso_label="13C1")))
# a satellite names WHAT it is a satellite of, not just which peak_id: the row
# has to be auditable on its own, because that is how the ledger is exported and
# read (P/S corroboration audit, ledger CSV in output/per_file/).
check("iso_child carries its parent's identity (formula + channel)",
      led.loc[led.peak_id == "p2", "parent_neutral_formula"].iloc[0] == "C9H14O4"
      and led.loc[led.peak_id == "p2", "parent_adduct"].iloc[0] == "[M-H]-",
      led.loc[led.peak_id == "p2", ["parent_neutral_formula", "parent_adduct"]].to_dict())

# --- locking + immutability (I3) ---
L.lock_peaks(led, ["p1", "p2"])
check("p1 locked", L.is_locked(led, "p1"))
check("commit to locked peak raises",
      raises(lambda: L.commit_assignment(
          led, "p1", neutral_formula="C10H16O4", adduct="[M-H]-", ion_score=0.99,
          pass_no=2, method="grid", confidence="High", commentary="try overwrite")))

# --- I4: iso_child of locked parent cannot become M0 ---
check("claiming locked iso_child as M0 raises",
      raises(lambda: L.commit_assignment(
          led, "p2", neutral_formula="C9H13O4", adduct="[M-H]-", ion_score=0.99,
          pass_no=2, method="grid", confidence="High", commentary="steal child")))

# --- overwrite escape hatch works ---
L.commit_assignment(led, "p1", neutral_formula="C10H16O4", adduct="[M-H]-",
                    ion_score=0.99, pass_no=2, method="grid", confidence="High",
                    commentary="forced", overwrite=True)
check("overwrite=True bypasses lock", led.loc[led.peak_id == "p1", "neutral_formula"].iloc[0] == "C10H16O4")

# --- validate clean ledger ---
# rebuild a clean, valid ledger
led2 = fresh()
L.commit_assignment(led2, "p1", neutral_formula="C9H14O4", adduct="[M-H]-",
                    ion_score=0.97, pass_no=1, method="cheminfo", confidence="High",
                    commentary="ok")
L.attach_isotopologue(led2, "p2", "p1", iso_label="13C1", iso_match_score=0.94)
problems = L.validate(led2)
check("clean ledger validates", problems == [], problems)

# a stamp that disagrees with the parent's own row would send a reviewer to the
# wrong compound -- as much an I2 violation as a dangling parent_peak_id
led2b = fresh()
L.commit_assignment(led2b, "p1", neutral_formula="C9H14O4", adduct="[M-H]-",
                    ion_score=0.97, pass_no=1, method="cheminfo", confidence="High",
                    commentary="ok")
L.attach_isotopologue(led2b, "p2", "p1", iso_label="13C1", iso_match_score=0.94)
led2b.loc[led2b.peak_id == "p2", "parent_neutral_formula"] = "C8H10N2O"
check("validate flags an iso_child naming the wrong parent formula",
      any("names parent" in p for p in L.validate(led2b)), L.validate(led2b))

# --- validate catches orphan iso_child ---
led3 = fresh()
led3.loc[led3.peak_id == "p3", "role"] = L.ROLE_ISO  # orphan: no parent / parent not M0
problems3 = L.validate(led3)
check("validate flags orphan iso_child", any("iso_child" in p for p in problems3), problems3)

# --- stats ---
s = L.stats(led2)
check("stats counts M0", s["by_role"][L.ROLE_M0] == 1, s["by_role"])
check("stats counts iso_child", s["by_role"][L.ROLE_ISO] == 1, s["by_role"])
check("stats signal fractions sum ~1",
      abs(sum(s["signal_by_role"].values()) - 1.0) < 1e-6, s["signal_by_role"])

# --- clear_assignment ---
led4 = fresh()
L.commit_assignment(led4, "p1", neutral_formula="C9H14O4", adduct="[M-H]-",
                    ion_score=0.7, ppm_error=3.9, pass_no=1, method="grid",
                    confidence="Low", commentary="weak fit")
L.attach_isotopologue(led4, "p2", "p1", iso_label="13C", iso_match_score=0.8)
L.clear_assignment(led4, "p1", reason="mass-gate: z=4.5")
check("clear_assignment demotes M0 to unexplained",
      L.role_of(led4, "p1") == L.ROLE_UNEXPLAINED)
check("clear_assignment wipes formula",
      pd.isna(led4.loc[led4.peak_id == "p1", "neutral_formula"].iloc[0]))
check("clear_assignment records reason",
      "mass-gate" in led4.loc[led4.peak_id == "p1", "commentary"].iloc[0])
check("clear_assignment orphan child also cleared",
      L.role_of(led4, "p2") == L.ROLE_UNEXPLAINED
      and pd.isna(led4.loc[led4.peak_id == "p2", "parent_neutral_formula"].iloc[0]))
check("cleared ledger validates clean", L.validate(led4) == [], L.validate(led4))
check("clear_assignment refuses non-M0",
      raises(lambda: L.clear_assignment(led4, "p3", reason="x")))

# --- displace_to_isotopologue (M0-vs-iso-child arbitration) ---
led5 = fresh()
L.commit_assignment(led5, "p1", neutral_formula="C10H16O5", adduct="[M+Br]-",
                    ion_score=0.87, ppm_error=-0.7, pass_no=1, method="grid",
                    confidence="Good", commentary="parent")
L.commit_assignment(led5, "p2", neutral_formula="C6H11NO7Si", adduct="[M+CO3]-",
                    ion_score=0.6, ppm_error=2.0, pass_no=2, method="gka-series",
                    confidence="Low (series)", commentary="bogus own M0")
L.attach_isotopologue(led5, "p3", "p2", iso_label="13C", iso_match_score=0.7)
L.displace_to_isotopologue(led5, "p2", "p1", iso_label="81Br", iso_match_score=0.95)
check("displaced peak is now iso_child of parent",
      L.role_of(led5, "p2") == L.ROLE_ISO
      and led5.loc[led5.peak_id == "p2", "parent_peak_id"].iloc[0] == "p1")
check("displaced peak formula wiped",
      pd.isna(led5.loc[led5.peak_id == "p2", "neutral_formula"].iloc[0]))
check("displacement audit trail kept",
      "DISPLACED" in led5.loc[led5.peak_id == "p2", "commentary"].iloc[0])
check("grandchild re-parented with combined label",
      led5.loc[led5.peak_id == "p3", "parent_peak_id"].iloc[0] == "p1"
      and led5.loc[led5.peak_id == "p3", "iso_label"].iloc[0] == "13C+81Br")
check("re-parented grandchild names the NEW parent, not the displaced one",
      led5.loc[led5.peak_id == "p3", "parent_neutral_formula"].iloc[0] == "C10H16O5"
      and led5.loc[led5.peak_id == "p2", "parent_neutral_formula"].iloc[0] == "C10H16O5",
      led5.loc[:, ["peak_id", "parent_neutral_formula"]].to_dict("records"))
check("displaced ledger validates clean", L.validate(led5) == [], L.validate(led5))
led6 = fresh()
L.commit_assignment(led6, "p1", neutral_formula="C5H10O", adduct="[M-H]-",
                    ion_score=0.9, ppm_error=0.1, pass_no=1, method="grid",
                    confidence="Good", commentary="m0")
L.commit_assignment(led6, "p2", neutral_formula="C5H9NO", adduct="[M-H]-",
                    ion_score=0.8, ppm_error=0.4, pass_no=1, method="grid",
                    confidence="Good", commentary="m0 locked")
L.lock_peaks(led6, ["p2"])
check("displace refuses a locked peak",
      raises(lambda: L.displace_to_isotopologue(led6, "p2", "p1", iso_label="81Br")))

def test_all():
    assert FAIL == 0, f"{FAIL} checks failed"


if __name__ == "__main__":
    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)
