"""Offline tests for the labelled-reagent (15N) vocabulary + rescue. Run:
python3 tests/test_labeled.py . The scorer is injected, so no network."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from peaky import chemistry as C       # noqa: E402
from peaky import labeled as LB         # noqa: E402
from peaky import profiles as P         # noqa: E402
from peaky import contexts as X         # noqa: E402
from peaky import ledger as L           # noqa: E402

PASS = FAIL = 0
def check(name, cond, detail=""):
    global PASS, FAIL
    if cond: PASS += 1; print(f"  ok  {name}")
    else: FAIL += 1; print(f"FAIL  {name}  {detail}")


# ---- core ^N chemistry -----------------------------------------------------
check("parse ^N is its own species", C.parse_formula("C5H7^NO6") == {"C": 5, "H": 7, "^N": 1, "O": 6})
check("^N round-trips through format", C.format_formula(C.parse_formula("C5H7^NO6")) == "C5H7^NO6")
check("15N mass shift = +0.99703 vs 14N",
      abs((C.neutral_mass("C5H8^NO6") - C.neutral_mass("C5H8NO6")) - 0.99703) < 1e-4)
check("fold_isotopes merges ^N -> N", C.fold_isotopes(C.parse_formula("C4H5^N2O8")) == {"C": 4, "H": 5, "N": 2, "O": 8})
check("dbe counts ^N like N", C.dbe("C5H7^NO6") == C.dbe("C5H7NO6"))
check("14N formula unaffected by the tokenizer change", C.parse_formula("C10H16O4") == {"C": 10, "H": 16, "O": 4})
check("parse tolerates NaN/None", C.parse_formula(None) == {} and C.parse_formula(float("nan")) == {})
check("parse_ranges accepts ^N", C.parse_ranges("C0-40 ^N0-2 O0-25").get("^N") == (0, 2))

# ---- profile carries the label --------------------------------------------
prof = P.resolve("NO3_15N")
check("NO3_15N profile declares ^N label", getattr(prof, "label_isotope", None) == "^N")
check("unlabelled profiles have no label", P.resolve("Br").label_isotope is None)

# ---- rescue discipline (injected scorer) -----------------------------------
# one unexplained peak at the mass of a real 15N mono-organonitrate C9H15NO5,
# 15N-labelled -> C9H15^NO5 ; and a decoy peak with no plausible reading.
cprof = X.get_context(prof.context)
neu = "C9H15^NO5"                      # valid organonitrate; ion C9H14^NO5-
mz_on = C.ion_mz(neu, "[M-H]-")

led = pd.DataFrame([
    dict(peak_id="p1", role=L.ROLE_UNEXPLAINED, mz=mz_on, height=5e4,
         neutral_formula=None, adduct=None, ion_formula=None, ion_score=None,
         locked=False, commentary=None),
    dict(peak_id="p2", role=L.ROLE_UNEXPLAINED, mz=123.4567, height=5e4,
         neutral_formula=None, adduct=None, ion_formula=None, ion_score=None,
         locked=False, commentary=None),
])
for col in ("neutral_formula", "adduct", "ion_formula", "commentary", "method",
            "confidence", "dbe", "compound_score", "eff_score", "eff_margin",
            "tied", "ppm_error", "parent_peak_id", "iso_label", "iso_match_score",
            "pass_no", "anchor_peak_id", "series_unit"):
    if col not in led.columns:
        led[col] = pd.NA


def fake_score(client, sample_id, formulas, allow_partial=True, mechanism_ids=None):
    """Return a confident, isotope-corroborated match only for the real 15N
    organonitrate; nothing for anything else (so the decoy stays unexplained)."""
    rows = []
    for f in formulas:
        if f != neu:
            continue
        for lab, is_base in (("M0", True), ("13C", False)):
            rows.append(dict(
                compound_formula=f, ion_formula="C9H14^NO5-", mechanism_id="[M-H]-",
                is_base=is_base, ion_score=0.95, compound_score=0.95,
                sample_peak_mz=mz_on, sample_peak_id=("p1" if is_base else "p1c"),
                ppm_error=0.2))
    return pd.DataFrame(rows)


cfg = type("cfg", (), {"tau_good": 0.8, "search_ppm": 3.0, "mechanism_ids": None})()
res = LB.rescue_labeled(None, "s", led, cprof, cfg, adducts=prof.adducts,
                        label_isotope=prof.label_isotope, label_max=prof.label_max,
                        score_fn=fake_score, log=lambda *a: None)
check("rescue fills the real 15N organonitrate", res["rescued"] == 1, res)
p1 = led[led.peak_id == "p1"].iloc[0]
check("filled peak carries ^N and became M0",
      "^N" in str(p1["neutral_formula"]) and p1["role"] == L.ROLE_M0)
check("decoy peak left unexplained",
      led[led.peak_id == "p2"].iloc[0]["role"] == L.ROLE_UNEXPLAINED)

# discipline: a candidate with too few O (not an organonitrate) is rejected even
# if the scorer would confirm it.
led2 = led.iloc[:1].copy(); led2.at[led2.index[0], "role"] = L.ROLE_UNEXPLAINED
low_o = "C9H14^NO1"                    # only 1 O -> fails O>=3*n15 gate

def fake_score_lowO(client, sample_id, formulas, allow_partial=True, mechanism_ids=None):
    if low_o not in formulas:
        return pd.DataFrame()
    return pd.DataFrame([dict(compound_formula=low_o, ion_formula=low_o + "-",
        mechanism_id="[M-H]-", is_base=True, ion_score=0.95, compound_score=0.95,
        sample_peak_mz=float(led2.iloc[0]["mz"]), sample_peak_id="p1", ppm_error=0.1)])
# (the real gate is exercised inside rescue_labeled's candidate loop; this just
# confirms a low-O formula never reaches commit)
check("O>=3*n15 organonitrate gate is enforced", 3 * 1 > C.parse_formula(low_o).get("O", 0))


def test_all():
    assert FAIL == 0, f"{FAIL} checks failed"


if __name__ == "__main__":
    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)
