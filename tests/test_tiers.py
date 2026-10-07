"""Offline tests for tiers.py. Run: python3 tests/test_tiers.py"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from peaky import ledger as L  # noqa: E402
from peaky import tiers as T  # noqa: E402

PASS = FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok  {name}")
    else:
        FAIL += 1
        print(f"FAIL  {name}  {detail}")


peaks = pd.DataFrame({
    "peak_id": list("ABCDEFGHIJKL"),
    "mz": [200.1, 201.1, 191.0, 510.99, 392.89, 250.0, 251.0, 252.0,
           300.0, 301.0, 555.5, 556.5],
    "height": [1e5, 1e4, 8e4, 1.1e4, 9e2, 5e4, 4e4, 3e4, 2e4, 1e4, 6e3, 5e3]})
led = L.new_ledger(peaks)

# A: High, isotopologue-confirmed, clear margin -> Assigned
L.commit_assignment(led, "A", neutral_formula="C10H16O4", adduct="[M-H]-",
                    ion_formula="C10H15O4-", ion_score=0.97, compound_score=0.96,
                    eff_score=0.95, eff_margin=0.20, tied=False,
                    ppm_error=-0.3, pass_no=1, method="cheminfo+grid",
                    confidence="High", commentary="Pass 1",
                    alternatives=[{"formula": "C9H13NO5", "ion_score": 0.80,
                                   "raw_score": 0.80, "eff_score": 0.75, "ppm": 0.8}],
                    isotopologues=[{"label": "13C", "score": 0.93, "peak_id": "B"}])
L.attach_isotopologue(led, "B", "A", iso_label="13C", iso_match_score=0.93)

# C: Good, NO corroboration, close alternative -> Candidate (density rule)
L.commit_assignment(led, "C", neutral_formula="C7H12O4", adduct="[M-H]-",
                    ion_formula="C7H11O4-", ion_score=0.83, compound_score=0.83,
                    eff_score=0.82, eff_margin=0.07, tied=False,
                    ppm_error=0.5, pass_no=1, method="cheminfo+grid",
                    confidence="Good", commentary="Pass 1",
                    alternatives=[{"formula": "C3H8N2O6", "ion_score": 0.81,
                                   "raw_score": 0.81, "eff_score": 0.75, "ppm": 0.9}])

# D: 'High' O>=12 lattice monster -> Candidate
L.commit_assignment(led, "D", neutral_formula="C14H12N2O19", adduct="[M-H]-",
                    ion_formula="C14H11N2O19-", ion_score=0.92, compound_score=0.92,
                    eff_score=0.90, eff_margin=0.3, tied=False,
                    ppm_error=0.4, pass_no=1, method="cheminfo+grid",
                    confidence="High", commentary="Pass 1",
                    isotopologues=[{"label": "13C", "score": 0.9, "peak_id": "E"}])

# E: mixed Br/Cl neutral -> Candidate (even at Good confidence)
L.commit_assignment(led, "E", neutral_formula="C9H4BrClO2", adduct="[M-H]-",
                    ion_formula="C9H3BrClO2-", ion_score=0.86, compound_score=0.86,
                    ppm_error=-0.2, pass_no=4, method="residual:iso-pair",
                    confidence="Good (iso-pair)", commentary="Pass 4 (iso-pair): BrCl doublet")

# F: known species (pass-0) -> Assigned regardless of alternatives
L.commit_assignment(led, "F", neutral_formula="C4H14O3Si2", adduct="[M+Br]-",
                    ion_formula="C4H14O3Si2.Br-", ion_score=0.95, compound_score=0.94,
                    ppm_error=0.1, pass_no=0, method="known:contaminant:silanediol",
                    confidence="Good (contaminant)", commentary="Pass 0 (known contaminant)")

# G: Low confidence -> Candidate
L.commit_assignment(led, "G", neutral_formula="C5H8O3", adduct="[M+Br]-",
                    ion_formula="C5H8O3.Br-", ion_score=0.62, compound_score=0.62,
                    ppm_error=1.2, pass_no=4, method="residual:series",
                    confidence="Low (deep-series)", commentary="Pass 4")

# H: Good, near-tie via stored arbitration columns -> Candidate
L.commit_assignment(led, "H", neutral_formula="C6H10O4", adduct="[M-H]-",
                    ion_formula="C6H9O4-", ion_score=0.85, compound_score=0.85,
                    eff_score=0.84, eff_margin=0.02, tied=True,
                    ppm_error=0.3, pass_no=1, method="cheminfo+grid",
                    confidence="Good", commentary="Pass 1",
                    alternatives=[{"formula": "C2H6N2O6", "ion_score": 0.84,
                                   "raw_score": 0.84, "eff_score": 0.82, "ppm": 0.4}],
                    isotopologues=[{"label": "13C", "score": 0.8, "peak_id": "I"}])

# I: OLD-LEDGER fallback -- no eff columns, tie only visible in commentary
L.commit_assignment(led, "I", neutral_formula="C8H14O5", adduct="[M-H]-",
                    ion_formula="C8H13O5-", ion_score=0.88, compound_score=0.88,
                    ppm_error=0.2, pass_no=1, method="cheminfo+grid",
                    confidence="Good",
                    commentary="Pass 1. Nearest competitor C4H10N2O7 trails by 0.03 (TIE)",
                    alternatives=[{"formula": "C4H10N2O7", "ion_score": 0.87,
                                   "raw_score": 0.87, "ppm": 0.5}])

# J: Good, cross-channel corroboration (same neutral as A, other adduct),
# close alternative present -> Assigned via corroboration
L.commit_assignment(led, "J", neutral_formula="C10H16O4", adduct="[M+Br]-",
                    ion_formula="C10H16O4.Br-", ion_score=0.84, compound_score=0.84,
                    eff_score=0.80, eff_margin=0.08, tied=False,
                    ppm_error=0.6, pass_no=5, method="completion:known-neutral",
                    confidence="Good (completion)", commentary="Pass 5",
                    alternatives=[{"formula": "C6H12N2O6", "ion_score": 0.82,
                                   "raw_score": 0.82, "eff_score": 0.74, "ppm": 0.7}])

t = T.compute_tiers(led).set_index("peak_id")

check("A High + iso + margin -> Assigned", t.at["A", "tier"] == "Assigned", t.loc["A"].to_dict())
check("A reason mentions isotopologue", "isotopologue" in t.at["A", "tier_reason"])
check("C close alt, no corroboration -> Candidate", t.at["C", "tier"] == "Candidate")
check("D O19 monster -> Candidate despite High", t.at["D", "tier"] == "Candidate")
check("D reason mentions lattice", "lattice" in t.at["D", "tier_reason"])
check("E mixed BrCl -> Candidate", t.at["E", "tier"] == "Candidate")
check("F known species -> Assigned", t.at["F", "tier"] == "Assigned")
check("G Low -> Candidate", t.at["G", "tier"] == "Candidate")
check("H stored near-tie -> Candidate", t.at["H", "tier"] == "Candidate")
check("H reason is the tie", "near-tie" in t.at["H", "tier_reason"], t.at["H", "tier_reason"])
check("I commentary-tie fallback -> Candidate", t.at["I", "tier"] == "Candidate")
check("J cross-channel corroboration -> Assigned", t.at["J", "tier"] == "Assigned",
      t.loc["J"].to_dict())
check("J reason mentions second channel", "second ionization channel" in t.at["J", "tier_reason"])
check("density: A counts winner only when alt is far",
      int(t.at["A", "candidate_density"]) == 1, t.at["A", "candidate_density"])
check("density: H counts the close alt",
      int(t.at["H", "candidate_density"]) == 2, t.at["H", "candidate_density"])

# CSV round-trip: tied becomes the string 'True'/'False'; tiers must agree
import io  # noqa: E402
rt = pd.read_csv(io.StringIO(led.to_csv(index=False)))
t2 = T.compute_tiers(rt).set_index("peak_id")
check("CSV round-trip preserves every tier",
      (t2["tier"] == t["tier"]).all(),
      t2.loc[t2["tier"] != t["tier"], "tier"].to_dict())

# apply_tiers stamps the ledger
T.apply_tiers(led)
m0 = led[led["role"] == "M0"]
check("apply_tiers fills every M0 row", m0["tier"].notna().all())
check("apply_tiers leaves iso/unexplained NA",
      led.loc[led["role"] != "M0", "tier"].isna().all())
check("candidate_density rendered as text", isinstance(m0["candidate_density"].iloc[0], str))

# base_confidence parsing
check("base_confidence strips suffix", T.base_confidence("Good (fluorinated)") == "Good")
check("base_confidence handles NA", T.base_confidence(None) == "")

# --- mass-error-distribution test (Gao 2024) ----------------------------------
# Small fixture above has <20 core peaks -> uncalibrated, rule never fires.
kids0 = led.loc[led["role"] == "MISSING"].set_index("peak_id") if False else \
    led.loc[led["role"] == L.ROLE_ISO, "parent_peak_id"].value_counts()
check("small ledger is uncalibrated (rule inert)",
      T._calibrate(led[led["role"] == "M0"], kids0) is None)

ppm_cycle = [-0.5, -0.6, -0.7, -0.55, -0.65]
N_CORE = 22
ids = [f"K{i:02d}" for i in range(N_CORE)] + ["U1", "X1", "U2", "B1", "B2a", "B2b"]
mzs = [150 + i for i in range(len(ids))]
peaks2 = pd.DataFrame({"peak_id": ids, "mz": mzs,
                       "height": [1e4] * len(ids)})
led2 = L.new_ledger(peaks2)
# 22 corroborated CHO core peaks tightly around -0.6 ppm -> defines calibration
for i in range(N_CORE):
    L.commit_assignment(led2, f"K{i:02d}", neutral_formula=f"C{6 + i}H{10 + i}O3",
                        adduct="[M-H]-", ion_formula=f"C{6 + i}H{9 + i}O3-",
                        ion_score=0.95, compound_score=0.95, eff_score=0.93,
                        eff_margin=0.3, tied=False, ppm_error=ppm_cycle[i % 5],
                        pass_no=1, method="cheminfo+grid", confidence="Good",
                        commentary="core",
                        isotopologues=[{"label": "13C", "score": 0.9, "peak_id": "z"}])
# U1: UNCORROBORATED, unique formula, +0.6 ppm (far off the -0.6 calibration)
L.commit_assignment(led2, "U1", neutral_formula="C10HF11N2O3", adduct="[M-H]-",
                    ion_formula="C10F11N2O3-", ion_score=0.9, compound_score=0.9,
                    eff_score=0.9, eff_margin=0.4, tied=False, ppm_error=0.6,
                    pass_no=3, method="contaminant:fluorinated", confidence="Good",
                    commentary="off-cal uncorroborated")
# X1: same off-calibration ppm but isotopologue-CORROBORATED -> spared
L.commit_assignment(led2, "X1", neutral_formula="C11H8F14", adduct="[M+Br]-",
                    ion_formula="C11H8F14.Br-", ion_score=0.96, compound_score=0.96,
                    eff_score=0.95, eff_margin=0.4, tied=False, ppm_error=0.6,
                    pass_no=3, method="contaminant:fluorinated",
                    confidence="Good", commentary="iso-backed",
                    isotopologues=[{"label": "81Br", "score": 0.9, "peak_id": "z"}])
# U2: uncorroborated but ON calibration (-0.6) -> stays Assigned
L.commit_assignment(led2, "U2", neutral_formula="C8H16O2", adduct="[M-H]-",
                    ion_formula="C8H15O2-", ion_score=0.9, compound_score=0.9,
                    eff_score=0.9, eff_margin=0.4, tied=False, ppm_error=-0.6,
                    pass_no=1, method="cheminfo+grid", confidence="Good",
                    commentary="on-cal uncorroborated")
# B1: background CO3 channel, ON calibration, uncorroborated -> Candidate
# (the background-channel rule fires; mass-error rule is inert here)
L.commit_assignment(led2, "B1", neutral_formula="C9H16O3", adduct="[M+CO3]-",
                    ion_formula="C10H16O6-", ion_score=0.88, compound_score=0.88,
                    eff_score=0.88, eff_margin=0.4, tied=False, ppm_error=-0.5,
                    pass_no=1, method="cheminfo+grid", confidence="Good",
                    commentary="background channel uncorroborated")
# B2a/B2b: same neutral on CO3 AND [M+Br]- -> cross-channel corroborated, CO3 kept
L.commit_assignment(led2, "B2a", neutral_formula="C5H10O", adduct="[M+CO3]-",
                    ion_formula="C6H10O4-", ion_score=0.9, compound_score=0.9,
                    eff_score=0.9, eff_margin=0.4, tied=False, ppm_error=-0.5,
                    pass_no=1, method="cheminfo+grid", confidence="Good",
                    commentary="CO3 but cross-channel")
L.commit_assignment(led2, "B2b", neutral_formula="C5H10O", adduct="[M+Br]-",
                    ion_formula="C5H10O.Br-", ion_score=0.9, compound_score=0.9,
                    eff_score=0.9, eff_margin=0.4, tied=False, ppm_error=-0.5,
                    pass_no=1, method="cheminfo+grid", confidence="Good",
                    commentary="Br partner")
kids2 = led2.loc[led2["role"] == L.ROLE_ISO, "parent_peak_id"].value_counts()
cal = T._calibrate(led2[led2["role"] == "M0"], kids2)
check("calibration computed from core", cal is not None and abs(cal[0] + 0.6) < 0.15, cal)
t3 = T.compute_tiers(led2).set_index("peak_id")
check("U1 off-cal + uncorroborated -> Candidate", t3.at["U1", "tier"] == "Candidate",
      t3.loc["U1"].to_dict())
check("U1 reason cites mass-error-distribution",
      "mass-error-distribution" in t3.at["U1", "tier_reason"], t3.at["U1", "tier_reason"])
check("X1 off-cal but iso-corroborated -> Assigned", t3.at["X1", "tier"] == "Assigned",
      t3.loc["X1"].to_dict())
check("U2 on-cal uncorroborated -> Assigned", t3.at["U2", "tier"] == "Assigned",
      t3.loc["U2"].to_dict())
check("core peaks stay Assigned", (t3.loc[[f"K{i:02d}" for i in range(N_CORE)],
                                            "tier"] == "Assigned").all())
check("B1 background CO3 uncorroborated -> Candidate", t3.at["B1", "tier"] == "Candidate",
      t3.loc["B1"].to_dict())
check("B1 reason cites background air-ion channel",
      "background air-ion channel" in t3.at["B1", "tier_reason"], t3.at["B1", "tier_reason"])
check("B2a CO3 but cross-channel corroborated -> Assigned",
      t3.at["B2a", "tier"] == "Assigned", t3.loc["B2a"].to_dict())

# --- degeneracy-aware tiering (ROADMAP: tiers read degeneracy.py's stamp) -----
# Regression for the v43 contradiction: two cleanup-recovered ions carried
# tier=Assigned "unique formula in the calibrated window" (candidate_density=1,
# i.e. unique inside their NARROW per-pass box) while the honest cross-family
# degeneracy audit stamped degeneracy_density=27 / 12 + a MASS-SATURATED note
# ("not identifiable from accurate mass alone"). Uncorroborated + mass-degenerate
# must cap at Candidate; genuinely corroborated degenerate masses must NOT be
# weakened.
dg_ids = ["R1", "R2", "R3", "R3c", "R4a", "R4b", "R5", "R6"]
dg_mz = [600.0, 350.0, 500.0, 501.0, 450.0, 370.0, 200.0, 250.0]
dg = L.new_ledger(pd.DataFrame({"peak_id": dg_ids, "mz": dg_mz,
                                "height": [1e4] * len(dg_ids)}))

# R1: the C18H14N2O3S [M+Br]- recovered ion -- uncorroborated, density 27
L.commit_assignment(dg, "R1", neutral_formula="C18H14N2O3S", adduct="[M+Br]-",
                    ion_formula="C18H14N2O3S.Br-", ion_score=0.9, compound_score=0.9,
                    ppm_error=-0.06, pass_no=4, method="cleanup:iso-recovery",
                    confidence="Good (recovered)", commentary="recovered")
# R2: the C14H22O4 [M+Br]- recovered ion -- uncorroborated, density 12
L.commit_assignment(dg, "R2", neutral_formula="C14H22O4", adduct="[M+Br]-",
                    ion_formula="C14H22O4.Br-", ion_score=0.9, compound_score=0.9,
                    ppm_error=-0.59, pass_no=4, method="cleanup:iso-recovery",
                    confidence="Good (recovered)", commentary="recovered")
# R3: SAME high degeneracy but isotopologue-CORROBORATED -> must stay Assigned
L.commit_assignment(dg, "R3", neutral_formula="C20H20O5", adduct="[M+Br]-",
                    ion_formula="C20H20O5.Br-", ion_score=0.9, compound_score=0.9,
                    ppm_error=0.1, pass_no=4, method="cleanup:iso-recovery",
                    confidence="Good (recovered)", commentary="iso-backed")
L.attach_isotopologue(dg, "R3c", "R3", iso_label="81Br", iso_match_score=0.9)
# R4: high degeneracy but CROSS-CHANNEL corroborated (same neutral, 2 adducts)
L.commit_assignment(dg, "R4a", neutral_formula="C16H24O5", adduct="[M+Br]-",
                    ion_formula="C16H24O5.Br-", ion_score=0.9, compound_score=0.9,
                    ppm_error=0.2, pass_no=4, method="cleanup:iso-recovery",
                    confidence="Good (recovered)", commentary="ch1")
L.commit_assignment(dg, "R4b", neutral_formula="C16H24O5", adduct="[M-H]-",
                    ion_formula="C16H23O5-", ion_score=0.9, compound_score=0.9,
                    ppm_error=0.2, pass_no=1, method="cheminfo+grid",
                    confidence="Good", commentary="ch2")
# R5: honestly unique (density 1) uncorroborated -> stays Assigned
L.commit_assignment(dg, "R5", neutral_formula="C8H12O3", adduct="[M+Br]-",
                    ion_formula="C8H12O3.Br-", ion_score=0.9, compound_score=0.9,
                    ppm_error=0.1, pass_no=1, method="cheminfo+grid",
                    confidence="Good", commentary="clean")
# R6: density 2 (one rival) uncorroborated -> below the >2 demote bar, Assigned
L.commit_assignment(dg, "R6", neutral_formula="C9H14O3", adduct="[M+Br]-",
                    ion_formula="C9H14O3.Br-", ion_score=0.9, compound_score=0.9,
                    ppm_error=0.1, pass_no=1, method="cheminfo+grid",
                    confidence="Good", commentary="borderline")

# stamp degeneracy_density / degeneracy_note exactly as degeneracy.apply_degeneracy would
for col in ("degeneracy_density", "degeneracy_note"):
    dg[col] = pd.Series(pd.NA, index=dg.index, dtype="object")
SAT = ("MASS-SATURATED: {n} plausible formulas (≤3 heteroatom types) within ±3σ "
       "calibrated window — not identifiable from accurate mass alone; needs "
       "isotope envelope / time-series corroboration")
DEG = ("MASS-DEGENERATE: {n} plausible ions within ±3σ calibrated window — "
       "competitors: ...")


def _stamp_degen(pid, density, note):
    i = dg.index[dg.peak_id == pid][0]
    dg.at[i, "degeneracy_density"] = density
    dg.at[i, "degeneracy_note"] = note


_stamp_degen("R1", 27, SAT.format(n=27))
_stamp_degen("R2", 12, SAT.format(n=12))
_stamp_degen("R3", 27, SAT.format(n=27))
_stamp_degen("R4a", 19, SAT.format(n=19))
_stamp_degen("R5", 1, "unique within ±3σ calibrated mass window")
_stamp_degen("R6", 2, DEG.format(n=2))

td = T.compute_tiers(dg).set_index("peak_id")

check("R1 (C18H14N2O3S, density 27) uncorroborated -> Candidate, not Assigned",
      td.at["R1", "tier"] == "Candidate", td.loc["R1"].to_dict())
check("R1 reason cites the mass degeneracy",
      "mass-degenerate" in td.at["R1", "tier_reason"]
      and "27" in td.at["R1", "tier_reason"], td.at["R1", "tier_reason"])
check("R1 reason says not identifiable from accurate mass alone",
      "not identifiable from accurate mass alone" in td.at["R1", "tier_reason"],
      td.at["R1", "tier_reason"])
check("R2 (C14H22O4, density 12) uncorroborated -> Candidate, not Assigned",
      td.at["R2", "tier"] == "Candidate", td.loc["R2"].to_dict())
check("R3 same density but isotopologue-corroborated -> stays Assigned",
      td.at["R3", "tier"] == "Assigned", td.loc["R3"].to_dict())
check("R3 reason mentions isotopologue (corroboration kept it)",
      "isotopologue" in td.at["R3", "tier_reason"], td.at["R3", "tier_reason"])
check("R4a high density but cross-channel corroborated -> stays Assigned",
      td.at["R4a", "tier"] == "Assigned", td.loc["R4a"].to_dict())
check("R5 honestly unique (density 1) -> Assigned",
      td.at["R5", "tier"] == "Assigned", td.loc["R5"].to_dict())
check("R5 reason is the unique-window text",
      "unique formula in the calibrated window" in td.at["R5", "tier_reason"],
      td.at["R5", "tier_reason"])
check("R6 density 2 (below >2 bar) -> stays Assigned (no over-demotion)",
      td.at["R6", "tier"] == "Assigned", td.loc["R6"].to_dict())

# CSV round-trip: degeneracy_density -> '27.0' string; verdict must be unchanged
rtd = pd.read_csv(io.StringIO(dg.to_csv(index=False)))
td2 = T.compute_tiers(rtd).set_index("peak_id")
check("CSV round-trip: R1/R2 stay Candidate",
      td2.at["R1", "tier"] == "Candidate" and td2.at["R2", "tier"] == "Candidate",
      {"R1": td2.at["R1", "tier"], "R2": td2.at["R2", "tier"]})
check("CSV round-trip: R3/R4a stay Assigned",
      td2.at["R3", "tier"] == "Assigned" and td2.at["R4a", "tier"] == "Assigned",
      {"R3": td2.at["R3", "tier"], "R4a": td2.at["R4a", "tier"]})

# a ledger with NO degeneracy columns (predates the audit): rule inert, R5-like
# unique row stays Assigned (the helper must not raise on a missing column)
dg_old = dg.drop(columns=["degeneracy_density", "degeneracy_note"])
td3 = T.compute_tiers(dg_old).set_index("peak_id")
check("no degeneracy columns -> rule inert (R1 falls back to Assigned)",
      td3.at["R1", "tier"] == "Assigned", td3.loc["R1"].to_dict())

# --- flag_below_assignability: O>=11 AND mass-saturated -> not a confident formula
ba = pd.DataFrame([
    dict(peak_id="mon", role="M0", neutral_formula="C20H20O29", degeneracy_density=52,
         degeneracy_note="MASS-SATURATED: 52 plausible formulas", tier="Candidate", tier_reason="x"),
    dict(peak_id="cho", role="M0", neutral_formula="C10H16O4", degeneracy_density=1,
         degeneracy_note="unique", tier="Assigned", tier_reason="y"),
    dict(peak_id="hiO_uniq", role="M0", neutral_formula="C10H16O11", degeneracy_density=1,
         degeneracy_note="unique", tier="Candidate", tier_reason="z"),
])
nba = T.flag_below_assignability(ba)
check("below-assign: O>=11 + mass-saturated flagged",
      bool(ba.loc[ba.peak_id == "mon", "below_assignability"].iloc[0]))
check("below-assign: sane CHO not flagged",
      not bool(ba.loc[ba.peak_id == "cho", "below_assignability"].iloc[0]))
check("below-assign: O>=11 but UNIQUE not flagged (needs the degeneracy too)",
      not bool(ba.loc[ba.peak_id == "hiO_uniq", "below_assignability"].iloc[0]))
check("below-assign: count==1", nba == 1, nba)

# ---------- calibration is offset-tolerant (uronium -2.4 ppm) ----------
# the backbone of a -2.4 ppm source must still calibrate the tier engine: the
# old |ppm|<=2 cut excluded every backbone row and left the mass-error gate off.
m0_off = pd.DataFrame([
    dict(peak_id=f"o{i}", neutral_formula="C8H18O3", adduct="[M+H]+",
         confidence="Good", ppm_error=-2.4 + (0.2 if i % 2 else -0.2),
         isotopologues=None)
    for i in range(30)])
kids_off = pd.Series({f"o{i}": 1 for i in range(30)})   # each has an iso child
cal_off = T._calibrate(m0_off, kids_off)
check("tiers _calibrate finds the -2.4 ppm center (offset-tolerant)",
      cal_off is not None and abs(cal_off[0] + 2.4) < 0.2
      and cal_off[1] < 0.6, cal_off)
# a gross +6 ppm monster mixed in does not pull the robust median off the core
m0_mix = pd.concat([m0_off, pd.DataFrame([
    dict(peak_id="big", neutral_formula="C9H20O3", adduct="[M+H]+",
         confidence="Good", ppm_error=6.0, isotopologues=None)])],
    ignore_index=True)
kids_mix = pd.concat([kids_off, pd.Series({"big": 1})])
cal_mix = T._calibrate(m0_mix, kids_mix)
check("tiers _calibrate median robust to a gross outlier",
      cal_mix is not None and abs(cal_mix[0] + 2.4) < 0.3, cal_mix)
# --- the stamped ppm_error_cal is the CONSTANT-centre view, by design ---------
# A backbone that DOES accept a masscal mass trend (ppm = a + b*1000/mz, here
# b = -0.12 mDa, so -2.24 ppm at m/z 61 and -0.56 at m/z 390). The gates judge
# such a peak against the centre at ITS m/z; the displayed/exported column
# deliberately removes only the constant offset, so the QC panel can still show
# the residual drift the trend models. Nothing reads a trend off ledger.attrs.
tr_mz = [61, 66, 72, 78, 85, 92, 99] + [115, 130, 150, 170, 190, 210, 230,
                                        250, 270, 290, 310, 330, 350, 370, 390]
tr_ppm = [-0.25 - 0.12 * 1000 / m + (0.02 if i % 2 else -0.02)
          for i, m in enumerate(tr_mz)]
tr_ids = [f"T{i:02d}" for i in range(len(tr_mz))]
led_tr = L.new_ledger(pd.DataFrame({"peak_id": tr_ids, "mz": tr_mz,
                                    "height": [1e4] * len(tr_ids)}))
for i, pid in enumerate(tr_ids):
    L.commit_assignment(led_tr, pid, neutral_formula=f"C{5 + i}H{10 + i}O3",
                        adduct="[M+H]+", ion_formula=f"C{5 + i}H{11 + i}O3+",
                        ion_score=0.95, compound_score=0.95, eff_score=0.93,
                        eff_margin=0.3, tied=False, ppm_error=tr_ppm[i],
                        pass_no=1, method="cheminfo+grid", confidence="Good",
                        commentary="trend backbone",
                        isotopologues=[{"label": "13C", "score": 0.9, "peak_id": "z"}])
kids_tr = led_tr.loc[led_tr["role"] == L.ROLE_ISO, "parent_peak_id"].value_counts()
cal_tr = T._calibrate(led_tr[led_tr["role"] == "M0"], kids_tr)
check("this backbone does accept a mass trend",
      cal_tr is not None and cal_tr.b is not None and abs(cal_tr.b + 0.12) < 0.02,
      getattr(cal_tr, "b", None))
mu_tr, _sig_tr = T.stamp_calibrated_ppm(led_tr)
_low = led_tr.loc[led_tr["peak_id"] == "T00"].iloc[0]          # m/z 61
_trend_centre = cal_tr.a + cal_tr.b * 1000 / 61                # ~= -2.22 ppm
check("ppm_error_cal removes the CONSTANT centre even when a trend was accepted",
      abs(_low["ppm_error_cal"] - (_low["ppm_error"] - mu_tr)) < 1e-9
      and abs(_low["ppm_error_cal"] - (_low["ppm_error"] - _trend_centre)) > 1.0,
      (_low["ppm_error"], _low["ppm_error_cal"], mu_tr, _trend_centre))
check("stamp_calibrated_ppm stashes only the centre the column used",
      {k for k in led_tr.attrs if k.startswith("cal_")} == {"cal_mu", "cal_sigma"},
      dict(led_tr.attrs))


# ---- the reagent-N isobar flag fires on BOTH sides of the pair --------------
# The donor side (an N-free neutral on an N-donating adduct) was always flagged;
# the amine side -- the protonated N-richer neutral, with the same-ion N-poorer
# alternative on the donor adduct -- reached Assigned as "unique formula in the
# calibrated window" once the alias was dropped, with nothing discriminating it
# (C5H12N2S [M+H]+ Assigned in 2 files at 133.079 against C5H9NS [M+NH4]+ in 10).
_rn = L.new_ledger(pd.DataFrame({"peak_id": ["am", "am2", "dn"],
                                 "mz": [133.0794, 193.1088, 133.0794],
                                 "height": [5e4, 2e4, 5e4]}))


def _commit_rn(pid, nf, ad, alt_nf, alt_ad):
    L.commit_assignment(_rn, pid, neutral_formula=nf, adduct=ad, ion_formula="C5H13N2S+",
                        ion_score=0.95, compound_score=0.95, eff_score=0.9, eff_margin=0.3,
                        tied=False, ppm_error=0.2, pass_no=1, method="cheminfo+grid",
                        confidence="High", commentary="Pass 1",
                        alternatives=[{"formula": alt_nf, "adduct": alt_ad, "ion_score": 0.95,
                                       "raw_score": 0.95, "eff_score": 0.9, "ppm": 0.2}])


_commit_rn("am", "C5H12N2S", "[M+H]+", "C5H9NS", "[M+NH4]+")     # the amine side, one channel
_commit_rn("dn", "C5H9NS", "[M+NH4]+", "C5H12N2S", "[M+H]+")     # the donor side, one channel
check("_reagent_n_isobar: the amine side is flagged, naming its alias",
      T._reagent_n_isobar(_rn[_rn.peak_id == "am"].iloc[0], [{"formula": "C5H9NS", "adduct": "[M+NH4]+"}])
      == ("amine", "C5H9NS", "[M+NH4]+"))
check("_reagent_n_isobar: the donor side is flagged as before",
      T._reagent_n_isobar(_rn[_rn.peak_id == "dn"].iloc[0], [{"formula": "C5H12N2S", "adduct": "[M+H]+"}])
      == ("donor", "C5H12N2S", "[M+H]+"))
check("_reagent_n_isobar: negative mode / a different ion / an N-richer alias on [M+H]+ -> None",
      T._reagent_n_isobar({"neutral_formula": "C5H12N2S", "adduct": "[M-H]-"},
                          [{"formula": "C5H9NS", "adduct": "[M+Br]-"}]) is None
      and T._reagent_n_isobar(_rn[_rn.peak_id == "am"].iloc[0],
                              [{"formula": "C6H12O2", "adduct": "[M+H]+"}]) is None
      and T._reagent_n_isobar({"neutral_formula": "C5H9NS", "adduct": "[M+H]+"},
                              [{"formula": "C5H12N2S", "adduct": "[M+H]+"}]) is None)
_trn = T.compute_tiers(_rn).set_index("peak_id")
check("amine-side winner with a single channel -> Candidate, the alias named on the row",
      _trn.at["am", "tier"] == "Candidate"
      and _trn.at["am", "tier_reason"] == ("reagent-N isobar unresolved: C5H12N2S [M+H]+ is the same "
                                           "ion as C5H9NS [M+NH4]+ (the N-poorer neutral on an N-donating "
                                           "reagent adduct), and no second channel / series anchor fixes "
                                           "the nitrogen count (isotopes cannot — identical ion)"),
      _trn.at["am", "tier_reason"])
check("donor-side winner with a single channel -> Candidate, the wording unchanged",
      _trn.at["dn", "tier"] == "Candidate" and "N-heavier" in _trn.at["dn", "tier_reason"],
      _trn.at["dn", "tier_reason"])
# a second channel of the protonated neutral fixes the nitrogen count on the amine side
L.commit_assignment(_rn, "am2", neutral_formula="C5H12N2S", adduct="[M+(CH4N2O)H]+",
                    ion_formula="C6H17N4OS+", ion_score=0.9, compound_score=0.9, eff_score=0.85,
                    eff_margin=0.3, tied=False, ppm_error=0.1, pass_no=5,
                    method="completion:known-neutral", confidence="Good", commentary="Pass 5")
_trn2 = T.compute_tiers(_rn).set_index("peak_id")
check("amine-side winner with its urea channel too -> Assigned, the row says which channel fixed it",
      _trn2.at["am", "tier"] == "Assigned"
      and _trn2.at["am", "tier_reason"] == ("reagent-N isobar: nitrogen count fixed by a second "
                                            "ionization channel of the protonated neutral"),
      _trn2.at["am", "tier_reason"])


# ---- the diagnostic-satellite verdict on every committed Br / Cl / S row -------------
# (assignment/satellites.py, read by the tier engine): a line the file could show and
# did not refutes the count whatever else corroborates the row; a line the file could
# not show leaves it untested, and an untested count with nothing else is Candidate.
from peaky.chem import isotopes as _ISO  # noqa: E402
from peaky.assignment.passes.config import PassConfig as _PC  # noqa: E402

_sat_peaks = pd.DataFrame([
    ("R", 300.0, 2000.0),                       # bromo-organic, 81Br line ABSENT (pred 1945 cps)
    ("Ranc", 314.0157, 1500.0),                 # its CH2 series partner (an anchor for R)
    ("S1", 320.0, 100.0),                       # organosulfate, 34S line predicted 4.4 cps: untestable
    ("S2", 340.0, 100.0),                       # same, but the neutral also sits on a second channel
    ("S2b", 419.9, 60.0),
    ("S3", 360.0, 5000.0),                      # organosulfate with its 34S line PRESENT (pred 222)
    ("S3t", 360.0 + _ISO.D_34S, 200.0),
    ("M", 380.0, 2000.0),                       # bromo-organic ON the bromide adduct: masked
    ("Mt", 380.0 + _ISO.D_81BR, 1900.0),
    ("K", 400.0, 2000.0),                       # a known-species commit: never re-judged here
], columns=["peak_id", "mz", "height"])
_sat = L.new_ledger(_sat_peaks)


def _c(pid, nf, ad, ion, **kw):
    a = dict(neutral_formula=nf, adduct=ad, ion_formula=ion, ion_score=0.9, compound_score=0.9,
             eff_score=0.9, eff_margin=0.3, tied=False, ppm_error=0.1, pass_no=1,
             method="cheminfo+grid", confidence="High", commentary="Pass 1")
    a.update(kw)
    L.commit_assignment(_sat, pid, **a)


_c("R", "C12H17BrO2", "[M-H]-", "C12H16BrO2-", method="residual:series", anchor_peak_id="Ranc", series_unit="CH2")
_c("S1", "C10H16O5S", "[M-H]-", "C10H15O5S-")
_c("S2", "C11H18O5S", "[M-H]-", "C11H17O5S-")
_c("S2b", "C11H18O5S", "[M+Br]-", "C11H18O5S.Br-", pass_no=5, method="completion:known-neutral")
_c("S3", "C12H20O5S", "[M-H]-", "C12H19O5S-")
_c("M", "C13H19BrO3", "[M+Br]-", "C13H19BrO3.Br-")
_c("K", "C14H21BrO2", "[M-H]-", "C14H20BrO2-", pass_no=0, method="known:contaminant:x")
_ts = T.compute_tiers(_sat, cfg=_PC(height_cutoff_cps=10.0)).set_index("peak_id")

check("satellite: an absent 81Br line predicted at 195x the floor refutes the Br count, anchor or not",
      _ts.at["R", "tier"] == "Candidate" and _ts.at["R", "tier_reason"].startswith("Br count refuted by its isotope envelope")
      and "no 81Br line" in _ts.at["R", "tier_reason"], _ts.at["R", "tier_reason"])
check("satellite: a 34S line predicted under 4x the floor is untestable; with nothing else it is Candidate",
      _ts.at["S1", "tier"] == "Candidate" and _ts.at["S1", "tier_reason"].startswith("S1 untestable at this intensity")
      and "for want of evidence" in _ts.at["S1", "tier_reason"], _ts.at["S1", "tier_reason"])
check("satellite: the same untestable S with a second channel stays Assigned and says the count is untested",
      _ts.at["S2", "tier"] == "Assigned" and "second ionization channel" in _ts.at["S2", "tier_reason"]
      and "S count untested" in _ts.at["S2", "tier_reason"], _ts.at["S2", "tier_reason"])
check("satellite: a present, consistent 34S line supports -- Assigned, and the row says the line is there",
      _ts.at["S3", "tier"] == "Assigned" and "S envelope line present" in _ts.at["S3", "tier_reason"],
      _ts.at["S3", "tier_reason"])
check("satellite: a bromo-organic on the bromide adduct is masked by the reagent's own 81Br line -> untestable, not refuted",
      _ts.at["M", "tier"] == "Candidate" and "untestable" in _ts.at["M", "tier_reason"]
      and "reagent adduct's 81Br line" in _ts.at["M", "tier_reason"], _ts.at["M", "tier_reason"])
check("satellite: a known-species commit is never re-judged by the tier's twin test",
      _ts.at["K", "tier"] == "Assigned" and "known species" in _ts.at["K", "tier_reason"], _ts.at["K", "tier_reason"])
_ts0 = T.compute_tiers(_sat).set_index("peak_id")
check("satellite: with no resolved floor nothing is refuted or capped as untestable (nothing can be ruled out)",
      _ts0.at["R", "tier"] == "Assigned" and _ts0.at["S1", "tier"] == "Assigned", (_ts0.at["R", "tier_reason"], _ts0.at["S1", "tier_reason"]))
# CSV round trip: the verdict is recomputed from the ledger, so it survives a string round trip
_ts_rt = T.compute_tiers(pd.read_csv(io.StringIO(_sat.to_csv(index=False))), cfg=_PC(height_cutoff_cps=10.0)).set_index("peak_id")
check("satellite: CSV round-trip keeps the refuted / untestable verdicts",
      _ts_rt.at["R", "tier"] == "Candidate" and _ts_rt.at["S1", "tier"] == "Candidate" and _ts_rt.at["S3", "tier"] == "Assigned")


# --- pd.NA in the string columns must not blow up the tier pass ----------------
# A ledger reaches compute_tiers with NULLABLE dtypes whenever rows were appended
# without a column, or when it was read back with nullable string dtypes.
# `str(v or "")` raises on pd.NA -- "boolean value of NA is ambiguous" -- which killed
# `peaky batch` on the NO3_15N profile while the single-sample path (float NaN,
# falsy) survived. tiers._txt has to absorb both.
check("_txt maps every missing flavour to ''",
      (T._txt(pd.NA), T._txt(float("nan")), T._txt(None)) == ("", "", ""),
      (T._txt(pd.NA), T._txt(float("nan")), T._txt(None)))
check("_txt leaves a real value alone", T._txt("occurrence") == "occurrence")

_na = led.copy()
for _col, _val in (("admitted_by", "occurrence"), ("method", "pass1"),
                   ("commentary", "note")):
    _na[_col] = pd.array([_val] + [pd.NA] * (len(_na) - 1), dtype="string")
_na["neutral_formula"] = pd.array(
    ["C10H16O2"] + [pd.NA] * (len(_na) - 1), dtype="string")
try:
    _t_na = T.compute_tiers(_na)              # must not raise; rows may be dropped
    _ok, _why = "tier" in _t_na.columns and len(_t_na) > 0, list(_t_na.columns)
except TypeError as exc:                      # the regression
    _ok, _why = False, repr(exc)
check("compute_tiers survives pd.NA in admitted_by/method/commentary/formula",
      _ok, _why)


def test_all():
    assert FAIL == 0, f"{FAIL} checks failed"


if __name__ == "__main__":
    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


def test_persist_only_read_is_null_safe():
    """`admitted_by` is float NaN on a row the admission gate never stamped and
    pd.NA on an offline ledger (the scorecard's decoy arms). The old read
    `str(r.get("admitted_by") or "")` gave "nan" on the first (False by accident)
    and RAISED on the second; both must tier like an unstamped row."""
    import numpy as np
    from peaky.assignment import ledger as L
    from peaky.assignment import tiers as T
    rows = []
    for pid, adm in (("a", np.nan), ("b", pd.NA), ("c", None), ("d", "occurrence"), ("e", "height")):
        rows.append(dict(peak_id=pid, mz=200.0 + len(rows), height=1000.0))
    led = L.new_ledger(pd.DataFrame(rows))
    led["admitted_by"] = [np.nan, pd.NA, None, "occurrence", "height"]
    for pid in "abcde":
        L.commit_assignment(led, pid, neutral_formula="C10H16O4", adduct="[M-H]-", ion_formula="C10H15O4-",
                            ion_score=0.9, compound_score=0.9, ppm_error=0.1, pass_no=1, method="cheminfo",
                            confidence="High", commentary="t")
    T.apply_tiers(led)                       # must not raise on the pd.NA row
    by = led.set_index("peak_id")
    assert by.loc["a", "tier"] == by.loc["b", "tier"] == by.loc["c", "tier"] == by.loc["e", "tier"]
    assert by.loc["d", "tier"] == "Candidate"   # persistence-admitted, uncorroborated


def test_below_assignability_reads_a_saturated_note_only():
    """`below_assignability` (O >= 11) needs the audit's MASS-SATURATED flag, as its reason
    says. A MASS-DEGENERATE window (3-8 ions, a lower bound included) stays degenerate for
    the tier's cap but is not flagged below assignability."""
    from peaky.assignment import tiers as T
    tail = (" — the committed formula lies outside this run's enumerated space (C41 > the "
            "context's C40), so the count is a lower bound")
    notes = {
        "sat": (12, "MASS-SATURATED: 12 plausible formulas (≤3 heteroatom types) within ±3σ "
                    "calibrated window — not identifiable from accurate mass alone"),
        "sat_lb": (9, "MASS-SATURATED: at least 9 plausible formulas (≤3 heteroatom types) within "
                      "±3σ calibrated window" + tail),
        "deg8": (8, "MASS-DEGENERATE: 8 plausible ions within ±3σ calibrated window — competitors: "
                    "C9H12O13 [M+NO3]- (+0.40 ppm)"),
        "deg_lb": (3, "MASS-DEGENERATE: at least 3 plausible ions within ±3σ calibrated window — "
                      "competitors: C9H12O13 [M+NO3]- (+0.40 ppm); C12H10O10 [M-H]- (-1.1 ppm)" + tail),
    }
    ba = pd.DataFrame([dict(peak_id=p, role="M0", neutral_formula="C10H16O11", degeneracy_density=d,
                            degeneracy_note=n, tier="Candidate", tier_reason="r")
                       for p, (d, n) in notes.items()])
    assert T.flag_below_assignability(ba) == 2
    by = ba.set_index("peak_id")
    assert bool(by.loc["sat", "below_assignability"]) and bool(by.loc["sat_lb", "below_assignability"])
    assert not bool(by.loc["deg8", "below_assignability"]) and not bool(by.loc["deg_lb", "below_assignability"])
    assert by.loc["sat", "tier_reason"] == "r | below-assignability (O>=11, mass-saturated)"
    assert by.loc["deg8", "tier_reason"] == "r"
    # still degenerate for the tier's cap
    assert T._degeneracy(by.loc["deg8"])[1] and T._degeneracy(by.loc["deg_lb"])[1]
    assert not T._saturated(by.loc["deg8"]) and T._saturated(by.loc["sat_lb"])
    assert not T._saturated({"degeneracy_note": pd.NA}) and not T._saturated({})
