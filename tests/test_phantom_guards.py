"""Offline regression tests for the Si/P phantom-assignment guards and exhaustive
isotopologue claiming. Run: python3 tests/test_phantom_guards.py

Covers three coupled fixes:
  (a) arbitration het-iso gate now includes Si (29Si/30Si) -> a bare-Si mass fit
      with no confirmed satellite loses to a CHO/CHON rival; a confirmed one wins.
  (b) tier engine demotes an uncorroborated Si (has an isotope twin that must
      appear) and an uncorroborated mono-isotopic P/I (no twin, needs a second
      channel) to Candidate; cross-channel / known: species are spared.
  (c) satellite claiming is exhaustive: the faint 15N/34S/29Si diagnostic line is
      now predicted (complete_isotope_envelopes) and swept (reclaim_satellites),
      so it is claimed as an iso_child / displaces a weak phantom M0 sitting on it
      instead of floating free for a mass-coincidence phantom to grab.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from peaky import ledger as L        # noqa: E402
from peaky import tiers as T         # noqa: E402
from peaky import cleanup as CU      # noqa: E402
from peaky import passes as P        # noqa: E402
from peaky import isotopes as ISO    # noqa: E402

PASS = FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok  {name}")
    else:
        FAIL += 1
        print(f"FAIL  {name}  {detail}")


def iso_row(**kw):
    base = dict(compound_formula=None, compound_score=None, compound_category=2,
                ion_formula=None, ion_score=None, ion_category=2, mechanism_id="m",
                isotope_formula=None, iso_label="M0", is_base=True, theo_mz=100.0,
                rel_abundance=1.0, iso_score=None, iso_category=2,
                sample_peak_id=None, sample_peak_mz=100.0, sample_peak_intensity=1e4,
                ppm_error=0.2, abundance_error=0.0)
    base.update(kw)
    return base


CFG = P.PassConfig(height_cutoff_cps=100.0)   # offline: explicit absolute gate (no edge stamped)

# ============================================================================
# (a) arbitration het-iso gate for Si  (mirrors the existing 34S gate)
# ============================================================================
# no 29Si evidence: the Si fit pays prior(0.20)+gate(0.12) and loses to the CHON.
scored_si = pd.DataFrame([
    iso_row(sample_peak_id="G", compound_formula="C8H16O5Si", compound_score=0.90,
            ion_formula="C8H17O5Si+", ion_score=0.90, ppm_error=0.3),
    iso_row(sample_peak_id="G", compound_formula="C9H15NO5", compound_score=0.86,
            ion_formula="C9H16NO5+", ion_score=0.86, ppm_error=0.5),
])
wg = P.arbitrate(scored_si, CFG)["winners"].iloc[0]
check("Si without 29Si loses to CHON (arbitration iso gate)",
      wg["neutral"] == "C9H15NO5", (wg["neutral"], round(float(wg["eff_score"]), 3)))

# with a confirmed 29Si satellite: the Si penalty is waived and the Si fit wins.
scored_si2 = pd.DataFrame([
    iso_row(sample_peak_id="H", compound_formula="C8H16O5Si", compound_score=0.90,
            ion_formula="C8H17O5Si+", ion_score=0.90, ppm_error=0.3),
    iso_row(sample_peak_id="H2", compound_formula="C8H16O5Si", compound_score=0.90,
            ion_formula="C8H17O5Si+", ion_score=0.90, isotope_formula="[29Si]C8H17O5+",
            iso_label="29Si", is_base=False, iso_score=0.88, ppm_error=0.2),
    iso_row(sample_peak_id="H", compound_formula="C9H15NO5", compound_score=0.86,
            ion_formula="C9H16NO5+", ion_score=0.86, ppm_error=0.5),
])
wsi = P.arbitrate(scored_si2, CFG)["winners"]
wh = wsi[wsi.peak_id == "H"].iloc[0]
check("Si WITH 29Si confirmed wins (penalty waived)",
      wh["neutral"] == "C8H16O5Si", (wh["neutral"], round(float(wh["eff_score"]), 3)))

# a confirmed 30Si (M+2 only) must also waive the gate (diag tuple accepts either)
scored_si3 = pd.DataFrame([
    iso_row(sample_peak_id="K", compound_formula="C8H16O5Si", compound_score=0.90,
            ion_formula="C8H17O5Si+", ion_score=0.90, ppm_error=0.3),
    iso_row(sample_peak_id="K2", compound_formula="C8H16O5Si", compound_score=0.90,
            ion_formula="C8H17O5Si+", ion_score=0.90, isotope_formula="[30Si]C8H17O5+",
            iso_label="30Si", is_base=False, iso_score=0.85, ppm_error=0.2),
    iso_row(sample_peak_id="K", compound_formula="C9H15NO5", compound_score=0.86,
            ion_formula="C9H16NO5+", ion_score=0.86, ppm_error=0.5),
])
wk = P.arbitrate(scored_si3, CFG)["winners"]
wk0 = wk[wk.peak_id == "K"].iloc[0]
check("Si confirmed on 30Si only also wins", wk0["neutral"] == "C8H16O5Si",
      wk0["neutral"])

# ============================================================================
# (b) tier engine: demote uncorroborated Si and mono-isotopic P/I; spare others
# ============================================================================
peaks = pd.DataFrame({
    "peak_id": ["si_bad", "si_ok", "si_ok_iso", "p_orphan", "p_h", "p_nh4",
                "p_known", "i_orphan"],
    "mz": [276.126, 355.07, 356.07, 298.214, 213.09, 230.12, 331.04, 340.10],
    "height": [3e3, 5e3, 4e2, 8e3, 9e3, 7e3, 2e4, 5e3]})
lg = L.new_ledger(peaks)

# uncorroborated Si on a single channel, no iso child -> Candidate
L.commit_assignment(lg, "si_bad", neutral_formula="C11H18O5Si", adduct="[M+NH4]+",
                    ion_formula="C11H22NO5Si+", ion_score=0.90, compound_score=0.90,
                    ppm_error=0.2, pass_no=3, method="grid", confidence="Good",
                    commentary="pdms family fit")
# real trisiloxane WITH a confirmed 29Si iso child -> corroborated -> Assigned
L.commit_assignment(lg, "si_ok", neutral_formula="C6H18O3Si3", adduct="[M+H]+",
                    ion_formula="C6H19O3Si3+", ion_score=0.92, compound_score=0.92,
                    ppm_error=0.1, pass_no=3, method="grid", confidence="Good",
                    isotopologues=[{"label": "29Si", "score": 0.9, "peak_id": "si_ok_iso"}], commentary="test")
L.attach_isotopologue(lg, "si_ok_iso", "si_ok", iso_label="29Si", iso_match_score=0.9)

# mono-isotopic P, single [M+NH4]+ orphan, no corroboration -> Candidate
L.commit_assignment(lg, "p_orphan", neutral_formula="C13H29O4P", adduct="[M+NH4]+",
                    ion_formula="C13H33NO4P+", ion_score=0.91, compound_score=0.91,
                    ppm_error=0.2, pass_no=3, method="grid", confidence="Good",
                    commentary="orphan P on NH4")
# same P neutral seen on TWO channels -> cross_channel -> Assigned (both rows)
L.commit_assignment(lg, "p_h", neutral_formula="C6H15O4P", adduct="[M+H]+",
                    ion_formula="C6H16O4P+", ion_score=0.90, compound_score=0.90,
                    ppm_error=0.2, pass_no=1, method="grid", confidence="Good", commentary="test")
L.commit_assignment(lg, "p_nh4", neutral_formula="C6H15O4P", adduct="[M+NH4]+",
                    ion_formula="C6H19NO4P+", ion_score=0.88, compound_score=0.88,
                    ppm_error=0.3, pass_no=3, method="grid", confidence="Good", commentary="test")
# pass-0 known: organophosphate -> Assigned regardless (known branch first)
L.commit_assignment(lg, "p_known", neutral_formula="C10H19O6PS2", adduct="[M+H]+",
                    ion_formula="C10H20O6PS2+", ion_score=0.93, compound_score=0.93,
                    ppm_error=0.1, pass_no=0, method="known:pesticide:malathion",
                    confidence="Good (known)", commentary="test")
# mono-isotopic I, single-channel, uncorroborated -> Candidate
L.commit_assignment(lg, "i_orphan", neutral_formula="C8H9IO", adduct="[M+H]+",
                    ion_formula="C8H10IO+", ion_score=0.90, compound_score=0.90,
                    ppm_error=0.2, pass_no=3, method="grid", confidence="Good", commentary="test")

tt = T.compute_tiers(lg).set_index("peak_id")
check("uncorroborated Si -> Candidate",
      tt.at["si_bad", "tier"] == T.TIER_CANDIDATE, tt.at["si_bad", "tier_reason"])
check("29Si-confirmed siloxane -> Assigned",
      tt.at["si_ok", "tier"] == T.TIER_ASSIGNED, tt.at["si_ok", "tier_reason"])
check("orphan mono-isotopic P (NH4 only) -> Candidate",
      tt.at["p_orphan", "tier"] == T.TIER_CANDIDATE, tt.at["p_orphan", "tier_reason"])
check("cross-channel P -> Assigned (spared)",
      tt.at["p_h", "tier"] == T.TIER_ASSIGNED, tt.at["p_h", "tier_reason"])
check("pass-0 known: organophosphate -> Assigned (spared)",
      tt.at["p_known", "tier"] == T.TIER_ASSIGNED, tt.at["p_known", "tier_reason"])
check("orphan mono-isotopic I -> Candidate",
      tt.at["i_orphan", "tier"] == T.TIER_CANDIDATE, tt.at["i_orphan", "tier_reason"])
check("Si demote reason mentions 29Si/30Si",
      "29Si" in tt.at["si_bad", "tier_reason"], tt.at["si_bad", "tier_reason"])

# ============================================================================
# (c1) reclaim_satellites now claims a leaked 15N / 34S / 29Si satellite
# ============================================================================
p0 = 298.1918
lr = L.new_ledger(pd.DataFrame({
    "peak_id": ["par", "n15", "s34src", "s34", "far"],
    "mz": [p0, p0 + ISO.D_15N,          # 15N satellite of the CHON parent
           260.10, 260.10 + ISO.D_34S,  # 34S satellite of a mono-S parent
           400.0],
    "height": [1.0e6, 3.6e3, 5.0e5, 2.1e4, 50.0]}))
L.commit_assignment(lr, "par", neutral_formula="C16H29NO4", adduct="[M-H]-",
                    ion_formula="C16H28NO4-", ion_score=0.95, compound_score=0.95,
                    ppm_error=0.1, pass_no=1, method="grid", confidence="High", commentary="test")
L.commit_assignment(lr, "s34src", neutral_formula="C10H13O5S", adduct="[M-H]-",
                    ion_formula="C10H13O5S-", ion_score=0.9, compound_score=0.9,
                    ppm_error=0.1, pass_no=1, method="grid", confidence="High", commentary="test")
res = CU.reclaim_satellites(lr, log=lambda *a: None)
check("reclaim: 15N satellite -> iso_child", L.role_of(lr, "n15") == L.ROLE_ISO,
      L.role_of(lr, "n15"))
check("reclaim: 34S satellite -> iso_child", L.role_of(lr, "s34") == L.ROLE_ISO,
      L.role_of(lr, "s34"))
check("reclaim: isolated peak untouched", L.role_of(lr, "far") == L.ROLE_UNEXPLAINED)
check("reclaim: parents untouched",
      L.role_of(lr, "par") == L.ROLE_M0 and L.role_of(lr, "s34src") == L.ROLE_M0)
check("reclaim: ledger valid", L.validate(lr) == [])
check("reclaim: count == 2", res["reclaimed"] == 2, res)

# ============================================================================
# (c2) complete_isotope_envelopes DISPLACES a weak phantom sitting on the 15N line
# ============================================================================
le = L.new_ledger(pd.DataFrame({
    "peak_id": ["parent", "phantom"],
    "mz": [p0, p0 + ISO.D_15N],
    "height": [1.0e6, 3.0e3]}))
L.commit_assignment(le, "parent", neutral_formula="C16H29NO4", adduct="[M-H]-",
                    ion_formula="C16H28NO4-", ion_score=0.95, compound_score=0.95,
                    ppm_error=0.1, pass_no=1, method="grid", confidence="High", commentary="test")
# the phantom: a weak (< tau_high), non-High mono-isotopic-P M0 on the 15N mass
L.commit_assignment(le, "phantom", neutral_formula="C13H28O4P", adduct="[M-H]-",
                    ion_formula="C13H28O4P-", ion_score=0.85, compound_score=0.85,
                    ppm_error=0.4, pass_no=4, method="residual", confidence="Good", commentary="test")
out = P.complete_isotope_envelopes(le, CFG, log=lambda *a: None)
check("iso-envelope: weak phantom on 15N line displaced to iso_child",
      L.role_of(le, "phantom") == L.ROLE_ISO, L.role_of(le, "phantom"))
check("iso-envelope: phantom re-parented onto the true CHON",
      le.loc[le.peak_id == "phantom", "parent_peak_id"].iloc[0] == "parent")
check("iso-envelope: parent stays M0", L.role_of(le, "parent") == L.ROLE_M0)
check("iso-envelope: at least one displacement recorded", out["displaced"] >= 1, out)
check("iso-envelope: ledger valid", L.validate(le) == [])

# ============================================================================
# (c3) STRONG-scoring phantom on a FAINT 15N line is displaced (score >= tau_high),
#      but excess-intensity and High-confidence victims are spared. Reproduces the
#      real C13H29O4P ghost = 15N satellite of a 153x-brighter CHON (ratio ~0.9).
# ============================================================================
pA = 297.2174
lg3 = L.new_ledger(pd.DataFrame({
    "peak_id": ["pA", "ghostP", "pB", "brightV", "pC", "highV"],
    "mz": [pA, pA + ISO.D_15N,            # ghost: strong P fit on pA's 15N line
           250.0, 250.0 + ISO.D_15N,      # control: intensity 5x the predicted line
           360.0, 360.0 + ISO.D_15N],     # control: victim is High-confidence
    "height": [142367.0, 929.0,           # ratio ~0.90  -> displace
               100000.0, 3640.0,          # ratio ~5.0   -> keep (real co-eluter)
               120000.0, 900.0]}))        # ratio ~1.0 but High -> keep
def _c(pid, neu, ion, sc, conf, mz_note=""):
    L.commit_assignment(lg3, pid, neutral_formula=neu, adduct="[M+H]+",
                        ion_formula=ion, ion_score=sc, compound_score=sc,
                        ppm_error=0.2, pass_no=2, method="grid", confidence=conf,
                        commentary="test")
_c("pA", "C16H30N2O3", "C16H29N2O3+", 0.95, "High")        # bright 2-N parent
_c("ghostP", "C13H29O4P", "C13H33NO4P+", 0.909, "Good")    # score >= tau_high (0.90)
_c("pB", "C10H21N2O2", "C10H20N2O2+", 0.95, "High")
_c("brightV", "C9H17N2O2", "C9H16N2O2+", 0.92, "Good")     # strong + too bright
_c("pC", "C14H25N2O4", "C14H24N2O4+", 0.95, "High")
_c("highV", "C11H21N2O3", "C11H20N2O3+", 0.95, "High")     # High-confidence victim
out3 = P.complete_isotope_envelopes(lg3, CFG, log=lambda *a: None)
check("iso-envelope: STRONG-score phantom on faint 15N line IS displaced",
      L.role_of(lg3, "ghostP") == L.ROLE_ISO, L.role_of(lg3, "ghostP"))
check("iso-envelope: displaced phantom re-parented onto the CHON",
      lg3.loc[lg3.peak_id == "ghostP", "parent_peak_id"].iloc[0] == "pA")
check("iso-envelope: excess-intensity victim (ratio~5) is KEPT as M0",
      L.role_of(lg3, "brightV") == L.ROLE_M0, L.role_of(lg3, "brightV"))
check("iso-envelope: High-confidence victim is KEPT as M0",
      L.role_of(lg3, "highV") == L.ROLE_M0, L.role_of(lg3, "highV"))
check("iso-envelope: bright parents stay M0",
      all(L.role_of(lg3, p) == L.ROLE_M0 for p in ("pA", "pB", "pC")))
check("iso-envelope (c3): ledger valid", L.validate(lg3) == [])


def test_all():
    assert FAIL == 0, f"{FAIL} checks failed"


if __name__ == "__main__":
    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)
