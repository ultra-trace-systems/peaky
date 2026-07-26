"""Offline tests for residual.py (Pass 4). Run: python3 tests/test_residual.py"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from peaky import residual as RD  # noqa: E402
from peaky import ledger as L  # noqa: E402
from peaky import contexts as X  # noqa: E402
from peaky import chemistry as C  # noqa: E402
from peaky import isotopes as ISO  # noqa: E402
from peaky.assignment.passes import PassConfig  # noqa: E402

PASS = FAIL = 0
CFG = PassConfig(height_cutoff=100)
PROF = X.get_context("ambient-air")
PRE = ISO.PrescanResult(estimated_max_C=20)


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok  {name}")
    else:
        FAIL += 1
        print(f"FAIL  {name}  {detail}")


# ---------- acceptance policy ----------
ok, _ = RD._accept(0.85, 0.4, False, CFG)
check("strict ppm accepts on score alone", ok)
ok, _ = RD._accept(0.85, 2.5, False, CFG)
check("loose ppm WITHOUT pattern rejected", not ok)
ok, why = RD._accept(0.85, 2.5, True, CFG)
check("loose ppm WITH pattern accepted", ok, why)
ok, _ = RD._accept(0.85, 6.0, True, CFG)
check("beyond pattern ppm rejected even with pattern", not ok)
ok, _ = RD._accept(0.40, 0.2, True, CFG)
check("below score floor rejected", not ok)
check("confidence capped to Good", RD._cap_conf("High (iso-pair)") == "Good (iso-pair)")

# ---------- find_iso_pairs ----------
# Br doublet (ratio ~1) at the REAL m/z of C6H10O3 [M+Br]-, a Cl doublet, orphan
MZ_BR = C.ion_mz("C6H10O3", "[M+Br]-")
peaks = pd.DataFrame({
    "peak_id": ["L1", "H1", "L2", "H2", "orph"],
    "mz": [MZ_BR, MZ_BR + RD.D_PAIR_BR, 200.0, 200.0 + RD.D_PAIR_CL, 123.4],
    "height": [1e4, 9.6e3, 1e4, 0.33e4, 5e3],
})
led = L.new_ledger(peaks)
pairs = RD.find_iso_pairs(led, min_height=100)
check("finds 2 isotope pairs", len(pairs) == 2, pairs.to_dict("records"))
check("Br pair n_halogen=1", (pairs[pairs.element == "Br"].n_halogen == 1).all())
check("Cl pair detected", (pairs.element == "Cl").any())
check("orphan not paired", "orph" not in set(pairs.light_pid) | set(pairs.heavy_pid))

# ---------- candidates_for_pair: ion must carry the halogen, DBE-only ----------
# light_mz for C6H10O3 as [M+Br]- ; n_Br=1 in ion via the adduct (need=0 neutral)
mz_br = C.ion_mz("C6H10O3", "[M+Br]-")
cands = RD.candidates_for_pair(mz_br, "Br", 1, ["[M+Br]-", "[M-H]-"],
                               ranges={"C": (1, 12), "H": (0, 28), "O": (0, 8),
                                       "N": (0, 1), "S": (0, 0)}, ppm=4.0)
check("pair candidates recover C6H10O3 ([M+Br]- gives the Br)", "C6H10O3" in cands,
      sorted(cands)[:6])
# all candidates obey DBE-only gate
check("all pair candidates pass DBE gate", all(C.dbe_ok(f)[0] for f in cands))

# ---------- stage A end-to-end with a fake scorer ----------
def fake_scorer(client, sample_id, formulas, *, mechanism_ids=None, **kw):
    """Return a flat per-isotopologue table: C6H10O3 [M+Br]- matches L1 strongly
    with a confirmed 81Br child on H1; everything else misses."""
    rows = []
    if "C6H10O3" in formulas:
        ion = "C6H10BrO3-"
        rows.append(dict(compound_formula="C6H10O3", compound_score=0.93,
                         compound_category=2, ion_formula=ion, ion_score=0.93,
                         ion_category=2, mechanism_id="m", isotope_formula=ion,
                         iso_label="M0", is_base=True, theo_mz=MZ_BR,
                         rel_abundance=1.0, iso_score=0.97, iso_category=2,
                         sample_peak_id="L1", sample_peak_mz=MZ_BR,
                         sample_peak_intensity=1e4, ppm_error=0.3, abundance_error=0.0))
        rows.append(dict(compound_formula="C6H10O3", compound_score=0.93,
                         compound_category=2, ion_formula=ion, ion_score=0.93,
                         ion_category=2, mechanism_id="m",
                         isotope_formula="[81Br]C6H10O3-", iso_label="81Br",
                         is_base=False, theo_mz=MZ_BR + RD.D_PAIR_BR,
                         rel_abundance=0.97, iso_score=0.95, iso_category=2,
                         sample_peak_id="H1", sample_peak_mz=MZ_BR + RD.D_PAIR_BR,
                         sample_peak_intensity=9.6e3, ppm_error=0.2, abundance_error=0.02))
    return pd.DataFrame(rows)


led2 = L.new_ledger(peaks)
res = RD.stage_a_iso_pairs(None, "SID", led2, PROF, PRE, CFG, ["[M+Br]-", "[M-H]-"],
                           score_fn=fake_scorer, log=lambda *_: None)
check("stage A commits the light member", res["committed"] == 1, res)
check("L1 assigned C6H10O3", led2.loc[led2.peak_id == "L1", "neutral_formula"].iloc[0] == "C6H10O3")
check("L1 confidence is Good (capped, not High)",
      "Good" in str(led2.loc[led2.peak_id == "L1", "confidence"].iloc[0]),
      led2.loc[led2.peak_id == "L1", "confidence"].iloc[0])
check("H1 attached as 81Br isotopologue child",
      L.role_of(led2, "H1") == L.ROLE_ISO
      and led2.loc[led2.peak_id == "H1", "parent_peak_id"].iloc[0] == "L1")
check("commentary names the doublet + DBE-only policy",
      "doublet" in led2.loc[led2.peak_id == "L1", "commentary"].iloc[0]
      and "DBE-only" in led2.loc[led2.peak_id == "L1", "commentary"].iloc[0])
check("ledger valid after stage A", L.validate(led2) == [], L.validate(led2))

# ---------- stage B: 2-anchor support waives loose ppm ----------
def fake_scorer_b(client, sample_id, formulas, *, mechanism_ids=None, **kw):
    rows = []
    if "C10H16O5" in formulas:   # = C10H16O4 + O, between two anchors
        ion = "C10H15O5-"
        rows.append(dict(compound_formula="C10H16O5", compound_score=0.82,
                         compound_category=2, ion_formula=ion, ion_score=0.82,
                         ion_category=2, mechanism_id="m", isotope_formula=ion,
                         iso_label="M0", is_base=True, theo_mz=MZ_S1,
                         rel_abundance=1.0, iso_score=0.9, iso_category=2,
                         sample_peak_id="S1", sample_peak_mz=MZ_S1,
                         sample_peak_intensity=2e3, ppm_error=2.3, abundance_error=0.0))
    return pd.DataFrame(rows)


MZ_S1 = C.ion_mz("C10H16O5", "[M-H]-")
pk = pd.DataFrame({"peak_id": ["A1", "A2", "S1"],
                   "mz": [C.ion_mz("C10H16O4", "[M-H]-"),
                          C.ion_mz("C10H16O6", "[M-H]-"), MZ_S1],
                   "height": [1e5, 1e5, 2e3]})
led3 = L.new_ledger(pk)
# seed two anchors one O below and above the target C10H16O5
L.commit_assignment(led3, "A1", neutral_formula="C10H16O4", adduct="[M-H]-",
                    ion_score=0.97, pass_no=1, method="x", confidence="High", commentary="anchor")
L.commit_assignment(led3, "A2", neutral_formula="C10H16O6", adduct="[M-H]-",
                    ion_score=0.97, pass_no=1, method="x", confidence="High", commentary="anchor")
resb = RD.stage_b_series(None, "SID", led3, PROF, CFG, ["[M-H]-"], reagent="Br",
                         score_fn=fake_scorer_b, log=lambda *_: None)
check("stage B commits C10H16O5 at 2.3 ppm via 2-anchor support", resb["committed"] == 1, resb)
check("S1 assigned C10H16O5",
      led3.loc[led3.peak_id == "S1", "neutral_formula"].iloc[0] == "C10H16O5")
check("S1 commentary cites supporting anchors",
      "supporting anchors" in led3.loc[led3.peak_id == "S1", "commentary"].iloc[0])

# ---------- carbon clamp from 13C satellite ----------
ccled = L.new_ledger(pd.DataFrame({
    "peak_id": ["m0", "c13", "lone"],
    "mz": [300.0000, 300.0000 + 1.0033548, 412.5],
    "height": [10000.0, 1070.0, 5000.0]}))
clamp = RD.carbon_count_from_13c(ccled, "m0")
check("carbon clamp brackets ~C10 from 13C ratio",
      clamp is not None and clamp[0] <= 10 <= clamp[1], clamp)
check("carbon clamp None when no satellite present",
      RD.carbon_count_from_13c(ccled, "lone") is None)

# ---------- residual characterization tiers ----------
chled = L.new_ledger(pd.DataFrame({
    "peak_id": ["L", "H", "C", "iso", "alone"],
    "mz": [400.0000, 400.0 + RD.D_PAIR_BR, 401.0033548,
           500.0 + RD.D_PAIR_BR, 250.123],
    "height": [10000.0, 9700.0, 1070.0, 4000.0, 8000.0]}))
ch = RD.characterize_residual(chled, min_height=100)
tier = dict(zip(ch["peak_id"], ch["tier"]))
check("characterize: Br-doublet light member -> has-constraints",
      tier["L"] == "has-constraints", tier)
check("characterize: 81Br twin -> iso-partner", tier["H"] == "iso-partner", tier)
check("characterize: isolated bright peak -> isolated",
      tier["alone"] == "isolated", tier)
check("characterize: light member records n_Br=1",
      int(ch.loc[ch.peak_id == "L", "n_Br"].iloc[0]) == 1)

# ---------- TFA regression: F earned by chain evidence reaches pair enumeration ----------
# TFA.Br- at 192.9116 with its 0.978-ratio 81Br twin was invisible v13-v18:
# not a chain member (no CF2 tail peak) and no F in the pass-4 grid.
franges = {"C": (1, 40), "H": (0, 84), "O": (0, 30), "N": (0, 3),
           "S": (0, 1), "F": (0, 17)}
fc = RD.candidates_for_pair(192.91161, "Br", 1, ["[M+Br]-", "[M-H]-"],
                            ranges=franges, ppm=4.0)
check("TFA enumerable for its Br pair once F is enabled", "C2HF3O2" in fc, fc)
check("TFA pair candidate set stays tiny (density bounded)", len(fc) <= 4, fc)
fc_nof = RD.candidates_for_pair(192.91161, "Br", 1, ["[M+Br]-", "[M-H]-"],
                                ranges={k: v for k, v in franges.items()
                                        if k != "F"}, ppm=4.0)
check("without F the pair has no candidates (the v13-v18 hole)",
      len(fc_nof) == 0, fc_nof)

# F+O cap in pair enumeration: the v19 junk class (Cl/F + O16) must not enumerate
jranges = {"C": (1, 40), "H": (0, 84), "O": (0, 30), "N": (0, 3),
           "S": (0, 1), "F": (0, 17), "Cl": (0, 2)}
jc = RD.candidates_for_pair(448.97890, "Cl", 1, ["[M-H]-"], ranges=jranges, ppm=4.0)
check("pair enumeration drops F-with-O>6 candidates",
      all(not (C.parse_formula(f).get("F", 0) >= 1
               and C.parse_formula(f).get("O", 0) > 6) for f in jc),
      [f for f in jc if C.parse_formula(f).get("F", 0) >= 1
       and C.parse_formula(f).get("O", 0) > 6])

# ---------- CH2O unit registered for series detection ----------
from peaky import series_detect as SD  # noqa: E402
check("CH2O in series unit library", "CH2O" in SD.UNIT_LIBRARY,
      sorted(SD.UNIT_LIBRARY))

# ---------- mixed BrCl pair classification (M+4-gated) ----------
# BrCl pattern: M+2/M ~ 1.29 (inside the Br1 band!), M+4/M ~ 0.31
pk_brcl = pd.DataFrame({
    "peak_id": ["L", "M2", "M4"],
    "mz": [400.0, 400.0 + RD.D_PAIR_BR, 400.0 + RD.D_PAIR_BR + RD.D_PAIR_CL],
    "height": [1e4, 1.29e4, 0.31e4]})
pr = RD.find_iso_pairs(L.new_ledger(pk_brcl), min_height=100)
check("BrCl pair classified via M+4 satellite",
      len(pr) == 1 and pr.iloc[0]["element"] == "BrCl"
      and pr.iloc[0]["m4_pid"] == "M4", pr.to_dict("records"))
# control: same M+2 ratio with NO M+4 -> read as plain Br1
pr2 = RD.find_iso_pairs(L.new_ledger(pk_brcl.iloc[:2]), min_height=100)
check("same ratio without M+4 stays Br1",
      len(pr2) == 1 and pr2.iloc[0]["element"] == "Br"
      and pr2.iloc[0]["n_halogen"] == 1, pr2.to_dict("records"))

# ---------- BrCl constrained enumeration ----------
mz_brcl = C.ion_mz("C15H17ClN2O10", "[M+Br]-")
bc = RD.candidates_for_pair(mz_brcl, "BrCl", 1, ["[M+Br]-", "[M-H]-"],
                            ranges={"C": (1, 20), "H": (0, 44), "O": (0, 14),
                                    "N": (0, 2), "S": (0, 0)}, ppm=3.0)
check("BrCl enumeration recovers the Cl neutral via [M+Br]-",
      "C15H17ClN2O10" in bc, sorted(bc)[:5])
check("every BrCl candidate carries exactly the pinned halogens",
      all((C.parse_formula(f).get("Cl", 0) == 1) for f in bc), sorted(bc)[:5])

# --- off-grid element invariant in the deep-series stage ---------------------
# stage_b_series applied only the STRUCTURAL gates (DBE + oxygen cap), so it walked
# +CO / +C2H2O / +O off the pass-0 iodine anchors (HOI/HIO2/INO3) into CHIO2, CHIO3,
# C2H3IO2, C2H3IO3, INO4 -- each really the I2 cluster of an acid already assigned.
# It now applies filter_by_context too, so ambient-air's max_I=0 rejects them.
from peaky import contexts as _X2   # noqa: E402
for _f in ("CHIO2", "CHIO3", "C2H3IO2", "C2H3IO3", "INO4"):
    _keep, _why = _X2.filter_by_context(_f, "ambient-air")
    check(f"residual off-grid: {_f} rejected by ambient-air ({_why})", not _keep)
for _f in ("C6H8O4", "C10H16O3", "C5H7NO4"):
    check(f"residual off-grid: {_f} still accepted",
          _X2.filter_by_context(_f, "ambient-air")[0])
check("residual off-grid: DBE/oxygen gates alone would NOT have caught CHIO2",
      C.dbe_ok("CHIO2")[0] and C.oxygen_ok("CHIO2")[0])


def test_all():
    assert FAIL == 0, f"{FAIL} checks failed"


if __name__ == "__main__":
    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)
