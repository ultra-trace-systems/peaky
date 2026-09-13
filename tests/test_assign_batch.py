"""Offline tests for assign_batch.py pure align/merge. Run: python3 tests/test_assign_batch.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from peaky import assign_batch as AB  # noqa: E402

PASS = FAIL = 0
def check(name, cond, detail=""):
    global PASS, FAIL
    if cond: PASS += 1; print(f"  ok  {name}")
    else: FAIL += 1; print(f"FAIL  {name}  {detail}")


def m0(rows):
    return pd.DataFrame(rows, columns=["mz", "neutral_formula", "adduct", "tier", "ion_score"])


# fileA + fileB share a peak (C10H16O2 @ ~217.12, mass jitter ~2 ppm), each has
# one unique peak; fileB also assigns the shared peak Candidate (A is Assigned).
A = m0([(217.1200, "C10H16O2", "[M+H]+", "Assigned", 0.91),
        (158.1539, "C9H19NO", "[M+H]+", "Assigned", 0.85)])      # A-only
B = m0([(217.1205, "C10H16O2", "[M+H]+", "Candidate", 0.72),
        (300.1000, "C15H17NO5", "[M+H]+", "Assigned", 0.80)])    # B-only

merged, jitter = AB.align({"A": A, "B": B}, tol_ppm=6.0)

check("3 distinct clusters (1 shared + 2 unique)", len(merged) == 3, len(merged))
shared = merged[merged["neutral_formula"] == "C10H16O2"].iloc[0]
check("shared peak seen in 2 files", shared["n_files"] == 2, shared.to_dict())
check("best tier wins (Assigned over Candidate)", shared["tier"] == "Assigned", shared["tier"])
check("formula_agree True for the shared peak", bool(shared["formula_agree"]))
check("raw mz jitter ~2.3 ppm measured",
      2.0 <= shared["mz_jitter_ppm_raw"] <= 2.6, shared["mz_jitter_ppm_raw"])
check("A-only peak flagged single-file",
      merged[merged["neutral_formula"] == "C9H19NO"].iloc[0]["n_files"] == 1)
check("jitter long-form has 4 rows (2 shared + 2 unique)", len(jitter) == 4, len(jitter))

# --- offset-awareness: same peak, files at different calibrations ------------
# fileC at +3 ppm, fileD at -3 ppm -> raw mz differ by ~6 ppm; with offsets the
# corrected positions coincide and they cluster as ONE peak.
mz0 = 250.0
C = m0([(mz0 * (1 + 3e-6), "C12H20O5", "[M+H]+", "Assigned", 0.88)])
D = m0([(mz0 * (1 - 3e-6), "C12H20O5", "[M+H]+", "Assigned", 0.87)])
mC, _ = AB.align({"C": C, "D": D}, tol_ppm=4.0)                    # raw spread 6ppm > 4
check("without offsets: 6ppm raw split into 2 clusters at tol 4", len(mC) == 2, len(mC))
mO, _ = AB.align({"C": C, "D": D}, tol_ppm=4.0, offsets={"C": 3.0, "D": -3.0})
check("offset-aware: corrected positions coincide -> 1 cluster", len(mO) == 1, len(mO))
check("offset-aware: raw jitter ~6 ppm but cal-adjusted ~0",
      len(mO) == 1 and mO.iloc[0]["mz_jitter_ppm_raw"] >= 5.5
      and mO.iloc[0]["mz_jitter_ppm_caldj"] < 0.5,
      mO.iloc[0][["mz_jitter_ppm_raw", "mz_jitter_ppm_caldj"]].to_dict() if len(mO) else "empty")

# --- formula disagreement is detected ---------------------------------------
E = m0([(400.0000, "C20H25NO7", "[M+H]+", "Assigned", 0.7)])
F = m0([(400.0010, "C16H29NO10", "[M+H]+", "Candidate", 0.6)])     # different formula, same m/z
mEF, _ = AB.align({"E": E, "F": F}, tol_ppm=6.0)
check("formula disagreement flagged (formula_agree False)",
      len(mEF) == 1 and not bool(mEF.iloc[0]["formula_agree"]), mEF.to_dict("records"))

# --- cross-file consensus: corroborated formula beats single-file outlier ----
# The Ur+ m/z 424.218 bug: a mass-degenerate competitor reads on-cal
# (Assigned) with a marginally higher local score in ONE file's calibration,
# while five files agree on the real reflist HOM. The old "best (tier, ion_score)
# row" let the single-file outlier win; the consensus vote must pick the formula
# Assigned across the most files.
def _file(mz, nf, tier, ion):
    return m0([(mz, nf, "[M+NH4]+", tier, ion)])

consensus = {
    "f1": _file(424.2177, "C18H30O10", "Assigned", 0.944),
    "f2": _file(424.2179, "C18H30O10", "Assigned", 0.922),
    "f3": _file(424.2177, "C18H30O10", "Assigned", 0.985),
    "f4": _file(424.2178, "C18H30O10", "Assigned", 0.961),
    "f5": _file(424.2177, "C18H30O10", "Assigned", 0.987),
    "f6": m0([(424.2165, "C17H31N5O6", "[M+Na]+", "Candidate", 0.939)]),
    "f7": m0([(424.2167, "C17H31N5O6", "[M+Na]+", "Assigned", 0.996)]),  # outlier
}
mc, _ = AB.align(consensus, tol_ppm=6.0)
row = mc.iloc[(mc["mz"] - 424.2174).abs().argmin()]
check("consensus winner is the 5-file Assigned formula, not the 1-file outlier",
      row["neutral_formula"] == "C18H30O10", row.to_dict())
check("consensus winner keeps Assigned tier", row["tier"] == "Assigned", row["tier"])
check("consensus cluster spans all 7 files", row["n_files"] == 7, row["n_files"])

# a single-file bright Assigned formula with no competitor is still chosen (no
# spurious override when there is nothing to out-vote).
solo, _ = AB.align({"a": m0([(500.0, "C20H30O8", "[M+H]+", "Assigned", 0.9)])}, tol_ppm=6.0)
check("solo Assigned formula chosen (vote is a no-op without a competitor)",
      len(solo) == 1 and solo.iloc[0]["neutral_formula"] == "C20H30O8")

# --- THE VOTE, stage 1: DIFFERENT IONS at one m/z -> the file count decides ----
# The Texas Ur+ minority-winner defect: the old rule ranked Assigned-file count
# first, so ONE file's Assigned reading outvoted FOURTEEN files' Candidate reading
# of another ion. File count decides; tier and score only break ties; the losers
# stay on the row.
def _files(n, mz, nf, ad, tier, ion, start=0):
    return {f"v{start + i:02d}": m0([(mz + 1e-4 * i, nf, ad, tier, ion)]) for i in range(n)}


vote = {**_files(14, 300.0, "C10H12O2", "[M+H]+", "Candidate", 0.90),
        **_files(1, 300.0, "C9H12N2O", "[M+H]+", "Assigned", 0.99, start=14)}   # C10H13O2+ vs C9H13N2O+
mv, jv = AB.align(vote, tol_ppm=6.0)
_v = mv.iloc[0]
check("vote: 14 Candidate files beat 1 Assigned file of a different ion",
      len(mv) == 1 and _v["neutral_formula"] == "C10H12O2" and _v["adduct"] == "[M+H]+",
      mv.to_dict("records"))
check("vote: the merged tier / score are the WINNER's best row, not the loser's",
      _v["tier"] == "Candidate" and abs(float(_v["ion_score"]) - 0.90) < 1e-9, _v.to_dict())
check("vote: n_files / n_files_ion / n_files_winner record it (15 / 14 / 14)",
      _v["n_files"] == 15 and _v["n_files_ion"] == 14 and _v["n_files_winner"] == 14, _v.to_dict())
check("vote: the losing reading stays visible on the merged row",
      _v["alternatives"] == "C9H12N2O [M+H]+ x1 Assigned 0.99", repr(_v["alternatives"]))
check("vote: ion_agree and formula_agree are both False (two ions), no note needed",
      not bool(_v["ion_agree"]) and not bool(_v["formula_agree"]) and pd.isna(_v["tier_reason"]),
      _v.to_dict())
check("vote: jitter.csv still carries every per-file reading (15 rows)",
      len(jv) == 15 and (jv["neutral_formula"] == "C9H12N2O").sum() == 1, len(jv))

# --- stage 2: the SAME ION read two ways -> corroboration decides, not the count
# m/z 252.123 on the Texas run: C13H14O4 [M+NH4]+ and C13H17NO4 [M+H]+ are ONE ion
# (C13H18NO4+, the reagent-N isobar). The tier engine marks such a reading Assigned
# only when a discriminating channel was present in that file, and Candidate when
# it had nothing to decide with -- so 6 Candidate files are 6 files that could not
# tell, and the 5 files that could win.
same = {**_files(6, 252.1230, "C13H17NO4", "[M+H]+", "Candidate", 0.98),
        **_files(5, 252.1230, "C13H14O4", "[M+NH4]+", "Assigned", 0.97, start=6)}
ms, _ = AB.align(same, tol_ppm=6.0)
_s = ms.iloc[0]
check("same ion: the label Assigned in 5 files beats the label Candidate in 6",
      _s["neutral_formula"] == "C13H14O4" and _s["adduct"] == "[M+NH4]+" and _s["tier"] == "Assigned",
      ms.to_dict("records"))
check("same ion: n_files_ion counts the ion (11), n_files_winner the label (5)",
      _s["n_files"] == 11 and _s["n_files_ion"] == 11 and _s["n_files_winner"] == 5, _s.to_dict())
check("same ion: ion_agree True, formula_agree False (one ion, two neutrals)",
      bool(_s["ion_agree"]) and not bool(_s["formula_agree"]), _s.to_dict())
check("same ion: the row explains the choice",
      _s["tier_reason"] == "same ion C13H18NO4+ read two ways: kept C13H14O4 [M+NH4]+ (Assigned in 5 "
                           "of its 5 files) over the 6-file C13H17NO4 [M+H]+ (Assigned in 0)",
      _s["tier_reason"])
check("same ion: the losing label is listed", _s["alternatives"] == "C13H17NO4 [M+H]+ x6 Candidate 0.98",
      repr(_s["alternatives"]))
# the task's own spec case, 14 Candidate vs 1 Assigned, read both ways: as two
# labels of ONE ion (m/z 160.072: C4H5NO2 uronium vs C5H6N2O3 [M+NH4]+, both
# C5H10N3O3+) the corroborated file wins; as two DIFFERENT ions the count does.
spec = {**_files(14, 160.0717, "C4H5NO2", "[M+(CH4N2O)H]+", "Candidate", 0.90),
        **_files(1, 160.0717, "C5H6N2O3", "[M+NH4]+", "Assigned", 0.99, start=14)}
msp, _ = AB.align(spec, tol_ppm=6.0)
check("same ion, 14 Candidate vs 1 Assigned: the one file that could decide wins, and says so",
      msp.iloc[0]["neutral_formula"] == "C5H6N2O3" and msp.iloc[0]["n_files_winner"] == 1
      and msp.iloc[0]["n_files_ion"] == 15 and bool(msp.iloc[0]["ion_agree"])
      and str(msp.iloc[0]["tier_reason"]).startswith("same ion C5H10N3O3+ read two ways"),
      msp.iloc[0].to_dict())
# same ion, nobody corroborated: the count decides, nothing to explain
nobody = {**_files(4, 160.0717, "C4H5NO2", "[M+(CH4N2O)H]+", "Candidate", 0.90),
          **_files(1, 160.0717, "C5H6N2O3", "[M+NH4]+", "Candidate", 0.99, start=4)}
mnb, _ = AB.align(nobody, tol_ppm=6.0)
check("same ion, no label Assigned anywhere: the count decides, no note",
      mnb.iloc[0]["neutral_formula"] == "C4H5NO2" and mnb.iloc[0]["n_files_winner"] == 4
      and pd.isna(mnb.iloc[0]["tier_reason"]), mnb.iloc[0].to_dict())
# a corroborated MAJORITY label needs no explanation either
maj = {**_files(14, 220.2059, "C15H22", "[M+NH4]+", "Assigned", 0.97),
       **_files(1, 220.2059, "C15H25N", "[M+H]+", "Candidate", 0.97, start=14)}
mmj, _ = AB.align(maj, tol_ppm=6.0)
check("same ion, the majority label is the corroborated one: it wins, 14 of 15, no note",
      mmj.iloc[0]["neutral_formula"] == "C15H22" and mmj.iloc[0]["n_files_winner"] == 14
      and pd.isna(mmj.iloc[0]["tier_reason"]), mmj.iloc[0].to_dict())

# a 1-vs-1 tie between different ions falls back to tier, then to ion_score
tie_t, _ = AB.align({"a": m0([(300.0, "C10H12O2", "[M+H]+", "Candidate", 0.90)]),
                     "b": m0([(300.0, "C9H12N2O", "[M+H]+", "Assigned", 0.85)])}, tol_ppm=6.0)
check("vote 1-vs-1: tier breaks the tie (Assigned beats Candidate, whatever the score)",
      tie_t.iloc[0]["neutral_formula"] == "C9H12N2O" and tie_t.iloc[0]["n_files_winner"] == 1,
      tie_t.to_dict("records"))
tie_s, _ = AB.align({"a": m0([(300.0, "C10H12O2", "[M+H]+", "Candidate", 0.90)]),
                     "b": m0([(300.0, "C9H12N2O", "[M+H]+", "Candidate", 0.95)])}, tol_ppm=6.0)
check("vote 1-vs-1, same tier: ion_score breaks the tie",
      tie_s.iloc[0]["neutral_formula"] == "C9H12N2O", tie_s.to_dict("records"))
check("vote 1-vs-1: the loser is listed with its own tier and score",
      tie_s.iloc[0]["alternatives"] == "C10H12O2 [M+H]+ x1 Candidate 0.90",
      repr(tie_s.iloc[0]["alternatives"]))
# a 2-vs-2 tie: the ion Assigned in more files wins before any score
tie_a, _ = AB.align({"a": m0([(300.0, "C10H12O2", "[M+H]+", "Assigned", 0.80)]),
                     "b": m0([(300.0, "C10H12O2", "[M+H]+", "Candidate", 0.80)]),
                     "c": m0([(300.0, "C9H12N2O", "[M+H]+", "Candidate", 0.99)]),
                     "d": m0([(300.0, "C9H12N2O", "[M+H]+", "Candidate", 0.98)])}, tol_ppm=6.0)
check("vote 2-vs-2: the Assigned-file count breaks the tie before ion_score",
      tie_a.iloc[0]["neutral_formula"] == "C10H12O2" and tie_a.iloc[0]["tier"] == "Assigned",
      tie_a.to_dict("records"))
# unanimous: the vote is a no-op and the row says so
check("vote: a unanimous cluster reports n_files_winner == n_files_ion == n_files, ion_agree, no alternatives",
      shared["n_files_winner"] == 2 and shared["n_files_ion"] == 2 and bool(shared["ion_agree"])
      and shared["alternatives"] == "", shared.to_dict())
check("vote: the 5-file consensus case still wins, with the 2-file Na reading listed",
      row["n_files_winner"] == 5 and row["n_files_ion"] == 5 and not bool(row["ion_agree"])
      and row["alternatives"] == "C17H31N5O6 [M+Na]+ x2 Assigned 1.00", row.to_dict())

# a CURATED identity (reference-list rescue / pass-0 known species: a list's
# identity, not the grid's) that reached Assigned somewhere is never outvoted by
# grid readings: at m/z 579.171 the D7 cyclosiloxane urea adduct, locked in ONE
# file, faced a 9-file C27H30O14 the per-file engine itself flags as an O14 monster.
prot = {**_files(9, 579.1710, "C27H30O14", "[M+H]+", "Candidate", 0.99),
        **_files(1, 579.1710, "C14H42O7Si7", "[M+(CH4N2O)H]+", "Assigned", 0.83, start=9)}
mp, _ = AB.align(prot, tol_ppm=6.0, curated={"C14H42O7Si7"})
check("vote: a curated identity beats a grid majority, and the vote is still recorded",
      mp.iloc[0]["neutral_formula"] == "C14H42O7Si7" and mp.iloc[0]["n_files_winner"] == 1
      and mp.iloc[0]["n_files"] == 10
      and mp.iloc[0]["alternatives"] == "C27H30O14 [M+H]+ x9 Candidate 0.99",
      mp.to_dict("records"))
check("vote: the exemption is explained on the row",
      mp.iloc[0]["tier_reason"] == "curated identity kept over the 9-file C27H30O14 [M+H]+ "
                                   "reading (vote 1 of 10 files)", mp.iloc[0]["tier_reason"])
mnp, _ = AB.align(prot, tol_ppm=6.0)
check("vote: without the curated set the same cluster goes to the majority, no note",
      mnp.iloc[0]["neutral_formula"] == "C27H30O14" and pd.isna(mnp.iloc[0]["tier_reason"]),
      mnp.to_dict("records"))
# ... and the exemption is from the file COUNT, not from corroboration: sulfolane
# (known list, 1 file, Assigned) at m/z 181.065 met fluorenone C13H8O [M+H]+
# Assigned in 9 files -- a real contest, which the count decides.
sulf = {**_files(9, 181.0647, "C13H8O", "[M+H]+", "Assigned", 0.99),
        **_files(1, 181.0647, "C4H8O2S", "[M+(CH4N2O)H]+", "Assigned", 0.95, start=9)}
msu, _ = AB.align(sulf, tol_ppm=6.0, curated={"C4H8O2S"})
check("vote: a curated reading does NOT override a grid reading Assigned in more files",
      msu.iloc[0]["neutral_formula"] == "C13H8O" and msu.iloc[0]["n_files_winner"] == 9
      and pd.isna(msu.iloc[0]["tier_reason"])
      and msu.iloc[0]["alternatives"] == "C4H8O2S [M+(CH4N2O)H]+ x1 Assigned 0.95",
      msu.to_dict("records"))
# ... but a curated reading that never reached Assigned has no claim on the vote
weak = {**_files(3, 346.0741, "C4H23NO2Si6", "[M+(CH4N2O)H]+", "Candidate", 0.98),
        **_files(1, 346.0741, "C19H8ClN", "[M+(CH4N2O)H]+", "Candidate", 0.90, start=3)}
mw, _ = AB.align(weak, tol_ppm=6.0, curated={"C19H8ClN"})
check("vote: a Candidate-only curated reading does not override the majority",
      mw.iloc[0]["neutral_formula"] == "C4H23NO2Si6" and mw.iloc[0]["n_files_winner"] == 3,
      mw.to_dict("records"))
check("vote: a unanimous curated reading carries no exemption note",
      pd.isna(AB.align({"a": m0([(500.0, "C21H21O4P", "[M+(CH4N2O)H]+", "Assigned", 0.9)])},
                       curated={"C21H21O4P"})[0].iloc[0]["tier_reason"]))

# an unparseable adduct is its own ion (no crash, no false merge of labels)
odd, _ = AB.align({"a": m0([(320.145, "C6H19N4PS2", "[M+1R+NH4]+", "Candidate", 0.9)]),
                   "b": m0([(320.145, "C11H17NO6", "[M+(CH4N2O)H]+", "Candidate", 0.95)])}, tol_ppm=6.0)
check("vote: a reagent-cluster adduct string is handled and stays a distinct ion",
      len(odd) == 1 and not bool(odd.iloc[0]["ion_agree"]) and odd.iloc[0]["n_files_winner"] == 1,
      odd.to_dict("records"))

# DETERMINISM: the vote must not depend on the order the files arrived in (the
# parallel path reduces in sample order, but the rule itself is order-free)
for _name, _d in (("different ions", vote), ("same ion", same)):
    _m1, _j1 = AB.align(_d, tol_ppm=6.0)
    _m2, _j2 = AB.align(dict(reversed(list(_d.items()))), tol_ppm=6.0)
    check(f"vote ({_name}): reversed file order -> identical merged frame",
          _m1.equals(_m2), (_m1.to_dict("records"), _m2.to_dict("records")))
    check(f"vote ({_name}): reversed file order -> identical jitter frame",
          _j1.reset_index(drop=True).equals(_j2.reset_index(drop=True)))
# a FULL tie (same count, tier and score) resolves by the reading's own text, so a
# serial and a parallel run cannot disagree on it
full = {"x": m0([(300.0, "C9H12N2O", "[M+H]+", "Candidate", 0.90)]),
        "y": m0([(300.0, "C10H12O2", "[M+H]+", "Candidate", 0.90)])}
f1, _ = AB.align(full, tol_ppm=6.0)
f2, _ = AB.align(dict(reversed(list(full.items()))), tol_ppm=6.0)
check("vote: a full tie resolves the same way in either file order",
      f1.iloc[0]["neutral_formula"] == f2.iloc[0]["neutral_formula"] == "C10H12O2",
      (f1.iloc[0]["neutral_formula"], f2.iloc[0]["neutral_formula"]))

# --- empty input ------------------------------------------------------------
me, je = AB.align({})
check("empty -> empty merged + jitter with schema",
      len(me) == 0 and {"n_files", "n_files_ion", "n_files_winner", "alternatives",
                        "tier_reason", "ion_agree"} <= set(me.columns)
      and "cluster" in je.columns)

# --- stage provenance: which stage first put the ion in the ledger ------------
# A and B are cover files, R a residual-stage file. The shared 217.12 ion is a
# cover row even though R holds it too; R's own 500.0 ion is a residual row.
R = m0([(217.1202, "C10H16O2", "[M+H]+", "Assigned", 0.90),
        (500.0000, "C25H40O8", "[M+H]+", "Candidate", 0.70)])
mS, _ = AB.align({"A": A, "B": B, "R": R}, tol_ppm=6.0,
                 stages={"A": AB.STAGE_COVER, "B": AB.STAGE_COVER, "R": AB.STAGE_RESIDUAL})
check("stages: a `stage` column sits right after `srcs`",
      list(mS.columns).index("stage") == list(mS.columns).index("srcs") + 1, list(mS.columns))
check("stages: an ion any cover file holds is a cover row, even when a residual file has it too",
      mS[mS["neutral_formula"] == "C10H16O2"].iloc[0]["stage"] == "cover"
      and mS[mS["neutral_formula"] == "C10H16O2"].iloc[0]["n_files"] == 3, mS.to_dict("records"))
check("stages: an ion only residual files hold is a residual row",
      mS[mS["neutral_formula"] == "C25H40O8"].iloc[0]["stage"] == "residual")
check("stages: a file missing from the record counts as a cover file",
      AB.align({"A": A, "R": R}, tol_ppm=6.0, stages={"R": AB.STAGE_RESIDUAL})[0]
      .set_index("neutral_formula")["stage"].to_dict()
      == {"C9H19NO": "cover", "C10H16O2": "cover", "C25H40O8": "residual"})
check("stages: without the record the column is absent (a cover-only ledger is unchanged)",
      "stage" not in merged.columns)
check("stages: the empty merge carries the column iff the record is given",
      "stage" in AB.align({}, stages={})[0].columns
      and "stage" not in AB.align({})[0].columns)

# --- _protected_neutrals: curated/known/certified provenance shields NH4 adducts
_ledp = pd.DataFrame([
    dict(neutral_formula="C10H15NO2S", method="reflist-rescue:contaminants_keller2008"),  # NBBS
    dict(neutral_formula="C10H19O6PS2", method="known:organophosphate"),                  # malathion
    dict(neutral_formula="C6H10O2",    method="certified:multi-channel"),
    dict(neutral_formula="C8H10O",     method="cheminfo+grid"),        # ordinary -> NOT protected
    dict(neutral_formula="C3H9NOSi",   method="contaminant:siloxane"), # NOT via this set (Si guard owns it)
])
_prot = AB._protected_neutrals(_ledp)
check("_protected_neutrals: reflist/known/certified in; grid/siloxane out",
      _prot == {"C10H15NO2S", "C10H19O6PS2", "C6H10O2"}, _prot)
check("_protected_neutrals: missing columns -> empty set",
      AB._protected_neutrals(pd.DataFrame({"x": [1]})) == set())
# the vote's exemption is the CURATED subset: a list's identity, not the file's
# own multi-channel evidence for a grid formula (that is already in the tier)
check("_curated_neutrals: reflist/known in; certified (and grid/siloxane) out",
      AB._curated_neutrals(_ledp) == {"C10H15NO2S", "C10H19O6PS2"}, AB._curated_neutrals(_ledp))

# --- admission provenance survives the merge --------------------------------
# The merged ledger is the deliverable: a reader must be able to see that a row
# only ever entered formula search because its m/z bin persists. The winning row
# donates `admitted_by` / `occurrence` along with its formula, so the provenance
# that is carried is the WINNER's, not an arbitrary file's.
def m0a(rows):
    return pd.DataFrame(rows, columns=["mz", "neutral_formula", "adduct", "tier",
                                       "ion_score", "admitted_by", "occurrence"])


# G: weak, persistence-admitted, Candidate. H: the same peak in a second file,
# also persistence-admitted, plus a bright height-admitted peak of its own.
G = m0a([(264.0361, "C8H10O6", "[M-H]-", "Candidate", 0.71, "occurrence", 0.93)])
H = m0a([(264.0365, "C8H10O6", "[M-H]-", "Candidate", 0.68, "occurrence", 0.91),
         (300.0000, "C10H16O4", "[M-H]-", "Assigned", 0.95, "height", 0.44)])
mGH, _ = AB.align({"G": G, "H": H}, tol_ppm=6.0)
check("merge carries admitted_by / occurrence for every row",
      {"admitted_by", "occurrence"} <= set(mGH.columns), list(mGH.columns))
_g = mGH[mGH["neutral_formula"] == "C8H10O6"].iloc[0]
check("merged persistence-only peak keeps admitted_by='occurrence' (2 files)",
      _g["n_files"] == 2 and _g["admitted_by"] == "occurrence" and _g["occurrence"] > 0.9,
      _g.to_dict())
check("the WINNING row donates it: G wins on ion_score, so G's occurrence (0.93) is carried",
      abs(float(_g["occurrence"]) - 0.93) < 1e-9, _g.to_dict())
check("a height-admitted merged peak keeps admitted_by='height'",
      mGH[mGH["neutral_formula"] == "C10H16O4"].iloc[0]["admitted_by"] == "height")

# backward compatibility: per-file ledgers written before the gate existed have
# neither column. They must still merge, with the admission columns present and
# empty rather than the merge failing on a missing key.
mOld, jOld = AB.align({"A": A, "B": B}, tol_ppm=6.0)          # A/B: the old 5-column schema
check("a frame WITHOUT the admission columns still merges (3 clusters, as before)",
      len(mOld) == 3 and len(jOld) == 4, (len(mOld), len(jOld)))
check("...and the merged schema still carries both columns, all-null",
      {"admitted_by", "occurrence"} <= set(mOld.columns)
      and mOld["admitted_by"].isna().all() and mOld["occurrence"].isna().all(),
      mOld[["admitted_by", "occurrence"]].to_dict("records"))
# mixed: one old-schema file, one new -> the new file's provenance still lands
mMix, _ = AB.align({"old": m0([(264.0361, "C8H10O6", "[M-H]-", "Candidate", 0.60)]),
                    "new": G}, tol_ppm=6.0)
check("old + new schema in one merge: the new file's row wins and keeps its provenance",
      len(mMix) == 1 and mMix.iloc[0]["admitted_by"] == "occurrence", mMix.to_dict("records"))


# ---------------------------------------------------------------------------
# the SELECTION block end to end through run(): the per-sample assign and the IO
# layer are stubbed, so this exercises the real selector -> summary -> CSV path.
# ---------------------------------------------------------------------------
import json  # noqa: E402
import os  # noqa: E402
import tempfile  # noqa: E402

from peaky.io import io_mascope as IO  # noqa: E402
from peaky.assignment import assign as _A  # noqa: E402
from peaky.assignment import ledger as _L  # noqa: E402
from peaky.assignment import passes as _PASSES  # noqa: E402
from peaky.assignment import tiers as _T  # noqa: E402
from peaky.batch import sampling as SS  # noqa: E402
from peaky.chem import chemistry as _C  # noqa: E402
from peaky.chem import profiles as P_PROF  # noqa: E402

_T0 = pd.Timestamp("2025-10-01 21:00:00", tz="UTC")


def _batch_table(spec, height=500.0):
    """Per-peak batch table: sample id -> the m/z values present in that sample."""
    rows = []
    for i, (sid, mzs) in enumerate(spec.items()):
        t = _T0 + pd.Timedelta(minutes=10 * i)
        rows += [dict(sample_item_id=sid, sample_item_name=f"n_{sid}", datetime_utc=t,
                      mz=float(mz), height=height) for mz in mzs]
    return pd.DataFrame(rows)


_BG = list(range(100, 120))                    # 20 bins every sample shares
_SPEC = {}
for _i in range(4):                            # 4 exclusive 20-bin blocks, each a PAIR
    _SPEC[f"a{_i}"] = _BG + list(range(200 + 20 * _i, 220 + 20 * _i))
    _SPEC[f"b{_i}"] = _BG + list(range(200 + 20 * _i, 220 + 20 * _i))
_PK = _batch_table(_SPEC)
_F = "C10H16O5"


_SEEN_CFG = []
_SEEN_KW = []


def _fake_assign(sid, context="ambient-air", **kw):
    """Stand-in for assign.run: one assigned M0, the real ledger schema. Keeps
    the cfg it was handed, so the gate resolution can be read back."""
    _SEEN_CFG.append(kw.get("cfg"))
    _SEEN_KW.append(dict(kw))
    led = _L.new_ledger(pd.DataFrame([("p1", _C.ion_mz(_F, "[M-H]-"), 1.0e5)],
                                     columns=["peak_id", "mz", "height"]))
    _L.commit_assignment(led, "p1", neutral_formula=_F, adduct="[M-H]-",
                         ion_formula="C10H15O5-", ion_score=0.9, compound_score=0.9,
                         ppm_error=0.1, pass_no=1, method="cheminfo+grid",
                         confidence="High", commentary="stub")
    _T.apply_tiers(led)
    return {"ledger": led, "stats": {"noise_edge_cps": 4.0, "height_gate_cps": 10.0},
            "plausibility_audit": [], "summaries": {}, "problems": []}


_saved = {"connect": IO.connect, "fetch_peaks": IO.fetch_peaks,
          "estimate_offset": IO.estimate_offset, "run": _A.run}
IO.connect = lambda *a, **k: "CLIENT"
IO.fetch_peaks = lambda client, sid, use_cache=True: pd.DataFrame(
    {"peak_id": ["p1"], "mz": [_C.ion_mz(_F, "[M-H]-")], "height": [1.0e5]})
IO.estimate_offset = lambda raw: 0.0
_A.run = _fake_assign
try:
    with tempfile.TemporaryDirectory() as _d:
        res = AB.run(peaks=_PK, ts_peaks=_PK, reagent="Br", batch="test batch",
                     out_dir=_d, k_min=2, k_max=3, min_gain=0.0, n_jobs=1,
                     log=lambda *a: None)
        summ = json.load(open(os.path.join(_d, "batch_summary.json")))
        s = summ["selection"]
        check("run: batch_summary carries the selection block",
              s["method"] == "presence-cover" and s["k"] == 3 and s["n_samples"] == 8
              and s["n_bins"] == 100, s)
        check("run: the k_max budget bound and is recorded with its rejected gain",
              s["stop_reason"] == "k_max" and s["k_max"] == 3
              and np.isclose(s["achieved_coverage"], 0.8)
              and np.isclose(s["next_gain"], 0.2), s)
        check("run: the selection's binning tolerance IS the merge tolerance",
              s["tol_ppm"] == summ["tol_ppm"] == SS.BATCH_TOL_PPM, (s.get("tol_ppm"),
                                                                    summ.get("tol_ppm")))
        sel = pd.read_csv(os.path.join(_d, "tables", "selected_samples.csv"))
        check("run: selected_samples.csv is in pick order with the cover columns",
              sel["pick"].tolist() == [1, 2, 3] and sel["role"].tolist() == ["cover"] * 3
              and sel["bins_new"].tolist() == [40, 20, 20]
              and np.isclose(sel["coverage"].tolist(), [0.4, 0.6, 0.8]).all(),
              sel.to_dict("records"))
        check("run: the CSV order IS the assignment order recorded in the summary",
              sel["sample_item_id"].tolist() == summ["sample_ids"] == res["sample_ids"],
              (sel["sample_item_id"].tolist(), summ["sample_ids"]))
        check("run: per-file stats keep the RESOLVED gate under height_gate_cps",
              all("height_gate_cps" in pf and "height_cutoff_cps" not in pf
                  for pf in summ["per_file"]), summ["per_file"][:1])
        check("run: every per-file run is told to leave the reagent-N re-read to the merge",
              len(_SEEN_KW) == 3 and all(k.get("reagent_n_relabel") is False for k in _SEEN_KW),
              [k.get("reagent_n_relabel") for k in _SEEN_KW])
        check("run: the merged ledger carries the vote and a tier_reason column",
              {"n_files_winner", "alternatives", "tier_reason"} <= set(res["merged"].columns),
              list(res["merged"].columns))
        check("run: batch_summary records the merged-level gates (none in negative mode)",
              summ.get("merge_gates") == {}, summ.get("merge_gates"))
        # 8 samples: fewer than the 10 spectra the persistence table needs, so the
        # 'auto' policy has nothing to derive the floor from and falls back to the
        # numeric default -- stamped as a NUMBER on every per-file cfg, with a
        # source that says why
        check("run: a profile with no opinion + nothing to derive from -> the package default 1x",
              all(c is not None and c.height_cutoff_x_edge == 1.0 for c in _SEEN_CFG)
              and summ.get("height_cutoff_x_edge") == 1.0
              and summ.get("height_cutoff_x_edge_source", "").startswith("the package default 1x")
              and "nothing to derive" in summ.get("height_cutoff_x_edge_source", "")
              and summ.get("gate") == {}, summ.get("height_cutoff_x_edge_source"))

    # ... and a profile that carries its own multiple hands it to every per-file
    # run and says so in the summary (the config-file path for a picker that
    # picks into the noise).
    _snap = (dict(P_PROF.PROFILES), dict(P_PROF._BY_ALIAS))
    P_PROF.register(P_PROF.ReagentProfile(
        name="BrPick", label="Br- picker", polarity="-",
        adducts=list(P_PROF.BR.adducts), normaliser="reagent",
        reagent_ion_re=P_PROF.BR.reagent_ion_re, ranges=P_PROF.BR.ranges,
        detect_adduct=None, height_cutoff_x_edge=5.0))
    try:
        with tempfile.TemporaryDirectory() as _d3:
            _SEEN_CFG.clear()
            AB.run(peaks=_PK, ts_peaks=_PK, reagent="BrPick", batch="test batch",
                   out_dir=_d3, k_min=2, k_max=3, min_gain=0.0, n_jobs=1,
                   log=lambda *a: None)
            summ3 = json.load(open(os.path.join(_d3, "batch_summary.json")))
            check("run: the profile's multiple reaches every per-file PassConfig",
                  len(_SEEN_CFG) == 3
                  and all(c.height_cutoff_x_edge == 5.0 for c in _SEEN_CFG),
                  [getattr(c, "height_cutoff_x_edge", None) for c in _SEEN_CFG])
            check("run: batch_summary records the multiple AND where it came from",
                  summ3.get("height_cutoff_x_edge") == 5.0
                  and summ3.get("height_cutoff_x_edge_source")
                  == "the BrPick reagent profile",
                  {k: summ3.get(k) for k in ("height_cutoff_x_edge",
                                             "height_cutoff_x_edge_source")})
        # an EXPLICIT cfg multiple outranks the profile at this layer -- including
        # one that equals the package default, which is the value a caller is
        # most likely to type and the one a `!= 1.0` test cannot distinguish from
        # "unset". (PassConfig.height_cutoff_x_edge is None when unset.)
        with tempfile.TemporaryDirectory() as _d5:
            _SEEN_CFG.clear()
            AB.run(peaks=_PK, ts_peaks=_PK, reagent="BrPick", batch="test batch",
                   out_dir=_d5, k_min=2, k_max=3, min_gain=0.0, n_jobs=1,
                   cfg=_PASSES.PassConfig(height_cutoff_x_edge=1.0),
                   log=lambda *a: None)
            summ4 = json.load(open(os.path.join(_d5, "batch_summary.json")))
            check("run: an explicit 1.0 beats a profile that says 5.0",
                  all(c.height_cutoff_x_edge == 1.0 for c in _SEEN_CFG)
                  and summ4.get("height_cutoff_x_edge") == 1.0
                  and "explicit" in summ4.get("height_cutoff_x_edge_source", ""),
                  {k: summ4.get(k) for k in ("height_cutoff_x_edge",
                                             "height_cutoff_x_edge_source")})
    finally:
        P_PROF.PROFILES.clear(); P_PROF.PROFILES.update(_snap[0])
        P_PROF._BY_ALIAS.clear(); P_PROF._BY_ALIAS.update(_snap[1])

    # ---- per-file cfg ISOLATION on the serial path (--jobs 1) ---------------
    # A.run MUTATES the cfg it is handed (noise edge, mechanism ids, the fitted
    # cal_mu/cal_sigma), and passes.calibrate RETURNS EARLY -- leaving the
    # previous fit in place -- when a file's backbone is smaller than cal_min_n.
    # The worker pool copies in _assign_one; the serial loop must do the same, or
    # file N+1 runs the calibrated mass gate on file N's calibration.
    _SEEN_CFG.clear()
    _seen_cal: list = []

    def _calibrating_assign(sid, context="ambient-air", **kw):
        _cfg = kw.get("cfg")
        _seen_cal.append(getattr(_cfg, "cal_mu", "no cfg"))
        _out = _fake_assign(sid, context, **kw)
        _cfg.cal_mu, _cfg.cal_sigma = -2.45, 0.3      # what calibrate() stamps
        return _out

    _A.run = _calibrating_assign
    try:
        with tempfile.TemporaryDirectory() as _d4:
            _parent = _PASSES.PassConfig()
            AB.run(peaks=_PK, ts_peaks=_PK, reagent="Br", batch="test batch",
                   out_dir=_d4, k_min=2, k_max=3, min_gain=0.0, n_jobs=1,
                   cfg=_parent, log=lambda *a: None)
        check("serial run: every file is handed its OWN cfg object",
              len(_SEEN_CFG) == 3 and len({id(c) for c in _SEEN_CFG}) == 3
              and all(c is not _parent for c in _SEEN_CFG),
              [id(c) for c in _SEEN_CFG])
        check("serial run: one file's fitted calibration never reaches the next",
              _seen_cal == [None, None, None], _seen_cal)
        check("serial run: the caller's cfg comes back unmutated by the assign",
              _parent.cal_mu is None and _parent.cal_sigma is None,
              (_parent.cal_mu, _parent.cal_sigma))
    finally:
        _A.run = _fake_assign

    # a per-SAMPLE table (samples.list) cannot be binned: selection refuses it
    # rather than silently falling back to some other rule.
    with tempfile.TemporaryDirectory() as _d2:
        try:
            AB.run(peaks=SS.sample_table(_PK), reagent="Br", out_dir=_d2,
                   n_jobs=1, log=lambda *a: None)
            check("run: per-sample peaks and no time series raises ValueError",
                  False, "no error")
        except ValueError as e:
            check("run: per-sample peaks and no time series raises ValueError",
                  "per-peak" in str(e), str(e))
finally:
    IO.connect, IO.fetch_peaks = _saved["connect"], _saved["fetch_peaks"]
    IO.estimate_offset, _A.run = _saved["estimate_offset"], _saved["run"]


# ---------------------------------------------------------------------------
# the POSITIVE path through run(): the hydrocarbon-on-N-cluster re-read and the
# ammonium/amine gate are decided ONCE on the merged ledger, and the merged row
# says what was done to it.
# ---------------------------------------------------------------------------
import inspect  # noqa: E402
import types  # noqa: E402

_stage = next(s_ for s_ in _A._STAGES if s_.name == "relabel_reagent_n")
check("per-file stage relabel_reagent_n runs by default (single-sample runs unchanged)",
      _stage.when(types.SimpleNamespace(reagent_n_relabel=True)))
check("per-file stage relabel_reagent_n stands down when a batch says so",
      not _stage.when(types.SimpleNamespace(reagent_n_relabel=False)))
check("assign.run exposes reagent_n_relabel, default True",
      inspect.signature(_A.run).parameters["reagent_n_relabel"].default is True)

_HC, _HC_MZ = "C15H22", _C.ion_mz("C15H22", "[M+NH4]+")       # a sesquiterpene on NH4
_HC_MH = _C.ion_mz("C15H22", "[M+H]+")                          # its own protonated form
_UR, _UR_MZ = "C11H20", _C.ion_mz("C11H20", "[M+(CH4N2O)H]+")   # a hydrocarbon on urea
_CALLS = []


def _positive_assign(sid, context="ambient-air", **kw):
    """Three files: every file reads C15H22 [M+NH4]+ and C11H20 uronium; only the
    FIRST also holds C15H22 [M+H]+ -- per file, the re-read would then have fired
    in two files and not the third (the Texas C15H22 case)."""
    _CALLS.append(sid)
    rows = [("nh4", _HC_MZ, 8.0e4), ("ur", _UR_MZ, 5.0e4)]
    if len(_CALLS) == 1:
        rows.append(("mh", _HC_MH, 3.0e4))
    led = _L.new_ledger(pd.DataFrame(rows, columns=["peak_id", "mz", "height"]))
    _L.commit_assignment(led, "nh4", neutral_formula=_HC, adduct="[M+NH4]+",
                         ion_formula="C15H26N+", ion_score=0.97, compound_score=0.97,
                         ppm_error=0.1, pass_no=1, method="cheminfo+grid",
                         confidence="High", commentary="stub")
    _L.commit_assignment(led, "ur", neutral_formula=_UR, adduct="[M+(CH4N2O)H]+",
                         ion_formula="C12H25N2O+", ion_score=0.98, compound_score=0.98,
                         ppm_error=0.1, pass_no=1, method="cheminfo+grid",
                         confidence="High", commentary="stub")
    if len(_CALLS) == 1:
        _L.commit_assignment(led, "mh", neutral_formula=_HC, adduct="[M+H]+",
                             ion_formula="C15H23+", ion_score=0.95, compound_score=0.95,
                             ppm_error=0.1, pass_no=1, method="cheminfo+grid",
                             confidence="High", commentary="stub")
    _T.apply_tiers(led)
    return {"ledger": led, "stats": {"noise_edge_cps": 4.0, "height_gate_cps": 10.0},
            "plausibility_audit": [], "summaries": {}, "problems": []}


IO.connect = lambda *a, **k: "CLIENT"
IO.fetch_peaks = lambda client, sid, use_cache=True: pd.DataFrame(
    {"peak_id": ["p1"], "mz": [_HC_MZ], "height": [1.0e5]})
IO.estimate_offset = lambda raw: 0.0
_A.run = _positive_assign
try:
    with tempfile.TemporaryDirectory() as _dp:
        resp = AB.run(peaks=_PK, ts_peaks=_PK, reagent="Ur", batch="test batch",
                      out_dir=_dp, k_min=2, k_max=3, min_gain=0.0, n_jobs=1,
                      log=lambda *a: None)
        mg = resp["merged"]
        summp = json.load(open(os.path.join(_dp, "batch_summary.json")))
        r_nh4 = mg.iloc[(mg["mz"] - _HC_MZ).abs().argmin()]
        r_ur = mg.iloc[(mg["mz"] - _UR_MZ).abs().argmin()]
        check("positive run: the per-file ledgers AGREE (no per-file re-read split the ion)",
              bool(r_nh4["formula_agree"]) and bool(r_nh4["ion_agree"])
              and r_nh4["n_files_winner"] == r_nh4["n_files_ion"] == r_nh4["n_files"] == 3
              and r_nh4["alternatives"] == "", r_nh4.to_dict())
        check("positive run: batch_summary counts ion disagreements (none here)",
              summp.get("ion_disagreements") == 0 and summp.get("formula_disagreements") == 0,
              (summp.get("ion_disagreements"), summp.get("formula_disagreements")))
        pf = pd.concat([pd.read_csv(os.path.join(_dp, "per_file", f"{sid}_ledger.csv"))
                        for sid in resp["sample_ids"]])
        check("positive run: every per-file ledger on disk keeps C15H22 [M+NH4]+",
              int(((pf["neutral_formula"] == _HC) & (pf["adduct"] == "[M+NH4]+")).sum()) == 3
              and not (pf["neutral_formula"] == "C15H25N").any(), pf["neutral_formula"].tolist())
        # C15H22 shows its own [M+H]+ in ONE file: pooled on the merged ledger, that
        # keeps the ammonium reading through the reagent-N pass for the whole batch;
        # the amine gate then has no trace to confirm the adduct against (the
        # synthetic TS holds no such ion) and, by its default-to-CHON policy, reads
        # it as the amine -- and SAYS so on the merged row.
        check("positive run: the reagent-N pass re-read ONE reading (C11H20) and kept C15H22, which protonates in one file",
              summp["merge_gates"]["reagent_n"] == {"reagent_n_relabeled": 1}, summp.get("merge_gates"))
        check("positive run: the amine gate's re-read of C15H22 [M+NH4]+ is explained in tier_reason",
              r_nh4["neutral_formula"] == "C15H25N" and r_nh4["adduct"] == "[M+H]+"
              and r_nh4["tier"] == "Candidate"
              and "protonated CHON" in str(r_nh4["tier_reason"]) and "C15H22" in str(r_nh4["tier_reason"]),
              r_nh4.to_dict())
        check("positive run: C11H20 uronium (no [M+H]+ anywhere) re-read ONCE, naming the reading it replaced",
              r_ur["neutral_formula"] == "C12H24N2O" and r_ur["adduct"] == "[M+H]+"
              and r_ur["tier"] == "Candidate" and r_ur["n_files_winner"] == 3
              and str(r_ur["tier_reason"]).startswith("re-read C11H20 [M+(CH4N2O)H]+ as [M+H]+ of C12H24N2O"),
              r_ur.to_dict())
        check("positive run: batch_summary carries both gates' counts",
              summp["merge_gates"]["amine"]["relabeled"] == 1
              and summp["merge_gates"]["amine"]["kept_covary"] == 0, summp["merge_gates"])
        check("positive run: merged_ledger.csv on disk has the vote + tier_reason columns",
              {"n_files_winner", "alternatives", "tier_reason"}
              <= set(pd.read_csv(os.path.join(_dp, "merged_ledger.csv"), nrows=1).columns))
finally:
    IO.connect, IO.fetch_peaks = _saved["connect"], _saved["fetch_peaks"]
    IO.estimate_offset, _A.run = _saved["estimate_offset"], _saved["run"]


# ---------------------------------------------------------------------------
# THE RESIDUAL STAGE end to end through run(): stubbed assign + IO, the real
# cover -> merge -> stamp -> residual universe -> residual cover -> second
# assignment -> ONE align over both stages -> summary / CSV / ledger provenance.
# ---------------------------------------------------------------------------
import concurrent.futures as _CF  # noqa: E402
import hashlib  # noqa: E402

_F2 = "C12H20O5"          # what a residual file assigns: an ion only that stage can bring


def _batch_table_h(spec):
    """Per-peak batch table: sample id -> {m/z: height}."""
    rows = []
    for i, (sid, mzh) in enumerate(spec.items()):
        t = _T0 + pd.Timedelta(minutes=10 * i)
        rows += [dict(sample_item_id=sid, sample_item_name=f"n_{sid}", datetime_utc=t,
                      mz=float(mz), height=float(h)) for mz, h in mzh.items()]
    return pd.DataFrame(rows)


# 3 exclusive 20-bin block PAIRS on a 20-bin background (every height 500 cps, so
# every sample's noise edge is 500) -- the cover fixture -- plus two samples the
# cover never picks (22 bins each < 40): bin 500 is bright in z1 (10x its edge),
# bin 502 bright in z2, bin 501 dim in both (1.2x); each bright bin stands at 20 %
# of its maximum in the other file. 8 samples keeps the persistence path off
# (< 10 spectra), as in the cover fixture above.
_BGH = {m: 500.0 for m in _BG}
_RSPEC = {}
for _i in range(3):
    _blk = {m: 500.0 for m in range(200 + 20 * _i, 220 + 20 * _i)}
    _RSPEC[f"a{_i}"] = {**_BGH, **_blk}
    _RSPEC[f"b{_i}"] = {**_BGH, **_blk}
_RSPEC["z1"] = {**_BGH, 500: 5000.0, 501: 600.0, 502: 1000.0}
_RSPEC["z2"] = {**_BGH, 500: 1000.0, 501: 600.0, 502: 5000.0}
_RPK = _batch_table_h(_RSPEC)
_EXTRA_M0: list = []      # (mz, neutral) rows EVERY fake ledger also assigns (stamp probes)


def _fake_assign_staged(sid, context="ambient-air", **kw):
    """assign.run stand-in: every file assigns _F; a residual file (z*) assigns
    _F2 as well -- a NEW ion the ledger can only gain through the residual stage."""
    _SEEN_CFG.append(kw.get("cfg"))
    rows = [("p1", _C.ion_mz(_F, "[M-H]-"), 1.0e5)]
    if sid.startswith("z"):
        rows.append(("p2", _C.ion_mz(_F2, "[M-H]-"), 5.0e3))
    rows += [(f"x{i}", mz, 2.0e3) for i, (mz, _nf) in enumerate(_EXTRA_M0)]
    led = _L.new_ledger(pd.DataFrame(rows, columns=["peak_id", "mz", "height"]))
    _L.commit_assignment(led, "p1", neutral_formula=_F, adduct="[M-H]-",
                         ion_formula="C10H15O5-", ion_score=0.9, compound_score=0.9,
                         ppm_error=0.1, pass_no=1, method="cheminfo+grid",
                         confidence="High", commentary="stub")
    if sid.startswith("z"):
        _L.commit_assignment(led, "p2", neutral_formula=_F2, adduct="[M-H]-",
                             ion_formula="C12H19O5-", ion_score=0.8, compound_score=0.8,
                             ppm_error=0.2, pass_no=1, method="cheminfo+grid",
                             confidence="High", commentary="stub")
    for i, (mz, nf) in enumerate(_EXTRA_M0):
        _L.commit_assignment(led, f"x{i}", neutral_formula=nf, adduct="[M-H]-",
                             ion_formula=nf + "-", ion_score=0.7, compound_score=0.7,
                             ppm_error=0.3, pass_no=1, method="cheminfo+grid",
                             confidence="Medium", commentary="stub")
    _T.apply_tiers(led)
    return {"ledger": led, "stats": {"noise_edge_cps": 500.0, "height_gate_cps": 500.0},
            "plausibility_audit": [], "summaries": {}, "problems": []}


class _FakePool:
    """A ProcessPoolExecutor stand-in that runs the REAL worker entry points
    (`_worker_init`, `_assign_one`) in this process -- so the stubbed assign is
    visible to them -- and hands the futures back already finished, in an order
    `as_completed` does not control. What it exercises is everything the pool
    path does around the workers: the raw-TS parquet hand-off, the banner and
    per-future 'done' lines, and the reduce in sample order."""

    def __init__(self, max_workers=None, mp_context=None, initializer=None, initargs=()):
        initializer(*initargs)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def submit(self, fn, *args):
        f = _CF.Future()
        f.set_result(fn(*args))
        return f


def _run_staged(d, **kw):
    lines = []
    res = AB.run(peaks=_RPK, ts_peaks=_RPK, reagent="Br", batch="test batch", out_dir=d,
                 k_min=2, k_max=2, min_gain=0.0, log=lines.append, **kw)
    summ = json.load(open(os.path.join(d, "batch_summary.json")))
    return res, summ, lines


def _sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


_ARTS = ("merged_ledger.csv", "tables/jitter.csv", "tables/selected_samples.csv",
         "tables/residual_bins.csv")
_saved_pool = _CF.ProcessPoolExecutor
IO.connect = lambda *a, **k: "CLIENT"
IO.fetch_peaks = lambda client, sid, use_cache=True: pd.DataFrame(
    {"peak_id": ["p1"], "mz": [_C.ion_mz(_F, "[M-H]-")], "height": [1.0e5]})
IO.estimate_offset = lambda raw: 0.0
_A.run = _fake_assign_staged
try:
    # ---- residual ON (the default): the bright uncovered bins get their samples ----
    with tempfile.TemporaryDirectory() as _d:
        _SEEN_CFG.clear()
        res, summ, lines = _run_staged(_d, n_jobs=1)
        r = summ["selection"]["residual"]
        check("residual run: the cover stops at k_max=2 (a0, a1) and the z files stay unpicked",
              summ["selection"]["k"] == 2 and summ["selection"]["stop_reason"] == "k_max"
              and summ["n_files_by_stage"] == {"cover": 2, "residual": 2},
              (summ["selection"].get("k"), summ.get("n_files_by_stage")))
        check("residual run: the funnel -- 23 bins in no assigned file, 21 below the 5x floor, 2 targeted",
              r["n_universe"] == 83 and r["n_uncovered"] == 23 and r["n_below_floor"] == 21
              and r["n_explained"] == 0 and r["n_bins_residual"] == 2, r)
        check("residual run: the cover of those bins -- z1 then z2, 100 %, exhausted",
              r["sample_ids"] == ["z1", "z2"] and r["k"] == 2 and r["k_max"] == SS.RESIDUAL_K_MAX
              and np.isclose(r["coverage_of_residual"], 1.0) and r["stop_reason"] == "exhausted"
              and r["frac_of_max"] == 0.5, r)
        check("residual run: the floor is 5x the sample's edge (the gate is 1x, so not raised)",
              r["floor"]["min_x_edge"] == 5.0 and r["floor"]["min_cps"] is None
              and r["floor"]["edge_median_cps"] == 500.0
              and "5x the sample's noise edge" in r["floor"]["source"], r["floor"])
        check("residual run: n_files / sample_ids count BOTH stages, in assignment order",
              summ["n_files"] == 4 and summ["sample_ids"] == ["a0", "a1", "z1", "z2"]
              == res["sample_ids"], summ["sample_ids"])
        check("residual run: every file went through the per-file path with its own cfg",
              len(_SEEN_CFG) == 4 and len({id(c) for c in _SEEN_CFG}) == 4)
        check("residual run: per-file ledgers exist for the residual files, with stage on their stats",
              all(os.path.exists(os.path.join(_d, "per_file", f"{sid}_ledger.csv"))
                  for sid in ("z1", "z2"))
              and [pf["stage"] for pf in summ["per_file"]] == ["cover", "cover", "residual", "residual"],
              [pf.get("stage") for pf in summ["per_file"]])
        merged = pd.read_csv(os.path.join(_d, "merged_ledger.csv"))
        row_c = merged[merged["neutral_formula"] == _F].iloc[0]
        row_r = merged[merged["neutral_formula"] == _F2].iloc[0]
        check("residual run: ONE align over both stages -- the cover ion now spans 4 files",
              len(merged) == 2 and row_c["n_files"] == 4 and row_c["srcs"] == "a0,a1,z1,z2",
              merged.to_dict("records"))
        check("residual run: the merged ledger gains the residual files' ion with stage 'residual'",
              row_r["stage"] == "residual" and row_r["n_files"] == 2 and row_r["srcs"] == "z1,z2"
              and row_c["stage"] == "cover", merged[["neutral_formula", "stage", "n_files"]].to_dict("records"))
        check("residual run: `stage` sits after `srcs` in the ledger, and the summary counts by stage",
              list(merged.columns).index("stage") == list(merged.columns).index("srcs") + 1
              and summ["merged_by_stage"] == {"cover": 1, "residual": 1}, summ.get("merged_by_stage"))
        sel = pd.read_csv(os.path.join(_d, "tables", "selected_samples.csv"))
        check("residual run: selected_samples.csv gains the picks, role 'residual', numbered on",
              sel["sample_item_id"].tolist() == ["a0", "a1", "z1", "z2"]
              and sel["pick"].tolist() == [1, 2, 3, 4]
              and sel["role"].tolist() == ["cover", "cover", "residual", "residual"]
              and sel["bins_new"].tolist() == [40, 20, 1, 1]
              and np.isclose(sel["coverage"].tolist()[2:], [0.5, 1.0]).all(),
              sel.to_dict("records"))
        rb = pd.read_csv(os.path.join(_d, "tables", "residual_bins.csv"))
        check("residual run: tables/residual_bins.csv lists the targeted bins and who carries each",
              rb["bin_mz"].round(0).tolist() == [500.0, 502.0]
              and rb["covered_by"].tolist() == ["z1", "z2"] and rb["max_x_edge"].tolist() == [10.0, 10.0],
              rb.to_dict("records"))
        check("residual run: the log counts on through the stage (progress.py reads these)",
              "[phase] residual" in lines
              and any(ln.startswith("[assign_batch] (3/4) assigning z1") for ln in lines)
              and "[assign_batch] (4/4) done z2" in lines
              and any(ln.startswith("[assign_batch] residual:") for ln in lines)
              and lines.index("[phase] residual") < lines.index("[assign_batch] (4/4) done z2"),
              [ln for ln in lines if "(3/4)" in ln or "(4/4)" in ln or "phase" in ln])
        check("residual run: the result hands the picks and the stage record back",
              res["residual_samples"]["sample_item_id"].tolist() == ["z1", "z2"]
              and res["stages"] == {"a0": "cover", "a1": "cover", "z1": "residual", "z2": "residual"})
        check("residual run: the stage's yield is logged on its own line after DONE",
              any(ln.startswith("[assign_batch] residual stage: 2 file(s), 1 ion(s) it alone holds")
                  for ln in lines)
              and [i for i, ln in enumerate(lines) if ln.startswith("[assign_batch] DONE:")]
              < [i for i, ln in enumerate(lines) if "ion(s) it alone holds" in ln],
              [ln for ln in lines if "it alone holds" in ln or ln.startswith("[assign_batch] DONE")])
        _sha_serial = {a: _sha(os.path.join(_d, a)) for a in _ARTS}
        _sha_serial_pf = {sid: _sha(os.path.join(_d, "per_file", f"{sid}_ledger.csv"))
                          for sid in ("a0", "a1", "z1", "z2")}

    # ---- the budget caps the stage ----------------------------------------------
    with tempfile.TemporaryDirectory() as _d:
        _SEEN_CFG.clear()
        res, summ, lines = _run_staged(_d, n_jobs=1, residual_k_max=1)
        r = summ["selection"]["residual"]
        merged = pd.read_csv(os.path.join(_d, "merged_ledger.csv"))
        check("residual run: residual_k_max=1 caps the picks (z1 only), stop 'k_max', 50 %",
              r["k"] == 1 and r["sample_ids"] == ["z1"] and r["stop_reason"] == "k_max"
              and np.isclose(r["coverage_of_residual"], 0.5) and summ["n_files"] == 3
              and len(_SEEN_CFG) == 3, r)
        check("residual run: ...and the residual ion is then a single-file row",
              merged[merged["neutral_formula"] == _F2].iloc[0]["n_files"] == 1
              and merged[merged["neutral_formula"] == _F2].iloc[0]["stage"] == "residual")

    # ---- the floor: excludes the dim bin; an absolute override; too high -> nothing ----
    with tempfile.TemporaryDirectory() as _d:
        res, summ, lines = _run_staged(_d, n_jobs=1, residual_min_x_edge=1.1)
        r = summ["selection"]["residual"]
        check("residual run: a 1.1x floor admits the 1.2x bin too (3 targeted), never the 1x tail",
              r["n_bins_residual"] == 3 and r["n_below_floor"] == 20 and r["sample_ids"] == ["z1", "z2"],
              r)
    with tempfile.TemporaryDirectory() as _d:
        res, summ, lines = _run_staged(_d, n_jobs=1, residual_min_cps=5000.0)
        r = summ["selection"]["residual"]
        check("residual run: --residual-min-cps is an absolute floor (5000 cps: the two bright bins)",
              r["floor"]["min_x_edge"] is None and r["floor"]["min_cps"] == 5000.0
              and r["n_bins_residual"] == 2 and r["sample_ids"] == ["z1", "z2"], r["floor"])
    with tempfile.TemporaryDirectory() as _d:
        _SEEN_CFG.clear()
        res, summ, lines = _run_staged(_d, n_jobs=1, residual_min_x_edge=20.0)
        r = summ["selection"]["residual"]
        merged = pd.read_csv(os.path.join(_d, "merged_ledger.csv"))
        check("residual run: a floor nothing reaches -> no target, no extra file, stop 'empty'",
              r["n_bins_residual"] == 0 and r["k"] == 0 and r["stop_reason"] == "empty"
              and r["sample_ids"] == [] and summ["n_files"] == 2 and len(_SEEN_CFG) == 2
              and summ["n_files_by_stage"] == {"cover": 2, "residual": 0}, r)
        check("residual run: ...the stage column is still there (all cover) and the bins table is empty",
              merged["stage"].tolist() == ["cover"]
              and len(pd.read_csv(os.path.join(_d, "tables", "residual_bins.csv"))) == 0
              and pd.read_csv(os.path.join(_d, "tables", "selected_samples.csv"))["role"].tolist()
              == ["cover", "cover"])
    # a gate multiple above the floor raises the floor to it (a bin no file would
    # admit is not worth a file)
    with tempfile.TemporaryDirectory() as _d:
        res, summ, lines = _run_staged(_d, n_jobs=1, cfg=_PASSES.PassConfig(height_cutoff_x_edge=8.0))
        r = summ["selection"]["residual"]
        check("residual run: the floor is never below the run's gate multiple (5x -> 8x, said so)",
              r["floor"]["min_x_edge"] == 8.0 and "raised from 5x" in r["floor"]["source"]
              and r["n_bins_residual"] == 2, r["floor"])
    with tempfile.TemporaryDirectory() as _d:
        res, summ, lines = _run_staged(_d, n_jobs=1, cfg=_PASSES.PassConfig(height_cutoff_x_edge=12.0))
        r = summ["selection"]["residual"]
        check("residual run: ...and at 12x the 10x bins are below it: nothing targeted",
              r["floor"]["min_x_edge"] == 12.0 and r["n_bins_residual"] == 0 and r["k"] == 0, r)

    # ---- sidelobe tiers through run(): a suspect is targeted LAST, and recorded ----
    # a saturating peak 5 mDa above bin 500 in every file (covered, in a0); bin 500
    # in z1 is 200x dimmer than it and shares only 2 samples with it -> a static-rule
    # suspect; bin 502 stays clean -> z2 (clean) is assigned before z1 (suspect)
    _RSPEC_SL = {sid: {**mzh, 500.005: 1.0e6} for sid, mzh in _RSPEC.items()}
    _RPK_SL = _batch_table_h(_RSPEC_SL)
    with tempfile.TemporaryDirectory() as _d:
        lines = []
        res = AB.run(peaks=_RPK_SL, ts_peaks=_RPK_SL, reagent="Br", batch="test batch",
                     out_dir=_d, k_min=2, k_max=2, min_gain=0.0, n_jobs=1, log=lines.append)
        summ = json.load(open(os.path.join(_d, "batch_summary.json")))
        r = summ["selection"]["residual"]
        rb = pd.read_csv(os.path.join(_d, "tables", "residual_bins.csv"))
        check("residual run: the static-rule suspect is counted and targeted after the clean bin",
              r["n_suspect"] == 1 and r["n_sidelobe"] == 0 and r["n_bins_residual"] == 2
              and r["sample_ids"] == ["z2", "z1"] and summ["sample_ids"] == ["a0", "a1", "z2", "z1"],
              r)
        check("residual run: residual_bins.csv carries the tier, the neighbour and the pairs",
              rb["tier"].tolist() == ["suspect", "clean"] and rb["covered_by"].tolist() == ["z1", "z2"]
              and rb["sidelobe_of"].round(3).tolist()[:1] == [500.005] and rb["n_pairs"].tolist() == [2, 0],
              rb.to_dict("records"))
        check("residual run: the log names the suspect",
              any("1 static-rule sidelobe suspect(s), covered last" in ln for ln in lines))

    # ---- (b) a bin the whole-batch stamp explains is not re-targeted ----------------
    _EXTRA_M0[:] = [(500.0, "C20H20O10")]     # every cover ledger assigns an ion AT bin 500
    with tempfile.TemporaryDirectory() as _d:
        res, summ, lines = _run_staged(_d, n_jobs=1)
        r = summ["selection"]["residual"]
        check("residual run: bin 500 is stamped from the cover's ledger -> explained, only 502 targeted",
              r["n_explained"] == 1 and r["n_bins_residual"] == 1 and r["sample_ids"] == ["z2"]
              and summ["n_files"] == 3, r)
    _EXTRA_M0[:] = []

    # ---- residual OFF: today's cover-only run, byte for byte in its schema -------------
    with tempfile.TemporaryDirectory() as _d:
        _SEEN_CFG.clear()
        res, summ, lines = _run_staged(_d, n_jobs=1, residual=False)
        merged = pd.read_csv(os.path.join(_d, "merged_ledger.csv"))
        sel = pd.read_csv(os.path.join(_d, "tables", "selected_samples.csv"))
        check("residual off: two cover files only, no second stage anywhere in the summary",
              summ["n_files"] == 2 and summ["sample_ids"] == ["a0", "a1"] and len(_SEEN_CFG) == 2
              and "residual" not in summ["selection"] and "n_files_by_stage" not in summ
              and "merged_by_stage" not in summ
              and all("stage" not in pf for pf in summ["per_file"]), summ["selection"].keys())
        check("residual off: no `stage` column, no residual_bins.csv, cover roles only",
              "stage" not in merged.columns and len(merged) == 1
              and not os.path.exists(os.path.join(_d, "tables", "residual_bins.csv"))
              and sel["role"].tolist() == ["cover", "cover"] and sel["pick"].tolist() == [1, 2],
              list(merged.columns))
        check("residual off: no residual phase or stage lines in the log",
              "[phase] residual" not in lines
              and not any("it alone holds" in ln or ln.startswith("[assign_batch] residual")
                          for ln in lines))
        check("residual off: run() still hands back the (empty) stage fields",
              res["residual_samples"] is None and res["stages"] == {"a0": "cover", "a1": "cover"})
        _sha_off_serial = {a: _sha(os.path.join(_d, a)) for a in _ARTS if a != "tables/residual_bins.csv"}

    # ---- determinism: the pool path is byte-identical to the serial one, both ways ----
    _CF.ProcessPoolExecutor = _FakePool
    with tempfile.TemporaryDirectory() as _d:
        _SEEN_CFG.clear()
        res, summ, lines = _run_staged(_d, n_jobs=2)
        check("residual + pool: both stages ran through the pool path (two banners, N counted on)",
              sum(ln.startswith("[assign_batch] parallel: 2 worker processes") for ln in lines) == 2
              and any(ln.endswith("over 2 samples") for ln in lines)
              and any(ln.endswith("over 4 samples") for ln in lines)
              and ("[assign_batch] (4/4) done z2" in lines or "[assign_batch] (3/4) done z2" in lines),
              [ln for ln in lines if "parallel" in ln or "done z" in ln])
        check("residual + pool: merged / jitter / selected / residual_bins are byte-identical to serial",
              {a: _sha(os.path.join(_d, a)) for a in _ARTS} == _sha_serial,
              {a: (_sha(os.path.join(_d, a))[:8], _sha_serial[a][:8]) for a in _ARTS})
        check("residual + pool: every per-file ledger is byte-identical to serial",
              {sid: _sha(os.path.join(_d, "per_file", f"{sid}_ledger.csv"))
               for sid in ("a0", "a1", "z1", "z2")} == _sha_serial_pf)
        check("residual + pool: the final _batch_ts.parquet is the ANNOTATED one, not the worker copy",
              "ion_formula" in pd.read_parquet(os.path.join(_d, "per_file", "_batch_ts.parquet")).columns)
        check("residual + pool: n_jobs recorded is the run's, sample order unchanged",
              summ["n_jobs"] == 2 and summ["sample_ids"] == ["a0", "a1", "z1", "z2"])
    with tempfile.TemporaryDirectory() as _d:
        res, summ, lines = _run_staged(_d, n_jobs=2, residual=False)
        check("residual off + pool: byte-identical to the serial cover-only run",
              {a: _sha(os.path.join(_d, a)) for a in _sha_off_serial} == _sha_off_serial)
    _CF.ProcessPoolExecutor = _saved_pool

    # ---- no time series: the stage has nothing to read and says so -------------------
    with tempfile.TemporaryDirectory() as _d:
        lines = []
        AB.run(peaks=_RPK, ts_peaks=None, reagent="Br", batch="test batch", out_dir=_d,
               k_min=2, k_max=2, min_gain=0.0, n_jobs=1, log=lines.append)
        summ = json.load(open(os.path.join(_d, "batch_summary.json")))
        r = summ["selection"]["residual"]
        check("residual run without a TS: skipped, recorded as such, cover files only",
              r["k"] == 0 and r["stop_reason"] == "empty" and "skipped" in r
              and summ["n_files"] == 2 and "[phase] residual" in lines, r)
finally:
    _CF.ProcessPoolExecutor = _saved_pool
    _EXTRA_M0[:] = []
    os.environ.pop("PEAKY_MATCH_WORKERS", None)
    IO.connect, IO.fetch_peaks = _saved["connect"], _saved["fetch_peaks"]
    IO.estimate_offset, _A.run = _saved["estimate_offset"], _saved["run"]


def test_all():
    assert FAIL == 0, f"{FAIL} checks failed"


if __name__ == "__main__":
    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)
