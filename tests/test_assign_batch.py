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
# The minority-winner defect (15-file uronium run): the old rule ranked Assigned-file count
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
# m/z 252.123 on that run: C13H14O4 [M+NH4]+ and C13H17NO4 [M+H]+ are ONE ion
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

# a KNOWN-SPECIES identity is no longer exempt from the vote: the vote is the
# files' count and corroboration, nothing else. What the curated exemption used
# to do is decided ONCE for the batch by the evidence the files pooled
# (`known_evidence` -> `lock_known_species`), on the merged ledger, AFTER the
# vote. The case: at m/z 579.171 the D7 cyclosiloxane urea adduct, locked in ONE
# file on its 29Si/30Si envelope, faced a 9-file C27H30O14 the per-file engine
# itself flags as an O14 monster -- and in those 9 files pass 0 had anchored D7
# on-cal but could not test it (a single channel, the twin below the floor).
check("align no longer takes a curated set (the exemption is gone)",
      "curated" not in __import__("inspect").signature(AB.align).parameters
      and not hasattr(AB, "_curated_neutrals") and not hasattr(AB, "_CURATED_METHODS"))
prot = {**_files(9, 579.1710, "C27H30O14", "[M+H]+", "Candidate", 0.99),
        **_files(1, 579.1710, "C14H42O7Si7", "[M+(CH4N2O)H]+", "Assigned", 0.83, start=9)}
mp, _ = AB.align(prot, tol_ppm=6.0)
check("vote: without the exemption the 9-file grid reading wins the vote itself, no note",
      mp.iloc[0]["neutral_formula"] == "C27H30O14" and mp.iloc[0]["n_files_winner"] == 9
      and pd.isna(mp.iloc[0]["tier_reason"])
      and mp.iloc[0]["alternatives"] == "C14H42O7Si7 [M+(CH4N2O)H]+ x1 Assigned 0.83",
      mp.to_dict("records"))
_D7 = dict(neutral="C14H42O7Si7", adduct="[M+(CH4N2O)H]+", family="cyclosiloxane",
           label="tetradecamethylcycloheptasiloxane (D7)", mz=579.1710)


def _ev(src, verdict, why, ion_score=0.83):
    return dict(_D7, src=src, verdict=verdict, why=why, ion_score=ion_score,
                tier="Assigned" if verdict == "confirmed" else None,
                admitted_by="height" if verdict == "confirmed" else None,
                occurrence=0.9 if verdict == "confirmed" else None)


pool = ([dict(_ev("v09", "confirmed", "corroborated by a confirmed 29Si/30Si envelope (2 satellites)"),
              n_channels=1, n_satellites=2)]
        + [_ev(f"v{i:02d}", "deferred", "single channel; the Si twin is predicted below the detection floor")
           for i in range(9)])
mk = mp.copy()
g = AB.lock_known_species(mk, pool, tol_ppm=6.0, log=lambda *a: None)
row = mk.iloc[0]
check("lock: confirmed in 1 file (two satellite lines), untestable in 9, refuted in 0 -> the merged row is D7, Assigned",
      row["neutral_formula"] == "C14H42O7Si7" and row["adduct"] == "[M+(CH4N2O)H]+"
      and row["tier"] == "Assigned" and abs(float(row["ion_score"]) - 0.83) < 1e-9
      and row["n_files_winner"] == 1 and row["n_files_ion"] == 1 and row["n_files"] == 10,
      row.to_dict())
check("lock: the vote's winner moves to the head of the alternatives, the known reading leaves them",
      row["alternatives"] == "C27H30O14 [M+H]+ x9 Candidate 0.99", repr(row["alternatives"]))
_tr = str(row["tier_reason"])
check("lock: the row says the evidence and what it overrode",
      _tr.startswith("known species decided once for the batch: tetradecamethylcycloheptasiloxane (D7) "
                     "(C14H42O7Si7 [M+(CH4N2O)H]+) -- confirmed in 1 file (corroborated by a confirmed "
                     "29Si/30Si envelope (2 satellites)); could not test it in 9 files (single channel; "
                     "the Si twin is predicted below the detection floor)")
      and _tr.endswith("; kept over the 9-file C27H30O14 [M+H]+ reading (vote 1 of 10 files)"), _tr)
check("lock: counts", g == {"pooled": 1, "locked": 1, "confirmed_kept": 0, "conflict": 0,
                            "lead_only": 0, "mass_only_outvoted": 0, "no_cluster": 0}, g)
check("lock: admitted_by / occurrence come from the confirmed file's row, not the loser's",
      row["admitted_by"] == "height" and abs(float(row["occurrence"]) - 0.9) < 1e-9, row.to_dict())
# ONE channel and ONE satellite line in every confirming file (the real D7: two
# files, the urea adduct alone, the 29Si line alone, the 30Si line never testable
# at that intensity -- J12) -> the reading is locked, the tier capped at Candidate
pool1 = ([dict(_ev(f"v{i:02d}", "confirmed", "corroborated by a confirmed 29Si/30Si envelope (1 satellite)"),
               n_channels=1, n_satellites=1) for i in (8, 9)]
         + [_ev(f"v{i:02d}", "deferred", "29Si and 30Si lines predicted under 4x the floor") for i in range(8)])
mk1 = mp.copy()
g1 = AB.lock_known_species(mk1, pool1, tol_ppm=6.0, log=lambda *a: None)
check("lock: one channel + one satellite in every confirming file -> locked as D7 but capped Candidate, said on the row",
      mk1.iloc[0]["neutral_formula"] == "C14H42O7Si7" and mk1.iloc[0]["tier"] == "Candidate" and g1["locked"] == 1
      and str(mk1.iloc[0]["tier_reason"]).endswith("(vote 2 of 10 files); capped Candidate (one ion channel and one "
                                                   "satellite line in every confirming file: two independent lines "
                                                   "are needed for Assigned)"),
      mk1.iloc[0].to_dict())
# ... and a second channel in ONE confirming file is enough for Assigned
pool2 = [dict(p) for p in pool1]; pool2[0]["n_channels"] = 2
mk2 = mp.copy(); AB.lock_known_species(mk2, pool2, tol_ppm=6.0, log=lambda *a: None)
check("lock: a second channel in one confirming file -> Assigned, no cap note",
      mk2.iloc[0]["tier"] == "Assigned" and "capped" not in str(mk2.iloc[0]["tier_reason"]), mk2.iloc[0].to_dict())
# the same cap on a row the vote already gave to the species (confirmed_kept)
kept1, _ = AB.align(_files(2, 283.0960, "C6H18O3Si3", "[M+(CH4N2O)H]+", "Assigned", 0.9), tol_ppm=6.0)
gk = AB.lock_known_species(kept1, [dict(neutral="C6H18O3Si3", adduct="[M+(CH4N2O)H]+", family="cyclosiloxane",
                                        label="D3", src=f"v{i:02d}", mz=283.0960, verdict="confirmed",
                                        why="corroborated by a confirmed 29Si/30Si envelope (1 satellite)",
                                        summary="corroborated by a confirmed 29Si/30Si envelope (1 satellite)",
                                        n_channels=1, n_satellites=1, ion_score=0.9, tier="Assigned",
                                        admitted_by="height", occurrence=0.5) for i in range(2)],
                           tol_ppm=6.0, log=lambda *a: None)
check("lock: a kept known row backed by one channel and one satellite everywhere is capped Candidate too",
      gk["confirmed_kept"] == 1 and kept1.iloc[0]["tier"] == "Candidate"
      and str(kept1.iloc[0]["tier_reason"]).endswith("two independent lines are needed for Assigned)"),
      kept1.iloc[0].to_dict())
# REFUTED anywhere: a file that could show the twin and did not contradicts the
# lock; the species is left to the vote and the row says so. Sulfolane at
# 181.065 (34S-confirmed in one file) vs fluorenone C13H8O [M+H]+ Assigned in
# nine: the nine bright files show no 34S, so the count decides -- by evidence.
sulf = {**_files(9, 181.0647, "C13H8O", "[M+H]+", "Assigned", 0.99),
        **_files(1, 181.0647, "C4H8O2S", "[M+(CH4N2O)H]+", "Assigned", 0.95, start=9)}
ms, _ = AB.align(sulf, tol_ppm=6.0)
_SF = dict(neutral="C4H8O2S", adduct="[M+(CH4N2O)H]+", family="indoor_sulfur", label="sulfolane",
           mz=181.0647, tier=None, admitted_by=None, occurrence=None)
pool_s = ([dict(_SF, src="v09", verdict="confirmed", ion_score=0.95, tier="Assigned",
                admitted_by="height", occurrence=0.8,
                why="corroborated by a confirmed 34S envelope (single channel)")]
          + [dict(_SF, src=f"v{i:02d}", verdict="refuted", ion_score=0.9,
                  why="single channel; the S twin is predicted above the floor and was not matched")
             for i in range(9)])
gs = AB.lock_known_species(ms, pool_s, tol_ppm=6.0, log=lambda *a: None)
check("lock: refuted in 9 files -> not locked, the vote's fluorenone stands, the conflict is on the row",
      ms.iloc[0]["neutral_formula"] == "C13H8O" and ms.iloc[0]["tier"] == "Assigned"
      and gs["conflict"] == 1 and gs["locked"] == 0
      and str(ms.iloc[0]["tier_reason"]) == ("known species sulfolane (C4H8O2S [M+(CH4N2O)H]+) confirmed "
                                             "in 1 file but refuted in 9 (single channel; the S twin is "
                                             "predicted above the floor and was not matched); left to the vote"),
      ms.iloc[0].to_dict())
# ... and when the vote's winner IS the conflicted species (the other files left
# the peak unexplained, so the one confirming file "won" 1-0) the merged tier is
# capped: one file's Assigned cannot stand for a batch that refuted it in nine
alone, _ = AB.align(_files(1, 181.0647, "C4H8O2S", "[M+(CH4N2O)H]+", "Assigned", 0.95, start=9), tol_ppm=6.0)
ga = AB.lock_known_species(alone, pool_s, tol_ppm=6.0, log=lambda *a: None)
check("lock: the conflicted species as the row's own reading -> capped Candidate, said on the row",
      alone.iloc[0]["neutral_formula"] == "C4H8O2S" and alone.iloc[0]["tier"] == "Candidate"
      and ga["conflict"] == 1
      and str(alone.iloc[0]["tier_reason"]).endswith("; left to the vote; capped Candidate (refuted in more "
                                                      "files than confirmed)"),
      alone.iloc[0].to_dict())
# a summary (the reason without the file's numbers) is what the note counts files by
pool_sum = [dict(_D7, src=f"v{i:02d}", verdict="deferred", ion_score=0.8, tier=None, admitted_by=None,
                 occurrence=None, why=f"single channel; 29Si predicted at {60 + i} cps, under 2x the 60-cps floor",
                 summary="29Si and 30Si lines predicted under 2x the floor") for i in range(3)] + \
           [dict(_D7, src="v09", verdict="deferred", ion_score=0.8, tier=None, admitted_by=None, occurrence=None,
                 why="single channel; 29Si line at 0.28x the parent (predicted 0.36) -- present in the ledger, not credited by the scorer",
                 summary="29Si line present at the predicted ratio, not credited by the scorer")]
check("_pool_summary: counts files per distinct summary, never repeating a file's numbers",
      AB._pool_summary([], pool_sum, []) == ("could not test it in 4 files (29Si and 30Si lines predicted "
                                             "under 2x the floor [3 files] / 29Si line present at the "
                                             "predicted ratio, not credited by the scorer [1 file])"),
      AB._pool_summary([], pool_sum, []))
# DEFERRED only (never confirmed anywhere): a lead on the row, not a lock --
# tricresyl phosphate anchored single-channel in 3 files against a grid reading
weak, _ = AB.align(_files(3, 429.1574, "C15H24N4O10", "[M+H]+", "Candidate", 0.98), tol_ppm=6.0)
pool_w = [dict(neutral="C21H21O4P", adduct="[M+(CH4N2O)H]+", family="organophosphate",
               label="tricresyl phosphate (TMPP / TCrP)", src=f"v{i:02d}", mz=429.1574,
               verdict="deferred", why="single channel; no diagnostic twin to test (monoisotopic)",
               ion_score=0.7, tier=None, admitted_by=None, occurrence=None) for i in range(3)]
gw = AB.lock_known_species(weak, pool_w, tol_ppm=6.0, log=lambda *a: None)
check("lock: a lead anchored in 3 files but confirmed in none is noted, not locked",
      weak.iloc[0]["neutral_formula"] == "C15H24N4O10" and gw["lead_only"] == 1 and gw["locked"] == 0
      and str(weak.iloc[0]["tier_reason"]) == ("known-species lead: tricresyl phosphate (TMPP / TCrP) "
                                               "(C21H21O4P [M+(CH4N2O)H]+) anchored on-cal in 3 files but "
                                               "never corroborated (single channel; no diagnostic twin to "
                                               "test (monoisotopic)); not locked"),
      weak.iloc[0].to_dict())
# confirmed AND already the vote's winner: the row gains the evidence, nothing else moves
uni, _ = AB.align({"a": m0([(500.0, "C21H21O4P", "[M+(CH4N2O)H]+", "Assigned", 0.9)])}, tol_ppm=6.0)
_TCP = dict(neutral="C21H21O4P", adduct="[M+(CH4N2O)H]+", family="organophosphate",
            label="tricresyl phosphate (TMPP / TCrP)", src="a", mz=500.0, verdict="confirmed",
            why="corroborated by 2 ion channels", ion_score=0.9, tier="Assigned",
            admitted_by="height", occurrence=1.0, n_channels=2, n_satellites=0)
gu = AB.lock_known_species(uni, [_TCP], tol_ppm=6.0, log=lambda *a: None)
check("lock: a confirmed species the vote already chose keeps its row and gains the evidence note",
      gu["confirmed_kept"] == 1 and uni.iloc[0]["neutral_formula"] == "C21H21O4P"
      and uni.iloc[0]["alternatives"] == ""
      and uni.iloc[0]["tier_reason"] == ("known species decided once for the batch: tricresyl phosphate "
                                         "(TMPP / TCrP) (C21H21O4P [M+(CH4N2O)H]+) -- confirmed in 1 file "
                                         "(corroborated by 2 ion channels)"),
      uni.iloc[0].to_dict())
gn = AB.lock_known_species(uni, [dict(_TCP, neutral="C6H15O4P", adduct="[M+H]+", mz=183.078)],
                           tol_ppm=6.0, log=lambda *a: None)
# a MASS-ONLY family (the PFCAs: no twin, no second channel demanded) carries
# nothing to pool beyond the count of files it fitted in: a PFCA [M-H]- on-cal in
# 2 files of a ~4k TOF must not displace an 11-file 81Br-corroborated CHOS [M+Br]-
# reading 6 ppm away -- the vote stands, the row says so
pf = {**_files(11, 362.9717, "C12H12O8", "[M+Br]-", "Assigned", 0.91),
      **_files(2, 362.9700, "C7HF13O2", "[M-H]-", "Assigned", 0.98, start=11)}
mpf, _ = AB.align(pf, tol_ppm=6.0)
_PF = dict(neutral="C7HF13O2", adduct="[M-H]-", family="perfluoroacid", label="perfluoro-C7 acid (PFCA)",
           verdict="confirmed", why=AB.MASS_ONLY_ROUTE, summary=AB.MASS_ONLY_ROUTE, ion_score=0.98,
           tier="Assigned", admitted_by="height", occurrence=0.6)
gpf = AB.lock_known_species(mpf, [dict(_PF, src="v11", mz=362.9700), dict(_PF, src="v12", mz=362.9701)],
                            tol_ppm=6.0, log=lambda *a: None)
check("lock: a mass-only species confirmed in 2 files does NOT override the 11-file vote; noted",
      mpf.iloc[0]["neutral_formula"] == "C12H12O8" and mpf.iloc[0]["tier"] == "Assigned"
      and gpf["mass_only_outvoted"] == 1 and gpf["locked"] == 0
      and str(mpf.iloc[0]["tier_reason"]) == ("mass-only known species perfluoro-C7 acid (PFCA) (C7HF13O2 "
                                              "[M-H]-) anchored on-cal in 2 files; the vote's 11-file C12H12O8 "
                                              "[M+Br]- reading stands (exact mass alone cannot overrule a "
                                              "reading carried by more files)"),
      mpf.iloc[0].to_dict())
# ... and a mass-only species the vote itself chose is simply confirmed on its row
uni_pf, _ = AB.align(_files(3, 112.9857, "C2HF3O2", "[M-H]-", "Assigned", 0.9), tol_ppm=6.0)
gpu = AB.lock_known_species(uni_pf, [dict(_PF, neutral="C2HF3O2", label="TFA", src=f"v{i:02d}", mz=112.9857)
                                     for i in range(3)], tol_ppm=6.0, log=lambda *a: None)
check("lock: a mass-only species the vote chose -> confirmed_kept",
      gpu["confirmed_kept"] == 1 and gpu["mass_only_outvoted"] == 0 and uni_pf.iloc[0]["neutral_formula"] == "C2HF3O2")
# MEMBERSHIP, not the m/z window: a chlorinated paraffin 37Cl-confirmed in ONE
# file whose own m/z sits 7.5 ppm from the mean of a cluster the 12-file grid
# reading dominates -- the vote lists it in `alternatives`, so the lock finds it
_mcp = pd.DataFrame([dict(mz=680.9847, neutral_formula="C25H18N2O16", adduct="[M+Br]-", tier="Candidate",
                          ion_score=0.90, n_files=13, n_files_ion=12, n_files_winner=12,
                          alternatives="C24H40Cl10 [M-H]- x1 Assigned 0.58", tier_reason=pd.NA,
                          srcs=",".join(f"v{i:02d}" for i in range(13)), admitted_by="height", occurrence=0.5)])
gcp = AB.lock_known_species(_mcp, [dict(neutral="C24H40Cl10", adduct="[M-H]-", family="chlorinated_paraffin",
                                        label="chlorinated paraffin C24Cl10", src="v12", mz=680.9898,
                                        verdict="confirmed", why="corroborated by a 37Cl envelope (2 satellites)",
                                        summary="corroborated by a 37Cl envelope (2 satellites)", ion_score=0.58,
                                        tier="Assigned", admitted_by="height", occurrence=0.1,
                                        n_channels=1, n_satellites=2)],
                            tol_ppm=6.0, log=lambda *a: None)
check("lock: a reading the vote lists in `alternatives` is found by membership, 7.5 ppm off the cluster mean",
      gcp["locked"] == 1 and gcp["no_cluster"] == 0 and _mcp.iloc[0]["neutral_formula"] == "C24H40Cl10"
      and _mcp.iloc[0]["alternatives"] == "C25H18N2O16 [M+Br]- x12 Candidate 0.90"
      and str(_mcp.iloc[0]["tier_reason"]).endswith("kept over the 12-file C25H18N2O16 [M+Br]- reading (vote 1 of 13 files)"),
      _mcp.iloc[0].to_dict())
check("lock: a pooled ion with no merged row within tolerance is counted, nothing moves",
      gn["no_cluster"] == 1 and gn["locked"] == 0 and uni.iloc[0]["neutral_formula"] == "C21H21O4P")
check("lock: an empty pool / an empty frame is a no-op",
      AB.lock_known_species(uni, [], tol_ppm=6.0, log=lambda *a: None)["pooled"] == 0
      and AB.lock_known_species(pd.DataFrame(columns=["mz"]), pool, tol_ppm=6.0,
                                log=lambda *a: None)["pooled"] == 0)
# known_evidence reads a per-file ledger: the committed known: rows (with the
# route pass 0 wrote into the commentary) and the known_lead records
_ledk = pd.DataFrame([
    dict(role="M0", peak_id="p1", mz=579.1710, neutral_formula="C14H42O7Si7", adduct="[M+(CH4N2O)H]+",
         method="known:cyclosiloxane", tier="Assigned", ion_score=0.83, admitted_by="height", occurrence=0.9,
         commentary=("Pass 0 (known methylsiloxane): C14H42O7Si7 [M+(CH4N2O)H]+ = "
                     "tetradecamethylcycloheptasiloxane (D7), ppm 0.30, ion score 0.83; volatile "
                     "methylsiloxane (PDMS monomer/oligomer, indoor/lab contaminant); corroborated by "
                     "a confirmed 29Si/30Si envelope (single channel)"), known_lead=pd.NA),
    dict(role="M0", peak_id="p2", mz=429.1574, neutral_formula="C15H24N4O10", adduct="[M+H]+",
         method="cheminfo+grid", tier="Candidate", ion_score=0.98, admitted_by="height", occurrence=0.5,
         commentary="Pass 1",
         known_lead=('{"formula": "C21H21O4P", "family": "organophosphate", "label": "tricresyl phosphate '
                     '(TMPP / TCrP)", "adduct": "[M+(CH4N2O)H]+", "ion_formula": "C22H26N2O5P+", '
                     '"mz": 429.1574, "ppm": 0.4, "ion_score": 0.7, "channels": 1, "verdict": "deferred", '
                     '"twin": null, "why": "single channel; no diagnostic twin to test (monoisotopic)", '
                     '"summary": "no diagnostic twin to test (monoisotopic)"}')),
    dict(role="M0", peak_id="p4", mz=112.9857, neutral_formula="C2HF3O2", adduct="[M-H]-",
         method="known:perfluoroacid", tier="Assigned", ion_score=0.9, admitted_by=pd.NA, occurrence=float("nan"),
         commentary=("Pass 0 (known perfluoroacid): C2HF3O2 [M-H]- = trifluoroacetic acid (TFA), ppm 0.10, "
                     "ion score 0.90; perfluorocarboxylic acid (F off the grid); known PFCA series formula, "
                     "exact-mass committed"), known_lead=pd.NA),
    dict(role="unexplained", peak_id="p3", mz=100.0, neutral_formula=pd.NA, adduct=pd.NA, method=pd.NA,
         tier=pd.NA, ion_score=float("nan"), admitted_by=pd.NA, occurrence=float("nan"), commentary=pd.NA,
         known_lead=pd.NA),
])
_ke = AB.known_evidence(_ledk, src="f1")
check("known_evidence: two confirmed records (route / exact mass), one lead with its verdict; nothing else",
      len(_ke) == 3 and _ke[0]["verdict"] == "confirmed" and _ke[0]["src"] == "f1"
      and _ke[0]["label"] == "tetradecamethylcycloheptasiloxane (D7)"
      and _ke[0]["why"] == "corroborated by a confirmed 29Si/30Si envelope (single channel)"
      and _ke[0]["family"] == "cyclosiloxane" and abs(_ke[0]["ion_score"] - 0.83) < 1e-9
      and _ke[0]["tier"] == "Assigned" and _ke[0]["admitted_by"] == "height"
      and _ke[1]["neutral"] == "C2HF3O2" and _ke[1]["label"] == "trifluoroacetic acid (TFA)"
      and _ke[1]["why"] == "exact mass, on-cal (the family's own rule)"
      and _ke[1]["admitted_by"] is None and _ke[1]["occurrence"] is None
      and _ke[1]["summary"] == "exact mass, on-cal (the family's own rule)"
      and _ke[0]["n_channels"] == 1 and _ke[0]["n_satellites"] == 0
      and _ke[2] == dict(src="f1", neutral="C21H21O4P", adduct="[M+(CH4N2O)H]+", mz=429.1574,
                         family="organophosphate", label="tricresyl phosphate (TMPP / TCrP)",
                         verdict="deferred", why="single channel; no diagnostic twin to test (monoisotopic)",
                         summary="no diagnostic twin to test (monoisotopic)", n_channels=1, n_satellites=0,
                         ion_score=0.7, tier=None, admitted_by=None, occurrence=None), _ke)
check("known_evidence: missing columns -> empty", AB.known_evidence(pd.DataFrame({"x": [1]})) == [])
# the route is read off the ROW: a chlorinated paraffin names no "corroborated by"
# but records the 37Cl lines it locked on; a PFCA's recorded 81Br line is the
# REAGENT's twin (no Br in the neutral) and buys it nothing; an iodine bromide's
# 81Br2 is the neutral's own
_ledr = pd.DataFrame([
    dict(role="M0", peak_id="cp", mz=680.9898, neutral_formula="C24H40Cl10", adduct="[M-H]-",
         method="known:chlorinated_paraffin", tier="Assigned", ion_score=0.58,
         commentary=("Pass 0 (known chlorinated-paraffin): C24H40Cl10 [M-H]- = chlorinated paraffin C24Cl10, "
                     "ppm 2.12, ion score 0.58; chlorinated paraffin (Cl off the grid); 37Cl envelope confirmed "
                     "(2 satellites), isotope-locked"),
         isotopologues='[{"label": "37Cl3", "score": 0.9, "peak_id": "a"}, {"label": "37Cl4", "score": 0.8, "peak_id": "b"}]'),
    dict(role="M0", peak_id="pf", mz=192.9118, neutral_formula="C2HF3O2", adduct="[M+Br]-",
         method="known:perfluoroacid", tier="Assigned", ion_score=0.9,
         commentary=("Pass 0 (known perfluoroacid): C2HF3O2 [M+Br]- = perfluoro-C2 acid (PFCA), ppm 1.5, ion score "
                     "0.90; perfluorocarboxylic acid (F off the grid); known PFCA series formula, exact-mass committed"),
         isotopologues='[{"label": "13C", "score": 0.9, "peak_id": "c"}, {"label": "81Br", "score": 0.95, "peak_id": "d"}]'),
    dict(role="M0", peak_id="ib", mz=286.7401, neutral_formula="IBr", adduct="[M+Br]-",
         method="known:reactive_iodine", tier="Assigned", ion_score=0.7,
         commentary="Pass 0 (known reactive-iodine): IBr [M+Br]- = iodine monobromide, ppm 1.54, ion score 0.70",
         isotopologues='[{"label": "81Br2", "score": 0.9, "peak_id": "e"}]'),
])
_kr = {r["neutral"]: (r["why"], r["n_channels"], r["n_satellites"]) for r in AB.known_evidence(_ledr, src="f")}
check("known_evidence: the paraffin's route is its recorded 37Cl envelope, the PFCA is mass-only, IBr its own 81Br",
      _kr == {"C24H40Cl10": ("corroborated by a confirmed 37Cl envelope (2 satellites)", 1, 2),
              "C2HF3O2": (AB.MASS_ONLY_ROUTE, 1, 0),
              "IBr": ("corroborated by a confirmed 81Br envelope (1 satellite)", 1, 1)}, _kr)
check("_known_route: a second ledger channel first; the commentary phrase ('by' or 'across') when nothing is recorded; else None",
      AB._known_route("x; corroborated across 2 ion channels (monoisotopic P)", "C6H15O4P", None) == "2 ion channels (monoisotopic P)"
      and AB._known_route("no phrase", "C6H15O4P", None, n_channels=2) == "2 ion channels"
      and AB._known_route("no phrase", "C24H40Cl10", None) is None
      and AB._known_route("no phrase", "C24H40Cl10", "not json") is None)
# a satellite recorded in the row's list AND as an iso_child row is one peak
check("_own_satellites: the same peak in the isotopologues list and as a child row counts once; a reagent 81Br is not the neutral's",
      AB._own_satellites("C24H40Cl10", '[{"label": "37Cl", "peak_id": "k1"}]', [("k1", "37Cl"), ("k2", "37Cl2")]) == ["37Cl", "37Cl2"]
      and AB._own_satellites("C2HF3O2", '[{"label": "81Br", "peak_id": "k1"}]', [("k1", "81Br")]) == [])
# what pass 0 really records on the TOF's paraffins: an `M0` two Da below the committed 37Cl line and
# an `M+6` are envelope lines and count; two labels on ONE peak (37Cl2 / 81Br+37Cl, unresolved) are
# one line; a 13C child never counts; the reagent's 81Br on a Cl-only neutral counts only as the
# 37Cl part of a composite label
check("_own_satellites: envelope members M0 / M+6 count, two labels on one peak count once, 13C does not",
      AB._own_satellites("C12H21Cl5", '[{"label": "M0", "peak_id": "a"}, {"label": "81Br+37Cl", "peak_id": "b"}]',
                         [("c", "M+6"), ("b", "81Br+37Cl"), ("d", "13C")]) == ["M0", "81Br+37Cl", "M+6"]
      and AB._own_satellites("C15H26Cl6", '[{"label": "37Cl2", "peak_id": "j"}, {"label": "81Br+37Cl", "peak_id": "j"}]',
                             [("j", "81Br+37Cl")]) == ["37Cl2"]
      and AB._own_satellites("C2HF3O2", '[{"label": "M0", "peak_id": "a"}, {"label": "13C", "peak_id": "b"}]', []) == ["M0"])
# ... and the pool of that one file locks nothing the file did not already read:
# the D7 row is confirmed where it stands, the TCP lead is noted on the grid row
_mk2 = pd.DataFrame([dict(mz=579.1710, neutral_formula="C14H42O7Si7", adduct="[M+(CH4N2O)H]+", tier="Assigned",
                          ion_score=0.83, n_files=1, n_files_ion=1, n_files_winner=1, alternatives="",
                          tier_reason=pd.NA, srcs="f1"),
                     dict(mz=429.1574, neutral_formula="C15H24N4O10", adduct="[M+H]+", tier="Candidate",
                          ion_score=0.98, n_files=1, n_files_ion=1, n_files_winner=1, alternatives="",
                          tier_reason=pd.NA, srcs="f1")])
_g2 = AB.lock_known_species(_mk2, _ke, tol_ppm=6.0, log=lambda *a: None)
check("lock over known_evidence: confirmed_kept 1, lead_only 1, no_cluster 1 (TFA has no row here)",
      _g2 == {"pooled": 3, "locked": 0, "confirmed_kept": 1, "conflict": 0, "lead_only": 1,
              "mass_only_outvoted": 0, "no_cluster": 1}
      and str(_mk2.iloc[1]["tier_reason"]).startswith("known-species lead: tricresyl phosphate"), _g2)

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

# --- THE VOTE READS THE EVIDENCE: the vote class ranks ions before the count
# With one TOF ion's readings finally in one row (the merge window sized from
# the batch's own scatter), the count alone handed the peak to bromide adducts
# of N-compounds read in more files over the reading the other instrument
# confirms. Each ion now takes the best per-file VOTE CLASS of its readings
# (the parent's `vote_class` column: neutral backed 2 > formula confirmed 1 >
# unconfirmed 0) and the count decides among equals; the label stage and the
# ion-only-last rule are unchanged. The four cases are the C2 board's own
# losses. The frames below state each reading's private decision (level, axes)
# and carry the class `_evidence_class` maps it to -- exactly what the parent
# computes per file (evidence.vote_classes) -- and never the level itself.
def _with_class(df):
    df = df.copy()
    df["vote_class"] = [AB._evidence_class(a, b) for a, b in zip(df["evidence_level"], df["evidence_axes"])]
    return df.drop(columns=["evidence_level", "evidence_axes"])


def m0e(rows):
    return _with_class(pd.DataFrame(rows, columns=["mz", "neutral_formula", "adduct", "tier", "ion_score",
                                                   "evidence_level", "evidence_axes"]))


def _filese(n, mz, nf, ad, tier, ion, level, axes, start=0):
    return {f"e{start + i:02d}": m0e([(mz + 1e-4 * i, nf, ad, tier, ion, level, axes)]) for i in range(n)}


check("vote class: the private decision's neutral established (2b/3a/3b/4a) or corroborated is 2, formula/ion confirmed 1, else 0",
      [AB._evidence_class(lv, "") for lv in ("2b", "3a", "3b", "4a", "4b", "4c", "4d", "5a", "5b", None)]
      == [2, 2, 2, 2, 1, 1, 1, 0, 0, 0]
      and AB._evidence_class("5b", "iso|corroborated|carbon") == 2 and AB._evidence_class("4b", "corroborated") == 2
      and AB._evidence_class("5b", "iso|uncorroborated") == 0)
# case 1: pinic acid, a 1-vs-1 tie the count could not decide -- the old vote
# fell through to ion_score and picked a silicon formula over the reading the
# other instrument corroborates
pin = {"a": m0e([(248.0762, "C12H15NO3Si", "[M-H]-", "Candidate", 0.947, "5b", "")]),
       "b": m0e([(248.0762, "C9H14O4", "[M+NO3]-", "Candidate", 0.917, "4b", "corroborated")])}
mp, jp = AB.align(pin, tol_ppm=12.0)
check("evidence: pinic acid C9H14O4 [M+NO3]- (4b corroborated, 1 file) beats the 1-file silicon formula (5b) that out-scored it",
      len(mp) == 1 and mp.iloc[0]["neutral_formula"] == "C9H14O4" and mp.iloc[0]["adduct"] == "[M+NO3]-",
      mp.to_dict("records"))
check("evidence: the row says the evidence decided, not the count",
      mp.iloc[0]["tier_reason"] == "evidence outranks the count: kept C9H14O4 [M+NO3]- (neutral backed in 1 of 2 "
                                   "files) over the 1-file C12H15NO3Si [M-H]- (unconfirmed)", mp.iloc[0]["tier_reason"])
check("evidence: the loser is listed as before", mp.iloc[0]["alternatives"] == "C12H15NO3Si [M-H]- x1 Candidate 0.95",
      repr(mp.iloc[0]["alternatives"]))
check("evidence: jitter.csv carries each file's own vote class (an int), never a level",
      "vote_class" in jp.columns and "evidence_level" not in jp.columns
      and sorted(int(x) for x in jp["vote_class"]) == [0, 2], jp.to_dict("records"))
pin_br = {"a": m0e([(265.0089, "C2H9N3O6S", "[M+NO3]-", "Candidate", 0.944, "5b", "")]),
          "b": m0e([(265.0089, "C9H14O4", "[M+Br]-", "Candidate", 0.909, "4b", "corroborated")])}
mpb, _ = AB.align(pin_br, tol_ppm=12.0)
check("evidence: pinic acid's [M+Br]- reading wins its row the same way",
      mpb.iloc[0]["neutral_formula"] == "C9H14O4" and mpb.iloc[0]["adduct"] == "[M+Br]-", mpb.to_dict("records"))
# case 2: C9H16O6 [M+NO3]- Assigned at 4a in 2 files against C14H21N [M+Br]- in 9
hom6 = {**_filese(9, 282.0853, "C14H21N", "[M+Br]-", "Candidate", 0.93, "5a", ""),
        **_filese(2, 282.0831, "C9H16O6", "[M+NO3]-", "Assigned", 0.95, "4a", "iso|chan2|corroborated|branch", start=9)}
m6, _ = AB.align(hom6, tol_ppm=12.0)
_h6 = m6.iloc[0]
check("evidence: C9H16O6 [M+NO3]- (4a, 2 files) beats C14H21N [M+Br]- (5a, 9 files) whatever the count",
      len(m6) == 1 and _h6["neutral_formula"] == "C9H16O6" and _h6["tier"] == "Assigned"
      and _h6["n_files"] == 11 and _h6["n_files_ion"] == 2 and _h6["n_files_winner"] == 2, _h6.to_dict())
check("evidence: the 9-file loser heads the alternatives and the note names it",
      _h6["alternatives"] == "C14H21N [M+Br]- x9 Candidate 0.93"
      and _h6["tier_reason"] == "evidence outranks the count: kept C9H16O6 [M+NO3]- (neutral backed in 2 of 11 "
                                "files) over the 9-file C14H21N [M+Br]- (unconfirmed)", _h6.to_dict())
# case 3: C10H16O9 [M+NO3]- read below assignability (5b) in 2 files, but the
# other instrument holds the neutral: the corroborated axis lifts it over the
# 3-file mass-only C15H21NO3 [M+Br]-
hom9 = {**_filese(3, 342.0698, "C15H21NO3", "[M+Br]-", "Candidate", 0.90, "5b", ""),
        **_filese(2, 342.0698, "C10H16O9", "[M+NO3]-", "Candidate", 0.85, "5b", "iso|corroborated|carbon", start=3)}
m9, _ = AB.align(hom9, tol_ppm=12.0)
check("evidence: a corroborated 5b reading (C10H16O9, 2 files) beats an uncorroborated 5b read in 3 files",
      m9.iloc[0]["neutral_formula"] == "C10H16O9" and m9.iloc[0]["n_files_ion"] == 2
      and str(m9.iloc[0]["tier_reason"]).startswith("evidence outranks the count: kept C10H16O9 [M+NO3]- (neutral backed in 2 of 5 files) over the 3-file C15H21NO3 [M+Br]- (unconfirmed)"),
      m9.iloc[0].to_dict())
# case 4: C10H18O9 [M+NO3]- in ONE file (4b corroborated) against two mass-only
# ions read in 4 and 3 files; both losers stay listed, biggest first
hom18 = {**_filese(4, 344.0857, "C15H23NO3", "[M+Br]-", "Candidate", 0.92, "5b", ""),
         **_filese(3, 344.0857, "C11H9NO9", "[M+NO3]-", "Candidate", 0.91, "5b", "", start=4),
         **_filese(1, 344.0840, "C10H18O9", "[M+NO3]-", "Candidate", 0.88, "4b", "corroborated", start=7)}
m18, _ = AB.align(hom18, tol_ppm=12.0)
check("evidence: C10H18O9 [M+NO3]- (1 file, 4b corroborated) beats the 4-file and the 3-file mass-only ions",
      m18.iloc[0]["neutral_formula"] == "C10H18O9" and m18.iloc[0]["n_files"] == 8 and m18.iloc[0]["n_files_ion"] == 1
      and m18.iloc[0]["alternatives"] == "C15H23NO3 [M+Br]- x4 Candidate 0.92; C11H9NO9 [M+NO3]- x3 Candidate 0.91",
      m18.iloc[0].to_dict())
check("evidence: the note names the biggest of the lower-class ions",
      m18.iloc[0]["tier_reason"] == "evidence outranks the count: kept C10H18O9 [M+NO3]- (neutral backed in 1 of 8 "
                                    "files) over the 4-file C15H23NO3 [M+Br]- (unconfirmed)", m18.iloc[0]["tier_reason"])
# the count decides among EQUALS: two class-2 ions, 3 files at 4a vs 2 files at 3b
eq = {**_filese(3, 300.0, "C10H12O2", "[M+H]+", "Candidate", 0.90, "4a", "iso|chan2"),
      **_filese(2, 300.0, "C9H12N2O", "[M+H]+", "Candidate", 0.99, "3b", "branch", start=3)}
meq, _ = AB.align(eq, tol_ppm=6.0)
check("evidence: among ions of one class the count decides (3 files at 4a beat 2 files at 3b), no note",
      meq.iloc[0]["neutral_formula"] == "C10H12O2" and meq.iloc[0]["n_files_ion"] == 3 and pd.isna(meq.iloc[0]["tier_reason"]),
      meq.iloc[0].to_dict())
eq0 = {**_filese(4, 300.0, "C10H12O2", "[M+H]+", "Candidate", 0.90, "5b", ""),
       **_filese(1, 300.0, "C9H12N2O", "[M+H]+", "Assigned", 0.99, "5a", "", start=4)}
meq0, _ = AB.align(eq0, tol_ppm=6.0)
check("evidence: two mass-only ions are equals, the count decides as before (4 Candidate files beat 1 Assigned)",
      meq0.iloc[0]["neutral_formula"] == "C10H12O2" and pd.isna(meq0.iloc[0]["tier_reason"]), meq0.iloc[0].to_dict())
# the middle class: the formula pinned (4b) in one file beats exact mass alone in nine
mid = {**_filese(9, 304.9071, "C16H2S", "[M+Br]-", "Candidate", 0.95, "5b", ""),
       **_filese(1, 304.9071, "C10HF3O3", "[M+Br]-", "Assigned", 0.90, "4b", "iso", start=9)}
mmid, _ = AB.align(mid, tol_ppm=12.0)
check("evidence: a 4b reading in 1 file beats a 5b reading in 9 (the middle class outranks mass-only)",
      mmid.iloc[0]["neutral_formula"] == "C10HF3O3"
      and mmid.iloc[0]["tier_reason"] == "evidence outranks the count: kept C10HF3O3 [M+Br]- (formula confirmed in 1 "
                                         "of 10 files) over the 9-file C16H2S [M+Br]- (unconfirmed)", mmid.iloc[0].to_dict())
# an Orbitrap-like cluster: the many-file reading is also the best-evidenced one -- nothing moves, no note
noop = {**_filese(8, 217.12, "C10H16O2", "[M+H]+", "Assigned", 0.91, "3b", "chan2|branch"),
        **_filese(2, 217.12, "C9H16N2O", "[M+H]+", "Candidate", 0.99, "5b", "", start=8)}
mno, _ = AB.align(noop, tol_ppm=6.0)
check("evidence: the many-file best-class reading wins as before, no note",
      mno.iloc[0]["neutral_formula"] == "C10H16O2" and mno.iloc[0]["n_files_ion"] == 8 and pd.isna(mno.iloc[0]["tier_reason"]),
      mno.iloc[0].to_dict())
# frames WITHOUT the evidence columns: every reading is class 0 and the vote is the count it was
_nocol, _ = AB.align(vote, tol_ppm=6.0)
_nacol, _ = AB.align({k: v.assign(vote_class=pd.NA) for k, v in vote.items()}, tol_ppm=6.0)
check("evidence: frames without the class (or with an NA class) vote by the count exactly as before",
      _nocol["neutral_formula"].iloc[0] == "C10H12O2" and _nocol.drop(columns=[]).equals(_nacol), (_nocol.to_dict("records"), _nacol.to_dict("records")))
# the ion-only-last rule is unchanged: an ion-only reading levelled 4d in 3 files
# still yields to a regular mass-only reading in 1
io = {**{f"i{i}": _with_class(pd.DataFrame([(100.0, "C4H6O2", "[M]-.", "Candidate", 0.8, "4d", "iso|ion_only", "p1")],
                                columns=["mz", "neutral_formula", "adduct", "tier", "ion_score", "evidence_level", "evidence_axes", "ion_only_of"]))
         for i in range(3)},
      "r": _with_class(pd.DataFrame([(100.0, "C3H2O3", "[M-H]-", "Candidate", 0.7, "5b", "", pd.NA)],
                        columns=["mz", "neutral_formula", "adduct", "tier", "ion_score", "evidence_level", "evidence_axes", "ion_only_of"]))}
mio, _ = AB.align(io, tol_ppm=6.0)
check("evidence: an ion-only reading (4d, 3 files) still ranks below a regular reading (5b, 1 file)",
      mio.iloc[0]["neutral_formula"] == "C3H2O3" and str(mio.iloc[0]["tier_reason"]).startswith("regular reading kept over the 3-file"),
      mio.iloc[0].to_dict())
# the label stage is unchanged: the same ion read two ways -- the label Assigned in
# a file wins over the label Candidate in more files even when the latter carries the better level
lbl = {**_filese(3, 252.1230, "C13H17NO4", "[M+H]+", "Candidate", 0.98, "3b", "branch"),
       **_filese(1, 252.1230, "C13H14O4", "[M+NH4]+", "Assigned", 0.97, "5b", "", start=3)}
mlb, _ = AB.align(lbl, tol_ppm=6.0)
check("evidence: within one ion the label stage still goes by corroboration (Assigned in 1 beats Candidate in 3)",
      mlb.iloc[0]["neutral_formula"] == "C13H14O4" and bool(mlb.iloc[0]["ion_agree"])
      and str(mlb.iloc[0]["tier_reason"]).startswith("same ion C13H18NO4+ read two ways"), mlb.iloc[0].to_dict())
# the notes name a class, never a level of any scale (the pre-0.10.0 decision is private to the vote)
import re as _re
_notes = [str(x) for _m in (mp, mpb, m6, m9, m18, mmid) for x in _m["tier_reason"].dropna()]
check("evidence: every vote note names the class text, no level token",
      _notes and all(("neutral backed" in n or "formula confirmed" in n) for n in _notes)
      and not any(_re.search(r"\(\s*[1-5][a-d]?\b|\b[1-5][a-d]\b", n) for n in _notes), _notes)
# determinism with the new key
for _name, _d in (("HOM C10H18O9", hom18), ("pinic acid", pin)):
    _m1, _j1 = AB.align(_d, tol_ppm=12.0)
    _m2, _j2 = AB.align(dict(reversed(list(_d.items()))), tol_ppm=12.0)
    check(f"evidence ({_name}): reversed file order -> identical merged frame", _m1.equals(_m2),
          (_m1.to_dict("records"), _m2.to_dict("records")))

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


#: what every fake file's degeneracy stage persists, as a real run's does (stats["degeneracy_cal"])
_CAL = {"mu": 0.0, "sigma": 0.3}


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
    return {"ledger": led, "stats": {"noise_edge_cps": 4.0, "height_gate_cps": 10.0,
                                     "degeneracy_cal": dict(_CAL)},
            "plausibility_audit": [], "summaries": {}, "problems": []}


def _pooled_levelled(d, summ):
    """(ok, detail): the pooled level stage really ran on the batch in ``d`` -- an Orbitrap-class width model
    (declared: the fakes give no measured one), the per-file calibrations in the summary, so a run window,
    and every pooled pair levelled on the scale (no NA, none at the no-run-window text)."""
    from peaky.assignment import evidence as _EV
    ev = pd.read_csv(os.path.join(d, "tables", "evidence_levels.csv"), keep_default_na=False)
    el = summ["evidence_levels"]
    cals = [pf.get("degeneracy_cal") for pf in summ["per_file"]]
    ok = (el["instrument"]["class"] == "orbitrap" and all(c == _CAL for c in cals)
          and el["n_pairs"] == len(ev) > 0 and set(ev["evidence_level"]) <= set(_EV.LEVELS) | {"reagent"}
          and _EV.NO_RUN_WINDOW_TEXT not in set(ev["evidence"]) and sum(el["pooled"].values()) == len(ev))
    return ok, (el.get("instrument"), el.get("pooled"), cals)


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
                     resolving_power=100_000, log=lambda *a: None)   # Orbitrap class: the pooled levels run
        summ = json.load(open(os.path.join(_d, "batch_summary.json")))
        s = summ["selection"]
        check("run: the pooled level stage ran (Orbitrap class, persisted calibrations, every pair levelled)",
              *_pooled_levelled(_d, summ))
        check("run: a calibrated batch has no levels-not-assessed reason",
              "levels_not_assessed_reason" in summ and summ["levels_not_assessed_reason"] is None,
              summ.get("levels_not_assessed_reason"))
        check("run: batch_summary names the scorer that judged the candidates",
              summ.get("scorer") == ("local" if IO._local_scoring_enabled() else "server"), summ.get("scorer"))
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
        # C46: the batch's typical detection edge -- the median of the files' own 1st-percentile heights
        # (the stub serves one 1e5-cps peak per file) -- recorded in the summary and carried on every
        # per-file cfg the assigner was handed
        check("run (C46): batch_summary records the batch detection edge and every per-file cfg carried it",
              summ.get("noise_edge_batch_cps") == 1.0e5
              and all(getattr(c, "noise_edge_batch_cps", None) == 1.0e5 for c in _SEEN_CFG if c is not None)
              and len(_SEEN_CFG) > 0, (summ.get("noise_edge_batch_cps"), len(_SEEN_CFG)))
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
        check("run: batch_summary records the merged-level gates (the known-species decision, "
              "the reagent-water ladder, the isotopologue rows, the element-signature removal and the "
              "TOF ion-M+2 gates always -- nothing pooled, no rung or signature veto here, the "
              "isotopologue gate skipped on a height-only series, the TOF gates skipped on an "
              "Orbitrap-class batch; no polarity gate in negative mode)",
              set(summ.get("merge_gates", {})) == {"known", "reagent_water", "isotopologue",
                                                    "element_signature", "tof_m2"}
              and summ["merge_gates"]["isotopologue"]["ran"] is False
              and "no peak areas" in summ["merge_gates"]["isotopologue"]["skipped"]
              and summ["merge_gates"]["element_signature"] == {"removed": 0, "pairs": []}
              and summ["merge_gates"]["known"] == {"pooled": 0, "locked": 0, "confirmed_kept": 0,
                                                   "conflict": 0, "lead_only": 0, "mass_only_outvoted": 0,
                                                   "no_cluster": 0}
              and summ["merge_gates"]["reagent_water"]["n_rungs"] == 0
              and summ["merge_gates"]["reagent_water"]["n_stripped"] == 0
              and summ["merge_gates"]["tof_m2"]["ran"] is False
              and summ["merge_gates"]["tof_m2"]["skipped"] == "not a TOF-class batch",
              summ.get("merge_gates"))
    # ... and on a TOF-class batch (a width model under the Orbitrap class's R) the TOF gates run
    with tempfile.TemporaryDirectory() as _d7:
        AB.run(peaks=_PK, ts_peaks=_PK, reagent="Br", batch="test batch", out_dir=_d7, k_min=2, k_max=3,
               min_gain=0.0, n_jobs=1, resolving_power=10_000, log=lambda *a: None)
        summ7 = json.load(open(os.path.join(_d7, "batch_summary.json")))
        check("run: a TOF-class batch runs the TOF ion-M+2 gates after the stamp (nothing to demote here)",
              summ7["merge_gates"]["tof_m2"] == {"ran": True, "req_demoted": 0, "known_demoted": 0,
                                                 "doublet_demoted": 0, "doublet_exempt": 0},
              summ7["merge_gates"].get("tof_m2"))
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

    # a sparse peak table: no file calibrates the degeneracy window, so the scale
    # assesses nothing -- and the run says so once, in the summary and the console
    def _uncalibrated_assign(sid, context="ambient-air", **kw):
        _out = _fake_assign(sid, context, **kw)
        _out["stats"]["degeneracy_cal"] = None
        return _out

    _A.run = _uncalibrated_assign
    _log6: list = []
    try:
        with tempfile.TemporaryDirectory() as _d6:
            AB.run(peaks=_PK, ts_peaks=_PK, reagent="Br", batch="test batch", out_dir=_d6,
                   k_min=2, k_max=3, min_gain=0.0, n_jobs=1, resolving_power=100_000,
                   log=lambda *a: _log6.append(" ".join(map(str, a))))
            summ6 = json.load(open(os.path.join(_d6, "batch_summary.json")))
        _why6 = summ6.get("levels_not_assessed_reason") or ""
        check("run: no file calibrated -> batch_summary says the levels were not assessed, and why",
              _why6.startswith("Evidence levels were not assessed") and "isotope-backed core rows" in _why6,
              _why6)
        check("run: ... and the console carries it once as a WARNING",
              sum("WARNING" in _l and "not assessed" in _l for _l in _log6) == 1,
              [_l for _l in _log6 if "WARNING" in _l])
    finally:
        _A.run = _fake_assign

    # a batch addressed by ID: the roster is fetched by that id (io_mascope.
    # resolve_batch settles the batch against the server's listing first) and
    # batch_summary records the batch's DISPLAY name, so a run addressed by id
    # still reads as its batch. Listing + roster surfaces are faked; no network.
    class _IdBatches:
        def list(self, dataset=None):   # noqa: A001
            return pd.DataFrame({"sample_batch_id": ["BID"],
                                 "sample_batch_name": ["Texas Ur 122-600"]})

    class _IdSamples:
        seen: list = []

        def list(self, *, batch, dataset=None, drop_columns=None):   # noqa: A001
            _IdSamples.seen.append(batch)
            return SS.sample_table(_PK)          # the roster: one row per sample

    class _IdClient:
        batches, samples = _IdBatches(), _IdSamples()

    IO.connect = lambda *a, **k: _IdClient()
    try:
        with tempfile.TemporaryDirectory() as _d5:
            AB.run(batch="BID", dataset="DS", ts_peaks=_PK, reagent="Br", out_dir=_d5,
                   k_min=2, k_max=3, min_gain=0.0, n_jobs=1, log=lambda *a: None)
            summ5 = json.load(open(os.path.join(_d5, "batch_summary.json")))
        check("run: a batch addressed by id fetches its roster by that id",
              _IdSamples.seen == ["BID"], _IdSamples.seen)
        check("run: ... and batch_summary records the batch's DISPLAY name, not the id",
              summ5["batch_name"] == "Texas Ur 122-600", summ5.get("batch_name"))
    finally:
        IO.connect = lambda *a, **k: "CLIENT"

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
    in two files and not the third (the C15H22 case)."""
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
    return {"ledger": led, "stats": {"noise_edge_cps": 4.0, "height_gate_cps": 10.0,
                                     "degeneracy_cal": dict(_CAL)},
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
                      resolving_power=100_000, log=lambda *a: None)   # Orbitrap class: the pooled levels run
        mg = resp["merged"]
        summp = json.load(open(os.path.join(_dp, "batch_summary.json")))
        r_nh4 = mg.iloc[(mg["mz"] - _HC_MZ).abs().argmin()]
        r_ur = mg.iloc[(mg["mz"] - _UR_MZ).abs().argmin()]
        check("positive run: the pooled level stage ran (Orbitrap class, persisted calibrations, every pair levelled)",
              *_pooled_levelled(_dp, summp))
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
    return {"ledger": led, "stats": {"noise_edge_cps": 500.0, "height_gate_cps": 500.0,
                                     "degeneracy_cal": dict(_CAL)},
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


def _run_staged(d, resolving_power=100_000, **kw):
    """One staged run; an Orbitrap-class width model by default, so the pooled levels run over both stages
    (the fakes give no measured one: class-less, every pair would read NA at the class gate)."""
    lines = []
    res = AB.run(peaks=_RPK, ts_peaks=_RPK, reagent="Br", batch="test batch", out_dir=d,
                 k_min=2, k_max=2, min_gain=0.0, log=lines.append, resolving_power=resolving_power, **kw)
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
        _ok, _det = _pooled_levelled(_d, summ)
        check("residual run: the pooled level stage ran over both stages (every pair levelled, per stage too)",
              _ok and set(summ["evidence_levels"]["per_stage"]) == {"cover", "residual"},
              (_det, summ["evidence_levels"].get("per_stage")))
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


# ---- the batch's width model: measured ONCE, handed to every per-file run --------------
from peaky.batch import tracefirst as _TFT  # noqa: E402
from peaky.chem import resolution as _RES  # noqa: E402

_saved_meas = _TFT.measure_resolution
_MEAS_CALLS = []


def _fake_measure(client, sid, peaks=None, log=print, **kw):
    _MEAS_CALLS.append(sid)
    return _RES.Resolution(coef=1.0 / 9500.0, exponent=1.0, n_peaks=9, source="measured")


IO.connect = lambda *a, **k: "CLIENT"
IO.fetch_peaks = lambda client, sid, use_cache=True: pd.DataFrame(
    {"peak_id": ["p1"], "mz": [_C.ion_mz(_F, "[M-H]-")], "height": [1.0e5]})
IO.estimate_offset = lambda raw: 0.0
_A.run = _fake_assign
_TFT.measure_resolution = _fake_measure
try:
    with tempfile.TemporaryDirectory() as _d:
        _SEEN_KW.clear(); _MEAS_CALLS.clear(); _lines = []
        AB.run(peaks=_PK, ts_peaks=_PK, reagent="Br", batch="test batch", out_dir=_d,
               k_min=2, k_max=3, min_gain=0.0, n_jobs=1, log=_lines.append)
        summ = json.load(open(os.path.join(_d, "batch_summary.json")))
        _rps = [kw.get("resolving_power") for kw in _SEEN_KW]
        check("width model: measured ONCE on a middling spectrum and handed to every per-file run",
              len(_MEAS_CALLS) == 1 and len(_rps) == 3
              and all(isinstance(r, _RES.Resolution) and r.r_at(200.0) == 9500.0 for r in _rps),
              (_MEAS_CALLS, _rps))
        check("width model: the middling spectrum by peak count is the probe",
              _MEAS_CALLS[0] in _PK["sample_item_id"].unique(), _MEAS_CALLS)
        check("width model: batch_summary records the model and the (empty here) class counts",
              summ["resolution"]["source"] == "measured" and summ["resolution"]["r_at_200"] == 9500.0
              and summ["resolvability"] == {}, (summ.get("resolution"), summ.get("resolvability")))
    with tempfile.TemporaryDirectory() as _d:
        _SEEN_KW.clear(); _MEAS_CALLS.clear()
        AB.run(peaks=_PK, ts_peaks=_PK, reagent="Br", batch="test batch", out_dir=_d,
               k_min=2, k_max=3, min_gain=0.0, n_jobs=1, resolving_power=6500, log=lambda *a: None)
        summ = json.load(open(os.path.join(_d, "batch_summary.json")))
        check("width model: a declared R is coerced, nothing is measured",
              not _MEAS_CALLS and all(kw["resolving_power"].r_at(200.0) == 6500.0 for kw in _SEEN_KW)
              and summ["resolution"]["source"] == "declared", (_MEAS_CALLS, summ.get("resolution")))
    with tempfile.TemporaryDirectory() as _d:
        _SEEN_KW.clear(); _MEAS_CALLS.clear()
        AB.run(peaks=_PK, ts_peaks=_PK, reagent="Br", batch="test batch", out_dir=_d,
               k_min=2, k_max=3, min_gain=0.0, n_jobs=1, resolving_power="none", log=lambda *a: None)
        summ = json.load(open(os.path.join(_d, "batch_summary.json")))
        check("width model: 'none' declines the stamp -- no measurement, None to every file, null in the summary",
              not _MEAS_CALLS and all(kw["resolving_power"] is None for kw in _SEEN_KW)
              and summ["resolution"] is None, (_MEAS_CALLS, summ.get("resolution")))
    _TFT.measure_resolution = lambda *a, **k: None
    with tempfile.TemporaryDirectory() as _d:
        _SEEN_KW.clear(); _lines = []
        AB.run(peaks=_PK, ts_peaks=_PK, reagent="Br", batch="test batch", out_dir=_d,
               k_min=2, k_max=3, min_gain=0.0, n_jobs=1, log=_lines.append)
        summ = json.load(open(os.path.join(_d, "batch_summary.json")))
        check("width model: a measurement that cannot be made is a log line on the cover path, never a failed run",
              all(kw["resolving_power"] is None for kw in _SEEN_KW) and summ["resolution"] is None
              and summ["n_files"] == 3, (summ.get("resolution"), summ.get("n_files")))
    check("_sum_counts adds class counts across files and skips None",
          AB._sum_counts([{"blended": 2, "isolated": 1}, None, {"blended": 3}]) == {"blended": 5, "isolated": 1}
          and AB._sum_counts([]) == {})
finally:
    _TFT.measure_resolution = _saved_meas
    IO.connect, IO.fetch_peaks = _saved["connect"], _saved["fetch_peaks"]
    IO.estimate_offset, _A.run = _saved["estimate_offset"], _saved["run"]


# --- FLATNESS SURVIVES THE MERGE AS A LABEL: the time series labels a row
# (ts_disposition / ts_cv_norm) and never tiers it, so the merged row carries the
# label of the row that donates its tier -- otherwise a batch reader would see a
# flat calibrant background as a plain Assigned analyte.
def m0t(rows):
    return pd.DataFrame(rows, columns=["mz", "neutral_formula", "adduct", "tier", "ion_score",
                                       "ts_disposition", "ts_cv_norm"])


flat_lab = {"a": m0t([(178.0777, "C14H10", "[M]+.", "Assigned", 0.95, "background:flat (TS-flat)", 0.07)]),
            "b": m0t([(178.0778, "C14H10", "[M]+.", "Candidate", 0.97, "ambient:variable", 0.61)])}
mfl, _ = AB.align(flat_lab, tol_ppm=6.0)
check("the merged row carries the donor row's ts_disposition (the Assigned file's label, not the other file's)",
      len(mfl) == 1 and mfl.iloc[0]["tier"] == "Assigned"
      and mfl.iloc[0]["ts_disposition"] == "background:flat (TS-flat)", mfl.to_dict("records"))
check("  -> and its ts_cv_norm", float(mfl.iloc[0]["ts_cv_norm"]) == 0.07, mfl.to_dict("records"))
nolab, _ = AB.align({"a": m0([(500.0, "C20H30O8", "[M+H]+", "Assigned", 0.9)])}, tol_ppm=6.0)
check("a batch without a time series merges as before (label NA, no crash)",
      len(nolab) == 1 and pd.isna(nolab.iloc[0]["ts_disposition"]) and pd.isna(nolab.iloc[0]["ts_cv_norm"]),
      nolab.to_dict("records"))


def test_all():
    assert FAIL == 0, f"{FAIL} checks failed"


if __name__ == "__main__":
    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)
