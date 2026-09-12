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

# --- empty input ------------------------------------------------------------
me, je = AB.align({})
check("empty -> empty merged + jitter with schema",
      len(me) == 0 and "n_files" in me.columns and "cluster" in je.columns)

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


_SEEN_CFG = []


def _fake_assign(sid, context="ambient-air", **kw):
    """Stand-in for assign.run: one assigned M0, the real ledger schema. Keeps
    the cfg it was handed, so the gate resolution can be read back."""
    _SEEN_CFG.append(kw.get("cfg"))
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
        check("run: a profile with no opinion leaves the package default gate",
              all(c is not None and c.height_cutoff_x_edge == 1.0 for c in _SEEN_CFG)
              and summ.get("height_cutoff_x_edge") == 1.0
              and summ.get("height_cutoff_x_edge_source") == "the package default", summ.get(
                  "height_cutoff_x_edge_source"))

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


def test_all():
    assert FAIL == 0, f"{FAIL} checks failed"


if __name__ == "__main__":
    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)
