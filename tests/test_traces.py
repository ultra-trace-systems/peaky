"""Offline tests for batch/traces.py -- the ONE trace primitive behind admission
(per-peak occurrence), the stamp (re-centring / collapse / window) and the
batch-derived brightness floor. Synthetic batches reproduce the mechanisms
measured on real data: a TOF-like ion whose ledger anchor is snapped to theory
while its trace lives ~6 ppm away, a phantom competitor on the same trace, an
Orbitrap-like batch where nothing should move, and a picker that picks into the
noise vs one that stops at the edge.
Run: python3 tests/test_traces.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from peaky.batch import traces as TR          # noqa: E402
from peaky.batch import timeseries as TS      # noqa: E402
from peaky.assignment import admission as ADM  # noqa: E402

PASS = FAIL = 0
def check(name, cond, detail=""):
    global PASS, FAIL
    if cond: PASS += 1; print(f"  ok  {name}")
    else: FAIL += 1; print(f"FAIL  {name}  {detail}")


# ---------------------------------------------------------------------------
# a TOF-like batch: 200 spectra. Ion A: theory 342.0678, but its TRACE sits +6.5
# ppm above theory with a 4 ppm (1 sigma) per-spectrum scatter, present in 90 %
# of spectra at 2-3 cps. Ion B: a clean ion at 250.1 (1 ppm scatter, 95 %,
# 50 cps). 300 transient noise peaks (1-2 spectra each, 1-3 cps). Every spectrum
# also holds 150 noise-floor peaks at random m/z with a SOFT lower tail
# (lognormal: a TOF picker keeps picking into the noise, so a thin tail of peaks
# sits well below the 1st-percentile edge -- 31 in 10 000 below 0.75x on a real
# TOF), and 300 recurrent weak ions (80 % of spectra, brighter than the noise on
# average, 3 ppm scatter) -- so, as on the real batch, most peaks are persistent
# and the transient share FALLS as the floor rises.
# ---------------------------------------------------------------------------
rng = np.random.default_rng(3)
N = 200
A_THEORY = 342.0678
A_TRACE = A_THEORY * (1 + 6.5e-6)
B = 250.1000
WEAK = np.linspace(105.03, 495.71, 300)                   # recurrent weak ions
rows = []
for s in range(N):
    sid = f"s{s:03d}"
    if rng.random() < 0.90:
        rows.append((sid, A_TRACE * (1 + rng.normal(0, 4e-6)), 2.0 + rng.random()))
    if rng.random() < 0.95:
        rows.append((sid, B * (1 + rng.normal(0, 1e-6)), 50.0 + rng.random()))
    for _ in range(150):                                  # the SOFT noise floor
        rows.append((sid, rng.uniform(100, 500), float(np.exp(rng.normal(0.0, 0.6)))))
    for m in WEAK[rng.random(len(WEAK)) < 0.8]:            # persistent, brighter on average
        rows.append((sid, m * (1 + rng.normal(0, 3e-6)), float(2.0 * np.exp(rng.normal(0.0, 0.7)))))
for mz in rng.uniform(100, 500, 300):
    for s in rng.choice(N, size=rng.integers(1, 3), replace=False):
        rows.append((f"s{s:03d}", mz * (1 + rng.normal(0, 2e-6)), 1.0 + 2 * rng.random()))
ts = pd.DataFrame(rows, columns=["sample_item_id", "mz", "height"])
# a duplicate (sample, mz) pair -- Mascope's match-expanded row -- must count once
ts = pd.concat([ts, ts.iloc[[0]]], ignore_index=True)

idx = TR.PeakIndex(ts, tol_ppm=6.0)
check("PeakIndex drops the exact duplicate (sample, mz) row", len(idx) == len(ts) - 1, (len(idx), len(ts)))
check("PeakIndex is sorted by m/z with one code per sample",
      np.all(np.diff(idx.mz) >= 0) and idx.n_samples == N and idx.sample.max() == N - 1)
shuf = TR.PeakIndex(ts.sample(frac=1.0, random_state=9), tol_ppm=6.0)
check("PeakIndex is deterministic under input row order (same codes, same m/z)",
      np.array_equal(shuf.mz, idx.mz) and np.array_equal(shuf.sample, idx.sample))

# --- occurrence: the sweep IS the per-probe rule ---------------------------------
occ = idx.occurrence()
k = rng.choice(len(idx), 500, replace=False)
check("occurrence(): the O(n) sweep equals occurrence_at() probe by probe",
      np.allclose(occ[k], idx.occurrence_at(idx.mz[k])), None)
in_b = np.abs(idx.mz - B) / B * 1e6 < 3
check("a clean recurrent ion's own peaks read its occurrence (~0.95)",
      in_b.any() and np.all(occ[in_b] > 0.85), occ[in_b].min() if in_b.any() else None)
in_a = np.abs(idx.mz - A_TRACE) / A_TRACE * 1e6 < 2       # the core of the wide cloud
check("a wide (4 ppm) TOF cloud: peaks near its centre still read a high occurrence",
      in_a.any() and np.median(occ[in_a]) > 0.6, np.median(occ[in_a]) if in_a.any() else None)
# the random-m/z noise-floor peaks: everything not within 20 ppm of a known ion
known = np.sort(np.r_[WEAK, A_TRACE, B])
j = np.clip(np.searchsorted(known, idx.mz), 1, len(known) - 1)
d_known = np.minimum(np.abs(known[j - 1] - idx.mz), np.abs(known[j] - idx.mz)) / idx.mz * 1e6
is_noise = d_known > 20
check("transient noise reads a low occurrence (< 0.05 for > 97 % of the random-m/z peaks)",
      is_noise.sum() > 20000 and (occ[is_noise] < 0.05).mean() > 0.97, (int(is_noise.sum()), (occ[is_noise] < 0.05).mean()))
check("...and the recurrent weak ions read a high one (the population the path exists for)",
      np.median(occ[d_known < 2]) > 0.6, np.median(occ[d_known < 2]))
check("PeakIndex.sample_edge is the 1st percentile of each sample's heights",
      np.isclose(idx.sample_edge()[0], np.percentile(idx.height[idx.sample == 0], 1.0)))
check("occurrence_at: NaN where NOTHING lies within tolerance",
      np.isnan(idx.occurrence_at([50.0, 900.0])).all())
# a noise mass 12 ppm from ion B sees only itself, not B's cloud: construction ==
# lookup, and no chained bin can hand B's persistence to it
probe = B * (1 + 12e-6)
check("a probe 12 ppm from a recurrent ion does NOT inherit its occurrence",
      np.nan_to_num(idx.occurrence_at([probe])[0], nan=0.0) < 0.1, idx.occurrence_at([probe]))

# --- mean_shift / coverage: the theory-snapped anchor finds its trace ---------------
cov_anchor = idx.coverage_at(A_THEORY, 6.0)
c = idx.mean_shift(A_THEORY, tol_ppm=6.0, max_drift_ppm=10.0)
cov_trace = idx.coverage_at(c, 6.0)
check("the anchor at theory covers a MINORITY of the ion's spectra (the TOF failure)",
      cov_anchor < 0.6, cov_anchor)
check("mean_shift walks the anchor onto the trace centre (+6.5 ppm, within 1.5 ppm)",
      abs((c - A_TRACE) / A_TRACE * 1e6) < 1.5, (c - A_TRACE) / A_TRACE * 1e6)
check("...and the re-centred window covers most of the ion's spectra",
      cov_trace > 0.75 and cov_trace > cov_anchor + 0.15, (cov_anchor, cov_trace))
check("the drift cap is honoured: a 3 ppm cap cannot reach a trace 6.5 ppm away",
      abs((idx.mean_shift(A_THEORY, tol_ppm=6.0, max_drift_ppm=3.0) - A_THEORY) / A_THEORY * 1e6) <= 3.0)
check("mean_shift on empty mass space returns the start unchanged",
      idx.mean_shift(700.0, tol_ppm=6.0, max_drift_ppm=10.0) == 700.0)
cb = idx.mean_shift(B, tol_ppm=6.0, max_drift_ppm=10.0)
check("an anchor already on its trace does not move (Orbitrap-like ion)",
      abs((cb - B) / B * 1e6) < 0.5, (cb - B) / B * 1e6)
sig_a, sig_b = idx.scatter_ppm(c, 6.0), idx.scatter_ppm(cb, 6.0)
check("scatter_ppm measures the per-ion cloud: wide ion ~4 ppm, clean ion ~1 ppm",
      2.5 < sig_a < 6.0 and sig_b < 2.0, (sig_a, sig_b))
check("batch_scatter_ppm is the third quartile over populated traces (NaN with none)",
      np.isfinite(TR.batch_scatter_ppm(idx, [c, cb])) and np.isnan(TR.batch_scatter_ppm(idx, [700.0])))
check("members(): one peak per spectrum inside the window",
      len(np.unique(idx.sample[idx.members(cb, 6.0)])) == len(idx.members(cb, 6.0)))

# --- height in edge units ----------------------------------------------------------
he = idx.height_in_edges()
edge_med = float(np.nanmedian(idx.sample_edge()))
check("height_in_edges: a sample's 1st-percentile peaks sit at ~1x its own edge",
      0.8 < np.nanmedian(he[np.abs(idx.height - edge_med) < 0.1 * edge_med]) < 1.25,
      np.nanmedian(he[np.abs(idx.height - edge_med) < 0.1 * edge_med]))

# ---------------------------------------------------------------------------
# the merged ledger: A minted at THEORY from 2 winner files (the snap), plus a
# phantom competitor C15H21NO3 [M+Br]- minted at +2 ppm from A's trace centre in
# ONE file, plus B. Re-centring + collapse must (1) move A onto its trace,
# (2) put A and the phantom on one trace, (3) let A win on n_files -- the phantom
# has the higher ion_score.
# ---------------------------------------------------------------------------
merged = pd.DataFrame({
    "mz": [A_THEORY, A_TRACE * (1 + 2e-6), B],
    "neutral_formula": ["C10H16O9", "C15H21NO3", "C8H12O6"],
    "adduct": ["[M+NO3]-", "[M+Br]-", "[M-H]-"],
    "tier": ["Candidate", "Candidate", "Assigned"],
    "ion_score": [0.83, 0.88, 0.95],
    "n_files": [2, 1, 6],
})
info = TS.recentre_ledger(merged, index=idx, tol_ppm=6.0, log=lambda *a: None)
check("recentre_ledger adds the trace columns",
      {"mz_anchor", "mz_trace", "trace_offset_ppm", "trace_cov_anchor", "trace_cov",
       "trace_moved", "trace_guarded"} <= set(merged.columns))
check("the theory-snapped anchor moved onto its trace (+5..8 ppm) and gained coverage",
      bool(merged.loc[0, "trace_moved"]) and 5.0 < merged.loc[0, "trace_offset_ppm"] < 8.0
      and merged.loc[0, "trace_cov"] > merged.loc[0, "trace_cov_anchor"] + 0.15,
      merged.loc[0, ["trace_offset_ppm", "trace_cov_anchor", "trace_cov"]].to_dict())
check("the clean ion did not move (its anchor IS its trace)",
      not bool(merged.loc[2, "trace_moved"]) and abs(merged.loc[2, "trace_offset_ppm"]) < 0.5)
check("mz_anchor keeps the merge's m/z; the summary counts the moves",
      merged.loc[0, "mz_anchor"] == A_THEORY and info["n_recentred"] >= 1
      and info["mean_cov_trace"] >= info["mean_cov_anchor"], info)
col = TS.collapse_trace_labels(merged, tol_ppm=6.0, log=lambda *a: None)
check("A and the phantom competitor land on ONE trace; B on its own",
      merged.loc[0, "trace_id"] == merged.loc[1, "trace_id"] != merged.loc[2, "trace_id"]
      and col["n_traces"] == 2, merged[["trace_id", "trace_role"]].to_dict("records"))
check("collapse arbitrates on n_files FIRST: the 2-file HOM beats the 1-file phantom "
      "despite the phantom's higher ion_score",
      merged.loc[0, "trace_role"] == "winner" and merged.loc[1, "trace_role"] == "collapsed"
      and merged.loc[2, "trace_role"] == "single" and col["n_collapsed"] == 1,
      merged[["neutral_formula", "trace_role"]].to_dict("records"))
check("nothing is dropped from the merged ledger", len(merged) == 3)
# a SAME-file, same-tier tie: the merge's own ordering decides by ion_score, and
# proximity to the trace centre only breaks a score tie -- on a TOF the anchors
# were snapped to theory, so the label whose anchor happens to sit nearer the
# centre is not thereby more likely right (live batch: a 1-file C15H21NO3 anchor
# 0.25 ppm from the centre, score 0.846, vs the known monomer's alias 8 ppm off,
# score 0.960)
tie = pd.DataFrame({
    "mz": [A_THEORY, A_TRACE * (1 + 0.3e-6)],
    "neutral_formula": ["C10H17NO12", "C15H21NO3"], "adduct": ["[M-H]-", "[M+Br]-"],
    "tier": ["Candidate", "Candidate"], "ion_score": [0.960, 0.846], "n_files": [1, 1]})
TS.recentre_ledger(tie, index=idx, tol_ppm=6.0, log=lambda *a: None)
TS.collapse_trace_labels(tie, tol_ppm=6.0, log=lambda *a: None)
check("collapse tie (same n_files, same tier): the higher ion_score wins, not the anchor nearer the centre",
      tie.loc[0, "trace_role"] == "winner" and tie.loc[1, "trace_role"] == "collapsed"
      and abs(tie.loc[1, "trace_offset_ppm"]) < abs(tie.loc[0, "trace_offset_ppm"]),
      tie[["neutral_formula", "trace_offset_ppm", "trace_role"]].to_dict("records"))
# the stamping frame stamps from the trace centre and skips the collapsed label
sf = TS.stamping_frame(merged, None)
check("stamping_frame drops the collapsed label and uses the TRACE centre as the stamp m/z",
      len(sf) == 2 and abs(sf.loc[sf.neutral_formula == "C10H16O9", "mz"].iloc[0] - merged.loc[0, "mz_trace"]) < 1e-9)
tol_s, sigma = TS.stamp_tolerance(idx, merged["mz_trace"], tol_ppm=6.0)
check("stamp_tolerance widens the window on a TOF-like batch, capped at 2x the merge tolerance",
      6.0 < tol_s <= 12.0 and np.isfinite(sigma), (tol_s, sigma))
# end to end: stamping coverage of the HOM before vs after
ann_before = TS.annotate_peaks(ts, merged.assign(mz=merged["mz_anchor"])[["mz", "neutral_formula", "adduct", "tier"]], tol_ppm=6.0)
ann_after = TS.annotate_peaks(ts, sf, tol_ppm=tol_s)
def _cov(a, f):
    return a.loc[a.neutral_formula == f, "sample_item_id"].nunique() / N
check("END TO END: the HOM's stamped coverage rises from a minority to most spectra",
      _cov(ann_before, "C10H16O9") < 0.6 and _cov(ann_after, "C10H16O9") > 0.8,
      (_cov(ann_before, "C10H16O9"), _cov(ann_after, "C10H16O9")))
check("...while the clean ion's coverage is unchanged (no-op where the anchor was right)",
      abs(_cov(ann_before, "C8H12O6") - _cov(ann_after, "C8H12O6")) < 0.03,
      (_cov(ann_before, "C8H12O6"), _cov(ann_after, "C8H12O6")))
check("...and the collapsed phantom stamps nothing", _cov(ann_after, "C15H21NO3") == 0.0)
check("one-to-one still holds after re-centring (one stamp per ion per spectrum)",
      ann_after[ann_after.neutral_formula.notna()].groupby(["neutral_formula", "sample_item_id"]).size().max() == 1)

# --- the low-evidence guard ---------------------------------------------------------
# an anchor covering almost nothing that wants to jump > 6 ppm onto a neighbour's
# trace, seen in ONE file and only a Candidate: refused. The same row seen in 2
# files: allowed (corroborated).
# (built on the CLEAN ion B, sigma 1 ppm: an anchor 7.5 ppm below it sees only the
# far tail of the cloud -- coverage well under 10 % -- yet enough peaks to walk)
lone = pd.DataFrame({"mz": [B * (1 - 7.5e-6)], "neutral_formula": ["C12H14O4"],
                     "adduct": ["[M+HBr+Br]-"], "tier": ["Candidate"], "ion_score": [0.7], "n_files": [1]})
g = TS.recentre_ledger(lone, index=idx, tol_ppm=6.0, log=lambda *a: None)
check("guard: an almost-empty 1-file Candidate anchor may not jump > 6 ppm onto a trace",
      bool(lone.loc[0, "trace_guarded"]) and not bool(lone.loc[0, "trace_moved"])
      and g["n_guarded"] == 1, lone.iloc[0].to_dict())
lone2 = lone.assign(n_files=2)[["mz", "neutral_formula", "adduct", "tier", "ion_score", "n_files"]]
TS.recentre_ledger(lone2, index=idx, tol_ppm=6.0, log=lambda *a: None)
check("guard: the same anchor seen in 2 files is corroborated and may move",
      bool(lone2.loc[0, "trace_moved"]) and not bool(lone2.loc[0, "trace_guarded"]))
# no time series: columns added, nothing moves, no crash
nots = merged[["mz", "neutral_formula", "adduct", "tier", "ion_score", "n_files"]].copy()
r0 = TS.recentre_ledger(nots, None, tol_ppm=6.0, log=lambda *a: None)
check("recentre_ledger without a TS is a no-op that still adds the columns",
      r0["n_recentred"] == 0 and (nots["mz_trace"] == nots["mz"]).all() and "trace_cov" in nots.columns)
check("collapse on a ledger without mz_trace is a no-op", TS.collapse_trace_labels(nots.drop(columns=["mz_trace"]), tol_ppm=6.0, log=None)["n_traces"] == 0)

# ---------------------------------------------------------------------------
# Orbitrap-like control: every ion's anchor IS its trace (0.2 ppm scatter), so
# re-centring must change nothing -- the no-op that validates the diagnosis.
# ---------------------------------------------------------------------------
rng2 = np.random.default_rng(5)
# A picker that STOPS at the noise edge: the weak end of its peak list is real
# weak ions (recurrent, 5-8 cps) with only a few transient peaks per spectrum --
# not a random-m/z noise cloud. (That is what the five real batches show: three
# clean Orbitrap modes are 11-16 % transient at 1x the edge.)
ions = np.linspace(150, 450, 40)
weak = np.linspace(151.3, 449.7, 30)
orows = []
for s in range(120):
    sid = f"o{s:03d}"
    for m in ions:
        if rng2.random() < 0.9:
            orows.append((sid, m * (1 + rng2.normal(0, 0.2e-6)), 500 + 100 * rng2.random()))
    for m in weak:
        if rng2.random() < 0.8:
            orows.append((sid, m * (1 + rng2.normal(0, 0.3e-6)), 5.0 + 3 * rng2.random()))
    for _ in range(3):
        orows.append((sid, rng2.uniform(100, 500), 5.0 + rng2.random()))
ots = pd.DataFrame(orows, columns=["sample_item_id", "mz", "height"])
oidx = TR.PeakIndex(ots, tol_ppm=6.0)
oled = pd.DataFrame({"mz": ions, "neutral_formula": [f"C{i}H2O" for i in range(40)],
                     "adduct": "[M-H]-", "tier": "Assigned", "ion_score": 0.9, "n_files": 6})
oi = TS.recentre_ledger(oled, index=oidx, tol_ppm=6.0, log=lambda *a: None)
check("Orbitrap control: no anchor moves more than 0.5 ppm",
      np.abs(oled["trace_offset_ppm"]).max() < 0.5, np.abs(oled["trace_offset_ppm"]).max())
oc = TS.collapse_trace_labels(oled, tol_ppm=6.0, log=lambda *a: None)
check("Orbitrap control: no collapse (one label per trace)", oc["n_collapsed"] == 0 and oc["n_traces"] == 40)
check("Orbitrap control: coverage is unchanged by re-centring",
      oi["mean_cov_trace"] == oi["mean_cov_anchor"], oi)
otol, osig = TS.stamp_tolerance(oidx, oled["mz_trace"], tol_ppm=6.0)
check("Orbitrap control: the stamping window stays at the merge tolerance",
      otol == 6.0 and osig < 1.0, (otol, osig))

# ---------------------------------------------------------------------------
# the batch-derived brightness floor
# ---------------------------------------------------------------------------
tab = ADM.bin_occurrence(ts, tol_ppm=6.0, index=idx)
thr = ADM.resolve_threshold(type("C", (), {"occurrence_min": "auto"})(), tab)
der = ADM.derive_height_cutoff_x_edge(tab, thr)
check("TOF-like batch (picker into the noise): the derived floor is raised above 1x",
      der is not None and der["x_edge"] > 1.0 and der["share_at_1"] > ADM.MAX_TRANSIENT_SHARE
      and der["share_at_x"] <= ADM.MAX_TRANSIENT_SHARE, der)
check("...because its picker leaves a tail below the edge (the second condition)",
      der is not None and der["picker_into_noise"]
      and der["picker_tail_fraction"] > ADM.PICKER_INTO_NOISE_FRACTION, der)
check("the derivation records the whole grid and its constants",
      der is not None and set(der["transient_share"]) <= set(map(float, ADM.X_EDGE_GRID))
      and der["max_transient_share"] == ADM.MAX_TRANSIENT_SHARE
      and der["picker_fraction_min"] == ADM.PICKER_INTO_NOISE_FRACTION
      and der["picker_tail_x"] == ADM.PICKER_TAIL_X and der["bound"] is False, der)
check("the transient share falls as the floor rises (the shape the rule relies on)",
      der is not None and der["transient_share"][1.0] > der["transient_share"][der["x_edge"]], der)
# THE REAGENT-IN-RANGE ORBITRAP LESSON: a hard-threshold picker whose admitted
# population is nevertheless very transient (an m/z-localised artefact: a
# scan-edge pile-up, a reagent ion's skirt) must KEEP 1x -- on the real mode a 2x
# floor removed 45 pile-up rows but also 14 multi-file Assigned ions at 1-2x the
# edge. Same TOF-like chemistry, but the noise floor has a hard lower bound and a
# dense cloud of transient peaks sits at 1-3x it.
rng3 = np.random.default_rng(11)
hrows = []
for s_ in range(N):
    sid = f"h{s_:03d}"
    if rng3.random() < 0.9:
        hrows.append((sid, B * (1 + rng3.normal(0, 1e-6)), 50.0 + rng3.random()))
    for _ in range(200):                                          # hard floor at 1.0
        hrows.append((sid, rng3.uniform(100, 500), 1.0 + 0.2 * rng3.random()))
    for _ in range(300):                                          # transient artefact 1-3x
        hrows.append((sid, rng3.uniform(518, 522), 1.0 + 2.0 * rng3.random()))
hts = pd.DataFrame(hrows, columns=["sample_item_id", "mz", "height"])
hidx = TR.PeakIndex(hts, tol_ppm=6.0)
htab = ADM.bin_occurrence(hts, tol_ppm=6.0, index=hidx)
hthr = ADM.resolve_threshold(type("C", (), {"occurrence_min": "auto"})(), htab)
hder = ADM.derive_height_cutoff_x_edge(htab, hthr)
check("hard-threshold picker + a transient artefact population: the floor STAYS 1x",
      hder is not None and hder["x_edge"] == 1.0 and not hder["picker_into_noise"]
      and hder["picker_tail_fraction"] <= ADM.PICKER_INTO_NOISE_FRACTION
      and hder["share_at_1"] > ADM.MAX_TRANSIENT_SHARE, hder)
check("picker_tail_fraction: ~0 for a hard floor, clearly above for a soft tail, NaN without a table",
      ADM.picker_tail_fraction(htab) < 1e-4 and ADM.picker_tail_fraction(tab) > ADM.PICKER_INTO_NOISE_FRACTION
      and np.isnan(ADM.picker_tail_fraction(None)), (ADM.picker_tail_fraction(htab), ADM.picker_tail_fraction(tab)))
otab = ADM.bin_occurrence(ots, tol_ppm=6.0, index=oidx)
othr = ADM.resolve_threshold(type("C", (), {"occurrence_min": "auto"})(), otab)
oder = ADM.derive_height_cutoff_x_edge(otab, othr)
check("Orbitrap-like batch (picker stops at the edge): the derived floor stays at 1x",
      oder is not None and oder["x_edge"] == 1.0 and not oder["picker_into_noise"], oder)
check("no threshold (path off) -> nothing to derive from", ADM.derive_height_cutoff_x_edge(tab, None) is None
      and ADM.derive_height_cutoff_x_edge(None, 0.5) is None)
check("persistent_trace_count ~ the number of persistent IONS (302 here: 300 weak + A + B), not their ~50k peaks",
      280 <= ADM.persistent_trace_count(tab, thr) <= 360, ADM.persistent_trace_count(tab, thr))


def test_all():
    assert FAIL == 0, f"{FAIL} checks failed"


if __name__ == "__main__":
    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)
