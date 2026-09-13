"""Offline tests for the PREDICTED diagnostic-satellite rows in the batch stamp
(timeseries.stamping_frame / predicted_satellite_rows) and the intensity gate
under which annotate_peaks stamps them. Run: python3 tests/test_predicted_satellites.py

The gap this closes: a per-file ledger claims a satellite only where the picker
picked it, so the faint 15N / 18O lines of a parent Assigned in EVERY assigned
file were missing from the batch stamp and surfaced as unexplained tracks
wherever a plume lifted them (a 6154-spectrum uronium batch: C12H27O4P [M+(CH4N2O)H]+, 15N at
m/z 328.2013 in 109 spectra; C12H14O [M+NH4]+, 18O at 194.1425). The stamp now
predicts those lines and stamps them only where the parent's same-sample height
licenses it. The per-file ledgers are never touched.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from peaky.batch import timeseries as TS  # noqa: E402
from peaky.chem import isotopes as ISO  # noqa: E402

PASS = FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok  {name}")
    else:
        FAIL += 1
        print(f"FAIL  {name}  {detail}")


# --- the predicted lines of an ion ------------------------------------------
F_UREA = "C13H32N2O5P+"      # C12H27O4P [M+(CH4N2O)H]+ (m/z 327.2043)
F_NH4 = "C12H18NO+"          # C12H14O [M+NH4]+ (m/z 192.1383)
lines = {lab: (d, rel) for d, rel, lab in TS._predicted_lines(F_UREA)}
check("lines: a C/N/O/P ion predicts exactly its 13C, 15N and 18O diagnostic lines",
      set(lines) == {"13C", "15N", "18O"}, sorted(lines))
check("lines: every line sits at its EXACT per-atom shift (not a merged centroid)",
      abs(lines["15N"][0] - ISO.D_15N) < 1e-9 and abs(lines["13C"][0] - ISO.D_13C) < 1e-9
      and abs(lines["18O"][0] - ISO.D_18O) < 1e-9, lines)
check("lines: the rel is atom-count aware (2 N -> ~2 x 0.36 %; 13 C -> ~14 %)",
      abs(lines["15N"][1] - 2 * ISO.R_15N_PER_N) < 0.001
      and abs(lines["13C"][1] - 13 * ISO.R_13C_PER_C) < 0.01, lines)
# isotope_pattern's default 6 mDa merge folds 18O (+2.0042) into 13C2 (+2.0067) on
# a carbon-rich ion and labels the centroid 13C2 -- the 18O track the stamp
# exists to explain (C12H18NO+, m/z 194.1425) would then be missed by 13 ppm.
nh4 = {lab: d for d, rel, lab in TS._predicted_lines(F_NH4)}
check("lines: the 18O line of a carbon-rich ion keeps its own shift instead of the 13C2 centroid",
      "18O" in nh4 and abs(nh4["18O"] - ISO.D_18O) < 1e-9, nh4)
check("lines: 13C2 / 2x81Br / M+4 envelope lines are NOT predicted (diagnostic set only)",
      not any(lab not in TS.PRED_SAT_LABELS for _d, _r, lab in TS._predicted_lines("C40H60Br2O2+")))
si = {lab: rel for _d, rel, lab in TS._predicted_lines("C6H19OSi2+")}
check("lines: a Si ion predicts 29Si and 30Si, 2 Si -> ~2 x per-atom",
      abs(si.get("29Si", 0) - 2 * ISO.R_29SI_PER_SI) < 0.005
      and abs(si.get("30Si", 0) - 2 * ISO.R_30SI_PER_SI) < 0.01, si)
check("lines: a labelled ^N adduct predicts no 15N line for the label itself (and no 14N)",
      not {"15N", "14N"} & {lab for _d, _r, lab in TS._predicted_lines("C10H20^NO3+")},
      TS._predicted_lines("C10H20^NO3+"))
check("lines: an unparseable / empty formula predicts nothing",
      TS._predicted_lines("") == () and TS._predicted_lines("nan") == ())


# --- stamping_frame: predicted rows, and what beats them ----------------------
MZ_U, MZ_A = 327.2043, 192.1383
merged = pd.DataFrame({"mz": [MZ_U, MZ_A], "neutral_formula": ["C12H27O4P", "C12H14O"],
                       "adduct": ["[M+(CH4N2O)H]+", "[M+NH4]+"],
                       "tier": ["Assigned", "Assigned"]})
COLS = ["mz", "role", "ion_formula", "iso_label", "neutral_formula", "adduct"]
# two per-file ledgers: both parents Assigned in both, only the 13C claimed (as
# in every ledger of that batch)
idf = pd.DataFrame([
    (MZ_U, "M0", F_UREA, None, "C12H27O4P", "[M+(CH4N2O)H]+"),
    (MZ_A, "M0", F_NH4, None, "C12H14O", "[M+NH4]+"),
    (MZ_U + ISO.D_13C, "iso_child", F_UREA, "13C", None, None),
    (MZ_U + 0.0001, "M0", F_UREA, None, "C12H27O4P", "[M+(CH4N2O)H]+"),
    (MZ_U + ISO.D_13C + 0.0001, "iso_child", F_UREA, "13C", None, None),
], columns=COLS)
sf = TS.stamping_frame(merged, idf, tol_ppm=6.0)
pred = sf[sf.stamp_source == "predicted"]
obs = sf[sf.stamp_source == "observed"]
m0 = sf[sf.stamp_source == "M0"]
check("stamping_frame: every row carries stamp_source (M0 / observed / predicted)",
      set(sf.stamp_source) == {"M0", "observed", "predicted"} and len(m0) == 2
      and sf.stamp_source.notna().all(), sf.stamp_source.value_counts().to_dict())
check("stamping_frame: an M0 carrying N and O gains PREDICTED 15N and 18O rows",
      set(zip(pred.ion_formula, pred.iso_label))
      == {(F_UREA, "15N"), (F_UREA, "18O"), (F_NH4, "15N"), (F_NH4, "18O"), (F_NH4, "13C")},
      pred[["ion_formula", "iso_label"]].to_dict("records"))
check("stamping_frame: the OBSERVED 13C row wins over the predicted 13C (one 13C row, source observed)",
      len(obs) == 1 and obs.iloc[0]["iso_label"] == "13C" and obs.iloc[0]["ion_formula"] == F_UREA
      and not ((pred.ion_formula == F_UREA) & (pred.iso_label == "13C")).any())
r15 = pred[(pred.ion_formula == F_UREA) & (pred.iso_label == "15N")].iloc[0]
check("stamping_frame: predicted row = role iso_child, parent's ion_formula, iso_label, iso_rel, no neutral",
      r15["role"] == "iso_child" and abs(r15["iso_rel"] - 2 * ISO.R_15N_PER_N) < 0.001
      and pd.isna(r15["neutral_formula"]) and pd.isna(r15["adduct"]),
      r15.to_dict())
check("stamping_frame: predicted m/z = the parent's stamped m/z + the exact line shift",
      abs(r15["mz"] - (MZ_U + ISO.D_15N)) < 1e-9, r15["mz"])
check("stamping_frame: a predicted row links to its parent (parent_stamp_id -> the M0's stamp_id)",
      int(r15["parent_stamp_id"]) == int(m0.loc[m0.ion_formula == F_UREA, "stamp_id"].iloc[0])
      and (m0.parent_stamp_id == -1).all() and (obs.parent_stamp_id == -1).all()
      and sf.stamp_id.is_unique)
check("stamping_frame: .attrs counts parents / lines / supersessions / rows kept",
      sf.attrs["predicted_satellites"] == {"n_parents": 2, "n_lines": 6,
                                           "n_superseded_observed": 1,
                                           "n_superseded_track": 0, "n_predicted": 5},
      sf.attrs)
check("stamping_frame: the inputs are not mutated",
      "stamp_source" not in merged.columns and "stamp_source" not in idf.columns
      and len(merged) == 2 and len(idf) == 5)

# an M0 sitting at a satellite offset (an assigned analyte 1.0034 Da above the
# NH4 parent, i.e. on its 13C track) beats the predicted line: no predicted row
# is minted on that track
merged_m0 = pd.concat([merged, pd.DataFrame({
    "mz": [MZ_A + ISO.D_13C + 0.0002], "neutral_formula": ["C11H18O2"],
    "adduct": ["[M+H]+"], "tier": ["Candidate"]})], ignore_index=True)
idf_m0 = pd.concat([idf, pd.DataFrame([(MZ_A + ISO.D_13C + 0.0002, "M0", "C11H19O2+", None,
                                         "C11H18O2", "[M+H]+")], columns=COLS)],
                   ignore_index=True)
sf2 = TS.stamping_frame(merged_m0, idf_m0, tol_ppm=6.0)
p2 = sf2[sf2.stamp_source == "predicted"]
check("stamping_frame: an M0 at a satellite offset wins -- no predicted 13C on its track",
      not ((p2.ion_formula == F_NH4) & (p2.iso_label == "13C")).any()
      and sf2.attrs["predicted_satellites"]["n_superseded_track"] == 1
      and (sf2.stamp_source == "M0").sum() == 3, sf2.attrs)
check("stamping_frame: ... and the other predicted lines of that parent survive",
      ((p2.ion_formula == F_NH4) & (p2.iso_label == "15N")).any()
      and ((p2.ion_formula == F_NH4) & (p2.iso_label == "18O")).any())
# the M0 at the offset also predicts its own satellites (C11H19O2+ -> 13C, 18O)
check("stamping_frame: the analyte on the satellite track predicts its own lines",
      {"13C", "18O"} <= set(p2.loc[p2.ion_formula == "C11H19O2+", "iso_label"]))

# a trace-reconciled ledger: the predicted line is offset from the TRACE centre,
# so the instrument's systematic offset carries over to the satellite
merged_tr = merged.assign(mz_anchor=merged.mz, mz_trace=merged.mz * (1 + 4e-6),
                          trace_role="single")
sf3 = TS.stamping_frame(merged_tr, idf, tol_ppm=6.0)
r3 = sf3[(sf3.stamp_source == "predicted") & (sf3.ion_formula == F_UREA)
         & (sf3.iso_label == "15N")].iloc[0]
check("stamping_frame: a predicted line is the parent's TRACE CENTRE + shift (offset carries over)",
      abs(r3["mz"] - (MZ_U * (1 + 4e-6) + ISO.D_15N)) < 1e-9, r3["mz"])

# switches and legacy inputs
sf_off = TS.stamping_frame(merged, idf, tol_ppm=6.0, predict_satellites=False)
check("stamping_frame: predict_satellites=False restores the observed-only stamp",
      not (sf_off.stamp_source == "predicted").any() and len(sf_off) == 3
      and sf_off.attrs["predicted_satellites"] == {})
check("stamping_frame: no ion_formula anywhere -> nothing to predict, analytes only",
      len(TS.stamping_frame(merged, None)) == 2
      and (TS.stamping_frame(merged, None).stamp_source == "M0").all())


# --- annotate_peaks: the intensity gate on predicted lines ---------------------
MZ_15N = MZ_U + ISO.D_15N          # 328.2013
REL_15N = float(r15["iso_rel"])    # 0.0073
MZ_18O = MZ_U + ISO.D_18O
REL_18O = float(pred[(pred.ion_formula == F_UREA) & (pred.iso_label == "18O")].iloc[0]["iso_rel"])
HP = 1.0e6
ts = pd.DataFrame({
    "sample_item_id": ["s1", "s1",          # parent + a CONSISTENT 15N peak (ratio 1.0)
                       "s2", "s2",          # parent + a peak 70x the predicted height
                       "s3",                # 15N peak, parent ABSENT in this sample
                       "s1",                # a consistent 18O peak (ratio 2.0)
                       "s4", "s4",          # parent + a peak at 0.29x (just below the window)
                       "s5", "s5"],         # parent + a peak at 3.4x (inside the window)
    "mz": [MZ_U, MZ_15N, MZ_U, MZ_15N, MZ_15N, MZ_18O, MZ_U, MZ_15N, MZ_U, MZ_15N],
    "height": [HP, HP * REL_15N, HP, HP * REL_15N * 70, HP * REL_15N, HP * REL_18O * 2.0,
               HP, HP * REL_15N * 0.29, HP, HP * REL_15N * 3.4],
})
ann = TS.annotate_peaks(ts, sf, tol_ppm=6.0)
check("annotate: stamp_source column is emitted on every row",
      "stamp_source" in ann.columns and len(ann) == len(ts))
check("annotate: a consistent 15N peak takes the predicted label (iso_child, parent formula, 'predicted')",
      ann.loc[1, "role"] == "iso_child" and ann.loc[1, "ion_formula"] == F_UREA
      and ann.loc[1, "iso_label"] == "15N" and ann.loc[1, "stamp_source"] == "predicted"
      and abs(ann.loc[1, "ion_mz"] - MZ_15N) < 1e-9, ann.loc[1].to_dict())
check("annotate: a predicted satellite carries NO neutral_formula / adduct / tier",
      pd.isna(ann.loc[1, "neutral_formula"]) and pd.isna(ann.loc[1, "adduct"])
      and pd.isna(ann.loc[1, "tier"]))
check("annotate: an INCONSISTENT height (70x the predicted line) takes no label",
      pd.isna(ann.loc[3, "ion_formula"]) and pd.isna(ann.loc[3, "ion_mz"])
      and not bool(ann.loc[3, "dup_candidate"]), ann.loc[3].to_dict())
check("annotate: parent ABSENT in that sample -> no label",
      pd.isna(ann.loc[4, "ion_formula"]) and pd.isna(ann.loc[4, "ion_mz"]))
check("annotate: the parent itself is stamped M0 in every sample, untouched by the gate",
      all(ann.loc[i, "role"] == "M0" and ann.loc[i, "stamp_source"] == "M0"
          for i in (0, 2, 6, 8)))
check("annotate: a consistent 18O peak (ratio 2.0) takes the predicted 18O label",
      ann.loc[5, "iso_label"] == "18O" and ann.loc[5, "stamp_source"] == "predicted")
check("annotate: the window is the per-file passes' 0.3-3.5 (0.29x rejected, 3.4x accepted)",
      pd.isna(ann.loc[7, "ion_formula"]) and ann.loc[9, "iso_label"] == "15N",
      (ann.loc[7, "ion_formula"], ann.loc[9, "iso_label"]))
check("annotate: pred_ratio is honoured (a 1.5-2.5 window rejects the 1.0x peak, keeps the 2.0x one)",
      pd.isna(TS.annotate_peaks(ts, sf, tol_ppm=6.0, pred_ratio=(1.5, 2.5)).loc[1, "ion_formula"])
      and TS.annotate_peaks(ts, sf, tol_ppm=6.0, pred_ratio=(1.5, 2.5)).loc[5, "iso_label"] == "18O")
check("annotate: without a height column the gate cannot pass -> no predicted stamps, no crash",
      TS.annotate_peaks(ts.drop(columns=["height"]), sf, tol_ppm=6.0)
      .stamp_source.eq("predicted").sum() == 0)
# observed rows are NOT gated: the observed 13C track stamps whatever its height
ts_obs = pd.DataFrame({"sample_item_id": ["s9"], "mz": [MZ_U + ISO.D_13C + 0.00005],
                       "height": [12345.0]})
ann_obs = TS.annotate_peaks(ts_obs, sf, tol_ppm=6.0)
check("annotate: an OBSERVED satellite row stamps without a parent in the sample (not gated)",
      ann_obs.loc[0, "iso_label"] == "13C" and ann_obs.loc[0, "stamp_source"] == "observed")

# the one-to-one contest for a predicted line runs among the GATED candidates: a
# nearer peak that fails the gate cannot beat the real satellite to the label
# (nor turn it into a dup_candidate)
ts_sh = pd.DataFrame({"sample_item_id": ["t1", "t1", "t1"],
                      "mz": [MZ_U, MZ_15N, MZ_15N * (1 + 0.4e-6)],
                      "height": [HP, 50.0, HP * REL_15N]})
ann_sh = TS.annotate_peaks(ts_sh, sf, tol_ppm=6.0)
check("annotate: a nearer peak that fails the gate does not steal the label from the real satellite",
      pd.isna(ann_sh.loc[1, "ion_formula"]) and not bool(ann_sh.loc[1, "dup_candidate"])
      and ann_sh.loc[2, "iso_label"] == "15N" and not bool(ann_sh.loc[2, "dup_candidate"]),
      ann_sh[["mz", "height", "iso_label", "dup_candidate"]].to_dict("records"))
# two gated candidates for one predicted line in one sample: still one stamp
ts_two = pd.DataFrame({"sample_item_id": ["u1"] * 3,
                       "mz": [MZ_U, MZ_15N, MZ_15N * (1 + 1.0e-6)],
                       "height": [HP, HP * REL_15N, HP * REL_15N * 0.9]})
ann_two = TS.annotate_peaks(ts_two, sf, tol_ppm=6.0)
check("annotate: one-to-one holds on predicted lines (one stamp, the loser flagged dup_candidate)",
      ann_two.iso_label.eq("15N").sum() == 1 and int(ann_two.dup_candidate.sum()) == 1
      and ann_two.loc[1, "iso_label"] == "15N" and bool(ann_two.loc[2, "dup_candidate"]))

# an M0 always beats a predicted line for the same track -- even when both rows
# reach annotate_peaks (the frame is built by hand here, bypassing
# stamping_frame's own supersession), and even if the peak sits NEARER the
# predicted m/z
frame = sf.copy()
m0_on_track = pd.DataFrame([{"mz": MZ_15N + 0.0008, "role": "M0", "neutral_formula": "C13H31N2O5P",
                             "adduct": "[M+H]+", "tier": "Candidate", "ion_formula": "C13H32N2O5P+",
                             "iso_label": None, "stamp_source": "M0",
                             "stamp_id": 99, "parent_stamp_id": -1, "iso_rel": np.nan}])
frame = pd.concat([frame, m0_on_track], ignore_index=True)
ts_m0 = pd.DataFrame({"sample_item_id": ["v1", "v1"], "mz": [MZ_U, MZ_15N + 0.0001],
                      "height": [HP, HP * REL_15N]})
ann_m0 = TS.annotate_peaks(ts_m0, frame, tol_ppm=6.0)
check("annotate: an M0 within the window beats the predicted line, even when the peak is nearer the prediction",
      ann_m0.loc[1, "role"] == "M0" and ann_m0.loc[1, "neutral_formula"] == "C13H31N2O5P"
      and ann_m0.loc[1, "stamp_source"] == "M0", ann_m0.loc[1].to_dict())
# a one-to-one LOSER inside a known row's window is never offered to a predicted line
ts_lose = pd.DataFrame({"sample_item_id": ["w1"] * 3,
                        "mz": [MZ_U, MZ_15N + 0.0008, MZ_15N + 0.0001],
                        "height": [HP, 5000.0, HP * REL_15N]})
ann_lose = TS.annotate_peaks(ts_lose, frame, tol_ppm=6.0)
check("annotate: a peak that LOST the contest for a known row is not re-offered to a predicted line",
      bool(ann_lose.loc[2, "dup_candidate"]) and pd.isna(ann_lose.loc[2, "ion_formula"])
      and ann_lose.loc[1, "role"] == "M0",
      ann_lose[["mz", "height", "role", "iso_label", "dup_candidate"]].to_dict("records"))

# legacy ledgers (no stamp_source) behave exactly as before, and emit the column
legacy = merged[["mz", "neutral_formula", "adduct", "tier"]]
ann_leg = TS.annotate_peaks(ts, legacy, tol_ppm=6.0)
check("annotate: a legacy analyte-only ledger -> stamp_source all <NA>, analytes still stamped",
      ann_leg.stamp_source.isna().all() and ann_leg.loc[0, "neutral_formula"] == "C12H27O4P"
      and pd.isna(ann_leg.loc[1, "neutral_formula"]))
# observed-only frame (predict_satellites=False): the 15N peak stays unexplained,
# exactly the pre-change behaviour this feature fixes
ann_off = TS.annotate_peaks(ts, sf_off, tol_ppm=6.0)
check("annotate: with the observed-only stamp the consistent 15N peak stays UNEXPLAINED (the old gap)",
      pd.isna(ann_off.loc[1, "ion_formula"]) and ann_off.loc[0, "role"] == "M0")


# --- track coherence: the rule the per-sample gate cannot express ----------------
# A true satellite's ratio is a constant of nature, so it passes the window in
# (nearly) every judged sample; an independent compound on the line fails in most
# and passes in the few where its height happens to fit. Measured on two live
# runs: 7 and 8 such tracks, 1-26 % pass share, 24 and 268 mislabelled peaks that
# the per-sample gate alone hands out.
SAMPLES = [f"c{i:02d}" for i in range(12)]
_rows = []
for i, s_ in enumerate(SAMPLES):
    _rows.append((s_, MZ_U, HP))                                   # parent everywhere
    _rows.append((s_, MZ_15N, HP * REL_15N * (0.8 + 0.03 * i)))    # coherent: ratios 0.80-1.13
    _rows.append((s_, MZ_18O, HP * REL_18O * (2.0 if i < 3 else 20.0)))   # 3 pass, 9 fail
ts_coh = pd.DataFrame(_rows, columns=["sample_item_id", "mz", "height"])
_stats = {}
ann_coh = TS.annotate_peaks(ts_coh, sf, tol_ppm=6.0, stats=_stats)
check("coherence: the coherent 15N line keeps all 12 stamps",
      int((ann_coh.iso_label == "15N").sum()) == 12)
check("coherence: the incoherent 18O line (3 of 12 judged samples pass) is rejected as a WHOLE",
      int((ann_coh.iso_label == "18O").sum()) == 0 and ann_coh.loc[2, "stamp_source"] is None
      or pd.isna(ann_coh.loc[2, "stamp_source"]))
check("coherence: a rejected line's peaks are neither stamped nor flagged dup_candidate",
      not ann_coh.dup_candidate.any() and ann_coh.loc[ann_coh.mz == MZ_18O, "ion_formula"].isna().all())
_tt = _stats["predicted_tracks"]
_r18 = _tt[_tt.iso_label == "18O"].iloc[0]
_r15 = _tt[_tt.iso_label == "15N"].iloc[0]
check("coherence: stats table tallies SAMPLES judged / passed and the verdict per line",
      list(_tt.columns) == TS.PRED_TRACK_COLS and len(_tt) == 2
      and _r18.n_eval == 12 and _r18.n_pass == 3 and bool(_r18.judged) and not bool(_r18.kept)
      and _r18.n_stamped == 0 and abs(_r18.pass_share - 0.25) < 1e-9
      and _r15.n_eval == 12 and _r15.n_pass == 12 and bool(_r15.kept) and _r15.n_stamped == 12
      and _r15.ion_formula == F_UREA and abs(_r15.ion_mz - MZ_15N) < 1e-9,
      _tt.to_dict("records"))
check("coherence: lines with no candidate peak are not in the table; no predicted rows -> empty table",
      "13C" not in set(_tt.iso_label)
      and (TS.annotate_peaks(ts_coh, sf_off, tol_ppm=6.0, stats=(_e := {})) is not None
           and list(_e["predicted_tracks"].columns) == TS.PRED_TRACK_COLS and len(_e["predicted_tracks"]) == 0))
check("coherence: pred_track_min_share=0 switches the rule off (the 3 passing samples stamp)",
      int((TS.annotate_peaks(ts_coh, sf, tol_ppm=6.0, pred_track_min_share=0).iso_label == "18O").sum()) == 3)
check("coherence: ... and so does pred_track_min_n=0",
      int((TS.annotate_peaks(ts_coh, sf, tol_ppm=6.0, pred_track_min_n=0).iso_label == "18O").sum()) == 3)
# too few judged samples: the line stands on the per-sample gate alone
ts_few = ts_coh[ts_coh.sample_item_id.isin(SAMPLES[2:6])].reset_index(drop=True)   # 18O: 1 pass, 3 fail
_few = {}
ann_few = TS.annotate_peaks(ts_few, sf, tol_ppm=6.0, stats=_few)
check("coherence: a line judged in fewer than min_n samples is NOT judged (1 of 4 passes -> 1 stamp)",
      int((ann_few.iso_label == "18O").sum()) == 1
      and not bool(_few["predicted_tracks"].set_index("iso_label").loc["18O", "judged"]))
check("coherence: lowering min_n to 4 judges it and rejects it (1 of 4 < 50 %)",
      int((TS.annotate_peaks(ts_few, sf, tol_ppm=6.0, pred_track_min_n=4).iso_label == "18O").sum()) == 0)
check("coherence: exactly the share passes -> kept (6 of 12 with min_share 0.5)",
      int((TS.annotate_peaks(pd.concat([ts_coh[ts_coh.mz != MZ_18O],
                                        pd.DataFrame([(s_, MZ_18O, HP * REL_18O * (2.0 if i < 6 else 20.0))
                                                      for i, s_ in enumerate(SAMPLES)],
                                                     columns=ts_coh.columns)], ignore_index=True),
                             sf, tol_ppm=6.0).iso_label == "18O").sum()) == 6)
# samples, not peaks: two failing shoulders beside the real peak in every sample
# must not outvote it (per peak that would be 1 of 3)
_sh = []
for i, s_ in enumerate(SAMPLES):
    _sh += [(s_, MZ_U, HP), (s_, MZ_15N, HP * REL_15N),
            (s_, MZ_15N * (1 + 0.8e-6), 20.0), (s_, MZ_15N * (1 - 0.8e-6), 25.0)]
ann_sh2 = TS.annotate_peaks(pd.DataFrame(_sh, columns=ts_coh.columns), sf, tol_ppm=6.0, stats=(_s2 := {}))
check("coherence: tallied per SAMPLE -- failing shoulders beside the real peak do not outvote it",
      int((ann_sh2.iso_label == "15N").sum()) == 12
      and _s2["predicted_tracks"].set_index("iso_label").loc["15N", "n_pass"] == 12
      and _s2["predicted_tracks"].set_index("iso_label").loc["15N", "n_candidates"] == 36)
check("coherence: samples where the parent is absent do not count as judged",
      _stats["predicted_tracks"].set_index("iso_label").loc["15N", "n_eval"] == 12
      and TS.annotate_peaks(pd.concat([ts_coh, pd.DataFrame([(f"x{i}", MZ_18O, 5.0) for i in range(20)],
                                                             columns=ts_coh.columns)], ignore_index=True),
                            sf, tol_ppm=6.0, stats=(_s3 := {})) is not None
      and _s3["predicted_tracks"].set_index("iso_label").loc["18O", "n_eval"] == 12)


# --- hand-built frames: annotate_peaks must degrade, never raise -----------------
# stamping_frame guarantees unique stamp_ids and the gate columns; a frame someone
# assembled by hand may not, and the public function has to cope
dup_ids = pd.concat([sf, sf[sf.stamp_source == "M0"].iloc[[0]]], ignore_index=True)
ann_dup = TS.annotate_peaks(ts, dup_ids, tol_ppm=6.0)
check("annotate: a frame with a repeated stamp_id does not raise, and the gate still resolves the parent",
      ann_dup.loc[1, "iso_label"] == "15N" and pd.isna(ann_dup.loc[3, "ion_formula"]))
orphan = sf.copy()
orphan.loc[orphan.stamp_source == "predicted", "parent_stamp_id"] = 777
check("annotate: a predicted row whose parent id is not in the frame never stamps",
      TS.annotate_peaks(ts, orphan, tol_ppm=6.0).stamp_source.eq("predicted").sum() == 0)
check("annotate: a frame with predicted rows but no gate columns stamps none of them (no crash)",
      TS.annotate_peaks(ts, sf.drop(columns=["iso_rel", "parent_stamp_id"]), tol_ppm=6.0)
      .stamp_source.eq("predicted").sum() == 0)
check("annotate: no sample column -> one spectrum, the gate still finds the parent",
      TS.annotate_peaks(ts.iloc[[0, 1]].drop(columns=["sample_item_id"]).reset_index(drop=True),
                        sf, tol_ppm=6.0).loc[1, "iso_label"] == "15N")
check("annotate: an empty peaks table emits the columns and no rows",
      "stamp_source" in TS.annotate_peaks(ts.iloc[0:0], sf, tol_ppm=6.0).columns)
check("annotate: predict_satellites off in the frame -> observed rows are still stamped, nothing predicted",
      TS.annotate_peaks(ts, sf_off, tol_ppm=6.0).stamp_source.isin(["M0", "observed"]).sum() == 4)
check("annotate: one_to_one=False keeps the gate (the brighter same-sample parent peak licenses the child)",
      TS.annotate_peaks(pd.DataFrame({"sample_item_id": ["x", "x", "x"],
                                      "mz": [MZ_U, MZ_U * (1 + 1e-6), MZ_15N],
                                      "height": [HP, HP * 0.2, HP * REL_15N]}),
                        sf, tol_ppm=6.0, one_to_one=False).loc[2, "iso_label"] == "15N")


# --- the per-file ledgers are untouched: identified_rows is the same function ----
led = pd.DataFrame({
    "peak_id": ["p1", "p2"], "mz": [MZ_U, MZ_U + ISO.D_13C], "role": ["M0", "iso_child"],
    "neutral_formula": ["C12H27O4P", None], "adduct": ["[M+(CH4N2O)H]+", None],
    "ion_formula": [F_UREA, None], "iso_label": [None, "13C"], "parent_peak_id": [None, "p1"],
    "commentary": ["Pass 1", None]})
idr = TS.identified_rows(led)
check("identified_rows: still reports only what the ledger CLAIMED (no predicted rows there)",
      len(idr) == 2 and set(idr.role) == {"M0", "iso_child"}
      and "stamp_source" not in idr.columns)


def test_all():
    assert FAIL == 0, f"{FAIL} checks failed"


if __name__ == "__main__":
    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)
