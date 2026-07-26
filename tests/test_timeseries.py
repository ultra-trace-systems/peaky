"""Offline tests for timeseries.py. Run: python3 tests/test_timeseries.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from peaky import timeseries as TS  # noqa: E402

PASS = FAIL = 0
def check(name, cond, detail=""):
    global PASS, FAIL
    if cond: PASS += 1; print(f"  ok  {name}")
    else: FAIL += 1; print(f"FAIL  {name}  {detail}")


# --- synthetic 20-sample time series ---------------------------------------
N = 20
k = np.arange(N)
var = 1.0 + 0.8 * np.sin(2 * np.pi * k / N)      # variable (diel-like), cv ~0.57
flat = np.ones(N)                                  # flat / background
PEAKS = {
    236.7555: ("reagent", 1e6 * flat),             # Br3- reagent (normaliser)
    409.0011: ("dibromide", 5000 * flat),          # flat di-bromide cluster
    463.0000: ("contaminant", 8000 * flat),        # flat fluorinated inlet
    279.0236: ("mono1", 5000 * var),               # monoterpene anchor (variable)
    311.0134: ("mono2", 4000 * var),               # monoterpene anchor (variable)
    265.0080: ("ambient", 3000 * var),             # co-varies with monoterpenes
    124.9243: ("formic", 10000 * var),             # formic ref
}
rows = []
for mz, (_tag, tr) in PEAKS.items():
    for s in range(N):
        rows.append(dict(sample_item_id=f"s{s}", mz=mz, height=float(tr[s])))
peaks = pd.DataFrame(rows)

# --- matching ledger -------------------------------------------------------
led = pd.DataFrame([
    dict(peak_id="r1", mz=236.7555, height=1e6, role="reagent",
         neutral_formula=None, ion_formula="Br3-", adduct=None, tier=None, tier_reason=""),
    dict(peak_id="db", mz=409.0011, height=5000, role="M0",
         neutral_formula="C15H22O3", ion_formula="C15H23Br2O3-", adduct="[M+HBr+Br]-",
         tier="Assigned", tier_reason="series"),
    dict(peak_id="ct", mz=463.0000, height=8000, role="M0",
         neutral_formula="C12H12F12", ion_formula="C12H12BrF12-", adduct="[M+Br]-",
         tier="Assigned", tier_reason="unique"),
    dict(peak_id="m1", mz=279.0236, height=5000, role="M0",
         neutral_formula="C10H16O4", ion_formula="C10H16BrO4-", adduct="[M+Br]-",
         tier="Assigned", tier_reason="iso"),
    dict(peak_id="m2", mz=311.0134, height=4000, role="M0",
         neutral_formula="C10H16O6", ion_formula="C10H16BrO6-", adduct="[M+Br]-",
         tier="Assigned", tier_reason="iso"),
    dict(peak_id="am", mz=265.0080, height=3000, role="M0",
         neutral_formula="C9H14O4", ion_formula="C9H14BrO4-", adduct="[M+Br]-",
         tier="Candidate", tier_reason="ladder"),
    dict(peak_id="fo", mz=124.9243, height=10000, role="M0",
         neutral_formula="CH2O2", ion_formula="CH2BrO2-", adduct="[M+Br]-",
         tier="Assigned", tier_reason="iso"),
])

summ = TS.apply_timeseries(led, peaks, log=lambda *a: None)

# --- assertions ------------------------------------------------------------
def disp(pid): return led.loc[led.peak_id == pid, "ts_disposition"].iloc[0]
def cv(pid): return led.loc[led.peak_id == pid, "ts_cv_norm"].iloc[0]
def tier(pid): return led.loc[led.peak_id == pid, "tier"].iloc[0]

check("matrix built + annotated all M0", summ["annotated"] == 6, summ)
check("di-bromide flat -> background disposition", disp("db").startswith("background:di-bromide"), disp("db"))
check("di-bromide flat -> DEMOTED Assigned->Candidate", tier("db") == "Candidate", tier("db"))
check("demote count >=1", summ["demoted"] >= 1, summ)
check("fluorinated flat -> inlet contaminant", "inlet/instrument" in disp("ct"), disp("ct"))
check("contaminant NOT demoted (not di-bromide/CO3)", tier("ct") == "Assigned", tier("ct"))
check("flat peaks have low cv_norm", cv("db") < 0.25 and cv("ct") < 0.25, (cv("db"), cv("ct")))
check("variable ambient peak high cv_norm", cv("am") > 0.4, cv("am"))
check("co-varying peak -> ambient disposition", disp("am").startswith("ambient"), disp("am"))
check("monoterpene anchor -> ambient", disp("m1").startswith("ambient"), disp("m1"))

# CO3-channel flat demotion
led2 = pd.DataFrame([
    dict(peak_id="r1", mz=236.7555, height=1e6, role="reagent", neutral_formula=None,
         ion_formula="Br3-", adduct=None, tier=None, tier_reason=""),
    dict(peak_id="co3", mz=361.0653, height=4000, role="M0", neutral_formula="C12H15NO8",
         ion_formula="C13H15NO11-", adduct="[M+CO3]-", tier="Assigned", tier_reason="x"),
])
peaks2 = pd.DataFrame([dict(sample_item_id=f"s{s}", mz=mz, height=float(h[s]))
                       for mz, h in {236.7555: 1e6*flat, 361.0653: 4000*flat}.items() for s in range(N)])
TS.apply_timeseries(led2, peaks2, log=lambda *a: None)
check("flat CO3-channel -> demoted", led2.loc[led2.peak_id=="co3","tier"].iloc[0] == "Candidate",
      led2.loc[led2.peak_id=="co3","tier"].iloc[0])

# no reagent -> graceful (cv still computed on raw)
led3 = led.copy()
s3 = TS.apply_timeseries(led3, peaks, reagent_mzs=[], log=lambda *a: None)
check("runs without a reagent normaliser", s3["annotated"] == 6, s3)

# --- auto_bin_minutes: native sample cadence (non-averaged), not span/50 --------
_base = pd.Timestamp("2026-06-03T00:00:00Z")
# 316 samples, 5-min cadence over ~26 h: native -> 5; the old span/50 would give ~32
_ts24 = pd.DataFrame([{"sample_item_id": f"s{i}", "datetime_utc": _base + pd.Timedelta(minutes=5 * i),
                       "mz": 100.0, "height": 1.0} for i in range(316)])
check("auto_bin_minutes uses native cadence (5 min), not span/50 (~32)",
      TS.auto_bin_minutes(_ts24) == 5, TS.auto_bin_minutes(_ts24))
_ts6 = pd.DataFrame([{"sample_item_id": f"s{i}", "datetime_utc": _base + pd.Timedelta(minutes=6 * i),
                      "mz": 100.0, "height": 1.0} for i in range(10)])
check("auto_bin_minutes returns the native cadence (6 min)", TS.auto_bin_minutes(_ts6) == 6)
check("auto_bin_minutes floors at >=1 and falls back on <3 samples",
      TS.auto_bin_minutes(_ts24) >= 1 and isinstance(TS.auto_bin_minutes(_ts6.head(2)), int))
# sub-minute / non-integer cadence must round UP, never down: a bin narrower than the
# real spacing aliases -> empty time bins -> a spurious drop-to-floor comb (a Br-
# was 73 s cadence -> the old round() gave a 60 s bin with ~19% empty bins).
_ts73 = pd.DataFrame([{"sample_item_id": f"s{i}", "datetime_utc": _base + pd.Timedelta(seconds=73 * i),
                       "mz": 100.0, "height": 1.0} for i in range(80)])
check("auto_bin_minutes rounds the 73s cadence UP to 2 min (not down to 1 -> aliasing)",
      TS.auto_bin_minutes(_ts73) == 2, TS.auto_bin_minutes(_ts73))
check("bin width >= sample cadence, so no INTERIOR time bin is empty",
      TS.auto_bin_minutes(_ts73) * 60 >= 73)

# --- trace(): pull one compound's time series from a run dir ---------------
import os as _os          # noqa: E402
import tempfile as _tf    # noqa: E402

_rd = _tf.mkdtemp()
_times = pd.to_datetime(["2026-06-03T00:00:00Z", "2026-06-03T06:00:00Z",
                         "2026-06-03T12:00:00Z"])
_rows = []
for _i, _t in enumerate(_times):
    _rows += [{"datetime_utc": _t, "peak_id": f"a{_i}", "mz": 200.0, "height": 100.0 * (_i + 1), "area": 1.0},
              {"datetime_utc": _t, "peak_id": f"b{_i}", "mz": 300.0, "height": 50.0, "area": 1.0}]
pd.DataFrame(_rows).to_parquet(_os.path.join(_rd, "X_ts.parquet"))
pd.DataFrame([{"mz": 200.0, "neutral_formula": "C10H8O4", "adduct": "[M-H]-",
               "tier": "Assigned", "ion_score": 0.9}]).to_csv(
    _os.path.join(_rd, "merged_ledger.csv"), index=False)

_tr = TS.trace(_rd, "C10H8O4")
check("trace by formula resolves the assignment + sums the m/z window per time",
      _tr.attrs["assignment"].startswith("C10H8O4 [M-H]- (Assigned)")
      and len(_tr) == 3 and list(_tr["height"]) == [100.0, 200.0, 300.0], _tr.attrs)
_tu = TS.trace(_rd, 300.0)
check("trace by m/z of an unexplained peak labels it 'unassigned'",
      _tu.attrs["assignment"] == "unassigned" and len(_tu) == 3
      and list(_tu["height"]) == [50.0, 50.0, 50.0], _tu.attrs)
check("trace tol_ppm window returns nothing for an absent mass",
      len(TS.trace(_rd, 250.0)) == 0)


# --- annotate_peaks: stamp ts peaks with assigned formula/channel ----------
_ann_ts = pd.DataFrame({
    "sample_item_id": ["s1", "s1", "s2", "s1", "s2"],
    "mz": [202.1438,          # exact match  -> C10H16O3 [M+NH4]+
           202.1450,          # +5.9 ppm     -> still within 6 ppm
           186.1489,          # match        -> C10H16O2 [M+H]+
           202.2000,          # +278 ppm     -> too far, unmatched
           999.9999],         # no ledger ion nearby, unmatched
    "height": [100, 90, 50, 30, 10],
})
_ann_led = pd.DataFrame({
    "mz": [202.1438, 186.1489],
    "neutral_formula": ["C10H16O3", "C10H16O2"],
    "adduct": ["[M+NH4]+", "[M+H]+"],
    "tier": ["Assigned", "Candidate"],
})
_ann = TS.annotate_peaks(_ann_ts, _ann_led, tol_ppm=6.0)
check("annotate: columns added",
      all(c in _ann.columns
          for c in ("neutral_formula", "adduct", "tier", "ion_mz", "dup_candidate")))
check("annotate: exact match -> formula+channel",
      _ann.loc[0, "neutral_formula"] == "C10H16O3" and _ann.loc[0, "adduct"] == "[M+NH4]+")
# rows 0 and 1 are BOTH sample s1 and both inside the C10H16O3 window: the
# many-to-one match would stamp the formula twice in one sample. One-to-one keeps
# the nearer peak (row 0, exact) and flags row 1 instead.
check("annotate: near-duplicate in the SAME sample is NOT stamped twice",
      pd.isna(_ann.loc[1, "neutral_formula"]) and pd.isna(_ann.loc[1, "ion_mz"]))
check("annotate: the losing near-duplicate is flagged dup_candidate",
      bool(_ann.loc[1, "dup_candidate"]))
check("annotate: the winner is NOT flagged dup_candidate",
      not bool(_ann.loc[0, "dup_candidate"]))
check("annotate: second ion matched",
      _ann.loc[2, "neutral_formula"] == "C10H16O2" and _ann.loc[2, "tier"] == "Candidate")
check("annotate: out-of-tol -> NA",
      pd.isna(_ann.loc[3, "neutral_formula"]) and pd.isna(_ann.loc[3, "ion_mz"]))
check("annotate: no-neighbour -> NA",
      pd.isna(_ann.loc[4, "neutral_formula"]))
check("annotate: ion_mz is the matched ledger m/z",
      abs(_ann.loc[0, "ion_mz"] - 202.1438) < 1e-9)
check("annotate: original rows/columns preserved",
      len(_ann) == len(_ann_ts) and _ann["height"].tolist() == _ann_ts["height"].tolist())
check("annotate: empty ledger -> all NA, no crash",
      TS.annotate_peaks(_ann_ts, pd.DataFrame(columns=["mz", "neutral_formula"]))
      ["neutral_formula"].isna().all())

# --- one-to-one stamping: THE invariant downstream software relies on ---------
# a raw doublet (the real peak + a 1.2 mDa shoulder, as seen for C8H4O3 [M+H]+ at
# m/z 149.0233 in every sample of the Wind-zone-2 batch) must yield ONE stamped
# trace per sample, not two.
_dup_led = pd.DataFrame({
    "mz": [149.023307], "neutral_formula": ["C8H4O3"],
    "adduct": ["[M+H]+"], "tier": ["Assigned"]})
_dup_ts = pd.DataFrame({
    "sample_item_id": ["s1", "s1", "s2", "s2", "s3"],
    "mz": [149.023312, 149.022113,        # s1: real peak + shoulder
           149.022146, 149.023308,        # s2: SAME pair, reversed row order
           149.023300],                   # s3: real peak only
    "height": [40134.0, 699.0, 970.0, 35770.0, 21000.0],
})
_d = TS.annotate_peaks(_dup_ts, _dup_led, tol_ppm=5.0)
_st = _d[_d.neutral_formula.notna()]
check("one-to-one: exactly one stamped peak per sample",
      _st.groupby("sample_item_id").size().max() == 1 and len(_st) == 3,
      _st.groupby("sample_item_id").size().to_dict())
check("one-to-one: the WINNER is the peak the assignment committed (~149.02331)",
      bool((_st.mz - 149.023307).abs().max() < 2e-5), _st.mz.tolist())
check("one-to-one: the 1.2 mDa shoulder is left unassigned",
      pd.isna(_d.loc[1, "neutral_formula"]) and pd.isna(_d.loc[2, "neutral_formula"]))
check("one-to-one: both shoulders flagged dup_candidate",
      bool(_d.loc[1, "dup_candidate"]) and bool(_d.loc[2, "dup_candidate"]))
check("one-to-one: winner is row-order independent (s2 picks the 2nd row)",
      _d.loc[3, "neutral_formula"] == "C8H4O3" and not bool(_d.loc[3, "dup_candidate"]))
check("one-to-one: a lone peak is unaffected (no false dup flag)",
      _d.loc[4, "neutral_formula"] == "C8H4O3" and not bool(_d.loc[4, "dup_candidate"]))
check("one-to-one: NO rows are dropped -- losers kept, just unstamped",
      len(_d) == len(_dup_ts) and _d.height.tolist() == _dup_ts.height.tolist())
# the escape hatch reproduces the old many-to-one behaviour
_raw = TS.annotate_peaks(_dup_ts, _dup_led, tol_ppm=5.0, one_to_one=False)
check("one_to_one=False restores the many-to-one stamp (back-compat)",
      int(_raw.neutral_formula.notna().sum()) == 5
      and not _raw.dup_candidate.any())
# EXACT tie (identical m/z) -> the brighter peak wins, regardless of row order.
# NB an "equidistant" pair either side of the ion is NOT a tie in floating point
# (149.022307 and 149.024307 differ in |delta| by 3e-14), so the tie-break is
# exercised with duplicate m/z, which is the case that genuinely reaches it.
_tie_ts = pd.DataFrame({
    "sample_item_id": ["s1", "s1", "s2", "s2"],
    "mz": [149.023307, 149.023307,        # s1: identical m/z, dim row first
           149.023307, 149.023307],       # s2: identical m/z, bright row first
    "height": [100.0, 9000.0, 9000.0, 100.0]})
_t = TS.annotate_peaks(_tie_ts, _dup_led, tol_ppm=5.0)
check("one-to-one: exact tie breaks to the BRIGHTER peak (dim row first)",
      pd.isna(_t.loc[0, "neutral_formula"]) and _t.loc[1, "neutral_formula"] == "C8H4O3",
      _t[["mz", "height", "neutral_formula"]].to_dict("records"))
check("one-to-one: exact tie is row-order independent (bright row first)",
      _t.loc[2, "neutral_formula"] == "C8H4O3" and pd.isna(_t.loc[3, "neutral_formula"]))
check("one-to-one: tie still yields exactly one stamp per sample",
      _t[_t.neutral_formula.notna()].groupby("sample_item_id").size().tolist() == [1, 1])
# two DIFFERENT ions in one sample must both still be stamped
_two_led = pd.DataFrame({
    "mz": [149.023307, 186.148900],
    "neutral_formula": ["C8H4O3", "C10H16O2"],
    "adduct": ["[M+H]+", "[M+H]+"], "tier": ["Assigned", "Assigned"]})
_two = TS.annotate_peaks(
    pd.DataFrame({"sample_item_id": ["s1", "s1"], "mz": [149.023307, 186.148900],
                  "height": [10.0, 20.0]}), _two_led, tol_ppm=5.0)
check("one-to-one: distinct ions in one sample are both stamped",
      int(_two.neutral_formula.notna().sum()) == 2 and not _two.dup_candidate.any())
# no sample column -> whole table treated as one spectrum, still one winner/ion
_nos = TS.annotate_peaks(
    pd.DataFrame({"mz": [149.023312, 149.022113], "height": [40134.0, 699.0]}),
    _dup_led, tol_ppm=5.0)
check("one-to-one: works without a sample column (single-spectrum table)",
      int(_nos.neutral_formula.notna().sum()) == 1 and bool(_nos.loc[1, "dup_candidate"]))

# --- consensus: the SAME physical track must win in every sample --------------
# Mirrors C18H30O6 [M+(CH4N2O)H]+ (ledger 403.24341) on the Wind-zone-2 batch: two
# raw tracks, A at -1.2 mDa (dominant, bright) and B at +0.45 mDa (nearer the
# ledger mass). Ranking on the bare ledger mass picks B wherever B exists and A
# elsewhere -> the trace alternates between two peaks. Consensus locks onto A.
_bi_led = pd.DataFrame({"mz": [403.24341], "neutral_formula": ["C18H30O6"],
                        "adduct": ["[M+(CH4N2O)H]+"], "tier": ["Assigned"]})
_A, _B = 403.24341 - 1.2e-3, 403.24341 + 0.45e-3
_bi_ts = pd.DataFrame({
    "sample_item_id": ["s1", "s1", "s2", "s2", "s3", "s4"],
    "mz":     [_A, _B, _A, _B, _A, _A],   # s1,s2: both tracks; s3,s4: A only
    "height": [8000., 500., 7500., 450., 9000., 8500.],
})
_bi_led_c = TS.annotate_peaks(_bi_ts, _bi_led, tol_ppm=5.0, consensus=False)
_bi_con = TS.annotate_peaks(_bi_ts, _bi_led, tol_ppm=5.0)
_wl = _bi_led_c[_bi_led_c.neutral_formula.notna()].mz.to_numpy()
_wc = _bi_con[_bi_con.neutral_formula.notna()].mz.to_numpy()
# NB absolute tolerance, NOT bare np.isclose: its default rtol=1e-5 is a 4.03 mDa
# window at m/z 403, wider than the 1.65 mDa track separation under test, which
# would make both of these assertions vacuous (and mutually contradictory) --
# every winner would satisfy both. The length guard also catches one_to_one=False.
check("consensus OFF: ledger-centred ranking picks track B where B exists (flips)",
      bool(len(_wl) == 4 and abs(_wl[:2] - _B).max() < 1e-6
           and abs(_wl[2:] - _A).max() < 1e-6),
      ((_wl - 403.24341) * 1e3).round(2).tolist())
check("consensus ON: the dominant track A wins in EVERY sample (no flip)",
      bool(len(_wc) == 4 and abs(_wc - _A).max() < 1e-6),
      ((_wc - 403.24341) * 1e3).round(2).tolist())
check("consensus: still exactly one stamp per sample",
      _bi_con[_bi_con.neutral_formula.notna()].groupby("sample_item_id").size().max() == 1)
check("consensus: the off-track peak is flagged, not stamped",
      bool(_bi_con.loc[1, "dup_candidate"]) and pd.isna(_bi_con.loc[1, "neutral_formula"]))
# a sample where ONLY the off-track peak exists still gets stamped (it is the same
# ion with a bistable fit -- 4.1 ppm apart is far below any real resolving power --
# so dropping it would lose a genuine data point)
_only_b = TS.annotate_peaks(
    pd.concat([_bi_ts, pd.DataFrame({"sample_item_id": ["s5"], "mz": [_B],
                                     "height": [400.]})], ignore_index=True),
    _bi_led, tol_ppm=5.0)
check("consensus: a sample with ONLY the off-track peak is still stamped",
      _only_b.iloc[-1]["neutral_formula"] == "C18H30O6")

# _consensus_offsets: the MODE, not the mean/median (which land in the empty gap)
_off = np.array([-1.2e-3] * 8 + [0.45e-3] * 3)
_hgt = np.array([8000.0] * 8 + [500.0] * 3)
_c = TS._consensus_offsets(1, np.zeros(11, dtype=int), _off, _hgt,
                           np.array([0.67e-3]))
check("_consensus_offsets: locks onto the dominant (heaviest) mode",
      abs(_c[0] - (-1.2e-3)) < 1e-6, _c[0])
check("_consensus_offsets: does NOT land between the two modes (mean would)",
      abs(_c[0] - _off.mean()) > 2e-4, (_c[0], _off.mean()))
check("_consensus_offsets: no candidates -> 0.0 offset, no crash",
      TS._consensus_offsets(2, np.array([], dtype=int), np.array([]),
                            np.array([]), np.array([1e-3, 1e-3])).tolist() == [0.0, 0.0])

# --- the sidelobe trap: ubiquitous-but-DIM must not outvote the ledger mass ----
# Real regression (Wind-zone-2, C12H19NO6 [M+H]+ @ 274.12850): an FT ringing
# sidelobe of a 601146-cps neighbour 5 mDa away recurs in 559 samples at 1576 cps,
# while the peak the ASSIGNMENT committed sits at the ledger mass in 70 samples at
# 2390 cps. Scoring tracks by SUMMED height hands the ion to the artifact; scoring
# by the brightest member (+ the ledger-mass anchor) keeps the real peak.
_side_off = np.r_[np.zeros(70), np.full(559, -1.27e-3)]          # on-mass, sidelobe
_side_h = np.r_[np.full(70, 2390.0), np.full(559, 1576.0)]
_side_c = TS._consensus_offsets(1, np.zeros(629, dtype=int), _side_off, _side_h,
                                np.array([0.81e-3]))
check("consensus: a dim-but-ubiquitous sidelobe does NOT capture the ion",
      abs(_side_c[0]) < 1e-6, _side_c[0] * 1e3)
check("consensus: summed height WOULD have lost it (the bug this guards)",
      (559 * 1576.0) > (70 * 2390.0))
# ... but a genuinely brighter off-ledger track still wins (C19H34O6Si @ 404.24605:
# 1205 cps in 1074 samples off-mass vs 473 cps in 20 on-mass) -- the anchor is a
# margin, not a veto.
_gen_off = np.r_[np.zeros(20), np.full(1074, 1.45e-3)]
_gen_h = np.r_[np.full(20, 473.0), np.full(1074, 1205.0)]
_gen_c = TS._consensus_offsets(1, np.zeros(1094, dtype=int), _gen_off, _gen_h,
                               np.array([0.81e-3]))
check("consensus: a decisively brighter off-ledger track still wins",
      abs(_gen_c[0] - 1.45e-3) < 1e-6, _gen_c[0] * 1e3)
# the ANCHOR (as opposed to the brightness rule above, which already settles the
# sidelobe case) is what decides the in-between band: an off-ledger track brighter
# than the on-mass one but not decisively so. Real case, C14H28O3Si [M+H]+ @
# 273.18802: 698 cps off-mass vs 375 on-mass = 1.86x, just under ANCHOR_MARGIN, so
# the assignment's own mass keeps the ion.
_amb_off = np.r_[np.zeros(75), np.full(686, 0.91e-3)]
_amb_h = np.r_[np.full(75, 375.0), np.full(686, 698.0)]
def _amb(margin):
    return TS._consensus_offsets(1, np.zeros(761, dtype=int), _amb_off, _amb_h,
                                 np.array([0.30e-3]), anchor_margin=margin)[0]
check("consensus: a merely-1.86x-brighter off-ledger track does NOT displace the "
      "ledger mass (anchor holds)", abs(_amb(TS.ANCHOR_MARGIN)) < 1e-6, _amb(2.0) * 1e3)
check("consensus: lowering anchor_margin below 1.86 lets it through (the anchor is "
      "the deciding rule here)", abs(_amb(1.5) - 0.91e-3) < 1e-6, _amb(1.5) * 1e3)


# --- flag_sidelobe_channels: trust the formula, distrust the channel ----------
# Modelled on the real case (Wind-zone-2): C18H30O6 [M+(CH4N2O)H]+ @ 403.24341 sits
# 11.5 mDa from a 520k-cps C20H34O8 [M+H]+ and holds a locked 0.71% of it (cv 0.033),
# while genuinely independent neighbours of bright peaks run cv 0.21-1.09.
_rng = np.random.default_rng(7)
_ns = 120
_sids = [f"s{i:03d}" for i in range(_ns)]
_parent_h = 520_000 * (1 + 0.25 * _rng.standard_normal(_ns))     # parent varies
_PMZ, _SMZ, _IMZ = 403.23200, 403.24380, 403.24800
_CLEAN = 343.21149          # C18H30O6's OTHER channel: clean, no bright neighbour
_rows = []
for i, sid in enumerate(_sids):
    _rows.append({"sample_item_id": sid, "mz": _PMZ, "height": _parent_h[i]})
    # sidelobe: a LOCKED 0.71% of the parent (+-1% jitter) -> cv ~0.01
    _rows.append({"sample_item_id": sid, "mz": _SMZ,
                  "height": _parent_h[i] * 0.0071 * (1 + 0.01 * _rng.standard_normal())})
    # independent ion of the SAME brightness near the same parent -> cv large
    _rows.append({"sample_item_id": sid, "mz": _IMZ,
                  "height": 3700 * float(np.exp(0.5 * _rng.standard_normal()))})
    # the corroborating [M+H]+ channel of the same neutral, far from any parent
    _rows.append({"sample_item_id": sid, "mz": _CLEAN,
                  "height": 10800 * float(np.exp(0.4 * _rng.standard_normal()))})
_sl_ts = pd.DataFrame(_rows)
_sl_led = pd.DataFrame({
    "mz": [_PMZ, _SMZ, _IMZ, _CLEAN],
    "neutral_formula": ["C20H34O8", "C18H30O6", "C9H17NO5", "C18H30O6"],
    "adduct": ["[M+H]+", "[M+(CH4N2O)H]+", "[M+(CH4N2O)H]+", "[M+H]+"],
    "tier": ["Assigned"] * 4})
_out = TS.flag_sidelobe_channels(_sl_led, _sl_ts, log=lambda *a: None)
check("sidelobe: the locked-ratio channel is flagged intensity_suspect",
      bool(_sl_led.loc[1, "intensity_suspect"]), _out)
check("sidelobe: an INDEPENDENT ion beside the same parent is NOT flagged",
      not bool(_sl_led.loc[2, "intensity_suspect"]))
check("sidelobe: the bright parent itself is NOT flagged",
      not bool(_sl_led.loc[0, "intensity_suspect"]))
check("sidelobe: the parent m/z is recorded for audit",
      abs(float(_sl_led.loc[1, "sidelobe_parent_mz"]) - _PMZ) < 1e-6,
      _sl_led.loc[1, "sidelobe_parent_mz"])
check("sidelobe: a CORROBORATED neutral keeps formula AND tier (only the "
      "channel's intensity is doubted)",
      _sl_led.loc[1, "neutral_formula"] == "C18H30O6"
      and _sl_led.loc[1, "tier"] == "Assigned", _sl_led.loc[1, "tier"])
check("sidelobe: the corroborating clean channel is itself untouched",
      _sl_led.loc[3, "tier"] == "Assigned"
      and not bool(_sl_led.loc[3, "intensity_suspect"]))
check("sidelobe: exactly one channel flagged", _out["suspect"] == 1, _out)
check("sidelobe: nothing demoted while corroboration exists",
      _out["demoted"] == 0, _out)

# UNCORROBORATED: the neutral's only evidence IS the sidelobe -> the assignment
# itself is unsupported, so it is demoted (never deleted). Real cases: C8H19NO9
# [M+H]+ and C31H30 [M+H]+, which have no second channel in their runs.
_un_led = pd.DataFrame({
    "mz": [_PMZ, _SMZ], "neutral_formula": ["C20H34O8", "C31H30"],
    "adduct": ["[M+H]+", "[M+H]+"], "tier": ["Assigned", "Assigned"]})
_uo = TS.flag_sidelobe_channels(_un_led, _sl_ts, log=lambda *a: None)
check("sidelobe: an UNCORROBORATED sidelobe assignment is demoted to Candidate",
      _un_led.loc[1, "tier"] == "Candidate" and _uo["demoted"] == 1, _uo)
check("sidelobe: demotion never deletes the row (no-drop rule)", len(_un_led) == 2)
check("sidelobe: the formula is still recorded after demotion",
      _un_led.loc[1, "neutral_formula"] == "C31H30")
check("sidelobe: demote_uncorroborated=False keeps the tier",
      TS.flag_sidelobe_channels(
          pd.DataFrame({"mz": [_PMZ, _SMZ], "neutral_formula": ["C20H34O8", "C31H30"],
                        "adduct": ["[M+H]+", "[M+H]+"], "tier": ["Assigned"] * 2}),
          _sl_ts, demote_uncorroborated=False, log=lambda *a: None)["demoted"] == 0)
# a peak far from any saturating parent is never considered
_far_led = pd.DataFrame({"mz": [200.1], "neutral_formula": ["C5H8O2"],
                         "adduct": ["[M+H]+"], "tier": ["Assigned"]})
TS.flag_sidelobe_channels(_far_led, _sl_ts, log=lambda *a: None)
check("sidelobe: an ion with no saturating neighbour is not flagged",
      not bool(_far_led.loc[0, "intensity_suspect"]))
# schema stability: no TS -> columns still exist, all False
_no_ts = pd.DataFrame({"mz": [_SMZ], "neutral_formula": ["C18H30O6"],
                       "adduct": ["[M+H]+"], "tier": ["Assigned"]})
TS.flag_sidelobe_channels(_no_ts, None, log=lambda *a: None)
check("sidelobe: without a TS the columns still exist (stable schema)",
      "intensity_suspect" in _no_ts.columns and "sidelobe_parent_mz" in _no_ts.columns
      and not bool(_no_ts.loc[0, "intensity_suspect"]))
# a brightness ratio under FACTOR is not a sidelobe situation at all
_weak = _sl_ts.copy()
_weak.loc[_weak.mz == _PMZ, "height"] = 60_000.0       # only ~16x the satellite
_weak_led = _sl_led[["mz", "neutral_formula", "adduct", "tier"]].copy()
_o2 = TS.flag_sidelobe_channels(_weak_led, _weak, log=lambda *a: None)
check("sidelobe: a neighbour below the 100x factor does not trigger it",
      _o2["suspect"] == 0, _o2)

# the flag rides through to the ts parquet stamp
_ann_sus = TS.annotate_peaks(_sl_ts, _sl_led, tol_ppm=6.0)
_st = _ann_sus[(_ann_sus.neutral_formula == "C18H30O6")
               & (_ann_sus.adduct == "[M+(CH4N2O)H]+")]
check("sidelobe: annotate_peaks carries intensity_suspect onto stamped rows",
      len(_st) > 0 and bool(_st.intensity_suspect.all()), len(_st))
# the SAME neutral's clean channel must stay unflagged -- the flag is per CHANNEL,
# not per compound, which is the whole point of the design
_cl = _ann_sus[(_ann_sus.neutral_formula == "C18H30O6")
               & (_ann_sus.adduct == "[M+H]+")]
check("sidelobe: the same neutral's CLEAN channel stays unflagged (per-channel)",
      len(_cl) > 0 and not bool(_cl.intensity_suspect.any()), len(_cl))
_ind = _ann_sus[_ann_sus.neutral_formula == "C9H17NO5"]
check("sidelobe: the independent ion's stamped rows stay unflagged",
      len(_ind) > 0 and not bool(_ind.intensity_suspect.any()))
check("sidelobe: unmatched rows default to False, never <NA>",
      _ann_sus.intensity_suspect.dtype == bool)


# --- identified_rows / stamping_frame: ion identity for NON-analyte tracks ----
# The parquet should stamp EVERY known ion (reagent ladder, isotope satellites,
# artifacts), not just analytes -- in an iodide spectrum the reagent tracks alone
# are ~77% of signal and looked "unassigned" before.
_full_led = pd.DataFrame({
    "peak_id": ["p1", "p2", "p3", "p4", "p5"],
    "mz": [172.9106, 253.8094, 173.9139, 126.9072, 401.5],
    "role": ["M0", "reagent", "iso_child", "artifact", "unexplained"],
    "neutral_formula": ["CH2O2", None, None, None, None],
    "adduct": ["[M+I]-", None, None, None, None],
    "ion_formula": ["CH3IO2-", "I2-", None, None, None],
    "iso_label": [None, None, "13C", None, None],
    "parent_peak_id": [None, None, "p1", None, None],
    "commentary": ["Pass 1", "reagent ion: [I2]-. (127I+127I) (-0.3 ppm)",
                   None, "FT ringing/sidelobe of 126.9051", None],
})
_idr = TS.identified_rows(_full_led)
check("identified_rows: one row per identified ion (unexplained excluded)",
      len(_idr) == 4 and set(_idr.role) == {"M0", "reagent", "iso_child", "artifact"},
      _idr.role.tolist())
check("identified_rows: reagent carries ion_formula + isotopologue tag",
      _idr.loc[_idr.role == "reagent", "ion_formula"].iloc[0] == "I2-"
      and _idr.loc[_idr.role == "reagent", "iso_label"].iloc[0] == "127I+127I")
check("identified_rows: iso child inherits the PARENT ion formula",
      _idr.loc[_idr.role == "iso_child", "ion_formula"].iloc[0] == "CH3IO2-"
      and _idr.loc[_idr.role == "iso_child", "iso_label"].iloc[0] == "13C")
check("identified_rows: iso child does NOT carry the neutral (no double-count)",
      pd.isna(_idr.loc[_idr.role == "iso_child", "neutral_formula"].iloc[0]))
check("identified_rows: ledger without role column -> empty",
      len(TS.identified_rows(pd.DataFrame({"mz": [1.0]}))) == 0)

_merged_sf = pd.DataFrame({"mz": [172.9106], "neutral_formula": ["CH2O2"],
                           "adduct": ["[M+I]-"], "tier": ["Assigned"]})
# two files: same reagent line twice (median mz), artifacts 1 mDa apart (one
# track) and 5 mDa away (a second track)
_aux2 = pd.concat([_idr, _idr.assign(mz=_idr.mz + 0.0002)], ignore_index=True)
_aux2 = pd.concat([_aux2, pd.DataFrame([{"mz": 126.9122, "role": "artifact",
                                         "ion_formula": None, "iso_label": None,
                                         "neutral_formula": None, "adduct": None}])],
                  ignore_index=True)
_sf = TS.stamping_frame(_merged_sf, _aux2)
check("stamping_frame: analyte row gains role=M0 + modal per-file ion_formula",
      _sf.loc[_sf.role == "M0", "ion_formula"].iloc[0] == "CH3IO2-")
check("stamping_frame: one aggregated row per (reagent ion, iso tag)",
      (_sf.role == "reagent").sum() == 1)
check("stamping_frame: artifact m/z-gap clustering (1 mDa merges, 5 mDa splits)",
      (_sf.role == "artifact").sum() == 2,
      _sf.loc[_sf.role == "artifact", "mz"].tolist())

# annotate_peaks stamps the identity columns from a union frame
_uts = pd.DataFrame({"sample_item_id": ["s1"] * 3,
                     "mz": [253.8095, 172.9107, 300.0],
                     "height": [9e6, 5e4, 200.0]})
_ust = TS.annotate_peaks(_uts, _sf, tol_ppm=6.0)
check("annotate: reagent track stamped ion_formula+role, NO neutral",
      _ust.loc[0, "ion_formula"] == "I2-" and _ust.loc[0, "role"] == "reagent"
      and pd.isna(_ust.loc[0, "neutral_formula"]))
check("annotate: analyte track carries neutral AND ion_formula",
      _ust.loc[1, "neutral_formula"] == "CH2O2"
      and _ust.loc[1, "ion_formula"] == "CH3IO2-" and _ust.loc[1, "role"] == "M0")
check("annotate: unknown track stays blank", pd.isna(_ust.loc[2, "ion_formula"]))
# backward compat: an analyte-only ledger (no role columns) still emits the
# identity columns, all-<NA>
_bc = TS.annotate_peaks(_uts, _merged_sf[["mz", "neutral_formula", "adduct", "tier"]],
                        tol_ppm=6.0)
check("annotate: legacy ledger -> identity columns exist, empty",
      {"role", "ion_formula", "iso_label"} <= set(_bc.columns)
      and _bc["ion_formula"].isna().all())


def test_all():
    assert FAIL == 0, f"{FAIL} checks failed"


if __name__ == "__main__":
    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)
