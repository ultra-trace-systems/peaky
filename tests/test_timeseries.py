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
      all(c in _ann.columns for c in ("neutral_formula", "adduct", "tier", "ion_mz")))
check("annotate: exact match -> formula+channel",
      _ann.loc[0, "neutral_formula"] == "C10H16O3" and _ann.loc[0, "adduct"] == "[M+NH4]+")
check("annotate: within-tol match kept",
      _ann.loc[1, "neutral_formula"] == "C10H16O3")
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


def test_all():
    assert FAIL == 0, f"{FAIL} checks failed"


if __name__ == "__main__":
    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)
