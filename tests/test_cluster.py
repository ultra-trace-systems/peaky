"""Offline tests for cluster.py. Run: python3 tests/test_cluster.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from peaky import cluster as CL  # noqa: E402

PASS = FAIL = 0
def check(name, cond, detail=""):
    global PASS, FAIL
    if cond: PASS += 1; print(f"  ok  {name}")
    else: FAIL += 1; print(f"FAIL  {name}  {detail}")


# --- panel_median: holes (zeroed traces) are below-detection LOWS, not dropped ---
_M = np.array([[1000.0] * 20 for _ in range(6)])
_M[0][0] = 300.0                       # a dim detected point -> floor ~300
for _r in range(5):                    # zero-air window t=8..11: 5 of 6 traces zero out
    for _t in range(8, 12):
        _M[_r][_t] = np.nan
_pm = CL.panel_median(_M)
check("panel_median DIPS during a zeroing window (holes counted as floor)",
      _pm[9] < 400 and _pm[2] == 1000, (_pm[9], _pm[2]))
check("panel_median != nanmedian (which would survivorship-bias up to 1000)",
      abs(_pm[9] - np.nanmedian(_M, axis=0)[9]) > 500)


# --- two anti-phase families: signed distance must keep them apart -----------
n = 40
t = np.linspace(0, 1, n)
rise = 10 ** (1 + 2 * t)          # rising
fall = 10 ** (3 - 2 * t)          # falling (anti-phase)
rng_noise = np.cos(np.arange(n))  # deterministic wiggle (no Math.random)
traces = {}
for i in range(4):
    traces[f"R{i}"] = rise * (1 + 0.02 * np.roll(rng_noise, i))
for i in range(4):
    traces[f"F{i}"] = fall * (1 + 0.02 * np.roll(rng_noise, i))
df = pd.DataFrame(traces)
cols = list(df.columns)

Lg, cm = CL.correlate(df, cols)
lab, big = CL.cluster(cm)
check("rising and falling families separate into 2 clusters", len(big) == 2, dict(lab))
rcl = {lab[c] for c in cols if c.startswith("R")}
fcl = {lab[c] for c in cols if c.startswith("F")}
check("all rising share one cluster", len(rcl) == 1, rcl)
check("all falling share one cluster", len(fcl) == 1, fcl)
check("rising and falling are DIFFERENT clusters (anti-phase not merged)",
      rcl.isdisjoint(fcl), (rcl, fcl))

# --- shape_of ----------------------------------------------------------------
check("shape_of rise", CL.shape_of(np.linspace(-2, 2, n)) == "rise")
check("shape_of fall", CL.shape_of(np.linspace(2, -2, n)) == "fall")
check("shape_of peak", CL.shape_of(np.concatenate([np.linspace(-1, 2, n // 2),
                                                   np.linspace(2, -1, n // 2)])) == "peak")

# --- threshold_scan returns sizes --------------------------------------------
ts = CL.threshold_scan(cm)
check("threshold_scan returns a dict keyed by cut", set(ts) >= {0.3, 0.4, 0.5})

# --- graceful on tiny / empty inputs -----------------------------------------
Lg0, cm0 = CL.correlate(df, [])
check("correlate([]) -> empty frames", len(cm0.columns) == 0)
lab0, big0 = CL.cluster(cm0)
check("cluster(empty) -> no clusters, no crash", big0 == [] and len(lab0) == 0)
lab1, big1 = CL.cluster(cm.iloc[:1, :1])     # single item
check("cluster(1 item) -> no >=3 cluster, no crash", big1 == [])
check("threshold_scan(empty) -> {}", CL.threshold_scan(cm0) == {})

# --- cluster_rows ------------------------------------------------------------
grid = t * 1.6
rows, Z = CL.cluster_rows(cols, lab, big, cm, df, grid)
check("cluster_rows builds one row per big cluster", len(rows) == len(big), len(rows))
check("cluster_rows carries (cid, members, rbar, shape, peak_hr)",
      all(len(r) == 5 for r in rows) and all(r[2] > 0.9 for r in rows),
      [(r[0], round(r[2], 2), r[3]) for r in rows])

# --- flatness gate: split varying vs flat, render the flat bunch --------------
# rise/fall traces vary strongly; add two genuinely flat traces (cv ~ 0).
flat_df = df.copy()
flat_df["FLAT0"] = 1000.0 + 0.0 * t                  # dead flat
flat_df["FLAT1"] = 500.0 * (1 + 0.001 * rng_noise)   # ~flat (tiny wiggle)
varying, flat = CL.split_varying(flat_df, list(flat_df.columns))
check("split_varying: rise/fall go to varying", set(cols) <= set(varying), varying)
check("split_varying: flat traces pulled out", set(flat) == {"FLAT0", "FLAT1"}, flat)
check("trace_cv: a varying trace clears the cv gate", CL.trace_cv(flat_df, "R0") >= CL.CHANGING)
check("trace_cv: a flat trace is below the gate", CL.trace_cv(flat_df, "FLAT0") < CL.CHANGING)
check("split_varying: a no-points trace is treated flat",
      "Z" in CL.split_varying(pd.DataFrame({"Z": [np.nan] * n}), ["Z"])[1])

import os, tempfile  # noqa: E402
with tempfile.TemporaryDirectory() as d:
    out = CL.render_flat_panel(flat, flat_df, grid, f"{d}/flat_p1.png", lambda c: str(c),
                               label="flat", title="flat test")
    check("render_flat_panel: writes one PNG", bool(out) and os.path.exists(out) and os.path.getsize(out) > 4000)
    check("render_flat_panel(empty) -> None", CL.render_flat_panel([], flat_df, grid, f"{d}/x.png", str) is None)

# --- merge_similar: collapse near-identical-shape clusters -------------------
T = 20
tt = np.linspace(0, 1, T)
up = 10 ** (1 + 2 * tt); dn = 10 ** (3 - 2 * tt)
mcols = {}
for i in range(3): mcols[f"A{i}"] = up * (1 + 0.01 * np.cos(np.arange(T) + i))   # rising fam 1
for i in range(3): mcols[f"B{i}"] = up * (1 + 0.01 * np.sin(np.arange(T) + i))   # rising fam 2 (~same)
for i in range(3): mcols[f"C{i}"] = dn * (1 + 0.01 * np.cos(np.arange(T) + i))   # falling (distinct)
Lgm = np.log10(pd.DataFrame(mcols))
labm = pd.Series({**{f"A{i}": 1 for i in range(3)}, **{f"B{i}": 2 for i in range(3)},
                  **{f"C{i}": 3 for i in range(3)}})
nlab, nbig = CL.merge_similar(Lgm, labm, [1, 2, 3], merge_r=0.9)
check("merge_similar: the two near-identical clusters merge (3 -> 2)", len(nbig) == 2, nbig)
check("merge_similar: rising fams share a merged id; falling stays apart",
      nlab["A0"] == nlab["B0"] and nlab["A0"] != nlab["C0"])
check("merge_similar: merged ids are disjoint from original singleton ids",
      set(nbig).isdisjoint({1, 2, 3}), nbig)
check("merge_similar: <2 clusters is a no-op", CL.merge_similar(Lgm, labm, [1])[1] == [1])

# --- split_flat_clusters: demote clusters whose family-MEAN doesn't move ------
T2 = 20
t2 = np.linspace(0, 1, T2)
flatfam = {f"P{i}": 1000.0 + 5 * np.cos(np.arange(T2) + i) for i in range(4)}      # ~constant
risefam = {f"Q{i}": 10 ** (1 + 2 * t2) * (1 + 0.01 * np.sin(np.arange(T2) + i)) for i in range(4)}
tf = pd.DataFrame({**flatfam, **risefam})
check("cluster_flatness: a flat family scores ~1 (below the gate)",
      CL.cluster_flatness(list(flatfam), tf) < CL.FLAT_CLUSTER_RANGE,
      CL.cluster_flatness(list(flatfam), tf))
check("cluster_flatness: a rising family clears the gate",
      CL.cluster_flatness(list(risefam), tf) >= CL.FLAT_CLUSTER_RANGE,
      CL.cluster_flatness(list(risefam), tf))
rows_in = [(1, list(flatfam), 0.9, "peak", 0.5), (2, list(risefam), 0.95, "rise", 1.5)]
dyn, flatd = CL.split_flat_clusters(rows_in, tf)
check("split_flat_clusters: rising stays dynamic, flat family demoted",
      [r[0] for r in dyn] == [2] and [r[0] for r in flatd] == [1], (dyn, flatd))

# --- big_changers: surface single channels that change a lot ------------------
gridc = np.linspace(0, 1.6, 20)
spike = np.full(20, 300.0); spike[2:5] = [1500, 4000, 2200]      # ~13x spike
flatc = 1000.0 + 8 * np.cos(np.arange(20))                       # ~flat
tc = pd.DataFrame({"BIG": spike, "DULL": flatc})
ch = CL.big_changers(tc, ["BIG", "DULL"], gridc, fold_min=CL.BIG_CHANGE_FOLD)
check("big_changers: picks the big spiker, not the flat one",
      [c[0] for c in ch] == ["BIG"], ch)
check("big_changers: reports a large fold and a peak hour in the spike window",
      ch and ch[0][1] >= CL.BIG_CHANGE_FOLD and gridc[1] <= ch[0][2] <= gridc[5], ch)
with tempfile.TemporaryDirectory() as d:
    out = CL.render_changers(ch, tc, gridc, f"{d}/ch", lambda c: str(c), title="t")
    check("render_changers: writes A4 page PNG(s) (list of paths)",
          isinstance(out, list) and out and os.path.exists(out[0]) and os.path.getsize(out[0]) > 4000, out)
    check("render_changers([]) -> []", CL.render_changers([], tc, gridc, f"{d}/x", str) == [])

# --- per-cluster workbook (one tab per cluster) ------------------------------
wb_rows = [(1, ["A", "B", "C"], 0.9, "rise", 0.5),
           ("remaining (singletons)", ["D", "E"], float("nan"), "n/a", 0.0)]
wb_meta = {"A": {"neutral_formula": "C6H14O4", "channel": "+H⁺", "match_score": 0.98, "tier": "Assigned"},
           "B": {"neutral_formula": "C6H14O4", "channel": "+Ur⁺", "match_score": 0.91, "tier": "Candidate"}}
with tempfile.TemporaryDirectory() as d:
    out = CL.write_cluster_workbook(wb_rows, f"{d}/wb.xlsx", meta=wb_meta,
                                    item_label=lambda k: f"ion-{k}",
                                    member_cols=["neutral_formula", "channel", "match_score", "tier"])
    check("write_cluster_workbook: file created", bool(out) and os.path.exists(out))
    import openpyxl  # noqa: E402
    names = openpyxl.load_workbook(out).sheetnames
    check("workbook: summary sheet + one sheet per cluster",
          names[0] == "summary" and "c1" in names and len(names) == 3, names)
    s1 = pd.read_excel(out, "c1")
    check("cluster sheet carries member + formula/channel/score/tier",
          "member" in s1.columns and "channel" in s1.columns and "tier" in s1.columns and len(s1) == 3,
          list(s1.columns))
    check("write_cluster_workbook([]) -> None", CL.write_cluster_workbook([], f"{d}/x.xlsx") is None)

    # reproducibility: two workbooks from the same data + SAME run time are byte-
    # identical; built at a DIFFERENT time they differ (the embedded stamp tracks the
    # run, like the report id/cover — it is NOT frozen to a constant).
    import hashlib
    import time
    t1 = 1_700_000_000
    kw = dict(meta=wb_meta, item_label=lambda k: f"ion-{k}",
              member_cols=["neutral_formula", "channel", "match_score", "tier"])
    a = CL.write_cluster_workbook(wb_rows, f"{d}/a.xlsx", when=t1, **kw)
    time.sleep(1.1)   # wall-clock gap: an un-normalised (now()) stamp would differ
    b = CL.write_cluster_workbook(wb_rows, f"{d}/b.xlsx", when=t1, **kw)
    c = CL.write_cluster_workbook(wb_rows, f"{d}/c.xlsx", when=t1 + 3600, **kw)
    H = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
    check("workbook byte-identical for same data + same run time", H(a) == H(b),
          f"{H(a)[:12]} vs {H(b)[:12]}")
    check("workbook differs when the run time differs (stamp not frozen)", H(a) != H(c))

    # render_changers: A4-PORTRAIT page(s) so the section keeps the report's format
    cg = np.linspace(0, 1.6, 40)
    ctr = pd.DataFrame({"C10H16O2|[M+H]+": np.r_[np.full(30, 600.0), np.linspace(600, 2400, 10)],
                        "C9H14O5|[M-H]-": np.r_[np.full(20, 800.0), np.linspace(800, 200, 20)]})
    cpaths = CL.render_changers([("C10H16O2|[M+H]+", 4.0, 1.6), ("C9H14O5|[M-H]-", 4.0, 0.1)],
                                ctr, cg, f"{d}/chg", lambda k: k.split("|")[0],
                                title="Large standalone changes — 2 channels")
    check("render_changers returns a list of page paths",
          isinstance(cpaths, list) and len(cpaths) == 1 and os.path.exists(cpaths[0]), cpaths)
    import matplotlib.image as _mpimg
    cimg = _mpimg.imread(cpaths[0])
    check("render_changers page is A4 portrait (taller than wide)", cimg.shape[0] > cimg.shape[1],
          cimg.shape)
    check("render_changers([]) -> []", CL.render_changers([], ctr, cg, f"{d}/e", str) == [])

# --- diurnal structure gate: amplitude-blind eta2 catches low-amplitude diel waves ---
# 7 days at ~0.5 h resolution
_h = np.linspace(0, 167, 336)
_wiggle = 0.01 * np.cos(13.0 * _h)                       # deterministic noise
_diel = 1000.0 * 10 ** (0.08 * np.sin(2 * np.pi * _h / 24) + _wiggle)   # ±20% daily wave
                                          # (real diel channels: range 1.24-1.40)
_flatn = 1000.0 * 10 ** _wiggle                          # structureless
check("diurnal_eta2 high for a low-amplitude diel wave",
      CL.diurnal_eta2(_diel, _h) > 0.8, CL.diurnal_eta2(_diel, _h))
check("diurnal_eta2 low for a structureless trace",
      CL.diurnal_eta2(_flatn, _h) < 0.2, CL.diurnal_eta2(_flatn, _h))
check("diurnal_eta2 = 0 when the span is < ~2 diel cycles",
      CL.diurnal_eta2(_diel[:60], _h[:60]) == 0.0)

# WEAK diel analyte (Ur+ regime): a real but noisy diel wave scores eta2 ~0.30-0.50
# -- below the old 0.50 gate (so it was BURIED in the flat panel, its wave leaking
# into the flat median) but above the tuned DIURNAL_ETA2, so it now surfaces as
# STRUCTURED. Guards the 2026-07 tune against silently drifting back up.
_hd = np.arange(0, 168, 0.5)
_weak = 1000.0 * 10 ** (0.18 * np.sin(2 * np.pi * _hd / 24)
                        + 0.15 * np.random.default_rng(0).standard_normal(len(_hd)))
_weak_eta2 = CL.diurnal_eta2(_weak, _hd)
check("weak diel analyte lands in the 0.30-0.50 band (missed by the old 0.50 gate)",
      0.30 <= _weak_eta2 < 0.50, _weak_eta2)
check("tuned DIURNAL_ETA2 surfaces the weak diel analyte (structured, not flat)",
      _weak_eta2 >= CL.DIURNAL_ETA2 and CL.DIURNAL_ETA2 < 0.50,
      (_weak_eta2, CL.DIURNAL_ETA2))

_tr = pd.DataFrame({"diel": _diel, "noise": _flatn})
check("diel wave fails the amplitude gates (cv + burst range) without hours",
      not CL.trace_varies(_tr, "diel"))
check("diel wave PROMOTED to varying when hours are passed",
      CL.trace_varies(_tr, "diel", hours=_h))
check("structureless trace stays flat even with hours",
      not CL.trace_varies(_tr, "noise", hours=_h))
_v, _f = CL.split_varying(_tr, ["diel", "noise"], hours=_h)
check("split_varying(hours=) -> diel varying, noise flat", _v == ["diel"] and _f == ["noise"])

# --- split_flat_clusters: structure-aware keep + relaxed structureless-settling demote ---
def _fam(base):
    return pd.DataFrame({f"m{i}": base * (1 + 0.02 * np.roll(_wiggle, i)) for i in range(3)})

_row = lambda mem: (1, mem, 0.9, "peak", 12.0)
_mem = ["m0", "m1", "m2"]
# (a) low-amplitude diel family: amplitude says flat -> kept only with hours
_trd = _fam(_diel)
_dyn, _flt = CL.split_flat_clusters([_row(_mem)], _trd)
check("diel family DEMOTED by the amplitude-only gate (old behavior, no hours)",
      len(_flt) == 1 and not _dyn)
_dyn, _flt = CL.split_flat_clusters([_row(_mem)], _trd, hours=_h)
check("diel family KEPT when hours enable the structure test",
      len(_dyn) == 1 and not _flt, (len(_dyn), len(_flt)))
# (b) fatty-acid signature: startup fall inflates full range; starts at ~0.65 of peak
#     (slips the strict 0.8 guard); flat after settling; NO diel structure
_fa = 500.0 + 1500.0 * np.exp(-_h / 8.0)
_trf = _fam(_fa)
_dyn, _flt = CL.split_flat_clusters([_row(_mem)], _trf)
check("structureless settling family KEPT by old gate (the cluster-280 bug)",
      len(_dyn) == 1 and not _flt)
_dyn, _flt = CL.split_flat_clusters([_row(_mem)], _trf, hours=_h)
check("structureless settling family DEMOTED with the relaxed start-high bar",
      len(_flt) == 1 and not _dyn, (len(_dyn), len(_flt)))
# (c) real early rise-event: starts LOW -> must be kept under both gates
_ev = 200.0 + 1800.0 * np.exp(-((_h - 30.0) ** 2) / 50.0)
_tre = _fam(_ev)
_dyn, _flt = CL.split_flat_clusters([_row(_mem)], _tre, hours=_h)
check("early rise-event family still KEPT (starts low, below even the loose bar)",
      len(_dyn) == 1 and not _flt, (len(_dyn), len(_flt)))
# (d) SHORT batch (< 2 diel cycles): eta2 is unmeasurable (forced 0.0), which must
#     NOT read as "measured structureless" — the loose settling bar must not fire,
#     so a rise-then-settle family with starts_high in [0.5, 0.8) keeps the legacy
#     STRICT bar and stays dynamic.
_h30 = np.linspace(0, 30, 90)
_short = 300.0 + 1700.0 * (np.exp(-((_h30 - 4.0) ** 2) / 18.0) + 0.12)
_trs = pd.DataFrame({f"m{i}": _short * (1 + 0.02 * np.cos(0.7 * i + _h30)) for i in range(3)})
_dyn_nh, _flt_nh = CL.split_flat_clusters([_row(_mem)], _trs)           # legacy (no hours)
_dyn_h, _flt_h = CL.split_flat_clusters([_row(_mem)], _trs, hours=_h30)  # hours, short span
check("short batch: hours= must not change the legacy settling verdict",
      (len(_dyn_nh), len(_flt_nh)) == (len(_dyn_h), len(_flt_h)),
      ((len(_dyn_nh), len(_flt_nh)), (len(_dyn_h), len(_flt_h))))


# --- residual (de-glued) correlation space -----------------------------------
# two channels with INDEPENDENT day-to-day behaviour but the SAME diel wave:
# raw correlation glues them; residual correlation must not.
_days = (_h // 24).astype(int)
_wave = 0.25 * np.sin(2 * np.pi * _h / 24)
# a realistic panel: 12 channels with INDEPENDENT day-to-day patterns + the SAME
# diel wave (the common_mode panel-median needs a population — with a tiny panel a
# 2-member family's own pattern would BE the median and get erased; documented
# trade-off, production panels have hundreds of channels).
_daypat = {i: np.array([np.cos(0.9 * i + 2.1 * d) for d in range(8)])[_days] for i in range(12)}
_panel = {f"p{i}": 1000.0 * 10 ** (_wave + 0.10 * _daypat[i] + 0.01 * np.cos((5 + i) * _h))
          for i in range(12)}
_panel["A"] = 1000.0 * 10 ** (_wave + 0.10 * _daypat[0] + 0.01 * np.cos(31 * _h))  # true partner of p0
_trg = pd.DataFrame(_panel)
_rawr = np.corrcoef(np.log10(_trg["p0"]), np.log10(_trg["p5"]))[0, 1]
check("raw log correlation GLUES two unrelated channels sharing the diel wave",
      _rawr > 0.6, _rawr)
_res, _r2, _cm = CL.decompose(_trg, list(_trg.columns), _h)
_resr = _res["p0"].corr(_res["p5"])
check("residual correlation de-glues them (r < 0.6)", abs(_resr) < 0.6, _resr)
_resr2 = _res["p0"].corr(_res["A"])
check("a REAL co-varying pair (same day-to-day pattern) survives residualization",
      _resr2 > 0.6, _resr2)
_pure = pd.DataFrame({"W": 1000.0 * 10 ** (_wave + 0.005 * np.cos(23 * _h)),
                      "N": _flatn})
# NOTE: the diel anomaly already removes a channel's own daily cycle, so a PURE
# wave carrier is identified by its R2 against the shared mode of a panel that
# still carries day-to-day common structure:
_dayc = np.array([0.1, -0.2, 0.15, -0.1, 0.25, -0.15, 0.2, -0.25])[_days]     # shared day-to-day
_carrier = {f"c{i}": 1000.0 * 10 ** (0.2 * _dayc + 0.01 * np.cos((7 + i) * _h)) for i in range(4)}
_trc = pd.DataFrame(_carrier)
_res2, _r2c, _ = CL.decompose(_trc, list(_trc.columns), _h)
check("shared day-to-day mode carriers have high R2 vs the common mode",
      float(_r2c.min()) > 0.7, dict(_r2c))
check("correlate(log=False) accepts a residual frame",
      CL.correlate(_res, ["p0", "A"], log=False)[1].shape == (2, 2))

# validate_cohesion: a tight family passes, a glued-loose one is flagged
_corr_ok = pd.DataFrame(np.array([[1, .8, .7], [.8, 1, .75], [.7, .75, 1]]),
                        index=["a", "b", "c"], columns=["a", "b", "c"])
_corr_bad = pd.DataFrame(np.array([[1, .2, .1], [.2, 1, .15], [.1, .15, 1]]),
                         index=["a", "b", "c"], columns=["a", "b", "c"])
_coh = CL.validate_cohesion([(1, ["a", "b", "c"], .9, "peak", 1.0)], _corr_ok)
check("validate_cohesion: tight family not flagged", _coh[1] == (_coh[1][0], False) and _coh[1][0] > 0.7)
_coh = CL.validate_cohesion([(2, ["a", "b", "c"], .9, "peak", 1.0)], _corr_bad)
check("validate_cohesion: loose family flagged", _coh[2][1] and _coh[2][0] < 0.5)

# isotope_satellite_of: a bin one 13C step above a brighter parent is a satellite
_kmz = np.array([200.0000, 250.0000])
_kmed = np.array([5000.0, 8000.0])
check("isotope_satellite_of catches the 13C satellite of a brighter parent",
      CL.isotope_satellite_of(201.003355, _kmz, _kmed, 300.0) == 200.0)
check("isotope_satellite_of ignores a BRIGHTER bin (satellites are dimmer)",
      CL.isotope_satellite_of(201.003355, _kmz, _kmed, 9000.0) is None)
check("isotope_satellite_of ignores a non-isotope spacing",
      CL.isotope_satellite_of(201.5, _kmz, _kmed, 300.0) is None)


# --- grouped structured-background pages --------------------------------------
check("formula_class basics",
      [CL.formula_class(f) for f in
       ("C7H8O5", "C6H11NO6", "C9H11ClO5S", "C3HF5", "C9H8OSi", "C10H19NO5S", "?179.0076", "")]
      == ["CHO", "CHON", "halogenated", "F-containing", "Si / siloxane", "CHONS",
          "unassigned", "unassigned"])
import tempfile as _tf, os as _os
with _tf.TemporaryDirectory() as _d:
    _gcols = {"CHO": ["diel", "noise"], "CHON": ["p0", "p1", "p2"], "empty": []}
    _gtr = pd.concat([_tr, _trg[["p0", "p1", "p2"]]], axis=1)
    _paths = CL.render_grouped_flat(_gcols, _gtr, _h, f"{_d}/sb", lambda k: k,
                                    title="Structured background", subtitle="test",
                                    group_note=lambda n, cs: f"{len(cs)} ch")
    check("render_grouped_flat writes paginated pages, skipping empty groups",
          len(_paths) == 1 and _os.path.exists(_paths[0]), _paths)
    import matplotlib.image as _mpimg2
    _img = _mpimg2.imread(_paths[0])
    check("grouped page is A4 portrait", _img.shape[0] > _img.shape[1], _img.shape)
    check("render_grouped_flat([]) -> []",
          CL.render_grouped_flat({}, _gtr, _h, f"{_d}/e", str) == [])


def test_all():
    assert FAIL == 0, f"{FAIL} checks failed"


if __name__ == "__main__":
    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)
