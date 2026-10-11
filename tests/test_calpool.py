"""The tier calibration pooled over a batch (assignment/calpool.py).

A file whose own core cannot calibrate the tier engine (fewer than
tiers.CAL_MIN_N isotope-backed core rows: a small scan window) borrows the
batch's pooled core: distinct carbon-bearing ions off the reagent, isotope-backed
in >= 2 files, the tallest row per ion, >= POOL_MIN_IONS of them spanning the
committed m/z range. This file pins the core, the re-tier on the tier-time
ledger (only rows still holding the uncalibrated tier stage's verdict move),
the window the levels read, and the batch wiring (batch_summary
`tier_calibration`, the per-file stats and rows, the levels assessed).

Synthetic ledgers only. Run: pytest tests/test_calpool.py -q
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from peaky.assignment import assign as A
from peaky.assignment import calpool as CP
from peaky.assignment import evidence as EV
from peaky.assignment import ledger as L
from peaky.assignment import tiers as T
from peaky.assignment.levels import context as CX
from peaky.assignment.passes import PassConfig
from peaky.chem import chemistry as C

ORBI = 120_000.0
#: carbon-bearing CHO neutrals read [M-H]-, spread over m/z ~100-330
NEUTRALS = ["C4H6O3", "C5H8O3", "C5H6O4", "C6H8O4", "C6H10O4", "C7H10O4", "C7H12O4", "C8H12O4",
            "C8H14O4", "C9H14O4", "C10H16O4", "C10H14O5", "C10H16O5", "C11H18O5", "C12H20O5", "C13H22O5",
            "C14H22O6", "C16H26O6"]
AD = "[M-H]-"


def _ion(nf, ad=AD):
    cnt = T._ion_counts(nf, ad)
    return C.format_formula(cnt) + ("-" if ad.endswith("-") else "+")


def _file(sid, rows, *, tier=True):
    """One file's ledger: `rows` = [(neutral, adduct, ppm, height, iso, confidence[, method])]; an `iso`
    row gets its 13C child. Tiered by the file's own tier stage when `tier` (as the run leaves it)."""
    peaks, commits = [], []
    for k, r in enumerate(rows):
        nf, ad, ppm, h, iso, conf = r[:6]
        meth = r[6] if len(r) > 6 else "cheminfo+grid"
        mz = C.ion_mz(nf, ad) * (1.0 + ppm * 1e-6)
        pid = f"{sid}-{k}"
        peaks.append((pid, mz, h))
        if iso:
            peaks.append((pid + "c", mz + 1.0033548, h * 0.08))
        commits.append((pid, nf, ad, ppm, iso, conf, meth))
    led = L.new_ledger(pd.DataFrame(peaks, columns=["peak_id", "mz", "height"]))
    for pid, nf, ad, ppm, iso, conf, meth in commits:
        L.commit_assignment(led, pid, neutral_formula=nf, adduct=ad, ion_formula=_ion(nf, ad), ion_score=0.95,
                            compound_score=0.95, ppm_error=ppm, pass_no=1, method=meth, confidence=conf,
                            commentary=f"Pass 1: {nf} {ad}")
        if iso:
            L.attach_isotopologue(led, pid + "c", pid, iso_label="13C", iso_match_score=0.9)
    if tier:
        T.apply_tiers(led)
    return led


def _batch(n_files=3, n_ions=16, *, iso_files=2, ppm=lambda f, k: 0.05 * ((k % 5) - 2) + 0.01 * f):
    """{sid: ledger}: every file commits the first `n_ions` NEUTRALS; the first `iso_files` files hold
    each with its 13C line."""
    out = {}
    for f in range(n_files):
        rows = [(nf, AD, ppm(f, k), 1e5 * (1 + k) * (1 + f), f < iso_files, "Good")
                for k, nf in enumerate(NEUTRALS[:n_ions])]
        out[f"s{f}"] = _file(f"s{f}", rows)
    return out


# --------------------------------------------------------------------------- the per-file core, unchanged
def test_the_file_core_is_the_shared_row_test_and_estimator():
    """_calibrate reads core_point + robust_centre: 20 isotope-backed rows calibrate a file, 19 do not."""
    rows = [(nf, AD, 0.1 * ((k % 7) - 3), 1e5, True, "High") for k, nf in enumerate(NEUTRALS)]
    rows += [("C17H28O6", AD, 0.2, 1e5, True, "Good"), ("C18H30O6", AD, -0.1, 1e5, True, "Good")]
    led = _file("f", rows, tier=False)
    m0 = led[led["role"] == "M0"]
    kids = led.loc[led["role"] == "iso_child", "parent_peak_id"].value_counts()
    cal = T._calibrate(m0, kids)
    ppms = [p for p in (T.core_point(r, kids) for _, r in m0.iterrows()) if p is not None]
    assert len(ppms) == 20 and cal is not None and T._cal_source(cal) == T.CAL_SOURCE_PER_FILE
    mu, sigma, keep = T.robust_centre([p[0] for p in ppms], T.CAL_MIN_N)
    assert (cal[0], cal[1]) == (mu, sigma) and len(keep) == 20
    assert T._calibrate(m0.iloc[1:], kids) is None


def test_apply_tiers_marks_where_the_tier_calibration_came_from():
    led = _batch(n_files=1)["s0"]
    m0 = led[led["role"] == "M0"]
    assert (m0[T.TIER_CALIBRATION] == T.CAL_SOURCE_NONE).all()
    assert led.loc[led["role"] != "M0", T.TIER_CALIBRATION].isna().all()
    assert m0["tier_reason"].str.contains("degeneracy not measured: file uncalibrated").any()
    rows = [(nf, AD, 0.0, 1e5, True, "High") for nf in NEUTRALS + ["C17H28O6", "C18H30O6"]]
    cal_led = _file("c", rows)
    assert (cal_led.loc[cal_led["role"] == "M0", T.TIER_CALIBRATION] == T.CAL_SOURCE_PER_FILE).all()


# --------------------------------------------------------------------------- the pooled core
def test_the_pool_counts_distinct_ions_isotope_backed_in_two_files():
    pool = CP.pool_calibration(_batch())
    assert pool.cal is not None and pool.reason is None
    assert pool.n_ions == 16 and pool.n_files == 2 and pool.n_files_read == 3
    assert T._cal_source(pool.cal) == T.CAL_SOURCE_POOLED and pool.cal.b is None   # constant model
    # the tallest row per ion (file s1: heights x2): its ppm, one row per ion
    assert set(pool.core["sid"]) == {"s1"} and pool.core["ion"].is_unique
    mu, sigma, _ = T.robust_centre(pool.core["ppm"], CP.POOL_MIN_IONS)
    assert (pool.cal[0], pool.cal[1]) == (mu, sigma)
    rec = pool.record()
    assert rec["used"] is True and rec["n_ions"] == 16 and rec["min_ions"] == CP.POOL_MIN_IONS
    json.dumps(rec)


def test_one_files_isotope_line_is_not_enough():
    pool = CP.pool_calibration(_batch(iso_files=1))
    assert pool.cal is None and pool.n_ions == 0 and pool.n_candidate_ions == 16
    assert "the pooled core needs 15" in pool.reason


def test_too_few_distinct_ions_stay_uncalibrated_and_say_why():
    pool = CP.pool_calibration(_batch(n_ions=14))
    assert pool.cal is None and pool.n_ions == 14
    assert pool.reason.startswith("14 distinct carbon-bearing ions isotope-backed in >= 2 files")
    assert CP.pool_calibration(_batch(n_ions=15)).cal is not None


def test_carbon_free_reagent_identity_and_reagent_mass_rows_stay_out():
    batch = _batch(n_ions=15)
    extra = [("H^NO3", "[M+^NO3]-", 0.0, 5e6, True, "High"),        # a carbon-free reagent cluster line
             ("HNO3", "[M+^NO3]-", 0.0, 5e6, True, "High")]
    for sid in ("s0", "s1"):
        rows = [(nf, AD, 0.0, 1e5, True, "Good") for nf in NEUTRALS[:15]] + extra
        batch[sid] = _file(sid, rows)
    pool = CP.pool_calibration(batch)
    assert pool.n_ions == 15 and not pool.core["ion"].str.contains("N").any()
    # a line on a reagent-library ion mass is the reagent's
    mz0 = C.ion_mz(NEUTRALS[0], AD)
    assert CP.pool_calibration(_batch(n_ions=15), reagent_masses=[mz0]).n_ions == 14


def test_a_reagent_identity_stays_out_on_the_runs_polarity(monkeypatch):
    """The nitrate / urea reagent molecules are carbon-free or carry N2, so C >= 1 and
    N <= 1 already keep them out; the identity test guards any other reagent set."""
    from peaky.assignment.levels import decide as DC
    seen = []

    def ident(n, a, pol):
        seen.append(pol)
        return "reagent" if n == NEUTRALS[0] else ""
    monkeypatch.setattr(DC, "reagent_identity", ident)
    assert CP.pool_calibration(_batch(n_ions=16), polarity="positive").n_ions == 15
    assert set(seen) == {"positive"}


def test_a_core_bunched_in_mz_is_not_used():
    """15 ions bunched at the low end of a batch whose files committed lines up to m/z ~480."""
    rows_far = [("C20H34O8", AD, 0.0, 1e4, False, "Good"), ("C22H36O9", AD, 0.0, 1e4, False, "Good")]
    small = ["C4H6O3", "C5H8O2", "C4H4O4", "C5H6O3", "C5H8O3", "C4H6O4", "C6H8O2", "C6H10O2", "C5H4O4",
             "C6H8O3", "C6H10O3", "C5H6O4", "C7H10O2", "C7H12O2", "C6H6O4"]
    batch = {s: _file(s, [(nf, AD, 0.0, 1e5, True, "Good") for nf in small] + rows_far) for s in ("a", "b")}
    pool = CP.pool_calibration(batch)
    assert pool.cal is None and pool.n_ions == 15 and "span" in pool.reason


# --------------------------------------------------------------------------- the tier texts and the transfer
def test_the_reasons_name_the_pooled_window():
    led = _batch(n_files=1)["s0"]
    pool = CP.pool_calibration(_batch())
    led.loc[led["role"] == "M0", "degeneracy_density"] = 1.0
    t = T.compute_tiers(led, cal=pool.cal).set_index("peak_id")
    assert t["tier_reason"].str.startswith("unique formula in the batch-pooled calibrated window").any()
    t0 = T.compute_tiers(led, cal=None)
    assert not t0["tier_reason"].str.contains("batch-pooled").any()
    # the file's own calibration keeps its words
    t1 = T.compute_tiers(led, cal=T._Cal(pool.cal[0], pool.cal[1]))
    assert t1["tier_reason"].str.startswith("unique formula in the calibrated window").any()


def test_retier_moves_only_rows_still_holding_the_tier_stages_verdict():
    led = pd.DataFrame({"peak_id": ["a", "b", "c", "d"], "role": ["M0"] * 4,
                        "tier": ["Assigned", "Candidate", "Assigned", "Candidate"],
                        "tier_reason": ["no rival", "a later re-read", "no rival", "no rival"]})
    base = pd.DataFrame({"peak_id": ["a", "b", "c", "d"], "tier": ["Assigned"] * 4, "tier_reason": ["no rival"] * 4,
                         "candidate_density": [1] * 4, "density_capped": [False] * 4})
    new = base.assign(tier=["Candidate", "Candidate", "Assigned", "Assigned"],
                      tier_reason=["mass error off", "mass error off", "unique in the pooled window",
                                   "unique in the pooled window"])
    out = T.retier_calibrated(led, base, new)
    # a: re-tiered; b: a later stage rewrote it, kept; c: reworded; d: a later stage only demoted it --
    # its tier stays, the tier stage's words are the calibrated ones
    assert list(led["tier"]) == ["Candidate", "Candidate", "Assigned", "Candidate"]
    assert list(led["tier_reason"]) == ["mass error off", "a later re-read", "unique in the pooled window",
                                        "unique in the pooled window"]
    assert out == {"rows": 4, "retiered": 3, "demoted": 1, "promoted": 0, "reworded": 2, "kept": 1}
    assert (led[T.TIER_CALIBRATION] == T.CAL_SOURCE_POOLED).all()


# --------------------------------------------------------------------------- one file, re-applied
def _inputs_for(at, sid, *, levels=True):
    ri = (EV.file_run_inputs(sample_id=sid, reagent="NO3", context="ambient-air", resolution=ORBI,
                             height_gate_cps=100.0, degeneracy_cal=None) if levels else None)
    return CP.PoolInputs(at_tiers=at, cfg=None, profile=None, adducts=[AD], families=(),
                         curated=frozenset(), level_inputs=ri)


def test_apply_pool_re_tiers_the_file_on_its_tier_time_ledger():
    pool = CP.pool_calibration(_batch())
    assert pool.cal is not None
    rows = [(nf, AD, 0.0, 1e5, False, "Good") for nf in NEUTRALS[:6]]
    rows[0] = (NEUTRALS[0], AD, 3.0, 1e5, False, "Good")     # off the pooled calibration, uncorroborated
    rows[1] = (NEUTRALS[1], AD, 3.0, 1e5, True, "Good")      # off it, but its 13C line corroborates
    at = _file("u", rows, tier=False)
    led = at.copy(deep=True)
    T.apply_tiers(led)                                        # the file's own (uncalibrated) tier stage
    i2 = led.index[led["peak_id"] == "u-2"][0]                # a later stage rewrote this row
    led.at[i2, "tier"], led.at[i2, "tier_reason"] = "Candidate", "a later stage's re-read"
    i3 = led.index[led["peak_id"] == "u-3"][0]                # ... and only demoted this one
    led.at[i3, "tier"] = "Candidate"
    lid = led.index[led["peak_id"] == "u-0"][0]
    assert led.at[lid, "tier"] == "Assigned"
    out = CP.apply_pool(led, pool, _inputs_for(at, "u"), audit=[], log=lambda *a: None)
    by = led.set_index("peak_id")
    assert by.at["u-0", "tier"] == "Candidate" and "batch-pooled instrument accuracy" in by.at["u-0", "tier_reason"]
    assert by.at["u-1", "tier"] == "Assigned" and "batch-pooled calibrated window" in by.at["u-1", "tier_reason"]
    assert by.at["u-2", "tier_reason"] == "a later stage's re-read"
    assert by.at["u-3", "tier"] == "Candidate" and "batch-pooled calibrated window" in by.at["u-3", "tier_reason"]
    assert out["tiers"]["demoted"] == 1 and out["tiers"]["kept"] == 1
    m0 = led[led["role"] == "M0"]
    assert (m0[T.TIER_CALIBRATION] == T.CAL_SOURCE_POOLED).all()
    assert pd.to_numeric(m0["degeneracy_density"], errors="coerce").notna().all()
    # the file's own level, in the pooled window: no longer the no-window text
    assert out["levels"]["n_levelled"] == len(m0)
    assert not (m0["evidence"] == EV.NO_WINDOW_TEXT).any()
    # the tier-time ledger is read, never changed
    assert "tier" not in at.columns or at["tier"].isna().all()


def test_the_lent_window_is_the_pooled_width_about_the_files_own_centre():
    """A file 0.6 ppm off the pooled centre (4 pooled sigma): centred on the pool its
    uncorroborated rows would all fall past the z-tail; centred on its own Assigned rows
    none does. With fewer than CENTRE_MIN_ROWS Assigned rows the pooled mu stands in."""
    pool = CP.pool_calibration(_batch())
    assert pool.cal is not None and pool.cal[1] == pytest.approx(T.CAL_SIGMA_FLOOR)
    cal, centre = CP.lent_calibration(pool, [0.6] * CP.CENTRE_MIN_ROWS)
    assert centre == "file" and cal[0] == pytest.approx(0.6) and cal[1] == pool.cal[1]
    assert T._cal_source(cal) == T.CAL_SOURCE_POOLED
    cal, centre = CP.lent_calibration(pool, [0.6] * (CP.CENTRE_MIN_ROWS - 1))
    assert centre == "pool" and cal[0] == pool.cal[0]
    assert CP.lent_calibration(CP.Pool(), [0.6] * 20) == (None, None)
    rows = [(nf, AD, 0.6 + 0.02 * ((k % 3) - 1), 1e5, False, "Good") for k, nf in enumerate(NEUTRALS[:16])]
    at = _file("o", rows, tier=False)
    led = at.copy(deep=True)
    T.apply_tiers(led)
    assert (led.loc[led["role"] == "M0", "tier"] == "Assigned").all()
    out = CP.apply_pool(led, pool, _inputs_for(at, "o", levels=False), audit=[], log=lambda *a: None)
    assert out["cal"]["centre"] == "file" and out["cal"]["mu"] == pytest.approx(0.6)
    m0 = led[led["role"] == "M0"]
    assert (m0["tier"] == "Assigned").all() and not m0["tier_reason"].str.contains("σ off").any()
    # the same file centred on the pool: every row is past the z-tail
    t = T.compute_tiers(at, cal=pool.cal)
    assert (t["tier"] == "Candidate").all() and t["tier_reason"].str.contains("σ off the batch-pooled").all()


def test_the_o11_flag_falls_only_on_rows_still_committing_their_tier_time_reading(monkeypatch):
    """The O >= 11 below-assignability flag is the tier stage's: a row a later stage re-read
    (a new reading on the peak) is not flagged off the stamp its tier-time reading got."""
    pool = CP.pool_calibration(_batch())
    rows = [("C10H16O11", AD, 0.0, 1e5, False, "Good"), ("C11H18O11", AD, 0.0, 1e5, False, "Good")]
    at = _file("x", rows, tier=False)
    led = at.copy(deep=True)
    T.apply_tiers(led)
    i1 = led.index[led["peak_id"] == "x-1"][0]
    led.at[i1, "neutral_formula"] = "C11H16O12"                 # a later re-read of that peak (O >= 11 too)
    monkeypatch.setattr(T, "_saturated", lambda r: True)
    out = CP.apply_pool(led, pool, _inputs_for(at, "x", levels=False), audit=[], log=lambda *a: None)
    by = led.set_index("peak_id")
    assert out["below"] == 1 and bool(by.at["x-0", "below_assignability"]) is True
    assert bool(by.at["x-1", "below_assignability"]) is False
    assert "below-assignability" in by.at["x-0", "tier_reason"]
    assert "below-assignability" not in str(by.at["x-1", "tier_reason"])


def test_apply_pool_does_nothing_without_a_calibration_or_a_tier_time_ledger():
    led = _batch(n_files=1)["s0"]
    before = led.copy(deep=True)
    assert CP.apply_pool(led, CP.Pool(), _inputs_for(led, "s0")) == {}
    pool = CP.pool_calibration(_batch())
    assert CP.apply_pool(led, pool, CP.PoolInputs(at_tiers=None)) == {}
    pd.testing.assert_frame_equal(led, before)


# --------------------------------------------------------------------------- the levels' window
def test_a_pooled_window_is_the_files_own_and_the_run_window_only_when_no_file_calibrated():
    win = {"a": {"fit": (0.10, 0.20)}, "b": {"fit": (0.30, 0.40)}, "p": {"fit": (0.0, 0.9), "pooled": True}}
    windows, run = CX.window_table(win)
    assert windows["p"] == (0.0, 0.9) and run == (0.2, pytest.approx(0.3))
    windows, run = CX.window_table({"p": {"fit": (0.0, 0.9), "pooled": True}, "q": {"fit": (0.0, 0.9), "pooled": True}})
    assert run == (0.0, 0.9)
    # without a pooled file the table is what it was
    assert CX.window_table({"a": {"fit": (0.1, 0.2)}, "u": {"mu_stamp": 0.05}}) == (
        {"a": (0.1, 0.2), "u": (0.05, 0.2)}, (0.1, 0.2))


# --------------------------------------------------------------------------- the batch
def _batch_table(sids, mzs, height=500.0):
    t0 = pd.Timestamp("2025-10-01 21:00:00", tz="UTC")
    return pd.DataFrame([dict(sample_item_id=s, sample_item_name=f"n_{s}", datetime_utc=t0 + pd.Timedelta(minutes=10 * i),
                              mz=float(mz), height=height) for i, s in enumerate(sids) for mz in mzs])


def _run_batch(tmp_path, monkeypatch, *, iso_files=8, lines=None, calibrated=False):
    from peaky.batch import assign_batch as AB
    from peaky.io import io_mascope as IO
    sids = [f"f{i}" for i in range(4)]
    pk = _batch_table(sids, list(range(100, 140)))
    k_of = {}

    def fake_assign(sid, context="ambient-air", **kw):
        k = k_of.setdefault(sid, len(k_of))
        rows = [(nf, AD, 0.05 * ((j % 5) - 2), 1e5 * (1 + j), k < iso_files, "Good")
                for j, nf in enumerate(NEUTRALS[:16])]
        rows.append(("C17H28O6", AD, 2.5, 1e5, False, "Good"))     # off-calibration, uncorroborated
        at = _file(sid, rows, tier=False)
        led = at.copy(deep=True)
        T.apply_tiers(led)
        ri = EV.file_run_inputs(sample_id=sid, reagent="Br", context="ambient-air", resolution=ORBI,
                                height_gate_cps=10.0, degeneracy_cal=None)
        EV.apply_levels(led, run_inputs=ri)
        if calibrated:
            return {"ledger": led, "plausibility_audit": [], "summaries": {}, "problems": [],
                    "stats": {"noise_edge_cps": 4.0, "height_gate_cps": 10.0,
                              "degeneracy_cal": {"mu": 0.0, "sigma": 0.3},
                              "tier_calibration": T.CAL_SOURCE_PER_FILE},
                    "pool_inputs": None}
        return {"ledger": led, "plausibility_audit": [], "summaries": {}, "problems": [],
                "stats": {"noise_edge_cps": 4.0, "height_gate_cps": 10.0, "degeneracy_cal": None,
                          "tier_calibration": T.CAL_SOURCE_NONE},
                "pool_inputs": CP.PoolInputs(at_tiers=at, adducts=[AD], level_inputs=ri)}

    monkeypatch.setattr(IO, "connect", lambda *a, **k: "CLIENT")
    monkeypatch.setattr(IO, "fetch_peaks", lambda client, sid, use_cache=True: pd.DataFrame(
        {"peak_id": ["A"], "mz": [200.1], "height": [1.0e5]}))
    monkeypatch.setattr(IO, "estimate_offset", lambda raw: 0.0)
    monkeypatch.setattr(A, "run", fake_assign)
    return AB.run(peaks=pk, ts_peaks=pk, reagent="Br", batch="test batch", out_dir=str(tmp_path),
                  k_min=3, k_max=4, min_gain=0.0, n_jobs=1, resolving_power=ORBI,
                  log=(lines.append if lines is not None else (lambda *a: None)))


def test_a_batch_of_uncalibrated_files_is_tiered_and_levelled_on_its_pooled_core(tmp_path, monkeypatch):
    lines = []
    _run_batch(tmp_path, monkeypatch, lines=lines)
    summ = json.load(open(tmp_path / "batch_summary.json"))
    tc = summ["tier_calibration"]
    n = len(summ["per_file"])
    assert n >= 2 and tc["files"] == {"per_file": 0, "pooled": n, "none": 0}
    pool = tc["pool"]
    assert pool["used"] is True and pool["n_ions"] == 16 and pool["reason"] is None
    assert set(pool["retiered_files"]) == {p["sample_id"] for p in summ["per_file"]}
    for p in summ["per_file"]:
        assert p["tier_calibration"] == "pooled"
        assert p["degeneracy_cal"] == {"mu": pool["mu"], "sigma": pool["sigma"]} or (
            abs(p["degeneracy_cal"]["mu"] - pool["mu"]) < 1e-4 and abs(p["degeneracy_cal"]["sigma"] - pool["sigma"]) < 1e-4)
        assert p["tier_pool"]["demoted"] == 1 and p["by_tier"].get("Candidate", 0) >= 1
        assert p["tier_pool"]["centre"] == "file"                 # 17 Assigned rows: the file's own centre
    assert tc["source"] == "pooled"
    assert summ["levels_not_assessed_reason"] is None
    assert sum(summ["evidence_levels"]["pooled"].values()) > 0
    led = pd.read_csv(tmp_path / "per_file" / f"{summ['per_file'][0]['sample_id']}_ledger.csv")
    m0 = led[led["role"] == "M0"]
    assert (m0[T.TIER_CALIBRATION] == "pooled").all()
    off = m0[m0["neutral_formula"] == "C17H28O6"]
    assert (off["tier"] == "Candidate").all() and off["tier_reason"].str.contains("batch-pooled").all()
    assert any(ln.startswith("[calpool]") and "pooled tier calibration" in ln for ln in lines)
    # the merged rows say where the tier they carry was calibrated
    merged = pd.read_csv(tmp_path / "merged_ledger.csv")
    assert T.TIER_CALIBRATION in merged.columns and (merged[T.TIER_CALIBRATION] == "pooled").all()


def test_a_batch_whose_pool_falls_short_stays_unlevelled_and_says_why(tmp_path, monkeypatch):
    _run_batch(tmp_path, monkeypatch, iso_files=1)
    summ = json.load(open(tmp_path / "batch_summary.json"))
    tc = summ["tier_calibration"]
    assert tc["files"]["pooled"] == 0 and tc["files"]["none"] == len(summ["per_file"]) and tc["source"] == "none"
    assert tc["pool"]["used"] is False and "the pooled core needs 15" in tc["pool"]["reason"]
    why = summ["levels_not_assessed_reason"]
    assert why.startswith("Evidence levels were not assessed")
    assert "A core pooled over the batch's files was tried and not used" in why
    assert all(p["degeneracy_cal"] is None for p in summ["per_file"])


def test_a_calibrated_batch_builds_no_pool(tmp_path, monkeypatch):
    """Every file calibrated alone: no pool is built, nothing is re-tiered."""
    lines = []
    _run_batch(tmp_path, monkeypatch, calibrated=True, lines=lines)
    summ = json.load(open(tmp_path / "batch_summary.json"))
    assert summ["tier_calibration"] == {"source": "per_file",
                                        "files": {"per_file": len(summ["per_file"]), "pooled": 0, "none": 0},
                                        "pool": None}
    assert not any(ln.startswith("[calpool]") for ln in lines)
    assert all("tier_pool" not in p for p in summ["per_file"])


# --------------------------------------------------------------------------- the per-file run's side
def _run_one(monkeypatch, rows, sid):
    """assign.run on a synthetic peak table with the real degeneracy, tier and evidence stages after a
    stand-in pass that commits `rows` (as in _file)."""
    import pickle
    from peaky.io import io_mascope as IO
    led0 = _file(sid, rows, tier=False)
    peaks = led0[["peak_id", "mz", "height"]].copy()

    def commit(st):
        for c in led0.columns:
            if c not in st.led.columns:
                st.led[c] = pd.Series(pd.NA, index=st.led.index, dtype="object")
        by = led0.set_index("peak_id")
        for i in st.led.index:
            pid = st.led.at[i, "peak_id"]
            for c in ("role", "neutral_formula", "adduct", "ion_formula", "ion_score", "compound_score",
                      "ppm_error", "pass_no", "method", "confidence", "commentary", "parent_peak_id",
                      "iso_label", "isotopologues"):
                st.led.at[i, c] = by.at[pid, c]
        return {}

    stages = [A._Stage("pass1", commit, safe=False)] + [s for s in A._STAGES if s.name in ("degeneracy", "tiers",
                                                                                            "evidence")]
    monkeypatch.setattr(A, "_STAGES", stages)
    try:
        res = A.run(sid, adducts=[AD], peaks=peaks, log=lambda *a: None)
    finally:
        IO.unregister_offline_sample(sid)
    return res, pickle.loads(pickle.dumps(res.get("pool_inputs")))


def test_an_uncalibrated_run_returns_its_pool_inputs_and_a_calibrated_one_none(monkeypatch):
    rows = [(nf, AD, 0.0, 1e5, True, "Good") for nf in NEUTRALS[:6]]
    res, pin = _run_one(monkeypatch, rows, "pool-u")
    assert res["stats"]["degeneracy_cal"] is None and res["stats"]["tier_calibration"] == T.CAL_SOURCE_NONE
    assert isinstance(pin, CP.PoolInputs) and pin.adducts == [AD] and pin.level_inputs is not None
    at = pin.at_tiers
    assert set(at.loc[at["role"] == "M0", "neutral_formula"]) == set(NEUTRALS[:6])
    assert "tier" not in at.columns or at["tier"].isna().all()          # as the degeneracy stage found it
    assert (res["ledger"].loc[res["ledger"]["role"] == "M0", T.TIER_CALIBRATION] == T.CAL_SOURCE_NONE).all()
    rows = [(nf, AD, 0.0, 1e5, True, "High") for nf in NEUTRALS + ["C17H28O6", "C18H30O6"]]
    res, pin = _run_one(monkeypatch, rows, "pool-c")
    assert res["stats"]["degeneracy_cal"] is not None and pin is None
    assert res["stats"]["tier_calibration"] == T.CAL_SOURCE_PER_FILE


def test_the_batch_word_is_the_weakest_files_source():
    from peaky.batch import assign_batch as AB
    assert AB._tier_cal_source([{"tier_calibration": "per_file"}] * 3) == "per_file"
    assert AB._tier_cal_source([{"tier_calibration": "per_file"}, {"tier_calibration": "pooled"}]) == "pooled"
    assert AB._tier_cal_source([{"tier_calibration": "pooled"}, {"tier_calibration": "none"}]) == "none"
    assert AB._tier_cal_source([{}, {"stage": "cover"}]) is None
