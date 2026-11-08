"""The tier calibration pooled over a batch (assignment/calpool.py).

A file whose own core cannot calibrate the tier engine (fewer than
tiers.CAL_MIN_N isotope-backed core rows: a small scan window) borrows the
batch's pooled core: distinct carbon-bearing ions off the reagent, isotope-backed
in >= 2 files, the tallest row per ion, >= POOL_MIN_IONS of them spanning the
committed m/z range. This file pins the core, the re-tier on the tier-time
ledger (only rows still holding the uncalibrated tier stage's verdict move),
the window the levels read, and the batch wiring (batch_summary
`tier_calibration`, the per-file stats and rows, the levels assessed, the
pool's polarity / reagent masses / instrument class, a later stage's files,
the keep flag serial and through the workers).

Synthetic ledgers only. Run: pytest tests/test_calpool.py -q
"""

from __future__ import annotations

import gc
import json
import weakref

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


def test_the_core_estimator_keeps_the_errors_within_2_ppm_of_their_median():
    """robust_centre (the file's core and the pooled one): an error 1.5 ppm off the median is in the
    core, one 2.5 ppm off is not -- unless fewer than `min_n` errors are within, then all are."""
    ppms = [0.02 * ((k % 5) - 2) for k in range(18)] + [1.5, 2.5]
    mu, sigma, keep = T.robust_centre(ppms, 15)
    assert 18 in keep and 19 not in keep and len(keep) == 19
    assert mu == pytest.approx(float(np.median(ppms[:19])))
    assert T.robust_centre(ppms, 20)[2] == list(range(20))


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
    # carbon-free readings that are NO reagent identity (a nitrite cluster, an iodic acid line): core rows
    # of the per-file test, isotope-backed in both files -- only the pool's C >= 1 keeps them out
    from peaky.assignment.levels.decide import reagent_identity
    cfree = [("HNO2", "[M+^NO3]-", 0.0, 5e6, True, "High"), ("HIO3", AD, 0.0, 5e6, True, "High")]
    assert not any(reagent_identity(nf, ad, "negative") for nf, ad, *_ in cfree)
    for sid in ("s0", "s1"):
        batch[sid] = _file(sid, [(nf, AD, 0.0, 1e5, True, "Good") for nf in NEUTRALS[:15]] + extra + cfree)
    led = batch["s0"]
    kids = led.loc[led["role"] == "iso_child", "parent_peak_id"].value_counts()
    m0 = led[led["role"] == "M0"].set_index("neutral_formula", drop=False)
    assert all(T.core_point(m0.loc[nf], kids) is not None for nf, *_ in cfree)
    assert not CP.core_rows(led, sid="s0")["neutral_formula"].isin(["HNO2", "HIO3"]).any()
    pool = CP.pool_calibration(batch)
    assert pool.n_ions == 15 and pool.n_candidate_ions == 15
    assert not pool.core["neutral_formula"].isin(["HNO2", "HIO3"]).any()
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


def test_a_tof_batch_is_counted_and_not_lent_a_pool():
    """A TOF file's own calibration fits a mass trend; the pooled constant model is not lent there."""
    pool = CP.pool_calibration(_batch(), klass="tof")
    assert pool.cal is None and pool.n_ions == 16 and "TOF" in pool.reason
    assert pool.record()["used"] is False and pool.record()["lent_sigma"] is None
    assert CP.pool_calibration(_batch(), klass="orbitrap").cal is not None


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
    led = pd.DataFrame({"peak_id": ["a", "b", "c", "d", "e"], "role": ["M0"] * 5,
                        "tier": ["Assigned", "Candidate", "Assigned", "Candidate", "Assigned"],
                        "tier_reason": ["no rival", "a later re-read", "no rival", "no rival", "a later rescue"],
                        T.TIER_CALIBRATION: [T.CAL_SOURCE_NONE] * 4 + [pd.NA]})
    base = pd.DataFrame({"peak_id": ["a", "b", "c", "d"], "tier": ["Assigned"] * 4, "tier_reason": ["no rival"] * 4,
                         "candidate_density": [1] * 4, "density_capped": [False] * 4})
    new = base.assign(tier=["Candidate", "Candidate", "Assigned", "Assigned"],
                      tier_reason=["mass error off", "mass error off", "unique in the pooled window",
                                   "unique in the pooled window"])
    out = T.retier_calibrated(led, base, new)
    # a: re-tiered; b: a later stage rewrote it, kept; c: reworded; d: a later stage only demoted it --
    # its tier stays, the tier stage's words are the calibrated ones; e: committed after the tier
    # stage (not in its frames), kept
    assert list(led["tier"]) == ["Candidate", "Candidate", "Assigned", "Candidate", "Assigned"]
    assert list(led["tier_reason"]) == ["mass error off", "a later re-read", "unique in the pooled window",
                                        "unique in the pooled window", "a later rescue"]
    assert out == {"rows": 5, "retiered": 3, "demoted": 1, "promoted": 0, "reworded": 2, "kept": 2}
    # the source names the calibration the row's verdict was given in: the re-judged rows the pooled
    # one; a kept row its own ('none', the uncalibrated stage's), a later commit none (NA, as unpooled)
    assert list(led[T.TIER_CALIBRATION].fillna("NA")) == ["pooled", "none", "pooled", "pooled", "NA"]


# --------------------------------------------------------------------------- one file, re-applied
def _inputs_for(at, sid, *, levels=True, guard=None):
    ri = (EV.file_run_inputs(sample_id=sid, reagent="NO3", context="ambient-air", resolution=ORBI,
                             height_gate_cps=100.0, degeneracy_cal=None) if levels else None)
    return CP.PoolInputs(at_tiers=at, cfg=None, profile=None, adducts=[AD], families=(),
                         curated=frozenset(), level_inputs=ri, isotope_guard=guard)


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
    src = m0.set_index("peak_id")[T.TIER_CALIBRATION]
    assert src["u-2"] == T.CAL_SOURCE_NONE and (src.drop("u-2") == T.CAL_SOURCE_POOLED).all()
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
    # files of the batch that calibrated alone: their median sigma is lent, their median mu the fallback centre
    pool.run_window = (-0.2, 0.31, 3)
    cal, centre = CP.lent_calibration(pool, [0.6] * CP.CENTRE_MIN_ROWS)
    assert centre == "file" and cal[0] == pytest.approx(0.6) and cal[1] == pytest.approx(0.31)
    cal, centre = CP.lent_calibration(pool, [])
    assert centre == "run" and cal[0] == pytest.approx(-0.2) and cal[1] == pytest.approx(0.31)
    rec = pool.record()
    assert rec["lent_width"] == "run" and rec["lent_sigma"] == pytest.approx(0.31) and rec["n_files_calibrated"] == 3
    assert rec["sigma"] == pytest.approx(T.CAL_SIGMA_FLOOR)            # the pool's own fit is still recorded
    pool.run_window = None
    assert pool.record()["lent_width"] == "pool" and pool.record()["n_files_calibrated"] == 0
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


def test_the_skeleton_cap_record_is_taken_on_the_re_run_tier_stage(monkeypatch):
    """stats nox_skeleton_gate is the tier stage's record (assign._stage_tiers): the pooled re-run's
    verdicts on the tier-time ledger, not the finished ledger a later stage rewrote."""
    seen = {}

    def spy(led, profile):
        seen.update(zip(led["peak_id"], led["tier_reason"]))
        return {"capped": 0, "noted": 1}
    monkeypatch.setattr(T, "nox_skeleton_summary", spy)
    pool = CP.pool_calibration(_batch())
    at = _file("k", [(nf, AD, 0.0, 1e5, False, "Good") for nf in NEUTRALS[:3]], tier=False)
    led = at.copy(deep=True)
    T.apply_tiers(led)
    i0 = led.index[led["peak_id"] == "k-0"][0]
    led.at[i0, "tier_reason"] = "a later stage's re-read"
    out = CP.apply_pool(led, pool, _inputs_for(at, "k", levels=False), audit=[], log=lambda *a: None)
    assert out["nox_skeleton_gate"] == {"capped": 0, "noted": 1}
    assert "batch-pooled" in seen["k-0"] and seen["k-0"] != "a later stage's re-read"


def test_clear_and_displace_reset_the_rows_tier_calibration():
    """A row that stops being an M0 (cleared, displaced to an isotopologue) loses its tier's source,
    as it loses its tier; a ledger that never had the column does not get one."""
    led = _batch(n_files=1)["s0"]
    L.clear_assignment(led, "s0-0", reason="a test")
    L.displace_to_isotopologue(led, "s0-1", "s0-2", iso_label="13C")
    by = led.set_index("peak_id")
    assert by.loc[["s0-0", "s0-1"], T.TIER_CALIBRATION].isna().all()
    assert by.at["s0-2", T.TIER_CALIBRATION] == T.CAL_SOURCE_NONE
    bare = _file("z", [(NEUTRALS[0], AD, 0.0, 1e5, False, "Good")], tier=False)
    L.clear_assignment(bare, "z-0", reason="a test")
    assert T.TIER_CALIBRATION not in bare.columns


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


def test_run_windows_marks_a_pooled_file_off_its_stats(monkeypatch):
    """A file whose batch_summary stats say `tier_calibration` 'pooled' is marked: its lent window is
    its own, and the run window is the file that calibrated alone."""
    monkeypatch.setattr(CX, "fit_windows", lambda *a, **k: pytest.fail("the fallback ran"))
    led = _file("w", [(nf, AD, 0.0, 1e5, False, "Good") for nf in NEUTRALS[:3]])
    summary = {"reagent": "NO3", "context": "ambient-air", "reflists_active": [],
               "per_file": [{"sample_id": "a", "degeneracy_cal": {"mu": 0.1, "sigma": 0.2},
                             "tier_calibration": T.CAL_SOURCE_PER_FILE},
                            {"sample_id": "p", "degeneracy_cal": {"mu": 0.0, "sigma": 0.9},
                             "tier_calibration": T.CAL_SOURCE_POOLED}]}
    win = CX.run_windows(summary, {"a": led, "p": led})
    assert win["p"].get("pooled") is True and "pooled" not in win["a"]
    windows, run = CX.window_table(win)
    assert run == (0.1, 0.2) and windows["p"] == (0.0, 0.9)


# --------------------------------------------------------------------------- the batch
def _batch_table(sids, mzs, height=500.0):
    t0 = pd.Timestamp("2025-10-01 21:00:00", tz="UTC")
    return pd.DataFrame([dict(sample_item_id=s, sample_item_name=f"n_{s}", datetime_utc=t0 + pd.Timedelta(minutes=10 * i),
                              mz=float(mz), height=height) for i, s in enumerate(sids) for mz in mzs])


def _staged_table():
    """A batch with two per-file stages: cover files a0-a2 / b0-b2 (an exclusive 20-bin block each pair, on a
    shared 20-bin background, all at 500 cps) and z1 / z2, which the cover never picks, each bright (10x its
    edge) on a bin no cover file holds. With k_max=2 the cover assigns a0, a1, the residual stage z1, z2."""
    bg = {m: 500.0 for m in range(100, 120)}
    spec = {}
    for i in range(3):
        blk = {m: 500.0 for m in range(200 + 20 * i, 220 + 20 * i)}
        spec[f"a{i}"] = {**bg, **blk}
        spec[f"b{i}"] = {**bg, **blk}
    spec["z1"] = {**bg, 500: 5000.0, 501: 600.0, 502: 1000.0}
    spec["z2"] = {**bg, 500: 1000.0, 501: 600.0, 502: 5000.0}
    t0 = pd.Timestamp("2025-10-01 21:00:00", tz="UTC")
    return pd.DataFrame([dict(sample_item_id=s, sample_item_name=f"n_{s}", datetime_utc=t0 + pd.Timedelta(minutes=10 * i),
                              mz=float(mz), height=float(h)) for i, (s, mzh) in enumerate(spec.items())
                         for mz, h in mzh.items()])


class _InProcessPool:
    """A ProcessPoolExecutor stand-in that runs the batch's real worker entry points (`_worker_init`,
    `_assign_one`) in this process, so the stubbed assign.run is the one they call."""

    def __init__(self, max_workers=None, mp_context=None, initializer=None, initargs=()):
        initializer(*initargs)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def submit(self, fn, *args):
        import concurrent.futures as cf
        f = cf.Future()
        f.set_result(fn(*args))
        return f


def _run_batch(tmp_path, monkeypatch, *, iso_files=8, lines=None, calibrated=False, refs=None,
               staged=False, n_jobs=1, keeps=None, skeleton=None, reagent="Br", guard_record=False):
    """`calibrated`: True = every file calibrates alone; a set = the files (in the order they are
    assigned) that do; False = none. `refs`: a list the uncalibrated files' PoolInputs are weakly
    referenced into. `staged`: the two-stage batch (_staged_table: cover a0, a1; residual z1, z2), else
    four files the cover assigns at once. `n_jobs` > 1: the worker path, in process. `keeps`: a dict the
    keep_pool_inputs each file's run was called with is recorded into. `skeleton`: the NOx-skeleton cap
    record (stats `nox_skeleton_gate`) every run returns. The stand-in run keeps its pool inputs as
    assign.run does: only when asked (no `pool_inputs` key otherwise; None on a calibrated file)."""
    import concurrent.futures as cf
    from peaky.batch import assign_batch as AB
    from peaky.io import io_mascope as IO
    if staged:
        pk, k_min, k_max = _staged_table(), 2, 2
    else:
        pk, k_min, k_max = _batch_table([f"f{i}" for i in range(4)], list(range(100, 140))), 3, 4
    k_of = {}

    def fake_assign(sid, context="ambient-air", **kw):
        k = k_of.setdefault(sid, len(k_of))
        keep = bool(kw.get("keep_pool_inputs", False))
        if keeps is not None:
            keeps[sid] = kw.get("keep_pool_inputs")
        rows = [(nf, AD, 0.05 * ((j % 5) - 2), 1e5 * (1 + j), k < iso_files, "Good")
                for j, nf in enumerate(NEUTRALS[:16])]
        rows.append(("C17H28O6", AD, 2.5, 1e5, False, "Good"))     # off-calibration, uncorroborated
        at = _file(sid, rows, tier=False)
        led = at.copy(deep=True)
        T.apply_tiers(led)
        ri = EV.file_run_inputs(sample_id=sid, reagent="Br", context="ambient-air", resolution=ORBI,
                                height_gate_cps=10.0, degeneracy_cal=None)
        EV.apply_levels(led, run_inputs=ri)
        skel = {} if skeleton is None else {"nox_skeleton_gate": dict(skeleton)}
        if guard_record:
            # the per-file isotope_guard stage's own record (no facts on an uncalibrated file: nothing demoted)
            skel["isotope_guard"] = {"tested": int((led["tier"] == "Assigned").sum()), "demoted": 0,
                                     "by_fact": {"13C": 0, "absent": 0, "unexplained": 0}, "rows": []}
        if calibrated is True or (calibrated and k in calibrated):
            return {"ledger": led, "plausibility_audit": [], "summaries": {}, "problems": [],
                    "stats": {"noise_edge_cps": 4.0, "height_gate_cps": 10.0,
                              "degeneracy_cal": {"mu": 0.0, "sigma": 0.3},
                              "tier_calibration": T.CAL_SOURCE_PER_FILE, **skel},
                    **({"pool_inputs": None} if keep else {})}
        inputs = CP.PoolInputs(at_tiers=at, adducts=[AD], level_inputs=ri) if keep else None
        if refs is not None and inputs is not None:
            refs.append(weakref.ref(inputs))
        return {"ledger": led, "plausibility_audit": [], "summaries": {}, "problems": [],
                "stats": {"noise_edge_cps": 4.0, "height_gate_cps": 10.0, "degeneracy_cal": None,
                          "tier_calibration": T.CAL_SOURCE_NONE, **skel},
                **({"pool_inputs": inputs} if keep else {})}

    monkeypatch.setattr(IO, "connect", lambda *a, **k: "CLIENT")
    monkeypatch.setattr(IO, "fetch_peaks", lambda client, sid, use_cache=True: pd.DataFrame(
        {"peak_id": ["A"], "mz": [200.1], "height": [1.0e5]}))
    monkeypatch.setattr(IO, "estimate_offset", lambda raw: 0.0)
    monkeypatch.setattr(A, "run", fake_assign)
    if n_jobs > 1:
        monkeypatch.setattr(cf, "ProcessPoolExecutor", _InProcessPool)
        monkeypatch.setattr(AB, "_W", {})
        monkeypatch.setenv("PEAKY_MATCH_WORKERS", "2")
    return AB.run(peaks=pk, ts_peaks=pk, reagent=reagent, batch="test batch", out_dir=str(tmp_path),
                  k_min=k_min, k_max=k_max, min_gain=0.0, n_jobs=n_jobs, resolving_power=ORBI,
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
    assert pool["lent_width"] == "pool" and pool["n_files_calibrated"] == 0
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


def test_a_partly_calibrated_batch_lends_the_run_sigma(tmp_path, monkeypatch):
    """One file calibrated alone (sigma 0.3): the others are lent that width about their own centres,
    not the pooled sigma (which also carries the spread between files)."""
    lines = []
    _run_batch(tmp_path, monkeypatch, calibrated={0}, lines=lines)
    summ = json.load(open(tmp_path / "batch_summary.json"))
    tc = summ["tier_calibration"]
    n = len(summ["per_file"])
    assert tc["files"] == {"per_file": 1, "pooled": n - 1, "none": 0} and tc["source"] == "pooled"
    pool = tc["pool"]
    assert pool["used"] is True and pool["lent_width"] == "run" and pool["n_files_calibrated"] == 1
    assert pool["lent_sigma"] == pytest.approx(0.3) and pool["sigma"] != pytest.approx(0.3)
    pooled = [p for p in summ["per_file"] if p["tier_calibration"] == "pooled"]
    assert len(pooled) == n - 1
    for p in pooled:
        assert p["degeneracy_cal"]["sigma"] == pytest.approx(0.3)
        assert p["tier_pool"]["width"] == "run" and p["tier_pool"]["centre"] == "file"
        assert p["tier_pool"]["demoted"] == 1
    assert any("a file is lent the run sigma 0.300 ppm of the 1 file(s)" in ln for ln in lines)


def test_the_merge_vote_reads_the_pooled_files_as_re_tiered(tmp_path, monkeypatch):
    """The pool is applied before a stage's files are folded: the vote class of every pooled file is
    read off its re-tiered ledger (the degeneracy stamps measured in the lent window, the off-calibration
    row Candidate), so the pooled calibration feeds the merge vote too."""
    seen = []
    real = EV.vote_classes

    def spy(trim_frame, **kw):
        m0 = trim_frame[trim_frame["role"] == "M0"]
        seen.append((bool(pd.to_numeric(m0["degeneracy_density"], errors="coerce").notna().all()),
                     set(m0.loc[m0["neutral_formula"] == "C17H28O6", "tier"])))
        return real(trim_frame, **kw)
    monkeypatch.setattr(EV, "vote_classes", spy)
    _run_batch(tmp_path, monkeypatch)
    assert seen and all(measured and tiers == {"Candidate"} for measured, tiers in seen)


def test_the_tier_time_copies_are_released_before_the_fold(tmp_path, monkeypatch):
    """The pooled step is the only reader of a run's pool inputs (its tier-time ledger copy): by the
    time the stage's files are folded (the vote class is read in the fold) none is still held."""
    refs, alive = [], []
    real = EV.vote_classes

    def spy(trim_frame, **kw):
        gc.collect()
        alive.append(sum(r() is not None for r in refs))
        return real(trim_frame, **kw)
    monkeypatch.setattr(EV, "vote_classes", spy)
    _run_batch(tmp_path, monkeypatch, refs=refs)
    assert len(refs) >= 2 and alive and set(alive) == {0}


def test_the_batch_worker_passes_the_keep_flag_to_the_run(monkeypatch):
    """A worker run keeps the pool inputs unless the batch says its pool has nothing to lend."""
    from peaky.batch import assign_batch as AB
    seen = []

    def fake(sid, context="ambient-air", **kw):
        seen.append(kw.get("keep_pool_inputs"))
        return {"ledger": pd.DataFrame(), "stats": {}}
    monkeypatch.setattr(A, "run", fake)
    monkeypatch.setattr(AB, "_W", {"context": "ambient-air", "reflists_active": None, "base_kw": {},
                                   "ts_path": None, "ts": None}, raising=False)
    AB._assign_one("w1")
    AB._assign_one("w2", False)
    assert seen == [True, False]


def test_a_batch_whose_pool_falls_short_stays_unlevelled_and_says_why(tmp_path, monkeypatch):
    lines = []
    _run_batch(tmp_path, monkeypatch, iso_files=1, lines=lines)
    assert any(ln.startswith("[calpool] WARNING: ") and "the pooled core needs 15" in ln for ln in lines)
    summ = json.load(open(tmp_path / "batch_summary.json"))
    tc = summ["tier_calibration"]
    assert tc["files"]["pooled"] == 0 and tc["files"]["none"] == len(summ["per_file"]) and tc["source"] == "none"
    assert tc["pool"]["used"] is False and "the pooled core needs 15" in tc["pool"]["reason"]
    why = summ["levels_not_assessed_reason"]
    assert why.startswith("Evidence levels were not assessed")
    assert "A core pooled over the batch's files was tried and not used" in why
    assert all(p["degeneracy_cal"] is None for p in summ["per_file"])


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_a_later_stage_file_is_lent_the_pool_the_first_stage_built(tmp_path, monkeypatch, n_jobs):
    """The pool is decided once, on the cover files; it has a calibration to lend, so the residual
    stage's runs still keep their pool inputs and its uncalibrated files are re-tiered in it too."""
    lines, keeps = [], {}
    _run_batch(tmp_path, monkeypatch, staged=True, n_jobs=n_jobs, lines=lines, keeps=keeps)
    summ = json.load(open(tmp_path / "batch_summary.json"))
    assert {p["sample_id"]: p["stage"] for p in summ["per_file"]} == {
        "a0": "cover", "a1": "cover", "z1": "residual", "z2": "residual"}
    assert keeps == {"a0": True, "a1": True, "z1": True, "z2": True}
    tc = summ["tier_calibration"]
    assert tc["files"] == {"per_file": 0, "pooled": 4, "none": 0}
    assert all(p["tier_calibration"] == "pooled" and p["tier_pool"]["demoted"] == 1 for p in summ["per_file"])
    assert tc["pool"]["n_files_read"] == 2 and set(tc["pool"]["retiered_files"]) == {"a0", "a1", "z1", "z2"}
    assert sum(ln.startswith("[calpool] ") and "uncalibrated on their own core" in ln for ln in lines) == 1
    led = pd.read_csv(tmp_path / "per_file" / "z2_ledger.csv")
    assert (led.loc[led["role"] == "M0", T.TIER_CALIBRATION] == "pooled").all()


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_once_the_pool_falls_short_later_stages_keep_no_copy(tmp_path, monkeypatch, n_jobs):
    """A pool decided on the cover files with nothing to lend: the residual stage's runs (serial, or
    through the workers) are asked to keep no tier-time copy, and their files say why they stay
    uncalibrated."""
    lines, keeps = [], {}
    _run_batch(tmp_path, monkeypatch, staged=True, iso_files=1, n_jobs=n_jobs, lines=lines, keeps=keeps)
    summ = json.load(open(tmp_path / "batch_summary.json"))
    assert keeps == {"a0": True, "a1": True, "z1": False, "z2": False}
    tc = summ["tier_calibration"]
    assert tc["files"] == {"per_file": 0, "pooled": 0, "none": 4} and tc["pool"]["used"] is False
    assert any(ln.startswith("[calpool] WARNING: 2 more file(s) uncalibrated") for ln in lines)


def test_a_tof_class_batch_keeps_its_files_uncalibrated(tmp_path, monkeypatch):
    """The batch hands the pool its instrument class: a TOF-class batch counts the core and lends nothing."""
    from peaky.batch import assign_batch as AB
    monkeypatch.setattr(AB, "_axis_class", lambda measured, roster: "tof")
    lines = []
    _run_batch(tmp_path, monkeypatch, lines=lines)
    summ = json.load(open(tmp_path / "batch_summary.json"))
    tc = summ["tier_calibration"]
    assert tc["pool"]["used"] is False and "TOF" in tc["pool"]["reason"] and tc["pool"]["n_ions"] == 16
    assert tc["files"] == {"per_file": 0, "pooled": 0, "none": len(summ["per_file"])} and tc["source"] == "none"
    assert any(ln.startswith("[calpool] WARNING: ") and "TOF" in ln for ln in lines)


def test_the_batch_builds_the_pool_on_its_reagents_polarity_and_library(tmp_path, monkeypatch):
    """The pool reads the batch reagent's polarity (the reagent-identity test) and its cluster library's
    ion masses, and the batch's instrument class."""
    seen = []
    real = CP.pool_calibration

    def spy(ledgers, **kw):
        seen.append(kw)
        return real(ledgers, **kw)
    monkeypatch.setattr(CP, "pool_calibration", spy)
    _run_batch(tmp_path, monkeypatch, reagent="Ur")
    assert len(seen) == 1 and seen[0]["polarity"] == "positive" and seen[0]["klass"] == "orbitrap"
    assert len(seen[0]["reagent_masses"]) > 0
    seen.clear()
    _run_batch(tmp_path / "neg", monkeypatch)
    assert len(seen) == 1 and seen[0]["polarity"] == "negative"


def test_a_pooled_files_skeleton_record_is_the_pooled_tier_stages(tmp_path, monkeypatch):
    """stats `nox_skeleton_gate` of a pooled file is replaced by apply_pool's record (the re-run tier
    stage's); a file that calibrated alone keeps its run's own; the batch totals add the replaced ones."""
    own, pooled = {"capped": 0, "noted": 5}, {"capped": 2, "noted": 1}
    monkeypatch.setattr(T, "nox_skeleton_summary", lambda led, profile: dict(pooled))
    _run_batch(tmp_path, monkeypatch, calibrated={0}, skeleton=own)
    summ = json.load(open(tmp_path / "batch_summary.json"))
    src = [p["tier_calibration"] for p in summ["per_file"]]
    assert src.count("per_file") == 1 and src.count("pooled") == len(src) - 1
    for p in summ["per_file"]:
        assert p["nox_skeleton_gate"] == (pooled if p["tier_calibration"] == "pooled" else own)
    n = src.count("pooled")
    assert (summ["nox_skeleton_gate"]["files_capped"], summ["nox_skeleton_gate"]["files_noted"]) == (2 * n, n + 5)


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
def _run_one(monkeypatch, rows, sid, *, keep=True, at_tiers=None):
    """assign.run on a synthetic peak table with the real degeneracy, tier and evidence stages after a
    stand-in pass that commits `rows` (as in _file); `keep`: as a batch's per-file run (keep_pool_inputs).
    `at_tiers`: a list the run state's tier-time copy (st.at_tiers) is appended to, right after the
    degeneracy stage."""
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
    if at_tiers is not None:
        stages.insert(2, A._Stage("spy", lambda st: at_tiers.append(st.at_tiers), safe=False, store=False))
    monkeypatch.setattr(A, "_STAGES", stages)
    try:
        res = A.run(sid, adducts=[AD], peaks=peaks, log=lambda *a: None, keep_pool_inputs=keep)
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


def test_a_single_sample_run_keeps_no_pool_inputs(monkeypatch):
    """Only a batch's per-file run asks for the pool inputs: a single-sample run (never pooled) returns
    no such key, so its manifest (every result key but the ledger) carries none."""
    rows = [(nf, AD, 0.0, 1e5, True, "Good") for nf in NEUTRALS[:6]]
    res, _ = _run_one(monkeypatch, rows, "pool-single", keep=False)
    assert res["stats"]["degeneracy_cal"] is None and "pool_inputs" not in res
    manifest = json.dumps({k: v for k, v in res.items() if k != "ledger"}, default=str)   # as the CLI writes it
    assert "pool_inputs" not in manifest and "PoolInputs" not in manifest


def test_the_tier_time_copy_is_taken_only_when_asked_and_only_uncalibrated(monkeypatch):
    """The degeneracy stage copies the ledger (st.at_tiers) for an uncalibrated file of a run that asked
    (keep_pool_inputs); a run that did not ask, and a calibrated file, hold no copy at all."""
    rows = [(nf, AD, 0.0, 1e5, True, "Good") for nf in NEUTRALS[:6]]
    seen = []
    _run_one(monkeypatch, rows, "copy-kept", keep=True, at_tiers=seen)
    _run_one(monkeypatch, rows, "copy-not-asked", keep=False, at_tiers=seen)
    cal_rows = [(nf, AD, 0.0, 1e5, True, "High") for nf in NEUTRALS + ["C17H28O6", "C18H30O6"]]
    _run_one(monkeypatch, cal_rows, "copy-calibrated", keep=True, at_tiers=seen)
    assert len(seen) == 3 and isinstance(seen[0], pd.DataFrame)
    assert set(seen[0].loc[seen[0]["role"] == "M0", "neutral_formula"]) == set(NEUTRALS[:6])
    assert seen[1] is None and seen[2] is None


def test_the_batch_word_is_the_weakest_files_source():
    from peaky.batch import assign_batch as AB
    assert AB._tier_cal_source([{"tier_calibration": "per_file"}] * 3) == "per_file"
    assert AB._tier_cal_source([{"tier_calibration": "per_file"}, {"tier_calibration": "pooled"}]) == "pooled"
    assert AB._tier_cal_source([{"tier_calibration": "pooled"}, {"tier_calibration": "none"}]) == "none"
    assert AB._tier_cal_source([{}, {"stage": "cover"}]) is None


# --------------------------------------------------------------------------- the isotope guard on the pooled re-level
def test_the_pooled_re_level_runs_the_per_file_isotope_guard():
    """A file its own core could not calibrate is not levelled by its run's evidence stage, so its
    per-file isotope guard had no facts there; the pooled re-level (apply_pool step 4) computes them
    and step 5 guards the file: a C10 reading whose own 13C line reads ~C6 (predicted S/N 40) is
    Candidate with the guard's note, a C10 reading whose line reads C10 stays Assigned."""
    from tests.test_isoguard import _ledger as _guard_ledger
    from peaky.assignment import isoguard as IG
    pool = CP.pool_calibration(_batch())
    assert pool.cal is not None
    for c13, tier in ((6.0, "Candidate"), (10.0, "Assigned")):
        led = _guard_ledger(c13_count=c13, snr13=40.0)
        at = led.copy(deep=True)
        at["tier"] = pd.NA
        out = CP.apply_pool(led, pool, _inputs_for(at, "s1", guard="demote"), audit=[], log=lambda *a: None)
        row = led[led["role"] == "M0"].iloc[0]
        assert row["tier"] == tier, (c13, row["tier"], row["tier_reason"])
        g = out["isotope_guard"]
        assert g["source"] == T.CAL_SOURCE_POOLED and g["tested"] >= 1
        if tier == "Candidate":
            assert g["demoted"] == 1 and g["by_fact"]["13C"] == 1
            assert f"{IG.NOTE_MARK}: 13C reads " in str(row["tier_reason"])
        else:
            assert g["demoted"] == 0 and IG.NOTE_MARK not in str(row["tier_reason"])
    # no level inputs: nothing levelled, no guard record
    led = _guard_ledger(c13_count=6.0, snr13=40.0)
    at = led.copy(deep=True)
    out = CP.apply_pool(led, pool, _inputs_for(at, "s1", levels=False), audit=[], log=lambda *a: None)
    assert "isotope_guard" not in out


def test_the_batch_records_the_pooled_files_guard(tmp_path, monkeypatch):
    """Every pooled file's stats carry the guard record of its pooled re-level -- replacing the
    stage's empty one -- naming the file's calibration source, `tested` = its Assigned M0 rows at the
    guard (those the written ledger still holds Assigned + the guard's demotions)."""
    _run_batch(tmp_path, monkeypatch, guard_record=True)
    summ = json.load(open(tmp_path / "batch_summary.json"))
    for p in summ["per_file"]:
        g = p["isotope_guard"]
        assert p["tier_calibration"] == "pooled" and g["source"] == p["tier_calibration"]
        led = pd.read_csv(tmp_path / "per_file" / f"{p['sample_id']}_ledger.csv")
        n_ass = int(((led["role"] == "M0") & (led["tier"] == "Assigned")).sum())
        assert g["tested"] == n_ass + g["demoted"] and g["tested"] >= 1


def test_a_rejected_pool_keeps_the_stages_guard_record(tmp_path, monkeypatch):
    _run_batch(tmp_path, monkeypatch, iso_files=1, guard_record=True)
    summ = json.load(open(tmp_path / "batch_summary.json"))
    assert summ["tier_calibration"]["pool"]["used"] is False
    for p in summ["per_file"]:
        g = p["isotope_guard"]
        assert "source" not in g and g["demoted"] == 0 and g["tested"] >= 1


def _detached(c13):
    """The guard test's C10 file with its 13C line left unexplained (not the reading's confirmed child)."""
    from tests.test_isoguard import _ledger as _guard_ledger
    led = _guard_ledger(c13_count=c13, snr13=40.0)
    i = led.index[led["peak_id"] == "B"][0]
    led.loc[i, ["role", "parent_peak_id", "iso_label", "parent_neutral_formula", "parent_adduct"]] = \
        ["unexplained", pd.NA, pd.NA, pd.NA, pd.NA]
    led.loc[led["role"] == "M0", "isotopologues"] = None
    return led


def test_the_pooled_guard_has_the_last_tier_word_after_the_re_tier():
    """The uncalibrated tier stage left the C10 reading Candidate (5 plausible formulas in its wide
    window); the pooled window re-tiers it Assigned -- and the guard, after the re-tier, takes it back
    to Candidate on its own 13C line."""
    pool = CP.pool_calibration(_batch())
    led = _detached(6.0)
    at = led.copy(deep=True)
    at["tier"] = pd.NA
    for c in ("degeneracy_density", "degeneracy_note"):
        at[c] = pd.Series([pd.NA] * len(at), dtype=object)
    at.loc[at["peak_id"] == "A", "degeneracy_density"] = 5
    base = T.compute_tiers(at, cal=None).set_index("peak_id")
    assert base.at["A", "tier"] == "Candidate"
    led.loc[led["peak_id"] == "A", ["tier", "tier_reason"]] = [base.at["A", "tier"], base.at["A", "tier_reason"]]
    out = CP.apply_pool(led, pool, _inputs_for(at, "s1", guard="demote"), audit=[], log=lambda *a: None)
    row = led[led["role"] == "M0"].iloc[0]
    assert out["tiers"]["promoted"] == 1 and out["isotope_guard"]["demoted"] == 1
    assert row["tier"] == "Candidate" and "isotope contradiction: 13C reads " in str(row["tier_reason"])


def test_the_pooled_guard_reads_the_finished_ledgers_facts():
    """The tier-time copy's 13C line read C10; the finished ledger's reads C6: the guard reads the
    finished ledger's re-level."""
    from tests.test_isoguard import _ledger as _guard_ledger
    pool = CP.pool_calibration(_batch())
    led = _guard_ledger(c13_count=6.0, snr13=40.0)
    at = _guard_ledger(c13_count=10.0, snr13=40.0)
    at["tier"] = pd.NA
    out = CP.apply_pool(led, pool, _inputs_for(at, "s1", guard="demote"), audit=[], log=lambda *a: None)
    assert out["isotope_guard"]["demoted"] == 1 and led[led["role"] == "M0"].iloc[0]["tier"] == "Candidate"


def test_the_guard_record_names_the_pools_source():
    from tests.test_isoguard import _ledger as _guard_ledger
    pool = CP.pool_calibration(_batch())
    pool.source = "campaign"                          # a lent calibration's source
    led = _guard_ledger(c13_count=6.0, snr13=40.0)
    at = led.copy(deep=True)
    at["tier"] = pd.NA
    out = CP.apply_pool(led, pool, _inputs_for(at, "s1"), audit=[], log=lambda *a: None)
    assert out["isotope_guard"]["source"] == "campaign"
