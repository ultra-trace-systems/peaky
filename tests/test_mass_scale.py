"""The batch's mass scale (traces.MassScale): one measured per-ion scatter, the
merge and stamping windows sized from it, the flat-window run recovered exactly
when nothing was measured."""
from __future__ import annotations

import json
import math
import os

import numpy as np
import pandas as pd
import pytest

from peaky.batch import assign_batch as AB
from peaky.batch import sampling as SS
from peaky.batch import timeseries as TS
from peaky.batch import traces as TR

TOL = 6.0


def _index(n_spectra: int, centres, sigma_ppm: float, seed: int = 7) -> TR.PeakIndex:
    """A batch time series: every spectrum holds one draw of every trace."""
    rng = np.random.default_rng(seed)
    rows = [dict(sample_item_id=f"s{s:03d}", mz=float(c) * (1.0 + rng.normal(0.0, sigma_ppm) * 1e-6),
                 height=1000.0)
            for s in range(n_spectra) for c in centres]
    return TR.PeakIndex(pd.DataFrame(rows), tol_ppm=TOL)


# ---------------------------------------------------------------- the window rule
def test_window_rule_floors_on_the_tolerance_and_caps_at_twice_it():
    assert TR.window_ppm(TOL, float("nan")) == TOL            # nothing measured
    assert TR.window_ppm(TOL, None) == TOL
    assert TR.window_ppm(TOL, 0.3) == TOL                     # an Orbitrap sits on the floor
    assert TR.window_ppm(TOL, 4.0, k_sigma=2.5) == pytest.approx(10.0)
    assert TR.window_ppm(TOL, 4.0, k_sigma=TR.MERGE_GAP_SIGMA) == pytest.approx(12.0)   # capped
    assert TR.window_ppm(TOL, 40.0) == pytest.approx(12.0)


def test_the_merge_gap_is_the_stamp_window_for_two_draws():
    assert TR.MERGE_GAP_SIGMA == pytest.approx(TR.WINDOW_SIGMA * math.sqrt(2.0))
    assert TS.STAMP_TOL_SIGMA == TR.WINDOW_SIGMA and TS.STAMP_TOL_MAX_X == TR.WINDOW_MAX_X


def test_an_unmeasured_scale_is_the_tolerance_on_both_windows():
    sc = TR.MassScale(tol_ppm=TOL)
    assert not sc.measured and sc.merge_ppm == TOL and sc.stamp_ppm == TOL
    d = sc.as_dict()
    assert d["source"] == "unmeasured" and d["sigma_ppm"] is None
    assert d["merge_ppm"] == TOL and d["stamp_ppm"] == TOL and d["tol_ppm"] == TOL
    assert "not measured" in sc.describe()
    assert not TR.measure_mass_scale(None, [200.0], tol_ppm=TOL).measured
    assert not TR.measure_mass_scale(_index(30, [200.0], 1.0), [], tol_ppm=TOL).measured


# ---------------------------------------------------------------- measuring it
def test_a_tof_like_batch_widens_the_merge_more_than_the_stamp():
    centres = [200.0, 250.0, 300.0, 350.0, 400.0, 450.0]
    idx = _index(40, centres, sigma_ppm=4.0)
    # three per-file anchors per trace, each one spectrum's draw (up to 5 ppm off)
    seeds = [c * (1 + d * 1e-6) for c in centres for d in (-5.0, 1.0, 4.5)]
    sc = TR.measure_mass_scale(idx, seeds, tol_ppm=TOL)
    assert sc.measured and sc.n_seeds == 18
    assert sc.n_traces == len(centres), "one centre per trace: the three anchors of a trace count once"
    assert 3.0 < sc.sigma_ppm < 5.5, sc
    assert sc.stamp_ppm == pytest.approx(min(2 * TOL, 2.5 * sc.sigma_ppm))
    assert sc.merge_ppm == pytest.approx(min(2 * TOL, TR.MERGE_GAP_SIGMA * sc.sigma_ppm))
    assert sc.merge_ppm > sc.stamp_ppm > TOL
    d = sc.as_dict()
    assert d["source"] == "measured" and d["n_traces"] == 6 and d["merge_ppm"] == pytest.approx(12.0)
    assert "merge window" in sc.describe()


def test_an_orbitrap_like_batch_keeps_the_tolerance_on_both_windows():
    centres = [200.0, 300.0, 400.0]
    idx = _index(40, centres, sigma_ppm=0.3)
    sc = TR.measure_mass_scale(idx, centres, tol_ppm=TOL)
    assert sc.measured and sc.sigma_ppm < 0.6
    assert sc.merge_ppm == TOL and sc.stamp_ppm == TOL


def test_seeds_walk_onto_their_trace_before_the_scatter_is_read():
    """An anchor 5 ppm off its trace reads a truncated (low) scatter through a
    +-6 ppm window; walked onto the trace first, the scatter is the trace's."""
    centres = [300.0, 400.0]
    idx = _index(60, centres, sigma_ppm=3.0)
    off = [c * (1 - 5.0e-6) for c in centres]
    raw = TR.batch_scatter_ppm(idx, off, tol_ppm=TOL)
    sc = TR.measure_mass_scale(idx, off, tol_ppm=TOL)
    assert sc.n_traces == 2 and sc.sigma_ppm > raw
    assert abs(sc.sigma_ppm - 3.0) < 1.0


def test_a_trace_too_sparse_to_measure_does_not_count():
    idx = _index(40, [200.0, 300.0], sigma_ppm=1.0)
    sc = TR.measure_mass_scale(idx, [200.0, 300.0, 700.0], tol_ppm=TOL)   # 700: no peaks at all
    assert sc.n_traces == 2 and sc.n_seeds == 3
    lone = TR.measure_mass_scale(idx, [700.0], tol_ppm=TOL)
    assert not lone.measured and lone.merge_ppm == TOL


def test_stamp_tolerance_is_the_same_rule_on_a_merged_ledger():
    centres = [200.0, 300.0, 400.0]
    idx = _index(40, centres, sigma_ppm=4.0)
    win, sig = TS.stamp_tolerance(idx, centres, tol_ppm=TOL)
    raw = TR.batch_scatter_ppm(idx, centres, tol_ppm=TOL)
    assert sig == round(raw, 3)
    assert win == pytest.approx(TR.window_ppm(TOL, raw))      # sized from the unrounded sigma
    assert TS.stamp_tolerance(None, centres, tol_ppm=TOL) == (TOL, pytest.approx(float("nan"), nan_ok=True))


# ---------------------------------------------------------------- through run()
from peaky.assignment import assign as _A  # noqa: E402
from peaky.assignment import ledger as _L  # noqa: E402
from peaky.assignment import tiers as _T  # noqa: E402
from peaky.chem import chemistry as _C  # noqa: E402
from peaky.io import io_mascope as IO  # noqa: E402

_F = "C10H16O5"
_MZ = _C.ion_mz(_F, "[M-H]-")
_T0 = pd.Timestamp("2025-10-01 21:00:00", tz="UTC")


def _tof_batch(sigma_ppm: float = 4.0, seed: int = 11) -> pd.DataFrame:
    """40 spectra; every one holds the shared background bins and one draw of the
    analyte's trace. Exclusive bins make s000 the first cover pick and s001 the
    second (the universe needs a bin in >= 2 samples, so each block is in two)."""
    rng = np.random.default_rng(seed)
    bg = list(range(100, 120))
    b0, b1, b2 = list(range(200, 240)), list(range(300, 320)), list(range(400, 410))
    b3, b4 = list(range(500, 510)), list(range(600, 620))
    # pick 1: s000 (91 universe bins) over s002 (71); pick 2: s001 adds b1 + b3 = 30
    # new bins over s003's 20 (b1) and s002's 10 (b3)
    extra = {"s000": b0 + b2 + b4, "s001": b1 + b2 + b3, "s002": b0 + b3, "s003": b1 + b4}
    rows = []
    for i in range(40):
        sid = f"s{i:03d}"
        t = _T0 + pd.Timedelta(minutes=10 * i)
        mzs = bg + extra.get(sid, [])
        if sid == "s000":
            draw = _MZ * (1 + 4.0e-6)
        elif sid == "s001":
            draw = _MZ * (1 - 4.0e-6)
        else:
            draw = _MZ * (1 + rng.normal(0.0, sigma_ppm) * 1e-6)
        rows += [dict(sample_item_id=sid, sample_item_name=f"n_{sid}", datetime_utc=t,
                      mz=float(m), height=500.0) for m in mzs]
        rows.append(dict(sample_item_id=sid, sample_item_name=f"n_{sid}", datetime_utc=t,
                         mz=float(draw), height=5000.0))
    return pd.DataFrame(rows)


def _stub_engine(monkeypatch, anchors: dict):
    """The IO layer and the per-file engine stubbed: each cover file's ledger holds
    ONE assigned M0 at that file's own anchor (its draw of the trace)."""
    monkeypatch.setattr(IO, "connect", lambda *a, **k: "CLIENT")
    monkeypatch.setattr(IO, "fetch_peaks", lambda client, sid, use_cache=True: pd.DataFrame(
        {"peak_id": ["p1"], "mz": [anchors[sid]], "height": [1.0e5]}))
    monkeypatch.setattr(IO, "estimate_offset", lambda raw: 0.0)

    def fake(sid, context="ambient-air", **kw):
        led = _L.new_ledger(pd.DataFrame([("p1", anchors[sid], 1.0e5)],
                                         columns=["peak_id", "mz", "height"]))
        _L.commit_assignment(led, "p1", neutral_formula=_F, adduct="[M-H]-",
                             ion_formula="C10H15O5-", ion_score=0.9, compound_score=0.9,
                             ppm_error=0.1, pass_no=1, method="cheminfo+grid",
                             confidence="High", commentary="stub")
        _T.apply_tiers(led)
        return {"ledger": led, "stats": {"noise_edge_cps": 4.0, "height_gate_cps": 10.0,
                                         "degeneracy_cal": {"mu": 0.0, "sigma": 0.3}},   # as a real run persists
                "plausibility_audit": [], "summaries": {}, "problems": []}
    monkeypatch.setattr(_A, "run", fake)


def _run(tmp_path, monkeypatch, *, ts: bool):
    pk = _tof_batch()
    # every file's anchor is its own draw of the trace (s000 +4 ppm, s001 -4 ppm)
    anchors = {sid: float(g.loc[g["height"].idxmax(), "mz"]) for sid, g in pk.groupby("sample_item_id")}
    assert anchors["s000"] == pytest.approx(_MZ * (1 + 4.0e-6)) and anchors["s001"] == pytest.approx(_MZ * (1 - 4.0e-6))
    _stub_engine(monkeypatch, anchors)
    lines = []
    AB.run(peaks=pk, ts_peaks=pk if ts else None, reagent="Br", batch="test batch",
           out_dir=str(tmp_path), k_min=2, k_max=2, min_gain=0.0, n_jobs=1, residual=False,
           resolving_power="none", log=lines.append)
    summ = json.load(open(tmp_path / "batch_summary.json"))
    # the pooled level stage ran with every file's persisted calibration, to its class gate: a declined
    # width model reads NA on every pair
    el = summ["evidence_levels"]
    assert el["n_pairs"] > 0 and el["pooled"] == {"NA": el["n_pairs"]}
    assert all(pf["degeneracy_cal"] == {"mu": 0.0, "sigma": 0.3} for pf in summ["per_file"])
    merged = pd.read_csv(tmp_path / "merged_ledger.csv")
    per_file = {sid: AB._m0(pd.read_csv(tmp_path / "per_file" / f"{sid}_ledger.csv"))
                for sid in summ["sample_ids"]}
    return summ, merged, per_file, lines


def test_run_without_a_time_series_is_the_flat_window_run(tmp_path, monkeypatch):
    summ, merged, per_file, lines = _run(tmp_path, monkeypatch, ts=False)
    ms = summ["mass_scale"]
    assert summ["sample_ids"] == ["s000", "s001"]
    assert ms["source"] == "unmeasured" and ms["sigma_ppm"] is None and ms["n_seeds"] == 2
    assert ms["merge_ppm"] == ms["stamp_ppm"] == ms["tol_ppm"] == summ["tol_ppm"] == SS.BATCH_TOL_PPM
    # the two anchors sit 8 ppm apart: at the flat 6 ppm window they are two rows,
    # exactly what align() gives at that window
    assert len(merged) == 2 and merged["n_files"].tolist() == [1, 1]
    flat, _ = AB.align(per_file, tol_ppm=SS.BATCH_TOL_PPM)
    assert len(flat) == 2
    assert any("[scale] per-ion mass scatter not measured" in x for x in lines)
    assert summ["traces"] == {}, "no time series: no trace block, no stamping window to report"


def test_run_with_a_tof_like_time_series_fuses_the_split_anchors(tmp_path, monkeypatch):
    summ, merged, per_file, lines = _run(tmp_path, monkeypatch, ts=True)
    ms = summ["mass_scale"]
    assert summ["sample_ids"] == ["s000", "s001"]
    assert ms["source"] == "measured" and 3.0 < ms["sigma_ppm"] < 5.5 and ms["n_traces"] == 1
    assert ms["n_seeds"] == 2 and ms["merge_ppm"] > ms["stamp_ppm"] > ms["tol_ppm"] == SS.BATCH_TOL_PPM
    assert ms["merge_ppm"] == pytest.approx(min(12.0, TR.MERGE_GAP_SIGMA * ms["sigma_ppm"]), abs=1e-3)
    # the same two anchors, 8 ppm apart, are ONE ion at the measured window ...
    assert len(merged) == 1 and int(merged["n_files"].iloc[0]) == 2
    assert merged["neutral_formula"].iloc[0] == _F and int(merged["n_files_ion"].iloc[0]) == 2
    # ... and two rows at the flat one, so the test discriminates
    assert len(AB.align(per_file, tol_ppm=SS.BATCH_TOL_PPM)[0]) == 2
    # the stamp reads the same scale, and the summary's trace block agrees with it
    tr = summ["traces"]
    assert tr["stamp_tol_ppm"] == pytest.approx(ms["stamp_ppm"], abs=1e-3) and tr["sigma_ppm"] == ms["sigma_ppm"]
    assert any("[scale] per-ion mass scatter" in x and "merge window" in x for x in lines)
    assert summ["tol_ppm"] == summ["selection"]["tol_ppm"] == summ["admission"]["tol_ppm"] == SS.BATCH_TOL_PPM, \
        "the binning tolerance is untouched: selection, admission and the trace index still bin at it"
