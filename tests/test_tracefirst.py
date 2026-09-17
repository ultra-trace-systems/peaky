"""batch/tracefirst.py + the offline engine path + the --trace-first batch run,
on a synthetic batch whose ions are known."""
import json
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from peaky.assignment import assign as A
from peaky.batch import assign_batch as AB
from peaky.batch import tracefirst as TFT
from peaky.chem import chemistry as C
from peaky.io import io_mascope as IO

T0 = pd.Timestamp("2026-01-01T00:00:00Z")
IONS = [("C9H14O4", 40.0), ("C10H16O4", 25.0), ("C5H8O4", 12.0)]     # neutral, height


def _batch(rng, n=800, sigma=1.5, noise_per_da=0.6):
    """Hourly spectra: three [M-H]- acids at their exact masses (jitter `sigma`
    ppm, present in 95 % of spectra); the 13C satellite of the brightest at the
    natural ratio in 90 % of spectra (bright enough to seed its own trace, as a
    real one does); the 13C of the second acid picked in only 3.3 % of spectra
    (too sparse to seed: the satellite path must find it); sparse uniform noise;
    and one noise-fill position: ONE random peak inside +-12 ppm of m/z 230.1 in
    40 % of the spectra (recurrent enough to seed, never an ion: its members
    fill the window; a fill that recurs in >= 50 % would be KEPT as a blended
    peak, by design)."""
    rows = []
    for i in range(n):
        sid = f"s{i:04d}"
        stamp = T0 + pd.Timedelta(hours=i)
        for f, h in IONS:
            if rng.uniform() < 0.95:
                rows.append((sid, C.ion_mz(f, "[M-H]-") * (1 + rng.normal(0, sigma) * 1e-6), h, h * 1.2, stamp))
        m13 = C.ion_mz("C9H14O4", "[M-H]-") + 1.0033548
        if rng.uniform() < 0.9:
            rows.append((sid, m13 * (1 + rng.normal(0, sigma * 1.5) * 1e-6), 40.0 * 0.0973, 40.0 * 0.0973 * 1.2, stamp))
        # an EPISODIC bright acid in 3 % of spectra: below the seed threshold, so
        # no trace -- the residual stage's domain
        if rng.uniform() < 0.03:
            me = C.ion_mz("C7H12O5", "[M-H]-")
            rows.append((sid, me * (1 + rng.normal(0, sigma) * 1e-6), 60.0, 72.0, stamp))
            rows.append((sid, (me + 1.0033548) * (1 + rng.normal(0, sigma) * 1e-6), 60.0 * 0.076, 72.0 * 0.076, stamp))
        d13 = C.ion_mz("C10H16O4", "[M-H]-") + 1.0033548
        if rng.uniform() < 0.033:
            rows.append((sid, d13 * (1 + rng.normal(0, sigma * 2) * 1e-6), 25.0 * 0.108, 25.0 * 0.108 * 1.2, stamp))
        for m in rng.uniform(100.0, 300.0, rng.poisson(noise_per_da * 200)):
            rows.append((sid, m, 1.0, 1.2, stamp))
        if rng.uniform() < 0.4:
            rows.append((sid, float(rng.uniform(230.1 * (1 - 12e-6), 230.1 * (1 + 12e-6))), 1.0, 1.2, stamp))
    return pd.DataFrame(rows, columns=["sample_item_id", "mz", "height", "area", "datetime_utc"])


@pytest.fixture(scope="module")
def ts():
    return _batch(np.random.default_rng(11))


@pytest.fixture(scope="module")
def sample(ts):
    lines = []
    s = TFT.build_trace_sample(ts, sample_id="traces-test", reagent="NO3", resolving_power=6500.0,
                               log=lines.append)
    s.notes.append("log:" + "\n".join(lines))
    return s


def _nearest(traces, mz):
    d = (traces["mz"] - mz).abs() / mz * 1e6
    return traces.loc[d.idxmin()], float(d.min())


def test_the_ions_become_seed_traces_and_the_satellite_finds_its_parent(sample):
    t = sample.traces
    for f, _ in IONS:
        row, d = _nearest(t, C.ion_mz(f, "[M-H]-"))
        assert d < 2.0 and row["kind"] == "seed" and row["trace_occurrence"] > 0.9, (f, d)
        assert row["resid_ppm"] < 3.0 and not row["fills_window"]
    # the bright 13C recurs in 90 % of spectra: it seeds its own trace, and the
    # engine's isotope passes attach it to the parent later (as on a real batch)
    parent, _ = _nearest(t, C.ion_mz("C9H14O4", "[M-H]-"))
    sat, d = _nearest(t, C.ion_mz("C9H14O4", "[M-H]-") + 1.0033548)
    assert d < 3.0 and sat["kind"] == "seed"
    # batch-mean heights keep the isotope ratio honest
    assert abs(sat["height_avg"] / parent["height_avg"] - 0.0973 * 0.9 / 0.95) < 0.02
    # the sparse 13C (4.5 % of spectra) cannot seed: the satellite path finds it,
    # links it to its parent and measures the co-occurrence
    parent2, _ = _nearest(t, C.ion_mz("C10H16O4", "[M-H]-"))
    sat2, d2 = _nearest(t, C.ion_mz("C10H16O4", "[M-H]-") + 1.0033548)
    assert d2 < 6.0 and sat2["kind"] == "sat:13C" and sat2["parent_peak"] == parent2["peak_id"], sat2
    assert sat2["parent_cooccur"] > 0.85 and 20 <= sat2["n_members"] < 0.05 * 800


def test_the_fill_position_is_rejected_and_the_noise_is_not_a_trace(sample):
    t = sample.traces
    _, d = _nearest(t, 230.1)
    assert d > 30.0                                   # nothing within the dedup cell
    assert len(t) <= 9                                # 3 ions + 2 satellites (+ maybe a 13C2/18O)
    assert (t["mz"] - C.ion_mz("C7H12O5", "[M-H]-")).abs().min() / 190 * 1e6 > 30   # the episodic acid is no trace
    assert (t["kind"] == "seed").sum() == 4           # the bright 13C seeds too


def test_the_sample_and_summary_carry_what_the_engine_and_the_manifest_need(sample):
    assert list(sample.peaks.columns[:6]) == ["sample_item_id", "peak_id", "mz", "sparsity", "area", "height"]
    assert set(TFT.MATCH_COLS) <= set(sample.peaks.columns)
    assert (sample.peaks["sample_item_id"] == "traces-test").all()
    assert sample.occurrence.attrs["n_samples"] == 800 and sample.occurrence.attrs["tol_ppm"] == sample.tol_ppm
    s = sample.summary()
    assert s["n_seeds"] == 4 and s["n_satellites"] >= 1 and s["resolving_power"] == 6500.0
    assert abs(s["dedup_ppm"] - 30.77) < 0.1
    # the nitrate reference ions are absent from this synthetic batch, so mass-qc says so
    assert s["mass_qc"]["verdict"] == "no_reference"
    assert sample.tol_ppm == 12.0


def test_the_wave_is_applied_only_inside_the_calibrant_range():
    tr = pd.DataFrame({"mz": [50.0, 100.0, 200.0, 400.0], "height_med": 1.0, "kind": "seed"})
    from peaky.batch import wave as WV
    mz = np.array([80.0, 120.0, 160.0, 200.0, 240.0, 280.0, 320.0, 360.0])
    w = WV.fit_wave(mz, 2.0 + 0.0 * mz)                      # a flat +2 ppm bias
    qc = {"wave": w.as_dict()}
    out = TFT.apply_wave(tr, qc, log=lambda *a: None)
    assert out.loc[0, "wave_ppm"] == 0.0 and out.loc[3, "wave_ppm"] == 0.0     # outside 80-360
    assert abs(out.loc[1, "wave_ppm"] - 2.0) < 1e-6 and abs(out.loc[2, "mz"] - 200.0 * (1 - 2e-6)) < 1e-9
    assert out.loc[1, "mz_raw"] == 100.0


def test_the_engine_runs_offline_on_the_trace_sample_and_finds_the_acids(sample):
    from peaky.assignment.passes import PassConfig
    cfg = PassConfig()
    TFT.engine_settings(cfg, sample, log=lambda *a: None)
    assert cfg.search_ppm == 12.0 and cfg.ppm == 5.0
    real_connect = IO.connect
    IO.connect = lambda *a, **k: (_ for _ in ()).throw(AssertionError("offline run must not connect"))
    try:
        res = A.run("traces-test", context="ambient-air", cfg=cfg, adducts=["[M-H]-"],
                    peaks=sample.peaks, occurrence=sample.occurrence, log=lambda *a: None)
    finally:
        IO.connect = real_connect
        IO.unregister_offline_sample("traces-test")
    led = res["ledger"]
    m0 = led[led["role"] == "M0"]
    got = set(m0["neutral_formula"])
    assert {"C9H14O4", "C10H16O4"} <= got, got
    kids = led[led["role"] == "iso_child"]
    assert (kids["iso_label"].astype(str) == "13C").any()


def test_the_batch_run_takes_the_trace_first_path(ts, tmp_path):
    real_connect = IO.connect
    IO.connect = lambda *a, **k: SimpleNamespace(name="stub-client")
    try:
        res = AB.run(peaks=ts, ts_peaks=ts, reagent="NO3", batch="test batch",
                     out_dir=str(tmp_path), trace_first=True, resolving_power=6500.0,
                     residual=False, n_jobs=1, log=lambda *a: None)
    finally:
        IO.connect = real_connect
    summ = res["summary"]
    tf = summ["trace_first"]
    assert tf["n_seeds"] == 4 and summ["selection"]["method"] == "trace-first"
    merged = res["merged"]
    assert {"C9H14O4", "C10H16O4"} <= set(merged["neutral_formula"])
    assert (merged["n_files"] == 1).all()
    assert os.path.exists(tmp_path / "per_file" / "traces-test-batch_ledger.csv")
    assert os.path.exists(tmp_path / "tables" / "traces.csv")
    js = json.load(open(tmp_path / "batch_summary.json"))
    assert js["trace_first"]["n_traces"] == tf["n_traces"]


def test_trace_first_refuses_to_run_without_the_resolving_power(ts, tmp_path):
    real_connect = IO.connect
    IO.connect = lambda *a, **k: SimpleNamespace(name="stub-client")
    try:
        with pytest.raises(ValueError, match="resolving_power"):
            AB.run(peaks=ts, ts_peaks=ts, reagent="NO3", batch="b", out_dir=str(tmp_path),
                   trace_first=True, log=lambda *a: None)
        with pytest.raises(ValueError, match="ts_peaks"):
            AB.run(peaks=ts, ts_peaks=None, reagent="NO3", batch="b", out_dir=str(tmp_path),
                   trace_first=True, resolving_power=6500.0, log=lambda *a: None)
    finally:
        IO.connect = real_connect


def test_offline_mechanism_lookups_resolve_only_the_declared_channels():
    IO.register_offline_sample("x", pd.DataFrame({"mz": [1.0]}), ["-H+", "+NO3-"])
    try:
        assert IO.resolve_mechanism_ids(None, ["-H+", "+NO3-", "+CO3-"]) == {"-H+": "-H+", "+NO3-": "+NO3-"}
        # a channel declared in the legacy spelling answers for the standard one
        assert IO.resolve_mechanism_ids(None, ["[M-H]-", "[M+CO3]-"]) == {"[M-H]-": "[M-H]-"}
        # handed to the scorer in the standard notation; '-H+' is a proton removed
        # (the anion), never flipped to the hydride loss '-H-'
        assert IO._mechanism_names(None, ["-H+", "+NO3-"]) == ["[M-H]-", "[M+NO3]-"]
        assert IO._mechanism_names(None, ["[M-H]-", "-H+"]) == ["[M-H]-"]
        assert IO.is_offline_sample("x") and len(IO.fetch_peaks(None, "x")) == 1
    finally:
        IO.unregister_offline_sample("x")
    assert not IO.is_offline_sample("x")


def _per_sample_tables(ts):
    """The per-file peak tables the engine would fetch, one per spectrum."""
    out = {}
    for sid, g in ts.groupby("sample_item_id"):
        t = pd.DataFrame({"sample_item_id": sid, "peak_id": [f"{sid}-{i}" for i in range(len(g))],
                          "mz": g["mz"].to_numpy(), "sparsity": 0.0,
                          "area": g["area"].to_numpy(), "height": g["height"].to_numpy()})
        for c in TFT.MATCH_COLS:
            t[c] = None
        out[sid] = t
    return out


def test_the_whole_batch_path_runs_offline_with_the_rolling_centre(ts, tmp_path):
    """The file cover, the per-file engine runs, the merge, the trace
    reconciliation WITH the rolling centre, the per-trace stamp and the summary
    -- all offline, every selected sample served from memory."""
    tabs = _per_sample_tables(ts)
    for sid, t in tabs.items():
        IO.register_offline_sample(sid, t, ["-H+"])
    real_connect, real_fetch = IO.connect, IO.fetch_peaks
    IO.connect = lambda *a, **k: None
    try:
        res = AB.run(peaks=ts, ts_peaks=ts, reagent="NO3", batch="test batch",
                     out_dir=str(tmp_path), residual=False, n_jobs=1, k_min=2, k_max=3,
                     rolling_centre=True, log=lambda *a: None)
    finally:
        IO.connect, IO.fetch_peaks = real_connect, real_fetch
        for sid in tabs:
            IO.unregister_offline_sample(sid)
    summ = res["summary"]
    tr = summ["traces"]
    assert tr["rolling"]["enabled"] and "stamp_tol_per_trace" in tr
    assert tr["rolling"]["n_rolling"] + tr["rolling"]["n_global"] >= 2
    merged = res["merged"]
    assert {"centre_scheme", "resid_ppm", "stamp_tol_ppm", "trace_key"} <= set(merged.columns)
    assert {"C9H14O4", "C10H16O4"} <= set(merged["neutral_formula"])
    ann = pd.read_parquet(tmp_path / "per_file" / "_batch_ts.parquet")   # the stamped time series
    n = ts["sample_item_id"].nunique()
    cov = ann[ann["neutral_formula"] == "C9H14O4"]["sample_item_id"].nunique() / n
    assert cov > 0.9


def test_trace_first_and_the_rolling_centre_compose(ts, tmp_path):
    real_connect = IO.connect
    IO.connect = lambda *a, **k: SimpleNamespace(name="stub-client")
    try:
        res = AB.run(peaks=ts, ts_peaks=ts, reagent="NO3", batch="test batch",
                     out_dir=str(tmp_path), trace_first=True, resolving_power=6500.0,
                     rolling_centre=True, residual=False, n_jobs=1, log=lambda *a: None)
    finally:
        IO.connect = real_connect
    summ = res["summary"]
    assert summ["trace_first"]["n_seeds"] == 4 and summ["traces"]["rolling"]["enabled"]
    assert "stamp_tol_ppm" in res["merged"].columns


def test_under_trace_first_a_residual_file_only_adds_what_the_stamp_left_unexplained(ts, tmp_path):
    """The episodic acid recurs in 3 % of spectra -- no trace -- so its bins are
    residual; the residual files that cover them also carry the three persistent
    acids, which the trace ledger already explains: those rows must NOT merge
    back in (they would out-vote the trace reading), the episodic one must."""
    tabs = _per_sample_tables(ts)
    for sid, t in tabs.items():
        IO.register_offline_sample(sid, t, ["-H+"])
    real_connect = IO.connect
    IO.connect = lambda *a, **k: None
    try:
        res = AB.run(peaks=ts, ts_peaks=ts, reagent="NO3", batch="test batch",
                     out_dir=str(tmp_path), trace_first=True, resolving_power=6500.0,
                     residual=True, residual_k_max=2, n_jobs=1, log=lambda *a: None)
    finally:
        IO.connect = real_connect
        for sid in tabs:
            IO.unregister_offline_sample(sid)
    summ = res["summary"]
    rmeta = summ["selection"]["residual"]
    assert rmeta["k"] >= 1, rmeta
    sc = rmeta["trace_first_scope"]
    assert 0 < sc["rows_kept"] < sc["rows_total"]
    merged = res["merged"]
    persistent = merged[merged["neutral_formula"].isin(["C9H14O4", "C10H16O4", "C5H8O4"])]
    assert (persistent["n_files"] == 1).all()                 # the trace reading, unchanged
    assert "C7H12O5" in set(merged["neutral_formula"])       # the episodic acid, added


def test_the_membership_window_never_exceeds_half_the_instruments_own_cell(ts):
    """At high resolving power the 0.4-HWHM cell is ~1 ppm wide, so the TOF's
    validated 12 ppm membership floor must not apply: a trace's members may
    never span more than one observable."""
    lines = []
    s = TFT.build_trace_sample(ts, sample_id="hi-res", reagent="NO3", resolving_power=155000.0,
                               log=lines.append)
    cell = TFT.dedup_ppm(200.0, 155000.0)
    assert cell < 2.0
    assert s.tol_ppm <= cell / 2 + 1e-9, (s.tol_ppm, cell)
    assert any("high-resolution instrument" in n for n in s.notes)
    assert any("NOTE:" in ln for ln in lines)
    # the TOF case is unchanged: the floor applies because the cell is wide
    t = TFT.build_trace_sample(ts, sample_id="tof", reagent="NO3", resolving_power=6500.0,
                               log=lambda *a: None)
    assert t.tol_ppm == TFT.MEMBER_MIN_PPM
    assert not any("high-resolution" in n for n in t.notes)


def test_trace_first_on_a_reagent_with_no_reference_table_says_so_and_runs(ts):
    """A positive-mode reagent has no reference-ion list yet: mass-qc is skipped,
    no wave is applied, and the run says both in its notes rather than failing."""
    s = TFT.build_trace_sample(ts, sample_id="no-refs", reagent="Ur", resolving_power=6500.0,
                               log=lambda *a: None)
    assert s.qc is None and s.summary()["mass_qc"] is None
    assert any("no reference-ion table" in n for n in s.notes)
    assert len(s.traces) and (s.traces["wave_ppm"] == 0).all()
    assert s.tol_ppm == TFT.MEMBER_MIN_PPM
