"""`peaky batch --mass-axis`: the batch's m/z axis measured against the reagent's
reference ions (batch.massqc) and corrected before anything is assigned --
assign_batch.measure_axis / restore_axis, the io registry every peak table passes
through, the hand-off to spawned workers, a whole offline batch run, the
pipeline's clean-up and report series, and the CLI flag."""
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from peaky.batch import assign_batch as AB
from peaky.batch import massqc as MQ
from peaky.batch.wave import WaveFit
from peaky.chem import reference_ions as RI
from peaky.io import io_mascope as IO
from tests.test_massqc import _batch, _hump, _orbi_batch


@pytest.fixture(autouse=True)
def _no_axis_left_behind():
    IO.clear_axis_correction()
    yield
    IO.clear_axis_correction()


@pytest.fixture(scope="module")
def refs():
    return RI.nitrate()


def _const(ppm=2.0, lo=70.0, hi=300.0) -> WaveFit:
    """A constant correction (an axis_offset): it reaches every m/z."""
    return WaveFit(coef=[ppm], p=-0.5, domain=(hi ** -0.5, lo ** -0.5), mz_range=(lo, hi), K=0,
                   n=10, n_clipped=0, resid_ppm=0.1, raw_ppm=0.1, loo_ppm=0.1, span_ppm=0.0,
                   share=0.0)


def _trend(lo=70.0, hi=300.0) -> WaveFit:
    """A linear trend in (m/z)^-1/2, +1 ppm at the low edge to +3 at the high one:
    it reaches its calibrants' range only."""
    return WaveFit(coef=[2.0, -1.0], p=-0.5, domain=(hi ** -0.5, lo ** -0.5), mz_range=(lo, hi),
                   K=1, n=10, n_clipped=0, resid_ppm=0.1, raw_ppm=1.0, loo_ppm=0.1, span_ppm=2.0,
                   share=0.9)


def _quiet(*a, **k):
    return None


# --- massqc: scope, constants, the cap -------------------------------------------

def test_a_trend_reaches_its_calibrants_range_and_both_slack_edges_only():
    fit = _trend()
    lo_in, hi_in = 70.0 * (1 - 10e-6), 300.0 * (1 + 10e-6)       # inside the 20 ppm slack
    lo_out, hi_out = 70.0 * (1 - 30e-6), 300.0 * (1 + 30e-6)
    assert MQ.in_scope(fit, [lo_in, hi_in]).all()
    assert not MQ.in_scope(fit, [lo_out, hi_out, 62.0, 340.0]).any()
    out = MQ.apply_correction(fit, [62.0, 150.0, 340.0])
    assert out[0] == 62.0 and out[2] == 340.0 and out[1] != 150.0


def test_an_edge_calibrant_measured_past_its_theoretical_m_z_is_still_corrected():
    """The scope is the calibrants' THEORETICAL range; the peaks sit at their
    measured m/z, which the axis error moves. A +0.9 ppm edge ion fell just
    outside a strict scope and kept its error."""
    fit = _trend()
    top = 300.0 * (1 + fit.predict(np.array([300.0]))[0] * 1e-6)
    bottom = 70.0 * (1 - 5e-6)
    assert MQ.in_scope(fit, [top, bottom]).all()
    assert abs(MQ.apply_correction(fit, [top])[0] - 300.0) < 1e-6


def test_a_constant_reaches_every_m_z():
    fit = _const(1.5)
    out = MQ.apply_correction(fit, [20.0, 150.0, 900.0, np.nan])
    assert np.allclose(out[:3], np.array([20.0, 150.0, 900.0]) * (1 - 1.5e-6), rtol=0, atol=1e-9)
    assert np.isnan(out[3])


# --- measure_axis ------------------------------------------------------------------

def test_measure_axis_corrects_an_orbitrap_hump_and_marks_the_series(refs):
    rng = np.random.default_rng(31)
    ts = _orbi_batch(rng, refs)
    m0 = ts["mz"].to_numpy(float).copy()
    out, info, table = AB.measure_axis(ts, "NO3", "orbitrap", log=_quiet)
    assert info["applied"] and info["verdict"] == "axis_trend" and table is not None
    fit = MQ.correction(info["qc"])
    inside = MQ.in_scope(fit, m0)
    m1 = out["mz"].to_numpy(float)
    assert (m1[~inside] == m0[~inside]).all()                  # outside: left as is
    assert info["n_corrected"] == int(inside.sum()) and info["n_peaks"] == len(ts)
    # the marker holds exactly what was removed, 0 where nothing was
    d = out[AB.AXIS_COL].to_numpy(float)
    assert (d[~inside] == 0).all() and np.allclose(m1, m0 * (1 - d * 1e-6), rtol=0, atol=1e-9)
    # the reference ions come back to theory, the edge ones included
    lo, hi = info["scope_mz"]
    reag = set(refs.loc[refs["identity"].isin(("nitric acid", "nitric acid . NO3-")), "mz"])
    for th in refs["mz"].to_numpy(float):
        if th in reag or not (lo <= th <= hi):
            continue
        near = np.abs(m1 - th) / th * 1e6 < 6
        assert abs(np.median((m1[near] - th) / th * 1e6)) < 0.15, th
    assert (ts["mz"].to_numpy(float) == m0).all()             # the input is not mutated
    assert 1.5 < info["shift_ppm"]["max"] < 2.2 and info["max_abs_ppm"] < MQ.MAX_CORRECTION_PPM


def test_a_series_fed_back_is_restored_and_corrected_exactly_once(refs):
    """A run's per_file/_batch_ts.parquet carries the corrected axis: fed back
    with --ts it used to read `clean`, so its files' peak tables stayed raw while
    the series was corrected."""
    rng = np.random.default_rng(31)
    ts = _orbi_batch(rng, refs)
    once, info1, _ = AB.measure_axis(ts, "NO3", "orbitrap", log=_quiet)
    back = AB.restore_axis(once, log=_quiet)
    assert AB.AXIS_COL not in back.columns
    assert np.allclose(back["mz"].to_numpy(float), ts["mz"].to_numpy(float), rtol=0, atol=1e-9)
    again, info2, _ = AB.measure_axis(back, "NO3", "orbitrap", log=_quiet)
    assert info2["applied"] and info2["verdict"] == info1["verdict"]
    assert np.allclose(again["mz"].to_numpy(float), once["mz"].to_numpy(float), rtol=0, atol=1e-9)
    assert AB.restore_axis(ts, log=_quiet) is ts               # nothing to restore


def test_measure_axis_leaves_a_clean_axis_alone(refs):
    rng = np.random.default_rng(32)
    ts = _orbi_batch(rng, refs, offset=lambda mz: 0.2)
    out, info, _ = AB.measure_axis(ts, "NO3", "orbitrap", log=_quiet)
    assert not info["applied"] and info["verdict"] == "clean" and "held" not in info
    assert out is ts


def test_a_flat_orbitrap_offset_is_removed_at_every_m_z(refs):
    rng = np.random.default_rng(33)
    ts = _orbi_batch(rng, refs, offset=lambda mz: 1.5)
    out, info, _ = AB.measure_axis(ts, "NO3", "orbitrap", log=_quiet)
    assert info["applied"] and info["verdict"] == "axis_offset"
    assert info["n_corrected"] == info["n_peaks"] and info["frac_outside"] == 0.0


def test_a_tof_is_measured_and_never_corrected(refs):
    rng = np.random.default_rng(34)
    lo, hi = np.sqrt(46.0), np.sqrt(308.0)
    ts = _batch(rng, refs, offset=lambda mz: 1.0 + 10.0 * (np.sqrt(mz) - lo) / (hi - lo), sigma=0.8)
    out, info, table = AB.measure_axis(ts, "NO3", "tof", log=_quiet)
    assert info["verdict"] == "axis_trend" and not info["applied"] and "TOF" in info["held"]
    assert out is ts and table is not None and info["qc"]["rules"]["name"] == "legacy"


def test_a_hold_measures_without_correcting(refs):
    ts = _orbi_batch(np.random.default_rng(35), refs)
    out, info, _ = AB.measure_axis(ts, "NO3", "orbitrap", hold="because", log=_quiet)
    assert info["verdict"] == "axis_trend" and not info["applied"] and info["held"] == "because"
    assert out is ts


def test_a_correction_above_the_cap_is_held(refs):
    ts = _orbi_batch(np.random.default_rng(36), refs, offset=lambda mz: 8.0)
    out, info, _ = AB.measure_axis(ts, "NO3", "orbitrap", log=_quiet)
    assert info["verdict"] == "axis_offset" and not info["applied"] and "above 4" in info["held"]
    assert out is ts


def test_the_probe_is_sized_for_the_analyser(refs, monkeypatch):
    seen = []
    real = MQ.run

    def spy(ts, refs, **kw):
        seen.append((kw["tol_ppm"], kw["tof"], kw["per_instrument"]))
        return real(ts, refs, **kw)

    monkeypatch.setattr(MQ, "run", spy)
    ts = _orbi_batch(np.random.default_rng(37), refs)
    AB.measure_axis(ts, "NO3", "orbitrap", log=_quiet)
    AB.measure_axis(ts, "NO3", "tof", log=_quiet)
    assert seen == [(6.0, False, True), (12.0, True, True)]


def test_a_series_without_times_is_probed_in_spectrum_order(refs):
    ts = _orbi_batch(np.random.default_rng(38), refs).drop(columns=["datetime_utc"])
    out, info, _ = AB.measure_axis(ts, "NO3", "orbitrap", log=_quiet)
    assert info["applied"] and "datetime_utc" not in out.columns


def test_a_failing_probe_never_stops_the_batch(refs, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("probe broke")
    monkeypatch.setattr(MQ, "run", boom)
    ts = _orbi_batch(np.random.default_rng(39), refs)
    out, info, table = AB.measure_axis(ts, "NO3", "orbitrap", log=_quiet)
    assert "probe broke" in info["skipped"] and out is ts and table is None


@pytest.mark.parametrize("ts_ok,klass,reagent,why", [
    (False, "orbitrap", "NO3", "no batch time series"),
    (True, None, "NO3", "instrument class unknown"),
    (True, "orbitrap", "Ur", "no reference-ion table"),
])
def test_measure_axis_says_why_it_did_not_measure(refs, ts_ok, klass, reagent, why):
    ts = _orbi_batch(np.random.default_rng(40), refs) if ts_ok else None
    out, info, table = AB.measure_axis(ts, reagent, klass, log=_quiet)
    assert why in info["skipped"] and not info["applied"] and table is None and out is ts


def test_the_roster_names_the_class_when_no_width_model_was_measured():
    assert AB._roster_class(pd.DataFrame({"instrument_type": ["orbi", "orbi"]})) == "orbitrap"
    assert AB._roster_class(pd.DataFrame({"instrument_type": ["tof"]})) == "tof"
    assert AB._roster_class(pd.DataFrame({"instrument_type": ["api", "tof"]})) == "tof"
    assert AB._roster_class(pd.DataFrame({"instrument_type": ["orbi", "tof"]})) is None
    assert AB._roster_class(pd.DataFrame({"mz": [1.0]})) is None and AB._roster_class(None) is None


# --- the io registry ------------------------------------------------------------------

def _raw_table():
    return pd.DataFrame({"peak_id": ["a", "b", "c"], "mz": [62.0, 150.0, 340.0],
                         "height": [1.0, 2.0, 3.0]})


def test_a_cached_table_is_served_corrected_and_the_cache_stays_raw(tmp_path):
    sid = "s-axis"
    (tmp_path / sid).mkdir()
    _raw_table().to_parquet(tmp_path / sid / "peaks.v2.parquet")
    IO._SCORING_CACHE[sid] = ("stale", {})
    IO._SCORING_TREND[sid] = "stale trend"
    IO.set_axis_correction([sid], _trend().as_dict())
    assert sid not in IO._SCORING_CACHE and sid not in IO._SCORING_TREND   # read off the old axis
    got = IO.fetch_peaks(None, sid, cache_root=tmp_path)
    assert got["mz"].tolist()[0] == 62.0 and got["mz"].tolist()[2] == 340.0     # outside 70-300
    assert abs(got["mz"].iloc[1] - MQ.apply_correction(_trend(), [150.0])[0]) < 1e-12
    assert pd.read_parquet(tmp_path / sid / "peaks.v2.parquet")["mz"].tolist() == [62.0, 150.0, 340.0]
    IO.clear_axis_correction([sid])
    assert IO.fetch_peaks(None, sid, cache_root=tmp_path)["mz"].tolist() == [62.0, 150.0, 340.0]


def test_a_fresh_server_table_is_cached_raw_and_served_corrected(tmp_path):
    sid = "s-fresh"
    client = SimpleNamespace(samples=SimpleNamespace(get_peaks=lambda sample_id, matches: _raw_table()))
    IO.set_axis_correction([sid], _const(2.0))
    got = IO.fetch_peaks(client, sid, cache_root=tmp_path)
    assert np.allclose(got["mz"], np.array([62.0, 150.0, 340.0]) * (1 - 2e-6), rtol=0, atol=1e-9)
    assert pd.read_parquet(tmp_path / sid / "peaks.v2.parquet")["mz"].tolist() == [62.0, 150.0, 340.0]


def test_a_registered_offline_sample_is_corrected_too():
    sid = "offline-axis"
    IO.register_offline_sample(sid, _raw_table(), mechanisms=["[M-H]-"])
    try:
        assert IO.fetch_peaks(None, sid)["mz"].tolist() == [62.0, 150.0, 340.0]
        IO.set_axis_correction([sid], _const(1.0))
        assert np.allclose(IO.fetch_peaks(None, sid)["mz"], np.array([62.0, 150.0, 340.0]) * (1 - 1e-6),
                           rtol=0, atol=1e-9)
    finally:
        IO.unregister_offline_sample(sid)


def test_clearing_one_runs_samples_leaves_anothers():
    """The MCP server runs jobs on threads of one process: a run that ends must
    not clear a concurrent run's correction."""
    IO.set_axis_correction(["a1", "a2"], _const(1.0))
    IO.set_axis_correction(["b1"], _const(2.0))
    IO.clear_axis_correction(["a1", "a2"])
    assert IO.axis_correction("a1") is None and IO.axis_correction("b1").coef == [2.0]
    IO.set_axis_correction(["b1"], None)
    assert IO.axis_correction("b1") is None


def test_a_spawned_worker_gets_the_parents_correction_or_none():
    AB._worker_init("ambient-air", [], {}, None, None, (["s1", "s2"], _const().as_dict()))
    assert IO.axis_correction("s1") is not None and IO.axis_correction("s2") is not None
    AB._worker_init("ambient-air", [], {}, None, None, None)
    assert IO.axis_correction("s1") is None


def test_the_pool_hands_the_correction_to_its_workers():
    """The worker pool's initargs carry the correction: a source-level guard on
    the one line that ships it (a spawned worker shares no memory)."""
    import inspect
    assert "P.registry_extras(), axis_args)" in inspect.getsource(AB.run)


# --- a whole batch, offline -------------------------------------------------------------

def _run_offline(ts, tmp_path, *, tables_from=None, roster=None, **kw):
    """assign_batch.run over `ts`, every sample served from memory, on an
    Orbitrap-class width model (a declared resolving power). The served per-file
    tables are built from `tables_from` (default `ts`: the server's axis);
    `roster` adds an instrument_type column to the peaks table."""
    from peaky.batch import tracefirst as TFT
    tabs = {}
    for sid, g in (ts if tables_from is None else tables_from).groupby("sample_item_id"):
        t = pd.DataFrame({"sample_item_id": sid, "peak_id": [f"{sid}-{i}" for i in range(len(g))],
                          "mz": g["mz"].to_numpy(), "sparsity": 0.0,
                          "area": g["height"].to_numpy() * 1e-3, "height": g["height"].to_numpy()})
        for c in TFT.MATCH_COLS:
            t[c] = None
        tabs[sid] = t
    for sid, t in tabs.items():
        IO.register_offline_sample(sid, t, ["[M-H]-", "[M+NO3]-"])
    real_connect = IO.connect
    IO.connect = lambda *a, **k: None
    peaks = ts if roster is None else ts.assign(instrument_type=roster)
    try:
        return AB.run(peaks=peaks, ts_peaks=ts, reagent="NO3", batch="axis batch",
                      out_dir=str(tmp_path), residual=False, n_jobs=1, k_min=2, k_max=3,
                      resolving_power=250000.0, log=_quiet, **kw), tabs
    finally:
        IO.connect = real_connect
        for sid in tabs:
            IO.unregister_offline_sample(sid)


def test_a_whole_batch_assigns_every_file_on_the_corrected_axis(refs, tmp_path):
    ts = _orbi_batch(np.random.default_rng(41), refs)
    res, tabs = _run_offline(ts, tmp_path)
    ax = res["summary"]["mass_axis"]
    assert ax["mode"] == "auto" and ax["applied"] and ax["verdict"] == "axis_trend"
    fit = WaveFit(**ax["wave"])
    assert len(res["sample_ids"]) >= 2
    # every assigned file's ledger holds the corrected m/z (the serial path reads
    # its tables through the parent's registry)
    for sid in res["sample_ids"]:
        led = pd.read_csv(tmp_path / "per_file" / f"{sid}_ledger.csv")
        raw = tabs[sid].set_index("peak_id")["mz"]
        got = led.drop_duplicates("peak_id").set_index("peak_id")["mz"]
        assert len(got) > 20, sid
        want = pd.Series(MQ.apply_correction(fit, raw.reindex(got.index).to_numpy()), index=got.index)
        assert np.allclose(got, want, rtol=0, atol=1e-7), sid
    # the series it returns and writes is the corrected one, marked
    assert AB.AXIS_COL in res["ts_peaks"].columns
    assert AB.AXIS_COL in pd.read_parquet(tmp_path / "per_file" / "_batch_ts.parquet").columns
    assert (tmp_path / "tables" / "mass_axis.csv").exists()
    assert json.load(open(tmp_path / "batch_summary.json"))["mass_axis"]["applied"]
    # and the correction is gone once the run returns
    assert all(IO.axis_correction(sid) is None for sid in tabs)


def test_off_and_trace_first_leave_the_axis_alone(refs, tmp_path):
    ts = _orbi_batch(np.random.default_rng(42), refs)
    res, _ = _run_offline(ts, tmp_path / "off", mass_axis="off")
    ax = res["summary"]["mass_axis"]
    assert ax["mode"] == "off" and not ax["applied"] and "off" in ax["skipped"]
    assert AB.AXIS_COL not in res["ts_peaks"].columns
    assert not (tmp_path / "off" / "tables" / "mass_axis.csv").exists()


def test_server_side_scoring_holds_the_correction(refs, tmp_path, monkeypatch):
    monkeypatch.setattr(IO, "_local_scoring_enabled", lambda: False)
    ts = _orbi_batch(np.random.default_rng(43), refs)
    seen = {}
    real = AB.measure_axis

    def spy(ts, reagent, klass, *, hold=None, log=print):
        seen["hold"] = hold
        return real(ts, reagent, klass, hold=hold, log=log)

    monkeypatch.setattr(AB, "measure_axis", spy)
    # the measurement alone: stop the run right after it
    monkeypatch.setattr(AB.SS, "select_cover_samples",
                        lambda *a, **k: (_ for _ in ()).throw(SystemExit("measured")))
    with pytest.raises(SystemExit):
        _run_offline(ts, tmp_path)
    assert "hold" in seen and "server-side scoring" in seen["hold"]


def test_run_refuses_an_unknown_mode_before_any_server_call(tmp_path, monkeypatch):
    monkeypatch.setattr(IO, "connect", lambda *a, **k: (_ for _ in ()).throw(AssertionError("connect")))
    with pytest.raises(ValueError, match="mass_axis"):
        AB.run(peaks=pd.DataFrame({"sample_item_id": ["s"], "mz": [100.0], "height": [1.0]}),
               reagent="NO3", out_dir=str(tmp_path), mass_axis="sometimes",
               sample_ids=["s"], resolving_power="none")


# --- the pipeline ---------------------------------------------------------------------------

def _stub_pipeline(monkeypatch, ts, *, raise_in_run=False):
    from peaky import pipeline as PL
    seen = {}

    def fake_run(**kw):
        seen.update(kw)
        IO.set_axis_correction(ts["sample_item_id"].unique(), _const())
        if raise_in_run:
            raise RuntimeError("assign failed")
        return {"summary": {"mass_axis": {"applied": False}}, "ts_peaks": None}

    monkeypatch.setattr(IO, "connect", lambda *a, **k: None)
    monkeypatch.setattr(IO, "resolve_batch",
                        lambda c, b, dataset=None: SimpleNamespace(id="bid", name="axis batch"))
    monkeypatch.setattr(AB, "run", fake_run)
    monkeypatch.setattr(PL, "generate_report", lambda ctx, ts, **k: {"report_ts": ts})
    return PL, seen


def test_the_pipeline_passes_the_mode_and_clears_its_samples_even_on_failure(refs, tmp_path, monkeypatch):
    ts = _orbi_batch(np.random.default_rng(44), refs)
    IO.set_axis_correction(["someone-else"], _const())
    PL, seen = _stub_pipeline(monkeypatch, ts, raise_in_run=True)
    with pytest.raises(RuntimeError, match="assign failed"):
        PL.run_batch(batch="axis batch", reagent="NO3", base_out=str(tmp_path), ts=ts,
                     mass_axis="off", do_report=False, log=_quiet)
    assert seen["mass_axis"] == "off"
    assert all(IO.axis_correction(s) is None for s in ts["sample_item_id"].unique())
    assert IO.axis_correction("someone-else") is not None          # another run's, kept


def test_the_pipeline_restores_a_fed_back_series_before_the_run(refs, tmp_path, monkeypatch):
    ts = _orbi_batch(np.random.default_rng(45), refs)
    marked, _, _ = AB.measure_axis(ts, "NO3", "orbitrap", log=_quiet)
    PL, seen = _stub_pipeline(monkeypatch, ts)
    PL.run_batch(batch="axis batch", reagent="NO3", base_out=str(tmp_path), ts=marked,
                 do_report=False, log=_quiet)
    fed = seen["ts_peaks"]
    assert AB.AXIS_COL not in fed.columns
    assert np.allclose(fed["mz"].to_numpy(float), ts["mz"].to_numpy(float), rtol=0, atol=1e-9)


def test_the_report_reads_the_corrected_series_and_the_run_keeps_the_raw_one(tmp_path, refs):
    from peaky import pipeline as PL
    raw = _batch(np.random.default_rng(46), refs, n=3, sigma=0.5)
    fixed = raw.assign(mz=raw["mz"] * (1 - 2e-6))
    ctx = PL.RunContext.__new__(PL.RunContext)
    ctx.out_dir, ctx.tag, ctx.ts_path = str(tmp_path), "NO3", None
    res = {"summary": {"mass_axis": {"applied": True}}, "ts_peaks": fixed}
    assert PL._report_ts(ctx, raw, res) is fixed
    assert pd.read_parquet(ctx.ts_path)["mz"].tolist() == raw["mz"].tolist()
    # not applied (or no series handed back): the report reads what it was given
    ctx.ts_path = None
    assert PL._report_ts(ctx, raw, {"summary": {"mass_axis": {"applied": False}}, "ts_peaks": fixed}) is raw
    assert PL._report_ts(ctx, raw, {"summary": {}}) is raw and ctx.ts_path is None


def test_the_run_manifest_records_the_axis(monkeypatch, tmp_path, refs):
    from peaky import pipeline as PL
    from peaky.reporting import provenance as PV
    ts = _orbi_batch(np.random.default_rng(47), refs)
    _stub_pipeline(monkeypatch, ts)
    rec, rep = {}, {}
    fixed = ts.assign(mz=ts["mz"] * (1 - 1e-6))
    monkeypatch.setattr(AB, "run", lambda **kw: {"summary": {"mass_axis": {"applied": True, "verdict": "x"}},
                                                  "ts_peaks": fixed})
    monkeypatch.setattr(PL, "generate_report", lambda ctx, ts, **k: rep.update(ts=ts) or {})
    monkeypatch.setattr(PV, "record_run", lambda **kw: rec.update(kw))
    PL.run_batch(batch="axis batch", reagent="NO3", base_out=str(tmp_path), ts=ts,
                 do_report=False, log=_quiet)
    assert rec["counts"]["mass_axis"] == {"applied": True, "verdict": "x"}
    assert rep["ts"] is fixed                              # the figures read the corrected series


# --- the CLI ----------------------------------------------------------------------------------

def test_the_cli_flag_defaults_to_auto_and_reaches_both_pipelines(monkeypatch):
    from peaky import cli
    from peaky import pipeline as PL
    p = cli.build_parser()
    assert p.parse_args(["batch", "--batch", "b"]).mass_axis == "auto"
    assert p.parse_args(["pool", "--batches", "b"]).mass_axis == "auto"
    with pytest.raises(SystemExit):
        p.parse_args(["batch", "--batch", "b", "--mass-axis", "maybe"])
    seen = {}

    def fake(**kw):
        seen.update(kw)
        raise SystemExit(0)

    monkeypatch.setattr(cli, "_require_creds", lambda *a, **k: None)
    monkeypatch.setattr(PL, "run_batch", fake)
    monkeypatch.setattr(PL, "run_pooled_batches", fake)
    with pytest.raises(SystemExit):
        cli.main(["batch", "--batch", "b", "--mass-axis", "off", "--no-progress"])
    assert seen.pop("mass_axis") == "off"
    with pytest.raises(SystemExit):
        cli.main(["pool", "--batches", "b", "--mass-axis", "off", "--no-progress"])
    assert seen.pop("mass_axis") == "off"


def test_a_pool_of_several_batches_is_held_restored_and_cleaned_up(refs, tmp_path, monkeypatch):
    """A pool: the fed-back series is restored before the pool trims its columns,
    several batches are measured as one but not corrected (one wave would correct
    each batch by the others' axis), and the pool's samples are cleared even when
    the assignment fails."""
    from peaky import pipeline as PL
    ts = _orbi_batch(np.random.default_rng(48), refs)
    ids = sorted(ts["sample_item_id"].unique())
    ts["sample_batch_name"] = np.where(ts["sample_item_id"].isin(ids[:4]), "batch one", "batch two")
    marked, _, _ = AB.measure_axis(ts, "NO3", "orbitrap", log=_quiet)
    seen = {}

    def fake_run(**kw):
        seen.update(kw)
        IO.set_axis_correction(ids, _const())
        raise RuntimeError("assign failed")

    monkeypatch.setattr(IO, "connect", lambda *a, **k: None)
    monkeypatch.setattr(AB, "run", fake_run)
    with pytest.raises(RuntimeError, match="assign failed"):
        PL.run_pooled_batches(batches="batch", reagent="NO3", base_out=str(tmp_path), ts=marked,
                              do_report=False, per_group_reports=False, log=_quiet)
    assert "2 batches" in seen["mass_axis_hold"]
    fed = seen["ts_peaks"]
    assert AB.AXIS_COL not in fed.columns
    raw = ts.set_index(["sample_item_id", "height", "datetime_utc"])["mz"]
    got = fed.set_index(["sample_item_id", "height", "datetime_utc"])["mz"]
    assert np.allclose(got.sort_index().to_numpy(), raw.sort_index().to_numpy(), rtol=0, atol=1e-9)
    assert all(IO.axis_correction(s) is None for s in ids)



# --- refute round 2 ---------------------------------------------------------------------------

def _ledger_mz(tmp_path, sid):
    led = pd.read_csv(tmp_path / "per_file" / f"{sid}_ledger.csv")
    return led.drop_duplicates("peak_id").set_index("peak_id")["mz"]


def test_the_cap_is_checked_across_the_range_and_both_signs(refs):
    # a hump peaking at ~6 ppm mid-range whose ends both sit under 4 ppm
    mid = _orbi_batch(np.random.default_rng(50), refs, offset=lambda mz: 3.2 * _hump(mz))
    _, info, _ = AB.measure_axis(mid, "NO3", "orbitrap", log=_quiet)
    assert info["verdict"] == "axis_trend" and not info["applied"] and "above 4" in info["held"]
    neg = _orbi_batch(np.random.default_rng(51), refs, offset=lambda mz: -8.0)
    _, info, _ = AB.measure_axis(neg, "NO3", "orbitrap", log=_quiet)
    assert info["verdict"] == "axis_offset" and not info["applied"] and "above 4" in info["held"]


def test_measure_axis_skips_a_series_without_its_columns(refs):
    ts = _orbi_batch(np.random.default_rng(52), refs).drop(columns=["sample_item_id"])
    out, info, table = AB.measure_axis(ts, "NO3", "orbitrap", log=_quiet)
    assert "lacks" in info["skipped"] and out is ts and table is None


def test_a_tof_stays_a_tof_whatever_resolving_power_it_is_given(refs, tmp_path):
    """A TOF declared (or measured) at R >= the Orbitrap bar read as an Orbitrap
    from its width model and was corrected; the roster's TOF now wins."""
    assert AB._axis_class("orbitrap", "tof") == "tof" and AB._axis_class("tof", "orbitrap") == "tof"
    assert AB._axis_class("orbitrap", None) == "orbitrap" and AB._axis_class(None, "orbitrap") == "orbitrap"
    assert AB._axis_class(None, None) is None
    # a swing big enough for the TOF rules to call a trend (and to prescribe a fix)
    ts = _orbi_batch(np.random.default_rng(53), refs, offset=lambda mz: 3.2 * _hump(mz))
    res, _ = _run_offline(ts, tmp_path, roster="tof")
    ax = res["summary"]["mass_axis"]
    assert ax["instrument"] == "tof" and ax["verdict"] == "axis_trend"
    assert not ax["applied"] and "TOF" in ax["held"]


def test_the_run_honours_its_hold(refs, tmp_path):
    ts = _orbi_batch(np.random.default_rng(54), refs)
    res, tabs = _run_offline(ts, tmp_path, mass_axis_hold="held by the caller")
    ax = res["summary"]["mass_axis"]
    assert ax["verdict"] == "axis_trend" and not ax["applied"] and ax["held"] == "held by the caller"
    sid = res["sample_ids"][0]
    raw = tabs[sid].set_index("peak_id")["mz"]
    got = _ledger_mz(tmp_path, sid)
    assert np.allclose(got, raw.reindex(got.index), rtol=0, atol=1e-9)        # raw axis


def test_the_run_restores_a_fed_back_series_and_corrects_its_files_once(refs, tmp_path):
    raw = _orbi_batch(np.random.default_rng(55), refs)
    marked, first, _ = AB.measure_axis(raw, "NO3", "orbitrap", log=_quiet)
    res, tabs = _run_offline(marked, tmp_path, tables_from=raw)
    ax = res["summary"]["mass_axis"]
    assert ax["applied"]
    fit = WaveFit(**ax["wave"])
    for sid in res["sample_ids"]:
        src = tabs[sid].set_index("peak_id")["mz"]
        got = _ledger_mz(tmp_path, sid)
        want = MQ.apply_correction(fit, src.reindex(got.index).to_numpy())
        assert np.allclose(got.to_numpy(), want, rtol=0, atol=1e-7), sid
    # the series the run used is the raw one corrected once, not twice
    ts_used = res["ts_peaks"]
    assert np.allclose(np.sort(ts_used["mz"].to_numpy()), np.sort(marked["mz"].to_numpy()), rtol=0, atol=1e-7)


def test_a_run_clears_its_own_samples_only_and_starts_from_a_clean_axis(refs, tmp_path):
    ts = _orbi_batch(np.random.default_rng(56), refs)
    ids = sorted(ts["sample_item_id"].unique())
    IO.set_axis_correction(["another-run"], _const(1.0))
    IO.set_axis_correction(ids, _const(3.0))           # stale, from an earlier failed run
    res, tabs = _run_offline(ts, tmp_path, mass_axis="off")
    sid = res["sample_ids"][0]
    raw = tabs[sid].set_index("peak_id")["mz"]
    got = _ledger_mz(tmp_path, sid)
    assert np.allclose(got, raw.reindex(got.index), rtol=0, atol=1e-9)        # the stale one is gone
    assert IO.axis_correction("another-run") is not None                          # not this run's
    assert all(IO.axis_correction(s) is None for s in ids)


def test_clearing_drops_the_cached_scoring_but_keeps_an_inherited_trend():
    IO._SCORING_CACHE["s-a"] = ("stale", {})
    IO._SCORING_CACHE["s-b"] = ("stale", {})
    IO._SCORING_TREND["s-b"] = "inherited trend"
    IO._SCORING_TREND_INHERITED.add("s-b")
    try:
        IO.clear_axis_correction(["s-a", "s-b"])
        assert "s-a" not in IO._SCORING_CACHE and "s-b" not in IO._SCORING_CACHE
        assert IO._SCORING_TREND.get("s-b") == "inherited trend"
    finally:
        IO._SCORING_TREND.pop("s-b", None)
        IO._SCORING_TREND_INHERITED.discard("s-b")


def _pool(monkeypatch, tmp_path, ts, *, result=None, group_by="sample_batch_name"):
    from peaky import pipeline as PL
    from peaky.reporting import provenance as PV
    seen, rep = {}, {}

    def fake_run(**kw):
        seen.update(kw)
        if result is None:
            raise RuntimeError("assign failed")
        return result

    monkeypatch.setattr(IO, "connect", lambda *a, **k: None)
    monkeypatch.setattr(AB, "run", fake_run)
    monkeypatch.setattr(PL, "generate_report", lambda ctx, ts, **k: rep.update(ts=ts) or {})
    monkeypatch.setattr(PV, "record_run", lambda **kw: None)
    kw = dict(batches="batch", reagent="NO3", base_out=str(tmp_path), ts=ts, group_by=group_by,
              do_report=False, per_group_reports=False, log=_quiet)
    if result is None:
        with pytest.raises(RuntimeError, match="assign failed"):
            PL.run_pooled_batches(**kw)
    else:
        PL.run_pooled_batches(**kw)
    return seen, rep


def test_a_one_batch_pool_is_corrected_and_its_figures_read_the_corrected_series(refs, tmp_path, monkeypatch):
    ts = _orbi_batch(np.random.default_rng(57), refs).assign(sample_batch_name="only batch")
    fixed = ts.assign(mz=ts["mz"] * (1 - 1e-6))
    seen, rep = _pool(monkeypatch, tmp_path, ts, result={
        "summary": {"mass_axis": {"applied": True}}, "ts_peaks": fixed, "residual_samples": None})
    assert seen["mass_axis_hold"] is None
    assert rep["ts"] is fixed


def test_the_hold_counts_batches_not_report_groups(refs, tmp_path, monkeypatch):
    ts = _orbi_batch(np.random.default_rng(58), refs)
    ids = sorted(ts["sample_item_id"].unique())
    ts["sample_batch_name"] = np.where(ts["sample_item_id"].isin(ids[:4]), "batch one", "batch two")
    ts["site"] = "one site"
    seen, _ = _pool(monkeypatch, tmp_path, ts, group_by="site")
    assert seen["mass_axis_hold"] and "2 batches" in seen["mass_axis_hold"]


def test_a_failed_pool_clears_its_samples_only(refs, tmp_path, monkeypatch):
    ts = _orbi_batch(np.random.default_rng(59), refs).assign(sample_batch_name="only batch")
    ids = sorted(ts["sample_item_id"].unique())
    IO.set_axis_correction(["another-run"], _const(1.0))
    IO.set_axis_correction(ids, _const(2.0))
    _pool(monkeypatch, tmp_path, ts)
    assert all(IO.axis_correction(s) is None for s in ids)
    assert IO.axis_correction("another-run") is not None


def test_the_report_series_never_rewrites_a_given_input(tmp_path, refs):
    from peaky import pipeline as PL
    raw = _batch(np.random.default_rng(60), refs, n=3, sigma=0.5)
    ctx = PL.RunContext.__new__(PL.RunContext)
    ctx.out_dir, ctx.tag, ctx.ts_path = str(tmp_path), "NO3", "/given/input.parquet"
    fixed = raw.assign(mz=raw["mz"] * (1 - 2e-6))
    assert PL._report_ts(ctx, raw, {"summary": {"mass_axis": {"applied": True}}, "ts_peaks": fixed}) is fixed
    assert ctx.ts_path == "/given/input.parquet"


def test_mass_qc_judges_the_servers_axis_of_a_corrected_parquet(tmp_path, monkeypatch, refs):
    from peaky import cli
    raw = _orbi_batch(np.random.default_rng(61), refs)
    marked, _, _ = AB.measure_axis(raw, "NO3", "orbitrap", log=_quiet)
    pq = tmp_path / "ts.parquet"
    marked.to_parquet(pq)
    monkeypatch.setattr(cli, "_require_creds", lambda *a, **k: (_ for _ in ()).throw(AssertionError("creds")))
    cli.main(["mass-qc", "--ts", str(pq), "--reagent", "NO3", "--orbitrap", "--out", str(tmp_path)])
    assert json.load(open(tmp_path / "mass_qc.json"))["verdict"] == "axis_trend"
