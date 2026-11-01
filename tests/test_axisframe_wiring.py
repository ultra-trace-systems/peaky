"""The band-step frame's plumbing (batch.axisframe in front of the lock model): the
fingerprint fetch and its cache, the per-item registry and the spawned workers, a whole
offline batch corrected by the frame, byte-identity with a run that read no fingerprint,
the gate holding a wrong grid, a mixed frame applied per item and put back per source, and
the CLI flags down to assign_batch.run. Every batch and document here is synthetic."""
import json
import os
import pickle
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from mascope_sdk.exceptions import NotFoundError

from peaky.batch import assign_batch as AB
from peaky.batch import axisframe as AF
from peaky.batch import axislock as AL
from peaky.batch import massqc as MQ
from peaky.chem import reference_ions as RI
from peaky.io import io_mascope as IO
from tests.test_axisframe import (B0, C0, K_MID, PERIOD, PHASE, SIDE_PPM, STEP_MZ, STEP_PPM, STEP_TOL, _document,
                                  _item, _k_at, _spline_inner, _two_document_batch)
from tests.test_axislock import _batch, _smooth, _truth
from tests.test_massqc import _orbi_batch


@pytest.fixture(autouse=True)
def _no_axis_left_behind():
    IO.clear_axis_correction()
    yield
    IO.clear_axis_correction()


@pytest.fixture(scope="module")
def refs():
    return RI.nitrate()


def _quiet(*a, **k):
    return None


# --- an offline batch run ------------------------------------------------------------------

RUN_KW = {"reagent": "NO3", "batch": "axis batch", "residual": False, "n_jobs": 1, "k_min": 2, "k_max": 2,
          "resolving_power": 250000.0}
MECHANISMS = ["[M-H]-", "[M+NO3]-"]


def _tables(ts):
    """Per sample the peak table the run is served (the server's axis), from the series."""
    from peaky.batch import tracefirst as TFT
    tabs = {}
    for sid, g in ts.groupby("sample_item_id"):
        t = pd.DataFrame({"sample_item_id": sid, "peak_id": [f"{sid}-{i}" for i in range(len(g))],
                          "mz": g["mz"].to_numpy(), "sparsity": 0.0,
                          "area": g["height"].to_numpy() * 1e-3, "height": g["height"].to_numpy()})
        for c in TFT.MATCH_COLS:
            t[c] = None
        tabs[sid] = t
    return tabs


def _run_offline(ts, out, *, client=None, roster=None, log=_quiet, **kw):
    """assign_batch.run over `ts`, every sample served from memory, on an Orbitrap-class
    width model (a declared resolving power); `client` is what IO.connect returns."""
    tabs = _tables(ts)
    for sid, t in tabs.items():
        IO.register_offline_sample(sid, t, MECHANISMS)
    real_connect = IO.connect
    IO.connect = lambda *a, **k: client
    peaks = ts if roster is None else ts.assign(instrument_type=roster)
    try:
        return AB.run(peaks=peaks, ts_peaks=ts, out_dir=str(out), log=log, **RUN_KW, **kw), tabs
    finally:
        IO.connect = real_connect
        for sid in tabs:
            IO.unregister_offline_sample(sid)


# the same run, self-contained, for another interpreter (a checkout of main): no test imports
_DRIVER = r'''
import json, sys
import pandas as pd
from peaky.batch import assign_batch as AB
from peaky.batch import tracefirst as TFT
from peaky.io import io_mascope as IO
ts = pd.read_parquet(sys.argv[1]); out = sys.argv[2]; kw = json.loads(sys.argv[3])
tabs = {}
for sid, g in ts.groupby("sample_item_id"):
    t = pd.DataFrame({"sample_item_id": sid, "peak_id": [f"{sid}-{i}" for i in range(len(g))],
                      "mz": g["mz"].to_numpy(), "sparsity": 0.0,
                      "area": g["height"].to_numpy() * 1e-3, "height": g["height"].to_numpy()})
    for c in TFT.MATCH_COLS:
        t[c] = None
    tabs[sid] = t
for sid, t in tabs.items():
    IO.register_offline_sample(sid, t, ["[M-H]-", "[M+NO3]-"])
IO.connect = lambda *a, **k: None
AB.run(peaks=ts, ts_peaks=ts, out_dir=out, log=lambda *a, **k: None, **kw)
'''


def _same_outputs(a: Path, b: Path, *, ignore_frame: bool):
    """Two run folders hold the same outputs: batch_summary.json as text (its wall clock
    dropped; with `ignore_frame`, mass_axis.frame dropped from either), the merged
    ledger, every per-file ledger, the stamped series and every table, byte for byte."""
    sa = json.loads((a / "batch_summary.json").read_text())
    sb = json.loads((b / "batch_summary.json").read_text())
    for s in (sa, sb):
        s.pop("elapsed_s", None)
        if ignore_frame:
            (s.get("mass_axis") or {}).pop("frame", None)
    assert json.dumps(sa) == json.dumps(sb), [k for k in sa if json.dumps(sa[k]) != json.dumps(sb.get(k))]
    files = (["merged_ledger.csv"] + sorted(str(p.relative_to(a)) for p in (a / "per_file").iterdir())
             + sorted(str(p.relative_to(a)) for p in (a / "tables").iterdir()))
    for rel in files:
        assert (b / rel).exists(), rel
        if (a / rel).read_bytes() != (b / rel).read_bytes():
            assert rel.endswith(".parquet"), rel
            assert pd.read_parquet(a / rel).equals(pd.read_parquet(b / rel)), rel
    assert {p.name for p in (a / "tables").iterdir()} == {p.name for p in (b / "tables").iterdir()}
    assert {p.name for p in (a / "per_file").iterdir()} == {p.name for p in (b / "per_file").iterdir()}


# --- the synthetic framed batch -------------------------------------------------------------

#: a STEP_PPM band at m/z ~294-315 and SIDE_PPM on every band above it in m/z, on the
#: smooth hump of test_axislock
STEPS = {K_MID: STEP_PPM, **{k: SIDE_PPM for k in range(18, K_MID)}}
#: the stepped band's d under STEPS: its offset from the local trend, which runs between the
#: levels on its two sides (0 below it in m/z, SIDE_PPM above)
D_MID = STEP_PPM - 0.5 * SIDE_PPM


def _framed_batch():
    """Four items from two documents of ONE calibration (grids 0.1 kHz apart, different
    unmeasured bands), their frame, the physical axis (hump + the frame's own band steps,
    so the steps sit exactly at the grid's edges) and the batch's time series."""
    docs = _two_document_batch(STEPS, n_items=4)
    frame, rec = AF.FrameSet.from_documents(docs, polarity="-")
    assert frame is not None, rec
    it = frame.for_item("s0")

    def axis(mz):
        return _smooth(mz) + it.d_of(mz)

    rng = np.random.default_rng(7)
    truth = [i for i in _truth(0) if rng.random() < 0.6]
    ts, truth = _batch(axis, n_files=4, seed=0, truth=truth)
    return docs, frame, axis, ts, truth


def _write_docs(docs, ddir: Path) -> Path:
    ddir.mkdir(parents=True, exist_ok=True)
    for sid, doc in docs.items():
        (ddir / f"{sid}.json").write_text(json.dumps(doc))
    return ddir


@pytest.fixture(scope="module")
def batch():
    return _framed_batch()


@pytest.fixture(scope="module")
def framed_run(batch, tmp_path_factory):
    docs, frame, axis, ts, truth = batch
    root = tmp_path_factory.mktemp("framed")
    ddir = _write_docs(docs, root / "docs")
    logs = []
    res, tabs = _run_offline(ts, root / "run", log=logs.append, axis_fingerprint_dir=str(ddir))
    return root / "run", res, tabs, logs


@pytest.fixture(scope="module")
def off_run(batch, tmp_path_factory):
    docs, frame, axis, ts, truth = batch
    root = tmp_path_factory.mktemp("off")
    res, tabs = _run_offline(ts, root / "run", axis_fingerprints="off")
    return root / "run", res, tabs


@pytest.fixture(scope="module")
def missing_run(batch, tmp_path_factory):
    """Fingerprints looked for (an Orbitrap, the lock path) but none served: no client."""
    docs, frame, axis, ts, truth = batch
    root = tmp_path_factory.mktemp("missing")
    logs = []
    res, tabs = _run_offline(ts, root / "run", log=logs.append)
    return root / "run", res, tabs, logs


@pytest.fixture(scope="module")
def held_run(batch, tmp_path_factory):
    """Documents whose grid is half a period off: the frame leaves the steps where they are."""
    docs, frame, axis, ts, truth = batch
    shifted = {sid: _document(steps=STEPS, phase=PHASE + PERIOD / 2, item=_item(sid)) for sid in docs}
    root = tmp_path_factory.mktemp("held")
    ddir = _write_docs(shifted, root / "docs")
    logs = []
    res, tabs = _run_offline(ts, root / "run", log=logs.append, axis_fingerprint_dir=str(ddir))
    return root / "run", res, tabs, logs


@pytest.fixture(scope="module")
def malformed_run(batch, tmp_path_factory):
    """One document of four with `item.scans` a count where the schema has a list."""
    docs, frame, axis, ts, truth = batch
    bad = json.loads(json.dumps(docs))
    bad["s1"]["item"]["scans"] = 79
    root = tmp_path_factory.mktemp("malformed")
    ddir = _write_docs(bad, root / "docs")
    logs = []
    res, tabs = _run_offline(ts, root / "run", log=logs.append, axis_fingerprint_dir=str(ddir))
    return root / "run", res, tabs, logs


# --- the fetch and its cache -----------------------------------------------------------------

class _Boom:
    """A client that must never be touched."""

    def __getattr__(self, name):
        raise AssertionError(f"the server was asked ({name})")


def test_fetch_prefers_the_sdk_method_caches_the_served_json_raw_and_reads_it_back(tmp_path):
    doc = _document(item=_item("s0"))
    calls = []

    def served(sid):
        calls.append(sid)
        return json.loads(json.dumps(doc))

    def route(path, params=None):
        raise AssertionError("the generic route was used although the SDK has the method")

    client = SimpleNamespace(samples=SimpleNamespace(get_axis_fingerprint=served, _get=route))
    got = IO.fetch_axis_fingerprint(client, "s0", cache_root=tmp_path)
    assert got == doc and calls == ["s0"]
    cfile = tmp_path / "s0" / IO.AXIS_FINGERPRINT_CACHE
    assert json.loads(cfile.read_text()) == doc                        # the served JSON, as served
    assert IO.fetch_axis_fingerprint(client, "s0", cache_root=tmp_path) == doc and calls == ["s0"]   # from the cache
    assert IO.fetch_axis_fingerprint(client, "s0", cache_root=tmp_path, use_cache=False) == doc
    assert calls == ["s0", "s0"]
    assert IO.axis_fingerprint_source(client) == "sdk"


def test_fetch_falls_back_to_the_route_on_an_older_sdk_and_never_raises(tmp_path):
    doc = _document(item=_item("s1"))
    paths = []

    def route(path, params=None):
        paths.append(path)
        return json.loads(json.dumps(doc))

    old_sdk = SimpleNamespace(samples=SimpleNamespace(_get=route))
    assert IO.fetch_axis_fingerprint(old_sdk, "s1", cache_root=tmp_path) == doc
    assert paths == ["samples/s1/axis_fingerprint"]
    assert IO.axis_fingerprint_source(old_sdk) == "route"
    said = []

    def not_found(sid):
        raise NotFoundError("Not Found", status_code=404)

    gone = SimpleNamespace(samples=SimpleNamespace(get_axis_fingerprint=not_found))
    assert IO.fetch_axis_fingerprint(gone, "s2", cache_root=tmp_path, log=said.append) is None
    assert len(said) == 1 and "s2" in said[0] and "NotFoundError" in said[0] and "404" in said[0]
    assert not (tmp_path / "s2").exists()

    def broken(sid):
        raise RuntimeError("the network went away")

    assert IO.fetch_axis_fingerprint(SimpleNamespace(samples=SimpleNamespace(get_axis_fingerprint=broken)), "s3",
                                     cache_root=tmp_path, log=said.append) is None
    assert "RuntimeError" in said[-1]
    # a document of status missing is handed back (the reader refuses it with its status) and NOT
    # cached: the server's backfill may serve a table later
    missing = {"schema": AF.SCHEMA, "version": 1, "status": "missing"}
    client = SimpleNamespace(samples=SimpleNamespace(get_axis_fingerprint=lambda sid: dict(missing)))
    assert IO.fetch_axis_fingerprint(client, "s4", cache_root=tmp_path) == missing
    assert not (tmp_path / "s4").exists()
    assert "status 'missing'" in AF.load_document(missing, polarity="-").why
    # no client, a client without a samples resource, a non-document answer: None, with a reason
    assert IO.fetch_axis_fingerprint(None, "s5", cache_root=tmp_path, log=said.append) is None
    assert "no server connection" in said[-1] and IO.axis_fingerprint_source(None) == "none"
    assert IO.fetch_axis_fingerprint(SimpleNamespace(), "s6", cache_root=tmp_path, log=said.append) is None
    assert IO.fetch_axis_fingerprint(SimpleNamespace(samples=SimpleNamespace(get_axis_fingerprint=lambda sid: [1])),
                                     "s7", cache_root=tmp_path, log=said.append) is None
    assert "not a document" in said[-1]


def test_a_directory_of_documents_is_read_instead_of_the_server(tmp_path):
    doc = _document(item=_item("s0"))
    ddir = _write_docs({"s0": doc}, tmp_path / "docs")
    (ddir / "bad.json").write_text("{not json")
    assert IO.fetch_axis_fingerprint(_Boom(), "s0", directory=str(ddir), cache_root=tmp_path / "cache") == doc
    said = []
    assert IO.fetch_axis_fingerprint(_Boom(), "s9", directory=str(ddir), log=said.append) is None
    assert "s9.json" in said[-1]
    assert IO.fetch_axis_fingerprint(_Boom(), "bad", directory=str(ddir), log=said.append) is None
    assert "unreadable" in said[-1]
    assert not (tmp_path / "cache").exists()                            # a directory is not cached
    assert IO.axis_fingerprint_source(_Boom(), directory=str(ddir)) == "directory"


# --- the registry, the peak tables, the workers --------------------------------------------------

def _two_calibrations() -> AF.FrameSet:
    """A hand-built frame of two items on two DIFFERENT calibrations (different step
    tables): what `from_documents` refuses as one batch (two calibration groups), used
    here to see that every per-item path keeps the items apart."""
    a = AF.load_document(_document(steps={K_MID: STEP_PPM, K_MID - 1: SIDE_PPM, _k_at(120.0): 1.1}, item=_item("a")),
                         polarity="-")
    b = AF.load_document(_document(steps={K_MID: -1.8, _k_at(250.0): -1.5}, item=_item("b")), polarity="-")
    assert a.ok and b.ok
    return AF.FrameSet(items={"a": a.fingerprint.frame_item("a"), "b": b.fingerprint.frame_item("b")})


def _raw_table(mzs):
    return pd.DataFrame({"peak_id": [f"p{i}" for i in range(len(mzs))], "mz": list(mzs),
                         "height": np.linspace(1.0, 2.0, len(mzs))})


def test_a_frame_record_registers_one_map_per_item_and_the_cache_keeps_the_servers_axis(tmp_path):
    fs = _two_calibrations()
    rec = json.loads(json.dumps(fs.as_dict()))
    mzs = [120.3, 250.4, STEP_MZ + 0.1, 700.2]
    for sid in ("a", "b"):
        (tmp_path / sid).mkdir()
        _raw_table(mzs).to_parquet(tmp_path / sid / "peaks.v2.parquet")
    IO._SCORING_CACHE["a"] = ("stale", {})
    IO.set_axis_correction(["a", "b", "c"], rec)
    assert "a" not in IO._SCORING_CACHE
    fa, fb = IO.axis_correction("a"), IO.axis_correction("b")
    assert isinstance(fa, AF.FrameItem) and isinstance(fb, AF.FrameItem) and fa.sample_item_id == "a"
    assert IO.axis_correction("c") is None                              # not in the frame: served raw
    assert not np.array_equal(fa.d, fb.d)
    for sid, it in (("a", fs.for_item("a")), ("b", fs.for_item("b"))):
        got = IO.fetch_peaks(None, sid, cache_root=tmp_path)["mz"].to_numpy()
        want = MQ.apply_correction(it, np.array(mzs))
        assert np.allclose(got, want, rtol=0, atol=1e-9) and not np.array_equal(got, mzs)
        assert pd.read_parquet(tmp_path / sid / "peaks.v2.parquet")["mz"].tolist() == mzs
    # the two items are corrected by their own maps, not by each other's
    assert not np.array_equal(IO.fetch_peaks(None, "a", cache_root=tmp_path)["mz"],
                              IO.fetch_peaks(None, "b", cache_root=tmp_path)["mz"])
    IO.register_offline_sample("a", _raw_table(mzs), MECHANISMS)
    try:
        assert np.allclose(IO.fetch_peaks(None, "a")["mz"], MQ.apply_correction(fs.for_item("a"), np.array(mzs)),
                           rtol=0, atol=1e-9)
    finally:
        IO.unregister_offline_sample("a")
    IO.clear_axis_correction(["a"])
    assert IO.fetch_peaks(None, "a", cache_root=tmp_path)["mz"].tolist() == mzs and IO.axis_correction("b") is fb


def test_a_spawned_worker_rebuilds_the_frame_from_the_pickled_record():
    fs = _two_calibrations().with_inner(_spline_inner(80.0, 430.0))
    ids = fs.sample_ids
    axis_args = (ids, fs.as_dict())                                     # what the pool's initargs carry
    ids2, rec2 = pickle.loads(pickle.dumps(axis_args))
    assert rec2 == axis_args[1]
    AB._worker_init("ambient-air", [], {}, None, None, (ids2, rec2))
    grid = np.linspace(70.0, 700.0, 3001)
    for sid in ids:
        worker = IO.axis_correction(sid)
        parent = fs.for_item(sid)
        assert isinstance(worker, AF.FrameItem) and worker.sample_item_id == sid
        assert np.array_equal(worker.in_scope(grid), parent.in_scope(grid))
        assert np.allclose(worker.predict(grid, extrapolate=True), parent.predict(grid, extrapolate=True),
                           rtol=0, atol=1e-4)
        assert np.allclose(MQ.apply_correction(worker, grid), MQ.apply_correction(parent, grid), rtol=0, atol=1e-7)
    AB._worker_init("ambient-air", [], {}, None, None, None)
    assert all(IO.axis_correction(sid) is None for sid in ids)


# --- a whole batch -------------------------------------------------------------------------------

def test_a_whole_batch_is_assigned_on_the_frame_corrected_axis(framed_run, batch):
    docs, frame0, axis, ts, truth = batch
    out, res, tabs, logs = framed_run
    ax = res["summary"]["mass_axis"]
    assert ax["mode"] == "auto" and ax["applied"] and ax["model"] == "frame"
    assert ax["verdict"] in AF.ACCEPTED_VERDICTS and ax["wave"]["model"] == "frame"
    fr = ax["frame"]
    assert fr["held"] is None and fr["n_items"] == 4 and fr["n_groups"] == 1 and fr["source"] == "directory"
    assert fr["gate"]["passed"] and fr["gate"]["failed"] == [] and set("abcde") <= set(fr["gate"])
    assert fr["gate"]["d"]["raw_applied"] and fr["gate"]["d"]["ok"] and fr["gate"]["e"]["ok"]
    assert fr["fits"]["raw"]["verdict"] == "axis_steps" and fr["fits"]["raw"]["steps"]     # the raw walk saw steps
    assert fr["fits"]["W"]["steps"] == [] and fr["fits"]["W"]["has_model"]               # the W walk saw none
    for fit in fr["fits"].values():
        assert fit["cv_ppm"] is not None and fit["raw_ppm"] is not None and len(fit["scope_mz"]) == 2
    assert set(fr["items"]) == set(tabs) and all(fr["documents"][s]["usable"] for s in tabs)
    for rec in fr["items"].values():
        assert rec["mode_id"] and rec["document_sha1"] and rec["n_measured"] > 40 and rec["n_edge_ambiguous"] >= 0
        assert rec["d_max_ppm"] > D_MID - STEP_TOL and rec["grid"]["zone_khz"] >= 0.1
    assert ax["inner"]["model"] == "locks" and ax["inner"]["verdict"] == ax["verdict"]
    # the tables: the W fit's locks, marked, and the band table per item
    locks = pd.read_csv(out / "tables" / "mass_axis_locks.csv")
    assert (locks["frame"] == "W").all() and len(locks) == ax["inner"]["locks"]["n_locks"]
    table = pd.read_csv(out / "tables" / "mass_axis_frame.csv")
    assert list(table.columns) == ["sample_item_id", "k", "mz_lo", "mz_hi", "d_ppm", "measured"]
    assert len(table) == 4 * fr["items"]["s0"]["n_bands"] and set(table["sample_item_id"]) == set(tabs)
    assert abs(table["d_ppm"].max() - D_MID) < 0.3
    # every assigned file's ledger holds the m/z corrected by ITS item's map
    fit = MQ.fit_from_record(ax["wave"])
    assert isinstance(fit, AF.FrameSet) and len(res["sample_ids"]) >= 2
    for sid in res["sample_ids"]:
        led = pd.read_csv(out / "per_file" / f"{sid}_ledger.csv")
        raw = tabs[sid].set_index("peak_id")["mz"]
        got = led.drop_duplicates("peak_id").set_index("peak_id")["mz"]
        assert len(got) > 20, sid
        want = MQ.apply_correction(fit.for_item(sid), raw.reindex(got.index).to_numpy())
        assert np.allclose(got.to_numpy(), want, rtol=0, atol=1e-7), sid
    # the true ions land on the axis: within 0.3 ppm, away from the edge zones, in scope
    th = np.array([AL.exact_mz(i) for i in sorted(truth)])
    errs = []
    for sid, g in res["ts_peaks"].groupby("sample_item_id"):
        it = fit.for_item(sid)
        measured = th * (1 + axis(th) * 1e-6)
        keep = it.in_scope(measured) & ~it.edge_ambiguous(measured)
        m = g["mz"].to_numpy()
        for t in th[keep]:
            j = int(np.argmin(np.abs(m - t)))
            e = (m[j] - t) / t * 1e6
            if abs(e) < 3.0:
                errs.append(e)
    errs = np.abs(np.array(errs))
    assert len(errs) > 400 and np.median(errs) < 0.15 and np.mean(errs <= 0.3) > 0.95, (np.median(errs), errs.max())
    # the stamped series restores to the server's axis exactly, and the correction is gone
    assert AB.AXIS_COL in res["ts_peaks"].columns
    back = AB.restore_axis(pd.read_parquet(out / "per_file" / "_batch_ts.parquet"), log=_quiet)
    key = ["sample_item_id", "height"]
    a = back.sort_values(key)["mz"].to_numpy()
    b = ts.sort_values(key)["mz"].to_numpy()
    assert len(a) == len(b) and np.allclose(a, b, rtol=0, atol=1e-9)
    assert all(IO.axis_correction(sid) is None for sid in tabs)
    assert any("frame: one calibration group" in line for line in logs)
    assert any("APPLIED" in line and "band steps" in line for line in logs)


def test_an_off_run_and_a_run_served_no_document_are_byte_for_byte_the_same(off_run, missing_run):
    """Fingerprints looked for (an Orbitrap, the lock path) but none served: nothing is
    recorded -- the summary carries no `frame` key -- and every output file is the off
    run's; the log says what was looked for."""
    out_off, res_off, _ = off_run
    out_missing, res_missing, _, logs = missing_run
    ax_off, ax_missing = res_off["summary"]["mass_axis"], res_missing["summary"]["mass_axis"]
    assert "frame" not in ax_off and ax_off["model"] == "locks" and ax_off["applied"]
    assert "frame" not in ax_missing and list(ax_missing) == list(ax_off)
    assert any("no server connection" in line for line in logs)
    assert any("0 of 4 items usable; no frame: item s0: no document served" in line for line in logs)
    _same_outputs(out_off, out_missing, ignore_frame=False)
    assert not (out_off / "tables" / "mass_axis_frame.csv").exists()
    assert not (out_missing / "tables" / "mass_axis_frame.csv").exists()


def test_a_404_and_a_tof_leave_every_output_as_without_fingerprints(refs, tmp_path, monkeypatch):
    ts = _orbi_batch(np.random.default_rng(41), refs)
    mech = pd.DataFrame({"ionization_mechanism": MECHANISMS, "ionization_mechanism_id": MECHANISMS})

    def not_found(sid):
        raise NotFoundError("Not Found", status_code=404)

    # a server before the route (every lookup 404s); the same client on both arms, since a
    # client that is not None reaches other stages too (the width model, the noise edges)
    older_server = SimpleNamespace(samples=SimpleNamespace(get_axis_fingerprint=not_found),
                                   ionization=SimpleNamespace(list=lambda: mech))
    logs404 = []
    res404, _ = _run_offline(ts, tmp_path / "r404", client=older_server, log=logs404.append)
    ax = res404["summary"]["mass_axis"]
    assert "frame" not in ax                                   # served nothing: nothing recorded
    assert any("NotFoundError" in line and "no document served" in line for line in logs404)
    assert any("items usable; no frame: item" in line and "no document served" in line for line in logs404)
    # off: nothing is fetched even with documents at hand, and the summary says nothing of frames
    ddir = _write_docs({sid: _document(item=_item(sid)) for sid in ts["sample_item_id"].unique()}, tmp_path / "docs")
    monkeypatch.setattr(IO, "fetch_axis_fingerprint",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("fetched under off or a hold")))
    logs = []
    res_off, _ = _run_offline(ts, tmp_path / "off", client=older_server, log=logs.append, axis_fingerprints="off",
                              axis_fingerprint_dir=str(ddir))
    assert "frame" not in res_off["summary"]["mass_axis"]
    assert any("axis fingerprints off" in line for line in logs)
    _same_outputs(tmp_path / "off", tmp_path / "r404", ignore_frame=False)          # byte for byte
    # a hold from the caller (a pool of several batches, server-side scoring): nothing is fetched
    # either, nothing recorded, and the held run is the held run without the feature
    res_hold, _ = _run_offline(ts, tmp_path / "hold", client=older_server, axis_fingerprint_dir=str(ddir),
                               mass_axis_hold="held by the caller")
    res_hold_off, _ = _run_offline(ts, tmp_path / "hold_off", client=older_server, axis_fingerprints="off",
                                   mass_axis_hold="held by the caller")
    for r in (res_hold, res_hold_off):
        assert r["summary"]["mass_axis"]["held"] == "held by the caller" and not r["summary"]["mass_axis"]["applied"]
        assert "frame" not in r["summary"]["mass_axis"]
    _same_outputs(tmp_path / "hold", tmp_path / "hold_off", ignore_frame=False)
    # a TOF: fingerprints are not looked for at all (no record either way)
    monkeypatch.undo()
    monkeypatch.setattr(IO, "fetch_axis_fingerprint",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("fetched on a TOF")))
    res_tof, _ = _run_offline(ts, tmp_path / "tof", roster="tof", axis_fingerprint_dir=str(ddir))
    res_tof_off, _ = _run_offline(ts, tmp_path / "tof_off", roster="tof", axis_fingerprints="off")
    assert res_tof["summary"]["mass_axis"]["instrument"] == "tof"
    assert "frame" not in res_tof["summary"]["mass_axis"] and "frame" not in res_tof_off["summary"]["mass_axis"]
    _same_outputs(tmp_path / "tof", tmp_path / "tof_off", ignore_frame=False)


def test_the_gate_holds_a_shifted_grid_and_todays_path_stands(off_run, held_run):
    out_off, res_off, _ = off_run
    out_held, res_held, _, logs = held_run
    ax = res_held["summary"]["mass_axis"]
    assert ax["model"] == "locks" and ax["applied"] and ax["verdict"] == res_off["summary"]["mass_axis"]["verdict"]
    fr = ax["frame"]
    assert fr["held"].startswith("gate (") and set(fr["gate"]["failed"]) & {"a", "b"} and not fr["gate"]["passed"]
    assert fr["fits"]["W"]["steps"] and fr["fits"]["raw"]["steps"]
    assert "items" not in fr and "inner" not in ax
    assert any("frame NOT applied: gate" in line for line in logs)
    _same_outputs(out_off, out_held, ignore_frame=True)
    assert not (out_held / "tables" / "mass_axis_frame.csv").exists()
    assert "frame" not in pd.read_csv(out_held / "tables" / "mass_axis_locks.csv").columns


@pytest.mark.skipif(not os.environ.get("PEAKY_MAIN_WORKTREE"),
                    reason="set PEAKY_MAIN_WORKTREE to a checkout of main with a synced .venv")
def test_an_off_run_reproduces_the_outputs_of_mains_package(batch, off_run, tmp_path):
    root = Path(os.environ["PEAKY_MAIN_WORKTREE"]).expanduser()
    py = root / ".venv" / "bin" / "python"
    assert py.exists(), py
    docs, frame, axis, ts, truth = batch
    ts.to_parquet(tmp_path / "ts.parquet")
    subprocess.run([str(py), "-c", _DRIVER, str(tmp_path / "ts.parquet"), str(tmp_path / "main"), json.dumps(RUN_KW)],
                   check=True, cwd=str(tmp_path), capture_output=True)
    _same_outputs(tmp_path / "main", off_run[0], ignore_frame=False)


# --- measure_axis with a frame, directly ----------------------------------------------------------

def test_measure_axis_records_the_build_record_and_honours_a_hold(refs, batch):
    ts_ref = _orbi_batch(np.random.default_rng(42), refs)
    plain = AB.measure_axis(ts_ref, "NO3", "orbitrap", log=_quiet)[1]
    unserved = {"held": "item x: no document served", "n_usable": 0,
                "items": {"x": {"usable": False, "status": None, "sha1": None, "why": "no document served"}}}
    nothing = AB.measure_axis(ts_ref, "NO3", "orbitrap", log=_quiet, frame_record=dict(unserved))[1]
    assert "frame" not in nothing                                        # served nothing: no key at all
    served = {"held": "item x: status 'missing'", "n_usable": 0,
              "items": {"x": {"usable": False, "status": "missing", "sha1": "a" * 40, "why": "status 'missing'"}}}
    with_rec = AB.measure_axis(ts_ref, "NO3", "orbitrap", log=_quiet, frame_record=dict(served))[1]
    assert with_rec.pop("frame") == {"held": served["held"], "n_usable": 0, "documents": served["items"]}
    assert list(with_rec) == list(plain) == list(nothing)
    for d in (plain, with_rec, nothing):
        d.pop("_locks_table", None)                                      # a DataFrame: compared apart
    assert json.dumps(with_rec, default=str) == json.dumps(plain, default=str) == json.dumps(nothing, default=str)
    assert not AB._documents_served(unserved) and AB._documents_served(served) and not AB._documents_served(None)
    docs, frame, axis, ts, truth = batch
    on_tof = AB.measure_axis(ts_ref, "NO3", "tof", log=_quiet, frame=frame)[1]
    assert "lock-model path" in on_tof["frame"]["held"] and not on_tof["applied"]
    out, info, _ = AB.measure_axis(ts, "NO3", "orbitrap", hold="because", log=_quiet, frame=frame)
    assert out is ts and not info["applied"] and info["held"] == "because"
    assert info["frame"]["held"] == "because" and "fits" not in info["frame"]


def test_run_refuses_an_unknown_fingerprint_mode_before_any_server_call(tmp_path, monkeypatch):
    monkeypatch.setattr(IO, "connect", lambda *a, **k: (_ for _ in ()).throw(AssertionError("connect")))
    with pytest.raises(ValueError, match="axis_fingerprints"):
        AB.run(peaks=pd.DataFrame({"sample_item_id": ["s"], "mz": [100.0], "height": [1.0]}),
               reagent="NO3", out_dir=str(tmp_path), axis_fingerprints="sometimes",
               sample_ids=["s"], resolving_power="none")


def test_the_fetch_condition_is_the_orbitrap_lock_path_without_a_hold():
    assert AB.AXIS_FINGERPRINT_MODES == ("auto", "refresh", "off")
    assert AB._frame_wanted("auto", "orbitrap", "auto", None, ["s0"]) == (True, None)
    assert AB._frame_wanted("refresh", "orbitrap", "locks", None, ["s0"]) == (True, None)
    for args, says in ((("off", "orbitrap", "auto", None, ["s0"]), "off"),
                       (("auto", "tof", "auto", None, ["s0"]), "Orbitrap"),
                       (("auto", None, "auto", None, ["s0"]), "Orbitrap"),
                       (("auto", "orbitrap", "reference", None, ["s0"]), "lock-model"),
                       (("auto", "orbitrap", "off", None, ["s0"]), "lock-model"),
                       (("auto", "orbitrap", "auto", "a pool of 3 batches", ["s0"]), "held"),
                       (("auto", "orbitrap", "auto", None, []), "no sample")):
        wanted, why = AB._frame_wanted(*args)
        assert not wanted and says in why, (args, why)


def test_refresh_reads_past_the_cache_and_replaces_it(tmp_path, monkeypatch):
    # _batch_frame hands the mode down as use_cache, and says so in the record
    seen = []

    def recorder(client, sid, *, directory=None, log=None, use_cache=True):
        seen.append(use_cache)
        return _document(item=_item(sid))

    monkeypatch.setattr(IO, "fetch_axis_fingerprint", recorder)
    client = SimpleNamespace(samples=SimpleNamespace(get_axis_fingerprint=lambda sid: None))
    frame, rec = AB._batch_frame(client, ["s0", "s1"], "-", None, _quiet, use_cache=False)
    assert frame is not None and seen == [False, False] and rec["refreshed"] is True
    frame, rec = AB._batch_frame(client, ["s0"], "-", None, _quiet)
    assert seen[-1] is True and "refreshed" not in rec
    monkeypatch.undo()
    # the fetch itself: use_cache False asks the server although a cached copy exists, and replaces it
    stale = _document(item=_item("s0"))
    stale["modes"][0]["grid"]["n_hits"] = 3
    (tmp_path / "s0").mkdir()
    (tmp_path / "s0" / IO.AXIS_FINGERPRINT_CACHE).write_text(json.dumps(stale))
    fresh = _document(item=_item("s0"))
    calls = []

    def served(sid):
        calls.append(sid)
        return json.loads(json.dumps(fresh))

    client = SimpleNamespace(samples=SimpleNamespace(get_axis_fingerprint=served))
    assert IO.fetch_axis_fingerprint(client, "s0", cache_root=tmp_path) == stale and calls == []
    assert IO.fetch_axis_fingerprint(client, "s0", cache_root=tmp_path, use_cache=False) == fresh and calls == ["s0"]
    assert json.loads((tmp_path / "s0" / IO.AXIS_FINGERPRINT_CACHE).read_text()) == fresh
    assert IO.fetch_axis_fingerprint(client, "s0", cache_root=tmp_path) == fresh and calls == ["s0"]


class _Stop(Exception):
    """Raised by a stub to end a run once it has seen what it was handed."""


@pytest.mark.parametrize("mode, use_cache", [("refresh", False), ("auto", True)])
def test_the_run_hands_refresh_to_the_fetch_as_use_cache_false(batch, tmp_path, monkeypatch, mode, use_cache):
    docs, frame, axis, ts, truth = batch
    seen = []

    def recorder(client, sid, *, directory=None, log=None, use_cache=True):
        seen.append((sid, use_cache))
        raise _Stop

    monkeypatch.setattr(IO, "fetch_axis_fingerprint", recorder)
    with pytest.raises(_Stop):
        _run_offline(ts, tmp_path / "run", axis_fingerprints=mode)
    assert seen == [("s0", use_cache)]


def test_a_malformed_document_is_handed_back_never_cached_and_a_malformed_cached_copy_is_fetched_again(tmp_path):
    bad = _document(item=_item("s0"))
    bad["item"]["streams"] = 1                                          # a count where the schema has a list
    good = _document(item=_item("s0"))
    served, calls, said = [bad], [], []

    def get(sid):
        calls.append(sid)
        return json.loads(json.dumps(served[0]))

    client = SimpleNamespace(samples=SimpleNamespace(get_axis_fingerprint=get))
    assert IO.fetch_axis_fingerprint(client, "s0", cache_root=tmp_path, log=said.append) == bad
    assert not (tmp_path / "s0").exists()
    assert said == ["[axis-fingerprint] sample s0: document not cached (malformed document: item.streams is int, "
                    "not a list)"]
    assert AF.load_document(bad, polarity="-").why == "malformed document: item.streams is int, not a list"
    served[0] = good                                                    # the server's document is fixed
    assert IO.fetch_axis_fingerprint(client, "s0", cache_root=tmp_path) == good and calls == ["s0", "s0"]
    cfile = tmp_path / "s0" / IO.AXIS_FINGERPRINT_CACHE
    assert json.loads(cfile.read_text()) == good
    cfile.write_text(json.dumps(bad))                                   # a malformed copy cached by an older reader
    assert IO.fetch_axis_fingerprint(client, "s0", cache_root=tmp_path) == good and calls == ["s0"] * 3
    assert json.loads(cfile.read_text()) == good
    # a malformed document of a status that is never cached says nothing of the cache
    served[0], said = {"schema": AF.SCHEMA, "version": 1, "status": "missing", "modes": 3}, []
    assert IO.fetch_axis_fingerprint(client, "s9", cache_root=tmp_path, log=said.append) == served[0] and not said


def test_a_batch_with_one_malformed_document_finishes_on_the_raw_path(off_run, malformed_run):
    out_off, res_off, _ = off_run
    out_bad, res_bad, _, logs = malformed_run
    ax = res_bad["summary"]["mass_axis"]
    assert ax["model"] == "locks" and ax["applied"] and ax["verdict"] == res_off["summary"]["mass_axis"]["verdict"]
    why = "malformed document: item.scans is int, not a list"
    assert ax["frame"]["held"] == f"item s1: {why}" and ax["frame"]["documents"]["s1"]["why"] == why
    assert ax["frame"]["documents"]["s1"]["status"] == "ok" and ax["frame"]["documents"]["s0"]["usable"]
    assert any(f"no frame: item s1: {why}" in line for line in logs)
    _same_outputs(out_off, out_bad, ignore_frame=True)
    assert not (out_bad / "tables" / "mass_axis_frame.csv").exists()


def test_a_reader_that_raises_holds_the_frame_naming_the_exception_type_only(refs, monkeypatch):
    served = {sid: _document(item=_item(sid)) for sid in ("s0", "s1")}
    monkeypatch.setattr(IO, "fetch_axis_fingerprint", lambda client, sid, **k: served[sid])

    def boom(cls, documents, **kw):
        raise TypeError("object of type 'int' has no len() at https://mascope.example.test/x")

    monkeypatch.setattr(AF.FrameSet, "from_documents", classmethod(boom))
    said = []
    frame, rec = AB._batch_frame(None, ["s0", "s1"], "-", None, said.append)
    assert frame is None and rec["held"] == "documents unreadable (TypeError)" and rec["n_usable"] == 0
    for sid, doc in served.items():
        assert rec["items"][sid] == {"usable": False, "status": "ok", "mode_id": None, "sha1": AF.document_sha1(doc),
                                     "notes": [], "why": "not read"}
    assert "example.test" not in json.dumps(rec) and "no len" not in json.dumps(rec)
    assert any("no frame: documents unreadable (TypeError)" in line for line in said)
    # the run records the documents it was served and stands on its own path
    ts_ref = _orbi_batch(np.random.default_rng(42), refs)
    plain = AB.measure_axis(ts_ref, "NO3", "orbitrap", log=_quiet)[1]
    info = AB.measure_axis(ts_ref, "NO3", "orbitrap", log=_quiet, frame_record=rec)[1]
    fr = info.pop("frame")
    assert fr["held"] == "documents unreadable (TypeError)" and set(fr["documents"]) == {"s0", "s1"}
    for d in (plain, info):
        d.pop("_locks_table", None)
    assert json.dumps(info, default=str) == json.dumps(plain, default=str)


def test_a_frame_path_that_raises_leaves_the_raw_fit_standing(monkeypatch, batch):
    docs, frame, axis, ts, truth = batch
    rec = AF.FrameSet.from_documents(docs, polarity="-")[1]
    out_plain, plain, _ = AB.measure_axis(ts, "NO3", "orbitrap", log=_quiet)

    def boom(frame_):
        raise RuntimeError("the band table could not be built")

    monkeypatch.setattr(AB, "_frame_table", boom)                       # after the composite was applied
    said = []
    out, info, _ = AB.measure_axis(ts, "NO3", "orbitrap", log=said.append, frame=frame, frame_record=rec)
    fr = info.pop("frame")
    assert fr["held"] == "the frame path failed (RuntimeError: the band table could not be built)"
    assert fr["documents"] == rec["items"] and "items" not in fr and "fits" not in fr
    assert any("frame NOT applied: the frame path failed" in line for line in said)
    assert info["model"] == "locks" and info["applied"] and "_frame_table" not in info and "inner" not in info
    for d in (plain, info):
        d.pop("_locks_table", None)
    assert json.dumps(info, default=str) == json.dumps(plain, default=str)
    assert np.array_equal(out["mz"].to_numpy(), out_plain["mz"].to_numpy())


def test_the_fetch_stops_at_the_first_item_served_no_document(monkeypatch):
    """The frame needs every item's document, so the items after the first one served
    none are not asked (a server that is down or rate-limiting would cost each of them
    the SDK's full retries) and are recorded as not asked; a served document of any
    status does not stop the loop."""
    asked = []
    served = {"s0": {"schema": AF.SCHEMA, "version": 1, "status": "missing"}}

    def fetch(client, sid, *, directory=None, log=None, use_cache=True):
        asked.append(sid)
        if sid in served:
            return served[sid]
        log(f"[axis-fingerprint] sample {sid}: no document served (MascopeAPIError: [HTTP 429] Too Many Requests)")
        return None

    monkeypatch.setattr(IO, "fetch_axis_fingerprint", fetch)
    ids, said = ["s0", "s1", "s2", "s3"], []
    frame, rec = AB._batch_frame(None, ids, "-", None, said.append)
    assert frame is None and asked == ["s0", "s1"] and rec["held"] == "item s0: status 'missing'"
    assert set(rec["items"]) == set(ids) and rec["n_items"] == 4
    assert rec["items"]["s1"]["fetch"] == "sample s1: no document served (MascopeAPIError: [HTTP 429] Too Many Requests)"
    assert rec["items"]["s2"]["fetch"] == rec["items"]["s3"]["fetch"] == "not asked: item s1 had no document"
    assert rec["items"]["s2"]["why"] == rec["items"]["s3"]["why"] == "not asked"
    assert rec["items"]["s1"]["why"] == "no document served"
    assert "fetch" not in rec["items"]["s0"]
    assert any("2 later item(s) not asked: item s1 had no document" in line for line in said)
    asked.clear()
    served.update({sid: _document(item=_item(sid)) for sid in ids})
    frame, rec = AB._batch_frame(None, ids, "-", None, _quiet)
    assert frame is not None and asked == ids and not any("fetch" in r for r in rec["items"].values())


def test_no_server_address_reaches_the_log_or_the_records(monkeypatch, tmp_path):
    from functools import partial
    from peaky import pipeline as PL
    host = "mascope.example.test"
    served = {"s0": _document(item=_item("s0"))}

    def get(sid):
        if sid in served:
            return json.loads(json.dumps(served[sid]))
        raise NotFoundError(message="Not Found", status_code=404, url=f"https://{host}/api/samples/{sid}/axis_fingerprint")

    client = SimpleNamespace(samples=SimpleNamespace(get_axis_fingerprint=get))
    monkeypatch.setattr(IO, "fetch_axis_fingerprint", partial(IO.fetch_axis_fingerprint, cache_root=tmp_path))
    said = []
    frame, rec = AB._batch_frame(client, ["s0", "s1", "s2"], "-", None, said.append)
    _, info, _ = AB.measure_axis(None, "NO3", "orbitrap", log=_quiet, frame=frame, frame_record=rec)
    docs = info["frame"]["documents"]
    assert docs["s1"]["fetch"] == "sample s1: no document served (NotFoundError: [HTTP 404] Not Found)"
    assert docs["s2"]["fetch"] == "not asked: item s1 had no document"
    text = json.dumps([info, PL._manifest_mass_axis(info), said], default=str)
    assert host not in text and "URL" not in text and "https://" not in text


# --- the gate's reach and quality rules, the caps and the holds, on stubbed W fits --------------------

def _stub_lock_model(model, cv, verdict="axis_trend", steps=()):
    """A `_lock_model` that returns `model` as the W fit with the info layout the real one writes."""
    def fake(ts_, pol, re, info, log):
        info["_locks_table"] = None
        info["locks"] = {"cv_ppm": cv, "raw_ppm": 1.2, "steps": list(steps), "n_pass1": 1, "n_links": 1,
                         "n_narrow": 0, "n_off_curve": 0}
        info.update(model="locks", wave=model.as_dict(), scope_mz=list(model.mz_range),
                    max_abs_ppm=round(model.max_abs_ppm(), 3), verdict=verdict)
        return model, True
    return fake


def _frame_path(monkeypatch, batch, model_raw, cv_raw, model_W, cv_W, *, raw_hold=None, caller_hold=None):
    """assign_batch._frame_path on the synthetic framed batch with the raw fit given and the W fit stubbed."""
    docs, frame, axis, ts, truth = batch
    if model_raw is not None:
        info = {"applied": False, "verdict": "axis_trend", "model": "locks", "wave": model_raw.as_dict(),
                "scope_mz": list(model_raw.mz_range), "max_abs_ppm": round(model_raw.max_abs_ppm(), 3),
                "locks": {"cv_ppm": cv_raw, "raw_ppm": 1.2, "steps": []}}
    else:
        info = {"applied": False, "verdict": None, "locks": {"why": "no lock model"}}
    monkeypatch.setattr(AB, "_lock_model", _stub_lock_model(model_W, cv_W))
    rec = AF.FrameSet.from_documents(docs, polarity="-")[1]
    out = AB._frame_path(ts, frame, rec, model_raw, caller_hold, raw_hold, info, "-", (), _quiet)
    return out, info


def test_gate_d_holds_a_w_model_that_reaches_less_than_the_raw_one_by_more_than_a_band(monkeypatch, batch):
    docs, frame, axis, ts, truth = batch
    it = frame.for_item("s0")
    raw = _spline_inner(80.0, 430.0)
    w_lo = float(it.band_width_at(np.array([80.0]))[0])
    w_hi = float(it.band_width_at(np.array([430.0]))[0])
    assert 2.0 < w_lo < 4.0 and 30.0 < w_hi < 45.0
    # shorter at the top by more than one band: held by (d), the rule recorded
    out, info = _frame_path(monkeypatch, batch, raw, 0.10, _spline_inner(80.0, 430.0 - w_hi - 1.0), 0.10)
    fr = info["frame"]
    assert out is None and fr["held"].startswith("gate (d)") and fr["gate"]["failed"] == ["d"] and not fr["gate"]["passed"]
    assert not fr["gate"]["d"]["ok"] and "upper end" in fr["gate"]["why"] and fr["gate"]["d"]["raw_applied"]
    assert fr["gate"]["d"]["raw_scope_mz"] == [80.0, 430.0] and fr["gate"]["d"]["W_scope_mz"][1] < 430.0 - w_hi
    assert info["model"] == "locks" and not info["applied"]            # today's model stands
    # shorter at the top by less than one band: passes, the composite applied
    out, info = _frame_path(monkeypatch, batch, raw, 0.10, _spline_inner(80.0, 430.0 - w_hi + 1.0), 0.10)
    assert out is not None and info["applied"] and info["model"] == "frame" and info["frame"]["gate"]["d"]["ok"]
    # shorter at the low end by more than one band: held
    out, info = _frame_path(monkeypatch, batch, raw, 0.10, _spline_inner(80.0 + w_lo + 1.0, 430.0), 0.10)
    assert out is None and "lower end" in info["frame"]["held"] and info["frame"]["gate"]["failed"] == ["d"]
    # the raw model held by its own cap: its reach is not compared, the frame may apply
    out, info = _frame_path(monkeypatch, batch, raw, 0.10, _spline_inner(120.0, 430.0), 0.10, raw_hold="over the cap")
    d_rec = info["frame"]["gate"]["d"]
    assert out is not None and info["model"] == "frame" and d_rec["ok"] and not d_rec["raw_applied"]
    assert d_rec["raw_scope_mz"] is None and d_rec["W_scope_mz"] == [120.0, 430.0]
    # no raw model at all: the W fit is still tried, (d) and (e) vacuous
    out, info = _frame_path(monkeypatch, batch, None, None, _spline_inner(120.0, 430.0), 0.9)
    assert out is not None and info["model"] == "frame" and info["frame"]["gate"]["passed"]
    assert not info["frame"]["gate"]["d"]["raw_applied"] and info["frame"]["gate"]["e"]["cv_raw_ppm"] is None


def test_gate_e_holds_a_w_fit_whose_held_out_error_passes_one_and_a_half_times_the_raw_fits(monkeypatch, batch):
    raw = _spline_inner(80.0, 430.0)
    assert AB.FRAME_QUALITY_CV_RATIO == 1.5
    out, info = _frame_path(monkeypatch, batch, raw, 0.10, raw, 0.151)
    assert out is None and info["frame"]["gate"]["failed"] == ["e"] and "1.5 x the raw" in info["frame"]["held"]
    assert info["frame"]["gate"]["e"] == {"ok": False, "cv_raw_ppm": 0.1, "cv_W_ppm": 0.151, "ratio_max": 1.5}
    assert info["model"] == "locks" and not info["applied"]
    out, info = _frame_path(monkeypatch, batch, raw, 0.10, raw, 0.15)              # exactly 1.5 x: the rule is '>'
    assert out is not None and info["frame"]["gate"]["e"]["ok"] and info["model"] == "frame"
    out, info = _frame_path(monkeypatch, batch, raw, 0.10, raw, 0.2)
    assert out is None and info["frame"]["gate"]["failed"] == ["e"]
    # both rules failing are both named
    out, info = _frame_path(monkeypatch, batch, raw, 0.10, _spline_inner(80.0, 300.0), 0.3)
    assert out is None and info["frame"]["gate"]["failed"] == ["d", "e"] and info["frame"]["held"].startswith("gate (d, e)")


def test_the_w_models_cap_and_the_callers_hold_hold_the_frame(monkeypatch, batch):
    raw = _spline_inner(80.0, 430.0)
    big = AL.AxisModel(segments=[{"kind": "spline", "lo": 80.0, "hi": 430.0, "n": 100, "x": np.linspace(80, 430, 400),
                                  "y": np.full(400, 9.0)}], n_locks=100)
    out, info = _frame_path(monkeypatch, batch, raw, 0.10, big, 0.10)
    assert out is None and info["frame"]["held"].startswith("the W model: the correction reaches 9.0 ppm, above 7")
    assert "gate" not in info["frame"] and info["frame"]["fits"]["W"]["max_abs_ppm"] == 9.0
    out, info = _frame_path(monkeypatch, batch, raw, 0.10, raw, 0.10, caller_hold="a pool of 3 batches")
    assert out is None and info["frame"]["held"] == "a pool of 3 batches" and "fits" not in info["frame"]


def _stub_axis_ok(model, cv):
    """A `_lock_model` that ends in its axis_ok branch: the W fit's model recorded (wave, scope)
    and no model returned -- nothing smooth to correct."""
    def fake(ts_, pol, re, info, log):
        info["_locks_table"] = None
        info["locks"] = {"cv_ppm": cv, "raw_ppm": 0.4, "max_abs_ppm": 0.6, "steps": [], "n_pass1": 1,
                         "n_links": 1, "n_narrow": 0, "n_off_curve": 0}
        info.update(model="locks", wave=model.as_dict(), scope_mz=list(model.mz_range), max_abs_ppm=0.6,
                    verdict="axis_ok")
        return None, True
    return fake


def test_under_axis_ok_the_band_steps_reach_only_the_w_fits_lock_range(monkeypatch, batch):
    """The W verdict axis_ok has no model to apply, and the band steps alone used to be
    applied over the whole table (m/z ~58-747), beyond any lock. Now the composite's inner
    model is the W fit's lock model set to zero: nothing moves outside the W scope, the
    recorded scope is the lock range, the record round-trips and the way back is exact;
    and gate (d) compares that real scope with the raw model's."""
    docs, frame, axis, ts, truth = batch
    raw = _spline_inner(90.0, 420.0)
    info = {"applied": False, "verdict": "axis_trend", "model": "locks", "wave": raw.as_dict(),
            "scope_mz": [90.0, 420.0], "max_abs_ppm": round(raw.max_abs_ppm(), 3),
            "locks": {"cv_ppm": 0.10, "raw_ppm": 1.2, "steps": []}}
    monkeypatch.setattr(AB, "_lock_model", _stub_axis_ok(_spline_inner(90.0, 420.0), 0.10))
    rec = AF.FrameSet.from_documents(docs, polarity="-")[1]
    said = []
    out = AB._frame_path(ts, frame, rec, raw, None, None, info, "-", (), said.append)
    assert out is not None and info["model"] == "frame" and info["verdict"] == "axis_ok", info["frame"].get("held")
    assert info["scope_mz"] == [90.0, 420.0] and info["frame"]["gate"]["d"]["W_scope_mz"] == [90.0, 420.0]
    assert info["inner"]["model"] is None and info["inner"]["verdict"] == "axis_ok"
    assert any("band steps alone inside the W fit's lock range m/z 90-420" in s for s in said)
    m = ts["mz"].to_numpy()
    sids = ts["sample_item_id"].astype(str).to_numpy()
    d = np.zeros(len(m))
    for s in pd.unique(sids):
        d[sids == s] = frame.for_item(s).d_of(m[sids == s])
    outside = (m < 90.0 * (1 - 2e-5)) | (m > 420.0 * (1 + 2e-5))
    assert (outside & (d != 0.0)).sum() > 50                       # what the steps alone moved before
    assert np.array_equal(out["mz"].to_numpy()[outside], m[outside])
    assert np.all(out[AB.AXIS_COL].to_numpy()[outside] == 0.0)
    inside = (m > 90.0 * (1 + 2e-5)) & (m < 420.0 * (1 - 2e-5))
    assert np.array_equal(out[AB.AXIS_COL].to_numpy()[inside], d[inside]) and (d[inside] != 0.0).any()
    # the record: JSON and pickle give the same composite; to_server_axis puts every row back
    fit = MQ.fit_from_record(json.loads(json.dumps(info["wave"])))
    assert fit.inner.max_abs_ppm() == 0.0 and fit.mz_range == (90.0, 420.0)
    n_in = sum(int(fit.for_item(s).in_scope(m[sids == s]).sum()) for s in pd.unique(sids))
    assert info["n_corrected"] == n_in <= int((~outside).sum()) and n_in >= int(inside.sum())
    for s in pd.unique(sids):
        sel = sids == s
        assert np.array_equal(MQ.apply_correction(fit.for_item(s), m[sel]), out["mz"].to_numpy()[sel])
        assert np.array_equal(MQ.apply_correction(pickle.loads(pickle.dumps(fit)).for_item(s), m[sel]),
                              out["mz"].to_numpy()[sel])
    back = AB.to_server_axis(out["mz"].to_numpy(), {"applied": True, "wave": info["wave"]}, list(sids), log=_quiet)
    assert np.array_equal(back[outside], m[outside]) and np.max(np.abs(back - m) / m) < 1e-9
    # gate (d) on the real scope: an axis_ok lock range shorter than the raw model's by more than a band holds
    monkeypatch.setattr(AB, "_lock_model", _stub_axis_ok(_spline_inner(90.0, 300.0), 0.10))
    info2 = dict(info, applied=False, verdict="axis_trend", model="locks", wave=raw.as_dict(), scope_mz=[90.0, 420.0])
    out2 = AB._frame_path(ts, frame, rec, raw, None, None, info2, "-", (), _quiet)
    assert out2 is None and info2["frame"]["gate"]["failed"] == ["d"] and "upper end" in info2["frame"]["held"]
    assert info2["frame"]["gate"]["d"]["W_scope_mz"] == [90.0, 300.0]
    # a W record without its lock model bounds the steps by its recorded scope; one without either holds
    for wave, scope, applied in (({}, [90.0, 420.0], True), ({}, None, False)):
        def fake(ts_, pol, re, inf, log, wave=wave, scope=scope):
            inf["_locks_table"] = None
            inf["locks"] = {"cv_ppm": 0.10, "raw_ppm": 0.4, "steps": []}
            inf.update(model="locks", wave=wave, scope_mz=scope, max_abs_ppm=0.6, verdict="axis_ok")
            return None, True
        monkeypatch.setattr(AB, "_lock_model", fake)
        info3 = dict(info2, frame=None)
        out3 = AB._frame_path(ts, frame, rec, raw, None, None, info3, "-", (), _quiet)
        assert (out3 is not None) == applied, info3["frame"].get("held")
        if applied:
            assert info3["scope_mz"] == [90.0, 420.0] and np.array_equal(out3[AB.AXIS_COL].to_numpy()[outside], 0.0 * m[outside])
        else:
            assert "records no lock scope" in info3["frame"]["held"]


def test_the_raw_models_own_cap_hold_does_not_hold_the_frame(monkeypatch, batch):
    """measure_axis hands _frame_path the caller's hold and the raw model's cap hold
    apart: a raw model over its own cap leaves the frame to be tried, and the frame applies."""
    docs, frame, axis, ts, truth = batch
    rec = AF.FrameSet.from_documents(docs, polarity="-")[1]
    real_cap, holds = AB._cap_lock_model, []

    def cap(model, info, hold, log):
        holds.append(hold)
        if len(holds) == 1:                                             # the raw fit, held by its own cap
            return model, hold if hold is not None else "the correction reaches 9.0 ppm, above 7"
        return real_cap(model, info, hold, log)

    monkeypatch.setattr(AB, "_cap_lock_model", cap)
    out, info, _ = AB.measure_axis(ts, "NO3", "orbitrap", log=_quiet, frame=frame, frame_record=rec)
    assert holds == [None, None] and out is not ts
    assert info["model"] == "frame" and info["applied"] and info["frame"]["held"] is None
    assert not info["frame"]["gate"]["d"]["raw_applied"] and "held" not in info
    # a caller's hold still holds it, before any W fit
    holds.clear()
    out, info, _ = AB.measure_axis(ts, "NO3", "orbitrap", hold="because", log=_quiet, frame=frame, frame_record=rec)
    assert out is ts and info["frame"]["held"] == "because" and info["held"] == "because" and len(holds) == 1


def test_an_applied_frame_keeps_the_refresh_in_its_record(monkeypatch, batch):
    docs, frame, axis, ts, truth = batch
    raw = _spline_inner(80.0, 430.0)
    monkeypatch.setattr(AB, "_lock_model", _stub_lock_model(raw, 0.10))
    for refreshed in (True, False):
        rec = dict(AF.FrameSet.from_documents(docs, polarity="-")[1], source="sdk")
        if refreshed:
            rec["refreshed"] = True
        info = {"applied": False, "verdict": "axis_trend", "model": "locks", "wave": raw.as_dict(),
                "scope_mz": list(raw.mz_range), "max_abs_ppm": round(raw.max_abs_ppm(), 3),
                "locks": {"cv_ppm": 0.10, "raw_ppm": 1.2, "steps": []}}
        out = AB._frame_path(ts, frame, rec, raw, None, None, info, "-", (), _quiet)
        assert out is not None and info["model"] == "frame" and info["frame"]["source"] == "sdk"
        assert info["frame"].get("refreshed") is (True if refreshed else None)


def test_the_manifest_carries_the_frame_record_summarised_and_batch_summary_keeps_it_whole(framed_run, held_run,
                                                                                         off_run):
    from peaky import pipeline as PL
    out, res, tabs, logs = framed_run
    ax = res["summary"]["mass_axis"]
    on_disk = json.loads((out / "batch_summary.json").read_text())["mass_axis"]
    assert on_disk["frame"]["items"] and on_disk["frame"]["documents"] and on_disk["wave"]["items"]
    slim = PL._manifest_mass_axis(ax)
    assert list(slim) == list(ax) and all(slim[k] is ax[k] for k in ax if k not in ("frame", "wave"))
    groups = slim["frame"].pop("groups")
    assert slim["frame"] == {"held": None, "n_items": 4, "n_usable": 4, "n_groups": 1, "source": "directory",
                             "gate": {"passed": True, "failed": []}}
    g = ax["frame"]["groups"][0]
    assert groups == [dict({k: g[k] for k in PL._MANIFEST_GROUP_KEYS}, zone_khz=g["grid"]["zone_khz"])]
    assert slim["wave"] == {"model": "frame", "version": 1, "policy": "raw", "inner": ax["wave"]["inner"],
                            "n_items": 4}
    assert len(json.dumps(slim)) < len(json.dumps(ax)) / 3
    assert ax["frame"]["items"] and ax["wave"]["items"]                 # the run's own record untouched
    # held by the gate: the frame record summarised, the lock model's wave as recorded
    hax = held_run[1]["summary"]["mass_axis"]
    held = PL._manifest_mass_axis(hax)
    assert held["wave"] is hax["wave"] and held["frame"]["held"] == hax["frame"]["held"]
    assert held["frame"]["gate"] == {"passed": False, "failed": hax["frame"]["gate"]["failed"]}
    assert PL._manifest_mass_axis({"frame": {"held": "h", "refreshed": True}})["frame"]["refreshed"] is True
    # no frame record: the same object, so the manifest and the registry are as without the feature
    oax = off_run[1]["summary"]["mass_axis"]
    assert PL._manifest_mass_axis(oax) is oax and PL._manifest_mass_axis(None) is None


def test_measure_axis_applies_the_frame_at_the_tables_real_scatter_and_the_true_ions_land():
    """The collapsing scale end to end: four items of one calibration at a band scatter of
    0.001 ppm (within the served tables' range) with ONE 3-ppm band. The old trend put
    -2.6 ppm on the step's neighbours and 0 on the step, the gate passed and 94.7 % of the true
    ions were within 0.5 ppm afterwards; now the step band alone carries it."""
    k_step = 40
    docs = {}
    for i in range(4):
        sid = f"s{i}"
        docs[sid] = _document(steps={k_step: 3.0}, interp=(60, 61) if i < 2 else (58, 59),
                              phase=PHASE + (0.0 if i < 2 else 0.1), item=_item(sid), noise=0.001,
                              n_hits=20 if i < 2 else 36)
    frame, rec = AF.FrameSet.from_documents(docs, polarity="-")
    assert frame is not None, rec
    bands = {b["k"]: b for b in rec["groups"][0]["bands"]}
    assert abs(bands[k_step]["d_ppm"] - 3.0) < 0.05
    assert all(abs(b["d_ppm"]) < 0.05 for kk, b in bands.items() if kk != k_step)
    it = frame.for_item("s0")

    def axis(mz):
        return _smooth(mz) + np.where(it.k_of(mz) == k_step, 3.0, 0.0)

    rng = np.random.default_rng(7)
    truth = [i for i in _truth(0) if rng.random() < 0.6]
    ts, truth = _batch(axis, n_files=4, seed=0, truth=truth)
    out, info, _ = AB.measure_axis(ts, "NO3", "orbitrap", log=_quiet, frame=frame, frame_record=rec)
    assert info["applied"] and info["model"] == "frame" and info["frame"]["gate"]["passed"], info.get("frame")
    assert info["frame"]["fits"]["W"]["steps"] == []
    fit = MQ.fit_from_record(info["wave"]).for_item("s0")
    th = np.array([AL.exact_mz(i) for i in sorted(truth)])
    measured = th * (1 + axis(th) * 1e-6)
    inside = np.asarray(fit.in_scope(measured), dtype=bool)
    err = (MQ.apply_correction(fit, measured) - th) / th * 1e6
    assert inside.sum() > 150 and np.mean(np.abs(err[inside]) <= 0.5) > 0.97, np.mean(np.abs(err[inside]) <= 0.5)
    on_step = inside & (np.asarray(it.k_of(measured)) == k_step)
    assert on_step.sum() >= 3 and np.abs(err[on_step]).max() < 0.3, err[on_step]


def test_to_server_axis_counts_the_rows_without_a_preimage_and_those_with_two():
    """At a zone boundary the frame's map jumps by half a step: where d steps down the
    images leave a GAP (a corrected value there has no preimage: returned unchanged, off
    the server's axis by up to half the step) and where it steps up they OVERLAP (two
    served m/z share one corrected value: the lower returned). Both are counted in the
    inverse's record and said in its log line."""
    it = AF.load_document(_document(steps={K_MID: STEP_PPM, K_MID - 1: SIDE_PPM, _k_at(120.0): 1.1},
                                    interp=(_k_at(200.0),)), polarity="-").fingerprint.frame_item("s0")
    down = next(s for s in it.edge_steps() if s["step_ppm"] < -1.0)
    zb = float(AF.mz_of_f(it.phase + it.period * int(down["k_below"]) + it.zone_khz, it.B, it.C))
    gap = zb * (1 - 0.5 * (down["d_below"] + 0.5 * (down["d_below"] + down["d_above"])) * 1e-6)
    assert it.to_W([zb * (1 - 1e-9)])[0] < gap < it.to_W([zb * (1 + 1e-9)])[0]
    up = next(s for s in it.edge_steps() if s["step_ppm"] > 1.0)
    zb2 = float(AF.mz_of_f(it.phase + it.period * int(up["k_below"]) + it.zone_khz, it.B, it.C))
    twin_lo, twin_hi = zb2 * (1 - 0.2e-6), zb2 * (1 + 0.2e-6)
    img_hi = it.to_W([twin_hi])[0]
    assert img_hi < it.to_W([twin_lo])[0]
    plain = 250.0
    fs = AF.FrameSet(items={"s0": it.with_inner(None)})
    axis_info = {"applied": True, "model": "frame", "wave": fs.as_dict()}
    stats, said = {}, []
    back = AB.to_server_axis(np.array([gap, img_hi, it.to_W([plain])[0]]), axis_info, ["s0"] * 3, log=said.append,
                             stats=stats)
    assert back[0] == gap and back[1] < twin_lo and abs(back[2] - plain) / plain < 1e-12
    assert stats["n_no_preimage"] == 1 and stats["n_several_preimages"] == 1 and stats["n_without_item"] == 0
    assert "1 rows had no preimage" in said[0] and "1 had two" in said[0]


# --- a mixed frame per item, and the way back -------------------------------------------------------

def test_a_mixed_frame_is_corrected_per_item_and_one_ion_merges_in_W():
    fs = _two_calibrations()
    a, b = fs.for_item("a"), fs.for_item("b")
    mid_above = float(np.mean(AF.band_edges_mz(K_MID - 1, B0, C0, PERIOD, PHASE)))
    th = np.array([120.3, 250.45, STEP_MZ + 0.1, mid_above])           # true m/z, one per stepped region
    rows = []
    for sid, it in (("a", a), ("b", b)):
        served = th * (1 + it.d_of(th) * 1e-6)                        # each item's axis carries ITS steps
        rows += [(sid, m, 100.0) for m in served]
    ts = pd.DataFrame(rows, columns=["sample_item_id", "mz", "height"])
    assert not np.allclose(ts["mz"][:4].to_numpy(), ts["mz"][4:].to_numpy(), rtol=0, atol=1e-4)
    tsW = fs.to_frame(ts)
    w = tsW["mz"].to_numpy()
    assert np.allclose(w[:4], w[4:], rtol=1e-10, atol=0) and np.allclose(w[:4], th, rtol=1e-10, atol=0)
    info = {"verdict": "axis_ok"}
    corrected = AB._apply_axis(ts, fs, info, "per item", _quiet)
    assert info["applied"] and info["n_corrected"] == 8
    c = corrected["mz"].to_numpy()
    assert np.allclose(c[:4], MQ.apply_correction(a, ts["mz"][:4].to_numpy()), rtol=0, atol=1e-12)
    assert np.allclose(c[4:], MQ.apply_correction(b, ts["mz"][4:].to_numpy()), rtol=0, atol=1e-12)
    assert np.allclose(c, ts["mz"].to_numpy() * (1 - corrected[AB.AXIS_COL].to_numpy() * 1e-6), rtol=0, atol=1e-12)
    # the way back, per source item: a row from one item comes back exactly; a merged row of
    # both items has two inverses -- the median is taken, the spread recorded
    axis_info = {"applied": True, "wave": fs.as_dict()}
    with pytest.raises(TypeError, match="srcs"):
        AB.to_server_axis(c, axis_info)
    srcs = ["a"] * 4 + ["b"] * 4
    back = AB.to_server_axis(c, axis_info, srcs)
    assert np.allclose(back, ts["mz"].to_numpy(), rtol=1e-12, atol=0)
    stats, said = {}, []
    merged = np.array([th[2], th[2], th[0]])
    back2 = AB.to_server_axis(merged, axis_info, ["a,b", "a", "zzz"], log=said.append, stats=stats)
    inv_a, inv_b = a.invert(merged[:1])[0], b.invert(merged[:1])[0]
    assert abs(back2[0] - 0.5 * (inv_a + inv_b)) < 1e-9 and abs(back2[1] - inv_a) < 1e-9 and back2[2] == merged[2]
    assert stats["n_rows"] == 3 and stats["n_multi_source"] == 1 and stats["n_without_item"] == 1
    assert abs(stats["spread_ppm_max"] - abs(inv_a - inv_b) / back2[0] * 1e6) < 1e-3 and stats["n_spread_over"] == 1
    assert said and "spread" in said[0]
    # a plain lock-model record ignores srcs, as today
    model = _spline_inner(80.0, 430.0)
    lock_info = {"applied": True, "wave": model.as_dict()}
    raw = np.array([150.0, 250.0, 400.0])
    assert np.array_equal(AB.to_server_axis(MQ.apply_correction(model, raw), lock_info, ["a", "b", "a,b"]),
                          AB.to_server_axis(MQ.apply_correction(model, raw), lock_info))


def test_one_calibration_puts_every_merged_row_back_exactly_whatever_its_sources(batch):
    docs, frame, axis, ts, truth = batch
    composite = frame.with_inner(_spline_inner(90.0, 420.0))
    axis_info = {"applied": True, "model": "frame", "wave": composite.as_dict()}
    rng = np.random.default_rng(3)
    raw = np.sort(rng.uniform(95.0, 415.0, 400))
    sids = frame.sample_ids
    srcs = [",".join(sorted(rng.choice(sids, size=int(rng.integers(1, 4)), replace=False))) for _ in raw]
    corrected = MQ.apply_correction(composite.for_item(sids[0]), raw)
    stats = {}
    back = AB.to_server_axis(corrected, axis_info, srcs, log=_quiet, stats=stats)
    assert np.max(np.abs(back - raw) / raw * 1e6) < 1e-3
    assert stats["n_spread_over"] == 0 and stats["spread_ppm_max"] < 1e-6 and stats["n_without_item"] == 0
    assert stats["n_multi_source"] == sum("," in s for s in srcs) > 50


# --- the CLI and the pipelines -----------------------------------------------------------------------

def test_the_cli_flags_default_to_auto_and_reach_both_pipelines(monkeypatch):
    from peaky import cli
    from peaky import pipeline as PL
    p = cli.build_parser()
    for argv in (["batch", "--batch", "b"], ["pool", "--batches", "b"]):
        a = p.parse_args(argv)
        assert a.axis_fingerprints == "auto" and a.axis_fingerprint_dir is None
    assert p.parse_args(["batch", "--batch", "b", "--axis-fingerprints", "refresh"]).axis_fingerprints == "refresh"
    with pytest.raises(SystemExit):
        p.parse_args(["batch", "--batch", "b", "--axis-fingerprints", "maybe"])
    seen = {}

    def fake(**kw):
        seen.update(kw)
        raise SystemExit(0)

    monkeypatch.setattr(cli, "_require_creds", lambda *a, **k: None)
    monkeypatch.setattr(PL, "run_batch", fake)
    monkeypatch.setattr(PL, "run_pooled_batches", fake)
    for argv in (["batch", "--batch", "b"], ["pool", "--batches", "b"]):
        with pytest.raises(SystemExit):
            cli.main(argv + ["--axis-fingerprints", "off", "--axis-fingerprint-dir", "docs", "--no-progress"])
        assert seen.pop("axis_fingerprints") == "off" and seen.pop("axis_fingerprint_dir") == "docs"
        with pytest.raises(SystemExit):
            cli.main(argv + ["--no-progress"])
        assert seen.pop("axis_fingerprints") == "auto" and seen.pop("axis_fingerprint_dir") is None


def test_the_pipelines_hand_the_flags_to_the_run_as_explicit_kwargs(refs, tmp_path, monkeypatch):
    import inspect
    from peaky import pipeline as PL
    from peaky.reporting import provenance as PV
    # the kwargs are the run's own, never assign_kw's: run() names them
    params = inspect.signature(AB.run).parameters
    assert params["axis_fingerprints"].default == "auto" and params["axis_fingerprint_dir"].default is None
    assert "P.registry_extras(), axis_args)" in inspect.getsource(AB.run)
    ts = _orbi_batch(np.random.default_rng(43), refs).assign(sample_batch_name="only batch")
    seen = {}

    def fake_run(**kw):
        seen.update(kw)
        return {"summary": {"mass_axis": {"applied": False}}, "ts_peaks": None, "residual_samples": None}

    monkeypatch.setattr(IO, "connect", lambda *a, **k: None)
    monkeypatch.setattr(IO, "resolve_batch", lambda c, b, dataset=None: SimpleNamespace(id="bid", name="axis batch"))
    monkeypatch.setattr(AB, "run", fake_run)
    monkeypatch.setattr(PL, "generate_report", lambda ctx, ts, **k: {})
    monkeypatch.setattr(PV, "record_run", lambda **kw: None)
    PL.run_batch(batch="axis batch", reagent="NO3", base_out=str(tmp_path), ts=ts, do_report=False, log=_quiet,
                 axis_fingerprints="off", axis_fingerprint_dir="docs")
    assert seen["axis_fingerprints"] == "off" and seen["axis_fingerprint_dir"] == "docs"
    assert "cfg" in seen and seen["mass_axis"] == "auto"
    seen.clear()
    PL.run_batch(batch="axis batch", reagent="NO3", base_out=str(tmp_path), ts=ts, do_report=False, log=_quiet)
    assert seen["axis_fingerprints"] == "auto" and seen["axis_fingerprint_dir"] is None
    seen.clear()
    PL.run_pooled_batches(batches="batch", reagent="NO3", base_out=str(tmp_path), ts=ts, do_report=False,
                          per_group_reports=False, log=_quiet, axis_fingerprints="off", axis_fingerprint_dir="docs")
    assert seen["axis_fingerprints"] == "off" and seen["axis_fingerprint_dir"] == "docs"
    assert seen["mass_axis_hold"] is None                              # a one-batch pool behaves like a batch
