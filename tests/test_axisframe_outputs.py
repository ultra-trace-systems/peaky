"""The band-step frame's outputs beyond the run (batch.axisframe, plumbed by
assign_batch): publishing a frame run puts every merged row back on the server's axis
per source item, the PDF's coverage and batch-max tables read the corrected series when
the frame reaches past 4 ppm, the manifest records the documents read and nothing
else changes without a frame. Synthetic documents and series only."""
import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from peaky import pipeline as PL
from peaky.batch import assign_batch as AB
from peaky.batch import axisframe as AF
from peaky.batch import axislock as AL
from peaky.batch import massqc as MQ
from peaky.chem import reference_ions as RI
from peaky.io import io_mascope as IO
from peaky.io import publish as P
from peaky.reporting import pdf_report as R
from peaky.reporting import provenance as PV
from tests.test_axisframe import (K_MID, SIDE_PPM, STEP_MZ, STEP_PPM, _document, _item, _spline_inner,
                                  _two_document_batch)
from tests.test_axislock import _truth
from tests.test_massqc import _orbi_batch

ROOT = Path(__file__).resolve().parents[1]
MODE = "-|R280000|50.0000-750.0000"
#: a STEP_PPM band at m/z ~294-315 and SIDE_PPM on every band above it (test_axisframe_wiring)
STEPS = {K_MID: STEP_PPM, **{k: SIDE_PPM for k in range(18, K_MID)}}


def _quiet(*a, **k):
    return None


@pytest.fixture(autouse=True)
def _no_axis_left_behind():
    IO.clear_axis_correction()
    yield
    IO.clear_axis_correction()


# --- helpers ---------------------------------------------------------------------------------

def _composite(steps=STEPS, n_items=2, inner="spline"):
    """A one-group frame of `n_items` items from two documents of one calibration
    (test_axisframe) around a spline lock model (or none): what a run that applied the
    frame records as batch_summary['mass_axis']['wave']."""
    docs = _two_document_batch(steps, n_items=n_items)
    frame, rec = AF.FrameSet.from_documents(docs, polarity="-")
    assert frame is not None, rec
    return frame.with_inner(_spline_inner() if inner == "spline" else inner)


def _raw_mz(frame, lo=100.0, hi=400.0, n=60):
    """Served m/z of true ions inside the composite's scope and outside its edge zones."""
    it = frame.for_item(frame.sample_ids[0])
    th = np.array(sorted(AL.exact_mz(i) for i in _truth(0)))
    th = th[(th > lo) & (th < hi)]
    keep = np.asarray(it.in_scope(th), dtype=bool) & ~np.asarray(it.edge_ambiguous(th), dtype=bool)
    return th[keep][:n]


def _corrected(frame, raw, srcs):
    """The merged ledger's m/z: the mean of the source items' corrected readings."""
    return np.array([np.mean([MQ.apply_correction(frame.for_item(s), np.array([m]))[0] for s in src.split(",")])
                     for m, src in zip(raw, srcs)])


def _write_run(out: Path, mz, srcs, record, *, model="frame", **summary):
    out.mkdir(parents=True, exist_ok=True)
    led = pd.DataFrame({"mz": mz, "neutral_formula": ["C3H4O4"] * len(mz), "adduct": ["[M-H]-"] * len(mz),
                        "tier": ["Assigned"] * len(mz)})
    if srcs is not None:
        led["srcs"] = list(srcs)
    led.to_csv(out / "merged_ledger.csv", index=False)
    ax = {"applied": True, "model": model, "wave": record.as_dict(), **summary}
    (out / "batch_summary.json").write_text(json.dumps({"mass_axis": ax, "sample_ids": ["s0", "s1"]}))


# --- publish: the merged rows back on the server's axis --------------------------------------

def test_publish_puts_a_frame_run_back_on_the_servers_axis_per_source_item(tmp_path):
    comp = _composite()
    raw = _raw_mz(comp)
    assert len(raw) >= 20
    srcs = [("s0", "s1", "s0,s1")[i % 3] for i in range(len(raw))]
    corr = _corrected(comp, raw, srcs)
    assert np.max(np.abs(corr - raw) / raw * 1e6) > 1.0            # the frame did move them
    _write_run(tmp_path, corr, srcs, comp)
    logs = []
    loaded = P.load_batch_run(str(tmp_path), log=logs.append)
    got = loaded["merged"]["mz"].to_numpy(dtype=float)
    assert np.max(np.abs(got - raw) / raw) < 1e-9                     # exact, row by row
    inv = loaded["axis_inverse"]
    assert inv["n_rows"] == len(raw) and inv["n_multi_source"] == sum("," in s for s in srcs)
    assert inv["n_without_item"] == 0 and inv["n_spread_over"] == 0
    assert inv["n_no_preimage"] == 0 and inv["n_several_preimages"] == 0      # no row at a zone boundary
    assert inv["spread_warn_ppm"] == 2.5 and inv["n_spread_over_warn"] == 0   # half of the 5-ppm default
    assert "spread_ppm" not in inv                                     # the per-row array stays out of the record
    assert any("frame inverse" in line for line in logs)
    assert not any(line.startswith("[publish-batch]") for line in logs)   # nothing to warn about
    # the rest of the ledger travels untouched
    assert list(loaded["merged"]["srcs"]) == srcs and loaded["summary"]["mass_axis"]["model"] == "frame"


def test_publish_refuses_a_frame_run_whose_ledger_has_no_srcs(tmp_path):
    comp = _composite()
    raw = _raw_mz(comp, n=5)
    _write_run(tmp_path, _corrected(comp, raw, ["s0"] * len(raw)), None, comp)
    with pytest.raises(P.PublishError, match="srcs"):
        P.load_batch_run(str(tmp_path))


def test_publish_warns_when_a_rows_sources_disagree_by_more_than_half_the_tolerance(tmp_path):
    """Two items on two calibrations in one hand-built frame (what a run never applies: the
    frame is one calibration group) -- the row's inverses differ by the band steps' gap."""
    a = AF.load_document(_document(steps={K_MID: STEP_PPM}, item=_item("a")), polarity="-")
    b = AF.load_document(_document(steps={K_MID: -1.8}, item=_item("b")), polarity="-")
    assert a.ok and b.ok
    frame = AF.FrameSet(items={"a": a.fingerprint.frame_item("a"), "b": b.fingerprint.frame_item("b")})
    ita, itb = frame.for_item("a"), frame.for_item("b")
    lo, hi = AF.band_edges_mz(np.array([K_MID]), ita.B, ita.C, ita.period, ita.phase)
    raw = np.linspace(float(lo[0]) + 2.0, float(hi[0]) - 2.0, 8)   # inside the stepped band, off its edges
    for it in (ita, itb):
        assert np.asarray(it.in_scope(raw), dtype=bool).all() and not np.asarray(it.edge_ambiguous(raw)).any()
    corr = _corrected(frame, raw, ["a,b"] * len(raw))
    _write_run(tmp_path, corr, ["a,b"] * len(raw), frame)
    logs = []
    loaded = P.load_batch_run(str(tmp_path), log=logs.append)      # tolerance 5 ppm: warn above 2.5
    inv = loaded["axis_inverse"]
    assert inv["n_multi_source"] == len(raw) and inv["spread_ppm_max"] > 3.5
    assert inv["n_spread_over_warn"] == len(raw) and inv["spread_warn_ppm"] == 2.5
    warned = [line for line in logs if line.startswith("[publish-batch]")]
    assert len(warned) == 1 and f"{len(raw)} merged row(s)" in warned[0] and "2.5 ppm" in warned[0]
    # the median was sent: between the two sources' readings
    got = loaded["merged"]["mz"].to_numpy(dtype=float)
    inv_a, inv_b = MQ.invert_correction(ita, corr), MQ.invert_correction(itb, corr)
    assert np.allclose(got, 0.5 * (inv_a + inv_b), rtol=0, atol=1e-9)
    # a wider tolerance: nothing to warn about
    quiet = []
    again = P.load_batch_run(str(tmp_path), tolerance_ppm=20.0, log=quiet.append)
    assert again["axis_inverse"]["n_spread_over_warn"] == 0 and again["axis_inverse"]["spread_warn_ppm"] == 10.0
    assert not any(line.startswith("[publish-batch]") for line in quiet)


def test_publish_loads_a_lock_model_run_exactly_as_before(tmp_path):
    inner = _spline_inner()
    raw = np.array([150.0, 250.0, 380.0, 410.0])
    corr = MQ.apply_correction(inner, raw)
    _write_run(tmp_path, corr, ["s0", "s1", "s0,s1", "s0"], inner, model="locks")
    logs = []
    loaded = P.load_batch_run(str(tmp_path), log=logs.append)
    got = loaded["merged"]["mz"].to_numpy(dtype=float)
    assert np.max(np.abs(got - raw) / raw) < 1e-9
    assert "axis_inverse" not in loaded and logs == []
    assert set(loaded) == {"merged", "summary", "ion_formulas", "root"}


def test_the_publish_cli_hands_load_batch_run_its_tolerance(monkeypatch, tmp_path):
    from peaky import cli

    class Stop(Exception):
        pass

    seen = {}

    def fake(run_dir, **kw):
        seen.update(run_dir=run_dir, **kw)
        raise Stop

    monkeypatch.setattr(P, "load_batch_run", fake)
    args = cli.build_parser().parse_args(["publish-batch", str(tmp_path), "--tolerance-ppm", "3", "--dry-run"])
    with pytest.raises(Stop):
        cli.cmd_publish_batch(args)
    assert seen["run_dir"] == str(tmp_path) and seen["tolerance_ppm"] == 3.0 and seen["log"] is print


# --- the report: the coverage tables read the corrected series past 4 ppm ---------------------

def _twenty_ppm_frame():
    """A one-group frame whose band at m/z ~294-315 is 20 ppm off and nothing else (no
    smooth part): the correction a lock model could never apply (its cap is 7 ppm)."""
    docs = _two_document_batch({K_MID: 20.0}, n_items=2)
    frame, rec = AF.FrameSet.from_documents(docs, polarity="-")
    assert frame is not None, rec
    it = frame.for_item("s0")
    assert abs(float(it.d_of(np.array([STEP_MZ]))[0]) - 20.0) < 2.0, it.d_of(np.array([STEP_MZ]))
    assert it.max_abs_ppm() > 15.0
    return frame


def _report_run(out: Path, frame):
    """A run folder around `frame`: the series on the served axis (data/NO3_ts.parquet), the
    corrected series, and the ledgers on the corrected axis. Returns (ctx, res, th, in_band)."""
    it = frame.for_item("s0")
    th = np.linspace(STEP_MZ - 60.0, STEP_MZ + 80.0, 41)    # one ion every 3.5 Da, 6 of them in the stepped band
    th = th[~np.asarray(it.edge_ambiguous(th), dtype=bool)]
    d = np.asarray(it.d_of(th), dtype=float)
    in_band = np.abs(d) > 10.0
    assert 4 <= in_band.sum() < len(th)
    raw_mz = th * (1 + d * 1e-6)                          # served: the band steps on the axis
    t0 = pd.Timestamp("2026-01-01T00:00:00Z")
    rows = []
    for s, sid in enumerate(("s0", "s1")):
        for m, h in zip(raw_mz, np.linspace(100.0, 500.0, len(th)) * (1 + 0.5 * s)):
            rows.append((sid, m, h, t0 + pd.Timedelta(hours=s)))
    raw = pd.DataFrame(rows, columns=["sample_item_id", "mz", "height", "datetime_utc"])
    corrected = raw.assign(mz=lambda f: MQ.apply_correction(it, f["mz"].to_numpy(dtype=float)))
    corrected[AB.AXIS_COL] = raw["mz"].to_numpy() * 0 + np.tile(d, 2)
    assert np.allclose(corrected["mz"].to_numpy(), np.tile(th, 2), rtol=0, atol=1e-6)
    (out / "per_file").mkdir(parents=True, exist_ok=True)
    (out / "tables").mkdir(exist_ok=True)
    (out / "data").mkdir(exist_ok=True)
    pd.DataFrame({"mz": th, "neutral_formula": [f"C{10 + i}H16O3" for i in range(len(th))],
                  "adduct": ["[M+NO3]-"] * len(th), "tier": ["Assigned"] * len(th), "ion_score": 0.9,
                  "n_files": 2, "formula_agree": True, "admitted_by": "height", "occurrence": 1.0,
                  "srcs": ["s0,s1"] * len(th)}).to_csv(out / "merged_ledger.csv", index=False)
    pd.DataFrame({"peak_id": np.arange(len(th)), "mz": th, "role": ["M0"] * len(th),
                  "neutral_formula": [f"C{10 + i}H16O3" for i in range(len(th))], "adduct": ["[M+NO3]-"] * len(th),
                  "height": np.linspace(100.0, 500.0, len(th)), "tier": ["Assigned"] * len(th),
                  "ppm_error": 0.1, "parent_peak_id": None}).to_csv(out / "per_file" / "s0_ledger.csv", index=False)
    ts_path = str(out / "data" / "NO3_ts.parquet")
    raw.to_parquet(ts_path)
    ctx = PL.RunContext(out_dir=str(out), batch_name="axis batch", tag="NO3", label="NO3- CIMS",
                        when=datetime(2026, 1, 1, tzinfo=timezone.utc), run_id="rid", generated="g",
                        dataset="D", ts_path=ts_path)
    ax = {"applied": True, "model": "frame", "verdict": "axis_ok", "wave": frame.as_dict(),
          "max_abs_ppm": round(it.max_abs_ppm(), 3),
          "shift_ppm": {"min": round(float(d.min()), 3), "median": 0.0, "max": round(float(d.max()), 3)}}
    res = {"summary": {"mass_axis": ax}, "ts_peaks": corrected}
    return ctx, res, th, in_band


def test_a_frame_past_4_ppm_hands_the_report_the_corrected_series_and_its_tables_match(tmp_path):
    frame = _twenty_ppm_frame()
    ctx, res, th, in_band = _report_run(tmp_path, frame)
    recorded = ctx.ts_path
    logs = []
    path = PL._report_ts_path(ctx, res, log=logs.append)
    assert path == str(tmp_path / "data" / "NO3_ts_corrected.parquet") and Path(path).exists()
    assert ctx.ts_path == recorded                                  # the recorded input never moves
    assert PL._report_ts(ctx, pd.read_parquet(recorded), res) is res["ts_peaks"]   # the figures' series
    written = pd.read_parquet(path)
    assert np.allclose(written["mz"].to_numpy(), res["ts_peaks"]["mz"].to_numpy(), rtol=0, atol=0)
    assert AB.AXIS_COL in written.columns
    assert any("corrected series" in line and "20." in line for line in logs)
    # the coverage and batch-max tables: on the recorded raw series the 20-ppm band's bins
    # miss the 8-ppm match; on the corrected one every ion's bin is explained
    on_raw = R.load_context(str(tmp_path), tag="NO3", label="x", ts_path=recorded)
    on_cor = R.load_context(str(tmp_path), tag="NO3", label="x", ts_path=path)
    assert on_raw["ts"]["nbins"] == on_cor["ts"]["nbins"] == len(th)
    assert on_raw["ts"]["expl_count"] == len(th) - int(in_band.sum())
    assert on_cor["ts"]["expl_count"] == len(th)
    assert len(on_raw.get("max_h_by_channel", {})) == len(th) - int(in_band.sum())
    assert len(on_cor["max_h_by_channel"]) == len(th) and on_cor["max_h_scope"] == "batch"
    # the whole-batch max is the brighter (second) sample's, read off the corrected bins
    assert all(v > 100.0 for v in on_cor["max_h_by_channel"].values())


def test_without_a_frame_past_4_ppm_the_report_reads_the_recorded_series(tmp_path):
    frame = _twenty_ppm_frame()
    ctx, res, th, in_band = _report_run(tmp_path, frame)
    ax = res["summary"]["mass_axis"]
    # a lock model, however far it reaches (it is capped at 7 ppm anyway)
    lock = {"summary": {"mass_axis": {**ax, "model": "locks", "max_abs_ppm": 6.5}}, "ts_peaks": res["ts_peaks"]}
    assert PL._report_ts_path(ctx, lock, log=_quiet) is None
    # a frame within the report's margin
    small = {"summary": {"mass_axis": {**ax, "max_abs_ppm": 3.9, "shift_ppm": {"min": -3.9, "median": 0, "max": 2}}},
             "ts_peaks": res["ts_peaks"]}
    assert PL._report_ts_path(ctx, small, log=_quiet) is None
    # a frame with no series handed back, nothing applied, no record
    assert PL._report_ts_path(ctx, {"summary": {"mass_axis": ax}}, log=_quiet) is None
    assert PL._report_ts_path(ctx, {"summary": {"mass_axis": {**ax, "applied": False}}, "ts_peaks": res["ts_peaks"]},
                              log=_quiet) is None
    assert PL._report_ts_path(ctx, {"summary": {}}, log=_quiet) is None and PL._report_ts_path(ctx, None) is None
    assert not (tmp_path / "data" / "NO3_ts_corrected.parquet").exists()
    assert PL._frame_reach_ppm({}) == 0.0 and PL._frame_reach_ppm({"max_abs_ppm": 2.0, "shift_ppm": {"min": -4.5}}) == 4.5
    assert PL.REPORT_FRAME_CORRECTED_TS_PPM == 4.0


def test_generate_report_hands_build_the_recorded_series_unless_given_a_report_path(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(R, "build", lambda out_dir, **kw: seen.append((out_dir, dict(kw))) or str(tmp_path / "r.pdf"))
    monkeypatch.setattr(R, "compress_pdf", lambda path, log=None: None)
    ts = pd.DataFrame({"sample_item_id": ["s0", "s1"], "mz": [200.0, 200.0], "height": [1.0, 2.0],
                       "datetime_utc": pd.to_datetime(["2026-01-01T00:00Z", "2026-01-01T01:00Z"])})
    (tmp_path / "data").mkdir()
    ts_path = str(tmp_path / "data" / "NO3_ts.parquet")
    ts.to_parquet(ts_path)
    ctx = PL.RunContext(out_dir=str(tmp_path), batch_name="axis batch", tag="NO3", label="NO3- CIMS",
                        when=datetime(2026, 1, 1, tzinfo=timezone.utc), run_id="rid", generated="g",
                        dataset="D", ts_path=ts_path)
    PL.generate_report(ctx, ts, do_cluster=False, do_vk=False, log=_quiet)
    want = {"tag": "NO3", "label": "NO3- CIMS", "ts_path": ts_path, "batch_name": "axis batch",
            "dataset": "D", "run_id": "rid", "generated": "g"}
    assert seen == [(str(tmp_path), want)]                        # byte for byte what build got before
    other = str(tmp_path / "data" / "NO3_ts_corrected.parquet")
    PL.generate_report(ctx, ts, do_cluster=False, do_vk=False, log=_quiet, report_ts_path=other)
    assert seen[1] == (str(tmp_path), {**want, "ts_path": other})
    assert ctx.ts_path == ts_path


# --- the pipeline: the report path and the manifest's inputs through run_batch -----------------

def _stub_run(monkeypatch, summary, fixed):
    """run_batch with the assign, the report and the manifest stubbed: what each was handed."""
    seen = {}
    monkeypatch.setattr(IO, "connect", lambda *a, **k: None)
    monkeypatch.setattr(IO, "resolve_batch", lambda c, b, dataset=None: SimpleNamespace(id="bid", name="axis batch"))
    monkeypatch.setattr(AB, "run", lambda **kw: {"summary": summary, "ts_peaks": fixed, "sample_ids": ["s0000"]})
    monkeypatch.setattr(PL, "generate_report", lambda ctx, ts, **k: seen.update(report=dict(k), ctx=ctx, ts=ts) or {})
    monkeypatch.setattr(PV, "record_run", lambda **kw: seen.update(record=dict(kw)))
    return seen


DOCS = {"s0000": {"usable": True, "status": "ok", "mode_id": MODE, "sha1": "a" * 40, "notes": []},
        "s0001": {"usable": False, "status": None, "mode_id": None, "sha1": None, "notes": [], "why": "no document served"},
        "s0002": {"usable": False, "status": "missing", "mode_id": None, "sha1": "b" * 40, "notes": [],
                  "why": "status 'missing'"}}


def test_the_run_writes_the_corrected_series_for_the_report_and_records_the_documents_read(monkeypatch, tmp_path):
    ts = _orbi_batch(np.random.default_rng(49), RI.nitrate())
    fixed = ts.assign(mz=ts["mz"] * (1 - 20e-6), **{AB.AXIS_COL: 20.0})
    summary = {"mass_axis": {"applied": True, "model": "frame", "verdict": "axis_ok", "max_abs_ppm": 20.0,
                             "shift_ppm": {"min": 0.0, "median": 20.0, "max": 20.0},
                             "frame": {"held": None, "documents": DOCS}}}
    seen = _stub_run(monkeypatch, summary, fixed)
    PL.run_batch(batch="axis batch", reagent="NO3", base_out=str(tmp_path), ts=ts, log=_quiet)
    ctx = seen["ctx"]
    assert seen["ts"] is fixed                                              # the figures' series
    assert ctx.ts_path.endswith("NO3_ts.parquet")                            # the recorded input: raw
    assert np.allclose(pd.read_parquet(ctx.ts_path)["mz"].to_numpy(), ts["mz"].to_numpy(), rtol=0, atol=1e-9)
    path = seen["report"]["report_ts_path"]
    assert path == str(Path(ctx.out_dir) / "data" / "NO3_ts_corrected.parquet") and Path(path).exists()
    assert np.allclose(pd.read_parquet(path)["mz"].to_numpy(), fixed["mz"].to_numpy(), rtol=0, atol=1e-9)
    # the manifest: the documents READ (a sha1 exists), whatever their status; not the unserved one
    assert seen["record"]["axis_fingerprints"] == {
        "s0000": {"sha1": "a" * 40, "status": "ok", "mode_id": MODE},
        "s0002": {"sha1": "b" * 40, "status": "missing", "mode_id": None}}
    assert seen["record"]["ts_path"] == ctx.ts_path
    # the manifest's counts carry the frame record summarised; the summary keeps it whole
    assert seen["record"]["counts"]["mass_axis"] == PL._manifest_mass_axis(summary["mass_axis"])
    assert seen["record"]["counts"]["mass_axis"]["frame"] == {"held": None, "n_items": 3, "n_usable": 1,
                                                              "n_groups": 0, "groups": [], "source": None}
    assert summary["mass_axis"]["frame"]["documents"] is DOCS


def test_a_lock_model_run_hands_the_report_and_the_manifest_what_they_got_before(monkeypatch, tmp_path):
    ts = _orbi_batch(np.random.default_rng(50), RI.nitrate())
    fixed = ts.assign(mz=ts["mz"] * (1 - 6e-6))
    summary = {"mass_axis": {"applied": True, "model": "locks", "verdict": "axis_trend", "max_abs_ppm": 6.0,
                             "shift_ppm": {"min": 0.0, "median": 6.0, "max": 6.0}}}
    seen = _stub_run(monkeypatch, summary, fixed)
    PL.run_batch(batch="axis batch", reagent="NO3", base_out=str(tmp_path), ts=ts, log=_quiet)
    assert seen["report"]["report_ts_path"] is None and seen["ts"] is fixed
    assert seen["record"]["axis_fingerprints"] is None
    assert seen["record"]["counts"]["mass_axis"] is summary["mass_axis"]        # no frame: as recorded
    assert not list(Path(seen["ctx"].out_dir).glob("data/*_corrected.parquet"))
    # a frame looked for but held (documents read): the manifest still records them, the report is as before
    held = {"mass_axis": {"applied": False, "model": "locks", "verdict": "locks_unpredictive",
                          "frame": {"held": "gate (a): a step beside a grid edge", "documents": DOCS}}}
    seen = _stub_run(monkeypatch, held, None)
    PL.run_batch(batch="axis batch", reagent="NO3", base_out=str(tmp_path), ts=ts, log=_quiet)
    assert seen["report"]["report_ts_path"] is None
    assert set(seen["record"]["axis_fingerprints"]) == {"s0000", "s0002"}
    # no report: nothing written for it either
    seen = _stub_run(monkeypatch, summary, fixed)
    PL.run_batch(batch="axis batch", reagent="NO3", base_out=str(tmp_path), ts=ts, do_report=False, log=_quiet)
    assert seen["report"]["report_ts_path"] is None


def test_a_pools_per_group_reports_read_their_slice_of_the_corrected_series_past_4_ppm(monkeypatch, tmp_path):
    """A one-batch pool is corrected like a batch; its per-group reports (groups by another
    column) read their own slice of the pool's corrected series when the frame passes 4
    ppm, written under the group's data/; a lock-model pool hands them nothing (the raw
    slice, as before)."""
    ts = _orbi_batch(np.random.default_rng(51), RI.nitrate()).assign(sample_batch_name="only batch")
    sids = sorted(ts["sample_item_id"].unique())
    zone_a = set(sids[: len(sids) // 2])
    ts["zone"] = np.where(ts["sample_item_id"].isin(zone_a), "zone A", "zone B")
    fixed = ts.assign(mz=ts["mz"] * (1 - 20e-6), **{AB.AXIS_COL: 20.0})
    calls = []
    monkeypatch.setattr(IO, "connect", lambda *a, **k: None)
    monkeypatch.setattr(PL, "generate_report", lambda ctx, t, **k: calls.append((ctx, t, dict(k))) or {})
    monkeypatch.setattr(PV, "record_run", lambda **kw: None)

    def run(summary, base):
        calls.clear()
        monkeypatch.setattr(AB, "run", lambda **kw: {"summary": summary, "ts_peaks": fixed, "residual_samples": None,
                                                     "sample_ids": sids})
        # each run under its own base: two pool runs within one second share a run-folder name
        PL.run_pooled_batches(batches="only", reagent="NO3", base_out=str(tmp_path / base), ts=ts, group_by="zone",
                              per_group_reports=True, log=_quiet)
        assert [c[0].batch_name for c in calls] == ["only (pooled)", "zone A", "zone B"]
        return calls

    framed = {"mass_axis": {"applied": True, "model": "frame", "verdict": "axis_ok", "max_abs_ppm": 20.0,
                            "shift_ppm": {"min": 0.0, "median": 20.0, "max": 20.0}, "frame": {"held": None}}}
    pool, a, b = run(framed, "framed")
    assert pool[2]["report_ts_path"] == str(Path(pool[0].out_dir) / "data" / "NO3_ts_corrected.parquet")
    for (gctx, dfg, kw), zone in ((a, zone_a), (b, set(sids) - zone_a)):
        path = kw["report_ts_path"]
        assert path == str(Path(gctx.out_dir) / "data" / "NO3_ts_corrected.parquet") and Path(path).exists()
        got = pd.read_parquet(path)
        assert set(got["sample_item_id"]) == zone and set(dfg["sample_item_id"]) == zone
        want = fixed[fixed["sample_item_id"].isin(zone)]
        assert len(got) == len(want) and np.allclose(np.sort(got["mz"]), np.sort(want["mz"]), rtol=0, atol=1e-9)
        assert AB.AXIS_COL in got.columns
    locks = {"mass_axis": {"applied": True, "model": "locks", "verdict": "axis_trend", "max_abs_ppm": 6.0,
                           "shift_ppm": {"min": 0.0, "median": 6.0, "max": 6.0}}}
    pool, a, b = run(locks, "locks")
    assert pool[2]["report_ts_path"] is None and a[2]["report_ts_path"] is None and b[2]["report_ts_path"] is None
    assert not list(Path(a[0].out_dir).glob("data/*_corrected.parquet"))


def test_the_manifest_inputs_name_the_documents_read_and_nothing_without_them():
    assert PL._axis_fingerprint_inputs(None) is None and PL._axis_fingerprint_inputs({}) is None
    assert PL._axis_fingerprint_inputs({"mass_axis": {"applied": True, "model": "locks"}}) is None
    assert PL._axis_fingerprint_inputs({"mass_axis": {"frame": {"held": "off"}}}) is None
    none_served = {"mass_axis": {"frame": {"documents": {"s1": {"usable": False, "sha1": None, "status": None}}}}}
    assert PL._axis_fingerprint_inputs(none_served) is None
    got = PL._axis_fingerprint_inputs({"mass_axis": {"frame": {"documents": DOCS}}})
    assert got == {"s0000": {"sha1": "a" * 40, "status": "ok", "mode_id": MODE},
                   "s0002": {"sha1": "b" * 40, "status": "missing", "mode_id": None}}


def test_the_manifest_carries_the_documents_read_and_no_key_otherwise(tmp_path):
    from peaky.assignment.passes import PassConfig
    (tmp_path / "merged_ledger.csv").write_text("mz,neutral_formula,tier\n100.0,CH4,Assigned\n")
    kw = dict(run_dir=str(tmp_path), batch_name="B", dataset="D", sample_ids=["s1"], reagent="NO3",
              cfg=PassConfig(), created_utc="2026-01-01T00:00:00Z")
    m0 = PV.build_manifest(**kw)
    assert "axis_fingerprints" not in m0["input"]
    docs = {"s1": {"sha1": "f" * 40, "status": "ok", "mode_id": MODE}}
    m1 = PV.build_manifest(**kw, axis_fingerprints=docs)
    assert m1["input"]["axis_fingerprints"] == docs
    assert {k: v for k, v in m1["input"].items() if k != "axis_fingerprints"} == m0["input"]
    assert list(m1) == list(m0)                                   # the same sections, in the same order
    # and through record_run onto disk, both ways
    PV.record_run(**kw, base_out=str(tmp_path / "base"), axis_fingerprints=docs, log=_quiet)
    assert json.loads((tmp_path / "run_manifest.json").read_text())["input"]["axis_fingerprints"] == docs
    PV.record_run(**kw, base_out=str(tmp_path / "base"), log=_quiet)
    assert "axis_fingerprints" not in json.loads((tmp_path / "run_manifest.json").read_text())["input"]


# --- the docs ---------------------------------------------------------------------------------

def test_the_docs_name_every_fingerprint_flag_and_record():
    files = [ROOT / "README.md", ROOT / "CHANGELOG.md", *sorted((ROOT / "docs").glob("*.md"))]
    texts = {p.name: p.read_text(encoding="utf-8") for p in files}
    for flag in ("--axis-fingerprints", "--axis-fingerprint-dir", "--mass-axis"):
        assert any(flag in t for t in texts.values()), flag
    page = texts["MASS_AXIS.md"]
    for token in ("--axis-fingerprints", "--axis-fingerprint-dir", "mass_axis_frame.csv", "mass_axis_locks.csv",
                  "axis_fingerprints", "all or nothing", "byte", "stitched", "interpolated", "_ts_corrected.parquet",
                  "mascope-sdk", "srcs"):
        assert token.lower() in page.lower(), token
    assert "MASS_AXIS.md" in texts["PIPELINE.md"] and "--axis-fingerprints" in texts["README.md"]
    assert "--axis-fingerprints" in texts["CHANGELOG.md"] and "--axis-fingerprint-dir" in texts["CHANGELOG.md"]
    assert "mass_axis_frame.csv" in texts["OUTPUTS.md"] and "axis_fingerprints" in texts["OUTPUTS.md"]
    assert "srcs" in texts["PUBLISH.md"] and "axis_fingerprint" in texts["DATA_IO.md"]
    # the CLI's own help names both flags too, on both pipelines
    import argparse
    from peaky.cli import build_parser
    sub = [a for a in build_parser()._actions if isinstance(a, argparse._SubParsersAction)][0]
    assert "--axis-fingerprints" in sub.choices["batch"].format_help()
    assert "--axis-fingerprint-dir" in sub.choices["pool"].format_help()
