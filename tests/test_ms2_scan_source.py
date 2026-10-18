"""Offline tests for the per-scan MS2 source wired into peaky: the io_mascope
fetchers (a raw file's `/metadata`, the file listing, a spectrum's time
series), `sources.extract_sample(per_scan=True)`, `sources.extract_file`, the
`peaky ms2 extract --per-scan | --file` and `peaky ms2 files` commands, and
`ms2 analyze` on a `--file` folder. No network: the client is a fake.

Own data (tests/fixtures/ms2, ids and names scrubbed): the 2026-09-11
monoterpene standard A, an ordinary MS2 file the server reads --

  * ms2_monoterpene_a_metadata.json -- its `/metadata`: all 140 scans, the 61
    trailer rows the server writes (its MS1 scans; trimmed to the 17 keys of
    the synthetic rows of tests/test_ms2_scans.py -- the 14 the source reads,
    plus MsType, StartTime and Monoisotopic M/Z:, which it does not), every
    MS2 centroid, and the MS1 centroids within 0.5 Da of a parent (the window
    inventory needs +-0.205);
  * ms2_monoterpene_a_times.json -- the `time` list of each spectrum's time
    series (get_timeseries per parent and activation);
  * ms2_monoterpene_a_centroids_sn3.json -- the server's by_activation spectra
    at S/N 3 (12 spectra, 230 peaks).

Semi-synthetic: `wrap_tms2` re-writes standard A as a targeted-MS2 file the
server could not read -- every MS2 scan given a tMS2 trailer (an event per
spectrum, its NCE, the scan-range rule's HighMass) under a garbled filter,
22 of those rows dropped inside their blocks, the server's headerless first
scan kept, the Full Scan block first -- with the real centroids. Synthetic:
`make_tms2` of tests/test_ms2_scans.py."""
import json
import math
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from peaky import cli
from peaky.io import io_mascope as IO
from peaky.ms2 import analyze as AN
from peaky.ms2 import scans as SC
from peaky.ms2 import sources as S
from tests.test_ms2_scans import PREC, make_tms2, the_list

FIX = Path(__file__).resolve().parent / "fixtures" / "ms2"
MTA, MTB = "SAMPLE2xxxxxxxxx", "SAMPLE3xxxxxxxxx"
FA, FROUTINE, FT, FFIXED, FNEG = ("FILE2xxxxxxxxxxx", "FILE1xxxxxxxxxxx", "FILE3xxxxxxxxxxx",
                                  "FILE4xxxxxxxxxxx", "FILE5xxxxxxxxxxx")
SENTINEL_URL = "https://sentinel-host.invalid"
SENTINEL_TOKEN = "TOKSENTINEL0123456789"
#: the windows' ions of standard A (as PR #80's known answers name them)
MT_IONS = ("C10H17+", "C10H20^N+", "C10H19O+")
GARBLED = ("ITMS - c EI Full ms", "TQMS - c EI Full ms", "SQMS + c FABI Full ms",
           "Sector - c Full ms")
WIDTH_A = 0.4


def _load(name):
    return json.loads((FIX / name).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def own():
    """Standard A: (metadata, times, the server's S/N-3 spectra, its summary)."""
    return (_load("ms2_monoterpene_a_metadata.json"), _load("ms2_monoterpene_a_times.json"),
            _load("ms2_monoterpene_a_centroids_sn3.json"), _load("ms2_monoterpene_a_summary.json"))


def _ions(*formulas):
    return AN.collect_ions((f, math.nan, AN.ion_mz(f)) for f in formulas)


def _e50(res, parent, ion, curve="F"):
    e = res.e50
    r = e[np.isclose(e["parent_mz"], parent) & (e["ion"] == ion) & (e["curve"] == curve)]
    assert len(r) == 1
    return r.iloc[0]


def _tables(spectra, parents, window, width=WIDTH_A):
    return AN.RunTables(spectra=spectra, parents=parents, window=window,
                        manifest={"sample_id": "S", "parameters": {"isolation_width_used": width}})


def wrap_tms2(meta, times, summary, *, width=WIDTH_A, drop_inside=True, keep_first=True):
    """Standard A re-written as a targeted-MS2 file (module docstring) ->
    (payload, one single-row inclusion list per experiment in event order,
    spectrum keys in event order, the dropped MS2 rows' scan numbers)."""
    st, _ = SC.assign_from_times(SC.scan_table(meta), times)
    ms2 = st[st["spectrum"] != ""].sort_values("scan")
    keys = list(dict.fromkeys(ms2["spectrum"]))            # acquisition order = event order
    volts = {(round(g["parent_peak_mz"], 6), g["activation"]): g["hcd_energy"][0]
             for g in summary["groups"]}
    fs_event = len(keys) + 1
    ms1 = st.loc[st["order"] == 1, "scan"].tolist()
    order = ([1] if keep_first else []) + ms1 + ms2["scan"].tolist()
    drops = set()
    for k in keys:
        s = ms2.loc[ms2["spectrum"] == k, "scan"].tolist()
        if drop_inside and len(s) >= 5:
            drops |= {s[1], s[2]}
    rows, data, tt, dropped = {}, [], [], []
    ms2_of = dict(zip(ms2["scan"], ms2["spectrum"]))
    for new, old in enumerate(order, start=1):
        data.append(meta["centroids_meta"]["data"][old - 1])
        tt.append(3.0 + 2.6 * (new - 1))
        if old in ms1:
            r = dict(meta["stats_per_scan"][str(old)])
            r["Scan Event:"] = fs_event
        elif old in ms2_of:
            k = ms2_of[old]
            p, act = S._parent_of(k, {})
            nce = S.parse_activation(act)[1]
            high = 1.02 * (p + width / 2) + 20
            r = {"ScanType": f"{GARBLED[new % 4]} [40.0000-{high:.4f}]", "MsType": "Ms",
                 "LowMass": 40.0, "HighMass": high, "Scan Node:": "tMS2",
                 "Scan Event:": keys.index(k) + 1, "HCD Energy:": f"{nce:.2f}",
                 "HCD Energy V:": f"{volts[(round(p, 6), act)]:.2f}",
                 "MS2 Isolation Width:": width, "FT Resolution:": 60000}
            if old in drops:
                r = None
                dropped.append(new)
        else:
            r = None                              # the server's headerless first scan
        if r is not None:
            rows[str(new)] = r
    lists = []
    for k in keys:
        f = pd.DataFrame({"Compound": [k], "Precursor (m/z)": [S._parent_of(k, {})[0]]})
        f.attrs["name"] = f"list_{len(lists) + 1}.csv"
        lists.append(f)
    payload = {"num_of_scans": len(order), "stats_per_scan": rows, "stats_per_file": {"Value": {}},
               "centroids_meta": {"time": tt, "data": data}}
    return payload, lists, keys, dropped


# --------------------------------------------------------- own data: API ----
def test_standard_a_payload_is_an_ordinary_ms2_file(own):
    """The server writes a trailer row for its MS1 scans only: 61 of 140, all
    readable 'FTMS + p NSI Full ms' filters, polarity '+'. Its 78 MS2 scans
    and the bad first scan have centroids and no row, so nothing reads MS2 off
    a trailer -- the file check sends it to the sample's per-scan mode."""
    meta = own[0]
    st = SC.scan_table(meta)
    assert len(st) == 140 and int(st["has_header"].sum()) == 61
    assert (st.loc[st["has_header"], "order"] == 1).all() and not (st["order"] == 2).any()
    v = SC.filter_verdict(st)
    assert v["filters_readable"] == SC.FILTERS_YES and v["filters_ok"] == 61
    assert SC.filter_polarity(st) == "+"
    chk = S.check_file(meta, {"sample_file_id": FA, "polarity": "+"})
    assert chk["ms2_by_trailer"] == 0 and chk["headerless"] == 79 and chk["command"] == ""
    assert chk["check_note"] == S.MS2_WITHOUT_TRAILER_NOTE.format(n=79)
    res = S.extract_file(None, FA, metadata=meta)
    assert res.row["status"] == S.MS2_WITHOUT_TRAILER and not len(res.spectra)
    assert res.row["polarity"] == "+" and res.params["polarity_source"] == "scan filters"


def test_standard_a_time_series_names_every_ms2_scan(own):
    """API mode: each spectrum's time-series times name its scans -- 78 of 78,
    each group's count the server's, none an MS1 row, none twice; the one
    scan with no row that no group holds is the first (the server's
    bad-first-scan rule)."""
    meta, times, _, summ = own
    st, rep = SC.assign_from_times(SC.scan_table(meta), times)
    assert int((st["spectrum"] != "").sum()) == 78
    assert rep["unmatched_times"] == 0 and rep["claimed_twice"] == [] and rep["on_ms1_rows"] == []
    counts = {S.spectrum_key(g["parent_peak_mz"], g["activation"]): g["scan_count"]
              for g in summ["groups"]}
    assert {k: v["matched"] for k, v in rep["spectra"].items()} == counts
    assert (rep["headerless_in_no_group"], rep["headerless_in_no_group_scans"]) == (1, "1")
    mapped = st[st["spectrum"] != ""]
    assert (mapped["precursor_source"] == SC.PRECURSOR_SERVER).all()
    assert not mapped["has_header"].any()


def test_per_scan_averages_reproduce_the_servers_spectra(own):
    """Our average of each group's scans against the server's S/N-3 spectrum:
    every one of its 230 peaks, nothing extra, the same count per spectrum;
    m/z within 0.005 ppm; resolution identical; S/N 1.0015-1.011 x; intensity
    1.020 x on the median peak (0.934-1.118; 5-95 %: 1.000-1.074) -- the
    server re-reads heights off an averaged profile the payload lacks
    (`scans.SERVER_HEIGHT_RATIO`)."""
    meta, times, cen, _ = own
    st, _ = SC.assign_from_times(SC.scan_table(meta), times)
    ours, _, _, rep = SC.build_spectra("S", meta, st)
    srv = S.spectra_table("S", cen, None)
    assert rep["peaks"] == len(ours) == len(srv) == 230 and rep["spectra"] == 12
    assert (ours.groupby("spectrum").size() == srv.groupby("spectrum").size()).all()
    m = []
    for k, g in srv.groupby("spectrum"):
        o = ours[ours["spectrum"] == k]
        for r in g.itertuples():
            d = np.abs(o["mz"].to_numpy() - r.mz) / r.mz * 1e6
            i = int(np.argmin(d))
            m.append((k, i, d[i], o["intensity"].iloc[i] / r.intensity,
                      o["signal_to_noise"].iloc[i] / r.signal_to_noise,
                      o["resolution"].iloc[i] / r.resolution))
    m = pd.DataFrame(m, columns=["k", "i", "ppm", "ri", "rsn", "rr"])
    assert not m.duplicated(["k", "i"]).any()               # one of ours per server peak
    assert m["ppm"].max() < 0.005
    assert np.allclose(m["rr"], 1.0, rtol=1e-6, atol=0)
    assert m["rsn"].min() > 1.001 and m["rsn"].max() < 1.012
    assert np.median(m["rsn"]) == pytest.approx(1.0055, abs=5e-4)
    q = np.quantile(m["ri"], [0, 0.05, 0.5, 0.95, 1])
    assert q[2] == pytest.approx(SC.SERVER_HEIGHT_RATIO, abs=5e-4)
    assert q[1] == pytest.approx(1.000, abs=2e-3) and q[3] == pytest.approx(1.0736, abs=2e-3)
    assert q[0] == pytest.approx(0.934, abs=1e-3) and q[4] == pytest.approx(1.1184, abs=1e-3)


def test_the_servers_peaks_read_in_every_scan(own):
    """Each server peak read in every scan of its group: 1517 rows (its peaks
    x its scans), no peak all zero, and the per-scan mean 1.020 x the
    server's intensity on the median peak -- the same ratio as our averages."""
    meta, times, cen, _ = own
    st, _ = SC.assign_from_times(SC.scan_table(meta), times)
    srv = S.spectra_table("S", cen, None)
    pk = SC.scan_peak_intensities(meta, st, srv)
    assert list(pk.columns) == list(SC.SCAN_PEAK_COLUMNS) and len(pk) == 1517
    n_scans = st[st["spectrum"] != ""].groupby("spectrum").size()
    assert len(pk) == int((srv.groupby("spectrum").size() * n_scans).sum())
    mean = pk.groupby(["spectrum", "peak_mz"])["intensity"].mean()
    ratio = mean / srv.set_index(["spectrum", "mz"])["intensity"].reindex(mean.index)
    assert (mean > 0).all()
    assert np.median(ratio) == pytest.approx(SC.SERVER_HEIGHT_RATIO, abs=5e-4)


# ------------------------------------------------- own data, wrapped tMS2 ----
@pytest.fixture(scope="module")
def wrapped(own):
    meta, times, _, summ = own
    return wrap_tms2(meta, times, summ)


def test_a_garbled_tms2_file_needs_its_polarity(wrapped):
    payload, lists, _, _ = wrapped
    with pytest.raises(S.PolarityUnknown, match=r"not readable \(61 of 117 agree"):
        S.extract_file(None, FT, metadata=payload, inclusion_lists=lists)
    with pytest.raises(ValueError, match="contradicts the file record"):
        S.extract_file(None, FT, metadata=payload, inclusion_lists=lists, polarity="-",
                       file_row={"polarity": "+"})
    res = S.extract_file(None, FT, metadata=payload, inclusion_lists=lists,
                         file_row={"polarity": "+-", "filename": "a tMS2 file"}, polarity="+")
    assert res.row["polarity"] == "+" and res.params["polarity_source"] == "explicit"
    assert res.row["name"] == "a tMS2 file"


def test_wrapped_standard_a_rebuilds_the_same_spectra(own, wrapped):
    """Garbled filters, 23 scans with no trailer row (the first scan and 22
    MS2 rows inside their blocks) and one list per experiment: every scan is
    placed (the 22 by the scan cycle, the first walked back into the Full
    Scan block), every precursor comes from its list, the list fits the
    scan-range rule to 0 mDa -- and the 12 spectra are exactly our averages of
    the scans the server's time series name."""
    meta, times, _, summ = own
    payload, lists, keys, dropped = wrapped
    assert len(dropped) == 22
    res = S.extract_file(None, FT, metadata=payload, inclusion_lists=lists, polarity="+")
    m = res.per_scan["mapping"]
    assert m["filters"]["filters_readable"] == SC.FILTERS_NO and m["filters"]["ms2_filters_ok"] == 0
    assert m["layout"] == SC.PRECURSOR_LIST and len(m["experiments"]) == 12
    assert m["full_scan_event"] == 13
    inf = m["inference"]
    assert (inf["headerless"], inf["inferred"], inf["unassigned"]) == (23, 23, 0)
    assert m["precursor_sources"] == {SC.PRECURSOR_LIST: 78}
    assert m["rule"]["applies"] and m["rule"]["checked"] and m["rule"]["max_residual_mda"] < 1e-6
    sc = res.scans
    assert sc.loc[sc["scan"].isin(dropped), "inferred"].all()
    assert sc.loc[sc["scan"] == 1, "order"].iloc[0] == 1             # walked back: a Full Scan
    # the same spectra as the time-series mapping gives, to the last bit
    st, _ = SC.assign_from_times(SC.scan_table(meta), times)
    api, api_par, _, _ = SC.build_spectra("S", meta, st)
    cols = ["spectrum", "mz", "intensity", "signal_to_noise", "resolution", "nce"]
    a = api[cols].sort_values(["spectrum", "mz"]).reset_index(drop=True)
    b = res.spectra[cols].sort_values(["spectrum", "mz"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)
    counts = {S.spectrum_key(g["parent_peak_mz"], g["activation"]): g["scan_count"]
              for g in summ["groups"]}
    assert dict(zip(res.parents["spectrum"], res.parents["scan_count"])) == counts
    # volts: the trailer's, one per group; the window at the trailer's width
    assert set(res.parents["hcd_v"].round(2)) == {g["hcd_energy"][0] for g in summ["groups"]}
    assert res.params["isolation_width_used"] == WIDTH_A
    assert res.params["isolation_width_source"] == "trailer"
    assert set(res.window["parent_mz"]) == {137.1321, 155.1429, 155.1559}
    assert res.window["offset_da"].abs().max() <= WIDTH_A / 2 + S.WINDOW_PAD_DA
    assert res.row["status"] == "ok" and res.source == S.SOURCE_FILE
    assert res.scan_peaks["spectrum"].nunique() == 12


def test_wrapped_standard_a_gives_the_servers_e50(own, wrapped):
    """`ms2 analyze` on the rebuilt spectra against the server's S/N-3
    spectra: [M+H]+ C10H17+ F within 0.002 at every NCE (0.0016 at most, at
    NCE 20), which at the crossing's slope (0.018 per NCE) bounds the E50s'
    difference by 0.09 NCE: 13.821 against 13.883 (the S/N-10 spectra of
    PR #80 read 13.891). Our intensities run 1.02 x the server's on the
    median peak, but F is a ratio of peaks of one spectrum."""
    meta, times, cen, summ = own
    srv = AN.analyze_run(_tables(S.spectra_table("S", cen, summ), S.parents_table("S", cen, summ),
                                 S.window_table("S", _load("ms2_monoterpene_a_ms1.json"),
                                                [137.1321, 155.1429, 155.1559], WIDTH_A)),
                         _ions(*MT_IONS))
    payload, lists, _, _ = wrapped
    res = S.extract_file(None, FT, metadata=payload, inclusion_lists=lists, polarity="+")
    ours = AN.analyze_run(_tables(res.spectra, res.parents, res.window), _ions(*MT_IONS))
    a, b = _e50(srv, 137.1321, "C10H17+"), _e50(ours, 137.1321, "C10H17+")
    assert a["flag"] == b["flag"] == "interpolated"
    assert a["e50_nce"] == pytest.approx(13.883, abs=1e-3)
    assert b["e50_nce"] == pytest.approx(13.821, abs=1e-3)
    assert abs(a["e50_nce"] - b["e50_nce"]) < 0.1
    assert b["e50_v"] == pytest.approx(a["e50_v"], abs=0.02)

    def curve(r):
        bd = r.breakdown
        g = bd[np.isclose(bd["parent_mz"], 137.1321) & (bd["ion"] == "C10H17+")]
        return g.sort_values("nce")["f"].to_numpy()
    assert np.abs(curve(srv) - curve(ours)).max() < 0.002


def test_a_list_that_does_not_fit_the_real_scans_is_refused(wrapped):
    payload, lists, _, _ = wrapped
    swapped = [lists[0], lists[5], *lists[2:5], lists[1], *lists[6:]]
    with pytest.raises(SC.ScanMapError, match="do not fit the inclusion list"):
        S.extract_file(None, FT, metadata=payload, inclusion_lists=swapped, polarity="+")
    with pytest.raises(SC.ScanMapError, match="the Full Scan is event 13"):
        S.extract_file(None, FT, metadata=payload, inclusion_lists=lists + lists[:1],
                       polarity="+")
    # without a list the scan range names each precursor to 0.1 mDa of its key;
    # nothing pins the layout, so no headerless scan is walked onto an event
    # (one-row experiments: every run wraps) -- the 56 MS2 scans with a row
    res = S.extract_file(None, FT, metadata=payload, polarity="+")
    assert res.per_scan["mapping"]["precursor_sources"] == {SC.PRECURSOR_RULE: 56}
    inf = res.per_scan["mapping"]["inference"]
    assert (inf["inferred"], inf["unassigned"]) == (0, 23)
    assert sorted(res.parents["parent_mz"].unique()) == pytest.approx(
        [137.1321, 155.1429, 155.1559], abs=1e-4)


# ------------------------------------------------------------- the server ----
class _Ms2:
    def __init__(self, server, sid):
        self.server, self.sid = server, sid

    def get_summary(self, **kw):
        return self.server.answer("summary", self.sid, kw)

    def get_averaged_centroids(self, **kw):
        return self.server.answer("centroids", self.sid, kw)

    def get_ms1_centroids(self, **kw):
        return self.server.answer("ms1", self.sid, kw)

    def get_timeseries(self, parent_peak_mz, **kw):
        return self.server.answer("timeseries", self.sid, dict(kw, parent=parent_peak_mz))


class _Samples:
    def __init__(self, server):
        self.server = server

    def ms2(self, sid):
        return _Ms2(self.server, sid)

    def get(self, sid):
        return self.server.answer("record", sid, {})


class FakeServer:
    """samples.get / samples.ms2(sid).* and, through io_mascope._api_get, a
    raw file's /metadata and the file listing. Every call is recorded."""
    url = SENTINEL_URL
    access_token = SENTINEL_TOKEN

    def __init__(self, store, files, listing=()):
        self.store, self.files, self.listing = store, files, list(listing)
        self.listing_total = None                 # the listing's `results`; None: its length
        self.calls = []
        self.samples = _Samples(self)

    def answer(self, what, sid, kw):
        self.calls.append((what, sid, kw))
        x = self.store[sid][what]
        if what == "timeseries":
            x = x[S.spectrum_key(kw["parent"], kw["activation"])]
        if isinstance(x, Exception):
            raise x
        return x

    def api_get(self, client, path, *, params=None, timeout=None, body=False):
        assert client is self
        if path == "sample/files":
            self.calls.append(("files", params.get("instrument"), dict(params)))
            rows = [dict(r) for r in self.listing]
            if not body:
                return rows
            return {"results": (len(rows) if self.listing_total is None
                                else self.listing_total), "data": rows}
        head, fid, tail = path.rsplit("/", 2)
        assert (head, tail) == ("sample/files", "metadata") and timeout == IO.METADATA_TIMEOUT
        self.calls.append(("metadata", fid, {}))
        x = self.files[fid]
        if isinstance(x, Exception):
            raise x
        return json.loads(json.dumps(x))


def _sample_store(own):
    meta, times, cen, summ = own
    rec = dict(_load("ms2_monoterpene_a_sample.json"), sample_file_id=FA)
    ts = {k: {"time": v["time"], "mz_values": [], "values": []} for k, v in times.items()}
    return {MTA: {"record": rec, "summary": summ, "centroids": cen,
                  "ms1": _load("ms2_monoterpene_a_ms1.json"), "timeseries": ts}}


def _server_error(status, text, fid):
    from mascope_sdk.exceptions import AuthenticationError, ServerError
    cls = ServerError if status >= 500 else AuthenticationError
    return cls(text, status_code=status,
               url=f"https://server.example/api/sample/files/{fid}/metadata")


def _routine_rec(fid, **kw):
    rec = {"sample_file_id": fid, "filename": f"file {fid[:5]}",
           "datetime_utc": "2026-09-11T10:00:00+00:00", "length": 300.0, "polarity": "+",
           "instrument_type": "orbi", "processing_status": "done",
           "processing_detail": "Matched 1 sample."}
    rec.update(kw)
    return rec


@pytest.fixture
def server(monkeypatch, own):
    """The CLI against the fake server: credentials stubbed, sentinel ones in
    the environment that nothing written may carry, sleeps recorded."""
    tms2, _ = make_tms2()
    fixed, _ = make_tms2(fixed_range=True)
    srv = FakeServer(_sample_store(own), {FA: own[0], FT: tms2, FFIXED: fixed},
                     listing=[_routine_rec(FROUTINE),
                              _routine_rec(FNEG, polarity="-"),
                              _routine_rec(FA, processing_status=None),
                              _routine_rec(FT, polarity="+-", processing_status="needs_chemistry",
                                           processing_detail="no sample built"),
                              _routine_rec(FFIXED, polarity="+-",
                                           processing_status="needs_chemistry")])
    monkeypatch.setenv("MASCOPE_URL", SENTINEL_URL)
    monkeypatch.setenv("MASCOPE_ACCESS_TOKEN", SENTINEL_TOKEN)
    monkeypatch.setattr(cli, "_require_creds", lambda *a, **k: None)
    monkeypatch.setattr(IO, "connect", lambda *a, **k: srv)
    monkeypatch.setattr(IO, "_api_get", srv.api_get)
    srv.sleeps = []
    monkeypatch.setattr(S, "time", type("_T", (), {"sleep": staticmethod(srv.sleeps.append),
                                                   "time": staticmethod(time.time)}))
    return srv


# --------------------------------------------------------------- fetchers ----
def test_the_raw_routes_go_through_the_sdk_transport(monkeypatch):
    """`_api_get` asks the client's server with its token, through the SDK's
    http_get, and unwraps `data`; the metadata waits 300 s; the listing asks
    by instrument and time range."""
    import mascope_sdk._http as H
    seen = []

    class _R:
        def __init__(self, body):
            self.body = body

        def json(self):
            return self.body

    def fake_get(url, path, token, **kw):
        seen.append((url, path, token, kw))
        return _R({"data": [{"sample_file_id": FA}]} if path == "sample/files"
                  else {"data": {"num_of_scans": 1}})
    monkeypatch.setattr(H, "http_get", fake_get)
    c = type("C", (), {"url": SENTINEL_URL, "access_token": SENTINEL_TOKEN, "_verify_ssl": True,
                       "_service_name": "peaky-test"})()
    assert IO.fetch_file_metadata(c, FA) == {"num_of_scans": 1}
    assert IO.fetch_sample_files(c, instrument="ORB", since="2026-09-11T10:00:00",
                                 until="2026-09-11T16:00:00") == [{"sample_file_id": FA}]
    (u1, p1, t1, k1), (u2, p2, t2, k2) = seen
    assert (u1, p1, t1) == (SENTINEL_URL, f"sample/files/{FA}/metadata", SENTINEL_TOKEN)
    assert k1["timeout"] == IO.METADATA_TIMEOUT == (15, 300)
    assert k1["verify_ssl"] is True and k1["service_name"] == "peaky-test"
    assert p2 == "sample/files" and k2["params"] == {
        "instrument": "ORB", "datetime_min": "2026-09-11T10:00:00",
        "datetime_max": "2026-09-11T16:00:00", "sort": "sample_file_id", "order": "asc",
        "page": 0, "limit": IO.FILE_LIST_PAGE}
    IO.fetch_sample_files(c, instrument="ORB", since="2026-09-11T10:00:00")
    assert "datetime_max" not in seen[-1][3]["params"]


def test_the_raw_routes_retry_as_the_ms2_calls_do(monkeypatch, server):
    """A proxy's WAF block is retried with back-off; an unknown file's 404 or
    a 500 on /metadata is final (asked once); the listing retries a 500."""
    sleeps = []
    monkeypatch.setattr(IO, "time", type("_T", (), {"sleep": staticmethod(sleeps.append)}))
    waf = _server_error(403, "Attention Required! | Cloudflare", FT)
    answers = [waf, waf, {"num_of_scans": 1}]

    def flaky(client, path, **kw):
        x = answers.pop(0)
        if isinstance(x, Exception):
            raise x
        return x
    monkeypatch.setattr(IO, "_api_get", flaky)
    assert IO.fetch_file_metadata(server, FT) == {"num_of_scans": 1} and sleeps == [2.0, 4.0]
    asked = []

    def final(client, path, **kw):
        asked.append(path)
        raise _server_error(500, "Unexpected error (ref: 403f00aa)", FT)
    monkeypatch.setattr(IO, "_api_get", final)
    with pytest.raises(Exception, match="Unexpected error"):
        IO.fetch_file_metadata(server, FT)
    assert len(asked) == 1
    asked.clear()
    with pytest.raises(Exception, match="Unexpected error"):
        IO.fetch_sample_files(server, instrument="ORB", since="2026-09-11T00:00:00")
    assert len(asked) == 4


def test_the_time_series_is_asked_per_spectrum_by_its_key(server):
    out = IO.fetch_ms2_timeseries(server, MTA, 137.1321, activation="hcd1.00",
                                  noise_threshold=3.0)
    assert len(out["time"]) == 7
    assert server.calls == [("timeseries", MTA, {"activation": "hcd1.00", "noise_threshold": 3.0,
                                                 "parent": 137.1321})]


# ------------------------------------------------- extract_sample(per_scan) ----
def test_extract_sample_per_scan_reads_the_scans_behind_the_servers_spectra(server):
    res = S.extract_sample(server, MTA, per_scan=True)
    asked = [c[0] for c in server.calls]
    assert asked == ["record", "summary", "centroids", "ms1", "metadata"] + ["timeseries"] * 12
    assert server.calls[4][1] == FA
    assert {c[2]["noise_threshold"] for c in server.calls if c[0] == "timeseries"} == {3.0}
    assert server.sleeps == [S.TIMESERIES_PAUSE_S] * 11
    # the spectra stay the server's
    want = S.spectra_table(MTA, server.store[MTA]["centroids"], server.store[MTA]["summary"])
    pd.testing.assert_frame_equal(res.spectra, want)
    assert res.source == S.SOURCE_SERVER_PER_SCAN and res.params["per_scan"] is True
    assert list(res.scans.columns) == list(SC.SCAN_COLUMNS) and len(res.scans) == 140
    assert int((res.scans["spectrum"] != "").sum()) == 78 and len(res.scan_peaks) == 1517
    ps = res.per_scan
    assert ps["sample_file_id"] == FA and ps["ms2_scans_mapped"] == 78
    assert (ps["headerless_in_no_group"], ps["headerless_in_no_group_scans"]) == (1, "1")
    assert ps["hidden_tail_scans"] == 0 and ps["unmatched_times"] == 0
    assert ps["filters"]["filters_readable"] == SC.FILTERS_YES
    assert any("1 scan(s) with no trailer row are in no server group" in n for n in res.notes)
    man = S.manifest(res)
    assert man["source"] == S.SOURCE_SERVER_PER_SCAN and man["per_scan"]["scans"] == 140
    assert man["counts"]["scans"] == 140 and man["counts"]["scan_peak_rows"] == 1517


def test_a_hidden_tail_is_counted_after_t1(server):
    """The server's time series stops at its ingest: the last group's scans
    become scans in no group, and those after the record's t1 are said to be
    the hidden tail's; a group whose time series names another number of scans
    than the server counts is noted."""
    st = server.store[MTA]
    t_last = st["timeseries"]["155.1429@hcd15.00"]["time"]
    st["timeseries"] = dict(st["timeseries"], **{"155.1429@hcd15.00": {"time": t_last[:1]}})
    st["record"] = dict(st["record"], t1=350.0)
    res = S.extract_sample(server, MTA, per_scan=True)
    ps = res.per_scan
    assert ps["headerless_in_no_group"] == 5 and ps["hidden_tail_scans"] == 4
    assert any("4 of the 5 scan(s) in no group lie after the last ingested scan (t1 = 350.0 s"
               in n for n in res.notes)
    assert any("155.1429@hcd15.00 (1 named, 5 counted)" in n for n in res.notes)


def test_a_per_scan_failure_keeps_the_ms2_tables(server):
    server.files[FA] = _server_error(500, "Unexpected error (ref: 0f1e2d3c)", FA)
    res = S.extract_sample(server, MTA, per_scan=True)
    assert len(res.spectra) == 230 and len(res.scans) == 0 and len(res.scan_peaks) == 0
    assert list(res.scans.columns) == list(SC.SCAN_COLUMNS)
    assert any(n.startswith("per-scan tables skipped: the server could not serve") and
               "server.example" not in n for n in res.notes)
    assert "timeseries" not in [c[0] for c in server.calls]
    # the server's or the connection's failure is not this sample's: raised
    server.files[FA] = _server_error(403, "Attention Required! | Cloudflare", FA)
    with pytest.raises(Exception, match="Attention Required"):
        S.extract_sample(server, MTA, per_scan=True)


def test_a_record_without_its_raw_file_skips_the_scans(server):
    st = server.store[MTA]
    st["record"] = {k: v for k, v in st["record"].items() if k != "sample_file_id"}
    res = S.extract_sample(server, MTA, per_scan=True)
    assert len(res.spectra) == 230 and len(res.scans) == 0
    assert res.notes[-1] == ("per-scan tables skipped: the sample record names no raw file "
                             "(sample_file_id)")
    assert not {"metadata", "timeseries"} & {c[0] for c in server.calls}


def test_no_spectrum_no_per_scan_request(server):
    st = server.store[MTA]
    st["summary"] = _load("ms2_empty_scaffold_summary.json")
    res = S.extract_sample(server, MTA, per_scan=True)
    assert res.row["status"] == "no MS2" and len(res.scans) == 0
    assert res.notes[-1] == "per-scan tables skipped: no MS2 spectrum to map scans to"
    assert "metadata" not in [c[0] for c in server.calls]


# -------------------------------------------------------------------- CLI ----
PER_SCAN_FILES = {"ms2_spectra.parquet", "ms2_spectra.csv", "ms2_parents.csv", "ms2_window.csv",
                  "ms2_manifest.json", "ms2_scans.csv", "ms2_scan_peaks.parquet"}


def test_cli_extract_per_scan_writes_the_scan_tables(server, tmp_path, capsys):
    assert cli.main(["ms2", "extract", "--sample-id", MTA, "--per-scan", "--save-metadata",
                     "--out-dir", str(tmp_path)]) == 0
    d = tmp_path / "ms2" / MTA
    assert {p.name for p in d.iterdir()} == PER_SCAN_FILES | {cli.MS2_METADATA_FILE}
    scans = pd.read_csv(d / "ms2_scans.csv", keep_default_na=False, na_values=[""])
    assert list(scans.columns) == list(SC.SCAN_COLUMNS) and len(scans) == 140
    pk = pd.read_parquet(d / "ms2_scan_peaks.parquet")
    assert list(pk.columns) == list(SC.SCAN_PEAK_COLUMNS) and len(pk) == 1517
    assert {c: str(t) for c, t in pk.dtypes.items() if c != "spectrum"} == {
        c: t for c, t in SC.SCAN_PEAK_COLUMNS.items() if c != "spectrum"}
    man = json.loads((d / "ms2_manifest.json").read_text())
    from peaky.reporting import provenance as PV
    for name in (PER_SCAN_FILES - {"ms2_manifest.json"}) | {cli.MS2_METADATA_FILE}:
        assert man["files"][name] == PV.sha1_file(str(d / name)), name
    kept = json.loads((d / cli.MS2_METADATA_FILE).read_text())
    assert kept["num_of_scans"] == 140
    printed = capsys.readouterr().out
    assert ("per-scan: 140 scans (61 with a trailer row), 78 MS2 scans named by the spectra's "
            "time series, 1517 scan-peak rows; scans with no trailer row in no group: 1 (1)"
            in printed)
    # a re-run without --per-scan leaves no stale scan tables
    assert cli.main(["ms2", "extract", "--sample-id", MTA, "--out-dir", str(tmp_path)]) == 0
    assert not (d / "ms2_scans.csv").exists() and not (d / "ms2_scan_peaks.parquet").exists()
    assert "per_scan" not in json.loads((d / "ms2_manifest.json").read_text())


def _write_lists(tmp_path, lists):
    paths = []
    for f in lists:
        p = tmp_path / f.attrs["name"]
        f.to_csv(p, index=False)
        paths.append(str(p))
    return paths


def test_cli_extract_file_offline_then_analyze(own, wrapped, tmp_path, capsys, monkeypatch):
    """`--file` with `--metadata-json`: no server, no credentials asked; the
    folder holds the four tables, the scan tables and the manifest, and
    `ms2 analyze` reads it as it is: C10H17+ E50 13.82."""
    monkeypatch.setattr(cli, "_require_creds",
                        lambda *a, **k: pytest.fail("no credentials needed offline"))
    monkeypatch.setattr(IO, "connect", lambda *a, **k: pytest.fail("no server offline"))
    payload, lists, _, _ = wrapped
    mj = tmp_path / "payload.json"
    mj.write_text(json.dumps({"data": payload}))          # the API's envelope is read too
    paths = _write_lists(tmp_path, lists)
    base = tmp_path / "out"
    assert cli.main(["ms2", "extract", "--file", FT, "--metadata-json", str(mj),
                     "--inclusion-list", *paths, "--polarity", "+", "--out-dir", str(base)]) == 0
    d = base / "ms2" / FT
    assert {p.name for p in d.iterdir()} == PER_SCAN_FILES
    man = json.loads((d / "ms2_manifest.json").read_text())
    assert man["source"] == S.SOURCE_FILE and man["server"] is None and man["polarity"] == "+"
    assert man["parameters"]["isolation_width_used"] == WIDTH_A
    from peaky.reporting import provenance as PV
    assert man["parameters"]["inclusion_lists"][0] == {
        "name": "list_1.csv", "rows": 1, "sha1": PV.sha1_file(paths[0]),
        "bytes": Path(paths[0]).stat().st_size}
    assert man["per_scan"]["mapping"]["inference"]["inferred"] == 23
    assert man["per_scan"]["tolerances"]["rule"]["tol_mda"] == SC.RULE_TOL_MDA
    assert man["counts"]["spectra"] == 12 and man["counts"]["scans"] == 140
    printed = capsys.readouterr().out
    assert (f"[ms2] {FT}: ok; 12 spectra over 3 parents, 230 peaks (S/N >= 3, averaged here from "
            "78 MS2 scans); polarity + (explicit)") in printed
    assert ("filters readable: no (61 of 117 agree with their trailers); precursors: inclusion "
            "list 78; scan range follows the precursor (max residual 0 mDa over 12 events); "
            "23 scans with no trailer row: 23 placed by the scan cycle, 0 unassigned") in printed
    out = tmp_path / "an"
    assert cli.main(["ms2", "analyze", "--run-dir", str(d), "--ion", "137.1323=C10H17+",
                     "--ion", "155.1558=C10H20^N+", "--ion", "155.1428=C10H19O+",
                     "--no-figures", "--out-dir", str(out)]) == 0
    e = pd.read_csv(out / "ms2_e50.csv")
    r = e[(e["parent_mz"] == 137.1321) & (e["curve"] == "F") & (e["ion"] == "C10H17+")].iloc[0]
    assert r["e50_nce"] == pytest.approx(13.821, abs=1e-3)
    am = json.loads((out / "ms2_analyze_manifest.json").read_text())
    assert am["parameters"]["isolation_width_source"] == "extract manifest"
    assert am["sample_id"] == FT


def test_the_manifest_tells_two_lists_of_one_name_and_length_apart(tmp_path):
    """The list decides the precursor mapping: an edited list with the same
    name and row count (one row moved 0.3 mDa, inside the rule's check) is
    recorded by its content (sha1, bytes), like the payload, so a run can be
    matched to the list it read."""
    from peaky.reporting import provenance as PV
    meta, _ = make_tms2()
    a, b = tmp_path / "a" / "list.csv", tmp_path / "b" / "list.csv"
    for path, shift in ((a, 0.0), (b, 0.0003)):
        path.parent.mkdir()
        lst = the_list()
        lst.loc[1, "Precursor (m/z)"] += shift
        lst.to_csv(path, index=False)
    got = [S.extract_file(None, FT, metadata=meta, inclusion_lists=[str(x)],
                          polarity="+").params["inclusion_lists"] for x in (a, b)]
    for g, x in zip(got, (a, b)):
        assert g == [{"name": "list.csv", "rows": len(PREC), "sha1": PV.sha1_file(str(x)),
                      "bytes": x.stat().st_size}]
    assert got[0][0]["sha1"] != got[1][0]["sha1"]
    # a frame handed in (the Python API) has no file: its content is hashed
    f = SC.read_inclusion_list(str(a))
    got = S.extract_file(None, FT, metadata=meta, inclusion_lists=[f],
                         polarity="+").params["inclusion_lists"][0]
    assert got["sha1"] and got["bytes"] is None


def test_cli_extract_file_from_the_server(server, tmp_path, capsys):
    """A synthetic garbled file through the fake server: its /metadata asked
    once, kept with --save-metadata; a missing polarity fails that file alone
    (its payload kept for a re-run); a wrong list fails it with the events."""
    lst = tmp_path / "list.csv"
    the_list().to_csv(lst, index=False)
    with pytest.raises(SystemExit) as exc:
        cli.main(["ms2", "extract", "--file", FT, FFIXED, "--save-metadata",
                  "--inclusion-list", str(lst), "--out-dir", str(tmp_path)])
    assert "2 of 2" in str(exc.value)
    printed = capsys.readouterr().out
    assert f"[ms2] {FT}: cannot extract: the polarity cannot be read off this file" in printed
    assert "re-run with --metadata-json" in printed
    for fid in (FT, FFIXED):
        assert {p.name for p in (tmp_path / "ms2" / fid).iterdir()} == {cli.MS2_METADATA_FILE}
    assert [c[:2] for c in server.calls] == [("metadata", FT), ("metadata", FFIXED)]
    # the kept payload, offline, with the polarity: written
    assert cli.main(["ms2", "extract", "--file", FT, "--metadata-json",
                     str(tmp_path / "ms2" / FT / cli.MS2_METADATA_FILE), "--polarity", "+",
                     "--inclusion-list", str(lst), "--out-dir", str(tmp_path)]) == 0
    d = tmp_path / "ms2" / FT
    sp = pd.read_parquet(d / "ms2_spectra.parquet")
    assert sp["spectrum"].nunique() == 15 and list(sp.columns) == list(S.SPECTRA_COLUMNS)
    # the list in the wrong order: refused, naming events
    rev = tmp_path / "rev.csv"
    the_list(PREC[::-1]).to_csv(rev, index=False)
    with pytest.raises(SystemExit):
        cli.main(["ms2", "extract", "--file", FT, "--polarity", "+", "--inclusion-list",
                  str(rev), "--out-dir", str(tmp_path / "x")])
    assert "do not fit the inclusion list" in capsys.readouterr().out
    assert not (tmp_path / "x" / "ms2" / FT).exists()


@pytest.mark.parametrize("argv, message", [
    (["--sample-id", MTA, "--inclusion-list", "a.csv"], "--inclusion-list goes with --file"),
    (["--sample-id", MTA, "--polarity", "+"], "--polarity goes with --file"),
    (["--sample-id", MTA, "--metadata-json", "m.json"], "--metadata-json goes with --file"),
    (["--sample-id", MTA, "--no-rule-check"], "--no-rule-check goes with --file"),
    (["--sample-id", MTA, "--save-metadata"], "--save-metadata goes with --file or --per-scan"),
    (["--file", FT, "--per-scan"], "--per-scan goes with --sample-id"),
    (["--file", FT, FA, "--metadata-json", "m.json"], "give one --file"),
    (["--file", FT, "--metadata-json", "m.json", "--save-metadata"], "--save-metadata keeps"),
    (["--file", "../x"], "--file '../x' is not a file id"),
    (["--file", FT, "--inclusion-list", "/no/such/list.csv"], "--inclusion-list /no/such"),
    (["--file", FT, "--noise-threshold", "-1"], "--noise-threshold must be >= 0"),
    (["--file", FT, FFIXED, "--noise-threshold", "inf", "--save-metadata"],
     "--noise-threshold must be >= 0"),
    (["--file", FT, FFIXED, "--iso-width", "inf", "--save-metadata"],
     "--iso-width must be a positive width"),
    (["--sample-id", MTA, "--noise-threshold", "inf"], "--noise-threshold must be >= 0"),
])
def test_cli_extract_checks_its_flags_before_asking(server, tmp_path, argv, message):
    with pytest.raises(SystemExit) as exc:
        cli.main(["ms2", "extract", *argv, "--out-dir", str(tmp_path)])
    assert message in str(exc.value)
    assert server.calls == [] and not (tmp_path / "ms2").exists()


def test_cli_parses_the_new_flags():
    P = cli.build_parser()
    a = P.parse_args(["ms2", "extract", "--file", FT, "--polarity", "-", "--inclusion-list",
                      "a.csv", "b.csv", "--inclusion-list", "c.csv"])
    assert a.file == [FT] and a.sample_id is None and a.polarity == "-"
    assert a.inclusion_list == ["a.csv", "b.csv", "c.csv"]
    for bad in (["ms2", "extract", "--file", FT, "--sample-id", MTA],
                ["ms2", "extract", "--file", FT, "--polarity", "x"],
                ["ms2", "files", "--since", "2026-09-11"]):
        with pytest.raises(SystemExit):
            P.parse_args(bad)
    a = P.parse_args(["ms2", "files", "--instrument", "ORB", "--since", "2026-09-11T10:00"])
    assert a.check is None and a.max_check == 5 and a.until is None
    a = P.parse_args(["ms2", "files", "--instrument", "ORB", "--since", "2026-09-11", "--check"])
    assert a.check == []


def test_cli_files_lists_and_checks_only_what_needs_it(server, tmp_path, capsys):
    out = tmp_path / "files.csv"
    assert cli.main(["ms2", "files", "--instrument", "ORB", "--since", "2026-09-11T00:00",
                     "--until", "2026-09-11T23:00", "--check", "--out", str(out)]) == 0
    assert server.calls == [("files", "ORB", {"instrument": "ORB",
                                              "datetime_min": "2026-09-11T00:00:00",
                                              "datetime_max": "2026-09-11T23:00:00",
                                              "sort": "sample_file_id", "order": "asc",
                                              "page": 0, "limit": IO.FILE_LIST_PAGE}),
                            ("metadata", FT, {}), ("metadata", FFIXED, {})]
    df = pd.read_csv(out, keep_default_na=False, na_values=[""],
                     dtype={"nce": str, "isolation_widths": str})
    assert list(df.columns) == list(S.FILES_COLUMNS) and len(df) == 5
    df = df.set_index("sample_file_id")
    for fid in (FROUTINE, FNEG, FA):                     # '+' / '-', 'done' / no status
        assert pd.isna(df.loc[fid, "needs_check"]), fid
    assert not df.loc[FROUTINE, "checked"] and pd.isna(df.loc[FROUTINE, "scans"])
    assert df.loc[FT, "needs_check"] == "polarity '+-', status 'needs_chemistry'"
    t = df.loc[FT]
    assert t["checked"] and t["filters_readable"] == SC.FILTERS_NO and t["events"] == 15
    assert t["nce"] == "5/15/30" and t["isolation_widths"] == "0.7" and t["spectra"] == 15
    # no list: the layout is not pinned, so the headerless scans at the
    # file's edges, the block switch and a wrap stay unassigned -- the list
    # is asked for (it places them)
    assert t["command"] == (f"peaky ms2 extract --file {FT} --polarity <+|-> "
                            "--inclusion-list <the method's list.csv>")
    assert t["precursors"] == f"{SC.PRECURSOR_RULE} 30"          # every MS2 scan with a row
    assert (t["headerless"], t["inferred"], t["unassigned"]) == (7, 0, 7)
    f = df.loc[FFIXED]
    assert f["command"] == (f"peaky ms2 extract --file {FFIXED} --polarity <+|-> "
                            "--inclusion-list <the method's list.csv>")
    assert "without a list: no precursor for event(s)" in f["check_note"]
    printed = capsys.readouterr().out
    assert "[ms2] 5 raw file(s) of ORB from 2026-09-11T00:00:00 to 2026-09-11T23:00:00; 2 need" \
        in printed
    assert f"  extract: peaky ms2 extract --file {FT} --polarity <+|->" in printed


def test_cli_files_caps_the_checks_and_checks_named_files(server, tmp_path, capsys):
    assert cli.main(["ms2", "files", "--instrument", "ORB", "--since", "2026-09-11",
                     "--check", "--max-check", "1"]) == 0
    assert [c[:2] for c in server.calls[1:]] == [("metadata", FT)]
    assert "checking 1 of 2 file(s) (--max-check 1" in capsys.readouterr().out
    server.calls.clear()
    assert cli.main(["ms2", "files", "--instrument", "ORB", "--since", "2026-09-11",
                     "--check", FA]) == 0
    assert [c[:2] for c in server.calls[1:]] == [("metadata", FA)]
    printed = capsys.readouterr().out
    assert S.MS2_WITHOUT_TRAILER_NOTE.format(n=79) in printed
    # named files are capped too: the first named one read, the other said
    server.calls.clear()
    assert cli.main(["ms2", "files", "--instrument", "ORB", "--since", "2026-09-11",
                     "--check", FT, FFIXED, "--max-check", "1"]) == 0
    assert [c[:2] for c in server.calls[1:]] == [("metadata", FT)]
    printed = capsys.readouterr().out
    assert "checking 1 of 2 file(s) (--max-check 1" in printed
    assert f"not checked: {FFIXED}" in printed
    assert cli.MS2_CHECK_ORDER not in printed
    # no --check: the listing only
    server.calls.clear()
    assert cli.main(["ms2", "files", "--instrument", "ORB", "--since", "2026-09-11"]) == 0
    assert [c[0] for c in server.calls] == ["files"]
    with pytest.raises(SystemExit, match="is not an ISO date-time"):
        cli.main(["ms2", "files", "--instrument", "ORB", "--since", "yesterday-ish"])


def test_cli_files_records_a_failing_check_and_goes_on(server, tmp_path, capsys):
    server.files[FT] = _server_error(500, "Unexpected error (ref: 0f1e2d3c)", FT)
    out = tmp_path / "f.csv"
    with pytest.raises(SystemExit) as exc:
        cli.main(["ms2", "files", "--instrument", "ORB", "--since", "2026-09-11", "--check",
                  "--out", str(out)])
    assert "1 of 2" in str(exc.value)
    df = pd.read_csv(out, keep_default_na=False).set_index("sample_file_id")
    assert df.loc[FT, "check_note"].startswith("error: ServerError: [HTTP 500]")
    assert df.loc[FFIXED, "checked"]


def test_nothing_written_carries_the_server_or_the_credentials(server, tmp_path):
    lst = tmp_path / "list.csv"
    the_list().to_csv(lst, index=False)
    assert cli.main(["ms2", "extract", "--sample-id", MTA, "--per-scan", "--save-metadata",
                     "--out-dir", str(tmp_path)]) == 0
    assert cli.main(["ms2", "extract", "--file", FT, "--polarity", "+", "--save-metadata",
                     "--inclusion-list", str(lst), "--out-dir", str(tmp_path)]) == 0
    server.files[FFIXED] = _server_error(500, "Unexpected error", FFIXED)
    with pytest.raises(SystemExit):
        cli.main(["ms2", "files", "--instrument", "ORB", "--since", "2026-09-11", "--check",
                  "--out", str(tmp_path / "files.csv")])
    files = [f for f in tmp_path.rglob("*") if f.is_file() and f.name != "list.csv"]
    assert len(files) == 1 + 2 * 8
    for f in files:
        texts = [f.read_bytes().decode("latin-1")]
        if f.suffix == ".parquet":
            texts.append(pd.read_parquet(f).to_csv())
        for text in texts:
            for b in ("sentinel-host", SENTINEL_TOKEN, "server.example", "ACCESS_TOKEN", "://"):
                assert b not in text, (f.name, b)


def test_a_key_off_its_precursor_names_where_the_key_came_from():
    """The key-offset note says what a key is: the list's precursor for a
    file's spectra (the server's key only for the server's). One row typed
    2 mDa high on an on-axis file (the other keys read the offset ~0): only
    that parent's rows carry it, read against its key x (1 + the offset)."""
    meta, _ = make_tms2(drops=())
    lst = the_list()
    lst.loc[1, "Precursor (m/z)"] += 0.002                           # typed 2 mDa high
    res = S.extract_file(None, FT, metadata=meta, inclusion_lists=[lst], polarity="+",
                         rule_check=False)
    assert abs(res.params["ms2_axis_offset"]["ppm"]) < 0.5
    first = res.parents.sort_values("nce").drop_duplicates("parent_mz")
    for r in first.itertuples():
        hit = re.search(r"-(1\.9|2\.0|2\.1) mDa from the parent key \(the inclusion list's "
                        r"precursor\)", r.note)
        assert bool(hit) == math.isclose(r.parent_mz, PREC[1] + 0.002), r.note
    assert res.per_scan["mapping"]["rule"]["checked"] is False
    with pytest.raises(SC.ScanMapError, match="do not fit the inclusion list"):
        S.extract_file(None, FT, metadata=meta, inclusion_lists=[lst], polarity="+")


# ------------------------------------------------------ analyze, per parent ----
def test_analyze_reads_each_window_at_its_own_parents_width(monkeypatch):
    """A `--file` folder whose parents were isolated at different widths has
    no single width in its manifest: each window is read at its parent's own
    (the parents table), never at the first parent's for all."""
    meta, _ = make_tms2(drops=())
    for r in meta["stats_per_scan"].values():
        if r["Scan Node:"] == "tMS2" and r["Scan Event:"] % 5 == 2:     # the 137 rows
            r["MS2 Isolation Width:"] = 1.2
            r["HighMass"] = 1.02 * (PREC[1] + 0.6) + 20
    res = S.extract_file(None, FT, metadata=meta, inclusion_lists=[the_list()], polarity="+")
    assert res.params["isolation_width_used"] is None
    assert res.params["isolation_width_source"] == "trailer, per parent"
    tables = AN.RunTables(spectra=res.spectra, parents=res.parents, window=res.window,
                          manifest=json.loads(json.dumps(S.manifest(res), default=str)))
    seen = {}
    real = AN.analyze_window

    def spy(tables, p, ions, *, width, **kw):
        seen[round(p, 4)] = width
        return real(tables, p, ions, width=width, **kw)
    monkeypatch.setattr(AN, "analyze_window", spy)
    out = AN.analyze_run(tables, _ions("C10H17+"))
    assert seen[round(PREC[1], 4)] == pytest.approx(1.2)
    others = [w for p, w in seen.items() if p != round(PREC[1], 4)]
    assert len(others) == len(PREC) - 1 and np.allclose(others, 0.7)
    assert out.params["isolation_width_source"] == AN.PER_PARENT_WIDTH
    assert out.params["isolation_widths"][repr(PREC[1])] == pytest.approx(1.2)
    assert out.params["isolation_width_used"] is None


# --------------------------------------------- a raw file's m/z axis off ----
#: window ions with formulas, for `ms2 analyze` on synthetic tMS2 files: two
#: below m/z ~175 (where max(10 ppm, 5 mDa) still reaches -28 ppm), three above
FORMS = ("C6H13O2+", "C10H19O+", "C10H19O3+", "C12H23O2+", "C11H21O4+")
FPREC = tuple(round(AN.ion_mz(f), 4) for f in FORMS)


def shift_axis(meta, truth, ppm, *, lobe=0.0, widths=4.0):
    """The payload with every centroid m/z scaled by (1 + ppm 1e-6) -- the axis
    off, the trailers not -- and, with `lobe`, a side lobe of that share of
    each precursor line `widths` line widths above it (MS2: the scan's
    precursor; MS1: every precursor line)."""
    out = json.loads(json.dumps(meta))
    prec_of = dict(zip(truth["scan"], truth["parent_mz"]))
    every = sorted({float(p) for p in truth["parent_mz"] if math.isfinite(p)})
    for s, c in enumerate(out["centroids_meta"]["data"], start=1):
        mz, inten = np.asarray(c["mzs"], float), np.asarray(c["intensities"], float)
        res, noise = np.asarray(c["resolutions"], float), np.asarray(c["noises"], float)
        p = prec_of.get(s, math.nan)
        if lobe and len(mz):
            add = []
            for q in ([p] if math.isfinite(p) else every):
                i = int(np.argmin(np.abs(mz - q)))
                if abs(mz[i] - q) < 1e-3:
                    add.append((mz[i] * (1 + widths / res[i]), lobe * inten[i], res[i], noise[i]))
            if add:
                a = np.asarray(add)
                mz, inten = np.concatenate([mz, a[:, 0]]), np.concatenate([inten, a[:, 1]])
                res, noise = np.concatenate([res, a[:, 2]]), np.concatenate([noise, a[:, 3]])
        o = np.argsort(mz, kind="stable")
        c.update(mzs=(mz[o] * (1 + ppm * 1e-6)).tolist(), intensities=inten[o].tolist(),
                 resolutions=res[o].tolist(), noises=noise[o].tolist())
    return out


def _analysed(res, *, mass_offset_ppm=None, manifest=None):
    man = manifest if manifest is not None else json.loads(json.dumps(S.manifest(res),
                                                                      default=str))
    tables = AN.RunTables(spectra=res.spectra, parents=res.parents, window=res.window,
                          manifest=man)
    kw = {} if mass_offset_ppm is None else {"mass_offset_ppm": mass_offset_ppm}
    return AN.analyze_run(tables, _ions(*FORMS), **kw)


@pytest.fixture(scope="module")
def off_axis():
    """(on-axis payload, the same 28 ppm low with a 0.5 % lobe 4 widths above
    every precursor line, truth, the list)."""
    meta, truth = make_tms2(prec=FPREC, drops=(), auto_low_block=None)
    return (shift_axis(meta, truth, 0.0, lobe=0.005), shift_axis(meta, truth, -28.0, lobe=0.005),
            truth, the_list(FPREC))


def test_an_off_axis_file_locates_its_precursors_not_their_lobes(off_axis):
    """28 ppm low: from m/z ~180 up the precursor lies outside max(10 ppm,
    5 mDa) of its key and only its upper lobe is inside. The file's offset is
    read off its precursors, the search centred on it: every parent's
    precursor is the line, never the lobe, and the MS1 window's is_parent too."""
    _, low, truth, lst = off_axis
    res = S.extract_file(None, FT, metadata=low, inclusion_lists=[lst], polarity="+")
    first = res.parents.sort_values("nce").drop_duplicates("parent_mz").set_index("parent_mz")
    assert np.allclose(first.loc[list(FPREC), "precursor_mz"], np.asarray(FPREC) * (1 - 28e-6),
                       rtol=0.5e-6, atol=0)
    for p in FPREC:
        assert re.search(r"the m/z axis sits -2[78]\.\d ppm off", first.at[p, "note"])
    axis = res.params["ms2_axis_offset"]
    # 117.091 and 155.143 read as 3-decimal keys (up to 4.3 / 3.2 ppm of
    # typing): the three 4-decimal ones read the offset
    assert axis["ppm"] == pytest.approx(-28.0, abs=0.5) and axis["parents"] == 3
    assert axis["source"] == "MS2 precursors" and axis["coarse_keys"] == 2
    assert res.params["precursor_lobe"]["guard_on"] is False      # centred on the offset
    w = res.window[res.window["is_parent"]].set_index("parent_mz")["mz"]
    assert sorted(w.index) == sorted(FPREC)
    assert np.allclose(w[list(FPREC)], np.asarray(FPREC) * (1 - 28e-6), rtol=0.5e-6)
    assert any(re.search(r"the m/z axis sits -2[78]\.\d ppm off", n) for n in res.notes)


def test_a_precursor_whose_only_candidate_is_a_side_lobe_is_not_located(off_axis):
    """Independently of the offset: with the search left on the key, a parent
    whose only candidate is a lobe of a line 200x brighter just outside the
    window is ambiguous -- its precursor not located, said -- never the lobe."""
    _, low, _, _ = off_axis
    res = SC.extract_payload("S", low, lists=[SC.inclusion_list_frame(the_list(FPREC))],
                             with_window=False)
    cen = {}
    for k, g in res.spectra.groupby("spectrum"):
        cen[k] = {"parent_peak_mz": float(g["parent_mz"].iloc[0]),
                  "activation": g["activation"].iloc[0], "mz": g["mz"].tolist(),
                  "intensity": g["intensity"].tolist(), "resolution": g["resolution"].tolist(),
                  "signal_to_noise": g["signal_to_noise"].tolist()}
    # the guard on (an unknown offset), the search left on the keys
    par = S.parents_table("S", cen, None, lobe_guard=True)
    first = par.sort_values("nce").drop_duplicates("parent_mz").set_index("parent_mz")
    for p in FPREC:
        if p * 28e-6 > S._search_tol(p):                    # the line is outside the window
            assert math.isnan(first.at[p, "precursor_mz"]), p
            assert "precursor ambiguous, not located" in first.at[p, "note"]
            assert "a side lobe of that line" in first.at[p, "note"]
        else:
            assert first.at[p, "precursor_mz"] == pytest.approx(p * (1 - 28e-6), abs=1e-4)
    assert sum(p * 28e-6 > S._search_tol(p) for p in FPREC) == 3
    # the guard is the chem.sidelobes rule: a candidate 49x weaker is kept
    mz = np.array([187.13, 187.1355])        # 5.5 mDa apart: 3.5 line widths at R 120000
    assert S.locate_precursor(mz, [100.0, 4900.0], 187.13, resolution=[1.2e5, 1.2e5]) == 0
    assert S.locate_precursor(mz, [100.0, 5000.0], 187.13, resolution=[1.2e5, 1.2e5]) is None
    assert S.locate_precursor(mz, [100.0, 5000.0], 187.13) == 0       # no widths: no guard


def test_analyze_reads_an_off_axis_folder_as_an_on_axis_one(off_axis):
    """`ms2 analyze` matches its named ions through the offset the extract
    measured (its manifest): every window analysed, each E50 the on-axis
    file's -- where it skipped every window ('no named ion matches') or read
    a lobe as the precursor."""
    on, low, _, lst = off_axis
    a = _analysed(S.extract_file(None, FT, metadata=on, inclusion_lists=[lst], polarity="+"))
    b = _analysed(S.extract_file(None, FT, metadata=low, inclusion_lists=[lst], polarity="+"))
    assert [w.status for w in a.windows] == ["analysed"] * 5
    assert [w.status for w in b.windows] == ["analysed"] * 5
    assert [w.target for w in b.windows] == list(FORMS)
    # each window's own offset (precursor against formula; the list typed to 4 decimals)
    assert all(w.offset_ppm == pytest.approx(-28.0, abs=1.0) for w in b.windows)
    ea = a.e50.set_index(["ion", "curve"])
    eb = b.e50.set_index(["ion", "curve"])
    assert list(ea.index) == list(eb.index)
    assert np.allclose(ea["e50_nce"], eb["e50_nce"], equal_nan=True, atol=1e-6)
    assert (ea["flag"] == eb["flag"]).all()
    assert b.params["mass_offset_source"] == "extract manifest"
    assert b.params["mass_offset_ppm"] == pytest.approx(-28.0, abs=0.5)


def test_a_folder_without_its_offset_says_which_one_to_pass(off_axis):
    """A folder whose axis sits off and whose manifest records no offset: the
    skip note says how far the named ion is and what to pass; with
    `--mass-offset-ppm` it is read like any other."""
    _, low, _, lst = off_axis
    res = S.extract_file(None, FT, metadata=low, inclusion_lists=[lst], polarity="+")
    out = _analysed(res, manifest={"sample_id": FT, "parameters": {"isolation_width_used": 0.7}})
    w = next(w for w in out.windows if w.parent_mz == FPREC[1])
    assert w.status == "skipped" and w.reason == AN.SKIP_NO_TARGET
    note = next(n for n in w.notes if "no named ion matches" in n)
    assert re.search(r"\(nearest C10H19O\+ at 155\.14304, \+4\.4 mDa = \+28\.\d ppm\)", note)
    assert re.search(r"pass --mass-offset-ppm -28\.\d$", note)
    out = _analysed(res, mass_offset_ppm=-28.0,
                    manifest={"sample_id": FT, "parameters": {"isolation_width_used": 0.7}})
    assert [w.status for w in out.windows] == ["analysed"] * 5
    assert out.params["mass_offset_source"] == "explicit"


def test_a_named_ion_da_away_gets_no_axis_offset_hint():
    """The skip note offers --mass-offset-ppm only within the reach the file's
    own offset is read over (sources.AXIS_SEARCH_PPM): a named ion 105 Da from
    the located precursor is another ion, and the note just says to name this
    one."""
    ions = {"C6H15N2O4+": 179.10263}
    _, _, note = AN.window_ions(74.0962, 74.09646, ions, 1.0, 0.3)
    assert "nearest C6H15N2O4+ at 179.10263" in note
    assert "--mass-offset-ppm" not in note and note.endswith("name it with --ion")
    # within the reach (0.8 of it: 40 ppm) the hint is still given; just past it
    # (1.2 of it: 60 ppm) the named ion is taken for another ion
    _, _, note = AN.window_ions(179.1, 179.10263 * (1 + 0.8 * S.AXIS_SEARCH_PPM * 1e-6), ions,
                                1.0, 0.3)
    assert re.search(r"pass --mass-offset-ppm \+40\.0$", note)
    _, _, note = AN.window_ions(179.1, 179.10263 * (1 + 1.2 * S.AXIS_SEARCH_PPM * 1e-6), ions,
                                1.0, 0.3)
    assert "no named ion matches" in note
    assert "--mass-offset-ppm" not in note and note.endswith("name it with --ion")


def test_cli_analyze_never_folds_away_the_axis_offset_a_folder_needs(off_axis, tmp_path, capsys,
                                                                      monkeypatch):
    """A folder 28 ppm off whose manifest records no offset, every window's ion
    named: each window misses its named ion by the axis offset, and its note
    says what to pass. Beyond MS2_ANALYZE_UNNAMED_SHOWN those windows are one
    line -- naming the offset they need, never 'name them with --ion' -- and
    the exit says to pass it, not to name ions already named."""
    _, low, _, lst = off_axis
    res = S.extract_file(None, FT, metadata=low, inclusion_lists=[lst], polarity="+")
    d = tmp_path / "f"
    cli._write_ms2_outputs(res, d)
    man = json.loads((d / "ms2_manifest.json").read_text())
    del man["parameters"]["ms2_axis_offset"]
    (d / "ms2_manifest.json").write_text(json.dumps(man))
    monkeypatch.setattr(cli, "MS2_ANALYZE_UNNAMED_SHOWN", 1)
    argv = ["ms2", "analyze", "--run-dir", str(d), "--no-figures", "--out-dir", str(tmp_path / "a")]
    for f, m in zip(FORMS, FPREC):
        argv += ["--ion", f"{m}={f}"]
    with pytest.raises(SystemExit) as exc:
        cli.main(argv)
    out = capsys.readouterr().out
    assert re.search(r"\[ms2\] 5 of 5 parents skipped: each named ion lies off its located "
                     r"precursor by the folder's m/z axis offset \(median -2[78]\.\d ppm\): pass "
                     r"--mass-offset-ppm -2[78]\.\d", out), out
    assert "name them with --ion" not in out
    assert re.search(r"pass --mass-offset-ppm -2[78]\.\d", str(exc.value)), exc.value
    assert "name each window's target ion" not in str(exc.value)
    # at or below the cap they are printed one by one, each with its note
    monkeypatch.setattr(cli, "MS2_ANALYZE_UNNAMED_SHOWN", 5)
    with pytest.raises(SystemExit):
        cli.main(argv)
    out = capsys.readouterr().out
    hints = re.findall(r"note: skipped: no named ion matches .* pass --mass-offset-ppm", out)
    assert len(hints) == 5


def test_the_docs_say_how_far_the_offset_hint_reaches_and_what_is_folded():
    """The skip note's --mass-offset-ppm hint is given only within the axis
    reach, and the console folds the windows skipped for want of a named
    target beyond MS2_ANALYZE_UNNAMED_SHOWN: docs §11 and the window_ions
    docstring say the first, §11.7 the second -- never 'one line per window'
    unqualified."""
    root = Path(__file__).resolve().parent.parent
    docs = " ".join((root / "docs" / "MS2.md").read_text(encoding="utf-8").split())
    target = docs[docs.index("**Target:**"):docs.index("**The folder's m/z axis**")]
    assert "`sources.AXIS_SEARCH_PPM` (50 ppm" in target
    assert "AXIS_SEARCH_PPM" in " ".join(AN.window_ions.__doc__.split())
    sec = docs[docs.index("It prints one line per window"):]
    sec = sec[:sec.index("Each window has a **status**")]
    assert "`cli.MS2_ANALYZE_UNNAMED_SHOWN` (5)" in sec and "ms2_analyze_manifest.json" in sec
    assert cli.MS2_ANALYZE_UNNAMED_SHOWN == 5


def test_cli_analyze_sums_up_the_windows_no_named_ion_matches(off_axis, tmp_path, capsys,
                                                               monkeypatch):
    """An inclusion-list folder holds many parents and the user names a few:
    beyond MS2_ANALYZE_UNNAMED_SHOWN the unnamed windows are one summary line,
    not a block each; at or below it they are printed as before."""
    on, _, _, lst = off_axis
    res = S.extract_file(None, FT, metadata=on, inclusion_lists=[lst], polarity="+")
    d = tmp_path / "f"
    cli._write_ms2_outputs(res, d)
    ion = f"{FPREC[0]}={FORMS[0]}"
    monkeypatch.setattr(cli, "MS2_ANALYZE_UNNAMED_SHOWN", 1)
    assert cli.main(["ms2", "analyze", "--run-dir", str(d), "--ion", ion, "--no-figures",
                     "--out-dir", str(tmp_path / "a")]) == 0
    out = capsys.readouterr().out
    n = len(FPREC) - 1
    assert f"[ms2] {n} of {len(FPREC)} parents skipped: no named ion at their located" in out
    heads = [x for x in out.splitlines() if x.startswith("[ms2] parent ")]
    assert not [x for x in heads if x.endswith(": skipped")] and "no named ion matches" not in out
    assert f"parent {S.spectrum_key(FPREC[0])}: analysed" in out
    man = json.loads((tmp_path / "a" / "ms2_analyze_manifest.json").read_text())
    assert sum("no named ion matches" in " ".join(w["notes"]) for w in man["windows"]) == n
    monkeypatch.setattr(cli, "MS2_ANALYZE_UNNAMED_SHOWN", n)
    assert cli.main(["ms2", "analyze", "--run-dir", str(d), "--ion", ion, "--no-figures",
                     "--out-dir", str(tmp_path / "b")]) == 0
    out = capsys.readouterr().out
    heads = [x for x in out.splitlines() if x.startswith("[ms2] parent ")]
    assert len([x for x in heads if x.endswith(": skipped")]) == n
    assert out.count("no named ion matches") == n and "parents skipped" not in out


# ---------------------------------------- a first mass that moves per NCE ----
def test_a_parents_spectra_are_cut_to_the_range_every_energy_recorded():
    """Experiment 2 (NCE 15) records from m/z 50-55 (an automatic first mass),
    the others from 40: a C3H7+ fragment (m/z 43.05) is in the NCE 5 and 30
    spectra only. Read as it is, the target's F jumps at NCE 15 and its E50
    moves; cut to the range every energy recorded, the breakdown is the one
    of a file without that fragment -- and said so."""
    def payload(frag):
        meta, truth = make_tms2(prec=FPREC, drops=(), auto_low_block=1)
        if frag:
            for t, c, r in zip(truth.itertuples(), meta["centroids_meta"]["data"],
                               [meta["stats_per_scan"][str(s)] for s in truth["scan"]]):
                if t.order == 2 and r["LowMass"] < 43:
                    surv = math.exp(-t.nce / 15.0)
                    for k, v in (("mzs", 43.05423), ("intensities", 3e4 * (1 - surv) + 200),
                                 ("resolutions", 1.5e5), ("noises", 100.0)):
                        c[k].insert(0, v)
        return meta
    lst = [the_list(FPREC)]
    with_frag = S.extract_file(None, FT, metadata=payload(True), inclusion_lists=lst,
                               polarity="+")
    without = S.extract_file(None, FT, metadata=payload(False), inclusion_lists=lst,
                             polarity="+")
    pd.testing.assert_frame_equal(with_frag.spectra, without.spectra)
    ea = _analysed(with_frag).e50.set_index(["ion", "curve"])
    eb = _analysed(without).e50.set_index(["ion", "curve"])
    assert np.allclose(ea["e50_nce"], eb["e50_nce"], equal_nan=True)
    par = with_frag.parents.set_index("spectrum")
    for j, p in enumerate(FPREC):
        lo = 50.0 + 1.3 * j
        sp = with_frag.spectra[np.isclose(with_frag.spectra["parent_mz"], p)]
        assert sp["mz"].min() >= lo
        note = par.at[S.spectrum_key(p, "hcd15.00"), "note"]
        assert (f"scan range differs between this parent's energies (first mass 40.00 at NCE "
                f"5/30, {lo:.2f} at NCE 15): every spectrum of it is cut to m/z {lo:.2f}-") in note
    assert with_frag.per_scan["averaging"]["range_cuts"] == {repr(p): 2 for p in FPREC}


# ------------------------------------------- review round 1: CLI and I/O ----
def _api_error(status, text, fid):
    from mascope_sdk.exceptions import AuthenticationError, NotFoundError
    cls = NotFoundError if status == 404 else AuthenticationError
    return cls(text, status_code=status,
               url=f"https://server.example/api/sample/files/{fid}/metadata")


@pytest.mark.parametrize("status,hint", [
    (403, "the server refuses this raw file: an unknown id, or one your token may not read"),
    (404, "no such raw file on this server")])
def test_a_raw_file_failure_points_to_the_file_listing(server, tmp_path, capsys, status, hint):
    server.files[FT] = _api_error(status, "You do not have permission to perform this action.",
                                  FT)
    with pytest.raises(SystemExit) as exc:
        cli.main(["ms2", "extract", "--file", FT, "--polarity", "+", "--out-dir", str(tmp_path)])
    assert str(exc.value) == f"[ms2] 1 of 1 raw file(s) could not be read: {FT}"
    printed = capsys.readouterr().out
    assert hint in printed
    assert "list the raw-file ids with `peaky ms2 files --instrument NAME --since ISO`" in printed
    assert "peaky list" not in printed.replace("peaky ms2 files", "")
    # `ms2 files --check` says the same of the file it could not read
    with pytest.raises(SystemExit) as exc:
        cli.main(["ms2", "files", "--instrument", "ORB", "--since", "2026-09-11", "--check", FT])
    assert str(exc.value) == f"[ms2] 1 of 1 raw file(s) could not be read: {FT}"
    assert hint in capsys.readouterr().out


def test_the_check_reads_the_longest_flagged_files_first(server, capsys):
    """Short flagged files (a composite MS1 run of 32 s that the server's
    reader saw as '+-') listed before a long tMS2 file: the cap goes to the
    long one, and the line says the order."""
    shorts = [_routine_rec(f"FILE{k}xxxxxxxxxxx", length=32.0, polarity="+-",
                           processing_status="needs_chemistry",
                           datetime_utc=f"2026-09-11T0{k}:00:00+00:00") for k in (6, 7, 8)]
    server.listing = shorts + [_routine_rec(FT, length=2466.0, polarity="+-",
                                            processing_status="needs_chemistry",
                                            datetime_utc="2026-09-11T09:00:00+00:00")]
    assert cli.main(["ms2", "files", "--instrument", "ORB", "--since", "2026-09-11",
                     "--check", "--max-check", "1"]) == 0
    assert [c[:2] for c in server.calls[1:]] == [("metadata", FT)]
    printed = capsys.readouterr().out
    assert "checking 1 of 4 file(s) (--max-check 1" in printed
    assert f"{cli.MS2_CHECK_ORDER}); not checked: FILE8xxxxxxxxxxx" in printed


def test_since_and_until_are_compared_in_utc(server, capsys):
    # 10:00+03:00 is 07:00 UTC: a valid two hours before 09:00 UTC
    assert cli.main(["ms2", "files", "--instrument", "ORB", "--since", "2026-09-11T10:00+03:00",
                     "--until", "2026-09-11T09:00Z"]) == 0
    p = server.calls[0][2]
    assert (p["datetime_min"], p["datetime_max"]) == ("2026-09-11T07:00:00", "2026-09-11T09:00:00")
    # 11:00+05:00 is 06:00 UTC: before 10:00 UTC, whatever the text says
    server.calls.clear()
    with pytest.raises(SystemExit, match="is not after --since"):
        cli.main(["ms2", "files", "--instrument", "ORB", "--since", "2026-09-11T10:00Z",
                  "--until", "2026-09-11T11:00+05:00"])
    assert server.calls == []


def test_an_unreadable_payload_fails_that_check_alone(server, tmp_path, capsys):
    server.files[FT] = {}                         # the server answered nothing usable
    out = tmp_path / "f.csv"
    with pytest.raises(SystemExit) as exc:
        cli.main(["ms2", "files", "--instrument", "ORB", "--since", "2026-09-11", "--check",
                  "--out", str(out)])
    assert str(exc.value) == f"[ms2] 1 of 2 raw file(s) could not be read: {FT}"
    assert [c[:2] for c in server.calls[1:]] == [("metadata", FT), ("metadata", FFIXED)]
    df = pd.read_csv(out, keep_default_na=False).set_index("sample_file_id")
    assert df.loc[FT, "check_note"].startswith("error: the file metadata holds no per-scan")
    assert df.loc[FFIXED, "checked"]
    assert FFIXED in capsys.readouterr().out


def test_a_per_scan_payload_without_centroids_keeps_the_ms2_tables(server, tmp_path):
    server.files[FA] = {}                         # `data: null` reads {}
    res = S.extract_sample(server, MTA, per_scan=True)
    assert len(res.spectra) == 230 and len(res.scans) == 0
    assert any(n.startswith("per-scan tables skipped: the server could not serve") and
               "no per-scan centroids" in n for n in res.notes)
    assert "timeseries" not in [c[0] for c in server.calls]
    assert cli.main(["ms2", "extract", "--sample-id", MTA, "--per-scan",
                     "--out-dir", str(tmp_path)]) == 0
    assert (tmp_path / "ms2" / MTA / "ms2_spectra.parquet").exists()


def test_a_kept_payload_is_the_runs_input_or_gone(server, tmp_path, own):
    """(a) An offline run's payload is named in its manifest (name, sha1,
    bytes); the folder's own payload, read back, is among its files. (b) A
    re-run that neither keeps nor reads the payload removes it, so every file
    of the folder is in the manifest."""
    from peaky.reporting import provenance as PV
    lst = tmp_path / "list.csv"
    the_list().to_csv(lst, index=False)
    with pytest.raises(SystemExit):              # no polarity: fails, payload kept
        cli.main(["ms2", "extract", "--file", FT, "--save-metadata", "--inclusion-list",
                  str(lst), "--out-dir", str(tmp_path)])
    d = tmp_path / "ms2" / FT
    kept = d / cli.MS2_METADATA_FILE
    assert cli.main(["ms2", "extract", "--file", FT, "--metadata-json", str(kept), "--polarity",
                     "+", "--inclusion-list", str(lst), "--out-dir", str(tmp_path)]) == 0
    man = json.loads((d / "ms2_manifest.json").read_text())
    assert man["parameters"]["metadata_json"] == {
        "name": cli.MS2_METADATA_FILE, "sha1": PV.sha1_file(str(kept)),
        "bytes": kept.stat().st_size, "offline": True}
    assert man["files"][cli.MS2_METADATA_FILE] == PV.sha1_file(str(kept))
    # another payload, offline into another folder: named, not listed as a file
    other = tmp_path / "elsewhere.json"
    other.write_text(json.dumps(make_tms2(seed=3)[0]))
    assert cli.main(["ms2", "extract", "--file", FFIXED, "--metadata-json", str(other),
                     "--polarity", "+", "--inclusion-list", str(lst),
                     "--out-dir", str(tmp_path)]) == 0
    man = json.loads((tmp_path / "ms2" / FFIXED / "ms2_manifest.json").read_text())
    assert man["parameters"]["metadata_json"]["name"] == "elsewhere.json"
    assert cli.MS2_METADATA_FILE not in man["files"]
    # (b) a sample's kept payload, then a plain re-run
    assert cli.main(["ms2", "extract", "--sample-id", MTA, "--per-scan", "--save-metadata",
                     "--out-dir", str(tmp_path)]) == 0
    assert (tmp_path / "ms2" / MTA / cli.MS2_METADATA_FILE).exists()
    assert cli.main(["ms2", "extract", "--sample-id", MTA, "--out-dir", str(tmp_path)]) == 0
    sd = tmp_path / "ms2" / MTA
    man = json.loads((sd / "ms2_manifest.json").read_text())
    assert not (sd / cli.MS2_METADATA_FILE).exists()
    assert {p.name for p in sd.iterdir()} == set(man["files"]) | {"ms2_manifest.json"}


def test_the_check_names_what_a_list_cannot_fix(own):
    """`ms2 files --check`: a file refused for a reason no list lifts (a
    stepped energy, two MS1 events) gets no command and keeps its scan-range
    verdict; a fixed range asks for the list; a layout the Full Scan event
    does not pin asks for it too."""
    row = {"sample_file_id": FT, "polarity": "+-"}
    meta, _ = make_tms2(drops=())
    meta["stats_per_scan"]["10"]["HCD Energy:"] = "15.00,30.00"
    chk = S.check_file(meta, row)
    assert chk["scan_range_rule"] == ("HighMass differs between the events (not judged: the "
                                      "mapping was refused)")
    assert chk["command"] == "" and "cannot be extracted as it is" in chk["check_note"]
    assert "stepped" in chk["check_note"]
    meta, _ = make_tms2(drops=())
    meta["stats_per_scan"]["2"]["Scan Event:"] = 99            # a second MS1 event
    chk = S.check_file(meta, row)
    assert chk["command"] == "" and "2 scan events" in chk["check_note"]
    chk = S.check_file(make_tms2(fixed_range=True)[0], row)
    assert chk["scan_range_rule"] == "does not apply (a fixed range in experiment(s) 1, 2, 3)"
    assert chk["command"].endswith(" --inclusion-list <the method's list.csv>")
    chk = S.check_file(make_tms2(fs_scans=0)[0], row)            # no Full Scan event
    assert chk["command"].endswith(" --inclusion-list <the method's list.csv>")
    assert "not pinned" in chk["check_note"]


def test_the_polarity_rule_record_and_filters(own):
    """An explicit sign that contradicts readable filters is refused; with
    garbled filters a one-sign file record names the polarity."""
    with pytest.raises(ValueError, match="contradicts the scan filters, which read"):
        S.extract_file(None, FT, metadata=make_tms2(garble=0.0)[0],
                       inclusion_lists=[the_list()], polarity="-")
    res = S.extract_file(None, FT, metadata=make_tms2()[0], inclusion_lists=[the_list()],
                         file_row={"polarity": "+"})
    assert (res.row["polarity"], res.params["polarity_source"]) == ("+", "file record")


def test_a_time_series_for_every_activation_leaves_the_parameter_out():
    seen = []

    class _M:
        def get_timeseries(self, parent, **kw):
            seen.append(kw)
            return {"time": []}

    class _C:
        samples = type("_S", (), {"ms2": staticmethod(lambda sid: _M())})()
    for act in (None, ""):
        IO.fetch_ms2_timeseries(_C(), MTA, 137.1321, activation=act)
    IO.fetch_ms2_timeseries(_C(), MTA, 137.1321, activation="hcd1.00")
    assert [k["activation"] for k in seen] == [None, None, "hcd1.00"]


def test_the_file_listing_is_read_page_by_page(monkeypatch):
    """Sorted by id (unique), `page` from 0, a pause between pages, until a
    short page; a server that ignores the paging (the whole range on every
    page) stops at the first page that adds nothing. Each file once, by time."""
    monkeypatch.setattr(IO, "FILE_LIST_PAGE", 2)
    sleeps = []
    monkeypatch.setattr(IO, "time", type("_T", (), {"sleep": staticmethod(sleeps.append)}))
    rows = [{"sample_file_id": f"FILE{k}xxxxxxxxxxx", "datetime_utc": f"2026-09-11T1{5 - k}:00"}
            for k in range(1, 6)]
    asked = []

    def paged(client, path, *, params=None, timeout=None, body=False):
        asked.append(params["page"])
        by_id = sorted(rows, key=lambda r: r["sample_file_id"])
        return by_id[params["page"] * params["limit"]:(params["page"] + 1) * params["limit"]]
    monkeypatch.setattr(IO, "_api_get", paged)
    out = IO.fetch_sample_files(object(), instrument="ORB", since="2026-09-11T00:00:00")
    assert asked == [0, 1, 2] and sleeps == [IO.FILE_LIST_PAUSE_S] * 2
    assert [r["sample_file_id"] for r in out] == [r["sample_file_id"] for r in rows[::-1]]
    asked.clear()
    monkeypatch.setattr(IO, "_api_get", lambda c, p, **kw: asked.append(1) or list(rows))
    out = IO.fetch_sample_files(object(), instrument="ORB", since="2026-09-11T00:00:00")
    assert len(asked) == 2 and len(out) == 5


def test_every_ms2_module_is_indexed():
    """docs/PIPELINE.md's MS2 row and the package docstring name every module
    of peaky/ms2 (the per-scan source among them)."""
    import peaky.ms2 as M
    root = Path(__file__).resolve().parent.parent
    row = next(line for line in (root / "docs" / "PIPELINE.md").read_text(encoding="utf-8")
               .splitlines() if "[`MS2.md`]" in line)
    for f in sorted((root / "peaky" / "ms2").glob("*.py")):
        if f.name != "__init__.py":
            assert f.name in row, f.name
            assert f"`{f.stem}`" in M.__doc__, f.stem
    assert "§12" in row and "ms2 files" in row


def test_the_console_names_a_fixed_range_experiment(server, tmp_path, capsys):
    meta, _ = make_tms2()
    for r in meta["stats_per_scan"].values():
        if r["Scan Event:"] in range(11, 16):
            r["HighMass"] = 600.0
    mj = tmp_path / "m.json"
    mj.write_text(json.dumps(meta))
    lst = tmp_path / "list.csv"
    the_list().to_csv(lst, index=False)
    assert cli.main(["ms2", "extract", "--file", FT, "--metadata-json", str(mj), "--polarity", "+",
                     "--inclusion-list", str(lst), "--out-dir", str(tmp_path)]) == 0
    assert ("scan range follows the precursor (max residual 0 mDa over 10 events), except in "
            "experiment(s) 3 (a fixed range)") in capsys.readouterr().out


def test_the_ms1_window_is_read_through_the_axis_offset():
    """The MS1 window of a file 28 ppm low: centred on key x (1 + offset), it
    holds a centroid 3 mDa inside its lower edge that a window centred on the
    key leaves out; analyze's MS1 shares read its centroids through the offset."""
    p, half = 200.0, 0.35 + S.WINDOW_PAD_DA
    low = (p - half + 0.003) * (1 - 28e-6)            # a co-isolate near the lower edge
    line = p * (1 - 28e-6)
    ms1 = {"mz": [low, line], "intensity": [100.0, 1000.0], "resolution": [1.2e5, 1.2e5],
           "signal_to_noise": [10.0, 100.0]}
    on = S.window_table("S", ms1, [p], 0.7, axis_offset_ppm=-28.0)
    assert len(on) == 2 and on.loc[on["is_parent"], "mz"].iloc[0] == pytest.approx(line)
    assert len(S.window_table("S", ms1, [p], 0.7)) == 1
    ions = {"A+": p, "B+": p - half + 0.003}
    shares = AN._ms1_shares(on, p, ions, 0.3, half, axis_offset_ppm=-28.0)
    assert shares == {"A+": pytest.approx(1000 / 1100), "B+": pytest.approx(100 / 1100)}
    assert AN._ms1_shares(on, p, ions, 0.3, half) == {"A+": 0.0, "B+": 0.0}


# ------------------------------------------------------------ review round 2 ----
def _typed_payload(prec, decimals, *, truth=False):
    """make_tms2 on `prec` (every centroid on its exact m/z) whose method typed
    its list with `decimals`: each tMS2 row's HighMass is derived from the
    TYPED value, as the instrument derives it -> (payload, the typed list[,
    the truth])."""
    meta, tr = make_tms2(prec=prec, drops=(), auto_low_block=None)
    typed = tuple(round(p, decimals) for p in prec)
    for r in meta["stats_per_scan"].values():
        if r["Scan Node:"] == "tMS2":
            hi = 1.02 * (typed[(int(r["Scan Event:"]) - 1) % len(prec)] + 0.35) + 20
            r["HighMass"] = hi
            r["ScanType"] = r["ScanType"].rsplit("[", 1)[0] + f"[{r['LowMass']:.4f}-{hi:.4f}]"
    return (meta, the_list(typed), tr) if truth else (meta, the_list(typed))


def test_keys_typed_with_two_decimals_read_no_axis_offset():
    """An on-axis file whose list was typed with two decimals (117.09 for
    117.0910): each key sits up to 5 mDa (23-43 ppm here) from its precursor
    by its typing alone, so no key can measure an axis offset of a few ppm.
    No offset is recorded, nothing says the axis is off, and `ms2 analyze`
    analyses every window -- with the E50s of the same file typed exactly."""
    meta, lst = _typed_payload(FPREC, 2)
    coarse = S.extract_file(None, FT, metadata=meta, inclusion_lists=[lst], polarity="+")
    m4, l4 = _typed_payload(FPREC, 4)
    exact = S.extract_file(None, FT, metadata=m4, inclusion_lists=[l4], polarity="+")
    ax = coarse.params["ms2_axis_offset"]
    assert ax["ppm"] is None and ax["coarse_keys"] == 5 and ax["parents"] == 0
    assert not any("m/z axis sits" in n for n in coarse.notes)
    assert any("m/z axis offset is not measured" in n for n in coarse.notes)
    assert not coarse.parents["note"].str.contains("m/z axis sits").any()
    first = coarse.parents.sort_values("nce").drop_duplicates("parent_mz")
    assert np.allclose(sorted(first["precursor_mz"]), sorted(FPREC), atol=2e-4)
    a, b = _analysed(exact), _analysed(coarse)
    assert [w.status for w in b.windows] == ["analysed"] * 5
    assert [w.target for w in b.windows] == list(FORMS)
    ea, eb = a.e50.set_index(["ion", "curve"]), b.e50.set_index(["ion", "curve"])
    assert list(ea.index) == list(eb.index)
    assert np.allclose(ea["e50_nce"], eb["e50_nce"], equal_nan=True, atol=1e-9)
    # the 4-decimal keys read the offset (the 3-decimal-looking 117.091 and
    # 155.143 do not): about 0 ppm
    ax = exact.params["ms2_axis_offset"]
    assert ax["parents"] == 3 and ax["coarse_keys"] == 2 and abs(ax["ppm"]) < 0.5
    # a server-style answer keyed 137.13 (our own standard B's key): no offset
    cen = {"137.13@hcd5.00": {"parent_peak_mz": 137.13, "activation": "hcd5.00",
                              "mz": [95.0855, 137.1323], "intensity": [1e3, 1e5]}}
    assert S.axis_offset(cen, None)["ppm"] is None


def test_an_unmeasured_offset_keeps_the_side_lobe_guard_on():
    """The same file 28 ppm low, its list typed with two decimals: no key can
    measure the offset, so the search stays on the keys with its side-lobe
    guard on. 199.17's precursor (199.1693 x (1 - 28e-6) = 199.16372) lies
    6.3 mDa below its key, outside the window, and only its upper lobe is
    inside: ambiguous, never the lobe. The others lie inside: located."""
    meta, lst, truth = _typed_payload(FPREC, 2, truth=True)
    low = shift_axis(meta, truth, -28.0, lobe=0.005)
    res = S.extract_file(None, FT, metadata=low, inclusion_lists=[lst], polarity="+")
    assert res.params["ms2_axis_offset"]["ppm"] is None
    assert res.params["precursor_lobe"]["guard_on"] is True
    first = res.parents.sort_values("nce").drop_duplicates("parent_mz").set_index("parent_mz")
    assert math.isnan(first.at[199.17, "precursor_mz"])
    assert "a side lobe of that line" in first.at[199.17, "note"]
    for p, key in zip(FPREC, (117.09, 155.14, 187.13, 199.17, 217.14)):
        if key != 199.17:
            assert first.at[key, "precursor_mz"] == pytest.approx(p * (1 - 28e-6), abs=2e-4)
    w = res.window
    assert not w.loc[np.isclose(w["parent_mz"], 199.17), "is_parent"].any()


@pytest.mark.parametrize("ppm", [5.0, -13.5, 28.6])
def test_per_scan_reads_a_calibrated_samples_peaks_on_the_raw_axis(server, ppm):
    """The server multiplies the m/z of the spectra it serves by the file's
    calibration factor (the record's `mz_calibration`); the per-scan centroids
    are the raw axis. Each server peak is matched at m/z / factor: the same
    per-scan readings as an uncalibrated sample, the server's m/z kept."""
    base = S.extract_sample(server, MTA, per_scan=True).scan_peaks
    f = 1.0 + ppm * 1e-6
    st = server.store[MTA]
    st["record"] = dict(st["record"], mz_calibration={"par": {"calibration_factor": f}})
    st["centroids"] = {k: dict(v, mz=[m * f for m in v["mz"]]) for k, v in st["centroids"].items()}
    res = S.extract_sample(server, MTA, per_scan=True)
    pk = res.scan_peaks
    assert len(pk) == len(base) == 1517
    assert np.array_equal(pk["intensity"], base["intensity"])
    assert np.array_equal(pk["n_centroids"], base["n_centroids"])
    assert np.allclose(pk["peak_mz"], base["peak_mz"] * f, rtol=1e-12, atol=0)
    assert res.per_scan["mz_calibration_factor"] == f and res.per_scan["peaks_all_zero"] == 0
    assert not any("read 0 in every scan" in n for n in res.notes)


def test_per_scan_says_when_server_peaks_read_nothing_in_any_scan(server):
    """A served axis the record does not explain (scaled, no calibration in
    the record): the peaks that read 0 in every scan of their spectrum are
    counted and noted -- never left as silent zeros."""
    st = server.store[MTA]
    st["centroids"] = {k: dict(v, mz=[m * (1 + 5e-6) for m in v["mz"]])
                       for k, v in st["centroids"].items()}
    res = S.extract_sample(server, MTA, per_scan=True)
    n = res.per_scan["peaks_all_zero"]
    assert n > 100
    assert any(f"per-scan: {n} of the 230 server peaks read 0 in every scan" in x
               for x in res.notes)
    # a calibration the server marked failed carries no factor: read as served, said
    st["record"] = dict(st["record"], mz_calibration={"status": "failed"})
    res = S.extract_sample(server, MTA, per_scan=True)
    assert res.per_scan["mz_calibration_factor"] == 1.0
    assert any("holds no calibration factor" in x for x in res.notes)


def _co_isolate_answer(*, decay=15.0):
    """A server-style by_activation answer: the weak target C10H15O4+ on its
    key (round(exact, 4)), with an unnamed co-isolate 8.0 mDa above it (4.8
    line widths) 100x brighter at NCE 0, decaying with NCE / `decay`."""
    m = AN.ion_mz("C10H15O4+")
    key, co = round(m, 4), m + 0.0080
    cen, groups = {}, []
    for k, nce in enumerate((5.0, 15.0, 30.0)):
        surv = math.exp(-nce / 15)
        mz = [m - 18.010565, m - 44.0, m, co]
        it = [3e3 * (1 - surv) + 200, 2e3 * (1 - surv) + 200, 2e3 * surv,
              2e5 * math.exp(-nce / decay)]
        act = f"hcd{nce:.2f}"
        cen[S.spectrum_key(key, act)] = {
            "parent_peak_mz": key, "activation": act, "mz": mz, "intensity": it,
            "resolution": [120000 * math.sqrt(200 / x) for x in mz], "signal_to_noise": [50] * 4}
        groups.append({"parent_peak_mz": key, "activation": act, "hcd_energy": [4 + nce / 5],
                       "scan_count": 5, "t_min": 10.0 * k, "t_max": 10.0 * k + 8})
    return m, cen, {"parent_peaks": [key], "groups": groups, "isolation_width": 0.7}


def test_a_server_folder_locates_a_weak_precursor_beside_a_bright_co_isolate():
    """The side-lobe guard is for a raw file whose m/z axis offset is unknown
    (the precursor can then lie beyond the window with only its lobe inside);
    the server's spectra are calibrated and their keys are the isolation
    centres. A weak target on its key, a 100x brighter ion 4.8 line widths
    above it: located and analysed (F-E50 7.43), as before the guard."""
    m, cen, summ = _co_isolate_answer()
    par = S.parents_table("S", cen, summ)
    assert np.allclose(par["precursor_mz"], m, atol=1e-6)
    assert not par["note"].str.contains("ambiguous").any()
    ms1 = {"mz": [m, m + 0.008], "intensity": [2e3, 2e5], "resolution": [1.2e5, 1.2e5],
           "signal_to_noise": [50.0, 50.0]}
    w = S.window_table("S", ms1, [round(m, 4)], 0.7)
    assert w.loc[w["is_parent"], "mz"].tolist() == [m]
    tabs = AN.RunTables(spectra=S.spectra_table("S", cen, summ), parents=par,
                        window=S._frame({}, S.WINDOW_COLUMNS),
                        manifest={"sample_id": "S", "parameters": {"isolation_width_used": 0.7}})
    out = AN.analyze_run(tabs, _ions("C10H15O4+"))
    assert out.windows[0].status == "analysed"
    assert _e50(out, round(m, 4), "C10H15O4+")["e50_nce"] == pytest.approx(7.43, abs=0.01)
    # with the guard on (a raw file's unknown axis) the candidate is refused at
    # every energy -- the co-isolate is 100x at each
    par = S.parents_table("S", cen, summ, lobe_guard=True)
    assert par["precursor_mz"].isna().all()
    assert par["note"].str.contains("precursor ambiguous, not located").all()


def test_the_guard_moves_on_to_the_next_energy():
    """Guard on: the candidate is refused at NCE 5 (the co-isolate 51x) but
    not at NCE 15, where the co-isolate has fragmented to 14x: the precursor
    is located there and followed across every energy."""
    m, cen, summ = _co_isolate_answer(decay=5.0)
    par = S.parents_table("S", cen, summ, lobe_guard=True)
    assert np.allclose(par["precursor_mz"], m, atol=1e-6)
    assert not par["note"].str.contains("ambiguous").any()


def test_the_check_reads_unprocessed_files_before_long_processed_ones(server, own, capsys):
    """A long dual-polarity file the server processed ('+-', 'done') and a
    shorter garbled tMS2 file ('+-', 'needs_chemistry'): the cap goes to the
    file the server could not build a sample from; the processed one is
    left unchecked, and the line says the order."""
    fdual = "FILE9xxxxxxxxxxx"
    server.files[fdual] = own[0]
    server.listing = [_routine_rec(fdual, length=3600.0, polarity="+-"),
                      _routine_rec(FT, length=900.0, polarity="+-",
                                   processing_status="needs_chemistry")]
    assert cli.main(["ms2", "files", "--instrument", "ORB", "--since", "2026-09-11",
                     "--check", "--max-check", "1"]) == 0
    assert [c[:2] for c in server.calls[1:]] == [("metadata", FT)]
    printed = capsys.readouterr().out
    assert (f"{cli.MS2_CHECK_ORDER}); not checked: {fdual}") in printed
    assert "a processing status other than 'done' first" in cli.MS2_CHECK_ORDER
    # a file with no status is in the second tier, as a 'done' one
    server.calls.clear()
    server.listing[0] = dict(server.listing[0], processing_status=None)
    assert cli.main(["ms2", "files", "--instrument", "ORB", "--since", "2026-09-11",
                     "--check", "--max-check", "1"]) == 0
    assert [c[:2] for c in server.calls[1:]] == [("metadata", FT)]
    # a server that lists no status: one tier, the longest first
    server.calls.clear()
    server.listing = [dict(r, processing_status=None) for r in server.listing]
    assert cli.main(["ms2", "files", "--instrument", "ORB", "--since", "2026-09-11",
                     "--check", "--max-check", "1"]) == 0
    assert [c[:2] for c in server.calls[1:]] == [("metadata", fdual)]


def _weak_fragment_payload():
    """make_tms2 with one more fragment in every MS2 scan (precursor - C2H4,
    S/N 4.2 per scan): S/N 5.9-7.3 in its spectrum, kept at 3, not at 10."""
    meta, truth = make_tms2()
    for t, c in zip(truth.itertuples(), meta["centroids_meta"]["data"]):
        if t.order == 2:
            for k, v in (("mzs", t.parent_mz - 28.0313), ("intensities", 420.0),
                         ("resolutions", 1.5e5), ("noises", 100.0)):
                c[k].append(v)
    return meta


def test_cli_extract_file_passes_every_flag_on(tmp_path, monkeypatch):
    """`--file` offline: --noise-threshold, --iso-width, --no-window and
    --no-rule-check each reach the extraction (and its manifest)."""
    monkeypatch.setattr(cli, "_require_creds", lambda *a, **k: pytest.fail("offline"))
    monkeypatch.setattr(IO, "connect", lambda *a, **k: pytest.fail("offline"))
    mj = tmp_path / "m.json"
    mj.write_text(json.dumps(_weak_fragment_payload()))
    lst, rev = tmp_path / "list.csv", tmp_path / "rev.csv"
    the_list().to_csv(lst, index=False)
    the_list(PREC[::-1]).to_csv(rev, index=False)

    def run(name, *flags, listing=lst):
        out = tmp_path / name
        rc = cli.main(["ms2", "extract", "--file", FT, "--metadata-json", str(mj), "--polarity",
                       "+", "--inclusion-list", str(listing), *flags, "--out-dir", str(out)])
        d = out / "ms2" / FT
        return (rc, json.loads((d / "ms2_manifest.json").read_text()),
                pd.read_parquet(d / "ms2_spectra.parquet"),
                pd.read_csv(d / "ms2_window.csv"))
    rc, man, sp, win = run("plain")
    assert rc == 0 and man["parameters"]["noise_threshold"] == 3.0
    assert man["parameters"]["isolation_width_used"] == 0.7 and man["parameters"]["rule_check"]
    assert ((sp["signal_to_noise"] >= 3) & (sp["signal_to_noise"] < 10)).sum() == 15
    assert len(win) and win["offset_da"].abs().max() <= 0.35 + S.WINDOW_PAD_DA
    rc, man, sp10, _ = run("sn10", "--noise-threshold", "10")
    assert man["parameters"]["noise_threshold"] == 10.0
    assert (sp10["signal_to_noise"] >= 10).all() and len(sp10) == len(sp) - 15
    rc, man, _, win = run("wide", "--iso-width", "2.0")
    assert man["parameters"]["isolation_width_used"] == 2.0
    assert man["parameters"]["isolation_width_source"] == "explicit"
    assert 0.6 < win["offset_da"].abs().max() <= 1.0 + S.WINDOW_PAD_DA
    rc, man, _, win = run("nowin", "--no-window")
    assert len(win) == 0 and man["counts"]["window_rows"] == 0
    assert "window inventory not asked for (--no-window)" in man["notes"]
    # a list in the wrong order passes with the rule check off (said in the manifest)
    with pytest.raises(SystemExit):
        cli.main(["ms2", "extract", "--file", FT, "--metadata-json", str(mj), "--polarity", "+",
                  "--inclusion-list", str(rev), "--out-dir", str(tmp_path / "refused")])
    rc, man, _, _ = run("norule", "--no-rule-check", listing=rev)
    assert rc == 0 and man["parameters"]["rule_check"] is False
    assert man["per_scan"]["mapping"]["rule"]["events_off"]


def test_a_listing_cut_short_is_an_error_never_a_silent_subset(monkeypatch):
    """The listing stops at `FILE_LIST_MAX_PAGES` full pages: what it read is
    not every file (ids are random, so the files left out are anywhere in the
    range) -- an error that says to narrow the range. A listing that ends
    with fewer files than the server counts is reported."""
    monkeypatch.setattr(IO, "FILE_LIST_PAGE", 2)
    monkeypatch.setattr(IO, "FILE_LIST_MAX_PAGES", 3)
    monkeypatch.setattr(IO, "time", type("_T", (), {"sleep": staticmethod(lambda s: None)}))
    rows = [{"sample_file_id": f"FILE{k:02d}xxxxxxxxxx", "datetime_utc": f"2026-09-11T{k:02d}:00"}
            for k in range(10)]
    total = {"n": 10}

    def paged(client, path, *, params=None, timeout=None, body=False):
        page = rows[params["page"] * params["limit"]:(params["page"] + 1) * params["limit"]]
        return {"results": total["n"], "data": page} if body else page
    monkeypatch.setattr(IO, "_api_get", paged)
    with pytest.raises(IO.FileListTruncated, match=r"narrow --since / --until"):
        IO.fetch_sample_files(object(), instrument="ORB", since="2026-09-11T00:00:00")
    del rows[5:]
    total["n"] = 6
    rep = {}
    out = IO.fetch_sample_files(object(), instrument="ORB", since="2026-09-11T00:00:00",
                                report=rep)
    assert len(out) == 5 and rep == {"server_total": 6, "files": 5, "pages": 3}


def test_cli_files_says_when_the_server_counts_more_files(server, capsys):
    server.listing_total = 7
    assert cli.main(["ms2", "files", "--instrument", "ORB", "--since", "2026-09-11"]) == 0
    assert ("the server counts 7 file(s) in this range and its pages held 5"
            in capsys.readouterr().out)


@pytest.mark.parametrize("name", ["", "   "])
def test_cli_files_refuses_an_empty_instrument(server, name):
    """An empty name would ask the listing with no instrument filter: every
    instrument's files. Refused before any request."""
    with pytest.raises(SystemExit, match="--instrument"):
        cli.main(["ms2", "files", "--instrument", name, "--since", "2026-09-11", "--check"])
    assert server.calls == []


def test_the_check_writes_the_listings_one_sign_into_the_command():
    """`--file` reads no file record: the listing's sign reaches the user only
    through the suggested command -- written when the listing has one sign
    and the filters name none, left out when the filters name it."""
    garbled = make_tms2()[0]
    row = S.file_row(_routine_rec(FT, polarity="-", processing_status="needs_chemistry"))
    chk = S.check_file(garbled, row)
    assert chk["command"].startswith(f"peaky ms2 extract --file {FT} --polarity - --inclusion")
    assert "the method's sign" not in chk["check_note"]
    row = S.file_row(_routine_rec(FT, polarity="+-", processing_status="needs_chemistry"))
    chk = S.check_file(make_tms2(garble=0.0, drops=())[0], row)
    assert chk["command"] == f"peaky ms2 extract --file {FT}"            # the filters name all
    assert "the method's sign" not in chk["check_note"]


def test_a_named_file_outside_the_listed_range_says_so(server, tmp_path, capsys):
    """`--check FILE_ID` for a file the listed range does not hold: checked,
    marked unlisted, and its note says no listing names its sign -- never
    \"the listing reads polarity ''\"."""
    server.listing = []
    out = tmp_path / "f.csv"
    assert cli.main(["ms2", "files", "--instrument", "ORB", "--since", "2026-09-11", "--check",
                     FT, "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "polarity ''" not in printed
    assert "not in the listed range: no listing names its sign" in printed
    df = pd.read_csv(out, keep_default_na=False).set_index("sample_file_id")
    assert df.loc[FT, "listed"] is False or df.loc[FT, "listed"] in (False, "False")
    assert df.loc[FT, "needs_check"] == "named with --check (not in the listed range)"
    assert df.loc[FT, "command"].startswith(f"peaky ms2 extract --file {FT} --polarity <+|->")


# ------------------------------------------------------------ review round 3 ----
def test_an_ordinary_file_given_a_list_reads_ms2_without_trailer(own):
    """Standard A (an ordinary MS2 file: its Full Scan is event 0, no MS2 scan
    has a trailer row) named with --file and a list by mistake: there is no
    MS2 event to fit the list to, so it is not used (said), and the file
    reads 'MS2 without trailer' as without one -- never a refusal counting -1
    MS2 events."""
    meta = own[0]
    for rows in ([137.1321, 155.1429, 155.1559], [137.1321]):
        res = S.extract_file(None, FA, metadata=meta,
                             inclusion_lists=[pd.DataFrame({"Precursor (m/z)": rows})])
        assert res.row["status"] == S.MS2_WITHOUT_TRAILER and not len(res.spectra)
        assert any("no MS2 scan has a trailer row: the inclusion list is not used" in n
                   for n in res.notes)


def _co_isolated_file(*, coiso=True, shift_ppm=0.0, key_form="C8H9O2+",
                      co_form="C3H9N2O4+"):
    """A one-row tMS2 file (garbled, NCE 5/15/30): the target `key_form` on its
    4-decimal key and, with `coiso`, `co_form` 4.0 mDa below it (-29 ppm),
    stronger than the target at NCE 5 (it fragments slower) and in the MS1
    scans; every m/z then scaled by (1 + `shift_ppm` 1e-6). -> (payload, the
    target's m/z, the co-isolate's, the list)."""
    from tests.test_ms2_scans import _res
    prec, co = AN.ion_mz(key_form), AN.ion_mz(co_form)
    key = round(prec, 4)
    meta, truth = make_tms2(prec=(key,), nces=(5.0, 15.0, 30.0),
                            cycles=((8, 0), (8, 0), (8, 0)), drops=(), split_scan=False,
                            auto_low_block=None)
    rng = np.random.default_rng(3)
    for c, t in zip(meta["centroids_meta"]["data"], truth.itertuples()):
        mz, inten = list(c["mzs"]), list(c["intensities"])
        j = int(np.argmin(np.abs(np.asarray(mz) - key)))
        mz[j] = prec * (1 + rng.normal(0, 0.3e-6))                # the target, exactly
        if coiso:
            mz.append(co * (1 + rng.normal(0, 0.3e-6)))
            inten.append(1.4e6 if t.order == 1 else 1.6e5 * math.exp(-t.nce / 40.0))
        o = np.argsort(mz)
        mz = np.asarray(mz)[o] * (1 + shift_ppm * 1e-6)
        c.update(mzs=mz.tolist(), intensities=np.asarray(inten)[o].tolist(),
                 resolutions=_res(mz).tolist(), noises=[100.0] * len(mz))
    return meta, prec, co, the_list((key,))


def test_one_parent_never_reads_its_co_isolate_as_the_axis_offset():
    """One parent, its window's strongest centroid at NCE 5 a co-isolate 29 ppm
    below the key -- while the target sits ON its key. A peak on the key is
    evidence against an axis shift: no offset is read off that parent (said),
    the precursor is the target at every energy, and `ms2 analyze` reads the
    target's breakdown with the co-isolate named as present. A one-parent file
    whose axis really sits 28 ppm off (nothing on its key) still reads it."""
    meta, prec, co, lst = _co_isolated_file()
    res = S.extract_file(None, FT, metadata=meta, inclusion_lists=[lst], polarity="+")
    ax = res.params["ms2_axis_offset"]
    assert ax["ppm"] is None and ax["contested_keys"] == 1
    assert any("hold a precursor candidate on their key" in n for n in res.notes)
    assert np.allclose(res.parents["precursor_mz"], prec, atol=2e-4)
    out = _analysed_with(res, "C8H9O2+", "C3H9N2O4+")
    w = out.windows[0]
    assert w.status == "analysed" and w.target == "C8H9O2+"
    e = _e50(out, round(prec, 4), "C8H9O2+")
    assert e["flag"] == "interpolated" and 10 < e["e50_nce"] < 17
    # a one-parent file 28 ppm off, no co-isolate: the offset is read (said: one parent)
    meta, prec, _, lst = _co_isolated_file(coiso=False, shift_ppm=-28.0)
    res = S.extract_file(None, FT, metadata=meta, inclusion_lists=[lst], polarity="+")
    ax = res.params["ms2_axis_offset"]
    assert ax["ppm"] == pytest.approx(-28.0, abs=0.5) and ax["contested_keys"] == 0
    assert ax["source"] == "MS2 precursors (only 1 parent(s) read one)"
    assert np.allclose(res.parents["precursor_mz"], prec * (1 - 28e-6), atol=2e-4)
    # the strongest ON its key, a 30 % neighbour 0.6 mDa away (also on the key,
    # resolved from it): nothing contests a reading on the key -- read
    meta, prec, _, lst = _co_isolated_file(coiso=False)
    for c in meta["centroids_meta"]["data"]:
        i = int(np.argmin(np.abs(np.asarray(c["mzs"]) - prec)))
        if abs(c["mzs"][i] - prec) < 1e-3:
            for k, v in (("mzs", c["mzs"][i] + 6e-4), ("intensities", 0.3 * c["intensities"][i]),
                         ("resolutions", c["resolutions"][i]), ("noises", c["noises"][i])):
                c[k].insert(i + 1, v)
    ax = S.extract_file(None, FT, metadata=meta, inclusion_lists=[lst],
                        polarity="+").params["ms2_axis_offset"]
    assert abs(ax["ppm"]) < 1.0 and ax["contested_keys"] == 0


def test_a_contested_reading_is_kept_and_said_on_its_parent():
    """A one-parent file 28 ppm off whose key holds an unrelated ion (15 % of
    the target at NCE 5, 30 % in MS1): the reading the target gives is
    contested by the ion on the key and dropped, so the search stays on the
    key and takes that ion. Nothing in one parent tells the two apart -- so
    the dropped reading is recorded (`contested`: key -> ppm), quoted in the
    extraction's note, and put on the parent's rows: if the axis sits there,
    the ion on the key is a co-isolate."""
    meta, prec, _, lst = _co_isolated_file(coiso=False, shift_ppm=-28.0)
    key = round(prec, 4)
    st = SC.scan_table(meta)
    for c, (order, nce) in zip(meta["centroids_meta"]["data"], zip(st["order"], st["nce"])):
        c["mzs"].append(key + 2e-5)
        c["intensities"].append(3e5 if order == 1 else 1.1e4)
        c["resolutions"].append(c["resolutions"][0])
        c["noises"].append(100.0)
        o = np.argsort(c["mzs"])
        for k in ("mzs", "intensities", "resolutions", "noises"):
            c[k] = [c[k][j] for j in o]
    res = S.extract_file(None, FT, metadata=meta, inclusion_lists=[lst], polarity="+")
    ax = res.params["ms2_axis_offset"]
    assert ax["ppm"] is None and ax["contested_keys"] == 1
    assert list(ax["contested"]) == [repr(key)]
    assert ax["contested"][repr(key)] == pytest.approx(-28.0, abs=0.5)
    note = next(n for n in res.notes if n.startswith("the m/z axis offset is not measured"))
    assert re.search(r"reads -2[78]\.\d ppm", note)
    assert np.allclose(res.parents["precursor_mz"], key, atol=2e-4)        # the ion on the key
    for n in res.parents["note"]:
        assert re.search(r"axis-offset reading contested: .* -2[78]\.\d ppm off its key", n), n
        assert "the ion on the key is a co-isolate" in n


def _two_parent_file(shift_ppm):
    """Two parents (C8H9O2+, C9H15O+; 4-decimal keys), the first's window
    holding C3H9N2O4+ 4.0 mDa below the target, stronger at NCE 5 and in the
    MS1 scans; every m/z scaled by (1 + `shift_ppm` 1e-6). -> (payload, the
    two targets' m/z, the list)."""
    from tests.test_ms2_scans import _res
    p1, p2, co = AN.ion_mz("C8H9O2+"), AN.ion_mz("C9H15O+"), AN.ion_mz("C3H9N2O4+")
    keys = (round(p1, 4), round(p2, 4))
    meta, truth = make_tms2(prec=keys, nces=(5.0, 15.0, 30.0), cycles=((8, 0), (8, 0), (8, 0)),
                            drops=(), split_scan=False, auto_low_block=None)
    rng = np.random.default_rng(3)
    for c, t in zip(meta["centroids_meta"]["data"], truth.itertuples()):
        mz, inten = list(c["mzs"]), list(c["intensities"])
        for k, pr in zip(keys, (p1, p2)):
            j = int(np.argmin(np.abs(np.asarray(mz) - k)))
            if abs(mz[j] - k) < 1e-3:
                mz[j] = pr * (1 + rng.normal(0, 0.3e-6))
        if t.order == 1 or abs(t.parent_mz - keys[0]) < 1e-6:
            mz.append(co * (1 + rng.normal(0, 0.3e-6)))
            inten.append(1.4e6 if t.order == 1 else 1.6e5 * math.exp(-t.nce / 40.0))
        o = np.argsort(mz)
        mz = np.asarray(mz)[o] * (1 + shift_ppm * 1e-6)
        c.update(mzs=mz.tolist(), intensities=np.asarray(inten)[o].tolist(),
                 resolutions=_res(mz).tolist(), noises=[100.0] * len(mz))
    return meta, (p1, p2), the_list(keys)


@pytest.mark.parametrize("shift", [-12.0, -6.0, 0.0])
def test_two_parents_never_average_a_co_isolate_into_the_offset(shift):
    """Two parents, the axis `shift` ppm off; the first parent's window holds
    a co-isolate 4 mDa below its target that outshines it at NCE 5. Off the
    axis the target is not on its key, so nothing contests the co-isolate's
    reading -- and the median of two is their mean, wrong for both. A
    reading is kept only where the other parent's window bears it out (a
    candidate at its key x (1 + that reading)): the co-isolate's is dropped
    (recorded, said), the offset is the other parent's, and `ms2 analyze`
    reads both windows."""
    meta, precs, lst = _two_parent_file(shift)
    res = S.extract_file(None, FT, metadata=meta, inclusion_lists=[lst], polarity="+")
    ax = res.params["ms2_axis_offset"]
    assert ax["ppm"] == pytest.approx(shift, abs=0.8), ax
    first = res.parents.sort_values("nce").drop_duplicates("parent_mz")
    assert np.allclose(sorted(first["precursor_mz"]), np.asarray(precs) * (1 + shift * 1e-6),
                       atol=3e-4)
    out = _analysed_with(res, "C8H9O2+", "C9H15O+", "C3H9N2O4+")
    assert [w.status for w in out.windows] == ["analysed"] * 2
    assert [w.target for w in out.windows] == ["C8H9O2+", "C9H15O+"]
    if shift:
        assert list(ax["unconfirmed"]) == [repr(round(precs[0], 4))]
        assert "1 reading(s) no other parent's window bears out" in ax["source"]


#: two list rows 3.5 mDa apart (their 0.7 Da windows one window), the second's
#: ion 20x the first's in both rows' scans; three more fine keys read the axis
CLOSE = (120.0808, 137.1325, 155.1431, 155.1466, 181.0859)


def _close_rows_file(shift_ppm=-28.0, ratio=20.0):
    """A tMS2 file whose rows 3 and 4 (`CLOSE`) co-isolate each other: each
    row's scans hold both ions, row 4's `ratio` x row 3's; every m/z then
    scaled by (1 + `shift_ppm` 1e-6). -> (payload, truth, the list)."""
    meta, truth = make_tms2(prec=CLOSE, drops=(), auto_low_block=None)
    a, b = CLOSE[2], CLOSE[3]
    for c, t in zip(meta["centroids_meta"]["data"], truth.itertuples()):
        if t.order != 2 or not (math.isclose(t.parent_mz, a) or math.isclose(t.parent_mz, b)):
            continue
        own = int(np.argmin(np.abs(np.asarray(c["mzs"]) - t.parent_mz)))
        other, f = (b, ratio) if math.isclose(t.parent_mz, a) else (a, 1.0 / ratio)
        c["mzs"].append(other)
        for k, v in (("intensities", f * c["intensities"][own]),
                     ("resolutions", c["resolutions"][own]), ("noises", c["noises"][own])):
            c[k].append(v)
        o = np.argsort(c["mzs"])
        for k in ("mzs", "intensities", "resolutions", "noises"):
            c[k] = [c[k][j] for j in o]
    return shift_axis(meta, truth, shift_ppm), truth, the_list(CLOSE)


def test_a_measured_offset_finds_each_rows_own_ion_before_a_brighter_neighbour():
    """Two list rows 3.5 mDa apart whose windows are one window, the second's
    ion 20x the first's, the axis 28 ppm off (measured: MAD a fraction of a
    ppm). Centred on its corrected key, the first row's own ion is there --
    it is the precursor, never the neighbour row's ion 3.5 mDa away (the wide
    search's 10 % share rule would drop the weak one). The key-offset note is
    read against the corrected key (none here), and no two parents share a
    located precursor."""
    meta, _, lst = _close_rows_file()
    res = S.extract_file(None, FT, metadata=meta, inclusion_lists=[lst], polarity="+")
    ax = res.params["ms2_axis_offset"]
    assert ax["ppm"] == pytest.approx(-28.0, abs=0.5) and ax["mad_ppm"] < 2.0
    par = res.parents.set_index("spectrum")
    for p in CLOSE:
        rows = res.parents[np.isclose(res.parents["parent_mz"], p)]
        assert np.allclose(rows["precursor_mz"], p * (1 - 28e-6), atol=3e-4), (p, rows)
        assert not rows["note"].str.contains("precursor measured at").any()
        assert not rows["note"].str.contains("coincides with").any()
    assert len(par) == len(CLOSE) * 3


def test_two_parents_that_locate_one_ion_are_said():
    """A server folder (no offset) whose two keys, 3.5 mDa apart, sit in one
    window holding one bright ion near the second and a weak one on the
    first: the wide search gives both parents the bright ion -- said on
    both, never two parents silently reading one ion's breakdown."""
    m1, m2 = 155.1431, 155.1466
    cen, groups = {}, []
    for k, (key, nce) in enumerate([(m1, 5.0), (m2, 5.0)]):
        act = f"hcd{nce:.2f}"
        mz = [key - 18.010565, m1, m2]
        it = [500.0, 300.0, 6000.0]
        cen[S.spectrum_key(key, act)] = {
            "parent_peak_mz": key, "activation": act, "mz": mz, "intensity": it,
            "resolution": [120000 * math.sqrt(200 / x) for x in mz], "signal_to_noise": [50] * 3}
        groups.append({"parent_peak_mz": key, "activation": act, "hcd_energy": [5.0],
                       "scan_count": 5, "t_min": 10.0 * k, "t_max": 10.0 * k + 8})
    summary = {"parent_peaks": [m1, m2], "groups": groups, "isolation_width": 0.7}
    par = S.parents_table("S", cen, summary)
    assert np.allclose(par["precursor_mz"], m2)
    for p, other in ((m1, m2), (m2, m1)):
        n = par.loc[np.isclose(par["parent_mz"], p), "note"].iloc[0]
        assert f"located precursor coincides with parent {other:.4f}'s" in n, n


def test_the_ms1_window_marks_the_precursor_the_ms2_located_or_none(off_axis):
    """28 ppm low (measured). One parent's MS1 scans lack its precursor and hold
    a weak centroid 2.2 mDa above it (inside the wide search, 12 ppm off the
    measured axis): `is_parent` is the ion the MS2 spectra located (within the
    track tolerance), so no centroid of that window is marked -- said on the
    parent's rows. The MS1 scans' m/z range and the MS1 / MS2 time spans are
    recorded; a parent whose window lies past the MS1 scans' range is said to
    be not measured, never silently absent."""
    _, low, truth, lst = off_axis
    p = FPREC[2]
    odd = json.loads(json.dumps(low))
    for s_ in truth.loc[truth["order"] == 1, "scan"]:
        c = odd["centroids_meta"]["data"][s_ - 1]
        keep = [i for i, m in enumerate(c["mzs"]) if abs(m - p) > 0.4]
        for k in ("mzs", "intensities", "resolutions", "noises"):
            c[k] = [c[k][i] for i in keep]
        c["mzs"].append(p * (1 - 28e-6) + 2.2e-3)
        for k, v in (("intensities", 2000.0), ("resolutions", 1e5), ("noises", 1000.0)):
            c[k].append(v)
        o = np.argsort(c["mzs"])
        for k in ("mzs", "intensities", "resolutions", "noises"):
            c[k] = [c[k][j] for j in o]
    res = S.extract_file(None, FT, metadata=odd, inclusion_lists=[lst], polarity="+")
    assert res.params["ms2_axis_offset"]["ppm"] == pytest.approx(-28.0, abs=0.5)
    w = res.window[np.isclose(res.window["parent_mz"], p)]
    assert len(w) and not w["is_parent"].any()
    others = res.window[~np.isclose(res.window["parent_mz"], p) & res.window["is_parent"]]
    assert sorted(others["parent_mz"]) == sorted(q for q in FPREC if q != p)
    rows = res.parents[np.isclose(res.parents["parent_mz"], p)]
    assert rows["note"].str.contains("the MS1 window holds no centroid at this parent's "
                                     "precursor").all()
    per = res.per_scan
    assert per["ms1_scan_range"] == [40.0, 600.0]
    assert per["ms1_time_s"][1] < per["ms2_time_s"][0]
    assert any(n.startswith("the MS1 scans (4, t 3.0-3.9 s) all precede the MS2 scans")
               for n in res.notes)
    # the MS1 scans recorded m/z 40-190: the two heavier parents are not measured there
    narrow = json.loads(json.dumps(low))
    for s_ in truth.loc[truth["order"] == 1, "scan"]:
        r = narrow["stats_per_scan"][str(s_)]
        r["HighMass"] = 190.0
        r["ScanType"] = r["ScanType"].replace("-600.0000]", "-190.0000]")
        c = narrow["centroids_meta"]["data"][s_ - 1]
        keep = [i for i, m in enumerate(c["mzs"]) if m <= 190.0]
        for k in ("mzs", "intensities", "resolutions", "noises"):
            c[k] = [c[k][i] for i in keep]
    res = S.extract_file(None, FT, metadata=narrow, inclusion_lists=[lst], polarity="+")
    for q in FPREC:
        n = res.parents.loc[np.isclose(res.parents["parent_mz"], q), "note"]
        outside = n.str.contains(r"outside the MS1 scans' m/z range \(40\.00-190\.00\)")
        assert outside.all() if q > 190 else not outside.any(), q
        assert (q > 190) == (not len(res.window[np.isclose(res.window["parent_mz"], q)]))


def _analysed_with(res, *forms):
    man = json.loads(json.dumps(S.manifest(res), default=str))
    tables = AN.RunTables(spectra=res.spectra, parents=res.parents, window=res.window,
                          manifest=man)
    return AN.analyze_run(tables, _ions(*forms))


def test_a_file_with_no_ms2_parent_averages_no_ms1_scan(own, monkeypatch):
    """An ordinary MS2 file named with --file ('MS2 without trailer'): no
    parent, so no window and no axis offset to read -- its MS1 scans are
    never averaged (an hour of them is a clusters x scans matrix of GB)."""
    calls = []
    real = SC.average_scans
    monkeypatch.setattr(SC, "average_scans",
                        lambda m, s, **kw: calls.append(len(list(s))) or real(m, s, **kw))
    for with_window in (False, True):
        res = S.extract_file(None, FA, metadata=own[0], with_window=with_window)
        assert res.row["status"] == S.MS2_WITHOUT_TRAILER
    assert calls == []


def test_the_help_says_what_a_metadata_read_costs_and_which_files_are_read(capsys):
    """One statement of a `/metadata` read's size wherever it is read (the
    --per-scan and --max-check help, docs §12): about 0.1 MB per MS1 scan --
    hundreds of MB for an hour of them, never 'a few MB'. And `--check` reads
    a file it names whatever the listing says: 'never fetched' holds only for
    the files it picks itself (and --max-check caps both)."""
    for argv in (["ms2", "extract", "--help"], ["ms2", "files", "--help"]):
        with pytest.raises(SystemExit):
            cli.main(argv)
    text = " ".join(capsys.readouterr().out.split())
    cost = " ".join(cli.MS2_METADATA_COST.split())
    assert "hundreds of MB for an hour of MS1 scans" in cost and "a few MB" not in text
    assert text.count(cost) == 2                                 # --per-scan, --max-check
    assert "Routine files are never fetched" not in text
    assert "a file named here is read whatever the listing says" in text
    assert "named ones included" in text
    root = Path(__file__).resolve().parent.parent
    docs = (root / "docs" / "MS2.md").read_text(encoding="utf-8")
    sec = " ".join(docs[docs.index("## 12."):].split())
    assert "a few to tens of MB" not in sec and "routine files are never fetched;" not in sec
    assert sec.count("about 0.1 MB per MS1 scan") >= 2
    assert "a file it names is read whatever its listing says" in sec
    doc = " ".join(S.needs_check.__doc__.split())
    assert "reads '' and is never fetched" not in doc
    assert "a file it names (`--check ID`) is read whatever this says" in doc


def test_the_manifest_names_the_per_scan_modules_version(own):
    """The per-scan core (scans.py) maps and averages every --file and
    --per-scan folder: its version is in the code identity beside sources'
    and io_mascope's (a plain install has no git to say which code ran)."""
    res = S.extract_file(None, FT, metadata=make_tms2()[0], inclusion_lists=[the_list()],
                         polarity="+")
    code = S.manifest(res)["code"]
    assert code["scans_version"] == SC.__version__
    assert code["sources_version"] == S.__version__ and code["io_mascope_version"] == IO.__version__


def test_the_scan_peaks_hold_every_scan_of_a_kept_peak(wrapped):
    """`ms2_scan_peaks` of a --file folder: every kept peak has a row for every
    scan of its spectrum, 0 where the scan has none (65 of wrapped standard
    A's 230 peaks miss some scan), so the mean over its rows IS the
    spectrum's intensity -- the same 1517 rows the server's peaks read."""
    payload, lists, _, _ = wrapped
    res = S.extract_file(None, FT, metadata=payload, inclusion_lists=lists, polarity="+")
    pk, sp = res.scan_peaks, res.spectra
    count = dict(zip(res.parents["spectrum"], res.parents["scan_count"]))
    g = pk.groupby(["spectrum", "peak_mz"])["intensity"]
    rows = g.size()
    assert len(rows) == len(sp) == 230 and len(pk) == 1517
    assert (rows.to_numpy() == rows.index.get_level_values(0).map(count).to_numpy()).all()
    assert int((g.apply(lambda v: int((v == 0).sum())) > 0).sum()) == 65
    want = sp.set_index(["spectrum", "mz"])["intensity"].reindex(g.mean().index)
    assert np.allclose(g.mean(), want, rtol=1e-12, atol=0)


def test_a_measured_offset_keeps_the_side_lobe_guard_off_on_a_file():
    """An on-axis file whose 4-decimal keys read its offset (~0 ppm): centred
    there, the precursor lies in its search window, so the side-lobe guard is
    off. 155.1430 on its key beside a co-isolate 100x brighter 4.8 line widths
    above it (just outside the window; in its MS2 and MS1 scans): located at
    every energy and the MS1 window's parent -- the guard would refuse it as
    that line's lobe (as it does with the guard on)."""
    meta, truth = make_tms2(drops=())
    p, q = PREC[2], PREC[3]

    def lobe(c, x):
        i = int(np.argmin(np.abs(np.asarray(c["mzs"]) - x)))
        c["mzs"].append(x + 4.8 * x / c["resolutions"][i])
        for k, v in (("intensities", 100 * c["intensities"][i]),
                     ("resolutions", c["resolutions"][i]), ("noises", c["noises"][i])):
            c[k].append(v)
        o = np.argsort(c["mzs"])
        for k in ("mzs", "intensities", "resolutions", "noises"):
            c[k] = [c[k][j] for j in o]
    for t, c in zip(truth.itertuples(), meta["centroids_meta"]["data"]):
        if t.order == 1 or math.isclose(t.parent_mz, p):
            lobe(c, p)
        if t.order == 1:
            lobe(c, q)
        elif math.isclose(t.parent_mz, q):
            # q's MS2 scans lose its precursor: the MS1 window is searched for it
            keep = [i for i, m in enumerate(c["mzs"]) if abs(m - q) > 0.4]
            for k in ("mzs", "intensities", "resolutions", "noises"):
                c[k] = [c[k][i] for i in keep]
    for iso in (None, 0.7):          # the trailer's widths, and --iso-width's
        res = S.extract_file(None, FT, metadata=meta, inclusion_lists=[the_list()], polarity="+",
                             iso_width=iso)
        assert abs(res.params["ms2_axis_offset"]["ppm"]) < 0.5
        assert res.params["precursor_lobe"]["guard_on"] is False
        par = res.parents[np.isclose(res.parents["parent_mz"], p)]
        assert len(par) == 3 and np.allclose(par["precursor_mz"], p, atol=2e-4)
        assert not par["note"].str.contains("ambiguous").any()
        w = res.window[np.isclose(res.window["parent_mz"], p)]
        assert w.loc[w["is_parent"], "mz"].tolist() == pytest.approx([p], abs=2e-4), iso
        # q, located in no MS2 spectrum: its MS1 window is searched with the
        # guard off -- its weak precursor beside the 100x lobe is the parent
        assert res.parents.loc[np.isclose(res.parents["parent_mz"], q), "precursor_mz"].isna().all()
        w = res.window[np.isclose(res.window["parent_mz"], q)]
        assert w.loc[w["is_parent"], "mz"].tolist() == pytest.approx([q], abs=2e-4), iso
    # the guard on, the same MS1 window refuses q's precursor as that lobe's
    ms1 = SC.extract_payload("S", meta, lists=[SC.inclusion_list_frame(the_list())]).ms1
    for guard, want in ((True, []), (False, [q])):
        w = S.window_table("S", ms1, [q], 0.7, lobe_guard=guard, measured=True)
        assert w.loc[w["is_parent"], "mz"].tolist() == pytest.approx(want, abs=2e-4), guard
    cen = {k: {"parent_peak_mz": float(g["parent_mz"].iloc[0]),
               "activation": g["activation"].iloc[0],
               "mz": g["mz"].tolist(), "intensity": g["intensity"].tolist(),
               "resolution": g["resolution"].tolist()}
           for k, g in res.spectra.groupby("spectrum")}
    on = S.parents_table("S", cen, None, lobe_guard=True)
    assert on.loc[np.isclose(on["parent_mz"], p), "precursor_mz"].isna().all()


def _drop_window(meta, truth, p, nce):
    """`meta` with every centroid of `p`'s isolation window removed from its
    scans at `nce` (that parent's lowest-energy window empty)."""
    out = json.loads(json.dumps(meta))
    for s in truth.loc[np.isclose(truth["parent_mz"], p) & (truth["nce"] == nce), "scan"]:
        c = out["centroids_meta"]["data"][s - 1]
        keep = [i for i, m in enumerate(c["mzs"]) if abs(m - p) > 0.4]
        for k in ("mzs", "intensities", "resolutions", "noises"):
            c[k] = [c[k][i] for i in keep]
    return out


def test_the_axis_offset_out_votes_one_parent_and_reads_the_ms1_window_for_few(off_axis):
    """28 ppm low. (a) One of the three parents that read the offset has an
    ion 3x its precursor off its key at its lowest energy: it reads that ion,
    and the MEDIAN out-votes it -- placed on the lowest key reading +20 ppm
    (above the others: a mean would read -12 ppm and leave every precursor
    16 ppm from its formula), and on the highest, finest-typed key reading
    -48 ppm (below them: a minimum, the last parent's or the largest reading
    would take it). (b) One parent's lowest-energy window holds nothing: 2
    MS2 parents read it, fewer than `AXIS_MIN_PARENTS` -- the MS1 window's 3
    parents name it."""
    _, low, truth, lst = off_axis
    fine = [p for p in FPREC if S.key_rounding_ppm(p) <= S.AXIS_KEY_PPM]
    assert len(fine) == S.AXIS_MIN_PARENTS == 3
    for p, ppm in ((fine[0], 20.0), (fine[-1], -48.0)):
        odd = json.loads(json.dumps(low))
        for s in truth.loc[np.isclose(truth["parent_mz"], p) & (truth["nce"] == 5.0), "scan"]:
            c = odd["centroids_meta"]["data"][s - 1]
            i = int(np.argmin(np.abs(np.asarray(c["mzs"]) - p * (1 - 28e-6))))
            for k, v in (("mzs", p * (1 + ppm * 1e-6)), ("intensities", 3 * c["intensities"][i]),
                         ("resolutions", c["resolutions"][i]), ("noises", c["noises"][i])):
                c[k].append(v)
        res = S.extract_file(None, FT, metadata=odd, inclusion_lists=[lst], polarity="+")
        ax = res.params["ms2_axis_offset"]
        assert ax["source"] == "MS2 precursors" and ax["parents"] == 3
        assert ax["per_parent"][repr(p)] == pytest.approx(ppm, abs=0.5)     # read, not contested
        assert ax["ppm"] == pytest.approx(-28.0, abs=1.0), (p, ppm)
        assert [w.status for w in _analysed(res).windows] == ["analysed"] * 5
    res = S.extract_file(None, FT, metadata=_drop_window(low, truth, fine[1], 5.0),
                         inclusion_lists=[lst], polarity="+")
    ax = res.params["ms2_axis_offset"]
    assert ax["source"] == "MS1 window" and ax["parents"] == 3
    assert ax["ppm"] == pytest.approx(-28.0, abs=0.5)


def test_the_file_window_keeps_every_ms1_cluster(wrapped):
    """The MS1 average of a --file folder applies no S/N floor (as the server's
    MS1 centroids: 95 of standard A's 209 read S/N < 3): wrapped standard A's
    window holds 114 rows, 34 of them below S/N 3 -- the weak co-isolates
    the inventory is there to list."""
    payload, lists, _, _ = wrapped
    res = S.extract_file(None, FT, metadata=payload, inclusion_lists=lists, polarity="+")
    w = res.window
    assert len(w) == 114 and int((w["signal_to_noise"] < 3).sum()) == 34


def test_a_negative_file_is_extracted_as_negative(tmp_path, capsys):
    """`--file --polarity -` on a garbled file: the folder says '-' in its
    manifest, its parameters and the console -- and a file whose readable
    filters carry both signs needs the sign given, which it then keeps."""
    meta, _ = make_tms2()
    mj, lst = tmp_path / "m.json", tmp_path / "list.csv"
    mj.write_text(json.dumps(meta))
    the_list().to_csv(lst, index=False)
    assert cli.main(["ms2", "extract", "--file", FT, "--metadata-json", str(mj), "--polarity", "-",
                     "--inclusion-list", str(lst), "--out-dir", str(tmp_path)]) == 0
    man = json.loads((tmp_path / "ms2" / FT / "ms2_manifest.json").read_text())
    assert man["polarity"] == "-" and man["parameters"]["polarity"] == "-"
    assert man["parameters"]["polarity_source"] == "explicit"
    assert "; polarity - (explicit);" in capsys.readouterr().out


def _signed(meta, sign, *, every=1):
    """The payload with every `every`-th trailer row's filter set to `sign`."""
    for k, r in meta["stats_per_scan"].items():
        if int(k) % every == 0:
            r["ScanType"] = r["ScanType"].replace("FTMS + p", f"FTMS {sign} p")
    return meta


def test_a_readable_negative_file_reads_its_sign_off_the_filters(tmp_path, capsys):
    """A clean negative-mode file (every filter readable, all '-'): the filters
    name '-'. `extract_file` without a polarity and `--file --metadata-json`
    without --polarity record ('-', 'scan filters') in the row, the manifest
    and the console; the check asks for no sign. A garbled file whose
    one-sign file record reads '-' takes it ('file record')."""
    neg = _signed(make_tms2(garble=0.0)[0], "-")
    assert SC.filter_polarity(SC.scan_table(neg)) == "-"
    res = S.extract_file(None, FT, metadata=neg, inclusion_lists=[the_list()])
    assert (res.row["polarity"], res.params["polarity_source"]) == ("-", "scan filters")
    mj, lst = tmp_path / "m.json", tmp_path / "list.csv"
    mj.write_text(json.dumps(neg))
    the_list().to_csv(lst, index=False)
    assert cli.main(["ms2", "extract", "--file", FT, "--metadata-json", str(mj),
                     "--inclusion-list", str(lst), "--out-dir", str(tmp_path)]) == 0
    man = json.loads((tmp_path / "ms2" / FT / "ms2_manifest.json").read_text())
    assert (man["polarity"], man["parameters"]["polarity"]) == ("-", "-")
    assert man["parameters"]["polarity_source"] == "scan filters"
    assert "; polarity - (scan filters);" in capsys.readouterr().out
    chk = S.check_file(neg, {"sample_file_id": FT, "polarity": "+-"})
    assert "--polarity" not in chk["command"]
    res = S.extract_file(None, FT, metadata=make_tms2()[0], inclusion_lists=[the_list()],
                         file_row={"polarity": "-"})
    assert (res.row["polarity"], res.params["polarity_source"]) == ("-", "file record")


def test_readable_filters_of_both_signs_are_refused_not_mixed(tmp_path, capsys):
    """Every filter agrees with its trailer, but they carry '+' and '-' -- a
    dual-polarity method with MS2 scans read by trailer. The spectra would
    average every MS2 scan and the window every MS1 scan whatever their sign,
    so a folder labelled with one sign would hold the other's scans: refused
    whatever --polarity says (no sign splits it), and the check suggests no
    command. A file whose MS2 scans have no trailer row (an ordinary
    dual-polarity file) is not refused here."""
    both = _signed(make_tms2(garble=0.0)[0], "-", every=2)
    v = SC.filter_verdict(SC.scan_table(both))
    assert v["filters_readable"] == SC.FILTERS_YES and v["polarities_in_filters"] == "+-"
    for sign in (None, "-", "+"):
        with pytest.raises(S.DualPolarityError, match=r"carry both signs \('\+-'\)"):
            S.extract_file(None, FT, metadata=both, inclusion_lists=[the_list()], polarity=sign)
    chk = S.check_file(both, {"sample_file_id": FT, "polarity": "+-"})
    assert chk["command"] == "" and "carry both signs" in chk["check_note"]
    mj = tmp_path / "m.json"
    mj.write_text(json.dumps(both))
    with pytest.raises(SystemExit, match="1 of 1 raw file"):
        cli.main(["ms2", "extract", "--file", FT, "--metadata-json", str(mj), "--polarity", "-",
                  "--out-dir", str(tmp_path)])
    assert "carry both signs" in capsys.readouterr().out
    ordinary, _ = make_tms2(garble=0.0, blocks=[], drops=())
    ordinary = _signed(ordinary, "-", every=2)
    data = ordinary["centroids_meta"]
    data["time"].append(data["time"][-1] + 1.0)
    data["data"].append(json.loads(json.dumps(data["data"][-1])))
    ordinary["num_of_scans"] += 1
    res = S.extract_file(None, FT, metadata=ordinary, polarity="-")
    assert res.row["status"] == S.MS2_WITHOUT_TRAILER


def test_a_file_with_no_ms2_or_none_mapped_reads_so():
    """An MS1-only file reads 'no MS2'; a file whose MS2 trailer rows name no
    scan event reads 'MS2 unmapped' -- never 'ok' with nothing in it."""
    ms1_only, _ = make_tms2(blocks=[], drops=())
    res = S.extract_file(None, FT, metadata=ms1_only, polarity="+")
    assert (res.row["status"], res.row["note"]) == ("no MS2", "no scan reads MS2 off its trailer")
    assert res.notes == ["window inventory skipped: no MS2 parent"]
    assert S.check_file(ms1_only, {"sample_file_id": FT, "polarity": "+"})["check_note"] == (
        "no MS2 scan")
    no_event, _ = make_tms2(drops=())
    for r in no_event["stats_per_scan"].values():
        if r["Scan Node:"] == "tMS2":
            del r["Scan Event:"]
    for lists in (None, [the_list()]):
        res = S.extract_file(None, FT, metadata=no_event, polarity="+", inclusion_lists=lists)
        assert (res.row["status"], res.row["note"]) == (
            "MS2 unmapped", "no MS2 scan could be mapped to a precursor and an energy")
        assert not len(res.spectra)
        assert res.per_scan["mapping"]["unmapped_ms2_scans"]
    assert "no MS2 trailer row names a scan event: the inclusion list is not used" in res.notes[0]


def test_a_file_with_no_full_scan_says_its_window_is_skipped():
    """MS2 scans with trailers and no Full Scan at all: the spectra are built,
    and the window inventory is skipped -- said, never an empty table unsaid."""
    no_fs, _ = make_tms2(fs_scans=0, drops=())
    res = S.extract_file(None, FT, metadata=no_fs, polarity="+", inclusion_lists=[the_list()])
    assert res.row["status"] == "ok" and len(res.spectra) and not len(res.window)
    assert "window inventory skipped: the file has no MS1 scan" in res.notes


def test_the_check_says_once_that_no_list_pins_the_layout():
    """A garbled file with headerless scans and no list: the check note says
    once that the experiments' event ranges are not pinned (the cycle walk's
    note says why), and once what to do -- never the same sentence twice."""
    chk = S.check_file(make_tms2()[0], {"sample_file_id": FT, "polarity": "+"})
    note = chk["check_note"]
    assert note.count("the experiments' event ranges are not pinned") == 1, note
    assert "pass the method's list: without it, headerless scans at block edges and on a wrap " \
           "stay unassigned" in note
    assert chk["command"].endswith(" --inclusion-list <the method's list.csv>")


def test_the_check_advises_the_list_for_rule_derived_precursors():
    """A garbled file with every trailer row and no list: the scan range names
    every precursor, so the list is advised (it checks them), not required."""
    chk = S.check_file(make_tms2(drops=())[0], {"sample_file_id": FT, "polarity": "+"})
    assert chk["command"] == (f"peaky ms2 extract --file {FT} --polarity + "
                              "[--inclusion-list <the method's list.csv>]")
    assert "precursors from the scan-range rule: the method's list checks them" in chk["check_note"]


def test_the_console_says_one_scan_range_for_every_event(tmp_path, capsys):
    """A one-row list over three experiments: every event reads one HighMass,
    so the rule says nothing -- the console says so."""
    meta, _ = make_tms2(prec=(PREC[1],), drops=(), auto_low_block=None)
    mj, lst = tmp_path / "m.json", tmp_path / "list.csv"
    mj.write_text(json.dumps(meta))
    the_list((PREC[1],)).to_csv(lst, index=False)
    assert cli.main(["ms2", "extract", "--file", FT, "--metadata-json", str(mj), "--polarity", "+",
                     "--inclusion-list", str(lst), "--out-dir", str(tmp_path)]) == 0
    printed = capsys.readouterr().out
    assert "precursors: inclusion list 12; one scan range for every event" in printed
