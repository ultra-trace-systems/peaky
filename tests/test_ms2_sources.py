"""Offline tests for the MS2 source (peaky/ms2/sources.py), its io_mascope
fetchers and the `peaky ms2 list|extract` commands. No network: the client is a
fake that serves captured payloads.

The fixtures (tests/fixtures/ms2) are our own lab's server answers, captured
read-only and trimmed (MS1 centroids within 2 Da of a parent; every MS2
spectrum kept), with placeholder ids and a placeholder host:

  * ms2_nh4_exploratory_* -- the 2026-09-10 exploratory 15N-ammonium file:
    29 (parent, activation) spectra over 7 parents;
  * ms2_monoterpene_a_* -- monoterpene standard A, 2026-09-11: 12 spectra over
    3 parents;
  * ms2_monoterpene_b_* -- monoterpene standard B, 2026-09-11: get_summary
    refused (HTTP 400, multiple isolation widths) while the centroids served
    six activations of its one parent, keyed 137.13; its listing row runs
    210.5 s past t1.

Also real: ms2_monoterpene_a_centroids_averaged.json -- standard A's centroids
asked WITHOUT by_activation on the same >= 1.8 server (one blended spectrum per
parent, 'activation' '').

Synthetic: ms2_old_server_* (a server older than Mascope 1.8: standard A's
energy-averaged answer without the 'activation' / 'parent_peak_mz' fields, and
its summary in the pre-1.8 shape -- no groups, and an `hcd_energy_map` holding
the scan-weighted MEAN volt of each parent, as Mascope computed it before 1.8:
5.16 / 4.92 / 6.07 V from standard A's groups) and
ms2_empty_scaffold_summary.json (an MS1-only sample: every key present,
ms2_scan_count 0)."""
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from mascope_sdk.exceptions import (AuthenticationError, MascopeAPIError, MascopeConnectionError,
                                    MascopeTimeoutError, NotFoundError, ServerError,
                                    ValidationError)

from peaky import cli
from peaky.io import io_mascope as IO
from peaky.ms2 import assign_fragments, best_readings, mass_offset
from peaky.ms2 import sources as S

FIX = Path(__file__).resolve().parent / "fixtures" / "ms2"

NH4, MTA, MTB, OLD, EMPTY, BAD, TOF, WAF = (f"SAMPLE{i}xxxxxxxxx" for i in range(1, 9))
#: a sample the API answers its own 403 for (an unknown or stale id), and one
#: whose request never reaches the server (a DNS failure, after the SDK's retries)
STALE, DOWN = "SAMPLE9xxxxxxxxx", "SAMPLExxxxxxxxxx"
REASON = "Multiple isolation widths found for MS2 scans"
#: What the fake server and the environment carry and nothing written may hold.
SENTINEL_URL = "https://sentinel-host.invalid"
SENTINEL_TOKEN = "TOKSENTINEL0123456789"

#: The table contracts, spelled out here (not read back from the module).
SPECTRA_DTYPES = {
    "sample_id": "str", "spectrum": "str", "parent_mz": "float64", "activation": "str",
    "kind": "str", "nce": "float64", "hcd_v": "float64", "hcd_v_values": "str",
    "scan_count": "float64", "t_min": "float64", "t_max": "float64",
    "energy_resolved": "bool", "mz": "float64", "intensity": "float64",
    "resolution": "float64", "signal_to_noise": "float64"}
PARENTS_DTYPES = {
    "sample_id": "str", "spectrum": "str", "parent_mz": "float64", "activation": "str",
    "kind": "str", "nce": "float64", "hcd_v": "float64", "hcd_v_values": "str",
    "scan_count": "float64", "t_min": "float64", "t_max": "float64",
    "isolation_width": "float64", "n_peaks": "int64", "tic": "float64",
    "precursor_mz": "float64", "precursor_intensity": "float64",
    "energy_resolved": "bool", "note": "str"}
WINDOW_DTYPES = {
    "sample_id": "str", "parent_mz": "float64", "mz": "float64", "offset_da": "float64",
    "intensity": "float64", "share": "float64", "resolution": "float64",
    "signal_to_noise": "float64", "is_parent": "bool"}
INVENTORY_DTYPES = {
    "sample_id": "str", "name": "str", "datetime_utc": "str", "polarity": "str",
    "ms1_scans": "float64", "ms2_scans": "float64", "n_parents": "int64", "parents": "str",
    "n_spectra": "int64", "nce": "str", "hcd_v_min": "float64", "hcd_v_max": "float64",
    "isolation_width": "float64", "energy_resolved": "bool", "hidden_tail_s": "float64",
    "status": "str", "note": "str"}


def _dtypes(df):
    return {c: str(t) for c, t in df.dtypes.items()}


class _Clock:
    """A module's `time`, its sleeps recorded instead of slept. Only that
    module's own name is replaced: patching the stdlib's time.sleep would also
    record every other sleep in the process -- subprocess polls with it while
    provenance.git_info runs git for the manifest."""

    def __init__(self, sleeps):
        self.sleep = sleeps.append

    def __getattr__(self, name):
        return getattr(time, name)


def _record_sleeps(monkeypatch, module=IO):
    """The back-off sleeps of `module` (io_mascope's by default), recorded."""
    sleeps = []
    monkeypatch.setattr(module, "time", _Clock(sleeps))
    return sleeps


def _load(name):
    return json.loads((FIX / name).read_text(encoding="utf-8"))


def _refusal_error(url_id="SAMPLE3xxxxxxxxx"):
    f = _load("ms2_monoterpene_b_summary_refused.json")
    return ValidationError(message=f["message"], status_code=f["status_code"],
                           url=f["url"].replace("SAMPLE3xxxxxxxxx", url_id))


def _server_500(sid):
    """What the server answers for a sample it cannot read MS2 from."""
    return ServerError("Failed to get ms2 summary. Unexpected error (ref: 0f1e2d3c)",
                       status_code=500, url=f"https://server.example/api/samples/{sid}/ms2/summary")


def _waf_403(sid):
    return AuthenticationError("Attention Required! | Cloudflare", status_code=403,
                               url=f"https://server.example/api/samples/{sid}/ms2/summary")


def _api_403(sid, what="ms2/summary"):
    """The API's own 403, as the SDK raises it: what the server answers for an
    unknown or stale sample id (its JSON `error`, then the SDK's advice)."""
    return AuthenticationError("You do not have permission to perform this action.. Please "
                               "check your API token.", status_code=403,
                               url=f"https://server.example/api/samples/{sid}/{what}")


def _html_403(sid):
    """A proxy's block page as the SDK raises it: the page's first 200 characters."""
    return AuthenticationError('<!DOCTYPE html>\n<!--[if lt IE 7]> <html class="no-js ie6 oldie" '
                               'lang="en-US"> <![endif]-->. Please check your API token.',
                               status_code=403,
                               url=f"https://server.example/api/samples/{sid}/ms2/summary")


#: The SDK's connection failures, worded as urllib3 words them (captured offline
#: through the SDK's own http_get with DNS / connect patched to fail), naming
#: the server's host bare -- not only in "host='...'" and the URL.
def _dns_failure(sid, host="sentinel-host.invalid"):
    return MascopeConnectionError(
        "Could not connect. Please check the URL and your network connection: "
        f"HTTPSConnectionPool(host='{host}', port=443): Max retries exceeded with url: "
        f"/api/samples/{sid}/ms2/summary (Caused by NameResolutionError(\"HTTPSConnection("
        f"host='{host}', port=443): Failed to resolve '{host}' ([Errno -2] Name or service "
        "not known)\"))", url=f"https://{host}/api/samples/{sid}/ms2/summary")


def _connect_timeout(sid, host="sentinel-host.invalid"):
    return MascopeTimeoutError(
        f"Request timed out: HTTPSConnectionPool(host='{host}', port=443): Max retries "
        f"exceeded with url: /api/samples/{sid}/ms2/summary (Caused by ConnectTimeoutError("
        f"<HTTPSConnection(host='{host}', port=443) at 0x72d884bbc2d0>, 'Connection to {host} "
        "timed out. (connect timeout=30)'))", url=f"https://{host}/api/samples/{sid}/ms2/summary")


def _compute_504(sid):
    """The MS2 endpoints' own timeout: the server gave up computing the sample."""
    return ServerError("MS2 summary timed out after 120s for sample", status_code=504,
                       url=f"https://server.example/api/samples/{sid}/ms2/summary")


class _Seq(list):
    """Answers in turn (an exception is raised); the last one repeats."""


def _store():
    """sample id -> what the fake server answers for it."""
    def sample(stem):
        out = {"centroids": _load(f"{stem}_centroids.json"), "ms1": _load(f"{stem}_ms1.json"),
               "record": _load(f"{stem}_sample.json")}
        p = FIX / f"{stem}_summary.json"
        out["summary"] = _load(p.name) if p.exists() else _refusal_error()
        return out

    def never(what):
        return AssertionError(f"{what} must not be asked for")
    old = sample("ms2_monoterpene_a")
    old.update(summary=_load("ms2_old_server_summary.json"),
               centroids=_load("ms2_old_server_centroids.json"))
    empty = {"summary": _load("ms2_empty_scaffold_summary.json"),
             "centroids": never("no MS2 counted: the centroids"),
             "ms1": never("no MS2 counted: the MS1"),
             "record": {"sample_item_name": "MS1-only file", "polarity": "+",
                        "instrument_type": "orbi", "t0": 5.0, "t1": 300.0, "length": 300.0}}
    bad = {"summary": _server_500(BAD), "centroids": _server_500(BAD), "ms1": _server_500(BAD),
           "record": {"sample_item_name": "raw file missing", "polarity": "+",
                      "instrument_type": "orbi", "t0": 5.0, "t1": 300.0, "length": 300.0}}
    tof = {w: never(f"a TOF sample's {w}") for w in ("summary", "centroids", "ms1")}
    tof["record"] = {"sample_item_name": "TOF file", "polarity": "-", "instrument_type": "tof",
                     "t0": 0.0, "t1": 600.0, "length": 600.0}
    waf = dict(bad, summary=_waf_403(WAF))
    mta = sample("ms2_monoterpene_a")
    # the same >= 1.8 server asked WITHOUT by_activation (the real answer)
    mta["centroids_averaged"] = _load("ms2_monoterpene_a_centroids_averaged.json")
    stale = {w: _api_403(STALE) for w in ("summary", "centroids", "ms1")}
    stale["record"] = _api_403(STALE, "")
    down = {w: _dns_failure(DOWN) for w in ("summary", "centroids", "ms1", "record")}
    return {NH4: sample("ms2_nh4_exploratory"), MTA: mta,
            MTB: sample("ms2_monoterpene_b"), OLD: old, EMPTY: empty, BAD: bad, TOF: tof,
            WAF: waf, STALE: stale, DOWN: down}


class _FakeMs2:
    def __init__(self, server, sid):
        self.server, self.sid = server, sid

    def get_summary(self, **kw):
        return self.server.answer("summary", self.sid, kw)

    def get_averaged_centroids(self, **kw):
        # a >= 1.8 server answers per activation only when asked to: without
        # by_activation it blends every activation of a parent into one spectrum
        if kw.get("by_activation") is True:
            return self.server.answer("centroids", self.sid, kw)
        if "centroids_averaged" not in self.server.store[self.sid]:
            raise AssertionError(f"{self.sid}: centroids asked without by_activation")
        return self.server.answer("centroids_averaged", self.sid, kw)

    def get_ms1_centroids(self, **kw):
        return self.server.answer("ms1", self.sid, kw)


class _FakeSamples:
    def __init__(self, server):
        self.server = server

    def ms2(self, sid):
        return _FakeMs2(self.server, sid)

    def get(self, sid):
        return self.server.answer("record", sid, {})


class FakeClient:
    #: what a real client holds; nothing of it may reach a written file
    base_url = SENTINEL_URL
    access_token = SENTINEL_TOKEN

    def __init__(self, store=None):
        self.store = store if store is not None else _store()
        self.calls = []
        self.samples = _FakeSamples(self)

    def answer(self, what, sid, kw):
        self.calls.append((what, sid, kw))
        x = self.store[sid][what]
        if isinstance(x, _Seq):
            x = x.pop(0) if len(x) > 1 else x[0]
        if isinstance(x, Exception):
            raise x
        return x


def _tables(stem, refusal=None):
    cent = _load(f"{stem}_centroids.json")
    p = FIX / f"{stem}_summary.json"
    summ = _load(p.name) if p.exists() else None
    return (S.spectra_table("S", cent, summ), S.parents_table("S", cent, summ, refusal=refusal))


# --------------------------------------------------------------- readers ----
def test_parse_activation():
    assert S.parse_activation("hcd40.00") == ("hcd", 40.0)
    assert S.parse_activation("hcd1.00") == ("hcd", 1.0)
    assert S.parse_activation("cid35") == ("cid", 35.0)
    # the server writes two decimals: a fractional NCE keeps them
    assert S.parse_activation("hcd27.50") == ("hcd", 27.5)
    assert S.parse_activation("hcd27.5") == ("hcd", 27.5)
    kind, n = S.parse_activation("ETD")
    assert kind == "etd" and math.isnan(n)
    for empty in ("", None, "  "):
        kind, n = S.parse_activation(empty)
        assert kind is None and math.isnan(n)
    # an unknown spelling is kept whole, never guessed into a number
    kind, n = S.parse_activation("sa-hcd 35")
    assert kind == "sa-hcd 35" and math.isnan(n)


def test_spectrum_keys_are_the_servers():
    assert S.spectrum_key(137.1321, "hcd1.00") == "137.1321@hcd1.00"
    assert S.spectrum_key(197.13) == "197.13"
    for stem in ("ms2_nh4_exploratory", "ms2_monoterpene_a", "ms2_monoterpene_b"):
        for key, entry in _load(f"{stem}_centroids.json").items():
            assert S.spectrum_key(entry["parent_peak_mz"], entry["activation"]) == key


def test_hidden_tail_from_the_listing_row():
    b = _load("ms2_monoterpene_b_sample.json")
    assert S.hidden_tail_s(b) == pytest.approx(210.479, abs=1e-3)
    assert S.hidden_tail_s(pd.Series(b)) == pytest.approx(210.479, abs=1e-3)
    assert S.hidden_tail_s(_load("ms2_monoterpene_a_sample.json")) == 0.0    # t1 == length
    # the exploratory file holds MS2 and still runs 94.7 s past its last ingested scan
    assert S.hidden_tail_s(_load("ms2_nh4_exploratory_sample.json")) == pytest.approx(94.6895,
                                                                                      abs=1e-3)
    assert S.hidden_tail_s({"t1": 100.0, "length": 100.4}) == 0.0           # under 0.5 s
    assert math.isnan(S.hidden_tail_s({"t1": 100.0}))
    assert math.isnan(S.hidden_tail_s(None))


# -------------------------------------------------------- real fixtures ----
def test_exploratory_file_pairs_every_nce_with_its_volts():
    spectra, parents = _tables("ms2_nh4_exploratory")
    assert len(parents) == 29 and parents["spectrum"].is_unique
    assert parents["parent_mz"].nunique() == 7
    assert spectra["spectrum"].nunique() == 29
    by = parents.set_index("spectrum")
    # NCE from the activation string, volts from the summary group: paired, never converted
    for key, nce, volts, scans in (("136.0213@hcd1.00", 1, 4.05, 24),
                                   ("136.0213@hcd91.00", 91, 19.84, 31),
                                   ("197.13@hcd9.00", 9, 6.16, 7),
                                   ("315.106@hcd1.00", 1, 4.28, 7),
                                   ("315.106@hcd5.00", 5, 5.90, 5)):
        assert by.loc[key, "nce"] == nce and by.loc[key, "kind"] == "hcd"
        assert by.loc[key, "hcd_v"] == pytest.approx(volts)
        assert by.loc[key, "scan_count"] == scans
    # one NCE is not one voltage: NCE 1 is 4.05 V at m/z 136 and 4.28 V at m/z 315
    assert set(by.loc[by["nce"] == 1, "hcd_v"]) == {4.05, 4.08, 4.10, 4.13, 4.15, 4.28}
    assert parents["energy_resolved"].all() and spectra["energy_resolved"].all()
    assert (parents["hcd_v_values"] == "").all()             # one volt value per group
    assert (parents["isolation_width"] == 0.4).all()
    assert (parents["t_min"] < parents["t_max"]).all()
    assert (parents["note"] == "").all()
    # the spectra and the parents tables agree on every spectrum
    n = spectra.groupby("spectrum").size()
    assert (by["n_peaks"] == n.reindex(by.index)).all()
    tic = spectra.groupby("spectrum")["intensity"].sum()
    assert np.allclose(by["tic"], tic.reindex(by.index))
    _check_against_payload("ms2_nh4_exploratory", spectra, parents)


#: The per-spectrum metadata the spectra table repeats on every row.
_META = ("sample_id", "parent_mz", "activation", "kind", "nce", "hcd_v", "hcd_v_values",
         "scan_count", "t_min", "t_max", "energy_resolved")


def _check_against_payload(stem, spectra, parents, summary=True):
    """Every spectrum's rows carry its parents-table metadata, the parents
    table carries the summary group's numbers, and the arrays are the
    payload's, m/z-sorted."""
    cent = _load(f"{stem}_centroids.json")
    groups = {(g["parent_peak_mz"], g["activation"]): g
              for g in (_load(f"{stem}_summary.json")["groups"] if summary else [])}
    by = parents.set_index("spectrum")
    assert set(spectra["spectrum"]) == set(cent)
    for key, rows in spectra.groupby("spectrum", sort=False):
        meta = rows[list(_META)].drop_duplicates()
        assert len(meta) == 1, key
        for c in _META:
            want, got = by.loc[key, c], meta[c].iloc[0]
            assert (pd.isna(want) and pd.isna(got)) or want == got, (key, c)
        entry = cent[key]
        assert (by.loc[key, "parent_mz"], by.loc[key, "activation"]) == (
            entry["parent_peak_mz"], entry["activation"])
        g = groups.get((entry["parent_peak_mz"], entry["activation"]))
        if g is not None:
            assert by.loc[key, "hcd_v"] == pytest.approx(np.mean(g["hcd_energy"]))
            assert (by.loc[key, "scan_count"], by.loc[key, "t_min"], by.loc[key, "t_max"]) == (
                g["scan_count"], g["t_min"], g["t_max"])
        order = np.argsort(entry["mz"], kind="stable")
        for name in ("mz", "intensity", "resolution", "signal_to_noise"):
            want = np.asarray(entry[name])[order]
            assert np.array_equal(rows[name].to_numpy(), want), (key, name)


def test_a_group_with_several_volts_reads_their_mean():
    """hcd_v is the mean of a group's hcd_energy list; hcd_v_values lists them
    (a stepped energy). Synthetic: standard A with one group at [4.0, 6.0] V."""
    summ = _load("ms2_monoterpene_a_summary.json")
    g = next(g for g in summ["groups"] if g["activation"] == "hcd1.00"
             and g["parent_peak_mz"] == 137.1321)
    g["hcd_energy"] = [4.0, 6.0]
    cent = _load("ms2_monoterpene_a_centroids.json")
    spectra = S.spectra_table("S", cent, summ)
    parents = S.parents_table("S", cent, summ).set_index("spectrum")
    assert parents.loc["137.1321@hcd1.00", "hcd_v"] == 5.0
    assert parents.loc["137.1321@hcd1.00", "hcd_v_values"] == "4;6"
    rows = spectra[spectra["spectrum"] == "137.1321@hcd1.00"]
    assert (rows["hcd_v"] == 5.0).all() and (rows["hcd_v_values"] == "4;6").all()
    assert (parents.drop(index="137.1321@hcd1.00")["hcd_v_values"] == "").all()
    row = S.inventory_row("S", summ)
    assert (row["hcd_v_min"], row["hcd_v_max"]) == (4.0, 8.88)


def test_monoterpene_standard_a():
    spectra, parents = _tables("ms2_monoterpene_a")
    assert len(parents) == 12 and parents["parent_mz"].nunique() == 3
    assert sorted(parents["parent_mz"].unique()) == [137.1321, 155.1429, 155.1559]
    assert list(parents.loc[parents["parent_mz"] == 155.1559, "nce"]) == [1, 5, 15, 25]
    assert parents["scan_count"].sum() == 78                 # = the summary's ms2_scan_count
    row = S.inventory_row(MTA, _load("ms2_monoterpene_a_summary.json"),
                          sample_row=_load("ms2_monoterpene_a_sample.json"))
    assert row["status"] == "ok" and row["note"] == "" and row["hidden_tail_s"] == 0.0
    assert row["n_spectra"] == 12 and row["n_parents"] == 3 and row["energy_resolved"]
    assert row["parents"] == "137.1321;155.1429;155.1559"    # the server's own keys
    assert row["nce"] == "1;3;5;9;15;20;25"
    assert (row["hcd_v_min"], row["hcd_v_max"]) == (4.05, 8.88)
    assert (row["ms1_scans"], row["ms2_scans"]) == (61, 78)
    _check_against_payload("ms2_monoterpene_a", spectra, parents)


def test_an_energy_averaged_answer_from_a_current_server():
    """Standard A's centroids asked WITHOUT by_activation (the real answer of
    the same >= 1.8 server): one blended spectrum per parent. Its summary
    groups are what the spectrum spans, never 12 'missing' spectra."""
    cent = _load("ms2_monoterpene_a_centroids_averaged.json")
    summ = _load("ms2_monoterpene_a_summary.json")
    assert not S.is_energy_averaged(cent)          # a current server: 'activation' is ''
    parents = S.parents_table("S", cent, summ)
    assert list(parents["spectrum"]) == ["137.1321", "155.1429", "155.1559"]
    assert not parents["energy_resolved"].any() and parents["nce"].isna().all()
    assert parents["hcd_v"].isna().all()           # no single energy of its own
    by = parents.set_index("spectrum")
    assert by.loc["137.1321", "hcd_v_values"] == "4.05;4.4;4.76;5.46;7.41"   # the volts spanned
    assert by.loc["137.1321", "scan_count"] == 7 + 5 + 9 + 6 + 6
    assert parents["scan_count"].sum() == 78
    assert by.loc["155.1559", "t_min"] == pytest.approx(218.0855, abs=1e-3)
    assert by.loc["155.1559", "t_max"] == pytest.approx(273.5023, abs=1e-3)
    assert not parents["note"].str.contains(S.SUMMARY_ONLY_NOTE, regex=False).any()
    assert (parents["note"] == "energy-averaged: one spectrum over every activation of the "
                               "parent").all()
    spectra = S.spectra_table("S", cent, summ)
    assert spectra["spectrum"].nunique() == 3 and not spectra["energy_resolved"].any()
    row = S.inventory_row("S", summ, centroids=cent)
    assert row["status"] == "ok" and not row["energy_resolved"]
    assert row["n_spectra"] == 3 and row["nce"] == "" and row["n_parents"] == 3
    assert "energy-averaged (asked without by_activation)" in row["note"]


def test_summary_and_centroid_keys_that_disagree():
    """Standard A with one group's spectrum missing (155.1559@hcd25.00) and one
    spectrum no group matches (155.1559@hcd30.00): each row says which side is
    missing, and only spectra the centroid answer holds are counted."""
    summ = _load("ms2_monoterpene_a_summary.json")
    cent = _load("ms2_monoterpene_a_centroids.json")
    cent["155.1559@hcd30.00"] = dict(cent.pop("155.1559@hcd25.00"), activation="hcd30.00")
    parents = S.parents_table("S", cent, summ)
    assert len(parents) == 13
    by = parents.set_index("spectrum")
    miss = by.loc["155.1559@hcd25.00"]
    assert miss["note"] == S.SUMMARY_ONLY_NOTE
    # the spec's own words: the keys disagree -- never blamed on the noise threshold
    assert miss["note"].startswith("the summary lists this group but no centroid spectrum is "
                                   "keyed to it")
    assert "noise" not in miss["note"]
    assert (miss["n_peaks"], miss["tic"], miss["nce"], miss["scan_count"]) == (0, 0.0, 25, 6)
    assert miss["hcd_v"] == 8.88 and math.isnan(miss["precursor_mz"]) and miss["energy_resolved"]
    extra = by.loc["155.1559@hcd30.00"]
    assert extra["note"] == S.NO_GROUP_NOTE
    assert extra["note"] == ("no summary group matches this spectrum's parent m/z and activation: "
                             "scan_count, t_min, t_max and hcd_v unknown")
    assert extra["nce"] == 30 and extra["n_peaks"] == 11 and extra["energy_resolved"]
    for c in ("scan_count", "t_min", "t_max", "hcd_v"):
        assert math.isnan(extra[c]), c
    assert extra["precursor_mz"] == pytest.approx(155.1558, abs=2e-4)
    assert (by.drop(index=["155.1559@hcd25.00", "155.1559@hcd30.00"])["note"] == "").all()
    # the spectra table holds only what the centroid answer holds
    spectra = S.spectra_table("S", cent, summ)
    assert set(spectra["spectrum"]) == set(cent) and spectra["spectrum"].nunique() == 12
    row = S.inventory_row("S", summ, centroids=cent)
    assert row["n_spectra"] == 12 and row["nce"] == "1;3;5;9;15;20;30"
    assert row["status"] == "ok" and row["energy_resolved"]
    # without a centroid answer (`ms2 list`) the summary's groups are the spectra
    assert S.inventory_row("S", summ)["nce"] == "1;3;5;9;15;20;25"


def test_refused_summary_builds_spectra_from_the_centroid_keys():
    client = FakeClient()
    res = S.extract_sample(client, MTB)
    assert res.refusal == REASON
    p = res.parents
    assert len(p) == 6 and set(p["parent_mz"]) == {137.13}
    assert list(p["nce"]) == [1, 5, 10, 15, 25, 35]
    assert list(p["spectrum"]) == ["137.13@hcd1.00", "137.13@hcd5.00", "137.13@hcd10.00",
                                   "137.13@hcd15.00", "137.13@hcd25.00", "137.13@hcd35.00"]
    # what only the summary knows stays unknown
    for col in ("scan_count", "t_min", "t_max", "hcd_v", "isolation_width"):
        assert p[col].isna().all(), col
    assert p["energy_resolved"].all()
    assert (p["note"] == (f"summary refused ({REASON}): scan_count, t_min, t_max, hcd_v and "
                          "isolation width unknown; precursor measured at 137.1323, +2.3 mDa "
                          "from the server's parent key (the method's isolation centre, by "
                          "which the server is queried)")).all()
    row = res.row
    assert row["status"] == f"summary refused: {REASON}"
    assert row["hidden_tail_s"] == pytest.approx(210.5, abs=0.05)
    assert "reprocess the sample on the server" in row["note"]
    assert row["note"].split(S.NOTE_SEP)[0] == (
        "spectra are built from the centroid keys, and scan counts, times, volts and the "
        "isolation width stay unknown (pass --iso-width to `ms2 extract` for the window "
        "inventory)")
    assert math.isnan(row["ms2_scans"]) and row["n_spectra"] == 6
    # the row still names the parent and the activations, from the centroid keys
    assert (row["n_parents"], row["parents"], row["nce"]) == (1, "137.13", "1;5;10;15;25;35")
    assert row["energy_resolved"] and math.isnan(row["isolation_width"])
    assert math.isnan(row["hcd_v_min"]) and math.isnan(row["ms1_scans"])
    # a refusal whose centroid answer is empty is still the refusal (the known
    # server defect and its workaround), never 'no summary'
    for empty in ({}, None):
        bare = S.inventory_row(MTB, None, centroids=empty, refusal=REASON)
        assert bare["status"] == f"summary refused: {REASON}" and bare["n_spectra"] == 0
        assert bare["note"].startswith("spectra are built from the centroid keys")
    lister = FakeClient()
    listed = S.list_sample(lister, MTB)                  # `ms2 list` reads the same row
    assert json.dumps(listed, sort_keys=True) == json.dumps(row, sort_keys=True)
    assert lister.calls[-1] == ("centroids", MTB, {"noise_threshold": 10.0,
                                                   "by_activation": True})
    assert len(res.spectra) == p["n_peaks"].sum() == 183
    _check_against_payload("ms2_monoterpene_b", res.spectra, p, summary=False)
    # no isolation width: the window is skipped and says why ...
    assert len(res.window) == 0 and list(res.window.columns) == list(S.WINDOW_COLUMNS)
    assert any("--iso-width" in n and "refused" in n for n in res.notes)
    assert not any(c[0] == "ms1" for c in client.calls)
    # ... unless one is given
    res = S.extract_sample(FakeClient(), MTB, iso_width=0.4)
    assert len(res.window) and res.params["isolation_width_source"] == "explicit"
    # the server's URL (host + sample id) reaches no note
    text = json.dumps([res.row["note"], res.row["status"], res.notes, list(res.parents["note"])])
    assert "server.example" not in text and "http" not in text and MTB not in text


def test_every_parent_keeps_its_window_even_without_a_peak():
    """The window is built per PARENT (the parents table), not per spectrum
    that holds a peak: a parent whose every spectrum is empty (all under the
    noise threshold) or that only the summary lists keeps its inventory."""
    store = _store()
    for key, e in store[NH4]["centroids"].items():
        if e["parent_peak_mz"] == 315.106:
            store[NH4]["centroids"][key] = dict(e, mz=[], intensity=[], resolution=[],
                                                signal_to_noise=[])
    res = S.extract_sample(FakeClient(store), NH4)
    assert 315.106 not in set(res.spectra["parent_mz"])
    empty = res.parents[res.parents["parent_mz"] == 315.106]
    assert len(empty) == 2 and (empty["n_peaks"] == 0).all()
    w = res.window[res.window["parent_mz"] == 315.106]
    assert len(w) and w["is_parent"].sum() == 1
    assert sorted(res.window["parent_mz"].unique()) == sorted(res.parents["parent_mz"].unique())
    assert res.notes == []
    # standard A with every 155.1559 spectrum missing: four summary-only rows, one window
    store = _store()
    store[MTA]["centroids"] = {k: e for k, e in store[MTA]["centroids"].items()
                               if e["parent_peak_mz"] != 155.1559}
    res = S.extract_sample(FakeClient(store), MTA)
    assert (res.parents["parent_mz"] == 155.1559).sum() == 4
    w = res.window[res.window["parent_mz"] == 155.1559]
    assert len(w) and w["is_parent"].sum() == 1
    assert sorted(res.window["parent_mz"].unique()) == [137.1321, 155.1429, 155.1559]


def test_extract_asks_for_the_weak_fragments_and_list_does_not_need_them():
    """`ms2 extract` asks the server for S/N >= 3, not the SDK's 10 (a
    breakdown curve needs the weak fragments); `ms2 list` reads only the
    spectrum keys, which no S/N floor removes, so it keeps the fetcher's
    default, the SDK's."""
    client = FakeClient()
    S.extract_sample(client, NH4)
    asked = [kw for what, _, kw in client.calls if what == "centroids"]
    assert asked == [{"noise_threshold": 3.0, "by_activation": True}]
    assert S.DEFAULT_NOISE_THRESHOLD == 3.0
    client = FakeClient()
    S.extract_sample(client, NH4, noise_threshold=10)
    assert [kw for what, _, kw in client.calls if what == "centroids"] == [
        {"noise_threshold": 10.0, "by_activation": True}]


#: the server's own wording: '<context>. <the error>.'
NO_SCANS_MSG = ("Failed to get ms2 averaged centroids. No scans found matching the specified "
                "filters: polarity='+', time_range=(5.0, 300.0), ms_type='Ms2'.")


def _no_scans(sid, status=400):
    cls = ValidationError if status == 400 else ServerError
    return cls(NO_SCANS_MSG, status_code=status,
               url=f"https://server.example/api/samples/{sid}/ms2/centroids")


def test_no_summary_and_no_ms2_scans_is_no_ms2():
    """With no summary to count MS2 scans on, an MS1-only sample reaches the
    centroid call, which answers HTTP 400 'No scans found' rather than an
    empty answer: that is 'no MS2', with empty tables and no MS1 request --
    and only then. The same 400 after a refused summary (whose refusal says
    MS2 scans exist), or any other 400, is still raised."""
    store = _store()
    store[EMPTY]["summary"] = None
    store[EMPTY]["centroids"] = _no_scans(EMPTY)
    client = FakeClient(store)
    res = S.extract_sample(client, EMPTY)
    assert res.row["status"] == "no MS2" and S.NO_MS2_SCANS_NOTE in res.row["note"]
    assert "server.example" not in res.row["note"]
    assert (len(res.spectra), len(res.parents), len(res.window)) == (0, 0, 0)
    assert [w for w, _, _ in client.calls] == ["record", "summary", "centroids"]
    assert res.row["note"].count(S.NO_MS2_SCANS_NOTE) == 1
    assert "answered no MS2 summary" in res.row["note"]   # said once, by inventory_row
    # a file that runs past its last ingested scan may hide its MS2: never 'no MS2'
    store = _store()
    store[EMPTY]["summary"] = None
    store[EMPTY]["centroids"] = _no_scans(EMPTY)
    store[EMPTY]["record"] = dict(store[EMPTY]["record"], t1=100.0, length=300.0)
    res = S.extract_sample(FakeClient(store), EMPTY)
    assert res.row["status"] == "MS2 may be hidden"
    assert S.HIDDEN_TAIL_NOTE.format(s=200.0) in res.row["note"]
    # the same 400 after a refused summary (MS2 scans exist), with a summary
    # that counts MS2 scans, or under another status, is raised
    store = _store()
    store[MTB]["centroids"] = _no_scans(MTB)               # its summary is refused
    with pytest.raises(ValidationError):
        S.extract_sample(FakeClient(store), MTB)
    store = _store()
    store[NH4]["centroids"] = _no_scans(NH4)               # its summary counts 213 MS2 scans
    with pytest.raises(ValidationError):
        S.extract_sample(FakeClient(store), NH4)
    store = _store()
    store[EMPTY]["summary"] = None
    store[EMPTY]["centroids"] = _no_scans(EMPTY, status=500)
    with pytest.raises(ServerError):
        S.extract_sample(FakeClient(store), EMPTY)
    store = _store()
    store[EMPTY]["summary"] = None
    store[EMPTY]["centroids"] = ValidationError("Bad parameter", status_code=400, url="u")
    with pytest.raises(ValidationError):
        S.extract_sample(FakeClient(store), EMPTY)


#: What the server answers a sample's MS1 centroid request it cannot serve:
#: no MS1 scan in a sample window that holds only the MS2 block, and its own
#: timeout on a long file.
_MS1_FAILURES = [
    lambda sid: ValidationError("No scans found matching the specified filters", status_code=400,
                                url=f"https://server.example/api/samples/{sid}/ms2/ms1_centroids"),
    lambda sid: ServerError("MS1 centroid extraction timed out after 120s for sample",
                            status_code=504,
                            url=f"https://server.example/api/samples/{sid}/ms2/ms1_centroids"),
]


@pytest.mark.parametrize("make", _MS1_FAILURES)
def test_a_failed_ms1_call_fails_only_the_window(monkeypatch, make):
    """The MS1 centroids feed the window only: when the server refuses them
    for this sample, the MS2 tables already read are kept, the window is
    empty, and a note gives the server's reason without its address."""
    sleeps = _record_sleeps(monkeypatch)
    store = _store()
    err = make(NH4)
    store[NH4]["ms1"] = err
    client = FakeClient(store)
    res = S.extract_sample(client, NH4)
    assert [c[0] for c in client.calls] == ["record", "summary", "centroids", "ms1"]
    assert sleeps == []                                   # the sample's own answer
    assert len(res.parents) == 29 and res.spectra["spectrum"].nunique() == 29
    assert len(res.spectra) == 249 and res.row["status"] == "ok"
    assert len(res.window) == 0 and _dtypes(res.window) == WINDOW_DTYPES
    (note,) = res.notes
    assert note == S.WINDOW_FAILED_NOTE.format(
        error=f"{type(err).__name__}: {IO.server_error_text(err)}")
    assert note.startswith("window inventory skipped: the server could not serve this "
                           "sample's MS1 centroids")
    assert "The MS2 tables are kept" in note and err.message in note
    assert "server.example" not in note and NH4 not in note and "://" not in note
    assert res.params["isolation_width_source"] == "server"


@pytest.mark.parametrize("make", [_dns_failure, _waf_403])
def test_a_failed_ms1_call_that_is_the_servers_still_propagates(monkeypatch, make):
    """A connection failure or a WAF block on the MS1 call is not the
    sample's: it propagates (after the back-off for a WAF page), so a loop over
    samples stops there."""
    sleeps = _record_sleeps(monkeypatch)
    store = _store()
    store[NH4]["ms1"] = make(NH4)
    with pytest.raises(type(make(NH4))):
        S.extract_sample(FakeClient(store), NH4)
    assert sleeps == ([] if make is _dns_failure else [2.0, 4.0, 8.0])


def test_extract_sample_refuses_a_bad_width_before_asking():
    client = FakeClient()
    for bad in (0, -0.4, float("nan"), "x"):
        with pytest.raises(ValueError, match="iso_width must be a positive width"):
            S.extract_sample(client, NH4, iso_width=bad)
    assert client.calls == []


def test_an_empty_window_says_so():
    store = _store()
    store[NH4]["ms1"] = {"mz": [50.0, 400.0], "intensity": [1.0, 1.0],
                         "resolution": [1e5, 1e5], "signal_to_noise": [5.0, 5.0]}
    res = S.extract_sample(FakeClient(store), NH4)
    assert len(res.window) == 0 and len(res.parents) == 29
    assert res.notes == ["window inventory empty: no MS1 centroid inside any window"]
    assert res.params["isolation_width_source"] == "server"


def test_an_unknown_instrument_type_is_asked():
    """Only a KNOWN TOF is skipped: a record that names no instrument type (nor
    an instrument or file name that resolves to one) is asked for its MS2."""
    for rec in ({"sample_item_name": "monoterpene standard A", "t1": 370.964825,
                 "length": 370.964825},
                {"sample_item_name": "x", "filename": "Exploris_2026_x.raw", "t1": 1.0,
                 "length": 1.0}):
        assert IO.instrument_type_for(rec) is None and not S.is_tof(rec)
        store = _store()
        store[MTA]["record"] = rec
        client = FakeClient(store)
        row = S.list_sample(client, MTA)
        assert row["status"] == "ok" and row["n_spectra"] == 12
        assert [c[0] for c in client.calls] == ["record", "summary"]
        client = FakeClient(store)
        res = S.extract_sample(client, MTA)
        assert res.row["status"] == "ok" and len(res.parents) == 12
        assert [c[0] for c in client.calls] == ["record", "summary", "centroids", "ms1"]
    assert S.is_tof({"instrument_type": "tof"}) and S.is_tof(pd.Series({"instrument": "APi-TOF"}))
    assert not S.is_tof(None)


def test_refusal_is_typed_never_retried_and_other_errors_propagate():
    store = _store()
    # a sample id that spells a transient code inside the URL ('500'): handed to
    # the retry loop raw, it would be retried three times with back-off
    store[MTB]["summary"] = _refusal_error(url_id="SAMPLE500xxxxxxx")
    client = FakeClient(store)
    with pytest.raises(IO.MS2SummaryRefused) as exc:
        IO.fetch_ms2_summary(client, MTB)
    assert exc.value.reason == REASON and "500" not in str(exc.value)
    assert [c[0] for c in client.calls] == ["summary"]
    # another 400 and a 404 are not refusals: they propagate as they came
    store[MTB]["summary"] = ValidationError("Bad parameter", status_code=400, url="u")
    with pytest.raises(ValidationError):
        IO.fetch_ms2_summary(FakeClient(store), MTB)
    store[MTB]["summary"] = NotFoundError("Sample not found", status_code=404, url="u")
    with pytest.raises(NotFoundError):
        IO.fetch_ms2_summary(FakeClient(store), MTB)
    assert IO.ms2_summary_refusal(_refusal_error()) == REASON
    assert IO.ms2_summary_refusal(RuntimeError("multiple isolation widths")) is None


def test_fetchers_ask_for_activations_and_pass_the_knobs():
    # the fake answers as a >= 1.8 server does: without by_activation, one
    # blended spectrum per parent
    client = FakeClient()
    assert IO.fetch_ms2_spectra(client, MTA, by_activation=False) == _load(
        "ms2_monoterpene_a_centroids_averaged.json")
    assert client.calls == [("centroids_averaged", MTA, {"noise_threshold": 10.0,
                                                         "by_activation": False})]
    client = FakeClient()
    assert IO.fetch_ms2_spectra(client, NH4, noise_threshold=5) == client.store[NH4]["centroids"]
    IO.fetch_ms2_ms1(client, NH4, ppm=2)
    assert client.calls == [("centroids", NH4, {"noise_threshold": 5.0, "by_activation": True}),
                            ("ms1", NH4, {"ppm": 2})]
    # the SDK's "unavailable": the summary passes None on (status 'no summary'),
    # the others answer {}
    for what in ("summary", "centroids", "ms1", "record"):
        client.store[NH4][what] = None
    assert IO.fetch_ms2_summary(client, NH4) is None
    assert IO.fetch_ms2_spectra(client, NH4) == {} and IO.fetch_ms2_ms1(client, NH4) == {}
    assert IO.fetch_sample_record(client, NH4) == {}


_FETCHERS = [
    ("summary", lambda c, sid: IO.fetch_ms2_summary(c, sid)),
    ("centroids", lambda c, sid: IO.fetch_ms2_spectra(c, sid)),
    ("ms1", lambda c, sid: IO.fetch_ms2_ms1(c, sid)),
    ("record", lambda c, sid: IO.fetch_sample_record(c, sid)),
]
#: Every transient status peaky retries, as the SDK raises it (a 429 is its
#: MascopeAPIError; a 5xx its ServerError). A proxy's 403 page and a gateway's
#: 504 page are in test_a_final_answer_is_asked_once_whatever_the_id_spells.
_TRANSIENT = [(429, MascopeAPIError, "Too many requests"),
              (502, ServerError, "Bad gateway"),
              (503, ServerError, "Service temporarily unavailable"),
              (521, ServerError, "Web server is down"),
              (522, ServerError, "Connection timed out")]


@pytest.mark.parametrize("what, call", _FETCHERS)
@pytest.mark.parametrize("status, kind, message", _TRANSIENT)
def test_every_fetcher_retries_every_transient_status(monkeypatch, what, call, status, kind,
                                                      message):
    """One transient answer is retried after the back-off and the second try's
    answer returned -- for each status in the retry set, on each fetcher; a
    status dropped from the set would end a listing on one blip."""
    sleeps = _record_sleeps(monkeypatch)
    store = _store()
    want = store[NH4][what]
    store[NH4][what] = _Seq([kind(message, status_code=status, url="https://server.example/x"),
                             want])
    client = FakeClient(store)
    assert call(client, NH4) == want
    assert [c[0] for c in client.calls] == [what, what] and sleeps == [2.0]


@pytest.mark.parametrize("what, call", _FETCHERS)
def test_a_500_is_final_on_the_ms2_calls_and_retried_on_the_record(monkeypatch, what, call):
    """An MS2 500 is the server's answer for this sample (a non-Orbitrap file,
    a missing raw file): raised at once, even when its URL spells '403'. The
    sample record retries a 500, as every other peaky read does."""
    sleeps = _record_sleeps(monkeypatch)
    store = _store()
    want = store[NH4][what]
    e500 = ServerError("Failed to get ms2 summary. Unexpected error (ref: x)", status_code=500,
                       url="https://server.example/api/samples/SAMPLE403xxxxxxx/ms2")
    store[NH4][what] = _Seq([e500, want])
    client = FakeClient(store)
    if what == "record":
        assert call(client, NH4) == want
        assert [c[0] for c in client.calls] == ["record", "record"] and sleeps == [2.0]
        return
    with pytest.raises(ServerError):
        call(client, NH4)
    assert [c[0] for c in client.calls] == [what] and sleeps == []


def _sdk_client():
    """A client whose MS2 calls run through the REAL SDK resource and its
    transport (mascope_sdk's http_get); `_stub_http_get` answers every HTTP
    GET, so no socket is opened."""
    from mascope_sdk.resources.ms2 import Ms2Resource
    conn = SimpleNamespace(url=SENTINEL_URL, access_token=SENTINEL_TOKEN, _timeout=(1, 1),
                           _verify_ssl=True, _service_name="peaky-test")
    return SimpleNamespace(samples=SimpleNamespace(ms2=lambda sid: Ms2Resource(conn, sid)))


def _stub_http_get(monkeypatch, status, body):
    import requests
    from mascope_sdk import _http
    sent = []

    def get(url, **kw):
        sent.append(url)
        r = requests.models.Response()
        r.status_code, r.url = status, url
        if isinstance(body, dict):
            r._content = json.dumps(body).encode()
            r.headers["Content-Type"] = "application/json"
        else:
            r._content = body.encode()
        return r
    monkeypatch.setattr(_http.requests, "get", get)
    return sent, _record_sleeps(monkeypatch, _http)


def test_the_sdk_resends_a_504_beneath_peaky(monkeypatch):
    """What docs/MS2.md §10.2 / §10.5 say of the SDK's transport, checked on
    the installed SDK: the MS2 compute-timeout 504 gets no retry of peaky's own,
    yet reaches the server four times (the SDK re-sends every 504, 2 / 4 / 8 s);
    a gateway's 504 page, which peaky retries too, reaches it 16 times."""
    sent, sdk_sleeps = _stub_http_get(
        monkeypatch, 504, {"error": "MS2 summary timed out after 120s for sample"})
    sleeps = _record_sleeps(monkeypatch)
    with pytest.raises(ServerError) as exc:
        IO.fetch_ms2_summary(_sdk_client(), NH4)
    assert IO.is_ms2_compute_timeout(exc.value) and IO.is_sample_error(exc.value)
    assert len(sent) == 4 and sdk_sleeps == [2, 4, 8] and sleeps == []
    assert all(u.startswith(SENTINEL_URL) for u in sent)
    sent, sdk_sleeps = _stub_http_get(monkeypatch, 504, "<html>504 Gateway Time-out</html>")
    sleeps = _record_sleeps(monkeypatch)
    with pytest.raises(ServerError) as exc:
        IO.fetch_ms2_summary(_sdk_client(), NH4)
    assert not IO.is_ms2_compute_timeout(exc.value) and not IO.is_sample_error(exc.value)
    assert len(sent) == 16 and sleeps == [2.0, 4.0, 8.0] and sdk_sleeps == [2, 4, 8] * 4


def test_server_errors_are_told_apart_and_lose_their_url():
    e500, e403 = _server_500(BAD), _waf_403(WAF)
    assert IO.server_error_text(e500) == ("[HTTP 500] Failed to get ms2 summary. Unexpected "
                                          "error (ref: 0f1e2d3c)")
    assert "server.example" not in IO.server_error_text(e403)
    conn = ConnectionError("HTTPSConnectionPool(host='server.example', port=443): Max retries "
                           "exceeded with url: https://server.example/api/x")
    text = IO.server_error_text(conn)
    assert "server.example" not in text and "://" not in text
    assert IO.is_sample_error(e500) and IO.is_sample_error(_refusal_error())
    assert IO.is_sample_error(NotFoundError("Sample not found", status_code=404, url="u"))
    assert IO.is_sample_error(ValueError("centroid list '1@hcd1.00': intensity has 2 values"))
    assert not IO.is_sample_error(e403) and not IO.is_sample_error(conn)
    assert not IO.is_sample_error(ServerError("Bad gateway", status_code=502, url="u"))
    # what the SDK raises when the server cannot be reached, after its own four
    # attempts: neither is an OSError, and neither is about one sample
    for exc in (_dns_failure(NH4), _connect_timeout(NH4),
                MascopeTimeoutError("Request timed out: Read timed out. (read timeout=300)",
                                    url="https://server.example/api/x")):
        assert not isinstance(exc, OSError)
        assert not IO.is_sample_error(exc), type(exc).__name__
    # the API's own 403 (an unknown or stale id) is the sample's; a proxy's is not
    assert IO.is_api_403(_api_403(STALE)) and IO.is_sample_error(_api_403(STALE))
    for proxy in (e403, _html_403(WAF),
                  AuthenticationError("Unknown error. Please check your API token.",
                                      status_code=403, url="u")):
        assert not IO.is_api_403(proxy) and not IO.is_sample_error(proxy)
    assert not IO.is_api_403(AuthenticationError("Not authenticated", status_code=401, url="u"))
    assert not IO.is_sample_error(AuthenticationError("Not authenticated", status_code=401,
                                                      url="u"))
    # the MS2 endpoints' own compute timeout is the sample's; a gateway's 504 page is not
    assert IO.is_ms2_compute_timeout(_compute_504(NH4)) and IO.is_sample_error(_compute_504(NH4))
    gateway = ServerError("<html><head><title>504 Gateway Time-out</title></head>",
                          status_code=504, url="u")
    assert not IO.is_ms2_compute_timeout(gateway) and not IO.is_sample_error(gateway)


@pytest.mark.parametrize("make", [_dns_failure, _connect_timeout])
def test_a_connection_error_loses_the_host_in_every_wording(make, monkeypatch):
    """urllib3 names the host bare ("Failed to resolve '<host>'", "Connection to
    <host> timed out"), not only in host='...' and the URL; none of it, nor the
    request path, survives -- whether the host is the error's URL's or only the
    configured MASCOPE_URL's."""
    text = IO.server_error_text(make(NH4))
    assert "sentinel-host" not in text and "://" not in text and NH4 not in text, text
    assert ("Failed to resolve '<host>'" in text) or ("Connection to <host> timed out" in text)
    # a host the error's URL does not carry, known from the environment
    exc = make(NH4, host="203.0.113.40")
    exc.url = None
    monkeypatch.setenv("MASCOPE_URL", "http://203.0.113.40:8090")
    text = IO.server_error_text(exc)
    assert "203.0.113.40" not in text and NH4 not in text, text
    # ... and by its wording alone, with neither
    monkeypatch.delenv("MASCOPE_URL")
    exc = make(NH4, host="lab-only-host.internal")
    exc.url = None
    assert "lab-only-host" not in IO.server_error_text(exc)


def test_the_servers_host_is_dropped_in_any_wording(monkeypatch):
    """Wordings no pattern knows (a TLS failure, a proxy's): the host the
    error's URL names, and the configured MASCOPE_URL's, are dropped wherever
    they stand -- whole names only."""
    monkeypatch.delenv("MASCOPE_URL", raising=False)
    exc = MascopeConnectionError("Request failed: SSLError(certificate is not valid for "
                                 "Sentinel-Host.invalid; not for mysentinel-host.invalid)",
                                 url="https://sentinel-host.invalid/api/samples/x/ms2/summary")
    text = IO.server_error_text(exc)
    assert text == ("Request failed: SSLError(certificate is not valid for <host>; not for "
                    "mysentinel-host.invalid)")
    monkeypatch.setenv("MASCOPE_URL", "http://203.0.113.40:8090")
    exc = MascopeConnectionError("Request failed: proxy refused 203.0.113.40:8090 (203.0.113.401)")
    assert IO.server_error_text(exc) == "Request failed: proxy refused <host>:8090 (203.0.113.401)"


@pytest.mark.parametrize("what, call", [
    ("summary", lambda c, sid: IO.fetch_ms2_summary(c, sid)),
    ("centroids", lambda c, sid: IO.fetch_ms2_spectra(c, sid)),
    ("ms1", lambda c, sid: IO.fetch_ms2_ms1(c, sid)),
    ("record", lambda c, sid: IO.fetch_sample_record(c, sid)),
])
def test_a_final_answer_is_asked_once_whatever_the_id_spells(monkeypatch, what, call):
    """Retrying is decided by the STATUS, never by the error's text, whose URL
    quotes the sample id: a 404 / 400 / 422 for an id spelling '500' or '403',
    the API's own 403 (a stale id), the MS2 compute timeout and an SDK
    connection failure (already retried four times by the SDK) are each asked
    once, with no back-off."""
    sleeps = _record_sleeps(monkeypatch)
    url = f"https://server.example/api/samples/SAMPLE500x403x429/{what}"
    finals = [NotFoundError("Sample not found", status_code=404, url=url),
              ValidationError("Bad parameter", status_code=422, url=url),
              ValidationError("Bad request", status_code=400, url=url),
              _api_403("SAMPLE500x403x429"), _dns_failure(NH4), _connect_timeout(NH4)]
    if what != "record":
        finals.append(_compute_504("SAMPLE500x403x429"))
    for exc in finals:
        store = _store()
        store[NH4][what] = exc
        client = FakeClient(store)
        with pytest.raises(type(exc)):
            call(client, NH4)
        assert [c[0] for c in client.calls] == [what], (what, exc)
    assert sleeps == []
    # a proxy's 403 page and a gateway's 504 page are retried
    for exc in (_html_403(NH4), ServerError("<html>504 Gateway Time-out</html>",
                                            status_code=504, url="u")):
        store = _store()
        want = store[NH4][what]
        store[NH4][what] = _Seq([exc, want])
        client = FakeClient(store)
        assert call(client, NH4) == want
        assert [c[0] for c in client.calls] == [what, what]
    assert sleeps == [2.0, 2.0]


def test_old_server_answers_energy_averaged_spectra():
    cent = _load("ms2_old_server_centroids.json")
    assert S.is_energy_averaged(cent)
    assert not S.is_energy_averaged(_load("ms2_monoterpene_a_centroids.json"))
    res = S.extract_sample(FakeClient(), OLD)
    assert res.row["status"] == "energy-averaged server" and not res.row["energy_resolved"]
    # the spec's own words, not read back from the module
    assert ("server returns energy-averaged MS2 only (Mascope < 1.8): per-energy spectra and "
            "breakdown curves need a newer server") in res.row["note"]
    assert S.ENERGY_AVERAGED_SERVER in res.row["note"]
    p = res.parents
    assert list(p["spectrum"]) == ["137.1321", "155.1429", "155.1559"]
    assert (p["activation"] == "").all() and p["nce"].isna().all()
    assert not p["energy_resolved"].any() and not res.spectra["energy_resolved"].any()
    # the older summary's energy map holds ONE scan-weighted mean volt per parent
    # (5.16 V for a parent measured at 4.05-7.41 V): it is hcd_v, said to be a
    # mean, and never passed off as the volts the spectrum spans
    assert list(p["hcd_v"]) == [5.16, 4.92, 6.07]
    assert (p["hcd_v_values"] == "").all()
    assert (p["note"] == S.NOTE_SEP.join([S.OLD_SERVER_MEAN_VOLTS,
                                          S.ENERGY_AVERAGED_SERVER])).all()
    # ... in so many words: a mean, not a voltage the instrument applied
    assert S.OLD_SERVER_MEAN_VOLTS == ("hcd_v is the server's mean over every scan of the parent "
                                       "(Mascope < 1.8), not a voltage the instrument applied")
    for key, rows in res.spectra.groupby("spectrum"):
        assert (rows["hcd_v"] == p.set_index("spectrum").loc[key, "hcd_v"]).all()
    # the precursor is still located in the blended spectrum
    assert p.loc[0, "precursor_mz"] == pytest.approx(137.13234, abs=2e-4)
    # `ms2 list` reads the same from the summary alone; a range of means is not
    # the volts applied
    for row in (res.row, S.inventory_row(OLD, _load("ms2_old_server_summary.json"))):
        assert row["status"] == "energy-averaged server" and row["n_spectra"] == 3
        assert not row["energy_resolved"] and row["nce"] == ""
        assert math.isnan(row["hcd_v_min"]) and math.isnan(row["hcd_v_max"])
        assert row["note"] == S.NOTE_SEP.join([S.ENERGY_AVERAGED_SERVER,
                                               S.OLD_SERVER_VOLTS_NOTE])


def test_empty_scaffold_is_no_ms2_by_its_count():
    client = FakeClient()
    res = S.extract_sample(client, EMPTY)
    assert res.row["status"] == "no MS2" and res.row["note"] == ""
    assert len(res.spectra) == len(res.parents) == len(res.window) == 0
    for df, want in ((res.spectra, SPECTRA_DTYPES), (res.parents, PARENTS_DTYPES),
                     (res.window, WINDOW_DTYPES)):
        assert _dtypes(df) == want and list(df.columns) == list(want)
    assert [c[0] for c in client.calls] == ["record", "summary"]     # nothing else asked
    # the same count with a tail past t1 is never called "no MS2"
    store = _store()
    store[EMPTY]["record"] = dict(store[EMPTY]["record"], length=420.0)
    row = S.list_sample(FakeClient(store), EMPTY)
    assert row["status"] == "MS2 may be hidden" and row["hidden_tail_s"] == 120.0
    assert "reprocess the sample on the server" in row["note"]
    # the count decides, not the shape: an empty-looking answer that counts scans is not
    # "no MS2" -- nor "ok": the server counts MS2 scans it could not group by parent
    shaped = dict(_load("ms2_empty_scaffold_summary.json"), ms2_scan_count=12)
    row = S.inventory_row(EMPTY, shaped)
    assert row["status"] == S.MS2_UNREADABLE == "MS2 unreadable"
    assert row["note"] == S.MS2_UNREADABLE_NOTE.format(n=12)
    assert "12 MS2 scans" in row["note"] and "noise threshold" in row["note"]
    assert S.inventory_row(EMPTY, None)["status"] == "no summary"


def test_window_inventory_shares_and_the_parent():
    win = S.window_table("S", _load("ms2_nh4_exploratory_ms1.json"),
                         [136.0213, 137.0572, 161.096, 179.1069, 197.13, 213.1612, 315.106], 0.4)
    assert sorted(win["parent_mz"].unique()) == [136.0213, 137.0572, 161.096, 179.1069,
                                                 197.13, 213.1612, 315.106]
    assert S.WINDOW_PAD_DA == 0.005
    assert (win["offset_da"].abs() <= 0.205).all()
    assert np.allclose(win.groupby("parent_mz")["share"].sum(), 1.0)
    assert (win.groupby("parent_mz")["is_parent"].sum() == 1).all()
    w = win[win["parent_mz"] == 137.0572]
    par = w[w["is_parent"]].iloc[0]
    assert par["offset_da"] == pytest.approx(0.00023, abs=5e-5)
    assert par["share"] == pytest.approx(0.446, abs=0.005)
    # the co-isolate 2.4 mDa above holds 10 % of the window and is not the parent
    co = w[(w["offset_da"] - 0.0024).abs() < 3e-4]
    assert len(co) == 1 and not co["is_parent"].iloc[0]
    assert co["share"].iloc[0] == pytest.approx(0.10, abs=0.01)
    # standard A: the 155.1429 window holds the 155.1559 parent too, at the larger share
    wa = S.window_table("S", _load("ms2_monoterpene_a_ms1.json"), [155.1429], 0.4)
    top = wa.loc[wa["share"].idxmax()]
    assert top["mz"] == pytest.approx(155.1558, abs=2e-4) and not top["is_parent"]
    assert wa.loc[wa["is_parent"], "mz"].iloc[0] == pytest.approx(155.1428, abs=2e-4)
    # standard B: the key 137.13 is the method's centre; the parent is C10H17+ 2.3 mDa
    # above it, not the sub-1 % side peaks nearer to the key
    wb = S.window_table("S", _load("ms2_monoterpene_b_ms1.json"), [137.13], 0.4)
    pb = wb[wb["is_parent"]].iloc[0]
    assert pb["offset_da"] == pytest.approx(0.0023, abs=1e-4)
    assert pb["share"] == pytest.approx(0.95, abs=0.01)
    nearer = wb[wb["offset_da"].abs() < abs(pb["offset_da"])]
    assert len(nearer) and (nearer["share"] < 0.01).all()
    with pytest.raises(ValueError):
        S.window_table("S", {}, [137.13], float("nan"))


@pytest.mark.parametrize("stem", ["ms2_nh4_exploratory", "ms2_monoterpene_a",
                                  "ms2_monoterpene_b"])
def test_window_rows_are_the_ms1_payloads(stem):
    """Every window row carries its MS1 centroid's own intensity, resolution and
    S/N (looked up by m/z in the payload), and the share is that intensity over
    the window's sum."""
    ms1 = _load(f"{stem}_ms1.json")
    parents = sorted({e["parent_peak_mz"] for e in _load(f"{stem}_centroids.json").values()})
    win = S.window_table("S", ms1, parents, 0.4)
    assert len(win)
    src = pd.DataFrame(ms1).set_index("mz")
    assert src.index.is_unique
    for name in ("intensity", "resolution", "signal_to_noise"):
        assert np.array_equal(win[name].to_numpy(), src.loc[win["mz"], name].to_numpy()), name
    total = win.groupby("parent_mz")["intensity"].transform("sum")
    assert np.allclose(win["share"], win["intensity"] / total)
    assert np.allclose(win["offset_da"], win["mz"] - win["parent_mz"])


def test_window_edge_is_half_the_width_plus_the_pad():
    """+-(isolation_width/2 + 5 mDa): a centroid 0.2049 Da off a 0.4-Da window's
    centre is listed (on either side), one 0.2051 Da off is not."""
    ms1 = {"mz": [99.7949, 99.7951, 99.85, 100.0, 100.2049, 100.2051],
           "intensity": [1.0, 1.0, 1.0, 5.0, 1.0, 1.0], "resolution": [1e5] * 6,
           "signal_to_noise": [5.0] * 6}
    win = S.window_table("S", ms1, [100.0], 0.4)
    assert list(win["mz"]) == [99.7951, 99.85, 100.0, 100.2049]
    assert list(win["offset_da"]) == pytest.approx([-0.2049, -0.15, 0.0, 0.2049])
    assert list(win["share"]) == pytest.approx([1 / 8, 1 / 8, 5 / 8, 1 / 8])
    assert list(win["is_parent"]) == [False, False, True, False]
    assert list(S.window_table("S", ms1, [100.0], 0.4, pad_da=0.0)["mz"]) == [99.85, 100.0]
    # on the real file: a centroid +0.2018 Da off 137.0572 is listed because of the pad only
    ms1 = _load("ms2_nh4_exploratory_ms1.json")
    w = S.window_table("S", ms1, [137.0572], 0.4)
    assert len(w) == 32 and w["offset_da"].max() == pytest.approx(0.20177, abs=1e-5)
    w0 = S.window_table("S", ms1, [137.0572], 0.4, pad_da=0.0)
    assert len(w0) == 31 and w0["offset_da"].max() < 0.2


def test_the_precursor_rule():
    """Among the peaks within max(10 ppm, 5 mDa) of the key that hold >= 10 % of
    the strongest one, the NEAREST to the key wins (the stronger on a tie)."""
    spectra, _ = _tables("ms2_nh4_exploratory")
    hcd5 = spectra[spectra["spectrum"] == "137.0572@hcd5.00"]
    i = S.locate_precursor(hcd5["mz"], hcd5["intensity"], 137.0572)
    assert hcd5["mz"].iloc[i] == pytest.approx(137.05746, abs=5e-5)
    strongest = hcd5.loc[(hcd5["mz"] - 137.0572).abs() <= 0.005, "intensity"].idxmax()
    assert hcd5.loc[strongest, "mz"] == pytest.approx(137.0596, abs=3e-4)    # not chosen
    # synthetic, at m/z 100 (search window 5 mDa): a near peak just under / over
    # 10 % of a far one
    mz = [100.001, 100.003]
    assert S.locate_precursor(mz, [9.9, 100.0], 100.0) == 1
    assert S.locate_precursor(mz, [10.1, 100.0], 100.0) == 0
    # equidistant (100 -+ 2**-10): the stronger
    assert S.locate_precursor([100 - 2 ** -10, 100 + 2 ** -10], [5.0, 6.0], 100.0) == 1
    assert S.locate_precursor([100 - 2 ** -10, 100 + 2 ** -10], [6.0, 5.0], 100.0) == 0
    # nothing within the search window
    assert S.locate_precursor([100.0051, 99.9949], [1.0, 1.0], 100.0) is None
    assert S.locate_precursor([], [], 100.0) is None
    # above m/z 500 the ppm clause is the wider: 10 ppm of 600 = 6 mDa
    assert S.locate_precursor([600.0058], [1.0], 600.0) == 0


def test_precursor_follows_one_ion_across_energies():
    spectra, parents = _tables("ms2_nh4_exploratory")
    by = parents.set_index("spectrum")
    # at NCE 5 the co-isolate 2.4 mDa above is the STRONGER peak; the precursor
    # stays the ion located at NCE 1
    hcd5 = spectra[spectra["spectrum"] == "137.0572@hcd5.00"]
    co = hcd5[(hcd5["mz"] - 137.0596).abs() < 3e-4]["intensity"].iloc[0]
    assert by.loc["137.0572@hcd5.00", "precursor_mz"] == pytest.approx(137.05746, abs=5e-5)
    assert by.loc["137.0572@hcd5.00", "precursor_intensity"] < co
    # the intensity is the spectrum's own peak
    for key, r in by.iterrows():
        s = spectra[(spectra["spectrum"] == key) & (spectra["mz"] == r["precursor_mz"])]
        assert len(s) == 1 and s["intensity"].iloc[0] == r["precursor_intensity"], key
    # it decays with the energy on the 197.13 parent
    p197 = parents[parents["parent_mz"] == 197.13]
    assert p197["precursor_intensity"].is_monotonic_decreasing
    # standard B: every energy reads the ion 2.3 mDa above the key, and says so
    _, pb = _tables("ms2_monoterpene_b", refusal=REASON)
    assert np.allclose(pb["precursor_mz"], 137.1323, atol=1.5e-4)
    assert pb["note"].str.contains("+2.3 mDa from the server's parent key", regex=False).all()
    # a spent precursor reads NaN, never a neighbour
    cent = _load("ms2_monoterpene_a_centroids.json")
    spent = cent["137.1321@hcd20.00"]
    keep = [i for i, m in enumerate(spent["mz"]) if abs(m - 137.1323) > 1e-3]
    cent["137.1321@hcd20.00"] = dict(spent, **{k: [spent[k][i] for i in keep]
                                               for k in ("mz", "intensity", "resolution",
                                                         "signal_to_noise")})
    pa = S.parents_table("S", cent, _load("ms2_monoterpene_a_summary.json")).set_index("spectrum")
    assert math.isnan(pa.loc["137.1321@hcd20.00", "precursor_mz"])
    assert "no peak at the precursor m/z" in pa.loc["137.1321@hcd20.00", "note"]
    assert pa.loc["137.1321@hcd9.00", "precursor_mz"] == pytest.approx(137.1323, abs=2e-4)


def _synthetic(parent, spectra):
    """A centroid answer of one parent: {activation: (mz list, intensity list)}."""
    return {S.spectrum_key(parent, act): {
        "parent_peak_mz": parent, "activation": act, "mz": list(mz), "intensity": list(inten),
        "resolution": [1e5] * len(mz), "signal_to_noise": [50.0] * len(mz)}
        for act, (mz, inten) in spectra.items()}


def test_precursor_is_located_in_the_lowest_energy_that_holds_it():
    """No candidate at NCE 1 (the precursor under the noise floor there): the
    precursor is located at NCE 5, the next energy that holds one."""
    cent = _synthetic(100.0, {"hcd1.00": ([60.0], [1.0]),
                              "hcd5.00": ([60.0, 100.0002], [1.0, 5.0]),
                              "hcd9.00": ([60.0, 100.0004], [3.0, 1.0])})
    p = S.parents_table("S", cent).set_index("spectrum")
    assert math.isnan(p.loc["100.0@hcd1.00", "precursor_mz"])
    assert p.loc["100.0@hcd1.00", "note"] == ("no peak at the precursor m/z (spent, or under the "
                                              "noise threshold)")
    assert p.loc["100.0@hcd5.00", "precursor_mz"] == 100.0002
    assert p.loc["100.0@hcd9.00", "precursor_mz"] == 100.0004
    assert p.loc["100.0@hcd9.00", "precursor_intensity"] == 1.0


def test_precursor_tracking_never_jumps_to_a_neighbour():
    """Followed within max(5 ppm, 0.5 mDa) of the located m/z: a peak 0.4 mDa
    off is the precursor, one 1.9 mDa off (the satellite distance seen in these
    files) is not; above m/z 100 the ppm clause widens it (5 ppm of 600 = 3 mDa)."""
    cent = _synthetic(100.0, {"hcd1.00": ([100.0], [10.0]), "hcd5.00": ([100.0004], [5.0]),
                              "hcd9.00": ([100.0019], [5.0])})
    p = S.parents_table("S", cent).set_index("spectrum")
    assert p.loc["100.0@hcd5.00", "precursor_mz"] == 100.0004
    assert math.isnan(p.loc["100.0@hcd9.00", "precursor_mz"])
    assert math.isnan(p.loc["100.0@hcd9.00", "precursor_intensity"])
    assert "no peak at the precursor m/z" in p.loc["100.0@hcd9.00", "note"]
    cent = _synthetic(600.0, {"hcd1.00": ([600.0], [10.0]), "hcd5.00": ([600.0028], [5.0]),
                              "hcd9.00": ([600.0032], [5.0])})
    p = S.parents_table("S", cent).set_index("spectrum")
    assert p.loc["600.0@hcd5.00", "precursor_mz"] == 600.0028
    assert math.isnan(p.loc["600.0@hcd9.00", "precursor_mz"])


def test_a_parent_whose_precursor_is_nowhere_says_so():
    """Standard A with the C10H17+ precursor removed from every 137.1321
    spectrum: NaN and the note on each of its rows, never a neighbour. A
    summary group with no spectrum keeps only its own note."""
    cent = _load("ms2_monoterpene_a_centroids.json")
    for key, e in cent.items():
        if e["parent_peak_mz"] == 137.1321:
            keep = [i for i, m in enumerate(e["mz"]) if abs(m - 137.1321) > 6e-3]
            cent[key] = dict(e, **{k: [e[k][i] for i in keep] for k in
                                   ("mz", "intensity", "resolution", "signal_to_noise")})
    summ = _load("ms2_monoterpene_a_summary.json")
    p = S.parents_table("S", cent, summ)
    gone = p[p["parent_mz"] == 137.1321]
    assert len(gone) == 5 and gone["precursor_mz"].isna().all()
    assert gone["precursor_intensity"].isna().all()
    assert (gone["note"] == "precursor not found near the parent key in any spectrum of this "
                            "parent").all()
    assert p.loc[p["parent_mz"] != 137.1321, "precursor_mz"].notna().all()
    # every spectrum of 155.1559 missing from the centroid answer: four summary-only rows
    # (no precursor to find in a spectrum that is not there)
    cent = {k: e for k, e in _load("ms2_monoterpene_a_centroids.json").items()
            if e["parent_peak_mz"] != 155.1559}
    p = S.parents_table("S", cent, summ)
    only = p[p["parent_mz"] == 155.1559]
    assert len(only) == 4 and (only["note"] == S.SUMMARY_ONLY_NOTE).all()
    assert only["note"].str.endswith("(the parent m/z and activation match no key): no peaks").all()


def test_ms2_scans_the_server_could_not_group():
    """The server counts every MS2 scan but groups (and serves a spectrum for)
    only those whose scan event it can read: none grouped is 'MS2 unreadable',
    never 'ok' and never blamed on the noise threshold; some ungrouped is
    noted on an 'ok' row; groups with an empty centroid answer say so."""
    unread = dict(_load("ms2_empty_scaffold_summary.json"), ms2_scan_count=57,
                  isolation_width=0.4)
    for row in (S.inventory_row("S", unread), S.inventory_row("S", unread, centroids={})):
        assert row["status"] == S.MS2_UNREADABLE and row["n_spectra"] == 0
        assert row["note"] == S.MS2_UNREADABLE_NOTE.format(n=57)
        assert "57 MS2 scans" in row["note"]
    store = _store()
    store[EMPTY]["summary"] = unread
    store[EMPTY]["centroids"] = {}
    res = S.extract_sample(FakeClient(store), EMPTY)
    assert res.row["status"] == S.MS2_UNREADABLE and res.row["ms2_scans"] == 57
    assert not any("noise threshold?" in n for n in res.row["note"].split(S.NOTE_SEP) + res.notes)
    # standard A's groups hold 78 scans; a summary counting 100 leaves 22 in no group
    summ = dict(_load("ms2_monoterpene_a_summary.json"), ms2_scan_count=100)
    row = S.inventory_row("S", summ)
    assert row["status"] == "ok"
    assert row["note"] == S.UNGROUPED_SCANS_NOTE.format(n=22, total=100)
    assert "22 of the 100 MS2 scans" in row["note"]
    assert S.inventory_row("S", _load("ms2_monoterpene_a_summary.json"))["note"] == ""
    # the summary's 12 groups and an empty centroid answer
    row = S.inventory_row("S", _load("ms2_monoterpene_a_summary.json"), centroids={})
    assert row["status"] == "ok" and row["n_spectra"] == 0
    assert row["note"] == S.NO_SPECTRUM_NOTE.format(k=12)
    assert "holds no spectrum for the summary's 12 group(s)" in row["note"]


def test_payload_arrays_are_checked_and_a_missing_one_is_nan():
    cent = _synthetic(100.0, {"hcd1.00": ([60.0, 80.0, 100.0], [1.0, 2.0, 3.0])})
    key = "100.0@hcd1.00"
    bad = {key: dict(cent[key], intensity=[1.0, 2.0])}
    with pytest.raises(ValueError, match=r"'100.0@hcd1.00': intensity has 2 values for 3 m/z"):
        S.spectra_table("S", bad)
    with pytest.raises(ValueError, match="signal_to_noise has 4 values"):
        S.parents_table("S", {key: dict(cent[key], signal_to_noise=[1.0] * 4)})
    # a list the server leaves out is unknown (NaN), never zero
    part = {key: {k: v for k, v in cent[key].items() if k != "resolution"}}
    sp = S.spectra_table("S", part)
    assert sp["resolution"].isna().all() and list(sp["intensity"]) == [1.0, 2.0, 3.0]
    assert list(sp["signal_to_noise"]) == [50.0] * 3
    ms1 = {"mz": [99.9, 100.0], "intensity": [1.0, 3.0], "resolution": [1e5, 1e5]}
    win = S.window_table("S", ms1, [100.0], 0.4)
    assert win["signal_to_noise"].isna().all() and list(win["share"]) == [0.25, 0.75]
    with pytest.raises(ValueError, match="MS1"):
        S.window_table("S", dict(ms1, intensity=[1.0]), [100.0], 0.4)


def test_order_dtypes_and_determinism():
    spectra, parents = _tables("ms2_nh4_exploratory")
    # the server lists the 197.13 groups as NCE 9, 7, 5, 3, 1; the tables go by NCE
    assert list(parents.loc[parents["parent_mz"] == 197.13, "nce"]) == [1, 3, 5, 7, 9]
    order = spectra.sort_values(["parent_mz", "nce", "mz"], kind="stable")
    assert spectra.reset_index(drop=True).equals(order.reset_index(drop=True))
    assert _dtypes(spectra) == SPECTRA_DTYPES and list(spectra.columns) == list(SPECTRA_DTYPES)
    assert _dtypes(parents) == PARENTS_DTYPES and list(parents.columns) == list(PARENTS_DTYPES)
    again = _tables("ms2_nh4_exploratory")
    assert spectra.equals(again[0]) and parents.equals(again[1])
    win = S.window_table("S", _load("ms2_nh4_exploratory_ms1.json"), [136.0213], 0.4)
    assert _dtypes(win) == WINDOW_DTYPES and list(win.columns) == list(WINDOW_DTYPES)
    inv = S.inventory_frame([S.inventory_row(NH4, _load("ms2_nh4_exploratory_summary.json"))])
    assert _dtypes(inv) == INVENTORY_DTYPES and list(inv.columns) == list(INVENTORY_DTYPES)


def test_a_spectrum_is_a_peak_list_assign_fragments_reads():
    """2026-09-10 file, the C11H18^NO2+ precursor ([C11H14O2 + ^NH4]+) at its
    highest NCE: the declustered C11H15O2+ (loss H3^N) and the dehydrated
    C11H13O+ (loss H2O + H3^N) read straight off the extracted spectrum."""
    spectra, parents = _tables("ms2_nh4_exploratory")
    p = parents[parents["parent_mz"] == 197.13]
    top = p.loc[p["nce"].idxmax()]
    assert top["spectrum"] == "197.13@hcd9.00"
    spec = spectra[spectra["spectrum"] == top["spectrum"]]
    off = mass_offset([(top["precursor_mz"], "C11H18^NO2+")]).offset_ppm
    best = best_readings(assign_fragments(spec, "C11H18^NO2+", offset_ppm=off))
    got = {(r.formula, r.loss_name) for r in best.itertuples() if r.formula}
    assert ("C11H15O2+", "H3^N") in got
    assert ("C11H13O+", "H2O + H3^N") in got
    assert ("C11H18^NO2+", "") in got
    assert set(best["mz"]) == set(spec["mz"])           # every peak of the spectrum is read


# ------------------------------------------------------------------ CLI ----
def _subparser(*path):
    p = cli.build_parser()
    for name in path:
        p = next(a for a in p._actions if a.__class__.__name__ == "_SubParsersAction"
                 ).choices[name]
    return p


def test_cli_parses_both_commands_and_renders_their_help():
    P = cli.build_parser()
    a = P.parse_args(["ms2", "list", "--batch", "B", "--dataset", "D"])
    assert a.func is cli.cmd_ms2 and a.ms2_verb == "list"
    assert (a.batch, a.dataset, a.sample_id, a.max_samples, a.out) == ("B", "D", None, 100, None)
    a = P.parse_args(["ms2", "list", "--sample-id", "X", "Y", "--out", "inv.csv"])
    assert a.sample_id == ["X", "Y"] and a.batch is None and a.out == "inv.csv"
    for bad in (["ms2", "list"], ["ms2", "list", "--batch", "B", "--sample-id", "X"],
                ["ms2"], ["ms2", "extract"]):
        with pytest.raises(SystemExit):
            P.parse_args(bad)
    a = P.parse_args(["ms2", "extract", "--sample-id", "X"])
    assert a.ms2_verb == "extract" and a.sample_id == ["X"]
    assert (a.noise_threshold, a.ms1_ppm, a.iso_width, a.out_dir) == (3.0, 1, None, None)
    assert a.noise_threshold == S.DEFAULT_NOISE_THRESHOLD
    a = P.parse_args(["ms2", "extract", "--sample-id", "X", "Y", "--noise-threshold", "5",
                      "--ms1-ppm", "2", "--iso-width", "0.4", "--out-dir", "base"])
    assert (a.noise_threshold, a.ms1_ppm, a.iso_width, a.out_dir) == (5.0, 2, 0.4, "base")
    for path in (("ms2",), ("ms2", "list"), ("ms2", "extract")):
        assert _subparser(*path).format_help().startswith("usage:")
    assert "--max-samples" in _subparser("ms2", "list").format_help()
    assert "--iso-width" in _subparser("ms2", "extract").format_help()


@pytest.fixture
def fake_server(monkeypatch):
    """The CLI against the fake server: credentials stubbed (CI has none), the
    environment holding sentinel ones nothing written may carry, back-off
    sleeps recorded instead of slept, and a batch listing of `client.listing`."""
    client = FakeClient()
    monkeypatch.setenv("MASCOPE_URL", SENTINEL_URL)
    monkeypatch.setenv("MASCOPE_ACCESS_TOKEN", SENTINEL_TOKEN)
    monkeypatch.setattr(cli, "_require_creds", lambda *a, **k: None)
    monkeypatch.setattr(IO, "connect", lambda *a, **k: client)
    client.sleeps = _record_sleeps(monkeypatch)
    client.listing = [NH4, MTA, MTB, EMPTY]

    def listing(c, batch, *, dataset=None, drop_columns=None):
        assert c is client and batch == "B"
        def row(s):          # a sample whose record answers an error: listed bare
            rec = client.store[s]["record"]
            return dict(rec if isinstance(rec, dict) else {"instrument_type": "orbi"},
                        sample_item_id=s)
        return pd.DataFrame([row(s) for s in client.listing])
    monkeypatch.setattr(IO, "fetch_batch_samples", listing)
    return client


def _read_inventory(path):
    return pd.read_csv(path, keep_default_na=False, dtype={"parents": str, "nce": str})


def test_cli_list_reads_one_summary_per_sample(fake_server, tmp_path, capsys):
    out = tmp_path / "inv" / "ms2_list.csv"
    assert cli.main(["ms2", "list", "--batch", "B", "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    df = _read_inventory(out)
    assert list(df.columns) == list(INVENTORY_DTYPES)
    assert list(df["sample_id"]) == [NH4, MTA, MTB, EMPTY]
    assert list(df["status"]) == ["ok", "ok", f"summary refused: {REASON}", "no MS2"]
    assert list(df["n_spectra"]) == [29, 12, 6, 0]
    assert list(df["n_parents"]) == [7, 3, 1, 0]
    # the refused sample's row names its parent and NCEs from the centroid keys
    mtb = df.set_index("sample_id").loc[MTB]
    assert (mtb["parents"], mtb["nce"], mtb["energy_resolved"]) == ("137.13", "1;5;10;15;25;35",
                                                                    True)
    assert mtb["isolation_width"] == "" and mtb["hidden_tail_s"] == pytest.approx(210.479)
    assert f"summary refused: {REASON}" in printed and "no MS2" in printed
    # the batch listing supplied the rows: no record fetch; the centroids only for the refusal
    assert fake_server.calls == [("summary", NH4, {}), ("summary", MTA, {}),
                                 ("summary", MTB, {}),
                                 ("centroids", MTB, {"noise_threshold": 10.0,
                                                     "by_activation": True}),
                                 ("summary", EMPTY, {})]
    # an 'ok' sample that runs past its last ingested scan says so too
    nh4 = df.set_index("sample_id").loc[NH4]
    assert nh4["status"] == "ok" and nh4["hidden_tail_s"] == pytest.approx(94.6895, abs=1e-3)
    assert nh4["note"] == S.HIDDEN_TAIL_NOTE.format(s=94.6895)
    assert "94.7 s past its last ingested scan" in nh4["note"]
    assert fake_server.sleeps == []
    with pytest.raises(SystemExit) as exc:
        cli.main(["ms2", "list", "--batch", "B", "--max-samples", "3"])
    # the refusal states every request a sample can cost
    assert "--max-samples" in str(exc.value) and "record" in str(exc.value)
    assert "centroids" in str(exc.value)
    with pytest.raises(SystemExit):
        cli.main(["ms2", "list", "--sample-id", NH4, "--dataset", "D"])


def test_cli_list_by_sample_id_reads_each_record(fake_server, capsys):
    assert cli.main(["ms2", "list", "--sample-id", MTB]) == 0
    printed = capsys.readouterr().out
    assert "210.5 s past its last ingested scan" in printed
    assert [c[0] for c in fake_server.calls] == ["record", "summary", "centroids"]
    help_text = " ".join(_subparser("ms2", "list").format_help().split())
    assert "record with --sample-id" in help_text and "centroids when" in help_text


def test_cli_list_records_a_failing_sample_and_goes_on(fake_server, tmp_path, capsys):
    """A TOF sample is asked nothing; a sample the server answers HTTP 500 for
    is asked once, gets an 'error' row without the URL, and the listing goes on
    and is written; the command then exits 1."""
    fake_server.listing = [NH4, TOF, BAD, MTA]
    out = tmp_path / "ms2_list.csv"
    with pytest.raises(SystemExit) as exc:
        cli.main(["ms2", "list", "--batch", "B", "--out", str(out)])
    assert "1 of 4" in str(exc.value) and BAD in str(exc.value)
    df = _read_inventory(out)
    assert list(df["sample_id"]) == [NH4, TOF, BAD, MTA]
    assert list(df["status"]) == ["ok", S.NOT_ORBITRAP, "error: ServerError", "ok"]
    bad = df.set_index("sample_id").loc[BAD]
    assert bad["note"] == "[HTTP 500] Failed to get ms2 summary. Unexpected error (ref: 0f1e2d3c)"
    assert df.set_index("sample_id").loc[TOF, "note"] == S.NOT_ORBITRAP_NOTE == (
        "the server reads MS2 from Orbitrap raw files only (this sample's instrument type is "
        "'tof'): nothing asked")
    assert [(c[0], c[1]) for c in fake_server.calls] == [
        ("summary", NH4), ("summary", BAD), ("summary", MTA)]
    assert fake_server.sleeps == []                       # a 500 is never retried
    printed = capsys.readouterr().out
    assert "server.example" not in printed and "://" not in printed


def test_cli_list_stops_when_the_server_blocks(fake_server, tmp_path):
    """A WAF block (403 after the back-off) is not the sample's: asking the
    next ones only extends it, so the listing stops there, writes what it read
    and says how many were not asked."""
    fake_server.listing = [NH4, WAF, MTA, EMPTY]
    out = tmp_path / "ms2_list.csv"
    with pytest.raises(SystemExit) as exc:
        cli.main(["ms2", "list", "--batch", "B", "--out", str(out)])
    assert f"stopped at {WAF}" in str(exc.value) and "2 later sample(s)" in str(exc.value)
    df = _read_inventory(out)
    assert list(df["sample_id"]) == [NH4, WAF]
    assert list(df["status"]) == ["ok", "error: AuthenticationError"]
    assert [c[1] for c in fake_server.calls] == [NH4] + [WAF] * 4
    assert fake_server.sleeps == [2.0, 4.0, 8.0]


def test_cli_list_stops_when_the_connection_fails(fake_server, tmp_path, capsys):
    """A server that cannot be reached (the SDK's MascopeConnectionError /
    MascopeTimeoutError, after its own four attempts) is not the sample's: the
    listing stops there, asks it once (no second round of back-off), and
    neither the CSV nor the console carries the host."""
    out = tmp_path / "ms2_list.csv"
    with pytest.raises(SystemExit) as exc:
        cli.main(["ms2", "list", "--sample-id", NH4, DOWN, MTA, "--out", str(out)])
    assert f"stopped at {DOWN}" in str(exc.value) and "1 later sample(s)" in str(exc.value)
    assert [(c[0], c[1]) for c in fake_server.calls] == [("record", NH4), ("summary", NH4),
                                                         ("record", DOWN)]
    assert fake_server.sleeps == []
    df = _read_inventory(out)
    assert list(df["sample_id"]) == [NH4, DOWN]
    assert list(df["status"]) == ["ok", "error: MascopeConnectionError"]
    text = out.read_text() + capsys.readouterr().out
    assert "sentinel-host" not in text and "://" not in text
    assert "Failed to resolve '<host>'" in text
    # a timeout from a batch listing's sample stops it the same way
    fake_server.calls.clear()
    fake_server.store[MTB]["summary"] = _connect_timeout(MTB)
    fake_server.listing = [NH4, MTB, MTA, EMPTY]
    with pytest.raises(SystemExit) as exc:
        cli.main(["ms2", "list", "--batch", "B", "--out", str(out)])
    assert f"stopped at {MTB}" in str(exc.value) and "2 later sample(s)" in str(exc.value)
    assert [(c[0], c[1]) for c in fake_server.calls] == [("summary", NH4), ("summary", MTB)]
    df = _read_inventory(out)
    assert list(df["status"]) == ["ok", "error: MascopeTimeoutError"]
    text = out.read_text() + capsys.readouterr().out
    assert "sentinel-host" not in text and "Connection to <host> timed out" in text


def test_cli_a_stale_id_is_that_samples_error(fake_server, tmp_path, capsys):
    """The API's own 403 -- how the server answers an unknown or stale sample
    id -- is final and the sample's: asked once, recorded, the others go on,
    and the hint is to re-list the ids, never to wait out a WAF block."""
    out = tmp_path / "ms2_list.csv"
    with pytest.raises(SystemExit) as exc:
        cli.main(["ms2", "list", "--sample-id", NH4, STALE, MTA, "--out", str(out)])
    assert "1 of 3" in str(exc.value) and STALE in str(exc.value)
    assert "stopped" not in str(exc.value)
    assert [(c[0], c[1]) for c in fake_server.calls] == [
        ("record", NH4), ("summary", NH4), ("record", STALE), ("record", MTA), ("summary", MTA)]
    assert fake_server.sleeps == []
    df = _read_inventory(out)
    assert list(df["status"]) == ["ok", "error: AuthenticationError", "ok"]
    printed = capsys.readouterr().out
    assert cli.MS2_API_403_HINT in printed and "WAF" not in printed
    # extract: the same
    fake_server.calls.clear()
    with pytest.raises(SystemExit) as exc:
        cli.main(["ms2", "extract", "--sample-id", STALE, MTA, "--out-dir", str(tmp_path)])
    assert "1 of 2" in str(exc.value)
    assert {p.name for p in (tmp_path / "ms2").iterdir()} == {MTA}
    assert cli.MS2_API_403_HINT in capsys.readouterr().out


def test_cli_the_servers_compute_timeout_is_that_samples_error(fake_server, tmp_path, capsys):
    """The MS2 endpoints' own 504 ('MS2 summary timed out after 120s') is asked
    once and recorded; the listing goes on. A 500 for an id that spells '403'
    prints no WAF advice."""
    fake_server.store[MTB]["summary"] = _compute_504(MTB)
    fake_server.store[BAD]["summary"] = ServerError(
        "Failed to get ms2 summary. Unexpected error (ref: x)", status_code=500,
        url="https://server.example/api/samples/SAMPLE403xxxxxxx/ms2/summary")
    fake_server.listing = [MTB, BAD, MTA]
    with pytest.raises(SystemExit) as exc:
        cli.main(["ms2", "list", "--batch", "B"])
    assert "2 of 3" in str(exc.value) and "stopped" not in str(exc.value)
    assert [(c[0], c[1]) for c in fake_server.calls] == [("summary", MTB), ("summary", BAD),
                                                         ("summary", MTA)]
    assert fake_server.sleeps == []
    printed = capsys.readouterr().out
    assert cli.MS2_COMPUTE_TIMEOUT_HINT in printed
    assert "WAF" not in printed and "rate-limiting" not in printed


def test_cli_extract_stops_when_the_server_blocks(fake_server, tmp_path):
    """A WAF block (403 after the back-off) stops `ms2 extract` as it stops
    `ms2 list`: the later samples are never asked; a dropped connection too."""
    # only io_mascope's back-off is recorded: a sleep elsewhere in the process
    # (subprocess polling while the manifest's git_info runs git) is not
    time.sleep(0)
    assert fake_server.sleeps == []
    with pytest.raises(SystemExit) as exc:
        cli.main(["ms2", "extract", "--sample-id", NH4, WAF, MTA, "--out-dir", str(tmp_path)])
    assert f"stopped at {WAF}" in str(exc.value) and "1 later sample(s)" in str(exc.value)
    assert MTA not in {c[1] for c in fake_server.calls}
    assert [c[0] for c in fake_server.calls if c[1] == WAF] == ["record"] + ["summary"] * 4
    assert fake_server.sleeps == [2.0, 4.0, 8.0]
    assert {p.name for p in (tmp_path / "ms2").iterdir()} == {NH4}
    fake_server.calls.clear()
    with pytest.raises(SystemExit) as exc:
        cli.main(["ms2", "extract", "--sample-id", DOWN, MTA, "--out-dir", str(tmp_path)])
    assert f"stopped at {DOWN}" in str(exc.value) and "1 later sample(s)" in str(exc.value)
    assert [(c[0], c[1]) for c in fake_server.calls] == [("record", DOWN)]


def test_cli_extract_writes_the_tables_and_the_manifest(fake_server, tmp_path, capsys):
    assert cli.main(["ms2", "extract", "--sample-id", NH4, MTB, EMPTY, OLD,
                     "--out-dir", str(tmp_path)]) == 0
    printed = capsys.readouterr().out
    names = {"ms2_spectra.parquet", "ms2_spectra.csv", "ms2_parents.csv", "ms2_window.csv",
             "ms2_manifest.json"}
    for sid in (NH4, MTB, EMPTY, OLD):
        assert {p.name for p in (tmp_path / "ms2" / sid).iterdir()} == names, sid
    d = tmp_path / "ms2" / NH4
    spectra = pd.read_parquet(d / "ms2_spectra.parquet")
    assert len(spectra) == 249 and spectra["spectrum"].nunique() == 29
    assert _dtypes(spectra) == SPECTRA_DTYPES
    assert len(pd.read_csv(d / "ms2_spectra.csv")) == 249
    assert len(pd.read_csv(d / "ms2_parents.csv")) == 29
    win = pd.read_csv(d / "ms2_window.csv")
    assert win["parent_mz"].nunique() == 7 and win["is_parent"].sum() == 7
    assert win["offset_da"].abs().max() > 0.155            # the server's 0.4-Da window
    man = json.loads((d / "ms2_manifest.json").read_text())
    assert man["sample_id"] == NH4 and man["status"] == "ok" and man["energy_resolved"]
    assert man["server"]["hidden_tail_s"] == pytest.approx(94.69, abs=1e-3)
    assert S.HIDDEN_TAIL_NOTE.format(s=94.6895) in man["notes"]
    assert man["server"]["summary_refused"] is None and man["server"]["isolation_width"] == 0.4
    assert (man["parameters"]["noise_threshold"], man["parameters"]["ms1_ppm"]) == (3.0, 1)
    assert man["parameters"]["isolation_width_source"] == "server"
    assert man["parameters"]["isolation_width_used"] == 0.4
    assert man["counts"] == {"parents": 7, "spectra": 29, "parents_rows": 29, "peaks": 249,
                             "window_rows": len(win)}
    from peaky.reporting import provenance as PV
    assert man["files"]["ms2_spectra.parquet"] == PV.sha1_file(str(d / "ms2_spectra.parquet"))
    assert man["code"]["sources_version"] == S.__version__
    # the refused sample: recorded, spectra written, window empty with its reason
    mb = json.loads((tmp_path / "ms2" / MTB / "ms2_manifest.json").read_text())
    assert mb["server"]["summary_refused"] == REASON
    assert mb["status"] == f"summary refused: {REASON}" and mb["energy_resolved"]
    assert mb["server"]["hidden_tail_s"] == pytest.approx(210.479, abs=1e-3)
    assert mb["server"]["isolation_width"] is None
    assert any("--iso-width" in n for n in mb["notes"])
    # each note comes back whole (none holds the separator the note cell joins them by)
    assert S.HIDDEN_TAIL_NOTE.format(s=210.479058) in mb["notes"]
    assert len(pd.read_csv(tmp_path / "ms2" / MTB / "ms2_window.csv")) == 0
    me = json.loads((tmp_path / "ms2" / EMPTY / "ms2_manifest.json").read_text())
    assert me["status"] == "no MS2" and me["counts"]["spectra"] == 0
    assert me["energy_resolved"] is False
    # a server older than 1.8: its manifest never claims energy-resolved spectra
    mo = json.loads((tmp_path / "ms2" / OLD / "ms2_manifest.json").read_text())
    assert mo["status"] == "energy-averaged server" and mo["energy_resolved"] is False
    assert ("server returns energy-averaged MS2 only (Mascope < 1.8): per-energy spectra and "
            "breakdown curves need a newer server") in mo["notes"]
    assert man["energy_resolved"] is True and mb["energy_resolved"] is True
    # every written table carries its whole contract, in order
    for sid in (NH4, MTB, EMPTY, OLD):
        d = tmp_path / "ms2" / sid
        for name, want in (("ms2_spectra.csv", SPECTRA_DTYPES), ("ms2_parents.csv", PARENTS_DTYPES),
                           ("ms2_window.csv", WINDOW_DTYPES)):
            assert list(pd.read_csv(d / name, nrows=0).columns) == list(want), (sid, name)
        assert list(pd.read_parquet(d / "ms2_spectra.parquet").columns) == list(SPECTRA_DTYPES)
    # ... and its values: NCE and volts per spectrum, the window's shares, the notes
    sp = pd.read_csv(tmp_path / "ms2" / NH4 / "ms2_spectra.csv")
    first = sp[sp["spectrum"] == "136.0213@hcd1.00"]
    assert len(first) and (first["nce"] == 1).all() and (first["hcd_v"] == 4.05).all()
    assert (first["kind"] == "hcd").all() and first["energy_resolved"].all()
    assert np.allclose(win.groupby("parent_mz")["share"].sum(), 1.0)
    pb = pd.read_csv(tmp_path / "ms2" / MTB / "ms2_parents.csv", keep_default_na=False)
    assert len(pb) == 6 and pb["note"].str.startswith(f"summary refused ({REASON})").all()
    po = pd.read_csv(tmp_path / "ms2" / OLD / "ms2_parents.csv", keep_default_na=False)
    assert po["note"].str.contains("not a voltage the instrument applied", regex=False).all()
    assert f"summary refused: {REASON}" in printed and "no window inventory" in printed
    # the console line names a mode only for spectra it wrote
    lines = {ln.split(":")[0].split()[-1]: ln for ln in printed.splitlines()
             if ln.startswith("[ms2] SAMPLE")}
    assert "(energy-resolved)" in lines[NH4] and "(energy-resolved)" in lines[MTB]
    assert "(energy-averaged)" in lines[OLD]
    assert lines[EMPTY].startswith(f"[ms2] {EMPTY}: no MS2; 0 spectra over 0 parents, 0 peaks")
    assert "energy-" not in lines[EMPTY]


@pytest.mark.parametrize("argv, message", [
    (["--sample-id", NH4, "--iso-width", "0"], "--iso-width must be a positive width in Da"),
    (["--sample-id", NH4, "--iso-width", "-0.4"], "--iso-width must be a positive width in Da"),
    (["--sample-id", NH4, "--iso-width", "nan"], "--iso-width must be a positive width in Da"),
    (["--sample-id", NH4, "--noise-threshold", "-1"], "--noise-threshold must be >= 0"),
    (["--sample-id", NH4, "--ms1-ppm", "0"], "--ms1-ppm must be a positive integer"),
    # an id names its own output folder
    (["--sample-id", NH4, "../elsewhere"], "--sample-id '../elsewhere' is not a sample id"),
    (["--sample-id", ".."], "--sample-id '..' is not a sample id"),
    (["--sample-id", "a/b"], "--sample-id 'a/b' is not a sample id"),
])
def test_cli_extract_checks_its_arguments_before_asking(fake_server, tmp_path, argv, message):
    """A bad argument is refused up front, by its own message, before any
    request and before anything is written."""
    with pytest.raises(SystemExit) as exc:
        cli.main(["ms2", "extract", *argv, "--out-dir", str(tmp_path)])
    assert message in str(exc.value)
    assert fake_server.calls == []
    assert not (tmp_path / "ms2").exists()


def test_cli_extract_keeps_the_spectra_when_the_ms1_call_fails(fake_server, tmp_path, capsys):
    """The server's MS1 timeout on one sample costs that sample its window
    only: its MS2 tables are written, the note says why, the next sample is
    read, and the command succeeds. --no-window skips the MS1 request."""
    err = _MS1_FAILURES[1](NH4)
    fake_server.store[NH4]["ms1"] = err
    assert cli.main(["ms2", "extract", "--sample-id", NH4, MTA, "--out-dir", str(tmp_path)]) == 0
    printed = capsys.readouterr().out
    d = tmp_path / "ms2" / NH4
    assert len(pd.read_parquet(d / "ms2_spectra.parquet")) == 249
    assert len(pd.read_csv(d / "ms2_parents.csv")) == 29
    win = pd.read_csv(d / "ms2_window.csv")
    assert len(win) == 0 and list(win.columns) == list(WINDOW_DTYPES)
    man = json.loads((d / "ms2_manifest.json").read_text())
    assert man["status"] == "ok" and man["counts"]["window_rows"] == 0
    note = S.WINDOW_FAILED_NOTE.format(error=f"ServerError: {IO.server_error_text(err)}")
    assert note in man["notes"] and f"note: {note}" in printed
    assert "MS1 centroid extraction timed out after 120s" in note
    assert len(pd.read_csv(tmp_path / "ms2" / MTA / "ms2_window.csv"))
    assert [c[0] for c in fake_server.calls if c[1] == NH4] == ["record", "summary", "centroids",
                                                               "ms1"]
    # --no-window: no MS1 request at all
    fake_server.calls.clear()
    assert cli.main(["ms2", "extract", "--sample-id", MTA, "--no-window",
                     "--out-dir", str(tmp_path)]) == 0
    assert [c[0] for c in fake_server.calls] == ["record", "summary", "centroids"]
    man = json.loads((tmp_path / "ms2" / MTA / "ms2_manifest.json").read_text())
    assert man["counts"]["window_rows"] == 0 and man["counts"]["spectra"] == 12
    assert "window inventory not asked for (--no-window): no MS1 request" in man["notes"]
    assert cli.build_parser().parse_args(["ms2", "extract", "--sample-id", "X"]).no_window is False


def test_the_console_hint_goes_by_the_status_never_the_messages_digits(capsys):
    """A 500's '(ref: <8 hex>)' can spell 401 / 403 / 404: it gets no WAF,
    token or not-found advice. A status that is what it says gets its hint; an
    error without one (a connection failure) is read by its words only."""
    url = "https://server.example/api/samples/SAMPLE1xxxxxxxxx/ms2/summary"
    for ref in ("4031c9ae", "9a4010ff", "ab404cde", "0f1e2d3c"):
        exc = ServerError(f"Failed to get ms2 summary. Unexpected error (ref: {ref})",
                          status_code=500, url=url)
        assert cli._ms2_failure(NH4, exc, IO) is True
        out = capsys.readouterr().out
        assert f"(ref: {ref})" in out and out.rstrip().endswith(f"(ref: {ref})"), out
    for exc, hint in ((AuthenticationError("Not authenticated", status_code=401, url=url),
                       "MASCOPE_ACCESS_TOKEN is likely expired"),
                      (_waf_403(NH4), "Cloudflare WAF 403"),
                      (NotFoundError("Sample not found", status_code=404, url=url),
                       "Not found on the server"),
                      (_api_403(NH4), cli.MS2_API_403_HINT),
                      (_compute_504(NH4), "recomputes them from scratch")):
        cli._ms2_failure(NH4, exc, IO)
        assert hint in capsys.readouterr().out, exc
    # urllib3 quotes a memory address, whose hex can spell 403
    timeout = MascopeTimeoutError(
        "Request timed out: HTTPSConnectionPool(host='sentinel-host.invalid', port=443): Max "
        "retries exceeded (Caused by ConnectTimeoutError(<HTTPSConnection at 0x7f4031c9ae10>, "
        "'Connection to sentinel-host.invalid timed out. (connect timeout=30)'))",
        url="https://sentinel-host.invalid/api/samples/x/ms2/summary")
    assert cli._ms2_failure(NH4, timeout, IO) is False
    out = capsys.readouterr().out
    assert "0x7f4031c9ae10" in out and "WAF" not in out and "token" not in out
    # ... while a status-less error's words still count
    text = "Request failed: Attention Required! | Cloudflare"
    assert "Cloudflare WAF 403" in cli._ms2_hint(MascopeConnectionError(text), text, IO)


def test_cli_extract_passes_every_knob_to_the_server(fake_server, tmp_path):
    """--noise-threshold, --ms1-ppm and --iso-width reach the server calls and
    the window, and the manifest records what was used; an explicit width
    replaces the server's 0.4 Da."""
    assert cli.main(["ms2", "extract", "--sample-id", NH4, "--noise-threshold", "3",
                     "--ms1-ppm", "2", "--iso-width", "0.3", "--out-dir", str(tmp_path)]) == 0
    assert fake_server.calls == [("record", NH4, {}), ("summary", NH4, {}),
                                 ("centroids", NH4, {"noise_threshold": 3.0,
                                                     "by_activation": True}),
                                 ("ms1", NH4, {"ppm": 2})]
    d = tmp_path / "ms2" / NH4
    man = json.loads((d / "ms2_manifest.json").read_text())
    assert (man["parameters"]["noise_threshold"], man["parameters"]["ms1_ppm"]) == (3.0, 2)
    assert man["parameters"]["iso_width"] == 0.3
    assert man["parameters"]["isolation_width_used"] == 0.3
    assert man["parameters"]["isolation_width_source"] == "explicit"
    assert man["server"]["isolation_width"] == 0.4          # the server's, kept apart
    win = pd.read_csv(d / "ms2_window.csv")
    assert len(win) and win["offset_da"].abs().max() <= 0.155


def test_cli_extract_goes_on_past_a_failing_sample(fake_server, tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["ms2", "extract", "--sample-id", NH4, BAD, TOF, MTA,
                  "--out-dir", str(tmp_path)])
    assert "1 of 4" in str(exc.value) and BAD in str(exc.value)
    assert {p.name for p in (tmp_path / "ms2").iterdir()} == {NH4, TOF, MTA}
    mt = json.loads((tmp_path / "ms2" / TOF / "ms2_manifest.json").read_text())
    assert mt["status"] == S.NOT_ORBITRAP and mt["counts"]["spectra"] == 0
    assert [c for c in fake_server.calls if c[1] in (BAD, TOF)] == [
        ("record", BAD, {}), ("summary", BAD, {}), ("record", TOF, {})]
    printed = capsys.readouterr().out
    assert f"[ms2] {BAD}: error: ServerError: [HTTP 500]" in printed
    assert "server.example" not in printed


def test_nothing_written_carries_the_server_or_the_credentials(fake_server, tmp_path):
    """Not the host, not the token -- whether from the environment, the client
    or an error's URL -- in any table, CSV, parquet or manifest."""
    fake_server.listing = [NH4, MTB, EMPTY, BAD, DOWN]
    with pytest.raises(SystemExit):
        cli.main(["ms2", "list", "--batch", "B", "--out", str(tmp_path / "list.csv")])
    with pytest.raises(SystemExit):
        cli.main(["ms2", "extract", "--sample-id", NH4, MTB, EMPTY, BAD, OLD, DOWN,
                  "--out-dir", str(tmp_path)])
    # the unreachable sample's row is written, without the host it could not reach
    assert DOWN in (tmp_path / "list.csv").read_text()
    files = [f for f in tmp_path.rglob("*") if f.is_file()]
    assert len(files) == 1 + 4 * 5
    banned = ("sentinel-host", SENTINEL_TOKEN, "server.example", "ACCESS_TOKEN")
    for f in files:
        texts = [f.read_bytes().decode("latin-1")]
        if f.suffix == ".parquet":
            texts.append(pd.read_parquet(f).to_csv())
        else:
            assert "://" not in texts[0], f.name
        for text in texts:
            for b in banned:
                assert b not in text, (f.name, b)
