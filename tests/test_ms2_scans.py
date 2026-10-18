"""Offline tests for the per-scan MS2 source (peaky/ms2/scans.py).

Everything here runs on a deterministic SYNTHETIC `/metadata` payload of a
targeted-MS2 (inclusion-list) file, built by `make_tms2` the way the
instrument and the server write one:

  * K experiments x n list rows, one NCE per experiment, events numbered
    1..n, n+1..2n, ... in experiment order, the Full Scan the last event
    (K*n + 1) and the first time block;
  * inside each experiment's time block the events cycle in list order, and
    a block may end mid-cycle (the next block starts at its first event);
  * HighMass = 1.02 x (precursor + width / 2) + 20 (the "define first mass"
    scan range), or one fixed range for every event;
  * an automatic first mass that moves LowMass in one block;
  * garbled scan filters (random analyzers, polarities and 'Full ms' on MS2
    scans) over a correct trailer, or sane ones;
  * trailer rows dropped (headerless scans): the first scan, both sides of a
    block switch, a run inside a block, the last scan;
  * known peaks (a precursor that fragments with the NCE, two fragments) with
    m/z and intensity jitter, noise centroids below S/N 3, and one scan whose
    fragment is written as two centroids 0.8 ppm apart.
"""
import json
import math
import re

import numpy as np
import pandas as pd
import pytest

from peaky.ms2 import scans as SC
from peaky.ms2 import sources as SRC

R200 = 120000.0
PREC = (120.0808, 137.1325, 155.1430, 181.0859, 203.1430)
NCES = (5.0, 15.0, 30.0)
WIDTH = 0.7
NOISE = 100.0
GARBLED = ("ITMS - c EI Full ms", "TQMS - c EI Full ms", "SQMS + c FABI Full ms",
           "MS + EI Full ms", "Sector - c Full ms", "ITMS + c Full ms")
#: (full cycles, extra events) per experiment's time block: blocks end mid-cycle
CYCLES = ((2, 2), (2, 3), (2, 1))
FS_SCANS = 4
#: the fragments every precursor loses (neutral masses)
LOSSES = (18.010565, 42.010565)

#: The table contract, spelled out here (not read back from the module).
SCAN_DTYPES = {
    "scan": "int64", "t_s": "float64", "has_header": "bool", "order": "float64",
    "event": "float64", "node": "str", "nce": "float64", "hcd_v": "float64",
    "iso_width": "float64", "low_mass": "float64", "high_mass": "float64",
    "inj_time_ms": "float64", "max_inj_time_ms": "float64", "microscans": "float64",
    "agc_fill": "float64", "tic": "float64", "ft_resolution": "float64",
    "filter": "str", "filter_ok": "bool", "filter_parent_mz": "float64",
    "n_centroids": "int64", "min_centroid_mz": "float64", "max_centroid_mz": "float64",
    "inferred": "bool", "parent_mz": "float64", "precursor_source": "str",
    "experiment": "float64", "slot": "float64", "spectrum": "str", "note": "str"}


def _dtypes(df):
    return {c: ("str" if t == object or str(t) in ("str", "string") else str(t))
            for c, t in df.dtypes.items()}


def _high(p, width=WIDTH):
    return 1.02 * (p + width / 2) + 20


def _res(m):
    return np.round(R200 * np.sqrt(200.0 / np.asarray(m, dtype=float)), -2)


def default_drops(seq):
    """Scan numbers whose trailer row is dropped: the first scan, the last scan
    of experiment 1 and the first of experiment 2 (both sides of a block
    switch), a run of three inside experiment 3, and the last scan."""
    first_of = {}
    last_of = {}
    for s, (kind, k, _) in enumerate(seq, start=1):
        first_of.setdefault((kind, k), s)
        last_of[(kind, k)] = s
    e3 = first_of[("ms2", 2)]
    return {1, last_of[("ms2", 0)], first_of[("ms2", 1)], e3 + 4, e3 + 5, e3 + 6, len(seq)}


def scan_sequence(n=None, cycles=CYCLES, fs_scans=FS_SCANS, blocks=None):
    """[(kind, experiment index, slot index)] in time order: the Full Scan block,
    then each experiment's time block cycling its list from the first row.
    `n`: the rows of every experiment's list, or one count per experiment.
    `blocks` = [(experiment index, full cycles, extra events[, first slot])]
    in time order replaces the default (every experiment once, in event
    order, `cycles`); ('fs', scans, 0) puts a Full Scan block there."""
    n = len(PREC) if n is None else n
    sizes = list(n) if isinstance(n, (list, tuple)) else None
    seq = [("fs", None, None)] * fs_scans
    for b in (blocks if blocks is not None else [(k, *c) for k, c in enumerate(cycles)]):
        k, full, extra = b[:3]
        if k == "fs":
            seq += [("fs", None, None)] * full
            continue
        start = b[3] if len(b) > 3 else 0
        nk = sizes[k] if sizes is not None else n
        seq += [("ms2", k, (start + j) % nk) for j in range(full * nk + extra)]
    return seq


def make_tms2(*, prec=PREC, nces=NCES, cycles=CYCLES, fs_scans=FS_SCANS, garble=1.0,
              drops="default", fixed_range=False, auto_low_block=1, split_scan=True,
              seed=7, blocks=None, width=WIDTH, precs=None):
    """(payload, truth): a synthetic `/metadata` payload and, per scan, what it
    really is (order, event, experiment, slot, precursor, nce, spectrum key).
    `garble`: the share of rows whose filter is garbled (deterministic).
    `precs`: one precursor tuple per experiment (lists of different lengths,
    events numbered on through them; the Full Scan event their sum + 1)
    instead of `prec` shared by every experiment."""
    rng = np.random.default_rng(seed)
    K = len(nces)
    precs = [tuple(prec)] * K if precs is None else [tuple(x) for x in precs]
    assert len(precs) == K
    sizes = [len(x) for x in precs]
    first_ev = [int(x) for x in np.concatenate(([0], np.cumsum(sizes)))]
    seq = scan_sequence(sizes, cycles, fs_scans, blocks)
    drop = default_drops(seq) if drops == "default" else set(drops or ())
    fs_event = first_ev[-1] + 1
    rows, data, times, truth = {}, [], [], []
    t = 3.0
    garbled_rows = set(rng.choice(len(seq), round(garble * len(seq)), replace=False).tolist())
    split_at = None
    for s, (kind, k, j) in enumerate(seq, start=1):
        if kind == "fs":
            mz = []
            inten = []
            for i, p in enumerate(dict.fromkeys(q for x in precs for q in x)):
                mz += [p, p + 0.2, p + 1.00336]
                inten += [1e6 * (1 + 0.1 * i), 5e4, 1.1e5 * (1 + 0.1 * i)]
            mz += list(rng.uniform(45, 590, 6))
            inten += list(rng.uniform(150, 250, 6))
            mz = np.asarray(mz) * (1 + rng.normal(0, 0.2e-6, len(mz)))
            inten = np.asarray(inten) * rng.normal(1, 0.03, len(mz))
            low, high = 40.0, 600.0
            row = {"Scan Node:": "Full Scan", "HCD Energy:": "", "HCD Energy V:": "",
                   "Scan Event:": fs_event, "MS2 Isolation Width:": 560.0}
            sane = "FTMS + p NSI Full ms"
            truth.append({"scan": s, "order": 1, "event": fs_event, "experiment": None,
                          "slot": None, "parent_mz": math.nan, "nce": math.nan, "spectrum": ""})
            dt = 0.3
        else:
            p, nce = precs[k][j], nces[k]
            ev = first_ev[k] + j + 1
            surv = math.exp(-nce / 15.0)
            mz = [p, p - LOSSES[0], p - LOSSES[1]]
            inten = [1e5 * surv, 5e4 * (1 - surv) + 400, 2e4 * (1 - surv) + 300]
            low = 40.0 if k != auto_low_block else 50.0 + 1.3 * j
            high = 600.0 if fixed_range else _high(p, width)
            mz += list(rng.uniform(low + 1, p - 1, 3))
            inten += list(rng.uniform(100, 250, 3))
            mz = np.asarray(mz) * (1 + rng.normal(0, 0.3e-6, len(mz)))
            inten = np.asarray(inten) * rng.normal(1, 0.05, len(mz))
            if split_scan and split_at is None and k == 1 and j == 1:
                # one scan writes its first fragment as two centroids 0.8 ppm apart
                split_at = s
                m0, i0 = mz[1], inten[1]
                mz = np.concatenate([mz, [m0 * (1 + 0.4e-6)]])
                mz[1] = m0 * (1 - 0.4e-6)
                inten = np.concatenate([inten, [i0 / 2]])
                inten[1] = i0 / 2
            row = {"Scan Node:": "tMS2", "HCD Energy:": f"{nce:.2f}",
                   "HCD Energy V:": f"{4.0 + nce * p / 1000:.2f}", "Scan Event:": ev,
                   "MS2 Isolation Width:": width}
            sane = f"FTMS + p NSI t ms2 {p:.4f}@hcd{nce:.2f}"
            truth.append({"scan": s, "order": 2, "event": ev, "experiment": k + 1, "slot": j + 1,
                          "parent_mz": p, "nce": nce,
                          "spectrum": SRC.spectrum_key(p, f"hcd{nce:.2f}")})
            dt = 1.2
        o = np.argsort(mz)
        mz, inten = np.asarray(mz)[o], np.asarray(inten)[o]
        filt = (GARBLED[s % len(GARBLED)] if (s - 1) in garbled_rows else sane)
        row.update({"ScanType": f"{filt} [{low:.4f}-{high:.4f}]", "MsType": "Ms",
                    "StartTime": t / 60.0, "LowMass": low, "HighMass": high,
                    "TIC": float(inten.sum()), "Ion Injection Time (ms):": 50.0,
                    "Max. Ion Time (ms):": 100.0, "AGC Fill:": 0.5, "Micro Scan Count:": 10,
                    "FT Resolution:": int(R200), "Monoisotopic M/Z:": 0.0})
        if s not in drop:
            rows[str(s)] = row
        data.append({"mzs": mz.tolist(), "intensities": inten.tolist(),
                     "resolutions": _res(mz).tolist(),
                     "noises": (NOISE * rng.normal(1, 0.02, len(mz))).tolist()})
        times.append(t)
        t += dt
    payload = {"num_of_scans": len(seq), "stats_per_scan": rows,
               "stats_per_file": {"Value": {}}, "centroids_meta": {"time": times, "data": data}}
    tr = pd.DataFrame(truth)
    tr.attrs["split_scan"] = split_at
    tr.attrs["dropped"] = sorted(drop)
    return payload, tr


def the_list(prec=PREC):
    return pd.DataFrame({"Compound": [f"ion{i + 1}" for i in range(len(prec))],
                         "Formula": [""] * len(prec), "Adduct": ["(no adduct)"] * len(prec),
                         "Precursor (m/z)": list(prec), "Precursor Charge (z)": [1] * len(prec)})


def _list_frame(prec=PREC):
    return SC.inclusion_list_frame(the_list(prec))


def _check_truth(scans, truth, *, cols=("order", "event", "parent_mz", "nce", "spectrum")):
    m = truth.merge(scans, on="scan", suffixes=("_true", ""))
    for c in cols:
        a, b = m[f"{c}_true"], m[c]
        if c == "spectrum":
            assert (a == b).all(), m.loc[a != b, ["scan", f"{c}_true", c]]
        else:
            ok = np.isclose(a.astype(float), b.astype(float), equal_nan=True, atol=1e-4)
            assert ok.all(), m.loc[~ok, ["scan", f"{c}_true", c]]


# ---------------------------------------------------------------------------
# the scan table and the filters
# ---------------------------------------------------------------------------
def test_scan_table_reads_the_trailer_never_the_filter():
    meta, truth = make_tms2()
    st = SC.scan_table(meta)
    assert _dtypes(st) == SCAN_DTYPES
    assert list(st["scan"]) == list(range(1, len(truth) + 1))
    head = st[st["has_header"]]
    # every garbled MS2 row still reads order 2 and its NCE off the trailer
    m = head.merge(truth, on="scan", suffixes=("", "_true"))
    assert (m["order"] == m["order_true"]).all()
    assert np.allclose(m["nce"], m["nce_true"], equal_nan=True)
    assert (m["event"] == m["event_true"]).all()
    # the dropped rows: no header, nothing read, centroids counted
    dropped = st[~st["has_header"]]
    assert sorted(dropped["scan"]) == truth.attrs["dropped"]
    assert dropped["order"].isna().all() and dropped["event"].isna().all()
    assert (dropped["n_centroids"] > 0).all()
    # times are the payload's seconds; centroid range per scan
    assert st["t_s"].iloc[0] == pytest.approx(3.0)
    assert (st["max_centroid_mz"] <= st["high_mass"].fillna(np.inf) + SC.RANGE_CHECK_DA).all()
    # a garbled filter never agrees: no MS2 precursor from it
    assert not head.loc[head["order"] == 2, "filter_ok"].any()
    assert head["filter_parent_mz"].isna().all()


def test_filter_verdicts_and_polarity():
    garbled, _ = make_tms2(garble=1.0)
    clean, _ = make_tms2(garble=0.0)
    some, _ = make_tms2(garble=0.3)
    v = SC.filter_verdict(SC.scan_table(garbled))
    assert v["filters_readable"] == SC.FILTERS_NO and v["ms2_filters_ok"] == 0
    assert SC.filter_polarity(SC.scan_table(garbled)) is None
    st = SC.scan_table(clean)
    v = SC.filter_verdict(st)
    assert v["filters_readable"] == SC.FILTERS_YES
    assert v["filters_ok"] == v["header_rows"] and v["polarities_in_filters"] == "+"
    assert SC.filter_polarity(st) == "+"
    ms2 = st[st["order"] == 2]
    assert np.allclose(ms2["filter_parent_mz"], [round(p, 4) for p in ms2["filter_parent_mz"]])
    v = SC.filter_verdict(SC.scan_table(some))
    assert v["filters_readable"] == SC.FILTERS_PARTLY
    # a partly garbled file can carry a sane-looking '-' row: polarity is still not read
    assert SC.filter_polarity(SC.scan_table(some)) is None


@pytest.mark.parametrize("text,ok", [
    ("FTMS + p NSI t ms2 137.1325@hcd15.00 [40.0000-180.2934]", True),
    ("FTMS + p NSI t ms2 137.1325@hcd5.00 [40.0000-180.2934]", False),     # energy != trailer
    ("FTMS + p NSI t ms2 137.1325@cid15.00 [40.0000-180.2934]", False),    # activation
    ("FTMS + p NSI Full ms [40.0000-180.2934]", False),                    # MS order
    ("ITMS + c NSI t ms2 137.1325@hcd15.00 [40.0000-180.2934]", False),    # analyzer
    ("FTMS p NSI t ms2 137.1325@hcd15.00 [40.0000-180.2934]", False),      # no polarity
    ("FTMS + p NSI t ms2 137.1325@hcd15.00 [40.0000-190.0000]", False),    # range
])
def test_a_filter_agrees_with_its_trailer_or_is_not_used(text, ok):
    f = SC.parse_filter(text)
    assert SC._filter_ok(f, 2.0, 15.0, 40.0, 180.29339, R200) is ok


def test_a_stepped_energy_is_ms2_with_no_single_nce():
    meta, _ = make_tms2(drops=())
    meta["stats_per_scan"]["10"]["HCD Energy:"] = "20.00,40.00"
    st = SC.scan_table(meta)
    r = st[st["scan"] == 10].iloc[0]
    assert r["order"] == 2 and math.isnan(r["nce"]) and "stepped" in r["note"]
    with pytest.raises(SC.ScanMapError, match="stepped"):
        SC.map_precursors(st, [_list_frame()])


def test_the_payload_envelope_is_unwrapped_and_checked():
    meta, _ = make_tms2()
    assert len(SC.scan_table({"data": meta})) == len(SC.scan_table(meta))
    bad = json.loads(json.dumps(meta))
    bad["centroids_meta"]["time"].pop()
    with pytest.raises(ValueError, match="scan times"):
        SC.scan_table(bad)
    with pytest.raises(ValueError, match="no per-scan centroids"):
        SC.scan_table({"num_of_scans": 0})


# ---------------------------------------------------------------------------
# lists and the method layout
# ---------------------------------------------------------------------------
def test_inclusion_list_reader(tmp_path):
    p = tmp_path / "list.csv"
    the_list().to_csv(p, index=False, encoding="utf-8-sig")       # an Excel-saved CSV
    lf = SC.read_inclusion_list(str(p))
    assert list(lf.columns) == ["row", "mz", "compound"]
    assert list(lf["row"]) == [1, 2, 3, 4, 5] and np.allclose(lf["mz"], PREC)
    assert lf["compound"].iloc[0] == "ion1"
    x = tmp_path / "list.xlsx"
    the_list().to_excel(x, index=False)
    assert np.allclose(SC.read_inclusion_list(str(x))["mz"], PREC)
    for col in ("Mass [m/z]", "m/z", "mz"):
        q = tmp_path / f"alt{len(col)}.csv"
        pd.DataFrame({col: PREC}).to_csv(q, index=False)
        assert np.allclose(SC.read_inclusion_list(str(q))["mz"], PREC)
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"Compound": ["a", "b"], "Precursor (m/z)": [120.0, None]}).to_csv(bad, index=False)
    with pytest.raises(ValueError, match="row"):
        SC.read_inclusion_list(str(bad))
    none = tmp_path / "none.csv"
    pd.DataFrame({"Compound": ["a"]}).to_csv(none, index=False)
    with pytest.raises(ValueError, match="no precursor m/z column"):
        SC.read_inclusion_list(str(none))


def test_events_number_the_rows_experiment_by_experiment():
    meta, _ = make_tms2()
    st = SC.scan_table(meta)
    lay = SC.method_layout(st, [_list_frame()])
    n = len(PREC)
    assert [(b.first, b.last, b.nce) for b in lay.blocks] == [
        (k * n + 1, (k + 1) * n, nce) for k, nce in enumerate(NCES)]
    assert lay.full_scan.first == len(NCES) * n + 1
    # one list per experiment, in event order, is the same layout
    lay3 = SC.method_layout(st, [_list_frame()] * 3)
    assert [(b.first, b.last) for b in lay3.blocks] == [(b.first, b.last) for b in lay.blocks]
    # no list: the blocks are read off the NCE runs, sized by the Full Scan event
    lay0 = SC.method_layout(st)
    assert [(b.first, b.last, b.nce) for b in lay0.blocks] == [
        (b.first, b.last, b.nce) for b in lay.blocks]


@pytest.mark.parametrize("lists,match", [
    (lambda: [_list_frame(PREC[:4])],                                    # a row missing
     r"15 MS2 event\(s\) before its Full Scan \(event 16\), not a whole number of 4-row"),
    (lambda: [_list_frame(PREC + (219.1,))],                             # a row extra
     r"15 MS2 event\(s\) before its Full Scan \(event 16\), not a whole number of 6-row"),
    (lambda: [_list_frame()] * 2,                                        # an experiment short
     r"the Full Scan is event 16, but the 2 lists end at event 10: the lists do not fit"),
])
def test_a_list_of_the_wrong_length_is_refused(lists, match):
    st = SC.scan_table(make_tms2()[0])
    with pytest.raises(SC.ScanMapError, match=match):
        SC.map_precursors(st, lists())


def test_lists_that_stop_short_of_the_full_scan_are_refused():
    """One list each for experiments 1 and 2 of a 3-experiment method whose
    experiment 3 lost every trailer row (and experiment 2's block ends on a
    full cycle): no event seen lies past the lists, so only the Full Scan
    event (16, not 10 + 1) says a list is missing -- refused, never accepted
    with experiment 3's scans walked on into experiment 2's spectra."""
    cycles = ((2, 2), (2, 0), (2, 1))
    seq = scan_sequence(cycles=cycles)
    exp3 = [s for s, x in enumerate(seq, 1) if x[:2] == ("ms2", 2)]
    meta, _ = make_tms2(drops=exp3, auto_low_block=None, cycles=cycles)
    st = SC.scan_table(meta)
    assert int(st["event"].max()) == 16 and int(st.loc[st["order"] == 2, "event"].max()) == 10
    with pytest.raises(SC.ScanMapError, match=r"the Full Scan is event 16, but the 2 lists end "
                                              r"at event 10: the lists do not fit the method"):
        SC.map_precursors(st, [_list_frame()] * 2)


def test_a_list_longer_than_the_method_says_what_would_fit():
    """A list with more rows than the method has MS2 events: the refusal names
    the method's MS2 events and, from its NCE runs, the list that would fit --
    never an 'event 0'."""
    st = SC.scan_table(make_tms2()[0])
    with pytest.raises(SC.ScanMapError) as e:
        SC.map_precursors(st, [_list_frame(PREC * 4)])
    msg = str(e.value)
    assert "event 0" not in msg
    assert ("the method has 15 MS2 event(s) before its Full Scan (event 16), not a whole number "
            "of 20-row experiments: this 20-row list does not fit the method") in msg
    assert ("its events read 3 runs of one NCE (NCE 5/15/30): a shared list of 5 rows, or 3 "
            "lists summing to 15 rows, would fit") in msg


def test_a_list_in_the_wrong_order_is_refused_by_the_scan_range_rule():
    st = SC.scan_table(make_tms2()[0])
    swapped = list(PREC)
    swapped[1], swapped[2] = swapped[2], swapped[1]
    with pytest.raises(SC.ScanMapError) as e:
        SC.map_precursors(st, [_list_frame(tuple(swapped))])
    msg = str(e.value)
    assert "event 2 (row 2: 155.1430" in msg and "implies 137.1325" in msg
    assert "wrong list, rows in the wrong order" in msg
    # the check can be switched off for a method that sets its range otherwise
    _, rep = SC.map_precursors(st, [_list_frame(tuple(swapped))], rule_check=False)
    assert rep["rule"]["events_off"] and not rep["rule"]["checked"]


def test_an_experiment_with_two_nces_is_refused():
    meta, _ = make_tms2(drops=())
    row = next(r for r in meta["stats_per_scan"].values() if r["Scan Event:"] == 3)
    row["HCD Energy:"] = "7.00"
    with pytest.raises(SC.ScanMapError, match="one NCE"):
        SC.map_precursors(SC.scan_table(meta), [_list_frame()])


def test_the_list_must_agree_with_readable_filters():
    st = SC.scan_table(make_tms2(garble=0.0)[0])
    shifted = tuple(p + 0.01 for p in PREC)
    with pytest.raises(SC.ScanMapError, match="disagrees with the scan filters"):
        SC.map_precursors(st, [_list_frame(shifted)], rule_check=False)


# ---------------------------------------------------------------------------
# precursors: filter > list > scan-range rule
# ---------------------------------------------------------------------------
def test_garbled_file_with_its_list_maps_every_scan():
    meta, truth = make_tms2()
    scans, rep = SC.map_precursors(SC.scan_table(meta), [_list_frame()])
    _check_truth(scans, truth)
    ms2 = scans[scans["order"] == 2]
    assert set(ms2["precursor_source"]) == {SC.PRECURSOR_LIST}
    assert rep["rule"]["applies"] and rep["rule"]["checked"]
    assert rep["rule"]["slope"] == pytest.approx(1.02, abs=1e-6)
    assert rep["rule"]["offset_da"] == pytest.approx(20.0, abs=1e-4)
    assert rep["rule"]["max_residual_mda"] < 0.01
    m = truth.merge(scans, on="scan", suffixes=("_true", ""))
    m2 = m[m["order_true"] == 2]
    assert (m2["experiment"] == m2["experiment_true"]).all()
    assert (m2["slot"] == m2["slot_true"]).all()


def test_garbled_file_without_a_list_derives_the_precursor_from_the_scan_range():
    meta, truth = make_tms2()
    scans, rep = SC.map_precursors(SC.scan_table(meta))
    _check_truth(scans[scans["has_header"]], truth[truth["scan"].isin(scans.loc[
        scans["has_header"], "scan"])])
    assert set(scans.loc[scans["order"] == 2, "precursor_source"]) == {SC.PRECURSOR_RULE}
    assert rep["layout"] == "events"
    # no list pins the layout: the headerless scans at the file's edges, at
    # the block switch and on a wrap stay unassigned (the list places them)
    assert not rep["layout_closed"] and _mis_keyed(scans, truth) == []
    inf = rep["inference"]
    assert (inf["inferred"], inf["unassigned_scans"]) == (0, truth.attrs["dropped"])


def test_readable_filters_name_the_precursor_first():
    meta, truth = make_tms2(garble=0.0)
    scans, _ = SC.map_precursors(SC.scan_table(meta), [_list_frame()])
    _check_truth(scans, truth)
    head = scans[(scans["order"] == 2) & scans["has_header"]]
    assert set(head["precursor_source"]) == {SC.PRECURSOR_FILTER}
    # a headerless scan of such an event takes the event's filter reading too
    inferred = scans[(scans["order"] == 2) & scans["inferred"]]
    assert len(inferred) and set(inferred["precursor_source"]) == {SC.PRECURSOR_FILTER}


def test_partly_readable_filters_mix_sources_per_event():
    meta, truth = make_tms2(garble=0.6, seed=3)
    scans, rep = SC.map_precursors(SC.scan_table(meta), [_list_frame()])
    _check_truth(scans, truth)
    assert {SC.PRECURSOR_FILTER, SC.PRECURSOR_LIST} <= set(rep["precursor_sources"])


def test_filters_that_measure_the_precursor_scan_by_scan_key_one_parent():
    meta, truth = make_tms2(garble=0.0, drops=())
    rng = np.random.default_rng(1)
    for r in meta["stats_per_scan"].values():
        if r["Scan Node:"] == "tMS2":
            p = float(r["ScanType"].split("ms2 ")[1].split("@")[0])
            r["ScanType"] = r["ScanType"].replace(f"{p:.4f}@", f"{p + rng.uniform(-3e-4, 3e-4):.4f}@")
    scans, rep = SC.map_precursors(SC.scan_table(meta))
    # every scan keeps its own reading, and the readings of one ion are one parent
    _check_truth(scans, truth, cols=("order", "event", "nce"))
    ms2 = scans[scans["order"] == 2]
    assert set(ms2["precursor_source"]) == {SC.PRECURSOR_FILTER}
    assert ms2["spectrum"].nunique() == len(PREC) * len(NCES)
    assert np.allclose(sorted(ms2["parent_mz"].unique()), PREC, atol=3e-4)
    assert any("one parent" in n for n in rep["notes"])


def test_an_event_whose_filters_name_several_precursors_maps_scan_by_scan():
    # a data-dependent event: one event, a different precursor per cycle
    meta, _ = make_tms2(garble=0.0, drops=(), fixed_range=True)
    rows = meta["stats_per_scan"]
    ev2 = sorted(int(k) for k, r in rows.items() if r["Scan Event:"] == 2)
    r0 = rows[str(ev2[0])]
    r0["ScanType"] = r0["ScanType"].replace(f"{PREC[1]:.4f}@", f"{PREC[3]:.4f}@")
    scans, _ = SC.map_precursors(SC.scan_table(meta))
    got = scans.set_index("scan")
    assert got.at[ev2[0], "parent_mz"] == pytest.approx(PREC[3])
    assert got.at[ev2[1], "parent_mz"] == pytest.approx(PREC[1])
    assert got.at[ev2[0], "spectrum"] == SRC.spectrum_key(PREC[3], f"hcd{NCES[0]:.2f}")


def test_list_rows_within_a_millidalton_are_one_parent():
    prec = (PREC[0], PREC[1], PREC[1] + 0.0004, PREC[3], PREC[4])
    meta, _ = make_tms2(prec=prec)
    scans, rep = SC.map_precursors(SC.scan_table(meta), [_list_frame(prec)])
    ms2 = scans[scans["order"] == 2]
    assert ms2["parent_mz"].nunique() == 4
    assert round(float(np.median([prec[1], prec[2]])), 4) in set(ms2["parent_mz"])
    assert any("one parent" in n for n in rep["notes"])
    assert SC.cluster_parents([1.0, 1.0009, 1.0018, 1.003]) == {
        1.0: 1.0009, 1.0009: 1.0009, 1.0018: 1.0009, 1.003: 1.003}


def test_a_fixed_scan_range_derives_nothing():
    meta, truth = make_tms2(fixed_range=True)
    st = SC.scan_table(meta)
    with pytest.raises(SC.ScanMapError, match="pass the method's inclusion list") as e:
        SC.map_precursors(st)
    assert e.value.needs_list and "a fixed range" in str(e.value)
    scans, rep = SC.map_precursors(st, [_list_frame()])
    assert not rep["rule"]["applies"]
    assert rep["rule"]["experiments"] == {1: SC.RULE_FIXED, 2: SC.RULE_FIXED, 3: SC.RULE_FIXED}
    # what the list IS checked against there: everything but the scan range
    assert any("the list rows there are checked against the Full Scan event, the experiments' "
               "NCEs and any readable filters, not against the scan range" in n
               for n in rep["notes"])
    # with one range for every event the walk is checked against that range only
    _check_truth(scans, truth)


def test_a_moved_first_mass_is_kept_per_scan_and_the_rule_still_holds():
    meta, truth = make_tms2(auto_low_block=1)
    scans, rep = SC.map_precursors(SC.scan_table(meta), [_list_frame()])
    blk = scans[(scans["experiment"] == 2) & scans["has_header"]]
    assert (blk["low_mass"] > 40).all()
    assert set(np.round(scans.loc[scans["experiment"] == 1, "low_mass"], 3)) == {40.0}
    assert not rep["rule"]["events_off"]
    _check_truth(scans, truth)


# ---------------------------------------------------------------------------
# headerless scans: the cycle walk
# ---------------------------------------------------------------------------
def test_every_headerless_scan_is_placed_by_the_cycle_including_block_switches():
    meta, truth = make_tms2()
    scans, rep = SC.map_precursors(SC.scan_table(meta), [_list_frame()])
    inf = rep["inference"]
    dropped = truth.attrs["dropped"]
    assert inf["headerless"] == len(dropped) == inf["inferred"]
    assert inf["unassigned"] == 0
    assert sorted(scans.loc[scans["inferred"], "scan"]) == dropped
    _check_truth(scans, truth)
    # an inferred scan takes its event's range, width and volts
    r = scans[scans["inferred"] & (scans["order"] == 2)].iloc[0]
    assert math.isfinite(r["high_mass"]) and r["iso_width"] == pytest.approx(WIDTH)


def test_a_leading_run_walks_back_in_the_first_block():
    # the file opens on experiment 1 (its Full Scan block comes later), its
    # first two rows lost: the list and the Full Scan event pin every block
    blocks = [(0, 2, 2), ("fs", FS_SCANS, 0), (1, 2, 3), (2, 2, 1)]
    meta, truth = make_tms2(fs_scans=0, blocks=blocks, drops={1, 2})
    scans, rep = SC.map_precursors(SC.scan_table(meta), [_list_frame()])
    assert rep["layout_closed"] and rep["inference"]["inferred"] == 2
    _check_truth(scans, truth)
    assert scans.loc[scans["scan"] <= 2, "slot"].tolist() == [1.0, 2.0]
    # no Full Scan event: the list cannot say how many experiments the method
    # has, and one it does not know of could fill the leading run
    meta, _ = make_tms2(fs_scans=0, drops={1, 2})
    scans, rep = SC.map_precursors(SC.scan_table(meta), [_list_frame()])
    assert not rep["layout_closed"]
    assert rep["inference"]["inferred"] == 0 and rep["inference"]["unassigned_scans"] == [1, 2]
    assert any("not pinned" in n for n in rep["notes"])


def test_an_empty_headerless_scan_takes_its_walk():
    meta, truth = make_tms2()
    s = truth.attrs["dropped"][3]                     # inside experiment 3's run
    for k in ("mzs", "intensities", "resolutions", "noises"):
        meta["centroids_meta"]["data"][s - 1][k] = []
    scans, rep = SC.map_precursors(SC.scan_table(meta), [_list_frame()])
    assert rep["inference"]["unassigned"] == 0
    _check_truth(scans, truth)


def test_a_gap_the_cycle_allows_two_ways_stays_unassigned():
    seq = scan_sequence()
    first2 = next(s for s, x in enumerate(seq, 1) if x[:2] == ("ms2", 1))
    # 3 scans before the switch and 6 after: the run spans more than a cycle
    # of the new block, so two switch positions fit
    drops = set(range(first2 - 3, first2 + 6))
    meta, _ = make_tms2(drops=drops)
    scans, rep = SC.map_precursors(SC.scan_table(meta), [_list_frame()])
    inf = rep["inference"]
    assert inf["runs_ambiguous"] == 1 and inf["unassigned"] == len(drops)
    assert set(inf["unassigned_scans"]) == drops
    assert scans.loc[scans["scan"].isin(drops), "spectrum"].eq("").all()
    assert any("left unassigned" in n for n in rep["notes"])


def test_a_walk_the_centroids_contradict_is_not_used():
    meta, truth = make_tms2()
    # a dropped scan inside experiment 3 holds a centroid above its event's HighMass
    s = truth.attrs["dropped"][4]
    meta["centroids_meta"]["data"][s - 1]["mzs"].append(599.0)
    for k in ("intensities", "resolutions", "noises"):
        meta["centroids_meta"]["data"][s - 1][k].append(1000.0)
    _, rep = SC.map_precursors(SC.scan_table(meta), [_list_frame()])
    inf = rep["inference"]
    assert inf["runs_failed_range"] == 1
    # the whole run of three is left out, the other runs are placed
    assert inf["unassigned"] == 3 and inf["inferred"] == len(truth.attrs["dropped"]) - 3


def test_a_known_pair_that_breaks_the_cycle_stops_all_inference():
    meta, _ = make_tms2()
    # scan 8's trailer is a copy of the next event's: two consecutive known
    # scans skip an event, and the cycle model does not hold
    rows = meta["stats_per_scan"]
    ev8 = rows["8"]["Scan Event:"]
    rows["8"] = dict(next(r for r in rows.values() if r["Scan Event:"] == ev8 + 1))
    _, rep = SC.map_precursors(SC.scan_table(meta), [_list_frame()], rule_check=False)
    inf = rep["inference"]
    assert inf["model_breaks"] >= 1 and inf["inferred"] == 0
    assert inf["unassigned"] == inf["headerless"]


def test_blocks_that_interleave_in_time_stop_all_inference():
    once = [(0, 1, 0), (1, 1, 0), (2, 1, 0)]
    meta, _ = make_tms2(blocks=once, drops={3, 12})
    _, rep = SC.map_precursors(SC.scan_table(meta), [_list_frame()])
    assert rep["inference"]["inferred"] == 2
    # experiment 1 runs again after experiment 3: a block could hide in any run
    meta, _ = make_tms2(blocks=once + [(0, 1, 0)], drops={3, 12})
    _, rep = SC.map_precursors(SC.scan_table(meta), [_list_frame()])
    assert rep["inference"]["model_breaks"] == 1 and rep["inference"]["inferred"] == 0
    assert rep["inference"]["unassigned"] == 2
    assert any("return to a block" in n for n in rep["notes"])


def test_an_ordinary_file_with_ms1_rows_only_infers_nothing():
    meta, _ = make_tms2(garble=0.0, drops=())
    rows = meta["stats_per_scan"]
    for k in [k for k, r in rows.items() if r["Scan Node:"] == "tMS2"]:
        del rows[k]
    st = SC.scan_table(meta)
    scans, rep = SC.map_precursors(st)
    assert rep["inference"]["inferred"] == 0
    assert rep["inference"]["unassigned"] == int((~st["has_header"]).sum())
    assert (scans["spectrum"] == "").all()
    assert any("no MS2 event has a trailer row" in n for n in rep["notes"])


def test_a_short_block_with_no_trailer_row_is_never_walked_over():
    # experiment 2 runs three scans, all headerless: a walk on in experiment 1
    # would fit the run and the scan ranges, and still be wrong
    blocks = [(0, 1, 2), (1, 0, 3), (2, 1, 0)]
    exp2 = set(range(FS_SCANS + 8, FS_SCANS + 11))
    meta, _ = make_tms2(blocks=blocks, drops=exp2)
    scans, rep = SC.map_precursors(SC.scan_table(meta), [_list_frame()])
    assert rep["inference"]["runs_ambiguous"] == 1
    assert rep["inference"]["unassigned"] == 3
    assert scans.loc[scans["scan"].isin(exp2), "spectrum"].eq("").all()


def test_an_experiment_with_no_trailer_row_leaves_switch_runs_unassigned():
    seq = scan_sequence()
    exp2 = {s for s, x in enumerate(seq, 1) if x[:2] == ("ms2", 1)}
    meta, _ = make_tms2(drops=exp2)
    scans, rep = SC.map_precursors(SC.scan_table(meta), [_list_frame()])
    assert any("experiment 2" in n and "NCE is unknown" in n for n in rep["notes"])
    assert scans.loc[scans["scan"].isin(exp2), "spectrum"].eq("").all()
    assert rep["inference"]["runs_ambiguous"] >= 1


def _mis_keyed(scans, truth):
    """The inferred scans placed on another event or spectrum than they are."""
    m = truth.merge(scans, on="scan", suffixes=("_true", ""))
    inf = m[m["inferred"]]
    wrong = (inf["order"] != inf["order_true"]) | (
        (inf["order_true"] == 2) & ((inf["event"] != inf["event_true"])
                                    | ((inf["spectrum"] != "")
                                       & (inf["spectrum"] != inf["spectrum_true"]))))
    return sorted(inf.loc[wrong, "scan"].astype(int))


def test_a_layout_read_off_the_events_walks_only_straight_runs():
    """No list and no Full Scan to size the blocks: experiment boundaries are
    the events seen, and an event never seen at a block's edge belongs to no
    block -- so a walk that wraps, or walks back from the first scan, could
    land on the wrong event. Only a straight run inside one block is placed."""
    # one 4-row experiment, no Full Scan, its event 1 never seen with a trailer
    seq = scan_sequence([4], fs_scans=0, blocks=[(0, 3, 1)])
    ev1 = {s for s, x in enumerate(seq, 1) if x == ("ms2", 0, 0)}
    meta, truth = make_tms2(precs=[PREC[:4]], nces=(5.0,), fs_scans=0, blocks=[(0, 3, 1)],
                            drops=ev1 | {7})
    scans, rep = SC.map_precursors(SC.scan_table(meta))
    assert _mis_keyed(scans, truth) == []
    assert not rep["layout_closed"] and rep["experiments"][0]["first_event"] == 2
    # the event-1 scans (scan 1 leads, the others wrap) stay unassigned; the
    # straight run 2 -> [3] -> 4 (scan 7) is placed
    assert set(rep["inference"]["unassigned_scans"]) == ev1
    assert scans.loc[scans["scan"] == 7, ["inferred", "event"]].iloc[0].tolist() == [True, 3.0]


def test_an_unseen_edge_event_with_unequal_lists_is_never_walked_onto():
    # lists of 5 / 3 / 4 rows, no list given, experiment 2's first event never
    # seen: the NCE runs read 1-5 / 7-8 / 9-12, so a walk across the switch
    # would put that event's scans in experiment 1 (event 1, NCE 5)
    precs = [PREC, PREC[:3], PREC[:4]]
    seq = scan_sequence([5, 3, 4], blocks=[(0, 2, 1), (1, 3, 1), (2, 2, 2)])
    ev6 = {s for s, x in enumerate(seq, 1) if x == ("ms2", 1, 0)}
    meta, truth = make_tms2(precs=precs, blocks=[(0, 2, 1), (1, 3, 1), (2, 2, 2)], drops=ev6)
    scans, rep = SC.map_precursors(SC.scan_table(meta))
    assert _mis_keyed(scans, truth) == []
    assert not rep["layout_closed"]
    assert set(rep["inference"]["unassigned_scans"]) == ev6
    # with the three lists the layout is pinned and every scan is placed
    lists = [_list_frame(p) for p in precs]
    scans, rep = SC.map_precursors(SC.scan_table(meta), lists)
    assert rep["layout_closed"] and rep["inference"]["unassigned"] == 0
    _check_truth(scans, truth)


@pytest.mark.parametrize("with_list", [False, True])
def test_a_short_experiment_with_no_trailer_row_is_never_walked_over_without_a_list(with_list):
    # the payload of test_a_short_block_with_no_trailer_row_is_never_walked_over,
    # with and without the list: no list cannot know experiment 2 exists
    blocks = [(0, 1, 2), (1, 0, 3), (2, 1, 0)]
    exp2 = set(range(FS_SCANS + 8, FS_SCANS + 11))
    meta, truth = make_tms2(blocks=blocks, drops=exp2)
    scans, rep = SC.map_precursors(SC.scan_table(meta), [_list_frame()] if with_list else None)
    assert _mis_keyed(scans, truth) == []
    assert scans.loc[scans["scan"].isin(exp2), "spectrum"].eq("").all()
    assert rep["inference"]["unassigned"] == 3


@pytest.mark.parametrize("fs_scans,with_list", [(FS_SCANS, False), (FS_SCANS, True), (0, True)])
def test_a_last_experiment_with_no_trailer_row_stays_unassigned(fs_scans, with_list):
    # experiment 3 runs two scans, both without a trailer row: a walk on in
    # experiment 2 fits their scan ranges (its precursors are the larger)
    blocks = [(0, 2, 2), (1, 2, 3), (2, 0, 2)]
    seq = scan_sequence(fs_scans=fs_scans, blocks=blocks)
    exp3 = {s for s, x in enumerate(seq, 1) if x[:2] == ("ms2", 2)}
    meta, truth = make_tms2(fs_scans=fs_scans, blocks=blocks, drops=exp3, auto_low_block=None)
    scans, rep = SC.map_precursors(SC.scan_table(meta), [_list_frame()] if with_list else None)
    assert _mis_keyed(scans, truth) == []
    assert set(rep["inference"]["unassigned_scans"]) == exp3


def test_a_headerless_full_scan_block_is_not_walked_into_experiment_1():
    # a fixed range and every Full Scan row lost: no Full Scan event, so the
    # MS1 scans before experiment 1 could be anything -- never its events
    meta, truth = make_tms2(fixed_range=True, drops=set(range(1, FS_SCANS + 1)))
    scans, rep = SC.map_precursors(SC.scan_table(meta), [_list_frame()])
    assert _mis_keyed(scans, truth) == []
    assert rep["inference"]["unassigned_scans"] == list(range(1, FS_SCANS + 1))


POOL = (99.0441, 113.0597, 120.0808, 137.1325, 141.0910, 151.0754, 155.1430, 159.1016,
        169.0859, 175.0965, 181.0859, 203.1430, 219.1380, 233.1536)


def test_seeded_fuzz_never_mis_keys_a_scan():
    """Random methods (1-3 experiments of 2-5 rows, shared or own lists, a Full
    Scan block or none, anywhere in time), random rows lost, sometimes an
    event never seen: an inferred scan is on its true event and spectrum, or
    left unassigned -- never on another."""
    import random
    rnd = random.Random(20261017)
    inferred = 0
    for trial in range(200):
        K = rnd.randint(1, 3)
        precs = [tuple(sorted(rnd.sample(POOL, rnd.randint(2, 5)))) for _ in range(K)]
        shared = rnd.random() < 0.4
        if shared:
            precs = [precs[0]] * K
        sizes = [len(p) for p in precs]
        blocks = [(k, rnd.randint(0, 2), rnd.randint(1, sizes[k])) for k in range(K)]
        if rnd.random() < 0.3:
            rnd.shuffle(blocks)
        fs = rnd.choice([0, 3])
        if fs and rnd.random() < 0.3:
            blocks.insert(rnd.randint(0, K), ("fs", fs, 0))
            fs = 0
        seq = scan_sequence(sizes, fs_scans=fs, blocks=blocks)
        rate = rnd.choice([0.1, 0.2, 0.3])
        drops = {s for s in range(1, len(seq) + 1) if rnd.random() < rate}
        if rnd.random() < 0.3:                # an event never seen with a trailer
            k = rnd.randrange(K)
            j = rnd.randrange(sizes[k])
            drops |= {s for s, x in enumerate(seq, 1) if x == ("ms2", k, j)}
        meta, truth = make_tms2(precs=precs, nces=tuple(5.0 + 9 * k for k in range(K)),
                                fs_scans=fs, blocks=blocks, drops=drops, seed=trial,
                                split_scan=False, auto_low_block=None)
        lists = rnd.choice([None, "lists"] + (["shared"] if shared else []))
        lists = (None if lists is None else [_list_frame(precs[0])] if lists == "shared"
                 else [_list_frame(p) for p in precs])
        try:
            scans, rep = SC.map_precursors(SC.scan_table(meta), lists)
        except SC.ScanMapError:
            continue
        assert _mis_keyed(scans, truth) == [], (trial, precs, blocks, sorted(drops), fs, lists)
        inferred += rep["inference"]["inferred"]
    assert inferred > 200                    # 247 scans placed over the 200 methods


def test_seeded_fuzz_with_shared_nces_skipped_events_and_hidden_blocks():
    """The cycle model's own edges: adjacent experiments at one NCE (one NCE
    run without a list); the instrument skipping one event once inside a
    block (e2 -> e4), both scans either side of the skip without a trailer
    row; a short block of another experiment hidden inside a block's time
    (no trailer row on any of its scans, the block resuming at its first
    row). Never a scan on another event or spectrum. A hidden block whose
    length completes the interrupted block's cycle is left out: its scans
    are then exactly what the block itself would have run, and nothing in
    the file tells them apart (docs/MS2.md §12.7)."""
    import random
    rnd = random.Random(20261018)
    counts = {"inferred": 0, "skips": 0, "hidden": 0, "shared_nce": 0, "inconsistent": 0}
    for trial in range(300):
        K = rnd.randint(1, 3)
        precs = [tuple(sorted(rnd.sample(POOL, rnd.randint(2, 5)))) for _ in range(K)]
        if rnd.random() < 0.4:
            precs = [precs[0]] * K
        sizes = [len(p) for p in precs]
        nces = [5.0 + 9 * k for k in range(K)]
        for k in range(1, K):
            if rnd.random() < 0.5:
                nces[k] = nces[k - 1]
        counts["shared_nce"] += len(set(nces)) < K
        blocks = [(k, rnd.randint(1, 2), rnd.randint(0, sizes[k] - 1), 0) for k in range(K)]
        k = max(range(K), key=lambda x: sizes[x])
        i = next(n for n, b in enumerate(blocks) if b[0] == k)
        n_k = sizes[k]
        end = (blocks[i][1] * n_k + blocks[i][2] - 1) % n_k      # the block's last slot
        mode = rnd.choice(["skip", "hide", "none"])
        if mode == "skip" and n_k > 2:                # ... e(end), then e(end + 2)
            blocks.insert(i + 1, (k, 1, 0, (end + 2) % n_k))
        elif mode == "hide" and K > 1:
            j = rnd.choice([x for x in range(K) if x != k])
            L = rnd.choice([n for n in range(1, 5) if (end + n + 1) % n_k])
            blocks[i + 1:i + 1] = [(j, 0, L, 0), (k, 1, 0, 0)]
        else:
            mode = "none"
        fs = rnd.choice([0, 3])
        seq = scan_sequence(sizes, fs_scans=fs, blocks=blocks)
        span, s0 = [], fs + 1                         # each block's first and last scan
        for b in blocks:
            n_b = b[1] * sizes[b[0]] + b[2]
            span.append((s0, s0 + n_b - 1))
            s0 += n_b
        rate = rnd.choice([0.1, 0.2, 0.3])
        drops = {s for s in range(1, len(seq) + 1) if rnd.random() < rate}
        if mode == "skip":                            # known, gone, (skip), gone, known
            (_, last1), (first2, _) = span[i], span[i + 1]
            drops = (drops - {last1 - 1, first2 + 1}) | {last1, first2}
            counts["skips"] += 1
        elif mode == "hide":                          # known, the hidden block, known
            a, b = span[i + 1]
            drops = (drops - {a - 1, b + 1}) | set(range(a, b + 1))
            counts["hidden"] += 1
        meta, truth = make_tms2(precs=precs, nces=tuple(nces), fs_scans=fs, blocks=blocks,
                                drops=drops, seed=trial, split_scan=False, auto_low_block=None)
        lists = None if rnd.random() < 0.5 else [_list_frame(p) for p in precs]
        try:
            scans, rep = SC.map_precursors(SC.scan_table(meta), lists)
        except SC.ScanMapError:
            continue
        assert _mis_keyed(scans, truth) == [], (trial, mode, precs, nces, blocks, sorted(drops),
                                                fs, lists is not None)
        counts["inferred"] += rep["inference"]["inferred"]
        counts["inconsistent"] += rep["inference"]["runs_inconsistent"]
    assert counts["skips"] > 50 and counts["hidden"] > 50 and counts["shared_nce"] > 60
    assert counts["inferred"] > 200 and counts["inconsistent"] > 30, counts


# ---------------------------------------------------------------------------
# averaging
# ---------------------------------------------------------------------------
def _payload_of(scans):
    """A tiny payload from [[(mz, intensity, noise, resolution), ...] per scan]."""
    return {"stats_per_scan": {}, "centroids_meta": {
        "time": [float(i) for i in range(len(scans))],
        "data": [{"mzs": [c[0] for c in s], "intensities": [c[1] for c in s],
                  "noises": [c[2] for c in s], "resolutions": [c[3] for c in s]}
                 for s in scans]}}


def test_a_jitter_split_merges_and_a_resolved_neighbour_stays_apart():
    # FWHM at m/z 100 = 5.9 ppm, half = 2.9 ppm; the neighbour 8 ppm away is
    # resolved (> 1 FWHM), the 1.6 ppm jitter is not
    r = 170000.0
    m = 100.0
    meta = _payload_of([
        [(m, 1000, 10, r), (m * (1 + 8e-6), 500, 10, r)],
        [(m * (1 + 0.2e-6), 1000, 10, r), (m * (1 + 8.1e-6), 500, 10, r)],
        [(m * (1 + 1.6e-6), 1000, 10, r)],          # beyond the 1 ppm seed: merged
    ])
    a = SC.average_scans(meta, [1, 2, 3], noise_threshold=0)
    assert len(a.mz) == 2
    assert a.n_scans_present.tolist() == [3, 2]
    assert a.intensity[0] == pytest.approx(1000.0)
    assert a.intensity[1] == pytest.approx(1000.0 / 3)
    # a flat 1 ppm rule alone would have split the first peak
    mz = np.sort([m, m * 1.0000002, m * 1.0000016])
    lab = SC.cluster_centroids(mz, np.ones(3) * 1000, np.ones(3) * r, merge_fwhm=0)
    assert len(set(lab)) == 2
    # with no resolution the split stays, unless the nominal resolving power fills it
    nan = np.full(3, np.nan)
    assert len(set(SC.cluster_centroids(mz, np.ones(3), nan))) == 2
    nom = SC.nominal_resolution(R200)
    assert len(set(SC.cluster_centroids(mz, np.ones(3), nan, nominal_resolution=nom))) == 1


def test_averaging_rules_mean_with_absence_zero_and_the_server_s_s_n():
    m, r = 150.0, 160000.0
    meta = _payload_of([
        [(m, 900, 100, r)],
        [(m * (1 + 0.3e-6), 1200, 120, r)],
        [],
        [(m * (1 - 0.2e-6), 600, 80, r)],
    ])
    a = SC.average_scans(meta, [1, 2, 3, 4], noise_threshold=3)
    assert len(a.mz) == 1
    i = np.array([900, 1200, 600.0])
    mz = np.array([m, m * (1 + 0.3e-6), m * (1 - 0.2e-6)])
    assert a.mz[0] == pytest.approx(float((i * mz).sum() / i.sum()), rel=1e-12)
    assert a.intensity[0] == pytest.approx(i.sum() / 4)          # the empty scan counts 0
    sn = i / np.array([100, 120, 80.0])
    expect = (sn * i).sum() / i.sum() * 3 / math.sqrt(4)
    assert a.signal_to_noise[0] == pytest.approx(expect)
    assert a.resolution[0] == pytest.approx(r)
    assert a.per_scan[0].tolist() == [900, 1200, 0, 600]
    # below the threshold the peak is dropped
    weak = SC.average_scans(_payload_of([[(m, 250, 100, r)], [], [], []]), [1, 2, 3, 4])
    assert weak.signal_to_noise[0] == pytest.approx(2.5 * 1 / 2)
    assert not weak.keep[0]


def test_two_centroids_of_one_scan_in_one_cluster_are_one_reading():
    m, r = 120.0, 180000.0
    meta = _payload_of([
        [(m * (1 - 0.4e-6), 500, 100, r), (m * (1 + 0.4e-6), 500, 100, r)],
        [(m, 1000, 100, r)],
    ])
    a = SC.average_scans(meta, [1, 2], noise_threshold=0)
    assert len(a.mz) == 1
    assert a.per_scan[0].tolist() == [1000, 1000]
    assert a.per_scan_n[0].tolist() == [2, 1]
    assert a.n_scans_present[0] == 2                     # scans, not centroids
    # the scan's S/N is its summed intensity over its noise: 10, not two 5s
    assert a.signal_to_noise[0] == pytest.approx(10 * 2 / math.sqrt(2))


def test_a_payload_without_resolutions_takes_the_nominal_resolving_power():
    meta, _ = make_tms2(drops=())
    for c in meta["centroids_meta"]["data"]:
        del c["resolutions"]
    scans, _ = SC.map_precursors(SC.scan_table(meta), [_list_frame()])
    sp, _, _, rep = SC.build_spectra("S", meta, scans)
    assert rep["resolution_source"].startswith("nominal")
    got = sp["resolution"].to_numpy()
    assert np.allclose(got, R200 * np.sqrt(200 / sp["mz"].to_numpy()), rtol=1e-9)


def test_spectra_follow_the_servers_schema_and_keys():
    meta, truth = make_tms2()
    res = SC.extract_payload("SAMPLE1xxxxxxxxx", meta, lists=[_list_frame()])
    sp, par = res.spectra, res.parents
    assert list(sp.columns) == list(SRC.SPECTRA_COLUMNS)
    assert list(par.columns) == list(SRC.PARENTS_COLUMNS)
    keys = {SRC.spectrum_key(p, f"hcd{e:.2f}") for p in PREC for e in NCES}
    assert set(par["spectrum"]) == keys
    assert sp["energy_resolved"].all() and set(sp["kind"]) == {"hcd"}
    # every group's scans, times and volts
    t = truth[truth["order"] == 2]
    for key, g in t.groupby("spectrum"):
        row = par[par["spectrum"] == key].iloc[0]
        assert row["scan_count"] == len(g)
        assert row["isolation_width"] == pytest.approx(WIDTH)
        assert math.isfinite(row["hcd_v"])
        # the precursor is located at the generator's m/z
        p = float(g["parent_mz"].iloc[0])
        if math.exp(-float(g["nce"].iloc[0]) / 15) > 0.3:
            assert row["precursor_mz"] == pytest.approx(p, abs=2e-4)
    # each spectrum: precursor and both fragments; no noise centroid (S/N < 3)
    for key, g in sp.groupby("spectrum"):
        p = float(g["parent_mz"].iloc[0])
        want = [p - LOSSES[1], p - LOSSES[0]]
        for w in want:
            assert (np.abs(g["mz"] - w) < 1e-3).any(), (key, w)
        assert (g["signal_to_noise"] >= 3).all()
        assert len(g) <= 3
    # the per-scan rows average to the spectrum, exactly
    pk = res.scan_peaks
    assert list(pk.columns) == list(SC.SCAN_PEAK_COLUMNS)
    m = pk.groupby(["spectrum", "peak_mz"])["intensity"].mean()
    s = sp.set_index(["spectrum", "mz"])["intensity"].reindex(m.index)
    assert np.allclose(m.to_numpy(), s.to_numpy(), rtol=1e-12)
    # the split scan: one reading of two centroids
    split = truth.attrs["split_scan"]
    row = pk[(pk["scan"] == split) & (pk["n_centroids"] == 2)]
    assert len(row) == 1


def test_extract_payload_is_deterministic_and_builds_the_window():
    meta, _ = make_tms2()
    a = SC.extract_payload("S", meta, lists=[_list_frame()])
    b = SC.extract_payload("S", json.loads(json.dumps(meta)), lists=[_list_frame()])
    for x, y in ((a.spectra, b.spectra), (a.parents, b.parents), (a.scans, b.scans),
                 (a.window, b.window), (a.scan_peaks, b.scan_peaks)):
        pd.testing.assert_frame_equal(x, y)
    w = a.window
    assert list(w.columns) == list(SRC.WINDOW_COLUMNS)
    assert set(np.round(w["parent_mz"], 4)) == set(PREC)
    # the 0.2-Da co-isolate is in each window, the 13C peak (+1.003) is not
    for p in PREC:
        g = w[np.isclose(w["parent_mz"], p)]
        assert (np.abs(g["mz"] - (p + 0.2)) < 1e-3).any()
        assert not (np.abs(g["mz"] - (p + 1.00336)) < 1e-2).any()
        assert g.loc[g["is_parent"], "mz"].iloc[0] == pytest.approx(p, abs=1e-3)
    assert a.report["mapping"]["inference"]["unassigned"] == 0
    c = SC.extract_payload("S", meta, lists=[_list_frame()], with_window=False)
    assert not len(c.window) and any("not asked" in n for n in c.notes)


def test_isolation_widths_per_parent_feed_the_window():
    meta, _ = make_tms2(drops=())
    for r in meta["stats_per_scan"].values():
        if r["Scan Node:"] == "tMS2" and r["Scan Event:"] % 5 == 2:     # the 137 rows
            r["MS2 Isolation Width:"] = 1.2
            r["HighMass"] = _high(PREC[1], 1.2)
    res = SC.extract_payload("S", meta, lists=[_list_frame()])
    assert res.widths[round(PREC[1], 6)] == pytest.approx(1.2)
    assert res.widths[round(PREC[0], 6)] == pytest.approx(WIDTH)
    par = res.parents.drop_duplicates("parent_mz").set_index("parent_mz")["isolation_width"]
    assert par[PREC[1]] == pytest.approx(1.2) and par[PREC[0]] == pytest.approx(WIDTH)
    w = res.window
    span = w.groupby(w["parent_mz"].round(4))["offset_da"].agg(lambda s: s.abs().max())
    assert span[round(PREC[1], 4)] <= 1.2 / 2 + SRC.WINDOW_PAD_DA


# ---------------------------------------------------------------------------
# a file the server groups: its time-series times
# ---------------------------------------------------------------------------
def test_assign_from_times_maps_the_server_s_groups():
    meta, truth = make_tms2(garble=0.0, drops=())
    rows = meta["stats_per_scan"]
    for k in [k for k, r in rows.items() if r["Scan Node:"] == "tMS2"]:
        del rows[k]                           # an ordinary file: MS2 scans have no row
    st = SC.scan_table(meta)
    t = truth[truth["order"] == 2]
    times = {}
    for key, g in t.groupby("spectrum"):
        ts = [st.loc[st["scan"] == s, "t_s"].iloc[0] for s in g["scan"]]
        # the server writes ISO strings from the epoch; one key also as numbers
        times[key] = ([float(x) + 3e-4 for x in ts] if key.startswith("120") else
                      [(pd.Timestamp(0) + pd.Timedelta(seconds=float(x))).isoformat() for x in ts])
    scans, rep = SC.assign_from_times(st, times)
    _check_truth(scans, truth, cols=("parent_mz", "nce", "spectrum"))
    assert set(scans.loc[scans["order"] == 2, "precursor_source"]) == {SC.PRECURSOR_SERVER}
    assert all(v["matched"] == v["times"] for v in rep["spectra"].values())
    assert rep["headerless_in_no_group"] == 0
    # a time 2 ms off matches nothing, and a hidden tail is counted as runs
    first = t["scan"].iloc[0]
    bad = {next(iter(times)): [st.loc[st["scan"] == first, "t_s"].iloc[0] + 0.002, 999.0]}
    _, rep2 = SC.assign_from_times(st, bad)
    assert rep2["unmatched_times"] == 2
    assert rep2["headerless_in_no_group"] == len(t)
    assert SC._runs_text([1, 154, 155, 156, 160]) == "1, 154-156, 160"


def test_scan_peak_intensities_read_the_server_s_own_peaks():
    meta, _ = make_tms2(garble=0.0, drops=())
    scans, _ = SC.map_precursors(SC.scan_table(meta))
    sp, _, ours, _ = SC.build_spectra("S", meta, scans)
    # server peaks = our clusters: identical per-scan readings
    got = SC.scan_peak_intensities(meta, scans, sp)
    pd.testing.assert_frame_equal(
        got.sort_values(["spectrum", "peak_mz", "scan"]).reset_index(drop=True),
        ours.sort_values(["spectrum", "peak_mz", "scan"]).reset_index(drop=True))
    # a server peak 0.5 ppm off every cluster still takes its centroids
    moved = sp.assign(mz=sp["mz"] * (1 + 0.5e-6))
    got2 = SC.scan_peak_intensities(meta, scans, moved)
    assert np.allclose(got2["intensity"].to_numpy(), got["intensity"].to_numpy())


# ---------------------------------------------------------------------------
# the scan-range rule, judged per experiment
# ---------------------------------------------------------------------------
def _set_high(meta, events, high):
    """Every trailer row of `events` reads HighMass `high` (a fixed range)."""
    for r in meta["stats_per_scan"].values():
        if r["Scan Event:"] in events:
            r["HighMass"] = high
            r["ScanType"] = r["ScanType"].rsplit("[", 1)[0] + f"[{r['LowMass']:.4f}-{high:.4f}]"


def test_a_fixed_range_per_experiment_derives_nothing_and_takes_its_list():
    """Each experiment set to its own fixed range (300 / 450 / 600): three
    distinct HighMass values in the file, but one per experiment -- no
    precursor can be read off them, and the right list is not refused."""
    meta, truth = make_tms2()
    for k, high in enumerate((300.0, 450.0, 600.0)):
        _set_high(meta, range(5 * k + 1, 5 * k + 6), high)
    st = SC.scan_table(meta)
    with pytest.raises(SC.ScanMapError, match="pass the method's inclusion list") as e:
        SC.map_precursors(st)
    assert e.value.needs_list and "event(s) 1, 2, 3, 4, 5" in str(e.value)
    scans, rep = SC.map_precursors(st, [_list_frame()])
    _check_truth(scans, truth)
    assert rep["rule"]["experiments"] == {1: SC.RULE_FIXED, 2: SC.RULE_FIXED, 3: SC.RULE_FIXED}
    assert not rep["rule"]["applies"] and not rep["rule"]["checked"]


def test_one_fixed_experiment_beside_rule_following_ones():
    meta, truth = make_tms2()
    _set_high(meta, range(11, 16), 600.0)                 # experiment 3 only
    st = SC.scan_table(meta)
    with pytest.raises(SC.ScanMapError, match=r"no precursor for event\(s\) 11, 12, 13, 14, 15"):
        SC.map_precursors(st)
    scans, rep = SC.map_precursors(st, [_list_frame()])
    _check_truth(scans, truth)
    assert rep["rule"]["experiments"] == {1: SC.RULE_FOLLOWS, 2: SC.RULE_FOLLOWS,
                                          3: SC.RULE_FIXED}
    # the rule still checks the list in the experiments it holds in (10 events)
    assert rep["rule"]["checked"] and rep["rule"]["n_events"] == 10
    assert any("does not follow the precursor in experiment(s) 3" in n for n in rep["notes"])
    swapped = list(PREC)
    swapped[1], swapped[2] = swapped[2], swapped[1]
    with pytest.raises(SC.ScanMapError, match="do not fit the inclusion list"):
        SC.map_precursors(st, [_list_frame(tuple(swapped))])


def test_one_precursor_over_several_energies_checks_its_list_against_the_rule():
    """A 1-row list over three NCEs: one HighMass in the file, which cannot
    tell a fixed range from one precursor. Nothing is derived; a list is
    still checked against the rule, and a misfit is said, not refused."""
    meta, truth = make_tms2(prec=(PREC[1],), cycles=((3, 0), (3, 0), (3, 0)), drops=())
    st = SC.scan_table(meta)
    with pytest.raises(SC.ScanMapError, match="one event seen in its experiment"):
        SC.map_precursors(st)
    scans, rep = SC.map_precursors(st, [_list_frame((PREC[1],))])
    _check_truth(scans, truth)
    assert not any("do not fit the scan-range rule" in n for n in rep["notes"])
    _, rep = SC.map_precursors(st, [_list_frame((PREC[3],))])
    assert rep["rule"]["unjudged_events_off"] == [1, 2, 3]
    assert any("event(s) 1, 2, 3 do not fit the scan-range rule" in n and "check them" in n
               for n in rep["notes"])


def test_a_list_whose_rows_divide_the_methods_is_refused_on_a_fixed_range():
    """6 rows x 3 NCEs on one fixed range: a 3-, 2- or 1-row list passes the
    Full Scan event (18 / n is whole) and one NCE per experiment, but its
    experiments carry one NCE in pairs and alternate in time."""
    prec = PREC + (219.1380,)
    for drops in ((), "default"):
        meta, truth = make_tms2(prec=prec, fixed_range=True, drops=drops)
        st = SC.scan_table(meta)
        for rows in (prec[:3], prec[:2], prec[:1], prec[3:]):
            with pytest.raises(SC.ScanMapError, match="alternate in time"):
                SC.map_precursors(st, [_list_frame(rows)])
        scans, _ = SC.map_precursors(st, [_list_frame(prec)])
        _check_truth(scans, truth)
    # two experiments of one NCE, one after the other in time: valid, noted
    meta, truth = make_tms2(nces=(15.0, 15.0, 30.0), fixed_range=True)
    scans, rep = SC.map_precursors(SC.scan_table(meta), [_list_frame()])
    _check_truth(scans, truth)
    assert any("experiments 1 and 2 of the list both read NCE 15" in n for n in rep["notes"])


def test_two_events_of_one_scan_range_in_a_rule_following_experiment_derive_nothing():
    # event 5 reads event 4's HighMass in an experiment whose other events
    # follow the rule: a precursor derived from it would coincide with event 4's
    meta, _ = make_tms2(drops=())
    high4 = next(r["HighMass"] for r in meta["stats_per_scan"].values() if r["Scan Event:"] == 4)
    _set_high(meta, (5,), high4)
    with pytest.raises(SC.ScanMapError, match="reads the same HighMass") as e:
        SC.map_precursors(SC.scan_table(meta))
    assert e.value.needs_list and "no precursor for event(s) 4, 5:" in str(e.value)


# ---------------------------------------------------------------------------
# the cycle walk at a block switch, and the lists of unequal length
# ---------------------------------------------------------------------------
def test_a_run_at_either_edge_of_a_block_switch_is_placed():
    seq = scan_sequence()
    first2 = next(s for s, x in enumerate(seq, 1) if x[:2] == ("ms2", 1))
    # the new block's first two scans (the switch at the run's first scan),
    # and the old block's last two (the switch right after the run)
    for drops in ({first2, first2 + 1}, {first2 - 2, first2 - 1}):
        meta, truth = make_tms2(drops=drops)
        scans, rep = SC.map_precursors(SC.scan_table(meta), [_list_frame()])
        assert rep["inference"]["inferred"] == 2, drops
        _check_truth(scans, truth)


def test_a_block_entered_past_its_first_event_breaks_the_model():
    # experiment 2's time block starts at its SECOND row: the known pair at the
    # switch (experiment 1's last scan, then event 7) is no transition the
    # cycle allows, so nothing is inferred
    blocks = [(0, 2, 2), (1, 2, 3, 1), (2, 2, 1)]
    seq = scan_sequence(blocks=blocks)
    e3 = next(s for s, x in enumerate(seq, 1) if x[:2] == ("ms2", 2))
    meta, _ = make_tms2(blocks=blocks, drops={1, e3 + 4})
    _, rep = SC.map_precursors(SC.scan_table(meta), [_list_frame()])
    assert rep["inference"]["model_breaks"] == 1 and rep["inference"]["inferred"] == 0


@pytest.mark.parametrize("sizes", [(3, 5, 5), (5, 3, 4)])
def test_per_experiment_lists_of_different_lengths_map_row_by_row(sizes):
    """One list per experiment, of different lengths, on a fixed range (so no
    scan-range check can rescue a wrong row rule): experiment, list row and
    precursor of every scan are the truth's."""
    pool = PREC + (219.1380, 233.1536)
    precs = [pool[k:k + n] for k, n in enumerate(sizes)]
    meta, truth = make_tms2(precs=precs, fixed_range=True)
    scans, rep = SC.map_precursors(SC.scan_table(meta), [_list_frame(p) for p in precs])
    assert [(x["first_event"], x["last_event"]) for x in rep["experiments"]] == [
        (1 + sum(sizes[:k]), sum(sizes[:k + 1])) for k in range(3)]
    _check_truth(scans, truth, cols=("order", "event", "parent_mz", "nce", "spectrum",
                                     "experiment", "slot"))


# ---------------------------------------------------------------------------
# guard branches
# ---------------------------------------------------------------------------
def test_a_walk_below_its_events_first_mass_is_not_used():
    meta, truth = make_tms2(auto_low_block=1)
    seq = scan_sequence()
    first2 = next(s for s, x in enumerate(seq, 1) if x[:2] == ("ms2", 1))   # LowMass 50
    c = meta["centroids_meta"]["data"][first2 - 1]
    for k, v in (("mzs", 49.0), ("intensities", 1000.0), ("resolutions", 1e5), ("noises", 100.0)):
        c[k].insert(0, v)
    _, rep = SC.map_precursors(SC.scan_table(meta), [_list_frame()])
    assert rep["inference"]["runs_failed_range"] == 1
    assert first2 in rep["inference"]["unassigned_scans"]


def test_a_walk_through_an_event_with_no_range_to_check_is_not_used():
    # no list, and event 8 never seen with a trailer: nothing predicts its range.
    # Its two straight runs inside the block are unverifiable; its last scan,
    # at the switch to experiment 3, is not walked at all (no list: open layout)
    seq = scan_sequence()
    ev8 = {s for s, x in enumerate(seq, 1) if x == ("ms2", 1, 2)}
    meta, _ = make_tms2(drops=ev8)
    scans, rep = SC.map_precursors(SC.scan_table(meta))
    assert (rep["inference"]["runs_unverifiable"], rep["inference"]["runs_ambiguous"]) == (2, 1)
    assert rep["inference"]["inferred"] == 0 and set(rep["inference"]["unassigned_scans"]) == ev8
    # with the list and the rule check off nothing predicts it either: all three
    _, rep = SC.map_precursors(SC.scan_table(meta), [_list_frame()], rule_check=False)
    assert rep["inference"]["runs_unverifiable"] == len(ev8)
    assert set(rep["inference"]["unassigned_scans"]) == ev8


def test_a_list_a_millidalton_off_is_refused_by_the_rule():
    lst = _list_frame(tuple(p + 0.001 for p in PREC))          # 1.02 mDa > RULE_TOL_MDA
    with pytest.raises(SC.ScanMapError, match="do not fit the inclusion list"):
        SC.map_precursors(SC.scan_table(make_tms2()[0]), [lst])


def test_a_scan_two_time_series_claim_stays_with_the_first():
    meta, _ = make_tms2(garble=0.0, drops=())
    st = SC.scan_table(meta)
    t5 = float(st.loc[st["scan"] == 5, "t_s"].iloc[0])
    a, b = "120.0808@hcd5.00", "137.1325@hcd5.00"
    scans, rep = SC.assign_from_times(st, {a: [t5], b: [t5]})
    assert rep["claimed_twice"] == [5]
    assert scans.loc[scans["scan"] == 5, "spectrum"].iloc[0] == a
    assert rep["spectra"][b]["matched"] == 0


def test_the_s_n_floor_keeps_a_peak_at_it_and_a_payload_without_noises():
    m, r = 150.0, 160000.0
    a = SC.average_scans(_payload_of([[(m, 300, 100, r)]]), [1], noise_threshold=3)
    assert a.signal_to_noise[0] == pytest.approx(3.0) and a.keep[0]
    meta, _ = make_tms2(drops=())
    for c in meta["centroids_meta"]["data"]:
        del c["noises"]
    res = SC.extract_payload("S", meta, lists=[_list_frame()], with_window=False)
    assert len(res.spectra) > 100 and res.spectra["signal_to_noise"].isna().all()


def test_the_manifest_tolerances_name_every_one():
    t = SC.tolerances()
    assert t["parent_tol_da"] == SC.PARENT_TOL_DA
    assert t["rule"]["distinct_da"] == SC.RULE_DISTINCT_DA
    assert t["rule"]["decimals"] == SC.RULE_DECIMALS


# ---------------------------------------------------------------------------
# the server's peaks per scan, where they are close
# ---------------------------------------------------------------------------
def _close_pair(server_mz):
    """Two lines 5.6 ppm apart at R 60000 (FWHM 16.7 ppm): our averaging merges
    them into one cluster; the server lists them at `server_mz`."""
    m, r = 150.0, 60000.0
    m2 = m * (1 + 5.6e-6)
    meta = _payload_of([[(m, 1000, 10, r), (m2, 300, 10, r)] for _ in range(4)])
    scans = pd.DataFrame({"scan": [1, 2, 3, 4], "t_s": [0.0, 1.0, 2.0, 3.0],
                          "spectrum": ["k"] * 4})
    spectra = pd.DataFrame({"spectrum": ["k", "k"], "mz": list(server_mz),
                            "resolution": [r, r]})
    return meta, scans, spectra, (m, m2)


def test_two_server_peaks_in_one_of_our_clusters_each_read_their_own():
    meta, scans, _, (m, m2) = _close_pair((0, 0))
    assert len(SC.average_scans(meta, [1, 2, 3, 4], noise_threshold=0).mz) == 1
    merged = float(SC.average_scans(meta, [1, 2, 3, 4], noise_threshold=0).mz[0])
    # the server's m/z at its lines, or one of them exactly at our cluster's
    for server in ((m, m2), (merged, m2)):
        _, _, spectra, _ = _close_pair(server)
        pk = SC.scan_peak_intensities(meta, scans, spectra)
        mean = pk.groupby("peak_mz")["intensity"].mean()
        assert mean.tolist() == [1000.0, 300.0], server


def test_a_server_peak_takes_only_centroids_within_half_its_width():
    # a server peak 0.5 ppm off our line, and a centroid 0.7 FWHM away that
    # the server does not list (its own cluster): not that peak's
    m, r = 150.0, 60000.0
    far = m * (1 + 0.7 / r)
    meta = _payload_of([[(m, 1000, 10, r), (far, 300, 10, r)] for _ in range(2)])
    scans = pd.DataFrame({"scan": [1, 2], "t_s": [0.0, 1.0], "spectrum": ["k", "k"]})
    spectra = pd.DataFrame({"spectrum": ["k"], "mz": [m * (1 + 0.5e-6)], "resolution": [r]})
    pk = SC.scan_peak_intensities(meta, scans, spectra)
    assert pk["intensity"].tolist() == [1000.0, 1000.0]
    # a server peak 5 ppm below a cluster of ours (a line and a centroid 7 ppm
    # above it, merged): only the cluster's m/z itself is a match; off it,
    # the centroids within half its width (the line, not the one 12 ppm away)
    meta = _payload_of([[(m, 1000, 10, r), (m * (1 + 7e-6), 300, 10, r)] for _ in range(2)])
    assert len(SC.average_scans(meta, [1, 2], noise_threshold=0).mz) == 1
    spectra = pd.DataFrame({"spectrum": ["k"], "mz": [m * (1 - 5e-6)], "resolution": [r]})
    pk = SC.scan_peak_intensities(meta, scans, spectra)
    assert pk["intensity"].tolist() == [1000.0, 1000.0]


# ---------------------------------------------------------------------------
# review round 2: the layout without a list, the cycle walk's guards, the
# alternation check, an unseen event's range, the nominal resolution
# ---------------------------------------------------------------------------
P4 = (99.0441, 120.0808, 141.0910, 159.1016)
P4B = (113.0597, 137.1325, 151.0754, 155.1430)


@pytest.mark.parametrize("precs,nces", [((P4, P4), (15.0, 15.0)),        # one list, one NCE
                                        ((P4, P4B), (15.0, 15.0)),       # two lists, one NCE
                                        ((P4, P4B), (5.0, 14.0))])       # experiment 2 unseen
def test_without_a_list_the_nce_runs_never_pin_the_layout(precs, nces):
    """Two 4-row experiments; experiment 2 runs two scans at the end, both
    without a trailer row, and experiment 1's wrap is lost too. Without a
    list one NCE run (two experiments at one NCE, or the second never seen)
    splits the Full Scan event 'evenly' as ONE 8-event experiment: nothing
    may then be walked across a block edge, or experiment 2's scans land on
    experiment 1's events."""
    blocks = [(0, 1, 2), (1, 0, 2)]
    meta, truth = make_tms2(precs=precs, nces=nces, fs_scans=3, blocks=blocks,
                            drops={8, 10, 11}, split_scan=False, auto_low_block=None)
    scans, rep = SC.map_precursors(SC.scan_table(meta))
    assert _mis_keyed(scans, truth) == []
    assert not rep["layout_closed"]
    assert scans.loc[scans["scan"].isin([10, 11]), "spectrum"].eq("").all()
    assert set(rep["inference"]["unassigned_scans"]) >= {10, 11}
    # the method's two lists pin it: the wrap (scan 8) is placed, the hidden
    # experiment's scans stay unassigned (no trailer row in that block)
    scans, rep = SC.map_precursors(SC.scan_table(meta), [_list_frame(p) for p in precs])
    assert rep["layout_closed"] and _mis_keyed(scans, truth) == []
    assert scans.loc[scans["scan"] == 8, ["inferred", "event"]].iloc[0].tolist() == [True, 1.0]


SKIP_PREC = (120.0808, 137.1325, 203.1430, 155.1430, 181.0859)     # rows not in m/z order
#: one 5-row experiment, the instrument skipping event 3 once: e1..e5 e1 e2 | e4 e5 e1 e2 e3
SKIP_BLOCKS = [(0, 1, 2, 0), (0, 1, 0, 3)]


def test_a_run_no_walk_of_its_length_fits_stays_unassigned():
    """Closed layout (list + Full Scan): the one scan between the known e2 and
    e5 has no trailer row. A walk of one step from e2 lands on e4, not e5:
    no walk fits, the scan stays unassigned (it is e4, not e3)."""
    meta, truth = make_tms2(prec=SKIP_PREC, nces=(5.0,), fs_scans=4, blocks=SKIP_BLOCKS,
                            drops={12}, auto_low_block=None, split_scan=False)
    scans, rep = SC.map_precursors(SC.scan_table(meta), [_list_frame(SKIP_PREC)])
    inf = rep["inference"]
    assert rep["layout_closed"] and inf["model_breaks"] == 0
    assert (inf["runs_inconsistent"], inf["runs_failed_range"], inf["unassigned_scans"]) == (1, 0, [12])
    assert _mis_keyed(scans, truth) == []
    assert any("1 run(s) no walk fits" in n for n in rep["notes"])


def test_a_straight_run_must_have_the_length_of_its_events():
    """Open layout (no list, no Full Scan): only a straight run a -> a+1 ->
    ... -> b of exactly b - a - 1 scans is placed. The one scan between the
    known e2 and e5 does not fit it (two events between them): unassigned."""
    meta, truth = make_tms2(prec=SKIP_PREC, nces=(5.0,), fs_scans=0, blocks=SKIP_BLOCKS,
                            drops={8}, auto_low_block=None, split_scan=False)
    scans, rep = SC.map_precursors(SC.scan_table(meta))
    inf = rep["inference"]
    assert not rep["layout_closed"] and inf["model_breaks"] == 0
    assert (inf["runs_ambiguous"], inf["unassigned_scans"]) == (1, [8])
    assert _mis_keyed(scans, truth) == []


def test_a_short_block_hidden_inside_another_is_not_walked_over():
    """Experiment 2 runs two scans with no trailer row inside experiment 1's
    time (experiment 1 resumes at its first row after them): from the known
    e2, two steps on in experiment 1 end on e4, not on the known e1 after
    them -- no walk fits, both stay unassigned."""
    blocks = [(0, 1, 2), (1, 0, 2), (0, 1, 0), (2, 2, 1)]
    seq = scan_sequence(blocks=blocks)
    exp2 = {s for s, x in enumerate(seq, 1) if x[:2] == ("ms2", 1)}
    meta, truth = make_tms2(prec=SKIP_PREC, blocks=blocks, drops=exp2, auto_low_block=None)
    scans, rep = SC.map_precursors(SC.scan_table(meta), [_list_frame(SKIP_PREC)])
    inf = rep["inference"]
    assert inf["runs_inconsistent"] == 1 and set(inf["unassigned_scans"]) == exp2
    assert _mis_keyed(scans, truth) == []


def test_two_experiments_of_one_nce_in_reverse_time_order_are_not_alternating():
    """Experiments 1 and 2 share NCE 15, and experiment 2's time block runs
    first (time order is free: the Full Scan, the last event, runs first):
    the correct shared list maps every scan; the blocks never alternate."""
    for blocks in ([(1, 2, 3), (0, 2, 2), (2, 2, 1)], [(2, 2, 1), (1, 2, 3), (0, 2, 2)]):
        meta, truth = make_tms2(nces=(15.0, 15.0, 30.0) if blocks[0][0] == 1 else (15.0,) * 3,
                                blocks=blocks)
        scans, rep = SC.map_precursors(SC.scan_table(meta), [_list_frame()])
        _check_truth(scans, truth)
        assert any("both read NCE 15 (one after the other in time)" in n for n in rep["notes"])
    # a real split (a 3-row list on a 6-row method) still alternates, whatever the order
    prec = PREC + (219.1380,)
    for blocks in ([(0, 2, 2), (1, 2, 3), (2, 2, 1)], [(2, 2, 1), (1, 2, 3), (0, 2, 2)]):
        st = SC.scan_table(make_tms2(prec=prec, fixed_range=True, blocks=blocks)[0])
        with pytest.raises(SC.ScanMapError, match="alternate in time"):
            SC.map_precursors(st, [_list_frame(prec[:3])])


def test_an_unseen_event_of_a_fixed_range_experiment_takes_that_range():
    """Every trailer row reads 40-600 (a fixed range) and event 2 never has
    one: its scans are checked against, and keep, the experiment's range --
    never the HighMass the scan-range rule would predict from the list (160.23),
    which does not hold here."""
    seq = scan_sequence()
    ev2 = {s for s, x in enumerate(seq, 1) if x == ("ms2", 0, 1)}
    meta, truth = make_tms2(fixed_range=True, drops=ev2, auto_low_block=None)
    for s in ev2:                                   # a centroid inside 40-600, above 160.23
        c = meta["centroids_meta"]["data"][s - 1]
        for k, v in (("mzs", 301.2), ("intensities", 150.0), ("resolutions", 9e4),
                     ("noises", 100.0)):
            c[k].append(v)
    scans, rep = SC.map_precursors(SC.scan_table(meta), [_list_frame()])
    assert rep["inference"]["unassigned"] == 0 and rep["inference"]["runs_failed_range"] == 0
    _check_truth(scans, truth)
    got = scans[scans["scan"].isin(ev2)]
    assert got["inferred"].all() and (got["high_mass"] == 600.0).all()
    _, _, _, avg = SC.build_spectra("S", meta, scans)
    assert avg["range_cuts"] == {}
    assert avg["scan_ranges"][SRC.spectrum_key(PREC[1], "hcd5.00")][1] == 600.0


def test_an_unseen_events_rule_range_is_a_check_never_a_recorded_range():
    """Where the rule holds, an unseen event's walk is checked against the
    HighMass the rule predicts from its list row -- and the scan keeps no
    range it never recorded (NaN). With the rule check off nothing predicts
    it: the run is unverifiable, never placed on a guess."""
    seq = scan_sequence()
    ev2 = {s for s, x in enumerate(seq, 1) if x == ("ms2", 0, 1)}
    meta, truth = make_tms2(drops=ev2, auto_low_block=None)
    scans, rep = SC.map_precursors(SC.scan_table(meta), [_list_frame()])
    _check_truth(scans, truth)
    assert scans.loc[scans["scan"].isin(ev2), "high_mass"].isna().all()
    _, rep = SC.map_precursors(SC.scan_table(meta), [_list_frame()], rule_check=False)
    assert rep["inference"]["runs_unverifiable"] == len(ev2)
    assert set(rep["inference"]["unassigned_scans"]) == ev2


def test_a_payload_without_resolutions_takes_each_ms_orders_nominal_resolving_power():
    """No per-centroid resolutions, the Full Scan at R 120000 and the MS2
    scans at 60000: each average takes its own scans' resolving power -- never
    none because the file holds two -- and the report says which."""
    meta, _ = make_tms2(drops=())
    for r in meta["stats_per_scan"].values():
        if r["Scan Node:"] == "tMS2":
            r["FT Resolution:"] = 60000
    for c in meta["centroids_meta"]["data"]:
        del c["resolutions"]
    scans, _ = SC.map_precursors(SC.scan_table(meta), [_list_frame()])
    sp, _, _, rep = SC.build_spectra("S", meta, scans)
    assert rep["resolution_source"] == "nominal: FT Resolution 60000 at m/z 200"
    assert np.allclose(sp["resolution"], 60000 * np.sqrt(200 / sp["mz"]), rtol=1e-9)
    ms1 = SC.ms1_average(meta, scans)
    assert np.allclose(ms1["resolution"], R200 * np.sqrt(200 / np.asarray(ms1["mz"])), rtol=1e-9)
    # neither a resolution nor an FT Resolution: said, never 'per-scan centroids'
    for r in meta["stats_per_scan"].values():
        del r["FT Resolution:"]
    scans, _ = SC.map_precursors(SC.scan_table(meta), [_list_frame()])
    sp, _, _, rep = SC.build_spectra("S", meta, scans)
    assert rep["resolution_source"] == "none" and sp["resolution"].isna().all()


def test_highmass_values_are_told_apart_at_a_millidalton():
    """`RULE_DISTINCT_DA`: two rows of one experiment 13 mDa apart (our own
    standard's 155.1429 / 155.1559) read two scan ranges, and each scan's
    precursor comes from its own; two rows 0.4 mDa apart read one (178.603),
    so neither is derived -- they would coincide."""
    prec = (120.0808, 137.1321, 155.1429, 155.1559, 181.0859)
    meta, truth = make_tms2(prec=prec, drops=())
    scans, rep = SC.map_precursors(SC.scan_table(meta))
    _check_truth(scans, truth)
    assert set(scans.loc[scans["order"] == 2, "precursor_source"]) == {SC.PRECURSOR_RULE}
    assert set(rep["rule"]["experiments"].values()) == {SC.RULE_FOLLOWS}
    close = (120.0808, 137.1321, 155.1429, 155.1433, 181.0859)
    assert round(_high(close[2]), 3) == round(_high(close[3]), 3)
    with pytest.raises(SC.ScanMapError, match="reads the same HighMass") as e:
        SC.map_precursors(SC.scan_table(make_tms2(prec=close)[0]))
    assert e.value.needs_list and "no precursor for event(s) 3, 4" in str(e.value)


def test_a_split_reading_s_noise_is_weighted_by_its_centroids_intensities():
    """Two centroids of one scan in one cluster (500 at noise 50, 1500 at
    noise 150): the scan's noise is their intensity-weighted noise, 125 --
    so its S/N is 2000 / 125 = 16, as the other scan's (2000 at noise 125)."""
    m, r = 120.0, 180000.0
    meta = _payload_of([
        [(m * (1 - 0.4e-6), 500, 50, r), (m * (1 + 0.4e-6), 1500, 150, r)],
        [(m, 2000, 125, r)],
    ])
    a = SC.average_scans(meta, [1, 2], noise_threshold=24)
    assert len(a.mz) == 1
    assert a.signal_to_noise[0] == pytest.approx(16 * 2 / math.sqrt(2))
    assert not a.keep[0]                                  # 22.6 < 24


def test_a_data_dependent_event_on_a_rule_following_range_is_refused():
    """Readable filters naming another precursor in one cycle of event 2: on
    a fixed range the event maps scan by scan; where the range follows each
    precursor, the event's scans read two ranges, and that is refused (its
    HighMass would name neither precursor) -- never one range picked."""
    meta, _ = make_tms2(garble=0.0, drops=())
    rows = meta["stats_per_scan"]
    ev2 = sorted(int(k) for k, r in rows.items() if r["Scan Event:"] == 2)
    r0 = rows[str(ev2[0])]
    r0["ScanType"] = (r0["ScanType"].replace(f"{PREC[1]:.4f}@", f"{PREC[3]:.4f}@")
                      .rsplit("[", 1)[0] + f"[{r0['LowMass']:.4f}-{_high(PREC[3]):.4f}]")
    r0["HighMass"] = _high(PREC[3])
    with pytest.raises(SC.ScanMapError, match=r"event 2: its scans read 2 scan ranges"):
        SC.map_precursors(SC.scan_table(meta))


# ---------------------------------------------------------------------------
# review round 3: scans whose trailer gives no MS order (e.g. a SIM experiment)
# ---------------------------------------------------------------------------
def insert_scans(meta, after, extra):
    """The payload with `extra` scans inserted after scan `after` (1-based):
    each (trailer row or None, centroids). Later scans are renumbered; the new
    scans' times sit between their neighbours', so every scan of `meta` keeps
    its time. -> (payload, the new scans' numbers, {old scan: new scan})."""
    out = json.loads(json.dumps(meta))
    cm, n, k = out["centroids_meta"], len(meta["centroids_meta"]["time"]), len(extra)
    t0 = cm["time"][after - 1]
    t1 = cm["time"][after] if after < n else t0 + 1.0
    rows = {str(int(s) + k if int(s) > after else int(s)): r
            for s, r in out["stats_per_scan"].items()}
    for j, (row, _) in enumerate(extra):
        if row is not None:
            rows[str(after + 1 + j)] = row
    cm["data"][after:after] = [json.loads(json.dumps(c)) for _, c in extra]
    cm["time"][after:after] = [t0 + (t1 - t0) * (j + 1) / (k + 1) for j in range(k)]
    out["stats_per_scan"], out["num_of_scans"] = rows, n + k
    return out, list(range(after + 1, after + 1 + k)), {
        s: s + k if s > after else s for s in range(1, n + 1)}


def _tsim(event, low=100.0, high=130.0, mz=(120.0808, 121.0842)):
    """(trailer row, centroids) of one tSIM scan: its own event, no HCD energy, a
    narrow scan range, a readable 'SIM ms' filter."""
    row = {"Scan Node:": "tSIM", "HCD Energy:": "", "HCD Energy V:": "", "Scan Event:": event,
           "MS2 Isolation Width:": high - low, "MsType": "Ms", "StartTime": 0.0,
           "ScanType": f"FTMS + p NSI SIM ms [{low:.4f}-{high:.4f}]", "LowMass": low,
           "HighMass": high, "TIC": 4.4e6, "Ion Injection Time (ms):": 50.0,
           "Max. Ion Time (ms):": 100.0, "AGC Fill:": 0.5, "Micro Scan Count:": 10,
           "FT Resolution:": int(R200), "Monoisotopic M/Z:": 0.0}
    cen = {"mzs": list(mz), "intensities": [4e6, 4e5][:len(mz)],
           "resolutions": [R200] * len(mz), "noises": [NOISE] * len(mz)}
    return row, cen


def _renumbered(truth, renum, new):
    t = truth.assign(scan=truth["scan"].map(renum))
    return t[~t["scan"].isin(new)]


def test_a_sim_block_between_two_experiments_is_left_as_it_is():
    """Three tSIM scans (trailer rows, no HCD energy, event 17) between
    experiment 1's and experiment 2's blocks of a closed layout: their
    trailer gives no MS order, so they belong to no block -- never walked
    over as experiment 1's next events, never in a spectrum, and the file's
    spectra are those of the same file without them. (Its headerless scans
    lie inside one block: a leading or trailing run would stay unassigned
    beside an experiment outside the layout, below.)"""
    seq = scan_sequence()
    drops = default_drops(seq)
    last1 = max(s for s, x in enumerate(seq, 1) if x[:2] == ("ms2", 0))
    first2 = last1 + 1
    drops -= {last1, first2, 1, len(seq)}          # no headerless scan beside them, none at an edge
    meta, truth = make_tms2(drops=drops)
    sim, new, renum = insert_scans(meta, last1, [_tsim(16 + 1)] * 3)
    st = SC.scan_table(sim)
    assert st.loc[st["scan"].isin(new), "order"].isna().all()
    scans, rep = SC.map_precursors(st, [_list_frame()])
    got = scans[scans["scan"].isin(new)]
    assert not got["inferred"].any() and (got["spectrum"] == "").all()
    assert got["order"].isna().all() and (got["event"] == 17).all()
    assert (got["node"] == "tSIM").all() and got["nce"].isna().all()
    inf = rep["inference"]
    assert inf["headerless"] == len(drops) and inf["inferred"] == len(drops)
    assert inf["runs_blocked_by_header_rows"] == 0
    assert _mis_keyed(scans, _renumbered(truth, renum, new)) == []
    assert any("3 scan(s) whose trailer gives no MS order (tSIM)" in n for n in rep["notes"])
    a = SC.extract_payload("S", meta, lists=[_list_frame()])
    b = SC.extract_payload("S", sim, lists=[_list_frame()])
    pd.testing.assert_frame_equal(a.spectra, b.spectra)
    pd.testing.assert_frame_equal(a.parents, b.parents)
    assert b.report["ms1_scans"] == a.report["ms1_scans"]


def test_a_run_beside_a_sim_scan_is_not_walked():
    """The same tSIM block between the two headerless scans of a block
    switch: the gap holds scans of no block, so where experiment 1 ends and
    experiment 2 starts is unknown -- the two stay unassigned (counted), the
    tSIM scans untouched, and no more scans are inferred than had no row.
    The file shows an experiment outside the layout, so the leading and the
    trailing run stay unassigned too; the run inside experiment 3 is placed."""
    seq = scan_sequence()
    last1 = max(s for s, x in enumerate(seq, 1) if x[:2] == ("ms2", 0))
    meta, truth = make_tms2()
    sim, new, renum = insert_scans(meta, last1, [_tsim(17)] * 3)
    scans, rep = SC.map_precursors(SC.scan_table(sim), [_list_frame()])
    inf = rep["inference"]
    assert inf["runs_blocked_by_header_rows"] == 1
    assert inf["runs_beside_outside_scans"] == 2
    assert set(inf["unassigned_scans"]) == {1, renum[last1], renum[last1 + 1], renum[len(seq)]}
    assert inf["inferred"] == inf["headerless"] - 4 and inf["inferred"] <= inf["headerless"]
    assert not scans.loc[scans["scan"].isin(new), "inferred"].any()
    assert scans.loc[scans["scan"].isin(new), "order"].isna().all()
    assert _mis_keyed(scans, _renumbered(truth, renum, new)) == []
    assert any("holding a scan whose trailer gives no MS order" in n for n in rep["notes"])


def test_sim_scans_after_the_full_scan_stay_out_of_the_ms1_average():
    """tSIM scans right after the Full Scan block: never walked into it (the
    MS1 average holds the Full Scans only), and on a file whose filters are
    readable their 'SIM ms' filter agrees with a trailer that names no
    fragmentation -- the file's filters stay readable, its polarity read.
    (The first scan keeps its row: beside an experiment outside the layout a
    leading run is not walked.)"""
    meta, truth = make_tms2(garble=0.0, drops=default_drops(scan_sequence()) - {1})
    sim, new, renum = insert_scans(meta, FS_SCANS, [_tsim(17, 130.0, 145.0,
                                                          (137.1325, 138.1358))] * 2)
    st = SC.scan_table(sim)
    assert st.loc[st["scan"].isin(new), "filter_ok"].all()
    assert st.loc[st["scan"].isin(new), "filter_parent_mz"].isna().all()
    assert SC.filter_verdict(st)["filters_readable"] == SC.FILTERS_YES
    assert SC.filter_polarity(st) == "+"
    scans, rep = SC.map_precursors(st, [_list_frame()])
    assert sorted(scans.loc[scans["order"] == 1, "scan"]) == list(range(1, FS_SCANS + 1))
    assert not scans.loc[scans["scan"].isin(new), "inferred"].any()
    assert rep["inference"]["inferred"] <= rep["inference"]["headerless"]
    a = SC.ms1_average(meta, SC.map_precursors(SC.scan_table(meta), [_list_frame()])[0])
    b = SC.ms1_average(sim, scans)
    assert a == b


@pytest.mark.parametrize("hidden", [1, 3])
def test_a_sim_block_that_lost_its_rows_at_a_switch_is_never_walked_as_ms2(hidden):
    """A tSIM experiment runs beside the list's: three of its scans keep their
    trailer rows (no MS order, event 17) between experiments 2 and 3, and
    another of its blocks lost every row at the switch between experiments 1
    and 2. The list and the Full Scan event close the layout, but it holds
    no block of that experiment, and the file shows it runs: any run across a
    block switch, leading or trailing, could hide its scans. Those runs are not
    walked (counted, said) -- the hidden SIM scans are never placed as
    experiment 1's next events, their 4e6 line in no MS2 spectrum; runs inside
    one block are still placed."""
    seq = scan_sequence()
    drops = default_drops(seq)
    last1 = max(s for s, x in enumerate(seq, 1) if x[:2] == ("ms2", 0))
    last2 = max(s for s, x in enumerate(seq, 1) if x[:2] == ("ms2", 1))
    drops -= {last1, last1 + 1, last2, last2 + 1}
    meta, truth = make_tms2(drops=drops)
    seen, new1, ren1 = insert_scans(meta, last2, [_tsim(17)] * 3)
    t1 = _renumbered(truth, ren1, new1)
    _, cen = _tsim(17)
    sim, new2, ren2 = insert_scans(seen, last1, [(None, cen)] * hidden)
    t2 = _renumbered(t1, ren2, new2)
    scans, rep = SC.map_precursors(SC.scan_table(sim), [_list_frame()])
    assert rep["layout_closed"] and rep["no_order_scans"] == 3
    got = scans[scans["scan"].isin(new2)]
    assert not got["inferred"].any() and (got["spectrum"] == "").all()
    assert got["order"].isna().all()
    inf = rep["inference"]
    assert set(new2) <= set(inf["unassigned_scans"])
    # the leading and the trailing run too; the run inside experiment 3 is placed
    first, last = 1, len(sim["centroids_meta"]["time"])
    assert {first, last} <= set(inf["unassigned_scans"])
    assert inf["inferred"] == 3 and inf["runs_beside_outside_scans"] == 3
    assert inf["runs_ambiguous"] == 0
    assert _mis_keyed(scans, t2) == []
    assert any("outside the layout" in n and "leading or trailing" in n for n in rep["notes"])
    assert not any("a block has no scan with a trailer" in n for n in rep["notes"])
    res = SC.extract_payload("S", sim, lists=[_list_frame()])
    sim_line = res.spectra[(res.spectra["mz"] - 120.0808).abs() < 0.002]
    assert set(sim_line["parent_mz"]) <= {PREC[0]}


def test_two_parallel_experiments_at_one_nce_take_one_shared_list():
    """Two tMS2 experiments at one NCE (differing in another setting) that run
    in parallel -- each cycle walks experiment 1's rows, then experiment 2's
    -- with one shared list, the scan range following the precursor: the rule
    checks every row of both, so their alternation is no sign of a list that
    only divides the method's. Mapped as with the list given once per
    experiment. On a fixed range (or with the rule check off) nothing checks
    the rows: refused, saying a parallel method takes the list once per
    experiment."""
    blocks = [(0, 1, 0), (1, 1, 0)] * 4
    meta, truth = make_tms2(nces=(30.0, 30.0), blocks=blocks, drops=(), auto_low_block=None)
    st = SC.scan_table(meta)
    once, rep = SC.map_precursors(st, [_list_frame()])
    twice, _ = SC.map_precursors(st, [_list_frame(), _list_frame()])
    _check_truth(once, truth)
    pd.testing.assert_frame_equal(once, twice)
    assert rep["rule"]["checked"] and rep["rule"]["max_residual_mda"] < 0.01
    assert any("alternate in time" in n and "the scan-range rule checks every row" in n
               for n in rep["notes"])
    fixed, _ = make_tms2(nces=(30.0, 30.0), blocks=blocks, drops=(), auto_low_block=None,
                         fixed_range=True)
    for s, kw in ((SC.scan_table(fixed), {}), (st, {"rule_check": False})):
        with pytest.raises(SC.ScanMapError, match=r"alternate in time.*the list once per "
                                                  r"experiment"):
            SC.map_precursors(s, [_list_frame()], **kw)
    twice, _ = SC.map_precursors(SC.scan_table(fixed), [_list_frame(), _list_frame()])
    _check_truth(twice, truth)


@pytest.mark.parametrize("rows", [PREC[:3], PREC[:2], (PREC + (219.1380,))[3:]])
def test_a_divisor_list_on_a_rule_following_method_is_refused_by_the_rule(rows):
    """A 6-row method (NCE 5/15/30) whose scan range follows the precursor,
    given a shared list of 3 or 2 rows: the experiments it implies pair up at
    one NCE and alternate in time, and both follow the precursor -- the rule
    checks their rows, and the list misses it. The refusal names the events
    the rule puts off the list and says the list's rows are wrong; it never
    says the range does not follow the precursor, nor sends the user to give
    the list twice (refused in turn)."""
    prec = PREC + (219.1380,)
    meta, _ = make_tms2(prec=prec)
    st = SC.scan_table(meta)
    with pytest.raises(SC.ScanMapError) as e:
        SC.map_precursors(st, [_list_frame(rows)])
    msg = str(e.value)
    assert "does not follow the precursor" not in msg and "same list twice" not in msg
    assert re.search(r"do not fit the scan-range rule \(event\(s\) \d+", msg), msg
    assert "wrong list, or rows missing" in msg
    # a 1-row list: one event seen per experiment, the rule unjudged -- the old text
    with pytest.raises(SC.ScanMapError, match="does not follow the precursor here"):
        SC.map_precursors(st, [_list_frame(prec[:1])])


def test_a_full_scan_numbered_before_the_ms2_events_is_said_so():
    """A list maps an inclusion-list method's numbering (MS2 events from 1, the
    Full Scan last). A method whose Full Scan is event 0 numbers otherwise:
    refused saying so -- never a count of -1 MS2 events, and with one list per
    experiment (three lists that fit events 1-15 exactly, or two) never blamed
    on the lists."""
    meta, _ = make_tms2(drops=())
    for r in meta["stats_per_scan"].values():
        if r["Scan Node:"] == "Full Scan":
            r["Scan Event:"] = 0
    for lists in ([PREC], [PREC[:1]], [PREC] * 3, [PREC] * 2):
        with pytest.raises(SC.ScanMapError, match=r"the Full Scan is event 0, not after the MS2 "
                                                  r"events \(1-15\)") as e:
            SC.map_precursors(SC.scan_table(meta), [_list_frame(rows) for rows in lists])
        assert "-1 MS2 event" not in str(e.value) and "-1 experiment" not in str(e.value)
        assert "lists do not fit" not in str(e.value)


# ---------------------------------------------------------------------------
# review round 3: the MS1 average, only where it is read, in memory linear in
# the centroids
# ---------------------------------------------------------------------------
def _ms1_calls(monkeypatch):
    calls = []
    real = SC.average_scans

    def spy(meta, scan_numbers, **kw):
        a = real(meta, scan_numbers, **kw)
        calls.append(len(a.scans))
        if len(a.scans) == FS_SCANS:                       # the MS1 average: no matrix
            assert kw.get("per_scan") is False and a.per_scan is None
        return a
    monkeypatch.setattr(SC, "average_scans", spy)
    return calls


def test_the_ms1_scans_are_averaged_only_where_they_are_read(monkeypatch):
    """The MS1 average feeds the window inventory, and the axis offset when
    fewer than `AXIS_MIN_PARENTS` MS2 precursors read one: with --no-window
    and three fine keys reading it, the file's MS1 scans are never averaged;
    with the window, once."""
    meta, _ = make_tms2()
    calls = _ms1_calls(monkeypatch)
    res = SC.extract_payload("S", meta, lists=[_list_frame()], with_window=False)
    assert res.report["averaging"]["axis_offset"]["parents"] == 3
    assert FS_SCANS not in calls and len(calls) == len(PREC) * len(NCES)
    assert res.ms1 is None
    calls.clear()
    res = SC.extract_payload("S", meta, lists=[_list_frame()])
    assert calls.count(FS_SCANS) == 1 and len(res.window)


def test_the_ms1_average_needs_no_clusters_x_scans_matrix():
    """60 scans of 150 noise centroids each (every one its own cluster): the
    MS1 average (no per-scan matrix asked) peaks below the size of ONE dense
    clusters x scans matrix -- memory linear in the centroids -- and gives the
    numbers of the average that keeps the matrix."""
    import tracemalloc
    rng = np.random.default_rng(11)
    scans = [[(m, i, 100.0, 1e5) for m, i in zip(np.sort(rng.uniform(50, 1000, 150)),
                                                 rng.uniform(200, 5000, 150))]
             for _ in range(60)]
    meta = _payload_of(scans)
    full = SC.average_scans(meta, range(1, 61), noise_threshold=-math.inf)
    k = len(full.mz)
    assert k > 8000 and full.per_scan.shape == (k, 60)
    tracemalloc.start()
    try:
        lean = SC.average_scans(meta, range(1, 61), noise_threshold=-math.inf, per_scan=False)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert lean.per_scan is None and lean.per_scan_n is None
    assert peak < k * 60 * 8, (peak, k * 60 * 8)
    for f in ("mz", "intensity", "resolution", "signal_to_noise", "n_scans_present", "keep"):
        assert np.allclose(getattr(lean, f), getattr(full, f), rtol=1e-12, atol=0,
                           equal_nan=True), f


def test_a_seed_bin_with_no_resolution_is_joined_to_no_neighbour():
    """A seed bin whose centroids carry no resolution has no width: it stays
    apart (`cluster_centroids`) -- never joined through its neighbour's
    FWHM. A payload where only some scans carry resolutions takes the nominal
    resolving power for the others (said): no peak of theirs is left with a
    NaN resolution, which would switch the side-lobe rules off unsaid."""
    mz, inten = np.array([200.0, 200.0003]), np.array([1e3, 1e3])      # 1.5 ppm: two seeds
    for res, labels in (([math.nan, 1e5], [0, 1]), ([1e5, math.nan], [0, 1]),
                        ([1e5, 1e5], [0, 0]), ([math.nan, math.nan], [0, 1])):
        assert SC.cluster_centroids(mz, inten, np.array(res)).tolist() == labels, res
    meta = _payload_of([[(200.0, 1000, 10, 1e5)],
                        [(200.0003, 1000, 10, 1e5), (250.0, 900, 10, 1e5)]])
    del meta["centroids_meta"]["data"][1]["resolutions"]
    a = SC.average_scans(meta, [1, 2], noise_threshold=0)
    assert len(a.mz) == 3 and np.isnan(a.resolution[1:]).all()
    # a file: one scan in three without resolutions
    meta, _ = make_tms2(drops=())
    for c in meta["centroids_meta"]["data"][1::3]:
        del c["resolutions"]
        c["mzs"].append(c["mzs"][-1] + 0.5)                   # a peak of that scan alone
        c["intensities"].append(5e4)
        c["noises"].append(NOISE)
    res = SC.extract_payload("S", meta, lists=[_list_frame()])
    avg = res.report["averaging"]
    assert avg["resolution_source"] == ("per-scan centroids; nominal (FT Resolution 120000 at "
                                        "m/z 200) where a scan carries none")
    lacking = set(range(2, len(meta["centroids_meta"]["data"]) + 1, 3))
    n_ms1 = len(lacking & set(range(1, FS_SCANS + 1)))
    assert avg["scans_without_resolutions"] == len(lacking) - n_ms1 and n_ms1 == 1
    assert avg["ms1_scans_without_resolutions"] == n_ms1
    assert res.spectra["resolution"].notna().all() and res.window["resolution"].notna().all()
    # the MS1 average itself, every cluster (the window holds a few of them):
    # the MS1 scan without resolutions takes the MS1 rows' nominal value
    assert np.isfinite(res.ms1["resolution"]).all()
    assert any("MS2 scan(s) carry no per-centroid resolution" in n for n in res.notes)
    assert any("MS1 scan(s) carry no per-centroid resolution" in n for n in res.notes)
    # a spectrum whose scans have neither a trailer row (placed by the cycle)
    # nor resolutions: the MS2 scans' resolving power fills it
    seq = scan_sequence()
    gone = [s for s, x in enumerate(seq, 1) if x[:3] == ("ms2", 0, 2)]
    meta, truth = make_tms2(drops=gone)
    for s in gone:
        del meta["centroids_meta"]["data"][s - 1]["resolutions"]
    res = SC.extract_payload("S", meta, lists=[_list_frame()])
    key = SRC.spectrum_key(PREC[2], "hcd5.00")
    assert res.scans.loc[res.scans["scan"].isin(gone), "spectrum"].eq(key).all()
    sp = res.spectra[res.spectra["spectrum"] == key]
    assert len(sp) and np.allclose(sp["resolution"], R200 * np.sqrt(200 / sp["mz"]), rtol=1e-9)


@pytest.mark.parametrize("ms1_r200", [120000, None])
def test_only_the_ms1_scans_without_resolutions_are_said_as_ms1(ms1_r200):
    """The tMS2 rows read FT Resolution 60000, the Full Scan rows 120000 (or
    none); only the 4 MS1 scans' centroids lack resolutions. No MS2 centroid
    needs a nominal value: the spectra's source stays 'per-scan centroids'.
    The MS1 average fills its own from the MS1 rows' 120000 -- recorded and
    said as the MS1 scans' -- or, with no FT Resolution on those rows, its
    window peaks read NaN, said so: never a fill at the MS2 groups' 60000
    that never happened."""
    meta, truth = make_tms2(drops=())
    for r in meta["stats_per_scan"].values():
        if r["Scan Node:"] == "tMS2":
            r["FT Resolution:"] = 60000
        elif ms1_r200 is None:
            del r["FT Resolution:"]
    ms1_scans = truth.loc[truth["order"] == 1, "scan"].tolist()
    for s_ in ms1_scans:
        del meta["centroids_meta"]["data"][s_ - 1]["resolutions"]
    res = SC.extract_payload("S", meta, lists=[_list_frame()])
    avg = res.report["averaging"]
    assert avg["resolution_source"] == "per-scan centroids"
    assert not avg.get("scans_without_resolutions") and not avg.get("nominal_r200")
    assert avg["ms1_scans_without_resolutions"] == len(ms1_scans)
    assert not any(n.startswith(f"{len(ms1_scans)} scan(s) carry no per-centroid")
                   for n in res.notes)
    w = res.window
    assert len(w)
    if ms1_r200:
        assert avg["ms1_resolution_source"] == "nominal: FT Resolution 120000 at m/z 200"
        assert np.allclose(w["resolution"], ms1_r200 * np.sqrt(200 / w["mz"]), rtol=1e-9)
        assert (f"{len(ms1_scans)} MS1 scan(s) carry no per-centroid resolution: the MS1 average "
                "takes the nominal resolving power (FT Resolution 120000 at m/z 200)") in res.notes
    else:
        assert avg["ms1_resolution_source"] == "none"
        assert w["resolution"].isna().all()
        assert any(n.startswith(f"{len(ms1_scans)} MS1 scan(s) carry no per-centroid resolution "
                                "and the MS1 rows read no single FT Resolution") for n in res.notes)


# ---------------------------------------------------------------------------
# review round 3: rules the code holds that no test pinned
# ---------------------------------------------------------------------------
PEREXP = [(120.0808, 137.1325), (155.1430, 181.0859, 203.1430), (99.0441, 113.0597)]


def test_per_experiment_lists_without_a_full_scan_leave_the_layout_open():
    """One list per experiment and no Full Scan event: nothing says the lists
    are every experiment of the method. The user passes experiments 1 and 2's
    lists; experiment 3 ran with no trailer row. The trailing run is not
    walked on inside experiment 2 (its scans would land in another ion's
    spectrum at another NCE): the layout stays open, the run unassigned."""
    blocks = [(0, 2, 1), (1, 2, 2), (2, 1, 0)]
    seq = scan_sequence([len(p) for p in PEREXP], fs_scans=0, blocks=blocks)
    exp3 = {s for s, x in enumerate(seq, 1) if x[:2] == ("ms2", 2)}
    meta, truth = make_tms2(precs=PEREXP, fs_scans=0, blocks=blocks, drops=exp3 | {3},
                            auto_low_block=None)
    scans, rep = SC.map_precursors(SC.scan_table(meta), [_list_frame(p) for p in PEREXP[:2]])
    assert not rep["layout_closed"]
    assert _mis_keyed(scans, truth) == []
    assert exp3 <= set(rep["inference"]["unassigned_scans"])
    assert (scans.loc[scans["scan"].isin(exp3), "spectrum"] == "").all()


def test_per_experiment_lists_must_reach_every_event_seen():
    """The same method with experiment 3's trailer rows: its events (6-7) lie
    past the 5 rows of the 2 lists given -- refused, never accepted with its
    scans in no spectrum."""
    blocks = [(0, 2, 1), (1, 2, 2), (2, 2, 0)]
    meta, _ = make_tms2(precs=PEREXP, fs_scans=0, blocks=blocks, drops={3}, auto_low_block=None)
    with pytest.raises(SC.ScanMapError, match=r"MS2 event 7 lies past the 5 rows of the 2 lists"):
        SC.map_precursors(SC.scan_table(meta), [_list_frame(p) for p in PEREXP[:2]])


def test_the_list_and_readable_filters_agree_within_a_millidalton():
    """A fixed range (the scan-range rule checks nothing) and partly readable
    filters: a list row typed 137.13 for 137.1325 (2.5 mDa) or every row
    1.5 mDa off is refused -- taken, one ion's breakdown would split over two
    parent keys (the filter's and the list's). 0.5 mDa off is one parent."""
    meta, _ = make_tms2(garble=0.6, fixed_range=True, drops=(), seed=3)
    st = SC.scan_table(meta)
    assert SC.filter_verdict(st)["filters_readable"] == SC.FILTERS_PARTLY
    for rows in ((PREC[0], 137.13, *PREC[2:]), tuple(p + 0.0015 for p in PREC)):
        with pytest.raises(SC.ScanMapError, match="disagrees with the scan filters"):
            SC.map_precursors(st, [_list_frame(rows)])
    scans, _ = SC.map_precursors(st, [_list_frame(tuple(p + 0.0005 for p in PREC))])
    ms2 = scans[scans["order"] == 2]
    assert ms2["parent_mz"].nunique() == len(PREC)
    assert set(ms2["precursor_source"]) == {SC.PRECURSOR_FILTER, SC.PRECURSOR_LIST}


@pytest.mark.parametrize("old,new", [
    ("FTMS + p NSI t ms2 137.1325@hcd15.00", "FTMS + p NSI t ms2 181.0859@hcd30.00"),   # energy
    ("FTMS + p NSI t ms2 137.1325@hcd15.00", "ITMS + c NSI t ms2 181.0859@hcd15.00"),   # analyzer
])
def test_a_filter_that_disagrees_with_its_trailer_names_no_precursor(old, new):
    """A readable-looking filter naming another precursor, but disagreeing with
    its trailer (energy, analyzer): it names nothing -- the scan keeps its
    event's precursor, with the list and without (where the event's other,
    agreeing filters name it), never the filter's ion."""
    meta, truth = make_tms2(garble=0.0, drops=())
    st0 = SC.scan_table(meta)
    s = int(st0.loc[(st0["order"] == 2) & (st0["event"] == 7), "scan"].iloc[0])
    r = meta["stats_per_scan"][str(s)]
    assert old in r["ScanType"]
    r["ScanType"] = r["ScanType"].replace(old, new)
    st = SC.scan_table(meta)
    row = st[st["scan"] == s].iloc[0]
    assert not row["filter_ok"] and math.isnan(row["filter_parent_mz"])
    for lists in ([_list_frame()], None):
        scans, _ = SC.map_precursors(st, lists)
        _check_truth(scans, truth)


def test_readable_filters_of_both_signs_name_no_polarity():
    """Every filter agrees with its trailer, but they carry '+' and '-' (a
    dual-polarity method): the filters name no polarity -- never the first
    sign read."""
    meta, _ = make_tms2(garble=0.0)
    for k, r in meta["stats_per_scan"].items():
        if int(k) % 2:
            r["ScanType"] = r["ScanType"].replace("FTMS + p", "FTMS - p")
    st = SC.scan_table(meta)
    v = SC.filter_verdict(st)
    assert v["filters_readable"] == SC.FILTERS_YES and v["polarities_in_filters"] == "+-"
    assert SC.filter_polarity(st) is None


@pytest.mark.parametrize("edge", ["low", "high"])
def test_a_walk_onto_the_full_scan_is_checked_against_its_range(edge):
    """The Full Scan records m/z 100-600 (or 40-150: narrower than the
    precursors); experiment 1's block is entered at its third row behind
    three scans with no trailer row. The only walk from the last Full Scan to
    experiment 1's first event calls them Full Scans -- their centroids below
    m/z 100 (above m/z 150) contradict it: unassigned, never averaged into
    the MS1 window. Both edges of the Full Scans' range are checked."""
    blocks = [(0, 2, 1, 2), (1, 2, 2), (2, 2, 1)]
    seq = scan_sequence(blocks=blocks)
    first1 = next(s for s, x in enumerate(seq, 1) if x[:2] == ("ms2", 0))
    drops = {first1, first1 + 1, first1 + 2}
    meta, truth = make_tms2(blocks=blocks, drops=drops, auto_low_block=None)
    for s, x in enumerate(seq, 1):
        if x[0] == "fs":
            c = meta["centroids_meta"]["data"][s - 1]
            keep = [i for i, m in enumerate(c["mzs"]) if (m >= 100 if edge == "low" else m <= 150)]
            for k in ("mzs", "intensities", "resolutions", "noises"):
                c[k] = [c[k][i] for i in keep]
            r = meta["stats_per_scan"][str(s)]
            if edge == "low":
                r["LowMass"] = 100.0
                r["ScanType"] = r["ScanType"].replace("[40.0000", "[100.0000")
            else:
                r["HighMass"] = 150.0
                r["ScanType"] = r["ScanType"].replace("-600.0000]", "-150.0000]")
    walked = SC.scan_table(meta).set_index("scan").loc[sorted(drops)]
    if edge == "high":       # every walked scan inside the low edge, one above the high one
        assert (walked["min_centroid_mz"] >= 40).all() and (walked["max_centroid_mz"] > 150).any()
    scans, rep = SC.map_precursors(SC.scan_table(meta), [_list_frame()])
    inf = rep["inference"]
    assert inf["runs_failed_range"] == 1 and inf["unassigned_scans"] == sorted(drops)
    assert int((scans["order"] == 1).sum()) == FS_SCANS
    assert _mis_keyed(scans, truth) == []


def test_an_ms2_scan_with_no_precursor_is_said():
    """A data-dependent event (its readable filters name two precursors) one of
    whose scans has a garbled filter: that scan has no precursor source and is
    in no spectrum -- counted and said, never dropped silently."""
    meta, _ = make_tms2(garble=0.0, drops=(), fixed_range=True)
    rows = meta["stats_per_scan"]
    ev2 = sorted(int(k) for k, r in rows.items() if r["Scan Event:"] == 2)
    rows[str(ev2[0])]["ScanType"] = rows[str(ev2[0])]["ScanType"].replace(
        f"{PREC[1]:.4f}@", f"{PREC[3]:.4f}@")
    r1 = rows[str(ev2[1])]
    r1["ScanType"] = "ITMS - c EI Full ms " + r1["ScanType"].rsplit(" ", 1)[1]
    scans, rep = SC.map_precursors(SC.scan_table(meta))
    assert rep["unmapped_ms2_scans"] == [ev2[1]]
    assert scans.set_index("scan").at[ev2[1], "spectrum"] == ""
    note = f"1 MS2 scan(s) with no precursor (scan(s) {ev2[1]})"
    assert any(n.startswith(note) for n in rep["notes"])
    res = SC.extract_payload("S", meta)
    assert any(n.startswith(note) for n in res.notes)


@pytest.mark.parametrize("delta_mda,mapped", [(0.5, True), (1.5, False)])
def test_one_events_filters_name_one_precursor_within_a_millidalton(delta_mda, mapped):
    """An event's readable filters name ONE precursor when they agree within
    `SOURCE_MATCH_MDA` (1 mDa): a scan of that event with a garbled filter
    takes it. Farther apart they name one per scan (a data-dependent event):
    the garbled scan has no precursor -- in no spectrum, counted."""
    meta, _ = make_tms2(garble=0.0, drops=(), fixed_range=True)
    rows = meta["stats_per_scan"]
    ev2 = sorted(int(k) for k, r in rows.items() if r["Scan Event:"] == 2)
    r0 = rows[str(ev2[0])]
    r0["ScanType"] = r0["ScanType"].replace(f"{PREC[1]:.4f}@", f"{PREC[1] + delta_mda / 1e3:.4f}@")
    r1 = rows[str(ev2[1])]
    r1["ScanType"] = "ITMS - c EI Full ms " + r1["ScanType"].rsplit(" ", 1)[1]
    st = SC.scan_table(meta)
    assert st.set_index("scan").at[ev2[0], "filter_ok"]
    assert not st.set_index("scan").at[ev2[1], "filter_ok"]
    scans, rep = SC.map_precursors(st)
    r = scans.set_index("scan").loc[ev2[1]]
    if mapped:
        assert ev2[1] not in rep["unmapped_ms2_scans"]
        assert r["precursor_source"] == SC.PRECURSOR_FILTER and r["spectrum"] != ""
        assert abs(r["parent_mz"] - PREC[1]) * 1e3 <= 1.0
    else:
        assert rep["unmapped_ms2_scans"] == [ev2[1]] and r["spectrum"] == ""


def test_an_acquisition_that_ends_inside_its_last_experiment_is_said():
    """The file stops after the last experiment ran its first 2 rows once:
    events 13-15 (rows 3-5 at NCE 30) were never scanned. The report counts
    each experiment's rows scanned and names the events never scanned, the
    notes say the acquisition ends inside experiment 3 and which parents
    have no NCE-30 spectrum, and those parents' rows say so -- never an
    experiment of 5 rows silently holding 2. Events never scanned lend their
    isolation width to no scan: no width note."""
    cycles = ((2, 2), (2, 3), (0, 2))
    meta, truth = make_tms2(drops=(), cycles=cycles)
    scans, rep = SC.map_precursors(SC.scan_table(meta), [_list_frame()])
    assert [x["rows"] for x in rep["experiments"]] == [5, 5, 5]
    assert [x["rows_scanned"] for x in rep["experiments"]] == [5, 5, 2]
    assert rep["events_never_scanned"] == [13, 14, 15]
    last = int(truth["scan"].max())
    note = next(n for n in rep["notes"] if "the acquisition ends inside it" in n)
    assert note.startswith(f"experiment 3 (NCE 30, events 11-15): rows 3-5 never scanned -- the "
                           f"acquisition ends inside it (last MS2 scan {last} = event 12, row 2)")
    assert "3 parent(s) have no NCE-30 spectrum" in note
    assert rep["missing_energies"] == {repr(p): [30.0] for p in PREC[2:]}
    assert not any("isolation width" in n for n in rep["notes"])
    res = SC.extract_payload("S", meta, lists=[_list_frame()])
    for p in PREC:
        n = res.parents.loc[np.isclose(res.parents["parent_mz"], p), "note"]
        said = n.str.contains("no spectrum at NCE 30: its event was never scanned")
        assert said.all() if p in PREC[2:] else not said.any(), p


def test_an_event_seen_only_by_the_cycle_takes_its_experiments_width_said_only_if_it_differs():
    """Every scan of event 7 lost its trailer row; the cycle places them. The
    event takes its experiment's isolation width -- said only where the
    trailers read more than one width (experiment 2's are 1.0 Da, the
    others' 0.7), else counted in the report alone."""
    seq = scan_sequence()
    ev7 = [s for s, x in enumerate(seq, 1) if x[:3] == ("ms2", 1, 1)]
    meta, _ = make_tms2(drops=ev7)
    scans, rep = SC.map_precursors(SC.scan_table(meta), [_list_frame()])
    assert rep["inference"]["inferred"] == len(ev7) and rep["events_width_from_experiment"] == [7]
    assert not any("isolation width" in n for n in rep["notes"])
    for r in meta["stats_per_scan"].values():
        if r["Scan Node:"] == "tMS2" and 6 <= r["Scan Event:"] <= 10:
            p = PREC[r["Scan Event:"] - 6]
            r["MS2 Isolation Width:"] = 1.0
            r["ScanType"] = r["ScanType"].replace(f"-{r['HighMass']:.4f}]",
                                                  f"-{_high(p, 1.0):.4f}]")
            r["HighMass"] = _high(p, 1.0)
    scans, rep = SC.map_precursors(SC.scan_table(meta), [_list_frame()])
    assert scans.loc[scans["scan"].isin(ev7), "iso_width"].eq(1.0).all()
    assert any(n.startswith("1 event(s) seen only through scans placed by the scan cycle (7) take "
                            "their experiment's isolation width") for n in rep["notes"])


def test_a_payload_without_any_resolution_and_an_empty_window_are_said():
    """No centroid carries a resolution and no trailer row says FT Resolution:
    every peak's resolution is NaN -- said. MS1 centroids nowhere near a
    precursor: the window inventory is empty -- said."""
    meta, _ = make_tms2(drops=())
    for c in meta["centroids_meta"]["data"]:
        del c["resolutions"]
    for r in meta["stats_per_scan"].values():
        del r["FT Resolution:"]
    res = SC.extract_payload("S", meta, lists=[_list_frame()])
    assert res.report["averaging"]["resolution_source"] == "none"
    assert any(n.startswith("no resolution: the centroids carry none") for n in res.notes)
    meta, truth = make_tms2(drops=())
    for s in truth.loc[truth["order"] == 1, "scan"]:
        c = meta["centroids_meta"]["data"][s - 1]
        c["mzs"] = [m + 3.0 for m in c["mzs"]]
    res = SC.extract_payload("S", meta, lists=[_list_frame()])
    assert len(res.window) == 0
    assert "window inventory empty: no MS1 centroid inside any window" in res.notes
