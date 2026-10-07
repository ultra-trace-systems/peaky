"""Rule K, the veto side: a labelled [Y+^NO3]- reading is refuted when the 14N
twin its reagent's impurity must carry is missing (EVIDENCE_LEVELS §3
`label_veto`, §4.2; batch/label_twins.py module docstring).

These tests pin the numbers of the twin test on small hand-made series: the
twin fraction, the batch's detection curve (13C lines of committed M0s of >= 4
C, 1.07 % per carbon, left-closed bins, an empty bin taking the populated bin
below, 0 below the first), the spectra that count (the labelled line seen, the
twin's m/z at or above the spectrum's scan start, p >= PMIN), E = sum of p, the
share boundaries, the 2-ppm tallest-peak window, the note, the table contract,
the funnel, and `veto()` reading a table back from CSV.

Run: pytest tests/test_label_twins_veto.py -q
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from peaky.assignment import evidence as EV
from peaky.batch import label_twins as LT
from peaky.chem import chemistry as C
from tests.test_evidence import ledger, m0
from tests.test_label_twins import LABEL, NO3, NO3L, _frames, _measure, _refs_spec, _row, _series

T0 = pd.Timestamp("2021-02-18 00:00", tz="UTC")
N = 120
N_BINS = len(LT.PDET_EDGES) - 1
F0 = 0.02 / 0.98
START = 50.0                                   # every hand-made spectrum opens with a peak here
Y = "C12H14O2"
MZ15 = C.ion_mz(Y, NO3L)
TWIN = MZ15 - LT.DELTA_15N


def _ts(lines, n=N, start=START) -> pd.DataFrame:
    """A stamped series of n spectra. lines: [(mz, heights[n]; NaN = no peak)];
    start: the scan-start peak of each spectrum (scalar or per spectrum; NaN or
    None = none)."""
    starts = np.broadcast_to(np.asarray(np.nan if start is None else start, float), (n,))
    rows = []
    for i in range(n):
        sid, when = f"s{i:03d}", T0 + pd.Timedelta(minutes=20 * i)
        if np.isfinite(starts[i]):
            rows.append(dict(sample_item_id=sid, datetime_utc=when, mz=float(starts[i]), height=1.0))
        for mz, h in lines:
            if np.isfinite(h[i]):
                rows.append(dict(sample_item_id=sid, datetime_utc=when, mz=mz, height=float(h[i])))
    return pd.DataFrame(rows)


def _heights(n_on, h=10000.0, n=N) -> np.ndarray:
    """h in the first n_on spectra, NaN after."""
    return np.where(np.arange(n) < n_on, h, np.nan)


def _veto(h15, twin, curve, f=F0, extra=(), start=START, twin_mz=TWIN) -> pd.Series:
    """_twin_veto of Y on a hand-made series: its 15N line at heights h15, a
    twin peak (200 counts) at twin_mz wherever `twin` is True."""
    tw = np.where(np.asarray(twin, bool), 200.0, np.nan)
    tr = LT._Traces(_ts([(MZ15, np.asarray(h15, float)), (twin_mz, tw), *extra], start=start))
    return LT._twin_veto(tr, [Y], f, np.asarray(curve, float)).iloc[0]


def _ones() -> np.ndarray:
    return np.ones(N_BINS)


def _step(e) -> np.ndarray:
    """A curve of 0 below the bin holding e and 1 from it."""
    return np.where(np.arange(N_BINS) >= _bin(e), 1.0, 0.0)


def _bin(e) -> int:
    return int(np.searchsorted(LT.PDET_EDGES, e, side="right") - 1)


def _curve(families, n=N) -> np.ndarray:
    """The detection curve of committed M0s. families: [(neutral, adduct, H0[n],
    has13C[n], extra m0 kwargs)]; the 13C line is written where has13C."""
    lines, rows = [], []
    for k, (neutral, adduct, h0, c13, kw) in enumerate(families):
        mz = C.ion_mz(neutral, adduct)
        nc = C.parse_formula(kw.get("ion") or neutral).get("C", 0)
        lines.append((mz, h0))
        lines.append((mz + LT.D13C, np.where(np.asarray(c13, bool) & np.isfinite(h0),
                                            np.nan_to_num(h0) * LT.C13_PER_C * nc, np.nan)))
        rows.append(m0(f"p{k}", neutral, adduct=adduct, mz=mz, **kw))
    tr = LT._Traces(_ts(lines, n))
    return LT._pdet_curve(tr, LT._committed({"f": EV.trim(ledger(rows))}))


def _const(h, n=N) -> np.ndarray:
    return np.full(n, float(h))


# --------------------------------------------------------------------------- the twin fraction
def test_an_undeclared_purity_reads_the_isotopes_default(monkeypatch):
    """f = (1 - p) / p; a purity that is None, 0 or not an attribute reads isotopes.LABEL_PURITY_15N."""
    from peaky.chem import isotopes
    monkeypatch.setattr(isotopes, "LABEL_PURITY_15N", 0.99)
    for prof in (SimpleNamespace(purity=None), SimpleNamespace(purity=0.0), SimpleNamespace()):
        assert LT.twin_fraction(prof) == pytest.approx(0.01 / 0.99), prof
    assert LT.twin_fraction(SimpleNamespace(purity=0.95)) == pytest.approx(0.05 / 0.95)


# --------------------------------------------------------------------------- the detection curve
def test_the_curve_counts_lines_of_four_carbons_and_up():
    """PDET_MIN_C = 4: a 4-C ion populates its bin, a 3-C ion leaves the curve empty."""
    c4 = _curve([("C4H6O4", "[M-H]-", _const(1500), np.ones(N), {})])            # e = 64.2
    assert list(c4) == [0.0, 0.0] + [1.0] * (N_BINS - 2)
    c3 = _curve([("C3H4O4", "[M-H]-", _const(1500), np.ones(N), {})])            # e = 48.2
    assert (c3 == 0.0).all()


def test_the_curve_counts_the_ions_carbons():
    """The carbons of the ION: malonic acid (3 C) on carbonate is a 4-C ion C4H4O7."""
    c = _curve([("C3H4O4", "[M+CO3]-", _const(1500), np.ones(N), {"ion": "C4H4O7"})])
    assert list(c) == [0.0, 0.0] + [1.0] * (N_BINS - 2)


def test_the_curve_expects_1_07_percent_per_carbon():
    """e = 0.0107 x nC x H0: C10 at 1800 counts expects 192.6, bin [160, 200)."""
    c = _curve([("C10H16O4", "[M-H]-", _const(1800), np.ones(N), {})])
    assert _bin(192.6) == 7
    assert list(c) == [0.0] * 7 + [1.0] * (N_BINS - 7)


def test_the_curve_bins_are_left_closed():
    """A line whose expected 13C height is exactly an edge (50.0) sits in the bin above it."""
    h = 584.1121495327103
    assert LT.C13_PER_C * 8.0 * h == 50.0
    c = _curve([("C8H12O4", "[M-H]-", _const(h), np.ones(N), {})])
    assert c[_bin(40.0)] == 0.0 and c[_bin(50.0)] == 1.0


def test_the_top_bin_holds_the_brightest_lines():
    """A line expected at >= 1000 counts populates the top bin alone; the bins below stay 0."""
    c = _curve([("C10H16O4", "[M-H]-", _const(12000), np.ones(N), {})])        # e = 1284
    assert list(c) == [0.0] * (N_BINS - 1) + [1.0]


def test_an_empty_bin_inherits_the_populated_bin_below():
    """A gap between populated bins takes the bin below; 0 below the first populated bin."""
    lo = ("C8H12O4", "[M-H]-", _const(700), np.ones(N), {})                       # e = 59.9: seen
    hi = ("C10H16O4", "[M-H]-", _const(4000), np.arange(N) % 2 == 0, {})          # e = 428: half
    c = _curve([lo, hi])
    assert list(c) == [0.0, 0.0] + [1.0] * 9 + [0.5] * 3


def test_a_measured_miss_is_a_zero_not_a_gap():
    """A bin whose lines are never detected reads 0, not the populated bin below it."""
    lo = ("C8H12O4", "[M-H]-", _const(700), np.ones(N), {})                       # e = 59.9: seen
    mid = ("C10H16O4", "[M-H]-", _const(1100), np.zeros(N), {})                   # e = 117.7: never
    c = _curve([lo, mid])
    assert list(c) == [0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def test_a_line_absent_from_a_spectrum_says_nothing_there():
    """Only spectra holding the M0 line enter its bin: absences are not misses."""
    always = ("C8H12O4", "[M-H]-", _const(300), np.ones(N), {})                   # e = 25.7
    part = ("C9H14O4", "[M-H]-", _heights(30, 500.0), np.ones(N), {})             # e = 48.2, 30 spectra
    c = _curve([always, part])
    assert list(c) == [1.0] * N_BINS


def test_a_pair_committed_in_two_files_counts_once():
    """The curve reads each committed (neutral, adduct) pair once, however many files commit it."""
    a = ("C8H12O4", "[M-H]-", _const(700), np.ones(N), {})                        # e = 59.9: seen
    b = ("C8H10O4", "[M-H]-", _const(650), np.zeros(N), {})                       # e = 55.6: never
    lines = []
    for neutral, adduct, h0, c13, _ in (a, b):
        mz = C.ion_mz(neutral, adduct)
        lines += [(mz, h0), (mz + LT.D13C, np.where(c13, h0 * LT.C13_PER_C * 8, np.nan))]
    row_a = m0("pa", a[0], adduct=a[1], mz=C.ion_mz(a[0], a[1]))
    row_b = m0("pb", b[0], adduct=b[1], mz=C.ion_mz(b[0], b[1]))
    frames = {"f1": EV.trim(ledger([row_a, row_b])), "f2": EV.trim(ledger([row_a]))}
    c = LT._pdet_curve(LT._Traces(_ts(lines)), LT._committed(frames))
    assert c[_bin(55.6)] == 0.5


def test_ion_only_rows_never_shape_the_curve():
    """An ion-only M0 ([M]-. from the ion_only stage) is not a committed pair of the curve."""
    good = ("C8H12O4", "[M-H]-", _const(700), np.ones(N), {})                     # e = 59.9
    io = ("C10H16O4", "[M]-.", _const(1100), np.zeros(N), {"method": "ion_only:ea"})   # e = 117.7
    c = _curve([good, io])
    assert c[_bin(117.7)] == 1.0 and list(c) == [0.0, 0.0] + [1.0] * (N_BINS - 2)


def test_measure_builds_the_curve_from_every_committed_m0():
    """measure() builds the curve from every committed M0: seen dim [M-H]- 13C lines make a dim cluster refutable."""
    spec = _refs_spec(**{Y: {"phase": 1.0, "q": None, "h15": 2500.0}})
    ts = _series(spec)
    extra, rows = [], []
    for k, n in enumerate(["C10H16O3", "C9H14O4", "C8H10O4", "C10H14O4"]):
        extra.append(m0(f"d{k}", n, adduct="[M-H]-", ion=n, mz=C.ion_mz(n, "[M-H]-")))
        for i in range(N):
            h = 400.0 * (1.2 + 0.8 * np.sin(i / 7.0 + k))            # e 4..86: bins 0-4
            for mz, hh in ((C.ion_mz(n, "[M-H]-"), h),
                           (C.ion_mz(n, "[M-H]-") + LT.D13C, h * LT.C13_PER_C * C.parse_formula(n)["C"])):
                rows.append(dict(sample_item_id=f"s{i:03d}", datetime_utc=T0 + pd.Timedelta(minutes=20 * i),
                                 mz=mz, height=hh, area=hh))
    frames = dict(_frames(list(spec)), f2=EV.trim(ledger(extra)))
    ts = pd.concat([ts, pd.DataFrame(rows)], ignore_index=True)
    r = _row(LT.measure(ts, frames, LABEL, log=lambda *a: None), Y, NO3L)
    assert (r.twin_verdict, bool(r.veto), r.E, r.obs) == ("refuted", True, 120.0, 0)


# --------------------------------------------------------------------------- the spectra that count
def test_E_MIN_is_inclusive_at_three_expected_detections():
    """E >= 3 is testable (refuted without a twin); E = 2 is untestable."""
    r = _veto(_heights(3), np.zeros(N), _ones())
    assert (r.E, r.obs, r.twin_verdict, bool(r.veto)) == (3.0, 0, "refuted", True)
    r = _veto(_heights(2), np.zeros(N), _ones())
    assert (r.E, r.obs, r.twin_verdict, bool(r.veto)) == (2.0, 0, "untestable", False)
    r = _veto(_heights(10), np.zeros(N), _ones())
    assert (r.E, r.twin_verdict) == (10.0, "refuted")


@pytest.mark.parametrize("seen,verdict", [(24, "refuted"), (25, "unclear"), (59, "unclear"), (60, "passes")])
def test_the_share_boundaries_are_inclusive(seen, verdict):
    """refuted at obs <= 0.2 E, passes at obs >= 0.5 E, unclear between (E = 120)."""
    r = _veto(_const(10000), np.arange(N) < seen, _ones())
    assert (r.E, r.obs, r.twin_verdict, bool(r.veto)) == (120.0, seen, verdict, verdict == "refuted")


def test_pmin_is_inclusive_and_E_sums_the_probabilities():
    """A spectrum counts at p >= 0.5 and adds p to E (not 1)."""
    for p, E in ((0.5, 5.0), (0.7, 7.0)):
        r = _veto(_heights(10), np.zeros(N), np.full(N_BINS, p))
        assert (r.E, r.twin_verdict) == (pytest.approx(E), "refuted"), p
    r = _veto(_heights(10), np.zeros(N), np.full(N_BINS, 0.49))
    assert (r.E, r.obs, r.twin_verdict) == (0.0, 0, "untestable")


def test_only_spectra_where_the_twin_is_expected_count():
    """Spectra with p < PMIN add nothing to E, and a twin seen only there is not counted."""
    curve = np.where(np.arange(N_BINS) >= _bin(160.0), 1.0, 0.3)
    h = np.where(np.arange(N) % 2 == 0, 20000.0, 1000.0)          # e 408 (p 1) / 20.4 (p 0.3)
    r = _veto(h, h < 5000.0, curve)                                 # the twin only in the dim spectra
    assert (r.E, r.obs, r.twin_verdict, bool(r.veto)) == (60.0, 0, "refuted", True)


def test_an_unseen_labelled_line_expects_nothing_and_counts_nothing():
    """Spectra without the 15N line add nothing to E, and a twin there is not counted."""
    r = _veto(_heights(30), np.ones(N), _ones())                    # the twin in all 120, Y in 30
    assert (r.E, r.obs, r.twin_verdict) == (30.0, 30, "passes")
    r = _veto(_heights(30), np.arange(N) < 30, _ones())
    assert (r.E, r.obs, r.twin_verdict) == (30.0, 30, "passes")


def test_the_top_bin_is_read():
    """An expected twin above 1000 counts reads the curve's top bin."""
    curve = np.zeros(N_BINS)
    curve[-1] = 1.0
    r = _veto(_const(100000), np.zeros(N), curve)                   # e = 2041
    assert (r.E, r.twin_verdict) == (120.0, "refuted")


def test_the_veto_bins_are_left_closed_and_scale_with_f():
    """e = f x H0 exactly on an edge (0.25 x 200 = 50) reads the bin above it."""
    r = _veto(_const(200), np.zeros(N), _step(50.0), f=0.25)
    assert (r.E, r.twin_verdict, r.f) == (120.0, "refuted", 0.25)
    r = _veto(_const(200), np.zeros(N), _step(50.0), f=0.2)         # e = 40: below the step
    assert (r.E, r.twin_verdict) == (0.0, "untestable")


def test_the_labelled_height_is_the_tallest_peak_in_the_window():
    """A tiny neighbour within 2 ppm of the 15N line never sets its height."""
    r = _veto(_const(20000), np.zeros(N), _step(160.0), extra=[(MZ15 * (1 + 1e-6), _const(5.0))])
    assert (r.E, r.twin_verdict) == (120.0, "refuted")


@pytest.mark.parametrize("ppm,seen", [(1.8, True), (-1.8, True), (2.2, False), (-2.2, False)])
def test_the_twin_window_is_two_ppm(ppm, seen):
    """The twin is found within +-2 ppm of mz15 - 0.99703, on both sides."""
    r = _veto(_const(10000), np.ones(N), _ones(), twin_mz=TWIN * (1 + ppm * 1e-6))
    assert (r.obs, r.twin_verdict) == ((120, "passes") if seen else (0, "refuted"))


# --------------------------------------------------------------------------- the scan start
def test_a_twin_at_the_scan_start_was_measured():
    """A twin that is itself each spectrum's lowest peak (at its exact m/z) is not below the scan start."""
    r = _veto(_const(10000), np.ones(N), _ones(), start=None)
    assert (r.E, r.obs, r.twin_verdict) == (120.0, 120, "passes")


def test_a_twin_below_the_scan_start_expects_nothing():
    """A scan starting between the twin and the 15N line adds nothing to E, whatever curve[0] says."""
    r = _veto(_const(10000), np.zeros(N), _ones(), start=TWIN + 0.5)
    assert (r.E, r.obs, r.twin_verdict, bool(r.veto)) == (0.0, 0, "untestable", False)


def test_each_spectrum_keeps_its_own_lower_scan_start():
    """70 spectra start above the twin (the batch median), 50 below it: only the 50 count."""
    start = np.where(np.arange(N) < 50, TWIN - 1.0, np.nan)         # the other 70 open at Y's line
    tr = LT._Traces(_ts([(MZ15, _const(10000))], start=start))
    assert list(tr.low) == [TWIN - 1.0] * 50 + [MZ15] * 70
    r = _veto(_const(10000), np.zeros(N), _ones(), start=start)
    assert (r.E, r.obs, r.twin_verdict) == (50.0, 0, "refuted")


def test_a_sparse_spectrum_keeps_the_batch_scan_start():
    """Each spectrum's scan start is capped at the batch MEDIAN lowest peak: a sparse spectrum counts."""
    start = np.where(np.arange(N) < 61, TWIN - 0.1, np.nan)
    tr = LT._Traces(_ts([(MZ15, _const(10000))], start=start))
    assert list(tr.low) == [TWIN - 0.1] * N
    r = _veto(_const(10000), np.zeros(N), _ones(), start=start)
    assert (r.E, r.obs, r.twin_verdict) == (120.0, 0, "refuted")


def test_a_spectrum_without_a_parseable_peak_does_not_move_the_scan_start():
    """The median scan start is taken over the spectra that have a parseable peak."""
    start = np.where(np.arange(N) < 60, TWIN - 0.1, np.nan)
    ts = _ts([(MZ15, np.where(np.arange(N) < N - 1, 10000.0, np.nan))], start=start)
    ts = pd.concat([ts, pd.DataFrame([dict(sample_item_id=f"s{N - 1:03d}", datetime_utc=T0 + pd.Timedelta(days=2),
                                           mz="n/a", height=1.0)])], ignore_index=True)
    tr = LT._Traces(ts)
    assert len(tr.order) == N and list(tr.low) == [TWIN - 0.1] * N
    r = LT._twin_veto(tr, [Y], F0, _ones()).iloc[0]
    assert (r.E, r.obs, r.twin_verdict) == (119.0, 0, "refuted")


# --------------------------------------------------------------------------- the note
def test_the_note_carries_the_numbers_and_only_on_a_veto():
    """'no 14N twin: seen in <obs> of <E, rounded> expected detections' on a vetoed row, '' elsewhere."""
    r = _veto(_heights(10), np.arange(N) < 1, _ones())
    assert r.veto and r.note == "no 14N twin: seen in 1 of 10 expected detections"
    r = _veto(_heights(7), np.zeros(N), np.full(N_BINS, 0.75))      # E = 5.25
    assert r.veto and r.note == "no 14N twin: seen in 0 of 5 expected detections"
    for twin, h, verdict in ((np.ones(N), _const(10000), "passes"), (np.arange(N) < 40, _const(10000), "unclear"),
                             (np.zeros(N), _heights(2), "untestable")):
        r = _veto(h, twin, _ones())
        assert (r.twin_verdict, bool(r.veto), r.note) == (verdict, False, ""), verdict


# --------------------------------------------------------------------------- the table and the funnel
def test_the_table_keeps_its_documented_columns():
    """label_twins.csv (docs/OUTPUTS.md): column order, bool flags, str notes, committed readings."""
    assert LT.TABLE_COLUMNS == (
        "neutral_formula", "adduct", "committed", "mz", "mz_partner", "n14", "n_codetected", "partner_share",
        "h15_median", "reference", "k_ratio", "k_sd", "k_r", "k_n", "cluster_k", "line_verdict",
        "alias_only_tie", "untie", "alien", "f", "E", "obs", "twin_verdict", "veto", "note")
    t = _measure(_refs_spec(**{Y: {"phase": 1.0, "q": None}}))
    assert list(t.columns) == list(LT.TABLE_COLUMNS)
    for c in ("committed", "reference", "cluster_k", "alias_only_tie", "untie", "alien", "veto"):
        assert t[c].dtype == bool, c
    assert t["note"].map(lambda v: isinstance(v, str)).all()
    assert set(t.loc[~t["veto"], "note"]) == {""}
    lab = t[t.adduct == NO3L]
    assert len(lab) == 7 and lab["committed"].all() and lab["f"].tolist() == pytest.approx([F0] * 7)
    assert list(lab.loc[lab["veto"], "neutral_formula"]) == [Y]


def _funnel_table() -> pd.DataFrame:
    """A label_twins table as measure writes it: two 14N lines (one excess, vetoed)
    and four readings (passes, unclear, refuted, untestable)."""
    rows = [dict(neutral_formula="A", adduct=NO3, committed=True, n_codetected=120, reference=True, cluster_k=True,
                 line_verdict="tracks", untie=True, alien=False, veto=False, note=""),
            dict(neutral_formula="B", adduct=NO3, committed=True, n_codetected=120, reference=False, cluster_k=False,
                 line_verdict="excess", untie=False, alien=True, veto=True, note="the 14N line runs 3.0x"),
            dict(neutral_formula="A", adduct=NO3L, f=0.25, E=120.0, obs=120, twin_verdict="passes", veto=False,
                 note=""),
            dict(neutral_formula="B", adduct=NO3L, f=0.25, E=120.0, obs=40, twin_verdict="unclear", veto=False,
                 note=""),
            dict(neutral_formula="Y", adduct=NO3L, f=0.25, E=120.0, obs=0, twin_verdict="refuted", veto=True,
                 note="no 14N twin: seen in 0 of 120 expected detections"),
            dict(neutral_formula="Z", adduct=NO3L, f=0.25, E=0.0, obs=0, twin_verdict="untestable", veto=False,
                 note="")]
    t = pd.DataFrame(rows)
    for c in LT.TABLE_COLUMNS:
        if c not in t.columns:
            t[c] = np.nan
    for c in ("committed", "reference", "cluster_k", "alias_only_tie", "untie", "alien", "veto"):
        t[c] = t[c].fillna(False).astype(bool)
    return t[list(LT.TABLE_COLUMNS)].sort_values(["neutral_formula", "adduct"]).reset_index(drop=True)


def test_the_funnel_counts_the_readings_apart_from_the_lines(tmp_path):
    """summary() counts the [M+^NO3]- readings (testable, passes, refuted) apart from the 14N lines."""
    t = _funnel_table()
    t.to_csv(tmp_path / "label_twins.csv", index=False)
    for table in (t, pd.read_csv(tmp_path / "label_twins.csv")):
        s = LT.summary(table, LABEL)
        got = {k: s[k] for k in ("f", "lines", "committed_lines", "readings", "testable", "passes", "refuted",
                                 "lines_refuted", "untie", "alien", "lines_excess", "lines_tracks")}
        assert got == {"f": 0.25, "lines": 2, "committed_lines": 2, "readings": 4, "testable": 3, "passes": 1,
                       "refuted": 1, "lines_refuted": 1, "untie": 1, "alien": 1, "lines_excess": 1,
                       "lines_tracks": 1}


# --------------------------------------------------------------------------- veto() on a table read back
def test_veto_reads_a_csv_table_strictly():
    """veto() reads True/'true'/'1'/'yes' only, a NaN note as '', and keys each row on its own adduct."""
    t = pd.DataFrame({"neutral_formula": ["A", "B", "C", "D", "E", "F", "G", "H", "I"],
                      "adduct": [NO3L, NO3L, NO3L, NO3, NO3L, NO3L, NO3L, NO3L, NO3L],
                      "veto": ["False", True, np.nan, True, "0", "yes", " TRUE ", "no", 0],
                      "note": ["x", np.nan, "y", "z", "w", "v", "u", "t", "s"]})
    assert LT.veto(t) == {("B", NO3L): "", ("D", NO3): "z", ("F", NO3L): "v", ("G", NO3L): "u"}
    assert LT.veto(pd.DataFrame({"neutral_formula": ["A"], "adduct": [NO3L]})) == {}
    assert LT.veto(pd.DataFrame({"neutral_formula": ["A"], "adduct": [NO3L], "veto": [True]})) == {("A", NO3L): ""}


def test_veto_reads_the_written_table_back_unchanged(tmp_path):
    """A written table read back from CSV vetoes the same keys and notes, [M+NO3]- lines included."""
    x, y = "C6H10O4", Y
    spec = _refs_spec(**{x: {"phase": 1.0, "q": 3.0}, y: {"phase": 1.0, "q": None}})
    t = _measure(spec, _frames(list(spec), no3=[x]))
    want = LT.veto(t)
    assert set(want) == {(x, NO3), (y, NO3L)}
    r = _row(t, y, NO3L)
    assert want[(y, NO3L)] == r.note == f"no 14N twin: seen in 0 of {r.E:.0f} expected detections"
    assert want[(x, NO3)] == _row(t, x).note and want[(x, NO3)].startswith("the 14N line runs 3.0x")
    t.to_csv(tmp_path / "label_twins.csv", index=False)
    back = pd.read_csv(tmp_path / "label_twins.csv")
    assert LT.veto(back) == want
    assert LT.facts(back)["veto"] == want
