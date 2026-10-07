"""Rule K core pins (peaky/batch/label_twins.py): the scope, the traces and the
scan start, the committed pairs, the cluster-k test at its edges, the line
verdicts, the untie assembly, the table contract, facts() / untie() / _truth()
on CSV cells, and the summary funnel (docs/EVIDENCE_LEVELS.md §3 `label_untie` /
`label_alien` / `label_veto`, §4.2; docs/OUTPUTS.md `label_twins.csv`).

Each test pins one rule with small synthetic inputs and exact expected values;
the helpers are tests/test_label_twins.py's.

Run: pytest tests/test_label_twins_core.py -q
"""
from __future__ import annotations

from types import SimpleNamespace as NS

import numpy as np
import pandas as pd
import pytest

from peaky.assignment import evidence as EV
from peaky.batch import label_twins as LT
from peaky.chem import chemistry as C
from tests import test_label_twins as TL
from tests.test_evidence import ledger, m0
from tests.test_label_twins import LABEL, NO3, NO3L, REFS, _alias_ledger, _frames, _measure, _refs_spec, _row, _series

T0 = pd.Timestamp("2026-08-11", tz="UTC")
QUIET = lambda *a: None   # noqa: E731


def _one(mzs, hs, sids=None):
    """A time series of explicit peaks (one spectrum unless `sids` says otherwise)."""
    sids = sids or ["a"] * len(mzs)
    return pd.DataFrame(dict(sample_item_id=sids, datetime_utc=[T0] * len(mzs), mz=mzs, height=hs))


def _lines(t):
    return t[t.adduct == NO3]


# =========================================================================== scope
def test_in_scope_needs_both_the_15n_label_and_the_labelled_adduct():
    """in_scope: label `^N` AND `[M+^NO3]-` among the adducts; either alone is out."""
    assert LT.in_scope(NS(label_isotope="^N", adducts=[NO3, NO3L]))
    assert LT.in_scope(NS(label_isotope="^N", adducts=[NO3L]))
    assert not LT.in_scope(NS(label_isotope=None, adducts=[NO3, NO3L]))
    assert not LT.in_scope(NS(label_isotope="13C", adducts=[NO3L]))
    assert not LT.in_scope(NS(label_isotope="^N", adducts=[NO3]))
    assert not LT.in_scope(NS(label_isotope="^N", adducts=None))
    assert not LT.in_scope(NS())


# =========================================================================== traces
def test_the_trace_is_the_tallest_peak_within_2_ppm():
    """_Traces.heights: the TALLEST peak of the window, found at +-1.9 ppm, not at +-2.1 ppm."""
    m = 300.0
    H = LT._Traces(_one([m, m * (1 + 1e-6), m * (1 - 1e-6)], [10.0, 500.0, 30.0])).heights([m])
    assert H.shape == (1, 1) and H[0, 0] == 500.0
    for off, found in ((1.9e-6, True), (-1.9e-6, True), (2.1e-6, False), (-2.1e-6, False)):
        got = LT._Traces(_one([m * (1 + off)], [7.0])).heights([m])[0, 0]
        assert np.isfinite(got) == found, off


def test_the_trace_window_is_closed_below_and_open_above():
    """_Traces.heights: the window is [m(1-tol), m(1+tol)) (the searchsorted convention)."""
    m, tol = 300.0, LT.TRACE_TOL_PPM * 1e-6
    tgt = np.asarray([m], float)
    lo, hi = float((tgt * (1 - tol))[0]), float((tgt * (1 + tol))[0])
    assert LT._Traces(_one([lo], [3.0])).heights([m])[0, 0] == 3.0
    assert np.isnan(LT._Traces(_one([hi], [3.0])).heights([m])[0, 0])


def test_the_scan_start_is_each_spectrums_lowest_peak_capped_at_the_median_of_the_measured_ones():
    """_Traces.low: per spectrum the lowest finite m/z, capped at the median over the
    spectra that have one; a spectrum below the median keeps its own start, and a
    spectrum with no finite m/z takes the median."""
    ts = pd.DataFrame(dict(
        sample_item_id=["a", "a", "b", "b", "c", "d", "d"],
        datetime_utc=[T0, T0, T0 + pd.Timedelta(minutes=1), T0 + pd.Timedelta(minutes=1),
                      T0 + pd.Timedelta(minutes=2), T0 + pd.Timedelta(minutes=3), T0 + pd.Timedelta(minutes=3)],
        mz=[10.0, 99.0, 12.0, 200.0, np.nan, 30.0, 31.0],
        height=[1.0] * 7))
    tr = LT._Traces(ts)
    assert tr.order == ["a", "b", "c", "d"]
    # measured lowest peaks 10, 12, 30 (c has none): median 12
    assert list(tr.low) == [10.0, 12.0, 12.0, 12.0]


def test_a_spectrum_that_starts_low_does_not_lower_the_others():
    """_Traces.low: one deep spectrum keeps its own start; every other spectrum keeps its own."""
    y = "C4H4O2"
    spec = _refs_spec(**{y: {"phase": 1.0, "h15": 20000.0, "q": None}})
    start = C.ion_mz(y, NO3L) - 2.0
    ts = TL._scan_from(_series(spec), start)
    deep = pd.concat([ts, pd.DataFrame([dict(sample_item_id="s001", datetime_utc=ts["datetime_utc"].iloc[0],
                                             mz=40.0, height=50.0, area=50.0, role="")])], ignore_index=True)
    tr = LT._Traces(deep)
    got = dict(zip(tr.order, tr.low))
    assert got["s001"] == pytest.approx(40.0)
    assert got["s002"] == pytest.approx(start + 0.001) and got["s119"] == pytest.approx(start + 0.001)


# =========================================================================== committed pairs
def test_committed_pools_each_regular_m0_pair_once():
    """_committed: one row per (neutral, adduct) over the files; M0 only, ion-only rows
    out, a row needs both a neutral and an adduct."""
    x = "C10H16O4"
    fr = _frames([x], no3=[x])["f1"]
    assert sorted(map(tuple, LT._committed({"f1": fr})[["neutral_formula", "adduct"]].values)) == [(x, NO3), (x, NO3L)]
    assert len(LT._committed({"f1": fr, "f2": fr.copy()})) == 2
    io = EV.trim(ledger([m0("io", "C11H18O4", adduct="[M]-.", ion="C11H18O4",
                            mz=C.ion_mz("C11H18O4", "[M-H]-") + 1.0078, method="ion_only:ea")]))
    assert len(LT._committed({"f1": fr, "f2": io})) == 2
    kid = m0("k", "C12H20O4", adduct=NO3L, ion="C12H20O4^NO3", mz=C.ion_mz("C12H20O4", NO3L))
    kid["role"] = "iso_child"
    assert len(LT._committed({"f1": fr, "f2": ledger([kid])})) == 2
    nulls = ledger([m0("n1", None, adduct=NO3L, mz=62.985), m0("n2", "C5H8O2", adduct=None)])
    assert len(LT._committed({"f1": fr, "f2": nulls})) == 2


def test_a_blank_or_missing_neutral_never_makes_a_row():
    """measure: '' and None neutrals on either nitrate adduct add no row (no reagent-ion row)."""
    spec = _refs_spec()
    frames = _frames(list(spec))
    frames["f2"] = ledger([m0("e", "", adduct=NO3L, mz=62.985), m0("n", None, adduct=NO3L, mz=62.985),
                           m0("e4", "", adduct=NO3, mz=61.988), m0("n4", None, adduct=NO3, mz=61.988)])
    t = _measure(spec, frames)
    assert sorted(set(t["neutral_formula"])) == sorted(spec)
    assert len(t) == 2 * len(spec)


def test_an_empty_series_is_an_empty_table():
    """measure: a zero-row series (with its columns) is 'no time series': the empty table
    with its header, nothing logged."""
    logs = []
    t = LT.measure(_series(_refs_spec()).iloc[0:0], _frames(REFS), LABEL, log=logs.append)
    assert t.empty and list(t.columns) == list(LT.TABLE_COLUMNS) and logs == []
    assert LT.facts(t) is None


def test_a_labelled_batch_without_a_labelled_commit_still_judges_its_14n_lines():
    """§3 label_alien: on a labelled-nitrate batch with a series, a committed [X+NO3]- line whose
    15N partner is in 0 of its 120 spectra is `absent` (alien, refuted) whether or not some other
    neutral is committed on [M+^NO3]-."""
    j = "C11H18O6"
    spec = _refs_spec(**{j: {"phase": 0.9, "share15": 0.0}})
    t = LT.measure(_series(spec), {"f1": EV.trim(TL._j1_rows(j))}, LABEL, log=QUIET)
    r = _row(t, j)
    assert (r.n14, r.n_codetected) == (120, 0) and r.line_verdict == "absent" and r.alien and r.veto
    assert LT.facts(t)["alien"] == {(j, NO3)}


# =========================================================================== cluster-k: references
def test_the_references_are_drawn_from_the_labelled_commits_only():
    """ref_pool: a bright, co-detected CHO pair committed on [M+NO3]- alone (its 15N line in
    the series but not committed) is tested and tracks, but never calibrates k_cl."""
    w = "C11H18O4"
    spec = _refs_spec(**{w: {"phase": 0.9}})
    t = _measure(spec, _frames(list(spec), no3=[w], only14=[w]))
    r = _row(t, w)
    assert r.n_codetected == 120 and r.h15_median >= LT.REF_MIN_H15 and r.k_n == 120
    assert not r.reference and r.cluster_k and r.line_verdict == "tracks" and r.committed
    assert [n for n in _lines(t)[_lines(t).reference].neutral_formula] == sorted(REFS)
    assert LT.summary(t, LABEL)["references"] == len(REFS)


def test_the_cluster_k_bands_and_counts_at_their_edges():
    """cluster-k: ratio band 0.5-2 (0.55 / 1.9 in, 0.45 / 2.1 out), sd <= 0.25 alone can
    fail, a reference needs >= 100 co-detections (exactly 100 counts, 99 does not) and a
    15N median >= 1000 read on the 15N line."""
    spec = _refs_spec(**{
        "C6H10O4": {"phase": 1.0, "q": 1.9}, "C6H8O4": {"phase": 1.1, "q": 2.1},
        "C6H10O5": {"phase": 1.2, "q": 0.55}, "C6H8O5": {"phase": 1.3, "q": 0.45},
        "C7H12O4": {"phase": 1.4, "noise": 0.3},
        "C8H14O4": {"phase": 1.3, "h15": 1000.0},
        "C9H16O4": {"phase": 1.8, "share": 99.5 / 120},
        "C9H16O5": {"phase": 1.9, "share": 98.5 / 120},
        "C10H18O5": {"phase": 0.3, "share": 0.6},
    })
    t = _measure(spec)
    assert _row(t, "C6H10O4").cluster_k and not _row(t, "C6H8O4").cluster_k
    assert _row(t, "C6H10O5").cluster_k and not _row(t, "C6H8O5").cluster_k
    r = _row(t, "C7H12O4")
    assert r.k_sd > LT.SD_MAX and r.k_r >= LT.R_MIN and LT.RATIO_LO <= r.k_ratio <= LT.RATIO_HI
    assert not r.cluster_k and r.line_verdict == "consistent"
    r = _row(t, "C8H14O4")
    ts = _series(spec)
    h15 = ts[(ts.neutral_formula == "C8H14O4") & (ts.adduct == NO3L)]["height"]
    assert r.reference and r.h15_median == pytest.approx(float(np.median(h15)))
    assert 1000.0 < r.h15_median < 2000.0
    r = _row(t, "C9H16O4")
    assert r.n_codetected == 100 and r.k_n == 100 and r.cluster_k and r.reference
    r = _row(t, "C9H16O5")
    assert r.n_codetected == 99 and pd.isna(r.k_ratio) and r.k_n == 0 and not r.reference
    r = _row(t, "C10H18O5")                                  # the 14N line in 72 spectra, its partner in each
    assert (r.n14, r.n_codetected, r.partner_share) == (72, 72, 1.0)
    assert not r.reference and r.line_verdict == "untestable"


def test_the_r_clause_fails_alone_on_a_quiet_reagent(monkeypatch):
    """cluster-k: r(log q, log k_cl) >= 0.8 is its own clause: on a reagent whose k_cl
    barely moves, a noisy or a constant-ratio line keeps sd <= 0.25 and the ratio in
    band and fails on r alone (0.3 < r < 0.8 for the noisy one)."""
    monkeypatch.setattr(TL, "k_of", lambda i: 0.15 * (1.4 + 0.4 * np.sin(2 * np.pi * i / 40.0)))
    spec = _refs_spec(**{"C7H12O5": {"phase": 1.6, "noise": 0.2}, "C7H12O4": {"phase": 1.4, "flat": True}})
    t = _measure(spec)
    for n in ("C7H12O5", "C7H12O4"):
        r = _row(t, n)
        assert r.k_sd <= LT.SD_MAX and LT.RATIO_LO <= r.k_ratio <= LT.RATIO_HI, n
        assert r.k_r < LT.R_MIN and not r.cluster_k and r.line_verdict == "consistent", n
    assert 0.3 < _row(t, "C7H12O5").k_r < 0.8
    for n in REFS:
        assert _row(t, n).cluster_k, n


def test_every_cluster_k_edge_is_inclusive(monkeypatch):
    """cluster-k: a pair exactly at RATIO_LO, RATIO_HI, SD_MAX, R_MIN or REF_MIN_H15 is
    in (each threshold set to the measured value); a line exactly at RATIO_HI that fails
    sd is `consistent`, not `excess` (excess is strictly above)."""
    spec = _refs_spec(**{"C6H10O5": {"phase": 1.2, "q": 0.55}, "C6H10O4": {"phase": 1.0, "q": 1.9},
                         "C7H12O4": {"phase": 1.4, "noise": 0.2}, "C7H10O5": {"phase": 1.5, "q": 1.9, "noise": 0.6}})
    t = _measure(spec)
    lo, hi = _row(t, "C6H10O5").k_ratio, _row(t, "C6H10O4").k_ratio
    sd, rr = _row(t, "C7H12O4").k_sd, _row(t, "C7H12O4").k_r
    h15 = float(_lines(t)[_lines(t).reference].h15_median.min())
    noisy_hi = _row(t, "C7H10O5").k_ratio
    assert 0.5 < lo < 0.6 and 1.8 < hi < 2.0 and sd < 0.25 and rr > 0.8 and 1.0 < noisy_hi < 2.5
    refs = set(_lines(t)[_lines(t).reference].neutral_formula)
    monkeypatch.setattr(LT, "RATIO_LO", lo)
    monkeypatch.setattr(LT, "RATIO_HI", hi)
    monkeypatch.setattr(LT, "SD_MAX", sd)
    monkeypatch.setattr(LT, "R_MIN", rr)
    monkeypatch.setattr(LT, "REF_MIN_H15", h15)
    t = _measure(spec)
    assert set(_lines(t)[_lines(t).reference].neutral_formula) == refs
    for n in ("C6H10O5", "C6H10O4", "C7H12O4"):
        assert _row(t, n).cluster_k and _row(t, n).line_verdict == "tracks", n
    monkeypatch.setattr(LT, "RATIO_HI", noisy_hi)
    monkeypatch.setattr(LT, "SD_MAX", 0.25)
    monkeypatch.setattr(LT, "R_MIN", 0.8)
    r = _row(_measure(spec), "C7H10O5")
    assert r.k_ratio == noisy_hi and not r.cluster_k and r.line_verdict == "consistent"


def test_the_stats_need_50_spectra_with_both_lines_above_zero():
    """cluster-k: a zero-height 14N reading is co-detected but carries no ratio; 50 usable
    spectra are tested, 49 are not and the line reads `untestable` (not `consistent`)."""
    x = "C9H16O4"
    spec = _refs_spec(**{x: {"phase": 1.8}})
    for zeros, n_ok, verdict in ((70, 50, "tracks"), (71, 49, "untestable")):
        ts = _series(spec)
        sel = ts.index[(ts.neutral_formula == x) & (ts.adduct == NO3)][-zeros:]
        ts.loc[sel, "height"] = 0.0
        r = _row(LT.measure(ts, _frames(list(spec)), LABEL, log=QUIET), x)
        assert r.n_codetected == 120 and r.k_n == n_ok and r.line_verdict == verdict, zeros
        assert r.cluster_k == (n_ok >= 50) and pd.isna(r.k_ratio) == (n_ok < 50)


def test_a_batch_without_references_tests_nothing():
    """cluster-k: no bright CHO reference (all 15N medians < 1000) -> no line has stats and
    every line reads `untestable`, however well co-detected."""
    spec = {n: {"phase": 0.6 * k, "h15": 300.0} for k, n in enumerate(REFS)}
    t = _measure(spec, _frames(list(spec), no3=list(spec)))
    L = _lines(t)
    assert (L.n_codetected == 120).all() and not L.reference.any() and (L.k_n == 0).all()
    assert L.k_ratio.isna().all() and (L.line_verdict == "untestable").all()
    assert L.alien.all() and not L.veto.any()


def test_the_tested_pair_is_left_out_of_its_own_k():
    """cluster-k: two references, one at 5x: its k_cl is the other's ratio alone (ratio ~5,
    excess); a lone reference has no k_cl at all (k_n 0, untestable)."""
    t = _measure({REFS[0]: {"phase": 0.0, "q": 5.0}, REFS[1]: {"phase": 0.6}})
    r = _row(t, REFS[0])
    assert not r.cluster_k and 4.5 < r.k_ratio < 5.5 and r.line_verdict == "excess"
    r = _row(_measure({REFS[1]: {"phase": 0.6}}), REFS[1])
    assert r.reference and pd.isna(r.k_ratio) and r.k_n == 0 and not r.cluster_k and r.line_verdict == "untestable"


def test_k_cl_is_the_median_over_the_references():
    """cluster-k: one reference at 5x does not drag the others' k_cl (a mean would put
    every other reference near 0.56)."""
    spec = _refs_spec()
    spec[REFS[0]] = {"phase": 0.0, "q": 5.0}
    t = _measure(spec)
    for n in REFS[1:]:
        assert 0.95 < _row(t, n).k_ratio < 1.05, n


def test_two_spikes_neither_move_the_ratio_nor_break_r():
    """cluster-k: the ratio is a median (two 5x spikes leave it within 1 %) and r is on the
    logs (the linear r drops under 0.8 on the spikes)."""
    x = "C7H12O5"
    spec = _refs_spec(**{x: {"phase": 1.6, "noise": 0.0}})
    ts = _series(spec)
    rows = ts.index[(ts.neutral_formula == x) & (ts.adduct == NO3)]
    ts.loc[rows[[10, 50]], "height"] *= 5.0                   # k_of peaks at i = 10, 50, 90
    r = _row(LT.measure(ts, _frames(list(spec)), LABEL, log=QUIET), x)
    assert abs(r.k_ratio - 1.0) < 0.01 and r.k_sd <= LT.SD_MAX and r.k_r >= LT.R_MIN and r.cluster_k


def test_k_sd_is_the_sample_sd_of_the_log_ratio():
    """cluster-k: noise-free references and a line at k_cl x exp(+-a), alternating: k_sd is
    a * sqrt(n / (n - 1)) and the ratio exactly 1."""
    x, a = "C7H12O5", 0.1
    spec = {n: {"phase": 0.6 * k, "noise": 0.0} for k, n in enumerate(REFS)}
    spec[x] = {"phase": 1.6, "noise": 0.0}
    ts = _series(spec)
    rows = ts.index[(ts.neutral_formula == x) & (ts.adduct == NO3)]
    ts.loc[rows, "height"] *= np.exp(a * (-1.0) ** np.arange(len(rows)))
    r = _row(LT.measure(ts, _frames(list(spec)), LABEL, log=QUIET), x)
    n = TL.N_SPECTRA
    assert r.k_n == n and r.k_sd == pytest.approx(a * np.sqrt(n / (n - 1)), rel=1e-6)
    assert r.k_ratio == pytest.approx(1.0, abs=1e-9)


# =========================================================================== line verdicts
def test_the_absent_edge_is_a_share_of_0_2_inclusive():
    """absent: the 15N partner in <= 20 % of the 14N line's spectra (24 of 120 is absent,
    25 of 120 is untestable), with its n14 / n_codetected / partner_share and exact note."""
    for n15, verdict in ((24, "absent"), (25, "untestable")):
        n = "C11H18O5"
        spec = _refs_spec(**{n: {"phase": 0.7, "share15": (n15 - 0.5) / 120}})
        t = _measure(spec, _frames(list(spec), no3=[n], only14=[n]))
        r = _row(t, n)
        assert (r.n14, r.n_codetected) == (120, n15) and r.partner_share == n15 / 120
        assert r.line_verdict == verdict and r.alien and r.veto == (verdict == "absent"), n15
        assert r.note == ("no 15N partner: seen in 24 of the 120 spectra of the 14N line; "
                          "cluster or organonitrate undecided" if verdict == "absent" else "")


def test_absent_needs_ten_spectra_of_the_14n_line():
    """absent: the 14N line in >= 10 spectra (10 without a partner is absent, 9 untestable)."""
    n = "C11H18O6"
    for k, verdict in ((10, "absent"), (9, "untestable")):
        spec = _refs_spec(**{n: {"phase": 0.9, "share15": 0.0, "share": (k - 0.5) / 120}})
        r = _row(_measure(spec, _frames(list(spec), no3=[n], only14=[n])), n)
        assert (r.n14, r.n_codetected, r.line_verdict) == (k, 0, verdict), k
        assert r.alien and r.veto == (verdict == "absent"), k


def test_only_a_committed_line_is_alien_vetoed_or_untied():
    """committed gates alien, the [M+NO3]- veto and the untie: tracking (alias-only tied in the
    alias table), excess and absent lines of neutrals committed on [M+^NO3]- alone carry none."""
    x, e, a = "C10H18O3", "C6H10O4", "C11H18O5"
    spec = _refs_spec(**{x: {"phase": 2.0}, e: {"phase": 1.0, "q": 3.0}, a: {"phase": 0.7, "share15": 0.1}})
    t = _measure(spec, _frames(list(spec)), {"f1": LT.alias_only_ties(_alias_ledger(x))})
    r = _row(t, x)
    assert r.cluster_k and r.alias_only_tie and not r.committed and not r.untie
    for n, verdict in ((e, "excess"), (a, "absent")):
        r = _row(t, n)
        assert r.line_verdict == verdict and not r.committed and not r.alien and not r.veto and r.note == "", n
    assert LT.facts(t) == {"untie": set(), "veto": {}, "alien": set()}
    s = LT.summary(t, LABEL)
    assert (s["committed_lines"], s["alien"], s["lines_refuted"], s["untie"]) == (0, 0, 0, 0)


def test_a_line_without_a_14n_reading_has_a_zero_partner_share():
    """n14 / partner_share: a labelled reading with no 14N line anywhere has n14 0 and a
    share of 0 (never NaN), and is untestable, not absent (fewer than 10 spectra)."""
    y = "C12H14O2"
    t = _measure(_refs_spec(**{y: {"phase": 1.0, "q": None}}))
    r = _row(t, y)
    assert (r.n14, r.n_codetected, r.partner_share) == (0, 0, 0.0)
    assert r.line_verdict == "untestable" and pd.isna(r.h15_median) and not r.committed


def test_the_notes_carry_the_numbers_and_only_on_a_vetoed_line():
    """_line_note: exact wording for excess and absent; every other 14N line has no note."""
    lines = {
        "C10H18O3": ({"phase": 2.0}, "tracks"),
        "C7H12O5": ({"phase": 1.4, "noise": 0.6}, "consistent"),
        "C6H10O4": ({"phase": 1.0, "q": 3.0}, "excess"),
        "C11H18O5": ({"phase": 0.7, "share15": 0.15}, "absent"),
        "C9H16O4": ({"phase": 1.8, "share15": 0.25}, "untestable"),
    }
    spec = _refs_spec(**{n: o for n, (o, _) in lines.items()})
    t = _measure(spec, _frames(list(spec), no3=list(lines), only14=["C11H18O5"]))
    for n, (_, v) in lines.items():
        assert _row(t, n).line_verdict == v, n
    k = _row(t, "C6H10O4").k_ratio
    assert 2.95 <= k < 3.05
    assert _row(t, "C6H10O4").note == ("the 14N line runs 3.0x its cluster share of the 15N line; "
                                       "cluster or organonitrate undecided")
    assert _row(t, "C11H18O5").note == ("no 15N partner: seen in 18 of the 120 spectra of the 14N line; "
                                        "cluster or organonitrate undecided")
    for n in ("C10H18O3", "C7H12O5", "C9H16O4", *REFS):
        assert _row(t, n).note == "", n
    assert LT.veto(t) == {("C6H10O4", NO3): _row(t, "C6H10O4").note, ("C11H18O5", NO3): _row(t, "C11H18O5").note}


def _mostly_absent(monkeypatch, **line):
    """A 600-spectrum batch: w's 14N line in every spectrum, its 15N partner in the first 108
    (18 %, co-detected >= 100 so the cluster-k stats exist); w committed and alias-only tied."""
    monkeypatch.setattr(TL, "N_SPECTRA", 600)
    w = "C11H18O4"
    spec = _refs_spec(**{w: {"phase": 0.9, "share15": 107.5 / 600, **line}})
    alias = {"f1": LT.alias_only_ties(_alias_ledger(w))}
    r = _row(_measure(spec, _frames(list(spec), no3=[w], tied=[w]), alias), w)
    assert (r.n14, r.n_codetected, r.partner_share) == (600, 108, 0.18) and r.alias_only_tie
    return r


@pytest.mark.parametrize("line", [{}, {"noise": 0.6}], ids=["tracking", "noisy"])
def test_a_line_whose_partner_is_mostly_absent_is_absent_even_when_tested(monkeypatch, line):
    """absent (EVIDENCE_LEVELS §3 label_alien): the 15N partner in <= 20 % of the >= 10 spectra
    of the 14N line -- 108 of 600 -- is absent (alien, refuted) whatever the ratio in the 108."""
    r = _mostly_absent(monkeypatch, **line)
    assert r.line_verdict == "absent" and r.alien and r.veto


def test_an_excess_line_gives_its_neutral_nothing_only_when_its_partner_is_mostly_missing(monkeypatch):
    """excess (2026-09-27 decision): the line is refuted (5b) either way; it leaves its
    neutral's pools only when its 15N partner is in < PARTNER_PRESENT_SHARE (0.8) of
    its spectra -- with the partner present the cluster is shown by the partner."""
    monkeypatch.setattr(TL, "N_SPECTRA", 600)
    w = "C11H18O4"
    for share15, alien in ((0.7, True), (0.8, False)):
        spec = _refs_spec(**{w: {"phase": 0.9, "q": 3.0, "share15": (share15 * 600 - 0.5) / 600 + 1e-9}})
        r = _row(_measure(spec, _frames(list(spec), no3=[w])), w)
        assert r.line_verdict == "excess" and r.veto, share15
        assert r.partner_share == pytest.approx(share15, abs=2e-3)
        assert bool(r.alien) == alien, share15


def test_a_line_whose_partner_is_mostly_absent_never_unties(monkeypatch):
    """untie = tracks & alias-only tie & committed: a line that is not X's cluster in 82 % of
    its spectra must not clear the organonitrate tie."""
    r = _mostly_absent(monkeypatch)
    assert not r.untie


# =========================================================================== the untie assembly
def test_an_alias_only_tie_on_a_line_off_k_does_not_untie():
    """untie = cluster_k & alias-only tie & committed: a committed, alias-only tied line at
    3x k_cl stays tied (excess, vetoed; its 15N partner is present, so not alien)."""
    x = "C10H18O3"
    spec = _refs_spec(**{x: {"phase": 2.0, "q": 3.0}})
    t = _measure(spec, _frames(list(spec), no3=[x], tied=[x]), {"f1": LT.alias_only_ties(_alias_ledger(x))})
    r = _row(t, x)
    assert r.alias_only_tie and r.committed and not r.cluster_k and not r.untie
    assert r.line_verdict == "excess" and not r.alien and r.veto and r.partner_share >= LT.PARTNER_PRESENT_SHARE
    assert LT.untie(t) == set() and LT.facts(t)["untie"] == set()


def test_a_columnless_alias_frame_is_ignored():
    """measure: an alias table without columns contributes nothing (no error, no untie)."""
    x = "C10H18O3"
    spec = _refs_spec(**{x: {"phase": 2.0}})
    t = _measure(spec, _frames(list(spec), no3=[x], tied=[x]), {"f1": pd.DataFrame(), "f2": None})
    assert _row(t, x).cluster_k and not _row(t, x).alias_only_tie and not _row(t, x).untie


def test_alias_only_ties_reads_only_tied_m0_nitrate_rows_with_an_alias():
    """alias_only_ties: needs an alias (no alternatives -> False), a `tied` column read as a
    truth value ('False' / NaN / '0' are not tied), role M0 and adduct [M+NO3]-."""
    x = "C10H18O4"
    led = _alias_ledger(x)
    led["alternatives"] = ["[]"]
    assert list(LT.alias_only_ties(led).itertuples(index=False, name=None)) == [(x, NO3, False)]
    assert LT.alias_only_ties(_alias_ledger(x).drop(columns=["tied"])).empty
    other = _alias_ledger(x)
    other["adduct"] = "[M-H]-"
    assert LT.alias_only_ties(other).empty
    kid = _alias_ledger(x)
    kid["role"] = "iso_child"
    assert LT.alias_only_ties(kid).empty
    for v in ("False", np.nan, "0"):
        s = _alias_ledger(x).astype({"tied": object})
        s["tied"] = [v]
        assert LT.alias_only_ties(s).empty, v
    s = _alias_ledger(x).astype({"tied": object})
    s["tied"] = ["True"]
    assert list(LT.alias_only_ties(s).alias_only) == [True]


# =========================================================================== the table contract
DOCUMENTED = ["neutral_formula", "adduct", "committed", "mz", "mz_partner", "n14", "n_codetected", "partner_share",
              "h15_median", "reference", "k_ratio", "k_sd", "k_r", "k_n", "cluster_k", "line_verdict",
              "alias_only_tie", "untie", "alien", "f", "E", "obs", "twin_verdict", "veto", "note"]


def test_the_table_columns_are_the_documented_order():
    """docs/OUTPUTS.md label_twins.csv: the column order, compared with a literal."""
    assert list(LT.TABLE_COLUMNS) == DOCUMENTED
    assert list(_measure(_refs_spec()).columns) == DOCUMENTED
    assert list(LT._empty().columns) == DOCUMENTED


def test_the_table_is_typed_and_sorted():
    """measure: the flags are bool on every row (NO3L rows read False where they carry no
    14N fact), the notes are strings, the rows sorted by (neutral, adduct), and every
    labelled reading is committed."""
    x, y = "C10H18O3", "C12H14O2"
    spec = _refs_spec(**{x: {"phase": 2.0}, y: {"phase": 1.0, "q": None}})
    t = _measure(spec, _frames(list(spec), no3=[x], tied=[x]), {"f1": LT.alias_only_ties(_alias_ledger(x))})
    for c in ("committed", "reference", "cluster_k", "alias_only_tie", "untie", "alien", "veto"):
        assert t[c].dtype == bool, c
    lab = t[t.adduct == NO3L]
    assert lab.committed.all() and not lab[["reference", "cluster_k", "alias_only_tie", "untie", "alien"]].any().any()
    assert t["note"].map(lambda v: isinstance(v, str)).all()
    keys = list(zip(t.neutral_formula, t.adduct))
    assert keys == sorted(keys) and len(keys) == 2 * len(spec)
    assert _row(t, x).untie and list(t.untie) == [(n, a) == (x, NO3) for n, a in keys]


# =========================================================================== facts / untie / veto readers
def test_truth_reads_csv_and_numpy_cells():
    """_truth: only real truth values and 'true' / '1' / 'yes' (any case, padded) are true."""
    yes = (True, np.True_, 1, "True", " true ", "1", "yes", "YES")
    no = (False, np.False_, 0, None, np.nan, float("nan"), "False", "0", "", "no", "nan")
    assert [LT._truth(v) for v in yes] == [True] * len(yes)
    assert [LT._truth(v) for v in no] == [False] * len(no)


def test_untie_reads_only_true_cells():
    """untie(): NaN and 'False' cells are no untie; a table without the column has none."""
    tab = pd.DataFrame({"neutral_formula": ["A", "B", "C"], "adduct": [NO3] * 3, "untie": [True, np.nan, "False"]})
    assert LT.untie(tab) == {("A", NO3)}
    assert LT.untie(pd.DataFrame({"neutral_formula": ["A"], "adduct": [NO3]})) == set()


def test_facts_reads_a_csv_table_like_the_engine(tmp_path):
    """facts(): the alien set reads truth values (a blank or 'False' cell is not alien) and keys
    each row by its own (neutral, adduct); a table without the alien column has an empty alien
    set; the keys are untie/veto/alien."""
    tab = pd.DataFrame({"neutral_formula": ["A", "B", "C", "D"], "adduct": [NO3, NO3, NO3, NO3L],
                        "untie": [False, "True", None, False], "alien": [True, None, "False", "true"],
                        "veto": [True, False, None, True], "note": ["n", "", None, None]})
    tab.to_csv(tmp_path / "t.csv", index=False)
    back = pd.read_csv(tmp_path / "t.csv")
    for table in (tab, back):
        f = LT.facts(table)
        assert f == {"untie": {("B", NO3)}, "veto": {("A", NO3): "n", ("D", NO3L): ""},
                     "alien": {("A", NO3), ("D", NO3L)}}
    assert LT.facts(back.drop(columns=["alien"])) == {"untie": {("B", NO3)}, "veto": {("A", NO3): "n", ("D", NO3L): ""},
                                                      "alien": set()}
    assert LT.facts(None) is None and LT.facts(LT._empty()) is None


# =========================================================================== the summary funnel
def test_the_summary_funnel_counts_each_stage():
    """summary(): every key on one batch of 14 neutrals. 14N lines: 6 refs + x (track,
    untie) + c100 (co-detected in exactly 100) + e (excess) + cons (consistent) tested;
    u (72 co-detected), a (absent, 14N only), y / z (no 14N line) not. Readings: 13
    labelled (a is 14N only), y refuted, z untestable."""
    x, y, z = "C10H18O3", "C12H14O2", "C11H12O2"
    e, u, c100, a, cons = "C6H10O4", "C9H16O4", "C9H16O5", "C11H18O6", "C7H12O5"
    spec = _refs_spec(**{
        x: {"phase": 2.0}, y: {"phase": 1.0, "q": None}, z: {"phase": 1.0, "q": None, "h15": 2500.0},
        e: {"phase": 1.0, "q": 3.0}, u: {"phase": 1.8, "share15": 0.6}, c100: {"phase": 1.9, "share15": 99.5 / 120},
        a: {"phase": 0.9, "share15": 0.0}, cons: {"phase": 1.4, "noise": 0.6}})
    frames = _frames(list(spec), no3=[x, e, u, a, cons], tied=[x], only14=[a])
    t = _measure(spec, frames, {"f1": LT.alias_only_ties(_alias_ledger(x))})
    assert _row(t, u).n_codetected == 72 and _row(t, c100).n_codetected == 100
    assert _row(t, z, NO3L).twin_verdict == "untestable" and _row(t, y, NO3L).twin_verdict == "refuted"
    assert LT.summary(t, LABEL) == {
        "in_scope": True, "f": pytest.approx(0.02 / 0.98), "lines": 14, "codetected": 10, "references": 10,
        "cluster_k": 8, "committed_lines": 5, "lines_tracks": 1, "lines_consistent": 1, "lines_excess": 1,
        "lines_absent": 1, "lines_untestable": 1, "alien": 2, "untie": 1, "readings": 13, "testable": 12,
        "passes": 11, "refuted": 1, "lines_refuted": 2}
    assert list(LT.summary(t, LABEL)) == [
        "in_scope", "f", "lines", "codetected", "references", "cluster_k", "committed_lines", "lines_tracks",
        "lines_consistent", "lines_excess", "lines_absent", "lines_untestable", "alien", "untie", "readings",
        "testable", "passes", "refuted", "lines_refuted"]


def test_the_summary_of_an_empty_in_scope_table_says_in_scope():
    """summary(): an empty table on a labelled channel is in scope with zero counts."""
    assert LT.summary(LT._empty(), LABEL) == {"in_scope": True, "lines": 0, "untie": 0, "readings": 0, "refuted": 0}
    assert LT.summary(None, P_NO3()) == {"in_scope": False, "lines": 0, "untie": 0, "readings": 0, "refuted": 0}


def P_NO3():
    from peaky.chem import profiles as P
    return P.PROFILES["NO3"]
