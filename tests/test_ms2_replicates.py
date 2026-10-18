"""Offline tests for `ms2 analyze` on the per-scan tables: replicate statistics
(peaky/ms2/replicates.py), the bootstrap interval of every E50, the vectorised
E50 (breakdown.e50_many), and the Orbitrap harmonic-ghost rule
(readings.harmonic_ghosts). No network.

Own data (tests/fixtures/ms2, ids scrubbed):

  * the 2026-09-11 monoterpene standard A -- its server spectra (S/N 10, the
    known answers of PR #80) read scan by scan from its `/metadata` and the
    time series of each spectrum (`scans.assign_from_times`,
    `scans.scan_peak_intensities`), as `ms2 extract --per-scan` builds them:
    [M+H]+ C10H17+ F E50 NCE 13.89 with its interval pinned; and the same
    standard rebuilt as a targeted-MS2 file (`ms2 extract --file`), whose
    spectra ARE the per-scan means;
  * ms2_nh4_exploratory_179_sn3.json -- the 2026-09-10 exploratory file's
    179.1069 window, the server's S/N-3 spectra (the committed S/N-10 ones
    do not hold the ghost): its precursor's 2f ghost at m/z 44.7766.

Synthetic: replicated windows of known per-scan intensities (one energy of
one scan, a crossing some resamples miss -- above the energies, below them,
across an unobserved energy, at an end energy some resamples do not observe
--, a crossing next to energies of one scan, S read against a lowest energy
without spread, a peak no scan reads, a split peak one of whose two peaks is
held, a spectrum with no per-scan row, an empty scan, volts unknown at one
energy), a ghost that lands on a formula, a ghost two window ions read, a 3f
ghost, a peak above the cap, two peaks in one band, a product that sits in a
harmonic band of its precursor, the S/N a reading needs to clear its place and
the 0.3 mDa floor of the place's tolerance."""
import json
import math
import os
import re
import subprocess
import sys
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from peaky import cli
from peaky.ms2 import analyze as AN
from peaky.ms2 import attribute as AT
from peaky.ms2 import breakdown as BD
from peaky.ms2 import readings as RD
from peaky.ms2 import replicates as RP
from peaky.ms2 import scans as SC
from peaky.ms2 import sources as S
from tests.test_ms2_breakdown import _sub, _synthetic, _write_tables
from tests.test_ms2_scan_source import FT, wrap_tms2

FIX = Path(__file__).resolve().parent / "fixtures" / "ms2"
NAN = float("nan")
MT_IONS = ("C10H17+", "C10H20^N+", "C10H19O+")
WIDTH_A = 0.4
#: the columns this PR adds (blank without the per-scan tables)
NEW_BREAKDOWN = ("n_harmonic", "harmonic_intensity", "n_scans", "intensity_cv", "f_sd",
                 "f_sem", "f_lo_sd", "f_lo_sem", "s_sd", "s_sem")
NEW_E50 = ("e50_nce_lo", "e50_nce_hi", "e50_v_lo", "e50_v_hi", "e50_ecm_lo", "e50_ecm_hi",
           "e50_lo_censor", "e50_hi_censor", "boot_n", "boot_cross", "n_one_scan",
           "one_scan_bracket")
NEW_FRAGMENTS = ("presence_at_max", "cv_at_max")


def _load(name):
    return json.loads((FIX / name).read_text(encoding="utf-8"))


def _mz(ion):
    return AN.ion_mz(ion)


def _ions(*formulas):
    return AN.collect_ions((f, NAN, _mz(f)) for f in formulas)


def _e50(res, parent, ion, curve="F"):
    e = res.e50
    r = e[np.isclose(e["parent_mz"], parent) & (e["ion"] == ion) & (e["curve"] == curve)]
    assert len(r) == 1, (parent, ion, curve)
    return r.iloc[0]


def _old(df, new):
    return df[[c for c in df.columns if c not in new]]


@pytest.fixture(scope="module")
def std_a():
    """Standard A as `ms2 extract --per-scan` writes it: the server's S/N-10
    spectra, their scans, every server peak in every scan."""
    meta, times = _load("ms2_monoterpene_a_metadata.json"), _load("ms2_monoterpene_a_times.json")
    cen, summ = _load("ms2_monoterpene_a_centroids.json"), _load("ms2_monoterpene_a_summary.json")
    spectra, parents = S.spectra_table("S", cen, summ), S.parents_table("S", cen, summ)
    window = S.window_table("S", _load("ms2_monoterpene_a_ms1.json"),
                            sorted(set(parents["parent_mz"])), WIDTH_A)
    st, _ = SC.assign_from_times(SC.scan_table(meta), times)
    peaks = SC.scan_peak_intensities(meta, st, spectra)
    man = {"sample_id": "S", "parameters": {"isolation_width_used": WIDTH_A}}
    plain = AN.RunTables(spectra=spectra, parents=parents, window=window, manifest=man)
    per = AN.RunTables(spectra=spectra, parents=parents, window=window, manifest=man,
                       scans=st, scan_peaks=peaks)
    return plain, per


@pytest.fixture(scope="module")
def std_a_res(std_a):
    plain, per = std_a
    return AN.analyze_run(plain, _ions(*MT_IONS)), AN.analyze_run(per, _ions(*MT_IONS))


# --------------------------------------------------------- e50_many ----
def test_e50_many_is_e50_column_by_column():
    """The vectorised E50 against `e50` on 600 random curves with holes,
    values at exactly 0.5, a repeated NCE, an NCE that is NaN and volts
    unknown at one energy: the same E50 on NCE and volts, the same flag."""
    rng = np.random.default_rng(3)
    nce = np.array([1.0, 5.0, 5.0, 10.0, np.nan, 20.0, 35.0, 3.0])
    volts = np.array([4.2, 5.0, 5.0, 6.0, 7.0, np.nan, 11.0, 4.6])
    y = rng.uniform(0.0, 1.0, size=(len(nce), 600))
    y = np.sort(y, axis=0)[::-1] + rng.normal(0, 0.15, size=y.shape)
    y[rng.random(y.shape) < 0.2] = np.nan
    y[rng.random(y.shape) < 0.05] = 0.5
    y[:, :20] = np.nan                                       # no data
    y[:, 20:40] = np.where(np.arange(len(nce))[:, None] == 3, 0.7, np.nan)    # one energy
    e, ev, code = BD.e50_many(nce, y, volts)
    for j in range(y.shape[1]):
        ref = BD.e50(nce, y[:, j], volts)
        assert BD.FLAGS[code[j]] == ref.flag, j
        np.testing.assert_equal(e[j], ref.nce)
        np.testing.assert_equal(ev[j], ref.volts)
    assert {BD.FLAGS[c] for c in code} == set(BD.FLAGS)      # every flag was met
    # the bracket: the two observed energies the crossing lies between
    e2, ev2, code2, (xa, xb, va, vb) = BD.e50_many(nce, y, volts, bracket=True)
    np.testing.assert_equal((e2, ev2, code2), (e, ev, code))
    fi = BD.FLAGS.index
    inter, gap = code == fi(BD.FLAG_INTERPOLATED), code == fi(BD.FLAG_GAP)
    assert np.isnan(xa[~(inter | gap)]).all() and np.isnan(xb[~(inter | gap)]).all()
    assert ((xa[inter] <= e[inter]) & (e[inter] <= xb[inter])).all()
    keep = np.isfinite(nce)
    x_s, y_s = nce[keep], y[keep]
    for j in np.flatnonzero(inter | gap):
        o = np.isfinite(y_s[:, j])
        assert not (o & (x_s > xa[j]) & (x_s < xb[j])).any(), j   # no observed energy between
        if inter[j]:
            assert not ((x_s > xa[j]) & (x_s < xb[j])).any(), j
        else:                                                      # an unobserved one about
            assert (~o & (x_s >= xa[j]) & (x_s <= xb[j])).any(), j
        assert va[j] == volts[keep][o & (x_s == xa[j])][-1] or np.isnan(va[j])


# ------------------------------------------------- synthetic replicates ----
X = "C10H17+"
FRAG = "C6H9+"


def _replicated(parent, steps, *, ions=(X,)):
    """A folder as `ms2 extract --file` writes it, from (nce, [per-scan {ion
    or m/z: intensity}]) steps: each spectrum the mean of its scans (absence
    = 0), every one of its peaks read in every scan (`SCAN_PEAK_COLUMNS`)."""
    cent, groups, rows, scan = {}, [], [], 0
    for nce, scans in steps:
        act = f"hcd{nce:.2f}"
        key = S.spectrum_key(parent, act)
        names = list(dict.fromkeys(k for s in scans for k in s))
        mzs = [_mz(k) if isinstance(k, str) else float(k) for k in names]
        cent[key] = {"parent_peak_mz": parent, "activation": act, "mz": mzs,
                     "intensity": [sum(s.get(k, 0.0) for s in scans) / len(scans) for k in names]}
        groups.append({"parent_peak_mz": parent, "activation": act,
                       "hcd_energy": [4.0 + 0.2 * nce], "scan_count": len(scans),
                       "t_min": 0.0, "t_max": 1.0})
        for s in scans:
            scan += 1
            for k, m in zip(names, mzs):
                rows.append({"spectrum": key, "peak_mz": m, "scan": scan, "t_s": float(scan),
                             "intensity": float(s.get(k, 0.0)), "n_centroids": int(k in s)})
    summ = {"parent_peaks": [parent], "groups": groups, "isolation_width": 0.4,
            "ms1_scan_count": 1, "ms2_scan_count": scan}
    peaks = SC._frame({c: [r[c] for r in rows] for c in SC.SCAN_PEAK_COLUMNS},
                      SC.SCAN_PEAK_COLUMNS)
    return AN.RunTables(spectra=S.spectra_table("S", cent, summ),
                        parents=S.parents_table("S", cent, summ),
                        window=S._frame({}, S.WINDOW_COLUMNS), manifest={}, scan_peaks=peaks)


def _three_steps():
    """X makes C6H9+ only: three energies of three scans each."""
    return [(1, [{X: 1000.0, FRAG: 10.0}, {X: 1100.0, FRAG: 30.0}, {X: 900.0, FRAG: 20.0}]),
            (10, [{X: 600.0, FRAG: 400.0}, {X: 640.0, FRAG: 380.0}, {X: 560.0, FRAG: 420.0}]),
            (20, [{X: 200.0, FRAG: 800.0}, {X: 260.0, FRAG: 700.0}, {X: 230.0, FRAG: 760.0}])]


def test_per_scan_curves_and_their_spread():
    """F_j = I_j / (I_j + frag_j) scan by scan, its sd (ddof 1) and sem; the
    precursor's CV; S_j = I_j / I(E_min) with I(E_min) the averaged lowest
    energy's precursor. The averaged F is the F of the per-scan MEANS, not
    the mean of the per-scan F."""
    steps = _three_steps()
    res = AN.analyze_run(_replicated(round(_mz(X), 4), steps), _ions(X))
    bd = res.breakdown.set_index("nce")
    i0 = np.mean([s[X] for s in steps[0][1]])
    for nce, scans in steps:
        i = np.array([s[X] for s in scans])
        fr = np.array([s[FRAG] for s in scans])
        f = i / (i + fr)
        r = bd.loc[float(nce)]
        assert r["n_scans"] == 3
        assert r["f"] == pytest.approx(i.mean() / (i.mean() + fr.mean()), rel=1e-12)
        assert r["f_sd"] == pytest.approx(f.std(ddof=1), rel=1e-12)
        assert r["f_sem"] == pytest.approx(f.std(ddof=1) / math.sqrt(3), rel=1e-12)
        assert r["intensity_cv"] == pytest.approx(i.std(ddof=1) / i.mean(), rel=1e-12)
        assert r["s_sd"] == pytest.approx((i / i0).std(ddof=1), rel=1e-12)
    # a ratio of means is not a mean of ratios
    i, fr = np.array([s[X] for s in steps[1][1]]), np.array([s[FRAG] for s in steps[1][1]])
    assert abs((i / (i + fr)).mean() - bd.loc[10.0, "f"]) > 1e-4
    rep = res.replicates
    assert list(rep.columns) == list(RP.REPLICATE_COLUMNS)
    one = rep[(rep["nce"] == 1.0) & (rep["item"] == "fragment")].iloc[0]
    assert (one["name"], one["n_present"], one["presence"]) == (FRAG, 3, 1.0)
    assert one["cv"] == pytest.approx(np.std([10.0, 30.0, 20.0], ddof=1) / 20.0)
    fr_row = res.fragments.set_index("fragment").loc[FRAG]
    assert fr_row["presence_at_max"] == 1.0                  # at NCE 20, its largest fraction
    top = np.array([800.0, 700.0, 760.0])
    assert fr_row["cv_at_max"] == pytest.approx(top.std(ddof=1) / top.mean(), rel=1e-9)


def test_a_fragment_absent_from_some_scans_has_its_presence():
    """Presence = the share of the spectrum's scans that hold it; absence
    counts 0 in its mean and its CV. The same for a window ion's precursor
    (one of the three scans at NCE 1 misses it): its row in
    `ms2_replicates.csv` and its `intensity_cv`."""
    steps = _three_steps()
    steps[0] = (1, [{X: 1000.0, FRAG: 10.0}, {FRAG: 30.0}, {X: 900.0, FRAG: 20.0}])
    steps[1] = (10, [{X: 600.0, FRAG: 400.0, "C5H7+": 30.0}, {X: 640.0, FRAG: 380.0},
                     {X: 560.0, FRAG: 420.0}])
    res = AN.analyze_run(_replicated(round(_mz(X), 4), steps), _ions(X))
    rep = res.replicates
    r = rep[(rep["name"] == "C5H7+")].iloc[0]
    assert (r["n_scans"], r["n_present"]) == (3, 1)
    assert r["presence"] == pytest.approx(1 / 3) and r["mean"] == pytest.approx(10.0)
    assert r["cv"] == pytest.approx(np.std([30.0, 0.0, 0.0], ddof=1) / 10.0)
    pre = rep[(rep["nce"] == 1.0) & (rep["item"] == "precursor")].iloc[0]
    i = np.array([1000.0, 0.0, 900.0])
    assert (pre["n_scans"], pre["n_present"], pre["matched"]) == (3, 2, True)
    assert pre["presence"] == pytest.approx(2 / 3) and pre["mean"] == pytest.approx(i.mean())
    assert pre["cv"] == pytest.approx(i.std(ddof=1) / i.mean(), rel=1e-12)
    assert res.breakdown.set_index("nce").loc[1.0, "intensity_cv"] == pytest.approx(pre["cv"])


def test_a_peak_read_on_another_scale_is_held():
    """A peak's per-scan readings x are put on its averaged height's scale
    (x I_avg / mean(x)) only where that factor lies within `SCALE_BAND`. A
    fragment of 400 that one stray centroid in one of three scans reads (380:
    a factor of 3.16) is not the spectrum's peak read scan by scan -- scaled,
    it would read 1200 in one scan and fake a CV of sqrt 3 -- so it is held
    at its averaged height as a peak no scan reads (`matched` False, in no
    scan, presence and CV NaN), and the window note names its factor."""
    p = round(_mz(X), 4)
    t = _replicated(p, _three_steps())
    sp = t.scan_peaks
    sel = (sp["spectrum"] == _spec(t, 10.0)) & np.isclose(sp["peak_mz"], _mz(FRAG))
    sp.loc[sel, "intensity"] = [0.0, 0.0, 380.0]
    res = AN.analyze_run(t, _ions(X))
    rep = res.replicates
    r = rep[(rep["nce"] == 10.0) & (rep["item"] == "fragment")].iloc[0]
    assert (r["n_present"], r["matched"]) == (0, False) and r["mean"] == pytest.approx(400.0)
    assert math.isnan(r["presence"]) and math.isnan(r["cv"])
    i = np.array([600.0, 640.0, 560.0])
    b = res.breakdown.set_index("nce").loc[10.0]
    assert b["f_sd"] == pytest.approx((i / (i + 400.0)).std(ddof=1), rel=1e-12)
    (w,) = res.windows
    assert any("1 precursor / fragment peak(s) are read in the scans on another scale than "
               "their spectrum's (averaged height / per-scan mean 3.16" in n for n in w.notes)
    assert not any("read in no scan" in n for n in w.notes)
    # the precursor is read: the energy is not one that adds no spread
    assert w.one_scan_energies == 0 and (res.e50["n_one_scan"] == 0).all()
    assert not any("is held: no spread there" in n for n in w.notes)
    # the band's edges (both in it) and every peak's factor
    rows = pd.DataFrame({"peak_mz": np.repeat([100.0, 200.0, 300.0, 400.0], 2),
                         "scan": [1, 2] * 4,
                         "intensity": [10.0, 10.0, 21.0, 21.0, 5.0, 5.0, 4.0, 4.0]})
    sc = RP.spectrum_scans("s", [100.0, 200.0, 300.0, 400.0], [19.0, 10.0, 10.0, 10.0], rows)
    assert RP.SCALE_BAND == (0.5, 2.0)
    np.testing.assert_allclose(sc.factor, [1.9, 10.0 / 21.0, 2.0, 2.5])
    assert list(sc.matched) == [True, False, True, False]
    np.testing.assert_allclose(sc.matrix, [[19.0, 19.0], [10.0, 10.0], [10.0, 10.0], [10.0, 10.0]])
    assert sc.present.tolist() == [[True, True], [False, False], [True, True], [False, False]]
    held = RP.spectrum_scans("s", [100.0], [19.0], rows.iloc[0:0])
    assert np.isnan(held.factor).all()


def test_a_curve_no_scan_reads_has_no_spread():
    """A held peak reads its averaged height in every scan: a curve made of
    held peaks only would scatter by exactly 0 -- a spread nobody measured.
    S (the precursor's alone) has none where its precursor is held, F and
    F_lo none where nothing that enters them is read (the precursor's CV,
    presence NaN; F keeps its read fragment's scatter). Where every
    precursor / fragment peak of a spectrum is held, the energy adds no
    spread, as one with no per-scan row: the same NaN spreads, the same
    count of energies without spread, the same note about the crossing and
    the same band. A spectrum that holds no precursor / fragment peak at all
    holds none: it is not one of those."""
    p = round(_mz(X), 4)
    t = _replicated(p, _three_steps())
    sp = t.scan_peaks
    sp.loc[(sp["spectrum"] == _spec(t, 10.0)) & np.isclose(sp["peak_mz"], _mz(X)),
           "intensity"] = 0.0
    res = AN.analyze_run(t, _ions(X))
    b = res.breakdown.set_index("nce").loc[10.0]
    assert math.isnan(b["s_sd"]) and math.isnan(b["s_sem"]) and math.isnan(b["intensity_cv"])
    fr = np.array([400.0, 380.0, 420.0])
    assert b["f_sd"] == pytest.approx((600.0 / (600.0 + fr)).std(ddof=1), rel=1e-12)
    assert b["f_lo_sd"] == pytest.approx(b["f_sd"], rel=1e-12)
    rep = res.replicates
    pre = rep[(rep["nce"] == 10.0) & (rep["item"] == "precursor")].iloc[0]
    assert (pre["matched"], pre["n_present"]) == (False, 0)
    assert math.isnan(pre["presence"]) and math.isnan(pre["sd"]) and math.isnan(pre["cv"])
    # one held peak among read ones: the energy keeps the read fragment's
    # spread -- not one that adds none (not counted, no 'held' note), and the
    # bootstrap draws its scans (a band of its own, not the all-held one below)
    (w,) = res.windows
    assert w.one_scan_energies == 0 and (res.e50["n_one_scan"] == 0).all()
    assert not any("is held: no spread there" in n for n in w.notes)
    part = _e50(res, p, X)
    assert part["one_scan_bracket"] == 0 and "E50 band" not in part["note"]
    assert (part["e50_nce_lo"], part["e50_nce_hi"]) == (pytest.approx(12.5638, abs=1e-4),
                                                        pytest.approx(12.9102, abs=1e-4))
    # every precursor / fragment peak at NCE 10 held, against no row at NCE 10
    held = _replicated(p, _three_steps())
    held.scan_peaks.loc[held.scan_peaks["spectrum"] == _spec(held, 10.0), "intensity"] = 0.0
    none = _replicated(p, _three_steps())
    none.scan_peaks = none.scan_peaks[none.scan_peaks["spectrum"]
                                      != _spec(none, 10.0)].reset_index(drop=True)
    a, z = AN.analyze_run(held, _ions(X)), AN.analyze_run(none, _ions(X))
    for res in (a, z):
        b = res.breakdown.set_index("nce").loc[10.0]
        assert b[["f_sd", "f_sem", "f_lo_sd", "f_lo_sem", "s_sd", "s_sem",
                  "intensity_cv"]].isna().all()
        assert res.windows[0].one_scan_energies == 1 and (res.e50["n_one_scan"] == 1).all()
        r = _e50(res, p, X)
        assert r["one_scan_bracket"] == 1 and r["e50_nce_lo"] < r["e50_nce"] < r["e50_nce_hi"]
    ra, rz = _e50(a, p, X), _e50(z, p, X)
    assert (ra["e50_nce_lo"], ra["e50_nce_hi"]) == (rz["e50_nce_lo"], rz["e50_nce_hi"])
    assert abs(ra["e50_nce_hi"] - part["e50_nce_hi"]) > 0.05     # the read fragment's scatter
    assert "E50 band: NCE 10 (3 scans, no spread) about the crossing adds no scatter" in ra["note"]
    assert "E50 band: NCE 10 (0 scans) about the crossing adds no scatter" in rz["note"]
    assert any("every precursor / fragment peak of the spectra at NCE 10 is held" in n
               for n in a.windows[0].notes)
    assert a.windows[0].scans_per_energy == [3, 3, 3]
    # an energy whose three scans hold no precursor / fragment peak at all (a
    # line no window ion makes, m/z 60.5): nothing there is held -- not
    # counted, no 'held' note -- and the band is the one without it
    lone = AN.analyze_run(_replicated(p, _three_steps() + [
        (30, [{60.5: 300.0}, {60.5: 320.0}, {60.5: 280.0}])]), _ions(X))
    assert lone.windows[0].scans_per_energy == [3, 3, 3, 3]
    assert lone.windows[0].one_scan_energies == 0 and (lone.e50["n_one_scan"] == 0).all()
    assert not any("is held" in n for n in lone.windows[0].notes)
    r, r0 = _e50(lone, p, X), _e50(AN.analyze_run(_replicated(p, _three_steps()), _ions(X)), p, X)
    assert r["one_scan_bracket"] == 0 and (r["e50_nce_lo"], r["e50_nce_hi"]) == (
        r0["e50_nce_lo"], r0["e50_nce_hi"])


def test_a_band_a_few_ulp_wide_has_zero_width():
    """Bounds that differ in their last bits only (held readings times
    resample counts / n that do not add up to exactly 1) are a band of zero
    width: the E50 note says so, and `ms2 analyze` prints it so, as it does
    an exactly equal pair."""
    lo, hi = 6.810663806263728, 6.810663806263729
    assert lo != hi and RP.zero_width(lo, hi) and RP.zero_width(lo, lo)
    assert not RP.zero_width(13.8428, 13.9412) and not RP.zero_width(NAN, NAN)
    assert not RP.zero_width(lo, NAN)
    row = {"e50_nce_lo": lo, "e50_nce_hi": hi, "e50_lo_censor": "", "e50_hi_censor": "",
           "boot_n": 1000, "one_scan_bracket": 0}
    assert cli._ms2_band(row) == "[16-84 %: 6.81 to 6.81, 1000 resamples; zero width]"
    bd = pd.DataFrame({"parent_mz": 137.1321, "ion": X, "nce": [1.0, 10.0, 20.0],
                       "hcd_v": [4.2, 6.0, 8.0], "f": [0.9, 0.6, 0.2], "f_lo": [0.9, 0.6, 0.2],
                       "s": [1.0, 0.6, 0.2], "n_scans": 3, "f_sd": 0.01, "f_lo_sd": 0.01,
                       "s_sd": 0.01})
    b = {"nce_lo": lo, "nce_hi": hi, "v_lo": 6.5, "v_hi": 6.5, "lo_where": "", "hi_where": "",
         "cross": 1.0, "below": 0.0, "above": 0.0, "gap": 0.0, "none": 0.0}
    e = AN._e50_rows(bd, [X], X, {X: _mz(X)}, BD.N2_MASS, boot={(X, "F"): b}, boot_n=1000,
                     per_scan=True).set_index("curve")
    assert "the band has zero width" in e.loc["F", "note"]


def test_one_scan_energies_add_no_spread():
    """An energy of one scan: no sd there (NaN), counted per window and in
    the E50 rows; the note names its NCE. With no energy of two scans there
    is no interval at all -- said."""
    steps = _three_steps()
    steps[1] = (10, steps[1][1][:1])
    res = AN.analyze_run(_replicated(round(_mz(X), 4), steps), _ions(X))
    (w,) = res.windows
    bd = res.breakdown.set_index("nce")
    assert bd.loc[10.0, "n_scans"] == 1 and math.isnan(bd.loc[10.0, "f_sd"])
    assert w.scans_per_energy == [3, 1, 3] and w.one_scan_energies == 1
    assert (res.e50["n_one_scan"] == 1).all() and (res.e50["boot_n"] == RP.BOOTSTRAP_B).all()
    assert any("1 of 3 energies hold one scan (NCE 10)" in n for n in w.notes)
    single = [(nce, scans[:1]) for nce, scans in _three_steps()]
    res = AN.analyze_run(_replicated(round(_mz(X), 4), single), _ions(X))
    assert (res.e50["boot_n"] == 0).all() and res.e50["e50_nce_lo"].isna().all()
    assert any("no energy holds two scans: no E50 interval" in n for n in res.windows[0].notes)


def test_bootstrap_off_keeps_the_replicate_statistics():
    """--bootstrap 0: the scan-to-scan statistics, no interval. The window's
    energies that add no spread (`n_one_scan`) are a fact of its scans, kept,
    and the window note still names them (without the E50 interval there is
    none to add nothing to); `one_scan_bracket` -- how many energies the
    BAND rests on add none -- is 0 with no band, and no E50 note names
    them, as where no energy holds two scans."""
    res = AN.analyze_run(_replicated(round(_mz(X), 4), _three_steps()), _ions(X), bootstrap=0)
    assert (res.e50["boot_n"] == 0).all() and res.e50["e50_nce_lo"].isna().all()
    assert res.e50["boot_cross"].isna().all()
    assert res.breakdown["f_sd"].notna().all() and res.params["replicates"]["bootstrap"] == 0
    base = _three_steps()
    thin = _replicated(round(_mz(X), 4), [base[0], base[1], (20, base[2][1][:1])])
    on = _e50(AN.analyze_run(thin, _ions(X)), round(_mz(X), 4), X)
    assert (on["boot_n"], on["n_one_scan"], on["one_scan_bracket"]) == (RP.BOOTSTRAP_B, 1, 1)
    assert "E50 band: NCE 20 (1 scan)" in on["note"]
    assert ("replicates: 1 of 3 energies hold one scan (NCE 20): no spread there, nothing to "
            "the E50 interval") in AN.analyze_run(thin, _ions(X)).windows[0].notes
    off_run = AN.analyze_run(thin, _ions(X), bootstrap=0)
    off = off_run.e50
    assert (off["boot_n"] == 0).all() and (off["n_one_scan"] == 1).all()
    assert (off["one_scan_bracket"] == 0).all() and not off["note"].str.contains("E50 band").any()
    assert "replicates: 1 of 3 energies hold one scan (NCE 20): no spread there" in \
        off_run.windows[0].notes
    held = _replicated(round(_mz(X), 4), _three_steps())
    held.scan_peaks.loc[held.scan_peaks["spectrum"] == _spec(held, 10.0), "intensity"] = 0.0
    notes = AN.analyze_run(held, _ions(X), bootstrap=0).windows[0].notes
    assert ("replicates: every precursor / fragment peak of the spectra at NCE 10 is held: no "
            "spread there (counted with the energies of one scan)") in notes
    assert not any("E50 interval" in n for n in notes)
    single = [(nce, scans[:1]) for nce, scans in base]
    none = AN.analyze_run(_replicated(round(_mz(X), 4), single), _ions(X)).e50
    assert (none["boot_n"] == 0).all() and (none["n_one_scan"] == 3).all()
    assert (none["one_scan_bracket"] == 0).all()
    for bad in (-1, 1.5, True):
        with pytest.raises(ValueError, match="bootstrap"):
            AN.analyze_run(_replicated(round(_mz(X), 4), _three_steps()), _ions(X), bootstrap=bad)


def test_too_few_resamples_are_refused(tmp_path, no_server):
    """Below `BOOTSTRAP_MIN` resamples the 16th / 84th percentiles rest on a
    handful of them -- one resample: a 'band' of zero width that is that
    resample's E50, which need not hold the averaged one, though the scans
    scatter. `analyze_run` and the CLI refuse it."""
    p = round(_mz(X), 4)
    t = _replicated(p, _three_steps())
    for bad in (1, 2, RP.BOOTSTRAP_MIN - 1):
        with pytest.raises(ValueError, match="bootstrap"):
            AN.analyze_run(t, _ions(X), bootstrap=bad)
    r = _e50(AN.analyze_run(t, _ions(X), bootstrap=RP.BOOTSTRAP_MIN), p, X)
    assert r["boot_n"] == RP.BOOTSTRAP_MIN and r["e50_nce_lo"] < r["e50_nce"] < r["e50_nce_hi"]
    t.manifest = {"parameters": {"isolation_width_used": 0.4}}
    run = _write_tables(tmp_path / "run", t)
    t.scan_peaks.to_parquet(run / "ms2_scan_peaks.parquet", index=False)
    argv = ["ms2", "analyze", "--run-dir", str(run), "--ion", f"{p}={X}", "--no-figures",
            "--out-dir", str(tmp_path / "out"), "--bootstrap"]
    for bad in ("1", str(RP.BOOTSTRAP_MIN - 1)):
        with pytest.raises(SystemExit, match="--bootstrap must be"):
            cli.main(argv + [bad])
    assert cli.main(argv + [str(RP.BOOTSTRAP_MIN)]) == 0


def test_a_crossing_some_resamples_miss():
    """F crosses 0.5 at the last step only just: some resamples stay above
    it. The interval's upper bound is then beyond the highest energy (NaN),
    the share that crosses is reported and the note says both."""
    steps = [(1, [{X: 1000.0, FRAG: 10.0}, {X: 1000.0, FRAG: 12.0}, {X: 1000.0, FRAG: 8.0}]),
             (10, [{X: 700.0, FRAG: 300.0}] * 3),
             (20, [{X: 520.0, FRAG: 480.0}, {X: 440.0, FRAG: 560.0}, {X: 520.0, FRAG: 480.0}])]
    res = AN.analyze_run(_replicated(round(_mz(X), 4), steps), _ions(X))
    r = _e50(res, round(_mz(X), 4), X)
    assert r["flag"] == BD.FLAG_INTERPOLATED and r["e50_nce"] == pytest.approx(19.68, abs=0.01)
    # a resample of only the two 520 scans stays above 0.5: (2/3)^3 of them
    assert r["boot_cross"] == pytest.approx(1 - (2 / 3) ** 3, abs=0.03)
    assert math.isnan(r["e50_nce_hi"]) and r["e50_nce_lo"] < r["e50_nce"]
    assert "crosses in" in r["note"] and "upper bound lies above the highest energy" in r["note"]
    # the resamples' own reading: above the highest energy where none crosses
    b = RP.interval(np.array([15.0, 16.0, np.nan, np.nan]), np.array([7.0, 7.2, np.nan, np.nan]),
                    np.array([0, 0, BD.FLAGS.index(BD.FLAG_ABOVE),
                              BD.FLAGS.index(BD.FLAG_ABOVE)]))
    assert (b["cross"], b["above"]) == (0.5, 0.5) and b["hi_censored"]
    assert b["nce_lo"] == pytest.approx(15.0 + 0.48 * 1.0)
    assert (b["lo_where"], b["hi_where"]) == ("", "above")
    assert (r["e50_lo_censor"], r["e50_hi_censor"]) == ("", "above")


def _lower_steps():
    """F crosses 0.5 between NCE 1 and 10 (F 0.51 at NCE 1): a resample of
    two or three of the 450-precursor scan at NCE 1 is below 0.5 there
    already -- 7 / 27 of them."""
    return [(1, [{X: 540.0, FRAG: 460.0}, {X: 450.0, FRAG: 550.0}, {X: 540.0, FRAG: 460.0}]),
            (10, [{X: 300.0, FRAG: 700.0}] * 3), (20, [{X: 100.0, FRAG: 900.0}] * 3)]


def test_a_lower_bound_below_the_lowest_energy():
    """The mirror of the test above: the resamples below 0.5 at the lowest
    energy count as E50s below every energy, the 16th percentile falls among
    them -- the lower bound is NaN and said, the upper one measured (the
    three-540 resamples, 8 / 27 of them: E50 2.5)."""
    p = round(_mz(X), 4)
    r = _e50(AN.analyze_run(_replicated(p, _lower_steps()), _ions(X)), p, X)
    assert r["flag"] == BD.FLAG_INTERPOLATED and r["e50_nce"] == pytest.approx(1.4286, abs=1e-4)
    assert r["boot_cross"] == pytest.approx(1 - 7 / 27, abs=0.03)
    assert math.isnan(r["e50_nce_lo"]) and r["e50_nce_hi"] == pytest.approx(2.5, abs=1e-9)
    assert math.isnan(r["e50_v_lo"]) and r["e50_v_hi"] == pytest.approx(4.5, abs=1e-9)
    assert (r["e50_lo_censor"], r["e50_hi_censor"]) == ("below", "")
    assert "(below the lowest energy in 25.9%, above the highest in 0.0%)" in r["note"]
    assert "the interval's lower bound lies below the lowest energy" in r["note"]
    assert "upper bound" not in r["note"]
    fi = BD.FLAGS.index
    b = RP.interval(np.array([np.nan, np.nan, 15.0, 16.0]), np.array([np.nan, np.nan, 7.0, 7.2]),
                    np.array([fi(BD.FLAG_BELOW)] * 2 + [0, 0]))
    assert (b["cross"], b["below"], b["lo_censored"], b["hi_censored"]) == (0.5, 0.5, True, False)
    assert math.isnan(b["nce_lo"]) and b["nce_hi"] == pytest.approx(15.52)
    assert (b["lo_where"], b["hi_where"]) == ("below", "")


def test_a_censored_bound_is_said_on_its_own_side():
    """A bound past the energies is NaN, and the row says on which side: a
    16th percentile among the resamples above the highest energy lies ABOVE
    it (not below the lowest because some other resample is below), and the
    mirror. The CSV carries the side (`e50_lo_censor` / `e50_hi_censor`)."""
    fi = BD.FLAGS.index
    for n_below, n_above, side in ((50, 900, "above"), (900, 50, "below")):
        code = np.array([fi(BD.FLAG_BELOW)] * n_below + [0] * 50 + [fi(BD.FLAG_ABOVE)] * n_above)
        e = np.full(len(code), np.nan)
        e[code == 0] = np.linspace(11.0, 13.0, 50)
        b = RP.interval(e, e / 2, code)
        assert math.isnan(b["nce_lo"]) and math.isnan(b["nce_hi"])
        assert (b["lo_where"], b["hi_where"]) == (side, side)
        assert b["lo_censored"] and b["hi_censored"]
    # a percentile between the last resample below and the first above:
    # the nearer one's side
    inf = np.inf
    assert RP._pct(np.array([-inf] * 160 + [inf] * 840), 0.16) == inf      # position 159.84
    assert RP._pct(np.array([-inf] * 161 + [inf] * 839), 0.16) == -inf
    assert RP._pct(np.array([-inf] * 160 + [inf] * 840), 0.159) == -inf    # position 158.84
    assert RP._pct(np.array([-inf, 3.0, inf]), 0.5) == 3.0
    # end to end: the averaged F 0.597 at both energies (no E50); a quarter
    # of the scans with the fragment x 3.75 -- 6 % of resamples cross, 87 %
    # stay above, 7 % are below at NCE 1
    sc = [{X: 1000.0, FRAG: 400.0}] * 3 + [{X: 1000.0, FRAG: 1500.0}]
    p = round(_mz(X), 4)
    r = _e50(AN.analyze_run(_replicated(p, [(1, sc), (20, sc)]), _ions(X)), p, X)
    assert r["flag"] == BD.FLAG_ABOVE and r["boot_cross"] == pytest.approx(0.062)
    assert (r["e50_lo_censor"], r["e50_hi_censor"]) == ("above", "above")
    assert "the interval's lower bound lies above the highest energy" in r["note"]
    assert "lies below the lowest energy" not in r["note"]


def _gap_steps():
    """F 0.501 at NCE 10 from five scattered scans, unobserved at NCE 20 (a
    peak neither the precursor nor its fragment), 0 at NCE 30: the averaged
    curve crosses across the hole (flag 'gap')."""
    return [(1, [{X: 1000.0, FRAG: 10.0}, {X: 1100.0, FRAG: 12.0}, {X: 900.0, FRAG: 9.0}]),
            (10, [{X: 600.0, FRAG: 400.0}, {X: 400.0, FRAG: 600.0}, {X: 700.0, FRAG: 300.0},
                  {X: 300.0, FRAG: 700.0}, {X: 510.0, FRAG: 500.0}]),
            (20, [{100.0: 50.0}, {100.0: 60.0}, {100.0: 55.0}]),
            (30, [{FRAG: 900.0}, {FRAG: 950.0}, {FRAG: 1000.0}])]


def test_resamples_that_cross_across_a_hole_count():
    """A resample whose curve crosses across an unobserved energy has an E50
    somewhere between the two energies about the hole: it is in the ranking,
    and a percentile that falls among such resamples is NaN ('gap', said) --
    never a bound read off the other resamples alone. The note's shares add
    up to 100 %."""
    p = round(_mz(X), 4)
    res = AN.analyze_run(_replicated(p, _gap_steps()), _ions(X))
    r = _e50(res, p, X)
    assert r["flag"] == BD.FLAG_GAP and math.isnan(r["e50_nce"])
    assert r["boot_cross"] == pytest.approx(0.447)
    # S too: every resample's S is NaN at the unobserved energy (0/0 there --
    # not 0 / I(E_min), which would let it cross through the hole)
    s = _e50(res, p, X, "S")
    assert s["flag"] == BD.FLAG_GAP and s["boot_cross"] == pytest.approx(0.472)
    assert math.isnan(s["e50_nce_hi"]) and s["e50_hi_censor"] == "gap"
    assert "across an unobserved energy in 52.8%" in s["note"]
    # 447 resamples cross below NCE 10, 553 across the hole (10-30): the
    # 16th percentile is one of the first, the 84th one of the second
    assert r["e50_nce_lo"] < 10.0 and math.isnan(r["e50_nce_hi"])
    assert (r["e50_lo_censor"], r["e50_hi_censor"]) == ("", "gap")
    assert ("crosses in 44.7% of resamples (below the lowest energy in 0.0%, above the highest "
            "in 0.0%, across an unobserved energy in 55.3%)") in r["note"]
    assert "the interval's upper bound lies across an unobserved energy" in r["note"]
    shares = [float(x) for x in re.findall(r"([0-9.]+)%", r["note"])]
    assert sum(shares) == pytest.approx(100.0, abs=0.15)
    # the bound is the 16th percentile of ALL the resamples with an answer
    nan, gi = np.nan, BD.FLAGS.index(BD.FLAG_GAP)
    br = (np.array([nan, nan, 10.0, 10.0]), np.array([nan, nan, 30.0, 30.0]),
          np.array([nan, nan, 6.0, 6.0]), np.array([nan, nan, 10.0, 10.0]))
    b = RP.interval(np.array([9.0, 9.5, nan, nan]), np.array([5.8, 5.9, nan, nan]),
                    np.array([0, 0, gi, gi]), bracket=br)
    assert b["nce_lo"] == pytest.approx(9.24) and b["v_lo"] == pytest.approx(5.848)
    assert math.isnan(b["nce_hi"]) and math.isnan(b["v_hi"])
    assert (b["gap"], b["lo_where"], b["hi_where"], b["hi_censored"]) == (0.5, "", "gap", True)
    with pytest.raises(ValueError, match="bracket"):
        RP.interval(np.array([9.0, nan]), np.array([5.8, nan]), np.array([0, gi]))
    # one hole resample (between NCE 1 and 30) among ten crossings: where the
    # percentile falls on a crossing it only moves its rank -- the bound is
    # the outer reading (9.06, not 9.16; 9.84, not 9.74), never NaN
    e = np.r_[nan, np.arange(9.0, 9.95, 0.1)]
    code = np.r_[gi, np.zeros(10, dtype=int)]
    br = (np.r_[1.0, [nan] * 10], np.r_[30.0, [nan] * 10], np.r_[4.2, [nan] * 10],
          np.r_[10.0, [nan] * 10])
    b = RP.interval(e, e / 2, code, bracket=br)
    assert (b["nce_lo"], b["nce_hi"]) == (pytest.approx(9.06), pytest.approx(9.84))
    assert (b["v_lo"], b["v_hi"]) == (pytest.approx(4.53), pytest.approx(4.92))
    assert (b["lo_where"], b["hi_where"], b["gap"]) == ("", "", pytest.approx(1 / 11))


def _gaps(n_inter, n_gap, *, inter_first=True, inter=(11.0, 19.0), hole=(10.0, 30.0),
          volts=(5.0, 15.0)):
    """`n_inter` resamples crossing at evenly spread NCE in `inter` (volts
    NCE / 2) and `n_gap` crossing across a hole between the NCE `hole` (its
    two energies' `volts`), in that order or the other."""
    fi, nan = BD.FLAGS.index, np.nan
    parts = [(np.linspace(*inter, n_inter), np.full(n_inter, fi(BD.FLAG_INTERPOLATED)),
              [np.full(n_inter, nan)] * 4),
             (np.full(n_gap, nan), np.full(n_gap, fi(BD.FLAG_GAP)),
              [np.full(n_gap, x) for x in (*hole, *volts)])]
    if not inter_first:
        parts.reverse()
    e = np.concatenate([a for a, _, _ in parts])
    code = np.concatenate([c for _, c, _ in parts])
    br = tuple(np.concatenate([b[k] for _, _, b in parts]) for k in range(4))
    return e, e / 2, code, br


def test_the_interval_s_censoring_rules():
    """`interval` case by case, each rule a resample of a hole ('gap') or
    without an answer can break:
      * a percentile interpolated between an E50 and a gap resample's end is
        NaN ('gap'): either order statistic it needs is enough (two
        resamples; 840 crossings at 11-19 and 160 across a hole at 20-30, the
        84th percentile between the last crossing and the first hole);
      * a hole resample at the top of ONE ordering only (its upper end, 30)
        is a gap there too: never a bound read at the hole's far energy;
      * a resample without an answer (one energy observed) is out of the
        ranking, counted only in `none`;
      * the volts bounds need the volts of each hole's two energies: NaN
        where one of them is unknown, though the crossings' volts are."""
    fi, nan = BD.FLAGS.index, np.nan
    b = RP.interval(np.array([9.0, nan]), np.array([4.5, nan]),
                    np.array([fi(BD.FLAG_INTERPOLATED), fi(BD.FLAG_GAP)]),
                    bracket=(np.array([nan, 10.0]), np.array([nan, 30.0]), np.array([nan, 5.0]),
                             np.array([nan, 15.0])))
    assert math.isnan(b["nce_lo"]) and math.isnan(b["nce_hi"])
    assert (b["lo_where"], b["hi_where"]) == ("gap", "gap")
    e, v, code, br = _gaps(840, 160, hole=(20.0, 30.0), volts=(10.0, 15.0))
    b = RP.interval(e, v, code, bracket=br)
    assert b["nce_lo"] == pytest.approx(12.5241, abs=1e-4) and b["lo_where"] == ""
    assert math.isnan(b["nce_hi"]) and math.isnan(b["v_hi"]) and b["hi_where"] == "gap"
    e, v, code, br = _gaps(700, 300)
    b = RP.interval(e, v, code, bracket=br)
    assert math.isnan(b["nce_hi"]) and b["hi_where"] == "gap" and math.isnan(b["v_hi"])
    # a hole about every crossing (NCE 1-30): its resamples at the bottom of
    # the lower ends' ordering and the top of the upper ends' -- a gap on
    # both sides, never the bound 1 or 30
    e, v, code, br = _gaps(700, 300, hole=(1.0, 30.0), volts=(4.2, 15.0))
    b = RP.interval(e, v, code, bracket=br)
    assert math.isnan(b["nce_lo"]) and math.isnan(b["nce_hi"])
    assert (b["lo_where"], b["hi_where"]) == ("gap", "gap")
    e = np.r_[np.linspace(11.0, 19.0, 800), np.full(200, nan)]
    code = np.r_[np.full(800, fi(BD.FLAG_INTERPOLATED)), np.full(200, fi(BD.FLAG_SINGLE))]
    b = RP.interval(e, e / 2, code)
    assert (b["nce_lo"], b["nce_hi"]) == (pytest.approx(12.28), pytest.approx(17.72))
    assert (b["v_lo"], b["v_hi"]) == (pytest.approx(6.14), pytest.approx(8.86))
    assert (b["cross"], b["none"], b["lo_where"], b["hi_where"]) == (0.8, 0.2, "", "")
    e, v, code, br = _gaps(700, 300, inter_first=False, hole=(1.0, 20.0), volts=(nan, 10.0))
    b = RP.interval(e, v, code, bracket=br)
    assert math.isnan(b["v_lo"]) and math.isnan(b["v_hi"])
    assert np.isfinite(e[code == fi(BD.FLAG_INTERPOLATED)] / 2).all()


#: A co-isolate of X and the one fragment only it makes (a synthetic pair).
Y, YF = "C5H9O4+", "C2H3O2+"


def _end_hole_steps(end):
    """X and its co-isolate Y, two scans per energy. 'start': Y (its
    precursor and its fragment) is read in one of the two NCE-1 scans only;
    'end': Y is gone at NCE 20 but for its fragment, read in one of the two
    scans. A resample that draws the other scan twice there does not observe
    Y's curve at that energy at all -- its scans hold neither Y nor anything
    Y makes."""
    if end == "start":
        return [(1, [{X: 1000.0, FRAG: 10.0, Y: 50.0, YF: 10.0}, {X: 1100.0, FRAG: 20.0}]),
                (10, [{X: 600.0, FRAG: 400.0, Y: 20.0, YF: 60.0},
                      {X: 640.0, FRAG: 380.0, Y: 25.0, YF: 50.0}]),
                (20, [{X: 200.0, FRAG: 800.0, Y: 5.0, YF: 80.0},
                      {X: 260.0, FRAG: 700.0, Y: 5.0, YF: 80.0}])]
    return [(1, [{X: 1000.0, FRAG: 10.0, Y: 50.0, YF: 2.0}, {X: 1100.0, FRAG: 20.0, Y: 48.0, YF: 3.0}]),
            (10, [{X: 600.0, FRAG: 400.0, Y: 30.0, YF: 20.0},
                  {X: 640.0, FRAG: 380.0, Y: 33.0, YF: 22.0}]),
            (20, [{X: 200.0, FRAG: 800.0, YF: 40.0}, {X: 260.0, FRAG: 700.0}])]


def test_a_resample_that_does_not_observe_an_end_energy_is_a_hole_there():
    """Y's averaged F is observed at every energy; a resample whose scans at
    the first (last) energy hold nothing of Y does not observe it there, and
    its reading at the first energy it observes -- already below 0.5 (still
    above 0.5 at the last) -- says only that its E50 lies at or below that
    energy (at or above it), not below (above) every energy: a hole at that
    end of the energies, ranked as the interval it is known to lie in, as a
    resample that crosses across an unobserved energy is ('gap'). About a
    quarter of the resamples ((1/2)^2) are such holes, so neither bound is
    known: NaN, 'gap' on both sides -- never a bound read off the crossing
    resamples alone (7.000 at the start, 11.667 at the end), and the note
    never counts them below the lowest / above the highest energy. With two
    energies only, such a hole leaves its resample one observed energy: what
    it reads there bounds its E50 all the same -- the same hole, never a
    resample left out of the ranking ('one energy observed'), whose band
    would be the crossings' alone. Read faintly in the other scan too, Y is
    observed in every resample: a measured band."""
    p = round(_mz(X), 4)
    ions = _ions(X, Y)
    for end, e50, share in (("start", 6.525, 0.267), ("end", 11.6667, 0.262)):
        res = AN.analyze_run(_replicated(p, _end_hole_steps(end), ions=(X, Y)), ions,
                             iso_width=60.0)
        _end_hole_row(res, p, e50, share, end)
    # two energies only: a hole leaves its resample ONE observed energy (F
    # 0.25 at NCE 10 at the start, 0.6 there at the end) -- what it reads
    # there still bounds its E50 (at or below NCE 10; at or above it), the
    # same hole, never a resample out of the ranking: the band is not the
    # crossings' alone (6.143-7.000; at the end 11.667-11.667, of zero width)
    for end, steps, e50, share in (("start", _end_hole_steps("start")[:2], 6.525, 0.267),
                                   ("end", _end_hole_steps("end")[1:], 11.6667, 0.262)):
        res = AN.analyze_run(_replicated(p, steps, ions=(X, Y)), ions, iso_width=60.0)
        r = _end_hole_row(res, p, e50, share, end)
        assert "one energy observed" not in r["note"] and "zero width" not in r["note"]
        assert r["one_scan_bracket"] == 1                  # the hole's energy: no F spread there
    # the control: Y read faintly in the other NCE-1 scan -- every resample
    # observes it there, and the band is measured
    steps = _end_hole_steps("start")
    steps[0][1][1].update({Y: 0.5, YF: 0.1})
    r = _e50(AN.analyze_run(_replicated(p, steps, ions=(X, Y)), ions, iso_width=60.0), p, Y)
    assert (r["e50_nce_lo"], r["e50_nce_hi"]) == (pytest.approx(6.143, abs=1e-3),
                                                  pytest.approx(7.0, abs=1e-9))
    assert (r["e50_lo_censor"], r["e50_hi_censor"], r["boot_cross"]) == ("", "", 1.0)
    # the ends are the averaged curve's, not the axis's: Y in no scan at NCE 1
    # (at NCE 20) -- the averaged curve itself starts at NCE 10 below 0.5
    # (ends there above it), every resample observes NCE 10, and all of them
    # lie below (above) its energies, as the averaged reading does
    start, end = _end_hole_steps("start"), _end_hole_steps("end")
    start[0] = (1, [{X: 1000.0, FRAG: 10.0}, {X: 1100.0, FRAG: 20.0}])
    end[2] = (20, [{X: 200.0, FRAG: 800.0}, {X: 260.0, FRAG: 700.0}])
    for steps, flag, side in ((start, BD.FLAG_BELOW, "below"), (end, BD.FLAG_ABOVE, "above")):
        r = _e50(AN.analyze_run(_replicated(p, steps, ions=(X, Y)), ions, iso_width=60.0), p, Y)
        assert r["flag"] == flag and r["boot_cross"] == 0.0
        assert (r["e50_lo_censor"], r["e50_hi_censor"]) == (side, side) and r["note"] == ""


def _end_hole_row(res, p, e50, share, end):
    """Y's F row of an end-hole window: the averaged crossing at `e50`, a
    `share` of the resamples holes at that `end`, so both bounds are NaN,
    'gap' -- and never counted below the lowest / above the highest energy."""
    r = _e50(res, p, Y)
    assert r["flag"] == BD.FLAG_INTERPOLATED and r["e50_nce"] == pytest.approx(e50, abs=1e-3)
    assert r["boot_cross"] == pytest.approx(1 - share, abs=1e-3)
    assert math.isnan(r["e50_nce_lo"]) and math.isnan(r["e50_nce_hi"]), end
    assert math.isnan(r["e50_v_lo"]) and math.isnan(r["e50_v_hi"])
    assert (r["e50_lo_censor"], r["e50_hi_censor"]) == ("gap", "gap"), end
    assert (f"(below the lowest energy in 0.0%, above the highest in 0.0%, across an "
            f"unobserved energy in {share:.1%})") in r["note"], r["note"]
    assert "lies below the lowest energy" not in r["note"]
    assert "lies above the highest energy" not in r["note"]
    assert ("the interval's lower bound lies across an unobserved energy; the interval's "
            "upper bound lies across an unobserved energy") in r["note"]
    return r


def test_the_interval_ranks_a_hole_at_an_end_as_its_interval():
    """`interval` given end holes (`bootstrap_e50`: a 'gap' resample whose
    bracket runs from -inf to the first energy it observes, or from the last
    to +inf): 100 resamples below the lowest energy, 100 holes at the start
    (E50 <= NCE 10) and 800 crossings at 11-19 -- the 16th percentile is
    -inf on the holes' lower ends, NCE 10 on their upper ends: not known,
    'gap' (not 'below'). With 300 resamples truly below, the 16th percentile
    is below whatever the holes hold: 'below'. The mirror at the end; the
    volts bounds of the crossings are kept (an end hole's volts are known on
    the side it observes)."""
    fi, nan, inf = BD.FLAGS.index, np.nan, np.inf
    for n_true, side in ((100, "gap"), (300, "below")):
        n_hole, n_x = 100, 1000 - n_true - 100
        e = np.r_[np.full(n_true + n_hole, nan), np.linspace(11.0, 19.0, n_x)]
        code = np.r_[np.full(n_true, fi(BD.FLAG_BELOW)), np.full(n_hole, fi(BD.FLAG_GAP)),
                     np.full(n_x, fi(BD.FLAG_INTERPOLATED))]
        br = (np.r_[np.full(n_true, nan), np.full(n_hole, -inf), np.full(n_x, nan)],
              np.r_[np.full(n_true, nan), np.full(n_hole, 10.0), np.full(n_x, nan)],
              np.r_[np.full(n_true, nan), np.full(n_hole, -inf), np.full(n_x, nan)],
              np.r_[np.full(n_true, nan), np.full(n_hole, 6.0), np.full(n_x, nan)])
        b = RP.interval(e, e / 2, code, bracket=br)
        assert math.isnan(b["nce_lo"]) and b["lo_where"] == side, (n_true, b)
        # the 84th percentile: a crossing, the same on both ends' orderings
        want = np.interp(0.84 * 999 - (n_true + n_hole), np.arange(n_x),
                         np.linspace(11.0, 19.0, n_x))
        assert b["nce_hi"] == pytest.approx(want, abs=1e-9)
        assert b["hi_where"] == "" and b["v_hi"] == pytest.approx(want / 2, abs=1e-9)
        assert (b["below"], b["gap"]) == (n_true / 1000, 0.1)
        # the mirror: holes at the end (E50 >= NCE 20) and resamples above
        m = RP.interval(40.0 - e, (40.0 - e) / 2,
                        np.where(code == fi(BD.FLAG_BELOW), fi(BD.FLAG_ABOVE), code),
                        bracket=(np.where(np.isfinite(br[1]), 30.0, nan), -br[0],
                                 np.where(np.isfinite(br[3]), 15.0, nan), -br[2]))
        assert math.isnan(m["nce_hi"]) and m["hi_where"] == {"gap": "gap",
                                                              "below": "above"}[side]
        assert m["lo_where"] == "" and m["nce_lo"] == pytest.approx(40.0 - want, abs=1e-9)
        assert m["v_lo"] == pytest.approx((40.0 - want) / 2, abs=1e-9)
        assert (m["above"], m["gap"]) == (n_true / 1000, 0.1)
    # the holes' observed end beyond the crossings (holes in [-inf, NCE 10],
    # crossings at 2-9): 100 holes, then 100 truly below, then 800 crossings
    # -- the 16th percentile is -inf on the lower ends, a crossing on the
    # upper ends, and an order statistic it needs is a resample below, not a
    # hole: still not known ('gap', NaN), never the bound -inf. The mirror:
    # 800 crossings at 11-18, 100 above, then 100 holes in [NCE 10, +inf]
    code = np.r_[np.full(100, fi(BD.FLAG_GAP)), np.full(100, fi(BD.FLAG_BELOW)),
                 np.full(800, fi(BD.FLAG_INTERPOLATED))]
    e = np.r_[np.full(200, nan), np.linspace(2.0, 9.0, 800)]
    hole = (-inf, 10.0, -inf, 6.0)
    br = tuple(np.r_[np.full(100, x), np.full(900, nan)] for x in hole)
    b = RP.interval(e, e / 2, code, bracket=br)
    assert math.isnan(b["nce_lo"]) and math.isnan(b["v_lo"]) and b["lo_where"] == "gap", b
    assert b["hi_where"] == "" and np.isfinite(b["nce_hi"])
    code = np.r_[np.full(800, fi(BD.FLAG_INTERPOLATED)), np.full(100, fi(BD.FLAG_ABOVE)),
                 np.full(100, fi(BD.FLAG_GAP))]
    e = np.r_[np.linspace(11.0, 18.0, 800), np.full(200, nan)]
    hole = (10.0, inf, 6.0, inf)
    br = tuple(np.r_[np.full(900, nan), np.full(100, x)] for x in hole)
    b = RP.interval(e, e / 2, code, bracket=br)
    assert math.isnan(b["nce_hi"]) and math.isnan(b["v_hi"]) and b["hi_where"] == "gap", b
    assert b["lo_where"] == "" and np.isfinite(b["nce_lo"])


def _few_end_holes(end):
    """Five scans per energy. 'start': Y (its precursor and its fragment) is
    read in three of the five NCE-1 scans; 'end': Y is gone at NCE 20 but
    for its fragment, read in three of the five. A resample draws none of
    those three in (2/5)^5 = 1 % of its draws: a few holes at that end."""
    if end == "start":
        return [(1, [{X: 1000.0, FRAG: 10.0, Y: 50.0, YF: 10.0}, {X: 1100.0, FRAG: 20.0},
                     {X: 950.0, FRAG: 15.0, Y: 46.0, YF: 12.0}, {X: 1050.0, FRAG: 12.0},
                     {X: 980.0, FRAG: 18.0, Y: 54.0, YF: 8.0}]),
                (10, [{X: 600.0, FRAG: 400.0, Y: 20.0, YF: 60.0},
                      {X: 640.0, FRAG: 380.0, Y: 25.0, YF: 50.0},
                      {X: 620.0, FRAG: 390.0, Y: 22.0, YF: 56.0},
                      {X: 580.0, FRAG: 410.0, Y: 18.0, YF: 62.0},
                      {X: 610.0, FRAG: 395.0, Y: 24.0, YF: 52.0}]),
                (20, [{X: 200.0, FRAG: 800.0, Y: 5.0, YF: 80.0}] * 5)]
    return [(1, [{X: 1000.0, FRAG: 10.0, Y: 50.0, YF: 2.0}] * 5),
            (10, [{X: 600.0, FRAG: 400.0, Y: 30.0, YF: 20.0},
                  {X: 640.0, FRAG: 380.0, Y: 33.0, YF: 22.0},
                  {X: 620.0, FRAG: 390.0, Y: 28.0, YF: 18.0},
                  {X: 580.0, FRAG: 410.0, Y: 35.0, YF: 21.0},
                  {X: 610.0, FRAG: 395.0, Y: 31.0, YF: 24.0}]),
            (20, [{X: 200.0, FRAG: 800.0, YF: 40.0}, {X: 260.0, FRAG: 700.0},
                  {X: 230.0, FRAG: 760.0, YF: 36.0}, {X: 210.0, FRAG: 780.0},
                  {X: 240.0, FRAG: 750.0, YF: 44.0}])]


@pytest.mark.parametrize("end, holes, band", [
    ("start", 0.006, (6.193477, 6.657989)),
    ("end", 0.009, (11.528662, 11.788079))])
def test_a_few_end_holes_leave_a_measured_band(end, holes, band):
    """A few holes at an end (under 1 % of the resamples) sit far from the
    16th / 84th percentiles: the band is measured on NCE and on volts (an
    end hole's volts are known at the energy it observes, its outer end
    infinite) and on E_cm, the holes counted in the note."""
    p = round(_mz(X), 4)
    ions = _ions(X, Y)
    r = _e50(AN.analyze_run(_replicated(p, _few_end_holes(end), ions=(X, Y)), ions,
                            iso_width=60.0), p, Y)
    assert r["flag"] == BD.FLAG_INTERPOLATED and r["boot_cross"] == pytest.approx(1 - holes)
    assert (r["e50_nce_lo"], r["e50_nce_hi"]) == (pytest.approx(band[0], abs=1e-6),
                                                  pytest.approx(band[1], abs=1e-6))
    # volts 4 + 0.2 NCE at every energy (`_replicated`)
    assert (r["e50_v_lo"], r["e50_v_hi"]) == (pytest.approx(4.0 + 0.2 * band[0], abs=1e-6),
                                              pytest.approx(4.0 + 0.2 * band[1], abs=1e-6))
    assert np.isfinite([r["e50_ecm_lo"], r["e50_ecm_hi"]]).all()
    assert (r["e50_lo_censor"], r["e50_hi_censor"]) == ("", "")
    assert f"across an unobserved energy in {holes:.1%})" in r["note"]
    assert "bound lies" not in r["note"]


def test_observed_ends_reads_each_curve_s_first_and_last_energy():
    """`breakdown.observed_ends` in `e50_many`'s order (NCE ascending, an
    energy of NaN NCE never one): each column's first and last observed
    NCE, their volts and (`with_values`) the curve's value there; NaN for a
    column observed nowhere (but at a NaN NCE)."""
    nan = np.nan
    nce, volts = [10.0, 1.0, nan, 20.0], [6.0, 4.2, 5.0, nan]
    y = np.array([[0.6, nan, nan, 0.5],              # NCE 10
                  [nan, 0.9, nan, 0.8],              # NCE 1
                  [nan, nan, 0.7, 0.3],              # NCE NaN
                  [0.2, nan, nan, 0.1]])             # NCE 20
    want = ([10.0, 1.0, nan, 1.0], [20.0, 1.0, nan, 20.0], [6.0, 4.2, nan, 4.2],
            [nan, 4.2, nan, nan], [0.6, 0.9, nan, 0.8], [0.2, 0.9, nan, 0.1])
    got = BD.observed_ends(nce, y, volts, with_values=True)
    assert len(got) == 6
    for g, w in zip(got, want):
        np.testing.assert_array_equal(g, w)
    four = BD.observed_ends(nce, y, volts)
    assert len(four) == 4
    for g, w in zip(four, want):
        np.testing.assert_array_equal(g, w)
    assert all(np.isnan(a).all() for a in BD.observed_ends([], np.zeros((0, 2)), [],
                                                           with_values=True))


def test_a_resample_of_one_observed_energy_or_none_is_ranked_as_what_it_reads():
    """`bootstrap_e50` where the averaged curve observes two energies (NCE 1
    and 10) and a resample observes one of them, or none: what it reads
    still bounds its E50, and it is ranked as that, never left out of the
    ranking ('single_energy' / 'no_data', counted in `none` only). Observed
    at the averaged curve's first energy alone: below 0.5 there, below every
    energy ('below_range'); at or above 0.5, a hole, its E50 in [NCE 1,
    +inf] ('gap'). At the last energy alone: at or above 0.5, above every
    energy ('above_range'); below 0.5, in [-inf, NCE 10]. Nowhere: in [-inf,
    +inf]. X reads F 0.8 / 0.2 in its two NCE-1 scans and is in one of the
    two NCE-10 scans (F 0.1); Z (no fragment: F 1) in one scan of each
    energy, not the same one; W (F 0.75 at NCE 1, 0.1 at NCE 10) in one of
    the two NCE-1 scans. All three are drawn from the same scans."""
    fi, inf = BD.FLAGS.index, np.inf
    w = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])            # fragments x ions (X, Z, W)
    per = [(np.array([[100.0, 100.0], [50.0, 0.0], [30.0, 0.0]]),
            np.array([[25.0, 400.0], [10.0, 0.0]])),
           (np.array([[10.0, 0.0], [0.0, 20.0], [5.0, 5.0]]), np.array([[90.0, 0.0], [45.0, 45.0]]))]
    b = 4000
    out = RP.bootstrap_e50([1.0, 10.0], [4.2, 6.0], per, w, w, b=b, seed=7)
    code, (xa, xb, va, vb) = out["f"][2], out["f"][3]
    assert not np.isin(code, [fi(BD.FLAG_SINGLE), fi(BD.FLAG_NO_DATA)]).any()
    gap, below, above = (code == fi(f) for f in (BD.FLAG_GAP, BD.FLAG_BELOW, BD.FLAG_ABOVE))
    inter = code == fi(BD.FLAG_INTERPOLATED)
    br = np.stack([xa, xb, va, vb])                               # 4 x ions x resamples
    # X: NCE 10 missed (1/4) -> NCE 1 alone, at 0.8 a hole above it (1/4 of
    # those), else below every energy
    assert set(map(tuple, br[:, 0, gap[0]].T)) == {(1.0, inf, 4.2, inf)}
    assert gap[0].mean() == pytest.approx(1 / 16, abs=0.02)
    assert (gap[0] | below[0] | inter[0]).all() and inter[0].mean() == pytest.approx(3 / 16, abs=0.02)
    # Z: both energies or NCE 10 alone -> above; NCE 1 alone -> [1, +inf];
    # neither -> [-inf, +inf]
    lone1 = gap[1] & (xa[1] == 1.0)
    nowhere = gap[1] & (xa[1] == -inf)
    assert (gap[1] == (lone1 | nowhere)).all() and (above[1] | gap[1]).all()
    assert set(map(tuple, br[:, 1, lone1].T)) == {(1.0, inf, 4.2, inf)}
    assert set(map(tuple, br[:, 1, nowhere].T)) == {(-inf, inf, -inf, inf)}
    assert (lone1.mean(), nowhere.mean()) == (pytest.approx(3 / 16, abs=0.02),
                                              pytest.approx(1 / 16, abs=0.02))
    # W: NCE 1 missed (1/4) -> NCE 10 alone at 0.1 -> [-inf, 10]
    assert set(map(tuple, br[:, 2, gap[2]].T)) == {(-inf, 10.0, -inf, 6.0)}
    assert (gap[2] | inter[2]).all() and gap[2].mean() == pytest.approx(1 / 4, abs=0.02)
    # one draw for all three: W misses NCE 1 exactly where Z is nowhere or
    # at NCE 10 alone, and where X reads 0.2 there (below, never crossing)
    assert not (gap[2] & lone1).any() and (nowhere <= gap[2]).all()
    assert not (gap[2] & (inter[0] | gap[0])).any()
    for i in range(3):
        r = RP.interval(out["f"][0][i], out["f"][1][i], code[i],
                        bracket=tuple(a[i] for a in out["f"][3]))
        assert r["none"] == 0.0
        assert r["cross"] + r["below"] + r["above"] + r["gap"] == pytest.approx(1.0)
    # an averaged curve of one energy: its resamples have no answer ('none')
    one = RP.bootstrap_e50([1.0, 10.0], [4.2, 6.0],
                           [(per[0][0][:1], per[0][1][:1]), (per[1][0][:1] * 0, per[1][1][:1] * 0)],
                           w[:1, :1], w[:1, :1], b=200, seed=7)
    assert (one["f"][2] == fi(BD.FLAG_SINGLE)).all()


def test_a_crossing_next_to_an_energy_of_one_scan_is_said():
    """The band holds the scatter of the scans it can draw. A crossing
    between two energies of one scan each: every resample IS the averaged
    curve there -- a band of zero width, no measurement. The row says how
    many of the two energies about the averaged crossing hold fewer than two
    scans (`one_scan_bracket`), names them, and flags the zero width; with
    one of them, the band holds the other's scatter only, said. F exactly 1
    in every scan at the energy before the crossing (no fragment read
    there) and one scan after it: zero width too."""
    p = round(_mz(X), 4)
    base = _three_steps()
    res = AN.analyze_run(_replicated(p, [base[0], (10, base[1][1][:1]), (20, base[2][1][:1])]),
                         _ions(X))
    r = _e50(res, p, X)
    assert r["e50_nce"] == pytest.approx(12.5) and r["e50_nce_lo"] == r["e50_nce_hi"] == r["e50_nce"]
    assert (r["one_scan_bracket"], r["boot_n"]) == (2, RP.BOOTSTRAP_B)
    assert ("E50 band: NCE 10 (1 scan) and 20 (1 scan) about the crossing add no scatter to it"
            in r["note"]) and "the band has zero width" in r["note"]
    res = AN.analyze_run(_replicated(p, [base[0], base[1], (20, base[2][1][:1])]), _ions(X))
    r = _e50(res, p, X)
    assert r["one_scan_bracket"] == 1 and r["e50_nce_lo"] < r["e50_nce"] < r["e50_nce_hi"]
    assert "E50 band: NCE 20 (1 scan) about the crossing adds no scatter to it" in r["note"]
    assert "zero width" not in r["note"]
    res = AN.analyze_run(_replicated(p, base), _ions(X))
    assert (res.e50["one_scan_bracket"] == 0).all()
    assert not res.e50["note"].str.contains("E50 band").any()
    flat = [(1, [{X: 1000.0}, {X: 1100.0}, {X: 900.0}]), (10, [{X: 600.0}, {X: 640.0}, {X: 560.0}]),
            (20, [{X: 230.0, FRAG: 760.0}])]
    r = _e50(AN.analyze_run(_replicated(p, flat), _ions(X)), p, X)
    assert r["e50_nce"] == pytest.approx(16.5132, abs=1e-4)
    assert r["e50_nce_lo"] == r["e50_nce_hi"] == r["e50_nce"] and r["one_scan_bracket"] == 1
    assert "NCE 20 (1 scan)" in r["note"] and "the band has zero width" in r["note"]


def _ref_steps(first):
    """F and S cross between NCE 10 and 20 (E50 16.667 both: the precursor
    falls 1000 -> 700 -> 400, its fragment takes the rest); NCE 1, the
    energy S is read against, holds the scans `first`."""
    return [(1, first),
            (10, [{X: 700.0, FRAG: 300.0}, {X: 760.0, FRAG: 280.0}, {X: 640.0, FRAG: 320.0}]),
            (20, [{X: 400.0, FRAG: 600.0}, {X: 440.0, FRAG: 560.0}, {X: 360.0, FRAG: 640.0}])]


def test_an_s_band_on_a_reference_without_spread_is_said():
    """S is read against each resample's own lowest energy, so its band rests
    on that energy's scatter as well as on the two about the crossing. A
    lowest energy of one scan -- or whose precursor is held, or with no
    per-scan row -- puts the same reference into every resample: the S band
    holds none of its scatter (too narrow), said on the S row and counted
    in its `one_scan_bracket`; F and F_lo, ratios within one spectrum, do
    not rest on it. A reference that is itself one of the two energies
    about the crossing is counted once. Whether S has a spread at an energy
    is its precursor's (`intensity_cv`), not the per-scan S's: a precursor
    read in one of three scans, the others empty, gives one finite S per
    scan, yet the resamples' means -- empty scans 0 -- scatter."""
    p = round(_mz(X), 4)
    three = [{X: 1000.0, FRAG: 20.0}, {X: 1100.0, FRAG: 15.0}, {X: 900.0, FRAG: 25.0}]
    held = _replicated(p, _ref_steps(three))
    held.scan_peaks.loc[(held.scan_peaks["spectrum"] == _spec(held, 1.0))
                        & np.isclose(held.scan_peaks["peak_mz"], _mz(X)), "intensity"] = 0.0
    none = _replicated(p, _ref_steps(three))
    none.scan_peaks = none.scan_peaks[none.scan_peaks["spectrum"]
                                      != _spec(none, 1.0)].reset_index(drop=True)
    for t, words in ((_replicated(p, _ref_steps(three[:1])), "1 (1 scan)"),
                     (held, "1 (3 scans, no spread)"), (none, "1 (0 scans)")):
        res = AN.analyze_run(t, _ions(X))
        s, f = _e50(res, p, X, "S"), _e50(res, p, X)
        assert s["e50_nce"] == pytest.approx(16.6667, abs=1e-4) and s["boot_n"] == RP.BOOTSTRAP_B
        assert s["one_scan_bracket"] == 1, words
        assert (f"E50 band: NCE {words}, the lowest energy S is read against, adds no scatter "
                "to it (too narrow)") in s["note"], s["note"]
        for c in ("F", "F_lo"):
            r = _e50(res, p, X, c)
            assert r["one_scan_bracket"] == 0 and "E50 band" not in r["note"], c
        assert f["e50_nce_lo"] < f["e50_nce"] < f["e50_nce_hi"]
    # three scans read at NCE 1: no caveat on any row
    res = AN.analyze_run(_replicated(p, _ref_steps(three)), _ions(X))
    assert (res.e50["one_scan_bracket"] == 0).all() and not res.e50["note"].str.contains(
        "E50 band").any()
    # the precursor read in one of the three NCE-1 scans (the same 1000 on
    # average), nothing of X in the other two: S's per-scan sd is NaN there
    # (one scan observes it), yet every resample's reference is the mean of
    # its draws, the empty scans 0 -- 0 to 3000: the reference scatters, and
    # no note calls the S band too narrow. A resample that draws only the
    # empty scans has no reference, no S anywhere: a hole about every energy
    # ((2/3)^3 of them), so neither bound is known.
    one_of_3 = _replicated(p, _ref_steps([{X: 3000.0, FRAG: 60.0}, {}, {}]))
    res = AN.analyze_run(one_of_3, _ions(X))
    b1 = res.breakdown.set_index("nce").loc[1.0]
    assert math.isnan(b1["s_sd"]) and b1["intensity_cv"] == pytest.approx(math.sqrt(3.0))
    s = _e50(res, p, X, "S")
    assert s["one_scan_bracket"] == 0 and "E50 band" not in s["note"], s["note"]
    assert (s["e50_lo_censor"], s["e50_hi_censor"]) == ("gap", "gap")
    assert s["boot_cross"] == pytest.approx(1 - (2 / 3) ** 3, abs=0.03)
    # the same at an energy about the crossing: X read in one of the three
    # NCE-20 scans -- S there (its mean precursor over the reference) moves
    # with the draws; F there does not (a ratio within the scan that holds
    # X: 0.4 in every resample that observes it), and F's band says so
    steps = _ref_steps(three)
    steps[2] = (20, [{X: 1200.0, FRAG: 1800.0}, {}, {}])
    res = AN.analyze_run(_replicated(p, steps), _ions(X))
    s, f = _e50(res, p, X, "S"), _e50(res, p, X)
    assert s["one_scan_bracket"] == 0 and "E50 band" not in s["note"], s["note"]
    assert f["one_scan_bracket"] == 1
    assert "E50 band: NCE 20 (3 scans, no spread) about the crossing" in f["note"]
    # the reference about the crossing (S 0.45 at NCE 10): one energy, named once
    steps = [(1, three[:1]), (10, [{X: 450.0, FRAG: 550.0}, {X: 500.0, FRAG: 500.0},
                                   {X: 400.0, FRAG: 600.0}]), (20, _ref_steps(three)[2][1])]
    s = _e50(AN.analyze_run(_replicated(p, steps), _ions(X)), p, X, "S")
    assert s["one_scan_bracket"] == 1 and s["note"].count("NCE 1 (1 scan)") == 1
    assert "about the crossing adds no scatter" in s["note"] and "read against" not in s["note"]


def _spec(t, nce):
    """The spectrum key of a synthetic folder's energy."""
    par = t.parents
    return str(par.loc[np.isclose(par["nce"], nce), "spectrum"].iloc[0])


def test_a_peak_no_scan_reads_is_held_at_its_average():
    """A fragment whose per-scan rows all read 0 (its averaged spectrum holds
    it at 400): held at 400 in every scan -- `matched` False, no scan holds
    it (n_present 0), presence / sd / CV NaN, the window note -- so F moves
    from scan to scan with the precursor alone, and F of the means is still
    the averaged F."""
    p = round(_mz(X), 4)
    t = _replicated(p, _three_steps())
    sp = t.scan_peaks
    sp.loc[(sp["spectrum"] == _spec(t, 10.0)) & np.isclose(sp["peak_mz"], _mz(FRAG)),
           "intensity"] = 0.0
    res = AN.analyze_run(t, _ions(X))
    rep = res.replicates
    r = rep[(rep["nce"] == 10.0) & (rep["item"] == "fragment")].iloc[0]
    assert (r["name"], r["n_scans"], r["n_present"], r["matched"]) == (FRAG, 3, 0, False)
    assert r["mean"] == pytest.approx(400.0)
    assert math.isnan(r["presence"]) and math.isnan(r["sd"]) and math.isnan(r["cv"])
    pre = rep[(rep["nce"] == 10.0) & (rep["item"] == "precursor")].iloc[0]
    assert (pre["matched"], pre["n_present"], pre["presence"]) == (True, 3, 1.0)
    b = res.breakdown.set_index("nce").loc[10.0]
    i = np.array([600.0, 640.0, 560.0])
    assert b["f"] == pytest.approx(0.6, rel=1e-12)
    assert b["f_sd"] == pytest.approx((i / (i + 400.0)).std(ddof=1), rel=1e-12)
    assert any("1 precursor / fragment peak(s) are read in no scan" in n
               for n in res.windows[0].notes)
    # the unit: a peak no row matches is held, and in no scan
    sc = RP.spectrum_scans("s", [100.0, 200.0], [10.0, 20.0],
                           pd.DataFrame({"peak_mz": [100.0, 100.0], "scan": [1, 2],
                                         "intensity": [8.0, 12.0]}))
    assert list(sc.matched) == [True, False] and sc.present.tolist() == [[True, True], [False, False]]
    np.testing.assert_allclose(sc.matrix, [[8.0, 12.0], [20.0, 20.0]])


@pytest.mark.parametrize("item,ion", [("fragment", FRAG), ("precursor", X)])
def test_an_item_of_two_peaks_with_one_held_is_unmatched_in_every_column(item, ion):
    """An item read over two peaks (a split peak 1.5 ppm above the first,
    read as the same formula and summed with it) of which one is held -- read
    in no scan at NCE 10 -- is unmatched: `matched` False, and then
    `n_present` 0 as presence, sd and CV are NaN, though its other peak is
    read in every scan (not 3 of 3 beside a NaN presence)."""
    p = round(_mz(X), 4)
    split = _mz(ion) * (1 + 1.5e-6)
    steps = [(nce, [{**s, split: 0.25 * s[ion]} for s in scans]) for nce, scans in _three_steps()]
    t = _replicated(p, steps)
    sp = t.scan_peaks
    sp.loc[(sp["spectrum"] == _spec(t, 10.0))
           & np.isclose(sp["peak_mz"], split, rtol=0.0, atol=1e-7), "intensity"] = 0.0
    res = AN.analyze_run(t, _ions(X))
    bd = res.breakdown.set_index("nce")
    if item == "precursor":
        assert (bd["n_precursor_peaks"] == 2).all()
    else:
        assert (res.fragments["fragment"] == FRAG).sum() == 1
    rep = res.replicates
    r = rep[(rep["nce"] == 10.0) & (rep["item"] == item)].iloc[0]
    assert (r["name"], r["n_scans"], r["matched"]) == (ion, 3, False)
    assert r["mean"] == pytest.approx(1.25 * (600.0 if item == "precursor" else 400.0))
    assert r["n_present"] == 0
    assert math.isnan(r["presence"]) and math.isnan(r["sd"]) and math.isnan(r["cv"])
    # at the other energies both peaks are read: matched, in every scan
    other = rep[(rep["nce"] != 10.0) & (rep["item"] == item)]
    assert other["matched"].all() and (other["n_present"] == 3).all()
    assert any("1 precursor / fragment peak(s) are read in no scan" in n
               for n in res.windows[0].notes)


def test_a_spectrum_with_no_per_scan_row():
    """A spectrum none of whose scans is in the per-scan table: held at its
    averaged spectrum (no spread there), said. It is an energy that adds no
    spread, counted with those of one scan (`n_one_scan`, the manifest's
    `one_scan_energies`) -- but not named by the one-scan note."""
    p = round(_mz(X), 4)
    t = _replicated(p, _three_steps())
    t.scan_peaks = t.scan_peaks[t.scan_peaks["spectrum"] != _spec(t, 10.0)].reset_index(drop=True)
    res = AN.analyze_run(t, _ions(X))
    (w,) = res.windows
    assert w.scans_per_energy == [3, 0, 3] and w.one_scan_energies == 1
    bd = res.breakdown.set_index("nce")
    assert bd.loc[10.0, "n_scans"] == 0 and math.isnan(bd.loc[10.0, "f_sd"])
    assert bd.loc[10.0, "f"] == pytest.approx(0.6)
    rep = res.replicates[res.replicates["nce"] == 10.0]
    assert (rep["n_scans"] == 0).all() and (rep["n_present"] == 0).all()
    assert not rep["matched"].any()
    assert any("no per-scan row for the spectra at NCE 10" in n for n in w.notes)
    assert not any("hold one scan" in n for n in w.notes)
    assert (res.e50["n_one_scan"] == 1).all() and (res.e50["boot_n"] == RP.BOOTSTRAP_B).all()
    man = AN.manifest(res, t, files={}, created_utc=None)
    assert man["counts"]["one_scan_energies"] == 1
    assert (man["windows"][0]["scans_per_energy"], man["windows"][0]["one_scan_energies"]) == \
        ([3, 0, 3], 1)


def test_a_scan_table_naming_other_scans_is_said():
    p = round(_mz(X), 4)
    t = _replicated(p, _three_steps())
    t.scans = pd.DataFrame({"spectrum": [_spec(t, nce) for nce, n in ((1.0, 3), (10.0, 4),
                                                                      (20.0, 3))
                                         for _ in range(n)]})
    res = AN.analyze_run(t, _ions(X))
    assert any("the per-scan peaks and the scan table name different scans for the spectra at "
               "NCE 10 (3 read, 4 in ms2_scans.csv)" in n for n in res.windows[0].notes)


def test_an_empty_scan_is_no_point_of_the_curves():
    """A scan holding neither the precursor nor any fragment it can make:
    0 / 0 there -- F, F_lo and S NaN in it, as in `survival` -- so the
    spread runs over the three scans that observe the curves (sd ddof 1,
    sem sd / sqrt 3), not over four."""
    p = round(_mz(X), 4)
    steps = _three_steps()
    steps[1] = (10, steps[1][1] + [{}])
    b = AN.analyze_run(_replicated(p, steps), _ions(X)).breakdown.set_index("nce").loc[10.0]
    i, fr = np.array([600.0, 640.0, 560.0]), np.array([400.0, 380.0, 420.0])
    f, s = i / (i + fr), i / 1000.0
    assert b["n_scans"] == 4
    assert (b["f_sd"], b["f_sem"]) == (pytest.approx(f.std(ddof=1), rel=1e-12),
                                       pytest.approx(f.std(ddof=1) / math.sqrt(3), rel=1e-12))
    assert (b["s_sd"], b["s_sem"]) == (pytest.approx(s.std(ddof=1), rel=1e-12),
                                       pytest.approx(s.std(ddof=1) / math.sqrt(3), rel=1e-12))


def test_volts_bounds_need_every_crossing_s_volts():
    """F 0.507 at NCE 10 from scattered scans: some resamples cross between
    NCE 1 and 10, the others between 10 and 20 -- where the volts are
    unknown. The volts band is then NaN (not the band of the crossings that
    have volts), the NCE band measured."""
    p = round(_mz(X), 4)
    t = _replicated(p, [(1, [{X: 1000.0, FRAG: 10.0}] * 3),
                        (10, [{X: 570.0, FRAG: 430.0}, {X: 440.0, FRAG: 560.0},
                              {X: 510.0, FRAG: 490.0}]),
                        (20, [{X: 200.0, FRAG: 800.0}] * 3)])
    t.parents.loc[np.isclose(t.parents["nce"], 20.0), "hcd_v"] = np.nan
    r = _e50(AN.analyze_run(t, _ions(X)), p, X)
    assert (r["e50_nce_lo"], r["e50_nce_hi"]) == (pytest.approx(9.704, abs=1e-3),
                                                  pytest.approx(11.4286, abs=1e-4))
    assert math.isnan(r["e50_v_lo"]) and math.isnan(r["e50_v_hi"])
    assert math.isnan(r["e50_ecm_lo"]) and math.isnan(r["e50_ecm_hi"])


def test_a_curve_without_an_e50_some_resamples_cross():
    """The averaged F stays above 0.5 (0.52 at NCE 20); a resample of two or
    three of the 450 scan crosses (7 / 27): the note says so, and the bound
    among the resamples that stay above is NaN on that side."""
    p = round(_mz(X), 4)
    steps = [(1, [{X: 1000.0, FRAG: 10.0}] * 3), (10, [{X: 700.0, FRAG: 300.0}] * 3),
             (20, [{X: 450.0, FRAG: 550.0}, {X: 550.0, FRAG: 450.0}, {X: 560.0, FRAG: 440.0}])]
    r = _e50(AN.analyze_run(_replicated(p, steps), _ions(X)), p, X)
    assert r["flag"] == BD.FLAG_ABOVE and r["boot_cross"] == pytest.approx(7 / 27, abs=0.03)
    assert r["note"].startswith("bootstrap (1000): crosses in 25.1% of resamples")
    assert math.isnan(r["e50_nce_hi"]) and r["e50_hi_censor"] == "above"
    assert "the interval's upper bound lies above the highest energy" in r["note"]


def test_the_bootstrap_note_counts_the_resamples_across_a_hole():
    """`_boot_note` where the resamples cross only across an unobserved
    energy: a curve whose averaged reading stays above 0.5 (no E50) while 30
    % of its resamples cross across a hole -- they disagree with it, said,
    though none crosses by interpolation; a curve whose averaged reading
    itself crosses across the hole ('gap') and every resample with it -- they
    agree, no note."""
    above = BD.e50([1.0, 10.0], [0.9, 0.8])
    gap = BD.e50([1.0, 10.0, 20.0], [0.9, np.nan, 0.2])
    assert (above.flag, gap.flag) == (BD.FLAG_ABOVE, BD.FLAG_GAP)
    b = {"cross": 0.0, "below": 0.0, "above": 0.7, "gap": 0.3, "none": 0.0,
         "lo_where": "gap", "hi_where": "above"}
    assert AN._boot_note(b, 1000, above) == (
        "bootstrap (1000): crosses in 0.0% of resamples (below the lowest energy in 0.0%, "
        "above the highest in 70.0%, across an unobserved energy in 30.0%); the interval's "
        "lower bound lies across an unobserved energy; the interval's upper bound lies above "
        "the highest energy")
    b = {"cross": 0.0, "below": 0.0, "above": 0.0, "gap": 1.0, "none": 0.0,
         "lo_where": "gap", "hi_where": "gap"}
    assert AN._boot_note(b, 1000, gap) == ""


def test_f_lo_s_spread_where_f_lo_is_not_f():
    """Two window ions share one fragment: its share weights split it (w <
    1), while F_lo gives it whole to each composition candidate (c = 1). Per
    scan F_lo_j = I_j / (I_j + frag_j) is then not F_j = I_j / (I_j + w
    frag_j), and `f_lo_sd` / `f_lo_sem` are its own spread (ddof 1, sem = sd
    / sqrt n), not F's."""
    a, b = "C10H19O+", "C10H20^N+"
    steps = [(1, [{a: 1000.0, b: 500.0, FRAG: 10.0}, {a: 1100.0, b: 450.0, FRAG: 30.0},
                  {a: 900.0, b: 520.0, FRAG: 20.0}]),
             (10, [{a: 600.0, b: 300.0, FRAG: 400.0}, {a: 640.0, b: 330.0, FRAG: 380.0},
                   {a: 560.0, b: 280.0, FRAG: 420.0}]),
             (20, [{a: 200.0, b: 100.0, FRAG: 800.0}, {a: 260.0, b: 120.0, FRAG: 700.0},
                   {a: 230.0, b: 90.0, FRAG: 760.0}])]
    res = AN.analyze_run(_replicated(round(_mz(a), 4), steps), _ions(a, b))
    fr = res.fragments.set_index("fragment").loc[FRAG]
    w = {k: float(x) for k, x in (kv.split("=") for kv in fr["weights"].split(";"))}
    assert set(w) == {a, b} and 0.0 < w[a] < 1.0 and 0.0 < w[b] < 1.0
    bd = res.breakdown.set_index(["nce", "ion"])
    for nce, scans in steps:
        frag = np.array([s[FRAG] for s in scans])
        for ion in (a, b):
            i = np.array([s[ion] for s in scans])
            f, f_lo = i / (i + w[ion] * frag), i / (i + frag)
            r = bd.loc[(float(nce), ion)]
            assert r["f_lo_sd"] == pytest.approx(f_lo.std(ddof=1), rel=1e-12)
            assert r["f_lo_sem"] == pytest.approx(f_lo.std(ddof=1) / math.sqrt(3), rel=1e-12)
            assert r["f_sd"] == pytest.approx(f.std(ddof=1), rel=1e-5)
            assert abs(r["f_lo_sd"] - r["f_sd"]) > 1e-4


def test_the_bootstrap_holds_no_array_of_every_scan_of_every_resample():
    """Each resample's means come from its scans' counts (resamples x scans)
    times the scans' intensities -- the same draws as indexing every scan of
    every resample, and the same means, without a fragments x resamples x
    scans array: 200 scans of 50 fragments, 1000 resamples, would hold 80 MB
    at once."""
    rng = np.random.default_rng(0)
    n, k, b = 200, 50, 1000
    per = [(rng.uniform(500.0, 1000.0, (1, n)), rng.uniform(0.0, 600.0 * e, (k, n)))
           for e in (0.1, 1.0, 3.0)]
    w, c = np.full((k, 1), 1.0 / k), np.ones((k, 1))
    tracemalloc.start()
    try:
        out = RP.bootstrap_e50([1.0, 10.0, 20.0], [5.0, 6.0, 7.0], per, w, c, b=b, seed=1)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak < 16e6
    # the same draws, means and E50s as indexing every scan of every resample
    g = np.random.default_rng(1)
    means = []
    for ion_m, frag_m in per:
        idx = g.integers(0, n, size=(b, n))
        means.append((ion_m[:, idx].mean(axis=2), frag_m[:, idx].mean(axis=2)))
    f = np.stack([im / (im + w.T @ fm) for im, fm in means])[:, 0, :]
    e, _, code = BD.e50_many([1.0, 10.0, 20.0], f, [5.0, 6.0, 7.0])
    np.testing.assert_allclose(out["f"][0][0], e, rtol=1e-12)
    np.testing.assert_array_equal(out["f"][2][0], code)


# ------------------------------------------------ own data: standard A ----
def test_standard_a_per_scan_e50_and_its_interval(std_a_res):
    """[M+H]+ C10H17+ of standard A, 5-9 scans per energy: F E50 NCE 13.891
    as without the scans (the server's spectra), and its 16-84 % band
    13.843-13.941 NCE (6.319-6.336 V, E_cm 1.0716-1.0745 eV on N2) over
    1000 resamples, every one of which crosses. The precursor's CV per
    spectrum is 1.3-4.3 %; F scatters by 0.002-0.004 from scan to scan."""
    plain, per = std_a_res
    a, b = _e50(plain, 137.1321, "C10H17+"), _e50(per, 137.1321, "C10H17+")
    assert a["e50_nce"] == b["e50_nce"] == pytest.approx(13.891, abs=1e-3)
    assert (b["e50_nce_lo"], b["e50_nce_hi"]) == (pytest.approx(13.8428, abs=1e-4),
                                                  pytest.approx(13.9412, abs=1e-4))
    assert (b["e50_v_lo"], b["e50_v_hi"]) == (pytest.approx(6.3185, abs=1e-4),
                                              pytest.approx(6.3359, abs=1e-4))
    assert (b["e50_ecm_lo"], b["e50_ecm_hi"]) == (pytest.approx(1.0716, abs=1e-4),
                                                  pytest.approx(1.0745, abs=1e-4))
    assert (b["boot_n"], b["boot_cross"], b["n_one_scan"]) == (1000, 1.0, 0)
    assert b["e50_nce_lo"] < b["e50_nce"] < b["e50_nce_hi"] and b["note"] == ""
    assert math.isnan(a["e50_nce_lo"]) and a["boot_n"] == 0
    w = next(w for w in per.windows if np.isclose(w.parent_mz, 137.1321))
    assert w.scans_per_energy == [7, 5, 9, 6, 6]
    g = per.breakdown[np.isclose(per.breakdown["parent_mz"], 137.1321)]
    assert g["intensity_cv"].between(0.013, 0.044).all()
    assert g["f_sd"].between(0.002, 0.0041).all()
    # the S curve crosses at its last step only just: some resamples do not
    s = _e50(per, 137.1321, "C10H17+", "S")
    assert s["boot_cross"] == pytest.approx(0.809, abs=1e-3) and math.isnan(s["e50_nce_hi"])


def test_the_scans_change_no_existing_column(std_a_res):
    """The per-scan tables add columns, and change nothing that was there:
    every table of the analysis without them, column by column."""
    plain, per = std_a_res
    pd.testing.assert_frame_equal(plain.readings, per.readings)
    pd.testing.assert_frame_equal(_old(plain.breakdown, NEW_BREAKDOWN),
                                  _old(per.breakdown, NEW_BREAKDOWN))
    pd.testing.assert_frame_equal(_old(plain.e50, NEW_E50 + ("note",)),
                                  _old(per.e50, NEW_E50 + ("note",)))
    pd.testing.assert_frame_equal(_old(plain.fragments, NEW_FRAGMENTS),
                                  _old(per.fragments, NEW_FRAGMENTS))
    pd.testing.assert_frame_equal(plain.branching, per.branching)
    # notes only where the resamples disagree with the averaged curve
    changed = per.e50[per.e50["note"] != plain.e50["note"]]
    assert set(changed["curve"]) == {"S"} and changed["note"].str.contains("bootstrap").all()


def test_the_averaged_f_is_f_of_the_per_scan_means(std_a_res):
    """Every item's mean over its scans (`ms2_replicates.csv`) is the
    averaged spectrum's intensity, and F rebuilt from those means with the
    analysis's weights (as `ms2_fragments.csv` prints them, 6 digits) is the
    analysis's F. The server's heights run ~2 % below the per-scan means
    (the test below), so each peak's per-scan readings are put on its
    averaged height's scale first. (Exact to the last digit: the synthetic test above.)"""
    _, per = std_a_res
    rep, bd, fr = per.replicates, per.breakdown, per.fragments
    for (p, spec), g in rep.groupby(["parent_mz", "spectrum"]):
        b = bd[(bd["parent_mz"] == p) & (bd["spectrum"] == spec)].set_index("ion")
        pre = g[g["item"] == "precursor"].set_index("name")
        frags = g[g["item"] == "fragment"].set_index("name")["mean"]
        for ion in b.index:
            assert pre.loc[ion, "mean"] == pytest.approx(b.loc[ion, "intensity"], rel=1e-12)
            w = {f: dict(kv.split("=") for kv in ws.split(";")).get(ion, "0")
                 for f, ws in zip(fr.loc[fr["parent_mz"] == p, "fragment"],
                                  fr.loc[fr["parent_mz"] == p, "weights"])}
            fw = sum(float(w.get(f, 0)) * m for f, m in frags.items())
            i = pre.loc[ion, "mean"]
            if i + fw > 0:
                assert i / (i + fw) == pytest.approx(b.loc[ion, "f"], abs=1e-6)
    assert rep["matched"].all() and (rep["n_scans"] >= 4).all()


def test_the_servers_heights_run_below_the_per_scan_means(std_a):
    """On a `--per-scan` folder each peak's per-scan readings are scaled by
    I_avg / their mean (`spectrum_scans`). The server's height runs ~2 %
    BELOW the per-scan mean on the median peak (the per-scan mean is
    `scans.SERVER_HEIGHT_RATIO` x it), so the factor is under 1 for nearly
    every peak: 0.89-1.01 on standard A's S/N-10 spectra."""
    _, per = std_a
    ratio = []
    for key, g in per.spectra.groupby("spectrum"):
        rows = per.scan_peaks[per.scan_peaks["spectrum"] == key]
        mean = rows.groupby("peak_mz")["intensity"].mean().reindex(g["mz"].to_numpy()).to_numpy()
        ok = mean > 0
        ratio.extend(g["intensity"].to_numpy()[ok] / mean[ok])
    r = np.array(ratio)
    assert np.median(r) == pytest.approx(1 / SC.SERVER_HEIGHT_RATIO, abs=0.005)
    assert (r < 1).mean() > 0.95 and 0.88 < r.min() and r.max() < 1.02


def test_a_file_folders_spectra_are_its_per_scan_means():
    """On an `ms2 extract --file` folder each spectrum IS the mean of its
    scans (absence 0): the scale factor every peak's readings take is 1, and
    `ms2 analyze` reads the folder's own E50 (13.821, PR #81) with an
    interval about it."""
    meta, times = _load("ms2_monoterpene_a_metadata.json"), _load("ms2_monoterpene_a_times.json")
    payload, lists, _, _ = wrap_tms2(meta, times, _load("ms2_monoterpene_a_summary.json"))
    res = S.extract_file(None, FT, metadata=payload, inclusion_lists=lists, polarity="+")
    for key, g in res.spectra.groupby("spectrum"):
        rows = res.scan_peaks[res.scan_peaks["spectrum"] == key]
        sc = RP.spectrum_scans(key, g["mz"], g["intensity"], rows)
        raw = rows.pivot_table(index="peak_mz", columns="scan", values="intensity").reindex(
            g["mz"].to_numpy()).to_numpy()
        assert sc.matched.all()
        np.testing.assert_allclose(raw.mean(axis=1), g["intensity"].to_numpy(), rtol=1e-12)
        np.testing.assert_allclose(sc.matrix, raw, rtol=1e-12)
    tables = AN.RunTables(spectra=res.spectra, parents=res.parents, window=res.window,
                          manifest={"sample_id": "S",
                                    "parameters": {"isolation_width_used": WIDTH_A}},
                          scans=res.scans, scan_peaks=res.scan_peaks)
    out = AN.analyze_run(tables, _ions(*MT_IONS))
    r = _e50(out, 137.1321, "C10H17+")
    assert r["e50_nce"] == pytest.approx(13.821, abs=1e-3)
    assert r["e50_nce_lo"] < r["e50_nce"] < r["e50_nce_hi"]
    assert r["e50_nce_hi"] - r["e50_nce_lo"] < 0.2


_SEED_SCRIPT = """
import json, math
from tests.test_ms2_replicates import _load, MT_IONS, _ions, WIDTH_A
from peaky.ms2 import analyze as AN, scans as SC, sources as S
meta, times = _load("ms2_monoterpene_a_metadata.json"), _load("ms2_monoterpene_a_times.json")
cen, summ = _load("ms2_monoterpene_a_centroids.json"), _load("ms2_monoterpene_a_summary.json")
sp, par = S.spectra_table("S", cen, summ), S.parents_table("S", cen, summ)
st, _ = SC.assign_from_times(SC.scan_table(meta), times)
t = AN.RunTables(spectra=sp, parents=par, window=S._frame({}, S.WINDOW_COLUMNS), manifest={},
                 scans=st, scan_peaks=SC.scan_peak_intensities(meta, st, sp))
e = AN.analyze_run(t, _ions(*MT_IONS)).e50
print(json.dumps([[round(v, 12) for v in r] for r in
                  e[["e50_nce_lo", "e50_nce_hi", "e50_v_lo", "boot_cross"]].fillna(-1).values]))
"""


def test_the_interval_is_deterministic(std_a, std_a_res):
    """The same interval on every run, under every string-hash seed (each
    seed its own process), and whatever other ions the window reads (the
    draws do not depend on them). That a window's interval does not depend
    on the other WINDOWS of the folder is
    `test_a_window_s_interval_is_its_own_whatever_else_the_folder_holds`."""
    _, per = std_a_res
    again = AN.analyze_run(std_a[1], _ions(*MT_IONS))
    pd.testing.assert_frame_equal(per.e50, again.e50)
    pd.testing.assert_frame_equal(per.breakdown, again.breakdown)
    alone = AN.analyze_run(std_a[1], _ions("C10H17+"))
    a = _e50(alone, 137.1321, "C10H17+")
    b = _e50(per, 137.1321, "C10H17+")
    assert (a["e50_nce_lo"], a["e50_nce_hi"]) == (b["e50_nce_lo"], b["e50_nce_hi"])
    root = Path(__file__).resolve().parents[1]
    outs = []
    for seed in ("0", "8"):
        env = dict(os.environ, PYTHONHASHSEED=seed,
                   PYTHONPATH=os.pathsep.join([str(root), os.environ.get("PYTHONPATH", "")]))
        p = subprocess.run([sys.executable, "-c", _SEED_SCRIPT], cwd=root, env=env,
                           capture_output=True, text=True, timeout=300)
        assert p.returncode == 0, p.stderr
        outs.append(json.loads(p.stdout.strip().splitlines()[-1]))
    assert outs[0] == outs[1]
    assert outs[0][0][0] == pytest.approx(b["e50_nce_lo"], abs=1e-9)


def _without_parent(t, parent):
    """A per-scan folder without one parent: its spectra, parents, window
    rows, scans and per-scan peaks dropped."""
    keep = ~np.isclose(t.parents["parent_mz"], parent)
    keys = set(t.parents.loc[keep, "spectrum"])
    win = t.window
    if "parent_mz" in win:
        win = win[~np.isclose(win["parent_mz"], parent)]
    return AN.RunTables(spectra=t.spectra[t.spectra["spectrum"].isin(keys)].reset_index(drop=True),
                        parents=t.parents[keep].reset_index(drop=True),
                        window=win.reset_index(drop=True), manifest=t.manifest,
                        scans=t.scans[t.scans["spectrum"].isin(keys)].reset_index(drop=True),
                        scan_peaks=t.scan_peaks[t.scan_peaks["spectrum"].isin(keys)]
                        .reset_index(drop=True))


def test_a_window_s_interval_is_its_own_whatever_else_the_folder_holds(std_a, std_a_res):
    """Not only the window the folder sorts first: the 155.1429 and 155.1559
    windows of standard A read the same bands -- every E50 row of theirs the
    same -- in the full folder and in the folder without the 137.1321
    window, which sorts before them. A stream shared by the windows (one
    generator for the run, or a seed by the window's place in it) would
    move them."""
    _, per = std_a_res
    part = AN.analyze_run(_without_parent(std_a[1], 137.1321), _ions(*MT_IONS))
    assert not np.isclose(part.e50["parent_mz"], 137.1321).any()
    for parent in (155.1429, 155.1559):
        a = per.e50[np.isclose(per.e50["parent_mz"], parent)].reset_index(drop=True)
        b = part.e50[np.isclose(part.e50["parent_mz"], parent)].reset_index(drop=True)
        assert len(a) and (a["boot_n"] == RP.BOOTSTRAP_B).all()
        pd.testing.assert_frame_equal(a, b)
    for parent, ion, lo, hi in ((155.1429, "C10H19O+", 6.76047, 6.86517),
                                (155.1559, "C10H20^N+", 8.55245, 8.65919)):
        r = _e50(part, parent, ion)
        assert (r["e50_nce_lo"], r["e50_nce_hi"]) == (pytest.approx(lo, abs=1e-5),
                                                      pytest.approx(hi, abs=1e-5))


# ------------------------------------- folders without per-scan tables ----
@pytest.mark.parametrize("stem,ions", [("ms2_monoterpene_a", MT_IONS),
                                       ("ms2_monoterpene_b", ("C10H17+",)),
                                       ("ms2_nh4_exploratory",
                                        ("C7H6NS+", "C4H10NS2+", "C11H13O+", "C11H15O2+",
                                         "C11H16^NO+", "C8H28^NO4Si4+"))])
def test_a_folder_without_per_scan_tables_reads_as_before(stem, ions):
    """No per-scan tables: no replicate table, every new column blank, the
    run says so -- and no peak of our own committed spectra is a harmonic
    ghost, so the rule changes nothing in them: with it and without it
    (`harmonics=False`, PR #80's reading) every table is the same."""
    from tests.test_ms2_breakdown import _tables
    t = _tables(stem)
    on, off = AN.analyze_run(t, _ions(*ions)), AN.analyze_run(t, _ions(*ions), harmonics=False)
    for name in ("readings", "breakdown", "e50", "fragments", "branching"):
        pd.testing.assert_frame_equal(getattr(on, name), getattr(off, name), obj=name)
    assert on.replicates is None and AN.NO_PER_SCAN_NOTE in on.notes
    assert (on.readings["harmonic"] == "").all()
    bd = on.breakdown
    assert (bd[["n_harmonic", "n_scans"]] == 0).all().all()
    assert (bd["harmonic_intensity"] == 0).all()
    spread = ["intensity_cv", "f_sd", "f_sem", "f_lo_sd", "f_lo_sem", "s_sd", "s_sem"]
    assert bd[spread].isna().all().all()
    e = on.e50
    assert (e["boot_n"] == 0).all() and (e["n_one_scan"] == 0).all()
    assert (e["one_scan_bracket"] == 0).all()
    assert (e["e50_lo_censor"] == "").all() and (e["e50_hi_censor"] == "").all()
    assert e[["e50_nce_lo", "e50_nce_hi", "e50_v_lo", "e50_v_hi", "e50_ecm_lo", "e50_ecm_hi",
              "boot_cross"]].isna().all().all()
    assert on.fragments[list(NEW_FRAGMENTS)].isna().all().all()
    assert not on.params["replicates"]["per_scan"] and on.params["harmonics"]["on"]


# ------------------------------------------------------- harmonic ghosts ----
def test_harmonic_ghosts_finds_a_lines_2f_and_3f_images():
    """2f at m/z / 4 within 40 ppm and <= 0.5 % of the line; 3f at m/z / 9
    and <= 8 %. Above the cap a peak is an ion; beyond the band, another
    peak; of two in one band the brighter is the ghost; a fragment line has
    its ghosts too."""
    P, F = 460.2500, 300.1000
    mz = [P / 9 * (1 + 15e-6), P / 4 * (1 + 12e-6), P / 4 * (1 - 45e-6), F / 4 * (1 + 5e-6),
          F / 4 * (1 + 20e-6), F, P]
    inten = [2.5e3, 100.0, 120.0, 30.0, 10.0, 5e4, 1e5]
    g = RD.harmonic_ghosts(mz, inten)
    assert sorted(g) == [0, 1, 3]
    assert g[0].startswith("3f ghost of the line at 460.2500 (m/z / 9): +15.0 ppm, 2.5 % of it")
    assert g[1].startswith("2f ghost of the line at 460.2500 (m/z / 4): +12.0 ppm, 0.1 % of it")
    assert g[3].startswith("2f ghost of the line at 300.1000")          # the brighter of two
    # above the cap: an ion
    assert RD.harmonic_ghosts([P / 4, P], [600.0, 1e5]) == {}
    assert RD.harmonic_ghosts([P / 9, P], [9e3, 1e5]) == {}
    assert RD.harmonic_ghosts([P / 9, P], [7.9e3, 1e5]) != {}
    # the band's edges, and nothing to test
    assert RD.harmonic_ghosts([P / 4 * (1 + 39.9e-6), P], [10.0, 1e5]) != {}
    assert RD.harmonic_ghosts([P / 4 * (1 + 40.1e-6), P], [10.0, 1e5]) == {}
    assert RD.harmonic_ghosts([], []) == {} and RD.harmonic_ghosts([P], [1.0]) == {}
    assert RD.harmonic_ghosts([P / 4, P], [10.0, NAN]) == {}
    # 4f is not read (never seen)
    assert RD.harmonic_ghosts([P / 16, P], [10.0, 1e5]) == {}
    # a peak that is the ghost of two lines (m/z / 4 of one, m/z / 9 of the
    # other) is named after the brighter
    assert RD.harmonic_ghosts([25.0, 100.0, 225.0], [10.0, 1e4, 1e5])[0].startswith(
        "3f ghost of the line at 225.0000")
    assert RD.harmonic_ghosts([25.0, 100.0, 225.0], [10.0, 1e5, 1e4])[0].startswith(
        "2f ghost of the line at 100.0000")
    # a place the window reads as an ion (`ions_at`) holds no ghost: of two
    # peaks in one band, the other is
    U, G = 61.0396, 549.3647 / 9 * (1 + 31e-6)
    assert list(RD.harmonic_ghosts([U, G, 549.3647], [5e3, 2.2e3, 1e5])) == [0]
    assert list(RD.harmonic_ghosts([U, G, 549.3647], [5e3, 2.2e3, 1e5], ions_at=[U])) == [1]


#: A synthetic [M+NH4]+ whose 2f ghost (m/z / 4, +34 ppm: inside the band)
#: lands 4.5 ppm from C4H7O+, a sub-formula it can make.
GHOST_ION = "C15H26NO4+"


def _ghost_steps(real=0.0):
    p = _mz(GHOST_ION)
    g = p / 4 * (1 + 34e-6)
    out = []
    for nce, i, a, b in ((1, 1e5, 0.0, 0.0), (10, 7e4, 3e4, 0.0), (20, 3e4, 6e4, 1e4)):
        peaks = {GHOST_ION: i, g: 0.003 * i}
        if a:
            peaks["C15H22O4+"] = a
        if b:
            peaks["C11H15O+"] = b
        if real:
            peaks[g] = real * i
        out.append((nce, peaks))
    return round(p, 4), out


def test_a_ghost_on_a_formula_is_never_read():
    """Read as a peak, the ghost is C4H7O+ (-4.5 ppm) at every energy: an
    'in-source fragment' of 0.3 %, and F 0.997 where nothing fragments.
    Read as a ghost: no formula, out of the TIC, F 1 -- and E50 15.0, the
    crossing of the products it really has. The same peak at 2 % of the line
    is above the cap: an ion, read."""
    p, steps = _ghost_steps()
    t = _synthetic(p, steps)
    on, off = AN.analyze_run(t, _ions(GHOST_ION)), AN.analyze_run(t, _ions(GHOST_ION),
                                                                 harmonics=False)
    assert "C4H7O+" in set(off.fragments["fragment"])
    assert "C4H7O+" not in set(on.fragments["fragment"])
    assert off.breakdown["f"].iloc[0] == pytest.approx(1 / 1.003)
    assert on.breakdown["f"].iloc[0] == 1.0
    assert _e50(on, p, GHOST_ION)["e50_nce"] == pytest.approx(15.0)
    rd = on.readings[on.readings["harmonic"] != ""]
    assert len(rd) == 3 and (rd["rank"] == 0).all() and (rd["formula"] == "").all()
    assert rd["harmonic"].str.startswith("2f ghost of the line at 284.1856 (m/z / 4): +34.0 ppm, "
                                         "0.3 % of it").all()
    bd = on.breakdown
    assert (bd["n_harmonic"] == 1).all()
    np.testing.assert_allclose(bd["tic"], off.breakdown["tic"] - bd["harmonic_intensity"])
    assert on.windows[0].harmonics == {"2f": 3}
    assert any(n.startswith("3 peak(s) are Orbitrap harmonic ghosts") for n in on.windows[0].notes)
    # out of the TIC and of the peak count, and explained by nothing
    assert list(bd["n_peaks"]) == list(off.breakdown["n_peaks"] - 1)
    np.testing.assert_allclose(bd["budget_explained"] * bd["tic"],
                               off.breakdown["budget_explained"] * off.breakdown["tic"]
                               - bd["harmonic_intensity"])
    # above the cap it is an ion
    p, steps = _ghost_steps(real=0.02)
    res = AN.analyze_run(_synthetic(p, steps), _ions(GHOST_ION))
    assert "C4H7O+" in set(res.fragments["fragment"]) and (res.readings["harmonic"] == "").all()


def test_a_ghost_two_window_ions_read_is_one_ghost_peak():
    """The ghost read by two window ions (the target and a co-isolate both
    read it as C4H7O+: two reading rows) becomes ONE row per spectrum, and
    every count -- the window's, its note, the manifest's -- counts peaks."""
    p, steps = _ghost_steps()
    co = "C16H30NO3+"
    for _, peaks in steps:
        peaks[co] = 5e3
    t = _synthetic(p, steps)
    ions = _ions(GHOST_ION, co)
    on, off = AN.analyze_run(t, ions), AN.analyze_run(t, ions, harmonics=False)
    g = _mz(GHOST_ION) / 4 * (1 + 34e-6)
    was = off.readings[np.isclose(off.readings["mz"], g, atol=1e-4)]
    assert (was.groupby("spectrum").size() == 2).all() and set(was["formula"]) == {"C4H7O+"}
    rd = on.readings[on.readings["harmonic"] != ""]
    assert (rd.groupby("spectrum").size() == 1).all() and len(rd) == 3
    assert on.windows[0].harmonics == {"2f": 3}
    assert any(n.startswith("3 peak(s) are Orbitrap harmonic ghosts") for n in on.windows[0].notes)
    man = AN.manifest(on, t, files={}, created_utc=None)
    assert (man["counts"]["harmonic_peaks"], man["counts"]["harmonic_by_order"]) == (3, {"2f": 3})


def test_a_fragments_place_weighs_its_peaks_by_the_tic_without_ghosts():
    """`fragment_mz` weighs each peak by its share of its spectrum's TIC --
    the TIC without the harmonic ghosts (the line seen again)."""
    roles = pd.DataFrame({
        "spectrum": ["a", "a", "a", "b", "b"],
        "role": ["precursor", "fragment", "harmonic", "precursor", "fragment"],
        "key": ["", "F", "", "", "F"], "mz_cal": [300.0, 100.000, 75.0, 300.0, 100.002],
        "intensity": [90.0, 10.0, 900.0, 90.0, 10.0]})
    assert AT.fragment_mz(roles)["F"] == pytest.approx(100.001, abs=1e-9)


#: A synthetic urea adduct [M + urea + H]+ (549.3647) whose main product,
#: protonated urea CH5N2O+ (61.0396), lies 14.4 ppm below m/z / 9 -- inside
#: the 3f band; [M + H]+ the other product (urea lost).
UREA_ADDUCT, UREA_H, UREA_LOSS = "C29H49N4O6+", "CH5N2O+", "C28H45N2O5+"


def _urea_steps(ghost):
    """The product holds 1 / 4 / 40 / 225 / 1500 % of the precursor at NCE
    1 / 3 / 6 / 10 / 15: under the 3f cap at the two lowest. `ghost`: the
    precursor's own 3f ghost too, +31 ppm of m/z / 9 at 2.2 % of it (a
    synthetic place and share: between the product's at NCE 1 and 3)."""
    p = _mz(UREA_ADDUCT)
    out = []
    for nce, ip, iu, im in ((1, 1e5, 1.0e3, 5e2), (3, 9e4, 3.6e3, 1.8e3), (6, 6e4, 2.4e4, 1.2e4),
                            (10, 2e4, 4.5e4, 2.2e4), (15, 4e3, 6e4, 3e4)):
        peaks = {UREA_ADDUCT: ip, UREA_H: iu, UREA_LOSS: im}
        if ghost:
            peaks[p / 9 * (1 + 31e-6)] = 0.022 * ip
        out.append((nce, peaks))
    return round(p, 4), out


@pytest.mark.parametrize("ghost", [False, True])
def test_a_product_in_a_harmonic_band_is_read_at_every_energy(ghost):
    """A product the window reads above the cap at some energy is an ion at
    that place at every energy: never half erased where it is still weak --
    its onset, in-source flag and spectra are those of the reading without
    the rule. Beside it, the precursor's true 3f ghost is the ghost at every
    energy (not the product where it is the brighter of the two)."""
    p, steps = _urea_steps(ghost)
    t = _synthetic(p, steps)
    on = AN.analyze_run(t, _ions(UREA_ADDUCT))
    off = AN.analyze_run(t, _ions(UREA_ADDUCT), harmonics=False)
    a = on.fragments.set_index("fragment").loc[UREA_H]
    b = off.fragments.set_index("fragment").loc[UREA_H]
    assert (a["onset_nce"], a["in_source"], a["n_spectra"]) == \
        (b["onset_nce"], b["in_source"], b["n_spectra"]) == (1.0, True, 5)
    rd = on.readings
    assert (rd.loc[np.isclose(rd["mz"], _mz(UREA_H), atol=1e-5), "harmonic"] == "").all()
    g = rd[rd["harmonic"] != ""]
    if not ghost:
        assert g.empty
        pd.testing.assert_frame_equal(on.breakdown, off.breakdown)
        return
    assert list(g["nce"]) == [1.0, 3.0, 6.0, 10.0, 15.0]
    assert np.allclose(g["mz"], _mz(UREA_ADDUCT) / 9 * (1 + 31e-6), atol=1e-6)
    assert g["harmonic"].str.startswith("3f ghost of the line at 549.3647 (m/z / 9): +31.0 "
                                        "ppm, 2.2 % of it").all()
    assert on.windows[0].harmonics == {"3f": 5}


def _same_reading(t, ion, frag, want, **kw):
    """The fragment's onset, in-source flag and spectra with the rule and
    without it (`kw`: `analyze_run`'s, the tolerance): both `want`; no
    harmonic row; F the same."""
    on = AN.analyze_run(t, _ions(ion), **kw)
    off = AN.analyze_run(t, _ions(ion), harmonics=False, **kw)
    a = on.fragments.set_index("fragment").loc[frag]
    b = off.fragments.set_index("fragment").loc[frag]
    assert (a["onset_nce"], a["in_source"], a["n_spectra"]) == \
        (b["onset_nce"], b["in_source"], b["n_spectra"]) == want
    assert (on.readings["harmonic"] == "").all()
    pd.testing.assert_series_equal(on.breakdown["f"], off.breakdown["f"])


@pytest.mark.parametrize("loss", [True, False])
def test_a_product_whose_line_is_gone_is_read_at_every_energy(loss):
    """A fragile urea adduct: at NCE 1 and 3 its protonated urea holds 1 %
    and 4 % of it (under the 3f cap, 14.4 ppm below m/z / 9), and from NCE
    6 on the adduct is gone -- its products are the spectrum. There no line's
    band holds the product: no ghost can be there, so its place is an ion's,
    and it is read at every energy, as without the rule. Without the [M+H]+
    product the spectra from NCE 6 on hold the protonated urea alone (one
    peak)."""
    p = _mz(UREA_ADDUCT)
    steps = []
    for nce, ip, iu, im in ((1, 1e5, 1.0e3, 5e2), (3, 9e4, 3.6e3, 1.8e3), (6, 0.0, 6e4, 3e4),
                            (10, 0.0, 6e4, 3e4), (15, 0.0, 5e4, 2.5e4)):
        peaks = {UREA_H: iu}
        if ip:
            peaks[UREA_ADDUCT] = ip
        if loss or ip:
            peaks[UREA_LOSS] = im
        steps.append((nce, peaks))
    _same_reading(_synthetic(round(p, 4), steps), UREA_ADDUCT, UREA_H, (1.0, True, 5))


def test_a_product_that_drifts_out_of_the_band_is_read_at_every_energy():
    """A product read 39.6 ppm above m/z / 4 of a line at NCE 1 (0.3 % of
    it: a 2f ghost, alone), at 40.4 and 40.6 ppm where it holds 3.75 % and 50
    % of the line (a ppm of jitter: just out of the band). Where no line's
    band holds it, it is no ghost: its place is an ion's at every energy --
    a spectrum of that one peak alone too."""
    P = 460.25
    spectra = [(np.array([P / 4 * (1 + 39.6e-6), P]), np.array([300.0, 1e5])),
               (np.array([P / 4 * (1 + 40.4e-6), P]), np.array([3e3, 8e4])),
               (np.array([P / 4 * (1 + 40.6e-6), P]), np.array([2e4, 4e4]))]
    assert RD.harmonic_ghosts(*spectra[0]) != {}
    ions_at = RD.harmonic_ions(spectra)
    assert all(RD.harmonic_ghosts(m, i, ions_at=ions_at) == {} for m, i in spectra)
    lone = [spectra[0], (np.array([P / 4 * (1 + 39.8e-6)]), np.array([2e4]))]
    assert RD.harmonic_ghosts(*lone[0], ions_at=RD.harmonic_ions(lone)) == {}


def test_a_product_s_place_holds_across_its_jitter_and_just_above_the_cap():
    """(1) A product's m/z moves between energies (0.06-0.96 ppm on our own
    179 window's lines): the urea product read 1.0 / -0.8 / 0.6 / -1.2 /
    0.4 ppm off its formula at NCE 1-15 is one place, within the reading
    tolerance -- read at every energy. (2) A product just above the 3f cap at
    ONE energy (8.5 % of the line at NCE 6; 5 % and 7 % at NCE 3 and 10) is
    an ion's place: read at all three. (3) `harmonic_ions` at the cap (a
    peak holding exactly the cap is under it) and in the bands of two lines
    (above one line's cap, under the other's: under)."""
    p = _mz(UREA_ADDUCT)
    steps = []
    for (nce, peaks), ppm in zip(_urea_steps(False)[1], (1.0, -0.8, 0.6, -1.2, 0.4)):
        steps.append((nce, {(_mz(k) * (1 + ppm * 1e-6) if k == UREA_H else k): v
                            for k, v in peaks.items()}))
    _same_reading(_synthetic(round(p, 4), steps), UREA_ADDUCT, UREA_H, (1.0, True, 5))
    steps = [(3, {UREA_ADDUCT: 1e5, UREA_H: 5e3, UREA_LOSS: 2e3}),
             (6, {UREA_ADDUCT: 8e4, UREA_H: 6.8e3, UREA_LOSS: 1e4}),
             (10, {UREA_ADDUCT: 6e4, UREA_H: 4.2e3, UREA_LOSS: 2e4})]
    _same_reading(_synthetic(round(p, 4), steps), UREA_ADDUCT, UREA_H, (3.0, True, 3))
    P = 460.25
    at = RD.HARMONIC_CAP[3] * 1e5
    assert not np.isclose(RD.harmonic_ions([([P / 9, P], [at, 1e5])]), P / 9).any()
    assert np.isclose(RD.harmonic_ions([([P / 9, P], [at * 1.001, 1e5])]), P / 9).any()
    # m/z 25 in the 2f band of the line at 100 and the 3f band of 225: above
    # one's cap, under the other's (either way round) -- under, no place
    for i25, i100, i225 in ((600.0, 1e5, 1e4), (400.0, 1e5, 4e3)):
        two = RD.harmonic_ions([([25.0, 100.0, 225.0], [i25, i100, i225])])
        assert not np.isclose(two, 25.0).any()


def _ghost_floor_steps():
    """GHOST_ION's 2f ghost holds 0.3 % of its line at NCE 1, 10 and 20; at
    NCE 30 the line has fallen to 4e3 and the ghost (24 counts) is read only
    where the noise lifts it: 0.6 % of the line, above the 2f cap."""
    p = _mz(GHOST_ION)
    g = p / 4 * (1 + 34e-6)
    out = []
    for nce, ip, a, b, share in ((1, 1e5, 0.0, 0.0, 0.003), (10, 7e4, 3e4, 0.0, 0.003),
                                 (20, 3e4, 6e4, 1e4, 0.003), (30, 4e3, 6e4, 3e4, 0.006)):
        peaks = {GHOST_ION: ip, g: share * ip}
        if a:
            peaks["C15H22O4+"] = a
        if b:
            peaks["C11H15O+"] = b
        out.append((nce, peaks))
    return round(p, 4), g, out


def test_a_ghost_read_near_the_floor_clears_no_place():
    """A ghost's share of its line reads high where the ghost is read near
    the S/N floor (only the noise's upswings are seen there): at NCE 30 it
    reads 0.6 %, above the 2f cap, at S/N 3.4. Such a reading clears no place
    (`HARMONIC_CLEAR_SN`): the ghost stays a ghost at NCE 1, 10 and 20, and is
    read (C4H7O+) only where it is above the cap itself. The same reading at
    S/N 24 is an ion's place, read at every energy -- and so where the
    spectra carry no S/N (as before)."""
    p, g, steps = _ghost_floor_steps()
    t = _synthetic(p, steps)
    t.spectra["signal_to_noise"] = t.spectra["intensity"] / 7.0
    on = AN.analyze_run(t, _ions(GHOST_ION))
    rd = on.readings[np.isclose(on.readings["mz"], g, atol=1e-5)]
    assert list(rd.loc[rd["harmonic"] != "", "nce"]) == [1.0, 10.0, 20.0]
    assert list(rd.loc[rd["harmonic"] == "", "formula"]) == ["C4H7O+"]
    fr = on.fragments.set_index("fragment").loc["C4H7O+"]
    assert (fr["onset_nce"], fr["in_source"], fr["n_spectra"]) == (30.0, False, 1)
    for sn in (t.spectra["intensity"] / 1.0, np.nan):
        t.spectra["signal_to_noise"] = sn
        _same_reading(t, GHOST_ION, "C4H7O+", (1.0, True, 4))
    # the unit: the S/N a reading above the cap needs to clear its place
    P = 460.25
    m, i = np.array([P / 4, P]), np.array([24.0, 4e3])
    for sn, cleared in ((RD.HARMONIC_CLEAR_SN, True), (RD.HARMONIC_CLEAR_SN * 0.99, False),
                        (np.nan, True)):
        ions_at = RD.harmonic_ions([(m, i, np.array([sn, 500.0]))])
        assert bool(np.isclose(ions_at, P / 4).any()) == cleared, sn
    # a peak in no band clears its place whatever its S/N
    assert np.isclose(RD.harmonic_ions([(m[:1], i[:1], np.array([1.0]))]), P / 4).any()


def test_the_s_n_a_reading_needs_to_clear_its_place_is_10():
    """`HARMONIC_CLEAR_SN` is 10, the value its docs argue (a 3f ghost of a
    few per cent read at S/N 10 stays under the 8 % cap): the floor scenario
    above with the NCE-30 ghost read at S/N 7.1 keeps it a ghost at NCE 1, 10
    and 20 (it is read only at NCE 30, where it is above the cap); read at
    S/N 12, its place is an ion's and it is read at all four energies."""
    assert RD.HARMONIC_CLEAR_SN == 10.0
    doc = (Path(__file__).resolve().parents[1] / "docs" / "MS2.md").read_text(encoding="utf-8")
    assert "| `readings.HARMONIC_CLEAR_SN` | 10 |" in doc and "`HARMONIC_CLEAR_SN` (10)" in doc
    p, g, steps = _ghost_floor_steps()
    for div, want in ((3.4, (30.0, False, 1)), (2.0, (1.0, True, 4))):
        t = _synthetic(p, steps)
        t.spectra["signal_to_noise"] = t.spectra["intensity"] / div      # ghost at NCE 30: 24 / div
        on = AN.analyze_run(t, _ions(GHOST_ION))
        rd = on.readings[np.isclose(on.readings["mz"], g, atol=1e-5)]
        fr = on.fragments.set_index("fragment").loc["C4H7O+"]
        assert (fr["onset_nce"], fr["in_source"], fr["n_spectra"]) == want, div
        assert list(rd.loc[rd["harmonic"] != "", "nce"]) == ([1.0, 10.0, 20.0] if want[1] is False
                                                             else []), div


def test_a_product_s_place_is_read_within_the_mda_floor_of_the_tolerance():
    """Below m/z 60 the reading tolerance is its 0.3 mDa floor, not 5 ppm:
    the place a product holds (`harmonic_ions` -> `harmonic_ghosts`'
    `ions_at`) is matched as `subformula` reads it, max(tol_ppm, tol_mda).
    C2H5O+ (45.0335) lies 24 ppm below m/z / 4 of C11H18NO+ (180.1383); at
    NCE 1 it holds 0.3 % of the line (under the 2f cap), read 6 ppm high
    (0.27 mDa: inside the floor, outside 5 ppm), and from NCE 10 on it is
    above the cap at its formula's mass -- one place, read at every energy,
    as without the rule (not a ghost at NCE 1, its onset moved to 10). The
    tolerance is the run's: read 8 ppm high (0.36 mDa) under `tol_mda` 0.4,
    the product is one place at every energy too -- matched at 0.4 mDa, as
    `subformula` reads it, not at the default floor."""
    P, Q = "C11H18NO+", "C2H5O+"
    mp, mq = _mz(P), _mz(Q)
    assert (mq - mp / 4) / (mp / 4) * 1e6 == pytest.approx(-24.0, abs=0.1)
    steps = [(1, {P: 1e5, mq * (1 + 6e-6): 300.0}), (10, {P: 7e4, Q: 3e3}),
             (20, {P: 3e4, Q: 6e3})]
    _same_reading(_synthetic(round(mp, 4), steps), P, Q, (1.0, True, 3))
    steps[0] = (1, {P: 1e5, mq * (1 + 8e-6): 300.0})
    assert mq * 8e-6 * 1e3 == pytest.approx(0.36, abs=0.005)
    _same_reading(_synthetic(round(mp, 4), steps), P, Q, (1.0, True, 3), tol_mda=0.4)


def _x179():
    """The 2026-09-10 exploratory file's 179.1069 window, the server's S/N-3
    spectra, as `ms2 extract` builds it (the summary's groups of this
    parent, its MS1 window)."""
    cen = _load("ms2_nh4_exploratory_179_sn3.json")
    summ = _load("ms2_nh4_exploratory_summary.json")
    summ = {**summ, "parent_peaks": [179.1069],
            "groups": [g for g in summ["groups"] if np.isclose(g["parent_peak_mz"], 179.1069)]}
    par = S.parents_table("S", cen, summ)
    win = S.window_table("S", _load("ms2_nh4_exploratory_ms1.json"), [179.1069],
                         summ["isolation_width"])
    return AN.RunTables(spectra=S.spectra_table("S", cen, summ), parents=par, window=win,
                        manifest={"sample_id": "S", "parameters": {
                            "isolation_width_used": summ["isolation_width"]}})


def test_our_own_2f_ghost_of_the_179_precursor():
    """The precursor C11H15O2+ (179.1068) carries its 2f ghost at m/z
    44.7766 at NCE 1, 5, 10 and 15: -1.9 to -2.7 ppm of m/z / 4, 0.098,
    0.098, 0.094 and 0.081 % of the line while the line falls from 3.21e5 to
    2.49e5 -- and at NCE 25 / 45 it is under the S/N floor. Nothing reads
    it (no formula has its mass defect), so F and every E50 are as before;
    out of the TIC, C11H15O2+'s branching sum at NCE 15 drops from 1.0506 to
    1.0488, under the flag."""
    t = _x179()
    ions = _ions("C11H15O2+", "C11H16^NO+")
    on, off = AN.analyze_run(t, ions), AN.analyze_run(t, ions, harmonics=False)
    rd = on.readings[on.readings["harmonic"] != ""]
    assert list(rd["nce"]) == [1.0, 5.0, 10.0, 15.0]
    assert np.allclose(rd["mz"], 44.7766, atol=1e-4)
    ppm = rd["harmonic"].str.extract(r": ([-+0-9.]+) ppm")[0].astype(float)
    pct = rd["harmonic"].str.extract(r"ppm, ([0-9.]+) % of it")[0].astype(float)
    assert list(ppm) == [-1.9, -1.9, -2.0, -2.7] and list(pct) == [0.0985, 0.0985, 0.0938, 0.0814]
    old = off.readings[np.isclose(off.readings["mz"], 44.7766, atol=1e-4)]
    assert (old["rank"] == 0).all()                         # never read before either
    for c in ("f", "f_lo", "s"):
        pd.testing.assert_series_equal(on.breakdown[c], off.breakdown[c])
    pd.testing.assert_series_equal(on.e50["e50_nce"], off.e50["e50_nce"])
    assert _e50(on, 179.1069, "C11H15O2+")["e50_nce"] == pytest.approx(21.094, abs=1e-3)
    bd, b0 = on.breakdown.set_index(["nce", "ion"]), off.breakdown.set_index(["nce", "ion"])
    assert b0.loc[(15.0, "C11H15O2+"), "branching_sum"] == pytest.approx(1.0506, abs=1e-4)
    assert bd.loc[(15.0, "C11H15O2+"), "branching_sum"] == pytest.approx(1.0488, abs=1e-4)
    assert b0.loc[(15.0, "C11H15O2+"), "branching_flag"]
    assert not bd.loc[(15.0, "C11H15O2+"), "branching_flag"]
    assert list(bd.xs("C11H15O2+", level="ion")["n_harmonic"]) == [1, 1, 1, 1, 0, 0]
    # out of the peak count too, and explained by nothing
    x, x0 = bd.xs("C11H15O2+", level="ion"), b0.xs("C11H15O2+", level="ion")
    assert list(x["n_peaks"]) == [25, 22, 28, 23, 19, 19]
    assert list(x["n_peaks"]) == list(x0["n_peaks"] - x["n_harmonic"])
    assert (x.loc[1.0, "budget_explained"], x0.loc[1.0, "budget_explained"]) == \
        (pytest.approx(0.953686, abs=1e-6), pytest.approx(0.952979, abs=1e-6))
    man = AN.manifest(on, t, files={}, created_utc=None)
    assert man["counts"]["harmonic_peaks"] == 4 and man["counts"]["harmonic_by_order"] == {"2f": 4}
    assert man["windows"][0]["harmonic_peaks"] == {"2f": 4}
    assert man["parameters"]["harmonics"] == {"on": True, "orders": [2, 3], "ppm": 40.0,
                                              "cap": {"2f": 0.005, "3f": 0.08},
                                              "clear_sn": RD.HARMONIC_CLEAR_SN}


# ----------------------------------------------------------------- CLI ----
def _per_scan_folder(d, std_a):
    _, per = std_a
    _write_tables(d, per)
    per.scans.to_csv(d / "ms2_scans.csv", index=False)
    per.scan_peaks.to_parquet(d / "ms2_scan_peaks.parquet", index=False)
    return d


def test_load_run_reads_the_per_scan_tables(tmp_path, std_a):
    d = _per_scan_folder(tmp_path / "run", std_a)
    t = AN.load_run(str(d))
    assert t.scan_peaks is not None and len(t.scan_peaks) == len(std_a[1].scan_peaks)
    assert {c: str(x) for c, x in t.scan_peaks.dtypes.items()} == SC.SCAN_PEAK_COLUMNS
    assert set(t.inputs) == set(AN.RUN_FILES) | set(AN.PER_SCAN_FILES)
    # skipped at extract: the tables are there with their headers only
    SC._frame({}, SC.SCAN_PEAK_COLUMNS).to_parquet(d / "ms2_scan_peaks.parquet", index=False)
    t = AN.load_run(str(d))
    assert t.scan_peaks is None and any("holds no row" in n for n in t.notes)
    res = AN.analyze_run(t, _ions("C10H17+"))
    assert AN.NO_PER_SCAN_NOTE not in res.notes and res.replicates is None


@pytest.fixture
def no_server(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("ms2 analyze asked for Mascope")
    from peaky.io import io_mascope as IO
    monkeypatch.setattr(cli, "_require_creds", boom)
    monkeypatch.setattr(IO, "connect", boom)


def test_cli_analyze_reads_the_scans(tmp_path, std_a, no_server, capsys):
    run = _per_scan_folder(tmp_path / "run", std_a)
    out = tmp_path / "out"
    argv = ["ms2", "analyze", "--run-dir", str(run), "--ion", "137.1323=C10H17+",
            "--no-figures", "--out-dir", str(out)]
    assert cli.main(argv) == 0
    printed = capsys.readouterr().out
    assert ("parent 137.1321: analysed; target C10H17+; offset -1.07 ppm; E50 NCE 13.9 (6.33 V, "
            "E_cm 1.073 eV) [16-84 %: 13.84 to 13.94, 1000 resamples]") in printed
    names = {p.name for p in out.iterdir()}
    assert "ms2_replicates.csv" in names
    rep = pd.read_csv(out / "ms2_replicates.csv", keep_default_na=False, na_values=[""])
    assert list(rep.columns) == list(RP.REPLICATE_COLUMNS)
    man = json.loads((out / "ms2_analyze_manifest.json").read_text())
    assert man["parameters"]["replicates"] == {"per_scan": True, "bootstrap": 1000,
                                               "bootstrap_min": RP.BOOTSTRAP_MIN,
                                               "seed": RP.BOOTSTRAP_SEED,
                                               "percentiles": [16.0, 84.0], "match_da": 1e-6,
                                               "scale_band": [0.5, 2.0]}
    assert man["files"]["ms2_replicates.csv"] and man["counts"]["replicate_rows"] == len(rep)
    assert man["counts"]["e50_intervals"] == 3 and man["counts"]["harmonic_peaks"] == 0
    assert set(man["inputs"]) == set(AN.RUN_FILES) | set(AN.PER_SCAN_FILES)
    # --bootstrap 0: the statistics, no interval
    assert cli.main(argv + ["--bootstrap", "0"]) == 0
    printed = capsys.readouterr().out
    assert "resamples]" not in printed
    e = pd.read_csv(out / "ms2_e50.csv")
    assert (e["boot_n"] == 0).all() and e["e50_nce_lo"].isna().all()
    # the same out-dir from a folder without the scans: no stale replicate table
    plain = _write_tables(tmp_path / "plain", std_a[0])
    assert cli.main(["ms2", "analyze", "--run-dir", str(plain), "--ion", "137.1323=C10H17+",
                     "--no-figures", "--out-dir", str(out)]) == 0
    assert not (out / "ms2_replicates.csv").exists()
    man = json.loads((out / "ms2_analyze_manifest.json").read_text())
    assert "ms2_replicates.csv" not in man["files"] and AN.NO_PER_SCAN_NOTE in man["notes"]
    with pytest.raises(SystemExit, match="--bootstrap must be"):
        cli.main(argv + ["--bootstrap", "-1"])


def test_cli_prints_the_band_with_its_caveats(tmp_path, no_server, capsys):
    """A bound past the energies is printed on its side; a band next to an
    energy of fewer than two scans says so, and a zero-width band is never
    printed bare."""
    p = round(_mz(X), 4)
    base = _three_steps()
    for name, steps, want in (
            ("lower", _lower_steps(), "[16-84 %: below range to 2.50, 1000 resamples]"),
            ("thin", [base[0], (10, base[1][1][:1]), (20, base[2][1][:1])],
             "[16-84 %: 12.50 to 12.50, 1000 resamples; 2 of the 2 energies about the crossing "
             "add no spread; zero width]")):
        t = _replicated(p, steps)
        t.manifest = {"parameters": {"isolation_width_used": 0.4}}
        run = _write_tables(tmp_path / name, t)
        t.scan_peaks.to_parquet(run / "ms2_scan_peaks.parquet", index=False)
        assert cli.main(["ms2", "analyze", "--run-dir", str(run), "--ion", f"{p}={X}",
                         "--no-figures", "--out-dir", str(tmp_path / f"out_{name}")]) == 0
        assert want in capsys.readouterr().out


def test_cli_keep_harmonics_reads_the_ghost(tmp_path, no_server):
    """`--keep-harmonics` reaches the analysis: by default the synthetic
    C15H26NO4+ window's 2f ghost is no formula (three `harmonic` rows, no
    C4H7O+ fragment, the manifest's rule on); with the flag it is read as
    C4H7O+, as any peak, and the manifest says the rule was off."""
    p, steps = _ghost_steps()
    t = _synthetic(p, steps)
    t.manifest = {"parameters": {"isolation_width_used": 0.4}}
    run = _write_tables(tmp_path / "run", t)
    for flag, rule in (([], True), (["--keep-harmonics"], False)):
        out = tmp_path / f"out{len(flag)}"
        assert cli.main(["ms2", "analyze", "--run-dir", str(run), "--ion", f"{p}={GHOST_ION}",
                         "--no-figures", "--out-dir", str(out)] + flag) == 0
        man = json.loads((out / "ms2_analyze_manifest.json").read_text())
        assert man["parameters"]["harmonics"]["on"] is rule
        assert man["counts"]["harmonic_peaks"] == (3 if rule else 0)
        rd = pd.read_csv(out / "ms2_readings.csv", keep_default_na=False, na_values=[""])
        assert int(rd["harmonic"].notna().sum()) == (3 if rule else 0)
        fr = pd.read_csv(out / "ms2_fragments.csv")
        assert ("C4H7O+" in set(fr["fragment"])) is not rule


def test_cli_parses_the_new_flags():
    P = cli.build_parser()
    a = P.parse_args(["ms2", "analyze", "--run-dir", "D", "--ion", "137.13=C10H17+"])
    assert (a.bootstrap, a.keep_harmonics) == (RP.BOOTSTRAP_B, False)
    a = P.parse_args(["ms2", "analyze", "--run-dir", "D", "--ion", "137.13=C10H17+",
                      "--bootstrap", "200", "--keep-harmonics"])
    assert (a.bootstrap, a.keep_harmonics) == (200, True)
    text = _sub("ms2", "analyze").format_help()
    assert "--bootstrap N" in text and "--keep-harmonics" in text


def test_the_docs_name_every_new_column():
    """§11.7 names the replicate table's columns and §11.11 every constant
    this PR adds."""
    doc = (Path(__file__).resolve().parents[1] / "docs" / "MS2.md").read_text(encoding="utf-8")
    sec = doc[doc.index("### 11.7 "):doc.index("### 11.8 ")]
    (row,) = [ln for ln in sec.splitlines() if ln.startswith("| `ms2_replicates.csv` |")]
    keys = {"parent_mz", "spectrum"}
    assert not [c for c in RP.REPLICATE_COLUMNS if c not in keys and f"`{c}`" not in row]
    for name in ("HARMONIC_ORDERS", "HARMONIC_PPM", "HARMONIC_CAP", "HARMONIC_CLEAR_SN",
                 "BOOTSTRAP_B", "BOOTSTRAP_MIN", "BOOTSTRAP_SEED", "BOOTSTRAP_PERCENTILES",
                 "MATCH_DA", "SCALE_BAND", "ZERO_WIDTH_REL"):
        assert name in doc, name
    (row,) = [ln for ln in sec.splitlines() if ln.startswith("| `ms2_e50.csv` |")]
    assert not [c for c in NEW_E50 if f"`{c}`" not in row]


def test_the_docs_state_what_was_measured():
    """The words of the docs, the module and the CHANGELOG on what the code
    does and what was measured: the server's height runs below the per-scan
    mean (the test above); the 16-84 % band is a percentile band of the
    resamples that holds the true E50 less often than 68 % at a few scans
    per energy, not a 1-sigma band; `n_one_scan` counts the energies of
    fewer than two scans (a spectrum with no per-scan row adds no spread
    either); the two-scan F agreement is a median, not a bound."""
    def words(text):
        return " ".join(text.split())
    root = Path(__file__).resolve().parents[1]
    doc = (root / "docs" / "MS2.md").read_text(encoding="utf-8")
    sec = words(doc[doc.index("### 11.11 "):doc.index("### 11.12 ")])
    src = words(Path(RP.__file__).read_text(encoding="utf-8"))
    for text in (sec, src):
        assert "about 2 % below the per-scan mean" in text
        assert "above the per-scan mean" not in text
        assert "1-sigma" not in text
    assert "1-sigma" not in doc
    assert "holds the true E50 in" in sec
    t7 = words(doc[doc.index("### 11.7 "):doc.index("### 11.8 ")])
    assert ("`n_one_scan` (the window's energies of fewer than two scans — one, or no per-scan "
            "row — or whose every precursor / fragment peak is held") in t7
    log = (root / "CHANGELOG.md").read_text(encoding="utf-8")
    entry = words(log[log.index("**`peaky ms2 analyze` reads the per-scan tables"):])
    entry = entry[:entry.index(" - **")]
    # two scans: the share of fragments is a mean over the pairs (the worst
    # pair holds far fewer), and the F the worst pair moves most is a
    # co-isolate's, twice the targets' worst
    for text in (sec, entry):
        assert "(0.009 at worst)" not in text and "any two scans" not in text
        assert "the worst pair moves a target's F by 0.009, a co-isolate's by up to 0.019" in text
    assert ("two scans of a spectrum, averaged over its pairs, hold every fragment of its S/N-3 "
            "spectrum on the median spectrum (95.5 % at the 10th percentile; the worst pair 67.5 "
            "% there, 50 % in one spectrum)") in sec
