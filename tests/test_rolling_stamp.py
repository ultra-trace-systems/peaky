"""The --rolling-centre path: recentre_ledger's rolling centre, the per-trace
stamping window and drift-following stamping in annotate_peaks. Off by default,
and the default path is untouched bit for bit."""
import numpy as np
import pandas as pd

from peaky.batch import timeseries as TS
from peaky.batch import traces as TR

T0 = pd.Timestamp("2026-01-01T00:00:00Z")


def _batch(rng, n=400, drift_ppm=24.0, sigma=0.5, static=250.0, moving=200.0):
    """Hourly spectra with one STATIC ion, one ion whose position WANDERS by
    +-drift_ppm/2 around its median over the batch (one slow oscillation --
    a linear ramp would have CONSTANT lag differences and no structure-function
    signal, which is correct: the estimator models a random walk, not a line),
    and sparse noise. Returns (ts, drift_ppm_per_spectrum)."""
    ramp = 0.5 * drift_ppm * np.sin(2 * np.pi * np.arange(n) / n)
    rows = []
    for i in range(n):
        sid = f"s{i:04d}"
        stamp = T0 + pd.Timedelta(hours=i)
        rows.append((sid, static * (1 + rng.normal(0, sigma) * 1e-6), 50.0, stamp))
        rows.append((sid, moving * (1 + (ramp[i] + rng.normal(0, sigma)) * 1e-6), 40.0, stamp))
        for m in rng.uniform(150.0, 300.0, 12):
            rows.append((sid, m, 1.0, stamp))
    ts = pd.DataFrame(rows, columns=["sample_item_id", "mz", "height", "datetime_utc"])
    return ts, ramp


def _merged(static=250.0, moving=200.0):
    return pd.DataFrame({"mz": [static, moving], "neutral_formula": ["C10H16O4", "C9H14O4"],
                         "adduct": ["[M-H]-", "[M-H]-"], "tier": ["Assigned", "Assigned"],
                         "n_files": [3, 3], "ion_score": [0.9, 0.9]})


def test_the_default_path_is_untouched():
    rng = np.random.default_rng(1)
    ts, _ = _batch(rng)
    idx = TR.PeakIndex(ts, tol_ppm=6.0)
    a, b = _merged(), _merged()
    ia = TS.recentre_ledger(a, index=idx, tol_ppm=6.0, log=None)
    ib = TS.recentre_ledger(b, index=idx, tol_ppm=6.0, rolling=False, log=None)
    pd.testing.assert_frame_equal(a, b)
    assert ia == ib and "tracks" not in ia and "rolling" not in ia
    assert "centre_scheme" not in a.columns and "trace_key" not in a.columns


def test_rolling_rolls_the_drifting_ion_and_leaves_the_static_one():
    rng = np.random.default_rng(2)
    ts, ramp = _batch(rng)
    idx = TR.PeakIndex(ts, tol_ppm=6.0)
    m = _merged()
    info = TS.recentre_ledger(m, index=idx, tol_ppm=6.0, rolling=True,
                              times_by_code=TS.sample_hours(idx, ts), log=None)
    assert list(m["centre_scheme"]) == ["global", "rolling"]
    assert info["rolling"]["n_rolling"] == 1 and info["rolling"]["n_global"] == 1
    assert set(info["tracks"]) == {1}
    t, c = info["tracks"][1]
    assert len(t) == len(c) and np.all(np.diff(t) > 0)
    assert 18.0 < m.loc[1, "track_span_ppm"] < 30.0          # the +-12 ppm wander
    assert m.loc[1, "resid_ppm"] < 3.0 < 4.0                 # about the MOVING centre ...
    off = (idx.mz[idx.members(m.loc[1, "mz_trace"], 12.0)] - m.loc[1, "mz_trace"]) / m.loc[1, "mz_trace"] * 1e6
    assert TR.MAD_TO_SIGMA * np.median(np.abs(off - np.median(off))) > 3.0   # ... not the batch one
    assert np.isfinite(m.loc[0, "sigma_ppm"]) and 0.3 < m.loc[0, "sigma_ppm"] < 0.8
    assert list(m["trace_key"]) == [0, 1]


def test_rolling_without_timestamps_is_skipped_and_says_so():
    rng = np.random.default_rng(3)
    ts, _ = _batch(rng)
    idx = TR.PeakIndex(ts, tol_ppm=6.0)
    m = _merged()
    lines = []
    info = TS.recentre_ledger(m, index=idx, tol_ppm=6.0, rolling=True, log=lines.append)
    # no ts_peaks were given, so no timestamps can be read
    assert info["rolling"]["skipped"] and "tracks" not in info
    assert any("no timestamps" in ln for ln in lines)
    assert m["centre_scheme"].eq("").all()


def test_stamp_tolerances_per_trace_clip_like_the_batch_rule():
    m = pd.DataFrame({"resid_ppm": [0.5, 2.0, 3.0, 10.0, np.nan]})
    got = TS.stamp_tolerances(m, tol_ppm=6.0, k_sigma=2.5, max_x=2.0, fallback=9.0)
    assert list(got) == [6.0, 6.0, 7.5, 12.0, 9.0]
    assert list(TS.stamp_tolerances(pd.DataFrame({"mz": [1.0]}), tol_ppm=6.0)) == [6.0]


def test_a_per_row_window_overrides_the_batch_window():
    ts = pd.DataFrame({"sample_item_id": ["a", "a"], "mz": [200.0 * (1 + 9e-6), 300.0 * (1 + 9e-6)],
                       "height": [1.0, 1.0]})
    led = pd.DataFrame({"mz": [200.0, 300.0], "neutral_formula": ["X", "Y"], "adduct": ["[M-H]-"] * 2,
                        "tier": ["Assigned"] * 2, "stamp_tol_ppm": [12.0, np.nan]})
    ann = TS.annotate_peaks(ts, led, tol_ppm=6.0)
    assert list(ann["neutral_formula"].fillna("-")) == ["X", "-"]     # 12 ppm row stamps, 6 ppm row not


def test_a_rolling_track_stamps_the_spectra_the_median_window_loses():
    rng = np.random.default_rng(4)
    ts, ramp = _batch(rng)
    idx = TR.PeakIndex(ts, tol_ppm=6.0)
    m = _merged()
    info = TS.recentre_ledger(m, index=idx, tol_ppm=6.0, rolling=True,
                              times_by_code=TS.sample_hours(idx, ts), log=None)
    m["stamp_tol_ppm"] = TS.stamp_tolerances(m, tol_ppm=6.0)
    frame = m[["mz_trace", "neutral_formula", "adduct", "tier", "trace_key", "track_span_ppm",
               "stamp_tol_ppm"]].rename(columns={"mz_trace": "mz"})
    without = TS.annotate_peaks(ts, frame.drop(columns=["trace_key", "track_span_ppm"]), tol_ppm=6.0)
    with_ = TS.annotate_peaks(ts, frame, tol_ppm=6.0, tracks=info["tracks"])
    n = ts["sample_item_id"].nunique()
    cov_w = with_[with_["neutral_formula"] == "C9H14O4"]["sample_item_id"].nunique() / n
    cov_o = without[without["neutral_formula"] == "C9H14O4"]["sample_item_id"].nunique() / n
    assert cov_o < 0.7 < 0.95 < cov_w, (cov_o, cov_w)
    # the static ion is unaffected either way
    for a in (with_, without):
        assert a[a["neutral_formula"] == "C10H16O4"]["sample_item_id"].nunique() / n > 0.97
    # and a track never stamps a peak that is far from the centre AT ITS TIME: the
    # wander reaches +-12 ppm from the median, so the median window (6 ppm, or
    # the 1.5 mDa floor = 7.5 ppm here) loses half the spectra while the track
    # keeps them, and no noise peak is pulled in
    stamped = with_[with_["neutral_formula"] == "C9H14O4"]
    assert (stamped["height"] == 40.0).all()


def test_tracks_are_ignored_without_peak_timestamps():
    rng = np.random.default_rng(5)
    ts, _ = _batch(rng)
    idx = TR.PeakIndex(ts, tol_ppm=6.0)
    m = _merged()
    info = TS.recentre_ledger(m, index=idx, tol_ppm=6.0, rolling=True,
                              times_by_code=TS.sample_hours(idx, ts), log=None)
    frame = m[["mz_trace", "neutral_formula", "adduct", "tier", "trace_key", "track_span_ppm"]] \
        .rename(columns={"mz_trace": "mz"})
    a = TS.annotate_peaks(ts.drop(columns=["datetime_utc"]), frame, tol_ppm=6.0, tracks=info["tracks"])
    b = TS.annotate_peaks(ts.drop(columns=["datetime_utc"]), frame.drop(columns=["trace_key"]), tol_ppm=6.0)
    pd.testing.assert_series_equal(a["neutral_formula"], b["neutral_formula"])


def test_the_cli_registers_the_rolling_flag_on_batch_and_pool():
    from peaky import cli
    ap = cli.build_parser()
    a = ap.parse_args(["batch", "--batch", "x", "--rolling-centre"])
    assert a.rolling_centre is True
    b = ap.parse_args(["pool", "--batches", "x", "--rolling-centre"])
    assert b.rolling_centre is True
    assert ap.parse_args(["batch", "--batch", "x"]).rolling_centre is False
