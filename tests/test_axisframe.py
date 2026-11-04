"""batch.axisframe: Mascope's Orbitrap axis fingerprint read into a batch's band-step
frame -- the law and the bands, s <-> r, the robust local (LOESS) trend and its pinned
convention, what is refused, the calibration groups (one pooled table and one grid per
calibration), the edge zone in frequency, the frame's exact composite with a lock model,
its inverse, the FrameSet, the massqc dispatch and the truth-free gate. Every document
and every batch here is synthetic: a schema-shaped document built from a chosen grid,
a smooth trend, chosen step offsets per band and a seeded band noise of the size a real
table scatters by."""
import json
import pickle
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from peaky.batch import axisframe as AF
from peaky.batch import axislock as AL
from peaky.batch import massqc as MQ
from tests.test_axislock import _batch, _smooth

# an Orbitrap-like law and band grid (generic magnitudes: B ~ 7e7, C ~ -3e8 law-kHz units,
# bands of 16.28 law-kHz)
B0, C0, PERIOD, PHASE = 6.78e7, -3.40e8, 16.28, 8.14
MODE = "-|R280000|50.0000-750.0000"
SCAN_COLUMNS = ["scan", "rt_min", "mode", "microscans", "B", "C", "n_lm_found", "lm_correction_ppm", "lock_used"]
NOISE_PPM = 0.05        # the band-to-band scatter about the smooth trend (served tables: 1e-4 to 0.05 ppm)
STEP_TOL = 0.25         # a step is recovered to the table's noise plus the local trend's own error


def _cubic(k):
    """A smooth trend of r across bands, ppm: what a calibration's common curve looks like."""
    k = np.asarray(k, dtype=float) - 40.0
    return 0.8 - 0.05 * k + 0.002 * k ** 2 - 2e-5 * k ** 3


def _quadratic_with_end_ramp(k):
    """A quadratic that bends up over the last bands (slope continuous): the shape whose end
    a global cubic in k misfit at the ends of served tables."""
    k = np.asarray(k, dtype=float)
    return 0.8 - 0.05 * (k - 40) + 0.002 * (k - 40) ** 2 + np.where(k > 57, 0.02 * (k - 57) ** 2, 0.0)


def _band_noise(seed: int) -> np.ndarray:
    """One standard-normal draw per PHYSICAL band index 0..199, fixed by the seed, so two
    documents of one calibration share their bands' noise exactly as two files of one
    calibration share their s."""
    return np.random.default_rng(1000 + seed).normal(0.0, 1.0, 200)


def _rows(k0, k1, trend, steps, *, interp=(), weak=(), B=B0, C=C0, period=PERIOD, phase=PHASE, rounded=False,
          noise=NOISE_PPM, seed=0, phys=lambda k: k):
    """`segments` rows in the schema's column order from a grid, a trend and step offsets
    per band (r = trend(K) + step_K + noise_K at the PHYSICAL band K = phys(k); the row
    stores s = s_from_r(r)). An `interp` band carries the table's own interpolation flag
    and a value the frame must ignore; a `weak` band too few chunks at its level."""
    nz = _band_noise(seed)
    rows = []
    for k in range(k0, k1 + 1):
        K = int(phys(k))
        f_lo = phase + period * k
        f_hi = f_lo + period
        mz_lo, mz_hi = AF.band_edges_mz(k, B, C, period, phase)
        r = float(trend(K)) + float(steps.get(K, 0.0)) + noise * float(nz[K])
        is_interp = k in interp
        s = float(AF.s_from_r(r)) if not is_interp else float(AF.s_from_r(r + 10.0))
        n_chunks, n_level, share, n_scans, sd = 16, 16, 1.0, 8, 0.0
        if k in weak:
            n_level, share = 2, 0.125
        if is_interp:
            n_chunks, n_level, share, n_scans, sd = 0, 0, 0.0, 0, None
        if rounded:
            rows.append([k, round(f_lo, 4), round(f_hi, 4), round(float(mz_lo), 4), round(float(mz_hi), 4),
                         round(s, 4), n_chunks, n_level, round(share, 3), n_scans, sd, is_interp])
        else:
            rows.append([k, f_lo, f_hi, float(mz_lo), float(mz_hi), s, n_chunks, n_level, share, n_scans, sd,
                         is_interp])
    return rows


def _mode(*, k0=18, k1=65, trend=_cubic, steps=None, interp=(), weak=(), polarity="-", grid_source="measured",
          flags=(), B=B0, C=C0, period=PERIOD, phase=PHASE, rounded=False, mode_id=None, n_hits=38,
          edge_uncertainty=0.1, resolution=280000, scan_range=(50.0, 750.0), c_over_b2=None, noise=NOISE_PPM,
          seed=0, phys=lambda k: k):
    mode_id = mode_id or MODE.replace("-", polarity, 1)
    cb2 = C / B ** 2 if c_over_b2 is None else c_over_b2
    return {
        "mode_id": mode_id, "polarity": polarity, "resolution": resolution, "scan_range": list(scan_range),
        "n_scans": 80, "scan_share": 1.0, "microscans": {"1": 80},
        "calibration": {"B_median": B, "C_median": C, "C_over_B2": cb2, "C_over_B2_min": cb2, "C_over_B2_max": cb2},
        "profile": {"scans_used": [2, 21, 41, 60, 79], "fft_step_hz_measured": [0.244140625] * 5,
                    "nominal_fft_step_hz": 0.244140625, "n_chunks": 800,
                    "scan_check_columns": ["scan", "n_points", "mz_min", "mz_max", "sum_mz"], "scan_check": []},
        "lock": {"used_share": {"100.0000": 1.0}, "used_count": {"100.0000": 80}, "m_lock_eff": 100.0,
                 "s_lock_eff_ppm": -1.2, "k_lock": {"100.0000": 46}, "configured_mz": [100.0],
                 "method": "reference_flag", "lm_correction_ppm_median_by_lock": {}},
        "levels": None, "brackets": None,
        "grid": {"source": grid_source, "period_khz": period, "phase_khz": phase,
                 "period_hypothesis": "fixed_frequency", "n_brackets": 40, "n_hits": n_hits, "hit_fraction": 0.95,
                 "random_hit_fraction_q95": 0.4, "edge_uncertainty_khz": edge_uncertainty,
                 "k_hit_range": [k0 + 2, k1 - 2], "significant": True},
        "segments": {"columns": list(AF.SEGMENT_COLUMNS),
                     "rows": _rows(k0, k1, trend, steps or {}, interp=interp, weak=weak, B=B, C=C,
                                   period=period, phase=phase, rounded=rounded, noise=noise, seed=seed, phys=phys)},
        "k_range_measured": [k0, k1], "flags": list(flags),
    }


def _document(*, status="ok", version=1, modes=None, dominant=None, file_flags=(), item=None, **mode_kw):
    """A schema-shaped `mascope.axis_fingerprint` document (version 1) as the file-level
    route serves it; with `item` as the per-item route does."""
    modes = modes if modes is not None else [_mode(**mode_kw)]
    if status == "no_fingerprint":                      # read, but no mode yielded a table
        modes = [dict(m, segments=None, k_range_measured=None, flags=list(m["flags"]) + ["no_segment_table"])
                 for m in modes]
    doc = {"schema": AF.SCHEMA, "version": version, "status": status,
           "extractor": {"name": "synthetic", "reader": "synthetic 1.0", "params": {}},
           "source": {"raw_name": "synthetic.RAW", "first_scan": 1, "last_scan": 80, "n_scans": 80},
           "scan_table": {"columns": SCAN_COLUMNS,
                          "rows": [[i, round(i * 0.1, 5), modes[0]["mode_id"] if modes else None, 1, B0, C0, 1,
                                    0.0, "100.0000"] for i in range(1, 81)]},
           "modes": modes, "dominant_mode": dominant or (modes[0]["mode_id"] if modes else None),
           "flags": list(file_flags), "cost_seconds": {}}
    if item is not None:
        doc["item"] = item
    return doc


def _item(sid="s0", *, mode_id=MODE, streams=None, stream_key=None, polarity="-"):
    return {"sample_item_id": sid, "sample_file_id": "f0", "polarity": polarity, "t0_s": 0.0, "t1_s": 480.0,
            "stream_key": stream_key, "scans": list(range(2, 81)), "mode_id": mode_id,
            "lock": {"used_share": {"100.0000": 1.0}, "used_count": {"100.0000": 79}, "m_lock_eff": 100.0,
                     "s_lock_eff_ppm": -1.25},
            "mz_calibration_factor": 1.0000012, "file_mz_calibration_factor": 1.0000012, "streams": streams}


def _k_at(mz, B=B0, C=C0, period=PERIOD, phase=PHASE) -> int:
    return int(AF.k_of_mz(mz, B, C, period, phase))


#: the tests' stepped band: the band holding m/z STEP_MZ under this law (m/z ~294-315) is
#: STEP_PPM off; SIDE_PPM sits on a neighbour (or on every band above it in m/z)
STEP_MZ, STEP_PPM, SIDE_PPM = 300.0, 2.4, 0.6
K_MID = _k_at(STEP_MZ)


def _spline_inner(lo=80.0, hi=430.0, shape=_smooth) -> AL.AxisModel:
    """A lock model with one spline segment over [lo, hi] in the W frame."""
    x = np.linspace(lo, hi, 400)
    return AL.AxisModel(segments=[{"kind": "spline", "lo": lo, "hi": hi, "n": 100, "x": x, "y": shape(x)}],
                        n_locks=100)


def _dense(item: AF.FrameItem, extra_edges=()) -> np.ndarray:
    """+-4 mDa around every band edge, every zone boundary and every inner-model edge, plus
    a 1-mDa grid over the reach."""
    lo, hi = item.reach
    edges = list(item.edges) + list(item.zone_bounds) + list(extra_edges)
    pts = [np.linspace(e - 0.004, e + 0.004, 801) for e in edges] + [np.arange(lo - 1.0, hi + 1.0, 0.001)]
    return np.unique(np.concatenate(pts))


def _ambiguous(item: AF.FrameItem, mz) -> np.ndarray:
    """Where the frame's map is not one-to-one: within half the step across the nearest
    edge (plus a hair) of a ZONE BOUNDARY in m/z, where the map jumps from a band's d to
    the two bands' mean (the images overlap or leave a gap there)."""
    mz = np.asarray(mz, dtype=float)
    out = np.zeros(mz.shape, dtype=bool)
    f = item.phase + item.period * item.edge_indices.astype(float)
    for s, fj in zip(item.edge_steps()[::-1], f):             # edge_steps ascend in m/z = descending j
        half = 0.5 * abs(s["step_ppm"]) + 1e-3
        for fb in (fj - item.zone_khz, fj + item.zone_khz):
            e = float(AF.mz_of_f(fb, item.B, item.C))
            out |= np.abs(mz - e) / e * 1e6 <= half
    return out


def _at(fp_or_item, k) -> int:
    return int(k) - int(fp_or_item.k0)


# --- the law, the bands, s <-> r ----------------------------------------------------------

def test_the_law_round_trips_and_the_bands_tile_the_axis_as_the_document_says():
    m = np.linspace(50.0, 750.0, 7001)
    back = AF.mz_of_f(AF.law_khz(m, B0, C0), B0, C0)
    assert np.allclose(back, m, rtol=1e-12, atol=0)
    k = AF.k_of_mz(m, B0, C0, PERIOD, PHASE)
    assert np.all(np.diff(k) <= 0)                                  # a higher k is a lower m/z
    # the document's edges (stored to 4 decimals, as Mascope rounds them) agree with the law's
    doc = _document(rounded=True, noise=0.0)
    seg = doc["modes"][0]["segments"]
    at = {c: i for i, c in enumerate(seg["columns"])}
    for row in seg["rows"]:
        kk, lo, hi = row[at["k"]], row[at["mz_lo"]], row[at["mz_hi"]]
        assert _k_at(0.5 * (lo + hi)) == kk
        elo, ehi = AF.band_edges_mz(kk, B0, C0, PERIOD, PHASE)
        assert abs(float(elo) - lo) < 1e-4 and abs(float(ehi) - hi) < 1e-4
    # just inside an exact edge the band is k, just above its upper edge k - 1 (lower m/z, higher k)
    for row in _document(noise=0.0)["modes"][0]["segments"]["rows"]:
        kk, lo, hi = row[at["k"]], row[at["mz_lo"]], row[at["mz_hi"]]
        assert _k_at(lo * (1 + 1e-9)) == kk and _k_at(hi * (1 - 1e-9)) == kk
        assert _k_at(lo * (1 - 1e-9)) == kk + 1 and _k_at(hi * (1 + 1e-9)) == kk - 1
    assert np.isnan(AF.law_khz([0.0, -1.0, np.nan], B0, C0)).all()


def test_r_from_s_is_the_exact_inverse_of_mascopes_spacing_relation():
    r = np.linspace(-150.0, 250.0, 4001)
    s = ((1.0 + r * 1e-6) ** -1.5 - 1.0) * 1e6                      # reference.s_from_fudge_ratio
    assert np.allclose(AF.s_from_r(r), s, rtol=0, atol=1e-12)
    assert np.allclose(AF.r_from_s(s), r, rtol=0, atol=1e-8)
    literal = ((1.0 + s * 1e-6) ** (-2.0 / 3.0) - 1.0) * 1e6
    assert np.array_equal(AF.r_from_s(s), literal)
    assert abs(AF.r_from_s(1.5) / 1.5 + 2.0 / 3.0) < 1e-5            # r ~ -(2/3) s


# --- the robust local trend --------------------------------------------------------------

def test_the_trend_follows_a_quadratic_with_an_end_ramp_that_a_global_cubic_misfits():
    """The regression of the cubic's end misfit: on a no-step table whose trend bends up
    over the last bands, the local quadratic stays within 0.05 ppm everywhere while a
    global cubic in k is off by > 0.2 ppm at the end (by more on served tables)."""
    k = np.arange(18, 66)
    true = _quadratic_with_end_ramp(k)
    r = true + 0.02 * _band_noise(0)[k]
    lo = AF.loess_trend(k, r)
    assert np.abs(lo["trend"] - true).max() < 0.05, np.abs(lo["trend"] - true).max()
    cubic = np.polyval(np.polyfit(k, r, 3), k)
    assert np.abs(cubic - true)[-1] > 0.2 and np.abs(cubic - true).max() > 0.2
    # through step_table: no band stands out as a step
    tab = AF.step_table(k, AF.s_from_r(r), np.ones(len(k), bool))
    assert np.abs(tab["d"]).max() < 0.1 and tab["measured"].all()
    assert lo["degree"].tolist() == [2] * len(k) and lo["n_window"].min() == 7 and lo["n_window"].max() == 13


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_the_trend_keeps_an_isolated_step_ignores_a_gross_outlier_and_spans_a_gap(seed):
    k = np.arange(18, 66)
    r = _cubic(k) + NOISE_PPM * _band_noise(seed)[k]
    i30, i45 = _at_k(k, 30), _at_k(k, 45)
    r[i30] += 3.0                                                   # an isolated 3-ppm band: a step
    r[i45] += 30.0                                                  # a gross outlier band
    measured = ~((k >= 50) & (k <= 56))                             # a 7-band gap: no window reaches across
    tab = AF.step_table(k, AF.s_from_r(r), measured)
    d, w = tab["d"], tab["weights"]
    assert abs(d[i30] - 3.0) < STEP_TOL and w[i30] == 0.0
    assert abs(d[i45] - 30.0) < STEP_TOL and w[i45] == 0.0
    rest = measured.copy()
    rest[[i30, i45]] = False
    assert np.abs(d[rest]).max() < STEP_TOL, np.abs(d[rest]).max()
    assert np.array_equal(tab["measured"], measured)                # every measured band keeps a trend
    assert not tab["measured"][~measured].any() and np.all(d[~measured] == 0.0) and np.all(w[~measured] == 0.0)
    assert np.isfinite(tab["trend"][~measured]).all()               # the gap's trend is still evaluated (record)
    # the same table through a document: interpolated bands are the gap
    doc = _document(steps={30: 3.0, 45: 30.0}, interp=tuple(range(50, 57)), seed=seed)
    fp = AF.load_document(doc, polarity="-").fingerprint
    assert fp is not None and abs(fp.d[_at(fp, 30)] - 3.0) < STEP_TOL and abs(fp.d[_at(fp, 45)] - 30.0) < STEP_TOL
    assert fp.n_measured == 48 - 7 and fp.d[_at(fp, 53)] == 0.0 and not fp.measured[_at(fp, 53)]


def _at_k(k, value) -> int:
    return int(np.flatnonzero(np.asarray(k) == value)[0])


def test_the_robust_scale_is_1_4826_times_the_median_absolute_residual_floored_at_a_tenth_of_a_ppm():
    """The pinned convention: sigma = max(1.4826 x median|res|, 0.1 ppm). Above the floor (a
    0.3-ppm table) the 1.4826 scale keeps a band 1 ppm off as smoothed where the raw
    median absolute residual zeroes it; under the floor (a 0.05-ppm table, or a served
    table smooth to its rounding) the floor is the scale, so no band within half a ppm of
    the local trend is ever discarded. Without the floor the scale collapses onto the
    rounding and an isolated step is attributed to its neighbours. The test fails if a
    constant or its use changes."""
    assert AF.MAD_TO_SIGMA == 1.4826 and AF.BISQUARE_C == 4.685 and AF.LOESS_ITERATIONS == 3 and AF.H_BANDS == 6
    assert AF.SIGMA_FLOOR_PPM == 0.1 and not hasattr(AF, "SIGMA_ZERO_PPM")
    k = np.arange(18, 66)
    i = _at_k(k, 40)
    # above the floor: the scale is 1.4826 x the median absolute residual of the previous fit, and
    # the convention matters -- a band 1 ppm off a 0.3-ppm table is smoothed under it, zeroed under
    # the raw median
    noisy = _cubic(k) + 0.3 * _band_noise(4)[k]
    noisy[i] += 1.0
    first = AF.loess_trend(k, noisy, iterations=0)["trend"]
    one = AF.loess_trend(k, noisy, iterations=1)
    assert one["scatter_ppm"] > AF.SIGMA_FLOOR_PPM and one["sigma_ppm"] == one["scatter_ppm"]
    assert abs(one["sigma_ppm"] - 1.4826 * np.median(np.abs(noisy - first))) < 1e-12
    assert abs(AF.loess_trend(k, noisy, iterations=1, scale=1.0)["sigma_ppm"] - np.median(np.abs(noisy - first))) < 1e-12
    pinned = AF.loess_trend(k, noisy)
    explicit = AF.loess_trend(k, noisy, scale=AF.MAD_TO_SIGMA)
    assert np.array_equal(pinned["trend"], explicit["trend"]) and np.array_equal(pinned["weights"], explicit["weights"])
    assert pinned["weights"][i] > 0.3, pinned["weights"][i]
    assert AF.loess_trend(k, noisy, scale=1.0)["weights"][i] == 0.0
    # under the floor: the floor is the scale (the scale convention no longer matters), and a
    # 0.16-ppm band on a 0.05-ppm table is smoothed, not discarded
    r = _cubic(k) + NOISE_PPM * _band_noise(4)[k]
    r[i] += 0.16
    lo = AF.loess_trend(k, r)
    assert lo["sigma_ppm"] == AF.SIGMA_FLOOR_PPM and 0.0 < lo["scatter_ppm"] < AF.SIGMA_FLOOR_PPM
    assert lo["weights"][i] > 0.8 and np.array_equal(lo["weights"], AF.loess_trend(k, r, scale=1.0)["weights"])
    # a table exact to rounding: every weight 1, the quadratic recovered, the scatter 0
    quad = 0.8 - 0.05 * (k - 40) + 0.002 * (k - 40.0) ** 2
    exact = AF.loess_trend(k, quad)
    assert np.all(exact["weights"] == 1.0) and exact["sigma_ppm"] == AF.SIGMA_FLOOR_PPM and exact["scatter_ppm"] < 1e-9
    assert np.abs(exact["trend"] - quad).max() < 1e-9
    # what the floor does: a served table (s to four decimals) with one 3-ppm band. Without the
    # floor the scale collapses to the rounding and the step's NEIGHBOURS are discarded with it (on
    # served documents whole windows then lose their trend and d differs between documents of one
    # calibration); with the floor the step band alone carries weight 0, and it carries the step
    rounded = AF.r_from_s(np.round(AF.s_from_r(quad + np.where(k == 40, 3.0, 0.0)), 4))
    bare = AF.loess_trend(k, rounded, floor=0.0)
    assert bare["sigma_ppm"] < 1e-3 and int((bare["weights"] == 0).sum()) > 3
    floored = AF.loess_trend(k, rounded)
    assert np.flatnonzero(floored["weights"] == 0).tolist() == [i]
    assert abs((rounded - floored["trend"])[i] - 3.0) < 0.01


@pytest.mark.parametrize("noise", [0.0, 0.0005, 0.001, 0.002, 0.005, 0.05])
@pytest.mark.parametrize("step", [3.0, 30.0])
def test_an_isolated_step_lands_on_its_own_band_whatever_the_tables_scatter(noise, step):
    """The collapsing scale, floored. Served tables scatter by 1e-4 to 0.05 ppm about their
    trend, and at the low end an isolated step used to be attributed to its NEIGHBOURS
    (d = 0 on the step band, -0.86 x the step beside it, the gate passing). Now the step band
    alone carries it -- on the second band, in the interior, on the second-last band, for
    any scatter and any step size; on the first and last band too, unless the scatter
    reaches 0.05 ppm (the end-band limit, see the known-limit test)."""
    k = np.arange(18, 68).astype(float)
    base = 0.8 - 0.05 * (k - 40) + 0.002 * (k - 40) ** 2
    positions = [19, 40, 66] + ([18, 67] if noise < 0.05 else [])
    for seed, pos in enumerate(positions):
        r = base + noise * _band_noise(10 + seed)[k.astype(int)]
        i = _at_k(k, pos)
        r[i] += step
        lo = AF.loess_trend(k, r)
        d = r - lo["trend"]
        assert np.isfinite(d).all(), (noise, step, pos)
        assert abs(d[i] - step) < 0.3, (noise, step, pos, d[i])
        assert np.abs(np.delete(d, i)).max() < 0.3, (noise, step, pos, np.abs(np.delete(d, i)).max())
        assert lo["weights"][i] == 0.0 and np.delete(lo["weights"], i).min() > 0.5, (noise, step, pos)
        assert lo["sigma_ppm"] == AF.SIGMA_FLOOR_PPM


def test_a_served_table_rounded_to_four_decimals_measures_its_step():
    """A document as Mascope serves it: s to four decimals, a trend smooth to that rounding
    (scatter 1e-4 ppm), one 3-ppm band. The floor is the scale; the step is its band's."""
    k_step = 40
    fp = AF.load_document(_document(steps={k_step: 3.0}, rounded=True, noise=0.0), polarity="-").fingerprint
    assert fp is not None
    i = _at(fp, k_step)
    assert abs(fp.d[i] - 3.0) < 0.01 and fp.weights[i] == 0.0
    rest = np.ones(len(fp.d), bool)
    rest[i] = False
    assert np.abs(fp.d[rest]).max() < 0.01 and fp.weights[rest].min() > 0.9
    assert fp.meta["sigma_ppm"] == AF.SIGMA_FLOOR_PPM and fp.meta["scatter_ppm"] < 1e-3


def test_a_run_of_gross_steps_is_measured_as_steps_and_never_comes_back_as_trend():
    """Six adjacent bands 40 ppm off (a broken calibration's staircase). They and their
    neighbours lose their weight in the first pass; since a band without a trend keeps its
    weight (never reset to 1), the plateau does not return as the trend: every plateau
    band is measured as its own 40 ppm (or left unmeasured), none is trusted as trend, and
    no other band carries a fake step."""
    k = np.arange(18, 68).astype(float)
    r = 0.8 - 0.05 * (k - 40) + 0.002 * (k - 40) ** 2 + 0.001 * _band_noise(3)[k.astype(int)]
    plateau = (k >= 35) & (k <= 40)
    r[plateau] += 40.0
    tab = AF.step_table(k, AF.s_from_r(r), np.ones(len(k), bool))
    d, meas, w = tab["d"], tab["measured"], tab["weights"]
    for i in np.flatnonzero(plateau):
        assert (not meas[i] and d[i] == 0.0) or abs(d[i] - 40.0) < 0.5, (k[i], meas[i], d[i])
    assert not (meas[plateau] & (w[plateau] > 0)).any()
    assert np.abs(d[~plateau]).max() < 0.5 and meas[~plateau].sum() >= 30


def test_the_known_limit_a_small_step_on_an_end_band_reads_as_trend():
    """The two end bands have a one-sided window and a leverage near 0.85: a step there
    under ~3 ppm leaves a first-pass residual under the 0.47-ppm cut-off and is followed
    as trend (d near 0, weight kept), with at most half a ppm left on its neighbour. Pinned
    so the limit is known: a leave-one-out residual would measure it but extrapolated the
    curved top end of real tables by up to 7 ppm."""
    k = np.arange(18, 68).astype(float)
    base = 0.8 - 0.05 * (k - 40) + 0.002 * (k - 40) ** 2
    for pos, seed in ((18, 0), (67, 1)):
        r = base + 0.001 * _band_noise(20 + seed)[k.astype(int)]
        i = _at_k(k, pos)
        r[i] += 1.0
        lo = AF.loess_trend(k, r)
        d = r - lo["trend"]
        assert lo["weights"][i] > 0.3 and abs(d[i]) < 0.6, (pos, lo["weights"][i], d[i])
        assert np.abs(np.delete(d, i)).max() < 0.6, pos


def test_the_known_limit_the_end_band_threshold_rises_beside_a_short_gap_and_with_the_first_pass_scale():
    """KNOWN LIMIT (1) above ~3 ppm. A lone first band beside a one-band gap has more
    leverage than a contiguous run's end band: a 5-ppm step there is followed as trend
    (weight near 1, d near 0) and the run past the gap takes about -0.4 x the step, in the
    pooled table too, while the same step on a contiguous end band is measured as itself.
    Gross steps elsewhere raise the first pass's scale above the floor and the threshold
    with it: four bands 40-49 ppm off, 14 bands away, hide a 5-ppm step on the first band
    past a gap that the same table without them measures, ~3 ppm landing beside it.
    Pinned so the limit is known."""
    for off in (5.0, -5.0):
        fp = AF.load_document(_document(steps={18: off}, item=_item("s0")), polarity="-").fingerprint
        assert fp.weights[0] == 0.0 and abs(fp.d[0] - off) < STEP_TOL       # contiguous: seen
        doc = _document(steps={18: off}, interp=(19,), item=_item("s0"))
        fp = AF.load_document(doc, polarity="-").fingerprint
        assert fp.measured[0] and fp.weights[0] > 0.9 and abs(fp.d[0]) < 0.1  # beside a gap: trend
        frame, rec = AF.FrameSet.from_documents({"s0": doc}, polarity="-")
        assert frame is not None, rec.get("held")
        pooled = {b["k"]: b for b in rec["groups"][0]["bands"]}
        assert pooled[18]["measured"] and abs(pooled[18]["d_ppm"]) < 0.1
        assert 1.5 < -np.sign(off) * pooled[20]["d_ppm"] < 2.6, pooled[20]    # the wrong d past the gap
    k = np.arange(18, 66).astype(float)
    gap = ~((k >= 21) & (k <= 30))
    i31 = _at_k(k, 31)
    near = (k >= 32) & (k <= 37)
    for gross in ({}, {45: 40.0, 46: -43.0, 47: 46.0, 48: -49.0}):
        r = _cubic(k) + 0.02 * _band_noise(0)[k.astype(int)]
        r[i31] += 5.0
        for kk, a in gross.items():
            r[_at_k(k, kk)] += a
        f0 = AF._local_fit(k[gap], r[gap], np.ones(int(gap.sum())), k[gap], h=AF.H_BANDS,
                           span=AF.TRICUBE_SPAN_BANDS)[0]
        sigma0 = max(AF.MAD_TO_SIGMA * float(np.nanmedian(np.abs(r[gap] - f0))), AF.SIGMA_FLOOR_PPM)
        tab = AF.step_table(k, AF.s_from_r(r), gap)
        for kk, a in gross.items():
            assert abs(tab["d"][_at_k(k, kk)] - a) < STEP_TOL                  # each its own step
        if not gross:
            assert sigma0 == AF.SIGMA_FLOOR_PPM
            assert tab["weights"][i31] == 0.0 and abs(tab["d"][i31] - 5.0) < STEP_TOL
            assert np.abs(tab["d"][near]).max() < STEP_TOL
        else:
            assert sigma0 > 0.2, sigma0
            assert tab["weights"][i31] > 0.9 and abs(tab["d"][i31]) < 0.1
            assert np.abs(tab["d"][near]).max() > 2.5


#: a broken calibration at one end of the table: a trend-less staircase of eight bands
#: (gross steps no local quadratic follows), then a short run whose trend bends towards it,
#: one large real band step, and a clean run
STAIRCASE = {58: 60.0, 59: -40.0, 60: 80.0, 61: -20.0, 62: 100.0, 63: 0.0, 64: 120.0, 65: 20.0}
BEND = {57: 3.5, 56: 2.0, 55: 1.2}
K_REAL = 54


@pytest.mark.parametrize("noise", [0.001, NOISE_PPM])
@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_a_trend_read_from_one_side_only_confirms_a_band_but_never_makes_it_a_step(noise, seed):
    """The real step zeroes the weights of the bent bands beside it in the first pass, the
    staircase leaves no band on their other side, and their trend is a line or quadratic
    extrapolated from the clean run over two bands or more: it read them as steps of 1-3.5
    ppm with nothing to check them against. Now such a band gets no d (extrapolated,
    unmeasured, not pooled) while the step band beside the clean run keeps its d (one band
    past the support: the reading that refused it), and nothing else moves."""
    doc = _document(steps={**STAIRCASE, **BEND, K_REAL: -20.0}, noise=noise, seed=seed, item=_item("s0"))
    loaded = AF.load_document(doc, polarity="-")
    assert loaded.ok, loaded.why
    fp = loaded.fingerprint
    bent = [_at(fp, k) for k in BEND]
    assert not fp.measured[[_at(fp, k) for k in STAIRCASE]].any()       # the staircase keeps no trend
    assert (np.abs((fp.r - fp.trend)[bent]) > 0.5).all()              # their trend reads them as steps ...
    assert not fp.measured[bent].any() and np.all(fp.d[bent] == 0.0)   # ... and they get no d
    assert fp.meta["extrapolated_k"] == sorted(BEND) and loaded.as_dict()["n_extrapolated"] == len(BEND)
    i = _at(fp, K_REAL)
    assert fp.measured[i] and fp.weights[i] == 0.0 and abs(fp.d[i] + 20.0) < 0.3
    rest = fp.measured.copy()
    rest[i] = False
    assert np.abs(fp.d[rest]).max() < STEP_TOL and rest.sum() == 48 - len(STAIRCASE) - len(BEND) - 1
    # pooled: a member contributes only the bands its own table reads a d on
    frame, rec = AF.FrameSet.from_documents({"s0": doc}, polarity="-")
    assert frame is not None, rec.get("held")
    bands = {b["k"]: b for b in rec["groups"][0]["bands"]}
    for k in BEND:
        assert not bands[k]["measured"] and bands[k]["n_measuring"] == 0 and bands[k]["d_ppm"] == 0.0
    assert bands[K_REAL]["measured"] and abs(bands[K_REAL]["d_ppm"] + 20.0) < 0.3
    assert rec["items"]["s0"]["extrapolated_k"] == sorted(BEND) and rec["groups"][0]["n_extrapolated"] == 0
    it = frame.for_item("s0")
    for k in BEND:
        lo, hi = AF.band_edges_mz(k, it.B, it.C, it.period, it.phase)
        assert np.all(it.d_of(np.linspace(lo, hi, 50)[5:-5]) == 0.0)  # the bent bands' peaks stay where served
    json.dumps(rec)


def test_the_reach_rule_leaves_an_interior_step_a_plateau_and_an_end_band_step_as_they_were():
    """Where the trend is an interpolation, or reaches the band from the support beside
    it, d is exactly r - trend as before the rule: an isolated interior step, a plateau of
    three steps (its middle band two bands from either side's support), a gross step on
    the table's first or last band, and a band refused beside a gap whose extrapolated
    trend still confirms it (within the cut-off)."""
    k = np.arange(18, 68).astype(float)
    base = 0.8 - 0.05 * (k - 40) + 0.002 * (k - 40) ** 2 + 0.001 * _band_noise(5)[k.astype(int)]
    for steps, measured in (({40: 3.0}, None), ({40: 3.0, 41: 3.0, 42: 3.0}, None), ({18: 10.0}, None),
                            ({67: -10.0}, None), ({40: 3.0}, ~((k >= 41) & (k <= 47)))):
        r = base.copy()
        for kk, a in steps.items():
            r[k == kk] += a
        meas = np.ones(len(k), bool) if measured is None else measured
        tab = AF.step_table(k, AF.s_from_r(r), meas)
        lo = AF.loess_trend(k[meas], tab["r"][meas], k_eval=k)
        assert not tab["extrapolated"].any() and np.array_equal(tab["measured"], meas), steps
        assert np.array_equal(tab["d"][meas], (tab["r"] - lo["trend"])[meas]), steps
        for kk, a in steps.items():
            i = _at_k(k, kk)
            assert tab["weights"][i] == 0.0 and abs(tab["d"][i] - a) < 0.05, (steps, kk, tab["d"][i])
    # the gross-outlier test's table (above): the band beside its 7-band gap had no trend in the
    # pass before the last and kept weight 0; its trend, two bands from the support, confirms it
    k2 = np.arange(18, 66)
    r2 = _cubic(k2) + NOISE_PPM * _band_noise(0)[k2]
    r2[_at_k(k2, 45)] += 30.0
    gap = ~((k2 >= 50) & (k2 <= 56))
    lo2 = AF.loess_trend(k2[gap], r2[gap], k_eval=k2)
    i49 = _at_k(k2, 49)
    assert lo2["reach"][i49] == 2.0 and not lo2["two_sided"][i49]
    tab2 = AF.step_table(k2, AF.s_from_r(r2), gap)
    assert tab2["measured"][i49] and not tab2["extrapolated"].any() and abs(tab2["d"][i49]) < 0.47


@pytest.mark.parametrize("off", [30.0, -30.0])
@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_a_band_without_a_trend_of_its_own_is_no_support(off, seed):
    """A lone band 30 ppm off its trend on the table's first band, then a two-band gap,
    then a clean run: once its neighbours lose their weight the lone band's window holds
    too few bands for a trend, so it keeps the weight an earlier pass's fit set (a fit
    through it in which its own leverage hid most of its offset). Counted as support it
    made the run's first bands two-sided and put ~17 ppm on the first of them in the
    document's own table. It is fitted through but is no support: those bands are read
    from one side only and extrapolated, the rest of the run keeps at most the ~2 ppm a
    30-ppm band's pull leaves (KNOWN LIMIT (5)), and the pooled table, which leaves the
    lone band out, applies nothing on the run."""
    k = np.arange(18, 66).astype(float)
    r = _cubic(k) + NOISE_PPM * _band_noise(seed)[k.astype(int)]
    r[0] += off
    meas = ~((k >= 19) & (k <= 20))
    lo = AF.loess_trend(k[meas], r[meas], k_eval=k)
    assert 0.9 < lo["weights"][0] < 1.0 and np.isnan(lo["trend"][0])    # an earlier pass's weight, no trend
    i21 = _at_k(k, 21)
    assert not lo["two_sided"][i21] and lo["reach"][i21] >= 2.0
    tab = AF.step_table(k, AF.s_from_r(r), meas)
    run = meas & (k > 18)
    assert not tab["measured"][0] and tab["d"][0] == 0.0
    assert tab["extrapolated"][i21] and tab["d"][i21] == 0.0
    assert np.abs(tab["d"][run]).max() < 2.5, np.abs(tab["d"][run]).max()
    doc = _document(steps={18: off}, interp=(19, 20), seed=seed, item=_item("s0"))
    fp = AF.load_document(doc, polarity="-").fingerprint
    assert fp is not None and fp.h_max < 2.5 and 21 in fp.meta["extrapolated_k"]
    frame, rec = AF.FrameSet.from_documents({"s0": doc}, polarity="-")
    assert frame is not None, rec.get("held")
    pooled = {b["k"]: b for b in rec["groups"][0]["bands"]}
    assert not pooled[18]["measured"] and pooled[18]["n_measuring"] == 0
    assert max(abs(b["d_ppm"]) for kk, b in pooled.items() if kk > 20) < STEP_TOL


def test_small_windows_fall_back_to_a_line_and_tiny_ones_to_no_trend():
    k = np.array([10, 11, 12, 13, 40, 41, 42, 43, 44, 45, 46, 47, 48], dtype=float)
    lo = AF.loess_trend(k, 0.1 * k)
    assert lo["n_window"].tolist() == [4, 4, 4, 4, 7, 8, 9, 9, 9, 9, 9, 8, 7]
    assert lo["degree"].tolist() == [1] * 4 + [2] * 9 and np.abs(lo["trend"] - 0.1 * k).max() < 1e-9
    k = np.array([10, 11, 30, 31, 32, 33, 34, 35, 36, 37], dtype=float)
    lo = AF.loess_trend(k, 0.1 * k)
    assert np.isnan(lo["trend"][:2]).all() and np.isfinite(lo["trend"][2:]).all() and lo["n_window"][:2].tolist() == [2, 2]
    tab = AF.step_table(k, AF.s_from_r(0.1 * k), np.ones(len(k), bool))
    assert tab["measured"].tolist() == [False, False] + [True] * 8 and tab["d"][0] == 0.0 and tab["weights"][0] == 0.0
    with pytest.raises(ValueError):
        AF.loess_trend(k[:0], k[:0])
    with pytest.raises(ValueError):
        AF.step_table(k, np.full(len(k), np.nan), np.ones(len(k), bool))


def test_d_is_zero_and_unmeasured_on_interpolated_and_weak_bands():
    km = K_MID                                                      # stepped; km + 1 (below it in m/z) SIDE_PPM
    i1, i2, wk = km + 5, km + 6, km + 15                            # interpolated, interpolated, weak
    steps = {km: STEP_PPM, km + 1: SIDE_PPM}
    doc = _document(steps=steps, interp=(i1, i2), weak=(wk,))
    fp, why = AF.load_document(doc, polarity="-")
    assert why is None and fp is not None
    at = lambda k: _at(fp, k)                                       # noqa: E731
    assert fp.d[at(i1)] == 0.0 and fp.d[at(i2)] == 0.0 and fp.d[at(wk)] == 0.0
    assert not fp.measured[[at(i1), at(i2), at(wk)]].any() and fp.interp[[at(i1), at(i2)]].all()
    assert abs(fp.d[at(km)] - STEP_PPM) < STEP_TOL and abs(fp.d[at(km + 1)] - SIDE_PPM) < STEP_TOL
    rest = np.ones(len(fp.d), bool)
    rest[[at(km), at(km + 1), at(i1), at(i2), at(wk)]] = False
    assert np.abs(fp.d[rest]).max() < STEP_TOL
    assert fp.n_measured == 48 - 3 and abs(fp.h_max - abs(STEP_PPM)) < STEP_TOL
    assert fp.n_hits == 38 and fp.edge_uncertainty_khz == 0.1 and fp.resolution == 280000
    steps = {s["k_above"]: s for s in fp.frame_item("s0").edge_steps()}
    # going up in m/z: into band km + 1 (SIDE_PPM), into band km (a step of STEP_PPM - SIDE_PPM),
    # out of it (-STEP_PPM)
    assert abs(steps[km + 1]["step_ppm"] - SIDE_PPM) < 2 * STEP_TOL
    assert abs(steps[km]["step_ppm"] - (STEP_PPM - SIDE_PPM)) < 2 * STEP_TOL
    assert abs(steps[km - 1]["step_ppm"] + STEP_PPM) < 2 * STEP_TOL
    assert steps[km]["measured"] and not steps[i1]["measured"]
    assert fp.meta["lock"] == {"m_lock_eff": 100.0, "s_lock_eff_ppm": -1.2}
    rec = AF.load_document(doc, polarity="-").as_dict()
    json.dumps(rec)
    assert rec["grid"]["n_hits"] == 38 and rec["grid"]["edge_uncertainty_khz"] == 0.1 and rec["scan_range"] == [50.0, 750.0]


# --- what is refused ------------------------------------------------------------------------

@pytest.mark.parametrize("doc, polarity, says", [
    (_document(grid_source="seed"), "-", "grid source 'seed'"),
    (_document(flags=("not_stepwise",)), "-", "not_stepwise"),
    (_document(flags=("grid_weak", "fft_step_not_power_of_two")), "-", "fft_step_not_power_of_two"),
    (_document(), "+", "polarity"),
    (_document(polarity="+"), "-", "polarity"),
    (_document(item=_item(streams=[{"stream_key": "a", "scans": [2], "mz_calibration_factor": 1.0},
                                   {"stream_key": "b", "scans": [3], "mz_calibration_factor": 1.0}],
                          stream_key="composite -")), "-", "stitched"),
    (_document(weak=tuple(range(18, 61))), "-", "5 measured bands, fewer than 8"),
    (_document(status="no_fingerprint"), "-", "status 'no_fingerprint'"),
    ({"schema": AF.SCHEMA, "version": 1, "status": "missing"}, "-", "status 'missing'"),
    ({"schema": AF.SCHEMA, "version": 1, "status": "error", "error": "X: y"}, "-", "status 'error' (X: y)"),
    (_document(version=2), "-", "version 2"),
    (_document(dominant="nothing"), "-", "not among the document's modes"),
    (_document(steps={30: 70.0}), "-", "ppm, above 60"),
    ({"schema": "something.else", "version": 1, "status": "ok"}, "-", "schema"),
    (None, "-", "no document"),
])
def test_an_unusable_document_is_refused_with_a_reason(doc, polarity, says):
    loaded = AF.load_document(doc, polarity=polarity)
    assert loaded.fingerprint is None and not loaded.ok
    assert says in loaded.why, loaded.why
    rec = loaded.as_dict()
    assert rec["usable"] is False and rec["why"] == loaded.why
    json.dumps(rec)


def test_grid_weak_is_advisory_and_the_item_route_picks_the_items_mode():
    other = _mode(polarity="-", phase=PHASE + 3.0, steps={33: 1.5}, mode_id="-|R70000|50.0000-750.0000",
                  flags=("grid_weak",), resolution=70000, n_hits=7, edge_uncertainty=None)
    doc = _document(modes=[other, _mode()], dominant=MODE, item=_item(mode_id=other["mode_id"]))
    loaded = AF.load_document(doc, polarity="-")
    assert loaded.ok and loaded.mode_id == other["mode_id"] and "grid_weak" in loaded.notes
    assert loaded.fingerprint.phase == PHASE + 3.0 and loaded.fingerprint.meta["grid_weak"]
    assert loaded.fingerprint.meta["lock"]["s_lock_eff_ppm"] == -1.25       # the item's lock, not the mode's
    assert loaded.fingerprint.meta["item"]["sample_item_id"] == "s0"
    assert loaded.fingerprint.n_hits == 7 and loaded.fingerprint.edge_uncertainty_khz == AF.DEFAULT_EDGE_UNCERTAINTY_KHZ
    # without an item block the file's dominant mode is read
    plain = AF.load_document(_document(modes=[other, _mode()], dominant=MODE), polarity="-")
    assert plain.ok and plain.mode_id == MODE and plain.fingerprint.phase == PHASE
    assert plain.sha1 != loaded.sha1 and len(plain.sha1) == 40


def _put(doc, path, value):
    node = doc
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    return doc


@pytest.mark.parametrize("path, name", [
    (("modes",), "modes"), (("flags",), "flags"), (("item", "streams"), "item.streams"),
    (("item", "scans"), "item.scans"), (("modes", 0, "flags"), "mode.flags"),
    (("modes", 0, "segments", "columns"), "segments.columns"), (("modes", 0, "segments", "rows"), "segments.rows")])
@pytest.mark.parametrize("value", [79, "79", {"n": 79}])
def test_a_field_of_an_unexpected_type_is_refused_with_a_reason_never_raised(path, name, value):
    """A field the schema gives as a list holding a count, a string or an object: the
    reader refuses the document with the field's name (the batch then holds the frame),
    it never raises out of the batch."""
    doc = _put(_document(item=_item("s0")), path, value)
    why = AF.document_shape_why(doc)
    assert why == f"malformed document: {name} is {type(value).__name__}, not a list"
    loaded = AF.load_document(doc, polarity="-")
    assert not loaded.ok and loaded.why == why and loaded.status == "ok" and len(loaded.sha1) == 40
    frame, rec = AF.FrameSet.from_documents({"s0": doc, "s1": _document(item=_item("s1"))}, polarity="-")
    assert frame is None and rec["held"] == f"item s0: {why}" and rec["items"]["s1"]["usable"]
    json.dumps(rec)


def test_a_band_index_or_a_number_no_table_has_is_refused_never_raised():
    good = _document(item=_item("s0"))
    assert AF.document_shape_why(good) is None and AF.load_document(good, polarity="-").ok
    assert AF.document_shape_why(_document(status="no_fingerprint")) is None      # segments None: the schema's own
    assert AF.document_shape_why([good]) == "not a document object"
    col = AF.SEGMENT_COLUMNS.index("k")
    for edit, says in ((lambda rows: [r.__setitem__(col, r[col] + 10 ** 300) for r in rows], "not a band table"),
                       (lambda rows: [r.__setitem__(col, r[col] - 10 ** 6) for r in rows], "not a band table"),
                       (lambda rows: rows[0].__setitem__(col, 10 ** 7), "not a band table"),
                       (lambda rows: rows[0].__setitem__(col, float("inf")), "k unreadable")):
        doc = json.loads(json.dumps(good))
        edit(doc["modes"][0]["segments"]["rows"])
        loaded = AF.load_document(doc, polarity="-")
        assert not loaded.ok and says in loaded.why, loaded.why
    # an integer too large for a float is an unreadable number, like a string
    doc = _put(json.loads(json.dumps(good)), ("modes", 0, "calibration", "B_median"), 10 ** 400)
    assert "B_median / C_median unreadable" in AF.load_document(doc, polarity="-").why


# --- calibration groups -------------------------------------------------------------------

STEPS = {K_MID: STEP_PPM, K_MID - 1: SIDE_PPM, _k_at(120.0): 1.1}


def _same_calibration_docs():
    """Two documents of ONE calibration (the same trend, steps and band noise) that differ
    the way two files of one calibration do: different low bands unmeasured (one band in
    neither), a different measured grid (phase 0.1 kHz apart, more hits in the longer
    file) and the lock-mass rescale (B, C 0.3 ppm apart)."""
    a = _document(steps=STEPS, interp=(60, 61), weak=(63,), item=_item("s0"), n_hits=20, edge_uncertainty=0.12)
    b = _document(steps=STEPS, interp=(58, 59, 61), phase=PHASE + 0.1, B=B0 * (1 + 3e-7), C=C0 * (1 + 3e-7),
                  item=_item("s1"), n_hits=36, edge_uncertainty=0.09)
    return {"s0": a, "s1": b}


def test_two_documents_of_one_calibration_pool_into_one_table_and_one_grid():
    docs = _same_calibration_docs()
    frame, rec = AF.FrameSet.from_documents(docs, polarity="-")
    assert frame is not None, rec.get("held")
    assert rec["n_groups"] == 1 and rec["groups"][0]["members"] == ["s0", "s1"] and "held" not in rec
    a, b = frame.for_item("s0"), frame.for_item("s1")
    assert a.d is b.d and a.measured is b.measured                 # one shared table
    assert (a.B, a.C, a.period, a.phase, a.k0, a.zone_khz) == (b.B, b.C, b.period, b.phase, b.k0, b.zone_khz)
    fa, fb = (AF.load_document(docs[s], polarity="-").fingerprint for s in ("s0", "s1"))
    # no band is left at d = 0 where either document measured it; 61 (in neither) stays unmeasured
    for k in (58, 59, 60, 63):
        assert a.measured[_at(a, k)] and (fa.measured[_at(fa, k)] != fb.measured[_at(fb, k)])
    assert not a.measured[_at(a, 61)] and a.d[_at(a, 61)] == 0.0
    g = rec["groups"][0]
    assert g["n_measured"] == 47 and g["n_unmeasured"] == 1 and g["n_bands"] == 48
    for k, step in STEPS.items():
        assert abs(a.d[_at(a, k)] - step) < STEP_TOL, (k, a.d[_at(a, k)])
    others = np.ones(len(a.d), bool)
    others[[_at(a, k) for k in STEPS] + [_at(a, 61)]] = False
    assert np.abs(a.d[others]).max() < STEP_TOL
    # the pooled grid: the heavier member's phase (weighted median of two), the zone = the larger
    # edge uncertainty (the phases are 0.1 apart: half of it is less)
    assert a.phase == PHASE + 0.1 and a.period == PERIOD and a.zone_khz == 0.12
    assert g["grid"]["phase_khz"] == PHASE + 0.1 and g["grid"]["zone_khz"] == 0.12 and g["grid"]["reference_member"] == 1
    assert abs(a.B - B0 * (1 + 1.5e-7)) < 1.0 and g["B"] == a.B
    # the per-band record
    row = next(r for r in g["bands"] if r["k"] == K_MID)
    assert row["measured"] and row["n_measuring"] == 2 and row["weight"] == 0.0
    assert abs(row["d_ppm"] - STEP_PPM) < STEP_TOL
    assert row["s_spread_ppm"] == 0.0 and row["mz_lo"] < STEP_MZ < row["mz_hi"]
    assert {"k", "mz_lo", "mz_hi", "s_ppm", "r_ppm", "trend_ppm", "weight", "d_ppm", "measured"} <= set(row)
    assert g["s_spread_max_ppm"] == 0.0 and g["h_max_ppm"] == round(a.h_max, 4)
    json.dumps(rec)
    json.dumps(frame.describe())
    assert frame.describe()["items"]["s1"]["document_grid"]["phase_khz"] == PHASE + 0.1
    # identical corrections for both items
    m = np.linspace(60.0, 700.0, 5001)
    assert np.array_equal(a.to_W(m), b.to_W(m)) and np.array_equal(a.predict(m), b.predict(m), equal_nan=True)


def test_two_calibrations_in_one_batch_hold_the_frame_with_the_groups_recorded():
    docs = _same_calibration_docs()
    other = _document(steps={**STEPS, 30: 2.0, 31: 1.8}, item=_item("s2"), seed=3)   # another day's calibration
    frame, rec = AF.FrameSet.from_documents(dict(docs, s2=other), polarity="-")
    assert frame is None and rec["held"] == "2 calibration groups in one batch"
    assert rec["n_groups"] == 2 and [g["members"] for g in rec["groups"]] == [["s0", "s1"], ["s2"]]
    assert "s differs by" in rec["groups"][1]["separated_by"] and "s2" in rec["groups"][1]["separated_by"]
    assert rec["groups"][1]["n_members"] == 1 and rec["n_usable"] == 3
    json.dumps(rec)
    # the members of one calibration still agree, and the stranger agrees with neither
    fps = {s: AF.load_document(d, polarity="-").fingerprint for s, d in dict(docs, s2=other).items()}
    assert AF.same_calibration(fps["s0"], fps["s1"])[0]
    assert not AF.same_calibration(fps["s0"], fps["s2"])[0] and not AF.same_calibration(fps["s1"], fps["s2"])[0]
    groups = AF.group_fingerprints(fps)
    assert [g.members for g in groups] == [["s0", "s1"], ["s2"]] and groups[1].index == 1


def test_same_calibration_needs_the_same_mode_law_and_s_table():
    base = AF.load_document(_document(steps=STEPS), polarity="-").fingerprint
    cases = [
        (_document(steps=STEPS, resolution=140000, mode_id="-|R140000|50.0000-750.0000"), "resolution"),
        (_document(steps=STEPS, scan_range=(50.0, 1000.0), mode_id="-|R280000|50.0000-1000.0000"), "scan range"),
        (_document(steps=STEPS, c_over_b2=C0 / B0 ** 2 + 1e-10), "C/B^2 differ"),
        (_document(steps={**STEPS, 50: 0.4}), "s differs by 0.6"),          # 0.4 ppm in r is 0.6 in s (s ~ -1.5 r)
        (_document(steps=STEPS, k0=45, k1=65, weak=tuple(range(45, 50)), interp=()), None),   # bands 50-65 only
    ]
    for doc, says in cases[:-1]:
        other = AF.load_document(doc, polarity="-").fingerprint
        ok, why = AF.same_calibration(base, other)
        assert not ok and says in why, why
    # no common measured band: cannot be told the same
    low = AF.load_document(_document(steps=STEPS, k0=18, k1=40), polarity="-").fingerprint
    high = AF.load_document(cases[-1][0], polarity="-").fingerprint
    ok, why = AF.same_calibration(low, high)
    assert not ok and "no band measured in both" in why
    # a band 0.15 ppm (in r; 0.22 in s) off between two documents is still one calibration
    near = AF.load_document(_document(steps={**STEPS, 50: 0.15}), polarity="-").fingerprint
    ok, how = AF.same_calibration(base, near)
    assert ok and "common bands" in how
    # a C/B^2 within tolerance (a pure rescale) is fine
    scaled = AF.load_document(_document(steps=STEPS, B=B0 * (1 + 2e-6), C=C0 * (1 + 2e-6)), polarity="-").fingerprint
    assert AF.same_calibration(base, scaled)[0]


def test_the_group_grid_is_the_hit_weighted_median_of_the_members_grids():
    fps = {s: AF.load_document(_document(steps=STEPS, phase=ph, n_hits=nh, edge_uncertainty=eu, item=_item(s)),
                               polarity="-").fingerprint
           for s, ph, nh, eu in (("a", PHASE, 38, 0.1), ("b", PHASE + 0.2, 20, 0.15), ("c", PHASE + 0.16, 30, 0.08))}
    g = AF.pool_grid([fps["a"], fps["b"], fps["c"]])
    # sorted phases 8.14 (38), 8.30 (30), 8.34 (20): the cumulative weight passes half at 8.30
    assert abs(g["phase_khz"] - (PHASE + 0.16)) < 1e-12 and g["period_khz"] == PERIOD and g["reference"] == 0
    assert abs(g["zone_khz"] - 0.15) < 1e-12 and abs(g["phase_spread_khz"] - 0.2) < 1e-12
    two = AF.pool_grid([fps["a"], fps["b"]])
    assert two["phase_khz"] == PHASE and two["zone_khz"] == 0.15          # the heavier member; max(0.15, 0.1)
    # a phase spread wider than the members' uncertainties sets the zone
    wide = AF.load_document(_document(steps=STEPS, phase=PHASE + 0.6, n_hits=10, edge_uncertainty=0.1), polarity="-").fingerprint
    assert abs(AF.pool_grid([fps["a"], wide])["zone_khz"] - 0.3) < 1e-12
    assert AF.weighted_median([8.14, 8.34], [38, 20]) == 8.14 and AF.weighted_median([8.14, 8.34], [20, 38]) == 8.34
    assert AF.weighted_median([1.0, 2.0], [1, 1]) == 1.0 and AF.weighted_median([3.0, 1.0, 2.0], [1, 1, 1]) == 2.0
    with pytest.raises(ValueError):
        AF.weighted_median([1.0], [0.0])
    # three members, one group, one table
    frame, rec = AF.FrameSet.from_documents({s: _document(steps=STEPS, phase=ph, n_hits=nh, edge_uncertainty=eu,
                                                          item=_item(s))
                                             for s, ph, nh, eu in (("a", PHASE, 38, 0.1), ("b", PHASE + 0.2, 20, 0.15),
                                                                   ("c", PHASE + 0.16, 30, 0.08))}, polarity="-")
    assert frame is not None and abs(frame.for_item("b").phase - (PHASE + 0.16)) < 1e-12
    assert len({id(it.d) for it in frame.items.values()}) == 1 and rec["groups"][0]["grid"]["zone_khz"] == 0.15


def test_the_pooled_table_takes_the_members_median_s_per_band():
    """Three documents of one calibration, the third 0.29 ppm off in s on one band (still
    within S_AGREE_PPM of both): the pooled band is the median of the three -- the two that
    agree -- not their mean."""
    docs = {s: _document(steps={**STEPS, 50: off}, item=_item(s)) for s, off in (("a", 0.0), ("b", 0.0), ("c", -0.19))}
    fps = {s: AF.load_document(d, polarity="-").fingerprint for s, d in docs.items()}
    i = _at(fps["a"], 50)
    diff = float(fps["c"].s[i] - fps["a"].s[i])
    assert 0.25 < diff < AF.S_AGREE_PPM and AF.same_calibration(fps["a"], fps["c"])[0]
    groups = AF.group_fingerprints(fps)
    assert len(groups) == 1 and groups[0].members == ["a", "b", "c"]
    g = groups[0]
    j = _at(g, 50)
    assert g.s[j] == fps["a"].s[i] == np.median([fp.s[i] for fp in fps.values()])
    assert abs(g.s[j] - np.mean([fp.s[i] for fp in fps.values()])) > 0.08 and abs(g.s_spread[j] - diff) < 1e-12
    others = np.ones(len(g.s), bool)
    others[j] = False
    assert np.allclose(g.s[others & g.measured], fps["a"].s[others & fps["a"].measured], rtol=0, atol=0)


def test_the_group_grid_period_is_the_hit_weighted_median_not_the_mean():
    periods = (PERIOD - 0.01, PERIOD, PERIOD + 0.02)
    fps = [AF.load_document(_document(steps=STEPS, period=pp, n_hits=nh, item=_item(s)), polarity="-").fingerprint
           for s, pp, nh in (("a", periods[0], 10), ("b", periods[1], 10), ("c", periods[2], 1))]
    g = AF.pool_grid(fps)
    assert g["period_khz"] == PERIOD                     # cumulative weight 10, 20, 21: half (10.5) falls on the second
    assert abs(g["period_khz"] - np.mean(periods)) > 1e-6
    assert AF.weighted_median(periods, [10, 10, 1]) == PERIOD and AF.weighted_median(periods, [1, 1, 10]) == periods[2]


def test_a_document_joins_a_group_only_when_it_agrees_with_every_member():
    """A, B and C with r at one band 0, +0.15 and +0.30: A-B and B-C agree within
    S_AGREE_PPM (0.225 in s), A-C do not (0.45). C agrees with B but not with A, so it
    starts its own group -- 'every member', not 'any member': a chain of near-agreements
    never merges two calibrations."""
    docs = {s: _document(steps={**STEPS, 50: off}, item=_item(s)) for s, off in (("a", 0.0), ("b", 0.15), ("c", 0.30))}
    fps = {s: AF.load_document(d, polarity="-").fingerprint for s, d in docs.items()}
    assert AF.same_calibration(fps["a"], fps["b"])[0] and AF.same_calibration(fps["b"], fps["c"])[0]
    assert not AF.same_calibration(fps["a"], fps["c"])[0]
    groups = AF.group_fingerprints(fps)
    assert [g.members for g in groups] == [["a", "b"], ["c"]]
    assert groups[1].meta["separated_by"].startswith("item c vs item a")
    frame, rec = AF.FrameSet.from_documents(docs, polarity="-")
    assert frame is None and rec["held"] == "2 calibration groups in one batch"


def test_group_hold_applies_the_two_limits_to_the_pooled_table(monkeypatch):
    """FrameSet.from_documents holds a group whose POOLED table has fewer than
    MIN_MEASURED_SEGMENTS bands with a trend or a step above H_MAX_PPM. The members pass
    the same limits one by one in load_document, so the pooled limits are reached here on
    groups built from hand-made fingerprints, and from_documents is shown to consult them."""
    fp = AF.load_document(_document(steps=STEPS, item=_item("a")), polarity="-").fingerprint
    assert AF.group_hold(AF.CalibrationGroup.from_fingerprints({"a": fp})) is None
    s = fp.s.copy()
    i = _at(fp, 33)
    s[i] = float(AF.s_from_r(float(AF.r_from_s(s[i])) + 70.0))
    broken = AF.CalibrationGroup.from_fingerprints({"a": replace(fp, s=s)})
    assert broken.h_max > AF.H_MAX_PPM and "above 60" in AF.group_hold(broken)
    few = AF.CalibrationGroup.from_fingerprints({"a": replace(fp, measured=fp.measured & (fp.ks <= 24))})
    assert few.n_measured == 7 and "7 measured bands, fewer than 8" in AF.group_hold(few)
    monkeypatch.setattr(AF, "group_hold", lambda g: "the pooled table: held for the test")
    frame, rec = AF.FrameSet.from_documents(_same_calibration_docs(), polarity="-")
    assert frame is None and rec["held"] == "the pooled table: held for the test" and rec["n_groups"] == 1


def test_grid_pooling_across_the_period_wrap_aligns_the_members_bands_by_frequency():
    """One calibration seen through two grids whose phases sit on either side of 0: A at
    16.20 kHz, B at 0.05 kHz (B's labels are one higher for the same physical band). The
    pooled grid keeps the heavier member's phase and B's bands land on A's labels."""
    a = _document(steps=STEPS, phase=16.20, n_hits=38, item=_item("a"))
    b = _document(steps=STEPS, phase=0.05, n_hits=20, k0=19, k1=66, phys=lambda k: k - 1, item=_item("b"))
    fa, fb = (AF.load_document(d, polarity="-").fingerprint for d in (a, b))
    assert fa.k0 == 18 and fb.k0 == 19
    ok, how = AF.same_calibration(fa, fb)
    assert ok, how
    assert AF.align_k(fb, fa.period, fa.phase).tolist() == list(range(18, 66))
    frame, rec = AF.FrameSet.from_documents({"a": a, "b": b}, polarity="-")
    assert frame is not None, rec.get("held")
    it = frame.for_item("b")
    assert it.phase == 16.20 and it.k0 == 18 and it.k1 == 65 and it.d is frame.for_item("a").d
    g = rec["groups"][0]["grid"]
    assert g["members"][1]["k_shift"] == -1 and g["members"][0]["k_shift"] == 0
    assert abs(g["members"][1]["phase_wrapped_khz"] - 16.33) < 1e-9 and abs(g["zone_khz"] - 0.1) < 1e-12
    for k, step in STEPS.items():
        assert abs(it.d[_at(it, k)] - step) < STEP_TOL
    assert rec["groups"][0]["n_measured"] == 48
    # the other way round: B heavier -> the pooled phase is B's (0.05) and A's labels shift up
    frame2, rec2 = AF.FrameSet.from_documents({"a": dict(a, modes=[dict(a["modes"][0], grid=dict(a["modes"][0]["grid"], n_hits=10))]),
                                               "b": b}, polarity="-")
    assert frame2 is not None, rec2.get("held")
    it2 = frame2.for_item("a")
    assert abs(it2.phase - 0.05) < 1e-12 and it2.k0 == 19 and rec2["groups"][0]["grid"]["members"][0]["k_shift"] == 1
    assert abs(it2.d[_at(it2, K_MID + 1)] - STEP_PPM) < STEP_TOL        # the same physical band, one label up
    assert np.allclose(it2.d, it.d, rtol=0, atol=1e-12)                  # one table, two labellings
    # the same map on both labellings away from the edges (the two grids' edges are 0.13 kHz
    # apart, so a peak between them sits in different bands under the two)
    m = np.linspace(60.0, 700.0, 3001)
    f = AF.law_khz(m, B0, C0)
    far = np.minimum(np.abs(((f - 16.20) / PERIOD) % 1.0), 1 - np.abs(((f - 16.20) / PERIOD) % 1.0)) * PERIOD > 0.3
    assert far.sum() > 2500 and np.allclose(it2.to_W(m[far]), it.to_W(m[far]), rtol=0, atol=1e-12)
    # the weighted median on the wrapped side: B (0.05 kHz, 30 hits) is the reference, two members at
    # 16.20 (20 hits each) wrap to -0.08 and carry the median -- the pooled phase is taken modulo the
    # period (16.20, labels as A's), never left negative
    three = {"b": dict(b, modes=[dict(b["modes"][0], grid=dict(b["modes"][0]["grid"], n_hits=30))]),
             "a": dict(a, modes=[dict(a["modes"][0], grid=dict(a["modes"][0]["grid"], n_hits=20))]),
             "c": dict(a, modes=[dict(a["modes"][0], grid=dict(a["modes"][0]["grid"], n_hits=20))], item=_item("c"))}
    frame3, rec3 = AF.FrameSet.from_documents(three, polarity="-")
    assert frame3 is not None, rec3.get("held")
    g3 = rec3["groups"][0]["grid"]
    assert g3["reference_member"] == 0 and abs(g3["members"][1]["phase_wrapped_khz"] + 0.08) < 1e-9
    it3 = frame3.for_item("b")
    assert 0.0 <= it3.phase < it3.period and abs(it3.phase - 16.20) < 1e-9 and it3.k0 == 18
    assert 0.0 <= g3["phase_khz"] < g3["period_khz"]


# --- one item's frame -------------------------------------------------------------------------

@pytest.fixture(scope="module")
def stepped_item() -> AF.FrameItem:
    """Steps of the size a calibration leaves on single bands: STEP_PPM on the band at m/z
    ~294-315, SIDE_PPM on the one above it, +1.1 on a low-m/z band; an interpolated band in
    between. Isolated bands, so the robust trend passes them by and d is the step itself."""
    doc = _document(steps=STEPS, interp=(_k_at(200.0),))
    loaded = AF.load_document(doc, polarity="-")
    assert loaded.ok, loaded.why
    return loaded.fingerprint.frame_item("s0")


def test_to_w_removes_each_bands_step_and_the_edge_zone_is_in_frequency(stepped_item):
    it = stepped_item
    lo, hi = it.reach
    m = np.linspace(lo + 0.5, hi - 0.5, 20001)
    w = it.to_W(m)
    d = it.d_of(m)
    assert np.allclose(w, m * (1 - d * 1e-6), rtol=0, atol=0)
    assert abs(it.d_of(STEP_MZ) - STEP_PPM) < STEP_TOL and abs(it.d_of(120.0) - 1.1) < STEP_TOL
    assert it.d_of(200.0) == 0.0
    assert it.to_W(hi + 50.0) == hi + 50.0 and it.to_W(lo - 5.0) == lo - 5.0     # off the table: untouched
    assert not it.measured_at(200.0) and it.in_reach(200.0) and not it.in_reach(hi + 50.0)
    # the zone: within zone_khz of an edge's frequency, on either side
    assert it.zone_khz == 0.1
    j = it.edge_indices[10]
    fe = it.phase + it.period * j
    at_f = lambda df: float(AF.mz_of_f(fe + df, B0, C0))               # noqa: E731
    assert it.edge_ambiguous([at_f(0.0), at_f(0.05), at_f(-0.05), at_f(0.099)]).all()
    assert not it.edge_ambiguous([at_f(0.11), at_f(-0.11), at_f(0.5 * it.period)]).any()
    assert np.isnan(it.to_W(np.nan)) and not it.edge_ambiguous([np.nan])[0]
    assert len(it.edges) == len(it.d) + 1 and np.all(np.diff(it.edges) > 0)
    assert len(it.zone_bounds) == 2 * len(it.edges) and np.all(np.diff(it.zone_bounds) > 0)
    assert abs(it.h_max - abs(STEP_PPM)) < STEP_TOL and abs(it.max_abs_ppm() - it.h_max) < 1e-9
    # the zone in ppm at an edge is 2 zone_khz / f: ~440 ppm at m/z 330
    e = float(it.edges[10])
    assert abs(float(it.zone_ppm_at(e)) - 2 * 0.1 / float(AF.law_khz(e, B0, C0)) * 1e6) < 1e-9


def test_a_peak_in_the_edge_zone_gets_the_mean_of_the_two_bands_and_inverts(stepped_item):
    it = stepped_item
    k_mid = K_MID                                                    # the STEP_PPM band; k_mid - 1 above it in m/z
    d_below, d_above = float(it.d[_at(it, k_mid)]), float(it.d[_at(it, k_mid - 1)])
    fe = it.phase + it.period * k_mid                                # the edge between them (edge index k_mid)
    peak = lambda df: float(AF.mz_of_f(fe + df, B0, C0))             # noqa: E731
    inside = np.array([peak(0.05), peak(-0.05), peak(0.0), peak(0.09)])
    outside = np.array([peak(0.15), peak(-0.15)])                    # +0.15 kHz: higher f = lower m/z = band k_mid
    assert it.edge_ambiguous(inside).all() and not it.edge_ambiguous(outside).any()
    assert np.allclose(it.d_of(inside), 0.5 * (d_below + d_above), rtol=0, atol=0)
    assert it.d_of(outside).tolist() == [d_below, d_above]
    # the W image: continuous across the edge, a half step at each zone boundary
    wi = it.to_W(inside)
    assert np.allclose(wi, inside * (1 - 0.5 * (d_below + d_above) * 1e-6), rtol=0, atol=0)
    # every one of them inverts exactly, from either side of the edge and from the zone
    allm = np.concatenate([inside, outside])
    back = it.invert(it.to_W(allm))
    assert np.max(np.abs(back - allm) / allm * 1e6) < 1e-6
    # counted when a series goes through the frame
    frame = AF.FrameSet(items={"s0": it.with_inner(None)})
    ts = pd.DataFrame({"sample_item_id": ["s0"] * 6, "mz": allm})
    frame.to_frame(ts)
    assert frame.stats["frame"]["s0"]["n_edge_ambiguous"] == 4 and frame.stats["frame"]["s0"]["n_moved"] == 6


def test_the_frame_alone_applies_and_inverts_exactly_everywhere_it_reaches(stepped_item):
    it = stepped_item
    raw = _dense(it)
    inside = it.in_scope(raw)
    assert inside.any() and not inside.all()
    corr = MQ.apply_correction(it, raw)
    assert np.array_equal(corr[~inside], raw[~inside])
    D = it.predict(raw)
    assert np.isnan(D[~inside]).all() and np.isfinite(D[inside]).all()
    assert np.allclose(corr[inside], raw[inside] * (1 - D[inside] * 1e-6), rtol=0, atol=0)
    back = MQ.invert_correction(it, corr)
    amb = _ambiguous(it, raw)
    ok = inside & ~amb
    assert ok.sum() > 0.9 * inside.sum()
    assert np.max(np.abs(back - raw)[ok] / raw[ok] * 1e6) < 1e-3
    assert np.array_equal(back[~inside & ~amb], raw[~inside & ~amb])
    # a point on an unmeasured band is in reach but moves nowhere
    assert not it.in_scope([200.0])[0] and MQ.apply_correction(it, [200.0])[0] == 200.0
    # where d steps DOWN going up in m/z (e.g. out of the STEP_PPM band) the map jumps UP at the zone
    # boundary below the edge (the band's d -> the mean): the images leave a GAP, and a
    # corrected value inside it has no preimage -- returned unchanged, counted
    down = next(s for s in it.edge_steps() if s["step_ppm"] < -1.0)
    j = int(down["k_below"])
    zb = float(AF.mz_of_f(it.phase + it.period * j + it.zone_khz, B0, C0))   # the zone boundary on the low-m/z side
    mean = 0.5 * (down["d_below"] + down["d_above"])
    gap = zb * (1 - 0.5 * (down["d_below"] + mean) * 1e-6)
    assert it.to_W([zb * (1 - 1e-9)])[0] < gap < it.to_W([zb * (1 + 1e-9)])[0]
    before = it.stats.get("invert_no_preimage", 0)
    assert it.invert([gap])[0] == gap and it.stats["invert_no_preimage"] == before + 1
    # where d steps UP (into the +1.1 band) the images OVERLAP at the zone boundary: two served
    # m/z share one corrected value, and the lower one is returned
    up = next(s for s in it.edge_steps() if s["step_ppm"] > 1.0)
    j = int(up["k_below"])
    zb = float(AF.mz_of_f(it.phase + it.period * j + it.zone_khz, B0, C0))
    twin_lo, twin_hi = zb * (1 - 0.2e-6), zb * (1 + 0.2e-6)            # below: the band's d; above: the mean
    img_lo, img_hi = it.to_W([twin_lo])[0], it.to_W([twin_hi])[0]
    assert img_hi < img_lo
    assert abs(it.invert([img_lo])[0] - twin_lo) / zb * 1e6 < 1e-9
    assert it.invert([img_hi])[0] < twin_lo and abs(it.to_W(it.invert([img_hi]))[0] - img_hi) / zb * 1e6 < 1e-9


def test_a_peak_exactly_on_a_zone_boundary_has_its_forward_image_as_a_preimage(stepped_item):
    """The inverse decides 'inside an edge zone' with the forward map's own test, so a
    served m/z placed on a zone boundary (to rounding) is found again whichever side the
    forward map put it on: no false 'no preimage' there."""
    it = stepped_item
    pts = []
    for s in it.edge_steps():
        j = int(s["k_below"])
        for df in (-it.zone_khz, it.zone_khz):
            fb = it.phase + it.period * j + df
            for eps in (-1e-12, 0.0, 1e-12):
                pts.append(float(AF.mz_of_f(fb + eps, B0, C0)))
    pts = np.array(pts)
    corr = it.to_W(pts)
    back, found, _ = it.preimage(corr)
    assert found.all()
    assert np.all(np.abs(it.to_W(back) - corr) / corr * 1e6 < 1e-9)


def test_the_composite_is_exact_and_falls_back_to_the_policy_outside_the_lock_model(stepped_item):
    inner = _spline_inner()
    raw_item = stepped_item.with_inner(inner)                       # OUTSIDE_POLICY 'raw'
    assert raw_item.policy == AF.OUTSIDE_POLICY == "raw"
    frame_item = stepped_item.with_inner(inner, policy="frame")
    lo, hi = raw_item.reach
    m = np.unique(np.concatenate([np.linspace(lo + 0.5, hi - 0.5, 40001), _dense(stepped_item, (80.0, 430.0))]))
    m_W = stepped_item.to_W(m)
    in_inner = inner.in_scope(m_W)
    for it in (raw_item, frame_item):
        scope = it.in_scope(m)
        D = it.predict(m)
        two_stage = m_W * (1 - np.where(in_inner, inner.predict(m_W, extrapolate=True), 0.0) * 1e-6)
        if it.policy == "raw":
            assert np.array_equal(scope, in_inner)
        else:
            assert np.array_equal(scope, in_inner | stepped_item.in_scope(m))
            alone = scope & ~in_inner
            assert alone.any() and np.allclose(D[alone], stepped_item.d_of(m[alone]), rtol=0, atol=0)
        assert np.isnan(D[~scope]).all()
        got = m[scope] * (1 - D[scope] * 1e-6)
        assert np.allclose(got, two_stage[scope], rtol=1e-12, atol=0)
        assert np.allclose(MQ.apply_correction(it, m), np.where(scope, two_stage, m), rtol=1e-12, atol=0)
        assert np.array_equal(MQ.apply_correction(it, m)[~scope], m[~scope])
    # outside the lock model's scope the 'raw' policy applies nothing, 'frame' the steps alone
    off = (~in_inner) & stepped_item.in_scope(m)
    assert off.any()
    assert np.array_equal(MQ.apply_correction(raw_item, m[off]), m[off])
    assert np.allclose(MQ.apply_correction(frame_item, m[off]), stepped_item.to_W(m[off]), rtol=0, atol=0)
    assert raw_item.mz_range == inner.mz_range and frame_item.mz_range == (min(lo, 80.0), max(hi, 430.0))
    expected = float(np.nanmax(np.abs(raw_item.predict(m))))
    assert expected > 2.0 and abs(raw_item.max_abs_ppm() - expected) < 1e-3


def test_the_composite_inverts_exactly_everywhere_it_reaches(stepped_item):
    inner = _spline_inner()
    for policy in ("raw", "frame"):
        it = stepped_item.with_inner(inner, policy=policy)
        raw = _dense(stepped_item, (80.0, 430.0))
        inside = it.in_scope(raw)
        corr = MQ.apply_correction(it, raw)
        back = MQ.invert_correction(it, corr)
        amb = _ambiguous(stepped_item, raw)
        near_inner = np.min(np.abs(raw[:, None] - np.array([80.0, 430.0])[None, :]), axis=1) < 0.01
        ok = inside & ~amb & ~near_inner
        assert ok.sum() > 0.9 * inside.sum()
        assert np.max(np.abs(back - raw)[ok] / raw[ok] * 1e6) < 1e-3, policy
        quiet = ~inside & ~amb & ~near_inner
        assert np.array_equal(corr[quiet], raw[quiet]) and np.array_equal(back[quiet], raw[quiet]), policy


def test_under_axis_ok_the_band_steps_apply_only_inside_the_w_fits_lock_domains(stepped_item):
    """The W verdict axis_ok leaves no smooth correction, but the composite must not reach
    beyond the locks: `scope_model` is the W fit's lock model with every segment a zero
    line over its domain (holes and the gap between segments kept), so the steps apply
    there alone and the scope is the lock range, not the table's reach (m/z ~55-750)."""
    w_fit = AL.AxisModel(segments=[
        {"kind": "spline", "lo": 90.0, "hi": 250.0, "n": 60, "x": np.linspace(90.0, 250.0, 50),
         "y": 0.3 * np.ones(50), "holes": [[150.0, 160.0]]},
        {"kind": "line", "lo": 270.0, "hi": 420.0, "n": 40, "a": 0.2, "b": 0.001}], n_locks=100)
    zero = AF.scope_model(json.loads(json.dumps(w_fit.as_dict())), [90.0, 420.0])
    assert [(s["kind"], s["lo"], s["hi"], s["a"], s["b"]) for s in zero.segments] == \
        [("line", 90.0, 250.0, 0.0, 0.0), ("line", 270.0, 420.0, 0.0, 0.0)]
    assert zero.segments[0]["holes"] == [[150.0, 160.0]] and zero.n_locks == 100 and zero.max_abs_ppm() == 0.0
    it = stepped_item.with_inner(zero)
    assert it.policy == "raw" and it.mz_range == (90.0, 420.0) == w_fit.mz_range
    lo, hi = stepped_item.reach
    m = np.unique(np.concatenate([np.linspace(lo + 0.5, hi - 0.5, 40001), _dense(stepped_item, (90.0, 420.0))]))
    inside = it.in_scope(m)
    assert np.array_equal(inside, w_fit.in_scope(stepped_item.to_W(m)))      # the W fit's own domains
    mid = (m > 92.0) & (m < 418.0) & ~((m > 148.0) & (m < 162.0)) & ~((m > 248.0) & (m < 272.0))
    assert inside[mid].all() and not inside[(m < 88.0) | (m > 422.0)].any()
    assert not inside[(m > 152.0) & (m < 158.0)].any() and not inside[(m > 252.0) & (m < 268.0)].any()
    D = it.predict(m)
    assert np.array_equal(D[inside], stepped_item.d_of(m[inside])) and np.isnan(D[~inside]).all()
    out = MQ.apply_correction(it, m)
    assert np.array_equal(out[~inside], m[~inside])                     # nothing moves outside the lock domains
    moved_without = stepped_item.with_inner(None).in_scope(m) & ~inside & (stepped_item.d_of(m) != 0.0)
    assert moved_without.sum() > 1000                                   # what the steps alone would have moved
    assert abs(it.max_abs_ppm() - float(np.abs(stepped_item.d_of(m[inside])).max())) < 1e-9
    # its records: JSON, the massqc dispatch and pickle give the same map; the inverse puts it back
    fs = AF.FrameSet(items={"s0": stepped_item}).with_inner(zero)
    back = MQ.fit_from_record(json.loads(json.dumps(fs.as_dict()))).for_item("s0")
    assert back.inner.segments[0]["holes"] == [[150.0, 160.0]] and back.mz_range == (90.0, 420.0)
    assert np.array_equal(back.in_scope(m), inside) and np.array_equal(back.predict(m), D, equal_nan=True)
    assert np.array_equal(pickle.loads(pickle.dumps(fs)).for_item("s0").predict(m), D, equal_nan=True)
    raw = _dense(stepped_item, (90.0, 420.0))
    ok = it.in_scope(raw) & ~_ambiguous(stepped_item, raw)
    assert np.max(np.abs(MQ.invert_correction(back, MQ.apply_correction(back, raw))[ok] - raw[ok]) / raw[ok]) < 1e-12
    # without a lock-model record: one zero line over the recorded scope; nothing usable: None
    one = AF.scope_model({}, [90.0, 420.0])
    assert [(s["lo"], s["hi"], s["a"], s["b"]) for s in one.segments] == [(90.0, 420.0, 0.0, 0.0)]
    assert AF.scope_model(None, None) is None and AF.scope_model({}, [420.0, 90.0]) is None
    assert AF.scope_model({"model": "locks", "segments": []}, None) is None


def test_the_inverse_tries_the_neighbouring_bands_where_a_big_step_crosses_an_edge():
    """A hand-built item with NO edge zone and a -300-ppm band: the image of a served m/z
    just below the band's upper edge lands in the band above, so its preimage is found
    only through the neighbouring band's d (k +- 1). That image also has a second
    preimage (a served m/z at the image itself, on the band above, d = 0): `several`,
    the lower m/z returned."""
    k0 = 30
    d = np.zeros(20)
    d[5] = -300.0
    it = AF.FrameItem(sample_item_id="x", B=B0, C=C0, period=PERIOD, phase=PHASE, k0=k0, d=d,
                      measured=np.ones(20, bool), zone_khz=0.0)
    lo, hi = (float(x) for x in AF.band_edges_mz(k0 + 5, B0, C0, PERIOD, PHASE))
    m = np.array([hi * (1 - 50e-6), hi * (1 - 200e-6), lo * (1 + 100e-6)])         # inside the -300-ppm band
    w = it.to_W(m)
    assert it.k_of(m).tolist() == [k0 + 5] * 3 and it.k_of(w).tolist() == [k0 + 4, k0 + 4, k0 + 5]
    back, found, several = it.preimage(w)
    assert found.all() and np.max(np.abs(back - m) / m) < 1e-12
    assert several.tolist() == [True, True, False]
    assert np.array_equal(it.invert(w), back)
    assert it.stats["invert_no_preimage"] == 0 and it.stats["invert_several_preimages"] == 2


def test_the_inverse_tries_the_band_on_the_other_side_where_a_big_positive_step_crosses_an_edge():
    """The mirror case: a +300-ppm band moves a served m/z just above its LOWER edge into
    the band below it in m/z (k + 1), so the preimage on the stepped band is found only
    through the k - 1 candidate. The image is also its own preimage on that d = 0 band:
    `several`, and the lower m/z -- the image itself -- is returned."""
    k0 = 30
    d = np.zeros(20)
    d[5] = 300.0
    it = AF.FrameItem(sample_item_id="x", B=B0, C=C0, period=PERIOD, phase=PHASE, k0=k0, d=d,
                      measured=np.ones(20, bool), zone_khz=0.0)
    lo, hi = (float(x) for x in AF.band_edges_mz(k0 + 5, B0, C0, PERIOD, PHASE))
    m = np.array([lo * (1 + 50e-6), lo * (1 + 200e-6), hi * (1 - 100e-6)])         # inside the +300-ppm band
    w = it.to_W(m)
    assert it.k_of(m).tolist() == [k0 + 5] * 3 and it.k_of(w).tolist() == [k0 + 6, k0 + 6, k0 + 5]
    back, found, several = it.preimage(w)
    assert found.all() and several.tolist() == [True, True, False]
    assert np.array_equal(back[:2], w[:2]) and abs(back[2] - m[2]) / m[2] < 1e-12
    assert np.max(np.abs(it.to_W(back) - w) / w) < 1e-12                       # each a preimage of its value


def test_a_broken_table_is_refused_by_h_max_and_a_bad_policy_by_the_frame():
    doc = _document(steps={30: -61.0})
    loaded = AF.load_document(doc, polarity="-")
    assert not loaded.ok and "ppm, above 60" in loaded.why
    ok = AF.load_document(_document(steps={30: -58.0}), polarity="-")
    assert ok.ok and abs(ok.fingerprint.h_max - 58.0) < STEP_TOL
    with pytest.raises(ValueError):
        ok.fingerprint.frame_item("s0", policy="nothing")
    with pytest.raises(ValueError):
        AF.FrameItem(sample_item_id="x", B=B0, C=C0, period=PERIOD, phase=PHASE, k0=18, d=[0.0], measured=[True],
                     zone_khz=-1.0)


# --- the batch's frames ---------------------------------------------------------------------

def test_a_frame_set_is_built_from_every_items_document_or_not_at_all():
    docs = _same_calibration_docs()
    frame, rec = AF.FrameSet.from_documents(docs, polarity="-")
    assert frame is not None and frame.sample_ids == ["s0", "s1"] and rec["n_usable"] == 2
    assert "held" not in rec and rec["items"]["s1"]["usable"] and abs(rec["items"]["s1"]["h_max_ppm"] - abs(STEP_PPM)) < STEP_TOL
    json.dumps(rec)
    bad = dict(docs, s1=_document(grid_source="seed", item=_item("s1")))
    frame2, rec2 = AF.FrameSet.from_documents(bad, polarity="-")
    assert frame2 is None and rec2["held"].startswith("item s1: grid source 'seed'") and rec2["n_usable"] == 1
    assert rec2["n_groups"] == 1 and rec2["groups"][0]["members"] == ["s0"]      # the usable ones are still grouped
    frame3, rec3 = AF.FrameSet.from_documents(dict(docs, s2=None), polarity="-")
    assert frame3 is None and rec3["held"] == "item s2: no document served"
    frame4, rec4 = AF.FrameSet.from_documents(docs, polarity="+")
    assert frame4 is None and "polarity" in rec4["held"]
    assert AF.FrameSet.from_documents({}, polarity="-") == (None, {"items": {}, "n_usable": 0, "n_items": 0,
                                                                   "policy": "raw", "n_groups": 0, "groups": [],
                                                                   "held": "no items"})
    # a pooled table over the limit is held with its reason
    broken = {s: _document(steps={**STEPS, 33: 59.0}, item=_item(s)) for s in ("s0", "s1")}
    frame5, rec5 = AF.FrameSet.from_documents(broken, polarity="-")
    assert frame5 is not None and rec5["groups"][0]["h_max_ppm"] > 58.0        # 59 passes per document and pooled
    broken["s1"] = _document(steps={**STEPS, 33: 59.0, 34: 59.0}, item=_item("s1"))
    frame6, rec6 = AF.FrameSet.from_documents(broken, polarity="-")
    assert frame6 is None and ("calibration groups" in rec6["held"] or "above 60" in rec6["held"])


def test_a_frame_set_maps_per_item_and_is_never_a_map_itself():
    frame, _ = AF.FrameSet.from_documents(_same_calibration_docs(), polarity="-")
    rng = np.random.default_rng(5)
    ts = pd.DataFrame({"sample_item_id": ["s0"] * 300 + ["s1"] * 300, "mz": rng.uniform(60, 700, 600),
                       "height": rng.uniform(10, 1000, 600)})
    out = frame.to_frame(ts)
    assert list(out.columns) == list(ts.columns) and len(out) == len(ts)
    for sid in ("s0", "s1"):
        sel = (ts["sample_item_id"] == sid).to_numpy()
        assert np.array_equal(out["mz"].to_numpy()[sel], frame.for_item(sid).to_W(ts["mz"].to_numpy()[sel]))
    s0, s1 = frame.stats["frame"]["s0"], frame.stats["frame"]["s1"]
    assert s0["n_peaks"] == 300 and s0["n_moved"] > 0 and abs(s0["d_ppm"]["max"] - STEP_PPM) < STEP_TOL
    assert abs(s1["d_ppm"]["min"]) < STEP_TOL and "n_edge_ambiguous" in s0
    assert np.array_equal(ts["mz"].to_numpy(), pd.DataFrame(ts)["mz"].to_numpy())  # the input is untouched
    for call in (lambda: frame.predict([100.0]), lambda: frame.in_scope([100.0]), lambda: frame.invert([100.0]),
                 lambda: MQ.apply_correction(frame, np.array([100.0])), lambda: MQ.in_scope(frame, [100.0])):
        with pytest.raises(TypeError):
            call()
    with pytest.raises(KeyError):
        frame.for_item("s9")
    with pytest.raises(KeyError):
        frame.to_frame(ts.assign(sample_item_id="s9"))
    assert abs(frame.h_max - abs(STEP_PPM)) < STEP_TOL and frame.mz_range[0] < 100 < frame.mz_range[1]
    desc = frame.describe()
    json.dumps(desc)
    assert desc["n_groups"] == 1 and desc["groups"][0]["members"] == ["s0", "s1"] and desc["items"]["s0"]["group"] == 0


def test_records_round_trip_through_json_pickle_and_the_massqc_dispatch():
    frame, _ = AF.FrameSet.from_documents(_same_calibration_docs(), polarity="-")
    with_inner = frame.with_inner(_spline_inner())
    assert with_inner.policy == "raw" and with_inner.for_item("s0").inner is with_inner.inner
    assert with_inner.groups == frame.groups and len(with_inner.groups) == 1
    # a set built from items that already share one model keeps it; mixed models are replaced
    kept = AF.FrameSet(items={sid: it for sid, it in with_inner.items.items()})
    assert kept.inner is with_inner.inner and kept.for_item("s1").inner is kept.inner
    mixed = AF.FrameSet(items={"s0": with_inner.for_item("s0").with_inner(_spline_inner()),
                               "s1": with_inner.for_item("s1").with_inner(_spline_inner(100.0, 400.0))})
    assert mixed.inner is None and mixed.for_item("s0").inner is None
    for fs in (frame, with_inner):
        rec = json.loads(json.dumps(fs.as_dict()))
        assert rec["model"] == "frame" and rec["version"] == 1 and set(rec["items"]) == {"s0", "s1"}
        assert rec["groups"][0]["members"] == ["s0", "s1"] and rec["items"]["s0"]["zone_khz"] == 0.12
        back = MQ.fit_from_record(rec)
        assert isinstance(back, AF.FrameSet) and back.policy == fs.policy and back.groups == rec["groups"]
        grid = np.linspace(60.0, 700.0, 20001)
        # the frame's own record is exact; a lock model's record keeps its knots to 1e-5 ppm
        tol = 0.0 if fs.inner is None else 1e-4
        for sid in fs.sample_ids:
            a, b = fs.for_item(sid), back.for_item(sid)
            assert np.allclose(a.predict(grid, extrapolate=True), b.predict(grid, extrapolate=True), rtol=0, atol=tol)
            assert np.array_equal(a.in_scope(grid), b.in_scope(grid)) and b.zone_khz == a.zone_khz
            assert b.inner is back.inner and b.policy == back.policy
        pickled = pickle.loads(pickle.dumps(fs))
        assert isinstance(pickled, AF.FrameSet) and pickled.sample_ids == fs.sample_ids
        assert np.array_equal(pickled.for_item("s1").predict(grid, extrapolate=True),
                              fs.for_item("s1").predict(grid, extrapolate=True))        # pickle is exact
        if fs.inner is not None:
            assert pickled.for_item("s0").inner is pickled.inner       # one shared model after unpickling
    item = with_inner.for_item("s0")
    rec = json.loads(json.dumps(item.as_dict()))
    one = MQ.fit_from_record(rec)
    assert isinstance(one, AF.FrameItem) and one.inner is not None and one.policy == "raw" and one.zone_khz == 0.12
    grid = np.linspace(60.0, 700.0, 2001)
    assert np.allclose(one.predict(grid, extrapolate=True), item.predict(grid, extrapolate=True), rtol=0, atol=1e-4)
    assert MQ.fit_from_record(with_inner) is with_inner and MQ.fit_from_record(item) is item
    # an older record without a zone reads with the default
    del rec["zone_khz"]
    assert AF.FrameItem.from_dict(rec).zone_khz == AF.DEFAULT_EDGE_UNCERTAINTY_KHZ
    locks = MQ.fit_from_record(json.loads(json.dumps(_spline_inner().as_dict())))
    assert isinstance(locks, AL.AxisModel)                              # the existing branch unchanged
    with pytest.raises(ValueError):
        AF.FrameSet.from_dict({"model": "locks"})


# --- the gate ---------------------------------------------------------------------------------

def _locks(mzs, ppms, bright=False):
    return pd.DataFrame({"mz": mzs, "ion": ["X-"] * len(mzs), "ppm": ppms, "how": ["unique +-5 ppm"] * len(mzs),
                         "bright": [bright] * len(mzs)})


def test_the_gate_c_needs_a_verdict_the_frame_may_be_applied_under(stepped_item):
    for v in ("locks_unpredictive", "locks_unvalidated", None, "axis_offset"):
        g = AF.gate(None, {"verdict": v, "locks": {"steps": []}}, None, stepped_item)
        assert not g["passed"] and g["failed"] == ["c"], (v, g)
        assert g["why"] and json.dumps(g)
    assert AF.gate(None, {"verdict": "axis_ok", "locks": {"steps": []}}, None, stepped_item)["passed"]
    assert AF.gate(None, {"verdict": "axis_trend", "locks": {"steps": []}}, None, stepped_item)["failed"] == ["c"]
    for v in ("locks_unpredictive", "locks_unvalidated"):               # a W model does not make them acceptable
        g = AF.gate(_spline_inner(), {"verdict": v, "locks": {"steps": []}}, None, stepped_item)
        assert g["failed"] == ["c"] and "held-out locks" in g["why"], (v, g)
    g = AF.gate(_spline_inner(), {"verdict": "axis_trend", "locks": {"steps": []}}, None, stepped_item)
    assert g["passed"] and g["b"]["edges_tested"] == 0


def test_the_gate_a_fails_on_a_step_beside_a_grid_edge_and_not_on_one_far_outside(stepped_item):
    e = stepped_item.edges[20]
    info = {"verdict": "axis_steps", "locks": {"steps": [{"between": [round(e - 0.6, 3), round(e + 0.4, 3)], "ppm": -2.4}]}}
    g = AF.gate(_spline_inner(), info, None, stepped_item)
    assert not g["passed"] and g["failed"] == ["a"] and g["a"]["near_edge"][0]["edge_mz"] == round(float(e), 4)
    far = {"verdict": "axis_steps", "steps": [{"between": [2000.0, 2010.0], "ppm": 1.5}]}
    g = AF.gate(_spline_inner(), far, None, stepped_item)
    assert g["passed"] and g["a"] == {"steps": 1, "near_edge": []}


def test_the_gate_a_counts_a_step_within_one_band_width_of_an_edge_not_only_across_it(stepped_item):
    """Every m/z inside the table lies within one band width of some edge, so a W step
    anywhere inside the table fails (a): a step the frame left or made. A bracket beside
    an edge (not across it) counts, with its distance recorded."""
    e = float(stepped_item.edges[20])
    w = float(stepped_item.band_width_at(e))
    beside = {"verdict": "axis_steps", "locks": {"steps": [{"between": [round(e + 0.1 * w, 3), round(e + 0.5 * w, 3)],
                                                           "ppm": -1.4}]}}
    g = AF.gate(_spline_inner(), beside, None, stepped_item)
    assert not g["passed"] and g["failed"] == ["a"]
    near = g["a"]["near_edge"][0]
    assert near["edge_mz"] == round(e, 4) and 0.0 < near["distance_mz"] <= w and "within one band" in g["why"]


def test_the_gate_b_leaves_out_zone_locks_on_either_side_of_the_edge(stepped_item):
    """Locks in the edge zone cannot be told a side; two LEFT-side locks in the zone (and
    one outside it) are one left lock: the edge is not tested. The same two just beyond
    the zone count, and the step is read."""
    inner = _spline_inner()
    e = float(stepped_item.edges[20])
    w = float(stepped_item.band_width_at(e))
    zone_ppm = float(stepped_item.zone_ppm_at(e))
    right = e + np.array([0.15, 0.3, 0.5, 0.7]) * w
    on_curve = lambda m, off: inner.predict(m, extrapolate=True) + off       # noqa: E731
    left_far = np.array([e - 0.7 * w])
    left_zone = e * (1 - np.array([0.5e-6, 0.9e-6]) * zone_ppm)
    locks = _locks(np.r_[left_zone, left_far, right],
                   np.r_[on_curve(left_zone, 0.9), on_curve(left_far, 0.0), on_curve(right, 0.9)])
    g = AF.gate(inner, {"verdict": "axis_trend"}, locks, stepped_item)
    assert g["passed"] and g["b"]["edges_tested"] == 0
    left_beyond = e * (1 - np.array([1.2e-6, 1.5e-6]) * zone_ppm)
    locks = _locks(np.r_[left_beyond, left_far, right],
                   np.r_[on_curve(left_beyond, 0.0), on_curve(left_far, 0.0), on_curve(right, 0.9)])
    g = AF.gate(inner, {"verdict": "axis_trend"}, locks, stepped_item)
    assert g["failed"] == ["b"] and g["b"]["edges_tested"] == 1 and g["b"]["failing"][0]["n_left"] == 3


def test_the_gate_b_reads_a_residual_step_across_an_edge_with_two_locks_each_side(stepped_item):
    inner = _spline_inner()
    e = float(stepped_item.edges[20])
    w = float(stepped_item.band_width_at(e))
    assert 80.0 < e - w and e + w < 430.0
    left = e - np.array([0.15, 0.3, 0.5, 0.7]) * w
    right = e + np.array([0.15, 0.3, 0.5, 0.7]) * w
    on_curve = lambda m, off: inner.predict(m, extrapolate=True) + off       # noqa: E731
    flat = _locks(np.r_[left, right], np.r_[on_curve(left, 0.1), on_curve(right, 0.2)])
    g = AF.gate(inner, {"verdict": "axis_trend", "locks": {"steps": []}}, flat, stepped_item)
    assert g["passed"] and g["b"]["edges_tested"] == 1 and abs(g["b"]["worst_jump_ppm"] - 0.1) < 1e-6
    stepped = _locks(np.r_[left, right], np.r_[on_curve(left, 0.0), on_curve(right, 0.8)])
    g = AF.gate(inner, {"verdict": "axis_trend", "locks": {"steps": []}}, stepped, stepped_item)
    assert not g["passed"] and g["failed"] == ["b"]
    assert g["b"]["failing"] == [{"edge_mz": round(e, 4), "n_left": 4, "n_right": 4, "jump_ppm": 0.8}]
    assert "0.80 ppm across the grid edge" in g["why"]
    # one lock on a side is not a test; bright locks and locks in the edge zone (in frequency:
    # ~440 ppm of m/z here) are left out
    one = _locks(np.r_[left, right[:1]], np.r_[on_curve(left, 0.0), on_curve(right[:1], 0.8)])
    assert AF.gate(inner, {"verdict": "axis_trend", "locks": {"steps": []}}, one, stepped_item)["b"]["edges_tested"] == 0
    bright = pd.concat([stepped.assign(bright=True), one.iloc[:4]])
    assert AF.gate(inner, {"verdict": "axis_trend"}, bright, stepped_item)["b"]["edges_tested"] == 0
    zone_ppm = float(stepped_item.zone_ppm_at(e))
    assert 100 < zone_ppm < 1000                                      # 0.1 kHz at m/z ~115 is ~260 ppm
    near = _locks(np.r_[left, e * (1 + 0.5e-6 * zone_ppm), e * (1 + 0.9e-6 * zone_ppm), right[:1]],
                  np.r_[on_curve(left, 0.0), on_curve(np.array([e, e]), 0.9), on_curve(right[:1], 0.9)])
    assert AF.gate(inner, {"verdict": "axis_trend"}, near, stepped_item)["b"]["edges_tested"] == 0
    beyond = _locks(np.r_[left, e * (1 + 1.2e-6 * zone_ppm), e * (1 + 1.5e-6 * zone_ppm), right[:1]],
                    np.r_[on_curve(left, 0.0), on_curve(np.array([e, e]), 0.9), on_curve(right[:1], 0.9)])
    assert AF.gate(inner, {"verdict": "axis_trend"}, beyond, stepped_item)["b"]["edges_tested"] == 1
    # the gate also reads axislock.fit's own info (its steps at the top level)
    g = AF.gate(inner, {"verdict": "axis_trend", "steps": []}, stepped, stepped_item)
    assert g["failed"] == ["b"]


# --- the frame in front of the lock model -----------------------------------------------------

def _two_document_batch(steps, *, phase_shift=0.1, n_items=6, **doc_kw):
    """A batch of `n_items` items from TWO documents of one calibration (the first half of
    the items from one, the second half from the other; different unmeasured low bands,
    grids 0.1 kHz apart); `doc_kw` (e.g. `noise`) goes to every document."""
    docs = {}
    for i in range(n_items):
        sid = f"s{i}"
        if i < n_items // 2:
            docs[sid] = _document(steps=steps, interp=(60, 61), item=_item(sid), n_hits=20, **doc_kw)
        else:
            docs[sid] = _document(steps=steps, interp=(58, 59), phase=PHASE + phase_shift, item=_item(sid), n_hits=36,
                                  **doc_kw)
    return docs


@pytest.fixture(scope="module")
def framed_batch():
    """A synthetic stepped batch whose steps sit at THIS grid's edges: a STEP_PPM band at
    m/z ~294-315 and SIDE_PPM on every band above it, on a smooth hump; its two documents
    (one calibration), the frame and the batch's time series."""
    steps = {K_MID: STEP_PPM, **{k: SIDE_PPM for k in range(18, K_MID)}}
    docs = _two_document_batch(steps)
    frame, rec = AF.FrameSet.from_documents(docs, polarity="-")
    assert frame is not None, rec
    it = frame.for_item("s0")

    def axis(mz):
        return _smooth(mz) + it.d_of(mz)

    ts, truth = _batch(axis, n_files=6, seed=0)
    return docs, steps, frame, axis, ts, truth


def test_the_frame_removes_the_band_steps_before_the_lock_model_and_the_gate_passes(framed_batch):
    docs, steps, frame, axis, ts, truth = framed_batch
    tsW = frame.to_frame(ts)
    assert frame.stats["frame"]["s0"]["n_moved"] > 100
    model, info = AL.fit(tsW, polarity="-", log=lambda *a, **k: None)
    assert model is not None, info.get("why")
    assert info["steps"] == [], info["steps"]
    assert [s["kind"] for s in model.segments][0] == "spline"
    locks = pd.DataFrame(info["locks"])
    assert len(locks) >= 150 and locks["ion"].isin(truth).all()
    composite = frame.with_inner(model)
    g = AF.gate(model, {"verdict": "axis_trend", "locks": info}, locks, composite)
    assert g["passed"], g
    assert g["b"]["edges_tested"] >= 3
    # true ions the model did not lock, away from the edge zones, land on the axis
    it = composite.for_item("s0")
    th = np.array([AL.exact_mz(i) for i in sorted(truth - set(locks["ion"]))])
    measured = th * (1 + axis(th) * 1e-6)
    keep = it.in_scope(measured) & ~it.edge_ambiguous(measured)
    assert keep.sum() > 20
    err = (MQ.apply_correction(it, measured[keep]) - th[keep]) / th[keep] * 1e6
    assert np.median(np.abs(err)) < 0.15 and np.mean(np.abs(err) <= 0.3) > 0.9, (np.median(np.abs(err)), np.abs(err).max())
    assert np.abs(err).max() < 0.6
    # the raw steps are gone from the W frame: the model reaches across both former edges
    lo, hi = model.mz_range
    e_lo, e_hi = AF.band_edges_mz(K_MID, B0, C0, PERIOD, PHASE)
    assert lo < e_lo - 20 and hi > e_hi + 20
    json.dumps(composite.describe())


def test_two_items_from_two_documents_of_one_calibration_get_identical_corrections(framed_batch):
    docs, steps, frame, axis, ts, truth = framed_batch
    a, b = frame.for_item("s0"), frame.for_item("s5")                 # from the two different documents
    assert docs["s0"]["modes"][0]["grid"]["phase_khz"] != docs["s5"]["modes"][0]["grid"]["phase_khz"]
    assert a.d is b.d and a.phase == b.phase and a.k0 == b.k0 and a.zone_khz == b.zone_khz
    m = ts["mz"].to_numpy()
    assert np.array_equal(a.to_W(m), b.to_W(m))
    composite = frame.with_inner(_spline_inner(80.0, 430.0))
    ca, cb = composite.for_item("s0"), composite.for_item("s5")
    assert np.array_equal(MQ.apply_correction(ca, m), MQ.apply_correction(cb, m))
    assert np.array_equal(ca.in_scope(m), cb.in_scope(m))
    # the per-item stats still count each item's own peaks
    frame.to_frame(ts)
    assert set(frame.stats["frame"]) == {f"s{i}" for i in range(6)}
    assert frame.describe()["items"]["s5"]["document_grid"]["n_hits"] == 36


def test_the_smooth_part_of_the_bands_is_left_to_the_lock_model():
    """The physical axis: the hump plus the calibration's whole band value r_k (its smooth
    trend AND the steps). The frame takes the steps out and leaves the trend, a staircase
    of < 0.15 ppm per band on the hump, which the lock model fits as one smooth curve."""
    steps = {K_MID: STEP_PPM, K_MID - 1: SIDE_PPM, _k_at(150.0): 1.3}
    frame, rec = AF.FrameSet.from_documents(_two_document_batch(steps), polarity="-")
    assert frame is not None, rec
    it = frame.for_item("s0")
    bands = rec["groups"][0]["bands"]
    r_of_k = {row["k"]: (row["r_ppm"] if row["r_ppm"] is not None else 0.0) for row in bands}
    d_of_k = {row["k"]: row["d_ppm"] for row in bands}
    assert all(abs(d_of_k[k] - s) < STEP_TOL for k, s in steps.items())

    def axis(mz):
        k = it.k_of(mz)
        return _smooth(mz) + np.array([r_of_k.get(int(kk), 0.0) if np.isfinite(kk) else 0.0 for kk in np.atleast_1d(k)])

    ts, truth = _batch(axis, n_files=6, seed=1)
    model, info = AL.fit(frame.to_frame(ts), polarity="-", log=lambda *a, **k: None)
    assert model is not None and info["steps"] == [], (info.get("why"), info.get("steps"))
    locks = pd.DataFrame(info["locks"])
    assert locks["ion"].isin(truth).all()
    composite = frame.with_inner(model)
    g = AF.gate(model, {"verdict": "axis_trend", "locks": info}, locks, composite)
    assert g["passed"], g
    cit = composite.for_item("s0")
    th = np.array([AL.exact_mz(i) for i in sorted(truth - set(locks["ion"]))])
    measured = th * (1 + axis(th) * 1e-6)
    keep = cit.in_scope(measured) & ~cit.edge_ambiguous(measured)
    err = (MQ.apply_correction(cit, measured[keep]) - th[keep]) / th[keep] * 1e6
    assert keep.sum() > 20 and np.median(np.abs(err)) < 0.15 and np.mean(np.abs(err) <= 0.3) > 0.9, \
        (np.median(np.abs(err)), np.abs(err).max())


def test_the_frame_measures_an_isolated_step_at_the_tables_real_scatter_and_the_lock_model_sees_none():
    """Two documents at a band scatter of 0.001 ppm (within what served tables show) with
    ONE 3-ppm band: the pooled table's trend is floored, the step is the band's alone, the W
    walk reports no step and the true ions land on the axis."""
    k_step = _k_at(300.0)
    frame, rec = AF.FrameSet.from_documents(_two_document_batch({k_step: 3.0}, noise=0.001), polarity="-")
    assert frame is not None, rec
    g = rec["groups"][0]
    assert g["trend_sigma_ppm"] == AF.SIGMA_FLOOR_PPM and g["trend_scatter_ppm"] < 0.01
    bands = {b["k"]: b for b in g["bands"]}
    assert abs(bands[k_step]["d_ppm"] - 3.0) < 0.05 and bands[k_step]["weight"] == 0.0
    assert max(abs(b["d_ppm"]) for kk, b in bands.items() if kk != k_step) < 0.05
    it = frame.for_item("s0")

    def axis(mz):
        return _smooth(mz) + it.d_of(mz)

    ts, truth = _batch(axis, n_files=6, seed=2)
    model, info = AL.fit(frame.to_frame(ts), polarity="-", log=lambda *a, **k: None)
    assert model is not None and info["steps"] == [], (info.get("why"), info.get("steps"))
    locks = pd.DataFrame(info["locks"])
    composite = frame.with_inner(model)
    g2 = AF.gate(model, {"verdict": "axis_trend", "locks": info}, locks, composite)
    assert g2["passed"], g2
    cit = composite.for_item("s0")
    th = np.array([AL.exact_mz(i) for i in sorted(truth - set(locks["ion"]))])
    measured = th * (1 + axis(th) * 1e-6)
    keep = cit.in_scope(measured) & ~cit.edge_ambiguous(measured)
    err = (MQ.apply_correction(cit, measured[keep]) - th[keep]) / th[keep] * 1e6
    assert keep.sum() > 20 and np.median(np.abs(err)) < 0.15 and np.abs(err).max() < 0.6, \
        (np.median(np.abs(err)), np.abs(err).max())


def test_the_gate_fails_on_a_shifted_grid_that_leaves_the_steps_at_the_edges(framed_batch):
    docs, steps, frame, axis, ts, truth = framed_batch
    shifted = {sid: _document(steps=steps, phase=PHASE + PERIOD / 2, item=_item(sid)) for sid in docs}
    wrong, rec = AF.FrameSet.from_documents(shifted, polarity="-")
    assert wrong is not None, rec
    tsW = wrong.to_frame(ts)
    model, info = AL.fit(tsW, polarity="-", log=lambda *a, **k: None)
    verdict = "axis_steps" if info.get("steps") else ("axis_trend" if model is not None else None)
    g = AF.gate(model, {"verdict": verdict, "locks": info}, pd.DataFrame(info.get("locks", [])),
                wrong.with_inner(model))
    assert not g["passed"], g
    assert set(g["failed"]) & {"a", "b", "c"}
    assert g["why"]
    json.dumps(g)
