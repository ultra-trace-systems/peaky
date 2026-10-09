"""batch/massqc.py + `peaky mass-qc` -- the external mass audit, on synthetic
batches whose axis state is known."""
import json

import numpy as np
import pandas as pd
import pytest

from peaky.batch import massqc as MQ
from peaky.chem import reference_ions as RI


def _batch(rng, refs, *, n=240, offset=lambda mz: 0.0, sigma=1.0, gamma=0.0,
           occ=1.0, noise_per_da=0.8, mz_lo=40.0, mz_hi=340.0, drop=(), height=None):
    """One row per picked peak: every reference ion in `refs` (unless in `drop`)
    placed at theory x (1 + offset(mz) ppm) with per-spectrum jitter `sigma`,
    a random walk `gamma`, present in a share `occ` of spectra; plus uniform
    noise peaks at `noise_per_da` per Da per spectrum. Hourly spectra. `height`
    (theory m/z -> height) overrides the reference ions' default 100."""
    rows = []
    t0 = pd.Timestamp("2026-01-01T00:00:00Z")
    walks = {i: np.cumsum(rng.normal(0, gamma, n)) if gamma > 0 else np.zeros(n)
             for i in range(len(refs))}
    for s in range(n):
        sid = f"s{s:04d}"
        stamp = t0 + pd.Timedelta(hours=s)
        for i, (_, r) in enumerate(refs.iterrows()):
            if r["identity"] in drop or rng.uniform() > occ:
                continue
            ppm = offset(r["mz"]) + rng.normal(0, sigma) + walks[i][s]
            h = (height(r["mz"]) if height else 100.0) * (0.97 if r["iso"] else 1.0)
            rows.append((sid, r["mz"] * (1 + ppm * 1e-6), h, stamp))
        k = rng.poisson(noise_per_da * (mz_hi - mz_lo))
        for m in rng.uniform(mz_lo, mz_hi, k):
            rows.append((sid, m, 2.0, stamp))
    return pd.DataFrame(rows, columns=["sample_item_id", "mz", "height", "datetime_utc"])


@pytest.fixture(scope="module")
def refs():
    return RI.nitrate()


def test_a_clean_batch_reads_clean_and_recovers_every_ion(refs):
    rng = np.random.default_rng(1)
    ts = _batch(rng, refs, offset=lambda mz: 0.3, sigma=1.0)
    table, v = MQ.run(ts, refs, tol_ppm=12.0)
    assert v["verdict"] == "clean", v
    assert v["n_found"] == 30 and v["calibrant_tier"] == "anchor" and v["calibrant_n"] == 11
    assert not v["calibrant_tier_fallback"]
    assert abs(v["median_offset_ppm"] - 0.3) < 0.3
    assert 0.7 < v["median_sigma_ppm"] < 1.3
    assert table["found"].all()


def test_a_flat_five_ppm_bias_is_an_axis_offset(refs):
    rng = np.random.default_rng(2)
    ts = _batch(rng, refs, offset=lambda mz: 5.0, sigma=1.0)
    _, v = MQ.run(ts, refs, tol_ppm=12.0)
    assert v["verdict"] == "axis_offset", v
    assert abs(v["median_offset_ppm"] - 5.0) < 0.4 and v["wave_span_ppm"] < 4.0
    assert "constant" in v["remedy"]


def test_a_ramp_in_sqrt_mz_is_an_axis_trend_with_a_scope(refs):
    rng = np.random.default_rng(3)
    lo, hi = np.sqrt(46.0), np.sqrt(308.0)
    ts = _batch(rng, refs, offset=lambda mz: 1.0 + 10.0 * (np.sqrt(mz) - lo) / (hi - lo), sigma=0.8)
    _, v = MQ.run(ts, refs, tol_ppm=12.0)
    assert v["verdict"] == "axis_trend", v
    assert v["wave_K"] >= 1 and v["wave_span_ppm"] > 6.0 and v["wave_share"] > 0.8
    lo, hi = v["wave_mz_range"]                 # the ANCHOR tier populates 62-220
    assert lo < 70 and hi > 200 and v["calibrant_tier"] == "anchor"
    assert "inside m/z" in v["remedy"]


def test_a_walking_axis_is_flagged_drifting(refs):
    rng = np.random.default_rng(4)
    ts = _batch(rng, refs, sigma=1.5, gamma=0.6, n=400)
    _, v = MQ.run(ts, refs, tol_ppm=12.0)
    assert "drifting" in v["verdict"], v
    assert v["rolling_frac"] >= 0.5 and v["W_star_median"] < 200
    assert "rolling centre" in v["remedy"]


def test_a_wave_that_does_not_predict_held_out_ions_is_not_a_trend(refs):
    """Ion-specific offsets with a weak smooth component: the fit explains a
    share of the calibrants it saw, but leave-one-out shows it predicts nothing
    -- that batch is BLENDED, and the remedy is not a recalibration."""
    rng = np.random.default_rng(12)
    lo, hi = np.sqrt(46.0), np.sqrt(308.0)
    # deterministic ion-to-ion jumps over the ANCHORS (the calibrant tier the
    # spread is measured on): eleven values from -10 to +10 ppm in a fixed
    # shuffled order, so the spread reads ~6 ppm (a two-valued +-8 set would
    # read 0 -- the MAD of a majority value is 0), the median 0, and no smooth
    # curve in sqrt(m/z) predicts them; the other ions sit on the ramp alone
    seq = [6.0, -8.0, 2.0, 10.0, -4.0, 0.0, -10.0, 4.0, -2.0, 8.0, -6.0]
    jumps, k = {}, 0
    for _, r in refs.sort_values("mz").iterrows():
        if r["anchor"]:
            jumps[float(r["mz"])] = seq[k % len(seq)]
            k += 1
        else:
            jumps[float(r["mz"])] = 0.0
    ts = _batch(rng, refs, offset=lambda mz: 3.0 * (np.sqrt(mz) - lo) / (hi - lo) + jumps[float(mz)], sigma=1.0)
    _, v = MQ.run(ts, refs, tol_ppm=12.0)
    assert v["verdict"].startswith("blended"), v
    assert v["ion_to_ion_spread_ppm"] >= MQ.BLEND_SPREAD_PPM
    assert not (v["wave_loo_ppm"] < MQ.TREND_LOO_GAIN * v["wave_raw_ppm"] and v["wave_span_ppm"] >= MQ.TREND_SPAN_PPM and v["wave_share"] >= MQ.TREND_SHARE)


def test_ion_to_ion_jumps_with_no_smooth_part_read_blended(refs):
    rng = np.random.default_rng(5)
    jumps = {float(m): float(j) for m, j in zip(refs["mz"], rng.choice([-9.0, 0.0, 9.0], len(refs)))}
    ts = _batch(rng, refs, offset=lambda mz: jumps[float(mz)], sigma=1.0)
    _, v = MQ.run(ts, refs, tol_ppm=12.0)
    assert v["verdict"].startswith("blended"), v
    assert v["ion_to_ion_spread_ppm"] > 5.0


def test_the_calibrant_tier_fallback_is_reported_not_silent(refs):
    rng = np.random.default_rng(6)
    anchors = set(refs.loc[refs["anchor"], "identity"])
    drop = tuple(sorted(anchors))[:8]           # identities; one names two anchor rows
    left = int((refs["anchor"] & ~refs["identity"].isin(drop)).sum())
    assert left < MQ.CAL_MIN_N
    ts = _batch(rng, refs, offset=lambda mz: 0.5, drop=drop)
    _, v = MQ.run(ts, refs, tol_ppm=12.0)
    assert v["calibrant_tier"] != "anchor" and v["calibrant_tier_fallback"]
    assert [n for n, _ in v["calibrant_tiers_tried"]][:2] == ["anchor", v["calibrant_tier"]]
    assert v["calibrant_tiers_tried"][0][1] == left


def test_no_reference_when_the_ions_are_absent(refs):
    rng = np.random.default_rng(7)
    ts = _batch(rng, refs, occ=0.01)
    _, v = MQ.run(ts, refs, tol_ppm=12.0)
    assert v["verdict"] == "no_reference"


def test_the_bromine_twin_test_withdraws_an_inconsistent_pair():
    rng = np.random.default_rng(8)
    refs = RI.bromide()
    bad = "acetic acid . 81Br-"
    ts = _batch(rng, refs, offset=lambda mz: 0.5, sigma=0.8, mz_lo=60.0, mz_hi=220.0)
    # displace the heavy acetic twin by 12 ppm: the pair no longer agrees
    m = refs.loc[refs["identity"] == bad, "mz"].iloc[0]
    sel = (ts["mz"] - m).abs() < m * 5e-6
    ts.loc[sel, "mz"] *= 1 + 12e-6
    table, v = MQ.run(ts, refs, tol_ppm=12.0)
    row = table[table["identity"].str.startswith("acetic")]
    assert (~row["twin_ok"]).all() and abs(row["twin_dppm"].iloc[0]) > 10
    assert v["n_twin_fail"] >= 2
    ok = table[table["identity"].str.startswith("formic")]
    assert ok["twin_ok"].all() and abs(ok["twin_dppm"].iloc[0]) < 2


def test_every_gate_is_absolute_a_batch_of_singletons_cannot_collapse_it(refs):
    """A1: a gate phrased 'at or below the batch median' deletes every persistent
    ion of a batch whose median jitter is 0. Here every ion is a singleton-free
    persistent trace and the noise is pure singletons: nothing is withdrawn."""
    rng = np.random.default_rng(9)
    ts = _batch(rng, refs, sigma=0.5, noise_per_da=3.0)
    _, v = MQ.run(ts, refs, tol_ppm=12.0)
    assert v["n_found"] == 30 and v["n_twin_fail"] == 0


def test_the_cli_runs_offline_on_a_parquet_and_writes_the_verdict(tmp_path, monkeypatch, refs):
    from peaky import cli
    rng = np.random.default_rng(10)
    ts = _batch(rng, refs, offset=lambda mz: 4.0, n=120)
    pq = tmp_path / "ts.parquet"
    ts.to_parquet(pq)
    monkeypatch.setattr(cli, "_require_creds", lambda *a, **k: (_ for _ in ()).throw(AssertionError("creds")))
    rc = cli.main(["mass-qc", "--ts", str(pq), "--reagent", "NO3", "--out", str(tmp_path)])
    assert rc in (0, None)
    v = json.load(open(tmp_path / "mass_qc.json"))
    assert v["verdict"] == "axis_offset" and v["reagent"] == "NO3" and v["tol_ppm"] == 12.0
    assert (tmp_path / "mass_qc.csv").exists()


# --- the instrument's own rules (per_instrument=True) and the few-spectra batch ---

REAGENT_IONS = ("nitric acid", "nitric acid . NO3-")    # NO3- and HNO3.NO3- (reagent / lock)


_HUMP_V = (330.0 ** -0.5, 46.0 ** -0.5)          # the hump's domain in v = (m/z)^-1/2
_HUMP_C = [0.825, -0.783, -0.385, 0.508]         # Chebyshev coefficients over that domain


def _hump(mz):
    """An Orbitrap axis that rises from ~0 ppm at m/z 62 to +1.9 at m/z 150-175
    and falls back to +0.9 by m/z 308 -- a rise and fall that a per-file
    a + b/(m/z) cannot follow. Smooth in (m/z)^-1/2 (a cubic there), like the
    axis it stands for, so the test is of the rules, not of the wave's reach."""
    v = np.asarray(mz, dtype=float) ** -0.5
    lo, hi = _HUMP_V
    return np.polynomial.chebyshev.chebval(2 * (v - lo) / (hi - lo) - 1, _HUMP_C)


def _orbi_batch(rng, refs, *, offset=_hump, n=9, bright_reagent=True):
    """A batch of `n` aggregated spectra (one per file) on an Orbitrap: the
    reference ions on `offset` at 0.05 ppm jitter, except the two reagent ions,
    which sit at 0 ppm (off the analyte axis) and, with `bright_reagent`, 5000x
    brighter than everything else."""
    reag = set(refs.loc[refs["identity"].isin(REAGENT_IONS), "mz"].astype(float))
    assert len(reag) == 2
    return _batch(rng, refs, n=n, sigma=0.05, noise_per_da=0.2,
                  offset=lambda mz: 0.0 if float(mz) in reag else float(offset(mz)),
                  height=(lambda mz: 5e5 if float(mz) in reag else 100.0) if bright_reagent else None)


def test_min_present_scales_with_a_small_batch_and_is_unchanged_from_twenty_spectra():
    assert [MQ.min_present_for(n) for n in (1, 2, 6, 7, 8, 9, 19, 20, 240)] == [3, 3, 3, 4, 4, 5, 10, 10, 10]


def test_a_batch_of_nine_aggregated_spectra_is_measured_not_no_reference(refs):
    """An absolute 10-spectrum presence bar read every 9-file batch as
    NO_REFERENCE although every reference ion sat in every file."""
    rng = np.random.default_rng(21)
    ts = _batch(rng, refs, n=9, offset=lambda mz: 0.3, sigma=0.5)
    table, v = MQ.run(ts, refs, tol_ppm=12.0)
    assert v["verdict"] != "no_reference", v
    assert table["found"].sum() == 30 and v["n_spectra"] == 9


def test_an_orbitrap_hump_is_an_axis_trend_by_its_own_rules_and_the_wave_follows_it(refs):
    rng = np.random.default_rng(22)
    ts = _orbi_batch(rng, refs)
    table, v = MQ.run(ts, refs, tol_ppm=6.0, tof=False, per_instrument=True)
    assert v["verdict"] == "axis_trend", v
    assert v["rules"]["name"] == "orbitrap" and v["calibrant_tier"] == "all-usable"
    assert not v["calibrant_tier_fallback"]           # the Orbitrap ladder's first tier
    # the two reagent ions are kept out of the calibrants, and nothing else is
    assert v["n_bright_excluded"] == 2
    assert all(any(i in b for i in ("HNO3 [M-H]-", "HNO3 [M+NO3]-")) for b in v["bright_excluded"])
    fit = MQ.correction(v)
    assert fit.mz_range[1] > 300          # the anchors alone stop at m/z 220
    assert v["calibrant_n"] == 28         # every reference ion but the two reagent ions
    probe_mz = np.array([100.0, 150.0, 200.0, 250.0, 300.0])
    assert np.abs(fit.predict(probe_mz) - _hump(probe_mz)).max() < 0.1
    # outside the calibrants the axis is unmeasured: left as is
    assert np.isnan(fit.predict(np.array([340.0]))).all()
    assert fit.correct(np.array([340.0]))[0] == 340.0


def test_the_legacy_rules_misread_the_same_hump(refs):
    """What the instrument's own rules fix: the TOF-sized ladder calibrates on
    the anchors (m/z 62-220, the off-axis reagent ions included) and never calls
    a 2.4 ppm Orbitrap swing a trend."""
    rng = np.random.default_rng(22)
    ts = _orbi_batch(rng, refs)
    _, v = MQ.run(ts, refs, tol_ppm=6.0, tof=False)
    assert v["rules"]["name"] == "legacy" and v["calibrant_tier"] == "anchor"
    assert v["verdict"] != "axis_trend" and v["n_bright_excluded"] == 0


def test_bright_ions_are_those_far_above_the_usable_median_height():
    t = pd.DataFrame({
        "found":      [True, True, True, True, True, True, False],
        "occurrence": [1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.0],
        "twin_ok":    [True, True, True, True, True, True, True],
        # usable median = 100 (rows 0-4): 5e4 is 500x it, 9e3 is 90x it; row 5 is
        # bright but not usable, row 6 has no height
        "height":     [80.0, 100.0, 120.0, 5e4, 9e3, 2e6, np.nan]})
    assert MQ.bright_mask(t).tolist() == [False, False, False, True, False, True, False]
    # nothing usable: the found ions' median is the reference
    t2 = t.assign(occurrence=0.1)
    assert MQ.bright_mask(t2).tolist() == [False, False, False, False, False, True, False]
    # no heights at all: nothing is bright
    assert not MQ.bright_mask(t.assign(height=np.nan)).any()


def test_a_clean_orbitrap_reads_clean_and_prescribes_nothing(refs):
    rng = np.random.default_rng(24)
    ts = _orbi_batch(rng, refs, offset=lambda mz: 0.2)
    _, v = MQ.run(ts, refs, tol_ppm=6.0, tof=False, per_instrument=True)
    assert v["verdict"] == "clean", v
    assert MQ.correction(v) is None


def test_a_flat_orbitrap_bias_is_an_axis_offset_corrected_as_a_constant(refs):
    rng = np.random.default_rng(25)
    ts = _orbi_batch(rng, refs, offset=lambda mz: 1.5)
    _, v = MQ.run(ts, refs, tol_ppm=6.0, tof=False, per_instrument=True)
    assert v["verdict"] == "axis_offset", v
    fit = MQ.correction(v)
    assert fit.K == 0
    got = fit.predict(np.array([100.0, 200.0, 300.0]))
    assert np.allclose(got, got[0]) and abs(got[0] - 1.5) < 0.1


def test_a_tof_is_judged_the_same_whichever_rules_are_asked_for(refs):
    rng = np.random.default_rng(26)
    lo, hi = np.sqrt(46.0), np.sqrt(308.0)
    ts = _batch(rng, refs, offset=lambda mz: 1.0 + 10.0 * (np.sqrt(mz) - lo) / (hi - lo), sigma=0.8)
    _, legacy = MQ.run(ts, refs, tol_ppm=12.0)
    _, own = MQ.run(ts, refs, tol_ppm=12.0, per_instrument=True)
    assert own == legacy and own["rules"]["name"] == "legacy"


def test_correction_follows_the_verdict():
    w = {"coef": [1.0, 0.5], "p": -0.5, "domain": [0.05, 0.12], "mz_range": [70.0, 300.0], "K": 1,
         "n": 20, "n_clipped": 0, "resid_ppm": 0.1, "raw_ppm": 0.6, "loo_ppm": 0.1,
         "span_ppm": 1.0, "share": 0.9}
    for v in ("clean", "drifting", "blended", "no_reference", "blended+drifting"):
        assert MQ.correction({"verdict": v, "wave": w, "median_offset_ppm": 1.0}) is None
    assert MQ.correction({"verdict": "axis_trend+drifting", "wave": w}).K == 1
    assert MQ.correction({"verdict": "axis_trend"}) is None          # no wave, nothing to apply
    off = MQ.correction({"verdict": "axis_offset", "wave": w, "median_offset_ppm": 1.7})
    assert off.K == 0 and off.coef == [1.7] and tuple(off.mz_range) == (70.0, 300.0)


def test_the_cli_judges_an_orbitrap_by_its_own_rules(tmp_path, monkeypatch, refs):
    from peaky import cli
    rng = np.random.default_rng(27)
    pq = tmp_path / "ts.parquet"
    _orbi_batch(rng, refs).to_parquet(pq)
    monkeypatch.setattr(cli, "_require_creds", lambda *a, **k: (_ for _ in ()).throw(AssertionError("creds")))
    cli.main(["mass-qc", "--ts", str(pq), "--reagent", "NO3", "--orbitrap", "--out", str(tmp_path)])
    v = json.load(open(tmp_path / "mass_qc.json"))
    assert v["verdict"] == "axis_trend" and v["rules"]["name"] == "orbitrap" and v["tol_ppm"] == 6.0


def _jittered_hump(refs, *, sd, seed):
    """The hump plus a fixed per-ion offset of sd `sd` (ion-specific scatter no
    smooth wave can predict), on a 9-spectrum Orbitrap batch."""
    rng = np.random.default_rng(seed)
    jit = {float(m): float(rng.normal(0, sd)) for m in refs["mz"]}
    return _orbi_batch(np.random.default_rng(seed + 1000), refs,
                       offset=lambda mz: float(_hump(mz)) + jit[float(mz)])


def test_an_orbitrap_wave_that_predicts_held_out_ions_poorly_is_not_a_trend(refs, monkeypatch):
    """LOO 0.61 x raw: the Orbitrap bar (half the constant's error) refuses it;
    the TOF-sized 0.8 would have called it a trend."""
    ts = _jittered_hump(refs, sd=0.3, seed=65)
    _, v = MQ.run(ts, refs, tol_ppm=6.0, tof=False, per_instrument=True)
    assert 0.5 < v["wave_loo_ppm"] / v["wave_raw_ppm"] < 0.8
    assert v["verdict"] == "axis_offset", v                  # the flat part still is one
    monkeypatch.setattr(MQ, "ORBI_LOO_GAIN", 0.8)
    _, v8 = MQ.run(ts, refs, tol_ppm=6.0, tof=False, per_instrument=True)
    assert v8["verdict"] == "axis_trend"


def test_an_orbitrap_wave_at_a_third_of_the_constants_error_is_a_trend(refs):
    ts = _jittered_hump(refs, sd=0.1, seed=73)
    _, v = MQ.run(ts, refs, tol_ppm=6.0, tof=False, per_instrument=True)
    assert 0.2 < v["wave_loo_ppm"] / v["wave_raw_ppm"] < 0.5
    assert v["verdict"] == "axis_trend", v


def test_a_flat_orbitrap_bias_under_one_ppm_reads_clean(refs):
    ts = _orbi_batch(np.random.default_rng(28), refs, offset=lambda mz: 0.7)
    _, v = MQ.run(ts, refs, tol_ppm=6.0, tof=False, per_instrument=True)
    assert v["verdict"] == "clean" and abs(v["median_offset_ppm"] - 0.7) < 0.1
