"""batch/massqc.py + `peaky mass-qc` -- the external mass audit, on synthetic
batches whose axis state is known."""
import json

import numpy as np
import pandas as pd
import pytest

from peaky.batch import massqc as MQ
from peaky.chem import reference_ions as RI


def _batch(rng, refs, *, n=240, offset=lambda mz: 0.0, sigma=1.0, gamma=0.0,
           occ=1.0, noise_per_da=0.8, mz_lo=40.0, mz_hi=340.0, drop=()):
    """One row per picked peak: every reference ion in `refs` (unless in `drop`)
    placed at theory x (1 + offset(mz) ppm) with per-spectrum jitter `sigma`,
    a random walk `gamma`, present in a share `occ` of spectra; plus uniform
    noise peaks at `noise_per_da` per Da per spectrum. Hourly spectra."""
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
            h = 100.0 * (0.97 if r["iso"] else 1.0)
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
