"""batch/tracefirst.py + the offline engine path + the --trace-first batch run,
on a synthetic batch whose ions are known."""
import json
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from peaky.assignment import assign as A
from peaky.batch import assign_batch as AB
from peaky.batch import tracefirst as TFT
from peaky.chem import chemistry as C
from peaky.io import io_mascope as IO

T0 = pd.Timestamp("2026-01-01T00:00:00Z")
IONS = [("C9H14O4", 40.0), ("C10H16O4", 25.0), ("C5H8O4", 12.0)]     # neutral, height


def _batch(rng, n=800, sigma=1.5, noise_per_da=0.6):
    """Hourly spectra: three [M-H]- acids at their exact masses (jitter `sigma`
    ppm, present in 95 % of spectra); the 13C satellite of the brightest at the
    natural ratio in 90 % of spectra (bright enough to seed its own trace, as a
    real one does); the 13C of the second acid picked in only 3.3 % of spectra
    (too sparse to seed: the satellite path must find it); sparse uniform noise;
    and one noise-fill position: ONE random peak inside +-12 ppm of m/z 230.1 in
    40 % of the spectra (recurrent enough to seed, never an ion: its members
    fill the window; a fill that recurs in >= 50 % would be KEPT as a blended
    peak, by design)."""
    rows = []
    for i in range(n):
        sid = f"s{i:04d}"
        stamp = T0 + pd.Timedelta(hours=i)
        for f, h in IONS:
            if rng.uniform() < 0.95:
                rows.append((sid, C.ion_mz(f, "[M-H]-") * (1 + rng.normal(0, sigma) * 1e-6), h, h * 1.2, stamp))
        m13 = C.ion_mz("C9H14O4", "[M-H]-") + 1.0033548
        if rng.uniform() < 0.9:
            rows.append((sid, m13 * (1 + rng.normal(0, sigma * 1.5) * 1e-6), 40.0 * 0.0973, 40.0 * 0.0973 * 1.2, stamp))
        # an EPISODIC bright acid in 3 % of spectra: below the seed threshold, so
        # no trace -- the residual stage's domain
        if rng.uniform() < 0.03:
            me = C.ion_mz("C7H12O5", "[M-H]-")
            rows.append((sid, me * (1 + rng.normal(0, sigma) * 1e-6), 60.0, 72.0, stamp))
            rows.append((sid, (me + 1.0033548) * (1 + rng.normal(0, sigma) * 1e-6), 60.0 * 0.076, 72.0 * 0.076, stamp))
        d13 = C.ion_mz("C10H16O4", "[M-H]-") + 1.0033548
        if rng.uniform() < 0.033:
            rows.append((sid, d13 * (1 + rng.normal(0, sigma * 2) * 1e-6), 25.0 * 0.108, 25.0 * 0.108 * 1.2, stamp))
        for m in rng.uniform(100.0, 300.0, rng.poisson(noise_per_da * 200)):
            rows.append((sid, m, 1.0, 1.2, stamp))
        if rng.uniform() < 0.4:
            rows.append((sid, float(rng.uniform(230.1 * (1 - 12e-6), 230.1 * (1 + 12e-6))), 1.0, 1.2, stamp))
    return pd.DataFrame(rows, columns=["sample_item_id", "mz", "height", "area", "datetime_utc"])


@pytest.fixture(scope="module")
def ts():
    return _batch(np.random.default_rng(11))


@pytest.fixture(scope="module")
def sample(ts):
    lines = []
    s = TFT.build_trace_sample(ts, sample_id="traces-test", reagent="NO3", resolving_power=6500.0,
                               log=lines.append)
    s.notes.append("log:" + "\n".join(lines))
    return s


def _nearest(traces, mz):
    d = (traces["mz"] - mz).abs() / mz * 1e6
    return traces.loc[d.idxmin()], float(d.min())


def test_the_ions_become_seed_traces_and_the_satellite_finds_its_parent(sample):
    t = sample.traces
    for f, _ in IONS:
        row, d = _nearest(t, C.ion_mz(f, "[M-H]-"))
        assert d < 2.0 and row["kind"] == "seed" and row["trace_occurrence"] > 0.9, (f, d)
        assert row["resid_ppm"] < 3.0 and not row["fills_window"]
    # the bright 13C recurs in 90 % of spectra: it seeds its own trace, and the
    # engine's isotope passes attach it to the parent later (as on a real batch)
    parent, _ = _nearest(t, C.ion_mz("C9H14O4", "[M-H]-"))
    sat, d = _nearest(t, C.ion_mz("C9H14O4", "[M-H]-") + 1.0033548)
    assert d < 3.0 and sat["kind"] == "seed"
    # batch-mean heights keep the isotope ratio honest
    assert abs(sat["height_avg"] / parent["height_avg"] - 0.0973 * 0.9 / 0.95) < 0.02
    # the sparse 13C (4.5 % of spectra) cannot seed: the satellite path finds it,
    # links it to its parent and measures the co-occurrence
    parent2, _ = _nearest(t, C.ion_mz("C10H16O4", "[M-H]-"))
    sat2, d2 = _nearest(t, C.ion_mz("C10H16O4", "[M-H]-") + 1.0033548)
    assert d2 < 6.0 and sat2["kind"] == "sat:13C" and sat2["parent_peak"] == parent2["peak_id"], sat2
    assert sat2["parent_cooccur"] > 0.85 and 20 <= sat2["n_members"] < 0.05 * 800


def test_the_fill_position_is_rejected_and_the_noise_is_not_a_trace(sample):
    t = sample.traces
    _, d = _nearest(t, 230.1)
    assert d > 30.0                                   # nothing within the dedup cell
    assert len(t) <= 9                                # 3 ions + 2 satellites (+ maybe a 13C2/18O)
    assert (t["mz"] - C.ion_mz("C7H12O5", "[M-H]-")).abs().min() / 190 * 1e6 > 30   # the episodic acid is no trace
    assert (t["kind"] == "seed").sum() == 4           # the bright 13C seeds too


def test_the_sample_and_summary_carry_what_the_engine_and_the_manifest_need(sample):
    assert list(sample.peaks.columns[:6]) == ["sample_item_id", "peak_id", "mz", "sparsity", "area", "height"]
    assert set(TFT.MATCH_COLS) <= set(sample.peaks.columns)
    assert (sample.peaks["sample_item_id"] == "traces-test").all()
    assert sample.occurrence.attrs["n_samples"] == 800 and sample.occurrence.attrs["tol_ppm"] == sample.tol_ppm
    s = sample.summary()
    assert s["n_seeds"] == 4 and s["n_satellites"] >= 1 and s["resolving_power"] == 6500.0
    assert abs(s["dedup_ppm"] - 30.77) < 0.1
    # the nitrate reference ions are absent from this synthetic batch, so mass-qc says so
    assert s["mass_qc"]["verdict"] == "no_reference"
    assert sample.tol_ppm == 12.0


def test_the_wave_is_applied_only_inside_the_calibrant_range():
    tr = pd.DataFrame({"mz": [50.0, 100.0, 200.0, 400.0], "height_med": 1.0, "kind": "seed"})
    from peaky.batch import wave as WV
    mz = np.array([80.0, 120.0, 160.0, 200.0, 240.0, 280.0, 320.0, 360.0])
    w = WV.fit_wave(mz, 2.0 + 0.0 * mz)                      # a flat +2 ppm bias
    qc = {"wave": w.as_dict()}
    out = TFT.apply_wave(tr, qc, log=lambda *a: None)
    assert out.loc[0, "wave_ppm"] == 0.0 and out.loc[3, "wave_ppm"] == 0.0     # outside 80-360
    assert abs(out.loc[1, "wave_ppm"] - 2.0) < 1e-6 and abs(out.loc[2, "mz"] - 200.0 * (1 - 2e-6)) < 1e-9
    assert out.loc[1, "mz_raw"] == 100.0


def test_the_engine_runs_offline_on_the_trace_sample_and_finds_the_acids(sample):
    from peaky.assignment.passes import PassConfig
    cfg = PassConfig()
    TFT.engine_settings(cfg, sample, log=lambda *a: None)
    assert cfg.search_ppm == 12.0 and cfg.ppm == 5.0
    real_connect = IO.connect
    IO.connect = lambda *a, **k: (_ for _ in ()).throw(AssertionError("offline run must not connect"))
    try:
        res = A.run("traces-test", context="ambient-air", cfg=cfg, adducts=["[M-H]-"],
                    peaks=sample.peaks, occurrence=sample.occurrence, log=lambda *a: None)
    finally:
        IO.connect = real_connect
        IO.unregister_offline_sample("traces-test")
    led = res["ledger"]
    m0 = led[led["role"] == "M0"]
    got = set(m0["neutral_formula"])
    assert {"C9H14O4", "C10H16O4"} <= got, got
    kids = led[led["role"] == "iso_child"]
    assert (kids["iso_label"].astype(str) == "13C").any()


def test_the_batch_run_takes_the_trace_first_path(ts, tmp_path):
    real_connect = IO.connect
    IO.connect = lambda *a, **k: SimpleNamespace(name="stub-client")
    try:
        res = AB.run(peaks=ts, ts_peaks=ts, reagent="NO3", batch="test batch",
                     out_dir=str(tmp_path), trace_first=True, resolving_power=6500.0,
                     residual=False, n_jobs=1, log=lambda *a: None)
    finally:
        IO.connect = real_connect
    summ = res["summary"]
    tf = summ["trace_first"]
    assert tf["n_seeds"] == 4 and summ["selection"]["method"] == "trace-first"
    merged = res["merged"]
    assert {"C9H14O4", "C10H16O4"} <= set(merged["neutral_formula"])
    assert (merged["n_files"] == 1).all()
    assert os.path.exists(tmp_path / "per_file" / "traces-test-batch_ledger.csv")
    assert os.path.exists(tmp_path / "tables" / "traces.csv")
    js = json.load(open(tmp_path / "batch_summary.json"))
    assert js["trace_first"]["n_traces"] == tf["n_traces"]


def test_the_trace_sample_stays_at_the_class_fallback_on_an_orbitrap_batch(ts, tmp_path):
    """The synthetic trace sample has no server record and no server matches: it is scored at the offline class
    fallback (a TOF's width and window, zero offset) even on an Orbitrap batch. Scoring it at the Orbitrap class
    instead lost a third of an Orbitrap batch's Assigned rows (R1 366 -> 233, R2 642 -> 433): the traces keep offsets
    the wave leaves uncorrected and carry no signal-to-noise, so a tight width with an assumed-zero offset rejects
    them (card C38). The stub serves an Orbitrap record for every real sample and none for the synthetic one, as the
    server does."""
    from mascope_tools.composition import resolve_fallback_sigma_ppm, resolve_match_tolerance_ppm, scoring_sigma_ppm
    real_connect = IO.connect

    def get(sid):
        if str(sid).startswith("traces-"):
            raise LookupError("404")
        return {"instrument_type": "orbi"}
    IO.connect = lambda *a, **k: SimpleNamespace(name="stub-client", samples=SimpleNamespace(get=get))
    try:
        AB.run(peaks=ts, ts_peaks=ts, reagent="NO3", batch="test batch", out_dir=str(tmp_path), trace_first=True,
               resolving_power=6500.0, residual=False, n_jobs=1, log=lambda *a: None)
    finally:
        IO.connect = real_connect
        IO.unregister_offline_sample("traces-test-batch")
    snap = json.load(open(tmp_path / "batch_summary.json"))["pattern_scoring"]["traces-test-batch"]
    assert snap["sigma_source"] == "instrument_class" and snap["mu_source"] == "assumed_zero"
    assert snap["instrument_type"] is None and snap["mz_tolerance_ppm"] == resolve_match_tolerance_ppm(None)
    assert snap["sigma_ppm"] == pytest.approx(scoring_sigma_ppm(None, resolve_fallback_sigma_ppm(None)), abs=1e-4)


def test_trace_first_measures_the_width_and_says_so_when_it_cannot(ts, tmp_path):
    """No --resolving-power means MEASURE it. When the profile cannot be read --
    here a client that serves none -- the run stops with an actionable message
    instead of guessing a number that would size the dedup cell wrong."""
    real_connect = IO.connect
    IO.connect = lambda *a, **k: SimpleNamespace(name="stub-client")
    try:
        with pytest.raises(ValueError, match="could not measure the peak width"):
            AB.run(peaks=ts, ts_peaks=ts, reagent="NO3", batch="b", out_dir=str(tmp_path),
                   trace_first=True, log=lambda *a: None)
        with pytest.raises(ValueError, match="ts_peaks"):
            AB.run(peaks=ts, ts_peaks=None, reagent="NO3", batch="b", out_dir=str(tmp_path),
                   trace_first=True, resolving_power=6500.0, log=lambda *a: None)
    finally:
        IO.connect = real_connect


def test_the_width_model_is_fitted_from_profile_segments_not_declared():
    """measure_resolution against a synthetic server: Gaussian segments whose
    width follows a known law. It must recover both terms -- the width AND how
    it scales -- so a TOF (constant R) and an Orbitrap (R ~ m^-1/2) are told
    apart without being declared."""
    rng = np.random.default_rng(5)
    for exponent, r_at_200 in ((1.0, 6500.0), (1.5, 150000.0)):
        coef = (200.0 ** (1 - exponent)) / r_at_200          # FWHM(200) = 200 / R
        mzs = np.geomspace(60, 600, 40)
        peaks = pd.DataFrame({"peak_id": [f"p{i}" for i in range(len(mzs))],
                              "mz": mzs, "height": 100.0})

        def server(_sid, lo, hi, coef=coef, exponent=exponent):
            m = 0.5 * (lo + hi)
            fwhm = coef * m ** exponent
            sigma = fwhm / 2.3548
            # sample the profile at a realistic density (~10 points per FWHM):
            # a fixed point count over a ppm-proportional window undersamples the
            # narrow low-mass peaks and flattens the very exponent under test
            n = int(np.clip((hi - lo) / (fwhm / 10.0), 51, 4001))
            x = np.linspace(lo, hi, n)
            return pd.DataFrame({"mz": x, "intensity": np.exp(-0.5 * ((x - m) / sigma) ** 2)})

        res = TFT.measure_resolution(None, "s", peaks=peaks, fetch_peaks=lambda *a, **k: peaks,
                                     get_spectrum=server, log=lambda *a: None)
        assert res is not None and res.source == "measured" and res.n_peaks >= 5
        assert res.exponent == pytest.approx(exponent, abs=0.08), (exponent, res.exponent)
        assert res.r_at(200.0) == pytest.approx(r_at_200, rel=0.05)
        # and the dedup cell follows the instrument instead of one constant
        wide, narrow = res.dedup_ppm(600.0), res.dedup_ppm(200.0)
        assert (wide > 1.3 * narrow) if exponent > 1.2 else (abs(wide - narrow) < 0.1 * narrow)


def test_a_measurement_that_cannot_be_made_returns_none_rather_than_raising():
    peaks = pd.DataFrame({"peak_id": ["a"], "mz": [100.0], "height": [1.0]})
    assert TFT.measure_resolution(None, "s", peaks=peaks, log=lambda *a: None) is None
    def boom(*a, **k):
        raise RuntimeError("server down")
    assert TFT.measure_resolution(None, "s", fetch_peaks=boom, log=lambda *a: None) is None
    many = pd.DataFrame({"peak_id": [f"p{i}" for i in range(50)],
                         "mz": np.geomspace(60, 600, 50), "height": 1.0})
    assert TFT.measure_resolution(None, "s", peaks=many, fetch_peaks=lambda *a, **k: many,
                                  get_spectrum=lambda *a, **k: None, log=lambda *a: None) is None


def test_offline_mechanism_lookups_resolve_only_the_declared_channels():
    IO.register_offline_sample("x", pd.DataFrame({"mz": [1.0]}), ["-H+", "+NO3-"])
    try:
        assert IO.resolve_mechanism_ids(None, ["-H+", "+NO3-", "+CO3-"]) == {"-H+": "-H+", "+NO3-": "+NO3-"}
        # a channel declared in the legacy spelling answers for the standard one
        assert IO.resolve_mechanism_ids(None, ["[M-H]-", "[M+CO3]-"]) == {"[M-H]-": "[M-H]-"}
        # handed to the scorer in the standard notation; '-H+' is a proton removed
        # (the anion), never flipped to the hydride loss '-H-'
        assert IO._mechanism_names(None, ["-H+", "+NO3-"]) == ["[M-H]-", "[M+NO3]-"]
        assert IO._mechanism_names(None, ["[M-H]-", "-H+"]) == ["[M-H]-"]
        assert IO.is_offline_sample("x") and len(IO.fetch_peaks(None, "x")) == 1
    finally:
        IO.unregister_offline_sample("x")
    assert not IO.is_offline_sample("x")


def _per_sample_tables(ts):
    """The per-file peak tables the engine would fetch, one per spectrum."""
    out = {}
    for sid, g in ts.groupby("sample_item_id"):
        t = pd.DataFrame({"sample_item_id": sid, "peak_id": [f"{sid}-{i}" for i in range(len(g))],
                          "mz": g["mz"].to_numpy(), "sparsity": 0.0,
                          "area": g["area"].to_numpy(), "height": g["height"].to_numpy()})
        for c in TFT.MATCH_COLS:
            t[c] = None
        out[sid] = t
    return out


def test_the_whole_batch_path_runs_offline_with_the_rolling_centre(ts, tmp_path):
    """The file cover, the per-file engine runs, the merge, the trace
    reconciliation WITH the rolling centre, the per-trace stamp and the summary
    -- all offline, every selected sample served from memory."""
    tabs = _per_sample_tables(ts)
    for sid, t in tabs.items():
        IO.register_offline_sample(sid, t, ["-H+"])
    real_connect, real_fetch = IO.connect, IO.fetch_peaks
    IO.connect = lambda *a, **k: None
    try:
        res = AB.run(peaks=ts, ts_peaks=ts, reagent="NO3", batch="test batch",
                     out_dir=str(tmp_path), residual=False, n_jobs=1, k_min=2, k_max=3,
                     rolling_centre=True, log=lambda *a: None)
    finally:
        IO.connect, IO.fetch_peaks = real_connect, real_fetch
        for sid in tabs:
            IO.unregister_offline_sample(sid)
    summ = res["summary"]
    tr = summ["traces"]
    assert tr["rolling"]["enabled"] and "stamp_tol_per_trace" in tr
    assert tr["rolling"]["n_rolling"] + tr["rolling"]["n_global"] >= 2
    merged = res["merged"]
    assert {"centre_scheme", "resid_ppm", "stamp_tol_ppm", "trace_key"} <= set(merged.columns)
    assert {"C9H14O4", "C10H16O4"} <= set(merged["neutral_formula"])
    ann = pd.read_parquet(tmp_path / "per_file" / "_batch_ts.parquet")   # the stamped time series
    n = ts["sample_item_id"].nunique()
    cov = ann[ann["neutral_formula"] == "C9H14O4"]["sample_item_id"].nunique() / n
    assert cov > 0.9


def test_trace_first_and_the_rolling_centre_compose(ts, tmp_path):
    real_connect = IO.connect
    IO.connect = lambda *a, **k: SimpleNamespace(name="stub-client")
    try:
        res = AB.run(peaks=ts, ts_peaks=ts, reagent="NO3", batch="test batch",
                     out_dir=str(tmp_path), trace_first=True, resolving_power=6500.0,
                     rolling_centre=True, residual=False, n_jobs=1, log=lambda *a: None)
    finally:
        IO.connect = real_connect
    summ = res["summary"]
    assert summ["trace_first"]["n_seeds"] == 4 and summ["traces"]["rolling"]["enabled"]
    # trace-first applies its own wave: the batch's axis step stands aside
    assert "trace-first" in summ["mass_axis"]["skipped"] and not summ["mass_axis"]["applied"]
    assert "stamp_tol_ppm" in res["merged"].columns


def test_under_trace_first_a_residual_file_only_adds_what_the_stamp_left_unexplained(ts, tmp_path):
    """The episodic acid recurs in 3 % of spectra -- no trace -- so its bins are
    residual; the residual files that cover them also carry the three persistent
    acids, which the trace ledger already explains: those rows must NOT merge
    back in (they would out-vote the trace reading), the episodic one must."""
    tabs = _per_sample_tables(ts)
    for sid, t in tabs.items():
        IO.register_offline_sample(sid, t, ["-H+"])
    real_connect = IO.connect
    IO.connect = lambda *a, **k: None
    try:
        res = AB.run(peaks=ts, ts_peaks=ts, reagent="NO3", batch="test batch",
                     out_dir=str(tmp_path), trace_first=True, resolving_power=6500.0,
                     residual=True, residual_k_max=2, n_jobs=1, log=lambda *a: None)
    finally:
        IO.connect = real_connect
        for sid in tabs:
            IO.unregister_offline_sample(sid)
    summ = res["summary"]
    rmeta = summ["selection"]["residual"]
    assert rmeta["k"] >= 1, rmeta
    sc = rmeta["trace_first_scope"]
    assert 0 < sc["rows_kept"] < sc["rows_total"]
    merged = res["merged"]
    persistent = merged[merged["neutral_formula"].isin(["C9H14O4", "C10H16O4", "C5H8O4"])]
    assert (persistent["n_files"] == 1).all()                 # the trace reading, unchanged
    assert "C7H12O5" in set(merged["neutral_formula"])       # the episodic acid, added


def test_the_membership_window_never_exceeds_half_the_instruments_own_cell(ts):
    """At high resolving power the 0.4-HWHM cell is ~1 ppm wide, so the TOF's
    validated 12 ppm membership floor must not apply: a trace's members may
    never span more than one observable."""
    lines = []
    s = TFT.build_trace_sample(ts, sample_id="hi-res", reagent="NO3", resolving_power=155000.0,
                               log=lines.append)
    cell = TFT.dedup_ppm(200.0, 155000.0)
    assert cell < 2.0
    assert s.tol_ppm <= cell / 2 + 1e-9, (s.tol_ppm, cell)
    assert any("high-resolution instrument" in n for n in s.notes)
    assert any("NOTE:" in ln for ln in lines)
    # the TOF case is unchanged: the floor applies because the cell is wide
    t = TFT.build_trace_sample(ts, sample_id="tof", reagent="NO3", resolving_power=6500.0,
                               log=lambda *a: None)
    assert t.tol_ppm == TFT.MEMBER_MIN_PPM
    assert not any("high-resolution" in n for n in t.notes)


def test_trace_first_on_a_reagent_with_no_reference_table_says_so_and_runs(ts):
    """A positive-mode reagent has no reference-ion list yet: mass-qc is skipped,
    no wave is applied, and the run says both in its notes rather than failing."""
    s = TFT.build_trace_sample(ts, sample_id="no-refs", reagent="Ur", resolving_power=6500.0,
                               log=lambda *a: None)
    assert s.qc is None and s.summary()["mass_qc"] is None
    assert any("no reference-ion table" in n for n in s.notes)
    assert len(s.traces) and (s.traces["wave_ppm"] == 0).all()
    assert s.tol_ppm == TFT.MEMBER_MIN_PPM


def test_the_spectra_bound_the_resolving_power_and_catch_a_value_from_another_instrument(ts):
    """The closest two maxima a picker reports bound the peak width. It is a
    bound, not a measurement -- how far into a flank a picker calls a maximum is
    the picker's property (a TOF picker reports down to 1.0 HWHM, an Orbitrap
    picker to 2.9) -- so the band is wide, and still twenty times tighter than
    the error it exists to catch."""
    b = TFT.spacing_bound(ts, 6500.0)
    assert b["d_min_ppm"] > 0 and b["hwhm_ppm"] == pytest.approx(76.92, abs=0.1)
    # the same spectra judged against an Orbitrap's resolving power: refused
    hi = TFT.spacing_bound(ts, 155000.0)
    assert hi["ratio"] > TFT.SPACING_MAX_RATIO and not hi["ok"]
    # ... and against an absurdly low one: also refused, from the other side
    lo = TFT.spacing_bound(ts, 1000.0)
    assert lo["ratio"] < TFT.SPACING_MIN_RATIO and not lo["ok"]
    # a table with no spectra to measure returns the empty bound, never raises
    empty = TFT.spacing_bound(ts.head(3), 6500.0)
    assert empty["ratio"] is None and empty["ok"]
    assert TFT.spacing_bound(None, 6500.0)["ok"]


def test_a_mismatched_resolving_power_is_noted_on_the_sample_not_swallowed(ts):
    lines = []
    s = TFT.build_trace_sample(ts, sample_id="wrong-R", reagent="NO3", resolving_power=155000.0,
                               log=lines.append)
    assert not s.spacing["ok"] and s.summary()["spacing_bound"]["ratio"] > 10
    assert any("different instrument" in n for n in s.notes)
    assert any("WARNING" in ln for ln in lines)


def test_the_model_holds_mascopes_own_resolution_functions():
    """Mascope fits a resolution function per instrument -- a rational polynomial
    for a TOF, an inverse square root for an Orbitrap -- and stores it against the
    instrument config. It is not reachable with a service token today, so the
    model just has to be able to HOLD it for when it is."""
    a, b = 1.0e-4, 2.0e-3
    tof = TFT.Resolution.from_mascope([a, b], "tofwerk")
    for m in (100.0, 400.0):
        assert tof.fwhm(m) == pytest.approx(a * m + b)          # R = m / (a m + b)
        assert tof.r_at(m) == pytest.approx(m / (a * m + b))
    orbi = TFT.Resolution.from_mascope([2.4e6], "orbitrap")
    for m in (200.0, 800.0):
        assert orbi.r_at(m) == pytest.approx(2.4e6 / np.sqrt(m), rel=1e-6)
    assert orbi.r_at(800.0) == pytest.approx(orbi.r_at(200.0) / 2, rel=1e-6)
    assert tof.source == "mascope" and orbi.source == "mascope"
    with pytest.raises(ValueError):
        TFT.Resolution.from_mascope([], "tofwerk")


def test_the_width_model_says_which_dispersion_the_analyser_works_in():
    # a scalar R, and Mascope's own TOF form, are both constant-R: flight time
    assert TFT.Resolution.from_r(6500).is_tof
    assert TFT.Resolution.from_mascope([1.2e-4, 2.0e-3], "tofwerk").is_tof
    # an Orbitrap's width grows as m^1.5 -- frequency
    assert not TFT.Resolution.from_mascope([1.1e4], "orbitrap").is_tof
    assert not TFT.Resolution(coef=1e-6, exponent=1.5).is_tof
    # the exponents measured off real batches land either side of the split
    assert TFT.Resolution(coef=1e-4, exponent=1.08).is_tof
    assert not TFT.Resolution(coef=1e-6, exponent=1.57).is_tof


def test_an_orbitrap_batch_fits_its_wave_in_frequency_not_flight_time(ts, monkeypatch):
    """The basis used to be hard-coded tof=True, so a trace-first Orbitrap run
    fitted its wave against (m/z)^+1/2 -- the wrong variable for the analyser."""
    seen = []
    real = TFT.MQ.verdict
    monkeypatch.setattr(TFT.MQ, "verdict",
                        lambda t, n, *, tof=True: (seen.append(tof), real(t, n, tof=tof))[1])
    for res, expect in ((TFT.Resolution.from_r(6500), True),
                        (TFT.Resolution(coef=1.0 / 155000.0, exponent=1.5), False)):
        seen.clear()
        TFT.build_trace_sample(ts, sample_id="s", reagent="NO3", resolving_power=res,
                               log=lambda *a: None)
        assert seen == [expect]


def test_the_fill_test_separates_an_ion_from_a_fill_at_every_member_count():
    """The property the old statistic did not have.

    It normalised by the members' OWN range, which divides out the width -- so a
    real ion and a true fill both read KS ~ 0.21-0.25 and `KS * sqrt(n)` was a
    disguised member count. Measured on the shipped code, a real ion was called a
    fill 100% of the time at n=10 and 94% at n=40.
    """
    rng = np.random.default_rng(0)
    W = 10.36
    for n in (10, 20, 40, 80, 200):
        ion = TFT.fill_ratio(rng.normal(0.0, 1.35, n), W)
        fill = TFT.fill_ratio(rng.uniform(-W, W, n), W)
        assert ion < TFT.FILL_SIGMA_FRAC <= fill, f"n={n}: ion {ion:.2f}, fill {fill:.2f}"
    # and the statistic means what it says: a fill reads ~1, an ion reads its
    # own sigma as a fraction of what a fill would give
    assert 0.85 < TFT.fill_ratio(rng.uniform(-W, W, 400), W) < 1.15
    assert abs(TFT.fill_ratio(rng.normal(0.0, 1.35, 400), W) - 1.35 / (TFT.FILL_FRAC * W)) < 0.05
    assert TFT.fill_ratio([0.1, -0.1], W) == 0.0          # too few points to judge


def test_a_real_ion_with_few_members_is_not_thrown_away_as_a_fill():
    """A tight ion in 12 of 230 spectra is exactly what the old gate destroyed:
    it had no power at that n, and read 'cannot reject uniform' as 'is a fill'."""
    rng = np.random.default_rng(4)
    W = 10.36
    for n in (12, 15, 25):
        offsets = rng.normal(0.0, 1.35, n)
        assert TFT.fill_ratio(offsets, W) < TFT.FILL_SIGMA_FRAC
    # an ion as wide as half the window is genuinely ambiguous and may go either
    # way -- that is a physical limit, not a test failure; it must not crash
    assert TFT.fill_ratio(rng.normal(0.0, 5.0, 40), W) > 0.0


def _episode_batch(rng, n_spectra=200, mz_e=301.1234, mz_s=411.2345, scattered=True):
    """A batch with a dim picked-noise floor (so the noise edge is realistic), a
    persistent background, one contiguous episode and optionally the same number
    of detections scattered across the campaign."""
    rows = []
    for s in range(n_spectra):
        for mz in (150.0 + 0.001 * s, 250.5, 350.75):
            rows.append((f"s{s:03d}", mz * (1 + rng.normal(0, 2e-6)), 40.0))
        for _ in range(12):                        # the picker's own floor
            rows.append((f"s{s:03d}", float(rng.uniform(120.0, 500.0)), float(rng.uniform(0.5, 2.0))))
        if 90 <= s < 97:
            rows.append((f"s{s:03d}", mz_e * (1 + rng.normal(0, 2e-6)), 60.0))
        if scattered and s in (5, 33, 61, 98, 140, 171, 195):
            rows.append((f"s{s:03d}", mz_s * (1 + rng.normal(0, 2e-6)), 60.0))
    ts = pd.DataFrame(rows, columns=["sample_item_id", "mz", "height"])
    ts["datetime_utc"] = pd.to_datetime(
        [int(x[1:]) for x in ts["sample_item_id"]], unit="h", utc=True)
    ts["area"] = ts["height"]
    return ts


def test_a_short_bright_episode_seeds_a_trace_but_the_same_count_scattered_does_not():
    """The seed floor asks an ion to recur across the whole batch, which a plume
    never does. An ion in 7 CONSECUTIVE spectra of 200 is 3.5% occurrence -- under
    the 5% floor -- and used never to become a trace at all."""
    ts = _episode_batch(np.random.default_rng(7))
    s = TFT.build_trace_sample(ts, sample_id="epi", reagent="Ur", episodes=True,
                               resolving_power=60000.0, log=lambda *a: None)
    t = s.traces
    near = lambda m: t[(t["mz"] - m).abs() / m * 1e6 < 15]
    epi, scat = near(301.1234), near(411.2345)
    assert len(epi) == 1 and epi.iloc[0]["kind"] == "episode"
    assert epi.iloc[0]["trace_occurrence"] < TFT.SEED_OCC       # under the seed floor
    assert epi.iloc[0]["episode_span"] <= TFT.EPISODE_IQR
    assert len(scat) == 0, "detections scattered across the campaign are not an episode"


def test_the_episode_pass_can_be_turned_off_and_never_outranks_a_persistent_ion():
    ts = _episode_batch(np.random.default_rng(8), scattered=False)
    idx = TFT.TR.PeakIndex(ts, tol_ppm=12.0, sample_col="sample_item_id")
    hours = TFT.sample_hours(idx, ts, sample_col="sample_item_id", time_col="datetime_utc")
    kw = dict(resolving_power=60000.0, tol_ppm=12.0, log=lambda *a: None)
    off = TFT.build_traces(idx, hours, episodes=False, **kw)
    on = TFT.build_traces(idx, hours, episodes=True, **kw)
    assert (off["kind"] == "episode").sum() == 0
    assert (on["kind"] == "episode").sum() >= 1
    assert any(abs(m - 301.1234) / 301.1234 * 1e6 < 15
               for m in on[on["kind"] == "episode"]["mz"])
    # the persistent ions are seeded either way, and identically: the episode pass
    # runs second and can never take a cell a persistent ion would have had
    assert off[off["kind"] == "seed"]["mz"].tolist() == on[on["kind"] == "seed"]["mz"].tolist()


def test_the_episode_pass_is_off_unless_asked_for():
    """It offers 322 more positions on a real TOF batch and only 20 of them are
    confirmed by a file cover; until an end-to-end run says what the rest are, the
    default must not change what a run produces."""
    ts = _episode_batch(np.random.default_rng(9), scattered=False)
    off = TFT.build_trace_sample(ts, sample_id="d", reagent="Ur",
                                 resolving_power=60000.0, log=lambda *a: None)
    on = TFT.build_trace_sample(ts, sample_id="d", reagent="Ur", episodes=True,
                                resolving_power=60000.0, log=lambda *a: None)
    assert (off.traces["kind"] == "episode").sum() == 0
    assert (on.traces["kind"] == "episode").sum() >= 1


def test_under_trace_first_the_vote_class_is_read_off_the_ledger_the_engine_returned(ts, tmp_path, monkeypatch):
    """The trace table carries its own `resolvability`, so the trace sample's
    ledger, once the trace columns are merged on, holds resolvability_x/_y. The
    vote class must be the one of the ledger the engine RETURNED (as the per-file
    stage before 0.10.0 read it): a blended exact-mass reading with one plausible
    ion in the window is unconfirmed (class 0) -- read off the merged ledger it
    sees no resolvability and counts as formula confirmed (class 1)."""
    from peaky.assignment import evidence as EV
    seen = {}
    real_run = A.run

    def run_blended(*a, **k):
        res = real_run(*a, **k)
        led = res["ledger"].copy()
        m0 = led["role"].astype(str) == "M0"
        led.loc[m0, "resolvability"] = "blended"
        led.loc[m0, "degeneracy_density"] = 1.0
        res["ledger"] = led
        seen["ledger"], seen["stats"] = led.copy(), dict(res.get("stats") or {})
        return res
    monkeypatch.setattr(A, "run", run_blended)
    monkeypatch.setattr(IO, "connect", lambda *a, **k: SimpleNamespace(name="stub-client"))
    AB.run(peaks=ts, ts_peaks=ts, reagent="NO3", batch="test batch", out_dir=str(tmp_path), trace_first=True,
           resolving_power=6500.0, residual=False, n_jobs=1, log=lambda *a: None)
    assert "resolvability" in pd.read_csv(tmp_path / "tables" / "traces.csv").columns
    led = seen["ledger"]
    hal = seen["stats"].get("reagent_halogen", EV.DETECT_HALOGEN)
    own = EV.vote_classes(EV.trim(led), resolution=6500.0, halogen=hal)
    # the same ledger without the measurement: what a read of the merged ledger sees
    unmeasured = EV.vote_classes(EV.trim(led.drop(columns=["resolvability"])), resolution=6500.0, halogen=hal)
    m0 = led[led["role"].astype(str) == "M0"]
    want = {(round(float(z), 6), n, a): int(c) for z, n, a, c in
            zip(m0["mz"], m0["neutral_formula"], m0["adduct"], own.reindex(m0.index))}
    jit = pd.read_csv(tmp_path / "tables" / "jitter.csv")
    got = {(round(float(z), 6), n, a): int(c) for z, n, a, c in
           zip(jit["mz"], jit["neutral_formula"], jit["adduct"], jit["vote_class"])}
    assert got and set(got) <= set(want)
    assert got == {k: want[k] for k in got}
    # the case is live: some reading is unconfirmed only because its peak is blended
    flipped = (own != unmeasured).reindex(m0.index)
    assert flipped.any() and (own[flipped] == 0).all()
    assert any(want[k] == 0 for k in got if (k in want))
    assert "__own_vote_class" not in pd.read_csv(tmp_path / "per_file" / "traces-test-batch_ledger.csv").columns
