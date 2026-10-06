"""C15: the reagent-ion water ladder of a batch (peaky/batch/reagent_water.py), the isotopologue tag
on the reagent library's cluster labels, and the batch wiring (strip + stamp + table + summary)."""
import json
import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from peaky.batch import assign_batch as AB
from peaky.batch import reagent_water as RW
from peaky.batch import timeseries as TS
from peaky.chem import chemistry as C
from peaky.chem import profiles as P
from peaky.chem import reagents as R

T0 = pd.Timestamp("2026-01-01 00:00", tz="UTC")
BR79 = 78.9183371 + C.M_E
BR81 = 80.9162906 + C.M_E
W = C.neutral_mass("H2O")


def _ts(spectra):
    """spectra: list of (minutes, [m/z ...]) -> a batch TS frame."""
    rows = []
    for i, (minute, mzs) in enumerate(spectra):
        for m in mzs:
            rows.append(dict(sample_item_id=f"s{i:03d}", datetime_utc=T0 + pd.Timedelta(minutes=minute),
                             mz=float(m), height=1000.0))
    return pd.DataFrame(rows)


# --- the library label carries the isotopologue (the stamp-collapse fix) ------------------------------
def test_water_cluster_labels_carry_the_isotopologue_tag():
    lib = {label: (mz, f) for label, mz, f in R.build_library("Br")}
    assert "[Br1+1xH2O]- (79Br)" in lib and "[Br1+1xH2O]- (81Br)" in lib
    assert "[Br2+1xH2O]- (79Br+81Br)" in lib
    assert "[Br1+1xHF]- (81Br)" in lib
    assert lib["[Br1+1xH2O]- (79Br)"][1] == lib["[Br1+1xH2O]- (81Br)"][1] == "H2BrO-"


def test_the_two_isotopologues_of_one_water_cluster_stamp_as_two_tracks():
    """Before the tag both lines shared (H2BrO-, no tag) and stamped once at their median (97.93)."""
    led = pd.DataFrame({"peak_id": ["a", "b"], "mz": [96.9295, 98.9274], "role": ["reagent", "reagent"],
                        "neutral_formula": [None, None], "adduct": [None, None],
                        "ion_formula": ["H2BrO-", "H2BrO-"], "iso_label": [None, None],
                        "parent_peak_id": [None, None], "commentary": [None, None]})
    for i, (label, mz, f) in enumerate(l for l in R.build_library("Br") if l[0].startswith("[Br1+1xH2O]-")):
        led.loc[i, "commentary"] = f"reagent ion: {label} (+0.2 ppm)"
    idr = TS.identified_rows(led)
    assert sorted(idr["iso_label"]) == ["79Br", "81Br"]
    merged = pd.DataFrame({"mz": [200.0], "neutral_formula": ["C5H6O5"], "adduct": ["[M-H]-"], "tier": ["Assigned"]})
    sf = TS.stamping_frame(merged, idr, predict_satellites=False)
    reag = sf[sf["role"] == "reagent"].sort_values("mz")
    assert len(reag) == 2
    assert np.allclose(reag["mz"], [96.9295, 98.9274])


# --- profiles -------------------------------------------------------------------------------------------
def test_profiles_declare_water_cores_and_compose_unions_them():
    assert P.PROFILES["Br"].water_cores == ("Br", "Br2", "Br3", "HBrNO3")
    assert P.PROFILES["NO3"].water_cores == ("NO3", "HN2O6", "H2N3O9", "NO2")
    assert P.PROFILES["NO3_15N"].water_cores == ("^NO3", "H^N2O6", "H2^N3O9")
    assert P.PROFILES["Ur"].water_cores == ()
    mixed = P.compose([P.PROFILES["NO3"], P.PROFILES["Br"]])
    assert mixed.water_cores == ("NO3", "HN2O6", "H2N3O9", "NO2", "Br", "Br2", "Br3", "HBrNO3")


def test_a_config_profile_can_declare_water_cores():
    p = P.from_dict(dict(name="X", label="X", polarity="-", adducts=["[M+X]-"], normaliser="tic",
                         reagent_ion_re=None, ranges="C0-10", detect_adduct=None, water_cores=["NO3"]))
    assert p.water_cores == ("NO3",)


# --- cores and formulas ----------------------------------------------------------------------------------
def test_cores_enumerate_the_halogen_isotopologues():
    cs = RW.cores(("Br", "Br2", "HBrNO3", "NO3"))
    assert [c.label for c in cs] == ["Br- (79Br)", "Br- (81Br)", "Br2- (79Br+79Br)", "Br2- (79Br+81Br)",
                                     "Br2- (81Br+81Br)", "HBrNO3- (79Br)", "HBrNO3- (81Br)", "NO3-"]
    assert cs[0].mz == pytest.approx(78.9189, abs=1e-4)
    assert cs[5].mz == pytest.approx(C.neutral_mass("HNO3") + BR79, abs=1e-6)
    assert cs[-1].mz == pytest.approx(61.9884, abs=1e-4)


def test_rung_formula_matches_the_reagent_library():
    assert RW.rung_formula(RW.cores(("Br",))[0], 1) == "H2BrO-"
    assert RW.rung_formula(RW.cores(("NO3",))[0], 2) == "H4NO5-"


# --- segments --------------------------------------------------------------------------------------------
def test_segments_cut_at_acquisition_gaps_and_merge_short_ones():
    spec = [(i * 20, [100.0]) for i in range(20)]                       # 20 spectra every 20 min
    spec += [(400 + 180 + i * 20, [100.0]) for i in range(12)]          # a 3 h gap, 12 more
    spec += [(400 + 180 + 240 + 1000 + i * 20, [100.0]) for i in range(4)]   # another gap, only 4
    seg = RW.segments(_ts(spec))
    assert seg.value_counts().sort_index().tolist() == [20, 16]          # the 4 join the 12


def test_segments_without_time_are_one():
    ts = _ts([(i, [100.0]) for i in range(5)]).drop(columns=["datetime_utc"])
    assert set(RW.segments(ts)) == {0}



def _two_blocks(spacing, gap, n=15):
    spec = [(i * spacing, [100.0]) for i in range(n)]
    spec += [((n - 1) * spacing + gap + i * spacing, [100.0]) for i in range(n)]
    return RW.segments(_ts(spec)).nunique()


def test_a_segment_boundary_is_a_gap_over_five_median_spacings_with_a_one_hour_floor():
    """gap > max(60 min, 5 x the median spacing): at 20-min spacing the bar is 100 min
    (the TOF batch's restart gap is 130 min at 20-min spacing), at 5-min spacing 60 min."""
    assert _two_blocks(20, 110) == 2 and _two_blocks(20, 90) == 1 and _two_blocks(20, 100) == 1   # strictly over
    assert _two_blocks(5, 70) == 2 and _two_blocks(5, 50) == 1

# --- detection ---------------------------------------------------------------------------------------
def _ladder(n_top, *, core=BR79, missing=(), extra=()):
    return [core] + [core + n * W for n in range(1, n_top + 1) if n not in missing] + list(extra)


def test_a_ladder_present_in_every_spectrum_passes_up_to_its_end():
    ts = _ts([(i * 10, _ladder(6)) for i in range(12)])
    r = RW.detect(ts, ("Br",), tol_ppm=5.0)
    assert r["n"].tolist() == [1, 2, 3, 4, 5, 6]
    assert set(r["iso_tag"]) == {"79Br"}
    assert r["ion_formula"].tolist()[0] == "H2BrO-"


def test_a_missing_rung_ends_the_ladder():
    ts = _ts([(i * 10, _ladder(6, missing=(3,))) for i in range(12)])
    assert RW.detect(ts, ("Br",), tol_ppm=5.0)["n"].tolist() == [1, 2]


def test_no_core_no_rungs():
    ts = _ts([(i * 10, [BR79 + n * W for n in range(1, 5)]) for i in range(12)])
    assert RW.detect(ts, ("Br",), tol_ppm=5.0).empty


def test_a_rung_whose_decoy_offset_is_as_present_fails_the_decoy_gate():
    decoy = BR79 + 4 * W + 0.035
    ts = _ts([(i * 10, _ladder(6, extra=(decoy,))) for i in range(12)])
    assert RW.detect(ts, ("Br",), tol_ppm=5.0)["n"].tolist() == [1, 2, 3, 5, 6]


def test_presence_below_half_of_the_segment_fails():
    spec = [(i * 10, _ladder(3) + ([BR79 + 4 * W] if i < 5 else [])) for i in range(12)]   # rung 4 in 5/12
    assert RW.detect(_ts(spec), ("Br",), tol_ppm=5.0)["n"].tolist() == [1, 2, 3]
    spec = [(i * 10, _ladder(3) + ([BR79 + 4 * W] if i < 6 else [])) for i in range(12)]   # 6/12 = 0.5
    assert RW.detect(_ts(spec), ("Br",), tol_ppm=5.0)["n"].tolist() == [1, 2, 3, 4]


def test_a_rung_counts_where_its_segment_shows_it():
    before = [(i * 10, _ladder(2)) for i in range(12)]
    after = [(300 + 200 + i * 10, _ladder(9)) for i in range(12)]          # a 200-min gap, then a longer ladder
    r = RW.detect(_ts(before + after), ("Br",), tol_ppm=5.0)
    assert r["n"].tolist() == list(range(1, 10))
    assert r.set_index("n").loc[1, "segments"] == "0|1" and r.set_index("n").loc[9, "segments"] == "1"


def test_outside_the_window_is_absent():
    ts = _ts([(i * 10, [BR79, BR79 + W + BR79 * 12e-6]) for i in range(12)])       # rung 1 at +12 ppm
    assert RW.detect(ts, ("Br",), tol_ppm=5.0).empty
    assert RW.detect(ts, ("Br",), tol_ppm=15.0)["n"].tolist() == [1]


# --- strip / stamp / table -------------------------------------------------------------------------------
def _rungs():
    ts = _ts([(i * 10, _ladder(5) + _ladder(2, core=BR81)) for i in range(12)])
    return RW.detect(ts, ("Br",), tol_ppm=5.0)


def test_a_merged_row_on_a_passing_rung_leaves_the_merged_ledger():
    r = _rungs()
    merged = pd.DataFrame({"mz": [BR79 + 4 * W + 1e-4, 250.0], "neutral_formula": ["C7H6O6", "C9H14O5"],
                           "adduct": ["[M-H]-", "[M-H]-"], "tier": ["Assigned", "Assigned"]})
    kept, stripped = RW.strip_rung_rows(merged, r, tol_ppm=5.0, log=lambda *a: None)
    assert kept["neutral_formula"].tolist() == ["C9H14O5"]
    assert stripped["neutral_formula"].tolist() == ["C7H6O6"]
    assert stripped["rung"].tolist() == ["Br(79Br).(H2O)4"]
    t = RW.table(r, stripped)
    assert t.loc[(t["n"] == 4) & (t["iso_tag"] == "79Br"), "displaced"].iloc[0] == "C7H6O6 [M-H]-"
    assert list(t.columns) == list(RW.TABLE_COLUMNS)


def test_nothing_is_stripped_without_rungs():
    merged = pd.DataFrame({"mz": [150.0], "neutral_formula": ["C5H6O5"], "adduct": ["[M-H]-"]})
    kept, stripped = RW.strip_rung_rows(merged, RW.detect(None, (), tol_ppm=5.0), tol_ppm=5.0)
    assert kept.equals(merged) and stripped.empty


def test_stamp_rows_keep_the_isotopologue_apart():
    s = RW.stamp_rows(_rungs())
    assert set(s["role"]) == {"reagent"}
    one = s[s["ion_formula"] == "H2BrO-"].sort_values("mz")
    assert one["iso_label"].tolist() == ["79Br", "81Br"]


def test_measure_is_a_noop_without_water_cores_or_time_series():
    out = RW.measure(None, P.PROFILES["Br"], tol_ppm=6.0, log=lambda *a: None)
    assert out["rungs"].empty and out["n_cores"] == 11
    ts = _ts([(i * 10, _ladder(4)) for i in range(12)])
    out = RW.measure(ts, P.PROFILES["Ur"], tol_ppm=6.0, log=lambda *a: None)
    assert out["rungs"].empty and out["n_cores"] == 0


# --- the batch wiring -----------------------------------------------------------------------------------
def test_a_batch_strips_the_rung_row_stamps_the_ladder_and_writes_the_table(monkeypatch):
    from peaky.assignment import assign as A
    from peaky.assignment import ledger as L
    from peaky.assignment import tiers as T
    from peaky.io import io_mascope as IO

    rung5 = BR79 + 5 * W
    spectra = [(i * 10, _ladder(6) + [250.1]) for i in range(12)]
    ts = _ts(spectra)

    def fake_assign(sid, context="ambient-air", **kw):
        led = L.new_ledger(pd.DataFrame([("p1", rung5, 900.0), ("p2", 250.1, 5e3), ("p3", rung5 + 1.00335, 40.0)],
                                        columns=["peak_id", "mz", "height"]))
        L.commit_assignment(led, "p1", neutral_formula="C4H4O8", adduct="[M-H]-", ion_formula="C4H3O8-",
                            ion_score=0.9, compound_score=0.9, ppm_error=0.1, pass_no=1,
                            method="cheminfo+grid", confidence="High", commentary="stub",
                            isotopologues=[{"label": "13C", "score": 0.9, "peak_id": "p3"}])
        # its own 13C line does not save it: on a TOF the M+1 line of a water rung
        # is its 15N-labelled partner as often as a 13C (no 13C exemption)
        L.attach_isotopologue(led, "p3", "p1", iso_label="13C", iso_match_score=0.9)
        L.commit_assignment(led, "p2", neutral_formula="C9H16O6", adduct="[M-H]-", ion_formula="C9H15O6-",
                            ion_score=0.9, compound_score=0.9, ppm_error=0.1, pass_no=1,
                            method="cheminfo+grid", confidence="High", commentary="stub")
        T.apply_tiers(led)
        return {"ledger": led, "stats": {"noise_edge_cps": 4.0, "height_gate_cps": 10.0,
                                         "degeneracy_cal": {"mu": 0.0, "sigma": 0.3}},   # as a real run persists
                "plausibility_audit": [], "summaries": {}, "problems": []}

    monkeypatch.setattr(IO, "connect", lambda *a, **k: "CLIENT")
    monkeypatch.setattr(IO, "fetch_peaks", lambda client, sid, use_cache=True: pd.DataFrame(
        {"peak_id": ["p1", "p2"], "mz": [rung5, 250.1], "height": [900.0, 5e3]}))
    monkeypatch.setattr(IO, "estimate_offset", lambda raw: 0.0)
    monkeypatch.setattr(A, "run", fake_assign)
    seen = {}
    real_strip = RW.strip_rung_rows

    def spy_strip(merged, rungs, **kw):
        seen["strip_tol"] = kw.get("tol_ppm")
        return real_strip(merged, rungs, **kw)
    monkeypatch.setattr(RW, "strip_rung_rows", spy_strip)
    with tempfile.TemporaryDirectory() as d:
        AB.run(peaks=ts, ts_peaks=ts, reagent="Br", batch="test batch", out_dir=d, k_min=2, k_max=3,
               min_gain=0.0, n_jobs=1, residual=False, tol_ppm=8.0, log=lambda *a: None)
        run = d
        merged = pd.read_csv(os.path.join(run, "merged_ledger.csv"))
        summ = json.load(open(os.path.join(run, "batch_summary.json")))
        table = pd.read_csv(os.path.join(run, "tables", "reagent_water.csv"))
        stamped = pd.read_parquet(os.path.join(run, "per_file", "_batch_ts.parquet"))
    assert merged["neutral_formula"].tolist() == ["C9H16O6"]                  # the rung reading left
    rw = summ["merge_gates"]["reagent_water"]
    assert rw["tol_ppm"] == pytest.approx(summ["traces"]["stamp_tol_ppm"])   # measured and stripped at the stamp window
    assert rw["tol_ppm"] >= 8.0                                               # the batch's window, not a fixed 6 ppm
    assert seen["strip_tol"] == pytest.approx(rw["tol_ppm"])
    assert rw["n_rungs"] == 6 and rw["n_stripped"] == 1 and rw["stripped"] == ["C4H4O8 [M-H]- (Br(79Br).(H2O)5)"]
    assert table.loc[table["n"] == 5, "displaced"].iloc[0] == "C4H4O8 [M-H]-"
    at5 = stamped[np.isclose(stamped["mz"], rung5)]
    assert set(at5["role"]) == {"reagent"} and set(at5["ion_formula"]) == {"H10BrO5-"}
    at250 = stamped[np.isclose(stamped["mz"], 250.1)]
    assert set(at250["neutral_formula"]) == {"C9H16O6"}
    # the pooled level stage ran with every file's persisted calibration, to its class gate (no width model: NA)
    el = summ["evidence_levels"]
    assert el["n_pairs"] > 0 and el["pooled"] == {"NA": el["n_pairs"]}
    assert all(pf["degeneracy_cal"] == {"mu": 0.0, "sigma": 0.3} for pf in summ["per_file"])


# --- the constants the rule stands on ----------------------------------------------------------------
def test_the_decoy_gate_is_three_times_the_most_present_decoy():
    def run(k_decoy):
        decoy = BR79 + 4 * W + 0.02
        spec = [(i * 10, _ladder(6) + ([decoy] if i < k_decoy else [])) for i in range(12)]
        return RW.detect(_ts(spec), ("Br",), tol_ppm=5.0)["n"].tolist()
    assert run(5) == [1, 2, 3, 5, 6]            # 3 x 5/12 = 1.25 > 1.0: fails (a 2x gate would pass it)
    assert run(3) == [1, 2, 3, 4, 5, 6]         # 3 x 3/12 = 0.75 <= 1.0: passes (a 5x gate would fail it)
    assert run(4) == [1, 2, 3, 4, 5, 6]         # 3 x 4/12 = 1.0: at least three times passes


@pytest.mark.parametrize("offset", (-0.05, -0.035, -0.02, 0.02, 0.035, 0.05))
def test_every_decoy_offset_is_read(offset):
    assert RW.DECOY_OFFSETS_DA == (-0.05, -0.035, -0.02, 0.02, 0.035, 0.05)
    decoy = BR79 + 4 * W + offset
    ts = _ts([(i * 10, _ladder(6, extra=(decoy,))) for i in range(12)])
    assert 4 not in RW.detect(ts, ("Br",), tol_ppm=5.0)["n"].tolist()


def test_a_ladder_is_read_up_to_n_45_and_no_further():
    ts = _ts([(i * 10, _ladder(50)) for i in range(12)])
    n = RW.detect(ts, ("Br",), tol_ppm=5.0)["n"].tolist()
    assert n == list(range(1, RW.N_MAX + 1)) and RW.N_MAX == 45


def test_the_core_must_be_present_in_half_the_segment():
    def run(k_core):
        spec = [(i * 10, ([BR79] if i < k_core else []) + [BR79 + n * W for n in range(1, 4)]) for i in range(12)]
        return RW.detect(_ts(spec), ("Br",), tol_ppm=5.0)["n"].tolist()
    assert run(5) == [] and run(6) == [1, 2, 3]


def test_a_rung_passes_in_any_segment_not_only_the_last():
    before = [(i * 10, _ladder(9)) for i in range(12)]
    after = [(300 + 200 + i * 10, _ladder(2)) for i in range(12)]
    assert RW.detect(_ts(before + after), ("Br",), tol_ppm=5.0)["n"].tolist() == list(range(1, 10))


def test_the_strip_window_is_the_stamp_window_and_takes_candidates_too():
    r = _rungs()
    at = float(r.loc[(r["n"] == 4) & (r["iso_tag"] == "79Br"), "mz_obs"].iloc[0])
    merged = pd.DataFrame({"mz": [at * (1 + 0.8 * 5e-6), at * (1 + 1.2 * 5e-6), at * (1 - 0.8 * 5e-6)],
                           "neutral_formula": ["C7H6O6", "C7H6O6x", "C8H10O5"],
                           "adduct": ["[M-H]-", "[M-H]-", "[M+NO3]-"], "tier": ["Assigned", "Assigned", "Candidate"]})
    kept, stripped = RW.strip_rung_rows(merged, r, tol_ppm=5.0, log=lambda *a: None)
    assert sorted(stripped["neutral_formula"]) == ["C7H6O6", "C8H10O5"]       # inside 5 ppm, Candidate too
    assert kept["neutral_formula"].tolist() == ["C7H6O6x"]                     # 6 ppm out


def test_strip_and_stamp_read_the_observed_rung_not_the_exact_mass():
    """A batch calibrated 8 ppm high: the ladder is found at its observed m/z, and a
    merged row on that peak is stripped although it is 8 ppm from the exact mass."""
    shift = 1 + 8e-6
    ts = _ts([(i * 10, [m * shift for m in _ladder(5)]) for i in range(12)])
    r = RW.detect(ts, ("Br",), tol_ppm=10.0)
    assert r["n"].tolist() == [1, 2, 3, 4, 5]
    obs = float(r.loc[r["n"] == 3, "mz_obs"].iloc[0])
    assert abs(obs / float(r.loc[r["n"] == 3, "mz"].iloc[0]) - shift) < 1e-7
    merged = pd.DataFrame({"mz": [obs], "neutral_formula": ["C6H8O7"], "adduct": ["[M-H]-"], "tier": ["Assigned"]})
    kept, stripped = RW.strip_rung_rows(merged, r, tol_ppm=5.0, log=lambda *a: None)
    assert kept.empty and len(stripped) == 1
    s = RW.stamp_rows(r)
    assert np.allclose(np.sort(s["mz"].to_numpy()), np.sort(r["mz_obs"].to_numpy()), rtol=0, atol=1e-6)


# --- the TOF rung test (a TOF-class width model) -------------------------------------------------------
from peaky.chem.resolution import Resolution  # noqa: E402

TOF = Resolution.from_r(10_000)          # R < 50 000 at m/z 200: TOF class
ORBI = Resolution.from_r(120_000)        # Orbitrap class


def _ts_h(spectra):
    """spectra: list of (minutes, [(m/z, height) ...]) -> a batch TS frame with heights."""
    rows = []
    for i, (minute, peaks) in enumerate(spectra):
        for m, h in peaks:
            rows.append(dict(sample_item_id=f"s{i:03d}", datetime_utc=T0 + pd.Timedelta(minutes=minute),
                             mz=float(m), height=float(h)))
    return pd.DataFrame(rows)


def _rung(n, core=BR79):
    return core + n * W


def test_the_tof_test_runs_only_on_a_tof_class_width_model():
    assert RW.tof_fwhm(None) is None and RW.tof_fwhm(ORBI) is None
    assert RW.tof_fwhm(TOF)(200.0) == pytest.approx(0.02)
    assert RW.tof_fwhm(TOF.as_dict())(600.0) == pytest.approx(0.06)
    assert RW.rung_test(ORBI) is None and RW.rung_test(None) is None
    rec = RW.rung_test(TOF)
    assert rec["test"] == "tof" and rec["decoy_fwhm"] == [2.0, 2.5, 3.0, 3.5, 4.0, 5.0]
    assert rec["link_presence"] == 0.25 and rec["covary_r"] == 0.8 and rec["m1_carbons"] == 5


def test_an_orbitrap_class_model_keeps_the_fixed_offset_test_exactly():
    decoy = BR79 + 4 * W + 0.035
    ts = _ts([(i * 10, _ladder(6, extra=(decoy,))) for i in range(12)])
    plain = RW.detect(ts, ("Br",), tol_ppm=5.0)
    orbi = RW.detect(ts, ("Br",), tol_ppm=5.0, resolution=ORBI)
    assert plain.to_csv(index=False) == orbi.to_csv(index=False)
    assert orbi["n"].tolist() == [1, 2, 3, 5, 6]
    out = RW.measure(ts, P.PROFILES["Br"], tol_ppm=5.0, log=lambda *a: None, resolution=ORBI)
    assert out["rung_test"] is None
    assert "rung_test" not in RW.summary(out["rungs"], None, n_cores=out["n_cores"], tol_ppm=5.0,
                                         segment_sizes=out["segment_sizes"], rung_test=out["rung_test"])


def test_the_lines_own_satellites_inside_two_fwhm_fail_the_fixed_offsets_not_the_tof_test():
    """A TOF picker reports weak satellites of a line about 1-1.5 FWHM below and ~1 FWHM above
    it. From m/z ~175 on (R 10 000) the fixed 0.035 / 0.02 Da decoys sit on them."""
    spec = []
    for i in range(12):
        peaks = [(BR79, 5000.0)]
        for n in range(1, 9):
            peaks.append((_rung(n), 1000.0))
            if n >= 6:
                peaks += [(_rung(n) - 0.035, 100.0), (_rung(n) + 0.02, 100.0)]
        spec.append((i * 10, peaks))
    ts = _ts_h(spec)
    assert RW.detect(ts, ("Br",), tol_ppm=5.0)["n"].tolist() == [1, 2, 3, 4, 5]
    r = RW.detect(ts, ("Br",), tol_ppm=5.0, resolution=TOF)
    assert r["n"].tolist() == list(range(1, 9))
    assert set(r["decoy_presence"]) == {"0.00"}


def _humid(*, link_hits, rung6=None, top=8, extra=()):
    """12 spectra whose ladder (Br79 n = 1..top) scales with a common factor A(t); rung 4 is
    seen in `link_hits` of them; rung 6's height is `rung6(A)` when given."""
    spec = []
    for i in range(12):
        a = 1.0 + i
        peaks = [(BR79, 5000.0 * a)]
        for n in range(1, top + 1):
            if n == 4 and i >= link_hits:
                continue
            h = 1000.0 * a / n
            if n == 6 and rung6 is not None:
                h = rung6(a)
            peaks.append((_rung(n), h))
        spec.append((i * 10, peaks + list(extra)))
    return _ts_h(spec)


def test_a_weak_rung_carries_the_ladder_past_it_and_the_rungs_beyond_must_co_vary():
    ts = _humid(link_hits=4)                                     # rung 4 in 4/12 = 0.33: weak, not absent
    assert RW.detect(ts, ("Br",), tol_ppm=5.0)["n"].tolist() == [1, 2, 3]       # fixed: the ladder ends
    assert RW.detect(ts, ("Br",), tol_ppm=5.0, resolution=TOF)["n"].tolist() == [1, 2, 3, 5, 6, 7, 8]
    # rung 6 runs AGAINST its ladder (r = -1 with its neighbours): it is not a rung of it; each
    # other rung takes the median over its present neighbours within +-2, so one such line
    # does not sink them
    anti = _humid(link_hits=4, rung6=lambda a: 1000.0 * (13.0 - a), top=9)
    assert RW.detect(anti, ("Br",), tol_ppm=5.0, resolution=TOF)["n"].tolist() == [1, 2, 3, 5, 7, 8, 9]


def test_an_absent_rung_ends_the_ladder_an_island_beyond_it_does_not_pass():
    ts = _humid(link_hits=2)                                     # rung 4 in 2/12 = 0.17 < 0.25: absent
    assert RW.detect(ts, ("Br",), tol_ppm=5.0, resolution=TOF)["n"].tolist() == [1, 2, 3]
    ts = _humid(link_hits=3)                                     # 3/12 = 0.25: a link
    assert RW.detect(ts, ("Br",), tol_ppm=5.0, resolution=TOF)["n"].tolist() == [1, 2, 3, 5, 6, 7, 8]


def test_the_tof_decoys_are_the_mean_local_chance_not_the_most_present_one():
    fw = TOF.fwhm(_rung(4))
    one = (_rung(4) + 2.0 * fw,)                                  # one real ion 2 FWHM above rung 4
    ts = _ts([(i * 10, _ladder(6, extra=one)) for i in range(12)])
    r = RW.detect(ts, ("Br",), tol_ppm=5.0, resolution=TOF)
    assert r["n"].tolist() == [1, 2, 3, 4, 5, 6]
    assert r.set_index("n").loc[4, "decoy_presence"] == "0.08"   # 1 of 12 decoys present in every spectrum
    crowded = tuple(_rung(4) + s * k * fw for k in RW.TOF_DECOY_FWHM for s in (-1, 1))
    ts = _ts([(i * 10, _ladder(6, extra=crowded)) for i in range(12)])
    assert 4 not in RW.detect(ts, ("Br",), tol_ppm=5.0, resolution=TOF)["n"].tolist()


def test_a_decoy_on_another_declared_ladder_is_skipped():
    """Br3(79Br+81Br+81Br).(H2O)21 sits 0.2 FWHM from the -4 FWHM decoy of Br(79Br).(H2O)30 at
    R 10 000: a peak there is that ladder, not chance -- with the Br3 core declared the decoy
    is skipped."""
    br3 = [c for c in RW.cores(("Br3",)) if c.tag == "79Br+81Br+81Br"][0]
    fw = TOF.fwhm(_rung(30))
    decoy = _rung(30) - 4.0 * fw
    assert abs(br3.mz + 21 * W - decoy) < 0.25 * fw
    ts = _ts([(i * 10, _ladder(30, extra=(decoy,))) for i in range(12)])
    alone = RW.detect(ts, ("Br",), tol_ppm=5.0, resolution=TOF).set_index("n")
    both = RW.detect(ts, ("Br", "Br3"), tol_ppm=5.0, resolution=TOF)
    both = both[both["core"] == "Br"].set_index("n")
    assert alone.loc[30, "decoy_presence"] == "0.08"            # counted as chance with Br3 undeclared
    assert both.loc[30, "decoy_presence"] == "0.00"             # skipped when its ladder is declared


def _bright(m1_height, *, rung_height=1000.0, noise=1.0):
    """12 spectra: Br79 n = 1..6 at `rung_height`, rung 5 with an M+1 line of `m1_height`,
    and weak picker-floor peaks half a mass unit off every rung."""
    spec = []
    for i in range(12):
        peaks = [(BR79, 5000.0)] + [(_rung(n), rung_height) for n in range(1, 7)]
        peaks += [(_rung(n) + 0.5, noise) for n in range(0, 7)] + [(_rung(n) - 0.5, noise) for n in range(1, 7)]
        if m1_height:
            peaks.append((_rung(5) + 1.003355, m1_height))
        spec.append((i * 10, peaks))
    return _ts_h(spec)


def test_a_bright_rung_whose_m1_line_holds_carbon_is_not_the_rung():
    # 13C line at 0.20x: about 19 carbons over the cluster's own M+1 -- an organic sits there
    assert RW.detect(_bright(200.0), ("Br",), tol_ppm=5.0, resolution=TOF)["n"].tolist() == [1, 2, 3, 4, 6]
    # at 0.01x the line is the cluster's own 17O / 2H (+ under one carbon): the rung passes
    assert RW.detect(_bright(10.0), ("Br",), tol_ppm=5.0, resolution=TOF)["n"].tolist() == [1, 2, 3, 4, 5, 6]
    # the fixed-offset test never asks
    assert RW.detect(_bright(200.0), ("Br",), tol_ppm=5.0)["n"].tolist() == [1, 2, 3, 4, 5, 6]


def test_the_m1_test_only_judges_a_rung_bright_enough_to_show_five_carbons():
    """At height 5 over a floor of 1 a C5 line (0.27) could not have been seen: the M+1
    line at 0.4x is not held against the rung."""
    assert RW.detect(_bright(2.0, rung_height=5.0), ("Br",), tol_ppm=5.0,
                     resolution=TOF)["n"].tolist() == [1, 2, 3, 4, 5, 6]


def test_the_cluster_m1_ratio_counts_its_own_atoms():
    br = RW.cores(("Br",))[0]
    no3 = RW.cores(("NO3",))[0]
    assert RW.cluster_m1_ratio(br, 0) == 0.0
    assert RW.cluster_m1_ratio(br, 10) == pytest.approx(20 * 0.000115 + 10 * 0.00038 / 0.99757, rel=1e-9)
    assert RW.cluster_m1_ratio(no3, 1) == pytest.approx(2 * 0.000115 + 0.00368 / 0.99632 + 4 * 0.00038 / 0.99757,
                                                        rel=1e-9)


def test_measure_and_summary_record_the_tof_test():
    ts = _ts([(i * 10, _ladder(4)) for i in range(12)])
    out = RW.measure(ts, P.PROFILES["Br"], tol_ppm=5.0, log=lambda *a: None, resolution=TOF.as_dict())
    assert out["rung_test"]["test"] == "tof" and out["rung_test"]["fwhm_at_200"] == pytest.approx(0.02)
    s = RW.summary(out["rungs"], None, n_cores=out["n_cores"], tol_ppm=5.0, segment_sizes=out["segment_sizes"],
                   rung_test=out["rung_test"])
    assert s["rung_test"] == out["rung_test"] and s["n_rungs"] == 4


def test_a_tof_batch_strips_the_rung_its_satellites_hid_and_records_the_test(monkeypatch, tmp_path):
    from peaky.assignment import assign as A
    from peaky.assignment import ledger as L
    from peaky.assignment import tiers as T
    from peaky.io import io_mascope as IO

    rung7 = _rung(7)
    spec = []
    for i in range(12):
        peaks = [(BR79, 5000.0)] + [(_rung(n), 1000.0) for n in range(1, 9)] + [(250.1, 3000.0)]
        peaks += [(m, 100.0) for n in range(6, 9) for m in (_rung(n) - 0.035, _rung(n) + 0.02)]
        spec.append((i * 10, peaks))
    ts = _ts_h(spec)

    def fake_assign(sid, context="ambient-air", **kw):
        led = L.new_ledger(pd.DataFrame([("p1", rung7, 1000.0), ("p2", 250.1, 3e3)],
                                        columns=["peak_id", "mz", "height"]))
        L.commit_assignment(led, "p1", neutral_formula="C7H10O7", adduct="[M-H]-", ion_formula="C7H9O7-",
                            ion_score=0.9, compound_score=0.9, ppm_error=0.1, pass_no=1,
                            method="cheminfo+grid", confidence="High", commentary="stub")
        L.commit_assignment(led, "p2", neutral_formula="C9H16O6", adduct="[M-H]-", ion_formula="C9H15O6-",
                            ion_score=0.9, compound_score=0.9, ppm_error=0.1, pass_no=1,
                            method="cheminfo+grid", confidence="High", commentary="stub")
        T.apply_tiers(led)
        return {"ledger": led, "stats": {"noise_edge_cps": 4.0, "height_gate_cps": 10.0,
                                         "degeneracy_cal": {"mu": 0.0, "sigma": 0.3}},
                "plausibility_audit": [], "summaries": {}, "problems": []}

    monkeypatch.setattr(IO, "connect", lambda *a, **k: "CLIENT")
    monkeypatch.setattr(IO, "fetch_peaks", lambda client, sid, use_cache=True: pd.DataFrame(
        {"peak_id": ["p1", "p2"], "mz": [rung7, 250.1], "height": [1000.0, 3e3]}))
    monkeypatch.setattr(IO, "estimate_offset", lambda raw: 0.0)
    monkeypatch.setattr(A, "run", fake_assign)

    def batch(out, rp):
        AB.run(peaks=ts, ts_peaks=ts, reagent="Br", batch="test batch", out_dir=str(out), k_min=2, k_max=3,
               min_gain=0.0, n_jobs=1, residual=False, tol_ppm=5.0, resolving_power=rp, log=lambda *a: None)
        return (pd.read_csv(out / "merged_ledger.csv"),
                json.load(open(out / "batch_summary.json"))["merge_gates"]["reagent_water"])

    merged, rw = batch(tmp_path / "tof", 10_000)
    assert merged["neutral_formula"].tolist() == ["C9H16O6"]                 # the rung reading left
    assert rw["rung_test"]["test"] == "tof" and rw["n_rungs"] == 8
    assert rw["stripped"] == ["C7H10O7 [M-H]- (Br(79Br).(H2O)7)"]
    merged, rw = batch(tmp_path / "orbi", 120_000)                           # the fixed-offset test
    assert sorted(merged["neutral_formula"]) == ["C7H10O7", "C9H16O6"]
    assert "rung_test" not in rw and rw["n_rungs"] == 5


def test_a_single_declared_core_runs_the_tof_test():
    no3 = RW.cores(("NO3",))[0].mz
    ts = _ts([(i * 10, [no3] + [no3 + n * W for n in range(1, 5)]) for i in range(12)])
    assert RW.detect(ts, ("NO3",), tol_ppm=5.0, resolution=TOF)["n"].tolist() == [1, 2, 3, 4]
    assert RW.detect(ts.drop(columns=["height"]), ("NO3",), tol_ppm=5.0, resolution=TOF)["n"].tolist() == [1, 2, 3, 4]
