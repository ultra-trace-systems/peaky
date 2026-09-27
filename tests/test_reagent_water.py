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
        return {"ledger": led, "stats": {"noise_edge_cps": 4.0, "height_gate_cps": 10.0},
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
