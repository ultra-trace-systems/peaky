"""scripts/scorecard.py — one synthetic run dir, every panel checked by hand.

The fixture is a four-spectrum batch with three assigned acids, one of them
with a 13C satellite, a merged row the stamp never carried, and the unstamped
tracks that make the misses: three +H lines (the electron-attachment family),
a shoulder, an unknown, and one that is not persistent. Each number the
panels report can be counted on paper.

Run: pytest tests/test_scorecard.py -q
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import scorecard as SC  # noqa: E402
from peaky.assignment import evidence as EV  # noqa: E402
from peaky.chem import chemistry as C  # noqa: E402
from peaky.chem.resolution import Resolution  # noqa: E402

D13C = 1.0033548
H = C.M["H"]
SAMPLES = ["s1", "s2", "s3", "s4"]
T0 = pd.Timestamp("2026-01-01T00:00:00Z")

# the three acids, their bare-channel ions and heights
A = ("C10H16O3", C.ion_mz("C10H16O3", "[M-H]-"), 1000.0)   # pinonic acid; 13C satellite in the series
B = ("C9H14O4", C.ion_mz("C9H14O4", "[M-H]-"), 500.0)      # pinic acid
Cc = ("C8H12O4", C.ion_mz("C8H12O4", "[M-H]-"), 250.0)     # terpenylic acid
E = ("C22H42O4", C.ion_mz("C22H42O4", "[M-H]-"), 120.0)    # DEHA: the C>20 contaminant
D = ("C5H10O6", C.ion_mz("C5H10O6", "[M-H]-"), 90.0)       # merged, never stamped
ORBI = Resolution(coef=0.002 / 200 ** 1.5, exponent=1.5, offset=0.0).as_dict()     # R(200) = 100 000
TOF = Resolution(coef=1.0 / 9500.0, exponent=1.0, n_peaks=9, source="measured").as_dict()


def ion_of(neutral: str, adduct: str = "[M-H]-") -> str:
    """The ion formula a ledger writes: the ion's composition with its charge sign."""
    return C.format_formula(EV.ion_composition(neutral, adduct, None)) + adduct[-1]

UNSTAMPED = [  # (mz, height, in how many spectra)
    (A[1] + H, 80.0, 4),          # +H family
    (B[1] + H, 60.0, 4),          # +H family
    (Cc[1] + H, 50.0, 4),         # +H family
    (D[1], 90.0, 4),              # the merged-but-unstamped row's own track
    (A[1] + 0.006, 30.0, 4),      # a shoulder of A
    (250.0, 40.0, 4),             # unknown
    (300.0, 500.0, 1),            # bright but in one spectrum: not persistent
]


def _ts() -> pd.DataFrame:
    rows = []
    for i, sid in enumerate(SAMPLES):
        t = T0 + pd.Timedelta(hours=i)
        for j, (neutral, mz, h) in enumerate((A, B, Cc, E)):
            rows.append(dict(sample_item_id=sid, datetime_utc=t, peak_id=f"{sid}-p{j}", mz=mz, height=h, area=h * 1.2,
                             neutral_formula=neutral, adduct="[M-H]-", tier="Assigned", ion_mz=mz, role="M0",
                             ion_formula=ion_of(neutral), iso_label=None,
                             stamp_source="M0", dup_candidate=False, intensity_suspect=False))
        # A's 13C satellite: 10 carbons x 1.07 %
        rows.append(dict(sample_item_id=sid, datetime_utc=t, peak_id=f"{sid}-iso", mz=A[1] + D13C, height=107.0, area=128.0,
                         neutral_formula=None, adduct=None, tier=None, ion_mz=A[1] + D13C, role="iso_child",
                         ion_formula="C10H15O3-", iso_label="13C", stamp_source="observed", dup_candidate=False, intensity_suspect=False))
        for k, (mz, h, n) in enumerate(UNSTAMPED):
            if i < n:
                rows.append(dict(sample_item_id=sid, datetime_utc=t, peak_id=f"{sid}-u{k}", mz=mz, height=h, area=h,
                                 neutral_formula=None, adduct=None, tier=None, ion_mz=np.nan, role=None, ion_formula=None,
                                 iso_label=None, stamp_source=None, dup_candidate=False, intensity_suspect=False))
    return pd.DataFrame(rows)


def _per_file(sid: str) -> pd.DataFrame:
    rows = []
    for j, (neutral, mz, h) in enumerate((A, B, Cc, E, D)):
        ion = ion_of(neutral)
        isos = json.dumps([{"label": "13C", "score": 0.95, "peak_id": f"{sid}-iso"}]) if neutral == A[0] else "[]"
        rows.append(dict(sample_item_id=sid, peak_id=f"{sid}-p{j}", mz=mz, height=h, area=h * 1.2, role="M0",
                         neutral_formula=neutral, adduct="[M-H]-", ion_formula=ion, tier="Assigned", method="cheminfo+grid",
                         confidence="High", tied=False, below_assignability=False, degeneracy_density=1.0, degeneracy_note="unique",
                         isotopologues=isos, parent_peak_id=None, iso_label=None, commentary="", anchor_peak_id=None, series_unit=None,
                         ppm_error=0.0, ppm_error_cal=0.0))
    rows.append(dict(sample_item_id=sid, peak_id=f"{sid}-iso", mz=A[1] + D13C, height=107.0, area=128.0, role="iso_child",
                     neutral_formula=None, adduct=None, ion_formula="C10H15O3-", tier=None, method=None, confidence=None, tied=False,
                     below_assignability=False, degeneracy_density=np.nan, degeneracy_note=None, isotopologues=None,
                     parent_peak_id=f"{sid}-p0", iso_label="13C", commentary="", anchor_peak_id=None, series_unit=None))
    for k, (mz, h, n) in enumerate(UNSTAMPED):
        if k == 3:
            continue  # D is an M0 in the per-file ledgers (the merge kept it; the stamp lost it)
        why = "CLEARED (mass-gate: z=8.7 > 4.0). Was: Pass 1 (cheminfo+grid): C9H17NO [M+NO3]-, ion score 0.73, ppm -2.09" if k == 0 else ""
        rows.append(dict(sample_item_id=sid, peak_id=f"{sid}-u{k}", mz=mz, height=h, area=h, role="unexplained",
                         neutral_formula=None, adduct=None, ion_formula=None, tier=None, method=None, confidence=None, tied=False,
                         below_assignability=False, degeneracy_density=np.nan, degeneracy_note=None, isotopologues=None,
                         parent_peak_id=None, iso_label=None, commentary=why, anchor_peak_id=None, series_unit=None))
    return pd.DataFrame(rows)


def _merged() -> pd.DataFrame:
    rows = []
    for neutral, mz, h in (A, B, Cc, E, D):
        rows.append(dict(mz=mz, neutral_formula=neutral, adduct="[M-H]-", tier="Assigned", n_files=2, stage="cover",
                         trace_role="single", tier_reason="", alternatives="", intensity_suspect=False))
    return pd.DataFrame(rows)


def write_run(root: Path, name: str = "TEST-BATCH_2026-01-01T000000Z", resolution=None) -> Path:
    """The fixture run; ``resolution`` = a width model recorded in its summary (none: class-less, levels read NA)."""
    run = root / name
    (run / "per_file").mkdir(parents=True)
    (run / "tables").mkdir()
    _merged().to_csv(run / "merged_ledger.csv", index=False)
    _ts().to_parquet(run / "per_file" / "_batch_ts.parquet", index=False)
    for sid in SAMPLES[:2]:
        _per_file(sid).to_csv(run / "per_file" / f"{sid}_ledger.csv", index=False)
    summary = {
        "reagent": "NO3", "batch_name": "TEST BATCH", "tol_ppm": 6.0, "n_files": 2, "trace_first": None,
        "merge_gates": {"reagent_n": {"reagent_n_relabeled": 1}}, "elapsed_s": 60.0,
        "n_in_all_files": 5, "n_single_file": 0, "ion_disagreements": 0}
    if resolution is not None:
        summary.update(resolution=resolution, context="ambient-air", reflists_active=[],
                       per_file=[dict(sample_id=sid, height_gate_cps=10.0, degeneracy_cal={"mu": 0.0, "sigma": 0.3})
                                 for sid in SAMPLES[:2]])
    (run / "batch_summary.json").write_text(json.dumps(summary))
    (run / "run_manifest.json").write_text(json.dumps({"code": {"package_version": "0.0-test", "git": {"commit": "abcdef0123456", "dirty": False}}}))
    return run


@pytest.fixture
def run_dir(tmp_path):
    return write_run(tmp_path / "out")


@pytest.fixture
def run(run_dir):
    return SC.load_run(str(run_dir))


@pytest.fixture
def rosters(tmp_path):
    d = tmp_path / "rosters"
    d.mkdir()
    pd.DataFrame({"name": ["pinonic acid", "pinic acid", "HOM C10H16O10", "hydroxy acid"],
                  "formula": ["C10H16O3", "C9H14O4", "C10H16O10", "C5H10O6"],
                  "class": ["monomer", "monomer", "HOM", "monomer"],
                  "reference": ["x"] * 4, "note": [""] * 4}).to_csv(d / "alpha_pinene.csv", index=False)
    pd.DataFrame({"name": ["DEHA"], "formula": ["C22H42O4"], "class": ["plasticiser"], "reference": ["x"], "note": [""]}).to_csv(d / "contaminants.csv", index=False)
    return SC.load_rosters(str(d))


def test_resolve_run_dir_accepts_parent_or_run_dir_and_refuses_ambiguity(run_dir, tmp_path):
    assert SC.resolve_run_dir(str(run_dir)) == str(run_dir)
    assert SC.resolve_run_dir(str(run_dir.parent)) == str(run_dir)
    write_run(tmp_path / "two", "A_2026-01-01T000000Z")
    write_run(tmp_path / "two", "B_2026-01-01T000000Z")
    with pytest.raises(SystemExit):
        SC.resolve_run_dir(str(tmp_path / "two"))


def test_load_run_reads_the_profile_and_the_artifacts(run):
    assert run.reagent == "NO3" and run.adducts == ["[M+NO3]-", "[M-H]-"] and run.polarity == "-"
    assert run.n_spectra == 4 and run.path_kind == "cover" and len(run.per_file) == 2 * 12
    assert run.channel == "TEST-BATCH|NO3|cover" and run.code == "0.0-test abcdef012"


def test_ion_table_and_tracks_count_what_the_series_holds(run):
    ions = SC.ion_table(run)
    assert len(ions) == 5 and (ions["role"] == "M0").sum() == 4 and (ions["role"] == "iso_child").sum() == 1
    assert (ions["presence"] == 1.0).all()
    tracks = SC.unstamped_tracks(run)
    assert len(tracks) == 7
    assert (tracks["presence"] >= 0.5).sum() == 6           # the 300.0 track sits in one spectrum


def test_headline_stamped_shares_and_stamp_coverage(run):
    ions = SC.ion_table(run)
    h = SC.headline(run, ions)
    stamped = 4 * (1000 + 500 + 250 + 120 + 107)
    unst = 4 * (80 + 60 + 50 + 90 + 30 + 40) + 500
    assert h["n_peaks"] == 4 * 5 + 25
    assert abs(h["stamped_signal_share"] - 100 * stamped / (stamped + unst)) < 1e-6
    assert h["assigned"] == 5 and h["candidate"] == 0 and h["neutrals"] == 5
    cov = h["stamp_coverage"]
    assert cov["n_unstamped"] == 1 and cov["rows"][0]["neutral"] == "C5H10O6" and cov["reagent_n_relabeled"] == 1


def _stamp_scale(run_dir: Path, levels: dict) -> None:
    """Write the evidence scale's columns on the merged ledger, as a run made with the scale does."""
    led = pd.read_csv(run_dir / "merged_ledger.csv")
    lv = [levels.get((n, a), "4b") for n, a in zip(led.neutral_formula, led.adduct)]
    for c in EV.COLUMNS:
        led[c] = ""
    led["evidence_level"] = lv
    led["would_lift"] = [f"lift {v}" for v in lv]
    led["claim"] = [EV.claim_class(v) for v in lv]
    led.to_csv(run_dir / "merged_ledger.csv", index=False)


def test_a_class_less_run_is_levelled_post_hoc_and_reads_na(run):
    lv = SC.levels_for(run, None, [])
    assert set(lv["level"]) == {"NA"} and set(lv["source"]) == {"post-hoc"} and len(lv) == 5
    assert SC.level_vector(lv) == {"3c": 0, "4a": 0, "4b": 0, "5a": 0, "5b": 0, "reagent": 0, "NA": 5}
    assert {SC.claim_of(v) for v in lv["level"]} == {"not assessed"}


def test_an_orbitrap_run_is_levelled_post_hoc_by_the_reference_exactly_as_the_engine_levels_it(tmp_path):
    rd = write_run(tmp_path / "orbi", resolution=ORBI)
    run = SC.load_run(str(rd))
    assert not run.scale
    lv = SC.levels_for(run, None, [])
    core = EV.level_source(EV.source_from_run_dir(str(rd)))
    got = dict(zip(zip(lv["neutral"], lv["adduct"]), lv["level"]))
    assert got == dict(zip(zip(core["neutral_formula"], core["adduct"]), core["evidence_level"]))
    assert "NA" not in set(lv["level"]) and set(lv["source"]) == {"post-hoc"}
    own = SC.own_levels_for(run)
    assert dict(zip(zip(own["neutral"], own["adduct"]), own["level"])) == got


def test_the_in_core_level_is_preferred_and_its_na_survives_the_csv(run_dir):
    _stamp_scale(run_dir, {(A[0], "[M-H]-"): "3c", (B[0], "[M-H]-"): "NA"})
    run = SC.load_run(str(run_dir))
    assert run.scale and run.ledger["evidence_level"].tolist().count("NA") == 1   # the literal bucket, not NaN
    lv = SC.levels_for(run, None, [])
    assert set(lv["source"]) == {"in-core"}
    assert SC.level_vector(lv) == {"3c": 1, "4a": 0, "4b": 3, "5a": 0, "5b": 0, "reagent": 0, "NA": 1}
    assert lv.set_index("neutral").loc[A[0], "would_lift"] == "lift 3c"


def test_a_pre_scale_merged_level_is_never_read_and_a_pre_scale_levels_csv_is_refused(run_dir, tmp_path):
    led = pd.read_csv(run_dir / "merged_ledger.csv")
    led["evidence_level"], led["evidence_axes"] = "3a", "iso|chan2"           # a pre-0.10.0 run
    led.to_csv(run_dir / "merged_ledger.csv", index=False)
    run = SC.load_run(str(run_dir))
    assert not run.scale and set(SC.levels_for(run, None, [])["level"]) == {"NA"}
    old = tmp_path / "old_levels.csv"
    pd.DataFrame({"source": [run.name], "neutral": [A[0]], "adduct": ["[M-H]-"], "level": ["4b"]}).to_csv(old, index=False)
    with pytest.raises(SystemExit, match="pre-0.10.0 levels table"):
        SC.levels_for(run, str(old), [])
    new = tmp_path / "levels.csv"
    pd.DataFrame({"source": [run.name], "neutral_formula": [A[0]], "adduct": ["[M-H]-"], "evidence_level": ["NA"],
                  "would_lift": [""]}).to_csv(new, index=False)
    lv = SC.levels_for(run, str(new), [])
    assert lv["level"].tolist() == ["NA"] and set(lv["source"]) == {"csv"}


def test_brightest_ranks_by_median_height_and_counts_the_unstamped(run_dir):
    _stamp_scale(run_dir, {(A[0], "[M-H]-"): "3c", (B[0], "[M-H]-"): "NA"})
    run = SC.load_run(str(run_dir))
    ions, tracks, lv = SC.ion_table(run), SC.unstamped_tracks(run), SC.levels_for(run, None, [])
    b = SC.brightest(run, ions, tracks, lv, n=3)
    assert [r["neutral"] for r in b["rows"]] == ["C10H16O3", "C9H14O4", "C8H12O4"]
    assert [(r["level"], r["claim"]) for r in b["rows"]] == [("3c", "identified"), ("NA", "not assessed"), ("4b", "ion")]
    assert b["m0_not_assigned"] == 0 and b["m0_not_identified"] == 1 and b["m0_not_assessed"] == 1
    # the batch's three brightest tracks overall: A 1000, B 500 and the 1-spectrum 500 track
    assert b["unstamped_in_top"] == 1


def test_best_evidence_orders_by_level_then_brightness(run_dir):
    _stamp_scale(run_dir, {(B[0], "[M-H]-"): "3c", (A[0], "[M-H]-"): "4a", (Cc[0], "[M-H]-"): "5b"})
    run = SC.load_run(str(run_dir))
    e = SC.best_evidence(run, SC.ion_table(run), SC.levels_for(run, None, []), n=3)
    assert [r["neutral"] for r in e["rows"]] == [B[0], A[0], E[0]]
    assert e["n_good"] == 1 and e["n_established"] == 2 and e["rows"][0]["would_lift"] == "lift 3c"
    assert e["by_claim"] == {"identified": 1, "neutral": 1, "ion": 2, "tentative": 1, "reagent": 0, "not assessed": 0}


def test_m1_groups_the_plus_h_comb_into_one_family_and_names_the_rest(run):
    ions, tracks = SC.ion_table(run), SC.unstamped_tracks(run)
    cov = SC.stamp_coverage(run, ions)["rows"]
    m1 = SC.missed_m1(run, ions, tracks, coverage_rows=cov)
    fam = {f["family"]: f for f in m1["families"]}
    assert m1["n_tracks"] == 6 and m1["n_families"] == 1
    assert fam["+H (M-. electron attachment / H adduct)"]["n_tracks"] == 3
    assert fam["merged but unstamped (stamp-coverage row)"]["n_tracks"] == 1
    assert fam["shoulder / sidelobe of a brighter stamped ion"]["n_tracks"] == 1
    assert fam["unknown"]["n_tracks"] == 1
    by_mz = {round(r["mz"], 3): r for r in m1["rows"]}
    comb = by_mz[round(A[1] + H, 3)]
    assert comb["parent"].startswith("C10H16O3 [M-H]-") and abs(comb["d_mda"] - (H - D13C) * 1000) < 0.05
    assert comb["reason"] == "mass-gate: z=8.7 > 4.0; tried C9H17NO [M+NO3]-"
    assert by_mz[round(D[1], 3)]["parent"] == "C5H10O6 [M-H]- Assigned"


def test_m2_tells_assigned_from_unstamped_from_absent(run, rosters):
    ions, tracks = SC.ion_table(run), SC.unstamped_tracks(run)
    m2 = SC.missed_m2(run, ions, tracks, rosters)
    status = {r["neutral"]: r["status"] for r in m2["rows"] if r["source"].startswith("roster:")}
    assert status["C10H16O3"] == "assigned" and status["C9H14O4"] == "assigned"
    assert status["C10H16O10"] == "absent"
    assert status["C5H10O6"] == "unstamped"                 # merged, in the series, never stamped
    assert status["C22H42O4"] == "assigned"
    assert m2["roster"]["alpha_pinene"] == {"n": 4, "assigned": 2, "candidate": 0, "read_as": 0, "unstamped": 1, "absent": 1, "present": 3}
    assert m2["roster_by_class"]["alpha_pinene"]["HOM"]["absent"] == 1
    names = {r["neutral"]: r["name"] for r in m2["rows"] if r["source"] == "roster:alpha_pinene"}
    assert names["C10H16O3"] == "pinonic acid"


def test_census_counts_heteroatoms_and_heavy_carbon(run):
    cz = SC.census(run)
    assert cz["n_assigned"] == 5 and cz["elements"]["C>20"] == 1 and cz["elements"]["Cl"] == 0
    assert cz["examples"]["C>20"].startswith("C22H42O4")


def test_falsification_reads_the_13c_carbon_count(run):
    fz = SC.falsification(run)
    assert fz["c13"]["n"] == 2 and fz["c13"]["within_1"] == 2 and fz["c13"]["median_abs_delta"] < 0.01
    assert fz["hetero"] is None and fz["iso_cov"] is None       # no heteroatoms; too few spectra for r


def test_decoy_transforms_and_the_wrong_adduct_sets(run):
    peaks = SC.raw_peaks_of(run, "s1")
    assert list(peaks.columns[:len(SC.PEAK_COLS)]) == SC.PEAK_COLS and len(peaks) == 12
    shifted = SC.decoy_peaks(peaks, 0.35)
    assert np.allclose(shifted["mz"] - peaks["mz"], 0.35) and shifted["peak_id"].str.endswith("_decoy").all()
    assert SC.wrong_adducts("-") == ["[M+Cl]-", "[M+I]-"] and SC.wrong_adducts("+") == ["[M+Na]+", "[M+NH4]+"]
    assert SC.brightest_files(run, 1) == ["s1"]


def test_decoy_runs_the_engine_offline_and_the_control_finds_the_acids(run):
    from peaky.io import io_mascope as IO
    real_connect = IO.connect
    IO.connect = lambda *a, **k: (_ for _ in ()).throw(AssertionError("offline run must not connect"))
    try:
        dc = SC.decoy(run, "shift", 0.35, 1)
    finally:
        IO.connect = real_connect
    assert dc["files"] == ["s1"] and dc["control"]["assigned"] >= 2
    assert dc["control"]["assigned_lt_350"] + dc["control"]["assigned_ge_350"] == dc["control"]["assigned"]
    assert dc["shift"]["assigned"] <= dc["control"]["assigned"] and "assigned_rate" in dc["shift"]


def test_m3_names_what_the_other_path_and_instrument_found(run, tmp_path):
    other_dir = write_run(tmp_path / "other")
    led = pd.read_csv(other_dir / "merged_ledger.csv")
    led = pd.concat([led, pd.DataFrame([dict(mz=200.0, neutral_formula="C7H10O5", adduct="[M-H]-", tier="Assigned", n_files=3,
                                             stage="cover", trace_role="single", tier_reason="", alternatives="", intensity_suspect=False)])])
    led.to_csv(other_dir / "merged_ledger.csv", index=False)
    other = SC.load_run(str(other_dir))
    lv_other = SC.levels_for(other, None, [])
    m3 = SC.missed_m3(run, other, other, lv_other, None, [], floor_cps=10.0, floor_share=0.8)
    assert m3["other_path"]["n_missing"] == 1 and m3["other_path"]["rows"][0]["neutral"] == "C7H10O5"
    # the other instrument's rows all exist here too (same fixture), so nothing is missing
    assert m3["other_instrument"]["n_missing"] == 0 and m3["other_instrument"]["n_spectra_in_window"] == 4
    # a window that excludes every spectrum leaves nothing to compare
    win = (pd.Timestamp("2027-01-01T00:00Z"), pd.Timestamp("2027-01-02T00:00Z"))
    m3w = SC.missed_m3(run, None, other, lv_other, win, [], 10.0, 0.8)
    assert m3w["other_instrument"]["n_spectra_in_window"] == 0 and m3w["other_instrument"]["n_missing"] == 0


def test_m3_cuts_at_4b_and_counts_the_assigned_rows_of_an_instrument_it_does_not_assess(run, tmp_path):
    """M3 counts the other instrument's neutrals at level <= 4b on the scale; where its rows are not assessed (a
    TOF: NA) it counts its Assigned rows instead, and its basis says so -- once by its in-core level and once by
    its own evidence (its files pooled with no other-source partner)."""
    import dataclasses
    other_dir = write_run(tmp_path / "other", resolution=TOF)
    _stamp_scale(other_dir, {(A[0], "[M-H]-"): "3c", (B[0], "[M-H]-"): "4b", (Cc[0], "[M-H]-"): "5a",
                             (E[0], "[M-H]-"): "5b", (D[0], "[M-H]-"): "4a"})
    other = SC.load_run(str(other_dir))
    in_core = SC.levels_for(other, None, [])
    own = SC.own_levels_for(other)
    assert set(own["level"]) == {"NA"}                               # a TOF-class source: not assessed
    lacks = dataclasses.replace(run, ledger=run.ledger[~run.ledger.neutral_formula.isin([A[0], B[0], Cc[0]])])
    m3 = SC.missed_m3(lacks, None, other, in_core, None, [], 10.0, 0.8, own_levels=own)
    oi, oo = m3["other_instrument"], m3["other_instrument_own"]
    assert oi["basis"] == SC.M3_BASIS_LEVEL and oi["n_good"] == 3          # A 3c, B 4b, D 4a (D never stamped)
    assert [r["neutral"] for r in oi["rows"]] == [A[0], B[0]] and oi["n_missing_by_claim"]["identified"] == 1
    assert oo["basis"] == SC.M3_BASIS_ASSIGNED and oo["n_good"] == 5         # every Assigned row
    assert oo["n_missing"] == 3 and oo["n_missing_by_claim"]["not assessed"] == 3
    card = SC.build_card(lacks, other_instrument=other, rosters=SC.load_rosters(), board=[], log=lambda *a: None)
    row = card["row"]
    assert row["m3_other_instrument_missing"] == 2 and row["m3_other_instrument_own_missing"] == 3
    assert row["m3_own_basis"] == SC.M3_BASIS_ASSIGNED
    crit = {a["key"]: a for a in card["acceptance"]}["m3_other_instrument_own_missing"]
    assert crit["value"] == 3 and crit["criterion"].endswith(SC.M3_BASIS_ASSIGNED)
    assert "rows (Assigned rows (the other instrument is not assessed on this scale)) pass the floor" in SC.render_md(card)
    assert SC.missed_m3(run, None, None, None, None, [], 10.0, 0.8)["other_instrument_own"] is None


def test_card_board_and_pages_round_trip_with_a_delta(run, rosters, tmp_path):
    out = tmp_path / "board"
    card = SC.build_card(run, rosters=rosters, board=[], log=lambda *a: None)
    assert card["previous"] is None and card["row"]["channel"] == "TEST-BATCH|NO3|cover"
    written = SC.write_outputs([card], str(out), log=lambda *a: None)
    md = Path(written["cards"][0]).read_text()
    assert "+H (M-. electron attachment / H adduct) | 3 |" in md and "C5H10O6" in md and "first row for this channel" in md
    board = SC.read_board(str(out / "scoreboard.jsonl"))
    assert len(board) == 1 and board[0]["assigned"] == 5 and board[0]["unstamped_merged"] == 1
    # a second card for the same channel gets a delta against the first
    card2 = SC.build_card(run, rosters=rosters, board=board, log=lambda *a: None)
    assert card2["previous"]["run"] == run.name
    d = {x["metric"]: x for x in card2["delta"]}
    assert d["Assigned rows"]["delta"] == 0 and d["merged rows unstamped"]["now"] == 1
    SC.write_outputs([card2], str(out), log=lambda *a: None)
    page = (out / "scoreboard.html").read_text()
    assert page.startswith("<title>Peaky Scoreboard</title>") and "TEST-BATCH|NO3|cover" in page and "data-theme" in page
    board_md = (out / "SCOREBOARD.md").read_text()
    assert "TEST-BATCH · NO3 · cover" in board_md and "2 rows in" in board_md   # the key's | would split the table


def test_cli_writes_the_board(run_dir, tmp_path, capsys):
    rc = SC.main([str(run_dir), "--out", str(tmp_path / "b"), "--quiet"])
    assert rc == 0 and (tmp_path / "b" / "scoreboard.jsonl").is_file()
    assert "Assigned 5 / Candidate 0" in capsys.readouterr().out


def test_a_decoy_arm_that_crashes_the_engine_is_recorded_not_fatal(run, monkeypatch):
    real = SC.run_engine_offline

    def boom(run_, peaks, sample_id, adducts, log=lambda *a: None, scoring=None):
        if sample_id.endswith("-adducts"):
            raise TypeError("boolean value of NA is ambiguous")
        return real(run_, peaks, sample_id, adducts, log, scoring=scoring)

    monkeypatch.setattr(SC, "run_engine_offline", boom)
    dc = SC.decoy(run, "adducts", 0.35, 1)
    assert dc["control"]["assigned"] >= 2 and dc["adducts"] == {"error": "TypeError: boolean value of NA is ambiguous"}
    card = SC.build_card(run, rosters=SC.load_rosters(), board=[], log=lambda *a: None)
    assert card["row"]["decoy_adducts_rate"] is None
    assert "engine error" not in SC.render_md(card)   # that card ran no decoy; the error text only appears with one


def test_m1_names_a_reagent_water_cluster_across_a_denser_ladder(run):
    """Br-.H2O sits 18 Da above Br- with BrO- in between: a mass window finds it."""
    ts = run.ts
    extra = []
    for i, sid in enumerate(SAMPLES):
        t = T0 + pd.Timedelta(hours=i)
        for mz, h, ion, role in ((78.9189, 5000.0, "Br-", "reagent"), (94.9139, 800.0, "BrO-", "reagent"),
                                 (96.9114, 780.0, "BrO-", "reagent"), (78.9189 + SC.WATER, 300.0, None, None)):
            extra.append(dict(sample_item_id=sid, datetime_utc=t, peak_id=f"{sid}-r{mz:.0f}", mz=mz, height=h, area=h,
                              neutral_formula=None, adduct=None, tier=None, ion_mz=mz if ion else np.nan, role=role,
                              ion_formula=ion, iso_label=None, stamp_source="observed" if ion else None,
                              dup_candidate=False, intensity_suspect=False))
    run.ts = pd.concat([ts, pd.DataFrame(extra)], ignore_index=True)
    ions, tracks = SC.ion_table(run), SC.unstamped_tracks(run)
    m1 = SC.missed_m1(run, ions, tracks, coverage_rows=[])
    row = next(r for r in m1["rows"] if abs(r["mz"] - (78.9189 + SC.WATER)) < 1e-3)
    assert row["family"] == "reagent + 1x H2O" and row["parent"].startswith("Br- @ 78.9189")


def test_decoy_ledger_counts_read_the_arm_levels_given_not_the_ledgers_own():
    """_ledger_counts rates a decoy arm by the levels it is handed (the arm levelled in its run's context), never
    by a per-file level the engine wrote on the arm's own ledger; strict minima in a field of their own."""
    base = dict(role="M0", tier="Assigned", mz=200.0, height=100.0, adduct="[M-H]-",
                ion_formula="C10H15O4", method="cheminfo", confidence="High")
    led = pd.DataFrame([dict(base, peak_id="a", neutral_formula="C10H16O4", evidence_level="3c"),
                        dict(base, peak_id="b", neutral_formula="C9H14O4", evidence_level="3c", tier="Candidate"),
                        dict(peak_id="r", role="reagent", tier=None, mz=62.0, height=1e5, adduct=None,
                             neutral_formula=None, ion_formula="NO3-", method=None, confidence=None, evidence_level=None)])
    lv = pd.DataFrame(dict(neutral=["C10H16O4", "C9H14O4"], adduct=["[M-H]-"] * 2, level=["4b", "5b"]))
    st = lv.assign(level=["5a", "5b"])
    c = SC._ledger_counts(led, "f", lv, st)
    assert c["m0"] == 2 and c["assigned"] == 1 and c["levels"]["4b"] == 1 and c["levels"]["5b"] == 1
    assert c["identified"] == 0 and c["by_claim"]["ion"]["pairs"] == 1 and c["strict"]["levels"]["5a"] == 1
    none = SC._ledger_counts(led, "f")
    assert sum(none["levels"].values()) == 0 and none["by_claim"]["tentative"]["pairs"] == 2   # no level: tentative


def test_the_offline_engine_run_carries_the_runs_own_width_model(run_dir, monkeypatch):
    """A decoy arm must be rated by the same separability rule as the run it
    bounds: the recorded width model rides into assign.run(peaks=)."""
    import json as _json
    from peaky.assignment import assign as A
    from peaky.chem import resolution as RES
    seen = {}

    def fake_run(sample_id, context="ambient-air", **kw):
        seen.update(kw)
        return {"ledger": pd.DataFrame({"role": [], "tier": []})}

    monkeypatch.setattr(A, "run", fake_run)
    peaks = pd.DataFrame({"peak_id": ["a"], "mz": [200.0], "height": [10.0]})
    r0 = SC.load_run(str(run_dir))
    SC.run_engine_offline(r0, peaks, "x-control", ["[M-H]-"])
    assert "resolving_power" not in seen
    summ = _json.loads((run_dir / "batch_summary.json").read_text())
    summ["resolution"] = RES.Resolution(coef=1.0 / 9500.0, exponent=1.0, n_peaks=9, source="measured").as_dict()
    (run_dir / "batch_summary.json").write_text(_json.dumps(summ))
    r1 = SC.load_run(str(run_dir))
    seen.clear()
    SC.run_engine_offline(r1, peaks, "x-control", ["[M-H]-"])
    assert isinstance(seen.get("resolving_power"), RES.Resolution)
    assert seen["resolving_power"].r_at(200.0) == pytest.approx(9500.0)
