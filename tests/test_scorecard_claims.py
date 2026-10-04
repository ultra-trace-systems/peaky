"""scripts/scorecard.py -- the claim on the evidence scale of peaky 0.10.0 beside the tier.

The fixture is test_scorecard's four-spectrum run with the scale's columns written
onto its merged ledger and three readings added to its per-file ledgers: the
pinic acid's nitrate cluster (a Candidate the scale identifies), an ion-only
[M]-. line, and a reading in one file that no merged row carries. Every number
the claim block reports can be counted on paper:

    merged row             tier       level  claim         per-file height (2 files)
    C10H16O3 [M-H]-        Assigned   3c     identified    2 x 1000
    C9H14O4  [M-H]-        Assigned   4a     neutral       2 x 500
    C9H14O4  [M+NO3]-      Candidate  3c     identified    2 x 200
    C8H12O4  [M-H]-        Assigned   5b     tentative     2 x 250
    C22H42O4 [M-H]-        Assigned   4b     ion           2 x 120
    C5H10O6  [M-H]-        Assigned   NA     not assessed  2 x 90
    C10H16O3 [M]-.         Candidate  4b     ion           2 x 60   (the ion-only bucket)
    (no merged row)        C7H10O5 [M-H]-, s1 only                  1 x 80

Run: pytest tests/test_scorecard_claims.py -q
"""

import gzip
import html
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import scorecard as SC  # noqa: E402
from peaky.assignment import evidence as EV  # noqa: E402
from peaky.chem import chemistry as C  # noqa: E402
from test_scorecard import ORBI, SAMPLES, A, B, Cc, D, E, H, write_run  # noqa: E402

COMMITTED = 2 * (1000 + 500 + 200 + 250 + 120 + 90 + 60) + 80
UNMATCHED = "C7H10O5"
LEVELS = {(A[0], "[M-H]-"): "3c", (B[0], "[M-H]-"): "4a", (B[0], "[M+NO3]-"): "3c", (Cc[0], "[M-H]-"): "5b",
          (E[0], "[M-H]-"): "4b", (D[0], "[M-H]-"): "NA", (A[0], "[M]-."): "4b"}

# the board-row keys a card carried before the claim; each keeps its name (axes_hist is gone: the scale has no axes)
OLD_ROW_KEYS = [
    "written_utc", "channel", "run", "run_dir", "code", "reagent", "path", "n_spectra", "n_peaks", "n_files",
    "stamped_peak_share", "stamped_signal_share", "merged_rows", "assigned", "candidate", "ion_only", "neutrals",
    "unstamped_merged", "bright_m0_not_assigned", "bright_unstamped", "levels", "good_levels",
    "m1_tracks", "m1_families", "m1_signal_share", "roster_n", "roster_present", "roster_assigned", "roster_misread",
    "roster_unstamped", "census", "census_halogen", "decoy_mode", "decoy_shift_rate", "decoy_adducts_rate",
    "decoy_control_assigned", "c13_n", "c13_within_1", "hetero_n", "hetero_present", "iso_cov_median_r",
    "adduct_cov_median_r", "m3_other_path_missing", "m3_other_instrument_missing", "m3_other_instrument_own_missing",
]
CLAIM_ROW_KEYS = [
    "claim_identified", "claim_neutral", "claim_ion", "claim_tentative", "claim_reagent", "claim_not_assessed",
    "claim_identified_signal", "claim_neutral_signal", "claim_ion_signal", "claim_tentative_signal",
    "claim_reagent_signal", "claim_not_assessed_signal", "claim_unmatched_signal", "claim_assigned_identified_signal",
    "claim_assigned_neutral_signal", "claim_assigned_ion_signal", "claim_assigned_tentative",
    "claim_candidate_identified", "claim_stamp_mismatch", "claim_level_source", "bright_m0_not_identified",
    "bright_m0_not_assessed", "roster_identified", "roster_neutral", "roster_misread_identified",
    "decoy_control_identified", "decoy_control_identified_lt_350", "decoy_control_established",
    "decoy_shift_identified", "decoy_shift_identified_lt_350", "decoy_shift_identified_rate",
    "decoy_shift_identified_lt_350_rate", "decoy_shift_established_rate", "decoy_shift_strict_identified_rate",
    "decoy_adducts_identified", "decoy_adducts_identified_rate", "decoy_adducts_identified_lt_350_rate",
    "decoy_adducts_established_rate", "m3_own_missing_identified", "m3_own_basis", "claims_schema",
]


def write_claim_run(root: Path, name: str = "TEST-BATCH_2026-01-01T000000Z", defect: bool = False,
                    resolution=None) -> Path:
    run = write_run(root, name, resolution=resolution)
    for sid in SAMPLES[:2]:
        pf = pd.read_csv(run / "per_file" / f"{sid}_ledger.csv")
        b = pf[pf["peak_id"] == f"{sid}-p1"].iloc[0].to_dict()
        extra = [
            dict(b, peak_id=f"{sid}-bn", adduct="[M+NO3]-", mz=C.ion_mz(B[0], "[M+NO3]-"), height=200.0,
                 ion_formula="C9H14NO7-", tier="Candidate"),
            dict(b, peak_id=f"{sid}-x", neutral_formula=A[0], adduct="[M]-.", mz=A[1] + H, height=60.0,
                 ion_formula="C10H16O3-", tier="Candidate", method="ion_only:electron_attachment"),
        ]
        if sid == "s1":
            extra.append(dict(b, peak_id=f"{sid}-u", neutral_formula=UNMATCHED, adduct="[M-H]-", height=80.0,
                              mz=C.ion_mz(UNMATCHED, "[M-H]-"), ion_formula="C7H9O5-"))
        pd.concat([pf, pd.DataFrame(extra)], ignore_index=True).to_csv(run / "per_file" / f"{sid}_ledger.csv", index=False)
    led = pd.read_csv(run / "merged_ledger.csv")
    led["ion_only_of"] = pd.NA
    extra = pd.DataFrame([
        dict(mz=C.ion_mz(B[0], "[M+NO3]-"), neutral_formula=B[0], adduct="[M+NO3]-", tier="Candidate", n_files=2, stage="cover",
             trace_role="single", tier_reason="", alternatives="", intensity_suspect=False, ion_only_of=pd.NA),
        dict(mz=A[1] + H, neutral_formula=A[0], adduct="[M]-.", tier="Candidate", n_files=2, stage="cover",
             trace_role="single", tier_reason="", alternatives="", intensity_suspect=False, ion_only_of="s1-p0"),
    ])
    led = pd.concat([led, extra], ignore_index=True)
    for c in EV.COLUMNS:
        led[c] = ""
    led["evidence_level"] = [LEVELS[k] for k in zip(led.neutral_formula, led.adduct)]
    led["would_lift"] = [f"{lv}: synthetic" for lv in led["evidence_level"]]
    led["claim"] = led["evidence_level"].map(EV.claim_class)
    if defect:
        led.loc[led["neutral_formula"] == E[0], "claim"] = "identified"      # one stamp defect
    led.to_csv(run / "merged_ledger.csv", index=False)
    return run


@pytest.fixture
def crun(tmp_path):
    return SC.load_run(str(write_claim_run(tmp_path / "out")))


def _claims(run):
    return SC.claims(run, SC.levels_for(run, None, []))


def test_claim_of_reads_blank_cells_as_no_level_and_good_levels_are_the_identified_ones():
    assert SC.claim_of("3c") == "identified" and SC.claim_of("4a") == "neutral" and SC.claim_of("4b") == "ion"
    assert SC.claim_of("NA") == "not assessed" and SC.claim_of("reagent") == "reagent"
    assert {SC.claim_of(v) for v in ("5a", "5b", "", "nan", "NaN", None, float("nan"), pd.NA)} == {"tentative"}
    assert SC.GOOD_LEVELS == {"3c"} and SC.ESTABLISHED_LEVELS == {"3c", "4a"} and SC.M3_LEVELS == {"3c", "4a", "4b"}
    assert SC.LEVELS == ["3c", "4a", "4b", "5a", "5b", "reagent", "NA"] and SC.CLAIMS_SCHEMA == 2
    assert SC.CLAIM_KEYS == ["identified", "neutral", "ion", "tentative", "reagent", "not assessed"]


def test_the_claim_block_counts_rows_crosstab_and_the_committed_signal(crun):
    cl = _claims(crun)
    assert cl["n_rows"] == 7 and cl["level_source"] == "in-core" and cl["scale"] == "peaky 0.10.0"
    assert cl["rows"] == {"identified": 2, "neutral": 1, "ion": 2, "tentative": 1, "reagent": 0, "not assessed": 1}
    # the ion-only row is its own bucket, out of the Candidate row
    zero = dict.fromkeys(SC.CLAIM_KEYS, 0)
    assert cl["by_tier"]["rows"] == {
        "Assigned": dict(zero, identified=1, neutral=1, ion=1, tentative=1, **{"not assessed": 1}),
        "Candidate": dict(zero, identified=1),
        "ion-only": dict(zero, ion=1),
    }
    sig = cl["signal"]
    assert sig["committed"] == COMMITTED and sig["n_m0"] == 15 and sig["n_unmatched"] == 1
    pct = lambda h: 100.0 * h / COMMITTED   # noqa: E731
    assert sig["share"] == pytest.approx({"identified": pct(2000 + 400), "neutral": pct(1000), "ion": pct(240 + 120),
                                          "tentative": pct(500), "reagent": 0.0, "not assessed": pct(180),
                                          "unmatched": pct(80)})
    assert sum(sig["share"].values()) == pytest.approx(100.0)
    assert sig["share_assigned"]["identified"] == pytest.approx(pct(2000))
    assert cl["by_tier"]["signal"]["Candidate"]["identified"] == pytest.approx(pct(400))
    assert cl["by_tier"]["signal"]["ion-only"]["ion"] == pytest.approx(pct(120))
    assert cl["stamp_mismatch"] == 0 and "corroboration" not in cl         # the stamp agrees; no axis to split by


def test_disagreeing_rows_are_listed_by_signal_never_fixed(crun):
    dis = _claims(crun)["disagree"]
    assert dis["assigned_tentative"] == 1 and dis["candidate_identified"] == 1
    assert [(r["kind"], r["neutral"], r["adduct"]) for r in dis["rows"]] == [
        ("Assigned but tentative", Cc[0], "[M-H]-"), ("Candidate but identified", B[0], "[M+NO3]-")]
    top = dis["rows"][0]
    assert top["tier"] == "Assigned" and top["level"] == "5b" and top["claim"] == "tentative" and top["n_files"] == 2
    assert top["signal_share"] == pytest.approx(100.0 * 500 / COMMITTED) and top["would_lift"] == "5b: synthetic"
    # the ledger is read, never written: the tier stays what the run wrote
    assert list(crun.ledger["tier"]).count("Assigned") == 5


def test_a_stored_claim_that_disagrees_with_its_level_is_counted(tmp_path):
    run = SC.load_run(str(write_claim_run(tmp_path / "s", defect=True)))
    cl = _claims(run)
    assert cl["stamp_mismatch"] == 1 and cl["stamp_mismatch_by_ledger"] == {"merged": 1, "per_file": None}
    assert cl["rows"]["identified"] == 2                       # derived from the level, not the stamp


def test_a_run_made_before_the_scale_is_levelled_post_hoc_and_its_old_claims_are_not_read(tmp_path):
    rd = write_run(tmp_path / "old")
    led = pd.read_csv(rd / "merged_ledger.csv")
    led["evidence_level"], led["evidence_axes"], led["claim"] = "4a", "iso|chan2", "identified"   # pre-0.10.0
    led.to_csv(rd / "merged_ledger.csv", index=False)
    run = SC.load_run(str(rd))
    cl = _claims(run)
    assert cl["level_source"] == "post-hoc" and cl["rows"]["not assessed"] == 5 and cl["rows"]["identified"] == 0
    assert cl["stamp_mismatch"] is None                        # an old claim stamp is not compared
    assert cl["signal"]["share"]["not assessed"] == pytest.approx(100.0) and cl["signal"]["n_unmatched"] == 0


def test_brightest_and_best_evidence_carry_the_claim(crun):
    ions, tracks, lv = SC.ion_table(crun), SC.unstamped_tracks(crun), SC.levels_for(crun, None, [])
    b = SC.brightest(crun, ions, tracks, lv, n=5)
    by = {r["neutral"]: r["claim"] for r in b["rows"] if r["role"] == "M0"}
    assert by == {A[0]: "identified", B[0]: "neutral", Cc[0]: "tentative", E[0]: "ion"}
    assert [r["claim"] for r in b["rows"] if r["role"] == "iso_child"] == [""]
    assert b["m0_not_assigned"] == 0 and b["m0_not_identified"] == 3 and b["m0_not_assessed"] == 0
    e = SC.best_evidence(crun, ions, lv)
    assert e["by_claim"] == {"identified": 2, "neutral": 1, "ion": 2, "tentative": 1, "reagent": 0, "not assessed": 1}
    assert e["rows"][0]["claim"] == "identified" and e["n_good"] == 2 and e["n_established"] == 3


def test_m2_joins_the_claim_of_the_reading_on_each_roster_line(crun, tmp_path):
    d = tmp_path / "rosters"
    d.mkdir()
    pd.DataFrame({"name": ["pinonic acid", "pinic acid", "terpenylic acid", "HOM", "hydroxy acid"],
                  "formula": [A[0], B[0], Cc[0], "C10H16O10", D[0]], "class": ["monomer"] * 5,
                  "reference": ["x"] * 5, "note": [""] * 5}).to_csv(d / "ap.csv", index=False)
    rosters = SC.load_rosters(str(d))
    ions, tracks, lv = SC.ion_table(crun), SC.unstamped_tracks(crun), SC.levels_for(crun, None, [])
    before = SC.missed_m2(crun, ions, tracks, rosters)
    m2 = SC.missed_m2(crun, ions, tracks, rosters, lv)
    assert m2["roster"] == before["roster"] and before["roster_claim"] == {}      # _recall untouched
    assert m2["roster_claim"]["ap"] == {"identified": 1, "neutral": 1, "ion": 0, "tentative": 1, "reagent": 0,
                                        "not assessed": 0, "misread_identified": 0}
    claim = {r["neutral"]: r["claim"] for r in m2["rows"] if r["source"] == "roster:ap"}
    assert claim == {A[0]: "identified", B[0]: "neutral", Cc[0]: "tentative", "C10H16O10": "", D[0]: ""}
    # the HOM's line read as another neutral the run identifies: an identified misread
    hom = C.ion_mz("C10H16O10", "[M-H]-")
    fake = pd.concat([ions, pd.DataFrame([dict(ion_mz=hom, n=4, med_h=50.0, sum_h=200.0, role="M0", ion_formula="C11H19O9-",
                                               iso_label=None, neutral="C11H20O9", adduct="[M-H]-", tier="Candidate",
                                               suspect=False, stamp_source="", presence=1.0)])], ignore_index=True)
    fake = fake.sort_values("ion_mz").reset_index(drop=True)
    lv2 = pd.concat([lv, pd.DataFrame([dict(neutral="C11H20O9", adduct="[M-H]-", level="3c")])])
    m2b = SC.missed_m2(crun, fake, tracks, rosters, lv2)
    assert m2b["roster"]["ap"]["read_as"] == 1 and m2b["roster_claim"]["ap"]["misread_identified"] == 1


def test_m3_counts_the_other_instruments_rows_by_claim_before_the_cut(crun, tmp_path):
    import dataclasses
    other = SC.load_run(str(write_claim_run(tmp_path / "other")))
    lacks = dataclasses.replace(crun, ledger=crun.ledger[~crun.ledger.neutral_formula.isin([A[0], B[0], E[0]])])
    own = SC.own_levels_for(other)                          # its files pooled: a class-less source reads NA
    m3 = SC.missed_m3(lacks, None, other, SC.levels_for(other, None, []), None, [], 10.0, 0.8, own_levels=own)
    oi, oo = m3["other_instrument"], m3["other_instrument_own"]
    # in-core: A 3c, B 4a, B-NO3 3c, E 4b, A-ion-only 4b, and D (NA, Assigned) by the fallback; the series stamps
    # A, B, E on [M-H]- (D never): those three are the misses
    assert oi["basis"] == SC.M3_BASIS_LEVEL + "; Assigned where not assessed"
    assert oi["n_good_by_claim"] == {"identified": 2, "neutral": 1, "ion": 2, "not assessed": 1}
    assert oi["n_missing"] == 3 and oi["n_missing_by_claim"] == {"identified": 1, "neutral": 1, "ion": 1, "not assessed": 0}
    assert oo["basis"] == SC.M3_BASIS_ASSIGNED and oo["n_good"] == 5
    assert oo["n_missing"] == 3 and oo["n_missing_by_claim"]["not assessed"] == 3


# --------------------------------------------------------------------------- decoys
def _decoy_ledger(rows) -> pd.DataFrame:
    base = dict(role="M0", tier="Assigned", adduct="[M-H]-", method="cheminfo", confidence="High", ion_only_of=pd.NA)
    return pd.DataFrame([dict(base, **r) for r in rows])


# `evidence_level` here is the level the arm's pair is GIVEN (by `_fake_level_arm`), not a ledger column the card reads
LEDGER = [
    # an identified pair seen twice: its m/z is the brightest row's (above 350), its tier that row's
    dict(peak_id="a1", neutral_formula="C10H16O4", mz=199.0, height=10.0, evidence_level="3c", tier="Candidate"),
    dict(peak_id="a2", neutral_formula="C10H16O4", mz=399.0, height=90.0, evidence_level="3c"),
    dict(peak_id="b", neutral_formula="C9H14O4", mz=185.0, height=50.0, evidence_level="3c"),
    dict(peak_id="c", neutral_formula="C8H12O4", mz=171.0, height=40.0, evidence_level="4a", tier="Candidate"),
    dict(peak_id="d", neutral_formula="C7H10O4", mz=157.0, height=30.0, evidence_level="4b"),
    dict(peak_id="e", neutral_formula="C6H8O4", mz=143.0, height=20.0, evidence_level="5b", tier="Candidate"),
    dict(peak_id="f", neutral_formula="C5H6O4", mz=500.0, height=20.0, evidence_level=None),
    # the ion-only bucket is no tier count: its pair is ion, not Assigned
    dict(peak_id="g", neutral_formula="C10H16O4", adduct="[M]-.", mz=200.0, height=15.0, evidence_level="4b",
         method="ion_only:electron_attachment", ion_only_of="a2"),
    dict(peak_id="r", role="reagent", neutral_formula=None, adduct=None, mz=62.0, height=1e5, tier=None, evidence_level=None),
]


def _given_levels(led: pd.DataFrame, strict: bool = False) -> pd.DataFrame:
    m0 = led[led["role"] == "M0"].drop_duplicates(["neutral_formula", "adduct"])
    lv = m0["evidence_level"].astype(object).where(m0["evidence_level"].notna(), None)
    if strict:                    # the run's minima: no 3c on one file here
        lv = lv.map(lambda v: "4b" if v == "3c" else v)
    return pd.DataFrame({"neutral_formula": m0["neutral_formula"].values, "adduct": m0["adduct"].values,
                         "evidence_level": lv.values})


def _fake_level_arm(main_src, led, file_id, arm, mode, wrong=(), control=None):
    return _given_levels(led, strict=mode == "strict")


def test_ledger_counts_split_the_decoy_pairs_by_claim():
    led = _decoy_ledger(LEDGER)
    c = SC._ledger_counts(led, "f", SC._levels_frame(_given_levels(led)), SC._levels_frame(_given_levels(led, True)))
    bc = c["by_claim"]
    assert bc["identified"] == {"pairs": 2, "lt_350": 1, "ge_350": 1, "assigned": 2, "assigned_lt_350": 1, "assigned_ge_350": 1}
    assert bc["neutral"] == {"pairs": 1, "lt_350": 1, "ge_350": 0, "assigned": 0, "assigned_lt_350": 0, "assigned_ge_350": 0}
    assert bc["ion"] == {"pairs": 2, "lt_350": 2, "ge_350": 0, "assigned": 1, "assigned_lt_350": 1, "assigned_ge_350": 0}
    assert bc["tentative"] == {"pairs": 2, "lt_350": 1, "ge_350": 1, "assigned": 1, "assigned_lt_350": 0, "assigned_ge_350": 1}
    assert (c["identified"], c["identified_lt_350"], c["identified_ge_350"]) == (2, 1, 1)
    assert (c["established"], c["established_lt_350"]) == (3, 2)
    # the level vector's unit: identified pairs are its 3c
    assert c["identified"] == sum(c["levels"][lv] for lv in SC.GOOD_LEVELS)
    assert c["m0"] == 8 and c["assigned"] == 4                       # the old row counts are untouched
    assert c["strict"]["identified"] == 0 and c["strict"]["levels"]["4b"] == 4
    assert set(c["by_claim"]) == set(SC.CLAIM_KEYS) and set(c["by_claim"]["ion"]) == set(SC.CLAIM_COUNTS)


def _arms(monkeypatch, shift_rows, adduct_rows=None, fail=None):
    def fake(run_, peaks, sample_id, adducts, log=lambda *a: None, scoring=None):
        arm = sample_id.rsplit("-", 1)[1]
        if arm == fail:
            raise TypeError("boolean value of NA is ambiguous")
        return _decoy_ledger({"control": LEDGER, "shift": shift_rows, "adducts": adduct_rows or []}[arm])
    monkeypatch.setattr(SC, "run_engine_offline", fake)
    monkeypatch.setattr(SC, "level_arm", _fake_level_arm)


def test_decoy_sums_the_claims_and_rates_them_against_the_control(crun, monkeypatch):
    _arms(monkeypatch, LEDGER[2:4] + LEDGER[5:6], LEDGER[4:5])
    dc = SC.decoy(crun, "both", 0.35, 1)
    ctrl, sh, ad = dc["control"], dc["shift"], dc["adducts"]
    assert ctrl["identified"] == 2 and sh["identified"] == 1 and sh["identified_lt_350"] == 1 and ad["identified"] == 0
    assert sh["identified_rate"] == pytest.approx(50.0) == pytest.approx(sh["good_level_rate"])
    assert sh["identified_lt_350_rate"] == pytest.approx(100.0) and sh["identified_ge_350_rate"] == 0.0
    assert sh["identified_assigned_lt_350_rate"] == pytest.approx(100.0)    # b Assigned / control's one Assigned below 350
    assert sh["established_rate"] == pytest.approx(100 * 2 / 3) and sh["neutral_rate"] == pytest.approx(100.0)
    assert sh["ion_rate"] == 0.0 and sh["tentative_rate"] == pytest.approx(50.0) and sh["not_assessed_rate"] is None
    assert ad["ion_rate"] == pytest.approx(50.0) and ad["identified_rate"] == 0.0
    assert ctrl["strict"]["identified"] == 0 and sh["strict"]["identified_rate"] is None    # strict, its own field
    for key in ("assigned_rate", "candidate_rate", "good_level_rate"):              # every old rate kept
        assert key in sh and key in ad
    assert dc["ledgers"] == {"source": "engine", "dir": None, "code": SC.engine_code()}


def test_a_claim_rate_against_a_control_with_none_of_that_claim_is_undefined(crun, monkeypatch):
    # the control holds one ion pair (4b); the shift arm one identified pair (3c)
    arms = {"control": [LEDGER[4]], "shift": [LEDGER[2]], "adducts": [LEDGER[4]]}
    monkeypatch.setattr(SC, "run_engine_offline", lambda run_, peaks, sample_id, adducts, log=None, scoring=None:
                        _decoy_ledger(arms[sample_id.rsplit("-", 1)[1]]))
    monkeypatch.setattr(SC, "level_arm", _fake_level_arm)
    card = SC.build_card(crun, rosters=SC.load_rosters(), board=[], log=lambda *a: None, decoy_mode="both")
    dc = card["decoy"]
    sh, ad = dc["shift"], dc["adducts"]
    assert dc["control"]["identified"] == 0 and sh["identified"] == 1
    for key in ("identified_rate", "identified_lt_350_rate", "identified_ge_350_rate", "identified_assigned_lt_350_rate",
                "tentative_rate", "established_rate"):
        assert sh[key] is None and ad[key] is None                     # undefined, not 0 %
    assert sh["ion_rate"] == 0.0 and ad["ion_rate"] == pytest.approx(100.0)
    assert sh["assigned_rate"] == pytest.approx(100.0) and sh["good_level_rate"] == 0.0   # the old rates as they were
    assert card["row"]["decoy_shift_identified_rate"] is None
    assert {a["key"]: a["value"] for a in card["acceptance"]}["decoy_shift_identified_rate"] is None
    line = next(ln for ln in SC.render_md(card).splitlines() if ln.startswith("| shift +0.35 Da |"))
    assert line.split(" | ")[11:13] == ["—", "—"]                   # identified rate, identified < 350 rate


def test_an_errored_arm_stays_exactly_its_error(crun, monkeypatch, tmp_path):
    _arms(monkeypatch, LEDGER[2:4], fail="adducts")
    save = tmp_path / "kept"
    dc = SC.decoy(crun, "both", 0.35, 1, save_dir=str(save))
    assert dc["adducts"] == {"error": "TypeError: boolean value of NA is ambiguous"}
    assert sorted(p.name for p in save.iterdir()) == ["manifest.json", "s1__control.csv.gz", "s1__shift.csv.gz"]   # nothing kept for it
    card = {"decoy": dc, "claims": {"rows": dict.fromkeys(SC.CLAIM_KEYS, 0)}}
    rows = SC.claim_table(card)
    assert rows[0]["class"] == "identified" and rows[0]["decoy"] == "2 / 1 / error"
    assert rows[0]["adducts_split"] == "error / error"


def test_the_decoy_arms_are_levelled_in_the_runs_context_exactly_as_the_reference_levels_a_kept_arm(tmp_path):
    """On an Orbitrap-class run the arms the engine makes in memory are levelled as
    `evidence.source_from_run_dir(<file>__<arm>.csv.gz, main=<run dir>, mode=...)` levels the kept file, adapted and
    strict: the run's window, gate and lists, the wrong adducts on the adducts arm, the control's calibration."""
    from peaky.assignment.levels import source as SRC
    rd = write_claim_run(tmp_path / "orbi", resolution=ORBI)
    run = SC.load_run(str(rd))
    kept = tmp_path / "kept"
    dc = SC.decoy(run, "both", 0.35, 1, save_dir=str(kept))
    fid = dc["files"][0]
    ctrl = dc["control"]
    kept_m0 = pd.read_csv(kept / f"{fid}__control.csv.gz", low_memory=False).query("role == 'M0'")
    n_pairs = len(kept_m0.drop_duplicates(["neutral_formula", "adduct"]))
    assert "error" not in ctrl and "level_error" not in ctrl and sum(ctrl["levels"].values()) == n_pairs > 0
    for mode, got in (("adapted", ctrl["levels"]), ("strict", ctrl["strict"]["levels"])):
        ref = EV.level_source(SRC.source_from_run_dir(str(kept / f"{fid}__control.csv.gz"), main=str(rd), mode=mode))
        assert got == SC.level_vector(SC._levels_frame(ref)), mode
    assert ctrl["levels"]["NA"] == 0 and ctrl["levels"] != ctrl["strict"]["levels"]
    # the wrong-adducts arm commits nothing on this file: no pair to level, its tier counts kept
    ad = dc["adducts"]
    assert ad["m0"] == 0 and sum(ad["levels"].values()) == 0 and "level_error" not in ad


def test_kept_arm_ledgers_recount_to_the_same_card(run_dir_claims, tmp_path, capsys):
    out = tmp_path / "board"
    assert SC.main([str(run_dir_claims), "--out", str(out), "--decoy", "both", "--quiet"]) == 0
    name = run_dir_claims.name
    kept = out / name / "decoy"
    assert sorted(p.name for p in kept.iterdir()) == ["manifest.json", "s1__adducts.csv.gz", "s1__control.csv.gz", "s1__shift.csv.gz"]
    with gzip.open(kept / "s1__control.csv.gz", "rt") as fh:
        assert "evidence_level" in fh.readline()
    manifest = json.loads((kept / "manifest.json").read_text())
    assert manifest == {"mode": "both", "offset_da": 0.35, "files": ["s1"], "adducts_used": SC.load_run(str(run_dir_claims)).adducts,
                        "wrong_adducts": SC.wrong_adducts("-"), "scoring": {"s1": "class-fallback"},   # no pattern_scoring: pre-0.9.0
                        "scoring_detail": {}, "code": SC.engine_code()}
    first = json.loads((out / name / "scorecard.json").read_text())["decoy"]
    assert first["control"]["assigned"] >= 2 and "by_claim" in first["control"]
    # a re-count from the kept ledgers: no engine run, the same numbers
    monkey = SC.run_engine_offline
    SC.run_engine_offline = lambda *a, **k: (_ for _ in ()).throw(AssertionError("the kept ledgers must be read"))
    try:
        assert SC.main([str(run_dir_claims), "--out", str(tmp_path / "again"), "--decoy-ledgers", str(out), "--quiet"]) == 0
    finally:
        SC.run_engine_offline = monkey
    again = json.loads((tmp_path / "again" / name / "scorecard.json").read_text())["decoy"]
    assert again["mode"] == "both" and again["ledgers"] == {"source": "saved", "dir": str(kept), "code": manifest["code"]}
    assert again["scoring"] == manifest["scoring"]                     # what the kept ledgers were judged at, as kept
    for arm in ("control", "shift", "adducts"):
        assert again[arm] == first[arm]
    assert not (tmp_path / "again" / name / "decoy").exists()          # a re-count keeps nothing new
    assert "identified " in capsys.readouterr().out


def test_a_recount_reports_what_the_kept_ledgers_were_made_with(run_dir_claims, tmp_path, monkeypatch):
    _arms(monkeypatch, LEDGER[2:4], LEDGER[4:5])
    out = tmp_path / "board"
    assert SC.main([str(run_dir_claims), "--out", str(out), "--decoy", "both", "--quiet"]) == 0
    name = run_dir_claims.name
    first = json.loads((out / name / "scorecard.json").read_text())["decoy"]
    # another offset and file count on the command line: the ledgers were made at +0.35 on one file
    assert SC.main([str(run_dir_claims), "--out", str(tmp_path / "again"), "--decoy-ledgers", str(out),
                    "--decoy-offset", "0.5", "--decoy-files", "3", "--quiet"]) == 0
    again = json.loads((tmp_path / "again" / name / "scorecard.json").read_text())["decoy"]
    for key in ("offset_da", "files", "adducts_used", "wrong_adducts", "shift"):
        assert again[key] == first[key]
    # an explicit --decoy none stays none
    assert SC.main([str(run_dir_claims), "--out", str(tmp_path / "none"), "--decoy", "none", "--decoy-ledgers", str(out), "--quiet"]) == 0
    assert json.loads((tmp_path / "none" / name / "scorecard.json").read_text())["decoy"]["control"] is None


def test_a_recount_from_a_dir_with_no_kept_ledger_stops_before_the_board(run_dir_claims, tmp_path):
    with pytest.raises(SystemExit, match="no kept decoy ledger"):
        SC.main([str(run_dir_claims), "--out", str(tmp_path / "board"), "--decoy-ledgers", str(tmp_path / "typo"), "--quiet"])
    assert not (tmp_path / "board").exists()


@pytest.fixture
def run_dir_claims(tmp_path):
    return write_claim_run(tmp_path / "out")


def test_the_card_prints_what_the_arms_were_judged_at_and_a_recount_keeps_it(run_dir_claims, tmp_path, monkeypatch):
    """C35: a file whose run recorded a snapshot has its arms scored at it; the card prints that per file with the
    inherited numbers and everything an arm does not inherit (md and html, the note escaped, whatever the control
    arm did), the board row carries one word, and a re-count from the kept ledgers gives back the same detail."""
    _arms(monkeypatch, LEDGER[2:4], LEDGER[4:5])
    bs = run_dir_claims / "batch_summary.json"
    summ = json.loads(bs.read_text())
    summ["pattern_scoring"] = {"s1": {"sigma_ppm": 0.71, "mu_ppm": -1.25, "mz_tolerance_ppm": 5.0, "abundance_floor": 0.02,
                                      "sigma_source": "fitted", "mu_source": "fitted", "fitted_anchors": 23}}
    bs.write_text(json.dumps(summ))
    out = tmp_path / "board"
    assert SC.main([str(run_dir_claims), "--out", str(out), "--decoy", "both", "--quiet"]) == 0
    name = run_dir_claims.name
    card = json.loads((out / name / "scorecard.json").read_text())
    assert card["decoy"]["scoring"] == {"s1": "inherited"} and card["row"]["decoy_scoring"] == "inherited"
    detail = {"sigma_ppm": 0.71, "mu_ppm": -1.25, "mz_tolerance_ppm": 5.0, "abundance_floor": 0.02, "fitted_anchors": 23,
              "sigma_source": "fitted"}
    assert card["decoy"]["scoring_detail"] == {"s1": detail}
    md = (out / name / "SCORECARD.md").read_text()
    assert "arms judged at: `s1` inherited (sigma 0.71 ppm, mu -1.25 ppm, window 5 ppm, floor 0.02, 23 anchors)" in md
    page = html.unescape(SC.render_html([card], [card["row"]]))          # the page writes non-ASCII as entities
    for text in (md, page):
        for item in ("opportunistic channels", "height cutoff", "prior offset", "occurrence table", "time series",
                     "active reference lists", "or corroboration (card C39)", "abundance floor"):
            assert item in text, item
    # the note stands whatever the control arm did, in md as in html
    broken = dict(card, decoy=dict(card["decoy"], control={"error": "TypeError: boom"}))
    assert "arms judged at" in SC.render_md(broken) and "arms judged at" in SC.render_html([broken], [card["row"]])
    odd = dict(card, decoy=dict(card["decoy"], scoring={"<img src=x>": "class-fallback"}))
    page = SC.render_html([odd], [card["row"]])
    assert "&lt;img src=x&gt;" in page and "<img src=x>" not in page
    # the kept manifest carries the scoring and its numbers; a re-count reads them back, not the run's summary
    kept = out / name / "decoy"
    manifest = json.loads((kept / "manifest.json").read_text())
    assert manifest["scoring"] == {"s1": "inherited"} and manifest["scoring_detail"] == {"s1": detail}
    summ["pattern_scoring"] = {"s1": dict(summ["pattern_scoring"]["s1"], sigma_ppm="garbled")}
    bs.write_text(json.dumps(summ))
    assert SC.main([str(run_dir_claims), "--out", str(tmp_path / "again"), "--decoy-ledgers", str(out), "--quiet"]) == 0
    again = json.loads((tmp_path / "again" / name / "scorecard.json").read_text())["decoy"]
    assert again["scoring"] == {"s1": "inherited"} and again["scoring_detail"] == {"s1": detail}


def test_the_board_row_says_class_fallback_and_mixed(run_dir_claims, tmp_path, monkeypatch):
    _arms(monkeypatch, LEDGER[2:4], LEDGER[4:5])
    name = run_dir_claims.name
    assert SC.main([str(run_dir_claims), "--out", str(tmp_path / "a"), "--decoy", "shift", "--quiet"]) == 0
    assert json.loads((tmp_path / "a" / name / "scorecard.json").read_text())["row"]["decoy_scoring"] == "class-fallback"
    files = SC.brightest_files(SC.load_run(str(run_dir_claims)), 2)
    assert len(files) == 2
    bs = run_dir_claims / "batch_summary.json"
    summ = json.loads(bs.read_text())
    summ["pattern_scoring"] = {files[1]: {"sigma_ppm": 3.2, "mu_ppm": -0.65, "mz_tolerance_ppm": 15.0,
                                          "abundance_floor": 0.02, "sigma_source": "fitted", "fitted_anchors": 29}}
    bs.write_text(json.dumps(summ))                                  # the second file inherits, the brightest not
    assert SC.main([str(run_dir_claims), "--out", str(tmp_path / "b"), "--decoy", "shift", "--decoy-files", "2",
                    "--quiet"]) == 0
    card = json.loads((tmp_path / "b" / name / "scorecard.json").read_text())
    assert card["decoy"]["scoring"] == {files[0]: "class-fallback", files[1]: "inherited"}
    assert card["row"]["decoy_scoring"] == "mixed"


# --------------------------------------------------------------------------- the card, the board, the page
def test_the_card_carries_claims_after_the_headline_and_the_acceptance_block(crun):
    card = SC.build_card(crun, rosters=SC.load_rosters(), board=[], log=lambda *a: None)
    keys = list(card)
    assert keys.index("claims") == keys.index("headline") + 1
    row = card["row"]
    assert list(row)[: len(OLD_ROW_KEYS)] == OLD_ROW_KEYS and "axes_hist" not in row
    assert list(row)[len(OLD_ROW_KEYS):] == CLAIM_ROW_KEYS + ["decoy_scoring"]
    assert (row["claim_identified"], row["claim_neutral"], row["claim_ion"], row["claim_tentative"],
            row["claim_reagent"], row["claim_not_assessed"]) == (2, 1, 2, 1, 0, 1)
    assert row["claim_unmatched_signal"] == pytest.approx(100.0 * 80 / COMMITTED)
    assert row["claim_assigned_tentative"] == 1 and row["claim_candidate_identified"] == 1
    assert row["claims_schema"] == 2 and row["claim_stamp_mismatch"] == 0 and row["claim_level_source"] == "in-core"
    assert row["levels"] == {"3c": 2, "4a": 1, "4b": 2, "5a": 0, "5b": 1, "reagent": 0, "NA": 1} and row["good_levels"] == 2
    assert row["bright_m0_not_identified"] == 3 and row["bright_m0_not_assigned"] == 0
    assert row["decoy_shift_identified_rate"] is None                  # no decoy ran
    acc = {a["key"]: a for a in card["acceptance"]}
    assert list(acc) == ["roster_identified", "roster_misread_identified", "decoy_shift_identified_rate",
                         "decoy_shift_identified_lt_350_rate", "decoy_shift_established_rate",
                         "decoy_adducts_identified_rate", "bright_m0_not_identified", "m1_families",
                         "m3_other_instrument_own_missing"]
    assert acc["bright_m0_not_identified"]["value"] == 3 and acc["bright_m0_not_identified"]["old_key"] == "bright_m0_not_assigned"
    assert acc["m1_families"]["old_value"] == acc["m1_families"]["value"]


def test_key_metrics_lead_with_the_claim_and_no_delta_crosses_a_scale():
    keys = [k for k, _, _ in SC.KEY_METRICS]
    assert keys[0] == "claim_identified" and keys[-1] == "m3_other_instrument_own_missing"
    assert ("ion_only", "ion-only rows", 0) in SC.KEY_METRICS and ("assigned", "Assigned rows", 0) in SC.KEY_METRICS
    labels = [lab for _, lab, _ in SC.KEY_METRICS]
    assert len(labels) == len(set(labels))
    assert SC.GOOD_DIRECTION["decoy_shift_identified_rate"] == -1 and SC.GOOD_DIRECTION["claim_identified"] == 1
    assert SC._delta_chip("m1_families", -2) == '<small class="d-flat">-2 vs previous</small>'
    # a board row from before the claim: the delta is None, not a crash
    d = {x["metric"]: x for x in SC.delta({"claim_identified": 4, "assigned": 5}, {"assigned": 5})}
    assert d["identified rows"]["prev"] is None and d["identified rows"]["delta"] is None and d["Assigned rows"]["delta"] == 0
    # a row on the pre-0.10.0 scale: its claim and level metrics are not diffed against the scale's; the tier is
    d = {x["metric"]: x for x in SC.delta({"claim_identified": 2, "good_levels": 2, "assigned": 5, "claims_schema": 2},
                                          {"claim_identified": 9, "good_levels": 9, "assigned": 4, "claims_schema": 1})}
    assert d["identified rows"]["prev"] is None and d["rows at level 3c (identified)"]["delta"] is None
    assert d["Assigned rows"]["delta"] == 1
    d = {x["metric"]: x for x in SC.delta({"claim_identified": 2, "claims_schema": 2},
                                          {"claim_identified": 3, "claims_schema": 2})}
    assert d["identified rows"]["delta"] == -1


def test_scorecard_md_opens_with_section_0(crun):
    card = SC.build_card(crun, rosters=SC.load_rosters(), board=[], log=lambda *a: None)
    md = SC.render_md(card)
    assert md.index("## 0. The claim — identified / neutral / ion / tentative") < md.index("## 1. What was assigned")
    assert "acceptance reads the identified class (evidence level 3c); tier unchanged and kept" in md
    s0 = md[: md.index("## 1. What was assigned")]
    assert "the evidence scale of peaky 0.10.0" in s0 and "Levels: in-core." in s0
    assert "| identified | 3c | 2 | 1 / 1 / 0 |" in s0 and "| not assessed | NA | 1 | 1 / 0 / 0 |" in s0
    assert "| unmatched |" in s0 and "| ion-only | 0 | 0 | 1 | 0 | 0 | 0 | 1 |" in s0
    assert "Assigned but tentative **1**, Candidate but identified **1**" in s0 and "5b: synthetic" in s0
    assert "Corroboration" not in s0
    assert "`roster_identified`" in s0 and "(`bright_m0_not_assigned` 0)" in s0
    assert "| level | claim | ion-only |" in md and "| level | claim | what would lift it |" in md
    assert "claims (identified / neutral / ion / tentative / reagent / not assessed): 2 / 1 / 2 / 1 / 0 / 1" in md
    assert "axes" not in md


def test_the_board_leads_with_claims_and_old_rows_show_dashes(crun, tmp_path):
    out = tmp_path / "board"
    old = SC.build_card(crun, rosters=SC.load_rosters(), board=[], log=lambda *a: None)
    # a card written before the claim: no block, no keys, no acceptance
    for k in ("claims", "acceptance"):
        old.pop(k)
    old["row"] = {k: v for k, v in old["row"].items() if k in OLD_ROW_KEYS}
    SC.write_outputs([old], str(out), log=lambda *a: None)
    board_md = (out / "SCOREBOARD.md").read_text()
    assert board_md.index("## Claims — acceptance reads the identified class") < board_md.index("## All metrics")
    claims_table = board_md[board_md.index("## Claims"): board_md.index("## All metrics")]
    assert "TEST-BATCH · NO3 · cover" in claims_table and "| — | — |" in claims_table
    page = (out / "scoreboard.html").read_text()
    assert "card predates C13" in page and page.index("<h2>Claims</h2>") < page.index("<h2>Channels</h2>")
    # the next card of the channel: its claims, its delta against a row that has none
    board = SC.read_board(str(out / "scoreboard.jsonl"))
    card = SC.build_card(crun, rosters=SC.load_rosters(), board=board, log=lambda *a: None)
    d = {x["metric"]: x for x in card["delta"]}
    assert d["identified rows"]["now"] == 2 and d["identified rows"]["prev"] is None
    SC.write_outputs([card], str(out), log=lambda *a: None)
    board_md = (out / "SCOREBOARD.md").read_text()
    claims_table = board_md[board_md.index("## Claims"): board_md.index("## All metrics")]
    assert "| 0.10.0 | 2 | 1 | 2 | 1 | 1 |" in claims_table and "2 rows in" in board_md
    assert "| 2/1/2/0/1/0/1 |" in board_md
    md = (out / crun.name / "SCORECARD.md").read_text()
    assert "| identified rows | — | 2 | — |" in md
    s6 = md[md.index("## 6. Delta"):].splitlines()
    assert s6[4] == "| metric | previous | now | delta |" and s6[5] == "|---|---:|---:|---:|"   # a dash does not left-align
    page = html.unescape((out / "scoreboard.html").read_text())         # the page is ASCII with entities
    assert "The claim — identified / neutral / ion / tentative" in page and "Acceptance" in page
    assert "card predates C13" not in page


def _schema_1_card(card: dict) -> dict:
    """A card as the pre-0.10.0 scorecard wrote it: three claims, a corroboration split, the old level vector."""
    old = json.loads(json.dumps(card, default=SC._json_default))
    old["claims"] = {"rows": {"identified": 4, "ion": 2, "tentative": 1},
                     "levels": {"identified": "1-4a", "ion": "4b-4d", "tentative": "5a, 5b, none"},
                     "n_rows": 7, "by_tier": {"rows": {"Assigned": {"identified": 3, "ion": 1, "tentative": 1}},
                                              "signal": {}}, "signal": {"share": {"identified": 70.0}},
                     "disagree": {"assigned_tentative": 1, "candidate_identified": 1,
                                  "rows": [{"kind": "Assigned but tentative", "level": "5b", "level_reason": "5b: old"}]},
                     "corroboration": {"identified": 4, "own_identified": 2, "via_cross": 2, "sources": ["X"]}}
    row = {k: v for k, v in old["row"].items() if k in OLD_ROW_KEYS}
    row.update(levels={"2b": 0, "3a": 1, "3b": 2, "4a": 1, "4b": 1, "4c": 1, "4d": 1, "5a": 0, "5b": 1},
               good_levels=4, claim_identified=4, claim_ion=2, claim_tentative=1, claims_schema=1)
    old["row"] = row
    return old


def test_a_schema_1_card_on_disk_renders_beside_a_new_one_and_is_never_diffed(crun, tmp_path):
    out = tmp_path / "board"
    new = SC.build_card(crun, rosters=SC.load_rosters(), board=[], log=lambda *a: None)
    old = _schema_1_card(new)
    SC.write_outputs([old], str(out), log=lambda *a: None)
    board = SC.read_board(str(out / "scoreboard.jsonl"))
    new = SC.build_card(crun, rosters=SC.load_rosters(), board=board, log=lambda *a: None)
    d = {x["metric"]: x for x in new["delta"]}
    assert d["identified rows"]["prev"] is None and d["Assigned rows"]["prev"] == 5
    # the page re-renders the old card from disk beside the new one
    import dataclasses
    other = dataclasses.replace(crun, path=crun.path + "-other", summary={**crun.summary, "batch_name": "OTHER BATCH"})
    old2 = _schema_1_card(SC.build_card(other, rosters=SC.load_rosters(), board=[], log=lambda *a: None))
    SC.write_outputs([old2], str(out), log=lambda *a: None)
    SC.write_outputs([new], str(out), log=lambda *a: None)
    page = html.unescape((out / "scoreboard.html").read_text())
    assert "OTHER-BATCH|NO3|cover" in page and "TEST-BATCH|NO3|cover" in page
    assert "pre-0.10.0: identified 4 = 2 on this channel's own evidence" in page      # the old card's own block
    assert "pre-0.10.0 0/1/2/1/1/1/1/0/1" in page                                      # its own vector, marked
    board_md = (out / "SCOREBOARD.md").read_text()
    claims_table = board_md[board_md.index("## Claims"): board_md.index("## All metrics")]
    assert "| pre-0.10.0 | 4 | — | 2 | 1 | — |" in claims_table and "| 0.10.0 | 2 | 1 | 2 | 1 | 1 |" in claims_table
    assert "(+" not in claims_table and "(-" not in claims_table                       # no delta across the scales


def test_an_old_card_on_disk_renders_beside_a_new_one(crun, tmp_path):
    out = tmp_path / "board"
    import dataclasses
    other = dataclasses.replace(crun, path=crun.path + "-other", summary={**crun.summary, "batch_name": "OTHER BATCH"})
    old = SC.build_card(other, rosters=SC.load_rosters(), board=[], log=lambda *a: None)
    old.pop("claims"), old.pop("acceptance")
    for arm in ("control", "shift", "adducts"):
        old["decoy"][arm] = None
    old["row"] = {k: v for k, v in old["row"].items() if k in OLD_ROW_KEYS}
    SC.write_outputs([old], str(out), log=lambda *a: None)
    new = SC.build_card(crun, rosters=SC.load_rosters(), board=SC.read_board(str(out / "scoreboard.jsonl")), log=lambda *a: None)
    SC.write_outputs([new], str(out), log=lambda *a: None)
    page = html.unescape((out / "scoreboard.html").read_text())
    assert "card predates C13" in page and "The claim — identified / neutral / ion / tentative" in page
    assert "OTHER-BATCH|NO3|cover" in page and "TEST-BATCH|NO3|cover" in page


def test_cli_summary_leads_with_the_claim(run_dir_claims, tmp_path, capsys):
    assert SC.main([str(run_dir_claims), "--out", str(tmp_path / "b"), "--quiet"]) == 0
    line = capsys.readouterr().out.strip().splitlines()[-1]
    frac = [100.0 * h / COMMITTED for h in (2400, 1000, 360, 500)]
    assert (f": identified 2 / neutral 1 / ion 2 / tentative 1 (signal {frac[0]:.1f} / {frac[1]:.1f} / {frac[2]:.1f} / "
            f"{frac[3]:.1f} %); not assessed 1; Assigned 5 / Candidate 1") in line
    assert "levels 2/1/2/0/1/0/1" in line
