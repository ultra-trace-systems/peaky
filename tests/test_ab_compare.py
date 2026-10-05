"""scripts/ab_compare.py — two synthetic run dirs, every metric checked by hand.

A path change (a cover of files vs `--trace-first`, a fixed centre vs
`--rolling-centre`) rebuilds the peak set from scratch, so the only honest A/B is
on chemistry: the ion a merged row reads, the share of samples carrying it, and
which of the reference run's multi-file neutrals survive. The fixtures here are
six rows wide so each of those numbers can be counted on paper and pinned.

Run: pytest tests/test_ab_compare.py -q
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import ab_compare as AB  # noqa: E402

# Run A -- the reference. Six merged rows, four of them in >=2 files.
LEDGER_A = pd.DataFrame(
    {
        "mz": [100.0000, 200.0000, 300.0000, 400.0000, 500.0000, 600.0000],
        "neutral_formula": ["C5H8O2", "C6H8O4", "C8H12O4", "C9H10O3", "C10H16O", None],
        "adduct": ["[M-H]-"] * 5 + [None],
        "tier": ["Assigned", "Assigned", "Candidate", "Candidate", "Assigned", "Candidate"],
        "n_files": [4, 3, 2, 2, 1, 5],
        "occurrence": [0.95, 0.60, 0.30, 0.08, 0.50, 0.99],
    }
)

# Run B -- the challenger. C8H12O4 and C9H10O3 are gone (the lost pair);
# 200.0004 is 2 ppm off A's 200.0000 and reads a DIFFERENT ion; C11H18O3 is new.
LEDGER_B = pd.DataFrame(
    {
        "mz": [100.0000, 200.0004, 500.0000, 700.0000],
        "neutral_formula": ["C5H8O2", "C5H4N2O3", "C10H16O", "C11H18O3"],
        "adduct": ["[M-H]-"] * 4,
        "tier": ["Assigned", "Candidate", "Assigned", "Assigned"],
        "n_files": [4, 3, 1, 2],
        "occurrence": [0.95, 0.60, 0.50, 0.40],
    }
)

# Four samples per run, and the series only ever carry what their own ledger
# reads. C5H8O2 sits in every sample of both (no move); C6H8O4 is in all four of
# A and none of B (a 100-point loss); C11H18O3 is in two of B and none of A (a
# 50-point gain).
TS_A = pd.DataFrame(
    {
        "sample_item_id": ["s1", "s2", "s3", "s4", "s1", "s2", "s3", "s4"],
        "neutral_formula": ["C5H8O2"] * 4 + ["C6H8O4"] * 4,
        "adduct": ["[M-H]-"] * 8,
    }
)
TS_B = pd.DataFrame(
    {
        "sample_item_id": ["s1", "s2", "s3", "s4", "s1", "s2"],
        "neutral_formula": ["C5H8O2"] * 4 + ["C11H18O3"] * 2,
        "adduct": ["[M-H]-"] * 6,
    }
)


def _write_run(root: Path, ledger: pd.DataFrame, ts: pd.DataFrame | None, summary: dict):
    """A run dir the way `peaky batch` leaves one: out-dir / timestamped / files."""
    import json

    run = root / "BATCH-2026-01-01_2026-01-01T000000Z"
    (run / "per_file").mkdir(parents=True)
    ledger.to_csv(run / "merged_ledger.csv", index=False)
    (run / "batch_summary.json").write_text(json.dumps(summary))
    if ts is not None:
        ts.to_parquet(run / "per_file" / "_batch_ts.parquet", index=False)
    return run


@pytest.fixture
def runs(tmp_path):
    a = _write_run(tmp_path / "A", LEDGER_A, TS_A, {"trace_first": None})
    b = _write_run(
        tmp_path / "B", LEDGER_B, TS_B, {"trace_first": {"n_traces": 9, "n_rolling": 3}}
    )
    return a, b


def test_resolve_run_dir_accepts_parent_or_run_dir(runs):
    a, _ = runs
    assert AB.resolve_run_dir(str(a)) == str(a)
    assert AB.resolve_run_dir(str(a.parent)) == str(a)


def test_resolve_run_dir_refuses_an_ambiguous_parent(tmp_path):
    _write_run(tmp_path / "two", LEDGER_A, None, {})
    second = tmp_path / "two" / "OTHER_2026-01-02T000000Z"
    second.mkdir(parents=True)
    LEDGER_A.to_csv(second / "merged_ledger.csv", index=False)
    with pytest.raises(SystemExit):
        AB.resolve_run_dir(str(tmp_path / "two"))


def test_load_run_reads_the_three_artifacts_and_the_flags(runs):
    a, b = runs
    ra, rb = AB.load_run(str(a)), AB.load_run(str(b))
    assert len(ra.ledger) == 6 and len(rb.ledger) == 4
    assert ra.ts is not None and list(ra.ts.columns) == AB.TS_COLUMNS
    assert not ra.trace_first and rb.trace_first
    assert not ra.rolling and rb.rolling


def test_load_run_tolerates_a_missing_time_series(tmp_path):
    run = _write_run(tmp_path / "noTS", LEDGER_A, None, {})
    assert AB.load_run(str(run)).ts is None


def test_match_by_mz_pairs_within_tolerance_only(runs):
    a, b = runs
    la, lb = AB.load_run(str(a)).ledger, AB.load_run(str(b)).ledger

    # 100.0000, 200.0004 (2 ppm) and 500.0000 pair; 300/400/600 have no partner.
    matched = AB.match_by_mz(la, lb, tol_ppm=6.0)
    assert len(matched) == 3
    assert matched["ppm"].max() == pytest.approx(2.0, abs=0.01)

    # Tighten below 2 ppm and the 200.0004 pair drops out.
    assert len(AB.match_by_mz(la, lb, tol_ppm=1.0)) == 2


def test_ion_disagreement_is_the_pair_that_reads_differently(runs):
    a, b = runs
    la, lb = AB.load_run(str(a)).ledger, AB.load_run(str(b)).ledger
    matched = AB.match_by_mz(la, lb, tol_ppm=6.0)
    disagree = matched[matched["ion_a"] != matched["ion_b"]]
    assert len(disagree) == 1
    assert disagree.iloc[0]["ion_a"] == "C6H8O4 [M-H]-"
    assert disagree.iloc[0]["ion_b"] == "C5H4N2O3 [M-H]-"


def test_ts_coverage_is_the_share_of_samples(runs):
    a, b = runs
    ca, cb = AB.ts_coverage(AB.load_run(str(a))), AB.ts_coverage(AB.load_run(str(b)))
    assert ca[("C5H8O2", "[M-H]-")] == pytest.approx(100.0)
    assert ca[("C6H8O4", "[M-H]-")] == pytest.approx(100.0)
    # B lost C6H8O4 outright and gained C11H18O3 in 2 of its 4 samples.
    assert ("C6H8O4", "[M-H]-") not in cb.index
    assert cb[("C5H8O2", "[M-H]-")] == pytest.approx(100.0)
    assert cb[("C11H18O3", "[M-H]-")] == pytest.approx(50.0)


def test_recovery_counts_a_multi_file_neutrals_that_b_keeps(runs):
    a, b = runs
    la, lb = AB.load_run(str(a)).ledger, AB.load_run(str(b)).ledger

    # A's n_files>=2 neutrals: C5H8O2, C6H8O4, C8H12O4, C9H10O3 (C10H16O is one
    # file, the None row carries no neutral). B keeps only C5H8O2 -- its row at
    # m/z 200 reads C5H4N2O3, so C6H8O4 is lost, not kept.
    per_neutral, kept, total = AB.recovery(la, lb, min_files=2)
    assert total == 4 and kept == 1
    lost = set(per_neutral.loc[~per_neutral["kept"], "neutral"])
    assert lost == {"C6H8O4", "C8H12O4", "C9H10O3"}

    # Raise the bar to 3 files and C8H12O4 / C9H10O3 stop counting at all.
    _, kept3, total3 = AB.recovery(la, lb, min_files=3)
    assert total3 == 2 and kept3 == 1

    # B keeping C10H16O earns nothing: it is a one-file neutral in A.
    assert "C10H16O" not in set(per_neutral["neutral"])


def test_evidence_hist_is_absent_until_the_column_is(runs, tmp_path):
    a, _ = runs
    assert AB.evidence_hist(AB.load_run(str(a))) is None

    levelled = LEDGER_A.assign(evidence_level=["4b", "3c", "4b", "5b", "NA", None])
    run = _write_run(tmp_path / "lvl", levelled, None, {})
    hist = AB.evidence_hist(AB.load_run(str(run)))
    # NA (not assessed) is a level token, re-read as written: never folded into "no level"
    assert hist["4b"] == 2 and hist["NA"] == 1 and hist["—"] == 1
    assert list(hist.index) == ["3c", "4b", "5b", "NA", "—"]          # the scale's order


def test_report_states_every_headline_number(runs):
    a, b = runs
    report = AB.build_report(
        AB.load_run(str(a)),
        AB.load_run(str(b)),
        tol_ppm=6.0,
        min_files=2,
        coverage_delta=5.0,
        top=25,
    )
    assert "| merged rows | 6 | 4 | -2 |" in report
    assert "**25.0 % recovery**" in report          # 1 of 4
    assert "**1 disagree on the ion**" in report
    assert "**1 gained**, **1 lost**" in report     # C11H18O3 +50, C6H8O4 -100
    assert "Neither ledger carries `evidence_level`." in report
    assert "A ran cover path, fixed centre; B ran trace-first, rolling." in report


def test_cli_writes_the_report(runs, tmp_path, capsys):
    a, b = runs
    out = tmp_path / "report.md"
    assert AB.main([str(a), str(b), "--out", str(out)]) == 0
    assert "**25.0 % recovery**" in out.read_text()
    assert str(out) in capsys.readouterr().out


def test_rolling_is_read_from_the_nested_traces_block(tmp_path):
    """A cover-path run writes traces.rolling, not a flat n_rolling.

    Reading only the flat key reported a batch that rolled 965 of its ions as
    "fixed centre" -- the shape this test pins.
    """
    nested = {
        "traces": {
            "rolling": {"enabled": True, "n_rolling": 965, "n_global": 6381},
            "n_traces": 7346,
        }
    }
    run = _write_run(tmp_path / "nested", LEDGER_A, None, nested)
    roll = AB.load_run(str(run)).rolling
    assert roll["n_rolling"] == 965 and roll["n_global"] == 6381

    # The trace-first path writes it flat instead; both must be seen.
    flat = {"trace_first": {"n_traces": 2544, "n_rolling": 1085}}
    run2 = _write_run(tmp_path / "flat", LEDGER_A, None, flat)
    assert AB.load_run(str(run2)).rolling["n_rolling"] == 1085

    # A run that did not roll reports nothing, and a disabled block is not a roll.
    off = {"traces": {"rolling": {"enabled": False, "n_rolling": 0}}}
    run3 = _write_run(tmp_path / "off", LEDGER_A, None, off)
    assert AB.load_run(str(run3)).rolling == {}


def test_report_names_the_rolling_share(tmp_path):
    a = _write_run(tmp_path / "a", LEDGER_A, None, {})
    b = _write_run(
        tmp_path / "b",
        LEDGER_B,
        None,
        {"traces": {"rolling": {"enabled": True, "n_rolling": 965, "n_global": 6381}}},
    )
    report = AB.build_report(
        AB.load_run(str(a)), AB.load_run(str(b)), 6.0, 2, 5.0, 25
    )
    assert "B ran rolling (965 of 7346 ions)" in report


def test_by_stage_prefers_the_summary_and_falls_back_to_the_ledger(tmp_path):
    recorded = _write_run(
        tmp_path / "rec", LEDGER_A, None, {"merged_by_stage": {"cover": 5, "residual": 1}}
    )
    assert AB.load_run(str(recorded)).by_stage == {"cover": 5, "residual": 1}

    staged = LEDGER_A.assign(stage=["cover"] * 5 + ["residual"])
    derived = _write_run(tmp_path / "der", staged, None, {})
    assert AB.load_run(str(derived)).by_stage == {"cover": 5, "residual": 1}

    assert AB.load_run(str(_write_run(tmp_path / "none", LEDGER_A, None, {}))).by_stage == {}


def test_report_warns_when_only_one_arm_ran_a_stage(tmp_path):
    """base ran no residual stage and roll_v2 added 1149 rows through one.

    Comparing the totals then credits a flag with a stage it never ran, which is
    exactly the reading this warning exists to stop.
    """
    a = _write_run(tmp_path / "a", LEDGER_A, None, {"merged_by_stage": {"cover": 6408}})
    b = _write_run(
        tmp_path / "b", LEDGER_B, None, {"merged_by_stage": {"cover": 6197, "residual": 1149}}
    )
    report = AB.build_report(AB.load_run(str(a)), AB.load_run(str(b)), 6.0, 2, 5.0, 25)
    assert "| cover | 6408 | 6197 | -211 |" in report
    assert "| residual | 0 | 1149 | +1149 |" in report
    assert "do not share a stage composition" in report

    # Same stages in both arms: the table is there, the warning is not.
    c = _write_run(tmp_path / "c", LEDGER_B, None, {"merged_by_stage": {"cover": 6000, "residual": 900}})
    ok = AB.build_report(AB.load_run(str(b)), AB.load_run(str(c)), 6.0, 2, 5.0, 25)
    assert "| residual | 1149 | 900 | -249 |" in ok
    assert "do not share a stage composition" not in ok


def test_a_run_without_stage_data_is_not_reported_as_zero(tmp_path):
    """base records no stage breakdown; printing 0 cover rows invents a finding."""
    a = _write_run(tmp_path / "old", LEDGER_A, None, {})            # no stages at all
    b = _write_run(
        tmp_path / "new", LEDGER_B, None, {"merged_by_stage": {"cover": 6197, "residual": 1149}}
    )
    report = AB.build_report(AB.load_run(str(a)), AB.load_run(str(b)), 6.0, 2, 5.0, 25)
    assert "| cover | n/a | 6197 | — |" in report
    assert "| cover | 0 | 6197 |" not in report
    assert "Run A records no stage breakdown" in report
    assert "do not share a stage composition" not in report


# --- claims and row changes ---------------------------------------------------
# A pre-claim run carries `evidence_level` and no `claim`; a claim run carries
# both. The row-change table is what shows two such runs agree on every ion,
# tier and level: the old run's claims are read off its levels, so the claim
# row reads 0 exactly when the stamped claim equals the derived one. The levels
# are on the evidence scale (3c 4a 4b 5a 5b + the reagent and NA buckets).
LEVELLED_A = LEDGER_A.assign(evidence_level=["3c", "4b", "NA", "5b", "4a", None])
CLAIMS_A = ["identified", "ion", "not assessed", "tentative", "neutral", "tentative"]


def test_the_claim_sets_are_the_package_sets():
    from peaky.assignment import evidence as EV

    assert AB.CLAIMS == EV.CLAIMS and AB.CLAIM_KEYS == EV.CLAIM_KEYS
    assert (AB.CLAIM_REAGENT, AB.CLAIM_NA) == (EV.CLAIM_REAGENT, EV.CLAIM_NA)
    assert AB.CLAIM_IDENTIFIED == EV.CLAIM_IDENTIFIED
    assert AB.CLAIM_NEUTRAL == EV.CLAIM_NEUTRAL
    assert AB.CLAIM_ION == EV.CLAIM_ION
    assert AB.LEVEL_ORDER == EV.LEVEL_ORDER and AB.BUCKETS == EV.BUCKETS
    for level in [*EV.LEVEL_ORDER, *EV.BUCKETS, None, float("nan"), "", "nan", "6", "2b", "4c", "4d"]:
        assert AB.claim_class(level) == EV.claim_class(level), level


def test_claims_of_reads_the_column_or_derives_it_from_the_level(tmp_path):
    old = AB.load_run(str(_write_run(tmp_path / "old", LEVELLED_A, None, {})))
    assert AB.claim_source(old) == "derived"
    assert list(AB.claims_of(old)) == CLAIMS_A          # NA read as written, not as "no level"

    stamped = LEVELLED_A.assign(claim=CLAIMS_A)
    new = AB.load_run(str(_write_run(tmp_path / "new", stamped, None, {})))
    assert AB.claim_source(new) == "stamped"
    assert list(AB.claims_of(new)) == list(stamped["claim"])

    bare = AB.load_run(str(_write_run(tmp_path / "bare", LEDGER_A, None, {})))
    assert AB.claim_source(bare) is None and AB.claims_of(bare) is None
    assert AB.claims_hist(bare) is None


def test_row_changes_is_zero_for_identical_runs(tmp_path):
    a = AB.load_run(str(_write_run(tmp_path / "a", LEVELLED_A, None, {})))
    b = AB.load_run(str(_write_run(tmp_path / "b", LEVELLED_A, None, {})))
    ch = AB.row_changes(a, b)
    # the None-neutral row is not an ion: five shared, none on one side only
    assert (ch["shared"], ch["only_a"], ch["only_b"]) == (5, 0, 0)
    assert (ch["tier"], ch["level"], ch["claim"]) == (0, 0, 0)
    assert ch["rows"].empty


def test_row_changes_counts_a_tier_change_and_a_level_change(tmp_path):
    b_led = LEVELLED_A.copy()
    b_led.loc[2, "tier"] = "Assigned"            # C8H12O4: Candidate -> Assigned, level kept
    b_led.loc[3, "evidence_level"] = "4b"        # C9H10O3: 5b -> 4b, tier kept (claim moves too)
    b_led = b_led.drop(index=[4])                # C10H16O -> only in A
    b_led = pd.concat([b_led, pd.DataFrame([{"mz": 700.0, "neutral_formula": "C11H18O3",
                                             "adduct": "[M-H]-", "tier": "Assigned",
                                             "n_files": 2, "occurrence": 0.4,
                                             "evidence_level": "4b"}])], ignore_index=True)
    a = AB.load_run(str(_write_run(tmp_path / "a", LEVELLED_A, None, {})))
    b = AB.load_run(str(_write_run(tmp_path / "b", b_led, None, {})))
    ch = AB.row_changes(a, b)
    assert (ch["shared"], ch["only_a"], ch["only_b"]) == (4, 1, 1)
    assert (ch["tier"], ch["level"], ch["claim"]) == (1, 1, 1)
    rows = ch["rows"]
    assert list(rows["neutral"]) == ["C8H12O4", "C9H10O3"]
    assert (rows.iloc[0]["tier_a"], rows.iloc[0]["tier_b"]) == ("Candidate", "Assigned")
    assert (rows.iloc[1]["level_a"], rows.iloc[1]["level_b"]) == ("5b", "4b")
    assert (rows.iloc[1]["claim_a"], rows.iloc[1]["claim_b"]) == ("tentative", "ion")


def test_a_stamped_claim_equal_to_the_derived_one_is_no_change(tmp_path):
    """The pre-claim run (A) has no claim column; the claim run (B) stamps it."""
    stamped = LEVELLED_A.assign(claim=LEVELLED_A["evidence_level"].map(AB.claim_class))
    a = AB.load_run(str(_write_run(tmp_path / "pre", LEVELLED_A, None, {})))
    b = AB.load_run(str(_write_run(tmp_path / "post", stamped, None, {})))
    ch = AB.row_changes(a, b)
    assert (ch["tier"], ch["level"], ch["claim"]) == (0, 0, 0)
    # a stamped claim that disagrees with its own level is a change, not hidden
    wrong = stamped.copy()
    wrong.loc[0, "claim"] = "ion"
    c = AB.load_run(str(_write_run(tmp_path / "wrong", wrong, None, {})))
    assert AB.row_changes(a, c)["claim"] == 1

    report = AB.build_report(a, b, 6.0, 2, 5.0, 25)
    for line in ("| ion | 0 |", "| tier | 0 |", "| level | 0 |", "| claim | 0 |",
                 "| rows only in A | 0 |", "| rows only in B | 0 |"):
        assert line in report, line
    assert "Run A's ledger has no `claim` column" in report
    assert "Run B's ledger has no `claim` column" not in report
    # the section sits after the ion disagreements, before the coverage
    assert (report.index("## Ion disagreements") < report.index("## Row changes")
            < report.index("## Time-series coverage"))


def test_row_changes_pairs_a_repeated_ion_row_by_row():
    a = pd.DataFrame({"mz": [100.0, 100.5], "neutral_formula": ["C5H8O2"] * 2,
                      "adduct": ["[M-H]-"] * 2, "tier": ["Assigned", "Candidate"]})
    b = a.iloc[[0]]
    ch = AB.row_changes(a, b)
    assert (ch["shared"], ch["only_a"], ch["only_b"], ch["tier"]) == (1, 1, 0, 0)


def test_report_writes_the_claims_histogram(runs, tmp_path):
    a, b = runs
    report = AB.build_report(AB.load_run(str(a)), AB.load_run(str(b)), 6.0, 2, 5.0, 25)
    assert "Neither ledger carries `claim` or `evidence_level`." in report

    stamped = LEVELLED_A.assign(claim=CLAIMS_A)
    ra = AB.load_run(str(_write_run(tmp_path / "old", LEVELLED_A, None, {})))
    rb = AB.load_run(str(_write_run(tmp_path / "new", stamped.iloc[:4], None, {})))
    report = AB.build_report(ra, rb, 6.0, 2, 5.0, 25)
    assert report.index("## Evidence levels") < report.index("## Claims")
    claims = report[report.index("## Claims"):]
    for line in ("| identified | 1 | 1 | +0 |", "| neutral | 1 | 0 | -1 |", "| ion | 1 | 1 | +0 |",
                 "| tentative | 2 | 1 | -1 |", "| not assessed | 1 | 1 | +0 |"):
        assert line in claims, line
    assert "| reagent |" not in claims                   # a bucket neither run has is not listed
    order = [claims.index(f"| {c} |") for c in ("identified", "neutral", "ion", "tentative", "not assessed")]
    assert order == sorted(order)


def test_a_run_without_levels_is_not_compared_on_level_or_claim(tmp_path):
    """A pre-level run (no `evidence_level`, no `claim`) against a levelled one:
    the missing columns are not a change on every shared ion. Only the tier is
    compared, only the real tier change is listed, and the claims histogram
    reads n/a for the run that has none rather than 0."""
    b_led = LEVELLED_A.copy()
    b_led.loc[2, "tier"] = "Assigned"            # C8H12O4: Candidate -> Assigned
    a = AB.load_run(str(_write_run(tmp_path / "pre", LEDGER_A, None, {})))
    b = AB.load_run(str(_write_run(tmp_path / "post", b_led, None, {})))
    ch = AB.row_changes(a, b)
    assert ch["shared"] == 5
    assert (ch["tier"], ch["level"], ch["claim"]) == (1, None, None)
    assert list(ch["rows"]["neutral"]) == ["C8H12O4"]

    report = AB.build_report(a, b, 6.0, 2, 5.0, 25)
    for line in ("| tier | 1 |", "| level | — |", "| claim | — |",
                 "| identified | n/a | 1 | — |", "| neutral | n/a | 1 | — |", "| ion | n/a | 1 | — |",
                 "| tentative | n/a | 2 | — |", "| not assessed | n/a | 1 | — |"):
        assert line in report, line
    assert ("Run A's ledger carries neither `claim` nor `evidence_level`: "
            "level and claim are not compared.") in report
    assert "Run B's ledger carries neither" not in report
    assert "| identified | 0 |" not in report
    assert "The 1 changed at the lowest m/z:" in report


def test_a_run_levelled_before_the_scale_is_read_on_it(tmp_path):
    """A run levelled on an older scale (B-series letters, `evidence_axes`, its
    claims stamped on that scale) against a run on the evidence scale: EVERY
    letter reads as no level (the shared 4a / 4b / 5b too), its claims re-read
    tentative, its levels are not compared letter for letter, and the report
    says so."""
    old_led = LEDGER_A.assign(evidence_level=["2b", "4b", "4c", "5b", "4a", None],
                              evidence_axes=["iso|chan2", "iso", "", "", "iso|anchor", None],
                              claim=["identified", "ion", "ion", "tentative", "identified", "tentative"])
    new_led = LEVELLED_A.assign(claim=CLAIMS_A)
    a = AB.load_run(str(_write_run(tmp_path / "old", old_led, None, {})))
    b = AB.load_run(str(_write_run(tmp_path / "new", new_led, None, {})))
    assert AB.before_scale(a) and not AB.before_scale(b)
    assert AB.claim_source(a) == "derived" and AB.claim_source(b) == "stamped"
    assert list(AB.claims_of(a)) == ["tentative"] * 6
    assert list(AB.levels_of(a).isna()) == [True] * 6
    ch = AB.row_changes(a, b)
    assert ch["level"] is None
    assert ch["claim"] == sum(1 for c in CLAIMS_A if c != "tentative")
    report = AB.build_report(a, b, 6.0, 2, 5.0, 25)
    assert (f"> Run A was levelled on a scale before the evidence scale: its 5 levelled row(s) read as "
            f"{AB.OLD_SCALE_NO_LEVEL} (the letters this scale shares too), so its claims read tentative, "
            "and its levels are not compared with the other run's.") in report
    from peaky.assignment import evidence as EV
    assert AB.OLD_SCALE_NO_LEVEL == f"no level (pre-{EV.SCALE_RELEASE} scale)"
    assert f"| {AB.OLD_SCALE_NO_LEVEL} | 5 |" in report and "| 4a | 0 |" in report
    # the changed-rows table prints no older-scale letter either: the old run's lettered rows read as no level
    table = report.split("| m/z (A) | ion | tier A → B | level A → B | claim A → B |")[1].split("\n\n")[0]
    rows = [r for r in table.splitlines() if r.startswith("| ") and not r.startswith("|---")]
    assert rows and all(f"| {AB.OLD_SCALE_NO_LEVEL} → " in r or "| — → " in r for r in rows), rows
    for letter in ("1", "2a", "2b", "3a", "3b", "3c", "3d", "4a", "4b", "4c", "4d", "5a", "5b"):
        assert f"| {letter} → " not in table, letter
    assert set(AB.row_changes(a, b)["rows"]["level_a"]) <= {AB.OLD_SCALE_NO_LEVEL, "—"}
    # two runs on the older scale: no letter of either is a level of this scale, so no level change is counted
    c = AB.load_run(str(_write_run(tmp_path / "old2", old_led.assign(evidence_level=["2a", "4c", "4c", "5b", "4a",
                                                                                     None]), None, {})))
    assert AB.row_changes(a, c)["level"] == 0
