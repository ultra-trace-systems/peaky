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

    levelled = LEDGER_A.assign(evidence_level=["2b", "2b", "4c", "5b", "4c", None])
    run = _write_run(tmp_path / "lvl", levelled, None, {})
    hist = AB.evidence_hist(AB.load_run(str(run)))
    assert hist["2b"] == 2 and hist["4c"] == 2 and hist["—"] == 1


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
