"""peaky's verdict at the publish boundary (offline).

A batch import row carries the m/z, the formulas and the mechanism id - no
verdict - so a merged row peaky holds Candidate would land in Mascope's batch
ledger looking exactly like an Assigned one. `publish-batch` therefore sends the
Assigned merged rows only, unless --include-candidates. The per-sample import
does carry peaky's verdict (`engine_tier`), but Mascope's own `tier` column is
derived server-side, so the publish summary leads with how many rows peaky holds
Candidate that Mascope will show as 'assigned'.
"""
import json

import numpy as np
import pandas as pd
import pytest

from peaky import cli
from peaky.io import publish as P


def _merged(**extra):
    frame = pd.DataFrame({
        "mz": [181.0707, 250.1, 203.0526, 300.5, 150.2],
        "neutral_formula": ["C6H12O6", "C10H20O5", "C6H12O6", None, "C5H8O4"],
        "adduct": ["[M+H]+", "[M+H]+", "[M+Na]+", None, "[M+H]+"],
        "tier": ["Assigned", "Candidate", "Candidate", None, "Identified"],
        "ion_score": [0.9, 0.5, 0.6, np.nan, 0.95],
    })
    return frame.assign(**extra)


def _run_dir(tmp_path, merged):
    run = tmp_path / "run"
    run.mkdir()
    merged.to_csv(run / "merged_ledger.csv", index=False)
    (run / "batch_summary.json").write_text(json.dumps({"merged_tiers": {"Assigned": 2, "Candidate": 2}}))
    return run


def test_batch_rows_send_the_assigned_rows_only_by_default():
    rows, summary = P.build_batch_rows(_merged())
    # the Assigned row and the legacy 'Identified' spelling of the same tier go;
    # the two Candidates are held back and counted by peaky's tier
    assert [r["formula"] for r in rows] == ["C6H12O6", "C5H8O4"]
    assert summary["rows"] == 2
    assert summary["by_tier"] == {"Assigned": 1, "Identified": 1}
    assert summary["held_back"] == {"Candidate": 2}
    assert summary["include_candidates"] is False
    # a row without a formula is still dropped as such, not counted as held back
    assert summary["dropped_no_formula"] == 1
    # the payload shape does not change: still the four fields the server reads
    assert set(rows[0]) == {"mz", "formula", "ion_formula", "ionization_mechanism_id"}


def test_include_candidates_sends_every_row_with_a_formula():
    rows, summary = P.build_batch_rows(_merged(), include_candidates=True)
    assert summary["rows"] == len(rows) == 4
    assert summary["by_tier"] == {"Assigned": 1, "Candidate": 2, "Identified": 1}
    assert summary["held_back"] == {}
    assert summary["dropped_no_formula"] == 1


def test_an_untiered_merged_row_is_held_back_by_default():
    rows, summary = P.build_batch_rows(_merged(tier=["Assigned", None, "Candidate", None, "Assigned"]))
    assert summary["rows"] == len(rows) == 2
    assert summary["held_back"] == {"untiered": 1, "Candidate": 1}


def test_a_merged_ledger_without_a_tier_column_is_refused_unless_every_row_is_asked_for():
    untiered = _merged().drop(columns=["tier"])
    with pytest.raises(P.PublishError, match="--include-candidates"):
        P.build_batch_rows(untiered)
    rows, summary = P.build_batch_rows(untiered, include_candidates=True)
    assert summary["rows"] == len(rows) == 4
    assert summary["by_tier"] == {"untiered": 4}


def test_batch_config_records_the_published_tiers():
    quiet = {"log": lambda *a: None}
    assert P.batch_config({"reagent": "Ur"}, published_tiers={"Assigned": 3}, **quiet)["published_tiers"] == {
        "Assigned": 3}
    assert "published_tiers" not in P.batch_config({"reagent": "Ur"}, **quiet)


def test_cli_publish_batch_defaults_to_assigned_and_says_what_it_held_back(tmp_path, capsys):
    run = _run_dir(tmp_path, _merged())
    out = tmp_path / "payload.json"
    args = cli.build_parser().parse_args(
        ["publish-batch", str(run), "--dry-run", "--no-resolve-mechanisms", "--out", str(out)])
    assert args.include_candidates is False
    cli.cmd_publish_batch(args)
    text = capsys.readouterr().out
    assert "rows       2 of 5 merged row(s)" in text
    assert cli.BATCH_HELD_BACK.format(n=2, tiers="Candidate 2") in text
    assert "WARNING" not in text
    payload = json.loads(out.read_text())
    assert [r["formula"] for r in payload["rows"]] == ["C6H12O6", "C5H8O4"]
    assert payload["config"]["published_tiers"] == {"Assigned": 1, "Identified": 1}


def test_cli_include_candidates_sends_them_with_a_warning(tmp_path, capsys):
    run = _run_dir(tmp_path, _merged())
    out = tmp_path / "payload.json"
    cli.cmd_publish_batch(cli.build_parser().parse_args(
        ["publish-batch", str(run), "--dry-run", "--no-resolve-mechanisms", "--include-candidates",
         "--out", str(out)]))
    text = capsys.readouterr().out
    assert "rows       4 of 5 merged row(s)" in text
    assert cli.BATCH_CANDIDATES_SENT.format(n=2, total=4, tiers="Candidate 2") in text
    assert "held back" not in text
    payload = json.loads(out.read_text())
    assert len(payload["rows"]) == 4
    assert payload["config"]["published_tiers"] == {"Assigned": 1, "Candidate": 2, "Identified": 1}


def test_cli_publish_batch_stops_when_nothing_is_assigned(tmp_path, capsys):
    run = _run_dir(tmp_path, _merged(tier=["Candidate"] * 5))
    with pytest.raises(SystemExit, match="no merged row with a formula is Assigned"):
        cli.cmd_publish_batch(cli.build_parser().parse_args(
            ["publish-batch", str(run), "--dry-run", "--no-resolve-mechanisms"]))


def test_cli_publish_batch_refuses_an_untiered_ledger_by_default(tmp_path, capsys):
    run = _run_dir(tmp_path, _merged().drop(columns=["tier"]))
    with pytest.raises(SystemExit, match="no 'tier' column"):
        cli.cmd_publish_batch(cli.build_parser().parse_args(
            ["publish-batch", str(run), "--dry-run", "--no-resolve-mechanisms"]))


def _ledger(tiers, ion_scores):
    n = len(tiers)
    return pd.DataFrame({
        "sample_item_id": "S0000000000000001",
        "peak_id": [f"P{k:018d}" for k in range(n)],
        "mz": [150.0 + k for k in range(n)],
        "height": [1000.0] * n,
        "role": "M0",
        "neutral_formula": "C6H6",
        "ion_score": ion_scores,
        "tier": tiers,
    })


def test_build_rows_counts_the_candidates_mascope_will_call_assigned():
    bands = P.DEFAULT_TIER_BANDS
    # a high fit peaky demoted -> Mascope 'assigned'; a low-fit Candidate stays
    # 'candidate' in Mascope too; an Assigned row agrees
    _, summary = P.build_rows(_ledger(["Candidate", "Candidate", "Assigned"], [0.99, 0.5, 0.99]),
                              intensity_column="height", bands=bands)
    assert summary["candidate_shown_assigned"] == 1
    assert summary["engine_tier_disagreements"] == 1
    _, none = P.build_rows(_ledger(["Assigned"], [0.99]), intensity_column="height", bands=bands)
    assert none["candidate_shown_assigned"] == 0


def _publish_text(tmp_path, capsys, led):
    path = tmp_path / "led.csv"
    led.to_csv(path, index=False)
    cli.cmd_publish(cli.build_parser().parse_args(
        ["publish", str(path), "--intensity", "height", "--dry-run", "--no-resolve-mechanisms"]))
    return capsys.readouterr().out


def test_cli_publish_summary_leads_with_the_candidates_mascope_will_call_assigned(tmp_path, capsys):
    text = _publish_text(tmp_path, capsys, _ledger(["Candidate", "Candidate", "Assigned"], [0.99, 0.98, 0.99]))
    lines = [ln for ln in text.splitlines() if ln.strip()]
    note = cli.CANDIDATE_SHOWN_ASSIGNED.format(n=2)
    assert note in lines
    # it leads the summary: the first line after the intensity line, before the sample
    assert lines.index(note) == lines.index(next(ln for ln in lines if ln.startswith("sample "))) - 1
    assert "engine tier" in note and "tier_disagrees" in note


def test_cli_publish_summary_has_no_note_when_no_candidate_reads_assigned(tmp_path, capsys):
    text = _publish_text(tmp_path, capsys, _ledger(["Assigned", "Candidate"], [0.99, 0.5]))
    assert "will show as Mascope 'assigned'" not in text
