"""C19(c) mutation hunt: the survivors of the `tentative_lead` split that the
first test file left alive, each pinned here.

Pinned here:
- the pooled `lead` fact holds when ANY row across a batch's files is a lead,
  whichever file holds it -- in the engine (evidence.level_pooled) and in the
  reference script (level_ledger over a run dir's per_file ledgers), in lockstep.
  The first file's row alone, or ALL rows, is the mutant;
- the Below assignability sheet reads either flag column: a ledger that carries
  `tentative_lead` and no `below_assignability` still lists its leads (a missing
  column reads False, ledger.has_flags is "either column").

Equivalent survivors (no test can kill them; the argument is in the hunt report):
- the ion-only stage's own `L.reset_flags(ledger, j)` (removed, or narrowed to one
  flag): the `L.commit_assignment` just above it already resets both flags on
  row j and nothing in between writes a flag;
- the reference-list dim rescue creating only `below_assignability` before
  `L.mark_lead`: mark_lead's own ensure_flags then creates the lead column, the
  same columns, order and values as `L.ensure_flags` first.

Run: pytest tests/test_tentative_lead_hunt.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

from peaky.assignment import evidence as EV
from peaky.reporting import report as R
from tests.test_tentative_lead import _m0, _frame, _report_ledger, _write

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import level_ledger as LL  # noqa: E402

BELOW, LEAD = "below_assignability", "tentative_lead"
PAIR = ("C8H12O4", "[M-H]-")


def _files(lead_in: str) -> dict[str, pd.DataFrame]:
    """Two files of one batch: the same pair in both, a lead in `lead_in` only; a
    second, never-flagged pair beside it. The anchor alone would give 4b."""
    out = {}
    for name in ("a", "b"):
        out[name] = _frame([_m0("p", PAIR[0], lead=(name == lead_in), anchor="x"),
                            _m0("q", "C9H14O4", anchor="x", mz=210.0)])
    return out


@pytest.mark.parametrize("lead_in", ["a", "b"])
def test_the_engine_pools_the_lead_over_every_file_whichever_holds_it(lead_in):
    pairs = EV.level_pooled(_files(lead_in)).set_index(["neutral_formula", "adduct"])
    assert bool(pairs.at[PAIR, "lead"]) and not bool(pairs.at[PAIR, "below"])
    assert pairs.at[PAIR, "evidence_level"] == "5b"
    assert pairs.at[PAIR, "level_reason"] == "5b: below assignability"
    clean = ("C9H14O4", "[M-H]-")
    assert not bool(pairs.at[clean, "lead"]) and pairs.at[clean, "evidence_level"] == "4b"
    # the mutant: neither file a lead -> the anchor's 4b
    none = EV.level_pooled(_files("none")).set_index(["neutral_formula", "adduct"])
    assert not bool(none.at[PAIR, "lead"]) and none.at[PAIR, "evidence_level"] == "4b"


@pytest.mark.parametrize("lead_in", ["a", "b"])
def test_the_reference_script_pools_the_lead_over_a_run_dirs_files(tmp_path, lead_in):
    files = _files(lead_in)
    run = tmp_path / "RUN"
    for name, frame in files.items():
        _write(run / "per_file" / f"{name}_ledger.csv", frame)
    got = LL.run([str(run)], []).set_index(["neutral", "adduct"])
    assert int(got.at[PAIR, "n_files"]) == 2                    # one source, both files
    assert bool(got.at[PAIR, "lead"]) and not bool(got.at[PAIR, "below"])
    assert got.at[PAIR, "level"] == "5b"
    assert got.at[("C9H14O4", "[M-H]-"), "level"] == "4b"
    # lockstep with the engine's pooled level, pair for pair
    eng = EV.level_pooled(files).set_index(["neutral_formula", "adduct"])["evidence_level"]
    assert got["level"].to_dict() == eng.to_dict()


def test_the_below_assignability_sheet_reads_a_ledger_with_the_lead_column_only():
    """A ledger carrying `tentative_lead` and no `below_assignability` (a missing
    column reads False) still lists its lead rows, as with both columns."""
    both = _report_ledger(LEAD)
    lead_only = both.drop(columns=[BELOW])
    assert LEAD in lead_only.columns and BELOW not in lead_only.columns
    sheet = R.build_sheets(lead_only)["Below assignability"]
    assert list(sheet["neutral_formula"]) == ["C5H6ClN2OP"]
    assert bool(sheet[LEAD].iloc[0])
    assert list(sheet["neutral_formula"]) == list(R.build_sheets(both)["Below assignability"]["neutral_formula"])
    # the mutant: no lead in the column either -> an empty sheet
    lead_only[LEAD] = False
    assert R.build_sheets(lead_only)["Below assignability"].empty
