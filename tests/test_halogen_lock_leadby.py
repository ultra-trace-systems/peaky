"""C11+b: `lead_by`, which setter made a row a tentative lead.

Rule H (the halogen lock) lifts a lead only where every setter behind it is one
the lock answers, and on the pooled batch the only record of the setter the
level can read is a per-row column: the commentary that also names it is not
among the columns `evidence.trim` keeps. This commit adds the column and moves
no level.

Pinned here:
- each lead setter writes its own code (the three speculative-residual reasons
  apart), each hard setter writes none, and a row two setters mark carries both;
- `ensure_flags` creates the column, commit / clear / displace reset it, and a
  reset never creates it;
- a column read back all-empty from a CSV (float NaN) takes a code;
- `evidence.trim`, the publish provenance and the Below assignability sheet
  carry it; the reference script reads it;
- nothing levels on it yet: the pooled levels with and without the column, and
  of a ledger written before it, are the same frame.

Run: pytest tests/test_halogen_lock_leadby.py -q
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest

from peaky.assignment import cleanup as CL
from peaky.assignment import evidence as EV
from peaky.assignment import ledger as L
from peaky.assignment import plausibility as PL
from peaky.assignment import reflists as RL
from peaky.io import publish as PUB
from peaky.reporting import report as R
from tests.test_tentative_lead import (
    HARD, LL, P, RLIST, _flagged_ledger, _frame, _m0, _mixed, _oracle, _report_ledger, _ResidualCfg,
    _residual, _tiered, _write)

BY = "lead_by"
QUIET = dict(log=lambda *a: None)


def _by(led, i) -> str:
    v = led.at[i, BY] if BY in led.columns else ""
    return "" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v)


# =========================================================================== the setters
def test_the_tier_stage_creates_the_column_empty():
    led = _tiered([dict(peak_id="p", role="M0", neutral_formula="C10H16O4", adduct="[M-H]-")])
    assert BY in led.columns and _by(led, 0) == ""
    assert L.LEAD_BY == BY and BY not in L.ASSIGNABILITY_FLAGS


@pytest.mark.parametrize("row, code", [
    (_residual("C6H5N3", "[M+Br]-"), "spec_n3"),
    (_residual("C12H6O2", "[M-H]-", comm="-2xCH2 (0 supporting anchors)"), "spec_gapfill"),
    (_residual("C9H14O4", "[M+CO3]-"), "spec_minor"),
    (_residual("C9H12O5", "[M-H]-", ppm=5.0, iso='[{"label": "13C"}]'), ""),      # off-calibration: hard
])
def test_the_speculative_residual_names_its_reason(row, code):
    led = _tiered([row])
    CL.demote_speculative_residual(led, _ResidualCfg(), **QUIET)
    assert _by(led, 0) == code
    assert bool(led.at[0, "tentative_lead"]) == bool(code)


def test_the_off_budget_demote_names_itself():
    led = _tiered([dict(peak_id="p", role="M0", mz=200.0, neutral_formula="C5H6ClN2OP", adduct="[M-H]-",
                        method="certified:multi-channel"),
                   dict(peak_id="q", role="M0", mz=201.0, neutral_formula="C10H16O5", adduct="[M-H]-")])
    PL.demote_implausible(led, audit=[], context="ambient-air", **QUIET)
    assert (_by(led, 0), _by(led, 1)) == ("off_budget", "")


def test_the_radical_anion_and_the_reagent_n_reread_name_themselves():
    led = _tiered([dict(peak_id="u", role="M0", neutral_formula="C6H6", adduct="[M+CO3]-", ion_formula="",
                        dbe=4.0)])
    CL.relabel_radical_anions(led, **QUIET)
    assert _by(led, 0) == "radical_anion"
    led = _tiered([dict(peak_id="u", role="M0", neutral_formula="C5H6", adduct="[M+(CH4N2O)H]+",
                        ion_formula="", dbe=3.0)])
    CL.relabel_reagent_n_adducts(led, **QUIET)
    assert _by(led, 0) == "reagent_n"


def test_a_corroborated_radical_anion_clears_the_code_with_the_flags():
    led = _tiered([dict(peak_id="c", role="M0", neutral_formula="C3H4", adduct="[M+CO3]-", ion_formula="",
                        dbe=2.0),
                   dict(peak_id="k", role="M0", neutral_formula="C4H4O3", adduct="[M-H]-", ion_formula="",
                        dbe=3.0)])
    led.loc[0, ["below_assignability", "tentative_lead"]] = True
    led.loc[0, BY] = "spec_minor"                          # whatever an earlier stage said
    CL.relabel_radical_anions(led, **QUIET)
    assert _by(led, 0) == "" and not bool(led.at[0, "tentative_lead"])


def test_the_reflist_dim_rescue_names_itself_and_the_confirmed_one_nothing():
    led = L.new_ledger(pd.DataFrame([("conf", P_MZ["conf"], 1.0e5), ("dim", P_MZ["dim"], 500.0)],
                                    columns=["peak_id", "mz", "height"]))
    cfg = P.PassConfig(height_cutoff_cps=100.0)
    cfg.cal_mu, cfg.cal_sigma, cfg.mechanism_ids = 0.0, 0.3, None
    RL.rescue_unexplained_by_reflist(None, "S", led, None, cfg, [RLIST], ["[M-H]-"], score_fn=_oracle, **QUIET)
    by = led.set_index("peak_id")
    assert (str(by.at["dim", BY]), str(by.at["conf", BY])) == ("reflist_dim", "")


def _p_mz():
    from peaky.chem import chemistry as C
    from tests.test_tentative_lead import F_CONF, F_DIM
    return {"conf": C.ion_mz(F_CONF, "[M-H]-"), "dim": C.ion_mz(F_DIM, "[M-H]-")}


P_MZ = _p_mz()


@pytest.mark.parametrize("name", sorted(HARD))
def test_each_hard_setter_writes_no_code(name):
    demote, row = HARD[name]
    led = _tiered([dict(dict(peak_id="p", role="M0", mz=300.0, adduct="[M-H]-"), **row)])
    demote(led)
    assert bool(led.at[0, "below_assignability"]) and _by(led, 0) == ""


def test_two_setters_on_one_row_both_named_sorted_once():
    led = _tiered([dict(peak_id="p", role="M0", neutral_formula="C10H16O4", adduct="[M-H]-")])
    L.mark_lead(led, 0, "spec_minor")
    L.mark_lead(led, 0, "off_budget")
    L.mark_lead(led, 0, "spec_minor")
    assert _by(led, 0) == "off_budget|spec_minor"
    assert L.lead_setters(led.at[0, BY]) == {"off_budget", "spec_minor"}
    assert L.lead_setters(np.nan) == L.lead_setters(None) == L.lead_setters("") == frozenset()


def test_an_unknown_code_is_refused():
    led = _tiered([dict(peak_id="p", role="M0", neutral_formula="C10H16O4", adduct="[M-H]-")])
    with pytest.raises(L.LedgerError):
        L.mark_lead(led, 0, "off-budget")
    assert not bool(led.at[0, "tentative_lead"])


def test_a_column_read_back_empty_from_a_csv_takes_a_code():
    led = _tiered([dict(peak_id="p", role="M0", neutral_formula="C10H16O4", adduct="[M-H]-"),
                   dict(peak_id="q", role="M0", neutral_formula="C9H14O4", adduct="[M-H]-")])
    buf = io.StringIO()
    led.to_csv(buf, index=False)
    back = pd.read_csv(io.StringIO(buf.getvalue()))
    assert back[BY].dtype.kind == "f"                      # all empty: float NaN
    assert L.mark_lead(back, 1, "spec_gapfill") is True
    assert (_by(back, 0), _by(back, 1)) == ("", "spec_gapfill")
    L.reset_flags(back, 1)
    assert _by(back, 1) == ""


# =========================================================================== the reset
def test_commit_clear_and_displace_reset_the_code():
    def flagged():
        led = _flagged_ledger()
        led[BY] = "spec_gapfill"
        return led
    led = flagged()
    a = led.index[led["peak_id"] == "A"][0]
    L.commit_assignment(led, "A", neutral_formula="C10H14O4", adduct="[M-H]-", ion_score=0.9, pass_no=2,
                        method="cheminfo", confidence="High", commentary="re-won", overwrite=True)
    assert _by(led, a) == "" and _by(led, led.index[led["peak_id"] == "C"][0]) == "spec_gapfill"
    led = flagged()
    L.clear_assignment(led, "A", reason="audit")
    assert _by(led, a) == ""
    led = flagged()
    L.displace_to_isotopologue(led, "C", "A", iso_label="13C")
    assert _by(led, led.index[led["peak_id"] == "C"][0]) == ""
    assert _by(led, a) == "spec_gapfill"                  # the parent keeps its own


def test_the_reset_never_creates_the_column():
    led = L.new_ledger(pd.DataFrame({"peak_id": ["A"], "mz": [200.0], "height": [1e5]}))
    L.commit_assignment(led, "A", neutral_formula="C10H16O4", adduct="[M-H]-", ion_score=0.9, pass_no=1,
                        method="cheminfo", confidence="High", commentary="Pass 1")
    L.clear_assignment(led, "A", reason="audit")
    assert BY not in led.columns


# =========================================================================== the readers
def test_trim_publish_and_the_sheet_carry_it():
    assert BY in EV.PREDICATE_COLUMNS
    frame = _frame([_m0("p", "C7H12O4", lead=True)])
    frame[BY] = "spec_n3"
    assert list(EV.trim(frame)[BY]) == ["spec_n3"]
    cols = PUB._ENGINE_PROVENANCE_COLUMNS
    assert BY in cols and cols.index(BY) == cols.index("tentative_lead") + 1
    prov = PUB._engine_provenance(pd.Series({"tier": "Candidate", "tentative_lead": True, BY: "off_budget"}), None)
    assert prov[BY] == "off_budget"
    assert BY not in PUB._engine_provenance(pd.Series({"tier": "Assigned", "tentative_lead": False, BY: ""}), None)
    led = _report_ledger("tentative_lead")
    led.loc[led["peak_id"] == "D", BY] = "off_budget"
    sheet = R.build_sheets(led)["Below assignability"]
    assert list(sheet[BY]) == ["off_budget"]


def test_nothing_levels_on_it_yet(tmp_path):
    """The column moves no level in the engine or the reference script: with it,
    without it, and on a ledger written before the split."""
    frame = _mixed("lead")
    coded = frame.assign(**{BY: np.where(frame["tentative_lead"].astype(bool), "off_budget", "")})
    base = EV.level_pooled({"f": frame})
    for f in (coded, frame.drop(columns=[BY])):
        pd.testing.assert_frame_equal(EV.level_pooled({"f": f}), base)
        pd.testing.assert_frame_equal(EV.compute_levels(f), EV.compute_levels(frame))
    got = LL.run([str(_write(tmp_path / "c" / "s_ledger.csv", coded))], [])
    ref = LL.run([str(_write(tmp_path / "n" / "s_ledger.csv", frame.drop(columns=[BY])))], [])
    assert list(got.level) == list(ref.level)
