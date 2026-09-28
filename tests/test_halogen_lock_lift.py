"""C11+b rule H, the lift: a halogen lock lifts a tentative lead (EVIDENCE_LEVELS
§3 `lead_lift`, §4.2).

On the pooled batch a pair the batch's time series locks (batch/iso_checks.py,
check H) stops reading its lead as hard where every lead row was set by a setter
the lock answers -- a speculative residual fit, a reference-list match too dim
to confirm, or (only where the locked halogen is the neutral's sole budget
violation, `budget_ok`) the element budget -- and no row is below
assignability. The lock is the pair's isotope axis; the anchor, second channel
and acid branch its flagged rows alone gave go with the flag, for that pair
only. The vetoes (the C11+a checks, rule K) outrank the lift; per-file levels
never see it. The reference script lifts in lockstep.

Equivalent survivor of the mutation hunt: dropping the iso_veto filter from the
lift set in evidence._level_pairs -- a vetoed pair is alien, and an alien pair is
outside the lift already (the filter states the precedence; rule K's label_veto,
which is not alien, needs its own and a test kills its removal).

Run: pytest tests/test_halogen_lock_lift.py -q
"""

from __future__ import annotations

import importlib.util
import inspect
import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from peaky.assignment import evidence as EV
from peaky.batch import iso_checks as IC
from peaky.chem import chemistry as C
from tests.test_tentative_lead import _child, _frame, _m0

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import level_ledger as LL  # noqa: E402

CL1, CL2 = "C6H9ClO3", "C6H10Cl2O4"
H, NO3, NO3L = "[M-H]-", "[M+NO3]-", "[M+^NO3]-"
NOTE = "a 0.39x line at the 37Cl offset (1 Cl: 0.21-0.46), r 0.94, in 98% of 250 spectra, +0.20 ppm"


def lock(*keys, budget_ok=True, note=NOTE) -> dict:
    return {"veto": {}, "lock": {k: {"element": "Cl", "n": 1, "budget_ok": budget_ok, "note": note} for k in keys}}


def _r(pid, neutral=CL1, adduct=H, *, lead_by=None, mz=None, **kw) -> dict:
    """One committed M0 row (test_tentative_lead._m0) with its `lead_by`: the
    speculative gap-fill code on a lead row unless told otherwise."""
    mz = mz if mz is not None else C.ion_mz(neutral, adduct)
    row = _m0(pid, neutral, adduct, mz=mz, **kw)
    row["lead_by"] = lead_by if lead_by is not None else ("spec_gapfill" if kw.get("lead") else "")
    return row


def _f(*rows) -> pd.DataFrame:
    return _frame(list(rows))


def _pairs(files: dict, **kw) -> pd.DataFrame:
    return EV.level_pooled(files, **kw).set_index(["neutral_formula", "adduct"])


def _one(files: dict, key=(CL1, H), **kw) -> pd.Series:
    return _pairs(files, **kw).loc[key]


# =========================================================================== the lift
def test_a_locked_lead_only_pair_lifts_to_4b():
    files = {"f1": _f(_r("p", lead=True))}
    before = _one(files)
    assert (before["evidence_level"], before["level_reason"]) == ("5b", "5b: below assignability")
    after = _one(files, iso=lock((CL1, H)))
    assert (after["evidence_level"], after["level_reason"]) == ("4b", "4b: one corroboration (iso)")
    assert bool(after["lead_lift"]) and not bool(after["lead"]) and bool(after["iso"])
    assert after["lock_note"] == NOTE and after["claim"] == "ion"
    assert after["evidence_axes"] == "iso|lead_lift|files:1"
    # the lift is recorded as a fact, never an axis
    assert "lead_lift" not in EV.AXES and after["n_axes"] == 1


def test_the_flag_in_either_column_no_longer_levels_alike_under_a_lock():
    """C19(c) made the two flags level alike; rule H lifts only the lead half --
    the lift never touches below assignability."""
    by_below = {"f1": _f(_r("p", below=True))}
    by_lead = {"f1": _f(_r("p", lead=True))}
    iso = lock((CL1, H))
    assert _one(by_below)["evidence_level"] == _one(by_lead)["evidence_level"] == "5b"
    assert _one(by_below, iso=iso)["evidence_level"] == "5b" and not bool(_one(by_below, iso=iso)["lead_lift"])
    assert _one(by_lead, iso=iso)["evidence_level"] == "4b"


def test_a_below_row_in_any_file_outranks_the_lift():
    files = {"f1": _f(_r("p", lead=True)), "f2": _f(_r("q", below=True))}
    r = _one(files, iso=lock((CL1, H)))
    assert r["evidence_level"] == "5b" and not bool(r["lead_lift"]) and bool(r["lead"])
    # a lead on one file of many is enough to be hard, and enough to lift (the pooled lead is any row)
    files = {"f1": _f(_r("p", lead=True)), "f2": _f(_r("q")), "f3": _f(_r("s"))}
    assert _one(files)["evidence_level"] == "5b"
    assert _one(files, iso=lock((CL1, H)))["evidence_level"] == "4b"


# =========================================================================== the axes the flag gave
def test_an_anchor_only_on_a_flagged_row_goes_with_the_flag():
    files = {"f1": _f(_r("p", lead=True, anchor="a1")), "f2": _f(_r("q"))}
    assert "anchor" in _one(files)["evidence_axes"]
    r = _one(files, iso=lock((CL1, H)))
    assert not bool(r["anchor"]) and r["level_reason"] == "4b: one corroboration (iso)"
    # an anchor on an unflagged row of the same pair stays: two axes
    files = {"f1": _f(_r("p", lead=True, anchor="a1")), "f2": _f(_r("q", anchor="a2"))}
    r = _one(files, iso=lock((CL1, H)))
    assert bool(r["anchor"]) and r["n_axes"] == 2 and r["level_reason"] == \
        "4b: iso + anchor, none outside the channel's chemistry"


def test_an_alien_sibling_gives_the_lifted_pair_nothing():
    files = {"f1": _f(_r("p", lead=True), _r("s", adduct=NO3))}
    label = {"untie": set(), "veto": {}, "alien": {(CL1, NO3)}}
    r = _one(files, iso=lock((CL1, H)), label=label)
    assert bool(r["lead_lift"]) and not bool(r["chan2"]) and not bool(r["branch"]) and r["evidence_level"] == "4b"
    iso = lock((CL1, H))
    iso["veto"] = {(CL1, NO3): "HIGH: h"}
    r = _one(files, iso=iso)
    assert bool(r["lead_lift"]) and not bool(r["branch"]) and r["evidence_level"] == "4b"


def test_what_the_card_does_not_drop_stays():
    """multiline, carbon and the curated family are not the flag's: a lifted pair
    keeps them (a lifted pair with two in-band lines of the neutral's own elements
    stays multiline)."""
    rows = [_r("p", lead=True), _child("c13", "p", "13C", 1000.0 * 6 * EV.C13_PER_CARBON),
            _child("c37", "p", "37Cl", 1000.0 * EV.ISOTOPE_ABUNDANCE["37Cl"])]
    files = {"f1": _f(*rows)}
    base = _one(files)
    assert bool(base["multiline"]) and bool(base["carbon_ev"]) and base["multiline_elements"] == "C|Cl"
    r = _one(files, iso=lock((CL1, H)))
    assert bool(r["lead_lift"]) and bool(r["multiline"]) and bool(r["carbon_ev"])
    assert r["multiline_elements"] == "C|Cl" and r["evidence_axes"] == "iso|multiline|carbon|lead_lift|files:1"


def test_a_second_channel_only_from_flagged_rows_goes_with_the_flag():
    flagged_sib = {"f1": _f(_r("p", lead=True), _r("s", adduct=NO3, below=True))}
    r = _one(flagged_sib, iso=lock((CL1, H)))
    # the sibling's below row: the branch and chan2 it gave the pair go (4b, not 3b)
    assert not bool(r["chan2"]) and not bool(r["branch"]) and r["evidence_level"] == "4b"
    lead_sib = {"f1": _f(_r("p", lead=True), _r("s", adduct=NO3, lead=True))}
    r = _one(lead_sib, iso=lock((CL1, H)))
    assert not bool(r["chan2"]) and not bool(r["branch"]) and r["evidence_level"] == "4b"
    # one unflagged row of the sibling keeps them: the acid branch, 3b
    clean_sib = {"f1": _f(_r("p", lead=True), _r("s", adduct=NO3, lead=True)), "f2": _f(_r("t", adduct=NO3))}
    r = _one(clean_sib, iso=lock((CL1, H)))
    assert bool(r["chan2"]) and bool(r["branch"]) and r["evidence_level"] == "3b" and bool(r["lead_lift"])


def test_the_label_fold_applies_to_the_lifted_pairs_second_channel():
    """On a labelled-nitrate batch the 14N and 15N nitrate clusters of one neutral
    are one channel: a lifted [M+^NO3]- takes no second channel from its own 14N
    twin."""
    files = {"f1": _f(_r("p", adduct=NO3L, lead=True), _r("s", adduct=NO3))}
    label = {"untie": set(), "veto": {}, "alien": set()}
    r = _one(files, key=(CL1, NO3L), iso=lock((CL1, NO3L)), label=label)
    assert bool(r["lead_lift"]) and not bool(r["chan2"]) and r["evidence_level"] == "4b"
    r = _one(files, key=(CL1, NO3L), iso=lock((CL1, NO3L)))
    assert bool(r["lead_lift"]) and bool(r["chan2"])                   # no label table: two channels


def test_no_other_pair_moves():
    """The full pairs frame with and without the lock: equal row for row but the
    lifted pair -- the sibling that took chan2 from the lifted pair's flagged
    rows keeps it, and the pair's own flagged rows still count for it."""
    files = {"f1": _f(_r("p", lead=True, anchor="a1"), _r("s", adduct=NO3), _r("x", "C7H12O4", anchor="z"),
                      _r("y", "C8H14O4", lead=True), _r("c", "C5H7ClO3"), _r("d", "C5H7ClO3", adduct=NO3, lead=True)),
             "f2": _f(_r("q", "C7H12O4", adduct=NO3))}
    base = EV.level_pooled(files)
    # a lock on a pair no file holds, and on a locked pair that is no lead (its flagged sibling still gives it
    # chan2 and the branch): nothing
    got = EV.level_pooled(files, iso=lock((CL1, H), ("C9H9ClO3", H), ("C5H7ClO3", H)))
    lifted = (got["neutral_formula"] == CL1) & (got["adduct"] == H)
    assert got.loc[lifted, "lead_lift"].all() and not got.loc[~lifted, "lead_lift"].any()
    pd.testing.assert_frame_equal(base[~lifted.values].reset_index(drop=True), got[~lifted].reset_index(drop=True))
    sib = got[(got["neutral_formula"] == CL1) & (got["adduct"] == NO3)].iloc[0]
    assert bool(sib["chan2"]) and bool(sib["branch"])                   # from the lifted pair's lead row
    me = got[lifted].iloc[0]
    assert bool(me["chan2"]) and bool(me["branch"])                     # its sibling is unflagged: kept


# =========================================================================== the vetoes outrank it
def test_an_isotope_veto_outranks_the_lift():
    files = {"f1": _f(_r("p", lead=True))}
    iso = lock((CL1, H))
    iso["veto"] = {(CL1, H): "REQ: r"}
    r = _one(files, iso=iso)
    assert r["evidence_level"] == "5b" and not bool(r["lead_lift"]) and bool(r["lead"])
    assert r["level_reason"] == "5b: below assignability; an isotope check refutes the formula (REQ: r)"


def test_rule_k_outranks_the_lift():
    files = {"f1": _f(_r("p", adduct=NO3L, lead=True))}
    label = {"untie": set(), "veto": {(CL1, NO3L): "the 14N twin is absent"}, "alien": set()}
    r = _one(files, key=(CL1, NO3L), iso=lock((CL1, NO3L)), label=label)
    assert r["evidence_level"] == "5b" and not bool(r["lead_lift"])
    assert "below assignability" in r["level_reason"] and "lead_lift" not in r["evidence_axes"]
    files = {"f1": _f(_r("p", adduct=NO3, lead=True))}
    label = {"untie": set(), "veto": {}, "alien": {(CL1, NO3)}}
    r = _one(files, key=(CL1, NO3), iso=lock((CL1, NO3)), label=label)
    assert r["evidence_level"] == "5b" and not bool(r["lead_lift"])


def test_an_ion_only_pair_is_never_lifted():
    row = _r("io", adduct="[M]-.", lead=True, mz=C.ion_mz(CL1, H) + 1.00728)
    row["method"] = "ion_only:ea"
    r = _one({"f1": _f(row)}, key=(CL1, "[M]-."), iso=lock((CL1, "[M]-.")))
    assert not bool(r["lead_lift"]) and r["evidence_level"] == "5b"


def test_the_other_hard_inputs_still_hold_but_the_lead_reason_goes():
    files = {"f1": _f(_r("p", lead=True, tied=True))}
    assert _one(files)["level_reason"] == "5b: near-tie broken by the arbiter; below assignability"
    r = _one(files, iso=lock((CL1, H)))
    assert (r["evidence_level"], r["level_reason"]) == ("5b", "5b: near-tie broken by the arbiter")
    assert bool(r["lead_lift"])
    files = {"f1": _f(_r("p", lead=True, confidence="Low (0.3)"))}
    r = _one(files, iso=lock((CL1, H)))
    assert (r["evidence_level"], r["level_reason"]) == ("5b", "5b: engine confidence Low/Suspect")


# =========================================================================== which leads lift
@pytest.mark.parametrize("lead_by, budget_ok, lifts", [
    ("spec_gapfill", False, True), ("spec_n3", False, True), ("spec_minor", False, True),
    ("reflist_dim", False, True),
    ("off_budget", True, True), ("off_budget", False, False),     # the CF2 analogue: Cl the only violation
    ("", True, True), ("", False, False),                          # no code (a ledger before lead_by): maybe budget
    ("radical_anion", True, False), ("reagent_n", True, False),     # never
    ("off_budget|spec_gapfill", True, True), ("off_budget|spec_gapfill", False, False),
    ("spec_gapfill|radical_anion", True, False), ("some_new_setter", True, False),
])
def test_the_setters_a_lock_answers(lead_by, budget_ok, lifts):
    assert EV.lead_liftable(EV.lead_setters(lead_by), budget_ok) is lifts
    files = {"f1": _f(_r("p", lead=True, lead_by=lead_by))}
    r = _one(files, iso=lock((CL1, H), budget_ok=budget_ok))
    assert bool(r["lead_lift"]) is lifts and r["evidence_level"] == ("4b" if lifts else "5b")


def test_every_lead_row_must_be_liftable_and_a_ledger_without_the_column_is_conservative():
    files = {"f1": _f(_r("p", lead=True, lead_by="spec_gapfill")), "f2": _f(_r("q", lead=True, lead_by="off_budget"))}
    assert not bool(_one(files, iso=lock((CL1, H), budget_ok=False))["lead_lift"])
    assert bool(_one(files, iso=lock((CL1, H), budget_ok=True))["lead_lift"])
    old = {"f1": _f(_r("p", lead=True)).drop(columns=["lead_by"])}
    assert not bool(_one(old, iso=lock((CL1, H), budget_ok=False))["lead_lift"])
    assert bool(_one(old, iso=lock((CL1, H), budget_ok=True))["lead_lift"])
    # an unflagged row with a stray code does not block anything
    files = {"f1": _f(_r("p", lead=True)), "f2": _f(_r("q", lead_by="radical_anion"))}
    assert bool(_one(files, iso=lock((CL1, H)))["lead_lift"])


def test_a_lifted_pair_takes_no_reagent_only_flag():
    """A Cl acid's [M-H]- on a bromide channel whose only per-file satellite is
    tagged 81Br (the TOF cannot tell it from the 37Cl line): without the lift
    the reagent-only flag would put it at 4d; the lock is a line of its own."""
    rows = [_r("p", lead=True), _child("c", "p", "81Br+2", 320.0),
            _r("b1", "C9H14O4", adduct="[M+Br]-"), _r("b2", "C8H12O4", adduct="[M+Br]-"),
            _r("b3", "C10H16O4", adduct="[M+Br]-")]
    files = {"f1": _f(*rows)}
    base = _one(files)
    assert bool(base["reagent_only_iso"]) and base["reagent_halogen"] == "Br"
    r = _one(files, iso=lock((CL1, H)))
    assert not bool(r["reagent_only_iso"]) and r["evidence_level"] == "4b" and "reagent_only_iso" not in \
        r["evidence_axes"]


# =========================================================================== never per file, never a leak
def test_per_file_levels_and_the_empty_facts_never_move():
    files = {"f1": _f(_r("p", lead=True, anchor="a"), _r("s", adduct=NO3), _child("c", "s", "13C", 70.0))}
    base = EV.level_pooled(files)
    assert {"lead_lift", "lock_note"} <= set(base.columns) and not base["lead_lift"].any()
    assert (base["lock_note"] == "").all()
    for iso in (None, {"veto": {}}, {"veto": {}, "lock": {}}):
        assert base.equals(EV.level_pooled(files, iso=iso)), iso
    for fn in (EV.compute_levels, EV.apply_levels, EV.source_neutrals, EV.corroborating_neutrals):
        assert "iso" not in inspect.signature(fn).parameters
    led = files["f1"]
    pd.testing.assert_frame_equal(EV.compute_levels(led), EV.compute_levels(led.drop(columns=["lead_by"])))
    # a lock keyed on the pair, held by a source: its neutral never enters another source's cross set
    assert EV.source_neutrals(files) == EV.source_neutrals({"f1": led.assign(lead_by="")})


def test_no_upair_neutral_carries_a_halogen():
    """Rule U's neutral pair is N-free CHO by construction and a lock needs a Cl / Br
    ion: a lifted pair never meets row 9' (which would read iso + upair as 4a)."""
    from tests.test_evidence import FIXTURES
    t = pd.read_csv(FIXTURES / "ur_neutral_pairs.csv")
    held = t.loc[t["upair"].astype(bool), "neutral_formula"]
    assert len(held) and not any(C.parse_formula(n).get(el, 0) for n in held for el in ("Cl", "Br"))


# =========================================================================== the reference script
def _run_dir(tmp_path: Path, name: str, frame: pd.DataFrame, table: pd.DataFrame | None) -> Path:
    run = tmp_path / name
    (run / "per_file").mkdir(parents=True)
    (run / "tables").mkdir()
    frame.rename(columns={"occurrence": "occurrence_y"}).to_csv(run / "per_file" / "s1_ledger.csv", index=False)
    if table is not None:
        table.to_csv(run / "tables" / "iso_checks.csv", index=False)
    return run


def _h_table(rows: list[dict]) -> pd.DataFrame:
    t = pd.DataFrame(rows)
    for c in IC.TABLE_COLUMNS:
        if c not in t.columns:
            t[c] = np.nan
    return t[list(IC.TABLE_COLUMNS)]


def _h_row(n, a, *, lock=True, budget_ok=True, veto=False, check="H", note=NOTE):
    return dict(neutral_formula=n, adduct=a, check=check, verdict="lock" if lock else "no_lock", veto=veto,
                lock=lock, element="Cl", n_halogen=1, budget_ok=budget_ok, note=note)


def _mixed_frame() -> pd.DataFrame:
    return _f(_r("p", lead=True, anchor="a1"), _r("s", adduct=NO3, below=True),                # lifts: 4b
              _r("p2", CL2, lead=True, lead_by="off_budget"), _r("s2", CL2, adduct=NO3),   # budget gate
              _r("p3", "C5H9ClO4", lead=True), _r("s3", "C5H9ClO4", adduct=NO3, below=True),  # vetoed
              _r("p4", "C7H11ClO5", lead=True, lead_by="reagent_n"),                        # never
              _r("p5", "C4H5ClO3", lead=True), _r("s5", "C4H5ClO3", adduct=NO3, lead=True),  # lifts: its
              _r("t5", "C4H5ClO3", adduct=NO3),                                            # sibling's clean row: 3b
              _r("p6", "C5H7ClO3"), _r("s6", "C5H7ClO3", adduct=NO3, lead=True),           # locked, no lead
              _r("x", "C7H12O4", anchor="z"))


def test_the_reference_script_lifts_like_the_engine(tmp_path):
    frame = _mixed_frame()
    table = _h_table([_h_row(CL1, H), _h_row(CL2, H, budget_ok=False), _h_row(CL2, NO3, lock=False),
                      _h_row("C5H9ClO4", H), _h_row("C5H9ClO4", H, check="REQ", lock=False, veto=True, note="r"),
                      _h_row("C7H11ClO5", H), _h_row("C4H5ClO3", H), _h_row("C5H7ClO3", H)])
    run = _run_dir(tmp_path, "RUN", frame, table)
    facts = LL.iso_check_facts(str(run))
    assert facts == IC.facts(table) and set(facts["lock"]) == {(CL1, H), (CL2, H), ("C5H9ClO4", H), ("C7H11ClO5", H),
                                                              ("C4H5ClO3", H), ("C5H7ClO3", H)}
    ref = LL.run([str(run)], [], None, None, "auto")
    core = EV.level_pooled({"s1": frame}, iso=IC.facts(table))
    m = core.merge(ref, left_on=["neutral_formula", "adduct"], right_on=["neutral", "adduct"], suffixes=("", "_ll"))
    assert len(m) == len(core) == len(ref)
    for c_core, c_ll in (("evidence_level", "level"), ("chan2", "chan2_ll"), ("anchor", "anchor_ll"),
                         ("branch", "branch_ll"), ("iso", "iso_ll"), ("lead", "lead_ll"),
                         ("lead_lift", "lead_lift_ll"), ("reagent_only_iso", "reagent_only_iso_ll")):
        assert (m[c_core].astype(str) == m[c_ll].astype(str)).all(), c_core
    lv = dict(zip(zip(m["neutral_formula"], m["adduct"]), m["evidence_level"]))
    lifted = set(zip(m.loc[m["lead_lift"], "neutral_formula"], m.loc[m["lead_lift"], "adduct"]))
    assert lifted == {(CL1, H), ("C4H5ClO3", H)} and lv[(CL1, H)] == "4b" and lv[("C4H5ClO3", H)] == "3b"
    assert lv[(CL2, H)] == lv[("C5H9ClO4", H)] == lv[("C7H11ClO5", H)] == "5b"
    assert lv[("C5H7ClO3", H)] == "3b"                                  # locked, no lead: its flagged sibling counts
    # the lock-only named-table guard: a table locking a pair this source does not hold is another batch's
    other = _run_dir(tmp_path, "OTHER", _f(_r("q", "C7H12O4", anchor="z")), None)
    lone = tmp_path / "lock_only.csv"
    _h_table([_h_row(CL1, H)]).to_csv(lone, index=False)
    import contextlib
    import io
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        got = LL.run([str(other)], [], None, None, str(lone))
    assert "does not hold 1 pair(s) the table vetoes or locks" in err.getvalue()
    assert not got["lead_lift"].any()


def _lockstep(tmp_path, name, frame, table, label_table=None):
    """The engine and the reference script on one run dir: every level input the
    lift touches, row for row."""
    from peaky.batch import label_twins as LT
    run = _run_dir(tmp_path, name, frame, table)
    if label_table is not None:
        label_table.to_csv(run / "tables" / "label_twins.csv", index=False)
    ref = LL.run([str(run)], [], None, "auto" if label_table is not None else None, "auto")
    core = EV.level_pooled({"s1": frame}, iso=IC.facts(table),
                           label=LT.facts(label_table) if label_table is not None else None)
    m = core.merge(ref, left_on=["neutral_formula", "adduct"], right_on=["neutral", "adduct"], suffixes=("", "_ll"))
    assert len(m) == len(core) == len(ref)
    for c in ("chan2", "anchor", "branch", "iso", "lead", "lead_lift", "reagent_only_iso", "multiline", "carbon_ev"):
        assert (m[c].astype(bool) == m[c + "_ll"].astype(bool)).all(), c
    assert (m["evidence_level"] == m["level"]).all()
    return m.set_index(["neutral_formula", "adduct"])


def test_the_reference_script_lifts_like_the_engine_on_a_labelled_batch(tmp_path):
    """Rule K's fold, veto and alien lines, a below row beside a lead, an ion-only
    lead: the reference script levels each like the engine."""
    io_row = _r("i5", "C5H9ClO4", adduct="[M]-.", lead=True, mz=C.ion_mz("C5H9ClO4", H) + 1.00728)
    io_row["method"] = "ion_only:ea"
    frame = _f(_r("a1", CL1, NO3L, lead=True), _r("b1", CL1, NO3),                   # lifted, one channel: 4b
               _child("a1c", "a1", "13C", 1000.0 * 6 * EV.C13_PER_CARBON),            # ... keeps multiline
               _child("a1l", "a1", "37Cl", 1000.0 * EV.ISOTOPE_ABUNDANCE["37Cl"]),
               _r("a2", "C7H11ClO5", lead=True), _r("b2", "C7H11ClO5", NO3),         # lifted, the sibling alien
               _r("a3", CL2, NO3L, lead=True),                                       # rule K refutes: 5b
               _r("a4", "C3H5ClO3", lead=True), _r("b4", "C3H5ClO3", below=True),    # a below row: 5b
               io_row,                                                               # ion-only: 5b
               _r("x", "C7H12O4", anchor="z"))
    table = _h_table([_h_row(CL1, NO3L), _h_row("C7H11ClO5", H), _h_row(CL2, NO3L), _h_row("C3H5ClO3", H),
                      _h_row("C5H9ClO4", "[M]-.")])
    label = pd.DataFrame({"neutral_formula": ["C7H11ClO5", CL2], "adduct": [NO3, NO3L], "untie": [False, False],
                          "veto": [False, True], "alien": [True, False], "note": ["", "the 14N twin is absent"]})
    m = _lockstep(tmp_path, "LAB", frame, table, label)
    assert set(m.index[m["lead_lift"]]) == {(CL1, NO3L), ("C7H11ClO5", H)}
    assert m.loc[(CL1, NO3L), "evidence_level"] == m.loc[("C7H11ClO5", H), "evidence_level"] == "4b"
    assert not m.loc[(CL1, NO3L), "chan2"] and not m.loc[("C7H11ClO5", H), "branch"]
    assert m.loc[(CL1, NO3L), "multiline"] and m.loc[(CL1, NO3L), "evidence_level"] == "4b"
    assert (m.loc[[(CL2, NO3L), ("C3H5ClO3", H), ("C5H9ClO4", "[M]-.")], "evidence_level"] == "5b").all()


def test_the_reference_script_clears_the_reagent_only_flag_like_the_engine(tmp_path):
    frame = _f(_r("p", lead=True), _child("c", "p", "81Br+2", 320.0),
               _r("b1", "C9H14O4", adduct="[M+Br]-"), _r("b2", "C8H12O4", adduct="[M+Br]-"),
               _r("b3", "C10H16O4", adduct="[M+Br]-"))
    m = _lockstep(tmp_path, "BR", frame, _h_table([_h_row(CL1, H)]))
    r = m.loc[(CL1, H)]
    assert bool(r["lead_lift"]) and not bool(r["reagent_only_iso"]) and r["evidence_level"] == "4b"


def test_the_reference_script_without_lead_by_is_conservative_like_the_engine(tmp_path):
    frame = _f(_r("p", lead=True)).drop(columns=["lead_by"])
    for ok in (True, False):
        run = _run_dir(tmp_path, f"RUN_{ok}", frame, _h_table([_h_row(CL1, H, budget_ok=ok)]))
        ref = LL.run([str(run)], [], None, None, "auto").set_index(["neutral", "adduct"])
        core = _one({"s1": frame}, iso=IC.facts(pd.read_csv(run / "tables" / "iso_checks.csv")))
        assert bool(ref.at[(CL1, H), "lead_lift"]) is ok is bool(core["lead_lift"])
        assert ref.at[(CL1, H), "level"] == core["evidence_level"] == ("4b" if ok else "5b")


def test_the_scorecard_reads_the_lift_as_the_runs_own_evidence(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("scorecard", ROOT / "scripts" / "scorecard.py")
    SC = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "scorecard", SC)
    spec.loader.exec_module(SC)
    frame = _f(_r("p", lead=True))
    run_dir = _run_dir(tmp_path, "RUN", frame, _h_table([_h_row(CL1, H)]))
    run = types.SimpleNamespace(path=str(run_dir), per_file=frame.assign(__file="s1"))
    own = SC.own_levels_for(run).set_index(["neutral", "adduct"])
    assert own.at[(CL1, H), "level"] == "4b" and "lead_lift" in own.at[(CL1, H), "axes"]


# =========================================================================== the batch, end to end
def _series_rows():
    from tests.test_iso_checks import N, REFS, _c13, _filler, _row
    rows = []
    for i in range(N):
        rows += _filler(i)
        for k, n in enumerate(REFS):
            rows += _c13(i, n, H, phase=0.7 * k)
        acid = _c13(i, CL1, H, phase=2.0)                                   # the acid and its 13C line
        rows += acid
        rows.append(_row(i, acid[0]["mz"] + IC.LOCK_D["Cl"], 0.3198 * acid[0]["height"]))   # its 37Cl line
    return rows, REFS


def test_a_batch_lifts_the_locked_lead_on_the_merged_ledger(tmp_path, monkeypatch):
    """assign_batch.run: a chlorinated acid every file commits as a speculative
    gap-fill lead, whose 37Cl line the batch's series carries: H locks it, the
    pooled pair and the merged row read 4b with lead_lift, the per-file rows keep
    their lead, and the summary counts the lift."""
    from peaky.assignment import assign as A
    from peaky.assignment import ledger as L
    from peaky.assignment import tiers as TT
    from peaky.batch import assign_batch as AB
    from peaky.io import io_mascope as IO
    from tests.test_iso_checks import _ion

    rows, refs = _series_rows()
    commits = [(n, H) for n in refs] + [(CL1, H)]

    def fake_assign(sid, context="ambient-air", **kw):
        ids = [f"P{j}" for j in range(len(commits))]
        led = L.new_ledger(pd.DataFrame({"peak_id": ids, "mz": [C.ion_mz(n, a) for n, a in commits],
                                         "height": [5000.0] * len(ids)}))
        for pid, (n, a) in zip(ids, commits):
            L.commit_assignment(led, pid, neutral_formula=n, adduct=a, ion_formula=_ion(n, a), ion_score=0.95,
                                compound_score=0.95, ppm_error=0.1, pass_no=1, method="cheminfo", confidence="High",
                                commentary=f"Pass 1: {n} {a}")
        TT.apply_tiers(led)
        led["degeneracy_density"] = 0.5
        led["resolvability"] = "resolved"
        L.mark_lead(led, led.index[led["neutral_formula"] == CL1][0], "spec_gapfill")
        return {"ledger": led, "stats": {"noise_edge_cps": 10.0, "height_gate_cps": 10.0},
                "plausibility_audit": [], "summaries": {}, "problems": []}

    monkeypatch.setattr(IO, "connect", lambda *a, **k: "CLIENT")
    monkeypatch.setattr(IO, "fetch_peaks", lambda client, sid, use_cache=True: pd.DataFrame(
        {"peak_id": ["A"], "mz": [200.1], "height": [1.0e5]}))
    monkeypatch.setattr(IO, "estimate_offset", lambda raw: 0.0)
    monkeypatch.setattr(A, "run", fake_assign)
    pk = pd.DataFrame(rows)[["sample_item_id", "datetime_utc", "mz", "height", "area"]]
    pk["sample_item_name"] = "n_" + pk["sample_item_id"]
    AB.run(peaks=pk, ts_peaks=pk, reagent="NO3", batch="test batch", out_dir=str(tmp_path), k_min=2, k_max=3,
           min_gain=0.0, n_jobs=1, resolving_power=100_000, log=lambda *a: None)
    table = pd.read_csv(tmp_path / "tables" / "iso_checks.csv")
    h = table[table["check"] == "H"].set_index(["neutral_formula", "adduct"])
    assert h.at[(CL1, H), "verdict"] == "lock" and bool(h.at[(CL1, H), "budget_ok"])
    ev = pd.read_csv(tmp_path / "tables" / "evidence_levels.csv").set_index(["neutral_formula", "adduct"])
    assert bool(ev.at[(CL1, H), "lead_lift"]) and ev.at[(CL1, H), "evidence_level"] == "4b"
    assert ev.at[(CL1, H), "lock_note"] == h.at[(CL1, H), "note"]
    merged = pd.read_csv(tmp_path / "merged_ledger.csv").set_index(["neutral_formula", "adduct"])
    assert merged.at[(CL1, H), "evidence_level"] == "4b" and "lead_lift" in merged.at[(CL1, H), "evidence_axes"]
    summ = json.load(open(tmp_path / "batch_summary.json"))["evidence_levels"]
    assert summ["lead_lifted"] == 1 and summ["iso_checks"]["locked_pairs"] == 1
    # the per-file rows keep their flag, and their own level reads it
    files = sorted((tmp_path / "per_file").glob("*_ledger.csv"))
    assert files
    for f in files:
        led = pd.read_csv(f)
        row = led[(led["neutral_formula"] == CL1) & (led["role"] == "M0")].iloc[0]
        assert bool(row["tentative_lead"]) and row["lead_by"] == "spec_gapfill"
        lv = EV.compute_levels(led)
        assert lv.loc[row.name, "evidence_level"] == "5b"


def test_a_table_written_before_rule_h_levels_as_before():
    """An iso_checks.csv from before rule H (no `lock` column, no H rows) gives
    no lock: the levels are those of its vetoes alone, in both readers."""
    files = {"f1": _f(_r("p", lead=True), _r("s", adduct=NO3), _r("x", "C7H12O4", anchor="z"))}
    t = _h_table([dict(neutral_formula="C7H12O4", adduct=H, check="REQ", verdict="absent", veto=True, note="r")])
    old = t.drop(columns=["lock", "element", "n_halogen", "ratio_lo", "ratio_hi", "heavy_cl", "heavy_br",
                          "budget_ok", "budget_why"])
    facts = IC.facts(old)
    assert facts == {"veto": {("C7H12O4", H): "REQ: r"}, "lock": {}}
    got = EV.level_pooled(files, iso=facts)
    assert got.equals(EV.level_pooled(files, iso={"veto": facts["veto"]})) and not got["lead_lift"].any()
    assert got.set_index(["neutral_formula", "adduct"]).at[(CL1, H), "evidence_level"] == "5b"
