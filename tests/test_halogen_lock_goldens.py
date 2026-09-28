"""C11+b rule H on the golden sets (decision D8, 2026-09-28; tests/fixtures/levels/README.md).

The fixtures predate C19(c): their leads sat in below_assignability, which rule H
never lifts. D8 moved the flag to `tentative_lead` (with the setter its source
row's note names in `lead_by`) on the four pairs the live runs lock -- the
Orbitrap set's HBr [M+^NO3]-, C6H9ClO3 and C6H10Cl2O4 [M-H]- (speculative
series gap-fills), the uronium set's C7H11ClO2 [M+(CH4N2O)H]+ (outside the
uronium element budget, Cl its only violation) -- and added each set's lock
table: the rule H lock rows of the live run's tables/iso_checks.csv for the pairs
the set holds. Levelled with its table each of the four lifts 5b -> 4b and
nothing else moves; without it, or with the flag back in below, the vectors are
the pre-C11+b ones. The engine and the reference script agree row for row.

Run: pytest tests/test_halogen_lock_goldens.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from peaky.assignment import evidence as EV
from peaky.batch import iso_checks as IC
from peaky.batch import label_twins as LT
from tests.test_evidence import (
    FIXTURES, GOLDEN, ORBI_NO_LOCK, UR_NO_LOCK, UR_WITHOUT_PAIR, _iso, _pooled, _ur_pairs, _vector)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import level_ledger as LL  # noqa: E402

K = ["neutral_formula", "adduct"]
D8 = {"orbi": {("HBr", "[M+^NO3]-"): "spec_gapfill", ("C6H9ClO3", "[M-H]-"): "spec_gapfill",
               ("C6H10Cl2O4", "[M-H]-"): "spec_gapfill"},
      "ur": {("C7H11ClO2", "[M+(CH4N2O)H]+"): "off_budget"}}
#: the Orbitrap set with rule K's table and its lock table
ORBI_KH = (1707, "0/11/174/9/209/151/0/38/1115")


def _key(frame: pd.DataFrame) -> list:
    return list(zip(frame["neutral_formula"].astype(str), frame["adduct"].astype(str)))


def test_the_d8_edit_moved_the_flag_of_the_four_live_locked_leads_only():
    for prefix, pairs in D8.items():
        seen = set()
        for name, f in _pooled(prefix).items():
            assert {"tentative_lead", "lead_by"} <= set(f.columns), name
            lead = f["tentative_lead"].map(EV.truthy)
            below = f["below_assignability"].map(EV.truthy)
            assert not (lead & below).any()
            rows = f[lead]
            assert (rows["role"] == "M0").all()
            for k, by in zip(_key(rows), rows["lead_by"]):
                assert pairs[k] == by, (name, k)
                seen.add(k)
            assert f.loc[~lead, "lead_by"].isna().all()
        assert seen == set(pairs)
    for prefix in ("tv_nitrate", "tv_bromide"):
        assert "tentative_lead" not in pd.read_csv(FIXTURES / f"{prefix}_ledger.csv.gz", nrows=1).columns
    assert all("tentative_lead" not in f.columns for f in _pooled("tof").values())


def test_the_lock_tables_are_lock_rows_of_pairs_the_sets_hold():
    for prefix, pairs in D8.items():
        t = pd.read_csv(FIXTURES / f"{prefix}_iso_checks.csv")
        assert list(t.columns) == list(IC.TABLE_COLUMNS)
        assert (t["check"] == "H").all() and t["lock"].all() and not t["veto"].any()
        assert (t["verdict"] == "lock").all() and (t["instrument"] == "orbitrap").all()
        held = set().union(*[set(_key(f[f["role"] == "M0"])) for f in _pooled(prefix).values()])
        assert set(_key(t)) <= held and set(pairs) <= set(_key(t))
        assert set(_iso(prefix)["lock"]) == set(_key(t)) and _iso(prefix)["veto"] == {}
    ur = _iso("ur")["lock"][("C7H11ClO2", "[M+(CH4N2O)H]+")]
    assert ur["budget_ok"] and ur["element"] == "Cl" and ur["n"] == 1
    t = pd.read_csv(FIXTURES / "ur_iso_checks.csv")
    assert t["budget_why"].iloc[0] == "Cl=1 > 0"


def _moves(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    m = a.merge(b, on=K, suffixes=("_a", "_b"))
    assert len(m) == len(a) == len(b)
    return m[(m["evidence_level_a"] != m["evidence_level_b"]) | (m["level_reason_a"] != m["level_reason_b"])
             | (m["evidence_axes_a"] != m["evidence_axes_b"])]


def test_the_lock_table_lifts_the_four_and_moves_nothing_else():
    tof, orbi, ur = _pooled("tof"), _pooled("orbi"), _pooled("ur")
    cross = EV.source_neutrals(tof)
    for name, before, after in (
            ("orbi", EV.level_pooled(orbi, cross=cross), EV.level_pooled(orbi, cross=cross, iso=_iso("orbi"))),
            ("ur", EV.level_pooled(ur, upair=_ur_pairs()), EV.level_pooled(ur, upair=_ur_pairs(), iso=_iso("ur")))):
        d = _moves(before, after)
        assert set(_key(d)) == set(D8[name]) == set(_key(after[after["lead_lift"]])), name
        assert (d["evidence_level_a"] == "5b").all() and (d["evidence_level_b"] == "4b").all()
        assert (d["level_reason_b"] == "4b: one corroboration (iso)").all()
        assert d["evidence_axes_b"].str.contains("lead_lift").all()
    assert _vector(EV.level_pooled(orbi, cross=cross, iso=_iso("orbi")).evidence_level) == GOLDEN["orbi"][1]
    assert _vector(EV.level_pooled(orbi, cross=cross).evidence_level) == ORBI_NO_LOCK[1]
    assert _vector(EV.level_pooled(ur, upair=_ur_pairs()).evidence_level) == UR_NO_LOCK
    assert _vector(EV.level_pooled(ur).evidence_level) == UR_WITHOUT_PAIR


def test_the_flag_back_in_below_and_the_lock_move_nothing():
    """The fixtures as they were (the lead in below_assignability) with the lock
    table: the golden before C11+b -- the lift never touches below."""
    tof = _pooled("tof")
    old = {}
    for name, f in _pooled("orbi").items():
        f = f.copy()
        lead = f["tentative_lead"].map(EV.truthy)
        f.loc[lead, "below_assignability"] = True
        old[name] = f.drop(columns=["tentative_lead", "lead_by"])
    out = EV.level_pooled(old, cross=EV.source_neutrals(tof), iso=_iso("orbi"))
    assert (len(out), _vector(out.evidence_level)) == ORBI_NO_LOCK and not out["lead_lift"].any()


def test_with_rule_ks_table_the_lock_lifts_the_same_three():
    tof, orbi = _pooled("tof"), _pooled("orbi")
    cross = EV.source_neutrals(tof)
    label = LT.facts(pd.read_csv(FIXTURES / "orbi_label_twins.csv"))
    k = EV.level_pooled(orbi, cross=cross, label=label)
    kh = EV.level_pooled(orbi, cross=cross, label=label, iso=_iso("orbi"))
    assert (len(kh), _vector(kh.evidence_level)) == ORBI_KH
    assert set(_key(_moves(k, kh))) == set(D8["orbi"])


def _unpack(tmp_path: Path, prefix: str) -> Path:
    d = tmp_path / prefix / "per_file"
    d.mkdir(parents=True)
    for p in sorted(FIXTURES.glob(f"{prefix}_*_ledger.csv.gz")):
        pd.read_csv(p, low_memory=False).to_csv(d / p.name[:-3], index=False)
    return d.parent


def test_the_reference_script_levels_the_golden_sets_with_their_lock_tables(tmp_path):
    """Live, beside expected_levels.csv: the reference script on the fixtures
    with --iso-checks agrees with the engine row for row, lifts included."""
    tof, orbi = _unpack(tmp_path, "tof"), _unpack(tmp_path, "orbi")
    ref = LL.run([str(orbi)], [str(tof)], None, None, str(FIXTURES / "orbi_iso_checks.csv"))
    core = EV.level_pooled(_pooled("orbi"), cross=EV.source_neutrals(_pooled("tof")), iso=_iso("orbi"))
    m = core.merge(ref, left_on=K, right_on=["neutral", "adduct"], suffixes=("", "_ll"))
    assert len(m) == len(core) == len(ref) == GOLDEN["orbi"][0]
    assert (m["evidence_level"] == m["level"]).all() and (m["lead_lift"] == m["lead_lift_ll"]).all()
    assert set(_key(m[m["lead_lift_ll"]])) == set(D8["orbi"])
    ur = _unpack(tmp_path, "ur")
    ref = LL.run([str(ur)], [], str(FIXTURES / "ur_neutral_pairs.csv"), None, str(FIXTURES / "ur_iso_checks.csv"))
    core = EV.level_pooled(_pooled("ur"), upair=_ur_pairs(), iso=_iso("ur"))
    m = core.merge(ref, left_on=K, right_on=["neutral", "adduct"], suffixes=("", "_ll"))
    assert len(m) == GOLDEN["ur"][0] and (m["evidence_level"] == m["level"]).all()
    assert set(_key(m[m["lead_lift_ll"]])) == set(D8["ur"])
    # the named-table guard: the Orbitrap set's table on the TOF set is another batch's
    ref = LL.run([str(tof)], [str(orbi)], None, None, str(FIXTURES / "orbi_iso_checks.csv"))
    assert not ref["lead_lift"].any() and _vector(ref["level"]) == GOLDEN["tof"][1]
