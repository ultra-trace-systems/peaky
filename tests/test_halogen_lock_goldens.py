"""C11+b rule H on the golden sets (decision D8, 2026-09-28; tests/fixtures/levels/README.md).

The fixtures predate C19(c): their leads sat in below_assignability, which rule H
never lifts. D8 moved the flag to `tentative_lead` (with the setter its source
row's note names in `lead_by`) on the four pairs the live runs lock -- the
Orbitrap set's HBr [M+^NO3]-, C6H9ClO3 and C6H10Cl2O4 [M-H]- (speculative
series gap-fills), the uronium set's C7H11ClO2 [M+(CH4N2O)H]+ (outside the
uronium element budget, Cl its only violation) -- and added each set's lock
table: the rule H lock rows of the live run's tables/iso_checks.csv for the pairs
the set holds. With its table each of the four is lifted (`lead_lift`, its
lead no longer a hard input: before peaky 0.10.0 that read 5b -> 4b) and no
other fact moves; without it, or with the flag back in below, the fact vectors
are the pre-C11+b ones. The pre-0.10.0 decision these levels came from is
private now and never sees a lock table (the merge vote reads each file alone),
so the tests pin the facts (tests/test_evidence.py `_vector`). The engine and
the reference script agree row for row while the script holds that decision.

Run: pytest tests/test_halogen_lock_goldens.py -q
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from peaky.assignment import evidence as EV
from peaky.batch import iso_checks as IC
from peaky.batch import label_twins as LT
from tests.test_evidence import (
    FACT_ORDER, FIXTURES, GOLDEN, ORBI_NO_LOCK, UR_NO_LOCK, UR_WITHOUT_PAIR, _iso, _pooled, _ur_pairs, _vector)
from tests.test_halogen_lock_lift import _hard, _ll

ROOT = Path(__file__).resolve().parents[1]

K = ["neutral_formula", "adduct"]
D8 = {"orbi": {("HBr", "[M+^NO3]-"): "spec_gapfill", ("C6H9ClO3", "[M-H]-"): "spec_gapfill",
               ("C6H10Cl2O4", "[M-H]-"): "spec_gapfill"},
      "ur": {("C7H11ClO2", "[M+(CH4N2O)H]+"): "off_budget"}}
#: the Orbitrap set's fact vector with rule K's table and its lock table (its level vector before peaky
#: 0.10.0: 0/10/174/9/208/150/0/38/1118), and with rule K's table alone
ORBI_KH = (1707, "238/39/218/393/176/391/0/0/411/890/0/3/14/706/91/0/0/24/0")
ORBI_K = "238/39/218/393/179/391/0/0/411/890/3/0/14/706/91/0/0/24/0"


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


#: the facts a pair can move by (FACT_ORDER + the counts and notes beside them)
MOVES = [*FACT_ORDER, "n_axes", "lock_note", "known_fam", "multiline_elements"]


def _moves(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    m = a.merge(b, on=K, suffixes=("_a", "_b"))
    assert len(m) == len(a) == len(b)
    moved = pd.Series(False, index=m.index)
    for c in MOVES:
        moved |= m[c + "_a"].astype(str) != m[c + "_b"].astype(str)
    return m[moved]


def test_the_lock_table_lifts_the_four_and_moves_nothing_else():
    tof, orbi, ur = _pooled("tof"), _pooled("orbi"), _pooled("ur")
    cross = EV._source_neutrals(tof)
    for name, before, after in (
            ("orbi", EV._series_pooled(orbi, cross=cross), EV._series_pooled(orbi, cross=cross, iso=_iso("orbi"))),
            ("ur", EV._series_pooled(ur, upair=_ur_pairs()), EV._series_pooled(ur, upair=_ur_pairs(), iso=_iso("ur")))):
        d = _moves(before, after)
        assert set(_key(d)) == set(D8[name]) == set(_key(after[after["lead_lift"]])), name
        assert d["lead_a"].all() and not d["lead_b"].any() and d["lead_lift_b"].all()
        lifted = after.set_index(K).loc[list(D8[name])]
        assert all(_hard(r) == () and bool(r["iso"]) for _k, r in lifted.iterrows())    # the lock: its one axis
        assert d["evidence_axes_b"].str.contains("lead_lift").all()
    assert _vector(EV._series_pooled(orbi, cross=cross, iso=_iso("orbi"))) == GOLDEN["orbi"][1]
    assert _vector(EV._series_pooled(orbi, cross=cross)) == ORBI_NO_LOCK[1]
    assert _vector(EV._series_pooled(ur, upair=_ur_pairs())) == UR_NO_LOCK
    assert _vector(EV._series_pooled(ur)) == UR_WITHOUT_PAIR


def test_the_flag_back_in_below_and_the_lock_move_nothing():
    """The fixtures as they were (the lead in below_assignability) with the lock
    table: the facts before C11+b -- the lift never touches below."""
    tof = _pooled("tof")
    old = {}
    for name, f in _pooled("orbi").items():
        f = f.copy()
        lead = f["tentative_lead"].map(EV.truthy)
        f.loc[lead, "below_assignability"] = True
        old[name] = f.drop(columns=["tentative_lead", "lead_by"])
    out = EV._series_pooled(old, cross=EV._source_neutrals(tof), iso=_iso("orbi"))
    assert not out["lead_lift"].any() and len(out) == ORBI_NO_LOCK[0]
    # the three leads read as below rows now: the lead count moves to below, nothing else
    base = EV._series_pooled(_pooled("orbi"), cross=EV._source_neutrals(tof))
    assert int(out["below"].sum()) == int(base["below"].sum()) + 3 and not out["lead"].any()
    assert _moves(base, out)[["neutral_formula", "adduct"]].apply(tuple, axis=1).isin(list(D8["orbi"])).all()


def test_with_rule_ks_table_the_lock_lifts_the_same_three():
    tof, orbi = _pooled("tof"), _pooled("orbi")
    cross = EV._source_neutrals(tof)
    label = LT.facts(pd.read_csv(FIXTURES / "orbi_label_twins.csv"))
    k = EV._series_pooled(orbi, cross=cross, label=label)
    kh = EV._series_pooled(orbi, cross=cross, label=label, iso=_iso("orbi"))
    assert (len(kh), _vector(kh)) == ORBI_KH and _vector(k) == ORBI_K
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
    LL = _ll()
    tof, orbi = _unpack(tmp_path, "tof"), _unpack(tmp_path, "orbi")
    ref = LL.series_run([str(orbi)], [str(tof)], None, None, str(FIXTURES / "orbi_iso_checks.csv"))
    core = EV._series_pooled(_pooled("orbi"), cross=EV._source_neutrals(_pooled("tof")), iso=_iso("orbi"))
    m = core.merge(ref, left_on=K, right_on=["neutral", "adduct"], suffixes=("", "_ll"))
    assert len(m) == len(core) == len(ref) == GOLDEN["orbi"][0]
    assert (m["evidence_level"] == m["level"]).all() and (m["lead_lift"] == m["lead_lift_ll"]).all()
    assert set(_key(m[m["lead_lift_ll"]])) == set(D8["orbi"])
    ur = _unpack(tmp_path, "ur")
    ref = LL.series_run([str(ur)], [], str(FIXTURES / "ur_neutral_pairs.csv"), None, str(FIXTURES / "ur_iso_checks.csv"))
    core = EV._series_pooled(_pooled("ur"), upair=_ur_pairs(), iso=_iso("ur"))
    m = core.merge(ref, left_on=K, right_on=["neutral", "adduct"], suffixes=("", "_ll"))
    assert len(m) == GOLDEN["ur"][0] and (m["evidence_level"] == m["level"]).all()
    assert set(_key(m[m["lead_lift_ll"]])) == set(D8["ur"])
    # the named-table guard: the Orbitrap set's table on the TOF set is another batch's
    ref = LL.series_run([str(tof)], [str(orbi)], None, None, str(FIXTURES / "orbi_iso_checks.csv"))
    core = EV._series_pooled(_pooled("tof"), cross=EV._source_neutrals(_pooled("orbi")))
    m = core.merge(ref, left_on=K, right_on=["neutral", "adduct"])
    assert not ref["lead_lift"].any() and len(m) == len(core) == GOLDEN["tof"][0]
    assert (m["evidence_level"] == m["level"]).all()
