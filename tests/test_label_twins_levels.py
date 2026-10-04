"""Rule K on the level facts (`label_untie` / `label_alien` / `label_veto`): what
the labelled-nitrate twin facts do to the pooled fact layer, pinned in the
engine (`evidence._measure` / `_level_pairs` / `_axes_string`), in the
reference script while it holds the pre-0.10.0 decision
(`level_ledger.relabel_pools` / `assign_levels` / `label_twin_facts` / `run`)
and in `assign_batch`'s wiring -- each where a mutant of the rule survived
`tests/test_label_twins.py`.

Before peaky 0.10.0 these tests pinned the pooled level the facts gave (3b on
an untied acid branch, 5b on a veto ...). That decision is private now and
never sees rule K (the merge vote reads each file alone); the evidence scale
reads a veto as a step-0 rejection. So the tests pin the hard inputs and the
facts string (`_levels`: (hard inputs, `evidence_axes`)), and the reference
script's private level only as equal to the engine's. The scorecard no longer
levels a run itself, so its rule-K test is gone.

The facts: an untie clears the arbiter's tie of its own reading only, and only
where that reading is tied; a veto is a hard input, joined last into the hard
list, with its note; an alien 14N line leaves its neutral's per-neutral pools in
both directions (no `chan2`, no `branch`, given or taken) but keeps its own
`anchor` and `iso`; given any facts, the 14N and 15N nitrate clusters of one
neutral are one channel; with none (None, `{}`, a missing or an empty table)
nothing of rule K fires, the fold included. The engine and the script agree row
for row on every one of these.

Run: pytest tests/test_label_twins_levels.py -q
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from peaky.assignment import evidence as EV
from peaky.batch import label_twins as LT
from peaky.chem import chemistry as C
from tests.test_evidence import child, ledger, m0
from tests.test_label_twins import (LABEL, NO3, NO3L, K, _frames, _hard, _j1_rows, _levels, _ll, _nitrate_rows,
                                    _refs_spec, _series)

X, Y, Z, J, W, V = "C10H18O4", "C9H14O4", "C12H14O2", "C11H18O6", "C4H6O4", "C5H8O4"
IO, NO3_15 = "[M]-.", "[M+15NO3]-"


# --------------------------------------------------------------------------- builders
def _prefixed(frame: pd.DataFrame, p: str) -> pd.DataFrame:
    """The ledger with every peak id (and its children's parent id) prefixed, so
    two builders' rows can share one file."""
    f = frame.copy()
    f["peak_id"] = p + f["peak_id"].astype(str)
    f["parent_peak_id"] = f["parent_peak_id"].map(lambda v: p + v if isinstance(v, str) else v)
    return f


def _combo() -> pd.DataFrame:
    """One file where every part of rule K matters: X's tied 14N line (untie) and
    its tied ion-only [M]-. line (untie + veto keyed, never taken); Y's untied
    14N line keyed in the untie and Y's untied ion-only line keyed in the veto; Z's
    labelled reading (veto); J's 14N-only cluster (alien + veto); W on both
    nitrate clusters and V on [M+15NO3]- + [M+^NO3]- (the fold)."""
    return pd.concat([
        _prefixed(_nitrate_rows(X), "x"),
        ledger([m0("xio", X, adduct=IO, ion=X, mz=C.ion_mz(X, "[M-H]-") + 1.0078, method="ion_only:ea", tied=True),
                m0("yh", Y, adduct="[M-H]-", mz=C.ion_mz(Y, "[M-H]-")),
                m0("y14", Y, adduct=NO3, ion=Y + "NO3", mz=C.ion_mz(Y, NO3)),
                m0("yio", Y, adduct=IO, ion=Y, mz=C.ion_mz(Y, "[M-H]-") + 1.0078, method="ion_only:ea"),
                m0("z15", Z, adduct=NO3L, ion=Z + "^NO3", mz=C.ion_mz(Z, NO3L)),
                child("zc", "z15", "13C+1", 1000.0 * EV.C13_PER_CARBON * 12),
                m0("w14", W, adduct=NO3, ion=W + "NO3", mz=C.ion_mz(W, NO3)),
                m0("w15", W, adduct=NO3L, ion=W + "^NO3", mz=C.ion_mz(W, NO3L)),
                m0("v14", V, adduct=NO3_15, ion=V + "N", mz=250.0),
                m0("v15", V, adduct=NO3L, ion=V + "^NO3", mz=251.0)]),
        _prefixed(_j1_rows(J), "j"),
    ], ignore_index=True)


#: the facts the combo is levelled with (what label_twins.facts hands over)
LAB = K(untie={(X, NO3), (X, IO), (Y, NO3)},
        veto={(Z, NO3L): "no 14N twin", (J, NO3): "no 15N partner", (X, IO): "x", (Y, IO): "y"},
        alien={(J, NO3)})

#: the combo's pooled facts with LAB, per (neutral, adduct): (hard inputs, facts string). Before peaky
#: 0.10.0 they read 3b (X, Y: the acid branch), 5b (X's tied ion-only line, J's and Z's vetoed clusters),
#: 4b (J's [M-H]-, its own 13C line), 4c (W, V: one channel each by the fold), 5a (Y's ion-only line).
COMBO_LEVELS = {
    (X, NO3): ((), "chan2|branch|label_untie|files:1"), (X, NO3L): ((), "iso|chan2|carbon|branch|files:1"),
    (X, "[M-H]-"): ((), "chan2|branch|files:1"), (X, IO): (("tied",), "ion_only|files:1"),
    (J, NO3): (("label_veto",), "iso|carbon|label_veto|files:1"), (J, "[M-H]-"): ((), "iso|carbon|files:1"),
    (Z, NO3L): (("label_veto",), "iso|carbon|label_veto|files:1"),
    (W, NO3): ((), "files:1"), (W, NO3L): ((), "files:1"), (V, NO3_15): ((), "files:1"), (V, NO3L): ((), "files:1"),
    (Y, NO3): ((), "chan2|branch|files:1"), (Y, "[M-H]-"): ((), "chan2|branch|files:1"),
    (Y, IO): ((), "ion_only|files:1"),
}

FACT_COLUMNS = ["chan2", "branch", "tied", "label_untie", "label_veto"]


def _engine(rows: pd.DataFrame, label=None) -> pd.DataFrame:
    return EV._series_pooled({"s1": rows}, label=label).set_index(["neutral_formula", "adduct"])


def _facts_of(frame: pd.DataFrame) -> dict:
    """{(neutral, adduct): (hard inputs, facts string)} of an indexed engine table."""
    return {k: (_hard(r), r["evidence_axes"]) for k, r in frame.iterrows()}


def _script(rows: pd.DataFrame, label=None) -> pd.DataFrame:
    LL = _ll()
    frame = LL.measure_source("s1", rows.assign(__file="s1"), None)
    return LL.assign_levels(frame, set(), None, label).set_index(["neutral", "adduct"])


def _run_dir(root: Path, rows: pd.DataFrame, table: pd.DataFrame | None = None) -> Path:
    """A batch run dir: per_file/s1_ledger.csv and, given, tables/label_twins.csv."""
    (root / "per_file").mkdir(parents=True)
    rows.to_csv(root / "per_file" / "s1_ledger.csv", index=False)
    if table is not None:
        (root / "tables").mkdir()
        table.to_csv(root / "tables" / "label_twins.csv", index=False)
    return root


def _table(label: dict) -> pd.DataFrame:
    """A label_twins.csv holding `label`'s facts (one row per keyed pair)."""
    keys = sorted(set(label["untie"]) | set(label["veto"]) | set(label["alien"]))
    return pd.DataFrame({
        "neutral_formula": [n for n, _ in keys], "adduct": [a for _, a in keys],
        "untie": [k in label["untie"] for k in keys], "alien": [k in label["alien"] for k in keys],
        "veto": [k in label["veto"] for k in keys], "note": [label["veto"].get(k, "") for k in keys]})


def _levels_of(frame: pd.DataFrame, col: str) -> dict:
    """{(neutral, adduct): value} of one column of an indexed level table."""
    return frame[col].to_dict()


# --------------------------------------------------------------------------- the untie
def test_the_untie_clears_only_its_own_reading():
    """An untie keyed on (X, [M+NO3]-) clears that pair's tie; X's other tied
    readings keep theirs (a fact about one reading, never about the neutral)."""
    rows = _nitrate_rows(X)
    rows.loc[rows.peak_id.isin(["h", "n15"]), "tied"] = True
    lv = _levels(rows, label=K(untie={(X, NO3)}))
    assert lv[NO3] == ((), "chan2|branch|label_untie|files:1")
    assert lv["[M-H]-"] == (("tied",), "chan2|branch|files:1")
    assert lv[NO3L] == (("tied",), "iso|chan2|carbon|branch|files:1")
    ref = _script(rows, K(untie={(X, NO3)}))
    assert {k: _hard(r) for k, r in ref.iterrows()} == {(X, NO3): (), (X, "[M-H]-"): ("tied",), (X, NO3L): ("tied",)}
    assert _levels_of(ref, "label_untie") == {(X, NO3): True, (X, "[M-H]-"): False, (X, NO3L): False}
    assert _levels_of(ref, "level") == _levels_of(_engine(rows, K(untie={(X, NO3)})), "evidence_level")


def test_the_untie_leaves_a_low_confidence_row_hard():
    """The untie clears `tied` only: a row the engine also rates Low/Suspect stays
    hard on that input alone (and the scale rejects it at step 0: lowconf)."""
    rows = _nitrate_rows(X)
    rows.loc[rows.peak_id == "n14", "confidence"] = "Low"
    lv = _levels(rows, label=K(untie={(X, NO3)}))
    assert lv[NO3] == (("lowconf",), "chan2|branch|label_untie|files:1")
    assert _hard(_script(rows, K(untie={(X, NO3)})).loc[(X, NO3)]) == ("lowconf",)


# --------------------------------------------------------------------------- the veto and the strings
def test_the_label_tokens_sit_in_the_documented_order():
    """The facts string reads ... upair, label_untie, label_veto, known:<family>, files:<n>."""
    rows = _nitrate_rows(X)
    rows.loc[rows.peak_id == "n14", "method"] = "known:atmospheric"
    lv = _levels(rows, label=K(untie={(X, NO3)}, veto={(X, NO3): "n"}))
    assert lv[NO3] == (("label_veto",), "chan2|branch|label_untie|label_veto|known:atmospheric|files:1")


def test_the_veto_joins_the_other_hard_inputs():
    """The veto is one hard input among tie, below, lowconf -- held together,
    never a separate branch -- and carries its note."""
    rows = _nitrate_rows(X, tied=False)
    rows.loc[rows.peak_id == "n15", ["tied", "below_assignability"]] = True
    rows.loc[rows.peak_id == "n15", "confidence"] = "Suspect"
    lv = _levels(rows, label=K(veto={(X, NO3L): "n"}))
    assert lv[NO3L] == (("tied", "below", "lowconf", "label_veto"), "iso|chan2|carbon|branch|label_veto|files:1")
    rows = _nitrate_rows(X, tied=False)
    rows.loc[rows.peak_id == "n15", "tied"] = True
    out = _engine(rows, K(veto={(X, NO3L): "n"}))
    assert _hard(out.loc[(X, NO3L)]) == ("tied", "label_veto") and out.loc[(X, NO3L), "label_note"] == "n"


def test_a_none_note_reads_as_no_note():
    """A veto whose note is None refutes and records ''."""
    out = _engine(_nitrate_rows(X, tied=False), K(veto={(X, NO3L): None}))
    assert _hard(out.loc[(X, NO3L)]) == ("label_veto",)
    assert out.loc[(X, NO3L), "label_note"] == "" and bool(out.loc[(X, NO3L), "label_veto"])


def test_an_ion_only_pair_carries_no_label_note():
    """An ion-only pair keyed in the veto takes neither the veto nor its note."""
    out = _engine(_combo(), LAB)
    assert not out.loc[(X, IO), "label_veto"] and out.loc[(X, IO), "label_note"] == ""
    assert not out.loc[(Y, IO), "label_veto"] and out.loc[(Y, IO), "label_note"] == ""
    assert out.loc[(Z, NO3L), "label_note"] == "no 14N twin"
    assert out.loc[(X, NO3), "label_note"] == ""


def test_the_rule_k_columns_follow_upair_in_order():
    """evidence_levels.csv carries upair, label_untie, label_veto, label_note in that
    order (OUTPUTS.md), with or without the facts."""
    for label in (None, LAB):
        cols = list(EV._series_pooled({"f": _combo()}, label=label).columns)
        i = cols.index("upair")
        assert cols[i:i + 4] == ["upair", "label_untie", "label_veto", "label_note"], label


# --------------------------------------------------------------------------- the alien line
def test_an_alien_line_takes_no_pool_fact_but_keeps_its_own_axes():
    """An alien 14N line takes neither chan2 nor the branch from its neutral's
    other readings (both of which hold), keeps its own anchor and 13C line, and
    the neutral's regular readings keep theirs."""
    rows = ledger([m0("h", X, adduct="[M-H]-", ion="C10H17O4", mz=C.ion_mz(X, "[M-H]-")),
                   m0("n15", X, adduct=NO3L, ion=X + "^NO3", mz=C.ion_mz(X, NO3L)),
                   child("c1", "n15", "13C+1", 1000.0 * EV.C13_PER_CARBON * 10),
                   m0("n14", X, adduct=NO3, ion=X + "NO3", mz=C.ion_mz(X, NO3), anchor="a"),
                   child("c2", "n14", "13C+1", 1000.0 * EV.C13_PER_CARBON * 10)])
    lab = K(alien={(X, NO3)})
    out = _engine(rows, lab)
    a = out.loc[(X, NO3)]
    assert (a.chan2, a.branch, a.anchor, a.iso, a.carbon_ev) == (False, False, True, True, True)
    assert (_hard(a), a.evidence_axes, a.n_axes, bool(a.cross)) == ((), "iso|anchor|carbon|files:1", 2, False)
    for adduct in ("[M-H]-", NO3L):
        assert out.loc[(X, adduct), "branch"] and out.loc[(X, adduct), "chan2"], adduct
    ref = _script(rows, lab)
    for key in out.index:
        assert ref.loc[key, "level"] == out.loc[key, "evidence_level"], key
        for c in ("chan2", "branch", "anchor", "iso"):
            assert bool(ref.loc[key, c]) == bool(out.loc[key, c]), (key, c)


def test_an_alien_line_gives_its_neutral_nothing_in_the_script_too():
    """J1: the C11 acid whose only cluster is an alien 14N line loses the branch in
    the engine and in the reference script alike (each reading keeps its own 13C line)."""
    lab = K(alien={(J, NO3)})
    for got in (_engine(_j1_rows(J), lab), _script(_j1_rows(J), lab)):
        assert not got["chan2"].any() and not got["branch"].any() and got["iso"].all()
    for got in (_engine(_j1_rows(J), K()), _script(_j1_rows(J), K())):
        assert got["branch"].all() and got["chan2"].all()


def test_a_reagent_dominated_cluster_keeps_the_acid_branch():
    """Trifluoroacetic acid's 14N line runs at 0.12 x the cluster ratio (the
    reagent's own impurity): consistent, not alien, not refuted -- its [M-H]- and
    its clusters read the acid branch; were the 14N line alien, the 15N cluster
    would still carry the branch."""
    tfa = "C2HF3O2"
    spec = _refs_spec(**{tfa: {"phase": 1.1, "q": 0.12}})
    frames = _frames(list(spec), no3=[tfa])
    table = LT.measure(_series(spec), frames, LABEL, log=lambda *a: None)
    line = table[(table.neutral_formula == tfa) & (table.adduct == NO3)].iloc[0]
    assert line.line_verdict == "consistent" and not line.alien and not line.veto
    facts = LT.facts(table)
    assert (tfa, NO3) not in facts["alien"] and not any(k[0] == tfa for k in facts["veto"])
    led = pd.concat([frames["f1"], EV.trim(ledger([m0("t", tfa, adduct="[M-H]-", ion="C2F3O2",
                                                        mz=C.ion_mz(tfa, "[M-H]-"))]))], ignore_index=True)
    out = EV._series_pooled({"f1": led}, label=facts).set_index(["neutral_formula", "adduct"])
    assert all(out.loc[(tfa, a), "branch"] and _hard(out.loc[(tfa, a)]) == () for a in ("[M-H]-", NO3, NO3L))
    out = EV._series_pooled({"f1": led}, label=K(alien={(tfa, NO3)})).set_index(["neutral_formula", "adduct"])
    assert out.loc[(tfa, "[M-H]-"), "branch"] and out.loc[(tfa, NO3L), "branch"]
    assert not out.loc[(tfa, NO3), "branch"] and not out.loc[(tfa, NO3), "chan2"] and out.loc[(tfa, NO3), "n_axes"] == 0
    # committed through the perfluoroacid family the curated class is a fact of its own, never below
    led.loc[led["adduct"].eq("[M-H]-") & led["neutral_formula"].eq(tfa), "method"] = "known:perfluoroacid"
    out = EV._series_pooled({"f1": led}, label=facts).set_index(["neutral_formula", "adduct"])
    assert out.loc[(tfa, "[M-H]-"), "known_fam"] == "perfluoroacid" and _hard(out.loc[(tfa, "[M-H]-")]) == ()
    assert out.loc[(tfa, NO3), "branch"]


# --------------------------------------------------------------------------- the fold
def test_both_spellings_of_the_14n_cluster_fold_into_the_15n_channel():
    """Given the facts, [M+NO3]- and [M+15NO3]- each count as the [M+^NO3]-
    channel (the engine's and the script's fold are one map); without them each
    is a channel of its own."""
    LL = _ll()
    assert EV.LABEL_FOLD == LL.LABEL_FOLD == {NO3: NO3L, NO3_15: NO3L}
    rows = _combo()
    keys = [(W, NO3), (W, NO3L), (V, NO3_15), (V, NO3L)]
    for got in (_engine(rows, K()), _script(rows, K())):
        assert [bool(got.loc[k, "chan2"]) for k in keys] == [False] * 4
    for got in (_engine(rows, None), _script(rows, None)):
        assert [bool(got.loc[k, "chan2"]) for k in keys] == [True] * 4
    assert [int(_engine(rows, None).loc[k, "n_axes"]) for k in keys] == [1] * 4      # the second channel
    assert [int(_engine(rows, K()).loc[k, "n_axes"]) for k in keys] == [0] * 4


def test_an_empty_fact_dict_is_no_rule_k():
    """label={} is label=None: no fact and no fold (the engine and the script)."""
    rows = _combo()
    assert EV._series_pooled({"s1": rows}, label={}).equals(EV._series_pooled({"s1": rows}, label=None))
    assert EV._series_pooled({"s1": rows}, label={}).equals(EV._series_pooled({"s1": rows}))
    assert _script(rows, {}).equals(_script(rows, None))
    assert bool(_script(rows, {}).loc[(W, NO3), "chan2"])


# --------------------------------------------------------------------------- engine / script lockstep
def test_the_reference_script_levels_rule_k_in_lockstep_with_the_engine():
    """Alien, fold, untie and veto together: the script's private level and every
    rule K fact column match the engine's, row for row; the untie never lands on
    an untied or an ion-only pair and the veto never on an ion-only one."""
    rows = _combo()
    core, ref = _engine(rows, LAB), _script(rows, LAB)
    assert sorted(core.index) == sorted(ref.index) == sorted(COMBO_LEVELS)
    assert _facts_of(core) == COMBO_LEVELS
    assert _levels_of(core, "evidence_level") == _levels_of(ref, "level")
    for c in FACT_COLUMNS:
        assert _levels_of(core, c) == {k: bool(v) for k, v in _levels_of(ref, c).items()}, c
    assert {k for k, v in _levels_of(ref, "label_untie").items() if v} == {(X, NO3)}
    assert {k for k, v in _levels_of(ref, "label_veto").items() if v} == {(Z, NO3L), (J, NO3)}
    assert bool(ref.loc[(X, IO), "tied"]) and not ref.loc[(X, NO3), "tied"]
    assert not ref.loc[(Y, NO3), "tied"]


def test_the_reference_script_reads_every_fact_from_a_run_table(tmp_path):
    """--label-twins on a run dir reads untie, veto and alien from its
    tables/label_twins.csv and levels exactly as the engine does on
    label_twins.facts of the same table."""
    LL = _ll()
    table = _table(LAB)
    run = _run_dir(tmp_path / "RUN_1", _combo(), table)
    assert LT.facts(table) == LAB
    assert LL.label_twin_facts(str(run)) == {"untie": LAB["untie"], "veto": set(LAB["veto"]), "alien": LAB["alien"]}
    got = LL.run([str(run)], [], None, "auto").set_index(["neutral", "adduct"])
    assert _levels_of(got, "level") == _levels_of(_engine(_combo(), LAB), "evidence_level")
    assert {k: _hard(r) for k, r in got.iterrows()} == {k: v[0] for k, v in COMBO_LEVELS.items()}
    assert _levels_of(got, "chan2") == _levels_of(_engine(_combo(), LAB), "chan2")


def test_a_named_table_reaches_only_the_source_holding_its_pairs(tmp_path):
    """--label-twins <csv> applies to each levelled source that holds every pair the
    table names -- another batch's table does not fire there."""
    LL = _ll()
    zcsv, jcsv = tmp_path / "z.csv", tmp_path / "j.csv"
    _table(K(veto={(Z, NO3L): "no 14N twin"})).to_csv(zcsv, index=False)
    _table(K(veto={(J, NO3): "n"}, alien={(J, NO3)})).to_csv(jcsv, index=False)
    zrows = ledger([m0("z15", Z, adduct=NO3L, ion=Z + "^NO3", mz=C.ion_mz(Z, NO3L)),
                    child("zc", "z15", "13C+1", 1000.0 * EV.C13_PER_CARBON * 12)])
    labelled = _run_dir(tmp_path / "labelled", zrows)
    other = _run_dir(tmp_path / "other", _j1_rows(J))
    got = LL.run([str(labelled), str(other)], [], None, str(zcsv)).set_index(["neutral", "adduct"])
    assert bool(got.loc[(Z, NO3L), "label_veto"])
    assert got.loc[(J, "[M-H]-"), "branch"] and got.loc[(J, NO3), "branch"] and not got.loc[(J, NO3), "label_veto"]
    got = LL.run([str(labelled), str(other)], [], None, str(jcsv)).set_index(["neutral", "adduct"])
    assert not got.loc[(Z, NO3L), "label_veto"]
    assert not got.loc[(J, "[M-H]-"), "branch"] and bool(got.loc[(J, NO3), "label_veto"])
    off = LL.run([str(labelled)], [], None, None).set_index(["neutral", "adduct"])
    assert not off.loc[(Z, NO3L), "label_veto"]


def test_a_missing_or_empty_table_is_no_rule_k(tmp_path):
    """No table, or a header-only one: label_twin_facts is None and the source
    levels as without --label-twins, the one-channel fold included."""
    LL = _ll()
    rows = _combo()
    bare = _run_dir(tmp_path / "bare", rows)
    empty = _run_dir(tmp_path / "empty", rows, pd.DataFrame(columns=list(LT.TABLE_COLUMNS)))
    assert LL.label_twin_facts(str(bare)) is None
    assert LL.label_twin_facts(str(empty)) is None
    assert LL.label_twin_facts(str(empty / "tables" / "label_twins.csv")) is None
    off = list(LL.run([str(bare)], [], None, None)["level"])
    for path, twins in ((bare, "auto"), (empty, "auto"), (bare, str(empty / "tables" / "label_twins.csv"))):
        got = LL.run([str(path)], [], None, twins).set_index(["neutral", "adduct"])
        assert list(got["level"]) == off, (path, twins)
        assert bool(got.loc[(W, NO3), "chan2"]) and not got["label_untie"].any() and not got["label_veto"].any()


def test_two_runs_under_one_out_dir_pick_no_table(tmp_path):
    """An --out-dir holding two runs' tables names no single batch: no facts."""
    LL = _ll()
    for name, flag in (("RUN_1", True), ("RUN_2", False)):
        _run_dir(tmp_path / "out" / name, _nitrate_rows(X),
                 pd.DataFrame({"neutral_formula": [X], "adduct": [NO3], "untie": [flag], "veto": [False],
                               "alien": [False], "note": [""]}))
    assert LL.label_twin_facts(str(tmp_path / "out")) is None
    _run_dir(tmp_path / "one" / "RUN_1", _nitrate_rows(X),
             pd.DataFrame({"neutral_formula": [X], "adduct": [NO3], "untie": [True], "veto": [False],
                           "alien": [False], "note": [""]}))
    assert LL.label_twin_facts(str(tmp_path / "one")) == {"untie": {(X, NO3)}, "veto": set(), "alien": set()}


def test_a_blank_cell_is_not_a_fact(tmp_path):
    """A blank fact cell reads False, as label_twins.facts reads it."""
    LL = _ll()
    csv = tmp_path / "label_twins.csv"
    t = pd.DataFrame({"neutral_formula": [X, Z, J], "adduct": [NO3, NO3L, NO3], "untie": [True, None, None],
                      "veto": [None, True, None], "alien": [None, None, True], "note": ["", "n", ""]})
    t.to_csv(csv, index=False)
    assert LL.label_twin_facts(str(csv)) == {"untie": {(X, NO3)}, "veto": {(Z, NO3L)}, "alien": {(J, NO3)}}
    f = LT.facts(pd.read_csv(csv))
    assert (f["untie"], set(f["veto"]), f["alien"]) == ({(X, NO3)}, {(Z, NO3L)}, {(J, NO3)})


def test_a_table_without_a_fact_column_reads_that_fact_empty(tmp_path):
    """A trimmed table (no veto, no alien column) reads those facts as empty."""
    LL = _ll()
    csv = tmp_path / "label_twins.csv"
    pd.DataFrame({"neutral_formula": [X], "adduct": [NO3], "untie": [True]}).to_csv(csv, index=False)
    assert LL.label_twin_facts(str(csv)) == {"untie": {(X, NO3)}, "veto": set(), "alien": set()}


# --------------------------------------------------------------------------- the batch, end to end
def _batch(tmp_path, monkeypatch, spec, commits, tie_alts):
    """assign_batch.run on the composed NO3+NO3_15N profile with a stub per-file
    assignment: every file commits `commits`; tie_alts(n_call, neutral, adduct)
    gives a row's alternatives (None = not tied). Returns the calls in order."""
    from peaky.assignment import assign as A
    from peaky.assignment import ledger as L
    from peaky.assignment import tiers as T
    from peaky.batch import assign_batch as AB
    from peaky.io import io_mascope as IOM

    ts = _series(spec)
    pk = ts[["sample_item_id", "datetime_utc", "mz", "height"]].copy()
    pk["sample_item_name"] = "n_" + pk["sample_item_id"]
    calls = []

    def fake_assign(sid, context="ambient-air", **kw):
        calls.append(sid)
        ids = [f"P{k}" for k in range(len(commits))]
        mzs = [C.ion_mz(n, NO3L if a == NO3_15 else a) + (0.5 if a == NO3_15 else 0.0) for n, a in commits]
        led = L.new_ledger(pd.DataFrame({"peak_id": ids, "mz": mzs,
                                         "height": [5000.0] * len(ids)}))
        for pid, (n, a) in zip(ids, commits):
            alts = tie_alts(len(calls), n, a)
            tie = alts is not None
            L.commit_assignment(led, pid, neutral_formula=n, adduct=a, ion_formula=n, ion_score=0.95,
                                eff_score=0.9, tied=tie, alternatives=alts, compound_score=0.95, ppm_error=0.1,
                                pass_no=1, method="cheminfo", confidence="Medium" if tie else "High",
                                commentary=f"Pass 1: {n} {a}" + (" (TIE)" if tie else ""))
        T.apply_tiers(led)
        led["degeneracy_density"] = 0.5
        led["resolvability"] = "resolved"
        return {"ledger": led, "stats": {"noise_edge_cps": 4.0, "height_gate_cps": 10.0},
                "plausibility_audit": [], "summaries": {}, "problems": []}

    monkeypatch.setattr(IOM, "connect", lambda *a, **k: "CLIENT")
    monkeypatch.setattr(IOM, "fetch_peaks", lambda client, sid, use_cache=True: pd.DataFrame(
        {"peak_id": ["A"], "mz": [200.1], "height": [1.0e5]}))
    monkeypatch.setattr(IOM, "estimate_offset", lambda raw: 0.0)
    monkeypatch.setattr(A, "run", fake_assign)
    AB.run(peaks=pk, ts_peaks=pk, reagent="NO3+NO3_15N", batch="test batch", out_dir=str(tmp_path),
           k_min=2, k_max=3, min_gain=0.0, n_jobs=1, resolving_power=100_000, log=lambda *a: None)
    return calls


def _alias(n):
    """The same-ion organonitrate alternative of a tied [n+NO3]- row."""
    f = C.parse_formula(n)
    alt = C.format_formula({k: f.get(k, 0) + d for k, d in (("C", 0), ("H", 1), ("N", 1), ("O", 3))})
    return {"formula": alt, "adduct": "[M-H]-", "ion_score": 0.9, "raw_score": 0.9, "eff_score": 0.88, "ppm": 0.3}


def test_one_file_with_a_foreign_tie_keeps_the_batch_tie(tmp_path, monkeypatch):
    """Every file's alias table reaches the twin test: X's 14N line tracks, but the
    first file's tie on it is also with a different ion (C11H22O5 [M+Cl]-), so X
    keeps its tie while X2 -- alias-only in every file -- unties; J's 14N line has
    no 15N partner: alien and refuted, so J's [M-H]- loses the branch (the facts
    reach the pooled fact table whole), the scale rejects J's line at step 0
    and the summary counts it."""
    x, x2, j = "C10H18O3", "C10H16O4", J
    spec = _refs_spec(**{x: {"phase": 2.0}, j: {"phase": 0.9, "share15": 0.0}})
    commits = ([(n, NO3L) for n in spec if n != j]
               + [(x, NO3), (x, "[M-H]-"), (x2, NO3), (x2, "[M-H]-"), (j, NO3), (j, "[M-H]-")])

    def tie_alts(n_call, n, a):
        if a != NO3 or n == j:
            return None
        alts = [_alias(n)]
        if n == x and n_call == 1:
            alts.append({"formula": "C11H22O5", "adduct": "[M+Cl]-", "ion_score": 0.9, "raw_score": 0.9,
                         "eff_score": 0.88, "ppm": 0.3})
        return alts

    calls = _batch(tmp_path, monkeypatch, spec, commits, tie_alts)
    assert len(calls) >= 2
    table = pd.read_csv(tmp_path / "tables" / "label_twins.csv")
    rows = table[table.adduct == NO3].set_index("neutral_formula")
    assert bool(rows.loc[x, "cluster_k"]) and not rows.loc[x, "alias_only_tie"] and not rows.loc[x, "untie"]
    assert bool(rows.loc[x2, "alias_only_tie"]) and bool(rows.loc[x2, "untie"])
    assert rows.loc[j, "line_verdict"] == "absent" and bool(rows.loc[j, "alien"]) and bool(rows.loc[j, "veto"])
    ev = pd.read_csv(tmp_path / "tables" / "evidence_levels.csv").set_index(["neutral_formula", "adduct"])
    assert bool(ev.loc[(x, NO3), "tied"]) and not bool(ev.loc[(x2, NO3), "tied"])
    assert bool(ev.loc[(j, NO3), "label_veto"])
    assert not ev.loc[(j, "[M-H]-"), "chan2"] and not ev.loc[(j, "[M-H]-"), "branch"]
    assert {k for k in ev.index if ev.loc[k, "label_untie"]} == {(x2, NO3)}
    merged = pd.read_csv(tmp_path / "merged_ledger.csv", keep_default_na=False) \
        .set_index(["neutral_formula", "adduct"])
    assert merged.loc[(j, NO3), "evidence_level"] == "5b" and "label_veto" in merged.loc[(j, NO3), "would_lift"]
    summ = json.load(open(tmp_path / "batch_summary.json"))["evidence_levels"]["label_twins"]
    assert (summ["in_scope"], summ["untie"], summ["alien"], summ["lines_absent"], summ["lines_refuted"]) == \
        (True, 1, 1, 1, 1)


def test_a_labelled_batch_with_no_labelled_reading_still_judges_its_14n_lines(tmp_path, monkeypatch):
    """On the labelled profile a batch that commits nothing on [M+^NO3]- still
    judges its committed 14N lines: with no committed labelled cluster there is
    no reference, so the line is untestable (alien: out of its neutral's pools,
    no veto), and the two nitrate spellings count as one channel."""
    w = W
    spec = {w: {"phase": 0.3}}
    commits = [(w, NO3), (w, NO3_15)]
    _batch(tmp_path, monkeypatch, spec, commits, lambda n_call, n, a: None)
    table = pd.read_csv(tmp_path / "tables" / "label_twins.csv")
    assert list(table.columns) == list(LT.TABLE_COLUMNS)
    row = table.set_index(["neutral_formula", "adduct"]).loc[(w, NO3)]
    assert row["line_verdict"] == "untestable" and bool(row["alien"]) and not bool(row["veto"])
    assert not table["reference"].any() and (table["adduct"] == NO3).all()
    summ = json.load(open(tmp_path / "batch_summary.json"))["evidence_levels"]["label_twins"]
    assert summ["in_scope"] is True and summ["committed_lines"] == 1 and summ["lines_untestable"] == 1
    assert summ["alien"] == 1 and summ["readings"] == 0 and summ["refuted"] == 0
    ev = pd.read_csv(tmp_path / "tables" / "evidence_levels.csv")
    assert not ev["label_untie"].any() and not ev["label_veto"].any()
    assert ev.set_index("adduct")["chan2"].to_dict() == {NO3: False, NO3_15: False}
