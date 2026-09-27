"""Rule U: the uronium neutral pair establishes the neutral (EVIDENCE_LEVELS §4 row 9').

`batch/neutral_pairs.measure` reads the stamped batch time series and the pooled
per-file ledgers; `upair(M)` holds when M is committed as [M+H]+ AND
[M+(CH4N2O)H]+, is C/H/O only, both ions are present at exact mass, co-vary,
are stamped as M's own readings, and their 13C lines do not contradict the
formula. The pooled level recompute then lifts M's rows to 4a -- after row 9,
before 4b -- when the formula has its own support (an isotope, or one plausible
ion on a resolved peak). The fact exists only where the profile declares a
pair: a leak to another channel would move the golden vectors, and a test here
says so.

Run: pytest tests/test_neutral_pairs.py -q
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from peaky.assignment import evidence as EV
from peaky.batch import neutral_pairs as NP
from peaky.chem import chemistry as C
from peaky.chem import profiles as P
from tests.test_evidence import GOLDEN, _pooled, _vector, child, ledger, m0

PAIR = ("[M+H]+", "[M+(CH4N2O)H]+")
N_SPECTRA = 60


def _ion(neutral, adduct):
    cnt = C.parse_formula(neutral)
    add = {"[M+H]+": {"H": 1}, "[M+(CH4N2O)H]+": {"C": 1, "H": 5, "N": 2, "O": 1}}[adduct]
    for k, v in add.items():
        cnt[k] = cnt.get(k, 0) + v
    return C.format_formula(cnt)


def _series(spec: dict) -> pd.DataFrame:
    """A stamped batch time series: per neutral, its two ions (and their 13C
    lines) in N_SPECTRA spectra. spec[neutral] options: phase (its own time
    course), cluster ('covary' | 'anti' | 'absent'), c13 (factor on the true
    13C area), cluster_reading (a stamp other than the neutral's own)."""
    t0 = pd.Timestamp("2026-08-11 00:00", tz="UTC")
    rows = []
    for i in range(N_SPECTRA):
        sid = f"s{i:03d}"
        when = t0 + pd.Timedelta(minutes=30 * i)
        for n, o in spec.items():
            level = 5000.0 * (1.6 + np.sin(2 * np.pi * i / 20.0 + o.get("phase", 0.0)))
            lines = [("[M+H]+", level)]
            if o.get("cluster", "covary") == "covary":
                lines.append(("[M+(CH4N2O)H]+", 0.6 * level * (1 + 0.02 * np.cos(i))))
            elif o["cluster"] == "anti":
                lines.append(("[M+(CH4N2O)H]+", 0.6 * (5000.0 * 3.2 - level)))
            for adduct, h in lines:
                mz = C.ion_mz(n, adduct)
                n_c = C.parse_formula(_ion(n, adduct)).get("C", 0)
                read_n, read_a = n, adduct
                if adduct == PAIR[1] and o.get("cluster_reading"):
                    read_n, read_a = o["cluster_reading"]
                rows.append(dict(sample_item_id=sid, datetime_utc=when, mz=mz, height=h, area=h, role="M0",
                                 neutral_formula=read_n, adduct=read_a))
                rows.append(dict(sample_item_id=sid, datetime_utc=when, mz=mz + NP.D13C,
                                 height=h * n_c * NP.C13_PER_C * o.get("c13", 1.0),
                                 area=h * n_c * NP.C13_PER_C * o.get("c13", 1.0), role="iso_child",
                                 neutral_formula=None, adduct=None))
    return pd.DataFrame(rows)


def _frames(neutrals, adducts=PAIR) -> dict:
    rows = []
    for k, n in enumerate(neutrals):
        for a in adducts:
            rows.append(m0(f"p{k}{a}", n, adduct=a, ion=_ion(n, a), mz=C.ion_mz(n, a)))
    return {"f1": EV.trim(ledger(rows))}


GOOD = ["C10H16O4", "C9H14O3", "C8H12O3", "C7H10O4"]


def _measure(spec, frames=None, pair=PAIR):
    ts = _series(spec)
    return NP.measure(ts, frames if frames is not None else _frames(list(spec)), pair, log=lambda *a: None)


def _held(table) -> dict:
    return dict(zip(table.neutral_formula, table.upair))


# --------------------------------------------------------------------------- the fact
def test_a_clean_co_varying_cho_pair_establishes_its_neutral():
    spec = {n: {"phase": k * 0.7} for k, n in enumerate(GOOD)}
    t = _measure(spec)
    assert _held(t) == {n: True for n in GOOD}
    assert NP.neutrals(t) == set(GOOD)
    s = NP.summary(t, PAIR)
    assert s["committed_both"] == 4 and s["upair"] == 4 and s["pair"] == list(PAIR)
    assert list(t.columns) == list(NP.TABLE_COLUMNS)


def test_each_clause_can_veto_the_pair():
    spec = {n: {"phase": k * 0.7} for k, n in enumerate(GOOD)}
    spec["C9H15NO4"] = {"phase": 0.3}                                   # carries N: the NH4 / urea-ladder alias
    spec["C10H16O4S"] = {"phase": 0.9}                                  # not C/H/O
    spec["C6H10O3"] = {"phase": 1.1, "cluster": "absent"}               # no cluster line in the series
    spec["C6H8O4"] = {"phase": 1.3, "cluster": "anti"}                  # the two lines do not co-vary
    spec["C5H8O3"] = {"phase": 1.5, "cluster_reading": ("C3H9N3O3", "[M+H]+")}   # another reading's M0
    spec["C11H18O4"] = {"phase": 1.7, "c13": 0.3}                       # 13C says ~3 C, not 11-12
    t = _measure(spec)
    held = _held(t)
    assert all(held[n] for n in GOOD)
    for n in ("C9H15NO4", "C10H16O4S", "C6H10O3", "C6H8O4", "C5H8O3", "C11H18O4"):
        assert not held[n], n
    row = t.set_index("neutral_formula")
    assert not row.loc["C9H15NO4", "cho"] and not row.loc["C10H16O4S", "cho"]
    assert not row.loc["C6H10O3", "present"]
    assert not row.loc["C6H8O4", "covary"] and row.loc["C6H8O4", "r_log"] < 0
    assert not row.loc["C5H8O3", "clean"]
    assert row.loc["C11H18O4", "c13_contradicts"]


def test_a_neutral_committed_under_one_adduct_only_holds_nothing():
    spec = {n: {"phase": k * 0.7} for k, n in enumerate(GOOD)}
    frames = _frames(GOOD[:3])
    frames["f2"] = EV.trim(ledger([m0("q", GOOD[3], adduct=PAIR[0], ion=_ion(GOOD[3], PAIR[0]))]))
    t = _measure(spec, frames)
    assert not _held(t)[GOOD[3]] and not t.set_index("neutral_formula").loc[GOOD[3], "committed_both"]


def test_no_pair_or_no_series_measures_nothing():
    spec = {n: {"phase": k * 0.7} for k, n in enumerate(GOOD)}
    assert _measure(spec, pair=()).empty
    assert NP.measure(None, _frames(GOOD), PAIR).empty
    assert NP.neutrals(NP.measure(None, _frames(GOOD), PAIR)) == set()
    assert NP.summary(None, ())["upair"] == 0


# --------------------------------------------------------------------------- profile scope
def test_only_the_uronium_profile_declares_a_pair():
    assert P.PROFILES["Ur"].neutral_pair == PAIR
    for name, prof in P.PROFILES.items():
        if name != "Ur":
            assert prof.neutral_pair == (), name
    assert P.compose([P.PROFILES["Ur"]]).neutral_pair == PAIR
    got = P.from_dict({**{k: getattr(P.PROFILES["Ur"], k) for k in ("name", "label", "polarity", "adducts",
                                                                    "normaliser", "reagent_ion_re", "ranges",
                                                                    "detect_adduct")},
                       "name": "UrX", "neutral_pair": list(PAIR)})
    assert got.neutral_pair == PAIR
    with pytest.raises(ValueError, match="two adducts"):
        P.from_dict({"name": "Bad", "label": "b", "polarity": "+", "adducts": ["[M+H]+"], "normaliser": "tic",
                     "reagent_ion_re": None, "ranges": "C0-10", "detect_adduct": None,
                     "neutral_pair": ["[M+H]+"]})


# --------------------------------------------------------------------------- the level (row 9')
def _pair_rows(neutral="C10H16O4", *, iso=False, degeneracy=0.5, resolvability="resolved", **kw):
    rows = []
    for k, a in enumerate(PAIR):
        ion = _ion(neutral, a)
        rows.append(m0(f"p{k}", neutral, adduct=a, ion=ion, mz=C.ion_mz(neutral, a), height=1000.0,
                       degeneracy=degeneracy, resolvability=resolvability, **kw))
        if iso:
            n_c = C.parse_formula(ion)["C"]
            rows.append(child(f"c{k}", f"p{k}", "13C+1", 1000.0 * EV.C13_PER_CARBON * n_c))
    return ledger(rows)


def _levels(frame, upair) -> dict:
    out = EV.level_pooled({"f": frame}, upair=upair)
    return dict(zip(out.adduct, zip(out.evidence_level, out.evidence_axes, out.level_reason)))


def test_row_9prime_lifts_the_pair_with_formula_support():
    for iso, deg, res in ((True, 2.0, "blended"), (False, 0.5, "resolved")):
        lv = _levels(_pair_rows(iso=iso, degeneracy=deg, resolvability=res), {"C10H16O4"})
        for a in PAIR:
            level, axes, reason = lv[a]
            assert level == "4a" and "upair" in axes and "neutral pair" in reason, (iso, deg, a)
        base = _levels(_pair_rows(iso=iso, degeneracy=deg, resolvability=res), set())
        assert {v[0] for v in base.values()} == {"4b"}


def test_row_9prime_needs_the_formula_supported():
    for deg, res in ((2.0, "resolved"), (0.5, "blended")):
        lv = _levels(_pair_rows(iso=False, degeneracy=deg, resolvability=res), {"C10H16O4"})
        assert {v[0] for v in lv.values()} == {"4b"}, (deg, res)


def test_rows_above_9prime_still_win():
    lv = _levels(_pair_rows(iso=True, tied=True), {"C10H16O4"})
    assert {v[0] for v in lv.values()} == {"5b"}                        # a hard 5b stays 5b
    lv = _levels(_pair_rows(iso=True, method="known:atmospheric"), {"C10H16O4"})
    assert {v[0] for v in lv.values()} == {"3a"}                        # curated first
    lv = _levels(_pair_rows(iso=True, below=True), {"C10H16O4"})
    assert {v[0] for v in lv.values()} == {"5b"}                        # outside the element budget


def test_the_pair_is_never_an_axis_nor_in_cross():
    out = EV.level_pooled({"f": _pair_rows(iso=True)}, upair={"C10H16O4"})
    assert (out["n_axes"] == 2).all()                                   # iso + chan2, upair adds none
    assert not out["cross"].any()
    assert EV.source_neutrals({"f": _pair_rows(iso=True)}) == {"C10H16O4"}   # 4b either way: the cross set is unchanged


def test_a_per_file_ledger_never_carries_the_pair():
    out = EV.compute_levels(_pair_rows(iso=True))
    assert set(out["evidence_level"]) == {"4b"}


# --------------------------------------------------------------------------- the leak guard
def test_a_leaked_fact_would_move_the_goldens():
    """U-design's leak mutant: the fact computed as 'two channels and N-free' on
    every channel lifts TOF and labelled-nitrate rows -- the golden vectors catch
    it. The profile-scoped fact is empty there, and the vectors stand."""
    tof, orbi = _pooled("tof"), _pooled("orbi")
    n_tof, n_orbi = EV.source_neutrals(tof), EV.source_neutrals(orbi)
    for pooled, cross, key in ((tof, n_orbi, "tof"), (orbi, n_tof, "orbi")):
        base = EV.level_pooled(pooled, cross=cross)
        assert (len(base), _vector(base.evidence_level)) == GOLDEN[key]
        leak = set(base.loc[base["chan2"] & ~base["neutral_formula"].str.contains("N"), "neutral_formula"])
        moved = EV.level_pooled(pooled, cross=cross, upair=leak)
        assert _vector(moved.evidence_level) != GOLDEN[key], key
        scoped = NP.neutrals(NP.measure(None, pooled, P.PROFILES["Br"].neutral_pair))
        assert _vector(EV.level_pooled(pooled, cross=cross, upair=scoped).evidence_level) == GOLDEN[key][1]


# --------------------------------------------------------------------------- the reference script
def test_level_ledger_reads_the_pair_table(tmp_path):
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location(
        "level_ledger", Path(__file__).resolve().parents[1] / "scripts" / "level_ledger.py")
    LL = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(LL)
    run = tmp_path / "run"
    (run / "per_file").mkdir(parents=True)
    (run / "tables").mkdir()
    _pair_rows(iso=True).to_csv(run / "per_file" / "s1_ledger.csv", index=False)
    pd.DataFrame({"neutral_formula": ["C10H16O4"], "upair": [True]}).to_csv(run / "tables" / "neutral_pairs.csv",
                                                                           index=False)
    off = LL.run([str(run)], [])
    on = LL.run([str(run)], [], "auto")
    explicit = LL.run([str(run)], [], str(run / "tables" / "neutral_pairs.csv"))
    assert set(off.level) == {"4b"} and set(on.level) == {"4a"} and set(explicit.level) == {"4a"}
    core = EV.level_pooled({"s1": _pair_rows(iso=True)}, upair={"C10H16O4"})
    assert sorted(core.evidence_level) == sorted(on.level)
