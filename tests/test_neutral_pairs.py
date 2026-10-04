"""Rule U: the uronium neutral pair (the `upair` fact).

`batch/neutral_pairs.measure` reads the stamped batch time series and the pooled
per-file ledgers; `upair(M)` holds when M is committed as [M+H]+ AND
[M+(CH4N2O)H]+, is C/H/O only, both ions are present at exact mass, co-vary,
are stamped as M's own readings, and their 13C lines do not contradict the
formula. The fact exists only where the profile declares a pair: a leak to
another channel would move the golden fact vectors, and a test here says so.

Before peaky 0.10.0 the pooled level lifted M's rows to 4a on the pair (row 9',
after row 9, before 4b) when the formula had its own support. That decision is
private now and never sees the pair table (the merge vote reads each file
alone); the evidence scale reads the two channels as a "two routes" tag. So the
tests here pin the fact, the formula-support facts the row read (an isotope
line, or one plausible ion on a resolved peak), the hard inputs that outranked
it, and -- on a batch -- the routes tag the scale prints; the reference script
is compared with the engine (its private level) while it still holds that
decision.

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
from tests.test_evidence import GOLDEN, ORBI_NO_LOCK, _pooled, _vector, child, ledger, m0

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


HARD = ("tied", "below", "lead", "lowconf", "label_veto", "iso_veto")


def _levels(frame, upair) -> dict:
    """{adduct: fact row} of the pooled fact layer over one file with the pair set `upair`."""
    out = EV._series_pooled({"f": frame}, upair=upair)
    return {r.adduct: r for r in out.itertuples(index=False)}


def _support(r) -> bool:
    """The formula support row 9' read: an isotope line, or one plausible ion on a resolved peak."""
    return bool(r.iso) or (r.degeneracy <= 1 and bool(r.res_ok))


def _hard(r) -> tuple:
    return tuple(h for h in HARD if bool(getattr(r, h)))


def test_the_pair_fact_lands_on_both_rows_with_or_without_formula_support():
    for iso, deg, res in ((True, 2.0, "blended"), (False, 0.5, "resolved")):
        lv = _levels(_pair_rows(iso=iso, degeneracy=deg, resolvability=res), {"C10H16O4"})
        for a in PAIR:
            assert lv[a].upair and "upair" in lv[a].evidence_axes and _support(lv[a]), (iso, deg, a)
        base = _levels(_pair_rows(iso=iso, degeneracy=deg, resolvability=res), set())
        assert not any(r.upair for r in base.values()) and all(r.chan2 for r in base.values())


def test_without_formula_support_the_pair_is_still_recorded():
    for deg, res in ((2.0, "resolved"), (0.5, "blended")):
        lv = _levels(_pair_rows(iso=False, degeneracy=deg, resolvability=res), {"C10H16O4"})
        assert all(r.upair and not _support(r) for r in lv.values()), (deg, res)


def test_hard_inputs_and_curated_identities_stand_beside_the_pair():
    lv = _levels(_pair_rows(iso=True, tied=True), {"C10H16O4"})
    assert {_hard(r) for r in lv.values()} == {("tied",)} and all(r.upair for r in lv.values())
    lv = _levels(_pair_rows(iso=True, method="known:atmospheric"), {"C10H16O4"})
    assert {r.known_fam for r in lv.values()} == {"atmospheric"}
    lv = _levels(_pair_rows(iso=True, below=True), {"C10H16O4"})
    assert {_hard(r) for r in lv.values()} == {("below",)}                # outside the element budget


def test_the_pair_is_never_an_axis_nor_in_cross():
    out = EV._series_pooled({"f": _pair_rows(iso=True)}, upair={"C10H16O4"})
    assert (out["n_axes"] == 2).all()                                   # iso + chan2, upair adds none
    assert not out["cross"].any()
    # the cross set is unchanged by the pair either way
    assert EV._source_neutrals({"f": _pair_rows(iso=True)}) == {"C10H16O4"}


def test_a_per_file_ledger_never_carries_the_pair():
    """The merge vote reads each file alone: no pair, the formula confirmed (class 1)."""
    out = EV._level_pairs({"": _pair_rows(iso=True)}, per_file=True)
    assert not out["upair"].any()
    assert set(EV.vote_classes(_pair_rows(iso=True))) == {1}


# --------------------------------------------------------------------------- the leak guard
def test_a_leaked_fact_would_move_the_goldens():
    """U-design's leak mutant: the fact computed as 'two channels and N-free' on
    every channel lifts TOF and labelled-nitrate rows -- the golden vectors catch
    it. The profile-scoped fact is empty there, and the vectors stand."""
    tof, orbi = _pooled("tof"), _pooled("orbi")
    n_tof, n_orbi = EV._source_neutrals(tof), EV._source_neutrals(orbi)
    for pooled, cross, key in ((tof, n_orbi, "tof"), (orbi, n_tof, "orbi")):
        base = EV._series_pooled(pooled, cross=cross)
        gold = {"tof": GOLDEN["tof"], "orbi": ORBI_NO_LOCK}[key]      # the orbi set without its lock table
        assert (len(base), _vector(base)) == gold
        leak = set(base.loc[base["chan2"] & ~base["neutral_formula"].str.contains("N"), "neutral_formula"])
        moved = EV._series_pooled(pooled, cross=cross, upair=leak)
        assert _vector(moved) != gold, key
    # the scope: the same series and ledgers hold pairs under the uronium profile's
    # declaration and none under any profile that declares no pair
    spec = {n: {"phase": k * 0.7} for k, n in enumerate(GOOD)}
    ts, frames = _series(spec), _frames(GOOD)
    assert NP.neutrals(NP.measure(ts, frames, P.PROFILES["Ur"].neutral_pair)) == set(GOOD)
    for name in ("Br", "NO3", "NO3_15N"):
        assert NP.measure(ts, frames, P.PROFILES[name].neutral_pair).empty, name


# --------------------------------------------------------------------------- the reference script
def test_level_ledger_reads_the_pair_table(tmp_path):
    LL = _ll()
    run = tmp_path / "run"
    (run / "per_file").mkdir(parents=True)
    (run / "tables").mkdir()
    _pair_rows(iso=True).to_csv(run / "per_file" / "s1_ledger.csv", index=False)
    pd.DataFrame({"neutral_formula": ["C10H16O4"], "upair": [True]}).to_csv(run / "tables" / "neutral_pairs.csv",
                                                                           index=False)
    off = LL.series_run([str(run)], [])
    on = LL.series_run([str(run)], [], "auto")
    explicit = LL.series_run([str(run)], [], str(run / "tables" / "neutral_pairs.csv"))
    assert not off["upair"].any() and on["upair"].all() and explicit["upair"].all()
    assert list(explicit.level) == list(on.level) != list(off.level)
    core = EV._series_pooled({"s1": _pair_rows(iso=True)}, upair={"C10H16O4"})
    assert sorted(core.evidence_level) == sorted(on.level)              # the private decision, alike


# --------------------------------------------------------------------------- the thresholds
def _one(spec_extra: dict, n="C12H20O4") -> pd.Series:
    """The table row of one extra neutral measured beside the four GOOD ones
    (which set the 13C scale)."""
    spec = {m: {"phase": k * 0.7} for k, m in enumerate(GOOD)}
    spec[n] = spec_extra
    ts = spec_extra.pop("_ts_hook", lambda t: t)(_series(spec))
    t = NP.measure(ts, _frames(list(spec)), PAIR, log=lambda *a: None)
    return t.set_index("neutral_formula").loc[n]


def _thin(ts, neutral, adduct, keep_share):
    """Drop an ion's line from all but `keep_share` of the spectra."""
    mz = C.ion_mz(neutral, adduct)
    near = (ts["mz"] - mz).abs() < 1e-6
    sids = sorted(ts["sample_item_id"].unique())
    drop = set(sids[int(round(keep_share * len(sids))):])
    return ts[~(near & ts["sample_item_id"].isin(drop))]


def test_presence_needs_half_the_spectra():
    n = "C12H20O4"
    assert not _one({"_ts_hook": lambda t: _thin(t, n, PAIR[1], 0.4)})["present"]
    assert _one({"_ts_hook": lambda t: _thin(t, n, PAIR[1], 0.6)})["present"]


def test_presence_needs_the_median_within_one_ppm():
    n = "C12H20O4"
    def shift(ts, ppm):
        mz = C.ion_mz(n, PAIR[0])
        near = (ts["mz"] - mz).abs() < 1e-6
        ts = ts.copy()
        ts.loc[near, "mz"] = mz * (1 + ppm * 1e-6)
        return ts
    r = _one({"_ts_hook": lambda t: shift(t, 1.5)})               # inside the 2 ppm window, off by 1.5
    assert r["det_bare"] == 1.0 and not r["present"] and not r["upair"]
    assert _one({"_ts_hook": lambda t: shift(t, 0.6)})["present"]


def test_co_variation_needs_r_of_one_half():
    n = "C12H20O4"
    def blend(ts, w):
        """cluster = w x the neutral's course + (1 - w) x an unrelated course."""
        mz = C.ion_mz(n, PAIR[1])
        near = (ts["mz"] - mz).abs() < 1e-6
        ts = ts.copy()
        i = ts.loc[near, "sample_item_id"].str[1:].astype(int).to_numpy()
        own = 5000.0 * (1.6 + np.sin(2 * np.pi * i / 20.0))
        other = 5000.0 * (1.6 + np.sin(2 * np.pi * i / 7.3 + 1.0))
        ts.loc[near, "height"] = 0.6 * (w * own + (1 - w) * other)
        return ts
    weak = _one({"_ts_hook": lambda t: blend(t, 0.25)})
    strong = _one({"_ts_hook": lambda t: blend(t, 0.8)})
    assert 0.0 < weak["r_log"] < NP.COVARY_R_MIN and not weak["covary"]
    assert strong["r_log"] >= NP.COVARY_R_MIN and strong["covary"]


def test_co_variation_reads_only_spectra_where_both_lines_are_bright():
    """A dim, flat protonated line (under the height floor) beside a bright,
    moving urea line is not evidence against the pair; counted, it would sink r."""
    n = "C12H20O4"
    def dim(ts):
        mz = C.ion_mz(n, PAIR[0])
        near = (ts["mz"] - mz).abs() < 1e-6
        ts = ts.copy()
        late = ts["sample_item_id"].str[1:].astype(int) >= 40
        ts.loc[near & late, "height"] = 100.0
        return ts
    r = _one({"_ts_hook": dim})
    assert r["n_both"] == 40 and r["covary"] and r["upair"]


def test_co_variation_is_read_on_the_log_scale():
    """One spectrum a hundred times brighter in the protonated line sinks the
    linear r and not the log r."""
    n = "C12H20O4"
    def spike(ts):
        mz = C.ion_mz(n, PAIR[0])
        near = (ts["mz"] - mz).abs() < 1e-6
        ts = ts.copy()
        ts.loc[near & (ts["sample_item_id"] == "s005"), "height"] *= 100.0
        return ts
    r = _one({"_ts_hook": spike})
    assert r["covary"]
    t = _series({n: {}})
    t = spike(t)
    a = t[(t["mz"] - C.ion_mz(n, PAIR[0])).abs() < 1e-6].sort_values("sample_item_id")["height"].to_numpy()
    b = t[(t["mz"] - C.ion_mz(n, PAIR[1])).abs() < 1e-6].sort_values("sample_item_id")["height"].to_numpy()
    assert np.corrcoef(a, b)[0, 1] < NP.COVARY_R_MIN                  # the linear r would veto it


def test_clean_reads_the_iso_child_and_artifact_stamps():
    n = "C12H20O4"
    for role in ("iso_child", "artifact"):
        def restamp(ts, role=role):
            mz = C.ion_mz(n, PAIR[0])
            near = (ts["mz"] - mz).abs() < 1e-6
            ts = ts.copy()
            ts.loc[near, "role"] = role
            return ts
        r = _one({"_ts_hook": restamp})
        assert not r["clean"] and not r["upair"], role


def test_clean_holds_below_a_majority_of_foreign_stamps():
    n = "C12H20O4"
    def partly(ts, share):
        mz = C.ion_mz(n, PAIR[1])
        near = (ts["mz"] - mz).abs() < 1e-6
        ts = ts.copy()
        sids = sorted(ts["sample_item_id"].unique())
        foreign = near & ts["sample_item_id"].isin(sids[:int(share * len(sids))])
        ts.loc[foreign, "neutral_formula"] = "C8H16N4O3"
        ts.loc[foreign, "adduct"] = "[M+H]+"
        return ts
    assert not _one({"_ts_hook": lambda t: partly(t, 0.7)})["clean"]
    assert _one({"_ts_hook": lambda t: partly(t, 0.3)})["clean"]


def test_a_13C_count_off_by_a_quarter_contradicts():
    """|n_obs - n| > max(1, 0.2 n): a C13 ion read as ~9.75 C contradicts, one
    read as ~12 C does not."""
    assert _one({"c13": 0.75}, n="C12H20O4")["c13_contradicts"]
    assert not _one({"c13": 0.93}, n="C12H20O4")["c13_contradicts"]


# --------------------------------------------------------------------------- composition
def test_a_composed_profile_keeps_one_declared_pair_and_drops_two():
    import dataclasses
    ur, easy = P.PROFILES["Ur"], P.PROFILES["EasyIC"]
    assert P.compose([ur, easy]).neutral_pair == PAIR                  # one component declares it
    assert P.compose([easy, ur]).neutral_pair == PAIR
    other = dataclasses.replace(ur, name="UrX", neutral_pair=("[M+H]+", "[M+NH4]+"), aliases=())
    assert P.compose([ur, other]).neutral_pair == ()                    # two different pairs: no rule
    assert P.compose([ur, dataclasses.replace(ur, name="UrY", aliases=())]).neutral_pair == PAIR
    assert P.compose([easy, P.PROFILES["NH4_15N"]]).neutral_pair == ()  # none declares one


# --------------------------------------------------------------------------- the order of the rows
def test_a_corroborated_pair_keeps_its_outside_axis():
    """Row 9' sat after row 9: a pair that two axes and an outside one (the
    corroborating source) already backed kept that reading -- the merge vote's
    class 2 -- and the private reason names the source, not the pair."""
    frame = _pair_rows(iso=True)
    out = EV._series_pooled({"f": frame}, cross={"C10H16O4"}, upair={"C10H16O4"})
    assert out["corroborated"].all() and out["upair"].all() and out["cross"].all()
    assert {EV._vote_class_of(a, b) for a, b in zip(out.evidence_level, out.evidence_axes)} == {2}
    assert all("neutral pair" not in r for r in out.level_reason)


# --------------------------------------------------------------------------- the reference script, live
def _ll():
    """The reference script with the pre-0.10.0 decision, or skip."""
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location(
        "level_ledger", Path(__file__).resolve().parents[1] / "scripts" / "level_ledger.py")
    LL = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(LL)
    if not all(hasattr(LL, x) for x in ("series_run", "series_main", "measure_source", "assign_levels")):
        pytest.skip("scripts/level_ledger.py no longer carries the pre-0.10.0 decision")
    return LL


def test_the_reference_script_levels_the_uronium_set_like_the_engine(tmp_path):
    """Live, not through expected_levels.csv: the script with the fixture pair
    table against the engine, row for row."""
    from tests.test_evidence import FIXTURES
    LL = _ll()
    d = tmp_path / "ur" / "per_file"
    d.mkdir(parents=True)
    for p in sorted(FIXTURES.glob("ur_*_ledger.csv.gz")):
        pd.read_csv(p, low_memory=False).to_csv(d / p.name[:-3], index=False)
    table = FIXTURES / "ur_neutral_pairs.csv"
    ref = LL.series_run([str(tmp_path / "ur")], [], str(table))
    t = pd.read_csv(table)
    core = EV._series_pooled(_pooled("ur"), upair=set(t.loc[t["upair"].astype(bool), "neutral_formula"]))
    m = core.merge(ref, left_on=["neutral_formula", "adduct"], right_on=["neutral", "adduct"])
    assert len(m) == len(core) == len(ref) == GOLDEN["ur"][0]
    assert (m["evidence_level"] == m["level"]).all()
    assert (m["multiline_elements_x"].fillna("") == m["multiline_elements_y"].fillna("")).all()


def test_the_reference_script_honours_the_verdict_and_the_cli(tmp_path):
    LL = _ll()
    run = tmp_path / "out" / "RUN_1"
    (run / "per_file").mkdir(parents=True)
    (run / "tables").mkdir()
    rows = pd.concat([_pair_rows("C10H16O4", iso=True), _pair_rows("C9H14O4", iso=True)])
    rows["peak_id"] = [f"q{i}" if r.role == "M0" else f"c{i}" for i, r in enumerate(rows.itertuples())]
    # re-link the children to their parents after the re-numbering
    rows = rows.reset_index(drop=True)
    for i in range(len(rows)):
        if rows.at[i, "role"] == "iso_child":
            rows.at[i, "parent_peak_id"] = rows.at[i - 1, "peak_id"]
    rows.to_csv(run / "per_file" / "s1_ledger.csv", index=False)
    pd.DataFrame({"neutral_formula": ["C10H16O4", "C9H14O4"], "upair": [True, False]}).to_csv(
        run / "tables" / "neutral_pairs.csv", index=False)
    got = LL.series_run([str(run)], [], "auto").set_index("neutral")
    assert got.loc["C10H16O4", "upair"].all() and not got.loc["C9H14O4", "upair"].any()   # the False row: nothing
    # an --out-dir holding one run finds the same table
    assert LL.series_run([str(tmp_path / "out")], [], "auto").set_index("neutral").loc["C10H16O4", "upair"].all()
    out = tmp_path / "levels.csv"
    assert LL.series_main([str(run), "--upair", "--out", str(out)]) == 0
    assert pd.read_csv(out).set_index("neutral").loc["C10H16O4", "upair"].all()
    assert LL.series_main([str(run), "--out", str(out)]) == 0
    assert not pd.read_csv(out).set_index("neutral").loc["C10H16O4", "upair"].any()


# --------------------------------------------------------------------------- the batch, end to end
def test_a_uronium_batch_measures_the_pair_and_lifts_the_rows(tmp_path, monkeypatch):
    """assign_batch.run on the uronium profile: the pair table is written from the
    stamped batch series, the summary carries the funnel, the pooled fact table
    carries `upair` on both rows of each pair, and the scale's record on the
    merged rows prints the two channels as a "two routes" tag (before peaky
    0.10.0 the pair lifted those rows to 4a)."""
    import json
    from peaky.assignment import assign as A
    from peaky.assignment import ledger as L
    from peaky.assignment import tiers as T
    from peaky.batch import assign_batch as AB
    from peaky.io import io_mascope as IO

    neutrals = GOOD
    t0 = pd.Timestamp("2026-08-11 00:00", tz="UTC")
    rows = []
    for i in range(40):
        sid = f"s{i:03d}"
        for k, n in enumerate(neutrals):
            level = 5000.0 * (1.6 + np.sin(2 * np.pi * i / 20.0 + 0.7 * k))
            for adduct, h in ((PAIR[0], level), (PAIR[1], 0.6 * level)):
                rows.append(dict(sample_item_id=sid, sample_item_name=f"n_{sid}",
                                 datetime_utc=t0 + pd.Timedelta(minutes=30 * i), mz=C.ion_mz(n, adduct), height=h))
        rows += [dict(sample_item_id=sid, sample_item_name=f"n_{sid}", datetime_utc=t0 + pd.Timedelta(minutes=30 * i),
                      mz=float(100 + j), height=500.0) for j in range(10)]
    pk = pd.DataFrame(rows)

    def fake_assign(sid, context="uronium", **kw):
        ids = [f"P{k}{j}" for k in range(len(neutrals)) for j in range(2)]
        mzs = [C.ion_mz(n, a) for n in neutrals for a in PAIR]
        led = L.new_ledger(pd.DataFrame({"peak_id": ids, "mz": mzs, "height": [5000.0] * len(ids)}))
        for k, n in enumerate(neutrals):
            for j, a in enumerate(PAIR):
                L.commit_assignment(led, f"P{k}{j}", neutral_formula=n, adduct=a, ion_formula=_ion(n, a) + "+",
                                    ion_score=0.95, compound_score=0.95, ppm_error=0.1, pass_no=1,
                                    method="cheminfo", confidence="High", commentary=f"Pass 1: {n} {a}")
        T.apply_tiers(led)
        led["degeneracy_density"] = 0.5
        led["resolvability"] = "resolved"
        return {"ledger": led, "stats": {"noise_edge_cps": 4.0, "height_gate_cps": 10.0},
                "plausibility_audit": [], "summaries": {}, "problems": []}

    monkeypatch.setattr(IO, "connect", lambda *a, **k: "CLIENT")
    monkeypatch.setattr(IO, "fetch_peaks", lambda client, sid, use_cache=True: pd.DataFrame(
        {"peak_id": ["A"], "mz": [200.1], "height": [1.0e5]}))
    monkeypatch.setattr(IO, "estimate_offset", lambda raw: 0.0)
    monkeypatch.setattr(A, "run", fake_assign)
    AB.run(peaks=pk, ts_peaks=pk, reagent="Ur", batch="test batch", out_dir=str(tmp_path),
           k_min=2, k_max=3, min_gain=0.0, n_jobs=1, resolving_power=100_000, log=lambda *a: None)
    table = pd.read_csv(tmp_path / "tables" / "neutral_pairs.csv")
    assert set(table.loc[table["upair"].astype(bool), "neutral_formula"]) == set(neutrals)
    assert (table["bare"] == PAIR[0]).all() and (table["cluster"] == PAIR[1]).all()
    summ = json.load(open(tmp_path / "batch_summary.json"))["evidence_levels"]["neutral_pairs"]
    assert summ["pair"] == list(PAIR) and summ["upair"] == len(neutrals)
    ev = pd.read_csv(tmp_path / "tables" / "evidence_levels.csv")
    pairs = ev[ev["neutral_formula"].isin(neutrals)]
    assert len(pairs) == 2 * len(neutrals) and pairs["upair"].map(EV.truthy).all()
    merged = pd.read_csv(tmp_path / "merged_ledger.csv", keep_default_na=False)
    mine = merged[merged["neutral_formula"].isin(neutrals)]
    assert len(mine) == 2 * len(neutrals) and "evidence_axes" not in merged.columns
    assert set(mine["evidence_level"]) <= set(EV.LEVELS)
    assert mine["tags"].map(lambda t: "two routes: protonated + urea cluster (" in t).all()
