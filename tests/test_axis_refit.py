"""The sub-ppm lock model is refitted without the locks that disagree with their neighbours.

A sub-ppm lock model (`assign_batch._lock_model`: its largest correction and the locks' raw
median both under 1 ppm) is applied once its held-out locks resolve it. Its smoothing spline
(GCV) is drawn through every non-bright lock, so a few co-located locks that read 0.6-0.9 ppm
off their neighbours (a lock on a line blended with a brighter ion's 13C line, a labelled alias)
bump it by 0.1-0.25 ppm over a few Da -- and at that size the bump moves a true line out of
Good. On that branch only, after the verdict (which stays the as-built model's), each spline
segment is refitted on its non-bright locks less the ones that sit more than LOCK_REFIT_PPM
(0.5) off a Theil-Sen line through their LOCAL_K (6) nearest locks, dropped one at a time, the
worst first (a cluster of high locks would otherwise drag a good neighbour out with it). Line
(walk) segments and every domain stay as built; a model from which nothing is dropped is the
as-built model, byte for byte; the edge carry is read off the kept locks; a model at 1 ppm or
more is never refitted.

Synthetic lock tables, AL.fit stubbed: no peaks, no network.
"""
import json

import numpy as np
import pandas as pd
import pytest

from peaky.batch import assign_batch as AB
from peaky.batch import axislock as AL
from peaky.batch import massqc as MQ
from peaky.io import io_mascope as IO

BUMP = {259.05: 0.65, 263.077: 0.6, 263.09: 0.9, 267.109: 0.8}


@pytest.fixture(autouse=True)
def _no_axis_left_behind():
    IO.clear_axis_correction()
    yield
    IO.clear_axis_correction()


def _quiet(*a, **k):
    return None


def _axis(m):
    """A gentle sub-ppm axis: +0.35 ppm at m/z 120 down to -0.21 at 400."""
    return 0.35 - 0.002 * (np.asarray(m, dtype=float) - 120.0)


def _table(off=None, *, seed=3, n=160, lo=120.0, hi=400.0, bright=()):
    """A lock table (axislock's rows: mz / ion / ppm / how / bright): `n` locks on the axis with
    0.05 ppm scatter, plus `off` {mz: ppm off the axis}; `bright` {mz: ppm off} bright locks."""
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(lo, hi, n))
    y = _axis(x) + rng.normal(0.0, 0.05, n)
    rows = [dict(mz=float(a), ion=f"L{i}", ppm=float(b), how="unique +-5 ppm", bright=False)
            for i, (a, b) in enumerate(zip(x, y))]
    for k, (m, d) in enumerate((off or {}).items()):
        rows.append(dict(mz=float(m), ion=f"X{k}", ppm=float(_axis(m) + d), how="unique +-5 ppm", bright=False))
    for k, (m, d) in enumerate(dict(bright).items()):
        rows.append(dict(mz=float(m), ion=f"B{k}", ppm=float(_axis(m) + d), how="unique +-5 ppm", bright=True))
    t = pd.DataFrame(rows).sort_values("mz").reset_index(drop=True)
    t["mz"] = t["mz"].round(5)                   # as axislock._lock_table writes it
    t["ppm"] = t["ppm"].round(3)
    return t


def _built(table, line=None):
    """The model fit() builds: a GCV spline through every non-bright lock on the 0.5 Da grid (plus
    an optional line segment above it, `line` = (lo, hi, a, b))."""
    t = table[~table["bright"]]
    if line is not None:
        t = t[t["mz"] < line[0]]
    x, y = t["mz"].to_numpy(float), t["ppm"].to_numpy(float)
    cur = AL._Curve(x, y)
    grid = np.unique(np.concatenate([np.arange(x[0], x[-1], 0.5), [x[-1]]]))
    segs = [{"kind": "spline", "lo": float(x[0]), "hi": float(x[-1]), "n": len(x), "x": grid, "y": cur(grid)}]
    if line is not None:
        segs.append({"kind": "line", "lo": line[0], "hi": line[1], "n": 8, "a": line[2], "b": line[3]})
    return AL.AxisModel(segments=segs, n_locks=len(table))


def _info(table, cv=0.09, raw=0.3, peak=0.6):
    return {"n_pass1": len(table), "n_links": 0, "n_narrow": 0, "n_off_curve": 0, "locks": table.to_dict("records"),
            "cv_ppm": cv, "raw_ppm": raw, "max_abs_ppm": peak}


def _measure(monkeypatch, model, table, **kw):
    monkeypatch.setattr(AL, "fit", lambda *a, **k: (AL.AxisModel.from_dict(model.as_dict()), _info(table, **kw)))
    logs = []
    _, got, _ = AB.measure_axis(pd.DataFrame({"sample_item_id": ["s0", "s1"], "mz": [150.0, 150.0],
                                              "height": [1e4, 1e4]}), "NO3", "orbitrap", log=logs.append)
    return got, logs


# --------------------------------------------------------------------------- the refit itself
def test_the_constants():
    """0.5 ppm: 0.4 starves the GCV fit of locks on a real batch (its lambda fell 297 -> 47),
    0.75 left the bump that moved a true line out of Good; k is the lock picker's own."""
    assert AB.LOCK_REFIT_PPM == 0.5
    assert AL.LOCAL_K == 6


def test_co_located_off_locks_bump_the_spline_and_the_refit_drops_them():
    table = _table(BUMP)
    built = _built(table)
    bump = float(built.predict([263.1])[0] - _axis(263.1))
    assert bump > 0.2                                   # the as-built spline follows them
    model, dropped = built.refit(table, off_ppm=AB.LOCK_REFIT_PPM)
    assert sorted(table.loc[dropped, "mz"]) == pytest.approx(sorted(BUMP))
    q = np.array([240.0, 255.0, 263.1, 265.0, 270.0, 290.0])
    assert np.abs(model.predict(q) - _axis(q)).max() < 0.04
    seg, seg0 = model.segments[0], built.segments[0]
    assert (seg["lo"], seg["hi"]) == (seg0["lo"], seg0["hi"]) and np.array_equal(seg["x"], seg0["x"])
    assert seg["n"] == seg0["n"] - len(BUMP) and seg["refit_dropped"] == len(BUMP)
    assert built.predict([263.1])[0] - _axis(263.1) == pytest.approx(bump)   # the input is not mutated


def test_the_worst_lock_goes_first_so_a_good_neighbour_stays():
    """Three high locks (+1.2 ppm) beside a good one: dropped one at a time, worst first, only the
    three go -- judged all at once their pull also flags the good lock beside them and two more."""
    x = np.arange(200.0, 400.0, 3.0)
    x = np.sort(np.r_[x[(x < 299) | (x > 303)], 300.2, 300.7, 301.2, 301.8])
    high = np.isin(x, [300.2, 300.7, 301.2])
    keep = AL._worst_first(x, np.where(high, 1.2, 0.0), 0.5)
    assert list(x[~keep]) == [300.2, 300.7, 301.2]


def test_a_model_with_nothing_to_drop_is_the_as_built_model():
    table = _table()
    built = _built(table)
    model, dropped = built.refit(table, off_ppm=AB.LOCK_REFIT_PPM)
    assert dropped == [] and json.dumps(model.as_dict()) == json.dumps(built.as_dict())


def test_bright_locks_and_line_segments_are_not_refitted():
    """A bright lock 2 ppm off is no spline lock (fit() leaves it out); a line segment's locks
    and line stay as built, its off lock included."""
    table = _table({**BUMP, 410.0: 1.5}, bright={250.0: 2.0}, hi=400.0)
    extra = pd.DataFrame([dict(mz=m, ion=f"W{i}", ppm=-0.3, how="mass difference", bright=False)
                          for i, m in enumerate((404.0, 406.0, 408.0, 412.0, 414.0, 416.0, 418.0))])
    table = pd.concat([table, extra], ignore_index=True).sort_values("mz").reset_index(drop=True)
    built = _built(table, line=(404.0, 418.0, -0.3, 0.0))
    model, dropped = built.refit(table, off_ppm=AB.LOCK_REFIT_PPM)
    assert sorted(table.loc[dropped, "mz"]) == pytest.approx(sorted(BUMP))
    assert model.segments[1] == built.segments[1]


def test_a_refit_steeper_than_the_slope_bound_keeps_the_as_built_segment(monkeypatch):
    table = _table(BUMP)
    built = _built(table)
    monkeypatch.setattr(AL, "SPLINE_MAX_SLOPE", 1e-6)
    model, dropped = built.refit(table, off_ppm=AB.LOCK_REFIT_PPM)
    assert dropped == [] and json.dumps(model.as_dict()) == json.dumps(built.as_dict())


# --------------------------------------------------------------------------- the batch's verdict and record
def test_the_sub_ppm_branch_applies_the_refitted_model_under_the_as_built_verdict(monkeypatch):
    table = _table(BUMP)
    built = _built(table)
    got, logs = _measure(monkeypatch, built, table)
    assert got["applied"] and got["verdict"] == "axis_trend"
    assert got["max_abs_ppm"] == 0.6                       # the verdict's (as built), as fit reported it
    applied = MQ.fit_from_record(got["wave"])
    assert abs(float(applied.predict([263.1])[0] - _axis(263.1))) < 0.04
    rec = got["refit"]
    assert rec["off_ppm"] == AB.LOCK_REFIT_PPM and rec["k"] == AL.LOCAL_K and rec["n_dropped"] == len(BUMP)
    assert sorted(d["mz"] for d in rec["dropped"]) == pytest.approx(sorted(BUMP))
    assert all(d["off_ppm"] > AB.LOCK_REFIT_PPM for d in rec["dropped"])
    assert rec["max_abs_ppm_built"] == pytest.approx(round(built.max_abs_ppm(), 3))
    assert applied.stats["refit"] == rec
    assert any("[mass-axis] refit" in str(m) for m in logs)
    # the edge carry is read off the kept locks
    kept = table.drop(index=[i for i in table.index if table.at[i, "mz"] in set(BUMP)])
    want = applied.edge_carry_record(AB.EDGE_CARRY_DA, locks=kept)
    for end in ("below", "above"):
        assert got["edge_carry"][end]["ppm"] == pytest.approx(want[end]["ppm"], abs=1e-4)
        assert got["edge_carry"][end]["n_locks"] == want[end]["n_locks"]


def test_the_edge_carry_reads_the_kept_locks_at_an_end(monkeypatch):
    """Two off locks among the first lock's neighbours: dropped, they no longer move the end's value."""
    table = _table({120.1: 0.0, 120.6: 0.0, 121.6: 0.0, 123.3: 0.0, 124.1: 0.0, 121.0: 0.9, 122.5: 0.95})
    built = _built(table)
    got, _ = _measure(monkeypatch, built, table)
    assert got["refit"]["n_dropped"] == 2
    lo = got["edge_carry"]["below"]
    assert lo["ppm"] == pytest.approx(float(_axis(lo["mz"])), abs=0.05)
    full = MQ.fit_from_record(got["wave"]).edge_carry_record(AB.EDGE_CARRY_DA, locks=table)["below"]
    assert lo["n_locks"] >= 5 and full["n_locks"] == lo["n_locks"] + 2


def test_nothing_dropped_leaves_the_applied_wave_as_built(monkeypatch):
    table = _table()
    built = _built(table)
    got, _ = _measure(monkeypatch, built, table)
    assert got["applied"] and got["refit"]["n_dropped"] == 0 and got["refit"]["dropped"] == []
    w = dict(got["wave"])
    stats = dict(w.pop("stats"))
    b = built.as_dict()
    b.pop("stats")
    assert w == json.loads(json.dumps(b))
    assert stats["refit"]["n_dropped"] == 0


@pytest.mark.parametrize("cv,raw,peak,applied", [
    (0.1, 0.5, 1.2, True),           # a model at 1 ppm or more: the reference bar's, never refitted
    (0.1, 1.1, 0.9, True),           # the locks' raw median at 1 ppm or more: never
    (0.1, 0.2, 0.35, False),         # not applied at all
])
def test_only_the_sub_ppm_branch_is_refitted(monkeypatch, cv, raw, peak, applied):
    table = _table(BUMP)
    built = _built(table)
    got, _ = _measure(monkeypatch, built, table, cv=cv, raw=raw, peak=peak)
    assert bool(got.get("applied")) == applied and "refit" not in got
    if applied:
        assert MQ.fit_from_record(got["wave"]).segments[0]["y"] == pytest.approx(built.segments[0]["y"], abs=1e-5)
