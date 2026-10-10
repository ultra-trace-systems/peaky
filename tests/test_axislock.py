"""batch.axislock: the batch's m/z axis from its own peaks -- calibration-free locks,
the smooth curve, the steps an Orbitrap's axis can take -- and its wiring into
`peaky batch --mass-axis` (measure_axis, the io registry, publish's way back to the
server's axis). Every batch here is synthetic: homologous ion series on a known
axis error, with their 13C lines and non-recurring noise."""
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from peaky.batch import assign_batch as AB
from peaky.batch import axislock as AL
from peaky.batch import massqc as MQ
from peaky.chem import reference_ions as RI
from peaky.io import io_mascope as IO
from tests.test_massqc import _orbi_batch

STEP1, STEP2 = 364.6, 394.0


@pytest.fixture(autouse=True)
def _no_axis_left_behind():
    IO.clear_axis_correction()
    yield
    IO.clear_axis_correction()


def _quiet(*a, **k):
    return None


def _smooth(mz):
    mz = np.asarray(mz, dtype=float)
    return 2.4 * np.exp(-((mz - 205.0) / 85.0) ** 2) - 0.03 * np.maximum(mz - 300.0, 0.0)


def _stepped(mz):
    """A hump peaking +2.4 ppm near m/z 205, then a -2.6 ppm step at STEP1 and a
    +2.2 ppm step at STEP2 -- the shape measured on one Orbitrap."""
    mz = np.asarray(mz, dtype=float)
    return _smooth(mz) - 2.6 * (mz > STEP1) + 2.2 * (mz > STEP2)


def _flat(mz):
    return np.zeros(np.shape(mz))


def _truth(seed=0):
    """Closed-shell anions in homologous families: CnH(2n+N+1-2DBE)NmOk seeds, each
    walked by CH2 and O steps -- a nitrate-CIMS-like spectrum, m/z 60-440."""
    rng = np.random.default_rng(seed)
    ions = set()
    for _ in range(46):
        c, n, o, dbe = int(rng.integers(2, 12)), int(rng.integers(0, 4)), int(rng.integers(2, 9)), int(rng.integers(1, 5))
        for dc in range(0, 6):
            for do in range(0, 5):
                cc, oo = c + dc, o + do
                h = 2 * cc + n + 1 - 2 * dbe
                if h < 1 or oo > 2 * cc + 3 * n + 2:
                    continue
                ion = AL.name({"C": cc, "H": h, "N": n, "O": oo})
                if 60 < AL.exact_mz(ion) < 440:
                    ions.add(ion)
    ions = sorted(ions)
    keep = rng.random(len(ions)) < 0.55
    return [i for i, k in zip(ions, keep) if k]


def _batch(axis=_stepped, *, n_files=6, seed=0, truth=None):
    rng = np.random.default_rng(seed + 100)
    truth = truth or _truth(seed)
    base = {ion: float(np.exp(rng.normal(np.log(400.0), 1.1))) for ion in truth}
    rows = []
    for f in range(n_files):
        sid = f"s{f}"
        fac = rng.uniform(0.7, 1.3)
        for ion, h0 in base.items():
            h = h0 * fac
            if h < 12.0:
                continue
            th = AL.exact_mz(ion)
            mz = th * (1 + (float(axis(np.array([th]))[0]) + rng.normal(0, 0.08)) * 1e-6)
            rows.append((sid, mz, h, h * 1e-3))
            nc = AL.parse(ion)["C"]
            h13 = h * nc * AL.R13 * rng.uniform(0.95, 1.05)
            if h13 >= 12.0:
                mz13 = (th + AL.D13C) * (1 + (float(axis(np.array([th + 1]))[0]) + rng.normal(0, 0.15)) * 1e-6)
                rows.append((sid, mz13, h13, h13 * 1e-3))
        for _ in range(250):                     # noise: never at the same m/z twice
            h = rng.uniform(10.0, 30.0)
            rows.append((sid, rng.uniform(55.0, 450.0), h, h * 1e-3))
    ts = pd.DataFrame(rows, columns=["sample_item_id", "mz", "height", "area"])
    return ts, set(truth)


@pytest.fixture(scope="module")
def stepped():
    ts, truth = _batch(_stepped)
    model, info = AL.fit(ts, polarity="-")
    return ts, truth, model, info


# --- the formula space --------------------------------------------------------------

def test_the_space_holds_closed_shell_ions_and_no_radicals():
    sp = AL.FormulaSpace("-")
    assert "NO3-" in set(sp.near(AL.exact_mz("NO3-"), 1.0).ion)
    assert "HO4S-" in set(sp.near(AL.exact_mz("HO4S-"), 1.0).ion)
    assert "C3H3O4-" in set(sp.near(AL.exact_mz("C3H3O4-"), 1.0).ion)
    # CO3- is a radical anion: (H + N) even
    assert "CO3-" not in set(sp.near(AL.exact_mz("CO3-"), 1.0).ion)
    pos = AL.FormulaSpace("+")
    assert "H4N+" in set(pos.near(AL.exact_mz("H4N+", "+"), 2.0).ion)
    assert abs(AL.exact_mz("H4N+", "+") - (AL.exact_mz("H4N-", "-") - 2 * AL.ME)) < 1e-12


def test_monoisotopic_elements_are_never_lockable_but_the_reagents_own_iodine_is():
    assert not AL.FormulaSpace("-").confirmable({"C": 2, "F": 3})
    assert not AL.FormulaSpace("-").confirmable({"C": 2, "I": 1})
    assert AL.FormulaSpace("-", reagent_elements=("I",)).confirmable({"C": 2, "I": 1})


# --- the model on a stepped axis -----------------------------------------------------

def test_every_lock_is_a_true_formula_and_no_isotopologue_is_locked(stepped):
    """The 13C lines recur in every file at the axis's error, and no monoisotopic
    formula is theirs: before isotopologues were kept out, ~60 of them locked
    as aromatic hydrocarbons (C14H9O2-, C22H25- ...) unique within +-1 ppm."""
    ts, truth, model, info = stepped
    locks = pd.DataFrame(info["locks"])
    assert len(locks) >= 150, len(locks)
    wrong = locks[~locks["ion"].isin(truth)]
    assert wrong.empty, wrong
    assert info["n_iso_children"] > 100


def test_both_steps_are_found_between_the_locks_that_bracket_them(stepped):
    _, _, model, info = stepped
    assert [s["kind"] for s in model.segments] == ["spline", "line", "line"]
    s1, s2 = info["steps"]
    assert s1["between"][0] < STEP1 < s1["between"][1] and s1["ppm"] < -1.5
    assert s2["between"][0] < STEP2 < s2["between"][1] and s2["ppm"] > 1.5
    assert info["joins"] == []


def test_the_model_puts_the_true_ions_it_did_not_lock_on_the_axis(stepped):
    ts, truth, model, info = stepped
    locked = set(pd.DataFrame(info["locks"])["ion"])
    th = np.array([AL.exact_mz(i) for i in sorted(truth - locked)])
    th = th[model.in_scope(th * (1 + _stepped(th) * 1e-6))]      # right up to the steps
    measured = th * (1 + _stepped(th) * 1e-6)
    resid = _stepped(th) - model.predict(measured, extrapolate=True)
    assert len(th) > 20
    assert np.median(np.abs(resid)) < 0.15 and np.mean(np.abs(resid) <= 0.5) > 0.9
    assert info["cv_ppm"] < 0.5 * info["raw_ppm"]


def test_the_model_round_trips_through_json_and_the_registry(stepped):
    ts, _, model, _ = stepped
    rec = json.loads(json.dumps(model.as_dict()))
    back = MQ.fit_from_record(rec)
    grid = np.linspace(70, model.mz_range[1] - 1, 500)
    assert np.allclose(back.predict(grid, extrapolate=True), model.predict(grid, extrapolate=True),
                       rtol=0, atol=1e-4)
    IO.set_axis_correction(["s0"], rec)
    assert isinstance(IO.axis_correction("s0"), AL.AxisModel)
    mz = ts["mz"].to_numpy()[:200]
    # the record keeps the spline's knots to 1e-5 ppm: 1e-7 Da at these masses
    assert np.allclose(MQ.apply_correction(IO.axis_correction("s0"), mz),
                       MQ.apply_correction(model, mz), rtol=0, atol=1e-7)


def test_outside_its_reach_the_model_leaves_the_axis_alone(stepped):
    _, _, model, _ = stepped
    lo, hi = model.mz_range
    far = np.array([lo * 0.9, hi + 1.0, 700.0])
    assert not model.in_scope(far).any()
    assert np.array_equal(MQ.apply_correction(model, far), far)


def test_main_body_splits_at_a_step_but_not_on_a_steep_smooth_rise():
    x = np.arange(60.0, 400.0, 2.0)
    rise = 1.5 * np.tanh((x - 70) / 15)                   # +-1.5 ppm over ~40 Da: smooth
    lo, hi = AL.main_body(x, rise)
    assert (lo, hi) == (60.0, 398.0)
    stepped = np.where(x > 300, -2.6, 0.0) + 0.02 * np.sin(x)
    lo, hi = AL.main_body(x, stepped)
    assert lo == 60.0 and 298.0 <= hi <= 300.0


# --- measure_axis: when the lock model is used, held or not ----------------------------

def test_auto_corrects_an_orbitrap_with_the_lock_model(stepped):
    ts, _, _, _ = stepped
    out, info, table = AB.measure_axis(ts, "NO3", "orbitrap", log=_quiet)
    assert info["model"] == "locks" and info["verdict"] == "axis_steps" and info["applied"]
    assert table is None and len(info.pop("_locks_table")) == info["wave"]["n_locks"]
    moved = out[AB.AXIS_COL].to_numpy()
    assert np.allclose(out["mz"], ts["mz"] * (1 - moved * 1e-6), rtol=0, atol=1e-9)
    json.dumps(info)                                        # the summary record is JSON-safe


def test_a_flat_axis_is_left_alone():
    ts, _ = _batch(_flat, seed=3)
    out, info, _ = AB.measure_axis(ts, "NO3", "orbitrap", log=_quiet)
    assert info["model"] == "locks" and info["verdict"] == "axis_ok" and not info["applied"]
    assert AB.AXIS_COL not in out.columns


def test_too_few_locks_falls_back_to_the_reference_ions_in_auto_only():
    rng = np.random.default_rng(22)
    ts = _orbi_batch(rng, RI.nitrate())
    _, info, table = AB.measure_axis(ts, "NO3", "orbitrap", log=_quiet)
    assert info["model"] == "reference" and table is not None
    assert "fewer than" in info["locks"]["why"]
    _, info, table = AB.measure_axis(ts, "NO3", "orbitrap", mode="locks", log=_quiet)
    assert not info["applied"] and table is None and info.get("model") is None


def test_reference_mode_skips_the_locks_and_a_tof_never_uses_them(stepped):
    ts, _, _, _ = stepped
    _, info, _ = AB.measure_axis(ts, "NO3", "orbitrap", mode="reference", log=_quiet)
    assert info.get("model") == "reference" and "locks" not in info
    _, info, _ = AB.measure_axis(ts, "NO3", "tof", log=_quiet)
    assert "locks" not in info and not info["applied"]


def test_a_hold_measures_the_lock_model_without_applying_it(stepped):
    ts, _, _, _ = stepped
    out, info, _ = AB.measure_axis(ts, "NO3", "orbitrap", hold="because", log=_quiet)
    assert info["model"] == "locks" and info["held"] == "because" and not info["applied"]
    assert AB.AXIS_COL not in out.columns


def test_the_lock_cap_holds_an_axis_too_far_off(monkeypatch, stepped):
    """A model whose smooth curve reaches past LOCK_MAX_CORRECTION_PPM is measured,
    not applied: the PDF report matches the ledger to the uncorrected series
    within 8 ppm."""
    ts, _, model, info = stepped
    d = model.as_dict()
    d["segments"][0]["y"] = [y + AB.LOCK_MAX_CORRECTION_PPM for y in d["segments"][0]["y"]]
    body = AL.AxisModel.from_dict(d)              # the smooth curve itself passes the cap
    far = dict(info, max_abs_ppm=body.max_abs_ppm())
    monkeypatch.setattr(AL, "fit", lambda *a, **k: (body, dict(far)))
    out, got, _ = AB.measure_axis(ts, "NO3", "orbitrap", log=_quiet)
    assert got["model"] == "locks" and not got["applied"] and "broken calibration" in got["held"]
    assert AB.AXIS_COL not in out.columns


# --- the way back to the server's axis --------------------------------------------------

def test_to_server_axis_inverts_the_correction_everywhere_it_reaches(stepped):
    """Densely beside every segment edge and step: a fixed-point inversion over the
    whole model returned a corrected m/z unchanged past the top edge (3.9 ppm) and
    took the wrong side of a step; it now inverts segment by segment."""
    _, _, model, _ = stepped
    info = {"applied": True, "wave": model.as_dict()}
    edges = [e for s in model.segments for e in (s["lo"], s["hi"])]
    raw = np.unique(np.concatenate([np.linspace(e - 0.004, e + 0.004, 801) for e in edges]
                                   + [np.linspace(60, 440, 4001)]))
    corr = MQ.apply_correction(model, raw)
    back = AB.to_server_axis(corr, info)
    inside = model.in_scope(raw)
    assert np.max(np.abs(back - raw)[inside] / raw[inside] * 1e6) < 1e-3
    # outside the model nothing moved, and nothing comes back moved
    assert np.array_equal(corr[~inside], raw[~inside])
    near_edge = np.min(np.abs(raw[:, None] - np.array(edges)[None, :]), axis=1) < 0.01
    assert np.array_equal(back[~inside & ~near_edge], raw[~inside & ~near_edge])
    assert np.array_equal(AB.to_server_axis(corr, {"applied": False}), corr)


def test_the_gap_around_a_step_is_left_uncorrected(stepped):
    """Corrected by either side, a peak between two segments was moved the wrong way
    (raw -1.8 ppm -> +2.6) whenever the step was not where the mid-point said."""
    _, _, model, info = stepped
    assert len(info["uncorrected_gaps"]) == len(info["steps"]) == len(model.segments) - 1
    for lo, hi in info["uncorrected_gaps"]:
        inner = np.linspace(lo, hi, 50)[1:-1]
        inner = inner[(inner > lo * (1 + 25e-6)) & (inner < hi * (1 - 25e-6))]
        assert not model.in_scope(inner).any()
        assert np.array_equal(MQ.apply_correction(model, inner), inner)


def test_the_cap_reads_the_correction_over_the_whole_model(stepped):
    """The cap was compared with the model at its locks only; the correction it then
    applied reached past it (5.36 read, 5.61 applied)."""
    ts, _, model, info = stepped
    grid = np.linspace(model.mz_range[0], model.mz_range[1], 20000)
    applied = np.abs(model.predict(grid[model.in_scope(grid)], extrapolate=True)).max()
    assert info["max_abs_ppm"] >= applied - 1e-3


def _odd_ions(lo, hi):
    """Odd-electron anions (RO2.NO3- style): (H + N) even, outside the closed-shell space."""
    out = []
    for c in range(6, 22):
        for h in range(5, 2 * c + 4):
            for n in range(0, 4):
                if (h + n) % 2:
                    continue
                for o in range(4, 16):
                    if o <= 2 * c + 3 * n + 2:
                        ion = AL.name({"C": c, "H": h, "N": n, "O": o})
                        if lo < AL.exact_mz(ion) < hi:
                            out.append(ion)
    return out


@pytest.mark.parametrize("seed", [1, 5])
def test_ions_outside_the_space_above_a_step_do_not_carry_the_curve_across_it(seed):
    """80 odd-electron ions above the step leave room for coincidental locks that sit
    on the UN-stepped curve: three of them bridged a 14 Da stretch with no true lock,
    the spline ran across the step, the cross-validation passed and the true ions
    above it were left ~3 ppm off. The yield cut, the in-segment step test and the
    gap rule stop it."""
    rng = np.random.default_rng(seed + 50)
    base = _truth(seed)
    truth = base + list(rng.choice(_odd_ions(355, 440), 80, replace=False))
    ts, _ = _batch(_stepped, seed=seed, truth=truth)
    model, info = AL.fit(ts, polarity="-", log=_quiet)
    locks = pd.DataFrame(info["locks"])
    th = np.array([AL.exact_mz(i) for i in locks["ion"]])
    assert (locks["ion"].isin(truth) & (np.abs(locks["ppm"] - _stepped(th)) < 0.6)).all()
    T = np.array(sorted(AL.exact_mz(i) for i in base))
    meas = T * (1 + _stepped(T) * 1e-6)
    err = (MQ.apply_correction(model, meas) - T) / T * 1e6
    assert not (model.in_scope(meas) & (np.abs(err) > 1.0)).any()
    assert model.segments[0]["hi"] < STEP1


def test_a_13c_line_across_a_step_is_never_locked():
    """An ion just below the step puts its 13C line just above it: the line sits 2.6
    ppm off the 13C spacing, outside a 2 ppm isotopologue window, and locked as a
    wrong monoisotopic formula (C25H33O2-)."""
    extra = ["C16H34N3O6-", "C15H30N3O7-"]
    for seed in (0, 1):
        truth = _truth(seed) + extra
        ts, _ = _batch(_stepped, seed=seed, truth=truth)
        _, info = AL.fit(ts, polarity="-", log=_quiet)
        locks = pd.DataFrame(info["locks"])
        assert locks["ion"].isin(truth).all(), locks[~locks["ion"].isin(truth)]


@pytest.mark.parametrize("seed", [0, 5])
def test_a_smooth_axis_reports_no_step_and_bridges_its_joins(seed):
    """Where the walk merely resumed past a sparse stretch, the log printed 'STEP
    -0.03 ppm ... no smooth curve follows it' and the verdict read axis_steps; and,
    once gaps were left uncorrected, such a join was too (a real batch lost 4 ions)."""
    ts, _ = _batch(_smooth, seed=seed)
    _, info, _ = AB.measure_axis(ts, "NO3", "orbitrap", log=_quiet)
    assert info["locks"]["steps"] == [] and info["verdict"] == "axis_trend"
    assert info["locks"]["uncorrected_gaps"] == [] and info["locks"]["holes"] == []
    if seed == 0:
        assert info["locks"]["joins"]                      # this one resumes past a sparse stretch
    model = MQ.fit_from_record(info["wave"])
    lo, hi = model.mz_range
    grid = np.linspace(lo, hi, 4000)
    assert model.in_scope(grid).all()


def test_a_heavy_line_needs_to_recur_at_the_predicted_ratio():
    """One stray peak at the 34S position in one spectrum 'confirmed' S."""
    row = SimpleNamespace(f_S=1 / 9, r_S=0.045, f_Cl=0.0, r_Cl=np.nan, f_Br=0.0, r_Br=np.nan)
    assert not AL._line_seen(row, "S", 1)
    row = SimpleNamespace(f_S=8 / 9, r_S=0.045, f_Cl=0.0, r_Cl=np.nan, f_Br=0.0, r_Br=np.nan)
    assert AL._line_seen(row, "S", 1) and not AL._line_seen(row, "S", 3)
    row = SimpleNamespace(f_S=8 / 9, r_S=0.30, f_Cl=0.0, r_Cl=np.nan, f_Br=0.0, r_Br=np.nan)
    assert not AL._line_seen(row, "S", 1)                   # 0.30 is a 37Cl ratio, not 34S


def test_a_structurally_impossible_link_never_locks():
    sp = AL.FormulaSpace("-")
    assert sp.valid(AL.parse("C3H3O4-")) and not sp.valid(AL.parse("C3H4O4-"))   # radical
    assert not sp.valid(AL.parse("C2H9O4-"))                                      # DBE < 0
    assert not sp.valid(AL.parse("CH1O9-"))                                       # O ceiling
    assert not sp.valid(AL.parse("C41H3O4-"))                                     # C range


def test_publish_reads_a_corrected_run_back_on_the_servers_axis(tmp_path, stepped):
    from peaky.io import publish as P
    _, _, model, _ = stepped
    raw = np.array([150.0, 250.0, 380.0, 410.0])
    corr = MQ.apply_correction(model, raw)
    pd.DataFrame({"mz": corr, "neutral_formula": ["C3H4O4"] * 4, "adduct": ["[M-H]-"] * 4,
                  "tier": ["Assigned"] * 4}).to_csv(tmp_path / "merged_ledger.csv", index=False)
    (tmp_path / "batch_summary.json").write_text(json.dumps(
        {"mass_axis": {"applied": True, "model": "locks", "wave": model.as_dict()}}))
    got = P.load_batch_run(str(tmp_path))["merged"]["mz"].to_numpy()
    assert np.max(np.abs(got - raw) / raw * 1e6) < 1e-3


def test_the_cli_offers_every_mode():
    from peaky.cli import build_parser
    for mode in AB.MASS_AXIS_MODES:
        a = build_parser().parse_args(["batch", "--batch", "b", "--dataset", "d", "--mass-axis", mode])
        assert a.mass_axis == mode


# --- pinned by mutation testing: each test below kills a mutant the suite above let live ----

def test_the_dbe_rule_excludes_oversaturated_ions():
    assert "CH5O-" not in set(AL.FormulaSpace("-").near(AL.exact_mz("CH5O-"), 1.0).ion)
    assert "CH7O+" not in set(AL.FormulaSpace("+").near(AL.exact_mz("CH7O+", "+"), 1.0).ion)


def test_a_flat_four_ppm_axis_is_locked_calibration_free():
    """The +-5 ppm window is what lets an axis 2-5 ppm off build a model at all."""
    ts, truth = _batch(lambda m: np.full(np.shape(m), 4.0), seed=7)
    model, info = AL.fit(ts, polarity="-", log=_quiet)
    assert model is not None, info.get("why")
    assert pd.DataFrame(info["locks"])["ion"].isin(truth).all()
    lo, hi = model.mz_range
    assert np.allclose(model.predict(np.linspace(lo + 1, hi - 1, 60), extrapolate=True), 4.0, atol=0.25)


def _inner(mz):
    return np.where(np.asarray(mz, dtype=float) > 330.0, -1.6, 1.2)


def test_a_step_inside_the_lock_window_ends_the_curve_there():
    ts, truth = _batch(_inner, seed=5)
    model, _ = AL.fit(ts, polarity="-", log=_quiet)
    assert model.segments[0]["hi"] < 330.0 < model.segments[1]["lo"]
    th = np.array([AL.exact_mz(i) for i in truth])
    meas = th * (1 + _inner(th) * 1e-6)
    th, meas = th[model.in_scope(meas)], meas[model.in_scope(meas)]
    resid = _inner(th) - model.predict(meas, extrapolate=True)
    assert np.median(np.abs(resid)) < 0.15 and np.mean(np.abs(resid) < 0.5) > 0.9


def _two(m):
    m = np.asarray(m, dtype=float)
    return 0.5 - 2.2 * (m > 300.0) + 2.0 * (m > 340.0)      # a 40 Da dip between two steps


@pytest.mark.parametrize("seed", [12, 13])
def test_two_steps_forty_da_apart_are_both_followed(seed):
    """One line once covered m/z 300-428 across the second step: the walk switched
    lines between rounds and kept the earlier locks; 90 of 539 ions >0.5 ppm off."""
    ts, truth = _batch(_two, seed=seed)
    model, _ = AL.fit(ts, polarity="-", log=_quiet)
    th = np.array(sorted(AL.exact_mz(i) for i in truth))
    meas = th * (1 + _two(th) * 1e-6)
    th, meas = th[model.in_scope(meas)], meas[model.in_scope(meas)]
    resid = _two(th) - model.predict(meas, extrapolate=True)
    assert np.mean(np.abs(resid) < 0.5) > 0.95, [(s["lo"], s["hi"]) for s in model.segments]


def test_a_thirty_da_hole_does_not_end_the_model():
    """The walk looked only SEG_SPAN_DA past its top: a hole that wide in the
    recurring ions left everything above it uncorrected."""
    truth = [i for i in _truth(5) if not 295 < AL.exact_mz(i) < 327]
    ts, truth = _batch(lambda m: np.full(np.shape(m), 2.0), seed=5, truth=truth)
    model, _ = AL.fit(ts, polarity="-", log=_quiet)
    th = np.array([AL.exact_mz(i) for i in truth])
    hi = th[th > 330] * (1 + 2e-6)
    assert model.in_scope(hi).mean() > 0.9


def test_the_first_locks_past_a_step_do_not_bend_the_curve():
    """Seed 3 puts one or two right formulas just past the step -- too few for the
    BREAK_K test -- and the spline bent to them (183 in-scope points >0.5 ppm off)."""
    ts, truth = _batch(_stepped, seed=3)
    model, info = AL.fit(ts, polarity="-", log=_quiet)
    assert model.segments[0]["hi"] < STEP1
    th = np.linspace(60, 440, 8000)
    meas = th * (1 + _stepped(th) * 1e-6)
    err = (MQ.apply_correction(model, meas) - th) / th * 1e6
    assert not (model.in_scope(meas) & (np.abs(err) > 0.5)).any()


@pytest.mark.parametrize("reagent,mode,want_pol,want_els", [
    ("Ur", "locks", "+", ()), ("Br", "auto", "-", ("Br",)), ("NO3", "reference", "-", ())])
def test_the_run_hands_measure_axis_its_mode_polarity_and_reagent(reagent, mode, want_pol, want_els,
                                                                  tmp_path, monkeypatch):
    seen = {}

    def spy(ts, reagent, klass, **kw):
        seen.update(kw)
        raise SystemExit("measured")

    monkeypatch.setattr(AB, "measure_axis", spy)
    ts = pd.DataFrame({"sample_item_id": ["a", "b"] * 5, "mz": np.linspace(100, 300, 10), "height": 100.0})
    monkeypatch.setattr(IO, "connect", lambda *a, **k: None)
    with pytest.raises(SystemExit):
        AB.run(peaks=ts, ts_peaks=ts, reagent=reagent, batch="b", out_dir=str(tmp_path),
               residual=False, n_jobs=1, resolving_power=250000.0, mass_axis=mode, log=_quiet)
    assert seen["mode"] == mode and seen["polarity"] == want_pol
    assert tuple(seen["reagent_elements"]) == want_els


def _row(**kw):
    d = {"r13": 5 * AL.R13, "h": 1e4, "frac": 1.0, "f_S": 0.0, "r_S": np.nan,
         "f_Cl": 0.0, "r_Cl": np.nan, "f_Br": 0.0, "r_Br": np.nan}
    d.update(kw)
    return pd.Series(d)


def test_the_isotope_strikes_and_confirmation_rules():
    bins = AL.Bins(table=pd.DataFrame(), n_spectra=6, floor=10.0, intensity="area")
    cands = pd.DataFrame({"ion": ["c0", "c5", "c5s", "c9"], "C": [0, 5, 5, 9], "S": [0, 0, 1, 0],
                          "Cl": [0, 0, 0, 0], "Br": [0, 0, 0, 0]})
    # a 13C line for 5 carbons strikes the carbon-free and the 9-carbon rival; the S one
    # goes too, its 34S line (4.5 % of 1e4, over the floor) being absent
    assert list(AL._strike(cands, _row(), bins).ion) == ["c5"]
    assert list(AL._strike(cands, _row(f_S=1.0, r_S=0.045), bins).ion) == ["c5", "c5s"]
    # no 13C line where one carbon would show: carbon-free only
    assert list(AL._strike(cands, _row(r13=np.nan), bins).ion) == ["c0"]
    sp = AL.FormulaSpace("-")
    d = {"C": 5, "H": 7, "O": 4, "S": 1}
    assert not AL._confirmed(d, _row(), sp) and AL._confirmed(d, _row(f_S=1.0, r_S=0.045), sp)


def test_no_isotopologue_is_ever_tried_as_a_lock(monkeypatch):
    ts, _ = _batch(seed=0)
    tried = []
    real = AL._unique_lock

    def spy(b, *a, **k):
        tried.append(bool(b["iso_child"]))
        return real(b, *a, **k)

    monkeypatch.setattr(AL, "_unique_lock", spy)
    AL.fit(ts, polarity="-", log=_quiet)
    assert tried and not any(tried)


def test_a_recurring_peak_off_the_axis_is_dropped_not_locked():
    """One recurring ion reads 3 ppm off an otherwise flat +2 ppm axis and its formula
    is still the only one within +-5 ppm: it locks in pass 1 and must come off the curve."""
    flat = lambda m: np.full(np.shape(m), 2.0)              # noqa: E731
    ts, truth = _batch(flat, seed=2)
    space = AL.FormulaSpace("-")
    bins = AL.bins_from_ts(ts)
    B = bins.table
    victim = None
    for ion in sorted(truth, key=AL.exact_mz):
        th = AL.exact_mz(ion)
        if not 150 < th < 300:
            continue
        row = B.iloc[int(np.argmin(np.abs(B.mz - th * (1 + 2e-6))))]
        if row.frac >= 1 and AL._unique_lock(row, space, bins, th * (1 + 5e-6), AL.LOCK_PPM) == ion:
            victim = ion
            break
    assert victim
    th = AL.exact_mz(victim)
    near = ((ts["mz"] - th).abs() < 0.002) | ((ts["mz"] - th - AL.D13C).abs() < 0.002)
    ts2 = ts.copy()
    ts2.loc[near, "mz"] *= 1 + 3.0e-6
    model, info = AL.fit(ts2, polarity="-", log=_quiet)
    assert victim not in set(pd.DataFrame(info["locks"]).ion)
    assert info["n_off_curve"] >= 1
    assert abs(model.predict([th], extrapolate=True)[0] - 2.0) < 0.1


def test_the_model_ends_at_its_last_lock():
    """Past the last lock nothing is known: an 8 Da reach crossed a missed step."""
    _, _, model, _ = _fitted_stepped()
    top = model.segments[-1]["hi"]
    assert model.mz_range[1] == pytest.approx(top)
    assert model.in_scope([top * (1 + 2e-6)]).all() and not model.in_scope([top * (1 + 5e-6)]).any()


def test_every_mass_difference_lock_has_two_agreeing_anchors_and_no_dissent():
    ts, truth = _batch(seed=0)
    _, info = AL.fit(ts, polarity="-", log=_quiet)
    L = pd.DataFrame(info["locks"])
    base = L[L.how.isin(["unique +-5 ppm", "mass difference"])]
    n_links = 0
    for _, t in L[L.how == "mass difference"].iterrows():
        props = {}
        for _, a in base.iterrows():
            dm = t.mz - a.mz
            if a.mz == t.mz or abs(dm) > AL.ANCHOR_DA:
                continue
            for u, um in AL.UNIT_MASS.items():
                for sgn in (1, -1):
                    if abs(dm - sgn * um) * 1e3 <= AL.LINK_MDA:
                        f = AL._shift(a.ion, u, sgn, "-")
                        if f:
                            props.setdefault(f, set()).add(a.mz)
        assert set(props) == {t.ion} and len(props[t.ion]) >= 2, (t.ion, props)
        n_links += 1
    assert n_links > 100


def test_main_body_cuts_at_a_gap_and_not_on_a_slow_rise_seen_sparsely():
    x = np.concatenate([np.arange(60.0, 200.0, 2.0), np.arange(220.0, 420.0, 2.0)])
    assert AL.main_body(x, np.zeros(len(x))) == (220.0, 418.0)      # the longer run, past the 20 Da gap
    xs = np.arange(60.0, 400.0, 5.0)                                  # sparse locks: k=4 spans 20 Da
    rise = 2.0 * np.tanh((xs - 200.0) / 20.0)                         # 0.1 ppm/Da at most: smooth
    assert AL.main_body(xs, rise) == (60.0, 395.0)


def _fitted_stepped():
    ts, truth = _batch(_stepped, seed=0)
    model, info = AL.fit(ts, polarity="-", log=_quiet)
    return ts, truth, model, info


def test_every_segment_follows_the_axis_on_its_own_span(stepped):
    """A global residual median hid a line segment with its slope's sign flipped."""
    _, truth, model, _ = stepped
    th = np.array(sorted(AL.exact_mz(i) for i in truth))
    for s in model.segments:
        q = th[(th >= s["lo"]) & (th <= s["hi"])]
        resid = _stepped(q) - model.predict(q * (1 + _stepped(q) * 1e-6), extrapolate=True)
        assert len(q) >= 3 and np.median(np.abs(resid)) < 0.1, (s["kind"], s["lo"])
    lines = [s for s in model.segments if s["kind"] == "line"]
    assert lines and all(abs(s["b"] + 0.03) < 0.01 for s in lines)      # the -0.03 ppm/Da above 300


def test_bright_lock_masses_reading_zero_stay_out_of_the_curve():
    ts, truth = _batch(lambda m: np.full(np.shape(m), 1.2), seed=4)
    rows = [(sid, AL.exact_mz(ion), 2e6, 2e3) for sid in ts["sample_item_id"].unique()
            for ion in ("NO3-", "HN2O6-", "H2N3O9-")]                 # the lock masses: exactly 0 ppm
    ts = pd.concat([ts, pd.DataFrame(rows, columns=ts.columns)], ignore_index=True)
    model, info = AL.fit(ts, polarity="-", log=_quiet)
    L = pd.DataFrame(info["locks"])
    lm = L[L.ion.isin(["NO3-", "HN2O6-", "H2N3O9-"])]
    assert len(lm) >= 2 and lm["bright"].all()
    q = np.linspace(model.mz_range[0] + 0.5, 200.0, 40)
    assert np.allclose(model.predict(q, extrapolate=True), 1.2, atol=0.15)


def test_a_flat_axis_keeps_its_whole_body():
    """The thinning-out cut started from the window with the highest lock SHARE -- a
    3-peak window at an edge -- and counted every window too sparse to judge as low:
    on a flat +2 ppm axis it cut 458 of 461 locks and built no model."""
    ts, truth = _batch(lambda m: np.full(np.shape(m), 2.0), seed=2)
    model, info = AL.fit(ts, polarity="-", log=_quiet)
    assert model is not None and info["n_outside_body"] < 20, {k: v for k, v in info.items() if k != "locks"}
    assert model.segments[0]["hi"] - model.segments[0]["lo"] > 250


def test_too_few_locks_build_no_model():
    rng = np.random.default_rng(9)
    truth = list(rng.choice(_truth(0), 24, replace=False))
    ts, _ = _batch(seed=0, truth=truth)
    model, info = AL.fit(ts, polarity="-", log=_quiet)
    assert model is None and "fewer than 30" in info["why"]


def test_a_crashing_lock_fit_falls_back_and_an_unvalidated_one_is_not_applied(monkeypatch, stepped):
    ts, _, model, info = stepped

    def boom(*a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr(AL, "fit", boom)
    _, got, _ = AB.measure_axis(ts, "NO3", "orbitrap", log=_quiet)
    assert got["model"] == "reference" and "boom" in got["locks"]["why"]
    bare = {k: v for k, v in info.items() if k not in ("cv_ppm", "raw_ppm")}
    monkeypatch.setattr(AL, "fit", lambda *a, **k: (model, dict(bare)))
    out, got, _ = AB.measure_axis(ts, "NO3", "orbitrap", log=_quiet)
    assert got["verdict"] == "locks_unvalidated" and not got["applied"] and AB.AXIS_COL not in out.columns


def test_the_run_writes_the_locks_and_keeps_no_frame_in_its_summary(tmp_path, monkeypatch, stepped):
    ts = stepped[0].copy()
    ts["datetime_utc"] = pd.Timestamp("2026-01-01", tz="UTC")
    holder = {}
    real = AB.measure_axis

    def spy(*a, **k):
        out = real(*a, **k)
        holder["info"] = out[1]
        return out

    monkeypatch.setattr(AB, "measure_axis", spy)
    monkeypatch.setattr(AB.SS, "select_cover_samples",
                        lambda *a, **k: (_ for _ in ()).throw(SystemExit("measured")))
    monkeypatch.setattr(IO, "connect", lambda *a, **k: None)
    with pytest.raises(SystemExit):
        AB.run(peaks=ts, ts_peaks=ts, reagent="NO3", batch="b", out_dir=str(tmp_path), residual=False,
               n_jobs=1, resolving_power=250000.0, log=_quiet)
    assert holder["info"]["model"] == "locks" and "_locks_table" not in holder["info"]
    locks = pd.read_csv(tmp_path / "tables" / "mass_axis_locks.csv")
    assert len(locks) == holder["info"]["wave"]["n_locks"]


@pytest.mark.parametrize("cv,raw,peak,applied,verdict", [
    (0.8, 1.0, 3.0, False, "locks_unpredictive"),     # held-out error 0.8x raw: not predictive
    (0.1, 0.5, 3.0, True, "axis_steps"),              # median fine, a 3 ppm step somewhere: correct it
    (0.1, 0.5, 0.9, False, "axis_ok"),
    (0.1, 2.0, 9.0, False, "axis_steps"),             # over the 7 ppm cap: held
])
def test_the_lock_verdicts(monkeypatch, stepped, cv, raw, peak, applied, verdict):
    ts, _, model, info = stepped
    if peak > AB.LOCK_MAX_CORRECTION_PPM:         # the smooth curve itself passes the cap
        d = model.as_dict()
        d["segments"][0]["y"] = [y + peak for y in d["segments"][0]["y"]]
        model = AL.AxisModel.from_dict(d)
    fake = dict(info, cv_ppm=cv, raw_ppm=raw, max_abs_ppm=peak)
    monkeypatch.setattr(AL, "fit", lambda *a, **k: (model, dict(fake)))
    _, got, _ = AB.measure_axis(ts, "NO3", "orbitrap", log=_quiet)
    assert got["verdict"] == verdict and bool(got.get("applied")) == applied
    if peak > AB.LOCK_MAX_CORRECTION_PPM:
        assert "broken calibration" in got["held"]


def test_massqc_reads_a_lock_models_own_scope(stepped):
    _, _, model, _ = stepped
    assert not MQ.in_scope(model, [model.mz_range[1] * (1 + 25e-6)]).any()


def test_a_transient_ion_is_never_a_lock(stepped):
    ts, truth, _, info = stepped
    L = pd.DataFrame(info["locks"])
    ion = L[L.how == "unique +-5 ppm"].iloc[10].ion
    th = AL.exact_mz(ion)
    keep = ~(((ts["mz"] - th).abs() < 0.003) & ~ts["sample_item_id"].isin(["s0", "s1"]))
    _, info2 = AL.fit(ts[keep], polarity="-", log=_quiet)       # now in 2 of 6 spectra
    assert ion not in set(pd.DataFrame(info2["locks"]).ion)


def test_the_narrow_window_absorbs_ion_scatter_without_wrong_locks():
    """Each ion's own +-0.35 ppm offset (the scatter real locks show)."""
    ts, truth = _batch(lambda m: 1.0 + 0.35 * np.sin(np.asarray(m, float) * 1000.0), seed=6)
    _, info = AL.fit(ts, polarity="-", log=_quiet)
    L = pd.DataFrame(info["locks"])
    assert info["n_narrow"] >= 10
    assert L[~L.ion.isin(truth)].empty


# --- refute round 2: each test reproduces a defect found on the round-1 fixes ---------------

def _radicals(lo, hi):
    """Odd-electron anions ((H + N) even), C3-21 -- RO2.NO3- style, outside the space."""
    out = []
    for c in range(3, 22):
        for h in range(3, 2 * c + 4):
            for n in range(0, 4):
                if (h + n) % 2:
                    continue
                for o in range(2, 16):
                    if o <= 2 * c + 3 * n + 2:
                        ion = AL.name({"C": c, "H": h, "N": n, "O": o})
                        if lo < AL.exact_mz(ion) < hi:
                            out.append(ion)
    return out


def _true_ions_off(model, truth, axis, tol=0.5):
    th = np.array(sorted(AL.exact_mz(i) for i in truth))
    meas = th * (1 + axis(th) * 1e-6)
    err = (MQ.apply_correction(model, meas) - th) / th * 1e6
    return th[model.in_scope(meas) & (np.abs(err) > tol)]


def test_one_wrong_lock_does_not_bend_the_curve():
    """A radical locked as C19H43N4O2- at +4.8 ppm; judged against a spline fitted
    THROUGH it, it stayed, the curve bent to 1.4 ppm/Da and 9 true ions near it were
    corrected 1-4 ppm the wrong way (applied)."""
    axis = lambda m: 1.0 - 2.5 * (np.asarray(m, float) > 420.0)       # noqa: E731
    base = _truth(4)
    rng = np.random.default_rng(4 + 999)
    truth = base + list(rng.choice(_radicals(60, 440), 80, replace=False))
    ts, _ = _batch(axis, seed=4, truth=truth)
    out, info, _ = AB.measure_axis(ts, "NO3", "orbitrap", log=_quiet)
    model = MQ.fit_from_record(info["wave"])
    assert len(_true_ions_off(model, base, axis)) == 0
    spline = model.segments[0]
    assert np.abs(np.diff(spline["y"]) / np.diff(spline["x"])).max() <= AL.SPLINE_MAX_SLOPE + 1e-6


@pytest.mark.parametrize("pos,step,seed", [(330.0, -2.5, 0), (330.0, 3.0, 1), (300.0, -2.0, 2), (360.0, 3.5, 3)])
def test_a_calibrated_body_with_a_stepped_top_is_corrected(pos, step, seed):
    """The gate compared the held-out error with the raw median, which a calibrated
    body holds near 0: a perfect stepped model read 'locks_unpredictive' and 46-129
    ions stayed 2-3.5 ppm off."""
    axis = lambda m: step * (np.asarray(m, float) > pos)              # noqa: E731
    truth = _truth(seed)
    ts, _ = _batch(axis, seed=seed, truth=truth)
    _, info, _ = AB.measure_axis(ts, "NO3", "orbitrap", log=_quiet)
    assert info["applied"] and info["verdict"] == "axis_steps"
    assert len(_true_ions_off(MQ.fit_from_record(info["wave"]), truth, axis)) == 0


@pytest.mark.parametrize("seed", [16, 25])
def test_a_ten_da_dip_is_never_corrected_the_wrong_way(seed):
    """Two steps 10 Da apart: one line ran straight across the dip (no locks in it),
    or kept a line its own locks did not sit on; ~30 ions corrected 2 ppm the wrong way."""
    axis = lambda m: 1.5 - 2.2 * (np.asarray(m, float) > 250.0) + 2.0 * (np.asarray(m, float) > 260.0)  # noqa: E731
    truth = _truth(seed)
    ts, _ = _batch(axis, seed=seed, truth=truth)
    _, info, _ = AB.measure_axis(ts, "NO3", "orbitrap", log=_quiet)
    if info.get("applied"):
        assert len(_true_ions_off(MQ.fit_from_record(info["wave"]), truth, axis)) == 0


def test_a_radical_band_builds_no_segment_off_by_a_jump_no_step_takes():
    """A band of radicals above a -3 ppm step mapped onto closed-shell formulas through
    N3 <-> C2H2O-type swaps and built a self-consistent segment at +8.6 ppm (only the
    7 ppm cap stopped it, discarding the whole model)."""
    axis = lambda m: 1.5 - 3.0 * (np.asarray(m, float) > 400.0)       # noqa: E731
    rng = np.random.default_rng(0 + 7)
    base = [i for i in _truth(0) if not (370 < AL.exact_mz(i) < 450) or rng.random() < 0.3]
    truth = base + list(rng.choice(_radicals(370, 450), 120, replace=False))
    ts, _ = _batch(axis, seed=0, truth=truth)
    _, info, _ = AB.measure_axis(ts, "NO3", "orbitrap", log=_quiet)
    assert info["applied"]
    model = MQ.fit_from_record(info["wave"])
    assert len(_true_ions_off(model, base, axis)) == 0
    assert all(abs(s["a"]) < 5 for s in model.segments if s["kind"] == "line")


def test_the_13c_ratio_reads_the_13c_line_not_a_taller_neighbour():
    """In a bromide batch an ion's 13C line and a neighbour's 81Br line sit 0.65 mDa
    apart and chain into one bin; the ratio was read off the taller one."""
    rows = []
    for f in range(6):
        sid = f"s{f}"
        rows += [(sid, 300.0, 1000.0, 1.0), (sid, 300.0 + AL.D13C, 54.0, 0.054),
                 (sid, 300.0 + AL.D13C + 0.00065, 400.0, 0.4)]
    ts = pd.DataFrame(rows, columns=["sample_item_id", "mz", "height", "area"])
    B = AL.bins_from_ts(ts).table
    parent = B.iloc[int(np.argmin(np.abs(B.mz - 300.0)))]
    assert abs(parent.r13 - 0.054) < 1e-6


def test_a_steep_rise_at_the_low_end_is_a_curve_not_outliers():
    """A real axis rose 0.37 ppm/Da at m/z 57-61 (the scan range's low end): a slope
    bound in ppm/Da and a neighbour-median outlier test read it as wrong locks, peeled
    them off one by one, and a whole batch built no model."""
    axis = lambda m: 1.0 - 2.2 * np.exp(-(np.asarray(m, float) - 55.0) / 3.0) + 0.0 * np.asarray(m, float)  # noqa: E731
    low = ["C3H5O-", "C2H3O2-", "C2HO3-", "C3H3O3-", "C2HO4-", "C3H5O3-", "C4H5O2-", "C3H3O2-",
           "C4H7O2-", "C2H3O3-", "C3H5O2-", "C4H3O3-"]
    truth = _truth(1) + low
    ts, _ = _batch(axis, seed=1, truth=truth)
    model, info = AL.fit(ts, polarity="-", log=_quiet)
    assert model is not None, info.get("why")
    assert model.segments[0]["lo"] < 62.0
    assert len(_true_ions_off(model, truth, axis)) == 0


# --- reach: one spectrum, restarts from seeds, steps the walk tilted across ------------

def _dimers():
    """Closed-shell CnHxNOy anions of three O-ladders one CH2 / H2 apart (C19H28,
    C20H30, C20H32) -- dimer-like, m/z 400-600, joined to nothing below by one unit."""
    out = []
    for c, h in ((19, 28), (20, 30), (20, 32)):
        for o in range(8, 19):
            ion = AL.name({"C": c, "H": h, "N": 1, "O": o})
            if 400 < AL.exact_mz(ion) < 600:
                out.append(ion)
    return out


def _residuals(model, ions, axis):
    th = np.array(sorted(AL.exact_mz(i) for i in ions))
    meas = th * (1 + axis(th) * 1e-6)
    inside = model.in_scope(meas)
    return th, inside, axis(th) - np.where(inside, model.predict(meas), 0.0)


def test_one_spectrum_is_modelled_without_a_recurrence_test():
    """A batch of one spectrum (a single file's whole-file peak list) was refused --
    'fewer than 2 spectra' -- and left uncorrected. Without recurrence the isotope
    strike and the curve are the guards: every lock true, the ions on the axis."""
    ts, truth = _batch(_stepped, n_files=1, seed=0)
    model, info = AL.fit(ts, polarity="-", log=_quiet)
    assert model is not None and info["single_spectrum"]
    locks = pd.DataFrame(info["locks"])
    # a noise peak recurs in the one spectrum it is in: a rare coincidence can lock, and
    # sits within the curve's tolerance
    assert len(locks) >= 100 and (~locks["ion"].isin(truth)).mean() <= 0.01
    _, inside, resid = _residuals(model, truth, _stepped)
    assert inside.mean() > 0.6 and np.mean(np.abs(resid[inside]) <= 0.5) > 0.9


@pytest.mark.parametrize("step", [1.5, 0.0])
def test_the_walk_restarts_from_seeds_past_a_stretch_no_unit_crosses(step):
    """Monomer-like ions end below m/z 300, dimer-like ones start above 400: no unit
    step joins them and the walk (anchors within ANCHOR_DA of its top) stopped at
    the curve's end, the dimers left 1.5-2 ppm off. It restarts from seeds; a gap
    that wide is never bridged, step or not -- its axis is unseen."""
    axis = lambda m: 0.5 + 0.004 * (np.asarray(m, float) - 200) + step * (np.asarray(m, float) > 350)  # noqa: E731
    low = [i for i in _truth(0) if AL.exact_mz(i) < 300]
    ts, _ = _batch(axis, seed=0, truth=low + _dimers())
    model, info = AL.fit(ts, polarity="-", log=_quiet)
    assert any(s.get("seeded") for s in model.segments)
    _, inside, resid = _residuals(model, _dimers(), axis)
    assert inside.mean() > 0.7 and np.all(np.abs(resid[inside]) <= 0.5)
    assert not model.in_scope(np.array([350.0])).any()
    if step:
        assert info["steps"] and not info["unseen_gaps"]
    else:
        assert info["unseen_gaps"] and not info["steps"] and not info["joins"]


def test_splits_finds_a_step_and_not_a_slope():
    """A walk's line accepts within SEG_TOL_PPM and refits as the walk climbs: across a
    -0.8 ppm step between sparse, scattered locks it tilted and left the ions beside
    the step 0.6-0.9 ppm off. The step is found: one slope, an offset per part."""
    m = np.array([464.2, 464.2, 466.1, 466.2, 478.2, 478.2, 480.2, 494.2, 496.2, 510.2, 524.2, 528.2, 542.2])
    noise = np.random.default_rng(0).normal(0, 0.08, len(m))
    cuts, slope, offs = AL._splits(m, 3.0 - 0.012 * (m - 464) - 1.2 * (m > 500) + noise)
    assert cuts == [0, 9] and abs(offs[1] - offs[0] + 1.2) < 0.25 and abs(slope + 0.012) < 0.01
    cuts, _, _ = AL._splits(m, 1.0 + 0.03 * (m - 400) + noise)
    assert cuts == [0]


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
@pytest.mark.parametrize("sep,floor", [(3.0, 0.0), (4.0, 0.0), (3.0, 12.0), (4.0, 12.0)])
def test_a_peak_list_of_two_axis_states_moves_no_true_ion_the_wrong_way(seed, sep, floor):
    """An acquisition whose axis moved 3-4 ppm part-way holds every ion twice in its
    whole-file peak list (a copy under the peak picker's floor missing; copies closer than
    the peak width merge into one centroid). Each copy's formula can be a seed, the seeds
    sit on two lines, and a restart built a segment on the wrong state in half the
    realisations, moving true ions ~3 ppm the wrong way."""
    axis = lambda m: 0.5 + 0.004 * (np.asarray(m, float) - 200)        # noqa: E731
    low = [i for i in _truth(seed) if AL.exact_mz(i) < 300]
    ts, _ = _batch(axis, seed=seed, truth=low + _dimers())
    up = ts[(ts["mz"] > 380) & (ts["height"] >= floor)].copy()
    up["mz"] *= 1 + sep * 1e-6
    ts = pd.concat([ts, up], ignore_index=True)
    _, info, _ = AB.measure_axis(ts, "NO3", "orbitrap", log=_quiet)
    assert not _no_true_ion_made_worse(info, _dimers(), axis)


def test_a_walk_segment_past_the_cap_is_cut_and_the_rest_applied(monkeypatch, stepped):
    """One walk segment's line reaching past LOCK_MAX_CORRECTION_PPM at its far end held
    the whole model; that stretch is now left uncorrected and the rest applied."""
    ts, _, model, info = stepped
    d = model.as_dict()
    d["segments"][-1]["a"] -= 3.6                       # the last line: -6.8 .. -7.3 ppm
    far = AL.AxisModel.from_dict(d)
    assert far.max_abs_ppm() > AB.LOCK_MAX_CORRECTION_PPM
    monkeypatch.setattr(AL, "fit", lambda *a, **k: (far, dict(info, max_abs_ppm=far.max_abs_ppm())))
    out, got, _ = AB.measure_axis(ts, "NO3", "orbitrap", log=_quiet)
    assert got["applied"] and got["max_abs_ppm"] <= AB.LOCK_MAX_CORRECTION_PPM and "clipped_from" in got
    assert np.abs(out[AB.AXIS_COL]).max() <= AB.LOCK_MAX_CORRECTION_PPM + 1e-6
    cut = MQ.fit_from_record(got["wave"])
    assert cut.segments[-1]["clipped"] and cut.segments[-1]["hi"] < far.segments[-1]["hi"]


# --- refute round: radical aliases, walk joins, split parts ----------------------------

def _rads(lo, hi, rng, n, *, nmin=0, nmax=1):
    pool = [r for r in _radicals(lo, hi) if AL.parse(r)["C"] >= 10 and nmin <= AL.parse(r).get("N", 0) <= nmax]
    return list(rng.choice(pool, n, replace=False))


def _no_true_ion_made_worse(info, ions, axis):
    """No true ion the applied model reaches ends > 0.5 ppm off AND further off than raw."""
    if not info.get("applied"):
        return []
    model = MQ.fit_from_record(info["wave"])
    th = np.array(sorted(AL.exact_mz(i) for i in ions))
    meas = th * (1 + axis(th) * 1e-6)
    err = (MQ.apply_correction(model, meas) - th) / th * 1e6
    bad = model.in_scope(meas) & (np.abs(err) > np.maximum(0.5, np.abs(axis(th)) + 0.1))
    return list(zip(th[bad].round(3), err[bad].round(2)))


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("n_rad,n_dim,nmin,nmax", [(80, 12, 0, 1), (40, 6, 0, 1), (80, 12, 3, 4)])
def test_a_radical_band_above_a_stall_seeds_no_segment_on_its_alias_line(seed, n_rad, n_dim, nmin, nmax):
    """Odd-electron ions (C >= 10) above a stretch no unit crosses read as their
    N3 <-> C2H2O aliases (1.35 mDa: +2.2..3.4 ppm for N <= 1, -2.2..3.4 for N >= 3) --
    unique, 13C-consistent, one family on one line. A restart seeded a segment there
    (contested never fired: the true dimers are the minority) and moved the true dimers
    ~3 ppm the wrong way, reporting a STEP the axis does not take."""
    rng = np.random.default_rng(seed + 7)
    axis = lambda m: 0.5 + 0.004 * (np.asarray(m, float) - 200)        # noqa: E731
    dims = list(rng.choice(_dimers(), n_dim, replace=False))
    rads = _rads(400, 600, rng, n_rad, nmin=nmin, nmax=nmax)
    low = [i for i in _truth(seed) if AL.exact_mz(i) < 300]
    ts, _ = _batch(axis, seed=seed, truth=low + dims + rads)
    _, info, _ = AB.measure_axis(ts, "NO3", "orbitrap", log=_quiet)
    assert not _no_true_ion_made_worse(info, dims, axis)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_radicals_past_a_step_out_of_the_seed_window_seed_nothing(seed):
    """A +3.8 ppm step puts the true dimers 5.7 ppm off -- outside LOCK_PPM, so none of
    them is a seed -- and 40 radicals' O3 <-> C3N(-H2) aliases (2.7 mDa) sit 3.6 ppm
    below the axis: a restart built that line and moved the dimers from 5.7 to 6.5 ppm off."""
    rng = np.random.default_rng(seed + 19)
    axis = lambda m: 1.5 + 0.002 * (np.asarray(m, float) - 200) + 3.8 * (np.asarray(m, float) > 350)  # noqa: E731
    low = [i for i in _truth(seed) if AL.exact_mz(i) < 300]
    rads = _rads(400, 600, rng, 40)
    ts, _ = _batch(axis, seed=seed, truth=low + _dimers() + rads)
    _, info, _ = AB.measure_axis(ts, "NO3", "orbitrap", log=_quiet)
    assert not _no_true_ion_made_worse(info, _dimers(), axis)


def test_a_walk_join_wider_than_join_max_is_still_bridged():
    """JOIN_MAX_DA (30 Da) was meant for gaps a RESTART leaves; it also cut a 41 Da join
    the plain walk made (no seed involved) on a flat +2 ppm axis, and the 80 true ions in
    the gap -- seen in half the spectra, so not locks -- went from 0.0 to 2.0 ppm off."""
    axis = lambda m: np.full(np.shape(m), 2.0)                          # noqa: E731
    truth = _truth(2)
    hole = [x for x in truth if 295.0 < AL.exact_mz(x) < 335.0]
    ts, _ = _batch(axis, seed=2, truth=truth)
    drop = np.zeros(len(ts), bool)
    for x in hole:
        th = AL.exact_mz(x)
        for t in (th, th + AL.D13C):
            drop |= ((ts["mz"] - t).abs() / t < 10e-6) & ts["sample_item_id"].isin(["s3", "s4", "s5"])
    model, info = AL.fit(ts[~drop].reset_index(drop=True), polarity="-", log=_quiet)
    assert not any(s.get("seeded") for s in model.segments)              # the walk alone
    th = np.array([AL.exact_mz(x) for x in hole])
    assert model.in_scope(th * (1 + 2e-6)).mean() > 0.9, info["unseen_gaps"]


def test_splits_never_makes_a_part_too_small_to_validate():
    """A 2-lock part is its own segment (n=2): _segment_cv needs 4 and never checks it,
    SEG_MIN_LOCKS (5, 'a family of radicals put 3-4 self-consistent wrong locks on one
    line') no longer applies to it. Two locks at the segment's end 0.8 ppm off split off."""
    m = np.arange(400.0, 426.0, 2.0)
    e = np.r_[np.zeros(11), 0.8, 0.82] + np.random.default_rng(1).normal(0, 0.03, 13)
    cuts, _, _ = AL._splits(m, e)
    assert min(np.diff(cuts + [len(m)])) >= 4, cuts


def test_clipped_drops_a_flat_line_segment_past_the_cap():
    spline = {"kind": "spline", "lo": 100.0, "hi": 300.0, "n": 50,
              "x": np.array([100.0, 300.0]), "y": np.array([1.0, 1.0])}
    flat = {"kind": "line", "lo": 400.0, "hi": 450.0, "n": 8, "a": 8.0, "b": 0.0}
    cut = AL.AxisModel(segments=[spline, flat]).clipped(7.0)
    assert [s["kind"] for s in cut.segments] == ["spline"]


def test_one_outlier_lock_is_never_its_own_part():
    m = np.arange(400.0, 426.0, 2.0)
    e = np.r_[np.zeros(12), 0.9] + np.random.default_rng(1).normal(0, 0.03, 13)
    assert AL._splits(m, e)[0] == [0]


def test_scatter_alone_seldom_splits():
    m = np.arange(400.0, 426.0, 2.0)
    n = sum(len(AL._splits(m, np.random.default_rng(s).normal(0, 0.3, 13))[0]) > 1 for s in range(200))
    assert n <= 10, n                       # 5 / 200 with the SSE rule, 63 / 200 without


def test_a_small_clean_step_is_no_split():
    m = np.arange(400.0, 426.0, 2.0)
    e = np.r_[np.zeros(7), np.full(6, 0.3)] + np.random.default_rng(2).normal(0, 0.02, 13)
    assert AL._splits(m, e)[0] == [0]


def test_a_seed_needs_a_measured_13c_line(monkeypatch):
    """After the curve is fitted, the only +-LOCK_PPM candidate search is the seed scan;
    it must never look at a bin without a measured 13C line. The dimers above m/z 500
    lose their 13C lines here."""
    truth = [i for i in _truth(0) if AL.exact_mz(i) < 300] + _dimers()
    ts, _ = _batch(lambda q: 0.5 + 0.004 * (np.asarray(q, float) - 200), seed=0, truth=truth)
    drop = np.zeros(len(ts), bool)
    for x in _dimers():
        t = AL.exact_mz(x) + AL.D13C
        if t > 500:
            drop |= (ts["mz"] - t).abs() / t < 10e-6
    ts = ts[~drop].reset_index(drop=True)
    calls, real = [], AL._candidates

    def spy(b, space, bins, mz, tol):
        calls.append((tol, bool(np.isfinite(b.r13))))
        return real(b, space, bins, mz, tol)

    monkeypatch.setattr(AL, "_candidates", spy)
    AL.fit(ts, polarity="-", log=_quiet)
    first_narrow = next(k for k, (tol, _) in enumerate(calls) if tol == AL.NARROW_PPM)
    late = [ok13 for tol, ok13 in calls[first_narrow:] if tol == AL.LOCK_PPM]
    assert late and all(late)


def test_a_true_family_with_a_minority_of_radicals_is_built_from_its_first_seed():
    for seed in (0, 1):
        rng = np.random.default_rng(seed + 29)
        axis = lambda q: 0.8 + 0.004 * (np.asarray(q, float) - 200)             # noqa: E731
        dims = []
        for c, h in ((19, 28), (20, 30), (20, 32), (18, 26), (19, 30)):
            dims += [AL.name({"C": c, "H": h, "N": 1, "O": o}) for o in range(8, 19)
                     if 400 < AL.exact_mz(AL.name({"C": c, "H": h, "N": 1, "O": o})) < 600]
        dims = list(rng.choice(dims, len(dims), replace=False))
        pool = [r for r in _radicals(400, 600) if AL.parse(r)["C"] >= 10 and AL.parse(r).get("N", 0) <= 1]
        rads = list(rng.choice(pool, 30, replace=False))
        low = [i for i in _truth(seed) if AL.exact_mz(i) < 300]
        ts, _ = _batch(axis, seed=seed, truth=low + dims + rads)
        _, info, _ = AB.measure_axis(ts, "NO3", "orbitrap", log=_quiet)
        model = MQ.fit_from_record(info["wave"])
        th = np.array([AL.exact_mz(i) for i in dims])
        meas = th * (1 + axis(th) * 1e-6)
        err = (MQ.apply_correction(model, meas) - th) / th * 1e6
        assert np.mean(np.abs(err) <= 0.5) >= 0.97 and not info["locks"].get("contested")


def test_a_twenty_da_hole_is_bridged_as_a_join():
    truth = [i for i in _truth(5) if not 300 < AL.exact_mz(i) < 322]
    ts, truth = _batch(lambda q: np.full(np.shape(q), 2.0), seed=5, truth=truth)
    model, info = AL.fit(ts, polarity="-", log=_quiet)
    assert not info["unseen_gaps"]
    grid = np.linspace(302.0, 320.0, 50) * (1 + 2e-6)
    assert model.in_scope(grid).all(), (info["joins"], info["uncorrected_gaps"])


def test_a_step_inside_a_walk_segment_is_followed():
    """A -2.5 ppm step at m/z 300 ends the curve; the walk climbs monomers + dimers and
    meets a +1.2 ppm step at 360 that its line tilts across. With the split: 2 true
    ions > 0.5 ppm off; with one line: 15."""
    axis = lambda q: (1.5 - 0.004 * (np.asarray(q, float) - 300) - 2.5 * (np.asarray(q, float) > 300)   # noqa: E731
                      + 1.2 * (np.asarray(q, float) > 360))
    truth = _truth(1) + [i for i in _dimers()]
    ts, _ = _batch(axis, seed=1, truth=truth)
    ts = ts.copy()
    _, info, _ = AB.measure_axis(ts, "NO3", "orbitrap", log=_quiet)
    model = MQ.fit_from_record(info["wave"])
    th = np.array(sorted(AL.exact_mz(i) for i in truth))
    meas = th * (1 + axis(th) * 1e-6)
    err = (MQ.apply_correction(model, meas) - th) / th * 1e6 if info.get("applied") else axis(th)
    assert int((np.abs(err) > 0.5).sum()) <= 5


def test_every_walk_lock_sits_on_its_segment_line():
    for seed in (1, 5):
        rng = np.random.default_rng(seed + 50)
        base = _truth(seed)
        odd = [i for i in _radicals(355, 440) if AL.parse(i)["C"] >= 6 and AL.parse(i)["O"] >= 4]
        ts, _ = _batch(lambda q: 2.4 * np.exp(-((np.asarray(q, float) - 205.0) / 85.0) ** 2)
                       - 0.03 * np.maximum(np.asarray(q, float) - 300.0, 0.0)
                       - 2.6 * (np.asarray(q, float) > 364.6) + 2.2 * (np.asarray(q, float) > 394.0),
                       seed=seed, truth=base + list(rng.choice(odd, 80, replace=False)))
        model, info = AL.fit(ts, polarity="-", log=_quiet)
        for k, s in enumerate(model.segments):
            if s["kind"] != "line":
                continue
            own = [v for v in info["locks"] if v["how"] == f"segment {k + 1}"]
            for v in own:
                assert abs(v["ppm"] - (s["a"] + s["b"] * (v["mz"] - 400.0))) <= AL.SEG_TOL_PPM + 0.05, (k, v)


def test_two_spectra_keep_the_recurrence_test():
    ts, _ = _batch(seed=0, n_files=2)
    _, info = AL.fit(ts, polarity="-", log=_quiet)
    assert info["single_spectrum"] is False


# --- refute round 2: a real step under a radical alias, the lazy scan, the clip record --

def _flood(seed, step, nmin, nmax):
    rng = np.random.default_rng(seed + 7)
    axis = lambda m: (0.5 + 0.004 * (np.asarray(m, float) - 200)            # noqa: E731
                      + step * (np.asarray(m, float) > 350))
    dims = list(rng.choice(_dimers(), 12, replace=False))
    rads = _rads(400, 600, rng, 80, nmin=nmin, nmax=nmax)
    low = [i for i in _truth(seed) if AL.exact_mz(i) < 300]
    ts, _ = _batch(axis, seed=seed, truth=low + dims + rads)
    return ts, dims, axis


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("step", [-2.6, -2.0])
def test_a_real_step_does_not_bring_a_radical_alias_under_the_jump_cap(seed, step):
    """The N3 <-> C2H2O alias of N <= 1 radicals sits +3.4 ppm off at m/z 400; a real
    -2.6 ppm step (the one measured on the Orbitrap) brings that line to +0.8..1.9 from the
    axis below -- under SEED_JUMP_MAX -- while the true family's -2.6 is over it: the cap
    picks the alias: 1-17 true ions 2.7-3.3 ppm the wrong way, a STEP of +1.3
    reported for a -2.6 one."""
    ts, dims, axis = _flood(seed, step, 0, 1)
    _, info, _ = AB.measure_axis(ts, "NO3", "orbitrap", log=_quiet)
    assert not _no_true_ion_made_worse(info, dims, axis)


def _radicals_n34(lo, hi):
    """Odd-electron anions C10-21 with N = 3-4 (their N3 <-> C2H2O alias drops 3 N)."""
    out = []
    for c in range(10, 22):
        for h in range(3, 2 * c + 4):
            for n in (3, 4):
                if (h + n) % 2:
                    continue
                for o in range(2, 16):
                    if o <= 2 * c + 3 * n + 2:
                        ion = AL.name({"C": c, "H": h, "N": n, "O": o})
                        if lo < AL.exact_mz(ion) < hi:
                            out.append(ion)
    return out


@pytest.mark.xfail(strict=True, reason="known limit: radicals with N >= 3 read as their N0-1 aliases "
                   "(C2H2O <-> N3) after a positive step, under SEED_JUMP_MAX and SEED_MAX_N")
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_no_segment_on_a_radical_alias_line_after_a_positive_step(seed):
    """N >= 3 radicals alias 3.4 ppm BELOW the axis; a real +2.6 step brings that line
    under the cap, and it was applied: the true dimers stay ~3 ppm off (from 4.2 raw), and
    no STEP is reported."""
    rng = np.random.default_rng(seed + 23)
    axis = lambda m: (0.5 + 0.004 * (np.asarray(m, float) - 200)            # noqa: E731
                      + 2.6 * (np.asarray(m, float) > 350))
    rads = list(rng.choice(_radicals_n34(400, 600), 80, replace=False))
    dims = list(rng.choice(_dimers(), 12, replace=False))
    low = [i for i in _truth(seed) if AL.exact_mz(i) < 300]
    ts, _ = _batch(axis, seed=seed, truth=low + dims + rads)
    model, info = AL.fit(ts, polarity="-", log=_quiet)
    alias = [v for v in info["locks"] if str(v["how"]).startswith("segment") and v["mz"] > 380
             and v["ion"] not in set(dims)]
    assert len(alias) < 5, (len(alias), [(s["lo"], s["hi"], s.get("a")) for s in model.segments])


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_alias_seeds_do_not_cost_the_true_family_its_segment(seed):
    """Dense dimers with 60 radicals among them: the true segment (jump ~0.4 ppm) reads
    41 seeds on its line and 28 off (the radicals' aliases, 3.4 ppm off -- a line the jump
    cap would never build), so it is contested, and the contested-span skip then gives the
    whole stretch up: the dimers stayed uncorrected."""
    rng = np.random.default_rng(seed + 29)
    axis = lambda m: 0.8 + 0.004 * (np.asarray(m, float) - 200)               # noqa: E731
    dims = []
    for c, h in ((19, 28), (20, 30), (20, 32), (18, 26), (19, 30)):
        dims += [AL.name({"C": c, "H": h, "N": 1, "O": o}) for o in range(8, 19)
                 if 400 < AL.exact_mz(AL.name({"C": c, "H": h, "N": 1, "O": o})) < 600]
    dims = list(rng.choice(dims, len(dims), replace=False))
    rads = _rads(400, 600, rng, 60)
    low = [i for i in _truth(seed) if AL.exact_mz(i) < 300]
    ts, _ = _batch(axis, seed=seed, truth=low + dims + rads)
    _, info, _ = AB.measure_axis(ts, "NO3", "orbitrap", log=_quiet)
    model = MQ.fit_from_record(info["wave"])
    th = np.array([AL.exact_mz(i) for i in dims])
    meas = th * (1 + axis(th) * 1e-6)
    err = (MQ.apply_correction(model, meas) - th) / th * 1e6 if info.get("applied") else axis(th)
    assert np.mean(np.abs(err) <= 0.5) >= 0.9


def test_n_seed_bins_counts_the_lazy_scan():
    """info['n_seed_bins'] is read before the lazy seed scan runs: 0 on every batch, also
    on one that restarts 2-6 times."""
    axis = lambda m: 0.5 + 0.004 * (np.asarray(m, float) - 200)               # noqa: E731
    ts, _ = _batch(axis, seed=0, truth=[i for i in _truth(0) if AL.exact_mz(i) < 300] + _dimers())
    _, info = AL.fit(ts, polarity="-", log=_quiet)
    assert info["n_restarts"] > 0 and info["n_seed_bins"] > 0


def test_clipped_lists_a_dropped_segment():
    spline = {"kind": "spline", "lo": 100.0, "hi": 300.0, "n": 50,
              "x": np.array([100.0, 300.0]), "y": np.array([1.0, 1.0])}
    over = {"kind": "line", "lo": 400.0, "hi": 450.0, "n": 8, "a": 8.0, "b": 0.01}
    cut = AL.AxisModel(segments=[spline, over]).clipped(7.0)
    assert cut.stats["clipped"] == [[400.0, 450.0]]


def test_the_clipped_stretch_is_an_uncorrected_gap(monkeypatch):
    ts, _ = _batch(_stepped, seed=0)
    model, info = AL.fit(ts, polarity="-", log=_quiet)
    d = model.as_dict()
    d["segments"][-1]["a"] -= 3.6
    far = AL.AxisModel.from_dict(d)
    monkeypatch.setattr(AL, "fit", lambda *a, **k: (far, dict(info, max_abs_ppm=far.max_abs_ppm())))
    _, got, _ = AB.measure_axis(ts, "NO3", "orbitrap", log=_quiet)
    cut = AL.AxisModel.from_dict(got["wave"]).segments[-1]
    gaps = got["locks"]["uncorrected_gaps"]
    assert any(abs(a - round(cut["hi"], 3)) < 1e-3 for a, _ in gaps), gaps


def test_a_radical_alias_contested_by_the_cap_is_recorded(monkeypatch):
    """With seeds of any N, a band of radicals seeds its aliases' line: the jump cap
    refuses it, and the refusal is recorded (and logged)."""
    monkeypatch.setattr(AL, "SEED_MAX_N", 4)
    rng = np.random.default_rng(7)
    axis = lambda m: 0.5 + 0.004 * (np.asarray(m, float) - 200)               # noqa: E731
    dims = list(rng.choice(_dimers(), 12, replace=False))
    rads = _rads(400, 600, rng, 80)
    ts, _ = _batch(axis, seed=0, truth=[i for i in _truth(0) if AL.exact_mz(i) < 300] + dims + rads)
    _, info = AL.fit(ts, polarity="-", log=_quiet)
    assert any(abs(c["jump_ppm"]) > AL.SEED_JUMP_MAX for c in info.get("contested", []))


def test_a_restart_past_a_twenty_da_stretch_is_joined():
    axis = lambda m: 0.5 + 0.004 * (np.asarray(m, float) - 200)               # noqa: E731
    low = [i for i in _truth(0) if AL.exact_mz(i) < 396]
    ts, _ = _batch(axis, seed=0, truth=low + _dimers())
    model, info = AL.fit(ts, polarity="-", log=_quiet)
    seeded = [s for s in model.segments if s.get("seeded")]
    assert seeded and not info["unseen_gaps"], (info["unseen_gaps"], info["joins"])


# --- refute round 3: a contested family must not hide the one above; chained restarts ---

def _ladders(lo, hi, ladders, olo, ohi):
    out = []
    for c, h in ladders:
        for o in range(olo, ohi + 1):
            ion = AL.name({"C": c, "H": h, "N": 1, "O": o})
            if lo < AL.exact_mz(ion) < hi:
                out.append(ion)
    return out


def _chain(seed, b=0.004, s1=0.0, s2=0.0):
    """Monomers below m/z 300, dimer ladders at 400-500, trimer ladders at 560-720: two
    restarts, the second past a gap wider than a join bridges."""
    axis = lambda m: (0.5 + b * (np.asarray(m, float) - 200) + s1 * (np.asarray(m, float) > 350)  # noqa: E731
                      + s2 * (np.asarray(m, float) > 530))
    dims = [x for x in _dimers() if AL.exact_mz(x) < 500]
    tri = _ladders(560, 720, ((29, 44), (30, 46), (31, 48)), 12, 24)
    low = [i for i in _truth(seed) if AL.exact_mz(i) < 300]
    ts, _ = _batch(axis, seed=seed, truth=low + dims + tri)
    return ts, tri, axis


@pytest.mark.parametrize("seed", [0, 1])
def test_a_contested_family_does_not_hide_the_family_above_it(seed):
    """Dimers 2.4 ppm above the curve are contested by the jump cap from seed after seed;
    each contest of the same span counted toward CONTEST_STREAK, the walk stopped, and the
    trimers 100 Da higher -- back at the curve's level -- stayed 2.3 ppm off."""
    ts, tri, axis = _chain(seed, s1=2.4, s2=-2.4)
    model, info = AL.fit(ts, polarity="-", log=_quiet)
    th, inside, resid = _residuals(model, tri, axis)
    assert inside.mean() > 0.8 and np.mean(np.abs(resid[inside]) <= 0.5) > 0.9, info.get("contested")


@pytest.mark.parametrize("seed", [0, 1])
def test_a_restart_above_a_segment_from_seeds_is_judged_against_it(seed):
    """On an axis rising 0.01 ppm/Da, trimers 60 Da past a segment from seeds were judged
    against the curve 300 Da below, held flat: the trend read as a 3 ppm jump, the
    trimers were contested and left 4 ppm off."""
    ts, tri, axis = _chain(seed, b=0.01)
    model, info = AL.fit(ts, polarity="-", log=_quiet)
    th, inside, resid = _residuals(model, tri, axis)
    assert inside.mean() > 0.8 and np.mean(np.abs(resid[inside]) <= 0.5) > 0.9, info.get("contested")
