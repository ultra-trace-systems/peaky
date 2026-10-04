"""The evidence scale's run context (peaky.assignment.levels.context): the
calibrated windows (the persisted degeneracy calibration first, the fit to the
stored densities as the fallback, the uncalibrated rule), the per-file peak
arrays and their probe, the class gate, the twin ratios and the run-dir loader.
Synthetic, offline."""
import json
import random

import numpy as np
import pandas as pd
import pytest

from peaky.assignment import degeneracy as DG
from peaky.assignment import ledger as L
from peaky.assignment.levels import context as CX
from peaky.assignment.levels import space as SP
from peaky.chem import chemistry as C
from peaky.chem.resolution import Resolution

ORBI = Resolution(coef=0.002 / 200 ** 1.5, exponent=1.5, offset=0.0).as_dict()     # R(200) = 100 000
TOF = Resolution(coef=200 / 9000 / 200 ** 1.0, exponent=1.0, offset=0.0).as_dict()  # R(200) = 9 000


# --------------------------------------------------------------------------- helpers
def _ledger(rows, *, ppm=None):
    """rows: (peak_id, neutral, adduct); each peak sits ppm[i] off its reading's m/z."""
    ppm = ppm or [0.0] * len(rows)
    mzs = [C.ion_mz(f, a) * (1 + p * 1e-6) for (_, f, a), p in zip(rows, ppm)]
    led = L.new_ledger(pd.DataFrame({"peak_id": [p for p, _, _ in rows], "mz": mzs,
                                     "height": [1e5] * len(rows)}))
    for (p, f, a), e in zip(rows, ppm):
        L.commit_assignment(led, p, neutral_formula=f, adduct=a, ion_formula=f"{f}{a[-1]}",
                            ion_score=0.9, compound_score=0.9, ppm_error=e, pass_no=1,
                            method="cheminfo+grid", confidence="Good", commentary="x")
    return led


def _peaks(rows):
    """rows: (peak_id, mz, height, role, parent, neutral, adduct)."""
    return pd.DataFrame([dict(peak_id=p, mz=m, height=h, area=h, role=r, parent_peak_id=par,
                              neutral_formula=n, adduct=a, iso_label="13C" if r == "iso_child" else None)
                         for p, m, h, r, par, n, a in rows])


def _summary(reagent="NO3", context="ambient-air", resolution=ORBI, per_file=()):
    return dict(reagent=reagent, context=context, reflists_active=[], resolution=resolution,
                per_file=list(per_file))


# --------------------------------------------------------------------------- windows
def test_window_table_medians_and_the_uncalibrated_rule():
    win = {"a": dict(fams=(), fit=(0.1, 0.30)), "b": dict(fams=(), fit=(0.3, 0.20)),
           "c": dict(fams=(), fit=(0.2, 0.25)),
           "d": dict(fams=(), cal=None, mu_stamp=-0.4),          # uncalibrated: its stamp centre, the run sigma
           "e": dict(fams=(), cal=None, mu_stamp=float("nan"))}  # no stamp either: the run window
    windows, run = CX.window_table(win)
    assert run == (0.2, 0.25)
    assert windows["a"] == (0.1, 0.30) and windows["d"] == (-0.4, 0.25) and windows["e"] == (0.2, 0.25)


def test_window_table_without_any_calibrated_file_is_nan():
    windows, run = CX.window_table({"a": dict(fams=(), cal=None, mu_stamp=0.1)})
    assert np.isnan(run[0]) and np.isnan(run[1]) and windows["a"][0] == 0.1


def test_run_windows_prefers_the_persisted_degeneracy_calibration(monkeypatch):
    led = _ledger([("p1", "C6H10O5", "[M+NO3]-")])
    led["ppm_error_cal"] = led["ppm_error"] - 0.7
    summary = _summary(per_file=[dict(sample_id="a", degeneracy_cal={"mu": 0.12, "sigma": 0.31}),
                                 dict(sample_id="b", degeneracy_cal=None)])
    monkeypatch.setattr(CX, "fit_windows", lambda *a, **k: pytest.fail("the fallback ran"))
    win = CX.run_windows(summary, {"a": led, "b": led})
    assert win["a"]["src"] == "degeneracy_cal" and win["a"]["fit"] == (0.12, 0.31)
    assert "fit" not in win["b"] and win["b"]["cal"] is None
    assert win["b"]["mu_stamp"] == pytest.approx(0.7)
    windows, run = CX.window_table(win)
    assert run == (0.12, 0.31) and windows["b"] == (pytest.approx(0.7), 0.31)


def test_run_windows_falls_back_per_file(monkeypatch):
    calls = []
    monkeypatch.setattr(CX, "fit_windows",
                        lambda summary, pf, **k: calls.append(sorted(pf)) or {s: dict(fams=(), src="fit", fit=(0.5, 0.2))
                                                                              for s in pf})
    led = _ledger([("p1", "C6H10O5", "[M+NO3]-")])
    summary = _summary(per_file=[dict(sample_id="a", degeneracy_cal={"mu": 0.1, "sigma": 0.3}),
                                 dict(sample_id="b")])
    win = CX.run_windows(summary, {"a": led, "b": led})
    assert calls == [["b"]] and win["b"]["fit"] == (0.5, 0.2) and win["a"]["fit"] == (0.1, 0.3)
    assert list(win) == ["a", "b"]


def test_parity_scorer_equals_density_in_window():
    rng = random.Random(3)
    rows = []
    for _ in range(300):
        found = {f"ion{j}": ("N", "A", rng.uniform(-3, 3)) for j in range(rng.randint(0, 9))}
        ai = rng.choice([None, "ion0", "ion1", "other"])
        srs = rng.choice([None, "outside"])
        st = rng.choice([float("nan"), 0.0, 1.0, 2.0, 3.0, 4.0, 6.0])
        rows.append((found, ai, srs, st))
    score = CX._ParityScorer(rows)
    for mu, sg in [(0.0, 0.3), (0.25, 0.5), (-0.4, 0.15), (1.0, 0.9)]:
        eq = w1 = 0
        for found, ai, srs, st in rows:
            d, _ = CX.density_in_window(found, ai, srs, mu, sg)
            if (np.isnan(d) and np.isnan(st)) or d == st:
                eq += 1
                w1 += 1
            elif np.isfinite(d) and np.isfinite(st) and abs(d - st) <= 1:
                w1 += 1
        assert score(mu, sg) == (eq, w1)


@pytest.fixture(scope="module")
def noted():
    """A nitrate ledger whose degeneracy notes were written at (0.30, 0.40) ppm."""
    sp = SP.Space("NO3", "ambient-air", [], ())
    rows = [(f"p{i}", f"C{n}H{2 * n - 2}O{k}", ad) for i, (n, k, ad) in enumerate(
        (n, k, ad) for n in range(5, 15) for k in (4, 6, 8) for ad in ("[M+NO3]-", "[M-H]-"))]
    rng = random.Random(11)
    led = _ledger(rows, ppm=[0.3 + 0.4 * rng.uniform(-2.5, 2.5) for _ in rows])
    DG.apply_degeneracy(led, cal=(0.30, 0.40), context="ambient-air", adducts=sp.adducts, families=(),
                        curated=sp.curated)
    return led, sp


def test_fit_window_keeps_a_calibration_that_reproduces_the_stored_densities(noted):
    led, sp = noted
    rec = CX.fit_window(led, sp, cal=(0.30, 0.40))
    assert rec["n_noted"] == len(led) and rec["parity_cal"][0] == rec["n_noted"]
    assert rec["fit"] == (0.30, 0.40)


def test_fit_window_grid_recovers_the_window_from_a_biased_calibration(noted):
    led, sp = noted
    rec = CX.fit_window(led, sp, cal=(0.10, 0.30))
    n = rec["n_noted"]
    assert rec["parity_cal"][0] < CX.FIT_PARITY_MIN * n       # the biased window misses some stored densities ...
    assert rec["parity_fit"] == (n, n)                         # ... the grid reproduces them all
    mu, sg = rec["fit"]
    assert 0.10 - 0.2625 <= mu <= 0.10 + 0.2626 and 0.15 - 0.011 <= sg <= 0.42 + 0.011


def test_fit_window_uncalibrated_file_takes_the_stamp_centre(noted):
    led, sp = noted
    led = led.copy()
    led["ppm_error_cal"] = pd.to_numeric(led["ppm_error"]) - 0.25
    rec = CX.fit_window(led, sp)          # no iso children: tiers._calibrate gives up
    assert rec["cal"] is None and "fit" not in rec
    assert rec["mu_stamp"] == pytest.approx(0.25)


def test_an_uncalibrated_ledger_without_the_calibrated_column_has_no_stamp_centre():
    """tiers.stamp_calibrated_ppm writes `ppm_error_cal` only when it can centre the
    file: a ledger that never got it (a decoy arm with nothing Assigned) reads no
    stamp centre -- the run window -- instead of crashing the stage (safe=False)."""
    led = _ledger([("p1", "C6H10O5", "[M+NO3]-")])
    assert "ppm_error_cal" not in led.columns
    assert np.isnan(CX.mu_stamp(led))
    summary = _summary(per_file=[dict(sample_id="a", degeneracy_cal={"mu": 0.12, "sigma": 0.31}),
                                 dict(sample_id="b", degeneracy_cal=None)])
    win = CX.run_windows(summary, {"a": led, "b": led})
    assert np.isnan(win["b"]["mu_stamp"])
    windows, run = CX.window_table(win)
    assert windows["b"] == run == (0.12, 0.31)


# --------------------------------------------------------------------------- per-file arrays
@pytest.fixture()
def fa():
    t = _peaks([("p1", 100.0, 1e5, "M0", None, "X", "A"),
                ("k1", 101.0034, 1.1e3, "iso_child", "p1", None, None),
                ("p2", 102.0, 5e4, "M0", None, "Y", "B"),
                ("u1", 103.0, 2e3, "unexplained", None, None, None),
                ("z0", 104.0, 0.0, "unexplained", None, None, None),     # height 0: not a peak
                ("p3", 110.0, 1e4, "M0", None, "Z", "C")])
    return CX.FileArr(t.sample(frac=1.0, random_state=1), gate=500.0)


def test_filearr_keeps_rows_with_height_sorted_with_scan_edges(fa):
    assert list(fa.pid) == ["p1", "k1", "p2", "u1", "p3"]
    assert fa.scan_start == 100.0 and fa.scan_end == 110.0 and fa.gate == 500.0
    assert fa.edge == pytest.approx(np.percentile([1e5, 1.1e3, 5e4, 2e3, 1e4], 1.0))
    assert fa.in_scan(100.5, 109.5) and not fa.in_scan(100.4, 105.0) and not fa.in_scan(101.0, 109.6)
    assert list(fa.pk[:1]) == ["X|A"] and fa.parent[1] == "p1"


def test_filearr_probe_free_occupied_absent_shadowed(fa):
    assert fa.probe((101.0034, 101.0034), 0.001, "p1")[0] == "free"            # the anchor's own child
    assert fa.probe((101.0034, 101.0034), 0.001, "p9")[0] == "occupied"        # another parent's child
    assert fa.probe((102.0, 102.0), 0.001, "p1")[0] == "occupied"              # another reading's M0
    fa.refuted = {"Y|B"}
    assert fa.probe((102.0, 102.0), 0.001, "p1")[0] == "free"                  # a refuted reading owns nothing
    assert fa.probe((102.0, 102.0), 0.001, "p1", own={"Y|B"})[0] == "free"
    assert fa.probe((103.0, 103.0), 0.001, "p1")[:2] == ("free", 2e3)          # an unexplained peak
    assert fa.probe((106.5, 106.5), 0.001, "p1") == ("absent", 0.0, -1)
    assert fa.probe((102.003, 102.003), 0.0005, "p1", shadow_da=0.01, h_exp=1e4)[0] == "shadowed"


def test_filearr_tallest_and_shoulder(fa):
    assert fa.tallest(102.0, 0.01) == 5e4 and fa.tallest(106.0, 0.01) == 0.0
    assert fa.shoulder(1, 2.0, 3.0) and not fa.shoulder(0, 2.0, 3.0)


# --------------------------------------------------------------------------- the context
def _pf():
    led = _ledger([("p1", "C6H10O5", "[M+NO3]-"), ("p2", "C7H12O5", "[M+H]+")])
    return {"a": led, "b": led.copy()}


def test_build_run_context_assesses_the_orbitrap_class_only():
    win = {"a": dict(fams=(), fit=(0.1, 0.3)), "b": dict(fams=(), fit=(0.2, 0.3))}
    pf = _pf()
    with pytest.raises(ValueError, match="not assessed"):
        CX.build_run_context(_summary(resolution=TOF, per_file=[dict(sample_id="a"), dict(sample_id="b")]), pf, win)
    with pytest.raises(ValueError, match="not assessed"):
        CX.build_run_context(_summary(resolution=None, per_file=[dict(sample_id="a"), dict(sample_id="b")]), pf, win)


def test_build_run_context_gates_channels_and_windows():
    pf = _pf()
    win = {"a": dict(fams=(), fit=(0.1, 0.3)), "b": dict(fams=("siloxane",), fit=(0.2, 0.3))}
    summary = _summary(reagent="Ur", context="uronium",
                       per_file=[dict(sample_id="a", height_gate_cps=250.0),
                                 dict(sample_id="b", height_gate_cps=0, noise_edge_cps=None)])
    ctx = CX.build_run_context(summary, pf, win, name="t")
    assert ctx.klass == "orbitrap" and ctx.run_window == (0.15000000000000002, 0.3)
    assert ctx.gates["a"] == 250.0 and ctx.gates["b"] == ctx.files["b"].edge   # no gate: the 1st-percentile height
    assert ctx.channels == ["[M+H]+", "[M+(CH4N2O)H]+"]
    assert ctx.decomp_adducts[-1] == "[M+NH4]+" and "[M+NO3]-" in ctx.decomp_adducts
    assert ctx.families == ("siloxane",) and not ctx.labelled and ctx.twin_q == {}
    assert ctx.tol_da(200.0, 1e5) == pytest.approx(200.0 * 1e-6)      # no position fit: 1 ppm
    assert ctx.fwhm(200.0) == pytest.approx(0.002)
    d = ctx.decompositions(SP.ion_counts_of("C7H12O5", "[M+H]+"), "C7H12O5", "[M+H]+")
    assert [x["adduct"] for x in d] == ctx.decomp_adducts


def test_twin_ratios_need_three_bright_pairs_on_a_labelled_run():
    rows = []
    for i, n in enumerate(("C5H8O4", "C6H10O4", "C7H12O4", "C8H14O4")):
        rows += [(f"a{i}", n, "[M+NO3]-"), (f"b{i}", n, "[M+^NO3]-")]
    led = _ledger(rows)
    led["height"] = [1e5, 2e4, 1e5, 4e4, 1e5, 3e4, 10.0, 1.0][:len(led)]
    summary = _summary(reagent="NO3+NO3_15N", per_file=[dict(sample_id="a", height_gate_cps=1000.0)])
    ctx = CX.build_run_context(summary, {"a": led}, {"a": dict(fams=(), fit=(0.0, 0.3))})
    assert ctx.labelled
    assert ctx.twin_q == {"a": pytest.approx(np.quantile([0.2, 0.4, 0.3], CX.TWIN_Q))}
    led2 = led.copy()
    led2.loc[led2.index[:2], "height"] = [100.0, 10.0]       # only two bright pairs left
    ctx2 = CX.build_run_context(summary, {"a": led2}, {"a": dict(fams=(), fit=(0.0, 0.3))})
    assert ctx2.twin_q == {}


def test_pairs_from_files_brightest_row_alts_and_own_twin():
    led = _ledger([("p1", "C6H10O5", "[M+^NO3]-"), ("p2", "C6H10O5", "[M+^NO3]-")])
    led["height"] = [1e4, 3e4]
    led["alternatives"] = ['[{"formula": "C5H6N2O4", "adduct": "[M-H]-", "ppm": 0.4}]', None]
    led["tied"] = [True, False]
    facts = pd.DataFrame([dict(neutral_formula="C6H10O5", adduct="[M+^NO3]-", mz=1.0, ion="", ppm=0.1)])
    pr = CX.pairs_from_files({"s": led}, facts)[("C6H10O5", "[M+^NO3]-")]
    assert [o["pid"] for o in pr["obs"]] == ["p2"]
    assert pr["alts"] == [({"formula": "C5H6N2O4", "adduct": "[M-H]-", "ppm": 0.4}, True, "s")]
    assert pr["own_pk"] == {"C6H10O5|[M+NO3]-"}


def test_source_inputs_from_run_dir(tmp_path):
    (tmp_path / "per_file").mkdir()
    (tmp_path / "tables").mkdir()
    (tmp_path / "batch_summary.json").write_text(json.dumps(_summary()))
    for sid in ("b", "a"):
        _ledger([("p1", "C6H10O5", "[M+NO3]-")]).to_csv(tmp_path / "per_file" / f"{sid}_ledger.csv", index=False)
    pd.DataFrame({"check": ["C"], "bias_area": [0.02]}).to_csv(tmp_path / "tables" / "iso_checks.csv", index=False)
    pd.DataFrame({"sample_item_id": ["a"], "mz": [1.0], "height": [2.0], "extra": [3]}).to_parquet(
        tmp_path / "per_file" / "_batch_ts.parquet")
    inp = CX.source_inputs_from_run_dir(str(tmp_path))
    assert list(inp.per_file) == ["a", "b"] and inp.summary["reagent"] == "NO3"
    assert inp.merged is None and inp.label_twins is None
    assert list(inp.ts.columns) == ["sample_item_id", "mz", "height"]
    assert CX.bias_area(inp.iso_checks) == 0.02


# --------------------------------------------------------------------------- the line model the context's efficiency reads
def test_cand_lines_purity_is_explicit_and_det_x_follows_the_substitution():
    from peaky.assignment.levels import lines as LN
    counts = SP.ion_counts_of("C6H10O5", "[M+^NO3]-")
    lines = {L["label"]: L for L in LN.cand_lines(counts, 0.002, 0.0, 0.98)}
    assert lines["13C"]["testable"] and lines["13C"]["det_x"] == LN.DET_X
    assert lines["18O"]["det_x"] == LN.DET_X_MINOR
    r14 = lines[LN.REAGENT14N_LABEL]
    assert r14["mode"] == "reagent14N" and r14["ratio"] == pytest.approx(0.02 / 0.98)
    other = {L["label"]: L for L in LN.cand_lines(counts, 0.002, 0.0, 0.99)}
    assert other[LN.REAGENT14N_LABEL]["ratio"] == pytest.approx(0.01 / 0.99)


def test_eval_candidate_counts_an_in_band_13c_line():
    from peaky.assignment.levels import lines as LN
    n, a = "C6H10O5", "[M+NO3]-"
    m0 = C.ion_mz(n, a)
    counts = SP.ion_counts_of(n, a)
    lines = LN.cand_lines(counts, 0.002, 0.0, 0.98)
    c13 = next(L for L in lines if L["label"] == "13C")
    t = _peaks([("lo", 150.0, 1e3, "unexplained", None, None, None),
                ("p1", m0, 1e6, "M0", None, n, a),
                ("k1", m0 + c13["d"], 1e6 * c13["ratio"], "iso_child", "p1", None, None),
                ("hi", 300.0, 1e3, "unexplained", None, None, None)])
    summary = _summary(per_file=[dict(sample_id=s, height_gate_cps=100.0) for s in "abc"])
    pf = {s: t for s in "abc"}
    ctx = CX.build_run_context(summary, pf, {s: dict(fams=(), fit=(0.0, 0.3)) for s in "abc"})
    obs = [dict(sid=s, mz=m0, h=1e6, area=1e6, pid="p1") for s in "abc"]
    rec = next(r for r in LN.eval_candidate(ctx, dict(lines=lines), obs, f"{n}|{a}") if r["L"]["label"] == "13C")
    assert (rec["n_det"], rec["n_test"], rec["n_ok"], rec["n_bad"]) == (3, 3, 3, 0)
    rec18 = next(r for r in LN.eval_candidate(ctx, dict(lines=lines), obs, f"{n}|{a}") if r["L"]["label"] == "18O")
    assert rec18["n_abs"] == 3 and rec18["n_bad"] == 3          # a detectable minor line tests by its absence
