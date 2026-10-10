"""The evidence scale's isotope-line layer (peaky.assignment.levels.lines) and
the step-1 competitor facts (peaky.assignment.levels.competitors): the line
model, a line probed in band / absent / shadowed / occupied / off-scan, the
exact 37Cl test, tests (a), (b), (c), (k), the matched elements, isotope-line
competitors and the pass-B competitor rows. Synthetic per-file arrays, offline."""
import re

import numpy as np
import pandas as pd
import pytest

from peaky.assignment.levels import competitors as CP
from peaky.assignment.levels import context as CX
from peaky.assignment.levels import lines as LN
from peaky.assignment.levels import space as SP
from peaky.chem import chemistry as C
from peaky.chem.resolution import Resolution

ORBI = Resolution(coef=0.002 / 200 ** 1.5, exponent=1.5, offset=0.0).as_dict()       # FWHM(200) = 2 mDa
ORBI_HI = Resolution(coef=0.0001 / 200 ** 1.5, exponent=1.5, offset=0.0).as_dict()   # FWHM(200) = 0.1 mDa
SIDS = ("a", "b", "c")
CL, CLA = "C6H5ClO", "[M-H]-"          # a chlorinated reading (its 37Cl line at +1.997, 0.32 x)
PH, PHA = "C6H6O", "[M-H]-"            # a chlorine-free reading


# --------------------------------------------------------------------------- helpers
def _t(rows):
    """rows: (peak_id, mz, height, role, parent, neutral, adduct, iso_label)."""
    return pd.DataFrame([dict(peak_id=p, mz=m, height=h, area=h, role=r, parent_peak_id=par, neutral_formula=n,
                              adduct=a, iso_label=lab) for p, m, h, r, par, n, a, lab in rows])


def _edges(lo=50.0, hi=400.0):
    """Two far peaks that put every probed line inside the scan."""
    return [("lo", lo, 1e3, "unexplained", None, None, None, None),
            ("hi", hi, 1e3, "unexplained", None, None, None, None)]


def _ctx(tables, *, reagent="NO3", resolution=ORBI, gate=100.0):
    """A context over {sid: peak table} (or one table for SIDS), windows (0, 0.3), no position fit, eff 1."""
    pf = tables if isinstance(tables, dict) else {s: tables.copy() for s in SIDS}
    summary = dict(reagent=reagent, context="ambient-air", reflists_active=[], resolution=resolution,
                   per_file=[dict(sample_id=s, height_gate_cps=gate) for s in pf])
    ctx = CX.build_run_context(summary, pf, {s: dict(fams=(), fit=(0.0, 0.3)) for s in pf})
    ctx.sig_fit = None
    return ctx


def _obs(mz, h=1e6, pid="p1", sids=SIDS):
    return [dict(sid=s, mz=mz, h=h, area=h, pid=pid, ppm=0.0) for s in sids]


def _line(lines, label):
    return next(L for L in lines if L["label"] == label)


def _rec(recs, label):
    return next(r for r in recs if r["L"]["label"] == label)


def _own_lines(n, a, h=1e6, gate=100.0, pid="p1", skip=()):
    """The reading's M0 plus every detectable testable line as an iso child at its predicted height."""
    m0 = C.ion_mz(n, a)
    rows = [(pid, m0, h, "M0", None, n, a, None)]
    for L in LN.cand_lines(SP.ion_counts_of(n, a), 0.001, 0.0, 0.98):
        if L["testable"] and h * L["ratio"] >= L["det_x"] * gate and L["label"] not in skip:
            rows.append((f"k{L['label']}", m0 + L["d"], h * L["ratio"], "iso_child", pid, None, None, L["label"]))
    return m0, rows


# --------------------------------------------------------------------------- the line model
def test_cand_lines_chlorine_line_is_single_heavy_and_18o_is_not_testable_beside_cl():
    lines = LN.cand_lines(SP.ion_counts_of(CL, CLA), 0.001, 0.0, 0.98)
    cl = _line(lines, "37Cl")
    assert cl["testable"] and cl["det_x"] == LN.DET_X and cl["elements"] == ["Cl"]
    assert cl["d"] == pytest.approx(1.99705, abs=1e-5) and cl["ratio"] == pytest.approx(0.32, abs=0.01)
    assert not _line(lines, "18O")["testable"]                 # an M+2 owner (Cl) hides the 18O line
    assert _line(lines, "13C+37Cl")["det_x"] == LN.DET_X_MINOR  # a composite line needs 5 x the gate
    assert all(0 < abs(L["nominal"]) <= LN.MAX_NOMINAL for L in lines if L["testable"])
    assert LN.cand_lines(SP.ion_counts_of(CL, CLA), 0.001, 0.0, 0.98) is lines      # memoised


def test_cand_lines_from_a_committed_peak_that_is_not_the_mono_line():
    counts = SP.ion_counts_of(CL, CLA)
    lines = LN.cand_lines(counts, 0.001, 1.997, 0.98)          # the observed peak is the 37Cl line
    mono = min(lines, key=lambda L: L["d"])                     # the mono line, read from the anchor down
    assert mono["d"] == pytest.approx(-1.99705, abs=1e-5) and mono["ratio"] == pytest.approx(1 / 0.32, rel=0.05)
    assert mono["label"] == "-1x37Cl"         # the reference's tag text for a removed substitution (kept for parity)


# --------------------------------------------------------------------------- probes and accounting
def test_line_in_band_is_matched_and_absent_line_refutes():
    m0, rows = _own_lines(CL, CLA)
    ctx = _ctx(_t(rows + _edges()))
    cand = dict(lines=LN.cand_lines(SP.ion_counts_of(CL, CLA), ctx.fwhm(m0), 0.0, 0.98))
    recs = LN.eval_candidate(ctx, cand, _obs(m0), f"{CL}|{CLA}")
    r = _rec(recs, "37Cl")
    assert (r["n_det"], r["n_test"], r["n_ok"], r["n_bad"]) == (3, 3, 3, 0)
    assert LN.contradiction_a(recs, 3) == []
    els, labs = LN.matched_elements(recs, 3)
    assert els == {"C", "Cl"} and labs[:2] == ["13C", "37Cl"] and "13C+37Cl" in labs
    assert LN.matched_elements(recs, 4) == (set(), [])          # fewer testable files than the minimum
    # the 37Cl line missing from every file: absent in 3/3 -> (a)
    _m0, rows2 = _own_lines(CL, CLA, skip=("37Cl",))
    ctx2 = _ctx(_t(rows2 + _edges()))
    recs2 = LN.eval_candidate(ctx2, cand, _obs(m0), f"{CL}|{CLA}")
    assert _rec(recs2, "37Cl")["n_abs"] == 3
    assert LN.contradiction_a(recs2, 3) == ["37Cl (0.32x) absent in 3/3 files"]


def test_line_too_high_counts_only_on_an_own_iso_child():
    m0, rows = _own_lines(PH, PHA)
    hi = [(p, m, h * 5 if lab == "13C" else h, r, par, n, a, lab) for p, m, h, r, par, n, a, lab in rows]
    cand = dict(lines=LN.cand_lines(SP.ion_counts_of(PH, PHA), 0.001, 0.0, 0.98))
    r = _rec(LN.eval_candidate(_ctx(_t(hi + _edges())), cand, _obs(m0), ""), "13C")
    assert (r["n_test"], r["n_hi"]) == (3, 3)                   # the anchor's own child, 5 x: too high
    unexpl = [(p, m, h, "unexplained" if lab == "13C" else r_, None if lab == "13C" else par, n, a, lab)
              for p, m, h, r_, par, n, a, lab in hi]
    r2 = _rec(LN.eval_candidate(_ctx(_t(unexpl + _edges())), cand, _obs(m0), ""), "13C")
    assert (r2["n_det"], r2["n_test"]) == (3, 0)                # too high on an unexplained peak: untestable


def test_occupied_line_in_band_above_band_and_too_small():
    m0, rows = _own_lines(PH, PHA)
    d13 = _line(LN.cand_lines(SP.ion_counts_of(PH, PHA), 0.001, 0.0, 0.98), "13C")
    exp = 1e6 * d13["ratio"]
    base = [x for x in rows if x[7] != "13C"]
    cand = dict(lines=LN.cand_lines(SP.ion_counts_of(PH, PHA), 0.001, 0.0, 0.98))

    def rec(h):
        occ = ("q1", m0 + d13["d"], h, "M0", None, "C5H2N2", "[M-H]-", None)    # another reading's M0
        return _rec(LN.eval_candidate(_ctx(_t(base + [occ] + _edges())), cand, _obs(m0), ""), "13C")

    r = rec(exp)
    assert (r["n_test"], r["n_occ"], r["n_ok"], r["n_bad"]) == (3, 3, 0, 0)   # present, not matched
    r = rec(exp * 10)
    assert (r["n_det"], r["n_test"]) == (3, 0)                  # an occupant above the band: untestable
    r = rec(exp * 0.1)
    assert (r["n_test"], r["n_lo"]) == (3, 3)                   # a too-small occupant: a major line too low


def test_shadowed_and_offscan_lines_are_untestable():
    m0, rows = _own_lines(PH, PHA)
    d13 = _line(LN.cand_lines(SP.ion_counts_of(PH, PHA), 0.001, 0.0, 0.98), "13C")
    base = [x for x in rows if x[7] != "13C"]
    # a big peak 0.5 mDa off the line: outside the 1 ppm window, inside 1.5 FWHM -> shadowed
    sh = ("u", m0 + d13["d"] + 0.0005, 1e6, "unexplained", None, None, None, None)
    ctx = _ctx(_t(base + [sh] + _edges()))
    cand = dict(lines=LN.cand_lines(SP.ion_counts_of(PH, PHA), ctx.fwhm(m0), 0.0, 0.98))
    r = _rec(LN.eval_candidate(ctx, cand, _obs(m0), ""), "13C")
    assert [p[0] for p in r["per"]] == ["shadowed"] * 3 and (r["n_det"], r["n_test"]) == (3, 0)
    # the scan ends 0.3 Da past the line: off-scan, not even detectable
    ctx2 = _ctx(_t([x for x in base if x[3] == "M0"] + _edges(hi=m0 + d13["d"] + 0.3)))
    r2 = _rec(LN.eval_candidate(ctx2, cand, _obs(m0), ""), "13C")
    assert [p[0] for p in r2["per"]] == ["offscan"] * 3 and r2["n_det"] == 0


def test_a_taller_side_lobe_beside_the_13c_line_neither_occupies_nor_empties_it():
    """A per-file stage marked a peak 'artifact' (an Orbitrap side lobe of a brighter
    neighbour) inside the position window of the reading's 13C line, taller than the
    13C child itself. The pick skips it: the 13C child is read, in band, and the
    reading keeps its 13C positive fact. A position only the artifact holds is
    untestable ('shadowed'), never absent."""
    m0, rows = _own_lines(PH, PHA)
    lines = LN.cand_lines(SP.ion_counts_of(PH, PHA), 0.001, 0.0, 0.98)
    d13 = _line(lines, "13C")
    x13 = m0 + d13["d"]
    lobe = ("lobe", x13 + 0.5e-6 * x13, 1e6 * d13["ratio"] * 1.4, "artifact", None, None, None, None)
    ctx = _ctx(_t(rows + [lobe] + _edges()))
    cand = dict(lines=LN.cand_lines(SP.ion_counts_of(PH, PHA), ctx.fwhm(m0), 0.0, 0.98))
    recs = LN.eval_candidate(ctx, cand, _obs(m0), "")
    r = _rec(recs, "13C")
    assert [p[0] for p in r["per"]] == ["free"] * 3 and (r["n_test"], r["n_ok"], r["n_occ"]) == (3, 3, 0)
    assert all(ctx.files[s].pid[p[5]] == "k13C" for s, p in zip(SIDS, r["per"]))
    els, labs = LN.matched_elements(recs, 3)
    assert "C" in els and "13C" in labs
    # the 13C child gone: only the artifact holds the position -> untestable, not absent
    ctx2 = _ctx(_t([x for x in rows if x[7] != "13C"] + [lobe] + _edges()))
    recs2 = LN.eval_candidate(ctx2, cand, _obs(m0), "")
    r2 = _rec(recs2, "13C")
    assert [p[0] for p in r2["per"]] == ["shadowed"] * 3 and (r2["n_det"], r2["n_test"], r2["n_abs"]) == (3, 0, 0)
    assert LN.contradiction_a(recs2, 3) == []
    # the exact 37Cl probe reads it the same way
    assert LN.probe_exact_37cl(ctx2, "a", (x13, x13), 1e-6 * x13, "p1", (), lobe[2])[0] == "shadowed"
    assert LN.probe_exact_37cl(ctx, "a", (x13, x13), 1e-6 * x13, "p1", (), lobe[2])[:1] == ("free",)


def test_reagent14n_line_holding_a_foreign_peak_is_present_not_too_low():
    """A labelled [M+^NO3]- reading predicts the reagent's 14N impurity line at
    -0.997 Da (~2 % of M0). A real ion the engine left unexplained can sit at
    that position, far above the impurity level: the line is present (an
    occupant in band -- tested, not bad, not matched), not 'too low'. The same
    peak below half the impurity level is still too low, and the engine's own
    child there still matches."""
    n, a = "C10H18O4", "[M+^NO3]-"
    m0, rows = _own_lines(n, a, skip=(LN.REAGENT14N_LABEL,))
    lines = LN.cand_lines(SP.ion_counts_of(n, a), 0.001, 0.0, 0.98)
    l14 = _line(lines, LN.REAGENT14N_LABEL)
    assert l14["mode"] == "reagent14N" and l14["ratio"] == pytest.approx(0.02 / 0.98)
    cand = dict(lines=lines)

    def recs(x_m0, role="unexplained", parent=None, reading=(None, None), refuted=False):
        peak = ("x", m0 + l14["d"], x_m0 * 1e6, role, parent, *reading, None)
        ctx = _ctx(_t(rows + [peak] + _edges()), reagent="NO3+NO3_15N")
        if refuted:
            for s in SIDS:
                ctx.files[s].refuted.add("|".join(reading))
        out = LN.eval_candidate(ctx, cand, _obs(m0), "")
        return out, _rec(out, LN.REAGENT14N_LABEL)

    out, r = recs(0.3)                                          # a foreign ion at 0.3 x M0 (15 x the impurity)
    assert [p[0] for p in r["per"]] == ["free"] * 3
    assert (r["n_det"], r["n_test"], r["n_occ"], r["n_ok"], r["n_lo"], r["n_bad"]) == (3, 3, 3, 0, 0, 0)
    assert LN.contradiction_a(out, 3) == []
    assert LN.REAGENT14N_LABEL not in LN.matched_elements(out, 3)[1]       # present, but no positive credit
    out, r = recs(0.005)                                        # a quarter of the impurity level: too low
    assert (r["n_test"], r["n_occ"], r["n_lo"]) == (3, 0, 3)
    assert LN.contradiction_a(out, 3) == [f"{LN.REAGENT14N_LABEL} ({l14['ratio']:.3g}x) too low in 3/3 files"]
    out, r = recs(0.3, role="iso_child", parent="p1")           # the engine's own child: matched
    assert (r["n_test"], r["n_ok"], r["n_occ"]) == (3, 3, 0)
    assert LN.REAGENT14N_LABEL in LN.matched_elements(out, 3)[1]
    other = ("C7H12O3", "[M+Br]-")                              # another reading's M0 on the line
    out, r = recs(0.3, role="M0", reading=other)                # that reading stands: present, not matched
    assert [p[0] for p in r["per"]] == ["occupied"] * 3 and (r["n_ok"], r["n_occ"], r["n_lo"]) == (0, 3, 0)
    assert LN.REAGENT14N_LABEL not in LN.matched_elements(out, 3)[1]
    out, r = recs(0.3, role="M0", reading=other, refuted=True)  # the levels refuted it: its M0 matches
    assert [p[0] for p in r["per"]] == ["free"] * 3 and (r["n_ok"], r["n_occ"], r["n_lo"]) == (3, 0, 0)
    assert LN.REAGENT14N_LABEL in LN.matched_elements(out, 3)[1]


# --------------------------------------------------------------------------- the exact 37Cl test
def _cl37_span(m0):
    d = _line(LN.cand_lines(SP.ion_counts_of(CL, CLA), 0.001, 0.0, 0.98), "37Cl")["d"]
    return m0 + d


def test_probe_exact_37cl_in_window_blend_and_absent():
    m0 = C.ion_mz(CL, CLA)
    x = _cl37_span(m0)
    tol = 1e-6 * m0                                             # the 1 ppm position window
    anchor = ("p1", m0, 1e6, "M0", None, CL, CLA, None)
    ctx = _ctx(_t([anchor, ("k", x + 0.5 * tol, 3.2e5, "iso_child", "p1", None, None, "37Cl")] + _edges()))
    assert LN.probe_exact_37cl(ctx, "a", (x, x), tol, "p1", (), 3.2e5)[:2] == ("free", 3.2e5)
    # 0.5 mDa off (outside the window, inside 1 FWHM = 1 mDa): an unresolved neighbour -> shadowed
    ctx = _ctx(_t([anchor, ("u", x + 0.0005, 3e5, "unexplained", None, None, None, None)] + _edges()))
    assert LN.probe_exact_37cl(ctx, "a", (x, x), tol, "p1", (), 3.2e5)[0] == "shadowed"
    assert ctx.cl37_blend == 1
    # the same neighbour too small to hide the line: absent (no 25 ppm neighbour reach on the 37Cl test)
    ctx = _ctx(_t([anchor, ("u", x + 0.0005, 1e4, "unexplained", None, None, None, None)] + _edges()))
    assert LN.probe_exact_37cl(ctx, "a", (x, x), tol, "p1", (), 3.2e5) == ("absent", 0.0, -1)


def test_probe_exact_37cl_ignores_a_resolved_labelled_reagent_line():
    m0 = C.ion_mz(CL, CLA)
    x = _cl37_span(m0)
    tol = 1e-6 * m0
    rows = [("p1", m0, 1e6, "M0", None, CL, CLA, None),
            ("t", x + 0.8 * tol, 3e5, "M0", None, "C2H4O", "[M+^NO3]-", None)] + _edges()   # a labelled M0
    # FWHM 0.1 mDa < its distance (0.1 mDa x 0.8 ... ~0.1 mDa): resolved -> not the 37Cl line
    hi = _ctx(_t(rows), reagent="NO3+NO3_15N", resolution=ORBI_HI)
    assert hi.fwhm(x) < 0.8 * tol
    assert LN.probe_exact_37cl(hi, "a", (x, x), tol, "p1", (), 3.2e5) == ("absent", 0.0, -1)
    assert hi.cl37_flips == 1
    # at FWHM 1 mDa it is not resolved from the line: it occupies it
    lo = _ctx(_t(rows), reagent="NO3+NO3_15N", resolution=ORBI)
    assert LN.probe_exact_37cl(lo, "a", (x, x), tol, "p1", (), 3.2e5)[0] == "occupied"
    # on an unlabelled run the labelled-position rule never applies
    assert LN.labelled_line_at(_ctx(_t(rows), resolution=ORBI_HI), "a", 1) is None


# --------------------------------------------------------------------------- (c)
def test_carbon_test_contradicts_agrees_and_skips():
    ctx = _ctx(_t(_edges()))
    k = ("X", "[M-H]-")
    assert LN.carbon_test(ctx, k, {"C": 6, "H": 5, "O": 1}) == (None, "")          # no observed count
    ctx.c13[k] = dict(c=6.4, se=0.3, n=5, src="rule C, 5 spectra")
    assert LN.carbon_test(ctx, k, {"C": 6, "H": 5, "O": 1}) == ("agrees", "13C reads 6.4 C for 6")
    v, txt = LN.carbon_test(ctx, k, {"C": 10, "H": 5, "O": 1})
    assert v == "contradicts" and txt == "13C reads 6.4 C for 10 (tol 2.5, rule C, 5 spectra)"
    v, _ = LN.carbon_test(ctx, k, {"H": 5, "N": 3, "O": 1})                         # carbon-free: tested
    assert v == "contradicts"
    ctx.c13[k] = dict(c=6.4, se=2.0, n=1, src="per-file 13C, 1 files")
    assert LN.carbon_test(ctx, k, {"C": 10, "H": 5, "O": 1})[0] == "agrees"          # tol 3 se = 6
    ctx.c13[k] = dict(c=np.nan, se=np.nan, n=0, src="exempt_14N")
    assert LN.carbon_test(ctx, k, {"C": 10}) == (None, "")


# --------------------------------------------------------------------------- (b), q1_isotopes
def _cands(ctx, m0):
    fw = ctx.fwhm(m0)
    J = dict(name=f"{PH} {PHA}", neutral=PH, adduct=PHA, counts=SP.ion_counts_of(PH, PHA), kind="committed",
             lines=ctx.lines(SP.ion_counts_of(PH, PHA), fw, 0.0))
    K = dict(name=f"{CL} {CLA}", neutral=CL, adduct=CLA, counts=SP.ion_counts_of(CL, CLA), kind="mass",
             lines=ctx.lines(SP.ion_counts_of(CL, CLA), fw, 0.0))
    return J, K


def test_b_a_competitors_observed_line_refutes_the_committed_reading():
    m0, rows = _own_lines(CL, CLA, gate=1e3)
    ctx = _ctx(_t(rows + _edges()), gate=1e3)
    J, K = _cands(ctx, m0)
    pair = dict(key=(PH, PHA), pairkey=f"{PH}|{PHA}", obs=_obs(m0), own_pk=set())
    LN.q1_isotopes(ctx, pair, [J, K], 3)
    assert J["a"] == [] and K["a"] == [] and K["b"] == []
    assert len(J["b"]) == 1 and J["b"][0].startswith("37Cl line at 0.32x seen in 3/3 files (in band for C6H5ClO [M-H]-)")
    assert J["contradicted"] and not K["contradicted"] and K["k"] == ""
    assert K["matched_els"] == {"C", "Cl"}


def test_b_a_reading_its_own_a_refutes_does_not_vouch_for_a_line():
    m0, rows = _own_lines(CL, CLA, gate=1e3, skip=("13C",))     # K's 13C line absent: K refuted by (a)
    ctx = _ctx(_t(rows + _edges()), gate=1e3)
    J, K = _cands(ctx, m0)
    pair = dict(key=(PH, PHA), pairkey=f"{PH}|{PHA}", obs=_obs(m0), own_pk=set())
    LN.q1_isotopes(ctx, pair, [J, K], 3)
    assert K["a"] and J["b"] == []


def test_b_line_closer_to_another_readings_m0_is_a_neighbour():
    m0, rows = _own_lines(CL, CLA, gate=1e3)
    ctx = _ctx(_t(rows + _edges()), gate=1e3)
    J, K = _cands(ctx, m0)
    line = next(r for r in rows if r[7] == "37Cl")
    ctx.m0_mz = np.array([line[1] + 1e-5])                      # another reading's M0, nearer than the prediction
    ctx.m0_pk = np.array(["Q|[M-H]-"], dtype=object)
    pair = dict(key=(PH, PHA), pairkey=f"{PH}|{PHA}", obs=_obs(m0), own_pk=set())
    rec = _rec(LN.eval_candidate(ctx, K, pair["obs"], ""), "37Cl")
    assert rec["n_ok"] == 3
    # the observed line sits on the predicted position (offset 0): no M0 is closer -> not a neighbour
    fa = ctx.files["a"]
    j = int(np.searchsorted(fa.mz, line[1]))
    ok = [(o, ("free", line[2], 0, True, True, j)) for o in pair["obs"]]
    assert LN._line_is_neighbour(ctx, ok, pair, _line(K["lines"], "37Cl")) == ""
    shifted = [(o, ("free", line[2], 0, True, True, j)) for o in
               [dict(o, mz=o["mz"] - 4e-5) for o in pair["obs"]]]      # prediction 40 uDa off the observed peak
    assert LN._line_is_neighbour(ctx, shifted, pair, _line(K["lines"], "37Cl")).startswith("another reading's M0")


# --------------------------------------------------------------------------- (k)
def test_twin_test_on_a_labelled_run(monkeypatch):
    n, a = "C6H10O5", "[M+NO3]-"
    m0 = C.ion_mz(n, a)
    ctx = _ctx(_t([("p1", m0, 1e5, "M0", None, n, a, None)] + _edges()), reagent="NO3+NO3_15N")
    ctx.twin_q = {s: 0.3 for s in SIDS}
    cand = dict(neutral=n, adduct=a, counts=SP.ion_counts_of(n, a))
    pair = dict(obs=_obs(m0, h=1e5))
    only_nitrate = [dict(neutral=n, adduct=a, ok=True)]
    monkeypatch.setattr(ctx, "decompositions", lambda *a_, **k: only_nitrate)
    assert LN.twin_test(ctx, pair, cand, 3) == "15N twin (+0.997, >= q_lo x) absent or too low in 3/3 files"
    twin = _ctx(_t([("p1", m0, 1e5, "M0", None, n, a, None),
                    ("t1", m0 + LN.D15N_TWIN, 3e4, "M0", None, n, "[M+^NO3]-", None)] + _edges()),
                reagent="NO3+NO3_15N")
    twin.twin_q = ctx.twin_q
    monkeypatch.setattr(twin, "decompositions", lambda *a_, **k: only_nitrate)
    assert LN.twin_test(twin, pair, cand, 3) == ""
    # another plausible decomposition: the twin says nothing about the ion
    monkeypatch.setattr(ctx, "decompositions",
                        lambda *a_, **k: only_nitrate + [dict(neutral="C6H11NO8", adduct="[M-H]-", ok=True)])
    assert LN.twin_test(ctx, pair, cand, 3) == ""
    assert LN.twin_test(ctx, pair, dict(cand, adduct="[M-H]-"), 3) == ""


# --------------------------------------------------------------------------- isotope-line competitors
def test_isoline_competitor_listed_explained_or_excluded():
    pn, pa = "C6H10O5", "[M+NO3]-"
    mp = C.ion_mz(pn, pa)
    d13 = _line(LN.cand_lines(SP.ion_counts_of(pn, pa), 0.002, 0.0, 0.98), "13C")
    sids = ("a", "b", "c", "d", "e")

    def run(heights):
        pf = {s: _t([("P", mp, 1e6, "M0", None, pn, pa, None),
                     ("Q", mp + d13["d"], h, "M0", None, "C9H6N", "[M-H]-", None)] + _edges())
              for s, h in zip(sids, heights)}
        ctx = _ctx(pf)
        CX.prepare_indexes(ctx, pf)
        pair = dict(key=("C9H6N", "[M-H]-"), pairkey="C9H6N|[M-H]-", own_pk=set(),
                    obs=[dict(sid=s, mz=mp + d13["d"], h=h, area=h, pid="Q") for s, h in zip(sids, heights)])
        return LN.isoline_competitors(ctx, pair, 3)

    exp = 1e6 * d13["ratio"]
    comps, expl = run([0.8 * exp] * 5)
    assert len(comps) == 1 and expl == set(sids)
    c = comps[0]
    assert c["name"] == f"13C line of {pn} {pa}" and c["kind"] == "isoline" and not c["excluded"]
    assert re.fullmatch(rf"13C line of {pn} \[M\+NO3\]- \(pred/obs median 1\.\d\d\) explains >= half the peak "
                        r"in 5/5 files; > 2x in 0", c["why"])
    comps, expl = run([0.8 * exp] + [10 * exp] * 4)              # listed in one file, > 2x in 4/5
    assert comps[0]["excluded"] and expl == {"a"}
    assert comps[0]["why"] == f"isotope height: the peak is > 2x {pn} {pa}'s predicted 13C line in 4/5 files"
    comps, _ = run([10 * exp] * 5)                               # never explains half the peak: not listed
    assert comps == []


def test_isoline_competitor_reads_no_artifact_as_the_other_readings_m0():
    """In a file where the other reading P is not committed, its predicted line is
    sized off the tallest peak at P's position -- never off an artifact row there: a
    taller side lobe beside a weaker real peak does not inflate the prediction, and a
    position only an artifact holds predicts nothing."""
    pn, pa = "C6H10O5", "[M+NO3]-"
    mp = C.ion_mz(pn, pa)
    d13 = _line(LN.cand_lines(SP.ion_counts_of(pn, pa), 0.002, 0.0, 0.98), "13C")
    sids = ("a", "b", "c", "d", "e")
    exp = 1e6 * d13["ratio"]
    q = ("Q", mp + d13["d"], 0.8 * exp, "M0", None, "C9H6N", "[M-H]-", None)

    def run(last):
        pf = {s: _t([("P", mp, 1e6, "M0", None, pn, pa, None), q] + _edges()) for s in sids[:-1]}
        pf["e"] = _t(last + [q] + _edges())
        ctx = _ctx(pf)
        CX.prepare_indexes(ctx, pf)
        pair = dict(key=("C9H6N", "[M-H]-"), pairkey="C9H6N|[M-H]-", own_pk=set(),
                    obs=[dict(sid=s, mz=q[1], h=q[2], area=q[2], pid="Q") for s in sids])
        return LN.isoline_competitors(ctx, pair, 3)

    lobe = ("lobe", mp + 0.2e-6 * mp, 1e6, "artifact", None, None, None, None)
    # a weak real peak at P's position plus a taller lobe: sized off the real peak (0.1x): not explained in e
    _, expl = run([("x", mp, 1e5, "unexplained", None, None, None, None), lobe])
    assert expl == {"a", "b", "c", "d"}
    # only the lobe holds P's position: no prediction there
    _, expl = run([lobe])
    assert expl == {"a", "b", "c", "d"}
    # the same peak not marked artifact is read as P's line in e
    _, expl = run([lobe[:3] + ("unexplained",) + lobe[4:]])
    assert expl == set(sids)


def test_the_neighbour_reach_takes_no_artifact_as_a_displaced_line():
    """An empty position window: a line displaced toward a >= 3x taller neighbour within
    25 ppm still counts (the reach) -- but an artifact there is no line. With a shadow
    window the span reads 'shadowed' (untestable), never 'free' off the artifact."""
    m0, _ = _own_lines(PH, PHA)
    d13 = _line(LN.cand_lines(SP.ion_counts_of(PH, PHA), 0.001, 0.0, 0.98), "13C")
    x13 = m0 + d13["d"]
    h_exp = 1e6 * d13["ratio"]
    tol = 1e-6 * x13

    def probe(role):
        rows = [("p1", m0, 1e6, "M0", None, PH, PHA, None),
                ("disp", x13 + 4e-6 * x13, h_exp, role, None, None, None, None),
                ("nb", x13 + 12e-6 * x13, 10 * h_exp, "unexplained", None, None, None, None)] + _edges()
        ctx = _ctx(_t(rows))
        return ctx.files["a"].probe((x13, x13), tol, "p1", shadow_da=30e-6 * x13, h_exp=h_exp)[0]

    assert probe("unexplained") == "free"          # a displaced line, credited by the reach
    assert probe("artifact") == "shadowed"         # a displaced artifact: untestable, not the line


# --------------------------------------------------------------------------- pass B
def test_pass_b_competitor_rows_and_facts():
    n, a = "C6H10O5", "[M+NO3]-"
    m0 = C.ion_mz(n, a)
    ctx = _ctx(_t([("p1", m0, 1e6, "M0", None, n, a, None)] + _edges()))
    facts = pd.DataFrame([dict(neutral_formula=n, adduct=a, mz=m0, ion="", ppm=0.0, ion_only=False, lead=True)])
    frames = {s: _t([("p1", m0, 1e6, "M0", None, n, a, None)] + _edges()) for s in SIDS}
    pairs = CP.pairs_from_files(frames, facts)
    assert len(pairs[(n, a)]["obs"]) == len(frames)
    base = dict(a=[], b=[], c="", c_agree="", k="", matched_els=set(), matched_labels=[], contradicted=False,
                stay="no testable isotope line", win="run", has_testable=False)
    com = dict(base, name=f"{n} {a}", neutral=n, adduct=a, counts=SP.ion_counts_of(n, a), kind="committed",
               ion_key="C6H10N1O8", ppm=0.0, relax_note="", enum_ok=True, matched_els={"C"}, matched_labels=["13C"],
               c_agree="13C reads 6.1 C for 6")
    ex = dict(base, name="C5H6N2O6 [M-H]-", neutral="C5H6N2O6", adduct="[M-H]-", kind="mass", ppm=0.4,
              a=["13C (0.055x) absent in 3/3 files"], contradicted=True)
    left = dict(base, name="C7H14O4 [M+NO3]-", neutral="C7H14O4", adduct="[M+NO3]-", kind="tie", ppm=-0.2,
                win="", stay="13C 0.077x testable in 1 files")
    iso = dict(name="13C line of C5H8O4 [M+NO3]-", neutral="C5H8O4", adduct="[M+NO3]-", kind="isoline", ppm=0.1,
               excluded=False, why="13C line of ... explains", explained_files=3, n_files=3)
    q1 = {(n, a): dict(cands=[com, ex, left], n_alt=1, outside=None, n_tie_same=0, isolines=[iso],
                       iso_explained=["a", "b"])}
    rows, comps = CP.pass_b(ctx, facts, pairs, q1)
    r = rows.iloc[0]
    assert r["n_competitors"] == 3 and r["n_excl_iso"] == 1 and r["n_left"] == 2 and r["n_excl_routes"] == 0
    assert r["competitors_left"] == "C7H14O4 [M+NO3]-; 13C line of C5H8O4 [M+NO3]-"
    assert r["competitors_excluded"] == "C5H6N2O6 [M-H]-: iso(a): 13C (0.055x) absent in 3/3 files"
    assert r["committed_matched"] == "C" and r["committed_matched_lines"] == "13C"
    assert r["committed_c13"] == "13C reads 6.1 C for 6" and not r["committed_contradicted"]
    assert not r["no_comp_info"] and r["committed_plausible"] and r["lead_fact"] and not r["ion_only_reading"]
    assert r["n_comp_isoline"] == 1 and r["n_isoline_left"] == 1 and r["iso_explained_files"] == "a|b"
    assert r["n_comp_mass"] == 1 and r["n_comp_alt"] == 1 and r["n_files_obs"] == 3
    assert list(comps.columns) == list(CP.COMP_COLUMNS)
    assert list(comps["status"]) == ["excluded", "left", "left"] and list(comps["how"]) == ["isotopes", "", ""]
    assert comps.iloc[1]["why"] == "13C 0.077x testable in 1 files"
    assert comps.iloc[2]["comp_neutral"] == "" and comps.iloc[2]["window"] == ""      # an isoline has no neutral
    # an unreadable committed reading has no competitor information
    q1[(n, a)]["cands"][0] = dict(com, ion_key=None)
    assert CP.pass_b(ctx, facts, pairs, q1)[0].iloc[0]["no_comp_info"]


def test_q1_pass_end_to_end_on_a_small_source():
    n, a = "C6H10O5", "[M+NO3]-"
    m0, rows = _own_lines(n, a)
    pf = {s: _t(rows + _edges()) for s in SIDS}
    ctx = _ctx(pf)
    CX.prepare_indexes(ctx, pf)
    facts = pd.DataFrame([dict(neutral_formula=n, adduct=a, mz=m0, ion="", ppm=0.0, ion_only=False, lead=False)])
    pairs = CP.pairs_from_files(pf, facts)
    q1 = CP.q1_pass(ctx, pairs, 3)
    com = q1[(n, a)]["cands"][0]
    assert com["kind"] == "committed" and com["enum_ok"] and not com["contradicted"]
    assert "C" in com["matched_els"] and "13C" in com["matched_labels"]
    assert all(c["kind"] != "committed" and c["name"] != f"{n} {a}" for c in q1[(n, a)]["cands"][1:])
    assert all("lines" not in c and "recs" not in c for c in q1[(n, a)]["cands"])
    rows_b, comps = CP.pass_b(ctx, facts, pairs, q1)
    assert rows_b.iloc[0]["n_competitors"] == len(comps) == len(q1[(n, a)]["cands"]) - 1
