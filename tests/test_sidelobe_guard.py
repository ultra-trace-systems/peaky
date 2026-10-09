"""The Orbitrap same-spectrum side-lobe guard (chem/sidelobes.py + the ledger
step assignment/sidelobe_guard.py, run by assign.run before pass 0).

Synthetic peak lists: a width model, bright isolated lines at a normal observed
width (0.9x the model, the file's calibration), dim fillers that set the noise
edge, and the peaks under test placed in units of the parent's width."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from peaky.assignment import assign as A
from peaky.assignment import ledger as L
from peaky.assignment import sidelobe_guard as SG
from peaky.assignment.passes.config import PassConfig
from peaky.chem import sidelobes as SL
from peaky.chem.resolution import Resolution

# an Orbitrap-like model, R = 120 000 at m/z 200, FWHM ~ m^1.5, MEASURED
ORB = Resolution(coef=(200.0 / 120_000) / 200.0 ** 1.5, exponent=1.5, n_peaks=40, source="measured")
W0 = 0.9                                   # the file's observed / model width ratio


def quiet(*a, **k):
    pass


def fw(mz, model=ORB, w=W0):
    """A normal line's observed FWHM at mz (Th)."""
    return w * float(SL.model_fwhm(model, mz))


def _pk(pid, mz, h, w=W0, model=ORB):
    return {"peak_id": pid, "mz": float(mz), "height": float(h),
            "area": SL.GAUSS_AREA * float(h) * w * float(SL.model_fwhm(model, mz))}


def _base(model=ORB, w=W0, n_bright=12, width_of=None):
    """Bright isolated lines (calibrate the widths) + dim fillers (the noise edge)."""
    rows = []
    for k in range(n_bright):
        mz = 108.05 + 23.17 * k
        ww = width_of(mz) if width_of else w
        rows.append(_pk(f"B{k}", mz, 2e4, ww, model))
    for k in range(40):
        mz = 101.37 + 7.31 * k
        ww = width_of(mz) if width_of else w
        rows.append(_pk(f"f{k}", mz, 3.0 + (k % 5), ww, model))
    return rows


def _table(extra, **kw):
    return pd.DataFrame(_base(**kw) + list(extra))


def _flags(t, model=ORB, **kw):
    r = SL.sidelobe_parents(t["mz"], t["height"], t["area"], model, **kw)
    pid = t["peak_id"].to_numpy()
    return {pid[i]: (pid[r["parent"][i]], r["signature"][i], r["offset_fwhm"][i])
            for i in np.flatnonzero(r["parent"] >= 0)}, r


P = 250.0                                  # the test parent's m/z (between bright lines)


def _parent(h=1e4, mz=P):
    return _pk("P", mz, h)


# --------------------------------------------------------------------------- the rule
def test_a_narrow_phantom_below_a_bright_line_is_flagged():
    t = _table([_parent(), _pk("lobe", P - 2.4 * fw(P), 100.0, w=0.45 * W0)])
    f, r = _flags(t)
    assert set(f) == {"lobe"} and f["lobe"][0] == "P"
    assert "narrow" in f["lobe"][1] and "neg-lobe" in f["lobe"][1]
    assert f["lobe"][2] == pytest.approx(-2.4, abs=0.01)
    assert r["calibration"]["ratio"] == pytest.approx(W0) and r["calibration"]["measured"]


def test_a_narrow_peak_anywhere_in_the_band_is_flagged_and_a_normal_one_off_the_lobe_is_not():
    t = _table([_parent(), _pk("narrow", P + 3.0 * fw(P), 50.0, w=0.4 * W0)])
    assert set(_flags(t)[0]) == {"narrow"}
    # normal width, at the negative lobe's offset: the offset alone is the signature
    t = _table([_parent(), _pk("neg", P - 2.35 * fw(P), 50.0)])
    assert _flags(t)[0]["neg"][1] == "neg-lobe"


def test_a_real_line_of_normal_width_at_ratio_20_is_kept():
    t = _table([_parent(), _pk("real", P - 2.35 * fw(P), 1e4 / 20)])
    f, r = _flags(t)
    assert f == {}


def test_a_normal_line_at_ratio_60_and_plus_4_fwhm_is_kept_unless_mirrored():
    side = _pk("side", P + 4.0 * fw(P), 1e4 / 60)
    f, r = _flags(_table([_parent(), side]))
    assert f == {} and r["n_no_signature"] == 1
    # a mirror lobe on the other side of the same parent is the signature
    mirror = _pk("mirror", P - 3.6 * fw(P), 1e4 / 80)
    f, r = _flags(_table([_parent(), side, mirror]))
    assert set(f) == {"side", "mirror"} and all("mirror" in v[1] for v in f.values())
    assert r["n_mirror"] == 2


def test_the_band_edges_are_in_parent_widths():
    near = _pk("near", P - 0.8 * fw(P), 10.0, w=0.4 * W0)              # below 1 FWHM: resolvability's
    far = _pk("far", P + 6.0 * fw(P), 10.0, w=0.4 * W0)                # beyond 5 FWHM
    assert _flags(_table([_parent(), near, far]))[0] == {}


def test_the_offset_scales_with_the_width_not_with_mDa():
    lo, hi = 100.2, 399.7
    t = _table([_parent(mz=lo), _pk("lo", lo - 2.35 * fw(lo), 100.0, w=0.45 * W0),
                _pk("P2", hi, 1e4), _pk("hi", hi - 2.35 * fw(hi), 100.0, w=0.45 * W0)])
    f, _ = _flags(t)
    assert set(f) == {"lo", "hi"}
    assert abs(t.loc[t.peak_id == "hi", "mz"].iloc[0] - hi) > 6 * abs(t.loc[t.peak_id == "lo", "mz"].iloc[0] - lo)
    # a fixed -4 mDa offset is ~8 FWHM at m/z 100: not a lobe there
    assert _flags(_table([_parent(mz=lo), _pk("x", lo - 4e-3, 100.0, w=0.45 * W0)]))[0] == {}


def test_only_the_weaker_of_a_pair_is_flagged_and_the_brightest_parent_is_named():
    t = _table([_parent(), _pk("twin", P + 2.0 * fw(P), 1e4, w=0.45 * W0)])
    assert _flags(t)[0] == {}                                             # equal peaks
    a, b = P, P + 6.5 * fw(P)
    mid = a + 3.25 * fw(P)                                                # in band of both
    t = _table([_pk("A", a, 1e4), _pk("Bp", b, 3e4), _pk("m", mid, 50.0, w=0.4 * W0)])
    f, _ = _flags(t)
    assert set(f) == {"m"} and f["m"][0] == "Bp"


def test_a_2h_fine_structure_line_of_a_bright_ions_13c_line_is_exempt():
    """C10H17-type ion: its 2H line sits +2.92 mDa above its 13C line at ~1/56
    of it -- a positive lobe's offset and ratio. The 13C line is itself an
    isotopologue line of an even brighter peak, so the 2H line is spared."""
    mono = 137.1325
    c13 = mono + 1.0033548
    h2 = mono + 1.0062767
    rows = [_pk("M", mono, 1e5), _pk("C13", c13, 1.1e4), _pk("D", h2, 1.1e4 / 56, w=0.5 * W0)]
    f, r = _flags(_table(rows))
    assert "D" not in f and r["n_exempt"] == {"2H-13C": 1}
    # without the brighter mono the 13C line is not an isotopologue line: flagged
    f, r = _flags(_table(rows[1:]))
    assert "D" in f and r["n_exempt"] == {}


def test_a_c20_single_n_ions_15n_line_is_exempt_not_a_negative_lobe():
    """A bright single-N C20 ion near m/z 460-490 (a nitrate cluster of an
    N-free dimer, an [M+NH4]+ of a C20 compound): its 15N line sits -6.32 mDa
    below its 13C line at 13C/15N = 2.95 nC/nN ~ 59 -- at -2.3..-2.4 FWHM, inside
    the negative-lobe window, at a lobe's ratio. It is an isotope line: kept."""
    model = Resolution(coef=1.232e-7, exponent=1.65, n_peaks=10, source="measured")
    w = 0.86
    for mono in (460.15, 474.19, 489.17):
        c13 = mono + 1.0033548
        n15 = mono + 0.9970349
        h2 = mono + 1.0062767
        rows = [_pk("M", mono, 2e5, w, model), _pk("C13", c13, 2e5 * 0.216, w, model),
                _pk("N15", n15, 2e5 * 0.00367, w, model)]
        f, r = _flags(_table(rows, model=model, w=w), model=model)
        off = (n15 - c13) / fw(c13, model, w)
        assert SL.NEG_LOBE_FWHM[0] <= off <= SL.NEG_LOBE_FWHM[1]          # where a lobe would be
        assert f == {} and r["n_exempt"] == {"15N-13C": 1}
        # with its 2H line on the other side of the 13C line: both are isotope
        # lines, and neither is the other's mirror lobe
        f, r = _flags(_table(rows + [_pk("D", h2, 2e5 * 0.216 / 58, w, model)], model=model, w=w),
                      model=model)
        assert f == {} and r["n_exempt"]["15N-13C"] == 1
        # without the brighter mono the 13C line is not an isotopologue line: a lobe
        f, _ = _flags(_table(rows[1:], model=model, w=w), model=model)
        assert f["N15"][0] == "C13" and f["N15"][1] == "neg-lobe"


def test_an_n_rich_ions_13c_line_beside_its_brighter_15n_line_is_exempt():
    mono = 300.05
    n15 = mono + 0.9970349
    c13 = mono + 1.0033548
    # the weak 13C line picked narrow (a signature): the spacing exempts it
    rows = [_pk("M", mono, 1e6), _pk("N15", n15, 6e4), _pk("C13", c13, 6e4 / 55, w=0.5 * W0)]
    f, r = _flags(_table(rows))
    assert 1.0 <= (c13 - n15) / fw(n15) <= 5.0                              # in band
    assert f == {} and r["n_exempt"] == {"13C-15N": 1}
    f, _ = _flags(_table(rows[1:]))                                         # no brighter mono: a lobe
    assert f["C13"][0] == "N15" and f["C13"][1] == "narrow"


def test_an_exempt_isotope_line_is_not_mirror_evidence():
    """The 2H line of a bright ion's 13C line is exempt; a real normal-width line
    on the other side of the 13C line (no narrow, no negative-lobe offset) must
    not read the 2H line as its mirror lobe."""
    mono = 137.1325
    c13 = mono + 1.0033548
    rows = [_pk("M", mono, 1e5), _pk("C13", c13, 1.1e4), _pk("D", mono + 1.0062767, 1.1e4 / 56),
            _pk("other", c13 - 3.5 * fw(c13), 1.1e4 / 60)]
    f, r = _flags(_table(rows))
    assert f == {} and r["n_exempt"] == {"2H-13C": 1} and r["n_no_signature"] == 1
    # a non-exempt line in the 2H line's place IS mirror evidence
    rows[2] = _pk("D", c13 + 4.2 * fw(c13), 1.1e4 / 56)
    f, _ = _flags(_table(rows))
    assert set(f) == {"D", "other"} and "mirror" in f["other"][1]


def test_the_exemption_needs_the_fine_structure_offset():
    """A narrow lobe of a 13C line (itself an isotopologue line of a brighter
    mono) at an offset that is no fine-structure spacing is still a lobe."""
    mono = 137.1325
    c13 = mono + 1.0033548
    rows = [_pk("M", mono, 1e5), _pk("C13", c13, 1.1e4),
            _pk("lobe", c13 - 2.4 * fw(c13), 1.1e4 / 56, w=0.45 * W0)]
    f, r = _flags(_table(rows))
    assert set(f) == {"lobe"} and f["lobe"][0] == "C13" and r["n_exempt"] == {}


D13C = SL.ISO.D_13C_EXACT


def _neighbour_13c(ratio=0.10, w=W0, k_h=2300.0, bright=None):
    """A neighbouring ion K at m/z 173.1172 and a line J ~100x brighter than K's 13C
    line, placed so that the 13C line sits at J's negative-lobe offset."""
    k = 173.1172
    c13 = k + D13C
    j = c13 + 2.35 * fw(c13)
    rows = [_pk("K", k, k_h), _pk("J", j, 100.0 * ratio * k_h), _pk("C13", c13, ratio * k_h, w=w)]
    if bright:
        rows.append(_pk("KB", k - 2.4 * fw(k), bright))
    return rows


def test_the_13c_line_of_a_neighbouring_ion_beside_a_brighter_line_is_exempt():
    """The real 13C line of a neighbouring ion (m/z 174.121, the 13C line of 173.117 at
    0.10x) two widths below a line ~100x brighter, at the negative lobe's offset: a line's
    width, on an exact 13C offset of a brighter line clear of every lobe band, at a share
    a C9 ion makes -- an isotope line, not a lobe."""
    rows = _neighbour_13c()
    f, r = _flags(_table(rows))
    assert f == {} and r["n_exempt"] == {"13C line": 1}
    # without the neighbour it is J's negative lobe
    f, r = _flags(_table([x for x in rows if x["peak_id"] != "K"]))
    assert set(f) == {"C13"} and f["C13"][1] == "neg-lobe" and r["n_exempt"] == {}


def test_the_isotopologue_exemption_needs_a_lines_width_a_plausible_share_and_a_clear_line():
    # narrower than a line: a lobe, whatever it sits on
    f, _ = _flags(_table(_neighbour_13c(w=0.45 * W0)))
    assert set(f) == {"C13"} and "narrow" in f["C13"][1]
    # 0.30x of an m/z 173 line would need ~28 carbons (cap 1.5 x 173/12 x 1.07 %): a lobe
    f, r = _flags(_table(_neighbour_13c(ratio=0.30)))
    assert set(f) == {"C13"} and r["n_exempt"] == {}
    # a "brighter line" that sits in the lobe band of an even brighter one exempts nothing
    f, r = _flags(_table(_neighbour_13c(bright=2300.0 * 60)))
    assert "C13" in f and r["n_exempt"] == {}
    # the offset must be the 13C spacing: 1 mDa off (~6 ppm, ~0.8 FWHM) it is a lobe
    rows = _neighbour_13c()
    rows[0] = _pk("K", 173.1172 - 1e-3, 2300.0)
    f, _ = _flags(_table(rows))
    assert set(f) == {"C13"}
    # ... within the exact-offset window (SL.ISO_LINK_PPM, 1 ppm): 0.8 ppm off it is
    # spared, 1.8 ppm off (a lobe population picked off a neighbour's 13C offset) it is not
    for dppm, spared in ((0.8, True), (1.8, False)):
        rows = _neighbour_13c()
        rows[0] = _pk("K", 173.1172 - dppm * 1e-6 * (173.1172 + D13C), 2300.0)
        f, _ = _flags(_table(rows))
        assert (f == {}) is spared, (dppm, f)


def test_a_lobe_of_an_ions_13c_line_is_not_spared_by_the_ions_own_lobe():
    """Lobes repeat one isotope spacing apart at the parents' own ratio: the negative lobe
    of an ion's 13C line sits one 13C spacing above the ion's own negative lobe, at the 13C
    share. That 'brighter line' is in the ion's lobe band, so both stay lobes."""
    mono = 200.0
    c13 = mono + D13C
    rows = [_pk("M", mono, 1e5), _pk("C13", c13, 1.1e4),
            _pk("L0", mono - 2.35 * fw(mono), 1e5 / 150), _pk("L1", c13 - 2.35 * fw(c13), 1.1e4 / 150)]
    f, r = _flags(_table(rows))
    assert set(f) == {"L0", "L1"} and r["n_exempt"] == {}


def test_a_real_line_at_ratio_40_in_the_negative_lobe_window_is_kept():
    t = _table([_parent(), _pk("real", P - 2.35 * fw(P), 1e4 / 40)])
    assert _flags(t)[0] == {}
    t = _table([_parent(), _pk("lobe", P - 2.35 * fw(P), 1e4 / 51)])
    assert set(_flags(t)[0]) == {"lobe"}


def test_a_parent_of_implausible_own_width_is_measured_by_the_scaled_model():
    """The band is in the parent's own width only when that width is plausible
    (0.6-1.6x the file's typical); a parent 3x too wide (a blend, a saturated
    line) falls back on the model x the file's median -- not the raw model."""
    lobe_at = P - 2.4 * fw(P)
    t = _table([_pk("P", P, 1e4, w=3.0), _pk("lobe", lobe_at, 100.0, w=0.45 * W0)])
    f, _ = _flags(t)
    assert set(f) == {"lobe"} and f["lobe"][2] == pytest.approx(-2.4, abs=0.01)
    # a file whose lines are 0.6x the model: the fallback is 0.6x the model, so a
    # normal-width line at -2.35 of THAT is at the negative lobe (-1.41 raw-model
    # widths, where it would have no signature)
    w = 0.6
    t = _table([_pk("P", P, 1e4, w=3.0), _pk("neg", P - 2.35 * fw(P, w=w), 1e4 / 60, w=w)], w=w)
    f, r = _flags(t)
    assert r["calibration"]["ratio"] == pytest.approx(w)
    assert set(f) == {"neg"} and f["neg"][1] == "neg-lobe"
    assert f["neg"][2] == pytest.approx(-2.35, abs=0.01)


def test_narrow_is_relative_to_the_files_typical_width():
    """A file whose lines are 0.6x the model: a line of THAT width is normal, not
    narrow (an absolute 0.7x-the-model cut would flag it); 0.4x of it is."""
    w = 0.6
    t = _table([_parent(), _pk("normal", P + 3.0 * fw(P, w=w), 1e4 / 60, w=w)], w=w)
    f, r = _flags(t)
    assert f == {} and r["n_no_signature"] == 1
    t = _table([_parent(), _pk("narrow", P + 3.0 * fw(P, w=w), 1e4 / 60, w=0.4 * w)], w=w)
    f, _ = _flags(t)
    assert set(f) == {"narrow"} and f["narrow"][1] == "narrow"


@pytest.mark.parametrize("w, runs", [(0.4, False), (0.55, True), (1.9, True), (2.3, False)])
def test_the_calibration_range_has_two_ends(w, runs):
    t = _table([_pk("P", P, 1e4, w=w), _pk("lobe", P - 2.4 * fw(P, w=w), 100.0, w=0.45 * w)], w=w)
    f, r = _flags(t)
    if runs:
        assert r["skipped"] is None and set(f) == {"lobe"}
    else:
        assert "outside 0.5-2" in r["skipped"] and f == {}


def test_an_uncalibrated_measured_model_uses_the_geometric_signatures_only():
    """A measured model with < 10 bright peaks cannot be scaled by the file's
    widths: 'narrow' would be read against the raw model -- 1.5x too wide here,
    so every line would read 0.67 -- and is not used. The negative lobe and the
    mirror still are."""
    w = 1 / 1.5
    real = _pk("real", P + 3.0 * fw(P, w=w), 1e4 / 60, w=w)            # a normal line
    narrow = _pk("narrow", P + 3.0 * fw(P, w=w), 1e4 / 60, w=0.4 * w)
    neg = _pk("neg", P - 2.35 * fw(P, w=w), 1e4 / 60, w=0.4 * w)
    for extra, want in (([real], {}), ([narrow], {}), ([neg], {"neg": "neg-lobe"})):
        t = _table([_pk("P", P, 1e4, w=w)] + extra, w=w, n_bright=3)
        f, r = _flags(t)
        assert r["skipped"] is None and not r["calibration"]["measured"]
        assert r["signatures"] == ("neg-lobe", "mirror")
        assert {k: v[1] for k, v in f.items()} == want
    # the ledger step says so
    msgs = []
    led = L.new_ledger(_table([_pk("P", P, 1e4, w=w), neg], w=w, n_bright=3))
    out = SG.flag_orbitrap_sidelobes(led, ORB, _cfg(), log=msgs.append)
    assert out["flagged"] == 1 and out["signatures"] == ["neg-lobe", "mirror"]
    assert out["calibration"]["measured"] is False
    assert any("UNCALIBRATED" in m and "'narrow' signature off" in m for m in msgs)
    # calibrated, all three are tested
    assert _flags(_table([_parent()]))[1]["signatures"] == SL.SIGNATURES


def test_the_config_defaults_are_the_rules_constants():
    cfg = PassConfig()
    assert cfg.sidelobe_ratio == SL.SIDELOBE_RATIO
    assert tuple(cfg.sidelobe_band_fwhm) == SL.SIDELOBE_BAND_FWHM
    assert cfg.sidelobe_narrow == SL.SIDELOBE_NARROW


# --------------------------------------------------------------------------- the skips
def _ledger(extra, **kw):
    return L.new_ledger(_table(extra, **kw))


LOBE = [_parent(), _pk("lobe", P - 2.4 * fw(P), 100.0, w=0.45 * W0)]


def _cfg(**kw):
    kw.setdefault("instrument_class", "orbitrap")
    return PassConfig(**kw)


@pytest.mark.parametrize("cfg_kw, why", [
    (dict(sidelobe_guard=False), "off"),
    (dict(instrument_class="tof"), "not an Orbitrap"),
    (dict(instrument_class=None), "not an Orbitrap"),
    (dict(trace_sample=True), "trace sample"),
])
def test_the_class_and_switch_gates(cfg_kw, why):
    led = _ledger(LOBE)
    out = SG.flag_orbitrap_sidelobes(led, ORB, _cfg(**cfg_kw), log=quiet)
    assert out["flagged"] == 0 and out["skipped"].startswith(why)
    assert (led["role"] == L.ROLE_UNEXPLAINED).all() and not led["locked"].any()


def test_a_tof_roster_with_an_orbitrap_class_r_is_skipped():
    from peaky.batch import assign_batch as AB
    rp = Resolution.from_r(250_000)
    klass = AB._axis_class(AB._instrument_of(rp)[0], "tof")          # the batch's resolution
    assert klass == "tof"
    out = SG.flag_orbitrap_sidelobes(_ledger(LOBE), rp, _cfg(instrument_class=klass), log=quiet)
    assert out["skipped"].startswith("not an Orbitrap")
    # a single-sample run: a declared number is not evidence of the class
    assert A.instrument_class_of(None, rp) is None


def test_no_width_model_and_no_area_are_skipped():
    out = SG.flag_orbitrap_sidelobes(_ledger(LOBE), None, _cfg(), log=quiet)
    assert out["skipped"] == "no width model"
    led = L.new_ledger(_table(LOBE).drop(columns=["area"]))
    out = SG.flag_orbitrap_sidelobes(led, ORB, _cfg(), log=quiet)
    assert out["skipped"].startswith("no area column") and not led["locked"].any()


def test_a_declared_r_the_observed_widths_cannot_calibrate_is_skipped():
    declared = Resolution.from_r(120_000)                  # FWHM ~ m, the instrument's ~ m^1.5
    out = SG.flag_orbitrap_sidelobes(_ledger(LOBE), declared, _cfg(), log=quiet)
    assert out["skipped"].startswith("declared width model") and "trends with m/z" in out["skipped"]
    # ... nor with too few bright peaks to calibrate from
    out = SG.flag_orbitrap_sidelobes(_ledger(LOBE, n_bright=3), declared, _cfg(), log=quiet)
    assert out["skipped"].startswith("declared width model") and "bright peaks" in out["skipped"]
    # a declared model the widths DO follow (their m/z dependence, any scale) runs
    rows = [_pk("P", P, 1e4, model=declared, w=0.8),
            _pk("lobe", P - 2.4 * 0.8 * P / 120_000, 100.0, model=declared, w=0.36)]
    led = L.new_ledger(_table(rows, model=declared, w=0.8))
    out = SG.flag_orbitrap_sidelobes(led, declared, _cfg(), log=quiet)
    assert out["skipped"] is None and out["flagged"] == 1


def test_a_file_whose_widths_disagree_with_the_model_is_skipped():
    out = SG.flag_orbitrap_sidelobes(_ledger(LOBE, w=3.0), ORB, _cfg(), log=quiet)
    assert "outside 0.5-2" in out["skipped"]


# --------------------------------------------------------------------------- the action
def test_flagged_rows_are_artifact_locked_and_cannot_be_committed():
    led = _ledger(LOBE + [_pk("held", P + 3.0 * fw(P), 60.0, w=0.4 * W0)])
    i_held = L._row_index(led, "held")
    L.mark_reagent(led, "held", "reagent ion")                          # not unexplained: untouched
    out = SG.flag_orbitrap_sidelobes(led, ORB, _cfg(), log=quiet)
    assert out["flagged"] == 1 and out["skipped"] is None and out["by_signature"]["narrow"] == 1
    row = led.loc[L._row_index(led, "lobe")]
    assert row["role"] == L.ROLE_ARTIFACT and bool(row["locked"])
    assert f"m/z {P:.4f}" in row["commentary"] and "100x brighter" in row["commentary"]
    assert "-2.40 FWHM" in row["commentary"] and "mDa" in row["commentary"]
    assert led.at[i_held, "role"] == L.ROLE_REAGENT and not bool(led.at[i_held, "locked"])
    assert L.validate(led) == []
    with pytest.raises(L.LedgerError):
        L.commit_assignment(led, "lobe", neutral_formula="C9H8O4", adduct="[M-H]-", ion_score=0.9,
                            pass_no=1, method="test", confidence="High", commentary="test")
    L.commit_assignment(led, "P", neutral_formula="C10H16O6", adduct="[M-H]-", ion_score=0.9,
                        pass_no=1, method="test", confidence="High", commentary="test")
    with pytest.raises(L.LedgerError):
        L.attach_isotopologue(led, "lobe", "P", iso_label="13C")
    assert L.stats(led)                                                   # still a healthy ledger


def test_an_already_locked_row_is_left_alone():
    led = _ledger(LOBE)
    L.lock_peaks(led, ["lobe"])
    out = SG.flag_orbitrap_sidelobes(led, ORB, _cfg(), log=quiet)
    assert out["flagged"] == 0 and led.loc[L._row_index(led, "lobe"), "role"] == L.ROLE_UNEXPLAINED


def test_the_guard_compares_lines_of_one_spectrum_only():
    lobe = _pk("lobe", P - 2.4 * fw(P), 100.0, w=0.45 * W0)
    with_parent, without = _ledger([_parent(), lobe]), _ledger([lobe])
    assert SG.flag_orbitrap_sidelobes(with_parent, ORB, _cfg(), log=quiet)["flagged"] == 1
    assert SG.flag_orbitrap_sidelobes(without, ORB, _cfg(), log=quiet)["flagged"] == 0


# --------------------------------------------------------------------------- assign.run wiring
SID = "offline-sidelobe-guard"
ION = "C10H15O6"                     # [M-H]- of C10H16O6


def _run_table(lobes=True):
    from mascope_tools.composition.heuristic_filter import anchor_on_monoisotopic, predict_isotopes
    mzs, ints, _ = anchor_on_monoisotopic(*predict_isotopes(ION, -1))
    rel = ints / ints[0]
    rows = [_pk(f"L{i}", m, 2e4 * r) for i, (m, r) in enumerate(zip(mzs, rel)) if r >= 0.01]
    mono = float(mzs[0])
    if lobes:
        rows += [_pk("lobe-", mono - 2.35 * fw(mono), 200.0, w=0.45 * W0),
                 _pk("lobe+", mono + 2.7 * fw(mono), 200.0, w=0.45 * W0)]
    rows += [_pk(f"B{k}", 113.07 + 31.1 * k, 1e4) for k in range(9)]     # calibrate the widths
    rows += [_pk(f"f{k}", 61.3 + 6.93 * k, 2.0 + (k % 4)) for k in range(40)]
    t = pd.DataFrame(rows)
    t["signal_to_noise"] = t["height"] / 2.0
    return t, mono


def _run(table, scoring="orbi", cfg=None, **kw):
    from peaky.io import io_mascope as IO
    IO.unregister_offline_sample(SID)
    try:
        return A.run(SID, context="ambient-air", cfg=cfg or PassConfig(), peaks=table, use_cache=False,
                     scoring=scoring, adducts=["[M-H]-"], reagent_n_relabel=False,
                     resolving_power=ORB, log=quiet, **kw)
    finally:
        IO.unregister_offline_sample(SID)


def test_assign_run_marks_the_lobes_before_pass_0_and_keeps_the_ion():
    table, mono = _run_table()
    res = _run(table)
    led, st = res["ledger"], res["stats"]
    assert st["sidelobe_guard"]["skipped"] is None and st["sidelobe_guard"]["flagged"] == 2
    lobes = led[led["peak_id"].isin(["lobe-", "lobe+"])]
    assert (lobes["role"] == L.ROLE_ARTIFACT).all() and lobes["locked"].all()
    assert lobes["neutral_formula"].isna().all()
    m0 = led[(led["role"] == L.ROLE_M0) & (led["peak_id"] == "L0")]
    assert len(m0) == 1 and m0["neutral_formula"].iloc[0] == "C10H16O6"
    assert (led.loc[led["parent_peak_id"] == "L0", "role"] == L.ROLE_ISO).any()     # its 13C child
    assert st["by_role"].get(L.ROLE_ARTIFACT, 0) >= 2                    # counted beside unexplained
    assert L.validate(led) == []


def test_assign_run_off_an_orbitrap_leaves_the_lobes_to_the_passes():
    table, _ = _run_table()
    res = _run(table, scoring="tof")
    assert res["stats"]["sidelobe_guard"]["skipped"].startswith("not an Orbitrap")
    led = res["ledger"]
    assert not led.loc[led["peak_id"].isin(["lobe-", "lobe+"]), "commentary"].astype(str) \
        .str.contains("Orbitrap side lobe").any()
    res = _run(table, cfg=PassConfig(sidelobe_guard=False))
    assert res["stats"]["sidelobe_guard"]["skipped"].startswith("off")


def test_a_c42_rerun_keeps_the_guards_artifacts(monkeypatch):
    """The guard runs before the re-run snapshot: a re-run from pass 0 starts
    from the guarded ledger, and the guard itself runs once."""
    calls = {"guard": 0, "step": 0}
    real_guard = SG.flag_orbitrap_sidelobes

    def counted(*a, **k):
        calls["guard"] += 1
        return real_guard(*a, **k)

    def one_rerun(*a, **k):
        calls["step"] += 1
        return "rerun" if calls["step"] == 1 else "done"

    real_cal = A.passes.calibrate

    def with_a_trend(led, cfg, **k):                     # a fitted trend for the re-run to log
        out = real_cal(led, cfg, **k)
        if cfg.cal_b is None:
            cfg.cal_a, cfg.cal_b, cfg.cal_sigma_trend, cfg.cal_trend_n = 0.0, 0.0, 0.3, 10
            cfg.cal_mz_lo, cfg.cal_mz_hi = 100.0, 400.0
        return out

    monkeypatch.setattr(A.passes, "calibrate", with_a_trend)
    monkeypatch.setattr(A.sidelobe_guard, "flag_orbitrap_sidelobes", counted)
    monkeypatch.setattr(A, "_trend_step", one_rerun)
    monkeypatch.setattr(A.io_mascope, "set_scoring_trend", lambda *a, **k: None)
    table, _ = _run_table()
    res = _run(table)
    assert calls == {"guard": 1, "step": 2}
    led = res["ledger"]
    lobes = led[led["peak_id"].isin(["lobe-", "lobe+"])]
    assert (lobes["role"] == L.ROLE_ARTIFACT).all() and lobes["locked"].all()


# --------------------------------------------------------------------------- CLI
def test_the_cli_switch():
    from peaky import cli
    ap = cli.build_parser()
    a = ap.parse_args(["assign", "--sample-id", "s", "--no-sidelobe-guard"])
    assert a.no_sidelobe_guard
    b = ap.parse_args(["batch", "--batch", "b", "--no-sidelobe-guard"])
    assert cli._sidelobe_cfg(b)["cfg"].sidelobe_guard is False
    assert cli._sidelobe_cfg(ap.parse_args(["batch", "--batch", "b"])) == {}
    assert PassConfig().sidelobe_guard is True
    assert "sidelobe_guard" not in PassConfig.RUNTIME_FIELDS          # a knob: fingerprinted
