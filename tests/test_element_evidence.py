"""The element-evidence gate (the `element_evidence` stage).

A formula carrying S / Cl / Br / Si / P / I always came from a widened search
(the per-peak grid proposes C / H / N / O only), and none of those searches asks
for the element's own line. The pieces tested here:

  * satellites.element_evidence -- the exact-offset predicate, SELF-CALIBRATED per
    file on its own 13C / 18O lines (an absence counts only where the file's own
    detection says the line would be there; 'low' is the file's own ratio
    percentile; under 30 calibration lines nothing is contradicted);
  * plausibility.gate_element_evidence -- the stage that CLEARS a contradicted,
    non-curated commit (and an off-budget P / I from a widened proposer);
  * pass 7 -- the peak list, not a scorer label, confirms the envelope on an
    Orbitrap-class run; a two-channel reagent-acid pair commits no off-budget P / I
    and no unconfirmed S / Cl / Br / Si; a gated winner never displaces;
  * the TOF path: Br / Cl by the ion's own M+2 line, only with the run's width model;
  * the siloxane families' channels, the Orbitrap-class pair / twin windows, and the
    merge-time removal of a reading whose element-signature line the batch refutes.

Every spectrum is synthetic: a flat background at the floor height, committed CHON
calibration rows with their 13C / 18O lines at a chosen ratio response, and the rows
under test at their ions' exact m/z.

Run: pytest tests/test_element_evidence.py -q
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from peaky import ledger as L
from peaky import passes as P
from peaky.assignment import plausibility as PL
from peaky.assignment import residual as RD
from peaky.assignment import satellites as SAT
from peaky.batch import iso_checks as IC
from peaky.chem import chemistry as CH
from peaky.chem import contexts as XC
from peaky.chem import isotopes as ISO

AIR = XC.get_context("ambient-air")
WATER = XC.get_context("water")              # negative mode, budgets P (max_P 1)
URO = XC.get_context("uronium")
FLOOR = 5.0
D13C = 1.0033548
NOLOG = dict(log=lambda *a: None)


def _cfg(klass="orbitrap", **kw):
    c = P.PassConfig(height_cutoff_cps=10.0)
    c.instrument_class = klass
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def _msn100(x, i):
    """A 100-microscan-like response: a line is seen from 1.0-2.5x the floor up (each line its own
    edge, so detection rises over that range) and read at 0.85-1.0x its prediction."""
    return (0.85 + 0.15 * ((i * 7) % 10) / 10.0) if x >= 1.0 + 1.5 * ((i * 37) % 10) / 10.0 else None


def _msn1(x, i):
    """A 1-microscan-like response: a line is seen only from 4-8x the floor up, and read at
    0.25-0.40x its first-order prediction."""
    return (0.25 + 0.15 * ((i * 3) % 10) / 10.0) if x >= 4.0 + 4.0 * ((i * 37) % 10) / 10.0 else None


def _file(lines=(), *, n_cal=60, response=_msn100, lo=100.0, hi=420.0):
    """A ledger: background at FLOOR every 0.2 Da, `n_cal` committed CHON calibration rows
    (C10H15O5-, heights 60 .. 20000) with their 13C / 18O lines at `response(x, i)` x their
    prediction (None: missing), and `lines` -- (peak_id, mz, height) -- as given. Nothing
    is placed within 0.08 Da of a given line."""
    reserved = np.array([m for _, m, _ in lines] or [-1e9])
    free = lambda m: np.min(np.abs(reserved - m)) > 0.08       # noqa: E731
    rows = list(lines)
    cal = []
    hs = np.geomspace(60.0, 20000.0, max(n_cal, 1))
    step = (hi - lo) / max(n_cal, 1)
    for i in range(n_cal):
        m0 = lo + i * step + 0.0321
        while not (free(m0) and free(m0 + D13C) and free(m0 + ISO.D_18O)):
            m0 += 0.11
        h0 = float(hs[i])
        rows.append((f"c{i}", m0, h0))
        kids = []
        for shift, per, lab in ((D13C, 10 * ISO.R_13C_PER_C, "13C"), (ISO.D_18O, 5 * ISO.R_18O_PER_O, "18O")):
            pred = h0 * per
            r = response(pred / FLOOR, i)
            if r is not None:
                rows.append((f"c{i}:{lab}", m0 + shift, pred * r))
                kids.append((f"c{i}:{lab}", lab))
        cal.append((f"c{i}", kids))
    for m in np.arange(lo - 25.0, hi + 25.0, 0.2) + 0.137:
        if free(m):
            rows.append((f"bg{m:.3f}", float(m), FLOOR))
    df = pd.DataFrame(rows, columns=["peak_id", "mz", "height"]).sort_values("mz").reset_index(drop=True)
    led = L.new_ledger(df)
    for pid, kids in cal:
        L.commit_assignment(led, pid, neutral_formula="C10H16O5", adduct="[M-H]-", ion_formula="C10H15O5-",
                            ion_score=0.9, ppm_error=0.1, pass_no=1, method="cheminfo+grid", confidence="Good",
                            commentary="calibration row")
        for k, lab in kids:
            L.attach_isotopologue(led, k, pid, iso_label=lab)
    return led


def _commit(led, pid, neutral, adduct, *, method="ladder:gapfill", conf="Good"):
    ion = CH.format_formula(_ion(neutral, adduct)) + "-"
    L.commit_assignment(led, pid, neutral_formula=neutral, adduct=adduct, ion_formula=ion, ion_score=0.8,
                        ppm_error=0.2, pass_no=6, method=method, confidence=conf, commentary="under test")


def _ion(neutral, adduct):
    from peaky.assignment.tiers import _ion_counts
    return _ion_counts(neutral, adduct)


def _ctx(led, klass="orbitrap"):
    return SAT.evidence_context(led, klass=klass)


def _role(led, pid):
    return L.role_of(led, pid)


# --------------------------------------------------------------------------- the predicate
BR_N, BR_A = "C10H13BrO8", "[M-H]-"              # a Br1 [M-H]- reading, as a widened search proposes it
BR_MZ = CH.ion_mz(BR_N, BR_A)


def test_the_file_calibrates_its_own_observable_level():
    led = _file()
    cal = SAT.evidence_context(led, klass="orbitrap").cal
    assert cal.calibrated and 2.0 <= cal.x_obs <= 4.0 and cal.n >= 100, (cal.x_obs, cal.n)
    assert 0.8 <= cal.low_ratio(20.0) <= 0.9
    led1 = _file(response=_msn1)
    cal1 = SAT.evidence_context(led1, klass="orbitrap").cal
    assert cal1.calibrated and 7.0 <= cal1.x_obs <= 14.0, cal1.x_obs
    assert 0.2 <= cal1.low_ratio(20.0) <= 0.3
    # every line found, or a detection that never rises: no curve, no contradiction
    flat = SAT.ElementCalibration([1.0] * 40 + [5.0] * 40, [False] * 40 + [True] * 20 + [False] * 20,
                                  [np.nan] * 80, 1.0)
    assert not flat.calibrated


def test_a_br_line_absent_where_observable_contradicts_and_a_present_one_confirms():
    h0 = 200.0                                   # 81Br predicted 195 cps: 39x the floor
    pred = h0 * ISO.R_81BR_PER_BR
    for line, verdict in ((pred * 0.95, SAT.EE_CONFIRMED), (None, SAT.EE_CONTRADICTED)):
        lines = [("br", BR_MZ, h0)] + ([("br81", BR_MZ + ISO.D_81BR, line)] if line else [])
        led = _file(lines)
        v = SAT.element_evidence(_ctx(led), BR_MZ, h0, {"Br": 1}, _ion(BR_N, BR_A), "Br")
        assert v["verdict"] == verdict, v


def test_a_side_lobe_on_the_81br_position_confirms_nothing_and_refutes_nothing():
    """A peak the side-lobe guard marked (sidelobe_guard.lobe_mask) is not a line of the
    profile: on the predicted 81Br position it no longer confirms the bromine, and since
    the blend guard still reads it the position is untestable -- never an absence."""
    from peaky.assignment import sidelobe_guard as SG
    h0 = 200.0
    pred = h0 * ISO.R_81BR_PER_BR
    led = _file([("br", BR_MZ, h0), ("br81", BR_MZ + ISO.D_81BR, pred * 0.95)])
    v = SAT.element_evidence(_ctx(led), BR_MZ, h0, {"Br": 1}, _ion(BR_N, BR_A), "Br")
    assert v["verdict"] == SAT.EE_CONFIRMED, v
    L.mark_artifact(led, "br81", f"{SG.LOBE_MARK} of m/z {BR_MZ + ISO.D_81BR + 0.003:.4f} (90x brighter)")
    assert SG.lobe_mask(led).sum() == 1
    v = SAT.element_evidence(_ctx(led), BR_MZ, h0, {"Br": 1}, _ion(BR_N, BR_A), "Br")
    assert v["verdict"] == SAT.EE_UNOBSERVABLE and v["lines"][0]["status"] == "blended", v

def test_a_ringing_artifact_on_the_81br_position_is_read_as_req_reads_it():
    """Cleanup's ringing artifact carries the same role as a side lobe, and the stamped series
    REQ reads cannot tell the two apart: on an Orbitrap the predicate reads it the same way
    (no line, the position untestable). The TOF M+2 test reads every peak, as REQ's TOF branch
    does -- its blend guards read the same list."""
    from peaky.assignment import sidelobe_guard as SG
    from peaky.chem.resolution import Resolution
    h0 = 200.0
    pred = h0 * ISO.R_81BR_PER_BR
    led = _file([("br", BR_MZ, h0), ("br81", BR_MZ + ISO.D_81BR, pred * 0.95)])
    L.mark_artifact(led, "br81", "FT ringing/sidelobe of a saturating parent")
    assert SG.lobe_mask(led).sum() == 0
    v = SAT.element_evidence(_ctx(led), BR_MZ, h0, {"Br": 1}, _ion(BR_N, BR_A), "Br")
    assert v["verdict"] == SAT.EE_UNOBSERVABLE and v["lines"][0]["status"] == "blended", v
    tof = SAT.evidence_context(led, klass="tof", tof_floor=1.0, tof_fwhm=Resolution.from_r(5000.0).fwhm)
    assert SAT.element_evidence(tof, BR_MZ, h0, {"Br": 1}, _ion(BR_N, BR_A), "Br")["verdict"] == SAT.EE_CONFIRMED


def test_a_neighbour_6p7_ppm_off_the_81br_position_is_not_credited():
    """The flagship case: a real neighbour line -6.7 ppm from the predicted 81Br position, at the
    predicted height. A 1 ppm window does not credit it, and as tall as the prediction alone it
    cannot be a blend holding the line (its centroid would sit on the position): contradicted."""
    h0 = 40.0
    pred = h0 * ISO.R_81BR_PER_BR
    nb = (BR_MZ + ISO.D_81BR) * (1 - 6.7e-6)
    led = _file([("br", BR_MZ, h0), ("nb", nb, pred)])
    v = SAT.element_evidence(_ctx(led), BR_MZ, h0, {"Br": 1}, _ion(BR_N, BR_A), "Br")
    assert v["verdict"] == SAT.EE_CONTRADICTED, v
    assert v["lines"][0]["status"] == "absent"


def test_a_dim_br_line_is_unobservable():
    h0 = 6.0                                     # 81Br predicted 5.8 cps: 1.2x the floor
    led = _file([("br", BR_MZ, h0)])
    v = SAT.element_evidence(_ctx(led), BR_MZ, h0, {"Br": 1}, _ion(BR_N, BR_A), "Br")
    assert v["verdict"] == SAT.EE_UNOBSERVABLE and not v["lines"][0]["observable"], v


def test_an_intensity_dependent_ratio_response_is_not_a_contradiction():
    """A 1-microscan-like file reads its real 13C lines at 0.25-0.40x their first-order
    prediction and misses every one under 4-8x the floor. A real S2 ion's 34S line there at 0.3x
    its prediction is CONFIRMED (the file's own low ratio, ~0.25), and one at 5x the floor that
    is missing is UNOBSERVABLE -- under fixed constants (0.5x low edge, 3x observable) the first
    would read low and the second absent, both contradictions."""
    n, a = "C8H14O6S2", "[M-H]-"
    mz = CH.ion_mz(n, a)
    ion = _ion(n, a)
    h_bright = 1200.0                              # 34S predicted 106 cps: 21x the floor
    led = _file([("s", mz, h_bright), ("s34", mz + ISO.D_34S, 0.3 * 2 * ISO.R_34S_PER_S * h_bright)],
                response=_msn1)
    v = SAT.element_evidence(_ctx(led), mz, h_bright, {"S": 2}, ion, "S")
    assert v["verdict"] == SAT.EE_CONFIRMED, v
    h_dim = 5.0 * FLOOR / (2 * ISO.R_34S_PER_S)    # 34S predicted at 5x the floor
    led = _file([("s", mz, h_dim)], response=_msn1)
    v = SAT.element_evidence(_ctx(led), mz, h_dim, {"S": 2}, ion, "S")
    assert v["verdict"] == SAT.EE_UNOBSERVABLE, v
    # the same two lines in a 100-microscan-like file: there the file says they would show
    led = _file([("s", mz, h_dim)])
    assert SAT.element_evidence(_ctx(led), mz, h_dim, {"S": 2}, ion, "S")["verdict"] == SAT.EE_CONTRADICTED


def test_too_few_calibration_lines_never_contradict():
    h0 = 200.0
    led = _file([("br", BR_MZ, h0)], n_cal=10)
    ctx = _ctx(led)
    assert not ctx.cal.calibrated
    v = SAT.element_evidence(ctx, BR_MZ, h0, {"Br": 1}, _ion(BR_N, BR_A), "Br")
    assert v["verdict"] == SAT.EE_UNOBSERVABLE, v
    # a line that is there still confirms (the default band)
    led = _file([("br", BR_MZ, h0), ("br81", BR_MZ + ISO.D_81BR, 0.9 * h0 * ISO.R_81BR_PER_BR)], n_cal=10)
    assert SAT.element_evidence(_ctx(led), BR_MZ, h0, {"Br": 1}, _ion(BR_N, BR_A), "Br")["verdict"] \
        == SAT.EE_CONFIRMED


def test_a_peak_2_ppm_off_the_34s_position_confirms_nothing():
    n, a = "C9H12O5", "[M-H]-"                     # a CHON ion asked as if it carried S
    mz = CH.ion_mz(n, a)
    h0 = 1000.0
    tgt = mz + ISO.D_34S
    led = _file([("x", mz, h0), ("near", tgt * (1 + 2e-6), 0.0443 * h0)])
    v = SAT.element_evidence(_ctx(led), mz, h0, {"S": 1}, {"C": 9, "H": 11, "O": 5, "S": 1}, "S")
    assert v["verdict"] != SAT.EE_CONFIRMED, v


def test_a_reagent_halogen_confirms_only_when_the_full_ion_count_fits():
    """A Br1 neutral on [M+Br]- (ion Br2): an M+2 at 1.95x is both bromines (confirmed); at 0.97x
    it is the reagent's bromine alone -- under the file's low ratio for the full count, and the
    file would show the full line: contradicted."""
    n, a = "C5H9BrO3", "[M+Br]-"
    mz = CH.ion_mz(n, a)
    h0 = 300.0
    for r, verdict in ((1.95, SAT.EE_CONFIRMED), (0.97, SAT.EE_CONTRADICTED)):
        led = _file([("b", mz, h0), ("b81", mz + ISO.D_81BR, r * h0)])
        v = SAT.element_evidence(_ctx(led), mz, h0, {"Br": 1}, _ion(n, a), "Br")
        assert v["verdict"] == verdict, (r, v)


def test_off_an_orbitrap_no_s_or_si_verdicts():
    n, a = "C8H14O6S2", "[M-H]-"
    mz = CH.ion_mz(n, a)
    led = _file([("s", mz, 1200.0)])
    for klass in ("tof", None):
        v = SAT.element_evidence(_ctx(led, klass), mz, 1200.0, {"S": 2}, _ion(n, a), "S")
        assert v["verdict"] == SAT.EE_UNOBSERVABLE, (klass, v)


# --------------------------------------------------------------------------- the stage
def _gate(led, profile=AIR, cfg=None, curated=frozenset()):
    audit = []
    s = PL.gate_element_evidence(led, profile=profile, curated=curated, cfg=cfg or _cfg(), audit=audit, **NOLOG)
    return s, audit


def test_the_stage_clears_a_contradicted_br_row_and_releases_its_children():
    h0 = 40.0
    nb = (BR_MZ + ISO.D_81BR) * (1 - 6.7e-6)
    led = _file([("br", BR_MZ, h0), ("nb", nb, h0 * ISO.R_81BR_PER_BR)])
    L.commit_assignment(led, "br", neutral_formula=BR_N, adduct=BR_A, ion_formula="C10H12BrO8-", ion_score=0.8,
                        ppm_error=0.1, pass_no=4, method="residual:iso-pair", confidence="Good (iso-pair)",
                        commentary="an iso-pair")
    L.attach_isotopologue(led, "nb", "br", iso_label="81Br(pair)")
    s, audit = _gate(led)
    assert s["cleared"] == 1 and s["cleared_isotope"] == 1, s
    assert _role(led, "br") == L.ROLE_UNEXPLAINED and _role(led, "nb") == L.ROLE_UNEXPLAINED
    assert pd.isna(led.loc[led.peak_id == "nb", "parent_peak_id"].iloc[0])
    assert len(audit) == 1 and audit[0]["reason"].startswith("element_evidence: Br1 contradicted")
    # every calibration row stays
    assert (led.loc[led.peak_id.str.match(r"c\d+$"), "role"] == L.ROLE_M0).all()


def test_a_cleared_peak_is_locked_against_new_readings_but_not_against_being_a_line():
    """A peak the stage cleared stays unexplained for the rest of the run: it is locked,
    so no later stage commits a new reading on it (an O-rich re-grid, a reference-list
    rescue). The final envelope sweep may still claim it as a committed M0's 13C line:
    that explains the line without proposing a formula."""
    h0 = 200.0
    rel = 10 * ISO.R_13C_PER_C                                    # C10H15O5-: its 13C line
    pp_mz = BR_MZ - ISO.D_13C
    led = _file([("br", BR_MZ, h0), ("pp", pp_mz, h0 / rel)])
    _commit(led, "br", BR_N, BR_A)
    s, _ = _gate(led)
    assert s["cleared"] == 1 and _role(led, "br") == L.ROLE_UNEXPLAINED
    assert L.is_locked(led, "br") and PL.cleared_by_element_evidence(led, led.index[led.peak_id == "br"][0])
    with pytest.raises(L.LedgerError):
        _commit(led, "br", "C9H10O13", BR_A, method="cheminfo+grid")
    # the envelope sweep: a committed C10 parent one 13C spacing below claims it as its line
    L.commit_assignment(led, "pp", neutral_formula="C10H16O5", adduct="[M-H]-", ion_formula="C10H15O5-",
                        ion_score=0.95, ppm_error=0.1, pass_no=1, method="cheminfo+grid", confidence="Good",
                        commentary="a parent committed late")
    P.complete_isotope_envelopes(led, _cfg(), **NOLOG)
    assert _role(led, "br") == L.ROLE_ISO and led.loc[led.peak_id == "br", "parent_peak_id"].iloc[0] == "pp"


def test_the_stage_keeps_a_dim_br_row_a_curated_one_a_known_one_and_a_locked_one():
    led = _file([("dim", BR_MZ, 6.0)])
    _commit(led, "dim", BR_N, BR_A)
    assert _gate(led)[0]["cleared"] == 0 and _role(led, "dim") == L.ROLE_M0
    for how in ("curated", "known", "locked"):
        led = _file([("br", BR_MZ, 200.0)])
        _commit(led, "br", BR_N, BR_A, method="known:atmospheric" if how == "known" else "ladder:gapfill")
        if how == "locked":
            led.loc[led.peak_id == "br", "locked"] = True
        s, _ = _gate(led, curated=frozenset({BR_N}) if how == "curated" else frozenset())
        assert s["cleared"] == 0 and _role(led, "br") == L.ROLE_M0, how
    led = _file([("br", BR_MZ, 200.0)])
    _commit(led, "br", BR_N, BR_A)
    assert _gate(led)[0]["cleared"] == 1


def test_the_stage_clears_a_reagent_halogen_reading_the_line_contradicts():
    n, a = "C5H9BrO3", "[M+Br]-"
    mz = CH.ion_mz(n, a)
    for r, cleared in ((1.95, 0), (0.97, 1)):
        led = _file([("b", mz, 300.0), ("b81", mz + ISO.D_81BR, r * 300.0)])
        _commit(led, "b", n, a, method="contaminant:halogen_dbp")
        assert _gate(led)[0]["cleared"] == cleared, r


def test_si_needs_its_line_unless_a_siloxane_step_pins_it():
    """An Si1 [M+NO3]- whose 29Si and 30Si lines the file would show is cleared; its 30Si line
    alone keeps it; a committed neighbour one C2H6OSi away on the same adduct pins the silicon
    (the CF2 rule's analogue) and keeps both rungs whatever their lines."""
    n, a = "C6H14O3Si", "[M+NO3]-"
    mz = CH.ion_mz(n, a)
    h0 = 2000.0
    led = _file([("si", mz, h0)])
    _commit(led, "si", n, a, method="contaminant:siloxane")
    assert _gate(led)[0]["cleared"] == 1
    led = _file([("si", mz, h0), ("si30", mz + ISO.D_30SI, ISO.R_30SI_PER_SI * h0)])
    _commit(led, "si", n, a, method="contaminant:siloxane")
    assert _gate(led)[0]["cleared"] == 0
    n2 = "C8H20O4Si2"
    mz2 = CH.ion_mz(n2, a)
    led = _file([("si", mz, h0), ("si2", mz2, h0)])
    _commit(led, "si", n, a, method="contaminant:siloxane")
    _commit(led, "si2", n2, a, method="contaminant:siloxane")
    s, _ = _gate(led)
    assert s["cleared"] == 0 and s["si_ladder_kept"] == 2, s


def test_s_counts_and_the_s1_allowance():
    """S1 within the profile's cap is not tested here (its own rules stand); S2 / S3 are."""
    for n, cleared in (("C9H10O4S", 0), ("C8H14O6S2", 1), ("C8H15PS3", 1)):
        a = "[M-H]-"
        mz = CH.ion_mz(n, a)
        led = _file([("s", mz, 2000.0)])
        _commit(led, "s", n, a, method="certified:multi-channel" if "P" in n else "ladder:gapfill")
        s, audit = _gate(led)
        assert s["cleared"] == cleared, (n, s)
        if "P" in n:          # a pass-7 P row is pass 7's to judge: the S3 line clears it here
            assert "S3 contradicted" in audit[0]["reason"] and "P1 outside" not in audit[0]["reason"]


def test_monoisotopic_p_from_a_widened_proposer_where_the_profile_budgets_it_at_zero():
    n, a = "C3H9O3P", "[M-H]-"
    mz = CH.ion_mz(n, a)
    for profile, method, curated, cleared in (
            (AIR, "ladder:gapfill", frozenset(), 1),
            (AIR, "contaminant:organosulfate", frozenset(), 1),
            (AIR, "certified:multi-channel", frozenset(), 0),     # pass 7 judges its own (channels)
            (AIR, "ladder:gapfill", frozenset({n}), 0),           # curated
            (AIR, "known:atmospheric", frozenset(), 0),
            (WATER, "ladder:gapfill", frozenset(), 0)):           # P budgeted
        led = _file([("p", mz, 500.0)])
        _commit(led, "p", n, a, method=method)
        s, audit = _gate(led, profile=profile, curated=curated)
        assert s["cleared"] == cleared, (profile.label, method, s)
        if cleared:
            assert s["cleared_mono"] == 1 and "P1 outside the ambient-air budget" in audit[0]["reason"]


def test_fluorine_is_left_alone():
    n, a = "C5H2F6O4", "[M-H]-"
    led = _file([("f", CH.ion_mz(n, a), 500.0)])
    _commit(led, "f", n, a, method="contaminant:fluorinated")
    assert _gate(led)[0]["cleared"] == 0 and _role(led, "f") == L.ROLE_M0


def test_unknown_class_and_the_trace_sample_run_only_the_p_rule():
    # (a TOF with its detection floor but no width model judges no M+2 line either: the TOF path
    # itself is test_on_a_tof_the_stage_judges_br_by_the_ions_own_m2_line)
    for cfg in (_cfg(None), _cfg("orbitrap", trace_sample=True),
                _cfg("tof", instrument_type="tof", noise_edge_cps=FLOOR)):
        led = _file([("br", BR_MZ, 200.0), ("p", CH.ion_mz("C3H9O3P", "[M-H]-"), 500.0)])
        _commit(led, "br", BR_N, BR_A)
        _commit(led, "p", "C3H9O3P", "[M-H]-")
        s, _ = _gate(led, cfg=cfg)
        assert s["cleared"] == 1 and s["cleared_mono"] == 1 and _role(led, "br") == L.ROLE_M0, s


def test_the_stage_sits_after_every_proposer_and_before_degeneracy_and_tiers():
    from peaky.assignment import assign as A
    names = [s.name for s in A._STAGES]
    i = names.index("element_evidence")
    for earlier in ("pass3", "pass4", "pass5", "pass_certified", "pass3_series", "pass6_ladder", "cleanup",
                    "siloxane", "labeled_15n", "rearbitrate"):
        assert names.index(earlier) < i, earlier
    for later in ("resolvability", "degeneracy", "tiers", "plausibility"):
        assert names.index(later) > i, later
    assert not A._STAGES[i].safe


# --------------------------------------------------------------------------- pass 7
def _srow(compound, ion, label, pid, mz, *, base, score=0.9, ppm=0.2, iso=0.9):
    return dict(compound_formula=compound, compound_score=score, compound_category=2, ion_formula=ion,
                ion_score=score, ion_category=2, mechanism_id="m", isotope_formula=ion, iso_label=label,
                is_base=base, theo_mz=mz, rel_abundance=1.0, iso_score=iso, iso_category=2,
                sample_peak_id=pid, sample_peak_mz=mz, sample_peak_intensity=1e4, ppm_error=ppm,
                abundance_error=0.0)


def _scorer(compound, members, kids=()):
    """A fake oracle that knows `compound` only: `members` = [(pid, ion, mz)] anchored base rows,
    `kids` = [(pid, ion, label, mz)] isotope rows (the scorer's labels, whatever the peak list holds)."""
    rows = [_srow(compound, ion, "M0", pid, mz, base=True) for pid, ion, mz in members]
    rows += [_srow(compound, ion, lab, pid, mz, base=False) for pid, ion, lab, mz in kids]

    def score(client, sid, formulas, *, mechanism_ids=None, **kw):
        return pd.DataFrame(rows) if compound in formulas else pd.DataFrame([])
    return score


def _p7(led, score, adducts, *, profile=AIR, cfg=None):
    s = P.run_pass_certified(None, "SID", led, profile, cfg or _cfg(), adducts, score_fn=score, **NOLOG)
    return s, led.set_index("peak_id")


def _weak(led, pid, neutral="C3H4N2O8", adduct="[M+NO3]-", ion="C3H4N3O11-"):
    L.commit_assignment(led, pid, neutral_formula=neutral, adduct=adduct, ion_formula=ion, ion_score=0.6,
                        ppm_error=-2.8, pass_no=1, method="cheminfo+grid", confidence="Low",
                        commentary="a single-channel fit")


MPA = "CH5O3P"
HN = ["[M-H]-", "[M+NO3]-"]


def test_a_two_channel_hno3_pair_commits_no_off_budget_p_and_displaces_nothing():
    """X- and X.HNO3- of a non-curated P winner in a profile that budgets P at 0: the ordinary
    nitrate-cluster pattern, no evidence for the element -- nothing committed, the weak incumbent
    keeps its peak. Curated (a reference list names it), the same pair commits as before."""
    mh, mn = CH.ion_mz(MPA, "[M-H]-"), CH.ion_mz(MPA, "[M+NO3]-")
    members = [("h", "CH4O3P-", mh), ("n", "CH5NO6P-", mn)]
    led = _file([("h", mh, 3000.0), ("n", mn, 3000.0)])
    _weak(led, "n")
    s, by = _p7(led, _scorer(MPA, members), HN)
    assert s["committed"] == 0 and s["gated"] == 1 and s["displaced"] == 0, s
    assert by.loc["n", "neutral_formula"] == "C3H4N2O8" and by.loc["h", "role"] == L.ROLE_UNEXPLAINED
    led = _file([("h", mh, 3000.0), ("n", mn, 3000.0)])
    _weak(led, "n")
    s, by = _p7(led, _scorer(MPA, members), HN, cfg=_cfg(reflist_formulas=frozenset({MPA})))
    assert s["committed"] == 1 and s["gated"] == 0 and by.loc["h", "neutral_formula"] == MPA, s
    assert s["displaced"] == 0 and by.loc["n", "neutral_formula"] == "C3H4N2O8"   # two channels, no line: weak


TAA = "C2H4OS"


def test_on_an_orbitrap_the_peak_list_not_the_scorer_label_confirms_the_envelope():
    """Thioacetic acid's [M-H]- / [M+NO3]- pair, bright enough that the file would show both 34S
    lines. The scorer labels a '34S' kid that the peak list does not hold at the 34S position:
    no line confirms, and on a reagent-acid pair the S winner is not committed. With the lines
    in the peak list it is, Good (certified), its envelope confirmed."""
    mh, mn = CH.ion_mz(TAA, "[M-H]-"), CH.ion_mz(TAA, "[M+NO3]-")
    members = [("h", "C2H3OS-", mh), ("n", "C2H4NO4S-", mn)]
    far = mn + ISO.D_34S + 0.0198                  # where the scorer matched its '34S' line
    kids = [("far", "C2H4NO4S-", "34S", far)]
    led = _file([("h", mh, 4000.0), ("n", mn, 3000.0), ("far", far, 140.0)])
    s, by = _p7(led, _scorer(TAA, members, kids), HN)
    assert s["committed"] == 0 and s["gated"] == 1, s
    lines = [("h", mh, 4000.0), ("n", mn, 3000.0), ("h34", mh + ISO.D_34S, 4000.0 * ISO.R_34S_PER_S),
             ("n34", mn + ISO.D_34S, 3000.0 * ISO.R_34S_PER_S)]
    led = _file(lines)
    kids = [("h34", "C2H3OS-", "34S", mh + ISO.D_34S), ("n34", "C2H4NO4S-", "34S", mn + ISO.D_34S)]
    s, by = _p7(led, _scorer(TAA, members, kids), HN)
    assert s["committed"] == 1, s
    for pid in ("h", "n"):
        assert by.loc[pid, "confidence"] == "Good (certified)"
        assert "diagnostic isotope envelope confirmed" in by.loc[pid, "commentary"]
    # off an Orbitrap the scorer-label rule stands (no exact-offset test there)
    led = _file([("h", mh, 4000.0), ("n", mn, 3000.0), ("far", far, 140.0)])
    s, by = _p7(led, _scorer(TAA, members, [("far", "C2H4NO4S-", "34S", far)]), HN, cfg=_cfg(None))
    assert s["committed"] == 1 and s["gated"] == 0, s


def test_a_contradicted_winner_never_displaces_an_incumbent():
    """[M+NO3]- + [M+Br]- (no reagent-acid pair): the scorer's '34S' label would have made the
    certificate strong, but the file shows neither member's 34S line where it would: the winner is
    gated -- committed on the unexplained member only, Low, the weak incumbent kept."""
    mn, mb = CH.ion_mz(TAA, "[M+NO3]-"), CH.ion_mz(TAA, "[M+Br]-")
    members = [("n", "C2H4NO4S-", mn), ("b", "C2H4BrOS-", mb)]
    far = mn + ISO.D_34S + 0.0198
    led = _file([("n", mn, 3000.0), ("b", mb, 3000.0), ("far", far, 140.0)])
    _weak(led, "n")
    s, by = _p7(led, _scorer(TAA, members, [("far", "C2H4NO4S-", "34S", far)]), HN + ["[M+Br]-"])
    assert s["displaced"] == 0 and s["committed"] == 1, s
    assert by.loc["n", "neutral_formula"] == "C3H4N2O8"
    assert by.loc["b", "neutral_formula"] == TAA and by.loc["b", "confidence"] == "Low (certified)"


# --------------------------------------------------------------------------- the siloxane families' channels
def test_the_siloxane_family_takes_no_anion_cluster_channel(monkeypatch):
    from peaky.assignment.passes import directors as D
    seen = {}

    def fake_enum(client, mzs, mech_ids, ranges, cfg, adducts, **kw):
        seen.setdefault(tuple(sorted(k for k in ranges if ranges[k][1] > 0)), []).append(list(adducts))
        return set()
    monkeypatch.setattr(D, "_enumerate", fake_enum)
    monkeypatch.setattr(D, "_mech_ids_for", lambda client, adducts: [])
    led = L.new_ledger(pd.DataFrame({"peak_id": ["a", "b"], "mz": [250.05, 300.07], "height": [500.0, 400.0]}))
    D.run_pass3(None, "s", led, AIR, None, _cfg(), ["[M-H]-", "[M+NO3]-"], reagent=None, **NOLOG)
    si = [v for k, v in seen.items() if "Si" in k]
    assert si and all("[M+NO3]-" not in ads and "[M-H]-" in ads for ads in si[0]), seen
    nitr = [v for k, v in seen.items() if "Si" not in k and "S" not in k]
    assert any("[M+NO3]-" in ads for v in nitr for ads in v)          # the other families keep the run's
    seen.clear()
    D.run_pass3(None, "s", led, URO, None, _cfg(), ["[M+H]+", "[M+(CH4N2O)H]+", "[M+Na]+", "[M+NH4]+"],
                reagent=None, **NOLOG)
    si = [ads for k, v in seen.items() if "Si" in k for ads in v]
    assert si and all({"[M+(CH4N2O)H]+", "[M+Na]+"} <= set(ads) for ads in si), seen   # the positive channels stay


# --------------------------------------------------------------------------- the Orbitrap-class windows
def test_the_pair_and_twin_windows_narrow_on_a_calibrated_orbitrap():
    """The exact-offset window max(1 ppm, 4 sigma) only once the file is calibrated
    (cfg.cal_sigma set by passes.calibrate); before that, and off an Orbitrap or on
    the trace sample, the wide windows the tests had before."""
    assert RD.iso_pair_ppm(_cfg("orbitrap", cal_sigma=0.1)) == 1.0
    assert RD.iso_pair_ppm(_cfg("orbitrap", cal_sigma=0.5)) == 2.0
    assert RD.iso_pair_ppm(_cfg("orbitrap")) == RD.ISO_PAIR_PPM                    # not calibrated yet
    assert RD.iso_pair_ppm(_cfg("tof", cal_sigma=0.1)) == RD.ISO_PAIR_PPM \
        == RD.iso_pair_ppm(_cfg("orbitrap", trace_sample=True, cal_sigma=0.1))
    assert SAT.twin_ppm(_cfg("orbitrap", cal_sigma=0.1)) == 1.0
    assert SAT.twin_ppm(_cfg("orbitrap")) == SAT.twin_ppm(_cfg(None)) == SAT.TWIN_PPM
    assert SAT.twin_ppm(_cfg("orbitrap", cal_sigma=float("nan"))) == SAT.TWIN_PPM
    light = 338.97204
    heavy = (light + RD.D_PAIR_BR) * (1 - 6.7e-6)          # a real neighbour, not the 81Br line
    led = L.new_ledger(pd.DataFrame({"peak_id": ["l", "x"], "mz": [light, heavy], "height": [40.0, 37.0]}))
    assert len(RD.find_iso_pairs(led, min_height=10.0)) == 1                       # 8 ppm: paired
    assert len(RD.find_iso_pairs(led, min_height=10.0,
                                 ppm_tol=RD.iso_pair_ppm(_cfg("orbitrap", cal_sigma=0.1)))) == 0


def test_pass0_does_not_refute_a_29si_line_before_the_file_is_calibrated():
    """Pass 0 runs before `calibrate`: a known Si5 urea adduct whose 29Si line sits 1.01 ppm
    off its exact offset (0.19x the parent against 0.26 predicted) and whose 30Si line is in
    place reads 'deferred' (the line is seen), not 'refuted: no 29Si line' -- the narrowed
    window would have refuted it and the batch's known-species lock would have counted it.
    Once the file is calibrated (sigma 0.1 ppm: a 1-ppm window) the same offset is an absence."""
    from peaky.assignment.passes import directors as D
    counts = {"C": 11, "H": 35, "N": 2, "O": 6, "Si": 5}            # D5.UrH+
    m0, h0 = 431.133634, 1500.0
    si29 = (m0 + ISO.D_29SI) * (1 + 1.01e-6)
    si30 = m0 + ISO.D_30SI
    led = L.new_ledger(pd.DataFrame({"peak_id": ["p", "s29", "s30"], "mz": [m0, si29, si30],
                                     "height": [h0, 0.19 * h0, 0.13 * h0]}))
    v = D._twin_verdict(led, "p", counts, _cfg("orbitrap"))
    assert v["verdict"] == "deferred" and v["twin"] == "Si", v
    assert "29Si line at 0.19x" in v["why"]
    v = D._twin_verdict(led, "p", counts, _cfg("orbitrap", cal_sigma=0.1))
    assert v["verdict"] == "refuted" and "no 29Si line" in v["why"], v


# --------------------------------------------------------------------------- the merge-time removal
def _table(rows):
    t = pd.DataFrame(rows)
    for c in IC.TABLE_COLUMNS:
        if c not in t.columns:
            t[c] = np.nan
    return t[list(IC.TABLE_COLUMNS)]


def test_a_req_veto_on_a_signature_line_removes_the_merged_row():
    t = _table([
        dict(neutral_formula=BR_N, adduct=BR_A, check="REQ", instrument="orbitrap", veto=True, verdict="absent",
             line="M+2 (81Br)", note="the M+2 (81Br) line absent in 9 of 9 detectable spectra"),
        dict(neutral_formula="C8H14O6S2", adduct="[M-H]-", check="REQ", instrument="orbitrap", veto=True,
             verdict="absent", line="34S", note="34S absent"),
        dict(neutral_formula="C12H20O4", adduct="[M-H]-", check="C", instrument="orbitrap", veto=True,
             verdict="contradict", line=np.nan, note="13C count"),
        dict(neutral_formula="CH4O3S", adduct="[M-H]-", check="REQ", instrument="orbitrap", veto=True,
             verdict="absent", line="34S", note="34S absent"),
        # a TOF's whole-M+2 REQ keeps its own demote (iso_checks.tof_m2_gates), whatever its label
        dict(neutral_formula="C6H9BrO3", adduct="[M+Br]-", check="REQ", instrument="tof", veto=True,
             verdict="absent", line="M+2 (81Br)", note="TOF M+2 absent"),
        # a species the known-species lock decided stays (its tier_reason mark)
        dict(neutral_formula="C2H3BrO2", adduct="[M-H]-", check="REQ", instrument="orbitrap", veto=True,
             verdict="absent", line="M+2 (81Br)", note="81Br absent"),
    ])
    assert set(IC.signature_vetoes(t)) == {(BR_N, BR_A), ("C8H14O6S2", "[M-H]-"), ("CH4O3S", "[M-H]-"),
                                           ("C2H3BrO2", "[M-H]-")}
    merged = pd.DataFrame({"neutral_formula": [BR_N, "C8H14O6S2", "C12H20O4", "CH4O3S", "C6H9BrO3", "C9H14O4",
                                               "C2H3BrO2"],
                           "adduct": [BR_A, "[M-H]-", "[M-H]-", "[M-H]-", "[M+Br]-", "[M-H]-", "[M-H]-"],
                           "mz": [338.97, 269.0, 227.1, 94.98, 286.9, 185.08, 136.92], "tier": "Candidate",
                           "tier_reason": ["", "", "", "", "", "", f"{IC.KNOWN_LOCK_MARK} by 4 files"]})
    out, s = IC.remove_signature_vetoed(merged, t, exempt=frozenset({"CH4O3S"}), **NOLOG)
    assert s["removed"] == 2 and {p["neutral_formula"] for p in s["pairs"]} == {BR_N, "C8H14O6S2"}
    assert list(out["neutral_formula"]) == ["C12H20O4", "CH4O3S", "C6H9BrO3", "C9H14O4", "C2H3BrO2"]


# --------------------------------------------------------------------------- calibration paths
def test_a_detection_curve_needs_ten_lines_at_its_observable_level():
    """The fitted curve reaches EE_DETECT near 4x the floor, but only six lines sit at or above it:
    the file stays uncalibrated (nothing contradicted). Twelve there: calibrated."""
    def cal(n_bright):
        x = list(np.geomspace(0.5, 3.0, 40)) + [20.0] * n_bright
        found = [(i % 4 == 0) if v < 1.0 else ((i % 4 != 0) if v < 2.0 else True)
                 for i, v in enumerate(x[:40])] + [True] * n_bright
        return SAT.ElementCalibration(x, found, [np.nan] * len(x), 1.0)
    few, enough = cal(6), cal(12)
    assert few.fit is not None and not few.calibrated, few.describe()
    assert enough.calibrated and enough.x_obs < 20.0, enough.describe()


def test_low_suspect_and_tied_rows_do_not_calibrate():
    """Only trusted CHON commits calibrate: a Low / Suspect / tied row's missing 13C line is not
    the file's detection."""
    extra = [(f"x{i}", 431.0 + 1.37 * i, 3000.0) for i in range(12)]
    base = SAT.evidence_context(_file(extra), klass="orbitrap").cal.n
    for conf, tied in (("Low", False), ("Suspect", False), ("Good", True)):
        led = _file(extra)
        for pid, _, _ in extra:
            L.commit_assignment(led, pid, neutral_formula="C10H16O5", adduct="[M-H]-", ion_formula="C10H15O5-",
                                ion_score=0.9, ppm_error=0.1, pass_no=1, method="cheminfo+grid", confidence=conf,
                                tied=tied, commentary="not trusted")
        assert SAT.evidence_context(led, klass="orbitrap").cal.n == base, (conf, tied)
    led = _file(extra)
    for pid, _, _ in extra:
        L.commit_assignment(led, pid, neutral_formula="C10H16O5", adduct="[M-H]-", ion_formula="C10H15O5-",
                            ion_score=0.9, ppm_error=0.1, pass_no=1, method="cheminfo+grid", confidence="Good",
                            commentary="trusted")
    assert SAT.evidence_context(led, klass="orbitrap").cal.n > base


def test_the_local_floor_widens_where_the_narrow_window_is_sparse():
    """Three tall lines within +-5 Da are no floor: under EE_FLOOR_MIN_N there, the floor is read
    over +-20 Da."""
    t = 300.0
    mz = np.sort(np.r_[t - 1.0, t + 0.5, t + 1.0, np.linspace(t - 19.0, t - 6.0, 20), np.linspace(t + 6.0, t + 19.0, 20)])
    h = np.where(np.abs(mz - t) <= 5.0, 100.0, 2.0)
    assert SAT.local_floor(mz, h, t) == 2.0
    dense = np.sort(np.r_[mz, t + np.linspace(-4.0, 4.0, 6)])
    hd = np.where(np.abs(dense - t) <= 5.0, 100.0, 2.0)
    assert SAT.local_floor(dense, hd, t) == 100.0          # enough lines nearby: the narrow window


def test_a_line_far_above_its_prediction_is_another_ions_and_confirms_nothing():
    h0 = 200.0
    pred = h0 * ISO.R_81BR_PER_BR
    led = _file([("br", BR_MZ, h0), ("big", BR_MZ + ISO.D_81BR, 5.0 * pred)])
    v = SAT.element_evidence(_ctx(led), BR_MZ, h0, {"Br": 1}, _ion(BR_N, BR_A), "Br")
    assert v["verdict"] == SAT.EE_UNOBSERVABLE and v["lines"][0]["status"] == "high", v


def test_a_taller_neighbour_inside_one_fwhm_could_hold_the_line():
    """A neighbour 6.7 ppm off the 81Br position, five times the predicted line's height: a real
    line merged into it would be picked at their centroid, so the absence is no evidence --
    unobservable, and the stage keeps the row (as tall as the prediction alone, it is no blend:
    test_a_neighbour_6p7_ppm_off_the_81br_position_is_not_credited)."""
    h0 = 200.0
    pred = h0 * ISO.R_81BR_PER_BR
    nb = (BR_MZ + ISO.D_81BR) * (1 - 6.7e-6)
    led = _file([("br", BR_MZ, h0), ("nb", nb, 5.0 * pred)])
    v = SAT.element_evidence(_ctx(led), BR_MZ, h0, {"Br": 1}, _ion(BR_N, BR_A), "Br")
    assert v["verdict"] == SAT.EE_UNOBSERVABLE and v["lines"][0]["status"] == "blended", v
    _commit(led, "br", BR_N, BR_A)
    assert _gate(led)[0]["cleared"] == 0 and _role(led, "br") == L.ROLE_M0


def test_the_stage_reads_a_composite_parent_by_its_own_share():
    """A Br row owning 3 % of a composite peak: its 81Br line is predicted from that share (1.2x the
    floor, unobservable) -- not from the whole peak, where it would be contradicted."""
    led = _file([("br", BR_MZ, 200.0)])
    _commit(led, "br", BR_N, BR_A)
    led.loc[led.peak_id == "br", "assigned_fraction"] = 0.03
    s, _ = _gate(led)
    assert s["cleared"] == 0 and s["unobservable"] == 1 and _role(led, "br") == L.ROLE_M0, s


def test_a_rearbitrated_row_keeps_its_proposers_exemptions():
    """The re-arbitration stage writes 'rearb<-<method>': a re-read known: row stays exempt, a
    re-read pass-7 P row is still pass 7's to judge, a re-read gap-fill P row is still a widened
    proposal."""
    led = _file([("br", BR_MZ, 200.0)])
    _commit(led, "br", BR_N, BR_A, method="rearb<-known:atmospheric")
    assert _gate(led)[0]["cleared"] == 0 and _role(led, "br") == L.ROLE_M0
    n, a = "C3H9O3P", "[M-H]-"
    for method, cleared in (("rearb<-certified:multi-channel", 0), ("rearb<-known:atmospheric", 0),
                            ("rearb<-ladder:gapfill", 1), ("rearb<-rearb<-ladder:gapfill", 1)):
        led = _file([("p", CH.ion_mz(n, a), 500.0)])
        _commit(led, "p", n, a, method=method)
        assert _gate(led)[0]["cleared"] == cleared, method
    assert SAT.base_method("rearb<-rearb<-known:x") == "known:x" and SAT.base_method(None) == ""


# --------------------------------------------------------------------------- the TOF path of the stage
def _tof_cfg(**kw):
    return _cfg("tof", instrument_type="tof", noise_edge_cps=FLOOR, **kw)    # TOF floor 3 x 5 = 15 cps


def test_on_a_tof_the_stage_judges_br_by_the_ions_own_m2_line():
    """TOF-class (instrument_type 'tof': the detection floor is set) with the run's width model:
    an absent M+2 line clears a non-curated Br row, a seen one keeps it, a neighbour 30 mDa off
    the line (inside one TOF FWHM) at its height keeps it (blended); without a width model nothing
    is judged (tiers.tof_m2_verdicts declines too), however absent the line."""
    from peaky.assignment import tiers as T
    from peaky.chem.resolution import Resolution
    rp = Resolution.from_r(5000.0)
    n, a = "C6H9BrO3", "[M-H]-"
    mz = CH.ion_mz(n, a)
    h0 = 400.0
    pred = h0 * ISO.R_81BR_PER_BR
    cfg = _tof_cfg()
    assert T.tof_assign_floor(cfg) == 15.0
    cases = (([], rp, 1, SAT.EE_CONTRADICTED),
             ([("m2", mz + ISO.D_81BR, 0.95 * pred)], rp, 0, SAT.EE_CONFIRMED),
             ([("nb", mz + ISO.D_81BR + 0.030, 1.2 * pred)], rp, 0, SAT.EE_UNOBSERVABLE),
             ([], None, 0, SAT.EE_UNOBSERVABLE))
    for extra, res, cleared, verdict in cases:
        led = _file([("br", mz, h0)] + extra)
        _commit(led, "br", n, a, method="residual:iso-pair")
        audit = []
        s = PL.gate_element_evidence(led, profile=AIR, cfg=cfg, resolution=res, audit=audit, **NOLOG)
        assert s["cleared"] == cleared and s[verdict] == 1, (extra, res, s)
        assert _role(led, "br") == (L.ROLE_UNEXPLAINED if cleared else L.ROLE_M0)
        if cleared:
            assert "Br1 contradicted" in audit[0]["reason"] and "(TOF)" in audit[0]["reason"]
        tv = T.tof_m2_verdicts(_file([("br", mz, h0)] + extra), cfg=cfg, resolving_power=res)
        assert (tv is None) == (res is None)
    # a curated Br row and S rows are never judged on a TOF
    led = _file([("br", mz, h0)])
    _commit(led, "br", n, a, method="residual:iso-pair")
    s = PL.gate_element_evidence(led, profile=AIR, cfg=cfg, resolution=rp, curated=frozenset({n}), **NOLOG)
    assert s["cleared"] == 0


def test_a_low_line_inside_the_window_is_reported_low_not_absent():
    """A picked line inside the search window under the fraction a sighting needs contradicts
    the element, but the reason says it is LOW with its share of the prediction -- not
    'absent within N ppm' -- on a TOF and on an Orbitrap; a truly empty window still reads absent."""
    from peaky.chem.resolution import Resolution
    rp = Resolution.from_r(5000.0)
    n, a = "C6H9BrO3", "[M-H]-"
    mz = CH.ion_mz(n, a)
    h0 = 400.0
    pred = h0 * ISO.R_81BR_PER_BR
    for extra, want in (([("m2", mz + ISO.D_81BR, 0.3 * pred)], "low (0.3"), ([], "absent within")):
        led = _file([("br", mz, h0)] + extra)
        _commit(led, "br", n, a, method="residual:iso-pair")
        audit = []
        s = PL.gate_element_evidence(led, profile=AIR, cfg=_tof_cfg(), resolution=rp, audit=audit, **NOLOG)
        assert s["cleared"] == 1, (extra, s)
        why = audit[0]["reason"]
        assert want in why and "(TOF)" in why, why
        if extra:
            assert "absent" not in why, why
    # Orbitrap: the 81Br line in place at 0.2x its prediction
    led = _file([("br", BR_MZ, 200.0), ("b81", BR_MZ + ISO.D_81BR, 0.2 * 200.0 * ISO.R_81BR_PER_BR)])
    v = SAT.element_evidence(_ctx(led), BR_MZ, 200.0, CH.parse_formula(BR_N), _ion(BR_N, BR_A), "Br")
    assert v["verdict"] == SAT.EE_CONTRADICTED and "line low (0.20 of prediction" in v["why"], v["why"]
    assert "no line within" not in v["why"]


# --------------------------------------------------------------------------- pass 7: the gate on displacement
def test_a_winner_with_its_line_contradicted_on_one_member_never_displaces():
    """Thioacetic acid's [M-H]- / [M+NO3]- pair: the [M-H]- member's 34S line is there (the
    envelope is confirmed on it), the [M+NO3]- member's -- bright enough that the file would show
    it -- is not. Confirmed is not enough: the contradicted member fails the gate, so the winner
    takes the unexplained member only and the weak incumbent keeps its peak."""
    mh, mn = CH.ion_mz(TAA, "[M-H]-"), CH.ion_mz(TAA, "[M+NO3]-")
    members = [("h", "C2H3OS-", mh), ("n", "C2H4NO4S-", mn)]
    led = _file([("h", mh, 4000.0), ("n", mn, 3000.0), ("h34", mh + ISO.D_34S, 4000.0 * ISO.R_34S_PER_S)])
    _weak(led, "n")
    s, by = _p7(led, _scorer(TAA, members), HN)
    assert s["committed"] == 1 and s["gated"] == 0 and s["displaced"] == 0, s
    assert by.loc["h", "neutral_formula"] == TAA and by.loc["n", "neutral_formula"] == "C3H4N2O8"
    # both lines there: the same certificate displaces the weak incumbent
    led = _file([("h", mh, 4000.0), ("n", mn, 3000.0), ("h34", mh + ISO.D_34S, 4000.0 * ISO.R_34S_PER_S),
                 ("n34", mn + ISO.D_34S, 3000.0 * ISO.R_34S_PER_S)])
    _weak(led, "n")
    s, by = _p7(led, _scorer(TAA, members), HN)
    assert s["displaced"] == 1 and by.loc["n", "neutral_formula"] == TAA, s


def test_three_channels_carry_an_off_budget_p_winner_past_the_gate():
    """An off-budget, non-curated P winner on THREE channels ([M-H]-, [M+NO3]-, [M+Br]-) passes the
    gate (three ions are independent evidence the pair is not) and displaces a weak incumbent."""
    mh, mn, mb = (CH.ion_mz(MPA, a) for a in ("[M-H]-", "[M+NO3]-", "[M+Br]-"))
    members = [("h", "CH4O3P-", mh), ("n", "CH5NO6P-", mn), ("b", "CH5BrO3P-", mb)]
    led = _file([("h", mh, 3000.0), ("n", mn, 3000.0), ("b", mb, 3000.0)])
    _weak(led, "n")
    s, by = _p7(led, _scorer(MPA, members), HN + ["[M+Br]-"])
    assert s["committed"] == 1 and s["gated"] == 0 and s["displaced"] == 1, s
    assert by.loc["n", "neutral_formula"] == MPA


def test_a_curated_p_winner_with_its_s_line_confirmed_displaces_on_two_channels():
    """A P + S winner a reference list names, its [M-H]- / [M+NO3]- members both showing their 34S
    line: the P budget does not gate a curated formula, the confirmed envelope makes the
    certificate strong, and it displaces the weak incumbent. Not curated, the same pair commits
    nothing (an off-budget P on a reagent-acid pair)."""
    w = "CH5O2PS"
    mh, mn = CH.ion_mz(w, "[M-H]-"), CH.ion_mz(w, "[M+NO3]-")
    members = [("h", "CH4O2PS-", mh), ("n", "CH5NO5PS-", mn)]
    lines = [("h", mh, 4000.0), ("n", mn, 3000.0), ("h34", mh + ISO.D_34S, 4000.0 * ISO.R_34S_PER_S),
             ("n34", mn + ISO.D_34S, 3000.0 * ISO.R_34S_PER_S)]
    led = _file(lines)
    _weak(led, "n")
    s, by = _p7(led, _scorer(w, members), HN, cfg=_cfg(reflist_formulas=frozenset({w})))
    assert s["committed"] == 1 and s["displaced"] == 1 and by.loc["n", "neutral_formula"] == w, s
    led = _file(lines)
    _weak(led, "n")
    s, by = _p7(led, _scorer(w, members), HN)
    assert s["committed"] == 0 and s["gated"] == 1 and by.loc["n", "neutral_formula"] == "C3H4N2O8", s
