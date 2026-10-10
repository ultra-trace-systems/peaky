"""The minor-line response of an Orbitrap peak list (assignment/iso_response.py) and where it is read:
the local scorer, the 13C carbon clamps and the residual carbon count."""
import math

import numpy as np
import pandas as pd
import pytest

from peaky.assignment import iso_response as IR
from peaky.assignment import ledger as L
from peaky.assignment import passes as P
from peaky.assignment.residual import carbon_count_from_13c

R = IR.R13C


def _peaks_with_response(resp_fn, n_ions=40, seed=0):
    """CHO ions C6..C15, each M0 with its 13C line read at resp_fn(expected line S/N) of its share."""
    rng = np.random.default_rng(seed)
    rows, ions = [], []
    for k in range(n_ions):
        c = 6 + k % 10
        m0 = 150.0 + 7.3 * k
        snr = 10 ** rng.uniform(2.0, 4.0)               # M0 S/N 100 .. 10000
        h0 = snr * 50.0
        x = math.log10(snr * c * R)
        f = resp_fn(x)
        rows += [dict(peak_id=f"m{k}", mz=m0, height=h0, area=h0 * 1e-3, signal_to_noise=snr),
                 dict(peak_id=f"c{k}", mz=m0 + IR.D13C, height=h0 * c * R * f, area=h0 * 1e-3 * c * R * f,
                      signal_to_noise=snr * c * R * f)]
        ions.append((m0, f"C{c}H10O4-"))
    return pd.DataFrame(rows), ions


def test_fit_recovers_a_rising_response_monotone_and_capped():
    resp = IR.fit(*_peaks_with_response(lambda x: min(1.0, 0.3 + 0.35 * max(x - 0.5, 0))))
    assert resp is not None and resp["n"] == 40 and resp["col"] == "area"
    assert all(b >= a for a, b in zip(resp["y"], resp["y"][1:]))          # non-decreasing
    assert max(resp["y"]) <= IR.CAP and min(resp["y"]) >= IR.FLOOR
    assert IR.scale(resp, 0.6) < 0.6 and IR.scale(resp, 2.9) > 0.9
    assert IR.is_material(resp)


def test_a_dim_bin_that_reads_high_never_raises_the_expected_line():
    # detection-selected dim lines read 2x high; bright ones read 0.85
    resp = IR.fit(*_peaks_with_response(lambda x: 2.0 if x < 1.2 else 0.85))
    assert resp is not None and max(resp["y"]) <= 1.0
    assert IR.scale(resp, 2.5) == pytest.approx(0.85, abs=0.05)


def test_too_few_lines_give_no_response():
    peaks, ions = _peaks_with_response(lambda x: 0.6, n_ions=IR.MIN_POINTS - 1)
    assert IR.fit(peaks, ions) is None
    assert IR.scale(None, 1.0) == 1.0 and not IR.is_material(None)


def test_heteroatom_ions_are_not_fitted_on():
    peaks, ions = _peaks_with_response(lambda x: 0.6)
    assert IR.fit(peaks, [(m, f.replace("O4", "O4Cl")) for m, f in ions]) is None


def test_scaled_labels_are_carbon_and_oxygen_lines_only():
    assert IR.is_scaled_label("13C") and IR.is_scaled_label("13C2") and IR.is_scaled_label("18O")
    assert IR.is_scaled_label("13C18O")
    assert not IR.is_scaled_label("M0") and not IR.is_scaled_label("37Cl") and not IR.is_scaled_label("15N")
    assert not IR.is_scaled_label("34S") and not IR.is_scaled_label("81Br")


def test_carbon_from_13c_reads_area_against_the_response():
    resp = {"x": [0.0, 3.0], "y": [0.8, 0.8], "n": 20, "col": "area"}
    # a C10 whose 13C reads 0.8 of its share by area (heights read lower still)
    c = IR.carbon_from_13c(resp, h0=1e4, h_sat=1e4 * 10 * R * 0.7, a0=10.0, a_sat=10.0 * 10 * R * 0.8,
                           snr0=500.0, n_c=10)
    assert c == pytest.approx(10.0, abs=0.05)
    assert IR.carbon_from_13c(None, h0=1e4, h_sat=1e4 * 10 * R * 0.7) == pytest.approx(7.0, abs=0.05)


# ---- the local scorer ------------------------------------------------------------------------------
def _envelope_peaks(f13):
    from mascope_tools.composition.heuristic_filter import anchor_on_monoisotopic, predict_isotopes
    pm, pi, lab = anchor_on_monoisotopic(*predict_isotopes("C10H16NO10", -1, None))
    rel = np.asarray(pi) / pi[0]
    rows = []
    for k, (m, r, l) in enumerate(zip(pm, rel, lab)):
        f = 1.0 if l == "M0" else f13
        h = 2e4 * r * f
        rows.append(dict(peak_id=f"p{k}", mz=float(m), height=h, area=h * 1e-3, signal_to_noise=h / 10.0))
    return pd.DataFrame(rows)


def _score(peaks, **kw):
    from mascope_tools.composition import PatternScoring
    from peaky.io import local_scoring as LS
    out = LS.score_candidates_local(peaks, ["C10H16O7"], ["[M+NO3]-"], scoring=PatternScoring(sigma_ppm=0.58),
                                    **kw)
    return float(out[out.is_base].ion_score.iloc[0])


def test_a_uniformly_under_read_envelope_passes_against_its_response():
    peaks = _envelope_peaks(0.75)
    bare = _score(peaks)
    resp = {"x": [0.0, 4.0], "y": [0.75, 0.75], "n": 20, "col": "area"}
    fixed = _score(peaks, intensity_col="area", iso_response=resp)
    assert bare < 0.7 <= fixed


def test_no_response_scores_exactly_as_before():
    peaks = _envelope_peaks(1.0)
    assert _score(peaks) == pytest.approx(_score(peaks, iso_response=None))


# ---- the run's mode ----------------------------------------------------------------------------------
def _raw(area=True):
    d = pd.DataFrame({"mz": [100.0, 200.0], "height": [1.0, 2.0]})
    if area:
        d["area"] = [0.1, 0.2]
    return d


@pytest.mark.parametrize("klass, knob, area, local, trace, want", [
    ("orbitrap", "auto", True, True, False, "auto"),
    ("orbitrap", "area", True, True, False, "area"),
    ("orbitrap", "off", True, True, False, "off"),
    ("tof", "auto", True, True, False, "off"),
    (None, "auto", True, True, False, "off"),
    ("orbitrap", "auto", False, True, False, "off"),
    ("orbitrap", "auto", True, False, False, "off"),        # network scorer: nothing to scale
    ("orbitrap", "auto", True, True, True, "off"),          # trace-first's synthetic sample
])
def test_iso_mode(klass, knob, area, local, trace, want):
    from peaky.assignment.assign import _iso_mode
    cfg = P.PassConfig(iso_response=knob)
    cfg.instrument_class, cfg.trace_sample = klass, trace
    assert _iso_mode(cfg, _raw(area), local) == want


def test_the_fitted_response_is_a_runtime_field():
    assert "iso_response_fit" in P.PassConfig.RUNTIME_FIELDS
    assert "iso_response" not in P.PassConfig.RUNTIME_FIELDS     # the knob is fingerprinted


# ---- the carbon counts that read 13C -----------------------------------------------------------------
def _ledger(rows):
    return L.new_ledger(pd.DataFrame(rows))


def test_residual_carbon_count_brackets_the_true_count_against_the_response():
    resp = {"x": [0.0, 4.0], "y": [0.7, 0.7], "n": 20, "col": "area"}
    led = _ledger([dict(peak_id="a", mz=300.0, height=1e4, area=10.0, signal_to_noise=500.0),
                   dict(peak_id="b", mz=301.00335, height=1e4 * 10 * R * 0.7, area=10.0 * 10 * R * 0.7,
                        signal_to_noise=40.0)])
    lo, hi = carbon_count_from_13c(led, "a")
    assert hi < 10                                               # read raw (C7, bracket 6-9): excludes C10
    lo, hi = carbon_count_from_13c(led, "a", resp=resp)
    assert lo <= 10 <= hi


def _audit_case(n_true, c_line, resp):
    led = _ledger([dict(peak_id="M", mz=310.078, height=2e4, area=20.0, signal_to_noise=800.0),
                   dict(peak_id="S", mz=311.0813, height=2e4 * c_line * R, area=20.0 * c_line * R,
                        signal_to_noise=800.0 * c_line * R)])
    L.commit_assignment(led, "M", neutral_formula=f"C{n_true}H16O7", adduct="[M+NO3]-",
                        ion_formula=f"C{n_true}H16NO10-", ion_score=0.9, ppm_error=0.1, pass_no=1,
                        method="test", confidence="Good", commentary="t")
    cfg = P.PassConfig(height_cutoff_cps=100.0)
    cfg.iso_response_fit = resp
    P.audit_isotopes(led, cfg, log=lambda *a: None)
    return L.role_of(led, "M")


def test_the_audit_keeps_a_true_c10_read_low_and_still_clears_an_overclaim():
    resp = {"x": [0.0, 4.0], "y": [0.6, 0.6], "n": 20, "col": "area"}
    # a C10 whose 13C reads 0.6 of its share (C6 by the raw ratio): cleared without the response
    assert _audit_case(10, 10 * 0.6, None) != L.ROLE_M0
    assert _audit_case(10, 10 * 0.6, resp) == L.ROLE_M0
    # a C17 claim on a line whose 13C reads C9 x 0.6: still cleared against the response
    assert _audit_case(17, 9 * 0.6, resp) != L.ROLE_M0


def test_a_bright_line_reading_its_natural_share_is_not_divided_up_into_a_clear():
    # the 13C reads 1.2x its share; the file's response at that S/N is 0.5: corrected it reads C24, raw C12
    resp = {"x": [0.0, 4.0], "y": [0.5, 0.5], "n": 20, "col": "area"}
    assert _audit_case(10, 10 * 1.2, resp) == L.ROLE_M0
    bad, c = IR.contradicts(resp, 10, h0=1e4, h_sat=1e4 * 12 * R, a0=10.0, a_sat=10.0 * 12 * R, snr0=100.0)
    assert not bad and c > 20


def test_the_residual_bracket_is_the_union_of_the_raw_and_the_corrected_reading():
    resp = {"x": [0.0, 4.0], "y": [0.5, 0.5], "n": 20, "col": "area"}
    led = _ledger([dict(peak_id="a", mz=300.0, height=1e4, area=10.0, signal_to_noise=500.0),
                   dict(peak_id="b", mz=301.00335, height=1e4 * 10 * R, area=10.0 * 10 * R, signal_to_noise=40.0)])
    lo, hi = carbon_count_from_13c(led, "a", resp=resp)
    assert lo <= 10 <= hi and hi >= 20


# ---- refute round 1: one test per defect --------------------------------------------------------------
def test_d1_area_scoring_still_reports_the_line_heights():
    # every reader of sample_peak_intensity compares it with height gates in cps (arbitration's
    # heteroatom observability waiver among them): an area must never stand in for it
    from mascope_tools.composition import PatternScoring
    from peaky.io import local_scoring as LS
    peaks = _envelope_peaks(1.0)
    out = LS.score_candidates_local(peaks, ["C10H16O7"], ["[M+NO3]-"], scoring=PatternScoring(sigma_ppm=0.58),
                                    intensity_col="area", iso_response={"x": [0.0, 4.0], "y": [0.8, 0.8],
                                                                        "n": 20, "col": "area"})
    m = out[out.sample_peak_id.notna()]
    h = peaks.set_index("peak_id").height
    assert len(m) >= 2
    assert np.allclose(m.sample_peak_intensity.astype(float), h.loc[m.sample_peak_id].to_numpy())


def test_d2_a_line_at_its_natural_share_still_matches_under_a_low_response():
    # a flat 0.5 response must not move the prediction so far that a line reading its natural share
    # falls outside the match gate and is charged as absent
    peaks = _envelope_peaks(1.0)
    resp = {"x": [0.0, 4.0], "y": [0.5, 0.5], "n": 20, "col": "area"}
    assert _score(peaks, intensity_col="area", iso_response=resp) == pytest.approx(
        _score(peaks, intensity_col="area"), abs=0.02)


def test_d2_the_response_returns_to_one_above_the_brightest_fitted_line():
    resp = {"x": [0.5, 1.5], "y": [0.4, 0.7], "n": 20, "col": "area"}
    assert IR.scale(resp, 1.5) == pytest.approx(0.7)
    assert IR.scale(resp, 1.5 + IR.RAMP / 2) == pytest.approx(0.85)
    assert IR.scale(resp, 1.5 + IR.RAMP) == 1.0 and IR.scale(resp, 9.0) == 1.0
    assert IR.scale(resp, 0.0) == pytest.approx(0.4)                # held at the dimmest point below it


def test_d2_the_band_spans_the_censored_to_the_natural_share():
    resp = {"x": [0.0, 1.0, 2.0], "y": [0.3, 0.6, 1.0], "n": 20, "col": "area"}
    assert IR.band(resp, 0.1, 10.0) == pytest.approx(0.1 * 0.3)      # line S/N 1
    assert IR.band(resp, 0.1, 1000.0) == pytest.approx(0.1)          # line S/N 100: a bright knot at 1.0
    assert IR.band(resp, 0.1, 1e6) == pytest.approx(0.1)             # held above it
    assert IR.band(None, 0.1, 10.0) == 0.1


def test_the_band_reads_the_lower_quantile_and_holds_a_bright_knot():
    resp = {"x": [1.0, 2.5], "y": [0.6, 0.95], "lo": [0.4, 0.9], "n": 30, "col": "area"}
    assert IR.band_scale(resp, 1.0) == pytest.approx(0.4) and IR.scale(resp, 1.0) == pytest.approx(0.6)
    assert IR.band_scale(resp, 4.0) == pytest.approx(0.9)            # bright data: held, never 0.8
    dim = {"x": [1.0, 1.4], "y": [0.5, 0.6], "lo": [0.35, 0.45], "n": 30, "col": "area"}
    assert IR.band_scale(dim, 1.4 + IR.RAMP / 2) == pytest.approx(0.45 + (IR.BAND_BRIGHT - 0.45) / 2)
    assert IR.band_scale(dim, 9.0) == pytest.approx(0.8)             # no bright data: the instrument residual
    assert IR.band_scale(None, 1.0) == 1.0


def test_the_fit_keeps_sparse_bright_lines_as_a_knot_and_reports_a_lower_quantile():
    rows, ions = [], []
    for k in range(14):
        snr = 300.0 if k < 12 else (3e4 if k == 12 else 6e4)   # 12 dim lines and 2 bright ones, far apart
        f = 0.5 + 0.02 * (k % 4) if k < 12 else 0.9
        h0 = snr * 50.0
        rows += [dict(peak_id=f"m{k}", mz=150.0 + 7.3 * k, height=h0, area=h0, signal_to_noise=snr),
                 dict(peak_id=f"c{k}", mz=150.0 + 7.3 * k + IR.D13C, height=h0 * 10 * R * f,
                      area=h0 * 10 * R * f, signal_to_noise=snr * 0.1 * f)]
        ions.append((150.0 + 7.3 * k, "C10H10O4-"))
    resp = IR.fit(pd.DataFrame(rows), ions)
    assert resp["n"] == 14 and len(resp["x"]) == 1                  # 2 bright lines alone make no knot: joined
    assert resp["x"][0] > math.log10(300.0 * 10 * R) + 0.2           # ... into the last knot, not dropped
    assert resp["lo"][0] <= resp["y"][0]
    rows2 = rows + [dict(r, peak_id=r["peak_id"] + "b", mz=r["mz"] + 500.0) for r in rows[24:]]
    ions2 = ions + [(m + 500.0, f) for m, f in ions[12:]]
    resp2 = IR.fit(pd.DataFrame(rows2), ions2)                       # 4 bright lines: a bright knot
    assert len(resp2["x"]) == 2 and resp2["y"][-1] == pytest.approx(0.9) and resp2["x"][-1] > IR.X_BRIGHT


def _alias_scores(f13, resp):
    """the documented sub-ppm alias of a 15N-nitrate run: truth C9H13NO5 [M+^NO3]- vs rival C9H14O6 [M+CO3]-
    (+1 C, 0.443 mDa), the M0 midway so the mass terms tie, its 13C reading f13 of the truth's natural share"""
    from mascope_tools.composition import PatternScoring
    from mascope_tools.composition.heuristic_filter import anchor_on_monoisotopic, predict_isotopes
    from peaky.io import local_scoring as LS

    def env(ion):
        pm, pi, lab = anchor_on_monoisotopic(*predict_isotopes(ion, -1, 0.986))
        return np.asarray(pm), np.asarray(pi) / pi[0], list(lab)
    t_mz, t_rel, t_lab = env("C9H13N^NO8")
    r_mz = env("C10H14O9")[0]
    mid = 0.5 * (t_mz[0] + r_mz[0])
    rows = []
    for k, (m, rr, lab) in enumerate(zip(t_mz, t_rel, t_lab)):
        if rr >= 1e-3:
            h = 1e6 * rr * (f13 if lab == "13C" else 1.0)
            rows.append(dict(peak_id=f"p{k}", mz=float(m - t_mz[0] + mid), height=h, area=h * 1e-3,
                             signal_to_noise=h / 200.0))
    out = LS.score_candidates_local(pd.DataFrame(rows), ["C9H13NO5", "C9H14O6"], ["[M+^NO3]-", "[M+CO3]-"],
                                    scoring=PatternScoring(sigma_ppm=0.58, mz_tolerance_ppm=5.0),
                                    intensity_col="area", iso_response=resp, purity=0.986)
    b = out[out.is_base]
    t = b[(b.compound_formula == "C9H13NO5") & b.mechanism_id.str.contains("NO3")].ion_score.iloc[0]
    r = b[(b.compound_formula == "C9H14O6") & b.mechanism_id.str.contains("CO3")].ion_score.iloc[0]
    return float(t), float(r)


@pytest.mark.parametrize("f13", [1.0, 1.02])
def test_on_a_file_whose_bright_lines_read_natural_the_13c_still_splits_the_15n_nitrate_alias(f13):
    resp = {"x": [0.3, 1.0, 2.6], "y": [0.45, 0.7, 1.0], "lo": [0.3, 0.55, 0.95], "n": 40, "col": "area"}
    t, r = _alias_scores(f13, resp)
    assert t > r                    # the band is the file's bright spread (0.95-1.0): narrower margin, same winner
    # the 0.8 floor the bright lines of OTHER files read at must not reach this file: the rival would tie or win
    t8, r8 = _alias_scores(f13, dict(resp, lo=[0.3, 0.55, 0.8]))
    assert r8 >= t8 - 1e-9


def test_on_a_file_whose_bright_lines_read_low_the_truth_reading_low_still_wins_the_alias():
    resp = {"x": [0.3, 1.0, 2.6], "y": [0.45, 0.7, 0.86], "lo": [0.3, 0.55, 0.8], "n": 40, "col": "area"}
    t, r = _alias_scores(0.86, resp)
    assert t > r


def test_a_reading_above_the_natural_share_is_charged_with_a_response():
    resp = {"x": [0.0, 4.0], "y": [0.5, 0.5], "lo": [0.4, 0.4], "n": 20, "col": "area"}
    at, above = _envelope_peaks(1.0), _envelope_peaks(1.3)
    assert _score(above, intensity_col="area", iso_response=resp) < _score(at, intensity_col="area", iso_response=resp)


def test_a_bright_line_reading_the_instrument_residual_is_inside_the_band():
    # 13C at 0.8 of its share on a bright envelope: inside the band once the file has a response, a 4-sigma
    # miss without one
    peaks = _envelope_peaks(0.8)
    resp = {"x": [1.3], "y": [0.5], "n": 16, "col": "area"}
    assert _score(peaks, intensity_col="area") < 0.7 <= _score(peaks, intensity_col="area", iso_response=resp)


def test_d3_readings_that_disagree_in_direction_do_not_contradict():
    # height reads C6 (low), area against the response reads C16 (high): neither says the same thing
    resp = {"x": [0.0, 4.0], "y": [0.6, 0.6], "n": 20, "col": "area"}
    bad, _ = IR.contradicts(resp, 10, h0=1e4, h_sat=1e4 * 6 * IR.R13C_RAW, a0=10.0, a_sat=10.0 * 9.6 * R,
                            snr0=500.0)
    assert not bad
    # both low: contradicted
    bad, c = IR.contradicts(resp, 17, h0=1e4, h_sat=1e4 * 6 * IR.R13C_RAW, a0=10.0, a_sat=10.0 * 5.4 * R,
                            snr0=500.0)
    assert bad and c == pytest.approx(9.0, abs=0.1)


def test_d5_without_a_response_or_area_mode_the_check_is_the_old_one():
    bad, c = IR.contradicts(None, 10, h0=1e4, h_sat=1e4 * 10 * 0.0107)
    assert not bad and c == pytest.approx(10.0)
    bad, c = IR.contradicts(None, 17, h0=1e4, h_sat=1e4 * 10 * 0.0107)
    assert bad and c == pytest.approx(10.0)


def test_d6_area_mode_without_a_response_clears_only_where_the_area_agrees():
    # the scorer accepted the line by area; a height that reads C6 alone must not clear the C10
    bad, _ = IR.contradicts(None, 10, h0=1e4, h_sat=1e4 * 6 * IR.R13C_RAW, a0=10.0, a_sat=10.0 * 10 * R,
                            area_mode=True)
    assert not bad
    bad, _ = IR.contradicts(None, 10, h0=1e4, h_sat=1e4 * 6 * IR.R13C_RAW, a0=10.0, a_sat=10.0 * 10 * R)
    assert bad


def _audit_area_case(n_true, c_height, c_area, *, iso_area, resp=None):
    led = _ledger([dict(peak_id="M", mz=310.078, height=2e4, area=20.0, signal_to_noise=800.0),
                   dict(peak_id="S", mz=311.0813, height=2e4 * c_height * IR.R13C_RAW, area=20.0 * c_area * R,
                        signal_to_noise=800.0 * c_height * R)])
    L.commit_assignment(led, "M", neutral_formula=f"C{n_true}H16O7", adduct="[M+NO3]-",
                        ion_formula=f"C{n_true}H16NO10-", ion_score=0.9, ppm_error=0.1, pass_no=1,
                        method="test", confidence="Good", commentary="t")
    cfg = P.PassConfig(height_cutoff_cps=100.0)
    cfg.iso_response_fit, cfg.iso_area = resp, iso_area
    P.audit_isotopes(led, cfg, log=lambda *a: None)
    return L.role_of(led, "M")


def test_d6_the_audit_reads_the_run_s_area_mode():
    assert _audit_area_case(10, 6, 10, iso_area=False) != L.ROLE_M0
    assert _audit_area_case(10, 6, 10, iso_area=True) == L.ROLE_M0
    assert _audit_area_case(17, 9, 9, iso_area=True) != L.ROLE_M0     # both read C9: still an overclaim


def test_d6_the_pre_pass_4_clamp_reads_the_run_s_area_mode_and_response():
    def case(c_height, c_area, *, iso_area, resp=None):
        led = _ledger([dict(peak_id="M", mz=310.078, height=2e4, area=20.0, signal_to_noise=800.0),
                       dict(peak_id="S", mz=311.0813, height=2e4 * c_height * IR.R13C_RAW,
                            area=20.0 * c_area * R, signal_to_noise=800.0 * c_height * R)])
        L.commit_assignment(led, "M", neutral_formula="C10H16O7", adduct="[M+NO3]-", ion_formula="C10H16NO10-",
                            ion_score=0.9, ppm_error=0.1, pass_no=1, method="test", confidence="Good",
                            commentary="t")
        cfg = P.PassConfig(height_cutoff_cps=100.0)
        cfg.iso_response_fit, cfg.iso_area = resp, iso_area
        return P.demote_carbon_inconsistent(led, cfg, log=lambda *a: None)
    assert case(6, 10, iso_area=False) == 1
    assert case(6, 10, iso_area=True) == 0
    resp = {"x": [0.0, 4.0], "y": [0.6, 0.6], "n": 20, "col": "area"}
    assert case(6, 6, iso_area=True) == 1                       # by area alone: C6 low
    assert case(6, 6, iso_area=True, resp=resp) == 0            # against the response: C10


def test_d6_the_missing_13c_clear_judges_against_the_censored_expectation():
    # a C10 at S/N 30: its 13C line (expected S/N ~3) is censored to 0.3 of its share by the response, so
    # an absent line is what the file predicts -- not grounds for a clear
    def case(resp):
        led = _ledger([dict(peak_id="M", mz=310.078, height=3000.0, area=3.0, signal_to_noise=30.0),
                       dict(peak_id="far", mz=400.0, height=500.0, area=0.5, signal_to_noise=5.0)])
        L.commit_assignment(led, "M", neutral_formula="C10H16O7", adduct="[M+NO3]-", ion_formula="C10H16NO10-",
                            ion_score=0.9, ppm_error=0.1, pass_no=1, method="test", confidence="Fair",
                            commentary="t")
        cfg = P.PassConfig(height_cutoff_cps=100.0)
        cfg.iso_response_fit = resp
        P.audit_isotopes(led, cfg, log=lambda *a: None)
        return L.role_of(led, "M")
    assert case(None) != L.ROLE_M0                    # 321 cps expected >= 1.5 x 100: cleared as before
    assert case({"x": [0.0, 2.0], "y": [0.3, 0.9], "n": 20, "col": "area"}) == L.ROLE_M0


def test_d6_the_sweeper_attaches_a_censored_13c_line():
    def case(resp, frac):
        led = _ledger([dict(peak_id="M", mz=310.078, height=3000.0, area=3.0, signal_to_noise=30.0),
                       dict(peak_id="S", mz=311.0813, height=3000.0 * 10 * IR.R13C_RAW * frac,
                            area=3.0 * 10 * R * frac, signal_to_noise=3.0)])
        L.commit_assignment(led, "M", neutral_formula="C10H16O7", adduct="[M+NO3]-", ion_formula="C10H16NO10-",
                            ion_score=0.9, ppm_error=0.1, pass_no=1, method="test", confidence="Fair",
                            commentary="t")
        cfg = P.PassConfig(height_cutoff_cps=10.0)
        cfg.iso_response_fit = resp
        out = P.audit_isotopes(led, cfg, log=lambda *a: None)
        return out["c13_attached"]
    assert case(None, 0.2) == 0                       # 0.2 of its share: outside the 0.3-2.5 band
    assert case({"x": [0.0, 2.0], "y": [0.3, 0.9], "n": 20, "col": "area"}, 0.2) == 1
    assert case({"x": [0.0, 2.0], "y": [0.3, 0.9], "n": 20, "col": "area"}, 3.0) == 0   # the upper end stays
    assert case({"x": [0.0, 2.0], "y": [0.3, 0.9], "n": 20, "col": "area"}, 2.0) == 1   # at the natural share


def test_d7_an_m0_without_an_area_is_read_by_height():
    peaks = _envelope_peaks(1.0)
    by_h = _score(peaks)
    peaks.loc[0, "area"] = np.nan
    assert _score(peaks, intensity_col="area") == pytest.approx(by_h)
    peaks.loc[0, "area"] = 0.0
    assert _score(peaks, intensity_col="area") == pytest.approx(by_h)


def test_d8_isospec_joined_labels_are_scaled():
    assert IR.is_scaled_label("13C+18O") and IR.is_scaled_label("13C2+18O")
    assert not IR.is_scaled_label("13C+34S") and not IR.is_scaled_label("37Cl+13C")


# ---- the stand-in (a decoy arm) inherits the isotope-line scoring -------------------------------------
def _orbi():
    from mascope_tools.composition import PatternScoring
    return PatternScoring(sigma_ppm=0.58, mu_ppm=0.0, mz_tolerance_ppm=5.0)


_ORBI = _orbi()


def _snap(**extra):
    return {"sigma_ppm": 0.58, "mu_ppm": 0.0, "mz_tolerance_ppm": 5.0, "sigma_source": "fitted",
            "mu_source": "fitted", **extra}


def test_d4_a_stand_in_inherits_the_measured_sample_s_isotope_scoring_and_keeps_it():
    from peaky.io import io_mascope as IO
    resp = {"x": [0.0, 4.0], "y": [0.6, 0.6], "n": 20, "col": "area"}
    peaks = _envelope_peaks(1.0)
    IO.register_offline_sample("iso-arm", peaks, ["[M+NO3]-"],
                               scoring=_snap(iso_scoring={"col": "area", "resp": resp}))
    try:
        IO.scoring_for_sample(None, "iso-arm", peaks)
        assert IO.iso_scoring("iso-arm") == {"col": "area", "resp": resp}
        assert IO.iso_scoring_inherited("iso-arm")
        IO.reset_iso_scoring("iso-arm")                       # a new run of the stand-in
        assert IO.iso_scoring("iso-arm")["resp"] == resp
        assert IO.scoring_snapshot(None, "iso-arm", peaks)["iso_scoring"]["resp"] == resp
    finally:
        IO.unregister_offline_sample("iso-arm")
    assert IO.iso_scoring("iso-arm") is None and not IO.iso_scoring_inherited("iso-arm")


def test_d4_a_run_s_own_isotope_scoring_is_forgotten_by_the_next_run():
    from peaky.io import io_mascope as IO
    peaks = _envelope_peaks(1.0)
    IO.register_offline_sample("iso-own", peaks, ["[M+NO3]-"], scoring=_ORBI)
    try:
        IO.scoring_for_sample(None, "iso-own", peaks)
        IO.set_iso_scoring("iso-own", col="area", resp={"x": [0.0], "y": [0.5], "n": 10, "col": "area"})
        assert IO.scoring_snapshot(None, "iso-own", peaks)["iso_scoring"]["col"] == "area"
        IO.reset_iso_scoring("iso-own")
        assert IO.iso_scoring("iso-own") is None
        assert "iso_scoring" not in IO.scoring_snapshot(None, "iso-own", peaks)
    finally:
        IO.unregister_offline_sample("iso-own")


def test_the_local_scorer_is_handed_the_sample_s_column_and_response(monkeypatch):
    from peaky.io import io_mascope as IO
    from peaky.io import local_scoring as LS
    seen = {}

    def spy(peaks, formulas, mechanisms=None, **kw):
        seen.update(kw)
        return pd.DataFrame()
    peaks = _envelope_peaks(1.0)
    IO.register_offline_sample("iso-pass", peaks, ["[M+NO3]-"], scoring=_ORBI)
    try:
        resp = {"x": [0.0], "y": [0.5], "n": 10, "col": "area"}
        IO.scoring_for_sample(None, "iso-pass", peaks)
        IO.set_iso_scoring("iso-pass", col="area", resp=resp)
        monkeypatch.setattr(LS, "score_candidates_local", spy)
        IO._score_candidates_local(None, "iso-pass", ["C10H16O7"], [])
        assert seen["intensity_col"] == "area" and seen["iso_response"] == resp
        IO.reset_iso_scoring("iso-pass")
        IO._score_candidates_local(None, "iso-pass", ["C10H16O7"], [])
        assert seen["intensity_col"] == "height" and seen["iso_response"] is None
    finally:
        IO.unregister_offline_sample("iso-pass")


# ---- surviving mutants of round 1 ---------------------------------------------------------------------
def test_the_fit_is_made_non_decreasing():
    # the middle reads low: pooled with its neighbour, never a dip
    resp = IR.fit(*_peaks_with_response(lambda x: 0.6 if x < 1.6 else (0.4 if x < 2.3 else 0.9), n_ions=120))
    assert resp is not None and all(b >= a for a, b in zip(resp["y"], resp["y"][1:]))
    assert len(resp["y"]) >= 4 and resp["y"][0] < 0.6


def test_the_fit_reads_the_column_it_is_given():
    peaks, ions = _peaks_with_response(lambda x: 0.95)
    peaks.loc[peaks.peak_id.str.startswith("c"), "height"] *= 0.5     # heights read lower still
    a, h = IR.fit(peaks, ions, col="area"), IR.fit(peaks, ions, col="height")
    assert min(a["y"]) > 0.9 and max(h["y"]) < 0.6 and h["col"] == "height"


def test_the_fit_needs_ten_lines_and_three_per_bin():
    def lines(n, snrs):
        rows, ions = [], []
        for k in range(n):
            snr = snrs(k)
            h0 = snr * 50.0
            rows += [dict(peak_id=f"m{k}", mz=150.0 + 7.3 * k, height=h0, area=h0, signal_to_noise=snr),
                     dict(peak_id=f"c{k}", mz=150.0 + 7.3 * k + IR.D13C, height=h0 * 10 * R * 0.6,
                          area=h0 * 10 * R * 0.6, signal_to_noise=snr * 0.06)]
            ions.append((150.0 + 7.3 * k, "C10H10O4-"))
        return pd.DataFrame(rows), ions
    assert IR.fit(*lines(9, lambda k: 300.0)) is None
    assert IR.fit(*lines(10, lambda k: 300.0)) is not None
    # 12 lines at one S/N and 2 far brighter: the 2 make no knot of their own
    rows, ions2 = [], []
    for k in range(14):
        snr = 300.0 if k < 12 else 3e5
        h0 = snr * 50.0
        rows += [dict(peak_id=f"m{k}", mz=150.0 + 7.3 * k, height=h0, area=h0, signal_to_noise=snr),
                 dict(peak_id=f"c{k}", mz=150.0 + 7.3 * k + IR.D13C, height=h0 * 10 * R * 0.6,
                      area=h0 * 10 * R * 0.6, signal_to_noise=snr * 0.06)]
        ions2.append((150.0 + 7.3 * k, "C10H10O4-"))
    resp = IR.fit(pd.DataFrame(rows), ions2)
    assert resp is not None and len(resp["x"]) == 1 and resp["n"] == 14


def test_the_fit_clips_at_the_floor():
    resp = IR.fit(*_peaks_with_response(lambda x: 0.1))
    assert resp is not None and min(resp["y"]) == pytest.approx(0.3)


def test_the_fit_skips_lines_off_the_13c_position_blends_and_small_ions():
    peaks, ions = _peaks_with_response(lambda x: 0.6)
    off = peaks.copy()
    off.loc[off.peak_id.str.startswith("c"), "mz"] *= 1 + 10e-6         # 10 ppm off: not this ion's 13C
    assert IR.fit(off, ions) is None
    blend = peaks.copy()
    blend.loc[blend.peak_id.str.startswith("c"), "area"] *= 6.0          # reads 3.6x: a blend, not a 13C
    assert IR.fit(blend, ions) is None
    assert IR.fit(peaks, [(m, "C3H10O4-") for m, _ in ions]) is None     # C3: not fitted on


def test_materiality():
    assert IR.is_material({"x": [0.0, 1.0], "y": [0.85, 1.0]})
    assert not IR.is_material({"x": [0.0], "y": [0.92]})


def test_the_scorer_scales_minor_c_o_lines_at_their_own_s_n_and_never_the_m0():
    from peaky.io import local_scoring as LS
    resp = {"x": [0.0, 1.0, 2.0], "y": [0.3, 0.6, 1.0], "n": 20, "col": "area"}
    pred = np.array([1.0, 0.1, 0.02, 0.3])
    labels = ["M0", "13C", "18O", "37Cl"]
    dim, bright = LS._respond(pred, labels, 10.0, resp), LS._respond(pred, labels, 1e4, resp)
    assert dim[0] == 1.0 and bright[0] == 1.0 and dim[3] == 0.3
    assert dim[1] == pytest.approx(0.1 * 0.3) and bright[1] == pytest.approx(0.1)
    assert dim[2] == pytest.approx(0.02 * 0.3)


@pytest.mark.parametrize("cover, want", [(1.0, "auto"), (0.95, "auto"), (0.5, "off")])
def test_iso_mode_needs_areas_on_nine_peaks_in_ten(cover, want):
    from peaky.assignment.assign import _iso_mode
    n = 20
    d = pd.DataFrame({"mz": np.arange(n, dtype=float), "height": np.ones(n),
                      "area": [1.0 if k < round(cover * n) else np.nan for k in range(n)]})
    cfg = P.PassConfig(iso_response="auto")
    cfg.instrument_class, cfg.trace_sample = "orbitrap", False
    assert _iso_mode(cfg, d, True) == want


def test_stage_a_reads_the_sample_s_isotope_scoring(monkeypatch):
    from peaky.assignment import residual as RS
    from peaky.io import io_mascope as IO
    resp = {"x": [0.0, 4.0], "y": [0.5, 0.5], "n": 20, "col": "area"}
    # a C10 whose 13C reads 0.5 of its share by height and by area
    led = _ledger([dict(peak_id="a", mz=300.0, height=1e4, area=10.0, signal_to_noise=500.0),
                   dict(peak_id="b", mz=301.00335, height=1e4 * 5 * R, area=10.0 * 5 * R, signal_to_noise=40.0)])
    assert RS._carbon_clamp(led, "a", "iso-stage-a")[1] < 10
    monkeypatch.setitem(IO._ISO_SCORING, "iso-stage-a", {"col": "area", "resp": resp})
    lo, hi = RS._carbon_clamp(led, "a", "iso-stage-a")
    assert lo <= 10 <= hi
    # by area, no response: the area reading joins the bracket
    led.loc[led.peak_id == "b", "area"] = 10.0 * 10 * R
    monkeypatch.setitem(IO._ISO_SCORING, "iso-stage-a", {"col": "area", "resp": None})
    lo, hi = RS._carbon_clamp(led, "a", "iso-stage-a")
    assert lo <= 10 <= hi and lo <= 5


# ---- assign.run: the wiring ---------------------------------------------------------------------------
_ACIDS = [f"C{n}H{2 * n - 2}O4" for n in range(5, 25)]


def _acid_peaks():
    from mascope_tools.composition.heuristic_filter import anchor_on_monoisotopic, predict_isotopes
    from peaky.chem import chemistry as C
    rows = []
    for k, nf in enumerate(_ACIDS):
        cnt = dict(C.parse_formula(nf))
        cnt["H"] -= 1
        ion = C.format_formula({e: v for e, v in cnt.items() if v > 0})
        mzs, ints, _ = anchor_on_monoisotopic(*predict_isotopes(ion, -1))
        for i, (m, h) in enumerate(zip(mzs, ints)):
            height = h / ints[0] * 2e5
            if height >= 300:
                rows.append({"peak_id": f"p{k}_{i}", "mz": float(m), "height": height, "area": height * 1e-3,
                             "signal_to_noise": height / 30.0})
    return pd.DataFrame(rows).sort_values("mz").reset_index(drop=True)


def test_the_run_fits_by_area_re_runs_on_a_material_response_and_a_stand_in_inherits_it(monkeypatch):
    from mascope_tools.composition import PatternScoring
    from peaky.assignment import assign as A
    from peaky.io import io_mascope as IO
    resp = {"x": [0.5, 2.0], "y": [0.5, 0.8], "n": 30, "col": "area"}
    fits, audits = [], []
    monkeypatch.setattr(IR, "fit", lambda peaks, ions, col="area", **k: fits.append(col) or resp)
    real_audit = P.audit_isotopes
    monkeypatch.setattr(P, "audit_isotopes", lambda led, cfg, **k: (
        audits.append((cfg.iso_response_fit, cfg.iso_area)), real_audit(led, cfg, **k))[1])

    def run(sid, scoring):
        log = []
        cfg = P.PassConfig(height_cutoff_cps=100)
        cfg.instrument_class = "orbitrap"
        res = A.run(sid, context="ambient-air", cfg=cfg, peaks=_acid_peaks(), adducts=["[M-H]-"],
                    do_pass2=False, do_pass3=False, do_pass4=False, do_pass5=False, do_pass_certified=False,
                    use_cache=False, log=log.append, scoring=scoring)
        return res, log
    try:
        res, log = run("iso-run", PatternScoring(sigma_ppm=0.58, mu_ppm=0.0, mz_tolerance_ppm=5.0))
        assert fits == ["area"]
        assert any("re-running from pass 0 against it" in str(m) for m in log), log
        assert audits and audits[-1] == (resp, True)
        rec = res["stats"]["iso_response"]
        assert rec["mode"] == "auto" and rec["fit"] == resp and rec["measured"] == resp and rec["area"]
        snap = res["pattern_scoring"]
        assert snap["iso_scoring"] == {"col": "area", "resp": resp}
        # a decoy arm judged at the run's snapshot: read as the sample was, from pass 0, and no fit of its own
        fits.clear(), audits.clear()
        res2, log2 = run("iso-run-arm", dict(snap))
        assert fits == []
        assert not any("re-running from pass 0 against it" in str(m) for m in log2)
        assert audits and all(a == (resp, True) for a in audits)
        assert res2["stats"]["iso_response"]["mode"] == "inherited"
    finally:
        IO.unregister_offline_sample("iso-run")
        IO.unregister_offline_sample("iso-run-arm")


def test_a_response_that_is_not_material_is_recorded_and_changes_nothing(monkeypatch):
    from mascope_tools.composition import PatternScoring
    from peaky.assignment import assign as A
    from peaky.io import io_mascope as IO
    resp = {"x": [0.5, 2.0], "y": [0.95, 1.0], "n": 30, "col": "area"}
    monkeypatch.setattr(IR, "fit", lambda peaks, ions, col="area", **k: resp)
    log = []
    cfg = P.PassConfig(height_cutoff_cps=100)
    cfg.instrument_class = "orbitrap"
    try:
        res = A.run("iso-flat", context="ambient-air", cfg=cfg, peaks=_acid_peaks(), adducts=["[M-H]-"],
                    do_pass2=False, do_pass3=False, do_pass4=False, do_pass5=False, do_pass_certified=False,
                    use_cache=False, log=log.append,
                    scoring=PatternScoring(sigma_ppm=0.58, mu_ppm=0.0, mz_tolerance_ppm=5.0))
    finally:
        IO.unregister_offline_sample("iso-flat")
    rec = res["stats"]["iso_response"]
    assert rec["fit"] is None and rec["measured"] == resp and "not material" in rec["note"]
    assert not any("re-running from pass 0 against it" in str(m) for m in log)


# ---- surviving mutants of round 2 ---------------------------------------------------------------------
def test_an_m0_without_an_area_is_read_by_height_without_the_response():
    peaks = _envelope_peaks(0.7)
    resp = {"x": [0.0, 4.0], "y": [0.7, 0.7], "lo": [0.7, 0.7], "n": 20, "col": "area"}
    peaks.loc[0, "area"] = np.nan
    assert _score(peaks, intensity_col="area", iso_response=resp) == pytest.approx(_score(peaks))


def test_an_absent_line_is_judged_visible_at_the_band_s_lower_end():
    # a C10 at M0 S/N 50: its 13C would be S/N 5.4 at its natural share (visible: absent is charged), S/N 1.6
    # at the response's censored share (not visible: absent is what the file predicts)
    peaks = _envelope_peaks(1.0)
    peaks = peaks[~peaks.peak_id.isin(peaks.peak_id.iloc[1:])].copy()
    peaks["signal_to_noise"] = 50.0
    resp = {"x": [0.0, 1.0], "y": [0.3, 0.3], "lo": [0.3, 0.3], "n": 20, "col": "area"}
    assert _score(peaks, intensity_col="area", iso_response=resp) > _score(peaks, intensity_col="area")


def test_the_clamp_reads_the_response_at_the_claimed_count_s_line():
    resp = {"x": [0.5, 1.5], "y": [0.4, 1.0], "n": 20, "col": "area"}
    snr0 = 30.0
    c = IR.carbon_from_13c(resp, h0=1e4, h_sat=1e4 * 6 * R, a0=10.0, a_sat=10.0 * 6 * R, snr0=snr0, n_c=10)
    x10 = math.log10(snr0 * 10 * R)
    assert c == pytest.approx(6.0 / IR.scale(resp, x10))
    assert c != pytest.approx(6.0 / IR.scale(resp, math.log10(snr0 * 6 * R)))


def test_a_line_of_unknown_s_n_reads_at_the_brightest_point():
    resp = {"x": [0.5, 1.5], "y": [0.4, 0.9], "lo": [0.3, 0.8], "n": 20, "col": "area"}
    assert IR.scale(resp, None) == 0.9 and IR.band_scale(resp, None) == 0.8
    assert IR.scale(resp, float("nan")) == 0.9


def test_the_pava_pools_by_weight():
    assert IR._pava([0.8, 0.4], [10.0, 1.0]) == pytest.approx([(8.0 + 0.4) / 11] * 2)


def test_a_bin_reads_its_median_not_its_mean():
    rows, ions = [], []
    for k in range(15):
        f = 0.5 if k < 12 else 2.9
        h0 = 300.0 * 50.0
        rows += [dict(peak_id=f"m{k}", mz=150.0 + 7.3 * k, height=h0, area=h0, signal_to_noise=300.0),
                 dict(peak_id=f"c{k}", mz=150.0 + 7.3 * k + IR.D13C, height=h0 * 10 * R * f, area=h0 * 10 * R * f,
                      signal_to_noise=30.0)]
        ions.append((150.0 + 7.3 * k, "C10H10O4-"))
    resp = IR.fit(pd.DataFrame(rows), ions)
    assert resp["y"] == [pytest.approx(0.5)]


def test_15n_labelled_ions_are_fitted_on():
    peaks, ions = _peaks_with_response(lambda x: 0.6)
    assert IR._carbons("C10H16^NO10-") == 10
    assert IR.fit(peaks, [(m, f.replace("O4", "^NO7")) for m, f in ions]) is not None


def test_only_committed_m0_rows_are_fitted_on():
    from peaky.assignment.assign import _committed_ions
    led = _ledger([dict(peak_id="a", mz=300.0, height=1e4, area=10.0, signal_to_noise=500.0),
                   dict(peak_id="b", mz=301.00335, height=1e3, area=1.0, signal_to_noise=50.0)])
    L.commit_assignment(led, "a", neutral_formula="C10H16O7", adduct="[M+NO3]-", ion_formula="C10H16NO10-",
                        ion_score=0.9, ppm_error=0.1, pass_no=1, method="test", confidence="Good", commentary="t")
    L.attach_isotopologue(led, "b", "a", iso_label="13C")
    led.loc[led.peak_id == "b", "ion_formula"] = "C10H16NO10-"      # an isotopologue row naming its parent's ion
    ions = _committed_ions(led)
    assert len(ions) == 1 and ions[0][1] == "C10H16NO10-"


def test_the_residual_bracket_reads_the_response_at_the_line_s_s_n_and_stays_bounded():
    resp = {"x": [0.5, 1.5], "y": [0.4, 1.0], "n": 20, "col": "area"}

    def bracket(snr):
        led = _ledger([dict(peak_id="a", mz=300.0, height=1e4, area=10.0, signal_to_noise=snr),
                       dict(peak_id="b", mz=301.00335, height=1e4 * 6 * R, area=10.0 * 6 * R, signal_to_noise=9.0)])
        return carbon_count_from_13c(led, "a", resp=resp)
    dim, bright = bracket(30.0), bracket(3000.0)
    assert dim[1] > bright[1]                       # the dim line is corrected up, the bright one is not
    assert bright[1] <= 8 and dim[1] <= 20           # and never to an unbounded count


def _clamp_case(fn, c_height, c_area, *, snr, resp, kid=False, iso_area=True):
    led = _ledger([dict(peak_id="M", mz=310.078, height=2e4, area=20.0, signal_to_noise=snr),
                   dict(peak_id="S", mz=311.0813, height=2e4 * c_height * IR.R13C_RAW, area=20.0 * c_area * R,
                        signal_to_noise=snr * c_height * R)])
    L.commit_assignment(led, "M", neutral_formula="C10H16O7", adduct="[M+NO3]-", ion_formula="C10H16NO10-",
                        ion_score=0.9, ppm_error=0.1, pass_no=1, method="test", confidence="Good", commentary="t")
    if kid:
        L.attach_isotopologue(led, "S", "M", iso_label="13C")
    cfg = P.PassConfig(height_cutoff_cps=10.0)
    cfg.iso_response_fit, cfg.iso_area = resp, iso_area
    fn(led, cfg, log=lambda *a: None)
    return L.role_of(led, "M")


def test_the_clamps_read_the_response_at_the_line_s_own_s_n():
    # the response is 0.5 at the dim end and 1.0 bright: a C10 read C5 is kept dim, cleared bright
    resp = {"x": [0.5, 1.6], "y": [0.5, 1.0], "n": 20, "col": "area"}
    for fn in (P.demote_carbon_inconsistent, P.audit_isotopes):
        assert _clamp_case(fn, 5, 5, snr=29.0, resp=resp) == L.ROLE_M0, fn.__name__        # line S/N ~3
        assert _clamp_case(fn, 5, 5, snr=3000.0, resp=resp) != L.ROLE_M0, fn.__name__      # line S/N ~300


def test_the_pre_pass_4_clamp_reads_a_committed_13c_child_s_area():
    # heights read C6, the area C10: a committed 13C child in area mode keeps the C10
    assert _clamp_case(P.demote_carbon_inconsistent, 6, 10, snr=800.0, resp=None, kid=True) == L.ROLE_M0
    assert _clamp_case(P.demote_carbon_inconsistent, 6, 10, snr=800.0, resp=None, kid=True,
                       iso_area=False) != L.ROLE_M0


def test_re_registering_a_stand_in_forgets_its_inherited_isotope_scoring():
    from peaky.io import io_mascope as IO
    peaks = _envelope_peaks(1.0)
    IO.register_offline_sample("iso-re", peaks, ["[M+NO3]-"], scoring=_snap(iso_scoring={"col": "area", "resp": None}))
    try:
        IO.scoring_for_sample(None, "iso-re", peaks)
        assert IO.iso_scoring_inherited("iso-re")
        IO.register_offline_sample("iso-re", peaks, ["[M+NO3]-"], scoring=_ORBI)
        assert IO.iso_scoring("iso-re") is None and not IO.iso_scoring_inherited("iso-re")
    finally:
        IO.unregister_offline_sample("iso-re")


def test_a_stand_in_of_a_snapshot_without_an_isotope_record_is_read_by_height():
    from peaky.io import io_mascope as IO
    peaks = _envelope_peaks(1.0)
    IO.register_offline_sample("iso-old", peaks, ["[M+NO3]-"], scoring=_snap())
    try:
        IO.scoring_for_sample(None, "iso-old", peaks)
        assert IO.iso_scoring("iso-old") == {"col": "height", "resp": None}
        assert IO.iso_scoring_inherited("iso-old")
    finally:
        IO.unregister_offline_sample("iso-old")


@pytest.mark.parametrize("knob, col", [("area", "area"), ("off", "height")])
def test_the_area_and_off_modes_fit_nothing_and_record_what_they_read(monkeypatch, knob, col):
    from mascope_tools.composition import PatternScoring
    from peaky.assignment import assign as A
    from peaky.io import io_mascope as IO
    from peaky.io import local_scoring as LS
    fits, cols = [], []
    monkeypatch.setattr(IR, "fit", lambda *a, **k: fits.append(1) or None)
    real = LS.score_candidates_local
    monkeypatch.setattr(LS, "score_candidates_local",
                        lambda *a, **k: cols.append(k.get("intensity_col")) or real(*a, **k))

    def run(sid, scoring, knob):
        cfg = P.PassConfig(height_cutoff_cps=100, iso_response=knob)
        cfg.instrument_class = "orbitrap"
        return A.run(sid, context="ambient-air", cfg=cfg, peaks=_acid_peaks(), adducts=["[M-H]-"],
                     do_pass2=False, do_pass3=False, do_pass4=False, do_pass5=False, do_pass_certified=False,
                     use_cache=False, log=lambda *a: None, scoring=scoring)
    try:
        res = run(f"iso-{knob}", PatternScoring(sigma_ppm=0.58, mu_ppm=0.0, mz_tolerance_ppm=5.0), knob)
        assert fits == [] and cols and set(cols) == {col}           # from pass 0, every call
        assert res["stats"]["iso_response"]["mode"] == knob and res["stats"]["iso_response"]["fit"] is None
        assert res["pattern_scoring"]["iso_scoring"] == {"col": col, "resp": None}
        cols.clear()
        arm = run(f"iso-{knob}-arm", dict(res["pattern_scoring"]), "auto")   # a decoy arm's default config
        assert fits == [] and set(cols) == {col}
        assert arm["stats"]["iso_response"]["mode"] == "inherited"
    finally:
        IO.unregister_offline_sample(f"iso-{knob}")
        IO.unregister_offline_sample(f"iso-{knob}-arm")


def test_the_band_s_lower_end_is_the_bin_s_lower_quantile():
    ys = np.linspace(0.4, 0.9, 15)
    rows, ions = [], []
    for k, f in enumerate(ys):
        h0 = 300.0 * 50.0
        rows += [dict(peak_id=f"m{k}", mz=150.0 + 7.3 * k, height=h0, area=h0, signal_to_noise=300.0),
                 dict(peak_id=f"c{k}", mz=150.0 + 7.3 * k + IR.D13C, height=h0 * 10 * R * f, area=h0 * 10 * R * f,
                      signal_to_noise=30.0)]
        ions.append((150.0 + 7.3 * k, "C10H10O4-"))
    resp = IR.fit(pd.DataFrame(rows), ions)
    assert resp["lo"] == [pytest.approx(np.quantile(ys, 0.2), abs=1e-3)]
    assert resp["y"] == [pytest.approx(np.median(ys), abs=1e-3)]


def test_a_minor_line_without_an_area_is_read_by_its_height_ratio():
    peaks = _envelope_peaks(1.0)
    by_h = _score(peaks)
    peaks.loc[1, "area"] = np.nan                                    # the 13C line
    assert _score(peaks, intensity_col="area") == pytest.approx(by_h)


def test_the_iso_response_switch_reaches_the_run_config():
    from peaky import cli
    ap = cli.build_parser()
    a = ap.parse_args(["batch", "--batch", "b", "--iso-response", "off"])
    assert cli._sidelobe_cfg(a)["cfg"].iso_response == "off"
    assert cli._sidelobe_cfg(ap.parse_args(["batch", "--batch", "b"])) == {}
    with pytest.raises(SystemExit):
        ap.parse_args(["batch", "--batch", "b", "--iso-response", "maybe"])
