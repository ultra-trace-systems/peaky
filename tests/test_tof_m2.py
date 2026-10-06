"""The TOF ion-M+2 test: one primitive (satellites.heavy_line_verdict) asks
whether the ION's own Br / Cl M+2 line is where its composition puts it -- the
reagent adduct's halogen included -- and the tier pass reads it
(tiers.apply_tof_m2, the assign stage `tof_m2`): an Assigned per-file M0 whose
line the file could show and does not is Candidate.

Synthetic, offline; every threshold tested at its edge.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from peaky.assignment import assign as A
from peaky.assignment import evidence as EV
from peaky.assignment import ledger as L
from peaky.assignment import satellites as SAT
from peaky.assignment import tiers as T
from peaky.assignment.passes.config import PassConfig
from peaky.chem import chemistry as C
from peaky.chem import isotopes as ISO
from peaky.chem.resolution import Resolution
from tests.test_iso_checks import _ion

BR = "[M+Br]-"
X = "C6H10O3"                       # a CHO neutral: on [M+Br]- its ion carries the reagent's one Br
RP = Resolution.from_r(10_000)
TOF_CFG = dict(instrument_type="tof", noise_edge_batch_cps=1.0)      # floor 3 cps


def quiet(*a, **k):
    pass


# --------------------------------------------------------------------------- the primitive
def test_the_prediction_is_the_ions_whole_m_plus_2_cluster_reagent_halogen_included():
    ion = T._ion_counts(X, BR)
    ratio, shift, label = SAT.heavy_line_prediction(ion)
    want = ISO.nominal_cluster(ion, 2)
    assert (ratio, shift) == pytest.approx(want) and label == "81Br"
    # the cluster adds 18O / 13C2 to the one 81Br: a little above the bare per-atom ratio
    assert ISO.R_81BR_PER_BR < ratio < ISO.R_81BR_PER_BR + 0.02
    assert shift == pytest.approx(ISO.D_81BR, abs=5e-4)
    r2, _s, lab2 = SAT.heavy_line_prediction(T._ion_counts("C30H58Cl4", BR))
    assert lab2 == "81Br/37Cl" and r2 == pytest.approx(ISO.R_81BR_PER_BR + 4 * ISO.R_37CL_PER_CL, abs=0.06)
    assert SAT.heavy_line_prediction(T._ion_counts("C6H10O3S", "[M+NO3]-"))[0] == 0.0     # 34S alone: no test
    assert SAT.heavy_line_prediction(T._ion_counts("C2H3ClO2", "[M+NO3]-"))[2] == "37Cl"


def test_the_window_is_the_scorers_or_three_fitted_sigmas():
    assert SAT.heavy_line_window_ppm() == SAT.HEAVY_LINE_MIN_PPM == 15.0
    assert SAT.heavy_line_window_ppm(4.0) == 15.0
    assert SAT.heavy_line_window_ppm(8.75) == pytest.approx(26.25)
    assert SAT.heavy_line_window_ppm(4.0, tol_ppm=20.0) == 20.0
    assert SAT.heavy_line_window_ppm(float("nan"), tol_ppm=None) == 15.0


def _verdict(lines, h0=100.0, floor=3.0, ratio=0.98, win=15.0, m0=300.0):
    mz = np.array([m0] + [m0 + ISO.D_81BR + d for d, _h in lines])
    h = np.array([h0] + [hh for _d, hh in lines])
    v = SAT.heavy_line_verdict(mz, h, m0, h0, ratio, ISO.D_81BR, floor, win_ppm=win, fwhm=m0 / 10_000)
    return v["status"][0]


def test_seen_absent_dim_and_none():
    fw = 300.0 / 10_000                                          # 0.03 Da = 100 ppm at m/z 300
    assert _verdict([(0.0, 97.0)]) == SAT.HL_SEEN
    assert _verdict([(14e-6 * 302, 60.0)]) == SAT.HL_SEEN       # 14 ppm, 0.61x the prediction
    assert _verdict([]) == SAT.HL_ABSENT
    assert _verdict([(0.0, 50.0)]) == SAT.HL_ABSENT             # 0.51x: under 0.6x, and alone in its reach
    assert _verdict([(0.75 * fw, 90.0)]) == SAT.HL_ABSENT       # outside the split reach, under pred in 1 FWHM
    assert _verdict([], h0=3.0) == SAT.HL_DIM                   # predicted 2.9 cps under the 3-cps floor
    v = SAT.heavy_line_verdict([300.0], [100.0], 300.0, 100.0, 0.0, float("nan"), 3.0, win_ppm=15, fwhm=0.03)
    assert v["status"][0] == SAT.HL_NONE
    v = SAT.heavy_line_verdict([300.0], [100.0], 300.0, 100.0, 0.98, ISO.D_81BR, None, win_ppm=15, fwhm=0.03)
    assert v["status"][0] == SAT.HL_DIM                          # no floor: nothing is testable


def test_the_two_blend_guards_make_a_held_or_split_position_untestable():
    fw = 300.0 / 10_000
    # another ion's line within one FWHM at >= the prediction holds the position
    assert _verdict([(0.9 * fw, 99.0)]) == SAT.HL_BLENDED
    assert _verdict([(0.9 * fw, 97.0)]) == SAT.HL_ABSENT        # under the prediction: no hold
    # an unresolved split: two sub-peaks within half a FWHM summing to >= 0.6x
    assert _verdict([(-0.3 * fw, 31.0), (0.3 * fw, 31.0)]) == SAT.HL_BLENDED
    assert _verdict([(-0.3 * fw, 28.0), (0.3 * fw, 28.0)]) == SAT.HL_ABSENT      # 56 < 0.6 x 98


# --------------------------------------------------------------------------- the tier pass
def _tof_ledger(*, m2=None, h0=100.0, neutral=X, adduct=BR, method="cheminfo+grid", tier="Assigned",
                extra=(), share=None):
    mz0 = C.ion_mz(neutral, adduct)
    rows = [("P", mz0, h0)] + ([("Q", mz0 + ISO.D_81BR + m2[0], m2[1])] if m2 else []) + list(extra)
    rows += [(f"f{k}", 100.0 + 3.1 * k, 2.0) for k in range(20)]
    led = L.new_ledger(pd.DataFrame(rows, columns=["peak_id", "mz", "height"]))
    L.commit_assignment(led, "P", neutral_formula=neutral, adduct=adduct, ion_formula=_ion(neutral, adduct),
                        ion_score=0.95, compound_score=0.95, ppm_error=0.1, pass_no=1, method=method,
                        confidence="High", commentary="stub")
    i = led.index[led["peak_id"] == "P"][0]
    led["tier"] = led["tier"].astype(object)
    led["tier_reason"] = led["tier_reason"].astype(object)
    led.at[i, "tier"], led.at[i, "tier_reason"] = tier, "unique formula in the calibrated window"
    if share is not None:
        led.at[i, "assigned_fraction"] = share
    return led, i


def test_an_absent_ion_m_plus_2_line_makes_the_tof_row_candidate_and_says_which_line():
    led, i = _tof_ledger()
    out = T.apply_tof_m2(led, cfg=PassConfig(**TOF_CFG), resolving_power=RP, log=quiet)
    assert out["tested"] == 1 and out["absent"] == 1 and out["demoted"] == 1 and out["skipped"] is None
    assert led.at[i, "tier"] == T.TIER_CANDIDATE
    why = led.at[i, "tier_reason"]
    assert why.startswith("ion M+2 line absent (TOF): C6H10BrO3- predicts its 81Br line at 0.98x")
    assert "(98 cps, over the 3-cps floor)" in why and "no line within 15 ppm" in why
    assert why.endswith("(otherwise Assigned: unique formula in the calibrated window)")


def test_the_line_present_dim_or_held_leaves_the_tier():
    held = 0.9 * C.ion_mz(X, BR) / 10_000                                 # 0.9 FWHM out, 1.2x: another line
    for m2, h0 in (((0.0, 97.0), 100.0), (None, 2.5), ((held, 120.0), 100.0)):
        led, i = _tof_ledger(m2=m2, h0=h0)
        T.apply_tof_m2(led, cfg=PassConfig(**TOF_CFG), resolving_power=RP, log=quiet)
        assert led.at[i, "tier"] == T.TIER_ASSIGNED, (m2, h0)
    led, i = _tof_ledger(m2=(0.0, 40.0))                                   # present at 0.4x: refuted
    T.apply_tof_m2(led, cfg=PassConfig(**TOF_CFG), resolving_power=RP, log=quiet)
    assert led.at[i, "tier"] == T.TIER_CANDIDATE and "the tallest line within 15 ppm is 0.41x it" in \
        led.at[i, "tier_reason"]


def test_the_window_widens_with_the_files_fitted_sigma():
    off = 20e-6 * C.ion_mz(X, BR)                                          # the line 20 ppm out, 0.97x
    for sigma, tier in ((4.0, T.TIER_ASSIGNED), (8.0, T.TIER_ASSIGNED)):
        led, i = _tof_ledger(m2=(off, 97.0))
        T.apply_tof_m2(led, cfg=PassConfig(**TOF_CFG), resolving_power=RP, scoring={"sigma_ppm": sigma}, log=quiet)
        assert led.at[i, "tier"] == tier
    # 15 ppm: not seen, but inside half a FWHM -> blended (untestable); 24 ppm: seen
    v4 = T.tof_m2_verdicts(_tof_ledger(m2=(off, 97.0))[0], cfg=PassConfig(**TOF_CFG), resolving_power=RP,
                           scoring={"sigma_ppm": 4.0})
    v8 = T.tof_m2_verdicts(_tof_ledger(m2=(off, 97.0))[0], cfg=PassConfig(**TOF_CFG), resolving_power=RP,
                           scoring={"sigma_ppm": 8.0, "mz_tolerance_ppm": 15.0})
    assert v4["status"].tolist() == [SAT.HL_BLENDED] and v4["window_ppm"].iloc[0] == 15.0
    assert v8["status"].tolist() == [SAT.HL_SEEN] and v8["window_ppm"].iloc[0] == 24.0


def test_off_a_tof_without_an_edge_or_a_width_model_nothing_moves():
    for cfg, rp, why in ((PassConfig(instrument_type="orbi", noise_edge_batch_cps=1.0), RP, "not a TOF"),
                         (PassConfig(instrument_type="tof"), RP, "not a TOF"),
                         (None, RP, "not a TOF"),
                         (PassConfig(**TOF_CFG), None, "no width model")):
        led, i = _tof_ledger()
        out = T.apply_tof_m2(led, cfg=cfg, resolving_power=rp, log=quiet)
        assert out["skipped"] == why and led.at[i, "tier"] == T.TIER_ASSIGNED


def test_known_species_rows_are_tested_and_candidate_rows_keep_their_reason():
    led, i = _tof_ledger(neutral="C30H58Cl4", method="known:chlorinated_paraffin",
                         m2=(0.0, 43.0))                      # the line one Br makes, not BrCl4's 2.3x
    T.apply_tof_m2(led, cfg=PassConfig(**TOF_CFG), resolving_power=RP, log=quiet)
    assert led.at[i, "tier"] == T.TIER_CANDIDATE and "81Br/37Cl line at 2.3" in led.at[i, "tier_reason"]
    led, i = _tof_ledger(tier="Candidate")
    T.apply_tof_m2(led, cfg=PassConfig(**TOF_CFG), resolving_power=RP, log=quiet)
    assert led.at[i, "tier"] == "Candidate" and led.at[i, "tier_reason"] == "unique formula in the calibrated window"


def test_a_composite_parent_predicts_from_its_own_share_and_a_heavy_commit_is_not_tested():
    led, i = _tof_ledger(m2=(0.0, 50.0), share=0.5)           # 50 cps is 1.0x of half the 100-cps peak
    T.apply_tof_m2(led, cfg=PassConfig(**TOF_CFG), resolving_power=RP, log=quiet)
    assert led.at[i, "tier"] == T.TIER_ASSIGNED
    led, i = _tof_ledger()
    led.at[i, "mz"] = float(led.at[i, "mz"]) + ISO.D_81BR       # committed on the 81Br line itself
    assert T.tof_m2_verdicts(led, cfg=PassConfig(**TOF_CFG), resolving_power=RP).empty


def test_the_demotion_moves_the_tier_only_never_the_vote_class():
    led, i = _tof_ledger()
    L.ensure_flags(led)
    before = EV.vote_classes(EV.trim(led))
    flags = led[list(L.ASSIGNABILITY_FLAGS)].copy()
    T.apply_tof_m2(led, cfg=PassConfig(**TOF_CFG), resolving_power=RP, log=quiet)
    assert led.at[i, "tier"] == T.TIER_CANDIDATE
    assert EV.vote_classes(EV.trim(led)).equals(before)
    assert led[list(L.ASSIGNABILITY_FLAGS)].equals(flags)


def test_the_stage_is_the_last_tier_word_before_the_evidence_level():
    names = [s.name for s in A._STAGES]
    k = names.index("tof_m2")
    assert names.index("tiers") < names.index("demote_speculative") < names.index("reflist_rescue") < k
    assert names[k - 1] == "iso_env_final" and names[k + 1] == "evidence"
    assert not A._STAGES[k].safe


# --------------------------------------------------------------------------- assign.run wiring
SID = "offline-tof-m2"


def _tof_table(n_filler: int = 60, seed: int = 3) -> pd.DataFrame:
    """A bromide-CIMS-like TOF peak list: every line of the [M+Br]- cluster of C10H16O6 and dim fillers; the
    table's signal-to-noise has nothing to do with height (as on a real TOF)."""
    from mascope_tools.composition.heuristic_filter import anchor_on_monoisotopic, predict_isotopes
    rng = np.random.default_rng(seed)
    mzs, ints, _labels = anchor_on_monoisotopic(*predict_isotopes("C10H16O6Br", -1))
    rel = ints / ints[0]
    rows = [{"peak_id": f"L{i}", "mz": float(m), "height": 500.0 * float(r)}
            for i, (m, r) in enumerate(zip(mzs, rel)) if r >= 0.01]
    rows += [{"peak_id": f"f{j}", "mz": 60.0 + j * 5.37, "height": float(rng.uniform(0.6, 6.0))}
             for j in range(n_filler)]
    t = pd.DataFrame(rows)
    t["signal_to_noise"] = rng.uniform(0.5, 2.0, len(t))
    return t


def test_assign_run_records_the_tests_counts_on_a_tof_and_skips_an_orbitrap():
    from peaky.io import io_mascope as IO
    IO.unregister_offline_sample(SID)
    try:
        table = _tof_table()
        res = A.run(SID, context="ambient-air", cfg=PassConfig(noise_edge_batch_cps=0.74), peaks=table,
                    use_cache=False, scoring="tof", adducts=["[M-H]-", "[M+Br]-", "[M+NO3]-"],
                    reagent_n_relabel=False, resolving_power=Resolution.from_r(10_000), log=quiet)
        st = res["stats"]["tof_m2"]
        assert st["skipped"] is None and st["tested"] >= 1
        led = res["ledger"]
        bright = led[(led["role"] == "M0") & (led["neutral_formula"] == "C10H16O6") & (led["adduct"] == BR)]
        assert len(bright) and (bright["tier"] == "Assigned").all()       # its own 81Br line is in the table
        IO.unregister_offline_sample(SID)
        table["signal_to_noise"] = table["height"] / 20.0
        res = A.run(SID, context="ambient-air", cfg=PassConfig(noise_edge_batch_cps=0.74), peaks=table,
                    use_cache=False, scoring="orbi", adducts=["[M-H]-", "[M+Br]-", "[M+NO3]-"],
                    reagent_n_relabel=False, resolving_power=Resolution.from_r(10_000), log=quiet)
        assert res["stats"]["tof_m2"]["skipped"] == "not a TOF"
    finally:
        IO.unregister_offline_sample(SID)
