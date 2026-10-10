"""The NOx-skeleton Van Krevelen readings and the C3-C4 small-acid band
(chem/contexts.py: vk_readings, run_profile), their run-level gating, the twin
tie-break in grid arbitration and the evidence level's relaxed space.
Synthetic formulas and frames only."""
from __future__ import annotations

import dataclasses
import itertools

import pandas as pd

from peaky.assignment import assign as A
from peaky.assignment import degeneracy as DG
from peaky.assignment import evidence as EV
from peaky.assignment.levels import context as CX
from peaky.assignment.levels import space as SP
from peaky.assignment.passes import core as PC
from peaky.assignment.passes.config import PassConfig
from peaky.chem import chemistry as C
from peaky.chem import contexts as X
from peaky.chem import profiles as P

AMB = X.get_context("ambient-air")
RUN = X.run_profile(AMB, reagent="NO3", instrument_class="orbitrap")


def keep(f, prof=RUN):
    return X.filter_by_profile(f, prof)[0]


# --- the switch: run level, nitrate reagents on Orbitrap-class axes only ----------------------------------
def test_switch_is_off_on_every_builtin_context():
    for name in ("ambient-air", "chamber", "combustion", "water", "uronium", "indoor-air", "easyic"):
        p = X.get_context(name)
        assert not p.nox_skeleton and not p.small_acid_band


def test_run_profile_gating():
    assert RUN.nox_skeleton and RUN.small_acid_band and RUN.label == "ambient-air"
    assert X.run_profile(AMB, reagent="NO3_15N", instrument_class="orbitrap").nox_skeleton
    for kw in (dict(reagent="NO3", instrument_class="tof"),
               dict(reagent="NO3", instrument_class=None),
               dict(reagent="NO3", instrument_class="orbitrap", trace_sample=True),
               dict(reagent="Br", instrument_class="orbitrap"),
               dict(reagent="I", instrument_class="orbitrap"),
               dict(reagent=None, instrument_class="orbitrap")):
        assert X.run_profile(AMB, **kw) is AMB, kw
    # a context that does not open the organonitrate family stays as it is
    ur = X.get_context("uronium")
    assert X.run_profile(ur, reagent="NO3", instrument_class="orbitrap") is ur
    assert X.profile_flags(RUN) == {"nox_skeleton": True, "small_acid_band": True}
    assert X.profile_flags(AMB) == {}
    assert X.as_profile("ambient-air", X.profile_flags(RUN)) == RUN


def test_reagent_name_resolves_aliases_and_channels():
    assert A._reagent_name("nitrate", None) == "NO3"
    assert A._reagent_name(None, ["[M+NO3]-", "[M-H]-"]) == "NO3"
    assert A._reagent_name(P.PROFILES["NO3_15N"], None) == "NO3_15N"
    assert A._reagent_name(None, None) is None


# --- the filter --------------------------------------------------------------------------------------------
def test_dinitrate_admitted_only_with_the_skeleton_reading():
    """Isoprene dihydroxy dinitrate C5H10N2O8 (raw O/C 1.60) reads as C5H12O4 (O/C 0.80)."""
    assert X.filter_by_profile("C5H10N2O8", AMB) == (False, "O/C=1.60 out of (0.0, 1.5)")
    assert X.filter_by_profile("C5H10N2O8", RUN) == (True, None)
    for f in ("C3H5N3O9",       # nitroglycerin: glycerol skeleton
              "C6H5NO4",        # 4-nitrocatechol: catechol
              "C7H5NO4",        # nitrobenzoic acid (the element_budget docstring's false reject)
              "C6H3N3O7",       # picric acid: phenol
              "C7H6N2O4"):      # 2,4-dinitrotoluene: toluene (two NITRO groups, no O left)
        assert not keep(f, AMB) and keep(f), f
    # the reason string of a failure is the RAW reading's, unchanged in shape
    ok, why = X.filter_by_profile("C5H6O8", RUN)
    assert not ok and why == "O/C=1.60 out of (0.0, 1.5)"
    assert DG.filter_kind(why) == DG.filter_kind(X.filter_by_profile("C5H6O8", AMB)[1])


def test_k_is_capped_at_three_and_at_the_carbon_count():
    """A tetranitrate's k=4 skeleton is never read: under a profile with N cap 5 and O/C <= 1.2,
    erythritol tetranitrate C4H6N4O12 would pass as erythritol (O/C 1.0) but its k=3 reading
    (C4H9NO6, O/C 1.5) fails. A C3 cannot carry four groups (k <= Ceff)."""
    prof = dataclasses.replace(RUN, max_N=5, o_to_c=(0.0, 1.2))
    sk = X.nox_skeletons(C.parse_formula("C4H6N4O12"), prof)
    assert max(k for k, _ in sk) == X.NOX_K_MAX == 3
    assert not keep("C4H6N4O12", prof)
    assert max(k for k, _ in X.nox_skeletons(C.parse_formula("C3H4N4O12"), prof)) == 3
    assert not keep("C3H4N4O12", prof)


def test_nitrate_stoichiometry_by_default_nitro_only_on_aromatic_skeletons():
    """k groups need O >= 3k (each nitrate leaves its bridging O on the skeleton) unless the
    skeleton is aromatic (DBE/C >= 0.5), where R-NO2 (2 O per N) is read too."""
    ks = lambda f: {k for k, _ in X.nox_skeletons(C.parse_formula(f), RUN)}  # noqa: E731
    assert ks("C7H6N2O4") == {2, 1}          # dinitrotoluene: k=2 skeleton toluene, DBE/C 0.57
    assert ks("C4H6N2O4") == {1}             # aliphatic: k=2 would leave O0 on a DBE/C 0.25 skeleton
    assert ks("C4H11NO4") == set()           # DBE 0: no N=O, no group
    assert not keep("C3H9NO2")               # ... so it is judged raw (H/C 3.0)
    assert ks("C3N2O6") == {2, 1}            # skeletons exist, but no reading of it passes
    assert not keep("C3N2O6") and not keep("C4N2O12")


def test_amine_n_keeps_the_n_window():
    """N not in a NOx group is not discounted: C6H4N4O (O < 2N) stays out by N/C (here also the
    N cap); C5H5N3O (N/C 0.6, one O) has no nitrate or aromatic-nitro reading that clears it."""
    assert not keep("C6H4N4O")
    assert not keep("C5H5N3O")


def test_the_change_only_loosens():
    """Every formula admitted without the switches is admitted with them (small CHNO grid)."""
    for c, h, n, o in itertools.product(range(1, 9), range(0, 19), range(0, 4), range(0, 13)):
        f = C.format_formula({k: v for k, v in dict(C=c, H=h, N=n, O=o).items() if v})
        if not C.dbe_ok(f)[0]:
            continue
        if keep(f, AMB):
            assert keep(f, RUN), f


def test_small_acid_band():
    for f in ("C4H2O4", "C3H2O4", "C3H4O5", "C3H2O5", "C4H2O3", "C3H6O5S"):
        assert not keep(f, AMB) and keep(f), f
    for f in ("C4H11NO7",    # N in the neutral: not a polycarbonyl acid (and DBE 0)
              "C3H8O6",      # DBE 0 at O/C 2: all gem-diols, no C=O
              "C4H2", "C3H2",  # O < 2
              "C5H6O8",      # C5: outside the band
              "C3N2O6", "C4N2O12"):   # H-free; the band reads the raw neutral only
        assert not keep(f), f
    # the band alone: a profile with only small_acid_band set
    band = dataclasses.replace(AMB, small_acid_band=True)
    assert keep("C4H2O4", band) and not keep("C5H10N2O8", band)


# --- every consumer sees the run's profile -----------------------------------------------------------------
def test_context_filter_and_degeneracy_take_the_run_profile():
    from peaky.assignment.passes import directors as D
    assert D._context_filter(["C5H10N2O8", "C6H8O4"], RUN) == ["C5H10N2O8", "C6H8O4"]
    assert D._context_filter(["C5H10N2O8", "C6H8O4"], "ambient-air") == ["C6H8O4"]
    assert X.filter_by_context("C5H10N2O8", RUN)[0] and not X.filter_by_context("C5H10N2O8")[0]
    mz = C.ion_mz("C5H10N2O8", "[M+NO3]-")
    on = DG.enumerate_window(mz, -0.5, 0.5, ["[M+NO3]-"], DG.enumeration_space(RUN))
    off = DG.enumerate_window(mz, -0.5, 0.5, ["[M+NO3]-"], DG.enumeration_space("ambient-air"))
    assert "C5H10N2O8" in {v[0] for v in on.values()}
    assert "C5H10N2O8" not in {v[0] for v in off.values()}


def test_level_space_carries_the_run_flags():
    ri = EV.file_run_inputs(sample_id="s", reagent="NO3", context="ambient-air",
                            context_flags=X.profile_flags(RUN))
    assert ri.summary["context_flags"] == {"nox_skeleton": True, "small_acid_band": True}
    reagent, ctx, _ = CX.run_space_args(ri.summary)
    assert ctx == RUN
    plain = EV.file_run_inputs(sample_id="s", reagent="NO3", context="ambient-air")
    assert "context_flags" not in plain.summary and CX.run_space_args(plain.summary)[1] == "ambient-air"


def test_relaxed_widens_on_the_reading_needing_least():
    """With O/C <= 0.7, C5H10N2O8's raw reading is 0.9 out, its k=1 skeleton 0.5 out and its k=2
    skeleton (C5H12O4, O/C 0.8) 0.1 out: the D5 widening takes O/C to 0.8, not to the raw 1.6,
    and leaves every other window as it was. Without the switch it widens to the raw 1.6."""
    tight = dataclasses.replace(RUN, o_to_c=(0.0, 0.7))
    sp = SP.Space("NO3", tight, [], (), catalog={})
    assert sp.space_reason("C5H10N2O8") is not None
    wide = sp.relaxed("C5H10N2O8")
    assert wide is not None
    p0 = wide.profiles[0]
    assert p0.o_to_c == (0.0, 0.8)
    for name in ("h_to_c", "n_to_c", "dbe_to_c"):
        assert getattr(p0, name) == getattr(tight, name)
    raw = SP.Space("NO3", dataclasses.replace(AMB, o_to_c=(0.0, 0.7)), [], (), catalog={}).relaxed("C5H10N2O8")
    assert raw.profiles[0].o_to_c == (0.0, 1.6)


# --- the twin tie-break ------------------------------------------------------------------------------------
def _row(**kw):
    base = dict(compound_formula=None, compound_score=None, compound_category=2, ion_formula=None,
                ion_score=None, ion_category=2, mechanism_id="m", isotope_formula=None, iso_label="M0",
                is_base=True, theo_mz=100.0, rel_abundance=1.0, iso_score=None, iso_category=2,
                sample_peak_id=None, sample_peak_mz=100.0, sample_peak_intensity=1e4, ppm_error=0.2,
                abundance_error=0.0)
    base.update(kw)
    return base


def test_cluster_reading_beats_a_skeleton_only_twin():
    """Propionic acid C3H6O2 [M+NO3]- and glyceryl nitrate C3H7NO5 [M-H]- are the same ion
    (C3H6NO5-). The [M-H]- reading of the +HNO3 neutral is admissible only through its k=1
    skeleton, so on the nitrate run the cluster reading is reported, however the scores fall.
    A twin the raw windows admit (C6H11NO6 [M-H]- beside C6H10O3 [M+NO3]-) keeps its order."""
    assert X.skeleton_only("C3H7NO5", RUN) and not X.skeleton_only("C6H11NO6", RUN)
    scored = pd.DataFrame([
        _row(sample_peak_id="P", compound_formula="C3H7NO5", compound_score=0.99,
             ion_formula="C3H6NO5-", ion_score=0.99),
        _row(sample_peak_id="P", compound_formula="C3H6O2", compound_score=0.90,
             ion_formula="C3H6NO5-", ion_score=0.99),
        _row(sample_peak_id="P", compound_formula="C2H6N2O5", compound_score=0.50,
             ion_formula="C2H5N2O5-", ion_score=0.50, ppm_error=0.9),
        _row(sample_peak_id="Q", compound_formula="C6H11NO6", compound_score=0.99,
             ion_formula="C6H10NO6-", ion_score=0.99),
        _row(sample_peak_id="Q", compound_formula="C6H10O3", compound_score=0.90,
             ion_formula="C6H10NO6-", ion_score=0.99),
    ])
    cfg = PassConfig()
    plain = PC.arbitrate(scored, cfg)["winners"].set_index("peak_id")
    assert plain.loc["P", "neutral"] == "C3H7NO5"           # without the switch: the score order
    w = PC.arbitrate(scored, cfg, RUN)["winners"].set_index("peak_id")
    assert w.loc["P", "neutral"] == "C3H6O2" and w.loc["P", "adduct"] == "[M+NO3]-"
    # its margin is to the best OTHER reading, not to its own same-ion twin
    assert w.loc["P", "eff_margin"] > 0 and not w.loc["P", "tied"]
    assert any(a["formula"] == "C3H7NO5" for a in w.loc["P", "alternatives"])
    assert w.loc["Q", "neutral"] == plain.loc["Q", "neutral"] == "C6H11NO6"
    # the switch off on the profile: no tie-break
    assert PC.arbitrate(scored, cfg, AMB)["winners"].set_index("peak_id").loc["P", "neutral"] == "C3H7NO5"


# --- boundaries ---------------------------------------------------------------------------------------------
def test_small_acid_band_boundaries():
    """Each leg of the band predicate on its own: O >= 2 (C4H2O), O <= 2 Ceff (C3H2O7), H >= 1
    (C4O4, C3O4) and the O/C 2.0 edge of its window (C3H2O6, DBE 3, is in)."""
    for f in ("C4H2O", "C3H2O7", "C4O4", "C3O4"):
        assert not X.small_acid_band_applies(C.parse_formula(f), RUN), f
        assert not keep(f), f
    assert X.small_acid_band_applies(C.parse_formula("C3H2O6"), RUN)
    assert not keep("C3H2O6", AMB) and keep("C3H2O6")


def test_a_failure_reports_the_raw_reading_even_with_skeletons():
    """C3N2O6 has k=2 and k=1 skeletons (C3H1NO4 would read H/C 0.33); none passes, and the reason
    is the raw reading's, word for word the named context's."""
    assert {k for k, _ in X.nox_skeletons(C.parse_formula("C3N2O6"), RUN)} == {2, 1}
    assert X.filter_by_profile("C3N2O6", RUN) == X.filter_by_profile("C3N2O6", AMB)
    assert X.filter_by_profile("C3N2O6", RUN)[1] == "(H+X)/C=0.00 out of (0.7, 2.75)"


def test_the_nitro_reading_starts_at_skeleton_dbe_per_c_one_half():
    """A NITRO group (no O left to the skeleton: O = 2k) is read only on an aromatic-enough
    skeleton: nitro-C4H6 (skeleton DBE/C exactly 0.5) is, nitro-C5H8 (0.4) is not."""
    ks = lambda f: {k for k, _ in X.nox_skeletons(C.parse_formula(f), RUN)}  # noqa: E731
    assert ks("C4H5NO2") == {1}
    assert ks("C5H7NO2") == set()


def test_k_is_capped_at_the_effective_carbon_count():
    """Each group needs its own carbon: a C2 trinitrate-like N3 formula reads at most k = 2."""
    prof = dataclasses.replace(RUN, max_N=5)
    assert max(k for k, _ in X.nox_skeletons(C.parse_formula("C2H3N3O9"), prof)) == 2
    assert max(k for k, _ in X.nox_skeletons(C.parse_formula("CH2N3O9"), prof)) == 1


def test_a_labelled_group_n_is_removed_first():
    """C5H10N^NO8 (one 14N, one 15N): the k=1 skeleton has given up the 15N and keeps the 14N."""
    sk = dict(X.nox_skeletons(C.parse_formula("C5H10N^NO8"), RUN))
    assert sk[1] == {"C": 5, "H": 11, "N": 1, "O": 6}
    assert sk[2] == {"C": 5, "H": 12, "O": 4}


def test_relaxed_judges_a_band_acid_on_the_band_windows():
    """Under a profile tighter than the band (O/C <= 0.7, H/C >= 0.9), acetylenedicarboxylic acid
    C4H2O4 (H/C 0.5, O/C 1.0, DBE/C 1.0) is inside the band's own windows, so the D5 widening
    leaves every Van Krevelen window as the profile has it."""
    tight = dataclasses.replace(RUN, o_to_c=(0.0, 0.7), h_to_c=(0.9, 2.75))
    sp = SP.Space("NO3", tight, [], (), catalog={})
    wide = sp.relaxed("C4H2O4")
    assert wide is not None
    for name in ("h_to_c", "o_to_c", "n_to_c", "dbe_to_c"):
        assert getattr(wide.profiles[0], name) == getattr(tight, name), name


# --- the off-calibration rearbitration judges its alternatives on the run's profile ---------------------------
def _rearb_ledger():
    """A 20-peak 13C-backed CHO calibration backbone near 0 ppm and one off-calibration (-1 ppm),
    uncorroborated, aromatic-monster winner whose only on-calibration alternative is the
    dinitrate C5H10N2O8 [M+NO3]- (raw O/C 1.6: an oxygen-lattice monster on the raw gates)."""
    from peaky.assignment import ledger as L
    rows = [(f"bb{i}", 150.0 + i, 1e5) for i in range(20)] + [(f"bb{i}c", 151.0 + i, 1e3) for i in range(20)]
    rows.append(("P", 288.03, 5e3))
    led = L.new_ledger(pd.DataFrame(rows, columns=["peak_id", "mz", "height"]))
    for i in range(20):
        nf = f"C{8 + i}H{14 + i}O4"
        L.commit_assignment(led, f"bb{i}", neutral_formula=nf, adduct="[M-H]-", ion_formula=nf,
                            ion_score=0.95, compound_score=0.95, eff_score=0.93, eff_margin=0.2, tied=False,
                            ppm_error=(-0.1, 0.0, 0.1, 0.05, -0.05)[i % 5], pass_no=1, method="cheminfo+grid",
                            confidence="High", commentary="backbone",
                            isotopologues=[{"label": "13C", "peak_id": f"bb{i}c"}])
        L.attach_isotopologue(led, f"bb{i}c", f"bb{i}", iso_label="13C", iso_match_score=0.9)
    L.commit_assignment(led, "P", neutral_formula="C20H5NO", adduct="[M-H]-", ion_formula="C20H4NO-",
                        ion_score=0.88, compound_score=0.88, eff_score=0.85, eff_margin=0.2, tied=False,
                        ppm_error=-1.0, pass_no=1, method="cheminfo+grid", confidence="Good", commentary="Pass 1",
                        alternatives=[{"formula": "C5H10N2O8", "adduct": "[M+NO3]-", "ion_score": 0.70,
                                       "raw_score": 0.70, "eff_score": 0.61, "ppm": 0.1}])
    return led


def test_rearbitration_reads_the_alternative_on_the_run_profile():
    from peaky.assignment import passes
    from peaky.assignment import plausibility as PL
    assert PL.implausible("C5H10N2O8") is not None and PL.implausible("C5H10N2O8", profile=RUN) is None
    raw = _rearb_ledger()
    assert passes.rearbitrate_offcal_degenerate(raw, PassConfig(), log=lambda *a: None)["swapped"] == 0
    run = _rearb_ledger()
    assert passes.rearbitrate_offcal_degenerate(run, PassConfig(), log=lambda *a: None, profile=RUN)["swapped"] == 1
    assert run.set_index("peak_id").loc["P", "neutral_formula"] == "C5H10N2O8"
    # a profile with the switches off is the raw gates
    off = _rearb_ledger()
    assert passes.rearbitrate_offcal_degenerate(off, PassConfig(), log=lambda *a: None, profile=AMB)["swapped"] == 0
