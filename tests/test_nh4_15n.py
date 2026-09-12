"""Offline tests for the ¹⁵N-labelled ammonium (NH4_15N) reagent profile: the
[M+^NH4]+ adduct + its dehydration aliases, mechanism mapping both ways, the
^NH3 cluster library, the ammonium-15n context, local scoring on the labelled
channel (with the 14N impurity satellite), and the in-source dehydration
relabel. Built from the 2026-09-10 exploratory file."""
import numpy as np
import pandas as pd
import pytest

from peaky.chem import chemistry as C, contexts as X, profiles as P, reagents as R
from peaky.io import local_scoring as LS, io_mascope as IO
from peaky.assignment import cleanup, ledger as L
from peaky.assignment.passes import core as PC

M15N_SHIFT = 19.03086   # ^N + 4H - e


def test_profile_resolves_and_is_labelled_positive():
    prof = P.resolve("NH4_15N")
    assert prof.polarity == "+"
    assert prof.adducts == ["[M+^NH4]+", "[M+H]+"]
    assert prof.detect_adduct == "[M+^NH4]+"
    assert prof.normaliser == "tic"           # reagent ions sit below m/z 40
    assert prof.purity == pytest.approx(0.98)
    assert prof.label_isotope is None         # no covalent 15N products
    assert prof.context == "ammonium-15n"
    for alias in ("15nh4", "^nh4+", "ammonium-15n", "15n-ammonium", "nh4-15n"):
        assert P.resolve(alias).name == "NH4_15N", alias
    # the 14N ammonium is NOT a profile of its own -- plain 'ur' still resolves to uronium
    assert P.resolve("ur").name == "Ur"


def test_auto_detect_from_server_mechanism():
    peaks = pd.DataFrame({"peak_id": [1, 2], "mz": [197.13, 179.107],
                          "ionization_mechanism": ["+^NH4+", "+H+"]})
    assert IO.detect_adducts(peaks) == ["[M+^NH4]+", "[M+H]+"]
    assert P.resolve("auto", peaks).name == "NH4_15N"


def test_adduct_masses_match_the_exploratory_file():
    # C11H14O2: 197.1304 / 179.1067 / 161.0961 / 179.1196 observed (all within 1 ppm)
    assert C.ion_mz("C11H14O2", "[M+^NH4]+") == pytest.approx(197.1302, abs=5e-4)
    assert C.ion_mz("C11H14O2", "[M+H]+") == pytest.approx(179.1067, abs=5e-4)
    assert C.ion_mz("C11H14O2", "[M+H-H2O]+") == pytest.approx(161.0961, abs=5e-4)
    assert C.ion_mz("C11H14O2", "[M+^NH4-H2O]+") == pytest.approx(179.1197, abs=5e-4)
    # the labelled shift is +19.0309, one 15N-14N above the 14N adduct
    assert C.ADDUCT_SHIFTS["[M+^NH4]+"] - C.ADDUCT_SHIFTS["[M+NH4]+"] == pytest.approx(0.99703, abs=1e-5)
    # D4 siloxane: 315.1059 / 297.0824 observed
    assert C.ion_mz("C8H24O4Si4", "[M+^NH4]+") == pytest.approx(315.1061, abs=5e-4)
    assert C.ion_mz("C8H24O4Si4", "[M+H]+") == pytest.approx(297.0825, abs=5e-4)


def test_mechanism_maps_both_ways():
    assert LS.adduct_to_mech("[M+^NH4]+") == "+^NH4+"
    assert IO.ADDUCT_TO_MECH["[M+^NH4]+"] == "+^NH4+"
    assert IO.MECH_TO_ADDUCT["+^NH4+"] == "[M+^NH4]+"
    # the dehydration aliases are relabel-only: no server mechanism
    assert "[M+^NH4-H2O]+" not in IO.ADDUCT_TO_MECH
    assert "[M+H-H2O]+" not in IO.ADDUCT_TO_MECH
    # element-diff -> label (the ^N in the ion string upgrades the 14N reading)
    assert PC._mech_to_adduct({"ion_formula": "C11H18^NO2+", "compound_formula": "C11H14O2"}) == "[M+^NH4]+"
    assert PC._mech_to_adduct({"ion_formula": "C11H18NO2+", "compound_formula": "C11H14O2"}) == "[M+NH4]+"
    assert PC._mech_to_adduct({"ion_formula": "C11H16^NO+", "compound_formula": "C11H14O2"}) == "[M+^NH4-H2O]+"
    assert PC._mech_to_adduct({"ion_formula": "C11H13O+", "compound_formula": "C11H14O2"}) == "[M+H-H2O]+"


def test_ammonium_cluster_library():
    lib = R.build_library("ammonium15N")
    mzs = {round(m, 3): f for _, m, f in lib}
    assert mzs[19.031] == "H4^N+"                 # ^NH4+
    assert 37.054 in mzs and 37.041 in mzs        # (^NH3)2H+ and ^NH4+·H2O
    assert 18.034 in mzs                          # the 14N impurity twin
    # the bare 15N clusters and their hydrates all sit below m/z 80 (the urea
    # crossover and CO2 source clusters are the only heavier entries)
    assert all(m < 80 for _, m, f in lib if f.startswith("H") and "C" not in f)
    assert R.reagent_for_adducts(["[M+^NH4]+", "[M+H]+"]) == "ammonium15N"
    assert R.reagent_for_adducts(["[M+H]+", "[M+(CH4N2O)H]+"]) == "urea"   # unchanged


def test_context_and_families():
    ctx = X.get_context("ammonium-15n")
    assert ctx.polarity == "positive"
    assert "[M+^NH4]+" in ctx.reagent_adducts and "[M+H]+" in ctx.reagent_adducts
    assert X.get_context("15nh4") is ctx
    # every alias SAYS labelled: a bare "ammonium" must NOT silently hand an
    # unlabelled-ammonium user the 15N channels
    assert all("15n" in a or "^" in a for a in X.CONTEXTS if X.CONTEXTS[a] is ctx)
    for bare in ("ammonium", "nh4", "nh4-cims"):
        with pytest.raises(ValueError):
            X.get_context(bare)
    for fam in ("siloxane", "pdms", "phthalate", "glycol_peg"):
        assert "[M+^NH4]+" in X.CONTAMINANT_FAMILIES[fam]["adducts"], fam
    # ammonium adduct of D4 passes the context (Si only as a siloxane scaffold)
    assert X.filter_by_profile("C8H24O4Si4", ctx)[0]


def test_local_scoring_on_the_labelled_channel_sees_the_14n_satellite():
    peaks = pd.DataFrame({
        "peak_id": ["a", "b", "c", "d"],
        "mz": [197.1304, 198.1336, 196.1332, 199.1370],
        "height": [670_000.0, 76_000.0, 12_200.0, 3_400.0],
    })
    res = LS.score_candidates_local(peaks, ["C11H14O2"], ["[M+^NH4]+"], purity=0.98)
    assert len(res), "no rows scored on [M+^NH4]+"
    ions = set(res["ion_formula"].astype(str)) if "ion_formula" in res.columns else set()
    assert any("^N" in i for i in ions), ions
    base = res[res["is_base"].astype(bool)] if "is_base" in res.columns else res
    assert (base["sample_peak_id"].astype(str) == "a").any()
    # the 2 % 14N impurity line at -0.99703 is part of the predicted envelope
    others = res[~res["is_base"].astype(bool)] if "is_base" in res.columns else res
    got = set(others["sample_peak_id"].dropna().astype(str))
    assert "c" in got, f"14N satellite not attributed: {res.to_dict(orient='records')[:6]}"


_ADDUCT_ATOMS = {"[M+H]+": {"H": 1}, "[M+^NH4]+": {"^N": 1, "H": 4},
                 "[M+H-H2O]+": {"H": -1, "O": -1}, "[M+^NH4-H2O]+": {"^N": 1, "H": 2, "O": -1}}


def _ion(nf, ad):
    """Real ION formula string (neutral + adduct atoms), e.g. C11H14O2 [M+^NH4]+ -> C11H18^NO2+."""
    d = C.parse_formula(nf)
    for el, n in _ADDUCT_ATOMS[ad].items():
        d[el] = d.get(el, 0) + n
    return C.format_formula({k: v for k, v in d.items() if v}) + "+"


def _ledger(rows):
    """rows: (peak_id, mz, height, neutral|None, adduct|None)"""
    peaks = pd.DataFrame([{"peak_id": pid, "mz": mz, "height": h} for pid, mz, h, *_ in rows])
    led = L.new_ledger(peaks)
    for pid, mz, h, nf, ad in rows:
        if nf:
            L.commit_assignment(led, pid, neutral_formula=nf, adduct=ad, ion_formula=_ion(nf, ad),
                                ion_score=0.9, pass_no=1, method="test", confidence="High",
                                commentary="test")
            # as production leaves it after apply_tiers: M0 rows tiered, every
            # other row (unexplained included) still at the ledger default pd.NA
            led.loc[led["peak_id"] == pid, "tier"] = "Assigned"
    return led


def test_dehydration_relabel_reads_the_cascade_onto_the_hydrate():
    X1, X2, X3 = "C11H14O2", "C12H18O2", "C6H12O2"
    Y1, Y2, Y3 = "C11H12O", "C12H16O", "C6H10O"
    rows = [
        ("p1", C.ion_mz(X1, "[M+^NH4]+"), 670_000, X1, "[M+^NH4]+"),
        ("p2", C.ion_mz(X1, "[M+H]+"), 232_000, X1, "[M+H]+"),
        ("p3", C.ion_mz(Y1, "[M+H]+"), 203_000, Y1, "[M+H]+"),         # == X1 [M+H-H2O]+
        ("p4", C.ion_mz(Y1, "[M+^NH4]+"), 28_700, Y1, "[M+^NH4]+"),    # == X1 [M+^NH4-H2O]+
        ("p5", C.ion_mz(X2, "[M+^NH4]+"), 134_000, X2, "[M+^NH4]+"),
        ("p6", C.ion_mz(X2, "[M+^NH4-H2O]+"), 42_700, None, None),     # unexplained, no [M+H]+ of X2 at all
        ("p7", C.ion_mz(Y2, "[M+H]+"), 113_000, Y2, "[M+H]+"),
        ("p8", C.ion_mz(X2, "[M+H-H2O]+") - C.neutral_mass("H2O"), 11_400, None, None),  # 2nd loss
        # a GENUINE Y3 next to a hydrate X3: Y3's own adduct is as strong as its [M+H]+
        ("p9", C.ion_mz(X3, "[M+^NH4]+"), 50_000, X3, "[M+^NH4]+"),
        ("p10", C.ion_mz(X3, "[M+H]+"), 20_000, X3, "[M+H]+"),
        ("p11", C.ion_mz(Y3, "[M+H]+"), 30_000, Y3, "[M+H]+"),
        ("p12", C.ion_mz(Y3, "[M+^NH4]+"), 40_000, Y3, "[M+^NH4]+"),
    ]
    led = _ledger(rows)
    out = cleanup.relabel_ammonium_dehydration(led, log=lambda *a: None)
    at = lambda pid, col: led.loc[led["peak_id"] == pid, col].iloc[0]
    assert (at("p3", "neutral_formula"), at("p3", "adduct")) == (X1, "[M+H-H2O]+")
    assert at("p3", "tier") == "Candidate"
    assert (at("p4", "neutral_formula"), at("p4", "adduct")) == (X1, "[M+^NH4-H2O]+")
    assert (at("p6", "role"), at("p6", "neutral_formula"), at("p6", "adduct")) == (L.ROLE_M0, X2, "[M+^NH4-H2O]+")
    # committed from an UNEXPLAINED peak (tier was pd.NA): lands in Candidate with a reason
    assert at("p6", "tier") == "Candidate"
    assert isinstance(at("p6", "tier_reason"), str) and "dehydration" in at("p6", "tier_reason")
    assert (at("p7", "neutral_formula"), at("p7", "adduct")) == (X2, "[M+H-H2O]+")
    assert at("p8", "role") == L.ROLE_UNEXPLAINED           # second loss: note only
    assert "second in-source water loss" in str(at("p5", "tier_reason"))
    # the genuine oxygenate keeps its reading, with the ambiguity noted
    assert (at("p11", "neutral_formula"), at("p11", "adduct")) == (Y3, "[M+H]+")
    assert (at("p12", "neutral_formula"), at("p12", "adduct")) == (Y3, "[M+^NH4]+")
    assert "ambiguity" in str(at("p11", "tier_reason"))
    assert out["nh4_deh_relabeled"] == 3 and out["nh4_deh_committed"] == 1
    # parents with a declustering product: X1, X2, X3 and the genuine Y3 (p12)
    assert out["nh4_deh_ambiguous"] == 2 and out["nh4_deh_parents"] == 4
    # the parent rows are untouched
    assert (at("p1", "neutral_formula"), at("p1", "adduct")) == (X1, "[M+^NH4]+")
    assert L.validate(led) == []


def test_dehydration_relabel_needs_a_declustering_product():
    # a labelled adduct with neither [M+H]+ nor [M+^NH4-H2O]+ present: nothing happens
    X1, Y1 = "C11H14O2", "C11H12O"
    led = _ledger([("p1", C.ion_mz(X1, "[M+^NH4]+"), 1e5, X1, "[M+^NH4]+"),
                   ("p3", C.ion_mz(Y1, "[M+H]+"), 5e4, Y1, "[M+H]+")])
    out = cleanup.relabel_ammonium_dehydration(led, log=lambda *a: None)
    assert out["nh4_deh_parents"] == 0
    assert led.loc[led["peak_id"] == "p3", "neutral_formula"].iloc[0] == Y1


# ---------------------------------------------------------------- round 2 ----
from peaky.assignment import masscal as MC
from peaky.assignment.passes import config as PCfg, postprocess as PP
from peaky.chem import isotopes as ISO
from peaky.assignment.passes import directors as D


def test_mass_trend_fit_recovers_an_absolute_offset():
    rng = np.random.default_rng(0)
    mz = rng.uniform(60, 420, 300)
    ppm = -0.25 + (-0.12) * 1000 / mz + rng.normal(0, 0.15, 300)
    fit = MC.fit_mass_trend(mz, ppm, min_n=20, sigma_floor=0.1)
    assert abs(fit.a + 0.25) < 0.08 and abs(fit.b + 0.12) < 0.02 and fit.n > 250
    assert 60 <= fit.mz_lo < 70 and 410 < fit.mz_hi <= 420   # backbone coverage recorded
    # a FLAT source keeps the constant model (trend not accepted)
    assert MC.fit_mass_trend(mz, -0.3 + rng.normal(0, 0.15, 300), min_n=20) is None


def test_z_of_uses_the_mass_dependent_centre_when_given_mz():
    cfg = PCfg.PassConfig()
    cfg.cal_mu, cfg.cal_sigma = -0.30, 0.33
    cfg.cal_a, cfg.cal_b, cfg.cal_sigma_trend = -0.25, -0.12, 0.25
    # ketene.^NH4+ at m/z 61: -2.0 ppm is ON the trend (-0.25 - 1.97 = -2.2)
    assert PC.z_of(-2.0, cfg) > 4.0            # constant model rejects it
    assert PC.z_of(-2.0, cfg, mz=61.04) < 1.0  # mass-dependent centre accepts it
    # and at m/z 300 the two agree to within a sigma
    assert abs(PC.z_of(-0.4, cfg, mz=300) - PC.z_of(-0.4, cfg)) < 1.0


def test_envelope_predictor_has_the_14n_line_for_a_labelled_ion():
    pat = ISO.isotope_pattern("C11H18^NO2+", min_rel=0.06, diag_min_rel=0.001)
    lab = {l: (d, r) for d, r, l in pat}
    assert "14N" in lab and abs(lab["14N"][0] + 0.99703) < 0.001
    assert 0.015 < lab["14N"][1] < 0.025          # 2 % impurity / 98 % label
    assert "13C" in lab                           # the normal envelope is intact
    # an unlabelled ion has no negative line
    assert not any(d < 0 for d, _, _ in ISO.isotope_pattern("C11H18NO2+", diag_min_rel=0.001))


def test_the_active_profile_purity_drives_both_purity_consumers():
    """ReagentProfile.purity used to be inert: nothing read it, the envelope
    predictor hard-coded 0.98 and the scorer passed no purity at all."""
    from peaky.io import local_scoring as LS

    def impurity(ion):
        return {l: r for _, r, l in ISO.isotope_pattern(ion, min_rel=0.0005,
                                                        diag_min_rel=0.0005)}["14N"]

    try:
        assert ISO.label_purity() == ISO.LABEL_PURITY_15N        # the default
        # the 14N line is (1 - purity)/purity of M0 -- 2.0 % at the 0.98 default ...
        assert abs(impurity("C11H18^NO2+") - 0.02 / 0.98) < 0.002
        # ... and a 90 %-pure bottle moves it, in the ENVELOPE PREDICTOR ...
        assert ISO.set_label_purity(0.90) == 0.90
        assert abs(impurity("C11H18^NO2+") - 0.10 / 0.90) < 0.005
        # ... and in the SCORER's predicted envelope (purity unset = the active value)
        peaks = pd.DataFrame([
            {"peak_id": "a", "mz": C.ion_mz("C11H14O2", "[M+^NH4]+"), "height": 1e5},
            {"peak_id": "b", "mz": C.ion_mz("C11H14O2", "[M+^NH4]+") - 0.99703,
             "height": 1e4},
        ])
        def scored_14n_rel(**kw):
            got = LS.score_candidates_local(peaks, ["C11H14O2"], ["[M+^NH4]+"], **kw)
            m0 = got.loc[got["is_base"].astype(bool), "theo_mz"].iloc[0]
            line = got[(got["theo_mz"] - (m0 - 0.99703)).abs() < 0.002]
            assert len(line) == 1, got[["iso_label", "theo_mz", "rel_abundance"]]
            return float(line["rel_abundance"].iloc[0])

        assert scored_14n_rel() > 0.08                 # 10 %, not the 2 % default
        # an explicit purity still wins over the active value
        assert scored_14n_rel(purity=0.98) < 0.04
    finally:
        ISO.set_label_purity(None)
    assert ISO.label_purity() == ISO.LABEL_PURITY_15N


def test_envelope_completion_claims_the_14n_satellite_even_from_a_locked_amine():
    X = "C11H14O2"
    rows = [("p1", C.ion_mz(X, "[M+^NH4]+"), 670_000, X, "[M+^NH4]+"),
            ("s14", C.ion_mz(X, "[M+^NH4]+") - 0.99703, 12_200, "C11H17NO2", "[M+H]+"),
            ("s13", C.ion_mz(X, "[M+^NH4]+") + 1.003355, 76_000, None, None)]
    led = _ledger(rows)
    L.lock_peaks(led, ["s14"])                    # pass-1 style lock on the phantom amine
    led.loc[led.peak_id == "s14", "confidence"] = "High"
    cfg = PCfg.PassConfig()
    PP.complete_isotope_envelopes(led, cfg, log=lambda *a: None)
    at = lambda pid, col: led.loc[led["peak_id"] == pid, col].iloc[0]
    assert at("s14", "role") == L.ROLE_ISO and at("s14", "iso_label") == "14N"
    assert at("s13", "role") == L.ROLE_ISO and at("s13", "iso_label") == "13C"


def test_the_14n_displacement_survives_a_missing_pass_no():
    """The pass-0 exemption read the cell as `int(pd.to_numeric(...) or 1)`:
    NaN is TRUTHY so it survives the `or` and int(nan) raises, and pd.NA raises
    on the truth-test itself. A row with no pass_no must read as a normal pass."""
    assert PP._pass_no(pd.NA) == 1 and PP._pass_no(float("nan")) == 1
    assert PP._pass_no("not a number") == 1          # coerced, then defaulted
    assert PP._pass_no(0) == 0 and PP._pass_no("3") == 3 and PP._pass_no(7.0) == 7

    X = "C11H14O2"
    led = _ledger([("p1", C.ion_mz(X, "[M+^NH4]+"), 670_000, X, "[M+^NH4]+"),
                   ("s14", C.ion_mz(X, "[M+^NH4]+") - 0.99703, 12_200,
                    "C11H17NO2", "[M+H]+")])
    led.loc[led.peak_id == "s14", "pass_no"] = pd.NA      # e.g. a CSV round-trip
    L.lock_peaks(led, ["s14"])
    PP.complete_isotope_envelopes(led, PCfg.PassConfig(), log=lambda *a: None)
    s14 = led.loc[led["peak_id"] == "s14"].iloc[0]
    assert s14["role"] == L.ROLE_ISO and s14["iso_label"] == "14N"


def test_dehydration_relabel_overrides_a_pass1_lock_but_not_a_pass0_lock():
    X1, Y1 = "C11H14O2", "C11H12O"
    led = _ledger([("p1", C.ion_mz(X1, "[M+^NH4]+"), 670_000, X1, "[M+^NH4]+"),
                   ("p2", C.ion_mz(X1, "[M+H]+"), 232_000, X1, "[M+H]+"),
                   ("p3", C.ion_mz(Y1, "[M+H]+"), 203_000, Y1, "[M+H]+"),
                   ("p4", C.ion_mz(Y1, "[M+^NH4]+"), 28_700, Y1, "[M+^NH4]+")])
    L.lock_peaks(led, ["p3", "p4"])                       # pass-1 backbone locks
    cleanup.relabel_ammonium_dehydration(led, log=lambda *a: None)
    at = lambda pid, col: led.loc[led["peak_id"] == pid, col].iloc[0]
    assert (at("p3", "neutral_formula"), at("p3", "adduct")) == (X1, "[M+H-H2O]+")
    assert (at("p4", "neutral_formula"), at("p4", "adduct")) == (X1, "[M+^NH4-H2O]+")
    # a pass-0 (known-species) lock stays
    led2 = _ledger([("p1", C.ion_mz(X1, "[M+^NH4]+"), 670_000, X1, "[M+^NH4]+"),
                    ("p2", C.ion_mz(X1, "[M+H]+"), 232_000, X1, "[M+H]+"),
                    ("p3", C.ion_mz(Y1, "[M+H]+"), 203_000, Y1, "[M+H]+")])
    led2.loc[led2.peak_id == "p3", "pass_no"] = 0
    L.lock_peaks(led2, ["p3"])
    cleanup.relabel_ammonium_dehydration(led2, log=lambda *a: None)
    assert led2.loc[led2.peak_id == "p3", "neutral_formula"].iloc[0] == Y1


def test_positive_known_species_include_siloxanes_and_indoor_sulfur():
    reg = D._known_species("positive", "ammonium-15n")
    assert reg["cyclosiloxane"]["C8H24O4Si4"].startswith("octamethylcyclotetrasiloxane")
    assert "C7H5NS" in reg["indoor_sulfur"] and "C4H9NS2" in reg["indoor_sulfur"]
    # negative mode untouched
    assert "cyclosiloxane" not in D._known_species("negative")


def test_ammonium_context_admits_small_amines_and_library_has_source_ions():
    ctx = X.get_context("ammonium-15n")
    assert X.filter_by_profile("C4H11N", ctx)[0] and X.filter_by_profile("C3H9N", ctx)[0]
    assert "[M+Na]+" not in ctx.reagent_adducts
    lib = {round(m, 3): f for _, m, f in R.build_library("ammonium15N")}
    assert lib[65.049] == "CH7^N2O+"            # (^NH3)2H+.CO / 15N-formamide.^NH4+
    assert 61.04 in lib                         # [urea+H]+ crossover


# ---------------------------------------------------------------- round 3 ----
def test_absolute_floor_and_trend_centre_in_confidence():
    cfg = PCfg.PassConfig()
    cfg.cal_mu, cfg.cal_sigma = -0.27, 0.275
    cfg.cal_a, cfg.cal_b, cfg.cal_sigma_trend = 0.319, -0.111, 0.25
    # dimethylamine [M+H]+ at m/z 46: -2.7 ppm, 0.6 ppm off the 1/mz trend =
    # 0.03 mDa -> inside the absolute floor
    assert PC.z_of(-2.73, cfg, mz=46.065) < 2.0
    assert PC.z_of(-2.73, cfg, mz=460.65) > 4.0      # the floor is inert at high mass
    assert abs(PC.cal_center(cfg, 60) - (0.319 - 0.111 * 1000 / 60)) < 1e-9
    # confidence: the same -2 ppm ion is Good against the trend centre, Low against the constant one
    # (-2.5 ppm at m/z 61 is 1.0 ppm from the trend centre -1.50 but 2.2 ppm from the constant one)
    assert PC.confidence_label(0.85, -2.5, 0, False, cfg, mz=61.04).startswith("Good")
    assert PC.confidence_label(0.85, -2.5, 0, False, cfg).startswith("Low")


def test_dehydration_relabel_respects_the_brightness_ceiling():
    # acetone [M+H]+ (9.2 kcps) must NOT become "propanediol - H2O" of a 2 kcps parent
    X, Y = "C3H8O2", "C3H6O"
    led = _ledger([("p1", C.ion_mz(X, "[M+^NH4]+"), 1_980, X, "[M+^NH4]+"),
                   ("p2", C.ion_mz(X, "[M+^NH4-H2O]+"), 3_970, None, None),
                   ("p3", C.ion_mz(Y, "[M+H]+"), 9_236, Y, "[M+H]+")])
    out = cleanup.relabel_ammonium_dehydration(led, log=lambda *a: None)
    at = lambda pid, col: led.loc[led["peak_id"] == pid, col].iloc[0]
    assert (at("p3", "neutral_formula"), at("p3", "adduct")) == (Y, "[M+H]+")
    assert "brighter" in str(at("p3", "tier_reason"))
    # the 3.97 kcps ion at the [M+^NH4-H2O]+ mass is ABOVE the 1.5x ceiling of a 1.98 kcps
    # parent too (in the real file it is acetone.^NH4+), so nothing is committed there either
    assert at("p2", "role") == L.ROLE_UNEXPLAINED
    assert out["nh4_deh_relabeled"] == 0 and out["nh4_deh_committed"] == 0


def test_library_labels_the_reagent_made_acetamide_and_context_opens_organosulfur():
    lib = {round(m, 3): (lab, f) for lab, m, f in R.build_library("ammonium15N")}
    assert lib[61.041][1] == "C2H6^NO+" and "reagent-derived" in lib[61.041][0]
    assert lib[79.065][1] == "C2H9^N2O+"
    ctx = X.get_context("ammonium-15n")
    assert "organosulfur" in ctx.pass3_families
    assert X.CONTAMINANT_FAMILIES["organosulfur"]["add"] == {"S": (1, 2)}
    assert "C5H12N2S" in D._known_species("positive")["indoor_sulfur"]
