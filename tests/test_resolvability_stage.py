"""assignment.resolvability: the per-file separability stamp, and the way the
run and the tier engine read it."""
import numpy as np
import pandas as pd

from peaky.assignment import ledger as L
from peaky.assignment import resolvability as RV
from peaky.assignment import tiers as T
from peaky.assignment.passes.config import PassConfig
from peaky.chem import resolution as RES

R = 10000.0


def _commit(led, pid, nf, adduct="[M-H]-", **kw):
    ion = nf + ("-" if adduct == "[M-H]-" else ".Br-")
    args = dict(neutral_formula=nf, adduct=adduct, ion_formula=ion, ion_score=0.9, compound_score=0.9,
                eff_score=0.9, eff_margin=0.3, tied=False, ppm_error=0.1, pass_no=1,
                method="cheminfo+grid", confidence="High", commentary="Pass 1")
    args.update(kw)
    L.commit_assignment(led, pid, **args)


def _ledger():
    hw = RES.hwhm(300.0, R)
    peaks = pd.DataFrame([
        ("A", 300.0, 1e4),                    # M0, blended with the unexplained U
        ("U", 300.0 + 1.5 * hw, 8e3),         # picked, never assigned: still a neighbour
        ("B", 320.0, 1e4),                    # M0, isolated
        ("C", 340.0, 1e4),                    # M0, a synthetic sub-peak shares its m/z
        ("C.2", 340.0, 2e3),                  # composite sub-peak: NOT a picked peak
        ("D", 360.0, 1e4),                    # M0 with a resolved 13C child
        ("D13", 360.0 + 1.003355, 1e3),
    ], columns=["peak_id", "mz", "height"])
    led = L.new_ledger(peaks)
    led.loc[led.peak_id == "C.2", ["synthetic", "host_peak_id"]] = [True, "C"]
    _commit(led, "A", "C10H16O5")
    _commit(led, "B", "C10H14O6")
    _commit(led, "C", "C9H14O7")
    _commit(led, "D", "C8H12O8", isotopologues=[{"label": "13C", "score": 0.9, "peak_id": "D13"}])
    L.attach_isotopologue(led, "D13", "D", iso_label="13C", iso_match_score=0.9)
    return led


def test_the_stamp_lands_on_m0_rows_only_from_the_pool_of_picked_peaks():
    led = _ledger()
    assert not RV.already_stamped(led)
    out = RV.stamp_resolvability(led, R, log=lambda *a: None)
    by = led.set_index("peak_id")
    assert by.at["A", "resolvability"] == "blended" and by.at["A", "sep_hwhm"] == pytest_approx(1.5)
    assert by.at["B", "resolvability"] == "isolated" and np.isnan(by.at["B", "sep_hwhm"])
    # the synthetic sub-peak at C's own m/z is not a picked peak: C is not 'unresolvable'
    assert by.at["C", "resolvability"] == "isolated"
    # D's 13C child is a picked peak 1 Da away: far beyond 8 HWHM at R 10k -> isolated;
    # the child itself (not M0) carries no stamp
    assert by.at["D", "resolvability"] == "isolated" and pd.isna(by.at["D13", "resolvability"])
    assert pd.isna(by.at["U", "resolvability"]) and pd.isna(by.at["C.2", "resolvability"])
    assert out["counts"] == {"isolated": 3, "blended": 1} and out["n_pool"] == 6
    assert out["model"]["r_at_200"] == pytest_approx(R)
    assert RV.already_stamped(led)


def pytest_approx(x, rel=0.02):
    import pytest
    return pytest.approx(x, rel=rel)


def test_the_tier_caps_a_blended_uncorroborated_peak_and_spares_a_corroborated_one():
    led = _ledger()
    RV.stamp_resolvability(led, R, log=lambda *a: None)
    # give A a second channel in another test copy: corroboration carries it
    led2 = led.copy()
    hw = RES.hwhm(300.0, R)
    led2 = pd.concat([led2, L.new_ledger(pd.DataFrame([("A2", 379.9, 5e3)], columns=["peak_id", "mz", "height"]))],
                     ignore_index=True)
    _commit(led2, "A2", "C10H16O5", adduct="[M+Br]-")
    t1 = T.compute_tiers(led, cfg=PassConfig(height_cutoff_cps=10.0)).set_index("peak_id")
    t2 = T.compute_tiers(led2, cfg=PassConfig(height_cutoff_cps=10.0)).set_index("peak_id")
    assert t1.at["A", "tier"] == "Candidate" and t1.at["A", "tier_reason"].startswith("blended peak")
    assert "1.50 HWHM" in t1.at["A", "tier_reason"] and "not this ion's own" in t1.at["A", "tier_reason"]
    assert t1.at["B", "tier"] == "Assigned" and t1.at["D", "tier"] == "Assigned"
    assert t2.at["A", "tier"] == "Assigned" and "blended peak" in t2.at["A", "tier_reason"]
    assert "carried by the corroboration" in t2.at["A", "tier_reason"]
    # an unresolvable neighbour (under the 0.4-HWHM fit floor) is worded as one observable
    led3 = _ledger()
    led3.loc[led3.peak_id == "U", "mz"] = 300.0 + 0.2 * hw
    RV.stamp_resolvability(led3, R, log=lambda *a: None)
    t3 = T.compute_tiers(led3, cfg=PassConfig(height_cutoff_cps=10.0)).set_index("peak_id")
    assert t3.at["A", "tier"] == "Candidate" and t3.at["A", "tier_reason"].startswith("unresolvable peak")


def test_without_a_stamp_the_tier_rule_is_inert_and_the_stage_skips_offline():
    led = _ledger()
    t = T.compute_tiers(led, cfg=PassConfig(height_cutoff_cps=10.0)).set_index("peak_id")
    assert t.at["A", "tier"] == "Assigned"
    from peaky.assignment import assign as A
    assert A._width_model(None, None, "s", None, lambda *a: None) is None
    assert A._width_model("auto", None, "s", None, lambda *a: None) is None      # offline: nothing to measure
    assert A._width_model("none", "CLIENT", "s", None, lambda *a: None) is None
    assert A._width_model("6500", None, "s", None, lambda *a: None).r_at(200.0) == pytest_approx(6500.0)
    m = RES.Resolution.from_r(6500.0)
    assert A._width_model(m, None, "s", None, lambda *a: None) is m
    assert A._width_model(6500.0, None, "s", None, lambda *a: None) == m


def test_a_ledger_without_roles_or_peaks_is_a_no_op():
    empty = pd.DataFrame(columns=["peak_id", "mz", "height"])
    out = RV.stamp_resolvability(empty, R, log=lambda *a: None)
    assert out["counts"] == {} and out["n_pool"] == 0 and "resolvability" in empty.columns
