"""Recovery re-reads a committed halogen-free M0 whose 37Cl / 81Br M+2 twin sits unexplained at the halogen's ratio
(cleanup._halogen_twins, recover_isotope_gated): a C12 CHON reading of a chlorinated C10 product whose 13C line,
read by area, fits the CHON reading's carbon count about as well as the truth's."""
import pandas as pd
import pytest

from peaky.assignment import cleanup as CU
from peaky.assignment import ledger as L
from peaky.assignment import passes as P
from peaky.chem import chemistry as C
from peaky.chem import contexts as X
from peaky.chem import isotopes as ISO

MZ = 285.0747                                    # C10H18ClO7- = C10H19ClO7 [M-H]-


def _led(r2=0.31, *, offset=ISO.D_37CL, twin_role=None, formula=("C12H14O4", "[M+^NO3]-", "C12H14^NO7-")):
    led = L.new_ledger(pd.DataFrame([
        dict(peak_id="p", mz=MZ, height=2400.0, area=2.4, signal_to_noise=300.0),
        dict(peak_id="c13", mz=MZ + 1.00336, height=290.0, area=0.29, signal_to_noise=35.0),
        dict(peak_id="twin", mz=MZ + offset, height=2400.0 * r2, area=2.4 * r2, signal_to_noise=90.0),
        dict(peak_id="other", mz=400.1, height=5e3, area=5.0, signal_to_noise=500.0)]))
    nf, ad, ion = formula
    L.commit_assignment(led, "p", neutral_formula=nf, adduct=ad, ion_formula=ion, ion_score=0.95, ppm_error=0.26,
                        pass_no=1, method="test", confidence="Good", commentary="Pass 1 (test)")
    if twin_role == "M0":
        L.commit_assignment(led, "twin", neutral_formula="C11H20O8", adduct="[M-H]-", ion_formula="C11H19O8-",
                            ion_score=0.9, ppm_error=0.1, pass_no=1, method="test", confidence="Good",
                            commentary="Pass 1 (test)")
    return led


def _score(client, sample_id, formulas, *, allow_partial=True, mechanism_ids=None):
    rows = []
    if "C10H19ClO7" in formulas:
        rows.append({"ion_formula": "C10H18ClO7-", "ion_score": 0.78, "ppm_error": 0.1, "sample_peak_id": "p",
                     "sample_peak_mz": MZ, "sample_peak_intensity": 2400.0})
    return pd.DataFrame(rows)


def _run(led):
    cfg = P.PassConfig()
    cfg.cal_mu, cfg.cal_sigma, cfg.mechanism_ids = 0.0, 0.3, None
    return CU.recover_isotope_gated(None, "S", led, X.get_context("ambient-air"), cfg, score_fn=_score,
                                    log=lambda *a: None)


def test_a_halogen_free_reading_that_leaves_the_37cl_twin_unexplained_is_replaced():
    led = _led()
    out = _run(led)
    assert out["halogen_twin_swaps"] == 1 and out["recovered"] == 1
    row = led[led.peak_id == "p"].iloc[0]
    assert row.neutral_formula == "C10H19ClO7" and row.adduct == "[M-H]-" and row.role == L.ROLE_M0
    assert L.validate(led) == []


@pytest.mark.parametrize("case", ["ratio", "explained", "offset", "has_halogen"])
def test_no_swap_without_the_twin(case):
    kw = {"ratio": dict(r2=0.08), "explained": dict(twin_role="M0"), "offset": dict(offset=ISO.D_37CL + 0.003),
          "has_halogen": dict(formula=("C10H19ClO7", "[M-H]-", "C10H18ClO7-"))}[case]
    led = _led(**kw)
    before = led[led.peak_id == "p"].iloc[0].neutral_formula
    out = _run(led)
    assert out.get("halogen_twin_swaps", 0) == 0
    assert led[led.peak_id == "p"].iloc[0].neutral_formula == before


def test_the_twins_are_found_at_the_exact_offset_only():
    assert set(CU._halogen_twins(_led())) == {"p"} and CU._halogen_twins(_led())["p"][3] == "Cl"
    t = MZ + ISO.D_37CL
    assert set(CU._halogen_twins(_led(offset=ISO.D_37CL - 1.5e-6 * t)))  == {"p"}      # 1.5 ppm below: found
    assert CU._halogen_twins(_led(offset=ISO.D_37CL - 2.5e-6 * t)) == {}               # 2.5 ppm below: not
    assert CU._halogen_twins(_led(offset=ISO.D_37CL + 2.5e-6 * t)) == {}


def test_the_swap_records_what_it_replaced():
    led = _led()
    _run(led)
    com = str(led[led.peak_id == "p"].iloc[0].commentary)
    assert "Replaces C12H14O4 [M+^NO3]-" in com and "Cl M+2 offset reads 0.31" in com


def test_a_br_twin_is_found_and_the_nearer_offset_names_the_halogen():
    tw = CU._halogen_twins(_led(r2=0.95, offset=ISO.D_81BR))
    assert tw["p"][3] == "Br"
    assert CU._halogen_twins(_led(r2=0.31, offset=ISO.D_81BR)) == {}     # a Cl ratio at the Br offset: neither


def test_a_si_bearing_ion_s_own_30si_line_is_not_a_cl_twin():
    led = _led(offset=ISO.D_30SI, formula=("C2H6O7Si3", "[M-H]-", "C2H5O7Si3-"))
    assert CU._halogen_twins(led) == {}


def test_a_line_that_may_be_another_ion_s_isotope_line_is_not_a_twin():
    led = _led()
    twin_mz = float(led.loc[led.peak_id == "twin", "mz"].iloc[0])
    extra = L.new_ledger(pd.DataFrame([dict(peak_id="nb", mz=twin_mz - ISO.D_18O, height=5e4, area=50.0,
                                            signal_to_noise=900.0)]))
    led = pd.concat([led, extra], ignore_index=True)
    L.commit_assignment(led, "nb", neutral_formula="C9H14O5", adduct="[M-H]-", ion_formula="C9H13O5-", ion_score=0.9,
                        ppm_error=0.1, pass_no=1, method="test", confidence="Good", commentary="t")
    assert CU._halogen_twins(led) == {}


def test_an_incumbent_with_its_own_label_line_attached_is_not_re_read():
    led = _led()
    extra = L.new_ledger(pd.DataFrame([dict(peak_id="n14", mz=MZ - 0.99703, height=60.0, area=0.06,
                                            signal_to_noise=8.0)]))
    led = pd.concat([led, extra], ignore_index=True)
    L.attach_isotopologue(led, "n14", "p", iso_label="14N")
    assert CU._halogen_twins(led) == {}


def test_the_swaps_reach_the_cleanup_summary(monkeypatch):
    monkeypatch.setattr(CU, "recover_isotope_gated", lambda *a, **k: {"recovered": 2, "halogen_twin_swaps": 1})
    cfg = P.PassConfig()
    out = CU.run_cleanup(None, "S", _led(), X.get_context("ambient-air"), cfg, log=lambda *a: None)
    assert out["halogen_twin_swaps"] == 1
