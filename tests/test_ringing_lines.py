"""iso_checks.release_ringing_lines: a ringing artifact on a committed reading's isotope line is released to
that reading when the batch series shows it follows the reading's M0, and kept when it follows the ringing
parent. Synthetic ledgers and series, offline."""
import numpy as np
import pandas as pd

from peaky.assignment import cleanup as CL
from peaky.assignment import ledger as L
from peaky.assignment.levels import lines as LN
from peaky.assignment.levels import space as SP
from peaky.batch import iso_checks as IC
from peaky.chem import chemistry as C
from peaky.chem import isotopes as ISO
from peaky.chem.resolution import Resolution

RES = Resolution(coef=0.0016 / 200 ** 1.5, exponent=1.5, offset=0.0)
N, A = "C10H16O3", "[M+NH4]+"
M0 = C.ion_mz(N, A)
H0, HSRC = 8e5, 1.5e6


def _o18():
    lines = LN.cand_lines(SP.ion_counts_of(N, A), RES.fwhm(M0), 0.0, ISO.LABEL_PURITY_15N)
    return next(ln for ln in lines if ln["label"] == "18O")


def _ledger():
    ln = _o18()
    line_mz, src_mz = M0 + ln["d"], M0 + ln["d"] + 0.011
    led = L.new_ledger(pd.DataFrame({"peak_id": ["m0", "line", "src"], "mz": [M0, line_mz, src_mz],
                                     "height": [H0, H0 * ln["ratio"], HSRC]}))
    L.commit_assignment(led, "m0", neutral_formula=N, adduct=A, ion_formula="C10H20NO3+", ion_score=0.95,
                        ppm_error=0.1, pass_no=1, method="t", confidence="Good", commentary="t")
    L.mark_artifact(led, "line", f"{CL.RING_LABEL} {src_mz:.4f} (1500000 cps, 300x brighter, Δ+11 mDa)")
    return led, line_mz, src_mz, ln["ratio"]


def _series(line_mz, src_mz, ratio, follows, n=40):
    rng = np.random.default_rng(0)
    m0 = H0 * np.exp(rng.normal(0, 0.5, n))
    src = HSRC * np.exp(rng.normal(0, 0.5, n))
    base = m0 * ratio if follows == "m0" else src * (H0 * ratio / HSRC)
    line = base * (1 + rng.normal(0, 0.04, n))
    rows = []
    for t in range(n):
        for mz, h in ((M0, m0[t]), (line_mz, line[t]), (src_mz, src[t])):
            rows.append(dict(sample_item_id=f"t{t:03d}", mz=mz, height=h, area=h))
    return pd.DataFrame(rows)


def test_a_line_that_follows_its_reading_is_released_to_it():
    led, line_mz, src_mz, ratio = _ledger()
    changed, summ = IC.release_ringing_lines({"s1": led}, _series(line_mz, src_mz, ratio, "m0"), RES,
                                             log=lambda *a: None)
    row = led.set_index("peak_id").loc["line"]
    assert changed == {"s1"} and summ["released"] == 1 and summ["by_label"] == {"18O": 1}
    assert (row["role"], row["parent_peak_id"], row["iso_label"]) == (L.ROLE_ISO, "m0", "18O")
    assert row["parent_neutral_formula"] == N and "released from the ringing flag" in row["commentary"]
    assert L.validate(led) == []


def test_a_line_locked_to_the_ringing_parent_stays_an_artifact():
    led, line_mz, src_mz, ratio = _ledger()
    changed, summ = IC.release_ringing_lines({"s1": led}, _series(line_mz, src_mz, ratio, "src"), RES,
                                             log=lambda *a: None)
    assert changed == set() and summ["checked"] == 1 and summ["released"] == 0
    assert led.set_index("peak_id").loc["line", "role"] == L.ROLE_ARTIFACT


def test_nothing_runs_without_a_series_or_a_width_model():
    led, *_ = _ledger()
    assert IC.release_ringing_lines({"s1": led}, None, RES)[1]["ran"] is False
    assert IC.release_ringing_lines({"s1": led}, pd.DataFrame({"mz": [1.0], "sample_item_id": ["t"],
                                                               "height": [1.0]}), None)[1]["ran"] is False
