"""Pass 1 records WHY each target peak it did not commit stayed unexplained
(directors.record_pass1_outcomes -> the ledger's `pass1_outcome` column)."""
import pandas as pd

from peaky import passes as P
from peaky.assignment import ledger as L
from peaky.chem import chemistry as C
from peaky.chem import contexts as X
from peaky.assignment.passes import directors as D


def _ledger(rows):
    return L.new_ledger(pd.DataFrame({"peak_id": [r[0] for r in rows], "mz": [r[1] for r in rows],
                                      "height": [r[2] for r in rows]}))


def _scored(rows):
    """flat scorer rows: (peak_id, formula, adduct, ion_score)"""
    return pd.DataFrame([dict(sample_peak_id=p, compound_formula=f, adduct_label=a, ion_score=s,
                              compound_score=s, is_base=True) for p, f, a, s in rows])


def test_each_way_out_of_pass1_is_named():
    prof = X.run_profile(X.CONTEXTS["ambient-air"], reagent="NO3_15N", instrument_class="orbitrap")
    m_ctx = C.ion_mz("C5H4O3", "[M-H]-")              # DBE/C 0.80: the ambient-air window refuses it
    m_low = C.ion_mz("C7H12O5", "[M-H]-")
    m_unsc = C.ion_mz("C6H10O4", "[M-H]-")
    m_ok = C.ion_mz("C8H12O4", "[M-H]-")
    led = _ledger([("ctx", m_ctx, 5e4), ("low", m_low, 5e4), ("uns", m_unsc, 5e4),
                   ("none", 150.5, 5e4), ("ok", m_ok, 5e4), ("done", 300.0, 5e4)])
    L.commit_assignment(led, "done", neutral_formula="C10H16O8", adduct="[M-H]-", ion_formula="C10H15O8-",
                        ion_score=0.9, ppm_error=0.1, pass_no=1, method="t", confidence="Good", commentary="t")
    grid = {"C5H4O3", "C7H12O5", "C6H10O4", "C8H12O4"}
    formulas = set(D._context_filter(grid, prof))
    assert "C5H4O3" not in formulas
    scored = _scored([("low", "C7H12O5", "[M-H]-", 0.55), ("ok", "C8H12O4", "[M-H]-", 0.85)])
    cfg = P.PassConfig(height_cutoff_cps=100.0)
    tgt = led[led["role"] == L.ROLE_UNEXPLAINED]
    out = D.record_pass1_outcomes(led, tgt, grid, formulas, scored, cfg, ["[M+^NO3]-", "[M-H]-"],
                                  profile=prof, log=lambda *a: None)
    o = dict(zip(led["peak_id"], led[D.PASS1_COL]))
    assert o["ctx"].startswith("pass1 context:") and "DBE/C" in o["ctx"]
    assert o["low"].startswith("pass1 below-tau:") and "C7H12O5" in o["low"] and "0.55" in o["low"]
    assert o["uns"].startswith("pass1 unscored:")
    assert o["none"].startswith("pass1 no-grid:")
    assert o["ok"].startswith("pass1 not-committed:")
    assert pd.isna(o["done"])                          # committed rows carry no pass-1 note
    assert out == {"no-grid": 1, "context": 1, "unscored": 1, "below-tau": 1, "not-committed": 1}


def test_no_targets_writes_an_empty_column():
    led = _ledger([("a", 100.0, 1.0)])
    out = D.record_pass1_outcomes(led, led.iloc[0:0], set(), set(), pd.DataFrame(), P.PassConfig(),
                                  ["[M-H]-"], log=lambda *a: None)
    assert D.PASS1_COL in led.columns and led[D.PASS1_COL].isna().all()
    assert sum(out.values()) == 0
