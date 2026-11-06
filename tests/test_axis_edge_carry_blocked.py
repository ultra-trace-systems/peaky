"""The edge carry in a refused reading's record (assignment/blocked.py).

The labelled rescue notes a reading it refuses off calibration as the peak's blocked
reading, with its z. Under a sub-ppm batch axis model's edge carry
(PassConfig.axis_edge_carry; tests/test_axis_edge_carry.py) that z is read at the
line's own centre -- cal_mu plus the model's edge value past its scope -- the centre
the gate refused it at, so the note and the verdict agree.

Synthetic ledger, injected scorer.
"""

from __future__ import annotations

import pandas as pd

from peaky.assignment import blocked as BL
from peaky.assignment import labeled as LB
from peaky.assignment import ledger as L
from peaky.assignment.passes import config as PCfg
from peaky.chem import chemistry as C
from peaky.chem import contexts as X


def test_the_labelled_rescues_off_cal_note_reads_the_carried_centre():
    neu = "C9H15^NO5"
    mz = C.ion_mz(neu, "[M-H]-")
    led = L.new_ledger(pd.DataFrame({"peak_id": ["p1"], "mz": [mz], "height": [5e4]}))

    def score(client, sid, formulas, allow_partial=True, mechanism_ids=None):
        # -0.3 ppm: 1.0 sigma off cal_mu, but 4.0 sigma off the carried centre (-1.5) there
        return pd.DataFrame([dict(compound_formula=neu, ion_formula="C9H14^NO5-", mechanism_id="[M-H]-",
                                  is_base=is_base, ion_score=0.95, compound_score=0.95, sample_peak_mz=mz,
                                  sample_peak_id=pid, ppm_error=-0.3)
                             for is_base, pid in ((True, "p1"), (False, "p1c")) if neu in formulas])
    cfg = PCfg.PassConfig(height_cutoff_cps=10)
    cfg.cal_mu, cfg.cal_sigma, cfg.mechanism_ids = 0.0, 0.3, None
    cfg.axis_edge_carry = {"reach_da": 30.0, "slack_ppm": 3.0, "below": {"mz": mz - 80.0, "ppm": 0.0},
                           "above": {"mz": mz - 10.0, "ppm": -1.5}}
    rec = BL.start()
    try:
        out = LB.rescue_labeled(None, "s", led, X.get_context("ambient-air"), cfg, adducts=["[M-H]-"],
                                label_isotope="^N", label_max=1, score_fn=score, log=lambda *a: None)
        b = BL.best(rec.blocks["p1"])
    finally:
        BL.stop()
    assert out["rescued"] == 0
    assert b.gate == "labeled:off-cal z=4.0>2.6" and b.neutral == neu
