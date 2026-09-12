"""Offline tests for the noise-edge height gate (passes/config.py): the edge is
the 1st percentile of the sample's picked heights, `cfg.height_cutoff` resolves
to x_edge * edge, an absolute override wins, and the default keeps every picked
peak (but the bottom 1 %) eligible on ANY instrument.
Run: python3 tests/test_edge.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from peaky.assignment import passes as P  # noqa: E402
from peaky.assignment.passes import config as C  # noqa: E402

PASS = FAIL = 0
def check(name, cond, detail=""):
    global PASS, FAIL
    if cond: PASS += 1; print(f"  ok  {name}")
    else: FAIL += 1; print(f"FAIL  {name}  {detail}")


rng = np.random.default_rng(1)
# a TOF-like peak list (edge ~0.8 cps) and an Orbitrap-like one (edge ~700 cps)
tof = np.exp(rng.normal(np.log(3.0), 1.0, 5000)) + 0.6
orbi = np.exp(rng.normal(np.log(3000.0), 1.0, 5000)) + 600.0

e_tof, e_orbi = C.noise_edge(tof), C.noise_edge(orbi)
check("noise_edge = 1st percentile of the heights",
      np.isclose(e_tof, np.percentile(tof, 1.0)) and np.isclose(e_orbi, np.percentile(orbi, 1.0)),
      (e_tof, e_orbi))
check("edge tracks the instrument: TOF ~1 cps, Orbitrap ~1e3 cps (~1000x apart)",
      e_tof < 2.0 and e_orbi > 600.0, (e_tof, e_orbi))
check("noise_edge ignores NaN/inf", np.isclose(C.noise_edge(np.r_[tof, np.nan, np.inf]), e_tof))
check("noise_edge of nothing is None", C.noise_edge([]) is None and C.noise_edge([np.nan]) is None)
check("noise_edge accepts a Series", np.isclose(C.noise_edge(pd.Series(orbi)), e_orbi))

cfg = P.PassConfig()
check("PassConfig has NO absolute height_cutoff default (x_edge 1.0, cps None)",
      cfg.height_cutoff_x_edge == 1.0 and cfg.height_cutoff_cps is None
      and cfg.noise_edge_cps is None)
check("before an edge is stamped the gate is 0 (no gate)", cfg.height_cutoff == 0.0)
cfg.noise_edge_cps = e_tof
check("height_cutoff property = x_edge * edge (TOF)", np.isclose(cfg.height_cutoff, e_tof))
check("default gate keeps 99% of picked TOF peaks eligible (old 100 cps kept ~3%)",
      (tof >= cfg.height_cutoff).mean() > 0.98 and (tof >= 100.0).mean() < 0.05,
      ((tof >= cfg.height_cutoff).mean(), (tof >= 100.0).mean()))
cfg.noise_edge_cps = e_orbi
check("same cfg on the Orbitrap list: gate follows the edge",
      np.isclose(cfg.height_cutoff, e_orbi) and (orbi >= cfg.height_cutoff).mean() > 0.98)
cfg.height_cutoff_x_edge = 2.5
check("x_edge scales the gate", np.isclose(cfg.height_cutoff, 2.5 * e_orbi))
cfg.height_cutoff_cps = 100.0
check("an absolute override wins over the edge", cfg.height_cutoff == 100.0)
check("height_cutoff is read-only (a property, not a field)",
      not hasattr(type(cfg), "__dataclass_fields__") or "height_cutoff" not in type(cfg).__dataclass_fields__)
try:
    P.PassConfig(height_cutoff=100.0)
    check("PassConfig(height_cutoff=) is rejected (use height_cutoff_cps)", False, "accepted")
except TypeError:
    check("PassConfig(height_cutoff=) is rejected (use height_cutoff_cps)", True)

# the call sites read cfg.height_cutoff -- an offline caller with an override
# behaves exactly as before
c2 = P.PassConfig(height_cutoff_cps=100.0)
led = pd.DataFrame({"height": [50.0, 150.0, 99.9, 100.0]})
check("call-site filter `height >= cfg.height_cutoff` with the override",
      (led["height"] >= c2.height_cutoff).tolist() == [False, True, False, True])


def test_all():
    assert FAIL == 0, f"{FAIL} checks failed"


if __name__ == "__main__":
    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)
