"""Integration tests for the edge-relative height gate: `assign.run` STAMPS the
sample's noise edge onto the shared config and records the RESOLVED gate, and a
height-gated pass then runs off that default (no absolute override anywhere).

Both drive real code paths offline -- the whole IO layer is stubbed (connect /
fetch_peaks / detect_adducts / resolve_mechanism_ids / estimate_offset /
score_candidates), the same oracle-injection style as test_reflists.py, so
`assign.run` itself is exercised end to end rather than a helper standing in
for it.  Run: python3 tests/test_edge_stamping.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from peaky.io import io_mascope as IO          # noqa: E402
from peaky.assignment import ledger as L       # noqa: E402
from peaky.assignment import passes as P       # noqa: E402
from peaky.assignment import reflists as RL    # noqa: E402
from peaky.chem import chemistry as C          # noqa: E402

PASS = FAIL = 0
def check(name, cond, detail=""):
    global PASS, FAIL
    if cond: PASS += 1; print(f"  ok  {name}")
    else: FAIL += 1; print(f"FAIL  {name}  {detail}")


# ===========================================================================
# (1) assign.run stamps the edge and records the resolved gate
# ===========================================================================
# A spread of picked heights whose 1st percentile is well inside the range, so a
# mutant reporting min()/median() instead of p1 cannot pass.
HEIGHTS = np.concatenate([np.linspace(7.0, 60.0, 20), np.linspace(100.0, 9000.0, 80)])
PEAKS = pd.DataFrame({"peak_id": [f"p{i}" for i in range(len(HEIGHTS))],
                      "mz": np.round(np.linspace(120.0, 420.0, len(HEIGHTS)), 4),
                      "height": HEIGHTS})
EDGE = float(np.percentile(HEIGHTS, 1.0))

_orig = {}
def _patch(name, fn):
    _orig[name] = getattr(IO, name, None)
    setattr(IO, name, fn)


_patch("connect", lambda *a, **k: "CLIENT")
_patch("fetch_peaks", lambda client, sid, use_cache=True: _RAW[0].copy())
_patch("detect_adducts", lambda raw: ["[M-H]-"])
_patch("resolve_mechanism_ids", lambda client, names: {})
_patch("estimate_offset", lambda raw: 0.0)
# every pass funnels its server scoring through score_candidates; an empty frame
# means "nothing matched", which leaves the ledger unexplained and the run short.
_patch("score_candidates", lambda *a, **k: pd.DataFrame())

_RAW = [PEAKS]
from peaky.assignment import assign as A  # noqa: E402  (import AFTER the IO stubs)

X_EDGE = 2.5                      # deliberately not 1.0: gate != edge
out = A.run("SID", "ambient-air", cfg=P.PassConfig(height_cutoff_x_edge=X_EDGE),
            log=lambda *a: None)
st = out["stats"]
check("assign.run stamps noise_edge_cps = the 1st percentile of the picked heights",
      np.isclose(st["noise_edge_cps"], EDGE), (st.get("noise_edge_cps"), EDGE))
check("assign.run records the RESOLVED gate as height_gate_cps = x_edge x edge",
      np.isclose(st["height_gate_cps"], X_EDGE * EDGE), (st.get("height_gate_cps"),
                                                         X_EDGE * EDGE))
check("the resolved gate is NOT filed under the knob's name (height_cutoff_cps)",
      "height_cutoff_cps" not in st, sorted(st))
check("the run resolved the gate from the edge, not from an override",
      st["height_gate_cps"] != st["noise_edge_cps"] and X_EDGE != 1.0)

# an absolute override wins and the edge is still measured and reported
out_o = A.run("SID", "ambient-air", cfg=P.PassConfig(height_cutoff_cps=777.0),
              log=lambda *a: None)
check("an absolute override is what gets recorded as the resolved gate",
      out_o["stats"]["height_gate_cps"] == 777.0
      and np.isclose(out_o["stats"]["noise_edge_cps"], EDGE), out_o["stats"])

# fail closed: no measurable edge and no override -> assign.run refuses, naming
# the sample, instead of running every height-gated pass un-floored.
_RAW[0] = PEAKS.assign(height=np.nan)
try:
    A.run("NOHEIGHTS", "ambient-air", log=lambda *a: None)
    check("a sample with no finite heights RAISES rather than running un-gated",
          False, "returned")
except RuntimeError as e:
    check("a sample with no finite heights RAISES rather than running un-gated",
          "NOHEIGHTS" in str(e) and "height_cutoff_cps" in str(e), str(e))
# ... unless the caller supplied an absolute gate
o2 = A.run("NOHEIGHTS", "ambient-air", cfg=P.PassConfig(height_cutoff_cps=50.0),
           log=lambda *a: None)
check("... but an explicit absolute gate still runs with no measurable edge",
      o2["stats"]["height_gate_cps"] == 50.0 and o2["stats"]["noise_edge_cps"] is None,
      o2["stats"])
_RAW[0] = PEAKS

for _k, _v in _orig.items():                   # leave IO as we found it
    if _v is not None:
        setattr(IO, _k, _v)


# ===========================================================================
# (2) a height-gated pass running off the EDGE DEFAULT (no absolute override)
# ===========================================================================
# The reflist rescue commits a "too dim to confirm" lead as tentative when the
# predicted 13C M+1 (0.011 * nC * height) falls BELOW the gate, and leaves a peak
# that is bright enough to show satellites but shows none unexplained. With the
# gate resolved from the stamped edge, that split is what the edge decides.
F_CONF, F_DIM, F_BRIGHT = "C10H16O5", "C9H14O5", "C8H12O5"
RLIST = RL.ReferenceList(
    id="edgelist", system="t", label="Edge test list", data_version="1",
    polarity="negative", native_detection="[M-H]-", applies_to_contexts=("x",),
    references=({"authors": "X", "title": "T"},),
    formulas=frozenset({F_CONF, F_DIM, F_BRIGHT}), radicals=frozenset(),
    conditions_of={f: () for f in (F_CONF, F_DIM, F_BRIGHT)},
    source_file="t.json", always_active=False, meta_of={})


def mh(f):
    return C.ion_mz(f, "[M-H]-")


def oracle(client, sample_id, formulas, *, allow_partial=True, mechanism_ids=None):
    """Server stand-in: every listed formula is a good mass match; only F_CONF
    also has a scored satellite (isotope-confirmed)."""
    rows = []
    def base(f, score):
        cnt = C.parse_formula(f); cnt["H"] = cnt.get("H", 0) - 1
        return dict(compound_formula=f, compound_score=score,
                    ion_formula=C.format_formula(cnt) + "-", ion_score=score,
                    iso_label="M0", is_base=True, iso_score=score,
                    sample_peak_id=f, sample_peak_mz=mh(f),
                    sample_peak_intensity=1e4, ppm_error=0.2)
    for f in formulas:
        rows.append(base(f, 0.90))
        if f == F_CONF:
            r = base(f, 0.90); r.update(is_base=False, iso_label="13C",
                                        sample_peak_mz=mh(f) + 1.00336, iso_score=0.95)
            rows.append(r)
    return pd.DataFrame(rows)


EDGE2 = 100.0                       # stamped noise edge, cps
H_DIM = 500.0                       # 0.011 * 9 * 500  =  49.5 cps  <  gate 100
H_BRIGHT = 1.0e5                    # 0.011 * 8 * 1e5  = 8800  cps  >  gate 100


def rescue(x_edge=1.0, edge=EDGE2):
    """Run the rescue on a fresh ledger with the gate resolved from the EDGE
    (no height_cutoff_cps anywhere). Returns (counts, ledger, cfg)."""
    led = L.new_ledger(pd.DataFrame(
        [("conf", mh(F_CONF), 1.0e5), ("dim", mh(F_DIM), H_DIM),
         ("bright", mh(F_BRIGHT), H_BRIGHT)],
        columns=["peak_id", "mz", "height"]))
    cfg = P.PassConfig(height_cutoff_x_edge=x_edge)
    cfg.noise_edge_cps = edge                    # what assign.run stamps per sample
    cfg.cal_mu, cfg.cal_sigma, cfg.mechanism_ids = 0.0, 0.3, None
    counts = RL.rescue_unexplained_by_reflist(None, "S", led, None, cfg, [RLIST],
                                              ["[M-H]-"], score_fn=oracle,
                                              log=lambda *a: None)
    return counts, led, cfg


counts, led, cfg = rescue()
check("edge default: no absolute override is involved (gate = 1.0 x the stamped edge)",
      cfg.height_cutoff_cps is None and cfg.height_cutoff == EDGE2, cfg.height_cutoff)
check("edge default: the isotope-confirmed lead is CONFIRMED",
      counts["rescued"] == 1 and L.role_of(led, "conf") == L.ROLE_M0
      and led.loc[led.peak_id == "conf", "tier"].iloc[0] == "Assigned", counts)
check("edge default: a predicted 13C satellite BELOW the gate stays tentative",
      counts["tentative"] == 1
      and "dim" in str(led.loc[led.peak_id == "dim", "confidence"].iloc[0])
      and bool(led.loc[led.peak_id == "dim", "below_assignability"].iloc[0]), counts)
check("edge default: a satellite ABOVE the gate that never showed is left unexplained",
      L.role_of(led, "bright") == L.ROLE_UNEXPLAINED, counts)

# the split really is the gate's: raise the same x_edge and the bright peak,
# whose predicted satellite (8800 cps) now sits UNDER the gate, flips to tentative.
counts_hi, led_hi, cfg_hi = rescue(x_edge=100.0)     # gate = 100 x 100 = 10 000 cps
check("raising x_edge moves the gate (edge-relative, not absolute)",
      cfg_hi.height_cutoff == 100.0 * EDGE2)
check("with the gate above its predicted satellite the bright peak flips to tentative",
      counts_hi["tentative"] == 2 and L.role_of(led_hi, "bright") == L.ROLE_M0
      and bool(led_hi.loc[led_hi.peak_id == "bright", "below_assignability"].iloc[0]),
      counts_hi)
# and a 1000x quieter instrument (edge 0.1 cps) confirms nothing changed but the
# edge: the dim peak's satellite is now ABOVE the gate, so it is no longer excused.
counts_lo, led_lo, _ = rescue(edge=0.1)
check("a low-edge instrument gates the same peaks differently (nothing absolute)",
      counts_lo["tentative"] == 0 and L.role_of(led_lo, "dim") == L.ROLE_UNEXPLAINED,
      counts_lo)


def test_all():
    assert FAIL == 0, f"{FAIL} checks failed"


if __name__ == "__main__":
    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)
