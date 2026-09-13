"""Offline tests for siloxane.py. Run: python3 tests/test_siloxane.py"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from peaky import siloxane as SX  # noqa: E402
from peaky import ledger as L     # noqa: E402
from peaky import chemistry as C  # noqa: E402
from peaky import contexts as X   # noqa: E402
from peaky import passes as P     # noqa: E402

PASS = FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok  {name}")
    else:
        FAIL += 1
        print(f"FAIL  {name}  {detail}")


# --- _find_ladders: detect a +C2H6OSi (74.0188) chain ---
base = 462.1455
mzs = [base, base + SX.UNIT_MASS, base + 2 * SX.UNIT_MASS, 300.0, 305.123]
hts = [1e5, 8e4, 6e4, 1e5, 9e4]
chains = SX._find_ladders(mzs, hts, min_height=100)
check("finds the 3-member +C2H6OSi ladder", len(chains) == 1 and len(chains[0]) == 3,
      [[round(mzs[i], 3) for i in ch] for ch in chains])
check("ladder ignores the unrelated peaks (300/305)",
      all(mzs[i] >= base - 0.1 for ch in chains for i in ch))
# a too-short run (2 members) is not a ladder
short = SX._find_ladders([100.0, 100.0 + SX.UNIT_MASS, 500.0], [1e5, 1e5, 1e5], min_height=100)
check("a 2-member run is not a ladder", len(short) == 0, short)

# --- assign_siloxane_ladder: commit on Si-isotope corroboration, displace an
#     UNLOCKED monster, never override a LOCKED id ---
PROF = X.get_context("uronium")
cfg = P.PassConfig(height_cutoff_cps=100.0)
cfg.cal_mu, cfg.cal_sigma = -2.45, 0.27          # the uronium offset
cfg.mechanism_ids = ["m"]

FORMS = ["C12H39NO6Si6", "C14H45NO7Si7", "C16H51NO8Si8"]   # +C2H6OSi ladder


def _mzp(f):                                     # [M+H]+ at -2.45 ppm
    t = C.ion_mz(f, "[M+H]+")
    return t * (1 - 2.45e-6)


pmz = {f: _mzp(f) for f in FORMS}
m1 = {f: pmz[f] + 0.9997 for f in FORMS}         # the 29Si M+1 peak
def _m1h(f):                                     # realistic 29Si+13C M+1 height for Si_k C_n
    c = C.parse_formula(f)
    return round((c.get("Si", 0) * 0.0468 + c.get("C", 0) * 0.0107) * 1e5)
rows = []
for i, f in enumerate(FORMS):
    rows.append({"peak_id": f"m0_{i}", "mz": round(pmz[f], 5), "height": 1e5})
    rows.append({"peak_id": f"m1_{i}", "mz": round(m1[f], 5), "height": _m1h(f)})
led = L.new_ledger(pd.DataFrame(rows))
pid_of = {round(pmz[f], 5): f"m0_{i}" for i, f in enumerate(FORMS)}
m1pid_of = {round(m1[f], 5): f"m1_{i}" for i, f in enumerate(FORMS)}
# member 1 (536) carries an UNLOCKED CHON monster; member 2 (610) a LOCKED one
L.commit_assignment(led, "m0_1", neutral_formula="C14H21NO15", adduct="[M+H]+",
                    ion_formula="C14H22NO15", ion_score=0.85, ppm_error=-2.4,
                    pass_no=1, method="cheminfo+grid", confidence="Good",
                    commentary="unlocked monster")
L.commit_assignment(led, "m0_2", neutral_formula="C16H27NO16", adduct="[M+H]+",
                    ion_formula="C16H28NO16", ion_score=0.9, ppm_error=-2.4,
                    pass_no=1, method="cheminfo+grid", confidence="High",
                    commentary="LOCKED id")
L.lock_peaks(led, ["m0_2"])


def fake_score(client, sid, formulas, *, mechanism_ids=None, allow_partial=False, **kw):
    out = []
    for f in FORMS:
        if f not in formulas:
            continue
        t = C.ion_mz(f, "[M+H]+")
        ion = C.format_formula({**C.parse_formula(f),
                                "H": C.parse_formula(f).get("H", 0) + 1}) + "+"
        out.append(dict(compound_formula=f, compound_score=0.80, ion_formula=ion,
                        ion_score=0.80, mechanism_id="m", iso_label="M0", is_base=True,
                        theo_mz=t, rel_abundance=1.0, iso_score=0.80,
                        sample_peak_id=pid_of[round(pmz[f], 5)], sample_peak_mz=pmz[f],
                        sample_peak_intensity=1e5, ppm_error=(pmz[f] - t) / t * 1e6,
                        abundance_error=0.0))
        out.append(dict(compound_formula=f, compound_score=0.80, ion_formula=ion,
                        ion_score=0.80, mechanism_id="m", iso_label="29Si", is_base=False,
                        theo_mz=t + 0.9997, rel_abundance=0.28, iso_score=0.72,
                        sample_peak_id=m1pid_of[round(m1[f], 5)], sample_peak_mz=m1[f],
                        sample_peak_intensity=2.8e4, ppm_error=0.1, abundance_error=0.02))
    return pd.DataFrame(out)


s = SX.assign_siloxane_ladder(None, "SID", led, PROF, cfg, adducts=["[M+H]+"],
                              score_fn=fake_score, log=lambda *a: None)
check("ladder detected + 2 members committed (3rd is locked)",
      s["ladders"] >= 1 and s["committed"] == 2 and s["displaced"] == 1, s)
check("commits the unexplained ladder member (462)",
      L.role_of(led, "m0_0") == L.ROLE_M0
      and led.loc[led.peak_id == "m0_0", "neutral_formula"].iloc[0] == "C12H39NO6Si6", s)
check("DISPLACES the unlocked CHON monster (536 -> siloxane)",
      led.loc[led.peak_id == "m0_1", "neutral_formula"].iloc[0] == "C14H45NO7Si7"
      and s["displaced"] >= 1, s)
check("does NOT override the LOCKED id (610 stays the monster)",
      led.loc[led.peak_id == "m0_2", "neutral_formula"].iloc[0] == "C16H27NO16", s)
check("committed members are locked + method siloxane:ladder",
      L.is_locked(led, "m0_0")
      and led.loc[led.peak_id == "m0_0", "method"].iloc[0] == "siloxane:ladder")
check("29Si satellite attached as iso_child",
      L.role_of(led, "m1_0") == L.ROLE_ISO and s["iso_attached"] >= 1, s)
check("ledger valid after siloxane pass", L.validate(led) == [], L.validate(led))

# inert when the context forbids Si (max_Si < 3)
led2 = L.new_ledger(pd.DataFrame(rows))
s2 = SX.assign_siloxane_ladder(None, "SID", led2, X.get_context("ambient-air"), cfg,
                               score_fn=fake_score, log=lambda *a: None)
check("inert when context forbids Si (ambient max_Si=1)", s2["committed"] == 0, s2)

# --- Si-count intensity gate: a too-small 29Si M+1 -> skipped (the C10H18O11 case) ---
check("_m1_ratio: present M+1 -> ratio", abs(SX._m1_ratio([200.0, 200.99957], [1e5, 3e4], 200.0, 1e5) - 0.3) < 1e-6)
check("_m1_ratio: absent M+1 -> 0.0", SX._m1_ratio([200.0, 250.0], [1e5, 1e5], 200.0, 1e5) == 0.0)
rows3 = []                                        # Si6 ladder but M+1 only ~13% (needs ~41%)
for i, f in enumerate(FORMS):
    rows3.append({"peak_id": f"m0_{i}", "mz": round(pmz[f], 5), "height": 1e5})
    rows3.append({"peak_id": f"m1_{i}", "mz": round(m1[f], 5), "height": 1.3e4})  # under-intensity
led3 = L.new_ledger(pd.DataFrame(rows3))
s3 = SX.assign_siloxane_ladder(None, "SID", led3, PROF, cfg, adducts=["[M+H]+"],
                               score_fn=fake_score, log=lambda *a: None)
check("Si-intensity gate: under-intensity M+1 -> NOT committed (over-claimed Si skipped)",
      s3["committed"] == 0 and s3.get("si_underclaimed", 0) >= 1, s3)

def test_all():
    assert FAIL == 0, f"{FAIL} checks failed"


if __name__ == "__main__":
    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)
