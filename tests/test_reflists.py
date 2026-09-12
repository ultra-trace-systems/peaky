"""Offline tests for reflists.py — catalog load, context unlock, and the
rescue-verify decision logic (oracle injected, no network).
Run: python3 tests/test_reflists.py"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from peaky import reflists as RL  # noqa: E402
from peaky import ledger as L     # noqa: E402
from peaky import chemistry as C  # noqa: E402
from peaky import passes as P     # noqa: E402

PASS = FAIL = 0
def check(name, cond, detail=""):
    global PASS, FAIL
    if cond: PASS += 1; print(f"  ok  {name}")
    else: FAIL += 1; print(f"FAIL  {name}  {detail}")


# ---- catalog loads + always_active / context gating ----
cat = RL.load_catalog()
check("catalog loads >=2 lists", len(cat) >= 2, list(cat))
check("contaminants list is always_active",
      any(rl.always_active for rl in cat.values()))
tags = RL.resolve_context_tags("Orange peeling (Br- CIMS)", "Br⁻ CIMS")
check("'orange' batch name unlocks monoterpene context", "monoterpene_ox" in tags, tags)
act = RL.active_lists(cat, context_tags=tags)
check("active set includes a context list + the always-active contaminants",
      len(act) >= 2, [rl.id for rl in act])
check("no context -> still get the always_active contaminants",
      len(RL.active_lists(cat, context_tags=set())) >= 1)

# ---- the DATASET name is metadata too. On a campaign whose batch names describe the
# instrument and the reagent ("... TOF NO3_Br mixed ...") the chemistry lives only in
# the dataset name ("AP oxidation demo-set"), and a resolver that never saw it left
# the alpha-pinene list inactive on alpha-pinene data. The bare abbreviation "AP" is
# still NOT a keyword (soap, grape); the PHRASE "ap oxidation" is.
ds_tags = RL.resolve_context_tags("MIONtof TOF NO3_Br mixed 2026-08-10 - 08-14",
                                  "AP oxidation demo-set", "Br- CIMS")
check("the dataset name unlocks the alpha-pinene contexts when the batch name does not",
      {"ap_ox", "monoterpene_ox"} <= ds_tags, ds_tags)
check("...and the same texts without the dataset name unlock nothing",
      RL.resolve_context_tags("MIONtof TOF NO3_Br mixed 2026-08-10 - 08-14", "", "Br- CIMS") == set())
check("a bare 'AP' is still not a keyword (false positives: soap, grape, AP Low temperature)",
      RL.resolve_context_tags("AP Low temperature", "soap and grape") == set())
check("the phrase variants unlock too",
      "ap_ox" in RL.resolve_context_tags("ap-oxidation chamber")
      and "ap_ox" in RL.resolve_context_tags("AP_oxidation run 3"))
check("the alpha-pinene HOM list activates under the dataset-derived tags",
      any("hom" in rl.id.lower() or "pinene" in rl.id.lower() or "monoterpene" in rl.id.lower()
          for rl in RL.active_lists(cat, context_tags=ds_tags)),
      [rl.id for rl in RL.active_lists(cat, context_tags=ds_tags)])

# ---- isoprene oxidation list (Wennberg 2018): loads, gated by the isoprene context ----
iso = cat.get("isoprene_ox_wennberg2018")
check("isoprene list is in the catalog with 27 closed-shell neutrals", iso is not None and len(iso.formulas) == 27, iso)
check("isoprene list carries the dihydroxy-dinitrate and ISOPN", iso is not None and {"C5H10N2O8", "C5H9NO4"} <= set(iso.formulas), None)
iso_tags = RL.resolve_context_tags("Isoprene + OH chamber (NO3- CIMS)")
check("'isoprene' batch name unlocks the isoprene context", "isoprene_ox" in iso_tags, iso_tags)
check("isoprene list active under the isoprene context", any(rl.id == "isoprene_ox_wennberg2018" for rl in RL.active_lists(cat, context_tags=iso_tags)), None)
check("isoprene list NOT active under a monoterpene-only context", not any(rl.id == "isoprene_ox_wennberg2018" for rl in RL.active_lists(cat, context_tags={"monoterpene_ox"})), None)

# `activate` is the one step every entry point runs (batch, `peaky assign`, MCP):
# resolve tags -> select lists, returning both so the caller can log the unlock.
_a_lists, _a_tags = RL.activate("Orange peeling (Br- CIMS)", "Br⁻ CIMS")
check("activate() == resolve_context_tags + active_lists",
      _a_tags == tags and {rl.id for rl in _a_lists} == {rl.id for rl in act},
      (sorted(_a_tags), [rl.id for rl in _a_lists]))
# a single-sample run's only metadata is a context label, which names no chemistry
_c_lists, _c_tags = RL.activate("ambient-air")
check("a context label alone unlocks nothing chemistry-specific", _c_tags == set(), _c_tags)
check("...but still activates the contaminants (so a lone sample is never list-free)",
      _c_lists and all(rl.always_active for rl in _c_lists), [rl.id for rl in _c_lists])


# ---- rescue-verify decision logic (injected oracle) ----
def mk(rows):
    return L.new_ledger(pd.DataFrame(rows, columns=["peak_id", "mz", "height"]))


def mh(formula):                              # [M-H]- ion m/z
    return C.ion_mz(formula, "[M-H]-")


# three formulas; each becomes one unexplained peak at its [M-H]- mass
F_CONF, F_DIM, F_BRIGHT = "C10H16O5", "C9H14O5", "C8H12O5"
rl = RL.ReferenceList(
    id="testlist", system="t", label="Test list", data_version="1",
    polarity="negative", native_detection="[M-H]-", applies_to_contexts=("x",),
    references=({"authors": "X", "title": "T"},),
    formulas=frozenset({F_CONF, F_DIM, F_BRIGHT}), radicals=frozenset(),
    conditions_of={f: () for f in (F_CONF, F_DIM, F_BRIGHT)},
    source_file="t.json", always_active=False, meta_of={})
led = mk([("conf", mh(F_CONF), 1.0e5),      # bright + isotopes confirmed -> M0
          ("dim", mh(F_DIM), 120.0),        # too dim to show isotopes -> tentative
          ("bright", mh(F_BRIGHT), 1.0e5)])  # bright but isotopes absent -> leave


def oracle(client, sample_id, formulas, *, allow_partial=True, mechanism_ids=None):
    rows = []
    def base(f, score):
        cnt = C.parse_formula(f); cnt["H"] = cnt.get("H", 0) - 1     # [M-H]- ion
        return dict(compound_formula=f, compound_score=score,
                    ion_formula=C.format_formula(cnt) + "-",
                    ion_score=score, iso_label="M0", is_base=True, iso_score=score,
                    sample_peak_id=f, sample_peak_mz=mh(f), sample_peak_intensity=1e4,
                    ppm_error=0.2)
    if F_CONF in formulas:
        rows.append(base(F_CONF, 0.90))
        r = base(F_CONF, 0.90); r.update(is_base=False, iso_label="13C",
                                          sample_peak_mz=mh(F_CONF) + 1.00336, iso_score=0.95)
        rows.append(r)
    if F_DIM in formulas:
        rows.append(base(F_DIM, 0.85))       # good mass match, NO iso row
    if F_BRIGHT in formulas:
        rows.append(base(F_BRIGHT, 0.85))    # good mass match, NO iso row
    return pd.DataFrame(rows)


cfg = P.PassConfig(height_cutoff_cps=100.0); cfg.cal_mu, cfg.cal_sigma = 0.0, 0.3; cfg.mechanism_ids = None  # offline: absolute gate (the synthetic heights are calibrated to 100 cps)
out = RL.rescue_unexplained_by_reflist(None, "S", led, None, cfg, [rl], ["[M-H]-"],
                                       score_fn=oracle, log=lambda *a: None)
check("rescue: 1 confirmed + 1 tentative", out == {"rescued": 1, "tentative": 1}, out)
check("rescue: confirmed peak -> M0 with the listed formula",
      L.role_of(led, "conf") == L.ROLE_M0
      and led[led.peak_id == "conf"].iloc[0]["neutral_formula"] == F_CONF)
_dim = led[led.peak_id == "dim"].iloc[0]
check("rescue: dim peak -> tentative Candidate (literature, dim) + below_assignability",
      _dim["role"] == L.ROLE_M0 and "dim" in str(_dim["confidence"])
      and bool(_dim["below_assignability"]))
check("rescue: bright-but-unconfirmed peak left unexplained (isotopes expected, absent)",
      L.role_of(led, "bright") == L.ROLE_UNEXPLAINED)
check("rescue: ledger valid", L.validate(led) == [])

# off-cal match is rejected even if mass-near
led2 = mk([("oc", mh(F_CONF), 1.0e5)])
cfg2 = P.PassConfig(height_cutoff_cps=100.0); cfg2.cal_mu, cfg2.cal_sigma = 5.0, 0.3   # peak sits ~17 sigma off
out2 = RL.rescue_unexplained_by_reflist(None, "S", led2, None, cfg2, [rl], ["[M-H]-"],
                                        score_fn=oracle, log=lambda *a: None)
check("rescue: off-calibration match rejected", out2["rescued"] == 0 and out2["tentative"] == 0, out2)


def test_all():
    assert FAIL == 0, f"{FAIL} checks failed"


if __name__ == "__main__":
    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)
