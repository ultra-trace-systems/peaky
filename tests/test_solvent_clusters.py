"""Offline tests for assignment/solvent_clusters.py -- the source-solvent cluster
channel (pure module: no client, no network).

The spectrum under test is the one the family was built for: a positive
charge-transfer source running on ethanol vapour, where the proton-bound ethanol
dimer is the single brightest unexplained ion and four more cluster rungs sit
around it. Decoy controls matter more than the positive cases here, because the
whole risk of this family is claiming a covalent molecule as a cluster: with the
ethanol monomer removed, with it present but weak, or with the ladder broken,
NOTHING may commit.

Run: python3 tests/test_solvent_clusters.py
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from peaky.assignment import solvent_clusters as SC  # noqa: E402
from peaky.assignment import ledger as L             # noqa: E402
from peaky.assignment import passes as P             # noqa: E402
from peaky.assignment import tiers as TI             # noqa: E402
from peaky.chem import chemistry as C                # noqa: E402
from peaky.chem import contexts as X                 # noqa: E402

PASS = FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok  {name}")
    else:
        FAIL += 1
        print(f"FAIL  {name}  {detail}")


EASYIC = X.get_context("easyic")
URONIUM = X.get_context("uronium")
AMBIENT = X.get_context("ambient-air")


# --- what a context opens -------------------------------------------------
check("easyic declares the four source solvents",
      [s.formula for s in SC.solvents_for(EASYIC)]
      == ["H2O", "CH4O", "C2H6O", "C3H6O"])
check("easyic has BOTH carriers (it abstracts hydride)",
      SC.carriers_for(EASYIC) == ("H", "-H"))
check("uronium opens the proton ladder only (a urea reagent protonates)",
      bool(SC.solvents_for(URONIUM)) and SC.carriers_for(URONIUM) == ("H",))
check("a negative-mode context declares no source solvents (channel off)",
      SC.solvents_for(AMBIENT) == ())


# --- the library ----------------------------------------------------------
DET = {"C2H6O": {"height": 3.1e7, "frac": 0.31, "name": "ethanol", "channels": []},
       "C3H6O": {"height": 8.3e5, "frac": 0.0082, "name": "acetone", "channels": []}}
def _lib(det=None, profile=None, **kw):
    """The library as the pass builds it: covalent alternatives are worked out
    against the CONTEXT's own ion forms, which is what makes them grow when the
    source gains a channel."""
    profile = profile or EASYIC
    return SC.build_library(det if det is not None else DET,
                            SC.solvents_for(profile), SC.carriers_for(profile),
                            channels=profile.reagent_adducts, **kw)


LIB = {c.ion_formula: c for c in _lib()}

dimer = LIB.get("C4H13O2+")
check("the proton-bound ethanol dimer is enumerated at 93.0910",
      dimer is not None and abs(dimer.mz - 93.09101) < 1e-4,
      None if dimer is None else dimer.mz)
check("...reported as ethanol on a cluster adduct, not as a C4 neutral",
      dimer.core == "C2H6O" and dimer.adduct == "[M+(C2H6O)H]+", dimer.adduct)
check("...and it has NO covalent reading (its neutral would be DBE -1)",
      dimer.covalent == () and C.dbe("C4H12O2") == -1.0, dimer.covalent)
check("the DBE gate really does refuse that neutral (it is not being relaxed)",
      C.dbe_ok("C4H12O2")[0] is False)

hyd = LIB.get("C4H11O2+")
check("the hydride-bound dimer is enumerated at 91.0754",
      hyd is not None and abs(hyd.mz - 91.07536) < 1e-4)
check("...and it DOES carry the covalent butanediol reading as an alternative",
      ("C4H10O2", "[M+H]+") in hyd.covalent, hyd.covalent)

mixed = LIB.get("C2H9O2+")
check("the EtOH/water mixed cluster is enumerated at 65.0597 with no covalent read",
      mixed is not None and abs(mixed.mz - 65.05971) < 1e-4 and mixed.covalent == ())

cond = LIB.get("C4H9O+")
check("the dimer's -H2O condensation rung is enumerated at 73.0648",
      cond is not None and cond.water_loss == 1 and abs(cond.mz - 73.06479) < 1e-4)
_order = [c.ion_formula for c in _lib()]
check("...ordered AFTER its parent rung, so the parent can license it",
      _order.index("C4H11O2+") < _order.index("C4H9O+"))
# the covalent competitors are the SOURCE's, not a fixed pair: EasyIC's
# [M-CH3]+ channel (the methylsiloxanes' quantifier) makes C4H9O+ the methyl
# loss of pentanol as well as protonated C4H8O and hydride-abstracted C4H10O
check("covalent alternatives follow the context's own ion forms",
      ("C5H12O", "[M-CH3]+") in cond.covalent
      and {a for _f, a in cond.covalent} == {"[M+H]+", "[M-H]+", "[M-CH3]+"},
      cond.covalent)
check("...and a reagent-CLUSTER channel in that list drops out on its own",
      all(a in ("[M+H]+", "[M-H]+", "[M-CH3]+")
          for c in LIB.values() for _f, a in c.covalent),
      sorted({a for c in LIB.values() for _f, a in c.covalent}))
_ur = {c.ion_formula: c for c in _lib(profile=URONIUM)}
check("a channel the source does NOT have contributes no alternative",
      _ur["C4H11O+"].covalent == (("C4H10O", "[M+H]+"),), _ur["C4H11O+"].covalent)
check("...and a source with no hydride channel builds no hydride rung at all",
      "C4H11O2+" not in _ur and "C4H13O2+" in _ur, sorted(_ur))

# the 57.0335 bug: acetone + water, hydride, then -H2O is EXACTLY [acetone-H]+.
# Enumerating it would let the family steal the very monomer it anchors on.
_monomers = {C.format_formula(SC._counts((f,), carrier, 0)) + "+"
             for f in ("H2O", "CH4O", "C2H6O", "C3H6O") for carrier in ("H", "-H")}
check("no library ion is a bare monomer channel (a water rung + water loss is a no-op)",
      not (set(LIB) & _monomers), sorted(set(LIB) & _monomers))
check("water is capped at one per cluster (it has no monomer to anchor on)",
      all(sum(1 for u in c.units if u == "H2O") <= SC.MAX_WATER for c in LIB.values()))
check("no rung is a lone molecule",
      all(c.n_units >= 2 for c in LIB.values()))

# the adduct label has to round-trip through BOTH grammars the ledger is read by
check("the adduct label reproduces the ion composition via tiers._ion_counts",
      all(C.format_formula(TI._ion_counts(c.core, c.adduct)) + "+" == c.ion_formula
          for c in LIB.values()),
      [c.adduct for c in LIB.values()
       if C.format_formula(TI._ion_counts(c.core, c.adduct)) + "+" != c.ion_formula])
check("...and the registered shift reproduces the rung mass via chemistry.ion_mz",
      all(abs(C.ion_mz(c.core, c.adduct) - c.mz) < 1e-9 for c in LIB.values()))
check("repeated units are written out, never '(C2H6O)2' (which parses as C2H6O2)",
      "[M+(C2H6O)(C2H6O)H]+" in C.ADDUCT_SHIFTS
      and C.format_formula(TI._ion_counts("C2H6O", "[M+(C2H6O)(C2H6O)H]+"))
      == "C6H19O3")

# in-family mass degeneracy: 2x ethanol (-H) and acetone+methanol (+H) are one ion
DET3 = dict(DET, CH4O={"height": 1.0e5, "frac": 0.001, "name": "methanol",
                       "channels": []})
LIB3 = {c.ion_formula: c for c in _lib(DET3, max_units=2)}
check("a same-composition multiset is collapsed to the best-evidenced reading",
      LIB3["C4H11O2+"].units == ("C2H6O", "C2H6O"), LIB3["C4H11O2+"].units)
check("...with the reading it beat kept as an alias, not silently dropped",
      ("C3H6O", "CH4O") in LIB3["C4H11O2+"].aliases, LIB3["C4H11O2+"].aliases)

# register_adduct refuses to move a label already in use
try:
    C.register_adduct("[M+(C2H6O)H]+", 0.0)
    _refused = False
except ValueError:
    _refused = True
check("register_adduct refuses a conflicting redefinition", _refused)
check("...but re-registering the same shift is a no-op",
      C.register_adduct("[M+(C2H6O)H]+", C.ADDUCT_SHIFTS["[M+(C2H6O)H]+"])
      == "[M+(C2H6O)H]+")


# --- the commit pass ------------------------------------------------------
# an ethanol-dominated EasyIC spectrum, masses and heights from a real run
SPECTRUM = [
    ("etoh_hyd", 45.033541, 2.718e7),    # [C2H6O-H]+   ethanol's main channel
    ("etoh_h", 47.049139, 4.011e6),      # [C2H6O+H]+
    ("acetone", 59.049124, 8.251e5),     # [C3H6O+H]+
    ("mixed", 65.059693, 6.201e4),       # EtOH.H3O+
    ("cond", 73.064847, 1.540e6),        # (EtOH)2-H-H2O
    ("hyd2", 91.075429, 9.650e5),        # [EtOH-H]+.EtOH
    ("dimer", 93.091136, 6.541e6),       # (EtOH)2+H
    ("other", 120.080000, 5.0e4),        # an unrelated ion
]


def _ledger(rows=SPECTRUM):
    return L.new_ledger(pd.DataFrame(
        [{"peak_id": p, "mz": m, "height": h} for p, m, h in rows]))


def _cfg():
    cfg = P.PassConfig(height_cutoff_cps=100.0)
    cfg.prior_offset = 0.0
    return cfg


def _run(rows=SPECTRUM, profile=EASYIC, **kw):
    led = _ledger(rows)
    out = SC.assign_solvent_clusters(led, profile, _cfg(), log=lambda *a: None, **kw)
    return led, out


led, out = _run()
by_pid = {r["peak_id"]: r for _, r in led.iterrows()}
check("the ethanol dimer is claimed (it was the brightest unexplained ion)",
      by_pid["dimer"]["role"] == L.ROLE_M0
      and by_pid["dimer"]["ion_formula"] == "C4H13O2+")
check("...committed as ethanol on the cluster channel, never as a C4 neutral",
      by_pid["dimer"]["neutral_formula"] == "C2H6O"
      and by_pid["dimer"]["adduct"] == "[M+(C2H6O)H]+")
check("...as a known species, since no covalent reading of it exists",
      by_pid["dimer"]["method"] == "known:solvent_cluster")
check("the EtOH/water mixed cluster is claimed too",
      by_pid["mixed"]["role"] == L.ROLE_M0
      and by_pid["mixed"]["adduct"] == "[M+(H2O)H]+")
check("the hydride dimer is claimed, but on the covalent-alias method",
      by_pid["hyd2"]["method"] == "cluster:solvent"
      and by_pid["hyd2"]["ion_formula"] == "C4H11O2+")
check("...with the butanediol reading recorded as an alternative, not discarded",
      "C4H10O2" in str(by_pid["hyd2"]["alternatives"]))
check("the -H2O condensation rung is claimed off its committed parent",
      by_pid["cond"]["method"] == "cluster:solvent"
      and by_pid["cond"]["adduct"] == "[M+(C2H6O)-H-H2O]+")
check("every cluster commit is LOCKED, so pass 1 cannot re-read it covalently",
      all(bool(by_pid[p]["locked"]) for p in ("dimer", "mixed", "hyd2", "cond")))
check("the monomer channels are NOT claimed (n=1 is never this family's)",
      by_pid["etoh_hyd"]["role"] == L.ROLE_UNEXPLAINED
      and by_pid["etoh_h"]["role"] == L.ROLE_UNEXPLAINED
      and by_pid["acetone"]["role"] == L.ROLE_UNEXPLAINED)
check("an unrelated ion is left alone",
      by_pid["other"]["role"] == L.ROLE_UNEXPLAINED)
check("the pass reports which solvents it found", set(out["solvents"]) == {"C2H6O",
                                                                           "C3H6O"},
      out["solvents"])
check("...and splits its commits by evidence class",
      out["committed"] == 4 and out["assigned_tier"] == 2
      and out["candidate_tier"] == 2, out)
check("the ledger stays structurally valid", L.validate(led) == [], L.validate(led))

# the commentary has to carry the evidence, not just the verdict
_note = str(by_pid["dimer"]["commentary"])
check("commentary names the ladder step it rests on",
      "exact C2H6O" in _note and "47.049" in _note, _note)
check("commentary says WHY the grid may not reach it",
      "C4H12O2" in _note and "DBE -1" in _note, _note)


# --- decoy controls: the gates are what make this safe --------------------
NO_MONOMER = [r for r in SPECTRUM if r[0] not in ("etoh_hyd", "etoh_h")]
led2, out2 = _run(NO_MONOMER)
check("DECOY no ethanol monomer -> not one ethanol rung is claimed",
      out2["committed"] == 0 and out2["solvents"].get("C2H6O") is None, out2)
check("...so the dimer peak stays unexplained rather than becoming a guess",
      str(led2.loc[led2["peak_id"] == "dimer", "role"].iloc[0])
      == L.ROLE_UNEXPLAINED)

# monomer present, but a trace: a SOURCE solvent is a major ion by definition
WEAK = [("etoh_hyd", 45.033541, 500.0), ("etoh_h", 47.049139, 300.0),
        ("dimer", 93.091136, 4.0e5), ("big", 200.1, 1.0e8)]
_l3, out3 = _run(WEAK)
check("DECOY a trace ethanol monomer (<0.2 % of signal) licenses nothing",
      out3["committed"] == 0, out3)

# ladder broken: the dimer is there, the monomer is strong, but the rung the
# condensation product needs was never picked
BROKEN = [r for r in SPECTRUM if r[0] != "hyd2"]
_l4, out4 = _run(BROKEN)
check("DECOY a missing parent rung blocks the condensation ion",
      str(_l4.loc[_l4["peak_id"] == "cond", "role"].iloc[0]) == L.ROLE_UNEXPLAINED)
check("...while the rungs that DO have their parent still commit",
      str(_l4.loc[_l4["peak_id"] == "dimer", "role"].iloc[0]) == L.ROLE_M0)

# a cluster reading may only OVERRULE a legal covalent molecule when it is a
# major ion: the same ladder, scaled down, must not take a trace peak off the
# grid (the gin-headspace C5H10O2/C6H12O2 esters that motivated the floor)
TRACE = [("etoh_hyd", 45.033541, 2.718e7), ("etoh_h", 47.049139, 4.011e6),
         ("hyd2", 91.075429, 300.0), ("dimer", 93.091136, 6.541e6)]
_lt, outt = _run(TRACE)
check("DECOY a trace rung may not displace its covalent reading",
      str(_lt.loc[_lt["peak_id"] == "hyd2", "role"].iloc[0]) == L.ROLE_UNEXPLAINED,
      outt)
check("...while a rung with NO covalent reading keeps the ordinary height gate",
      str(_lt.loc[_lt["peak_id"] == "dimer", "role"].iloc[0]) == L.ROLE_M0)
TRACE_MIXED = [("etoh_hyd", 45.033541, 2.718e7), ("etoh_h", 47.049139, 4.011e6),
               ("mixed", 65.059693, 300.0), ("big", 300.2, 1.0e7)]
_lm, _ = _run(TRACE_MIXED)
check("...even a trace one, because its alternative is 'unexplained', not a molecule",
      str(_lm.loc[_lm["peak_id"] == "mixed", "role"].iloc[0]) == L.ROLE_M0)

# a WATER rung may never displace a covalent reading, however bright. Modelled
# on an orange-peel headspace where this cost 11 of 18 files an Assigned
# C3H6O2 [M+H]+: water is anchored by nothing, so the whole claim is its
# 18.0106 step -- and 26 % of the peaks in that spectrum had a +H2O partner
# within 4 ppm, against 7.7 % for +C2H6O.
WATERY = [("acetone_hyd", 57.033491, 3.0e3), ("acetone_h", 59.049141, 9.4e4),
          ("aq", 75.044056, 4.8e3), ("bulk", 150.1, 2.0e6)]
_lw, outw = _run(WATERY)
check("DECOY a water rung may not displace a covalent reading, however bright",
      str(_lw.loc[_lw["peak_id"] == "aq", "role"].iloc[0]) == L.ROLE_UNEXPLAINED,
      outw)
check("...and that is a rule about WATER, not about the rung being small",
      _lw.loc[_lw["peak_id"] == "aq", "height"].iloc[0]
      > SC.ALIAS_MIN_RUNG_FRAC * _lw["height"].sum())
# ...but a water rung with NO covalent reading is still welcome: its implied
# neutral is DBE -1, so there is no molecule for it to take
check("a water rung with no covalent reading still commits (nothing to displace)",
      LIB["C2H9O2+"].covalent == ()
      and str(led.loc[led["peak_id"] == "mixed", "role"].iloc[0]) == L.ROLE_M0)
check("...while the water rung that DOES have one is refused on that ground",
      LIB["C3H7O2+"].covalent != ())

# a condensation ion far brighter than its parent is a real analyte on that mass
LOUD = [(p, m, 3.0e7 if p == "cond" else h) for p, m, h in SPECTRUM]
_l5, out5 = _run(LOUD)
check("DECOY a -H2O ion 30x its parent is left for the covalent reading",
      str(_l5.loc[_l5["peak_id"] == "cond", "role"].iloc[0]) == L.ROLE_UNEXPLAINED)

# off-ladder spacing: the guard has to be wired, not just described
_saved, SC.LADDER_PPM = SC.LADDER_PPM, 0.01
_l6, out6 = _run()
SC.LADDER_PPM = _saved
check("DECOY an off-exact rung spacing refuses the rung",
      out6["committed"] == 0, out6)

# a context that declares no solvents is a hard no-op
_l7, out7 = _run(profile=AMBIENT)
check("DECOY a context with no source solvents commits nothing",
      out7["committed"] == 0 and (_l7["role"] == L.ROLE_UNEXPLAINED).all())

# a uronium-style source has no hydride channel, so no hydride ladder
_l8, out8 = _run(profile=URONIUM)
check("a proton-only source claims the proton rungs...",
      str(_l8.loc[_l8["peak_id"] == "dimer", "role"].iloc[0]) == L.ROLE_M0)
check("...and never the hydride ones (that source does not abstract H-)",
      str(_l8.loc[_l8["peak_id"] == "hyd2", "role"].iloc[0]) == L.ROLE_UNEXPLAINED)

# a locked peak (a pass-0 known species) is not claimable
led9 = _ledger()
L.lock_peaks(led9, ["dimer"])
out9 = SC.assign_solvent_clusters(led9, EASYIC, _cfg(), log=lambda *a: None)
check("a peak locked by an earlier pass is left alone",
      str(led9.loc[led9["peak_id"] == "dimer", "role"].iloc[0]) == L.ROLE_UNEXPLAINED
      and out9["committed"] == 3, out9)


# --- the tier engine reads both evidence classes --------------------------
led10, _ = _run()
cfg10 = _cfg()
P.complete_isotope_envelopes(led10, cfg10, log=lambda *a: None)
TI.apply_tiers(led10, cfg=cfg10)
t = {r["peak_id"]: (r["tier"], str(r["tier_reason"])) for _, r in led10.iterrows()}
check("a cluster with no covalent reading is Assigned",
      t["dimer"][0] == "Assigned", t["dimer"])
check("...for the cluster's OWN reason, not the halide own-twin one",
      "source-solvent cluster" in t["dimer"][1] and "own-twin" not in t["dimer"][1],
      t["dimer"][1])
check("a cluster with a covalent alias is capped at Candidate",
      t["hyd2"][0] == "Candidate" and t["cond"][0] == "Candidate",
      (t["hyd2"], t["cond"]))
check("...and the reason names the isomer problem, not a score failure",
      "MS1 cannot discriminate" in t["hyd2"][1]
      and "below the identification bar" not in t["hyd2"][1], t["hyd2"][1])

# the dimer's 13C satellite -- the second peak the real run left unexplained --
# is claimed by the ordinary envelope sweep off the cluster's ion formula
SAT = SPECTRUM + [("dimer_13c", 93.091136 + 1.00336, 3.187e5)]
led11, _ = _run(SAT)
P.complete_isotope_envelopes(led11, _cfg(), log=lambda *a: None)
_sat = led11.loc[led11["peak_id"] == "dimer_13c"].iloc[0]
check("the cluster's 13C satellite is attached to it, not left in the residual",
      _sat["role"] == L.ROLE_ISO and _sat["parent_peak_id"] == "dimer"
      and str(_sat["iso_label"]) == "13C", (_sat["role"], _sat["iso_label"]))


# --- the annotation layer (the rows the commit pass could not take) -------
def _covalent_ledger():
    """The same spectrum as read WITHOUT the cluster channel: the ladder ions
    carry covalent formulas from the grid."""
    led = _ledger()
    for pid, neutral, adduct, ionf in (
            ("hyd2", "C4H10O2", "[M+H]+", "C4H11O2+"),
            ("cond", "C4H8O", "[M+H]+", "C4H9O+"),
            ("other", "C6H10NO", "[M+H]+", "C6H12NO+")):
        L.commit_assignment(led, pid, neutral_formula=neutral, adduct=adduct,
                            ion_formula=ionf, ion_score=0.95, ppm_error=0.3,
                            pass_no=1, method="cheminfo+grid", confidence="High",
                            commentary="Pass 1 (cheminfo+grid).")
    return led


ledc = _covalent_ledger()
notes = SC.covalent_alias_notes(ledc, EASYIC)
_idx = {ledc.at[i, "peak_id"]: i for i in ledc.index}
check("a covalent C4H11O2+ commit gets the cluster-vs-covalent note",
      _idx["hyd2"] in notes and "[(C2H6O)2-H]+" in notes[_idx["hyd2"]],
      notes.get(_idx["hyd2"]))
check("a covalent C4H9O+ commit gets it too",
      _idx["cond"] in notes and "ethanol" in notes[_idx["cond"]])
check("the note quantifies the solvent that motivates it",
      "% of this spectrum's signal" in notes[_idx["hyd2"]], notes[_idx["hyd2"]])
check("an unrelated commit gets no note", _idx["other"] not in notes)

# ...and it is silent when the solvent is not in the spectrum at all
ledd = _covalent_ledger()
ledd = ledd[~ledd["peak_id"].isin(["etoh_hyd", "etoh_h"])].reset_index(drop=True)
check("DECOY no solvent monomer -> no cluster note is invented",
      SC.covalent_alias_notes(ledd, EASYIC) == {})

# a row the commit pass already read as a cluster must not be re-annotated
lede, _ = _run()
check("a committed cluster row is not given the 'kept the covalent reading' note",
      SC.covalent_alias_notes(lede, EASYIC) == {})


def test_all():
    assert FAIL == 0, f"{FAIL} checks failed"


if __name__ == "__main__":
    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)
