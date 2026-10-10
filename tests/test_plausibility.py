"""Offline tests for plausibility.py (chemical-plausibility QC of assignments).
Run: python3 tests/test_plausibility.py"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from peaky import plausibility as PL  # noqa: E402

PASS = FAIL = 0
def check(name, cond, detail=""):
    global PASS, FAIL
    if cond: PASS += 1; print(f"  ok  {name}")
    else: FAIL += 1; print(f"FAIL  {name}  {detail}")


# --- implausible() rules ---
check("N-monster high O/C flagged (Candidate)",
      PL.implausible("C9H12N4O12", tier="Candidate") is not None)
check("N5O20 monster flagged", PL.implausible("C15H37N5O20", tier="Candidate") is not None)
check("very low H/C flagged", PL.implausible("C35H4", tier="Candidate") is not None)
check("positive-mode halogen flagged",
      PL.implausible("C10H18Br2O12", tier="Candidate", polarity="+") is not None)
check("same halogen NOT flagged in negative mode",
      PL.implausible("C10H18Br2O12", tier="Candidate", polarity="-") is None
      or "halogen" not in (PL.implausible("C10H18Br2O12", tier="Candidate", polarity="-") or ""))

# real molecules are NOT flagged
check("ordinary CHO not flagged", PL.implausible("C10H16O2", tier="Candidate") is None)
check("monoterpene-oxidation product not flagged", PL.implausible("C10H16O5", tier="Candidate") is None)
check("small N1 amine not flagged", PL.implausible("C10H19NO2", tier="Candidate") is None)
check("modest N3 low-O species not flagged (O/C<1)",
      PL.implausible("C12H21N3O3", tier="Candidate") is None)
check("HOM dimer not flagged", PL.implausible("C20H30O14", tier="Candidate") is None)

# Assigned tier is never second-guessed
check("Assigned N-monster NOT flagged (corroborated by isotope score)",
      PL.implausible("C9H12N4O12", tier="Assigned") is None)

# --- scan() de-duplicates and respects best-tier-per-neutral ---
merged = pd.DataFrame([
    dict(neutral_formula="C9H12N4O12", adduct="[M+Na]+", tier="Candidate", ion_score=0.94),
    dict(neutral_formula="C10H16O2", adduct="[M+H]+", tier="Assigned", ion_score=0.9),
    # same monster appears Assigned in one channel -> must NOT be flagged
    dict(neutral_formula="C8H13N3O10", adduct="[M+H]+", tier="Candidate", ion_score=0.6),
    dict(neutral_formula="C8H13N3O10", adduct="[M+NH4]+", tier="Assigned", ion_score=0.8),
    dict(neutral_formula="C10H18Br2O12", adduct="[M+H]+", tier="Candidate", ion_score=0.65),
])
flagged = PL.scan(merged, polarity="+")
names = {d["neutral_formula"] for d in flagged}
check("scan flags the Candidate-only N-monster", "C9H12N4O12" in names, names)
check("scan flags the positive-mode dibromo", "C10H18Br2O12" in names, names)
check("scan excludes a neutral that is Assigned in any channel",
      "C8H13N3O10" not in names, names)
check("scan excludes ordinary CHO", "C10H16O2" not in names, names)
check("scan carries reason + score", all("reason" in d and "ion_score" in d for d in flagged))

# negative mode: the dibromo is NOT flagged for halogen (covalent organohalogen is plausible)
flagged_neg = PL.scan(merged, polarity="-")
check("negative mode does not flag the dibromo on halogen grounds",
      "C10H18Br2O12" not in {d["neutral_formula"] for d in flagged_neg},
      {d["neutral_formula"] for d in flagged_neg})

# empty / missing columns degrade gracefully
check("scan empty frame -> []", PL.scan(pd.DataFrame()) == [])


# ===========================================================================
# Stage 3: HARDENED demote gates (the shared oracle + the demotes)
# ===========================================================================
from peaky import chemistry as C        # noqa: E402

# --- shared oracle: is_oxygen_monster / is_carbon_cluster ---
def cf(s): return C.parse_formula(s)

check("oracle: O-monster O/C 1.6 flagged (C5H4O8)", PL.is_oxygen_monster(cf("C5H4O8")))
check("oracle: O-monster lattice fit (C3H5ClO17)", PL.is_oxygen_monster(cf("C3H5ClO17")))
check("oracle: real HOM O/C 0.7 NOT an O-monster (C10H16O7)", not PL.is_oxygen_monster(cf("C10H16O7")))
check("oracle: HOM dimer O/C 0.7 NOT an O-monster (C20H30O14)", not PL.is_oxygen_monster(cf("C20H30O14")))
check("oracle: O/C exactly 1.3 NOT a monster (strict >)", not PL.is_oxygen_monster(cf("C10H10O13")))

check("oracle: carbon cluster DBE/C 1.0 flagged (C24H2)", PL.is_carbon_cluster(cf("C24H2")))
# real aromatics sit below DBE/C 1.0 and MUST be spared (the 0.75 cutoff caught them)
check("oracle: pyridine C5H5N (DBE/C 0.80) NOT a carbon cluster", not PL.is_carbon_cluster(cf("C5H5N")))
check("oracle: coumarin C9H6O2 (DBE/C 0.78) NOT a carbon cluster", not PL.is_carbon_cluster(cf("C9H6O2")))
check("oracle: furfural C5H4O2 (DBE/C 0.80) NOT a carbon cluster", not PL.is_carbon_cluster(cf("C5H4O2")))
check("oracle: umbelliferone C9H6O3 NOT a carbon cluster", not PL.is_carbon_cluster(cf("C9H6O3")))
check("oracle: phthalic anhydride C8H4O3 (0.88) NOT a carbon cluster", not PL.is_carbon_cluster(cf("C8H4O3")))
# HALF-INTEGER DBE (radical) is EXEMPT even at high DBE/C
check("oracle: half-integer-DBE radical C10H3 (DBE 9.5) EXEMPT",
      abs(C.dbe(cf("C10H3")) - round(C.dbe(cf("C10H3")))) > 1e-9 and not PL.is_carbon_cluster(cf("C10H3")))
check("oracle: F-rich low-H/C NOT a carbon cluster (F-free rule)", not PL.is_carbon_cluster(cf("C11H6F16")))
check("oracle: single carbon C1 not a cluster (C>=2)", not PL.is_carbon_cluster(cf("CH2")))

# implausible() shares the same oracle (one source of truth)
check("implausible: O-monster reason via shared oracle",
      "oxygen-lattice monster" in (PL.implausible("C5H4O8", tier="Candidate") or ""))
check("implausible: carbon-cluster reason via shared oracle",
      "carbon cluster" in (PL.implausible("C24H2", tier="Candidate") or ""))
check("implausible: pyridine NOT flagged by either gate", PL.implausible("C5H5N", tier="Candidate") is None)
check("implausible: coumarin NOT flagged", PL.implausible("C9H6O2", tier="Candidate") is None)
check("implausible: furfural NOT flagged", PL.implausible("C5H4O2", tier="Candidate") is None)

# --- demote_oxygen_monsters: O/C>1.3 AND mass-saturated (NOT niso-gated) ---
ledo = pd.DataFrame([
    dict(role="M0", mz=300.0, neutral_formula="C5H4O8", tier="Assigned", commentary="",
         below_assignability=False, degeneracy_note="MASS-SATURATED: 27 plausible formulas", isotopologues="[]"),
    # O-monster carrying a real 13C twin -> STILL demoted (niso must NOT exempt it)
    dict(role="M0", mz=305.0, neutral_formula="C6H4O9", tier="Assigned", commentary="",
         below_assignability=False, degeneracy_note="MASS-SATURATED: 19 plausible formulas",
         isotopologues='[{"label": "13C", "score": 0.9}]'),
    # high O/C but NOT saturated -> spared (the second leg)
    dict(role="M0", mz=310.0, neutral_formula="C4H4O7", tier="Assigned", commentary="",
         below_assignability=False, degeneracy_note="unique within 3 sigma window", isotopologues="[]"),
    # real HOM O/C 0.7, even if saturated -> spared (the ratio leg)
    dict(role="M0", mz=320.0, neutral_formula="C10H16O7", tier="Assigned", commentary="",
         below_assignability=False, degeneracy_note="MASS-SATURATED: 14 plausible formulas", isotopologues="[]"),
])
audit_o = []
outo = PL.demote_oxygen_monsters(ledo, audit=audit_o, log=lambda *a: None)
check("O-monster demote: 2 saturated O-monsters demoted (incl. one with a 13C twin)",
      outo == {"o_demoted": 2}, outo)
check("O-monster demote: C5H4O8 -> Candidate + below_assignability",
      ledo.loc[0, "tier"] == "Candidate" and bool(ledo.loc[0, "below_assignability"]))
check("O-monster demote: NOT niso-gated (C6H4O9 with 13C still demoted)",
      ledo.loc[1, "tier"] == "Candidate" and bool(ledo.loc[1, "below_assignability"]))
check("O-monster demote: high-O but NOT saturated spared (C4H4O7)",
      ledo.loc[2, "tier"] == "Assigned" and not bool(ledo.loc[2, "below_assignability"]))
check("O-monster demote: real HOM C10H16O7 spared (ratio leg)",
      ledo.loc[3, "tier"] == "Assigned" and not bool(ledo.loc[3, "below_assignability"]))
check("O-monster demote: audit one row per touched peak", len(audit_o) == 2, audit_o)
check("O-monster demote: audit carries O/C evidence + degeneracy note",
      all("O/C" in a["evidence"] and "SATUR" in a["degeneracy_note"].upper() for a in audit_o))

# --- demote_carbon_clusters: DBE/C>=1.0, F-free, radical-exempt ---
ledc = pd.DataFrame([
    dict(role="M0", mz=290.0, neutral_formula="C24H2", tier="Assigned", commentary="",
         below_assignability=False, isotopologues="[]"),                    # DBE/C 1.0 -> demote
    dict(role="M0", mz=300.0, neutral_formula="C5H5N",  tier="Assigned", commentary="",
         below_assignability=False, isotopologues="[]"),                    # pyridine -> spare
    dict(role="M0", mz=310.0, neutral_formula="C9H6O2", tier="Assigned", commentary="",
         below_assignability=False, isotopologues="[]"),                    # coumarin -> spare
    dict(role="M0", mz=320.0, neutral_formula="C5H4O2", tier="Assigned", commentary="",
         below_assignability=False, isotopologues="[]"),                    # furfural -> spare
    dict(role="M0", mz=330.0, neutral_formula="C10H3",  tier="Assigned", commentary="",
         below_assignability=False, isotopologues="[]"),                    # half-int DBE radical -> EXEMPT
    dict(role="M0", mz=340.0, neutral_formula="C10H16O4", tier="Assigned", commentary="",
         below_assignability=False, isotopologues="[]"),                    # ordinary SOA -> spare
])
audit_c = []
outc = PL.demote_carbon_clusters(ledc, audit=audit_c, log=lambda *a: None)
check("carbon-cluster demote: only the DBE/C>=1.0 integer cluster (C24H2)", outc == {"c_cluster_demoted": 1}, outc)
check("carbon-cluster demote: C24H2 -> Candidate + below_assignability",
      ledc.loc[0, "tier"] == "Candidate" and bool(ledc.loc[0, "below_assignability"]))
check("carbon-cluster demote: pyridine C5H5N spared", ledc.loc[1, "tier"] == "Assigned")
check("carbon-cluster demote: coumarin C9H6O2 spared", ledc.loc[2, "tier"] == "Assigned")
check("carbon-cluster demote: furfural C5H4O2 spared", ledc.loc[3, "tier"] == "Assigned")
check("carbon-cluster demote: half-integer-DBE radical C10H3 EXEMPT",
      ledc.loc[4, "tier"] == "Assigned" and not bool(ledc.loc[4, "below_assignability"]))
check("carbon-cluster demote: ordinary SOA C10H16O4 spared", ledc.loc[5, "tier"] == "Assigned")

# demote_implausible runs both, and NEVER deletes a row (row count is preserved)
ledboth = pd.concat([ledo, ledc], ignore_index=True)
ledboth["tier"] = "Assigned"; ledboth["below_assignability"] = False
n_before = len(ledboth)
PL.demote_implausible(ledboth, audit=[], log=lambda *a: None)
check("demote_implausible: demote-only, never deletes a row", len(ledboth) == n_before)
check("demote_implausible: no row left without a tier value",
      ledboth["tier"].isin(["Assigned", "Candidate"]).all())

# --- write_audit: deterministic + always a header ---
import tempfile, os as _os    # noqa: E402
with tempfile.TemporaryDirectory() as _d:
    _n0 = PL.write_audit([], _os.path.join(_d, "empty.csv"))
    _hdr = open(_os.path.join(_d, "empty.csv")).readline().strip()
    check("audit: empty -> 0 rows but a header line", _n0 == 0 and _hdr.startswith("mz,neutral_formula"))
    _rows_a = [dict(mz=300.0, neutral_formula="C5H4O8", before_tier="Assigned",
                    after_tier_or_role="Candidate", reason="x", evidence="O/C=1.6",
                    degeneracy_note="SAT", n_iso=0),
               dict(mz=200.0, neutral_formula="C24H2", before_tier="Assigned",
                    after_tier_or_role="Candidate", reason="y", evidence="DBE/C=1.0",
                    degeneracy_note="", n_iso=0)]
    _n1 = PL.write_audit(_rows_a, _os.path.join(_d, "a.csv"))
    _df = pd.read_csv(_os.path.join(_d, "a.csv"))
    check("audit: nonempty rows written + sorted by mz", _n1 == 2 and list(_df["mz"]) == [200.0, 300.0])


def test_all():
    assert FAIL == 0, f"{FAIL} checks failed"


if __name__ == "__main__":
    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


def test_oxygen_monster_demote_needs_a_degenerate_window():
    """The O-monster demote's second leg is the tier engine's own "degenerate" window
    (>= 3 plausible ions, a lower bound included, or MASS-SATURATED). The audit counts the
    commit itself, so a small high-O/C acid with one competitor reads density 2 (malonic
    acid [M+^NO3]- beside a fluorinated [M-H]- on the nitrate Orbitrap) -- spared; an
    O-rich formula in a 4-ion window is not."""
    from peaky.assignment import plausibility as PL
    from peaky.assignment import tiers as T
    deg2 = ("MASS-DEGENERATE: 2 plausible ions within ±3σ calibrated window — competitors: "
            "C5H3F3O3 [M-H]- (+1.12 ppm)")
    deg_lb = ("MASS-DEGENERATE: at least 4 plausible ions within ±3σ calibrated window — competitors: "
              "a; b; c — the committed formula lies outside this run's enumerated space (x), so the "
              "count is a lower bound")
    sat = ("MASS-SATURATED: 11 plausible formulas (≤3 heteroatom types) within ±3σ calibrated "
           "window — not identifiable from accurate mass alone")
    nm = ("not measured: the committed formula lies outside this run's enumerated space (x); "
          "1 other plausible ion(s) within ±3σ calibrated window: a")
    rows = [("C3H4O4", 165.0, 2, deg2), ("C6H14O9", 309.0, 4, deg_lb), ("C3H6O4", 167.0, 11, sat),
            ("C2H4O4", 168.0, pd.NA, nm), ("C2H2O4", 170.0, 1, "unique within ±3σ calibrated mass window")]
    led = pd.DataFrame([dict(role="M0", mz=mz, neutral_formula=nf, tier="Assigned", commentary="",
                             below_assignability=False, degeneracy_density=d, degeneracy_note=n,
                             isotopologues="[]") for nf, mz, d, n in rows])
    assert PL.demote_oxygen_monsters(led, log=lambda *a: None) == {"o_demoted": 2}
    assert list(led["tier"]) == ["Assigned", "Candidate", "Candidate", "Assigned", "Assigned"]
    assert [bool(v) for v in led["below_assignability"]] == [False, True, True, False, False]
    assert "mass-degenerate" in led.at[1, "commentary"]
    # the second leg is the tier engine's own predicate, row for row
    for i in led.index:
        assert PL._mass_degenerate(led.loc[i]) == T._degeneracy(led.loc[i])[1]
    assert not PL._mass_degenerate({}) and not PL._mass_degenerate({"degeneracy_note": pd.NA})

# ===========================================================================
# The ambient NITRATE run (contexts.run_profile: a nitrate-reagent run on an
# Orbitrap-class axis switches nox_skeleton + small_acid_band on). Every check
# above passes no profile, and the positive-mode / every-other-context case
# below passes one with the switches off: those expectations are unchanged.
# Only the nitrate run reads the carbon skeleton of an organonitrate.
# ===========================================================================
def _nitrate_run():
    from peaky.chem import contexts as X
    return X.run_profile(X.get_context("ambient-air"), reagent="NO3", instrument_class="orbitrap")


def test_nitrate_run_reads_organonitrate_skeletons_in_the_oxygen_gate():
    """A C5 hydroxy nitrate C5H9NO7 is O/C 1.4 raw, 1.0 on its skeleton (C5H10O5); a C7
    trihydroxy-carbonyl nitrate C7H11NO10 is 1.43 raw, 1.14 on its skeleton -- both under the
    HOM ceiling the gate was set for (1.14), so NOT oxygen-lattice monsters on the nitrate run.
    A CHO C5H2O8 has no NOx group to discount and stays one."""
    prof = _nitrate_run()
    for f in ("C5H9NO7", "C7H11NO10"):
        assert PL.is_oxygen_monster(cf(f))                    # the raw gate (no profile): unchanged
        assert not PL.is_oxygen_monster(cf(f), prof)
    assert PL.is_oxygen_monster(cf("C5H2O8"), prof)


def test_nitrate_run_spares_a_dinitrate_in_a_degenerate_window():
    """C6H10N2O9 (raw O/C 1.5, skeleton C6H12O5 0.83) in a MASS-SATURATED window: demoted by the
    raw gate, kept Assigned on the nitrate run."""
    def led():
        return pd.DataFrame([dict(role="M0", mz=250.0, neutral_formula="C6H10N2O9", tier="Assigned",
                                  commentary="", below_assignability=False, isotopologues="[]",
                                  degeneracy_note="MASS-SATURATED: 11 plausible formulas")])
    raw, run = led(), led()
    assert PL.demote_oxygen_monsters(raw, log=lambda *a: None) == {"o_demoted": 1}
    assert PL.demote_oxygen_monsters(run, log=lambda *a: None, profile=_nitrate_run()) == {"o_demoted": 0}
    assert run.at[0, "tier"] == "Assigned"
    # demote_implausible takes the run's profile as its context and passes it on
    both = led()
    PL.demote_implausible(both, log=lambda *a: None, context=_nitrate_run())
    assert both.at[0, "tier"] == "Assigned"


def test_nitrate_run_scrutiny_flags():
    """A Candidate trinitrate C5H9N3O10 (glycerol-like C5 skeleton C5H12O4 after 3 nitrate groups)
    is not a 'heteroatom coincidence' on the nitrate run. The N-cap scrutiny stays: a formula with
    more N than the context's cap (C8H6N4O9, C9H12N4O12: N4 > ambient max_N 3, beyond the three
    groups the skeleton reading credits) is judged raw and still flagged."""
    prof = _nitrate_run()
    assert PL.implausible("C5H9N3O10", tier="Candidate") is not None
    assert PL.implausible("C5H9N3O10", tier="Candidate", profile=prof) is None
    assert PL.implausible("C8H6N4O9", tier="Candidate", profile=prof) is not None
    assert PL.implausible("C9H12N4O12", tier="Candidate", profile=prof) is not None
    # an ordinary CHO coincidence is untouched by the profile
    assert "oxygen-lattice monster" in (PL.implausible("C5H4O8", tier="Candidate", profile=prof) or "")
    led = pd.DataFrame({"neutral_formula": ["C5H9N3O10", "C9H12N4O12"], "tier": ["Candidate"] * 2,
                        "ion_score": [0.9, 0.9]})
    assert [d["neutral_formula"] for d in PL.scan(led, profile=prof)] == ["C9H12N4O12"]


def test_nitrate_run_carbon_cluster_reads_the_skeleton_and_exempts_small_acids():
    """Dinitrophenol C6H4N2O5 is DBE/C 1.0 raw (a 'carbon cluster') and phenol (0.67) on its
    skeleton; acetylenedicarboxylic acid C4H2O4 (DBE/C 1.0) is a C3-C4 polycarbonyl acid the
    small-acid band admits. Neither is demoted on the nitrate run; C24H2 still is."""
    prof = _nitrate_run()
    for f in ("C6H4N2O5", "C4H2O4", "C3H2O4"):
        assert PL.is_carbon_cluster(cf(f))
        assert not PL.is_carbon_cluster(cf(f), prof)
    assert PL.is_carbon_cluster(cf("C24H2"), prof)
    led = pd.DataFrame([dict(role="M0", mz=m, neutral_formula=f, tier="Assigned", commentary="",
                             below_assignability=False, isotopologues="[]")
                        for f, m in (("C4H2O4", 113.0), ("C6H4N2O5", 183.0), ("C24H2", 290.0))])
    assert PL.demote_carbon_clusters(led, log=lambda *a: None, profile=prof) == {"c_cluster_demoted": 1}
    assert list(led["tier"]) == ["Assigned", "Assigned", "Candidate"]


def test_other_contexts_keep_the_raw_gates():
    """Positive mode (uronium) and a TOF / trace-first nitrate run get no switches: every gate
    reads the raw neutral exactly as with no profile."""
    from peaky.chem import contexts as X
    offs = [X.run_profile(X.get_context("uronium"), reagent="Ur", instrument_class="orbitrap"),
            X.run_profile(X.get_context("ambient-air"), reagent="NO3", instrument_class="tof"),
            X.run_profile(X.get_context("ambient-air"), reagent="NO3", instrument_class="orbitrap",
                          trace_sample=True)]
    for prof in offs:
        assert not prof.nox_skeleton and not prof.small_acid_band
        for f in ("C5H9NO7", "C6H4N2O5", "C4H2O4", "C5H9N3O10", "C9H12N4O12"):
            assert PL.implausible(f, tier="Candidate", profile=prof) == PL.implausible(f, tier="Candidate")
            assert PL.is_oxygen_monster(cf(f), prof) == PL.is_oxygen_monster(cf(f))
            assert PL.is_carbon_cluster(cf(f), prof) == PL.is_carbon_cluster(cf(f))


def test_nitrate_run_judges_c1_c2_and_skeleton_free_formulas_raw():
    """The skeleton reading needs Ceff >= 3, as the context filter's ratio test: C2H4N2O2 stays a
    carbon cluster and CH3NO3 / C2H3NO4 stay oxygen monsters on the nitrate run. A formula with
    no NOx skeleton at all (C4H13N3O5, DBE 0) keeps its raw heteroatom flag."""
    prof = _nitrate_run()
    assert PL.is_carbon_cluster(cf("C2H4N2O2"), prof)
    for f in ("CH3NO3", "C2H3NO4"):
        assert PL.is_oxygen_monster(cf(f), prof), f
    for f in ("C2H4N2O2", "CH3NO3", "C2H3NO4", "C4H13N3O5"):
        assert PL.implausible(f, tier="Candidate", profile=prof) == PL.implausible(f, tier="Candidate"), f
    assert "heteroatom coincidence" in PL.implausible("C4H13N3O5", tier="Candidate", profile=prof)
