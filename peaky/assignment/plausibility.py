"""Chemical-plausibility QC for assigned formulas.

High-resolution accurate mass alone lets the fitter hit almost any target once it
is allowed many heteroatoms (every extra N/O/S/halogen adds free parameters), so a
small fraction of assignments are mass-coincidence "monsters" rather than real
molecules in an organic-aerosol matrix: extreme heteroatom counts, implausibly
carbon-rich (very low H/C) skeletons, or a covalent halogen in a positive-mode
spectrum whose reagent provides no halogen.

`scan` flags these for SCRUTINY — it does not delete or re-assign anything. It is
deliberately conservative and only inspects CANDIDATE-tier neutrals: an Assigned
assignment cleared the server's isotope-pattern score, which is independent
corroboration we don't second-guess from element ratios. A neutral seen as
Assigned in ANY channel is therefore never flagged.

Pure formula arithmetic; deterministic. Thresholds are intentionally loose so the
flagged set is small and defensible (the clear coincidences), not a dragnet.
"""
from __future__ import annotations

import dataclasses
import json

import numpy as np
import pandas as pd

from peaky.chem import chemistry as C
from peaky.assignment import ledger as L
from peaky.assignment import tiers as T

__version__ = "0.6.2"   # skeleton_reliance / rests_on_skeleton (the tier cap's predicate); 0.6.1 a peak gate_element_evidence clears is locked; 0.6.0 NOx-skeleton readings (profile=); 0.5.0 the element-evidence gate (gate_element_evidence); 0.4.0 element-budget demote (demote_off_budget); 0.3.0 carbon-cluster rule: F no longer exempts

# thresholds (loose on purpose — flag the clear coincidences only)
N_HIGH_OC = 3       # N>=3 combined with...
OC_HIGH = 1.0       # ...O/C >= this  -> high-heteroatom coincidence
N_VERY_HIGH = 4     # N>=4 combined with...
O_HIGH = 8          # ...O>= this
HC_FLOOR = 0.35     # H/C below this -> implausibly carbon-rich (F-FREE formulas only)
F_HIGH = 4          # F>=this -> heavily fluorinated; F is monoisotopic, so the fit
                    # has NO isotope twin to confirm it (a fluorine mass coincidence).
                    # Flagged with the RIGHT reason: low H/C here is F displacing H
                    # (fluorine-rich), NOT a carbon-rich skeleton.

# --- the SHARED plausibility oracle (Stage 3) ------------------------------
# The demote/relabel steps (cleanup-level + this module's scrutiny scan) all read
# the SAME predicates so a flagged formula and a demoted formula are never out of
# step. Two gates were hardened (live-data calibrated, see assign.py wiring):
#
#   * O-MONSTER: an extreme O/C (> OC_MONSTER) is the oxygen-lattice mass-fit
#     signature. It is NOT a niso gate -- a 13C satellite confirms the CARBON
#     count, not the O count, so an O-monster carrying a real 13C twin is still an
#     O-monster. Real HOMs top out at O/C ~1.14, so OC_MONSTER=1.3 spares every
#     genuine oxidation product. (The DEMOTE additionally requires a mass-degenerate
#     window from the degeneracy audit -- the tier engine's own threshold, >= 3
#     plausible ions or MASS-SATURATED; the plain reason-string oracle reports the ratio.)
#
#   * CARBON-CLUSTER: DBE/C >= DBE_PER_C_MONSTER (equivalently H <= N+2) on an
#     C>=2 skeleton is a bare-carbon mass coincidence (e.g. C5H2, C24H2, C12HF --
#     DBE/C >= 1). H-poor-but-DBE/C<1 skeletons like C27H8 (0.89) are NOT this gate;
#     they are the separate H/C<0.35 carbon-rich demote in cleanup.py.
#     A HALF-INTEGER DBE is EXEMPT: radicals carry half-integer DBE, whereas the
#     carbon-cluster monsters are all integer-DBE. The >= 1.0 cutoff (NOT the
#     earlier 0.75 proposal) is deliberate -- pyridine (0.80), coumarin (0.78),
#     umbelliferone (0.78), furfural (0.80) and phthalic anhydride (0.88) are real
#     aromatics that sit below 1.0 and MUST be spared.
OC_MONSTER = 1.3            # O/C strictly above this -> oxygen-lattice monster
DBE_PER_C_MONSTER = 1.0     # DBE/C at or above this (C>=2) -> carbon cluster


def _oc(cnt: dict) -> float:
    """O/C ratio (0 when carbon-free; carbon-free is handled elsewhere)."""
    nc = cnt.get("C", 0)
    return cnt.get("O", 0) / nc if nc else 0.0


# --- the NOx-skeleton readings (a run whose context profile sets nox_skeleton) --
# An -ONO2 / -NO2 group adds 2 O and one DBE to its carbon skeleton, so the raw
# O/C and DBE/C of a dinitrate or a nitroaromatic read as an oxygen lattice or a
# carbon cluster (a C5 dihydroxy dinitrate is O/C 1.6; dinitrophenol DBE/C 1.0).
# With the profile's switch on, these gates read the skeleton too
# (contexts.nox_skeletons: up to 3 groups) and fire only when EVERY reading
# does. A formula with more N than the context's cap or than three groups can
# carry is judged raw, as before, and so is a C1-C2 neutral: the context filter
# reads no Van Krevelen ratio below Ceff 3 (contexts.vk_readings), and an
# "aromatic" nitro skeleton means nothing at C2. Without a profile, or with the
# switch off (positive mode, every other context), nothing changes.
def _readings(cnt: dict, profile) -> list[dict]:
    """The raw counts, then the NOx skeletons when the profile reads them, the
    neutral has Ceff = C + Si >= 3 (as the filter's ratio test) and its N is
    within min(3, the context's N cap)."""
    out = [cnt]
    if profile is None or not getattr(profile, "nox_skeleton", False):
        return out
    if cnt.get("C", 0) + cnt.get("Si", 0) < 3:
        return out
    from peaky.chem import contexts as X
    n_all = cnt.get("N", 0) + cnt.get("^N", 0)
    if n_all > min(X.NOX_K_MAX, int(getattr(profile, "max_N", 99))):
        return out
    return out + [sk for _k, sk in X.nox_skeletons(cnt, profile)]


def is_oxygen_monster(cnt: dict, profile=None) -> bool:
    """O/C strictly above OC_MONSTER (the oxygen-lattice mass-fit ratio) on every
    reading (`_readings`: the raw neutral, and its NOx skeletons when `profile`
    reads them). Pure arithmetic -- the DEMOTE additionally gates on degeneracy
    mass-saturation."""
    return cnt.get("C", 0) > 0 and min(_oc(r) for r in _readings(cnt, profile)) > OC_MONSTER


def _small_acid(cnt: dict, profile) -> bool:
    """A C3-C4 polycarbonyl acid the profile's small-acid band admits."""
    if profile is None or not getattr(profile, "small_acid_band", False):
        return False
    from peaky.chem import contexts as X
    return X.small_acid_band_applies(cnt, profile)


def is_carbon_cluster(cnt: dict, profile=None) -> bool:
    """C>=2 skeleton whose DBE/C >= DBE_PER_C_MONSTER, EXCLUDING radicals
    (half-integer DBE are exempt). H+halogen <= N+2 is the equivalent integer
    test (C.dbe already counts F/Cl/Br like H), but we compute the real DBE so
    the half-integer radical exemption is exact. F-bearing skeletons are NOT
    exempt: dbe() counting F as an H-equivalent means real perfluoro classes
    (PFCA DBE/C~0.2) can never trip this gate, while C12HF / C25H3F3 style
    bare-carbon fits (DBE/C~1) previously slipped through on an F-free clause
    (2026-07-04 [15N]-nitrate run: 15 such F-decorated clusters tier-Assigned).
    With a `profile` that reads NOx skeletons, a nitroaromatic is a cluster only
    when its skeleton is one too (dinitrophenol reads as phenol, DBE/C 0.67);
    with its small-acid band, a C3-C4 polycarbonyl acid the band admits
    (acetylenedicarboxylic C4H2O4, DBE/C 1.0) is exempt.
    """
    nc = cnt.get("C", 0)
    if nc < 2:
        return False
    if C.odd_electron(cnt):            # half-integer DBE -> radical, EXEMPT
        return False
    if _small_acid(cnt, profile):
        return False
    return min(C.dbe(r) / nc for r in _readings(cnt, profile)) >= DBE_PER_C_MONSTER


def _hetero_flag(cnt: dict) -> str | None:
    """The heteroatom-coincidence flags of one reading (None when neither fires)."""
    nc = cnt.get("C", 0)
    n, o = cnt.get("N", 0), cnt.get("O", 0)
    oc = o / nc if nc else 0.0
    if n >= N_HIGH_OC and oc >= OC_HIGH:
        return f"N{n}, O/C {oc:.1f} (heteroatom coincidence)"
    if n >= N_VERY_HIGH and o >= O_HIGH:
        return f"N{n}O{o} (heteroatom coincidence)"
    return None


def implausible(neutral_formula: str, *, tier: str | None = None,
                polarity: str | None = None, profile=None) -> str | None:
    """Return a short reason string if `neutral_formula` looks like a mass-coincidence
    fit rather than a real molecule, else None. Only Candidate-tier is scrutinised
    (pass tier=None to scrutinise regardless). `polarity` ('+'/'-') enables the
    wrong-mode-halogen check. `profile` (the run's ContextProfile): with its
    nox_skeleton switch the O/C, heteroatom and carbon-cluster checks read the
    NOx skeletons too (`_readings`) and flag only when every reading does; a
    formula with more N than min(3, the context's N cap) is judged raw."""
    if tier is not None and str(tier) != "Candidate":
        return None
    c = C.parse_formula(str(neutral_formula))
    nc = c.get("C", 0)
    if nc == 0:
        return None                      # carbon-free handled elsewhere (reagent/inorganic)
    h, o = c.get("H", 0), c.get("O", 0)
    f = c.get("F", 0)
    br, cl = c.get("Br", 0), c.get("Cl", 0)
    hc, oc = h / nc, o / nc
    # Terse labels (the full meaning is spelled out in the scrutiny-page legend);
    # keeping them short stops the table overflowing the page width.
    if is_oxygen_monster(c, profile):    # O/C beyond the HOM ceiling -> oxygen-lattice monster
        return f"O/C {oc:.1f} (oxygen-lattice monster)"
    het = _hetero_flag(c)
    if het and all(_hetero_flag(r) for r in _readings(c, profile)[1:]):
        return het
    if f >= F_HIGH:           # heavily fluorinated: 19F is 100% monoisotopic
        # NB any 13C/81Br satellites the row carries confirm the CARBON count / the
        # adduct halogen, NOT the fluorine -- 19F has no heavier stable isotope, so
        # the F COUNT is never isotope-confirmable (do NOT say "no isotope twin").
        return f"F{f}: 19F monoisotopic, fluorine count not isotope-confirmable"
    if is_carbon_cluster(c, profile):  # DBE/C>=1.0, integer-DBE (radicals exempt)
        return f"DBE/C {C.dbe(c) / nc:.2f} (carbon cluster, H<=N+2)"
    if f == 0 and hc < HC_FLOOR:     # genuine carbon-rich skeleton (F not displacing H)
        return f"H/C {hc:.2f} (carbon-rich)"
    if polarity == "+" and (br > 0 or cl > 0):
        return "halogen in neutral, +mode"
    return None


#: skeleton_reliance's clause for a reading the context filter admits only on a skeleton
SKELETON_ONLY_REASON = "passes the context windows only through its organonitrate-skeleton reading"


def skeleton_reliance(neutral_formula: str, profile, *, filtered: bool = True, mass_degenerate: bool = False,
                      rearbitrated: bool = False) -> str | None:
    """How the run's NOx-skeleton reading changed the outcome for a row reading
    `neutral_formula` (a short clause), or None when it did not -- or the profile
    reads no skeletons:

      * `filtered` (the row's proposer runs the context filter before it commits,
        tiers.consults_context_filter): the filter passes the formula only on a
        skeleton reading (contexts.skeleton_only);
      * any row: the carbon-cluster demote (unconditional) would have fired on the
        raw reading but not on the skeleton, or the oxygen-monster demote would
        have, on a `mass_degenerate` row (its own second leg, `_mass_degenerate`);
      * a `rearbitrated` row (re-arbitration accepts an alternative only when
        `implausible` reads it clean under the run's profile): `implausible`
        flags its raw reading only.

    The raw reading is the same profile with `nox_skeleton` off, so the small-acid
    band is never counted as a skeleton. Heavy-isotope labels fold first ('^N' is
    N), as the labelled rescue reads its formulas."""
    if profile is None or not getattr(profile, "nox_skeleton", False):
        return None
    from peaky.chem import contexts as X
    cnt = C.fold_isotopes(C.parse_formula(str(neutral_formula)))
    f = C.format_formula(cnt)
    if filtered and X.skeleton_only(f, profile):
        return SKELETON_ONLY_REASON
    raw = dataclasses.replace(profile, nox_skeleton=False)
    if is_carbon_cluster(cnt, raw) and not is_carbon_cluster(cnt, profile):
        return "is spared the carbon-cluster demote only by its organonitrate-skeleton reading"
    if mass_degenerate and is_oxygen_monster(cnt, raw) and not is_oxygen_monster(cnt, profile):
        return ("is spared the oxygen-monster demote (a mass-degenerate window) only by its "
                "organonitrate-skeleton reading")
    if rearbitrated and implausible(f, profile=raw) is not None and implausible(f, profile=profile) is None:
        return "was accepted by re-arbitration only on its organonitrate-skeleton reading"
    return None


def rests_on_skeleton(neutral_formula: str, profile, *, filtered: bool = True, mass_degenerate: bool = False,
                      rearbitrated: bool = False) -> bool:
    """True when the NOx-skeleton reading changed the outcome for the row
    (`skeleton_reliance`)."""
    return skeleton_reliance(neutral_formula, profile, filtered=filtered, mass_degenerate=mass_degenerate,
                             rearbitrated=rearbitrated) is not None


def scan(merged, *, polarity: str | None = None, profile=None) -> list[dict]:
    """Flag Candidate-only neutrals that look implausible. Returns one dict per
    distinct neutral: {neutral_formula, reason, ion_score, tier}. A neutral that is
    Assigned in any ion channel is excluded (it is corroborated). `profile`: the
    run's ContextProfile (`implausible`)."""
    if merged is None or "neutral_formula" not in getattr(merged, "columns", []):
        return []
    g = merged.dropna(subset=["neutral_formula"]).copy()
    if not len(g):
        return []
    g["neutral_formula"] = g["neutral_formula"].astype(str)
    has_tier = "tier" in g.columns
    out = []
    for f, sub in g.groupby("neutral_formula"):
        best = ("Assigned" if has_tier and (sub["tier"] == "Assigned").any()
                else "Candidate")
        reason = implausible(f, tier=best, polarity=polarity, profile=profile)
        if reason:
            sc = sub["ion_score"].max() if "ion_score" in sub.columns else None
            out.append({"neutral_formula": f, "reason": reason, "tier": best,
                        "ion_score": (float(sc) if sc is not None and sc == sc else None)})
    out.sort(key=lambda d: (d["reason"], d["neutral_formula"]))
    return out


# ===========================================================================
# Stage 3: demote-ONLY hardening (never deletes a row)
#
# Every function below DEMOTES an Assigned M0 to Candidate (+ stamps
# below_assignability; the element-budget demote stamps tentative_lead instead).
# None of them can clear/delete a row, so the worst-case
# failure is an over-cautious Candidate, never a lost peak. Each appends one dict
# per touched peak to `audit` (when a list is passed) so assign/assign_batch can
# write tables/plausibility_audit_*.
# ===========================================================================

def _mass_degenerate(row) -> bool:
    """The second leg of the O-monster demote: the tier engine's own "degenerate"
    window (`tiers._degeneracy`: more than DEGEN_DEMOTE_DENSITY plausible ions, a
    lower bound included, or the audit's MASS-SATURATED flag) -- the ratio alone is
    not enough, the mass must also be one pick of a degenerate set. Two ions is not:
    the audit counts the commit itself, so a small high-O/C acid with one competitor
    reads density 2 -- a choice between two, not an arbitrary pick."""
    return T._degeneracy(row)[1]


def _iso_count(s) -> int:
    """Number of server-confirmed isotopologues recorded on a row (0 if none)."""
    if isinstance(s, str) and s.strip().startswith("["):
        try:
            return len(json.loads(s))
        except Exception:
            return 0
    return 0


def _m0_index(ledger):
    """M0 rows when the ledger has a role column; otherwise every row (the merged
    ledger is already M0-only and carries no role column)."""
    return (ledger.index[ledger["role"] == L.ROLE_M0]
            if "role" in ledger.columns else ledger.index)


def _append_note(ledger, i, note):
    if "commentary" in ledger.columns:
        cur = ledger.at[i, "commentary"]
        prev = "" if cur is None or (isinstance(cur, float) and pd.isna(cur)) or cur is pd.NA else str(cur)
        ledger.at[i, "commentary"] = (prev + "; " + note) if prev and prev != "nan" else note


def _demote_row(ledger, i, *, reason, audit, evidence, degeneracy_note, n_iso, lead=None):
    """Demote one M0 -> Candidate + below_assignability (`lead`, a setter code
    of ledger.LEAD_SETTERS: + the tentative_lead flag set by it instead -- the
    proposal is unsupported, not contradicted; ledger.ASSIGNABILITY_FLAGS),
    append the note, and log one audit record. Demote-only: tier moves
    Assigned->Candidate, nothing is cleared."""
    before = str(ledger.at[i, "tier"]) if "tier" in ledger.columns else ""
    if "tier" in ledger.columns and before == "Assigned":
        ledger.at[i, "tier"] = "Candidate"
    if lead:
        L.mark_lead(ledger, i, lead)
    elif "below_assignability" in ledger.columns:
        ledger.at[i, "below_assignability"] = True
    _append_note(ledger, i, reason)
    if audit is not None:
        audit.append({
            "mz": ledger.at[i, "mz"] if "mz" in ledger.columns else None,
            "neutral_formula": ledger.at[i, "neutral_formula"],
            "before_tier": before, "after_tier_or_role": "Candidate",
            "reason": reason, "evidence": evidence,
            "degeneracy_note": ("" if degeneracy_note is None
                                or (isinstance(degeneracy_note, float) and pd.isna(degeneracy_note))
                                else str(degeneracy_note)),
            "n_iso": n_iso})


def demote_oxygen_monsters(ledger: pd.DataFrame, *, audit=None, log=print, profile=None) -> dict:
    """Demote M0 assignments that are oxygen-lattice 'monsters': O/C > OC_MONSTER
    AND mass-degenerate (the degeneracy audit counts >= 3 plausible ions in the
    calibrated window, or flags it MASS-SATURATED). NOT niso-gated -- a 13C satellite
    confirms the carbon count, not the oxygen count, so it would wrongly exempt a
    real O-monster. Real HOMs (O/C<=1.14) are spared by the ratio cut; high-O fits
    on a unique or two-ion window (the small polyacids: oxalic, malonic ...) are
    spared by the second leg. Assigned->Candidate + below_assignability. Demote-only.
    `profile`: the run's ContextProfile -- with nox_skeleton an organonitrate
    is judged on its skeleton O/C too (`is_oxygen_monster`)."""
    n = 0
    has_note = "degeneracy_note" in ledger.columns
    for i in _m0_index(ledger):
        cnt = C.parse_formula(str(ledger.at[i, "neutral_formula"] or ""))
        if not is_oxygen_monster(cnt, profile):
            continue
        note = ledger.at[i, "degeneracy_note"] if has_note else None
        if not _mass_degenerate(ledger.loc[i]):   # ratio alone is not enough -- needs a degenerate mass
            continue
        ni = _iso_count(ledger.at[i, "isotopologues"]) if "isotopologues" in ledger.columns else 0
        reason = (f"oxygen-lattice monster (O/C {_oc(cnt):.2f} > {OC_MONSTER}, "
                  "mass-degenerate) -- one pick of a sub-ppm-degenerate set")
        _demote_row(ledger, i, reason=reason, audit=audit,
                    evidence=f"O/C={_oc(cnt):.2f}", degeneracy_note=note, n_iso=ni)
        n += 1
    log(f"[plausibility] demoted {n} oxygen-lattice monsters (O/C>{OC_MONSTER}, mass-degenerate)")
    return {"o_demoted": n}


def demote_carbon_clusters(ledger: pd.DataFrame, *, audit=None, log=print, profile=None) -> dict:
    """Demote M0 assignments resting on a bare-carbon skeleton: DBE/C >=
    DBE_PER_C_MONSTER (H+halogen<=N+2), C>=2, with the HALF-INTEGER-DBE radical
    EXEMPTION (radicals carry half-integer DBE; carbon-cluster monsters are
    integer-DBE). This is distinct from the H/C<0.35 carbon-rich demote in
    cleanup.py (kept unchanged): the two together cover the carbon-coincidence
    family without catching real aromatics (pyridine/coumarin/furfural sit below
    DBE/C 1.0). Assigned->Candidate + below_assignability. Demote-only.
    `profile`: the run's ContextProfile (`is_carbon_cluster`: the NOx skeleton
    and the small-acid exemption)."""
    n = 0
    for i in _m0_index(ledger):
        cnt = C.parse_formula(str(ledger.at[i, "neutral_formula"] or ""))
        if not is_carbon_cluster(cnt, profile):
            continue
        nc = cnt.get("C", 0)
        dpc = C.dbe(cnt) / nc
        ni = _iso_count(ledger.at[i, "isotopologues"]) if "isotopologues" in ledger.columns else 0
        reason = (f"carbon cluster (DBE/C {dpc:.2f} >= {DBE_PER_C_MONSTER}, "
                  "H+halogen<=N+2) -- bare-carbon mass coincidence, not a molecule")
        note = ledger.at[i, "degeneracy_note"] if "degeneracy_note" in ledger.columns else None
        _demote_row(ledger, i, reason=reason, audit=audit,
                    evidence=f"DBE/C={dpc:.2f}", degeneracy_note=note, n_iso=ni)
        n += 1
    log(f"[plausibility] demoted {n} carbon clusters (DBE/C>={DBE_PER_C_MONSTER}, "
        "integer-DBE)")
    return {"c_cluster_demoted": n}


def demote_off_budget(ledger: pd.DataFrame, *, context,
                      curated=frozenset(), audit=None, log=print) -> dict:
    """Demote M0 commits whose neutral lies outside the run context's ELEMENT
    BUDGET (contexts.element_budget: the structural gate, the carbon-free
    allowlist, the heteroatom caps) and that no curated list names.

    The per-peak grid never proposes such a formula -- ambient-air keeps P, F and
    I at zero because they are monoisotopic and can never be isotope-confirmed,
    and S at one. Other commit paths widen the search on evidence of their own
    (a multi-channel certificate, a series extrapolation, a contaminant family)
    and CAN commit one; that evidence proposes the neutral MASS, and it is then
    read back as the axes (chan2, the acid branch, an anchor) that the tier and
    the merge vote's class count as confirmation. So an off-budget formula that no
    curated list names is Candidate + tentative_lead (C19(c): the proposal is
    unsupported, not contradicted; before the split it was below_assignability)
    -- the evidence level prints that as the `lead` tag, with no level effect
    (docs/EVIDENCE_LEVELS.md section 3.2). `curated` is every formula the pass-0 registry names for
    this polarity/context plus the active reference lists
    (assign._stage_plausibility), exempt whichever pass committed it. Measured on
    a same-air TOF/Orbitrap pair before the rule: the TOF's Assigned
    phosphorus and multi-sulfur neutrals were ALL such commits.

    One element has evidence of its own that no isotope can give: fluorine, whose
    only stable isotope is 19F, is pinned by a CF2 STEP -- two committed rows on
    the same adduct whose neutrals differ by exactly CF2 carry, between them, two
    more fluorines (at Orbitrap accuracy no other composition difference sits
    within the window; the nearest, CH3Cl and O3H2, are 4.5 and 3.6 mDa away).
    So a formula whose only budget violation is fluorine is KEPT when the ledger
    commits its CF2 neighbour (neutral +/- CF2, same adduct) -- a consistent
    member of a fluorinated homologous series, the same standing an isotope line
    gives Cl, Br or S. The step pins the fluorine difference, not the rest of the
    formula; the level still reads the row's own axes. Measured on the same pair:
    peak pairs one CF2 apart were 20x chance on the labelled-nitrate Orbitrap and
    4x on the ~10k TOF (where ~1 in 4 such pairs is chance), and the fluorinated
    rows the budget would demote from Assigned were almost all NOT chain members.
    Demote-only; `context=None` is a no-op."""
    if not context:
        return {"budget_demoted": 0, "budget_cf2_kept": 0}
    from peaky.chem import contexts as X
    profile = X.as_profile(context)
    open_f = dataclasses.replace(profile, max_F=10 ** 6)   # the budget with fluorine lifted
    curated = frozenset(curated or ())
    verdict: dict = {}
    n = kept = 0
    has_method = "method" in ledger.columns
    committed = _committed_pairs(ledger)
    for i in _m0_index(ledger):
        neutral = ledger.at[i, "neutral_formula"]
        if not isinstance(neutral, str) or not neutral.strip() or neutral in curated:
            continue
        # an ion-only row carries its parent acid's composition (evidence.py);
        # it is levelled on its own satellite and never judged here
        if has_method and str(ledger.at[i, "method"]).startswith("ion_only:"):
            continue
        if neutral not in verdict:
            verdict[neutral] = X.element_budget(neutral, profile)
        ok, why = verdict[neutral]
        if ok:
            continue
        adduct = ledger.at[i, "adduct"] if "adduct" in ledger.columns else None
        if (X.element_budget(neutral, open_f)[0]
                and any((nb, adduct) in committed for nb in _cf2_neighbours(neutral))):
            kept += 1          # fluorine is the only violation and its CF2 step is committed
            continue
        ni = _iso_count(ledger.at[i, "isotopologues"]) if "isotopologues" in ledger.columns else 0
        note = ledger.at[i, "degeneracy_note"] if "degeneracy_note" in ledger.columns else None
        reason = (f"outside the {profile.label} element budget ({why}) and on no curated list "
                  "-- a widened search proposed this formula; its axes confirm a neutral mass, "
                  "not this composition")
        # a lead, not a contradiction: the widened search's evidence proposed the
        # neutral mass and nothing tests this composition either way (C19(c))
        _demote_row(ledger, i, reason=reason, audit=audit, evidence=str(why),
                    degeneracy_note=note, n_iso=ni, lead="off_budget")
        n += 1
    log(f"[plausibility] demoted {n} commits outside the {profile.label} element budget "
        f"(not on a curated list); kept {kept} fluorinated CF2-series members")
    return {"budget_demoted": n, "budget_cf2_kept": kept}


def _committed_pairs(ledger: pd.DataFrame) -> set:
    """(neutral_formula, adduct) of every committed M0 row of the ledger."""
    if "neutral_formula" not in ledger.columns or "adduct" not in ledger.columns:
        return set()
    m0 = ledger.loc[_m0_index(ledger)]
    return {(str(n), a) for n, a in zip(m0["neutral_formula"], m0["adduct"]) if isinstance(n, str) and n}


def _cf2_neighbours(neutral: str) -> list:
    """The neutral one CF2 lighter and one CF2 heavier (the lighter only when it
    exists: at least one C and two F to remove)."""
    cnt = C.parse_formula(neutral)
    out = []
    for k in (-1, 1):
        c = dict(cnt)
        c["C"] = c.get("C", 0) + k
        c["F"] = c.get("F", 0) + 2 * k
        if c["C"] >= 1 and c["F"] >= 0:
            out.append(C.format_formula({el: v for el, v in c.items() if v}))
    return out


# ===========================================================================
# The element-evidence gate: the one stage here that CLEARS a commit
# ===========================================================================
#: the methods of the per-peak grid (it never proposes S / P / Si / F / Cl / Br
#: / I: build_ranges keeps them at zero), so a heteroatom there is not a widened
#: proposal; and the methods a curated list stands behind
#: the reason a peak cleared by `gate_element_evidence` carries (its commentary
#: reads 'CLEARED (element_evidence: ...)'): `cleared_by_element_evidence`
EE_REASON = "element_evidence"


def cleared_by_element_evidence(ledger: pd.DataFrame, i) -> bool:
    """Whether row `i` (an index label) is a peak `gate_element_evidence` cleared:
    unexplained, locked, its commentary 'CLEARED (element_evidence: ...)'."""
    if "commentary" not in ledger.columns or str(ledger.at[i, "role"]) != L.ROLE_UNEXPLAINED:
        return False
    c = ledger.at[i, "commentary"]
    return isinstance(c, str) and c.startswith(f"CLEARED ({EE_REASON}:")


_GRID_METHODS = ("cheminfo+grid", "grid")
_EXEMPT_METHODS = ("known:", "ion_only:")
#: the siloxane step: a committed neutral one C2H6OSi apart on the same adduct
#: pins the silicon the way a CF2 step pins fluorine (demote_off_budget)
_SILOXANE_UNIT = {"C": 2, "H": 6, "O": 1, "Si": 1}


def _siloxane_neighbours(neutral: str) -> list:
    """The neutral one C2H6OSi lighter (when it exists) and one heavier."""
    cnt = C.parse_formula(neutral)
    out = []
    for k in (-1, 1):
        c = {el: cnt.get(el, 0) + k * _SILOXANE_UNIT.get(el, 0) for el in set(cnt) | set(_SILOXANE_UNIT)}
        if all(v >= 0 for v in c.values()) and c.get("Si", 0) >= 1 and c.get("C", 0) >= 1:
            out.append(C.format_formula({el: v for el, v in c.items() if v}))
    return out


def gate_element_evidence(ledger: pd.DataFrame, *, profile, curated=frozenset(), cfg=None,
                          resolution=None, scoring=None, audit=None, log=print) -> dict:
    """CLEAR (L.clear_assignment) a committed M0 whose heteroatom the file's
    peak list contradicts -- the one rule of this module that removes a
    commit rather than demoting it.

    Every S / Cl / Br / Si / P / I formula a run commits came from a widened
    search: the per-peak grid proposes C / H / N / O only. Those searches
    propose a neutral MASS; nothing in them asks for the element's own line.
    For each M0 row whose neutral no curated list names (`curated`: the pass-0
    registry for this polarity/context plus the active reference lists), whose
    method is not known: / ion_only:, and which is not locked:

      * isotope elements -- Br, Cl and Si at any count, S at two or more, and
        any S / Cl / Br / Si count over the profile's cap: cleared when
        satellites.element_evidence says 'contradicted' (an exact-offset line
        the file's own calibration calls observable is absent or too low).
        'unobservable' keeps today's outcome (the budget demote's lead, the
        tier's twin test). Si whose committed C2H6OSi neighbour on the same
        adduct pins it (a siloxane ladder) is exempt, as the CF2 step exempts F.
      * monoisotopic P / I -- cleared where the profile budgets the element at
        0 (what demote_off_budget flags) and the formula came from a widened
        proposer other than pass 7 (which applies the same rule at its source,
        where it can see the certificate's channels: a reagent-acid pair is no
        independent evidence, three channels are).
      * F is untouched (demote_off_budget's CF2 rule).

    The class decides what is tested: Orbitrap-class (cfg.instrument_class)
    every isotope element at its exact offset; TOF-class Br / Cl by the ion's
    own M+2 line (tiers.tof_m2_verdicts' primitive, with the run's width model
    `resolution` -- none: no TOF verdict, as there), no S / Si verdicts;
    unknown class, or trace-first's synthetic sample (batch-mean heights),
    only the P / I rule. A cleared peak is locked (it stays unexplained; only
    the final envelope sweep may claim it as an isotope line). Returns counts;
    one audit row per cleared peak."""
    from peaky.assignment import satellites as SAT
    out = {"tested": 0, "cleared": 0, "cleared_isotope": 0, "cleared_mono": 0, "si_ladder_kept": 0,
           "confirmed": 0, "contradicted": 0, "unobservable": 0}
    if ledger is None or not len(ledger) or profile is None or "role" not in ledger.columns:
        return out
    curated = frozenset(curated or ())
    klass = getattr(cfg, "instrument_class", None) if cfg is not None else None
    if cfg is not None and getattr(cfg, "trace_sample", False):
        klass = None
    tof_floor = tof_win = tof_fwhm = None
    if klass == "tof":
        tof_floor = T.tof_assign_floor(cfg)
        sc = scoring if isinstance(scoring, dict) else {}
        tof_win = SAT.heavy_line_window_ppm(sc.get("sigma_ppm"), sc.get("mz_tolerance_ppm"))
        # the run's width model, as tiers.tof_m2_verdicts reads it; none (offline,
        # --resolving-power none): no TOF verdict, as there
        if resolution is not None:
            from peaky.chem.resolution import Resolution
            try:
                _rp = Resolution.coerce(resolution)
            except (TypeError, ValueError):
                _rp = None
            if _rp is not None:
                tof_fwhm = lambda m, _rp=_rp: float(T._fwhm_at(_rp, m))   # noqa: E731
    ctx = SAT.evidence_context(ledger, klass=klass, cal_sigma=getattr(cfg, "cal_sigma", None),
                               resolution=resolution, tof_floor=tof_floor, tof_win_ppm=tof_win,
                               tof_fwhm=tof_fwhm)
    committed = _committed_pairs(ledger)
    has_locked = "locked" in ledger.columns
    todo = []
    for i in _m0_index(ledger):
        neutral = ledger.at[i, "neutral_formula"]
        if not isinstance(neutral, str) or not neutral.strip() or neutral in curated:
            continue
        # the proposer's own method: a re-arbitrated row ('rearb<-known:...')
        # keeps its proposer's exemption and pass 7's own P / I judgement
        method = SAT.base_method(ledger.at[i, "method"]) if "method" in ledger.columns else ""
        if method.startswith(_EXEMPT_METHODS):
            continue
        if has_locked and L._truthy(ledger.at[i, "locked"]):
            continue
        nc = C.parse_formula(neutral)
        if not any(nc.get(el, 0) for el in ("Br", "Cl", "S", "Si", "P", "I")):
            continue
        adduct = ledger.at[i, "adduct"] if "adduct" in ledger.columns else None
        reasons, evid = [], []
        # monoisotopic P / I, budgeted at 0, from a widened proposer (not pass 7)
        widened = not method.startswith(_GRID_METHODS) and not method.startswith("certified:")
        for el in ("P", "I"):
            if nc.get(el, 0) and getattr(profile, f"max_{el}", 99) == 0 and widened:
                reasons.append(f"{el}{nc[el]} outside the {profile.label} budget (max_{el} 0): {el} is "
                               f"monoisotopic, no line can confirm it, and {method or 'a widened search'} "
                               "proposed only the neutral mass")
                evid.append(f"{el}{nc[el]}")
        mono = bool(reasons)
        # the isotope elements
        ion = T._ion_counts(neutral, adduct) or SAT._ion_body(ledger.at[i, "ion_formula"]
                                                                if "ion_formula" in ledger.columns else None)
        tested_any = False
        if ion and klass in ("orbitrap", "tof"):
            try:
                mono_mz = C.ion_mz(neutral, str(adduct))
            except Exception:  # noqa: BLE001 -- an unparseable adduct: no mono check
                mono_mz = float("nan")
            mz0 = pd.to_numeric(ledger.at[i, "mz"], errors="coerce")
            h0 = pd.to_numeric(ledger.at[i, "height"], errors="coerce")
            if "assigned_fraction" in ledger.columns:
                af = pd.to_numeric(ledger.at[i, "assigned_fraction"], errors="coerce")
                if pd.notna(af) and 0 < float(af) <= 1 and pd.notna(h0):
                    h0 = float(h0) * float(af)
            on_mono = not (np.isfinite(mono_mz) and pd.notna(mz0) and abs(float(mz0) - mono_mz) > SAT.EE_MONO_DA)
            for el in ("Br", "Cl", "Si", "S"):
                n = int(nc.get(el, 0))
                if not n:
                    continue
                cap = getattr(profile, f"max_{el}", 99)
                if el == "S" and not (n >= 2 or n > cap):
                    continue
                if el == "Si" and any((nb, adduct) in committed for nb in _siloxane_neighbours(neutral)):
                    out["si_ladder_kept"] += 1
                    continue
                if not on_mono:
                    continue
                v = SAT.element_evidence(ctx, mz0, h0, nc, ion, el)
                tested_any = True
                out[v["verdict"]] += 1
                if v["verdict"] == SAT.EE_CONTRADICTED:
                    reasons.append(f"{el}{n} contradicted by its isotope line: {v['why']}")
                    evid.append(f"{el}{n} contradicted")
        out["tested"] += int(tested_any)
        if reasons:
            todo.append((i, neutral, mono, reasons, evid))
    for i, neutral, mono, reasons, evid in todo:
        pid = ledger.at[i, "peak_id"]
        before = str(ledger.at[i, "tier"]) if "tier" in ledger.columns else ""
        ni = _iso_count(ledger.at[i, "isotopologues"]) if "isotopologues" in ledger.columns else 0
        note = ledger.at[i, "degeneracy_note"] if "degeneracy_note" in ledger.columns else None
        mz = ledger.at[i, "mz"] if "mz" in ledger.columns else None
        reason = f"{EE_REASON}: " + "; ".join(reasons)
        try:
            L.clear_assignment(ledger, pid, reason=reason)
        except L.LedgerError:
            continue
        # ... and LOCKED for the rest of the run: the file's own evidence just
        # emptied it, so no later stage commits a new reading on it (the run's
        # proposers have all run; the reference-list rescue and the ion-only
        # rows would put one there). The final envelope sweep may still claim it
        # as a committed M0's isotope line (passes.complete_isotope_envelopes):
        # that explains the line, it proposes nothing.
        L.lock_peaks(ledger, [pid])
        out["cleared"] += 1
        out["cleared_mono" if mono else "cleared_isotope"] += 1
        if audit is not None:
            audit.append({"mz": mz, "neutral_formula": neutral, "before_tier": before,
                          "after_tier_or_role": L.ROLE_UNEXPLAINED, "reason": reason,
                          "evidence": "; ".join(evid),
                          "degeneracy_note": ("" if note is None or (isinstance(note, float) and pd.isna(note))
                                              or note is pd.NA else str(note)),
                          "n_iso": ni})
    log(f"[element_evidence] {ctx.describe()}; rules: "
        + ("isotope lines + " if klass in ("orbitrap", "tof") else "")
        + f"P/I off-budget (widened proposers); {out['tested']} rows tested "
        f"({out['confirmed']} confirmed, {out['contradicted']} contradicted, {out['unobservable']} unobservable), "
        f"cleared {out['cleared']} ({out['cleared_isotope']} isotope, {out['cleared_mono']} P/I); "
        f"{out['si_ladder_kept']} Si rows pinned by a siloxane step")
    out["instrument_class"] = klass
    out["calibration"] = ctx.cal.describe() if klass == "orbitrap" else None
    return out


def demote_implausible(ledger: pd.DataFrame, *, audit=None, log=print,
                       context=None, curated=frozenset()) -> dict:
    """The shared-oracle demotes that fire on a single-file or merged ledger
    without a time series: O-monster + carbon-cluster, and -- given the run's
    `context` (a name, or the run's ContextProfile with its run-level switches,
    which the O-monster and carbon-cluster reads honour) -- the element-budget
    demote (`demote_off_budget`). All are demote-only and feed the same audit
    list."""
    profile = None
    if context:
        from peaky.chem import contexts as X
        profile = X.as_profile(context)
    o = demote_oxygen_monsters(ledger, audit=audit, log=log, profile=profile)
    c = demote_carbon_clusters(ledger, audit=audit, log=log, profile=profile)
    if not context:
        return {**o, **c}
    b = demote_off_budget(ledger, context=context, curated=curated, audit=audit, log=log)
    return {**o, **c, **b}


_AUDIT_COLS = ["mz", "neutral_formula", "before_tier", "after_tier_or_role",
               "reason", "evidence", "degeneracy_note", "n_iso"]


def write_audit(audit: list, path: str) -> int:
    """Write the plausibility audit (one row per touched peak) to `path`, sorted
    deterministically by mz then formula. Always writes the header (an empty audit
    still produces a 1-line CSV so the artifact set is stable). Returns the row
    count."""
    df = pd.DataFrame(audit, columns=_AUDIT_COLS)
    if len(df):
        df = df.sort_values(["mz", "neutral_formula"], na_position="last").reset_index(drop=True)
    df.to_csv(path, index=False)
    return len(df)
