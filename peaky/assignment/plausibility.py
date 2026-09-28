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

__version__ = "0.4.0"   # element-budget demote (demote_off_budget); 0.3.0 carbon-cluster rule: F no longer exempts

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


def is_oxygen_monster(cnt: dict) -> bool:
    """O/C strictly above OC_MONSTER (the oxygen-lattice mass-fit ratio). Pure
    arithmetic -- the DEMOTE additionally gates on degeneracy mass-saturation."""
    return cnt.get("C", 0) > 0 and _oc(cnt) > OC_MONSTER


def is_carbon_cluster(cnt: dict) -> bool:
    """C>=2 skeleton whose DBE/C >= DBE_PER_C_MONSTER, EXCLUDING radicals
    (half-integer DBE are exempt). H+halogen <= N+2 is the equivalent integer
    test (C.dbe already counts F/Cl/Br like H), but we compute the real DBE so
    the half-integer radical exemption is exact. F-bearing skeletons are NOT
    exempt: dbe() counting F as an H-equivalent means real perfluoro classes
    (PFCA DBE/C~0.2) can never trip this gate, while C12HF / C25H3F3 style
    bare-carbon fits (DBE/C~1) previously slipped through on an F-free clause
    (2026-07-04 [15N]-nitrate run: 15 such F-decorated clusters tier-Assigned).
    """
    nc = cnt.get("C", 0)
    if nc < 2:
        return False
    if C.odd_electron(cnt):            # half-integer DBE -> radical, EXEMPT
        return False
    return C.dbe(cnt) / nc >= DBE_PER_C_MONSTER


def implausible(neutral_formula: str, *, tier: str | None = None,
                polarity: str | None = None) -> str | None:
    """Return a short reason string if `neutral_formula` looks like a mass-coincidence
    fit rather than a real molecule, else None. Only Candidate-tier is scrutinised
    (pass tier=None to scrutinise regardless). `polarity` ('+'/'-') enables the
    wrong-mode-halogen check."""
    if tier is not None and str(tier) != "Candidate":
        return None
    c = C.parse_formula(str(neutral_formula))
    nc = c.get("C", 0)
    if nc == 0:
        return None                      # carbon-free handled elsewhere (reagent/inorganic)
    h, n, o = c.get("H", 0), c.get("N", 0), c.get("O", 0)
    f = c.get("F", 0)
    br, cl = c.get("Br", 0), c.get("Cl", 0)
    hc, oc = h / nc, o / nc
    # Terse labels (the full meaning is spelled out in the scrutiny-page legend);
    # keeping them short stops the table overflowing the page width.
    if is_oxygen_monster(c):    # O/C beyond the HOM ceiling -> oxygen-lattice monster
        return f"O/C {oc:.1f} (oxygen-lattice monster)"
    if n >= N_HIGH_OC and oc >= OC_HIGH:
        return f"N{n}, O/C {oc:.1f} (heteroatom coincidence)"
    if n >= N_VERY_HIGH and o >= O_HIGH:
        return f"N{n}O{o} (heteroatom coincidence)"
    if f >= F_HIGH:           # heavily fluorinated: 19F is 100% monoisotopic
        # NB any 13C/81Br satellites the row carries confirm the CARBON count / the
        # adduct halogen, NOT the fluorine -- 19F has no heavier stable isotope, so
        # the F COUNT is never isotope-confirmable (do NOT say "no isotope twin").
        return f"F{f}: 19F monoisotopic, fluorine count not isotope-confirmable"
    if is_carbon_cluster(c):  # DBE/C>=1.0, F-free, integer-DBE (radicals exempt)
        return f"DBE/C {C.dbe(c) / nc:.2f} (carbon cluster, H<=N+2)"
    if f == 0 and hc < HC_FLOOR:     # genuine carbon-rich skeleton (F not displacing H)
        return f"H/C {hc:.2f} (carbon-rich)"
    if polarity == "+" and (br > 0 or cl > 0):
        return "halogen in neutral, +mode"
    return None


def scan(merged, *, polarity: str | None = None) -> list[dict]:
    """Flag Candidate-only neutrals that look implausible. Returns one dict per
    distinct neutral: {neutral_formula, reason, ion_score, tier}. A neutral that is
    Assigned in any ion channel is excluded (it is corroborated)."""
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
        reason = implausible(f, tier=best, polarity=polarity)
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


def demote_oxygen_monsters(ledger: pd.DataFrame, *, audit=None, log=print) -> dict:
    """Demote M0 assignments that are oxygen-lattice 'monsters': O/C > OC_MONSTER
    AND mass-degenerate (the degeneracy audit counts >= 3 plausible ions in the
    calibrated window, or flags it MASS-SATURATED). NOT niso-gated -- a 13C satellite
    confirms the carbon count, not the oxygen count, so it would wrongly exempt a
    real O-monster. Real HOMs (O/C<=1.14) are spared by the ratio cut; high-O fits
    on a unique or two-ion window (the small polyacids: oxalic, malonic ...) are
    spared by the second leg. Assigned->Candidate + below_assignability. Demote-only."""
    n = 0
    has_note = "degeneracy_note" in ledger.columns
    for i in _m0_index(ledger):
        cnt = C.parse_formula(str(ledger.at[i, "neutral_formula"] or ""))
        if not is_oxygen_monster(cnt):
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


def demote_carbon_clusters(ledger: pd.DataFrame, *, audit=None, log=print) -> dict:
    """Demote M0 assignments resting on a bare-carbon skeleton: DBE/C >=
    DBE_PER_C_MONSTER (H+halogen<=N+2), C>=2, with the HALF-INTEGER-DBE radical
    EXEMPTION (radicals carry half-integer DBE; carbon-cluster monsters are
    integer-DBE). This is distinct from the H/C<0.35 carbon-rich demote in
    cleanup.py (kept unchanged): the two together cover the carbon-coincidence
    family without catching real aromatics (pyridine/coumarin/furfural sit below
    DBE/C 1.0). Assigned->Candidate + below_assignability. Demote-only."""
    n = 0
    for i in _m0_index(ledger):
        cnt = C.parse_formula(str(ledger.at[i, "neutral_formula"] or ""))
        if not is_carbon_cluster(cnt):
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


def demote_off_budget(ledger: pd.DataFrame, *, context: str | None,
                      curated=frozenset(), audit=None, log=print) -> dict:
    """Demote M0 commits whose neutral lies outside the run context's ELEMENT
    BUDGET (contexts.element_budget: the structural gate, the carbon-free
    allowlist, the heteroatom caps) and that no curated list names.

    The per-peak grid never proposes such a formula -- ambient-air keeps P, F and
    I at zero because they are monoisotopic and can never be isotope-confirmed,
    and S at one. Other commit paths widen the search on evidence of their own
    (a multi-channel certificate, a series extrapolation, a contaminant family)
    and CAN commit one; that evidence proposes the neutral MASS, and it is then
    read back as the axes (chan2, the acid branch, an anchor) that the evidence
    level and the tier count as confirmation. So an off-budget formula that no
    curated list names is Candidate + tentative_lead (C19(c): the proposal is
    unsupported, not contradicted; before the split it was below_assignability)
    -- the evidence level reads that as 5b. `curated` is every formula the pass-0 registry names for
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
    profile = X.get_context(context)
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


def demote_implausible(ledger: pd.DataFrame, *, audit=None, log=print,
                       context: str | None = None, curated=frozenset()) -> dict:
    """The shared-oracle demotes that fire on a single-file or merged ledger
    without a time series: O-monster + carbon-cluster, and -- given the run's
    `context` -- the element-budget demote (`demote_off_budget`). All are
    demote-only and feed the same audit list."""
    o = demote_oxygen_monsters(ledger, audit=audit, log=log)
    c = demote_carbon_clusters(ledger, audit=audit, log=log)
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
