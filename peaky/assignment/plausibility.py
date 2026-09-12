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

import json

import numpy as np
import pandas as pd

from peaky.chem import chemistry as C
from peaky.assignment import ledger as L

__version__ = "0.3.0"   # carbon-cluster rule: F no longer exempts (F counts as H)

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
#     genuine oxidation product. (The DEMOTE additionally requires mass-saturation
#     from the degeneracy audit; the plain reason-string oracle reports the ratio.)
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
# below_assignability). None of them can clear/delete a row, so the worst-case
# failure is an over-cautious Candidate, never a lost peak. Each appends one dict
# per touched peak to `audit` (when a list is passed) so assign/assign_batch can
# write tables/plausibility_audit_*.
# ===========================================================================

def _is_saturated(note) -> bool:
    """A degeneracy_note that flags the mass as saturated/degenerate -- the
    second leg of the O-monster demote (the ratio alone is not enough; the mass
    must also be one arbitrary pick of a degenerate set)."""
    s = "" if note is None or (isinstance(note, float) and pd.isna(note)) else str(note)
    low = s.lower()
    return "satur" in low or "degener" in low


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


def _demote_row(ledger, i, *, reason, audit, evidence, degeneracy_note, n_iso):
    """Demote one M0 -> Candidate + below_assignability, append the note, and log
    one audit record. Demote-only: tier moves Assigned->Candidate, nothing is
    cleared."""
    before = str(ledger.at[i, "tier"]) if "tier" in ledger.columns else ""
    if "tier" in ledger.columns and before == "Assigned":
        ledger.at[i, "tier"] = "Candidate"
    if "below_assignability" in ledger.columns:
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
    AND mass-saturated (the degeneracy audit flags ~dozens of plausible ions on
    the mass). NOT niso-gated -- a 13C satellite confirms the carbon count, not the
    oxygen count, so it would wrongly exempt a real O-monster. Real HOMs (O/C<=1.14)
    are spared by the ratio cut; non-saturated high-O fits are spared by the second
    leg. Assigned->Candidate + below_assignability. Demote-only."""
    n = 0
    has_note = "degeneracy_note" in ledger.columns
    for i in _m0_index(ledger):
        cnt = C.parse_formula(str(ledger.at[i, "neutral_formula"] or ""))
        if not is_oxygen_monster(cnt):
            continue
        note = ledger.at[i, "degeneracy_note"] if has_note else None
        if not _is_saturated(note):     # ratio alone is not enough -- needs saturation
            continue
        ni = _iso_count(ledger.at[i, "isotopologues"]) if "isotopologues" in ledger.columns else 0
        reason = (f"oxygen-lattice monster (O/C {_oc(cnt):.2f} > {OC_MONSTER}, "
                  "mass-saturated) -- one arbitrary pick of a sub-ppm-degenerate set")
        _demote_row(ledger, i, reason=reason, audit=audit,
                    evidence=f"O/C={_oc(cnt):.2f}", degeneracy_note=note, n_iso=ni)
        n += 1
    log(f"[plausibility] demoted {n} oxygen-lattice monsters (O/C>{OC_MONSTER}, mass-saturated)")
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


def demote_implausible(ledger: pd.DataFrame, *, audit=None, log=print) -> dict:
    """The two shared-oracle demotes that fire on a single-file or merged ledger
    without a time series: O-monster + carbon-cluster. Both are demote-only and
    feed the same audit list."""
    o = demote_oxygen_monsters(ledger, audit=audit, log=log)
    c = demote_carbon_clusters(ledger, audit=audit, log=log)
    return {**o, **c}


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
