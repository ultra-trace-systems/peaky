"""Reference peaklists — a curated catalog of known-molecule formula lists from
published chemical systems (oxidation HOM, contaminant families, ...), used as a
SOFT, context-gated prior.

Two uses, both decoupled from the expensive server scoring (one offline pass):
  * corroborate  -- a Candidate-tier neutral whose formula is on a list for the
                    sample's chemistry gets a literature-corroboration tag.
  * rescue/annotate -- an UNEXPLAINED peak (no formula) is matched BY MASS, under
                    the run's reagent adducts, against the list formulas, turning
                    the remainder into leads the reader can chase.

This is NOT the Pass-0 known-species lock (`passes._known_species`, for
contaminants the grid cannot reach). It is a post-hoc PRIOR: it never overrides an
isotope-scored Assigned, never fabricates a peak, and every match carries the
source list id for provenance. Masses are recomputed for whatever reagent adduct
the run uses (`chemistry.ion_mz`), so one list serves Br-/NO3-/I-/urea+ runs alike.

A list is one self-describing JSON in data/peaklists/ (see README.md). Add a
credible source = drop in one JSON; nothing else couples.
"""
from __future__ import annotations

import bisect
import glob
import json
import os
import warnings
from dataclasses import dataclass

from peaky.chem import chemistry as C
from peaky.paths import pkg_data

__version__ = "0.1.0"

# Bundled peaklist catalog. Resolved from the package root (paths.pkg_data), not
# this module's __file__, so it survives `reflists` moving into a sub-package.
_DIR = pkg_data("peaklists")

# batch-name / label keywords -> experimental-context tag (the metadata "unlock").
# Conservative: only fire on clearly source-diagnostic words.
CONTEXT_KEYWORDS = {
    "monoterpene_ox": ("monoterpene", "pinene", "a-pinene", "α-pinene", "apinene",
                       "b-pinene", "β-pinene", "limonene", "terpene", "orange",
                       "carene", "sabinene", "myrcene", "ap oxidation",
                       "ap-oxidation", "ap_oxidation"),
    "limonene_ox": ("limonene", "orange", "d-limonene"),
    # "ap oxidation" as a PHRASE: a dataset named for the chemistry ("AP oxidation
    # demo-set") must unlock the alpha-pinene lists; the bare abbreviation may not
    # (see below).
    "ap_ox": ("pinene", "a-pinene", "α-pinene", "apinene", "ap oxidation",
              "ap-oxidation", "ap_oxidation"),
    "isoprene_ox": ("isoprene", "isopn", "iepox", "isopooh", "methacrolein"),
    # NB a bare "AP" abbreviation (e.g. "AP Low temperature") is intentionally NOT a
    # keyword (too many false positives like "soap"/"grape"); tag such runs via the
    # explicit context/--reagent-config path instead.
}


@dataclass(frozen=True)
class ReferenceList:
    id: str
    system: str
    label: str
    data_version: str
    polarity: str
    native_detection: str
    applies_to_contexts: tuple
    references: tuple
    formulas: frozenset            # closed-shell neutral formulas (matchable)
    radicals: frozenset            # odd-electron (half-integer DBE) formulas (excluded by default)
    conditions_of: dict            # formula -> tuple(conditions)
    source_file: str
    always_active: bool = False    # universal lists (e.g. contaminants) ignore context gating
    meta_of: dict = None           # formula -> {name, origin, ...} (display extras)
    skipped: tuple = ()            # entries the loader refused: not a neutral molecule (`_not_a_neutral`)

    def pool(self, include_radicals: bool = False) -> frozenset:
        return self.formulas | self.radicals if include_radicals else self.formulas

    def cite(self) -> str:
        if not self.references:
            return self.label
        r = self.references[0]
        bits = [r.get("authors"), r.get("title"), r.get("publisher"),
                str(r.get("year", "")), r.get("section")]
        return ", ".join(b for b in bits if b)


# ---------------------------------------------------------------------------
def is_radical(formula: str) -> bool:
    """True for an odd-electron neutral: a half-integer DBE, read by
    `chemistry.odd_electron` (the parity test the grid gate and the plausibility
    exemption share). Parity is the rule, not the H count -- an organic nitrate
    like C10H15NO8 has odd H and an integer DBE, so it is closed-shell."""
    return C.odd_electron(formula)


def _not_a_neutral(formula) -> str | None:
    """Why `formula` cannot be handed to the parity test, or None when it can.

    The parity test only means something for a neutral molecule written over the
    mass table: `chemistry.dbe` scores an element it does not know as divalent
    (sodium acetate C2H3NaO2 would come out DBE 1.5 and pool as a radical, with
    no message), and `parse_formula` drops charge and bracket notation (the ion
    '[C10H14NO8]-' would load as the closed-shell neutral C10H14NO8). So an entry
    must round-trip `format_formula(parse_formula(f)) == f` (Hill notation, no
    charge), use mass-table elements only, and have a non-negative DBE (a
    negative one is an ion or a salt: a quaternary-ammonium cation sits at -0.5)."""
    cnt = C.parse_formula(formula)
    unknown = sorted(e for e in cnt if e not in C.M)
    if unknown:
        return f"element {', '.join(unknown)} not in the mass table"
    if C.format_formula(cnt) != formula:
        return (f"does not read back as {C.format_formula(cnt) or 'a formula'} "
                "(charge, bracket or non-Hill notation)")
    d = C.dbe(cnt)
    if d < 0:
        return f"DBE {d:g} < 0, an ion or a salt"
    return None


def load_catalog(directory: str | None = None) -> dict:
    """Load every *.json reference list under `directory` (default: packaged
    data/peaklists). Returns {id: ReferenceList}.

    A species goes to `formulas` or `radicals` by its formula's DBE parity
    (`is_radical`). Its `radical` value (false when absent) is the author's
    claim, checked but never deciding: a list whose claims disagree with parity
    loads with a warning naming them. An entry whose formula is not a neutral
    molecule over the mass table (`_not_a_neutral`) is skipped with a warning
    and recorded in `ReferenceList.skipped`, never pooled."""
    directory = directory or _DIR
    out: dict = {}
    for p in sorted(glob.glob(os.path.join(directory, "*.json"))):
        with open(p, encoding="utf-8") as fh:
            d = json.load(fh)
        sp = d.get("species", [])
        closed, rad, cond, meta, wrong, skipped, why = set(), set(), {}, {}, [], [], []
        for s in sp:
            f = s.get("formula")
            if not f:
                continue
            reason = _not_a_neutral(f)
            if reason:
                skipped.append(f)
                why.append(f"{f}: {reason}")
                continue
            radical = is_radical(f)
            if bool(s.get("radical", False)) != radical:
                wrong.append(f)
            (rad if radical else closed).add(f)
            cond[f] = tuple(s.get("conditions", ()))
            extra = {k: s[k] for k in ("name", "origin", "modes") if k in s}
            if extra:
                meta[f] = extra
        if skipped:
            warnings.warn(
                f"{os.path.basename(p)}: {len(skipped)} species skipped, not a neutral "
                f"molecule over the mass table ({'; '.join(why[:5])}"
                f"{'; ...' if len(why) > 5 else ''})", stacklevel=2)
        if wrong:
            warnings.warn(
                f"{os.path.basename(p)}: the radical flag (false when absent) of "
                f"{len(wrong)} species disagrees with its formula's DBE parity "
                f"({', '.join(wrong[:5])}"
                f"{', ...' if len(wrong) > 5 else ''}); parity decides -- run "
                "tests/test_peaklists.py for the full list", stacklevel=2)
        out[d["id"]] = ReferenceList(
            id=d["id"], system=d.get("system", ""), label=d.get("label", d["id"]),
            data_version=str(d.get("data_version", "")),
            polarity=d.get("polarity", ""), native_detection=d.get("native_detection", ""),
            applies_to_contexts=tuple(d.get("applies_to_contexts", ())),
            references=tuple(d.get("references", ())),
            formulas=frozenset(closed), radicals=frozenset(rad), conditions_of=cond,
            source_file=os.path.basename(p),
            always_active=bool(d.get("always_active", False)), meta_of=meta,
            skipped=tuple(skipped))
    return out


def resolve_context_tags(*texts: str) -> set:
    """Infer experimental-context tags from run metadata: the batch name, the
    DATASET name and the reagent label, all passed as `texts`. This is the
    'unlock with metadata' step — only the matched contexts' lists become active,
    keeping precision high on unrelated samples. Pass the dataset name too: on a
    campaign whose batch names describe instrument and reagent ("... TOF NO3_Br
    mixed ...") the chemistry lives only in the dataset name ("AP oxidation
    demo-set"), and a resolver that never saw it left the alpha-pinene reference
    list inactive on alpha-pinene data."""
    blob = " ".join(t for t in texts if t).lower()
    return {tag for tag, kws in CONTEXT_KEYWORDS.items()
            if any(k.lower() in blob for k in kws)}


def active_lists(catalog: dict, *, context_tags=()) -> list:
    """Select active lists: every `always_active` list (ubiquitous, e.g. lab
    contaminants) PLUS any whose `applies_to_contexts` intersect the run's context
    tags (the metadata unlock). Polarity is NOT a filter: a neutral-formula library
    transfers across reagent ion forms."""
    tags = set(context_tags)
    return [L for L in catalog.values()
            if L.always_active or (tags and set(L.applies_to_contexts) & tags)]


def active_versions(lists) -> list:
    """`[(id, data_version), ...]` of the lists a run had active -- the manifest
    line that says which list versions shaped its selection prior and rescue
    (a list's split changes with its `data_version`, e.g. the HOM list's 2024.2
    parity correction moved 118 nitrates into the matched pool)."""
    return [(L.id, L.data_version) for L in (lists or ())]


# ---------------------------------------------------------------------------
def match_assigned(neutral_formulas, lists, *, include_radicals: bool = False) -> dict:
    """Formula-membership corroboration: which assigned neutrals appear on a list.
    Returns {formula: [{list, conditions}]}."""
    res: dict = {}
    for f in {str(x) for x in neutral_formulas if x and str(x) != "nan"}:
        hits = [{"list": L.id, "conditions": list(L.conditions_of.get(f, ()))}
                for L in lists if f in L.pool(include_radicals)]
        if hits:
            res[f] = hits
    return res


def _target_table(lists, adducts, include_radicals: bool):
    """Sorted [(mz, formula, adduct, list_id)] of every list formula under every
    reagent adduct — built once per match call."""
    rows = []
    for L in lists:
        for f in L.pool(include_radicals):
            for a in adducts:
                try:
                    rows.append((C.ion_mz(f, a), f, a, L.id))
                except Exception:
                    continue
    rows.sort(key=lambda r: r[0])
    return rows


def rescue_unexplained_by_reflist(client, sample_id, ledger, profile, cfg, lists,
                                  adducts, *, score_fn=None, tol_ppm: float = 4.0,
                                  log=print) -> dict:
    """RESCUE-VERIFY: match UNEXPLAINED peaks by mass to active reference-list
    formulas, then SCORE those specific formulas with the server (match_compounds).

    Decision per matched peak (mass gate: server ion_score >= tau_low AND on-cal z):
      * isotope-CONFIRMED  -> commit literature-anchored M0 (Good/Assigned-grade);
      * too DIM to confirm (the predicted 13C M+1 falls below height_cutoff, so no
        satellite COULD show) -> commit a low-quality Candidate + below_assignability
        so the lead is never lost back to 'unexplained' (the user's small-peak rule);
      * isotopes EXPECTED (bright enough) but absent, or off-cal / poor score
        -> leave unexplained (a real mass coincidence, not corroborated).

    A soft, provenance-tagged rescue: never overrides an existing assignment (only
    ROLE_UNEXPLAINED peaks are touched), every commit records the source list."""
    import pandas as pd

    from peaky.io import io_mascope as IO
    from peaky.assignment import ledger as L
    score_fn = score_fn or IO.score_candidates
    if not lists:
        return {"rescued": 0, "tentative": 0}
    un = ledger[ledger["role"] == L.ROLE_UNEXPLAINED].dropna(subset=["mz"])
    if not len(un):
        return {"rescued": 0, "tentative": 0}
    mz_by_pid = {pid: float(m) for pid, m in zip(un["peak_id"], un["mz"])}
    matches = match_by_mass(list(mz_by_pid.values()), lists, adducts, tol_ppm=tol_ppm)
    by_mz = {round(m, 5): pid for pid, m in mz_by_pid.items()}
    want, allf = {}, set()
    for m in matches:
        pid = by_mz.get(round(m["obs_mz"], 5))
        if pid is not None and pid not in want:
            want[pid] = (m["formula"], m["adduct"], m["list"])
            allf.add(m["formula"])
    if not allf:
        return {"rescued": 0, "tentative": 0}
    fr = score_fn(client, sample_id, sorted(allf), allow_partial=True,
                  mechanism_ids=getattr(cfg, "mechanism_ids", None))
    if fr is None or not len(fr):
        return {"rescued": 0, "tentative": 0}
    fr = fr[fr["sample_peak_id"].notna()]
    mu = getattr(cfg, "cal_mu", None)
    sigma = getattr(cfg, "cal_sigma", None) or 0.5
    z_acc = getattr(cfg, "cal_z_accept", 2.0)
    floor = getattr(cfg, "tau_low", 0.70)
    hcut = cfg.height_cutoff          # resolved gate (raises if unresolved: fail closed)
    rescued = tentative = 0
    for pid, (formula, adduct, lid) in want.items():
        idx = ledger.index[ledger["peak_id"] == pid]
        if not len(idx) or ledger.at[idx[0], "role"] != L.ROLE_UNEXPLAINED:
            continue
        i = idx[0]
        mz = float(ledger.at[i, "mz"])
        h = float(ledger.at[i, "height"]) if pd.notna(ledger.at[i, "height"]) else 0.0
        sub = fr[(fr["compound_formula"] == formula)
                 & ((fr["sample_peak_mz"] - mz).abs() < 0.006)]
        base = sub[sub["is_base"] & sub["ion_score"].notna()]
        if base.empty:
            continue
        top = base.sort_values("ion_score", ascending=False).iloc[0]
        score = float(top["ion_score"])
        ppm = float(top["ppm_error"]) if pd.notna(top["ppm_error"]) else None
        z = abs((ppm - mu) / sigma) if (mu is not None and ppm is not None) else 0.0
        if score < floor or ppm is None or z > z_acc:
            continue                                   # poor mass match / off-cal -> leave
        # isotope confirmation: a confirmed satellite sits at the M+1/M+2 mass, not
        # the base m/z -- scan the full scored frame for this formula, not `sub`.
        iso_ok = bool(len(fr[(fr["compound_formula"] == formula) & (~fr["is_base"])
                             & (pd.to_numeric(fr["iso_score"], errors="coerce").fillna(0) > 0.4)]))
        nC = C.parse_formula(formula).get("C", 0)
        iso_observable = 0.011 * nC * h >= hcut        # predicted 13C M+1 vs the floor
        srcs = next((Ls.cite().split(",")[0] for Ls in lists if Ls.id == lid), lid)
        # runs AFTER apply_tiers (like the F/carbon demotes), so set tier explicitly.
        if iso_ok:
            L.commit_assignment(ledger, pid, neutral_formula=formula, adduct=adduct,
                                ion_formula=str(top["ion_formula"]), ion_score=score,
                                compound_score=score, ppm_error=ppm, pass_no=8,
                                method=f"reflist-rescue:{lid}", confidence="Good (literature)",
                                commentary=(f"Reference-list match ({srcs}); server score "
                                            f"{score:.2f}, isotope-confirmed, z={z:.1f}."))
            ledger.at[i, "tier"] = "Assigned"
            rescued += 1
        elif not iso_observable:                       # too dim to confirm -> tentative
            L.commit_assignment(ledger, pid, neutral_formula=formula, adduct=adduct,
                                ion_formula=str(top["ion_formula"]), ion_score=score,
                                compound_score=score, ppm_error=ppm, pass_no=8,
                                method=f"reflist-rescue:{lid}",
                                confidence="Candidate (literature, dim)",
                                commentary=(f"Reference-list match ({srcs}); server score "
                                            f"{score:.2f}, z={z:.1f}. Too dim ({h:.0f} cps) to "
                                            "confirm isotopes -- tentative lead, not confirmed."))
            ledger.at[i, "tier"] = "Candidate"
            if "below_assignability" not in ledger.columns:
                ledger["below_assignability"] = False
            ledger.at[i, "below_assignability"] = True
            tentative += 1
        # else: bright enough to show isotopes but none confirmed -> mass coincidence, leave
    log(f"[reflist] rescue-verify: {rescued} confirmed + {tentative} tentative "
        f"(of {len(want)} reference-matched unexplained)")
    return {"rescued": rescued, "tentative": tentative}


def match_by_mass(mz_values, lists, adducts, *, tol_ppm: float = 5.0,
                  include_radicals: bool = False) -> list:
    """Rescue/annotate UNEXPLAINED peaks (which have no formula) BY MASS: for each
    observed m/z, find the closest list-formula ion (any reagent adduct) within
    `tol_ppm`. Returns one dict per matched observed peak (best match only):
    {obs_mz, formula, adduct, list, ppm, target_mz}."""
    targets = _target_table(lists, adducts, include_radicals)
    if not targets:
        return []
    tmz = [t[0] for t in targets]
    out = []
    for obs in mz_values:
        try:
            obs = float(obs)
        except (TypeError, ValueError):
            continue
        if obs <= 0:
            continue
        w = obs * tol_ppm * 1e-6
        lo = bisect.bisect_left(tmz, obs - w)
        hi = bisect.bisect_right(tmz, obs + w)
        best = None
        for j in range(lo, hi):
            m, f, a, lid = targets[j]
            ppm = (obs - m) / obs * 1e6
            if abs(ppm) <= tol_ppm and (best is None or abs(ppm) < abs(best["ppm"])):
                best = {"obs_mz": round(obs, 5), "formula": f, "adduct": a,
                        "list": lid, "ppm": round(ppm, 2), "target_mz": round(m, 5)}
        if best:
            out.append(best)
    return out
