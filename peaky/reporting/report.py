"""Reporting: turn the finished ledger into human-facing outputs.

  * build_sheets(ledger)  -> dict of DataFrames (pure; unit-tested offline)
  * write_excel(...)      -> styled multi-sheet .xlsx (needs openpyxl)
  * write_markdown(...)   -> narrative summary with commentary + alternatives

Tiered presentation (ROADMAP 2): committed assignments are split into
  Assigned    -- unique-in-window or independently corroborated
  Candidates  -- honest ambiguity, shown one row PER CANDIDATE FORMULA
and the unexplained residual keeps its evidence characterization
(iso-partner / has-constraints / isolated) as the below-assignability tier.

The commentary, alternatives, and tier reasons are generated mechanically
from ledger columns, so they are reproducible.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from peaky.chem import contexts as X
from peaky.assignment import evidence as EV
from peaky.assignment import ledger as L
from peaky.assignment import mass_only as MO
from peaky.assignment import tiers as T

__version__ = "0.5.0"  # the evidence scale of peaky 0.10.0 (the level columns, By claim over six keys)
                       # (history) By claim sheet; solvent-cluster method legend; Below-assignability sheet


def _alts_list(cell) -> list[dict]:
    try:
        v = json.loads(cell) if isinstance(cell, str) else (cell or [])
        return v if isinstance(v, list) else []
    except Exception:
        return []


def _alts_to_text(cell) -> str:
    parts = []
    for a in _alts_list(cell)[:3]:
        s = a.get("ion_score") or a.get("raw_score")
        ppm = a.get("ppm")
        seg = a.get("formula", "?")
        if s is not None:
            seg += f" (score {s:.2f}"
            if ppm is not None:
                seg += f", {ppm:.1f} ppm"
            seg += ")"
        parts.append(seg)
    return "; ".join(parts)


def _iso_to_text(cell) -> str:
    try:
        iso = json.loads(cell) if isinstance(cell, str) else (cell or [])
    except Exception:
        return ""
    # a satellite confirmed against the LEDGER (the isotope-locked recovery path)
    # carries no per-line server score; render its label rather than dropping it,
    # so the evidence a commit rests on is never silently blank in the report.
    return "; ".join(
        f"{i.get('label')}={i.get('score'):.2f}" if i.get("score") is not None
        else str(i.get("label"))
        for i in iso if i.get("label")
    )


_RESIDUAL_INTERPRETATION = {
    "iso-partner": ("heavy-isotope satellite of another residual peak -- "
                    "explained the moment its light partner is"),
    "has-constraints": ("isotope structure measured (carbon and/or halogen "
                        "count) -- a constrained formula solve is possible"),
    "isolated": ("no measurable isotope structure -- needs orthogonal "
                 "evidence (e.g. time-series correlation)"),
}


# ---------------------------------------------------------------------------
# the evidence scale (docs/EVIDENCE_LEVELS.md): reading a ledger's level columns
# ---------------------------------------------------------------------------
#: the level tokens the scale defines: LEVEL_ORDER (1 and 2 never assigned) + the
#: two buckets. Any other token in `evidence_level` is a level of a scale before
#: this one (a ledger written by an older peaky).
KNOWN_LEVELS = frozenset(EV.LEVEL_ORDER) | frozenset(EV.BUCKETS)
#: columns only a ledger levelled before the scale carries
OLD_LEVEL_COLUMNS = ("evidence_axes", "level_reason", "n_plausible_structures")
#: the long text columns the scale writes beside `evidence_level` and `claim`
LEVEL_TEXT_COLUMNS = tuple(c for c in EV.COLUMNS if c not in ("evidence_level", "claim"))
#: where a sheet shows the level: the claim, the level, then its text columns
LEVEL_COLUMNS = ["claim", "evidence_level", *LEVEL_TEXT_COLUMNS]
#: the scale's name in user-facing text (the release lives in one constant)
SCALE_NAME = f"the evidence scale of peaky {EV.SCALE_RELEASE}"
#: what every level letter of a ledger levelled on an older scale reads as -- the
#: letters both scales share (4a, 4b, 5a, 5b) included: an old 4a is not a new 4a
OLD_SCALE_NO_LEVEL = f"no level (pre-{EV.SCALE_RELEASE} scale)"
#: claim rank, best first: the four claims, then the two buckets
_CLAIM_RANK = {c: k for k, c in enumerate(EV.CLAIM_KEYS)}


def _blank(v) -> bool:
    """A cell with no value: None, NaN, pd.NA, or the text forms a CSV round
    trip leaves ('', 'nan', '<NA>')."""
    if v is None:
        return True
    if not isinstance(v, str):
        try:
            return bool(pd.isna(v))
        except (TypeError, ValueError):
            return False
    return v.strip() in ("", "nan", "<NA>")


def scale_view(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """A copy of `frame` with its level columns read for display, and what was
    found. Two repairs:

    - ``NA`` (not assessed) is a literal token that pandas' default CSV parser
      turns into NaN; a row whose `claim` says "not assessed" gets its ``NA``
      back (docs/OUTPUTS.md: read the claim column to tell NA from no level).
    - A ledger levelled on a scale before this one (it carries `evidence_axes`
      / `level_reason`, or a level letter this scale does not define): EVERY
      level letter reads as no level (`OLD_SCALE_NO_LEVEL`) -- the letters both
      scales share too, since an old 4a is not a 4a of this scale -- and the
      stored claim, read on the old scale, is dropped: every claim re-reads
      tentative.

    Returns (frame, info) with info = {old_scale, n_unknown (rows whose old
    letter reads as no level), unknown (the sorted old letters), n_na,
    na_reason}. A frame without `evidence_level` comes back unchanged with an
    empty info."""
    out = frame.copy()
    info = {"old_scale": False, "n_unknown": 0, "unknown": [], "n_na": 0, "na_reason": ""}
    if "evidence_level" not in out.columns:
        return out, info
    lv = pd.Series([pd.NA if _blank(v) else str(v).strip() for v in out["evidence_level"]],
                   index=out.index, dtype=object)
    if "claim" in out.columns:
        na_claim = out["claim"].astype(object).map(lambda v: not _blank(v) and str(v) == EV.CLAIM_NA)
        lv = lv.where(~(lv.isna() & na_claim), "NA")
    old = bool((lv.notna() & ~lv.isin(KNOWN_LEVELS)).any()) or any(c in out.columns for c in OLD_LEVEL_COLUMNS)
    unknown = lv.notna() if old else pd.Series(False, index=lv.index)
    if old:
        info.update(old_scale=True, n_unknown=int(unknown.sum()),
                    unknown=sorted(set(lv[unknown].astype(str))))
        lv = lv.where(~unknown, pd.NA)
        if "claim" in out.columns:
            out = out.drop(columns=["claim"])
    out["evidence_level"] = lv
    na = lv == "NA"
    info["n_na"] = int(na.sum())
    if na.any():
        reason = ""
        if "evidence" in out.columns:
            texts = [str(t) for t in out.loc[na, "evidence"] if not _blank(t)]
            reason = texts[0].split(" · ", 1)[1] if texts and " · " in texts[0] else ""
        info["na_reason"] = reason or EV.LEVEL_MEANING["NA"]
    return out, info


def scale_note(info: dict) -> str:
    """One sentence on a ledger levelled before the scale ('' otherwise)."""
    if not info.get("old_scale"):
        return ""
    if info.get("n_unknown"):
        letters = ", ".join(info["unknown"])
        return (f"levelled on a scale older than {SCALE_NAME}: its {info['n_unknown']} levelled row(s) "
                f"({letters}; the letters this scale shares too) read as {OLD_SCALE_NO_LEVEL}, and their claims "
                "as tentative")
    return (f"levelled on a scale older than {SCALE_NAME} (its level columns are not shown); no row carries a "
            "level of this scale")


def na_detail(info: dict) -> str:
    """The parenthetical of a not-assessed reason ('width model R(200) = 9 652 <
    50 000'), or the whole reason when it has none."""
    reason = str(info.get("na_reason") or EV.LEVEL_MEANING["NA"])
    head = EV.LEVEL_MEANING["NA"] + " ("
    return reason[len(head):-1] if reason.startswith(head) and reason.endswith(")") else reason


def tag_kind(tag: str) -> str:
    """The kind of one tag text: its words before the first ':', ' (' or ' -- ',
    without a trailing count ('series exclusion x2' -> 'series exclusion')."""
    head = re.split(r":| \(| -- ", str(tag), maxsplit=1)[0].strip()
    return re.sub(r" x\d+$", "", head)


def tag_kinds(tags) -> list[str]:
    """The sorted distinct kinds of a `tags` cell (tag texts joined by ' | ')."""
    if _blank(tags):
        return []
    return sorted({tag_kind(t) for t in str(tags).split(" | ") if t.strip()})


def _claims(m0: pd.DataFrame) -> pd.Series:
    """The claim of each M0 row (evidence.claim_class): the stored `claim`, else --
    a ledger written before the column existed, or one `scale_view` re-reads --
    read off `evidence_level`."""
    lv = m0["evidence_level"] if "evidence_level" in m0.columns else pd.Series(pd.NA, index=m0.index)
    derived = pd.Series([EV.claim_class(v) for v in lv], index=m0.index, dtype=object)
    if "claim" not in m0.columns:
        return derived
    stored = m0["claim"].astype(object)
    return stored.where(~stored.map(_blank), derived)


def claim_levels(claim: str) -> str:
    """The levels behind a claim, best first: '1 2 3c' (1 and 2 never assigned) /
    '4a' / '4b' / '5a 5b, no level' / 'reagent' / 'NA'."""
    if claim == "identified":
        return " ".join(lv for lv in EV.LEVEL_ORDER if lv in EV.CLAIM_IDENTIFIED)
    if claim == "neutral":
        return " ".join(lv for lv in EV.LEVEL_ORDER if lv in EV.CLAIM_NEUTRAL)
    if claim == "ion":
        return " ".join(lv for lv in EV.LEVEL_ORDER if lv in EV.CLAIM_ION)
    if claim == EV.CLAIM_REAGENT:
        return "reagent"
    if claim == EV.CLAIM_NA:
        return "NA"
    rest = [lv for lv in EV.LEVEL_ORDER if lv not in EV.CLAIM_IDENTIFIED | EV.CLAIM_NEUTRAL | EV.CLAIM_ION]
    return " ".join(rest) + ", no level"


def _enrich_m0(m0: pd.DataFrame) -> pd.DataFrame:
    cls = m0["neutral_formula"].map(lambda f: X.classify_compound(f))
    m0["compound_class"] = [c[0] for c in cls]
    m0["oxidation"] = [c[1] for c in cls]
    m0["heteroatoms"] = [c[2] for c in cls]
    m0["alternatives_text"] = m0["alternatives"].map(_alts_to_text)
    m0["isotopologues_text"] = m0["isotopologues"].map(_iso_to_text)
    return m0


_ASSIGN_COL_ORDER = [
    "mz", "height", "neutral_formula", "adduct", "ion_formula", "dbe",
    "compound_class", "oxidation", "heteroatoms", "ion_score",
    "compound_score", "ppm_error", "confidence", "candidate_density",
    "degeneracy_note", "composite_note", "tier_reason", "isotopologues_text",
    "alternatives_text", "pass_no", "method", "commentary", "peak_id",
]


def _candidate_rows(cand: pd.DataFrame, *, levels: bool = False) -> pd.DataFrame:
    """Explode Candidate peaks into one row per candidate formula: rank 1 is
    the committed winner, ranks 2+ are the stored alternatives. This is the
    'stop presenting one formula per peak' sheet. With `levels`, the rank-1 row
    carries the committed reading's `claim`, `evidence_level` and the level's
    text columns (evidence.COLUMNS); an alternative makes no claim."""
    rows = []
    for _, r in cand.iterrows():
        first = {
            "mz": r["mz"], "height": r["height"], "rank": 1,
            "formula": r["neutral_formula"], "adduct": r["adduct"],
            "score": r["ion_score"],
            "eff_score": r.get("eff_score", np.nan),
            "ppm_error": r["ppm_error"], "confidence": r["confidence"],
            "claim": r.get("claim", pd.NA),
            "evidence_level": r.get("evidence_level", pd.NA),
            **{c: r.get(c, pd.NA) for c in LEVEL_TEXT_COLUMNS},
            "candidate_density": r.get("candidate_density", pd.NA),
            "degeneracy_note": r.get("degeneracy_note", ""),
            "why_candidate": r.get("tier_reason", ""),
            "isotopologues": r.get("isotopologues_text", ""),
            "commentary": r["commentary"], "peak_id": r["peak_id"],
        }
        rows.append(first)
        for k, a in enumerate(_alts_list(r.get("alternatives")), start=2):
            rows.append({
                "mz": r["mz"], "height": r["height"], "rank": k,
                "formula": a.get("formula"), "adduct": a.get("adduct"),
                "score": a.get("raw_score") or a.get("ion_score"),
                "eff_score": a.get("eff_score"),
                "ppm_error": a.get("ppm"),
                "confidence": "", "claim": "", "evidence_level": "", "candidate_density": "",
                **{c: "" for c in LEVEL_TEXT_COLUMNS},
                "why_candidate": "", "isotopologues": "", "commentary": "",
                "peak_id": r["peak_id"],
            })
    cols = ["mz", "height", "rank", "formula", "adduct", "score", "eff_score",
            "ppm_error", "confidence", "candidate_density", "degeneracy_note",
            "why_candidate", "isotopologues", "commentary", "peak_id"]
    if levels:
        cols[cols.index("confidence") + 1:cols.index("confidence") + 1] = LEVEL_COLUMNS
    df = pd.DataFrame(rows, columns=cols)
    if len(df):
        df = (df.sort_values(["height", "mz", "rank"],
                             ascending=[False, True, True])
              .reset_index(drop=True))
    return df


def build_sheets(ledger: pd.DataFrame, context: str = "ambient-air",
                 sample_id: str = "", *, tof_flag_mz=None) -> dict[str, pd.DataFrame]:
    """Return the report sheets as DataFrames (insertion order == sheet order).
    A ledger carrying the TOF mass-only flag (assignment/mass_only.py) shows it on
    the Assigned sheet and counts it in the Summary, split at `tof_flag_mz`
    (None = the package default); a ledger without it renders as before."""
    led = ledger.copy()
    # tiering: stamp if the ledger does not carry it (e.g. an old CSV)
    if "tier" not in led.columns or led.loc[led["role"] == L.ROLE_M0, "tier"].isna().all():
        T.apply_tiers(led)
    if "composite_note" not in led.columns:   # old ledgers predate composite detection
        led["composite_note"] = pd.NA
    if "degeneracy_note" not in led.columns:   # old ledgers predate the degeneracy audit
        led["degeneracy_note"] = pd.NA
    # evidence levels (docs/EVIDENCE_LEVELS.md): the column set and the extra sheets
    # exist only when the ledger carries the level, so an unlevelled run renders
    # unchanged. `scale_view` restores the literal NA (not assessed) and reads a
    # ledger levelled before the scale (its unknown letters read as no level)
    has_levels = "evidence_level" in led.columns
    if has_levels:
        led, _info = scale_view(led)
        led = led.drop(columns=[c for c in OLD_LEVEL_COLUMNS if c in led.columns])
        for c in EV.COLUMNS:
            if c not in led.columns:
                led[c] = pd.NA
        # the claim (evidence.claim_class): an older ledger predates the column, so it
        # is read off the level -- on M0 rows only; an isotope child, a reagent ion or
        # an unexplained peak makes no claim
        is_m0 = led["role"] == L.ROLE_M0
        claim = (led["claim"].astype(object) if "claim" in led.columns
                 else pd.Series(pd.NA, index=led.index, dtype=object))
        claim[is_m0] = _claims(led[is_m0]).values
        claim[~is_m0] = pd.NA
        led["claim"] = claim
    m0 = led[led["role"] == L.ROLE_M0].copy()
    if len(m0):
        m0 = _enrich_m0(m0)

    ident = m0[m0["tier"] == T.TIER_ASSIGNED]
    cand = m0[m0["tier"] == T.TIER_CANDIDATE]

    acols = list(_ASSIGN_COL_ORDER)
    if has_levels:
        _j = acols.index("confidence") + 1
        acols[_j:_j] = LEVEL_COLUMNS
    # the TOF mass-only flag, beside the tier's own reason (only where the ledger carries it)
    has_flag = MO.counts(led) is not None
    if has_flag:
        _j = acols.index("tier_reason") + 1
        acols[_j:_j] = list(MO.COLUMNS)
    # the tier's own reason keeps its ledger name: `evidence` is the level's column
    identified = (ident[acols].sort_values("height", ascending=False)) if len(ident) else \
        pd.DataFrame(columns=acols)

    candidates = _candidate_rows(cand, levels=has_levels)

    # unassigned -- characterized by isotope structure (carbon/halogen count,
    # iso-partner class) so the residual is described, not just listed
    from peaky.assignment import residual as RD
    un = RD.characterize_residual(led)
    if len(un):
        un = un.rename(columns={"tier": "residual_evidence"})
        un["interpretation"] = un["residual_evidence"].map(_RESIDUAL_INTERPRETATION)
        un = un[["mz", "height", "residual_evidence", "twin_of", "c_count",
                 "n_Br", "n_Cl", "interpretation", "peak_id"]]

    # by class, with the tier split visible
    if len(m0):
        by_class = (m0.assign(is_ident=(m0["tier"] == T.TIER_ASSIGNED))
                    .groupby(["compound_class", "heteroatoms"])
                    .agg(n_peaks=("peak_id", "count"),
                         n_identified=("is_ident", "sum"),
                         signal=("height", "sum"))
                    .reset_index())
        by_class["n_candidates"] = by_class["n_peaks"] - by_class["n_identified"]
        by_class = (by_class[["compound_class", "heteroatoms", "n_peaks",
                              "n_identified", "n_candidates", "signal"]]
                    .sort_values("signal", ascending=False))
    else:
        by_class = pd.DataFrame()

    # unique formulas across channels
    if len(m0):
        uniq = (m0.assign(is_ident=(m0["tier"] == T.TIER_ASSIGNED))
                .groupby("neutral_formula")
                .agg(n_peaks=("peak_id", "count"),
                     adducts=("adduct", lambda s: "; ".join(sorted(set(map(str, s))))),
                     best_tier=("is_ident", lambda s: T.TIER_ASSIGNED if s.any()
                                else T.TIER_CANDIDATE),
                     best_score=("ion_score", "max"),
                     signal=("height", "sum"))
                .reset_index().sort_values("signal", ascending=False))
        if has_levels:
            # the neutral's best claim over its channels, beside its best tier (the
            # four claims first, then the reagent and not-assessed buckets)
            best = (m0.assign(_r=m0["claim"].map(_CLAIM_RANK))
                    .groupby("neutral_formula")["_r"].min().map(dict(enumerate(EV.CLAIM_KEYS))))
            uniq.insert(uniq.columns.get_loc("best_tier") + 1, "best_claim",
                        uniq["neutral_formula"].map(best).values)
    else:
        uniq = pd.DataFrame()

    # isotopologues, joined back to the parent so each row is self-describing
    iso = led[led["role"] == L.ROLE_ISO][[
        "peak_id", "mz", "height", "parent_peak_id", "iso_label",
        "iso_match_score"]].copy()
    if len(iso) and len(m0):
        parents = m0.set_index("peak_id")
        iso["parent_formula"] = iso["parent_peak_id"].map(parents["neutral_formula"])
        iso["parent_adduct"] = iso["parent_peak_id"].map(parents["adduct"])
        iso["parent_mz"] = iso["parent_peak_id"].map(parents["mz"])
        iso = (iso[["mz", "height", "iso_label", "iso_match_score",
                    "parent_formula", "parent_adduct", "parent_mz",
                    "parent_peak_id", "peak_id"]]
               .sort_values(["parent_mz", "mz"]))

    # ownership audit (one row per physical peak). A satellite's row names its
    # owner, not just its peak_id: this sheet is what a reviewer reads to ask who
    # claimed a peak, and "iso_child of 4f9a..." does not answer that.
    _own_cols = ["peak_id", "mz", "height", "role", "tier"] + (LEVEL_COLUMNS if has_levels else []) + [
                 "neutral_formula", "adduct", "ion_score", "ppm_error",
                 "confidence", "composite_note", "parent_peak_id",
                 "parent_neutral_formula", "parent_adduct",
                 "iso_label", "pass_no", "method", "commentary"]
    ownership = (led[[c for c in _own_cols if c in led.columns]]
                 .sort_values("height", ascending=False))

    # target list (formula + adduct + best ppm), Assigned first
    # ('Assigned' < 'Candidate' lexically, hence the ascending tier sort)
    _tcols = ["neutral_formula", "adduct", "ion_formula", "mz",
              "ppm_error", "ion_score", "confidence", "tier"] + (LEVEL_COLUMNS if has_levels else [])
    target = (m0[_tcols]
              .sort_values(["tier", "mz"], ascending=[True, True])) if len(m0) else pd.DataFrame()

    reag = led[led["role"] == L.ROLE_REAGENT][[
        "mz", "height", "commentary", "peak_id"]].copy().sort_values(
        "height", ascending=False)

    # below-assignability: M0 commits flagged as mass-saturated O-monsters -- the
    # base mass fits but ~dozens of plausible ions sit within <=1 ppm, so the
    # formula is one arbitrary pick, NOT an identification. Listed as a constrained
    # mass + the tie-set size, separated from the real Candidates. Since C19(c) the
    # flag has two halves -- below_assignability (the assignment argues with
    # itself) and tentative_lead (unsupported, not contradicted) -- and the sheet
    # lists both, with the `tentative_lead` column saying which and `lead_by`
    # which setter made it a lead (C11+b).
    if L.has_flags(led):
        bamask = (led["role"] == L.ROLE_M0) & L.flagged(led)
        bcols = [c for c in ["mz", "neutral_formula", "adduct", "ion_formula", "ppm_error",
                             "ion_score", "degeneracy_density", "degeneracy_note", "tier_reason",
                             L.FLAG_LEAD, L.LEAD_BY]
                 if c in led.columns]
        below = led[bamask][bcols].copy().sort_values("mz") if bamask.any() else pd.DataFrame(columns=bcols)
    else:
        below = pd.DataFrame()

    sheets = {
        "Summary": summary_stats(led, context=context, sample_id=sample_id,
                                 scale_info=_info if has_levels else None, tof_flag_mz=tof_flag_mz),
        "Read me": legend_sheet(claims=has_levels, mass_only=has_flag),
        "Assigned": identified,
        "Candidates": candidates,
        "Below assignability": below,
        "Unassigned": un,
        "By class": by_class,
        "Unique formulas": uniq,
        "Isotopologues": iso,
        "Peak ownership": ownership,
        "Target list": target,
        "Reagent ions": reag,
    }
    if has_levels:
        # the claim leads (the workbook opens on it); the level histogram sits right
        # after the two tier sheets it re-reads
        no_level = OLD_SCALE_NO_LEVEL if _info.get("old_scale") else "no level"
        out = {"By claim": claim_sheet(m0, no_level=no_level) if len(m0) else pd.DataFrame()}
        for k, v in sheets.items():
            out[k] = v
            if k == "Candidates":
                out["By evidence level"] = evidence_level_sheet(m0, no_level=no_level) if len(m0) else pd.DataFrame()
        sheets = out
    return sheets


def claim_sheet(m0: pd.DataFrame, n_bright: int = 20, *, no_level: str = "no level") -> pd.DataFrame:
    """The **By claim** sheet, the workbook's first: one `summary` row per claim
    key (evidence.CLAIM_KEYS: the four claims, then the reagent and not-assessed
    buckets, zeros kept) -- count and share of the M0 rows, summed height and its
    share, the tier split, the levels behind it -- then the `tier disagrees` rows
    (Assigned but tentative, Candidate but identified; brightest first) and the
    `n_bright` brightest rows of each claim. The tier is shown beside the claim,
    never read off it. Ion-only rows (`ion_only_of` set) are counted by their
    level and split out of their tier as `n_ion_only`."""
    cl = _claims(m0)
    lv = m0["evidence_level"].astype(object)
    h = pd.to_numeric(m0["height"], errors="coerce").fillna(0.0)
    h_m0 = float(h.sum())
    tier = m0["tier"].astype(object)
    io = (m0["ion_only_of"].notna() if "ion_only_of" in m0.columns
          else pd.Series(False, index=m0.index))
    pos = {k: i for i, k in enumerate([*EV.LEVEL_ORDER, *EV.BUCKETS])}
    rows = []
    for c in EV.CLAIM_KEYS:
        g = cl == c
        counts = lv[g].where(lv[g].notna(), no_level).astype(str).value_counts()
        hist = "; ".join(f"{k}: {v}" for k, v in sorted(counts.items(),
                                                         key=lambda kv: (-kv[1], pos.get(kv[0], 99))))
        rows.append({"section": "summary", "claim": c, "meaning": EV.CLAIM_MEANING[c],
                     "levels": claim_levels(c), "n": int(g.sum()),
                     "share": g.sum() / max(len(m0), 1),
                     "signal": float(h[g].sum()), "signal_share": float(h[g].sum()) / h_m0 if h_m0 else 0.0,
                     "n_assigned": int((g & ~io & (tier == T.TIER_ASSIGNED)).sum()),
                     "n_candidate": int((g & ~io & (tier == T.TIER_CANDIDATE)).sum()),
                     "n_ion_only": int((g & io).sum()),
                     "level_hist": hist})

    def _row(section, c, r):
        return {"section": section, "claim": c, "evidence_level": r.get("evidence_level"),
                "tier": r.get("tier"), "mz": r.get("mz"), "height": r.get("height"),
                "neutral_formula": r.get("neutral_formula"), "adduct": r.get("adduct"),
                "evidence": r.get("evidence"), "would_lift": r.get("would_lift"),
                "tier_reason": r.get("tier_reason"), "peak_id": r.get("peak_id")}
    # the two verdicts part: a tier that prints a row the claim calls tentative, or
    # offers one the claim calls identified -- listed, never reconciled
    part = ~io & (((tier == T.TIER_ASSIGNED) & (cl == "tentative"))
                  | ((tier == T.TIER_CANDIDATE) & (cl == "identified")))
    for i in h[part].sort_values(ascending=False, kind="mergesort").index:
        rows.append(_row("tier disagrees", cl[i], m0.loc[i]))
    for c in EV.CLAIM_KEYS:
        for i in h[cl == c].sort_values(ascending=False, kind="mergesort").head(n_bright).index:
            rows.append(_row(f"brightest {c}", c, m0.loc[i]))
    cols = ["section", "claim", "evidence_level", "tier", "meaning", "levels", "n", "share",
            "signal", "signal_share", "n_assigned", "n_candidate", "n_ion_only", "level_hist",
            "mz", "height", "neutral_formula", "adduct", "evidence", "would_lift",
            "tier_reason", "peak_id"]
    return pd.DataFrame(rows, columns=cols)


def _kind_hist(tags: pd.Series, top: int = 8) -> str:
    """'kind: n; ...' over the rows' tag kinds (a row counts once per kind),
    commonest first; '(none)' counts the rows with no tag."""
    counts: dict = {}
    for t in tags:
        for k in (tag_kinds(t) or ["(none)"]):
            counts[k] = counts.get(k, 0) + 1
    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return "; ".join(f"{k}: {v}" for k, v in ordered[:top])


def evidence_level_sheet(m0: pd.DataFrame, n_bright: int = 20, *, no_level: str = "no level") -> pd.DataFrame:
    """The **By evidence level** sheet (docs/EVIDENCE_LEVELS.md): one `summary` row
    per level and bucket (evidence.LEVELS + BUCKETS, then `no level` when a
    committed row carries none) -- count, share of the M0 rows, tier split, the
    commonest tag kinds -- then the `n_bright` brightest M0 rows of each with
    the evidence, what would lift it, the competitors left and the tags, so a
    reader sees what a 4a looks like beside a 5b."""
    lv = m0["evidence_level"].astype(object)
    keys = [*EV.LEVELS, *EV.BUCKETS]
    groups = [(k, lv == k) for k in keys]
    if lv.isna().any():
        groups.append((no_level, lv.isna()))
    meaning = {**EV.LEVEL_MEANING,
               "no level": "a committed row the stage did not level (its evidence says why) -- reads tentative",
               OLD_SCALE_NO_LEVEL: f"a level of a scale older than {SCALE_NAME} -- not a level of this one; "
                                   "reads tentative"}
    _col = lambda g, c: g[c] if c in g.columns else pd.Series(pd.NA, index=g.index)   # noqa: E731
    rows = []
    for level, mask in groups:
        g = m0[mask]
        if not len(g):
            continue
        rows.append({"section": "summary", "level": level, "meaning": meaning[level],
                     "n": int(len(g)), "share": len(g) / max(len(m0), 1),
                     "n_assigned": int((g["tier"] == T.TIER_ASSIGNED).sum()),
                     "n_candidate": int((g["tier"] == T.TIER_CANDIDATE).sum()),
                     "tag_kinds": _kind_hist(_col(g, "tags"))})
    for level, mask in groups:
        g = m0[mask].sort_values("height", ascending=False).head(n_bright)
        for _, r in g.iterrows():
            rows.append({"section": f"brightest {level}", "level": level,
                         "mz": r.get("mz"), "height": r.get("height"),
                         "neutral_formula": r.get("neutral_formula"), "adduct": r.get("adduct"),
                         "tier": r.get("tier"), "evidence": r.get("evidence"),
                         "would_lift": r.get("would_lift"), "competitors_left": r.get("competitors_left"),
                         "tags": r.get("tags"), "peak_id": r.get("peak_id")})
    cols = ["section", "level", "meaning", "n", "share", "n_assigned", "n_candidate", "tag_kinds",
            "mz", "height", "neutral_formula", "adduct", "tier", "evidence", "would_lift",
            "competitors_left", "tags", "peak_id"]
    return pd.DataFrame(rows, columns=cols)


def summary_stats(ledger: pd.DataFrame, *, context: str = "",
                  sample_id: str = "", scale_info: dict | None = None,
                  tof_flag_mz=None) -> pd.DataFrame:
    """The **Summary** sheet. `scale_info` is `scale_view`'s info when the caller
    has already read the level columns (build_sheets); else they are read here.
    A ledger carrying the TOF mass-only flag gets a section of its counts, split
    at `tof_flag_mz` (None = the package default)."""
    if "evidence_level" in ledger.columns and scale_info is None:
        ledger, scale_info = scale_view(ledger)
    st = L.stats(ledger)
    n = st["n_peaks"]
    rows = []

    def add(section, metric, value):
        rows.append({"section": section, "metric": metric, "value": value})

    if sample_id:
        add("Run", "sample_id", sample_id)
    if context:
        add("Run", "context", context)
    # NB: no "generated (UTC)" cell here on purpose — this workbook is MATERIAL DATA
    # and must be byte-identical for identical inputs regardless of when it's run.
    # The run timestamp lives only on the PDF report cover + the run-folder name.
    add("Run", "peaks total", n)

    role_label = {L.ROLE_M0: "M0 (has formula)", L.ROLE_ISO: "isotopologue children",
                  L.ROLE_REAGENT: "reagent ions",
                  L.ROLE_UNEXPLAINED: "unexplained"}
    for role, label in role_label.items():
        cnt = st["by_role"].get(role, 0)
        add("Coverage", label,
            f"{cnt}  ({100 * st['count_frac_by_role'].get(role, 0):.1f}% of peaks, "
            f"{100 * st['signal_by_role'].get(role, 0):.1f}% of signal)")
    expl = (st["signal_by_role"].get(L.ROLE_M0, 0)
            + st["signal_by_role"].get(L.ROLE_ISO, 0)
            + st["signal_by_role"].get(L.ROLE_REAGENT, 0))
    add("Coverage", "signal explained", f"{100 * expl:.1f}%")

    m0 = ledger[ledger["role"] == L.ROLE_M0]
    if "evidence_level" in ledger.columns and len(m0):
        # the claim each level supports, on the Tiers rows' own denominators
        cl = _claims(m0)
        h_m0 = m0["height"].sum(skipna=True)
        for claim in EV.CLAIM_KEYS:
            sub = m0[cl == claim]
            sig = (100 * sub["height"].sum(skipna=True) / h_m0) if h_m0 else 0.0
            add("Claims", claim,
                f"{len(sub)}  ({100 * len(sub) / len(m0):.0f}% of assignments, "
                f"{sig:.0f}% of assigned signal) -- {EV.CLAIM_MEANING[claim]}")
    if "tier" in ledger.columns and len(m0):
        h_m0 = m0["height"].sum(skipna=True)
        for tier in (T.TIER_ASSIGNED, T.TIER_CANDIDATE):
            sub = m0[m0["tier"] == tier]
            sig = (100 * sub["height"].sum(skipna=True) / h_m0) if h_m0 else 0.0
            add("Tiers", tier,
                f"{len(sub)}  ({100 * len(sub) / len(m0):.0f}% of assignments, "
                f"{sig:.0f}% of assigned signal)")
        add("Tiers", "Below assignability",
            f"{st['by_role'].get(L.ROLE_UNEXPLAINED, 0)} unexplained peaks "
            "(see Unassigned sheet for per-peak evidence)")
    if "evidence_level" in ledger.columns and len(m0):
        lv = m0["evidence_level"]
        add("Evidence levels", "scale", SCALE_NAME)
        info = scale_info or {}
        if info.get("n_na"):
            add("Evidence levels", "not assessed", info["na_reason"])
        _why = (EV.levels_not_assessed_reason(m0["evidence"], lv) if "evidence" in m0.columns else None)
        if _why:          # no file calibrated: the scale assessed nothing (said once, not only per row)
            add("Evidence levels", "not assessed", _why)
        note = scale_note(info)
        if note:
            add("Evidence levels", "older scale", note)
        for level, cnt in EV.summarize(lv).items():
            add("Evidence levels", level,
                f"{cnt}  ({100 * cnt / len(m0):.0f}% of assignments) -- "
                f"{EV.LEVEL_MEANING[level]}")
        n_none = int(lv.isna().sum())
        if n_none and info.get("old_scale"):
            add("Evidence levels", OLD_SCALE_NO_LEVEL,
                f"{n_none}  ({100 * n_none / len(m0):.0f}% of assignments) -- levelled on an older scale; "
                "reads tentative")
        elif n_none:
            add("Evidence levels", "no level",
                f"{n_none}  ({100 * n_none / len(m0):.0f}% of assignments) -- not levelled "
                "(the row's evidence says why); reads tentative")

    mo = MO.counts(ledger, MO.DEFAULT_TOF_FLAG_MZ if tof_flag_mz is None else tof_flag_mz)
    if mo is not None:
        thr = mo["threshold_mz"]
        sec = "TOF mass-only flag"
        add(sec, "Assigned, flagged", f"{mo['n_flagged']} of {mo['n_assigned']} -- no attached isotope line of "
                                      "the neutral's own elements at its expected height; tier unchanged")
        add(sec, f"below m/z {thr:g}", f"{mo['below']['flagged']} of {mo['below']['assigned']} Assigned -- "
                                      "the reading rests on mass alone")
        add(sec, f"at or above m/z {thr:g}",
            f"{mo['at_or_above']['flagged']} of {mo['at_or_above']['assigned']} Assigned -- a TOF's formula "
            "space is saturated here (decoy-measured on two TOFs); an unflagged reading here is not "
            "supported either")
    if len(m0):
        base = m0["confidence"].map(T.base_confidence)
        for lab in ("High", "Good", "Low", "Suspect"):
            cnt = int((base == lab).sum())
            if cnt:
                add("Confidence", lab, cnt)
        for meth, cnt in m0["method"].value_counts().items():
            add("Methods", str(meth), int(cnt))
    return pd.DataFrame(rows, columns=["section", "metric", "value"])


#: the Read me rows of the scale: every level and bucket with its meaning
_LEVEL_LEGEND = [(lv, EV.LEVEL_MEANING[lv]) for lv in [*EV.LEVEL_ORDER, *EV.BUCKETS]]

#: the Read me rows of the level's columns
_LEVEL_COLUMN_LEGEND = [
    ("evidence_level", "What the evidence behind a committed formula is worth on " + SCALE_NAME
     + ": 3c (best assigned) .. 5b, plus the reagent bucket and NA (not assessed on this "
     "instrument class). Levels 1 (authentic standard) and 2 (library / MS2 spectrum) are "
     "defined and never assigned. Computed per (neutral, adduct) on M0 rows only; the tier "
     "is not an input. See docs/EVIDENCE_LEVELS.md."),
    ("evidence", "The level's reasons in one line, segments joined by ' · ': the level, how the ion "
     "is established, whether the neutral / adduct split is pinned and how, the positive fact, "
     "the named list entry, how the context lists were activated and the tags."),
    ("would_lift", "What the next level up needs (or, on 5a / 5b, what refuted or blocked the reading)."),
    ("competitors_left", "The competitor ions left in the calibrated window after the isotope "
     "tests ('; '-joined; empty when none)."),
    ("tags", "Facts that never unlock a level, joined by ' | ': routes, other-source partners, "
     "ladders, class-list entries, side channels locked, NH4 gate outcomes, a tentative lead. "
     "Routes, ladders and partners can anchor the series exclusion that lifts a pair out of 5a."),
    ("context", "The context-list entries that match the neutral (reflist:<id> = <name> for a "
     "named entry, reflist:<id> for a class entry; registry:<family> likewise)."),
    ("context_source", "How those lists were activated (always active, a keyword in the batch / "
     "dataset / reagent label name, or the pass-0 registry)."),
    ("NA (the token)", "Written literally. Spreadsheet and pandas readers may show it as empty: the claim "
     "column ('not assessed') tells it apart from a row with no level."),
]


def legend_sheet(*, claims: bool = False, mass_only: bool = False) -> pd.DataFrame:
    """The **Read me** sheet. With `claims` (a ledger that carries evidence levels)
    it opens on the claim classes; with `mass_only` (a ledger carrying the TOF
    mass-only flag) it explains the flag's two columns."""
    rows = [
        ("Tiers", "Assigned", "Formula unique in the calibrated mass window, "
         "or corroborated by independent evidence: Mascope-confirmed "
         "isotopologues, the same neutral in a second ionization channel, or "
         "series-anchor support. The 'tier_reason' column states which."),
        ("Tiers", "Candidate", "A plausible formula that the evidence cannot "
         "single out. All competing formulas are listed, one row per candidate "
         "(rank 1 = the committed best guess). The 'why_candidate' column "
         "gives the demotion reason."),
        ("Tiers", "Below assignability", "Unexplained peaks (Unassigned sheet)."
         " 'residual_evidence' says what the isotope pattern DOES tell us: iso-partner "
         "(satellite of another residual peak), has-constraints (carbon/"
         "halogen count measured), isolated (no isotope structure)."),
        *[("Evidence levels", _col, _text) for _col, _text in _LEVEL_COLUMN_LEGEND[:1]],
        *[("Evidence levels", _lvl, _meaning) for _lvl, _meaning in _LEVEL_LEGEND],
        *[("Evidence levels", _col, _text) for _col, _text in _LEVEL_COLUMN_LEGEND[1:]],
        ("Confidence", "High", "Score >= tau_high, |ppm| within 1.5x the gate, "
         "at least one Mascope-confirmed isotopologue, no near-tie."),
        ("Confidence", "Good", "Score >= tau_good and |ppm| within 2x the gate. "
         "Pattern-driven passes (series, iso-pair) are capped here by design."),
        ("Confidence", "Low / Suspect", "Mass fit only, or pattern evidence "
         "with a weaker score. Always tier Candidate."),
        ("Columns", "height", "Peak intensity (cps) from the summed spectrum."),
        ("Columns", "ion_score / compound_score", "Mascope match_compounds "
         "scores for the matched ion / whole compound (isotope pattern "
         "included). Server scoring is authoritative."),
        ("Columns", "eff_score", "Arbitration score: raw score minus the "
         "complexity prior (heteroatom skepticism, waived when the diagnostic "
         "isotope is confirmed) and minor-channel penalty."),
        ("Columns", "ppm_error", "Observed minus theoretical m/z, ppm. The "
         "pipeline self-calibrates (mu, sigma) on the pass-1 CHO/CHON "
         "backbone and gates later commits by z-score."),
        ("Columns", "candidate_density", "How many formulas live within 0.10 "
         "effective score of the winner (winner included). 1 = unique. "
         "'>=N' means the stored alternatives list saturated."),
        ("Columns", "dbe", "Double-bond equivalents of the NEUTRAL (can be "
         "half-integer for radicals, e.g. HO2)."),
        ("Columns", "isotopologues", "Mascope-scored isotope satellites "
         "attributed to this assignment (label=score)."),
        ("Methods", "known:*", "Pass 0: locked list of known instrument "
         "contaminants (silanediol/PDMS ladder) and small atmospheric "
         "acids/radicals; mass + own-81Br-twin self-consistency gated."),
        ("Methods", "known:solvent_cluster / cluster:solvent", "Pass 0 "
         "(positive mode): a SOURCE-SOLVENT cluster ion -- solvent vapour the "
         "ion source clusters with itself ([S_n+H]+, [S_n-H]+, and their -H2O "
         "condensation rung). The neutral reported is the SOLVENT, on a cluster "
         "adduct; the ion is not a covalent molecule and the grid cannot reach "
         "it. 'known:' = no covalent reading of that composition exists (its "
         "neutral would have DBE < 0); 'cluster:' = one does, so the row is "
         "capped at Candidate with the covalent reading in `alternatives`. "
         "ion_score is 0 on these rows: the family is scored OFFLINE (exact "
         "cluster masses + an observed ladder), so there is no server score."),
        ("Methods", "cheminfo+grid", "Pass 1: CHO/CHON backbone candidates "
         "from the cheminfo m/z query + local formula grid, scored by "
         "Mascope; self-calibration is fitted on these."),
        ("Methods", "gka-series", "Pass 2: generalized-Kendrick homologous "
         "series extension of assigned anchors."),
        ("Methods", "contaminant:* / cluster:HBr", "Pass 3: evidence-opened "
         "contaminant families (repeat units validated against decoys) and "
         "HBr cluster reading of reagent-halogen compositions."),
        ("Methods", "residual:iso-pair", "Pass 4: Br/Cl/BrCl isotope doublets "
         "in the residual anchor a constrained enumeration (carbon-clamped "
         "by the 13C satellite where measurable)."),
        ("Methods", "residual:series", "Pass 4: bright residual peaks 1-2 "
         "exact repeat units from >= 1 assigned anchors."),
        ("Methods", "completion:known-neutral", "Pass 5: cross-channel "
         "partners + series-gap fills of already-assigned neutrals "
         "(no new formula space)."),
        ("Provenance", "ledger.csv", "The per-peak ledger is the source of "
         "truth; every sheet here is a mechanical view of it. Commentary "
         "strings are generated from ledger columns and are reproducible."),
    ]
    if mass_only:
        rows += [
            ("TOF mass-only flag", "mass_only", "TRUE on an Assigned row of a TOF-class run when "
             + MO.DEFINITION + ". Information, not a verdict: the tier and the evidence level are "
             "unchanged. Empty on Candidate rows and off a TOF."),
            ("TOF mass-only flag", "mass_only_reason", "Why: below the threshold (default m/z 350) the "
             "reading rests on mass alone; at or above it a TOF's formula space is saturated -- a "
             "shifted-mass decoy spectrum is Assigned as often as the real one (measured on two TOFs). "
             "An isotope line, an MS2 spectrum or a standard would support the reading. FALSE is not "
             "support at or above the threshold: there even a present line is weak (a shifted spectrum "
             "keeps real isotope spacings, and a 0.5-2x 13C band bounds the carbon count only to a "
             "factor of two), and on a TOF's shifted-mass decoy the line test left more decoy readings "
             "unflagged there than real ones."),
        ]
    if claims:
        rows = [
            ("Claims", "claim", "What a committed formula lets you say, read from its "
             "evidence_level: identified (level 3c; 1 and 2 are never assigned), neutral "
             "(4a), ion (4b), tentative (5a, 5b or no level). Two buckets sit beside the "
             "claims, never folded into them: reagent (a reagent ion or cluster) and not "
             "assessed (NA). The tier is a separate verdict from the same columns and can "
             "disagree; the By claim sheet lists where."),
            *[("Claims", _c, EV.CLAIM_MEANING[_c]) for _c in EV.CLAIM_KEYS],
            ("Claims", "n_identified (By class)", "The number of tier-Assigned rows: a "
             "column name older than the claim. It does not count the identified claim."),
        ] + rows
    return pd.DataFrame(rows, columns=["section", "topic", "explanation"])


# ---------------------------------------------------------------------------
# Excel styling
# ---------------------------------------------------------------------------
_NUM_FMT = {
    "mz": "0.0000", "parent_mz": "0.0000",
    "height": "#,##0", "signal": "#,##0",
    "ppm_error": "+0.00;-0.00;0.00",
    "ion_score": "0.000", "compound_score": "0.000", "eff_score": "0.000",
    "score": "0.000", "best_score": "0.000", "iso_match_score": "0.000",
    "dbe": "0.0",
    "share": "0.0%", "signal_share": "0.0%",
}
_WRAP_COLS = {"commentary": 70, "why_candidate": 46,
              "tier_reason": 46, "alternatives_text": 44, "composite_note": 50,
              "degeneracy_note": 60,
              "isotopologues_text": 30, "isotopologues": 30,
              # the level's text columns (evidence.COLUMNS): long, one line each
              "evidence": 80, "would_lift": 50, "competitors_left": 44, "tags": 60,
              "context": 40, "context_source": 44, "tag_kinds": 52,
              "meaning": 60, "level_hist": 36,
              "interpretation": 52, "explanation": 90, "value": 46,
              # the TOF mass-only flag's reason (assignment/mass_only.py)
              "mass_only_reason": 50}

_FILL = {
    "good":    ("C6EFCE", "006100"),
    "okay":    ("E2EFDA", "375623"),
    "warn":    ("FFEB9C", "9C6500"),
    "bad":     ("FFC7CE", "9C0006"),
    "info":    ("DDEBF7", "1F4E79"),
    "neutral": ("EDEDED", "3B3838"),
}


# evidence levels (docs/EVIDENCE_LEVELS.md): named (3c) green, neutral established
# (4a) light green, ion only (4b) blue, a competitor left (5a) amber, rejected (5b)
# red; the reagent and not-assessed buckets grey. Read only on a level column, so
# "NA" or "4a" elsewhere is never coloured
_LEVEL_CHIP = {"1": "good", "2": "good", "3c": "good", "4a": "okay", "4b": "info",
               "5a": "warn", "5b": "bad", "reagent": "neutral", "NA": "neutral"}
_LEVEL_COLS = ("evidence_level", "level")


# claims (evidence.CLAIM_MEANING): read only on a claim column, so a word that
# happens to match elsewhere is never coloured
_CLAIM_CHIP = {"identified": "good", "neutral": "okay", "ion": "info", "tentative": "warn",
               EV.CLAIM_REAGENT: "neutral", EV.CLAIM_NA: "neutral"}
_CLAIM_COLS = ("claim", "best_claim")


def _chip(label: str) -> str | None:
    s = str(label)
    if s == T.TIER_ASSIGNED or s.startswith("High"):
        return "good"
    if s == T.TIER_CANDIDATE or s.startswith("Low"):
        return "warn"
    if s.startswith("Good"):
        return "okay"
    if s.startswith("Suspect"):
        return "bad"
    if s == "iso-partner":
        return "info"
    if s == "has-constraints":
        return "warn"
    if s == "isolated":
        return "neutral"
    return None


def _style_sheet(ws, df, *, chip_cols=(), band_by=None):
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    header_fill = PatternFill("solid", fgColor="1F3864")
    header_font = Font(bold=True, color="FFFFFF")
    for j in range(1, len(df.columns) + 1):
        c = ws.cell(row=1, column=j)
        c.fill = header_fill
        c.font = header_font
        c.alignment = Alignment(vertical="center")
    ws.freeze_panes = "A2"
    if len(df):
        ws.auto_filter.ref = ws.dimensions

    # column widths + number formats + wrapping
    for j, col in enumerate(df.columns, start=1):
        letter = get_column_letter(j)
        if col in _WRAP_COLS:
            width = _WRAP_COLS[col]
        else:
            lens = df[col].head(200).map(lambda v: len(str(v)))
            width = max(int(lens.max()) if len(lens) else 8, len(str(col)) + 1)
            width = min(width + 2, 26)
        ws.column_dimensions[letter].width = width
        fmt = _NUM_FMT.get(col)
        wrap = col in _WRAP_COLS
        if fmt or wrap:
            align = Alignment(wrap_text=True, vertical="top") if wrap else None
            for i in range(2, len(df) + 2):
                cell = ws.cell(row=i, column=j)
                if fmt:
                    cell.number_format = fmt
                if align:
                    cell.alignment = align

    # colour chips on verdict-like columns
    for col in chip_cols:
        if col not in df.columns:
            continue
        j = list(df.columns).index(col) + 1
        for i, val in enumerate(df[col].tolist(), start=2):
            if not pd.notna(val):
                kind = None
            elif col in _CLAIM_COLS:
                kind = _CLAIM_CHIP.get(str(val))
            elif col in _LEVEL_COLS:
                kind = _LEVEL_CHIP.get(str(val))
            elif col == MO.COLUMN:
                # the flag is information, not a verdict: blue, never red
                kind = "info" if EV.truthy(val) else None
            else:
                kind = _chip(val)
            if kind:
                bg, fg = _FILL[kind]
                cell = ws.cell(row=i, column=j)
                cell.fill = PatternFill("solid", fgColor=bg)
                cell.font = Font(color=fg)

    # alternating banding by a grouping key (Candidates: one band per peak)
    if band_by is not None and band_by in df.columns and len(df):
        band_fill = PatternFill("solid", fgColor="F2F2F2")
        bold = Font(bold=True)
        prev, band = None, False
        rank_j = (list(df.columns).index("rank") + 1) if "rank" in df.columns else None
        formula_j = (list(df.columns).index("formula") + 1) if "formula" in df.columns else None
        for i, key in enumerate(df[band_by].tolist(), start=2):
            if key != prev:
                band = not band
                prev = key
            if band:
                for j in range(1, len(df.columns) + 1):
                    if ws.cell(row=i, column=j).fill.start_color.rgb in (None, "00000000"):
                        ws.cell(row=i, column=j).fill = band_fill
            if rank_j and formula_j and ws.cell(row=i, column=rank_j).value == 1:
                ws.cell(row=i, column=formula_j).font = bold


def _style_summary(ws, df):
    """Section-grouped key/value look for Summary and Read me."""
    from openpyxl.styles import Border, Font, Side
    top = Border(top=Side(style="thin", color="B0B0B0"))
    bold = Font(bold=True)
    prev = None
    sec_j = 1
    for i, val in enumerate(df.iloc[:, 0].tolist(), start=2):
        if val != prev:
            for j in range(1, len(df.columns) + 1):
                ws.cell(row=i, column=j).border = top
            ws.cell(row=i, column=sec_j).font = bold
            prev = val
        else:
            ws.cell(row=i, column=sec_j).value = None


def write_excel(ledger: pd.DataFrame, path: str | Path,
                context: str = "ambient-air", sample_id: str = "", *, tof_flag_mz=None):
    sheets = build_sheets(ledger, context, sample_id, tof_flag_mz=tof_flag_mz)
    chip_cols = ("tier", "confidence", "residual_evidence", "best_tier", "evidence_level", "level",
                 "claim", "best_claim", MO.COLUMN)
    with pd.ExcelWriter(path, engine="openpyxl") as xl:
        for name, df in sheets.items():
            out = df if len(df) else pd.DataFrame({"(empty)": []})
            out.to_excel(xl, sheet_name=name[:31], index=False)
            ws = xl.sheets[name[:31]]
            if not len(df):
                continue
            _style_sheet(ws, out,
                         chip_cols=[c for c in chip_cols if c in out.columns],
                         band_by="peak_id" if name == "Candidates" else None)
            if name in ("Summary", "Read me"):
                _style_summary(ws, out)
    # content-stable bytes (fixed SOURCE_DATE_EPOCH) so the assignment workbook is a
    # pure function of the ledger, matching the cluster workbook.
    from peaky.batch.cluster import _make_xlsx_deterministic
    _make_xlsx_deterministic(path)
    return path


def write_markdown(result: dict, path: str | Path) -> Path:
    led = result["ledger"]
    st = result["stats"]
    m0 = led[led["role"] == L.ROLE_M0]
    tier_line = ""
    if "by_tier" in st and st["by_tier"]:
        tier_line = "  |  ".join(f"{k}: {v}" for k, v in st["by_tier"].items())
    info: dict = {}
    if "evidence_level" in led.columns:
        led, info = scale_view(led)
    m0 = led[led["role"] == L.ROLE_M0]
    cl = _claims(m0) if "evidence_level" in led.columns else None
    lines = [
        f"# Peak assignment — sample {result['sample_id']}",
        "",
        f"- Context: **{result['context']}**",
        f"- Peaks: {st['n_peaks']}  |  M0 (has formula): {st['by_role']['M0']}  "
        f"|  isotopologues: {st['by_role']['iso_child']}  "
        f"|  unexplained: {st['by_role']['unexplained']}",
        (f"- Peaks explained: "
         f"{100*(1 - st['count_frac_by_role']['unexplained']):.1f}% "
         f"({st['by_role']['unexplained']}/{st['n_peaks']} unexplained)  |  "
         if "count_frac_by_role" in st else "- ")
        + f"Signal explained: "
        f"{100*(st['signal_by_role']['M0']+st['signal_by_role']['iso_child']+st['signal_by_role']['reagent']):.1f}%",
    ]
    if cl is not None:
        n_cl = EV.summarize_claims(cl)
        lines.append("- Claims: " + " | ".join(f"{k} {n_cl[k]}" for k in EV.CLAIMS)
                     + "".join(f" | {k} {n_cl[k]}" for k in (EV.CLAIM_REAGENT, EV.CLAIM_NA) if n_cl[k]))
        if info.get("n_na"):
            lines.append(f"- Evidence levels: {info['n_na']} of {len(m0)} not assessed -- {info['na_reason']}")
        if info.get("old_scale"):
            lines.append(f"- Evidence levels: {scale_note(info)}")
    if tier_line:
        lines.append(f"- Tiers: {tier_line}")
    lines += [
        f"- Confidence: {st.get('by_confidence', {})}",
        f"- Prescan: {result['prescan']}",
        "",
        "## Top assignments",
        "",
    ]
    top = m0.sort_values("ion_score", ascending=False).head(20)
    for i, r in top.iterrows():
        tags = [str(r["tier"])] if "tier" in m0.columns and pd.notna(r.get("tier")) else []
        if cl is not None:
            tags.append(str(cl[i]))
        tier = f" [{' · '.join(tags)}]" if tags else ""
        lines.append(f"- **{r['neutral_formula']}** {r['adduct']} "
                     f"(m/z {r['mz']:.4f}, {r['confidence']}{tier}) — {r['commentary']}")
    if result.get("problems"):
        lines += ["", "## ⚠ Ledger validation problems", ""]
        lines += [f"- {p}" for p in result["problems"]]
    Path(path).write_text("\n".join(lines), encoding="utf-8")  # non-ASCII (—, ⁻, ≥, ⚠)
    return Path(path)
