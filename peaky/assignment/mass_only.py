"""The TOF mass-only flag -- an annotation on a TOF-class run's Assigned readings
that no isotope line of their own supports. It is information, never a verdict:
it changes no tier, no level, no reading.

WHY. On a TOF the formula grid saturates with m/z. Measured with a shifted-mass
decoy on two TOFs (a bromide/nitrate TOF at R ~9 700 and a nitrate TOF at
R ~4 100): at m/z >= 350 the decoy spectrum is Assigned as often as the real
one -- the CHNOS mass-defect gap has closed and an accurate mass picks one of
dozens of formulas. Below that the tier still rests on mass for most readings
whose own isotope lines are not seen. Demoting every such reading would hide
what does make it at high masses; flagging it lets the reader see both the
reading and what it rests on. `PassConfig.tof_flag_mz` (default
`DEFAULT_TOF_FLAG_MZ`, the decoy's constant, unmeasured on iodide TOFs) only
chooses which of two reasons a flagged row carries.

POSITIVE ISOTOPE EVIDENCE (what keeps a reading unflagged). In at least one
file that holds the reading at tier Assigned, an isotope child of the
reading's M0 -- an `iso_child` row whose `parent_peak_id` is the M0's peak --
that

  1. sits at its label's exact spacing from the committed parent line, inside
     the file's self-fitted position window (the evidence scale's placement
     test, chem/isotopes.judge_source; a file with too few 13C children to fit
     the window is not position-tested);
  2. stands at 0.5-2x its count-aware expected height relative to that line
     (isotopes.RATIO_BAND): a minor line matched at a consistent ratio; and
  3. measures an element the NEUTRAL supplies most of in the ion
     (evidence.neutral_elements): the 13C line of an organic neutral, a
     37Cl / 81Br / 34S / 29Si / 30Si line of a neutral that carries the
     element -- never the reagent's own twin (the 81Br line of a Br-free
     neutral's [M+Br]- adduct says the ion holds the reagent, not what the
     neutral is).

That is the evidence scale's own per-file fact (the `multiline_elements` of
evidence._measure, read with `per_file=True`), so the flag and the scale read
one line test. The scorer's `isotopologues` list counts only through the
children it attached: a listed line another reading owns is that reading's
peak, not this one's evidence.

EXEMPT. A pass-0 known species -- a `known:` commit at tier Assigned in one of
the reading's files, or the batch's known-species decision on the merged row
(`KNOWN_DECIDED`) -- is never flagged: its identity is the curated list's.

Entry points: `instrument_class`, `flag_merged` (a batch's merged ledger, from
its per-file ledgers), `flag_ledger` (one sample's ledger), `counts` (re-read
the tallies off a flagged frame), `flag_mz` / `check_threshold`.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from peaky.assignment import evidence as EV
from peaky.assignment.passes.config import PassConfig

#: the m/z at and above which a flagged reading says the formula space is
#: saturated: where the shifted-mass decoy's Assigned rate met the real
#: spectrum's on two TOFs. The one home of the number is PassConfig's field
#: default; a run's own value rides on its cfg (`flag_mz`)
DEFAULT_TOF_FLAG_MZ = float(PassConfig.tof_flag_mz)

#: the merged / per-sample ledger columns
COLUMN = "mass_only"
REASON_COLUMN = "mass_only_reason"
COLUMNS = (COLUMN, REASON_COLUMN)

#: the note batch/assign_batch.lock_known_species writes on a merged row whose
#: known species the batch decided (confirmed, kept or locked)
KNOWN_DECIDED = "known species decided once for the batch"

#: one sentence for docs, the batch summary and the report legends
DEFINITION = ("no isotope line of the neutral's own elements (13C of an organic neutral; 37Cl / 81Br / "
              "34S / 29Si / 30Si of a neutral that carries the element; never the reagent's own twin) "
              "sits at its exact spacing and 0.5-2x its expected height in any file that holds the "
              "reading at tier Assigned; pass-0 known species are exempt; tier and level unchanged")

_TOF = "tof"


def check_threshold(value) -> float:
    """`value` as the flag's m/z threshold: a finite number above 0, else ValueError."""
    try:
        x = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"the TOF flag threshold is an m/z above 0, not {value!r}") from None
    if not math.isfinite(x) or x <= 0:
        raise ValueError(f"the TOF flag threshold is an m/z above 0, not {value!r}")
    return x


def flag_mz(cfg=None) -> float:
    """The threshold a run's PassConfig carries (`tof_flag_mz`), the default without one."""
    v = getattr(cfg, "tof_flag_mz", None) if cfg is not None else None
    return check_threshold(DEFAULT_TOF_FLAG_MZ if v is None else v)


def instrument_class(*, resolution=None, instrument_types=()) -> str | None:
    """'tof' / 'orbitrap' / None: the width model's class (evidence.instrument, a
    Resolution, its `as_dict` or a resolving power) when there is one, else the
    files' scoring classes when they all agree ('tof'; 'orbi' reads 'orbitrap'),
    else None (unknown: the flag does not run)."""
    klass = EV.instrument(resolution)[0] if resolution is not None else None
    if klass:
        return klass
    kinds = {str(t).strip().lower() for t in instrument_types or ()
             if t is not None and str(t).strip() and str(t).strip().lower() not in ("nan", "none")}
    if kinds == {_TOF}:
        return _TOF
    if kinds and kinds <= {"orbi", "orbitrap"}:
        return "orbitrap"
    return None


def is_tof(klass) -> bool:
    return str(klass or "").strip().lower() == _TOF


def reason(mz, threshold: float) -> str:
    """The short reason a flagged row carries: at or above `threshold` the
    formula space is saturated there; below it the reading rests on mass alone."""
    try:
        hi = float(mz) >= float(threshold)
    except (TypeError, ValueError):
        hi = False
    if hi:
        return (f"mass only at m/z >= {float(threshold):g}: a TOF's formula space is saturated here "
                "(decoy-measured on two TOFs); no isotope line of the neutral's own elements in any "
                "Assigned file")
    return ("mass only: the reading rests on mass alone -- no isotope line of the neutral's own "
            "elements at its expected height in any Assigned file")


def _ion_key(neutral, adduct) -> tuple:
    """The ION a reading names (its element counts), so a reading the batch
    re-read finds the files that hold the same ion under another label."""
    try:
        comp = EV.ion_composition(neutral, adduct, None) or {}
    except Exception:          # noqa: BLE001 -- an unparsable reading is its own ion
        comp = {}
    if not comp:
        return ("?", str(neutral), str(adduct))
    sign = "+" if "]+" in str(adduct) else "-" if "]-" in str(adduct) else ""
    return tuple(sorted((k, int(v)) for k, v in comp.items() if v)) + (sign,)


_FACT_COLUMNS = ["neutral_formula", "adduct", "assigned", "known", "positive", "own_lines"]


def pair_facts(ledger: pd.DataFrame, *, halogen=None, resolution=None) -> pd.DataFrame:
    """One row per (neutral_formula, adduct) the ledger (ONE file) commits:
    `assigned` (a row of the pair is tier Assigned), `known` (a row is a pass-0
    `known:` commit), `own_lines` (the elements of its positive lines,
    '|'-joined) and `positive` (any). See the module note for the line test."""
    if ledger is None or not len(ledger) or "role" not in ledger.columns:
        return pd.DataFrame(columns=_FACT_COLUMNS)
    frame = EV.trim(ledger)
    if not (frame["role"].astype(str) == "M0").any():
        return pd.DataFrame(columns=_FACT_COLUMNS)
    frame["__file"] = "file"
    # the evidence scale's per-file pair facts (one line test for the flag and the scale)
    facts = EV._measure(frame, halogen=halogen, resolution=resolution, per_file=True)
    if facts.empty:
        return pd.DataFrame(columns=_FACT_COLUMNS)
    own = facts["multiline_elements"].fillna("").astype(str)
    return pd.DataFrame({
        "neutral_formula": facts["neutral_formula"].astype(str),
        "adduct": facts["adduct"].astype(str),
        "assigned": facts["tier"].astype(str).eq("Assigned").to_numpy(),
        "known": facts["known_fam"].fillna("").astype(str).ne("").to_numpy(),
        "positive": own.ne("").to_numpy(),
        "own_lines": own.to_numpy(),
    })


def _blank(frame: pd.DataFrame) -> None:
    frame[COLUMN] = pd.Series(pd.NA, index=frame.index, dtype=object)
    frame[REASON_COLUMN] = pd.Series(pd.NA, index=frame.index, dtype=object)


def _block(klass, threshold: float) -> dict:
    return {"instrument": klass, "applied": is_tof(klass), "threshold_mz": float(threshold),
            "n_assigned": 0, "n_flagged": 0, "n_known": 0, "n_positive": 0,
            "below": {"assigned": 0, "flagged": 0}, "at_or_above": {"assigned": 0, "flagged": 0},
            "definition": DEFINITION}


def _tally(block: dict, mz: np.ndarray, assigned: np.ndarray, flagged: np.ndarray) -> dict:
    hi = np.asarray(mz, dtype=float) >= block["threshold_mz"]
    block["n_assigned"] = int(assigned.sum())
    block["n_flagged"] = int(flagged.sum())
    block["below"] = {"assigned": int((assigned & ~hi).sum()), "flagged": int((flagged & ~hi).sum())}
    block["at_or_above"] = {"assigned": int((assigned & hi).sum()), "flagged": int((flagged & hi).sum())}
    return block


def describe(block: dict) -> str:
    """One log line for a flag block."""
    if not block.get("applied"):
        return (f"[tof-flag] not a TOF-class run ({block.get('instrument') or 'class unknown'}): "
                "no mass-only flag")
    b, a = block["below"], block["at_or_above"]
    thr = block["threshold_mz"]
    return (f"[tof-flag] {block['n_flagged']} of {block['n_assigned']} Assigned readings rest on mass alone "
            f"(no own isotope line in any Assigned file; {block['n_known']} known species exempt): "
            f"{b['flagged']} of {b['assigned']} below m/z {thr:g}, {a['flagged']} of {a['assigned']} at or "
            "above it (formula space saturated there); tier and level unchanged")


def flag_merged(merged: pd.DataFrame, per_file: dict, *, klass, threshold=DEFAULT_TOF_FLAG_MZ,
                halogen=None, resolution=None, log=print) -> dict:
    """Stamp `mass_only` / `mass_only_reason` on a batch's merged ledger, in
    place, from the per-file ledgers it was merged from ({sample id: full
    ledger}). On a TOF-class run (`klass` 'tof') every Assigned row reads True
    (flagged) or False; every other row -- and every row of any other class --
    stays empty. A row's files are its `srcs` that hold its reading at tier
    Assigned (a reading the batch re-read: the files holding its ion). Returns
    the batch_summary block: instrument, applied, threshold_mz, n_assigned,
    n_flagged, n_known (exempt), n_positive, below / at_or_above {assigned,
    flagged}, definition. Touches no other column."""
    thr = check_threshold(threshold)
    block = _block(klass, thr)
    _blank(merged)
    if not block["applied"] or merged is None or not len(merged) or "tier" not in merged.columns:
        log(describe(block))
        return block
    by_reading: dict = {}
    by_ion: dict = {}
    for sid, led in (per_file or {}).items():
        f = pair_facts(led, halogen=halogen, resolution=resolution)
        for nf, ad, asg, kn, pos in zip(f["neutral_formula"], f["adduct"], f["assigned"], f["known"],
                                        f["positive"]):
            if not asg:
                continue
            by_reading[(nf, ad, str(sid))] = (bool(pos), bool(kn))
            by_ion.setdefault((_ion_key(nf, ad), str(sid)), []).append((bool(pos), bool(kn)))
    tier = merged["tier"].astype(str)
    srcs = (merged["srcs"].fillna("").astype(str) if "srcs" in merged.columns
            else pd.Series("", index=merged.index))
    notes = (merged["tier_reason"].fillna("").astype(str) if "tier_reason" in merged.columns
             else pd.Series("", index=merged.index))
    mz = pd.to_numeric(merged["mz"], errors="coerce").to_numpy(dtype=float)
    assigned = tier.eq("Assigned").to_numpy()
    flagged = np.zeros(len(merged), dtype=bool)
    n_known = n_pos = 0
    for k, i in enumerate(merged.index):
        if not assigned[k]:
            continue
        nf, ad = str(merged.at[i, "neutral_formula"]), str(merged.at[i, "adduct"])
        files = [s for s in srcs.at[i].split(",") if s]
        hold = [by_reading[(nf, ad, s)] for s in files if (nf, ad, s) in by_reading]
        if not hold:
            key = _ion_key(nf, ad)
            hold = [x for s in files for x in by_ion.get((key, s), [])]
        known = any(kn for _p, kn in hold) or KNOWN_DECIDED in notes.at[i]
        positive = any(p for p, _k in hold)
        n_known += int(known)
        n_pos += int(positive and not known)
        flagged[k] = not known and not positive
        merged.at[i, COLUMN] = bool(flagged[k])
        if flagged[k]:
            merged.at[i, REASON_COLUMN] = reason(mz[k], thr)
    _tally(block, mz, assigned, flagged)
    block.update(n_known=int(n_known), n_positive=int(n_pos))
    log(describe(block))
    return block


def flag_ledger(ledger: pd.DataFrame, *, klass, threshold=DEFAULT_TOF_FLAG_MZ, halogen=None,
                resolution=None) -> dict:
    """The same flag on ONE sample's ledger, in place: on a TOF-class sample
    every Assigned M0 row reads True / False from this file's own lines (a
    `known:` commit is exempt); every other row, and every row of any other
    class, stays empty. Returns the block `flag_merged` returns."""
    thr = check_threshold(threshold)
    block = _block(klass, thr)
    _blank(ledger)
    if not block["applied"] or ledger is None or not len(ledger) \
            or not {"role", "tier", "neutral_formula", "adduct"} <= set(ledger.columns):
        return block
    f = pair_facts(ledger, halogen=halogen, resolution=resolution)
    pos = {(nf, ad): bool(p) for nf, ad, p in zip(f["neutral_formula"], f["adduct"], f["positive"])}
    role = ledger["role"].astype(str)
    method = ledger["method"].fillna("").astype(str) if "method" in ledger.columns \
        else pd.Series("", index=ledger.index)
    asg = (role.eq("M0") & ledger["tier"].astype(str).eq("Assigned")).to_numpy()
    mz = pd.to_numeric(ledger["mz"], errors="coerce").to_numpy(dtype=float)
    flagged = np.zeros(len(ledger), dtype=bool)
    n_known = n_pos = 0
    for k, i in enumerate(ledger.index):
        if not asg[k]:
            continue
        known = method.at[i].startswith("known:")
        positive = pos.get((str(ledger.at[i, "neutral_formula"]), str(ledger.at[i, "adduct"])), False)
        n_known += int(known)
        n_pos += int(positive and not known)
        flagged[k] = not known and not positive
        ledger.at[i, COLUMN] = bool(flagged[k])
        if flagged[k]:
            ledger.at[i, REASON_COLUMN] = reason(mz[k], thr)
    _tally(block, mz, asg, flagged)
    block.update(n_known=int(n_known), n_positive=int(n_pos))
    return block


def flagged(frame: pd.DataFrame) -> pd.Series:
    """Boolean mask of the flagged rows (False where the column is empty or absent);
    reads a CSV round trip ('True' / 'False' / empty) as written."""
    if frame is None or COLUMN not in getattr(frame, "columns", ()):
        return pd.Series(False, index=getattr(frame, "index", None), dtype=bool)
    return frame[COLUMN].map(EV.truthy).astype(bool)


def counts(frame: pd.DataFrame, threshold=DEFAULT_TOF_FLAG_MZ) -> dict | None:
    """The tallies re-read off a flagged frame (a merged ledger, or one sample's
    ledger: its M0 rows): {threshold_mz, n_assigned, n_flagged, below,
    at_or_above}; None when the frame carries no flag value (no column, or an
    empty one: not a TOF-class run, or a run made before the flag)."""
    if frame is None or COLUMN not in getattr(frame, "columns", ()) or not frame[COLUMN].notna().any():
        return None
    thr = check_threshold(threshold)
    rows = frame[frame["role"].astype(str) == "M0"] if "role" in frame.columns else frame
    assigned = (rows["tier"].astype(str).eq("Assigned").to_numpy() if "tier" in rows.columns
                else rows[COLUMN].notna().to_numpy())
    mz = pd.to_numeric(rows["mz"], errors="coerce").to_numpy(dtype=float)
    fl = flagged(rows).to_numpy() & assigned
    out = _tally({"threshold_mz": thr}, mz, assigned, fl)
    return out
