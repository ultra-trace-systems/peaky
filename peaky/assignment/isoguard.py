"""The isotope-contradiction guard: a reading whose own line's isotope facts
contradict it is never Assigned (docs/ASSIGNMENT_DETAIL.md §5.9).

The facts are the evidence scale's, computed once by
`levels.lines.q1_isotopes` (via `levels.competitors.q1_pass` / `pass_b`) for
the committed reading of every (neutral, adduct) pair and kept in its pass-B
record `committed_reasons`: test (a) `contradiction_a`, test (b) (another
candidate's in-band line the reading cannot explain) and test (c)
`carbon_test`. The guard re-derives none of them; it reads the texts those
functions write (`facts_of`) and keeps the three kinds the guard binds on:

* **13C count** -- test (c): the carbon count measured on the row's own 13C
  line (`levels.context.carbon_counts`: a file counts where the line's
  predicted S/N >= `iso_checks.C_MIN_SNR13` = 10 and >= C_KDL x the edge)
  differs from the formula's C by more than the levels' band
  max(C_TOL_ABS, C_TOL_REL x nC, C_TOL_SE x se) (`levels.lines.C_TOL_*`);
* **required 34S / 37Cl line absent** -- test (a) on the reading's own 34S or
  37Cl line, outcome `absent`, in the files where the line is detectable
  (`levels.lines.eval_candidate`: 3 x the gate and, with the server S/N, a
  predicted line S/N >= `levels.lines.DET_SNR` = 10; a file under it cannot
  refute, a89d0a3);
* **unexplained 34S / 37Cl line** -- test (b): a line at the 34S / 37Cl spacing,
  in band for a candidate carrying the element, where the reading has no S /
  no Cl and predicts nothing comparable; "not otherwise owned" is test (b)'s
  own neighbour rule (`levels.lines._line_is_neighbour`: another committed
  reading's M0 closer to the line takes it).

Every other contradiction of the scale (18O, 15N, 2x13C, 81Br, the labelled
reagent's 14N line, ...) stays what it was: a level fact (5b), not a tier gate.

Two call sites, one rule (`contradiction`):

* per file, `guard_ledger`: the `isotope_guard` stage of assign.run, right
  after the `evidence` stage that computed the facts (the file levelled alone)
  and before `timeseries` -- so before the batch's vote and M0.b's sideband
  hold, which read the per-file tiers;
* on the merged ledger, `guard_merged`: after the batch's pooled levels
  (assign_batch.run), on the pooled facts.

The outcome is tier Candidate and the tier_reason note
``isotope contradiction: <fact>`` (`NOTE_MARK`); the reading itself, the
evidence level and the claim are untouched. A row that is not Assigned is not
touched (the guard only demotes).
"""
from __future__ import annotations

import re

import pandas as pd

__version__ = "0.1.0"

#: the note's lead words (tier_reason; `cleanup._note` joins notes with ' | ')
NOTE_MARK = "isotope contradiction"
#: the isotope lines whose absence / unexplained presence binds the tier
GUARD_LABELS = {"34S": "S", "37Cl": "Cl"}

# the texts the facts' producers write (levels.lines): one pattern each
#   carbon_test       -> "13C reads 6.0 C for 10 (tol 2.5, per-file 13C, 1 files)"
#   contradiction_a   -> "34S (0.0442x) absent in 2/2 files"
#   q1_isotopes (b)   -> "34S line at 0.045x seen in 2/2 files (in band for C9H14O4S [M-H]-), predicts 0x"
_C_RE = re.compile(r"^13C reads (?P<c>-?[\d.]+) C for (?P<n>\d+) \(tol (?P<tol>[\d.]+), (?P<src>[^)]*)\)$")
_A_RE = re.compile(r"^(?P<label>\S+) \((?P<ratio>[^()]+)x\) (?P<what>absent|too high|too low) "
                   r"in (?P<k>\d+)/(?P<n>\d+) files$")
_B_RE = re.compile(r"^(?P<label>\S+) line at (?P<x>\S+)x seen in (?P<k>\d+)/(?P<n>\d+) files "
                   r"\(in band for (?P<other>.+)\), predicts (?P<pred>\S+)x$")


def _txt(v) -> str:
    if v is None:
        return ""
    try:
        if pd.isna(v):
            return ""
    except (TypeError, ValueError):
        pass
    return str(v)


def facts_of(reasons) -> list[dict]:
    """The committed reading's contradiction facts, one dict per text of the
    pass-B ``committed_reasons`` cell (' | '-joined): kind 'c' | 'a' | 'b' |
    'other' (a text no producer's pattern reads), the text, and the parsed
    fields (label, what, other, ...)."""
    out = []
    for t in (s.strip() for s in _txt(reasons).split(" | ")):
        if not t:
            continue
        m = _C_RE.match(t)
        if m:
            out.append(dict(kind="c", text=t, **m.groupdict()))
            continue
        m = _A_RE.match(t)
        if m:
            out.append(dict(kind="a", text=t, **m.groupdict()))
            continue
        m = _B_RE.match(t)
        if m:
            out.append(dict(kind="b", text=t, **m.groupdict()))
            continue
        out.append(dict(kind="other", text=t))
    return out


def _has(counts: dict, el: str) -> bool:
    return int(counts.get(el, 0) or 0) > 0


def contradiction(reasons, ion_counts: dict | None) -> list[str]:
    """The guard's facts among a pair's ``committed_reasons`` (see the module
    note): the 13C count (test c), an own 34S / 37Cl line absent (test a), a
    34S / 37Cl line the reading cannot explain while it carries no S / Cl
    (test b). ``ion_counts`` = the reading's ion element counts (the element
    test of the third kind); None reads as carrying neither."""
    cnt = ion_counts or {}
    out = []
    for f in facts_of(reasons):
        if f["kind"] == "c":
            out.append(f["text"])
        elif f["kind"] == "a" and f["label"] in GUARD_LABELS and f["what"] == "absent":
            out.append(f["text"])
        elif f["kind"] == "b" and f["label"] in GUARD_LABELS and not _has(cnt, GUARD_LABELS[f["label"]]):
            out.append(f["text"])
    return out


def _ion_counts(n, a) -> dict:
    from peaky.assignment.levels import space as SP
    try:
        return SP.ion_counts_of(str(n), str(a)) or {}
    except Exception:  # noqa: BLE001 -- an unparsable reading carries no element
        return {}


def pair_facts(pairs: pd.DataFrame | None) -> dict:
    """{(neutral, adduct): [guard facts]} over a levels frame (per-file
    `evidence.apply_levels`' pairs or the batch's `level_batch`) -- only the
    pairs with at least one."""
    out = {}
    if pairs is None or not len(pairs) or "committed_reasons" not in pairs.columns:
        return out
    for n, a, r in zip(pairs["neutral_formula"], pairs["adduct"], pairs["committed_reasons"]):
        if not _txt(r):
            continue
        f = contradiction(r, _ion_counts(n, a))
        if f:
            out[(str(n), str(a))] = f
    return out


def note_text(facts: list[str]) -> str:
    return f"{NOTE_MARK}: " + "; ".join(facts)


def _apply(frame: pd.DataFrame, rows, facts: dict) -> list[dict]:
    from peaky.assignment import cleanup as CU
    hits = []
    for i in rows:
        k = (str(frame.at[i, "neutral_formula"]), str(frame.at[i, "adduct"]))
        f = facts.get(k)
        if not f:
            continue
        frame.at[i, "tier"] = "Candidate"
        CU._note(frame, i, note_text(f))
        hits.append(dict(index=i, neutral_formula=k[0], adduct=k[1],
                         mz=float(frame.at[i, "mz"]) if "mz" in frame.columns else float("nan"), fact=f[0]))
    return hits


def _kinds(hits: list[dict]) -> dict:
    out = {"13C": 0, "absent": 0, "unexplained": 0}
    for h in hits:
        t = h["fact"]
        out["13C" if t.startswith("13C reads") else "absent" if " absent in " in t else "unexplained"] += 1
    return out


def guard_ledger(ledger: pd.DataFrame, pairs: pd.DataFrame | None, *, log=print) -> dict:
    """The per-file guard (stage `isotope_guard`): every Assigned M0 row whose
    pair carries a guard fact in ``pairs`` (the file's own levels, the frame
    `evidence.apply_levels` computed) is Candidate with the note. In place;
    returns {tested, demoted, by_fact, rows} (rows: neutral, adduct, mz, fact)."""
    if ledger is None or not len(ledger) or "tier" not in ledger.columns:
        return {"tested": 0, "demoted": 0, "by_fact": _kinds([]), "rows": []}
    facts = pair_facts(pairs)
    m = (ledger["role"].astype(str) == "M0") & (ledger["tier"].astype(str) == "Assigned")
    hits = _apply(ledger, ledger.index[m], facts) if facts else []
    if hits:
        log(f"[isotope_guard] {len(hits)} Assigned row(s) contradicted by their own isotope lines -> Candidate "
            f"({_kinds(hits)})")
    return {"tested": int(m.sum()), "demoted": len(hits), "by_fact": _kinds(hits),
            "rows": [{k: v for k, v in h.items() if k != "index"} for h in hits]}


def merged_notes(merged: pd.DataFrame, ledgers: dict, *, log=print) -> int:
    """Say the per-file guard on the merged row: a merged row whose ION
    (assign_batch._ion_key, the vote's key: a batch-level re-read of the same ion
    counts) the guard's note marks on a per-file M0 row of k of the n files in its
    ``srcs`` gets ``isotope contradiction in k of n file(s): <the first file's
    fact>`` -- the vote reads the guarded tiers but carries no per-file note, so
    a reading every file's guard held would sit at Candidate with no reason. A
    note only; returns the rows noted."""
    from peaky.assignment import cleanup as CU
    from peaky.batch.assign_batch import _ion_key, _s
    if merged is None or not len(merged) or not ledgers or "srcs" not in merged.columns:
        return 0
    if "tier_reason" not in merged.columns:
        merged["tier_reason"] = pd.NA
    idx = {}
    for sid, led in ledgers.items():
        if led is None or not len(led) or "tier_reason" not in led.columns:
            continue
        m = led[(led["role"].astype(str) == "M0")
                & led["tier_reason"].astype(str).str.contains(NOTE_MARK + ":", regex=False)]
        for n, a, tr in zip(m["neutral_formula"], m["adduct"], m["tier_reason"].astype(str)):
            fact = tr[tr.index(NOTE_MARK + ":") + len(NOTE_MARK) + 2:].split(" | ")[0]
            idx.setdefault((str(sid), _ion_key(_s(n), _s(a))), fact)
    if not idx:
        return 0
    noted = 0
    for i in merged.index:
        srcs = [x for x in _txt(merged.at[i, "srcs"]).split(",") if x]
        k = _ion_key(_s(merged.at[i, "neutral_formula"]), _s(merged.at[i, "adduct"]))
        facts = [idx[(s, k)] for s in srcs if (s, k) in idx]
        if facts:
            CU._note(merged, i, f"{NOTE_MARK} in {len(facts)} of {len(srcs)} file(s): {facts[0]}")
            noted += 1
    if noted:
        log(f"[isotope_guard] merged: {noted} row(s) carry the per-file guard's note")
    return noted


def guard_merged(merged: pd.DataFrame, levels: pd.DataFrame | None, *, log=print) -> dict:
    """The merge-side guard: every Assigned merged row whose POOLED pair
    (`evidence.level_batch`) carries a guard fact is Candidate with the note.
    In place; returns {tested, demoted, by_fact, rows}."""
    if merged is None or not len(merged) or "tier" not in merged.columns:
        return {"tested": 0, "demoted": 0, "by_fact": _kinds([]), "rows": []}
    if "tier_reason" not in merged.columns:
        merged["tier_reason"] = pd.NA
    facts = pair_facts(levels)
    m = merged["tier"].astype(str) == "Assigned"
    hits = _apply(merged, merged.index[m], facts) if facts else []
    if hits:
        log(f"[isotope_guard] merged: {len(hits)} Assigned row(s) contradicted by the pooled isotope facts -> "
            f"Candidate ({_kinds(hits)})")
    return {"tested": int(m.sum()), "demoted": len(hits), "by_fact": _kinds(hits),
            "rows": [{k: v for k, v in h.items() if k != "index"} for h in hits]}
