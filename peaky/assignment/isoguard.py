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

Three call sites, one rule (`contradiction`):

* per file, `guard_ledger`: the `isotope_guard` stage of assign.run, right
  after the `evidence` stage that computed the facts (the file levelled alone)
  and before `timeseries` -- so before the batch's vote and M0.b's sideband
  hold, which read the per-file tiers;
* per file again, `guard_ledger` inside `calpool.apply_pool` (step 5): a file
  its own core could not calibrate has no level (no facts) at that stage; the
  batch's pooled calibration re-levels it and guards it on those facts before
  the vote folds it in -- every calibration lent through apply_pool;
* on the merged ledger, `guard_merged`: after the batch's pooled levels
  (assign_batch.run), on the pooled facts.

The outcome (`MODES`): by default ('note') the tier_reason note
``isotope contradiction: <fact>`` (`NOTE_MARK`) and the record, the tier kept;
in the 'demote' mode tier Candidate too. The reading itself, the evidence
level and the claim are untouched. A row that is not Assigned is not touched.
Every call site runs the reliability gate (`reliability`) and the per-row
conditions (`LineView.condition`, `_merged_condition`) first, and returns one
record shape (`_record`; docs/OUTPUTS.md).
"""
from __future__ import annotations

import re

import pandas as pd

__version__ = "0.3.1"  # + binding_facts / merged_condition / noted_in_every_file: the gated verdict for the
#                        episode evidence's hold; 0.3.0 NOTE-ONLY by default (MODES, `PassConfig.isotope_guard`) + the demote mode's per-row
#                        conditions (blend / untested) + one record shape; 0.2.0 the reliability gate

#: a stats record keeps at most this many of the demoted rows (the counts are complete)
ROWS_KEPT = 50
#: the note's lead words (tier_reason; `cleanup._note` joins notes with ' | ')
NOTE_MARK = "isotope contradiction"
#: THE RELIABILITY GATE. A file's (a batch's) isotope facts may touch a tier only when the file proves its own
#: 13C count trustworthy: on its REFERENCE rows -- M0 rows Assigned before the guard, engine confidence High /
#: Good (per file; the merged rows carry none), carbon in the ion, a 13C count the levels tested (they count it
#: only where the line's predicted S/N >= 10) -- at least GATE_MIN_AGREE of the counts agree with the formula
#: within the levels' band, over at least GATE_MIN_REF such rows. Otherwise the guard records
#: `facts_unreliable` with the agreement and demotes nothing there (the levels keep the facts as before).
#: Measured (M1 report §12-13): 1-microscan Orbitrap files whose isotope lines under-respond at low ion counts
#: (their response fit sits at its floor, <= 0.3, at predicted line S/N ~25, and the fit cannot see the lines
#: that are missing altogether) agree on 0.22-0.27 of their reference rows, a 100-microscan replicate of the same
#: air on 0.98-1.00, the field files whose gate opens on >= 0.96; three nitrate sentinels of other chamber
#: studies 0.52-0.79. The gate closes on 27 of the 60 guard-relevant files of a 12-batch field arm (all 15N-nitrate
#: 128-600 files, n_ref 4-9): there the guard is off.
GATE_MIN_AGREE = 0.9
GATE_MIN_REF = 10
#: THE MODE (`PassConfig.isotope_guard`, CLI `--isotope-guard`). 'note' (the DEFAULT): the guard records its
#: facts -- the per-file and merged records, the tier_reason note on the rows it would demote, the facts the
#: unidentified table carries -- and never changes a tier. 'demote': those rows are Candidate. Measured so far,
#: against ion-level truth, 0 of 511 demotions was a wrong formula (M1 report §13): the guard's precision is
#: unproven, so it moves no tier unless asked.
MODES = ("note", "demote")
DEFAULT_MODE = "note"
#: THE PER-ROW CONDITIONS (both modes; on 13C facts only -- a row's 34S / 37Cl facts still bind): read on the
#: file's own picks (`LineView`; every ledger row, any role), with the predicted 13C line = the row's area x nC x
#: R13C x ((m+1)/m)^1.5 and its predicted S/N = the row's server S/N x nC x R13C (the levels' own footing).
#: * blend: a pick BLEND_PPM_MIN..BLEND_PPM_MAX ppm off the 13C position at >= BLEND_MIN_AREA x the predicted
#:   area -- an unresolved neighbour shadows the 13C line (outside the levels' 3-ppm window the count reads ~0);
#: * untested: no pick within BLEND_PPM_MIN ppm and a predicted 13C S/N under UNTESTED_MAX_SNR13 -- a missing
#:   faint line is not a measured count.
#: On the merged ledger the condition is read in the source files (`_merged_condition`).
BLEND_PPM_MIN = 3.0          # = iso_checks.C_TOL_PPM, the levels' 13C window
BLEND_PPM_MAX = 10.0
BLEND_MIN_AREA = 0.5
UNTESTED_MAX_SNR13 = 30.0
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


def resolve_mode(mode) -> str:
    """The guard's mode: 'note' (the default: record the facts, never move a tier) or 'demote'."""
    m = DEFAULT_MODE if mode is None else str(mode).strip().lower()
    if m not in MODES:
        raise ValueError(f"isotope guard mode {mode!r}: expected one of {MODES}")
    return m


def mode_of(cfg) -> str:
    """The mode a run's PassConfig carries (`PassConfig.isotope_guard`; no cfg / no field -> the default)."""
    return resolve_mode(getattr(cfg, "isotope_guard", None))


def _is_c13(fact: str) -> bool:
    return str(fact).startswith("13C reads")


class LineView:
    """A per-file ledger's picks sorted by m/z: the geometry the per-row conditions read (module note). Every
    row of the ledger is a pick (any role); a ledger without `area` cannot show a blend, one without
    `signal_to_noise` cannot call a line untested."""

    def __init__(self, ledger: pd.DataFrame):
        import numpy as np
        n = len(ledger)
        self._np = np
        self.mz = pd.to_numeric(ledger["mz"], errors="coerce").to_numpy(float) if "mz" in ledger.columns \
            else np.full(n, np.nan)
        self.area = pd.to_numeric(ledger["area"], errors="coerce").to_numpy(float) if "area" in ledger.columns \
            else np.full(n, np.nan)
        self.snr = (pd.to_numeric(ledger["signal_to_noise"], errors="coerce").to_numpy(float)
                    if "signal_to_noise" in ledger.columns else np.full(n, np.nan))
        ok = np.isfinite(self.mz)
        self.order = np.flatnonzero(ok)[np.argsort(self.mz[ok], kind="stable")]
        self.sorted_mz = self.mz[self.order]
        self.pos = {ix: k for k, ix in enumerate(ledger.index)}

    def condition(self, index, neutral, adduct) -> str | None:
        """'blend' / 'untested' / None for the 13C count of the row at ``index`` read as (neutral, adduct)."""
        from peaky.batch import iso_checks as IC
        np = self._np
        k = self.pos.get(index)
        if k is None or not np.isfinite(self.mz[k]):
            return None
        nC = int(_ion_counts(neutral, adduct).get("C", 0) or 0)
        if nC < 1:
            return None
        x0 = float(self.mz[k])
        t = x0 + IC.D13C
        lo = np.searchsorted(self.sorted_mz, t * (1 - BLEND_PPM_MAX * 1e-6), side="left")
        hi = np.searchsorted(self.sorted_mz, t * (1 + BLEND_PPM_MAX * 1e-6), side="right")
        picks = [(abs(self.mz[j] - t) / t * 1e6, self.area[j]) for j in self.order[lo:hi] if j != k]
        pred_a = self.area[k] * nC * IC.R13C * ((x0 + 1.0) / x0) ** 1.5
        if np.isfinite(pred_a) and pred_a > 0 and any(
                BLEND_PPM_MIN < d <= BLEND_PPM_MAX and np.isfinite(a) and a >= BLEND_MIN_AREA * pred_a
                for d, a in picks):
            return "blend"
        pred_snr = self.snr[k] * nC * IC.R13C
        if not any(d <= BLEND_PPM_MIN for d, _a in picks) and np.isfinite(pred_snr) \
                and pred_snr < UNTESTED_MAX_SNR13:
            return "untested"
        return None


def _contradicted(frame: pd.DataFrame, rows, facts: dict) -> list[dict]:
    """The rows of ``rows`` whose pair carries a guard fact: [{index, neutral_formula, adduct, mz, facts}]."""
    out = []
    for i in rows:
        k = (str(frame.at[i, "neutral_formula"]), str(frame.at[i, "adduct"]))
        f = facts.get(k)
        if f:
            out.append(dict(index=i, neutral_formula=k[0], adduct=k[1],
                            mz=float(frame.at[i, "mz"]) if "mz" in frame.columns else float("nan"), facts=list(f)))
    return out


def _apply(frame: pd.DataFrame, hits: list[dict], *, demote: bool) -> None:
    """The note on every binding row (``hits``); with ``demote`` its tier goes to Candidate too."""
    from peaky.assignment import cleanup as CU
    for h in hits:
        if demote:
            frame.at[h["index"], "tier"] = "Candidate"
        CU._note(frame, h["index"], note_text(h["facts"]))


def _kinds(hits: list[dict]) -> dict:
    out = {"13C": 0, "absent": 0, "unexplained": 0}
    for h in hits:
        t = h["facts"][0]
        out["13C" if _is_c13(t) else "absent" if " absent in " in t else "unexplained"] += 1
    return out


def _row_rec(h: dict, **extra) -> dict:
    return dict(neutral_formula=h["neutral_formula"], adduct=h["adduct"], mz=h["mz"], fact=h["facts"][0], **extra)


def _tested_c13(text) -> str | None:
    """'agree' / 'contradict' for a pair's `committed_c13` (the levels' carbon_test text), None when the
    count was not tested (empty) or the reading has no carbon ('C for 0')."""
    t = _txt(text)
    if not t.startswith("13C reads") or " C for 0" in t:
        return None
    return "contradict" if "(tol" in t else "agree"


def reliability(rows: pd.DataFrame, pairs: pd.DataFrame | None, *, confidence: bool = True) -> dict:
    """The gate's measurement on the reference rows (module note) of ``rows`` (a per-file ledger, or the merged
    ledger with ``confidence=False``) against the levels' pair frame ``pairs``: {n_ref, n_agree, agreement,
    reliable}."""
    out = {"n_ref": 0, "n_agree": 0, "agreement": None, "reliable": False}
    if rows is None or not len(rows) or pairs is None or not len(pairs) or "committed_c13" not in pairs.columns:
        return out
    c13 = dict(zip(zip(pairs["neutral_formula"].astype(str), pairs["adduct"].astype(str)), pairs["committed_c13"]))
    m = rows["tier"].astype(str) == "Assigned"
    if "role" in rows.columns:
        m &= rows["role"].astype(str) == "M0"
    if confidence and "confidence" in rows.columns:
        m &= rows["confidence"].astype(str).str.startswith(("High", "Good"))
    n = a = 0
    for nf, ad in zip(rows.loc[m, "neutral_formula"].astype(str), rows.loc[m, "adduct"].astype(str)):
        v = _tested_c13(c13.get((nf, ad)))
        if v is None:
            continue
        n += 1
        a += v == "agree"
    out.update(n_ref=n, n_agree=a, agreement=round(a / n, 4) if n else None,
               reliable=bool(n >= GATE_MIN_REF and a / n >= GATE_MIN_AGREE))
    return out


def _record(mode: str, tested: int, gate: dict) -> dict:
    """The guard's record, one shape at every call site (docs/OUTPUTS.md, `isotope_guard`)."""
    return {"mode": mode, "tested": int(tested), "demoted": 0, "would_demote": 0, "by_fact": _kinds([]),
            "held": {"gate": 0, "blend": 0, "untested": 0}, "rows": [], "held_rows": [],
            "facts_unreliable": not gate["reliable"], "agreement": gate["agreement"], "n_ref": gate["n_ref"],
            "n_agree": gate["n_agree"]}


def _guard(frame: pd.DataFrame, m, facts: dict, gate: dict, cond, mode: str, where: str, log) -> dict:
    """The one rule of every call site: the contradicted Assigned rows of ``m``; none binds when the gate failed;
    a 13C fact the per-row condition ``cond(hit)`` calls a blend / untested does not bind (the row's other facts
    still do); every binding row is noted, and Candidate in the demote mode."""
    rec = _record(mode, int(m.sum()), gate)
    hits = _contradicted(frame, frame.index[m], facts) if facts else []
    if not hits:
        return rec
    if not gate["reliable"]:
        rec["held"]["gate"] = len(hits)
        rec["held_rows"] = [_row_rec(h, condition="gate") for h in hits]
        log(f"[isotope_guard] {where}facts unreliable: the 13C count agrees on {gate['n_agree']} of "
            f"{gate['n_ref']} reference rows (needs >= {GATE_MIN_AGREE:.0%} of >= {GATE_MIN_REF}): "
            f"{len(hits)} contradicted row(s) held, none noted")
        return rec
    bind = []
    for h in hits:
        c = cond(h) if (cond is not None and any(_is_c13(f) for f in h["facts"])) else None
        if c is not None:
            rest = [f for f in h["facts"] if not _is_c13(f)]
            if not rest:
                rec["held"][c] += 1
                rec["held_rows"].append(_row_rec(h, condition=c))
                continue
            h = {**h, "facts": rest}
        bind.append(h)
    demote = mode == "demote"
    _apply(frame, bind, demote=demote)
    rec.update(would_demote=len(bind), demoted=len(bind) if demote else 0, by_fact=_kinds(bind),
               rows=[_row_rec(h) for h in bind])
    if bind or any(rec["held"].values()):
        log(f"[isotope_guard] {where}{len(bind)} Assigned row(s) contradicted by their own isotope lines "
            f"({_kinds(bind)}) -> {'Candidate' if demote else 'noted (mode note: tier kept)'}; held: "
            f"{rec['held']}")
    return rec


def guard_ledger(ledger: pd.DataFrame, pairs: pd.DataFrame | None, *, mode=None, log=print) -> dict:
    """The per-file guard (stage `isotope_guard`, and `calpool.apply_pool` step 5): every Assigned M0 row whose
    pair carries a guard fact in ``pairs`` (the file's own levels, the frame `evidence.apply_levels` computed),
    under the file's reliability gate and the per-row conditions on its own picks (module note), gets the note
    -- and tier Candidate in the 'demote' mode. In place; returns the record (`_record`)."""
    mode = resolve_mode(mode)
    if ledger is None or not len(ledger) or "tier" not in ledger.columns:
        return _record(mode, 0, reliability(None, None))
    facts = pair_facts(pairs)
    m = (ledger["role"].astype(str) == "M0") & (ledger["tier"].astype(str) == "Assigned")
    gate = reliability(ledger, pairs)
    view = []

    def cond(h):
        if not view:
            view.append(LineView(ledger))
        return view[0].condition(h["index"], h["neutral_formula"], h["adduct"])

    return _guard(ledger, m, facts, gate, cond, mode, "", log)


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


def _merged_condition(merged: pd.DataFrame, ledgers: dict | None):
    """The per-row condition of a merged row: its source files (``srcs``) are read at their own M0 row of the
    same ion (assign_batch._ion_key, the vote's key; the nearest one by m/z); the 13C fact does not bind when at
    least half of the files that carry the row call it a blend / untested (the commoner of the two). None when
    no per-file ledger is at hand."""
    if not ledgers or "srcs" not in merged.columns:
        return None
    from peaky.batch.assign_batch import _ion_key, _s
    views, index = {}, {}

    def rows_of(sid):
        if sid not in index:
            led = ledgers.get(sid)
            d = {}
            if led is not None and len(led) and {"role", "neutral_formula", "adduct", "mz"} <= set(led.columns):
                mm = led[led["role"].astype(str) == "M0"]
                for i, n, a, x in zip(mm.index, mm["neutral_formula"], mm["adduct"],
                                      pd.to_numeric(mm["mz"], errors="coerce")):
                    d.setdefault(_ion_key(_s(n), _s(a)), []).append((float(x), i, _s(n), _s(a)))
            index[sid] = d
        return index[sid]

    def cond(h):
        k = _ion_key(h["neutral_formula"], h["adduct"])
        cs = []
        for sid in (x for x in _txt(merged.at[h["index"], "srcs"]).split(",") if x):
            cand = rows_of(sid).get(k)
            if not cand:
                continue
            x, i, n, a = min(cand, key=lambda r: abs(r[0] - h["mz"]))
            if sid not in views:
                views[sid] = LineView(ledgers[sid])
            cs.append(views[sid].condition(i, n, a))
        held = [c for c in cs if c is not None]
        if not cs or 2 * len(held) < len(cs):
            return None
        return max(("blend", "untested"), key=held.count)

    return cond


def merged_condition(merged: pd.DataFrame, ledgers: dict | None):
    """The merged guard's per-row condition (`_merged_condition`), for a reader outside this module."""
    return _merged_condition(merged, ledgers)


def binding_facts(facts, cond=None, hit: dict | None = None) -> list[str]:
    """The guard facts of one reading that BIND -- the gated verdict -- given that the gate is open: a 13C fact
    the per-row condition ``cond(hit)`` calls a blend / untested is dropped (the reading's 34S / 37Cl facts still
    bind), exactly as `_guard` reads them. ``hit``: {index, neutral_formula, adduct, mz} of the row."""
    facts = list(facts or [])
    if not facts or cond is None or hit is None or not any(_is_c13(f) for f in facts):
        return facts
    return facts if cond(hit) is None else [f for f in facts if not _is_c13(f)]


_IN_RE = re.compile(re.escape(NOTE_MARK) + r" in (\d+) of (\d+) file")


def noted_in_every_file(tier_reason) -> bool:
    """True when ``merged_notes``' note on a merged row says the per-file guard bound in EVERY file carrying the
    row (k of n with k == n): only then would the demote mode leave no Assigned file of the reading for the vote."""
    m = _IN_RE.search(_txt(tier_reason))
    return bool(m) and int(m.group(1)) >= int(m.group(2)) > 0


def guard_merged(merged: pd.DataFrame, levels: pd.DataFrame | None, *, ledgers: dict | None = None, mode=None,
                 log=print) -> dict:
    """The merge-side guard: every Assigned merged row whose POOLED pair (`evidence.level_batch`) carries a guard
    fact, under the batch's own gate on its pooled counts (the merged Assigned rows; no confidence there) and the
    per-row conditions read in its source files' ledgers (``ledgers``: {sample_id: per-file ledger};
    `_merged_condition`), gets the note -- and tier Candidate in the 'demote' mode. In place; returns the record."""
    mode = resolve_mode(mode)
    if merged is None or not len(merged) or "tier" not in merged.columns:
        return _record(mode, 0, reliability(None, None))
    if "tier_reason" not in merged.columns:
        merged["tier_reason"] = pd.NA
    facts = pair_facts(levels)
    m = merged["tier"].astype(str) == "Assigned"
    gate = reliability(merged, levels, confidence=False)
    return _guard(merged, m, facts, gate, _merged_condition(merged, ledgers), mode, "merged: ", log)
