"""A SOURCE of the evidence scale and the pipeline that levels it.

A source is one or more ledgers levelled together: a batch's pooled files
(``mode="run"``, every file-count minimum 3), one file alone (``"adapted"``:
the minima 1; ``"strict"``: the batch minima, so no isotope evidence on one
file), or one file inside a MAIN run's context (a decoy arm, a post-hoc
re-level): the main run's window, gate, line efficiency, position sigma,
carbon-count se model, 15N twin ratio, decomposition grid, context lists and
scan.

`level_source` runs the scale in order: the instrument-class gate (a source
whose width model does not resolve >= 50 000 at m/z 200, or that has none,
reads NA and nothing else is computed) -> the pair facts -> the run context ->
pass A (competitors and isotope tests) -> pass B -> the decision -> the record.
"""
from __future__ import annotations

import dataclasses
import glob
import json
import os
import re

import numpy as np
import pandas as pd

from peaky.assignment.levels import competitors as CP
from peaky.assignment.levels import context as CX
from peaky.assignment.levels import decide as DC
from peaky.assignment.levels import lists as LS
from peaky.assignment.levels import routes as RT
from peaky.assignment.levels import scale as SC
from peaky.assignment.levels import space as SP
from peaky.assignment.levels import split as SPL
from peaky.chem import chemistry as C
from peaky.chem import profiles as PR

MODES = ("run", "adapted", "strict")
# the pair-fact columns of the pooled facts table read as booleans / texts (NaN-safe)
BOOL_COLS = ["iso", "multiline", "carbon_ev", "chan2", "anchor", "branch", "reagent_only_iso", "ion_only", "tied",
             "below", "lead", "lead_lift", "lowconf", "saturated", "res_ok", "corroborated", "upair", "label_untie",
             "label_veto", "iso_veto", "cross", "neutral_backed"]
STR_COLS = ["known_fam", "multiline_elements", "resolvability", "reagent_halogen", "label_note", "iso_note",
            "lock_note", "iso_labels", "neutral_formula", "adduct", "ion"]
# the B-series LEVEL columns of the facts table: never part of the scale's output
SERIES_LEVEL_COLS = ("evidence_level", "evidence_axes", "level_reason", "n_plausible_structures", "claim", "cur")
ARM_FILE_RE = re.compile(r"^(?P<fid>.+?)__(?P<arm>[A-Za-z0-9_-]+)\.csv(?:\.gz)?$")


def _ev():
    from peaky.assignment import evidence as EV
    return EV


@dataclasses.dataclass
class RunInputs:
    """What a source needs beyond its per-file ledgers: the batch summary
    (reagent, context, label, reflists_active [+ reflists_context], resolution,
    per_file stats [height gates, degeneracy_cal], amine_r_min), the merged
    ledger, the batch time series, the iso_checks / label_twins / neutral_pairs
    tables, the protected neutrals (None: from the per-file ledgers) and the
    run's reference-list activation record (None: from the summary's
    ``reflists_context``)."""
    summary: dict
    merged: pd.DataFrame | None = None
    ts: pd.DataFrame | None = None
    iso_checks: pd.DataFrame | None = None
    label_twins: pd.DataFrame | None = None
    neutral_pairs: pd.DataFrame | None = None
    protected: set | None = None
    activation: dict | None = None


class Source:
    """A source to level (see the module docstring). Build with
    `source_from_frames` / `source_from_run_dir`."""

    def __init__(self, per_file: dict, inputs: RunInputs, *, mode: str = "run", name: str = "", label: str = "",
                 main: "Source | None" = None, fid: str | None = None, arm: str | None = None,
                 wrong_adducts=(), control_ledger: pd.DataFrame | None = None):
        if mode not in MODES:
            raise ValueError(f"mode {mode!r} not in {MODES}")
        if main is not None and len(per_file) != 1:
            raise ValueError("a source in a main run's context holds exactly one file")
        self.per_file = dict(sorted(per_file.items()))
        self.inputs = inputs
        self.mode = mode
        self.name = name
        self.label = label or name
        self.main = main
        self.fid = fid if fid is not None else (next(iter(self.per_file)) if main is not None else None)
        self.arm = arm
        self.wrong_adducts = list(wrong_adducts or ())
        self.control_ledger = control_ledger
        # caches (set by the pipeline; the parity harness may preset facts / win / q1)
        self.facts: pd.DataFrame | None = None
        self.win: dict | None = None
        self.q1: dict | None = None
        self.ctx = None
        self._lists = None
        self._scan = None

    # ---- the minima ----
    @property
    def nmin(self) -> int:
        return CX.NMIN_FILES_ARM if self.mode == "adapted" else CX.NMIN_FILES

    @property
    def one_file_minima(self) -> bool:
        """Route co-files / ladder files 1 and no partners (the adapted minima)."""
        return self.mode == "adapted"

    @property
    def summary(self) -> dict:
        return self.main.summary if self.main is not None else self.inputs.summary

    # ---- the instrument class ----
    def instrument(self) -> tuple[str | None, float]:
        """(class or None, R at m/z 200) of the source's width model (a main run's for an arm)."""
        res = self.summary.get("resolution")
        klass, _fw = _ev().instrument(res)
        if klass is None:
            return None, float("nan")
        from peaky.chem.resolution import Resolution
        return klass, float(Resolution.from_dict(res).r_at(200.0))


# ---------------------------------------------------------------------------
# construction
# ---------------------------------------------------------------------------
def source_from_frames(per_file: dict, *, run_inputs: RunInputs, mode: str = "run", name: str = "",
                       label: str = "", main: Source | None = None, arm: str | None = None, wrong_adducts=(),
                       control_ledger=None) -> Source:
    """A source from in-memory FULL ledgers ({sample_id: ledger}, every row;
    sorted by sample id) and the run's other inputs."""
    return Source(per_file, run_inputs, mode=mode, name=name, label=label, main=main, arm=arm,
                  wrong_adducts=wrong_adducts, control_ledger=control_ledger)


def _read_csv(p, float_precision=None):
    return pd.read_csv(p, low_memory=False, float_precision=float_precision) if os.path.isfile(p) else None


def _activation_from_run(path: str, summary: dict):
    """The run's reference-list activation record: batch_summary's
    ``reflists_context`` when the run recorded it, else recomputed from the run
    manifest's batch / dataset names and the profile label (the texts the run's
    `RL.activate` read); None when neither exists."""
    rc = summary.get("reflists_context")
    if isinstance(rc, dict) and "matched" in rc:
        return rc
    mp = os.path.join(path, "run_manifest.json")
    if not os.path.isfile(mp):
        return None
    try:
        inp = (json.load(open(mp)) or {}).get("input") or {}
    except (OSError, ValueError):
        return None
    return LS.activation_record(batch=inp.get("batch_name") or summary.get("batch_name") or "",
                                dataset=inp.get("dataset") or "", label=summary.get("label") or "")


def source_from_run_dir(path, *, main=None, mode: str | None = None, name: str | None = None,
                        label: str | None = None, float_precision: str | None = None) -> Source:
    """A source from disk. ``path`` = a batch run directory (merged_ledger.csv,
    per_file/*_ledger.csv, per_file/_batch_ts.parquet, batch_summary.json,
    tables/{iso_checks,label_twins,neutral_pairs}.csv) or ONE ledger CSV (.csv /
    .csv.gz). With ``main`` (a run dir or a Source) the file is levelled in the
    main run's context exactly as a decoy arm: a file named
    ``<sample id>__<arm>.csv[.gz]`` takes its sample id and arm from the name
    and, on the 'adducts' arm, the decoy manifest's wrong adducts. ``mode``:
    'run' for a run dir, 'adapted' (default) or 'strict' for one file. Every CSV
    is read with ``low_memory=False`` and pandas' default parser."""
    path = os.path.expanduser(str(path).rstrip("/"))
    if os.path.isdir(path):
        inp = CX.source_inputs_from_run_dir(path, float_precision=float_precision)
        np_tab = _read_csv(os.path.join(path, "tables", "neutral_pairs.csv"), float_precision)
        ri = RunInputs(summary=inp.summary, merged=inp.merged, ts=inp.ts, iso_checks=inp.iso_checks,
                       label_twins=inp.label_twins, neutral_pairs=np_tab,
                       activation=_activation_from_run(path, inp.summary))
        nm = name or os.path.basename(path)
        return Source(inp.per_file, ri, mode=mode or "run", name=nm, label=label or nm)
    led = pd.read_csv(path, low_memory=False, float_precision=float_precision)
    base = os.path.basename(path)
    m = ARM_FILE_RE.match(base)
    fid, arm = (m.group("fid"), m.group("arm")) if m else (re.sub(r"_ledger\.csv(\.gz)?$|\.csv(\.gz)?$", "", base), None)
    if main is not None and not isinstance(main, Source):
        main = source_from_run_dir(main)
    wrong, control = [], None
    if main is not None and arm is not None:
        mf = os.path.join(os.path.dirname(path), "manifest.json")
        man = json.load(open(mf)) if os.path.isfile(mf) else {}
        if arm == "adducts":
            wrong = list(man.get("wrong_adducts", []))
        cp = glob.glob(os.path.join(os.path.dirname(path), f"{fid}__control.csv*"))
        if cp and arm != "control":
            control = pd.read_csv(cp[0], low_memory=False, float_precision=float_precision)
        elif arm == "control":
            control = led
    summary = main.summary if main is not None else {}
    ri = RunInputs(summary=summary)
    nm = name or (f"{main.name}-{arm}-{mode or 'adapted'}" if main is not None else fid)
    return Source({fid: led}, ri, mode=mode or "adapted", name=nm, label=label or nm, main=main, fid=fid, arm=arm,
                  wrong_adducts=wrong, control_ledger=control)


# ---------------------------------------------------------------------------
# the pair facts and the context
# ---------------------------------------------------------------------------
def norm_facts(df: pd.DataFrame) -> pd.DataFrame:
    """The pair facts with NaN-safe booleans and texts, without the B-series level columns."""
    df = df.copy()
    truthy = _ev().truthy
    for c in BOOL_COLS:
        if c not in df.columns:
            df[c] = False
        df[c] = df[c].map(truthy).astype(bool)
    for c in STR_COLS:
        if c not in df.columns:
            df[c] = ""
        df[c] = df[c].map(lambda v: "" if v is None or v is pd.NA or (isinstance(v, float) and np.isnan(v))
                          or str(v).lower() in ("nan", "<na>") else str(v))
    for c in ("degeneracy", "height", "mz", "ppm"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.drop(columns=[c for c in SERIES_LEVEL_COLS if c in df.columns])


def reagent_halogen(summary: dict):
    """The reagent halogen a source's pair facts read (C43,
    `evidence.channel_halogen`): the one the run recorded (`reagent_halogen`:
    a batch's profile channels, a file's declared channels), else the declared
    channels of the run's reagent profile (`reagent`, a run made before the
    record), else `evidence.DETECT_HALOGEN` (count the committed clusters) for a
    source that names no profile."""
    EV = _ev()
    summary = summary or {}
    if "reagent_halogen" in summary:
        return summary["reagent_halogen"]
    name = summary.get("reagent")
    if name:
        try:
            return EV.channel_halogen(PR.resolve(str(name)).adducts)
        except Exception:  # noqa: BLE001 -- an unknown profile name: no declared channels
            pass
    return EV.DETECT_HALOGEN


def pair_facts(src: Source) -> pd.DataFrame:
    """The pair facts the scale's step 0 reads (iso_veto, label_veto, lowconf,
    below, ion_only, lead, tied, the pooled m/z ...): one row per committed
    (neutral, adduct). A batch: the per-file ledgers pooled (``tied`` /
    ``lowconf`` need ALL rows, ``below`` / ``lead`` ANY) with the batch checks'
    vetoes; one file: that file alone."""
    if src.facts is not None:
        return src.facts
    EV = _ev()
    if src.main is not None or len(src.per_file) == 1 and src.mode != "run":
        led = next(iter(src.per_file.values()))
        res = src.summary.get("resolution")
        rp = None
        if (res or {}).get("coef") is not None:
            from peaky.chem.resolution import Resolution
            rp = Resolution.from_dict(res)
        F = EV._level_pairs({"": led}, cross=None, resolution=rp, per_file=True,
                            halogen=reagent_halogen(src.summary))
    else:
        from peaky.batch import iso_checks as IC
        from peaky.batch import label_twins as LT
        from peaky.batch import neutral_pairs as NP
        ri = src.inputs
        F = EV._level_pairs({k: EV.trim(v) for k, v in src.per_file.items()}, cross=None, with_files=True,
                            upair=NP.neutrals(ri.neutral_pairs) if ri.neutral_pairs is not None else None,
                            label=LT.facts(ri.label_twins) if ri.label_twins is not None else None,
                            iso=IC.facts(ri.iso_checks) if ri.iso_checks is not None else None,
                            resolution=src.summary.get("resolution"), halogen=reagent_halogen(src.summary))
    src.facts = norm_facts(F)
    return src.facts


def _source_inputs(src: Source) -> CX.SourceInputs:
    ri = src.inputs
    return CX.SourceInputs(summary=ri.summary, per_file=src.per_file, merged=ri.merged, ts=ri.ts,
                           iso_checks=ri.iso_checks, label_twins=ri.label_twins)


def context_of(src: Source):
    """The source's run context (cached on the source)."""
    if src.ctx is not None:
        return src.ctx
    facts = pair_facts(src)
    if src.main is None:
        win = src.win if src.win is not None else CX.run_windows(src.summary, src.per_file)
        src.win = win
        src.ctx = CX.prepare_context(_source_inputs(src), facts, win=win, nmin=src.nmin, name=src.name)
        return src.ctx
    # one file in a main run's context (a decoy arm, a post-hoc re-level)
    ctx0 = context_of(src.main)
    fid, led = src.fid, src.per_file[src.fid]
    w = (ctx0.window_records or {}).get(fid, {})
    if w.get("fit"):
        win = w["fit"]
    else:
        cal = CX.file_cal(src.control_ledger) if src.control_ledger is not None else None
        if cal is not None:
            win = cal
        else:
            mu = w.get("mu_stamp", np.nan)
            win = (mu if np.isfinite(mu) else ctx0.run_window[0], ctx0.run_window[1])
    gate = ctx0.gates.get(fid)
    fa = CX.FileArr(led, gate if gate else np.nan)
    if not np.isfinite(fa.gate) or fa.gate <= 0:
        fa.gate = fa.edge
    arm_adducts = [a for a in led.loc[led["role"] == "M0", "adduct"].dropna().astype(str).unique()
                   if a in C.ADDUCT_SHIFTS and a not in SP.ION_ONLY]
    chans = list(ctx0.channels) + ([a for a in src.wrong_adducts + arm_adducts if a not in ctx0.channels]
                                   if src.arm == "adducts" else [])
    chans = list(dict.fromkeys(chans))
    fams = CX.file_families(src.summary, led)
    sp = SP.Space(*CX.run_space_args(src.summary), fams)
    dch = list(dict.fromkeys(list(ctx0.decomp_adducts) + chans))
    ctx = CX.RunContext(src.name, {fid: fa}, {fid: fa.gate}, {fid: win}, win, sp, ctx0.klass, ctx0.rp, ctx0.sig_fit,
                        ctx0.eff, src.nmin, chans, dch, ctx0.labelled, purity=ctx0.purity)
    ctx.engine_channels = list(ctx0.engine_channels)
    ctx.cse = ctx0.cse
    ctx.families = fams
    CX.prepare_indexes(ctx, {fid: led})
    ctx.twin_q = {fid: ctx0.twin_q[fid]} if fid in ctx0.twin_q else {}
    ctx.c13, _dev = CX.carbon_counts(ctx, None, facts, {fid: led}, src.nmin, ctx0.space.adducts, 0.0)
    ctx.alien = frozenset()
    src.ctx = ctx
    return ctx


def lists_of(src: Source) -> LS.ContextLists:
    if src.main is not None:
        return lists_of(src.main)
    if src._lists is None:
        s = src.summary
        ctx = context_of(src)
        src._lists = LS.ContextLists(_polarity(ctx), s.get("context"), s.get("reflists_active"),
                                     activation=src.inputs.activation or (s.get("reflists_context") or None))
    return src._lists


def scan_of(src: Source):
    """(first, last) m/z of the source's peaks (a main run's for an arm)."""
    if src.main is not None:
        return scan_of(src.main)
    if src._scan is None:
        src._scan = SPL.scan_range(src.per_file)
    return src._scan


def _polarity(ctx) -> str:
    return "positive" if any(str(x).endswith("+") for x in ctx.engine_channels) else "negative"


def _lead_by(per_file: dict) -> dict:
    from peaky.assignment import evidence as EV
    acc: dict = {}
    for led in per_file.values():
        if "tentative_lead" not in led.columns:
            continue
        m = led[(led["role"] == "M0") & led["tentative_lead"].map(EV.truthy)]
        lb = m["lead_by"] if "lead_by" in m.columns else pd.Series("", index=m.index)
        for nn, aa, b in zip(m["neutral_formula"].astype(str), m["adduct"].astype(str), lb):
            t = DC._txt(b)
            acc.setdefault((nn, aa), set())
            if t:
                acc[(nn, aa)].add(t)
    return {k: ",".join(sorted(v)) for k, v in acc.items()}


def prepared(src: Source):
    """Pass A + pass B and the decision's view of the source (levels.decide.Prepared)."""
    facts = pair_facts(src)
    ctx = context_of(src)
    pairs = CP.pairs_from_files(src.per_file, facts)
    if src.q1 is None:
        src.q1 = CP.q1_pass(ctx, pairs, src.nmin)
    core, comps = CP.pass_b(ctx, facts, pairs, src.q1)
    P = facts.merge(core, on=["neutral_formula", "adduct"], how="left")
    pol = _polarity(ctx)
    main_ctx = context_of(src.main) if src.main is not None else ctx
    tol_ppm = CX.K_SIGMA * main_ctx.run_window[1]
    gate = None
    if pol == "positive":
        rec = src.summary.get("amine_r_min")          # the run's own value when it recorded one (0.0 included)
        r_min = SPL.AMINE_R_MIN if rec is None else float(rec)
        ri = src.inputs
        prot = ri.protected if (ri.protected is not None and src.main is None) else SPL.protected_neutrals(src.per_file)
        if src.main is None and ri.merged is not None:
            gate = SPL.AmineGate(ri.merged, ri.ts, prot, scan_of(src), per_file=src.per_file, r_min=r_min)
        else:
            gate = SPL.AmineGate(P, None, prot, scan_of(src), per_file=src.per_file, r_min=r_min)
    return DC.Prepared(name=src.name, P=P, comps=comps, pf=src.per_file, ctx=ctx, lists=lists_of(src),
                       ts=RT.ts_groups(src.inputs.ts) if src.main is None else None, tol_ppm=tol_ppm, pol=pol,
                       run_classes=DC.run_classes_of(ctx.engine_channels), below=DC.below_classes(src.per_file),
                       alien=ctx.alien if src.main is None else frozenset(), arm=src.one_file_minima,
                       skip_m0=CP.skip_m0_of(src.q1), gate=gate, lead_by=_lead_by(src.per_file))


# ---------------------------------------------------------------------------
# levelling
# ---------------------------------------------------------------------------
def na_text(klass, r200) -> str:
    if klass is None:
        return "NA · not assessed on this instrument class (no width model: the instrument class is unknown)"
    return (f"NA · not assessed on this instrument class (width model R(200) = {r200:,.0f} < "
            f"{_ev().ORBITRAP_R200:,.0f})").replace(",", " ")


def committed_pairs(per_file: dict) -> list[tuple[str, str]]:
    """The sorted committed (neutral, adduct) pairs of the per-file M0 rows."""
    out = set()
    for led in per_file.values():
        m0 = led[led["role"].astype(str) == "M0"]
        out |= {(n, a) for n, a in zip(m0["neutral_formula"].fillna("").astype(str), m0["adduct"].fillna("").astype(str))
                if n and a}
    return sorted(out)


def na_frame(src: Source, *, facts: bool = True) -> pd.DataFrame:
    """The level frame of a source the scale does not assess (a TOF-class or
    class-less width model): every committed pair NA with the reason, then --
    ``facts`` -- the pair-fact table's columns (`pair_facts`: the cheap per-pair
    facts the vetoes and step 0 read, pooled as on an Orbitrap batch), so
    tables/evidence_levels.csv keeps the facts on every run (D17). No
    enumeration, no gate, no pass A. The per-file stage asks for the scale's
    columns only (``facts=False``: no fact work at all)."""
    klass, r200 = src.instrument()
    return _unlevelled_frame(src, "NA", na_text(klass, r200), SC.CLAIM_NA, facts=facts)


def no_run_window(src: Source) -> bool:
    """Whether an Orbitrap-class source has no run window: none of its files is
    calibrated (every persisted `degeneracy_cal` null, no refit), so there is
    no sigma to borrow and no window to enumerate competitors in."""
    return not np.isfinite(context_of(src).run_window[1])


def no_window_frame(src: Source) -> pd.DataFrame:
    """The level frame of a source with no run window (`no_run_window`): every
    committed pair without a level, claim tentative, the reason in `evidence`
    (`evidence.NO_RUN_WINDOW_TEXT`), the pair facts beside it -- as the
    per-file stage writes for an uncalibrated file levelled alone."""
    return _unlevelled_frame(src, "", _ev().NO_RUN_WINDOW_TEXT, SC.claim_class(""), facts=True)


def _unlevelled_frame(src: Source, level: str, evidence: str, claim: str, *, facts: bool) -> pd.DataFrame:
    keys = committed_pairs(src.per_file)
    if not keys and facts:
        return empty_levels(src)          # no committed pair: the empty frame of every stage (a blank batch)
    out = pd.DataFrame({"neutral_formula": pd.Series([k[0] for k in keys], dtype=object),
                        "adduct": pd.Series([k[1] for k in keys], dtype=object)})
    for c in SC.COLUMNS:
        out[c] = ""
    out["evidence_level"] = level
    out["evidence"] = evidence
    out["claim"] = claim
    if facts:
        F = pair_facts(src)
        F = F[[c for c in F.columns if c not in out.columns or c in ("neutral_formula", "adduct")]]
        F = F.drop_duplicates(["neutral_formula", "adduct"])
        out = out.merge(F, on=["neutral_formula", "adduct"], how="left")
    return out


def empty_levels(src: Source) -> pd.DataFrame:
    """The level frame of a source with no committed pair: no row, the columns
    `level_source` writes (neutral_formula, adduct, the scale's COLUMNS, the
    step facts, pass B's and the pair-fact table's columns) -- so the per-file
    stage, the pooled stage and their tables see the same shape on a run that
    committed nothing (an empty decoy arm, a blank file)."""
    front = ["neutral_formula", "adduct", *SC.COLUMNS]
    cols = front + [c for c in DC.RECORD_COLUMNS if c not in front]
    facts = pair_facts(src)
    for c in [*facts.columns, *CP.PASS_B_COLUMNS]:
        if c not in cols:
            cols.append(c)
    return pd.DataFrame({c: pd.Series(dtype=object) for c in cols})


def level_source(src: Source, *, partners=None) -> pd.DataFrame:
    """One row per pair: neutral_formula, adduct, the scale's COLUMNS, then the
    step facts (split, positive fact, named list, NH4 / window / chloride
    notes, tag kinds, the internal pass's level and why) and the pair / pass-B
    fact columns. ``partners`` = {neutral: {route class: [partner text]}}
    (`partners_from`; never on a one-file adapted source)."""
    klass, _r = src.instrument()
    if klass != "orbitrap":
        return na_frame(src)
    if not committed_pairs(src.per_file):
        return empty_levels(src)
    if no_run_window(src):
        return no_window_frame(src)
    S = prepared(src)
    if src.one_file_minima:
        partners = None
    res = DC.inpass(S, partners)
    texts = DC.inpass_texts(S, res)
    lv = DC.relevel(S, res)
    rec = DC.records(S, res, texts, lv)
    rec["claim"] = rec["evidence_level"].map(SC.claim_class)
    front = ["neutral_formula", "adduct", *SC.COLUMNS]
    rest = [c for c in rec.columns if c not in front]
    facts = S.P[[c for c in S.P.columns if c not in rec.columns]]
    return pd.concat([rec[front + rest], facts.reset_index(drop=True)], axis=1)
