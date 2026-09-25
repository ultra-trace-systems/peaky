"""Batch assignment over a PRESENCE-COVER sample subset, with per-file records.

This realises the sample-selection RULE (see sampling.py): rather than assign a
single averaged file (which misses analytes present only part of the run), we
assign each selected file SEPARATELY and combine. match_compounds is per-sample
— a synthetic union spectrum can't be scored — so combining real per-file
ledgers is the only principled path. The subset is a greedy presence set-cover
over the batch's m/z bins (no height floor, prevalence >= 2, marginal-gain stop);
the achieved coverage and the stop reason are recorded in batch_summary.json.

We keep every per-file ledger on disk (out_dir/per_file/<sid>_ledger.csv) so the
file-to-file JITTER can be investigated: does the same m/z get the same formula /
tier in every file, and is its mass spread real or just per-file calibration?

The combine step is OFFSET-AWARE: each file carries a median mass offset
(io_mascope.estimate_offset); clustering aligns peaks on offset-corrected m/z so a
genuine same-peak is not split by a per-file calibration shift, while the reported
jitter separates the raw spread from the calibration-removed (residual) spread.

Within a cluster the files VOTE, in two stages: the ION of the best EVIDENCE
CLASS wins -- each ion takes the best per-file evidence level of its readings
(the neutral established or corroborated > the formula / ion pinned > exact mass
alone; `_evidence_class`) -- and among ions of one class the ion carried by the
most files (tier and ion_score break ties); among that ion's labels -- the same
ion read as C13H14O4 [M+NH4]+ or as C13H17NO4 [M+H]+ -- the one Assigned in the
most files wins, because on such a pair Assigned means a discriminating channel
was present and Candidate means the file had nothing to decide with. The losing
readings stay on the merged row (`alternatives`, `n_files_ion`, `n_files_winner`,
`ion_agree`) as well as in jitter.csv (with each file's own evidence level).
The two positive-mode re-reads that can change a reading -- the hydrocarbon-on-
N-cluster re-read and the ammonium/amine gate -- are decided ONCE on the merged
ledger, from the union of every file's evidence, and say so in `tier_reason`.

THE RESIDUAL STAGE (sampling.py, THE RESIDUAL STAGE). The cover's merge is
followed by a second, TARGETED selection over the universe bins that no assigned
file holds, that the whole-batch stamp does not explain and that are bright
somewhere; the samples in which those bins peak are assigned through the same
per-file path, and `align()` then runs ONCE over every per-file ledger (cover +
residual) so the merged ledger, the trace reconciliation and the time-series
stamp include them. Merged rows carry `stage` (the stage that first held the
ion), `tables/selected_samples.csv` the extra picks (role 'residual'),
`tables/residual_bins.csv` the targeted bins, and
`batch_summary.json['selection']['residual']` the record. `residual=False`
reproduces the cover-only run exactly.

`align()` / `merge_union()` are PURE (offline-tested). `run()` does the network
assignment loop.
"""
from __future__ import annotations

import copy
import json
import os
import re
import time

import numpy as np
import pandas as pd

from peaky import paths as PT
from peaky.chem import profiles as P
from peaky.batch import sampling as SS

__version__ = "0.10.0"  # the vote reads the per-file EVIDENCE: a cluster's ions are
                        # ranked by the best evidence class of their readings before the
                        # file count (_evidence_class; align carries evidence_level /
                        # evidence_axes; jitter.csv carries the level)
                        # (0.9.0: the merge window is sized from the batch's own mass scatter
                        # -- traces.MassScale: one sigma per batch, merge + stamp windows from
                        # it; batch_summary['mass_scale'] -- the flat DEFAULT_TOL_PPM stays
                        # the BINNING tolerance and the floor of both windows;
                        # 0.8.1: traces.stamp block + tables/predicted_satellites.csv: the stamp's
                        # predicted satellites counted apart from observed, track coherence;
                        # 0.8.0: the targeted residual stage: a second selection + assignment
                        # after the cover's merge, one align() over both, stage provenance;
                        # 0.7.0: the merge is a VOTE -- n_files_ion / n_files_winner /
                        # alternatives / ion_agree, the batch-level gates' tier_reason)

# the BINNING tolerance: the selector's bins, the admission table, the trace index
# (one constant for every batch-level binning; see sampling.BATCH_TOL_PPM). It is
# also the default -- and the floor -- of the merge window: a batch run with a time
# series sizes its merge window from the batch's own measured mass scatter
# (traces.MassScale, measured once before the first merge; `merge_ppm`), never
# narrower than this and never wider than 2x it. Pure `align()` callers pass the
# window they mean; without a measured scale it is this constant.
DEFAULT_TOL_PPM = SS.BATCH_TOL_PPM
TIER_ASSIGNED = "Assigned"
TIER_CANDIDATE = "Candidate"
STAGE_COVER = "cover"          # a file of the presence cover (incl. its k_min pads)
STAGE_RESIDUAL = "residual"    # a file the residual stage targeted
TIER_RANK = {"Assigned": 2, "Candidate": 1}
_M0_COLS = ["mz", "neutral_formula", "adduct", "tier", "ion_score",
            "admitted_by", "occurrence",   # admission provenance, when present
            "ion_only_of",                 # the ion-only link (the winner file's parent peak), when present
            "resolvability", "sep_hwhm",   # the winner file's peak separability (assignment/resolvability.py), when present
            "evidence_level", "evidence_axes"]   # the file's own evidence level + axes (assignment/evidence.py): the vote's
                                                 # evidence class reads them (_evidence_class), when present

# THE VOTE'S EVIDENCE CLASS of one per-file reading (the key after the ion-only
# rule and before the file count in `_vote`). The tier engine levels every
# committed reading per file (docs/EVIDENCE_LEVELS.md): a curated identity (2b,
# 3a), the acid branch (3b) and a neutral established by two axes with one
# outside the channel (4a) say the NEUTRAL is right; a reading the `--corroborate`
# source holds too (the `corroborated` axis, at whatever level) has the other
# instrument's / channel's word for its neutral; 4b / 4c / 4d say the FORMULA
# or the ION is pinned and the neutral is not; 5a / 5b are exact mass alone or
# an assignment that argues with itself. Three classes, compared before any
# count: a reading of the first kind outranks one of the second whatever the
# file count, and one of the second outranks a mass-only reading; the count
# decides among equals. Measured on a 28-file TOF batch merged at its own 12 ppm
# window: with one ion's readings finally in one row, the count alone handed the
# peak of the Orbitrap-confirmed acid C9H16O6 [M+NO3]- (Assigned, 4a, 2 files)
# to C14H21N [M+Br]- (5a, 9 files), and the roster's pinic acid C9H14O4 (4b
# corroborated, 1 file) lost a 1-vs-1 tie on ion_score to a silicon formula --
# neither a bromide adduct of an amine nor an organosilicon is a reading a
# negative-mode CIMS should carry over the one the other instrument confirms.
EVIDENCE_CLASS_GOOD = frozenset({"1", "2a", "2b", "3a", "3b", "4a"})   # the neutral established
EVIDENCE_CLASS_MID = frozenset({"4b", "4c", "4d"})                      # the formula / the ion pinned
CORROBORATED_AXIS = "corroborated"   # the `evidence_axes` token of the --corroborate source's agreement
_LEVEL_RANK = {lv: i for i, lv in enumerate(["1", "2a", "2b", "3a", "3b", "4a", "4b", "4c", "4d", "5a", "5b"])}


def _has_axis(axes, name: str) -> bool:
    """True when `name` is one of the `|`-separated tokens of an evidence_axes string."""
    return name in _s(axes).split("|")


def _evidence_class(level, axes) -> int:
    """2 = the neutral established (2b / 3a / 3b / 4a) or corroborated by the
    --corroborate source (the `corroborated` axis, at any level); 1 = the formula
    or the ion pinned (4b / 4c / 4d); 0 = exact mass alone / self-contradicting
    (5a / 5b) or no level at all."""
    lv = _s(level)
    if lv in EVIDENCE_CLASS_GOOD or _has_axis(axes, CORROBORATED_AXIS):
        return 2
    if lv in EVIDENCE_CLASS_MID:
        return 1
    return 0


def _level_text(level_rank, corroborated) -> str:
    """'3b', '4b corroborated', '-' (no level): the evidence a vote note names."""
    lv = next((k for k, v in _LEVEL_RANK.items() if v == int(level_rank)), "-")
    return lv + (" corroborated" if int(corroborated) else "")


# ---------------------------------------------------------------------------
# pure: cross-file alignment + union  (no network)
# ---------------------------------------------------------------------------
def _cluster_mz(mz_sorted: np.ndarray, tol_ppm: float) -> np.ndarray:
    """Single-linkage gap clustering of an ASCENDING m/z array -> cluster ids."""
    cid = np.zeros(len(mz_sorted), dtype=np.int64)
    if len(mz_sorted) > 1:
        gaps = np.diff(mz_sorted) / mz_sorted[:-1] * 1e6
        cid[1:] = np.cumsum(gaps > tol_ppm)
    return cid


def _sum_counts(dicts) -> dict:
    """Element-wise sum of count dicts (None / empty skipped); {} when none."""
    out: dict = {}
    for d in dicts:
        for k, v in (d or {}).items():
            out[k] = out.get(k, 0) + int(v)
    return out


def _s(v) -> str:
    """A reading's label component as text: '' for any NA (None / NaN / pd.NA)."""
    try:
        if v is None or pd.isna(v):
            return ""
    except (TypeError, ValueError):
        pass
    return str(v)


def _ion_key(nf: str, ad: str) -> str:
    """The ION a reading names: the element counts of neutral + adduct (the tier
    engine's own `_ion_counts`) as a Hill-ordered formula with the charge sign,
    so the two labels of one ion -- C13H14O4 [M+NH4]+ and C13H17NO4 [M+H]+, both
    'C13H18NO4+' -- share a key. A reading whose adduct cannot be parsed is its
    own ion (the key is its text)."""
    if not nf or not ad:
        return f"{nf} {ad}"
    from peaky.assignment import tiers as _TI
    try:
        cnt = _TI._ion_counts(nf, ad)
    except Exception:          # noqa: BLE001 -- an odd adduct string is its own ion
        cnt = None
    if not cnt:
        return f"{nf} {ad}"
    order = sorted(cnt.items(), key=lambda kv: (kv[0] != "C", kv[0] != "H", kv[0]))
    sign = "+" if "]+" in ad else "-" if "]-" in ad else ""
    return "".join(f"{e}{n if n != 1 else ''}" for e, n in order) + sign


def _vote(g: pd.DataFrame):
    """Rank one cluster's readings in two stages. Returns (ions, labels):
    `ions` one row per ion (_ion, n_files, n_assigned, best_ion, regular,
    best_cls, best_lr, corroborated), best first; `labels` one row per
    (neutral_formula, adduct) reading of EVERY ion (_ion, _nf, _ad, n_files,
    n_assigned, best_ion, ...), the winning ion's readings ranked best first
    and listed first, the other ions' readings after them in ion order.

    1. WHICH ION sits at this m/z is what files can genuinely disagree on, and
       the EVIDENCE decides it before the count: each ion takes the best
       evidence class of its per-file readings (`_evidence_class` over the
       file's own `evidence_level` / `evidence_axes` -- the neutral established
       or corroborated, above the formula / ion pinned, above exact mass alone;
       every reading is class 0 when the frames carry no level, and the vote is
       then the pure count it was), and the ion of the best class wins; among
       ions of one class the ion carried by the most FILES wins, the number of
       files carrying it at Assigned tier and the best ion_score only break
       ties, and the ion's own text is the last key -- so a full tie resolves
       the same way whatever order the files arrived in (serial and parallel
       runs stay byte-identical). The count-first order is the one
       collapse_trace_labels applies to competing labels on one trace.

       Why the evidence first: once the merge window put one TOF ion's readings
       in one row (they used to sit in rows of their own, 8-12 ppm apart, each
       looking unanimous), a 4-file mass-only reading (5b) outvoted a 1-file
       reading with an acid-branch corroboration (3b) at the same peak, and the
       board lost the Orbitrap-confirmed HOMs and the roster's pinic acid to
       bromide adducts of N-compounds read in more files. A per-file level is
       a measurement of THAT file's evidence for the reading; the count of
       files is a measurement of persistence. The first says which reading is
       right, the second how often it was seen -- and a reading no file could
       establish does not become right by being fitted in more of them.

       Nothing is exempt from the count. A known-species identity (the pass-0
       list) used to be -- an ion carrying one ranked first once its label had
       reached Assigned somewhere, so a list identity locked in one file on its
       own evidence was not outvoted by grid guesses -- and that exemption is
       retired: the batch pools every file's known-species evidence and decides
       each such ion ONCE, by evidence, on the merged ledger AFTER the vote
       (`lock_known_species`). A species confirmed somewhere and refuted nowhere
       takes its cluster whatever the count, a species some file could test and
       found wanting is left to the count, and the row says which.

    2. WHICH LABEL of that ion -- the same ion read as C13H14O4 [M+NH4]+ or as
       C13H17NO4 [M+H]+ (the reagent-N isobar) -- is decided by corroboration,
       not by count: the tier engine marks such a reading Assigned only when a
       discriminating channel was present in that file (an N-free sibling, the
       joint NH4+urea pair, a series anchor) and Candidate when it had nothing
       to decide with, so counting Candidate files would be counting silence.
       The label Assigned in the most files wins; file count and score break
       ties. On the 15-file uronium run 16 of the 43 same-ion splits had a majority
       label nobody had corroborated against a minority label some file had."""
    assigned = g["_r"] >= TIER_RANK[TIER_ASSIGNED]
    # the evidence class of every per-file reading (0 everywhere when the frames
    # carry no level: a pure `align()` caller without the evidence stage), the
    # rank of its level (for the row's note) and its corroborated axis
    if "evidence_level" in g.columns:
        axes = g["evidence_axes"] if "evidence_axes" in g.columns else pd.Series("", index=g.index)
        cls = pd.Series([_evidence_class(a, b) for a, b in zip(g["evidence_level"], axes)], index=g.index)
        lrank = pd.Series([_LEVEL_RANK.get(_s(v), len(_LEVEL_RANK)) for v in g["evidence_level"]], index=g.index)
        corr = pd.Series([int(_has_axis(b, CORROBORATED_AXIS)) for b in axes], index=g.index)
    else:
        cls = pd.Series(0, index=g.index)
        lrank = pd.Series(len(_LEVEL_RANK), index=g.index)
        corr = pd.Series(0, index=g.index)
    # an ION-ONLY reading (the `ion_only` stage: the acid's own composition as
    # a radical anion, committed beside its [M-H]- parent with an `ion_only_of`
    # link) is a bucket kept apart from the tiers, and it must not MOVE a
    # regular reading at the merge either: however many files carry it, it
    # ranks below every regular ion in the cluster and goes to `alternatives`.
    # The case: the radical anion of a C_n acid sits 0.44 mDa from the
    # labelled-nitrate cluster of the C_{n-1} organonitrate (the CO3 / ^NO3
    # degeneracy), so a file that left the peak unexplained and read it ion-only
    # must not outvote the files that read the organonitrate.
    io = (g["ion_only_of"].notna() if "ion_only_of" in g.columns
          else pd.Series(False, index=g.index))
    gg = g.assign(_asrc=g["src"].where(assigned),       # the file, when Assigned there
                  _reg=(~io).astype(int),                # 1 = a regular reading in this file
                  _cls=cls, _lr=lrank, _corr=corr)
    lab = (gg.groupby(["_ion", "_nf", "_ad"], sort=True)  # text order = last key
             .agg(n_files=("src", "nunique"),
                  n_assigned=("_asrc", "nunique"),        # FILES at Assigned, not rows
                  best_ion=("ion_score", "max"), regular=("_reg", "max"),
                  best_cls=("_cls", "max"), best_lr=("_lr", "min"), corroborated=("_corr", "max"))
             .reset_index())
    ions = (gg.groupby("_ion", sort=True)
              .agg(n_files=("src", "nunique"), n_assigned=("_asrc", "nunique"),
                   best_ion=("ion_score", "max"), regular=("_reg", "max"),
                   best_cls=("_cls", "max"), best_lr=("_lr", "min"), corroborated=("_corr", "max"))
              .reset_index())
    ions = ions.sort_values(["regular", "best_cls", "n_files", "n_assigned", "best_ion"],
                            ascending=False, kind="mergesort")   # stable: keeps text order
    rank = {k: i for i, k in enumerate(ions["_ion"])}
    lab = (lab.assign(_k=lab["_ion"].map(rank))
              .sort_values(["_k", "n_assigned", "n_files", "best_ion"],
                           ascending=[True, False, False, False], kind="mergesort")
              .drop(columns="_k"))
    return ions, lab


def _describe(r) -> str:
    """One losing reading for the merged row: 'C15H25N [M+H]+ x1 Candidate 0.97'
    (the best tier and score any file gave it)."""
    tier = TIER_ASSIGNED if int(r["n_assigned"]) > 0 else "Candidate"
    score = "" if pd.isna(r["best_ion"]) else f" {float(r['best_ion']):.2f}"
    return f"{r['_nf']} {r['_ad']} x{int(r['n_files'])} {tier}{score}"


def align(per_file: dict, *, tol_ppm: float = DEFAULT_TOL_PPM,
          offsets: dict | None = None, stages: dict | None = None):
    """Align the M0 rows of several files by m/z and let the files VOTE on each
    cluster's reading (see `_vote`: the evidence class, then the count, decides
    WHICH ION; corroboration decides WHICH LABEL of it).

    per_file : {src -> DataFrame with _M0_COLS; `evidence_level` / `evidence_axes`
    are the file's own levels (assignment/evidence.py) and feed the vote's
    evidence class -- without them every reading is class 0 and the vote is
    the pure count}. offsets : {src -> median ppm}
    (subtracted before clustering so a per-file calibration shift does not split
    a peak). A known-species identity gets no exemption here: the batch decides
    it after the vote, by pooled evidence (`lock_known_species`). stages : {src ->
    STAGE_COVER | STAGE_RESIDUAL}, the stage that assigned each file
    (assign_batch.run's record); when given, every merged row carries `stage`
    -- STAGE_COVER if any file of the cluster is a cover file, else
    STAGE_RESIDUAL: the stage that first put the ion in the ledger. None (the
    default) leaves the column out, so a cover-only run's ledger is unchanged.

    Returns (merged, jitter):

      merged  one row per m/z cluster: consensus mz; the WINNING reading's
              neutral_formula / adduct / tier / ion_score / admitted_by /
              occurrence (its best per-file row: tier, then ion_score, then
              src); the vote -- n_files (files in the cluster), n_files_ion
              (files carrying the winning ion), n_files_winner (files carrying
              the winning reading), alternatives (the losing readings, best
              first, '' when unanimous), ion_agree (one ion in the cluster),
              formula_agree (one neutral), tier_reason (NA unless the vote had
              something to explain: an ion chosen by its evidence class over a
              lower-class ion carried by at least as many files, a label chosen
              by corroboration over a bigger count, or a regular reading kept
              over an ion-only one carried by more files); srcs[, stage],
              mz_jitter_ppm_raw, mz_jitter_ppm_caldj.
      jitter  long form, one row per (cluster, file): cluster, src, mz,
              neutral_formula, adduct, tier, ion_score, evidence_level -- every
              reading, winner or not.

    The previous rule ranked the number of ASSIGNED files first, which let one
    file's Assigned reading outvote many files' Candidate reading of a
    different ion: on the 15-file uronium run 12 of the 73 split clusters
    were decided that way -- and the merged row then carried nothing to show
    the other files had read it differently."""
    offsets = offsets or {}
    frames = []
    for src, df in per_file.items():
        if df is None or not len(df):
            continue
        d = df[[c for c in _M0_COLS if c in df.columns]].dropna(subset=["mz"]).copy()
        d["src"] = src
        off = float(offsets.get(src, 0.0) or 0.0)
        d["_mz_adj"] = d["mz"] * (1.0 - off / 1e6)   # offset-corrected for alignment
        frames.append(d)
    if not frames:
        return (pd.DataFrame(columns=["mz", "neutral_formula", "adduct", "tier",
                                      "ion_score", "admitted_by", "occurrence", "ion_only_of",
                                      "resolvability", "sep_hwhm",
                                      "n_files", "n_files_ion", "n_files_winner",
                                      "alternatives", "tier_reason", "srcs",
                                      *(["stage"] if stages is not None else []),
                                      "ion_agree", "formula_agree",
                                      "mz_jitter_ppm_raw", "mz_jitter_ppm_caldj"]),
                pd.DataFrame(columns=["cluster", "src", *_M0_COLS]))
    allm = pd.concat(frames, ignore_index=True).sort_values("_mz_adj").reset_index(drop=True)
    allm["cluster"] = _cluster_mz(allm["_mz_adj"].to_numpy(), tol_ppm)

    merged_rows, jitter_rows = [], []
    for cid, g in allm.groupby("cluster"):
        g = g.assign(_r=g["tier"].map(lambda t: TIER_RANK.get(str(t), 0)),
                     _nf=g["neutral_formula"].map(_s), _ad=g["adduct"].map(_s))
        g["_ion"] = [_ion_key(a, b) for a, b in zip(g["_nf"], g["_ad"])]
        ions, lab = _vote(g)
        win_ion, win = ions.iloc[0], lab.iloc[0]
        n_total = int(g["src"].nunique())
        gw = g[(g["_nf"] == win["_nf"]) & (g["_ad"] == win["_ad"])]
        # the winning reading's best per-file row donates tier / score / provenance;
        # src last so an exact tie is settled by name, not by arrival order
        best = gw.sort_values(["_r", "ion_score", "src"], ascending=[False, False, True],
                              kind="mergesort").iloc[0]
        # what the vote had to explain, on the row (a 1-of-10 winner needs a reason)
        notes = []
        others = ions.iloc[1:]
        if len(others) and "best_cls" in others.columns:
            # an ion of a lower evidence class carried by at least as many files
            # as the winner: the evidence decided the ion, not the count
            lower = others[(others.get("regular", 1) == 1) & (others["best_cls"] < int(win_ion["best_cls"]))
                           & (others["n_files"] >= int(win_ion["n_files"]))]
            if len(lower):
                big = lower.sort_values(["n_files", "best_cls"], ascending=False, kind="mergesort").iloc[0]
                big_lab = lab[lab["_ion"] == big["_ion"]].iloc[0]
                notes.append(f"evidence outranks the count: kept {win['_nf']} {win['_ad']} "
                             f"({_level_text(win_ion['best_lr'], win_ion['corroborated'])} in "
                             f"{int(win_ion['n_files'])} of {n_total} files) over the "
                             f"{int(big['n_files'])}-file {big_lab['_nf']} {big_lab['_ad']} "
                             f"({_level_text(big['best_lr'], big['corroborated'])})")
        if int(win_ion.get("regular", 1)) and len(others):
            # an ion-only reading carried by MORE files than the regular winner
            # stayed an alternative on purpose (see _vote): say so on the row
            io_more = others[(others.get("regular", 1) == 0) & (others["n_files"] > int(win_ion["n_files"]))]
            if len(io_more):
                top = io_more.iloc[0]
                top_lab = lab[lab["_ion"] == top["_ion"]].iloc[0]
                notes.append(f"regular reading kept over the {int(top['n_files'])}-file "
                             f"{top_lab['_nf']} {top_lab['_ad']} ion-only reading (vote "
                             f"{int(win_ion['n_files'])} of {n_total} files)")
        same = lab[lab["_ion"] == win_ion["_ion"]]
        if len(same) > 1 and int(same["n_files"].max()) > int(win["n_files"]):
            big = same.iloc[1:].sort_values("n_files", ascending=False, kind="mergesort").iloc[0]
            notes.append(f"same ion {win_ion['_ion']} read two ways: kept {win['_nf']} "
                         f"{win['_ad']} (Assigned in {int(win['n_assigned'])} of its "
                         f"{int(win['n_files'])} files) over the {int(big['n_files'])}-file "
                         f"{big['_nf']} {big['_ad']} (Assigned in {int(big['n_assigned'])})")
        mz_raw = g["mz"].to_numpy(); mz_adj = g["_mz_adj"].to_numpy()
        def _spread(a):
            return float((a.max() - a.min()) / a.mean() * 1e6) if len(a) > 1 else 0.0
        forms = set(g["neutral_formula"].dropna())
        srcs = sorted(set(g["src"]))
        rec = dict(
            mz=float(g["mz"].mean()), neutral_formula=best["neutral_formula"],
            adduct=best.get("adduct"), tier=best["tier"],
            ion_score=best.get("ion_score"),
            admitted_by=best.get("admitted_by"), occurrence=best.get("occurrence"),
            ion_only_of=best.get("ion_only_of", pd.NA),
            resolvability=best.get("resolvability", pd.NA), sep_hwhm=best.get("sep_hwhm", np.nan),
            n_files=n_total, n_files_ion=int(win_ion["n_files"]),
            n_files_winner=int(win["n_files"]),
            alternatives="; ".join(_describe(r) for _, r in lab.iloc[1:].iterrows()),
            tier_reason=" | ".join(notes) if notes else pd.NA,
            srcs=",".join(srcs))
        if stages is not None:
            # the stage that FIRST held the ion: a cover file anywhere in the
            # cluster makes it a cover row; only an ion seen in residual files
            # alone was found by the residual stage
            rec["stage"] = (STAGE_COVER if any(stages.get(x, STAGE_COVER) == STAGE_COVER
                                               for x in srcs) else STAGE_RESIDUAL)
        rec.update(ion_agree=(len(ions) <= 1),
                   formula_agree=(len(forms) <= 1),
                   mz_jitter_ppm_raw=round(_spread(mz_raw), 3),
                   mz_jitter_ppm_caldj=round(_spread(mz_adj), 3))
        merged_rows.append(rec)
        for _, r in g.sort_values("src").iterrows():
            jitter_rows.append(dict(cluster=int(cid), src=r["src"], mz=float(r["mz"]),
                                    neutral_formula=r.get("neutral_formula"),
                                    adduct=r.get("adduct"), tier=r.get("tier"),
                                    ion_score=r.get("ion_score"),
                                    evidence_level=r.get("evidence_level", pd.NA)))
    merged = pd.DataFrame(merged_rows).sort_values("mz").reset_index(drop=True)
    jitter = pd.DataFrame(jitter_rows)
    return merged, jitter


def merge_union(per_file: dict, **kw):
    """Just the merged union frame from align()."""
    return align(per_file, **kw)[0]


def _theo_ppm(mz, neutral, adduct):
    """Observed-vs-theoretical ppm for an assigned (neutral, adduct) at mz."""
    from peaky.chem import chemistry as C
    try:
        theo = C.ion_mz(str(neutral), str(adduct))
        return (float(mz) - theo) / theo * 1e6
    except Exception:
        return None


def jitter_report(per_file: dict, *, tol_ppm: float = DEFAULT_TOL_PPM):
    """File-to-file JITTER analysis over the per-file M0 frames (the user's goal:
    'investigate the jitter'). For each assignment we compute the observed-vs-
    theoretical ppm, giving each FILE a calibration offset (median ppm); peaks are
    then compared two ways:

      by_formula : keyed on (neutral_formula, adduct) — the same assignment seen
                   in >=2 files. `mz_jitter_raw` = ppm spread of the raw masses;
                   `mz_jitter_resid` = spread AFTER removing each file's offset
                   (the genuine peak-position noise vs mere calibration drift).
      by_mz      : keyed on offset-corrected m/z clusters — exposes FORMULA
                   DISAGREEMENTS (same peak, different formula across files).

    Returns dict: {offsets, by_formula (DataFrame), by_mz (DataFrame), summary}.
    """
    # per-file offset = median observed-vs-theoretical ppm of its assignments
    offsets, rows = {}, []
    for src, df in per_file.items():
        if df is None or not len(df):
            offsets[src] = None
            continue
        d = df.dropna(subset=["mz", "neutral_formula", "adduct"]).copy()
        d["ppm"] = [_theo_ppm(m, n, a) for m, n, a in
                    zip(d["mz"], d["neutral_formula"], d["adduct"])]
        d = d.dropna(subset=["ppm"])
        offsets[src] = float(d["ppm"].median()) if len(d) else None
        d["src"] = src
        rows.append(d)
    if not rows:
        return {"offsets": offsets, "by_formula": pd.DataFrame(),
                "by_mz": pd.DataFrame(), "summary": {}}
    allm = pd.concat(rows, ignore_index=True)

    # --- by_formula: same (neutral, adduct) across files ---
    frows = []
    for (nf, ad), g in allm.groupby(["neutral_formula", "adduct"]):
        if g["src"].nunique() < 2:
            continue
        ppm = g["ppm"].to_numpy()
        resid = np.array([p - (offsets[s] or 0.0) for p, s in zip(g["ppm"], g["src"])])
        frows.append(dict(
            neutral_formula=nf, adduct=ad, n_files=int(g["src"].nunique()),
            mz=float(g["mz"].mean()),
            mz_jitter_raw=round(float(ppm.max() - ppm.min()), 3),
            mz_jitter_resid=round(float(resid.max() - resid.min()), 3),
            tiers=",".join(sorted(set(map(str, g["tier"])))),
            tier_stable=(g["tier"].nunique() == 1),
            ion_score_min=round(float(g["ion_score"].min()), 3) if "ion_score" in g else None,
            ion_score_max=round(float(g["ion_score"].max()), 3) if "ion_score" in g else None))
    by_formula = (pd.DataFrame(frows).sort_values("mz_jitter_resid", ascending=False)
                  .reset_index(drop=True)) if frows else pd.DataFrame()

    # --- by_mz: offset-corrected m/z clusters -> formula disagreements ---
    allm["_mz_adj"] = [m * (1 - (offsets[s] or 0.0) / 1e6)
                       for m, s in zip(allm["mz"], allm["src"])]
    a = allm.sort_values("_mz_adj").reset_index(drop=True)
    a["cluster"] = _cluster_mz(a["_mz_adj"].to_numpy(), tol_ppm)
    mrows = []
    for cid, g in a.groupby("cluster"):
        forms = sorted(set(g["neutral_formula"].dropna()))
        if g["src"].nunique() < 2:
            continue
        mrows.append(dict(mz=float(g["mz"].mean()), n_files=int(g["src"].nunique()),
                          n_formulas=len(forms), formulas="; ".join(forms),
                          disagree=(len(forms) > 1)))
    by_mz = pd.DataFrame(mrows)

    shared = len(by_formula)
    disagree = int(by_mz["disagree"].sum()) if len(by_mz) else 0
    jr = by_formula["mz_jitter_raw"] if shared else pd.Series(dtype=float)
    jres = by_formula["mz_jitter_resid"] if shared else pd.Series(dtype=float)
    summary = {
        "offsets_ppm": {k: (round(v, 3) if v is not None else None) for k, v in offsets.items()},
        "offset_spread_ppm": round(float(max(v for v in offsets.values() if v is not None)
                                         - min(v for v in offsets.values() if v is not None)), 3)
        if any(v is not None for v in offsets.values()) else None,
        "shared_assignments": shared,
        "tier_unstable": int((~by_formula["tier_stable"]).sum()) if shared else 0,
        "formula_disagreements": disagree,
        "mz_jitter_raw_median": round(float(jr.median()), 3) if shared else None,
        "mz_jitter_raw_p95": round(float(jr.quantile(0.95)), 3) if shared else None,
        "mz_jitter_resid_median": round(float(jres.median()), 3) if shared else None,
        "mz_jitter_resid_p95": round(float(jres.quantile(0.95)), 3) if shared else None,
    }
    return {"offsets": offsets, "by_formula": by_formula, "by_mz": by_mz,
            "summary": summary}


def _m0(ledger: pd.DataFrame) -> pd.DataFrame:
    """Extract the M0 (assigned-compound) rows in the _M0_COLS schema."""
    role = ledger["role"] if "role" in ledger.columns else None
    m = ledger[role == "M0"] if role is not None else ledger
    cols = [c for c in _M0_COLS if c in m.columns]
    return m[cols].copy()


# provenance prefixes whose neutral identity is established independently of the
# [M+NH4]+ channel (curated reference lists, pass-0 known species, cross-channel
# certified neutrals) -- the amine gate keeps their ammonium adducts regardless of
# the NH4-vs-parent tracking test. The merged ledger drops `method`, so the set is
# gathered here from the full per-file ledgers.
_PROTECTED_METHODS = ("reflist-rescue", "known:", "certified:")


def _neutrals_by_method(ledger: pd.DataFrame, prefixes: tuple) -> set:
    if not {"method", "neutral_formula"} <= set(ledger.columns):
        return set()
    meth = ledger["method"].astype(str)
    keep = meth.str.startswith(prefixes)
    return set(ledger.loc[keep, "neutral_formula"].dropna().astype(str)) - {"nan", ""}


def _protected_neutrals(ledger: pd.DataFrame) -> set:
    """Neutrals the amine gate must not re-read (see _PROTECTED_METHODS)."""
    return _neutrals_by_method(ledger, _PROTECTED_METHODS)


# ---------------------------------------------------------------------------
# known species, decided once per batch by pooled evidence
# ---------------------------------------------------------------------------
# what pass 0 writes into a `known:` commit's commentary: "... = <label>, ppm
# <x>, ion score <y>; ...; corroborated by <route>"
_KNOWN_LABEL_RE = re.compile(r"=\s*(.+?),\s*ppm\s")
_KNOWN_ROUTE_RE = re.compile(r"corroborated (?:by|across) (.+?)(?:;|$)")
#: the route of a family whose own rule is exact mass alone (the perfluoroacids,
#: the nitroaromatics, the C0 atmospheric acids, reactive iodine ...): such a
#: commit carries no evidence the batch can pool beyond the count of files it
#: fitted in, so it never overrides the vote (see lock_known_species)
MASS_ONLY_ROUTE = "exact mass, on-cal (the family's own rule)"
#: the recorded satellite labels that corroborate an element of the NEUTRAL
#: (tiers._DIAG_ISO, per element): a `[M+Br]-` reading's 81Br line is the
#: reagent's twin, evidence of the adduct, not of the neutral -- it counts only
#: when the neutral itself carries bromine
_DIAG_TAGS = {"Br": ("81Br",), "Cl": ("37Cl",), "S": ("34S",), "Si": ("29Si", "30Si")}
#: label parts that name a line of the ENVELOPE as a whole rather than one
#: isotope: pass 0 records them on a multi-halogen species whose commit landed
#: on one member of the pattern (an `M0` two Da below a chlorinated paraffin's
#: committed 37Cl line, an `M+6` three lines up). They are the envelope's own
#: lines and count; a carbon-only tag (13C, 13C2) never does.
_ENVELOPE_PART = re.compile(r"^M(?:0|\+\d+)$")


def _own_satellites(neutral: str, isotopologues, kid_labels=()) -> list[str]:
    """The recorded lines of the neutral's own isotope envelope -- one per
    distinct peak: the row's `isotopologues` entries (each names its peak) and
    the iso_child rows pointing at the row (`kid_labels`: (peak_id, label)
    pairs, or bare labels), a peak recorded in both places -- or under two
    labels, a 37Cl2 and an 81Br+37Cl the instrument does not separate --
    counted once. A line counts when its label carries a diagnostic isotope of
    an element the neutral contains (`_DIAG_TAGS`) or names an envelope line
    outright (`_ENVELOPE_PART`: M0, M+6 -- pass 0 records those on a
    multi-halogen species whose commit sits on one member of the pattern). A
    `[M+Br]-` reading's bare 81Br line is the reagent's twin -- evidence of the
    adduct, not of the neutral -- and is left out unless the neutral itself
    carries bromine; a 13C line never counts."""
    seen: dict = {}
    try:
        entries = (json.loads(isotopologues) if isinstance(isotopologues, str)
                   else (isotopologues or []))
    except (TypeError, ValueError):
        entries = []
    for d in entries:
        if isinstance(d, dict):
            seen.setdefault(str(d.get("peak_id")) if d.get("peak_id") is not None else f"#{len(seen)}",
                            str(d.get("label", "")))
    for k in kid_labels:
        pid, lab = (k if isinstance(k, tuple) else (None, k))
        if lab is None or str(lab) in ("nan", "<NA>", ""):
            continue
        key = str(pid) if pid is not None and str(pid) not in ("nan", "<NA>", "") else f"#{len(seen)}"
        seen.setdefault(key, str(lab))
    from peaky.chem import chemistry as _C
    counts = _C.parse_formula(str(neutral or ""))
    tags = [t for el, ts in _DIAG_TAGS.items() if counts.get(el, 0) > 0 for t in ts]

    def _counts(lab: str) -> bool:
        # a label is one line; its '+'-joined parts name what the line carries.
        # It counts when any part is an envelope line (M0, M+6) or an isotope of
        # a diagnostic element the neutral contains; 13C / 15N / 18O parts and
        # the reagent's own twin on a halogen-free neutral do not.
        if _ENVELOPE_PART.match(lab.strip()):
            return True
        return any(_ENVELOPE_PART.match(part.strip()) or any(t in part for t in tags)
                   for part in lab.split("+"))

    return [lab for lab in seen.values() if _counts(lab)]


def _known_route(commentary: str, neutral: str, isotopologues, *, n_channels: int = 0,
                 kid_labels=()) -> str | None:
    """The evidence a `known:` commit rests on, read off its own row: a second
    ion channel of the neutral in the same ledger (`n_channels`, counted by the
    caller), else the recorded satellites of an element the neutral contains
    (`_own_satellites`: a chlorinated paraffin's 37Cl envelope, an iodine
    bromide's 81Br2 -- pass 0 records the lines it locked on, and the recovery
    path does too), else the commentary's own "corroborated by/across <route>",
    else None: the family's rule was exact mass."""
    if n_channels >= 2:
        return f"{n_channels} ion channels"
    hit = _own_satellites(neutral, isotopologues, kid_labels)
    if hit:
        from peaky.chem import chemistry as _C
        counts = _C.parse_formula(str(neutral or ""))
        tags = [t for el, ts in _DIAG_TAGS.items() if counts.get(el, 0) > 0 for t in ts]
        seen = sorted({t for t in tags if any(t in lab for lab in hit)})
        return f"a confirmed {'/'.join(seen)} envelope ({len(hit)} satellite{'s' if len(hit) != 1 else ''})"
    m = _KNOWN_ROUTE_RE.search(commentary or "")
    if m:
        return m.group(1).strip()
    return None


def known_evidence(ledger: pd.DataFrame, *, src=None) -> list[dict]:
    """One record per known-species reading this file anchored on-cal, for the
    batch to pool (`lock_known_species`): the committed `known:` M0 rows
    (verdict `confirmed`; `why` = the route read off the row (`_known_route`):
    the ion channels or the diagnostic envelope the commentary names, else the
    recorded satellites of an element the neutral contains, else exact mass --
    the family's own rule) and the `known_lead` records pass 0 left on
    the claims it refused (verdict `deferred` = this file could not test it,
    `refuted` = it tested it and it failed; ledger.py). `src` tags the file.
    Record: src, neutral, adduct, mz, family, label, verdict, why, summary (the
    reason without this file's numbers -- what the merged row counts files by),
    n_channels (the neutral's ion channels in that ledger), n_satellites (its
    recorded own-element diagnostic lines), ion_score, tier, admitted_by,
    occurrence (the last three None on a lead)."""
    out: list[dict] = []
    if ledger is None or not len(ledger) \
            or not {"neutral_formula", "adduct", "mz"} <= set(ledger.columns):
        return out

    def _num(v):
        try:
            f = float(v)
        except (TypeError, ValueError):
            return None
        return None if np.isnan(f) else f

    def _col(i, name):
        return _s(ledger.at[i, name]) if name in ledger.columns else ""

    if "method" in ledger.columns:
        meth = ledger["method"].map(_s)
        role = ledger["role"].map(_s) if "role" in ledger.columns \
            else pd.Series("M0", index=ledger.index)
        m0 = ledger[role == "M0"]
        # the neutral's ion channels in THIS ledger (the tier engine's own
        # cross-channel count) and the satellites attached to the row
        chan = m0.groupby(m0["neutral_formula"].map(_s))["adduct"].nunique() if len(m0) else pd.Series(dtype=int)
        kids: dict = {}
        if {"parent_peak_id", "iso_label", "peak_id"} <= set(ledger.columns):
            for pid, own, lab_ in zip(ledger.loc[role == "iso_child", "parent_peak_id"],
                                      ledger.loc[role == "iso_child", "peak_id"],
                                      ledger.loc[role == "iso_child", "iso_label"]):
                kids.setdefault(pid, []).append((own, lab_))
        for i in ledger.index[(role == "M0") & meth.str.startswith("known:")]:
            com = _col(i, "commentary")
            lab = _KNOWN_LABEL_RE.search(com)
            nf = _s(ledger.at[i, "neutral_formula"])
            n_ch = int(chan.get(nf, 0))
            kid_labels = kids.get(ledger.at[i, "peak_id"], ()) if "peak_id" in ledger.columns else ()
            iso_cell = ledger.at[i, "isotopologues"] if "isotopologues" in ledger.columns else None
            route = _known_route(com, nf, iso_cell, n_channels=n_ch, kid_labels=kid_labels)
            why = ("corroborated by " + route) if route else MASS_ONLY_ROUTE
            out.append(dict(
                src=src, neutral=nf,
                adduct=_s(ledger.at[i, "adduct"]), mz=float(ledger.at[i, "mz"]),
                family=meth[i][len("known:"):],
                label=lab.group(1).strip() if lab else "",
                verdict="confirmed", why=why, summary=why,
                n_channels=n_ch, n_satellites=len(_own_satellites(nf, iso_cell, kid_labels)),
                ion_score=_num(ledger.at[i, "ion_score"]) if "ion_score" in ledger.columns else None,
                tier=_col(i, "tier") or None,
                admitted_by=_col(i, "admitted_by") or None,
                occurrence=_num(ledger.at[i, "occurrence"]) if "occurrence" in ledger.columns else None))
    if "known_lead" in ledger.columns:
        for i in ledger.index[ledger["known_lead"].notna()]:
            try:
                d = json.loads(_s(ledger.at[i, "known_lead"]))
            except (TypeError, ValueError):
                continue
            if not isinstance(d, dict) or not d.get("formula"):
                continue
            out.append(dict(
                src=src, neutral=str(d["formula"]), adduct=_s(d.get("adduct")),
                mz=float(d["mz"]) if d.get("mz") is not None else float(ledger.at[i, "mz"]),
                family=_s(d.get("family")), label=_s(d.get("label")),
                verdict=_s(d.get("verdict")) or "deferred", why=_s(d.get("why")),
                summary=_s(d.get("summary")) or _s(d.get("why")),
                n_channels=int(d.get("channels") or 0), n_satellites=0,
                ion_score=_num(d.get("ion_score")), tier=None, admitted_by=None,
                occurrence=None))
    return out


def _pool_summary(conf: list, dfr: list, ref: list) -> str:
    """'confirmed in 2 files (...); could not test it in 3 files (...);
    refuted in 1 file (...)' -- each group's distinct reasons (a record's
    `summary`, the reason without that file's numbers, else its `why`), with
    the file count of each reason when there are several."""
    def _n(rows):
        return len({r["src"] for r in rows})

    def _whys(rows):
        order: list = []
        files: dict = {}
        for r in rows:
            w = str(r.get("summary") or r.get("why") or "").strip()
            if not w:
                continue
            if w not in files:
                order.append(w)
                files[w] = set()
            files[w].add(r["src"])
        if len(order) <= 1:
            return " / ".join(order)
        return " / ".join(f"{w} [{len(files[w])} file{'s' if len(files[w]) != 1 else ''}]" for w in order)

    parts = []
    for rows, verb in ((conf, "confirmed in"), (dfr, "could not test it in"), (ref, "refuted in")):
        if rows:
            n = _n(rows)
            parts.append(f"{verb} {n} file{'s' if n != 1 else ''} ({_whys(rows)})")
    return "; ".join(parts)


def lock_known_species(merged: pd.DataFrame, pool: list, *, tol_ppm: float = DEFAULT_TOL_PPM,
                       mz_floor_da: float = 1.5e-3, log=print) -> dict:
    """Decide every known-species ion ONCE for the batch, by the evidence the
    files pooled (`known_evidence`), on the merged ledger after the vote.

    The pass-0 lock is per file: it commits a species where THAT file shows the
    corroboration its family demands (>= 2 ion channels, or a diagnostic
    29Si/30Si / 34S / 37Cl / 81Br twin; exact mass alone for the monoisotopic
    families) and refuses it elsewhere. On a batch the twin clears the picker's
    floor in one file of ten, so the D7 cyclosiloxane urea adduct was known in
    one file and grid-fit as an O14 formula -- Candidate, flagged implausible
    by the engine itself -- in the nine others. The vote used to carry a
    "curated exemption" for that (a list identity Assigned somewhere ranked
    first); this replaces the exemption with the evidence:

      * a species CONFIRMED in at least one file and REFUTED in none takes its
        merged cluster whatever the file count -- the files that could not test
        it (`deferred`: a single channel, no twin the file could have shown) do
        not vote against it, because silence is not evidence;
      * a species some file could test and found wanting (`refuted`: the twin
        predicted above the floor and absent; an own-twin ratio or the Si M+1
        check failed) is left to the vote, and the row records the conflict --
        sulfolane, 34S-confirmed in one file, against fluorenone [M+H]+ in nine
        bright files that show no 34S, stays fluorenone, by evidence now; and
        where the vote's own winner IS the conflicted species and it was refuted
        in more files than it was confirmed in, the merged tier is capped at
        Candidate (the one file's Assigned cannot stand for the batch);
      * a lead never confirmed anywhere is noted on the row, not locked;
      * a species whose family's rule is exact mass alone (MASS_ONLY_ROUTE: no
        diagnostic twin, no second channel demanded) carries nothing to pool
        beyond the count of files it fitted in, so it never overrides the vote:
        a PFCA `[M-H]-` on-cal in two files of a ~4k TOF cannot displace an
        11-file, 81Br-corroborated CHOS `[M+Br]-` reading 6 ppm away; the row
        notes it, the vote stands;
      * a species of a family that demands corroboration is Assigned only when
        some confirming file holds TWO independent lines of evidence beyond
        the exact mass -- a second ion channel, or two own-element diagnostic
        satellites (`n_channels >= 2 or n_satellites >= 2`); one channel and one
        satellite line in every confirming file is capped at Candidate, locked
        or kept (the D7 cyclosiloxane: two files, one channel, the 29Si line
        alone, the 30Si line never testable at that intensity -- J12).

    The merged row a pooled reading belongs to is found by MEMBERSHIP -- the
    row whose own reading or whose `alternatives` lists it (the vote names
    every losing reading as "NF AD xN ...") -- and only for a reading no file
    committed by the m/z window: a minority reading's own m/z sits outside the
    merge window of a cluster whose mean the majority ion pulls 6-8 ppm away,
    which is exactly where a lock matters.

    A locked row takes the known reading (neutral, adduct, the confirmed files'
    best tier / score / admission provenance), moves the vote's winner to the
    head of `alternatives`, and says in `tier_reason` what the evidence was and
    what it overrode. A row already reading the species gains the note only.
    Mutates `merged` in place; returns counts (pooled, locked, confirmed_kept,
    conflict, lead_only, no_cluster) for `batch_summary["merge_gates"]`."""
    from peaky.assignment.cleanup import _note
    counts = {k: 0 for k in ("pooled", "locked", "confirmed_kept", "conflict",
                             "lead_only", "mass_only_outvoted", "no_cluster")}
    if not pool or merged is None or not len(merged) or "mz" not in merged.columns:
        return counts
    if "tier_reason" not in merged.columns:
        merged["tier_reason"] = pd.NA
    if "alternatives" not in merged.columns:
        merged["alternatives"] = ""
    mz = pd.to_numeric(merged["mz"], errors="coerce").to_numpy(dtype=float)
    pos = {i: k for k, i in enumerate(merged.index)}
    members: dict = {}                     # (neutral, adduct) -> merged rows holding it
    for i in merged.index:
        keys = {(_s(merged.at[i, "neutral_formula"]), _s(merged.at[i, "adduct"]))}
        for entry in _s(merged.at[i, "alternatives"]).split("; "):
            parts = entry.split(" ")
            if len(parts) >= 3 and parts[2].startswith("x"):
                keys.add((parts[0], parts[1]))
        for k in keys:
            members.setdefault(k, []).append(i)
    groups: dict = {}
    for rec in pool:
        groups.setdefault((str(rec["neutral"]), str(rec["adduct"])), []).append(rec)
    for (nf, ad), recs in sorted(groups.items()):
        counts["pooled"] += 1
        conf = [r for r in recs if r["verdict"] == "confirmed"]
        dfr = [r for r in recs if r["verdict"] == "deferred"]
        ref = [r for r in recs if r["verdict"] == "refuted"]
        mz0 = float(np.median([float(r["mz"]) for r in recs]))
        d = np.abs(mz - mz0)
        held = [j for j in members.get((nf, ad), ()) if np.isfinite(d[pos[j]])]
        if held:
            i = min(held, key=lambda j: d[pos[j]])
        else:
            cand = np.flatnonzero(np.isfinite(d) & (d <= max(mz0 * tol_ppm * 1e-6, mz_floor_da)))
            if not len(cand):
                counts["no_cluster"] += 1
                continue
            i = merged.index[cand[np.argmin(d[cand])]]
        label = next((str(r["label"]) for r in conf + dfr + ref if r.get("label")), "") or nf
        same = _s(merged.at[i, "neutral_formula"]) == nf and _s(merged.at[i, "adduct"]) == ad
        n_c = len({r["src"] for r in conf})
        mass_only = bool(conf) and all((r.get("summary") or r.get("why")) == MASS_ONLY_ROUTE for r in conf)
        two_lines = any(int(r.get("n_channels") or 0) >= 2 or int(r.get("n_satellites") or 0) >= 2
                        for r in conf)
        weak_note = ("; capped Candidate (one ion channel and one satellite line in every "
                     "confirming file: two independent lines are needed for Assigned)")
        old_nf, old_ad = _s(merged.at[i, "neutral_formula"]), _s(merged.at[i, "adduct"])
        old_n = (int(merged.at[i, "n_files_winner"])
                 if "n_files_winner" in merged.columns and pd.notna(merged.at[i, "n_files_winner"])
                 else 0)
        head = (f"known species decided once for the batch: {label} ({nf} {ad}) -- "
                f"{_pool_summary(conf, dfr, ref)}")
        if conf and not ref:
            if same:
                capped = (not mass_only and not two_lines and "tier" in merged.columns
                          and _s(merged.at[i, "tier"]) == TIER_ASSIGNED)
                if capped:
                    merged.at[i, "tier"] = TIER_CANDIDATE
                _note(merged, i, head + (weak_note if capped else ""))
                counts["confirmed_kept"] += 1
                continue
            if mass_only:
                _note(merged, i, f"mass-only known species {label} ({nf} {ad}) anchored on-cal in "
                                 f"{n_c} file{'s' if n_c != 1 else ''}; the vote's {old_n}-file {old_nf} "
                                 f"{old_ad} reading stands (exact mass alone cannot overrule a reading "
                                 "carried by more files)")
                counts["mass_only_outvoted"] += 1
                continue
            best = max(conf, key=lambda r: (TIER_RANK.get(str(r.get("tier")), 0),
                                            r.get("ion_score") or 0.0))
            old_sc = merged.at[i, "ion_score"] if "ion_score" in merged.columns else np.nan
            old = (f"{old_nf} {old_ad} x{old_n} {_s(merged.at[i, 'tier'])}"
                   + ("" if pd.isna(old_sc) else f" {float(old_sc):.2f}"))
            keep = [x for x in _s(merged.at[i, "alternatives"]).split("; ")
                    if x and not x.startswith(f"{nf} {ad} x")]
            srcs = set(_s(merged.at[i, "srcs"]).split(",")) if "srcs" in merged.columns else set()
            n_in = max(1, len({r["src"] for r in conf} & srcs) if srcs else n_c)
            n_total = (int(merged.at[i, "n_files"])
                       if "n_files" in merged.columns and pd.notna(merged.at[i, "n_files"])
                       else max(n_in, old_n))
            merged.at[i, "neutral_formula"] = nf
            merged.at[i, "adduct"] = ad
            merged.at[i, "tier"] = (best.get("tier") or TIER_ASSIGNED) if two_lines else TIER_CANDIDATE
            if "ion_score" in merged.columns:
                scores = [r["ion_score"] for r in conf if r.get("ion_score") is not None]
                merged.at[i, "ion_score"] = max(scores) if scores else np.nan
            for col in ("admitted_by", "occurrence"):
                if col in merged.columns:
                    merged.at[i, col] = best.get(col) if best.get(col) is not None else pd.NA
            if "ion_only_of" in merged.columns:
                merged.at[i, "ion_only_of"] = pd.NA
            for col in ("n_files_winner", "n_files_ion"):
                if col in merged.columns:
                    merged.at[i, col] = n_in
            merged.at[i, "alternatives"] = "; ".join([old] + keep)
            _note(merged, i, f"{head}; kept over the {old_n}-file {old_nf} {old_ad} reading "
                             f"(vote {n_in} of {n_total} files)" + ("" if two_lines else weak_note))
            counts["locked"] += 1
        elif conf and ref:
            n_r = len({r["src"] for r in ref})
            capped = (same and n_r > n_c and "tier" in merged.columns
                      and _s(merged.at[i, "tier"]) == TIER_ASSIGNED)
            if capped:
                merged.at[i, "tier"] = TIER_CANDIDATE
            _note(merged, i, f"known species {label} ({nf} {ad}) confirmed in {n_c} "
                             f"file{'s' if n_c != 1 else ''} but refuted in {n_r} "
                             f"({_pool_summary([], [], ref).split(' (', 1)[1][:-1]}); left to the vote"
                             + ("; capped Candidate (refuted in more files than confirmed)" if capped else ""))
            counts["conflict"] += 1
        elif dfr and not same:
            n_d = len({r["src"] for r in dfr})
            _note(merged, i, f"known-species lead: {label} ({nf} {ad}) anchored on-cal in {n_d} "
                             f"file{'s' if n_d != 1 else ''} but never corroborated "
                             f"({_pool_summary([], dfr, []).split(' (', 1)[1][:-1]}); not locked")
            counts["lead_only"] += 1
    log(f"[known] pooled {counts['pooled']} known-species ion(s): {counts['locked']} locked over the "
        f"vote, {counts['confirmed_kept']} confirmed where the vote already stood, "
        f"{counts['conflict']} in conflict (left to the vote), {counts['lead_only']} lead(s) only, "
        f"{counts['mass_only_outvoted']} mass-only species outvoted, "
        f"{counts['no_cluster']} without a merged row")
    return counts


# ---------------------------------------------------------------------------
# sample-level parallelism (process pool)
# ---------------------------------------------------------------------------
# Each A.run is self-contained: it builds its OWN Mascope client via connect()
# (which reads MASCOPE_URL/TOKEN/WORKSPACE from the env that 'spawn' inherits),
# owns a per-sample disk cache, and re-derives every mutated cfg field. So samples
# parallelise cleanly across processes. The heavy passes (degeneracy audit, pass2/4
# scoring) are pure-Python and GIL-bound, so a PROCESS pool -- not threads -- is
# what actually uses the extra cores. Determinism is preserved by reducing results
# in sample_ids order (align() has order-sensitive tie-breaks), NOT completion order.
_W: dict = {}   # per-worker-process read-only context, populated by _worker_init


def _worker_init(context, reflists_active, base_kw, ts_path):
    global _W
    _W = {"context": context, "reflists_active": reflists_active,
          "base_kw": base_kw, "ts_path": ts_path, "ts": None}


def _assign_one(sid: str) -> dict:
    """Top-level (spawn-picklable) worker: assign ONE sample and return the
    picklable pieces the parent reduces. The parent writes the per-file CSV and
    computes the offset -- in sample order -- so all disk I/O and the align() input
    order stay single-writer and deterministic."""
    import copy
    from peaky.assignment import assign as A
    kw = dict(_W["base_kw"])
    if kw.get("cfg") is not None:
        kw["cfg"] = copy.deepcopy(kw["cfg"])   # isolate per-sample cfg mutation
    if _W["ts_path"] is not None:
        if _W["ts"] is None:
            _W["ts"] = pd.read_parquet(_W["ts_path"])   # load once per process
        kw["ts_peaks"] = _W["ts"]
    lines: list = []
    res = A.run(sid, context=_W["context"], reflists_active=_W["reflists_active"],
                log=lines.append, **kw)
    return {"sid": sid, "ledger": res["ledger"],
            "plausibility_audit": res.get("plausibility_audit") or [],
            "stats": dict(res.get("stats", {})), "log": lines}


def _physical_cores() -> int:
    """Physical (not logical) core count -- the pool is CPU/GIL-bound, so
    hyperthreads give no speedup. os.cpu_count() returns logical cores."""
    import subprocess
    import sys
    if sys.platform == "darwin":
        try:
            out = subprocess.run(["sysctl", "-n", "hw.physicalcpu"],
                                 capture_output=True, text=True, timeout=2).stdout.strip()
            if out.isdigit():
                return max(1, int(out))
        except Exception:
            pass
    n = os.cpu_count() or 2
    return max(1, n // 2)


def _resolve_jobs(n_jobs, n_samples: int) -> int:
    """min(requested-or-physical-cores, n_samples), >=1. n_jobs=None consults
    $PEAKY_JOBS then falls back to physical cores; <=0 also means auto."""
    if n_jobs is None:
        env = os.environ.get("PEAKY_JOBS", "").strip()
        n_jobs = int(env) if env.lstrip("-").isdigit() else 0
    n_jobs = int(n_jobs)
    if n_jobs <= 0:
        n_jobs = _physical_cores()
    return max(1, min(n_jobs, max(1, n_samples)))


# ---------------------------------------------------------------------------
# network: assign each selected file, keep per-file records, combine
# ---------------------------------------------------------------------------
def _width_model_for_batch(resolving_power, client, table, log):
    """The batch's peak-width model (chem.resolution.Resolution) or None. None /
    'auto' measures it from the raw profile of a MIDDLING spectrum of `table`
    (the richest is the most crowded, so the worst place to look for an isolated
    peak, and the sparsest may have none); 'none' declines; a number is a
    constant R; a model is taken as is. A measurement that cannot be made
    returns None with a log line."""
    from peaky.batch import tracefirst as TFT
    rp = resolving_power
    if isinstance(rp, str) and rp.strip().lower() == "none":
        log("[resolution] resolvability stamp declined (--resolving-power none)")
        return None
    if rp is not None and not (isinstance(rp, str) and rp.strip().lower() == "auto"):
        rp = TFT.Resolution.coerce(float(rp) if isinstance(rp, str) else rp)
        log(f"[resolution] resolving power as given: {rp.describe()}")
        return rp
    if client is None or table is None or not len(table) or "sample_item_id" not in table.columns:
        log("[resolution] no per-peak table / server to measure the peak width from; "
            "resolvability is not stamped (pass --resolving-power <R> to declare one)")
        return None
    counts = table.groupby("sample_item_id").size().sort_values()
    probe = str(counts.index[len(counts) // 2])
    return TFT.measure_resolution(client, probe, log=log)


def run(peaks=None, *, batch: str | None = None, dataset: str | None = None,
        reagent: str = "auto", context: str | None = None,
        k_min: int = SS.K_MIN, k_max: int = SS.K_MAX, min_gain: float = SS.MIN_GAIN,
        min_prevalence: int = SS.MIN_PREVALENCE,
        out_dir: str, tol_ppm: float = DEFAULT_TOL_PPM,
        sample_ids: list | None = None, selection_meta: dict | None = None,
        residual: bool = SS.RESIDUAL_DEFAULT,
        residual_min_x_edge: float = SS.RESIDUAL_MIN_X_EDGE,
        residual_min_cps: float | None = None,
        residual_k_max: int = SS.RESIDUAL_K_MAX,
        residual_frac_of_max: float = SS.RESIDUAL_FRAC_OF_MAX,
        ts_peaks=None, amine_r_min: float = 0.6,
        n_jobs: int | None = None, rolling_centre: bool = False,
        trace_first: bool = False, resolving_power=None, trace_episodes: bool = False,
        corroborate=None, log=print, **assign_kw) -> dict:
    """Assign the presence-cover subset of a batch and combine, keeping per-file
    ledgers. Provide EITHER `peaks` (a batch peak/sample table) OR `batch` (a
    batch id or name -- exact id > exact name > unique substring, an ambiguous
    string raises; the per-sample list is fetched fresh from the live server,
    which also guarantees the selected sample ids are valid for get_peaks — cached
    ids go stale / 404 when the server copy is renamed). Selection needs the per-PEAK
    table: pass it as `ts_peaks` (the full-batch time series) or as a per-peak
    `peaks`. `k_min`/`k_max`/`min_gain`/`min_prevalence` tune the cover (see
    sampling.py). `sample_ids` skips selection (the pooled path); pass its
    `selection_meta` so batch_summary still records how they were chosen.
    `context` defaults to the reagent profile's context. Extra kwargs pass
    through to assign.run. Writes (see paths.RunPaths): merged_ledger.csv +
    batch_summary.json at the run root, per_file/<sid>_ledger.csv, and
    tables/{selected_samples,jitter}.csv.

    `residual` (default `sampling.RESIDUAL_DEFAULT`) runs the residual stage
    after the cover's merge (module note): the universe bins in no assigned
    file, unexplained by the stamp and reaching `residual_min_x_edge` x their
    sample's noise edge somewhere (never below the run's own gate multiple;
    `residual_min_cps` is an absolute floor instead) are covered by at most
    `residual_k_max` more samples, each counting for a bin only where the bin
    stands at >= `residual_frac_of_max` of its maximum and above that file's
    admission gate. Those files go through the same per-file path, and the
    merge + stamp are redone once over everything. Needs `ts_peaks`; the
    pooled path (`sample_ids=`) gets the extra picks back as
    `residual_samples` to append to its own selection table.

    `corroborate`: run dirs / ledger CSVs (or a ready set of neutral formulas)
    whose M0 neutrals corroborate this run -- the other reagent channel, or the
    other instrument on the same air. It is the `corroborated` axis of the
    evidence levels (docs/EVIDENCE_LEVELS.md): the per-file `evidence` stage
    reads it, and so does the batch level, which is recomputed on the POOLED
    per-file ledgers (cover + residual files as one source) and stamped on the
    merged ledger by (neutral_formula, adduct) -- the merged rows carry none of
    the predicate columns. batch_summary['evidence_levels'] records the counts;
    tables/evidence_levels.csv the facts behind every pair."""
    from peaky.assignment import assign as A
    from peaky.assignment import evidence as EV
    from peaky.batch import timeseries as _TSN
    from peaky.io import io_mascope as IO

    t_start = time.time()          # wall clock for summary['elapsed_s'] (see below)
    # ONE row per physical peak before anything reads the time series: Mascope
    # returns one row per target MATCH, which would double-count every peak two
    # targets claim (selection, the amine gate, sidelobe flagging and the
    # _batch_ts.parquet this writes). No-op on an already-collapsed frame.
    ts_peaks = _TSN.collapse_peak_matches(ts_peaks, log=log)
    out_dir = os.path.expanduser(out_dir)
    TAB = PT.run_paths(out_dir).ensure().tables    # .csv tables -> tables/
    pfdir = os.path.join(out_dir, "per_file")
    os.makedirs(pfdir, exist_ok=True)

    client = IO.connect()
    if peaks is None:
        if not batch:
            raise ValueError("need peaks= or batch=")
        # settle the batch once (exact id > exact name > unique substring): the
        # roster is fetched by id, and the DISPLAY name is what the reference
        # lists read and batch_summary records, so a run addressed by id still
        # reads as its batch
        rb = IO.resolve_batch(client, batch, dataset=dataset)
        peaks = IO.fetch_batch_samples(client, rb.id, dataset=dataset)
        batch = rb.name
        log(f"[assign_batch] fetched {len(peaks)} samples for batch {batch!r} "
            f"(id {rb.id})")

    prof = P.resolve(reagent, peaks)
    context = context or prof.context
    # The height-gated passes gate on a MULTIPLE of each sample's own noise edge.
    # Resolve that multiple ONCE for the batch -- the profile's own value when it
    # carries one, else the package default; a cfg the caller already set wins --
    # so every per-file run gates identically and the summary can record it.
    from peaky.assignment import passes as PA

    cfg = assign_kw.get("cfg") or PA.PassConfig()
    x_edge, x_edge_source = P.apply_height_cutoff_x_edge(cfg, prof, log=log)
    # the ion-only channels the profile opens ("[M]-." on the nitrate profiles),
    # copied by the same explicitness rule: a cfg that already carries a tuple wins
    P.apply_ion_only_channels(cfg, prof, log=log)
    assign_kw["cfg"] = cfg
    selection = dict(selection_meta or {})
    sel = None                 # our own cover table (None on the sample_ids= path)
    trace_sample = None
    # The peak-width model, resolved ONCE for the batch (chem.resolution) and
    # handed to every per-file run: it sizes trace-first's dedup cell and, on
    # every path, the `resolvability` stamp each per-file ledger carries (a
    # blended, uncorroborated peak is capped at Candidate; level 4c reads it).
    # MEASURED from the raw profile by default -- a TOF can be tuned anywhere
    # and a declared number is a guess -- and the caller's only when they gave
    # one. A measurement that cannot be made stops trace-first (nothing sizes
    # its cell) and is a log line on the cover path (the stage skips, the
    # columns stay NA).
    rp = _width_model_for_batch(resolving_power, client,
                                ts_peaks if ts_peaks is not None else peaks, log)
    if trace_first:
        # TRACE-FIRST: no files are selected -- the batch's persistent ions are
        # built as traces, centred, gated and handed to the engine as ONE
        # synthetic sample (batch.tracefirst), which then goes through the same
        # merge / reconciliation / stamp / residual stages as a cover file.
        if ts_peaks is None:
            raise ValueError("trace-first needs the batch time series (ts_peaks=)")
        from peaky.batch import tracefirst as TFT
        log("[phase] traces")
        if rp is None:
            raise ValueError(
                "could not measure the peak width from this batch's raw profile; pass "
                "--resolving-power <R> (the instrument's resolving power) instead")
        trace_sample = TFT.build_trace_sample(
            ts_peaks, sample_id=f"traces-{TFT.slug(batch or 'batch')}", reagent=prof.name,
            resolving_power=rp, episodes=trace_episodes, log=log)
        sample_ids = [trace_sample.sample_id]
        selection = {"method": "trace-first", "k": 1, **trace_sample.summary()}
        log(f"[assign_batch] trace-first: {selection['n_traces']} traces ({selection['n_seeds']} "
            f"seeds + {selection['n_satellites']} satellite positions) from {selection['n_spectra']} "
            f"spectra -> one synthetic sample")
        log("[phase] assign")
    if sample_ids is None:
        # greedy presence set-cover over the batch's m/z bins. Needs the per-PEAK
        # table: the pipeline passes it as ts_peaks; `peaks` may already be one.
        # The cover runs before any sample is assigned and is not quick on a big
        # batch, so it owns the phase for its duration (the caller already said
        # `assign`); the marker is handed back below, once the picks are in.
        log("[phase] select")
        src = ts_peaks if ts_peaks is not None else peaks
        if not SS.is_per_peak(src):
            raise ValueError("sample selection needs the per-peak batch table "
                             "(mz + height per peak): pass ts_peaks= (the batch "
                             "time series) or a per-peak peaks=")
        sel = SS.select_cover_samples(src, k_min=k_min, k_max=k_max,
                                      min_gain=min_gain, min_prevalence=min_prevalence,
                                      tol_ppm=tol_ppm)      # = the merge's tolerance
        selection = dict(sel.attrs.get("selection", {}))
        sample_ids = sel["sample_item_id"].tolist()
        sel.to_csv(os.path.join(TAB, "selected_samples.csv"), index=False)
        log(f"[assign_batch] {SS.describe(selection)}")
        _warn = SS.k_max_warning(selection)
        if _warn:
            log(f"[assign_batch] WARNING: {_warn}")
        log("[phase] assign")     # cover picked; everything past here is the run
    log(f"[assign_batch] {prof.label} context={context!r}: "
        f"{len(sample_ids)} selected files -> {pfdir}")

    # Force the reagent's analyte channels (we know the reagent at batch level) so
    # a per-sample match gap can't flip polarity / mis-assign a file. Caller can
    # still override via assign_kw['adducts'].
    assign_kw.setdefault("adducts", list(prof.adducts))
    # The hydrocarbon-on-N-cluster re-read (cleanup.relabel_reagent_n_adducts) is
    # decided ONCE on the merged ledger below, not per file. Its skip key -- does
    # this hydrocarbon show its own [M+H]+ -- is a presence test that flips with
    # each file's S/N: on the 15-file uronium run C15H22 [M+NH4]+ kept its reading in
    # the 14 files that also held C15H22 [M+H]+ and was re-read to C15H25N [M+H]+
    # in the one file that did not -- a "disagreement" the spectra never had,
    # and a phantom minority reading for the vote. The merged ledger holds the
    # union of every file's [M+H]+ rows, so the same rule applied there gives the
    # batch one answer. An explicit reagent_n_relabel=True in assign_kw restores
    # the per-file re-read (and the merged-level pass then stands down).
    assign_kw.setdefault("reagent_n_relabel", False)
    # --corroborate: the neutrals of the named sources, or a ready set. Read by
    # the per-file evidence stage (through assign_kw) and by the pooled batch
    # level below; the paths are recorded so the run says what corroborated it.
    if isinstance(corroborate, (set, frozenset)):
        cross_sources, cross = [], {str(x) for x in corroborate}
    else:
        cross_sources = [str(x) for x in (corroborate or [])]
        cross = EV.corroborating_neutrals(cross_sources)
    assign_kw["corroborate"] = cross
    # the batch's width model -> every per-file `resolvability` stage (never re-measured per file)
    assign_kw["resolving_power"] = rp
    if cross_sources:
        log(f"[assign_batch] --corroborate: {len(cross)} neutral(s) from {len(cross_sources)} source(s)")
    # labelled-reagent covalent-product rescue (e.g. 15N-organonitrates); no-op
    # for every unlabelled reagent profile.
    if getattr(prof, "label_isotope", None):
        assign_kw.setdefault("label_isotope", prof.label_isotope)
        assign_kw.setdefault("label_max", prof.label_max)
    # the bottle's isotopic purity -> the '^X' impurity line in both the local
    # scorer's envelope and peaky's own (isotopes.set_label_purity). None => default.
    if getattr(prof, "purity", None) is not None:
        assign_kw.setdefault("label_purity", prof.purity)
    # thread the batch TS to the per-sample run so pass-7 (certified-neutral)
    # can use member-channel co-variation as OPTIONAL corroboration. Guarded:
    # the pass is fully functional with ts_peaks=None (single-sample runs, or
    # batches whose mass range excludes the reagent ions).
    if ts_peaks is not None:
        assign_kw.setdefault("ts_peaks", ts_peaks)
    # Admission gate, persistence path: the batch's per-bin occurrence table
    # (fraction of spectra in which each m/z bin holds a peak), computed ONCE
    # from the batch time series and handed to every per-sample run. A peak
    # whose bin recurs in >= the resolved threshold (a number given as
    # occurrence_min, or the batch's Otsu split for "auto") of the spectra is
    # eligible for formula search even below the height gate (see
    # assignment/admission.py). Binned at this run's `tol_ppm` -- the same
    # tolerance the merge below uses (default sampling.BATCH_TOL_PPM).
    from peaky.assignment import admission as ADM
    from peaky.batch import traces as TR
    # READ the cfg built + height-resolved above; never rebuild one here, which
    # would gate the batch on a config that skipped that resolution.
    _cfg = assign_kw["cfg"]
    occurrence_min = getattr(_cfg, "occurrence_min", ADM.DEFAULT_OCCURRENCE_MIN)
    _on = isinstance(occurrence_min, str) or (occurrence_min is not None and float(occurrence_min) > 0)
    occ_info = {"occurrence_min": occurrence_min, "occurrence_threshold": None,
                "n_peaks": 0, "n_persistent_peaks": 0, "n_persistent_traces": 0,
                "n_spectra": 0, "tol_ppm": tol_ppm}
    # ONE m/z-sorted index of the batch's peaks (batch.traces.PeakIndex) serves the
    # admission table here AND the trace reconciliation of the merged ledger below,
    # so a peak's occurrence, its trace and its stamp are one object at one rule.
    _idx = TR.PeakIndex(ts_peaks, tol_ppm=tol_ppm) if ts_peaks is not None and len(ts_peaks) else None
    # absolute hours per index sample code, for the rolling centre and the
    # drift-following stamp (`rolling_centre`); None without timestamps
    _hours = _TSN.sample_hours(_idx, ts_peaks) if rolling_centre and _idx is not None else None
    _occ, _thr = assign_kw.get("occurrence"), None
    if _on and _idx is not None and _occ is None:
        _occ = ADM.bin_occurrence(ts_peaks, tol_ppm=tol_ppm, index=_idx)
        assign_kw["occurrence"] = _occ
    if _on and _occ is not None:
        _thr = ADM.resolve_threshold(_cfg, _occ)
        _n_pers = int((pd.to_numeric(_occ["occurrence"], errors="coerce") >= _thr).sum()) \
            if _thr is not None else 0
        occ_info.update(n_peaks=int(len(_occ)), occurrence_threshold=_thr,
                        n_spectra=int(_occ.attrs.get("n_samples", 0)),
                        tol_ppm=float(_occ.attrs.get("tol_ppm", tol_ppm)),
                        n_persistent_peaks=_n_pers,
                        n_persistent_traces=ADM.persistent_trace_count(_occ, _thr))
        if _thr is None:
            log(f"[assign_batch] admission: persistence path off -- {ADM.why_off(_cfg, _occ)}")
        else:
            log(f"[assign_batch] admission: {occ_info['n_persistent_peaks']} of {occ_info['n_peaks']} "
                f"batch peaks (~{occ_info['n_persistent_traces']} ions) persist in >= {_thr:.2f} of "
                f"{occ_info['n_spectra']} spectra (threshold {occurrence_min!r}"
                + (f" = Otsu split {_occ.attrs.get('auto_threshold'):.2f}" if isinstance(occurrence_min, str) else "")
                + ") -> eligible below the height gate")
    # The brightness floor. An UNSET multiple resolved to the 'auto' policy above:
    # derive it from this batch's own peaks (admission.derive_height_cutoff_x_edge
    # -- the smallest grid multiple whose admitted peaks are at most
    # MAX_TRANSIENT_SHARE transient), stamp the NUMBER on the cfg every per-file
    # run copies, and say where it came from. No table to derive from (path off,
    # too few spectra, no time series) -> the package default, and the source
    # says why.
    gate_info: dict = {}
    if PA.is_auto_x_edge(_cfg.height_cutoff_x_edge):
        _der = ADM.derive_height_cutoff_x_edge(_occ, _thr)
        if _der is not None:
            _cfg.height_cutoff_x_edge = x_edge = float(_der["x_edge"])
            _pf = _der.get("picker_tail_fraction")
            _pf_txt = "n/a" if _pf is None else f"{_pf * 1e4:.1f} in 10 000"
            _tail = (f"{_pf_txt} peaks sit below {_der['picker_tail_x']:.2f}x their sample's edge, "
                     f"threshold {_der['picker_fraction_min'] * 1e4:.0f}")
            if _der.get("picker_into_noise"):
                x_edge_source = (f"derived from the batch's own peaks: the picker picks into the noise "
                                 f"({_tail}) and {_der['share_at_x']:.0%} of the peaks above {x_edge:g}x "
                                 f"the noise edge are transient (at most {_der['max_transient_share']:.0%} "
                                 f"allowed; {_der['share_at_1']:.0%} at 1x)"
                                 + ("; the grid's top value still exceeded it" if _der.get("bound") else "")
                                 + f" -- requested by {x_edge_source}")
            else:
                x_edge_source = (f"derived from the batch's own peaks: the picker stops at the noise edge "
                                 f"({_tail}), so the floor stays {x_edge:g}x "
                                 f"({_der['share_at_1']:.0%} of the peaks above it are transient)"
                                 f" -- requested by {x_edge_source}")
            gate_info = dict(_der, source=x_edge_source)
        else:
            _cfg.height_cutoff_x_edge = x_edge = float(PA.DEFAULT_HEIGHT_CUTOFF_X_EDGE)
            _why = ADM.why_off(_cfg, _occ) if _on else "occurrence_min switches the persistence path off"
            x_edge_source = (f"the package default {x_edge:g}x -- 'auto' had nothing to derive "
                             f"from ({_why})")
        log(f"[gate] height cutoff = {x_edge:g}x the sample's noise edge (from {x_edge_source})")
    # context-unlock the reference peaklists (contaminants always; chemistry-
    # specific lists when the batch OR DATASET name -- the chemistry often lives
    # only in the latter -- matches) -> selection prior + rescue.
    from peaky.assignment import reflists as RL
    reflists_active, _tags = RL.activate(batch or "", dataset or "", getattr(prof, "label", ""))
    if reflists_active:
        log(f"[assign_batch] reference lists active: {RL.active_versions(reflists_active)} "
            f"(context {sorted(_tags) or 'contaminants-only'})")
    per_file, offsets, per_stats = {}, {}, []
    scorings: dict = {}        # per-sample pattern_scoring, for the run manifest
    level_frames: dict = {}    # sid -> its ledger's M0/iso rows + predicate columns
                               # (evidence.trim): the pooled batch level's input
    identified_aux: list = []  # per-file identified-ion rows (reagent/iso/artifact
                               # + analyte ion_formula) for the parquet stamp
    plaus_audit: list = []     # per-file O-monster / carbon-cluster demotes, pooled
    protected_neutrals: set = set()   # curated/cross-channel identities the amine
    #   gate must not re-read (reflist / known-species / certified provenance) --
    #   e.g. NBBS, whose weak isobar-contaminated NH4 trace fails the tracking test
    #   yet is a genuine Keller-list contaminant adduct.
    known_pool: list = []      # every file's known-species evidence (known_evidence):
    #   the committed known: rows and the leads pass 0 left on the claims it
    #   refused -- pooled and decided once on the merged ledger (lock_known_species)
    stages: dict = {}          # sid -> STAGE_COVER | STAGE_RESIDUAL (align() reads it)
    n_jobs = _resolve_jobs(n_jobs, len(sample_ids))
    ts_path_written: list = []  # the raw-TS parquet the worker pool loads: written once

    residual_scope: list = []      # [sorted residual-bin m/z] once the residual stage runs
    scope_counts: dict = {}        # sid -> (kept, total) M0 rows under the trace-first scope

    def _apply(sid, led, plaus, stats, stage):
        """Parent-side reduce (called in sample_ids order): write the per-file CSV
        and fold this sample into the accumulators. Order-fixed so align() -- which
        has order-sensitive tie-breaks -- yields byte-identical output either path.

        Under TRACE-FIRST a residual file may only ADD what the trace stamp left
        unexplained: its M0 rows are kept within `tol_ppm` of a residual bin and
        dropped elsewhere. Otherwise ten per-file ledgers of a noisy TOF would
        merge back in on top of the traces -- the per-file lottery trace-first
        exists to avoid -- and out-vote a trace's reading (measured: a Candidate
        on the trace ledger re-read as Assigned by three residual files)."""
        led.to_csv(os.path.join(pfdir, f"{sid}_ledger.csv"), index=False)
        level_frames[sid] = EV.trim(led)
        plaus_audit.extend(plaus)
        protected_neutrals.update(_protected_neutrals(led))
        known_pool.extend(known_evidence(led, src=sid))
        m0 = _m0(led)
        if stage == STAGE_RESIDUAL and trace_sample is not None and residual_scope:
            bmz = residual_scope[0]
            pmz = pd.to_numeric(m0["mz"], errors="coerce").to_numpy(dtype=float)
            j = np.searchsorted(bmz, pmz)
            jl = np.clip(j - 1, 0, len(bmz) - 1)
            jr = np.clip(j, 0, len(bmz) - 1)
            d = np.minimum(np.abs(bmz[jl] - pmz), np.abs(bmz[jr] - pmz))
            keep = np.isfinite(pmz) & (d <= np.maximum(pmz * tol_ppm * 1e-6, TR.MZ_FLOOR_DA))
            scope_counts[sid] = (int(keep.sum()), int(len(m0)))
            log(f"[assign_batch]   {sid}: trace-first scope keeps {int(keep.sum())} of "
                f"{len(m0)} M0 rows (those on a residual bin)")
            m0 = m0[keep]
        per_file[sid] = m0
        stages[sid] = stage
        from peaky.batch import timeseries as _TSI
        identified_aux.append(_TSI.identified_rows(led))
        try:
            offsets[sid] = IO.estimate_offset(IO.fetch_peaks(client, sid, use_cache=True))
        except Exception:
            offsets[sid] = None
        # What this sample's candidates were scored at. Read here rather than
        # carried back from the worker: it is a property of the sample and its
        # peaks are cached, so the parent computes the same answer the worker did.
        try:
            scorings[sid] = IO.scoring_snapshot(client, sid)
        except Exception:      # provenance must not fail a completed sample
            scorings[sid] = None
        st = dict(stats)
        st.update(sample_id=sid, offset_ppm=offsets[sid],
                  n_M0=int((led["role"] == "M0").sum()) if "role" in led.columns else None)
        if residual:
            st["stage"] = stage        # stage provenance rides with the switch
        per_stats.append(st)
        log(f"[assign_batch]   {sid}: offset={offsets[sid]}")

    def _assign_files(ids: list, stage: str, n_jobs: int) -> None:
        """Assign `ids` -- the TAIL of `sample_ids`, which already holds them, so
        the (i/N) progress lines count on through the run whichever stage is on --
        serially or across a process pool, and fold each into the accumulators
        in id order (`_apply`). The cover stage and the residual stage share it."""
        offset = len(sample_ids) - len(ids)
        if n_jobs <= 1:
            for i, sid in enumerate(ids, offset + 1):
                log(f"[assign_batch] ({i}/{len(sample_ids)}) assigning {sid} ...")
                # Per-file cfg COPY -- the same isolation the worker pool gets in
                # _assign_one. A.run mutates the cfg it is handed (noise edge,
                # mechanism ids, and the fitted cal_mu/cal_sigma), and `calibrate`
                # LEAVES A PREVIOUS FIT IN PLACE when this file's backbone is
                # smaller than cal_min_n -- so one shared object would gate file
                # N+1's mass z-scores on file N's calibration.
                kw = dict(assign_kw, cfg=copy.deepcopy(cfg))
                res = A.run(sid, context=context, log=log,
                            reflists_active=reflists_active, **kw)
                _apply(sid, res["ledger"], res.get("plausibility_audit") or [],
                       dict(res.get("stats", {})), stage)
                # same line the parallel branch logs per completed future: it is what
                # advances a progress reader's samples bar (peaky/progress.py)
                log(f"[assign_batch] ({i}/{len(sample_ids)}) done {sid}")
        else:
            # Write the batch TS to disk once so workers load it from the parquet
            # rather than re-pickling the full-batch DataFrame into every process.
            base_kw = {k: v for k, v in assign_kw.items() if k != "ts_peaks"}
            ts_path = None
            _ts = assign_kw.get("ts_peaks")
            if _ts is not None:
                ts_path = os.path.join(pfdir, "_batch_ts.parquet")
                if not ts_path_written:
                    _ts.to_parquet(ts_path)
                    ts_path_written.append(ts_path)
            # Bound total match_compounds concurrency at the flaky server: each worker
            # runs PEAKY_MATCH_WORKERS threads, so keep n_jobs * that modest (~12).
            os.environ.setdefault("PEAKY_MATCH_WORKERS", str(max(2, 12 // n_jobs)))
            import multiprocessing as _mp
            from concurrent.futures import ProcessPoolExecutor, as_completed
            log(f"[assign_batch] parallel: {n_jobs} worker processes "
                f"(match-workers/proc={os.environ['PEAKY_MATCH_WORKERS']}) "
                f"over {len(sample_ids)} samples")
            results: dict = {}
            with ProcessPoolExecutor(
                    max_workers=n_jobs, mp_context=_mp.get_context("spawn"),
                    initializer=_worker_init,
                    initargs=(context, reflists_active, base_kw, ts_path)) as ex:
                futs = {ex.submit(_assign_one, sid): sid for sid in ids}
                for done, fut in enumerate(as_completed(futs), offset + 1):
                    out = fut.result()
                    results[out["sid"]] = out
                    log(f"[assign_batch] ({done}/{len(sample_ids)}) done {out['sid']}")
            # Reduce STRICTLY in sample_ids order (not completion order) so align()'s
            # input is fixed and the merged output is byte-identical to a serial run.
            for sid in ids:
                out = results[sid]
                for ln in out["log"]:              # replay worker logs, grouped per sid
                    log(ln)
                _apply(sid, out["ledger"], out["plausibility_audit"], out["stats"], stage)

    if trace_sample is not None:
        from peaky.batch import tracefirst as TFT
        kw = dict(assign_kw, cfg=copy.deepcopy(cfg), occurrence=trace_sample.occurrence)
        TFT.engine_settings(kw["cfg"], trace_sample, log=log)
        log(f"[assign_batch] (1/1) assigning {trace_sample.sample_id} (offline, "
            f"{len(trace_sample.peaks)} trace peaks) ...")
        res = A.run(trace_sample.sample_id, context=context, log=log,
                    reflists_active=reflists_active, peaks=trace_sample.peaks, **kw)
        led = res["ledger"].merge(
            trace_sample.traces[[c for c in TFT.TRACE_COLS if c in trace_sample.traces.columns]],
            on="peak_id", how="left")
        trace_sample.traces.to_csv(os.path.join(TAB, "traces.csv"), index=False)
        _apply(trace_sample.sample_id, led, res.get("plausibility_audit") or [],
               dict(res.get("stats", {})), STAGE_COVER)
        log(f"[assign_batch] (1/1) done {trace_sample.sample_id}")
    else:
        _assign_files(list(sample_ids), STAGE_COVER, n_jobs)

    from peaky.chem import reagents as _RG
    from peaky.assignment import plausibility as PL
    from peaky.batch import timeseries as _TS
    _rgk = _RG.reagent_for_adducts(list(prof.adducts or []))

    scale = None   # the batch's traces.MassScale, measured at the first merge

    def _merge() -> dict:
        """align() over EVERY per-file ledger so far, then the merged-level
        guards and re-reads, the trace reconciliation and the whole-batch stamp
        -- returned as one record (merged, jitter, merge_gates, trace_info,
        stamp_tol, ts_annot), nothing written. Runs once per stage: the cover's
        record is what the residual universe is read from, and the last call
        (cover + residual files, ONE align) is what the run writes.

        The batch's mass scale (traces.MassScale) is measured ONCE, at the first
        call, from the time series at the traces the per-file anchors label
        (offset-corrected, walked onto their trace, one centre per trace): its
        `merge_ppm` is the gap that still means one ion here, in the known-species
        lock and in the trace-label collapse, its `stamp_ppm` the stamping
        half-window. Both are the binning tolerance when nothing was measured (no
        time series), so that path is the flat-window run exactly; on an Orbitrap
        (0.2-0.3 ppm scatter) both sit on the tolerance floor and nothing moves
        either; a TOF (3-4 ppm) merges at up to 2x the tolerance."""
        nonlocal scale
        if scale is None:
            seeds = [pd.to_numeric(df["mz"], errors="coerce").to_numpy(dtype=float)
                     * (1.0 - float(offsets.get(sid, 0.0) or 0.0) / 1e6)
                     for sid, df in per_file.items() if df is not None and len(df)]
            seeds = np.concatenate(seeds) if seeds else np.empty(0)
            scale = (TR.measure_mass_scale(_idx, seeds, tol_ppm=tol_ppm) if _idx is not None
                     else TR.MassScale(tol_ppm=float(tol_ppm), n_seeds=int(len(seeds))))
            log(f"[scale] {scale.describe()}")
        merged, jitter = align(per_file, tol_ppm=scale.merge_ppm, offsets=offsets,
                               stages=stages if residual else None)
        # Merge guard: drop reagent-cluster ions a per-file pass mislabelled as analyte
        # (urea [R_n+H]+/[R_n+NH4]+ read as CHNO/CH4N2O on the [M+NH4]+/urea channel) --
        # they otherwise dominate the 'assigned' signal. Belt-and-braces with the
        # per-file reagent lock/reclaim (older per-file ledgers predate that fix).
        if _rgk:
            merged, _rgstrip = _RG.strip_reagent_cluster_rows(merged, _rgk, log=log)
        # The merged row's tier_reason (from align: the vote's exemption, else NA)
        # also takes the batch-level gates' notes below (cleanup._note appends to
        # it): a re-read can leave the merged formula different from EVERY per-file
        # reading, and the row itself must say why.
        if "tier_reason" not in merged.columns:
            merged["tier_reason"] = pd.NA
        merge_gates: dict = {}
        # Known species, decided ONCE for the batch by the evidence every file
        # pooled (the pass-0 commits and the leads it left on refused claims):
        # what the vote's curated exemption used to do, by evidence instead of
        # by rank, and written on the row either way.
        merge_gates["known"] = lock_known_species(merged, known_pool, tol_ppm=scale.merge_ppm, log=log)
        if prof.polarity == "+":
            from peaky.assignment import cleanup
            # Hydrocarbon on an N-cluster channel -> [M+H]+ of the N-heterocycle,
            # decided once here from the union of every file's [M+H]+ rows (the
            # per-file stage was deferred above; it runs there instead only when the
            # caller forced reagent_n_relabel=True, and then this pass stands down).
            if not assign_kw.get("reagent_n_relabel"):
                merge_gates["reagent_n"] = cleanup.relabel_reagent_n_adducts(merged, log=log)
            # Re-read uncorroborated [M+NH4]+ adducts as [M+H]+ of the +NH3 amine
            # (mass/isotope-identical; simpler in an N-rich source). Done at the
            # MERGED level where cross-channel corroboration is complete.
            merge_gates["amine"] = cleanup.prefer_amine_over_ammonium(
                merged, ts_peaks=ts_peaks, r_min=amine_r_min,
                protected=protected_neutrals, log=log)
        # Sidelobe-contaminated CHANNELS: an assigned ion whose m/z lands on the ringing
        # sidelobe of a saturating neighbour keeps its formula (the neutral is usually
        # corroborated on another channel) but its HEIGHT is the neighbour's, not the
        # analyte's. Only the time series separates that from a real ion that merely sits
        # near a bright peak, so it is decided HERE, not in per-file cleanup.
        # Called unconditionally so the merged-ledger SCHEMA is stable: without a TS it
        # no-ops and the two columns are still present (all False / NaN).
        _TS.flag_sidelobe_channels(merged, ts_peaks, log=log)
        # Trace-level reconciliation (timeseries.recentre_ledger / collapse_trace_labels):
        # re-centre every merged anchor on its own trace, collapse the rows that
        # converge on one trace, and size the stamping window to the batch's own
        # per-ion scatter. Columns are added on the merged ledger (mz_anchor, mz_trace,
        # trace_offset_ppm, trace_cov_anchor, trace_cov, trace_moved, trace_guarded,
        # trace_id, trace_role); the stamp below reads them. No-op without a TS.
        trace_info: dict = {}
        stamp_tol = scale.stamp_ppm     # = tol_ppm when nothing was measured
        tracks = None
        if _idx is not None and len(merged):
            # the trace question (membership, re-centring) stays at the binning
            # tolerance -- the index's own rule, with its mDa floor; the one-ion
            # question (which rows compete for one trace) is the merge window
            trace_info = _TS.recentre_ledger(merged, index=_idx, tol_ppm=tol_ppm,
                                             rolling=rolling_centre, times_by_code=_hours,
                                             log=log)
            tracks = trace_info.pop("tracks", None)
            trace_info.update(_TS.collapse_trace_labels(merged, tol_ppm=scale.merge_ppm, log=log))
            trace_info.update(stamp_tol_ppm=float(stamp_tol),
                              sigma_ppm=round(float(scale.sigma_ppm), 3) if scale.measured else None)
            log(f"[traces] stamping window +-{stamp_tol:g} ppm from the batch's mass scale "
                f"(scatter {scale.sigma_ppm:.3f} ppm; merge window {scale.merge_ppm:g}, "
                f"binning tolerance {tol_ppm:g})" if scale.measured else
                f"[traces] stamping window +-{stamp_tol:g} ppm = the binning tolerance "
                f"(mass scatter not measured)")
            if rolling_centre:
                # the per-TRACE window: each row's own post-centring residual,
                # the batch window where a row has none
                merged["stamp_tol_ppm"] = _TS.stamp_tolerances(merged, tol_ppm=tol_ppm,
                                                                fallback=stamp_tol, floor=stamp_tol)
                _pt = merged["stamp_tol_ppm"]
                trace_info["stamp_tol_per_trace"] = {
                    "median_ppm": float(_pt.median()), "min_ppm": float(_pt.min()),
                    "max_ppm": float(_pt.max()),
                    "n_wider_than_batch": int((_pt > stamp_tol + 1e-9).sum()),
                    "n_tighter_than_batch": int((_pt < stamp_tol - 1e-9).sum())}
                log(f"[traces] per-trace stamping windows: median "
                    f"{trace_info['stamp_tol_per_trace']['median_ppm']:.2f} ppm "
                    f"({trace_info['stamp_tol_per_trace']['min_ppm']:.2f}-"
                    f"{trace_info['stamp_tol_per_trace']['max_ppm']:.2f}); "
                    f"{len(tracks or {})} rows stamp along a rolling track")
        out = {"merged": merged, "jitter": jitter, "merge_gates": merge_gates,
               "trace_info": trace_info, "stamp_tol": stamp_tol, "ts_annot": None,
               "predicted_rows": {}, "predicted_tracks": None}
        # Stamp the batch time-series peaks with their assigned formula/channel.
        # Downstream time-series analysis then has neutral_formula / adduct / tier /
        # ion_mz per peak, not just m/z. No-op when ts_peaks is unavailable.
        if ts_peaks is not None:
            # union stamping frame: merged analytes + every identified NON-analyte
            # ion (reagent ladder / isotope satellites / ringing artifacts) from the
            # per-file ledgers -- so `ion_formula` marks every KNOWN ion, analyte or
            # not, and only true unknowns stay blank (in an iodide spectrum the 10
            # reagent tracks alone are ~77% of total signal).
            _aux = (pd.concat(identified_aux, ignore_index=True)
                    if identified_aux else None)
            # ... plus the PREDICTED diagnostic isotope satellites (13C / 81Br /
            # 37Cl / 15N / 34S / 29Si / 30Si / 18O) of every merged M0 with a known
            # ion formula that no per-file ledger claimed -- the faint 15N / 18O
            # lines sit below the picker's edge in most files, so they were
            # missing from the stamp even with the parent Assigned everywhere, and
            # surfaced as unexplained tracks wherever a plume lifted them. Observed
            # rows beat predicted ones, an M0 always beats a predicted line on its
            # track, and annotate_peaks stamps a predicted line only where the
            # parent's same-sample height licenses it (the per-file passes'
            # 0.3-3.5 window) and only on lines that pass coherently across the
            # batch. The per-file ledgers and their coverage figures are untouched.
            # A track explained this way carries an ion_formula, so the residual
            # stage (which reads the cover's stamp) no longer targets it.
            _stamp = _TS.stamping_frame(merged, _aux, tol_ppm=stamp_tol)
            _stats: dict = {}
            out["ts_annot"] = _TS.annotate_peaks(ts_peaks, _stamp, tol_ppm=stamp_tol,
                                                 stats=_stats, tracks=tracks)
            out["predicted_rows"] = dict(_stamp.attrs.get("predicted_satellites") or {})
            out["predicted_tracks"] = _stats.get("predicted_tracks")
        return out

    res_m = _merge()

    # ---- the residual stage (module note) ---------------------------------------
    residual_meta: dict = {}
    rsel = None
    n_cover = len(sample_ids)
    if residual:
        log("[phase] residual")
        if ts_peaks is None:
            residual_meta = {"n_bins_residual": 0, "k": 0, "coverage_of_residual": 0.0,
                             "stop_reason": SS.STOP_EMPTY, "sample_ids": [],
                             "skipped": "no batch time series to read the residual from"}
            log("[residual] skipped: no batch time series to read the residual from")
        else:
            _annot = res_m["ts_annot"]
            stamped = _annot["ion_formula"].notna().to_numpy() if _annot is not None else None
            # THE FLOOR. Edge-relative by default, and never below the run's own
            # gate multiple: a bin that no file would admit is not worth a file.
            # An absolute --residual-min-cps replaces the multiple; an absolute
            # gate (cfg.height_cutoff_cps) is a floor as well.
            gate_x = float(_cfg.height_cutoff_x_edge_resolved)
            gate_cps = _cfg.height_cutoff_cps
            if residual_min_cps is not None:
                min_x, min_cps = None, float(residual_min_cps)
                floor_src = f"an absolute floor of {min_cps:g} cps (residual_min_cps)"
            else:
                min_x, min_cps = float(residual_min_x_edge), None
                floor_src = f"{min_x:g}x the sample's noise edge (residual_min_x_edge)"
                if gate_cps is None and gate_x > min_x:
                    min_x = gate_x
                    floor_src = (f"{min_x:g}x the sample's noise edge -- raised from "
                                 f"{float(residual_min_x_edge):g}x to the run's admission gate")
            if gate_cps is not None:
                min_cps = max(float(gate_cps), min_cps if min_cps is not None else 0.0)
                floor_src += f"; the absolute admission gate of {float(gate_cps):g} cps"
            bins = SS.residual_universe(ts_peaks, assigned=sample_ids, stamped=stamped,
                                        min_x_edge=min_x, min_cps=min_cps,
                                        min_prevalence=min_prevalence, tol_ppm=tol_ppm)
            umeta = dict(bins.attrs.get("residual", {}))
            if "bin_mz" in bins.columns:
                residual_scope[:] = [np.sort(pd.to_numeric(bins["bin_mz"], errors="coerce")
                                             .dropna().to_numpy(dtype=float))]
            # a sample counts for a bin only where its OWN gate would admit it:
            # the multiple x that sample's edge, or the absolute gate everywhere
            edge = pd.Series(bins.attrs.get("edge_cps") or {}, dtype=float)
            min_height = (pd.Series(float(gate_cps), index=edge.index) if gate_cps is not None
                          else edge * gate_x)
            rsel = SS.select_residual_cover(ts_peaks, bins, frac_of_max=residual_frac_of_max,
                                            k_max=residual_k_max, min_height=min_height,
                                            tol_ppm=tol_ppm)
            smeta = dict(rsel.attrs.get("selection", {}))
            log(f"[assign_batch] {SS.describe_residual(smeta, umeta)}")
            residual_ids = rsel["sample_item_id"].tolist()
            residual_meta = {
                "n_bins_residual": int(umeta.get("n_residual", 0)),
                "n_universe": int(umeta.get("n_universe", 0)),
                "n_uncovered": int(umeta.get("n_uncovered", 0)),
                "n_explained": int(umeta.get("n_explained", 0)),
                "n_below_floor": int(umeta.get("n_below_floor", 0)),
                "n_sidelobe": int(umeta.get("n_sidelobe", 0)),
                "n_suspect": int(umeta.get("n_suspect", 0)),
                "floor": {"min_x_edge": min_x, "min_cps": min_cps,
                          "edge_median_cps": umeta.get("edge_median_cps"),
                          "source": floor_src},
                "frac_of_max": float(residual_frac_of_max),
                "k": int(smeta.get("k", 0)), "k_max": int(residual_k_max),
                "coverage_of_residual": float(smeta.get("achieved_coverage", 0.0)),
                "stop_reason": smeta.get("stop_reason", SS.STOP_EMPTY),
                "next_gain": float(smeta.get("next_gain", 0.0)),
                "sample_ids": residual_ids,
            }
            # the targeted bins, each with the pick that carries it (empty = none
            # did), then the confirmed sidelobes the stage dropped (tier 'sidelobe')
            cb = rsel.attrs.get("covered_by", {})
            rb = bins.copy()
            rb["covered_by"] = [cb.get(int(b_), "") for b_ in rb["bin"]] if len(rb) else []
            _sl = bins.attrs.get("sidelobes") or []
            if _sl:
                rb = pd.concat([rb, pd.DataFrame(_sl).assign(covered_by="")],
                               ignore_index=True)
            rb.to_csv(os.path.join(TAB, "residual_bins.csv"), index=False)
            if residual_ids:
                if sel is not None:
                    # our own cover table gains the picks, numbered on from the cover
                    extra = rsel.copy()
                    extra["pick"] = np.arange(len(sel) + 1, len(sel) + 1 + len(extra))
                    pd.concat([sel, extra], ignore_index=True).to_csv(
                        os.path.join(TAB, "selected_samples.csv"), index=False)
                log("[phase] assign")
                log(f"[assign_batch] residual stage: {len(residual_ids)} more file(s) "
                    f"-> {pfdir}")
                sample_ids = list(sample_ids) + residual_ids
                _assign_files(residual_ids, STAGE_RESIDUAL, min(n_jobs, len(residual_ids)))
                if scope_counts:
                    residual_meta["trace_first_scope"] = {
                        "rows_kept": int(sum(k for k, _ in scope_counts.values())),
                        "rows_total": int(sum(t for _, t in scope_counts.values())),
                        "per_file": {sid: {"kept": k, "total": t} for sid, (k, t) in scope_counts.items()}}
                    log(f"[assign_batch] trace-first scope: residual files contribute "
                        f"{residual_meta['trace_first_scope']['rows_kept']} of "
                        f"{residual_meta['trace_first_scope']['rows_total']} M0 rows (on residual bins)")
                res_m = _merge()          # ONE align over cover + residual files

    # ---- write the run ------------------------------------------------------------
    merged, jitter, merge_gates = res_m["merged"], res_m["jitter"], res_m["merge_gates"]
    trace_info, stamp_tol, ts_annot = res_m["trace_info"], res_m["stamp_tol"], res_m["ts_annot"]
    summary_plaus = {}
    # one audit row per touched peak (per-file O/C-monster + carbon-cluster demotes);
    # always written for a stable artifact set.
    n_audit = PL.write_audit(plaus_audit, os.path.join(TAB, f"plausibility_audit_{prof.name}.csv"))
    log(f"[assign_batch] plausibility audit: {n_audit} touched peaks "
        f"-> tables/plausibility_audit_{prof.name}.csv")
    # evidence levels: recomputed on the POOLED per-file ledgers (cover + residual
    # files as ONE source -- chan2 sees a second adduct in any file, tied/lowconf
    # need all rows across files) and stamped on the merged ledger by ion. The
    # merged rows carry none of the predicate columns, so they are never read for
    # this; a merged row whose reading no per-file ledger holds (a batch-level
    # re-read) stays NA, and the count says so.
    levels = EV.level_pooled(level_frames, cross=cross)
    merged = EV.stamp_merged(merged, levels)
    levels.to_csv(os.path.join(TAB, "evidence_levels.csv"), index=False)
    ev_summary = {
        "pooled": EV.summarize(levels["evidence_level"]) if len(levels) else {},
        "merged": EV.summarize(merged["evidence_level"]) if len(merged) else {},
        "per_stage": ({str(s_): EV.summarize(merged.loc[merged["stage"] == s_, "evidence_level"])
                       for s_ in sorted(merged["stage"].dropna().astype(str).unique())}
                      if len(merged) and "stage" in merged.columns else {}),
        "n_pairs": int(len(levels)),
        "n_unstamped": int(merged["evidence_level"].isna().sum()) if len(merged) else 0,
        "n_corroborate": int(len(cross)), "cross_source": cross_sources,
    }
    log(f"[assign_batch] evidence levels over {len(level_frames)} pooled file(s): "
        f"{ev_summary['pooled']} ({ev_summary['n_pairs']} neutral/adduct pairs); "
        f"{ev_summary['n_unstamped']} merged row(s) without a per-file reading "
        f"-> tables/evidence_levels.csv")
    # ion-only rows (the `ion_only` stage): merged rows carrying the link, the
    # per-file rows behind them, and the files that hold any -- so the bucket
    # is on record beside the tiers it is deliberately kept apart from
    _io_merged = int(merged["ion_only_of"].notna().sum()) if len(merged) and "ion_only_of" in merged.columns else 0
    _io_files = {sid: int(EV.is_ion_only(fr).sum()) for sid, fr in level_frames.items()}
    ion_only_summary = {
        "channels": list(getattr(cfg, "ion_only_channels", None) or ()),
        "merged": _io_merged,
        "per_file_rows": int(sum(_io_files.values())),
        "n_files_with": int(sum(1 for v in _io_files.values() if v)),
        "merged_levels": (EV.summarize(merged.loc[merged["ion_only_of"].notna(), "evidence_level"])
                          if _io_merged else {}),
    }
    if ion_only_summary["channels"]:
        log(f"[assign_batch] ion-only rows: {_io_merged} merged ({ion_only_summary['per_file_rows']} "
            f"per-file rows in {ion_only_summary['n_files_with']} of {len(_io_files)} files) on "
            f"{ion_only_summary['channels']}")
    merged.to_csv(os.path.join(out_dir, "merged_ledger.csv"), index=False)
    jitter.to_csv(os.path.join(TAB, "jitter.csv"), index=False)
    # the FINAL per_file/_batch_ts.parquet (in parallel mode this overwrites the raw
    # worker-transfer copy)
    if ts_annot is not None:
        ts_annot.to_parquet(os.path.join(pfdir, "_batch_ts.parquet"))
        _n_ass = int(ts_annot["neutral_formula"].notna().sum())
        _n_ion = int(ts_annot["ion_formula"].notna().sum())
        log(f"[assign_batch] _batch_ts.parquet: {len(ts_annot)} peaks, {_n_ass} "
            f"({_n_ass / max(len(ts_annot), 1):.0%}) matched to an assigned "
            f"formula/channel, {_n_ion} ({_n_ion / max(len(ts_annot), 1):.0%}) "
            f"to a known ion incl. reagent/isotope (window +-{stamp_tol:g} ppm around "
            f"each ion's trace centre)")
        # one-to-one guarantee: each (sample, ion) is stamped on at most ONE peak,
        # so a downstream groupby(formula, adduct) sees one trace per sample. The
        # shoulder/split peaks that lost are kept, unstamped, flagged dup_candidate.
        _n_dup = int(ts_annot["dup_candidate"].sum())
        if _n_dup:
            _ions = ts_annot.loc[ts_annot["dup_candidate"], "mz"].round(3).nunique()
            log(f"[assign_batch] one-to-one: {_n_dup} near-duplicate peak(s) at "
                f"~{_ions} m/z left unstamped (flagged dup_candidate) so no ion is "
                f"stamped twice in one sample")
        # what explained the stamped peaks: analytes, per-file-observed ions and
        # predicted satellites are counted apart, so the predicted share is
        # auditable and the observed figures compare with earlier runs; plus one
        # audit row per predicted line that had a candidate peak (samples judged /
        # passed, whether the track was kept, peaks stamped), always written for a
        # stable artifact set. These figures describe the FINAL stamp (cover +
        # residual files); the residual stage read the cover's.
        _tracks = res_m.get("predicted_tracks")
        if _tracks is None:
            _tracks = pd.DataFrame(columns=_TS.PRED_TRACK_COLS)
        _tracks.to_csv(os.path.join(TAB, "predicted_satellites.csv"), index=False)
        _role = ts_annot["role"].astype(object)
        _src = ts_annot["stamp_source"].astype(object)
        _pred_rows = dict(res_m.get("predicted_rows") or {})
        _judged = _tracks["judged"].astype(bool) if len(_tracks) else pd.Series(dtype=bool)
        _kept = _tracks["kept"].astype(bool) if len(_tracks) else pd.Series(dtype=bool)
        _rej = _tracks[_judged & ~_kept] if len(_tracks) else _tracks
        trace_info["stamp"] = {
            "n_peaks": int(len(ts_annot)),
            "n_M0": int((_role == "M0").sum()),
            "n_reagent": int((_role == "reagent").sum()),
            "n_artifact": int((_role == "artifact").sum()),
            "n_iso_observed": int(((_role == "iso_child") & (_src == "observed")).sum()),
            "n_iso_predicted": int((_src == "predicted").sum()),
            "n_dup_candidate": int(ts_annot["dup_candidate"].sum()),
            # the predicted lines as TRACKS: how many carry a stamp, how many the
            # coherence rule could judge, how many it rejected, and the
            # per-sample passes those rejections discarded
            "n_tracks_predicted_stamped": int((_tracks["n_stamped"] > 0).sum()) if len(_tracks) else 0,
            "n_tracks_judged": int(_judged.sum()),
            "n_tracks_rejected": int(len(_rej)),
            "n_samples_rejected": int(_rej["n_pass"].sum()) if len(_rej) else 0,
            "track_min_n": _TS.PRED_TRACK_MIN_N,
            "track_min_share": _TS.PRED_TRACK_MIN_SHARE,
            "predicted_rows": _pred_rows,
        }
        _st = trace_info["stamp"]
        log(f"[assign_batch] satellites: {_st['n_iso_observed']} peak(s) on tracks the "
            f"per-file ledgers claimed, {_st['n_iso_predicted']} on PREDICTED lines "
            f"({_st['n_tracks_predicted_stamped']} tracks; {_pred_rows.get('n_predicted', 0)} "
            f"predicted rows for {_pred_rows.get('n_parents', 0)} parents; "
            f"{_pred_rows.get('n_superseded_observed', 0)} superseded by an observed "
            f"satellite, {_pred_rows.get('n_superseded_track', 0)} by a known track); "
            f"coherence rule: {_st['n_tracks_rejected']} of {_st['n_tracks_judged']} judged "
            f"track(s) rejected, {_st['n_samples_rejected']} per-sample pass(es) discarded "
            f"-> tables/predicted_satellites.csv")

    if residual:
        # the record of the second stage, beside the cover's (`selection` is our
        # own dict: a caller's selection_meta was copied above)
        selection["residual"] = residual_meta
    summary = {
        "reagent": prof.name, "label": prof.label, "context": context,
        "batch_name": batch,
        "selection": selection,
        "trace_first": trace_sample.summary() if trace_sample is not None else None,
        "admission": occ_info,
        # the batch-derived brightness floor (empty when the multiple was pinned
        # by a flag / cfg / profile, or could not be derived): the transient share
        # per grid multiple and the one chosen
        "gate": gate_info,
        # the trace reconciliation of the merged ledger (empty without a TS)
        "traces": trace_info,
        "n_files": len(sample_ids), "sample_ids": sample_ids,
        # the gate actually used, and where the multiple came from (a
        # profile-supplied value reads differently from the package default);
        # the resolved cps gate per file is in per_file[].height_gate_cps.
        "height_cutoff_x_edge": x_edge,
        "height_cutoff_x_edge_source": x_edge_source,
        "tol_ppm": tol_ppm, "offsets_ppm": offsets,
        "pattern_scoring": scorings,
        # the batch's mass scale (traces.MassScale): the measured per-ion scatter
        # and the merge / stamping windows sized from it (tol_ppm above is the
        # BINNING tolerance and the floor of both; unmeasured, both equal it)
        "mass_scale": (scale if scale is not None
                       else TR.MassScale(tol_ppm=float(tol_ppm))).as_dict(),
        "merged_M0": int(len(merged)),
        # the width model the per-file resolvability stamp used (None = not stamped)
        # and the per-file M0 class counts summed over the files
        "resolution": rp.as_dict() if rp is not None else None,
        "resolvability": _sum_counts(x.get("resolvability") for x in per_stats),
        "merged_tiers": merged["tier"].value_counts().to_dict() if len(merged) else {},
        "n_in_all_files": int((merged["n_files"] == len(sample_ids)).sum()) if len(merged) else 0,
        "n_single_file": int((merged["n_files"] == 1).sum()) if len(merged) else 0,
        "formula_disagreements": int((~merged["formula_agree"]).sum()) if len(merged) else 0,
        # ... of which clusters where the files named DIFFERENT IONS (the rest are
        # two labels of one ion, e.g. the reagent-N isobar)
        "ion_disagreements": int((~merged["ion_agree"]).sum()) if len(merged) else 0,
        # the batch-level re-reads applied to the merged ledger (positive mode):
        # what the reagent-N pass and the ammonium/amine gate each did, so the
        # counts are on record and not only in the log
        "merge_gates": merge_gates,
        "plausibility": summary_plaus,
        "plausibility_audit_rows": n_audit,
        # the evidence levels (docs/EVIDENCE_LEVELS.md): pooled = one count per
        # (neutral, adduct) pair over the pooled files; merged = per merged row;
        # per_stage = merged rows by cover / residual; cross_source = what
        # corroborated the run
        "evidence_levels": ev_summary,
        # the ion-only bucket (docs/EVIDENCE_LEVELS.md §3, the `ion_only` stage):
        # channels opened, merged rows carrying an `ion_only_of` link, per-file
        # rows behind them, files holding any, and their levels (4d / 5a)
        "ion_only": ion_only_summary,
        "reflists_active": RL.active_versions(reflists_active),   # [(id, data_version)]
        "per_file": per_stats,
        # RUN-TIME metadata, not material data: how long the assignment actually
        # took, alongside the n_jobs that produced it (a duration is meaningless
        # without it). Safe to keep here -- batch_summary.json is a counts/offsets
        # file and is NOT part of the reproducibility fingerprint, which hashes
        # merged_ledger.csv and the input TS (see reporting/provenance.py).
        "elapsed_s": round(time.time() - t_start, 1),
        "n_jobs": n_jobs,
    }
    if residual:
        # files per stage, and the ions each stage brought in (`stage` column)
        summary["n_files_by_stage"] = {STAGE_COVER: int(n_cover),
                                       STAGE_RESIDUAL: int(len(sample_ids) - n_cover)}
        summary["merged_by_stage"] = (merged["stage"].value_counts().to_dict()
                                      if len(merged) and "stage" in merged.columns else {})
    with open(os.path.join(out_dir, "batch_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    log(f"[assign_batch] DONE: {summary['merged_M0']} merged M0 "
        f"({summary['merged_tiers']}); {summary['n_in_all_files']} in all files, "
        f"{summary['n_single_file']} single-file, "
        f"{summary['formula_disagreements']} formula disagreements")
    # the DONE line above is parsed by the progress panel (progress.RE_ASSIGN_DONE)
    # and pinned by tests/test_progress.py, so the ion split goes on its own line
    log(f"[assign_batch] disagreements: {summary['ion_disagreements']} between different "
        f"ions, {summary['formula_disagreements'] - summary['ion_disagreements']} two "
        f"labels of one ion (see ion_agree / alternatives on the merged ledger)")
    if residual:
        # ... and so does the residual stage's yield
        log(f"[assign_batch] residual stage: {summary['n_files_by_stage'][STAGE_RESIDUAL]} "
            f"file(s), {summary['merged_by_stage'].get(STAGE_RESIDUAL, 0)} ion(s) it alone holds")
    log(f"[assign_batch] assigned {len(sample_ids)} samples in "
        f"{summary['elapsed_s']:.1f}s (n_jobs={n_jobs})")
    return {"profile": prof, "context": context, "sample_ids": sample_ids,
            "per_file": per_file, "offsets": offsets, "merged": merged,
            "jitter": jitter, "summary": summary, "out_dir": out_dir,
            "evidence": levels,
            "residual_samples": rsel, "stages": dict(stages)}
