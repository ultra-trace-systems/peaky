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
CLASS wins -- each ion takes the best per-file vote class of its readings
(neutral backed > formula confirmed > unconfirmed; `evidence.vote_classes`,
private to the vote and never a user-facing level) -- and among ions of one class the ion carried by the
most files (tier and ion_score break ties); among that ion's labels -- the same
ion read as C13H14O4 [M+NH4]+ or as C13H17NO4 [M+H]+ -- the one Assigned in the
most files wins, because on such a pair Assigned means a discriminating channel
was present and Candidate means the file had nothing to decide with. The losing
readings stay on the merged row (`alternatives`, `n_files_ion`, `n_files_winner`,
`ion_agree`) as well as in jitter.csv (with each file's own vote class).
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
from peaky.batch import iso_checks as _IC
from peaky.batch import label_twins as _LT
from peaky.batch import neutral_pairs as _NP
from peaky.batch import sampling as SS

__version__ = "0.10.0"  # the vote reads the per-file EVIDENCE: a cluster's ions are
                        # ranked by the best vote class of their readings before the
                        # file count (evidence.vote_classes, computed in the parent and
                        # carried as `vote_class`; jitter.csv carries the class); the
                        # merged ledger is levelled on the evidence scale of the release
                        # (evidence.level_batch on the pooled per-file ledgers)
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
            "ts_disposition", "ts_cv_norm",  # the winner file's time-series label (batch/timeseries.py): flatness is a
                                             # label, never a tier, so the merged row must carry it, when present
            "vote_class"]   # the reading's vote class (0 / 1 / 2, evidence.vote_classes; computed in the parent,
                            # never written to a ledger): `_vote` ranks ions by it before the count, when present

# THE VOTE'S EVIDENCE CLASS of one per-file reading (the key after the ion-only
# rule and before the file count in `_vote`): 2 = neutral backed (the formula
# confirmed and the neutral backed by two axes or a corroborating source -- the
# `--corroborate` source holds the neutral on its own evidence), 1 = formula
# confirmed, 0 = unconfirmed (exact mass alone, or the reading argues with
# itself). It is computed per file in the parent (`_apply`: evidence.vote_classes
# over the file's ledger, the --corroborate cross set and the batch's width
# model) by the decision the vote has always read, kept PRIVATE to it: the
# evidence scale of the release is a reader's grade of the committed formula and
# never moves a reading. A reading of the first class outranks one of the second
# whatever the file count, and one of the second outranks an unconfirmed one;
# the count decides among equals. Measured on a 28-file TOF batch merged at its
# own 12 ppm window: with one ion's readings finally in one row, the count alone
# handed the peak of the Orbitrap-confirmed acid C9H16O6 [M+NO3]- (neutral
# backed, 2 files) to C14H21N [M+Br]- (unconfirmed, 9 files), and the roster's
# pinic acid C9H14O4 (corroborated, 1 file) lost a 1-vs-1 tie on ion_score to a
# silicon formula.
VOTE_CLASS = "vote_class"
_OWN_VOTE_CLASS = "__own_vote_class"   # a ledger's vote class carried through a merge (trace-first), never written


def _evidence_class(level, axes) -> int:
    """The vote class of one reading from the private decision's (level, axes)
    -- evidence._vote_class_of: 2 for an established neutral or the
    --corroborate source's agreement, 1 for a confirmed formula / ion, else 0."""
    from peaky.assignment import evidence as _EV
    return _EV._vote_class_of(level, axes)


def _class_text(cls) -> str:
    """'neutral backed' / 'formula confirmed' / 'unconfirmed': what a vote note
    names (evidence.VOTE_CLASS_TEXT; never a level)."""
    from peaky.assignment import evidence as _EV
    try:
        return _EV.VOTE_CLASS_TEXT.get(int(cls), "unconfirmed")
    except (TypeError, ValueError):
        return "unconfirmed"


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
    best_cls), best first; `labels` one row per
    (neutral_formula, adduct) reading of EVERY ion (_ion, _nf, _ad, n_files,
    n_assigned, best_ion, ...), the winning ion's readings ranked best first
    and listed first, the other ions' readings after them in ion order.

    1. WHICH ION sits at this m/z is what files can genuinely disagree on, and
       the EVIDENCE decides it before the count: each ion takes the best
       vote class of its per-file readings (the `vote_class` column,
       evidence.vote_classes -- neutral backed, above formula confirmed, above
       unconfirmed; every reading is class 0 when the frames carry no class,
       and the vote is then the pure count it was), and the ion of the best class wins; among
       ions of one class the ion carried by the most FILES wins, the number of
       files carrying it at Assigned tier and the best ion_score only break
       ties, and the ion's own text is the last key -- so a full tie resolves
       the same way whatever order the files arrived in (serial and parallel
       runs stay byte-identical). The count-first order is the one
       collapse_trace_labels applies to competing labels on one trace.

       Why the evidence first: once the merge window put one TOF ion's readings
       in one row (they used to sit in rows of their own, 8-12 ppm apart, each
       looking unanimous), a 4-file mass-only reading (5b) outvoted a 1-file
       reading with an acid-branch corroboration (neutral backed) at the same peak, and the
       board lost the Orbitrap-confirmed HOMs and the roster's pinic acid to
       bromide adducts of N-compounds read in more files. A per-file class is
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
    # the vote class of every per-file reading (0 everywhere when the frames
    # carry none: a pure `align()` caller without the parent's class)
    if VOTE_CLASS in g.columns:
        cls = pd.to_numeric(g[VOTE_CLASS], errors="coerce").fillna(0).astype(int)
    else:
        cls = pd.Series(0, index=g.index)
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
                  _cls=cls)
    lab = (gg.groupby(["_ion", "_nf", "_ad"], sort=True)  # text order = last key
             .agg(n_files=("src", "nunique"),
                  n_assigned=("_asrc", "nunique"),        # FILES at Assigned, not rows
                  best_ion=("ion_score", "max"), regular=("_reg", "max"),
                  best_cls=("_cls", "max"))
             .reset_index())
    ions = (gg.groupby("_ion", sort=True)
              .agg(n_files=("src", "nunique"), n_assigned=("_asrc", "nunique"),
                   best_ion=("ion_score", "max"), regular=("_reg", "max"),
                   best_cls=("_cls", "max"))
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

    per_file : {src -> DataFrame with _M0_COLS; `vote_class` is each reading's
    vote class (evidence.vote_classes) -- without it every reading is class 0
    and the vote is the pure count}. offsets : {src -> median ppm}
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
              something to explain: an ion chosen by its vote class over a
              lower-class ion carried by at least as many files, a label chosen
              by corroboration over a bigger count, or a regular reading kept
              over an ion-only one carried by more files); srcs[, stage],
              mz_jitter_ppm_raw, mz_jitter_ppm_caldj.
      jitter  long form, one row per (cluster, file): cluster, src, mz,
              neutral_formula, adduct, tier, ion_score, vote_class -- every
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
                                      "resolvability", "sep_hwhm", "ts_disposition", "ts_cv_norm",
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
                             f"({_class_text(win_ion['best_cls'])} in "
                             f"{int(win_ion['n_files'])} of {n_total} files) over the "
                             f"{int(big['n_files'])}-file {big_lab['_nf']} {big_lab['_ad']} "
                             f"({_class_text(big['best_cls'])})")
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
            ts_disposition=best.get("ts_disposition", pd.NA), ts_cv_norm=best.get("ts_cv_norm", np.nan),
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
                                    vote_class=(int(r[VOTE_CLASS]) if VOTE_CLASS in r.index
                                                and pd.notna(r[VOTE_CLASS]) else pd.NA)))
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
    from peaky.assignment.mass_only import KNOWN_DECIDED
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
        # every confirming file holds the reading at Candidate (its own tier rules:
        # the lock score floor, the counting-detector floor) -- say so on the row
        cand_note = ("; Candidate: no confirming file holds it at Assigned (see the "
                     "per-file tier reasons)"
                     if conf and all(str(r.get("tier")) != TIER_ASSIGNED for r in conf) else "")
        old_nf, old_ad = _s(merged.at[i, "neutral_formula"]), _s(merged.at[i, "adduct"])
        old_n = (int(merged.at[i, "n_files_winner"])
                 if "n_files_winner" in merged.columns and pd.notna(merged.at[i, "n_files_winner"])
                 else 0)
        # the note's opening words are what the TOF mass-only flag reads as
        # "the batch decided this known species" (an exempt row)
        head = (f"{KNOWN_DECIDED}: {label} ({nf} {ad}) -- "
                f"{_pool_summary(conf, dfr, ref)}")
        if conf and not ref:
            if same:
                capped = (not mass_only and not two_lines and "tier" in merged.columns
                          and _s(merged.at[i, "tier"]) == TIER_ASSIGNED)
                if capped:
                    merged.at[i, "tier"] = TIER_CANDIDATE
                _note(merged, i, head + (weak_note if capped else "")
                      + (cand_note if not capped and "tier" in merged.columns
                         and _s(merged.at[i, "tier"]) == TIER_CANDIDATE else ""))
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
                             f"(vote {n_in} of {n_total} files)"
                             + ((cand_note if _s(merged.at[i, "tier"]) == TIER_CANDIDATE else "")
                                if two_lines else weak_note))
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


def batch_noise_edge(client, sample_ids, *, edges=None) -> float | None:
    """The batch's typical detection edge: the median of its files' own
    (`passes.noise_edge`, the 1st percentile of each file's picked heights),
    from their cached peak tables (`edges` given: those numbers, for tests and
    offline callers). None when no file has one. The tier pass's
    counting-detector floor is sized from it (tiers.tof_assign_floor, C46)."""
    from peaky.assignment import passes as PA
    from peaky.io import io_mascope as IO

    vals = []
    if edges is None:
        edges = []
        for sid in sample_ids:
            try:
                pk = IO.fetch_peaks(client, sid, use_cache=True)
                if "peak_id" in pk.columns:
                    pk = pk.drop_duplicates("peak_id")   # the raw table has one row per match
                edges.append(PA.noise_edge(pk["height"]))
            except Exception:            # noqa: BLE001 -- a file with no peaks has no edge
                edges.append(None)
    for e in edges:
        try:
            e = float(e) if e is not None else None
        except (TypeError, ValueError):
            e = None
        if e is not None and np.isfinite(e) and e > 0:
            vals.append(e)
    return float(np.median(vals)) if vals else None


def _worker_init(context, reflists_active, base_kw, ts_path, reagents=None, axis=None):
    global _W
    from peaky.io import io_mascope as IO
    # A spawned worker imports the reagent registry afresh -- built-ins only. The
    # parent's added profiles (--reagent-config, register()) ride in as
    # `reagents` (profiles.registry_extras) and are registered here, before any
    # per-file stage resolves the run's reagent by name (the evidence space
    # does), or a config-only profile is an unknown reagent in every worker.
    P.register_extras(reagents)
    # Likewise the batch's m/z-axis correction (`axis` = (sample ids, wave
    # record), see measure_axis): it lives in the parent's io registry, and a
    # worker that did not get it would assign every file on the raw axis.
    if axis:
        IO.set_axis_correction(*axis)
    else:
        IO.clear_axis_correction()
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
            "stats": dict(res.get("stats", {})), "log": lines,
            # what the worker scored at: a mass trend its calibration set lives
            # in the worker's process only (C42), so the parent cannot re-read it
            "pattern_scoring": res.get("pattern_scoring")}


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


#: `--mass-axis`: 'auto' models the batch's m/z axis from its own peaks (batch.axislock:
#: calibration-free locks, a spline and the axis's steps) on an Orbitrap, falling back to
#: the reagent's reference ions (batch.massqc) when the locks cannot build a model;
#: 'locks' / 'reference' force one of the two; 'off' skips the step
MASS_AXIS_MODES = ("auto", "locks", "reference", "off")
#: the largest correction the LOCK model applies. Its corrections are evidence-backed
#: by hundreds of locks, so it may exceed massqc.MAX_CORRECTION_PPM (sized for a wave
#: fitted on ~25 reference ions); the bound left is the PDF report's: its coverage
#: tables match the ledger to the run's recorded, uncorrected series within 8 ppm
LOCK_MAX_CORRECTION_PPM = 7.0
#: the lock model is predictive when it predicts held-out locks to within this (ppm), or
#: half their raw error: a raw median alone read a calibrated body with a stepped top as
#: 'unpredictive' (raw 0.02 ppm) and left 2-3.5 ppm steps uncorrected
LOCK_CV_OK_PPM = 0.3


def _roster_class(table) -> str | None:
    """'orbitrap' / 'tof' from a roster's `instrument_type` (Mascope's 'orbi' /
    'tof'), or None -- the fallback when no peak-width model was measured."""
    if table is None or "instrument_type" not in getattr(table, "columns", ()):
        return None
    kinds = {str(k).lower() for k in table["instrument_type"].dropna().unique()}
    if kinds == {"orbi"}:
        return "orbitrap"
    if kinds and kinds <= {"tof", "api"}:
        return "tof"
    return None


#: the column a corrected time series carries: the ppm removed from each peak's
#: m/z (0 where the correction does not reach). A series fed back in (`--ts` on a
#: run's per_file/_batch_ts.parquet) is restored to the server's axis first
#: (`restore_axis`), so its files' peak tables and its series sit on one axis.
AXIS_COL = "mz_axis_ppm"


def restore_axis(ts, *, log=print):
    """`ts` on the server's m/z axis: an earlier run's correction (AXIS_COL)
    undone, the column dropped. Unchanged when it carries none."""
    if ts is None or AXIS_COL not in getattr(ts, "columns", ()):
        return ts
    d = pd.to_numeric(ts[AXIS_COL], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    out = ts.drop(columns=[AXIS_COL])
    if np.any(d != 0):
        out = out.copy()
        out["mz"] = out["mz"].to_numpy(dtype=float) / (1 - d * 1e-6)
        log(f"[mass-axis] the time series carries an earlier run's axis correction "
            f"({int((d != 0).sum())} peaks): restored to the server's axis first")
    return out


def _axis_class(measured: str | None, roster: str | None) -> str | None:
    """The class the axis step acts on: 'tof' when EITHER the width model or the
    roster says TOF -- a TOF declared or measured at R >= the Orbitrap bar
    (--resolving-power 60000) read as an Orbitrap and was corrected -- else the
    width model's class, else the roster's."""
    if "tof" in (measured, roster):
        return "tof"
    return measured or roster


def to_server_axis(mz, axis_info):
    """Corrected m/z back on the server's axis (massqc.invert_correction: segment by
    segment for a lock model). `axis_info` is a run's batch_summary['mass_axis'];
    unchanged when it applied nothing. A batch's merged ledger is matched to the
    server's batch peaks by m/z (`peaky publish --batch`), so it must travel on the
    server's axis."""
    from peaky.batch import massqc as MQ
    mz = np.asarray(mz, dtype=float)
    if not axis_info or not axis_info.get("applied") or not axis_info.get("wave"):
        return mz
    return MQ.invert_correction(MQ.fit_from_record(axis_info["wave"]), mz)


def _apply_axis(ts, fit, info, where, log):
    """`ts` with `mz` corrected by `fit` where it reaches (AXIS_COL holds the ppm
    removed), and `info` updated with what moved."""
    from peaky.batch import massqc as MQ
    mz = ts["mz"].to_numpy(dtype=float)
    inside = MQ.in_scope(fit, mz)
    shift = np.zeros(len(mz))
    if inside.any():
        shift[inside] = fit.predict(mz[inside], extrapolate=True)
    ts = ts.copy()
    ts["mz"] = MQ.apply_correction(fit, mz)
    ts[AXIS_COL] = shift
    moved = shift[inside] if inside.any() else np.array([0.0])
    info.update(applied=True, n_peaks=int(len(mz)), n_corrected=int(inside.sum()),
                frac_outside=round(float(1 - inside.mean()), 4) if len(mz) else None,
                shift_ppm={"min": round(float(np.min(moved)), 3),
                           "median": round(float(np.median(moved)), 3),
                           "max": round(float(np.max(moved)), 3)})
    log(f"[mass-axis] APPLIED {info.get('verdict')}: m/z corrected by "
        f"{info['shift_ppm']['min']:+.2f}..{info['shift_ppm']['max']:+.2f} ppm "
        f"({info['n_corrected']} of {info['n_peaks']} peaks) {where}")
    return ts


def _lock_model(ts, polarity, reagent_elements, info, log):
    """axislock.fit on the batch and its verdict: (model to apply or None, done) --
    `done` False when the locks could not build a model at all (the caller may
    fall back to the reference ions)."""
    from peaky.batch import axislock as AL
    from peaky.batch import massqc as MQ
    try:
        model, li = AL.fit(ts, polarity=polarity, reagent_elements=reagent_elements, log=log)
    except Exception as exc:      # noqa: BLE001 -- a diagnostic never stops the batch
        info["locks"] = {"why": f"the lock model failed ({type(exc).__name__}: {exc})"}
        log(f"[mass-axis] lock model failed: {type(exc).__name__}: {exc}")
        return None, False
    table = pd.DataFrame(li.pop("locks", []))
    info["_locks_table"] = table
    info["locks"] = li
    if model is None:
        log(f"[mass-axis] no lock model: {li.get('why')}")
        return None, False
    cv, raw, peak = li.get("cv_ppm"), li.get("raw_ppm"), li.get("max_abs_ppm")
    segs = ", ".join(f"m/z {s['lo']:.0f}-{s['hi']:.0f} ({s['kind']}, {s['n']} locks)" for s in model.segments)
    log(f"[mass-axis] LOCKS: {li.get('n_pass1')} unique within +-{AL.LOCK_PPM:g} ppm, "
        f"{li.get('n_links')} by mass difference, {li.get('n_narrow')} on the fitted curve, "
        f"{li.get('n_off_curve')} dropped off it; segments {segs}")
    for st in li.get("steps", []):
        log(f"[mass-axis] STEP {st['ppm']:+.2f} ppm between m/z {st['between'][0]:.2f} and "
            f"{st['between'][1]:.2f} (the instrument's axis; no smooth curve follows it)")
    for st in li.get("joins", []):
        log(f"[mass-axis] segments join between m/z {st['between'][0]:.2f} and {st['between'][1]:.2f} "
            f"({st['ppm']:+.2f} ppm: no step, the walk resumed past a sparse stretch)")
    if li.get("uncorrected_gaps"):
        log("[mass-axis] left uncorrected between segments (the step lies somewhere in there): "
            + ", ".join(f"m/z {a:.2f}-{b:.2f}" for a, b in li["uncorrected_gaps"]))
    log(f"[mass-axis] lock model: |error| median {raw} ppm raw -> {cv} ppm on held-out locks "
        f"(5-fold), largest correction {peak} ppm")
    info.update(model="locks", wave=model.as_dict(), scope_mz=list(model.mz_range),
                max_abs_ppm=peak)
    if cv is None or raw is None:
        info["verdict"] = "locks_unvalidated"
        return None, True
    if peak is not None and peak < MQ.ORBI_OFFSET_FLAT_PPM and raw < MQ.ORBI_OFFSET_FLAT_PPM:
        info["verdict"] = "axis_ok"
        log(f"[mass-axis] axis_ok: no lock reads {MQ.ORBI_OFFSET_FLAT_PPM:g} ppm off; nothing to correct")
        return None, True
    if cv > max(LOCK_CV_OK_PPM, 0.5 * raw):
        info["verdict"] = "locks_unpredictive"
        log("[mass-axis] the lock model does not predict held-out locks: not applied")
        return None, True
    info["verdict"] = "axis_steps" if li.get("steps") else "axis_trend"
    return model, True


def measure_axis(ts, reagent: str, klass: str | None, *, hold: str | None = None,
                 mode: str = "auto", polarity: str = "-", reagent_elements=(), log=print):
    """The batch's m/z axis and the correction it prescribes. Returns (ts, info, table):

      * `ts` -- the time series with `mz` corrected where the correction reaches
        and the ppm removed in AXIS_COL, when it was applied; else as given;
      * `info` -- the JSON-safe record batch_summary['mass_axis'] keeps (`model`
        'locks' or 'reference', the verdict, the model and its scope, how many
        peaks moved and by how much, or why nothing was applied: `held`);
        info['_locks_table'] (popped by the caller) holds every lock;
      * `table` -- the per-reference-ion probe table, None when not probed.

    On an Orbitrap, 'auto' and 'locks' model the axis from the batch's own peaks
    (batch.axislock: calibration-free locks, a smoothing spline, the steps the
    instrument's axis takes); it is applied when it predicts held-out locks to
    within half the raw error and some lock reads >= 1 ppm off. 'auto' falls back
    to the reagent's reference ions (batch.massqc, the instrument's own rules)
    only when the locks could not build a model; 'reference' goes there directly,
    and a TOF always does (it is measured there, never corrected).

    Not measured (info['skipped']) without a time series or an instrument class,
    or -- reference path -- without a reference table for the reagent or when the
    probe fails; the measurement never stops a batch. Measured but NOT applied
    (info['held']) on a TOF, when the caller holds it (`hold`: server-side
    scoring; a pool of several batches), or when the correction would exceed the
    model's cap (LOCK_MAX_CORRECTION_PPM / massqc.MAX_CORRECTION_PPM)."""
    from peaky.batch import massqc as MQ
    from peaky.chem import reference_ions as RI

    info: dict = {"applied": False, "verdict": None}

    def _skip(why):
        info["skipped"] = why
        log(f"[mass-axis] not measured: {why}")
        return ts, info, None

    if ts is None or not len(ts):
        return _skip("no batch time series")
    if not {"sample_item_id", "mz", "height"} <= set(ts.columns):
        return _skip("the time series lacks sample_item_id / mz / height")
    if klass not in ("orbitrap", "tof"):
        return _skip("instrument class unknown (no peak-width model)")
    tof = klass == "tof"

    if not tof and mode in ("auto", "locks"):
        log(f"[mass-axis] modelling the axis from the batch's own peaks "
            f"({ts['sample_item_id'].nunique()} spectra, {klass}, polarity {polarity})")
        model, done = _lock_model(ts, polarity, tuple(reagent_elements or ()), info, log)
        if done or mode == "locks":
            if model is None:
                return ts, info, None
            if hold is None and info["max_abs_ppm"] > LOCK_MAX_CORRECTION_PPM:
                hold = (f"the correction reaches {info['max_abs_ppm']:.1f} ppm, above "
                        f"{LOCK_MAX_CORRECTION_PPM:g}: a broken calibration to fix at the instrument")
            if hold is not None:
                info["held"] = hold
                log(f"[mass-axis] {info['verdict']} NOT applied: {hold}")
                return ts, info, None
            lo, hi = model.mz_range
            ts = _apply_axis(ts, model, info, f"inside m/z {lo:.0f}-{hi:.0f} "
                             f"({len(model.segments)} segment(s)); outside them -- below the first "
                             "lock, above the last, in the gaps between segments -- left as is", log)
            return ts, info, None
        log("[mass-axis] falling back to the reagent's reference ions")

    try:
        refs = RI.get(reagent)
    except KeyError:
        return _skip(f"no reference-ion table for reagent {reagent!r}")
    tol = MQ.member_ppm(tof)
    probe_ts = ts
    if "datetime_utc" not in ts.columns:
        # the times feed only the drift statistics; the spectra's order stands in
        order = {s: i for i, s in enumerate(pd.unique(ts["sample_item_id"]))}
        probe_ts = ts.assign(datetime_utc=pd.Timestamp("2000-01-01", tz="UTC")
                             + pd.to_timedelta(ts["sample_item_id"].map(order), unit="h"))
    log(f"[mass-axis] probing {len(refs)} {reagent} reference ions in "
        f"{ts['sample_item_id'].nunique()} spectra ({klass}, membership +-{tol:g} ppm)")
    try:
        table, v = MQ.run(probe_ts, refs, tol_ppm=tol, tof=tof, per_instrument=True)
    except Exception as exc:      # noqa: BLE001 -- a diagnostic never stops the batch
        return _skip(f"the reference-ion probe failed ({type(exc).__name__}: {exc})")
    MQ.report(table, v, log=lambda s: log(s.replace("[mass-qc]", "[mass-axis]")))
    info.update(model="reference", instrument=klass, verdict=v.get("verdict"),
                remedy=v.get("remedy"), qc=v)
    fit = MQ.correction(v)
    if fit is None:
        log(f"[mass-axis] axis left as measured (verdict {v.get('verdict')})")
        return ts, info, table
    lo, hi = (float(x) for x in fit.mz_range)
    grid = np.linspace(lo, hi, 200)
    peak = float(np.nanmax(np.abs(fit.predict(grid, extrapolate=True))))
    held = hold
    if held is None and tof:
        held = ("a TOF: measured, not applied (its calibrants are the anchor ions, clean "
                "at TOF resolution over a narrow m/z span, and its offsets are mostly "
                "ion-specific; see `peaky mass-qc`)")
    if held is None and peak > MQ.MAX_CORRECTION_PPM:
        held = (f"the correction reaches {peak:.1f} ppm, above {MQ.MAX_CORRECTION_PPM:g}: "
                "a broken calibration to fix at the instrument, not a residual to model")
    info.update(wave=fit.as_dict(), scope_mz=[lo, hi], max_abs_ppm=round(peak, 3))
    if held is not None:
        info["held"] = held
        log(f"[mass-axis] {v.get('verdict')} NOT applied: {held}")
        return ts, info, table
    where = ("at every m/z (a constant)" if int(fit.K) == 0 else
             f"inside m/z {lo:.0f}-{hi:.0f}; outside it the shape is unmeasured and the "
             "axis is left as is")
    ts = _apply_axis(ts, fit, info, where, log)
    return ts, info, table


def _reparsed(frame: pd.DataFrame | None) -> pd.DataFrame | None:
    """`frame` as a reader of its CSV sees it (written and re-read with the
    parser a post-hoc re-level uses), so the in-run evidence level and a re-level
    of the run dir read the same values."""
    if frame is None:
        return None
    import io as _io
    return pd.read_csv(_io.StringIO(frame.to_csv(index=False)), low_memory=False)


def _ts_for_levels(ts: pd.DataFrame | None) -> pd.DataFrame | None:
    """The stamped batch time series as the evidence level reads it from the
    run's per_file/_batch_ts.parquet: its level columns, through a parquet
    round trip (the same dtypes a post-hoc re-level reads)."""
    if ts is None:
        return None
    from peaky.assignment.levels import context as _CX
    cols = [c for c in _CX.TS_COLUMNS if c in ts.columns]
    import io as _io
    try:
        buf = _io.BytesIO()
        ts[cols].to_parquet(buf)
        buf.seek(0)
        return pd.read_parquet(buf)
    except Exception:  # noqa: BLE001 -- no parquet engine: the frame itself
        return ts[cols].copy()


def _instrument_of(rp) -> tuple:
    """(class or None, R at m/z 200 or None) of the batch's width model."""
    from peaky.assignment import evidence as EV
    if rp is None:
        return None, None
    klass, _fw = EV.instrument(rp)
    try:
        r = float(rp.r_at(200.0))
    except Exception:  # noqa: BLE001
        r = float("nan")
    return klass, (round(r, 1) if np.isfinite(r) else None)


def _run_dir_of(path: str) -> str | None:
    """The batch run directory a --corroborate source names (the dir itself, or
    the one run an out-dir holds); None for a ledger CSV or anything else."""
    path = os.path.expanduser(str(path).rstrip("/"))
    if not os.path.isdir(path):
        return None
    if os.path.isfile(os.path.join(path, "batch_summary.json")) and os.path.isdir(os.path.join(path, "per_file")):
        return path
    inner = [os.path.join(path, d) for d in sorted(os.listdir(path))]
    runs = [d for d in inner if os.path.isfile(os.path.join(d, "batch_summary.json"))
            and os.path.isdir(os.path.join(d, "per_file"))]
    return runs[0] if len(runs) == 1 else None


def _corroborate_partners(sources, *, log=print) -> tuple[dict, dict]:
    """The evidence scale's other-source partners from the --corroborate
    sources (D9): each Orbitrap-class RUN DIR levelled once by the scale with
    no partners of its own, `evidence.partners_from` taken under the run dir's
    name. A TOF-class or class-less run dir, a ledger CSV or a merged-only
    source gives none (its levels would be NA, or it carries no run context),
    nor does an Orbitrap run dir none of whose files is calibrated (no run
    window: its pairs carry no level), or whose reagent profile or context this
    process does not know (`evidence.partner_source_problem`): logged and skipped. Returns (partners, {source label: n neutrals})."""
    from peaky.assignment import evidence as EV
    from peaky.assignment.levels import routes as _RT
    parts, counts = [], {}
    for src in sources or []:
        run = _run_dir_of(src)
        label = os.path.basename(os.path.expanduser(str(src).rstrip("/")))
        if run is None:
            log(f"[assign_batch] --corroborate {label}: not a batch run dir -- feeds the merge vote only, "
                "no other-source partners")
            continue
        klass = EV.instrument(EV.source_resolution(run))[0]
        if klass != "orbitrap":
            log(f"[assign_batch] --corroborate {label}: instrument class {klass or 'unknown'} -- its evidence "
                "levels are NA, so it gives no other-source partners (the merge vote still reads it)")
            continue
        why = EV.partner_source_problem(run)
        if why:
            log(f"[assign_batch] --corroborate {label}: {why} -- no other-source partners "
                "(the merge vote still reads it)")
            continue
        src = EV.source_from_run_dir(run)
        if EV.no_run_window(src):
            log(f"[assign_batch] --corroborate {label}: {EV.NO_RUN_WINDOW_PARTNERS} (the merge vote still reads it)")
            continue
        lv = EV.level_source(src)
        p = EV.partners_from(lv, os.path.basename(run))
        parts.append(p)
        counts[os.path.basename(run)] = int(len(p))
        log(f"[assign_batch] --corroborate {label}: {len(p)} neutral(s) with an other-source partner")
    return (_RT.merge_partners(*parts) if parts else {}), counts


def _claims_summary(merged: pd.DataFrame, levels: pd.DataFrame) -> dict:
    """batch_summary['claims']: the claim each level supports (evidence.claim_class:
    identified / neutral / ion / tentative + the reagent and not-assessed
    buckets, every tally over evidence.CLAIM_KEYS) counted per merged row, per
    pooled (neutral, adduct) pair, per stage and per tier. An ion-only merged row (an `ion_only_of` link) counts under
    its own key 'ion-only', not under its tier. Tier and claim are separate
    verdicts: they are tallied side by side and neither is read off the other.
    `n_unlevelled` = merged rows with no level (their claim reads tentative)."""
    from peaky.assignment import evidence as EV

    pooled = EV.summarize_claims(levels["claim"] if len(levels) and "claim" in levels.columns else [])
    if not len(merged) or "claim" not in merged.columns:
        return {"merged": EV.summarize_claims([]), "pooled": pooled,
                "per_stage": {}, "by_tier": {}, "n_unlevelled": 0}
    io = (merged["ion_only_of"].notna() if "ion_only_of" in merged.columns
          else pd.Series(False, index=merged.index))
    tiers = (merged["tier"].fillna("").astype(str) if "tier" in merged.columns
             else pd.Series("", index=merged.index))
    by_tier = {t: EV.summarize_claims(merged.loc[~io & (tiers == t), "claim"])
               for t in sorted(tiers[~io].unique(), key=lambda t: (-TIER_RANK.get(t, 0), t))}
    if io.any():
        by_tier["ion-only"] = EV.summarize_claims(merged.loc[io, "claim"])
    return {
        "merged": EV.summarize_claims(merged["claim"]),
        "pooled": pooled,
        "per_stage": ({str(s_): EV.summarize_claims(merged.loc[merged["stage"] == s_, "claim"])
                       for s_ in sorted(merged["stage"].dropna().astype(str).unique())}
                      if "stage" in merged.columns else {}),
        "by_tier": by_tier,
        "n_unlevelled": (int(merged["evidence_level"].isna().sum())
                         if "evidence_level" in merged.columns else int(len(merged))),
    }


def _auto_reagent_table(reagent, peaks, ts_peaks):
    """The table `reagent` is resolved on. A name never reads one (`peaks` passes
    through). 'auto' reads the server's matches -- an `ionization_mechanism`
    column -- so it takes `peaks` when that carries matches, else `ts_peaks`, else
    whichever of the two has the (empty) column, which then raises naming what it
    saw. The batch= roster is one row per sample with a polarity and no match:
    resolving on it could only guess, so when neither table carries the column
    this raises, asking for the reagent or the per-peak time series."""
    if reagent != "auto":
        return peaks
    tables = [t for t in (peaks, ts_peaks)
              if t is not None and "ionization_mechanism" in getattr(t, "columns", [])]
    for t in tables:
        if t["ionization_mechanism"].notna().any():
            return t
    if tables:
        return tables[0]
    raise ValueError(
        "reagent='auto' reads the reagent from the batch's server matches (an "
        "ionization_mechanism column), and neither peaks= nor ts_peaks= carries "
        "one (the batch= sample roster never does): pass reagent=NAME, or the "
        "batch's per-peak time series as ts_peaks=")


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
        corroborate=None, mass_axis: str = "auto", mass_axis_hold: str | None = None,
        isotopologue_rows: bool = True,
        log=print, **assign_kw) -> dict:
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
    `reagent='auto'` reads the server matches of `peaks`, else of `ts_peaks`;
    the batch= roster carries none, so batch= alone needs a reagent name
    (`_auto_reagent_table`). `context` defaults to the reagent profile's
    context. The side channels every file may open are the profile's declared
    ones (`ReagentProfile.side_channels`; the uronium profile's [M+NH4]+) unless
    `cfg.side_channels` is already set (a `--side-channels` choice, () = closed);
    batch_summary records the requested set and what the files opened.
    Extra kwargs pass through to assign.run. Writes (see paths.RunPaths): merged_ledger.csv +
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
    whose neutrals corroborate this run -- the other reagent channel, or the
    other instrument on the same air. It feeds the merge vote's evidence class
    (each source by the neutrals it holds by its own evidence,
    `evidence.vote_cross_neutrals`) and, from an Orbitrap-class run dir, the
    evidence scale's other-source partner tag (`evidence.partners_from` of the
    source levelled once with no partners), which never unlocks an evidence
    level but, as a route, can anchor the series exclusion that lifts a pair out
    of 5a.
    The evidence level of the merged ledger is computed on the POOLED per-file
    ledgers (cover + residual files as one source, every file-count minimum 3;
    `evidence.level_batch`) and stamped on the merged ledger by
    (neutral_formula, adduct); the per-file ledgers keep their own per-file
    (adapted) level. batch_summary['evidence_levels'] records the counts,
    tables/evidence_levels.csv the level and the facts behind every pair. The
    claim each level supports (identified / neutral / ion / tentative, plus the
    reagent and not-assessed buckets) is stamped on every merged row and
    tallied in batch_summary['claims'] (merged, pooled, per stage, per tier with
    the ion-only rows apart); it changes no ion, tier or level.

    `mass_axis` ('auto', the default; 'locks', 'reference' or 'off'): before this
    run picks its cover or assigns anything, the batch's m/z axis is modelled
    (`measure_axis`) and, on an Orbitrap whose model finds an axis error, the
    time series and every file's peak table are corrected where the model
    reaches (io_mascope.set_axis_correction; the spawned workers get it too).
    'auto' models it from the batch's own peaks (batch.axislock: calibration-free
    locks, a smoothing spline and the steps the instrument's axis takes) and
    falls back to the reagent's formula-certain reference ions (batch.massqc)
    only when the locks cannot build a model. Pass 1's per-file
    self-calibration only models a constant plus a 1/(m/z) term, so an axis
    that rises and falls -- or steps -- across the range left correct formulas
    off-centre by several of its widths. A TOF is measured, never corrected;
    `mass_axis_hold` (a reason) measures without correcting -- the pooled path
    passes one for a pool of several batches -- and so does server-side scoring
    (PEAKY_LOCAL_SCORING=0), which scores the server's own peaks. A series that
    carries an earlier run's correction is restored first (`restore_axis`).
    batch_summary['mass_axis'] records the model and what moved (`offsets_ppm`
    and the per-file scoring are then read on the corrected axis);
    tables/mass_axis_locks.csv the locks, tables/mass_axis.csv the reference
    ions when probed. Trace-first keeps its own wave
    (batch.tracefirst) and 'off' reproduces a run without the step.

    `isotopologue_rows` (default on; CLI --no-isotopologue-gate turns it off): on an
    Orbitrap-class batch whose time series carries peak areas, a merged row whose
    line is another merged ion's (or a reagent ion's) 13C / 18O / 15N / 34S / 37Cl /
    81Br / Si isotopologue at its expected area ratio over the batch leaves the
    merged ledger before the stamp, which then gives the line to the parent
    (iso_checks.satellite_rows; tables/isotopologue_rows.csv,
    merge_gates['isotopologue']). Its pooled pair reads an isotope-check veto in the
    evidence levels (check 'SAT' in tables/iso_checks.csv). The per-file ledgers keep
    their own reading, and a single-sample `peaky assign` has no such gate."""
    from peaky.assignment import assign as A
    from peaky.assignment import evidence as EV
    from peaky.batch import timeseries as _TSN
    from peaky.io import io_mascope as IO

    if mass_axis not in MASS_AXIS_MODES:      # before any server call or folder
        raise ValueError(f"mass_axis must be one of {MASS_AXIS_MODES}, got {mass_axis!r}")
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

    prof = P.resolve(reagent, _auto_reagent_table(reagent, peaks, ts_peaks))
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
    # the side channels the profile declares (uronium: [M+NH4]+), same rule: a cfg
    # that already carries a tuple (--side-channels, () included) wins. The cfg
    # rides base_kw into every spawned worker, so each file opens exactly these.
    P.apply_side_channels(cfg, prof, log=log)
    side_source = P.side_channels_source(cfg.side_channels, prof)
    assign_kw["cfg"] = cfg
    selection = dict(selection_meta or {})
    sel = None                 # our own cover table (None on the sample_ids= path)
    trace_sample = None
    # The peak-width model, resolved ONCE for the batch (chem.resolution) and
    # handed to every per-file run: it sizes trace-first's dedup cell and, on
    # every path, the `resolvability` stamp each per-file ledger carries (a
    # blended, uncorroborated peak is capped at Candidate; the vote's class reads it)
    # and, per file and pooled, the evidence level's instrument class.
    # MEASURED from the raw profile by default -- a TOF can be tuned anywhere
    # and a declared number is a guess -- and the caller's only when they gave
    # one. A measurement that cannot be made stops trace-first (nothing sizes
    # its cell) and is a log line on the cover path (the stage skips, the
    # columns stay NA).
    rp = _width_model_for_batch(resolving_power, client,
                                ts_peaks if ts_peaks is not None else peaks, log)
    # the class the class-gated per-file stages read (PassConfig.instrument_class):
    # resolved ONCE here, a TOF roster winning over a width model that reads
    # Orbitrap-class, and carried to every file by the cfg (workers included)
    cfg.instrument_class = _axis_class(_instrument_of(rp)[0], _roster_class(peaks))
    log(f"[assign_batch] instrument class: {cfg.instrument_class or 'unknown (class-gated stages off)'}")
    # The batch's m/z axis (see the docstring's `mass_axis`), measured and, on an
    # axis error, corrected HERE -- before the cover is picked and before any
    # file's peak table is read, so every stage sees one axis.
    ts_peaks = restore_axis(ts_peaks, log=log)
    axis_ids = ([str(s) for s in ts_peaks["sample_item_id"].unique()]
                if ts_peaks is not None and "sample_item_id" in ts_peaks.columns else [])
    IO.clear_axis_correction(axis_ids)     # this run's samples only (a thread may run another)
    axis_args = None          # (sample ids, wave record) for the spawned workers
    if mass_axis == "off":
        axis_info = {"applied": False, "verdict": None, "skipped": "--mass-axis off"}
    elif trace_first:
        axis_info = {"applied": False, "verdict": None,
                     "skipped": "trace-first applies its own wave (batch.tracefirst)"}
    else:
        hold = mass_axis_hold
        if hold is None and not IO._local_scoring_enabled():
            hold = ("server-side scoring (PEAKY_LOCAL_SCORING=0) scores the server's own "
                    "peaks, on the server's axis")
        halogen = EV.channel_halogen(prof.adducts)
        ts_peaks, axis_info, axis_table = measure_axis(
            ts_peaks, prof.name, _axis_class(_instrument_of(rp)[0], _roster_class(peaks)),
            hold=hold, mode=mass_axis, polarity=prof.polarity,
            reagent_elements=(halogen,) if halogen else (), log=log)
        if axis_table is not None:
            axis_table.to_csv(os.path.join(TAB, "mass_axis.csv"), index=False)
        locks_table = axis_info.pop("_locks_table", None)
        if locks_table is not None and len(locks_table):
            locks_table.to_csv(os.path.join(TAB, "mass_axis_locks.csv"), index=False)
        if axis_info["applied"]:
            axis_args = (axis_ids, axis_info["wave"])
            IO.set_axis_correction(*axis_args)
    axis_info = {"mode": mass_axis, **axis_info}
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
    # --corroborate: the neutrals of the named sources, or a ready set -- the
    # merge vote's cross set (the vote class is computed per file in `_apply`);
    # the Orbitrap-class run dirs among the sources also give the evidence
    # scale's other-source partners (below, at the pooled level). The paths are
    # recorded so the run says what corroborated it.
    if isinstance(corroborate, (set, frozenset)):
        cross_sources, cross = [], {str(x) for x in corroborate}
    else:
        cross_sources = [str(x) for x in (corroborate or [])]
        cross = EV.vote_cross_neutrals(cross_sources)
    # the batch's width model -> every per-file `resolvability` stage (never re-measured per file)
    # and the per-file evidence stage's instrument class
    assign_kw["resolving_power"] = rp
    # the reagent profile whose adducts span the evidence level's enumeration space
    assign_kw.setdefault("reagent_profile", prof.name)
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
    reflists_active, _tags, rl_record = RL.activate(batch or "", dataset or "", getattr(prof, "label", ""),
                                                    record=True)
    # how they were activated (which keyword in which name): the evidence
    # scale's context source, per file and pooled (batch_summary["reflists_context"])
    assign_kw.setdefault("reflists_context", rl_record)
    if reflists_active:
        log(f"[assign_batch] reference lists active: {RL.active_versions(reflists_active)} "
            f"(context {sorted(_tags) or 'contaminants-only'})")
    per_file, offsets, per_stats = {}, {}, []
    scorings: dict = {}        # per-sample pattern_scoring, for the run manifest
    level_frames: dict = {}    # sid -> its ledger's M0/iso rows + predicate columns
                               # (evidence.trim): the batch checks' input
    alias_ties: dict = {}      # sid -> its tied [M+NO3]- rows and whether the tie is alias-only (rule K)
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

    def _vote_class(led, stats):
        """The reading's vote class over one file's ledger AS ITS RUN RETURNED IT
        (evidence.vote_classes; see VOTE_CLASS), indexed by the ledger's rows."""
        return EV.vote_classes(EV.trim(led), cross=cross, resolution=rp,
                               halogen=(stats or {}).get("reagent_halogen", EV.DETECT_HALOGEN))

    def _apply(sid, led, plaus, stats, stage, scoring=None):
        """Parent-side reduce (called in sample_ids order): write the per-file CSV
        and fold this sample into the accumulators. Order-fixed so align() -- which
        has order-sensitive tie-breaks -- yields byte-identical output either path.

        Under TRACE-FIRST a residual file may only ADD what the trace stamp left
        unexplained: its M0 rows are kept within `tol_ppm` of a residual bin and
        dropped elsewhere. Otherwise ten per-file ledgers of a noisy TOF would
        merge back in on top of the traces -- the per-file lottery trace-first
        exists to avoid -- and out-vote a trace's reading (measured: a Candidate
        on the trace ledger re-read as Assigned by three residual files)."""
        # the trace-first sample hands in its class computed BEFORE the trace
        # columns were merged on (see the call): carried as a private column,
        # never written
        own_class = led.pop(_OWN_VOTE_CLASS) if _OWN_VOTE_CLASS in led.columns else None
        led.to_csv(os.path.join(pfdir, f"{sid}_ledger.csv"), index=False)
        level_frames[sid] = EV.trim(led)
        alias_ties[sid] = _LT.alias_only_ties(led)
        plaus_audit.extend(plaus)
        protected_neutrals.update(_protected_neutrals(led))
        known_pool.extend(known_evidence(led, src=sid))
        m0 = _m0(led)
        # the reading's vote class (private to the vote: the decision it has
        # always read, over this file's facts, the --corroborate cross set, the
        # batch's width model and the reagent halogen the file's own run read --
        # the declared channels' (C43), carried back in its stats) -- joined by
        # row before any subset below
        vc = own_class if own_class is not None else _vote_class(led, stats)
        m0[VOTE_CLASS] = pd.to_numeric(vc.reindex(m0.index), errors="coerce").fillna(0).astype(int)
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
        # What this sample's candidates were scored at: the record the run
        # returned. Since C42 a run may re-score at the mass trend its own
        # calibration accepted, and that lives in the process that ran it -- a
        # spawned worker's -- so the parent re-reading the sample would record the
        # constant offset for a file scored at the trend (and a decoy arm would
        # inherit the wrong one). The parent's own read is the fallback only.
        if scoring is not None:
            scorings[sid] = dict(scoring)
        else:
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
        # C46: the batch's typical detection edge (the median of its files' own),
        # the footing of the tier pass's counting-detector floor. Set once, from
        # the first stage's files (the cover; on a trace-first run the residual
        # picks, the only per-file stage it runs); every per-file cfg copy below
        # carries it. The trace sample itself (averaged traces, no scoring
        # snapshot, no class) never sees the floor: its heights sit on another
        # footing (PassConfig.audit_floor_cps).
        if getattr(cfg, "noise_edge_batch_cps", None) is None:
            cfg.noise_edge_batch_cps = batch_noise_edge(client, ids)
            log(f"[assign_batch] batch detection edge (median of {len(ids)} files' own): "
                f"{cfg.noise_edge_batch_cps if cfg.noise_edge_batch_cps is None else round(cfg.noise_edge_batch_cps, 4)} cps")
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
                       dict(res.get("stats", {})), stage, res.get("pattern_scoring"))
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
                    initargs=(context, reflists_active, base_kw, ts_path,
                              P.registry_extras(), axis_args)) as ex:
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
                _apply(sid, out["ledger"], out["plausibility_audit"], out["stats"], stage,
                       out.get("pattern_scoring"))

    if trace_sample is not None:
        from peaky.batch import tracefirst as TFT
        kw = dict(assign_kw, cfg=copy.deepcopy(cfg), occurrence=trace_sample.occurrence)
        kw["cfg"].trace_sample = True      # batch-mean heights: no same-spectrum stages
        TFT.engine_settings(kw["cfg"], trace_sample, log=log)
        log(f"[assign_batch] (1/1) assigning {trace_sample.sample_id} (offline, "
            f"{len(trace_sample.peaks)} trace peaks) ...")
        res = A.run(trace_sample.sample_id, context=context, log=log,
                    reflists_active=reflists_active, peaks=trace_sample.peaks, **kw)
        # The vote class is read off the ledger the engine returned, BEFORE the
        # trace columns are merged on: the trace table carries its own
        # `resolvability` / `sep_hwhm`, so after the merge the ledger's columns
        # are suffixed (_x / _y) and a class read there sees no resolvability --
        # a blended exact-mass reading would count as "formula confirmed" (the
        # per-file stage before 0.10.0 read the unmerged ledger: unconfirmed).
        own = res["ledger"]
        own = own.assign(**{_OWN_VOTE_CLASS: _vote_class(own, res.get("stats"))})
        led = own.merge(
            trace_sample.traces[[c for c in TFT.TRACE_COLS if c in trace_sample.traces.columns]],
            on="peak_id", how="left")
        trace_sample.traces.to_csv(os.path.join(TAB, "traces.csv"), index=False)
        _apply(trace_sample.sample_id, led, res.get("plausibility_audit") or [],
               dict(res.get("stats", {})), STAGE_COVER, res.get("pattern_scoring"))
        log(f"[assign_batch] (1/1) done {trace_sample.sample_id}")
    else:
        _assign_files(list(sample_ids), STAGE_COVER, n_jobs)

    from peaky.chem import reagents as _RG
    from peaky.assignment import plausibility as PL
    from peaky.batch import timeseries as _TS
    from peaky.batch import reagent_water as _RW
    _rgk = _RG.reagent_for_adducts(list(prof.adducts or []))

    scale = None   # the batch's traces.MassScale, measured at the first merge
    rwater = None  # the reagent-water ladder (batch.reagent_water), measured at the first merge

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
        nonlocal scale, rwater
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
        # The reagent-water ladder (C15, batch/reagent_water.py): the rungs
        # core.(H2O)n of the profile's water cores that this batch's own time series
        # shows -- core and lower rungs present in half an acquisition segment's
        # spectra, the rung 3x above its decoy offsets. A merged analyte row on a
        # passing rung is the water cluster and leaves the merged ledger (listed in
        # tables/reagent_water.csv); the rung joins the stamp as a reagent row below.
        if rwater is None:
            # the rung test follows the width model: decoys at fixed Da offsets on an
            # Orbitrap, scaled to the line width (gaps allowed, M+1 carbon test) on a TOF
            rwater = _RW.measure(ts_peaks, prof, tol_ppm=scale.stamp_ppm, log=log, resolution=rp)
        # after the TOF test a row leaves only when a file carrying its winning reading
        # (jitter) lies in a segment where its rung passed (segment_of; None otherwise)
        merged, rw_stripped = _RW.strip_rung_rows(merged, rwater["rungs"], tol_ppm=scale.stamp_ppm, log=log,
                                                  jitter=jitter, segment_of=rwater.get("segment_of"))
        # The merged row's tier_reason (from align: the vote's exemption, else NA)
        # also takes the batch-level gates' notes below (cleanup._note appends to
        # it): a re-read can leave the merged formula different from EVERY per-file
        # reading, and the row itself must say why.
        if "tier_reason" not in merged.columns:
            merged["tier_reason"] = pd.NA
        merge_gates: dict = {}
        merge_gates["reagent_water"] = _RW.summary(rwater["rungs"], rw_stripped, n_cores=rwater["n_cores"],
                                                   tol_ppm=rwater["tol_ppm"],
                                                   segment_sizes=rwater["segment_sizes"],
                                                   rung_test=rwater.get("rung_test"))
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
        # The isotopologue rows (iso_checks.satellite_rows; Orbitrap-class batches
        # with peak areas): a merged row whose line is another merged ion's (or a
        # reagent ion's) isotopologue at the expected area ratio across the batch
        # leaves the merged ledger HERE -- after the trace reconciliation (it reads
        # mz_trace) and before the stamp, so the line is stamped as the parent's
        # satellite and the residual stage does not target it. Stateless: the
        # second merge (cover + residual) re-strips a row a residual file re-adds.
        iso_rows = _IC._sat_empty()
        if isotopologue_rows:
            _rg = None
            if identified_aux:
                _ia = pd.concat(identified_aux, ignore_index=True)
                _rg = _ia[_ia["role"].astype(str) == "reagent"] if "role" in _ia.columns else None
            merged, iso_rows, merge_gates["isotopologue"] = _IC.satellite_rows(
                merged, ts_peaks, resolution=rp, mass_scale=scale, klass=cfg.instrument_class, prof=prof,
                reagents=_rg, per_file=level_frames, log=log)
        else:
            merge_gates["isotopologue"] = {"ran": False, "n_stripped": 0, "n_mixed": 0, "n_exempt": 0,
                                           "skipped": "--no-isotopologue-gate"}
        out = {"merged": merged, "jitter": jitter, "merge_gates": merge_gates,
               "trace_info": trace_info, "stamp_tol": stamp_tol, "ts_annot": None,
               "predicted_rows": {}, "predicted_tracks": None,
               "reagent_water": _RW.table(rwater["rungs"], rw_stripped),
               "isotopologue": iso_rows}
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
            # ... plus the batch's passing reagent-water rungs (reagent rows)
            _rw_rows = _RW.stamp_rows(rwater["rungs"])
            if len(_rw_rows):
                _aux = _rw_rows if _aux is None else pd.concat([_aux, _rw_rows], ignore_index=True)
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
    # the batch checks over the pooled per-file ledgers (their facts are step-0
    # inputs of the evidence level below: a vetoed pair is rejected).
    # rule U: the profile's neutral pair, measured on the stamped batch time
    # series over the pooled ledgers; the table is written for every run (empty
    # without a pair or a time series)
    _pair = tuple(getattr(prof, "neutral_pair", ()) or ())
    pairs_table = _NP.measure(ts_annot, level_frames, _pair, log=log)
    pairs_table.to_csv(os.path.join(TAB, "neutral_pairs.csv"), index=False)
    # rule K (label_untie / label_veto): on a 15N-labelled
    # nitrate channel the 14N and 15N lines of one cluster arbitrate each other's
    # reading on the same series; written for every run (empty out of scope)
    twins_table = _LT.measure(ts_annot, level_frames, prof, alias_ties=alias_ties, log=log)
    twins_table.to_csv(os.path.join(TAB, "label_twins.csv"), index=False)
    # the isotope checks (iso_veto): rule C, REQ and HIGH read each committed
    # formula's isotope claims off the same stamped series (the instrument class
    # from the batch's width model), rule H its exact halogen line (a lock,
    # judged against the batch's element budget: `context`); written for every
    # run (empty without a time series); a refuted pair is rejected (5b)
    iso_table = _IC.measure(ts_annot, level_frames, prof, resolution=rp, mass_scale=scale,
                            x_edge=x_edge, context=context, edge_cps=getattr(cfg, "noise_edge_batch_cps", None),
                            log=log)
    # ... plus the merged-ledger isotopologue gate's strips (iso_checks.veto_rows,
    # check 'SAT'): the pooled pair of a line the merged ledger gave to its parent is
    # refuted the same way, so evidence_levels.csv agrees with the merged ledger
    _sat_veto = _IC.veto_rows(res_m.get("isotopologue"))
    if len(_sat_veto):
        iso_table = (pd.concat([iso_table, _sat_veto], ignore_index=True) if len(iso_table)
                     else _sat_veto)
    iso_table.to_csv(os.path.join(TAB, "iso_checks.csv"), index=False)
    # the TOF ion-M+2 gates (iso_checks.tof_m2_gates; TOF-class batches only): a
    # merged winner whose own M+2 line REQ refutes over the batch -- a species the
    # known-species lock decided included -- and a merged line that is the 81Br
    # partner of the line one spacing below it are Candidate. No re-vote: the
    # winner and its reading stay, the row says why.
    merge_gates["tof_m2"] = _IC.tof_m2_gates(merged, iso_table, ts_annot, resolution=rp, mass_scale=scale,
                                             log=log)
    # THE EVIDENCE LEVEL (the scale of peaky 0.10.0): the batch's per-file
    # ledgers pooled as ONE source (cover + residual files; every file-count
    # minimum 3), re-read from the per_file/<sid>_ledger.csv files just written
    # in sorted sample-id order with the parser a post-hoc re-level uses, so the
    # run and `scripts/level_ledger.py` on its run dir see the same inputs; with
    # the stamped time series, the merged ledger (the NH4 rule's ion index), the
    # protected neutrals, the per-file gates and degeneracy calibrations, and the
    # batch checks' vetoes. Stamped on the merged ledger by (neutral, adduct); a
    # merged row no pooled pair holds (a batch-level re-read) gets no level.
    from peaky.assignment.levels import lists as _LS
    rl_context = _LS.reflists_context(reflists_active, rl_record)
    level_summary = {
        "reagent": prof.name, "label": prof.label, "context": context,
        "reflists_active": [list(x) for x in RL.active_versions(reflists_active)],
        "reflists_context": rl_context,
        "resolution": rp.as_dict() if rp is not None else None,
        "per_file": copy.deepcopy(per_stats), "amine_r_min": float(amine_r_min),
        # the pair facts' reagent halogen: the profile's declared channels (C43)
        "reagent_halogen": EV.channel_halogen(prof.adducts),
    }
    level_ledgers = {sid: pd.read_csv(os.path.join(pfdir, f"{sid}_ledger.csv"), low_memory=False)
                     for sid in sorted(level_frames)}
    run_inputs = EV.RunInputs(
        summary=level_summary, merged=_reparsed(merged), ts=_ts_for_levels(ts_annot),
        iso_checks=iso_table, label_twins=twins_table, neutral_pairs=pairs_table,
        protected=set(protected_neutrals), activation=rl_record)
    partners, partner_counts = _corroborate_partners(cross_sources, log=log)
    levels = EV.level_batch(level_ledgers, run_inputs=run_inputs, partners=partners or None)
    merged = EV.stamp_merged(merged, levels)
    EV.for_output(levels).to_csv(os.path.join(TAB, "evidence_levels.csv"), index=False)
    klass, r200 = _instrument_of(rp)
    ev_summary = {
        "scale": f"peaky {EV.SCALE_RELEASE}",
        "instrument": {"class": klass, "r200": r200},
        "pooled": EV.summarize(levels["evidence_level"]) if len(levels) else {},
        "merged": EV.summarize(merged["evidence_level"]) if len(merged) else {},
        "per_stage": ({str(s_): EV.summarize(merged.loc[merged["stage"] == s_, "evidence_level"])
                       for s_ in sorted(merged["stage"].dropna().astype(str).unique())}
                      if len(merged) and "stage" in merged.columns else {}),
        "n_pairs": int(len(levels)),
        "n_unstamped": int(merged["evidence_level"].isna().sum()) if len(merged) else 0,
        "side_channels_locked": bool(EV.SIDE_CHANNELS_LOCKED),
        "unlocked": sorted(EV.UNLOCKED),
        # the --corroborate sources: what fed the vote's cross set, and the
        # other-source partners each Orbitrap-class run dir gave
        "n_corroborate": int(len(cross)), "cross_source": cross_sources,
        "partners": partner_counts,
        "amine_r_min": float(amine_r_min),
        "neutral_pairs": _NP.summary(pairs_table, _pair),
        "label_twins": _LT.summary(twins_table, prof),
        "iso_checks": _IC.summary(iso_table, rp),
    }
    log(f"[assign_batch] evidence levels (peaky {EV.SCALE_RELEASE}) over {len(level_ledgers)} pooled "
        f"file(s), instrument class {klass or 'unknown'}: {ev_summary['pooled']} "
        f"({ev_summary['n_pairs']} neutral/adduct pairs); {ev_summary['n_unstamped']} merged row(s) "
        f"without a pooled pair -> tables/evidence_levels.csv")
    # an Orbitrap-class batch none of whose files calibrated the degeneracy window
    # (a sparse peak table: too few isotope-backed core rows) has no level at all --
    # said once for the run, in the summary, the console and the report, not only
    # row by row in the evidence column
    levels_na_reason = (EV.levels_not_assessed_reason(levels["evidence"], levels["evidence_level"])
                        if len(levels) and {"evidence", "evidence_level"} <= set(levels.columns) else None)
    if levels_na_reason:
        log(f"[assign_batch] WARNING: {levels_na_reason}")
    # the claim each level supports, tallied beside the tier (never read off it)
    claims_summary = _claims_summary(merged, levels)
    log("[assign_batch] claims (merged): "
        + " | ".join(f"{k} {claims_summary['merged'][k]}" for k in EV.CLAIM_KEYS))
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
    # The TOF mass-only flag (assignment/mass_only.py): on a TOF-class run, every
    # Assigned reading that no attached isotope line speaking for the neutral supports
    # in any of its Assigned files is FLAGGED -- `mass_only` + a short reason on
    # the merged row, never a tier or a level: a decoy-measured fact about what
    # the reading rests on (the formula space saturates with m/z on a TOF), kept
    # visible instead of hiding what makes it at high masses. Read off the same
    # per-file ledgers the levels pooled; the class is the batch width model's,
    # else the files' scoring class. The columns exist on every run (empty off a TOF).
    from peaky.assignment import mass_only as _MO
    tof_flag = _MO.flag_merged(
        merged, level_ledgers,
        klass=_MO.instrument_class(resolution=rp, instrument_types=[(s_ or {}).get("instrument_type")
                                                                    for s_ in scorings.values()]),
        threshold=_MO.flag_mz(_cfg), halogen=level_summary["reagent_halogen"], resolution=rp, log=log)
    merged.to_csv(os.path.join(out_dir, "merged_ledger.csv"), index=False)
    jitter.to_csv(os.path.join(TAB, "jitter.csv"), index=False)
    # the reagent-water rungs and the merged readings they displaced (always written:
    # a stable artifact set; header only when the profile declares no water cores)
    res_m["reagent_water"].to_csv(os.path.join(TAB, "reagent_water.csv"), index=False)
    # the isotopologue rows the merged ledger gave to their parent, and the mixed /
    # exempt lines it only noted (always written: header only when none)
    res_m["isotopologue"].to_csv(os.path.join(TAB, "isotopologue_rows.csv"), index=False)
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
    # the side channels the files OPENED (each file's stats carries its own list),
    # unioned in file order, and how many files opened each
    side_opened: list = []
    side_files: dict = {}
    for x in per_stats:
        for a in (x.get("side_channels") or ()):
            if a not in side_opened:
                side_opened.append(a)
            side_files[a] = side_files.get(a, 0) + 1
    if side_opened:
        log(f"[assign_batch] side channels opened: "
            + ", ".join(f"{a} ({side_files[a]} of {len(per_stats)} files)" for a in side_opened))
    summary = {
        "reagent": prof.name, "label": prof.label, "context": context,
        # the side channels the run asked for (`side_channels_requested`, from
        # `side_channels_source`) and the run-level union the files actually opened
        # (`side_channels`; per file: per_file[].side_channels, files per channel:
        # `side_channels_files`) -- a decoy arm opens what this records
        "side_channels": side_opened,
        "side_channels_files": side_files,
        "side_channels_requested": list(cfg.side_channels or ()),
        "side_channels_source": side_source,
        # the reagent halogen of the profile's declared channels (C43): what the
        # pooled pair facts read, and what a post-hoc re-level of this run reads
        "reagent_halogen": level_summary["reagent_halogen"],
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
        # C46: the median of the files' own detection edges, the footing of the
        # tier pass's counting-detector floor (per file: tof_assign_floor_cps)
        "noise_edge_batch_cps": getattr(cfg, "noise_edge_batch_cps", None),
        "tol_ppm": tol_ppm, "offsets_ppm": offsets,
        # the batch's m/z axis against the reagent's reference ions, and the
        # correction applied to it before the cover was picked (measure_axis)
        "mass_axis": axis_info,
        "pattern_scoring": scorings,
        # which scorer judged the candidates: 'local' (in-process, the default) or
        # 'server' (match_compounds; PEAKY_LOCAL_SCORING=0) -- the report's Methods name it
        "scorer": "local" if IO._local_scoring_enabled() else "server",
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
        # per_stage = merged rows by cover / residual; over the levels and the
        # reagent / NA buckets
        "evidence_levels": ev_summary,
        # the claim each level supports (identified / neutral / ion / tentative +
        # the reagent and not-assessed buckets), merged / pooled / per stage / per tier
        "claims": claims_summary,
        # why the scale assessed nothing on this run (no file calibrated), else null
        "levels_not_assessed_reason": levels_na_reason,
        # the ion-only bucket (the `ion_only` stage): channels opened, merged rows
        # carrying an `ion_only_of` link, per-file rows behind them, files holding
        # any, and their merged levels
        "ion_only": ion_only_summary,
        # the TOF mass-only flag (assignment/mass_only.py): the class it keyed on,
        # whether it ran, the m/z threshold of its two reasons, and the Assigned /
        # flagged counts below and at or above it (tier and level unchanged)
        "tof_flag": tof_flag,
        "reflists_active": RL.active_versions(reflists_active),   # [(id, data_version)]
        # how the lists were activated: {tags, matched: {tag: [[field, keyword]]},
        # active: [[id, version, how]]} (the evidence level's context source)
        "reflists_context": rl_context,
        "amine_r_min": float(amine_r_min),
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
    IO.clear_axis_correction(axis_ids)     # the correction belongs to this run only
    return {"profile": prof, "context": context, "sample_ids": sample_ids,
            "per_file": per_file, "offsets": offsets, "merged": merged,
            "jitter": jitter, "summary": summary, "out_dir": out_dir,
            "evidence": levels,
            "residual_samples": rsel, "stages": dict(stages),
            # the time series every stage read (axis-corrected, with AXIS_COL, when
            # summary['mass_axis']['applied']); the pipeline's cluster and Van
            # Krevelen figures read it
            "ts_peaks": ts_peaks}
