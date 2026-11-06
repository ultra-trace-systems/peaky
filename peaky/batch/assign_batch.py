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
ion read as C13H14O4 [M+NH4]+ or as C13H17NO4 [M+H]+ -- the one the most files
could fix wins: it ranks by the vote's private tier (VOTE_TIER), which reads a
file's Assigned reading as Candidate where the file left its reagent-N split open
(the tier itself answers "is the ion right?", and a reagent-N pair is one ion).
The merged tier is the best per-file tier of the winning ion. The losing
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

THE EPISODE STAGE (sampling.py, THE EPISODE STAGE). Between the cover and the
first assignment, every short bright episode of the batch gets its peak file and
up to two neighbours, and every episodic bin peaking at >= 5x its sample's noise
edge its own peak spectrum -- no budget (`k_episode_max` is an optional safety
cap); those files are
assigned with the cover's (`stage` cover; `tables/selected_samples.csv` role
'episode'), `tables/episodes.csv` and `batch_summary.json['selection']['episodes']`
record the episodes, and the per-file ledgers of a batch with an episode file
carry an `episode` column (True on an episode file's rows on an episodic bin)
for the merge-side readers. `episodes=False` reproduces the run without it.

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
from peaky.batch import sideband as _SB
from peaky.batch import episode_evidence as _EE
from peaky.batch import unidentified as _UN
from peaky.assignment import isoguard as _IG
from peaky.batch import sampling as SS

__version__ = "0.14.1"  # the episode evidence demotes only on a contradicting measurement; one curated set
                        #   (episode_evidence.curated_exempt); batch_summary records episode_evidence_mode
                        # 0.14.0: + the episode evidence in the merge (batch/episode_evidence.py: the `episode` flag on
                        #   the merged row, the single-episode judgment, the plume-solvent clusters; tables/
                        #   episode_evidence.csv, solvent_clusters.csv) and the satellite-artefact hold (sideband.py)
                        # 0.13.0: + the episode stage between the cover and the first assignment (sampling.py,
                        #   tables/episodes.csv, selection["episodes"], the per-file `episode` column; its
                        #   the coverage rule: every episodic bin >= 5x edge gets its peak spectrum, no budget)
                        # 0.12.2: + the vote-drop recorder (tables/vote_drops.csv), the merged isotope-contradiction
                        #   guard (isoguard.guard_merged, merge_gates["isotope_guard"]) and the bright unidentified
                        #   lines (tables/unidentified.csv, batch_summary["unidentified"]; batch/unidentified.py)
                        # 0.12.1: + the isotopologue hold / satellite note of a reading Assigned in <= 2 files (sideband.py,
                        #   tables/sideband_holds.csv, merge_gates["sideband"])
                        # 0.12.0: the vote ranks by `vote_tier` (a file's reagent-N reading with its split open
                        # is Candidate to the label stage), the merged tier is the winning ion's best per-file tier;
                        # the monotone known-species cap (own 13C counts; an exact-mass-only pool of a
                        # corroboration-demanding species is capped)
                        # 0.11.1: + tables/unexplained_blocked.csv (why the unexplained lines are unexplained, blocked.py)
                        # 0.11.0: the merged-ledger isotopologue gate and the element-signature
                        # removal (one curated exempt set, reconciled: 'parent removed';
                        # the per-file ledgers record both decisions -- the gate's lines,
                        # a removed reading released --, the parent's merged row lists
                        # its lines: `isotopologue_lines`),
                        # the batch's resolved class in every isotope check, the n = 0
                        # reagent core, context_flags and batch_summary['warnings']
                        # (0.10.0: the vote reads the per-file EVIDENCE: a cluster's ions are
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
            "vote_class",   # the reading's vote class (0 / 1 / 2, evidence.vote_classes; computed in the parent,
                            # never written to a ledger): `_vote` ranks ions by it before the count, when present
            "tier_calibration",   # where the winner file's tier calibration came from (per_file / pooled /
                                  # none: tiers.TIER_CALIBRATION, assignment/calpool.py), when present
            "episode"]   # the episode stage's row flag (sampling.episode_row_flags): the merged row of a bin
                         # any of whose files' rows is flagged is flagged (batch/episode_evidence.py reads it),
                         # when present -- a batch with no episode file has no such column

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
# THE TIER THE VOTE RANKS BY (`_m0`, from the file's own tier and tier_reason; never
# written to a ledger or the merged row): the reading's tier, but Candidate where
# the file left the reading's reagent-N split open (tiers.split_open: the nitrogen
# count not fixed). The tier answers "is the ion right?" and a reagent-N pair is one
# ion, so such a reading is Assigned on its ion's evidence -- while the vote's label
# stage asks which LABEL of the ion a file could fix, and the best-row pick keeps
# the row it took before. The merged `tier` is the best per-file TIER of the winning
# ion (any of its labels). Frames without the column vote on the tier exactly as before.
VOTE_TIER = "vote_tier"


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
       One exception to the class: an aromatic-heteroatom neutral at the dimer
       mass (plausibility.aromatic_heteroatom: C >= 20, DBE/C >= 0.5, N/S/P)
       counts as class 0, because its "confirmation" is an isotope fit on a
       sub-ppm alias of an oxygenated dimer cluster (C28H30O7S [M-H]- on
       C20H30O11 [M+^NO3]-).

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
       not by count. The tier no longer says it (it reads the ion, and the pair
       is one ion), so the stage ranks by the vote's tier (`_r`, VOTE_TIER):
       Assigned only where the file's reading is Assigned AND its reagent-N
       split was not left open (an N-free sibling, the joint NH4+urea pair or a
       series anchor fixed the nitrogen count, or the file saw no same-ion
       alias), Candidate where the file had nothing to decide with -- counting
       those files would be counting silence. The label Assigned in the most
       files wins; file count and score break ties. On the 15-file uronium run
       16 of the 43 same-ion splits had a majority label nobody had corroborated
       against a minority label some file had."""
    assigned = g["_r"] >= TIER_RANK[TIER_ASSIGNED]
    # the vote class of every per-file reading (0 everywhere when the frames
    # carry none: a pure `align()` caller without the parent's class)
    if VOTE_CLASS in g.columns:
        cls = pd.to_numeric(g[VOTE_CLASS], errors="coerce").fillna(0).astype(int)
    else:
        cls = pd.Series(0, index=g.index)
    # an aromatic-heteroatom neutral at the dimer mass carries no class: its isotope
    # "confirmation" is no evidence over the oxygenated reading it aliases
    # (plausibility.aromatic_heteroatom), so the count decides between them
    if "_nf" in g.columns:
        from peaky.assignment import plausibility as _PL
        cls = cls.where(~g["_nf"].astype(str).map(_PL.aromatic_heteroatom).astype(bool), 0)
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
    # the TIER (not the vote's tier, `_r`): what the losers' description prints
    tiered = (g["_rt"] if "_rt" in g.columns else g["_r"]) >= TIER_RANK[TIER_ASSIGNED]
    gg = g.assign(_asrc=g["src"].where(assigned),       # the file, when Assigned there
                  _tsrc=g["src"].where(tiered),
                  _reg=(~io).astype(int),                # 1 = a regular reading in this file
                  _cls=cls)
    lab = (gg.groupby(["_ion", "_nf", "_ad"], sort=True)  # text order = last key
             .agg(n_files=("src", "nunique"),
                  n_assigned=("_asrc", "nunique"),        # FILES at Assigned, not rows
                  n_tier_assigned=("_tsrc", "nunique"),
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
    tier = TIER_ASSIGNED if int(r.get("n_tier_assigned", r["n_assigned"])) > 0 else "Candidate"
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
        d = df[[c for c in (*_M0_COLS, VOTE_TIER) if c in df.columns]].dropna(subset=["mz"]).copy()
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
        # `_r` = the vote's tier (VOTE_TIER where a file carries it: the label
        # stage, the ion count's tie-break and the best-row pick read it), `_rt`
        # = the tier itself (the merged tier and the losers' description)
        _rt = g["tier"].map(lambda t: TIER_RANK.get(str(t), 0))
        _vt = (g[VOTE_TIER].where(g[VOTE_TIER].notna(), g["tier"]) if VOTE_TIER in g.columns
               else g["tier"])
        g = g.assign(_r=_vt.map(lambda t: TIER_RANK.get(str(t), 0)), _rt=_rt,
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
            # where a file of this ion left its reagent-N split open, the label stage
            # counted the files Assigned with the split not left open (VOTE_TIER)
            gi = g[g["_ion"] == win_ion["_ion"]]
            how = ("Assigned, nitrogen count not left open, in"
                   if bool((gi["_r"] != gi["_rt"]).any()) else "Assigned in")
            notes.append(f"same ion {win_ion['_ion']} read two ways: kept {win['_nf']} "
                         f"{win['_ad']} ({how} {int(win['n_assigned'])} of its "
                         f"{int(win['n_files'])} files) over the {int(big['n_files'])}-file "
                         f"{big['_nf']} {big['_ad']} ({how} {int(big['n_assigned'])})")
        # the best per-file TIER of the winning ION, whichever label a file read
        # it with (the tier answers "is the ion right?"): the best row's own
        # unless the vote's tier set a file's reagent-N reading back, or a file
        # Assigned the ion under the label the vote did not keep -- then that
        # file's row is the one the tier (and its calibration footing) comes from
        gi_win = g[g["_ion"] == win_ion["_ion"]]
        tier = (TIER_ASSIGNED if int(gi_win["_rt"].max()) >= TIER_RANK[TIER_ASSIGNED] else best["tier"])
        tier_row = best if str(best["tier"]) == str(tier) else (
            gi_win[gi_win["_rt"] == gi_win["_rt"].max()]
            .sort_values(["ion_score", "src"], ascending=[False, True], kind="mergesort").iloc[0])
        mz_raw = g["mz"].to_numpy(); mz_adj = g["_mz_adj"].to_numpy()
        def _spread(a):
            return float((a.max() - a.min()) / a.mean() * 1e6) if len(a) > 1 else 0.0
        forms = set(g["neutral_formula"].dropna())
        srcs = sorted(set(g["src"]))
        rec = dict(
            mz=float(g["mz"].mean()), neutral_formula=best["neutral_formula"],
            adduct=best.get("adduct"),
            tier=tier,
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
        if "tier_calibration" in g.columns:
            # the tier's footing: where the calibration of the file whose row
            # donated it came from (its own core, the batch's pooled core, none)
            rec["tier_calibration"] = tier_row.get("tier_calibration", pd.NA)
        if "episode" in g.columns:
            # the episode stage's flag, per bin: any file's row of it flagged
            rec["episode"] = bool(g["episode"].map(_EE.truthy).any())
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


#: the per-file stats keys `ledger.stats` computes: recounted on a ledger the batch rewrote
_ROLE_STATS = ("n_peaks", "by_role", "signal_by_role", "count_frac_by_role", "n_synthetic", "by_confidence",
               "by_tier")


def _recount_roles(st: dict, ledger: pd.DataFrame, gate: dict | None = None, *,
                   signature: dict | None = None) -> dict:
    """A file's stats (its batch_summary['per_file'] entry) after the batch rewrote
    its ledger: the role / signal / confidence / tier counts, `n_M0` and the
    resolvability class counts recounted from the rewritten ledger (`ledger.stats`,
    the M0 rows' `resolvability`), what the isotopologue rewrite did under
    `isotopologue_gate` and what the element-signature release did under
    `element_signature_gate`. Everything else stays the file's own run's record."""
    from peaky.assignment import ledger as L
    new = L.stats(ledger)
    for k in _ROLE_STATS:
        if k in new:
            st[k] = new[k]
    st["n_M0"] = int((ledger["role"] == L.ROLE_M0).sum()) if "role" in ledger.columns else st.get("n_M0")
    if "resolvability" in ledger.columns and "role" in ledger.columns and st.get("resolvability") is not None:
        # the class counts of the M0 rows (resolvability.stamp_resolvability), recounted
        cls = ledger.loc[ledger["role"] == L.ROLE_M0, "resolvability"].dropna()
        st["resolvability"] = {str(k): int(v) for k, v in cls.value_counts().to_dict().items()}
    if gate:
        st["isotopologue_gate"] = dict(gate)
    if signature:
        st["element_signature_gate"] = dict(signature)
    return st


def _m0(ledger: pd.DataFrame) -> pd.DataFrame:
    """Extract the M0 (assigned-compound) rows in the _M0_COLS schema, plus the
    vote's private `VOTE_TIER` where the ledger carries tier and tier_reason."""
    role = ledger["role"] if "role" in ledger.columns else None
    m = ledger[role == "M0"] if role is not None else ledger
    cols = [c for c in _M0_COLS if c in m.columns]
    out = m[cols].copy()
    if {"tier", "tier_reason"} <= set(m.columns):
        from peaky.assignment import tiers as _TI
        out[VOTE_TIER] = [TIER_CANDIDATE if _TI.split_open(why) and _s(t) == TIER_ASSIGNED else t
                          for t, why in zip(m["tier"], m["tier_reason"])]
    return out


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


#: the label of a row's own 13C line (one line; a '13C+34S' line is the 34S line's)
_C13_LABEL = "13C"


def _own_c13(isotopologues, kid_labels=()) -> int:
    """1 when the row records its own 13C line -- an `isotopologues` entry or an
    iso_child labelled exactly '13C' -- else 0. It counts toward the known
    species' two lines (`lock_known_species`): the carbon count is the
    composition's own, a line no mass fit forges."""
    try:
        entries = (json.loads(isotopologues) if isinstance(isotopologues, str)
                   else (isotopologues or []))
    except (TypeError, ValueError):
        entries = []
    labs = [str(d.get("label", "")) for d in entries if isinstance(d, dict)]
    labs += [str(k[1] if isinstance(k, tuple) else k) for k in kid_labels]
    return int(any(lab.strip() == _C13_LABEL for lab in labs))


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
    recorded own-element diagnostic lines), n_c13 (1 when its own 13C line is
    recorded, `_own_c13`), ion_score, tier, admitted_by, occurrence (the last
    three None on a lead)."""
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
                n_c13=_own_c13(iso_cell, kid_labels),
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
                n_channels=int(d.get("channels") or 0), n_satellites=0, n_c13=0,
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
      * TWO independent lines of evidence beyond the exact mass, in some
        confirming file, are a second ion channel or two of the neutral's own
        lines -- its diagnostic satellites and its own 13C line (`n_channels
        >= 2 or n_satellites + n_c13 >= 2`). A species that LOCKS over the
        vote's reading takes the confirmed files' best tier only with two
        lines, whatever its family; with fewer it is locked at Candidate (an
        exact-mass-only species never locks, above). Where the vote already
        reads the species, the cap applies to a species whose pass-0 rule
        demands corroboration (directors.known_demands_corroboration: the
        organophosphates, the organo(thio)phosphates, the indoor organosulfur,
        the cyclosiloxanes, the chlorinated paraffins, the silanediols -- their
        Si M+1 gate --, H2SO4 and MSA): one line in every confirming file is
        capped at Candidate (the D7 cyclosiloxane: two files, one channel, the
        29Si line alone, the 30Si line never testable at that intensity --
        J12), and so is exact mass alone in every one. The cap is MONOTONE: a
        pool with more confirming lines never ranks below one with fewer (a
        pure exact-mass H2SO4 pool had escaped it while one with a 34S line in
        a file was capped). A species whose family's rule is exact mass (the
        PFCAs, the nitroaromatics, the C0 acids, reactive iodine) is never
        capped where the vote reads it, at any line count.

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
    from peaky.assignment.passes.directors import known_demands_corroboration
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
        # the cap's domain is the species' own rule (monotone: the lines it counts
        # are a second channel, the neutral's diagnostic satellites and its 13C)
        demands = known_demands_corroboration(
            next((str(r.get("family")) for r in recs if r.get("family")), ""), nf)
        lines = max((2 if int(r.get("n_channels") or 0) >= 2
                     else int(r.get("n_satellites") or 0) + int(r.get("n_c13") or 0) for r in conf), default=0)
        two_lines = lines >= 2
        # the cap's note names what it counted: a diagnostic satellite, the
        # row's own 13C line where no confirming file recorded a satellite, or
        # no line at all (exact mass, or a corroboration the rows did not record)
        if lines >= 1:
            one_line = ("one satellite line" if any(int(r.get("n_satellites") or 0) for r in conf)
                        else "its own 13C line alone")
            weak_note = (f"; capped Candidate (one ion channel and {one_line} in every "
                         "confirming file: two independent lines are needed for Assigned)")
        else:
            weak_note = ("; capped Candidate ("
                         + ("exact mass alone in every confirming file" if mass_only else
                            "no second ion channel or own isotope line recorded in any confirming file")
                         + (": the species' rule demands corroboration, and two independent lines are "
                            "needed for Assigned)" if demands else
                            ": two independent lines are needed to override the vote)"))
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
                capped = (demands and not two_lines and "tier" in merged.columns
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
            # overriding the vote takes two lines whatever the species' family: a
            # species outside the cap's domain (exact mass is its rule) that
            # displaces the majority reading on one line is the KNOWN-LOCK-DECOY
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


def _assign_one(sid: str, keep_pool_inputs: bool = True) -> dict:
    """Top-level (spawn-picklable) worker: assign ONE sample and return the
    picklable pieces the parent reduces. The parent writes the per-file CSV and
    computes the offset -- in sample order -- so all disk I/O and the align() input
    order stay single-writer and deterministic. `keep_pool_inputs`: the run keeps
    what a batch-pooled tier calibration re-applies (assign.run; off once the
    batch's pool is known to have no calibration)."""
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
                log=lines.append, keep_pool_inputs=keep_pool_inputs, **kw)
    return {"sid": sid, "ledger": res["ledger"],
            "plausibility_audit": res.get("plausibility_audit") or [],
            "stats": dict(res.get("stats", {})), "log": lines,
            # what the worker scored at: a mass trend its calibration set lives
            # in the worker's process only (C42), so the parent cannot re-read it
            "pattern_scoring": res.get("pattern_scoring"),
            # an uncalibrated file's inputs to the batch-pooled tier calibration
            # (calpool.apply_pool); None on a calibrated file
            "pool_inputs": res.get("pool_inputs")}


def _tier_cal_source(per_stats: list) -> str | None:
    """The batch's tier calibration in one word (batch_summary
    `tier_calibration.source`): the weakest of its files' `tier_calibration` --
    'none' when any file stayed uncalibrated, else 'lent' / 'campaign' when any
    file was lent a calibration file's window (assignment/calfile.py), else
    'pooled' when any file was lent the batch's pooled core
    (assignment/calpool.py), else 'per_file'; None when no file records one (a
    run made before the key existed)."""
    seen = {x.get("tier_calibration") for x in per_stats} - {None}
    for word in ("none", "lent", "campaign", "pooled", "per_file"):
        if word in seen:
            return word
    return None


def _levels_calibration(per_stats: list, record, file_state: dict, cfg) -> dict:
    """batch_summary `levels_calibration` (calfile.levels_record) plus the files
    the calibration file was lent to and the batch facts it was matched on."""
    from peaky.assignment import calfile as _CF
    out = _CF.levels_record(per_stats, record or None, used_as=file_state.get("as"),
                            reason=file_state.get("reason"),
                            lent_flag=bool(getattr(cfg, "tier_calibration_lent", False)))
    if out["file"] is not None:
        out["file"]["decided"] = bool(file_state.get("decided"))
        out["file"]["batch_facts"] = file_state.get("facts")
        out["file"]["lent_files"] = dict(file_state.get("files") or {})
        out["file"]["transfer"] = file_state.get("transfer")
    return out


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
#: `run(axis_fingerprints=)`: 'auto' reads every item's served Orbitrap axis fingerprint
#: (batch.axisframe) in front of the lock model on an Orbitrap batch; 'refresh' reads it
#: from the server again, past the cached copy (a re-fingerprinted file: a new reader or
#: a backfill), and caches the final, readable documents it gets; 'off' never asks for
#: one and reproduces a run without the frame.
AXIS_FINGERPRINT_MODES = ("auto", "refresh", "off")
#: the largest correction the LOCK model applies. Its corrections are evidence-backed
#: by hundreds of locks, so it may exceed massqc.MAX_CORRECTION_PPM (sized for a wave
#: fitted on ~25 reference ions); the bound left is the PDF report's: its coverage
#: tables match the ledger to the run's recorded, uncorrected series within 8 ppm
LOCK_MAX_CORRECTION_PPM = 7.0
#: the lock model is predictive when it predicts held-out locks to within this (ppm), or
#: half their raw error: a raw median alone read a calibrated body with a stepped top as
#: 'unpredictive' (raw 0.02 ppm) and left 2-3.5 ppm steps uncorrected
LOCK_CV_OK_PPM = 0.3
#: the lock model is applied once its largest correction (or the locks' raw median) reaches
#: this (ppm): about 0.7x the Orbitrap score width (0.58 ppm), where a pattern-perfect M0
#: already drops below Good. The reference ions' 1 ppm bar (massqc.ORBI_OFFSET_FLAT_PPM)
#: left +0.5..+0.8 ppm plateaus uncorrected, and on a labelled-nitrate batch the whole
#: correction hung on one lock reading 1.08 ppm against it
LOCK_APPLY_PPM = 0.4
#: a correction under massqc.ORBI_OFFSET_FLAT_PPM is applied only when the model predicts
#: held-out locks to within this share of it: the predictive bar above would let a model
#: with a 0.3 ppm held-out error correct a 0.4 ppm axis, i.e. add about as much as it removes
LOCK_SMALL_CV_SHARE = 0.5
#: a model applied through that sub-ppm branch is refitted, after its verdict, without the
#: locks that sit more than this (ppm) off a Theil-Sen line through their axislock.LOCAL_K
#: nearest locks, dropped one at a time, the worst first (axislock.AxisModel.refit; spline
#: segments only, every domain as built). A few co-located locks 0.6-0.9 ppm off their
#: neighbours (a lock on a line blended with a brighter ion's 13C line, its labelled alias)
#: bumped a labelled-nitrate batch's spline by +0.1..+0.15 ppm over m/z 261-269 -- +0.23 ppm
#: of correction at a true cluster line that then fell out of Good in every file. 0.4 starves
#: the GCV fit there (its smoothing collapsed 297 -> 47); 0.75 leaves the bump
LOCK_REFIT_PPM = 0.5
#: a model applied through that sub-ppm branch carries the axis at its first and its last
#: lock (the model there, put back on its end locks: axislock.EDGE_LOCK_DA) into the scoring
#: centre of the lines up to this far (Da) beyond them
#: (axislock.AxisModel.edge_carry_record; masscal.edge_carry_at). Peaks outside the
#: scope keep their measured m/z while a file's centres are fitted on the corrected
#: ones, so without it a line just past the last lock was judged against a centre read
#: off the other axis (an EasyIC batch: the C9-C12 hydrocarbons above it, 0.7 ppm off,
#: out of Assigned). Measured against an independent axis: within 30 Da of the edge its
#: value is the axis to ~0.1 ppm on the flat EasyIC axes; further out a curved
#: labelled-nitrate axis bends away from it (0.2 -> 0.4 ppm), so past the reach, and in
#: the gaps between segments, nothing is carried. A model at 1 ppm or more -- one the
#: reference bar applied before the sub-ppm branch existed -- carries nothing: those
#: batches run as before
EDGE_CARRY_DA = 30.0


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


#: `to_server_axis` on a frame: the spread (ppm) between the sources' inverses of one
#: merged row above which the row is counted as disagreeing in its log line.
FRAME_INVERSE_SPREAD_PPM = 0.05


def to_server_axis(mz, axis_info, srcs=None, *, log=None, stats: dict | None = None):
    """Corrected m/z back on the server's axis (massqc.invert_correction: segment by
    segment for a lock model). `axis_info` is a run's batch_summary['mass_axis'];
    unchanged when it applied nothing. A batch's merged ledger is matched to the
    server's batch peaks by m/z (`peaky publish --batch`), so it must travel on the
    server's axis.

    A run that applied a band-step frame (`axis_info['wave']['model'] == 'frame'`,
    batch.axisframe) corrected each sample item on its own map, so a merged row has
    no single inverse: `srcs` (one entry per row, the merged ledger's comma-joined
    sample item ids) is REQUIRED then (TypeError without it), each row is put back
    by every source item's map and the median taken; the spread between the
    sources' inverses is logged (`log`, else loguru) and, with `stats` (a dict),
    recorded in it (`n_rows`, `n_multi_source`, `spread_ppm_max`,
    `n_spread_over`, `n_without_item`, `n_no_preimage` (rows a source's map leaves
    in a gap at a zone boundary: unchanged under it, up to half the band step off
    the server's axis), `n_several_preimages` (rows with two preimages under a
    source's map: the lower m/z taken) and `spread_ppm`: every row's spread as an
    array, for a caller that warns at its own threshold -- publish.load_batch_run
    at half the match tolerance). Under one calibration group every item's
    map is the same, so the inverses agree and the row comes back exactly; a row
    whose sources the frame does not hold is returned unchanged and counted."""
    from peaky.batch import massqc as MQ
    mz = np.asarray(mz, dtype=float)
    if not axis_info or not axis_info.get("applied") or not axis_info.get("wave"):
        return mz
    fit = MQ.fit_from_record(axis_info["wave"])
    if not hasattr(fit, "for_item"):
        return MQ.invert_correction(fit, mz)
    if srcs is None:
        raise TypeError("to_server_axis: the run applied a band-step frame, which corrects each sample item on "
                        "its own map; pass the rows' `srcs` (comma-joined sample item ids) to put them back")
    return _frame_to_server_axis(fit, mz, srcs, log, stats)


def _frame_to_server_axis(frame, mz, srcs, log, stats):
    """The frame's inverse per merged row: the median over its source items' inverses
    (`FrameItem.invert`), the spread between them counted (see `to_server_axis`)."""
    srcs = list(srcs)
    if len(srcs) != len(mz):
        raise ValueError(f"to_server_axis: {len(srcs)} srcs for {len(mz)} m/z")
    rows = []
    for s in srcs:
        if s is None or (isinstance(s, float) and np.isnan(s)):
            rows.append(())
        else:
            rows.append(tuple(x.strip() for x in str(s).split(",") if x.strip()))
    known = sorted({sid for row in rows for sid in row if sid in frame.items})
    inv, found, several = {}, {}, {}
    for sid in known:
        inv[sid], found[sid], several[sid] = frame.for_item(sid).preimage(mz)
    out = mz.copy()
    spread = np.zeros(len(mz))
    n_multi = n_without = n_no_preimage = n_several = 0
    for i, row in enumerate(rows):
        vals = [inv[sid][i] for sid in row if sid in inv]
        if not vals:
            n_without += 1
            continue
        if any(not found[sid][i] for sid in row if sid in inv):
            n_no_preimage += 1           # some source's map leaves this value in a gap: unchanged under it
        if any(several[sid][i] for sid in row if sid in inv):
            n_several += 1               # some source's map has two preimages here: the lowest m/z taken
        v = np.array(vals, dtype=float)
        v = v[np.isfinite(v)]
        if not len(v):
            continue
        out[i] = float(np.median(v))
        if len(v) > 1:
            n_multi += 1
            spread[i] = float((v.max() - v.min()) / out[i] * 1e6) if out[i] > 0 else 0.0
    rec = {"n_rows": int(len(mz)), "n_multi_source": int(n_multi),
           "spread_ppm_max": round(float(spread.max()), 4) if len(spread) else 0.0,
           "n_spread_over": int((spread > FRAME_INVERSE_SPREAD_PPM).sum()), "n_without_item": int(n_without),
           "n_no_preimage": int(n_no_preimage), "n_several_preimages": int(n_several)}
    if stats is not None:
        stats.update(rec)
        stats["spread_ppm"] = spread
    msg = (f"[mass-axis] frame inverse: {rec['n_rows']} rows put back on the server's axis by their source "
           f"items' maps ({rec['n_multi_source']} from several items, the inverses spread by up to "
           f"{rec['spread_ppm_max']:.3f} ppm, {rec['n_spread_over']} rows by more than "
           f"{FRAME_INVERSE_SPREAD_PPM:g}; {rec['n_without_item']} rows without an item in the frame, unchanged; "
           f"at a zone boundary {rec['n_no_preimage']} rows had no preimage under a source's map (left where they "
           f"were under it) and {rec['n_several_preimages']} had two (the lower m/z taken))")
    if log is not None:
        log(msg)
    else:
        from loguru import logger
        flag = rec["n_spread_over"] or rec["n_without_item"] or rec["n_no_preimage"] or rec["n_several_preimages"]
        (logger.warning if flag else logger.info)(msg)
    return out


def _apply_axis(ts, fit, info, where, log):
    """`ts` with `mz` corrected by `fit` where it reaches (AXIS_COL holds the ppm
    removed), and `info` updated with what moved. A band-step frame (`for_item`,
    batch.axisframe.FrameSet) is applied per sample item by that item's own map;
    an item the frame does not hold (it is all-or-nothing, so none) is left as is."""
    from peaky.batch import massqc as MQ
    mz = ts["mz"].to_numpy(dtype=float)
    if hasattr(fit, "for_item"):
        new = mz.copy()
        shift = np.zeros(len(mz))
        inside = np.zeros(len(mz), dtype=bool)
        sids = ts["sample_item_id"].astype(str).to_numpy()
        for sid in pd.unique(sids):
            sel = np.flatnonzero(sids == sid)
            try:
                it = fit.for_item(sid)
            except KeyError:
                continue
            ins = np.asarray(MQ.in_scope(it, mz[sel]), dtype=bool)
            inside[sel] = ins
            if ins.any():
                shift[sel[ins]] = it.predict(mz[sel[ins]], extrapolate=True)
            new[sel] = MQ.apply_correction(it, mz[sel])
    else:
        inside = MQ.in_scope(fit, mz)
        shift = np.zeros(len(mz))
        if inside.any():
            shift[inside] = fit.predict(mz[inside], extrapolate=True)
        new = MQ.apply_correction(fit, mz)
    ts = ts.copy()
    ts["mz"] = new
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


def axis_reagent_elements(adducts) -> tuple:
    """What the reagent adds to the lock model's formula space (axislock.FormulaSpace):
    its halogen ([M+Br]- ...), and the heavy label of a labelled anion reagent ('^N' for a
    channel holding a '^N' anion adduct: [M+^NO3]-) -- a labelled batch's ions carry it, and
    a space without it reads them as their unlabelled aliases. A labelled cation reagent
    ([M+^NH4]+) is left out: its positive-mode family is not validated against a measured
    axis, and the space's positive DBE rule does not admit all of it."""
    from peaky.assignment import evidence as EV
    halogen = EV.channel_halogen(adducts)
    els = [halogen] if halogen else []
    if any("^N" in str(a) and str(a).strip().endswith("-") for a in adducts or ()):
        els.append("^N")
    return tuple(els)


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
    segs = ", ".join(f"m/z {s['lo']:.0f}-{s['hi']:.0f} ({s['kind']}, {s['n']} locks"
                     f"{', from seeds' if s.get('seeded') else ''})" for s in model.segments)
    if li.get("single_spectrum"):
        log("[mass-axis] one spectrum: no recurrence to test, every peak counts (the isotope "
            "strike and the curve's consistency are the guards)")
    log(f"[mass-axis] LOCKS: {li.get('n_pass1')} unique within +-{AL.LOCK_PPM:g} ppm, "
        f"{li.get('n_links')} by mass difference, {li.get('n_narrow')} on the fitted curve, "
        f"{li.get('n_off_curve')} dropped off it; segments {segs}")
    for c in li.get("contested", []):
        log(f"[mass-axis] m/z {c['lo']:.0f}-{c['hi']:.0f}: a segment from seeds, contested "
            f"({c['seeds_on']} seeds on its line, {c['seeds_off']} off it, {c['jump_ppm']:+.2f} ppm "
            "from the axis below): not built")
    for st in li.get("steps", []):
        log(f"[mass-axis] STEP {st['ppm']:+.2f} ppm between m/z {st['between'][0]:.2f} and "
            f"{st['between'][1]:.2f} (the instrument's axis; no smooth curve follows it)")
    for st in li.get("joins", []):
        why = ("a step under the STEP's bar inside one walk segment" if st.get("split") else
               "no step, the walk resumed past a sparse stretch")
        log(f"[mass-axis] segments join between m/z {st['between'][0]:.2f} and {st['between'][1]:.2f} "
            f"({st['ppm']:+.2f} ppm: {why}): bridged at the midpoint")
    for a, b in li.get("unseen_gaps", []):
        log(f"[mass-axis] no lock between m/z {a:.2f} and {b:.2f} (wider than {AL.JOIN_MAX_DA:g} Da): "
            "the axis in between is unseen")
    if li.get("uncorrected_gaps"):
        log("[mass-axis] left uncorrected between segments (a step, or an unseen stretch): "
            + ", ".join(f"m/z {a:.2f}-{b:.2f}" for a, b in li["uncorrected_gaps"]))
    log(f"[mass-axis] lock model: |error| median {raw} ppm raw -> {cv} ppm on held-out locks "
        f"(5-fold), largest correction {peak} ppm")
    info.update(model="locks", wave=model.as_dict(), scope_mz=list(model.mz_range),
                max_abs_ppm=peak)
    if cv is None or raw is None:
        info["verdict"] = "locks_unvalidated"
        return None, True
    if peak is not None and peak < LOCK_APPLY_PPM and raw < LOCK_APPLY_PPM:
        info["verdict"] = "axis_ok"
        log(f"[mass-axis] axis_ok: no lock reads {LOCK_APPLY_PPM:g} ppm off; nothing to correct")
        return None, True
    if cv > max(LOCK_CV_OK_PPM, 0.5 * raw):
        info["verdict"] = "locks_unpredictive"
        log("[mass-axis] the lock model does not predict held-out locks: not applied")
        return None, True
    small = peak is not None and peak < MQ.ORBI_OFFSET_FLAT_PPM and raw < MQ.ORBI_OFFSET_FLAT_PPM
    if small and cv > LOCK_SMALL_CV_SHARE * peak:
        # a sub-ppm correction its own held-out locks do not resolve: the axis is flat
        # within what the model can tell
        info["verdict"] = "axis_ok"
        log(f"[mass-axis] axis_ok: the largest correction ({peak:g} ppm) is under "
            f"{MQ.ORBI_OFFSET_FLAT_PPM:g} ppm and the model predicts held-out locks only to "
            f"{cv:g} ppm (more than {LOCK_SMALL_CV_SHARE:g}x it); nothing to correct")
        return None, True
    info["verdict"] = "axis_steps" if li.get("steps") else "axis_trend"
    if small:
        # the sub-ppm branch: the verdict above is the as-built model's; the model applied is
        # its splines refitted without the locks that disagree with their neighbours
        # (LOCK_REFIT_PPM; a model that drops nothing is the as-built model)
        built_peak = round(model.max_abs_ppm(), 3)
        model, gone = model.refit(table, off_ppm=LOCK_REFIT_PPM, k=AL.LOCAL_K)
        kept = table.drop(index=gone) if gone else table
        dropped = [{"mz": float(table.at[i, "mz"]), "ion": str(table.at[i, "ion"]) if "ion" in table.columns else "",
                    "ppm": float(table.at[i, "ppm"])} for i in gone]
        if dropped:
            off = {round(d["mz"], 5): d for d in dropped}
            for seg in model.segments:
                if seg.get("refit_dropped"):
                    for d in off.values():
                        if seg["lo"] - AL.REFIT_SEL_DA <= d["mz"] <= seg["hi"] + AL.REFIT_SEL_DA:
                            d["off_ppm"] = round(abs(d["ppm"] - float(model._eval(seg, d["mz"]))), 3)
        refit = {"off_ppm": LOCK_REFIT_PPM, "k": int(AL.LOCAL_K), "n_dropped": len(dropped), "dropped": dropped,
                 "max_abs_ppm_built": built_peak, "max_abs_ppm": round(model.max_abs_ppm(), 3)}
        model.stats["refit"] = refit
        info["refit"] = refit
        log(f"[mass-axis] refit (sub-ppm): {len(dropped)} lock(s) more than {LOCK_REFIT_PPM:g} ppm off their "
            f"{AL.LOCAL_K} nearest dropped, worst first"
            + (": " + ", ".join(f"m/z {d['mz']:.4f} {d['ion']} {d['ppm']:+.2f}" for d in dropped[:12])
               + (" ..." if len(dropped) > 12 else "")
               + f"; the spline refitted (largest correction {built_peak:g} -> {refit['max_abs_ppm']:g} ppm)"
               if dropped else "; the model stands as built"))
        # its correction at the first and the last lock is carried into the scoring centre
        # of the lines up to EDGE_CARRY_DA beyond them (no m/z moves); recorded on the model,
        # so every worker and a re-read run judge alike (the value at each end: the model
        # there, put back on its kept end locks; axislock.EDGE_LOCK_DA)
        rec = model.edge_carry_record(EDGE_CARRY_DA, locks=kept)
        model.stats["edge_carry"] = rec
        info.update(wave=model.as_dict(), edge_carry=dict(rec))
        log(f"[mass-axis] edge carry: a line outside the model's scope within {EDGE_CARRY_DA:g} Da "
            f"of its first / last lock is scored at the axis there, read off its end locks (m/z "
            + "; m/z ".join(f"{e['mz']:.1f}: {e['ppm']:+.2f} ppm (the model {e['model_ppm']:+.2f}, "
                            f"{e['n_locks']} locks)" for e in (rec["below"], rec["above"]))
            + "); no m/z is moved, gaps carry nothing")
    return model, True


def _cap_lock_model(model, info, hold, log):
    """The cap on a lock model's correction (LOCK_MAX_CORRECTION_PPM): a model that
    passes it anywhere is held as a broken calibration -- unless only a walk
    segment's line passes it (its far end): that stretch is cut (AxisModel.clipped)
    and left uncorrected, the rest applied, `info` recording the uncut model
    (`clipped_from`) and the new scope. Returns (model, hold); a hold already given
    stands and skips the check."""
    if hold is None and info["max_abs_ppm"] > LOCK_MAX_CORRECTION_PPM:
        cut = model.clipped(LOCK_MAX_CORRECTION_PPM)
        if cut is None:
            hold = (f"the correction reaches {info['max_abs_ppm']:.1f} ppm, above "
                    f"{LOCK_MAX_CORRECTION_PPM:g}: a broken calibration to fix at the instrument")
        else:
            # a walk segment's line past the cap (its far end): that stretch is left
            # uncorrected, the rest applied
            log(f"[mass-axis] the correction passes {LOCK_MAX_CORRECTION_PPM:g} ppm in a walk "
                f"segment (up to {info['max_abs_ppm']:.1f}): that stretch left uncorrected")
            info["clipped_from"] = info["wave"]
            model = cut
            info.update(wave=model.as_dict(), scope_mz=list(model.mz_range),
                        max_abs_ppm=round(model.max_abs_ppm(), 3))
            li = info.get("locks") or {}
            li["uncorrected_gaps"] = sorted((li.get("uncorrected_gaps") or []) + model.stats["clipped"])
            li["max_abs_ppm"] = info["max_abs_ppm"]
    return model, hold


def _fit_summary(info: dict, model) -> dict:
    """What one lock fit found, for the record of the two fits a framed batch makes
    (the raw one, today's model and the fallback; the one in the W frame): verdict,
    held-out and raw errors, the largest correction and the scope after the cap, the
    lock counts, the steps and the stretches left uncorrected."""
    li = info.get("locks") or {}
    wave = info.get("wave") if isinstance(info.get("wave"), dict) else {}
    return {"verdict": info.get("verdict"), "has_model": model is not None,
            "cv_ppm": li.get("cv_ppm"), "raw_ppm": li.get("raw_ppm"), "max_abs_ppm": info.get("max_abs_ppm"),
            "scope_mz": list(model.mz_range) if model is not None else info.get("scope_mz"),
            "n_locks": wave.get("n_locks"), "n_pass1": li.get("n_pass1"), "n_links": li.get("n_links"),
            "n_narrow": li.get("n_narrow"), "n_off_curve": li.get("n_off_curve"),
            "n_segments": len(model.segments) if model is not None else None,
            "steps": li.get("steps"), "uncorrected_gaps": li.get("uncorrected_gaps"),
            "clipped": "clipped_from" in info, "single_spectrum": li.get("single_spectrum"), "why": li.get("why")}


def _frame_table(frame) -> pd.DataFrame:
    """tables/mass_axis_frame.csv: item x band -- the band's index, its m/z bounds, the
    step the frame took out of it and whether the band was measured."""
    from peaky.batch import axisframe as AF
    rows = []
    for sid, it in frame.items.items():
        ks = np.arange(it.k0, it.k1 + 1)
        lo, hi = AF.band_edges_mz(ks, it.B, it.C, it.period, it.phase)
        rows += [{"sample_item_id": sid, "k": int(k), "mz_lo": round(float(a), 4), "mz_hi": round(float(b), 4),
                  "d_ppm": round(float(d), 4), "measured": bool(m)}
                 for k, a, b, d, m in zip(ks, lo, hi, it.d, it.measured)]
    return pd.DataFrame(rows, columns=["sample_item_id", "k", "mz_lo", "mz_hi", "d_ppm", "measured"])


def _frame_wanted(axis_fingerprints, axis_klass, mass_axis, hold, sample_ids) -> tuple:
    """Whether a run looks for the items' axis fingerprints at all: (True, None) on an
    Orbitrap batch in the lock-model path (`mass_axis` auto / locks) with no hold from
    the caller and some items, `axis_fingerprints` 'auto' or 'refresh'; else (False, why)
    -- nothing is fetched, nothing recorded, and the run is byte for byte the one
    without the feature."""
    if axis_fingerprints == "off":
        return False, "axis fingerprints off (--axis-fingerprints off): the calibration's band steps are not read"
    if axis_klass != "orbitrap":
        return False, "not an Orbitrap batch"
    if mass_axis not in ("auto", "locks"):
        return False, f"--mass-axis {mass_axis}: not the lock-model path"
    if hold is not None:
        return False, f"the axis is held ({hold})"
    if not sample_ids:
        return False, "no sample items"
    return True, None


def _batch_frame(client, sample_ids, polarity, directory, log, *, use_cache: bool = True):
    """The batch's band-step frame from every item's served Orbitrap axis fingerprint
    (io_mascope.fetch_axis_fingerprint; `directory` reads `<directory>/<id>.json`
    instead of the server; `use_cache` False (--axis-fingerprints refresh) reads past
    the cached copy), all or nothing (batch.axisframe.FrameSet.from_documents: every
    item needs a usable document of the batch's polarity and the documents must form
    ONE calibration group). The items are asked in turn until one has no document: the
    frame is then ruled out, so the rest are not asked (each request to a server that
    is down or rate-limiting waits out the SDK's retries) and are recorded as not asked.
    Returns (FrameSet or None, record): the record carries every item's document
    verdict (and why none was served, `fetch`), the groups found, how the documents
    were read (`source`: 'sdk', 'route' for an SDK without the method, 'directory',
    'none'; `refreshed` True when the cache was passed over) and, with no frame,
    `held` -- 'documents unreadable (<exception type>)' when reading them raised
    (`_unread_record`): a document is a reason, never a stop."""
    from peaky.batch import axisframe as AF
    from peaky.io import io_mascope as IO
    source = IO.axis_fingerprint_source(client, directory)
    if source == "route":
        log("[mass-axis] the installed mascope-sdk has no samples.get_axis_fingerprint (mascope-sdk >= "
            "2026.10.28 has it): reading the axis-fingerprint route directly")
    if not use_cache and directory is None:
        log("[mass-axis] axis fingerprints refreshed from the server (--axis-fingerprints refresh): "
            "the cached documents are read again, and every final, readable one served "
            "replaces its copy")
    docs, missing = {}, {}
    first, n_not_asked = None, 0      # the first item with no document rules the frame out
    for sid in sample_ids:
        if first is not None:
            docs[sid] = None
            missing[sid] = f"not asked: item {first} had no document"
            n_not_asked += 1
            continue
        said: list = []
        docs[sid] = IO.fetch_axis_fingerprint(client, sid, directory=directory, log=said.append, use_cache=use_cache)
        said = [line.replace("[axis-fingerprint] ", "") for line in said]
        for line in said:
            log(f"[mass-axis] fingerprint: {line}")
        if docs[sid] is None:
            missing[sid] = said[-1] if said else "no document"
            first = sid
    if n_not_asked:
        log(f"[mass-axis] fingerprint: {n_not_asked} later item(s) not asked: item {first} had no document "
            "and the frame needs every item's")
    try:
        frame, record = AF.FrameSet.from_documents(docs, polarity=polarity)
    except Exception as exc:          # noqa: BLE001 -- a document the reader cannot take holds the frame, not the batch
        frame, record = None, _unread_record(docs, exc)
    record["source"] = source
    if not use_cache:
        record["refreshed"] = True
    for sid, why in missing.items():
        if sid in record["items"]:
            record["items"][sid]["fetch"] = why
            if why.startswith("not asked"):
                record["items"][sid]["why"] = "not asked"
    n_ok, n = record.get("n_usable", 0), len(sample_ids)
    if frame is None:
        log(f"[mass-axis] axis fingerprints: {n_ok} of {n} items usable; no frame: {record.get('held')}")
    else:
        g = record["groups"][0] if record.get("groups") else {}
        log(f"[mass-axis] axis fingerprints: {n_ok} of {n} items usable, one calibration group "
            f"({g.get('n_measured')} measured bands of {g.get('n_bands')}, largest band step "
            f"{g.get('h_max_ppm')} ppm, edge zone {(g.get('grid') or {}).get('zone_khz')} kHz)")
    return frame, record


def _unread_record(docs: dict, exc: Exception) -> dict:
    """The build record when FrameSet.from_documents raised on the served documents
    (load_document refuses every malformed document it knows of; this is the backstop):
    no frame, every item unusable, a served document's status and sha1 kept (the run
    records that it was read), `held` naming the exception's type only."""
    from peaky.batch import axisframe as AF
    items = {}
    for sid, doc in docs.items():
        status = doc.get("status") if isinstance(doc, dict) else None
        items[str(sid)] = {"usable": False, "status": status if isinstance(status, str) else None,
                           "mode_id": None, "sha1": AF.document_sha1(doc) if isinstance(doc, dict) else None,
                           "notes": [], "why": "no document served" if doc is None else "not read"}
    return {"items": items, "n_usable": 0, "n_items": len(docs), "policy": AF.OUTSIDE_POLICY,
            "n_groups": 0, "groups": [], "held": f"documents unreadable ({type(exc).__name__})"}


#: gate (e): the frame is held when the W fit's held-out error passes this multiple of the raw fit's
FRAME_QUALITY_CV_RATIO = 1.5


def _frame_path(ts, frame, frame_record, model_raw, caller_hold, raw_hold, info, polarity, reagent_elements, log):
    """The band-step frame in front of the lock model (batch.axisframe), tried AFTER the
    raw fit, which stays the fallback: the series in the W frame (every item's band
    steps taken out), the lock model fitted there UNCHANGED (`_lock_model`), today's
    cap on the W model, then the gate -- (a) a W step beside a grid edge, (b) a
    residual lock step across an edge, (c) a verdict the frame may not be applied
    under (axisframe.gate), (d) REACH: the W model's applied scope shorter than the
    raw model's by more than one band width at either end, (e) QUALITY: the W fit's
    held-out error above FRAME_QUALITY_CV_RATIO x the raw fit's (both finite). Every
    rule passing, the composite (frame + W model) is applied per item inside the W
    model's scope and `info` describes it (`model` 'frame', `scope_mz` the W scope);
    under the W verdict axis_ok (nothing smooth to correct) the W model is the W fit's
    own lock model set to zero (axisframe.scope_model): the band steps alone, inside the
    W fit's lock domains, never beyond them. Any failure holds the frame
    (info['frame']['held'] names the rule) and returns None, today's path then
    standing. Both fits are recorded (info['frame']['fits']). A hold from the caller
    (a pool of several batches, server-side scoring) holds the frame too; the raw
    model's own cap hold (`raw_hold`) does not -- the frame may be what makes the
    smooth part correctable."""
    from peaky.batch import axisframe as AF
    fr: dict = {"policy": frame.policy, "n_items": len(frame.items), "n_groups": len(frame.groups)}
    if frame_record:
        fr["source"] = frame_record.get("source")
        fr["documents"] = frame_record.get("items")
        if frame_record.get("refreshed"):
            fr["refreshed"] = True
    fr["groups"] = list(frame.groups)
    info["frame"] = fr

    def held(why):
        fr["held"] = why
        log(f"[mass-axis] frame NOT applied: {why}")
        return None

    if caller_hold is not None:
        return held(caller_hold)
    try:
        tsW = frame.to_frame(ts)
    except Exception as exc:          # noqa: BLE001 -- a diagnostic never stops the batch
        return held(f"the frame could not map the series ({type(exc).__name__}: {exc})")
    per = frame.stats.get("frame", {})
    it0 = next(iter(frame.items.values()))
    d_lo = min((p["d_ppm"]["min"] for p in per.values() if p["d_ppm"]["min"] is not None), default=0.0)
    d_hi = max((p["d_ppm"]["max"] for p in per.values() if p["d_ppm"]["max"] is not None), default=0.0)
    log(f"[mass-axis] frame: one calibration group over {len(frame.items)} spectra, {len(it0.d)} bands "
        f"(k {it0.k0}-{it0.k1}, {int(it0.measured.sum())} measured), edge zone {it0.zone_khz:.3f} kHz; band "
        f"steps of {d_lo:+.2f}..{d_hi:+.2f} ppm taken out of {sum(p['n_moved'] for p in per.values())} peaks "
        f"({sum(p['n_edge_ambiguous'] for p in per.values())} in an edge zone, "
        f"{sum(p['n_unmeasured_band'] for p in per.values())} on unmeasured bands); fitting the lock model "
        "in the W frame")
    infoW: dict = {"applied": False, "verdict": None}
    modelW, doneW = _lock_model(tsW, polarity, reagent_elements, infoW, log)
    locksW = infoW.pop("_locks_table", None)
    holdW = None
    if modelW is not None:
        modelW, holdW = _cap_lock_model(modelW, infoW, None, log)
    fr["fits"] = {"raw": _fit_summary(info, model_raw), "W": _fit_summary(infoW, modelW)}
    if not doneW:
        return held(f"no lock model in the W frame ({(infoW.get('locks') or {}).get('why')})")
    if holdW is not None:
        return held(f"the W model: {holdW}")
    inner = modelW
    if modelW is None and infoW.get("verdict") == "axis_ok":
        # nothing smooth to correct: the band steps alone, but only where the W fit saw locks
        inner = AF.scope_model(infoW.get("wave"), infoW.get("scope_mz"))
        if inner is None:
            return held("the W fit's verdict is axis_ok but it records no lock scope to bound the band steps")
    composite = frame.with_inner(inner)
    g = AF.gate(modelW, infoW, locksW, composite)
    failed = list(g.get("failed") or [])
    why = g.get("why")
    # (d) reach: the frame must not shorten what today's model would have corrected
    lo_w, hi_w = (float(x) for x in composite.mz_range)
    raw_applied = model_raw is not None and raw_hold is None
    d_rec: dict = {"ok": True, "raw_applied": raw_applied, "raw_scope_mz": None, "W_scope_mz": [lo_w, hi_w]}
    if raw_applied:
        lo_r, hi_r = (float(x) for x in model_raw.mz_range)
        w_lo = float(np.asarray(it0.band_width_at(np.array([lo_r])))[0])
        w_hi = float(np.asarray(it0.band_width_at(np.array([hi_r])))[0])
        d_rec.update(raw_scope_mz=[lo_r, hi_r], band_width_mz={"at_lo": round(w_lo, 3), "at_hi": round(w_hi, 3)})
        short = []
        if hi_w < hi_r - w_hi:
            short.append(f"the upper end {hi_w:.1f} against {hi_r:.1f} (band {w_hi:.1f} wide there)")
        if lo_w > lo_r + w_lo:
            short.append(f"the lower end {lo_w:.1f} against {lo_r:.1f} (band {w_lo:.1f} wide there)")
        if short:
            d_rec["ok"] = False
            failed.append("d")
            why = why or ("the W model reaches less than the raw model by more than one band: "
                          + "; ".join(short))
    g["d"] = d_rec
    # (e) quality: the frame must beat nothing
    cv_raw = (info.get("locks") or {}).get("cv_ppm")
    cv_w = (infoW.get("locks") or {}).get("cv_ppm")
    e_rec: dict = {"ok": True, "cv_raw_ppm": cv_raw, "cv_W_ppm": cv_w, "ratio_max": FRAME_QUALITY_CV_RATIO}
    if (cv_raw is not None and cv_w is not None and np.isfinite(cv_raw) and np.isfinite(cv_w)
            and cv_w > FRAME_QUALITY_CV_RATIO * cv_raw):
        e_rec["ok"] = False
        failed.append("e")
        why = why or (f"the W fit's held-out error {cv_w} ppm exceeds {FRAME_QUALITY_CV_RATIO:g} x the raw "
                      f"fit's {cv_raw} ppm")
    g["e"] = e_rec
    g["failed"] = sorted(set(failed))
    g["passed"] = not g["failed"]
    g["why"] = None if g["passed"] else why
    fr["gate"] = g
    if not g["passed"]:
        return held(f"gate ({', '.join(g['failed'])}): {why}")
    # apply the composite per item
    info["verdict"] = infoW["verdict"]
    n_moved = sum(p["n_moved"] for p in per.values())
    if modelW is None:
        where = (f"per sample item: the calibration's band steps alone inside the W fit's lock range m/z "
                 f"{lo_w:.0f}-{hi_w:.0f} (axis_ok: no smooth correction; {n_moved} peaks moved in the W frame); "
                 "outside it left as is")
    else:
        where = (f"per sample item: the calibration's band steps taken out ({n_moved} peaks moved), then the "
                 f"lock model fitted in the W frame inside m/z {lo_w:.0f}-{hi_w:.0f} ({len(modelW.segments)} "
                 "segment(s)); outside its scope left as is")
    out = _apply_axis(ts, composite, info, where, log)
    info.pop("clipped_from", None)      # the raw model's cut is recorded under frame.fits.raw
    info.update(model="frame", wave=composite.as_dict(), locks=infoW.get("locks"), scope_mz=[lo_w, hi_w],
                max_abs_ppm=round(composite.max_abs_ppm(), 3),
                inner={"model": None if modelW is None else "locks", "verdict": infoW.get("verdict"),
                       "max_abs_ppm": infoW.get("max_abs_ppm"), "scope_mz": infoW.get("scope_mz"),
                       "clipped_from": infoW.get("clipped_from"), "locks": _fit_summary(infoW, modelW)})
    if locksW is not None and len(locksW):
        info["_locks_table"] = locksW.assign(frame="W")
    info["_frame_table"] = _frame_table(frame)
    fr["held"] = None
    fr["items"] = {}
    for sid, it in composite.items.items():
        st = per.get(sid, {})
        m = it.meta
        fr["items"][sid] = {
            "sample_file_id": (m.get("item") or {}).get("sample_file_id"), "mode_id": m.get("mode_id"),
            "document_sha1": m.get("document_sha1"),
            "grid": {"period_khz": it.period, "phase_khz": it.phase, "zone_khz": it.zone_khz,
                     "document": m.get("document_grid")},
            "n_bands": int(len(it.d)), "n_measured": int(it.measured.sum()),
            "n_interp": int(len(it.d) - it.measured.sum()),
            "d_min_ppm": (st.get("d_ppm") or {}).get("min"), "d_max_ppm": (st.get("d_ppm") or {}).get("max"),
            "flags": m.get("mode_flags"), "grid_weak": m.get("grid_weak"), "lock": m.get("lock"),
            "n_peaks": st.get("n_peaks"), "n_moved": st.get("n_moved"),
            "n_unmeasured_band": st.get("n_unmeasured_band"), "n_edge_ambiguous": st.get("n_edge_ambiguous")}
    return out


def measure_axis(ts, reagent: str, klass: str | None, *, hold: str | None = None,
                 mode: str = "auto", polarity: str = "-", reagent_elements=(), log=print,
                 frame=None, frame_record: dict | None = None):
    """The batch's m/z axis and the correction it prescribes. Returns (ts, info, table):

      * `ts` -- the time series with `mz` corrected where the correction reaches
        and the ppm removed in AXIS_COL, when it was applied; else as given;
      * `info` -- the JSON-safe record batch_summary['mass_axis'] keeps (`model`
        'locks', 'frame' or 'reference', the verdict, the model and its scope, how
        many peaks moved and by how much, or why nothing was applied: `held`);
        info['_locks_table'] (popped by the caller) holds every lock and
        info['_frame_table'] (likewise) the frame's band table when one was applied;
      * `table` -- the per-reference-ion probe table, None when not probed.

    On an Orbitrap, 'auto' and 'locks' model the axis from the batch's own peaks
    (batch.axislock: calibration-free locks, a smoothing spline, the steps the
    instrument's axis takes); it is applied when it predicts held-out locks to
    within half the raw error and its largest correction (or the locks' raw
    median) reaches LOCK_APPLY_PPM (0.4 ppm) -- a correction under 1 ppm only when
    held-out locks sit within half of it (LOCK_SMALL_CV_SHARE), and such a model
    carries the axis at its first / last lock (read off its end locks) into the scoring centre of the
    lines up to EDGE_CARRY_DA beyond them (`info['edge_carry']`; no m/z moves). 'auto' falls back
    to the reagent's reference ions (batch.massqc, the instrument's own rules)
    only when the locks could not build a model; 'reference' goes there directly,
    and a TOF always does (it is measured there, never corrected).

    `frame` (a batch.axisframe.FrameSet built from every item's served axis
    fingerprint, one calibration group) puts the calibration's band steps in front
    of the lock model: the raw fit is made first and stays the fallback, then the
    lock model is fitted again on the series in the W frame and, when the gate
    passes (`_frame_path`: no step left at a grid edge, no residual step across one,
    a predictive verdict, no shorter reach, no worse held-out error), the composite
    of the frame and the W model is applied per item (`model` 'frame';
    info['frame'] records the group, the items, both fits and the gate); otherwise
    the frame is held (info['frame']['held']) and today's path stands. The frame
    is tried only in the lock-model path of an Orbitrap. `frame_record` is what
    building the frame found (FrameSet.from_documents' record): with no frame it is
    recorded as info['frame'] (its `held` says why none was built; the per-document
    verdicts under `documents`) when at least one item's document was SERVED,
    whatever its status -- the only key a run that read a document adds to a run that
    did not; a run served nothing (a 404 from an older server, no connection and no
    cached copy, no file in `axis_fingerprint_dir`) records nothing and is byte for byte the run
    without the feature (its log says what was looked for).

    Not measured (info['skipped']) without a time series or an instrument class,
    or -- reference path -- without a reference table for the reagent or when the
    probe fails; the measurement never stops a batch. Measured but NOT applied
    (info['held']) on a TOF, when the caller holds it (`hold`: server-side
    scoring; a pool of several batches), or when the correction would exceed the
    model's cap (LOCK_MAX_CORRECTION_PPM / massqc.MAX_CORRECTION_PPM; a lock model
    whose walk segments alone pass it has those stretches cut and the rest applied)."""
    out, info, table = _measure_axis(ts, reagent, klass, hold=hold, mode=mode, polarity=polarity,
                                     reagent_elements=reagent_elements, log=log, frame=frame,
                                     frame_record=frame_record)
    if "frame" not in info:
        if frame is not None:
            info["frame"] = dict(_frame_build_record(frame_record),
                                 held="the frame applies in the lock-model path of an Orbitrap batch "
                                      "(mode auto / locks) only")
        elif frame_record is not None and _documents_served(frame_record):
            info["frame"] = _frame_build_record(frame_record)
    return out, info, table


def _documents_served(frame_record) -> bool:
    """Whether FrameSet.from_documents' record holds at least one SERVED document (a
    status was read off it, usable or not): the condition for recording the record. A
    run that was served nothing adds no key (byte identity with a run without the
    feature)."""
    items = (frame_record or {}).get("items") or (frame_record or {}).get("documents") or {}
    return any(isinstance(rec, dict) and (rec.get("status") is not None or rec.get("sha1"))
               for rec in items.values())


def _frame_build_record(frame_record) -> dict:
    """FrameSet.from_documents' record as batch_summary['mass_axis']['frame'] carries it:
    the per-document verdicts (usable, status, sha1, notes, why, how fetched) under
    `documents`, the key the applied record uses too (`items` there is the per-item
    FRAME record), so every reader finds them in one place."""
    rec = dict(frame_record or {})
    if "items" in rec:
        rec["documents"] = rec.pop("items")
    return rec


def _measure_axis(ts, reagent, klass, *, hold, mode, polarity, reagent_elements, log, frame, frame_record):
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
        caller_hold = hold
        if model is not None:
            model, hold = _cap_lock_model(model, info, hold, log)
        if frame is not None:
            # the raw fit first, then the frame: the composite when the gate passes,
            # else the raw fit stands (its hold, its verdict, its reference fallback)
            raw_info = dict(info)
            try:
                framed = _frame_path(ts, frame, frame_record, model, caller_hold, hold, info, polarity,
                                     tuple(reagent_elements or ()), log)
            except Exception as exc:      # noqa: BLE001 -- the frame is a refinement: its failure leaves the raw fit
                info.clear()
                info.update(raw_info)
                info["frame"] = dict(_frame_build_record(frame_record),
                                     held=f"the frame path failed ({type(exc).__name__}: {exc})")
                log(f"[mass-axis] frame NOT applied: {info['frame']['held']}")
                framed = None
            if framed is not None:
                return framed, info, None
        if done or mode == "locks":
            if model is None:
                return ts, info, None
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


def scoring_centres(scorings: dict) -> dict:
    """batch_summary['scoring_centres'] from the per-file scoring snapshots
    (`pattern_scoring`): `by_mu_source` (files per centre source: `fitted` anchors,
    the batch's `axis`, a `self`-centre, a `trend`, `inherited`, `assumed_zero`;
    `unrecorded` for a file with no snapshot) and `self` (`{sample id: centre ppm}`
    of the files scored at their self-centre). A self-centred file whose own
    calibration then accepted a mass trend was scored at the trend: it counts as
    `trend` and is not listed (its per-file `self_centre` keeps the measurement).
    The merge does not read the self-centre: it aligns the files on their
    pre-calibration offset (`per_file.offset_ppm`, io_mascope.estimate_offset)."""
    by: dict = {}
    own: dict = {}
    for sid, snap in (scorings or {}).items():
        src = (snap or {}).get("mu_source") or "unrecorded"
        by[src] = by.get(src, 0) + 1
        if src == "self":
            own[sid] = (snap or {}).get("mu_ppm")
    return {"by_mu_source": dict(sorted(by.items())), "self": own}


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
        episodes: bool = SS.EPISODE_DEFAULT, episode_min_x_edge: float = SS.EPISODE_MIN_X_EDGE,
        k_episode_max: int | None = SS.K_EPISODE_MAX, episode_evidence: str = _EE.MODE,
        ts_peaks=None, amine_r_min: float = 0.6,
        n_jobs: int | None = None, rolling_centre: bool = False,
        trace_first: bool = False, resolving_power=None, trace_episodes: bool = False,
        corroborate=None, mass_axis: str = "auto", mass_axis_hold: str | None = None,
        axis_fingerprints: str = "auto", axis_fingerprint_dir: str | None = None,
        isotopologue_rows: bool = True, lists=None,
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

    `episodes` (default `sampling.EPISODE_DEFAULT`) runs the episode stage
    right after the cover (sampling.py, THE EPISODE STAGE, THE COVERAGE RULE):
    every episodic bin peaking at >= `episode_min_x_edge` (5) x its sample's
    noise edge gets its peak spectrum, and its episode its peak file and up to
    two neighbours -- no budget (a coverage guarantee, not a cost bound);
    `episode_evidence` (episode_evidence.MODES; default 'measured') is what an episodic
    reading in <= 3 files the vote made Assigned becomes when its episode's own evidence
    fails (docs/MERGE.md step 6g): under 'measured' it is Candidate only when a measurement
    contradicts it; batch_summary['episode_evidence_mode'] records the mode.
    `k_episode_max` is an optional safety cap, flagged when it binds -- assigned with the cover's files
    (tables/selected_samples.csv role 'episode'); the record is
    batch_summary['selection']['episodes'] and tables/episodes.csv, and the rows
    of an episode's files on an episodic bin carry `episode = True` in the
    per-file ledgers (the column is written only when the batch has such a file).
    Cover path only: the `sample_ids=` (pooled) and trace-first paths skip it.

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
    and the per-file scoring are then read on the corrected axis; a file with no
    anchors the correction reached is scored at zero there, `mu_source` "axis",
    and one it did not reach measures its own centre, "self":
    batch_summary['scoring_centres'] counts the files per source);
    tables/mass_axis_locks.csv the locks, tables/mass_axis.csv the reference
    ions when probed. Trace-first keeps its own wave
    (batch.tracefirst) and 'off' reproduces a run without the step.

    `axis_fingerprints` ('auto', the default, 'refresh' or 'off'): on an Orbitrap batch in
    the lock-model path (no hold), 'auto' reads every item's served axis
    fingerprint (Mascope's per-file measurement of the calibration's band-wise m/z
    steps; io_mascope.fetch_axis_fingerprint, cached beside the peaks) and, when
    every item has a usable document of one calibration, puts the band steps in
    front of the lock model (batch.axisframe; `measure_axis(frame=)`): the raw lock
    fit is made first and kept as the fallback, the lock model is fitted again on
    the step-free series and the composite is applied only when its gate passes.
    batch_summary['mass_axis']['frame'] records the documents, the calibration
    group, both fits and the gate, or why no frame was built or applied (`held`);
    tables/mass_axis_frame.csv the band table and tables/mass_axis_locks.csv's
    `frame` column marks the W fit's locks. With documents served but no frame
    (unusable, several calibrations, the gate) the run is today's plus that `held`
    record; served nothing (an older server, a file that is not an Orbitrap's, no
    connection and no cached copy), on a TOF or under a hold the run is today's byte
    for byte. `axis_fingerprint_dir` reads
    `<dir>/<sample id>.json` instead of the server (offline runs, locally computed
    documents). 'refresh' reads past the cached documents (a file fingerprinted anew
    by a newer reader or a backfill) and caches the final, readable documents it
    gets; 'off' never asks and adds nothing.

    `isotopologue_rows` (default on; CLI --no-isotopologue-gate turns it off): on an
    Orbitrap-class batch with a width model whose time series carries peak areas, a
    merged row (not a curated reading: passes.curated_formulas) whose
    line is another merged ion's (or a reagent ion's) 13C / 18O / 15N / 34S / 37Cl /
    81Br / Si isotopologue at its expected area ratio over the batch leaves the
    merged ledger before the stamp, which then gives the line to the parent
    (iso_checks.satellite_rows; tables/isotopologue_rows.csv,
    merge_gates['isotopologue']). Its pooled pair reads an isotope-check veto in the
    evidence levels (check 'SAT' in tables/iso_checks.csv). The per-file ledgers record
    the line (iso_checks.reconcile_per_file), as they release a reading the
    element-signature removal took out (iso_checks.release_signature_removed); a
    single-sample `peaky assign` has neither."""
    from peaky.assignment import assign as A
    from peaky.assignment import evidence as EV
    from peaky.batch import timeseries as _TSN
    from peaky.io import io_mascope as IO

    if mass_axis not in MASS_AXIS_MODES:      # before any server call or folder
        raise ValueError(f"mass_axis must be one of {MASS_AXIS_MODES}, got {mass_axis!r}")
    if axis_fingerprints not in AXIS_FINGERPRINT_MODES:
        raise ValueError(f"axis_fingerprints must be one of {AXIS_FINGERPRINT_MODES}, got {axis_fingerprints!r}")
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
    episode_univ = None        # the episode stage's universe (sampling.episode_universe), when it ran
    episode_flag_files: list = []   # the files whose episodic rows carry `episode = True`
    n_episode_files = None     # files the episode stage added (None: the stage did not run)
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
        axis_klass = _axis_class(_instrument_of(rp)[0], _roster_class(peaks))
        # The calibration's band steps (batch.axisframe), read from every item's served
        # axis fingerprint and put in front of the lock model: an Orbitrap, the lock-model
        # path, no hold, `axis_fingerprints` 'auto' or 'refresh'. All or nothing: every item
        # needs a usable document of one calibration, else no frame (the record says why).
        frame, frame_record = None, None
        wanted, why_not = _frame_wanted(axis_fingerprints, axis_klass, mass_axis, hold, axis_ids)
        if wanted:
            frame, frame_record = _batch_frame(client, axis_ids, prof.polarity, axis_fingerprint_dir, log,
                                               use_cache=axis_fingerprints != "refresh")
        elif axis_fingerprints == "off":
            log(f"[mass-axis] {why_not}")
        ts_peaks, axis_info, axis_table = measure_axis(
            ts_peaks, prof.name, axis_klass,
            hold=hold, mode=mass_axis, polarity=prof.polarity,
            reagent_elements=axis_reagent_elements(prof.adducts), log=log,
            frame=frame, frame_record=frame_record)
        if axis_table is not None:
            axis_table.to_csv(os.path.join(TAB, "mass_axis.csv"), index=False)
        locks_table = axis_info.pop("_locks_table", None)
        if locks_table is not None and len(locks_table):
            locks_table.to_csv(os.path.join(TAB, "mass_axis_locks.csv"), index=False)
        frame_table = axis_info.pop("_frame_table", None)
        if frame_table is not None and len(frame_table):
            frame_table.to_csv(os.path.join(TAB, "mass_axis_frame.csv"), index=False)
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
        log(f"[assign_batch] {SS.describe(selection)}")
        _warn = SS.k_max_warning(selection)
        if _warn:
            log(f"[assign_batch] WARNING: {_warn}")
        if episodes:
            # THE EPISODE STAGE (sampling.py): THE COVERAGE RULE -- every episodic
            # bin peaking at >= episode_min_x_edge x its sample's edge gets its peak
            # spectrum (and its episode's, + neighbours); no budget, outside the
            # cover's stop rule; k_episode_max is an optional safety cap
            episode_univ = SS.episode_universe(src, tol_ppm=tol_ppm, min_x_edge=episode_min_x_edge)
            esel = SS.select_episode_samples(src, episode_univ, assigned=sample_ids,
                                             k_max=k_episode_max)
            emeta = dict(esel.attrs.get("selection", {}))
            episode_flag_files = list(esel.attrs.get("flag_files") or [])
            SS.episodes_table(emeta).to_csv(os.path.join(TAB, "episodes.csv"), index=False)
            emeta["episodes"] = [{k: v for k, v in e.items() if k not in ("bins", "n_bright")}
                                 for e in emeta.get("episodes") or []]
            selection["episodes"] = emeta
            n_episode_files = int(len(esel))
            if len(esel):
                esel = esel.copy()
                esel.attrs = {}
                esel["pick"] = np.arange(len(sel) + 1, len(sel) + 1 + len(esel))
                esel["episode"] = esel["episode"].astype("Int64")
                _sel_attrs = dict(sel.attrs)
                sel = sel.copy()
                sel.attrs = {}
                sel = pd.concat([sel, esel], ignore_index=True)
                sel.attrs = _sel_attrs
                sample_ids = sample_ids + esel["sample_item_id"].tolist()
            log(f"[assign_batch] {SS.describe_episodes(emeta)}")
            _warn = SS.episode_budget_warning(emeta)
            if _warn:
                log(f"[assign_batch] WARNING: {_warn}")
        sel.to_csv(os.path.join(TAB, "selected_samples.csv"), index=False)
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
    # `lists` names lists to activate by id (--lists / $PEAKY_LISTS), beside the
    # keyword-unlocked ones; an unknown id stops the batch here, before any work
    reflists_active, _tags, rl_record = RL.activate(batch or "", dataset or "", getattr(prof, "label", ""),
                                                    record=True, include=lists)
    # how they were activated (which keyword in which name): the evidence
    # scale's context source, per file and pooled (batch_summary["reflists_context"])
    assign_kw.setdefault("reflists_context", rl_record)
    if reflists_active:
        log(f"[assign_batch] reference lists active: {RL.active_versions(reflists_active)} "
            f"(context {sorted(_tags) or 'contaminants-only'})")
    per_file, offsets, per_stats = {}, {}, []
    full_ledgers: dict = {}    # sid -> its full ledger as written: the merged-ledger
                               # isotopologue gate's decision is recorded there after
                               # the merge (iso_checks.reconcile_per_file)
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
    # the batch-pooled tier calibration (assignment/calpool.py): decided ONCE, the
    # first time a per-file stage returns a file its own core could not calibrate,
    # from every per-file ledger assigned so far; then lent to every such file
    pool_ledgers: dict = {}        # sid -> the per-file stages' ledgers, as their runs returned them
    pool_cals: dict = {}           # sid -> (mu, sigma) of each file that calibrated alone
    pool_state: dict = {"pool": None, "files": {}}   # the Pool; sid -> what apply_pool did
    # the calibration file (--tier-calibration, assignment/calfile.py): read only for
    # a file neither its own core nor the batch's pool calibrates; decided once
    # (`_file_pool`): the Pool it lends as campaign / lent, or why it lends nothing
    file_state: dict = {"decided": False, "pool": None, "as": None, "reason": None, "files": {}}
    _cal_record = getattr(cfg, "tier_calibration", None)

    def _file_pool(need_ledgers: dict):
        """The calibration file's Pool (source campaign / lent) or None, decided
        once against the batch's reagent, polarity, instrument class, scan range
        and days (calfile.choose; lending across reagents only with
        `tier_calibration_lent`; a lent window only when the mass-accuracy transfer
        check passes on `need_ledgers`, the uncalibrated files' own core scatter,
        calfile.receiving_scatter); the reason it lends nothing is logged once."""
        if file_state["decided"]:
            return file_state["pool"]
        file_state["decided"] = True
        if not _cal_record:
            return None
        from peaky.assignment import calfile as _CF
        from peaky.assignment import tiers as _TP
        facts = _CF.series_facts(ts_peaks if ts_peaks is not None else peaks)
        recv = _CF.receiving_scatter(need_ledgers, reagent=prof.name,
                                     polarity="positive" if prof.polarity == "+" else "negative",
                                     carry=axis_info.get("edge_carry"))
        det: dict = {}
        src_, why_ = _CF.choose(_cal_record, reagent=prof.name, polarity=prof.polarity,
                                instrument_class=cfg.instrument_class, scan_range=facts["scan_range"],
                                days=facts["days"], lent_ok=bool(getattr(cfg, "tier_calibration_lent", False)),
                                recv=recv, detail=det)
        file_state.update(**{"as": src_, "reason": why_, "facts": facts, "transfer": det or None})
        if src_ is None:
            log(f"[calpool] WARNING: the calibration file is not lent: {why_}")
            return None
        file_state["pool"] = _CF.as_pool(_cal_record, src_, abs_floor_mda=_TP._abs_floor(cfg))
        w_ = _cal_record["width"]
        log(f"[calpool] the calibration file {str(_cal_record.get('_sha256'))[:12]} is read as {src_}: "
            f"{float(w_['mu']):+.3f} ± {float(w_['sigma']):.3f} ppm (basis {w_.get('basis')}; reagent "
            f"{_cal_record.get('reagent')}, scan range {_cal_record.get('scan_range')}; the batch "
            f"{prof.name}, {facts['scan_range']})")
        return file_state["pool"]

    def _keep_pool_inputs() -> bool:
        """Whether a per-file run keeps what the pooled calibration re-applies
        (assign.run keep_pool_inputs: an uncalibrated file's tier-time ledger
        copy): until the batch's pool is decided, and after only when it, or the
        calibration file, has a calibration to lend."""
        if pool_state["pool"] is None or pool_state["pool"].cal is not None:
            return True
        return bool(_cal_record) and (not file_state["decided"] or file_state["pool"] is not None)

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
        if episode_flag_files:
            # the episode stage's row flag (sampling.episode_row_flags): on every
            # per-file ledger of a batch with an episode file, True on the rows of
            # an episode's files whose peak is on an episodic bin
            led["episode"] = SS.episode_row_flags(led, sid, episode_univ, episode_flag_files,
                                                  tol_ppm=tol_ppm)
        led.to_csv(os.path.join(pfdir, f"{sid}_ledger.csv"), index=False)
        full_ledgers[sid] = led
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

    def _pool_tiers(runs: list) -> None:
        """The batch-pooled tier calibration over one per-file stage's results
        (`runs`: dicts of sid, ledger, plaus, stats, pool_inputs, in sample_ids
        order), before they are folded: every file whose own core did not
        calibrate the tier engine (stats `degeneracy_cal` null) is given the
        batch's pooled calibration (calpool.apply_pool) when the pool has one --
        its degeneracy, tiers, degeneracy-keyed flags and its own evidence level
        re-applied, its stats saying so (`tier_calibration` 'pooled', the lent
        `degeneracy_cal`, which the pooled levels then read as its window). The
        width lent is the run sigma of the files that calibrated alone, when any
        did (`Pool.run_window`). A calibrated file is untouched; with no
        uncalibrated file the pool is never built. Every run's `pool_inputs`
        (the tier-time ledger copy) is released here, before the fold."""
        try:
            _pool_tiers_apply(runs)
        finally:
            for r in runs:
                r["pool_inputs"] = None

    def _pool_tiers_apply(runs: list) -> None:
        from peaky.assignment import calpool as _CP
        from peaky.assignment import ledger as _L
        from peaky.assignment import tiers as _TP
        from peaky.chem import reagents as _RGP
        for r in runs:
            pool_ledgers[r["sid"]] = r["ledger"]
            _own = (r["stats"] or {}).get("degeneracy_cal")
            if _own is not None:
                pool_cals[r["sid"]] = (float(_own["mu"]), float(_own["sigma"]))
        need = [r for r in runs if "degeneracy_cal" in (r["stats"] or {}) and r["stats"]["degeneracy_cal"] is None]
        if not need:
            return
        if pool_state["pool"] is None:
            _key = _RGP.reagent_for_adducts(list(prof.adducts or []))
            _masses = [m for (_l, m, _f) in _RGP.build_library(_key)] if _key else []
            pool_state["pool"] = _CP.pool_calibration(
                pool_ledgers, polarity="positive" if prof.polarity == "+" else "negative",
                reagent_masses=_masses, abs_floor_mda=_TP._abs_floor(cfg), klass=cfg.instrument_class,
                # the batch axis model's edge carry: a core row past its edge on the carried scale
                carry=axis_info.get("edge_carry"))
            if pool_cals:
                # the files that calibrated alone measure one file's width directly
                pool_state["pool"].run_window = (float(np.median([c[0] for c in pool_cals.values()])),
                                                 float(np.median([c[1] for c in pool_cals.values()])),
                                                 len(pool_cals))
            log(f"[calpool] {'' if pool_state['pool'].cal is not None else 'WARNING: '}{len(need)} file(s) "
                f"uncalibrated on their own core: {pool_state['pool'].describe()}")
        elif pool_state["pool"].cal is None:
            log(f"[calpool] WARNING: {len(need)} more file(s) uncalibrated on their own core; the batch's "
                f"pool is not used: {pool_state['pool'].reason}")
        pool = pool_state["pool"]
        if pool.cal is None:
            # neither the files' own cores nor the batch's pool: the calibration
            # file, when one was given and the batch reads it (calfile.choose)
            pool = _file_pool({r["sid"]: r["ledger"] for r in need})
            if pool is None:
                return
        for r in need:
            res_p = (_CP.apply_pool(r["ledger"], pool, r["pool_inputs"], audit=r["plaus"],
                                    log=lambda m, _s=r["sid"]: log(m.replace("[calpool]  ", f"[calpool]   {_s}:", 1)))
                     if r.get("pool_inputs") is not None else {})
            if not res_p:
                log(f"[calpool]   {r['sid']}: no pooled re-tier (its run kept no tier-time ledger)")
                continue
            st_ = r["stats"]
            # the window the file was lent: the lent width about its own centre
            st_["degeneracy_cal"] = {"mu": res_p["cal"]["mu"], "sigma": res_p["cal"]["sigma"]}
            st_["tier_calibration"] = pool.source
            st_["tier_pool"] = {"centre": res_p["cal"]["centre"], "width": res_p["cal"]["width"],
                                "retiered": res_p["tiers"]["retiered"], "demoted": res_p["tiers"]["demoted"],
                                "kept": res_p["tiers"]["kept"], "below": res_p["below"],
                                "o_monster": res_p["o_monster"],
                                "degenerate": res_p["degeneracy"]["degenerate"]}
            st_["by_tier"] = _L.stats(r["ledger"]).get("by_tier", st_.get("by_tier"))
            if st_.get("nox_skeleton_gate") is not None:
                # the skeleton cap as the pooled tier stage left it (taken where the
                # run takes it: before later stages rewrote rows)
                st_["nox_skeleton_gate"] = res_p["nox_skeleton_gate"]
            (pool_state if pool is pool_state["pool"] else file_state)["files"][r["sid"]] = st_["tier_pool"]
            if res_p.get("isotope_guard") is not None:
                # the per-file isotope guard ran on the pooled / lent re-level (calpool.apply_pool step 5)
                st_["isotope_guard"] = res_p["isotope_guard"]

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
        runs: list = []      # the stage's results, folded (_apply) after the pooled calibration
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
                res = A.run(sid, context=context, log=log, reflists_active=reflists_active,
                            keep_pool_inputs=_keep_pool_inputs(), **kw)
                runs.append({"sid": sid, "ledger": res["ledger"], "plaus": res.get("plausibility_audit") or [],
                             "stats": dict(res.get("stats", {})), "scoring": res.get("pattern_scoring"),
                             "pool_inputs": res.pop("pool_inputs", None)})
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
                keep = _keep_pool_inputs()
                futs = {ex.submit(_assign_one, sid, keep): sid for sid in ids}
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
                runs.append({"sid": sid, "ledger": out["ledger"], "plaus": out["plausibility_audit"],
                             "stats": out["stats"], "scoring": out.get("pattern_scoring"),
                             "pool_inputs": out.pop("pool_inputs", None)})
        # the batch-pooled tier calibration for the files that could not calibrate
        # alone (calpool), then the fold, in sample_ids order either path
        _pool_tiers(runs)
        for r in runs:
            _apply(r["sid"], r["ledger"], r["plaus"], r["stats"], stage, r["scoring"])

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
        _t_cover = time.time()
        _assign_files(list(sample_ids), STAGE_COVER, n_jobs)
        if n_episode_files and isinstance(selection.get("episodes"), dict):
            # information only (no gate): the share of the cover pool's wall the
            # episode files took, the files being alike on average
            _w = time.time() - _t_cover
            selection["episodes"]["est_extra_wall_s"] = round(_w * n_episode_files / max(len(sample_ids), 1), 1)
            selection["episodes"]["assign_wall_s"] = round(_w, 1)

    from peaky.chem import reagents as _RG
    from peaky.assignment import plausibility as PL
    from peaky.batch import timeseries as _TS
    from peaky.batch import reagent_water as _RW
    _rgk = _RG.reagent_for_adducts(list(prof.adducts or []))

    scale = None   # the batch's traces.MassScale, measured at the first merge
    rwater = None  # the reagent-water ladder (batch.reagent_water), measured at the first merge

    # the formulas a curated list stands behind (the pass-0 registry for the
    # batch's polarity / context and the active reference lists): one exempt set
    # for both merged-ledger removal gates, the isotopologue gate and the
    # element-signature removal (passes.curated_formulas)
    curated = _EE.curated_exempt(context, reflists_active)

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
        # the vote-drop recorder (unidentified.DropLog): the merged rows the gates below
        # remove after the vote -- this call's own record, returned with it (the last call is the run's)
        drop_log = _UN.DropLog()
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
        # A persistent ion's cluster with a plume solvent (episode_evidence.read_solvent_clusters;
        # only in a batch with episode files): a merged row at a persistent Assigned ion + the
        # solvent's neutral mass, confined to the spectra where the solvent's own ion is bright, is
        # read as that ion's cluster -- before the reagent-cluster strip, which spares it
        sc_table, sc_summary = _EE.empty_clusters(), None
        if "episode" in merged.columns:
            sc_table, sc_summary = _EE.read_solvent_clusters(merged, ts_peaks, adducts=list(prof.adducts or []),
                                                             log=log)
        # Merge guard: drop reagent-cluster ions a per-file pass mislabelled as analyte
        # (urea [R_n+H]+/[R_n+NH4]+ read as CHNO/CH4N2O on the [M+NH4]+/urea channel) --
        # they otherwise dominate the 'assigned' signal. Belt-and-braces with the
        # per-file reagent lock/reclaim (older per-file ledgers predate that fix).
        if _rgk:
            _m_before = merged
            _sc = _EE.cluster_rows(merged)
            if _sc.any():
                _kept, _rgstrip = _RG.strip_reagent_cluster_rows(merged[~_sc].reset_index(drop=True), _rgk,
                                                                 log=log)
                merged = (pd.concat([_kept, merged[_sc]], ignore_index=True)
                          .sort_values("mz", kind="mergesort").reset_index(drop=True))
            else:
                merged, _rgstrip = _RG.strip_reagent_cluster_rows(merged, _rgk, log=log)
            drop_log.note(_m_before, merged, "reagent-cluster strip")
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
        _m_before = merged
        merged, rw_stripped = _RW.strip_rung_rows(merged, rwater["rungs"], tol_ppm=scale.stamp_ppm, log=log,
                                                  jitter=jitter, segment_of=rwater.get("segment_of"))
        drop_log.note(_m_before, merged, "reagent-water rung")
        # The merged row's tier_reason (from align: the vote's exemption, else NA)
        # also takes the batch-level gates' notes below (cleanup._note appends to
        # it): a re-read can leave the merged formula different from EVERY per-file
        # reading, and the row itself must say why.
        if "tier_reason" not in merged.columns:
            merged["tier_reason"] = pd.NA
        merge_gates: dict = {}
        if sc_summary is not None:
            merge_gates["solvent_clusters"] = sc_summary
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
                protected=protected_neutrals, nh4_main=(prof.detect_adduct == "[M+NH4]+"), log=log)
            # urea adducts of hydrocarbons / N-rich neutrals: the same ion as a protonated N-richer
            # compound -- a note on the split, the tier kept
            merge_gates["urea"] = cleanup.note_implausible_urea_adducts(merged, log=log)
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
            _m_before = merged
            merged, iso_rows, merge_gates["isotopologue"] = _IC.satellite_rows(
                merged, ts_peaks, resolution=rp, mass_scale=scale, klass=cfg.instrument_class, prof=prof,
                reagents=_rg, per_file=level_frames, exempt=curated, log=log)
            drop_log.note(_m_before, merged, "isotopologue strip")
        else:
            merge_gates["isotopologue"] = {"ran": False, "n_stripped": 0, "n_mixed": 0, "n_exempt": 0,
                                           "skipped": "--no-isotopologue-gate"}
        # A reading Assigned in no more than two files, on a brighter Assigned row's
        # isotopologue line in each of them and with no 13C line of its own, is held at
        # Candidate; beside a tracking satellite partner (Orbitrap-class: within +-4 mDa,
        # 10x brighter) it is only noted (batch/sideband.py; tables/sideband_holds.csv)
        _sig = {str(st_.get("sample_id")): (st_.get("degeneracy_cal") or {}).get("sigma")
                for st_ in per_stats if isinstance(st_.get("degeneracy_cal"), dict)}
        sb_rows, merge_gates["sideband"] = _SB.hold_low_file_promotions(
            merged, full_ledgers, ts_peaks, tol_ppm=scale.merge_ppm, sigma_ppm=_sig,
            orbitrap=(cfg.instrument_class == "orbitrap"), log=log)
        out = {"merged": merged, "jitter": jitter, "merge_gates": merge_gates,
               "trace_info": trace_info, "stamp_tol": stamp_tol, "ts_annot": None,
               "predicted_rows": {}, "predicted_tracks": None,
               "reagent_water": _RW.table(rwater["rungs"], rw_stripped),
               "isotopologue": iso_rows, "sideband": sb_rows, "drop_log": drop_log,
               "solvent_clusters": sc_table}
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
            out.update(_stamp_series(merged, _aux, stamp_tol, tracks))
            # kept so the re-stamp after the batch checks (a merged-row removal, a
            # per-file rewrite; below) can stamp with the same inputs: the rungs
            # apart, so the per-file rows can be re-read from rewritten ledgers
            out["stamp_inputs"] = (_aux, tracks, _rw_rows)
        return out

    def _stamp_series(merged, aux, stamp_tol, tracks) -> dict:
        """The whole-batch stamp of `merged` (+ the identified non-analyte rows
        `aux`): {ts_annot, predicted_rows, predicted_tracks}."""
        _stamp = _TS.stamping_frame(merged, aux, tol_ppm=stamp_tol)
        _stats: dict = {}
        ts = _TS.annotate_peaks(ts_peaks, _stamp, tol_ppm=stamp_tol, stats=_stats, tracks=tracks)
        return {"ts_annot": ts, "predicted_rows": dict(_stamp.attrs.get("predicted_satellites") or {}),
                "predicted_tracks": _stats.get("predicted_tracks")}

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
    twins_table = _LT.measure(ts_annot, level_frames, prof, alias_ties=alias_ties,
                              klass=_IC.batch_class(rp, cfg.instrument_class), log=log)
    twins_table.to_csv(os.path.join(TAB, "label_twins.csv"), index=False)
    # ... and its 13C-shadow lines (Orbitrap-class): a 14N line that follows the 13C line
    # of the committed ion 8.1 mDa above it is no reading's own ion -- every Assigned
    # merged reading on it is Candidate (label_twins.shadow_gate; no re-vote, as the TOF
    # gates below); the pooled pair's label veto alone leaves the vote's tier
    merge_gates["label_shadow"] = _LT.shadow_gate(merged, twins_table, mass_scale=scale, log=log)
    # the isotope checks (iso_veto): rule C, REQ and HIGH read each committed
    # formula's isotope claims off the same stamped series (the instrument class
    # the batch resolved once, cfg.instrument_class: a TOF roster wins), rule H its exact halogen line (a lock,
    # judged against the batch's element budget: `context`); written for every
    # run (empty without a time series); a refuted pair is rejected (5b)
    iso_table = _IC.measure(ts_annot, level_frames, prof, resolution=rp, mass_scale=scale,
                            x_edge=x_edge, context=context, edge_cps=getattr(cfg, "noise_edge_batch_cps", None),
                            klass=cfg.instrument_class, log=log)
    # a REQ veto on an element-signature line (81Br / 37Cl / 34S / 29Si / 30Si,
    # Orbitrap-class) is the batch's own refutation of the element: the reading
    # leaves the merged ledger (iso_checks.remove_signature_vetoed), not only 5b;
    # a curated formula and a known-species decision stay
    drop_log = res_m["drop_log"]
    _m_before = merged
    merged, merge_gates["element_signature"] = _IC.remove_signature_vetoed(
        merged, iso_table, exempt=curated, klass=_IC.batch_class(rp, cfg.instrument_class), log=log)
    drop_log.note(_m_before, merged, "element-signature removal")
    # the vote-drop list: every per-file M0 row with no merged row left at its m/z, and the
    # gate that removed it (unidentified.vote_drops) -- read before the per-file rewrite
    # below turns the stripped readings' rows into iso_child / unexplained rows
    try:
        vote_drop_table = _UN.vote_drops(full_ledgers, per_file, merged, drop_log,
                                         tol_ppm=float(getattr(scale, "merge_ppm", tol_ppm)), offsets=offsets,
                                         jitter=jitter)
    except Exception as e:  # noqa: BLE001 -- a diagnostic table never fails a finished batch
        log(f"[assign_batch] WARNING the vote-drop list was not built: {type(e).__name__}: {e}")
        vote_drop_table = _UN.empty_drops()
    # ... a row the isotopologue gate stripped as the satellite of a reading that
    # has just left is no longer that reading's satellite (iso_checks.parent_removed:
    # verdict 'parent removed', no SAT veto) ...
    res_m["isotopologue"], _n_orph = _IC.parent_removed(res_m.get("isotopologue"),
                                                        merge_gates["element_signature"].get("pairs"), log=log)
    if isinstance(merge_gates.get("isotopologue"), dict):
        merge_gates["isotopologue"]["n_parent_removed"] = _n_orph
    # rule K's refutation on the merged tier (label_twins.demote_vetoed): a refutation that
    # names another ion (a 13C shadow, an [Y+^NO3]- reading with no 14N twin) caps an Assigned
    # reading Candidate; an [X+NO3]- line "cluster or organonitrate undecided" is the same ion
    # as the organonitrate, so the row keeps the vote's tier with a note (the pair stays 5b)
    merge_gates["label_veto"] = _LT.demote_vetoed(merged, twins_table, exempt=curated, log=log)
    # ... and the per-file ledgers RECORD the decision (iso_checks.reconcile_per_file):
    # a stripped reading's M0 row in a file becomes the parent's iso_child where that
    # file commits the parent (a reagent parent's: a reagent isotopologue), else --
    # and wherever the parent has just left -- it is released to unexplained. The
    # rewritten ledgers replace per_file/<sid>_ledger.csv, the levels' frames and the
    # file's role counts, so the evidence levels below, the report and a per-file
    # publish read what the merged ledger decided. The batch checks above (neutral
    # pairs, label twins, iso checks) were measured on the ledgers as the files wrote
    # them; only the rows of the stripped and the removed readings differ there (their
    # SAT / REQ vetoes stay in iso_checks.csv).
    # ... and then the element-signature removal (iso_checks.release_signature_removed):
    # every M0 row committing a reading that left is released to unexplained, so a
    # per-file ledger, its levels and a per-file publish name no reading the batch
    # refuted. iso_checks.record_per_file runs the two in that order.
    _rewritten, _rw_summ, _sg_summ = _IC.record_per_file(
        full_ledgers, res_m.get("isotopologue"), merge_gates["element_signature"].get("pairs"),
        mass_scale=scale, log=log)
    if isinstance(merge_gates.get("isotopologue"), dict):
        merge_gates["isotopologue"]["per_file"] = _rw_summ
    if merge_gates["element_signature"].get("removed"):
        merge_gates["element_signature"]["per_file"] = _sg_summ
    # ... then cleanup's ringing flags that the batch series shows to be a committed reading's isotope line
    # (iso_checks.release_ringing_lines: the line follows its reading's M0, not the ringing parent) become that
    # reading's iso_child; those ledgers are saved, re-levelled and re-stamped with the rest
    _rl, merge_gates["ringing_lines"] = _IC.release_ringing_lines(full_ledgers, ts_peaks, rp, log=log)
    _rewritten = set(_rewritten) | _rl
    for sid in sorted(_rewritten):
        _led = full_ledgers[sid]
        _led.to_csv(os.path.join(pfdir, f"{sid}_ledger.csv"), index=False)
        level_frames[sid] = EV.trim(_led)
        for st_ in per_stats:
            if str(st_.get("sample_id")) == str(sid):
                _recount_roles(st_, _led, _rw_summ["files"].get(sid), signature=_sg_summ["files"].get(sid))
    # why the batch's unexplained lines are unexplained (blocked.py): the per-file
    # ledgers as the merged-ledger gates left them, pooled by m/z at the merge's
    # window (offset-corrected, as align does) -- a diagnostic table, read by nothing
    from peaky.assignment import blocked as _BL
    try:
        _BL.batch_table(full_ledgers, tol_ppm=getattr(scale, "merge_ppm", DEFAULT_TOL_PPM),
                        offsets=offsets).to_csv(os.path.join(TAB, "unexplained_blocked.csv"), index=False)
    except Exception as e:  # noqa: BLE001 -- a diagnostic table never fails a finished batch
        log(f"[assign_batch] WARNING tables/unexplained_blocked.csv not written: {type(e).__name__}: {e}")
    # the parent rows name the lines the gate gave them (one row per parent ion)
    merged = _IC.parent_lines(merged, res_m.get("isotopologue"))
    # the NOx-skeleton tier cap, said on the merged row: how many of the files that
    # commit the reading capped it (tiers.merged_skeleton_notes; the tier is the vote's)
    from peaky.assignment import tiers as _TI
    merged, nox_skeleton_merged = _TI.merged_skeleton_notes(merged, full_ledgers)
    # ... plus the merged-ledger isotopologue gate's strips (iso_checks.veto_rows,
    # check 'SAT'): the pooled pair of a line the merged ledger gave to its parent is
    # refuted the same way, so evidence_levels.csv agrees with the merged ledger
    _sat_veto = _IC.veto_rows(res_m.get("isotopologue"))
    if len(_sat_veto):
        iso_table = (pd.concat([iso_table, _sat_veto], ignore_index=True) if len(iso_table)
                     else _sat_veto)
    iso_table.to_csv(os.path.join(TAB, "iso_checks.csv"), index=False)
    # ... and the batch series is RE-STAMPED from the merged ledger without them
    # (the first stamp named them: _batch_ts.parquet, the levels' series and the
    # TOF gates below must not), with the same inputs as the merge's stamp -- the
    # per-file identified rows re-read from the rewritten ledgers when either gate
    # rewrote one (a released reading's isotope lines, a stripped reading's own,
    # no longer name its ion)
    _restamp = bool(merge_gates["element_signature"]["removed"] or _rewritten)
    if _restamp and ts_annot is not None and res_m.get("stamp_inputs"):
        _aux_s, _tracks_s, _rw_s = res_m["stamp_inputs"]
        if _rewritten:
            from peaky.batch import timeseries as _TSI
            _parts = [_TSI.identified_rows(_l) for _l in full_ledgers.values()]
            if _rw_s is not None and len(_rw_s):
                _parts.append(_rw_s)
            _aux_s = pd.concat(_parts, ignore_index=True) if _parts else None
        res_m.update(_stamp_series(merged, _aux_s, stamp_tol, _tracks_s))
        ts_annot = res_m["ts_annot"]
        if merge_gates["element_signature"]["removed"]:
            merge_gates["element_signature"]["restamped"] = True
        if _rewritten and isinstance(merge_gates.get("isotopologue"), dict) and _rw_summ["files"]:
            merge_gates["isotopologue"]["restamped"] = True
        log("[iso_checks] the batch series re-stamped from the merged ledger and the per-file ledgers as "
            "the merged-ledger gates left them")
    # the TOF ion-M+2 gates (iso_checks.tof_m2_gates; TOF-class batches only): a
    # merged winner whose own M+2 line REQ refutes over the batch -- a species the
    # known-species lock decided included -- and a merged line that is the 81Br
    # partner of the line one spacing below it are Candidate. No re-vote: the
    # winner and its reading stay, the row says why.
    merge_gates["tof_m2"] = _IC.tof_m2_gates(merged, iso_table, ts_annot, resolution=rp, mass_scale=scale,
                                             klass=cfg.instrument_class, log=log)
    # THE EVIDENCE LEVEL (the scale of peaky 0.10.0): the batch's per-file
    # ledgers pooled as ONE source (cover + residual files; every file-count
    # minimum 3), re-read from the per_file/<sid>_ledger.csv files just written
    # in sorted sample-id order with the parser a post-hoc re-level uses, so the
    # run and `scripts/level_ledger.py` on its run dir see the same inputs; with
    # the stamped time series, the merged ledger (the NH4 rule's ion index), the
    # protected neutrals, the per-file gates and degeneracy calibrations, and the
    # batch checks' vetoes. Stamped on the merged ledger by (neutral, adduct); a
    # merged row no pooled pair holds (a batch-level re-read) takes the level of the
    # pooled pair it replaced, the same ion (4b with its split open, or 5a), and
    # gets no level only where that pair is 5b or absent (evidence.stamp_merged).
    from peaky.assignment.levels import lists as _LS
    rl_context = _LS.reflists_context(reflists_active, rl_record)
    # the run-level switches of the batch's context profile, derived by the very
    # helper assign.run calls on each file, from the same reagent, channels,
    # instrument class and trace-first flag the files were handed; a file whose
    # own run recorded other switches (a trace-first batch's residual files run
    # without the trace flag) is named in the log -- the record is the batch's
    from peaky.chem import contexts as _X
    context_flags = _X.profile_flags(A.run_context_profile(
        context, reagent_profile=assign_kw.get("reagent_profile") or prof.name,
        adducts=assign_kw.get("adducts") or list(prof.adducts),
        instrument_class=cfg.instrument_class, trace_sample=trace_sample is not None))
    _off = sorted(str(s_.get("sample_id")) for s_ in per_stats
                  if "context_flags" in s_ and s_["context_flags"] != context_flags)
    # what a reader of batch_summary.json must know about the run's own record
    # (`warnings`; [] when nothing): here, files whose own run recorded other
    # context switches than the batch record the levels and the report read
    run_warnings: list = []
    if _off:
        log(f"[assign_batch] context switches: the batch records {context_flags or 'none'}; "
            f"{len(_off)} file(s) ran with other switches ({', '.join(_off[:5])}"
            f"{', ...' if len(_off) > 5 else ''})")
        run_warnings.append({
            "what": "context_flags_mismatch",
            "message": (f"{len(_off)} file(s) committed under other context switches than the batch "
                        f"record ({context_flags or 'none'}) that the evidence levels and the report "
                        "read; their skeleton-only readings are judged on the batch's switches"),
            "batch": context_flags,
            "files": {sid: next((s_.get("context_flags") for s_ in per_stats
                                 if str(s_.get("sample_id")) == sid), None) for sid in _off}})
    level_summary = {
        "reagent": prof.name, "label": prof.label, "context": context,
        "context_flags": context_flags,
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
    # the isotope-contradiction guard on the merged ledger (isoguard.guard_merged): an
    # Assigned row whose POOLED pair the levels' own isotope facts contradict (13C count,
    # an absent own 34S / 37Cl line, a 34S / 37Cl line it cannot explain) is Candidate with
    # the note. The per-file guard (stage isotope_guard) already held the files' tiers
    # before the vote; this reads the facts of every file together
    _ig_noted = _IG.merged_notes(merged, full_ledgers, log=log)
    merge_gates["isotope_guard"] = _IG.guard_merged(merged, levels, log=log)
    merge_gates["isotope_guard"]["rows"] = merge_gates["isotope_guard"]["rows"][:50]
    merge_gates["isotope_guard"]["per_file_noted"] = int(_ig_noted)
    # the episode evidence (episode_evidence.judge; only in a batch with episode files): an episodic
    # reading in <= 3 files is judged on its episode's own spectra -- Assigned from that episode only
    # when its mass, its isotope lines, its alias rivals, no hold and a second fact all say so
    ee_table = None
    if "episode" in merged.columns:
        _win = {str(st_.get("sample_id")): ((st_.get("degeneracy_cal") or {}).get("mu"),
                                            (st_.get("degeneracy_cal") or {}).get("sigma"))
                for st_ in per_stats if isinstance(st_.get("degeneracy_cal"), dict)}
        _edg = {str(st_.get("sample_id")): st_.get("noise_edge_cps") for st_ in per_stats}
        ee_table, merge_gates["episode_evidence"] = _EE.judge(
            merged, full_ledgers, ts_annot if ts_annot is not None else ts_peaks, windows=_win, edges=_edg,
            adducts=list(prof.adducts or []), labelled=any("^N" in str(a) for a in (prof.adducts or [])),
            levels=levels, tol_ppm=float(getattr(scale, "merge_ppm", tol_ppm)), exempt=frozenset(curated or ()),
            orbitrap=(cfg.instrument_class == "orbitrap"), mode=episode_evidence, log=log)
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
        "iso_checks": _IC.summary(iso_table, rp, klass=cfg.instrument_class),
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
    if levels_na_reason and pool_state["pool"] is not None and pool_state["pool"].cal is None:
        # ... and the batch-pooled core (calpool) could not stand in: say why
        levels_na_reason += (" A core pooled over the batch's files was tried and not used: "
                             f"{pool_state['pool'].reason}.")
    if levels_na_reason and file_state["reason"]:
        # ... nor the calibration file (--tier-calibration): say why
        levels_na_reason += f" A calibration file was given and not used: {file_state['reason']}."
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
    # the bright lines no batch output explains (unidentified.build): every per-file row
    # bright in its file that ends unexplained, an isotope child / artifact whose parent
    # the merged ledger does not hold Assigned, or an M0 row the vote dropped -- binned by
    # m/z, with why; written for every run (header only when none)
    try:
        unident = _UN.build(full_ledgers, per_stats, merged, drops=vote_drop_table, ts=ts_annot,
                            tol_ppm=float(getattr(scale, "merge_ppm", tol_ppm)), offsets=offsets,
                            axis_info=axis_info, resolution=rp)
    except Exception as e:  # noqa: BLE001 -- a diagnostic table never fails a finished batch
        log(f"[assign_batch] WARNING tables/unidentified.csv not built: {type(e).__name__}: {e}")
        unident = _UN.empty_table()
    unident_path = _UN.write(unident, TAB)
    vote_drop_table.to_csv(os.path.join(TAB, "vote_drops.csv"), index=False)
    unident_summary = _UN.summary(unident, vote_drop_table)
    log(f"[assign_batch] unidentified bright lines: {unident_summary['rows']} "
        f"({unident_summary['by_why']}); vote drops {unident_summary['vote_drops']} -> "
        f"tables/{os.path.basename(unident_path)}, tables/vote_drops.csv")
    merged.to_csv(os.path.join(out_dir, "merged_ledger.csv"), index=False)
    jitter.to_csv(os.path.join(TAB, "jitter.csv"), index=False)
    # the reagent-water rungs and the merged readings they displaced (always written:
    # a stable artifact set; header only when the profile declares no water cores)
    res_m["reagent_water"].to_csv(os.path.join(TAB, "reagent_water.csv"), index=False)
    # the isotopologue rows the merged ledger gave to their parent, and the mixed /
    # exempt lines it only noted (always written: header only when none)
    res_m["isotopologue"].to_csv(os.path.join(TAB, "isotopologue_rows.csv"), index=False)
    res_m.get("sideband", _SB.empty_table()).to_csv(os.path.join(TAB, "sideband_holds.csv"), index=False)
    if ee_table is not None:
        # the episode evidence and the plume-solvent clusters (a batch with episode files only)
        ee_table.to_csv(os.path.join(TAB, "episode_evidence.csv"), index=False)
        res_m.get("solvent_clusters", _EE.empty_clusters()).to_csv(os.path.join(TAB, "solvent_clusters.csv"),
                                                                    index=False)
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
    _centres = scoring_centres(scorings)
    if axis_info.get("edge_carry"):
        log(f"[assign_batch] axis edge carry: "
            f"{sum(int((x.get('axis_edge_carry') or {}).get('m0_carried', 0)) for x in per_stats)} committed "
            f"M0 rows outside the model's scope judged at its edge, in "
            f"{sum(1 for x in per_stats if x.get('axis_edge_carry'))} of {len(per_stats)} files")
    log("[assign_batch] scoring centres: "
        + ", ".join(f"{n} {src}" for src, n in _centres["by_mu_source"].items())
        + (f"; self-centred: " + ", ".join(f"{s} {v:+.2f} ppm" for s, v in _centres["self"].items()
                                           if v is not None) if _centres["self"] else ""))
    summary = {
        "reagent": prof.name, "label": prof.label, "context": context,
        # the context profile's run-level switches ({} = the named context as is)
        "context_flags": context_flags,
        # what the run's record cannot say on its own (a list of {'what',
        # 'message', ...}; [] when nothing): a trace-first batch's files that ran
        # with other context switches than `context_flags`
        "warnings": run_warnings,
        # the NOx-skeleton tier cap (null when the batch reads no skeletons): the
        # merged rows whose reading a file capped / noted, and the per-file totals
        "nox_skeleton_gate": ({**nox_skeleton_merged,
                               "files_capped": int(sum((s_.get("nox_skeleton_gate") or {}).get("capped", 0)
                                                       for s_ in per_stats)),
                               "files_noted": int(sum((s_.get("nox_skeleton_gate") or {}).get("noted", 0)
                                                      for s_ in per_stats))}
                              if context_flags.get("nox_skeleton")
                              # a trace-first batch records no switches, but its residual files read the skeleton
                              or any(s_.get("nox_skeleton_gate") is not None for s_ in per_stats) else None),
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
        # where each file's scoring centre came from (its snapshot's mu_source),
        # counted, and the files that measured their own (selfcentre.py)
        "scoring_centres": _centres,
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
        # the episode evidence's mode (episode_evidence.MODES; --episode-evidence), recorded whether or
        # not the batch had an episode file (the run manifest reads it: pipeline._switch_record)
        "episode_evidence_mode": episode_evidence,
        # the bright unidentified lines (tables/unidentified.csv): `unidentified_bright` the row
        # count (a later drop to 0 shows), `unidentified` by why + the vote-drop counts
        "unidentified_bright": int(unident_summary["rows"]),
        "unidentified": unident_summary,
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
        # where each file's tier calibration came from (its own core, the batch's
        # pooled core, none): the batch's weakest in one word, the files per
        # source, and the pooled core itself (assignment/calpool.py; null when
        # every file calibrated alone and no pool was built)
        "tier_calibration": {
            "source": _tier_cal_source(per_stats),
            "files": {k: sum(1 for x in per_stats if x.get("tier_calibration") == k)
                      for k in ("per_file", "pooled", "none")
                      + (("campaign", "lent") if _cal_record else ())},
            "pool": (dict(pool_state["pool"].record(), retiered_files=dict(pool_state["files"]))
                     if pool_state["pool"] is not None else None)},
        # where the evidence levels' windows came from (assignment/calfile.py): the
        # batch's weakest file source in one word (per_file / pooled / campaign /
        # lent / none), the files per source, and the calibration file given with
        # --tier-calibration (sha256, source runs, width; how the batch read it and
        # why not; the files it was lent to), null when none was given
        "levels_calibration": _levels_calibration(per_stats, _cal_record, file_state, cfg),
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
        # the pinned record of each list the run read (id, version, sha256, roles,
        # pinned, how activated): what run_manifest.json carries as `lists`
        "lists": RL.lists_record(reflists_active, rl_record),
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
    n_ep = int(n_episode_files or 0)
    if residual or n_ep:
        # files per stage, and the ions each stage brought in (`stage` column). The
        # episode stage's files are assigned with the cover's (their ions read `stage`
        # = cover) but counted apart here, so its yield is readable; the key is there
        # only when the stage added a file (a batch without one keeps its summary)
        summary["n_files_by_stage"] = {STAGE_COVER: int(n_cover - n_ep),
                                       **({SS.ROLE_EPISODE: n_ep} if n_ep else {}),
                                       **({STAGE_RESIDUAL: int(len(sample_ids) - n_cover)} if residual else {})}
    if residual:
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
