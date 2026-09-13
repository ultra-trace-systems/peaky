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

Within a cluster the files VOTE, in two stages: the ION carried by the most files
wins (tier and ion_score break ties), and among that ion's labels -- the same
ion read as C13H14O4 [M+NH4]+ or as C13H17NO4 [M+H]+ -- the one Assigned in the
most files wins, because on such a pair Assigned means a discriminating channel
was present and Candidate means the file had nothing to decide with. The losing
readings stay on the merged row (`alternatives`, `n_files_ion`, `n_files_winner`,
`ion_agree`) as well as in jitter.csv.
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
import time

import numpy as np
import pandas as pd

from peaky import paths as PT
from peaky.chem import profiles as P
from peaky.batch import sampling as SS

__version__ = "0.8.1"  # traces.stamp block + tables/predicted_satellites.csv: the stamp's
                       # predicted satellites counted apart from observed, track coherence
                       # (0.8.0: the targeted residual stage: a second selection + assignment
                       # after the cover's merge, one align() over both, stage provenance;
                       # 0.7.0: the merge is a VOTE -- n_files_ion / n_files_winner /
                       # alternatives / ion_agree, the batch-level gates' tier_reason)

# the merge's m/z tolerance IS the selector's binning tolerance (one constant for
# every batch-level binning; see sampling.BATCH_TOL_PPM)
DEFAULT_TOL_PPM = SS.BATCH_TOL_PPM
TIER_ASSIGNED = "Assigned"
STAGE_COVER = "cover"          # a file of the presence cover (incl. its k_min pads)
STAGE_RESIDUAL = "residual"    # a file the residual stage targeted
TIER_RANK = {"Assigned": 2, "Candidate": 1}
_M0_COLS = ["mz", "neutral_formula", "adduct", "tier", "ion_score",
            "admitted_by", "occurrence"]   # the last two: admission provenance, when present


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


def _vote(g: pd.DataFrame, curated: set):
    """Rank one cluster's readings in two stages. Returns (ions, labels):
    `ions` one row per ion (_ion, curated, n_files, n_assigned, best_ion), best
    first; `labels` one row per (neutral_formula, adduct) reading of EVERY ion
    (_ion, _nf, _ad, curated, n_files, n_assigned, best_ion), the winning ion's
    readings ranked best first and listed first, the other ions' readings after
    them in ion order.

    1. WHICH ION sits at this m/z is what files can genuinely disagree on, and
       the count decides it: the ion carried by the most FILES wins, the number
       of files carrying it at Assigned tier and the best ion_score only break
       ties, and the ion's own text is the last key -- so a full tie resolves
       the same way whatever order the files arrived in (serial and parallel
       runs stay byte-identical). This is the order collapse_trace_labels
       already applies to competing labels on one trace.

       `curated` exempts an ion from the file count, not from corroboration: an
       ion one of whose labels is a neutral from a curated list
       (`_curated_neutrals`) ranks first when that label reached Assigned in at
       least one file and in no fewer files than any grid ion did, so a list
       identity is not outvoted by grid GUESSES; a grid ion the tier engine
       corroborated in more files is a real contest, and the count decides it
       (sulfolane from the known list in one file lost to fluorenone [M+H]+
       Assigned in nine).

    2. WHICH LABEL of that ion -- the same ion read as C13H14O4 [M+NH4]+ or as
       C13H17NO4 [M+H]+ (the reagent-N isobar) -- is decided by corroboration,
       not by count: the tier engine marks such a reading Assigned only when a
       discriminating channel was present in that file (an N-free sibling, the
       joint NH4+urea pair, a series anchor) and Candidate when it had nothing
       to decide with, so counting Candidate files would be counting silence.
       The label Assigned in the most files wins; file count and score break
       ties. On the Texas Ur+ run 16 of the 43 same-ion splits had a majority
       label nobody had corroborated against a minority label some file had."""
    assigned = g["_r"] >= TIER_RANK[TIER_ASSIGNED]
    gg = g.assign(_asrc=g["src"].where(assigned),       # the file, when Assigned there
                  _c=g["_nf"].isin(curated).astype(int))
    lab = (gg.groupby(["_ion", "_nf", "_ad"], sort=True)  # text order = last key
             .agg(curated=("_c", "max"), n_files=("src", "nunique"),
                  n_assigned=("_asrc", "nunique"),        # FILES at Assigned, not rows
                  best_ion=("ion_score", "max"))
             .reset_index())
    ions = (gg.groupby("_ion", sort=True)
              .agg(n_files=("src", "nunique"), n_assigned=("_asrc", "nunique"),
                   best_ion=("ion_score", "max"))
              .reset_index())
    cur_lab = lab[lab["curated"] == 1]
    grid = ions[~ions["_ion"].isin(cur_lab["_ion"])]
    bar = max(1, int(grid["n_assigned"].max()) if len(grid) else 1)
    exempt = set(cur_lab.loc[cur_lab["n_assigned"] >= bar, "_ion"])
    ions["curated"] = ions["_ion"].isin(exempt).astype(int)
    ions = ions.sort_values(["curated", "n_files", "n_assigned", "best_ion"],
                            ascending=False, kind="mergesort")   # stable: keeps text order
    rank = {k: i for i, k in enumerate(ions["_ion"])}
    lab = (lab.assign(_k=lab["_ion"].map(rank))
              .sort_values(["_k", "curated", "n_assigned", "n_files", "best_ion"],
                           ascending=[True, False, False, False, False], kind="mergesort")
              .drop(columns="_k"))
    return ions, lab


def _describe(r) -> str:
    """One losing reading for the merged row: 'C15H25N [M+H]+ x1 Candidate 0.97'
    (the best tier and score any file gave it)."""
    tier = TIER_ASSIGNED if int(r["n_assigned"]) > 0 else "Candidate"
    score = "" if pd.isna(r["best_ion"]) else f" {float(r['best_ion']):.2f}"
    return f"{r['_nf']} {r['_ad']} x{int(r['n_files'])} {tier}{score}"


def align(per_file: dict, *, tol_ppm: float = DEFAULT_TOL_PPM,
          offsets: dict | None = None, curated=None, stages: dict | None = None):
    """Align the M0 rows of several files by m/z and let the files VOTE on each
    cluster's reading (see `_vote`: the count decides WHICH ION, corroboration
    decides WHICH LABEL of it).

    per_file : {src -> DataFrame with _M0_COLS}. offsets : {src -> median ppm}
    (subtracted before clustering so a per-file calibration shift does not split
    a peak). curated : neutral formulas whose identity is a curated list's, not
    the grid's (a reference-peaklist rescue or the pass-0 known-species list --
    see `_curated_neutrals`); an ion carrying one of these is not outvoted by
    grid ions Assigned in fewer files than it (see `_vote`), and the merged
    row's tier_reason says so when that decided the cluster. stages : {src ->
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
              something to explain: the curated exemption, or a label chosen
              by corroboration over a bigger count); srcs[, stage],
              mz_jitter_ppm_raw, mz_jitter_ppm_caldj.
      jitter  long form, one row per (cluster, file): cluster, src, mz,
              neutral_formula, adduct, tier, ion_score -- every reading, winner
              or not.

    The previous rule ranked the number of ASSIGNED files first, which let one
    file's Assigned reading outvote many files' Candidate reading of a
    different ion: on the Texas Ur+ run (15 files) 12 of the 73 split clusters
    were decided that way -- and the merged row then carried nothing to show
    the other files had read it differently."""
    offsets = offsets or {}
    curated = {str(p) for p in (curated or ())}
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
                                      "ion_score", "admitted_by", "occurrence",
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
        ions, lab = _vote(g, curated)
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
        if int(win_ion["curated"]) and len(others) and int(others["n_files"].max()) > int(win_ion["n_files"]):
            top = others.iloc[0]
            top_lab = lab[lab["_ion"] == top["_ion"]].iloc[0]
            notes.append(f"curated identity kept over the {int(top['n_files'])}-file "
                         f"{top_lab['_nf']} {top_lab['_ad']} reading (vote "
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
                                    ion_score=r.get("ion_score")))
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
# the subset whose identity comes from OUTSIDE the formula grid -- a curated
# reference peaklist or the pass-0 known-species list (mass + own-twin gate).
# `certified:` is left out on purpose: a multi-channel certification is the
# same file's own evidence for a grid formula, which the tier already credits
# (and a Candidate-tier certified C19H8ClN once outvoted a 3-file reading).
_CURATED_METHODS = ("reflist-rescue", "known:")


def _neutrals_by_method(ledger: pd.DataFrame, prefixes: tuple) -> set:
    if not {"method", "neutral_formula"} <= set(ledger.columns):
        return set()
    meth = ledger["method"].astype(str)
    keep = meth.str.startswith(prefixes)
    return set(ledger.loc[keep, "neutral_formula"].dropna().astype(str)) - {"nan", ""}


def _protected_neutrals(ledger: pd.DataFrame) -> set:
    """Neutrals the amine gate must not re-read (see _PROTECTED_METHODS)."""
    return _neutrals_by_method(ledger, _PROTECTED_METHODS)


def _curated_neutrals(ledger: pd.DataFrame) -> set:
    """Neutrals whose identity is a curated list's, not the grid's: the merge
    vote's exemption (see align / _vote; _CURATED_METHODS)."""
    return _neutrals_by_method(ledger, _CURATED_METHODS)


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
        n_jobs: int | None = None, log=print, **assign_kw) -> dict:
    """Assign the presence-cover subset of a batch and combine, keeping per-file
    ledgers. Provide EITHER `peaks` (a batch peak/sample table) OR `batch` (a
    batch name; the per-sample list is fetched fresh from the live server, which
    also guarantees the selected sample ids are valid for get_peaks — cached ids
    go stale / 404 when the server copy is renamed). Selection needs the per-PEAK
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
    `residual_samples` to append to its own selection table."""
    from peaky.assignment import assign as A
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
        peaks = IO.fetch_batch_samples(client, batch, dataset=dataset)
        log(f"[assign_batch] fetched {len(peaks)} samples for batch {batch!r}")

    prof = P.resolve(reagent, peaks)
    context = context or prof.context
    # The height-gated passes gate on a MULTIPLE of each sample's own noise edge.
    # Resolve that multiple ONCE for the batch -- the profile's own value when it
    # carries one, else the package default; a cfg the caller already set wins --
    # so every per-file run gates identically and the summary can record it.
    from peaky.assignment import passes as PA

    cfg = assign_kw.get("cfg") or PA.PassConfig()
    x_edge, x_edge_source = P.apply_height_cutoff_x_edge(cfg, prof, log=log)
    assign_kw["cfg"] = cfg
    selection = dict(selection_meta or {})
    sel = None                 # our own cover table (None on the sample_ids= path)
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
    # each file's S/N: on the Texas Ur+ run C15H22 [M+NH4]+ kept its reading in
    # the 14 files that also held C15H22 [M+H]+ and was re-read to C15H25N [M+H]+
    # in the one file that did not -- a "disagreement" the spectra never had,
    # and a phantom minority reading for the vote. The merged ledger holds the
    # union of every file's [M+H]+ rows, so the same rule applied there gives the
    # batch one answer. An explicit reagent_n_relabel=True in assign_kw restores
    # the per-file re-read (and the merged-level pass then stands down).
    assign_kw.setdefault("reagent_n_relabel", False)
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
    identified_aux: list = []  # per-file identified-ion rows (reagent/iso/artifact
                               # + analyte ion_formula) for the parquet stamp
    plaus_audit: list = []     # per-file O-monster / carbon-cluster demotes, pooled
    protected_neutrals: set = set()   # curated/cross-channel identities the amine
    #   gate must not re-read (reflist / known-species / certified provenance) --
    #   e.g. NBBS, whose weak isobar-contaminated NH4 trace fails the tracking test
    #   yet is a genuine Keller-list contaminant adduct.
    curated_neutrals: set = set()     # the reflist / known-species subset: the merge
    #   vote's exemption (a list identity is not outvoted by grid readings)
    stages: dict = {}          # sid -> STAGE_COVER | STAGE_RESIDUAL (align() reads it)
    n_jobs = _resolve_jobs(n_jobs, len(sample_ids))
    ts_path_written: list = []  # the raw-TS parquet the worker pool loads: written once

    def _apply(sid, led, plaus, stats, stage):
        """Parent-side reduce (called in sample_ids order): write the per-file CSV
        and fold this sample into the accumulators. Order-fixed so align() -- which
        has order-sensitive tie-breaks -- yields byte-identical output either path."""
        led.to_csv(os.path.join(pfdir, f"{sid}_ledger.csv"), index=False)
        plaus_audit.extend(plaus)
        protected_neutrals.update(_protected_neutrals(led))
        curated_neutrals.update(_curated_neutrals(led))
        per_file[sid] = _m0(led)
        stages[sid] = stage
        from peaky.batch import timeseries as _TSI
        identified_aux.append(_TSI.identified_rows(led))
        try:
            offsets[sid] = IO.estimate_offset(IO.fetch_peaks(client, sid, use_cache=True))
        except Exception:
            offsets[sid] = None
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

    _assign_files(list(sample_ids), STAGE_COVER, n_jobs)

    from peaky.chem import reagents as _RG
    from peaky.assignment import plausibility as PL
    from peaky.batch import timeseries as _TS
    _rgk = _RG.reagent_for_adducts(list(prof.adducts or []))

    def _merge() -> dict:
        """align() over EVERY per-file ledger so far, then the merged-level
        guards and re-reads, the trace reconciliation and the whole-batch stamp
        -- returned as one record (merged, jitter, merge_gates, trace_info,
        stamp_tol, ts_annot), nothing written. Runs once per stage: the cover's
        record is what the residual universe is read from, and the last call
        (cover + residual files, ONE align) is what the run writes."""
        merged, jitter = align(per_file, tol_ppm=tol_ppm, offsets=offsets,
                               curated=curated_neutrals,
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
        stamp_tol = tol_ppm
        if _idx is not None and len(merged):
            trace_info = _TS.recentre_ledger(merged, index=_idx, tol_ppm=tol_ppm, log=log)
            trace_info.update(_TS.collapse_trace_labels(merged, tol_ppm=tol_ppm, log=log))
            stamp_tol, _sigma = _TS.stamp_tolerance(_idx, merged["mz_trace"], tol_ppm=tol_ppm)
            trace_info.update(stamp_tol_ppm=float(stamp_tol),
                              sigma_ppm=None if not np.isfinite(_sigma) else float(_sigma))
            log(f"[traces] per-ion mass scatter {_sigma if np.isfinite(_sigma) else 'n/a'} ppm -> "
                f"stamping window +-{stamp_tol:g} ppm (merge tolerance {tol_ppm:g})")
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
                                                 stats=_stats)
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
        "merged_M0": int(len(merged)),
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
            "residual_samples": rsel, "stages": dict(stages)}
