"""Batch assignment over a REPRESENTATIVE sample subset, with per-file records.

This realises the sample-selection RULE (see sampling.py): rather than assign a
single averaged file (which misses analytes present only part of the run), we
assign each of the representative files SEPARATELY and combine. match_compounds
is per-sample — a synthetic union spectrum can't be scored — so combining real
per-file ledgers is the only principled path.

We keep every per-file ledger on disk (out_dir/per_file/<sid>_ledger.csv) so the
file-to-file JITTER can be investigated: does the same m/z get the same formula /
tier in every file, and is its mass spread real or just per-file calibration?

The combine step is OFFSET-AWARE: each file carries a median mass offset
(io_mascope.estimate_offset); clustering aligns peaks on offset-corrected m/z so a
genuine same-peak is not split by a per-file calibration shift, while the reported
jitter separates the raw spread from the calibration-removed (residual) spread.

`align()` / `merge_union()` are PURE (offline-tested). `run()` does the network
assignment loop.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from peaky import paths as PT
from peaky.chem import profiles as P
from peaky.batch import sampling as SS

__version__ = "0.3.0"  # + sample-level process parallelism (--jobs)

DEFAULT_TOL_PPM = 6.0
TIER_ASSIGNED = "Assigned"
TIER_RANK = {"Assigned": 2, "Candidate": 1}
_M0_COLS = ["mz", "neutral_formula", "adduct", "tier", "ion_score"]


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


def align(per_file: dict, *, tol_ppm: float = DEFAULT_TOL_PPM,
          offsets: dict | None = None):
    """Align the M0 rows of several files by m/z.

    per_file : {src -> DataFrame with _M0_COLS}. offsets : {src -> median ppm}
    (subtracted before clustering so a per-file calibration shift does not split
    a peak). Returns (merged, jitter):

      merged  one row per m/z cluster: consensus mz, best assignment
              (Assigned>Candidate>ion_score), n_files, srcs, formula_agree,
              mz_jitter_ppm_raw, mz_jitter_ppm_caldj.
      jitter  long form, one row per (cluster, file): cluster, src, mz,
              neutral_formula, adduct, tier, ion_score.
    """
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
                                      "ion_score", "n_files", "srcs", "formula_agree",
                                      "mz_jitter_ppm_raw", "mz_jitter_ppm_caldj"]),
                pd.DataFrame(columns=["cluster", "src", *_M0_COLS]))
    allm = pd.concat(frames, ignore_index=True).sort_values("_mz_adj").reset_index(drop=True)
    allm["cluster"] = _cluster_mz(allm["_mz_adj"].to_numpy(), tol_ppm)

    merged_rows, jitter_rows = [], []
    for cid, g in allm.groupby("cluster"):
        g = g.assign(_r=g["tier"].map(lambda t: TIER_RANK.get(str(t), 0)))
        # Cross-file consensus: prefer the formula with the broadest CORROBORATED
        # (Assigned-tier) support across files, not the single highest per-file
        # ion_score. Per-file mass-calibration jitter can flip a degenerate pair in
        # one file (a competitor reads on-cal there, Assigned, with a marginally
        # higher local score) while the other files agree on the real formula; the
        # old "best (tier, ion_score) row" then let that single-file outlier
        # displace a formula Assigned across many files. Rank formulas by
        # (Assigned-file count, file count, best tier, best score), then take that
        # formula's best per-file row. With one formula per cluster this is a no-op.
        supp = (g.assign(_a=(g["_r"] >= TIER_RANK[TIER_ASSIGNED]).astype(int))
                  .groupby("neutral_formula", dropna=False)
                  .agg(n_assigned=("_a", "sum"), n_files=("src", "nunique"),
                       best_r=("_r", "max"), best_ion=("ion_score", "max")))
        win_formula = supp.sort_values(
            ["n_assigned", "n_files", "best_r", "best_ion"], ascending=False).index[0]
        gw = g[g["neutral_formula"] == win_formula] if win_formula == win_formula else g
        best = gw.sort_values(["_r", "ion_score"], ascending=False).iloc[0]
        mz_raw = g["mz"].to_numpy(); mz_adj = g["_mz_adj"].to_numpy()
        def _spread(a):
            return float((a.max() - a.min()) / a.mean() * 1e6) if len(a) > 1 else 0.0
        forms = set(g["neutral_formula"].dropna())
        merged_rows.append(dict(
            mz=float(g["mz"].mean()), neutral_formula=best["neutral_formula"],
            adduct=best.get("adduct"), tier=best["tier"],
            ion_score=best.get("ion_score"),
            n_files=int(g["src"].nunique()), srcs=",".join(sorted(set(g["src"]))),
            formula_agree=(len(forms) <= 1),
            mz_jitter_ppm_raw=round(_spread(mz_raw), 3),
            mz_jitter_ppm_caldj=round(_spread(mz_adj), 3)))
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


def _protected_neutrals(ledger: pd.DataFrame) -> set:
    if not {"method", "neutral_formula"} <= set(ledger.columns):
        return set()
    meth = ledger["method"].astype(str)
    keep = meth.str.startswith(_PROTECTED_METHODS)
    return set(ledger.loc[keep, "neutral_formula"].dropna().astype(str)) - {"nan", ""}


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
# network: assign each representative file, keep per-file records, combine
# ---------------------------------------------------------------------------
def run(peaks=None, *, batch: str | None = None, dataset: str | None = None,
        reagent: str = "auto", context: str | None = None,
        n_time: int = SS.N_TIME, include_max_tic: bool = True,
        select: str = "representative", coverage_target: float = 0.85,
        k_max: int = 10, height_floor: float = 1000.0,
        out_dir: str, tol_ppm: float = DEFAULT_TOL_PPM,
        sample_ids: list | None = None, ts_peaks=None, amine_r_min: float = 0.6,
        n_jobs: int | None = None, log=print, **assign_kw) -> dict:
    """Assign the representative subset of a batch and combine, keeping per-file
    ledgers. Provide EITHER `peaks` (a batch peak/sample table) OR `batch` (a
    batch name; the per-sample list is fetched fresh from the live server, which
    also guarantees the selected sample ids are valid for get_peaks — cached ids
    go stale / 404 when the server copy is renamed). `context` defaults to the
    reagent profile's context. Extra kwargs pass through to assign.run. Writes
    (see paths.RunPaths): merged_ledger.csv + batch_summary.json at the run root,
    per_file/<sid>_ledger.csv, and tables/{selected_samples,jitter}.csv."""
    from peaky.assignment import assign as A
    from peaky.io import io_mascope as IO

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
    if sample_ids is None:
        if select == "brightest":
            # bin ALL batch peaks -> assign each significant bin's BRIGHTEST sample.
            # Needs the per-PEAK table (height per peak): the pipeline passes it as
            # ts_peaks; fall back to `peaks` if it already is per-peak.
            src = ts_peaks if ts_peaks is not None else peaks
            sel = SS.select_brightest_coverage_samples(
                src, coverage_target=coverage_target, k_max=k_max,
                height_floor=height_floor)
            log(f"[assign_batch] brightest-coverage: {len(sel)} winner samples "
                f"(target {coverage_target:.0%}, floor {height_floor:g} cps)")
        else:
            sel = SS.select_representative_samples(peaks, n_time=n_time,
                                                   include_max_tic=include_max_tic)
        sample_ids = sel["sample_item_id"].tolist()
        sel.to_csv(os.path.join(TAB, "selected_samples.csv"), index=False)
    log(f"[assign_batch] {prof.label} context={context!r}: "
        f"{len(sample_ids)} representative files -> {pfdir}")

    # Force the reagent's analyte channels (we know the reagent at batch level) so
    # a per-sample match gap can't flip polarity / mis-assign a file. Caller can
    # still override via assign_kw['adducts'].
    assign_kw.setdefault("adducts", list(prof.adducts))
    # labelled-reagent covalent-product rescue (e.g. 15N-organonitrates); no-op
    # for every unlabelled reagent profile.
    if getattr(prof, "label_isotope", None):
        assign_kw.setdefault("label_isotope", prof.label_isotope)
        assign_kw.setdefault("label_max", prof.label_max)
    # thread the batch TS to the per-sample run so pass-7 (certified-neutral)
    # can use member-channel co-variation as OPTIONAL corroboration. Guarded:
    # the pass is fully functional with ts_peaks=None (single-sample runs, or
    # batches whose mass range excludes the reagent ions).
    if ts_peaks is not None:
        assign_kw.setdefault("ts_peaks", ts_peaks)
    # context-unlock the reference peaklists (contaminants always; chemistry-
    # specific lists when the batch metadata matches) -> selection prior + rescue.
    from peaky.assignment import reflists as RL
    _tags = RL.resolve_context_tags(batch or "", getattr(prof, "label", ""))
    reflists_active = RL.active_lists(RL.load_catalog(), context_tags=_tags)
    if reflists_active:
        log(f"[assign_batch] reference lists active: {[rl.id for rl in reflists_active]} "
            f"(context {sorted(_tags) or 'contaminants-only'})")
    per_file, offsets, per_stats = {}, {}, []
    plaus_audit: list = []     # per-file O-monster / carbon-cluster demotes, pooled
    protected_neutrals: set = set()   # curated/cross-channel identities the amine
    #   gate must not re-read (reflist / known-species / certified provenance) --
    #   e.g. NBBS, whose weak isobar-contaminated NH4 trace fails the tracking test
    #   yet is a genuine Keller-list contaminant adduct.
    n_jobs = _resolve_jobs(n_jobs, len(sample_ids))

    def _apply(sid, led, plaus, stats):
        """Parent-side reduce (called in sample_ids order): write the per-file CSV
        and fold this sample into the accumulators. Order-fixed so align() -- which
        has order-sensitive tie-breaks -- yields byte-identical output either path."""
        led.to_csv(os.path.join(pfdir, f"{sid}_ledger.csv"), index=False)
        plaus_audit.extend(plaus)
        protected_neutrals.update(_protected_neutrals(led))
        per_file[sid] = _m0(led)
        try:
            offsets[sid] = IO.estimate_offset(IO.fetch_peaks(client, sid, use_cache=True))
        except Exception:
            offsets[sid] = None
        st = dict(stats)
        st.update(sample_id=sid, offset_ppm=offsets[sid],
                  n_M0=int((led["role"] == "M0").sum()) if "role" in led.columns else None)
        per_stats.append(st)
        log(f"[assign_batch]   {sid}: offset={offsets[sid]}")

    if n_jobs <= 1:
        for i, sid in enumerate(sample_ids, 1):
            log(f"[assign_batch] ({i}/{len(sample_ids)}) assigning {sid} ...")
            res = A.run(sid, context=context, log=log,
                        reflists_active=reflists_active, **assign_kw)
            _apply(sid, res["ledger"], res.get("plausibility_audit") or [],
                   dict(res.get("stats", {})))
    else:
        # Write the batch TS to disk once so workers load it from the parquet
        # rather than re-pickling the full-batch DataFrame into every process.
        base_kw = {k: v for k, v in assign_kw.items() if k != "ts_peaks"}
        ts_path = None
        _ts = assign_kw.get("ts_peaks")
        if _ts is not None:
            ts_path = os.path.join(pfdir, "_batch_ts.parquet")
            _ts.to_parquet(ts_path)
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
            futs = {ex.submit(_assign_one, sid): sid for sid in sample_ids}
            for done, fut in enumerate(as_completed(futs), 1):
                out = fut.result()
                results[out["sid"]] = out
                log(f"[assign_batch] ({done}/{len(sample_ids)}) done {out['sid']}")
        # Reduce STRICTLY in sample_ids order (not completion order) so align()'s
        # input is fixed and the merged output is byte-identical to a serial run.
        for sid in sample_ids:
            out = results[sid]
            for ln in out["log"]:              # replay worker logs, grouped per sid
                log(ln)
            _apply(sid, out["ledger"], out["plausibility_audit"], out["stats"])

    merged, jitter = align(per_file, tol_ppm=tol_ppm, offsets=offsets)
    # Merge guard: drop reagent-cluster ions a per-file pass mislabelled as analyte
    # (urea [R_n+H]+/[R_n+NH4]+ read as CHNO/CH4N2O on the [M+NH4]+/urea channel) --
    # they otherwise dominate the 'assigned' signal. Belt-and-braces with the
    # per-file reagent lock/reclaim (older per-file ledgers predate that fix).
    from peaky.chem import reagents as _RG
    _rgk = _RG.reagent_for_adducts(list(prof.adducts or []))
    if _rgk:
        merged, _rgstrip = _RG.strip_reagent_cluster_rows(merged, _rgk, log=log)
    # Positive urea-CIMS: re-read uncorroborated [M+NH4]+ adducts as [M+H]+ of the
    # +NH3 amine (mass/isotope-identical; simpler in an N-rich source). Done at the
    # MERGED level where cross-channel corroboration is complete.
    if prof.polarity == "+":
        from peaky.assignment import cleanup
        cleanup.prefer_amine_over_ammonium(merged, ts_peaks=ts_peaks, r_min=amine_r_min,
                                           protected=protected_neutrals, log=log)
    from peaky.assignment import plausibility as PL
    summary_plaus = {}
    # one audit row per touched peak (per-file O/C-monster + carbon-cluster demotes);
    # always written for a stable artifact set.
    n_audit = PL.write_audit(plaus_audit, os.path.join(TAB, f"plausibility_audit_{prof.name}.csv"))
    log(f"[assign_batch] plausibility audit: {n_audit} touched peaks "
        f"-> tables/plausibility_audit_{prof.name}.csv")
    # Sidelobe-contaminated CHANNELS: an assigned ion whose m/z lands on the ringing
    # sidelobe of a saturating neighbour keeps its formula (the neutral is usually
    # corroborated on another channel) but its HEIGHT is the neighbour's, not the
    # analyte's. Only the time series separates that from a real ion that merely sits
    # near a bright peak, so it is decided HERE, not in per-file cleanup.
    # Called unconditionally so the merged-ledger SCHEMA is stable: without a TS it
    # no-ops and the two columns are still present (all False / NaN).
    from peaky.batch import timeseries as _TSF
    _TSF.flag_sidelobe_channels(merged, ts_peaks, log=log)
    merged.to_csv(os.path.join(out_dir, "merged_ledger.csv"), index=False)
    jitter.to_csv(os.path.join(TAB, "jitter.csv"), index=False)
    # Stamp the batch time-series peaks with their assigned formula/channel and write
    # the FINAL per_file/_batch_ts.parquet (in parallel mode this overwrites the raw
    # worker-transfer copy). Downstream time-series analysis then has neutral_formula /
    # adduct / tier / ion_mz per peak, not just m/z. No-op when ts_peaks is unavailable.
    if ts_peaks is not None:
        from peaky.batch import timeseries as _TS
        ts_annot = _TS.annotate_peaks(ts_peaks, merged, tol_ppm=tol_ppm)
        ts_annot.to_parquet(os.path.join(pfdir, "_batch_ts.parquet"))
        _n_ass = int(ts_annot["neutral_formula"].notna().sum())
        log(f"[assign_batch] _batch_ts.parquet: {len(ts_annot)} peaks, {_n_ass} "
            f"({_n_ass / max(len(ts_annot), 1):.0%}) matched to an assigned "
            f"formula/channel (tol {tol_ppm:.0f} ppm)")
        # one-to-one guarantee: each (sample, ion) is stamped on at most ONE peak,
        # so a downstream groupby(formula, adduct) sees one trace per sample. The
        # shoulder/split peaks that lost are kept, unstamped, flagged dup_candidate.
        _n_dup = int(ts_annot["dup_candidate"].sum())
        if _n_dup:
            _ions = ts_annot.loc[ts_annot["dup_candidate"], "mz"].round(3).nunique()
            log(f"[assign_batch] one-to-one: {_n_dup} near-duplicate peak(s) at "
                f"~{_ions} m/z left unstamped (flagged dup_candidate) so no ion is "
                f"stamped twice in one sample")

    summary = {
        "reagent": prof.name, "label": prof.label, "context": context,
        "batch_name": batch,
        "select": select,
        "coverage_target": (coverage_target if select == "brightest" else None),
        "n_files": len(sample_ids), "sample_ids": sample_ids,
        "tol_ppm": tol_ppm, "offsets_ppm": offsets,
        "merged_M0": int(len(merged)),
        "merged_tiers": merged["tier"].value_counts().to_dict() if len(merged) else {},
        "n_in_all_files": int((merged["n_files"] == len(sample_ids)).sum()) if len(merged) else 0,
        "n_single_file": int((merged["n_files"] == 1).sum()) if len(merged) else 0,
        "formula_disagreements": int((~merged["formula_agree"]).sum()) if len(merged) else 0,
        "plausibility": summary_plaus,
        "plausibility_audit_rows": n_audit,
        "per_file": per_stats,
    }
    with open(os.path.join(out_dir, "batch_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    log(f"[assign_batch] DONE: {summary['merged_M0']} merged M0 "
        f"({summary['merged_tiers']}); {summary['n_in_all_files']} in all files, "
        f"{summary['n_single_file']} single-file, "
        f"{summary['formula_disagreements']} formula disagreements")
    return {"profile": prof, "context": context, "sample_ids": sample_ids,
            "per_file": per_file, "offsets": offsets, "merged": merged,
            "jitter": jitter, "summary": summary, "out_dir": out_dir}
