"""Pipeline spine — `run(batch, reagent)` instead of a copy-pasted script per
dataset. Resolves a reagent profile, loads the batch (SDK or a cached parquet),
and dispatches the requested stages, all parameterised by the profile.

This is the thin orchestrator the rest of agent-peaky hangs off. Stage functions
live in their own modules and are called with profile params — nothing reagent-
or batch-specific is hardcoded here.

Stages: 'matrix' (TS intensity matrix) is wired; 'assign' / 'cluster' / 'validate'
are folded in as their scratch logic is consolidated into the package.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone

import pandas as pd

from peaky.io import io_mascope as IO
from peaky import paths as PT
from peaky.chem import profiles as P
from peaky.batch import sampling as SS
from peaky.batch import timeseries as TS

__version__ = "0.2.0"  # + representative-sample selection (5 time-grid + max-TIC)

# Content-stable epoch for SOURCE_DATE_EPOCH. NOT the run time: figures, workbooks
# and the ledger are a PURE FUNCTION of the input data, so their embedded metadata
# timestamps must be constant — a re-run of the same data, whenever it happens,
# yields byte-identical figures/tables. The RUN time reaches output only as visible
# text (PDF cover "generated" + Report ID), the run-folder name, and run_manifest.json.
# 315532800 = 1980-01-01T00:00:00Z (also the zip-format mtime floor the xlsx writer needs).
CONTENT_EPOCH = 315532800

STAGES = ("matrix", "assign", "cluster", "validate")


# ---------------------------------------------------------------------------
# run versioning: every set of outputs goes in its OWN timestamped folder so a
# re-run never overwrites a previous one. The folder name = the run id = the
# batch slug + date + time, and that same id is stamped on the report cover.
# Timestamps are in UTC (explicitly marked Z / "UTC") — the run environment's
# local clock can differ from the user's, so UTC is unambiguous for sharing.
# Pass `when` explicitly (one now() per run) so folder/id/cover all agree.
# ---------------------------------------------------------------------------
def slugify(name: str) -> str:
    """Filesystem-safe batch slug: 'Sample run (Ur+ CIMS)' -> 'Sample-run-Ur-CIMS'."""
    return re.sub(r"[^0-9A-Za-z]+", "-", str(name)).strip("-") or "run"


def run_stamp(when: datetime | None = None) -> tuple[str, str]:
    """(folder stamp 'YYYY-MM-DDTHHMMSSZ', human 'YYYY-MM-DD HH:MM UTC') in UTC.
    `when`: None -> now (UTC); a tz-aware datetime is converted to UTC; a naive
    datetime is assumed to already be UTC."""
    if when is None:
        when = datetime.now(timezone.utc)
    elif when.tzinfo is not None:
        when = when.astimezone(timezone.utc)
    return when.strftime("%Y-%m-%dT%H%M%SZ"), when.strftime("%Y-%m-%d %H:%M UTC")


def run_id(batch_name: str, when: datetime | None = None) -> str:
    """Versioned run id = batch slug + timestamp; also the output folder's name."""
    return f"{slugify(batch_name)}_{run_stamp(when)[0]}"


def stamp_source_date_epoch(when=None) -> str:
    """Export the FIXED ``CONTENT_EPOCH`` as ``SOURCE_DATE_EPOCH`` so every renderer
    stamps a constant time, not ``now()`` and not the run time. matplotlib reads it
    for PNG/PDF metadata and the xlsx writer reads it via ``cluster._resolve_when``;
    this is matplotlib's ONLY reproducible-stamp hook. Pinning it to a constant makes
    figures, workbooks and tables a PURE FUNCTION OF THE INPUT DATA — the same data
    re-run at any time yields byte-identical figures/tables. Run time is NOT carried
    here; it reaches output only as visible PDF-cover text + the Report ID + the
    run-folder name. `when` is accepted for back-compat but IGNORED. Returns the epoch."""
    epoch = str(CONTENT_EPOCH)
    os.environ["SOURCE_DATE_EPOCH"] = epoch
    return epoch


def make_run_dir(base: str, batch_name: str, when: datetime | None = None) -> str:
    """Create and return a fresh per-run output folder `base/<run_id>`."""
    d = os.path.join(os.path.expanduser(base), run_id(batch_name, when))
    os.makedirs(d, exist_ok=True)
    return d


def load(*, batch: str | None = None, dataset: str | None = None,
         peaks: "str | pd.DataFrame | None" = None, save_path: str | None = None
         ) -> pd.DataFrame:
    """Get the batch peak time-series — from a parquet/DataFrame if given (offline,
    cached), else fetched from Mascope via the SDK."""
    if peaks is not None:
        return pd.read_parquet(os.path.expanduser(peaks)) if isinstance(peaks, str) else peaks
    if not (batch and dataset):
        raise ValueError("need peaks=, or both batch= and dataset=")
    return IO.fetch_batch_peaks(IO.connect(), dataset, batch, save_path=save_path)


def run(*, batch: str | None = None, dataset: str | None = None,
        peaks: "str | pd.DataFrame | None" = None, reagent: str = "auto",
        stages: tuple = ("matrix",), out_dir: str | None = None,
        n_time: int = SS.N_TIME, include_max_tic: bool = True) -> dict:
    """Run the pipeline on one batch.

    Returns a dict with at least {profile, peaks, n_samples, assign_samples}
    plus per-stage outputs. Pass reagent by name ('Br'/'Ur') or 'auto' to detect
    from the data.

    THE RULE (always computed, regardless of stages): `assign_samples` is the
    representative subset to assign + merge — `n_time` samples evenly spaced in
    TIME plus the max-TIC sample (see sampling.py). `assign_sample_ids` is the
    bare id list. Assignment runs on these, not on a single averaged file, so the
    merged peak list covers analytes that only appear at part of the run.
    """
    pk = load(batch=batch, dataset=dataset, peaks=peaks)
    prof = P.resolve(reagent, pk)
    n_samples = pk["sample_item_id"].nunique() if "sample_item_id" in pk.columns else None
    assign_samples = SS.select_representative_samples(
        pk, n_time=n_time, include_max_tic=include_max_tic)
    out: dict = {"profile": prof, "peaks": pk, "n_samples": n_samples,
                 "assign_samples": assign_samples,
                 "assign_sample_ids": assign_samples["sample_item_id"].tolist()
                 if "sample_item_id" in assign_samples.columns else [],
                 "stages": tuple(stages)}

    if "matrix" in stages:
        mat, bin_mz = TS.build_matrix(pk)
        out["matrix"] = mat
        out["bin_mz"] = bin_mz

    # 'assign' / 'cluster' / 'validate' are added as the scratch modules
    # (cluster_*, isotope_validate) are consolidated into the package against
    # isotopes.py. Flag clearly until then rather than silently no-op. NOTE the
    # 'assign' stage's sample SELECTION is already live (assign_samples above);
    # what remains is to run assign.run per selected id and merge the ledgers.
    todo = [s for s in stages if s in ("assign", "cluster", "validate")]
    if todo:
        out["pending_stages"] = todo
        if n_samples and n_samples < 12 and "cluster" in todo:
            out["cluster_warning"] = (f"{n_samples} samples < 12: temporal clustering "
                                      "unreliable; isotope/coverage layer preferred")

    if out_dir:
        os.makedirs(os.path.expanduser(out_dir), exist_ok=True)
    return out


# ---------------------------------------------------------------------------
# RunContext + the batch pipeline (assign -> cluster -> Van Krevelen -> report).
# Replaces the out-of-repo run_peaky.py orchestrator and its PEAKY_* env-var +
# subprocess threading: ONE object carries the run's identity (out_dir / run_id /
# generated stamp / batch / profile) into every stage as a plain argument.
# ---------------------------------------------------------------------------
@dataclass
class RunContext:
    out_dir: str                 # the versioned run folder (holds all artifacts)
    batch_name: str
    tag: str                     # filename token (e.g. the profile name)
    label: str                   # display label for figure/report titles
    when: datetime
    run_id: str                  # == folder basename; stamped on the report cover
    generated: str               # human stamp 'YYYY-MM-DD HH:MM UTC'
    profile: object = None       # ReagentProfile
    dataset: str | None = None
    ts_path: str | None = None   # parquet the report reads for the event TIC


def make_run_context(base_out: str, batch_name: str, profile, *, when=None,
                     tag=None, label=None, dataset=None) -> RunContext:
    """Create a fresh timestamped run folder and the context that identifies it.
    Pass ONE `when` per run so folder / run_id / cover stamp all agree."""
    when = when or datetime.now(timezone.utc)
    return RunContext(
        out_dir=make_run_dir(base_out, batch_name, when), batch_name=batch_name,
        tag=tag or profile.name, label=label or profile.label, when=when,
        run_id=run_id(batch_name, when), generated=run_stamp(when)[1],
        profile=profile, dataset=dataset)


def generate_report(ctx: RunContext, ts, *, subject: str | None = None,
                    do_cluster=True, do_vk=True, do_report=True, log=print) -> dict:
    """Offline generation half: cluster figures + Van Krevelen + the PDF report,
    from the merged / per-file ledgers already in `ctx.out_dir` and the batch time
    series. No network. Same artifacts the run_clusters / run_vankrevelen /
    run_report scratch chain produced, but in-process via the RunContext."""
    from peaky.reporting import analyte_viz as V
    from peaky.batch import clustering as CLU
    from peaky.reporting import pdf_report as R

    # Pin figures/PDF/workbooks to a FIXED content epoch (not the run time) so the
    # scientific content is byte-identical regardless of WHEN it's run. The run time
    # appears only as cover text + Report ID (ctx.generated / ctx.run_id below).
    stamp_source_date_epoch()

    if isinstance(ts, str):
        ctx.ts_path = os.path.expanduser(ts)        # reference an existing parquet, never copy
        ts = pd.read_parquet(ctx.ts_path)
    elif ctx.ts_path is None:
        # ts was fetched live (no on-disk source) — keep ONE copy with the run, in
        # data/, so the report/provenance can read it. (When the caller passed a
        # parquet path, run_batch points ctx.ts_path at it instead of copying.)
        ctx.ts_path = os.path.join(PT.run_paths(ctx.out_dir).ensure().data,
                                   f"{ctx.tag}_ts.parquet")
        ts.to_parquet(ctx.ts_path)

    out: dict = {"ctx": ctx}
    if do_cluster:
        out["cluster"] = CLU.cluster_batch(ctx.out_dir, ts, ctx.profile,
                                           tag=ctx.tag, label=ctx.label, log=log)
    if do_vk:
        out["vk"] = V.van_krevelen_batch(ctx.out_dir, ts, ctx.profile, tag=ctx.tag,
                                         label=ctx.label, batch_name=ctx.batch_name,
                                         subject=subject, log=log)
    if do_report:
        out["report_pdf"] = R.build(ctx.out_dir, tag=ctx.tag, label=ctx.label,
                                     ts_path=ctx.ts_path, batch_name=ctx.batch_name,
                                     run_id=ctx.run_id, generated=ctx.generated)
        log(f"[report] wrote {out['report_pdf']}")
        # also emit a size-reduced companion for emailing (optional deps; no-op if
        # absent or already small). The full report above is left untouched so it
        # stays byte-for-byte deterministic.
        small = R.compress_pdf(out["report_pdf"], log=log)
        if small:
            out["report_pdf_small"] = small
            log(f"[report] compressed -> {small} ({os.path.getsize(small) / 1e6:.1f} MB)")
    return out


def run_batch(*, batch: str, dataset: str | None = None, reagent: str = "auto",
              base_out: str, ts=None, when=None, subject: str | None = None,
              amine_r_min: float = 0.6, do_report=True, config: str | None = None,
              select: str = "representative", coverage_target: float = 0.85,
              k_max: int = 10, height_floor: float = 1000.0,
              n_jobs: int | None = None, log=print, **assign_kw) -> dict:
    """Full batch pipeline in ONE call: sample-subset ASSIGN (live match_compounds)
    -> merge -> cluster figures -> Van Krevelen -> PDF report, into one versioned run
    folder. `ts` is the full-batch per-sample peak time series (DataFrame or parquet
    path); if None it is fetched live and reused for the amine gate + clustering.

    `select` picks the assigned-sample strategy: 'representative' (THE RULE: 5
    time-spaced + max-TIC) or 'brightest' (bin all peaks -> assign each significant
    m/z bin's brightest sample; `coverage_target`/`k_max`/`height_floor` tune it).
    Returns {ctx, assign, cluster, vk, report_pdf}."""
    from peaky.batch import assign_batch as AB

    ts_src = None
    if isinstance(ts, str):
        ts_src = os.path.expanduser(ts)            # an on-disk parquet we can reference
        ts = pd.read_parquet(ts_src)
    if ts is None:
        log(f"[batch] fetching full-batch time series for {batch!r} ...")
        ts = load(batch=batch, dataset=dataset)
    prof = P.resolve(reagent, ts, config=config)
    ctx = make_run_context(base_out, batch, prof, when=when, dataset=dataset)
    if ts_src:
        ctx.ts_path = ts_src     # reference the caller's parquet; don't re-copy it into the run dir
    log(f"[batch] {ctx.run_id} -> {ctx.out_dir}")

    res = AB.run(batch=batch, dataset=dataset, reagent=prof.name,
                 out_dir=ctx.out_dir, ts_peaks=ts, amine_r_min=amine_r_min,
                 select=select, coverage_target=coverage_target, k_max=k_max,
                 height_floor=height_floor, n_jobs=n_jobs, log=log, **assign_kw)
    gen = generate_report(ctx, ts, subject=subject, do_report=do_report, log=log)

    # provenance: pin this run to its exact code + input-data hash + config +
    # output hash, and append it to the cross-run registry. Best-effort (never
    # fatal). Runs LAST so ts_path / merged_ledger.csv exist to be hashed.
    from peaky.assignment import passes as PA
    from peaky.reporting import provenance as PV
    summ = res.get("summary", {}) if isinstance(res, dict) else {}
    PV.record_run(
        run_dir=ctx.out_dir, base_out=os.path.expanduser(base_out),
        batch_name=batch, dataset=dataset,
        sample_ids=(res.get("sample_ids") if isinstance(res, dict) else None),
        reagent=prof.name, cfg=assign_kw.get("cfg") or PA.PassConfig(),
        ts_path=ctx.ts_path,
        counts={"merged_M0": summ.get("merged_M0"),
                "merged_tiers": summ.get("merged_tiers"),
                "n_samples": summ.get("n_files"),
                "select": summ.get("select"),
                "coverage_target": summ.get("coverage_target")},
        created_utc=ctx.when.isoformat(), log=log)
    return {"ctx": ctx, "assign": res, **gen}


def pool_name(batches_regex: str) -> str:
    """A readable run-folder label from a pooling regex: strip metacharacters,
    collapse whitespace, tag as pooled. `'HR-CIMS 100-500.*zone'` ->
    `'HR-CIMS 100-500 zone (pooled)'`."""
    base = re.sub(r"[.*+^$()\[\]{}\\?|]", " ", batches_regex)
    base = re.sub(r"\s+", " ", base).strip()
    return f"{base} (pooled)" if base else "pooled-batches"


def _write_selected_samples(run_dir: str, prov) -> None:
    """Emit tables/selected_samples.csv (the report's representative-sample section
    reads it). The pooled path passes sample_ids= to assign_batch.run, which skips
    that module's own writer, so we write the union provenance here instead."""
    tab = PT.run_paths(run_dir).ensure().tables
    prov.to_csv(os.path.join(tab, "selected_samples.csv"), index=False)


def _stage_pool_child(pool_dir: str, child_dir: str, group_prov) -> None:
    """Populate a per-group report folder from the finished pool run: the shared
    unified ledger + per-file ledgers + batch summary, plus this group's own
    selected-samples table. ONE place that knows what a report needs from a run
    folder, so a new report input is added here rather than silently missed."""
    import shutil
    for name in ("merged_ledger.csv", "batch_summary.json"):
        src = os.path.join(pool_dir, name)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(child_dir, name))
    pf_src = os.path.join(pool_dir, "per_file")
    if os.path.isdir(pf_src):
        shutil.copytree(pf_src, os.path.join(child_dir, "per_file"), dirs_exist_ok=True)
    _write_selected_samples(child_dir, group_prov)


def run_pooled_batches(*, batches: str, dataset: str | None = None,
                       reagent: str = "auto", base_out: str,
                       out_name: str | None = None,
                       group_by: str = "sample_batch_name", ts=None, when=None,
                       subject: str | None = None, amine_r_min: float = 0.6,
                       do_report: bool = True, per_group_reports: bool = True,
                       config: str | None = None, coverage_target: float = 0.90,
                       k_max: int = 6, height_floor: float = 1000.0,
                       n_jobs: int | None = None, log=print, **assign_kw) -> dict:
    """Pool the batches matching `batches` (a regex over batch names) into ONE
    unified ledger, then emit a whole-pool report plus one report per group.

    The value: for a campaign split into many batches of the SAME chemistry (e.g.
    per-zone / per-segment batches of one mode x range), a single unified ledger is the
    right peak list -- every analyte present ANYWHERE is discovered once, and each
    group's own time series is then read against that shared list. Selection uses a
    per-GROUP brightest-coverage UNION (`sampling.select_pooled_union`) so no group
    is starved by a louder one (see that function). `A.run` fetches each selected
    sample's peaks from the server itself; the pooled `ts` only drives selection,
    clustering and the amine gate, so it is trimmed to the TS columns for the workers.

    `group_by` is the column that splits the pool (default `sample_batch_name`: one
    group per matched batch). Per-group reports share the unified `merged_ledger.csv`
    + `per_file/` (copied in) and re-cluster on that group's TS slice only.

    Returns {ctx, assign, groups, group_runs, selection, cluster, vk, report_pdf}.
    """
    from peaky.batch import assign_batch as AB

    when = when or datetime.now(timezone.utc)     # ONE stamp: pool + every group ctx agree
    if isinstance(ts, str):
        ts = pd.read_parquet(os.path.expanduser(ts))
    if ts is None:
        log(f"[pool] loading pooled TS for /{batches}/ in {dataset!r} ...")
        ts = IO.fetch_pooled_peaks(IO.connect(), dataset, batches)
    if group_by not in ts.columns:
        raise ValueError(f"group_by {group_by!r} not in pooled peaks "
                         f"(got {list(ts.columns)[:8]})")
    n_ungrouped = int(ts[group_by].isna().sum())
    if n_ungrouped:
        log(f"[pool] WARNING: {n_ungrouped} peak rows have no {group_by!r} value "
            f"-- those samples are excluded from selection and reports")
    log(f"[pool] {len(ts):,} peak rows, {ts['sample_item_id'].nunique()} samples, "
        f"{ts[group_by].nunique()} groups by {group_by!r}")

    union, prov = SS.select_pooled_union(
        ts, group_col=group_by, coverage_target=coverage_target, k_max=k_max,
        height_floor=height_floor)
    # group iteration order == selection order (prov is groupby-sorted), so the
    # per-group reports and selection_provenance never disagree on ordering.
    groups = list(dict.fromkeys(prov[group_by].astype(str)))
    log(f"[pool] per-group brightest union: {len(union)} samples (k_max={k_max})\n"
        + prov.groupby(group_by).size().to_string())

    # trim to the TS columns so the worker-side parquet stays small (A.run fetches
    # each sample's assignment peaks itself; ts only drives selection/cluster/amine).
    ts_cols = [c for c in ("sample_item_id", "mz", "height", "datetime_utc")
               if c in ts.columns]
    prof = P.resolve(reagent, ts[ts_cols], config=config)
    pool_label = out_name or pool_name(batches)
    ctx = make_run_context(base_out, pool_label, prof, when=when, dataset=dataset)
    log(f"[pool] {ctx.run_id} -> {ctx.out_dir}")
    prov.to_csv(os.path.join(ctx.out_dir, "selection_provenance.csv"), index=False)

    # batch= carries the pool identity: it names batch_summary AND is what
    # reflists.resolve_context_tags reads to unlock chemistry-specific reference
    # lists (a chamber pool named e.g. 'apinene ...' -> the monoterpene list).
    res = AB.run(peaks=ts[ts_cols], ts_peaks=ts[ts_cols], reagent=prof.name,
                 batch=pool_label, sample_ids=union, out_dir=ctx.out_dir,
                 amine_r_min=amine_r_min, n_jobs=n_jobs, log=log, **assign_kw)
    # the report's representative-sample section reads tables/selected_samples.csv;
    # the sample_ids= path skips AB.run's own writer, so emit it from the union prov.
    _write_selected_samples(ctx.out_dir, prov)
    gen = generate_report(ctx, ts[ts_cols], subject=subject, do_report=do_report, log=log)

    group_runs = []
    if per_group_reports:
        gcol = ts[group_by].astype(str)
        for g in groups:
            dfg = ts.loc[gcol == g, ts_cols]           # label-safe slice on ts itself
            gctx = make_run_context(base_out, g, prof, when=when, dataset=dataset)
            _stage_pool_child(ctx.out_dir, gctx.out_dir,
                              prov[prov[group_by].astype(str) == g])
            generate_report(gctx, dfg, subject=subject, do_report=do_report, log=log)
            group_runs.append(gctx.out_dir)
            log(f"[pool] group {g!r} report -> {gctx.out_dir}")

    # provenance parity with run_batch: pin the pool run to its code/data/config/output.
    from peaky.assignment import passes as PA
    from peaky.reporting import provenance as PV
    summ = res.get("summary", {}) if isinstance(res, dict) else {}
    PV.record_run(
        run_dir=ctx.out_dir, base_out=os.path.expanduser(base_out),
        batch_name=pool_label, dataset=dataset, sample_ids=union, reagent=prof.name,
        cfg=assign_kw.get("cfg") or PA.PassConfig(), ts_path=ctx.ts_path,
        counts={"merged_M0": summ.get("merged_M0"),
                "merged_tiers": summ.get("merged_tiers"),
                "n_samples": summ.get("n_files"), "n_groups": len(groups),
                "select": "pooled-union", "coverage_target": coverage_target},
        created_utc=ctx.when.isoformat(), log=log)
    return {"ctx": ctx, "assign": res, "groups": groups, "group_runs": group_runs,
            "selection": prov, **gen}
