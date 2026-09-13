#!/usr/bin/env python
"""Score the trace-level stamp against the anchor stamp on a FINISHED batch run.

    python scripts/eval_trace_stamping.py <run_dir> [--targets targets.csv] [--tol 6]

Reads the run's `merged_ledger.csv` and its batch time series (`data/*_ts.parquet`
or `per_file/_batch_ts.parquet`), then stamps the time series twice from the SAME
ledger rows: (a) today's way -- a +-tol window around each merge anchor -- and
(b) the trace way -- `timeseries.recentre_ledger` + `collapse_trace_labels` +
`stamp_tolerance`, i.e. what `assign_batch.run` now does between the merge and
the stamp. For every ion it reports the fraction of spectra that received a
stamp, both ways, plus the batch-level summary the run writes.

`--targets` is a CSV with a `formula` column (and optionally `adducts`, a
`;`-separated list; default `[M-H]-;[M+NO3]-;[M+Br]-`): the ions known to be
present (an Orbitrap's ledger on the same air, a chamber's known products). The
benchmark rule for any stamping change is coverage of ions KNOWN to be present --
never how many rows were admitted or assigned -- and every coverage is a one-to-one
stamp count, so a wider window cannot inflate it by chaining. Runs offline; no
network; nothing is written into the run folder.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from peaky.batch import timeseries as TS   # noqa: E402
from peaky.batch import traces as TR       # noqa: E402
from peaky.chem import chemistry as C      # noqa: E402

DEFAULT_ADDUCTS = ("[M-H]-", "[M+NO3]-", "[M+Br]-")


def _ts_path(run_dir: str) -> str:
    for pat in ("per_file/_batch_ts.parquet", "data/*_ts.parquet"):
        hits = sorted(glob.glob(os.path.join(run_dir, pat)))
        if hits:
            return hits[0]
    raise FileNotFoundError(f"no time-series parquet under {run_dir}")


def _read_targets(path: str) -> pd.DataFrame:
    """A targets CSV: header `formula[,adducts][,name...]`; free-text columns may
    carry unquoted commas, so split each line on the first len(header)-1 commas
    only (a `name` column is never needed here)."""
    with open(path) as fh:
        lines = [ln.rstrip("\n") for ln in fh if ln.strip()]
    header = [h.strip() for h in lines[0].split(",")]
    want = [c for c in ("formula", "adducts") if c in header]
    rows = []
    for ln in lines[1:]:
        parts = ln.split(",", len(header) - 1)
        rec = dict(zip(header, parts))
        rows.append({c: rec.get(c, "") for c in want})
    return pd.DataFrame(rows)


def coverage_by_ion(annot: pd.DataFrame, n_spectra: int) -> pd.Series:
    """Stamped spectra per (neutral_formula, adduct) as a fraction of the batch --
    one-to-one stamps only (dup_candidate rows never count)."""
    d = annot[annot["neutral_formula"].notna() & ~annot["dup_candidate"].astype(bool)]
    return d.groupby(["neutral_formula", "adduct"])["sample_item_id"].nunique() / n_spectra


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--targets", default=None, help="CSV with a `formula` column (+ optional `adducts`)")
    ap.add_argument("--tol", type=float, default=None,
                    help="merge tolerance (default: the run's batch_summary tol_ppm, else 6)")
    ap.add_argument("--stamp-tol", type=float, default=None,
                    help="override the trace stamp's window (default: derived from the "
                         "batch's per-ion scatter, timeseries.stamp_tolerance)")
    ap.add_argument("--no-nitrate-alias", action="store_true",
                    help="do not count a target's [M+NO3]- through its exact isobar, "
                         "the organonitrate (M+HNO3) [M-H]- (a Br-only profile has no "
                         "nitrate channel, so that is how the ion appears in its ledger)")
    args = ap.parse_args(argv)

    run = os.path.expanduser(args.run_dir)
    merged = pd.read_csv(os.path.join(run, "merged_ledger.csv"))
    ts = pd.read_parquet(_ts_path(run), columns=["sample_item_id", "mz", "height"])
    ts = TS.collapse_peak_matches(ts)
    tol = args.tol
    if tol is None:
        try:
            import json
            tol = float(json.load(open(os.path.join(run, "batch_summary.json"))).get("tol_ppm", 6.0))
        except Exception:  # noqa: BLE001
            tol = 6.0
    n_spectra = int(ts["sample_item_id"].nunique())
    print(f"run: {run}\n  {len(merged)} merged rows, {len(ts)} peaks, {n_spectra} spectra, tol {tol:g} ppm")

    base_cols = [c for c in ("mz", "neutral_formula", "adduct", "tier", "ion_score", "n_files")
                 if c in merged.columns]
    led = merged[base_cols].copy()
    # (a) today's stamp: the merge anchor, the merge tolerance
    ann_a = TS.annotate_peaks(ts, led[["mz", "neutral_formula", "adduct", "tier"]], tol_ppm=tol)
    cov_a = coverage_by_ion(ann_a, n_spectra)
    # (b) the trace stamp
    idx = TR.PeakIndex(ts, tol_ppm=tol)
    led_t = led.copy()
    info = TS.recentre_ledger(led_t, index=idx, tol_ppm=tol, log=print)
    info.update(TS.collapse_trace_labels(led_t, tol_ppm=tol, log=print))
    stamp_tol, sigma = TS.stamp_tolerance(idx, led_t["mz_trace"], tol_ppm=tol)
    if args.stamp_tol is not None:
        stamp_tol = float(args.stamp_tol)
    print(f"  per-ion scatter {sigma} ppm -> stamping window +-{stamp_tol:g} ppm"
          + (" (overridden)" if args.stamp_tol is not None else ""))
    sf = TS.stamping_frame(led_t, None)
    ann_b = TS.annotate_peaks(ts, sf[["mz", "neutral_formula", "adduct", "tier"]], tol_ppm=stamp_tol)
    cov_b = coverage_by_ion(ann_b, n_spectra)

    both = pd.concat([cov_a.rename("cov_anchor"), cov_b.rename("cov_trace")], axis=1).fillna(0.0)
    print(f"\n  all {len(both)} stamped ions: mean coverage {both.cov_anchor.mean():.3f} -> "
          f"{both.cov_trace.mean():.3f}; median {both.cov_anchor.median():.3f} -> {both.cov_trace.median():.3f}")
    gain = both.cov_trace - both.cov_anchor
    print(f"  ions gaining > 5 pp: {(gain > 0.05).mean():.1%}; losing > 5 pp: {(gain < -0.05).mean():.1%}")
    print(f"  ledger rows re-centred {info['n_recentred']} / {info['n_rows']} (median move "
          f"{info['median_abs_move_ppm']} ppm, {info['n_guarded']} guarded); "
          f"{info['n_collapsed']} labels collapsed onto {info['n_traces']} traces")
    stamped_a = int(ann_a["neutral_formula"].notna().sum()); stamped_b = int(ann_b["neutral_formula"].notna().sum())
    print(f"  peaks stamped with a formula: {stamped_a} -> {stamped_b} ({stamped_a / len(ts):.1%} -> {stamped_b / len(ts):.1%})")

    if args.targets:
        tg = _read_targets(os.path.expanduser(args.targets))
        rows = []
        for _, r in tg.iterrows():
            f = str(r["formula"]).strip()
            adds = [a.strip() for a in str(r.get("adducts", "") or "").split(";") if a.strip()] or list(DEFAULT_ADDUCTS)
            best_a = best_b = 0.0; where = ""
            probes = [(f, a) for a in adds]
            if "[M+NO3]-" in adds and not args.no_nitrate_alias:
                # [M+NO3]- of M is the exact isobar of [M-H]- of M+HNO3 (5.7e-14 Da):
                # a ledger with no nitrate channel carries the ion under that label
                cnt = dict(C.parse_formula(f))
                cnt["H"] = cnt.get("H", 0) + 1; cnt["N"] = cnt.get("N", 0) + 1; cnt["O"] = cnt.get("O", 0) + 3
                alias = C.format_formula(cnt) if hasattr(C, "format_formula") else \
                    "".join(f"{el}{n if n > 1 else ''}" for el, n in cnt.items() if n)
                probes.append((alias, "[M-H]-"))
            for f2, a in probes:
                try:
                    C.ion_mz(f2, a)
                except Exception:  # noqa: BLE001
                    continue
                ca = float(cov_a.get((f2, a), 0.0)); cb = float(cov_b.get((f2, a), 0.0))
                if cb > best_b or (cb == best_b and ca > best_a):
                    best_a, best_b, where = ca, cb, (a if f2 == f else f"{a} via {f2} [M-H]-")
            rows.append(dict(formula=f, adduct=where, cov_anchor=round(best_a, 3), cov_trace=round(best_b, 3)))
        T = pd.DataFrame(rows)
        print("\n  targets (ions known to be present):")
        print(T.to_string(index=False))
        print(f"\n  target mean coverage {T.cov_anchor.mean():.3f} -> {T.cov_trace.mean():.3f}; "
              f"targets stamped in >= 50 % of spectra {int((T.cov_anchor >= 0.5).sum())} -> "
              f"{int((T.cov_trace >= 0.5).sum())} of {len(T)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
