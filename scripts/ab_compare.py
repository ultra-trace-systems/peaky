"""Compare two `peaky batch` runs over the same batch — path against path.

`diff_runs.py` answers "same peaks, same formula?" and keys on
`(sample_item_id, peak_id)`. This script answers a different question: two runs
took DIFFERENT routes to the same batch (a cover of files vs `--trace-first`,
one batch centre vs `--rolling-centre`), so their peak sets do not line up at
all. Everything here is therefore keyed on chemistry — the merged row's m/z, its
`(neutral_formula, adduct)` ion, or the neutral alone.

    python scripts/ab_compare.py <run_dir_A> <run_dir_B> [--out report.md]

A run dir is either the timestamped directory that holds `merged_ledger.csv` or
its parent (the `--out-dir` given to `peaky batch`, which holds exactly one).

Reported, A as the reference and B as the challenger:

* headline — merged rows, tier counts, distinct neutrals, distinct ions;
* ion disagreements — merged rows matched across the runs within `--tol-ppm`
  whose `(neutral_formula, adduct)` reading differs;
* time-series coverage — per `(neutral, adduct)`, the share of the batch's
  samples carrying the ion in `_batch_ts.parquet`, and every gain or loss above
  `--coverage-delta` points;
* recovery — the neutrals A found in at least `--min-files` files that B lacks,
  split by occurrence and median m/z, the headline number for a path change;
* evidence levels — the `evidence_level` histogram, when the column is there.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

TS_COLUMNS = ["sample_item_id", "neutral_formula", "adduct"]


def resolve_run_dir(path: str) -> str:
    """Accept the run dir itself or the --out-dir that holds exactly one."""
    path = os.path.expanduser(path.rstrip("/"))
    if os.path.isfile(os.path.join(path, "merged_ledger.csv")):
        return path
    inner = sorted(
        d
        for d in glob.glob(os.path.join(path, "*/"))
        if os.path.isfile(os.path.join(d, "merged_ledger.csv"))
    )
    if len(inner) == 1:
        return inner[0].rstrip("/")
    if not inner:
        raise SystemExit(f"no merged_ledger.csv in {path} or any child")
    raise SystemExit(
        f"{path} holds {len(inner)} run dirs; name the one you mean:\n  "
        + "\n  ".join(inner)
    )


@dataclass
class Run:
    """One run dir, loaded lazily enough to survive a missing time series."""

    path: str
    ledger: pd.DataFrame
    summary: dict
    ts: pd.DataFrame | None

    @property
    def name(self) -> str:
        return os.path.basename(self.path)

    @property
    def trace_first(self) -> bool:
        return bool(self.summary.get("trace_first"))

    @property
    def rolling(self) -> bool:
        tf = self.summary.get("trace_first") or {}
        traces = self.summary.get("traces") or {}
        return bool(tf.get("n_rolling") or traces.get("n_rolling"))


def load_run(path: str) -> Run:
    run_dir = resolve_run_dir(path)
    ledger = pd.read_csv(os.path.join(run_dir, "merged_ledger.csv"), low_memory=False)

    summary_path = os.path.join(run_dir, "batch_summary.json")
    summary = {}
    if os.path.isfile(summary_path):
        with open(summary_path) as fh:
            summary = json.load(fh)

    ts = None
    ts_path = os.path.join(run_dir, "per_file", "_batch_ts.parquet")
    if os.path.isfile(ts_path):
        try:
            ts = pd.read_parquet(ts_path, columns=TS_COLUMNS)
        except (ValueError, KeyError):  # an older run without the columns
            ts = None

    return Run(path=run_dir, ledger=ledger, summary=summary, ts=ts)


def _neutrals(df: pd.DataFrame) -> set[str]:
    return set(df["neutral_formula"].dropna().astype(str)) - {""}


def _ions(df: pd.DataFrame) -> set[tuple[str, str]]:
    sub = df.dropna(subset=["neutral_formula"])
    return set(
        zip(sub["neutral_formula"].astype(str), sub["adduct"].fillna("").astype(str))
    )


def match_by_mz(a: pd.DataFrame, b: pd.DataFrame, tol_ppm: float) -> pd.DataFrame:
    """Nearest-in-m/z pairing of merged rows, within tol_ppm. One B row per A row."""
    if a.empty or b.empty:
        return pd.DataFrame(columns=["mz_a", "mz_b", "ion_a", "ion_b", "ppm"])

    bs = b.sort_values("mz").reset_index(drop=True)
    bmz = bs["mz"].to_numpy(dtype=float)
    amz = a["mz"].to_numpy(dtype=float)

    idx = np.searchsorted(bmz, amz)
    left = np.clip(idx - 1, 0, len(bmz) - 1)
    right = np.clip(idx, 0, len(bmz) - 1)
    take_left = np.abs(bmz[left] - amz) <= np.abs(bmz[right] - amz)
    nearest = np.where(take_left, left, right)

    ppm = np.abs(bmz[nearest] - amz) / amz * 1e6
    keep = ppm <= tol_ppm

    a_keep = a.loc[keep].reset_index(drop=True)
    b_keep = bs.iloc[nearest[keep]].reset_index(drop=True)

    def ion(df: pd.DataFrame) -> pd.Series:
        return (
            df["neutral_formula"].fillna("—").astype(str)
            + " "
            + df["adduct"].fillna("").astype(str)
        ).str.strip()

    return pd.DataFrame(
        {
            "mz_a": a_keep["mz"].to_numpy(),
            "mz_b": b_keep["mz"].to_numpy(),
            "ion_a": ion(a_keep),
            "ion_b": ion(b_keep),
            "tier_a": a_keep["tier"].astype(str),
            "tier_b": b_keep["tier"].astype(str),
            "ppm": ppm[keep],
        }
    )


def ts_coverage(run: Run) -> pd.Series:
    """Share of the run's samples that carry each (neutral, adduct) ion."""
    if run.ts is None or run.ts.empty:
        return pd.Series(dtype=float)
    ts = run.ts.dropna(subset=["neutral_formula"])
    n_samples = run.ts["sample_item_id"].nunique()
    if not n_samples:
        return pd.Series(dtype=float)
    cov = ts.groupby(
        [ts["neutral_formula"].astype(str), ts["adduct"].fillna("").astype(str)]
    )["sample_item_id"].nunique()
    return (cov / n_samples * 100.0).sort_values(ascending=False)


def recovery(
    a: pd.DataFrame, b: pd.DataFrame, min_files: int
) -> tuple[pd.DataFrame, int, int]:
    """A's multi-file neutrals, and which of them B kept."""
    col = "n_files" if "n_files" in a.columns else "n_files_ion"
    multi = a[(a[col].fillna(0) >= min_files) & a["neutral_formula"].notna()]
    kept_names = _neutrals(b)

    per_neutral = (
        multi.groupby(multi["neutral_formula"].astype(str))
        .agg(occurrence=("occurrence", "max"), median_mz=("mz", "median"))
        .reset_index()
        .rename(columns={"neutral_formula": "neutral"})
    )
    per_neutral["kept"] = per_neutral["neutral"].isin(kept_names)
    return per_neutral, int(per_neutral["kept"].sum()), len(per_neutral)


def _bucket_table(
    lost: pd.DataFrame, column: str, edges: list[float], labels: list[str]
) -> pd.DataFrame:
    """Count the lost neutrals per bucket, labelled for a human not a repr."""
    if lost.empty:
        return pd.DataFrame(columns=[column, "lost"])
    cut = pd.cut(lost[column], bins=edges, labels=labels, include_lowest=True)
    out = cut.value_counts().sort_index().reset_index()
    out.columns = [column, "lost"]
    out[column] = out[column].astype(str)
    return out[out["lost"] > 0]


def evidence_hist(run: Run) -> pd.Series | None:
    if "evidence_level" not in run.ledger.columns:
        return None
    return run.ledger["evidence_level"].fillna("—").astype(str).value_counts().sort_index()


def _tier_table(a: pd.DataFrame, b: pd.DataFrame) -> str:
    ta = a["tier"].astype(str).value_counts()
    tb = b["tier"].astype(str).value_counts()
    lines = ["| tier | A | B | delta |", "|---|---:|---:|---:|"]
    for tier in sorted(set(ta.index) | set(tb.index)):
        na, nb = int(ta.get(tier, 0)), int(tb.get(tier, 0))
        lines.append(f"| {tier} | {na} | {nb} | {nb - na:+d} |")
    return "\n".join(lines)


def build_report(
    run_a: Run,
    run_b: Run,
    tol_ppm: float,
    min_files: int,
    coverage_delta: float,
    top: int,
) -> str:
    a, b = run_a.ledger, run_b.ledger
    out: list[str] = []
    w = out.append

    w("# A/B — two paths over one batch\n")
    w(f"* **A** `{run_a.name}`")
    w(f"* **B** `{run_b.name}`\n")

    def flags(run: Run) -> str:
        on = [n for n, v in (("trace-first", run.trace_first), ("rolling", run.rolling)) if v]
        return ", ".join(on) if on else "cover path, fixed centre"

    w(f"A ran {flags(run_a)}; B ran {flags(run_b)}.\n")

    w("## Headline\n")
    w(_tier_table(a, b))
    na, nb = _neutrals(a), _neutrals(b)
    ia, ib = _ions(a), _ions(b)
    w("")
    w("| quantity | A | B | delta |")
    w("|---|---:|---:|---:|")
    w(f"| merged rows | {len(a)} | {len(b)} | {len(b) - len(a):+d} |")
    w(f"| distinct neutrals | {len(na)} | {len(nb)} | {len(nb) - len(na):+d} |")
    w(f"| distinct ions | {len(ia)} | {len(ib)} | {len(ib) - len(ia):+d} |")
    w(f"| neutrals in both | {len(na & nb)} | | |")
    w(f"| neutrals only in A | {len(na - nb)} | | |")
    w(f"| neutrals only in B | | {len(nb - na)} | |")
    w("")

    w(f"## Ion disagreements (matched within {tol_ppm:g} ppm)\n")
    matched = match_by_mz(a, b, tol_ppm)
    if matched.empty:
        w("No merged rows matched across the runs.\n")
    else:
        disagree = matched[matched["ion_a"] != matched["ion_b"]]
        share = len(disagree) / len(matched) * 100.0
        w(
            f"{len(matched)} of A's {len(a)} rows found a partner in B; "
            f"**{len(disagree)} disagree on the ion** ({share:.1f} %).\n"
        )
        if not disagree.empty:
            w(f"The {min(top, len(disagree))} at the lowest m/z:\n")
            w("| m/z (A) | A reads | B reads | ppm apart |")
            w("|---:|---|---|---:|")
            for _, r in disagree.nsmallest(top, "mz_a").iterrows():
                w(
                    f"| {r.mz_a:.4f} | {r.ion_a} ({r.tier_a}) | "
                    f"{r.ion_b} ({r.tier_b}) | {r.ppm:.2f} |"
                )
            w("")

    w("## Time-series coverage per ion\n")
    cov_a, cov_b = ts_coverage(run_a), ts_coverage(run_b)
    if cov_a.empty and cov_b.empty:
        w("Neither run wrote `per_file/_batch_ts.parquet`.\n")
    else:
        joined = pd.concat([cov_a.rename("A"), cov_b.rename("B")], axis=1).fillna(0.0)
        joined["delta"] = joined["B"] - joined["A"]
        gains = joined[joined["delta"] > coverage_delta].sort_values(
            "delta", ascending=False
        )
        losses = joined[joined["delta"] < -coverage_delta].sort_values("delta")
        w(
            f"{len(joined)} ions carry a series in either run. "
            f"Above {coverage_delta:g} points: **{len(gains)} gained**, "
            f"**{len(losses)} lost**.\n"
        )
        for title, frame in (("Gains", gains), ("Losses", losses)):
            if frame.empty:
                continue
            w(f"### {title} (top {min(top, len(frame))})\n")
            w("| ion | A % | B % | delta |")
            w("|---|---:|---:|---:|")
            for (neutral, adduct), r in frame.head(top).iterrows():
                w(
                    f"| {neutral} {adduct} | {r.A:.1f} | {r.B:.1f} | "
                    f"{r.delta:+.1f} |"
                )
            w("")

    w(f"## Recovery of A's ≥{min_files}-file neutrals\n")
    per_neutral, kept, total = recovery(a, b, min_files)
    if not total:
        w(f"A has no neutral in {min_files} or more files.\n")
    else:
        w(
            f"A found **{total}** neutrals in {min_files} or more files. "
            f"B keeps **{kept}** of them — **{kept / total * 100:.1f} % recovery** "
            f"({total - kept} lost).\n"
        )
        lost = per_neutral[~per_neutral["kept"]]
        if not lost.empty:
            occ = _bucket_table(
                lost,
                "occurrence",
                [0.0, 0.1, 0.25, 0.5, 0.75, 1.0],
                ["≤10 %", "10–25 %", "25–50 %", "50–75 %", "75–100 %"],
            )
            if not occ.empty:
                w("By occurrence in A:\n")
                w("| occurrence | lost |")
                w("|---|---:|")
                for _, r in occ.iterrows():
                    w(f"| {r.occurrence} | {r.lost} |")
                w("")
            mz = _bucket_table(
                lost,
                "median_mz",
                [0, 100, 200, 300, 400, 600, np.inf],
                ["<100", "100–200", "200–300", "300–400", "400–600", ">600"],
            )
            if not mz.empty:
                w("By median m/z:\n")
                w("| m/z | lost |")
                w("|---|---:|")
                for _, r in mz.iterrows():
                    w(f"| {r.median_mz} | {r.lost} |")
                w("")

    w("## Evidence levels\n")
    ha, hb = evidence_hist(run_a), evidence_hist(run_b)
    if ha is None and hb is None:
        w("Neither ledger carries `evidence_level`.\n")
    else:
        ha = ha if ha is not None else pd.Series(dtype=int)
        hb = hb if hb is not None else pd.Series(dtype=int)
        w("| level | A | B | delta |")
        w("|---|---:|---:|---:|")
        for level in sorted(set(ha.index) | set(hb.index)):
            va, vb = int(ha.get(level, 0)), int(hb.get(level, 0))
            w(f"| {level} | {va} | {vb} | {vb - va:+d} |")
        w("")

    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Compare two peaky batch runs over the same batch."
    )
    p.add_argument("run_a", help="reference run dir (or its --out-dir parent)")
    p.add_argument("run_b", help="challenger run dir (or its --out-dir parent)")
    p.add_argument("--out", help="write the markdown report here (else stdout)")
    p.add_argument(
        "--tol-ppm",
        type=float,
        default=6.0,
        help="m/z window for pairing merged rows across the runs (default 6)",
    )
    p.add_argument(
        "--min-files",
        type=int,
        default=2,
        help="a neutral counts as A's when it appears in this many files (default 2)",
    )
    p.add_argument(
        "--coverage-delta",
        type=float,
        default=5.0,
        help="report coverage moves above this many points (default 5)",
    )
    p.add_argument(
        "--top", type=int, default=25, help="rows per listing (default 25)"
    )
    args = p.parse_args(argv)

    report = build_report(
        load_run(args.run_a),
        load_run(args.run_b),
        tol_ppm=args.tol_ppm,
        min_files=args.min_files,
        coverage_delta=args.coverage_delta,
        top=args.top,
    )

    if args.out:
        with open(args.out, "w") as fh:
            fh.write(report)
        print(f"wrote {args.out}")
    else:
        sys.stdout.write(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
