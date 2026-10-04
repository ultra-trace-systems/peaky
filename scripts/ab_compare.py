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
* row changes — the merged rows joined on the ion `(neutral_formula, adduct)`:
  how many of the shared ions changed tier, evidence level or claim, and the
  rows only one run holds (a pre-claim run's claims are read off its levels; a
  field one run does not carry at all is not compared);
* time-series coverage — per `(neutral, adduct)`, the share of the batch's
  samples carrying the ion in `_batch_ts.parquet`, and every gain or loss above
  `--coverage-delta` points;
* recovery — the neutrals A found in at least `--min-files` files that B lacks,
  split by occurrence and median m/z, the headline number for a path change;
* evidence levels — the `evidence_level` histogram, when the column is there;
* claims — the claim histogram (identified / neutral / ion / tentative, then the
  reagent and not-assessed buckets), when either column is there.

The level token `NA` (not assessed on this instrument class) is literal: the
merged ledger's `evidence_level` is re-read as written, so it is never mistaken
for "no level". A run levelled on a scale before the evidence scale (it carries
`evidence_axes` / `level_reason`, or a letter this scale does not define) has
every letter read as no level (an old 4a is not a 4a of this scale) and its
claims re-read on this scale (tentative); its levels are not compared letter
for letter with a run on the current scale.
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

# The evidence scale's vocabulary (peaky.assignment.evidence: LEVEL_ORDER,
# BUCKETS, CLAIMS, the buckets' claims and the level sets), kept here so the
# script reads a run dir without peaky installed; a test pins them equal to the
# package's.
LEVEL_ORDER = ["1", "2", "3c", "4a", "4b", "5a", "5b"]
BUCKETS = ["reagent", "NA"]
CLAIMS = ("identified", "neutral", "ion", "tentative")
CLAIM_REAGENT = "reagent"
CLAIM_NA = "not assessed"
CLAIM_KEYS = CLAIMS + (CLAIM_REAGENT, CLAIM_NA)
CLAIM_IDENTIFIED = frozenset({"1", "2", "3c"})
CLAIM_NEUTRAL = frozenset({"4a"})
CLAIM_ION = frozenset({"4b"})
#: every letter the scale defines; any other is a level of an older scale
KNOWN_LEVELS = frozenset(LEVEL_ORDER) | frozenset(BUCKETS)
#: columns only a ledger levelled before the scale carries
OLD_LEVEL_COLUMNS = ("evidence_axes", "level_reason", "n_plausible_structures")
try:        # the release names the older scale; without peaky installed it is just "older"
    from peaky.assignment.levels.scale import SCALE_RELEASE as _RELEASE
    OLD_SCALE_NO_LEVEL = f"no level (pre-{_RELEASE} scale)"
except ImportError:  # pragma: no cover - peaky not importable
    OLD_SCALE_NO_LEVEL = "no level (older scale)"
# What a shared ion is compared on: (report field, ledger column).
CHANGE_FIELDS = (("tier", "tier"), ("level", "evidence_level"), ("claim", "claim"))
CHANGED_ROWS_SHOWN = 20


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
    def by_stage(self) -> dict:
        """Merged rows per selection stage — `cover`, `residual`.

        Two runs whose stage composition differs are not comparable on totals:
        a run with the residual stage on carries rows a run without it never
        looked for. Prefer the summary's own count; fall back to the ledger.
        """
        recorded = self.summary.get("merged_by_stage")
        if isinstance(recorded, dict) and recorded:
            return {str(k): int(v) for k, v in recorded.items()}
        if "stage" in self.ledger.columns:
            return {
                str(k): int(v)
                for k, v in self.ledger["stage"].fillna("—").value_counts().items()
            }
        return {}

    @property
    def rolling(self) -> dict:
        """The `--rolling-centre` block, `{}` when the run did not roll.

        It is written in two places and two shapes: `traces.rolling` on the
        cover path (a dict with `enabled` / `n_rolling` / `n_global`) and a flat
        `n_rolling` under `trace_first`. Reading only one of them reports a run
        that plainly rolled as a fixed-centre run.
        """
        traces = self.summary.get("traces") or {}
        block = traces.get("rolling")
        if isinstance(block, dict) and (block.get("enabled") or block.get("n_rolling")):
            return block
        tf = self.summary.get("trace_first") or {}
        if tf.get("n_rolling"):
            return {"enabled": True, "n_rolling": tf["n_rolling"]}
        return {}


def _literal_na(ledger: pd.DataFrame, path: str) -> pd.DataFrame:
    """Put back the literal level token `NA` the default CSV parser read as NaN
    (the column re-read as written)."""
    if "evidence_level" not in ledger.columns:
        return ledger
    literal = pd.read_csv(path, usecols=["evidence_level"], dtype=str,
                          keep_default_na=False)["evidence_level"]
    level = ledger["evidence_level"].astype(object).where(literal.ne("NA").values, "NA")
    return ledger.assign(evidence_level=level)


def load_run(path: str) -> Run:
    run_dir = resolve_run_dir(path)
    merged_path = os.path.join(run_dir, "merged_ledger.csv")
    ledger = _literal_na(pd.read_csv(merged_path, low_memory=False), merged_path)

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


def _ledger(run) -> pd.DataFrame:
    return run.ledger if isinstance(run, Run) else run


def _unknown_levels(led: pd.DataFrame) -> pd.Series:
    """The rows whose level letter this scale does not define."""
    level = led["evidence_level"].astype(object)
    return level.notna() & ~level.astype(str).str.strip().isin(KNOWN_LEVELS)


def before_scale(run) -> bool:
    """Whether the run was levelled on a scale before the evidence scale: it
    carries a pre-scale level column or a letter this scale does not define."""
    led = _ledger(run)
    if "evidence_level" not in led.columns:
        return False
    return any(c in led.columns for c in OLD_LEVEL_COLUMNS) or bool(_unknown_levels(led).any())


def _lettered(led: pd.DataFrame) -> pd.Series:
    """The rows that carry a level letter."""
    level = led["evidence_level"].astype(object)
    return level.notna() & level.astype(str).str.strip().ne("")


def levels_of(run) -> pd.Series | None:
    """The merged rows' levels on this scale: on a run levelled before it EVERY
    letter reads as no level (OLD_SCALE_NO_LEVEL) -- the letters both scales
    share too, since an old 4a is not a 4a of this scale. None when the ledger
    carries no `evidence_level`."""
    led = _ledger(run)
    if "evidence_level" not in led.columns:
        return None
    level = led["evidence_level"].astype(object)
    return level.where(~_lettered(led), None) if before_scale(led) else level


def evidence_hist(run: Run) -> pd.Series | None:
    if "evidence_level" not in run.ledger.columns:
        return None
    level = run.ledger["evidence_level"].astype(object)
    if before_scale(run.ledger):            # an old letter is no level of this scale, whatever its name
        level = level.where(~_lettered(run.ledger), OLD_SCALE_NO_LEVEL)
    counts = level.fillna("—").astype(str).value_counts()
    order = {k: i for i, k in enumerate([*LEVEL_ORDER, *BUCKETS])}
    return counts.loc[sorted(counts.index, key=lambda k: (order.get(k, len(order)), k))]


def claim_class(level) -> str:
    """The claim a level supports, read as evidence.claim_class reads it:
    'identified' (3c; 1 and 2 never assigned), 'neutral' (4a), 'ion' (4b),
    'tentative' (5a, 5b or no level), and the buckets 'reagent' / 'not assessed'
    (NA)."""
    if level is None or (not isinstance(level, str) and pd.isna(level)):
        return "tentative"
    lv = str(level).strip()
    if lv in CLAIM_IDENTIFIED:
        return "identified"
    if lv in CLAIM_NEUTRAL:
        return "neutral"
    if lv in CLAIM_ION:
        return "ion"
    if lv == "reagent":
        return CLAIM_REAGENT
    if lv == "NA":
        return CLAIM_NA
    return "tentative"


def claim_source(run) -> str | None:
    """'stamped' (the ledger carries `claim` on this scale), 'derived' (read
    off `evidence_level`: the run predates the column, or was levelled before
    the scale so its stored claim is re-read) or None (neither)."""
    led = _ledger(run)
    cols = led.columns
    if "claim" in cols and not before_scale(led):
        return "stamped"
    return "derived" if "evidence_level" in cols else None


def claims_of(run) -> pd.Series | None:
    """The claim of every merged row: the `claim` column when the ledger
    carries one on this scale, else read off its levels on this scale; None
    when it has neither."""
    led = _ledger(run)
    source = claim_source(led)
    if source == "stamped":
        return led["claim"].astype(object)
    if source == "derived":
        return levels_of(led).map(claim_class).astype(object)
    return None


def claims_hist(run) -> pd.Series | None:
    claims = claims_of(run)
    if claims is None:
        return None
    return claims.fillna("—").astype(str).value_counts()


def _keyed(led: pd.DataFrame) -> pd.DataFrame:
    """The merged rows that read an ion, keyed on (neutral, adduct, k) with the
    compared fields as strings (no value reads '—'). k numbers the repeats of
    one ion in m/z order, so a repeated ion pairs row by row, never n x m."""
    claims = claims_of(led)
    sub = led[led["neutral_formula"].notna()]
    out = pd.DataFrame(
        {
            "neutral": sub["neutral_formula"].astype(str),
            "adduct": (sub["adduct"].fillna("").astype(str)
                       if "adduct" in sub.columns else ""),
            "mz": sub["mz"] if "mz" in sub.columns else np.nan,
        },
        index=sub.index,
    )
    for field, col in CHANGE_FIELDS:
        values = claims if col == "claim" else (led[col] if col in led.columns else None)
        out[field] = (values.loc[sub.index].fillna("—").astype(str)
                      if values is not None else "—")
    out = out.sort_values("mz", kind="stable")
    out["k"] = out.groupby(["neutral", "adduct"]).cumcount()
    return out


def _carries(led: pd.DataFrame, col: str) -> bool:
    """Whether a ledger holds a compared field at all: the claim when it is
    stamped or can be read off a level, any other field by its column."""
    return claim_source(led) is not None if col == "claim" else col in led.columns


def row_changes(a, b) -> dict:
    """The merged rows of two runs (Run or ledger) joined on the ion
    (neutral_formula, adduct): {shared, only_a, only_b, tier, level, claim,
    rows}. A field counts as changed on a shared ion when its values differ as
    strings, no value on both sides being equal; `rows` lists those ions. A
    field one ledger does not carry at all is not compared: it reads None, and
    no row is listed for it (a missing column is not a change)."""
    la, lb = _ledger(a), _ledger(b)
    ka, kb = _keyed(la), _keyed(lb)
    joined = ka.merge(kb, on=["neutral", "adduct", "k"], how="outer",
                      suffixes=("_a", "_b"), indicator=True)
    shared = joined[joined["_merge"] == "both"]
    # a pre-scale run's letters are not this scale's: its level is compared only
    # with another pre-scale run (its claims, re-read on this scale, always are)
    same_scale = before_scale(la) == before_scale(lb)
    changed = {field: shared[f"{field}_a"] != shared[f"{field}_b"]
               for field, col in CHANGE_FIELDS
               if _carries(la, col) and _carries(lb, col) and (field != "level" or same_scale)}
    rows = (shared[pd.concat(changed.values(), axis=1).any(axis=1)]
            if len(shared) and changed else shared.iloc[:0])
    return {
        "shared": int(len(shared)),
        "only_a": int((joined["_merge"] == "left_only").sum()),
        "only_b": int((joined["_merge"] == "right_only").sum()),
        **{field: int(changed[field].sum()) if field in changed else None
           for field, _col in CHANGE_FIELDS},
        "rows": rows.sort_values("mz_a", kind="stable").reset_index(drop=True),
    }


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
        on = []
        if run.trace_first:
            on.append("trace-first")
        roll = run.rolling
        if roll:
            n, g = roll.get("n_rolling"), roll.get("n_global")
            on.append(
                f"rolling ({n} of {n + g} ions)" if n is not None and g is not None
                else "rolling"
            )
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

    sa, sb = run_a.by_stage, run_b.by_stage
    if sa or sb:
        stages = sorted(set(sa) | set(sb))
        w("Merged rows by selection stage:\n")
        w("| stage | A | B | delta |")
        w("|---|---:|---:|---:|")
        for stage in stages:
            # A run that records no stages at all has not run zero rows through
            # them -- it predates the breakdown. Saying 0 invents a finding.
            va = str(sa.get(stage, 0)) if sa else "n/a"
            vb = str(sb.get(stage, 0)) if sb else "n/a"
            delta = (
                f"{sb.get(stage, 0) - sa.get(stage, 0):+d}" if sa and sb else "—"
            )
            w(f"| {stage} | {va} | {vb} | {delta} |")
        w("")
        if not sa or not sb:
            which = "A" if not sa else "B"
            w(
                f"> **Run {which} records no stage breakdown** — it predates the "
                "field, so its rows cannot be attributed to a stage and the two "
                "runs' composition cannot be compared here.\n"
            )
        else:
            missing = [s_ for s_ in stages if bool(sa.get(s_)) != bool(sb.get(s_))]
            if missing:
                w(
                    f"> **The runs do not share a stage composition** "
                    f"({', '.join(missing)} ran in one arm only). The totals above "
                    "are not a like-for-like comparison: rerun with the same "
                    "stages, or read the shared stage alone.\n"
                )

    w(f"## Ion disagreements (matched within {tol_ppm:g} ppm)\n")
    matched = match_by_mz(a, b, tol_ppm)
    n_ion = None
    if matched.empty:
        w("No merged rows matched across the runs.\n")
    else:
        disagree = matched[matched["ion_a"] != matched["ion_b"]]
        n_ion = len(disagree)
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

    w("## Row changes (merged rows joined on the ion)\n")
    ch = row_changes(a, b)
    w(
        f"{ch['shared']} ions are merged rows in both runs, {ch['only_a']} only in A "
        f"and {ch['only_b']} only in B. The ion row counts the m/z-matched pairs "
        "above; tier, level and claim count the shared ions, no value on both "
        "sides being no change.\n"
    )
    for label, run in (("A", run_a), ("B", run_b)):
        if before_scale(run):
            n_old = int(_lettered(run.ledger).sum())
            w(f"> Run {label} was levelled on a scale before the evidence scale: "
              + (f"its {n_old} levelled row(s) read as {OLD_SCALE_NO_LEVEL} (the letters this scale shares "
                 "too), so its claims read tentative" if n_old else "it carries no level of this scale")
              + ("" if before_scale(run_a) == before_scale(run_b)
                 else ", and its levels are not compared with the other run's")
              + ".\n")
        elif claim_source(run) == "derived":
            w(f"> Run {label}'s ledger has no `claim` column: its claims are read "
              "off `evidence_level`.\n")
        elif claim_source(run) is None:
            w(f"> Run {label}'s ledger carries neither `claim` nor `evidence_level`: "
              "level and claim are not compared.\n")
    w("| field | changed |")
    w("|---|---:|")
    w(f"| ion | {n_ion if n_ion is not None else '—'} |")
    for field, _col in CHANGE_FIELDS:
        w(f"| {field} | {ch[field] if ch[field] is not None else '—'} |")
    w(f"| rows only in A | {ch['only_a']} |")
    w(f"| rows only in B | {ch['only_b']} |")
    w("")
    if not ch["rows"].empty:
        shown = ch["rows"].head(CHANGED_ROWS_SHOWN)
        w(f"The {len(shown)} changed at the lowest m/z:\n")
        w("| m/z (A) | ion | tier A → B | level A → B | claim A → B |")
        w("|---:|---|---|---|---|")
        for _, r in shown.iterrows():
            w(
                f"| {r.mz_a:.4f} | {r.neutral} {r.adduct} | {r.tier_a} → {r.tier_b} | "
                f"{r.level_a} → {r.level_b} | {r.claim_a} → {r.claim_b} |"
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

    w("## Claims\n")
    ca, cb = claims_hist(run_a), claims_hist(run_b)
    if ca is None and cb is None:
        w("Neither ledger carries `claim` or `evidence_level`.\n")
    else:
        seen = set(ca.index if ca is not None else ()) | set(cb.index if cb is not None else ())
        buckets = [c for c in (CLAIM_REAGENT, CLAIM_NA) if c in seen]
        extra = sorted(seen - set(CLAIM_KEYS))
        w("| claim | A | B | delta |")
        w("|---|---:|---:|---:|")
        for claim in (*CLAIMS, *buckets, *extra):
            # A run with neither column has no claims to count, not zero of
            # each -- as with the stages, saying 0 invents a finding.
            va = str(int(ca.get(claim, 0))) if ca is not None else "n/a"
            vb = str(int(cb.get(claim, 0))) if cb is not None else "n/a"
            delta = (f"{int(cb.get(claim, 0)) - int(ca.get(claim, 0)):+d}"
                     if ca is not None and cb is not None else "—")
            w(f"| {claim} | {va} | {vb} | {delta} |")
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
