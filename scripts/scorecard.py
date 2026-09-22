"""The scorecard — what one `peaky batch` run assigned, how good it is, what it missed.

A run answers three questions, and every session that ran the pipeline answers
them the same way so the sessions can be compared:

    what was assigned      spectra, peaks, the stamped share of peaks and of
                           signal; merged rows by tier, stage and channel; the
                           file vote; STAMP COVERAGE (merged rows the time
                           series never carries — a stamping defect, not a
                           chemistry question)
    how good is it         the brightest 50 ions with their reading, tier,
                           evidence level and axes — every bright ion is either
                           explained or a named miss; the best-evidence 50; the
                           level and axes histograms; the roster recall; the
                           element census; the DECOY false-discovery bound (the
                           engine run offline on m/z-shifted input, and on the
                           wrong adduct set); falsification survival (the 13C
                           carbon count, the heteroatom lines, time covariance)
    what was missed        M1 the brightest UNSTAMPED tracks, grouped into
                           families by their shift from the nearest parent so a
                           300-line electron-attachment comb is one row; M2 the
                           roster / reference / known species present but not
                           assigned, with the engine's own reason; M3 what the
                           other instrument on the same air, or the other path
                           on the same batch, found and this run did not

    python scripts/scorecard.py <run_dir>... [--levels levels.csv]
        [--other <run_dir>] [--other-instrument <run_dir>]
        [--decoy none|shift|adducts|both] [--out DIR]

Per run: `<out>/<run name>/SCORECARD.md` and `scorecard.json`, one row appended
to `<out>/scoreboard.jsonl`, and `<out>/SCOREBOARD.md` + `scoreboard.html`
regenerated over every channel's latest row with its delta. The level column
comes from `scripts/level_ledger.py` (run in-process) until the in-core
`evidence_level` column exists, which is then preferred.

Never read `tables/residual_bins.csv` for the misses: it lists only the bins
absent from every cover file, and every headline miss of the first cut sat
inside the cover files. M1 is built from the per-file ledgers' `unexplained`
rows and the time series.
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import html
import json
import os
import re
import sys
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
if REPO not in sys.path:
    sys.path.insert(0, REPO)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import level_ledger as LL  # noqa: E402  (scripts/level_ledger.py)
from peaky.chem import chemistry as C  # noqa: E402
from peaky.paths import pkg_data  # noqa: E402

__version__ = "0.1.0"

DEFAULT_OUT = os.path.expanduser("~/peaky-output/scoreboard")
LEVELS = [k for k in LL.LEVEL_ORDER if k not in ("1", "2a")]  # 2b .. 5b
GOOD_LEVELS = {"2b", "3a", "3b", "4a"}  # "best evidence" = level <= 4a
BARE = {"[M-H]-", "[M+H]+"}
TOP_N = 50
PRESENCE = 0.5          # M1: a track present in >= this share of spectra
ROSTER_PRESENCE = 0.2   # M2: a roster ion counts as present at this share
FAMILY_MIN = 3          # M1: tracks sharing a shift before it is a family
SHOULDER_MDA = 20.0     # M1: |delta| to a brighter stamped ion that reads as a shoulder
SHIFT_TOL_MDA = 1.5     # M1: how close a delta must sit to a named shift
CENSUS = ("F", "Si", "P", "Cl", "Br", "S", "N")
C_HEAVY = 20            # census: neutrals with more carbons than this

# named shifts from a bare-adduct parent (Da); the sign matters
M_H = C.M["H"]
NAMED_SHIFTS = {
    "+H (M-. electron attachment / H adduct)": M_H,
    "13C": 1.0033548,
    "2x13C": 2 * 1.0033548,
    "15N": 0.9970349,
    "18O": 2.0042450,
    "34S": 1.9957959,
    "81Br": 1.9979521,
    "37Cl": 1.9970499,
    "-H (radical / dehydrogenated)": -M_H,
    "+H2O": 18.0105647,
    "-H2O": -18.0105647,
    "-CO2": -43.9898292,
    "-CO": -27.9949146,
    "+O": 15.9949146,
    "-O": -15.9949146,
    "+CH2": 14.0156501,
    "-CH2": -14.0156501,
}
SHIFT_PREFERENCE = {"+H (M-. electron attachment / H adduct)": 0, "13C": 1, "2x13C": 1, "15N": 1, "18O": 1, "34S": 1,
                    "81Br": 1, "37Cl": 1, "-H (radical / dehydrogenated)": 9}
WATER = 18.0105647
HNO3 = C.neutral_mass("HNO3")

# decoy: the adduct set a run of this polarity does NOT produce
WRONG_ADDUCTS = {"-": ["[M+Cl]-", "[M+I]-"], "+": ["[M+Na]+", "[M+NH4]+"]}
DECOY_MZ_SPLIT = 350.0   # below it a 0.35 Da shift sits in the CHNOS mass-defect gap


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
def resolve_run_dir(path: str) -> str:
    """The run dir itself, or the --out-dir that holds exactly one."""
    path = os.path.expanduser(path.rstrip("/"))
    if os.path.isfile(os.path.join(path, "merged_ledger.csv")):
        return path
    inner = sorted(
        d.rstrip("/")
        for d in glob.glob(os.path.join(path, "*/"))
        if os.path.isfile(os.path.join(d, "merged_ledger.csv"))
    )
    if len(inner) == 1:
        return inner[0]
    if not inner:
        raise SystemExit(f"no merged_ledger.csv in {path} or any child")
    raise SystemExit(f"{path} holds {len(inner)} run dirs; name the one you mean")


def _read_json(path: str) -> dict:
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def profile_for(reagent: str | None):
    """The reagent profile a run resolved, composed when it names several."""
    if not reagent:
        return None
    from peaky.chem import profiles as P

    parts = []
    for name in str(reagent).split("+"):
        prof = P.PROFILES.get(name) or P._BY_ALIAS.get(name.lower())
        if prof is None:
            return None
        parts.append(prof)
    return parts[0] if len(parts) == 1 else P.compose(parts)


@dataclass
class Run:
    path: str
    ledger: pd.DataFrame
    summary: dict
    manifest: dict
    ts: pd.DataFrame | None
    per_file: pd.DataFrame
    profile: object = None
    tables: dict = field(default_factory=dict)

    @property
    def name(self) -> str:
        return os.path.basename(self.path)

    @property
    def reagent(self) -> str:
        return str(self.summary.get("reagent") or "")

    @property
    def path_kind(self) -> str:
        return "trace-first" if self.summary.get("trace_first") else "cover"

    @property
    def polarity(self) -> str:
        if self.profile is not None:
            return self.profile.polarity
        adducts = self.ledger.get("adduct", pd.Series(dtype=str)).dropna().astype(str)
        return "+" if adducts.str.endswith("+").any() else "-"

    @property
    def adducts(self) -> list[str]:
        if self.profile is not None:
            return list(self.profile.adducts)
        return sorted(self.ledger.get("adduct", pd.Series(dtype=str)).dropna().astype(str).unique())

    @property
    def tol_ppm(self) -> float:
        return float(self.summary.get("tol_ppm") or self.summary.get("selection", {}).get("tol_ppm") or 6.0)

    @property
    def n_spectra(self) -> int:
        if self.ts is not None and "sample_item_id" in self.ts.columns:
            return int(self.ts["sample_item_id"].nunique())
        return int(self.summary.get("selection", {}).get("n_samples") or 0)

    @property
    def channel(self) -> str:
        batch = self.summary.get("batch_name") or self.manifest.get("batch") or self.name.rsplit("_", 1)[0]
        slug = re.sub(r"[^A-Za-z0-9]+", "-", str(batch)).strip("-")
        return f"{slug}|{self.reagent}|{self.path_kind}"

    @property
    def code(self) -> str:
        """`<package version> <git commit>` from run_manifest.json (`code.package_version`,
        `code.git.commit`; older shapes read too)."""
        code = self.manifest.get("code") or {}
        git = (code.get("git") or {}).get("commit") or self.manifest.get("git_commit") or (self.manifest.get("git") or {}).get("commit") or ""
        pkg = code.get("package_version") or self.manifest.get("package_version") or self.manifest.get("version") or ""
        return f"{pkg} {str(git)[:9]}".strip()


def load_run(path: str) -> Run:
    path = resolve_run_dir(path)
    ledger = pd.read_csv(os.path.join(path, "merged_ledger.csv"), low_memory=False)
    summary = _read_json(os.path.join(path, "batch_summary.json"))
    manifest = _read_json(os.path.join(path, "run_manifest.json"))
    ts_path = os.path.join(path, "per_file", "_batch_ts.parquet")
    ts = pd.read_parquet(ts_path) if os.path.isfile(ts_path) else None
    frames = []
    for f in sorted(glob.glob(os.path.join(path, "per_file", "*_ledger.csv"))):
        frame = pd.read_csv(f, low_memory=False)
        frame["__file"] = re.sub(r"_ledger\.csv$", "", os.path.basename(f))
        frames.append(frame)
    per_file = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    tables = {}
    for name in ("predicted_satellites", "residual_bins", "plausibility_audit"):
        hits = glob.glob(os.path.join(path, "tables", f"{name}*.csv"))
        if hits:
            try:
                tables[name] = pd.read_csv(hits[0], low_memory=False)
            except (OSError, ValueError, pd.errors.EmptyDataError):
                pass
    run = Run(path, ledger, summary, manifest, ts, per_file, tables=tables)
    run.profile = profile_for(run.reagent)
    return run


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def col(frame: pd.DataFrame, name: str, default=np.nan) -> pd.Series:
    if frame is not None and name in frame.columns:
        return frame[name]
    n = 0 if frame is None else len(frame)
    return pd.Series([default] * n, index=None if frame is None else frame.index)


def is_str(v) -> bool:
    return isinstance(v, str) and v != "" and v.lower() != "nan"


def elem(formula, e: str) -> int:
    return int(C.fold_isotopes(C.parse_formula(formula)).get(e, 0)) if is_str(formula) else 0


def pct(a, b) -> float:
    return float(100.0 * a / b) if b else 0.0


def ppm(mz, ref) -> float:
    return float((mz - ref) / ref * 1e6) if ref else float("nan")


def fmt(v, nd=0) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return ""
    if isinstance(v, (int, np.integer)):
        return f"{int(v):,}"
    if isinstance(v, (float, np.floating)):
        return f"{v:,.{nd}f}"
    return str(v)


def md_table(rows: list[dict], columns: list[tuple[str, str]], nd: dict | None = None) -> list[str]:
    """A GitHub table from records; `columns` = [(key, header)]; numbers right-aligned."""
    nd = nd or {}
    if not rows:
        return ["*(none)*"]
    head = "| " + " | ".join(h for _, h in columns) + " |"
    align = []
    for key, _ in columns:
        sample = next((r.get(key) for r in rows if r.get(key) is not None), None)
        align.append("---:" if isinstance(sample, (int, float, np.integer, np.floating)) else "---")
    lines = [head, "|" + "|".join(align) + "|"]
    for r in rows:
        cells = []
        for key, _ in columns:
            v = r.get(key)
            cells.append(fmt(v, nd.get(key, 0)) if isinstance(v, (float, np.floating)) else fmt(v))
        lines.append("| " + " | ".join(str(c).replace("|", "\\|") for c in cells) + " |")
    return lines


def gap_bins(mz: np.ndarray, tol_ppm: float) -> np.ndarray:
    """Group sorted m/z into tracks: a gap wider than tol_ppm starts a new one."""
    if len(mz) == 0:
        return np.array([], dtype=int)
    gaps = np.diff(mz) > tol_ppm * 1e-6 * mz[1:]
    return np.concatenate([[0], np.cumsum(gaps)])


def nearest(sorted_mz: np.ndarray, mz: float):
    """(index, value) of the nearest entry of a sorted array, or (None, nan)."""
    if len(sorted_mz) == 0:
        return None, float("nan")
    j = int(np.searchsorted(sorted_mz, mz))
    cands = [k for k in (j - 1, j) if 0 <= k < len(sorted_mz)]
    k = min(cands, key=lambda k: abs(sorted_mz[k] - mz))
    return k, float(sorted_mz[k])


def as_list(value) -> list:
    """A ledger cell holding a JSON / repr'd list of dicts."""
    if not is_str(value):
        return []
    try:
        parsed = json.loads(value)
    except ValueError:
        return LL.as_list(value)
    return parsed if isinstance(parsed, list) else []


# ---------------------------------------------------------------------------
# the ion table: one row per stamped ion of the time series
# ---------------------------------------------------------------------------
def ion_table(run: Run) -> pd.DataFrame:
    """Every stamped ion of the batch time series with its median height, its
    presence and the reading it carries. Keyed on `ion_mz` (the trace centre)."""
    ts = run.ts
    if ts is None or "ion_formula" not in ts.columns:
        return pd.DataFrame(columns=["ion_mz"])
    st = ts[ts["ion_formula"].notna() & ~col(ts, "dup_candidate", False).fillna(False).astype(bool)]
    if st.empty:
        return pd.DataFrame(columns=["ion_mz"])
    g = st.groupby("ion_mz", sort=True).agg(
        n=("sample_item_id", "nunique"),
        med_h=("height", "median"),
        sum_h=("height", "sum"),
        role=("role", "first"),
        ion_formula=("ion_formula", "first"),
        iso_label=("iso_label", "first"),
        neutral=("neutral_formula", "first"),
        adduct=("adduct", "first"),
        tier=("tier", "first"),
        suspect=("intensity_suspect", "any") if "intensity_suspect" in st.columns else ("height", "size"),
        stamp_source=("stamp_source", "first") if "stamp_source" in st.columns else ("height", "size"),
    ).reset_index()
    if "intensity_suspect" not in st.columns:
        g["suspect"] = False
    if "stamp_source" not in st.columns:
        g["stamp_source"] = ""
    g["presence"] = g["n"] / max(run.n_spectra, 1)
    return g


def unstamped_tracks(run: Run) -> pd.DataFrame:
    """The unstamped peaks of the time series, binned into tracks by tol_ppm."""
    ts = run.ts
    if ts is None or "ion_formula" not in ts.columns:
        return pd.DataFrame()
    un = ts[ts["ion_formula"].isna()].sort_values("mz")
    if un.empty:
        return pd.DataFrame()
    un = un.assign(track=gap_bins(un["mz"].to_numpy(float), run.tol_ppm))
    g = un.groupby("track").agg(
        mz=("mz", "median"),
        n=("sample_item_id", "nunique"),
        n_peaks=("mz", "size"),
        med_h=("height", "median"),
        sum_h=("height", "sum"),
    ).reset_index(drop=True)
    g["presence"] = g["n"] / max(run.n_spectra, 1)
    return g


# ---------------------------------------------------------------------------
# levels
# ---------------------------------------------------------------------------
def levels_for(run: Run, levels_csv: str | None, corroborate: list[str]) -> pd.DataFrame:
    """One row per (neutral, adduct) with `level`, `n_axes` and the axes.

    Prefers an in-core `evidence_level` column on the merged ledger; else the
    CSV `scripts/level_ledger.py --out` wrote; else levels the run in-process,
    corroborated by the run dirs in `corroborate` (the other instrument)."""
    led = run.ledger
    if "evidence_level" in led.columns:
        out = led[["neutral_formula", "adduct", "evidence_level"]].rename(
            columns={"neutral_formula": "neutral", "evidence_level": "level"}
        )
        axes = col(led, "evidence_axes", "")
        out["axes"] = axes.fillna("").astype(str)
        # the in-core string lists the four axes first, then modifiers
        # (multiline / carbon / branch / reagent_only_iso / known:<fam> / files:<n>)
        # -- only the axes count
        out["n_axes"] = out["axes"].map(
            lambda s: len([a for a in re.split(r"[|,;]", s)
                           if a in ("iso", "chan2", "anchor", "corroborated")]))
        out["source"] = "in-core"
        return out.drop_duplicates(["neutral", "adduct"])
    if levels_csv:
        df = pd.read_csv(levels_csv, low_memory=False)
        if "source" in df.columns and run.name in set(df["source"]):
            df = df[df["source"] == run.name]
    else:
        if run.per_file.empty and "role" not in led.columns:
            return pd.DataFrame(columns=["neutral", "adduct", "level", "n_axes", "axes"])
        df = LL.run([run.path], list(corroborate))
        if not df.empty and "source" in df.columns:
            df = df[df["source"] == run.name] if run.name in set(df["source"]) else df
    if df.empty:
        return pd.DataFrame(columns=["neutral", "adduct", "level", "n_axes", "axes"])
    axes_cols = [c for c in ("iso", "chan2", "anchor", "corroborated") if c in df.columns]
    df = df.copy()
    df["axes"] = df.apply(lambda r: ",".join(c for c in axes_cols if LL.truthy(r[c])), axis=1)
    if "n_axes" not in df.columns:
        df["n_axes"] = df["axes"].map(lambda s: len([a for a in s.split(",") if a]))
    keep = ["neutral", "adduct", "level", "n_axes", "axes"] + [c for c in ("known_fam",) if c in df.columns]
    return df[keep].drop_duplicates(["neutral", "adduct"])


def level_vector(levels: pd.DataFrame) -> dict:
    counts = levels["level"].value_counts() if "level" in levels.columns else pd.Series(dtype=int)
    return {k: int(counts.get(k, 0)) for k in LEVELS}


def level_rank(level) -> int:
    return LEVELS.index(level) if level in LEVELS else len(LEVELS)


# ---------------------------------------------------------------------------
# 1. headline
# ---------------------------------------------------------------------------
def headline(run: Run, ions: pd.DataFrame) -> dict:
    led = run.ledger
    ts = run.ts
    tier = col(led, "tier", "")
    out = {
        "run": run.name,
        "channel": run.channel,
        "reagent": run.reagent,
        "path": run.path_kind,
        "code": run.code,
        "n_files": int(run.summary.get("n_files") or col(led, "n_files").max() or 0),
        "n_spectra": run.n_spectra,
        "merged_rows": int(len(led)),
        "assigned": int((tier == "Assigned").sum()),
        "candidate": int((tier == "Candidate").sum()),
        "neutrals": int(col(led, "neutral_formula").dropna().nunique()),
        "neutrals_assigned": int(led.loc[tier == "Assigned", "neutral_formula"].dropna().nunique()) if "neutral_formula" in led.columns else 0,
        "by_stage": col(led, "stage", "cover").fillna("cover").value_counts().to_dict(),
        "by_adduct": col(led, "adduct", "").fillna("").value_counts().to_dict(),
        "by_adduct_assigned": led.loc[tier == "Assigned", "adduct"].fillna("").value_counts().to_dict() if "adduct" in led.columns else {},
        "n_in_all_files": run.summary.get("n_in_all_files"),
        "n_single_file": run.summary.get("n_single_file"),
        "ion_disagreements": run.summary.get("ion_disagreements"),
        "elapsed_s": run.summary.get("elapsed_s"),
    }
    if ts is not None and "ion_formula" in ts.columns:
        stamped = ts["ion_formula"].notna()
        out.update(
            n_peaks=int(len(ts)),
            stamped_peak_share=pct(stamped.sum(), len(ts)),
            stamped_signal_share=pct(ts.loc[stamped, "height"].sum(), ts["height"].sum()),
            stamped_M0_signal_share=pct(ts.loc[stamped & (col(ts, "role", "") == "M0"), "height"].sum(), ts["height"].sum()),
            unstamped_signal_share=pct(ts.loc[~stamped, "height"].sum(), ts["height"].sum()),
        )
    else:
        out.update(n_peaks=0, stamped_peak_share=float("nan"), stamped_signal_share=float("nan"),
                   stamped_M0_signal_share=float("nan"), unstamped_signal_share=float("nan"))
    # stamp coverage: merged M0 rows the time series never carries
    out["stamp_coverage"] = stamp_coverage(run, ions)
    return out


def stamp_coverage(run: Run, ions: pd.DataFrame) -> dict:
    """Merged rows with no stamped peak in the time series, and why."""
    led = run.ledger
    if run.ts is None or "neutral_formula" not in led.columns or ions.empty:
        return {"n_rows": int(len(led)), "n_unstamped": None, "rows": []}
    stamped_keys = set(
        zip(ions.loc[ions["role"] == "M0", "neutral"].astype(str), ions.loc[ions["role"] == "M0", "adduct"].astype(str))
    )
    keys = list(zip(led["neutral_formula"].astype(str), col(led, "adduct", "").astype(str)))
    collapsed = col(led, "trace_role", "").astype(str) == "collapsed"
    miss = [i for i, k in enumerate(keys) if k not in stamped_keys and not collapsed.iloc[i]]
    rows = []
    for i in miss:
        r = led.iloc[i]
        rows.append(
            {
                "mz": float(r["mz"]),
                "neutral": str(r["neutral_formula"]),
                "adduct": str(r.get("adduct", "")),
                "tier": str(r.get("tier", "")),
                "n_files": int(r["n_files"]) if pd.notna(r.get("n_files")) else None,
                "tier_reason": str(r["tier_reason"]) if is_str(r.get("tier_reason")) else "",
                "alternatives": str(r["alternatives"])[:80] if is_str(r.get("alternatives")) else "",
                "trace_role": str(r.get("trace_role", "")),
            }
        )
    rows.sort(key=lambda r: -(r["n_files"] or 0))
    gates = run.summary.get("merge_gates") or {}
    relabeled = (gates.get("reagent_n") or {}).get("reagent_n_relabeled")
    return {
        "n_rows": int(len(led)),
        "n_unstamped": int(len(miss)),
        "n_collapsed": int(collapsed.sum()),
        "reagent_n_relabeled": relabeled,
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# 2 + 3. the brightest and the best-evidence ions
# ---------------------------------------------------------------------------
def brightest(run: Run, ions: pd.DataFrame, tracks: pd.DataFrame, levels: pd.DataFrame, n: int = TOP_N) -> dict:
    """The n brightest stamped ions (median height over the batch) with their
    reading, tier, level and axes; and how many of the batch's n brightest
    TRACKS overall are unstamped (those are named in M1)."""
    if ions.empty:
        return {"rows": [], "m0_not_assigned": 0, "unstamped_in_top": 0, "n": 0}
    lv = levels.set_index(["neutral", "adduct"]) if not levels.empty else None
    top = ions.sort_values("med_h", ascending=False).head(n)
    rows = []
    for r in top.itertuples():
        level, axes = "", ""
        if lv is not None and is_str(r.neutral) and (r.neutral, r.adduct) in lv.index:
            hit = lv.loc[(r.neutral, r.adduct)]
            hit = hit.iloc[0] if isinstance(hit, pd.DataFrame) else hit
            level, axes = str(hit["level"]), str(hit["axes"])
        rows.append(
            {
                "ion_mz": float(r.ion_mz),
                "med_cps": float(r.med_h),
                "in": f"{int(r.n)}/{run.n_spectra}",
                "role": str(r.role),
                "ion": str(r.ion_formula) + (f" ({r.iso_label})" if is_str(r.iso_label) and r.role == "iso_child" else ""),
                "neutral": str(r.neutral) if is_str(r.neutral) else "",
                "adduct": str(r.adduct) if is_str(r.adduct) else "",
                "tier": str(r.tier) if is_str(r.tier) else "",
                "level": level,
                "axes": axes,
                "suspect": bool(r.suspect),
            }
        )
    m0_not_assigned = sum(1 for r in rows if r["role"] == "M0" and r["tier"] != "Assigned")
    # the batch's brightest tracks overall: stamped ions and unstamped tracks together
    all_tracks = pd.concat(
        [
            ions[["med_h"]].assign(kind="stamped"),
            (tracks[["med_h"]].assign(kind="unstamped") if not tracks.empty else pd.DataFrame(columns=["med_h", "kind"])),
        ],
        ignore_index=True,
    ).sort_values("med_h", ascending=False).head(n)
    return {
        "rows": rows,
        "n": len(rows),
        "m0_not_assigned": int(m0_not_assigned),
        "by_role": {k: int(v) for k, v in pd.Series([r["role"] for r in rows]).value_counts().items()},
        "unstamped_in_top": int((all_tracks["kind"] == "unstamped").sum()),
    }


def best_evidence(run: Run, ions: pd.DataFrame, levels: pd.DataFrame, n: int = TOP_N) -> dict:
    """The n best-evidence M0 rows: level <= 4a first, then most axes, then
    brightest; plus the level and axes histograms over every levelled row."""
    if levels.empty:
        return {"rows": [], "levels": level_vector(levels), "axes_hist": {}, "n_good": 0}
    m0 = ions[ions["role"] == "M0"][["neutral", "adduct", "ion_mz", "med_h", "n", "tier"]] if not ions.empty else pd.DataFrame()
    df = levels.merge(m0, on=["neutral", "adduct"], how="left") if not m0.empty else levels.assign(ion_mz=np.nan, med_h=np.nan, n=np.nan, tier="")
    df["rank"] = df["level"].map(level_rank)
    df["good"] = df["level"].isin(GOOD_LEVELS)
    top = df.sort_values(["good", "n_axes", "med_h"], ascending=[False, False, False]).head(n)
    rows = [
        {
            "neutral": str(r.neutral),
            "adduct": str(r.adduct),
            "ion_mz": float(r.ion_mz) if pd.notna(r.ion_mz) else None,
            "med_cps": float(r.med_h) if pd.notna(r.med_h) else None,
            "in": f"{int(r.n)}/{run.n_spectra}" if pd.notna(r.n) else "",
            "tier": str(r.tier) if is_str(r.tier) else "",
            "level": str(r.level),
            "axes": str(r.axes),
        }
        for r in top.itertuples()
    ]
    return {
        "rows": rows,
        "levels": level_vector(levels),
        "axes_hist": {int(k): int(v) for k, v in levels["n_axes"].value_counts().sort_index().items()},
        "n_good": int(df["good"].sum()),
        "n_levelled": int(len(df)),
    }


# ---------------------------------------------------------------------------
# 4. what was missed
# ---------------------------------------------------------------------------
def _cleared_reason(commentary) -> str:
    """'CLEARED (mass-gate: z=8.9 > 4.0). Was: Pass 1 (...): C9H17NO [M+^NO3]-, ...'
    -> 'mass-gate: z=8.9 > 4.0; tried C9H17NO [M+^NO3]-'."""
    if not is_str(commentary):
        return ""
    m = re.match(r"CLEARED \((.*?)\)\. Was: (.*)", commentary)
    if not m:
        return commentary[:90]
    why, was = m.group(1), m.group(2)
    tried = re.search(r"([A-Z][A-Za-z0-9^]*\d*[A-Za-z0-9^]*) (\[M[^\]]*\][+-]\.?)", was)
    return why + (f"; tried {tried.group(1)} {tried.group(2)}" if tried else "")


def residual_tags(run: Run) -> pd.DataFrame:
    """`characterize_residual`'s tag for every per-file unexplained row, plus
    the engine's own reason for clearing it. One row per (file, m/z)."""
    pf = run.per_file
    if pf.empty or "role" not in pf.columns:
        return pd.DataFrame(columns=["mz", "tag", "c_count", "n_Br", "n_Cl", "reason"])
    from peaky.assignment import residual as RS

    out = []
    for _, frame in pf.groupby("__file"):
        frame = frame.reset_index(drop=True)
        if not (frame["role"] == "unexplained").any():
            continue
        try:
            tags = RS.characterize_residual(frame)
        except Exception:  # a ledger shape the characterizer does not expect
            continue
        reasons = frame.set_index("peak_id")["commentary"] if "commentary" in frame.columns else pd.Series(dtype=str)
        for r in tags.itertuples():
            out.append(
                {
                    "mz": float(r.mz),
                    "tag": str(r.tier),
                    "c_count": r.c_count or "",
                    "n_Br": int(r.n_Br),
                    "n_Cl": int(r.n_Cl),
                    "reason": _cleared_reason(reasons.get(r.peak_id)),
                }
            )
    return pd.DataFrame(out, columns=["mz", "tag", "c_count", "n_Br", "n_Cl", "reason"])


def _tag_for(tags: pd.DataFrame, mz: float, tol_ppm: float) -> dict:
    if tags.empty:
        return {"tag": "", "c_count": "", "reason": "", "n_files": 0}
    d = (tags["mz"] - mz).abs()
    hit = tags[d <= tol_ppm * 1e-6 * mz]
    if hit.empty:
        return {"tag": "", "c_count": "", "reason": "", "n_files": 0}
    reasons = hit["reason"][hit["reason"] != ""]
    return {
        "tag": hit["tag"].mode().iloc[0],
        "c_count": next((c for c in hit["c_count"] if c), ""),
        "reason": reasons.mode().iloc[0] if not reasons.empty else "",
        "n_files": int(len(hit)),
    }


def _named_shift(delta: float) -> str | None:
    for name, d in NAMED_SHIFTS.items():
        if abs(delta - d) * 1000 <= SHIFT_TOL_MDA:
            return name
    return None


def _channel_shift(run: Run, delta: float) -> str | None:
    """A shift that turns a bare-adduct parent into one of the run's other
    channels (an unstamped cluster line of an assigned neutral)."""
    bare = next((a for a in run.adducts if a in BARE), None)
    if bare is None or bare not in C.ADDUCT_SHIFTS:
        return None
    for a in run.adducts:
        if a == bare or a not in C.ADDUCT_SHIFTS:
            continue
        if abs(delta - (C.ADDUCT_SHIFTS[a] - C.ADDUCT_SHIFTS[bare])) * 1000 <= SHIFT_TOL_MDA:
            return f"{a} line of the {bare} parent"
    return None


def _reagent_shift(delta: float) -> str | None:
    for n in (1, 2, 3):
        if abs(delta - n * WATER) * 1000 <= SHIFT_TOL_MDA:
            return f"reagent + {n}x H2O"
        if abs(delta - n * HNO3) * 1000 <= SHIFT_TOL_MDA:
            return f"reagent + {n}x HNO3"
    return None


def _named_parent(run: Run, mz: float, bare: pd.DataFrame, bare_mz: np.ndarray):
    """(label, parent text) for the brightest bare-adduct parent within +-3
    neighbours whose shift to `mz` is a named one. Brightness decides a tie
    (a +H comb rides on bright parents; a -H reading of the next acid up is the
    same peak seen from the wrong side)."""
    if not len(bare_mz):
        return None, None
    k, _ = nearest(bare_mz, mz)
    best = None
    for cand in range(max(0, k - 4), min(len(bare_mz), k + 5)):
        d = mz - bare_mz[cand]
        name = _named_shift(d) or _channel_shift(run, d)
        if not name:
            continue
        # a track between two acids that differ by H2 fits +H of the lighter
        # and -H of the heavier exactly; the +H reading (M-.) is the family
        # the first cut established, so it wins before brightness does
        key = (SHIFT_PREFERENCE.get(name, 5), float(bare.iloc[cand]["med_h"]))
        if best is None or (key[0] < best[0][0]) or (key[0] == best[0][0] and key[1] > best[0][1]):
            best = (key, name, f"{bare.iloc[cand]['neutral']} {bare.iloc[cand]['adduct']} @ {bare_mz[cand]:.4f}")
    return (best[1], best[2]) if best else (None, None)


def missed_m1(run: Run, ions: pd.DataFrame, tracks: pd.DataFrame, presence: float = PRESENCE, n: int = 25,
              coverage_rows: list | None = None) -> dict:
    """M1: the brightest unstamped tracks present in >= `presence` of the
    spectra, each joined to the nearest stamped ion and to the nearest bare
    parent, labelled and GROUPED into families by that shift. The rules, in
    order: a merged row the stamp never carried; a named shift from a bare
    parent (+H, isotopes, a cluster line, a neutral loss); a reagent-ladder
    line; an isotope line of any stamped M0; an isotope line of a brighter
    unstamped track; a shoulder of a brighter stamped ion; else unknown."""
    if tracks.empty:
        return {"families": [], "rows": [], "n_tracks": 0, "signal_share": float("nan")}
    persistent = tracks[tracks["presence"] >= presence].copy()
    ts_sum = float(run.ts["height"].sum()) if run.ts is not None else float("nan")
    tags = residual_tags(run)
    st_mz = ions["ion_mz"].to_numpy(float) if not ions.empty else np.array([])
    st = ions.reset_index(drop=True)
    m0 = ions[(ions["role"] == "M0")]
    bare = m0[m0["adduct"].isin(BARE)] if not m0.empty else m0
    bare_mz = bare["ion_mz"].to_numpy(float)
    bare = bare.reset_index(drop=True)
    reag = ions[ions["role"] == "reagent"].reset_index(drop=True)
    reag_mz = reag["ion_mz"].to_numpy(float)
    cov = pd.DataFrame(coverage_rows or [])
    cov_mz = cov["mz"].to_numpy(float) if not cov.empty else np.array([])
    order = np.argsort(cov_mz)
    cov_mz, cov = cov_mz[order], (cov.iloc[order].reset_index(drop=True) if not cov.empty else cov)
    un_sorted = persistent.sort_values("mz")
    un_mz = un_sorted["mz"].to_numpy(float)
    un_h = un_sorted["med_h"].to_numpy(float)
    ISO_ONLY = {"13C", "2x13C", "15N", "18O", "34S", "81Br", "37Cl"}
    rows = []
    for r in persistent.sort_values("med_h", ascending=False).itertuples():
        mz = float(r.mz)
        j, near_mz = nearest(st_mz, mz)
        near = st.iloc[j] if j is not None else None
        d_near = (mz - near_mz) if j is not None else float("nan")
        label, parent = "", ""
        # (0) a merged row the stamp never carried
        if len(cov_mz):
            k, c_mz = nearest(cov_mz, mz)
            if abs(c_mz - mz) <= run.tol_ppm * 1e-6 * mz:
                c = cov.iloc[k]
                label, parent = "merged but unstamped (stamp-coverage row)", f"{c['neutral']} {c['adduct']} {c['tier']}"
        # (1) a named shift from a bare-adduct parent
        if not label:
            name, ptxt = _named_parent(run, mz, bare, bare_mz)
            if name:
                label, parent = name, ptxt
        # (2) a reagent-ladder line: every reagent ion within 3 x HNO3 below the
        # track (a mass window, not an index window -- the ladder is dense near
        # the reagent and Br-.H2O sits 18 Da above a Br- that has BrO- between)
        if not label and len(reag_mz):
            lo = int(np.searchsorted(reag_mz, mz - 3 * HNO3 - 0.01))
            hi = int(np.searchsorted(reag_mz, mz))
            for cand in sorted(range(lo, hi), key=lambda q: abs(reag_mz[q] - mz)):
                name = _reagent_shift(mz - reag_mz[cand])
                if name:
                    label, parent = name, f"{reag.iloc[cand]['ion_formula']} @ {reag_mz[cand]:.4f}"
                    break
        # (3) an isotope line of any stamped M0 (not only the bare channel)
        if not label and near is not None and near["role"] == "M0":
            name = _named_shift(d_near)
            if name in ISO_ONLY:
                label, parent = f"unstamped {name} line", f"{near['neutral']} {near['adduct']} @ {near_mz:.4f}"
        # (4) an isotope line of a brighter unstamped track
        if not label and len(un_mz):
            k, u_mz = nearest(un_mz, mz - 1.0033548)
            for cand in range(max(0, k - 2), min(len(un_mz), k + 3)):
                d = mz - un_mz[cand]
                name = _named_shift(d)
                if name in ISO_ONLY and un_h[cand] > r.med_h:
                    label, parent = f"{name} of an unstamped track", f"unstamped @ {un_mz[cand]:.4f}"
                    break
        # (5) a shoulder / sidelobe of a brighter stamped ion within SHOULDER_MDA
        if not label and near is not None and abs(d_near) * 1000 <= SHOULDER_MDA and near["med_h"] > r.med_h:
            label = "shoulder / sidelobe of a brighter stamped ion" + (" (intensity_suspect)" if bool(near["suspect"]) else "")
            parent = f"{near['ion_formula']} @ {near_mz:.4f}"
        tag = _tag_for(tags, mz, run.tol_ppm)
        rows.append(
            {
                "mz": mz,
                "med_cps": float(r.med_h),
                "in": f"{int(r.n)}/{run.n_spectra}",
                "signal_share": pct(r.sum_h, ts_sum),
                "nearest_stamped": f"{near['ion_formula']}" if near is not None else "",
                "nearest_mz": near_mz if j is not None else None,
                "d_mda": d_near * 1000 if j is not None else None,
                "d_ppm": ppm(mz, near_mz) if j is not None else None,
                "family": label or "unknown",
                "parent": parent,
                "tag": tag["tag"],
                "c_count": tag["c_count"],
                "reason": tag["reason"],
            }
        )
    df = pd.DataFrame(rows)
    families = []
    if not df.empty:
        for fam, g in df.groupby("family", sort=False):
            top3 = g.sort_values("med_cps", ascending=False).head(3)
            families.append(
                {
                    "family": fam,
                    "n_tracks": int(len(g)),
                    "signal_share": float(g["signal_share"].sum()),
                    "max_cps": float(g["med_cps"].max()),
                    "examples": "; ".join(f"{t.mz:.4f} ({t.parent})" if t.parent else f"{t.mz:.4f}" for t in top3.itertuples()),
                }
            )
        families.sort(key=lambda f: -f["signal_share"])
    return {
        "families": families,
        "rows": rows[:n],
        "n_tracks": int(len(persistent)),
        "n_tracks_total": int(len(tracks)),
        "signal_share": float(persistent["sum_h"].sum() / ts_sum * 100) if ts_sum else float("nan"),
        "n_families": int(sum(1 for f in families if f["n_tracks"] >= FAMILY_MIN and f["family"] != "unknown")),
    }


# --- M2: expected but absent ------------------------------------------------
def load_rosters(directory: str | None = None) -> pd.DataFrame:
    """Every roster CSV under peaky/data/rosters (or `directory`), with `roster`."""
    directory = directory or pkg_data("rosters")
    frames = []
    for p in sorted(glob.glob(os.path.join(directory, "*.csv"))):
        df = pd.read_csv(p)
        df["roster"] = os.path.basename(p)[:-4]
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["name", "formula", "class", "reference", "note", "roster"])
    return pd.concat(frames, ignore_index=True)


def expectations(run: Run, rosters: pd.DataFrame) -> pd.DataFrame:
    """What this run is expected to see: the rosters, the reagent's reference
    ions and the pass-0 known species of its polarity. One row per (source,
    neutral); `channels` lists the adducts the source pins, if any."""
    rows = [
        {"source": f"roster:{r['roster']}", "name": r["name"], "neutral": r["formula"], "cls": r["class"], "channels": ""}
        for _, r in rosters.iterrows()
    ]
    try:
        from peaky.chem import reference_ions as RI

        ref = RI.get(run.reagent) if run.reagent else None
    except Exception:
        ref = None
    if ref is not None and not ref.empty:
        for r in ref[ref["iso"].fillna("") == ""].itertuples():
            rows.append({"source": "reference_ion", "name": r.identity, "neutral": r.neutral, "cls": f"grade {r.grade}", "channels": r.channel})
    try:
        from peaky.assignment.passes.directors import _known_species

        known = _known_species("positive" if run.polarity == "+" else "negative")
    except Exception:
        known = {}
    for fam, table in (known or {}).items():
        for formula, name in (table or {}).items():
            rows.append({"source": f"known:{fam}", "name": str(name), "neutral": str(formula), "cls": fam, "channels": ""})
    df = pd.DataFrame(rows, columns=["source", "name", "neutral", "cls", "channels"])
    return df.drop_duplicates(["source", "neutral"]).reset_index(drop=True)


def missed_m2(run: Run, ions: pd.DataFrame, tracks: pd.DataFrame, rosters: pd.DataFrame) -> dict:
    """M2: every expected neutral, looked for on the run's own channels in the
    time series: assigned as itself / read as something else / present but
    unstamped (with the engine's reason) / absent. Roster recall per class."""
    exp = expectations(run, rosters)
    if exp.empty:
        return {"rows": [], "roster": {}, "sources": {}}
    st_mz = ions["ion_mz"].to_numpy(float) if not ions.empty else np.array([])
    st = ions.reset_index(drop=True)
    tr_mz = tracks["mz"].to_numpy(float) if not tracks.empty else np.array([])
    tr = tracks.reset_index(drop=True)
    tags = residual_tags(run) if not tracks.empty else pd.DataFrame(columns=["mz", "tag", "c_count", "n_Br", "n_Cl", "reason"])
    led_neutrals = set(run.ledger["neutral_formula"].dropna().astype(str)) if "neutral_formula" in run.ledger.columns else set()
    led_tier = run.ledger.groupby("neutral_formula")["tier"].agg(lambda s: "Assigned" if (s == "Assigned").any() else "Candidate").to_dict() if "tier" in run.ledger.columns else {}
    rows = []
    for e in exp.itertuples():
        channels = [e.channels] if is_str(e.channels) else run.adducts
        best = None
        for adduct in channels:
            if adduct not in C.ADDUCT_SHIFTS:
                continue
            try:
                target = C.ion_mz(e.neutral, adduct)
            except Exception:
                continue
            tol = run.tol_ppm * 1e-6 * target
            # stamped ion on the line?
            j, mz = nearest(st_mz, target)
            if j is not None and abs(mz - target) <= tol:
                s = st.iloc[j]
                same = is_str(s["neutral"]) and s["neutral"] == e.neutral
                status = ("assigned" if s["tier"] == "Assigned" else "candidate") if same else "read as"
                cand = {"status": status, "adduct": adduct, "mz": float(mz), "cps": float(s["med_h"]), "in": f"{int(s['n'])}/{run.n_spectra}",
                        "read": "" if same else f"{s['ion_formula']}" + (f" = {s['neutral']} {s['adduct']}" if is_str(s["neutral"]) else f" ({s['role']})"),
                        "reason": ""}
            else:
                k, mz = nearest(tr_mz, target)
                if k is not None and abs(mz - target) <= tol and tr.iloc[k]["presence"] >= ROSTER_PRESENCE:
                    t = tr.iloc[k]
                    tag = _tag_for(tags, float(mz), run.tol_ppm)
                    cand = {"status": "unstamped", "adduct": adduct, "mz": float(mz), "cps": float(t["med_h"]), "in": f"{int(t['n'])}/{run.n_spectra}",
                            "read": "", "reason": tag["reason"] or (tag["tag"] and f"residual tag: {tag['tag']}") or ""}
                else:
                    cand = {"status": "absent", "adduct": adduct, "mz": None, "cps": None, "in": "", "read": "", "reason": ""}
            order = {"assigned": 0, "candidate": 1, "read as": 2, "unstamped": 3, "absent": 4}
            if best is None or order[cand["status"]] < order[best["status"]] or (
                cand["status"] == best["status"] and (cand["cps"] or 0) > (best["cps"] or 0)
            ):
                best = cand
        if best is None:
            continue
        in_ledger = e.neutral in led_neutrals
        rows.append(
            {
                "source": e.source, "name": e.name, "neutral": e.neutral, "cls": e.cls,
                "status": best["status"], "adduct": best["adduct"], "mz": best["mz"], "cps": best["cps"], "in": best["in"],
                "read": best["read"], "reason": best["reason"],
                "ledger": led_tier.get(e.neutral, "") if in_ledger else "",
            }
        )
    df = pd.DataFrame(rows)
    roster_summary, per_class = {}, {}
    if not df.empty:
        ros = df[df["source"].str.startswith("roster:")].drop_duplicates("neutral")
        for name, g in ros.groupby(ros["source"].str[7:]):
            roster_summary[name] = _recall(g)
            per_class[name] = {c: _recall(gc) for c, gc in g.groupby("cls")}
        sources = {s: _recall(g) for s, g in df[~df["source"].str.startswith("roster:")].groupby("source")}
    else:
        sources = {}
    return {"rows": rows, "roster": roster_summary, "roster_by_class": per_class, "sources": sources}


def _recall(g: pd.DataFrame) -> dict:
    n = int(len(g))
    st = g["status"]
    return {
        "n": n,
        "assigned": int((st == "assigned").sum()),
        "candidate": int((st == "candidate").sum()),
        "read_as": int((st == "read as").sum()),
        "unstamped": int((st == "unstamped").sum()),
        "absent": int((st == "absent").sum()),
        "present": int((st != "absent").sum()),
    }


# --- M3: found elsewhere ----------------------------------------------------
def _window_mask(ts: pd.DataFrame, overlap: tuple | None, masks: list[tuple]) -> pd.Series:
    if "datetime_utc" not in ts.columns:
        return pd.Series(True, index=ts.index)
    t = pd.to_datetime(ts["datetime_utc"], utc=True)
    keep = pd.Series(True, index=ts.index)
    if overlap:
        keep &= (t >= overlap[0]) & (t <= overlap[1])
    for a, b in masks:
        keep &= ~((t >= a) & (t <= b))
    return keep


def _parse_window(text: str | None) -> tuple | None:
    if not text:
        return None
    a, b = [pd.Timestamp(x.strip(), tz="UTC") for x in text.split(",")]
    return (a, b)


def missed_m3(run: Run, other: Run | None, other_instrument: Run | None, other_levels: pd.DataFrame | None,
              overlap: tuple | None, masks: list[tuple], floor_cps: float, floor_share: float) -> dict:
    """M3: neutrals the other path found (>= 2 files or Assigned) that this run
    lacks; and neutrals the other INSTRUMENT holds at level <= 4b, above the
    detection floor inside the overlap window, that this run lacks."""
    mine = set(run.ledger["neutral_formula"].dropna().astype(str)) if "neutral_formula" in run.ledger.columns else set()
    out = {"other_path": None, "other_instrument": None}
    if other is not None and "neutral_formula" in other.ledger.columns:
        o = other.ledger
        solid = o[(col(o, "n_files", 0).fillna(0) >= 2) | (col(o, "tier", "") == "Assigned")]
        missing = solid[~solid["neutral_formula"].astype(str).isin(mine)].dropna(subset=["neutral_formula"])
        missing = missing.sort_values("n_files", ascending=False) if "n_files" in missing.columns else missing
        out["other_path"] = {
            "run": other.name, "path": other.path_kind,
            "n_solid": int(solid["neutral_formula"].nunique()),
            "n_missing": int(missing["neutral_formula"].nunique()),
            "rows": [
                {"neutral": str(r.neutral_formula), "adduct": str(getattr(r, "adduct", "")), "mz": float(r.mz),
                 "tier": str(getattr(r, "tier", "")), "n_files": int(getattr(r, "n_files", 0) or 0)}
                for r in missing.drop_duplicates("neutral_formula").head(25).itertuples()
            ],
        }
    if other_instrument is not None and other_levels is not None and not other_levels.empty and other_instrument.ts is not None:
        ots = other_instrument.ts
        keep = _window_mask(ots, overlap, masks)
        w = ots[keep & ots["neutral_formula"].notna() & ~col(ots, "dup_candidate", False).fillna(False).astype(bool)]
        n_in_window = int(ots.loc[keep, "sample_item_id"].nunique()) if "sample_item_id" in ots.columns else 0
        good = other_levels[other_levels["level"].map(level_rank) <= level_rank("4b")]
        rows = []
        for r in good.itertuples():
            tr = w[(w["neutral_formula"] == r.neutral) & (w["adduct"] == r.adduct)]
            if tr.empty or n_in_window == 0:
                continue
            med = float(tr["height"].median())
            share = tr["sample_item_id"].nunique() / n_in_window
            if med < floor_cps or share < floor_share:
                continue
            if r.neutral in mine:
                continue
            rows.append({"neutral": str(r.neutral), "adduct": str(r.adduct), "level": str(r.level), "axes": str(r.axes),
                         "med_cps": med, "share": float(share * 100)})
        rows.sort(key=lambda x: (level_rank(x["level"]), -x["med_cps"]))
        out["other_instrument"] = {
            "run": other_instrument.name, "reagent": other_instrument.reagent,
            "n_good": int(len(good)), "n_spectra_in_window": n_in_window,
            "n_missing": int(len(rows)),
            "rows": rows[:40],
            "window": [str(overlap[0]), str(overlap[1])] if overlap else None,
            "floor": {"cps": floor_cps, "share": floor_share},
        }
    return out


# ---------------------------------------------------------------------------
# 5. is it right
# ---------------------------------------------------------------------------
def census(run: Run) -> dict:
    """Element census of the Assigned neutrals: heteroatoms in a clean
    oxidation experiment are contamination or error until named."""
    led = run.ledger
    if "neutral_formula" not in led.columns:
        return {"n_assigned": 0, "elements": {}, "examples": {}}
    ass = led[(col(led, "tier", "") == "Assigned") & led["neutral_formula"].notna()]
    out, examples = {}, {}
    for e in CENSUS:
        hit = ass[ass["neutral_formula"].map(lambda f: elem(f, e) > 0)]
        out[e] = int(len(hit))
        if not hit.empty:
            top = hit.sort_values("n_files", ascending=False) if "n_files" in hit.columns else hit
            examples[e] = ", ".join(f"{r.neutral_formula} {getattr(r, 'adduct', '')}" for r in top.head(4).itertuples())
    heavy = ass[ass["neutral_formula"].map(lambda f: elem(f, "C") > C_HEAVY)]
    out[f"C>{C_HEAVY}"] = int(len(heavy))
    if not heavy.empty:
        examples[f"C>{C_HEAVY}"] = ", ".join(f"{r.neutral_formula} {getattr(r, 'adduct', '')}" for r in heavy.head(4).itertuples())
    return {"n_assigned": int(len(ass)), "elements": out, "examples": examples}


# --- decoy ------------------------------------------------------------------
PEAK_COLS = ["sample_item_id", "peak_id", "mz", "sparsity", "area", "height"]
MATCH_COLS = ("match_score_isotope", "relative_abundance", "target_isotope_id", "target_isotope_formula",
              "target_ion_id", "target_ion_formula", "target_compound_id", "target_compound_name",
              "target_compound_formula", "target_collection_ids", "match_score_ion", "match_score_compound",
              "ionization_mechanism")


def raw_peaks_of(run: Run, file_id: str) -> pd.DataFrame:
    """The raw peak table of one assigned file, rebuilt from its per-file
    ledger: what `fetch_peaks` would have returned, minus the server matches."""
    pf = run.per_file[run.per_file["__file"] == file_id]
    if "synthetic" in pf.columns:
        pf = pf[~pf["synthetic"].fillna(False).astype(bool)]
    t = pd.DataFrame({c: pf[c].to_numpy() if c in pf.columns else np.nan for c in PEAK_COLS})
    t["sample_item_id"] = file_id
    for c in MATCH_COLS:
        t[c] = None
    return t.dropna(subset=["mz", "height"]).reset_index(drop=True)


def decoy_peaks(peaks: pd.DataFrame, offset_da: float) -> pd.DataFrame:
    """Every m/z shifted by `offset_da`: the same spectrum, on a mass axis no
    CHNOS formula sits on (0.35 Da lands in the mass-defect gap below ~m/z 350)."""
    out = peaks.copy()
    out["mz"] = out["mz"] + offset_da
    out["peak_id"] = out["peak_id"].astype(str) + "_decoy"
    return out


def wrong_adducts(polarity: str) -> list[str]:
    return list(WRONG_ADDUCTS.get(polarity, WRONG_ADDUCTS["-"]))


def brightest_files(run: Run, n: int) -> list[str]:
    pf = run.per_file
    if pf.empty:
        return []
    order = pf.groupby("__file")["height"].sum().sort_values(ascending=False)
    return list(order.index[:n])


def run_engine_offline(run: Run, peaks: pd.DataFrame, sample_id: str, adducts: list[str], log=lambda *a: None) -> pd.DataFrame:
    """`assign.run(peaks=)` on one table with the run's own profile settings.
    Returns the ledger. Needs the local scorer (the default)."""
    import copy

    from peaky.assignment import assign as A
    from peaky.assignment import passes as PA
    from peaky.chem import profiles as P
    from peaky.io import io_mascope as IO

    cfg = PA.PassConfig()
    P.apply_height_cutoff_x_edge(cfg, run.profile, log=log)
    kw = {"adducts": list(adducts), "reagent_n_relabel": False}
    if run.profile is not None:
        if getattr(run.profile, "label_isotope", None):
            kw["label_isotope"] = run.profile.label_isotope
            kw["label_max"] = run.profile.label_max
        if getattr(run.profile, "purity", None):
            kw["label_purity"] = run.profile.purity
    context = (run.profile.context if run.profile is not None else None) or run.summary.get("context") or "ambient-air"
    try:
        res = A.run(sample_id, context=context, cfg=copy.deepcopy(cfg), peaks=peaks, use_cache=False, log=log, **kw)
    finally:
        IO.unregister_offline_sample(sample_id)
    return res["ledger"]


def _ledger_counts(led: pd.DataFrame, label: str) -> dict:
    m0 = led[led["role"] == "M0"] if "role" in led.columns else led.iloc[0:0]
    tier = col(m0, "tier", "")
    frame = m0.assign(__file=label)
    levels = pd.DataFrame()
    if not frame.empty and "evidence_level" in m0.columns and m0["evidence_level"].notna().any():
        # the engine at this tip levels its own rows (the `evidence` stage): the
        # decoy arms are then rated by the SAME leveller as the run they bound
        levels = (m0.drop_duplicates(["neutral_formula", "adduct"])
                    .rename(columns={"evidence_level": "level"})[["neutral_formula", "adduct", "level"]])
    elif not frame.empty:
        # an older engine: level the ledger the way level_ledger does (no corroboration)
        full = led.assign(__file=label)
        halogen = LL.detect_reagent_halogen(frame)
        measured = LL.measure_source(label, full, halogen)
        if not measured.empty:
            levels = LL.assign_levels(measured, set())
    mz = pd.to_numeric(col(m0, "mz"), errors="coerce")
    return {
        "m0": int(len(m0)),
        "assigned": int((tier == "Assigned").sum()),
        # the mass-defect gap a 0.35 Da shift lands in closes above ~m/z 350,
        # so a decoy's Assigned rows are reported on either side of it
        "assigned_lt_350": int(((tier == "Assigned") & (mz < DECOY_MZ_SPLIT)).sum()),
        "assigned_ge_350": int(((tier == "Assigned") & (mz >= DECOY_MZ_SPLIT)).sum()),
        "candidate": int((tier == "Candidate").sum()),
        "neutrals": int(col(m0, "neutral_formula").dropna().nunique()),
        "levels": level_vector(levels),
        "examples": [
            f"{r.neutral_formula} {r.adduct} @ {float(r.mz):.4f}"
            for r in m0[tier == "Assigned"].sort_values("height", ascending=False).head(6).itertuples()
        ] if not m0.empty else [],
    }


def decoy(run: Run, mode: str, offset_da: float, n_files: int, log=lambda *a: None) -> dict:
    """The decoy false-discovery bound: the engine, offline, on the brightest
    cover file(s) as they are (the control), with every m/z shifted, and with
    the wrong adduct set. What is still Assigned is the error bound."""
    if mode == "none" or run.per_file.empty:
        return {"mode": mode, "files": [], "control": None, "shift": None, "adducts": None}
    files = brightest_files(run, n_files)
    out = {"mode": mode, "offset_da": offset_da, "files": files, "control": None, "shift": None, "adducts": None,
           "adducts_used": run.adducts, "wrong_adducts": wrong_adducts(run.polarity)}
    agg: dict[str, list] = {"control": [], "shift": [], "adducts": []}
    errors: dict[str, str] = {}

    def arm(key: str, peaks: pd.DataFrame, sample_id: str, adducts: list[str]) -> None:
        # a decoy arm that crashes the engine is a finding, not a reason to lose
        # the card: record it and carry on
        try:
            agg[key].append(_ledger_counts(run_engine_offline(run, peaks, sample_id, adducts, log), sample_id))
        except Exception as exc:  # noqa: BLE001 - anything the engine raises
            errors[key] = f"{type(exc).__name__}: {exc}"
            log(f"[decoy] {sample_id}: {key} arm failed -- {errors[key]}")

    for f in files:
        peaks = raw_peaks_of(run, f)
        if peaks.empty:
            continue
        log(f"[decoy] {f}: {len(peaks)} peaks, control run")
        arm("control", peaks, f"{f}-control", run.adducts)
        if mode in ("shift", "both"):
            log(f"[decoy] {f}: shift {offset_da:+.3f} Da")
            arm("shift", decoy_peaks(peaks, offset_da), f"{f}-shift", run.adducts)
        if mode in ("adducts", "both"):
            log(f"[decoy] {f}: wrong adducts {wrong_adducts(run.polarity)}")
            arm("adducts", peaks, f"{f}-adducts", wrong_adducts(run.polarity))
    for key, err in errors.items():
        out[key] = {"error": err}
    for key, items in agg.items():
        if not items:
            continue
        total = {k: int(sum(i[k] for i in items)) for k in ("m0", "assigned", "assigned_lt_350", "assigned_ge_350", "candidate", "neutrals")}
        total["levels"] = {lv: int(sum(i["levels"].get(lv, 0) for i in items)) for lv in LEVELS}
        total["examples"] = items[0]["examples"]
        out[key] = total
    ctrl = out["control"] if out["control"] and "error" not in out["control"] else None
    for key in ("shift", "adducts"):
        if out[key] and ctrl and "error" not in out[key]:
            out[key]["assigned_rate"] = pct(out[key]["assigned"], ctrl["assigned"])
            out[key]["candidate_rate"] = pct(out[key]["candidate"], ctrl["candidate"])
            out[key]["good_level_rate"] = pct(
                sum(out[key]["levels"].get(lv, 0) for lv in GOOD_LEVELS),
                sum(ctrl["levels"].get(lv, 0) for lv in GOOD_LEVELS),
            )
    return out


# --- falsification survival -------------------------------------------------
HETERO_LINES = {"S": "34S", "Cl": "37Cl", "Br": "81Br", "Si": "29Si"}


def falsification(run: Run) -> dict:
    """(d) the 13C-implied carbon count against the formula on every Assigned
    row with a measured 13C satellite; the heteroatom line where the formula
    demands one; and the time covariance of satellites and adduct pairs with
    their parent."""
    pf = run.per_file
    out = {"c13": None, "hetero": None, "iso_cov": None, "adduct_cov": None}
    if not pf.empty and "isotopologues" in pf.columns and "role" in pf.columns:
        rows, hetero = [], {"n": 0, "present": 0, "missing": []}
        for fid, frame in pf.groupby("__file"):
            heights = frame.set_index("peak_id")["height"].to_dict()
            m0 = frame[(frame["role"] == "M0") & (frame["tier"] == "Assigned")]
            # the satellites a file's picker claimed live on iso_child rows keyed by
            # parent_peak_id (an iso-pair or series commit leaves the parent's own
            # `isotopologues` cell empty), so both sources are read
            kids = frame[frame["role"] == "iso_child"] if "parent_peak_id" in frame.columns else frame.iloc[0:0]
            kid_labels = kids.groupby("parent_peak_id")["iso_label"].agg(lambda s: {str(v) for v in s if is_str(v)}).to_dict() if not kids.empty else {}
            kid_13c = kids[kids["iso_label"].astype(str) == "13C"].drop_duplicates("parent_peak_id").set_index("parent_peak_id")["peak_id"].to_dict() if not kids.empty else {}
            for r in m0.itertuples():
                isos = as_list(r.isotopologues)
                labels = {str(i.get("label", "")).split("+")[0] for i in isos if isinstance(i, dict)} | kid_labels.get(r.peak_id, set())
                n_c = elem(r.ion_formula, "C")
                sat = next((i for i in isos if isinstance(i, dict) and str(i.get("label", "")) == "13C"), None)
                sat_pid = sat.get("peak_id") if sat else kid_13c.get(r.peak_id)
                if sat_pid is not None and n_c and pd.notna(r.height) and r.height > 0 and sat_pid in heights:
                    implied = heights[sat_pid] / float(r.height) / LL.C13_PER_CARBON
                    rows.append({"neutral": r.neutral_formula, "adduct": r.adduct, "mz": float(r.mz), "height": float(r.height),
                                 "c_formula": n_c, "c_implied": float(implied), "delta": float(implied - n_c)})
                for e, line in HETERO_LINES.items():
                    if elem(r.ion_formula, e) > 0:
                        hetero["n"] += 1
                        # '81Br', '81Br(pair)', '2x37Cl', '37Cl2' all carry the line
                        present = any(line in l for l in labels) or (e == "Si" and any("30Si" in l for l in labels))
                        if present:
                            hetero["present"] += 1
                        elif len(hetero["missing"]) < 12:
                            hetero["missing"].append(f"{r.neutral_formula} {r.adduct} ({line})")
        if rows:
            df = pd.DataFrame(rows)
            tol = np.maximum(1.0, 0.25 * df["c_formula"])
            out["c13"] = {
                "n": int(len(df)),
                "within_1": int((df["delta"].abs() <= 1.0).sum()),
                "within_tol": int((df["delta"].abs() <= tol).sum()),
                "median_abs_delta": float(df["delta"].abs().median()),
                "worst": [
                    f"{r.neutral} {r.adduct}: C{r.c_formula} vs ~C{r.c_implied:.1f}"
                    for r in df[df["delta"].abs() > tol].sort_values("height", ascending=False).head(8).itertuples()
                ],
            }
        if hetero["n"]:
            out["hetero"] = hetero
    ts = run.ts
    if ts is not None and "ion_formula" in ts.columns and "role" in ts.columns:
        st = ts[ts["ion_formula"].notna() & ~col(ts, "dup_candidate", False).fillna(False).astype(bool)]
        wide = st.pivot_table(index="sample_item_id", columns="ion_mz", values="height", aggfunc="max")
        meta = st.drop_duplicates("ion_mz").set_index("ion_mz")
        parents = meta[meta["role"] == "M0"]
        by_ion = {}
        for mz, r in parents.iterrows():
            by_ion.setdefault(str(r["ion_formula"]), []).append(mz)
        rs = []
        for mz, r in meta[meta["role"] == "iso_child"].iterrows():
            for p in by_ion.get(str(r["ion_formula"]), []):
                if p in wide.columns and mz in wide.columns:
                    pair = wide[[p, mz]].dropna()
                    if len(pair) >= 20 and pair.iloc[:, 0].std() > 0 and pair.iloc[:, 1].std() > 0:
                        rs.append(float(np.corrcoef(pair.iloc[:, 0], pair.iloc[:, 1])[0, 1]))
                    break
        if rs:
            arr = np.array(rs)
            out["iso_cov"] = {"n": int(len(arr)), "median_r": float(np.median(arr)), "share_r_ge_0_8": pct((arr >= 0.8).sum(), len(arr))}
        rs = []
        chan = parents.reset_index().groupby("neutral_formula")["ion_mz"].apply(list)
        for neutral, mzs in chan.items():
            if len(mzs) < 2:
                continue
            a, b = mzs[0], mzs[1]
            if a in wide.columns and b in wide.columns:
                pair = wide[[a, b]].dropna()
                if len(pair) >= 20 and pair.iloc[:, 0].std() > 0 and pair.iloc[:, 1].std() > 0:
                    rs.append(float(np.corrcoef(pair.iloc[:, 0], pair.iloc[:, 1])[0, 1]))
        if rs:
            arr = np.array(rs)
            out["adduct_cov"] = {"n": int(len(arr)), "median_r": float(np.median(arr)), "share_r_ge_0_6": pct((arr >= 0.6).sum(), len(arr))}
    return out


# ---------------------------------------------------------------------------
# 6. delta against the previous row of the same channel
# ---------------------------------------------------------------------------
KEY_METRICS = [
    ("assigned", "Assigned rows", 0),
    ("candidate", "Candidate rows", 0),
    ("neutrals", "distinct neutrals", 0),
    ("stamped_signal_share", "stamped signal %", 1),
    ("unstamped_merged", "merged rows unstamped", 0),
    ("bright_m0_not_assigned", "bright M0 not Assigned", 0),
    ("bright_unstamped", "unstamped in brightest 50", 0),
    ("good_levels", "rows at level <= 4a", 0),
    ("m1_families", "M1 families", 0),
    ("m1_signal_share", "M1 unstamped signal %", 1),
    ("roster_present", "roster present", 0),
    ("roster_assigned", "roster Assigned", 0),
    ("roster_misread", "roster read as other", 0),
    ("decoy_shift_rate", "decoy (shift) Assigned %", 1),
    ("decoy_adducts_rate", "decoy (adducts) Assigned %", 1),
    ("c13_within_1", "13C carbon count within 1", 0),
    ("hetero_present", "heteroatom line present", 0),
    ("census_halogen", "Assigned with Cl/Br/F", 0),
]


def read_board(path: str) -> list[dict]:
    if not os.path.isfile(path):
        return []
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    continue
    return rows


def previous_row(board: list[dict], channel: str, before: dict | None = None) -> dict | None:
    """The channel's last row -- before `before` when that row is itself on the
    board (the board render), else the last one there is (a card being built)."""
    rows = list(board)
    if before is not None and before in rows:
        rows = rows[: rows.index(before)]
    rows = [r for r in rows if r.get("channel") == channel]
    return rows[-1] if rows else None


def delta(row: dict, prev: dict | None) -> list[dict]:
    out = []
    for key, label, nd in KEY_METRICS:
        now = row.get(key)
        was = prev.get(key) if prev else None
        d = None
        if isinstance(now, (int, float)) and isinstance(was, (int, float)) and np.isfinite(now) and np.isfinite(was):
            d = now - was
        out.append({"metric": label, "prev": was, "now": now, "delta": d, "nd": nd})
    return out


# ---------------------------------------------------------------------------
# the card: every panel, one dict
# ---------------------------------------------------------------------------
def build_card(run: Run, *, levels_csv=None, other=None, other_instrument=None, rosters=None,
               decoy_mode="none", decoy_offset=0.35, decoy_files=1, overlap=None, masks=(),
               floor_cps=10.0, floor_share=0.8, board=None, log=print) -> dict:
    log(f"[scorecard] {run.name}: {run.reagent} {run.path_kind}, {run.n_spectra} spectra")
    ions = ion_table(run)
    tracks = unstamped_tracks(run)
    corroborate = [other_instrument.path] if other_instrument is not None else []
    levels = levels_for(run, levels_csv, corroborate)
    other_levels = None
    if other_instrument is not None:
        other_levels = levels_for(other_instrument, None, [run.path])
    rosters = load_rosters() if rosters is None else rosters
    head = headline(run, ions)
    card = {
        "scorecard_version": __version__,
        "written_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "run_dir": run.path,
        "headline": head,
        "brightest": brightest(run, ions, tracks, levels),
        "evidence": best_evidence(run, ions, levels),
        "m1": missed_m1(run, ions, tracks, coverage_rows=head["stamp_coverage"].get("rows")),
        "m2": missed_m2(run, ions, tracks, rosters),
        "m3": missed_m3(run, other, other_instrument, other_levels, overlap, list(masks), floor_cps, floor_share),
        "census": census(run),
        "decoy": decoy(run, decoy_mode, decoy_offset, decoy_files, log=log),
        "falsification": falsification(run),
        "rosters_unreviewed": True,
    }
    card["row"] = board_row(card)
    prev = previous_row(board or [], card["row"]["channel"])
    card["delta"] = delta(card["row"], prev)
    card["previous"] = {"run": prev.get("run"), "written_utc": prev.get("written_utc")} if prev else None
    return card


def board_row(card: dict) -> dict:
    """The one-line summary appended to scoreboard.jsonl."""
    h, b, e, m1, m2, cz, dc, fz = (card[k] for k in ("headline", "brightest", "evidence", "m1", "m2", "census", "decoy", "falsification"))
    roster = {}
    for name, r in (m2.get("roster") or {}).items():
        for k, v in r.items():
            roster[k] = roster.get(k, 0) + v
    row = {
        "written_utc": card["written_utc"],
        "channel": h["channel"],
        "run": h["run"],
        "run_dir": card["run_dir"],
        "code": h["code"],
        "reagent": h["reagent"],
        "path": h["path"],
        "n_spectra": h["n_spectra"],
        "n_peaks": h["n_peaks"],
        "n_files": h["n_files"],
        "stamped_peak_share": h["stamped_peak_share"],
        "stamped_signal_share": h["stamped_signal_share"],
        "merged_rows": h["merged_rows"],
        "assigned": h["assigned"],
        "candidate": h["candidate"],
        "neutrals": h["neutrals"],
        "unstamped_merged": h["stamp_coverage"].get("n_unstamped"),
        "bright_m0_not_assigned": b["m0_not_assigned"],
        "bright_unstamped": b["unstamped_in_top"],
        "levels": e["levels"],
        "good_levels": sum(e["levels"].get(k, 0) for k in GOOD_LEVELS),
        "axes_hist": e["axes_hist"],
        "m1_tracks": m1["n_tracks"],
        "m1_families": m1.get("n_families", 0),
        "m1_signal_share": m1["signal_share"],
        "roster_n": roster.get("n"),
        "roster_present": roster.get("present"),
        "roster_assigned": roster.get("assigned"),
        "roster_misread": roster.get("read_as"),
        "roster_unstamped": roster.get("unstamped"),
        "census": cz["elements"],
        "census_halogen": sum(cz["elements"].get(e_, 0) for e_ in ("F", "Cl", "Br")),
        "decoy_mode": dc.get("mode"),
        "decoy_shift_rate": (dc.get("shift") or {}).get("assigned_rate"),
        "decoy_adducts_rate": (dc.get("adducts") or {}).get("assigned_rate"),
        "decoy_control_assigned": (dc.get("control") or {}).get("assigned"),
        "c13_n": (fz.get("c13") or {}).get("n"),
        "c13_within_1": (fz.get("c13") or {}).get("within_1"),
        "hetero_n": (fz.get("hetero") or {}).get("n"),
        "hetero_present": (fz.get("hetero") or {}).get("present"),
        "iso_cov_median_r": (fz.get("iso_cov") or {}).get("median_r"),
        "adduct_cov_median_r": (fz.get("adduct_cov") or {}).get("median_r"),
        "m3_other_path_missing": ((card["m3"].get("other_path") or {}).get("n_missing")),
        "m3_other_instrument_missing": ((card["m3"].get("other_instrument") or {}).get("n_missing")),
    }
    return row


# ---------------------------------------------------------------------------
# rendering: SCORECARD.md
# ---------------------------------------------------------------------------
def _p(v, nd=1) -> str:
    return "" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{nd}f}"


def render_md(card: dict) -> str:
    h, b, e, m1, m2, m3, cz, dc, fz = (card[k] for k in ("headline", "brightest", "evidence", "m1", "m2", "m3", "census", "decoy", "falsification"))
    sc = h["stamp_coverage"]
    L = [f"# Scorecard — {h['run']}", "",
         f"channel `{h['channel']}` · reagent `{h['reagent']}` · path {h['path']} · code {h['code'] or '?'} · written {card['written_utc']}", ""]
    # 1 headline
    L += ["## 1. What was assigned", "",
          f"- spectra {fmt(h['n_spectra'])}, peaks {fmt(h['n_peaks'])}, assigned files {fmt(h['n_files'])}"
          + (f", {h['elapsed_s'] / 60:.1f} min" if h.get("elapsed_s") else ""),
          f"- stamped: {_p(h['stamped_peak_share'])} % of peaks, {_p(h['stamped_signal_share'])} % of signal "
          f"(M0 alone {_p(h['stamped_M0_signal_share'])} %); unstamped signal {_p(h['unstamped_signal_share'])} %",
          f"- merged rows {fmt(h['merged_rows'])}: Assigned {fmt(h['assigned'])}, Candidate {fmt(h['candidate'])}; "
          f"distinct neutrals {fmt(h['neutrals'])} ({fmt(h['neutrals_assigned'])} Assigned)",
          "- by stage: " + ", ".join(f"{k} {v}" for k, v in h["by_stage"].items()),
          "- by channel (Assigned / all): " + ", ".join(f"{k or '?'} {h['by_adduct_assigned'].get(k, 0)}/{v}" for k, v in h["by_adduct"].items()),
          f"- file vote: in all files {fmt(h['n_in_all_files'])}, single-file {fmt(h['n_single_file'])}, ion disagreements {fmt(h['ion_disagreements'])}",
          ""]
    L += [f"**Stamp coverage:** {fmt(sc.get('n_unstamped'))} of {fmt(sc['n_rows'])} merged rows have no stamped peak in the time series"
          + (f" ({sc['n_collapsed']} collapsed rows excluded; the merge's reagent-N re-read relabelled {sc['reagent_n_relabeled']})" if sc.get("reagent_n_relabeled") is not None else "")
          + ". A merged row the series never carries is a stamping defect, not a chemistry question.", ""]
    if sc.get("rows"):
        L += md_table(sc["rows"][:15], [("mz", "m/z"), ("neutral", "neutral"), ("adduct", "adduct"), ("tier", "tier"), ("n_files", "files"), ("alternatives", "alternatives"), ("tier_reason", "tier reason")], {"mz": 4})
        L += [""]
    # 2 brightest
    L += [f"## 2. The brightest {b['n']} ions (median height over the batch)", "",
          f"roles: " + ", ".join(f"{k} {v}" for k, v in b.get("by_role", {}).items())
          + f" · **bright M0 not Assigned: {b['m0_not_assigned']}** · unstamped tracks among the batch's brightest {TOP_N}: **{b['unstamped_in_top']}** (named in M1)", ""]
    L += md_table(b["rows"], [("ion_mz", "ion m/z"), ("med_cps", "med cps"), ("in", "in"), ("role", "role"), ("ion", "ion"), ("neutral", "neutral"), ("adduct", "adduct"), ("tier", "tier"), ("level", "level"), ("axes", "axes")], {"ion_mz": 4})
    L += [""]
    # 3 best evidence
    vec = "/".join(str(e["levels"].get(k, 0)) for k in LEVELS)
    L += [f"## 3. The best-evidence {len(e['rows'])} rows", "",
          f"levels ({'/'.join(LEVELS)}): **{vec}** over {fmt(e.get('n_levelled', 0))} rows; at level <= 4a: {e['n_good']}; "
          "axes histogram: " + ", ".join(f"{k} axes {v}" for k, v in e["axes_hist"].items()), ""]
    L += md_table(e["rows"], [("neutral", "neutral"), ("adduct", "adduct"), ("ion_mz", "ion m/z"), ("med_cps", "med cps"), ("in", "in"), ("tier", "tier"), ("level", "level"), ("axes", "axes")], {"ion_mz": 4})
    L += [""]
    # 4 missed
    L += ["## 4. What was missed", "",
          f"### M1 · unstamped tracks present in >= {int(PRESENCE * 100)} % of spectra: {fmt(m1['n_tracks'])} of {fmt(m1.get('n_tracks_total', 0))} tracks, "
          f"{_p(m1['signal_share'])} % of the batch's signal, {m1.get('n_families', 0)} families", "",
          "Families group tracks by their shift from the nearest parent, so a comb is one row. `unknown` = no parent relation found.", ""]
    L += md_table(m1["families"], [("family", "family"), ("n_tracks", "tracks"), ("signal_share", "signal %"), ("max_cps", "max med cps"), ("examples", "brightest examples (parent)")], {"signal_share": 2})
    L += ["", "The brightest 25 tracks, one by one:", ""]
    L += md_table(m1["rows"], [("mz", "m/z"), ("med_cps", "med cps"), ("in", "in"), ("nearest_stamped", "nearest stamped"), ("d_mda", "d mDa"), ("d_ppm", "d ppm"), ("family", "family"), ("tag", "residual tag"), ("c_count", "13C carbons"), ("reason", "engine's reason")], {"mz": 4, "d_mda": 1, "d_ppm": 1})
    L += [""]
    L += ["### M2 · expected but not assigned", ""]
    for name, r in (m2.get("roster") or {}).items():
        L.append(f"- roster `{name}` ({r['n']} formulas): present {r['present']}, Assigned as itself {r['assigned']}, Candidate {r['candidate']}, "
                 f"read as something else {r['read_as']}, present but unstamped {r['unstamped']}, absent {r['absent']}")
        for cls, rc in (m2.get("roster_by_class", {}).get(name) or {}).items():
            L.append(f"  - {cls}: {rc['assigned']} Assigned / {rc['present']} present / {rc['n']}")
    for src, r in (m2.get("sources") or {}).items():
        L.append(f"- `{src}` ({r['n']}): Assigned {r['assigned']}, Candidate {r['candidate']}, read as other {r['read_as']}, unstamped {r['unstamped']}, absent {r['absent']}")
    L += ["", "Rows that are present but NOT assigned as themselves (the misses):", ""]
    miss_rows = [r for r in m2["rows"] if r["status"] in ("read as", "unstamped", "candidate")]
    L += md_table(miss_rows[:40], [("source", "source"), ("name", "name"), ("neutral", "neutral"), ("status", "status"), ("adduct", "channel"), ("mz", "m/z"), ("cps", "med cps"), ("in", "in"), ("read", "read as"), ("reason", "engine's reason"), ("ledger", "in ledger")], {"mz": 4})
    L += [""]
    L += ["### M3 · found elsewhere", ""]
    op = m3.get("other_path")
    if op:
        L += [f"- other path `{op['run']}` ({op['path']}): {op['n_missing']} of its {op['n_solid']} solid neutrals (>= 2 files or Assigned) are absent here", ""]
        L += md_table(op["rows"], [("neutral", "neutral"), ("adduct", "adduct"), ("mz", "m/z"), ("tier", "tier"), ("n_files", "files")], {"mz": 4})
        L += [""]
    oi = m3.get("other_instrument")
    if oi:
        L += [f"- other instrument `{oi['run']}` ({oi['reagent']}): {oi['n_missing']} of its {oi['n_good']} rows at level <= 4b pass the floor "
              f"({oi['floor']['cps']} cps median in {int(oi['floor']['share'] * 100)} % of {oi['n_spectra_in_window']} spectra"
              + (f", window {oi['window'][0][:16]} -> {oi['window'][1][:16]}" if oi.get("window") else "") + ") and are absent here", ""]
        L += md_table(oi["rows"], [("neutral", "neutral"), ("adduct", "other adduct"), ("level", "level"), ("axes", "axes"), ("med_cps", "med cps there"), ("share", "share %")], {"share": 0})
        L += [""]
    if not op and not oi:
        L += ["*(no --other / --other-instrument given)*", ""]
    # 5 is it right
    L += ["## 5. Is it right", "", "### (a) roster recall — see M2 above (rosters are UNREVIEWED; the user signs them off before the numbers are quoted)", "",
          f"### (b) element census of the {fmt(cz['n_assigned'])} Assigned rows", "",
          "| element | Assigned rows | examples |", "|---|---:|---|"]
    for k, v in cz["elements"].items():
        L.append(f"| {k} | {v} | {cz['examples'].get(k, '')} |")
    L += ["", "### (c) decoy false-discovery bound", ""]
    if dc.get("control") and "error" not in dc["control"]:
        L += [f"offline engine on the brightest {len(dc['files'])} cover file(s) `{', '.join(dc['files'])}`; control = the file as it is.", "",
              f"| arm | M0 rows | Assigned | < {DECOY_MZ_SPLIT:.0f} / >= | Candidate | neutrals | level <= 4a | rate vs control (Assigned) |", "|---|---:|---:|---|---:|---:|---:|---:|"]
        for key, label in (("control", "control (as is)"), ("shift", f"shift {dc.get('offset_da', 0):+.2f} Da"), ("adducts", f"wrong adducts {dc.get('wrong_adducts')}")):
            a = dc.get(key)
            if a and "error" in a:
                L.append(f"| {label} | engine error: {a['error']} | | | | | | |")
            elif a:
                good = sum(a["levels"].get(lv, 0) for lv in GOOD_LEVELS)
                L.append(f"| {label} | {a['m0']} | {a['assigned']} | {a.get('assigned_lt_350', '')} / {a.get('assigned_ge_350', '')} | {a['candidate']} | {a['neutrals']} | {good} | {_p(a.get('assigned_rate'))} |")
        for key in ("shift", "adducts"):
            a = dc.get(key)
            if a and a.get("examples"):
                L.append(f"\n{key} arm, brightest decoy Assigned: " + "; ".join(a["examples"]))
    else:
        L += [f"*(decoy mode `{dc.get('mode')}`: not run)*"]
    L += ["", "### (d) falsification survival", ""]
    c13 = fz.get("c13")
    if c13:
        L.append(f"- 13C carbon count on {c13['n']} Assigned rows with a measured satellite: within 1 carbon {c13['within_1']}, within max(1, 25 %) {c13['within_tol']}, median |delta| {c13['median_abs_delta']:.2f}"
                 + (f"; worst: {'; '.join(c13['worst'])}" if c13["worst"] else ""))
    het = fz.get("hetero")
    if het:
        L.append(f"- heteroatom line (34S/37Cl/81Br/29Si) where the formula demands one: present {het['present']} of {het['n']}"
                 + (f"; missing: {'; '.join(het['missing'])}" if het["missing"] else ""))
    ic = fz.get("iso_cov")
    if ic:
        L.append(f"- satellite-parent time covariance over {ic['n']} pairs (>= 20 shared samples): median r {ic['median_r']:.2f}, r >= 0.8 in {ic['share_r_ge_0_8']:.0f} %")
    ac = fz.get("adduct_cov")
    if ac:
        L.append(f"- adduct-pair time covariance over {ac['n']} neutrals on two channels: median r {ac['median_r']:.2f}, r >= 0.6 in {ac['share_r_ge_0_6']:.0f} %")
    if not any((c13, het, ic, ac)):
        L.append("*(nothing measurable)*")
    # 6 delta
    L += ["", "## 6. Delta against the previous row of this channel", ""]
    if card.get("previous"):
        L += [f"previous: `{card['previous']['run']}` written {card['previous']['written_utc']}", ""]
        L += md_table([{"metric": d["metric"], "prev": d["prev"], "now": d["now"], "delta": d["delta"]} for d in card["delta"]],
                      [("metric", "metric"), ("prev", "previous"), ("now", "now"), ("delta", "delta")], {"prev": 1, "now": 1, "delta": 1})
    else:
        L += ["*(first row for this channel)*"]
    L += [""]
    return "\n".join(L)


# ---------------------------------------------------------------------------
# rendering: SCOREBOARD.md over every channel's latest row
# ---------------------------------------------------------------------------
def latest_rows(board: list[dict]) -> list[dict]:
    seen: dict[str, dict] = {}
    for r in board:
        seen[r.get("channel", "?")] = r
    return list(seen.values())


def render_board_md(board: list[dict]) -> str:
    rows = latest_rows(board)
    L = ["# Peaky Scoreboard", "", f"regenerated {dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} · {len(rows)} channels · {len(board)} rows in `scoreboard.jsonl`", "",
         "One row per channel, its latest scorecard, and the delta to the row before it. Levels are 2b/3a/3b/4a/4b/4c/4d/5a/5b.", ""]
    L += ["| channel | run | code | Assigned | Candidate | neutrals | stamped signal % | unstamped merged | bright M0 not Assigned | unstamped in top 50 | levels | <= 4a | M1 fam. | M1 signal % | roster A/present/n | Cl+Br+F | decoy shift % | decoy adducts % | 13C ok/n | hetero ok/n |",
          "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---|---:|---:|---:|---|---|"]
    for r in rows:
        prev = previous_row(board, r["channel"], before=r)

        def cell(key, nd=0):
            v = r.get(key)
            s = fmt(v, nd) if isinstance(v, (int, float)) else (v or "")
            if prev is not None and isinstance(v, (int, float)) and isinstance(prev.get(key), (int, float)):
                d = v - prev[key]
                if abs(d) >= (0.05 if nd else 1):
                    s += f" ({d:+.{nd}f})"
            return s

        lv = r.get("levels") or {}
        L.append("| " + " | ".join([
            r["channel"].replace("|", " · "), r["run"][-24:], r.get("code", ""), cell("assigned"), cell("candidate"), cell("neutrals"), cell("stamped_signal_share", 1),
            cell("unstamped_merged"), cell("bright_m0_not_assigned"), cell("bright_unstamped"),
            "/".join(str(lv.get(k, 0)) for k in LEVELS), cell("good_levels"), cell("m1_families"), cell("m1_signal_share", 1),
            f"{r.get('roster_assigned', '')}/{r.get('roster_present', '')}/{r.get('roster_n', '')}", cell("census_halogen"),
            cell("decoy_shift_rate", 1), cell("decoy_adducts_rate", 1),
            f"{r.get('c13_within_1', '')}/{r.get('c13_n', '')}", f"{r.get('hetero_present', '')}/{r.get('hetero_n', '')}",
        ]) + " |")
    L += ["", "Per-run cards: `<run name>/SCORECARD.md` beside this file. The page `scoreboard.html` is the same data.", ""]
    return "\n".join(L)


# ---------------------------------------------------------------------------
# rendering: scoreboard.html — the "Peaky Scoreboard" page
# ---------------------------------------------------------------------------
GOOD_DIRECTION = {  # +1: up is good, -1: down is good; absent = neutral
    "bright_m0_not_assigned": -1, "unstamped_merged": -1, "bright_unstamped": -1, "m1_signal_share": -1,
    "decoy_shift_rate": -1, "decoy_adducts_rate": -1, "roster_assigned": 1, "roster_present": 1,
    "roster_misread": -1, "good_levels": 1, "c13_within_1": 1, "hetero_present": 1,
}
METRIC_KEYS = {label: key for key, label, _ in KEY_METRICS}

PAGE_CSS = """
:root{--bg:#FAFAF7;--surface:#F5F5F2;--rule:#E6E7E2;--rule-strong:#C9CAC5;--body:#5C5D5A;--head:#161718;
--accent:#B84500;--good:#1F7A46;--warn:#8A6200;--bad:#B0281A;--info:#0B7E8E;--chip:#F5F5F2;--mono:"IBM Plex Mono",ui-monospace,SFMono-Regular,Menlo,monospace;
--sans:"IBM Plex Sans",system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#161718;--surface:#161718;--rule:#5C5D5A;--rule-strong:#B8BAB5;
--body:#B8BAB5;--head:#F1F1ED;--accent:#FF6700;--good:#6CC08B;--warn:#E0B84A;--bad:#F08A7C;--info:#5AC8D8;--chip:#26282A}}
:root[data-theme="dark"]{--bg:#161718;--surface:#161718;--rule:#5C5D5A;--rule-strong:#B8BAB5;--body:#B8BAB5;--head:#F1F1ED;
--accent:#FF6700;--good:#6CC08B;--warn:#E0B84A;--bad:#F08A7C;--info:#5AC8D8;--chip:#26282A}
body{background:var(--bg);color:var(--body);font-family:var(--sans);font-size:14px;line-height:1.5;margin:0}
main{max-width:1280px;margin:0 auto;padding:24px 24px 64px}
h1,h2,h3{color:var(--head);font-weight:600;text-wrap:balance;margin:0}
h1{font-size:28px;letter-spacing:-.01em}h2{font-size:18px;margin-top:40px}h3{font-size:15px;margin-top:24px}
.eyebrow,.mono,th,.tile b,.tabs button,.chip{font-family:var(--mono);font-variant-numeric:tabular-nums}
.eyebrow{text-transform:uppercase;letter-spacing:.1em;font-size:11px;color:var(--accent)}
header{display:flex;flex-direction:column;gap:8px;padding-bottom:16px;border-bottom:1px solid var(--rule-strong)}
header p{margin:0;max-width:65ch}
.meta{font-family:var(--mono);font-size:12px;color:var(--body)}
.scroll{overflow-x:auto;border:1px solid var(--rule);background:var(--surface)}
table{border-collapse:collapse;width:100%;font-size:13px}
th{text-align:left;text-transform:uppercase;letter-spacing:.08em;font-size:10.5px;font-weight:500;color:var(--body);
border-bottom:1px solid var(--rule-strong);padding:8px 10px;white-space:nowrap;position:sticky;top:0;background:var(--surface)}
td{padding:6px 10px;border-bottom:1px solid var(--rule);vertical-align:top;font-variant-numeric:tabular-nums}
td.n{text-align:right;font-family:var(--mono);white-space:nowrap}td.f{font-family:var(--mono);white-space:nowrap}
tr:last-child td{border-bottom:0}
.tiles{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:16px;margin-top:16px}
.tile{background:var(--surface);border:1px solid var(--rule);padding:12px 14px;display:flex;flex-direction:column;gap:4px}
.tile span{font-size:11px;text-transform:uppercase;letter-spacing:.08em;font-family:var(--mono);color:var(--body)}
.tile b{font-size:22px;font-weight:500;color:var(--head)}
.tile small{font-family:var(--mono);font-size:11px}
.d-good{color:var(--good)}.d-bad{color:var(--bad)}.d-flat{color:var(--body)}
.tabs{display:flex;flex-wrap:wrap;gap:8px;margin-top:24px}
.tabs button{background:var(--surface);color:var(--head);border:1px solid var(--rule-strong);padding:6px 12px;font-size:12px;cursor:pointer}
.tabs button[aria-selected="true"]{border-color:var(--accent);color:var(--accent)}
.tabs button:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.chip{display:inline-block;font-size:11px;padding:1px 7px;border:1px solid var(--rule-strong);background:var(--chip);color:var(--head);margin-right:4px}
.chip.good{border-color:var(--good);color:var(--good)}.chip.bad{border-color:var(--bad);color:var(--bad)}.chip.warn{border-color:var(--warn);color:var(--warn)}
.note{max-width:70ch;margin:8px 0 0;font-size:13px}
.two{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:24px}
@media (prefers-reduced-motion: no-preference){.tabs button{transition:border-color .15s}}
"""

PAGE_JS = """
(function(){
  var tabs=document.querySelectorAll('.tabs button');var panels=document.querySelectorAll('.panel');
  function show(id){tabs.forEach(function(b){b.setAttribute('aria-selected',b.dataset.panel===id?'true':'false')});
    panels.forEach(function(p){p.hidden=p.id!==id});try{localStorage.setItem('peaky-scoreboard-tab',id)}catch(e){}}
  tabs.forEach(function(b){b.addEventListener('click',function(){show(b.dataset.panel)})});
  var first=tabs.length?tabs[0].dataset.panel:null;var saved=null;try{saved=localStorage.getItem('peaky-scoreboard-tab')}catch(e){}
  if(saved&&document.getElementById(saved))first=saved;if(first)show(first);
})();
"""


def _h(v) -> str:
    return html.escape("" if v is None else str(v))


def _td(v, nd=0, cls=None) -> str:
    if isinstance(v, bool):
        return f"<td>{'yes' if v else ''}</td>"
    if isinstance(v, (int, float, np.integer, np.floating)):
        return f"<td class=\"n\">{fmt(v, nd)}</td>"
    return f"<td class=\"{cls}\">{_h(v)}</td>" if cls else f"<td>{_h(v)}</td>"


def html_table(rows: list[dict], columns: list[tuple[str, str]], nd: dict | None = None, mono: tuple = ()) -> str:
    nd = nd or {}
    if not rows:
        return "<p class=\"note\">none</p>"
    head = "".join(f"<th>{_h(h)}</th>" for _, h in columns)
    body = []
    for r in rows:
        body.append("<tr>" + "".join(_td(r.get(k), nd.get(k, 0), "f" if k in mono else None) for k, _ in columns) + "</tr>")
    return f"<div class=\"scroll\"><table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table></div>"


def _delta_chip(key: str, d, nd=0) -> str:
    if d is None or (isinstance(d, float) and not np.isfinite(d)):
        return ""
    if abs(d) < (0.05 if nd else 0.5):
        return "<small class=\"d-flat\">no change</small>"
    good = GOOD_DIRECTION.get(key)
    cls = "d-flat" if good is None else ("d-good" if d * good > 0 else "d-bad")
    return f"<small class=\"{cls}\">{d:+.{nd}f} vs previous</small>"


def render_html(cards: list[dict], board: list[dict]) -> str:
    rows = latest_rows(board)
    by_run = {c["headline"]["run"]: c for c in cards}
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    out = ["<title>Peaky Scoreboard</title>",
           "<link rel=\"stylesheet\" href=\"https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap\">",
           f"<style>{PAGE_CSS}</style>", "<main>", "<header>",
           "<div class=\"eyebrow\">peaky · assignment quality, every session</div>", "<h1>Peaky Scoreboard</h1>",
           "<p>One row per channel: what the latest run assigned, how much of the signal it explains, what its evidence is worth, and what it missed. "
           "The delta is against the row before it on the same channel. Rosters are unreviewed until the user signs them off.</p>",
           f"<div class=\"meta\">regenerated {stamp} · {len(rows)} channels · {len(board)} rows</div>", "</header>"]
    # channel table
    out += ["<h2>Channels</h2>"]
    ch_rows = []
    for r in rows:
        prev = previous_row(board, r["channel"], before=r)
        lv = r.get("levels") or {}

        def d(key):
            if prev is None or not isinstance(r.get(key), (int, float)) or not isinstance(prev.get(key), (int, float)):
                return None
            return r[key] - prev[key]

        ch_rows.append({
            "channel": r["channel"], "run": r["run"][-24:], "code": r.get("code", ""), "assigned": r.get("assigned"), "d_assigned": d("assigned"),
            "candidate": r.get("candidate"), "neutrals": r.get("neutrals"), "signal": r.get("stamped_signal_share"),
            "unst": r.get("unstamped_merged"), "bright": r.get("bright_m0_not_assigned"), "bright_un": r.get("bright_unstamped"),
            "levels": "/".join(str(lv.get(k, 0)) for k in LEVELS), "good": r.get("good_levels"),
            "fam": r.get("m1_families"), "m1": r.get("m1_signal_share"),
            "roster": f"{r.get('roster_assigned', '')}/{r.get('roster_present', '')}/{r.get('roster_n', '')}",
            "hal": r.get("census_halogen"), "dshift": r.get("decoy_shift_rate"), "dadd": r.get("decoy_adducts_rate"),
            "c13": f"{r.get('c13_within_1', '')}/{r.get('c13_n', '')}", "het": f"{r.get('hetero_present', '')}/{r.get('hetero_n', '')}",
        })
    out.append(html_table(ch_rows, [("channel", "channel"), ("run", "run"), ("code", "code"), ("assigned", "Assigned"), ("d_assigned", "Δ"), ("candidate", "Candidate"),
                                    ("neutrals", "neutrals"), ("signal", "stamped signal %"), ("unst", "unstamped merged"), ("bright", "bright M0 not Assigned"),
                                    ("bright_un", "unstamped in top 50"), ("levels", "levels 2b…5b"), ("good", "≤ 4a"), ("fam", "M1 families"), ("m1", "M1 signal %"),
                                    ("roster", "roster A/present/n"), ("hal", "Cl+Br+F"), ("dshift", "decoy shift %"), ("dadd", "decoy adducts %"), ("c13", "13C ok/n"), ("het", "hetero ok/n")],
                          {"signal": 1, "m1": 1, "dshift": 1, "dadd": 1, "d_assigned": 0}, mono=("channel", "run", "code", "levels", "roster", "c13", "het")))
    # per-run panels
    panel_rows = [r for r in rows if r["run"] in by_run]
    if panel_rows:
        out.append("<h2>Runs</h2><div class=\"tabs\" role=\"tablist\">")
        for r in panel_rows:
            out.append(f"<button role=\"tab\" data-panel=\"p-{_h(r['run'])}\" aria-selected=\"false\">{_h(r['channel'].split('|')[1] or r['channel'])} · {_h(r['path'])} · {_h(r['run'][-18:])}</button>")
        out.append("</div>")
    for r in panel_rows:
        c = by_run[r["run"]]
        h, b, e, m1, m2, m3, cz, dc, fz = (c[k] for k in ("headline", "brightest", "evidence", "m1", "m2", "m3", "census", "decoy", "falsification"))
        dmap = {d["metric"]: d for d in c.get("delta", [])}

        def tile(label, key, nd=0, value=None):
            v = c["row"].get(key) if value is None else value
            dd = dmap.get(next((lab for k_, lab, _ in KEY_METRICS if k_ == key), ""), {})
            return f"<div class=\"tile\"><span>{_h(label)}</span><b>{fmt(v, nd) if isinstance(v, (int, float)) else _h(v)}</b>{_delta_chip(key, dd.get('delta'), nd)}</div>"

        out.append(f"<section class=\"panel\" id=\"p-{_h(r['run'])}\" hidden>")
        out.append(f"<h3>{_h(r['run'])}</h3><div class=\"meta\">channel {_h(h['channel'])} · reagent {_h(h['reagent'])} · {_h(h['path'])} · code {_h(h['code'])} · {_h(c['written_utc'])}</div>")
        out.append("<div class=\"tiles\">" + "".join([
            tile("spectra", "n_spectra"), tile("Assigned", "assigned"), tile("Candidate", "candidate"), tile("neutrals", "neutrals"),
            tile("stamped signal %", "stamped_signal_share", 1), tile("merged rows unstamped", "unstamped_merged"),
            tile("bright M0 not Assigned", "bright_m0_not_assigned"), tile("unstamped in top 50", "bright_unstamped"),
            tile("rows at level ≤ 4a", "good_levels"), tile("M1 families", "m1_families"), tile("M1 signal %", "m1_signal_share", 1),
            tile("roster Assigned", "roster_assigned"), tile("roster present", "roster_present"), tile("Assigned with Cl/Br/F", "census_halogen"),
            tile("decoy shift Assigned %", "decoy_shift_rate", 1), tile("decoy adducts Assigned %", "decoy_adducts_rate", 1),
        ]) + "</div>")
        out.append(f"<p class=\"note\">levels ({'/'.join(LEVELS)}): <span class=\"mono\">{'/'.join(str(e['levels'].get(k, 0)) for k in LEVELS)}</span> · "
                   f"axes histogram {_h(', '.join(f'{k}: {v}' for k, v in e['axes_hist'].items()))} · by channel (Assigned/all) "
                   f"{_h(', '.join(f'{k or chr(63)} {h['by_adduct_assigned'].get(k, 0)}/{v}' for k, v in h['by_adduct'].items()))}</p>")
        sc = h["stamp_coverage"]
        if sc.get("rows"):
            out.append(f"<h3>Stamp coverage — {sc['n_unstamped']} merged rows the time series never carries</h3>")
            out.append(html_table(sc["rows"][:15], [("mz", "m/z"), ("neutral", "neutral"), ("adduct", "adduct"), ("tier", "tier"), ("n_files", "files"), ("alternatives", "alternatives"), ("tier_reason", "tier reason")], {"mz": 4}, mono=("neutral", "adduct")))
        out.append(f"<h3>Brightest {b['n']} ions — bright M0 not Assigned: {b['m0_not_assigned']}, unstamped tracks among the brightest {TOP_N}: {b['unstamped_in_top']}</h3>")
        out.append(html_table(b["rows"], [("ion_mz", "ion m/z"), ("med_cps", "med cps"), ("in", "in"), ("role", "role"), ("ion", "ion"), ("neutral", "neutral"), ("adduct", "adduct"), ("tier", "tier"), ("level", "level"), ("axes", "axes")], {"ion_mz": 4}, mono=("ion", "neutral", "adduct", "in", "level")))
        out.append(f"<h3>Best evidence — {e['n_good']} rows at level ≤ 4a</h3>")
        out.append(html_table(e["rows"][:25], [("neutral", "neutral"), ("adduct", "adduct"), ("ion_mz", "ion m/z"), ("med_cps", "med cps"), ("in", "in"), ("tier", "tier"), ("level", "level"), ("axes", "axes")], {"ion_mz": 4}, mono=("neutral", "adduct", "in", "level")))
        out.append(f"<h3>M1 — {m1['n_tracks']} unstamped tracks in ≥ {int(PRESENCE * 100)} % of spectra, {_p(m1['signal_share'])} % of the signal, grouped into families</h3>")
        out.append(html_table(m1["families"], [("family", "family"), ("n_tracks", "tracks"), ("signal_share", "signal %"), ("max_cps", "max med cps"), ("examples", "brightest examples (parent)")], {"signal_share": 2}, mono=("examples",)))
        out.append(html_table(m1["rows"], [("mz", "m/z"), ("med_cps", "med cps"), ("in", "in"), ("nearest_stamped", "nearest stamped"), ("d_mda", "Δ mDa"), ("family", "family"), ("tag", "residual tag"), ("c_count", "13C carbons"), ("reason", "engine's reason")], {"mz": 4, "d_mda": 1}, mono=("nearest_stamped", "in")))
        out.append("<h3>M2 — expected but not assigned</h3><div class=\"two\"><div>")
        for name, rr in (m2.get("roster") or {}).items():
            out.append(f"<p class=\"note\"><span class=\"chip\">{_h(name)}</span> {rr['n']} formulas: <span class=\"chip good\">{rr['assigned']} Assigned</span>"
                       f"<span class=\"chip\">{rr['candidate']} Candidate</span><span class=\"chip bad\">{rr['read_as']} read as other</span>"
                       f"<span class=\"chip warn\">{rr['unstamped']} unstamped</span><span class=\"chip\">{rr['absent']} absent</span></p>")
        for src, rr in (m2.get("sources") or {}).items():
            out.append(f"<p class=\"note\"><span class=\"chip\">{_h(src)}</span> {rr['n']}: {rr['assigned']} Assigned, {rr['candidate']} Candidate, {rr['read_as']} read as other, {rr['unstamped']} unstamped, {rr['absent']} absent</p>")
        out.append("</div><div>")
        cz_rows = [{"element": k, "rows": v, "examples": cz["examples"].get(k, "")} for k, v in cz["elements"].items()]
        out.append("<p class=\"note\"><b>Element census of Assigned neutrals</b></p>" + html_table(cz_rows, [("element", "element"), ("rows", "Assigned rows"), ("examples", "examples")], mono=("element", "examples")))
        out.append("</div></div>")
        miss_rows = [x for x in m2["rows"] if x["status"] in ("read as", "unstamped", "candidate")]
        out.append(html_table(miss_rows[:40], [("source", "source"), ("name", "name"), ("neutral", "neutral"), ("status", "status"), ("adduct", "channel"), ("mz", "m/z"), ("cps", "med cps"), ("in", "in"), ("read", "read as"), ("reason", "engine's reason"), ("ledger", "in ledger")], {"mz": 4}, mono=("neutral", "adduct", "read", "in")))
        op, oi = m3.get("other_path"), m3.get("other_instrument")
        if op or oi:
            out.append("<h3>M3 — found elsewhere</h3>")
            if op:
                out.append(f"<p class=\"note\">other path <span class=\"mono\">{_h(op['run'])}</span> ({_h(op['path'])}): {op['n_missing']} of its {op['n_solid']} solid neutrals are absent here</p>")
                out.append(html_table(op["rows"], [("neutral", "neutral"), ("adduct", "adduct"), ("mz", "m/z"), ("tier", "tier"), ("n_files", "files")], {"mz": 4}, mono=("neutral", "adduct")))
            if oi:
                out.append(f"<p class=\"note\">other instrument <span class=\"mono\">{_h(oi['run'])}</span> ({_h(oi['reagent'])}): {oi['n_missing']} of its {oi['n_good']} rows at level ≤ 4b pass the floor and are absent here</p>")
                out.append(html_table(oi["rows"], [("neutral", "neutral"), ("adduct", "other adduct"), ("level", "level"), ("axes", "axes"), ("med_cps", "med cps there"), ("share", "share %")], {"share": 0}, mono=("neutral", "adduct", "level")))
        out.append("<h3>Decoy false-discovery bound</h3>")
        if dc.get("control") and "error" not in dc["control"]:
            dc_rows = []
            for key, label in (("control", "control (as is)"), ("shift", f"shift {dc.get('offset_da', 0):+.2f} Da"), ("adducts", f"wrong adducts {dc.get('wrong_adducts')}")):
                a = dc.get(key)
                if a and "error" in a:
                    dc_rows.append({"arm": label, "examples": f"engine error: {a['error']}"})
                elif a:
                    dc_rows.append({"arm": label, "m0": a["m0"], "assigned": a["assigned"], "split": f"{a.get('assigned_lt_350', '')} / {a.get('assigned_ge_350', '')}", "candidate": a["candidate"], "neutrals": a["neutrals"],
                                    "good": sum(a["levels"].get(lv, 0) for lv in GOOD_LEVELS), "rate": a.get("assigned_rate"), "examples": "; ".join(a.get("examples", [])[:4])})
            out.append(f"<p class=\"note\">offline engine on <span class=\"mono\">{_h(', '.join(dc['files']))}</span></p>" + html_table(dc_rows, [("arm", "arm"), ("m0", "M0 rows"), ("assigned", "Assigned"), ("split", f"< {DECOY_MZ_SPLIT:.0f} / ≥"), ("candidate", "Candidate"), ("neutrals", "neutrals"), ("good", "≤ 4a"), ("rate", "Assigned % of control"), ("examples", "brightest Assigned")], {"rate": 1}, mono=("examples",)))
        else:
            out.append(f"<p class=\"note\">decoy mode <span class=\"mono\">{_h(dc.get('mode'))}</span>: not run</p>")
        out.append("<h3>Falsification survival</h3><ul class=\"note\">")
        c13, het, ic, ac = fz.get("c13"), fz.get("hetero"), fz.get("iso_cov"), fz.get("adduct_cov")
        if c13:
            out.append(f"<li>13C carbon count on {c13['n']} Assigned rows: within 1 carbon {c13['within_1']}, within max(1, 25 %) {c13['within_tol']}, median |Δ| {c13['median_abs_delta']:.2f}" + (f" — worst: <span class=\"mono\">{_h('; '.join(c13['worst']))}</span>" if c13['worst'] else "") + "</li>")
        if het:
            out.append(f"<li>heteroatom line present {het['present']} of {het['n']}" + (f" — missing: <span class=\"mono\">{_h('; '.join(het['missing']))}</span>" if het['missing'] else "") + "</li>")
        if ic:
            out.append(f"<li>satellite–parent time covariance over {ic['n']} pairs: median r {ic['median_r']:.2f}, r ≥ 0.8 in {ic['share_r_ge_0_8']:.0f} %</li>")
        if ac:
            out.append(f"<li>adduct-pair time covariance over {ac['n']} neutrals: median r {ac['median_r']:.2f}, r ≥ 0.6 in {ac['share_r_ge_0_6']:.0f} %</li>")
        if not any((c13, het, ic, ac)):
            out.append("<li>nothing measurable</li>")
        out.append("</ul>")
        if c.get("previous"):
            out.append(f"<h3>Delta vs <span class=\"mono\">{_h(c['previous']['run'])}</span></h3>")
            out.append(html_table([{"metric": d_["metric"], "prev": d_["prev"], "now": d_["now"], "delta": d_["delta"]} for d_ in c["delta"]],
                                  [("metric", "metric"), ("prev", "previous"), ("now", "now"), ("delta", "Δ")], {"prev": 1, "now": 1, "delta": 1}))
        out.append("</section>")
    out += [f"<script>{PAGE_JS}</script>", "</main>"]
    # entities, not raw UTF-8: the page must read the same however it is served
    return "\n".join(out).encode("ascii", "xmlcharrefreplace").decode("ascii")


# ---------------------------------------------------------------------------
# outputs + main
# ---------------------------------------------------------------------------
def write_outputs(cards: list[dict], out_dir: str, board_path: str | None = None, log=print) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    board_path = board_path or os.path.join(out_dir, "scoreboard.jsonl")
    board = read_board(board_path)
    written = {"cards": [], "board": board_path}
    with open(board_path, "a") as fh:
        for card in cards:
            run_dir = os.path.join(out_dir, card["headline"]["run"])
            os.makedirs(run_dir, exist_ok=True)
            md_path = os.path.join(run_dir, "SCORECARD.md")
            with open(md_path, "w") as md:
                md.write(render_md(card))
            with open(os.path.join(run_dir, "scorecard.json"), "w") as js:
                json.dump(card, js, indent=1, default=_json_default)
            fh.write(json.dumps(card["row"], default=_json_default) + "\n")
            board.append(card["row"])
            written["cards"].append(md_path)
            log(f"[scorecard] wrote {md_path}")
    # the page and the board carry every channel's latest card
    latest = latest_rows(board)
    all_cards = {c["headline"]["run"]: c for c in cards}
    for r in latest:
        if r["run"] not in all_cards:
            p = os.path.join(out_dir, r["run"], "scorecard.json")
            if os.path.isfile(p):
                all_cards[r["run"]] = _read_json(p)
    md_board = os.path.join(out_dir, "SCOREBOARD.md")
    with open(md_board, "w") as fh:
        fh.write(render_board_md(board))
    html_path = os.path.join(out_dir, "scoreboard.html")
    with open(html_path, "w") as fh:
        fh.write(render_html([c for c in all_cards.values() if c], board))
    written.update(board_md=md_board, html=html_path)
    log(f"[scorecard] board -> {md_board}, page -> {html_path}")
    return written


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (pd.Timestamp, dt.datetime)):
        return o.isoformat()
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return str(o)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="The scorecard: what a peaky batch run assigned, how good it is, what it missed.")
    ap.add_argument("run_dirs", nargs="+", help="run dirs (or the --out-dir holding exactly one)")
    ap.add_argument("--levels", help="levels CSV from scripts/level_ledger.py --out (default: level in-process)")
    ap.add_argument("--other", help="the other PATH on the same batch (cover vs trace-first)")
    ap.add_argument("--other-instrument", help="the other analyser on the same air (corroborates the levels; feeds M3)")
    ap.add_argument("--overlap", help="UTC window 'start,end' both instruments recorded, e.g. '2026-01-01T06:00,2026-01-03T18:00'")
    ap.add_argument("--mask", action="append", default=[], help="UTC window 'start,end' to exclude (a gap); repeatable")
    ap.add_argument("--floor-cps", type=float, default=10.0, help="other-instrument detection floor: median cps (default 10)")
    ap.add_argument("--floor-share", type=float, default=0.8, help="... present in this share of spectra (default 0.8)")
    ap.add_argument("--decoy", choices=("none", "shift", "adducts", "both"), default="none", help="run the offline engine on decoy input")
    ap.add_argument("--decoy-offset", type=float, default=0.35, help="Da added to every m/z in the shift decoy (default 0.35)")
    ap.add_argument("--decoy-files", type=int, default=1, help="brightest cover files to run the decoy on (default 1)")
    ap.add_argument("--rosters", help="directory of roster CSVs (default: the packaged peaky/data/rosters)")
    ap.add_argument("--out", default=DEFAULT_OUT, help=f"scoreboard directory (default {DEFAULT_OUT})")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    log = (lambda *a, **k: None) if args.quiet else print

    other = load_run(args.other) if args.other else None
    other_instrument = load_run(args.other_instrument) if args.other_instrument else None
    rosters = load_rosters(args.rosters)
    board = read_board(os.path.join(args.out, "scoreboard.jsonl"))
    cards = []
    for path in args.run_dirs:
        run = load_run(path)
        cards.append(build_card(
            run, levels_csv=args.levels, other=other, other_instrument=other_instrument, rosters=rosters,
            decoy_mode=args.decoy, decoy_offset=args.decoy_offset, decoy_files=args.decoy_files,
            overlap=_parse_window(args.overlap), masks=[_parse_window(m) for m in args.mask],
            floor_cps=args.floor_cps, floor_share=args.floor_share, board=board, log=log,
        ))
    write_outputs(cards, args.out, log=log)
    for card in cards:
        r = card["row"]
        lv = r["levels"]
        print(f"{r['run']}: Assigned {r['assigned']} / Candidate {r['candidate']} / neutrals {r['neutrals']}; stamped signal {_p(r['stamped_signal_share'])} %; "
              f"unstamped merged {r['unstamped_merged']}; bright M0 not Assigned {r['bright_m0_not_assigned']}; levels {'/'.join(str(lv.get(k, 0)) for k in LEVELS)}; "
              f"M1 {r['m1_families']} families / {_p(r['m1_signal_share'])} %; roster {r['roster_assigned']}/{r['roster_present']}/{r['roster_n']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
