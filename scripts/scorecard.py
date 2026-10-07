"""The scorecard — what one `peaky batch` run assigned, how good it is, what it missed.

A run answers three questions, and every session that ran the pipeline answers
them the same way so the sessions can be compared:

    what was assigned      spectra, peaks, the stamped share of peaks and of
                           signal; merged rows by tier, stage and channel; the
                           file vote; STAMP COVERAGE (merged rows the time
                           series never carries — a stamping defect, not a
                           chemistry question)
    how good is it         the brightest 50 ions with their reading, tier,
                           evidence level and claim — every bright ion is either
                           explained or a named miss; the best-evidence 50; the
                           level vector; the roster recall; the
                           element census; the DECOY false-discovery bound (the
                           engine run offline on m/z-shifted input, and on the
                           wrong adduct set); falsification survival (the 13C
                           carbon count, the heteroatom lines, time covariance)
    what was missed        M1 the brightest UNSTAMPED tracks, grouped into
                           families by their shift from the nearest parent so a
                           300-line electron-attachment comb is one row; M2 the
                           roster / reference / known species present but not
                           assigned, with the engine's own reason; M3 what the
                           other instrument on the same air (by its merged level
                           and by its own evidence), or the other path on the
                           same batch, found and this run did not

Beside the tier, every committed reading carries the CLAIM its level on the
evidence scale of peaky 0.10.0 supports: identified (3c), neutral (4a), ion
(4b), tentative (5a, 5b, none), and two buckets reported beside them, reagent
and not assessed (NA: the instrument class is not assessed). §0 counts it by
merged row and by committed per-file M0 signal, crosses it with the tier and
lists the rows where the two disagree. The acceptance metrics read the
identified class; every older metric keeps its key beside it. A board row
carries `claims_schema` 2; rows written on the pre-0.10.0 scale (schema 1) or
before the claim (none) still render, and no delta is taken across scales.

    python scripts/scorecard.py <run_dir>... [--levels levels.csv]
        [--other <run_dir>] [--other-instrument <run_dir>]
        [--decoy none|shift|adducts|both] [--decoy-ledgers DIR] [--out DIR]

Per run: `<out>/<run name>/SCORECARD.md` and `scorecard.json` (and the decoy
arms' engine ledgers under `decoy/`), one row appended to
`<out>/scoreboard.jsonl`, and `<out>/SCOREBOARD.md` + `scoreboard.html`
regenerated over every channel's latest row with its delta. The level is the
merged ledger's in-core `evidence_level` when the run wrote the scale's columns;
a run made before the scale is levelled post hoc by `scripts/level_ledger.py`
(in-process, the same fact layer); a decoy arm is levelled in its run's context
exactly as the reference levels it (adapted, and strict in a field of its own).

Never read `tables/residual_bins.csv` for the misses: it lists only the bins
absent from every cover file, and every headline miss of the first cut sat
inside the cover files. M1 is built from the per-file ledgers' `unexplained`
rows and the time series.
"""

from __future__ import annotations

import argparse
import contextlib
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

import level_ledger as LL  # noqa: E402  (scripts/level_ledger.py: the scale's executable reference)
from peaky.assignment import evidence as EV  # noqa: E402
from peaky.chem import chemistry as C  # noqa: E402
from peaky.paths import pkg_data  # noqa: E402

__version__ = "0.1.0"

DEFAULT_OUT = os.path.expanduser("~/peaky-output/scoreboard")
LEVELS = list(EV.LEVELS) + list(EV.BUCKETS)   # the level vector: 3c 4a 4b 5a 5b reagent NA
# the pre-0.10.0 vector a schema-1 board row carries (rendered as it was, never diffed against the scale's)
OLD_LEVELS = ["2b", "3a", "3b", "4a", "4b", "4c", "4d", "5a", "5b"]
# "best evidence" = the levels of the identified claim (EV.CLAIM_IDENTIFIED): 3c
GOOD_LEVELS = {lv for lv in LEVELS if EV.claim_class(lv) == "identified"}
# the neutral established: identified + neutral (3c, 4a)
ESTABLISHED_LEVELS = {lv for lv in LEVELS if EV.claim_class(lv) in ("identified", "neutral")}
# M3: the other instrument's rows at level <= 4b
M3_LEVELS = {lv for lv in LEVELS if EV.claim_class(lv) in ("identified", "neutral", "ion")}
CLAIM_KEYS = list(EV.CLAIM_KEYS)               # identified neutral ion tentative + reagent, not assessed
CLAIM_LEVELS = {"identified": "3c", "neutral": "4a", "ion": "4b", "tentative": "5a, 5b, none",
                "reagent": "reagent bucket", "not assessed": "NA"}
CLAIMS_SCHEMA = 2       # 2: the evidence scale of peaky 0.10.0; 1: the pre-0.10.0 scale; none: before the claim
C13_PER_CARBON = 0.0107  # the 13C satellite per carbon (falsification)
BARE = {"[M-H]-", "[M+H]+"}
TOP_N = 50
PRESENCE = 0.5          # M1: a track present in >= this share of spectra
ROSTER_PRESENCE = 0.2   # M2: an expected line counts as present at this share of spectra (stamped or not)
ROSTER_SIGMA_K = 4.0    # M2: the presence window is this many of the run's measured mass sigmas ...
ROSTER_MIN_PPM = 1.0    # ... but never narrower than this, and never wider than the run's tolerance
ROSTER_TEST = 2         # M2's presence test: 2 = sigma window, share, isotope / reagent lines out, same ion apart
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
    def scale(self) -> bool:
        """True when the merged ledger carries the columns of the evidence scale of peaky 0.10.0 (its
        `evidence_level` is then the scale's; a run made before it carries the pre-0.10.0 level, never read)."""
        return set(EV.COLUMNS) <= set(self.ledger.columns)

    @property
    def code(self) -> str:
        """`<package version> <git commit>` from run_manifest.json (`code.package_version`,
        `code.git.commit`; older shapes read too)."""
        code = self.manifest.get("code") or {}
        git = (code.get("git") or {}).get("commit") or self.manifest.get("git_commit") or (self.manifest.get("git") or {}).get("commit") or ""
        pkg = code.get("package_version") or self.manifest.get("package_version") or self.manifest.get("version") or ""
        return f"{pkg} {str(git)[:9]}".strip()


def read_ledger(path: str) -> pd.DataFrame:
    """A ledger CSV with its `evidence_level` read literally: the scale writes the bucket `NA` as text, which
    pandas' default parser would turn into NaN (no level); an empty cell stays NaN."""
    frame = pd.read_csv(path, low_memory=False)
    if "evidence_level" in frame.columns:
        lv = pd.read_csv(path, usecols=["evidence_level"], keep_default_na=False, dtype=str)["evidence_level"]
        frame["evidence_level"] = lv.where(lv != "", np.nan).values
    return frame


def load_run(path: str) -> Run:
    path = resolve_run_dir(path)
    ledger = read_ledger(os.path.join(path, "merged_ledger.csv"))
    summary = _read_json(os.path.join(path, "batch_summary.json"))
    manifest = _read_json(os.path.join(path, "run_manifest.json"))
    ts_path = os.path.join(path, "per_file", "_batch_ts.parquet")
    ts = pd.read_parquet(ts_path) if os.path.isfile(ts_path) else None
    frames = []
    for f in sorted(glob.glob(os.path.join(path, "per_file", "*_ledger.csv"))):
        frame = read_ledger(f)
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


def pct_or_none(a, b) -> float | None:
    """pct, or None where there is nothing to take it of (undefined, not 0 %)."""
    return pct(a, b) if b else None


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


def md_table(rows: list[dict], columns: list[tuple[str, str]], nd: dict | None = None, missing: str = "") -> list[str]:
    """A GitHub table from records; `columns` = [(key, header)]; numbers right-aligned;
    a None cell prints `missing` (the alignment is read from the cells that are there)."""
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
            if v is None:
                cells.append(missing)
                continue
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
        return EV.as_list(value)
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
LEVEL_FRAME = ["neutral", "adduct", "level", "would_lift", "source"]


def levels_for(run: Run, levels_csv: str | None, corroborate: list[str], log=lambda *a: None) -> pd.DataFrame:
    """One row per (neutral, adduct) with its `level` on the evidence scale of peaky 0.10.0 and what would lift it.

    The merged ledger's in-core level when the run wrote the scale's columns (`source` in-core); else the CSV
    `scripts/level_ledger.py --out` wrote (`csv`; a CSV of the pre-0.10.0 script is refused); else the run
    levelled post hoc by `scripts/level_ledger.py` in-process (`post-hoc`), with the run dirs in `corroborate`
    (the other instrument) as other-source partners -- a tag, never a level."""
    led = run.ledger
    if run.scale:
        out = pd.DataFrame({"neutral": led["neutral_formula"], "adduct": led["adduct"],
                            "level": led["evidence_level"], "would_lift": col(led, "would_lift", ""),
                            "source": "in-core"})
        return out.drop_duplicates(["neutral", "adduct"])
    if levels_csv:
        df = read_ledger(levels_csv)
        if "evidence_level" not in df.columns:
            raise SystemExit(f"--levels {levels_csv}: no evidence_level column (a pre-0.10.0 levels table); "
                             "re-level with scripts/level_ledger.py")
        if "source" in df.columns and run.name in set(df["source"]):
            df = df[df["source"] == run.name]
        src = "csv"
    else:
        if run.per_file.empty:
            return pd.DataFrame(columns=LEVEL_FRAME)
        df = LL.run([run.path], list(corroborate), log=log)
        df["evidence_level"] = df["evidence_level"].where(df["evidence_level"].astype(str) != "", np.nan)
        src = "post-hoc"
    if df.empty:
        return pd.DataFrame(columns=LEVEL_FRAME)
    out = pd.DataFrame({"neutral": df["neutral_formula"].astype(str).values, "adduct": df["adduct"].astype(str).values,
                        "level": df["evidence_level"].values, "would_lift": col(df, "would_lift", "").values,
                        "source": src})
    return out.drop_duplicates(["neutral", "adduct"])


def own_levels_for(run: Run, log=lambda *a: None) -> pd.DataFrame:
    """One row per (neutral, adduct) the run's per-file ledgers commit, levelled on the run's OWN evidence: its
    files pooled as one source with no other-source partners (`evidence.level_source` on
    `evidence.source_from_run_dir`, the pooled entry the batch itself uses; a run made before the scale is
    levelled the same way). Ion-only readings are left out. The level a run's in-core stage gave can owe a series
    anchor to its own --corroborate partners -- for M3 the run being scored -- so M3 is counted on this too."""
    empty = pd.DataFrame(columns=LEVEL_FRAME)
    if run.per_file is None or run.per_file.empty or "role" not in run.per_file.columns:
        return empty
    pairs = EV.level_source(EV.source_from_run_dir(run.path))
    if pairs.empty:
        return empty
    if "ion_only_reading" in pairs.columns:
        pairs = pairs[~pairs["ion_only_reading"].map(EV.truthy).astype(bool)]
    out = pd.DataFrame({"neutral": pairs["neutral_formula"].astype(str).values,
                        "adduct": pairs["adduct"].astype(str).values,
                        "level": pairs["evidence_level"].where(pairs["evidence_level"].astype(str) != "", np.nan).values,
                        "would_lift": col(pairs, "would_lift", "").values, "source": "own"})
    return out.drop_duplicates(["neutral", "adduct"])


def level_vector(levels: pd.DataFrame) -> dict:
    """{level: n} over 3c 4a 4b 5a 5b reagent NA (zeros kept)."""
    counts = levels["level"].astype(str).value_counts() if "level" in levels.columns else pd.Series(dtype=int)
    return {k: int(counts.get(k, 0)) for k in LEVELS}


def level_rank(level) -> int:
    return LEVELS.index(level) if level in LEVELS else len(LEVELS)


def claim_of(level) -> str:
    """The claim a level supports (`evidence.claim_class`): identified (3c), neutral (4a), ion (4b), tentative
    (5a, 5b, no level), and the buckets reagent and not assessed (NA). A blank or 'nan' cell -- what a CSV round
    trip leaves of a missing level -- reads as no level; the literal `NA` is not assessed. The card always
    derives the claim from the level."""
    if isinstance(level, str) and level.strip().lower() in ("", "nan", "none", "<na>"):
        level = None
    return EV.claim_class(level)


def level_map(levels: pd.DataFrame | None) -> dict:
    """{(neutral, adduct): level} of a levels frame (`levels_for` / `own_levels_for`)."""
    if levels is None or levels.empty or "level" not in levels.columns:
        return {}
    key = "neutral" if "neutral" in levels.columns else "neutral_formula"
    return dict(zip(zip(levels[key].fillna("").astype(str), levels["adduct"].fillna("").astype(str)), levels["level"]))


def pair_keys(frame: pd.DataFrame) -> list[tuple[str, str]]:
    """(neutral_formula, adduct) of every row, blanks as ''."""
    return list(zip(col(frame, "neutral_formula").fillna("").astype(str), col(frame, "adduct").fillna("").astype(str)))


# ---------------------------------------------------------------------------
# 1. headline
# ---------------------------------------------------------------------------
def ion_only_mask(led: pd.DataFrame) -> pd.Series:
    """Rows the engine's ion-only stage wrote (merged or per-file ledger)."""
    if led is None or not len(led):
        return pd.Series(dtype=bool)
    return EV.is_ion_only(led)


def headline(run: Run, ions: pd.DataFrame) -> dict:
    led = run.ledger
    ts = run.ts
    tier = col(led, "tier", "")
    # the ion-only bucket (the engine's `ion_only` stage: `[M]-.` rows carrying an
    # `ion_only_of` link) is reported on its own and kept OUT of the Candidate
    # tile: the composition is pinned, the neutral is open, and the count would
    # otherwise inflate the tier it is deliberately kept apart from
    ion_only = ion_only_mask(led)
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
        "candidate": int(((tier == "Candidate") & ~ion_only).sum()),
        "ion_only": int(ion_only.sum()),
        "ion_only_levels": (col(led, "evidence_level", "")[ion_only].fillna("").astype(str).value_counts().to_dict()
                            if ion_only.any() and run.scale else {}),
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
# 0. the claim: identified / neutral / ion / tentative (+ reagent, not assessed)
# ---------------------------------------------------------------------------
TIER_BUCKETS = ("Assigned", "Candidate", "ion-only")


def tier_bucket(tier: pd.Series, ion_only: pd.Series) -> pd.Series:
    """Assigned / Candidate / ion-only per merged row: the ion-only rows are
    their own bucket, out of the Candidate count as `headline` keeps them."""
    t = tier.fillna("").astype(str)
    io = ion_only.reindex(t.index, fill_value=False).astype(bool)
    out = pd.Series([None] * len(t), index=t.index, dtype=object)
    out[t == "Assigned"] = "Assigned"
    out[t == "Candidate"] = "Candidate"
    out[io] = "ion-only"
    return out


def merged_claims(run: Run, levels: pd.DataFrame) -> pd.DataFrame:
    """One row per merged row: its level (the in-core `evidence_level` of a run on the scale, else the `levels`
    join), the claim derived from it, the tier bucket, and the stored `claim` cell when the run wrote one on the
    scale (a pre-0.10.0 claim is not read)."""
    led = run.ledger
    if run.scale:
        level = led["evidence_level"]
    else:
        lv = level_map(levels)
        level = pd.Series([lv.get(k) for k in pair_keys(led)], index=led.index, dtype=object)
    out = pd.DataFrame({
        "neutral": col(led, "neutral_formula").fillna("").astype(str),
        "adduct": col(led, "adduct").fillna("").astype(str),
        "level": level.map(lambda v: str(v) if is_str(v) else ""),
        "tier": col(led, "tier", "").fillna("").astype(str),
    }, index=led.index)
    out["claim"] = level.map(claim_of)
    out["bucket"] = tier_bucket(out["tier"], ion_only_mask(led))
    if run.scale and "claim" in led.columns:
        out["stored"] = led["claim"].map(lambda v: str(v) if is_str(v) else None)
    return out


def per_file_signal(run: Run, merged: pd.DataFrame) -> pd.DataFrame:
    """Every committed per-file M0 reading (`role == 'M0'`) with its height,
    joined on (neutral_formula, adduct) to the merged ledger: it takes the
    MERGED claim and tier bucket; a reading no merged row carries (the merge
    vote kept another) is `unmatched`."""
    pf = run.per_file
    if pf is None or pf.empty or "role" not in pf.columns:
        return pd.DataFrame(columns=["neutral", "adduct", "height", "claim", "bucket"])
    m0 = pf[pf["role"] == "M0"]
    sig = pd.DataFrame({
        "neutral": col(m0, "neutral_formula").fillna("").astype(str).values,
        "adduct": col(m0, "adduct").fillna("").astype(str).values,
        "height": pd.to_numeric(col(m0, "height"), errors="coerce").fillna(0.0).values,
    })
    right = merged.drop_duplicates(["neutral", "adduct"])[["neutral", "adduct", "claim", "bucket"]]
    sig = sig.merge(right, on=["neutral", "adduct"], how="left")
    sig["claim"] = sig["claim"].fillna("unmatched")
    return sig


def stamp_mismatch(run: Run, merged: pd.DataFrame) -> dict:
    """Rows whose stored `claim` differs from the one derived from their level
    (a per-file row other than M0 should carry none). None where the run wrote
    no `claim` column on the scale -- it predates it; expected 0 otherwise."""
    out = {"merged": None, "per_file": None}
    if "stored" in merged.columns:
        out["merged"] = int((merged["stored"].fillna("") != merged["claim"]).sum())
    pf = run.per_file
    if run.scale and pf is not None and not pf.empty and "claim" in pf.columns and "role" in pf.columns \
            and "evidence_level" in pf.columns:
        stored = pf["claim"].map(lambda v: str(v) if is_str(v) else "")
        m0 = pf["role"] == "M0"
        derived = col(pf, "evidence_level").map(claim_of).where(m0, "")
        out["per_file"] = int((stored != derived).sum())
    return out


def claims(run: Run, levels: pd.DataFrame) -> dict:
    """The claim each committed reading supports, beside its tier: merged rows
    per class and the tier x claim crosstab (rows and signal); the committed
    per-file M0 signal per class, with the readings no merged row carries as an
    explicit `unmatched` bucket; and the rows where tier and claim disagree.
    The classes are the four claims and the two buckets reported beside them
    (reagent, not assessed). The acceptance metrics read the identified class;
    the tier is a separate verdict and is never changed from it."""
    led = run.ledger
    merged = merged_claims(run, levels)
    sig = per_file_signal(run, merged)
    committed = float(sig["height"].sum())
    mismatch = stamp_mismatch(run, merged)
    checked = [v for v in mismatch.values() if v is not None]

    def share(mask) -> float:
        return pct(float(sig.loc[mask, "height"].sum()), committed) if committed else float("nan")

    by_rows = {b: {c: int(((merged["bucket"] == b) & (merged["claim"] == c)).sum()) for c in CLAIM_KEYS} for b in TIER_BUCKETS}
    by_signal = {b: {c: share((sig["bucket"] == b) & (sig["claim"] == c)) for c in CLAIM_KEYS} for b in TIER_BUCKETS}
    signal = {
        "committed": committed,
        "n_m0": int(len(sig)),
        "n_unmatched": int((sig["claim"] == "unmatched").sum()),
        "share": {c: share(sig["claim"] == c) for c in (*CLAIM_KEYS, "unmatched")},
        "share_assigned": {c: share((sig["claim"] == c) & (sig["bucket"] == "Assigned")) for c in CLAIM_KEYS},
    }
    # the signal each merged row's reading carries over the per-file ledgers
    pair_h = sig.groupby(["neutral", "adduct"])["height"].sum().to_dict() if len(sig) else {}
    row_share = pd.Series([pct(pair_h.get(k, 0.0), committed) if committed else float("nan")
                           for k in zip(merged["neutral"], merged["adduct"])], index=merged.index)
    lift = {(str(n), str(a)): str(w) for n, a, w in zip(levels.get("neutral", []), levels.get("adduct", []),
                                                         levels.get("would_lift", [])) if is_str(w)}

    def row_of(i, kind: str | None = None) -> dict:
        r = led.loc[i]
        out = {
            "mz": float(r["mz"]) if pd.notna(r.get("mz")) else None,
            "neutral": merged.at[i, "neutral"], "adduct": merged.at[i, "adduct"],
            "tier": merged.at[i, "tier"], "level": merged.at[i, "level"], "claim": merged.at[i, "claim"],
            "n_files": int(r["n_files"]) if pd.notna(r.get("n_files")) else None,
            "signal_share": float(row_share.at[i]),
            "would_lift": lift.get((merged.at[i, "neutral"], merged.at[i, "adduct"]), ""),
            "tier_reason": str(r["tier_reason"]) if is_str(r.get("tier_reason")) else "",
        }
        if kind:
            out = {"kind": kind, **out}
        return out

    a_t = merged.index[(merged["bucket"] == "Assigned") & (merged["claim"] == "tentative")]
    c_i = merged.index[(merged["bucket"] == "Candidate") & (merged["claim"] == "identified")]
    dis = [row_of(i, "Assigned but tentative") for i in a_t] + [row_of(i, "Candidate but identified") for i in c_i]
    dis.sort(key=lambda r: -(r["signal_share"] if np.isfinite(r["signal_share"]) else -1.0))
    return {
        "scale": f"peaky {EV.SCALE_RELEASE}",
        "meaning": dict(EV.CLAIM_MEANING),
        "levels": dict(CLAIM_LEVELS),
        "level_source": str(levels["source"].iloc[0]) if len(levels) and "source" in levels.columns else None,
        "n_rows": int(len(merged)),
        "rows": EV.summarize_claims(merged["claim"]),
        "by_tier": {"rows": by_rows, "signal": by_signal},
        "signal": signal,
        "disagree": {
            "assigned_tentative": int(len(a_t)),
            "candidate_identified": int(len(c_i)),
            "rows": dis[:60],
        },
        "stamp_mismatch": int(sum(checked)) if checked else None,
        "stamp_mismatch_by_ledger": mismatch,
    }


# ---------------------------------------------------------------------------
# 2 + 3. the brightest and the best-evidence ions
# ---------------------------------------------------------------------------
def brightest(run: Run, ions: pd.DataFrame, tracks: pd.DataFrame, levels: pd.DataFrame, n: int = TOP_N) -> dict:
    """The n brightest stamped ions (median height over the batch) with their
    reading, tier, level and claim; and how many of the batch's n brightest
    TRACKS overall are unstamped (those are named in M1). A bright M0 not
    assessed (NA) is counted apart from the ones not identified."""
    if ions.empty:
        return {"rows": [], "m0_not_assigned": 0, "m0_not_identified": 0, "m0_not_assessed": 0,
                "unstamped_in_top": 0, "n": 0}
    lv = levels.set_index(["neutral", "adduct"]) if not levels.empty else None
    top = ions.sort_values("med_h", ascending=False).head(n)
    # the ion-only bucket (`[M]-.` rows carrying an `ion_only_of` link on the merged
    # ledger): bright by nature and Candidate by design, so it is flagged on its
    # row and left out of "bright M0 not Assigned" -- that count is for readings
    # the engine could not confirm, not for a bucket it deliberately keeps open
    led = run.ledger
    io_pairs = set()
    if led is not None and len(led) and "neutral_formula" in led.columns:
        io = ion_only_mask(led)
        io_pairs = set(zip(led.loc[io, "neutral_formula"].astype(str), col(led, "adduct", "").astype(str)[io]))
    rows = []
    for r in top.itertuples():
        level = ""
        if lv is not None and is_str(r.neutral) and (r.neutral, r.adduct) in lv.index:
            hit = lv.loc[(r.neutral, r.adduct)]
            hit = hit.iloc[0] if isinstance(hit, pd.DataFrame) else hit
            level = str(hit["level"]) if is_str(hit["level"]) else ""
        ion_only = is_str(r.neutral) and (str(r.neutral), str(r.adduct)) in io_pairs
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
                "suspect": bool(r.suspect),
                "ion_only": bool(ion_only),
                # only a committed M0 reading makes a claim; one with no level reads tentative
                "claim": claim_of(level) if str(r.role) == "M0" else "",
            }
        )
    m0_not_assigned = sum(1 for r in rows if r["role"] == "M0" and r["tier"] != "Assigned" and not r["ion_only"])
    m0_not_identified = sum(1 for r in rows if r["role"] == "M0" and r["claim"] not in ("identified", EV.CLAIM_NA)
                            and not r["ion_only"])
    m0_not_assessed = sum(1 for r in rows if r["role"] == "M0" and r["claim"] == EV.CLAIM_NA and not r["ion_only"])
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
        "m0_not_identified": int(m0_not_identified),
        "m0_not_assessed": int(m0_not_assessed),
        "ion_only_in_top": int(sum(1 for r in rows if r["ion_only"])),
        "by_role": {k: int(v) for k, v in pd.Series([r["role"] for r in rows]).value_counts().items()},
        "unstamped_in_top": int((all_tracks["kind"] == "unstamped").sum()),
    }


def best_evidence(run: Run, ions: pd.DataFrame, levels: pd.DataFrame, n: int = TOP_N) -> dict:
    """The n best-evidence M0 rows: in level order (3c first), then brightest;
    plus the level vector and the claims over every levelled row."""
    if levels.empty:
        return {"rows": [], "levels": level_vector(levels), "n_good": 0, "n_established": 0, "n_levelled": 0,
                "by_claim": EV.summarize_claims([])}
    m0 = ions[ions["role"] == "M0"][["neutral", "adduct", "ion_mz", "med_h", "n", "tier"]] if not ions.empty else pd.DataFrame()
    df = levels.merge(m0, on=["neutral", "adduct"], how="left") if not m0.empty else levels.assign(ion_mz=np.nan, med_h=np.nan, n=np.nan, tier="")
    df["rank"] = df["level"].map(level_rank)
    df["good"] = df["level"].isin(GOOD_LEVELS)
    top = df.sort_values(["rank", "med_h"], ascending=[True, False], na_position="last").head(n)
    rows = [
        {
            "neutral": str(r.neutral),
            "adduct": str(r.adduct),
            "ion_mz": float(r.ion_mz) if pd.notna(r.ion_mz) else None,
            "med_cps": float(r.med_h) if pd.notna(r.med_h) else None,
            "in": f"{int(r.n)}/{run.n_spectra}" if pd.notna(r.n) else "",
            "tier": str(r.tier) if is_str(r.tier) else "",
            "level": str(r.level) if is_str(r.level) else "",
            "claim": claim_of(r.level),
            "would_lift": str(r.would_lift)[:120] if is_str(getattr(r, "would_lift", "")) else "",
        }
        for r in top.itertuples()
    ]
    return {
        "rows": rows,
        "levels": level_vector(levels),
        "n_good": int(df["good"].sum()),
        "n_established": int(df["level"].isin(ESTABLISHED_LEVELS).sum()),
        "n_levelled": int(len(df)),
        "by_claim": EV.summarize_claims(levels["level"].map(claim_of)),
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


def roster_window_ppm(run: Run) -> float:
    """M2's presence window: ROSTER_SIGMA_K x the run's measured mass sigma (batch_summary `mass_scale.sigma_ppm`),
    at least ROSTER_MIN_PPM, at most the run's tolerance (the tolerance itself when no sigma was measured). A flat
    6 ppm on an Orbitrap at ~0.2 ppm sigma reaches a neighbour 25 sigma away."""
    try:
        sigma = float(((run.summary or {}).get("mass_scale") or {}).get("sigma_ppm") or 0.0)
    except (TypeError, ValueError):
        sigma = 0.0
    if not np.isfinite(sigma) or sigma <= 0:
        return float(run.tol_ppm)
    return float(min(run.tol_ppm, max(ROSTER_SIGMA_K * sigma, ROSTER_MIN_PPM)))


#: M2 statuses, best first; `present` counts every one but the last two
M2_ORDER = {"assigned": 0, "candidate": 1, "same ion": 2, "read as": 3, "unstamped": 4, "isotope/reagent line": 5,
            "absent": 6}


def line_ion_key(ion) -> str | None:
    """The composition of a stamped line's own ion formula (a charge-signed formula; a reagent line carries no
    neutral), in `ion_key`'s form; None when there is none."""
    if not is_str(ion) or not ion.strip().endswith(("+", "-")):
        return None
    try:
        counts = C.parse_formula(ion.strip())
    except Exception:  # noqa: BLE001 - an unparseable line has no key
        return None
    return C.format_formula(counts) if counts else None


def _m2_line_status(s, neutral: str, source: str, want: str | None) -> str:
    """What one stamped line in the window says about an expected neutral whose ion on this channel is `want`:
    read as itself (Assigned / Candidate), another split of the same ion composition, another reading, or no
    sighting (an isotope satellite of another ion, a reagent line of another composition). A reagent line of
    the expected ion's own composition is that ion, read as the reagent: the reagent's reference ions (its ion,
    its water clusters) are read as themselves there; for any other source it is the same ion."""
    if str(s["role"]) == "M0":
        if is_str(s["neutral"]) and s["neutral"] == neutral:
            return "assigned" if s["tier"] == "Assigned" else "candidate"
        same_ion = want is not None and ion_key(s["neutral"], s["adduct"], s["ion_formula"]) == want
        return "same ion" if same_ion else "read as"
    if str(s["role"]) == "reagent" and want is not None and line_ion_key(s["ion_formula"]) == want:
        return "assigned" if source == "reference_ion" else "same ion"
    return "isotope/reagent line"


def missed_m2(run: Run, ions: pd.DataFrame, tracks: pd.DataFrame, rosters: pd.DataFrame,
              levels: pd.DataFrame | None = None) -> dict:
    """M2: every expected neutral, looked for on the run's own channels in the
    time series: assigned as itself / the same ion read as another split of it /
    read as something else / present but unstamped (with the engine's reason) /
    on an isotope or reagent line / absent. Roster recall per class.

    A line is looked for inside `roster_window_ppm` (the run's measured mass
    sigma, not a flat tolerance) and counts only in >= ROSTER_PRESENCE of the
    spectra, stamped or not; of the lines that pass, the best-read one stands
    for the formula (`_m2_line_status`). An isotope satellite of another ion, or
    a reagent line of another composition, is no sighting of the formula; a
    reagent line of the formula's own ion composition is (the reagent's
    reference ions are read as themselves there). A line read as another
    neutral / adduct split of the SAME ion composition (C10H15NO7 [M-H]- and
    C10H14O4 [M+NO3]- are one composition, one exact mass) is `same ion`, not a
    misread: nothing in that line's mass tells the two apart.

    With `levels`, each row carries the claim of the reading on its line and
    `roster_claim` counts, per roster, the formulas read as themselves per
    claim, `neutral_or_better` (identified + neutral) and the misreads whose
    other reading is identified; `scale` names the level scale the claims read."""
    exp = expectations(run, rosters)
    window = roster_window_ppm(run)
    if exp.empty:
        return {"rows": [], "roster": {}, "sources": {}, "roster_claim": {}, "scale": f"peaky {EV.SCALE_RELEASE}",
                "test": ROSTER_TEST, "window_ppm": window}
    lv = level_map(levels) if levels is not None else None
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
            tol = window * 1e-6 * target
            # the stamped lines in the window, present in enough spectra: the best-read one stands for the formula
            # (a reading of the formula before another split of its ion, before another reading, before an
            # isotope / reagent line), the nearest among equals
            want = ion_key(e.neutral, adduct)
            lo, hi = np.searchsorted(st_mz, target - tol, "left"), np.searchsorted(st_mz, target + tol, "right")
            j, status = None, None
            for i in range(lo, hi):
                if float(st.iloc[i]["presence"]) < ROSTER_PRESENCE:
                    continue
                st_i = _m2_line_status(st.iloc[i], e.neutral, e.source, want)
                if j is None or M2_ORDER[st_i] < M2_ORDER[status] or (
                        st_i == status and abs(st_mz[i] - target) < abs(st_mz[j] - target)):
                    j, status = i, st_i
            if j is not None:
                s, mz = st.iloc[j], st_mz[j]
                same = is_str(s["neutral"]) and s["neutral"] == e.neutral
                m0_line = str(s["role"]) == "M0"
                cand = {"status": status, "adduct": adduct, "mz": float(mz), "cps": float(s["med_h"]), "in": f"{int(s['n'])}/{run.n_spectra}",
                        "read": "" if same and m0_line else f"{s['ion_formula']}" + (f" = {s['neutral']} {s['adduct']}" if is_str(s["neutral"]) and m0_line else f" ({s['role']})"),
                        "reason": "",
                        # the committed reading on the line, whose claim the row carries
                        "pair": (str(s["neutral"]), str(s["adduct"])) if is_str(s["neutral"]) and m0_line else None}
            else:
                k, mz = nearest(tr_mz, target)
                if k is not None and abs(mz - target) <= tol and tr.iloc[k]["presence"] >= ROSTER_PRESENCE:
                    t = tr.iloc[k]
                    tag = _tag_for(tags, float(mz), window)
                    cand = {"status": "unstamped", "adduct": adduct, "mz": float(mz), "cps": float(t["med_h"]), "in": f"{int(t['n'])}/{run.n_spectra}",
                            "read": "", "reason": tag["reason"] or (tag["tag"] and f"residual tag: {tag['tag']}") or "", "pair": None}
                else:
                    cand = {"status": "absent", "adduct": adduct, "mz": None, "cps": None, "in": "", "read": "", "reason": "", "pair": None}
            order = M2_ORDER
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
                "claim": claim_of(lv.get(best["pair"])) if lv is not None and best["pair"] else "",
            }
        )
    df = pd.DataFrame(rows)
    roster_summary, per_class, roster_claim = {}, {}, {}
    if not df.empty:
        ros = df[df["source"].str.startswith("roster:")].drop_duplicates("neutral")
        for name, g in ros.groupby(ros["source"].str[7:]):
            roster_summary[name] = _recall(g)
            per_class[name] = {c: _recall(gc) for c, gc in g.groupby("cls")}
            if lv is not None:
                itself = g.loc[g["status"].isin(("assigned", "candidate")), "claim"]
                roster_claim[name] = {c: int((itself == c).sum()) for c in CLAIM_KEYS}
                # identified (3c) beside neutral or better (3c + 4a): a class list (no named entry per formula)
                # can never reach 3c, so identified alone is no recall metric for it. Per line read: the claim of
                # the one line M2 picked for the formula, not its best claim over every channel it is read on
                roster_claim[name]["neutral_or_better"] = roster_claim[name]["identified"] + roster_claim[name]["neutral"]
                roster_claim[name]["misread_identified"] = int(((g["status"] == "read as") & (g["claim"] == "identified")).sum())
        sources = {s: _recall(g) for s, g in df[~df["source"].str.startswith("roster:")].groupby("source")}
    else:
        sources = {}
    return {"rows": rows, "roster": roster_summary, "roster_by_class": per_class, "sources": sources,
            "roster_claim": roster_claim, "scale": f"peaky {EV.SCALE_RELEASE}", "test": ROSTER_TEST,
            "window_ppm": window}


def _recall(g: pd.DataFrame) -> dict:
    n = int(len(g))
    st = g["status"]
    return {
        "n": n,
        "assigned": int((st == "assigned").sum()),
        "candidate": int((st == "candidate").sum()),
        "same_ion": int((st == "same ion").sum()),
        "read_as": int((st == "read as").sum()),
        "unstamped": int((st == "unstamped").sum()),
        "iso_reagent": int((st == "isotope/reagent line").sum()),
        "absent": int((st == "absent").sum()),
        # an isotope or reagent line is no sighting of the formula
        "present": int((~st.isin(("absent", "isotope/reagent line"))).sum()),
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


#: M3's basis when the other instrument's rows are not assessed (a TOF): its Assigned rows instead (D23)
M3_BASIS_LEVEL = "level <= 4b (3c / 4a / 4b)"
M3_BASIS_ASSIGNED = "Assigned rows (the other instrument is not assessed on this scale)"


def _m3_other_instrument(mine: set, other_instrument: Run | None, levels: pd.DataFrame | None,
                         overlap: tuple | None, masks: list[tuple], floor_cps: float, floor_share: float) -> dict | None:
    """The neutrals the other INSTRUMENT holds at level <= 4b (by `levels`), above the detection floor inside the
    overlap window, that this run lacks. Where the other instrument's rows are not assessed (NA: a TOF), its
    Assigned rows count instead, and the basis says so."""
    if other_instrument is None or levels is None or levels.empty or other_instrument.ts is None:
        return None
    ots = other_instrument.ts
    keep = _window_mask(ots, overlap, masks)
    w = ots[keep & ots["neutral_formula"].notna() & ~col(ots, "dup_candidate", False).fillna(False).astype(bool)]
    n_in_window = int(ots.loc[keep, "sample_item_id"].nunique()) if "sample_item_id" in ots.columns else 0
    oled = other_instrument.ledger
    is_a = (col(oled, "tier", "") == "Assigned").to_numpy()
    assigned = {k for k, a in zip(zip(col(oled, "neutral_formula").fillna("").astype(str),
                                      col(oled, "adduct").fillna("").astype(str)), is_a) if a}
    lv = levels["level"].astype(str)
    na = lv == "NA"
    by_level = lv.isin(M3_LEVELS)
    by_tier = na & pd.Series([k in assigned for k in zip(levels["neutral"].astype(str), levels["adduct"].astype(str))],
                             index=levels.index)
    good = levels[by_level | by_tier]
    basis = M3_BASIS_ASSIGNED if (na.all() and len(levels)) else (
        M3_BASIS_LEVEL + "; Assigned where not assessed" if na.any() else M3_BASIS_LEVEL)
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
        rows.append({"neutral": str(r.neutral), "adduct": str(r.adduct), "level": str(r.level),
                     "med_cps": med, "share": float(share * 100)})
    rows.sort(key=lambda x: (level_rank(x["level"]), -x["med_cps"]))
    # the same split by the claim the other's level supports
    good_claim = good["level"].map(claim_of)
    miss_claim = [claim_of(x["level"]) for x in rows]
    keys = ("identified", "neutral", "ion", EV.CLAIM_NA)
    return {
        "run": other_instrument.name, "reagent": other_instrument.reagent, "basis": basis,
        "n_good": int(len(good)), "n_spectra_in_window": n_in_window,
        "n_missing": int(len(rows)),
        "n_good_by_claim": {c: int((good_claim == c).sum()) for c in keys},
        "n_missing_by_claim": {c: int(sum(1 for m in miss_claim if m == c)) for c in keys},
        "rows": rows[:40],
        "window": [str(overlap[0]), str(overlap[1])] if overlap else None,
        "floor": {"cps": floor_cps, "share": floor_share},
    }


def missed_m3(run: Run, other: Run | None, other_instrument: Run | None, other_levels: pd.DataFrame | None,
              overlap: tuple | None, masks: list[tuple], floor_cps: float, floor_share: float,
              own_levels: pd.DataFrame | None = None) -> dict:
    """M3: neutrals the other path found (>= 2 files or Assigned) that this run
    lacks; and neutrals the other INSTRUMENT holds at level <= 4b (its Assigned
    rows where it is not assessed), above the detection floor inside the overlap
    window, that this run lacks -- once by `other_levels` (its merged ledger's
    in-core level) and once by `own_levels` (its own evidence, `own_levels_for`:
    no other-source partners, so none of THIS run's pairs anchors its series)."""
    mine = set(run.ledger["neutral_formula"].dropna().astype(str)) if "neutral_formula" in run.ledger.columns else set()
    out = {"other_path": None, "other_instrument": None, "other_instrument_own": None}
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
    out["other_instrument"] = _m3_other_instrument(mine, other_instrument, other_levels, overlap, masks, floor_cps, floor_share)
    out["other_instrument_own"] = _m3_other_instrument(mine, other_instrument, own_levels, overlap, masks, floor_cps, floor_share)
    return out


# ---------------------------------------------------------------------------
# 5. is it right
# ---------------------------------------------------------------------------
def mass_only_census(run: Run) -> dict | None:
    """The TOF mass-only flag of the run's Assigned rows (peaky/assignment/mass_only.py), re-read off the merged
    ledger and split at the run's own threshold (batch_summary['tof_flag'], else the package default):
    {threshold_mz, n_assigned, n_flagged, below, at_or_above}; None on a run without flag values (not a
    TOF-class run, or made before the flag)."""
    from peaky.assignment import mass_only as MO
    thr = ((run.summary or {}).get("tof_flag") or {}).get("threshold_mz") or MO.DEFAULT_TOF_FLAG_MZ
    return MO.counts(run.ledger, thr)


def census(run: Run) -> dict:
    """Element census of the Assigned neutrals: heteroatoms in a clean
    oxidation experiment are contamination or error until named. On a TOF-class
    run it also counts the Assigned rows the mass-only flag marks (`mass_only`)."""
    led = run.ledger
    if "neutral_formula" not in led.columns:
        return {"n_assigned": 0, "elements": {}, "examples": {}, "mass_only": None}
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
    return {"n_assigned": int(len(ass)), "elements": out, "examples": examples, "mass_only": mass_only_census(run)}


def mass_only_line(mo: dict | None) -> str:
    """The census line on the TOF mass-only flag ('' without one)."""
    if not mo:
        return ""
    b, a, thr = mo["below"], mo["at_or_above"], mo["threshold_mz"]
    return (f"TOF mass-only flag: {mo['n_flagged']} of {mo['n_assigned']} Assigned rows have no attached isotope line "
            f"that speaks for the neutral in any Assigned file -- {b['flagged']} of {b['assigned']} below m/z {thr:g} "
            f"(the reading rests on mass alone), {a['flagged']} of {a['assigned']} at or above it (formula space "
            "saturated on a TOF); tier unchanged")


# --- decoy ------------------------------------------------------------------
# signal_to_noise: the v2 fit charges a missing line only where the noise says it was visible, so a decoy arm
# without it would be judged in the no-SNR mode while its file was not (a run before 0.9.0 has no such column)
PEAK_COLS = ["sample_item_id", "peak_id", "mz", "sparsity", "area", "height", "signal_to_noise"]
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


def decoy_ppm_peaks(peaks: pd.DataFrame, k_ppm: float) -> pd.DataFrame:
    """Every m/z scaled by (1 + k_ppm * 1e-6): the same spectrum a few ppm off its true formulas. Outside the
    match window the true formula is out of reach, but the shifted line still sits in the populated mass-defect
    band, where the formula grid is dense -- so the engine proposes wrong formulas and the mass, degeneracy and
    pattern gates are exercised. A shift keeps every isotope and label spacing, so the label / isotope vetoes
    are not: no shift decoy tests them. (The 0.35 Da arm moves the lines into the empty gap below ~m/z 350,
    where nothing is proposed and no gate is tested.)"""
    out = peaks.copy()
    out["mz"] = out["mz"] * (1.0 + float(k_ppm) * 1e-6)
    out["peak_id"] = out["peak_id"].astype(str) + "_decoy"
    return out


#: the populated-defect shift arms run by default (ppm; each must sit outside the file's match window)
DECOY_PPM = (9.0, -9.0)
#: m/z bins of the decoy-by-mass table
DECOY_BIN_DA = 50


def ppm_arm(k_ppm: float) -> str:
    """The arm key (and kept-ledger file tag) of a ppm-shift arm: +9 -> 'ppmp9', -4.5 -> 'ppmm4d5'."""
    return "ppm" + ("p" if float(k_ppm) > 0 else "m") + f"{abs(float(k_ppm)):g}".replace(".", "d")


def parse_ppm_list(text: str | None) -> list[float]:
    """`--decoy-ppm` '9,-9' -> [9.0, -9.0]; 'none' / '' / '0' -> [] (no ppm arm)."""
    if text is None:
        return list(DECOY_PPM)
    vals = []
    for part in str(text).replace(";", ",").split(","):
        part = part.strip()
        if not part or part.lower() == "none":
            continue
        v = float(part)
        if v != 0.0 and v not in vals:
            vals.append(v)
    return vals


def arm_window_ppm(run: "Run", file_id: str) -> float:
    """How far from a line the engine still reaches for a formula on this file: the wider of the grid's
    enumeration window (`PassConfig.search_ppm`) and the match window the arm is scored at (the file's
    `pattern_scoring` snapshot, else the class fallback's). A ppm-shift arm inside it is no decoy: the true
    formula stays in reach."""
    from mascope_tools.composition import resolve_match_tolerance_ppm

    from peaky.assignment import passes as PA

    snap = decoy_scoring(run, file_id)
    window = None
    if snap:
        try:
            window = float(snap.get("mz_tolerance_ppm"))
        except (TypeError, ValueError):
            window = None
    if window is None or not np.isfinite(window):
        window = float(resolve_match_tolerance_ppm(None))
    return max(float(PA.PassConfig().search_ppm), window)


def decoy_instrument(run: "Run", file_id: str) -> str | None:
    """The instrument class the run's scoring snapshot names for the file ('orbi', 'tof'), or None."""
    snap = decoy_scoring(run, file_id) or {}
    kind = snap.get("instrument_type")
    return str(kind) if kind else None


def wrong_adducts(polarity: str) -> list[str]:
    return list(WRONG_ADDUCTS.get(polarity, WRONG_ADDUCTS["-"]))


def own_adducts(run: "Run") -> set[str]:
    """Every channel the run itself reads: its profile's adducts, every adduct its merged and per-file ledgers
    commit (a side channel the run opened included), and any channel list its batch summary
    records."""
    own = set(run.adducts)
    for frame in (run.ledger, run.per_file):
        if frame is not None and "adduct" in frame.columns:
            own |= set(frame["adduct"].dropna().astype(str))
    for key, val in (run.summary or {}).items():
        if str(key).endswith("channels") and isinstance(val, (list, tuple)):
            own |= {str(v) for v in val if isinstance(v, str)}
    return {a for a in own if a and a.lower() != "nan"}


def wrong_adducts_for(run: "Run") -> list[str]:
    """The wrong-adducts arm's channels: the polarity's wrong set minus every channel the run itself reads -- an
    adduct the run's own chemistry makes (iodide's [M+I]-, a side channel the run opened) is never 'wrong'."""
    own = own_adducts(run)
    return [a for a in wrong_adducts(run.polarity) if a not in own]


def ion_key(neutral, adduct, ion=None) -> str | None:
    """The ion composition of a reading (`evidence.ion_composition`) as one formula string; None when the row
    carries no reading. Two splits of one ion (C10H14O4 [M+NO3]- and C10H15NO7 [M-H]-) share a key."""
    if not is_str(neutral) or not is_str(adduct):
        return None
    try:
        counts = EV.ion_composition(neutral, adduct, ion)
    except Exception:  # noqa: BLE001 - an unparseable reading has no key
        return None
    return C.format_formula(counts) if counts else None


def ion_level_counts(arm_led: pd.DataFrame, control_led: pd.DataFrame | None) -> dict:
    """The wrong-adducts arm at the ION level: of its Assigned M0 rows (ion-only rows aside), how many carry the
    same ion composition as a control M0 reading on the same peak (any tier: another split of an ion the run
    already reads, by construction of the arm) and how many a NEW ion (a different composition, or a peak the
    control left unexplained). Only the new ions are wrong at the ion level."""
    out = {"assigned_same_ion": 0, "assigned_new_ion": 0, "assigned_new_ion_lt_350": 0, "new_ion_examples": []}
    if arm_led is None or arm_led.empty or "role" not in arm_led.columns:
        return out
    m0 = arm_led[arm_led["role"] == "M0"]
    if m0.empty:
        return out
    m0 = m0[(col(m0, "tier", "") == "Assigned") & ~ion_only_mask(m0)]
    ctrl_ions: dict[str, set] = {}
    if control_led is not None and not control_led.empty and "role" in control_led.columns:
        cm = control_led[control_led["role"] == "M0"]
        for pid, n, a, i in zip(col(cm, "peak_id").astype(str), col(cm, "neutral_formula"), col(cm, "adduct"),
                                col(cm, "ion_formula")):
            k = ion_key(n, a, i)
            if k:
                ctrl_ions.setdefault(pid, set()).add(k)
    new = []
    for r in m0.itertuples(index=False):
        k = ion_key(getattr(r, "neutral_formula", None), getattr(r, "adduct", None), getattr(r, "ion_formula", None))
        if k is not None and k in ctrl_ions.get(str(getattr(r, "peak_id", "")), set()):
            out["assigned_same_ion"] += 1
        else:
            out["assigned_new_ion"] += 1
            mz = float(getattr(r, "mz", np.nan))
            out["assigned_new_ion_lt_350"] += int(mz < DECOY_MZ_SPLIT)
            new.append((float(getattr(r, "height", 0) or 0), f"{r.neutral_formula} {r.adduct} @ {mz:.4f}"))
    out["new_ion_examples"] = [t for _h, t in sorted(new, key=lambda x: -x[0])[:6]]
    return out


def brightest_files(run: Run, n: int) -> list[str]:
    pf = run.per_file
    if pf.empty:
        return []
    order = pf.groupby("__file")["height"].sum().sort_values(ascending=False)
    return list(order.index[:n])


def decoy_scoring(run: Run, file_id: str) -> dict | None:
    """What the run judged `file_id` at: its `pattern_scoring` snapshot in the
    batch summary (a 0.9.0 run records one per sample), or None for a run that
    predates it. A decoy arm of that file is judged at the same measurement --
    width, offset, window and abundance floor -- so the arm bounds the run as scored, not a
    forgiving class fallback (card C35)."""
    from peaky.io import io_mascope as IO

    ps = run.summary.get("pattern_scoring")
    snap = ps.get(file_id) if isinstance(ps, dict) else None
    if not isinstance(snap, dict):
        return None
    try:            # the registration's own check, so the card's label and the arm's scoring cannot disagree
        IO._check_offline_scoring(snap)
    except (TypeError, ValueError):
        return None
    return snap


def decoy_scoring_summary(dc: dict) -> str | None:
    """One word for what a card's decoy arms were judged at: 'inherited' (every
    file at the run's own measurement of it), 'class-fallback', 'mixed', or
    'unrecorded' (a kept manifest from before the field existed)."""
    sc = (dc or {}).get("scoring")
    if not sc:
        return None
    kinds = set(sc.values()) if isinstance(sc, dict) else {str(sc)}
    return kinds.pop() if len(kinds) == 1 else "mixed"


def _g(v) -> str:
    """A small number to three significant figures (an abundance floor of 0.004 is not 0.00)."""
    return f"{float(v):.3g}" if isinstance(v, (int, float)) and not isinstance(v, bool) else DASH


#: what a decoy arm does not take from the run it bounds (card C39)
DECOY_NOT_INHERITED = ("its batch height cutoff, pre-calibration prior offset, batch occurrence "
                       "table, batch time series, active reference lists (the reflist prior and the pass-8 rescue) "
                       "or corroboration")


def _scoring_note(dc: dict) -> str:
    sc = (dc or {}).get("scoring")
    if not isinstance(sc, dict) or not sc:
        return ""
    det = (dc or {}).get("scoring_detail") or {}
    parts = []
    for f, k in sc.items():
        d = det.get(f) or {}
        if k == "inherited" and d:
            src = "" if d.get("sigma_source") == "fitted" else f", the run's own {d.get('sigma_source')} width"
            parts.append(f"`{f}` inherited (sigma {_d(d.get('sigma_ppm'), 2)} ppm, mu {_d(d.get('mu_ppm'), 2)} ppm, "
                         f"window {_d(d.get('mz_tolerance_ppm'), 0)} ppm, floor {_g(d.get('abundance_floor'))}, "
                         f"{_d(d.get('fitted_anchors'))} anchors{src})")
        else:
            parts.append(f"`{f}` {k}")
    return ("arms judged at: " + ", ".join(parts)
            + f". Inherited = the run's width, offset, window and abundance floor, the file's signal-to-noise and, on "
            f"the arms that read the run's own channels, the side channels the run recorded opening; an "
            f"arm does not take {DECOY_NOT_INHERITED} (card C39).")


CALIBRATION_WORDS = {"control": "its file's control arm (inherited)", "own": "its own commits (no control ledger)",
                     "unrecorded": "unrecorded (ledgers kept before the field)"}


def _calibration_note(dc: dict) -> str:
    cal = (dc or {}).get("calibration")
    if not isinstance(cal, dict) or not cal:
        return ""
    return ("arm mass calibration: " + ", ".join(f"`{f}` {CALIBRATION_WORDS.get(k, k)}" for f, k in cal.items())
            + ". A decoy arm's own backbone is made of wrong readings; calibrated on it, an arm that fails to "
              "calibrate runs with the mass z-test and the degeneracy audit off.")


def run_side_channels(run: "Run") -> tuple:
    """The side channels the run's files opened (`batch_summary['side_channels']`, the union over its files),
    () when the summary records none -- a run made before the record existed, which on the measured runs is a
    closed run; its decoy arms keep the offline default (only the declared channels resolve)."""
    rec = (run.summary or {}).get("side_channels")
    if not isinstance(rec, (list, tuple)):
        return ()
    return tuple(str(a) for a in rec if isinstance(a, str) and a)


def run_engine_offline(run: Run, peaks: pd.DataFrame, sample_id: str, adducts: list[str], log=lambda *a: None,
                       scoring=None) -> pd.DataFrame:
    """`assign.run(peaks=)` on one table with the run's own profile settings,
    judged at `scoring` (`decoy_scoring`; None = the offline class fallback).
    Returns the ledger. Needs the local scorer (the default).

    An arm on the run's own channels (`adducts` = the run's: control, shift, ppm)
    opens the side channels the run recorded opening (`run_side_channels`), so it
    bounds the run as it was assigned; the wrong-adducts arm reads its wrong set
    alone. A run that recorded none (made before the record existed) opens none."""
    import copy

    from peaky.assignment import assign as A
    from peaky.assignment import passes as PA
    from peaky.chem import profiles as P
    from peaky.io import io_mascope as IO

    cfg = PA.PassConfig()
    P.apply_height_cutoff_x_edge(cfg, run.profile, log=log)
    P.apply_ion_only_channels(cfg, run.profile, log=log)
    own_chemistry = set(map(str, adducts)) == set(map(str, run.adducts or ()))
    P.apply_side_channels(cfg, run.profile, explicit=run_side_channels(run) if own_chemistry else (), log=log)
    # the batch's typical detection edge the run's tier pass sized its counting-
    # detector floor from (C46): an arm bounds the run AS TIERED, so it takes the
    # same footing (its own file's edge alone would put a low-count file's floor
    # 5x lower than the run's)
    _be = run.summary.get("noise_edge_batch_cps")
    try:
        cfg.noise_edge_batch_cps = float(_be) if _be is not None and float(_be) > 0 else None
    except (TypeError, ValueError):
        cfg.noise_edge_batch_cps = None
    kw = {"adducts": list(adducts), "reagent_n_relabel": False}
    model = run.summary.get("resolution")
    if isinstance(model, dict) and model.get("coef"):
        # the run's own width model, so a decoy arm carries the resolvability
        # stamp and its tier cap exactly as the run it bounds did
        from peaky.chem import resolution as RES
        kw["resolving_power"] = RES.Resolution.from_dict(model)
    if run.profile is not None:
        if getattr(run.profile, "label_isotope", None):
            kw["label_isotope"] = run.profile.label_isotope
            kw["label_max"] = run.profile.label_max
        if getattr(run.profile, "purity", None):
            kw["label_purity"] = run.profile.purity
    context = (run.profile.context if run.profile is not None else None) or run.summary.get("context") or "ambient-air"
    try:
        res = A.run(sample_id, context=context, cfg=copy.deepcopy(cfg), peaks=peaks, use_cache=False, log=log,
                    scoring=scoring, **kw)
    finally:
        IO.unregister_offline_sample(sample_id)
    return res["ledger"]


@contextlib.contextmanager
def inherited_calibration(control: pd.DataFrame | None, log=lambda *a: None):
    """While a decoy arm's engine runs, its mass calibration is the one its file's CONTROL arm gives: the
    pass-stage fit (`passes.calibrate`, the mass gate's mu / sigma and 1/mz trend) and the tier engine's
    (`tiers._calibrate`, read by the tiers, the degeneracy audit and the winner selection) are both taken on the
    control's ledger, not on the arm's own commits. An arm's own backbone is made of wrong formulas (a shifted
    spectrum) or wrong channels (the wrong adducts); when it is too small to calibrate, the arm runs with the
    mass z-test and the degeneracy audit OFF and keeps every mass fit -- it would bound an engine no run ever
    is. A real file always calibrates, so the arm is judged at the file's real instrument accuracy.

    Yields True when the calibration is inherited, False when there is no control ledger to take it from (the
    arm then calibrates on its own commits). Process-local: both functions are restored on exit."""
    if control is None or control.empty or "role" not in control.columns:
        yield False
        return
    from peaky.assignment import passes as PA
    from peaky.assignment import tiers as TI

    m0 = control[control["role"] == "M0"]
    kids = (control.loc[control["role"] == "iso_child", "parent_peak_id"].value_counts()
            if "parent_peak_id" in control.columns else pd.Series(dtype=int))
    own_pass, own_tier = PA.calibrate, TI._calibrate
    cache: dict = {}

    def pass_cal(_ledger, cfg, *, log=print):
        log("[calibrate] inherited: fitted on the control arm's ledger")
        return own_pass(control, cfg, log=log)

    def tier_cal(_m0, _kids_of, **kw):
        key = repr(sorted(kw.items()))
        if key not in cache:
            cache[key] = own_tier(m0, kids, **kw)
        return cache[key]

    PA.calibrate, TI._calibrate = pass_cal, tier_cal
    try:
        yield True
    finally:
        PA.calibrate, TI._calibrate = own_pass, own_tier


def calibration_summary(dc: dict) -> str | None:
    """One word for what a card's decoy arms were calibrated at: 'control' (every arm took its file's control
    calibration), 'own' (an arm's own commits), 'mixed', or 'unrecorded' (ledgers kept before the field)."""
    cal = (dc or {}).get("calibration")
    if not cal:
        return None
    kinds = set(cal.values()) if isinstance(cal, dict) else {str(cal)}
    return kinds.pop() if len(kinds) == 1 else "mixed"


CLAIM_COUNTS = ("pairs", "lt_350", "ge_350", "assigned", "assigned_lt_350", "assigned_ge_350")
IDENTIFIED_KEYS = ("identified", "identified_lt_350", "identified_ge_350",
                   "identified_assigned", "identified_assigned_lt_350", "identified_assigned_ge_350")
ESTABLISHED_KEYS = ("established", "established_lt_350")


def _claim_counts(m0: pd.DataFrame, tier: pd.Series, levels: pd.DataFrame) -> dict:
    """{claim: {pairs, lt_350, ge_350, assigned, ...}} in the level vector's
    unit: one per distinct (neutral_formula, adduct) M0 pair of any tier, with
    the pair's level (the one the level vector counts) and the m/z and tier of
    its brightest M0 row. A pair with no level reads tentative."""
    out = {c: dict.fromkeys(CLAIM_COUNTS, 0) for c in CLAIM_KEYS}
    if m0.empty:
        return out
    lv = level_map(levels)
    pairs = pd.DataFrame({
        "key": pair_keys(m0),
        "mz": pd.to_numeric(col(m0, "mz"), errors="coerce").values,
        "height": pd.to_numeric(col(m0, "height"), errors="coerce").values,
        "assigned": (tier == "Assigned").values,
    }).sort_values("height", ascending=False, kind="stable").drop_duplicates("key")
    pairs["claim"] = [claim_of(lv.get(k)) for k in pairs["key"]]
    for c, g in pairs.groupby("claim"):
        lt, ge = g["mz"] < DECOY_MZ_SPLIT, g["mz"] >= DECOY_MZ_SPLIT
        out[c] = {"pairs": int(len(g)), "lt_350": int(lt.sum()), "ge_350": int(ge.sum()),
                  "assigned": int(g["assigned"].sum()), "assigned_lt_350": int((g["assigned"] & lt).sum()),
                  "assigned_ge_350": int((g["assigned"] & ge).sum())}
    return out


def mz_bins(mz: pd.Series, width: int = DECOY_BIN_DA) -> dict:
    """{lower edge as text: n} of the finite m/z values, `width` Da bins (text keys: the card is JSON)."""
    v = pd.to_numeric(pd.Series(mz), errors="coerce")
    v = v[np.isfinite(v)]
    if v.empty:
        return {}
    lo = (np.floor(v.to_numpy(float) / width) * width).astype(int)
    vals, counts = np.unique(lo, return_counts=True)
    return {str(int(a)): int(b) for a, b in zip(vals, counts)}


def _add_bins(a: dict, b: dict) -> dict:
    out = dict(a or {})
    for k, n in (b or {}).items():
        out[k] = out.get(k, 0) + int(n)
    return out


def _levels_frame(pairs: pd.DataFrame | None) -> pd.DataFrame:
    """An arm's levelled pairs (`evidence.level_source`) as a levels frame (neutral, adduct, level)."""
    if pairs is None or pairs.empty:
        return pd.DataFrame(columns=LEVEL_FRAME)
    lv = pairs["evidence_level"].where(pairs["evidence_level"].astype(str) != "", np.nan)
    return pd.DataFrame({"neutral": pairs["neutral_formula"].astype(str).values,
                         "adduct": pairs["adduct"].astype(str).values, "level": lv.values,
                         "would_lift": col(pairs, "would_lift", "").values, "source": "arm"})


def _ledger_counts(led: pd.DataFrame, label: str, levels: pd.DataFrame | None = None,
                   strict: pd.DataFrame | None = None) -> dict:
    """One decoy arm's counts: the tier counts of its ledger and, per (neutral, adduct) pair, the claim of its
    level -- `levels` (the arm levelled in its run's context, adapted minima, as the reference levels it) and,
    in a field of its own, `strict` (the run's minima)."""
    m0 = led[led["role"] == "M0"] if "role" in led.columns else led.iloc[0:0]
    tier = col(m0, "tier", "")
    if len(m0):
        tier = tier.where(~ion_only_mask(m0), "")    # the ion-only bucket is not a tier count
    levels = levels if levels is not None else pd.DataFrame(columns=LEVEL_FRAME)
    mz = pd.to_numeric(col(m0, "mz"), errors="coerce")
    by_claim = _claim_counts(m0, tier, levels)
    ident = by_claim["identified"]
    est = {k: by_claim["identified"][k] + by_claim["neutral"][k] for k in CLAIM_COUNTS}
    out = {
        "m0": int(len(m0)),
        # what the arm committed below m/z 350 at all: an arm that commits few rows there has tested few gates
        "m0_lt_350": int((mz < DECOY_MZ_SPLIT).sum()),
        "assigned": int((tier == "Assigned").sum()),
        # the mass-defect gap a 0.35 Da shift lands in closes above ~m/z 350,
        # so a decoy's Assigned rows are reported on either side of it
        "assigned_lt_350": int(((tier == "Assigned") & (mz < DECOY_MZ_SPLIT)).sum()),
        "assigned_ge_350": int(((tier == "Assigned") & (mz >= DECOY_MZ_SPLIT)).sum()),
        # Assigned per m/z bin (key = the bin's lower edge, DECOY_BIN_DA wide)
        "assigned_by_bin": mz_bins(mz[tier == "Assigned"]),
        "candidate": int((tier == "Candidate").sum()),
        "neutrals": int(col(m0, "neutral_formula").dropna().nunique()),
        "levels": level_vector(levels),
        "examples": [
            f"{r.neutral_formula} {r.adduct} @ {float(r.mz):.4f}"
            for r in m0[tier == "Assigned"].sort_values("height", ascending=False).head(6).itertuples()
        ] if not m0.empty else [],
        # the claim per (neutral, adduct) pair of any tier, and the identified
        # class flat: the acceptance reads it (the Assigned-only variants beside)
        "by_claim": by_claim,
        "identified": ident["pairs"],
        "identified_lt_350": ident["lt_350"],
        "identified_ge_350": ident["ge_350"],
        "identified_assigned": ident["assigned"],
        "identified_assigned_lt_350": ident["assigned_lt_350"],
        "identified_assigned_ge_350": ident["assigned_ge_350"],
        # the neutral established (identified + neutral: 3c, 4a)
        "established": est["pairs"],
        "established_lt_350": est["lt_350"],
    }
    if strict is not None or levels.empty:
        strict = strict if strict is not None else pd.DataFrame(columns=LEVEL_FRAME)
        sb = _claim_counts(m0, tier, strict)
        out["strict"] = {"levels": level_vector(strict), "by_claim": sb, "identified": sb["identified"]["pairs"],
                         "identified_lt_350": sb["identified"]["lt_350"],
                         "established": sb["identified"]["pairs"] + sb["neutral"]["pairs"]}
    return out


#: the decoy arm's level modes: adapted (the reference's arm minima, 1) and strict (the run's minima)
ARM_MODES = ("adapted", "strict")


def _reparsed(led: pd.DataFrame) -> pd.DataFrame:
    """A ledger as a CSV round trip reads it back: an arm levelled from memory and the same arm re-counted from
    its kept file see the same values."""
    import io
    if led.empty:
        return led.copy()
    return pd.read_csv(io.StringIO(led.to_csv(index=False)), low_memory=False)


def level_arm(main_src, led: pd.DataFrame, file_id: str, arm: str, mode: str, wrong=(), control=None):
    """One decoy arm's ledger levelled in its run's context, exactly as `evidence.source_from_run_dir(<file>__<arm>
    .csv, main=<run dir>, mode=...)` levels a kept arm file: the run's window, gate, lists and scan; on the
    'adducts' arm the wrong adducts join the grid; the control arm's calibration where the run has none."""
    src = EV.source_from_frames({file_id: led}, run_inputs=EV.RunInputs(summary=main_src.summary), mode=mode,
                                name=f"{main_src.name}-{arm}-{mode}", main=main_src, arm=arm,
                                wrong_adducts=list(wrong) if arm == "adducts" else (),
                                control_ledger=led if arm == "control" else control)
    return EV.level_source(src)


#: beside the kept arm ledgers: what they were made with (mode, offset, files,
#: adduct sets, engine code), which a re-count reports instead of its command line
DECOY_MANIFEST = "manifest.json"


def arm_ledger_path(directory: str, file_id: str, arm: str) -> str:
    """Where a decoy arm's engine ledger is kept: `<dir>/<file>__<arm>.csv.gz`."""
    return os.path.join(directory, f"{file_id}__{arm}.csv.gz")


def decoy_ledgers_dir(base: str, run_name: str) -> str | None:
    """The directory holding a run's saved arm ledgers: `<base>/<run>/decoy/`
    (`base` is a scoreboard out dir), `<base>/decoy/`, or `base` itself; None
    when none of them holds a ledger."""
    for d in (os.path.join(base, run_name, "decoy"), os.path.join(base, "decoy"), base):
        if glob.glob(os.path.join(d, "*__*.csv.gz")):
            return d
    return None


def engine_code() -> str:
    """`<package version> <git commit>` of the engine the decoy arms run here
    (the shape of `Run.code`)."""
    import peaky
    from peaky.reporting import provenance as PV

    commit = (PV.git_info(os.path.dirname(peaky.__file__)) or {}).get("commit") or ""
    return f"{peaky.__version__} {commit[:9]}".strip()


#: the count keys a decoy arm's totals sum over its files
_SUM_KEYS = ("m0", "m0_lt_350", "assigned", "assigned_lt_350", "assigned_ge_350", "candidate", "neutrals")
#: the decoy modes and the arm families each runs (the control always runs)
DECOY_MODES = {"none": (), "shift": ("shift", "ppm"), "ppm": ("ppm",), "adducts": ("adducts",),
               "both": ("shift", "ppm", "adducts")}


def _total(items: list[dict]) -> dict:
    """One arm's counts summed over its files (or a ppm arm's over its files and shifts)."""
    total = {k: int(sum(i[k] for i in items)) for k in _SUM_KEYS}
    total["levels"] = {lv: int(sum(i["levels"].get(lv, 0) for i in items)) for lv in LEVELS}
    total["examples"] = items[0]["examples"]
    bins: dict = {}
    for i in items:
        bins = _add_bins(bins, i.get("assigned_by_bin"))
    total["assigned_by_bin"] = dict(sorted(bins.items(), key=lambda kv: int(kv[0])))
    if any("level_error" in i for i in items):
        total["level_error"] = "; ".join(i["level_error"] for i in items if "level_error" in i)
    total["by_claim"] = {c: {k: int(sum(i["by_claim"][c][k] for i in items)) for k in CLAIM_COUNTS} for c in CLAIM_KEYS}
    total.update({k: int(sum(i[k] for i in items)) for k in IDENTIFIED_KEYS + ESTABLISHED_KEYS})
    if all("strict" in i for i in items):
        st = [i["strict"] for i in items]
        total["strict"] = {
            "levels": {lv: int(sum(x["levels"].get(lv, 0) for x in st)) for lv in LEVELS},
            "by_claim": {c: {k: int(sum(x["by_claim"][c][k] for x in st)) for k in CLAIM_COUNTS} for c in CLAIM_KEYS},
            **{k: int(sum(x[k] for x in st)) for k in ("identified", "identified_lt_350", "established")}}
    # the wrong-adducts arm's ion level (`ion_level_counts`)
    if all("assigned_new_ion" in i for i in items):
        total.update({k: int(sum(i[k] for i in items)) for k in ION_LEVEL_KEYS})
        total["new_ion_examples"] = [t for i in items for t in i.get("new_ion_examples", [])][:6]
    return total


#: the wrong-adducts arm's ion-level counts
ION_LEVEL_KEYS = ("assigned_same_ion", "assigned_new_ion", "assigned_new_ion_lt_350")


def _ion_rates(a: dict, ctrl: dict) -> None:
    """The wrong-adducts arm's ion-level rate (in place): its Assigned NEW ions against the control's Assigned,
    beside the reading-level `assigned_rate`; and the share of its Assigned that only re-split an ion the control
    already reads."""
    if "assigned_new_ion" not in a:
        return
    a["new_ion_rate"] = pct(a["assigned_new_ion"], ctrl["assigned"])
    a["new_ion_lt_350_rate"] = pct_or_none(a["assigned_new_ion_lt_350"], ctrl["assigned_lt_350"])
    a["same_ion_share"] = pct_or_none(a["assigned_same_ion"], a["assigned"])


def _rates(a: dict, ctrl: dict) -> None:
    """An arm's rates against the control it is matched with (in place)."""
    a["assigned_rate"] = pct(a["assigned"], ctrl["assigned"])
    # the two sides of m/z 350, each against the control's Assigned on the same side (None = the control has none)
    a["assigned_lt_350_rate"] = pct_or_none(a["assigned_lt_350"], ctrl["assigned_lt_350"])
    a["assigned_ge_350_rate"] = pct_or_none(a["assigned_ge_350"], ctrl["assigned_ge_350"])
    a["candidate_rate"] = pct(a["candidate"], ctrl["candidate"])
    a["good_level_rate"] = pct(
        sum(a["levels"].get(lv, 0) for lv in GOOD_LEVELS),
        sum(ctrl["levels"].get(lv, 0) for lv in GOOD_LEVELS),
    )
    # per claim, against the control's pairs of the same claim (and m/z side);
    # a control with no pair of that claim leaves the rate undefined (None), not 0 %
    a["identified_rate"] = pct_or_none(a["identified"], ctrl["identified"])
    a["identified_lt_350_rate"] = pct_or_none(a["identified_lt_350"], ctrl["identified_lt_350"])
    a["identified_ge_350_rate"] = pct_or_none(a["identified_ge_350"], ctrl["identified_ge_350"])
    a["identified_assigned_lt_350_rate"] = pct_or_none(a["identified_assigned_lt_350"], ctrl["identified_assigned_lt_350"])
    a["established_rate"] = pct_or_none(a["established"], ctrl["established"])
    a["established_lt_350_rate"] = pct_or_none(a["established_lt_350"], ctrl["established_lt_350"])
    for c in ("neutral", "ion", "tentative", "not assessed"):
        a[f"{c.replace(' ', '_')}_rate"] = pct_or_none(a["by_claim"][c]["pairs"], ctrl["by_claim"][c]["pairs"])
    if "strict" in a and "strict" in ctrl:
        a["strict"]["identified_rate"] = pct_or_none(a["strict"]["identified"], ctrl["strict"]["identified"])
        a["strict"]["established_rate"] = pct_or_none(a["strict"]["established"], ctrl["strict"]["established"])


def decoy_bins(dc: dict) -> list[dict]:
    """Decoy against control Assigned per DECOY_BIN_DA m/z bin: the 0.35 Da arm against the control, the ppm arms
    (pooled) against the control counted once per ppm arm that ran on the file. A bin no arm holds is left out."""
    ctrl = dc.get("control") if isinstance(dc.get("control"), dict) and "error" not in dc["control"] else None
    if not ctrl or "assigned_by_bin" not in ctrl:
        return []
    sh = dc.get("shift") if isinstance(dc.get("shift"), dict) and "error" not in dc["shift"] else None
    pp = dc.get("ppm") if isinstance(dc.get("ppm"), dict) and "error" not in dc["ppm"] else None
    pc = (pp or {}).get("control") or {}
    keys = set(ctrl["assigned_by_bin"]) | set((sh or {}).get("assigned_by_bin") or {}) | set((pp or {}).get("assigned_by_bin") or {})
    rows = []
    for k in sorted(keys, key=int):
        c = int(ctrl["assigned_by_bin"].get(k, 0))
        rec = {"bin": f"{int(k)}-{int(k) + DECOY_BIN_DA}", "control": c}
        if sh is not None:
            s_ = int((sh.get("assigned_by_bin") or {}).get(k, 0))
            rec.update(shift=s_, shift_rate=pct_or_none(s_, c))
        if pp is not None:
            p_, pcn = int((pp.get("assigned_by_bin") or {}).get(k, 0)), int((pc.get("assigned_by_bin") or {}).get(k, 0))
            rec.update(ppm=p_, ppm_control=pcn, ppm_rate=pct_or_none(p_, pcn))
        rows.append(rec)
    return rows


def decoy_headline(dc: dict) -> dict | None:
    """The shift arm the card quotes below m/z 350: the ppm arms (pooled) when any ran, else the 0.35 Da arm. On an
    Orbitrap the 0.35 Da arm is blind below ~350 (its lines land in the empty mass-defect gap, no formula is
    proposed, no tier gate is tested), so its rate there is no bound."""
    pp = dc.get("ppm") if isinstance(dc.get("ppm"), dict) and "error" not in dc["ppm"] else None
    sh = dc.get("shift") if isinstance(dc.get("shift"), dict) and "error" not in dc["shift"] else None
    ctrl = dc.get("control") if isinstance(dc.get("control"), dict) and "error" not in dc["control"] else None
    if pp is not None and pp.get("control"):
        a, c, arm = pp, pp["control"], "ppm"
        label = "ppm shift " + "/".join(f"{k:+g}" for k in dc.get("ppm_k") or []) + " ppm, pooled"
    elif sh is not None and ctrl is not None:
        a, c, arm = sh, ctrl, "shift"
        label = f"shift {dc.get('offset_da', 0):+.2f} Da" + (" (blind below ~m/z 350 on an Orbitrap)" if dc.get("orbitrap") else "")
    else:
        return None
    return {"arm": arm, "label": label,
            "assigned_lt_350": a["assigned_lt_350"], "control_assigned_lt_350": c["assigned_lt_350"],
            "rate_lt_350": pct_or_none(a["assigned_lt_350"], c["assigned_lt_350"]),
            "assigned_ge_350": a["assigned_ge_350"], "control_assigned_ge_350": c["assigned_ge_350"],
            "rate_ge_350": pct_or_none(a["assigned_ge_350"], c["assigned_ge_350"]),
            "identified_lt_350_rate": a.get("identified_lt_350_rate"),
            # how much the arm committed below 350 at all, against the control: an arm that commits few rows there
            # (the Da arm in the mass-defect gap) has tested few gates, whatever its rate
            "m0_lt_350": a.get("m0_lt_350"), "control_m0_lt_350": c.get("m0_lt_350")}


def decoy(run: Run, mode: str, offset_da: float, n_files: int, log=lambda *a: None,
          save_dir: str | None = None, ledgers_dir: str | None = None, ppm_k=()) -> dict:
    """The decoy false-discovery bound: the engine, offline, on the brightest
    cover file(s) as they are (the control), with every m/z shifted, and with
    the wrong adduct set. What is still Assigned is the error bound; each arm's
    pairs are levelled in the run's context as the reference levels a decoy arm
    (`level_arm`: adapted minima; strict in a field of its own).

    Two shift arms: `shift` adds `offset_da` to every m/z (0.35 Da, kept for continuity; on an Orbitrap it lands
    in the empty mass-defect gap below ~m/z 350 and tests nothing there), and one `ppm` arm per k of `ppm_k`
    scales every m/z by (1 + k * 1e-6) -- outside the file's match window, inside the populated band
    (`decoy_ppm_peaks`). A k inside the window (`arm_window_ppm`) is skipped and says why. Every arm runs at
    the file's control calibration (`inherited_calibration`).

    `save_dir` keeps each arm's engine ledger (`arm_ledger_path`) and what they
    were made with (`DECOY_MANIFEST`); `ledgers_dir` counts the ledgers kept
    there instead of running the engine again -- the same counts at no engine
    cost (a missing ledger is that arm's error), with the offset, ppm shifts, files and
    adduct sets of the manifest when there is one."""
    if mode == "none" or run.per_file.empty:
        return {"mode": mode, "files": [], "control": None, "shift": None, "adducts": None}
    # a mode with no arm family of its own (the API's "control") runs the control alone, as it always did
    families = DECOY_MODES.get(mode, ())
    kept = _read_json(os.path.join(ledgers_dir, DECOY_MANIFEST)) if ledgers_dir else {}
    files = kept["files"] if "files" in kept else brightest_files(run, n_files)
    offset_da = kept.get("offset_da", offset_da)
    # a re-count reads the shifts the kept ledgers were made at (none before the ppm arm existed)
    ppm_k = [float(k) for k in (kept.get("ppm_k", []) if ledgers_dir else ppm_k)] if "ppm" in families else []
    out = {"mode": mode, "offset_da": offset_da, "files": files, "control": None, "shift": None, "adducts": None,
           "adducts_used": kept.get("adducts_used", run.adducts), "wrong_adducts": kept.get("wrong_adducts", wrong_adducts_for(run)),
           "ledgers": {"source": "saved" if ledgers_dir else "engine", "dir": ledgers_dir or save_dir,
                       "code": kept.get("code") if ledgers_dir else engine_code()},
           "ppm_k": ppm_k, "ppm": None, "ppm_arms": {},
           # per arm key, why it did not run on which file (a ppm shift inside the file's match window)
           "ppm_skipped": dict(kept.get("ppm_skipped") or {}) if ledgers_dir else {}}
    out["instrument"] = {f: decoy_instrument(run, f) for f in files}
    # a run's own channel in the wrong set (ledgers kept before the arm left them out): its rate is no error rate
    out["wrong_adducts_own"] = sorted(set(out["wrong_adducts"]) & own_adducts(run))
    if "adducts" in families and not out["wrong_adducts"]:
        out["adducts_skipped"] = (f"every wrong-adduct candidate {wrong_adducts(run.polarity)} is one of the run's "
                                  "own channels")
    out["orbitrap"] = any(v == "orbi" for v in out["instrument"].values())
    if kept:
        log(f"[decoy] re-count of the ledgers made at {offset_da:+.3f} Da"
            + (f" and {'/'.join(f'{k:+g}' for k in ppm_k)} ppm" if ppm_k else "") + f" on {files} by {kept.get('code')}")
    # per arm key: [(file, counts)]
    agg: dict[str, list] = {"control": [], "shift": [], "adducts": [], **{ppm_arm(k): [] for k in ppm_k}}
    errors: dict[str, str] = {}
    saved: list[str] = []
    ctx: dict = {}            # the run's source (built once, at the first arm) and each file's control ledger
    # per file, what its arms were judged at (attempted: an arm that errors keeps its file's label) and the inherited
    # numbers -- on a TOF the brightest file can be the batch's worst-fitted one
    out["scoring"] = (kept.get("scoring") or {f: "unrecorded" for f in files} if ledgers_dir
                      else {f: ("inherited" if decoy_scoring(run, f) else "class-fallback") for f in files})
    # per file, the mass calibration its decoy arms ran at (`inherited_calibration`): its control arm's, or the
    # arm's own where the control gave none
    out["calibration"] = (kept.get("calibration") or {f: "unrecorded" for f in files} if ledgers_dir else {})
    if ledgers_dir:
        out["scoring_detail"] = kept.get("scoring_detail") or {}
    else:
        from mascope_tools.composition import PatternScoring

        out["scoring_detail"] = {}
        for f in files:
            snap = decoy_scoring(run, f)
            if snap:
                det = {k: snap.get(k) for k in ("sigma_ppm", "mu_ppm", "mz_tolerance_ppm", "abundance_floor",
                                                "fitted_anchors", "sigma_source")}
                if det["abundance_floor"] is None:          # an omitted floor is the library's: what the arm got
                    det["abundance_floor"] = PatternScoring().abundance_floor
                out["scoring_detail"][f] = det

    def arm(key: str, peaks: pd.DataFrame, file_id: str, adducts: list[str]) -> None:
        # a decoy arm that crashes the engine is a finding, not a reason to lose
        # the card: record it and carry on
        sample_id = f"{file_id}-{key}"
        try:
            if ledgers_dir:
                led = pd.read_csv(arm_ledger_path(ledgers_dir, file_id, key), low_memory=False)
                read = led
            elif key == "control":
                led = run_engine_offline(run, peaks, sample_id, adducts, log, scoring=decoy_scoring(run, file_id))
                read = _reparsed(led)
            else:
                with inherited_calibration(ctx.get(file_id), log) as inherited:
                    led = run_engine_offline(run, peaks, sample_id, adducts, log, scoring=decoy_scoring(run, file_id))
                read = _reparsed(led)
                if out["calibration"].get(file_id) != "own":
                    out["calibration"][file_id] = "control" if inherited else "own"
            if key == "control":
                ctx[file_id] = read
            lv, level_error = {m: None for m in ARM_MODES}, None
            if "role" in read.columns and (read["role"].astype(str) == "M0").any():
                # an arm the engine committed nothing on has no pair to level
                try:
                    if "main" not in ctx:
                        ctx["main"] = EV.source_from_run_dir(run.path)
                    lv = {m: _levels_frame(level_arm(ctx["main"], read, file_id, key, m, wrong=out["wrong_adducts"],
                                                     control=ctx.get(file_id))) for m in ARM_MODES}
                except Exception as exc:  # noqa: BLE001 - a levelling failure keeps the arm's tier counts
                    level_error = f"{type(exc).__name__}: {exc}"
                    log(f"[decoy] {sample_id}: {key} arm not levelled -- {level_error}")
            counts = _ledger_counts(led, sample_id, lv["adapted"], lv["strict"])
            if key == "adducts":
                # the arm at the ion level: a re-split of an ion the control reads is no new ion
                counts.update(ion_level_counts(read, ctx.get(file_id)))
            if level_error:
                counts["level_error"] = level_error
            agg[key].append((file_id, counts))
        except Exception as exc:  # noqa: BLE001 - anything the engine raises
            errors[key] = f"{type(exc).__name__}: {exc}"
            log(f"[decoy] {sample_id}: {key} arm failed -- {errors[key]}")
            return
        if save_dir and not ledgers_dir:
            try:
                os.makedirs(save_dir, exist_ok=True)
                led.to_csv(arm_ledger_path(save_dir, file_id, key), index=False)
                saved.append(key)
            except OSError as exc:
                log(f"[decoy] {sample_id}: ledger not kept -- {exc}")

    for f in files:
        peaks = raw_peaks_of(run, f)
        if peaks.empty:
            continue
        log(f"[decoy] {f}: {len(peaks)} peaks, control {'re-count' if ledgers_dir else 'run'}")
        arm("control", peaks, f, run.adducts)
        if "shift" in families:
            log(f"[decoy] {f}: shift {offset_da:+.3f} Da")
            arm("shift", decoy_peaks(peaks, offset_da), f, run.adducts)
        for k in ppm_k:
            key = ppm_arm(k)
            if ledgers_dir:
                if f in (out["ppm_skipped"].get(key) or {}):
                    continue
            else:
                window = arm_window_ppm(run, f)
                if abs(k) <= window:
                    why = (f"{k:+g} ppm is inside the {window:g} ppm match window: the true formula stays in "
                           f"reach, so the arm is no decoy")
                    out["ppm_skipped"].setdefault(key, {})[f] = why
                    log(f"[decoy] {f}: ppm shift {k:+g} skipped -- {why}")
                    continue
            log(f"[decoy] {f}: ppm shift {k:+g}")
            arm(key, decoy_ppm_peaks(peaks, k), f, run.adducts)
        if "adducts" in families and out["wrong_adducts"]:
            log(f"[decoy] {f}: wrong adducts {out['wrong_adducts']}")
            arm("adducts", peaks, f, list(out["wrong_adducts"]))
    if saved:
        manifest = {k: out[k] for k in ("mode", "offset_da", "ppm_k", "ppm_skipped", "files", "adducts_used",
                                        "wrong_adducts", "scoring", "scoring_detail", "calibration")}
        try:
            with open(os.path.join(save_dir, DECOY_MANIFEST), "w") as fh:
                json.dump(manifest | {"code": out["ledgers"]["code"]}, fh, indent=1, default=_json_default)
        except OSError as exc:
            log(f"[decoy] manifest not kept -- {exc}")
    ppm_keys = {ppm_arm(k): k for k in ppm_k}
    for key, err in errors.items():
        if key in ppm_keys:
            out["ppm_arms"][key] = {"error": err, "k": ppm_keys[key]}
        else:
            out[key] = {"error": err}
    for key, items in agg.items():
        if not items:
            continue
        total = _total([c for _f, c in items])
        if key in ppm_keys:
            out["ppm_arms"][key] = dict(total, k=ppm_keys[key])
        else:
            out[key] = total
    ctrl = out["control"] if out["control"] and "error" not in out["control"] else None
    for key in ("shift", "adducts"):
        if out[key] and ctrl and "error" not in out[key]:
            _rates(out[key], ctrl)
            if key == "adducts":
                _ion_rates(out[key], ctrl)
    # the ppm arms: each against the control of the files it ran on; pooled over every shift, against the control
    # counted once per arm that ran on the file
    ctrl_of = {f: c for f, c in agg["control"]}
    pooled, matched = [], []
    for key, k in ppm_keys.items():
        items = [(f, c) for f, c in agg[key] if f in ctrl_of]
        if not items:
            continue
        mine = _total([ctrl_of[f] for f, _c in items])
        _rates(out["ppm_arms"][key], mine)
        out["ppm_arms"][key]["control_assigned"] = mine["assigned"]
        pooled += [c for _f, c in items]
        matched += [ctrl_of[f] for f, _c in items]
    if pooled:
        out["ppm"] = _total(pooled)
        out["ppm"]["control"] = _total(matched)
        out["ppm"]["n_arms"] = len(pooled)
        _rates(out["ppm"], out["ppm"]["control"])
    elif ppm_k and any(isinstance(v, dict) and "error" in v for v in out["ppm_arms"].values()):
        out["ppm"] = {"error": "; ".join(v["error"] for v in out["ppm_arms"].values() if "error" in v)}
    out["bins"] = decoy_bins(out)
    out["headline"] = decoy_headline(out)
    return out


# --- falsification survival -------------------------------------------------
HETERO_LINES = {"S": "34S", "Cl": "37Cl", "Br": "81Br", "Si": "29Si"}


def falsification(run: Run) -> dict:
    """(d) the 13C-implied carbon count against the formula on every Assigned
    row with a measured 13C satellite; the heteroatom line where the formula
    demands one; and the time covariance of satellites and adduct pairs with
    their parent."""
    pf = run.per_file
    out = {"c13": None, "hetero": None, "iso_cov": None, "adduct_cov": None, "resolvability": None}
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
                    implied = heights[sat_pid] / float(r.height) / C13_PER_CARBON
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
        # the separability stamp (assignment/resolvability.py) on the Assigned rows,
        # and how many per-file rows the separability / satellite tier rules capped
        m0all = pf[pf["role"] == "M0"]
        if "resolvability" in m0all.columns and m0all["resolvability"].notna().any():
            cls = m0all.loc[m0all["tier"] == "Assigned", "resolvability"].dropna().astype(str).value_counts()
            reasons = col(m0all, "tier_reason", "").fillna("").astype(str)
            out["resolvability"] = {
                "assigned": {k: int(v) for k, v in cls.items()},
                "capped": {"blended": int(reasons.str.startswith(("blended peak", "unresolvable peak")).sum()),
                           "refuted": int(reasons.str.contains("refuted by its isotope envelope", regex=False).sum()),
                           "untestable": int(reasons.str.contains("untestable at this intensity", regex=False).sum())}}
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
    # the claim leads: the acceptance reads the identified class
    ("claim_identified", "identified rows", 0),
    ("claim_neutral", "neutral rows", 0),
    ("claim_ion", "ion rows", 0),
    ("claim_tentative", "tentative rows", 0),
    ("claim_not_assessed", "not-assessed rows", 0),
    ("claim_identified_signal", "identified signal %", 1),
    ("claim_neutral_signal", "neutral signal %", 1),
    ("claim_unmatched_signal", "unmatched signal %", 1),
    ("claim_assigned_tentative", "Assigned but tentative", 0),
    ("claim_candidate_identified", "Candidate but identified", 0),
    ("bright_m0_not_identified", "bright M0 not identified", 0),
    ("roster_identified", "roster identified", 0),
    ("roster_neutral_or_better", "roster neutral or better", 0),
    ("roster_misread_identified", "roster read as other (identified)", 0),
    ("decoy_shift_identified_rate", "decoy (shift) identified %", 1),
    ("decoy_shift_identified_lt_350_rate", "decoy (shift) identified below 350 %", 1),
    ("decoy_adducts_identified_rate", "decoy (adducts) identified %", 1),
    ("decoy_shift_established_rate", "decoy (shift) neutral established %", 1),
    ("m3_own_missing_identified", "M3 own missing (identified)", 0),
    ("assigned", "Assigned rows", 0),
    ("candidate", "Candidate rows", 0),
    ("ion_only", "ion-only rows", 0),
    ("neutrals", "distinct neutrals", 0),
    ("stamped_signal_share", "stamped signal %", 1),
    ("unstamped_merged", "merged rows unstamped", 0),
    ("bright_m0_not_assigned", "bright M0 not Assigned", 0),
    ("bright_unstamped", "unstamped in brightest 50", 0),
    ("good_levels", "rows at level 3c (identified)", 0),
    ("m1_families", "M1 families", 0),
    ("m1_signal_share", "M1 unstamped signal %", 1),
    ("roster_present", "roster present", 0),
    ("roster_assigned", "roster Assigned", 0),
    ("roster_misread", "roster read as other", 0),
    ("decoy_shift_rate", "decoy (shift) Assigned %", 1),
    ("decoy_adducts_rate", "decoy (adducts) Assigned %", 1),
    ("decoy_adducts_new_ion_rate", "decoy (adducts) new ions Assigned %", 1),
    # the populated-defect ppm-shift arms (pooled) and the shift arm the card quotes below m/z 350
    ("decoy_headline_lt_350_rate", "decoy (headline shift arm) Assigned below 350 %", 1),
    ("decoy_ppm_rate", "decoy (ppm shift) Assigned %", 1),
    ("decoy_ppm_identified_rate", "decoy (ppm shift) identified %", 1),
    ("c13_within_1", "13C carbon count within 1", 0),
    ("hetero_present", "heteroatom line present", 0),
    ("census_halogen", "Assigned with Cl/Br/F", 0),
    ("census_mass_only", "Assigned mass-only (TOF flag)", 0),
    ("m3_other_instrument_own_missing", "M3 own missing", 0),
]
#: the metrics that read the evidence scale: never diffed between rows of two scales (`claims_schema`)
SCALE_KEYS = frozenset({k for k, _l, _n in KEY_METRICS if k.startswith(("claim_", "decoy_shift_identified",
                                                                          "decoy_adducts_identified",
                                                                          "decoy_shift_established",
                                                                          "decoy_ppm_identified"))}
                       | {"bright_m0_not_identified", "roster_identified", "roster_misread_identified",
                          "roster_neutral_or_better", "m3_own_missing_identified", "good_levels",
                          "m3_other_instrument_missing", "m3_other_instrument_own_missing"})
#: the metrics M2's presence test counts: never diffed between rows of two presence tests (`roster_test`)
ROSTER_KEYS = frozenset({"roster_present", "roster_assigned", "roster_misread", "roster_unstamped", "roster_same_ion",
                         "roster_identified", "roster_neutral", "roster_neutral_or_better", "roster_misread_identified"})


def row_scale(row: dict) -> str:
    """The evidence scale a board row's levels and claims read: its `scale` key, else its `claims_schema`'s
    (a row written before the key), else 'before the claim'."""
    if row.get("scale"):
        return str(row["scale"])
    cs = row.get("claims_schema")
    return "before the claim" if cs is None else SCALE_NAME.get(cs, str(cs))


def same_scale(row: dict, prev: dict | None) -> bool:
    """Two board rows read the same evidence scale (`row_scale`; none = before the claim)."""
    return prev is not None and row_scale(row) == row_scale(prev)


def ppm_shifts(row: dict) -> str:
    """The ppm shifts a board row's ppm arms ran at, as one word ('+9/-9'; 'none' for a row without them)."""
    ks = row.get("decoy_ppm_k")
    if not ks:
        return "none"
    try:
        return "/".join(f"{k:+g}" for k in sorted({float(k) for k in ks}, reverse=True))
    except (TypeError, ValueError):
        return str(ks)


def comparable(row: dict, prev: dict | None, key: str) -> tuple[bool, str]:
    """(True, '') when the metric `key` was measured alike on both rows; else (False, why): a level-scale change
    for a metric that reads the scale, a change of M2's presence test for a roster count, a change of what the
    decoy arms were calibrated at (control vs own; a row before the field ran its arms on their own) or scored at
    (`decoy_scoring`) for a decoy rate; for the headline below m/z 350 a change of the arm it quotes
    (`decoy_headline_arm`: the ppm arms or the Da arm, blind there on an Orbitrap), and for a ppm-arm rate (the
    headline when it quotes the ppm arms) a change of the shifts they ran at (`decoy_ppm_k`)."""
    if prev is None:
        return False, "no previous row"
    why = []
    if key in SCALE_KEYS and not same_scale(row, prev):
        why.append(f"level scale {row_scale(prev)} -> {row_scale(row)}")
    if key in ROSTER_KEYS and (row.get("roster_test") or 1) != (prev.get("roster_test") or 1):
        why.append(f"roster presence test {prev.get('roster_test') or 1} -> {row.get('roster_test') or 1}")
    if key.startswith("decoy_"):
        for field, before in (("decoy_calibration", "own"), ("decoy_scoring", "unrecorded")):
            a, b = prev.get(field) or before, row.get(field) or before
            if a != b:
                why.append(f"{field.replace('_', ' ')} {a} -> {b}")
    arm_a, arm_b = prev.get("decoy_headline_arm") or "none", row.get("decoy_headline_arm") or "none"
    if key.startswith("decoy_headline") and arm_a != arm_b:
        why.append(f"decoy headline arm {arm_a} -> {arm_b}")
    if key.startswith("decoy_ppm") or (key.startswith("decoy_headline") and arm_a == arm_b == "ppm"):
        a, b = ppm_shifts(prev), ppm_shifts(row)
        if a != b:
            why.append(f"decoy ppm shifts {a} -> {b}")
    return (not why), "; ".join(why)


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


def _num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and np.isfinite(v)


def delta(row: dict, prev: dict | None) -> list[dict]:
    """Each key metric against the previous row. A metric measured differently on the two rows (`comparable`: a
    level-scale change, a new presence test, a new decoy calibration) has no previous value and no delta; where
    both rows hold a number it says so in `note` ('not comparable: ...'), so 20 -> 2 across a scale change never
    reads as a regression."""
    out = []
    for key, label, nd in KEY_METRICS:
        now = row.get(key)
        ok, why = comparable(row, prev, key) if prev else (False, "")
        raw = prev.get(key) if prev else None
        was = raw if ok else None
        d = now - was if _num(now) and _num(was) else None
        rec = {"metric": label, "prev": was, "now": now, "delta": d, "nd": nd}
        if prev and not ok and _num(now) and _num(raw):
            rec["note"] = f"not comparable: {why} (previous {fmt(raw, nd)})"
        out.append(rec)
    return out


# ---------------------------------------------------------------------------
# the card: every panel, one dict
# ---------------------------------------------------------------------------
def build_card(run: Run, *, levels_csv=None, other=None, other_instrument=None, rosters=None,
               decoy_mode="none", decoy_offset=0.35, decoy_files=1, overlap=None, masks=(),
               floor_cps=10.0, floor_share=0.8, board=None, log=print,
               decoy_save_dir=None, decoy_ledgers=None, decoy_ppm=DECOY_PPM) -> dict:
    log(f"[scorecard] {run.name}: {run.reagent} {run.path_kind}, {run.n_spectra} spectra")
    ions = ion_table(run)
    tracks = unstamped_tracks(run)
    corroborate = [other_instrument.path] if other_instrument is not None else []
    levels = levels_for(run, levels_csv, corroborate, log=log)
    other_levels = other_own = None
    if other_instrument is not None:
        other_levels = levels_for(other_instrument, None, [run.path], log=log)
        other_own = own_levels_for(other_instrument, log=log)
    rosters = load_rosters() if rosters is None else rosters
    head = headline(run, ions)
    card = {
        "scorecard_version": __version__,
        "written_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "run_dir": run.path,
        "headline": head,
        "claims": claims(run, levels),
        "brightest": brightest(run, ions, tracks, levels),
        "evidence": best_evidence(run, ions, levels),
        "m1": missed_m1(run, ions, tracks, coverage_rows=head["stamp_coverage"].get("rows")),
        "m2": missed_m2(run, ions, tracks, rosters, levels),
        "m3": missed_m3(run, other, other_instrument, other_levels, overlap, list(masks), floor_cps, floor_share,
                        own_levels=other_own),
        "census": census(run),
        "decoy": decoy(run, decoy_mode, decoy_offset, decoy_files, log=log,
                       save_dir=decoy_save_dir, ledgers_dir=decoy_ledgers, ppm_k=decoy_ppm),
        "falsification": falsification(run),
        "rosters_unreviewed": True,
    }
    card["row"] = board_row(card)
    card["acceptance"] = acceptance(card["row"])
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
        "ion_only": h.get("ion_only", 0),
        "neutrals": h["neutrals"],
        "unstamped_merged": h["stamp_coverage"].get("n_unstamped"),
        "bright_m0_not_assigned": b["m0_not_assigned"],
        "bright_unstamped": b["unstamped_in_top"],
        "levels": e["levels"],
        "good_levels": sum(e["levels"].get(k, 0) for k in GOOD_LEVELS),
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
        "m3_other_instrument_own_missing": ((card["m3"].get("other_instrument_own") or {}).get("n_missing")),
    }
    # the claim: appended after every older key; `claims_schema` names the scale they read
    cl = card.get("claims") or {}
    rows_c = cl.get("rows") or {}
    sig = cl.get("signal") or {}
    share, share_a = sig.get("share") or {}, sig.get("share_assigned") or {}
    dis = cl.get("disagree") or {}
    rc = {}
    for r in (m2.get("roster_claim") or {}).values():
        for k, v in r.items():
            rc[k] = rc.get(k, 0) + v
    ctrl, shift, add = (dc.get(k) or {} for k in ("control", "shift", "adducts"))
    own = card["m3"].get("other_instrument_own") or {}
    row.update({
        "claim_identified": rows_c.get("identified"),
        "claim_neutral": rows_c.get("neutral"),
        "claim_ion": rows_c.get("ion"),
        "claim_tentative": rows_c.get("tentative"),
        "claim_reagent": rows_c.get("reagent"),
        "claim_not_assessed": rows_c.get(EV.CLAIM_NA),
        "claim_identified_signal": share.get("identified"),
        "claim_neutral_signal": share.get("neutral"),
        "claim_ion_signal": share.get("ion"),
        "claim_tentative_signal": share.get("tentative"),
        "claim_reagent_signal": share.get("reagent"),
        "claim_not_assessed_signal": share.get(EV.CLAIM_NA),
        "claim_unmatched_signal": share.get("unmatched"),
        "claim_assigned_identified_signal": share_a.get("identified"),
        "claim_assigned_neutral_signal": share_a.get("neutral"),
        "claim_assigned_ion_signal": share_a.get("ion"),
        "claim_assigned_tentative": dis.get("assigned_tentative"),
        "claim_candidate_identified": dis.get("candidate_identified"),
        "claim_stamp_mismatch": cl.get("stamp_mismatch"),
        "claim_level_source": cl.get("level_source"),
        "bright_m0_not_identified": b.get("m0_not_identified"),
        "bright_m0_not_assessed": b.get("m0_not_assessed"),
        "roster_identified": rc.get("identified"),
        "roster_neutral": rc.get("neutral"),
        "roster_misread_identified": rc.get("misread_identified"),
        "decoy_control_identified": ctrl.get("identified"),
        "decoy_control_identified_lt_350": ctrl.get("identified_lt_350"),
        "decoy_control_established": ctrl.get("established"),
        "decoy_shift_identified": shift.get("identified"),
        "decoy_shift_identified_lt_350": shift.get("identified_lt_350"),
        "decoy_shift_identified_rate": shift.get("identified_rate"),
        "decoy_shift_identified_lt_350_rate": shift.get("identified_lt_350_rate"),
        "decoy_shift_established_rate": shift.get("established_rate"),
        "decoy_shift_strict_identified_rate": (shift.get("strict") or {}).get("identified_rate"),
        "decoy_adducts_identified": add.get("identified"),
        "decoy_adducts_identified_rate": add.get("identified_rate"),
        "decoy_adducts_identified_lt_350_rate": add.get("identified_lt_350_rate"),
        "decoy_adducts_established_rate": add.get("established_rate"),
        "m3_own_missing_identified": (own.get("n_missing_by_claim") or {}).get("identified"),
        "m3_own_basis": own.get("basis"),
        "claims_schema": CLAIMS_SCHEMA,
    })
    # what the decoy arms were judged at (C35): appended, so the board's older columns keep their order
    row["decoy_scoring"] = decoy_scoring_summary(dc)
    # and the mass calibration they ran at: the control's (inherited) or their own
    row["decoy_calibration"] = calibration_summary(dc)
    # the populated-defect ppm-shift arms (pooled over their shifts and files), the two sides of m/z 350 of the
    # 0.35 Da arm, and the shift arm the card quotes below 350 (ppm when it ran; the 0.35 Da arm is blind there on
    # an Orbitrap)
    ppm = dc.get("ppm") if isinstance(dc.get("ppm"), dict) and "error" not in dc["ppm"] else {}
    hl = dc.get("headline") or {}
    row.update({
        "decoy_ppm_k": dc.get("ppm_k"),
        "decoy_ppm_rate": ppm.get("assigned_rate"),
        "decoy_ppm_lt_350_rate": ppm.get("assigned_lt_350_rate"),
        "decoy_ppm_ge_350_rate": ppm.get("assigned_ge_350_rate"),
        "decoy_ppm_identified_rate": ppm.get("identified_rate"),
        "decoy_ppm_identified_lt_350_rate": ppm.get("identified_lt_350_rate"),
        "decoy_ppm_established_rate": ppm.get("established_rate"),
        "decoy_ppm_control_assigned": (ppm.get("control") or {}).get("assigned"),
        "decoy_shift_lt_350_rate": shift.get("assigned_lt_350_rate"),
        "decoy_shift_ge_350_rate": shift.get("assigned_ge_350_rate"),
        "decoy_headline_arm": hl.get("arm"),
        "decoy_headline_lt_350_rate": hl.get("rate_lt_350"),
        # the wrong-adducts arm at the ion level (new ions only) beside its reading-level `decoy_adducts_rate`
        "decoy_wrong_adducts": dc.get("wrong_adducts"),
        "decoy_adducts_new_ion_rate": add.get("new_ion_rate"),
        "decoy_adducts_new_ion_lt_350_rate": add.get("new_ion_lt_350_rate"),
        "decoy_adducts_same_ion_share": add.get("same_ion_share"),
    })
    # the roster block: identified (3c) beside neutral or better (3c + 4a), the presence test that counted it and
    # the level scale it reads -- a delta across either is not comparable (`comparable`)
    row.update({
        "roster_neutral_or_better": rc.get("neutral_or_better"),
        "roster_same_ion": roster.get("same_ion"),
        "roster_iso_reagent": roster.get("iso_reagent"),
        "roster_test": m2.get("test"),
        "roster_window_ppm": m2.get("window_ppm"),
        "scale": EV.SCALE_RELEASE,
    })
    # an Orbitrap decoy file: the Da arm's numbers below m/z 350 are blind there (`DA_ARM_BLIND`)
    row["decoy_orbitrap"] = bool(dc.get("orbitrap")) if dc.get("mode") not in (None, "none") else None
    # the TOF mass-only flag (None off a TOF, or on a run made before it): the flagged Assigned rows, and the
    # split at the run's threshold -- appended, so the board's older columns keep their order
    mo = cz.get("mass_only") or {}
    row["census_mass_only"] = mo.get("n_flagged")
    row["census_mass_only_split"] = ({k: mo.get(k) for k in ("threshold_mz", "n_assigned", "below", "at_or_above")}
                                     if mo else None)
    return row


#: what an Orbitrap row's Da-shift-arm number below m/z 350 is worth: its lines land in the empty mass-defect gap
DA_ARM_BLIND = "blind on an Orbitrap: no formula is proposed below ~m/z 350, the ppm arms carry the bound"

#: today's acceptance criteria on the evidence scale of peaky 0.10.0, read on the identified class (3c) and, where
#: the scale makes identified rare, the neutral established (3c + 4a); each keeps the metric it replaces beside it:
#: (criterion, key, the key it was read on)
ACCEPTANCE = [
    ("roster recall not lower (identified)", "roster_identified", "roster_assigned"),
    ("roster recall not lower (neutral or better)", "roster_neutral_or_better", "roster_assigned"),
    ("roster misreads not higher (identified)", "roster_misread_identified", "roster_misread"),
    ("decoy rate not higher (Da shift arm, identified)", "decoy_shift_identified_rate", "decoy_shift_rate"),
    ("decoy rate not higher (Da shift arm, identified, below m/z 350)", "decoy_shift_identified_lt_350_rate",
     "decoy_shift_rate"),
    ("decoy rate not higher (Da shift arm, neutral established)", "decoy_shift_established_rate", "decoy_shift_rate"),
    # the ppm arms beside their own Assigned rate; the headline is Assigned already and has no older metric
    ("decoy rate not higher (ppm shift arms, identified)", "decoy_ppm_identified_rate", "decoy_ppm_rate"),
    ("decoy rate not higher (headline shift arm, Assigned, below m/z 350)", "decoy_headline_lt_350_rate", None),
    ("decoy rate not higher (wrong-adducts arm, identified)", "decoy_adducts_identified_rate", "decoy_adducts_rate"),
    ("bright M0 not identified not higher", "bright_m0_not_identified", "bright_m0_not_assigned"),
    ("M1 families not worse", "m1_families", "m1_families"),
    ("cross-instrument agreement not lower (own-evidence M3)", "m3_other_instrument_own_missing",
     "m3_own_missing_identified"),
]


def acceptance(row: dict) -> list[dict]:
    """The acceptance criteria on a board row: the scale's value and the old one; the M3 criterion names its
    basis (level <= 4b, or the other instrument's Assigned rows where it is not assessed)."""
    out = []
    for c, k, ok in ACCEPTANCE:
        if k == "m3_other_instrument_own_missing" and row.get("m3_own_basis"):
            c = f"{c}: {row['m3_own_basis']}"
        if k == "decoy_shift_identified_lt_350_rate" and row.get("decoy_orbitrap"):
            c = f"{c[:-1]}; {DA_ARM_BLIND})" if c.endswith(")") else f"{c} ({DA_ARM_BLIND})"
        out.append({"criterion": c, "key": k, "value": row.get(k), "old_key": ok,
                    "old_value": row.get(ok) if ok else None})
    return out


# ---------------------------------------------------------------------------
# rendering: SCORECARD.md
# ---------------------------------------------------------------------------
def _p(v, nd=1) -> str:
    return "" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{nd}f}"


DASH = "—"


def _d(v, nd=0) -> str:
    """A cell: the number, or a dash where it is missing (a card or a board row
    from before the claim, an arm that did not run)."""
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return DASH
    return fmt(v, nd) if isinstance(v, (int, float, np.integer, np.floating)) else str(v)


CLAIM_TABLE_COLUMNS = [
    ("class", "class"), ("levels", "levels"), ("rows", "merged rows"), ("tiers", "Assigned / Candidate / ion-only"),
    ("signal", "signal %"), ("signal_assigned", "Assigned signal %"),
    ("decoy", "decoy pairs: control / shift / adducts"), ("shift_split", "shift below 350 / at or above"),
    ("adducts_split", "adducts below 350 / at or above"),
    ("ppm_split", "ppm shift below 350 / at or above"),
]
DISAGREE_COLUMNS = [
    ("kind", "disagreement"), ("mz", "m/z"), ("neutral", "neutral"), ("adduct", "adduct"), ("tier", "tier"),
    ("level", "level"), ("claim", "claim"), ("n_files", "files"), ("signal_share", "signal %"),
    ("would_lift", "what would lift it"), ("tier_reason", "tier reason"),
]


def claim_keys_of(cl: dict | None) -> list[str]:
    """The claim classes a card's claim block counts, in its own order: the scale's six on a card of the evidence
    scale of peaky 0.10.0, the three of the pre-0.10.0 scale on an older card."""
    rows = (cl or {}).get("rows") or {}
    return list(rows) if rows else list(CLAIM_KEYS)


def crosstab_columns(keys) -> list[tuple[str, str]]:
    return ([("tier", "tier")] + [(c, c) for c in keys] + [("total", "rows")]
            + [(f"{c}_sig", f"{c} signal %") for c in keys])


def claim_table(card: dict) -> list[dict]:
    """The class table: per claim its levels, merged rows (and by tier bucket),
    its share of the committed per-file M0 signal (all tiers / Assigned), and
    the decoy arms' (neutral, adduct) pairs of that claim; then the signal no
    merged row carries (`unmatched`)."""
    cl = card.get("claims") or {}
    dc = card.get("decoy") or {}
    bt = (cl.get("by_tier") or {}).get("rows") or {}
    sig = cl.get("signal") or {}
    share, share_a = sig.get("share") or {}, sig.get("share_assigned") or {}
    levels_of = cl.get("levels") or CLAIM_LEVELS

    def arm(key: str, c: str, field: str = "pairs"):
        a = dc.get(key)
        if not a:
            return None
        return "error" if "error" in a else ((a.get("by_claim") or {}).get(c) or {}).get(field)

    out = []
    for c in claim_keys_of(cl):
        out.append({
            "class": c, "levels": levels_of.get(c, CLAIM_LEVELS.get(c, "")), "rows": (cl.get("rows") or {}).get(c),
            "tiers": " / ".join(_d((bt.get(k) or {}).get(c)) for k in TIER_BUCKETS),
            "signal": share.get(c), "signal_assigned": share_a.get(c),
            "decoy": " / ".join(_d(arm(k, c)) for k in ("control", "shift", "adducts")),
            "shift_split": f"{_d(arm('shift', c, 'lt_350'))} / {_d(arm('shift', c, 'ge_350'))}",
            "adducts_split": f"{_d(arm('adducts', c, 'lt_350'))} / {_d(arm('adducts', c, 'ge_350'))}",
            "ppm_split": f"{_d(arm('ppm', c, 'lt_350'))} / {_d(arm('ppm', c, 'ge_350'))}",
        })
    out.append({"class": "unmatched", "levels": "per-file reading no merged row carries", "rows": None, "tiers": "",
                "signal": share.get("unmatched"), "signal_assigned": None, "decoy": "", "shift_split": "", "adducts_split": "",
                "ppm_split": ""})
    return out


def crosstab_rows(cl: dict) -> list[dict]:
    """Tier x claim: merged rows per tier bucket and claim, and their signal %."""
    bt = cl.get("by_tier") or {}
    keys = claim_keys_of(cl)
    out = []
    for k in TIER_BUCKETS:
        r = (bt.get("rows") or {}).get(k) or {}
        g = (bt.get("signal") or {}).get(k) or {}
        out.append({"tier": k, **{c: r.get(c, 0) for c in keys}, "total": int(sum(r.get(c, 0) for c in keys)),
                    **{f"{c}_sig": g.get(c) for c in keys}})
    return out


def corroboration_line(cl: dict) -> str:
    """A pre-0.10.0 card's split of its identified rows by the channel's own evidence (the scale has no
    corroboration axis: an other-source partner is a tag); '' on a card of the scale."""
    co = cl.get("corroboration")
    if not co:
        return ""
    src = ", ".join(co.get("sources") or []) or "the corroborating source"
    lv = ", ".join(f"{k} {v}" for k, v in (co.get("via_cross_own_levels") or {}).items())
    line = (f"identified {co['identified']} = {co['own_identified']} on this channel's own evidence (its per-file ledgers "
            f"levelled with no cross set) + {co['via_cross']} only with the corroborating source ({src})")
    if lv:
        line += f"; their own levels: {lv}"
    if co.get("own_5b_identified"):
        line += (f"; {co['own_5b_identified']} identified rows are 5b on their own evidence (a flag): "
                 + "; ".join(co.get("own_5b_rows") or []))
    return line


def acceptance_lines(card: dict) -> list[str]:
    out = []
    for a in card.get("acceptance") or []:
        nd = 1 if "rate" in a["key"] else 0
        out.append(f"- {a['criterion']}: `{a['key']}` **{_d(a['value'], nd)}**"
                   + (f" (`{a['old_key']}` {_d(a['old_value'], nd)})" if a.get("old_key") else " (no older metric)"))
    return out


CLAIM_SENTENCE = (
    "The claim is read off the level on the evidence scale of peaky " + EV.SCALE_RELEASE + ": identified = 3c (ion "
    "established, split pinned, a named context-list entry), neutral = 4a (split pinned and a positive fact), ion = "
    "4b (the ion composition; the split or the process open), tentative = 5a, 5b or no level; reagent (a reagent "
    "ion or cluster) and not assessed (NA: the instrument class is not assessed) are reported beside the claims, "
    "never folded into them.")


def render_claims_md(card: dict) -> list[str]:
    """§0 of SCORECARD.md: the claim beside the tier."""
    cl = card.get("claims")
    L = ["## 0. The claim — identified / neutral / ion / tentative", "",
         "acceptance reads the identified class (evidence level 3c); tier unchanged and kept", ""]
    if not cl:
        return L + ["*(card predates C13)*", ""]
    sig = cl.get("signal") or {}
    old = "neutral" not in claim_keys_of(cl)
    src = cl.get("level_source")
    L += [(("A card on the pre-0.10.0 scale (identified 1-4a, ion 4b-4d, tentative 5a, 5b or none). ") if old
           else CLAIM_SENTENCE + (f" Levels: {src}." if src else ""))
          + f" Merged rows {fmt(cl['n_rows'])}; signal = the committed per-file "
          f"M0 height ({fmt(sig.get('n_m0'))} readings), each reading taking its merged row's claim and tier; "
          f"{fmt(sig.get('n_unmatched'))} readings no merged row carries are the `unmatched` bucket. Decoy pairs are the "
          f"arms' distinct (neutral, adduct) M0 pairs of any tier, levelled in the run's context (adapted minima)."
          + (f" Stored `claim` cells differing from the level: **{cl['stamp_mismatch']}**." if cl.get("stamp_mismatch") is not None else ""), ""]
    L += md_table(claim_table(card), CLAIM_TABLE_COLUMNS, {"signal": 1, "signal_assigned": 1})
    L += ["", "Tier x claim (the two are separate verdicts and are not nested; neither is changed from the other):", ""]
    keys = claim_keys_of(cl)
    L += md_table(crosstab_rows(cl), crosstab_columns(keys), {f"{c}_sig": 1 for c in keys})
    dis = cl.get("disagree") or {}
    L += ["", f"Disagreeing rows: Assigned but tentative **{dis.get('assigned_tentative')}**, Candidate but identified "
          f"**{dis.get('candidate_identified')}** (brightest first, up to 40):", ""]
    rows = [dict(r, would_lift=str(r.get("would_lift") or r.get("level_reason") or "")[:100],
                 tier_reason=r.get("tier_reason", "")[:100]) for r in (dis.get("rows") or [])[:40]]
    L += md_table(rows, DISAGREE_COLUMNS, {"mz": 4, "signal_share": 3})
    co = corroboration_line(cl)
    if co:
        L += ["", f"**Corroboration (pre-0.10.0):** {co}"]
    L += ["", "**Acceptance** — read on the identified class; the metric each criterion was read on before, in parentheses:", ""]
    L += acceptance_lines(card) + [""]
    return L


DECOY_BIN_COLUMNS = [("bin", "m/z"), ("control", "control Assigned"), ("shift", "shift (Da) Assigned"),
                     ("shift_rate", "shift (Da) % of control"), ("ppm", "ppm shifts Assigned"),
                     ("ppm_control", "control x ppm arms"), ("ppm_rate", "ppm % of control")]


def decoy_arm_rows(dc: dict) -> list[tuple[str, str, dict | None]]:
    """(key, label, counts) of every arm the card shows, in order: control, the 0.35 Da shift (labelled blind
    below ~m/z 350 on an Orbitrap), each ppm shift, the ppm shifts pooled, the wrong adducts."""
    rows = [("control", "control (as is)", dc.get("control"))]
    if dc.get("shift"):
        rows.append(("shift", f"shift {dc.get('offset_da', 0):+.2f} Da"
                     + (" (blind below ~m/z 350 on an Orbitrap)" if dc.get("orbitrap") else ""), dc.get("shift")))
    for key, a in (dc.get("ppm_arms") or {}).items():
        rows.append((key, f"shift {a.get('k', 0):+g} ppm", a))
    if dc.get("ppm") and len(dc.get("ppm_arms") or {}) > 1:
        rows.append(("ppm", "ppm shifts pooled", dc.get("ppm")))
    if dc.get("adducts"):
        rows.append(("adducts", f"wrong adducts {dc.get('wrong_adducts')}", dc.get("adducts")))
    return rows


def decoy_skip_lines(dc: dict) -> list[str]:
    """One line per ppm shift that did not run on a file, with why."""
    out = []
    for key, per_file in (dc.get("ppm_skipped") or {}).items():
        for f, why in (per_file or {}).items():
            out.append(f"{key} on `{f}` not run: {why}")
    return out


def decoy_adducts_lines(dc: dict) -> list[str]:
    """The wrong-adducts arm at the ion level, beside its reading-level rate; and why it did not run, or which
    of its adducts the run reads itself."""
    out = []
    if dc.get("adducts_skipped"):
        out.append(f"wrong adducts not run: {dc['adducts_skipped']}")
    if dc.get("wrong_adducts_own"):
        out.append(f"the kept wrong-adducts ledgers include the run's own channel(s) {dc['wrong_adducts_own']}: "
                   "their reading-level rate is no error rate; read the ion level")
    a = dc.get("adducts")
    if isinstance(a, dict) and "error" not in a and "assigned_new_ion" in a:
        out.append(f"wrong adducts at the ion level: of {a['assigned']} Assigned, {a['assigned_same_ion']} "
                   f"({_d(a.get('same_ion_share'), 1)} %) carry the ion composition of the control's reading on the "
                   f"same peak (another split of an ion the run reads); **{a['assigned_new_ion']} new ions = "
                   f"{_d(a.get('new_ion_rate'), 1)} % of the control's Assigned** (below {DECOY_MZ_SPLIT:.0f}: "
                   f"{_d(a.get('new_ion_lt_350_rate'), 1)} %), against the reading-level {_d(a.get('assigned_rate'), 1)} %"
                   + (f"; brightest new ions: {'; '.join(a['new_ion_examples'])}" if a.get("new_ion_examples") else ""))
    return out


#: what the headline line says the arms were calibrated at, per `calibration_summary`
HEADLINE_CALIBRATION = {
    "control": "the arms at the control's calibration",
    "own": "the arms calibrated on their own commits: no control ledger",
    "unrecorded": "the arms' calibration unrecorded: ledgers kept before the field, calibrated on their own commits",
    "mixed": "the arms' calibration mixed: see the calibration note",
}


def decoy_headline_lines(dc: dict) -> list[str]:
    """The shift arm the card quotes below m/z 350, as one bold line (empty when no shift arm ran). It names the
    calibration the arms actually ran at (`calibration_summary`) and what the arm committed below 350 at all."""
    hl = dc.get("headline") or decoy_headline(dc)
    if not hl:
        return []
    cal = HEADLINE_CALIBRATION.get(calibration_summary(dc) or "")
    m0, cm0 = hl.get("m0_lt_350"), hl.get("control_m0_lt_350")
    return [f"**Shift decoy below m/z {DECOY_MZ_SPLIT:.0f}** ({hl['label']}" + (f", {cal}" if cal else "") + "): "
            f"{hl['assigned_lt_350']} Assigned against {hl['control_assigned_lt_350']} in the control = "
            f"**{_d(hl.get('rate_lt_350'), 1)} %**; at or above {DECOY_MZ_SPLIT:.0f}: {hl['assigned_ge_350']} / "
            f"{hl['control_assigned_ge_350']} = {_d(hl.get('rate_ge_350'), 1)} %."
            + (f" Below {DECOY_MZ_SPLIT:.0f} the arm commits {m0} M0 rows of any tier against the control's {cm0} "
               "(an arm that commits few there tests few gates)." if m0 is not None and cm0 is not None else "")
            + (" The 0.35 Da arm is kept for continuity; below ~m/z 350 on an Orbitrap it proposes almost nothing "
               "and bounds no tier gate." if hl["arm"] == "ppm" and dc.get("orbitrap") and dc.get("shift") else ""),
            ""]


def render_md(card: dict) -> str:
    h, b, e, m1, m2, m3, cz, dc, fz = (card[k] for k in ("headline", "brightest", "evidence", "m1", "m2", "m3", "census", "decoy", "falsification"))
    sc = h["stamp_coverage"]
    L = [f"# Scorecard — {h['run']}", "",
         f"channel `{h['channel']}` · reagent `{h['reagent']}` · path {h['path']} · code {h['code'] or '?'} · written {card['written_utc']}", ""]
    # 0 the claim
    L += render_claims_md(card)
    # 1 headline
    L += ["## 1. What was assigned", "",
          f"- spectra {fmt(h['n_spectra'])}, peaks {fmt(h['n_peaks'])}, assigned files {fmt(h['n_files'])}"
          + (f", {h['elapsed_s'] / 60:.1f} min" if h.get("elapsed_s") else ""),
          f"- stamped: {_p(h['stamped_peak_share'])} % of peaks, {_p(h['stamped_signal_share'])} % of signal "
          f"(M0 alone {_p(h['stamped_M0_signal_share'])} %); unstamped signal {_p(h['unstamped_signal_share'])} %",
          f"- merged rows {fmt(h['merged_rows'])}: Assigned {fmt(h['assigned'])}, Candidate {fmt(h['candidate'])}"
          + (f", ion-only {fmt(h['ion_only'])} (levels {h.get('ion_only_levels')})" if h.get("ion_only") else "") + "; "
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
          + f" · **bright M0 not Assigned: {b['m0_not_assigned']}** · bright M0 not identified: {_d(b.get('m0_not_identified'))}"
          + (f" (not assessed: {b['m0_not_assessed']})" if b.get("m0_not_assessed") else "")
          + f" · unstamped tracks among the batch's brightest {TOP_N}: **{b['unstamped_in_top']}** (named in M1)", ""]
    L += md_table(b["rows"], [("ion_mz", "ion m/z"), ("med_cps", "med cps"), ("in", "in"), ("role", "role"), ("ion", "ion"), ("neutral", "neutral"), ("adduct", "adduct"), ("tier", "tier"), ("level", "level"), ("claim", "claim"), ("ion_only", "ion-only")], {"ion_mz": 4})
    L += [""]
    # 3 best evidence
    vec = "/".join(str(e["levels"].get(k, 0)) for k in LEVELS)
    L += [f"## 3. The best-evidence {len(e['rows'])} rows", "",
          f"levels ({'/'.join(LEVELS)}): **{vec}** over {fmt(e.get('n_levelled', 0))} rows; identified (3c): {e['n_good']}; "
          f"neutral established (3c + 4a): {_d(e.get('n_established'))}", "",
          f"claims ({' / '.join(CLAIM_KEYS)}): " + " / ".join(_d((e.get("by_claim") or {}).get(c)) for c in CLAIM_KEYS), ""]
    L += md_table(e["rows"], [("neutral", "neutral"), ("adduct", "adduct"), ("ion_mz", "ion m/z"), ("med_cps", "med cps"), ("in", "in"), ("tier", "tier"), ("level", "level"), ("claim", "claim"), ("would_lift", "what would lift it")], {"ion_mz": 4})
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
    if m2.get("window_ppm") is not None:
        L += [f"A line counts within {m2['window_ppm']:.2f} ppm ({ROSTER_SIGMA_K:g} x the run's measured mass sigma, "
              f"{ROSTER_MIN_PPM:g} ppm to the run's tolerance) in >= {int(ROSTER_PRESENCE * 100)} % of the spectra, the "
              "best-read line in the window standing for the formula; an isotope satellite of another ion, or a reagent "
              "line of another composition, is no sighting (a reagent line of the formula's own ion composition is: the "
              "reagent's reference ions are read as themselves there); a line read as another split of the same ion is "
              f"`same ion`, not a misread (presence test {m2.get('test')}). Claims on the level scale "
              f"{m2.get('scale', '')}. The roster claims are per line read: the claim of the one line M2 picked for a "
              "formula, not its best claim over every channel it is read on.", ""]
    for name, r in (m2.get("roster") or {}).items():
        rc = (m2.get("roster_claim") or {}).get(name) or {}
        L.append(f"- roster `{name}` ({r['n']} formulas): present {r['present']}, Assigned as itself {r['assigned']}, Candidate {r['candidate']}, "
                 f"the same ion read as another split {r.get('same_ion', 0)}, read as something else {r['read_as']}, "
                 f"present but unstamped {r['unstamped']}, on an isotope / reagent line only {r.get('iso_reagent', 0)}, absent {r['absent']}"
                 + (f"; read as itself and identified (3c) {rc.get('identified', 0)}, neutral or better (3c + 4a) "
                    f"{rc.get('neutral_or_better', rc.get('identified', 0) + rc.get('neutral', 0))}" if rc else ""))
        for cls, rc in (m2.get("roster_by_class", {}).get(name) or {}).items():
            L.append(f"  - {cls}: {rc['assigned']} Assigned / {rc['present']} present / {rc['n']}")
    for src, r in (m2.get("sources") or {}).items():
        L.append(f"- `{src}` ({r['n']}): Assigned {r['assigned']}, Candidate {r['candidate']}, the same ion read as "
                 f"another split {r.get('same_ion', 0)}, read as other {r['read_as']}, unstamped {r['unstamped']}, on an "
                 f"isotope / reagent line only {r.get('iso_reagent', 0)}, absent {r['absent']}")
    L += ["", "Rows that are present but NOT assigned as themselves (the misses):", ""]
    miss_rows = [r for r in m2["rows"] if r["status"] in ("read as", "same ion", "unstamped", "candidate")]
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
        L += [f"- other instrument `{oi['run']}` ({oi['reagent']}): {oi['n_missing']} of its {oi['n_good']} rows ({oi.get('basis', M3_BASIS_LEVEL)}) pass the floor "
              f"({oi['floor']['cps']} cps median in {int(oi['floor']['share'] * 100)} % of {oi['n_spectra_in_window']} spectra"
              + (f", window {oi['window'][0][:16]} -> {oi['window'][1][:16]}" if oi.get("window") else "") + ") and are absent here", ""]
        L += md_table(oi["rows"], [("neutral", "neutral"), ("adduct", "other adduct"), ("level", "level"), ("med_cps", "med cps there"), ("share", "share %")], {"share": 0})
        L += [""]
    oo = m3.get("other_instrument_own")
    if oo:
        L += [f"- the same by the other instrument's OWN evidence (its files pooled with no other-source partners; its "
              f"in-core series can owe an anchor to this run): "
              f"{oo['n_missing']} of its {oo['n_good']} rows ({oo.get('basis', M3_BASIS_LEVEL)}) pass the floor and are absent here", ""]
        L += md_table(oo["rows"], [("neutral", "neutral"), ("adduct", "other adduct"), ("level", "own level"), ("med_cps", "med cps there"), ("share", "share %")], {"share": 0})
        L += [""]
    if not op and not oi:
        L += ["*(no --other / --other-instrument given)*", ""]
    # 5 is it right
    L += ["## 5. Is it right", "", "### (a) roster recall — see M2 above (rosters are UNREVIEWED; the user signs them off before the numbers are quoted)", "",
          f"### (b) element census of the {fmt(cz['n_assigned'])} Assigned rows", "",
          "| element | Assigned rows | examples |", "|---|---:|---|"]
    for k, v in cz["elements"].items():
        L.append(f"| {k} | {v} | {cz['examples'].get(k, '')} |")
    if cz.get("mass_only"):
        L += ["", f"- {mass_only_line(cz['mass_only'])}"]
    L += ["", "### (c) decoy false-discovery bound", ""]
    if _scoring_note(dc):
        L += [_scoring_note(dc), ""]
    if _calibration_note(dc):
        L += [_calibration_note(dc), ""]
    if dc.get("control") and "error" not in dc["control"]:
        L += [f"offline engine on the brightest {len(dc['files'])} cover file(s) `{', '.join(dc['files'])}`; control = the file as it is.", ""]
        L += decoy_headline_lines(dc)
        L += [f"| arm | M0 rows | Assigned | < {DECOY_MZ_SPLIT:.0f} / >= | Candidate | neutrals | level 3c | rate vs control (Assigned) "
              f"| identified pairs | identified < {DECOY_MZ_SPLIT:.0f} / >= | identified Assigned < {DECOY_MZ_SPLIT:.0f} "
              f"| identified rate | identified < {DECOY_MZ_SPLIT:.0f} rate | ion / tentative pairs "
              f"| Assigned rate < {DECOY_MZ_SPLIT:.0f} / >= |",
              "|---|---:|---:|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---|---|"]
        for key, label, a in decoy_arm_rows(dc):
            if a and "error" in a:
                L.append(f"| {label} | engine error: {a['error']} | | | | | | | | | | | | | |")
            elif a:
                good = sum(a["levels"].get(lv, 0) for lv in GOOD_LEVELS)
                bc = a.get("by_claim") or {}
                L.append(f"| {label} | {a['m0']} | {a['assigned']} | {a.get('assigned_lt_350', '')} / {a.get('assigned_ge_350', '')} | {a['candidate']} | {a['neutrals']} | {good} | {_p(a.get('assigned_rate'))} "
                         f"| {_d(a.get('identified'))} | {_d(a.get('identified_lt_350'))} / {_d(a.get('identified_ge_350'))} | {_d(a.get('identified_assigned_lt_350'))} "
                         f"| {_d(a.get('identified_rate'), 1)} | {_d(a.get('identified_lt_350_rate'), 1)} "
                         f"| {_d((bc.get('ion') or {}).get('pairs'))} / {_d((bc.get('tentative') or {}).get('pairs'))} "
                         f"| {_d(a.get('assigned_lt_350_rate'), 1)} / {_d(a.get('assigned_ge_350_rate'), 1)} |")
        L.append("")
        for why in decoy_skip_lines(dc):
            L.append(f"- {why}")
        L += [f"- {x}" for x in decoy_adducts_lines(dc)]
        for key in ("control", "shift", "ppm", "adducts"):
            a = dc.get(key)
            if a and "error" not in a:
                st = a.get("strict") or {}
                L.append(f"- {key}: levels ({'/'.join(LEVELS)}) {'/'.join(str(a['levels'].get(k, 0)) for k in LEVELS)}; "
                         f"neutral established (3c + 4a) {_d(a.get('established'))}"
                         + (f", rate {_d(a.get('established_rate'), 1)} %" if key != "control" else "")
                         + (f"; strict minima: {'/'.join(str(st['levels'].get(k, 0)) for k in LEVELS)}, identified "
                            f"{_d(st.get('identified'))}" if st else ""))
        for key in ("shift", "ppm", "adducts"):
            a = dc.get(key)
            if a and a.get("examples"):
                L.append(f"\n{key} arm, brightest decoy Assigned: " + "; ".join(a["examples"]))
        bins = dc.get("bins") or decoy_bins(dc)
        if bins:
            L += ["", f"Assigned by {DECOY_BIN_DA}-Da m/z bin (the ppm arms pooled, against the control counted once per "
                      "ppm arm that ran on the file):", ""]
            L += md_table(bins, DECOY_BIN_COLUMNS, {"shift_rate": 1, "ppm_rate": 1}, missing=DASH)
    else:
        L += [f"*(decoy mode `{dc.get('mode')}`: not run)*"]
    L += ["", "### (d) falsification survival", ""]
    c13 = fz.get("c13")
    if c13:
        L.append(f"- 13C carbon count on {c13['n']} Assigned rows with a measured satellite: within 1 carbon {c13['within_1']}, within max(1, 25 %) {c13['within_tol']}, median |delta| {c13['median_abs_delta']:.2f}"
                 + (f"; worst: {'; '.join(c13['worst'])}" if c13["worst"] else ""))
    rv = fz.get("resolvability")
    if rv:
        L.append(f"- separability of the Assigned peaks (per-file rows): {rv['assigned']}; per-file rows capped at "
                 f"Candidate by the separability / satellite rules: {rv['capped']}")
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
        # a metric measured differently on the two rows carries its note (and no delta)
        cols = [("metric", "metric"), ("prev", "previous"), ("now", "now"), ("delta", "delta")]
        if any(d.get("note") for d in card["delta"]):
            cols.append(("note", "note"))
        L += md_table([{"metric": d["metric"], "prev": d["prev"], "now": d["now"], "delta": d["delta"],
                        "note": d.get("note") or ""} for d in card["delta"]],
                      cols, {"prev": 1, "now": 1, "delta": 1}, missing=DASH)
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


CLAIM_BOARD_COLUMNS = [
    ("channel", "channel"), ("run", "run"), ("scale", "scale"), ("identified", "identified"), ("neutral", "neutral"),
    ("ion", "ion"), ("tentative", "tentative"), ("not_assessed", "not assessed"),
    ("identified_sig", "identified sig %"), ("neutral_sig", "neutral sig %"), ("ion_sig", "ion sig %"),
    ("tentative_sig", "tentative sig %"), ("unmatched_sig", "unmatched sig %"),
    ("assigned_tentative", "Assigned but tentative"), ("candidate_identified", "Candidate but identified"),
    ("bright", "bright M0 not identified"), ("roster", "roster identified / neutral or better / present / n"),
    ("misread", "roster misread identified"), ("dshift", "decoy shift identified % (below 350), headline arm"),
    ("dadd", "decoy adducts identified %"), ("m3", "M3 own missing"), ("basis", "vs previous row"),
]
#: what a board row's `claims_schema` reads
SCALE_NAME = {2: EV.SCALE_RELEASE, 1: "pre-0.10.0"}


def level_cell(r: dict) -> str:
    """A board row's level vector: the scale's on a schema-2 row, the pre-0.10.0 one (marked) on an older row."""
    lv = r.get("levels") or {}
    if r.get("claims_schema") == CLAIMS_SCHEMA:
        return "/".join(str(lv.get(k, 0)) for k in LEVELS)
    return "pre-0.10.0 " + "/".join(str(lv.get(k, 0)) for k in OLD_LEVELS)


def dshift_cell(r: dict) -> str:
    """The board's lead decoy cell: the identified rate (below m/z 350 in brackets) of the shift arm the row's
    headline quotes -- the ppm arms when they ran, named with their shifts -- else the Da arm's, marked blind
    below 350 on an Orbitrap row (a row before the headline shows the Da arm's, unmarked)."""
    if r.get("decoy_headline_arm") == "ppm":
        return (f"ppm {ppm_shifts(r)}: {_d(r.get('decoy_ppm_identified_rate'), 1)} "
                f"({_d(r.get('decoy_ppm_identified_lt_350_rate'), 1)})")
    s = f"{_d(r.get('decoy_shift_identified_rate'), 1)} ({_d(r.get('decoy_shift_identified_lt_350_rate'), 1)})"
    if r.get("decoy_headline_arm") == "shift":
        s = "Da arm: " + s
    return s + (" blind below 350 on an Orbitrap" if r.get("decoy_orbitrap") and r.get("decoy_shift_rate") is not None
                else "")


def claim_board_rows(board: list[dict]) -> list[dict]:
    """The claims table of the board: every channel's latest row, its claim
    metrics as text cells with the delta to the row before it on the same
    scale; a row written before the claim shows dashes, a pre-0.10.0 row its
    own numbers under its scale's name (dashes where the scale has no key)."""
    out = []
    for r in latest_rows(board):
        prev = previous_row(board, r["channel"], before=r)
        rec = {"channel": r["channel"], "run": r["run"][-24:]}
        if r.get("claims_schema") is None:
            out.append(rec | {k: DASH for k, _ in CLAIM_BOARD_COLUMNS[2:]})
            continue
        notes: list[str] = []

        def cell(key, nd=0):
            v = r.get(key)
            s = _d(v, nd)
            if prev is None or not _num(v) or not _num(prev.get(key)):
                return s
            ok, why = comparable(r, prev, key)
            if not ok:
                notes.extend(part for part in why.split("; ") if part not in notes)
                return s
            d = v - prev[key]
            if abs(d) >= (0.05 if nd else 1):
                s += f" ({d:+.{nd}f})"
            return s

        rec.update({
            "scale": SCALE_NAME.get(r.get("claims_schema"), str(r.get("claims_schema"))),
            "identified": cell("claim_identified"), "neutral": cell("claim_neutral"), "ion": cell("claim_ion"),
            "tentative": cell("claim_tentative"), "not_assessed": cell("claim_not_assessed"),
            "identified_sig": cell("claim_identified_signal", 1), "neutral_sig": cell("claim_neutral_signal", 1),
            "ion_sig": cell("claim_ion_signal", 1),
            "tentative_sig": cell("claim_tentative_signal", 1), "unmatched_sig": cell("claim_unmatched_signal", 1),
            "assigned_tentative": cell("claim_assigned_tentative"), "candidate_identified": cell("claim_candidate_identified"),
            "bright": cell("bright_m0_not_identified"),
            "roster": f"{cell('roster_identified')} / {cell('roster_neutral_or_better')} / {cell('roster_present')} / "
                      f"{_d(r.get('roster_n'))}",
            "misread": cell("roster_misread_identified"),
            "dshift": dshift_cell(r),
            "dadd": cell("decoy_adducts_identified_rate", 1),
            "m3": cell("m3_other_instrument_own_missing") if r.get("claims_schema") == CLAIMS_SCHEMA
            else cell("m3_own_missing_identified"),
        })
        # the decoy cell carries no delta, but a change of the arm or shifts it reads is a change of measure
        dkey = "decoy_ppm_identified_rate" if r.get("decoy_headline_arm") == "ppm" else "decoy_shift_identified_rate"
        if prev is not None and _num(r.get(dkey)) and any(_num(prev.get(k)) for k in (
                "decoy_ppm_identified_rate", "decoy_shift_identified_rate", "decoy_headline_lt_350_rate")):
            for k in (dkey, "decoy_headline_lt_350_rate"):
                ok, why = comparable(r, prev, k)
                if not ok:
                    notes.extend(part for part in why.split("; ") if part not in notes)
        rec["basis"] = ("first row" if prev is None else
                        ("not comparable: " + "; ".join(notes)) if notes else "same measure")
        out.append(rec)
    return out


def render_board_md(board: list[dict]) -> str:
    rows = latest_rows(board)
    L = ["# Peaky Scoreboard", "", f"regenerated {dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} · {len(rows)} channels · {len(board)} rows in `scoreboard.jsonl`", "",
         "One row per channel, its latest scorecard, and the delta to the row before it on the same scale. Levels "
         f"are {'/'.join(LEVELS)} (the evidence scale of peaky {EV.SCALE_RELEASE}); a row written on the pre-0.10.0 "
         f"scale shows its own vector ({'/'.join(OLD_LEVELS)}), marked.", ""]
    # the claim first: the acceptance reads the identified class (evidence level 3c)
    L += ["## Claims — acceptance reads the identified class", "",
          "identified = 3c (a named context-list entry on a pinned split), neutral = 4a (pinned + a positive fact), "
          "ion = 4b, tentative = 5a, 5b or none; not assessed = NA (the instrument class is not assessed). Signal % = "
          "share of the committed per-file M0 signal, `unmatched` = readings no merged row carries. The tier is kept "
          "beside it; a row written before the claim shows dashes, a pre-0.10.0 row its own claims (identified "
          "1-4a, ion 4b-4d) under its scale's name, never diffed against the scale's.", ""]
    L += ["| " + " | ".join(h for _, h in CLAIM_BOARD_COLUMNS) + " |",
          "|---|---|---|" + "---:|" * (len(CLAIM_BOARD_COLUMNS) - 3)]
    for rec in claim_board_rows(board):
        L.append("| " + " | ".join([rec["channel"].replace("|", " · ")] + [str(rec[k]) for k, _ in CLAIM_BOARD_COLUMNS[1:]]) + " |")
    L += ["", "## All metrics", ""]
    L += ["| channel | run | code | Assigned | Candidate | ion-only | neutrals | stamped signal % | unstamped merged | bright M0 not Assigned | unstamped in top 50 | levels | identified level | M1 fam. | M1 signal % | roster A/present/n | Cl+Br+F | decoy Da shift % | decoy adducts % | 13C ok/n | hetero ok/n | decoy headline < 350 % | decoy adducts new-ion % |",
          "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---|---:|---:|---:|---|---|---:|---:|"]
    for r in rows:
        prev = previous_row(board, r["channel"], before=r)

        def cell(key, nd=0):
            v = r.get(key)
            s = fmt(v, nd) if isinstance(v, (int, float)) else (v or "")
            if prev is not None and comparable(r, prev, key)[0] \
                    and isinstance(v, (int, float)) and isinstance(prev.get(key), (int, float)):
                d = v - prev[key]
                if abs(d) >= (0.05 if nd else 1):
                    s += f" ({d:+.{nd}f})"
            return s

        L.append("| " + " | ".join([
            r["channel"].replace("|", " · "), r["run"][-24:], r.get("code", ""), cell("assigned"), cell("candidate"), cell("ion_only"), cell("neutrals"), cell("stamped_signal_share", 1),
            cell("unstamped_merged"), cell("bright_m0_not_assigned"), cell("bright_unstamped"),
            level_cell(r), cell("good_levels"), cell("m1_families"), cell("m1_signal_share", 1),
            f"{r.get('roster_assigned', '')}/{r.get('roster_present', '')}/{r.get('roster_n', '')}", cell("census_halogen"),
            cell("decoy_shift_rate", 1), cell("decoy_adducts_rate", 1),
            f"{r.get('c13_within_1', '')}/{r.get('c13_n', '')}", f"{r.get('hetero_present', '')}/{r.get('hetero_n', '')}",
            cell("decoy_headline_lt_350_rate", 1), cell("decoy_adducts_new_ion_rate", 1),
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
    # the claim (C13)
    "decoy_shift_identified_rate": -1, "decoy_shift_identified_lt_350_rate": -1, "decoy_adducts_identified_rate": -1,
    "decoy_adducts_identified_lt_350_rate": -1, "bright_m0_not_identified": -1, "roster_misread_identified": -1,
    "claim_assigned_tentative": -1, "claim_unmatched_signal": -1, "m3_own_missing_identified": -1,
    "roster_identified": 1, "claim_identified": 1, "claim_identified_signal": 1, "roster_neutral_or_better": 1,
    # the evidence scale of peaky 0.10.0
    "decoy_shift_established_rate": -1, "decoy_adducts_established_rate": -1, "m3_other_instrument_own_missing": -1,
    # the ppm-shift arms
    "decoy_headline_lt_350_rate": -1, "decoy_ppm_rate": -1, "decoy_ppm_identified_rate": -1,
    "decoy_adducts_new_ion_rate": -1,
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
           f"<p>Acceptance reads the <b>identified</b> class (evidence level 3c on the evidence scale of peaky {EV.SCALE_RELEASE}: "
           "a named context-list entry on a pinned split); <b>neutral</b> is 4a (pinned + a positive fact), <b>ion</b> 4b, "
           "<b>tentative</b> 5a, 5b or none, and <b>not assessed</b> NA (the instrument class is not assessed). The tier is "
           "unchanged and kept beside it. A pre-0.10.0 row shows its own claims, never diffed against the scale's.</p>",
           f"<div class=\"meta\">regenerated {stamp} · {len(rows)} channels · {len(board)} rows</div>", "</header>"]
    # the claim first
    out += ["<h2>Claims</h2>"]
    out.append(html_table(claim_board_rows(board), CLAIM_BOARD_COLUMNS, mono=("channel", "run", "scale", "roster", "dshift")))
    # channel table
    out += ["<h2>Channels</h2>"]
    ch_rows = []
    for r in rows:
        prev = previous_row(board, r["channel"], before=r)
        def d(key):
            if prev is None or not isinstance(r.get(key), (int, float)) or not isinstance(prev.get(key), (int, float)):
                return None
            if not comparable(r, prev, key)[0]:
                return None
            return r[key] - prev[key]

        ch_rows.append({
            "channel": r["channel"], "run": r["run"][-24:], "code": r.get("code", ""), "assigned": r.get("assigned"), "d_assigned": d("assigned"),
            "candidate": r.get("candidate"), "ion_only": r.get("ion_only"), "neutrals": r.get("neutrals"), "signal": r.get("stamped_signal_share"),
            "unst": r.get("unstamped_merged"), "bright": r.get("bright_m0_not_assigned"), "bright_un": r.get("bright_unstamped"),
            "levels": level_cell(r), "good": r.get("good_levels"),
            "fam": r.get("m1_families"), "m1": r.get("m1_signal_share"),
            "roster": f"{r.get('roster_assigned', '')}/{r.get('roster_present', '')}/{r.get('roster_n', '')}",
            "hal": r.get("census_halogen"), "dshift": r.get("decoy_shift_rate"), "dadd": r.get("decoy_adducts_rate"),
            "c13": f"{r.get('c13_within_1', '')}/{r.get('c13_n', '')}", "het": f"{r.get('hetero_present', '')}/{r.get('hetero_n', '')}",
        })
    out.append(html_table(ch_rows, [("channel", "channel"), ("run", "run"), ("code", "code"), ("assigned", "Assigned"), ("d_assigned", "Δ"), ("candidate", "Candidate"),
                                    ("ion_only", "ion-only"), ("neutrals", "neutrals"), ("signal", "stamped signal %"), ("unst", "unstamped merged"), ("bright", "bright M0 not Assigned"),
                                    ("bright_un", "unstamped in top 50"), ("levels", "levels " + "/".join(LEVELS)), ("good", "identified level"), ("fam", "M1 families"), ("m1", "M1 signal %"),
                                    ("roster", "roster A/present/n"), ("hal", "Cl+Br+F"), ("dshift", "decoy Da shift %"), ("dadd", "decoy adducts %"), ("c13", "13C ok/n"), ("het", "hetero ok/n")],
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
        # the claim beside the tier; a card from before it has no `claims`
        cl = c.get("claims")
        if cl:
            out.append("<div class=\"tiles\">" + "".join([
                tile("identified", "claim_identified"), tile("neutral", "claim_neutral"), tile("ion", "claim_ion"),
                tile("tentative", "claim_tentative"), tile("not assessed", "claim_not_assessed"),
                tile("identified signal %", "claim_identified_signal", 1), tile("unmatched signal %", "claim_unmatched_signal", 1),
                tile("Assigned but tentative", "claim_assigned_tentative"), tile("Candidate but identified", "claim_candidate_identified"),
                tile("bright M0 not identified", "bright_m0_not_identified"), tile("roster identified", "roster_identified"),
                tile("decoy Da shift identified %", "decoy_shift_identified_rate", 1),
                tile("decoy Da shift identified < 350 %" + (" (blind on an Orbitrap)" if r.get("decoy_orbitrap") else ""),
                     "decoy_shift_identified_lt_350_rate", 1),
                tile("decoy ppm shift identified %", "decoy_ppm_identified_rate", 1),
                tile("decoy adducts identified %", "decoy_adducts_identified_rate", 1),
                tile("decoy shift neutral established %", "decoy_shift_established_rate", 1),
                tile("M3 own missing", "m3_other_instrument_own_missing"),
            ]) + "</div>")
            out.append("<h3>The claim — identified / neutral / ion / tentative</h3>")
            out.append(html_table(claim_table(c), CLAIM_TABLE_COLUMNS, {"signal": 1, "signal_assigned": 1},
                                  mono=("levels", "tiers", "decoy", "shift_split", "adducts_split")))
            out.append("<p class=\"note\">Tier × claim — separate verdicts, not nested; neither is changed from the other.</p>")
            keys = claim_keys_of(cl)
            out.append(html_table(crosstab_rows(cl), crosstab_columns(keys), {f"{c}_sig": 1 for c in keys}))
            dis = cl.get("disagree") or {}
            out.append(f"<p class=\"note\">Disagreeing rows: Assigned but tentative {_h(dis.get('assigned_tentative'))}, "
                       f"Candidate but identified {_h(dis.get('candidate_identified'))} (brightest 25)</p>")
            out.append(html_table((dis.get("rows") or [])[:25], DISAGREE_COLUMNS, {"mz": 4, "signal_share": 3},
                                  mono=("neutral", "adduct", "level")))
            if corroboration_line(cl):
                out.append(f"<p class=\"note\">pre-0.10.0: {_h(corroboration_line(cl))}</p>")
            acc = c.get("acceptance") or []
            if acc:
                items = []
                for x in acc:
                    nd = 1 if "rate" in x["key"] else 0
                    items.append(f"<li>{_h(x['criterion'])}: <span class=\"mono\">{_h(x['key'])}</span> <b>{_h(_d(x['value'], nd))}</b> "
                                 + (f"(<span class=\"mono\">{_h(x['old_key'])}</span> {_h(_d(x['old_value'], nd))})"
                                    if x.get("old_key") else "(no older metric)") + "</li>")
                out.append("<p class=\"note\"><b>Acceptance</b> — read on the identified class; the old metric in parentheses</p>"
                           "<ul class=\"note\">" + "".join(items) + "</ul>")
        else:
            out.append("<p class=\"note\">card predates C13: no claim block</p>")
        out.append("<div class=\"tiles\">" + "".join([
            tile("spectra", "n_spectra"), tile("Assigned", "assigned"), tile("Candidate", "candidate"), tile("ion-only", "ion_only"), tile("neutrals", "neutrals"),
            tile("stamped signal %", "stamped_signal_share", 1), tile("merged rows unstamped", "unstamped_merged"),
            tile("bright M0 not Assigned", "bright_m0_not_assigned"), tile("unstamped in top 50", "bright_unstamped"),
            tile("rows at level 3c", "good_levels"), tile("M1 families", "m1_families"), tile("M1 signal %", "m1_signal_share", 1),
            tile("roster Assigned", "roster_assigned"), tile("roster present", "roster_present"), tile("Assigned with Cl/Br/F", "census_halogen"),
            tile("decoy Da shift Assigned %", "decoy_shift_rate", 1), tile("decoy adducts Assigned %", "decoy_adducts_rate", 1),
            tile("decoy headline Assigned < 350 %", "decoy_headline_lt_350_rate", 1),
        ]) + "</div>")
        out.append(f"<p class=\"note\">levels ({_h('/'.join(LEVELS) if r.get('claims_schema') == CLAIMS_SCHEMA else '/'.join(OLD_LEVELS))}): "
                   f"<span class=\"mono\">{_h(level_cell(r))}</span> · by channel (Assigned/all) "
                   f"{_h(', '.join(f'{k or chr(63)} {h['by_adduct_assigned'].get(k, 0)}/{v}' for k, v in h['by_adduct'].items()))}</p>")
        sc = h["stamp_coverage"]
        if sc.get("rows"):
            out.append(f"<h3>Stamp coverage — {sc['n_unstamped']} merged rows the time series never carries</h3>")
            out.append(html_table(sc["rows"][:15], [("mz", "m/z"), ("neutral", "neutral"), ("adduct", "adduct"), ("tier", "tier"), ("n_files", "files"), ("alternatives", "alternatives"), ("tier_reason", "tier reason")], {"mz": 4}, mono=("neutral", "adduct")))
        out.append(f"<h3>Brightest {b['n']} ions — bright M0 not Assigned: {b['m0_not_assigned']}, unstamped tracks among the brightest {TOP_N}: {b['unstamped_in_top']}</h3>")
        out.append(html_table(b["rows"], [("ion_mz", "ion m/z"), ("med_cps", "med cps"), ("in", "in"), ("role", "role"), ("ion", "ion"), ("neutral", "neutral"), ("adduct", "adduct"), ("tier", "tier"), ("level", "level"), ("claim", "claim")], {"ion_mz": 4}, mono=("ion", "neutral", "adduct", "in", "level")))
        out.append(f"<h3>Best evidence — {e['n_good']} rows at level 3c (identified)</h3>")
        out.append(html_table(e["rows"][:25], [("neutral", "neutral"), ("adduct", "adduct"), ("ion_mz", "ion m/z"), ("med_cps", "med cps"), ("in", "in"), ("tier", "tier"), ("level", "level"), ("claim", "claim"), ("would_lift", "what would lift it")], {"ion_mz": 4}, mono=("neutral", "adduct", "in", "level")))
        out.append(f"<h3>M1 — {m1['n_tracks']} unstamped tracks in ≥ {int(PRESENCE * 100)} % of spectra, {_p(m1['signal_share'])} % of the signal, grouped into families</h3>")
        out.append(html_table(m1["families"], [("family", "family"), ("n_tracks", "tracks"), ("signal_share", "signal %"), ("max_cps", "max med cps"), ("examples", "brightest examples (parent)")], {"signal_share": 2}, mono=("examples",)))
        out.append(html_table(m1["rows"], [("mz", "m/z"), ("med_cps", "med cps"), ("in", "in"), ("nearest_stamped", "nearest stamped"), ("d_mda", "Δ mDa"), ("family", "family"), ("tag", "residual tag"), ("c_count", "13C carbons"), ("reason", "engine's reason")], {"mz": 4, "d_mda": 1}, mono=("nearest_stamped", "in")))
        out.append("<h3>M2 — expected but not assigned</h3><div class=\"two\"><div>")
        for name, rr in (m2.get("roster") or {}).items():
            out.append(f"<p class=\"note\"><span class=\"chip\">{_h(name)}</span> {rr['n']} formulas: <span class=\"chip good\">{rr['assigned']} Assigned</span>"
                       f"<span class=\"chip\">{rr['candidate']} Candidate</span><span class=\"chip\">{rr.get('same_ion', 0)} same ion</span>"
                       f"<span class=\"chip bad\">{rr['read_as']} read as other</span>"
                       f"<span class=\"chip warn\">{rr['unstamped']} unstamped</span><span class=\"chip\">{rr.get('iso_reagent', 0)} isotope / reagent line</span>"
                       f"<span class=\"chip\">{rr['absent']} absent</span></p>")
        for src, rr in (m2.get("sources") or {}).items():
            out.append(f"<p class=\"note\"><span class=\"chip\">{_h(src)}</span> {rr['n']}: {rr['assigned']} Assigned, {rr['candidate']} Candidate, "
                       f"{rr.get('same_ion', 0)} same ion, {rr['read_as']} read as other, {rr['unstamped']} unstamped, "
                       f"{rr.get('iso_reagent', 0)} isotope / reagent line, {rr['absent']} absent</p>")
        out.append("</div><div>")
        cz_rows = [{"element": k, "rows": v, "examples": cz["examples"].get(k, "")} for k, v in cz["elements"].items()]
        out.append("<p class=\"note\"><b>Element census of Assigned neutrals</b></p>" + html_table(cz_rows, [("element", "element"), ("rows", "Assigned rows"), ("examples", "examples")], mono=("element", "examples")))
        if cz.get("mass_only"):
            out.append(f"<p class=\"note\">{_h(mass_only_line(cz['mass_only']))}</p>")
        out.append("</div></div>")
        miss_rows = [x for x in m2["rows"] if x["status"] in ("read as", "same ion", "unstamped", "candidate")]
        out.append(html_table(miss_rows[:40], [("source", "source"), ("name", "name"), ("neutral", "neutral"), ("status", "status"), ("adduct", "channel"), ("mz", "m/z"), ("cps", "med cps"), ("in", "in"), ("read", "read as"), ("reason", "engine's reason"), ("ledger", "in ledger")], {"mz": 4}, mono=("neutral", "adduct", "read", "in")))
        op, oi = m3.get("other_path"), m3.get("other_instrument")
        if op or oi:
            out.append("<h3>M3 — found elsewhere</h3>")
            if op:
                out.append(f"<p class=\"note\">other path <span class=\"mono\">{_h(op['run'])}</span> ({_h(op['path'])}): {op['n_missing']} of its {op['n_solid']} solid neutrals are absent here</p>")
                out.append(html_table(op["rows"], [("neutral", "neutral"), ("adduct", "adduct"), ("mz", "m/z"), ("tier", "tier"), ("n_files", "files")], {"mz": 4}, mono=("neutral", "adduct")))
            if oi:
                out.append(f"<p class=\"note\">other instrument <span class=\"mono\">{_h(oi['run'])}</span> ({_h(oi['reagent'])}): {oi['n_missing']} of its {oi['n_good']} rows ({_h(oi.get('basis', 'level ≤ 4b'))}) pass the floor and are absent here</p>")
                out.append(html_table(oi["rows"], [("neutral", "neutral"), ("adduct", "other adduct"), ("level", "level"), ("med_cps", "med cps there"), ("share", "share %")], {"share": 0}, mono=("neutral", "adduct", "level")))
            oo = m3.get("other_instrument_own")
            if oo:
                out.append(f"<p class=\"note\">by the other instrument's <b>own</b> evidence (no other-source partners): {oo['n_missing']} of its {oo['n_good']} rows ({_h(oo.get('basis', 'level ≤ 4b'))}) pass the floor and are absent here</p>")
                out.append(html_table(oo["rows"], [("neutral", "neutral"), ("adduct", "other adduct"), ("level", "own level"), ("med_cps", "med cps there"), ("share", "share %")], {"share": 0}, mono=("neutral", "adduct", "level")))
        out.append("<h3>Decoy false-discovery bound</h3>")
        if _scoring_note(dc):
            out.append(f"<p class=\"note\">{html.escape(_scoring_note(dc).replace('`', ''))}</p>")
        if _calibration_note(dc):
            out.append(f"<p class=\"note\">{html.escape(_calibration_note(dc).replace('`', ''))}</p>")
        if dc.get("control") and "error" not in dc["control"]:
            dc_rows = []
            for key, label, a in decoy_arm_rows(dc):
                if a and "error" in a:
                    dc_rows.append({"arm": label, "examples": f"engine error: {a['error']}"})
                elif a:
                    dc_rows.append({"arm": label, "m0": a["m0"], "assigned": a["assigned"], "split": f"{a.get('assigned_lt_350', '')} / {a.get('assigned_ge_350', '')}", "candidate": a["candidate"], "neutrals": a["neutrals"],
                                    "good": sum(a["levels"].get(lv, 0) for lv in GOOD_LEVELS), "rate": a.get("assigned_rate"),
                                    "rate_split": f"{_d(a.get('assigned_lt_350_rate'), 1)} / {_d(a.get('assigned_ge_350_rate'), 1)}",
                                    "ident": a.get("identified"), "ident_split": f"{_d(a.get('identified_lt_350'))} / {_d(a.get('identified_ge_350'))}",
                                    "ident_rate": a.get("identified_rate"), "ident_lt_rate": a.get("identified_lt_350_rate"),
                                    "examples": "; ".join(a.get("examples", [])[:4])})
            hl = [x for x in decoy_headline_lines(dc) if x]
            if hl:
                out.append(f"<p class=\"note\">{html.escape(hl[0].replace('**', ''))}</p>")
            out.append(f"<p class=\"note\">offline engine on <span class=\"mono\">{_h(', '.join(dc['files']))}</span></p>" + html_table(dc_rows, [("arm", "arm"), ("m0", "M0 rows"), ("assigned", "Assigned"), ("split", f"< {DECOY_MZ_SPLIT:.0f} / ≥"), ("candidate", "Candidate"), ("neutrals", "neutrals"), ("good", "level 3c"), ("rate", "Assigned % of control"),
                                                                                                                                  ("rate_split", f"Assigned % < {DECOY_MZ_SPLIT:.0f} / ≥"),
                                                                                                                                  ("ident", "identified pairs"), ("ident_split", f"identified < {DECOY_MZ_SPLIT:.0f} / ≥"), ("ident_rate", "identified % of control"), ("ident_lt_rate", f"identified < {DECOY_MZ_SPLIT:.0f} % of control"),
                                                                                                                                  ("examples", "brightest Assigned")], {"rate": 1, "ident_rate": 1, "ident_lt_rate": 1}, mono=("examples", "ident_split", "rate_split")))
            for why in decoy_skip_lines(dc) + decoy_adducts_lines(dc):
                out.append(f"<p class=\"note\">{html.escape(why.replace('`', '').replace('**', ''))}</p>")
            bins = dc.get("bins") or decoy_bins(dc)
            if bins:
                out.append(f"<p class=\"note\">Assigned by {DECOY_BIN_DA}-Da m/z bin (ppm arms pooled, against the control counted once per ppm arm)</p>"
                           + html_table(bins, DECOY_BIN_COLUMNS, {"shift_rate": 1, "ppm_rate": 1}, mono=("bin",)))
        else:
            out.append(f"<p class=\"note\">decoy mode <span class=\"mono\">{_h(dc.get('mode'))}</span>: not run</p>")
        out.append("<h3>Falsification survival</h3><ul class=\"note\">")
        c13, het, ic, ac = fz.get("c13"), fz.get("hetero"), fz.get("iso_cov"), fz.get("adduct_cov")
        rv = fz.get("resolvability")
        if rv:
            out.append(f"<li>separability of the Assigned peaks (per-file rows): <span class=\"mono\">{_h(str(rv['assigned']))}</span>; "
                       f"capped by the separability / satellite rules: <span class=\"mono\">{_h(str(rv['capped']))}</span></li>")
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
            out.append(html_table([{"metric": d_["metric"], "prev": d_["prev"], "now": d_["now"], "delta": d_["delta"],
                                    "note": d_.get("note") or ""} for d_ in c["delta"]],
                                  [("metric", "metric"), ("prev", "previous"), ("now", "now"), ("delta", "Δ")]
                                  + ([("note", "note")] if any(d_.get("note") for d_ in c["delta"]) else []),
                                  {"prev": 1, "now": 1, "delta": 1}))
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
    ap.add_argument("--levels", help="levels CSV from scripts/level_ledger.py --out, for a run made before the scale "
                                     "(default: the in-core level, else level_ledger.py in-process)")
    ap.add_argument("--other", help="the other PATH on the same batch (cover vs trace-first)")
    ap.add_argument("--other-instrument", help="the other analyser on the same air (feeds M3; an other-source partner "
                                               "when a run is levelled post hoc -- a tag, never a level)")
    ap.add_argument("--overlap", help="UTC window 'start,end' both instruments recorded, e.g. '2026-01-01T06:00,2026-01-03T18:00'")
    ap.add_argument("--mask", action="append", default=[], help="UTC window 'start,end' to exclude (a gap); repeatable")
    ap.add_argument("--floor-cps", type=float, default=10.0, help="other-instrument detection floor: median cps (default 10)")
    ap.add_argument("--floor-share", type=float, default=0.8, help="... present in this share of spectra (default 0.8)")
    ap.add_argument("--decoy", choices=tuple(DECOY_MODES), default=None,
                    help="run the offline engine on decoy input: shift = the Da arm and the ppm arms, ppm = the ppm "
                         "arms only, adducts = the wrong adducts, both = every arm (default none; both with "
                         "--decoy-ledgers)")
    ap.add_argument("--decoy-offset", type=float, default=0.35, help="Da added to every m/z in the shift decoy (default 0.35)")
    ap.add_argument("--decoy-ppm", default=None,
                    help="ppm shifts of the populated-defect shift arms, comma-separated (default "
                         + ",".join(f"{k:+g}" for k in DECOY_PPM) + "; 'none' = no ppm arm); a shift inside a "
                         "file's match window is skipped")
    ap.add_argument("--decoy-files", type=int, default=1, help="brightest cover files to run the decoy on (default 1)")
    ap.add_argument("--decoy-ledgers", help="count the arm ledgers an earlier card kept (<DIR>/<run>/decoy/<file>__<arm>.csv.gz, "
                                            "DIR a scoreboard out dir or the decoy dir itself) instead of running the engine, "
                                            "with the offset, files and adduct sets they were made with (decoy/manifest.json); "
                                            "implies --decoy both unless --decoy is given")
    ap.add_argument("--rosters", help="directory of roster CSVs (default: the packaged peaky/data/rosters)")
    ap.add_argument("--out", default=DEFAULT_OUT, help=f"scoreboard directory (default {DEFAULT_OUT})")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    log = (lambda *a, **k: None) if args.quiet else print

    decoy_mode = args.decoy or ("both" if args.decoy_ledgers else "none")
    ledgers_of: dict[str, str] = {}
    if args.decoy_ledgers and decoy_mode != "none":
        # a DIR with no kept ledger for a run stops here, before a card or a
        # board row without its decoy numbers is written
        for path in args.run_dirs:
            name = os.path.basename(resolve_run_dir(path))
            found = decoy_ledgers_dir(os.path.expanduser(args.decoy_ledgers), name)
            if found is None:
                raise SystemExit(f"--decoy-ledgers {args.decoy_ledgers}: no kept decoy ledger for {name}")
            ledgers_of[path] = found
    other = load_run(args.other) if args.other else None
    other_instrument = load_run(args.other_instrument) if args.other_instrument else None
    rosters = load_rosters(args.rosters)
    board = read_board(os.path.join(args.out, "scoreboard.jsonl"))
    cards = []
    for path in args.run_dirs:
        run = load_run(path)
        # the arm ledgers are kept beside the card (<out>/<run>/decoy/) so a
        # later card can re-count them with --decoy-ledgers
        ledgers = ledgers_of.get(path)
        cards.append(build_card(
            run, levels_csv=args.levels, other=other, other_instrument=other_instrument, rosters=rosters,
            decoy_mode=decoy_mode, decoy_offset=args.decoy_offset, decoy_files=args.decoy_files,
            decoy_ppm=parse_ppm_list(args.decoy_ppm),
            overlap=_parse_window(args.overlap), masks=[_parse_window(m) for m in args.mask],
            floor_cps=args.floor_cps, floor_share=args.floor_share, board=board, log=log,
            decoy_save_dir=None if ledgers else os.path.join(args.out, run.name, "decoy"), decoy_ledgers=ledgers,
        ))
    write_outputs(cards, args.out, log=log)
    for card in cards:
        r = card["row"]
        lv = r["levels"]
        print(f"{r['run']}: identified {r['claim_identified']} / neutral {r['claim_neutral']} / ion {r['claim_ion']} / "
              f"tentative {r['claim_tentative']} (signal {_p(r['claim_identified_signal'])} / {_p(r['claim_neutral_signal'])} / "
              f"{_p(r['claim_ion_signal'])} / {_p(r['claim_tentative_signal'])} %); not assessed {r['claim_not_assessed']}; "
              f"Assigned {r['assigned']} / Candidate {r['candidate']} / neutrals {r['neutrals']}; stamped signal {_p(r['stamped_signal_share'])} %; "
              f"unstamped merged {r['unstamped_merged']}; bright M0 not Assigned {r['bright_m0_not_assigned']}; levels {'/'.join(str(lv.get(k, 0)) for k in LEVELS)}; "
              f"M1 {r['m1_families']} families / {_p(r['m1_signal_share'])} %; roster {r['roster_assigned']}/{r['roster_present']}/{r['roster_n']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
