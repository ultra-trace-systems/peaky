"""Rate every assigned neutral in a ledger on the CIMS-adapted Schymanski scale.

Schymanski's confidence levels were written for LC-HRMS with a fragment spectrum
and a compound library. A chemical-ionization run has neither: there is one
adduct channel, no MS2, and the reagent takes part in the ion. This script is
the executable reference for the adaptation — it says what a level MEANS in
terms of columns a peaky ledger already carries, and it is the yardstick the
in-core `evidence_level` column (Phase B) must reproduce row for row.

    python scripts/level_ledger.py <source>... [--corroborate <source>]
                                   [--out levels.csv]

A *source* is a batch run dir (its `per_file/*_ledger.csv`, or `merged_ledger.csv`
when there is no per-file directory) or a single ledger CSV. Every source is
levelled; naming two of them ALSO gives each the other as corroboration, which
is what "the same neutral, seen through a second, independent channel" means —
the other reagent channel of one instrument, or the other instrument sampling
the same air. `--corroborate` adds a source that corroborates but is not itself
levelled.

One row out per `(source, neutral_formula, adduct)` over the source's M0 rows.

The scale, in the order the predicates are tried:

    5b  the assignment argues with itself — a near-tie the arbiter broke, a row
        below assignability, or a score the engine itself calls Low/Suspect;
        also a mass-degenerate row with no corroborating axis at all
    2b  a curated identity on a formula that admits essentially one structure
    3a  a named compound class, isomers open (PFCA, nitroaromatic, …)
    3b  a substituent only, via the gas-phase acidity branch: the same neutral
        appears both deprotonated and clustered
    4c  formula unopposed — unique at the calibrated sigma on a peak the width
        model says is separable — but nothing corroborates it
    5a  the same, on a peak that is not unique
    4d  ION formula only: the sole isotope support is the reagent halogen, which
        pins the ion and says nothing about the neutral (CIMS-specific; there is
        no Schymanski analogue)
    4a  formula confirmed and the NEUTRAL established: two orthogonal axes, at
        least one from outside this channel's ionization chemistry
    4b  formula confirmed, one corroboration

The four axes are: a verified isotopologue, a second adduct channel, a
homologous-series or anchor tie, and the corroborating source. Levels 1 and 2a
need an authentic standard or a library spectrum and never fire here.

Recovered on 2026-09-21 from the transcript that first ran it; the golden count
vectors it must reproduce are in `tests/test_level_ledger.py`.
"""

from __future__ import annotations

import argparse
import ast
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

LEVEL_ORDER = ["1", "2a", "2b", "3a", "3b", "4a", "4b", "4c", "4d", "5a", "5b"]

# Natural abundance of the heavy isotope relative to the light one, for the
# satellite the audit looks for. 13C is per carbon and computed from the ion.
ISOTOPE_ABUNDANCE = {
    "34S": 0.0443,
    "37Cl": 0.3196,
    "81Br": 0.9728,
    "29Si": 0.0508,
    "30Si": 0.0335,
}
C13_PER_CARBON = 0.0107
RATIO_LO, RATIO_HI = 0.5, 2.0

# Curated families whose formula admits essentially one structure in this
# chemistry, and those that name a class only.
UNIQUE_FAMILIES = {"atmospheric", "reactive_iodine"}
CLASS_ONLY_FAMILIES = {
    "nitroaromatic",
    "perfluoroacid",
    "chlorinated_paraffin",
    "organophosphate",
    "organothiophosphate",
    "indoor_sulfur",
    "cyclosiloxane",
}

BARE_ADDUCTS = {"[M-H]-"}
CLUSTER_ADDUCTS = {
    "[M+NO3]-",
    "[M+15NO3]-",
    "[M+^NO3]-",
    "[M+Br]-",
    "[M+HBr+Br]-",
    "[M+CO3]-",
}
RESOLVED = {"resolved", "isolated"}
LOW_CONFIDENCE = {"Low", "Suspect"}

# The heavy satellite a reagent halogen contributes. Iodine is monoisotopic, so
# an iodide reagent can never produce a reagent-only isotope pattern.
HALOGEN_SATELLITE = {"Br": "81Br", "Cl": "37Cl", "I": None}


def truthy(value) -> bool:
    """Null-safe truthiness: NaN, NA, None and 'false'/'' are all False."""
    if value is None or value is pd.NA:
        return False
    if isinstance(value, float) and np.isnan(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes")
    return bool(value)


def first_word(value) -> str:
    """The grade a `confidence` cell opens with ('High (0.91)' -> 'High')."""
    if not isinstance(value, str):
        return ""
    parts = value.split()
    return parts[0] if parts else ""


def as_list(value) -> list:
    """A ledger cell holding a repr'd list, or nothing at all."""
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        parsed = ast.literal_eval(value)
    except Exception:
        return []
    return list(parsed) if isinstance(parsed, (list, tuple)) else []


def count_element(formula, element: str) -> int:
    """How many of `element` a formula carries; 0 when it carries none."""
    if not isinstance(formula, str):
        return 0
    match = re.search(element + r"(\d*)(?![a-z])", formula)
    if not match:
        return 0
    return int(match.group(1)) if match.group(1) else 1


def column(frame: pd.DataFrame, name: str, default=np.nan) -> pd.Series:
    """The column if the ledger has it, a constant column if it does not."""
    if name in frame.columns:
        return frame[name]
    return pd.Series([default] * len(frame), index=frame.index)


def resolve_source(path: str) -> tuple[str, list[str]]:
    """(label, ledger csv paths) for a run dir, a run dir's parent, or a csv."""
    path = os.path.expanduser(path.rstrip("/"))
    if os.path.isfile(path):
        return os.path.basename(path)[:-4] if path.endswith(".csv") else path, [path]
    if not os.path.isdir(path):
        raise SystemExit(f"no such source: {path}")
    label = os.path.basename(path)
    per_file = sorted(glob.glob(os.path.join(path, "per_file", "*_ledger.csv")))
    if per_file:
        return label, per_file
    merged = os.path.join(path, "merged_ledger.csv")
    if os.path.isfile(merged):
        return label, [merged]
    # the --out-dir that holds exactly one run dir
    inner = [d for d in sorted(glob.glob(os.path.join(path, "*"))) if os.path.isdir(d)]
    runs = [
        d
        for d in inner
        if os.path.isdir(os.path.join(d, "per_file"))
        or os.path.isfile(os.path.join(d, "merged_ledger.csv"))
    ]
    if len(runs) == 1:
        return resolve_source(runs[0])
    raise SystemExit(f"no ledger under {path}")


def load_source(path: str) -> tuple[str, pd.DataFrame]:
    """Every ledger row of one source, stamped with the file it came from."""
    label, files = resolve_source(path)
    frames = []
    for f in files:
        frame = pd.read_csv(f, low_memory=False)
        frame["__file"] = os.path.basename(f)[:16]
        frames.append(frame)
    return label, pd.concat(frames, ignore_index=True)


def detect_reagent_halogen(m0: pd.DataFrame) -> str | None:
    """The halogen of the channel's commonest CLUSTER adduct, or None.

    A bromide channel clusters on Br; a nitrate channel does not, even when a
    stray `[M+Br]-` row is present. Counting only cluster adducts separates the
    two without naming any instrument.
    """
    adducts = column(m0, "adduct").dropna().astype(str)
    clusters = adducts[adducts.isin(CLUSTER_ADDUCTS)]
    if clusters.empty:
        return None
    top = clusters.value_counts().idxmax()
    for halogen in ("Br", "Cl", "I"):
        if count_element(top, halogen):
            return halogen
    return None


def measure_source(
    label: str, ledger: pd.DataFrame, halogen: str | None
) -> pd.DataFrame:
    """One row of evidence per (neutral, adduct) the source committed."""
    role = column(ledger, "role").astype(str)
    m0 = ledger[role == "M0"].copy()
    iso = ledger[role == "iso_child"]
    if m0.empty:
        return pd.DataFrame()

    m0["neutral_formula"] = column(m0, "neutral_formula").fillna("").astype(str)
    m0["adduct"] = column(m0, "adduct").fillna("").astype(str)

    # parent lookup: the M0 row an isotope child hangs off, within its own file
    parents = {}
    for _, row in m0.iterrows():
        parents.setdefault((row["__file"], row.get("peak_id")), row)

    labels: dict[tuple, set] = {}
    ratio_ok: dict[tuple, bool] = {}
    for _, child in iso.iterrows():
        parent = parents.get((child["__file"], child.get("parent_peak_id")))
        if parent is None:
            continue
        key = (parent["neutral_formula"], parent["adduct"])
        tag = str(child.get("iso_label")).split("+")[0].strip()
        labels.setdefault(key, set()).add(tag)
        height = parent.get("height")
        height = float(height) if pd.notna(height) and height > 0 else np.nan
        if height != height or not pd.notna(child.get("height")):
            continue
        ratio = float(child["height"]) / height
        if not np.isfinite(ratio):
            continue
        if tag.startswith("13C"):
            carbons = count_element(parent.get("ion_formula"), "C")
            expected = C13_PER_CARBON * carbons if carbons else None
        else:
            expected = ISOTOPE_ABUNDANCE.get(tag)
        if expected and RATIO_LO <= ratio / expected <= RATIO_HI:
            ratio_ok[key] = True

    # per-neutral facts, over the source's M0 rows
    channels = m0.groupby("neutral_formula")["adduct"].nunique()
    adduct_sets = m0.groupby("neutral_formula")["adduct"].agg(
        lambda s: set(s.dropna().astype(str))
    )
    satellite = HALOGEN_SATELLITE.get(halogen) if halogen else None

    for name in (
        "method",
        "confidence",
        "degeneracy_note",
        "resolvability",
        "series_unit",
        "anchor_peak_id",
        "isotopologues",
        "tied",
        "below_assignability",
        "degeneracy_density",
        "ion_formula",
        "tier",
        "height",
        "mz",
        "ppm_error_cal",
        "occurrence_y",
    ):
        if name not in m0.columns:
            m0[name] = np.nan

    rows = []
    for (neutral, adduct), group in m0.groupby(["neutral_formula", "adduct"]):
        key = (neutral, adduct)
        tags = labels.get(key, set()) - {"M0"}
        methods = group["method"].astype(str)
        known = methods[methods.str.startswith("known:")]
        degeneracy = pd.to_numeric(group["degeneracy_density"], errors="coerce")
        seen = {
            str(v).strip()
            for v in group["resolvability"].dropna()
            if str(v).strip() and str(v).strip().lower() != "nan"
        }
        carbon_ev = any(t.startswith("13C") for t in tags)
        rows.append(
            dict(
                source=label,
                neutral=neutral,
                adduct=adduct,
                ion=str(group["ion_formula"].iloc[0]),
                mz=float(pd.to_numeric(group["mz"], errors="coerce").median()),
                tier="Assigned" if (group["tier"] == "Assigned").any() else "Candidate",
                known_fam=known.iloc[0][6:] if len(known) else "",
                iso=bool(ratio_ok.get(key, False))
                or bool(group["isotopologues"].map(lambda v: len(as_list(v)) > 0).any()),
                multiline=len(tags) >= 2,
                carbon_ev=carbon_ev,
                chan2=int(channels.get(neutral, 0)) >= 2,
                anchor=bool(
                    group["anchor_peak_id"].notna().any()
                    or group["series_unit"].notna().any()
                ),
                branch=bool(adduct_sets.get(neutral, set()) & BARE_ADDUCTS)
                and bool(adduct_sets.get(neutral, set()) & CLUSTER_ADDUCTS),
                reagent_only_iso=bool(satellite)
                and bool(tags)
                and not carbon_ev
                and all(t.startswith(satellite) for t in tags),
                iso_labels="|".join(sorted(tags)),
                tied=bool(group["tied"].map(truthy).all()),
                below=bool(group["below_assignability"].map(truthy).any()),
                lowconf=bool(
                    group["confidence"].map(first_word).isin(LOW_CONFIDENCE).all()
                ),
                degeneracy=float(degeneracy.median())
                if degeneracy.notna().any()
                else np.nan,
                saturated=bool(
                    group["degeneracy_note"]
                    .astype(str)
                    .str.contains("MASS-SATURATED")
                    .any()
                ),
                # resolvability is measured on the trace-first path only; a
                # source that never measured it is not penalised for it.
                res_ok=(not seen) or bool(seen & RESOLVED),
                resolvability="|".join(sorted(seen)),
                n_files=int(group["__file"].nunique()),
                height=float(pd.to_numeric(group["height"], errors="coerce").median()),
                ppm=float(
                    pd.to_numeric(group["ppm_error_cal"], errors="coerce").median()
                ),
                occurrence=float(
                    pd.to_numeric(group["occurrence_y"], errors="coerce").median()
                ),
            )
        )
    return pd.DataFrame(rows)


def level_of(row) -> str:
    """The decision table. Order matters: the first predicate that holds wins."""
    hard = bool(row.tied) or bool(row.below) or bool(row.lowconf)
    degenerate = bool(row.saturated) or (
        pd.notna(row.degeneracy) and row.degeneracy >= 3
    )
    unique = pd.notna(row.degeneracy) and row.degeneracy <= 1
    if hard:
        return "5b"
    if degenerate and row.n_axes == 0:
        return "5b"
    if row.known_fam in UNIQUE_FAMILIES:
        return "2b"
    if row.known_fam in CLASS_ONLY_FAMILIES:
        return "3a"
    if row.branch:
        return "3b"
    if row.n_axes == 0:
        return "4c" if (unique and row.res_ok) else "5a"
    if (not row.neutral_backed) and row.reagent_only_iso:
        return "4d"
    if row.n_axes >= 2 and row.cross:
        return "4a"
    return "4b"


def assign_levels(df: pd.DataFrame, corroborating: set[str]) -> pd.DataFrame:
    """Add the axes, the derived flags and the level to measured rows."""
    df = df.copy()
    df["known_fam"] = df["known_fam"].fillna("")
    df["corroborated"] = df["neutral"].isin(corroborating)
    df["n_axes"] = df[["iso", "chan2", "anchor", "corroborated"]].sum(axis=1)
    df["cross"] = df.corroborated | df.multiline | df.known_fam.ne("")
    df["neutral_backed"] = (
        df.corroborated | df.chan2 | df.anchor | df.known_fam.ne("") | df.carbon_ev
    )
    df["level"] = df.apply(level_of, axis=1)
    return df


def run(sources: list[str], corroborate: list[str]) -> pd.DataFrame:
    """Level every source, each corroborated by the others plus --corroborate."""
    measured = {}
    neutrals = {}
    for path in sources:
        label, ledger = load_source(path)
        halogen = detect_reagent_halogen(ledger[column(ledger, "role").astype(str) == "M0"])
        frame = measure_source(label, ledger, halogen)
        if frame.empty:
            print(f"  {label}: no M0 rows, skipped", file=sys.stderr)
            continue
        frame["reagent_halogen"] = halogen or ""
        measured[path] = (label, frame)
        neutrals[path] = set(frame["neutral"])
    for path in corroborate:
        label, ledger = load_source(path)
        role = column(ledger, "role").astype(str)
        neutrals[f"--corroborate:{path}"] = set(
            column(ledger[role == "M0"], "neutral_formula").dropna().astype(str)
        )
    out = []
    for path, (label, frame) in measured.items():
        others: set[str] = set()
        for other, values in neutrals.items():
            if other != path:
                others |= values
        out.append(assign_levels(frame, others))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="CIMS-adapted Schymanski evidence levels over a peaky ledger."
    )
    parser.add_argument(
        "sources",
        nargs="+",
        help="run dirs or ledger CSVs; two or more corroborate each other",
    )
    parser.add_argument(
        "--corroborate",
        action="append",
        default=[],
        help="a source that corroborates but is not itself levelled",
    )
    parser.add_argument("--out", help="write the levelled rows here as CSV")
    args = parser.parse_args(argv)

    df = run(args.sources, args.corroborate)
    if df.empty:
        print("nothing to level", file=sys.stderr)
        return 1
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        df.to_csv(args.out, index=False)

    order = [k for k in LEVEL_ORDER if k not in ("1", "2a")]
    for label, group in df.groupby("source", sort=False):
        halogen = group["reagent_halogen"].iloc[0] or "none"
        counts = group.level.value_counts()
        vector = "/".join(str(int(counts.get(k, 0))) for k in order)
        print(f"{label}  n={len(group)}  reagent halogen {halogen}")
        print(f"  {'/'.join(order)}")
        print(f"  {vector}")
    if args.out:
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
