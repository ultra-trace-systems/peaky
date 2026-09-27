"""Rate every assigned neutral in a ledger on the CIMS-adapted Schymanski scale.

Schymanski's confidence levels were written for LC-HRMS with a fragment spectrum
and a compound library. A chemical-ionization run has neither: there is one
adduct channel, no MS2, and the reagent takes part in the ion. This script is
the executable reference for the adaptation — it says what a level MEANS in
terms of columns a peaky ledger already carries, and it is the yardstick the
in-core `evidence_level` column (Phase B) must reproduce row for row.

    python scripts/level_ledger.py <source>... [--corroborate <source>]
                                   [--upair [<neutral_pairs.csv>]]
                                   [--out levels.csv]

A *source* is a batch run dir (its `per_file/*_ledger.csv`, or `merged_ledger.csv`
when there is no per-file directory) or a single ledger CSV. Every source is
levelled; naming two of them ALSO gives each the other as corroboration, which
is what "the same neutral, seen through a second, independent channel" means —
the other reagent channel of one instrument, or the other instrument sampling
the same air. `--corroborate` adds a source that corroborates but is not itself
levelled. A source corroborates only the neutrals it holds at level 4b or
better by its OWN evidence — levelled first with no corroboration at all, so
two sources can never lift each other on nothing but their agreement (a 5b
formula the other grid also enumerated is two grids agreeing, not a second
sighting). A merged ledger has no predicate columns: its stored level is read
without its own `corroborated` axis.

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
        least one from outside this channel's ionization chemistry -- or (9',
        rule U, --upair) the channel's neutral pair on a supported formula
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
# 15N and 18O are expected per atom of the ion (heavy / light natural abundance;
# a caret '^N' atom is already 15N and is not counted)
PER_ATOM_ABUNDANCE = {"15N": ("N", 0.00368 / 0.99632), "18O": ("O", 0.00205 / 0.99757)}
RATIO_LO, RATIO_HI = 0.5, 2.0
# the element an isotope child's tag measures ('13C2' -> C, '2x81Br' -> Br,
# '81Br(pair)' -> Br, '14N' -> N); 'M' (a generic M+n child) names none
TAG_ELEMENT = re.compile(r"^(?:\d+x)?\d+([A-Z][a-z]?)\d*(?:\(pair\))?$")
# an ion carrying Br or Cl owns its M+2 region: its 81Br / 37Cl line buries an
# 18O satellite, so an '18O' line there is not measured
M2_OWNERS = ("Br", "Cl")
FORMULA_TOKEN = re.compile(r"(\^?)([A-Z][a-z]?)(\d*)")

# The scope of each pass-0 family (docs/EVIDENCE_LEVELS.md section 4.1): a family
# whose entries are hand-listed compounds asserts a COMPOUND, one generated from a
# formula loop asserts a CLASS. 2b needs compound scope AND a one-structure formula
# in peaky/data/isomer_space.csv; any other curated commit is 3a. A family not
# listed is read as a class. (Until 2026-09-22 this script kept two hand-made sets
# -- {atmospheric, reactive_iodine} always 2b, six others always 3a -- which
# agreed with the spec on every golden row but not on the positive-mode families:
# cyclosiloxane D3/D5/D7 read 3a here and 2b in core, and contaminant:silanediol
# was in neither set. The spec is the design; the sets were the bug.)
KNOWN_FAMILY_SCOPE = {
    "atmospheric": "compound",
    "reactive_iodine": "compound",
    "ambient_inorganic": "compound",
    "nitroaromatic": "compound",
    "cyclosiloxane": "compound",
    "indoor_sulfur": "compound",
    "organophosphate": "compound",
    "organothiophosphate": "compound",
    "easyic_hydride": "compound",
    "perfluoroacid": "class",
    "chlorinated_paraffin": "class",
    "contaminant:silanediol": "class",
}
ISOMER_SPACE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "peaky", "data", "isomer_space.csv")
_STRUCTURES: dict | None = None


def plausible_structures(formula: str) -> int | None:
    """The isomer space's structure count for `formula`, None when absent."""
    global _STRUCTURES
    if _STRUCTURES is None:
        _STRUCTURES = {}
        if os.path.isfile(ISOMER_SPACE):
            space = pd.read_csv(ISOMER_SPACE)
            for f, n in zip(space["formula"].astype(str).str.strip(),
                            pd.to_numeric(space["n_plausible_structures"], errors="coerce")):
                if pd.notna(n):
                    _STRUCTURES[f] = int(n)
    return _STRUCTURES.get(str(formula).strip())

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
# ION-ONLY rows (the engine's `ion_only` stage): the electron-attachment line
# beside a committed [M-H]- acid, on "[M]-." with method `ion_only:*` (a merged
# ledger carries the `ion_only_of` link instead). Levelled on their own
# satellite alone (4d with one, 5a without); never a second channel or a
# corroboration for anything, in either direction -- as evidence.py does.
ION_ONLY_ADDUCTS = {"[M]-."}
ION_ONLY_METHOD_PREFIX = "ion_only:"

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


def composition(formula, fold: bool = True, labelled: bool = False) -> dict:
    """Element counts of a formula; a caret isotope ('^N') folds into its
    element unless `fold` is False, when it is left out (it is already heavy),
    or `labelled` is True, when it is its own key ('^N')."""
    out: dict = {}
    if not isinstance(formula, str):
        return out
    for caret, element, n in FORMULA_TOKEN.findall(formula):
        if caret and labelled:
            element = caret + element
        elif caret and not fold:
            continue
        out[element] = out.get(element, 0) + (int(n) if n else 1)
    return out


def expected_ratio(tag: str, ion_formula) -> float | None:
    """Natural height ratio of an isotope child to its M0, None when unknown."""
    if tag.startswith("13C"):
        carbons = count_element(ion_formula, "C")
        return C13_PER_CARBON * carbons if carbons else None
    if tag in PER_ATOM_ABUNDANCE:
        element, per_atom = PER_ATOM_ABUNDANCE[tag]
        counts = composition(ion_formula, fold=False)
        if element == "O" and any(counts.get(e, 0) for e in M2_OWNERS):
            return None
        atoms = counts.get(element, 0)
        return per_atom * atoms if atoms else None
    return ISOTOPE_ABUNDANCE.get(tag)


def neutral_elements(neutral, ion) -> set:
    """Elements whose isotope line speaks for the neutral: the neutral supplies
    more than half of the ion's atoms of the element (else the line measures
    the reagent -- 15N on a urea adduct of an N-free neutral, 81Br on a bromide
    adduct)."""
    own, whole = composition(neutral, labelled=True), composition(ion, labelled=True)
    return {e for e, n in own.items() if n > 0 and 2 * n > whole.get(e, 0)}


def is_ion_only(frame: pd.DataFrame) -> pd.Series:
    """Rows the ion-only stage wrote (see ION_ONLY_ADDUCTS)."""
    adduct = column(frame, "adduct", "").fillna("").astype(str).isin(ION_ONLY_ADDUCTS)
    method = column(frame, "method", "").fillna("").astype(str).str.startswith(ION_ONLY_METHOD_PREFIX)
    mask = adduct & method
    if "ion_only_of" in frame.columns:
        mask = mask | (adduct & frame["ion_only_of"].notna())
    return mask


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
    m0["ion_only"] = is_ion_only(m0).to_numpy()
    regular = m0[~m0["ion_only"]]

    # parent lookup: the M0 row an isotope child hangs off, within its own file
    parents = {}
    for _, row in m0.iterrows():
        parents.setdefault((row["__file"], row.get("peak_id")), row)

    labels: dict[tuple, set] = {}
    ratio_ok: dict[tuple, bool] = {}
    in_band: dict[tuple, set] = {}   # the elements with an in-band line
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
        expected = expected_ratio(tag, parent.get("ion_formula"))
        if expected and RATIO_LO <= ratio / expected <= RATIO_HI:
            ratio_ok[key] = True
            match = TAG_ELEMENT.match(tag)
            if match:
                element = "^N" if tag.startswith("14N") else match.group(1)
                in_band.setdefault(key, set()).add(element)

    # per-neutral facts, over the source's REGULAR M0 rows (an ion-only row is
    # its parent's composition on another adduct, not a second channel for it)
    channels = regular.groupby("neutral_formula")["adduct"].nunique()
    adduct_sets = regular.groupby("neutral_formula")["adduct"].agg(
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
        ion_only = bool(group["ion_only"].any())
        own = in_band.get(key, set()) & neutral_elements(
            neutral, str(group["ion_formula"].iloc[0]))
        aset = set() if ion_only else adduct_sets.get(neutral, set())
        rows.append(
            dict(
                source=label,
                neutral=neutral,
                adduct=adduct,
                ion_only=ion_only,
                ion=str(group["ion_formula"].iloc[0]),
                mz=float(pd.to_numeric(group["mz"], errors="coerce").median()),
                tier="Assigned" if (group["tier"] == "Assigned").any() else "Candidate",
                known_fam=known.iloc[0][6:] if len(known) else "",
                iso=bool(ratio_ok.get(key, False))
                or bool(group["isotopologues"].map(lambda v: len(as_list(v)) > 0).any()),
                multiline=len(own) >= 2,
                multiline_elements="|".join(sorted(own)),
                carbon_ev=carbon_ev,
                chan2=(not ion_only) and int(channels.get(neutral, 0)) >= 2,
                anchor=(not ion_only) and bool(
                    group["anchor_peak_id"].notna().any()
                    or group["series_unit"].notna().any()
                ),
                branch=bool(aset & BARE_ADDUCTS) and bool(aset & CLUSTER_ADDUCTS),
                reagent_only_iso=(not ion_only)
                and bool(satellite)
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
    if bool(getattr(row, "ion_only", False)):
        return "4d" if row.iso else "5a"
    if degenerate and row.n_axes == 0:
        return "5b"
    if row.known_fam:
        if (KNOWN_FAMILY_SCOPE.get(row.known_fam, "class") == "compound"
                and plausible_structures(row.neutral) == 1):
            return "2b"
        return "3a"
    if row.branch:
        return "3b"
    if row.n_axes == 0:
        return "4c" if (unique and row.res_ok) else "5a"
    if (not row.neutral_backed) and row.reagent_only_iso:
        return "4d"
    if row.n_axes >= 2 and row.cross:
        return "4a"
    # 9' (rule U): the profile's neutral pair establishes the neutral; the
    # formula needs its own support -- an isotope, or one plausible ion on a
    # resolved peak
    if bool(getattr(row, "upair", False)) and (row.iso or (unique and row.res_ok)):
        return "4a"
    return "4b"


def assign_levels(df: pd.DataFrame, corroborating: set[str], upair: set[str] | None = None) -> pd.DataFrame:
    """Add the axes, the derived flags and the level to measured rows. `upair`
    is the batch's neutral-pair set (rule U); it is a fact, never an axis."""
    df = df.copy()
    df["known_fam"] = df["known_fam"].fillna("")
    if "ion_only" not in df.columns:
        df["ion_only"] = False
    df["corroborated"] = df["neutral"].isin(corroborating) & ~df["ion_only"].astype(bool)
    df["upair"] = df["neutral"].isin(upair or set()) & ~df["ion_only"].astype(bool)
    df["n_axes"] = df[["iso", "chan2", "anchor", "corroborated"]].sum(axis=1)
    df["cross"] = df.corroborated | df.multiline | df.known_fam.ne("")
    df["neutral_backed"] = (
        df.corroborated | df.chan2 | df.anchor | df.known_fam.ne("") | df.carbon_ev
    )
    df["level"] = df.apply(level_of, axis=1)
    return df


#: a source corroborates the neutrals it holds at one of these levels by its own
#: evidence (4b or better; 1 and 2a never fire)
CORROBORATING_LEVELS = {"1", "2a", "2b", "3a", "3b", "4a", "4b"}
OWN_AXES = {"iso", "chan2", "anchor"}


def own_good_neutrals(frame: pd.DataFrame) -> set[str]:
    """The neutrals a measured source holds at 4b or better when it is levelled
    with NO corroboration — its own evidence only. Ion-only pairs never count."""
    own = assign_levels(frame, set())
    ok = own["level"].isin(CORROBORATING_LEVELS) & ~own["ion_only"].astype(bool)
    return set(own.loc[ok, "neutral"].astype(str)) - {""}


def stored_good_neutrals(ledger: pd.DataFrame, label: str) -> set[str]:
    """A merged ledger (no predicate columns): the rows its stored level puts at
    4b or better that still hold an axis of their own once `corroborated` is
    taken away (a level the axis alone produced does not count)."""
    if "evidence_level" not in ledger.columns:
        raise SystemExit(f"{label}: neither per-file predicate columns nor an evidence_level column")
    ledger = ledger[~is_ion_only(ledger)]
    keep = []
    for level, axes in zip(ledger["evidence_level"], column(ledger, "evidence_axes", "")):
        parts = set(str(axes).split("|")) if pd.notna(axes) else set()
        good = str(level) in CORROBORATING_LEVELS
        if good and "corroborated" in parts:
            own = parts & OWN_AXES
            good = bool(own) and not (own == {"iso"} and "reagent_only_iso" in parts)
        keep.append(good)
    return set(column(ledger, "neutral_formula")[np.array(keep, dtype=bool)].dropna().astype(str)) - {""}


def source_good_neutrals(path: str) -> tuple[str, pd.DataFrame | None, set[str]]:
    """(label, measured frame or None, the neutrals it corroborates) for one source."""
    label, ledger = load_source(path)
    if "role" not in ledger.columns:
        return label, None, stored_good_neutrals(ledger, label)
    halogen = detect_reagent_halogen(ledger[column(ledger, "role").astype(str) == "M0"])
    frame = measure_source(label, ledger, halogen)
    if frame.empty:
        return label, None, set()
    frame["reagent_halogen"] = halogen or ""
    return label, frame, own_good_neutrals(frame)


def upair_neutrals(path: str) -> set[str]:
    """The neutrals whose neutral pair holds, read from a batch's
    tables/neutral_pairs.csv (a run dir, or an --out-dir holding one run) or
    from that CSV itself. A source without the table says so on stderr."""
    table = path
    if os.path.isdir(path):
        table = os.path.join(path, "tables", "neutral_pairs.csv")
        if not os.path.isfile(table):
            found = sorted(glob.glob(os.path.join(path, "*", "tables", "neutral_pairs.csv")))
            if len(found) == 1:
                table = found[0]
    if not os.path.isfile(table):
        print(f"  --upair: no neutral_pairs.csv for {path}; rule U does not fire there", file=sys.stderr)
        return set()
    frame = pd.read_csv(table)
    if "upair" not in frame.columns:
        return set()
    held = frame["upair"].map(truthy)
    return set(frame.loc[held, "neutral_formula"].astype(str))


def run(sources: list[str], corroborate: list[str], upair: str | None = None) -> pd.DataFrame:
    """Level every source, each corroborated by the others plus --corroborate —
    by the neutrals each of them holds at 4b or better on its own evidence.
    `upair`: 'auto' reads each run-dir source's own tables/neutral_pairs.csv;
    a CSV path applies that table to every levelled source; None = rule U off."""
    measured = {}
    neutrals = {}
    for path in sources:
        label, frame, good = source_good_neutrals(path)
        if frame is None:
            print(f"  {label}: no M0 rows, skipped", file=sys.stderr)
            continue
        measured[path] = (label, frame)
        neutrals[path] = good
    for path in corroborate:
        neutrals[f"--corroborate:{path}"] = source_good_neutrals(path)[2]
    out = []
    for path, (label, frame) in measured.items():
        others: set[str] = set()
        for other, values in neutrals.items():
            if other != path:
                others |= values
        held = set()
        if upair == "auto":
            held = upair_neutrals(path)
        elif upair:
            held = upair_neutrals(upair)
        out.append(assign_levels(frame, others, held))
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
    parser.add_argument(
        "--upair",
        nargs="?",
        const="auto",
        default=None,
        help="rule U: read each batch source's tables/neutral_pairs.csv (no value), "
        "or apply this neutral-pair CSV to every levelled source",
    )
    parser.add_argument("--out", help="write the levelled rows here as CSV")
    args = parser.parse_args(argv)

    df = run(args.sources, args.corroborate, args.upair)
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
