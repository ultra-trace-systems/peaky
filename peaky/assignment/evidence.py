"""Evidence levels -- what the evidence behind a committed formula is worth.

`tier` says whether the engine will *print* a formula (Assigned) or only *offer*
it (Candidate). A level says what the evidence behind a committed formula is
worth on the scale a reader of an identification paper already knows --
Schymanski et al. (2014), Environ. Sci. Technol. 48, 2097 -- numbered downward
(1 = best) and adapted to chemical ionization, where there is no chromatography,
no fragment spectrum, no library, and the reagent is part of the ion. The
contract is docs/EVIDENCE_LEVELS.md; the executable reference every predicate
here reproduces row for row is scripts/level_ledger.py.

One level per committed M0 row. It is computed per (neutral_formula, adduct)
over the rows that share the pair inside one SOURCE -- a file, or a batch's
pooled files -- and stamped on every M0 row of the pair. Isotope children,
reagent ions, artifacts and unexplained peaks carry no level.

The scale, in the order the predicates are tried (the first that holds wins):

    5b  the assignment argues with itself: a near-tie the arbiter broke, a row
        below assignability, or a score the engine calls Low/Suspect; also a
        mass-degenerate pair with no corroborating axis at all
    2b  a curated identity (compound-scope pass-0 family) on a formula the
        isomer space says admits one structure
    3a  any other curated commit: a named class, isomers open
    3b  the gas-phase-acidity branch: the same neutral deprotonated AND clustered
    4c  formula unopposed -- one plausible ion in the calibrated window on a
        separable peak -- but nothing corroborates it
    5a  exact mass only; no discriminating test was possible
    4d  ION formula only: the sole isotope support is the reagent halogen, which
        pins the ion and says nothing about the neutral (CIMS-specific)
    4a  formula confirmed AND the neutral established: two orthogonal axes, at
        least one from outside this channel's ionization chemistry
    4b  formula confirmed, one corroboration

The four axes: a verified isotopologue (`iso`), a second adduct channel
(`chan2`), a homologous-series or anchor tie (`anchor`), and a corroborating
source (`corroborated`: the other reagent channel, the other instrument on the
same air, or a `--corroborate` source). Levels 1 and 2a need an authentic
standard or a library spectrum and never fire.

Entry points: `apply_levels` (the `evidence` stage of assign.run: writes the
four columns in place), `compute_levels` (pure, one row per M0), `level_pooled`
(a batch's per-file ledgers as ONE source, one row per pair) and `stamp_merged`
(join the pooled result onto the merged ledger by ion).
"""

from __future__ import annotations

import ast
import glob
import os
import re
from functools import lru_cache

import numpy as np
import pandas as pd

from peaky import paths as PT

# ---------------------------------------------------------------------------
# the scale
# ---------------------------------------------------------------------------
LEVEL_ORDER = ["1", "2a", "2b", "3a", "3b", "4a", "4b", "4c", "4d", "5a", "5b"]
#: the levels that can fire today (1 and 2a are defined and never assigned)
LEVELS = LEVEL_ORDER[2:]
LEVEL_MEANING = {
    "1": "confirmed by an authentic standard in the same source and chemistry (never fires)",
    "2a": "matched to a library spectrum (never fires: CIMS has none)",
    "2b": "curated identity on a formula that admits one structure",
    "3a": "curated class, isomers open",
    "3b": "acid branch: the same neutral deprotonated and clustered (a substituent only)",
    "4a": "formula confirmed and the neutral established: two axes, one outside the channel",
    "4b": "formula confirmed, one corroboration",
    "4c": "formula unopposed on a separable peak, nothing corroborates it",
    "4d": "ion formula only: the reagent halogen pins the ion, not the neutral",
    "5a": "exact mass only; no discriminating test was possible",
    "5b": "the assignment argues with itself (near-tie, below assignability, Low/Suspect, or degenerate with no axis)",
}
#: the four columns the stage writes, in order
COLUMNS = ("evidence_level", "evidence_axes", "level_reason", "n_plausible_structures")
#: the four axes, in the order `evidence_axes` lists them
AXES = ("iso", "chan2", "anchor", "corroborated")
#: what the pooled batch recompute reads -- `trim()` keeps these of a ledger
PREDICATE_COLUMNS = (
    "role", "peak_id", "parent_peak_id", "iso_label", "neutral_formula", "adduct",
    "ion_formula", "mz", "height", "tier", "method", "confidence", "tied",
    "below_assignability", "degeneracy_density", "degeneracy_note", "resolvability",
    "series_unit", "anchor_peak_id", "isotopologues", "ppm_error_cal", "occurrence",
)

# Natural abundance of the heavy isotope relative to the light one, for the
# satellite the audit looks for. 13C is per carbon and computed from the ion.
ISOTOPE_ABUNDANCE = {"34S": 0.0443, "37Cl": 0.3196, "81Br": 0.9728, "29Si": 0.0508, "30Si": 0.0335}
C13_PER_CARBON = 0.0107
RATIO_LO, RATIO_HI = 0.5, 2.0

BARE_ADDUCTS = {"[M-H]-"}
CLUSTER_ADDUCTS = {"[M+NO3]-", "[M+15NO3]-", "[M+^NO3]-", "[M+Br]-", "[M+HBr+Br]-", "[M+CO3]-"}
#: ION-ONLY rows (the `ion_only` stage, cleanup.commit_ion_only_electron_attachment):
#: the +1.0078 Da electron-attachment line beside a committed [M-H]- acid,
#: committed on this adduct with method `ion_only:*`. The composition is pinned
#: (exact mass, own 13C); the ionization process and the neutral are open. Such a
#: row is levelled on its own satellite alone -- 4d with one, 5a without -- and
#: is kept OUT of the per-neutral pools in both directions: it never gives its
#: parent a second channel (`chan2`) or the acid branch, never takes an axis from
#: the parent, and never corroborates (or is corroborated by) another source.
ION_ONLY_ADDUCTS = {"[M]-."}
ION_ONLY_METHOD_PREFIX = "ion_only:"
RESOLVED = {"resolved", "isolated"}
LOW_CONFIDENCE = {"Low", "Suspect"}
# The heavy satellite a reagent halogen contributes. Iodine is monoisotopic, so
# an iodide reagent can never produce a reagent-only isotope pattern.
HALOGEN_SATELLITE = {"Br": "81Br", "Cl": "37Cl", "I": None}

#: Scope of each pass-0 family (passes/directors.py `_known_species`): a family
#: whose entries are hand-listed compounds asserts a COMPOUND; one generated from
#: a formula loop asserts a CLASS. A property of the registry, not of the level:
#: 2b needs compound scope AND a one-structure formula in the isomer space. A
#: family not listed here is read as a class (it can be 3a, never 2b).
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


def family_scope(family: str) -> str:
    """`compound` or `class` for a pass-0 family name (unknown -> class)."""
    return KNOWN_FAMILY_SCOPE.get(str(family), "class")


# ---------------------------------------------------------------------------
# null-safe cell readers (the script's, verbatim in behaviour)
# ---------------------------------------------------------------------------
def truthy(value) -> bool:
    """Null-safe truthiness: NaN, NA, None and 'false'/'' are all False.
    `bool(numpy.nan)` is True and once inflated 5b by 127 rows."""
    if value is None or value is pd.NA:
        return False
    if isinstance(value, float) and np.isnan(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes")
    try:
        return bool(value)
    except (TypeError, ValueError):
        return False


def first_word(value) -> str:
    """The grade a `confidence` cell opens with ('High (0.91)' -> 'High')."""
    if not isinstance(value, str):
        return ""
    parts = value.split()
    return parts[0] if parts else ""


def as_list(value) -> list:
    """A ledger cell holding a repr'd / JSON list, or nothing at all.

    The ledger writes the server-attributed satellite list with json.dumps, so a
    line without a per-line score carries `null`; the reference script's
    `ast.literal_eval` cannot read that and counts such a list as empty. JSON is
    tried first here so the list is read as written; a cell neither parser
    reads is empty."""
    if not isinstance(value, str) or not value.strip():
        return []
    parsed = None
    try:
        import json
        parsed = json.loads(value)
    except Exception:
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


def is_ion_only(frame: pd.DataFrame) -> pd.Series:
    """Boolean mask of the rows the ion-only stage wrote: an ION_ONLY_ADDUCTS
    adduct carrying an `ion_only:` method, or (a merged ledger, which has no
    method column) an `ion_only_of` link."""
    adduct = _col(frame, "adduct", "").fillna("").astype(str).isin(ION_ONLY_ADDUCTS)
    method = _col(frame, "method", "").fillna("").astype(str).str.startswith(ION_ONLY_METHOD_PREFIX)
    mask = adduct & method
    if "ion_only_of" in frame.columns:
        mask = mask | (adduct & frame["ion_only_of"].notna())
    return mask


def _col(frame: pd.DataFrame, name: str, default=np.nan) -> pd.Series:
    """The column if the frame has it, a constant column if it does not."""
    if name in frame.columns:
        return frame[name]
    return pd.Series([default] * len(frame), index=frame.index, dtype=object)


# ---------------------------------------------------------------------------
# the isomer space
# ---------------------------------------------------------------------------
@lru_cache(maxsize=4)
def _isomer_space_cached(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["formula"] = df["formula"].astype(str).str.strip()
    df["n_plausible_structures"] = pd.to_numeric(df["n_plausible_structures"], errors="coerce").astype("Int64")
    return df


def load_isomer_space(path: str | None = None) -> pd.DataFrame:
    """`peaky/data/isomer_space.csv` (or `path`): formula, n_plausible_structures,
    family, name, rationale. One row per formula a pass-0 family can commit."""
    return _isomer_space_cached(os.path.abspath(path or PT.pkg_data("isomer_space.csv"))).copy()


def _structures(isomer_space: pd.DataFrame | None) -> dict:
    space = load_isomer_space() if isomer_space is None else isomer_space
    return dict(zip(space["formula"].astype(str), space["n_plausible_structures"]))


# ---------------------------------------------------------------------------
# the source
# ---------------------------------------------------------------------------
def detect_reagent_halogen(m0: pd.DataFrame) -> str | None:
    """The halogen of the channel's commonest CLUSTER adduct, or None.

    A bromide channel clusters on Br; a nitrate channel does not, even when a
    stray `[M+Br]-` row is present (two against 346 `[M+NO3]-` on a real
    nitrate channel). Counting only cluster adducts separates the two without
    naming any instrument."""
    adducts = _col(m0, "adduct").dropna().astype(str)
    clusters = adducts[adducts.isin(CLUSTER_ADDUCTS)]
    if clusters.empty:
        return None
    top = clusters.value_counts().idxmax()
    for halogen in ("Br", "Cl", "I"):
        if count_element(top, halogen):
            return halogen
    return None


def _measure(frame: pd.DataFrame, *, halogen: str | None) -> pd.DataFrame:
    """One row of evidence per (neutral, adduct) the source committed. `frame`
    carries a `__file` column (the file each row came from)."""
    role = _col(frame, "role").astype(str)
    m0 = frame[role == "M0"].copy()
    if m0.empty:
        return pd.DataFrame()
    for name in PREDICATE_COLUMNS:
        if name not in m0.columns:
            m0[name] = np.nan
    m0["__neutral"] = m0["neutral_formula"].fillna("").astype(str)
    m0["__adduct"] = m0["adduct"].fillna("").astype(str)
    m0["__ion_only"] = is_ion_only(m0).to_numpy()
    # the per-neutral facts (second channel, acid branch) are read over the
    # REGULAR rows only: an ion-only row is the parent's own composition on
    # another adduct and must not count as a second channel for it
    regular = m0[~m0["__ion_only"]]

    # satellites: each child hangs off the M0 with peak_id == parent_peak_id in
    # the same file; the tag is the label before any '+' (13C+1 -> 13C)
    iso = frame[role == "iso_child"]
    parents = (m0.drop_duplicates(["__file", "peak_id"])
               [["__file", "peak_id", "__neutral", "__adduct", "height", "ion_formula"]]
               .rename(columns={"peak_id": "__pid", "height": "__h_parent"}))
    tags: dict[tuple, set] = {}
    ratio_ok: dict[tuple, bool] = {}
    if len(iso) and len(parents):
        ch = iso.loc[iso["parent_peak_id"].notna(),
                     ["__file", "parent_peak_id", "iso_label", "height"]]
        ch = ch.merge(parents, left_on=["__file", "parent_peak_id"], right_on=["__file", "__pid"], how="inner")
        ch["__tag"] = ch["iso_label"].map(lambda v: str(v).split("+")[0].strip() if pd.notna(v) else "")
        ch = ch[ch["__tag"] != ""]
        hp = pd.to_numeric(ch["__h_parent"], errors="coerce")
        hc = pd.to_numeric(ch["height"], errors="coerce")
        ratio = hc / hp.where(hp > 0)
        expected = pd.Series(
            [C13_PER_CARBON * count_element(f, "C") if t.startswith("13C") else ISOTOPE_ABUNDANCE.get(t, 0.0)
             for t, f in zip(ch["__tag"], ch["ion_formula"])], index=ch.index, dtype=float)
        rel = ratio / expected.where(expected > 0)
        ch["__ok"] = rel.between(RATIO_LO, RATIO_HI) & np.isfinite(rel)
        for (n, a), g in ch.groupby(["__neutral", "__adduct"], sort=False):
            tags[(n, a)] = set(g["__tag"]) - {"M0"}
            ratio_ok[(n, a)] = bool(g["__ok"].any())

    # per-neutral facts over the whole source (regular rows: see above)
    channels = regular.groupby("__neutral")["__adduct"].nunique()
    adduct_sets = regular.groupby("__neutral")["__adduct"].agg(lambda s: set(s))
    satellite = HALOGEN_SATELLITE.get(halogen) if halogen else None

    methods = m0["method"].astype(str)
    m0["__known"] = methods.where(methods.str.startswith("known:")).str[6:]
    m0["__iso_list"] = m0["isotopologues"].map(lambda v: len(as_list(v)) > 0)
    m0["__tied"] = m0["tied"].map(truthy)
    m0["__below"] = m0["below_assignability"].map(truthy)
    m0["__lowconf"] = m0["confidence"].map(first_word).isin(LOW_CONFIDENCE)
    m0["__deg"] = pd.to_numeric(m0["degeneracy_density"], errors="coerce")
    m0["__sat"] = m0["degeneracy_note"].astype(str).str.contains("MASS-SATURATED", regex=False)
    m0["__anchor"] = m0["anchor_peak_id"].notna() | m0["series_unit"].notna()
    m0["__res"] = m0["resolvability"].map(
        lambda v: str(v).strip() if pd.notna(v) and str(v).strip() and str(v).strip().lower() != "nan" else "")
    m0["__mz"] = pd.to_numeric(m0["mz"], errors="coerce")
    m0["__height"] = pd.to_numeric(m0["height"], errors="coerce")
    m0["__ppm"] = pd.to_numeric(m0["ppm_error_cal"], errors="coerce")
    m0["__occ"] = pd.to_numeric(m0["occurrence"], errors="coerce")

    rows = []
    for (neutral, adduct), g in m0.groupby(["__neutral", "__adduct"], sort=True):
        key = (neutral, adduct)
        t = tags.get(key, set())
        carbon_ev = any(x.startswith("13C") for x in t)
        known = g["__known"].dropna()
        seen = {v for v in g["__res"] if v}
        ion_only = bool(g["__ion_only"].any())
        aset = set() if ion_only else adduct_sets.get(neutral, set())
        rows.append(dict(
            neutral_formula=neutral, adduct=adduct,
            ion=str(g["ion_formula"].iloc[0]),
            mz=float(g["__mz"].median()) if g["__mz"].notna().any() else np.nan,
            height=float(g["__height"].median()) if g["__height"].notna().any() else np.nan,
            ppm=float(g["__ppm"].median()) if g["__ppm"].notna().any() else np.nan,
            occurrence=float(g["__occ"].median()) if g["__occ"].notna().any() else np.nan,
            tier="Assigned" if (g["tier"].astype(str) == "Assigned").any() else "Candidate",
            known_fam=str(known.iloc[0]) if len(known) else "",
            iso=bool(ratio_ok.get(key, False)) or bool(g["__iso_list"].any()),
            multiline=len(t) >= 2,
            carbon_ev=carbon_ev,
            chan2=(not ion_only) and int(channels.get(neutral, 0)) >= 2,
            anchor=(not ion_only) and bool(g["__anchor"].any()),
            branch=bool(aset & BARE_ADDUCTS) and bool(aset & CLUSTER_ADDUCTS),
            reagent_only_iso=(not ion_only) and bool(satellite) and bool(t) and not carbon_ev
            and all(x.startswith(satellite) for x in t),
            ion_only=ion_only,
            iso_labels="|".join(sorted(t)),
            tied=bool(g["__tied"].all()),
            below=bool(g["__below"].any()),
            lowconf=bool(g["__lowconf"].all()),
            degeneracy=float(g["__deg"].median()) if g["__deg"].notna().any() else np.nan,
            saturated=bool(g["__sat"].any()),
            # resolvability is measured on the trace-first path only; a source
            # that never measured it is not penalised for it
            res_ok=(not seen) or bool(seen & RESOLVED),
            resolvability="|".join(sorted(seen)),
            n_files=int(g["__file"].nunique()),
            reagent_halogen=halogen or "",
        ))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# the decision table
# ---------------------------------------------------------------------------
def _decide(r) -> tuple[str, str]:
    """(level, reason) for one evidence record. Order matters: the first
    predicate that holds wins -- docs/EVIDENCE_LEVELS.md §4."""
    deg = r.degeneracy
    hard = []
    if r.tied:
        hard.append("near-tie broken by the arbiter")
    if r.below:
        hard.append("below assignability")
    if r.lowconf:
        hard.append("engine confidence Low/Suspect")
    if hard:
        return "5b", "5b: " + "; ".join(hard)
    if r.ion_only:
        # the composition is pinned by exact mass; its own 13C line pins the ion
        # (4d: ion pinned, neutral not -- the same rung the reagent-halogen case
        # reaches by the other route); without one it is exact mass only
        if r.iso:
            return "4d", "4d: ion-only channel, composition pinned by exact mass + 13C; process open"
        return "5a", "5a: ion-only channel, exact mass only; process open"
    degenerate = bool(r.saturated) or (pd.notna(deg) and deg >= 3)
    if degenerate and r.n_axes == 0:
        what = "mass-saturated window" if r.saturated else f"{deg:g} plausible ions in the window"
        return "5b", f"5b: mass-degenerate ({what}) and no axis"
    fam = r.known_fam
    n_struct = r.n_plausible_structures
    if fam:
        if family_scope(fam) == "compound":
            if pd.notna(n_struct) and int(n_struct) == 1:
                return "2b", f"2b: curated identity ({fam}) on a one-structure formula"
            why = (f"{int(n_struct)} plausible structures" if pd.notna(n_struct)
                   else "formula not in the isomer space")
            return "3a", f"3a: curated identity ({fam}), {why}"
        return "3a", f"3a: curated class ({fam}), isomers open"
    if r.branch:
        return "3b", "3b: acid branch, the same neutral seen deprotonated and clustered"
    held = " + ".join(a for a in AXES if getattr(r, a))
    if r.n_axes == 0:
        unique = pd.notna(deg) and deg <= 1
        if unique and r.res_ok:
            res = r.resolvability if r.resolvability else "resolvability not measured"
            return "4c", f"4c: {deg:g} plausible ion in the window, {res}, no axis"
        why = []
        if pd.isna(deg):
            why.append("degeneracy not measured")
        elif deg > 1:
            why.append(f"{deg:g} plausible ions in the window")
        if not r.res_ok:
            why.append(f"peak {r.resolvability}")
        return "5a", "5a: exact mass only, no axis" + (f" ({'; '.join(why)})" if why else "")
    if not r.neutral_backed and r.reagent_only_iso:
        sat = HALOGEN_SATELLITE.get(r.reagent_halogen) or "halogen"
        return "4d", f"4d: only the reagent {sat} satellite, the ion is pinned, the neutral is not"
    if r.n_axes >= 2 and r.cross:
        outside = ("corroborated by the other source" if r.corroborated
                   else "two-line isotope envelope")
        return "4a", f"4a: {held}, {outside}"
    if r.n_axes >= 2:
        return "4b", f"4b: {held}, none outside the channel's chemistry"
    return "4b", f"4b: one corroboration ({held})"


def _axes_string(r, *, with_files: bool) -> str:
    parts = [a for a in AXES if getattr(r, a)]
    if r.multiline:
        parts.append("multiline")
    if r.carbon_ev:
        parts.append("carbon")
    if r.branch:
        parts.append("branch")
    if r.reagent_only_iso:
        parts.append("reagent_only_iso")
    if r.ion_only:
        parts.append("ion_only")
    if r.known_fam:
        parts.append(f"known:{r.known_fam}")
    if with_files:
        parts.append(f"files:{int(r.n_files)}")
    return "|".join(parts)


def _level_pairs(frames: dict, *, cross=None, isomer_space=None, with_files: bool = False) -> pd.DataFrame:
    """Level every (neutral, adduct) pair of the frames pooled as ONE source."""
    parts = []
    for label, frame in frames.items():
        f = frame.copy()
        f["__file"] = str(label)
        parts.append(f)
    if not parts:
        return pd.DataFrame(columns=["neutral_formula", "adduct", *COLUMNS])
    frame = pd.concat(parts, ignore_index=True, sort=False)
    role = _col(frame, "role").astype(str)
    halogen = detect_reagent_halogen(frame[role == "M0"])
    facts = _measure(frame, halogen=halogen)
    if facts.empty:
        return pd.DataFrame(columns=["neutral_formula", "adduct", *COLUMNS])
    cross = {str(x) for x in (cross or set())}
    structures = _structures(isomer_space)
    # an ion-only row is never corroborated: its neutral is the parent's, and a
    # source that holds it is not a second independent sighting of that neutral
    facts["corroborated"] = facts["neutral_formula"].isin(cross) & ~facts["ion_only"]
    facts["n_axes"] = facts[list(AXES)].sum(axis=1).astype(int)
    facts["cross"] = facts["corroborated"] | facts["multiline"] | facts["known_fam"].ne("")
    facts["neutral_backed"] = (facts["corroborated"] | facts["chan2"] | facts["anchor"]
                               | facts["known_fam"].ne("") | facts["carbon_ev"])
    facts["n_plausible_structures"] = pd.array(
        [structures.get(f, pd.NA) for f in facts["neutral_formula"]], dtype="Int64")
    decided = [_decide(r) for r in facts.itertuples(index=False)]
    facts["evidence_level"] = [d[0] for d in decided]
    facts["level_reason"] = [d[1] for d in decided]
    facts["evidence_axes"] = [_axes_string(r, with_files=with_files) for r in facts.itertuples(index=False)]
    return facts


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------
def compute_levels(ledger: pd.DataFrame, *, cfg=None, isomer_space=None, cross=None) -> pd.DataFrame:
    """Pure: one row per M0 row of `ledger` (index = the ledger's index) with
    `peak_id` and the four columns. `cross` = the corroborating neutral formulas
    (the other reagent channel / instrument / `--corroborate` source); `cfg` is
    accepted for stage-call symmetry and not read -- no predicate is tunable."""
    role = _col(ledger, "role").astype(str)
    m0 = ledger[role == "M0"]
    empty = pd.DataFrame({"peak_id": pd.Series(dtype=object),
                          "evidence_level": pd.Series(dtype=object),
                          "evidence_axes": pd.Series(dtype=object),
                          "level_reason": pd.Series(dtype=object),
                          "n_plausible_structures": pd.Series(dtype="Int64")})
    if m0.empty:
        return empty
    pairs = _level_pairs({"": ledger}, cross=cross, isomer_space=isomer_space, with_files=False)
    if pairs.empty:
        return empty
    key = pd.DataFrame({
        "peak_id": m0["peak_id"].values,
        "neutral_formula": _col(m0, "neutral_formula").fillna("").astype(str).values,
        "adduct": _col(m0, "adduct").fillna("").astype(str).values,
    }, index=m0.index)
    out = key.merge(pairs[["neutral_formula", "adduct", *COLUMNS]],
                    on=["neutral_formula", "adduct"], how="left")
    out.index = m0.index
    out["n_plausible_structures"] = out["n_plausible_structures"].astype("Int64")
    return out[["peak_id", *COLUMNS]]


def summarize(levels: pd.Series) -> dict:
    """{level: n} over the nine levels that can fire, in scale order, zeros dropped."""
    counts = pd.Series(levels).dropna().astype(str).value_counts()
    return {k: int(counts[k]) for k in LEVELS if k in counts.index and counts[k]}


def apply_levels(ledger: pd.DataFrame, *, cfg=None, cross=None, isomer_space=None) -> dict:
    """The `evidence` stage: write the four columns onto `ledger` in place (NA
    on every non-M0 row) and return the stage summary."""
    cross = {str(x) for x in (cross or set())}
    out = compute_levels(ledger, cfg=cfg, isomer_space=isomer_space, cross=cross)
    n = len(ledger)
    for c in ("evidence_level", "evidence_axes", "level_reason"):
        ledger[c] = pd.Series([pd.NA] * n, index=ledger.index, dtype=object)
    ledger["n_plausible_structures"] = pd.array([pd.NA] * n, dtype="Int64")
    if len(out):
        for c in ("evidence_level", "evidence_axes", "level_reason"):
            ledger.loc[out.index, c] = out[c].astype(object).values
        ledger.loc[out.index, "n_plausible_structures"] = out["n_plausible_structures"].values
    axes = {}
    for s in out["evidence_axes"].dropna().astype(str) if len(out) else []:
        for a in s.split("|"):
            if a in AXES:
                axes[a] = axes.get(a, 0) + 1
    return {"levels": summarize(out["evidence_level"]) if len(out) else {},
            "n_levelled": int(out["evidence_level"].notna().sum()) if len(out) else 0,
            "n_pairs": _n_pairs(ledger),
            "n_corroborate": len(cross), "axes": axes}


def _n_pairs(ledger: pd.DataFrame) -> int:
    role = _col(ledger, "role").astype(str)
    m0 = ledger[role == "M0"]
    if m0.empty:
        return 0
    return int(pd.DataFrame({"n": _col(m0, "neutral_formula").fillna("").astype(str),
                             "a": _col(m0, "adduct").fillna("").astype(str)}).drop_duplicates().shape[0])


def level_pooled(per_file: dict, *, cross=None, isomer_space=None) -> pd.DataFrame:
    """A batch's per-file ledgers ({label: frame}) pooled as ONE source: one row
    per (neutral_formula, adduct) over all files with the four columns and every
    fact of §3 (`chan2` sees a second adduct in ANY file, `iso` any file's
    satellite, `tied`/`lowconf` need ALL rows across files, `below` any).
    `evidence_axes` ends with `files:<n>`."""
    return _level_pairs(dict(per_file), cross=cross, isomer_space=isomer_space, with_files=True)


def stamp_merged(merged: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    """Join the pooled result onto a merged ledger by (neutral_formula, adduct)
    -- each merged row is one ion, so the join is one-to-one. A merged row whose
    reading exists in no per-file ledger (a batch-level re-read) stays NA."""
    out = merged.copy()
    n = len(out)
    for c in ("evidence_level", "evidence_axes", "level_reason"):
        out[c] = pd.Series([pd.NA] * n, index=out.index, dtype=object)
    out["n_plausible_structures"] = pd.array([pd.NA] * n, dtype="Int64")
    if not n or pairs is None or pairs.empty:
        return out
    right = pairs[["neutral_formula", "adduct", *COLUMNS]].copy()
    right["__n"] = right["neutral_formula"].fillna("").astype(str)
    right["__a"] = right["adduct"].fillna("").astype(str)
    right = right.drop_duplicates(["__n", "__a"]).set_index(["__n", "__a"])
    idx = pd.MultiIndex.from_arrays([_col(out, "neutral_formula").fillna("").astype(str).values,
                                     _col(out, "adduct").fillna("").astype(str).values])
    hit = right.reindex(idx)
    for c in ("evidence_level", "evidence_axes", "level_reason"):
        out[c] = pd.Series(hit[c].astype(object).values, index=out.index, dtype=object).where(hit[c].notna().values, pd.NA)
    out["n_plausible_structures"] = pd.array(hit["n_plausible_structures"].values, dtype="Int64")
    return out


def trim(ledger: pd.DataFrame) -> pd.DataFrame:
    """The M0 + isotope rows and the predicate columns of a ledger -- what the
    pooled batch recompute keeps per file (the rest of the ledger is on disk)."""
    role = _col(ledger, "role").astype(str)
    keep = ledger[role.isin(["M0", "iso_child"])]
    return keep[[c for c in PREDICATE_COLUMNS if c in keep.columns]].copy()


# ---------------------------------------------------------------------------
# --corroborate sources
# ---------------------------------------------------------------------------
def resolve_source(path: str) -> tuple[str, list[str]]:
    """(label, ledger csv paths) for a run dir (its per_file/*_ledger.csv, else
    merged_ledger.csv), an out-dir holding exactly one run, or a ledger CSV."""
    path = os.path.expanduser(str(path).rstrip("/"))
    if os.path.isfile(path):
        base = os.path.basename(path)
        return (base[:-4] if base.endswith(".csv") else base), [path]
    if not os.path.isdir(path):
        raise FileNotFoundError(f"no such source: {path}")
    label = os.path.basename(path)
    per_file = sorted(glob.glob(os.path.join(path, "per_file", "*_ledger.csv")))
    if per_file:
        return label, per_file
    merged = os.path.join(path, "merged_ledger.csv")
    if os.path.isfile(merged):
        return label, [merged]
    inner = [d for d in sorted(glob.glob(os.path.join(path, "*"))) if os.path.isdir(d)]
    runs = [d for d in inner if os.path.isdir(os.path.join(d, "per_file"))
            or os.path.isfile(os.path.join(d, "merged_ledger.csv"))]
    if len(runs) == 1:
        return resolve_source(runs[0])
    raise FileNotFoundError(f"no ledger under {path}")


def corroborating_neutrals(sources) -> set[str]:
    """The M0 neutral formulas of every source in `sources` (paths, or frames):
    the cross set a run is corroborated by. Ion-only rows are left out: they
    carry their parent's composition, not an independent sighting of it."""
    out: set[str] = set()
    for src in sources or []:
        frames = []
        if isinstance(src, pd.DataFrame):
            frames = [src]
        else:
            _, files = resolve_source(src)
            frames = [pd.read_csv(f, low_memory=False) for f in files]
        for f in frames:
            if "role" in f.columns:
                f = f[f["role"].astype(str) == "M0"]
            f = f[~is_ion_only(f)]
            out |= set(_col(f, "neutral_formula").dropna().astype(str))
    out.discard("")
    return out
