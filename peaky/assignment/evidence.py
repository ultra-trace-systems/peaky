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
        below assignability, a score the engine calls Low/Suspect, or (rule K,
        batch only) the labelled reagent's 14N twin refuting the cluster
        reading; also a mass-degenerate pair with no corroborating axis at all
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
        least one from outside this channel's ionization chemistry -- or, on a
        batch whose profile declares a neutral pair (rule U, row 9'), the pair
        on a formula with its own support
    4b  formula confirmed, one corroboration

The four axes: a verified isotopologue (`iso`), a second adduct channel
(`chan2`), a homologous-series or anchor tie (`anchor`), and a corroborating
source (`corroborated`: the other reagent channel, the other instrument on the
same air, or a `--corroborate` source, holding the neutral at 4b or better by
its OWN evidence -- `source_neutrals`). Levels 1 and 2a need an authentic
standard or a library spectrum and never fire.

Entry points: `apply_levels` (the `evidence` stage of assign.run: writes the
five columns in place), `compute_levels` (pure, one row per M0), `level_pooled`
(a batch's per-file ledgers as ONE source, one row per pair), `stamp_merged`
(join the pooled result onto the merged ledger by ion) and
`corroborating_neutrals` (the cross set of the `--corroborate` sources).

The claim (C13): `claim_class` reads a level as what a reader may say about
the committed formula -- `identified` (1-4a, the neutral established), `ion`
(4b-4d, the ion composition pinned, the neutral / adduct split open) or
`tentative` (5a, 5b, no level). It is stamped with the level on every committed
M0 row (per file and merged) and read by nothing upstream; `tier` stays the
engine's print-or-offer verdict and can disagree with it.
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
from peaky.chem import chemistry as C

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
#: what a committed formula lets a reader say, read off its level (C13). The
#: tier is a separate verdict (print / offer) and can disagree with it.
CLAIMS = ("identified", "ion", "tentative")
#: identified = the neutral established (a curated identity or class, the acid
#: branch, or two axes with one outside the channel); these are also the vote's
#: EVIDENCE_CLASS_GOOD / _MID sets in assign_batch, which add the corroborated
#: axis on top -- the vote class is not the claim.
CLAIM_IDENTIFIED = frozenset({"1", "2a", "2b", "3a", "3b", "4a"})
CLAIM_ION = frozenset({"4b", "4c", "4d"})
CLAIM_MEANING = {
    "identified": "the neutral is established (levels 1-4a): the formula can be reported as a compound or class",
    "ion": "the ion composition is pinned, the neutral / adduct split is open (levels 4b-4d)",
    "tentative": "exact mass only, or the assignment argues with itself (5a, 5b, or no level)",
}
#: the five columns the stage writes, in order (`claim` is a pure function of
#: `evidence_level` and is read by nothing upstream: not the tiers, not the
#: merge vote, not the cross set)
COLUMNS = ("evidence_level", "evidence_axes", "level_reason", "n_plausible_structures", "claim")
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
#: 15N and 18O are expected per atom of the ion, like 13C per carbon (heavy /
#: light natural abundance; a caret '^N' atom is already 15N and is not counted)
PER_ATOM_ABUNDANCE = {"15N": ("N", 0.00368 / 0.99632), "18O": ("O", 0.00205 / 0.99757)}
RATIO_LO, RATIO_HI = 0.5, 2.0
#: an ion carrying Br or Cl owns its M+2 region: the 81Br / 37Cl line (97 % /
#: 32 % per atom) sits 6.3 / 7.2 mDa from an 18O satellite -- one peak on a TOF,
#: a shoulder on an Orbitrap -- so an '18O' line of such an ion is not measured
#: (and a bromide ion showing an '18O' line with no 81Br line contradicts itself)
M2_OWNERS = ("Br", "Cl")
#: the atoms an isotope child's tag measures, as a key of the UNFOLDED ion
#: composition: '13C' / '13C2' -> C, '81Br' / '2x81Br' / '81Br2' / '81Br(pair)'
#: -> Br, '15N' -> N (the 14N atoms), '14N' (the light line of a 15N label) ->
#: '^N' (the labelled atoms). A generic 'M+n' child ('M') names none.
_TAG_ELEMENT = re.compile(r"^(?:\d+x)?\d+([A-Z][a-z]?)\d*(?:\(pair\))?$")

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
def claim_class(level) -> str:
    """The claim a level supports: 'identified' (1-4a), 'ion' (4b-4d) or
    'tentative' (5a, 5b, or no level). Callers decide which rows carry a claim
    at all: a committed M0 row always does; an isotope child, a reagent ion or
    an unexplained peak does not."""
    if level is None or (not isinstance(level, str) and pd.isna(level)):
        return "tentative"
    lv = str(level).strip()
    if lv in CLAIM_IDENTIFIED:
        return "identified"
    if lv in CLAIM_ION:
        return "ion"
    return "tentative"


def summarize_claims(claims) -> dict:
    """{claim: n} over the three classes in CLAIMS order, zeros kept; NA skipped."""
    counts = pd.Series(claims, dtype=object).dropna().astype(str).value_counts()
    return {k: int(counts.get(k, 0)) for k in CLAIMS}


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


def expected_ratio(tag: str, ion_formula) -> float:
    """The natural height ratio of an isotope child to its M0: 13C per carbon
    of the ion, 15N / 18O per atom of the ion, the tabulated satellite
    otherwise; 0.0 when there is none (such a line is never in the band)."""
    if tag.startswith("13C"):
        return C13_PER_CARBON * count_element(ion_formula, "C")
    if tag in PER_ATOM_ABUNDANCE:
        counts = C.parse_formula(str(ion_formula or ""))
        element, per_atom = PER_ATOM_ABUNDANCE[tag]
        if element == "O" and any(counts.get(e, 0) for e in M2_OWNERS):
            return 0.0
        return per_atom * counts.get(element, 0)
    return ISOTOPE_ABUNDANCE.get(tag, 0.0)


def tag_element(tag) -> str | None:
    """The composition key an isotope child's tag measures (see `_TAG_ELEMENT`)."""
    tag = str(tag).strip()
    match = _TAG_ELEMENT.match(tag)
    if not match:
        return None
    return "^N" if tag.startswith("14N") else match.group(1)


def neutral_elements(neutral, ion) -> set[str]:
    """The elements whose isotope line speaks for the NEUTRAL (C17): the neutral
    supplies MORE THAN HALF of the ion's atoms of the element. A 15N line of a
    urea adduct of an N-free neutral, the 81Br line of a bromide adduct, or the
    18O line of formic acid's nitrate cluster (2 of 5 O) measure the reagent;
    the 18O line of a C10H16O4 urea adduct (4 of 5 O) measures the neutral.
    Counted on the unfolded composition: a natural 15N line measures the 14N
    atoms, so a labelled adduct's '^N' is not the neutral's N."""
    own = C.parse_formula(str(neutral or ""))
    ion_counts = C.parse_formula(str(ion or ""))
    return {el for el, n in own.items() if n > 0 and 2 * n > ion_counts.get(el, 0)}


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


#: rule K's one-channel fold: on a labelled-nitrate batch the 14N and the 15N
#: nitrate cluster of one neutral are one channel seen through the reagent's two
#: isotopologues, never two
LABEL_FOLD = {"[M+NO3]-": "[M+^NO3]-", "[M+15NO3]-": "[M+^NO3]-"}


def _measure(frame: pd.DataFrame, *, halogen: str | None, alien=None, fold=None) -> pd.DataFrame:
    """One row of evidence per (neutral, adduct) the source committed. `frame`
    carries a `__file` column (the file each row came from). `alien` (rule K):
    {(neutral, adduct)} kept out of the per-neutral pools in both directions,
    like an ion-only row; `fold` maps adducts that count as one channel."""
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
    alien = {(str(n), str(a)) for n, a in (alien or set())}
    m0["__alien"] = [k in alien for k in zip(m0["__neutral"], m0["__adduct"])] if alien else False
    # the per-neutral facts (second channel, acid branch) are read over the
    # REGULAR rows only: an ion-only row is the parent's own composition on
    # another adduct and must not count as a second channel for it (nor a 14N
    # nitrate line rule K could not tie to its neutral's 15N cluster)
    regular = m0[~m0["__ion_only"] & ~m0["__alien"]]

    # satellites: each child hangs off the M0 with peak_id == parent_peak_id in
    # the same file; the tag is the label before any '+' (13C+1 -> 13C)
    iso = frame[role == "iso_child"]
    parents = (m0.drop_duplicates(["__file", "peak_id"])
               [["__file", "peak_id", "__neutral", "__adduct", "height", "ion_formula"]]
               .rename(columns={"peak_id": "__pid", "height": "__h_parent"}))
    tags: dict[tuple, set] = {}
    ratio_ok: dict[tuple, bool] = {}
    lined: dict[tuple, set] = {}
    if len(iso) and len(parents):
        ch = iso.loc[iso["parent_peak_id"].notna(),
                     ["__file", "parent_peak_id", "iso_label", "height"]]
        ch = ch.merge(parents, left_on=["__file", "parent_peak_id"], right_on=["__file", "__pid"], how="inner")
        ch["__tag"] = ch["iso_label"].map(lambda v: str(v).split("+")[0].strip() if pd.notna(v) else "")
        ch = ch[ch["__tag"] != ""]
        hp = pd.to_numeric(ch["__h_parent"], errors="coerce")
        hc = pd.to_numeric(ch["height"], errors="coerce")
        ratio = hc / hp.where(hp > 0)
        expected = pd.Series([expected_ratio(t, f) for t, f in zip(ch["__tag"], ch["ion_formula"])],
                             index=ch.index, dtype=float)
        rel = ratio / expected.where(expected > 0)
        ch["__ok"] = rel.between(RATIO_LO, RATIO_HI) & np.isfinite(rel)
        ch["__el"] = ch["__tag"].map(tag_element)
        for (n, a), g in ch.groupby(["__neutral", "__adduct"], sort=False):
            tags[(n, a)] = set(g["__tag"]) - {"M0"}
            ratio_ok[(n, a)] = bool(g["__ok"].any())
            # the elements with an in-band line (C17: `multiline` counts these)
            lined[(n, a)] = set(g.loc[g["__ok"], "__el"].dropna())

    # per-neutral facts over the whole source (regular rows: see above)
    channels = regular.assign(__ch=regular["__adduct"].map(lambda a: (fold or {}).get(a, a))) \
        .groupby("__neutral")["__ch"].nunique()
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
        ion = str(g["ion_formula"].iloc[0])
        own = lined.get(key, set()) & neutral_elements(neutral, ion)
        known = g["__known"].dropna()
        seen = {v for v in g["__res"] if v}
        ion_only = bool(g["__ion_only"].any())
        outside = ion_only or bool(np.any(g["__alien"]))
        aset = set() if outside else adduct_sets.get(neutral, set())
        rows.append(dict(
            neutral_formula=neutral, adduct=adduct,
            ion=ion,
            mz=float(g["__mz"].median()) if g["__mz"].notna().any() else np.nan,
            height=float(g["__height"].median()) if g["__height"].notna().any() else np.nan,
            ppm=float(g["__ppm"].median()) if g["__ppm"].notna().any() else np.nan,
            occurrence=float(g["__occ"].median()) if g["__occ"].notna().any() else np.nan,
            tier="Assigned" if (g["tier"].astype(str) == "Assigned").any() else "Candidate",
            known_fam=str(known.iloc[0]) if len(known) else "",
            iso=bool(ratio_ok.get(key, False)) or bool(g["__iso_list"].any()),
            multiline=len(own) >= 2,
            multiline_elements="|".join(sorted(own)),
            carbon_ev=carbon_ev,
            chan2=(not outside) and int(channels.get(neutral, 0)) >= 2,
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
    if getattr(r, "label_veto", False):
        # rule K (C18): the 14N twin of a labelled cluster is absent where the
        # reagent's own impurity puts it, or a 14N line has no 15N partner or
        # runs above its cluster share -- the cluster reading is refuted
        note = getattr(r, "label_note", "") or ""
        hard.append("the reagent's two isotopologues refute the cluster reading" + (f" ({note})" if note else ""))
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
        els = getattr(r, "multiline_elements", "") or ""
        outside = ("corroborated by the other source" if r.corroborated
                   else f"isotope lines of two of the neutral's elements ({els.replace('|', ', ')})"
                   if els else "isotope lines of two of the neutral's elements")
        return "4a", f"4a: {held}, {outside}"
    if getattr(r, "upair", False):
        # row 9' (rule U): the profile's neutral pair -- the bare and the cluster
        # ion at exact mass, co-varying, clean, N-free CHO -- establishes the
        # neutral as the acid branch does; the formula needs its own support
        unique = pd.notna(deg) and deg <= 1
        if r.iso:
            return "4a", f"4a: {held}, the neutral pair establishes the neutral (upair), formula by isotope"
        if unique and r.res_ok:
            return "4a", (f"4a: {held}, the neutral pair establishes the neutral (upair), "
                          f"formula unique on a resolved peak")
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
    if getattr(r, "upair", False):
        parts.append("upair")
    if getattr(r, "label_untie", False):
        parts.append("label_untie")
    if getattr(r, "label_veto", False):
        parts.append("label_veto")
    if r.known_fam:
        parts.append(f"known:{r.known_fam}")
    if with_files:
        parts.append(f"files:{int(r.n_files)}")
    return "|".join(parts)


def _level_pairs(frames: dict, *, cross=None, isomer_space=None, with_files: bool = False,
                 upair=None, label=None) -> pd.DataFrame:
    """Level every (neutral, adduct) pair of the frames pooled as ONE source.
    `upair`: the neutrals whose declared neutral pair holds (rule U, measured by
    batch/neutral_pairs.py on the batch time series; pooled only). `label`: the
    labelled-nitrate twin facts of rule K (batch/label_twins.facts; pooled only):
    {'untie': {(n, a)} whose arbiter tie the 15N sibling breaks, 'veto': {(n, a):
    note} the reagent's isotopologues refute, 'alien': {(n, a)} 14N lines kept out
    of their neutral's pools}; given, the 14N and 15N nitrate clusters of one
    neutral also count as one channel."""
    parts = []
    for src, frame in frames.items():
        f = frame.copy()
        f["__file"] = str(src)
        parts.append(f)
    if not parts:
        return pd.DataFrame(columns=["neutral_formula", "adduct", *COLUMNS])
    frame = pd.concat(parts, ignore_index=True, sort=False)
    role = _col(frame, "role").astype(str)
    halogen = detect_reagent_halogen(frame[role == "M0"])
    label = label or None
    facts = _measure(frame, halogen=halogen, alien=(label or {}).get("alien"),
                     fold=LABEL_FOLD if label else None)
    if facts.empty:
        return pd.DataFrame(columns=["neutral_formula", "adduct", *COLUMNS])
    cross = {str(x) for x in (cross or set())}
    structures = _structures(isomer_space)
    # an ion-only row is never corroborated: its neutral is the parent's, and a
    # source that holds it is not a second independent sighting of that neutral
    facts["corroborated"] = facts["neutral_formula"].isin(cross) & ~facts["ion_only"]
    # the neutral pair is a fact about the neutral, never an axis and never in
    # `cross`; like chan2 it never lands on an ion-only pair
    upair = {str(x) for x in (upair or set())}
    facts["upair"] = facts["neutral_formula"].isin(upair) & ~facts["ion_only"]
    # rule K: facts about one reading (a pair), never an axis, never in `cross`;
    # the untie clears the arbiter's tie only where the pair is tied at all
    keys = list(zip(facts["neutral_formula"].astype(str), facts["adduct"].astype(str)))
    untie = {(str(n), str(a)) for n, a in ((label or {}).get("untie") or set())}
    facts["label_untie"] = pd.Series([k in untie for k in keys], index=facts.index, dtype=bool) \
        & facts["tied"] & ~facts["ion_only"]
    facts.loc[facts["label_untie"], "tied"] = False
    veto = {(str(n), str(a)): str(v or "") for (n, a), v in ((label or {}).get("veto") or {}).items()}
    facts["label_veto"] = pd.Series([k in veto for k in keys], index=facts.index, dtype=bool) & ~facts["ion_only"]
    facts["label_note"] = [veto.get(k, "") if v else "" for k, v in zip(keys, facts["label_veto"])]
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
    facts["claim"] = facts["evidence_level"].map(claim_class)
    return facts


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------
def compute_levels(ledger: pd.DataFrame, *, cfg=None, isomer_space=None, cross=None) -> pd.DataFrame:
    """Pure: one row per M0 row of `ledger` (index = the ledger's index) with
    `peak_id` and the five columns. `cross` = the corroborating neutral formulas
    (the other reagent channel / instrument / `--corroborate` source); `cfg` is
    accepted for stage-call symmetry and not read -- no predicate is tunable."""
    role = _col(ledger, "role").astype(str)
    m0 = ledger[role == "M0"]
    empty = pd.DataFrame({"peak_id": pd.Series(dtype=object),
                          "evidence_level": pd.Series(dtype=object),
                          "evidence_axes": pd.Series(dtype=object),
                          "level_reason": pd.Series(dtype=object),
                          "n_plausible_structures": pd.Series(dtype="Int64"),
                          "claim": pd.Series(dtype=object)})
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
    # every committed M0 row carries a claim; one with no level reads tentative
    out["claim"] = out["evidence_level"].map(claim_class)
    return out[["peak_id", *COLUMNS]]


def summarize(levels: pd.Series) -> dict:
    """{level: n} over the nine levels that can fire, in scale order, zeros dropped."""
    counts = pd.Series(levels).dropna().astype(str).value_counts()
    return {k: int(counts[k]) for k in LEVELS if k in counts.index and counts[k]}


def apply_levels(ledger: pd.DataFrame, *, cfg=None, cross=None, isomer_space=None) -> dict:
    """The `evidence` stage: write the five columns onto `ledger` in place (NA
    on every non-M0 row, `claim` included: only a committed formula makes a
    claim) and return the stage summary."""
    cross = {str(x) for x in (cross or set())}
    out = compute_levels(ledger, cfg=cfg, isomer_space=isomer_space, cross=cross)
    n = len(ledger)
    for c in ("evidence_level", "evidence_axes", "level_reason", "claim"):
        ledger[c] = pd.Series([pd.NA] * n, index=ledger.index, dtype=object)
    ledger["n_plausible_structures"] = pd.array([pd.NA] * n, dtype="Int64")
    if len(out):
        for c in ("evidence_level", "evidence_axes", "level_reason", "claim"):
            ledger.loc[out.index, c] = out[c].astype(object).values
        ledger.loc[out.index, "n_plausible_structures"] = out["n_plausible_structures"].values
    axes = {}
    for s in out["evidence_axes"].dropna().astype(str) if len(out) else []:
        for a in s.split("|"):
            if a in AXES:
                axes[a] = axes.get(a, 0) + 1
    return {"levels": summarize(out["evidence_level"]) if len(out) else {},
            "claims": summarize_claims(out["claim"] if len(out) else []),
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


def level_pooled(per_file: dict, *, cross=None, isomer_space=None, upair=None, label=None) -> pd.DataFrame:
    """A batch's per-file ledgers ({label: frame}) pooled as ONE source: one row
    per (neutral_formula, adduct) over all files with the four columns and every
    fact of §3 (`chan2` sees a second adduct in ANY file, `iso` any file's
    satellite, `tied`/`lowconf` need ALL rows across files, `below` any).
    `upair` is the neutral-pair set of rule U (batch/neutral_pairs.neutrals);
    `label` is rule K's labelled-nitrate twin facts (batch/label_twins.facts);
    both exist only here, on the pooled batch.
    `evidence_axes` ends with `files:<n>`."""
    return _level_pairs(dict(per_file), cross=cross, isomer_space=isomer_space, with_files=True, upair=upair,
                        label=label)


def stamp_merged(merged: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    """Join the pooled result onto a merged ledger by (neutral_formula, adduct)
    -- each merged row is one ion, so the join is one-to-one. A merged row whose
    reading exists in no per-file ledger (a batch-level re-read) stays NA and
    its `claim` reads tentative: every merged row is a committed reading."""
    out = merged.copy()
    n = len(out)
    for c in ("evidence_level", "evidence_axes", "level_reason"):
        out[c] = pd.Series([pd.NA] * n, index=out.index, dtype=object)
    out["n_plausible_structures"] = pd.array([pd.NA] * n, dtype="Int64")
    out["claim"] = pd.Series(["tentative"] * n, index=out.index, dtype=object)
    if not n or pairs is None or pairs.empty:
        return out
    # the claim is re-read off the joined level, so a pairs frame without one joins too
    right = pairs[["neutral_formula", "adduct", *(c for c in COLUMNS if c != "claim")]].copy()
    right["__n"] = right["neutral_formula"].fillna("").astype(str)
    right["__a"] = right["adduct"].fillna("").astype(str)
    right = right.drop_duplicates(["__n", "__a"]).set_index(["__n", "__a"])
    idx = pd.MultiIndex.from_arrays([_col(out, "neutral_formula").fillna("").astype(str).values,
                                     _col(out, "adduct").fillna("").astype(str).values])
    hit = right.reindex(idx)
    for c in ("evidence_level", "evidence_axes", "level_reason"):
        out[c] = pd.Series(hit[c].astype(object).values, index=out.index, dtype=object).where(hit[c].notna().values, pd.NA)
    out["n_plausible_structures"] = pd.array(hit["n_plausible_structures"].values, dtype="Int64")
    out["claim"] = pd.Series([claim_class(v) for v in out["evidence_level"]], index=out.index, dtype=object)
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


#: The level a --corroborate source must hold a neutral at, by its OWN evidence,
#: for its sighting to count as the `corroborated` axis: the formula confirmed
#: by an axis of the source's own -- 4b -- or better. Below that the source's
#: grid only ENUMERATED the same formula at a peak (4c unopposed but
#: unconfirmed, 4d the ion only, 5a exact mass alone, 5b arguing with itself):
#: two grids agreeing, not a second sighting of the neutral. Measured on the
#: same-air pair before the rule: of 1521 neutrals the labelled-nitrate
#: Orbitrap offered the TOF, 379 pass; of 3728 the TOF offered the Orbitrap,
#: 435 (3437 of its pairs are 5b -- a 10k-resolution TOF can seldom pin a
#: formula), and of the TOF's 49 vote winners lifted by the axis alone, 28
#: rested on an Orbitrap 5a / 5b (docs/EVIDENCE_LEVELS.md §6.4).
CORROBORATE_MAX_LEVEL = "4b"
_AXES_OWN = frozenset(a for a in AXES if a != "corroborated")


def _rank(level) -> int:
    return LEVEL_ORDER.index(level) if level in LEVEL_ORDER else len(LEVEL_ORDER)


def _stored_own_good(level, axes, max_level: str) -> bool:
    """A merged ledger's stored level read WITHOUT its own `corroborated` axis
    (a merged row carries no predicate column to re-level it from): good when
    the level is `max_level` or better and it still holds an axis of its own
    once `corroborated` is taken away -- a level the axis alone produced (a 4b
    of one corroboration, a known-family row with no other axis) does not count."""
    lv = str(level) if pd.notna(level) else ""
    if _rank(lv) > _rank(max_level):
        return False
    parts = set(str(axes).split("|")) if pd.notna(axes) else set()
    if "corroborated" not in parts:
        return True
    own = parts & _AXES_OWN
    if not own or (own == {"iso"} and "reagent_only_iso" in parts):
        return False
    return True


def source_neutrals(per_file: dict, *, isomer_space=None, max_level: str = CORROBORATE_MAX_LEVEL) -> set[str]:
    """The neutral formulas a source ({label: ledger}) holds at `max_level` or
    better by its OWN evidence: the ledgers pooled as ONE source and levelled
    with NO cross set (`level_pooled`, the batch's own merged-row level), so a
    source that was itself run with --corroborate cannot hand a run back the
    agreement it got from it. Ion-only pairs never count. A ledger without a
    `role` column is a merged ledger: it carries none of the predicate columns,
    so its stored `evidence_level` is read instead, without its own
    `corroborated` axis (`_stored_own_good`)."""
    frames = {k: f for k, f in per_file.items() if "role" in f.columns}
    merged = {k: f for k, f in per_file.items() if "role" not in f.columns}
    out: set[str] = set()
    if frames:
        pairs = level_pooled({k: trim(f) for k, f in frames.items()}, cross=None, isomer_space=isomer_space)
        if len(pairs):
            ok = (pairs["evidence_level"].map(_rank) <= _rank(max_level)) & ~pairs["ion_only"].astype(bool)
            out |= set(pairs.loc[ok, "neutral_formula"].astype(str))
    for label, f in merged.items():
        if "evidence_level" not in f.columns:
            raise ValueError(
                f"--corroborate source {label!r} carries neither the per-file predicate columns nor an "
                "evidence_level column (a merged ledger from before the levels?) -- name the run dir, "
                "whose per_file/ ledgers the source's own levels are computed from")
        f = f[~is_ion_only(f)]
        ok = np.array([_stored_own_good(lv, ax, max_level)
                       for lv, ax in zip(f["evidence_level"], _col(f, "evidence_axes", ""))], dtype=bool)
        out |= set(_col(f, "neutral_formula").loc[ok].dropna().astype(str))
    out.discard("")
    return out


def corroborating_neutrals(sources) -> set[str]:
    """The cross set a run is corroborated by: for every source in `sources`
    (a run dir, an out-dir holding one run, a ledger CSV, or a frame), the
    neutral formulas it holds at level <= 4b by its own evidence
    (`source_neutrals`). A run dir is read through its per-file ledgers, which
    carry every fact a level reads."""
    out: set[str] = set()
    for src in sources or []:
        if isinstance(src, pd.DataFrame):
            per_file = {"": src}
        else:
            label, files = resolve_source(src)
            per_file = {(os.path.basename(f) if len(files) > 1 else label): pd.read_csv(f, low_memory=False)
                        for f in files}
        out |= source_neutrals(per_file)
    return out
