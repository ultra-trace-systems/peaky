"""Evidence levels -- what the evidence behind a committed formula is worth.

`tier` says whether the engine will *print* a formula (Assigned) or only *offer*
it (Candidate). The evidence level says what the evidence behind a committed
formula is worth, on a scale numbered downward (1 = best) after Schymanski et
al. (2014), Environ. Sci. Technol. 48, 2097, adapted to chemical ionization:
the evidence scale of peaky 0.10.0 (`SCALE_RELEASE`; the vocabulary lives in
peaky.assignment.levels.scale, the machinery in peaky.assignment.levels, the
contract in docs/EVIDENCE_LEVELS.md).

    3c  ion established, the neutral / adduct split pinned, and a NAMED
        context-list entry names the neutral (claim: identified)
    4a  ion established, split pinned, and a positive fact (an own isotope line
        of the neutral's elements, the 15N label, an NH4 adduct tracking its
        parent) (claim: neutral)
    4b  ion established; the split open, pinned without a positive fact, or the
        channel reads the ion only (claim: ion)
    5a  a competitor ion is left in the calibrated window (claim: tentative)
    5b  rejected by a check, or nothing could be enumerated or tested
    reagent  a reagent ion or reagent cluster (a bucket, not a level)
    NA  not assessed on this instrument class: the scale needs a width model
        resolving >= 50 000 at m/z 200 (`ORBITRAP_R200`); a TOF-class or
        class-less source reads NA and nothing else is computed

Levels 1 and 2 (an authentic standard, a library spectrum) are defined and
never assigned. One level per committed M0 row, computed per (neutral_formula,
adduct) over a SOURCE: the batch's pooled files (`level_batch`, every
file-count minimum 3; stamped on the merged ledger by `stamp_merged`) or one
file alone (`apply_levels`, the per-file `evidence` stage, "adapted" minima 1).

Entry points: `apply_levels`, `level_batch`, `stamp_merged`, `level_source`
(any `Source`: `source_from_frames` / `source_from_run_dir`),
`partners_from`, `summarize`, `claim_class`, `summarize_claims`.

THE MERGE VOTE'S EVIDENCE CLASS (private to the vote, never a user-facing
level): the batch merge ranks a cluster's ions by the class of their per-file
readings before the file count (batch/assign_batch.py `_vote`). That class is
computed here, privately, by the pre-0.10.0 decision over the per-file facts
(`_measure` / `_level_pairs` / `_decide` and the `--corroborate` cross set
`vote_cross_neutrals`), so the vote -- the only path from evidence to
assignments -- is unchanged by the scale: `vote_classes` returns 2 (the
formula confirmed and the neutral backed by two axes or a corroborating
source), 1 (the formula confirmed) or 0 (unconfirmed) per M0 row. The facts
`_level_pairs` measures (iso_veto, label_veto, lowconf, below, ion_only, ...)
are also the scale's step-0 inputs.
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
from peaky.assignment.ledger import lead_setters
from peaky.assignment.levels import scale as SCALE
from peaky.chem import chemistry as C
from peaky.chem import isotopes as ISO

# ---------------------------------------------------------------------------
# the scale (peaky.assignment.levels.scale): levels, buckets, claims, columns
# ---------------------------------------------------------------------------
SCALE_RELEASE = SCALE.SCALE_RELEASE
LEVEL_ORDER = list(SCALE.LEVEL_ORDER)
LEVELS = list(SCALE.LEVELS)
BUCKETS = list(SCALE.BUCKETS)
LEVEL_MEANING = dict(SCALE.LEVEL_MEANING)
CLAIMS = SCALE.CLAIMS
CLAIM_REAGENT = SCALE.CLAIM_REAGENT
CLAIM_NA = SCALE.CLAIM_NA
#: the four claims and the two buckets reported beside them, in that order
CLAIM_KEYS = SCALE.CLAIM_KEYS
CLAIM_IDENTIFIED = SCALE.CLAIM_IDENTIFIED
CLAIM_NEUTRAL = SCALE.CLAIM_NEUTRAL
CLAIM_ION = SCALE.CLAIM_ION
CLAIM_MEANING = dict(SCALE.CLAIM_MEANING)
#: the columns every committed M0 row carries (empty off M0)
COLUMNS = SCALE.COLUMNS
claim_class = SCALE.claim_class
summarize = SCALE.summarize
summarize_claims = SCALE.summarize_claims
#: THE side-channel switch: formate, acetate, CO3-, O2-, O3-, NH4+ (except a
#: positive run's own [M+NH4]+), Na+ and chloride never enter the split grid or
#: the route classes while locked. UNLOCKED is the sweep hook: a channel name
#: there (e.g. "formate") is unlocked alone.
SIDE_CHANNELS_LOCKED = True
UNLOCKED: frozenset = frozenset()

# ---------------------------------------------------------------------------
# the merge vote's private evidence class (the pre-0.10.0 decision; never shown)
# ---------------------------------------------------------------------------
_SERIES_ORDER = ["1", "2a", "2b", "3a", "3b", "4a", "4b", "4c", "4d", "5a", "5b"]
#: the vote's class 2: the neutral established (a curated identity or class,
#: the acid branch, or two axes with one outside the channel) -- plus any
#: reading the --corroborate source backs (`_AXES` 'corroborated')
_VOTE_GOOD = frozenset({"1", "2a", "2b", "3a", "3b", "4a"})
#: the vote's class 1: the formula or the ion confirmed, the neutral not
_VOTE_MID = frozenset({"4b", "4c", "4d"})
#: what a vote note prints for a class (never a level name)
VOTE_CLASS_TEXT = {2: "neutral backed", 1: "formula confirmed", 0: "unconfirmed"}
VOTE_CLASS_MEANING = {
    2: "the formula is confirmed and the neutral backed by two axes or a corroborating source",
    1: "the formula is confirmed",
    0: "unconfirmed: exact mass alone, or the reading argues with itself",
}
_SERIES_COLUMNS = ("evidence_level", "evidence_axes", "level_reason", "n_plausible_structures", "claim")
#: the four axes of the private decision, in the order its axes string lists them
_AXES = ("iso", "chan2", "anchor", "corroborated")


def _series_claim(level) -> str:
    """The private decision's own claim column (kept so `_level_pairs` returns
    the frame it always did; read by nothing user-facing)."""
    if level is None or (not isinstance(level, str) and pd.isna(level)):
        return "tentative"
    lv = str(level).strip()
    if lv in _VOTE_GOOD:
        return "identified"
    if lv in _VOTE_MID:
        return "ion"
    return "tentative"


#: what the batch checks and the vote class read per file -- `trim()` keeps these of a ledger
PREDICATE_COLUMNS = (
    "role", "peak_id", "parent_peak_id", "iso_label", "neutral_formula", "adduct",
    "ion_formula", "mz", "height", "tier", "method", "confidence", "tied",
    "below_assignability", "degeneracy_density", "degeneracy_note", "resolvability",
    "series_unit", "anchor_peak_id", "isotopologues", "ppm_error_cal", "occurrence",
    # C19(c): the lead half of the old below_assignability (ledger.ASSIGNABILITY_FLAGS);
    # a ledger written before the split has no such column and reads False
    "tentative_lead",
    # C11+b: which setter made a row a lead (ledger.LEAD_BY); a ledger written
    # before it has no such column
    "lead_by",
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
#: an Orbitrap-class width model resolves at least this at m/z 200 (the batch
#: checks' split, batch/iso_checks.ORBITRAP_R200)
ORBITRAP_R200 = 50_000.0


def instrument(resolution) -> tuple[str | None, object]:
    """(class, FWHM) of a width model -- a chem.resolution.Resolution, its
    `as_dict` (a run's batch_summary `resolution`) or a resolving power:
    ('orbitrap' | 'tof', FWHM(m/z) in Da). (None, None) without one: the
    source is class-less (a ledger CSV, a fixture, an old run)."""
    from peaky.chem.resolution import Resolution
    if resolution is None:
        return None, None
    if isinstance(resolution, dict):
        if resolution.get("coef") is None:
            return None, None
        rp = Resolution.from_dict(resolution)
    else:
        rp = Resolution.coerce(resolution)
    r = rp.r_at(200.0)
    return ("orbitrap" if np.isfinite(r) and r >= ORBITRAP_R200 else "tof"), rp.fwhm

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


def expected_ratio(tag: str, ion_formula, committed: dict | None = None) -> float:
    """The natural height ratio of an isotope line to its parent (C11+c: the
    count-aware expectation relative to the COMMITTED parent line, `committed`
    its heavy configuration, {} / None = the mono line): the joint probability
    of the label's heavy atoms over the ion's atoms (13C per carbon, 13C2
    C(n, 2) x 0.0107^2, one of two Br 2 x 0.9728, 15N / 18O per atom) divided
    by the committed line's; 0.0 when there is none (such a line is never in
    the band: an element the ion lacks, an '18O' line of a Br / Cl ion, a
    '14N' line, a generic 'M' label). The levels read every child through
    chem/isotopes.resolve_child; this is its parent-relative reading."""
    counts = {e: v for e, v in C.parse_formula(str(ion_formula or "")).items() if v}
    shift = ISO.resolve_child(tag, counts, {}, 0.0)["shift"]
    r = ISO.resolve_child(tag, counts, dict(committed or {}), shift if np.isfinite(shift) else 0.0)
    return float(r["expected"])


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


def ion_composition(neutral, adduct, ion) -> dict:
    """The ION's element counts: its stored ion formula when that carries a
    charge sign, else neutral + adduct (a ledger row can hold the NEUTRAL in
    `ion_formula` -- then the reagent's atoms are only in the adduct). A
    labelled reagent atom is '^N' either way ('[M+^NO3]-' adds 15N, not N:
    read as 14N the ion's mass sits 0.997 Da low)."""
    s = str(ion or "").strip() if not (isinstance(ion, float) and np.isnan(ion)) else ""
    if s.endswith(("+", "-")):
        return C.parse_formula(s)
    from peaky.assignment.tiers import _ion_counts
    return _ion_counts(str(neutral or ""), str(adduct or ""), labelled=True) or C.parse_formula(s)


def carries_reagent(neutral, adduct, ion, halogen) -> bool:
    """The ION carries more of the reagent halogen than the neutral does: only
    then can a line of it be the reagent's (an 81Br line of a Br-free ion, or of
    a brominated neutral's nitrate cluster, is not the bromide reagent's). The
    reagent-only flag reads this alone since C11+c released the 2026-09-27 hold
    on Br-free ions: under the count-aware band a Br-free ion's 81Br line is no
    line of it (expected 0, never in band), so its own pattern refutes the
    reading instead of the flag holding it at 4d."""
    if not halogen:
        return False
    return ion_composition(neutral, adduct, ion).get(halogen, 0) > C.parse_formula(str(neutral or "")).get(halogen, 0)


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
#: `halogen=` default of the level entry points: name it from the committed
#: cluster adducts (`detect_reagent_halogen`), for a caller that knows no channels.
DETECT_HALOGEN = "detect"


def channel_halogen(adducts) -> str | None:
    """The reagent halogen named by a run's DECLARED analyte channels (C43): Br
    for a channel set holding [M+Br]- or [M+HBr+Br]- (a bromide or mixed
    nitrate/bromide reagent), likewise Cl / I; None for a halogen-free set.

    The run's profile says which reagent it is; a count of the committed cluster
    adducts (`detect_reagent_halogen`) only guesses it, and on a mixed Br-/NO3-
    TOF it flips with every change that moves a few readings between the two
    channels (3907 [M+NO3]- to 3847 [M+Br]- per-file M0 on the rebased trunk:
    None, and the reagent-81Br rule went off for the whole run)."""
    for a in adducts or ():
        a = str(a).strip()
        for x in ("Br", "Cl", "I"):
            if a in (f"[M+{x}]-", f"[M+H{x}+{x}]-"):
                return x
    return None


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


#: the lead setters a halogen lock answers (rule H, C11+b; ledger.LEAD_SETTERS):
#: a speculative residual fit (its three reasons), a reference-list match too dim
#: to confirm its isotopes and -- only where the locked halogen is its sole
#: violation (`budget_ok`, the CF2 analogue) -- a commit outside the element
#: budget. The radical anion and the reagent-N re-read are never lifted: their
#: neutrals carry no halogen, and their Low confidence stays hard.
LIFTABLE_LEADS = frozenset({"spec_n3", "spec_gapfill", "spec_minor", "reflist_dim", "off_budget"})


def lead_liftable(setters, budget_ok: bool) -> bool:
    """Whether a lock lifts one lead row whose `lead_by` names `setters`: every
    setter liftable, the element budget's only where `budget_ok`. A row with no
    code (a ledger written before `lead_by`) may be the budget setter's: it lifts
    only where `budget_ok`."""
    setters = frozenset(setters or ())
    if not setters:
        return bool(budget_ok)
    return setters <= LIFTABLE_LEADS and ("off_budget" not in setters or bool(budget_ok))


_NO_LINES = ISO.line_facts([], None)


def _num(v) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return x


def _judge_children(frame: pd.DataFrame, role: pd.Series, m0: pd.DataFrame, *, satellite, resolution,
                    per_file: bool) -> tuple[dict, list]:
    """The isotope children and lists of one source judged against their
    committed parents (chem/isotopes.judge_source): ({(neutral, adduct): the
    facts its kept lines give (isotopes.line_facts)}, [one bool per m0 row: its
    isotopologues list holds a line that answers the same question])."""
    klass, fwhm = instrument(resolution)
    files = m0["__file"].astype(str)
    pids = m0["peak_id"].astype(str)
    ions = m0["ion_formula"]
    counts_of: dict = {}
    parents: dict = {}
    pair_of: dict = {}
    for f, pid, n, a, ion, mz, h, pc in zip(files, pids, m0["__neutral"], m0["__adduct"], ions, m0["mz"],
                                            m0["height"], m0["ppm_error_cal"]):
        key = (f, pid)
        if key in parents:
            continue
        ck = (n, a, ion if isinstance(ion, str) else "")
        if ck not in counts_of:
            counts_of[ck] = ({e: v for e, v in ion_composition(n, a, ion).items() if v}, ISO.ion_sign(ion, a))
        counts, sign = counts_of[ck]
        parents[key] = dict(mz=_num(mz), height=_num(h), pcal=_num(pc), counts=counts, sign=sign)
        pair_of[key] = (n, a)
    iso = frame[role == "iso_child"]
    children = []
    if len(iso) and parents:
        labels = _col(iso, "iso_label")
        for f, pid, par, lab, mz, h in zip(iso["__file"].astype(str), _col(iso, "peak_id").astype(str),
                                           _col(iso, "parent_peak_id"), labels, _col(iso, "mz"), _col(iso, "height")):
            if not isinstance(par, str) and pd.isna(par):
                continue
            key = (f, str(par))
            if key not in parents:
                continue
            lab = str(lab).strip() if pd.notna(lab) else ""
            if not lab.split("+")[0].strip():
                continue
            children.append(dict(file=f, peak_id=pid, parent=str(par), label=lab, mz=_num(mz), height=_num(h)))
    lists = []
    for f, pid, v in zip(files, pids, m0["isotopologues"]):
        entries = as_list(v)
        lists.append((f, pid, entries))
    readable = frame[role.isin(["M0", "iso_child"])]
    rows = {str(f): (pd.to_numeric(_col(g, "mz"), errors="coerce").to_numpy(float),
                     pd.to_numeric(_col(g, "height"), errors="coerce").to_numpy(float),
                     _col(g, "peak_id").astype(str).to_numpy(object))
            for f, g in readable.groupby(readable["__file"].astype(str), sort=False)}
    judged = ISO.judge_source(children, parents, [x for x in lists if x[2]], rows, klass=klass, fwhm=fwhm,
                              guard=per_file)
    kept: dict = {}
    for c, verdict in zip(children, judged["children"]):
        if verdict["keep"]:
            kept.setdefault(pair_of[(c["file"], c["parent"])], []).append(verdict)
    # the neutral's atoms of the reagent halogen: what the adduct supplies is the rest (D4's full-count line)
    halogen = ISO.ISOTOPE_ELEMENT[satellite] if satellite else None
    facts = {k: ISO.line_facts(v, satellite, C.parse_formula(str(k[0])).get(halogen, 0) if halogen else 0)
             for k, v in kept.items()}
    listed = iter(judged["lists"])
    list_ok = [bool(next(listed)) if entries else False for _f, _p, entries in lists]
    return facts, list_ok


def _measure(frame: pd.DataFrame, *, halogen: str | None, alien=None, fold=None, lift=None,
             resolution=None, per_file: bool = False) -> pd.DataFrame:
    """One row of evidence per (neutral, adduct) the source committed. `frame`
    carries a `__file` column (the file each row came from). `resolution`: the
    source's width model (`instrument`), None for a class-less source;
    `per_file`: the source is one file levelled inside its run. `alien` (rule K):
    {(neutral, adduct)} kept out of the per-neutral pools in both directions,
    like an ion-only row; `fold` maps adducts that count as one channel. `lift`
    (rule H): {(neutral, adduct): lock fact} -- a pair it names whose flagged
    rows are all liftable leads (`lead_liftable`) and none below assignability
    is lifted: its lead reads False, the lock is its isotope axis, and its
    anchor / second channel / acid branch are read over unflagged rows only."""
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

    # satellites (C11+c): each child hangs off the M0 with peak_id ==
    # parent_peak_id in the same file and counts only where it sits at its
    # label's exact spacing from the COMMITTED parent line (the window
    # max(1 ppm, 4 sigma(h)) self-fitted on the source's '13C' children, pcal,
    # N1; 'M+n' exempt); a kept line is in band when its height ratio is 0.5-2x
    # its count-aware expectation relative to that line. A dropped child is
    # dropped for every fact. The isotopologues list answers the same question
    # (`_judge_children`, chem/isotopes.judge_source).
    satellite = HALOGEN_SATELLITE.get(halogen) if halogen else None
    line_facts, list_ok = _judge_children(frame, role, m0, satellite=satellite, resolution=resolution,
                                          per_file=per_file)

    # per-neutral facts over the whole source (regular rows: see above)
    channels = regular.assign(__ch=regular["__adduct"].map(lambda a: (fold or {}).get(a, a))) \
        .groupby("__neutral")["__ch"].nunique()
    adduct_sets = regular.groupby("__neutral")["__adduct"].agg(lambda s: set(s))

    methods = m0["method"].astype(str)
    m0["__known"] = methods.where(methods.str.startswith("known:")).str[6:]
    m0["__iso_list"] = list_ok
    m0["__tied"] = m0["tied"].map(truthy)
    m0["__below"] = m0["below_assignability"].map(truthy)
    m0["__lead"] = m0["tentative_lead"].map(truthy)
    m0["__flagged"] = m0["__below"] | m0["__lead"]
    m0["__lead_by"] = m0["lead_by"].map(lead_setters)
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
    # rule H: the adducts each neutral commits on an UNFLAGGED regular row -- the
    # only channels a lifted pair's second channel / acid branch may come from
    lift = {(str(n), str(a)): dict(v or {}) for (n, a), v in (lift or {}).items()}
    clean = m0[~m0["__ion_only"] & ~m0["__alien"] & ~m0["__flagged"]] if lift else m0.iloc[:0]
    clean_adducts = clean.groupby("__neutral")["__adduct"].agg(lambda s: set(s)) if len(clean) else {}

    rows = []
    for (neutral, adduct), g in m0.groupby(["__neutral", "__adduct"], sort=True):
        key = (neutral, adduct)
        lines = line_facts.get(key) or _NO_LINES
        carbon_ev = lines["carbon"]
        ion = str(g["ion_formula"].iloc[0])
        own = lines["lined"] & neutral_elements(neutral, ion)
        known = g["__known"].dropna()
        seen = {v for v in g["__res"] if v}
        ion_only = bool(g["__ion_only"].any())
        outside = ion_only or bool(np.any(g["__alien"]))
        aset = set() if outside else adduct_sets.get(neutral, set())
        chan2 = (not outside) and int(channels.get(neutral, 0)) >= 2
        anchor = (not ion_only) and bool(g["__anchor"].any())
        # rule H (C11+b): a lock lifts the pair's lead where every lead row's
        # setter is one the lock answers and no row is below assignability; the
        # axes the flagged rows alone gave (anchor, a second channel, the acid
        # branch) go with the flag -- the lifted pair's own rows still give its
        # siblings what they gave before
        spec = lift.get(key)
        lifted = (spec is not None and not outside and not bool(g["__below"].any()) and bool(g["__lead"].any())
                  and all(lead_liftable(s, spec.get("budget_ok", False))
                          for s in g.loc[g["__lead"], "__lead_by"]))
        if lifted:
            chans = {adduct} | set(clean_adducts.get(neutral, set()))
            chan2 = len({(fold or {}).get(x, x) for x in chans}) >= 2
            aset = chans
            anchor = bool((g["__anchor"] & ~g["__flagged"]).any())
        rows.append(dict(
            neutral_formula=neutral, adduct=adduct,
            ion=ion,
            mz=float(g["__mz"].median()) if g["__mz"].notna().any() else np.nan,
            height=float(g["__height"].median()) if g["__height"].notna().any() else np.nan,
            ppm=float(g["__ppm"].median()) if g["__ppm"].notna().any() else np.nan,
            occurrence=float(g["__occ"].median()) if g["__occ"].notna().any() else np.nan,
            tier="Assigned" if (g["tier"].astype(str) == "Assigned").any() else "Candidate",
            known_fam=str(known.iloc[0]) if len(known) else "",
            # the lifted pair's isotope axis is the lock
            iso=lifted or lines["iso"] or bool(g["__iso_list"].any()),
            multiline=len(own) >= 2,
            multiline_elements="|".join(sorted(own)),
            carbon_ev=carbon_ev,
            chan2=chan2,
            anchor=anchor,
            branch=bool(aset & BARE_ADDUCTS) and bool(aset & CLUSTER_ADDUCTS),
            # the sole satellite is the reagent halogen's -- every kept line
            # naming a heavy atom names only its heavy isotope, none adds 13C,
            # none is a line only the ion's full halogen count makes (D4) --
            # and the reagent put that halogen on the ion: the ION carries more
            # of it than the neutral (C11+a; the Br-free hold released by
            # C11+c); a lifted pair's lock is a line of its own
            reagent_only_iso=(not ion_only) and not lifted and lines["reagent_only"]
            and carries_reagent(neutral, adduct, ion, halogen),
            ion_only=ion_only,
            iso_labels="|".join(sorted(lines["labels"])),
            tied=bool(g["__tied"].all()),
            below=bool(g["__below"].any()),
            # C19(c): any row a tentative lead (unsupported, not contradicted);
            # rule H (C11+b): False where a halogen lock lifts it
            lead=bool(g["__lead"].any()) and not lifted,
            lead_lift=lifted,
            lock_note=str(spec.get("note", "") or "") if lifted else "",
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
    if r.below or getattr(r, "lead", False):
        # C19(c) split the old flag in two; a lead is still hard and still reads
        # "below assignability", so no level and no reason moved with the split.
        # On the pooled batch a halogen lock (rule H, C11+b) lifts a lead: `lead`
        # reads False there and the pair levels on its other inputs.
        hard.append("below assignability")
    if r.lowconf:
        hard.append("engine confidence Low/Suspect")
    if getattr(r, "label_veto", False):
        # rule K (C18): the 14N twin of a labelled cluster is absent where the
        # reagent's own impurity puts it, or a 14N line has no 15N partner or
        # runs above its cluster share -- the cluster reading is refuted
        note = getattr(r, "label_note", "") or ""
        hard.append("the reagent's two isotopologues refute the cluster reading" + (f" ({note})" if note else ""))
    if getattr(r, "iso_veto", False):
        # C11+ (batch/iso_checks.py): the batch's time series refutes the
        # formula's own isotope claim -- its 13C carbon count, a required heavy
        # line absent, or a heavy line too high for it
        note = getattr(r, "iso_note", "") or ""
        hard.append("an isotope check refutes the formula" + (f" ({note})" if note else ""))
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
    held = " + ".join(a for a in _AXES if getattr(r, a))
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
    parts = [a for a in _AXES if getattr(r, a)]
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
    if getattr(r, "iso_veto", False):
        parts.append("iso_veto")
    if getattr(r, "lead_lift", False):
        parts.append("lead_lift")
    if r.known_fam:
        parts.append(f"known:{r.known_fam}")
    if with_files:
        parts.append(f"files:{int(r.n_files)}")
    return "|".join(parts)


def _level_pairs(frames: dict, *, cross=None, isomer_space=None, with_files: bool = False,
                 upair=None, label=None, iso=None, resolution=None, per_file: bool = False,
                 halogen=DETECT_HALOGEN) -> pd.DataFrame:
    """Level every (neutral, adduct) pair of the frames pooled as ONE source.
    `upair`: the neutrals whose declared neutral pair holds (rule U, measured by
    batch/neutral_pairs.py on the batch time series; pooled only). `label`: the
    labelled-nitrate twin facts of rule K (batch/label_twins.facts; pooled only):
    {'untie': {(n, a)} whose arbiter tie the 15N sibling breaks, 'veto': {(n, a):
    note} the reagent's isotopologues refute, 'alien': {(n, a)} 14N lines kept out
    of their neutral's pools}; given, the 14N and 15N nitrate clusters of one
    neutral also count as one channel. `iso`: the isotope checks (C11+,
    batch/iso_checks.facts; pooled only): {'veto': {(n, a): note}, 'lock': {(n,
    a): fact}} -- a vetoed pair is hard 5b and, like rule K's alien lines, leaves
    its neutral's chan2 / branch pools in both directions; a locked pair (rule
    H, C11+b) whose lead the lock answers is lifted (`_measure`), unless a check
    or rule K refutes the reading. `resolution`: the source's width model (the
    isotope children's committed-line tolerance and 'M+n' window, C11+c);
    `per_file`: the frames are one file levelled inside its run. `halogen`: the
    reagent halogen (`channel_halogen` of the run's declared channels, C43);
    the default names it from the committed cluster adducts instead."""
    parts = []
    for src, frame in frames.items():
        f = frame.copy()
        f["__file"] = str(src)
        parts.append(f)
    if not parts:
        return pd.DataFrame(columns=["neutral_formula", "adduct", *_SERIES_COLUMNS])
    frame = pd.concat(parts, ignore_index=True, sort=False)
    role = _col(frame, "role").astype(str)
    if halogen == DETECT_HALOGEN:
        halogen = detect_reagent_halogen(frame[role == "M0"])
    label = label or None
    iso_veto = {(str(n), str(a)): str(v or "") for (n, a), v in (((iso or {}).get("veto")) or {}).items()}
    alien = set((label or {}).get("alien") or set()) | set(iso_veto)
    # rule H (C11+b): the halogen locks lift a lead -- never on a reading a check
    # or rule K refutes (the vetoes outrank the lift; an alien pair is outside)
    label_veto = {(str(n), str(a)) for n, a in ((label or {}).get("veto") or {})}
    lift = {(str(n), str(a)): dict(v or {}) for (n, a), v in (((iso or {}).get("lock")) or {}).items()
            if (str(n), str(a)) not in iso_veto and (str(n), str(a)) not in label_veto}
    facts = _measure(frame, halogen=halogen, alien=alien or None, fold=LABEL_FOLD if label else None,
                     lift=lift or None, resolution=resolution, per_file=per_file)
    if facts.empty:
        return pd.DataFrame(columns=["neutral_formula", "adduct", *_SERIES_COLUMNS])
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
    # C11+: the isotope checks refute a formula -- a fact about one reading
    # (a pair), never an axis, never in `cross`; unlike rule K it reaches an
    # ion-only pair too (a refuted composition is refuted on any channel)
    facts["iso_veto"] = pd.Series([k in iso_veto for k in keys], index=facts.index, dtype=bool)
    facts["iso_note"] = [iso_veto.get(k, "") if v else "" for k, v in zip(keys, facts["iso_veto"])]
    facts["n_axes"] = facts[list(_AXES)].sum(axis=1).astype(int)
    facts["cross"] = facts["corroborated"] | facts["multiline"] | facts["known_fam"].ne("")
    facts["neutral_backed"] = (facts["corroborated"] | facts["chan2"] | facts["anchor"]
                               | facts["known_fam"].ne("") | facts["carbon_ev"])
    facts["n_plausible_structures"] = pd.array(
        [structures.get(f, pd.NA) for f in facts["neutral_formula"]], dtype="Int64")
    decided = [_decide(r) for r in facts.itertuples(index=False)]
    facts["evidence_level"] = [d[0] for d in decided]
    facts["level_reason"] = [d[1] for d in decided]
    facts["evidence_axes"] = [_axes_string(r, with_files=with_files) for r in facts.itertuples(index=False)]
    facts["claim"] = facts["evidence_level"].map(_series_claim)
    return facts


# ---------------------------------------------------------------------------
# the merge vote's class: the private decision per file
# ---------------------------------------------------------------------------
def _series_levels(ledger: pd.DataFrame, *, isomer_space=None, cross=None, resolution=None,
                   halogen=DETECT_HALOGEN) -> pd.DataFrame:
    """Pure: one row per M0 row of `ledger` (index = the ledger's index) with
    `peak_id` and the private decision's five columns (`_SERIES_COLUMNS`),
    the file levelled alone (``per_file=True``). `cross` = the --corroborate
    cross set (`vote_cross_neutrals`); `resolution` = the run's width model;
    `halogen` = the reagent halogen (`channel_halogen`, C43; the default counts
    the committed clusters)."""
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
    pairs = _level_pairs({"": ledger}, cross=cross, isomer_space=isomer_space, with_files=False,
                         resolution=resolution, per_file=True, halogen=halogen)
    if pairs.empty:
        return empty
    key = pd.DataFrame({
        "peak_id": m0["peak_id"].values,
        "neutral_formula": _col(m0, "neutral_formula").fillna("").astype(str).values,
        "adduct": _col(m0, "adduct").fillna("").astype(str).values,
    }, index=m0.index)
    out = key.merge(pairs[["neutral_formula", "adduct", *_SERIES_COLUMNS]],
                    on=["neutral_formula", "adduct"], how="left")
    out.index = m0.index
    out["n_plausible_structures"] = out["n_plausible_structures"].astype("Int64")
    out["claim"] = out["evidence_level"].map(_series_claim)
    return out[["peak_id", *_SERIES_COLUMNS]]


def _vote_class_of(level, axes) -> int:
    """The vote class of one private (level, axes) reading: 2 when the level
    says the neutral is established (2b / 3a / 3b / 4a) or the axes hold the
    --corroborate source's agreement (`corroborated`: the source pins the
    neutral on its own, this reading at any level); 1 when the formula or the
    ion is confirmed (4b / 4c / 4d); 0 otherwise (5a, 5b, no level)."""
    lv = "" if level is None or (not isinstance(level, str) and pd.isna(level)) else str(level)
    ax = "" if axes is None or (not isinstance(axes, str) and pd.isna(axes)) else str(axes)
    if lv in _VOTE_GOOD or "corroborated" in ax.split("|"):
        return 2
    if lv in _VOTE_MID:
        return 1
    return 0


def vote_classes(trim_frame: pd.DataFrame, *, cross=None, resolution=None, isomer_space=None,
                 halogen=DETECT_HALOGEN) -> pd.Series:
    """The merge vote's evidence class (0 / 1 / 2, see `VOTE_CLASS_TEXT`) of
    every M0 row of one file's ledger (`trim(ledger)` or the full ledger:
    the same answer), indexed like those rows. `cross` = the --corroborate
    cross set (`vote_cross_neutrals`), `resolution` = the batch's width model,
    `halogen` = the reagent halogen the file's own run read (`channel_halogen` of
    its declared channels, C43; the default counts the committed clusters).
    Private to the vote: never written to a ledger, never a level."""
    lv = _series_levels(trim_frame, isomer_space=isomer_space, cross={str(x) for x in (cross or set())},
                        resolution=resolution, halogen=halogen)
    if lv.empty:
        return pd.Series([], dtype="int64")
    return pd.Series([_vote_class_of(a, b) for a, b in zip(lv["evidence_level"], lv["evidence_axes"])],
                     index=lv.index, dtype="int64")


def _n_pairs(ledger: pd.DataFrame) -> int:
    role = _col(ledger, "role").astype(str)
    m0 = ledger[role == "M0"]
    if m0.empty:
        return 0
    return int(pd.DataFrame({"n": _col(m0, "neutral_formula").fillna("").astype(str),
                             "a": _col(m0, "adduct").fillna("").astype(str)}).drop_duplicates().shape[0])


def _series_pooled(per_file: dict, *, cross=None, isomer_space=None, upair=None, label=None,
                   iso=None, resolution=None, halogen=DETECT_HALOGEN) -> pd.DataFrame:
    """The private decision over a batch's per-file ledgers ({label: frame})
    pooled as ONE source (the --corroborate cross set's own levels)."""
    return _level_pairs(dict(per_file), cross=cross, isomer_space=isomer_space, with_files=True, upair=upair,
                        label=label, iso=iso, resolution=resolution, halogen=halogen)


def trim(ledger: pd.DataFrame) -> pd.DataFrame:
    """The M0 + isotope rows and the predicate columns of a ledger -- what the
    batch checks (iso_checks, label_twins, neutral_pairs) and the vote class
    read per file (the rest of the ledger is on disk)."""
    role = _col(ledger, "role").astype(str)
    keep = ledger[role.isin(["M0", "iso_child"])]
    return keep[[c for c in PREDICATE_COLUMNS if c in keep.columns]].copy()


# ---------------------------------------------------------------------------
# --corroborate sources: the vote's cross set
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


#: The private level a --corroborate source must hold a neutral at, by its OWN
#: evidence, for its sighting to count for the vote: the formula confirmed by
#: an axis of the source's own -- 4b -- or better. Below that the source's grid
#: only ENUMERATED the same formula at a peak: two grids agreeing, not a second
#: sighting of the neutral. Measured on the same-air pair before the rule: of
#: 1521 neutrals the labelled-nitrate Orbitrap offered the TOF, 379 pass; of
#: 3728 the TOF offered the Orbitrap, 435, and of the TOF's 49 vote winners
#: lifted by the agreement alone, 28 rested on an Orbitrap mass-only reading.
_VOTE_CROSS_MAX_LEVEL = "4b"
_AXES_OWN = frozenset(a for a in _AXES if a != "corroborated")


def _rank(level) -> int:
    return _SERIES_ORDER.index(level) if level in _SERIES_ORDER else len(_SERIES_ORDER)


def _stored_own_good(level, axes, max_level: str) -> bool:
    """A merged ledger's stored private level read WITHOUT its own `corroborated`
    axis (a merged row carries no predicate column to re-level it from): good
    when the level is `max_level` or better and it still holds an axis of its
    own once `corroborated` is taken away."""
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


def _source_neutrals(per_file: dict, *, isomer_space=None, max_level: str = _VOTE_CROSS_MAX_LEVEL,
                     resolution=None) -> set[str]:
    """The neutral formulas a source ({label: ledger}) holds at `max_level` or
    better by its OWN evidence (the private decision, the ledgers pooled as ONE
    source with NO cross set, so a source run with --corroborate cannot hand a
    run back the agreement it got from it). Ion-only pairs never count. A
    ledger without a `role` column is a merged ledger: its stored private level
    (`evidence_level` + `evidence_axes`, written before peaky 0.10.0) is read
    instead; a merged ledger of the evidence scale carries no such class -- name
    the run dir. `resolution`: the source's own width model."""
    frames = {k: f for k, f in per_file.items() if "role" in f.columns}
    merged = {k: f for k, f in per_file.items() if "role" not in f.columns}
    out: set[str] = set()
    if frames:
        pairs = _series_pooled({k: trim(f) for k, f in frames.items()}, cross=None, isomer_space=isomer_space,
                               resolution=resolution)
        if len(pairs):
            ok = (pairs["evidence_level"].map(_rank) <= _rank(max_level)) & ~pairs["ion_only"].astype(bool)
            out |= set(pairs.loc[ok, "neutral_formula"].astype(str))
    for label, f in merged.items():
        if "evidence_level" not in f.columns:
            raise ValueError(
                f"--corroborate source {label!r} carries neither the per-file predicate columns nor an "
                "evidence_level column (a merged ledger from before the levels?) -- name the run dir, "
                "whose per_file/ ledgers the source's own levels are computed from")
        if "evidence_axes" not in f.columns:
            raise ValueError(
                f"--corroborate source {label!r} is a merged ledger of the evidence scale (no evidence_axes): "
                "its levels are not the merge vote's class -- name the run dir, whose per_file/ ledgers "
                "the source's own class is computed from")
        f = f[~is_ion_only(f)]
        ok = np.array([_stored_own_good(lv, ax, max_level)
                       for lv, ax in zip(f["evidence_level"], _col(f, "evidence_axes", ""))], dtype=bool)
        out |= set(_col(f, "neutral_formula").loc[ok].dropna().astype(str))
    out.discard("")
    return out


def vote_cross_neutrals(sources) -> set[str]:
    """The merge vote's cross set from the --corroborate sources: for every
    source (a run dir, an out-dir holding one run, a ledger CSV, or a frame),
    the neutral formulas it holds by its own evidence at the private level 4b
    or better (`_source_neutrals`). A run dir is read through its per-file
    ledgers with its own width model. It feeds the vote's class only; it never
    moves an evidence level."""
    out: set[str] = set()
    for src in sources or []:
        resolution = None
        if isinstance(src, pd.DataFrame):
            per_file = {"": src}
        else:
            label, files = resolve_source(src)
            per_file = {(os.path.basename(f) if len(files) > 1 else label): pd.read_csv(f, low_memory=False)
                        for f in files}
            resolution = source_resolution(src)
        out |= _source_neutrals(per_file, resolution=resolution)
    return out


def source_resolution(path) -> dict | None:
    """A run dir's width model as its batch_summary.json records it (the
    `resolution` dict), for a run dir or an out-dir holding one run; None for a
    ledger CSV or a run that recorded none (a class-less source)."""
    import json
    path = os.path.expanduser(str(path).rstrip("/"))
    if not os.path.isdir(path):
        return None
    run = path
    if not (os.path.isdir(os.path.join(path, "per_file")) or os.path.isfile(os.path.join(path, "merged_ledger.csv"))):
        inner = [d for d in sorted(glob.glob(os.path.join(path, "*"))) if os.path.isdir(d)
                 and (os.path.isdir(os.path.join(d, "per_file")) or os.path.isfile(os.path.join(d, "merged_ledger.csv")))]
        if len(inner) != 1:
            return None
        run = inner[0]
    summary = os.path.join(run, "batch_summary.json")
    if not os.path.isfile(summary):
        return None
    try:
        res = json.load(open(summary)).get("resolution")
    except (OSError, ValueError):
        return None
    return res if isinstance(res, dict) and res.get("coef") is not None else None


# ---------------------------------------------------------------------------
# the evidence scale of peaky 0.10.0: entry points (machinery in peaky.assignment.levels)
# ---------------------------------------------------------------------------
def __getattr__(name):
    # `RunInputs` (levels.source) without importing the machinery at module load
    if name == "RunInputs":
        from peaky.assignment.levels.source import RunInputs
        return RunInputs
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def source_from_frames(per_file: dict, *, run_inputs, mode: str = "run", name: str = "", label: str = "",
                       main=None, arm=None, wrong_adducts=(), control_ledger=None):
    """A Source of the evidence scale from in-memory FULL ledgers
    ({sample_id: ledger}) and `RunInputs` (levels/source.py)."""
    from peaky.assignment.levels import source as SRC
    return SRC.source_from_frames(per_file, run_inputs=run_inputs, mode=mode, name=name, label=label, main=main,
                                  arm=arm, wrong_adducts=wrong_adducts, control_ledger=control_ledger)


def source_from_run_dir(path, *, main=None, mode: str | None = None, name: str | None = None,
                        label: str | None = None):
    """A Source from a batch run directory, or from ONE ledger CSV (levelled
    alone, or in a ``main`` run's context like a decoy arm; ``mode``
    'adapted' or 'strict')."""
    from peaky.assignment.levels import source as SRC
    return SRC.source_from_run_dir(path, main=main, mode=mode, name=name, label=label)


def level_source(src, *, partners=None) -> pd.DataFrame:
    """Level a Source: one row per (neutral_formula, adduct) with the scale's
    `COLUMNS` and the step facts. ``partners``: other-source partners
    (`partners_from`)."""
    from peaky.assignment.levels import source as SRC
    return SRC.level_source(src, partners=partners)


def level_batch(per_file: dict, *, run_inputs, partners=None) -> pd.DataFrame:
    """A batch's FULL per-file ledgers ({sample_id: ledger}) levelled as ONE
    pooled source (minima 3) with the run's other inputs (`RunInputs`)."""
    return level_source(source_from_frames(per_file, run_inputs=run_inputs, mode="run"), partners=partners)


def partners_from(levels: pd.DataFrame, label: str, **kw) -> dict:
    """The other-source partners a levelled source gives (its route / ladder /
    listed pairs, not ion-only; side-channel route classes dropped while
    locked): {neutral: {route class: ['<label> <neutral> <adduct> <how>']}}."""
    from peaky.assignment.levels import routes as RT
    return RT.partners_from(levels, label, **kw)


def _resolution_dict(resolution) -> dict | None:
    """A width model as the dict a batch summary records (None without one)."""
    if resolution is None:
        return None
    if isinstance(resolution, dict):
        return resolution if resolution.get("coef") is not None else None
    from peaky.chem.resolution import Resolution
    return Resolution.coerce(resolution).as_dict()


def file_run_inputs(*, sample_id: str, reagent: str | None, context: str | None, resolution=None,
                    reflists_active=None, height_gate_cps=None, noise_edge_cps=None, degeneracy_cal="absent",
                    label: str = "", activation=None, reagent_halogen=DETECT_HALOGEN):
    """The `RunInputs` of ONE file levelled alone (the per-file stage, D2 b):
    no time series, no merged ledger, no batch checks; the file's gate and,
    when the degeneracy stage persisted one, its calibration (``degeneracy_cal``
    = (mu, sigma) | None = uncalibrated; the default 'absent' = not persisted:
    the window is then refitted from the file's own degeneracy counts).
    ``reagent_halogen``: the halogen of the run's declared channels
    (`channel_halogen`, C43) the pair facts read; the default counts the
    committed clusters."""
    from peaky.assignment.levels.source import RunInputs
    st = dict(sample_id=str(sample_id), height_gate_cps=height_gate_cps, noise_edge_cps=noise_edge_cps)
    if degeneracy_cal != "absent":
        st["degeneracy_cal"] = (None if degeneracy_cal is None
                                else {"mu": float(degeneracy_cal[0]), "sigma": float(degeneracy_cal[1])})
    summary = dict(reagent=reagent, context=context or "ambient-air", label=label or "",
                   reflists_active=[list(x) for x in (reflists_active or [])],
                   resolution=_resolution_dict(resolution), per_file=[st], reagent_halogen=reagent_halogen)
    if activation is not None:
        summary["reflists_context"] = activation
    return RunInputs(summary=summary, activation=activation)


#: the evidence text of a committed M0 row the per-file stage could not level
NO_REAGENT_TEXT = ("no level · the run's reagent profile is unknown (its adducts match no registered profile): "
                   "the enumeration space cannot be built")


#: ... and of one whose file has no calibrated window (an uncalibrated file levelled alone)
NO_WINDOW_TEXT = ("no level · the file is uncalibrated (no calibration core for the degeneracy audit) and, "
                  "levelled alone, has no run sigma to borrow: no calibrated window to enumerate competitors in")


def _blank_columns(frame: pd.DataFrame) -> None:
    n = len(frame)
    for c in COLUMNS:
        frame[c] = pd.Series([pd.NA] * n, index=frame.index, dtype=object)


def _stamp_pairs(frame: pd.DataFrame, rows: pd.Series, pairs: pd.DataFrame | None) -> pd.Series:
    """Write `pairs`' COLUMNS onto the `rows` (a boolean mask) of `frame` by
    (neutral_formula, adduct), in place; returns the mask of the rows that
    found a pair."""
    hit_mask = pd.Series(False, index=frame.index)
    if pairs is None or pairs.empty or not rows.any():
        return hit_mask
    right = pairs[["neutral_formula", "adduct", *COLUMNS]].copy()
    right["__n"] = right["neutral_formula"].fillna("").astype(str)
    right["__a"] = right["adduct"].fillna("").astype(str)
    right = right.drop_duplicates(["__n", "__a"]).set_index(["__n", "__a"])
    sub = frame.loc[rows]
    idx = pd.MultiIndex.from_arrays([_col(sub, "neutral_formula").fillna("").astype(str).values,
                                     _col(sub, "adduct").fillna("").astype(str).values])
    hit = right.reindex(idx)
    found = hit["evidence_level"].notna().to_numpy()
    for c in COLUMNS:
        vals = hit[c].astype(object).to_numpy()
        vals = np.where(found, vals, pd.NA)
        frame.loc[sub.index, c] = pd.Series(vals, index=sub.index, dtype=object)
    hit_mask.loc[sub.index] = found
    return hit_mask


def apply_levels(ledger: pd.DataFrame, *, cfg=None, resolution=None, run_inputs=None,
                 sample_id: str | None = None) -> dict:
    """The per-file `evidence` stage (D2 b): the file levelled ALONE in
    "adapted" mode (every file-count minimum 1, no time series, no merged
    ledger, no partners; the amine gate on its no-time-series path with the
    file's own ledger as the ion index and protected set; the window = the
    degeneracy stage's calibration). Writes `COLUMNS` on every committed M0
    row, in place (empty on every other row); returns {levels, claims,
    n_levelled, n_pairs, instrument}. The instrument class comes from
    ``resolution`` (else the run inputs' width model): a TOF-class or
    class-less file reads NA before any fact work. ``cfg`` is accepted for
    stage-call symmetry and not read."""
    sid = str(sample_id or "file")
    if run_inputs is None:
        run_inputs = file_run_inputs(sample_id=sid, reagent=None, context=None, resolution=resolution)
    summary = run_inputs.summary
    if resolution is not None:
        summary["resolution"] = _resolution_dict(resolution)
    pf = summary.get("per_file") or []
    if pf:
        sid = str(pf[0].get("sample_id") or sid)
    _blank_columns(ledger)
    role = _col(ledger, "role").astype(str)
    m0 = role == "M0"
    from peaky.assignment.levels import source as SRC
    src = SRC.source_from_frames({sid: ledger}, run_inputs=run_inputs, mode="adapted", name=sid, label=sid)
    klass, r200 = src.instrument()
    pairs, why = None, ""
    if klass == "orbitrap" and not summary.get("reagent"):
        why = NO_REAGENT_TEXT
    elif klass == "orbitrap":
        # the step-1 window: the degeneracy stage's calibration (else the D10
        # refit); a file with no calibration has no window of its own and, alone,
        # no run sigma to borrow -- it is not levelled, and says so
        from peaky.assignment.levels import context as CX
        src.win = CX.run_windows(summary, src.per_file)
        if not np.isfinite(CX.window_table(src.win)[1][1]):
            why = NO_WINDOW_TEXT
    if why:
        ledger.loc[m0, "evidence"] = why
    else:
        pairs = level_source(src)
        _stamp_pairs(ledger, m0, pairs)
    ledger.loc[m0, "claim"] = [claim_class(v) for v in ledger.loc[m0, "evidence_level"]]
    lv = ledger.loc[m0, "evidence_level"]
    return {"levels": summarize(lv), "claims": summarize_claims(ledger.loc[m0, "claim"]),
            "n_levelled": int(lv.notna().sum()), "n_pairs": _n_pairs(ledger),
            "instrument": {"class": klass, "r200": None if not np.isfinite(r200) else round(float(r200), 1)}}


#: the evidence text of a merged row whose reading no pooled pair holds
NO_POOLED_PAIR_TEXT = "no pooled pair: a batch-level re-read"


def stamp_merged(merged: pd.DataFrame, pairs: pd.DataFrame | None) -> pd.DataFrame:
    """Join the pooled levels (`level_batch`) onto a merged ledger by
    (neutral_formula, adduct) -- each merged row is one ion, so the join is
    one-to-one. Every merged row is a committed reading and carries a claim: a
    row whose reading no pooled pair holds (a batch-level re-read) gets no
    level, claim tentative and the evidence `NO_POOLED_PAIR_TEXT`."""
    out = merged.copy()
    _blank_columns(out)
    if not len(out):
        return out
    hit = _stamp_pairs(out, pd.Series(True, index=out.index), pairs)
    out.loc[~hit, "evidence"] = NO_POOLED_PAIR_TEXT
    out["claim"] = pd.Series([claim_class(v) for v in out["evidence_level"]], index=out.index, dtype=object)
    return out
