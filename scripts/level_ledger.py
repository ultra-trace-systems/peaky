"""Rate every assigned neutral in a ledger on the CIMS-adapted Schymanski scale.

Schymanski's confidence levels were written for LC-HRMS with a fragment spectrum
and a compound library. A chemical-ionization run has neither: there is one
adduct channel, no MS2, and the reagent takes part in the ion. This script is
the executable reference for the adaptation — it says what a level MEANS in
terms of columns a peaky ledger already carries, and it is the yardstick the
in-core `evidence_level` column (Phase B) must reproduce row for row.

    python scripts/level_ledger.py <source>... [--corroborate <source>]
                                   [--upair [<neutral_pairs.csv>]]
                                   [--label-twins [<label_twins.csv>]]
                                   [--iso-checks [<iso_checks.csv>]]
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
        below assignability or a tentative lead (with --iso-checks, one a
        halogen lock of the batch answers is lifted: rule H, C11+b), or a score
        the engine itself calls Low/Suspect, or (rule K, --label-twins) the
        labelled reagent's 14N twin refutes the cluster reading, or (C11+,
        --iso-checks) an isotope check of the batch's time series refutes the
        formula; also a mass-degenerate row with no corroborating axis at all
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
import json
import math
import os
import re
import sys
from functools import lru_cache
from typing import NamedTuple

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
    """A ledger cell holding a JSON or repr'd list, or nothing at all.

    The ledger writes the satellite list with json.dumps, so a line without a
    per-line score carries `null`, which `ast.literal_eval` cannot read: JSON
    is tried first, as the engine's `evidence.as_list` does (C11+c; before it
    such a list read EMPTY here). A cell neither parser reads is empty."""
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        parsed = json.loads(value)
    except Exception:
        try:
            parsed = ast.literal_eval(value)
        except Exception:
            return []
    return list(parsed) if isinstance(parsed, (list, tuple)) else []


def to_float(value) -> float:
    """A cell as a float; NaN when it is none."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


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


ADDUCT_TOKEN = re.compile(r"([+-])\^?([A-Za-z0-9]+)")


def ion_composition(neutral, adduct, ion) -> dict:
    """The ION's element counts: its stored ion formula when that carries a
    charge sign, else neutral + adduct ('[M+HBr+Br]-' adds H and two Br; a
    ledger row can hold the NEUTRAL in ion_formula)."""
    s = str(ion).strip() if isinstance(ion, str) else ""
    if s.endswith(("+", "-")):
        return composition(s)
    a = str(adduct).strip() if isinstance(adduct, str) else ""
    if not neutral or not a.startswith("[M"):
        return composition(s)
    counts = composition(neutral)
    inner = a.split("]")[0][2:].replace("(", "").replace(")", "")
    for sign, token in ADDUCT_TOKEN.findall(inner):
        for element, n in composition(token).items():
            counts[element] = counts.get(element, 0) + (n if sign == "+" else -n)
    return {k: v for k, v in counts.items() if v} or composition(s)


ADDUCT_TOKEN_LABELLED = re.compile(r"([+-])(\^?[A-Za-z0-9]+)")


def ion_counts(neutral, adduct, ion) -> dict:
    """The ION's element counts as the engine's evidence.ion_composition reads
    them for the isotope lines, the labelled '^N' kept as its own key: the
    stored ion formula when it carries a charge sign, else neutral + adduct (a
    labelled reagent atom kept too: '[M+^NO3]-' adds 15N, as the engine's
    tiers._ion_counts(labelled=True) reads it), else the stored string.
    `ion_composition` above folds '^N' into N (the reagent halogen count it
    serves does not care)."""
    s = str(ion).strip() if isinstance(ion, str) else ""
    if s.endswith(("+", "-")):
        return composition(s, labelled=True)
    a = str(adduct).strip() if isinstance(adduct, str) else ""
    n = "" if neutral is None else str(neutral)
    counts: dict = {}
    if n and a.startswith("[M"):
        counts = composition(n, labelled=True)
        inner = a.split("]")[0][2:].replace("(", "").replace(")", "")
        for sign, token in ADDUCT_TOKEN_LABELLED.findall(inner):
            for element, k in composition(token, labelled=True).items():
                counts[element] = counts.get(element, 0) + (k if sign == "+" else -k)
    return {k: v for k, v in counts.items() if v} or composition(s, labelled=True)


def carries_reagent(neutral, adduct, ion, halogen) -> bool:
    """The ION carries more of the reagent halogen than the neutral: only then
    can a line of it be the reagent's (the reagent-only flag reads this alone
    since C11+c released the Br-free hold)."""
    if not halogen:
        return False
    return ion_composition(neutral, adduct, ion).get(halogen, 0) > composition(neutral).get(halogen, 0)


def expected_ratio(tag: str, ion_formula, committed: dict | None = None) -> float:
    """The natural height ratio of an isotope line to its parent (C11+c: the
    count-aware expectation relative to the COMMITTED parent line, `committed`
    its heavy configuration, {} / None = the mono line); 0.0 when there is none
    -- the engine's evidence.expected_ratio. The levels read every child through
    `resolve_child` (below); this is its parent-relative reading."""
    counts = {e: v for e, v in composition(ion_formula, labelled=True).items() if v}
    shift = resolve_child(tag, counts, {}, 0.0)["shift"]
    r = resolve_child(tag, counts, dict(committed or {}), shift if np.isfinite(shift) else 0.0)
    return float(r["expected"])


def neutral_elements(neutral, ion) -> set:
    """Elements whose isotope line speaks for the neutral: the neutral supplies
    more than half of the ion's atoms of the element (else the line measures
    the reagent -- 15N on a urea adduct of an N-free neutral, 81Br on a bromide
    adduct)."""
    own, whole = composition(neutral, labelled=True), composition(ion, labelled=True)
    return {e for e, n in own.items() if n > 0 and 2 * n > whole.get(e, 0)}


# ===========================================================================
# ISOTOPE CHILDREN JUDGED AGAINST THE COMMITTED LINE (C11+c) -- the standalone
# twin of peaky/chem/isotopes.py's section of the same name (this script imports
# no peaky code). The committed line is read off the parent's m/z, each child
# label both ways (parent-relative / mono-counted), a line counts within
# max(1 ppm, 4 sigma(h)) of its exact spacing (sigma(h) self-fitted on the
# source's '13C' children; pcal and N1 rescue it, 'M+n' is exempt) and in band
# under its count-aware expectation. tests/test_isotope_children.py pins every
# function here to the engine's, text and values.
# ===========================================================================

#: exact mass (NIST / AME2020) of the heavy isotope a child label can name and
#: the element it replaces; the light isotope's mass is chemistry.M's
HEAVY_ISOTOPES: dict[str, tuple[str, float]] = {
    "13C": ("C", 13.0033548378), "15N": ("N", 15.0001088984), "17O": ("O", 16.9991317565),
    "18O": ("O", 17.9991596129), "33S": ("S", 32.9714589098), "34S": ("S", 33.967867004),
    "37Cl": ("Cl", 36.965902602), "81Br": ("Br", 80.9162906), "29Si": ("Si", 28.9764946649),
    "30Si": ("Si", 29.973770136), "2H": ("H", 2.0141017781),
}
#: monoisotopic element masses an ion's mono m/z is summed from (chemistry.M
#: plus the two alkali adduct metals -- a copy: the script imports no peaky code)
ELEMENT_MASS: dict[str, float] = {
    "C": 12.0, "H": 1.0078250319, "O": 15.9949146221, "N": 14.0030740052, "S": 31.97207069, "P": 30.97376163,
    "Si": 27.976926535, "F": 18.9984031627, "Cl": 34.96885268, "Br": 78.9183371, "I": 126.9044719,
    "^N": 15.0001088984, "Na": 22.989769282, "K": 38.9637064864,
}
ELECTRON_MASS = 0.0005485799
#: exact heavy - light spacing (Da); '14N' is the labelled reagent's light line,
#: one 15N - 14N spacing BELOW a '^N' atom
ISOTOPE_SPACING: dict[str, float] = {iso: m - ELEMENT_MASS[el] for iso, (el, m) in HEAVY_ISOTOPES.items()}
ISOTOPE_SPACING["14N"] = -ISOTOPE_SPACING["15N"]
#: the composition key each heavy isotope counts atoms of ('14N' counts the
#: labelled '^N' atoms)
ISOTOPE_ELEMENT: dict[str, str] = {iso: el for iso, (el, _m) in HEAVY_ISOTOPES.items()}
ISOTOPE_ELEMENT["14N"] = "^N"
#: per-atom heavy / light ratio -- the evidence band's (evidence.C13_PER_CARBON,
#: ISOTOPE_ABUNDANCE, PER_ATOM_ABUNDANCE); 33S, 17O, 2H natural; '14N' the
#: labelled reagent's impurity at LABEL_PURITY_15N. Card B7 re-rounds 30Si.
ISOTOPE_RATIO: dict[str, float] = {
    "13C": 0.0107, "81Br": 0.9728, "37Cl": 0.3196, "34S": 0.0443, "29Si": 0.0508, "30Si": 0.0335,
    "15N": 0.00368 / 0.99632, "18O": 0.00205 / 0.99757, "33S": 0.0075 / 0.9499, "17O": 0.00038 / 0.99757,
    "2H": 0.000115, "14N": 0.02 / 0.98,
}
#: an ion carrying Br or Cl owns its M+2 region: an '18O' line there is not
#: measured (evidence.M2_OWNERS, C17) -- its expectation reads 0
M2_OWNERS = ("Br", "Cl")

#: the heavy isotopes the committed line may carry (the M+2 drivers; S and Si
#: to two atoms), and the least probability a candidate configuration needs
COMMITTED_ISOTOPES = ("81Br", "37Cl", "34S", "30Si", "29Si")
COMMITTED_MIN_P = 1e-4
#: how far (ppm of the parent m/z) the parent may sit from a configuration's
#: exact position and still be read as committed on it: 5 ppm on an
#: Orbitrap-class width model, 20 ppm on a TOF-class one and without one
COMMITTED_TOL_PPM = {"orbitrap": 5.0, "tof": 20.0}
COMMITTED_TOL_CLASSLESS_PPM = 20.0
#: a generic 'M+n' child: the ion's lines within this half-width (Da) of its
#: measured shift, or half the width model's FWHM there if wider
GENERIC_HALF_WIDTH_DA = 0.012

#: the position test: a child counts within max(POSITION_MIN_PPM, POSITION_K x
#: sigma(h)) of its exact position, sigma(h)^2 = a^2 + b^2 / h fitted on the
#: source's own '13C' children -- pre-clipped at max(SIGMA_CLIP_PPM,
#: SIGMA_CLIP_K x the global 1.4826 x MAD), SIGMA_BINS height-quantile bins of
#: >= SIGMA_MIN_PER_BIN children, per-bin 1.4826 x MAD, weighted least squares
#: of s^2 on 1/h (weights = bin counts), a >= SIGMA_A_FLOOR_PPM, b >= 0; fewer
#: than SIGMA_MIN_CHILDREN such children and the source is not tested at all
POSITION_MIN_PPM = 1.0
POSITION_K = 4.0
SIGMA_MIN_CHILDREN = 40
SIGMA_BINS = 8
SIGMA_MIN_PER_BIN = 5
SIGMA_A_FLOOR_PPM = 0.02
SIGMA_CLIP_PPM = 5.0
SIGMA_CLIP_K = 6.0
MAD_TO_SIGMA = 1.4826
#: N1, the neighbour's pull: a child outside the window still counts when a row
#: the levelling reads (an M0 or iso child of its file, not the child's own
#: parent) >= NEIGHBOUR_RATIO x its height sits on the side it is displaced
#: toward, within NEIGHBOUR_REACH_PPM of its exact position, and the child's
#: residual is <= NEIGHBOUR_FRACTION of the distance to it
NEIGHBOUR_RATIO = 3.0
NEIGHBOUR_REACH_PPM = 25.0
NEIGHBOUR_FRACTION = 0.5

_PART_KX = re.compile(r"^(\d+)x(\d+)([A-Z][a-z]?)$")                     # '2x81Br'
_PART_TWO = re.compile(r"^(\d+)([A-Z][a-z]?)(\d+)([A-Z][a-z]?)\(pair\)$")  # '81Br37Cl(pair)': one of each
_PART_ONE = re.compile(r"^(\d+)([A-Z][a-z]?)(\d*)(\(pair\))?$")          # '81Br', '81Br2', '13C2', '37Cl(pair)'
_GENERIC = re.compile(r"^M\+(\d+)$")                                      # 'M+5'


def split_label(label) -> list[str]:
    """The '+' parts of a child label: 'M+n' is one part ('13C+M+4' -> ['13C',
    'M+4']); a bare number after an isotope part is the synthetic tests'
    nominal-shift note ('13C+1', '81Br+2') and is no part."""
    out: list[str] = []
    for tok in (t.strip() for t in str(label).split("+")):
        if tok.isdigit() and out:
            if out[-1] == "M":
                out[-1] = "M+" + tok
            continue
        out.append(tok)
    return [t for t in out if t]


def parse_label_part(part: str) -> tuple[str, object]:
    """One '+' part -> ('set', {isotope: count}) | ('alt', [{..}, {..}]) (either
    line, '81Br/37Cl(pair)') | ('gen', n) ('M+n') | ('mono', {}) ('M0') |
    ('bad', None). 'kx' multiplies ('2x81Br'), a trailing count counts
    ('81Br2', '37Cl3', '13C2'), '(pair)' is one atom ('81Br(pair)'),
    '81Br37Cl(pair)' is one of each."""
    p = str(part).strip()
    if p == "M0":
        return "mono", {}
    m = _GENERIC.match(p)
    if m:
        return "gen", int(m.group(1))
    if "/" in p and p.endswith("(pair)"):
        alts = []
        for a in p[:-6].split("/"):
            kind, v = parse_label_part(a)
            if kind != "set":
                return "bad", None
            alts.append(v)
        return "alt", alts
    m = _PART_KX.match(p)
    if m:
        iso = m.group(2) + m.group(3)
        return ("set", {iso: int(m.group(1))}) if iso in ISOTOPE_SPACING else ("bad", None)
    m = _PART_TWO.match(p)
    if m:
        a, b = m.group(1) + m.group(2), m.group(3) + m.group(4)
        return ("set", {a: 1, b: 1}) if a in ISOTOPE_SPACING and b in ISOTOPE_SPACING else ("bad", None)
    m = _PART_ONE.match(p)
    if m:
        iso = m.group(1) + m.group(2)
        return ("set", {iso: int(m.group(3) or 1)}) if iso in ISOTOPE_SPACING else ("bad", None)
    return "bad", None


def heavy_key(h: dict) -> tuple:
    """A heavy configuration {isotope: count} as a sorted tuple (zeros dropped)."""
    return tuple(sorted((k, int(v)) for k, v in h.items() if v))


def heavy_shift(h) -> float:
    """The exact mass shift of a heavy configuration from the all-light line."""
    return sum(ISOTOPE_SPACING[i] * k for i, k in (h.items() if isinstance(h, dict) else h))


def _heavy_nominal(h) -> int:
    return sum(int(round(ISOTOPE_SPACING[i])) * k for i, k in (h.items() if isinstance(h, dict) else h))


def _heavy_add(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        out[k] = out.get(k, 0) + v
    return {k: v for k, v in out.items() if v}


def heavy_probability(h: dict, counts: dict) -> float:
    """P(configuration h) relative to the all-light line of an ion with element
    `counts`: per element the multinomial C(n; k1, k2, ..) x prod ratio^k over
    its n atoms. 0.0 when the ion cannot form it (an element it lacks, more
    heavy atoms than it has). '13C2' is C(nC, 2) x 0.0107^2."""
    p = 1.0
    per_el: dict[str, dict] = {}
    for iso, k in h.items():
        if k < 0:
            return 0.0
        per_el.setdefault(ISOTOPE_ELEMENT[iso], {})[iso] = k
    for el, ks in per_el.items():
        n = int(counts.get(el, 0))
        tot = sum(ks.values())
        if tot > n:
            return 0.0
        mult = math.factorial(n) / (math.factorial(n - tot) * math.prod(math.factorial(k) for k in ks.values()))
        p *= mult * math.prod(ISOTOPE_RATIO[iso] ** k for iso, k in ks.items())
    return p


def ion_sign(ion, adduct) -> str:
    """'+' / '-' of an ion: its stored ion formula's sign, else its adduct's;
    '' when neither carries one."""
    s = ion.strip().rstrip(".") if isinstance(ion, str) else ""
    if s.endswith(("+", "-")):
        return s[-1]
    a = adduct.strip().rstrip(".") if isinstance(adduct, str) else ""
    return a[-1] if a.endswith(("+", "-")) else ""


def mono_mz(counts: dict, sign: str) -> float:
    """The monoisotopic m/z of a singly charged ion (NaN for an unknown element
    or no sign)."""
    if sign not in ("+", "-"):
        return float("nan")
    mass = sum(ELEMENT_MASS.get(el, float("nan")) * n for el, n in counts.items())
    return mass + (ELECTRON_MASS if sign == "-" else -ELECTRON_MASS)


@lru_cache(maxsize=200000)
def _committed_candidates(ckey: tuple) -> tuple:
    counts = dict(ckey)
    opts: list[dict] = [{}]
    for iso in COMMITTED_ISOTOPES:
        el = ISOTOPE_ELEMENT[iso]
        n = int(counts.get(el, 0))
        kmax = min(n, 2) if iso in ("34S", "30Si", "29Si") else n
        new = []
        for o in opts:
            used = sum(v for k, v in o.items() if ISOTOPE_ELEMENT[k] == el)
            for k in range(0, min(kmax, n - used) + 1):
                new.append(_heavy_add(o, {iso: k}) if k else dict(o))
        opts = new
    out = []
    for o in opts:
        p = heavy_probability(o, counts)
        if p >= COMMITTED_MIN_P:
            out.append((heavy_key(o), heavy_shift(o), p))
    return tuple(out)


def committed_tolerance_ppm(klass: str | None) -> float:
    """COMMITTED_TOL_PPM of an instrument class ('orbitrap' / 'tof'); 20 ppm
    without one."""
    return COMMITTED_TOL_PPM.get(klass, COMMITTED_TOL_CLASSLESS_PPM) if klass else COMMITTED_TOL_CLASSLESS_PPM


def committed_configuration(counts: dict, d: float, mz: float, tol_ppm: float) -> dict:
    """The heavy configuration of the committed parent line that sits `d` Da
    above the ion's mono m/z: among the ion's Br / Cl / S / Si configurations
    (P >= COMMITTED_MIN_P) whose shift lies within `tol_ppm` of `d` (ppm of
    `mz`), the most probable; none within -> the mono line ({})."""
    if not (np.isfinite(d) and np.isfinite(mz)):
        return {}
    ck = heavy_key({k: v for k, v in counts.items() if k in ("Br", "Cl", "S", "Si")})
    tol = tol_ppm * 1e-6 * mz
    inside = [c for c in _committed_candidates(ck) if abs(d - c[1]) <= tol]
    if not inside:
        return {}
    return dict(max(inside, key=lambda c: c[2])[0])


_ENUM_ISOTOPES = ("13C", "81Br", "37Cl", "34S", "33S", "29Si", "30Si", "18O", "17O", "15N")


@lru_cache(maxsize=50000)
def _ion_lines(ckey: tuple, max_nominal: int = 12, floor: float = 1e-7) -> tuple:
    """Every heavy configuration of the ion (vs its mono line) with P >= floor
    and nominal shift <= max_nominal: ((key, shift, P), ...); 13C to four
    atoms, the halogens to their count, S / Si / O / N to two."""
    counts = dict(ckey)
    lines: list[tuple[dict, float]] = [({}, 1.0)]
    for iso in _ENUM_ISOTOPES:
        el = ISOTOPE_ELEMENT[iso]
        n = int(counts.get(el, 0))
        if n <= 0:
            continue
        kmax = {"13C": 4, "81Br": n, "37Cl": n}.get(iso, 2)
        new = []
        for h, _p in lines:
            used = sum(v for k, v in h.items() if ISOTOPE_ELEMENT[k] == el)
            for k in range(0, min(kmax, n - used) + 1):
                h2 = _heavy_add(h, {iso: k}) if k else dict(h)
                if _heavy_nominal(h2) > max_nominal:
                    break
                p = heavy_probability(h2, counts)
                if p < floor:
                    break
                new.append((h2, p))
        lines = new
    return tuple((heavy_key(h), heavy_shift(h), p) for h, p in lines)


def generic_expectation(counts: dict, hp: dict, delta: float, half_width: float) -> tuple[float, float]:
    """A generic 'M+n' child `delta` Da from its parent: (the ion's lines within
    `half_width` Da of that shift summed, relative to the committed line; their
    probability-weighted shift). (0.0, the nearest line's shift) when none."""
    sp = heavy_shift(hp)
    pp = heavy_probability(hp, counts)
    lines = _ion_lines(heavy_key(counts))
    near = [(s - sp, p) for _k, s, p in lines if abs((s - sp) - delta) <= half_width]
    if near and pp > 0:
        tot = sum(p for _s, p in near)
        return tot / pp, sum(s * p for s, p in near) / tot
    if not lines:
        return 0.0, delta
    return 0.0, min((s - sp for _k, s, _p in lines), key=lambda x: abs(x - delta))


def _label_elements(parsed) -> list[str]:
    els = []
    for kind, v in parsed:
        if kind == "set":
            els.extend(ISOTOPE_ELEMENT[i] for i in v)
        elif kind == "alt":
            for a in v:
                els.extend(ISOTOPE_ELEMENT[i] for i in a)
    return sorted(set(els))


def _expand(parsed) -> list[dict]:
    outs: list[dict] = [{}]
    for kind, v in parsed:
        if kind == "set":
            outs = [_heavy_add(o, v) for o in outs]
        elif kind == "alt":
            outs = [_heavy_add(o, a) for o in outs for a in v]
    return outs


def resolve_child(label, counts: dict, hp: dict, delta: float, *,
                  generic_half_width: float = GENERIC_HALF_WIDTH_DA) -> dict:
    """Read one child label against its committed parent (heavy set `hp`) at
    the measured child - parent shift `delta` (Da). Returns

      kind      'set' | 'mono' (a pure 'M0' label) | 'gen' ('M+n') | 'bad'
      shift     the exact child - parent shift of the reading nearer `delta`
      readings  the exact shifts the position test may place it at: both
                conventions (parent-relative, mono-counted); a pure 'M0' label
                only the mono line; none for 'M+n' (exempt) or an unreadable label
      expected  P(line) / P(committed line) under the nearer reading, the '+'
                parts' joint probability; 0.0 for a line the ion cannot make, a
                '14N' line (rule K judges it), an '18O' line of a Br / Cl ion,
                an 'M0' child of a mono-committed parent, an unreadable label
      elements  the elements whose heavy-atom count differs between the line and
                the committed line (a pure 'M0' line: the parent's heavy
                elements); none for 'M+n'
      parts     the label's parts other than 'M0' (their '+'-join is the label
                `iso_labels` lists)"""
    raw = split_label(label)
    parsed = [parse_label_part(p) for p in raw]
    parts = [p for p, (k, _v) in zip(raw, parsed) if k != "mono"]
    kinds = {k for k, _v in parsed}
    out = dict(kind="bad", shift=float("nan"), readings=(), expected=0.0, elements=[], parts=parts)
    if not parsed or "bad" in kinds:
        return out
    if "gen" in kinds:
        if len(parsed) != 1:
            return out
        e, s = generic_expectation(counts, hp, delta, generic_half_width)
        return dict(out, kind="gen", shift=s, expected=e)
    sp = heavy_shift(hp)
    pp = heavy_probability(hp, counts) if hp else 1.0
    alts = _expand(parsed)
    mono_only = not parts
    if not hp:
        conventions = ("same",)
    elif mono_only:
        conventions = ("M",)          # 'M0' names the mono line, never the parent itself
    else:
        conventions = ("P", "M")
    best = None
    readings = []
    for conv in conventions:
        cands = [_heavy_add(hp, a) for a in alts] if conv in ("same", "P") else alts
        ps = [heavy_probability(h, counts) for h in cands]
        tot = sum(ps)
        shift = (sum(heavy_shift(h) * p for h, p in zip(cands, ps)) / tot if tot > 0 else heavy_shift(cands[0])) - sp
        expected = tot / pp if pp > 0 else 0.0
        if any("14N" in h for h in cands):
            expected = 0.0
        if any("18O" in h for h in cands) and any(counts.get(e, 0) for e in M2_OWNERS):
            expected = 0.0
        readings.append(shift)
        cand = dict(shift=shift, expected=expected, heavy=cands[0])
        if best is None or abs(delta - shift) < abs(delta - best["shift"]):
            best = cand
    if mono_only:
        elements = sorted({ISOTOPE_ELEMENT[i] for i in hp})
        expected = best["expected"] if hp else 0.0
    else:
        diff = dict(best["heavy"])
        for k, v in hp.items():
            diff[k] = diff.get(k, 0) - v
        elements = sorted({ISOTOPE_ELEMENT[i] for i, v in diff.items() if v}) or _label_elements(parsed)
        expected = best["expected"]
    return dict(kind="mono" if mono_only else "set", shift=best["shift"], readings=tuple(readings),
                expected=expected, elements=elements, parts=parts)


def reagent_part(part: str, satellite: str | None) -> bool:
    """A label part naming the reagent halogen's heavy line alone ('81Br',
    '81Br2', '2x81Br', '81Br(pair)' on a bromide batch); a mixed or alternative
    part ('81Br37Cl(pair)', '81Br/37Cl(pair)') is not."""
    if not satellite:
        return False
    kind, v = parse_label_part(part)
    return kind == "set" and set(v) == {satellite}


class PositionSigma(NamedTuple):
    """sigma(h)^2 = a^2 + b^2 / h (ppm of the parent m/z; h = the child's
    height), fitted on a source's own '13C' children. `floored`: the
    intercept sits on SIGMA_A_FLOOR_PPM (the fit found no height-independent
    scatter at all)."""
    a: float
    b: float
    floored: bool
    n: int


def fit_position_sigma(residual_ppm, height) -> PositionSigma | None:
    """The height-aware sigma of a source's '13C' children (see the constants
    above); None below SIGMA_MIN_CHILDREN children (no position test there)."""
    r, h = np.asarray(residual_ppm, float), np.asarray(height, float)
    ok = np.isfinite(r) & np.isfinite(h) & (h > 0)
    r, h = r[ok], h[ok]
    if len(r) < SIGMA_MIN_CHILDREN:
        return None
    s0 = MAD_TO_SIGMA * np.median(np.abs(r - np.median(r)))
    k = np.abs(r) <= max(SIGMA_CLIP_PPM, SIGMA_CLIP_K * s0)
    r, h = r[k], h[k]
    q = np.quantile(h, np.linspace(0, 1, SIGMA_BINS + 1))
    xs, ys, ws = [], [], []
    for i in range(SIGMA_BINS):
        m = (h >= q[i]) & (h <= q[i + 1])
        if m.sum() < SIGMA_MIN_PER_BIN:
            continue
        s = MAD_TO_SIGMA * np.median(np.abs(r[m] - np.median(r[m])))
        xs.append(1 / np.median(h[m]))
        ys.append(s * s)
        ws.append(m.sum())
    X, Y, W = np.asarray(xs, float), np.asarray(ys, float), np.asarray(ws, float)
    A = np.vstack([np.ones_like(X), X]).T * np.sqrt(W)[:, None]
    coef, *_ = np.linalg.lstsq(A, Y * np.sqrt(W), rcond=None)
    floor = SIGMA_A_FLOOR_PPM ** 2
    return PositionSigma(a=float(np.sqrt(max(coef[0], floor))), b=float(np.sqrt(max(coef[1], 0.0))),
                         floored=bool(coef[0] <= floor), n=int(len(r)))


def position_window_ppm(height, fit: PositionSigma) -> float:
    """max(POSITION_MIN_PPM, POSITION_K x sigma(h)) at a child of `height`."""
    h = float(height)
    return float(max(POSITION_MIN_PPM, POSITION_K * math.sqrt(fit.a ** 2 + fit.b ** 2 / h))) \
        if np.isfinite(h) and h > 0 else float("nan")


#: the height-ratio band a line must fall in to count (evidence.RATIO_LO / RATIO_HI)
RATIO_BAND = (0.5, 2.0)


def _in_band(ratio: float, expected: float) -> bool:
    if not (expected and expected > 0 and np.isfinite(ratio)):
        return False
    rel = ratio / expected
    return bool(np.isfinite(rel) and RATIO_BAND[0] <= rel <= RATIO_BAND[1])


def _placed(c_mz: float, bases: list, readings, w_da: float) -> bool:
    return any(np.isfinite(c_mz - (b + e)) and abs(c_mz - (b + e)) <= w_da for b in bases for e in readings)


def _pulled(c_mz: float, c_h: float, p_mz: float, readings, own: str, parent: str, rows) -> bool:
    """N1: a row of the file >= NEIGHBOUR_RATIO x the child's height, within
    NEIGHBOUR_REACH_PPM of the exact position, on the side the child is
    displaced toward, the residual <= NEIGHBOUR_FRACTION of the distance to it
    (not the child itself, not its parent)."""
    if rows is None:
        return False
    fm, fh, fp = rows
    for e in readings:
        x = p_mz + e
        r = c_mz - x
        if not np.isfinite(r):
            continue
        sel = ((fm >= x * (1 - NEIGHBOUR_REACH_PPM * 1e-6)) & (fm <= x * (1 + NEIGHBOUR_REACH_PPM * 1e-6))
               & (fh >= NEIGHBOUR_RATIO * c_h) & (fp != own) & (fp != parent))
        dj = fm[sel] - x
        if np.any((np.sign(dj) == np.sign(r)) & (abs(r) <= NEIGHBOUR_FRACTION * np.abs(dj))):
            return True
    return False


def judge_source(children, parents: dict, lists, rows: dict, *, klass: str | None = None, fwhm=None,
                 guard: bool = False) -> dict:
    """The isotope children of ONE source (a file, or a batch's pooled files)
    judged against their committed parents: position (the self-fitted window,
    pcal, N1; 'M+n' exempt), count-aware expectation and band.

      parents   {(file, peak_id): {'mz', 'height', 'pcal', 'counts', 'sign'}} --
                every M0 row (the first per (file, peak_id)); `counts` the ION's
                element counts, `sign` its charge sign
      children  [{'file', 'peak_id', 'parent', 'label', 'mz', 'height'}] -- the
                iso_child rows joined to a parent
      lists     [(file, parent peak_id, [entry dict, ...])] -- the M0 rows'
                isotopologues lists (the scorer's record of the lines it scored)
      rows      {file: (mz, height, peak_id) arrays} -- the M0 + iso_child rows
                of each file: N1's neighbours and the list entries' lookup
      klass     'orbitrap' / 'tof' / None (the width model's class): the
                committed line's tolerance (COMMITTED_TOL_PPM)
      fwhm      the width model's FWHM(m/z) in Da, or None: an 'M+n' child's
                half-width
      guard     per file only: a TOF-class source whose fit sits on the
                intercept floor is not tested (its '13C' scatter collapsed)

    Returns {'fit': PositionSigma | None, 'tested': bool, 'children': [one
    verdict per child: keep (placed), ok (in band), expected, ratio, elements,
    parts, kind], 'lists': [one bool per list row: an entry that is not this
    parent's dropped child, names an M0 / iso row of the file, is placed at its
    own height (pcal allowed, no N1) and is in band]}."""
    tol = committed_tolerance_ppm(klass)
    hp_of: dict = {}
    for key, p in parents.items():
        mono = mono_mz(p["counts"], p["sign"])
        pmz = float(p["mz"]) if p["mz"] is not None else float("nan")
        hp_of[key] = committed_configuration(p["counts"], pmz - mono, pmz, tol)

    def half_width(mz: float) -> float:
        if fwhm is None or not np.isfinite(mz):
            return GENERIC_HALF_WIDTH_DA
        return max(GENERIC_HALF_WIDTH_DA, float(fwhm(mz)) / 2)

    # the self-fit on the source's own '13C' children
    s13 = ISOTOPE_SPACING["13C"]
    res13, h13 = [], []
    for c in children:
        if str(c["label"]).strip() == "13C":
            p = parents[(c["file"], c["parent"])]
            pmz = float(p["mz"])
            res13.append((float(c["mz"]) - pmz - s13) / pmz * 1e6)
            h13.append(float(c["height"]))
    fit = fit_position_sigma(res13, h13)
    tested = fit is not None and not (guard and klass == "tof" and fit.floored)

    verdicts = []
    dropped: dict = {}
    for c in children:
        key = (c["file"], c["parent"])
        p = parents[key]
        hp = hp_of[key]
        pmz, ph = float(p["mz"]), float(p["height"])
        cmz, ch = float(c["mz"]), float(c["height"])
        r = resolve_child(c["label"], p["counts"], hp, cmz - pmz, generic_half_width=half_width(cmz))
        keep = True
        if tested and r["kind"] != "gen":
            w = position_window_ppm(ch, fit) * pmz / 1e6
            pcal = float(p["pcal"]) if p["pcal"] is not None else float("nan")
            bases = [pmz] + ([pmz * (1 - pcal / 1e6)] if np.isfinite(pcal) else [])
            keep = (_placed(cmz, bases, r["readings"], w)
                    or _pulled(cmz, ch, pmz, r["readings"], str(c["peak_id"]), str(c["parent"]),
                               rows.get(c["file"])))
        ratio = ch / ph if (np.isfinite(ph) and ph > 0) else float("nan")
        verdicts.append(dict(keep=bool(keep), ok=_in_band(ratio, r["expected"]), expected=r["expected"],
                             ratio=ratio, elements=r["elements"], parts=r["parts"], kind=r["kind"]))
        if not keep:
            dropped.setdefault(key, set()).add(str(c["peak_id"]))

    look: dict = {}
    for f, (fm, fh, fp) in rows.items():
        for mz, h, pid in zip(fm, fh, fp):
            look.setdefault((f, str(pid)), (float(mz), float(h)))
    list_ok = []
    for f, parent, entries in lists:
        key = (f, parent)
        p = parents.get(key)
        passed = False
        if p is not None:
            hp = hp_of[key]
            pmz, ph = float(p["mz"]), float(p["height"])
            pcal = float(p["pcal"]) if p["pcal"] is not None else float("nan")
            bases = [pmz] + ([pmz * (1 - pcal / 1e6)] if np.isfinite(pcal) else [])
            gone = dropped.get(key, set())
            for e in entries:
                if not isinstance(e, dict) or str(e.get("peak_id")) in gone:
                    continue
                hit = look.get((f, str(e.get("peak_id"))))
                if hit is None or not np.isfinite(hit[0]):
                    continue
                emz, eh = hit
                r = resolve_child(e.get("label"), p["counts"], hp, emz - pmz, generic_half_width=half_width(emz))
                placed = True
                if tested and r["kind"] != "gen":
                    placed = _placed(emz, bases, r["readings"], position_window_ppm(eh, fit) * pmz / 1e6)
                ratio = eh / ph if (np.isfinite(ph) and ph > 0) else float("nan")
                if placed and _in_band(ratio, r["expected"]):
                    passed = True
                    break
        list_ok.append(passed)
    return dict(fit=fit, tested=tested, children=verdicts, lists=list_ok)


def line_facts(verdicts, satellite: str | None) -> dict:
    """The facts one pair's KEPT lines give (`verdicts` = its placed children):
    `iso` an in-band line; `lined` the elements of its in-band lines (C17's
    multiline reads those the neutral supplies); `carbon` a line that adds
    13C; `labels` the lines' labels without 'M0' parts; `reagent_only` every
    line naming a heavy atom names only the reagent halogen's, and none adds
    13C (a pure 'M0' line names none)."""
    lines = list(verdicts)
    ok = [v for v in lines if v["ok"]]
    naming = [v for v in lines if v["parts"]]
    carbon = any("C" in v["elements"] for v in lines)
    return dict(
        iso=bool(ok),
        lined=set().union(*(set(v["elements"]) for v in ok)) if ok else set(),
        carbon=carbon,
        labels={"+".join(v["parts"]) for v in naming},
        reagent_only=bool(satellite) and bool(naming) and not carbon
        and all(reagent_part(pt, satellite) for v in naming for pt in v["parts"]),
    )


#: the lead setters a halogen lock answers (rule H, C11+b; the engine's
#: evidence.LIFTABLE_LEADS): the three speculative-residual reasons, the
#: reference-list dim rescue and -- where the locked halogen is its only
#: violation -- the element budget; never the radical anion or the reagent-N re-read
LIFTABLE_LEADS = {"spec_n3", "spec_gapfill", "spec_minor", "reflist_dim", "off_budget"}


def lead_setters(value) -> set:
    """The setter codes a `lead_by` cell names; empty for "", NaN or None."""
    if not isinstance(value, str):
        return set()
    return {s for s in value.split("|") if s.strip() and s.strip().lower() != "nan"}


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


def source_resolution(path: str) -> dict | None:
    """A run dir's width model as its batch_summary.json records it (the
    `resolution` dict), for a run dir or an out-dir holding one run; None for a
    ledger CSV or a run that recorded none (a class-less source) -- the engine's
    evidence.source_resolution."""
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


#: an Orbitrap-class width model resolves at least this at m/z 200 (the engine's
#: evidence.ORBITRAP_R200 / batch/iso_checks.ORBITRAP_R200)
ORBITRAP_R200 = 50_000.0


def instrument(resolution) -> tuple:
    """(class, FWHM) of a run's `resolution` dict (batch_summary.json):
    ('orbitrap' | 'tof', FWHM(m/z) in Da); (None, None) without one -- the
    engine's evidence.instrument on the recorded model."""
    if not isinstance(resolution, dict) or resolution.get("coef") is None:
        return None, None
    coef, expo = float(resolution["coef"]), float(resolution.get("exponent", 1.0))
    off = float(resolution.get("offset", 0.0) or 0.0)

    def fwhm(mz: float) -> float:
        return coef * float(mz) ** expo + off
    r = 200.0 / fwhm(200.0)
    return ("orbitrap" if np.isfinite(r) and r >= ORBITRAP_R200 else "tof"), fwhm


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
    label: str, ledger: pd.DataFrame, halogen: str | None, resolution: dict | None = None,
    per_file: bool = False,
) -> pd.DataFrame:
    """One row of evidence per (neutral, adduct) the source committed.
    `resolution`: the source's width model (`source_resolution`; None =
    class-less); `per_file`: the source is one file of a run."""
    role = column(ledger, "role").astype(str)
    m0 = ledger[role == "M0"].copy()
    iso = ledger[role == "iso_child"]
    if m0.empty:
        return pd.DataFrame()

    m0["neutral_formula"] = column(m0, "neutral_formula").fillna("").astype(str)
    m0["adduct"] = column(m0, "adduct").fillna("").astype(str)
    m0["ion_only"] = is_ion_only(m0).to_numpy()
    regular = m0[~m0["ion_only"]]

    # the isotope children and the isotopologues lists, judged against their
    # committed parents (C11+c; `judge_source` above -- the engine's
    # evidence._judge_children): a child counts where it sits at its label's
    # exact spacing from the committed line (the source's self-fitted window,
    # pcal, N1; 'M+n' exempt), in band under its count-aware expectation; a
    # dropped child is dropped for every fact
    satellite = HALOGEN_SATELLITE.get(halogen) if halogen else None
    for name in ("peak_id", "ion_formula", "mz", "height", "ppm_error_cal", "isotopologues"):
        if name not in m0.columns:
            m0[name] = np.nan
    klass, fwhm = instrument(resolution)
    counts_of: dict = {}
    parents: dict = {}
    pair_of: dict = {}
    for f, pid, n, a, ion, mz, h, pc in zip(m0["__file"].astype(str), m0["peak_id"].astype(str),
                                            m0["neutral_formula"], m0["adduct"], m0["ion_formula"], m0["mz"],
                                            m0["height"], m0["ppm_error_cal"]):
        key = (f, pid)
        if key in parents:
            continue
        ck = (n, a, ion if isinstance(ion, str) else "")
        if ck not in counts_of:
            counts_of[ck] = ({e: v for e, v in ion_counts(n, a, ion).items() if v}, ion_sign(ion, a))
        counts, sign = counts_of[ck]
        parents[key] = dict(mz=to_float(mz), height=to_float(h), pcal=to_float(pc), counts=counts, sign=sign)
        pair_of[key] = (n, a)
    children = []
    for f, pid, par, lab, mz, h in zip(iso["__file"].astype(str), column(iso, "peak_id").astype(str),
                                       column(iso, "parent_peak_id"), column(iso, "iso_label"), column(iso, "mz"),
                                       column(iso, "height")):
        if not isinstance(par, str) and pd.isna(par):
            continue
        key = (f, str(par))
        if key not in parents:
            continue
        lab = str(lab).strip() if pd.notna(lab) else ""
        if not lab.split("+")[0].strip():
            continue
        children.append(dict(file=f, peak_id=pid, parent=str(par), label=lab, mz=to_float(mz), height=to_float(h)))
    lists = [(f, pid, as_list(v)) for f, pid, v in zip(m0["__file"].astype(str), m0["peak_id"].astype(str),
                                                        m0["isotopologues"])]
    readable = ledger[role.isin(["M0", "iso_child"])]
    rows_by_file = {str(f): (pd.to_numeric(column(g, "mz"), errors="coerce").to_numpy(float),
                             pd.to_numeric(column(g, "height"), errors="coerce").to_numpy(float),
                             column(g, "peak_id").astype(str).to_numpy(object))
                    for f, g in readable.groupby(readable["__file"].astype(str), sort=False)}
    judged = judge_source(children, parents, [x for x in lists if x[2]], rows_by_file, klass=klass, fwhm=fwhm,
                          guard=per_file)
    kept: dict = {}
    for c, verdict in zip(children, judged["children"]):
        if verdict["keep"]:
            kept.setdefault(pair_of[(c["file"], c["parent"])], []).append(verdict)
    facts_of = {k: line_facts(v, satellite) for k, v in kept.items()}
    no_lines = line_facts([], None)
    listed = iter(judged["lists"])
    m0["__list_ok"] = [bool(next(listed)) if entries else False for _f, _p, entries in lists]

    # per-neutral facts, over the source's REGULAR M0 rows (an ion-only row is
    # its parent's composition on another adduct, not a second channel for it)
    channels = regular.groupby("neutral_formula")["adduct"].nunique()
    adduct_sets = regular.groupby("neutral_formula")["adduct"].agg(
        lambda s: set(s.dropna().astype(str))
    )

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
        # C19(c): the lead half of the old flag; a ledger written before the
        # split carries none, reads False, and keeps its leads in below
        "tentative_lead",
        # C11+b: the lead's setter (ledger.LEAD_SETTERS); missing = any setter
        "lead_by",
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
        lines = facts_of.get(key) or no_lines
        methods = group["method"].astype(str)
        known = methods[methods.str.startswith("known:")]
        degeneracy = pd.to_numeric(group["degeneracy_density"], errors="coerce")
        seen = {
            str(v).strip()
            for v in group["resolvability"].dropna()
            if str(v).strip() and str(v).strip().lower() != "nan"
        }
        carbon_ev = lines["carbon"]
        ion_only = bool(group["ion_only"].any())
        below_row = group["below_assignability"].map(truthy).astype(bool)
        lead_row = group["tentative_lead"].map(truthy).astype(bool)
        clean_row = ~group["ion_only"].astype(bool) & ~below_row & ~lead_row
        lead_codes = [lead_setters(v) for v in group.loc[lead_row, "lead_by"]]
        own = lines["lined"] & neutral_elements(
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
                # an in-band kept line, or a list entry that answers the same question
                iso=lines["iso"] or bool(group["__list_ok"].any()),
                multiline=len(own) >= 2,
                multiline_elements="|".join(sorted(own)),
                carbon_ev=carbon_ev,
                chan2=(not ion_only) and int(channels.get(neutral, 0)) >= 2,
                anchor=(not ion_only) and bool(
                    group["anchor_peak_id"].notna().any()
                    or group["series_unit"].notna().any()
                ),
                branch=bool(aset & BARE_ADDUCTS) and bool(aset & CLUSTER_ADDUCTS),
                # every kept line naming a heavy atom names only the reagent
                # halogen's, none adds 13C -- and the reagent put it there: the
                # ION carries more of the reagent halogen than the neutral
                # (C11+a; the Br-free hold released by C11+c)
                reagent_only_iso=(not ion_only)
                and lines["reagent_only"]
                and carries_reagent(neutral, adduct, group["ion_formula"].iloc[0], halogen),
                iso_labels="|".join(sorted(lines["labels"])),
                tied=bool(group["tied"].map(truthy).all()),
                below=bool(below_row.any()),
                lead=bool(lead_row.any()),
                # rule H (C11+b): the setters behind the pair's lead rows (a lead
                # row naming none -- a ledger written before `lead_by` -- may be
                # the element budget's), whether it has an unflagged regular row,
                # and the anchor its unflagged rows give on their own
                lead_by="|".join(sorted(set().union(*lead_codes))) if lead_codes else "",
                lead_unknown=any(not c for c in lead_codes),
                clean_row=bool(clean_row.any()),
                # a pair with a regular (not ion-only) row gives its adduct to its
                # neutral's pools, row by row like the engine (relabel_pools)
                has_regular=bool((~group["ion_only"].astype(bool)).any()),
                anchor_clean=(not ion_only) and bool(
                    (group["anchor_peak_id"].notna() | group["series_unit"].notna())[~below_row & ~lead_row].any()),
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
    # a tentative lead (C19(c)) is hard like below assignability; a halogen lock
    # (rule H, C11+b) that lifts it has already set `lead` False (lift_leads)
    hard = (bool(row.tied) or bool(row.below) or bool(getattr(row, "lead", False))
            or bool(row.lowconf) or bool(getattr(row, "label_veto", False))
            or bool(getattr(row, "iso_veto", False)))
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


#: rule K's one-channel fold (the 14N and 15N nitrate clusters of one neutral)
LABEL_FOLD = {"[M+NO3]-": "[M+^NO3]-", "[M+15NO3]-": "[M+^NO3]-"}


def relabel_pools(df: pd.DataFrame, alien: set, fold: bool = True) -> pd.DataFrame:
    """Rule K / C11+ on the per-neutral facts: chan2 and branch recomputed over
    the regular rows minus the `alien` pairs (rule K's 14N lines, the pairs an
    isotope check refutes), with `fold` the two nitrate clusters of one neutral
    counted as one channel (rule K only); an alien row takes neither. Row by
    row, like the engine: a pair holding an ion-only row beside a regular one
    still gives its adduct to its siblings (`has_regular`; a measured frame
    without the column reads the pair-level ion-only flag) and takes neither
    itself."""
    df = df.copy()
    keys = list(zip(df["neutral"].astype(str), df["adduct"].astype(str)))
    alien_row = pd.Series([k in alien for k in keys], index=df.index, dtype=bool)
    out = alien_row | df["ion_only"].astype(bool)
    has_regular = (df["has_regular"].astype(bool) if "has_regular" in df.columns
                   else ~df["ion_only"].astype(bool))
    reg = df[~alien_row & has_regular]
    chans = reg.assign(ch=reg["adduct"].map(lambda a: LABEL_FOLD.get(a, a) if fold else a)) \
        .groupby("neutral")["ch"].nunique()
    adds = reg.groupby("neutral")["adduct"].agg(lambda s: set(s.astype(str)))
    df["chan2"] = [(not o) and int(chans.get(n, 0)) >= 2 for n, o in zip(df["neutral"], out)]
    df["branch"] = [(not o) and bool(adds.get(n, set()) & BARE_ADDUCTS) and bool(adds.get(n, set()) & CLUSTER_ADDUCTS)
                    for n, o in zip(df["neutral"], out)]
    return df


def lift_leads(df: pd.DataFrame, lock: dict, alien: set, fold: bool) -> pd.DataFrame:
    """Rule H (C11+b): a locked pair whose lead rows were all set by a setter
    the lock answers (the element budget's only where `budget_ok`), none below
    assignability, not alien / ion-only / refuted, is lifted: `lead` False, the
    lock its isotope axis, no reagent-only flag, and its second channel / acid
    branch / anchor read over unflagged rows alone -- its own adduct plus the
    adducts its neutral commits on an unflagged regular row of a pair that is
    not alien (`fold`: the two nitrate clusters count as one channel). Row by
    row, like the engine: a pair holding an ion-only row beside such a regular
    row still gives its adduct (`clean_row` already leaves the ion-only rows out)."""
    df = df.copy()
    df["lead_lift"] = False
    df["lock_note"] = ""
    if not lock:
        return df
    keys = list(zip(df["neutral"].astype(str), df["adduct"].astype(str)))
    alien_row = pd.Series([k in alien for k in keys], index=df.index, dtype=bool)
    out = alien_row | df["ion_only"].astype(bool)
    clean = df[df["clean_row"].astype(bool) & ~alien_row].groupby("neutral")["adduct"].agg(
        lambda s: set(s.astype(str)))
    for idx, k in zip(df.index, keys):
        spec = lock.get(k)
        if spec is None or out[idx] or bool(df.at[idx, "below"]) or not bool(df.at[idx, "lead"]):
            continue
        if bool(df.at[idx, "label_veto"]) or bool(df.at[idx, "iso_veto"]):
            continue
        setters = lead_setters(df.at[idx, "lead_by"])
        budget_ok = bool(spec.get("budget_ok", False))
        if not (setters <= LIFTABLE_LEADS and ("off_budget" not in setters or budget_ok)
                and (not bool(df.at[idx, "lead_unknown"]) or budget_ok)):
            continue
        own = {k[1]} | set(clean.get(k[0], set()))
        df.at[idx, "chan2"] = len({LABEL_FOLD.get(a, a) if fold else a for a in own}) >= 2
        df.at[idx, "branch"] = bool(own & BARE_ADDUCTS) and bool(own & CLUSTER_ADDUCTS)
        df.at[idx, "anchor"] = bool(df.at[idx, "anchor_clean"])
        df.at[idx, "iso"] = True
        df.at[idx, "reagent_only_iso"] = False
        df.at[idx, "lead"] = False
        df.at[idx, "lead_lift"] = True
        df.at[idx, "lock_note"] = str(spec.get("note", "") or "")
    return df


def assign_levels(df: pd.DataFrame, corroborating: set[str], upair: set[str] | None = None,
                  label: dict | None = None, iso: dict | None = None) -> pd.DataFrame:
    """Add the axes, the derived flags and the level to measured rows. `upair`
    is the batch's neutral-pair set (rule U); `label` the labelled-nitrate twin
    facts (rule K: {'untie', 'veto', 'alien'}, `label_twin_facts`): the 14N lines
    it could not tie to their neutral's 15N cluster leave the neutral's pools,
    the two clusters count as one channel, a tie the 15N sibling breaks is
    cleared and a refuted reading is hard; `iso` the isotope checks' vetoes and
    locks (C11+: {'veto': {(n, a): note}, 'lock': {(n, a): fact}},
    `iso_check_facts`): a refuted pair is hard and leaves its neutral's pools
    like an alien line; a locked pair's lead the lock answers is lifted
    (`lift_leads`, rule H). Facts, never axes."""
    df = df.copy()
    df["known_fam"] = df["known_fam"].fillna("")
    if "ion_only" not in df.columns:
        df["ion_only"] = False
    df["corroborated"] = df["neutral"].isin(corroborating) & ~df["ion_only"].astype(bool)
    df["upair"] = df["neutral"].isin(upair or set()) & ~df["ion_only"].astype(bool)
    iso_veto = {(str(n), str(a)): str(v or "") for (n, a), v in ((iso or {}).get("veto") or {}).items()}
    if label or iso_veto:
        alien = {(str(n), str(a)) for n, a in ((label or {}).get("alien") or set())} | set(iso_veto)
        df = relabel_pools(df, alien, fold=bool(label))
    keys = list(zip(df["neutral"].astype(str), df["adduct"].astype(str)))
    untie = {(str(n), str(a)) for n, a in ((label or {}).get("untie") or set())}
    veto = {(str(n), str(a)) for n, a in ((label or {}).get("veto") or {})}
    df["label_untie"] = (pd.Series([k in untie for k in keys], index=df.index, dtype=bool)
                         & df["tied"].astype(bool) & ~df["ion_only"].astype(bool))
    df.loc[df["label_untie"], "tied"] = False
    df["label_veto"] = pd.Series([k in veto for k in keys], index=df.index, dtype=bool) & ~df["ion_only"].astype(bool)
    # C11+: an isotope check refutes the formula (an ion-only pair too)
    df["iso_veto"] = pd.Series([k in iso_veto for k in keys], index=df.index, dtype=bool)
    df["iso_note"] = [iso_veto.get(k, "") if v else "" for k, v in zip(keys, df["iso_veto"])]
    # rule H (C11+b): the halogen locks lift the leads they answer
    lock = {(str(n), str(a)): dict(v or {}) for (n, a), v in ((iso or {}).get("lock") or {}).items()}
    for col, default in (("lead_by", ""), ("lead_unknown", True), ("clean_row", False), ("anchor_clean", False)):
        if col not in df.columns:
            df[col] = default
    df = lift_leads(df, lock, {(str(n), str(a)) for n, a in ((label or {}).get("alien") or set())} | set(iso_veto),
                    fold=bool(label))
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
    frame = measure_source(label, ledger, halogen, source_resolution(path))
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


def label_twin_facts(path: str) -> dict | None:
    """{'untie', 'veto', 'alien'} -- the labelled-nitrate twin facts (rule K)
    read from a batch's tables/label_twins.csv (a run dir, or an --out-dir holding
    one run) or from that CSV itself; None when there is no table or it is empty
    (out of scope). A source without the table says so on stderr."""
    table = path
    if os.path.isdir(path):
        table = os.path.join(path, "tables", "label_twins.csv")
        if not os.path.isfile(table):
            found = sorted(glob.glob(os.path.join(path, "*", "tables", "label_twins.csv")))
            if len(found) == 1:
                table = found[0]
    if not os.path.isfile(table):
        print(f"  --label-twins: no label_twins.csv for {path}; rule K does not fire there", file=sys.stderr)
        return None
    frame = pd.read_csv(table)
    if frame.empty:
        return None
    out = {}
    for col in ("untie", "veto", "alien"):
        if col not in frame.columns:
            out[col] = set()
            continue
        held = frame[col].map(truthy)
        out[col] = set(zip(frame.loc[held, "neutral_formula"].astype(str), frame.loc[held, "adduct"].astype(str)))
    return out


#: the order the isotope checks' notes join in, and each check's name there
ISO_CHECKS = ("C", "REQ", "HIGH", "H")
ISO_CHECK_NAME = {"C": "rule C", "REQ": "REQ", "HIGH": "HIGH", "H": "rule H"}


def iso_check_facts(path: str) -> dict | None:
    """{'veto': {(neutral, adduct): note}, 'lock': {(neutral, adduct): {'element',
    'n', 'budget_ok', 'note'}}} -- the isotope checks (C11+) read from a batch's
    tables/iso_checks.csv (a run dir, or an --out-dir holding one run) or from
    that CSV itself: one note per pair joined over the checks that refute it
    ('rule C: ...; REQ: ...'), and rule H's locks (C11+b; a table written before
    rule H has none); None when there is no table or it is empty (no time
    series). A source without the table says so on stderr."""
    table = path
    if os.path.isdir(path):
        table = os.path.join(path, "tables", "iso_checks.csv")
        if not os.path.isfile(table):
            found = sorted(glob.glob(os.path.join(path, "*", "tables", "iso_checks.csv")))
            if len(found) == 1:
                table = found[0]
    if not os.path.isfile(table):
        print(f"  --iso-checks: no iso_checks.csv for {path}; the isotope checks do not fire there", file=sys.stderr)
        return None
    frame = pd.read_csv(table)
    if frame.empty:
        return None
    veto: dict = {}
    if "veto" in frame.columns:
        held = frame[frame["veto"].map(truthy)].copy()
        order = {c: i for i, c in enumerate(ISO_CHECKS)}
        held["__o"] = held["check"].map(lambda c: order.get(str(c), len(order))) if "check" in held.columns else 0
        held = held.sort_values("__o", kind="mergesort")
        for _, r in held.iterrows():
            key = (str(r["neutral_formula"]), str(r["adduct"]))
            name = ISO_CHECK_NAME.get(str(r.get("check", "")), str(r.get("check", "")))
            note = r.get("note")
            piece = f"{name}: {note}" if isinstance(note, str) and note else name
            veto[key] = f"{veto[key]}; {piece}" if key in veto else piece
    lock: dict = {}
    if "lock" in frame.columns and "check" in frame.columns:
        for _, r in frame[(frame["check"].astype(str) == "H") & frame["lock"].map(truthy)].iterrows():
            n = pd.to_numeric(r.get("n_halogen"), errors="coerce")
            el, note = r.get("element"), r.get("note")
            lock[(str(r["neutral_formula"]), str(r["adduct"]))] = {
                "element": el if isinstance(el, str) else "", "n": int(n) if pd.notna(n) else 0,
                "budget_ok": truthy(r.get("budget_ok")), "note": note if isinstance(note, str) else ""}
    return {"veto": veto, "lock": lock}


def run(sources: list[str], corroborate: list[str], upair: str | None = None,
        twins: str | None = None, iso: str | None = None) -> pd.DataFrame:
    """Level every source, each corroborated by the others plus --corroborate —
    by the neutrals each of them holds at 4b or better on its own evidence.
    `upair`: 'auto' reads each run-dir source's own tables/neutral_pairs.csv;
    a CSV path applies that table to every levelled source; None = rule U off.
    `twins`: the same for tables/label_twins.csv (rule K). `iso`: the same for
    tables/iso_checks.csv (C11+, its vetoes and rule H's locks); a named table
    applies only to a source that holds every pair it vetoes or locks -- the
    checks were measured on one batch's time series and name its pairs, so a
    source missing one of them is another batch and the table does not fire
    there."""
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
    for path, (label_, frame) in measured.items():
        others: set[str] = set()
        for other, values in neutrals.items():
            if other != path:
                others |= values
        held = set()
        if upair == "auto":
            held = upair_neutrals(path)
        elif upair:
            held = upair_neutrals(upair)
        label = None
        if twins == "auto":
            label = label_twin_facts(path)
        elif twins:
            # one named table applies only to the source it was measured on: every
            # pair it names (untie, veto, alien) must be one of the source's pairs
            facts = label_twin_facts(twins)
            named = set().union(*(set(facts[k]) for k in ("untie", "veto", "alien"))) if facts else set()
            pairs = set(zip(frame["neutral"].astype(str), frame["adduct"].astype(str)))
            if facts is not None and named <= pairs:
                label = facts
            elif facts is not None:
                print(f"  --label-twins: {label_} does not hold the pairs the table names; rule K does not fire there",
                      file=sys.stderr)
        checks = None
        if iso == "auto":
            checks = iso_check_facts(path)
        elif iso:
            checks = iso_check_facts(iso)
            pairs = set(zip(frame["neutral"].astype(str), frame["adduct"].astype(str)))
            missing = (set((checks or {}).get("veto") or {}) | set((checks or {}).get("lock") or {})) - pairs
            if missing:
                print(f"  --iso-checks: {label_} does not hold {len(missing)} pair(s) the table vetoes or locks "
                      f"(another batch's table); the isotope checks do not fire there", file=sys.stderr)
                checks = None
        out.append(assign_levels(frame, others, held, label, checks))
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
    parser.add_argument(
        "--label-twins",
        nargs="?",
        const="auto",
        default=None,
        dest="twins",
        help="rule K: read each batch source's tables/label_twins.csv (no value), "
        "or apply this label-twin CSV to every levelled source",
    )
    parser.add_argument(
        "--iso-checks",
        nargs="?",
        const="auto",
        default=None,
        dest="iso",
        help="C11+: read each batch source's tables/iso_checks.csv (no value), "
        "or apply this isotope-check CSV to the levelled sources that hold every pair it vetoes or locks "
        "(its vetoes refute; rule H's locks lift the tentative leads they answer)",
    )
    parser.add_argument("--out", help="write the levelled rows here as CSV")
    args = parser.parse_args(argv)

    df = run(args.sources, args.corroborate, args.upair, args.twins, args.iso)
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
