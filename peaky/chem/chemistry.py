"""Chemistry core: exact masses, formula algebra, DBE / Senior rules, and the
theoretical neutral-formula grid.

This is the lowest layer of the package. It knows nothing about Mascope, about
experimental contexts, or about peaks. It provides the primitives every other
module builds on:

  * exact monoisotopic masses + electron mass
  * parse / format chemical formulas
  * neutral_mass(formula), ion_mz(neutral_mass, adduct)
  * dbe(formula)  -- Double Bond Equivalents of the NEUTRAL
  * dbe_ok / seniors_ok -- the hard structural gates
  * enumerate_grid(ranges) -- all plausible NEUTRAL formulas in a box
  * candidates_for_peaks(...) -- grid pre-filtered to observed peak m/z

Design decision -- DBE is enforced on the NEUTRAL molecule:
    A real neutral molecule always has a non-negative INTEGER DBE. Half-integer
    DBE only appears on an *ion* formula because (de)protonation / adduct
    formation flips H parity. So the grid emits neutrals with integer DBE >= 0
    only; organic nitrates are still produced (neutral C8H15NO12 has integer
    DBE; it is the observed ion C8H14NO12- that is half-integer, which is
    expected and handled at the ion layer, not here).
"""
from __future__ import annotations

import bisect
import re
from typing import Iterable, Iterator

__version__ = "0.1.0"

# ---------------------------------------------------------------------------
# Exact monoisotopic masses (most-abundant isotope) and the electron mass.
# ---------------------------------------------------------------------------
M: dict[str, float] = {
    "C": 12.0,
    "H": 1.0078250319,
    "O": 15.9949146221,
    "N": 14.0030740052,
    "S": 31.97207069,
    "P": 30.97376163,
    "Si": 27.9769265350,
    "F": 18.9984031627,
    "Cl": 34.96885268,
    "Br": 78.9183371,
    "I": 126.9044719,
    # caret-labelled heavy isotope for labelled-reagent chemistry: '^N' = 15N.
    # A covalent 15N enters a product only via labelled-reagent chemistry (e.g. a
    # 15NO3 radical adding to a VOC -> organonitrate); the reagent cluster's 15N
    # is separate (handled by the [M+^NO3]- adduct). Same '^N' notation the
    # mascope scoring stack uses (predict_isotopes), so an '^N' ion formula scores
    # directly and even yields a '14N' isotopologue (the ~2% 14N reagent impurity)
    # as a real confirmation channel.
    "^N": 15.0001088984,
}
M_E = 0.0005485799  # electron mass (Da)
# Heavy stable isotopes used when a REAGENT is isotopically labelled (the analyte
# grid itself is monoisotopic). ¹⁵N nitrate CIMS uses a ¹⁵N-labelled NO3 reagent,
# so its cluster adduct adds ¹⁵N (15.0001089), not ¹⁴N -- a ~0.997 Da shift that
# would otherwise put every nitrate-adduct assignment ~1 Da off.
_M_15N = 15.0001088989

# Valence used for the Senior / DBE accounting. O and S are treated divalent
# (zero DBE contribution); Si tetravalent like C; P trivalent like N for the
# standard organic DBE formula.
_DBE_PLUS = ("C", "Si")            # contribute +1 each (valence 4)
_DBE_HALF_PLUS = ("N", "P", "^N")  # contribute +1/2 each (valence 3; ^N = 15N)
_DBE_HALF_MINUS = ("H", "F", "Cl", "Br", "I")  # contribute -1/2 each (valence 1)

# Element set we model. Output uses Hill notation (C, H, then alphabetical),
# matching how Mascope formats formula strings -- important for string-level
# comparison against match_compounds output.
_ORDER = ["C", "H", "N", "O", "S", "P", "Si", "F", "Cl", "Br", "I"]

# ---------------------------------------------------------------------------
# Adduct / ion-form shifts:  ion_mz = neutral_mass + shift
# Negative-mode and positive-mode singly-charged forms.
# ---------------------------------------------------------------------------
ADDUCT_SHIFTS: dict[str, float] = {
    # negative mode
    "[M-H]-":   -(M["H"]) + M_E,
    "[M+Br]-":  M["Br"] + M_E,
    "[M+Cl]-":  M["Cl"] + M_E,
    "[M+I]-":   M["I"] + M_E,
    # iodide-CIMS poly-iodide analyte clusters (server mechanisms +I2- / +I3-,
    # already in ADDUCT_TO_MECH). I is monoisotopic, so these are the ONLY way a
    # neutral picks up 2/3 iodines -- the reagent I2⁻·/I3⁻ ladder clustering onto
    # an analyte (mostly small inorganics: NO2/Cl/O give the bright I2NO2⁻/I2Cl⁻/
    # I2O⁻ source-background lines). The bare In⁻ clusters are labelled reagent.
    "[M+I2]-":  2 * M["I"] + M_E,
    "[M+I3]-":  3 * M["I"] + M_E,
    # DEPROTONATED-acid + I2 cluster (A⁻·I₂): ion = M - H + 2 I. The iodide
    # analog of [M+HBr+Br]- -- a cluster-DECOMPOSITION alias, NOT a server
    # mechanism (deliberately absent from ADDUCT_TO_MECH; scoring goes through
    # the covalent alias (M-H+I) [M+I]-, the SAME ion). Acids the run already
    # believes on [M+I]-/[M-H]- also appear as their conjugate base bound to I₂
    # (I₂⁻· is the brightest reagent ion): [HCOOH-H+I2]- @298.807 -- exactly
    # degenerate with the off-grid covalent CHIO2 [M+I]- reading the series
    # passes once invented. [M+I2]- of A-H can never claim these (A-H is a
    # radical, blocked by integer-DBE), which is why they sat unexplained.
    "[M-H+I2]-": 2 * M["I"] - M["H"] + M_E,
    "[M+NO3]-": M["N"] + 3 * M["O"] + M_E,
    # ¹⁵N-labelled nitrate reagent cluster (server mechanism '+^NO3-'); the added
    # N is ¹⁵N, so this is +62.9855, not the +61.9885 of the ¹⁴N adduct above.
    "[M+^NO3]-": _M_15N + 3 * M["O"] + M_E,
    "[M+HSO4]-": 2 * M["H"] + M["S"] + 4 * M["O"] - M["H"] + M_E,  # = H + S + 4O
    "[M+CHO2]-": M["C"] + M["H"] + 2 * M["O"] + M_E,   # formate
    "[M+C2H3O2]-": 2 * M["C"] + 3 * M["H"] + 2 * M["O"] + M_E,  # acetate
    # background air-ion channels (atmospheric-pressure negative sources):
    # carbonate and superoxide adducts, plus electron-attachment radical anion
    "[M+CO3]-": M["C"] + 3 * M["O"] + M_E,
    "[M+O2]-": 2 * M["O"] + M_E,
    "[M]-.": M_E,
    # reagent-HBr cluster on a background channel: the SAME ion as a covalent
    # bromo-analyte + CO3, decomposed so the reagent Br sits in the cluster
    # (Y.HBr) and the reported neutral Y is bromine-free (user reagent rule).
    "[M+HBr+CO3]-": M["H"] + M["Br"] + M["C"] + 3 * M["O"] + M_E,
    # di-bromide analyte clusters (the "C/H lattice" peaks, 2026-06-12). The
    # bright n_Br=2 residual is biogenic SOA (mono-/sesquiterpene oxidation
    # products + N/S species) detected as reagent di-bromide clusters, NOT
    # exotic covalent organohalogens:
    #   [M+Br2]-     M . Br2 radical-anion adduct (server mechanism +Br2-)
    #   [M+HBr+Br]-  M . HBr . Br-  (the H-bearing cluster most peaks actually
    #                fit; same ion as covalent Y(Br) [M+Br]-, so _prefer_adduct_
    #                reading relabels onto it -- needs to be registered here).
    "[M+Br2]-":    2 * M["Br"] + M_E,
    "[M+HBr+Br]-": M["H"] + 2 * M["Br"] + M_E,
    # tribromide channels (Br3- is a DOMINANT reagent ion in this CH2Br2 source,
    # ~129k cps; server mechanism +Br3- registered 2026-06-12):
    #   [M+Br3]-      M . Br3- tribromide cluster (3 Br, native mechanism)
    #   [M+HBr+Br2]-  M . HBr . Br2-  (covalent-Br alias of [M+Br2]- analytes,
    #                 for _prefer_adduct_reading to relabel onto)
    "[M+Br3]-":     3 * M["Br"] + M_E,
    "[M+HBr+Br2]-": M["H"] + 3 * M["Br"] + M_E,
    # positive mode
    "[M+H]+":   M["H"] - M_E,
    "[M+Na]+":  22.9897692820 - M_E,
    "[M+NH4]+": M["N"] + 4 * M["H"] - M_E,
    "[M+K]+":   38.9637064864 - M_E,
    # protonated-urea (URONIUM) adduct -- the analyte channel of a urea-CIMS
    # positive-mode source: [M + (CH4N2O) + H]+ = M + urea + proton (server
    # mechanism '+(CH4N2O)H+'). Shift = C + 5H + 2N + O - e  (= 61.0396).
    "[M+(CH4N2O)H]+": M["C"] + 5 * M["H"] + 2 * M["N"] + M["O"] - M_E,
    # EasyIC⁺ (fluoranthene-cation charge transfer, low pressure) channels:
    #   [M]+.   molecular RADICAL cation (server mechanism '+') -- charge
    #           transfer keeps aromatics intact (benzene 78.0464, toluene
    #           92.0621); the positive twin of the [M]-. above
    #   [M-H]+  HYDRIDE abstraction: ion = M - H⁻ (alcohols/alkanes -- ethanol's
    #           ONLY channel, C2H5O+ @45.0335). NO server mechanism (the
    #           [M-H+I2]- ruling: deliberately absent from ADDUCT_TO_MECH, a
    #           local-scoring channel). Writing the ion as [M-H]+ of the intact
    #           molecule keeps the NEUTRAL closed-shell, so the integer-DBE
    #           gate holds ([M]+. of the M-H radical is the same ion, blocked).
    "[M]+.": -M_E,
    "[M-H]+": -(M["H"]) - M_E,
}


# ---------------------------------------------------------------------------
# Formula parsing / formatting
# ---------------------------------------------------------------------------
# optional caret prefix = a labelled heavy isotope (e.g. '^N' = 15N). Matched as
# part of the element token so 'C5H7^NO6' parses ^N as its own species, not 14N.
_TOKEN = re.compile(r"(\^?[A-Z][a-z]?)(\d*)")


def parse_formula(formula: str) -> dict[str, int]:
    """'C10H16O4' -> {'C':10,'H':16,'O':4}. Robust to empty / None. A caret
    element ('^N' = 15N) is kept as a distinct key so its mass/isotope pattern is
    correct; fold_isotopes() merges it back to the base element for classification."""
    out: dict[str, int] = {}
    if not isinstance(formula, str):        # tolerate NaN / None from arrow columns
        return out
    for el, n in _TOKEN.findall(formula):
        if not el:
            continue
        out[el] = out.get(el, 0) + (int(n) if n else 1)
    return out


# base element each caret isotope collapses to for VALENCE / CLASSIFICATION
# (mass keeps them distinct; chemistry-class semantics do not).
_ISOTOPE_BASE = {"^N": "N"}


def fold_isotopes(counts: dict[str, int]) -> dict[str, int]:
    """Merge caret heavy-isotope counts into their base element ('^N' -> 'N').
    Use where only the element TOTAL matters (CHON classification, N-count gates,
    Van Krevelen); never for mass (isotopes differ) or isotope-pattern scoring."""
    if not any(k in _ISOTOPE_BASE for k in counts):
        return dict(counts)
    out: dict[str, int] = {}
    for k, v in counts.items():
        out[_ISOTOPE_BASE.get(k, k)] = out.get(_ISOTOPE_BASE.get(k, k), 0) + v
    return out


def format_formula(counts: dict[str, int]) -> str:
    """{'C':10,'H':16,'O':4} -> 'C10H16O4' in Hill notation: C first, H second,
    then every other element in alphabetical order of its symbol (matches
    Mascope's formatting, e.g. 'CH2BrO2')."""
    s = ""
    if counts.get("C", 0) > 0:
        s += "C" + (str(counts["C"]) if counts["C"] > 1 else "")
    if counts.get("H", 0) > 0:
        s += "H" + (str(counts["H"]) if counts["H"] > 1 else "")
    # sort ignoring a leading caret so '^N' files next to 'N' rather than last;
    # the caret is preserved in the emitted symbol so parse_formula round-trips.
    for el in sorted((k for k in counts if k not in ("C", "H")),
                     key=lambda e: e.lstrip("^")):
        k = counts[el]
        if k > 0:
            s += el + (str(k) if k > 1 else "")
    return s


def neutral_mass(formula: str | dict[str, int]) -> float:
    cnt = formula if isinstance(formula, dict) else parse_formula(formula)
    return sum(M[el] * n for el, n in cnt.items() if el in M)


def ion_mz(neutral: str | dict[str, int] | float, adduct: str) -> float:
    """Theoretical m/z of a singly-charged ion of `neutral` under `adduct`."""
    if adduct not in ADDUCT_SHIFTS:
        raise KeyError(f"unknown adduct {adduct!r}; known: {sorted(ADDUCT_SHIFTS)}")
    mass = neutral if isinstance(neutral, (int, float)) else neutral_mass(neutral)
    return mass + ADDUCT_SHIFTS[adduct]


# ---------------------------------------------------------------------------
# DBE / Senior rules  (computed on the NEUTRAL)
# ---------------------------------------------------------------------------
def dbe(formula: str | dict[str, int]) -> float:
    """Double Bond Equivalents of the neutral.

    DBE = 1 + (C+Si) + (N+P)/2 - (H+F+Cl+Br+I)/2
    O and S are divalent -> no contribution.
    """
    cnt = formula if isinstance(formula, dict) else parse_formula(formula)
    val = 1.0
    val += sum(cnt.get(el, 0) for el in _DBE_PLUS)
    val += sum(cnt.get(el, 0) for el in _DBE_HALF_PLUS) / 2.0
    val -= sum(cnt.get(el, 0) for el in _DBE_HALF_MINUS) / 2.0
    return val


def seniors_cap(cnt: dict[str, int]) -> float:
    """Maximum DBE permitted by Senior's rule for these element counts."""
    return (cnt.get("C", 0) + cnt.get("Si", 0)
            + (cnt.get("N", 0) + cnt.get("^N", 0)) / 2.0 + 1.0)


def oxygen_ok(formula: str | dict[str, int]) -> tuple[bool, str | None]:
    """Structural oxygen cap: O <= 2*(C+N+S+P) + 4.

    This is valence chemistry, not a statistical Van Krevelen prior: every
    oxygen needs a bonding site on the C/N/S/P skeleton (2 per skeletal atom,
    +4 headroom for peroxide chains / terminal acids). DBE cannot constrain
    oxygen (divalent O contributes zero), so without this cap the O-rich corner
    of formula space can hit ANY mass within tolerance (e.g. C3H5ClO17).
    Real HOMs (C10H18O7, O/C 0.7) and inorganic acids (H2SO4, HNO3) pass."""
    cnt = formula if isinstance(formula, dict) else parse_formula(formula)
    skeleton = (cnt.get("C", 0) + cnt.get("N", 0) + cnt.get("^N", 0)
                + cnt.get("S", 0) + cnt.get("P", 0))
    cap = 2 * skeleton + 4
    O = cnt.get("O", 0)
    if O > cap:
        return False, f"O={O} > structural cap {cap} (2*(C+N+S+P)+4)"
    return True, None


def dbe_ok(formula: str | dict[str, int], tol: float = 1e-9) -> tuple[bool, str | None]:
    """Hard structural gate on the neutral: DBE must be a non-negative INTEGER
    and satisfy Senior's rule. Returns (ok, reason_if_not)."""
    cnt = formula if isinstance(formula, dict) else parse_formula(formula)
    d = dbe(cnt)
    if d < -tol:
        return False, f"DBE={d:g} < 0"
    if abs(d - round(d)) > tol:
        return False, f"DBE={d:g} is half-integer (not a valid neutral)"
    cap = seniors_cap(cnt)
    if d > cap + tol:
        return False, f"DBE={d:g} > Senior cap {cap:g}"
    return True, None


# ---------------------------------------------------------------------------
# Theoretical neutral-formula grid
# ---------------------------------------------------------------------------
def parse_ranges(s: str) -> dict[str, tuple[int, int]]:
    """'C0-30 H0-60 O0-15 N0-5 Br0-2' -> {'C':(0,30), ...}. A caret element
    ('^N0-2') is accepted for labelled-reagent grids."""
    out: dict[str, tuple[int, int]] = {}
    for tok in s.split():
        m = re.match(r"(\^?[A-Z][a-z]?)(\d+)-(\d+)", tok)
        if m:
            out[m.group(1)] = (int(m.group(2)), int(m.group(3)))
    return out


def enumerate_grid(
    ranges: dict[str, tuple[int, int]],
    mass_min: float = 30.0,
    mass_max: float = 900.0,
) -> list[tuple[float, str]]:
    """Enumerate (neutral_mass, formula) for all formulas in the element box
    that satisfy DBE >= 0, integer DBE, and Senior's rule.

    H is *derived* from DBE rather than iterated, so the only valid (integer,
    non-negative) DBE values are visited. This is both correct and fast.
    """
    elements = _ORDER
    b = {el: ranges.get(el, (0, 0)) for el in elements}
    out: list[tuple[float, str]] = []

    for C in range(b["C"][0], b["C"][1] + 1):
        if C * M["C"] > mass_max:
            break
        for Si in range(b["Si"][0], b["Si"][1] + 1):
            for N in range(b["N"][0], b["N"][1] + 1):
                for P in range(b["P"][0], b["P"][1] + 1):
                    for F in range(b["F"][0], b["F"][1] + 1):
                        for Cl in range(b["Cl"][0], b["Cl"][1] + 1):
                            for Br in range(b["Br"][0], b["Br"][1] + 1):
                                for I in range(b["I"][0], b["I"][1] + 1):
                                    # Senior cap fixes the max DBE; iterate
                                    # integer DBE 0..cap and derive H.
                                    cap = int(C + Si + N / 2.0 + 1.0)
                                    halo = F + Cl + Br + I
                                    for d in range(0, cap + 1):
                                        # DBE = 1 + (C+Si) + (N+P)/2 - (H+halo)/2
                                        # -> H = 2*(1+C+Si) + (N+P) - 2*d - halo
                                        H = 2 * (1 + C + Si) + (N + P) - 2 * d - halo
                                        if H < 0:
                                            continue
                                        if not (b["H"][0] <= H <= b["H"][1]):
                                            continue
                                        for S in range(b["S"][0], b["S"][1] + 1):
                                            # structural oxygen cap (see oxygen_ok)
                                            o_hi = min(b["O"][1],
                                                       2 * (C + N + S + P) + 4)
                                            for O in range(b["O"][0], o_hi + 1):
                                                mass = (
                                                    C * M["C"] + H * M["H"]
                                                    + N * M["N"] + O * M["O"]
                                                    + S * M["S"] + P * M["P"]
                                                    + Si * M["Si"] + F * M["F"]
                                                    + Cl * M["Cl"] + Br * M["Br"]
                                                    + I * M["I"]
                                                )
                                                if mass < mass_min or mass > mass_max:
                                                    continue
                                                cnt = {
                                                    "C": C, "H": H, "N": N, "O": O,
                                                    "S": S, "P": P, "Si": Si,
                                                    "F": F, "Cl": Cl, "Br": Br, "I": I,
                                                }
                                                out.append((mass, format_formula(cnt)))
    return out


# Per-atom complexity penalty used by arbitration to break score ties in favour
# of the simpler (CHO < CHON < heteroatom-rich) interpretation. A neutral
# halogen must out-score a CHO competitor by a real margin to win.
_COMPLEXITY_WEIGHT = {"N": 3, "S": 8, "P": 25, "Cl": 50, "Br": 50, "Si": 80, "I": 80, "F": 30}


def complexity_penalty(formula: str | dict[str, int], scale: float = 0.01,
                       cap: float = 0.20) -> float:
    """Score penalty in [0, cap] proportional to heteroatom complexity."""
    cnt = formula if isinstance(formula, dict) else parse_formula(formula)
    raw = sum(_COMPLEXITY_WEIGHT.get(el, 0) * n for el, n in cnt.items())
    return min(raw * scale, cap)


_GRID_CACHE: dict = {}
_GRID_CACHE_MAX = 16


def _grid_cached(ranges: dict[str, tuple[int, int]], mass_min: float,
                 mass_max: float) -> list[tuple[float, str]]:
    """Memoised sorted grid -- enumeration is pure, and callers like chain-head
    candidate generation hit the same element box many times per run."""
    key = (tuple(sorted(ranges.items())), round(mass_min, 3), round(mass_max, 3))
    g = _GRID_CACHE.get(key)
    if g is None:
        g = sorted(enumerate_grid(ranges, mass_min, mass_max))
        if len(_GRID_CACHE) >= _GRID_CACHE_MAX:
            _GRID_CACHE.clear()
        _GRID_CACHE[key] = g
    return g


def candidates_for_peaks(
    peak_mzs: Iterable[float],
    ranges: dict[str, tuple[int, int]],
    adducts: Iterable[str],
    ppm_tolerance: float,
    mass_min: float = 30.0,
    mass_max: float = 900.0,
) -> set[str]:
    """Return neutral formulas whose theoretical mass matches at least one peak
    m/z under at least one adduct, within ppm tolerance."""
    grid = _grid_cached(ranges, mass_min, mass_max)
    masses = [g[0] for g in grid]
    shifts = [ADDUCT_SHIFTS[a] for a in adducts if a in ADDUCT_SHIFTS]
    accepted: set[str] = set()
    for mz in peak_mzs:
        for shift in shifts:
            m_neu = mz - shift
            if m_neu < mass_min or m_neu > mass_max:
                continue
            tol = m_neu * ppm_tolerance * 1e-6
            lo = bisect.bisect_left(masses, m_neu - tol)
            hi = bisect.bisect_right(masses, m_neu + tol)
            for i in range(lo, hi):
                accepted.add(grid[i][1])
    return accepted
