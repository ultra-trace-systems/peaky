"""Generalized Kendrick Analysis (GKA) -- the Pass-2 series engine.

After Pass 1 locks a high-confidence backbone, Pass 2 propagates those anchors
along exact repeat units to explain neighbouring unassigned peaks:

  * organic growth / oxidation: CH2, H2O, O, CO, CO2, C2H2O
  * siloxane (PDMS) contaminant series: C2H6OSi
  * fluorinated (PFAS/CF2) contaminant series: CF2

This module proposes candidate NEUTRAL formulas; it never decides truth.
passes.py validates every proposal with Mascope match_compounds and keeps a
candidate only if the isotopologue pattern corroborates it. A candidate
supported by >=2 independent anchors (the formula one unit below AND above are
both anchors) is the GKA strength signal and earns higher confidence.

GKA math (Alton et al., AMT 2023):
    GKM(mz, base, X) = mz * X / mass(base)
    GKD = GKM - round(GKM)
X = nucleon number of the base reproduces traditional Kendrick analysis; a
larger integer X expands the defect scale to separate congested series.
"""
from __future__ import annotations

from dataclasses import dataclass

from peaky.chem import chemistry as C

__version__ = "0.1.0"

# Repeat units as element-count deltas.
REPEAT_UNITS: dict[str, dict[str, int]] = {
    "CH2":    {"C": 1, "H": 2},
    "H2O":    {"H": 2, "O": 1},
    "O":      {"O": 1},
    "CO":     {"C": 1, "O": 1},
    "CO2":    {"C": 1, "O": 2},
    "C2H2O":  {"C": 2, "H": 2, "O": 1},
    "C2H6OSi": {"C": 2, "H": 6, "O": 1, "Si": 1},   # PDMS / siloxane
    "CF2":    {"C": 1, "F": 2},                       # fluorinated / PFAS
    "HBr":    {"H": 1, "Br": 1},   # cluster unit (halide CIMS), not growth
    "HCl":    {"H": 1, "Cl": 1},   # cluster unit
    "HI":     {"H": 1, "I": 1},    # cluster unit (iodide CIMS; also lets
                                   # _prefer_adduct_reading strip covalent I)
    "C2H4O2": {"C": 2, "H": 4, "O": 2},   # acetic-acid cluster / esterification
    "C2F4":   {"C": 2, "F": 4},           # double-CF2 step (PFAS ladders)
    "C2H4O":  {"C": 2, "H": 4, "O": 1},   # PEG / ethoxylate ladder
    "C3H6O":  {"C": 3, "H": 6, "O": 1},   # PPG / propoxylate ladder
}

# Organic-growth units used for CHO/CHON propagation by default.
ORGANIC_UNITS = ("CH2", "H2O", "O", "CO", "CO2", "C2H2O")
CONTAMINANT_UNITS = ("C2H6OSi", "CF2")


def unit_mass(unit: str) -> float:
    return C.neutral_mass(REPEAT_UNITS[unit])


# ---------------------------------------------------------------------------
# Formula arithmetic
# ---------------------------------------------------------------------------
def formula_add(formula: str | dict[str, int], unit: str, n: int) -> str | None:
    """Return formula +/- n*unit as a canonical string, or None if any element
    count would go negative."""
    cnt = dict(C.parse_formula(formula) if isinstance(formula, str) else formula)
    delta = REPEAT_UNITS[unit]
    for el, k in delta.items():
        cnt[el] = cnt.get(el, 0) + n * k
        if cnt[el] < 0:
            return None
    cnt = {el: v for el, v in cnt.items() if v > 0}
    if not cnt:
        return None
    return C.format_formula(cnt)


# ---------------------------------------------------------------------------
# GKA / KMD math
# ---------------------------------------------------------------------------
def gkm(mz: float, base_unit: str, X: int) -> float:
    return mz * X / unit_mass(base_unit)


def gkd(mz: float, base_unit: str, X: int) -> float:
    g = gkm(mz, base_unit, X)
    return g - round(g)


# ---------------------------------------------------------------------------
# Anchor-based propagation
# ---------------------------------------------------------------------------
@dataclass
class Proposal:
    target_mz: float
    neutral_formula: str
    adduct: str
    anchor_formula: str
    unit: str
    n_steps: int
    ppm_error: float
    n_supporting_anchors: int = 1   # filled by support count

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def propose_for_peak(target_mz: float,
                     anchor_formulas: set[str],
                     adducts: list[str],
                     *, units=ORGANIC_UNITS, ppm: float = 3.0,
                     max_steps: int = 1) -> list[Proposal]:
    """Propose neutral formulas for one unassigned peak m/z by stepping each
    anchor by +/- n*unit (n up to max_steps) and checking the ion m/z under
    each adduct against the target within ppm."""
    out: list[Proposal] = []
    for anchor in anchor_formulas:
        for unit in units:
            for n in range(-max_steps, max_steps + 1):
                if n == 0:
                    continue
                cand = formula_add(anchor, unit, n)
                if cand is None:
                    continue
                m_neu = C.neutral_mass(cand)
                for add in adducts:
                    if add not in C.ADDUCT_SHIFTS:
                        continue
                    theo = m_neu + C.ADDUCT_SHIFTS[add]
                    err = (target_mz - theo) / theo * 1e6
                    if abs(err) <= ppm:
                        out.append(Proposal(
                            target_mz=target_mz, neutral_formula=cand, adduct=add,
                            anchor_formula=anchor, unit=unit, n_steps=n,
                            ppm_error=err))
    # support count: how many anchors sit one unit away (below/above) from cand
    for p in out:
        p.n_supporting_anchors = _support_count(p.neutral_formula, anchor_formulas)
    # best ppm first, then more-supported
    out.sort(key=lambda p: (-(p.n_supporting_anchors), abs(p.ppm_error)))
    return out


def _support_count(candidate: str, anchor_formulas: set[str]) -> int:
    """Number of anchors reachable from candidate by +/-1 of any organic unit
    (a proxy for 'sits in an established homologous series')."""
    n = 0
    for unit in ORGANIC_UNITS:
        for s in (+1, -1):
            nb = formula_add(candidate, unit, s)
            if nb and nb in anchor_formulas:
                n += 1
    return n


def find_homolog_series(formula_mz: dict[str, float], unit: str,
                        min_len: int = 3) -> list[list[str]]:
    """Group formulas into chains spaced by `unit` (each consecutive member is
    +1 unit from the previous). Returns chains of length >= min_len.
    `formula_mz` maps neutral formula -> some sortable value (m/z or mass)."""
    forms = set(formula_mz)
    visited: set[str] = set()
    chains: list[list[str]] = []
    for f in sorted(forms, key=lambda x: formula_mz[x]):
        if f in visited:
            continue
        # only start at chain heads (no -1 neighbour present)
        prev = formula_add(f, unit, -1)
        if prev in forms:
            continue
        chain = [f]
        cur = f
        while True:
            nxt = formula_add(cur, unit, +1)
            if nxt and nxt in forms:
                chain.append(nxt)
                cur = nxt
            else:
                break
        for m in chain:
            visited.add(m)
        if len(chain) >= min_len:
            chains.append(chain)
    return chains
