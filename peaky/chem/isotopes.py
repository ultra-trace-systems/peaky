"""Isotope-pattern prescan -> grid constraints.

This is deliberately NOT a scorer. Mascope (match_compounds) is the authoritative
judge of whether an isotopologue pattern fits a formula. The prescan only looks
at the raw peak list to answer two cheap questions that shrink the candidate
search before we ever call the server:

  1. Which heteroatoms show isotope-pair evidence (Br, Cl, S, Si)? -> only put
     those elements in the grid ranges (huge combinatorial saving).
  2. What is the largest carbon number implied by the brightest 13C satellites?
     -> cap C in the grid.

It walks the deduplicated, intensity-sorted peak list looking for satellite
pairs at characteristic delta-m with plausible intensity ratios.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import NamedTuple

import numpy as np
import pandas as pd

__version__ = "0.5.2"  # + D_13C_EXACT: the one exact 13C spacing the batch checks, the element evidence and the side-lobe rule share
# delta-m (Da) between an isotopologue satellite and its monoisotopic parent
# (AME2020 exact masses; 81Br-79Br = 80.9162897 - 78.9183376 = 1.9979521 --
# was 1.997795, 0.16 mDa low and inconsistent with passes._DBR; fixed 2026-06-13)
D_13C = 1.003355
#: the exact 13C - 12C spacing (AME2020), for tests that read offsets at the
#: 0.1 ppm level; D_13C above is the same spacing rounded to 1 microDa
D_13C_EXACT = 1.0033548378
D_37CL = 1.997050
D_81BR = 1.9979521
D_34S = 1.995796
D_29SI = 0.999568
D_30SI = 1.996840
D_15N = 0.997035
D_18O = 2.004246

# natural-abundance per-atom ratio of the +N satellite to the monoisotopic peak
R_13C_PER_C = 0.0107          # 1.1% per carbon
R_37CL_PER_CL = 0.3196        # 37Cl/35Cl
R_81BR_PER_BR = 0.9728        # 81Br/79Br
R_34S_PER_S = 0.0443          # 34S/32S
R_29SI_PER_SI = 0.0510        # 29Si/28Si
R_30SI_PER_SI = 0.0309        # 30Si/28Si
R_15N_PER_N = 0.003640        # 15N/14N (faint: 0.36% per N)
R_18O_PER_O = 0.002050        # 18O/16O (faint: 0.20% per O)


# ---------------------------------------------------------------------------
# OBSERVABILITY: could a satellite have been SEEN at all?
#
# Several rules in the engine argue from a MISSING satellite -- "a real Br would
# show its 81Br twin, this one does not, so discount it" (passes.core's het-iso
# gate, tiers' Si demote, the reflist rescue's isotope check). That argument is
# only valid when the twin was within reach of the instrument: on a peak too dim
# for its satellite to clear the picker's floor, ABSENCE IS NOT EVIDENCE, and
# charging it anyway turns "we could not have looked" into "we looked and it was
# not there". The predicate below is the shared test; every one of those sites
# asks it before holding a missing satellite against a formula.
#
# Per-atom heavy/light ratio of the line that DIAGNOSES each element -- the one
# a missing-satellite argument is actually about. Si has two (29Si at M+1, 30Si
# at M+2) and either confirms it, so the brighter line decides whether the
# question could have been answered at all. Elements with no
# minor isotope (N, P, I, F) are absent on purpose: nothing about brightness can
# make their count testable, so they never reach this predicate.
DIAG_SATELLITE_RATIO: dict[str, float] = {
    "C": R_13C_PER_C,
    "Cl": R_37CL_PER_CL,
    "Br": R_81BR_PER_BR,
    "S": R_34S_PER_S,
    "Si": max(R_29SI_PER_SI, R_30SI_PER_SI),
}


def satellite_height(element: str, n_atoms, parent_height) -> float:
    """First-order predicted height of `element`'s diagnostic satellite on a
    parent of `parent_height`: per-atom abundance ratio x atom count x parent.
    First-order is the right precision here -- the question is "order of the
    detection floor or not", and the multi-atom combinatorial correction (which
    only ever RAISES the line) never decides it. Returns 0.0 for an element with
    no diagnostic line, a non-positive count, or an unknown/absent height."""
    r = DIAG_SATELLITE_RATIO.get(element)
    if r is None or n_atoms is None or parent_height is None:
        return 0.0
    n, h = float(n_atoms), float(parent_height)
    if not (n > 0) or not (h > 0):     # NaN-safe: a non-finite count/height is no line
        return 0.0
    return h * r * n


def satellite_observable(element: str, n_atoms, parent_height, floor) -> bool:
    """Would `element`'s diagnostic satellite clear `floor` on this parent?

    False means the instrument COULD NOT HAVE SHOWN it, so its absence says
    nothing about the formula. Callers differ on what an unknown parent height
    means and must decide that themselves before calling: a height of 0/None
    reads here as "not observable" (the reflist rescue's reading -- too dim to
    confirm), so a caller that wants the opposite (passes.core: waive a penalty
    only on POSITIVE evidence of unobservability) checks the height first.
    `floor` None = no resolved detection floor -> nothing can be ruled out, so
    True."""
    if floor is None:
        return True
    return satellite_height(element, n_atoms, parent_height) >= float(floor)


# Isotopic purity of the '^' labelled reagents -- the DEFAULT only. The value in
# force for a run is the active reagent profile's `purity` (ReagentProfile.purity,
# published by assign.run via set_label_purity); both consumers read it through
# `label_purity()`: this module's envelope predictor (`isotope_pattern`, the '^N'
# per-atom distribution below) and the local scorer's `predict_isotopes` call
# (io.local_scoring.score_candidates_local). Process-global on purpose: batch runs
# fan out over PROCESSES, so one active reagent per interpreter holds.
LABEL_PURITY_15N = 0.98

_ACTIVE_LABEL_PURITY = LABEL_PURITY_15N


def set_label_purity(purity: float | None) -> float:
    """Publish the active reagent's isotopic purity (None restores the default).
    Returns the value now in force."""
    global _ACTIVE_LABEL_PURITY
    _ACTIVE_LABEL_PURITY = (LABEL_PURITY_15N if purity is None
                            else min(max(float(purity), 0.0), 1.0))
    return _ACTIVE_LABEL_PURITY


def label_purity() -> float:
    """The active labelled-reagent isotopic purity (see set_label_purity)."""
    return _ACTIVE_LABEL_PURITY

# Per-atom isotope distributions: element -> [(mass_shift_from_lightest, abundance)].
# Only isotopes that move the M+1/M+2/... envelope are listed (2H, 17O kept tiny).
# Masses are heavy-minus-light exact deltas; abundances are natural fractions.
_ISO_DIST: dict[str, list[tuple[float, float]]] = {
    "C":  [(0.0, 0.98930), (1.003355, 0.01070)],
    "H":  [(0.0, 0.999885), (1.006277, 0.000115)],
    "N":  [(0.0, 0.996360), (0.997035, 0.003640)],
    "O":  [(0.0, 0.997570), (1.004217, 0.000380), (2.004246, 0.002050)],
    "S":  [(0.0, 0.949900), (0.999388, 0.007500), (1.995796, 0.042500)],
    "Cl": [(0.0, 0.757600), (1.997050, 0.242400)],
    "Br": [(0.0, 0.506900), (1.9979521, 0.493100)],
    "Si": [(0.0, 0.922230), (0.999568, 0.046850), (1.996840, 0.030920)],
    # NB no '^N' entry -- a labelled element's distribution depends on the REAGENT
    # BOTTLE, not on nature, so it is built per lookup in _per_atom() from the
    # active purity rather than frozen into this natural-abundance table.
}
_HEAVY_ELEMENTS = ("Br", "Cl", "Si", "S")   # the M+2 drivers


def _per_atom(el: str) -> list[tuple[float, float]] | None:
    """Per-atom isotope distribution of an element, or None when the element does
    not move the envelope. A caret element ('^N' = 15N) is LABELLED: its
    "monoisotopic" line is the HEAVY isotope, so the reagent's unlabelled impurity
    sits at a NEGATIVE shift (-0.99703, the 14N line) at 1 - purity. Built from the
    ACTIVE purity (label_purity()) so a profile declaring a different bottle moves
    the line. Claiming it matters: left unpredicted it floats free and a CHON
    [M+H]+ mass-fit grabs it."""
    if el.startswith("^"):
        if el != "^N":
            return None
        p = label_purity()
        return [(0.0, p), (-0.997035, 1.0 - p)]
    return _ISO_DIST.get(el)


# (mass shift, label, {element: min count required to form it})
_LABEL_TABLE = [
    (-0.997035, "14N", {"^N": 1}),      # labelled-reagent impurity line (BELOW M0)
    (1.003355, "13C", {"C": 1}),
    (0.999568, "29Si", {"Si": 1}),
    (0.997035, "15N", {"N": 1}),
    (1.9979521, "81Br", {"Br": 1}),
    (1.997050, "37Cl", {"Cl": 1}),
    (1.996840, "30Si", {"Si": 1}),
    (1.995796, "34S", {"S": 1}),
    (2.004246, "18O", {"O": 1}),
    (2.006710, "13C2", {"C": 2}),
    (2.997520, "81Br+29Si", {"Br": 1, "Si": 1}),
    (3.001307, "81Br+13C", {"Br": 1, "C": 1}),
    (3.994792, "81Br+30Si", {"Br": 1, "Si": 1}),
    (3.995904, "2x81Br", {"Br": 2}),
    (3.994100, "2x37Cl", {"Cl": 2}),
    (3.993680, "2x30Si", {"Si": 2}),
]


def _label_for_shift(dmass: float, counts: dict | None = None) -> str:
    """Name the dominant isotopologue at a mass shift, restricted to combos the
    formula can actually form (a 1-Br ion's M+4 is 81Br+30Si, never 81Br2).
    Falls back to a generic M+N when nothing achievable is close."""
    cand = [t for t in _LABEL_TABLE
            if counts is None or all(counts.get(e, 0) >= n for e, n in t[2].items())]
    if not cand:
        return f"M+{round(dmass)}"
    best = min(cand, key=lambda t: abs(t[0] - dmass))
    return best[1] if abs(best[0] - dmass) <= 0.012 else f"M+{round(dmass)}"


def isotope_pattern(ion_formula: str, *, min_rel: float = 0.03,
                    max_shift: float = 6.5, merge_da: float = 0.006,
                    diag_min_rel: float | None = None
                    ) -> list[tuple[float, float, str]]:
    """Predict the isotopologue envelope of an ION formula.

    Returns [(delta_mass, rel_intensity, label), ...] for every resolved line
    ABOVE min_rel relative to the monoisotopic (M0) line, sorted by mass shift.
    The pattern is the convolution of each element's per-atom distribution; lines
    within ~3 mDa are merged keeping the intensity-weighted exact mass. This is
    what lets the envelope-completion pass recognise an unexplained peak as the
    M+2/M+4 satellite of a committed parent (the silanediol Si4+Br case, where
    the M+4/M+2 ratio of ~0.26 otherwise mimics a Cl doublet).

    diag_min_rel (when set below min_rel) keeps the FAINT single-heteroatom
    diagnostic M+1/M+2 lines -- 15N (0.36%/N), 18O, a single 34S/29Si/30Si -- that
    the plain min_rel floor prunes. A single-N/O/S/Si analyte's diagnostic
    satellite sits below any sane plausibility floor, yet it must still be CLAIMED
    (else it floats free as a base peak a mass-coincidence phantom grabs -- the
    15N-satellite-of-a-CHON leak). The intensity-consistency gate at the call site,
    not min_rel, is the discriminator against a coincidental neighbour."""
    from peaky.chem import chemistry as C
    counts = C.parse_formula(ion_formula)
    # distribution as {rounded_shift: [prob, weighted_mass_sum]}
    dist: dict[int, list[float]] = {0: [1.0, 0.0]}
    # prune hard enough to bound the convolution but soft enough that the
    # multi-atom cross terms survive (a 4-Si M+4 is a sum of many ~1e-3 paths --
    # 81Br.30Si, 81Br.29Si2, 2x30Si... -- so an aggressive prune underpredicts
    # the M+4 height and the silanediol M+4 wrongly survives as a contaminant)
    PRUNE = 1e-6
    for el, n in counts.items():
        per = _per_atom(el)
        if per is None or n <= 0:
            continue
        for _ in range(n):
            nxt: dict[int, list[float]] = {}
            for k, (p, wm) in dist.items():
                base_m = wm / p if p else 0.0
                for dm, ab in per:
                    if ab <= 0:
                        continue
                    np_ = p * ab
                    if np_ < PRUNE:
                        continue
                    newm = base_m + dm
                    key = int(round(newm * 1000))
                    slot = nxt.setdefault(key, [0.0, 0.0])
                    slot[0] += np_
                    slot[1] += np_ * newm
            # renormalise pruning loss negligibly; keep top lines only
            dist = nxt
    m0 = dist.get(0)
    if not m0 or m0[0] <= 0:
        # monoisotopic not the lightest key (rounding); take the smallest shift
        k0 = min(dist)
        m0 = dist[k0]
    base = m0[0]
    # raw lines (mass, prob) above a loose floor, sorted by mass
    raw = []
    for k, (p, wm) in sorted(dist.items()):
        if k == 0:
            continue
        dmass = wm / p
        # negative shifts exist only for a labelled element (the ^N 14N line)
        if abs(dmass) <= 0.4 or dmass > max_shift or dmass < -1.5:
            continue
        raw.append([dmass, p])
    # merge lines closer than merge_da -- the peak picker resolves them as ONE
    # peak, so their intensities add (e.g. 81Br at +1.9978 + 30Si at +1.9968).
    merged: list[list[float]] = []
    for dmass, p in raw:
        if merged and dmass - (merged[-1][1] / merged[-1][0]) < merge_da:
            merged[-1][0] += p
            merged[-1][1] += p * dmass
        else:
            merged.append([p, p * dmass])
    # single-heteroatom diagnostic lines whose floor diag_min_rel can lower
    _DIAG_LABELS = ("13C", "14N", "15N", "29Si", "30Si", "34S", "18O", "37Cl", "81Br")
    out = []
    for p, wm in merged:
        dmass = wm / p
        rel = p / base
        label = _label_for_shift(dmass, counts)
        floor = min_rel
        if diag_min_rel is not None and label in _DIAG_LABELS:
            floor = min(floor, diag_min_rel)
        if rel >= floor:
            out.append((round(dmass, 4), round(rel, 4), label))
    return sorted(out)


@dataclass
class PrescanResult:
    has_Br: bool = False
    has_Cl: bool = False
    has_S: bool = False
    has_Si: bool = False
    has_multi_Br: bool = False          # Br2 triplet seen
    estimated_max_C: int = 0
    evidence: list = field(default_factory=list)   # human-readable hits

    def as_dict(self) -> dict:
        return {
            "has_Br": self.has_Br, "has_Cl": self.has_Cl, "has_S": self.has_S,
            "has_Si": self.has_Si, "has_multi_Br": self.has_multi_Br,
            "estimated_max_C": self.estimated_max_C, "n_evidence": len(self.evidence),
        }


def _find_partner(mz_sorted, height_by_mz, target_mz, ppm_tol):
    """Return (mz, height) of a peak near target_mz within ppm_tol, else None."""
    tol = target_mz * ppm_tol * 1e-6
    lo, hi = target_mz - tol, target_mz + tol
    import bisect
    i = bisect.bisect_left(mz_sorted, lo)
    best = None
    while i < len(mz_sorted) and mz_sorted[i] <= hi:
        m = mz_sorted[i]
        h = height_by_mz[m]
        if best is None or h > best[1]:
            best = (m, h)
        i += 1
    return best


def prescan(peaks: pd.DataFrame, *, mz_col="mz", height_col="height",
            ppm_tol=8.0, min_height=0.0,
            reagent_mzs: list[float] | None = None,
            reagent_ppm=15.0) -> PrescanResult:
    """Scan a peak table for isotope-pair signatures.

    `reagent_mzs` lets the caller strip known reagent-cluster peaks (e.g. bare
    Br_n clusters) before the Br scan, so they don't masquerade as analyte Br.
    """
    res = PrescanResult()
    df = peaks[[mz_col, height_col]].dropna().copy()
    df = df[df[height_col] >= min_height]
    if reagent_mzs:
        keep = []
        for mz in df[mz_col]:
            is_reagent = any(abs(mz - r) / r * 1e6 <= reagent_ppm for r in reagent_mzs)
            keep.append(not is_reagent)
        df = df[keep]
    df = df.sort_values(height_col, ascending=False)
    mz_sorted = sorted(df[mz_col].tolist())
    height_by_mz = dict(zip(df[mz_col], df[height_col]))

    n_scan = min(len(df), 400)   # brightest peaks carry the isotope information
    for parent_mz, parent_h in zip(df[mz_col].head(n_scan), df[height_col].head(n_scan)):
        if parent_h <= 0:
            continue
        # --- 13C: estimate carbon count from the +1.00336 satellite ratio ---
        p = _find_partner(mz_sorted, height_by_mz, parent_mz + D_13C, ppm_tol)
        if p:
            ratio = p[1] / parent_h
            if 0.003 <= ratio <= 0.9:
                n_c = round(ratio / R_13C_PER_C)
                if n_c > res.estimated_max_C:
                    res.estimated_max_C = int(n_c)
        # --- 81Br: +1.99795. One Br -> M+2/M ~ 1.0; Br2 (1:2:1) -> M+2/M ~ 2.0
        #     with an M+4 at ~1.0*M. Either pattern confirms Br. ---
        p = _find_partner(mz_sorted, height_by_mz, parent_mz + D_81BR, ppm_tol)
        if p:
            ratio = p[1] / parent_h
            if 0.6 <= ratio <= 1.4:        # single Br
                res.has_Br = True
                res.evidence.append(("Br", round(parent_mz, 4), round(ratio, 2)))
                continue
            if 1.5 <= ratio <= 2.6:        # Br2: check M+4 ~ 1.0*M
                p2 = _find_partner(mz_sorted, height_by_mz, parent_mz + 2 * D_81BR, ppm_tol)
                if p2 and 0.6 <= p2[1] / parent_h <= 1.4:
                    res.has_Br = True
                    res.has_multi_Br = True
                    res.evidence.append(("Br2", round(parent_mz, 4), round(ratio, 2)))
                    continue
        # --- 37Cl: +1.99705, ratio ~0.32 (one Cl) ---
        p = _find_partner(mz_sorted, height_by_mz, parent_mz + D_37CL, ppm_tol)
        if p:
            ratio = p[1] / parent_h
            if 0.22 <= ratio <= 0.45:
                res.has_Cl = True
                res.evidence.append(("Cl", round(parent_mz, 4), round(ratio, 2)))
        # --- 34S: +1.9958, ratio ~0.045 per S ---
        p = _find_partner(mz_sorted, height_by_mz, parent_mz + D_34S, ppm_tol)
        if p:
            ratio = p[1] / parent_h
            if 0.025 <= ratio <= 0.09:
                res.has_S = True
                res.evidence.append(("S", round(parent_mz, 4), round(ratio, 2)))
        # --- 29Si: +0.99957, ratio ~0.05 per Si ---
        p = _find_partner(mz_sorted, height_by_mz, parent_mz + D_29SI, ppm_tol)
        if p:
            ratio = p[1] / parent_h
            if 0.035 <= ratio <= 0.08:
                res.has_Si = True
                res.evidence.append(("Si", round(parent_mz, 4), round(ratio, 2)))
    return res


def constrain_ranges(base_ranges: dict[str, tuple[int, int]],
                     pre: PrescanResult,
                     context_caps: dict[str, int]) -> dict[str, tuple[int, int]]:
    """Apply prescan evidence to grid ranges:
      * cap C at estimated_max_C (+ small headroom) when we have an estimate
      * zero out Br/Cl/S/Si that show NO spectral evidence (subject to context)
    `context_caps` gives the per-element max the context allows."""
    r = dict(base_ranges)
    if pre.estimated_max_C and "C" in r:
        cmax = min(r["C"][1], pre.estimated_max_C + 4)
        r["C"] = (r["C"][0], max(cmax, r["C"][0]))
    for el, flag in (("Br", pre.has_Br), ("Cl", pre.has_Cl),
                     ("S", pre.has_S), ("Si", pre.has_Si)):
        cap = context_caps.get(el, 0)
        if not flag or cap <= 0:
            r[el] = (0, 0)
        else:
            r[el] = (0, min(r.get(el, (0, cap))[1] or cap, cap))
    return r


# ===========================================================================
# ISOTOPE CHILDREN JUDGED AGAINST THE COMMITTED LINE (C11+c)
#
# An isotope child of a committed M0 row (ledger role `iso_child`) says "this
# line is the parent's isotopologue". The per-file isotope facts
# (assignment/evidence.py `_measure`, read by the merge vote's class) read it as
# evidence only where the line sits at its label's exact spacing and is as tall
# as the ion's composition makes it. (The evidence scale probes the lines itself,
# assignment/levels/lines.py.) Three
# things the ledger does not record decide both:
#
#   * the COMMITTED line. The scorer commits an ion's most abundant
#     isotopologue (a Br2 ion on its 79Br81Br line, a Cl4 ion on a 37Cl line),
#     so the parent's heavy configuration is read off its m/z against the ion's
#     monoisotopic m/z (`committed_configuration`);
#   * the label CONVENTION. Scorer labels count heavy atoms from the MONO line
#     ('81Br2', 'M0' = the mono line below a heavy parent, '13C+81Br'); peaky's
#     own labels count them from the PARENT line ('2x81Br', '81Br(pair)',
#     '81Br+13C', 'M+5'). Nothing records the producer, so a child is read both
#     ways: the reading nearer its measured shift gives the expectation
#     (`resolve_child`), and either reading may place it;
#   * the POSITION. A line counts only within max(1 ppm, 4 sigma(h)) of its
#     label's exact spacing, sigma(h)^2 = a^2 + b^2 / h fitted on the source's
#     own '13C' children (`fit_position_sigma`), rescued by the calibrated
#     parent position (pcal) or a brighter neighbour's pull (N1)
#     (`judge_source`).
#
# The expectation of a line is its configuration's probability relative to the
# committed line's, multinomial per element over the ion's atoms with the
# evidence band's per-atom heavy / light ratios (`heavy_probability`).
#
# scripts/level_ledger.py carries a standalone twin of everything below (the
# script imports no peaky code); tests/test_isotope_children.py pins the two.
# ===========================================================================

#: exact mass (NIST / AME2020) of the heavy isotope a child label can name and
#: the element it replaces; the light isotope's mass is chemistry.M's
HEAVY_ISOTOPES: dict[str, tuple[str, float]] = {
    "13C": ("C", 13.0033548378), "15N": ("N", 15.0001088984), "17O": ("O", 16.9991317565),
    "18O": ("O", 17.9991596129), "33S": ("S", 32.9714589098), "34S": ("S", 33.967867004),
    "37Cl": ("Cl", 36.965902602), "81Br": ("Br", 80.9162906), "29Si": ("Si", 28.9764946649),
    "30Si": ("Si", 29.973770136), "2H": ("H", 2.0141017781),
}


def _element_masses() -> dict[str, float]:
    from peaky.chem import chemistry as C
    return {**C.M, "Na": 22.989769282, "K": 38.9637064864}


#: monoisotopic element masses an ion's mono m/z is summed from (chemistry.M
#: plus the two alkali adduct metals)
ELEMENT_MASS: dict[str, float] = _element_masses()
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


def nominal_cluster(counts: dict, nominal: int = 2) -> tuple[float, float]:
    """The ion's lines at one NOMINAL heavy shift, read as one: (their summed
    probability relative to the all-light line, their probability-weighted
    shift). What a peak wider than the cluster's fine structure shows -- a
    TOF's M+2 is 81Br, 37Cl, 34S, 30Si, 18O, 13C2 and 13C15N within ~11 mDa,
    one line at R ~10 000. (0.0, nan) when the ion forms no line there."""
    lines = _ion_lines(heavy_key({k: v for k, v in counts.items() if v}), max_nominal=int(nominal))
    sel = [(s, p) for k, s, p in lines if k and _heavy_nominal(k) == int(nominal)]
    tot = float(sum(p for _s, p in sel))
    if tot <= 0:
        return 0.0, float("nan")
    return tot, float(sum(s * p for s, p in sel) / tot)


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
                `iso_labels` lists)
      heavy     the line's own heavy configuration under the nearer reading
                ({} for a pure 'M0' line: the mono line); None for 'M+n' or an
                unreadable label (no configuration)"""
    raw = split_label(label)
    parsed = [parse_label_part(p) for p in raw]
    parts = [p for p, (k, _v) in zip(raw, parsed) if k != "mono"]
    kinds = {k for k, _v in parsed}
    out = dict(kind="bad", shift=float("nan"), readings=(), expected=0.0, elements=[], parts=parts, heavy=None)
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
                expected=expected, elements=elements, parts=parts, heavy=dict(best["heavy"]))


def reagent_part(part: str, satellite: str | None) -> bool:
    """A label part naming the reagent halogen's heavy line alone ('81Br',
    '81Br2', '2x81Br', '81Br(pair)' on a bromide batch); a mixed or alternative
    part ('81Br37Cl(pair)', '81Br/37Cl(pair)') is not."""
    if not satellite:
        return False
    kind, v = parse_label_part(part)
    return kind == "set" and set(v) == {satellite}


def most_probable_heavy(n: int, iso: str) -> int:
    """k_c(n): how many `iso` atoms the most probable line of an ion carrying
    n atoms of its element holds (per-atom ISOTOPE_RATIO) -- the line a scorer
    commits: 81Br 0, 1, 1, 2 for Br1-Br4; 37Cl 0, 0, 0, 1 for Cl1-Cl4."""
    n = max(int(n), 0)
    r = ISOTOPE_RATIO[iso]
    return max(range(n + 1), key=lambda k: math.comb(n, k) * r ** k)


def full_count_line(v: dict, satellite: str | None, own: int) -> bool:
    """D4's last sub-point, read as a POSITION rule (2026-09-30): a kept,
    IN-BAND line (`v` a judge_source verdict) that only the ion's full count
    of the reagent halogen makes -- evidence of the NEUTRAL's halogen. The ion
    carries n atoms of it, the adduct s = n - `own` (the neutral's atoms); the
    line's heavy index relative to the ACTUAL committed line, j = k - k_P (k
    its heavy atoms, k_P the committed configuration's: `committed` on the
    verdict, read from the parent m/z), lies outside [-k_c(s), s - k_c(s)]
    (k_c `most_probable_heavy`) -- the lines an s-atom ion committed on its
    most probable line makes. A line that changes no atom of the halogen
    (j = 0: a 13C or 34S line of any parent) never counts. A line where an
    s-atom ion puts one stays the reagent's whatever its height; an 'M+n' or
    unreadable line has no index; an adduct that supplies none of the halogen
    (s <= 0) leaves nothing to tell apart, nor does a neutral that carries
    none (`own` <= 0: every line of its ion is the reagent's)."""
    if not satellite or not v["ok"] or v.get("heavy") is None:
        return False
    n = int(v["counts"].get(ISOTOPE_ELEMENT[satellite], 0))
    s = n - int(own)
    if s <= 0 or int(own) <= 0:
        return False
    j = int(v["heavy"].get(satellite, 0)) - int(v["committed"].get(satellite, 0))
    kc = most_probable_heavy(s, satellite)
    return not (-kc <= j <= s - kc)


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
    parts, kind, heavy (the line's configuration), committed (the parent's
    committed configuration), counts (the ion's)],
    'lists': [one bool per list row: an entry that is not this
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
                             ratio=ratio, elements=r["elements"], parts=r["parts"], kind=r["kind"],
                             heavy=r["heavy"], committed=hp, counts=p["counts"]))
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


def _reagent_line(v: dict, satellite: str) -> bool:
    if v["parts"]:
        return all(reagent_part(pt, satellite) for pt in v["parts"])
    return v["elements"] == [ISOTOPE_ELEMENT[satellite]]


def line_facts(verdicts, satellite: str | None, own: int = 0) -> dict:
    """The facts one pair's KEPT lines give (`verdicts` = its placed children):
    `iso` an in-band line; `lined` the elements of its in-band lines (C17's
    multiline reads those the neutral supplies); `carbon` a line that adds
    13C; `labels` the lines' labels without 'M0' parts; `reagent_only` every
    line naming a heavy atom names only the reagent halogen's (by its label
    parts; a pure 'M0' line by what it differs in from the committed line, so
    the lighter line of a heavy-committed halogen pattern names the halogen
    and the 'M0' child of a mono-committed parent names nothing), none adds
    13C, and none is a line only the ion's full halogen count makes
    (`full_count_line`; `own` = the neutral's atoms of the halogen)."""
    lines = list(verdicts)
    ok = [v for v in lines if v["ok"]]
    naming = [v for v in lines if v["parts"] or (v.get("kind") == "mono" and v["elements"])]
    carbon = any("C" in v["elements"] for v in lines)
    return dict(
        iso=bool(ok),
        lined=set().union(*(set(v["elements"]) for v in ok)) if ok else set(),
        carbon=carbon,
        labels={"+".join(v["parts"]) for v in lines if v["parts"]},
        reagent_only=bool(satellite) and bool(naming) and not carbon
        and all(_reagent_line(v, satellite) for v in naming)
        and not any(full_count_line(v, satellite, own) for v in lines),
    )
