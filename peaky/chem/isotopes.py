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

from dataclasses import dataclass, field

import pandas as pd

__version__ = "0.3.0"  # + diag_min_rel: claim faint 15N/18O/single-heavy satellites (D_15N, D_30SI)
# delta-m (Da) between an isotopologue satellite and its monoisotopic parent
# (AME2020 exact masses; 81Br-79Br = 80.9162897 - 78.9183376 = 1.9979521 --
# was 1.997795, 0.16 mDa low and inconsistent with passes._DBR; fixed 2026-06-13)
D_13C = 1.003355
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


LABEL_PURITY_15N = 0.98   # isotopic purity of the ^N reagents (NO3_15N / NH4_15N)

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
    # LABELLED reagent nitrogen ('^N' = 15N at the reagent's isotopic purity): the
    # "monoisotopic" line is the HEAVY isotope, so the impurity sits at a NEGATIVE
    # shift (-0.99703, the 14N line). 2 % = the nominal 98 atom % of both the
    # 15N-nitrate and the 15N-ammonium reagents (measured 0.018-0.021 on the
    # 2026-09-10 ^NH4+ file). Lets the envelope-completion sweep CLAIM that line
    # (else it floats free and a CHON [M+H]+ mass-fit grabs it).
    "^N": [(0.0, LABEL_PURITY_15N), (-0.997035, 1.0 - LABEL_PURITY_15N)],
}
_HEAVY_ELEMENTS = ("Br", "Cl", "Si", "S")   # the M+2 drivers


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
        per = _ISO_DIST.get(el)
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
