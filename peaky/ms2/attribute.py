"""Which window ion made each MS2 fragment: composition, then the intact test.

Pure, on the readings of ONE parent window across its energies
(`readings.read_window`): no Mascope, no I/O. `subformula` answers which window
ions CAN make a peak (`n_parents`); this module decides, per fragment, which of
them did, as far as the spectra say -- and no further. No least-squares
deconvolution: with two or three window ions and five or six energies a fit
has as many unknowns as points, and it would hand back a number for a
fragment the data cannot attribute.

  * A fragment is one physical peak across the window's energies
    (`merge_fragment_peaks`: linked by calibrated m/z within the reading
    tolerance, or by a shared rank-1 formula, never two peaks of one
    spectrum more than the tolerance apart), named by every formula read
    there. Where it sits is its signal-weighted m/z (`fragment_mz`: each
    peak weighted by the share of its spectrum's TIC it holds); its
    candidates are the window ions that read it as a formula within the
    reading tolerance of that m/z (`fragment_compositions`,
    `candidates_across`) -- a centroid of a few counts neither makes nor
    vetoes a candidate -- and an organic reading outranks a carbon-poor
    polyoxy one (`polyoxy_inorganic`: the C2F3 = H2^NO4 alias of 15N data),
    when the organic reader is in the sample.
  * A fragment every peak of which sits a few line widths beside a peak
    `SATELLITE_RATIO` times brighter is a side lobe of that line, not an ion
    (`satellite_fragments`): set aside, unexplained.
  * A named ion the sample lacks (`absent_ions`: its precursor in no MS2
    spectrum of the window, an MS1 share of 0) decides nothing: it weighs 0,
    is set aside before the intact test and the guard, and takes nothing --
    a fragment only absent ions can make is no ion's ('absent only').
  * One candidate (or one the sample holds): the fragment is that ion's
    ('composition').
  * Several: the INTACT TEST on TIC-normalised fractions. A candidate whose own
    fraction at the fragment's onset energy is still >= `INTACT_FRACTION` of
    its fraction at the window's lowest energy has lost nothing while the
    fragment appeared, and is excluded -- unless the fragment is already there
    at the lowest energy (an in-source ion, or one formed below the lowest
    step), where an onset says nothing. The onset the test reads is the
    first energy at which EVERY candidate reads the fragment
    (`shared_presence`) when what some of them read before it is a stray
    (<= `ONSET_STRAY_SHARE` of the fragment's TIC fraction there): a weak
    peak that only some of them can read sits where the others cannot make
    it, so it is not yet the fragment they share; a larger one is the
    fragment, and its own onset stands. One
    candidate left: called ('intact
    test') -- unless the MASS-BALANCE GUARD withdraws the call: the fragment's
    TIC-fraction gain at some energy exceeds everything the called ion held
    at the lowest energy (x `BRANCHING_FLAG`) -- its precursor plus the
    fragments only it can make by composition (`held_at_lowest`: an adduct
    that is mostly in-source products at the lowest step still carries what
    those products go on to make) -- so most of it must come from a
    candidate the onset excluded. Else 'ambiguous', split by share weights,
    chosen per candidate (`share_weights`): its intensity at the lowest energy
    (its MS2 survivor); a candidate with none, its share of the MS1 isolation
    window put on the survivors' scale; 0 for a named ion in neither (an MS1
    share of 0 is measured, not missing) -- so naming an ion the sample lacks
    moves no other candidate's weight. Excluded candidates weigh 0.

    Why the guard: in the 2026-09-10 file's 179.107 window C9H11+ (119.085)
    appears at NCE 5, where C11H15O2+ is still 98.7 % of its NCE 1 fraction
    and the 0.55 % co-isolate C11H16^NO+ lost 1132 counts -- the fragment
    holds 1118. The onset is the co-isolate's; but by NCE 45 C9H11+ holds
    39k counts, 17x everything that co-isolate ever held, and is C11H15O2+'s
    (- C2H4O2) from NCE 10 on. One call per fragment cannot be both.
  * Branching of X into a fragment = the fragment's weighted TIC-fraction gain
    over X's TIC-fraction loss, from the lowest energy to each energy. Summed
    over X's fragments it cannot exceed 1 (a precursor cannot give more than it
    lost); a sum above `BRANCHING_FLAG` is flagged, never clipped -- and so is
    any growth of X's products at an energy where X lost nothing or gained,
    where the ratio is NaN (`branching_breach`).
  * Every one of these reads OBSERVED peaks: a product outside the observed
    m/z range (below the scan's low-mass limit) leaves the TIC, which falls
    and inflates every fraction that stays -- a co-isolate that fragments can
    then hold its fraction and read intact, and branching sums exceed 1.
"""
from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from peaky.chem import chemistry as C
from peaky.chem import sidelobes as SL
from peaky.ms2 import subformula as SF

__all__ = [
    "INTACT_FRACTION", "BRANCHING_FLAG", "ONSET_STRAY_SHARE", "SATELLITE_RATIO",
    "SATELLITE_FWHM", "BASIS_COMPOSITION", "BASIS_INTACT",
    "BASIS_AMBIGUOUS", "BASIS_ABSENT", "WEIGHT_SOURCES", "peak_roles", "merge_fragment_peaks", "fragment_key",
    "satellite_fragments", "set_aside_fragments",
    "fragment_mz", "fragment_compositions", "candidates_across", "shared_presence",
    "intensity_matrices", "absent_ions", "live_candidates",
    "share_weights", "intact_test", "held_at_lowest", "attribute_fragments", "branching",
    "branching_breach", "format_weights", "POLYOXY_MAX_C", "POLYOXY_MIN_O", "polyoxy_inorganic",
]

#: A candidate still at >= this share of its lowest-energy TIC fraction where
#: the fragment is present has not fragmented. Not a fit: a margin over the
#: scatter of an ion that does not fragment. In the 2026-09-10 exploratory file
#: the C7H6NS+ precursor (136.02 window) reads 82.2 % of its spectrum at NCE 1
#: and 83.9 % at NCE 11 -- an ion that has not started to fragment wanders
#: about +-2 % between steps.
INTACT_FRACTION = 0.98
#: A branching sum above this is flagged: the products of X grew by more than X
#: lost. 5 % over 1 is room for the averaged spectra's noise; a sum beyond it
#: says a fragment is mis-attributed or a weight is wrong.
BRANCHING_FLAG = 1.05
#: The intact test moves a fragment's onset to the first energy at which every
#: candidate reads it only when what some of them read before it is a stray:
#: at most this share of the fragment's TIC fraction at that energy, at every
#: earlier energy. A larger presence is the fragment itself, placed by a
#: sub-ppm jitter where only some candidates read it, and its own onset stands
#: (at the lowest energy: in-source, no test). The synthetic stray centroid of
#: the docs holds 1.7 % of the fragment's fraction at the shared onset; an
#: in-source fragment read by one candidate at the lowest energy, 100 %.
ONSET_STRAY_SHARE = 0.05
#: A fragment every peak of which sits within `SATELLITE_FWHM` line widths of a
#: peak at least `SATELLITE_RATIO` times brighter in the same spectrum is a side
#: lobe of that line (an Orbitrap processing artifact), not an ion: the MS1
#: side-lobe guard's ratio and band (`chem.sidelobes`), the width being the
#: brighter line's m/z / its reported resolution.
SATELLITE_RATIO = SL.SIDELOBE_RATIO
SATELLITE_FWHM = SL.SIDELOBE_BAND_FWHM[1]

BASIS_COMPOSITION = "composition"
BASIS_INTACT = "intact test"
BASIS_AMBIGUOUS = "ambiguous"
#: A fragment only named ions the sample lacks (`absent_ions`) can make: no
#: window ion takes it (call '', every weight 0), so an absent ion never shows
#: a breakdown curve built from a peak the ions the sample holds cannot make.
BASIS_ABSENT = "absent only"
#: Where a fragment's weights came from: one candidate, the intact test, the
#: candidates' lowest-energy MS2 survivors, those survivors with the MS1 window
#: shares of the candidates that have none put on their scale, the MS1 window
#: shares alone (no candidate has a survivor), an equal split (a candidate
#: with no survivor and no known MS1 share), or none (`BASIS_ABSENT`) -- said,
#: not hidden.
WEIGHT_SOURCES = ("composition", "intact test", "ms2 survivors", "ms2 survivors + ms1 window",
                  "ms1 window", "equal", "none")


def fragment_key(formulas: Sequence[str]) -> str:
    """The fragment's name: its formula when every window ion that reads the
    peak reads it as the same formula (the usual case), else the formulas,
    sorted and joined by ' | '."""
    return " | ".join(sorted(set(formulas)))


def peak_roles(readings: pd.DataFrame, order: Sequence[str] | None = None) -> pd.DataFrame:
    """One row per (spectrum, peak) of a window's readings, with its role.

    'precursor' (`ion` = the window ion it is the intact precursor of: its
    precursor reading wins over any fragment reading of the same peak;
    `candidates` = every window ion whose rank-1 precursor reading it is --
    more than one only when two window ions lie within the reading tolerance
    of the peak, and then `ion` is the first of them), 'fragment' (`key` =
    `fragment_key` of the rank-1 formulas, `candidates` = the window ions that
    read it, `formulas` / `losses` = each candidate's rank-1 formula / loss, as
    'ion: formula' / 'ion: loss' in the same order), or 'unexplained'.
    `order` is the window order (target first, then the co-isolates);
    without it, the order in which the window ions first appear in
    `readings`. The key is THIS spectrum's: `merge_fragment_peaks`
    gives one physical fragment one name across the window's spectra."""
    cols = {"spectrum": [], "nce": [], "hcd_v": [], "peak": [], "mz": [], "mz_cal": [],
            "intensity": [], "role": [], "ion": [], "key": [], "candidates": [],
            "formulas": [], "losses": []}
    if len(readings):
        seen = list(dict.fromkeys(readings.loc[readings["parent"] != "", "parent"]))
        order = list(dict.fromkeys([*(order or []), *seen]))
        pos = {p: i for i, p in enumerate(order)}
        for (spec, peak), g in readings.groupby(["spectrum", "peak"], sort=False):
            r0 = g.iloc[0]
            best = g[g["rank"] == 1]
            best = best.sort_values("parent", key=lambda s: s.map(pos), kind="stable")
            prec = best[best["is_precursor"]]
            role, ion, key, cands, forms, losses = "unexplained", "", "", "", "", ""
            if len(prec):
                role, ion = "precursor", str(prec["parent"].iloc[0])
                cands = ";".join(dict.fromkeys(prec["parent"]))
            elif len(best):
                role = "fragment"
                key = fragment_key(best["formula"].tolist())
                cands = ";".join(best["parent"])
                forms = "; ".join(f"{p}: {f}" for p, f in zip(best["parent"], best["formula"]))
                losses = "; ".join(f"{p}: {n}" for p, n in zip(best["parent"], best["loss_name"]))
            for c, v in (("spectrum", spec), ("nce", r0["nce"]), ("hcd_v", r0["hcd_v"]),
                         ("peak", int(peak)), ("mz", r0["mz"]), ("mz_cal", r0["mz_cal"]),
                         ("intensity", r0["intensity"]), ("role", role), ("ion", ion),
                         ("key", key), ("candidates", cands), ("formulas", forms),
                         ("losses", losses)):
                cols[c].append(v)
    return pd.DataFrame(cols).astype({"spectrum": "str", "nce": "float64", "hcd_v": "float64",
                                      "peak": "int64", "mz": "float64", "mz_cal": "float64",
                                      "intensity": "float64", "role": "str", "ion": "str",
                                      "key": "str", "candidates": "str", "formulas": "str",
                                      "losses": "str"})


def merge_fragment_peaks(roles: pd.DataFrame, *, tol_ppm: float, tol_mda: float
                         ) -> pd.DataFrame:
    """One name per physical fragment across a window's spectra.

    `peak_roles` keys a fragment peak by THAT spectrum's rank-1 formulas, and
    the rank-1 formula of one peak can change between energies: where two
    compositions lie within the reading tolerance of each other (the
    N + ^N / C + O + H alias, 0.44 mDa, at the edge of 5 ppm near m/z 85;
    Cl / C2H-4^N, 0.04 mDa) a sub-ppm jitter of the peak brings the second
    one in and the ranking may prefer it. Keyed per spectrum, one fragment
    became two rows -- one with a late onset, neither with its whole
    intensity. Here the fragment peaks of the window (every spectrum) are
    linked when they share a rank-1 formula or when their calibrated m/z lie
    within the reading tolerance of each other (single linkage along the m/z
    axis), and each linked group is named by `fragment_key` of every rank-1
    formula read in it -- a name with ' | ' says the peak read as more than
    one composition (two window ions, or two energies). Two peaks of one
    spectrum in a group are one fragment, summed (`intensity_matrices`).

    Never two RESOLVED peaks: two peaks of one spectrum more than the reading
    tolerance apart are two ions, and no chain of links makes them one group
    (a cannot-link; the links are tried shortest first, the formula links
    before the m/z ones). A peak that reaches two groups which cannot be one
    -- a weak centroid between two resolved fragments, within the tolerance
    of both -- joins neither link: its position cannot say whose it is, and
    it keeps its own name (read as one formula only, it is that formula's
    fragment; read as two, a fragment of its own), so a few counts never
    merge two fragments.
    Every other row and column is returned as it is."""
    out = roles.copy()
    fr = list(out.index[out["role"] == "fragment"])
    if not fr:
        return out
    mz = {i: float(out.at[i, "mz_cal"]) for i in fr}
    spec = {i: str(out.at[i, "spectrum"]) for i in fr}
    keys = {i: str(out.at[i, "key"]).split(" | ") for i in fr}

    def tol(a: float, b: float) -> float:
        return SF._tolerance_da(max(a, b), tol_ppm, tol_mda)

    def gap(i: int, j: int) -> float:
        d = abs(mz[i] - mz[j])
        return d if math.isfinite(d) else math.inf

    def links(skip: set[int]) -> list[tuple[int, int]]:
        """The candidate links among the peaks not in `skip`: a shared rank-1
        formula (neighbours in m/z among the peaks that read it), then m/z
        within the tolerance (every such pair); each kind shortest first."""
        by_formula: dict[str, list[int]] = {}
        for i in fr:
            if i not in skip:
                for f in keys[i]:
                    by_formula.setdefault(f, []).append(i)
        f_links = set()
        for members in by_formula.values():
            ms = sorted(members, key=lambda i: (mz[i] if math.isfinite(mz[i]) else math.inf, i))
            f_links.update((min(a, b), max(a, b)) for a, b in zip(ms[:-1], ms[1:]))
        finite = sorted((i for i in fr if i not in skip and math.isfinite(mz[i])),
                        key=lambda i: (mz[i], i))
        m_links = set()
        for k, a in enumerate(finite):
            for b in finite[k + 1:]:
                if mz[b] - mz[a] > tol(mz[a], mz[b]):
                    break
                m_links.add((min(a, b), max(a, b)))
        return (sorted(f_links, key=lambda e: (gap(*e), e))
                + sorted(m_links - f_links, key=lambda e: (gap(*e), e)))

    def clash(a: dict, b: dict) -> bool:
        """Two groups' peaks, {spectrum: (lowest, highest m/z)}: does one
        spectrum hold a peak of each more than the tolerance apart?"""
        for s_, (lo_a, hi_a) in a.items():
            if s_ in b:
                lo_b, hi_b = b[s_]
                if max(hi_a - lo_b, hi_b - lo_a) > tol(max(hi_a, hi_b), 0.0):
                    return True
        return False

    def span(i: int) -> dict:
        return {spec[i]: (mz[i], mz[i])} if math.isfinite(mz[i]) else {}

    alone: set[int] = set()
    while True:
        parent = {i: i for i in fr}
        reach = {i: span(i) for i in fr}

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i
        refused = []
        for a, b in links(alone):
            ra, rb = find(a), find(b)
            if ra == rb:
                continue
            if clash(reach[ra], reach[rb]):
                refused.append((a, b))
                continue
            keep, gone = min(ra, rb), max(ra, rb)
            parent[gone] = keep
            merged = dict(reach[keep])
            for s_, (lo, hi) in reach[gone].items():
                lo0, hi0 = merged.get(s_, (lo, hi))
                merged[s_] = (min(lo, lo0), max(hi, hi0))
            reach[keep] = merged
            del reach[gone]
        # a peak that reaches a group it could join on its own, but which
        # its own group cannot join, is torn between two: it joins neither
        torn = {p for a, b in refused for p, q in ((a, b), (b, a))
                if not clash(span(p), reach[find(q)])}
        if not torn - alone:
            break
        alone |= torn
    groups: dict[int, list[str]] = {}
    for i in fr:
        groups.setdefault(find(i), []).extend(keys[i])
    out.loc[fr, "key"] = [fragment_key(groups[find(i)]) for i in fr]
    return out


def satellite_fragments(roles: pd.DataFrame, resolution: Mapping[tuple[str, int], float], *,
                        ratio: float = SATELLITE_RATIO, fwhm: float = SATELLITE_FWHM
                        ) -> dict[str, str]:
    """The fragments that are side lobes of a brighter line, with why.

    A peak list processed from Orbitrap profiles can carry weak centroids a
    few line widths beside a much brighter line that the profile does not
    hold (`chem.sidelobes`). Read against window ions they are sub-formulas
    like any other centroid -- of a large O/N-rich ion almost anything is
    -- and were given to that ion by composition: an intact ion read F < 1
    and its branching flagged. A fragment (one key of `merge_fragment_peaks`)
    is a satellite when EVERY one of its peaks, in every spectrum it is in,
    lies within `fwhm` line widths of a peak of the same spectrum (any role)
    that is at least `ratio` times brighter; the width is that brighter
    peak's m/z over its reported resolution (`resolution`: (spectrum, peak)
    -> resolving power). A peak with no such neighbour, or whose neighbours
    report no resolution, keeps its fragment: a fragment that grows out of
    the lobe band at some energy is an ion. -> {key: note}."""
    out: dict[str, str] = {}
    fr = roles[roles["role"] == "fragment"]
    if not len(fr):
        return out
    spec = {}
    for s_, g in roles.groupby("spectrum", sort=False):
        mz = g["mz_cal"].to_numpy(dtype=float)
        res = np.array([float(resolution.get((str(s_), int(p)), math.nan)) for p in g["peak"]])
        with np.errstate(invalid="ignore", divide="ignore"):
            width = np.where(np.isfinite(res) & (res > 0), mz / res, math.nan)
        spec[s_] = (g["peak"].to_numpy(dtype=int), mz,
                    g["intensity"].fillna(0.0).to_numpy(dtype=float), width)
    for key, g in fr.groupby("key", sort=False):
        worst = None                                  # (offset / width, ratio, offset Da)
        for s_, p, m, i in zip(g["spectrum"], g["peak"], g["mz_cal"], g["intensity"]):
            peaks, mz, inten, width = spec[s_]
            i, m = float(i), float(m)
            if not (math.isfinite(i) and i > 0 and math.isfinite(m)):
                worst = None
                break
            d = np.abs(mz - m)
            with np.errstate(invalid="ignore"):
                near = (peaks != int(p)) & (inten >= ratio * i) & (d <= fwhm * width)
            if not near.any():
                worst = None
                break
            j = int(np.flatnonzero(near)[np.argmax(inten[near])])     # the brightest
            this = (float(d[j] / width[j]), float(inten[j] / i), float(d[j]))
            worst = this if worst is None else (max(worst[0], this[0]), min(worst[1], this[1]),
                                                max(worst[2], this[2]))
        if worst is not None:
            n, n_s = len(g), g["spectrum"].nunique()
            out[key] = (f"{key}: each of its {n} peak{'s' if n > 1 else ''} ({n_s} "
                        f"spectr{'a' if n_s > 1 else 'um'}) sits within {fwhm:g} line widths "
                        f"of a peak >= {ratio:g}x brighter in the same spectrum (up to "
                        f"{worst[2] * 1e3:.2f} mDa = {worst[0]:.2f} widths away, from a peak "
                        f"{worst[1]:.3g}x brighter): a side lobe of that line, not attributed "
                        "(its peaks count as unexplained)")
    return out


def set_aside_fragments(roles: pd.DataFrame, keys) -> pd.DataFrame:
    """`roles` with the fragment peaks of `keys` made 'unexplained' (no key,
    candidates, formulas or losses): in the TIC and the ion budget, in no
    fragment. The readings themselves are not touched."""
    out = roles.copy()
    sel = (out["role"] == "fragment") & out["key"].isin(list(keys))
    for c, v in (("role", "unexplained"), ("key", ""), ("candidates", ""), ("formulas", ""),
                 ("losses", "")):
        out.loc[sel, c] = v
    return out


def _pairs(text: str) -> list[tuple[str, str]]:
    """`peak_roles`' 'ion: value; ion: value' cells -> [(ion, value)]."""
    return [(a, b) for a, _, b in (item.partition(": ") for item in str(text).split("; ")
                                   if item)]


def _formula_mz(formula: str) -> float:
    """A fragment formula's exact m/z, as the readings compute it
    (`chemistry.ion_mass` of its counts and charge)."""
    counts, charge = SF.parse_ion(formula)
    return float(C.ion_mass(counts, charge))


def fragment_mz(roles: pd.DataFrame) -> dict[str, float]:
    """Where each fragment of a window sits: the mean of its peaks' calibrated
    m/z over the window's spectra, each peak weighted by the share of its
    spectrum's TIC it holds (every peak of `roles` counts in the TIC, as in
    `breakdown.ion_budget`). The best estimate of the physical ion's m/z: a
    peak counts by the signal it holds, so a centroid of a few counts --
    whose position says little -- moves it little; and a TIC fraction, not a
    raw intensity, so a spectrum scaled as a whole (a drift of the precursor
    supply, space charge) moves it not at all. A fragment whose
    peaks hold no intensity gets their plain mean; one with no finite m/z,
    NaN. -> {fragment key: m/z}."""
    r = roles.assign(intensity=roles["intensity"].fillna(0.0).clip(lower=0.0))
    tic = r.groupby("spectrum", sort=False)["intensity"].sum()
    out: dict[str, float] = {}
    for key, g in r[r["role"] == "fragment"].groupby("key", sort=False):
        mz = g["mz_cal"].to_numpy(dtype=float)
        t = tic.reindex(g["spectrum"]).to_numpy(dtype=float)
        w = np.divide(g["intensity"].to_numpy(dtype=float), t, out=np.zeros(len(t)),
                      where=t > 0)
        ok = np.isfinite(mz)
        if not ok.any():
            out[key] = math.nan
        elif w[ok].sum() > 0:
            out[key] = float(np.average(mz[ok], weights=w[ok]))
        else:
            out[key] = float(mz[ok].mean())
    return out


def fragment_compositions(roles: pd.DataFrame, *, tol_ppm: float = 5.0, tol_mda: float = 0.3
                          ) -> dict[str, tuple[float, list[str]]]:
    """What each fragment of a window can be where it sits: its m/z
    (`fragment_mz`) and the formulas read anywhere in it (rank 1, any window
    ion) whose exact m/z lies within the reading tolerance of that m/z --
    the same tolerance a single peak is read with (`subformula`), on the best
    estimate of the peak's position. -> {key: (m/z, formulas, sorted)}."""
    where = fragment_mz(roles)
    fr = roles[roles["role"] == "fragment"]
    read: dict[str, set[str]] = {}
    for key, text in zip(fr["key"], fr["formulas"]):
        read.setdefault(key, set()).update(f for _, f in _pairs(text))
    out = {}
    for key, formulas in read.items():
        mz = where.get(key, math.nan)
        tol = SF._tolerance_da(mz, tol_ppm, tol_mda) if math.isfinite(mz) else math.nan
        out[key] = (mz, sorted(f for f in formulas if abs(_formula_mz(f) - mz) <= tol))
    return out


#: A carbon-poor polyoxy composition: at most this many C, only H / N / ^N /
#: O, and at least `POLYOXY_MIN_O` O -- a cluster of inorganic pieces (HNO3,
#: H2O, H2O2, O2) on a nitrate or carbonate core, never an organic fragment.
POLYOXY_MAX_C = 1
POLYOXY_MIN_O = 4
_POLYOXY_ELEMENTS = frozenset({"C", "H", "N", "^N", "O"})


def polyoxy_inorganic(formula: str) -> bool:
    """Is `formula` a carbon-poor polyoxy H/N/O composition (<= `POLYOXY_MAX_C`
    C, >= `POLYOXY_MIN_O` O, nothing but C, H, N, ^N and O)? In 15N data such a
    composition aliases a fluorinated organic ion: C2F3 = H2^NO4 within
    0.21 mDa, so CF3COO- (C2F3O2-) reads as H2^NO6- and C3H2F3O2- as
    CH4^NO6-, both sub-formulas of an [M+^NO3]- cluster."""
    counts, _ = SF.parse_ion(formula)
    return (set(counts) <= _POLYOXY_ELEMENTS and counts.get("C", 0) <= POLYOXY_MAX_C
            and counts.get("O", 0) >= POLYOXY_MIN_O)


def _organic(formula: str) -> bool:
    return SF.parse_ion(formula)[0].get("C", 0) > POLYOXY_MAX_C


def candidates_across(roles: pd.DataFrame, order: Sequence[str], *, tol_ppm: float = 5.0,
                      tol_mda: float = 0.3, absent: Sequence[str] = ()
                      ) -> tuple[dict[str, list[str]], dict[str, str]]:
    """The composition candidates of every fragment of a window, across its
    energies: the window ions that read it, in some spectrum, as one of the
    formulas it can be where it sits (`fragment_compositions`: within the
    reading tolerance of its signal-weighted m/z).

    Which ions can make a fragment is a composition question, and a reading
    of one centroid answers it only as well as that centroid is placed. At
    the edge of the tolerance a sub-ppm jitter of the peak lets an ion read
    it in one spectrum and not the next (the N + ^N / C + O + H alias,
    `merge_fragment_peaks`); counted spectrum by spectrum, a weak spectrum
    could make a candidate -- or, unanimity asked of every spectrum, veto
    one: a stray centroid of a few counts where only the target can read
    the peak took the fragment from the co-isolate that made it. Weighted by
    the signal each spectrum holds, it does neither. An ion that reads it
    only as formulas beyond the tolerance of that m/z cannot make it there:
    left out, and said; when no formula read in it lies within the
    tolerance (the readings disagree), every ion that reads it is a
    candidate, and said. On our own files every fragment is read as one
    formula, by the same ions, in every spectrum.

    When the formulas it can be there hold an organic one (more than
    `POLYOXY_MAX_C` C) and a carbon-poor polyoxy one (`polyoxy_inorganic`),
    an ion that can read it only as the latter is not a candidate, said: in
    15N data C2F3 = H2^NO4 within 0.21 mDa, and a fluorinated co-isolate's
    CF3COO- (C2F3O2-) reads as H2^NO6-, a sub-formula of every [M+^NO3]-
    cluster. Only an organic reader the sample holds can do that: one in
    `absent` (`absent_ions`: in no MS2 spectrum of the window nor
    the MS1 window) cannot have made the peak, so it takes no fragment from
    the ions that read it as polyoxy -- said.

    -> ({fragment key: candidates in window `order`}, {key: note})."""
    pos = {f: i for i, f in enumerate(order)}
    rank = lambda f: (pos.get(f, len(pos)), f)  # noqa: E731
    comp = fragment_compositions(roles, tol_ppm=tol_ppm, tol_mda=tol_mda)
    fr = roles[roles["role"] == "fragment"]
    read: dict[str, dict[str, set[str]]] = {}
    for key, text in zip(fr["key"], fr["formulas"]):
        d = read.setdefault(key, {})
        for ion, f in _pairs(text):
            d.setdefault(ion, set()).add(f)
    cand, notes = {}, {}
    for key, d in read.items():
        mz, inside = comp[key]
        ok = [ion for ion, fs in d.items() if fs & set(inside)]
        cand[key] = sorted(ok or d, key=rank)
        if not ok:
            notes[key] = ("no formula read in it lies within the reading tolerance of its "
                          f"signal-weighted m/z {mz:.5f}: every window ion that reads it is a "
                          "candidate")
            continue
        said_out = []
        out = sorted(set(d) - set(ok), key=rank)
        if out:
            said = "; ".join(
                f"{ion} reads it only as "
                + ", ".join(f"{f} ({(mz - _formula_mz(f)) / _formula_mz(f) * 1e6:+.2f} ppm)"
                            for f in sorted(d[ion]))
                for ion in out)
            said_out.append(f"not a candidate: {said} -- beyond the reading tolerance of its "
                            f"signal-weighted m/z {mz:.5f}")
        # an organic reading of the peak (> POLYOXY_MAX_C C) outranks a
        # carbon-poor polyoxy one: an ion that can read it only as the latter
        # is not a candidate when another reads it as the former
        organic = [f for f in inside if _organic(f)]
        poor = {f for f in inside if polyoxy_inorganic(f)}
        if organic and poor:
            kept = [ion for ion in ok if (d[ion] & set(inside)) - poor]
            gone = [ion for ion in cand[key] if ion not in kept]
            held = [ion for ion in kept if ion not in absent]
            if held and gone:
                cand[key] = kept
                said_out.append(
                    "not a candidate: " + "; ".join(
                        f"{ion} reads it only as "
                        + ", ".join(sorted(d[ion] & poor)) for ion in gone)
                    + f" -- a carbon-poor polyoxy composition, while {', '.join(organic)} "
                    "(organic) is read there too")
            elif kept and gone:
                one, sole = len(kept) == 1, len(gone) == 1
                said_out.append(
                    f"{', '.join(kept)} read{'s' if one else ''} it as {', '.join(organic)} "
                    f"(organic) but {'is' if one else 'are'} absent from this sample: "
                    f"{', '.join(gone)}, which read{'s' if sole else ''} it only as carbon-poor "
                    f"polyoxy, stay{'s' if sole else ''} a candidate")
        if said_out:
            notes[key] = "; ".join(said_out)
    return cand, notes


def live_candidates(candidates: Mapping[str, Sequence[str]], absent: Sequence[str]
                    ) -> dict[str, list[str]]:
    """Each fragment's candidates the sample holds: the `absent` ones
    (`absent_ions`) set aside, in the order given; a fragment whose every
    candidate is absent keeps them all (nothing to set aside them for)."""
    gone = set(absent)
    return {k: ([c for c in v if c not in gone] or list(v)) for k, v in candidates.items()}


def shared_presence(roles: pd.DataFrame, candidates: Mapping[str, Sequence[str]],
                    spectra: Sequence[str], absent: Sequence[str] = ()) -> pd.DataFrame:
    """Where each fragment is read by EVERY one of its candidates: rows =
    `spectra` (in the order given), columns = the fragment keys (first-seen
    order, as `intensity_matrices`), True where every candidate reads one of
    the fragment's peaks in that spectrum (rank 1, any formula). A weak peak
    only some candidates can read sits where the others cannot make it: the
    intact test does not take it for the onset of the fragment they share
    -- unless it holds more than a stray of the fragment's signal
    (`ONSET_STRAY_SHARE`, `attribute_fragments`).

    `absent`: the named ions the sample lacks (`absent_ions`). They are set
    aside before the intact test (`attribute_fragments`), so they are not
    waited for here either (`live_candidates`): where only an absent ion
    fails to read a fragment, every candidate the sample holds reads it, and
    that is where it is shared. Waiting for it let an ion list naming an ion
    this sample lacks move the onset the test reads, and with it the call."""
    live = live_candidates(candidates, absent)
    fr = roles[roles["role"] == "fragment"]
    keys = list(dict.fromkeys(fr["key"]))
    out = pd.DataFrame(False, index=list(spectra), columns=keys)
    for (key, spec), g in fr.groupby(["key", "spectrum"], sort=False):
        if spec in out.index:
            readers = set().union(*(str(c).split(";") for c in g["candidates"]))
            out.at[spec, key] = set(live.get(key, ())) <= readers
    return out


def intensity_matrices(roles: pd.DataFrame, spectra: Sequence[str], ions: Sequence[str]
                       ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(ion intensities, fragment intensities, precursor-peak counts): rows
    = `spectra` in the order given (lowest energy first), columns = `ions` /
    the fragment keys (in first-seen order). A precursor or fragment split over
    two peaks within tolerance is summed; absent = 0."""
    # select, THEN fill: `.assign(intensity=<the whole column>)` on an empty
    # selection takes that column's index and makes rows of NaN (a fragment
    # key NaN for a window with no fragment at all)
    def rows(role: str) -> pd.DataFrame:
        sel = roles[roles["role"] == role]
        return sel.assign(intensity=sel["intensity"].fillna(0.0))
    pre = rows("precursor")
    ion_i = (pre.pivot_table(index="spectrum", columns="ion", values="intensity",
                             aggfunc="sum") if len(pre) else pd.DataFrame())
    ion_n = (pre.pivot_table(index="spectrum", columns="ion", values="intensity",
                             aggfunc="size") if len(pre) else pd.DataFrame())
    fr = rows("fragment")
    keys = list(dict.fromkeys(fr["key"]))
    frag_i = (fr.pivot_table(index="spectrum", columns="key", values="intensity",
                             aggfunc="sum") if len(fr) else pd.DataFrame())
    idx = list(spectra)
    return (ion_i.reindex(index=idx, columns=list(ions)).fillna(0.0),
            frag_i.reindex(index=idx, columns=keys).fillna(0.0),
            ion_n.reindex(index=idx, columns=list(ions)).fillna(0).astype("int64"))


def absent_ions(ions: Sequence[str], survivors: Mapping[str, float],
                ms1_shares: Mapping[str, float] | None = None) -> list[str]:
    """The named ions the sample lacks: no precursor intensity in `survivors`
    (missing, NaN or <= 0) AND a measured MS1 window share of 0. For 'absent
    from this sample' (`attribute_fragments`, `analyze`) `survivors` is each
    ion's precursor at ANY energy of the window (its maximum): an ion located
    only at a later energy is not absent. `share_weights` passes the lowest-
    energy survivors, which is what its split reads. An unknown MS1 share (NaN, or
    no entry) is not 0: such an ion is not absent. In `ions` order.

    Such an ion weighs 0 in every fragment it shares (`share_weights`), is
    set aside before the intact test and the mass-balance guard
    (`attribute_fragments`), and never wins a fragment from a present ion by
    the polyoxy rule (`candidates_across`): an ion list naming an ion this
    sample lacks moves nothing."""
    ms1 = ms1_shares or {}
    out = []
    for c in ions:
        s, m = float(survivors.get(c, math.nan)), float(ms1.get(c, math.nan))
        if not (math.isfinite(s) and s > 0) and math.isfinite(m) and m <= 0:
            out.append(c)
    return out


def share_weights(candidates: Sequence[str], survivors: Mapping[str, float],
                  ms1_shares: Mapping[str, float] | None = None
                  ) -> tuple[dict[str, float], str, str]:
    """Split a shared fragment among `candidates`, each by what is measured of
    IT -- one candidate's missing number never changes how the others split.

    A candidate's survivor is its precursor intensity in the window's lowest
    energy MS2 spectrum (> 0: there); its MS1 share, its share of the MS1
    isolation window (0: measured, and not there; NaN or missing: not known).

      * In neither (no survivor, an MS1 share of 0): weight 0 -- an ion list
        naming an ion the sample lacks takes nothing, and moves nothing.
      * Every other candidate has a survivor: split by the survivors
        ('ms2 survivors'). An ion in the MS2 spectrum keeps its survivor
        whatever the MS1 window holds.
      * Some have a survivor, the rest only an MS1 share: the survivors as
        they are, the rest put on their scale through the MS1 shares (survivor
        per unit share, from the candidates that have both)
        ('ms2 survivors + ms1 window').
      * None has a survivor: the MS1 shares ('ms1 window').
      * Else -- a candidate with no survivor whose MS1 share is not known, or
        no candidate with both to set the scale -- an equal split over the
        candidates not absent from both ('equal').

    -> ({candidate: weight}, source, note): the note names every candidate
    whose weight came from another source than the survivors, and why."""
    cands = list(candidates)
    ms1 = ms1_shares or {}
    s = {c: float(survivors.get(c, math.nan)) for c in cands}
    m = {c: float(ms1.get(c, math.nan)) for c in cands}
    alive = {c for c in cands if math.isfinite(s[c]) and s[c] > 0}
    absent = absent_ions(cands, survivors, ms1)
    live = [c for c in cands if c not in absent]
    rest = [c for c in live if c not in alive]
    known = [c for c in rest if math.isfinite(m[c])]

    def norm(v: Mapping[str, float]) -> dict[str, float]:
        tot = sum(v.values())
        return {c: (float(v[c]) / tot if c in v else 0.0) for c in cands}

    def equal(why: str) -> tuple[dict[str, float], str, str]:
        pool = live or cands
        return {c: (1.0 / len(pool) if c in pool else 0.0) for c in cands}, "equal", why

    if not live:
        return equal("every candidate is in neither the lowest-energy MS2 spectrum nor the "
                     "MS1 window: an equal split")
    if not rest:
        return norm({c: s[c] for c in alive}), "ms2 survivors", ""
    if len(known) < len(rest):
        miss = ", ".join(c for c in rest if c not in known)
        return equal(f"{miss}: no MS2 survivor at the lowest energy and no MS1 share known: "
                     "an equal split")
    if not alive:
        return norm({c: m[c] for c in rest}), "ms1 window", ""
    both = [c for c in cands if c in alive and math.isfinite(m[c]) and m[c] > 0]
    if not both:
        return equal(f"{', '.join(rest)}: no MS2 survivor at the lowest energy, and no "
                     "candidate with a survivor has an MS1 share to set the scale: an equal "
                     "split")
    k = sum(s[c] for c in both) / sum(m[c] for c in both)
    w = norm({**{c: s[c] for c in alive}, **{c: m[c] * k for c in rest}})
    said = ", ".join(f"{c} (share {m[c]:.4g})" for c in rest)
    return w, "ms2 survivors + ms1 window", (
        f"no MS2 survivor at the lowest energy for {said}: weighed by its MS1 window share "
        f"on the survivors' scale ({k:.4g} per unit share, from {', '.join(both)})")


def intact_test(candidates: Sequence[str], frac: pd.DataFrame, present: pd.Series
                ) -> tuple[list[str], str]:
    """The candidates the intact test excludes, and why it could not run.

    `frac`: TIC-normalised precursor fractions, rows = spectra lowest energy
    first, columns = window ions; `present`: the fragment's presence in the
    same rows. A candidate is excluded when its fraction at the fragment's
    onset is >= `INTACT_FRACTION` of its fraction at the lowest energy; a
    candidate absent at the lowest energy cannot be tested and stays. Not run
    (-> [], reason) when the fragment is present at the lowest energy, never
    present, or when it would exclude every candidate."""
    present = present.astype(bool)
    if not present.any():
        return [], "never present"
    if bool(present.iloc[0]):
        return [], "present at the lowest energy (in-source, or formed below the lowest step)"
    onset = present.index[int(np.argmax(present.to_numpy()))]
    lo = frac.iloc[0]
    out = [c for c in candidates
           if lo[c] > 0 and frac.loc[onset, c] >= INTACT_FRACTION * lo[c]]
    if len(out) == len(candidates):
        return [], "every candidate intact at the fragment's onset"
    return out, ""


def format_weights(w: Mapping[str, float]) -> str:
    return ";".join(f"{k}={v:.6g}" for k, v in w.items())


def held_at_lowest(ion: str, frac: pd.DataFrame, phi_f: pd.DataFrame,
                   candidates: Mapping[str, Sequence[str]]) -> float:
    """What `ion` held at the lowest energy, as a TIC fraction: its own
    precursor plus every fragment only it can make (one candidate by
    composition) -- its in-source products are its signal too, and a
    fragment they go on to make is still carried by it. A shared fragment
    is not counted: whose it is is what is being decided."""
    own = [k for k in phi_f.columns if list(candidates.get(k, ())) == [ion]]
    return float(frac[ion].iloc[0] + sum(float(phi_f[k].iloc[0]) for k in own))


def _tested_onset(present: pd.Series, shared: pd.Series, phi: pd.Series
                  ) -> tuple[pd.Series, str]:
    """Where the intact test reads a fragment's presence, and why it moved
    the onset or did not: (rows, note). `present`: the fragment's presence;
    `shared`: where every candidate reads it (both on the same rows, lowest
    energy first); `phi`: its TIC fraction.

    The onset moves to the first shared spectrum only when what was present
    before it is a stray -- at every earlier spectrum at most
    `ONSET_STRAY_SHARE` of the fragment's TIC fraction at that first shared
    one. Else the fragment's own onset stands (at the lowest energy: in-source,
    and the test does not run): a sub-ppm jitter that lets only some candidates
    read the fragment where it already holds signal does not make it a later
    one. No shared spectrum: no onset, no test. Note '' when nothing moved."""
    if not present.any():
        return present, ""
    tested = present & shared
    first = present.index[int(np.argmax(present.to_numpy()))]
    if not tested.any():
        return tested, (f"no spectrum in which every candidate reads it (present from {first}, "
                        "read there by some of them only): the intact test has no onset")
    on = tested.index[int(np.argmax(tested.to_numpy()))]
    if on == first:
        return tested, ""
    before = present & (np.arange(len(present)) < present.index.get_loc(on))
    at_on = float(phi.loc[on])
    stray = float(phi[before].max()) / at_on if at_on > 0 else math.inf
    if stray <= ONSET_STRAY_SHARE:
        return tested, (f"the intact test reads its onset at {on}, the first spectrum in which "
                        f"every candidate reads it (present from {first}, read there by some of "
                        f"them only, holding at most {stray:.3g}x its TIC fraction at {on}: a "
                        "stray)")
    return present, (f"present from {first}, read there by some of its candidates only, but "
                     f"holding {stray:.3g}x its TIC fraction at {on} (the first spectrum in "
                     f"which every candidate reads it) -- more than a stray (<= "
                     f"{ONSET_STRAY_SHARE:g}x): the intact test reads the fragment's own onset")


def attribute_fragments(ion_int: pd.DataFrame, frag_int: pd.DataFrame, tic: pd.Series,
              candidates: Mapping[str, Sequence[str]],
              ms1_shares: Mapping[str, float] | None = None,
              shared: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per fragment key: its candidates, the call, its basis and the weights.

    `ion_int` / `frag_int` from `intensity_matrices` (lowest energy first),
    `tic` per spectrum (same index), `candidates` key -> window ions that read
    it (composition), `ms1_shares` ion -> MS1 window share (the fallback),
    `shared` (`shared_presence`, same rows) where every candidate reads each
    fragment: the intact test takes its onset there when what was present
    before it is a stray (`_tested_onset`; without `shared`, wherever the
    fragment is present). A candidate the sample lacks (`absent_ions`) is
    set aside first, said: weight 0, out of the intact test, the guard and
    the guard's bound; one candidate left so is the fragment's by
    composition. `shared` must set the same ions aside (`shared_presence`
    with `absent`): an absent ion it waits for moves the onset the test
    reads. Columns: key, candidates, n_candidates, call (an ion,
    'ambiguous', or '' for a fragment only absent ions can make), basis,
    weights (dict), weight_source, excluded (';'), test_note."""
    frac = ion_int.div(tic.replace(0, np.nan), axis=0).fillna(0.0)
    phi_f = frag_int.div(tic.replace(0, np.nan), axis=0).fillna(0.0)
    survivors = ion_int.iloc[0].to_dict() if len(ion_int) else {}
    # a named ion the sample lacks is set aside before anything is decided:
    # it weighs 0, so it must not keep the intact test from leaving one ion
    # (and the guard from reading that call), nor make a fragment shared that
    # is otherwise one ion's own (the guard's bound)
    named = list(dict.fromkeys(c for k in frag_int.columns for c in candidates[k]))
    # absent = in no MS2 spectrum of the window (its precursor at ANY energy),
    # not only the lowest: an ion located at a later energy -- the target too
    # -- is no absent ion, and keeps its own fragments
    gone = absent_ions(named, ion_int.max().to_dict() if len(ion_int) else {}, ms1_shares)
    live = live_candidates(candidates, gone)
    rows = []
    for key in frag_int.columns:
        cands = list(candidates[key])
        present = frag_int[key] > 0
        excluded, note = [], ""
        lv = live[key]
        aside = [c for c in cands if c not in lv]
        aside_note = (f"{', '.join(aside)}: in no MS2 spectrum of the window nor in the "
                      "MS1 window (absent from this sample): weight 0, set aside before the "
                      "intact test") if aside else ""
        if cands and all(c in gone for c in cands):
            # only ions the sample lacks can make it: it is no window ion's
            # (an absent ion decides nothing, so it takes nothing either)
            one = len(cands) == 1
            call, basis, src = "", BASIS_ABSENT, "none"
            w = {c: 0.0 for c in cands}
            note = (f"only {', '.join(cands)} read{'s' if one else ''} it, and "
                    f"{'it is' if one else 'they are'} in no MS2 spectrum of the window nor in "
                    "the MS1 window (absent from this sample): no window ion takes it")
        elif len(cands) == 1:
            call, basis, w, src = cands[0], BASIS_COMPOSITION, {cands[0]: 1.0}, "composition"
        elif len(lv) == 1:
            # one candidate the sample holds: as if the absent ones were not named
            call, basis, src = lv[0], BASIS_COMPOSITION, "composition"
            w = {c: float(c == lv[0]) for c in cands}
            note = aside_note
        else:
            sh = (shared[key].reindex(present.index, fill_value=False).astype(bool)
                  if shared is not None and key in shared.columns else present)
            tested, onset_note = _tested_onset(present, sh, phi_f[key])
            excluded, note = intact_test(lv, frac, tested)
            if onset_note:
                note = onset_note if not tested.any() else "; ".join(
                    t for t in (onset_note, note) if t)
            left = [c for c in lv if c not in excluded]
            if len(left) == 1:
                # the mass-balance guard: the called ion must be able to carry
                # the fragment's growth with what it held at the lowest energy
                gain = (phi_f[key] - phi_f[key].iloc[0])
                had = held_at_lowest(left[0], frac, phi_f, live)
                over = gain[gain > BRANCHING_FLAG * had]
                if len(over):
                    withdrawn = (f"intact test withdrawn: it left {left[0]}, but the fragment's "
                                 f"TIC fraction grew by {float(over.iloc[0]):.4g} at "
                                 f"{over.index[0]}, more than the {had:.4g} {left[0]} held at "
                                 "the lowest energy (its precursor and the fragments only it "
                                 "can make)")
                    note = "; ".join(t for t in (onset_note, withdrawn) if t)
                    excluded, left = [], lv
            if len(left) == 1:
                call, basis, src = left[0], BASIS_INTACT, "intact test"
                w = {c: float(c == left[0]) for c in cands}
            else:
                call, basis = BASIS_AMBIGUOUS, BASIS_AMBIGUOUS
                w_left, src, w_note = share_weights(left, survivors, ms1_shares)
                w = {c: w_left.get(c, 0.0) for c in cands}
                note = "; ".join(t for t in (note, w_note) if t)
            note = "; ".join(t for t in (aside_note, note) if t)
        rows.append({"key": key, "candidates": ";".join(cands), "n_candidates": len(cands),
                     "call": call, "basis": basis, "weights": w, "weight_source": src,
                     "excluded": ";".join(excluded), "test_note": note})
    return pd.DataFrame(rows, columns=["key", "candidates", "n_candidates", "call", "basis",
                                       "weights", "weight_source", "excluded", "test_note"])


def branching(ion_int: pd.DataFrame, frag_int: pd.DataFrame, tic: pd.Series,
              weights: Mapping[str, Mapping[str, float]],
              candidates: Mapping[str, Sequence[str]] | None = None) -> pd.DataFrame:
    """Branching of every window ion into each of its fragments, per energy.

    b(X, f, E) = w(X, f) * (phi_f(E) - phi_f(E0)) / (phi_X(E0) - phi_X(E)),
    phi = intensity / TIC, E0 = the lowest energy (the first row). NaN where X
    lost nothing (phi_X(E0) - phi_X(E) <= 0); a fragment that shrinks (it
    fragments on) counts negative. With `candidates` (fragment key -> the
    window ions that can make it), an energy where X is unobserved -- neither
    its precursor nor any fragment it can make is there, the 0/0 of
    `breakdown.survival` -- has no branching, gain or loss (NaN): X did not
    lose everything there, it was not seen. Columns: spectrum, ion, key,
    weight, frag_gain, ion_loss, branching -- one row per (energy above E0,
    ion, fragment with weight > 0)."""
    cols = ["spectrum", "ion", "key", "weight", "frag_gain", "ion_loss", "branching"]
    if len(ion_int) < 2:
        return pd.DataFrame(columns=cols)
    t = tic.replace(0, np.nan)
    phi_i = ion_int.div(t, axis=0)
    phi_f = frag_int.div(t, axis=0).fillna(0.0)
    seen = None
    if candidates is not None:
        seen = ion_int > 0
        for key in frag_int.columns:
            for ion in candidates.get(key, ()):
                if ion in seen.columns:
                    seen[ion] = seen[ion] | (frag_int[key] > 0)
    rows = []
    for spec in ion_int.index[1:]:
        for ion in ion_int.columns:
            loss = float(phi_i[ion].iloc[0] - phi_i.loc[spec, ion])
            hole = seen is not None and not bool(seen.loc[spec, ion])
            for key, w in weights.items():
                wx = float(w.get(ion, 0.0))
                if wx <= 0:
                    continue
                gain = wx * float(phi_f.loc[spec, key] - phi_f[key].iloc[0])
                if hole:
                    rows.append({"spectrum": spec, "ion": ion, "key": key, "weight": wx,
                                 "frag_gain": math.nan, "ion_loss": math.nan,
                                 "branching": math.nan})
                    continue
                b = gain / loss if (math.isfinite(loss) and loss > 0) else math.nan
                rows.append({"spectrum": spec, "ion": ion, "key": key, "weight": wx,
                             "frag_gain": gain, "ion_loss": loss, "branching": b})
    return pd.DataFrame(rows, columns=cols)


def branching_breach(gain: float, loss: float) -> bool:
    """The mass-balance flag of one (ion, energy): X's products' weighted
    TIC-fraction gain (summed over its fragments) exceeds `BRANCHING_FLAG`
    times what X lost. Where X lost something that is a branching sum above
    the flag; where it lost nothing or GAINED, the ratio is NaN, and any
    growth of its products breaks the invariant -- worst of all, so it is
    flagged too, never let through as a NaN."""
    gain, loss = float(gain), float(loss)
    if not (math.isfinite(gain) and math.isfinite(loss)):
        return False
    return gain > 0 and gain > BRANCHING_FLAG * max(loss, 0.0)
