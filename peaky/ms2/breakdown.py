"""Precursor breakdown curves across collision energies: survival, E50, onsets
and the ion budget of each spectrum.

Pure arithmetic on the intensity matrices of ONE parent window
(`attribute.intensity_matrices`, lowest energy first): no Mascope, no I/O.

Survival. F_X = I_X / (I_X + sum over X's fragments of w * I_fragment): the
precursor's share of what it and its products hold in the SAME spectrum. A
ratio within one spectrum holds whatever happened between the spectra of two
energies -- a drift of the precursor's supply (source, transmission,
isolation), space charge or an AGC misprediction scaling one spectrum as a
whole. (Orbitrap intensities are reported per unit injection time, so the
injection time itself moves nothing.) A fragment only X can make counts fully;
a shared one by X's weight (`attribute`). F_lo gives every fragment X can make
(by composition) fully to X: the lower bound over the observed products, and
F_lo..F is the band (F itself is not an upper bound: a survivor split can give
X more of a shared fragment than it made, docs/MS2.md §11.5). Both see only the
products the spectrum HOLDS: a product outside the observed m/z range (below
the scan's low-mass limit, as the small ion an adduct releases) is in neither
sum, and F and F_lo read high -- an ion whose only channel is unobserved keeps
F = 1. S_X = I_X(E) / I_X(E_min) is reported too: it compares spectra recorded
at different times, so it is a breakdown curve only while the precursor supply
holds steady -- and then it sees a loss to unobserved products that F cannot.

E50: linear interpolation at the FIRST downward crossing of 0.5 on the NCE
axis, and in the same segment on the volts axis; E_cm = V * m_gas / (m_gas +
m/z), with m_gas the collision gas (N2 by default: an assumption, a
parameter). The crossing is bracketed by measured points or it is not given:
'below_range' (already < 0.5 at the lowest energy), 'above_range' (never
below 0.5), 'single_energy', 'gap' (the crossing falls across energies
where the ion is unobserved), 'no_data'. A curve that climbs back by more
than `MONOTONIC_TOL` is flagged non-monotonic, and its first crossing is
still the one reported.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "N2_MASS", "E50_THRESHOLD", "MONOTONIC_TOL", "FLAG_INTERPOLATED", "FLAG_BELOW",
    "FLAG_ABOVE", "FLAG_SINGLE", "FLAG_NO_DATA", "FLAG_GAP", "E50", "e50", "e_cm", "survival",
    "onset", "ion_budget",
]

#: The collision gas's mass (Da) for the centre-of-mass energy: N2, 28.0061.
#: An assumption -- the HCD cell's gas is not in the data Mascope serves -- and
#: a parameter (`--gas-mass`).
N2_MASS = 28.0061
E50_THRESHOLD = 0.5
#: A curve that rises by more than this between two observed energies is
#: flagged non-monotonic. On our own files no window ion's F rose at all
#: between neighbouring steps (2026-09-10, seven windows; 2026-09-11, four),
#: while an intact ion's precursor intensity wanders by about +-2 % between
#: steps (S of the 2026-09-10 file's C7H6NS+ reads 1.006 at NCE 11; of its
#: 197.13 co-isolate C12H21O2+, 0.872 -> 0.837 -> 0.858, a rise of 0.021 that
#: is flagged). 0.02 is that wander: a margin, not a fit.
MONOTONIC_TOL = 0.02

FLAG_INTERPOLATED = "interpolated"
FLAG_BELOW = "below_range"
FLAG_ABOVE = "above_range"
FLAG_SINGLE = "single_energy"
FLAG_NO_DATA = "no_data"
FLAG_GAP = "gap"


@dataclass(frozen=True)
class E50:
    """One curve's E50: `nce`, `volts` (NaN when the volts are unknown or the
    crossing is not bracketed), `flag`, `monotonic`, and the curve's span."""
    nce: float
    volts: float
    flag: str
    monotonic: bool
    n_energies: int
    nce_min: float
    nce_max: float


def e_cm(volts, precursor_mz: float, gas_mass: float = N2_MASS):
    """Centre-of-mass energy (eV) of a singly-charged ion accelerated through
    `volts` onto a gas molecule of `gas_mass`: E_lab * m_gas / (m_gas + m)."""
    g = float(gas_mass)
    if not (math.isfinite(g) and g > 0):
        raise ValueError(f"gas_mass must be a positive mass in Da, got {gas_mass!r}")
    m = float(precursor_mz)
    if not (math.isfinite(m) and m > 0):
        raise ValueError(f"precursor_mz must be positive, got {precursor_mz!r}")
    return np.asarray(volts, dtype=float) * g / (g + m) if np.ndim(volts) else \
        float(volts) * g / (g + m)


def e50(nce: Sequence[float], value: Sequence[float], volts: Sequence[float] | None = None,
        *, threshold: float = E50_THRESHOLD) -> E50:
    """E50 of one curve: the first downward crossing of `threshold`, linearly
    interpolated between the bracketing energies (on NCE, and with the same
    fraction on volts). The points are taken in NCE order (stable); a NaN
    value is an energy where the curve is unobserved (no precursor and no
    fragment: 0/0). A crossing between two observed energies with unobserved
    ones between them is not interpolated across the hole: 'gap', E50 NaN --
    in the 2026-09-10 file's 136.02 window the 0.9 % co-isolate C8H10NO+ is
    seen at NCE 1 and next at NCE 81 (through a fragment), and a straight line
    across would read E50 41."""
    x = np.asarray(nce, dtype=float)
    y = np.asarray(value, dtype=float)
    v = np.full(len(x), np.nan) if volts is None else np.asarray(volts, dtype=float)
    keep = np.isfinite(x)
    x, y, v = x[keep], y[keep], v[keep]
    order = np.argsort(x, kind="stable")
    x, y, v = x[order], y[order], v[order]
    seen = np.flatnonzero(np.isfinite(y))
    n = len(seen)
    if not n:
        return E50(math.nan, math.nan, FLAG_NO_DATA, True, 0, math.nan, math.nan)
    mono = bool(n < 2 or (np.diff(y[seen]) <= MONOTONIC_TOL).all())
    span = dict(n_energies=n, nce_min=float(x[seen[0]]), nce_max=float(x[seen[-1]]))
    if n == 1:
        return E50(math.nan, math.nan, FLAG_SINGLE, mono, **span)
    if y[seen[0]] < threshold:
        return E50(math.nan, math.nan, FLAG_BELOW, mono, **span)
    for i, j in zip(seen[:-1], seen[1:]):
        if y[i] >= threshold > y[j]:
            if j - i > 1:
                return E50(math.nan, math.nan, FLAG_GAP, mono, **span)
            t = (y[i] - threshold) / (y[i] - y[j])
            e = float(x[i] + t * (x[j] - x[i]))
            ev = float(v[i] + t * (v[j] - v[i]))
            return E50(e, ev if math.isfinite(ev) else math.nan, FLAG_INTERPOLATED, mono,
                       **span)
    return E50(math.nan, math.nan, FLAG_ABOVE, mono, **span)


def survival(ion_int: pd.DataFrame, frag_int: pd.DataFrame,
             weights: Mapping[str, Mapping[str, float]],
             candidates: Mapping[str, Sequence[str]]) -> dict[str, pd.DataFrame]:
    """The three curves of every window ion, per spectrum (rows lowest energy
    first): {'f': F, 'f_lo': F_lo, 's': S, 'frag_w': the weighted fragment sum,
    'frag_all': the composition fragment sum}, each a DataFrame shaped like
    `ion_int`. Where the ion is unobserved -- no precursor and no fragment it
    can make (0/0) -- every curve is NaN, S too (its 0 there would be a
    crossing read through a hole that F and F_lo call a gap); S is NaN as
    well where the precursor is absent at the lowest energy."""
    w = pd.DataFrame(0.0, index=frag_int.columns, columns=ion_int.columns)
    c = w.copy()
    for key in frag_int.columns:
        for ion, wx in weights.get(key, {}).items():
            if ion in w.columns:
                w.loc[key, ion] = float(wx)
        for ion in candidates.get(key, ()):
            if ion in c.columns:
                c.loc[key, ion] = 1.0
    frag_w = frag_int.to_numpy() @ w.to_numpy() if len(frag_int.columns) else \
        np.zeros(ion_int.shape)
    frag_all = frag_int.to_numpy() @ c.to_numpy() if len(frag_int.columns) else \
        np.zeros(ion_int.shape)
    i = ion_int.to_numpy(dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        f = i / (i + frag_w)
        f_lo = i / (i + frag_all)
        s = i / i[0:1, :] if len(i) else i
    s = np.where(np.isfinite(s) & (i + frag_all > 0), s, np.nan)
    mk = lambda a: pd.DataFrame(a, index=ion_int.index, columns=ion_int.columns)  # noqa: E731
    return {"f": mk(f), "f_lo": mk(f_lo), "s": mk(s), "frag_w": mk(frag_w),
            "frag_all": mk(frag_all)}


def onset(nce: Sequence[float], present: Sequence[bool]) -> float:
    """The lowest energy at which a fragment is present (NaN: never)."""
    x = np.asarray(nce, dtype=float)
    p = np.asarray(present, dtype=bool) & np.isfinite(x)
    return float(x[p].min()) if p.any() else math.nan


def ion_budget(roles: pd.DataFrame) -> pd.DataFrame:
    """Per spectrum: its TIC (summed intensity of every peak), the fraction
    the window ions and their readings explain (precursor + fragment peaks)
    and the unexplained fraction, with the peak counts."""
    r = roles.assign(intensity=roles["intensity"].fillna(0.0))
    r = r.assign(explained=np.where(r["role"] != "unexplained", r["intensity"], 0.0),
                 unexp=(r["role"] == "unexplained").astype(int))
    g = r.groupby("spectrum", sort=False).agg(tic=("intensity", "sum"),
                                              explained=("explained", "sum"),
                                              n_peaks=("peak", "size"),
                                              n_unexplained=("unexp", "sum"))
    with np.errstate(invalid="ignore", divide="ignore"):
        g["budget_explained"] = np.where(g["tic"] > 0, g["explained"] / g["tic"], np.nan)
    g["budget_unexplained"] = 1.0 - g["budget_explained"]
    return g.drop(columns="explained")
