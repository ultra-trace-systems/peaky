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
still the one reported. `e50_many` reads the same rule off many curves on one
energy axis at once (a bootstrap's resamples, `replicates`).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "N2_MASS", "E50_THRESHOLD", "MONOTONIC_TOL", "FLAG_INTERPOLATED", "FLAG_BELOW",
    "FLAG_ABOVE", "FLAG_SINGLE", "FLAG_NO_DATA", "FLAG_GAP", "FLAGS", "E50", "e50", "e50_many",
    "observed_ends", "e_cm", "survival", "onset", "ion_budget",
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
#: The flags by code, as `e50_many` returns them (index = code).
FLAGS = (FLAG_INTERPOLATED, FLAG_BELOW, FLAG_ABOVE, FLAG_SINGLE, FLAG_NO_DATA, FLAG_GAP)


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


def _axis(nce: Sequence[float], values, volts: Sequence[float] | None
          ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(NCE, values energies x curves, volts) in `e50_many`'s order: a finite
    NCE only, the points in stable NCE order."""
    x = np.asarray(nce, dtype=float)
    y = np.asarray(values, dtype=float)
    if y.ndim == 1:
        y = y[:, None]
    v = np.full(len(x), np.nan) if volts is None else np.asarray(volts, dtype=float)
    keep = np.isfinite(x)
    x, y, v = x[keep], y[keep], v[keep]
    order = np.argsort(x, kind="stable")
    return x[order], y[order], v[order]


def observed_ends(nce: Sequence[float], values, volts: Sequence[float] | None = None, *,
                  with_values: bool = False) -> tuple:
    """The first and the last energy at which each column of `values`
    (energies x curves, NaN where a curve is unobserved) is observed, in
    `e50_many`'s order: (NCE first, NCE last, volts first, volts last), NaN
    for a column observed nowhere; `with_values` also the curve's value
    there (value first, value last). A bootstrap resample observes only the
    energies its averaged curve observes; one that misses the averaged
    curve's first (last) one reads its 'below_range' ('above_range') at an
    energy that is not the curve's end, and one left one energy reads its
    E50 on the side of it its value lies (`replicates.bootstrap_e50`)."""
    x, y, v = _axis(nce, values, volts)
    n_e, n_c = y.shape
    out = tuple(np.full(n_c, np.nan) for _ in range(6 if with_values else 4))
    if not n_e:
        return out
    obs = np.isfinite(y)
    seen = obs.any(axis=0)
    first = np.argmax(obs, axis=0)
    last = n_e - 1 - np.argmax(obs[::-1], axis=0)
    for a, src, idx in zip(out[:4], (x, x, v, v), (first, last, first, last)):
        a[seen] = src[idx[seen]]
    if with_values:
        cols = np.flatnonzero(seen)
        out[4][cols] = y[first[cols], cols]
        out[5][cols] = y[last[cols], cols]
    return out


def e50_many(nce: Sequence[float], values, volts: Sequence[float] | None = None, *,
             threshold: float = E50_THRESHOLD, bracket: bool = False) -> tuple:
    """`e50` of many curves on one energy axis at once (a bootstrap's
    resamples): `values` is energies x curves, one column per curve, NaN
    where a curve is unobserved. -> (E50 on NCE, on volts, flag code), one
    per column; the code indexes `FLAGS`. The same rule, column by column,
    as `e50`: the points in stable NCE order, a finite NCE only; 'no_data',
    'single_energy', 'below_range' first, then the first downward crossing
    between two observed energies -- 'gap' when unobserved ones lie between
    them -- else 'above_range' (tested against `e50` on random curves with
    holes). Vectorised over the columns: one pass over the energies.
    `bracket`: also the two observed energies the crossing lies between, a
    4th item (NCE before, NCE after, volts before, volts after; NaN where
    the curve does not cross) -- where a 'gap' crossing's E50 lies, and the
    energies whose scans an interpolated one rests on."""
    x, y, v = _axis(nce, values, volts)
    n_e, n_c = y.shape
    obs = np.isfinite(y)
    n_obs = obs.sum(axis=0)
    code = np.full(n_c, FLAGS.index(FLAG_ABOVE), dtype=np.int64)
    e = np.full(n_c, np.nan)
    ev = np.full(n_c, np.nan)
    first = np.argmax(obs, axis=0)
    y_first = y[first, np.arange(n_c)] if n_e else np.full(n_c, np.nan)
    done = n_obs < 2
    code[n_obs == 0] = FLAGS.index(FLAG_NO_DATA)
    code[n_obs == 1] = FLAGS.index(FLAG_SINGLE)
    below = ~done & (y_first < threshold)
    code[below] = FLAGS.index(FLAG_BELOW)
    done = done | below
    prev_i = np.full(n_c, -1, dtype=np.int64)
    prev_y = np.full(n_c, np.nan)
    xa, xb, va, vb = (np.full(n_c, np.nan) for _ in range(4))
    for i in range(n_e):
        o = obs[i]
        yi = y[i]
        with np.errstate(invalid="ignore"):
            cross = o & ~done & (prev_i >= 0) & (prev_y >= threshold) & (yi < threshold)
        gap = cross & (i - prev_i > 1)
        code[gap] = FLAGS.index(FLAG_GAP)
        if cross.any():
            pc = prev_i[cross]
            xa[cross], xb[cross], va[cross], vb[cross] = x[pc], x[i], v[pc], v[i]
        hit = cross & ~gap
        if hit.any():
            pi = prev_i[hit]
            t = (prev_y[hit] - threshold) / (prev_y[hit] - yi[hit])
            e[hit] = x[pi] + t * (x[i] - x[pi])
            with np.errstate(invalid="ignore"):
                ev[hit] = v[pi] + t * (v[i] - v[pi])
            code[hit] = FLAGS.index(FLAG_INTERPOLATED)
        done = done | cross
        prev_i = np.where(o, i, prev_i)
        prev_y = np.where(o, yi, prev_y)
    ev[~np.isfinite(ev)] = np.nan
    if bracket:
        return e, ev, code, (xa, xb, va, vb)
    return e, ev, code


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
    and the unexplained fraction, with the peak counts. A peak read as an
    Orbitrap harmonic of a brighter line (role 'harmonic', `readings`) is
    that line seen again, no ion: out of the TIC and the peak counts, counted
    apart (`n_harmonic`, `harmonic_intensity`)."""
    r = roles.assign(intensity=roles["intensity"].fillna(0.0))
    harm = (r["role"] == "harmonic").to_numpy()
    r = r.assign(counted=np.where(harm, 0.0, r["intensity"]),
                 explained=np.where(~harm & (r["role"] != "unexplained"), r["intensity"], 0.0),
                 unexp=(r["role"] == "unexplained").astype(int),
                 n=(~harm).astype(int), harm=harm.astype(int),
                 harm_i=np.where(harm, r["intensity"], 0.0))
    g = r.groupby("spectrum", sort=False).agg(tic=("counted", "sum"),
                                              explained=("explained", "sum"),
                                              n_peaks=("n", "sum"),
                                              n_unexplained=("unexp", "sum"),
                                              n_harmonic=("harm", "sum"),
                                              harmonic_intensity=("harm_i", "sum"))
    with np.errstate(invalid="ignore", divide="ignore"):
        g["budget_explained"] = np.where(g["tic"] > 0, g["explained"] / g["tic"], np.nan)
    g["budget_unexplained"] = 1.0 - g["budget_explained"]
    return g.drop(columns="explained")
