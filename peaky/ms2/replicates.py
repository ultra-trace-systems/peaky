"""Replicate statistics of a window's breakdown, from the scans behind its spectra.

Pure: no Mascope, no I/O. An `ms2 extract --file` or `--per-scan` folder holds,
beside each averaged spectrum, every one of its peaks read in every scan the
spectrum averages (`scans.SCAN_PEAK_COLUMNS`: 0 where a scan has none). The
breakdown itself is decided ONCE, on the averaged spectra (`analyze`): which
peak is which window ion's precursor, which fragment, and every fragment's
share weights. Here the same roles and weights are re-read scan by scan:

  * Per scan j of a spectrum: I_X,j (X's precursor peaks), and its fragments
    -> F_j = I_X,j / (I_X,j + sum w I_frag,j), F_lo_j likewise with every
    composition candidate's fragment given whole, S_j = I_X,j / I_X(E_min)
    with I_X(E_min) the averaged lowest-energy precursor (`spectrum_curves`).
    Their spread (sd, sem) is the scan-to-scan scatter of the curves; the
    averaged F is NOT their mean (a ratio of means, not a mean of ratios):
    the per-scan MEANS of every peak give back the averaged spectrum, and F
    from them is F, exactly (tested).
  * Each peak's per-scan readings are put on its averaged height's scale
    (x I_avg / their mean, `spectrum_scans`): on an `--file` folder that is
    a factor of 1 (the averaged intensity IS the per-scan mean), on a
    `--per-scan` folder it carries the server's height, which runs about 2 %
    below the per-scan mean on the median peak (the per-scan mean is
    `scans.SERVER_HEIGHT_RATIO`, 1.020, times it) and varies by peak: the
    factor is 0.89-1.01 on our own standard A's S/N-10 spectra (median
    0.979) -- so the per-scan means are the averaged spectrum in both, and F
    from them is the analysis's F. A factor outside `SCALE_BAND` says the
    rows are not that peak read scan by scan (a stray centroid in one scan
    of several, an m/z axis the rows do not share): the peak is held.
  * The E50 interval: a bootstrap over the scans WITHIN each energy (each
    energy's scans drawn with replacement, all of a spectrum's peaks from
    the same draw -- a scan's precursor and its fragments go together), the
    curves rebuilt from the resampled means and the same E50 rule
    (`breakdown.e50_many`) read off every resample (`bootstrap_e50`); the
    band is the 16th / 84th percentiles of the resampled E50s (`interval`),
    each resample ranked as what it is known to be -- one whose draws at an
    energy hold nothing of the ion does not observe it there, and where
    that is the curve's first (last) energy and it reads below (above) 0.5
    at the first (last) one it observes, its E50 is known only to lie at or
    below (above) that one: a hole at that end, not an E50 beyond the
    energies. So is one left a single observed energy (read on the side of
    it its value lies), and one left none (anywhere): never a resample
    dropped, a band read off the others alone.
    A one-scan energy has no spread to draw from: it adds nothing to the
    interval, and is counted -- and where it is one of the two energies
    the crossing lies between, the band holds none of its scatter (said
    per E50, `analyze`).
  * What the band is: a percentile bootstrap of a few scans. Drawing n of n
    scans with replacement spreads a mean by sqrt((n - 1) / n) of its
    standard error, and the scatter itself is read off those few scans, so
    the band holds the true E50 less often than a 68 % band would at a few
    scans per energy -- measured on synthetic windows (3 energies, 10 %
    scan-to-scan scatter of the precursor and its fragment, 4000 datasets
    each, 1000 resamples): in 51 % of them at 2 scans per energy, 54 % at
    3, 59 % at 5, 62 % at 7 and 66 % at 20. It is the scans' scatter, not a
    confidence interval of the method.

What the scans cannot carry: a product outside the scan's m/z range (both
the averaged and the per-scan readings see only what the spectrum holds), and
a peak of the averaged spectrum that no scan reads, or reads on another scale
(held at its averaged height in every scan: its spread is not in the
statistics, counted; a curve that only held peaks enter has no spread at all,
`analyze`).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from peaky.ms2 import breakdown as BD

__all__ = [
    "BOOTSTRAP_B", "BOOTSTRAP_MIN", "BOOTSTRAP_SEED", "BOOTSTRAP_PERCENTILES", "MATCH_DA",
    "SCALE_BAND", "ZERO_WIDTH_REL", "REPLICATE_COLUMNS", "SpectrumScans", "spectrum_scans",
    "window_scans", "window_seed", "spectrum_curves", "spread", "bootstrap_e50", "interval",
    "zero_width",
]

#: Resamples per window (`ms2 analyze --bootstrap`, 0 = off). On our own
#: 2026-09-11 standard A ([M+H]+ C10H17+, F E50 NCE 13.89, 5-9 scans per
#: energy) the 16th / 84th percentiles of 1000 resamples scatter by 0.003 /
#: 0.002 NCE (sd over 20 seeds) about a band 0.10 NCE wide; 200 and 5000
#: resamples read the same band within 0.004 NCE. A window's 1000 resamples
#: cost about a millisecond.
BOOTSTRAP_B = 1000
#: The fewest resamples a band is read off (fewer are refused; 0 is none).
#: The 16th / 84th percentiles of a few resamples rest on one or two of them
#: -- of one, the 'band' is that resample's E50, of zero width and not
#: holding the averaged one -- and they scatter with the seed: on standard A
#: by 0.017 NCE (sd over 20 seeds) at 10 resamples, 0.006 at 100 and 0.002
#: at 1000, about a band 0.10 NCE wide.
BOOTSTRAP_MIN = 100
#: The resampling seed: each window draws from its own stream, seeded by this
#: and the parent key (`window_seed`), so a window's interval does not depend
#: on the other windows of the folder, their order, or the interpreter's hash
#: seed.
BOOTSTRAP_SEED = 20261018
#: The interval's percentiles of the resampled E50s (16 / 84 %): the band a
#: normal scatter's +-1 sd would span -- a percentile bootstrap of a few
#: scans holds the true E50 less often than that (module docstring).
BOOTSTRAP_PERCENTILES = (16.0, 84.0)
#: A spectrum's peak and its per-scan rows are one peak when their m/z agree
#: within this (Da): `ms2_scan_peaks.parquet` carries each peak's m/z as the
#: spectra table does (the same float), and two peaks of one spectrum are
#: never this close (the averaging seeds split at 1 ppm).
MATCH_DA = 1e-6
#: The factor (averaged height / per-scan mean) a peak's per-scan readings
#: may be scaled by (`spectrum_scans`), both ends in. It is 0.89-1.01 on our
#: own standard A's S/N-10 spectra and 0.85-1.14 on our own three files'
#: S/N-3 spectra, where the rows are the server's peaks read scan by scan; a
#: peak that a stray centroid reads in one of n scans has a factor of about
#: n, and rows read off another m/z axis than the spectra's (a calibration
#: the rows do not share) read a few peaks partly, at factors of several.
#: Outside the band the rows are not the peak's: it is held.
SCALE_BAND = (0.5, 2.0)
#: An E50 band whose bounds agree to this share of their size (or 1e-9 NCE)
#: is of zero width (`zero_width`): bounds read off resamples of held
#: readings differ in their last bits only (a resample's counts / n do not
#: add up to exactly 1).
ZERO_WIDTH_REL = 1e-9

#: `ms2_replicates.csv`: per (window, spectrum, item), an item being a window
#: ion's precursor or a fragment (one name across the energies, as
#: `ms2_fragments.csv`): its scans, how many hold it, its mean (on the
#: averaged spectrum's scale: the averaged intensity) and its CV over the
#: scans (absence = 0). `matched` False: a peak of it is held -- read in no
#: scan, or on another scale than its spectrum's (`SCALE_BAND`) -- whether
#: or not another peak of it (a split peak, summed) is read: `n_present` 0,
#: presence, sd and CV NaN.
REPLICATE_COLUMNS: dict[str, str] = {
    "parent_mz": "float64", "spectrum": "str", "nce": "float64", "hcd_v": "float64",
    "item": "str", "name": "str", "n_scans": "int64", "n_present": "int64",
    "presence": "float64", "mean": "float64", "sd": "float64", "cv": "float64",
    "matched": "bool",
}


@dataclass
class SpectrumScans:
    """One spectrum's peaks in its scans: `matrix` peaks x scans on the
    averaged spectrum's scale (rows in the spectrum's peak order), `present`
    the same shape (a scan holds the peak), `matched` per peak (read in the
    scans; an unmatched peak is held at its averaged height), `scans` the
    scan numbers (ascending), `factor` per peak the averaged height over its
    per-scan mean (NaN where no scan reads it)."""
    spectrum: str
    scans: np.ndarray
    matrix: np.ndarray
    present: np.ndarray
    matched: np.ndarray
    factor: np.ndarray

    @property
    def n(self) -> int:
        return int(self.matrix.shape[1])


def spectrum_scans(spectrum: str, peak_mz, peak_intensity, rows: pd.DataFrame) -> SpectrumScans:
    """A spectrum's peaks (`peak_mz`, `peak_intensity`: the spectra table's,
    in its order) read in the scans of `rows` (its `ms2_scan_peaks.parquet`
    rows). A peak matches the rows whose `peak_mz` lies within `MATCH_DA`;
    its per-scan readings x are scaled to x I_avg / mean(x) when that factor
    lies within `SCALE_BAND`. A peak no row matches, whose rows read 0 in
    every scan, or whose factor lies outside the band (the rows are not the
    peak read scan by scan: one stray centroid in one of n scans reads it n
    times too high) is held at I_avg in every scan (`matched` False) and is
    present in none: no scan's reading is used. No rows at all: one
    pseudo-scan, the averaged spectrum (n 0 scans said by the caller), no
    peak matched or present."""
    mz = np.asarray(peak_mz, dtype=float)
    avg = np.asarray(peak_intensity, dtype=float)
    scans = np.sort(rows["scan"].astype("int64").unique()) if len(rows) else np.zeros(0, "int64")
    n_p, n_s = len(mz), len(scans)
    factor = np.full(n_p, np.nan)
    if not n_s:
        return SpectrumScans(str(spectrum), scans, avg[:, None].copy(),
                             np.zeros((n_p, 1), dtype=bool), np.zeros(n_p, dtype=bool), factor)
    col = {int(s): j for j, s in enumerate(scans)}
    pmz = rows["peak_mz"].to_numpy(dtype=float)
    uniq = np.unique(pmz)
    raw = np.zeros((len(uniq), n_s))
    ui = np.searchsorted(uniq, pmz)
    sj = np.asarray([col[int(s)] for s in rows["scan"]], dtype=np.int64)
    np.add.at(raw, (ui, sj), rows["intensity"].to_numpy(dtype=float))
    matrix = np.repeat(avg[:, None], n_s, axis=1)
    present = np.ones((n_p, n_s), dtype=bool)
    matched = np.zeros(n_p, dtype=bool)
    if len(uniq):
        k = np.clip(np.searchsorted(uniq, mz), 0, len(uniq) - 1)
        k0 = np.clip(k - 1, 0, len(uniq) - 1)
        near = np.where(np.abs(uniq[k0] - mz) < np.abs(uniq[k] - mz), k0, k)
        hit = np.abs(uniq[near] - mz) <= MATCH_DA
        x = raw[near]
        mean = x.mean(axis=1)
        read = hit & (mean > 0) & np.isfinite(avg)
        factor[read] = avg[read] / mean[read]
        lo, hi = SCALE_BAND
        with np.errstate(invalid="ignore"):
            ok = read & (factor >= lo) & (factor <= hi)
        matrix[ok] = x[ok] * factor[ok][:, None]
        present[ok] = x[ok] > 0
        present[~ok] = False
        matched = ok
    return SpectrumScans(str(spectrum), scans, matrix, present, matched, factor)


def window_scans(roles: pd.DataFrame, by_spectrum: Mapping[str, SpectrumScans],
                 spectra: Sequence[str], ions: Sequence[str], keys: Sequence[str]
                 ) -> dict[str, tuple[np.ndarray, ...]]:
    """Per spectrum of `spectra`: (ions x scans, fragments x scans) intensity
    matrices summed over each item's peaks exactly as
    `attribute.intensity_matrices` sums the averaged ones (`roles`: the
    breakdown's -- side lobes and harmonics already set aside), then the
    same shapes' presence (any of the item's peaks in the scan), then per
    ion / fragment whether every peak of it was matched in the scans. Rows
    follow `ions` / `keys` (the fragment columns of the averaged matrices)."""
    out = {}
    ion_pos = {f: i for i, f in enumerate(ions)}
    key_pos = {k: i for i, k in enumerate(keys)}
    for spec in spectra:
        sc = by_spectrum[spec]
        r = roles[roles["spectrum"] == spec]
        n = sc.n
        im = np.zeros((len(ions), n))
        fm = np.zeros((len(keys), n))
        ip = np.zeros((len(ions), n), dtype=bool)
        fp = np.zeros((len(keys), n), dtype=bool)
        ok_i = np.ones(len(ions), dtype=bool)
        ok_f = np.ones(len(keys), dtype=bool)
        for role, name, peak in zip(r["role"], r["ion"].where(r["role"] == "precursor", r["key"]),
                                    r["peak"]):
            if role == "precursor" and name in ion_pos:
                a, b, ok, i = im, ip, ok_i, ion_pos[name]
            elif role == "fragment" and name in key_pos:
                a, b, ok, i = fm, fp, ok_f, key_pos[name]
            else:
                continue
            p = int(peak)
            a[i] += np.nan_to_num(sc.matrix[p])
            b[i] |= sc.present[p]
            ok[i] &= bool(sc.matched[p])
        out[spec] = (im, fm, ip, fp, ok_i, ok_f)
    return out


def _weights_matrix(weights: Mapping[str, Mapping[str, float]],
                    candidates: Mapping[str, Sequence[str]], ions: Sequence[str],
                    keys: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    """(fragments x ions weights, fragments x ions composition 0/1), as
    `breakdown.survival` builds them."""
    w = np.zeros((len(keys), len(ions)))
    c = np.zeros((len(keys), len(ions)))
    pos = {f: i for i, f in enumerate(ions)}
    for k, key in enumerate(keys):
        for ion, wx in weights.get(key, {}).items():
            if ion in pos:
                w[k, pos[ion]] = float(wx)
        for ion in candidates.get(key, ()):
            if ion in pos:
                c[k, pos[ion]] = 1.0
    return w, c


def spectrum_curves(ion_m: np.ndarray, frag_m: np.ndarray, w: np.ndarray, c: np.ndarray,
                    ref: np.ndarray) -> dict[str, np.ndarray]:
    """F, F_lo and S of every window ion in every column of `ion_m` (ions x
    columns; `frag_m` fragments x columns): the rule of `breakdown.survival`
    column by column -- 0/0 (no precursor, no fragment it can make) NaN for
    all three, S NaN where `ref` (the ion's lowest-energy precursor) is 0.
    A column is a scan, or a resample's mean."""
    fw = w.T @ frag_m if len(frag_m) else np.zeros_like(ion_m)
    fa = c.T @ frag_m if len(frag_m) else np.zeros_like(ion_m)
    with np.errstate(invalid="ignore", divide="ignore"):
        f = ion_m / (ion_m + fw)
        f_lo = ion_m / (ion_m + fa)
        s = ion_m / np.asarray(ref, dtype=float)[:, None]
    s = np.where(np.isfinite(s) & (ion_m + fa > 0), s, np.nan)
    return {"f": f, "f_lo": f_lo, "s": s}


def spread(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Row-wise (sd, sem, n finite) over the finite values (ddof 1; NaN with
    fewer than two)."""
    v = np.asarray(values, dtype=float)
    ok = np.isfinite(v)
    n = ok.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(n > 0, np.where(ok, v, 0.0).sum(axis=1) / np.maximum(n, 1), np.nan)
        ss = np.where(ok, (v - mean[:, None]) ** 2, 0.0).sum(axis=1)
        sd = np.where(n >= 2, np.sqrt(ss / np.maximum(n - 1, 1)), np.nan)
        sem = np.where(n >= 2, sd / np.sqrt(np.maximum(n, 1)), np.nan)
    return sd, sem, n


def window_seed(parent_mz: float, seed: int = BOOTSTRAP_SEED) -> np.random.SeedSequence:
    """A window's own seed: `seed` and its parent key (to 1e-4 Da, the
    precision a key is typed with)."""
    return np.random.SeedSequence([int(seed), int(round(float(parent_mz) * 1e4))])


def bootstrap_e50(nce: Sequence[float], volts: Sequence[float],
                  per_spectrum: Sequence[tuple[np.ndarray, np.ndarray]], w: np.ndarray,
                  c: np.ndarray, *, b: int = BOOTSTRAP_B,
                  seed: np.random.SeedSequence | int = BOOTSTRAP_SEED
                  ) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, tuple]]:
    """The E50 of every window ion and curve, read off `b` resamples.

    `per_spectrum`: (ions x scans, fragments x scans) per energy, in the
    breakdown's order (lowest NCE first) with its `nce` / `volts`. Each
    resample draws every energy's scans with replacement (as many as it
    has) and takes their means -- the precursor and its fragments from the
    same scans -- and rebuilds the curves (`spectrum_curves`: S against the
    resample's own lowest energy). A resample's means are its scans' counts
    (resamples x scans, / n) times the scans' intensities: the memory is
    resamples x (scans + items), never an items x resamples x scans array
    (1000 resamples of 1000 scans of 100 fragments would hold 0.8 GB).
    -> {curve: (E50 NCE, E50 V, flag code, (NCE before, NCE after, volts
    before, volts after) of the crossing), each ions x b; the bracket is
    `breakdown.e50_many`'s, where a 'gap' resample's E50 lies}.

    A hole at an end of the energies: a resample observes only the energies
    the averaged curve (every scan once) observes, and where all the scans
    it draws at the averaged curve's first energy hold nothing of the ion
    (no precursor, nothing it makes) it does not observe the curve there. Its
    'below_range' -- already below 0.5 at the first energy IT observes --
    then says only that its E50 lies at or below that energy, not below
    every energy: it is a 'gap', its E50 in [-inf, that energy] (the
    bracket's NCE before and volts before -inf). The mirror at the last
    energy: an 'above_range' that does not observe it is a 'gap' in [the
    last energy it observes, +inf]. A resample that observes both ends is
    read as `e50_many` reads it.
    Where the averaged curve observes two energies or more, a resample
    left ONE of them ('single_energy': its holes took the others) still
    reads which side of 0.5 it lies on there: below 0.5 at the averaged
    curve's first energy it is below every energy ('below_range'), below
    0.5 at a later one a 'gap' in [-inf, that energy]; at or above 0.5 at
    the last energy above every energy ('above_range'), at an earlier one a
    'gap' in [that energy, +inf]. One left none ('no_data') is a 'gap' in
    [-inf, +inf]. So 'single_energy' / 'no_data' come back only for a curve
    the averaged spectrum itself observes at fewer than two energies: no
    resample is dropped from the ranking for its holes (`interval`).
    Deterministic: one generator from `seed`, the draws in energy order."""
    rng = np.random.default_rng(seed)
    means = []
    for ion_m, frag_m in per_spectrum:
        n = ion_m.shape[1]
        if n == 0:
            raise ValueError("a spectrum with no scan column")
        if n > 1:
            idx = rng.integers(0, n, size=(b, n))
            flat = (idx + n * np.arange(b, dtype=np.int64)[:, None]).ravel()
            del idx
            cnt = np.bincount(flat, minlength=b * n).reshape(b, n) / float(n)
            del flat
        else:
            cnt = np.ones((b, 1))
        im = ion_m @ cnt.T                                 # ions x b
        fm = frag_m @ cnt.T if len(frag_m) else np.zeros((0, b))
        means.append((im, fm))
    n_ions = per_spectrum[0][0].shape[0] if per_spectrum else 0
    curves = _resampled_curves(means, w, c, n_ions, b)
    # the averaged curves (every scan once): the energies a resample can observe
    avg = _resampled_curves([(ion_m.mean(axis=1, keepdims=True),
                              frag_m.mean(axis=1, keepdims=True) if len(frag_m)
                              else np.zeros((0, 1))) for ion_m, frag_m in per_spectrum],
                            w, c, n_ions, 1)
    fi = BD.FLAGS.index
    out = {}
    for k, arr in curves.items():
        e_n = np.full((n_ions, b), np.nan)
        e_v = np.full((n_ions, b), np.nan)
        code = np.zeros((n_ions, b), dtype=np.int64)
        br = tuple(np.full((n_ions, b), np.nan) for _ in range(4))
        for i in range(n_ions):
            e_n[i], e_v[i], code[i], bi = BD.e50_many(nce, arr[:, i, :], volts, bracket=True)
            for a, v in zip(br, bi):
                a[i] = v
            x0, x1, v0, v1, y0, _ = BD.observed_ends(nce, arr[:, i, :], volts,
                                                     with_values=True)
            a0, a1, _, _ = BD.observed_ends(nce, avg[k][:, i, :], volts)
            ci = code[i]                                     # a view: set in place
            with np.errstate(invalid="ignore"):
                ends = bool(a1[0] > a0[0])                   # the averaged curve: 2+ energies
                one = ends & (ci == fi(BD.FLAG_SINGLE))      # one energy left: x0 == x1
                lo, hi = one & (y0 < BD.E50_THRESHOLD), one & (y0 >= BD.E50_THRESHOLD)
                start = ((ci == fi(BD.FLAG_BELOW)) | lo) & (x0 > a0[0])
                end = ((ci == fi(BD.FLAG_ABOVE)) | hi) & (x1 < a1[0])
                nowhere = ends & (ci == fi(BD.FLAG_NO_DATA))
            ci[lo & ~start] = fi(BD.FLAG_BELOW)             # below at the curve's first energy
            ci[hi & ~end] = fi(BD.FLAG_ABOVE)               # above at its last
            ci[start | end | nowhere] = fi(BD.FLAG_GAP)
            xa, xb, va, vb = (a[i] for a in br)              # views: set in place
            xa[start], xb[start], va[start], vb[start] = -np.inf, x0[start], -np.inf, v0[start]
            xa[end], xb[end], va[end], vb[end] = x1[end], np.inf, v1[end], np.inf
            xa[nowhere], xb[nowhere], va[nowhere], vb[nowhere] = -np.inf, np.inf, -np.inf, np.inf
        out[k] = (e_n, e_v, code, br)
    return out


def _resampled_curves(means: Sequence[tuple[np.ndarray, np.ndarray]], w: np.ndarray,
                      c: np.ndarray, n_ions: int, b: int) -> dict[str, np.ndarray]:
    """F, F_lo and S (energies x ions x columns) from each energy's (ions x
    columns, fragments x columns) means: `spectrum_curves`' rule, S against
    the columns' own lowest energy."""
    curves = {k: np.full((len(means), n_ions, b), np.nan) for k in ("f", "f_lo", "s")}
    ref = means[0][0] if means else np.zeros((n_ions, b))
    for e, (im, fm) in enumerate(means):
        fw = w.T @ fm if len(fm) else np.zeros_like(im)
        fa = c.T @ fm if len(fm) else np.zeros_like(im)
        with np.errstate(invalid="ignore", divide="ignore"):
            curves["f"][e] = im / (im + fw)
            curves["f_lo"][e] = im / (im + fa)
            s = im / ref
        curves["s"][e] = np.where(np.isfinite(s) & (im + fa > 0), s, np.nan)
    return curves


def _pct(sorted_vals: np.ndarray, q: float) -> float:
    """The q-th fraction quantile (linear between order statistics) of
    sorted values that may hold -inf / +inf (censored resamples), its sign
    kept: -inf / +inf when an order statistic it needs is infinite (the
    bound lies below / above the measured energies; between a -inf and a
    +inf, the nearer one's). NaN with no value, or a NaN one."""
    n = len(sorted_vals)
    if not n:
        return math.nan
    pos = q * (n - 1)
    i = int(math.floor(pos))
    g = pos - i
    a = float(sorted_vals[i])
    bb = float(sorted_vals[min(i + 1, n - 1)])
    if math.isnan(a) or math.isnan(bb):
        return math.nan
    if g == 0.0 or a == bb:
        return a
    if math.isinf(a) and math.isinf(bb):
        return a if g < 0.5 else bb
    if math.isinf(a) or math.isinf(bb):
        return a if math.isinf(a) else bb
    return a + (bb - a) * g


def _bound(ends: Sequence[tuple[np.ndarray, np.ndarray]], q: float, *,
           upper: bool) -> tuple[float, str]:
    """One percentile of resamples each known to lie in [lower end, upper
    end]: `ends` the two ends' values, each sorted, with which of them are a
    hole's ('gap' resamples). The true quantile lies between the two ends'
    quantiles. -> (bound, where): NaN and where it lies when the order
    statistics it needs are censored -- 'below' the lowest energy (on both
    ends: so every resample it needs is), 'above' the highest (likewise),
    'gap' (a resample whose E50 is known only to lie across energies it does
    not observe: between the two about an unobserved one, or at or below the
    first energy it observes, at or above the last, where the averaged curve
    observes more -- one end of the quantile infinite, the other not, is
    such holes at an end); else the bound, the outer of the two quantiles
    (the lower end's for a lower bound, the upper end's for an upper one: a
    hole resample that only moves the rank of an interpolated E50 widens the
    band, never narrows it)."""
    (lv, lg), (uv, ug) = ends
    a, b = _pct(lv, q), _pct(uv, q)
    if math.isnan(a) or math.isnan(b):
        return math.nan, ""
    if b == -math.inf:                                  # a <= b: both ends below
        return math.nan, "below"
    if a == math.inf:                                   # both ends above
        return math.nan, "above"
    if a == -math.inf or b == math.inf:                 # holes at an end decide it
        return math.nan, "gap"
    pos = q * (len(lv) - 1)
    i = int(math.floor(pos))
    used = [i] if pos == i else [i, min(i + 1, len(lv) - 1)]
    if lg[used].any() or ug[used].any():
        return math.nan, "gap"
    return (b if upper else a), ""


def interval(e_nce: np.ndarray, e_v: np.ndarray, code: np.ndarray,
             percentiles: Sequence[float] = BOOTSTRAP_PERCENTILES, *,
             bracket: Sequence[np.ndarray] | None = None) -> dict:
    """One curve's resamples -> {nce_lo, nce_hi, v_lo, v_hi; the shares
    cross (interpolated), below, above, gap (the holes: between two
    observed energies, at an end, or about every energy) and none (no
    answer: one energy observed or none -- from `bootstrap_e50` only for a
    curve the averaged spectrum observes at fewer than two energies), which
    add up to 1; lo_where / hi_where ('' or where a NaN bound lies: 'below'
    the lowest energy, 'above' the highest, 'gap' across an unobserved
    energy), lo_censored / hi_censored}.

    Every resample with an answer is ranked, each as an interval it is known
    to lie in: an interpolated E50 is itself; a resample already below 0.5
    at the lowest energy lies below every energy (-inf), one never below 0.5
    above every energy (+inf); a 'gap' resample crosses across an
    unobserved energy, its E50 somewhere between the two observed energies
    about the hole (`bracket`: `bootstrap_e50`'s NCE before / after, volts
    before / after; needed when a resample is a gap) -- or it is a hole at
    an end of the energies, its E50 in [-inf, the first energy it observes]
    or [the last, +inf] (`bootstrap_e50`: the bracket's outer end infinite),
    or in [-inf, +inf] where it observes none.
    A percentile that falls among the resamples beyond the energies or
    across a hole is NaN, and `*_where` says on which side -- never a bound
    read off the other resamples alone; one that falls on an interpolated
    E50 whose rank a few hole resamples could move is the outer reading
    (`_bound`). The volts bounds need the volts of every crossing and every
    hole's observed energies (NaN otherwise)."""
    lo_q, hi_q = (p / 100.0 for p in percentiles)
    fi = BD.FLAGS.index
    code = np.asarray(code)
    below, above = code == fi(BD.FLAG_BELOW), code == fi(BD.FLAG_ABOVE)
    inter, gap = code == fi(BD.FLAG_INTERPOLATED), code == fi(BD.FLAG_GAP)
    answered = below | above | inter | gap
    b = len(code)
    res = {k: (float(m.mean()) if b else math.nan)
           for k, m in (("cross", inter), ("below", below), ("above", above), ("gap", gap),
                        ("none", ~answered))}
    if gap.any() and bracket is None:
        raise ValueError("a resample that crosses across an unobserved energy needs its "
                         "bracket (bootstrap_e50 / breakdown.e50_many(..., bracket=True))")
    xa, xb, va, vb = ((np.full(b, np.nan),) * 4 if bracket is None
                      else tuple(np.asarray(a, dtype=float) for a in bracket))
    e_nce, e_v = np.asarray(e_nce, dtype=float), np.asarray(e_v, dtype=float)
    # a hole's volts: known at the energies it observes (an end hole's outer
    # end is -inf / +inf, no volts needed there)
    volts_known = bool(np.isfinite(e_v[inter]).all() and not np.isnan(va[gap]).any()
                       and not np.isnan(vb[gap]).any())
    for name, vals, ga, gb in (("nce", e_nce, xa, xb), ("v", e_v, va, vb)):
        ends = []
        for g in (ga, gb):
            v = np.where(below, -np.inf, np.where(above, np.inf, np.where(inter, vals, g)))
            v = v[answered]
            o = np.argsort(v, kind="stable")
            ends.append((v[o], gap[answered][o]))
        for side, q in (("lo", lo_q), ("hi", hi_q)):
            value, where = _bound(ends, q, upper=side == "hi")
            if name == "nce":
                res[f"nce_{side}"], res[f"{side}_where"] = value, where
            else:
                res[f"v_{side}"] = value if volts_known else math.nan
    res["lo_censored"] = bool(res["lo_where"])
    res["hi_censored"] = bool(res["hi_where"])
    return res


def zero_width(lo: float, hi: float) -> bool:
    """Is the band [lo, hi] of zero width: both bounds finite and equal to
    `ZERO_WIDTH_REL` of their size (1e-9 NCE at least)? Not `lo == hi`: the
    bounds of resamples that read held peaks differ in their last bits
    (6.810663806263728 against ...729), and a band no scan's scatter widens
    is no measurement whether its floats agree or not."""
    lo, hi = float(lo), float(hi)
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return False
    return abs(hi - lo) <= ZERO_WIDTH_REL * max(1.0, abs(lo), abs(hi))
