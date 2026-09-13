"""Trace primitives over the FULL-BATCH peak list -- one m/z-sorted index and the
three questions every batch-level stage asks of it.

A *trace* is one ion followed across the batch's spectra: the set of peaks, at
most one per spectrum, that sit within a mass window around a common centre.
Three stages need exactly that object and used to approximate it three ways:

  * ADMISSION asked "does this m/z recur across the batch?" through a global
    gap-binner whose bins chain on a TOF (91 % of the persistent bins of a
    230-spectrum TOF batch were wider than the 6 ppm tolerance, one 134 ppm),
    then looked a peak up by a DIFFERENT rule (within tol of the bin's centre) --
    so a peak could fail to find its own bin, and the "persistence" of a chained
    bin was partly the chaining.
  * STAMPING asked "which spectra hold this ledger ion?" with a window centred on
    the ledger anchor -- an m/z minted from the few assigned samples and, on a
    TOF, snapped to theory by the assignment itself (a formula is only committed
    where a sample's draw lands near it): sd 1.8 ppm from theory for real anchors
    vs 5.7 ppm for a same-size anchor drawn at random from the same trace, while
    the ion's trace genuinely sits ~5.8 ppm away. 22.6 % of that batch's anchors
    were outside their own trace's window; re-centring lifted mean coverage from
    61 % to 74 % on 867 ions and moved an Orbitrap ledger by 0.11 ppm (a no-op,
    which is what validates the diagnosis).
  * The MERGE clustered per-file anchors, so two winner samples 8 ppm apart
    minted two competing rows for ONE ion (26.8 % of well-populated TOF rows
    shared a trace with another row).

This module is the one implementation of the trace question, so the three stages
cannot disagree with each other:

  occurrence_at(mz)   fraction of the batch's spectra holding a peak within
                      +-tol of mz (one peak per spectrum counts once)
  mean_shift(mz)      the local mode of the peak density near mz -- the trace's
                      own centre -- found by iterating the robust median of the
                      one-peak-per-spectrum window, never further than a drift
                      cap from where it started
  members(centre)     the trace: the nearest peak per spectrum within the window

The per-peak occurrence of EVERY batch peak comes from one O(n) sweep
(`occurrence`), so the persistence threshold can be derived from the whole
batch and a peak's own occurrence is read by exactly the rule it was derived
with -- construction and lookup are one function. Pure numpy/pandas.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__version__ = "0.1.0"

MZ_FLOOR_DA = 1.5e-3      # window half-width floor (annotate_peaks' own; absorbs the
                          # raw-vs-calibrated offset at low m/z). EVERY window here
                          # is +-max(tol ppm, this), so below m/z ~250 (at 6 ppm)
                          # the floor is what defines "within tolerance" -- for the
                          # occurrence, the trace and the stamp alike.
MEAN_SHIFT_MAX_ITER = 40
MEAN_SHIFT_CONVERGED_PPM = 0.02
MEAN_SHIFT_MIN_PEAKS = 3  # fewer peaks in the window than this -> nothing to shift to
SCATTER_MIN_MEMBERS = 20  # spectra a trace needs before its scatter is trusted
SCATTER_WIDEN_X = 2.0     # re-measure the scatter in a window this many times wider ...
SCATTER_WIDEN_AT = 3.0    # ... when this many sigmas of the first estimate reach the edge
                          # (a cloud truncated at +-1.5 sigma reads ~0.6 sigma, so a
                          # wide TOF cloud measures ~2.1-2.4 ppm through a 6 ppm window
                          # -- the trigger has to fire there; an Orbitrap's 0.1-0.6 never)
MAD_TO_SIGMA = 1.4826


def _halfwin(mz, tol_ppm: float) -> np.ndarray | float:
    return np.maximum(np.asarray(mz, dtype=float) * tol_ppm * 1e-6, MZ_FLOOR_DA)


class PeakIndex:
    """The batch's peaks sorted by m/z, with integer sample codes.

    Built once per batch from the per-peak time series (`sample_item_id`, `mz`,
    `height`); exact duplicate (sample, m/z) rows -- Mascope's match-expanded
    copies of one physical peak -- are dropped so a peak counts once. Sample codes
    come from a SORTED factorisation, so the same table always yields the same
    codes regardless of row order (deterministic downstream)."""

    def __init__(self, peaks: pd.DataFrame, *, tol_ppm: float,
                 mz_col: str = "mz", sample_col: str = "sample_item_id",
                 height_col: str = "height"):
        cols = [c for c in (sample_col, mz_col, height_col) if c in peaks.columns]
        if sample_col not in cols or mz_col not in cols:
            raise ValueError(f"PeakIndex needs {sample_col!r} and {mz_col!r} columns "
                             f"(got {list(peaks.columns)[:8]})")
        d = peaks[cols].dropna(subset=[sample_col, mz_col])
        d = d.drop_duplicates(subset=[sample_col, mz_col])
        d = d.sort_values(mz_col, kind="mergesort").reset_index(drop=True)
        self.tol_ppm = float(tol_ppm)
        self.mz = d[mz_col].to_numpy(dtype=float)
        codes, uniq = pd.factorize(d[sample_col], sort=True)
        self.sample = np.asarray(codes, dtype=np.int64)
        self.sample_ids = np.asarray(uniq)
        self.n_samples = int(len(uniq))
        if height_col in d.columns:
            self.height = pd.to_numeric(d[height_col], errors="coerce").to_numpy(dtype=float)
        else:
            self.height = np.full(len(d), np.nan)
        self._occ: np.ndarray | None = None
        self._edge: np.ndarray | None = None

    def __len__(self) -> int:
        return int(len(self.mz))

    # ------------------------------------------------------------------ windows
    def window(self, lo: float, hi: float) -> tuple[int, int]:
        """Index range [i0, i1) of the peaks with lo <= m/z <= hi."""
        return (int(np.searchsorted(self.mz, lo, side="left")),
                int(np.searchsorted(self.mz, hi, side="right")))

    def members(self, centre: float, tol_ppm: float | None = None) -> np.ndarray:
        """The trace at `centre`: indices of the nearest peak per spectrum within
        +-tol (one peak per spectrum -- a shoulder or split peak in the same
        window does not make the ion present twice)."""
        tol = self.tol_ppm if tol_ppm is None else float(tol_ppm)
        hw = float(_halfwin(centre, tol))
        i0, i1 = self.window(centre - hw, centre + hw)
        if i1 <= i0:
            return np.empty(0, dtype=np.int64)
        m, s = self.mz[i0:i1], self.sample[i0:i1]
        order = np.lexsort((np.abs(m - centre), s))     # per sample, nearest first
        ss = s[order]
        first = np.ones(len(order), dtype=bool)
        first[1:] = ss[1:] != ss[:-1]
        return (i0 + order[first]).astype(np.int64)

    def coverage_at(self, centre: float, tol_ppm: float | None = None) -> float:
        """Fraction of the batch's spectra holding a peak within +-tol of `centre`."""
        if not np.isfinite(centre) or self.n_samples == 0:
            return 0.0
        return float(len(self.members(centre, tol_ppm))) / self.n_samples

    def occurrence_at(self, mzs, tol_ppm: float | None = None) -> np.ndarray:
        """`coverage_at` for an array of probe m/z -- NaN where NO peak at all lies
        within +-tol (nothing there to be persistent or not)."""
        tol = self.tol_ppm if tol_ppm is None else float(tol_ppm)
        mzs = np.asarray(mzs, dtype=float)
        out = np.full(mzs.shape, np.nan)
        if not len(self.mz):
            return out
        hw = _halfwin(mzs, tol)
        lo = np.searchsorted(self.mz, mzs - hw, side="left")
        hi = np.searchsorted(self.mz, mzs + hw, side="right")
        for k in np.flatnonzero(np.isfinite(mzs) & (hi > lo)):
            out[k] = np.unique(self.sample[lo[k]:hi[k]]).size / self.n_samples
        return out

    # --------------------------------------------------------------- occurrence
    def occurrence(self, tol_ppm: float | None = None) -> np.ndarray:
        """Per-peak occurrence of EVERY indexed peak: the fraction of spectra with
        a peak within +-tol of it (itself included). One two-pointer sweep over
        the sorted list -- each peak enters and leaves the moving window once, so
        this is O(n) and a full field batch (~1e6 peaks) takes seconds. Cached for
        the index's own tolerance."""
        tol = self.tol_ppm if tol_ppm is None else float(tol_ppm)
        if tol == self.tol_ppm and self._occ is not None:
            return self._occ
        n = len(self.mz)
        out = np.zeros(n, dtype=float)
        if n and self.n_samples:
            hw = _halfwin(self.mz, tol)
            lo = np.searchsorted(self.mz, self.mz - hw, side="left").tolist()
            hi = np.searchsorted(self.mz, self.mz + hw, side="right").tolist()
            cnt = [0] * self.n_samples
            samp = self.sample.tolist()
            distinct = L = R = 0
            res = [0] * n
            for i in range(n):
                h = hi[i]
                while R < h:                    # admit peaks entering the window
                    s = samp[R]
                    if cnt[s] == 0:
                        distinct += 1
                    cnt[s] += 1
                    R += 1
                l = lo[i]
                while L < l:                    # retire peaks leaving it
                    s = samp[L]
                    cnt[s] -= 1
                    if cnt[s] == 0:
                        distinct -= 1
                    L += 1
                res[i] = distinct
            out = np.asarray(res, dtype=float) / self.n_samples
        if tol == self.tol_ppm:
            self._occ = out
        return out

    # ---------------------------------------------------------------- brightness
    def sample_edge(self, q: float = 0.01) -> np.ndarray:
        """Per-sample noise edge (the `q` quantile of that sample's heights, the
        same statistic assign.run stamps as noise_edge_cps), indexed by sample
        code. NaN for a sample with no finite height."""
        if self._edge is None:
            edge = np.full(self.n_samples, np.nan)
            ok = np.isfinite(self.height)
            if ok.any():
                g = pd.Series(self.height[ok]).groupby(self.sample[ok]).quantile(q)
                edge[g.index.to_numpy()] = g.to_numpy()
            self._edge = edge
        return self._edge

    def height_in_edges(self) -> np.ndarray:
        """Each peak's height as a multiple of its own sample's noise edge -- the
        one height scale that transfers between instruments (NaN where either is
        missing)."""
        edge = self.sample_edge()[self.sample]
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(edge > 0, self.height / edge, np.nan)

    # ------------------------------------------------------------- re-centring
    def mean_shift(self, mz0: float, *, tol_ppm: float | None = None,
                   max_drift_ppm: float, max_iter: int = MEAN_SHIFT_MAX_ITER,
                   min_peaks: int = MEAN_SHIFT_MIN_PEAKS) -> float:
        """The trace centre nearest `mz0`: iterate the robust median of the
        one-peak-per-spectrum window until it stops moving. Never returns a
        centre further than `max_drift_ppm` from `mz0` -- a walk that wants to go
        further stops at its last in-cap position, so a re-centring can only ever
        move an anchor onto a trace it plausibly labels, not onto a neighbour.
        Fewer than `min_peaks` in the window -> `mz0` unchanged."""
        tol = self.tol_ppm if tol_ppm is None else float(tol_ppm)
        c = float(mz0)
        if not np.isfinite(c) or not len(self.mz):
            return c
        for _ in range(max_iter):
            idx = self.members(c, tol)
            if len(idx) < min_peaks:
                break
            new = float(np.median(self.mz[idx]))
            if abs(new - mz0) / mz0 * 1e6 > max_drift_ppm:
                break
            if abs(new - c) / max(c, 1.0) * 1e6 < MEAN_SHIFT_CONVERGED_PPM:
                c = new
                break
            c = new
        return c

    def scatter_ppm(self, centre: float, tol_ppm: float | None = None,
                    min_members: int = SCATTER_MIN_MEMBERS,
                    widen_x: float = SCATTER_WIDEN_X) -> float:
        """Robust per-ion mass scatter: 1.4826 x MAD of the members' ppm offsets
        from `centre`. NaN for a trace with fewer than `min_members` spectra.

        A window of +-tol truncates a cloud wider than itself and the MAD then
        reads low (a 4 ppm sigma cloud seen through a +-6 ppm window measures
        ~2.4 ppm). So when the first estimate says the cloud fills the window
        (SCATTER_WIDEN_AT sigmas reach the edge) the scatter is re-measured in a
        window `widen_x` times wider. A narrow cloud (an Orbitrap's 0.1-0.6 ppm)
        never triggers the widening, so a neighbouring isobar a few mDa away can
        never be pulled into its estimate."""
        tol = self.tol_ppm if tol_ppm is None else float(tol_ppm)
        idx = self.members(centre, tol)
        if len(idx) < min_members:
            return float("nan")
        off = (self.mz[idx] - centre) / centre * 1e6
        sigma = float(MAD_TO_SIGMA * np.median(np.abs(off - np.median(off))))
        if widen_x > 1.0 and SCATTER_WIDEN_AT * sigma > tol:
            wide = self.members(centre, tol * widen_x)
            if len(wide) >= min_members:
                off = (self.mz[wide] - centre) / centre * 1e6
                sigma = float(MAD_TO_SIGMA * np.median(np.abs(off - np.median(off))))
        return sigma


SCATTER_Q = 0.75          # the per-trace scatter quantile that sizes a batch's window


def batch_scatter_ppm(index: PeakIndex, centres, *, tol_ppm: float | None = None,
                      min_members: int = SCATTER_MIN_MEMBERS, q: float = SCATTER_Q) -> float:
    """The per-ion mass scatter a stamping window has to fit: the `q` quantile
    (third quartile) of `scatter_ppm` over the given trace centres that are
    populated enough to measure. NaN when none is.

    Not the median: scatter grows as S/N falls (measured on a TOF batch, dim third
    of the traces 3.6 ppm vs bright third 1.2 ppm; median 2.2), and the dim traces
    are exactly the ones a window sized from the bright ones loses. Not the top
    decile either: on one Orbitrap mode it is 8 ppm -- a scan-edge pile-up 40x the
    median -- while the third quartile there is 0.34 ppm. Measured third quartiles:
    3.8-4.2 ppm on a TOF, 0.24-0.34 ppm on Orbitrap modes."""
    vals = [index.scatter_ppm(float(c), tol_ppm, min_members)
            for c in np.asarray(centres, dtype=float) if np.isfinite(c)]
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.quantile(vals, q)) if vals else float("nan")
