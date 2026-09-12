"""Admission: which peaks are ELIGIBLE for formula search -- persistence OR brightness.

Every pass that hunts for new formulas (pass-1 grid enumeration, the ladder /
siloxane / residual explainers) draws its candidate peaks through one gate. It
used to be brightness alone (`height >= cfg.height_cutoff`), and on a
low-sensitivity TOF that threw away almost all of the real chemistry: measured on
a 230-spectrum mixed-reagent TOF batch, an absolute 100 cps cutoff kept 32 of
4025 m/z bins, while ~500 bins recur in >= 80 % of the spectra at a median 3-4 cps.
Noise does not recur at a fixed m/z across hundreds of spectra; ions do. Probing
that batch at the exact masses of 21 highly oxygenated molecules an Orbitrap saw
on the same air hit 21/21 (present in a median 93 % of spectra, ~2 cps) while
mass-shifted decoys hit ~0 -- so the recurrent weak population is real, and it
sits 50x below the old cutoff.

Occurrence and brightness do not trade off (no bright bin is transient), so the
persistence path is purely ADDITIVE: it admits peaks for consideration and does
not lower the bar for confirming them. At 2-3 cps the isotopologues are
sub-count, so an occurrence-admitted peak normally lands as a Candidate -- that
is correct, and the isotope rules are untouched.

  occurrence(bin) = fraction of the batch's spectra in which that m/z bin holds a
                    peak (bins = `timeseries.build_matrix` at `sampling.
                    BATCH_TOL_PPM`, the one tolerance the selection, this table
                    and the merge share -- so a peak, its bin and its merged
                    cluster are the same object at the same m/z rule)
  admitted        = height >= cfg.height_cutoff  OR  occurrence >= threshold
                    (threshold = `resolve_threshold`: the number given as
                    `cfg.occurrence_min`, or the batch's Otsu split for "auto";
                    stored on `cfg.occurrence_threshold`)
  admitted_by     = 'height' | 'occurrence' (persistence only) | '' (below the
                    gate the gated passes use)

WHICH PASSES ARE GATED. `admissible()` is consulted by exactly four sites:
pass-1 grid enumeration (passes/directors), the pass-6 ladder gap-fill
(ladders), residual stage B (residual) and the siloxane ladder (siloxane, both
the work set and the seed test). `admitted_by` describes eligibility at THOSE
sites. Not yet gated (still brightness-only, `height >= cfg.height_cutoff`):
pass-2/3 series growth, residual stage A (the ~2-Da isotope-doublet scan) and
the reflist rescue -- a documented follow-up, not a promise this module makes.

The threshold is DERIVED FROM THE BATCH, not a constant: the bin-occurrence
distribution is cleanly bimodal (transient bins pile up below 0.1, persistent
ones above 0.9, a thin middle) on every instrument measured, and Otsu's split of
it lands at 0.40-0.55 on all of them. A fixed 0.8 was tried first and discarded
two-thirds of the recurring weak ions the path exists to recover (their
occurrence spreads 0.5-0.95); a fixed low value would not transfer either. So
`occurrence_min="auto"` = Otsu's threshold clamped to [AUTO_MIN, AUTO_MAX]; a
number overrides it; `0` disables the path. Fewer than MIN_SPECTRA spectra give
too coarse an occurrence to trust, and the path is off.

Single-sample runs have no batch context: without an occurrence table the gate
is brightness alone, unchanged.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from peaky.batch import sampling as SS

__version__ = "0.3.0"  # one batch tolerance (BATCH_TOL_PPM); lookup requires the
                       # table's tol; why_off() explains a disabled path

DEFAULT_OCCURRENCE_MIN = "auto"   # Otsu split of the batch's bin-occurrence distribution
AUTO_MIN, AUTO_MAX = 0.25, 0.75   # clamp for the derived threshold
MIN_SPECTRA = 10                  # below this, occurrence is too coarse -> path off
ADMIT_HEIGHT = "height"
ADMIT_OCCURRENCE = "occurrence"
ADMIT_NONE = ""


def bin_occurrence(ts_peaks: pd.DataFrame, *, tol_ppm: float = SS.BATCH_TOL_PPM,
                   sample_col: str = "sample_item_id", mz_col: str = "mz",
                   height_col: str = "height") -> pd.DataFrame:
    """Per-bin occurrence from the full-batch per-peak table: one row per m/z bin
    with `mz` (height-weighted bin centre), `occurrence` (fraction of samples in
    which the bin holds a peak) and `n_samples` (count). Bins come from
    `timeseries.build_matrix` (ppm gap-clustering) at `tol_ppm`, which defaults
    to `sampling.BATCH_TOL_PPM` -- the same tolerance the selection and the
    merge use, so `batch` and `assign --ts-batch` bin identically. `.attrs`
    carries `tol_ppm` (required by `lookup_occurrence`) and the batch's
    `n_samples`."""
    from peaky.batch import timeseries as TS

    tol = float(tol_ppm)
    cols = [c for c in (sample_col, mz_col, height_col) if c in ts_peaks.columns]
    if len(cols) < 3:
        raise ValueError(f"bin_occurrence needs {sample_col!r}, {mz_col!r}, {height_col!r} "
                         f"columns (got {list(ts_peaks.columns)[:8]})")
    mat, bin_mz = TS.build_matrix(ts_peaks[cols], tol_ppm=tol, mz_col=mz_col,
                                  height_col=height_col, sample_col=sample_col)
    if mat.shape[0] == 0 or mat.shape[1] == 0:
        out = pd.DataFrame({"mz": pd.Series(dtype=float), "occurrence": pd.Series(dtype=float),
                            "n_samples": pd.Series(dtype=int)})
        out.attrs.update(tol_ppm=tol, n_samples=int(mat.shape[0]))
        return out
    present = mat > 0
    out = pd.DataFrame({"mz": bin_mz.to_numpy(dtype=float),
                        "occurrence": present.mean(axis=0).to_numpy(dtype=float),
                        "n_samples": present.sum(axis=0).to_numpy(dtype=int)})
    out = out.sort_values("mz").reset_index(drop=True)
    out.attrs.update(tol_ppm=tol, n_samples=int(mat.shape[0]),
                     auto_threshold=auto_threshold(out["occurrence"].to_numpy()))
    return out


def otsu_threshold(x, *, n_bins: int = 50) -> float | None:
    """Otsu's split of a distribution on [0, 1]: the cut that maximises the
    between-class variance of the two sides. None for fewer than 2 values."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return None
    h, e = np.histogram(x, bins=n_bins, range=(0.0, 1.0))
    w = h / max(h.sum(), 1)
    c = (e[:-1] + e[1:]) / 2
    var = np.full(n_bins + 1, -1.0)
    for i in range(1, n_bins):
        w0, w1 = w[:i].sum(), w[i:].sum()
        if w0 <= 0 or w1 <= 0:
            continue
        m0 = (w[:i] * c[:i]).sum() / w0
        m1 = (w[i:] * c[i:]).sum() / w1
        var[i] = w0 * w1 * (m0 - m1) ** 2
    best = var.max()
    if best < 0:
        return None
    # every cut through an EMPTY stretch between the two modes is equally
    # optimal; take the middle of that stretch rather than its first edge
    opt = np.where(var >= best - 1e-12 * max(best, 1e-300))[0]
    return float(e[int(round((opt[0] + opt[-1]) / 2))])


def auto_threshold(occurrence) -> float | None:
    """The batch-derived persistence threshold: Otsu's split of the bin-occurrence
    distribution, clamped to [AUTO_MIN, AUTO_MAX]. None when undefined."""
    t = otsu_threshold(occurrence)
    if t is None:
        return None
    return float(min(max(t, AUTO_MIN), AUTO_MAX))


def resolve_threshold(cfg, table: pd.DataFrame | None) -> float | None:
    """The numeric persistence threshold for this run: `cfg.occurrence_min` when
    it is a number > 0, the table's auto threshold when it is 'auto'; None
    (path off) for 0 / no table / fewer than MIN_SPECTRA spectra."""
    om = getattr(cfg, "occurrence_min", DEFAULT_OCCURRENCE_MIN)
    if table is None or len(table) == 0:
        return None
    if int(table.attrs.get("n_samples", 0) or 0) < MIN_SPECTRA:
        return None
    if isinstance(om, str):
        if om.strip().lower() != "auto":
            raise ValueError(f"occurrence_min must be a number or 'auto', got {om!r}")
        t = table.attrs.get("auto_threshold")
        if t is None:
            t = auto_threshold(table["occurrence"].to_numpy())
        return float(t) if t is not None else None
    om = float(om)
    return om if om > 0 else None


def why_off(cfg, table: pd.DataFrame | None) -> str:
    """Why `resolve_threshold` gave no threshold, as one human phrase (`''` when
    it gave one). There are FOUR distinct reasons and a log line that says
    "path off" without saying which is unactionable -- a batch that is one
    spectrum short of MIN_SPECTRA and a batch whose occurrences carry no split
    need opposite responses. Every caller that reports the path being off should
    use this so `assign` and `batch` give the same account of the same run."""
    om = getattr(cfg, "occurrence_min", DEFAULT_OCCURRENCE_MIN)
    # the knob first: 0 disables the path whatever the table says
    if not isinstance(om, str) and (om is None or float(om) <= 0):
        return f"occurrence_min={om!r} -- the knob switches it off"
    if table is None or len(table) == 0:
        return ("no batch occurrence table (a single-sample run without "
                "--ts-batch has no batch context)")
    n = int(table.attrs.get("n_samples", 0) or 0)
    if n < MIN_SPECTRA:
        return f"only {n} spectra, fewer than the {MIN_SPECTRA} an occurrence needs to mean anything"
    if isinstance(om, str) and resolve_threshold(cfg, table) is None:
        return (f"{len(table)} bins over {n} spectra, but their occurrences carry no "
                "split for 'auto' to derive a threshold from")
    return ""


def lookup_occurrence(mz, table: pd.DataFrame | None) -> np.ndarray:
    """Occurrence of the nearest bin within the table's OWN binning tolerance of
    each m/z (NaN when no bin is that close, or without a table). The tolerance
    is `table.attrs["tol_ppm"]`, stamped by `bin_occurrence`, so a peak matches
    its bin by exactly the rule it was binned with; a table without it is
    refused rather than guessed at."""
    mz = np.asarray(mz, dtype=float)
    out = np.full(mz.shape, np.nan)
    if table is None or len(table) == 0:
        return out
    if table.attrs.get("tol_ppm") is None:
        raise ValueError("occurrence table lacks attrs['tol_ppm'] (build it with "
                         "admission.bin_occurrence, which stamps the binning tolerance)")
    tol = float(table.attrs["tol_ppm"])
    bm = table["mz"].to_numpy(dtype=float)
    occ = table["occurrence"].to_numpy(dtype=float)
    order = np.argsort(bm)
    bs, os_ = bm[order], occ[order]
    ok = np.isfinite(mz)
    if not ok.any():
        return out
    if len(bs) == 1:                      # one bin: nothing to bracket, it is the only candidate
        pick = np.zeros(int(ok.sum()), dtype=int)
    else:
        j = np.clip(np.searchsorted(bs, mz[ok]), 1, len(bs) - 1)
        left, right = j - 1, j
        pick = np.where(np.abs(bs[left] - mz[ok]) <= np.abs(bs[right] - mz[ok]), left, right)
    near = np.abs(bs[pick] - mz[ok]) / mz[ok] * 1e6 <= tol
    vals = np.where(near, os_[pick], np.nan)
    out[ok] = vals
    return out


def admissible(ledger: pd.DataFrame, cfg) -> pd.Series:
    """Boolean mask: `height >= cfg.height_cutoff` OR `occurrence >= threshold`,
    where the threshold is `cfg.occurrence_threshold` (resolved by
    `stamp_admission`) or, on an unstamped cfg, a numeric `cfg.occurrence_min`.
    The persistence path needs an `occurrence` column (stamped by
    `stamp_admission`) and a threshold; otherwise this is exactly the brightness
    gate."""
    hcut = float(getattr(cfg, "height_cutoff", 0.0) or 0.0)
    h = ledger["height"].fillna(0).astype(float) >= hcut if "height" in ledger.columns \
        else pd.Series(False, index=ledger.index)
    thr = getattr(cfg, "occurrence_threshold", None)
    if thr is None:                       # no stamped run: a numeric occurrence_min still counts
        om = getattr(cfg, "occurrence_min", 0.0)
        thr = float(om) if not isinstance(om, str) and om and float(om) > 0 else None
    if thr is not None and "occurrence" in ledger.columns:
        occ = pd.to_numeric(ledger["occurrence"], errors="coerce").fillna(-1.0)
        h = h | (occ >= float(thr))
    return h


def stamp_admission(ledger: pd.DataFrame, cfg, table: pd.DataFrame | None = None) -> dict:
    """Annotate the ledger IN PLACE with `occurrence` (its bin's occurrence, NaN
    without batch context) and `admitted_by` ('height' | 'occurrence' | ''), using
    the cfg's RESOLVED `height_cutoff` (call after the noise edge is stamped) and
    the persistence threshold resolved from `cfg.occurrence_min` + the table
    (stored on `cfg.occurrence_threshold` for `admissible`). Returns the counts
    {height, occurrence, rejected, n_bins, occurrence_min, occurrence_threshold,
    height_gate_cps} (the last = the resolved brightness gate in cps)."""
    ledger["occurrence"] = lookup_occurrence(ledger["mz"].to_numpy(dtype=float), table) \
        if "mz" in ledger.columns else np.nan
    hcut = float(getattr(cfg, "height_cutoff", 0.0) or 0.0)
    thr = resolve_threshold(cfg, table)
    try:
        cfg.occurrence_threshold = thr
    except AttributeError:
        pass
    by_h = ledger["height"].fillna(0).astype(float) >= hcut
    by_o = (pd.to_numeric(ledger["occurrence"], errors="coerce").fillna(-1.0) >= thr) \
        if thr is not None else pd.Series(False, index=ledger.index)
    ledger["admitted_by"] = np.where(by_h, ADMIT_HEIGHT, np.where(by_o, ADMIT_OCCURRENCE, ADMIT_NONE))
    return {"height": int(by_h.sum()), "occurrence": int((~by_h & by_o).sum()),
            "rejected": int((~by_h & ~by_o).sum()),
            "n_bins": int(len(table)) if table is not None else 0,
            "occurrence_min": getattr(cfg, "occurrence_min", None),
            "occurrence_threshold": thr, "height_gate_cps": hcut}
