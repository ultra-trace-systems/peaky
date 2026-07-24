"""Time-resolved disposition layer.

A sum spectrum cannot tell a bright stable INLET/instrument contaminant from a
real ambient ANALYTE, nor a degenerate reagent-background cluster from chemistry.
A *time series* of the same sample over a day can: in a halide-CIMS the physically
meaningful quantity is the analyte normalised to the reagent ion (removes
instrument-sensitivity + reagent-flow common-mode drift), and then

  * a FLAT normalised trace (low coefficient of variation, no diel) == inlet /
    instrument background or a constant reagent cluster -- NOT ambient chemistry;
  * a VARIABLE trace that co-varies with a known chemical family == real ambient
    analyte.

This module ingests a batch's per-sample peak table, builds the reagent-normalised
intensity matrix, measures each peak's variability (`cv_norm`) and (optionally)
its correlation to reference family traces, and stamps a `ts_*` disposition onto
the ledger. It then applies CONSERVATIVE auto-actions: demote a flat di-bromide /
background-channel commit (TS-confirmed background) and flag inlet contaminants.
It never changes a formula -- only the tier/role annotation, with commentary.

All pure pandas/numpy; no network. Reference (2026-06-16 time-series unlock).
"""
from __future__ import annotations

import bisect
import os
import re

import numpy as np
import pandas as pd

from peaky.assignment import ledger as L

__version__ = "0.1.0"

DEFAULT_TOL_PPM = 5.0
FLAT_CV = 0.25          # cv_norm below this == flat / background
COVARY_R = 0.70         # correlation above this == co-varies with the family


def auto_bin_minutes(ts: pd.DataFrame, *, target_bins: int = 50,
                     time_col: str = "datetime_utc") -> int:
    """Time-bin width (minutes) for the correlation / cluster / Van Krevelen layer.

    Bins at the NATIVE sample cadence (median inter-sample spacing) so the traces
    are NOT downsampled. A coarse bin (the old span/target_bins ~= 29 min on a 24 h
    batch) smears sharp features -- zero-air periods, fast trends -- that drive the
    real co-variation, pushing genuinely-changing channels into the flat bucket
    (validated on the June-3 uronium batch: native 5-min recovered 1018 changing
    channels / 95 families vs 752 / 54 at 29-min). Floored at 1 min; falls back to
    span/target_bins only when per-sample times are unavailable (<3 samples).
    Shared by clustering + VK so they bin identically.

    The bin is rounded UP (ceil) to the cadence, never down: a bin narrower than the
    real inter-sample spacing aliases — the fixed-width grid periodically catches
    ZERO samples (a beat between the grid pitch and the slightly-irregular sample
    times), leaving empty bins that render as a spurious regular comb of drop-to-floor
    teeth. ceil guarantees the bin is >= the sample spacing, so every bin holds >= 1
    sample. (e.g. a 73 s cadence -> 2 min, not the aliasing 1 min.)"""
    if "sample_item_id" in ts.columns:
        t = pd.to_datetime(ts.drop_duplicates("sample_item_id")[time_col], utc=True)
    else:
        t = pd.to_datetime(ts[time_col], utc=True).drop_duplicates()
    t = t.dropna().sort_values()
    if len(t) >= 3:
        cadence_min = t.diff().dropna().dt.total_seconds().median() / 60.0
        if cadence_min > 0:
            return max(1, int(np.ceil(cadence_min)))
    span_min = (t.max() - t.min()).total_seconds() / 60.0 if len(t) >= 2 else 30.0
    return max(1, int(round(span_min / target_bins)))


# ---------------------------------------------------------------------------
# matrix construction
# ---------------------------------------------------------------------------
def build_matrix(peaks: pd.DataFrame, *, tol_ppm: float = DEFAULT_TOL_PPM,
                 mz_col="mz", height_col="height", sample_col="sample_item_id"
                 ) -> tuple[pd.DataFrame, pd.Series]:
    """Gap-cluster peaks into m/z bins (ppm tolerance) and pivot to a
    samples x bin intensity matrix. Returns (matrix, bin_mz)."""
    d = peaks[[sample_col, mz_col, height_col]].dropna().sort_values(mz_col).reset_index(drop=True)
    mz = d[mz_col].to_numpy()
    if len(mz) == 0:
        return pd.DataFrame(), pd.Series(dtype=float)
    gaps = np.diff(mz) / mz[:-1] * 1e6
    binid = np.zeros(len(mz), dtype=np.int64)
    binid[1:] = np.cumsum(gaps > tol_ppm)
    d["_bin"] = binid
    wsum = (d[mz_col] * d[height_col]).groupby(d["_bin"]).sum()
    hsum = d[height_col].groupby(d["_bin"]).sum()
    bin_mz = (wsum / hsum).rename("mz")
    mat = d.pivot_table(index=sample_col, columns="_bin", values=height_col, aggfunc="sum")
    return mat, bin_mz


def annotate_peaks(peaks: pd.DataFrame, ledger: pd.DataFrame, *,
                   tol_ppm: float = DEFAULT_TOL_PPM, mz_floor_da: float = 1.5e-3,
                   mz_col: str = "mz") -> pd.DataFrame:
    """Stamp every time-series peak with the assigned formula/channel of the nearest
    ledger ion within tolerance. Returns a COPY of ``peaks`` with four added columns:

      * ``neutral_formula`` -- the assigned neutral formula (or <NA> if unmatched)
      * ``adduct``          -- the ionisation channel (e.g. ``[M+H]+`` / ``[M+NH4]+``)
      * ``tier``            -- the ledger tier (Assigned / Candidate / ...)
      * ``ion_mz``          -- the matched ledger ion m/z (NaN if unmatched)

    A raw ts peak is matched to the *nearest* assigned ion whose m/z is within
    ``max(mz*tol_ppm*1e-6, mz_floor_da)`` -- the mDa floor absorbs the small
    raw-vs-calibrated offset at low m/z. Nearest-wins so dense regions don't
    mis-fan-out. Fully vectorised (searchsorted); safe on multi-million-row batch
    time-series. A peak matching no assigned ion keeps <NA>/NaN (unassigned)."""
    out = peaks.copy()
    n = len(out)
    nf = np.full(n, None, dtype=object)
    ad = np.full(n, None, dtype=object)
    ti = np.full(n, None, dtype=object)
    im = np.full(n, np.nan, dtype=float)
    cols = getattr(ledger, "columns", None)
    if n and cols is not None and "mz" in cols and mz_col in out.columns:
        led = ledger.dropna(subset=["mz"]).sort_values("mz").reset_index(drop=True)
        if len(led):
            lmz = led["mz"].to_numpy(dtype=float)
            pmz = pd.to_numeric(out[mz_col], errors="coerce").to_numpy(dtype=float)
            j = np.searchsorted(lmz, pmz)
            jl = np.clip(j - 1, 0, len(lmz) - 1)
            jr = np.clip(j, 0, len(lmz) - 1)
            near = np.where(np.abs(lmz[jl] - pmz) <= np.abs(lmz[jr] - pmz), jl, jr)
            tol = np.maximum(pmz * tol_ppm * 1e-6, mz_floor_da)
            ok = np.isfinite(pmz) & (np.abs(lmz[near] - pmz) <= tol)
            lnf = led["neutral_formula"].to_numpy()
            lad = (led["adduct"].to_numpy() if "adduct" in led.columns
                   else np.full(len(led), None, dtype=object))
            lti = (led["tier"].to_numpy() if "tier" in led.columns
                   else np.full(len(led), None, dtype=object))
            nf[ok] = lnf[near[ok]]
            ad[ok] = lad[near[ok]]
            ti[ok] = lti[near[ok]]
            im[ok] = lmz[near[ok]]
    out["neutral_formula"] = nf
    out["adduct"] = ad
    out["tier"] = ti
    out["ion_mz"] = im
    return out


def reagent_total(mat: pd.DataFrame, bin_mz: pd.Series, reagent_mzs, *, tol_ppm=8.0):
    """Per-sample sum of the reagent bins (the normaliser). reagent_mzs is a list
    of reagent ion m/z (e.g. the Br3- isotopologues)."""
    cols = []
    bm = bin_mz.sort_values()
    arr = bm.to_numpy(); idx = bm.index.to_numpy()
    for r in reagent_mzs:
        i = bisect.bisect_left(arr, r)
        for j in (i - 1, i):
            if 0 <= j < len(arr) and abs(arr[j] - r) / r * 1e6 <= tol_ppm:
                cols.append(idx[j])
    cols = [c for c in set(cols) if c in mat.columns]
    if not cols:
        return None
    return mat[cols].sum(axis=1)


def normalize(mat: pd.DataFrame, reagent_series) -> pd.DataFrame:
    """Divide every bin by the per-sample reagent total (concentration proxy)."""
    if reagent_series is None:
        return mat
    return mat.div(reagent_series.replace(0, np.nan), axis=0)


def bin_metrics(norm: pd.DataFrame, bin_mz: pd.Series) -> pd.DataFrame:
    """Per-bin presence + cv_norm on the (reagent-normalised) matrix."""
    n = len(norm)
    presence = norm.notna().sum() / n if n else norm.notna().sum()
    mean = norm.mean(); std = norm.std()
    cv = (std / mean).replace([np.inf, -np.inf], np.nan)
    out = pd.DataFrame({"mz": bin_mz.reindex(norm.columns), "presence": presence,
                        "median": norm.median(), "cv_norm": cv})
    out.index.name = "_bin"
    return out


def family_trace(norm: pd.DataFrame, bin_ids):
    """z-scored mean log-trace of a set of bins (a reference family trace)."""
    bb = [b for b in bin_ids if b in norm.columns]
    if not bb:
        return None
    lg = np.log10(norm[bb].clip(lower=norm[norm > 0].min().min() or 1e-9))
    z = (lg - lg.mean()) / lg.std()
    return z.mean(axis=1)


def correlate(norm: pd.DataFrame, trace) -> pd.Series:
    if trace is None:
        return pd.Series(np.nan, index=norm.columns)
    lg = np.log10(norm.clip(lower=norm[norm > 0].min().min() or 1e-9))
    with np.errstate(invalid="ignore", divide="ignore"):  # flat bins -> NaN r (fine)
        return lg.apply(lambda c: c.corr(trace))


# ---------------------------------------------------------------------------
# disposition + ledger application
# ---------------------------------------------------------------------------
def _nearest(bin_mz_sorted_vals, bin_mz_sorted_idx, mz, tol_ppm):
    i = bisect.bisect_left(bin_mz_sorted_vals, mz)
    best = None
    for j in (i - 1, i):
        if 0 <= j < len(bin_mz_sorted_vals):
            ppm = abs(bin_mz_sorted_vals[j] - mz) / mz * 1e6
            if ppm <= tol_ppm and (best is None or ppm < best[1]):
                best = (bin_mz_sorted_idx[j], ppm)
    return best[0] if best else None


def _disposition(row, cv, r_mono, r_formic):
    """Classify one M0 row from its formula + time-series behavior."""
    ion = str(row.get("ion_formula", "")); adduct = str(row.get("adduct", ""))
    neutral = str(row.get("neutral_formula", ""))
    if "CO3" in adduct:
        return "background:CO3-channel (TS-flat)" if (pd.notna(cv) and cv < FLAT_CV) else "CO3-channel"
    if re.search(r"Br[23]", ion):
        return "background:di-bromide cluster (TS-flat)" if (pd.notna(cv) and cv < FLAT_CV) else "di-bromide"
    if pd.notna(cv) and cv < FLAT_CV:
        if "Si" in neutral or "F" in ion:
            return "background:inlet/instrument contaminant (TS-flat)"
        return "background:flat (TS-flat)"
    if pd.notna(r_mono) and r_mono >= COVARY_R:
        return "ambient:biogenic-SOA (co-varies)"
    if pd.notna(r_formic) and r_formic >= 0.9:
        return "ambient:acid/oxygenate pool (co-varies)"
    if pd.notna(cv) and cv >= 0.45:
        return "ambient:variable"
    return "intermediate"


def apply_timeseries(ledger: pd.DataFrame, peaks: pd.DataFrame, *,
                     reagent_mzs=None, mono_anchor_mzs=None, formic_mz=None,
                     tol_ppm: float = DEFAULT_TOL_PPM, demote=True, log=print) -> dict:
    """Annotate `ledger` (in place) with ts_cv_norm / ts_r_mono / ts_r_formic /
    ts_disposition from the time-series `peaks` table, and (if demote) cap a flat
    di-bromide / CO3-channel Assigned commit at Candidate (TS-confirmed
    background). Returns a summary dict. Reagent normaliser + anchors are taken
    from the ledger when not supplied.
    """
    summary = {"annotated": 0, "demoted": 0, "ambient": 0, "background": 0}
    for col in ("ts_cv_norm", "ts_r_mono", "ts_r_formic", "ts_disposition"):
        if col not in ledger.columns:
            ledger[col] = np.nan if col != "ts_disposition" else ""

    mat, bin_mz = build_matrix(peaks, tol_ppm=tol_ppm)
    if mat.empty:
        log("[timeseries] no peaks; skipped"); return summary

    # reagent normaliser: explicit, else the ledger's reagent Br_n rows
    if reagent_mzs is None:
        rr = ledger[(ledger["role"] == L.ROLE_REAGENT)
                    & ledger["ion_formula"].astype(str).str.match(r"Br\d-")]
        reagent_mzs = rr["mz"].dropna().tolist()
    rt = reagent_total(mat, bin_mz, reagent_mzs) if reagent_mzs else None
    norm = normalize(mat, rt)
    met = bin_metrics(norm, bin_mz)

    # reference family traces (optional)
    bmz_s = bin_mz.sort_values(); bvals = bmz_s.to_numpy(); bidx = bmz_s.index.to_numpy()
    def bins_for(mzs):
        out = []
        for m in (mzs or []):
            b = _nearest(bvals, bidx, m, tol_ppm)
            if b is not None:
                out.append(b)
        return out
    if mono_anchor_mzs is None:
        mono_anchor_mzs = ledger.loc[
            ledger["neutral_formula"].astype(str).isin(
                {"C10H16O3", "C10H16O4", "C10H16O5", "C10H16O6"}), "mz"].dropna().tolist()
    mono_tr = family_trace(norm, bins_for(mono_anchor_mzs))
    formic_b = _nearest(bvals, bidx, formic_mz, tol_ppm) if formic_mz else \
        _nearest(bvals, bidx, 124.9243, tol_ppm)
    formic_tr = norm[formic_b].pipe(lambda c: np.log10(c.clip(lower=1e-9))) if formic_b in norm.columns else None
    r_mono = correlate(norm, mono_tr)
    r_formic = correlate(norm, formic_tr)

    # stamp the ledger (M0 rows)
    for i in ledger.index[ledger["role"] == L.ROLE_M0]:
        mz = ledger.at[i, "mz"]
        if pd.isna(mz):
            continue
        b = _nearest(bvals, bidx, float(mz), tol_ppm)
        cv = float(met.at[b, "cv_norm"]) if (b is not None and b in met.index and pd.notna(met.at[b, "cv_norm"])) else np.nan
        rm = float(r_mono.get(b, np.nan)) if b is not None else np.nan
        rf = float(r_formic.get(b, np.nan)) if b is not None else np.nan
        disp = _disposition(ledger.loc[i], cv, rm, rf)
        ledger.at[i, "ts_cv_norm"] = cv
        ledger.at[i, "ts_r_mono"] = rm
        ledger.at[i, "ts_r_formic"] = rf
        ledger.at[i, "ts_disposition"] = disp
        summary["annotated"] += 1
        if disp.startswith("ambient"):
            summary["ambient"] += 1
        elif disp.startswith("background"):
            summary["background"] += 1
            # conservative auto-demote: a flat di-bromide / CO3 background commit
            # must not stay Assigned once the time series shows it is background
            if demote and str(ledger.at[i, "tier"]) == "Assigned" and (
                    "di-bromide" in disp or "CO3-channel" in disp):
                ledger.at[i, "tier"] = "Candidate"
                ledger.at[i, "tier_reason"] = (str(ledger.at[i, "tier_reason"] or "")
                    + " | time-series: flat background (reagent/inlet), demoted").strip(" |")
                summary["demoted"] += 1
    log(f"[timeseries] {summary}")
    return summary


# ---------------------------------------------------------------------------
# Reproducible single-compound time-series query
# ---------------------------------------------------------------------------
def find_ts_parquet(run_dir: str) -> str:
    """The cached batch time series in a run dir (`*_ts.parquet`)."""
    import glob
    hits = sorted(glob.glob(os.path.join(os.path.expanduser(run_dir), "*_ts.parquet")))
    if not hits:
        raise FileNotFoundError(f"no *_ts.parquet in {run_dir}")
    return hits[0]


def trace(run_dir: str, query, *, tol_ppm: float = DEFAULT_TOL_PPM,
          value: str = "height", ts: "pd.DataFrame | None" = None,
          ledger: "pd.DataFrame | None" = None) -> pd.DataFrame:
    """Pull the temporal trace of ONE compound from a finished run -- assigned OR
    unassigned. Reproducible: reads the run's own ``*_ts.parquet`` (full per-sample
    peak table) and ``merged_ledger.csv`` (the pipeline's assignments), so the
    answer is fixed by the run, not by re-deriving anything.

    query : a NEUTRAL FORMULA (str; resolved to its m/z via merged_ledger, taking
            the highest-ion-score adduct) or a float M/Z (use any peak, assigned
            or not). tol_ppm sets the m/z window summed per time point.

    Returns a tidy DataFrame [datetime_utc, <value>] (one row per sample time,
    summed over the window, time-sorted). ``df.attrs`` carries: mz, assignment
    ('<formula> <adduct> (<tier>)' or 'unassigned'), n_peak_ids, tol_ppm.
    """
    import os as _os
    ts = ts if ts is not None else pd.read_parquet(find_ts_parquet(run_dir))
    if ledger is None:
        mlp = _os.path.join(_os.path.expanduser(run_dir), "merged_ledger.csv")
        ledger = pd.read_csv(mlp) if _os.path.exists(mlp) else pd.DataFrame()

    assignment = "unassigned"
    if isinstance(query, str):
        hit = ledger[ledger.get("neutral_formula").astype(str) == query] \
            if "neutral_formula" in ledger.columns else ledger.iloc[0:0]
        if not len(hit):
            raise KeyError(f"{query!r} is not an assigned neutral in {run_dir} "
                           "(pass a float m/z to trace an unassigned peak)")
        if "ion_score" in hit.columns:
            hit = hit.sort_values("ion_score", ascending=False, na_position="last")
        row = hit.iloc[0]
        mz = float(row["mz"])
        assignment = (f"{query} {row.get('adduct', '')}".strip()
                      + f" ({row.get('tier', '?')})")
    else:
        mz = float(query)
        if len(ledger) and "mz" in ledger.columns:
            d = (ledger["mz"].astype(float) - mz).abs() / mz * 1e6
            j = d.idxmin() if len(d) else None
            if j is not None and d.loc[j] <= tol_ppm:
                r = ledger.loc[j]
                assignment = (f"{r.get('neutral_formula')} {r.get('adduct', '')}".strip()
                              + f" ({r.get('tier', '?')})")

    win = (ts["mz"].astype(float) - mz).abs() / mz * 1e6 <= tol_ppm
    sub = ts[win]
    if not len(sub):
        out = pd.DataFrame({"datetime_utc": [], value: []})
    else:
        out = (sub.groupby("datetime_utc", as_index=False)[value].sum()
               .sort_values("datetime_utc").reset_index(drop=True))
    out.attrs.update({"mz": mz, "assignment": assignment,
                      "n_peak_ids": int(sub["peak_id"].nunique()) if len(sub) else 0,
                      "tol_ppm": tol_ppm, "value": value})
    return out
