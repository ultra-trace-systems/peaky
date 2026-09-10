"""Consistent analyte visualisation: Van Krevelen + raw time-series.

Both plots are derived here from one committed ledger (the assignment) plus an
optional batch time-series peak table, so the negative Br-CIMS and positive
urea-CIMS figures are computed *identically* -- same analyte definition, same
H/C-O/C convention, same "changing" threshold, same RAW intensity choice.

Design choices (deliberate, so the figures are comparable):
- ANALYTE = a committed M0 with an organic neutral, EXCLUDING contamination
  (Si-bearing siloxane/silanediol) and the reagent/artifact roles. Polarity- and
  reagent-agnostic: it reads the neutral formula, not the adduct.
- Van Krevelen on the NEUTRAL: x = O/C, y = H/C (halogens are not in these
  neutrals -- the reagent halogen lives in the adduct).
- A compound's time trace is the RAW summed intensity of its bins under the
  given adducts. RAW (not reagent-normalised): a positive urea-CIMS spectrum
  often excludes the reagent ions from its mass range, and TIC is
  analyte-dominated, so reagent/TIC normalisation introduces closure artifacts.
  `mode='reagent'/'tic'` is available when a real normaliser exists.
- `cv` = std/mean of the (mode) trace; `changing` = cv >= CHANGING_CV. Same
  threshold both instruments.

Pure data functions (no plotting deps) + lazy-matplotlib renderers + a
`widget_payload()` that returns the arrays the interactive chat widget consumes.
"""
from __future__ import annotations

import bisect

import numpy as np
import pandas as pd

from peaky.chem import chemistry as C
from peaky.batch import timeseries as TS

__version__ = "0.1.0"

CHANGING_CV = 0.30           # cv_raw at/above which a peak is "changing"
DEFAULT_BIN_MIN = 30         # time-bin width (minutes) for the trace plot
_CONTAM_ELEMENTS = ("Si",)   # siloxane / silanediol = instrument contamination


def _is_contaminant(formula: str) -> bool:
    cnt = C.parse_formula(str(formula))
    return any(cnt.get(e, 0) for e in _CONTAM_ELEMENTS)


# Composition categories for the FULL Van Krevelen (every assigned peak shown,
# nothing excluded). Order = legend order; the first matching wins.
FULL_CLASS_ORDER = ("CHO", "CHON", "CHOS", "F-containing", "halogenated", "siloxane")
FULL_CLASS_COLORS = {
    "CHO": "#1D9E75", "CHON": "#7F77DD", "CHOS": "#D85A30",
    "F-containing": "#D4537E", "halogenated": "#BA7517", "siloxane": "#378ADD"}


def full_class(formula: str) -> str:
    """Composition class for the full VK: Si -> siloxane; else F -> F-containing;
    else Cl/Br-in-neutral -> halogenated; else CHOS / CHON / CHO."""
    c = C.parse_formula(str(formula))
    if c.get("Si", 0):
        return "siloxane"
    if c.get("F", 0):
        return "F-containing"
    if c.get("Cl", 0) or c.get("Br", 0):
        return "halogenated"
    if c.get("S", 0):
        return "CHOS"
    if c.get("N", 0):
        return "CHON"
    return "CHO"


def analyte_table(ledger: pd.DataFrame, *, exclude_contaminant: bool = True
                  ) -> pd.DataFrame:
    """The committed organic analytes (one row per M0). Adds H/C, O/C, N count
    and a CHO/CHON/CHOS class. Contamination (Si) and non-M0 roles dropped."""
    m0 = ledger[ledger["role"] == "M0"].copy()
    m0 = m0[m0["neutral_formula"].notna() & (m0["neutral_formula"].astype(str) != "")]
    if exclude_contaminant:
        m0 = m0[~m0["neutral_formula"].map(_is_contaminant)]
    cnt = m0["neutral_formula"].map(lambda f: C.parse_formula(str(f)))
    for col, el in (("nC", "C"), ("nH", "H"), ("nO", "O"), ("nN", "N"), ("nS", "S")):
        m0[col] = cnt.map(lambda c, el=el: c.get(el, 0))
    m0 = m0[m0["nC"] > 0].copy()
    m0["hc"] = m0["nH"] / m0["nC"]
    m0["oc"] = m0["nO"] / m0["nC"]
    m0["klass"] = np.where(m0["nS"] > 0, "CHOS",
                           np.where(m0["nN"] > 0, "CHON", "CHO"))
    # channels (adducts) a neutral was seen in -- the "ion" half of the hover.
    if "adduct" in m0.columns:
        chmap = (m0.groupby("neutral_formula")["adduct"]
                 .apply(lambda s: " / ".join(dict.fromkeys(s.dropna().astype(str)))).to_dict())
        m0["channels"] = m0["neutral_formula"].map(chmap)
    else:
        m0["channels"] = ""
    # ONE row per NEUTRAL: a compound seen in two channels ([M+Br]- and [M-H]-,
    # or [M+H]+ and [M+urea+H]+) is a single analyte. Keep the best tier.
    if "tier" in m0.columns:
        m0["_tr"] = m0["tier"].map({"Assigned": 0, "Candidate": 1}).fillna(2)
        m0 = m0.sort_values("_tr").drop_duplicates("neutral_formula", keep="first").drop(columns="_tr")
    else:
        m0 = m0.drop_duplicates("neutral_formula", keep="first")
    return m0.reset_index(drop=True)


# ---------------------------------------------------------------------------
# time series
# ---------------------------------------------------------------------------
def _bins_for_formula(arr, idx, cols_set, formula, adducts):
    """Bin indices whose m/z match the neutral under any adduct (<=8 ppm)."""
    out = []
    for a in adducts:
        if a not in C.ADDUCT_SHIFTS:
            continue
        mz = C.ion_mz(formula, a)
        i = bisect.bisect_left(arr, mz)
        for j in (i - 1, i):
            if 0 <= j < len(arr) and abs(arr[j] - mz) / mz * 1e6 <= 8:
                out.append(idx[j])
    return [c for c in set(out) if c in cols_set]


def time_traces(ts_peaks: pd.DataFrame, formulas, adducts, *, mode: str = "raw",
                bin_minutes: int | None = None, reagent_mzs=None
                ) -> tuple[np.ndarray, pd.DataFrame]:
    """Raw (or normalised) intensity trace per formula (summed across its adducts).

    `bin_minutes=None` (default) = NATIVE per-sample resolution (one point per
    sample at its real hour, no re-gridding); pass an int to time-bin (legacy).
    Returns (x-hours, DataFrame columns=formulas). NaN where a formula has no bin."""
    ts_peaks = ts_peaks.copy()
    ts_peaks["datetime_utc"] = pd.to_datetime(ts_peaks["datetime_utc"], utc=True)
    tstamp = ts_peaks.groupby("sample_item_id")["datetime_utc"].first()
    mat, bin_mz = TS.build_matrix(ts_peaks)
    mat = mat.reindex(tstamp.sort_values().index)
    t0 = tstamp.min()
    hr = (tstamp.reindex(mat.index) - t0).dt.total_seconds().values / 3600.0
    bm = bin_mz.sort_values(); arr = bm.values; idx = bm.index.values
    cols_set = set(mat.columns)
    if mode == "reagent" and reagent_mzs:
        rt = TS.reagent_total(mat, bin_mz, reagent_mzs)
    elif mode == "tic":
        rt = mat.sum(axis=1)
    else:
        rt = None

    def _series(b):
        s = mat[b].sum(axis=1)
        return (s / rt.replace(0, np.nan)) if rt is not None else s

    if bin_minutes is None:                                # NATIVE — one point per sample
        out = {}
        for f in formulas:
            b = _bins_for_formula(arr, idx, cols_set, f, adducts)
            out[f] = (pd.Series(_series(b).values) if b
                      else pd.Series(np.nan, index=range(len(hr))))
        return hr, pd.DataFrame(out)

    grid = np.arange(0, np.nanmax(hr) + bin_minutes / 60.0, bin_minutes / 60.0)
    digit = np.digitize(hr, grid)
    out = {}
    for f in formulas:
        b = _bins_for_formula(arr, idx, cols_set, f, adducts)
        if not b:
            out[f] = pd.Series(np.nan, index=range(1, len(grid) + 1))
            continue
        out[f] = pd.Series(_series(b).values).groupby(digit).median().reindex(range(1, len(grid) + 1))
    return grid, pd.DataFrame(out)


# compact per-ion channel labels (formula + adduct suffix), e.g. C6H14O4+Ur⁺
ADDUCT_SUFFIX = {
    "[M+H]+": "+H⁺", "[M-H]-": "−H⁻",
    "[M+(CH4N2O)H]+": "+Ur⁺", "[M+NH4]+": "+NH₄⁺", "[M+Na]+": "+Na⁺",
    "[M+^NH4]+": "+¹⁵NH₄⁺", "[M+^NH4-H2O]+": "+¹⁵NH₄⁺−H₂O", "[M+H-H2O]+": "+H⁺−H₂O",
    "[M+Br]-": "+Br⁻", "[M+HBr+Br]-": "+Br₂H⁻", "[M+Br2]-": "+Br₂⁻",
    "[M+CO3]-": "+CO₃⁻", "[M+HBr+CO3]-": "+HBrCO₃⁻", "[M+HSO4]-": "+HSO₄⁻",
    "[M+Cl]-": "+Cl⁻", "[M+I]-": "+I⁻",
}


def ion_label(formula, adduct) -> str:
    """Compact ion name: neutral formula + a short adduct suffix (C6H14O4+Ur⁺)."""
    return f"{formula}{ADDUCT_SUFFIX.get(str(adduct), ' ' + str(adduct))}"


def _bin_for_mz(arr, idx, cols_set, mz, tol=8.0):
    """Index of the single TS bin closest to `mz` within `tol` ppm, or None."""
    i = bisect.bisect_left(arr, mz)
    best, bestd = None, tol
    for j in (i - 1, i):
        if 0 <= j < len(arr):
            d = abs(arr[j] - mz) / mz * 1e6
            if d <= bestd and idx[j] in cols_set:
                best, bestd = idx[j], d
    return best


def ion_traces(ts_peaks: pd.DataFrame, ion_mz_map: dict, *, mode: str = "raw",
               bin_minutes: int | None = None, reagent_mzs=None
               ) -> tuple[np.ndarray, pd.DataFrame]:
    """One trace PER ION (no summing across adducts — the opposite of time_traces).
    `ion_mz_map` maps an arbitrary key -> the ion's measured m/z; each key gets the
    trace of the single TS bin matching that m/z (<=8 ppm). Keeping channels separate
    lets divergent ion channels of one neutral cluster apart.

    `bin_minutes=None` (default) = NATIVE per-sample resolution: the samples are
    already the common time axis, so return one point per sample at its real hour —
    no re-gridding onto a uniform lattice (which aliases into spurious empty time
    bins). Pass an int to time-bin instead (legacy). Returns (x-hours, traces df)."""
    ts_peaks = ts_peaks.copy()
    ts_peaks["datetime_utc"] = pd.to_datetime(ts_peaks["datetime_utc"], utc=True)
    tstamp = ts_peaks.groupby("sample_item_id")["datetime_utc"].first()
    mat, bin_mz = TS.build_matrix(ts_peaks)
    mat = mat.reindex(tstamp.sort_values().index)          # rows in time order
    hr = (tstamp.reindex(mat.index) - tstamp.min()).dt.total_seconds().values / 3600.0
    bm = bin_mz.sort_values(); arr = bm.values; idx = bm.index.values
    cols_set = set(mat.columns)
    if mode == "reagent" and reagent_mzs:
        rt = TS.reagent_total(mat, bin_mz, reagent_mzs)
    elif mode == "tic":
        rt = mat.sum(axis=1)
    else:
        rt = None

    def _series(b):                                        # the (normalised) per-sample trace
        s = mat[b]
        return (s / rt.replace(0, np.nan)) if rt is not None else s

    if bin_minutes is None:                                # NATIVE — one point per sample
        out = {}
        for key, mz in ion_mz_map.items():
            b = _bin_for_mz(arr, idx, cols_set, float(mz))
            out[key] = (pd.Series(_series(b).values) if b is not None
                        else pd.Series(np.nan, index=range(len(hr))))
        return hr, pd.DataFrame(out)

    grid = np.arange(0, np.nanmax(hr) + bin_minutes / 60.0, bin_minutes / 60.0)
    digit = np.digitize(hr, grid)
    out = {}
    for key, mz in ion_mz_map.items():
        b = _bin_for_mz(arr, idx, cols_set, float(mz))
        if b is None:
            out[key] = pd.Series(np.nan, index=range(1, len(grid) + 1))
            continue
        out[key] = pd.Series(_series(b).values).groupby(digit).median().reindex(range(1, len(grid) + 1))
    return grid, pd.DataFrame(out)


def _logcorr(a, b, min_points: int = 8):
    """log10 Pearson r over the time points where BOTH traces are positive."""
    df = pd.DataFrame({"a": np.asarray(a, float), "b": np.asarray(b, float)})
    df = df[(df > 0).all(axis=1)]
    if len(df) < min_points:
        return np.nan, len(df)
    lg = np.log10(df)
    return float(lg["a"].corr(lg["b"])), len(df)


def channel_agreement(ts_peaks: pd.DataFrame, ion_table: pd.DataFrame, *,
                      floor: float = 150.0, min_points: int = 8,
                      bin_minutes: int | None = None) -> pd.DataFrame:
    """QC: do the ion channels of the SAME neutral track in time? (We otherwise
    SUM them — `time_traces`.) For every neutral with >=2 testable channels
    (>=min_points finite points AND median >= floor cps), correlate every channel
    pair (log10). Returns one row per neutral: n_channels, worst_r (least-agreeing
    pair), top2_r (the two BRIGHTEST channels — the pair that dominates the sum),
    channels (list), verdict ('agree' r>=0.7 / 'marginal' / 'disagree' r<0.4 on
    worst_r). `ion_table` needs neutral_formula, adduct, mz."""
    t = ion_table.dropna(subset=["neutral_formula"]).copy()
    t = t[t["neutral_formula"].astype(str) != ""]
    t["ckey"] = t["neutral_formula"].astype(str) + "|" + t["adduct"].astype(str)
    grid, traces = ion_traces(ts_peaks, dict(zip(t["ckey"], t["mz"])),
                              mode="raw", bin_minutes=bin_minutes)
    rows = []
    for f, g in t.groupby("neutral_formula"):
        chans = []
        for r in g.itertuples():
            tr = traces[r.ckey] if r.ckey in traces else None
            if tr is None:
                continue
            med = float(np.nanmedian(tr.values.astype(float)))
            if tr.notna().sum() >= min_points and med >= floor:
                chans.append((str(r.adduct), tr.values.astype(float), med))
        if len(chans) < 2:
            continue
        prs = []
        for i in range(len(chans)):
            for j in range(i + 1, len(chans)):
                rr, _ = _logcorr(chans[i][1], chans[j][1], min_points)
                if not np.isnan(rr):
                    prs.append(rr)
        if not prs:
            continue
        top2 = sorted(chans, key=lambda c: -c[2])[:2]
        t2, _ = _logcorr(top2[0][1], top2[1][1], min_points)
        worst = float(min(prs))
        rows.append({"neutral_formula": str(f), "n_channels": len(chans),
                     "worst_r": round(worst, 3),
                     "top2_r": (round(float(t2), 3) if not np.isnan(t2) else None),
                     "channels": ", ".join(f"{a}~{m:.0f}" for a, _, m in
                                           sorted(chans, key=lambda c: -c[2])),
                     "verdict": "agree" if worst >= 0.7 else
                                ("marginal" if worst >= 0.4 else "disagree")})
    return pd.DataFrame(rows).sort_values("worst_r").reset_index(drop=True) if rows \
        else pd.DataFrame(columns=["neutral_formula", "n_channels", "worst_r",
                                   "top2_r", "channels", "verdict"])


def attach_dynamics(analytes: pd.DataFrame, ts_peaks: pd.DataFrame, adducts, *,
                    mode: str = "raw", reagent_mzs=None,
                    bin_minutes: int | None = None) -> pd.DataFrame:
    """Add median_cps / cv / changing to the analyte table from the time series.
    `bin_minutes=None` (default) computes cv at NATIVE per-sample resolution (every
    sample is a point), so a short densely-sampled batch is no longer starved of
    points; pass an int to time-bin (legacy)."""
    df = analytes.copy()
    _, traces = time_traces(ts_peaks, df["neutral_formula"].tolist(), adducts,
                            mode=mode, reagent_mzs=reagent_mzs, bin_minutes=bin_minutes)
    med, cv = {}, {}
    for f in df["neutral_formula"]:
        tr = traces[f].dropna().values if f in traces else np.array([])
        med[f] = float(np.median(tr)) if len(tr) else 0.0
        cv[f] = float(np.std(tr) / np.mean(tr)) if len(tr) and np.mean(tr) > 0 else np.nan
    df["median_cps"] = df["neutral_formula"].map(med)
    df["cv"] = df["neutral_formula"].map(cv)
    df["changing"] = df["cv"] >= CHANGING_CV
    return df


# ---------------------------------------------------------------------------
# widget payload (the arrays the interactive chat plot consumes)
# ---------------------------------------------------------------------------
def widget_payload(analytes: pd.DataFrame, grid: np.ndarray | None = None,
                   traces: pd.DataFrame | None = None, *, top_ts: int = 28) -> dict:
    """{'vk': [[oc,hc,n,changing,logI,formula,channels,tier],...],
        'ts': {'grid':[h...], 'series':[{f,n,ch,y:[...]}, ...]}}. `channels` is
        the adduct(s) the neutral was detected as -- the 'ion' shown on hover."""
    vk = []
    for r in analytes.itertuples():
        logI = round(float(np.log10(max(getattr(r, "median_cps", 0.0), 1))), 2)
        vk.append([round(r.oc, 3), round(r.hc, 3), int(r.nN > 0),
                   int(bool(getattr(r, "changing", False))), logI, r.neutral_formula,
                   str(getattr(r, "channels", "")), str(getattr(r, "tier", ""))])
    ts = None
    if grid is not None and traces is not None and "changing" in analytes:
        chg = analytes[analytes["changing"]].sort_values("median_cps", ascending=False).head(top_ts)
        series = []
        for r in chg.itertuples():
            y = [None if pd.isna(v) or v <= 0 else round(float(v), 0)
                 for v in traces[r.neutral_formula].values]
            series.append({"f": r.neutral_formula, "n": int(r.nN > 0),
                           "ch": str(getattr(r, "channels", "")), "y": y})
        ts = {"grid": [round(float(x), 1) for x in grid], "series": series}
    return {"vk": vk, "ts": ts}


# ---------------------------------------------------------------------------
# matplotlib renderers (lazy import; standalone PNGs)
# ---------------------------------------------------------------------------
def render_van_krevelen(analytes: pd.DataFrame, path: str, *, title: str = "") -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 5))
    sz = (np.maximum(2.0, (np.log10(analytes["median_cps"].clip(lower=1)) - 2.5) * 6))
    flat = analytes[~analytes.get("changing", pd.Series(False, index=analytes.index))]
    chg = analytes[analytes.get("changing", pd.Series(False, index=analytes.index))]
    ax.scatter(flat["oc"], flat["hc"], s=sz.loc[flat.index], c="#B4B2A9", alpha=0.4,
               linewidths=0, label="flat background")
    for kl, col in (("CHO", "#1D9E75"), ("CHON", "#7F77DD"), ("CHOS", "#D85A30")):
        g = chg[chg["klass"] == kl]
        if len(g):
            ax.scatter(g["oc"], g["hc"], s=sz.loc[g.index], c=col, alpha=0.85,
                       linewidths=0.3, edgecolors="white", label=f"changing — {kl}")
    ax.set_xlabel("O/C"); ax.set_ylabel("H/C"); ax.set_xlim(0, 1.3); ax.set_ylim(0.3, 3.0)
    ax.set_title(title); ax.legend(fontsize=8, loc="upper right"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
    return path


_BACKBONE_ORDER = ("CHO", "CHON", "CHOS")
_BACKBONE_COLORS = {"CHO": "#1D9E75", "CHON": "#7F77DD", "CHOS": "#D85A30"}


def backbone_class(formula: str) -> str:
    """Backbone class: CHOS if S, else CHON if N, else CHO. Heteroatoms Si/F/Cl/Br
    are 'additions' to this backbone, NOT a separate class (a siloxane with no N is
    CHO; a fluorinated species with N is CHON)."""
    c = C.parse_formula(str(formula))
    if c.get("S", 0):
        return "CHOS"
    if c.get("N", 0):
        return "CHON"
    return "CHO"


def render_van_krevelen_full(analytes: pd.DataFrame, path: str, *, title: str = "",
                             xmax: float = 1.4, ymax: float = 4.2, dpi: int = 150) -> str:
    """Van Krevelen showing EVERY assigned peak, coloured by CHO/CHON/CHOS BACKBONE
    (Si/F/halogen folded into their backbone class, not split out). Changing
    analytes solid with a white edge; flat ones dimmed. Size = log intensity.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    a = analytes.copy()
    if "klass" not in a.columns:
        a["klass"] = a["neutral_formula"].map(backbone_class)
    chg = a.get("changing", pd.Series(False, index=a.index)).astype(bool)
    sz = np.maximum(6.0, (np.log10(a["median_cps"].clip(lower=1)) - 2.0) * 9)
    fig, ax = plt.subplots(figsize=(8.5, 6))
    for kl in _BACKBONE_ORDER:
        col = _BACKBONE_COLORS[kl]
        g = a[a["klass"] == kl]
        if not len(g):
            continue
        gf, gc = g[~chg.loc[g.index]], g[chg.loc[g.index]]
        if len(gf):
            ax.scatter(gf["oc"], gf["hc"], s=sz.loc[gf.index], c=col, alpha=0.30, linewidths=0)
        if len(gc):
            ax.scatter(gc["oc"], gc["hc"], s=sz.loc[gc.index], c=col, alpha=0.9,
                       linewidths=0.4, edgecolors="white")
        ax.scatter([], [], c=col, s=60, label=f"{kl}  (n={len(g)})")
    ax.scatter([], [], c="0.5", s=60, alpha=0.3, label="faded = flat · solid = changing")
    ax.set_xlabel("O/C", fontsize=12); ax.set_ylabel("H/C", fontsize=12)
    ax.set_xlim(0, xmax); ax.set_ylim(0.3, ymax)
    ax.tick_params(labelsize=11)
    ax.set_title(title, fontsize=13)
    # legend OUTSIDE the plot (to the right) so it never crowds the data
    ax.legend(fontsize=10.5, loc="upper left", bbox_to_anchor=(1.02, 1.0),
              framealpha=0.95, markerscale=1.2, borderaxespad=0.0)
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=dpi, bbox_inches="tight"); plt.close(fig)
    return path


def render_timeseries(grid: np.ndarray, traces: pd.DataFrame, analytes: pd.DataFrame,
                      path: str, *, top: int = 28, title: str = "") -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    nN = dict(zip(analytes["neutral_formula"], analytes["nN"]))
    chg = analytes[analytes.get("changing", pd.Series(False, index=analytes.index))]
    chg = chg.sort_values("median_cps", ascending=False).head(top)
    fig, ax = plt.subplots(figsize=(8, 5))
    for f in chg["neutral_formula"]:
        col = "#7F77DD" if nN.get(f, 0) else "#1D9E75"
        ax.plot(grid, traces[f].values, color=col, lw=1.1, alpha=0.7)
    ax.set_yscale("log"); ax.set_xlabel("hour of experiment (UTC)")
    ax.set_ylabel("raw intensity (cps)"); ax.set_title(title); ax.grid(alpha=0.3, which="both")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
    return path


def van_krevelen_batch(out_dir, ts, profile, *, merged=None, tag=None, label=None,
                       batch_name=None, subject=None, bin_minutes=None, log=print) -> dict:
    """Both Van Krevelen figures + the full-VK CSV for one batch — the in-package
    home of the `run_vankrevelen.py` scratch driver. (1) organic-only VK (Si
    excluded), changing analytes coloured; (2) FULL VK — every assigned peak by
    CHO/CHON/CHOS backbone (Si/F/halogen folded in) + a `van_krevelen_full_<tag>.csv`.
    """
    import os

    from peaky import paths as PT
    from peaky.batch import timeseries as TS

    OUT = os.path.expanduser(out_dir)
    P = PT.run_paths(OUT).ensure()
    FIG, TAB = P.figures, P.tables       # .png -> figures/, .csv -> tables/
    tag = tag or profile.name
    label = label or profile.label
    batch_name = batch_name or label
    if merged is None:
        merged = os.path.join(OUT, "merged_ledger.csv")
    merged = pd.read_csv(merged) if isinstance(merged, str) else merged.copy()
    merged["role"] = "M0"   # the merged ledger is assigned-compound rows; analyte_table expects a role col
    if isinstance(ts, str):
        ts = pd.read_parquet(os.path.expanduser(ts))
    ts = ts.copy()
    ts["datetime_utc"] = pd.to_datetime(ts["datetime_utc"], utc=True)
    BIN_MIN = bin_minutes        # None = NATIVE per-sample resolution (no time grid)

    # (1) organic-only VK (Si excluded) — the clean atmospheric view
    an = analyte_table(merged, exclude_contaminant=True)
    an = attach_dynamics(an, ts, profile.adducts, mode="raw", bin_minutes=BIN_MIN)
    nchg = int(an["changing"].sum())
    log(f"{tag}: {len(an)} organic analytes ({nchg} changing); "
        f"res={'native/sample' if BIN_MIN is None else str(BIN_MIN)+'min'}")
    subj = f" · {subject}" if subject else ""
    render_van_krevelen(
        an, f"{FIG}/van_krevelen_{tag}.png",
        title=f"{label}{subj} — Van Krevelen ({len(an)} analytes, {nchg} changing)")

    # (2) FULL VK — EVERY assigned peak, coloured by composition (Si/F/halogen shown).
    anf = analyte_table(merged, exclude_contaminant=False)
    anf = attach_dynamics(anf, ts, profile.adducts, mode="raw", bin_minutes=BIN_MIN)
    anf["fclass"] = anf["neutral_formula"].map(full_class)
    log(f"{tag} FULL: {len(anf)} assigned neutrals; classes {anf['fclass'].value_counts().to_dict()}")
    render_van_krevelen_full(
        anf, f"{FIG}/van_krevelen_full_{tag}.png",
        title=f"Van Krevelen — {batch_name}   ({len(anf)} assigned compounds)")
    anf[["neutral_formula", "adduct", "tier", "oc", "hc", "fclass", "median_cps", "cv", "changing"]] \
        .to_csv(f"{TAB}/van_krevelen_full_{tag}.csv", index=False)
    log(f"wrote {FIG}/van_krevelen_full_{tag}.png")
    return {"organic": an, "full": anf, "bin_minutes": BIN_MIN, "out_dir": OUT, "tag": tag}
