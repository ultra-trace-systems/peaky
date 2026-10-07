"""The neutral-pair fact of rule U (docs/EVIDENCE_LEVELS.md §3 `upair`, §4 row 9').

A reagent chemistry whose profile declares a (bare, cluster) adduct pair --
the uronium profile: ([M+H]+, [M+(CH4N2O)H]+), `ReagentProfile.neutral_pair` --
sees one neutral M as TWO ions, protonated and urea-clustered, 60.0324 Da apart.
When both are committed and the batch's raw time series says one neutral makes
both, the pair establishes M the way the acid branch (deprotonated AND clustered)
does on the anion channels. The fact is measured once per batch, over the
stamped batch time series and the pooled per-file ledgers, and read by the
pooled level recompute (`evidence.level_pooled(..., upair=...)`): never per
file, never an axis, never in `cross`, never on a profile without a pair.

`upair(M)` holds when ALL of:
  committed : M is committed under both adducts of the pair (regular M0 rows of
              the pooled per-file ledgers)
  elements  : M is C/H/O only -- N-free (the NH4 alias [X+NH4]+ = [X+NH3+H]+ and
              the urea ladder both need N) and no S/Si/P/halogen (curated or
              contaminant evidence, not ion chemistry)
  presence  : each ion has a raw peak within PRESENCE_TOL_PPM of its exact m/z in
              >= PRESENCE_SHARE of the spectra, |median ppm| <= MEDIAN_PPM_MAX
              (the signed median: a centroid biased to one side fails)
  co-vary   : r(log10 h_bare, log10 h_cluster) >= COVARY_R_MIN over >=
              COVARY_MIN_SPECTRA spectra where both are >= COVARY_MIN_HEIGHT
  clean     : neither ion's matched peak is stamped iso_child / artifact, or as
              ANOTHER reading's M0, in more than CLEAN_SHARE of the spectra where
              it is found
  13C       : the area-13C carbon count of neither ion contradicts its formula
              (|n_obs / scale - n| > max(1, 0.2 n); the scale is the median of
              n_obs / n over both ions of every committed-both neutral)
The level rule adds formula support (the pair's own `iso`, or one plausible ion
on a resolved peak) and the rows above 9' (hard 5b, curated, branch) win first.
Measured on the regression batches (2026-09-27 output audit, round 3): exact
mass plus degeneracy <= 1 give the specificity (as built, 0 neutrals pass at
each of 12 decoy offsets on the uronium run); r >= 0.5 is a weak veto. A
composed profile keeps the pair when exactly one component declares one. The round-3 step-proportionality veto and
bright-parent guard read hand-dated steady states of one batch and are not
built (the user's decision, 2026-09-27).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from peaky.chem import chemistry as C

PRESENCE_TOL_PPM = 2.0
PRESENCE_SHARE = 0.5
MEDIAN_PPM_MAX = 1.0
COVARY_R_MIN = 0.5
COVARY_MIN_SPECTRA = 30
COVARY_MIN_HEIGHT = 150.0
CLEAN_SHARE = 0.5
#: the 13C check: M+1 13C line / M0 per carbon; the expected 13C line must be >=
#: C13_MIN_EXPECTED counts in >= C13_MIN_SPECTRA spectra to test; a line found in
#: under C13_FOUND_SHARE of those (with >= C13_MISSING_MIN of them) reads 0 C
C13_PER_C = 0.0107 / 0.9893
D13C = 1.0033548378
C13_MIN_EXPECTED = 300.0
C13_MIN_SPECTRA = 8
C13_FOUND_SHARE = 0.8
C13_MISSING_MIN = 10
ALLOWED_ELEMENTS = frozenset({"C", "H", "O"})
TABLE_COLUMNS = (
    "neutral_formula", "bare", "cluster", "mz_bare", "mz_cluster", "committed_both", "cho",
    "det_bare", "ppm_bare", "det_cluster", "ppm_cluster", "n_both", "r_log",
    "clean", "nC_bare", "nC_cluster", "nC_obs_bare", "nC_obs_cluster", "c13_scale", "c13_contradicts",
    "present", "covary", "upair",
)


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=list(TABLE_COLUMNS))


def _committed(frames: dict, pair: tuple) -> pd.DataFrame:
    """One row per neutral committed under an adduct of the pair: the adducts it
    has and each ion's formula (regular M0 rows; ion-only rows never count)."""
    from peaky.assignment import evidence as EV
    parts = []
    for f in frames.values():
        if f is None or not len(f) or "role" not in f.columns:
            continue
        m0 = f[f["role"].astype(str) == "M0"]
        m0 = m0[~EV.is_ion_only(m0).to_numpy()]
        m0 = m0[m0["adduct"].astype(str).isin(pair)]
        if len(m0):
            parts.append(m0[["neutral_formula", "adduct", "ion_formula"]])
    if not parts:
        return pd.DataFrame(columns=["neutral_formula", "adducts", "ion_bare", "ion_cluster"])
    m = pd.concat(parts, ignore_index=True)
    m = m[m["neutral_formula"].notna()]
    m["neutral_formula"] = m["neutral_formula"].astype(str)
    ions = m.dropna(subset=["ion_formula"]).drop_duplicates(["neutral_formula", "adduct"])
    ions = ions.set_index(["neutral_formula", "adduct"])["ion_formula"].astype(str)
    out = m.groupby("neutral_formula")["adduct"].agg(lambda s: set(s.astype(str))).rename("adducts").reset_index()
    out["ion_bare"] = [ions.get((n, pair[0]), "") for n in out["neutral_formula"]]
    out["ion_cluster"] = [ions.get((n, pair[1]), "") for n in out["neutral_formula"]]
    return out


class _Series:
    """The batch time series as per-spectrum sorted arrays, for nearest-peak lookups."""

    def __init__(self, ts: pd.DataFrame):
        t = ts.reset_index(drop=True)
        self.role = t["role"].astype(object).where(t["role"].notna(), "").astype(str).to_numpy()
        nf = t["neutral_formula"].astype(object).where(t["neutral_formula"].notna(), "").astype(str)
        ad = t["adduct"].astype(object).where(t["adduct"].notna(), "").astype(str)
        self.reading = (nf + " " + ad).str.strip().to_numpy()
        when = pd.to_datetime(t.groupby("sample_item_id")["datetime_utc"].first(), utc=True).sort_values()
        self.order = list(when.index)
        area = t["area"] if "area" in t.columns else pd.Series(np.nan, index=t.index)
        t = t.assign(__area=pd.to_numeric(area, errors="coerce"))
        self.spectra = {}
        for sid, g in t.sort_values("mz").groupby("sample_item_id", sort=False):
            self.spectra[sid] = (g["mz"].to_numpy(float), pd.to_numeric(g["height"], errors="coerce").to_numpy(float),
                                 g["__area"].to_numpy(float), g.index.to_numpy())

    def extract(self, targets) -> tuple:
        """(height, area, ppm, row) arrays of shape (spectra, targets): the nearest
        peak within PRESENCE_TOL_PPM of each target, NaN / -1 where there is none."""
        targets = np.asarray(targets, float)
        shape = (len(self.order), len(targets))
        H = np.full(shape, np.nan); A = np.full(shape, np.nan); P = np.full(shape, np.nan)
        R = np.full(shape, -1, dtype=int)
        if not len(targets):
            return H, A, P, R
        for i, sid in enumerate(self.order):
            mz, h, a, ix = self.spectra[sid]
            if len(mz) < 2:
                continue
            j = np.clip(np.searchsorted(mz, targets), 1, len(mz) - 1)
            jj = np.where(np.abs(mz[j] - targets) < np.abs(mz[j - 1] - targets), j, j - 1)
            d = (mz[jj] - targets) / targets * 1e6
            ok = np.abs(d) <= PRESENCE_TOL_PPM
            H[i, ok] = h[jj[ok]]; A[i, ok] = a[jj[ok]]; P[i, ok] = d[ok]; R[i, ok] = ix[jj[ok]]
        return H, A, P, R


def _majority(values: np.ndarray) -> tuple[str, float]:
    if not len(values):
        return "", 0.0
    v, c = np.unique(values.astype(str), return_counts=True)
    return str(v[c.argmax()]), float(c.max() / len(values))


def _carbon_obs(H0, A0, H1, A1, n_c) -> list:
    """Per ion: median area(13C) / area(M0) over the spectra whose expected 13C
    line is >= C13_MIN_EXPECTED, divided by the per-carbon ratio (a carbon
    count); NaN when untestable, 0.0 when the line is missing too often."""
    out = []
    for j in range(H0.shape[1]):
        expected = H0[:, j] * n_c[j] * C13_PER_C
        use = np.isfinite(H0[:, j]) & (np.nan_to_num(expected) >= C13_MIN_EXPECTED)
        n = int(use.sum())
        if n < C13_MIN_SPECTRA:
            out.append(np.nan)
            continue
        f = use & np.isfinite(A1[:, j]) & np.isfinite(A0[:, j]) & (np.nan_to_num(A0[:, j]) > 0)
        if f.sum() / n < C13_FOUND_SHARE:
            out.append(0.0 if n >= C13_MISSING_MIN else np.nan)
            continue
        out.append(float(np.median(A1[f, j] / A0[f, j]) / C13_PER_C))
    return out


def measure(ts: pd.DataFrame | None, frames: dict, pair, *, log=print) -> pd.DataFrame:
    """The neutral-pair table: one row per neutral committed under an adduct of
    `pair`, every clause of the fact and `upair`. Empty when the profile declares
    no pair, the batch has no time series, or nothing is committed on the pair."""
    pair = tuple(pair or ())
    if len(pair) != 2 or ts is None or not len(ts):
        return _empty()
    bare, cluster = pair
    com = _committed(frames, pair)
    if com.empty:
        return _empty()
    com["committed_both"] = com["adducts"].map(lambda s: {bare, cluster} <= s)
    com["cho"] = com["neutral_formula"].map(
        lambda n: set(k for k, v in C.fold_isotopes(C.parse_formula(n)).items() if v) <= ALLOWED_ELEMENTS)
    # the raw-series clauses are measured on every committed-both neutral (the
    # 13C scale needs the whole population), the CHO clause applied after
    test = com[com["committed_both"]].reset_index(drop=True)
    rows = com[~com["committed_both"]].assign(upair=False, bare=bare, cluster=cluster)
    if test.empty:
        return _finish(rows, pd.DataFrame())
    series = _Series(ts)
    neutrals = test["neutral_formula"].tolist()
    mz_b = np.array([C.ion_mz(n, bare) for n in neutrals])
    mz_c = np.array([C.ion_mz(n, cluster) for n in neutrals])
    Hb, Ab, Pb, Rb = series.extract(mz_b)
    Hc, Ac, Pc, Rc = series.extract(mz_c)
    _, Ab1, _, _ = series.extract(mz_b + D13C)
    _, Ac1, _, _ = series.extract(mz_c + D13C)
    nc_b = np.array([C.parse_formula(f).get("C", 0) if f else C.parse_formula(n).get("C", 0)
                     for f, n in zip(test["ion_bare"], neutrals)])
    nc_c = np.array([C.parse_formula(f).get("C", 0) if f else C.parse_formula(n).get("C", 0)
                     for f, n in zip(test["ion_cluster"], neutrals)])
    ob_b = _carbon_obs(Hb, Ab, Hb, Ab1, nc_b)
    ob_c = _carbon_obs(Hc, Ac, Hc, Ac1, nc_c)
    rec = []
    for j, n in enumerate(neutrals):
        r = dict(neutral_formula=n, mz_bare=mz_b[j], mz_cluster=mz_c[j])
        clean = True
        for tag, H, P, R, want in (("bare", Hb, Pb, Rb, f"{n} {bare}"), ("cluster", Hc, Pc, Rc, f"{n} {cluster}")):
            found = np.isfinite(H[:, j])
            r[f"det_{tag}"] = float(found.mean()) if len(found) else 0.0
            r[f"ppm_{tag}"] = float(np.nanmedian(P[found, j])) if found.any() else np.nan
            ix = R[:, j]
            ix = ix[ix >= 0]
            role, role_share = _majority(series.role[ix])
            read, read_share = _majority(series.reading[ix])
            if role in ("iso_child", "artifact") and role_share > CLEAN_SHARE:
                clean = False
            if role == "M0" and read not in ("", want) and read_share > CLEAN_SHARE:
                clean = False
        both = (np.nan_to_num(Hb[:, j]) >= COVARY_MIN_HEIGHT) & (np.nan_to_num(Hc[:, j]) >= COVARY_MIN_HEIGHT)
        r["n_both"] = int(both.sum())
        r["r_log"] = (float(np.corrcoef(np.log10(Hb[both, j]), np.log10(Hc[both, j]))[0, 1])
                      if both.sum() >= COVARY_MIN_SPECTRA else np.nan)
        r["clean"] = clean
        r["nC_bare"], r["nC_cluster"] = int(nc_b[j]), int(nc_c[j])
        r["nC_obs_bare"], r["nC_obs_cluster"] = ob_b[j], ob_c[j]
        rec.append(r)
    meas = pd.DataFrame(rec)
    rel = pd.concat([meas["nC_obs_bare"] / meas["nC_bare"].replace(0, np.nan),
                     meas["nC_obs_cluster"] / meas["nC_cluster"].replace(0, np.nan)])
    scale = float(rel[rel > 0].median()) if (rel > 0).any() else np.nan
    meas["c13_scale"] = scale
    contra = np.zeros(len(meas), bool)
    for tag in ("bare", "cluster"):
        obs = meas[f"nC_obs_{tag}"] / scale
        n_c = meas[f"nC_{tag}"]
        bad = obs.notna() & ((obs - n_c).abs() > np.maximum(1.0, 0.2 * n_c))
        contra |= bad.to_numpy()
    meas["c13_contradicts"] = contra
    test = test.merge(meas, on="neutral_formula", how="left")
    test["bare"], test["cluster"] = bare, cluster
    test["present"] = ((test["det_bare"] >= PRESENCE_SHARE) & (test["ppm_bare"].abs() <= MEDIAN_PPM_MAX)
                       & (test["det_cluster"] >= PRESENCE_SHARE) & (test["ppm_cluster"].abs() <= MEDIAN_PPM_MAX))
    test["covary"] = test["r_log"].fillna(-9.0) >= COVARY_R_MIN
    test["upair"] = (test["cho"] & test["present"] & test["covary"] & test["clean"].astype(bool)
                     & ~test["c13_contradicts"].astype(bool))
    out = _finish(rows, test)
    log(f"[neutral_pairs] {bare} + {cluster}: {int(com['committed_both'].sum())} neutrals committed on both, "
        f"{int(out['upair'].sum())} establish their neutral (upair)")
    return out


def _finish(rows: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    parts = [p for p in (rows, test) if len(p)]
    if not parts:
        return _empty()
    out = pd.concat(parts, ignore_index=True, sort=False)
    for c in TABLE_COLUMNS:
        if c not in out.columns:
            out[c] = np.nan
    out["upair"] = out["upair"].fillna(False).astype(bool)
    return out[list(TABLE_COLUMNS)].sort_values("neutral_formula").reset_index(drop=True)


def neutrals(table: pd.DataFrame | None) -> set[str]:
    """The neutrals whose pair holds."""
    if table is None or not len(table) or "upair" not in table.columns:
        return set()
    return set(table.loc[table["upair"].astype(bool), "neutral_formula"].astype(str))


def summary(table: pd.DataFrame | None, pair) -> dict:
    """The funnel, for batch_summary.json."""
    pair = list(pair or ())
    if table is None or not len(table):
        return {"pair": pair, "neutrals": 0, "upair": 0}
    t = table
    both = t["committed_both"].astype(bool)
    cho = both & t["cho"].astype(bool)
    pres = cho & t["present"].fillna(False).astype(bool)
    cov = pres & t["covary"].fillna(False).astype(bool)
    cln = cov & t["clean"].fillna(False).astype(bool)
    return {"pair": pair, "neutrals": int(len(t)), "committed_both": int(both.sum()), "cho": int(cho.sum()),
            "present": int(pres.sum()), "covary": int(cov.sum()), "clean": int(cln.sum()),
            "upair": int(t["upair"].astype(bool).sum())}
