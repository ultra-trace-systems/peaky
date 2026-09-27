"""The isotope checks of a batch (C11+a; docs/EVIDENCE_LEVELS.md §3 `iso_veto`, §4.2).

A committed formula makes a claim about its isotope lines: how many carbons the
13C line counts, which heavy lines MUST be there, and how tall the M+2 region
may be. Three checks read those claims off the batch's stamped time series and
refute the formula where the series says otherwise -- a hard input (5b), and the
refuted pair leaves its neutral's chan2 / branch pools in both directions (rule
K's `alien` mechanism). All three are batch facts: they need the stamped time
series, never run per file, are never an axis, never in `cross`.

The instrument class is batch-generic: Orbitrap-class when the batch's peak-width
model resolves >= ORBITRAP_R200 at m/z 200 (`batch_summary.resolution.r_at_200`),
TOF-class otherwise; a batch with no width model is TOF-class (conservative: no
rule C).

  rule C  (the 13C carbon count; Orbitrap-class only) -- per (neutral, adduct)
        pair the batch stamped, not ion-only: in each spectrum the brightest M0
        stamp and the nearest peak to it + 1.0033548 within C_TOL_PPM; a spectrum
        is used when h0 x C x R13C >= C_KDL x its noise edge (1st-percentile
        height); >= C_NMIN used spectra. The pooled area ratio, corrected for the
        width ((m+1)/m)^1.5 and 17O (nO x R17O), and the pooled height ratio
        (17O only) read a carbon count each (/ R13C), with a delta-method se;
        both are divided by (1 + bias), the LEVEL-FREE median relative misread
        over every testable heteroatom-free pair. The formula is contradicted
        when area AND height both miss the ion's carbons by more than
        max(1.5, 0.25 C, 3 se); a 'too many' reading whose M+1 slot is another
        named pair's line in > C_OCC_MAX of the used spectra is untestable.
        Exempt: ions below the scan start + 1 Da (the batch's lowest m/z), and a
        14N nitrate cluster on a profile that also clusters on the 15N-labelled
        nitrate (its M+1 sits 6.32 mDa above the taller 15N sibling's line).
  REQ     (a required heavy line is absent) -- per pooled pair whose ION carries
        Br / Cl (TOF-class) or Br / Cl / S / Si (Orbitrap-class): the ion's
        count-aware fine structure, components closer than one FWHM of the width
        model merged into observable lines, placed relative to the STAMPED line
        (a Br2 ion is committed on its 79Br81Br line); required: every halogen
        group >= REQ_FRAC of the stamped line (81Br per Br, 37Cl per Cl, the
        Br2 1:2:1 and Cl2 patterns), and, on an Orbitrap-class batch without Br
        or Cl, 34S, 29Si and 30Si when the component is >= REQ_SHARE of its
        line. A line is detectable in a spectrum when its expected height >=
        REQ_DET_X x the spectrum's floor (the engine's height gate: noise edge x
        the batch's height_cutoff_x_edge); present when a peak sits within the
        window of the blend centroid or the pure component (Orbitrap-class
        max(1 ppm, 4 sigma); TOF-class the stamp window AND REQ_TOF_WIDE_PPM,
        absent only when absent at both). Refuted: >= REQ_NMIN detectable
        spectra and the line present in <= REQ_ABSENT_FRAC of them. A pair with
        no stamp takes the tallest peak within the stamp window of its pooled m/z.
  HIGH    (a heavy line too high for the formula; variant V4) -- per pooled
        pair: at any heavy offset (81Br, 37Cl, 34S, 30Si, 18O, 13C2) the nearest
        peak within HIGH_TOL_PPM (Orbitrap 1, TOF 10) co-varies with the M0
        (r(log area) >= HIGH_RMIN over >= HIGH_NMIN spectra, present in >=
        HIGH_FRAC of them) at a pooled area AND height ratio >= HIGH_XEXP x the
        ion's count-aware expected M+2 and >= HIGH_FLOOR -- unless the line is
        another committed M0 or a plain '13C' child in > HIGH_SLOT_MAX of its
        spectra (slot guards), sits nearer a non-isotopic +2 alias than the heavy
        spacing (Orbitrap-class), or runs above expected + HIGH_CAP (no isotope
        envelope is that tall).

Measured on the three regression batches through this engine (the offline replay
of the 2026-09-27 C17+U baselines): rule C vetoes
19 pairs on the labelled-nitrate Orbitrap (2.087 % of the batch signal) and 5 on
the uronium Orbitrap, none on the TOF (TOF-class); REQ 152 pooled / 104 merged
pairs on the labelled-nitrate batch (identified -0.947 points), 6 / 4 (all Si) on
the uronium batch, 29 merged on the TOF (both windows); HIGH V4 moves 7 aromatic
[M+^NO3]- rows 4b -> 5b on the labelled-nitrate batch (a C2 + 15N <-> H4 + Cl exact
alias: their 37Cl line is there), 1 on the uronium batch and 6 on the TOF, none
identified. The false-veto controls: 0 of 63 testable 3b acids under rule C, 1 in
~280 real lines absent at 1 ppm under REQ, 0 decoy flags per +-4/8 mDa offset
under HIGH V4 on both Orbitraps.
"""
from __future__ import annotations

import math
import warnings
from itertools import product

import numpy as np
import pandas as pd

from peaky.chem import chemistry as C

CHECKS = ("C", "REQ", "HIGH")
CHECK_NAME = {"C": "rule C", "REQ": "REQ", "HIGH": "HIGH"}
#: Orbitrap-class: the batch's width model resolves at least this at m/z 200
ORBITRAP_R200 = 50_000.0
D13C = 1.0033548378
# --- rule C
R13C = 0.010816
R17O = 0.000381
C_TOL_PPM = 3.0
C_KDL = 5.0
C_NMIN = 8
C_TOL_ABS, C_TOL_REL, C_TOL_SE = 1.5, 0.25, 3.0
C_OCC_MAX = 0.2
C_HETERO = ("F", "S", "Si", "Cl", "P", "I", "Br")
# --- REQ
REQ_FRAC = 0.25          # a halogen group is required when >= 0.25x the stamped line
REQ_SHARE = 0.5          # an S / Si component must be >= 50 % of its observable line
REQ_DET_X = 3.0          # detectable: expected height >= 3x the spectrum's floor
REQ_NMIN = 3
REQ_ABSENT_FRAC = 0.2
REQ_ORBI_MIN_PPM = 1.0
REQ_ORBI_SIGMA_K = 4.0
REQ_TOF_WIDE_PPM = 20.0
REQ_MERGE_FWHM = 1.0
_MIN_REL = 1e-5
_ISO = {
    "C": [(0.0, 0.98930), (1.0033548378, 0.01070, "13C")],
    "H": [(0.0, 0.999885), (1.0062767, 0.000115, "2H")],
    "N": [(0.0, 0.996360), (0.9970349, 0.003640, "15N")],
    "O": [(0.0, 0.997570), (1.0042169, 0.000380, "17O"), (2.0042464, 0.002050, "18O")],
    "S": [(0.0, 0.949900), (0.9993878, 0.007500, "33S"), (1.9957959, 0.042500, "34S")],
    "Cl": [(0.0, 0.757600), (1.9970499, 0.242400, "37Cl")],
    "Br": [(0.0, 0.506900), (1.9979521, 0.493100, "81Br")],
    "Si": [(0.0, 0.922230), (0.9995683, 0.046850, "29Si"), (1.9968442, 0.030920, "30Si")],
}
_MAXK = {"Br": 6, "Cl": 6, "C": 3, "Si": 3, "S": 2, "O": 2, "H": 1, "N": 1}
# --- HIGH
HIGH_TOL_PPM = {"orbitrap": 1.0, "tof": 10.0}
HIGH_PARENT_PPM = {"orbitrap": 2.0, "tof": 12.0}
HIGH_NMIN = 8
HIGH_FRAC, HIGH_RMIN, HIGH_XEXP, HIGH_FLOOR = 0.6, 0.8, 3.0, 0.2
HIGH_SLOT_MAX = 0.2
HIGH_CAP = 3.0
_A13, _A15, _A18 = 0.0107 / 0.9893, 0.00364 / 0.99636, 0.00205 / 0.99757
_A34, _A37, _A81, _A30 = 0.0425 / 0.9499, 0.2424 / 0.7576, 0.4931 / 0.5069, 0.03092 / 0.92223
HIGH_OFFSETS = {"81Br": 1.9979535, "37Cl": 1.9970499, "34S": 1.9957959, "30Si": 1.9968442, "18O": 2.0042463,
                "13C2": 2.0067097}
#: the element each heavy offset needs in the ion (18O / 13C2: any formula makes them)
_OFFSET_ELEMENT = {"81Br": "Br", "37Cl": "Cl", "34S": "S", "30Si": "Si"}
#: the non-isotopic +2 aliases an Orbitrap can tell from a heavy spacing
HIGH_ALIASES = {"F<->OH": 1.995660, "C3<->F2": 1.996810, "N2<->CH2O": 2.004410}

TABLE_COLUMNS = (
    "neutral_formula", "adduct", "check", "instrument", "ion", "mz", "stamped", "n_spectra", "n_used",
    "verdict", "veto",
    # rule C
    "n_carbon", "c_area", "c_height", "se_area", "se_height", "bias_area", "bias_height", "occupied",
    # REQ
    "line", "expected", "n_present", "det_frac", "window_ppm", "n_present_wide", "det_frac_wide",
    # HIGH
    "offset", "ratio_area", "ratio_height", "r", "presence", "offset_mda", "other_m0", "other_13c",
    "note",
)
VERDICTS = {
    "C": ("agree", "ambiguous", "contradict", "untestable", "scan_edge", "exempt_14N"),
    "REQ": ("present", "absent", "untestable"),
    "HIGH": ("consistent", "guarded", "too_high"),
}


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=list(TABLE_COLUMNS))


# --------------------------------------------------------------------------- the batch
def _resolution(resolution):
    from peaky.chem.resolution import Resolution
    if resolution is None:
        return None
    if isinstance(resolution, Resolution):
        return resolution
    if isinstance(resolution, dict):
        return Resolution.from_dict(resolution) if resolution.get("coef") is not None else None
    return Resolution.coerce(resolution)


def instrument_class(resolution) -> str:
    """'orbitrap' when the batch's width model resolves >= ORBITRAP_R200 at m/z
    200, else 'tof' -- and 'tof' when there is no width model at all."""
    rp = _resolution(resolution)
    if rp is None:
        return "tof"
    r = rp.r_at(200.0)
    return "orbitrap" if np.isfinite(r) and r >= ORBITRAP_R200 else "tof"


def _scale(mass_scale) -> tuple[float, float]:
    """(sigma_ppm, stamp_ppm) of the batch's traces.MassScale (or its as_dict)."""
    if mass_scale is None:
        return float("nan"), 6.0
    get = mass_scale.get if isinstance(mass_scale, dict) else (lambda k, d=None: getattr(mass_scale, k, d))
    sigma = get("sigma_ppm")
    stamp = get("stamp_ppm")
    sigma = float(sigma) if sigma is not None and np.isfinite(float(sigma)) else float("nan")
    stamp = float(stamp) if stamp is not None and np.isfinite(float(stamp)) else float(get("tol_ppm", 6.0) or 6.0)
    return sigma, stamp


class _Series:
    """The stamped batch time series, one array per column, sorted by (spectrum,
    m/z), with each spectrum's noise edge (its 1st-percentile height)."""

    def __init__(self, ts: pd.DataFrame):
        t = ts.copy()
        t["mz"] = pd.to_numeric(t["mz"], errors="coerce")
        t["height"] = pd.to_numeric(t["height"], errors="coerce")
        t["area"] = pd.to_numeric(t["area"], errors="coerce") if "area" in t.columns else t["height"]
        t = t[np.isfinite(t["mz"]) & (t["height"] > 0) & (t["area"] > 0)]
        for c in ("role", "neutral_formula", "adduct", "iso_label", "ion_formula"):
            t[c] = t[c].astype(object).where(t[c].notna(), "").astype(str) if c in t.columns else ""
        self.sids = np.array(sorted(t["sample_item_id"].astype(str).unique()))
        code = {s: i for i, s in enumerate(self.sids)}
        t["code"] = t["sample_item_id"].astype(str).map(code).astype(int)
        t = t.sort_values(["code", "mz"], kind="mergesort").reset_index(drop=True)
        self.t = t
        self.code = t["code"].to_numpy()
        self.mz = t["mz"].to_numpy(float)
        self.h = t["height"].to_numpy(float)
        self.a = t["area"].to_numpy(float)
        self.role = t["role"].to_numpy()
        self.nf = t["neutral_formula"].to_numpy()
        self.ad = t["adduct"].to_numpy()
        self.label = t["iso_label"].to_numpy()
        self.key = self.code * 1e4 + self.mz
        self.n = len(self.sids)
        edge = t.groupby("code")["height"].apply(lambda h: float(np.percentile(h.to_numpy(float), 1.0)))
        self.edge = edge.reindex(range(self.n)).to_numpy(float)
        self.scan_start = float(self.mz.min()) if len(self.mz) else float("nan")
        # the brightest M0 stamp of each (neutral, adduct) per spectrum
        m0 = t[(t["role"] == "M0") & (t["neutral_formula"] != "") & (t["adduct"] != "")]
        m0 = m0.sort_values("height", ascending=False, kind="mergesort").drop_duplicates(
            ["neutral_formula", "adduct", "code"])
        m0 = m0.sort_values(["neutral_formula", "adduct", "code"], kind="mergesort")
        self.m0 = {k: g for k, g in m0.groupby(["neutral_formula", "adduct"], sort=False)}
        # pair ids of every stamped row (the slot guards)
        pk = pd.Series(list(zip(self.nf, self.ad)))
        self.pair_code, self.pair_index = pd.factorize(pk)

    def window(self, codes, targets, ppm):
        """(lo, hi) index ranges of the peaks within +-ppm of each (spectrum, target)."""
        codes = np.asarray(codes)
        targets = np.asarray(targets, float)
        tol = targets * ppm * 1e-6
        lo = np.searchsorted(self.key, codes * 1e4 + targets - tol, "left")
        hi = np.searchsorted(self.key, codes * 1e4 + targets + tol, "right")
        return lo, hi

    def tallest(self, codes, targets, ppm):
        """Index of the tallest peak within +-ppm of each (spectrum, target), -1 where none."""
        lo, hi = self.window(codes, targets, ppm)
        out = np.full(len(lo), -1, dtype=np.int64)
        for i in np.nonzero(hi > lo)[0]:
            out[i] = lo[i] + int(np.argmax(self.h[lo[i]:hi[i]]))
        return out

    def nearest(self, codes, targets, ppm_of=None, ppm=None, reach=(-1, 0)):
        """Index of the peak nearest each (spectrum, target) in the same spectrum
        (-1 where the spectrum is empty or, with `ppm`, nothing sits within ppm x
        `ppm_of` / 1e6 of the target)."""
        codes = np.asarray(codes)
        targets = np.asarray(targets, float)
        idx = np.searchsorted(self.key, codes * 1e4 + targets)
        best = np.full(len(targets), -1, dtype=np.int64)
        bestd = np.full(len(targets), np.inf)
        for d in reach:
            k = np.clip(idx + d, 0, max(len(self.key) - 1, 0))
            if not len(self.key):
                break
            ok = self.code[k] == codes
            dist = np.abs(self.mz[k] - targets)
            upd = ok & (dist < bestd)
            best[upd], bestd[upd] = k[upd], dist[upd]
        if ppm is not None:
            ref = targets if ppm_of is None else np.asarray(ppm_of, float)
            best[~(bestd / ref * 1e6 <= ppm)] = -1
        return best


def _pooled(frames: dict) -> pd.DataFrame:
    """One row per (neutral, adduct) the pooled per-file ledgers commit: the ion
    formula of its first M0 row and the median m/z (as evidence._measure pools
    them), and whether it is an ion-only pair."""
    from peaky.assignment import evidence as EV
    parts = [f for f in (frames or {}).values() if f is not None and len(f) and "role" in f.columns]
    cols = ["neutral_formula", "adduct", "ion", "mz", "ion_only"]
    if not parts:
        return pd.DataFrame(columns=cols)
    frame = pd.concat(parts, ignore_index=True, sort=False)
    m0 = frame[frame["role"].astype(str) == "M0"].copy()
    if m0.empty:
        return pd.DataFrame(columns=cols)
    m0["__n"] = m0["neutral_formula"].fillna("").astype(str) if "neutral_formula" in m0.columns else ""
    m0["__a"] = m0["adduct"].fillna("").astype(str) if "adduct" in m0.columns else ""
    m0["__io"] = EV.is_ion_only(m0).to_numpy()
    m0["__mz"] = pd.to_numeric(m0["mz"], errors="coerce") if "mz" in m0.columns else np.nan
    m0["__ion"] = m0["ion_formula"].astype(str) if "ion_formula" in m0.columns else ""
    g = m0.groupby(["__n", "__a"], sort=True)
    out = pd.DataFrame({"ion": g["__ion"].first(), "mz": g["__mz"].median(), "ion_only": g["__io"].any()})
    out = out.reset_index().rename(columns={"__n": "neutral_formula", "__a": "adduct"})
    return out[(out["neutral_formula"] != "") & (out["adduct"] != "")].reset_index(drop=True)[cols]


def ion_counts(neutral: str, adduct: str, ion) -> dict:
    """The ion's composition: its ion formula when it carries a charge sign, else
    neutral + adduct (a ledger can hold the NEUTRAL in `ion_formula`)."""
    s = str(ion or "").strip()
    if s.endswith(("-", "+")):
        return C.parse_formula(s)
    from peaky.assignment.tiers import _ion_counts
    return _ion_counts(neutral, adduct) or C.parse_formula(s)


# --------------------------------------------------------------------------- rule C
def _labelled_sibling(adduct: str, prof) -> bool:
    """A 14N nitrate cluster on a profile that also clusters on the 15N-labelled
    nitrate: the M+1 slot is the 15N sibling's line (6.32 mDa below it)."""
    from peaky.assignment import evidence as EV
    lab = EV.LABEL_FOLD.get(adduct)
    return bool(lab and lab != adduct and lab in set(getattr(prof, "adducts", None) or ()))


def _rule_c(S: _Series, pooled: pd.DataFrame, prof) -> pd.DataFrame:
    scope = set(zip(pooled.loc[~pooled["ion_only"].astype(bool), "neutral_formula"],
                    pooled.loc[~pooled["ion_only"].astype(bool), "adduct"]))
    keys = [k for k in S.m0 if k in scope]
    if not keys:
        return _empty()
    rows = []
    for k in keys:
        g = S.m0[k]
        codes = g["code"].to_numpy()
        mz0 = g["mz"].to_numpy(float)
        j = S.nearest(codes, mz0 + D13C, ppm=C_TOL_PPM)
        hit = j >= 0
        jj = np.where(hit, j, 0)
        h1 = np.where(hit, S.h[jj], 0.0)
        a1 = np.where(hit, S.a[jj], 0.0)
        same = (S.nf[jj] == k[0]) & (S.ad[jj] == k[1])
        occ = hit & (((S.role[jj] == "M0") & ~same) | ((S.role[jj] == "iso_child") & ~same & (S.nf[jj] != "")))
        ion = g["ion_formula"].mode()
        ion = str(ion.iloc[0]) if len(ion) else ""
        ic = _parse_ion(ion)
        nC = int(ic.get("C", 0))
        if nC < 1:
            continue
        mzm = float(np.median(mz0))
        wf = ((mzm + 1.0) / mzm) ** 1.5
        oth = ic.get("O", 0) * R17O
        h0, a0 = g["height"].to_numpy(float), g["area"].to_numpy(float)
        e = S.edge[codes]
        u = h0 * nC * R13C >= C_KDL * e
        n = int(u.sum())
        rec = dict(neutral_formula=k[0], adduct=k[1], ion=ion, mz=mzm, n_spectra=int(len(g)), n_used=n,
                   n_carbon=nC, het=sum(C.parse_formula(k[0]).get(x, 0) for x in C_HETERO))
        if n > 0:
            for nm, x0, x1, w in (("area", a0[u], a1[u], wf), ("height", h0[u], h1[u], 1.0)):
                s0, s1 = x0.sum(), x1.sum()
                rr = s1 / s0
                res = x1 - rr * x0
                se = np.sqrt(n / max(n - 1, 1) * (res ** 2).sum()) / s0 if n > 1 else np.nan
                rec[f"c_{nm}_raw"] = (rr / w - oth) / R13C
                rec[f"se_{nm}"] = se / w / R13C
            rec["occupied"] = float(occ[u].mean())
        rows.append(rec)
    d = pd.DataFrame(rows)
    for c in ("c_area_raw", "c_height_raw", "se_area", "se_height", "occupied"):
        if c not in d.columns:
            d[c] = np.nan
    edge = d["mz"] < S.scan_start + 1.0
    # the level-free bias: the median relative misread over every testable
    # heteroatom-free pair (the same veto set as a bias fitted on levelled acids)
    pop = d[(d["het"] == 0) & (d["n_used"] >= C_NMIN) & ~edge]
    bias = {u: (float(((pop[f"c_{u}_raw"] - pop["n_carbon"]) / pop["n_carbon"]).median()) if len(pop) else 0.0)
            for u in ("area", "height")}
    d["bias_area"], d["bias_height"] = bias["area"], bias["height"]
    d["c_area"] = d["c_area_raw"] / (1.0 + bias["area"])
    d["c_height"] = d["c_height_raw"] / (1.0 + bias["height"])
    nc = d["n_carbon"].astype(float)
    tola = np.maximum.reduce([np.full(len(d), C_TOL_ABS), C_TOL_REL * nc, C_TOL_SE * d["se_area"].fillna(0)])
    tolh = np.maximum.reduce([np.full(len(d), C_TOL_ABS), C_TOL_REL * nc, C_TOL_SE * d["se_height"].fillna(0)])
    oa = (d["c_area"] - nc).abs() > tola
    oh = (d["c_height"] - nc).abs() > tolh
    v = np.where(d["n_used"] < C_NMIN, "untestable", np.where(~oa, "agree", np.where(oh, "contradict", "ambiguous")))
    v = pd.Series(v, index=d.index, dtype=object)
    v[(v == "contradict") & (d["c_area"] > nc) & (d["occupied"] > C_OCC_MAX)] = "untestable"
    ex14 = d["adduct"].map(lambda a: _labelled_sibling(a, prof))
    d["verdict"] = np.where(edge, "scan_edge", np.where(ex14, "exempt_14N", v))
    d["veto"] = d["verdict"].eq("contradict")
    d["note"] = [_c_note(r) for r in d.itertuples(index=False)]
    d["check"] = "C"
    d["stamped"] = True
    return d


def _parse_ion(ion: str) -> dict:
    return C.parse_formula(str(ion).rstrip("+-."))


def _c_note(r) -> str:
    finite = isinstance(r.c_area, float) and np.isfinite(r.c_area)
    txt = (f"13C reads {r.c_area:.1f} C for {r.n_carbon} (height {r.c_height:.1f}, se {r.se_area:.1f}) "
           f"over {r.n_used} spectra") if finite else ""
    why = {"scan_edge": "below the scan start + 1 Da, not tested",
           "exempt_14N": "its M+1 slot is the 15N sibling's line, not tested"}.get(r.verdict)
    if why:
        return f"{txt}; {why}" if txt else why
    if not finite or r.n_used < C_NMIN:
        return f"{r.n_used} of {r.n_spectra} spectra bright enough for the 13C line (needs {C_NMIN})"
    if r.verdict == "untestable":
        return txt + f"; the M+1 slot is another pair's line in {r.occupied:.0%} of them"
    return txt


# --------------------------------------------------------------------------- REQ
def _element_dist(el: str, n: int):
    lev = _ISO[el]
    a0 = lev[0][1]
    heavy = lev[1:]
    kmax = min(n, _MAXK.get(el, 2))
    out = []
    for ks in product(*[range(kmax + 1)] * len(heavy)):
        t = sum(ks)
        if t > kmax:
            continue
        coef = math.comb(n, t) * math.factorial(t) / math.prod(math.factorial(k) for k in ks)
        rel = coef * float(np.prod([(h[1] / a0) ** k for h, k in zip(heavy, ks)]))
        if rel < _MIN_REL:
            continue
        out.append((sum(h[0] * k for h, k in zip(heavy, ks)), rel, {h[2]: k for h, k in zip(heavy, ks) if k}))
    return out


def fine_structure(counts: dict) -> pd.DataFrame:
    """The ion's isotopologues relative to its all-light line (count-aware,
    multinomial per element; components < 1e-5 dropped): shift, rel, tags."""
    comps = [(0.0, 1.0, {})]
    for el, n in counts.items():
        if el not in _ISO or n <= 0:
            continue
        new = []
        for s0, r0, t0 in comps:
            for s1, r1, t1 in _element_dist(el, int(n)):
                r = r0 * r1
                if r < _MIN_REL or s0 + s1 > 8.5:
                    continue
                t = dict(t0)
                t.update(t1)
                new.append((s0 + s1, r, t))
        comps = new
    df = pd.DataFrame({"shift": [c[0] for c in comps], "rel": [c[1] for c in comps], "tags": [c[2] for c in comps]})
    return df.sort_values("shift", kind="mergesort").reset_index(drop=True)


def _lines(fs: pd.DataFrame, fwhm: float):
    """Observable lines: consecutive components closer than one FWHM merge."""
    g = np.concatenate([[0], np.cumsum(np.diff(fs["shift"].to_numpy()) >= REQ_MERGE_FWHM * fwhm)]).astype(int)
    fs = fs.assign(line=g)
    w = fs["rel"].to_numpy()
    rel = np.bincount(g, weights=w)
    cen = np.bincount(g, weights=w * fs["shift"].to_numpy()) / rel
    return fs, cen, rel


def required_lines(counts: dict, fwhm: float, stamped_shift: float, tof: bool):
    """(stamped centroid, [dict(element, label, centroid, pure, ratio, share)]) --
    the heavy lines the ion's formula requires, relative to the stamped line."""
    fs, cen, rel = _lines(fine_structure(counts), fwhm)
    sl = int(np.argmin(np.abs(cen - stamped_shift)))
    sc, sr = float(cen[sl]), float(rel[sl])
    req = []
    nbr, ncl = counts.get("Br", 0), counts.get("Cl", 0)
    if nbr + ncl > 0:
        halo = fs[[all(k in ("81Br", "37Cl") for k in t) for t in fs["tags"]]]
        nominal = [int(t.get("81Br", 0) + t.get("37Cl", 0)) for t in halo["tags"]]
        for k, g in halo.groupby(nominal):
            main = g.loc[g["rel"].idxmax()]
            ln = int(main["line"])
            if ln == sl:
                continue
            ratio = float(rel[ln]) / sr
            if ratio < REQ_FRAC:
                continue
            el = "BrCl" if (nbr and ncl) else ("Br" if nbr else "Cl")
            lab = (f"M+{2 * k}" if k else "M0") + (f" ({_tag_text(main['tags'])})" if main["tags"] else " (all-light)")
            req.append(dict(element=el, label=lab, centroid=float(cen[ln]), pure=float(main["shift"]), ratio=ratio,
                            share=float(g["rel"].sum() / rel[ln])))
        return sc, req
    if tof:
        return sc, req        # 34S / 29Si / 30Si are unresolved inside a TOF's M+1 / M+2 clusters
    for el, lab in (("S", "34S"), ("Si", "29Si"), ("Si", "30Si")):
        if counts.get(el, 0) <= 0:
            continue
        m = fs[[t == {lab: 1} for t in fs["tags"]]]
        if m.empty:
            continue
        main = m.iloc[0]
        ln = int(main["line"])
        if ln == sl:
            continue
        share = float(main["rel"] / rel[ln])
        if share >= REQ_SHARE:
            req.append(dict(element=el, label=lab, centroid=float(cen[ln]), pure=float(main["shift"]),
                            ratio=float(rel[ln]) / sr, share=share))
    return sc, req


def _tag_text(tags: dict) -> str:
    return " ".join(f"{k}" if v == 1 else f"{v}x{k}" for k, v in tags.items())


def _req(S: _Series, pooled: pd.DataFrame, klass: str, rp, sigma: float, stamp: float,
         x_edge: float) -> pd.DataFrame:
    tof = klass == "tof"
    heavy = ("Br", "Cl") if tof else ("Br", "Cl", "S", "Si")
    win = stamp if tof else max(REQ_ORBI_MIN_PPM, REQ_ORBI_SIGMA_K * sigma if np.isfinite(sigma) else 0.0)
    floor = S.edge * x_edge
    rows = []
    for r in pooled.itertuples(index=False):
        counts = ion_counts(r.neutral_formula, r.adduct, r.ion)
        if not any(counts.get(el, 0) > 0 for el in heavy):
            continue
        try:
            mz0 = C.ion_mz(r.neutral_formula, r.adduct)
        except Exception:
            continue
        g = S.m0.get((r.neutral_formula, r.adduct))
        stamped = g is not None and len(g) > 0
        if stamped:
            codes, pm, ph = g["code"].to_numpy(), g["mz"].to_numpy(float), g["height"].to_numpy(float)
        else:
            if not np.isfinite(r.mz):
                continue
            codes = np.arange(S.n)
            j = S.tallest(codes, np.full(S.n, float(r.mz)), stamp)
            ok = j >= 0
            codes, pm, ph = codes[ok], S.mz[j[ok]], S.h[j[ok]]
            if not len(codes):
                continue
        sc, req = required_lines(counts, rp.fwhm(mz0), float(np.median(pm)) - mz0, tof)
        if not req:
            continue
        fl = floor[codes]
        lines = []
        for q in req:
            det = ph * q["ratio"] >= REQ_DET_X * fl
            nd = int(det.sum())
            t1 = pm + (q["centroid"] - sc)
            t2 = pm + (q["pure"] - sc)
            pres = _present(S, codes, t1, t2, win)
            npd = int((pres & det).sum())
            frac = npd / nd if nd else np.nan
            testable = nd >= REQ_NMIN
            absent = testable and frac <= REQ_ABSENT_FRAC
            npw = frw = np.nan
            if tof:
                presw = pres | _present(S, codes, t1, t2, REQ_TOF_WIDE_PPM)
                npw = int((presw & det).sum())
                frw = npw / nd if nd else np.nan
                absent = absent and frw <= REQ_ABSENT_FRAC
            lines.append(dict(q, n_det=nd, n_present=npd, det_frac=frac, n_present_wide=npw, det_frac_wide=frw,
                              testable=testable, absent=absent))
        ab = [x for x in lines if x["absent"]]
        tst = [x for x in lines if x["testable"]]
        pick = (min(ab, key=lambda x: (x["det_frac"], -x["n_det"])) if ab else
                min(tst, key=lambda x: (x["det_frac"], -x["n_det"])) if tst else
                max(lines, key=lambda x: x["n_det"]))
        verdict = "absent" if ab else ("present" if tst else "untestable")
        note = "; ".join(_req_note(x, tof, win) for x in ab) if ab else _req_note(pick, tof, win)
        rows.append(dict(neutral_formula=r.neutral_formula, adduct=r.adduct, ion=str(r.ion), mz=float(np.median(pm)),
                         stamped=stamped, n_spectra=int(len(codes)), n_used=pick["n_det"], line=pick["label"],
                         expected=pick["ratio"], n_present=pick["n_present"], det_frac=pick["det_frac"],
                         window_ppm=win, n_present_wide=pick["n_present_wide"], det_frac_wide=pick["det_frac_wide"],
                         verdict=verdict, veto=bool(ab), note=note))
    if not rows:
        return _empty()
    d = pd.DataFrame(rows)
    d["check"] = "REQ"
    return d


def _present(S: _Series, codes, t1, t2, ppm) -> np.ndarray:
    lo1, hi1 = S.window(codes, t1, ppm)
    lo2, hi2 = S.window(codes, t2, ppm)
    return (hi1 > lo1) | (hi2 > lo2)


def _req_note(x: dict, tof: bool, win: float) -> str:
    what = f"the {x['label']} line ({x['ratio']:.2f}x the stamped line)"
    if not x["testable"]:
        return f"{what} detectable in {x['n_det']} spectra (needs {REQ_NMIN})"
    where = (f"within {win:.3g} and {REQ_TOF_WIDE_PPM:g} ppm" if tof else f"within {win:.3g} ppm")
    seen = x["n_present_wide"] if tof else x["n_present"]
    if x["absent"]:
        return f"{what} absent in {x['n_det'] - int(seen)} of {x['n_det']} detectable spectra ({where})"
    return f"{what} present in {int(seen)} of {x['n_det']} detectable spectra ({where})"


# --------------------------------------------------------------------------- HIGH
def expected_m2(ion: dict) -> float:
    """The ion's count-aware expected M+2 / M0 (first order per element, plus
    the 13C2 and 13C15N pairs)."""
    nc, nn = ion.get("C", 0), ion.get("N", 0)
    return (ion.get("Br", 0) * _A81 + ion.get("Cl", 0) * _A37 + ion.get("S", 0) * _A34
            + ion.get("Si", 0) * _A30 + ion.get("O", 0) * _A18 + nc * (nc - 1) / 2 * _A13 ** 2
            + nc * nn * _A13 * _A15)


def _high(S: _Series, pooled: pd.DataFrame, klass: str) -> pd.DataFrame:
    tol, par = HIGH_TOL_PPM[klass], HIGH_PARENT_PPM[klass]
    keys = list(zip(pooled["neutral_formula"], pooled["adduct"]))
    P = len(keys)
    if not P or not S.n:
        return _empty()
    PM = np.full((S.n, P), np.nan)
    PA, PH = PM.copy(), PM.copy()
    stamped = np.zeros(P, bool)
    for i, k in enumerate(keys):
        g = S.m0.get(k)
        if g is not None and len(g):
            c = g["code"].to_numpy()
            PM[c, i], PA[c, i], PH[c, i] = g["mz"].to_numpy(float), g["area"].to_numpy(float), g["height"].to_numpy(float)
            stamped[i] = True
    for i in np.flatnonzero(~stamped):
        mz = float(pooled["mz"].iat[i])
        if not np.isfinite(mz):
            continue
        codes = np.arange(S.n)
        j = S.tallest(codes, np.full(S.n, mz), par)
        ok = j >= 0
        PM[codes[ok], i], PA[codes[ok], i], PH[codes[ok], i] = S.mz[j[ok]], S.a[j[ok]], S.h[j[ok]]
    npar = np.isfinite(PM).sum(0)
    ions = [ion_counts(n, a, x) for (n, a), x in zip(keys, pooled["ion"])]
    exp = np.array([expected_m2(x) for x in ions])
    # the pair id of each column in the series' factorized (neutral, adduct) codes (-2: never stamped)
    pid = {k: i for i, k in enumerate(S.pair_index)}
    pcol = np.array([pid.get(k, -2) for k in keys])
    is_m0 = S.role == "M0"
    is_13c = (S.role == "iso_child") & (S.label == "13C")
    have = np.isfinite(PM)
    cc, pp = np.nonzero(have)
    per = {}
    for lab, D in HIGH_OFFSETS.items():
        ci = np.full(PM.shape, -1, dtype=np.int64)
        if len(cc):
            pm = PM[cc, pp]
            ci[cc, pp] = S.nearest(cc, pm + D, ppm_of=pm, ppm=tol, reach=(-2, -1, 0, 1))
        hasc = ci >= 0
        cj = np.where(hasc, ci, 0)
        CM = np.where(hasc, S.mz[cj], np.nan)
        CA = np.where(hasc, S.a[cj], np.nan)
        CH = np.where(hasc, S.h[cj], np.nan)
        both = hasc & have
        n = both.sum(0)
        frac = np.where(npar > 0, n / np.maximum(npar, 1), 0.0)
        ra = np.where(n > 0, np.where(both, CA, 0).sum(0) / np.maximum(np.where(both, PA, 0).sum(0), 1e-12), np.nan)
        rh = np.where(n > 0, np.where(both, CH, 0).sum(0) / np.maximum(np.where(both, PH, 0).sum(0), 1e-12), np.nan)
        X = np.where(both, np.log(np.where(both, PA, 1.0)), np.nan)
        Y = np.where(both, np.log(np.where(both, CA, 1.0)), np.nan)
        with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)       # a column with no spectrum holding both
            xm, ym = np.nanmean(X, 0), np.nanmean(Y, 0)
            cov = np.nansum((X - xm) * (Y - ym), 0)
            vx, vy = np.nansum((X - xm) ** 2, 0), np.nansum((Y - ym) ** 2, 0)
            r = np.where((n >= HIGH_NMIN) & (vx > 0) & (vy > 0), cov / np.sqrt(vx * vy), np.nan)
            off = np.nanmedian(np.where(both, (CM - PM - D) * 1e3, np.nan), 0)
        nh = hasc.sum(0)
        oth = np.where(nh > 0, (hasc & is_m0[cj] & (S.pair_code[cj] != pcol[None, :])).sum(0) / np.maximum(nh, 1), 0.0)
        o13 = np.where(nh > 0, (hasc & is_13c[cj]).sum(0) / np.maximum(nh, 1), 0.0)
        v0 = ((npar >= HIGH_NMIN) & (frac >= HIGH_FRAC) & (r >= HIGH_RMIN) & (ra >= HIGH_FLOOR) & (rh >= HIGH_FLOOR)
              & (ra >= HIGH_XEXP * exp) & (rh >= HIGH_XEXP * exp))
        if klass == "orbitrap":
            pos = D + off / 1e3
            da = np.min([np.abs(pos - a) for a in HIGH_ALIASES.values()], axis=0)
            alias = np.where(np.isnan(pos), False, da < np.abs(off) / 1e3)
        else:
            alias = np.zeros(P, bool)
        cap = (ra <= exp + HIGH_CAP) & (rh <= exp + HIGH_CAP)
        v4 = v0 & (oth <= HIGH_SLOT_MAX) & ~alias & cap & (o13 <= HIGH_SLOT_MAX)
        per[lab] = dict(n=n, frac=frac, ra=ra, rh=rh, r=r, off=off, oth=oth, o13=o13, alias=alias, cap=cap,
                        v0=v0, v4=v4)
    rows = []
    for i in np.flatnonzero(npar >= HIGH_NMIN):
        hit = [lab for lab in HIGH_OFFSETS if per[lab]["v4"][i]]
        cand = [lab for lab in HIGH_OFFSETS if per[lab]["v0"][i]]
        if hit:
            lab, verdict = max(hit, key=lambda x: per[x]["ra"][i]), "too_high"
        elif cand:
            lab, verdict = max(cand, key=lambda x: per[x]["ra"][i]), "guarded"
        else:
            fin = [x for x in HIGH_OFFSETS if np.isfinite(per[x]["ra"][i])]
            lab = max(fin, key=lambda x: per[x]["ra"][i] / max(exp[i], 1e-12)) if fin else "81Br"
            verdict = "consistent"
        st = per[lab]
        rows.append(dict(neutral_formula=keys[i][0], adduct=keys[i][1], ion=str(pooled["ion"].iat[i]),
                         mz=float(np.nanmedian(PM[:, i])), stamped=bool(stamped[i]), n_spectra=int(npar[i]),
                         n_used=int(st["n"][i]), expected=float(exp[i]), offset=lab, ratio_area=float(st["ra"][i]),
                         ratio_height=float(st["rh"][i]), r=float(st["r"][i]), presence=float(st["frac"][i]),
                         offset_mda=float(st["off"][i]), other_m0=float(st["oth"][i]), other_13c=float(st["o13"][i]),
                         verdict=verdict, veto=verdict == "too_high",
                         note=_high_note(st, i, lab, exp[i], int(npar[i]), ions[i], verdict)))
    if not rows:
        return _empty()
    d = pd.DataFrame(rows)
    d["check"] = "HIGH"
    return d


def _high_note(st: dict, i: int, lab: str, exp: float, npar: int, ion: dict, verdict: str) -> str:
    ra = st["ra"][i]
    if not np.isfinite(ra):
        return f"no line at any heavy offset in {npar} spectra"
    txt = (f"a {ra:.2f}x line at the {lab} offset ({ra / max(exp, 1e-12):.0f}x the formula's M+2 {exp:.3f}), "
           f"r {st['r'][i]:.2f}, in {st['frac'][i]:.0%} of {npar} spectra")
    if verdict == "too_high":
        el = _OFFSET_ELEMENT.get(lab)
        if el and not ion.get(el, 0):
            return txt + f": the ion carries {el} the formula lacks"
        return txt + ": more than the formula can make"
    if verdict == "guarded":
        why = []
        if st["oth"][i] > HIGH_SLOT_MAX:
            why.append(f"another committed M0 in {st['oth'][i]:.0%} of its spectra")
        if st["o13"][i] > HIGH_SLOT_MAX:
            why.append(f"a plain 13C line in {st['o13'][i]:.0%} of its spectra")
        if st["alias"][i]:
            why.append("nearer a non-isotopic +2 alias than the heavy spacing")
        if not st["cap"][i]:
            why.append(f"above expected + {HIGH_CAP:g}: no isotope envelope")
        return txt + "; not an isotope line of this ion (" + "; ".join(why) + ")"
    return txt


# --------------------------------------------------------------------------- the table
def measure(ts: pd.DataFrame | None, frames: dict, prof=None, *, resolution=None, mass_scale=None,
            x_edge: float = 1.0, log=print) -> pd.DataFrame:
    """The isotope-check table: one row per tested pooled pair and check (module
    docstring). `resolution` is the batch's width model (chem.resolution.Resolution
    or its as_dict; it decides the instrument class and REQ's observable lines),
    `mass_scale` the batch's traces.MassScale (sigma_ppm, stamp_ppm), `x_edge` the
    batch's height_cutoff_x_edge (the height gate each spectrum's floor is: its
    noise edge, the 1st-percentile height, x x_edge -- the per-file
    height_gate_cps where a file is also a ledger). Empty with a header when the
    batch has no time series or no committed pair."""
    if ts is None or not len(ts):
        return _empty()
    pooled = _pooled(frames)
    if pooled.empty:
        return _empty()
    klass = instrument_class(resolution)
    rp = _resolution(resolution)
    sigma, stamp = _scale(mass_scale)
    S = _Series(ts)
    if not S.n:
        return _empty()
    try:
        x_edge = float(x_edge)
    except (TypeError, ValueError):     # an unresolved 'auto' multiple reads as the edge itself
        x_edge = 1.0
    x_edge = x_edge if np.isfinite(x_edge) and x_edge > 0 else 1.0
    parts = []
    if klass == "orbitrap":
        parts.append(_rule_c(S, pooled, prof))
    if rp is not None:
        parts.append(_req(S, pooled, klass, rp, sigma, stamp, x_edge))
    parts.append(_high(S, pooled, klass))
    parts = [p for p in parts if p is not None and len(p)]
    if not parts:
        return _empty()
    out = pd.concat(parts, ignore_index=True, sort=False)
    out["instrument"] = klass
    for c in TABLE_COLUMNS:
        if c not in out.columns:
            out[c] = np.nan
    out["veto"] = out["veto"].fillna(False).astype(bool)
    out["stamped"] = out["stamped"].fillna(False).astype(bool)
    out["note"] = out["note"].fillna("")
    order = {c: i for i, c in enumerate(CHECKS)}
    out = out[list(TABLE_COLUMNS)]
    out = out.assign(__o=out["check"].map(order)).sort_values(["__o", "neutral_formula", "adduct"], kind="mergesort")
    out = out.drop(columns="__o").reset_index(drop=True)
    log(f"[iso_checks] {klass}-class batch: "
        + "; ".join(f"{CHECK_NAME[c]} {int((out['check'] == c).sum())} tested, "
                    f"{int((out['veto'] & (out['check'] == c)).sum())} refuted" for c in CHECKS)
        + f" -> {len(veto(out))} pair(s) vetoed"
        + ("" if klass == "orbitrap" else " (TOF-class: no rule C)")
        + ("" if rp is not None else " (no width model: no REQ)"))
    return out


def veto(table: pd.DataFrame | None) -> dict:
    """{(neutral, adduct): note} the checks refute -- one joined note per pair
    ('rule C: ...; REQ: ...') in the order of CHECKS."""
    if table is None or not len(table) or "veto" not in table.columns:
        return {}
    t = table[table["veto"].map(_truth)]
    out: dict = {}
    order = {c: i for i, c in enumerate(CHECKS)}
    t = t.assign(__o=t["check"].map(lambda c: order.get(str(c), len(order)))).sort_values("__o", kind="mergesort")
    for n, a, c, x in zip(t["neutral_formula"], t["adduct"], t["check"], t["note"]):
        k = (str(n), str(a))
        piece = f"{CHECK_NAME.get(str(c), str(c))}: {x}" if isinstance(x, str) and x else CHECK_NAME.get(str(c), str(c))
        out[k] = f"{out[k]}; {piece}" if k in out else piece
    return out


def facts(table: pd.DataFrame | None) -> dict | None:
    """What evidence.level_pooled reads (its `iso=`): {'veto': {(n, a): note}} --
    None for an empty table (no time series, nothing committed)."""
    if table is None or not len(table):
        return None
    return {"veto": veto(table)}


def _truth(v) -> bool:
    if isinstance(v, str):
        return v.strip().lower() in ("true", "1", "yes")
    return bool(v) if v is not None and not (isinstance(v, float) and np.isnan(v)) else False


def summary(table: pd.DataFrame | None, resolution=None) -> dict:
    """The funnel, for batch_summary.json: per check the pairs tested and each
    verdict's count, and the pairs vetoed in all."""
    klass = instrument_class(resolution)
    if table is None or not len(table):
        return {"instrument": klass, "tested": 0, "vetoed_pairs": 0}
    out = {"instrument": klass, "tested": int(len(table)), "vetoed_pairs": len(veto(table))}
    for c in CHECKS:
        t = table[table["check"] == c]
        v = t["verdict"].astype(str)
        out[c] = {"tested": int(len(t)), "vetoed": int(t["veto"].map(_truth).sum()),
                  **{k: int((v == k).sum()) for k in VERDICTS[c]}}
    return out
