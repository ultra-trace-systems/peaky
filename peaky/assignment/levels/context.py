"""The run context of the evidence scale: everything one SOURCE (a batch's
pooled files, one file, or a decoy arm) needs before its pairs are levelled.

A `RunContext` carries the width model and instrument class, the per-file peak
arrays (`FileArr`: every row with height > 0, its gate and scan edges), the
calibrated mass window of every file and of the run, the enumeration space and
channels, the isotope position window (`position_sigma`), the measured
per-element line efficiency, the observed carbon counts (+ their single-file se
model), the labelled reagent's 15N twin ratios and the indexes of committed
readings and their predicted isotope lines.

The calibrated window of a file is the degeneracy audit's own calibration when
the run persisted it (``per_file[i].degeneracy_cal``); runs made before it was
persisted fall back to `fit_windows`: the (mu, sigma) that reproduces the
stored ``degeneracy_density`` values, found by a grid around the recomputed
calibration. An uncalibrated file takes (median(ppm_error - ppm_error_cal), the
run sigma); the run window is the median of the per-file windows.

Pure (frames in, objects out) except `source_inputs_from_run_dir`, which reads
a run directory into the in-memory inputs.
"""
from __future__ import annotations

import dataclasses
import glob
import json
import math
import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd

from peaky.assignment import degeneracy as DG
from peaky.assignment import tiers as T
from peaky.assignment.levels import lines as LN
from peaky.assignment.levels import space as SP
from peaky.batch import iso_checks as IC
from peaky.chem import chemistry as C
from peaky.chem import isotopes as ISO
from peaky.chem import contexts as X
from peaky.chem import profiles as PR
from peaky.chem import reagents as RG

K_SIGMA = 3.0                 # the calibrated half-window, in sigma
NMIN_FILES = 3                # files where an isotope test is possible (a batch)
# C47: a contaminant family opens the run's space (its competitors for EVERY
# pair) only from >= this many files (1 on a one-file source) -- the other
# 2-file minima of the scale (ROUTE_COFILES, LADDER_MIN_FILES). On the uronium
# run the scale was validated on, ONE Candidate row in ONE file opened
# `fluorinated` for all 1196 pairs (131 moved behind F competitors).
FAMILY_MIN_FILES = 2
NMIN_FILES_ARM = 1            # a single-file source ("adapted" minima)
ORBI_MIN_PPM, ORBI_K = 1.0, 4.0   # the isotope position window without a fit: 1 ppm (ISO.position_window_ppm's floor)
DEFAULT_FWHM_DA = 0.02        # the line merge width without a width model
NH4_ALIAS = {"[M+H]+": "[M+NH4]+"}   # the same-ion alias [X+NH4]+ = [X+NH3+H]+ (decompositions)
TWIN_Q = 0.10                 # a file's 15N twin ratio: the 10th percentile of h(15N twin)/h(14N cluster) ...
TWIN_MIN_PAIRS = 3            # ... over >= 3 neutrals ...
TWIN_BRIGHT_X = 10.0          # ... whose 14N cluster is >= 10 x the file's height gate
SCAN_MARGIN = 0.5             # a line within 0.5 Da of the file's first / last peak (or beyond) is untestable
EDGE_PERCENTILE = 1.0         # a file's "edge" height: the 1st percentile of its peak heights (no gate known)
C_NMIN_TS = IC.C_NMIN         # rule C's own minimum (time-series spectra)
# the window fit (D10 fallback): enumerate each noted row once in a wide window, then grid (mu, sigma) until the
# stored densities are reproduced. Each tuple keeps the reference's literal operands (np.arange's float grid).
FIT_PARITY_MIN = 0.99         # the recomputed calibration is kept when it reproduces >= 99 % of the stored densities
FIT_WIDE_K, FIT_WIDE_PAD = 2.5, 0.6          # wide window: mu0 -/+ (2.5 K_SIGMA sigma0 + 0.6) ppm
FIT_MU_COARSE = (0.25, 0.2501, 0.025)        # np.arange(mu0 - 0.25, mu0 + 0.2501, 0.025)
FIT_SIGMA_COARSE = (0.15, 0.85, 1.4, 0.01)   # np.arange(max(0.15, 0.85 s0), 1.4 s0, 0.01)
FIT_MU_FINE = (0.0125, 0.01251, 0.0025)      # np.arange(mu1 - 0.0125, mu1 + 0.01251, 0.0025)
FIT_SIGMA_FINE = (0.01, 0.01001, 0.002)      # np.arange(s1 - 0.01, s1 + 0.01001, 0.002)
DENSITY_EPS = 1e-12           # the stored density's inclusive window edges
# the batch time-series columns the scale reads (pairs' TS groups; the amine gate's binned matrix)
TS_COLUMNS = ("sample_item_id", "datetime_utc", "peak_id", "mz", "height", "role", "neutral_formula", "adduct",
              "ion_formula", "iso_label")


def _ev():
    from peaky.assignment import evidence as EV
    return EV


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class SourceInputs:
    """A source's in-memory inputs: the batch summary (reagent, context,
    reflists_active, resolution, per_file stats), the FULL per-file ledgers
    ({sample_id: frame}, sorted by sample id), the merged ledger, the batch
    time series, and the iso_checks / label_twins tables (None when absent)."""
    summary: dict
    per_file: dict
    merged: pd.DataFrame | None = None
    ts: pd.DataFrame | None = None
    iso_checks: pd.DataFrame | None = None
    label_twins: pd.DataFrame | None = None


def source_inputs_from_run_dir(path: str, *, float_precision: str | None = None,
                               ts_columns=TS_COLUMNS) -> SourceInputs:
    """Read a batch run directory: batch_summary.json, per_file/*_ledger.csv
    (sorted), merged_ledger.csv, per_file/_batch_ts.parquet (``ts_columns``
    that exist; None = all), tables/iso_checks.csv, tables/label_twins.csv.
    Every CSV is read with ``low_memory=False`` and ``float_precision``: by
    default pandas' own parser, the one the batch's level stage reads its
    per-file ledgers with, so a post-hoc re-level sees the same floats
    ('round_trip' reproduces the written in-memory floats exactly; the default
    parser can differ in the last ulp, which moves a fitted window by ~1e-16 ppm)."""
    def _csv(p):
        return pd.read_csv(p, low_memory=False, float_precision=float_precision) if os.path.isfile(p) else None

    with open(os.path.join(path, "batch_summary.json")) as fh:
        summary = json.load(fh)
    per_file = {}
    for fp in sorted(glob.glob(os.path.join(path, "per_file", "*_ledger.csv"))):
        per_file[re.sub(r"_ledger\.csv$", "", os.path.basename(fp))] = _csv(fp)
    ts = None
    tp = os.path.join(path, "per_file", "_batch_ts.parquet")
    if os.path.isfile(tp):
        cols = None
        if ts_columns is not None:
            import pyarrow.parquet as pq
            have = set(pq.ParquetFile(tp).schema.names)
            cols = [c for c in ts_columns if c in have]
        ts = pd.read_parquet(tp, columns=cols)
    return SourceInputs(summary=summary, per_file=per_file,
                        merged=_csv(os.path.join(path, "merged_ledger.csv")), ts=ts,
                        iso_checks=_csv(os.path.join(path, "tables", "iso_checks.csv")),
                        label_twins=_csv(os.path.join(path, "tables", "label_twins.csv")))


def pair_key(frame: pd.DataFrame, n="neutral_formula", a="adduct") -> list[tuple[str, str]]:
    return list(zip(frame[n].fillna("").astype(str), frame[a].fillna("").astype(str)))


def label_alien(label_twins: pd.DataFrame | None) -> frozenset:
    """The (neutral, adduct) pairs label_twins.csv marks ``alien`` (rule K 14N lines)."""
    if label_twins is None or "alien" not in label_twins.columns or not len(label_twins):
        return frozenset()
    return frozenset(pair_key(label_twins[label_twins["alien"].map(_ev().truthy)]))


# ---------------------------------------------------------------------------
# per-file arrays
# ---------------------------------------------------------------------------
class FileArr:
    """One file's peaks (every row with a finite m/z and height > 0, whatever
    its role), sorted by m/z, with its height gate, scan edges and the set of
    refuted readings ('neutral|adduct') whose M0 does not own its peak."""

    def __init__(self, led: pd.DataFrame, gate: float):
        d = led.copy()
        d["mz"] = pd.to_numeric(d["mz"], errors="coerce")
        d["height"] = pd.to_numeric(d["height"], errors="coerce")
        d["area"] = pd.to_numeric(d["area"], errors="coerce") if "area" in d.columns else d["height"]
        d = d[np.isfinite(d["mz"]) & (d["height"] > 0)].sort_values("mz", kind="mergesort")
        self.mz = d["mz"].to_numpy(float)
        self.h = d["height"].to_numpy(float)
        self.a = d["area"].fillna(0).to_numpy(float)
        self.role = d["role"].astype(str).to_numpy()
        #: a peak a per-file stage marked 'artifact' (an Orbitrap side lobe, cleanup's ringing): not a
        #: line of the profile. It is left out of every line pick; a position only artifacts hold is
        #: untestable ('shadowed'), never empty -- the rule of the element-evidence predicate and REQ
        self.art = self.role == "artifact"
        self.parent = d["parent_peak_id"].astype(object).where(d["parent_peak_id"].notna(), "").astype(str).to_numpy()
        self.pid = d["peak_id"].astype(str).to_numpy()
        self.pk = (d["neutral_formula"].fillna("").astype(str) + "|" + d["adduct"].fillna("").astype(str)).to_numpy()
        self.label = d["iso_label"].astype(object).where(d["iso_label"].notna(), "").astype(str).to_numpy() \
            if "iso_label" in d.columns else np.array([""] * len(d))
        self.gate = float(gate)
        self.scan_start = float(self.mz.min()) if len(self.mz) else float("nan")
        self.scan_end = float(self.mz.max()) if len(self.mz) else float("nan")
        self.refuted: set = set()
        self.edge = float(np.percentile(self.h, EDGE_PERCENTILE)) if len(self.h) else float("nan")

    def window(self, t: float, tol_da: float):
        lo = np.searchsorted(self.mz, t - tol_da, "left")
        hi = np.searchsorted(self.mz, t + tol_da, "right")
        return lo, hi

    def probe(self, span, tol_da: float, anchor_pid: str, shadow_da: float = 0.0, h_exp: float = 0.0, own=()):
        """('absent'|'shadowed'|'occupied'|'free', height, index) of the tallest
        peak within tol of the line's span. Nothing there: a line displaced
        toward a >= 3x taller neighbour within 25 ppm still counts (the
        isotopes module's N1 reach); else 'shadowed' when a peak >= SHADOW_FRAC
        x the expected height sits within ``shadow_da``. A refuted reading's M0
        and the reading's own twin (``own``) are free; any other M0 / reagent
        row or another parent's iso child is 'occupied'. An artifact row (a
        side lobe, ringing) is no line: the pick is the tallest OTHER peak
        within tol, and a span only artifacts hold is 'shadowed' (untestable)
        -- a lobe beside a line neither occupies it nor empties it."""
        lo_t, hi_t = span
        lo = np.searchsorted(self.mz, lo_t - tol_da, "left")
        hi = np.searchsorted(self.mz, hi_t + tol_da, "right")
        if hi > lo and self.art[lo:hi].all():
            return "shadowed", 0.0, -1
        if hi <= lo:
            reach = ISO.NEIGHBOUR_REACH_PPM * 1e-6 * hi_t
            l2 = np.searchsorted(self.mz, lo_t - reach, "left")
            h2 = np.searchsorted(self.mz, hi_t + reach, "right")
            best_c = -1
            for c in range(l2, h2):
                if self.pid[c] == anchor_pid or self.art[c]:
                    continue
                cm = self.mz[c]
                r = cm - hi_t if cm > hi_t else cm - lo_t
                ref = hi_t if cm > hi_t else lo_t
                for nb in range(l2, h2):
                    if nb == c or self.h[nb] < ISO.NEIGHBOUR_RATIO * self.h[c]:
                        continue
                    dj = self.mz[nb] - ref
                    if np.sign(dj) == np.sign(r) and abs(r) <= ISO.NEIGHBOUR_FRACTION * abs(dj):
                        if best_c < 0 or abs(r) < abs(self.mz[best_c] - ref):
                            best_c = c
                        break
            if best_c >= 0:
                lo, hi = best_c, best_c + 1
        if hi <= lo:
            if shadow_da > tol_da:
                lo2 = np.searchsorted(self.mz, lo_t - shadow_da, "left")
                hi2 = np.searchsorted(self.mz, hi_t + shadow_da, "right")
                if hi2 > lo2 and float(self.h[lo2:hi2].max()) >= LN.SHADOW_FRAC * h_exp:
                    return "shadowed", 0.0, -1
            return "absent", 0.0, -1
        best = lo + int(np.argmax(np.where(self.art[lo:hi], -np.inf, self.h[lo:hi])))
        r = self.role[best]
        if r == "M0" and (self.pk[best] in self.refuted or self.pk[best] in own):
            return "free", float(self.h[best]), best
        if r in ("M0", "reagent") or (r == "iso_child" and self.parent[best] != anchor_pid):
            return "occupied", float(self.h[best]), best
        return "free", float(self.h[best]), best

    def any_peak(self, t: float, tol_da: float) -> bool:
        lo, hi = self.window(t, tol_da)
        return hi > lo

    def tallest(self, t: float, tol_da: float) -> float:
        """The tallest peak (any role) within tol of t, 0 when none."""
        lo, hi = self.window(t, tol_da)
        return float(self.h[lo:hi].max()) if hi > lo else 0.0

    def shoulder(self, j: int, da: float, factor: float) -> bool:
        """Peak j sits within ``da`` of a peak more than ``factor`` x taller."""
        lo, hi = self.window(self.mz[j], da)
        return bool(hi > lo and float(self.h[lo:hi].max()) > factor * self.h[j])

    def in_scan(self, lo_t: float, hi_t: float) -> bool:
        """The line lies inside the scan (>= SCAN_MARGIN inside the first and last peaks)."""
        return bool(lo_t >= self.scan_start + SCAN_MARGIN and hi_t <= self.scan_end - SCAN_MARGIN)


# ---------------------------------------------------------------------------
# pairs and their per-file observations
# ---------------------------------------------------------------------------
def parse_alts(v) -> list[dict]:
    """A per-file row's ``alternatives`` JSON (formula, adduct, ppm, scores); [] otherwise."""
    if not isinstance(v, str) or not v.strip().startswith("["):
        return []
    try:
        x = json.loads(v)
    except Exception:  # noqa: BLE001
        return []
    return [e for e in x if isinstance(e, dict) and e.get("formula") and e.get("adduct")]


def pairs_from_files(files_led: dict, facts: pd.DataFrame) -> dict:
    """One record per pair of ``facts`` (neutral_formula, adduct, mz, ion,
    ppm): ``obs`` = per file, the pair's BRIGHTEST M0 row (sid, mz, h, area,
    pid, ppm); ``alts`` = the ``alternatives`` of all its M0 rows, each as
    (alt, the row's tied flag, sid); ``own_pk`` = the 14N twin of a
    [M+^NO3]- reading (its own line, probed as free)."""
    truthy = _ev().truthy
    obs = defaultdict(list)
    alts = defaultdict(list)
    for sid, led in files_led.items():
        m0 = led[led["role"] == "M0"].copy()
        m0["hh"] = pd.to_numeric(m0["height"], errors="coerce")
        m0["aa"] = pd.to_numeric(m0["area"], errors="coerce") if "area" in m0.columns else m0["hh"]
        m0 = m0.sort_values("hh", ascending=False, kind="mergesort")
        seen = set()
        for r in m0.itertuples(index=False):
            k = (str(r.neutral_formula) if isinstance(r.neutral_formula, str) else "",
                 str(r.adduct) if isinstance(r.adduct, str) else "")
            if not k[0] or not k[1]:
                continue
            tied = truthy(getattr(r, "tied", False))
            for e in parse_alts(getattr(r, "alternatives", None)):
                alts[k].append((e, tied, sid))
            if k in seen:
                continue
            seen.add(k)
            obs[k].append(dict(sid=sid, mz=float(r.mz), h=float(r.hh) if np.isfinite(r.hh) else 0.0,
                               area=float(r.aa) if np.isfinite(r.aa) else 0.0, pid=str(r.peak_id),
                               ppm=float(pd.to_numeric(getattr(r, "ppm_error", np.nan), errors="coerce"))))
    out = {}
    for r in facts.itertuples(index=False):
        k = (r.neutral_formula, r.adduct)
        own = {f"{k[0]}|[M+NO3]-"} if k[1] == "[M+^NO3]-" else set()
        out[k] = dict(key=k, pairkey=f"{k[0]}|{k[1]}", mz=float(r.mz), ion=r.ion, ppm=getattr(r, "ppm", np.nan),
                      obs=obs.get(k, []), alts=alts.get(k, []), own_pk=own)
    return out


# ---------------------------------------------------------------------------
# the calibrated windows
# ---------------------------------------------------------------------------
def file_cal(led: pd.DataFrame):
    """The file's mass calibration (mu, sigma) ppm recomputed on its ledger
    (``tiers._calibrate``), None when uncalibrated."""
    m0 = led[led["role"] == "M0"]
    kids = led.loc[led["role"] == "iso_child", "parent_peak_id"].value_counts()
    cal = T._calibrate(m0, kids)
    return None if cal is None else (float(cal[0]), float(cal[1]))


def mu_stamp(led: pd.DataFrame) -> float:
    """An uncalibrated file's window centre: median(ppm_error - ppm_error_cal) over its M0 rows (NaN when none,
    and when the ledger has no ``ppm_error_cal`` at all: tiers.stamp_calibrated_ppm writes it only when it could
    centre the file -- a decoy arm with nothing Assigned has none)."""
    if "ppm_error" not in led.columns or "ppm_error_cal" not in led.columns:
        return np.nan
    m0 = led[led["role"] == "M0"]
    pce = pd.to_numeric(m0["ppm_error"], errors="coerce") - pd.to_numeric(m0["ppm_error_cal"], errors="coerce")
    return float(pce.median()) if pce.notna().any() else np.nan


def run_space_args(summary: dict) -> tuple:
    """(reagent, context, reflists_active) of a batch summary. The context is
    the run's ContextProfile when the summary records run-level switches
    (``context_flags``: the NOx-skeleton reading), else its name."""
    ctx = summary["context"]
    flags = summary.get("context_flags")
    if flags:
        ctx = X.as_profile(ctx, flags)
    return summary["reagent"], ctx, summary.get("reflists_active")


def file_families(summary: dict, led: pd.DataFrame) -> tuple[str, ...]:
    """The contaminant families a file opened, without GKA evidence
    (``degeneracy.opened_families`` on its final ledger)."""
    ctx = X.get_context(summary["context"])
    reagent = RG.reagent_for_adducts(list(PR.resolve(summary["reagent"]).adducts))
    return DG.opened_families(ctx, reagent, None, led)


def density_in_window(found: dict, assigned_ion, outside, mu: float, sigma: float, k: float = K_SIGMA):
    """``measure_degeneracy``'s density for a (mu, sigma) window, from a wide
    enumeration ``found``: (density or NaN, competitors [(neutral, adduct, ppm)])."""
    lo, hi = mu - k * sigma, mu + k * sigma
    inwin = {ion: v for ion, v in found.items() if lo - DENSITY_EPS <= v[2] <= hi + DENSITY_EPS}
    comp = [v for ion, v in inwin.items() if ion != assigned_ion]
    density = float(len(inwin))
    if assigned_ion is None or assigned_ion not in inwin:
        if outside is not None:
            density = float(len(comp) + 1)
            if density < DG.DEGEN_LOWER_DECIDES:
                return float("nan"), comp
    return density, comp


class _ParityScorer:
    """Vectorised ``density_in_window`` over a file's noted rows: (exact, within
    +-1) counts of the stored densities a (mu, sigma) window reproduces."""

    def __init__(self, rows: list):
        self.n = len(rows)
        ppm, rid, own = [], [], []
        for i, (found, ai, _srs, _st) in enumerate(rows):
            for ion, v in found.items():
                ppm.append(v[2])
                rid.append(i)
                own.append(ai is not None and ion == ai)
        self.ppm = np.array(ppm, float)
        self.rid = np.array(rid, np.int64)
        self.own = np.array(own, bool)
        self.outside = np.array([r[2] is not None for r in rows], bool)
        self.stored = np.array([r[3] for r in rows], float)

    def __call__(self, mu: float, sigma: float, k: float = K_SIGMA) -> tuple[int, int]:
        lo, hi = mu - k * sigma, mu + k * sigma
        inw = (self.ppm >= lo - DENSITY_EPS) & (self.ppm <= hi + DENSITY_EPS)
        cnt = np.bincount(self.rid[inw], minlength=self.n).astype(float)
        own_in = np.bincount(self.rid[inw & self.own], minlength=self.n) > 0
        d = cnt.copy()
        low = ~own_in & self.outside
        d[low] = cnt[low] + 1.0
        d[low & (d < DG.DEGEN_LOWER_DECIDES)] = np.nan
        st = self.stored
        eq = (np.isnan(d) & np.isnan(st)) | (d == st)
        w1 = eq | (np.isfinite(d) & np.isfinite(st) & (np.abs(d - st) <= 1))
        return int(eq.sum()), int(w1.sum())


def fit_window(led: pd.DataFrame, space: SP.Space, cal=None) -> dict:
    """D10 fallback for one file: the (mu, sigma) that reproduces the stored
    ``degeneracy_density`` of its noted M0 rows. ``cal`` defaults to the
    recomputed calibration; kept when it reproduces >= FIT_PARITY_MIN of them,
    else a coarse then a fine grid around it (first best kept). Returns
    {cal, n_m0, n_noted, mu_stamp (uncalibrated) | parity_cal, parity_fit, fit}."""
    cal = file_cal(led) if cal is None else cal
    m0 = led[led["role"] == "M0"]
    rec = dict(cal=cal, n_m0=len(m0))
    if cal is None:
        rec["mu_stamp"] = mu_stamp(led)
        rec["n_noted"] = int(m0["degeneracy_note"].notna().sum()) if "degeneracy_note" in m0.columns else 0
        return rec
    mu0, s0 = cal
    lo, hi = mu0 - FIT_WIDE_K * K_SIGMA * s0 - FIT_WIDE_PAD, mu0 + FIT_WIDE_K * K_SIGMA * s0 + FIT_WIDE_PAD
    rows = []
    if "degeneracy_note" in m0.columns:
        for r in m0.itertuples(index=False):
            notev = r.degeneracy_note if isinstance(r.degeneracy_note, str) else None
            if notev is None:
                continue
            nf, ad = str(r.neutral_formula), str(r.adduct)
            chans = space.adducts + ([ad] if ad in C.ADDUCT_SHIFTS and ad not in space.adducts else [])
            found = space.enumerate(float(r.mz), lo, hi, chans)
            ai = space.canon(nf, ad) if ad in C.ADDUCT_SHIFTS else None
            srs = space.space_reason(nf) if ai is not None else "no ion composition"
            rows.append((found, ai, srs, float(pd.to_numeric(r.degeneracy_density, errors="coerce"))))
    score = _ParityScorer(rows)
    e0, w0 = score(mu0, s0)
    best = (e0, w0, mu0, s0)
    if rows and e0 < FIT_PARITY_MIN * len(rows):
        for mu in np.arange(mu0 - FIT_MU_COARSE[0], mu0 + FIT_MU_COARSE[1], FIT_MU_COARSE[2]):
            for sg in np.arange(max(FIT_SIGMA_COARSE[0], s0 * FIT_SIGMA_COARSE[1]), s0 * FIT_SIGMA_COARSE[2],
                                FIT_SIGMA_COARSE[3]):
                e, w = score(mu, sg)
                if e > best[0]:
                    best = (e, w, mu, sg)
        m1, s1 = best[2], best[3]
        for mu in np.arange(m1 - FIT_MU_FINE[0], m1 + FIT_MU_FINE[1], FIT_MU_FINE[2]):
            for sg in np.arange(s1 - FIT_SIGMA_FINE[0], s1 + FIT_SIGMA_FINE[1], FIT_SIGMA_FINE[2]):
                e, w = score(mu, sg)
                if e > best[0]:
                    best = (e, w, mu, sg)
    rec.update(n_noted=len(rows), parity_cal=(e0, w0), parity_fit=(best[0], best[1]),
               fit=(float(best[2]), float(best[3])))
    return rec


def fit_windows(summary: dict, per_file: dict, *, catalog=None) -> dict:
    """D10 fallback for every file: {sid: {fams, src: 'fit'} + fit_window's record}.
    Each file is fitted in ITS OWN space (its own opened families)."""
    reagent, context, active = run_space_args(summary)
    spaces: dict = {}
    out = {}
    for sid, led in per_file.items():
        fams = file_families(summary, led)
        sp = spaces.get(fams)
        if sp is None:
            sp = spaces[fams] = SP.Space(reagent, context, active, fams, catalog=catalog)
        out[sid] = dict(fams=fams, src="fit", **fit_window(led, sp))
    return out


def family_union(win: dict) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """(the run's families, the families dropped): a family counts when at
    least FAMILY_MIN_FILES of the files' window records (`fams`) opened it
    (every family on a one-file source), in first-seen order (C47)."""
    need = min(FAMILY_MIN_FILES, max(1, len(win)))
    counts: dict = {}
    for w in win.values():
        for f in dict.fromkeys(w.get("fams", ()) or ()):
            counts[f] = counts.get(f, 0) + 1
    kept = tuple(f for f, c in counts.items() if c >= need)
    dropped = tuple(f for f, c in counts.items() if c < need)
    return kept, dropped


def run_windows(summary: dict, per_file: dict, *, catalog=None) -> dict:
    """The window record of every file: PRIMARY the persisted degeneracy-stage
    calibration (``per_file[i].degeneracy_cal`` = {mu, sigma} | null), else
    the D10 fallback (`fit_windows`) for the files that lack it."""
    stats = {p.get("sample_id"): p for p in (summary.get("per_file") or [])}
    out, missing = {}, {}
    for sid, led in per_file.items():
        st = stats.get(sid) or {}
        if "degeneracy_cal" in st:
            cal = st["degeneracy_cal"]
            rec = dict(fams=file_families(summary, led), src="degeneracy_cal", n_m0=int((led["role"] == "M0").sum()))
            if cal:
                rec.update(cal=(float(cal["mu"]), float(cal["sigma"])), fit=(float(cal["mu"]), float(cal["sigma"])))
            else:
                rec.update(cal=None, mu_stamp=mu_stamp(led))
            out[sid] = rec
        else:
            missing[sid] = led
    fitted = fit_windows(summary, missing, catalog=catalog) if missing else {}
    return {sid: out[sid] if sid in out else fitted[sid] for sid in per_file}


def window_table(win: dict) -> tuple[dict, tuple[float, float]]:
    """({sid: (mu, sigma)}, run window): the run window is the median of the
    calibrated files' (mu, sigma); an uncalibrated file takes (its mu_stamp,
    else the run mu; the run sigma)."""
    fitted = [w["fit"] for w in win.values() if w.get("fit")]
    run_window = (float(np.median([f[0] for f in fitted])), float(np.median([f[1] for f in fitted]))) \
        if fitted else (float("nan"), float("nan"))
    windows = {}
    for sid, w in win.items():
        if w.get("fit"):
            windows[sid] = w["fit"]
        else:
            mu = w.get("mu_stamp", np.nan)
            windows[sid] = (mu if np.isfinite(mu) else run_window[0], run_window[1])
    return windows, run_window


# ---------------------------------------------------------------------------
# the run context
# ---------------------------------------------------------------------------
class RunContext:
    """Everything one source needs to level its pairs (Orbitrap class)."""

    def __init__(self, name, files: dict, gates: dict, windows: dict, run_window, space: SP.Space, klass: str, rp,
                 sig_fit, eff: dict, nmin: int, channels: list, decomp_adducts: list, labelled: bool, *,
                 purity: float = ISO.LABEL_PURITY_15N):
        self.name = name
        self.files = files            # sid -> FileArr
        self.gates = gates            # sid -> height gate (cps)
        self.windows = windows        # sid -> (mu, sigma) ppm
        self.run_window = run_window  # (mu, sigma) ppm
        self.space = space
        self.klass = klass
        self.rp = rp
        self.sig_fit = sig_fit        # isotopes.PositionSigma | None
        self.eff = eff                # element -> line efficiency
        self.nmin = nmin
        self.channels = channels      # the enumeration channels (the profile's adducts)
        self.engine_channels = list(channels)
        self.run_channels: list = []
        self.decomp_adducts = decomp_adducts
        self.labelled = labelled
        self.purity = float(purity)   # the labelled reagent's isotopic purity (the 14N impurity line)
        self.c13: dict = {}           # (n, a) -> dict(c, se, n, src)
        self.cse = None               # single-file se model
        self.twin_q: dict = {}        # sid -> lower-bound h(15N twin) / h(14N cluster) (labelled run)
        self.m0_mz = np.array([])     # every committed, unrefuted M0 m/z of the source (all files), sorted
        self.m0_pk = np.array([], dtype=object)
        self.isolines: dict = {}      # sid -> the predicted isotope lines of its committed readings
        self.alien: frozenset = frozenset()
        self.window_records: dict = {}
        self.families: tuple = ()
        self.families_dropped: tuple = ()   # opened by fewer than FAMILY_MIN_FILES files (C47)
        self.n_13c_children = 0
        self.cl37_flips = 0
        self.cl37_blend = 0

    def fwhm(self, mz) -> float:
        return float(self.rp.fwhm(mz)) if self.rp is not None else DEFAULT_FWHM_DA

    def tol_da(self, mz, h_exp) -> float:
        """The isotope position window (Da) at ``mz`` for a line of predicted
        height ``h_exp``: max(1 ppm, 4 sigma(h)) from the position-sigma fit, 1 ppm without one."""
        if self.sig_fit is not None and h_exp and h_exp > 0:
            w = ISO.position_window_ppm(h_exp, self.sig_fit)
        else:
            w = ORBI_MIN_PPM
        return w * mz * 1e-6

    def decompositions(self, counts: dict, n, a, adducts=None) -> list[dict]:
        return SP.decompositions(self.space, counts, n, a, self.decomp_adducts if adducts is None else adducts)

    def lines(self, counts: dict, fwhm: float, anchor_shift: float) -> list[dict]:
        return LN.cand_lines(counts, fwhm, anchor_shift, self.purity)


def position_sigma(per_file: dict):
    """(isotopes.fit_position_sigma over every 13C iso child's residual (ppm vs
    its parent + D13C) by height, the number of children)."""
    res, hs = [], []
    for _sid, led in per_file.items():
        m0 = led[led["role"] == "M0"].set_index("peak_id")
        iso = led[(led["role"] == "iso_child") & (led["iso_label"].astype(str).str.strip() == "13C")]
        for r in iso.itertuples(index=False):
            p = r.parent_peak_id
            if p not in m0.index:
                continue
            pm = m0.loc[p, "mz"]
            pm = float(pm.iloc[0]) if isinstance(pm, pd.Series) else float(pm)
            res.append((float(r.mz) - pm - IC.D13C) / pm * 1e6)
            hs.append(float(r.height))
    return ISO.fit_position_sigma(res, hs), len(res)


def build_run_context(summary: dict, per_file: dict, win: dict, *, nmin: int = NMIN_FILES, name: str = "",
                      catalog=None) -> RunContext:
    """The context of a source before its line efficiency / carbon counts /
    indexes (`prepare_context` adds them): width model + class, FileArr per file
    with its gate (``height_gate_cps``, else ``noise_edge_cps``, else the file's
    1st-percentile height), the windows (``win`` = `run_windows`), the space
    over the families >= FAMILY_MIN_FILES files opened (`family_union`), the position-sigma fit, the
    channels (profile adducts), the decomposition adducts (+ every committed
    non-ion-only adduct with a composition, + [M+NH4]+ beside [M+H]+), and the
    15N twin ratios. Raises ValueError on a source the scale does not assess."""
    from peaky.chem.resolution import Resolution
    res = summary.get("resolution")
    klass = _ev().instrument(res)[0] or "tof"
    if klass != "orbitrap":
        raise ValueError(f"not assessed on this instrument class ({klass})")
    rp = Resolution.from_dict(res) if (res or {}).get("coef") else None
    gates = {p["sample_id"]: float(p.get("height_gate_cps") or p.get("noise_edge_cps") or 0.0)
             for p in summary["per_file"]}
    files = {sid: FileArr(led, gates.get(sid, np.nan)) for sid, led in per_file.items()}
    for sid, fa in files.items():
        if not np.isfinite(fa.gate) or fa.gate <= 0:
            fa.gate = fa.edge
            gates[sid] = fa.edge
    windows, run_window = window_table(win)
    fams, fams_dropped = family_union(win)
    space = SP.Space(*run_space_args(summary), fams, catalog=catalog)
    sig_fit, n13 = position_sigma(per_file)
    labelled = any("^" in a for a in space.adducts)
    committed_adducts = []
    for led in per_file.values():
        committed_adducts += [x for x in led.loc[led["role"] == "M0", "adduct"].dropna().astype(str).unique()]
    run_channels = list(dict.fromkeys(list(space.adducts) + [x for x in committed_adducts if x in C.ADDUCT_SHIFTS]))
    channels = list(space.adducts)
    decomp = list(dict.fromkeys(channels + [x for x in committed_adducts
                                            if x not in SP.ION_ONLY and SP.adduct_delta(x) is not None]))
    decomp += [v for k_, v in NH4_ALIAS.items() if k_ in decomp and v not in decomp]
    purity = space.prof.purity if getattr(space.prof, "purity", None) is not None else ISO.LABEL_PURITY_15N
    ctx = RunContext(name, files, gates, windows, run_window, space, klass, rp, sig_fit, {}, nmin, channels, decomp,
                     labelled, purity=purity)
    ctx.run_channels = run_channels
    ctx.window_records = win
    ctx.families = fams
    ctx.families_dropped = fams_dropped
    ctx.n_13c_children = n13
    ctx.twin_q = twin_ratios(ctx, per_file)
    return ctx


def twin_ratios(ctx: RunContext, files_led: dict) -> dict:
    """Labelled run: per file, the TWIN_Q quantile of h([M+^NO3]-)/h([M+NO3]-)
    over the neutrals committed on both with the 14N cluster >= TWIN_BRIGHT_X
    x the gate (>= TWIN_MIN_PAIRS such neutrals)."""
    out = {}
    if not ctx.labelled:
        return out
    for sid, led in files_led.items():
        m = led[led["role"] == "M0"]
        h = pd.to_numeric(m["height"], errors="coerce")
        nf = m["neutral_formula"].astype(str)
        h14 = h[m["adduct"] == "[M+NO3]-"].groupby(nf[m["adduct"] == "[M+NO3]-"]).max()
        h15 = h[m["adduct"] == "[M+^NO3]-"].groupby(nf[m["adduct"] == "[M+^NO3]-"]).max()
        j = pd.concat([h14, h15], axis=1, keys=["h14", "h15"]).dropna()
        j = j[j["h14"] >= TWIN_BRIGHT_X * ctx.gates.get(sid, np.inf)]
        if len(j) >= TWIN_MIN_PAIRS:
            out[sid] = float(np.quantile(j["h15"] / j["h14"], TWIN_Q))
    return out


def _anchor_shift(mz: float, n: str, a: str) -> float:
    """m/z minus the reading's mono ion m/z when the peak is not the mono line (>= 0.3 Da), else 0."""
    try:
        ash = mz - C.ion_mz(n, a)
    except Exception:  # noqa: BLE001
        ash = 0.0
    return 0.0 if abs(ash) < 0.3 else ash


def efficiency(ctx: RunContext, facts: pd.DataFrame, files_led: dict, nmin: int) -> dict:
    """Per element: the median seen/theory height of the committed readings'
    single-element lines that are present (free) in >= 50 % of >= ``nmin``
    detectable files, clipped to [0.25, 1], over >= 3 pairs: {el: (eff, n pairs)}.
    Evaluated with no efficiency applied."""
    pairs = pairs_from_files(files_led, facts)
    eff0 = ctx.eff
    ctx.eff = {}
    per = defaultdict(list)
    for k, pr in pairs.items():
        if not pr["obs"]:
            continue
        n, a = k
        cnt = SP.ion_counts_of(n, a, pr["ion"])
        if not cnt:
            continue
        ash = _anchor_shift(pr["mz"], n, a)
        c = dict(name="", counts=cnt, lines=ctx.lines(cnt, ctx.fwhm(pr["mz"]), ash))
        for r in LN.eval_candidate(ctx, c, pr["obs"], pr["pairkey"]):
            L = r["L"]
            if len(L["elements"]) != 1 or L["mode"] == "reagent14N":
                continue
            det = [p for p in r["per"] if p[3] and p[0] not in ("occupied", "shadowed")]
            if len(det) < nmin:
                continue
            seen = [p for p in det if p[0] == "free"]
            if len(seen) < 0.5 * len(det):
                continue
            per[L["elements"][0]].append(float(np.median([p[1] / p[2] for p in seen])))
    ctx.eff = eff0
    return {el: (float(np.clip(np.median(v), 0.25, 1.0)), len(v)) for el, v in per.items() if len(v) >= 3}


def bias_area(iso_tab: pd.DataFrame | None) -> float:
    """Rule C's area bias from tables/iso_checks.csv (0 when absent)."""
    if iso_tab is None or "bias_area" not in iso_tab.columns:
        return 0.0
    b = pd.to_numeric(iso_tab.loc[iso_tab["check"] == "C", "bias_area"], errors="coerce").dropna()
    return float(b.iloc[0]) if len(b) else 0.0


def carbon_counts(ctx: RunContext, iso_tab: pd.DataFrame | None, facts: pd.DataFrame, files_led: dict, nmin: int,
                  prof_adducts, bias: float):
    """Test (c)'s observed carbon counts {(n, a): dict(c, se, n, src)}: rule
    C's pooled time-series reading (tables/iso_checks.csv) where it is testable
    (>= C_NMIN spectra, occupancy <= C_OCC_MAX; scan_edge / exempt_14N kept as
    NaN), else the same reading over the per-file ledgers (>= ``nmin`` used
    files, occupancy <= C_OCC_MAX x used). Also returns the per-file deviations
    from rule C that `se_model` reads."""
    truthy = _ev().truthy
    out = {}
    src_ts = {}
    if iso_tab is not None and len(iso_tab):
        c = iso_tab[iso_tab["check"] == "C"]
        for r in c.itertuples(index=False):
            k = (r.neutral_formula, r.adduct)
            if r.verdict in ("scan_edge", "exempt_14N"):
                src_ts[k] = dict(c=np.nan, se=np.nan, n=int(r.n_used), src=f"rule C {r.verdict}")
                continue
            occ = float(r.occupied) if pd.notna(r.occupied) else 0.0
            if pd.notna(r.c_area) and int(r.n_used) >= C_NMIN_TS and not (occ > IC.C_OCC_MAX):
                src_ts[k] = dict(c=float(r.c_area), se=float(r.se_area) if pd.notna(r.se_area) else np.nan,
                                 n=int(r.n_used), src=f"rule C, {int(r.n_used)} spectra")
    pairs = pairs_from_files(files_led, facts)
    io_set = set(pair_key(facts[facts["ion_only"].map(truthy)])) if "ion_only" in facts.columns else set()
    prof = type("P", (), {"adducts": prof_adducts})()
    per_file_dev = []
    for k, pr in pairs.items():
        if k in src_ts and (np.isfinite(src_ts[k]["c"]) or "exempt" in src_ts[k]["src"]
                            or "scan_edge" in src_ts[k]["src"]):
            out[k] = src_ts[k]
        n, a = k
        cnt = SP.ion_counts_of(n, a, pr["ion"])
        nC = int(cnt.get("C", 0))
        if nC < 1 or not pr["obs"]:
            continue
        if a in SP.ION_ONLY or (k in io_set):
            continue          # rule C's scope: stamped, not ion-only pairs
        if IC._labelled_sibling(a, prof):
            out.setdefault(k, dict(c=np.nan, se=np.nan, n=0, src="exempt_14N"))
            continue
        mzm = pr["mz"]
        wf = ((mzm + 1.0) / mzm) ** 1.5
        oth = cnt.get("O", 0) * IC.R17O
        a0s, a1s, used, occ, cf, xs = [], [], 0, 0, [], []
        for o in pr["obs"]:
            fa = ctx.files[o["sid"]]
            if not (o["h"] * nC * IC.R13C >= IC.C_KDL * fa.edge):
                continue
            t = o["mz"] + IC.D13C
            lo, hi = fa.window(t, IC.C_TOL_PPM * t * 1e-6)
            a1 = 0.0
            if hi > lo:
                j = lo + int(np.argmin(np.abs(fa.mz[lo:hi] - t)))
                a1 = fa.a[j]
                r_ = fa.role[j]
                if r_ == "M0" or (r_ == "iso_child" and fa.parent[j] != o["pid"]):
                    occ += 1
            if o["area"] <= 0:
                continue
            used += 1
            a0s.append(o["area"])
            a1s.append(a1)
            cf.append(((a1 / o["area"]) / wf - oth) / IC.R13C / (1 + bias))
            xs.append(math.log10(o["h"] * nC * IC.R13C / max(fa.edge, 1e-9)))
        if used == 0:
            continue
        x0, x1 = np.array(a0s), np.array(a1s)
        rr = x1.sum() / x0.sum()
        res_ = x1 - rr * x0
        se = (np.sqrt(used / max(used - 1, 1) * (res_ ** 2).sum()) / x0.sum() / wf / IC.R13C) if used > 1 else np.nan
        cpool = ((rr / wf - oth) / IC.R13C) / (1 + bias)
        if k in src_ts and np.isfinite(src_ts[k]["c"]):
            for c_, x_ in zip(cf, xs):
                per_file_dev.append((x_, c_ / src_ts[k]["c"] - 1.0, src_ts[k]["c"], cpool))
        if k not in out and used >= nmin and occ <= IC.C_OCC_MAX * used:
            if used == 1 and ctx.cse is not None:
                se = ctx.cse(xs[0]) * cpool
            out[k] = dict(c=float(cpool), se=float(se) if np.isfinite(se) else np.nan, n=used,
                          src=f"per-file 13C, {used} files")
    return out, per_file_dev


SE_BINS = [-1, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0, 10]   # log10 S/N of the expected 13C line
SE_MIN_N = 5                                       # deviations per bin for a robust sd
SE_DEFAULT = 0.3                                   # relative se where no bin has one


def se_model(dev):
    """The single-file 13C-count relative se by the log10 S/N of the expected
    13C line (1.4826 MAD of the per-file deviations from rule C, per bin;
    empty bins take the nearest filled one): (f(x) -> se, table rows)."""
    if not dev:
        return None, []
    d = pd.DataFrame(dev, columns=["x", "rel", "c_ts", "c_pf"])
    bins = SE_BINS
    d["b"] = pd.cut(d["x"], bins)
    tab = d.groupby("b", observed=False)["rel"].agg(
        lambda s: 1.4826 * np.median(np.abs(s - np.median(s))) if len(s) >= SE_MIN_N else np.nan)
    edges = np.array(bins)
    vals = np.array(tab.to_numpy(float), dtype=float, copy=True)
    fin = np.where(np.isfinite(vals))[0]
    for i in range(len(vals)):
        if not np.isfinite(vals[i]) and len(fin):
            vals[i] = vals[fin[np.argmin(np.abs(fin - i))]]

    def f(x):
        i = int(np.clip(np.searchsorted(edges, x) - 1, 0, len(vals) - 1))
        return float(vals[i]) if np.isfinite(vals[i]) else SE_DEFAULT
    rows = [dict(bin=str(b), rel_sd=float(v), n=int((d["b"] == b).sum())) for b, v in zip(tab.index, vals)]
    return f, rows


ISOLINE_MIN_RATIO = 1e-4      # an indexed isotope line predicts >= 1e-4 x its reading's M0


def prepare_indexes(ctx: RunContext, files_led: dict) -> None:
    """The source's committed, unrefuted M0 positions (``m0_mz`` / ``m0_pk``)
    and, per file, every isotope line (ratio >= 1e-4, x efficiency) of every
    committed, unrefuted reading's brightest M0 (``isolines``)."""
    mzs, pks = [], []
    for sid, led in files_led.items():
        fa = ctx.files[sid]
        m0 = led[led["role"] == "M0"].copy()
        m0["hh"] = pd.to_numeric(m0["height"], errors="coerce")
        m0["mm"] = pd.to_numeric(m0["mz"], errors="coerce")
        m0 = m0[np.isfinite(m0["mm"]) & (m0["hh"] > 0) & m0["neutral_formula"].notna() & m0["adduct"].notna()]
        m0 = m0.sort_values("hh", ascending=False, kind="mergesort").drop_duplicates(["neutral_formula", "adduct"])
        rows = []
        pk_pos = {}
        for r in m0.itertuples(index=False):
            pk = f"{r.neutral_formula}|{r.adduct}"
            if pk in fa.refuted:
                continue
            mzs.append(float(r.mm))
            pks.append(pk)
            pk_pos[pk] = (float(r.mm), float(r.hh))
            cnt = SP.ion_counts_of(str(r.neutral_formula), str(r.adduct), getattr(r, "ion_formula", None))
            if not cnt:
                continue
            ash = _anchor_shift(float(r.mm), str(r.neutral_formula), str(r.adduct))
            for L in ctx.lines(cnt, ctx.fwhm(float(r.mm)), ash):
                if L["mode"] == "reagent14N" or L["ratio"] < ISOLINE_MIN_RATIO:
                    continue
                eff = min([ctx.eff.get(e, 1.0) for e in L["elements"]] or [1.0])
                rows.append((float(r.mm) + L["d"], min(L["span"][0], L["d"]), max(L["span"][1], L["d"]),
                             L["ratio"] * eff, pk, str(r.neutral_formula), L["label"]))
        rows.sort(key=lambda t: t[0])
        ctx.isolines[sid] = dict(c=np.array([t[0] for t in rows]), lo=np.array([t[1] for t in rows]),
                                 hi=np.array([t[2] for t in rows]), r=np.array([t[3] for t in rows]),
                                 pk=[t[4] for t in rows], neutral=[t[5] for t in rows], label=[t[6] for t in rows],
                                 pos=pk_pos)
    o = np.argsort(np.array(mzs, float), kind="mergesort")
    ctx.m0_mz = np.array(mzs, float)[o]
    ctx.m0_pk = np.array(pks, dtype=object)[o]


def refuted_readings(facts: pd.DataFrame) -> set:
    """'neutral|adduct' of the pairs an isotope check or the label twin vetoes (iso_veto | label_veto)."""
    truthy = _ev().truthy
    m = pd.Series(False, index=facts.index)
    for c in ("iso_veto", "label_veto"):
        if c in facts.columns:
            m = m | facts[c].map(truthy).astype(bool)
    return {f"{n}|{a}" for n, a in pair_key(facts[m])}


def prepare_context(inputs: SourceInputs, facts: pd.DataFrame, *, win: dict | None = None, nmin: int = NMIN_FILES,
                    name: str = "", catalog=None) -> RunContext:
    """A batch source's full context, in the reference's order: windows
    (``win``, default `run_windows`) -> `build_run_context` -> line efficiency
    -> carbon counts + se model -> refuted readings (``facts``' iso / label
    vetoes) -> `prepare_indexes`. ``facts`` = the pair table (neutral_formula,
    adduct, mz, ion, ppm, ion_only, iso_veto, label_veto)."""
    pf = inputs.per_file
    win = run_windows(inputs.summary, pf, catalog=catalog) if win is None else win
    ctx = build_run_context(inputs.summary, pf, win, nmin=nmin, name=name, catalog=catalog)
    eff = efficiency(ctx, facts, pf, nmin)
    ctx.eff = {k: v[0] for k, v in eff.items()}
    ctx.eff_pairs = {k: v[1] for k, v in eff.items()}
    c13, dev = carbon_counts(ctx, inputs.iso_checks, facts, pf, nmin, ctx.space.adducts, bias_area(inputs.iso_checks))
    ctx.cse, ctx.cse_rows = se_model(dev)
    ctx.c13 = c13
    ctx.alien = label_alien(inputs.label_twins)
    refuted = refuted_readings(facts)
    for fa in ctx.files.values():
        fa.refuted = refuted
    prepare_indexes(ctx, pf)
    return ctx
