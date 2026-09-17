"""Trace-first: assign a batch's persistent ions ONCE, from their centred
traces, instead of assigning a handful of files and arbitrating afterwards.

On a TOF the per-file mass of one ion scatters ~13 ppm while its trace centre
is good to ~2 ppm, so a per-file assignment is a lottery and merging lotteries
is an argument. This module builds the traces, centres each one adaptively
(`batch.centre`), measures the batch's axis against reference ions
(`batch.massqc`) and applies the fitted wave inside its range, gates the traces
that are noise, and hands the engine ONE synthetic sample -- one peak per
trace, batch-mean height -- through `assign.run(peaks=...)`. Everything
downstream (enumeration, scoring, arbitration, the isotope audit, the tiers,
the merge, the stamp) is peaky as shipped; only the INPUT changes.

The rules, each measured on a month-long TOF batch before it was written down:

  * dedup at the resolution floor. Two positions closer than 0.4 HWHM (Cubison
    & Jimenez 2015) are one observable; a fixed-ppm dedup five times finer
    sliced one peak into a comb and gave each slice its own formula.
  * membership at the measured noise: 4 x the reference ions' per-spectrum
    sigma, floored at the validated 12 ppm TOF cap but never above half the
    cell -- so on a high-resolution instrument, whose cell is ~1 ppm, the floor
    steps aside rather than gluing ten resolved neighbours into one trace.
  * a seed whose members FILL their window is not an ion: a nearest-peak-per-
    spectrum collection in +-W is UNIFORM across the window (robust sd 0.74 W),
    and the seeds that clear a 5 % occurrence gate alone pile up exactly there.
    The test is the shape, not the width -- a Kolmogorov-Smirnov statistic of
    the members against a uniform on their own range, so it is free of the
    window and of the instrument's noise: a seed that cannot reject uniform at
    p ~ 0.001 is rejected, unless it recurs in half the spectra (that is a
    blended peak, flagged `fills_window`, to tier-cap, not to drop). A real ion
    dimmer than the window is wide -- sigma above W/2 -- also reads uniform,
    which is what the 4-sigma membership rule is for.
  * a satellite position (13C, 34S, 81Br, 37Cl, 13C2, 18O of every trace
    present in >= 10 % of spectra) needs 20 members and must co-occur with its
    parent in 60 % of its spectra; NO residual gate on it, because a dim
    satellite scatters at the picker's floor and reads as a fill (the version
    that gated it lost half the 13C evidence). Its own test is the height
    ratio, which the engine's audit applies.
  * heights are batch MEANS with absent = 0: the co-registered estimator the
    isotope ratios rest on (observed / expected 1.02 for 13C on the batch that
    validated it). The per-spectrum picker floor goes to the audit separately.

Not done here, on purpose: nothing decides which FILES to assign (the trace IS
the unit), the reagent ions stay the assignment passes' business, and the
resolvability flag is carried, never used as a filter.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from peaky.batch import centre as CE
from peaky.batch import massqc as MQ
from peaky.batch import traces as TR
from peaky.batch import wave as WV
from peaky.batch.timeseries import sample_hours
from peaky.chem import reference_ions as RI

__version__ = "0.1.0"

SEED_OCC = 0.05          # a peak seeds a trace when it recurs in >= this share of spectra
MIN_MEMBERS = 10
MIN_SAT_MEMBERS = 20
FIT_FLOOR_HWHM = 0.4     # Cubison & Jimenez (2015): below it two peaks are one observable
FILL_FRAC = 0.74         # robust sd of a uniform fill of +-W, in units of W (reported)
KS_CRIT = 1.95           # Kolmogorov-Smirnov statistic x sqrt(n) below which the members
                         # are indistinguishable from a uniform fill of their own range
                         # (two-sided p ~ 0.001): a seed that fails to reject uniform is noise
KEEP_OCC = 0.5           # ... unless it recurs in this share of spectra (blended, not noise)
SAT_COOCCUR = 0.6        # share of a satellite's spectra that must hold its parent
SAT_PARENT_OCC = 0.10    # satellites are probed for traces at least this persistent
MEMBER_SIGMA_X = 4.0     # membership half-window = this x the reference ions' sigma ...
MEMBER_MIN_PPM = 12.0    # ... never below this (the validated TOF cap) UNLESS the
                         # instrument's own cell is narrower ...
                         # ... and never above half the dedup cell, so a trace's
                         # members can never span more than one observable
HI_RES_CELL_PPM = 6.0    # a dedup cell this narrow means a high-resolution instrument:
                         # trace-first is a TOF remedy, and says so rather than pretend
WAVE_LOO_GAIN = 0.8      # apply a wave when its LOO error is below this x the raw scatter
WAVE_MIN_OFFSET = 1.0    # or when a constant wave carries at least this bias (ppm)
ISO_OFFSETS = {"13C": 1.0033548, "13C2": 2.0067096, "34S": 1.9957960,
               "18O": 2.0042460, "81Br": 1.9979521, "37Cl": 1.9970499}
# the columns a fetched Mascope peak table carries and the ledger expects
MATCH_COLS = ("match_score_isotope", "relative_abundance", "target_isotope_id",
              "target_isotope_formula", "target_ion_id", "target_ion_formula",
              "target_compound_id", "target_compound_name", "target_compound_formula",
              "target_collection_ids", "match_score_ion", "match_score_compound",
              "ionization_mechanism")
# engine settings for a trace sample where the caller left the package default
TRACE_DEFAULTS = {"search_ppm": 12.0, "ppm": 5.0, "cal_sigma_floor": 3.0}
# TOF audit knobs, set when the installed PassConfig has them
TRACE_TOF_KNOBS = {"pass0_ppm": 8.0, "audit_sat_ppm": 15.0}
TRACE_COLS = ["peak_id", "mz_raw", "wave_ppm", "n_members", "trace_occurrence", "sigma_ppm",
              "gamma_ppm", "centre_window", "centre_scheme", "resid_ppm", "se_ppm", "span_ppm",
              "fills_window", "ks_uniform", "height_med", "kind", "parent_peak", "parent_cooccur",
              "resolvability", "sep_hwhm", "d_crit_hwhm"]


# ------------------------------------------------------------------ resolution
def hwhm(mz: float, resolving_power: float) -> float:
    """Half width at half maximum (Th) at `mz` for a constant-R instrument."""
    return 0.5 * float(mz) / float(resolving_power)


def dedup_ppm(mz: float, resolving_power: float) -> float:
    """The dedup half-window: 0.4 HWHM in ppm (constant for constant R)."""
    return FIT_FLOOR_HWHM * hwhm(mz, resolving_power) / float(mz) * 1e6


_SIG_PER_HWHM = 1.0 / 1.1774396
_LOGR = np.array([0.0, -0.30103, -0.69897, -1.0, -1.30103, -1.69897, -2.0, -2.30103, -2.69897, -3.0])
_DSIG = np.array([2.000, 2.628, 3.079, 3.354, 3.598, 3.888, 4.088, 4.277, 4.511, 4.679])


def d_crit_hwhm(ratio: float) -> float:
    """Separation (HWHM) at which two Gaussians of height ratio `ratio` (<= 1)
    become bimodal -- what a local-maximum picker needs to report two peaks."""
    r = float(min(max(ratio, 1e-12), 1.0))
    lr = np.log10(r)
    if lr >= _LOGR[0]:
        d = _DSIG[0]
    elif lr <= _LOGR[-1]:
        slope = (_DSIG[-1] - _DSIG[-2]) / (_LOGR[-1] - _LOGR[-2])
        d = _DSIG[-1] + slope * (lr - _LOGR[-1])
    else:
        d = float(np.interp(lr, _LOGR[::-1], _DSIG[::-1]))
    return d / _SIG_PER_HWHM


def classify_pair(mz_a: float, h_a: float, mz_b: float, h_b: float, resolving_power: float) -> dict:
    """'unresolvable' (< 0.4 HWHM: one observable), 'blended' (a fit could
    separate them, the picker cannot: the centroid is displaced) or 'resolved'."""
    hw = hwhm(0.5 * (mz_a + mz_b), resolving_power)
    d = abs(float(mz_a) - float(mz_b))
    sep = d / hw
    hi, lo = max(h_a, h_b), min(h_a, h_b)
    ratio = (lo / hi) if hi > 0 else 0.0
    dc = d_crit_hwhm(ratio)
    cls = "unresolvable" if sep < FIT_FLOOR_HWHM else ("blended" if sep < dc else "resolved")
    return {"sep_hwhm": sep, "d_crit_hwhm": dc, "resolvability": cls, "height_ratio": ratio}


def stamp_resolvability(traces: pd.DataFrame, resolving_power: float) -> pd.DataFrame:
    """Every trace's separability from its nearest neighbour within 8 HWHM. A
    FLAG (and a tier input), never a filter."""
    t = traces.sort_values("mz").reset_index(drop=True)
    mz = t["mz"].to_numpy(float)
    h = np.nan_to_num(t["height_med"].to_numpy(float), nan=0.0)
    cls, sep, dcr = [], [], []
    for i in range(len(t)):
        hw = hwhm(mz[i], resolving_power)
        lo = int(np.searchsorted(mz, mz[i] - 8 * hw))
        hi = int(np.searchsorted(mz, mz[i] + 8 * hw))
        best = None
        for j in range(lo, hi):
            if j != i and (best is None or abs(mz[j] - mz[i]) < best[0]):
                best = (abs(mz[j] - mz[i]), j)
        if best is None:
            cls.append("isolated"); sep.append(np.nan); dcr.append(np.nan)
            continue
        c = classify_pair(mz[i], h[i], mz[best[1]], h[best[1]], resolving_power)
        cls.append(c["resolvability"]); sep.append(c["sep_hwhm"]); dcr.append(c["d_crit_hwhm"])
    t["resolvability"] = cls
    t["sep_hwhm"] = sep
    t["d_crit_hwhm"] = dcr
    return t


# ---------------------------------------------------------------------- traces
def ks_uniform(offsets) -> float:
    """Kolmogorov-Smirnov distance of `offsets` from a uniform on their own
    range: ~0 for a fill of the window, large for a peaked cloud. 0 for fewer
    than three points."""
    d = np.sort(np.asarray(offsets, dtype=float))
    n = len(d)
    if n < 3 or d[-1] <= d[0]:
        return 0.0
    fu = (d - d[0]) / (d[-1] - d[0])
    fe = np.arange(1, n + 1) / n
    return float(max(np.max(np.abs(fe - fu)), np.max(np.abs(fe - 1.0 / n - fu))))


def build_traces(index, hours, *, resolving_power: float, tol_ppm: float,
                 seed_occ: float = SEED_OCC, area_index=None, log=print) -> pd.DataFrame:
    """One row per trace: seeds (peaks recurring in >= seed_occ of spectra,
    brightest first, each consuming its dedup cell) and the satellite positions
    of the persistent ones. See the module note for the gates."""
    R = float(resolving_power)
    tol = float(tol_ppm)
    dedup = lambda m: dedup_ppm(m, R)
    fill = FILL_FRAC * tol
    occ = index.occurrence()
    n = index.n_samples
    order = np.lexsort((-np.nan_to_num(index.height), -occ))
    consumed = np.zeros(len(index), dtype=bool)
    centres: list[float] = []
    rows: list[dict] = []
    rejected: list[tuple] = []
    spectra_of: dict[str, set] = {}

    def near_existing(c: float) -> bool:
        return bool(centres) and bool(np.any(np.abs(np.asarray(centres) - c) / c * 1e6 <= dedup(c)))

    def take(c0: float, kind: str, min_members: int, parent: str | None = None) -> bool:
        mem = index.members(c0, tol)
        if len(mem) < min_members or near_existing(c0):
            return False
        t = hours[index.sample[mem]]
        o = np.argsort(t, kind="mergesort")
        mem, t = mem[o], t[o]
        cen, info = CE.trace_centre(index.mz[mem], t)
        if info["scheme"] == "rolling":
            mem2 = CE.rolling_members(index, t, cen, hours, tol, float(np.median(cen)))
            if len(mem2) >= min_members:
                t2 = hours[index.sample[mem2]]
                o2 = np.argsort(t2, kind="mergesort")
                mem, t = mem2[o2], t2[o2]
                cen, info = CE.trace_centre(index.mz[mem], t)
        c = float(np.median(cen))
        if near_existing(c):
            return False
        q = info["resid_ppm"]
        occ_c = len(mem) / n
        ks = ks_uniform((index.mz[mem] - c) / c * 1e6)
        fills = bool(ks * np.sqrt(len(mem)) < KS_CRIT)      # cannot reject a uniform fill
        cooc = np.nan
        if parent is not None:
            cooc = float(np.isin(index.sample[mem], list(spectra_of[parent])).mean())
        bad = (kind == "seed" and fills and occ_c < KEEP_OCC) \
            or (kind != "seed" and cooc < SAT_COOCCUR)
        if bad:
            rejected.append((kind, q, cooc, ks))
            rhw = c * dedup(c) * 1e-6
            r0, r1 = index.window(c - rhw, c + rhw)
            consumed[r0:r1] = True
            consumed[mem] = True
            return False
        hw = c * dedup(c) * 1e-6
        i0, i1 = index.window(c - hw, c + hw)
        consumed[i0:i1] = True
        consumed[mem] = True
        h = np.nan_to_num(index.height[mem])
        a = np.nan_to_num(area_index.height[mem]) if area_index is not None else h
        centres.append(c)
        pid = f"T{len(rows):06d}"
        spectra_of[pid] = set(index.sample[mem].tolist())
        rows.append(dict(peak_id=pid, mz=c, n_members=int(len(mem)), trace_occurrence=occ_c,
                         sigma_ppm=info["sigma"], gamma_ppm=info["gamma"], rise=info["rise"],
                         centre_window=int(info["window"]), centre_scheme=info["scheme"],
                         resid_ppm=q, se_ppm=info["se_ppm"],
                         span_ppm=float(np.ptp(cen) / c * 1e6), fills_window=bool(fills),
                         ks_uniform=float(ks),
                         height_avg=float(h.sum() / n), area_avg=float(a.sum() / n),
                         height_med=float(np.median(h)), kind=kind, parent_peak=parent,
                         parent_cooccur=cooc))
        return True

    for i in order:
        if occ[i] < seed_occ:
            break
        if consumed[i]:
            continue
        take(index.mean_shift(float(index.mz[i]), tol_ppm=tol, max_drift_ppm=tol), "seed", MIN_MEMBERS)
    n_seed = len(rows)
    rej_seed = [q for k, q, _, _ in rejected if k == "seed"]
    log(f"[traces] R={R:g}: dedup {FIT_FLOOR_HWHM} HWHM = {dedup(200.0):.1f} ppm; membership "
        f"+-{tol:g} ppm; a uniform fill reads {fill:.2f} ppm; gate rejects a seed whose "
        f"members cannot reject uniform (KS x sqrt(n) < {KS_CRIT}) unless occurrence >= "
        f"{KEEP_OCC:.0%}")
    log(f"[traces] {n_seed} seed traces from peaks recurring in >= {seed_occ:.0%} of {n} spectra; "
        f"{len(rej_seed)} seed positions rejected as fills"
        + (f" (median residual {np.median(rej_seed):.2f} ppm)" if rej_seed else ""))
    log(f"[traces] {sum(r['fills_window'] for r in rows)} kept seeds fill their window but recur "
        f"in >= {KEEP_OCC:.0%} (`fills_window`: blended, not noise); "
        f"{sum(r['centre_scheme'] == 'rolling' for r in rows)} of {n_seed} roll")
    base = [(r["peak_id"], r["mz"]) for r in rows if r["trace_occurrence"] >= SAT_PARENT_OCC]
    for pid, m in base:
        for lab, dm in ISO_OFFSETS.items():
            c0 = m + dm
            if len(index.members(c0, tol)) < MIN_SAT_MEMBERS:
                continue
            take(index.mean_shift(c0, tol_ppm=tol, max_drift_ppm=0.7 * tol), f"sat:{lab}",
                 MIN_SAT_MEMBERS, parent=pid)
    n_sat = len(rows) - n_seed
    rej_sat = [c for k, _, c, _ in rejected if k != "seed"]
    log(f"[traces] + {n_sat} isotopologue-position traces (>= {MIN_SAT_MEMBERS} members, "
        f"co-occurrence >= {SAT_COOCCUR:.0%} with the parent); {len(rej_sat)} satellite "
        f"positions rejected")
    if not rows:
        return pd.DataFrame(columns=["peak_id", "mz"] + [c for c in TRACE_COLS if c not in ("peak_id", "mz_raw", "wave_ppm", "resolvability", "sep_hwhm", "d_crit_hwhm")])
    return pd.DataFrame(rows).sort_values("mz").reset_index(drop=True)


def apply_wave(traces: pd.DataFrame, qc: dict | None, log=print) -> pd.DataFrame:
    """Correct the centres by the mass-qc wave INSIDE the calibrant range only,
    when the wave predicts held-out reference ions better than the constant
    (LOO < WAVE_LOO_GAIN x raw) or the constant itself is a bias."""
    t = traces.copy()
    t["mz_raw"] = t["mz"]
    t["wave_ppm"] = 0.0
    w = (qc or {}).get("wave")
    if not w:
        log("[wave] no wave to apply")
        return t
    K, loo, raw, c0 = w["K"], w["loo_ppm"], w["raw_ppm"], w["coef"][0]
    if not ((K >= 1 and loo < WAVE_LOO_GAIN * raw) or (K == 0 and abs(c0) >= WAVE_MIN_OFFSET)):
        log(f"[wave] not applied: degree {K}, LOO {loo:.2f} vs raw {raw:.2f} ppm")
        return t
    fit = WV.WaveFit(**w)
    lo, hi = fit.mz_range
    inside = ((t["mz"] >= lo) & (t["mz"] <= hi)).to_numpy()
    d = fit.predict(t["mz"].to_numpy(float))
    t.loc[inside, "wave_ppm"] = d[inside]
    t.loc[inside, "mz"] = t.loc[inside, "mz_raw"] * (1 - t.loc[inside, "wave_ppm"] * 1e-6)
    log(f"[wave] applied degree-{K} wave (swing {fit.span_ppm:.1f} ppm, LOO {loo:.2f} vs raw "
        f"{raw:.2f}) to {int(inside.sum())} of {len(t)} traces inside m/z {lo:.0f}-{hi:.0f}; "
        f"{int((~inside).sum())} outside left uncorrected")
    return t


def synthetic_sample(traces: pd.DataFrame, sample_id: str) -> pd.DataFrame:
    """The engine's input: one peak per trace, batch-mean height and area."""
    t = pd.DataFrame({"sample_item_id": sample_id, "peak_id": traces["peak_id"],
                      "mz": traces["mz"], "sparsity": 1.0 - traces["trace_occurrence"],
                      "area": traces["area_avg"], "height": traces["height_avg"]})
    for c in MATCH_COLS:
        t[c] = None
    return t


def occurrence_table(traces: pd.DataFrame, tol_ppm: float, n_samples: int) -> pd.DataFrame:
    """The admission table for the trace sample: every trace's own occurrence."""
    occ = pd.DataFrame({"mz": traces["mz"].to_numpy(float),
                        "occurrence": traces["trace_occurrence"].to_numpy(float),
                        "height": traces["height_avg"].to_numpy(float)})
    occ.attrs["tol_ppm"] = float(tol_ppm)
    occ.attrs["n_samples"] = int(n_samples)
    return occ


@dataclass
class TraceSample:
    sample_id: str
    peaks: pd.DataFrame            # the engine's input
    traces: pd.DataFrame           # one row per trace (TRACE_COLS and more)
    occurrence: pd.DataFrame       # admission table
    qc: dict | None                # mass-qc verdict (None: no reference table)
    tol_ppm: float
    resolving_power: float
    n_spectra: int
    picker_floor_cps: float | None = None
    notes: list = field(default_factory=list)

    def summary(self) -> dict:
        t = self.traces
        return {"n_traces": int(len(t)), "n_seeds": int((t["kind"] == "seed").sum()) if len(t) else 0,
                "n_satellites": int(t["kind"].astype(str).str.startswith("sat").sum()) if len(t) else 0,
                "n_spectra": int(self.n_spectra), "membership_ppm": float(self.tol_ppm),
                "resolving_power": float(self.resolving_power),
                "dedup_ppm": float(dedup_ppm(200.0, self.resolving_power)),
                "n_rolling": int((t["centre_scheme"] == "rolling").sum()) if len(t) else 0,
                "n_fills_kept": int(t["fills_window"].fillna(False).astype(bool).sum()) if len(t) else 0,
                "n_wave_corrected": int((t["wave_ppm"] != 0).sum()) if "wave_ppm" in t.columns else 0,
                "resolvability": (t["resolvability"].value_counts().to_dict()
                                  if "resolvability" in t.columns else {}),
                "mass_qc": ({k: self.qc.get(k) for k in ("verdict", "remedy", "calibrant_tier",
                                                          "calibrant_n", "median_offset_ppm",
                                                          "ion_to_ion_spread_ppm", "median_sigma_ppm",
                                                          "wave_K", "wave_span_ppm", "wave_loo_ppm",
                                                          "wave_raw_ppm", "wave_mz_range")}
                            if self.qc else None),
                "notes": list(self.notes)}


def build_trace_sample(ts_peaks: pd.DataFrame, *, sample_id: str, reagent: str,
                       resolving_power: float, tol_ppm: float | None = None,
                       seed_occ: float = SEED_OCC, sample_col: str = "sample_item_id",
                       time_col: str = "datetime_utc", log=print) -> TraceSample:
    """Traces -> mass-qc -> membership -> gate -> wave -> one synthetic sample."""
    notes: list[str] = []
    idx = TR.PeakIndex(ts_peaks, tol_ppm=MEMBER_MIN_PPM, sample_col=sample_col)
    hours = sample_hours(idx, ts_peaks, sample_col=sample_col, time_col=time_col)
    if hours is None or not np.isfinite(hours).any():
        hours = np.arange(idx.n_samples, dtype=float)      # spectrum order
        notes.append("no timestamps: traces centred in spectrum order")
    hours = np.nan_to_num(np.asarray(hours, dtype=float), nan=0.0)
    log(f"[traces] {idx.n_samples} spectra, {len(idx)} picked peaks")
    qc = None
    sig = None
    try:
        refs = RI.get(reagent)
    except KeyError:
        refs = None
        notes.append(f"no reference-ion table for {reagent!r}: no mass-qc, membership "
                     f"{MEMBER_MIN_PPM:g} ppm")
        log(f"[mass-qc] no reference-ion table for {reagent!r}; skipped")
    if refs is not None:
        table = MQ.twin_check(MQ.probe(idx, hours, refs, trace_ppm=MEMBER_MIN_PPM))
        qc = MQ.verdict(table, idx.n_samples, tof=True)
        MQ.report(table, qc, log=log)
        sig = qc.get("median_sigma_ppm")
    cell = dedup_ppm(200.0, resolving_power)
    half = cell / 2.0
    if tol_ppm is None:
        # the floor is the validated TOF cap -- but never wider than half the
        # instrument's OWN cell, or a trace spans more than one observable: at
        # R = 155 000 the cell is ~1.3 ppm, where a 12 ppm window would collect
        # ten resolved neighbours into one trace
        lo = min(MEMBER_MIN_PPM, half)
        tol = float(np.clip(MEMBER_SIGMA_X * sig if sig and np.isfinite(sig) else lo, lo, half))
        log(f"[traces] membership +-{tol:.2f} ppm = clip({MEMBER_SIGMA_X:g} x sigma_ref "
            f"{None if sig is None else round(sig, 2)}, {lo:.2f}, {half:.2f})")
    else:
        tol = float(tol_ppm)
        log(f"[traces] membership +-{tol:.2f} ppm (given)")
    if tol != MEMBER_MIN_PPM:
        idx = TR.PeakIndex(ts_peaks, tol_ppm=tol, sample_col=sample_col)
    area_idx = (TR.PeakIndex(ts_peaks, tol_ppm=tol, sample_col=sample_col, height_col="area")
                if "area" in ts_peaks.columns else None)
    traces = build_traces(idx, hours, resolving_power=resolving_power, tol_ppm=tol,
                          seed_occ=seed_occ, area_index=area_idx, log=log)
    if len(traces):
        traces = stamp_resolvability(traces, resolving_power)
        traces = apply_wave(traces, qc, log=log)
        log(f"[traces] resolvability {traces['resolvability'].value_counts().to_dict()}")
    else:
        traces["mz_raw"] = traces.get("mz", pd.Series(dtype=float))
        traces["wave_ppm"] = 0.0
        for c in ("resolvability", "sep_hwhm", "d_crit_hwhm"):
            traces[c] = np.nan
        notes.append("no traces built")
    from peaky.assignment import passes as PA
    floor = float(PA.noise_edge(ts_peaks["height"])) if "height" in ts_peaks.columns and len(ts_peaks) else None
    if cell <= HI_RES_CELL_PPM:
        msg = (f"R = {resolving_power:g} puts the resolution floor at {cell:.2f} ppm: on a "
               f"high-resolution instrument a per-file mass is already good to a fraction of a "
               f"ppm, so the per-file route has little to lose to. Trace-first is a TOF remedy; "
               f"it will run, but expect a reorganisation, not a gain")
        notes.append(msg)
        log(f"[traces] NOTE: {msg}")
    log(f"[traces] {len(traces)} traces -> synthetic sample {sample_id!r}")
    return TraceSample(sample_id=sample_id, peaks=synthetic_sample(traces, sample_id), traces=traces,
                       occurrence=occurrence_table(traces, tol, idx.n_samples), qc=qc, tol_ppm=tol,
                       resolving_power=float(resolving_power), n_spectra=int(idx.n_samples),
                       picker_floor_cps=floor, notes=notes)


def engine_settings(cfg, sample: TraceSample, log=print) -> dict:
    """Point the engine at a trace sample: the wider TOF windows where the
    caller left the package default, the TOF audit knobs where this PassConfig
    has them, and the per-spectrum picker floor for the isotope audit. Returns
    what was set."""
    from peaky.assignment import passes as PA
    base = PA.PassConfig()
    applied = {}
    for k, v in TRACE_DEFAULTS.items():
        if getattr(cfg, k, None) == getattr(base, k, None):
            setattr(cfg, k, v)
            applied[k] = v
    for k, v in TRACE_TOF_KNOBS.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
            applied[k] = v
    if hasattr(cfg, "audit_floor_cps") and sample.picker_floor_cps is not None:
        cfg.audit_floor_cps = sample.picker_floor_cps
        applied["audit_floor_cps"] = sample.picker_floor_cps
    missing = [k for k in (*TRACE_TOF_KNOBS, "audit_floor_cps") if not hasattr(cfg, k)]
    if missing:
        log(f"[traces] this PassConfig has no {missing}: the isotope audit judges the "
            f"trace sample by its own batch-mean floor (dim formulas may be cleared)")
    log(f"[traces] engine settings for the trace sample: {applied}")
    return applied


def slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", str(name)).strip("-") or "batch"
