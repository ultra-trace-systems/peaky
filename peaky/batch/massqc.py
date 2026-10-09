"""`peaky mass-qc`: measure a batch's mass state against an EXTERNAL reference
and return a verdict with the remedy it implies.

Today peaky's only mass reference is its own output: pass 1 self-calibrates on
the Assigned backbone, so an axis 30 ppm wrong yields a self-consistent
calibration and a batch of confident wrong formulas, and nothing in the run log
says the axis moved. This probes formula-certain reference ions
(`chem.reference_ions`) in the batch's time series instead, and reports per ion:
occurrence, offset from theory, the trace's white noise `sigma`, its random-walk
step `gamma`, the optimal averaging window `W*`, and whether a second picked
peak shares its window.

Verdicts (`verdict`):

  clean         offsets small and flat                -> nothing to do
  axis_offset   a flat bias, no swing worth a wave    -> recalibrate with a constant
  axis_trend    a smooth wave in (m/z)^p fits AND
                predicts held-out ions               -> apply the wave (batch.wave)
  blended       offsets jump ion to ion and no smooth
                component predicts them              -> centre only, widen the
                                                         tolerance, cap the tier
  drifting      (appended) gamma resolvable, W* << n  -> roll the centre
  no_reference  too few reference ions present

Two procedural rules, both learned from a calibration session that got them
wrong: every gate here is ABSOLUTE (a residual in ppm, an occurrence fraction)
-- a gate phrased "at or below the batch median" deletes every persistent ion
of a TOF batch whose median jitter is 0 because most of its ions are singletons;
and a rule that relaxes when the strict calibrant set is too small REPORTS WHICH
TIER IT USED (`calibrant_tier`, `calibrant_tiers_tried`), because a fallback
that walks silently through four tiers publishes rows that are not what they
claim to be.

The instrument's own rules (`per_instrument=True`; `peaky batch --mass-axis`
and `peaky mass-qc` judge with them, trace-first keeps the rules above). The
tier ladder and the thresholds above were sized on TOF batches; on an Orbitrap
they misread the one axis error this module exists to catch -- a nitrate
Orbitrap whose axis rose from 0 ppm at m/z 62 to +2.4 ppm near m/z 200 and fell
back to ~0 by m/z 350, identically in every file:
  * the anchor tier exists for TOF PURITY (an anchor is an ion clean at TOF
    resolution); on an Orbitrap every reference ion is resolved, and the
    anchors (m/z 62-220) leave the upper half of the range unmeasured. There
    the calibrants are every usable reference ion;
  * minus the BRIGHT ones (`BRIGHT_X` times the usable ions' median height):
    the reagent ions sit off the analyte axis (there the lock masses NO3- and
    HNO3.NO3- read 0.0 ppm while their neighbours read +1), and a wave forced
    through them swung 7 ppm;
  * a 4 ppm swing / 2 ppm offset is ~1 TOF scoring width but ~10 Orbitrap ones;
    the Orbitrap trend and offset bars are `ORBI_TREND_SPAN_PPM` /
    `ORBI_OFFSET_FLAT_PPM`, and a trend must PREDICT held-out ions with half
    the constant's error (`ORBI_LOO_GAIN`), not 0.8 of it.
A TOF batch is judged by the TOF rules above whichever rules are asked for
(`min_present_for` applies to every caller).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from peaky.batch.centre import trace_centre, rsd
from peaky.batch.wave import WaveFit, fit_wave

__version__ = "0.2.0"
__all__ = ["probe", "twin_check", "calibrant_set", "verdict", "report", "run",
           "correction", "apply_correction", "in_scope", "member_ppm", "min_present_for",
           "CAL_TIERS", "PROBE_PPM", "MAX_CORRECTION_PPM"]

PROBE_PPM = 50.0        # wide enough to find an axis error before it hides one
MIN_PRESENT = 10        # peaks in the probe window / members before an ion counts
MIN_PRESENT_FLOOR = 3   # ... scaled down for a batch of few spectra (min_present_for)
OFFSET_FLAT_PPM = 2.0   # |constant| at or above this -> axis_offset
TREND_SPAN_PPM = 4.0    # wave swing at or above this ...
TREND_SHARE = 0.4       # ... explaining at least this share ...
TREND_LOO_GAIN = 0.8    # ... AND predicting held-out ions better than the constant
                        # (LOO < this x the raw scatter) -> axis_trend; a fit that
                        # explains the calibrants it saw but not the one it did not
                        # is not a trend, whatever its share
BLEND_SPREAD_PPM = 5.0  # ion-to-ion spread at or above this, with no PREDICTIVE
                        # smooth component -> blended (a wave can always be fitted to
                        # jumps; only one that predicts held-out ions is a trend)
ROLL_FRAC = 0.5         # share of usable ions whose centre rolls ...
ROLL_W_FRAC = 0.5       # ... with W* below this fraction of the batch -> drifting
USABLE_OCC = 0.5        # an ion is usable at this occurrence or better

# the instrument's own rules on an Orbitrap (per_instrument=True; module note)
BRIGHT_X = 100.0            # a reference ion this many times the usable ions' median
                            # height does not calibrate the axis (reagent / lock mass)
ORBI_TREND_SPAN_PPM = 1.0   # Orbitrap: a wave swinging at least this much ...
ORBI_LOO_GAIN = 0.5         # ... and predicting held-out ions this much better than
                            # the constant (LOO < this x raw) -> axis_trend
ORBI_OFFSET_FLAT_PPM = 1.0  # Orbitrap: |median offset| at or above this -> axis_offset
MEMBER_PPM_TOF = 12.0       # trace membership half-window (`peaky mass-qc` defaults)
MEMBER_PPM_ORBI = 6.0


def member_ppm(tof: bool) -> float:
    """The trace-membership half-window an axis is probed with on this analyser."""
    return MEMBER_PPM_TOF if tof else MEMBER_PPM_ORBI


def min_present_for(n_samples: int) -> int:
    """The presence count an ion needs in a batch of `n_samples` spectra. An
    absolute MIN_PRESENT made every batch of fewer spectra unmeasurable: a time
    series of one aggregated spectrum per file (9 files) found 0 of 30 nitrate
    reference ions that sat in every file. Scaled to what USABLE_OCC of the
    batch is, floored at MIN_PRESENT_FLOOR and capped at MIN_PRESENT, so a batch
    of 20 spectra or more is judged exactly as before."""
    return int(min(MIN_PRESENT,
                   max(MIN_PRESENT_FLOOR, math.ceil(USABLE_OCC * max(int(n_samples), 0)))))


def probe(index, hours, refs: pd.DataFrame, *, probe_ppm: float = PROBE_PPM,
          trace_ppm: float | None = None) -> pd.DataFrame:
    """One row per reference ion: found, centre, offset, sigma/gamma/W*, second-peak
    share. `index` is a batch.traces.PeakIndex, `hours` a float time per sample
    code, `refs` a reference_ions table. An ion counts once it is present in
    `min_present_for(index.n_samples)` spectra."""
    trace_ppm = float(trace_ppm or index.tol_ppm)
    need = min_present_for(index.n_samples)
    rows = []
    for _, r in refs.iterrows():
        th = float(r["mz"])
        hw = th * probe_ppm * 1e-6
        i0, i1 = index.window(th - hw, th + hw)
        row = dict(mz_theory=th, neutral=r["neutral"], channel=r["channel"],
                   identity=r["identity"], grade=r["grade"], anchor_prior=bool(r["anchor"]),
                   iso=r.get("iso", ""), twin_key=r.get("twin_key", f"{r['neutral']}|{r['channel']}"),
                   n_in_probe=int(i1 - i0))
        if i1 - i0 < need:
            rows.append({**row, "occurrence": 0.0, "found": False})
            continue
        c0 = index.mean_shift(th, tol_ppm=trace_ppm, max_drift_ppm=probe_ppm)
        mem = index.members(c0, trace_ppm)
        if len(mem) < need:
            rows.append({**row, "occurrence": len(mem) / index.n_samples, "found": False})
            continue
        t = hours[index.sample[mem]]
        o = np.argsort(t, kind="mergesort")
        mem, t = mem[o], t[o]
        cen, info = trace_centre(index.mz[mem], t)
        centre = float(np.median(cen))
        # a SECOND picked peak sharing the window in most spectra? counted at the
        # final centre against the members re-collected there
        mem2 = index.members(centre, trace_ppm)
        j0, j1 = index.window(centre * (1 - trace_ppm * 1e-6), centre * (1 + trace_ppm * 1e-6))
        n_second = max((j1 - j0) - len(mem2), 0)
        rows.append({**row, "found": True, "centre_mz": centre,
                     "offset_ppm": (centre - th) / th * 1e6,
                     "occurrence": len(mem) / index.n_samples, "n": int(len(mem)),
                     "sigma_ppm": info["sigma"], "gamma_ppm": info["gamma"],
                     "rise": info["rise"],
                     "W_star": info["window"] if info["scheme"] == "rolling" else np.nan,
                     "scheme": info["scheme"], "resid_ppm": info["resid_ppm"],
                     "se_ppm": info["se_ppm"],
                     "second_peak_frac": n_second / max(len(mem2), 1),
                     "height": float(np.nanmedian(index.height[mem]))})
    cols = ["mz_theory", "neutral", "channel", "identity", "grade", "anchor_prior", "iso",
            "twin_key", "n_in_probe", "found", "occurrence", "centre_mz", "offset_ppm", "n",
            "sigma_ppm", "gamma_ppm", "rise", "W_star", "scheme", "resid_ppm", "se_ppm",
            "second_peak_frac", "height"]
    t = pd.DataFrame(rows)
    for c in cols:
        if c not in t.columns:
            t[c] = np.nan
    return t[cols]


def twin_check(t: pd.DataFrame, *, max_dppm: float = 3.0,
               ratio_win: tuple = (0.55, 1.65), max_docc: float = 0.25) -> pd.DataFrame:
    """The heavy-isotope twin as an internal axis check. Rows sharing a `twin_key`
    are one ion's light (iso '') and heavy (iso '81Br') lines; both sit on the same
    axis, so if their offsets disagree by more than `max_dppm`, their height ratio
    leaves `ratio_win`, or their occurrences differ by more than `max_docc`, the
    position is not one clean ion and must not calibrate anything -- whatever the
    formula says. Adds twin_dppm / twin_ratio / twin_ok (True where no twin)."""
    t = t.copy()
    t["twin_dppm"] = np.nan
    t["twin_ratio"] = np.nan
    t["twin_ok"] = True
    for _, g in t.groupby("twin_key"):
        if len(g) < 2:
            continue
        light = g[g["iso"].astype(str) == ""]
        heavy = g[g["iso"].astype(str) != ""]
        if not len(light) or not len(heavy):
            continue
        li, hv = light.iloc[0], heavy.iloc[0]
        if not (bool(li["found"]) and bool(hv["found"])):
            t.loc[g.index, "twin_ok"] = False
            continue
        d = float(hv["offset_ppm"] - li["offset_ppm"])
        rr = float(hv["height"] / li["height"]) if li["height"] else np.nan
        ok = (abs(d) <= max_dppm and ratio_win[0] <= rr <= ratio_win[1]
              and abs(float(hv["occurrence"]) - float(li["occurrence"])) <= max_docc)
        t.loc[g.index, "twin_dppm"] = d
        t.loc[g.index, "twin_ratio"] = rr
        t.loc[g.index, "twin_ok"] = bool(ok)
    return t


# calibrant tiers, strictest first
CAL_TIERS = [
    ("anchor", lambda t: t["found"] & (t["occurrence"] >= USABLE_OCC) & t["twin_ok"]
                         & t["anchor_prior"]),
    ("grade-A", lambda t: t["found"] & (t["occurrence"] >= USABLE_OCC) & t["twin_ok"]
                          & t["grade"].astype(str).str.startswith("A")),
    ("all-usable", lambda t: t["found"] & (t["occurrence"] >= USABLE_OCC) & t["twin_ok"]),
    ("relaxed-occ", lambda t: t["found"] & (t["occurrence"] >= 0.2) & t["twin_ok"]),
    ("twin-only", lambda t: t["found"] & t["twin_ok"]),
]
# an Orbitrap's tiers (per_instrument=True): no purity ladder -- every reference ion
# is resolved there -- just the occurrence relaxations, each without the bright ions
ORBI_CAL_TIERS = [t for t in CAL_TIERS if t[0] in ("all-usable", "relaxed-occ", "twin-only")]
CAL_MIN_N = 6


def bright_mask(t: pd.DataFrame) -> pd.Series:
    """The reference ions too bright to calibrate an Orbitrap axis: height above
    BRIGHT_X x the median height of the usable ions (found, USABLE_OCC, twin-ok;
    of the found ones when none is usable). False where the height is unknown."""
    found = t["found"].fillna(False).astype(bool)
    usable = found & (t["occurrence"] >= USABLE_OCC) & t["twin_ok"].fillna(False).astype(bool)
    base = t.loc[usable if usable.any() else found, "height"]
    med = float(np.nanmedian(base)) if base.notna().any() else float("nan")
    if not np.isfinite(med) or med <= 0:
        return pd.Series(False, index=t.index)
    return (t["height"] > BRIGHT_X * med).fillna(False)


def calibrant_set(t: pd.DataFrame, min_n: int = CAL_MIN_N, *, tof: bool = True,
                  per_instrument: bool = False):
    """The calibrants and the NAME of the tier they came from, plus every tier
    tried with its count. Never silent. With `per_instrument` an Orbitrap
    (`tof=False`) walks ORBI_CAL_TIERS with the bright ions (`bright_mask`)
    taken out of every tier; a TOF walks CAL_TIERS either way."""
    tried = []
    orbi = per_instrument and not tof
    pool = t[~bright_mask(t)] if orbi else t
    for name, rule in (ORBI_CAL_TIERS if orbi else CAL_TIERS):
        sel = pool[rule(pool).fillna(False).astype(bool)]
        tried.append((name, int(len(sel))))
        if len(sel) >= min_n:
            return sel, name, tried
    sel = pool[pool["found"].astype(bool)]
    tried.append(("found-any", int(len(sel))))
    return sel, "found-any", tried


def _f(v):
    return None if v is None or (isinstance(v, float) and not np.isfinite(v)) else float(v)


def _nanmed(x):
    """The median of the finite values, or None (no all-NaN RuntimeWarning: a batch
    of one aggregated spectrum per file has no per-spectrum sigma at all)."""
    x = pd.to_numeric(pd.Series(x), errors="coerce")
    x = x[np.isfinite(x)]
    return float(x.median()) if len(x) else None


def _rules(tof: bool, per_instrument: bool) -> dict:
    """The verdict thresholds in force (recorded in the verdict)."""
    if per_instrument and not tof:
        return dict(name="orbitrap", trend_span_ppm=ORBI_TREND_SPAN_PPM,
                    trend_loo_gain=ORBI_LOO_GAIN, offset_flat_ppm=ORBI_OFFSET_FLAT_PPM,
                    bright_x=BRIGHT_X)
    return dict(name="legacy", trend_span_ppm=TREND_SPAN_PPM, trend_loo_gain=TREND_LOO_GAIN,
                offset_flat_ppm=OFFSET_FLAT_PPM, bright_x=None)


def verdict(t: pd.DataFrame, n_spectra: int, *, tof: bool = True,
            per_instrument: bool = False) -> dict:
    """The batch's mass state from a probed (and twin-checked) table. JSON-safe.
    `per_instrument` judges an Orbitrap (`tof=False`) by its own rules (module
    note); a TOF is judged the same either way."""
    if "twin_ok" not in t.columns:
        t = twin_check(t)
    found = t["found"].astype(bool)
    usable = t[found & (t["occurrence"] >= USABLE_OCC) & t["twin_ok"].astype(bool)]
    cal, tier, tried = calibrant_set(t, tof=tof, per_instrument=per_instrument)
    rules = _rules(tof, per_instrument)
    first = (ORBI_CAL_TIERS if rules["name"] == "orbitrap" else CAL_TIERS)[0][0]
    bright = t[bright_mask(t)] if rules["name"] == "orbitrap" else t.iloc[0:0]
    out = dict(n_probed=int(len(t)), n_found=int(len(usable)),
               n_twin_fail=int((~t["twin_ok"].astype(bool)).sum()),
               calibrant_tier=tier, calibrant_n=int(len(cal)),
               calibrant_tier_fallback=bool(tier != first),
               calibrant_tiers_tried=tried, n_spectra=int(n_spectra), rules=rules,
               n_bright_excluded=int(len(bright)),
               bright_excluded=[f"{r.neutral} {r.channel} {r.mz_theory:.4f}"
                                for r in bright.itertuples()])
    if len(cal) < 4:
        out.update(verdict="no_reference",
                   remedy="too few reference ions present to measure this batch's axis "
                          "-- widen the probe, or build a reference list for this reagent")
        return out
    med = float(np.median(cal["offset_ppm"]))
    spread = rsd(cal["offset_ppm"])
    w = fit_wave(cal["mz_theory"].to_numpy(float), cal["offset_ppm"].to_numpy(float), tof=tof)
    roll_frac = float((usable["scheme"] == "rolling").mean()) if len(usable) else float("nan")
    Wmed = float(np.nanmedian(usable["W_star"])) if usable["W_star"].notna().any() else float("nan")
    out.update(median_offset_ppm=med, ion_to_ion_spread_ppm=spread,
               rolling_frac=_f(roll_frac), W_star_median=_f(Wmed),
               blend_frac=_f((usable["second_peak_frac"] > 0.5).mean()) if len(usable) else None,
               median_sigma_ppm=_nanmed(cal["sigma_ppm"]),
               median_gamma_ppm=_nanmed(cal["gamma_ppm"]),
               median_resid_ppm=_nanmed(cal["resid_ppm"]))
    if w is None:
        out.update(wave=None, verdict="clean",
                   remedy="no fittable wave (too few calibrants); assign as usual")
        return out
    out.update(wave=w.as_dict(), wave_K=w.K, wave_resid_ppm=w.resid_ppm, wave_raw_ppm=w.raw_ppm,
               wave_loo_ppm=w.loo_ppm, wave_span_ppm=w.span_ppm, wave_share=w.share,
               wave_mz_range=[float(w.mz_range[0]), float(w.mz_range[1])],
               wave_n=w.n, wave_n_clipped=w.n_clipped)
    v, rem = "clean", "nothing to do -- assign as usual"
    trend = (w.K >= 1 and w.span_ppm >= rules["trend_span_ppm"] and w.share >= TREND_SHARE
             and w.loo_ppm < rules["trend_loo_gain"] * w.raw_ppm)
    if trend:
        v, rem = "axis_trend", (f"a degree-{w.K} wave in (m/z)^{w.p:+g} explains {w.share:.0%} "
                                f"of a {w.span_ppm:.1f} ppm swing ({w.raw_ppm:.2f} -> "
                                f"{w.resid_ppm:.2f} ppm, LOO {w.loo_ppm:.2f}) -- apply it "
                                f"inside m/z {w.mz_range[0]:.0f}-{w.mz_range[1]:.0f}")
    elif spread >= BLEND_SPREAD_PPM:
        v, rem = "blended", (f"offsets jump ion to ion ({spread:.1f} ppm spread) and no smooth "
                             f"component predicts them (LOO {w.loo_ppm:.2f} vs raw {w.raw_ppm:.2f}) "
                             "-- unresolved neighbours: centre only, widen the stamping "
                             "tolerance, cap the confidence tier")
    elif abs(med) >= rules["offset_flat_ppm"]:
        # a flat bias: the wave is (nearly) constant whatever degree the CV picked
        v, rem = "axis_offset", (f"flat {med:+.2f} ppm bias (wave degree {w.K}, swing "
                                 f"{w.span_ppm:.1f} ppm, residual {w.resid_ppm:.2f} ppm) -- "
                                 "recalibrate with a constant")
    if roll_frac >= ROLL_FRAC and np.isfinite(Wmed) and Wmed < ROLL_W_FRAC * n_spectra:
        v = "drifting" if v == "clean" else v + "+drifting"
        rem += (f" | positions move: {roll_frac:.0%} of reference ions roll, W* median "
                f"{Wmed:.0f} of {n_spectra} spectra -- use the rolling centre")
    out.update(verdict=v, remedy=rem)
    return out


def report(t: pd.DataFrame, v: dict, log=print) -> None:
    log(f"[mass-qc] {v['n_found']}/{v['n_probed']} reference ions usable (present in "
        f">={USABLE_OCC:.0%} of spectra and twin-consistent); {v.get('n_twin_fail', 0)} "
        f"withdrawn by the isotope-twin test")
    tiers = ", ".join(f"{n}={k}" for n, k in v.get("calibrant_tiers_tried", []))
    log(f"[mass-qc] CALIBRANT TIER USED: {v['calibrant_tier']} (n={v.get('calibrant_n')})"
        + ("  <-- FELL BACK from the strict tier" if v.get("calibrant_tier_fallback") else "")
        + f"   [tiers tried: {tiers}]")
    rules = v.get("rules") or {}
    if rules.get("name") == "orbitrap":
        log(f"[mass-qc] Orbitrap rules: trend >= {rules['trend_span_ppm']:g} ppm swing with LOO "
            f"< {rules['trend_loo_gain']:g} x raw, offset >= {rules['offset_flat_ppm']:g} ppm; "
            f"{v.get('n_bright_excluded', 0)} reference ion(s) > {rules['bright_x']:g}x the median "
            f"height kept out of the calibrants"
            + (f" ({'; '.join(v['bright_excluded'])})" if v.get("bright_excluded") else ""))
    if v["verdict"] == "no_reference":
        log(f"[mass-qc] VERDICT NO_REFERENCE -- {v['remedy']}")
        return
    log(f"[mass-qc] offset median {v['median_offset_ppm']:+.2f} ppm, ion-to-ion spread "
        f"{v['ion_to_ion_spread_ppm']:.2f} ppm")
    if v.get("wave"):
        lo, hi = v["wave_mz_range"]
        log(f"[mass-qc] wave: degree {v['wave_K']} in (m/z)^{v['wave']['p']:+g}, "
            f"{v['wave_raw_ppm']:.2f} -> {v['wave_resid_ppm']:.2f} ppm (LOO {v['wave_loo_ppm']:.2f}), "
            f"swing {v['wave_span_ppm']:.1f} ppm, explains {v['wave_share']:.0%}; "
            f"{v['wave_n']} calibrants, {v['wave_n_clipped']} clipped")
        log(f"[mass-qc] SCOPE: calibrants populate m/z {lo:.1f}-{hi:.1f} only -- the wave "
            f"must not be applied outside it")
    s, g = v.get("median_sigma_ppm"), v.get("median_gamma_ppm")
    log(f"[mass-qc] sigma {s if s is None else round(s, 2)} ppm/spectrum, gamma "
        f"{g if g is None else round(g, 3)} ppm/spectrum, W* median {v.get('W_star_median')}, "
        f"{(v.get('rolling_frac') or 0):.0%} of ions roll")
    log(f"[mass-qc] VERDICT {v['verdict'].upper()} -- {v['remedy']}")


#: how far past its calibrants' THEORETICAL m/z range a trend still reaches: the
#: peaks it corrects sit at their MEASURED m/z, which the axis error itself moves
#: (an edge calibrant read +0.9 ppm fell just outside a theory-m/z scope and stayed
#: uncorrected). Sized for the corrections the batch step applies, which it caps
#: at MAX_CORRECTION_PPM; the probe itself searches PROBE_PPM wide.
SCOPE_SLACK_PPM = 20.0
#: the largest correction `peaky batch --mass-axis` applies: an Orbitrap axis further
#: off than this is a broken calibration to fix at the instrument, not a residual to
#: model. `peaky publish` matches a batch's merged m/z back to the server's peaks
#: within 5 ppm and does not undo the correction: a correction this size leaves ~1
#: ppm of that budget
MAX_CORRECTION_PPM = 4.0
#: the verdicts whose remedy is a correction of the m/z axis (`correction`); a
#: "+drifting" suffix keeps the axis remedy and adds the rolling centre's
APPLY_VERDICTS = ("axis_trend", "axis_offset")


def correction(v: dict) -> WaveFit | None:
    """The m/z correction a verdict prescribes, or None: an axis_trend's wave as
    fitted, an axis_offset as a constant (the calibrants' median offset, K = 0).
    A trend is applied ONLY inside its calibrants' m/z range (`in_scope`,
    unchanged outside): outside it its shape is unmeasured, and on the batch this
    was built for, both edges' neighbours sat back near 0 ppm -- holding the edge
    value would have been the larger error there. A constant has no shape to
    extrapolate and reaches every m/z: corrected in scope only, a flat offset
    left the rest of the batch at the full offset and a step at each edge."""
    base = str(v.get("verdict") or "").split("+")[0]
    w = v.get("wave")
    if base not in APPLY_VERDICTS or not w:
        return None
    fit = WaveFit(**{k: (tuple(x) if k in ("domain", "mz_range") else x) for k, x in w.items()})
    if base == "axis_offset":
        fit = WaveFit(coef=[float(v["median_offset_ppm"])], p=fit.p, domain=fit.domain,
                      mz_range=fit.mz_range, K=0, n=fit.n, n_clipped=fit.n_clipped,
                      resid_ppm=fit.raw_ppm, raw_ppm=fit.raw_ppm, loo_ppm=fit.loo_ppm,
                      span_ppm=0.0, share=0.0)
    return fit


def in_scope(fit: WaveFit, mz) -> np.ndarray:
    """Which m/z a correction reaches: a trend, its calibrants' m/z range widened
    by SCOPE_SLACK_PPM at each edge; a constant (K = 0), every finite m/z."""
    mz = np.asarray(mz, dtype=float)
    if int(fit.K) == 0:
        return np.isfinite(mz)
    lo, hi = (float(x) for x in fit.mz_range)
    return (mz >= lo * (1 - SCOPE_SLACK_PPM * 1e-6)) & (mz <= hi * (1 + SCOPE_SLACK_PPM * 1e-6))


def apply_correction(fit: WaveFit, mz) -> np.ndarray:
    """`mz` with the correction removed in scope (`in_scope`; a trend is a smooth
    polynomial, evaluated a few ppm past its edge) and unchanged outside it."""
    mz = np.asarray(mz, dtype=float)
    inside = in_scope(fit, mz)
    out = mz.copy()
    if inside.any():
        out[inside] = mz[inside] * (1 - fit.predict(mz[inside], extrapolate=True) * 1e-6)
    return out


def run(ts: pd.DataFrame, refs: pd.DataFrame, *, tol_ppm: float = 12.0, tof: bool = True,
        probe_ppm: float = PROBE_PPM, sample_col: str = "sample_item_id",
        time_col: str = "datetime_utc", per_instrument: bool = False
        ) -> tuple[pd.DataFrame, dict]:
    """Probe a batch time series (one row per picked peak: sample_item_id, mz,
    height, datetime_utc) against `refs`. Returns (table, verdict); see `verdict`
    for `per_instrument`."""
    from peaky.batch import traces as TR

    idx = TR.PeakIndex(ts, tol_ppm=tol_ppm, sample_col=sample_col)
    t = ts.drop_duplicates(sample_col).set_index(sample_col)[time_col]
    tt = pd.to_datetime(t.reindex(idx.sample_ids))
    hours = ((tt - tt.min()).dt.total_seconds() / 3600.0).to_numpy(dtype=float)
    hours = np.nan_to_num(hours, nan=0.0)
    table = twin_check(probe(idx, hours, refs, probe_ppm=probe_ppm, trace_ppm=tol_ppm))
    return table, verdict(table, idx.n_samples, tof=tof, per_instrument=per_instrument)
