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
    The test is the WIDTH: a seed whose members scatter as widely as a uniform
    fill of the window would is rejected, unless it recurs in half the spectra
    (that is a blended peak, flagged `fills_window`, to tier-cap, not to drop).
    A real ion dimmer than the window is wide -- sigma near W/2 -- and is
    genuinely indistinguishable from a fill there; that is the physical limit
    the 4-sigma membership rule exists to keep us away from.

    This used to be a Kolmogorov-Smirnov statistic of the members against a
    uniform ON THEIR OWN RANGE, chosen to be "free of the window and of the
    instrument's noise". Dividing by the observed range is what made it
    useless: it scales away the very width that separates an ion from a fill,
    so the statistic no longer depends on sigma at all (a sigma = 1.35 ppm ion
    and a true fill both read KS ~ 0.21-0.25) and KS * sqrt(n) becomes a
    disguised member count. Measured: at n = 10 a real ion was called a fill
    100 % of the time, at n = 40 still 94 %, and only above n ~ 86 did anything
    pass. On a 230-spectrum TOF batch that silently made the rule "keep an ion
    only if it occurs in ~37 % of spectra". It also read "cannot reject
    uniform" as "is a fill", which turns no evidence into a rejection.
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
FILL_FRAC = 0.74         # robust sd of a uniform fill of +-W, in units of W
FILL_SIGMA_FRAC = 0.70   # ... and a seed reads as a fill at this fraction of it. Chosen
                         # from both sides: a real ion (measured sigma median 1.35 ppm,
                         # p90 3.03) is never rejected, while >= 91 % of true fills with
                         # 30+ members are caught. Fills with ~12 members still leak
                         # through ~30 % of the time -- there is no width test with power
                         # that low, and MIN_MEMBERS is the floor that bounds it.
KEEP_OCC = 0.5           # ... unless it recurs in this share of spectra (blended, not noise)
# The EPISODE path. The seed floor asks an ion to recur across the batch, which a
# short plume never does: on a four-day mixed-reagent TOF batch, 41 of the ions a
# file cover found in >=2 files sit below a 5 % occurrence floor, and 24 of those
# have their detections packed into one stretch of the campaign. Peaky's per-file admission has admitted
# a peak by HEIGHT or by PERSISTENCE for as long as it has existed; the trace
# seeder only ever had the persistence half. This is the other half.
EPISODE_OCC = 0.01       # an episode may sit this far below the seed floor
EPISODE_MIN_MEMBERS = 5  # below this, contiguity in time cannot be judged at all
EPISODE_IQR = 0.05       # its detections' interquartile span, over the campaign's length.
                         # A run of k CONSECUTIVE spectra reads ~k/2n (0.011 at k=5,
                         # 0.024 at k=11, the most the episode path ever sees), while k
                         # detections scattered at random read 0.37-0.43. At this cut a
                         # scatter slips through 1.9 % of the time at k=5 and 0.01 % at
                         # k=8 -- the loosest point on the curve is the smallest k, which
                         # is why EPISODE_MIN_MEMBERS exists.
EPISODE_X_EDGE = 2.0     # and it must reach this multiple of the batch's noise edge
                         # somewhere. Brightness is the WEAKER half of this test -- of the
                         # 24 recoverable ions only 10 reach 3x and 5 reach 5x -- so it is
                         # set to exclude the floor, not to select plumes.
SAT_COOCCUR = 0.6        # share of a satellite's spectra that must hold its parent
SAT_PARENT_OCC = 0.10    # satellites are probed for traces at least this persistent
MEMBER_SIGMA_X = 4.0     # membership half-window = this x the reference ions' sigma ...
MEMBER_MIN_PPM = 12.0    # ... never below this (the validated TOF cap) UNLESS the
                         # instrument's own cell is narrower ...
                         # ... and never above half the dedup cell, so a trace's
                         # members can never span more than one observable
HI_RES_CELL_PPM = 6.0    # a dedup cell this narrow means a high-resolution instrument:
                         # trace-first is a TOF remedy, and says so rather than pretend
# Sanity band for the supplied resolving power, read off the spectra themselves.
# Two reported maxima cannot be closer than the summed profile is bimodal --
# 2.36 HWHM at equal heights, more when unequal -- so the smallest spacings a
# picker reports bound the peak width. The bound is only a bound, because how
# far into a flank a picker will call a maximum is the PICKER's property, not
# the instrument's: measured at the 1st percentile of the within-spectrum
# nearest-neighbour spacing, a TOF picker reports down to 1.00 HWHM (it calls
# shoulders, below the bimodality limit) and an Orbitrap picker to 2.88 (it does
# not). Three times apart, so this cannot SET the resolving power -- but it
# catches the error that matters, a value from the wrong instrument, which is
# twenty times out.
SPACING_MIN_RATIO = 0.3   # d_min / HWHM below this: the claimed peaks are far wider
SPACING_MAX_RATIO = 10.0  # ... above this: far narrower ... than the spectra show
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
              "fills_window", "fill_ratio", "episode_span", "height_med", "kind", "parent_peak", "parent_cooccur",
              "resolvability", "sep_hwhm", "d_crit_hwhm"]


# ------------------------------------------------------------------ resolution
def spacing_bound(ts_peaks: pd.DataFrame, resolving_power, *,
                  sample_col: str = "sample_item_id", mz_col: str = "mz",
                  q: float = 0.01) -> dict:
    """What the spectra themselves say about the peak width, against the
    resolving power the caller supplied.

    `d_min` is the `q` quantile of the within-spectrum nearest-neighbour
    spacing in ppm -- the closest two maxima this picker will report. Returned
    with the supplied HWHM, their ratio, and `ok` (the ratio inside the
    calibrated band). A bound, never a measurement: see SPACING_MIN_RATIO."""
    out = {"d_min_ppm": None, "hwhm_ppm": None, "ratio": None, "ok": True, "q": q}
    if ts_peaks is None or mz_col not in ts_peaks.columns or sample_col not in ts_peaks.columns:
        return out
    d = ts_peaks[[sample_col, mz_col]].dropna().sort_values([sample_col, mz_col])
    gap = d.groupby(sample_col)[mz_col].diff()
    ppm = (gap / d[mz_col] * 1e6).replace([np.inf, -np.inf], np.nan).dropna()
    ppm = ppm[ppm > 0]
    if len(ppm) < 100:
        return out
    d_min = float(np.quantile(ppm, q))
    hw = hwhm(200.0, resolving_power) / 200.0 * 1e6          # HWHM in ppm (constant R)
    ratio = d_min / hw if hw > 0 else np.nan
    out.update(d_min_ppm=d_min, hwhm_ppm=hw, ratio=float(ratio),
               ok=bool(SPACING_MIN_RATIO <= ratio <= SPACING_MAX_RATIO))
    return out


@dataclass(frozen=True)
class Resolution:
    """How wide a peak is, as a function of m/z: FWHM(m) = coef * m ** exponent.

    A single resolving power is a model, not a fact -- it says FWHM is
    proportional to m, i.e. R is the same at every mass. That holds on a TOF
    (measured exponent 1.08 on a field batch) and fails on an Orbitrap, whose R
    falls as m^-1/2: measured across one spectrum, 204 000 at m/z 152 and
    96 000 at m/z 558, a factor of two that a scalar cannot express and that
    would size the dedup cell wrong at both ends.

    `from_r(R)` is the scalar model, kept because a caller who knows their
    instrument may say so; `measure` (batch.tracefirst.measure_resolution) fits
    both terms from the raw profile and reports which it found."""

    coef: float            # FWHM(m) = coef * m ** exponent + offset, in Th
    exponent: float = 1.0  # 1.0 = constant R (TOF); 1.5 = R ~ m^-1/2 (Orbitrap)
    offset: float = 0.0    # the TOF rational form's constant width term
    n_peaks: int = 0       # peaks the fit used (0 = declared, not measured)
    r_spread: tuple = ()   # (p25, p75) of the per-peak R, when measured
    source: str = "declared"

    @classmethod
    def from_r(cls, resolving_power: float) -> "Resolution":
        """The constant-R model: FWHM(m) = m / R."""
        r = float(resolving_power)
        if not np.isfinite(r) or r <= 0:
            raise ValueError(f"resolving power must be a positive number, got {resolving_power!r}")
        return cls(coef=1.0 / r, exponent=1.0)

    @classmethod
    def coerce(cls, value) -> "Resolution":
        return value if isinstance(value, cls) else cls.from_r(value)

    @classmethod
    def from_mascope(cls, coefficients, instrument_type: str = "") -> "Resolution":
        """Mascope's own instrument resolution function, as its processors fit it
        (`mascope_signal.instrument_func.fit`): a TOF gets the rational polynomial
        R(m) = m / (a*m + b), so FWHM = a*m + b; an Orbitrap gets R(m) = a / sqrt(m),
        so FWHM = m**1.5 / a. Both are exactly representable here, so the day the
        server exposes them to a service token they drop straight in -- today the
        /api/instrument_configs route answers only a user session."""
        c = [float(x) for x in coefficients]
        if len(c) >= 2 and "orbi" not in instrument_type.lower():
            return cls(coef=c[0], exponent=1.0, offset=c[1], source="mascope")
        if len(c) == 1:
            return cls(coef=1.0 / c[0], exponent=1.5, source="mascope")
        raise ValueError(f"unrecognised resolution-function coefficients: {coefficients!r}")

    def fwhm(self, mz: float) -> float:
        """Peak width at half maximum, in Th."""
        return float(self.coef) * float(mz) ** float(self.exponent) + float(self.offset)

    def hwhm(self, mz: float) -> float:
        return 0.5 * self.fwhm(mz)

    def r_at(self, mz: float) -> float:
        """The resolving power AT this mass -- constant only when exponent is 1."""
        return float(mz) / self.fwhm(mz)

    def dedup_ppm(self, mz: float) -> float:
        """The dedup half-window in ppm at `mz`: 0.4 HWHM, the fit floor."""
        return FIT_FLOOR_HWHM * self.hwhm(mz) / float(mz) * 1e6

    @property
    def is_tof(self) -> bool:
        """Which dispersion the analyser works in, read off the width model.

        A TOF's width grows about linearly with mass (exponent ~1, constant R);
        an Orbitrap's as m^1.5 (R ~ m^-1/2). That is the same split the mass
        wave needs -- flight time goes as (m/z)^+1/2, frequency as (m/z)^-1/2 --
        so the measured exponent decides the basis, and nothing has to be
        declared twice. 1.3 is the midpoint of the two, far from both."""
        return float(self.exponent) < 1.3

    def describe(self) -> str:
        at = f"R = {self.r_at(200.0):.0f} at m/z 200"
        if abs(self.exponent - 1.0) > 0.15:
            at += f", {self.r_at(600.0):.0f} at m/z 600 (FWHM ~ m^{self.exponent:.2f})"
        if self.n_peaks:
            at += f" [fitted on {self.n_peaks} profile peaks"
            if self.r_spread:
                at += f", per-peak IQR {self.r_spread[0]:.0f}-{self.r_spread[1]:.0f}"
            at += "]"
        return at

    def as_dict(self) -> dict:
        return {"coef": float(self.coef), "exponent": float(self.exponent),
                "offset": float(self.offset),
                "r_at_200": float(self.r_at(200.0)), "r_at_600": float(self.r_at(600.0)),
                "n_peaks": int(self.n_peaks), "r_spread": list(self.r_spread),
                "source": self.source}


def _profile_fwhm(x, y):
    """FWHM of the tallest peak in a profile segment, by half-maximum crossings."""
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    if len(x) < 8 or not np.isfinite(y).any():
        return None
    k = int(np.argmax(y)); half = y[k] / 2.0
    left = np.where(y[:k] <= half)[0]
    right = np.where(y[k:] <= half)[0]
    if not len(left) or not len(right):
        return None                                  # apex on the segment's edge
    li, ri = int(left[-1]), int(k + right[0])
    xl = np.interp(half, [y[li], y[li + 1]], [x[li], x[li + 1]])
    xr = np.interp(half, [y[ri], y[ri - 1]], [x[ri], x[ri - 1]])
    return float(xr - xl) if xr > xl else None


def measure_resolution(client, sample_id: str, peaks: pd.DataFrame | None = None, *,
                       fetch_peaks=None, get_spectrum=None, n_bands: int = 5,
                       per_band: int = 2, iso_ppm=(400.0, 200.0, 100.0),
                       window_ppm: float = 600.0, min_peaks: int = 3,
                       log=print) -> "Resolution | None":
    """Measure the peak-width model from the sample's RAW PROFILE.

    Picks the brightest well-isolated peak in each of `n_bands` log-spaced mass
    bands, fetches a narrow profile window around each, measures the FWHM at
    half-maximum crossings, and fits log FWHM against log m/z -- so the width
    model and the instrument class come out of the data instead of a flag.
    Returns None (and says why) when too few peaks yield a usable profile.

    On a TOF the profile is the instrument's own. On an Orbitrap the served
    profile is a Gaussian rendering built from each centroid's stored
    resolution, so this reads that recorded resolution back -- right for sizing
    a window, and not evidence about the picker.
    """
    from peaky.io import io_mascope as IO
    fetch_peaks = fetch_peaks or IO.fetch_peaks
    get_spectrum = get_spectrum or (lambda sid, lo, hi: client.samples.get_spectrum(
        sid, mz_min=lo, mz_max=hi))
    try:
        pk = peaks if peaks is not None else fetch_peaks(client, sample_id)
    except Exception as exc:                     # a measurement may fail; a run may not
        log(f"[resolution] could not read {sample_id}'s peaks ({type(exc).__name__})")
        return None
    if pk is None or not len(pk) or "mz" not in pk.columns:
        log("[resolution] no peaks to measure from")
        return None
    d = pk.drop_duplicates("peak_id") if "peak_id" in pk.columns else pk
    d = d[["mz", "height"]].dropna().sort_values("mz")
    if len(d) < 20:
        log("[resolution] too few peaks to measure from")
        return None
    mz = d["mz"].to_numpy(float); h = d["height"].to_numpy(float)
    gapL = np.r_[np.inf, np.diff(mz)] / mz * 1e6
    gapR = np.r_[np.diff(mz), np.inf] / mz * 1e6
    lo, hi = np.quantile(mz, 0.02), np.quantile(mz, 0.98)
    edges = np.exp(np.linspace(np.log(max(lo, 1.0)), np.log(max(hi, lo * 1.1)), n_bands + 1))
    picks: list[float] = []
    for a, b in zip(edges[:-1], edges[1:]):
        band = (mz >= a) & (mz < b)
        for iso in iso_ppm:
            sel = band & (gapL > iso) & (gapR > iso)
            if sel.any():
                order = np.argsort(-np.where(sel, h, -np.inf))[:per_band]
                picks += [float(mz[i]) for i in order if sel[i]]
                break
    rows = []
    for m in picks:
        w = m * window_ppm * 1e-6
        try:
            sp = get_spectrum(sample_id, m - w, m + w)
        except Exception:
            continue
        if sp is None or not len(sp):
            continue
        f = _profile_fwhm(sp["mz"].to_numpy(), sp["intensity"].to_numpy())
        if f and f > 0:
            rows.append((m, f))
    if len(rows) < min_peaks:
        log(f"[resolution] only {len(rows)} of {len(picks)} probes gave a usable profile "
            f"(need {min_peaks}); pass --resolving-power instead")
        return None
    a_mz = np.array([r[0] for r in rows]); a_f = np.array([r[1] for r in rows])
    exponent, intercept = np.polyfit(np.log(a_mz), np.log(a_f), 1)
    exponent = float(np.clip(exponent, 0.5, 2.5))
    coef = float(np.exp(intercept))
    per_r = a_mz / a_f
    res = Resolution(coef=coef, exponent=exponent, n_peaks=len(rows),
                     r_spread=(float(np.quantile(per_r, 0.25)), float(np.quantile(per_r, 0.75))),
                     source="measured")
    log(f"[resolution] measured from the raw profile of {len(rows)} isolated peaks "
        f"(m/z {a_mz.min():.0f}-{a_mz.max():.0f}): {res.describe()}")
    return res


def hwhm(mz: float, resolving_power) -> float:
    """Half width at half maximum (Th) at `mz`. `resolving_power` is a number
    (constant R) or a `Resolution` model."""
    return Resolution.coerce(resolving_power).hwhm(mz)


def dedup_ppm(mz: float, resolving_power) -> float:
    """The dedup half-window at `mz`: 0.4 HWHM in ppm."""
    return Resolution.coerce(resolving_power).dedup_ppm(mz)


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


def stamp_resolvability(traces: pd.DataFrame, resolving_power) -> pd.DataFrame:
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
def episode_span(positions, n_spectra: int) -> float:
    """How tightly a trace's detections sit together in the campaign: the
    interquartile span of the spectra it appears in, over the number of spectra.

    ~0 for one contiguous episode, 0.37-0.48 for the same number of detections
    scattered at random. `positions` are the members' places in TIME order, not
    their spectrum ids. 1.0 (i.e. "scattered") for fewer than three points, so a
    caller that forgets the member floor fails closed."""
    p = np.asarray(positions, dtype=float)
    if len(p) < 3 or not n_spectra:
        return 1.0
    return float((np.percentile(p, 75) - np.percentile(p, 25)) / float(n_spectra))


def fill_ratio(offsets, tol_ppm: float) -> float:
    """How much of its window a trace's members occupy: their robust sd over the
    robust sd a uniform fill of +-`tol_ppm` would give. ~1 for a fill, small for
    a real ion (0.18 for the measured median sigma of 1.35 ppm in a +-10.4 ppm
    window). 0 for fewer than three points.

    The comparison is against the WINDOW, deliberately. An earlier version
    normalised by the members' own range to be scale-free, which divided out the
    width and left a statistic that could not tell an ion from a fill at all
    (see the module note)."""
    d = np.asarray(offsets, dtype=float)
    if len(d) < 3:
        return 0.0
    ref = FILL_FRAC * float(tol_ppm)
    if not np.isfinite(ref) or ref <= 0:
        return 0.0
    return float(1.4826 * np.median(np.abs(d - np.median(d))) / ref)


def build_traces(index, hours, *, resolving_power: float, tol_ppm: float,
                 seed_occ: float = SEED_OCC, area_index=None, episodes: bool = True,
                 log=print) -> pd.DataFrame:
    """One row per trace: seeds (peaks recurring in >= seed_occ of spectra,
    brightest first, each consuming its dedup cell) and the satellite positions
    of the persistent ones. See the module note for the gates."""
    R = Resolution.coerce(resolving_power)
    tol = float(tol_ppm)
    dedup = R.dedup_ppm
    fill = FILL_FRAC * tol
    occ = index.occurrence()
    n = index.n_samples
    order = np.lexsort((-np.nan_to_num(index.height), -occ))
    # where each spectrum falls in TIME, and how bright a peak has to be to be
    # more than the picker's floor -- both only used by the episode path below
    time_rank = np.empty(n, dtype=float)
    time_rank[np.argsort(np.asarray(hours, dtype=float), kind="mergesort")] = np.arange(n)
    from peaky.assignment.passes.config import noise_edge as _edge
    edge = _edge(index.height)
    epi_floor = (EPISODE_X_EDGE * float(edge)) if edge else 0.0
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
        fr = fill_ratio((index.mz[mem] - c) / c * 1e6, tol)
        fills = bool(fr >= FILL_SIGMA_FRAC)       # as wide as a fill of the window
        cooc = np.nan
        if parent is not None:
            cooc = float(np.isin(index.sample[mem], list(spectra_of[parent])).mean())
        span = episode_span(time_rank[index.sample[mem]], n) if kind == "episode" else np.nan
        bad = (kind == "seed" and fills and occ_c < KEEP_OCC) \
            or (kind == "episode" and not (span <= EPISODE_IQR
                                           and np.nanmax(index.height[mem]) >= epi_floor)) \
            or (kind not in ("seed", "episode") and cooc < SAT_COOCCUR)
        if bad:
            rejected.append((kind, q, cooc, fr))
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
                         fill_ratio=float(fr), episode_span=float(span),
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
    # EPISODE PASS: below the occurrence floor, where a short plume lives. Ordered
    # by height, because the floor is the one thing brightness is good for here,
    # and run AFTER the seeds so an episode can never take a persistent ion's cell.
    n_before_epi = len(rows)
    if episodes:
        epi_order = [i for i in np.lexsort((-np.nan_to_num(index.height), occ))
                     if EPISODE_OCC <= occ[i] < seed_occ]
        for i in epi_order:
            if consumed[i]:
                continue
            take(index.mean_shift(float(index.mz[i]), tol_ppm=tol, max_drift_ppm=tol),
                 "episode", EPISODE_MIN_MEMBERS)
    n_epi = len(rows) - n_before_epi
    rej_seed = [q for k, q, _, _ in rejected if k == "seed"]
    log(f"[traces] {R.describe()}: dedup {FIT_FLOOR_HWHM} HWHM = {dedup(200.0):.1f} ppm; membership "
        f"+-{tol:g} ppm; a uniform fill reads {fill:.2f} ppm; gate rejects a seed whose "
        f"members scatter >= {FILL_SIGMA_FRAC:.0%} of that ({FILL_SIGMA_FRAC * fill:.2f} ppm) "
        f"unless occurrence >= {KEEP_OCC:.0%}")
    log(f"[traces] {n_seed} seed traces from peaks recurring in >= {seed_occ:.0%} of {n} spectra; "
        f"{len(rej_seed)} seed positions rejected as fills"
        + (f" (median residual {np.median(rej_seed):.2f} ppm)" if rej_seed else ""))
    log(f"[traces] {sum(r['fills_window'] for r in rows)} kept seeds fill their window but recur "
        f"in >= {KEEP_OCC:.0%} (`fills_window`: blended, not noise); "
        f"{sum(r['centre_scheme'] == 'rolling' for r in rows)} of {n_seed} roll")
    if episodes:
        rej_epi = sum(1 for k, *_ in rejected if k == "episode")
        log(f"[traces] + {n_epi} episode traces below the {seed_occ:.0%} floor: detections packed "
            f"into <= {EPISODE_IQR:.0%} of the campaign (scattered reads 0.37-0.43) and reaching "
            f"{EPISODE_X_EDGE:g}x the noise edge; {rej_epi} candidate positions rejected")
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
    resolution: "Resolution"
    n_spectra: int
    picker_floor_cps: float | None = None
    spacing: dict | None = None
    notes: list = field(default_factory=list)

    def summary(self) -> dict:
        t = self.traces
        return {"n_traces": int(len(t)), "n_seeds": int((t["kind"] == "seed").sum()) if len(t) else 0,
                "n_satellites": int(t["kind"].astype(str).str.startswith("sat").sum()) if len(t) else 0,
                "n_spectra": int(self.n_spectra), "membership_ppm": float(self.tol_ppm),
                "resolution": self.resolution.as_dict(),
                "resolving_power": float(self.resolution.r_at(200.0)),
                "dedup_ppm": float(self.resolution.dedup_ppm(200.0)),
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
                "spacing_bound": self.spacing,
                "notes": list(self.notes)}


def build_trace_sample(ts_peaks: pd.DataFrame, *, sample_id: str, reagent: str,
                       resolving_power, tol_ppm: float | None = None,
                       seed_occ: float = SEED_OCC, sample_col: str = "sample_item_id",
                       time_col: str = "datetime_utc", log=print) -> TraceSample:
    """Traces -> mass-qc -> membership -> gate -> wave -> one synthetic sample."""
    notes: list[str] = []
    res = Resolution.coerce(resolving_power)      # a number, or a measured model
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
        # the wave's basis is the analyser's dispersion, and the width model
        # already knows which one this is -- an Orbitrap fitted in flight time
        # is a wave fitted against the wrong variable
        qc = MQ.verdict(table, idx.n_samples, tof=res.is_tof)
        MQ.report(table, qc, log=log)
        sig = qc.get("median_sigma_ppm")
    cell = res.dedup_ppm(200.0)
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
    traces = build_traces(idx, hours, resolving_power=res, tol_ppm=tol,
                          seed_occ=seed_occ, area_index=area_idx, log=log)
    if len(traces):
        traces = stamp_resolvability(traces, res)
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
    sb = spacing_bound(ts_peaks, res, sample_col=sample_col)
    if sb["ratio"] is not None:
        line = (f"{res.describe()} puts HWHM at {sb['hwhm_ppm']:.2f} ppm; the closest two "
                f"maxima this picker reports are {sb['d_min_ppm']:.2f} ppm apart "
                f"({sb['ratio']:.2f} HWHM)")
        if sb["ok"]:
            log(f"[traces] {line} -- consistent")
        else:
            msg = (line + ". That is outside the band both measured instruments sit in "
                   f"({SPACING_MIN_RATIO}-{SPACING_MAX_RATIO} HWHM): check --resolving-power, it "
                   "looks like a value from a different instrument")
            notes.append(msg)
            log(f"[traces] WARNING: {msg}")
    if cell <= HI_RES_CELL_PPM:
        msg = (f"{res.describe()} puts the resolution floor at {cell:.2f} ppm: on a "
               f"high-resolution instrument a per-file mass is already good to a fraction of a "
               f"ppm, so the per-file route has little to lose to. Trace-first is a TOF remedy; "
               f"it will run, but expect a reorganisation, not a gain")
        notes.append(msg)
        log(f"[traces] NOTE: {msg}")
    log(f"[traces] {len(traces)} traces -> synthetic sample {sample_id!r}")
    return TraceSample(sample_id=sample_id, peaks=synthetic_sample(traces, sample_id), traces=traces,
                       occurrence=occurrence_table(traces, tol, idx.n_samples), qc=qc, tol_ppm=tol,
                       resolution=res, n_spectra=int(idx.n_samples),
                       picker_floor_cps=floor, spacing=sb, notes=notes)


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
