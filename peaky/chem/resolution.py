"""The instrument's peak-width model, and what it says about two neighbouring peaks.

FWHM(m) = coef * m ** exponent + offset. A single resolving power is a model, not
a fact -- it says the width is proportional to m, i.e. R is the same at every
mass. That holds on a TOF (measured exponent ~1.0 on field batches) and fails on
an Orbitrap, whose R falls as m^-1/2 (exponent ~1.5): measured across one
spectrum, 204 000 at m/z 152 and 96 000 at m/z 558, a factor of two a scalar
cannot express. `batch.tracefirst.measure_resolution` fits both terms from the
raw profile of isolated peaks; `Resolution.from_r` is the declared scalar.

Two picked peaks a distance d apart, in units of the HWHM at that mass, are:

* `unresolvable` below FIT_FLOOR_HWHM (0.4 HWHM, Cubison & Jimenez 2015): one
  observable, whatever the fit;
* `blended` below the bimodality threshold for their height ratio
  (`d_crit_hwhm`): a fit could separate them, a local-maximum picker cannot, and
  the reported centroid is displaced toward the neighbour -- the mass is not the
  ion's own;
* `resolved` above it; `isolated` when no picked peak sits within
  NEIGHBOUR_WINDOW_HWHM at all.

This is a leaf module (numpy only) so the assignment stages and the batch code
can both read it without an import cycle; `batch.tracefirst` re-exports the
names it used to own.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FIT_FLOOR_HWHM = 0.4        # Cubison & Jimenez (2015): below it two peaks are one observable
NEIGHBOUR_WINDOW_HWHM = 8.0  # the nearest picked peak within this many HWHM decides the class


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
    def from_dict(cls, d: dict) -> "Resolution":
        """The inverse of `as_dict`: a model read back from a run's summary."""
        return cls(coef=float(d["coef"]), exponent=float(d.get("exponent", 1.0)),
                   offset=float(d.get("offset", 0.0)), n_peaks=int(d.get("n_peaks", 0) or 0),
                   r_spread=tuple(d.get("r_spread") or ()), source=str(d.get("source", "recorded")))

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


def hwhm(mz: float, resolving_power) -> float:
    """Half width at half maximum (Th) at `mz`. `resolving_power` is a number
    (constant R) or a `Resolution` model."""
    return Resolution.coerce(resolving_power).hwhm(mz)


def dedup_ppm(mz: float, resolving_power) -> float:
    """The dedup half-window at `mz`: 0.4 HWHM in ppm."""
    return Resolution.coerce(resolving_power).dedup_ppm(mz)


_HWHM_IN_SIGMA = 1.1774396  # HWHM = sqrt(2 ln 2) sigma, so d[HWHM] = d[sigma] / this
# the bimodality threshold in sigma, at log10(height ratio): 2 sigma for equal heights
_LOGR = np.array([0.0, -0.30103, -0.69897, -1.0, -1.30103, -1.69897, -2.0, -2.30103, -2.69897, -3.0])
_DSIG = np.array([2.000, 2.628, 3.079, 3.354, 3.598, 3.888, 4.088, 4.277, 4.511, 4.679])


def d_crit_hwhm(ratio: float) -> float:
    """Separation (HWHM) at which two Gaussians of height ratio `ratio` (<= 1)
    become bimodal -- what a local-maximum picker needs to report two peaks.
    Equal heights: 2 sigma = 1.70 HWHM (0.85 FWHM); 1:50, 3.89 sigma = 1.65 FWHM."""
    r = float(min(max(ratio, 1e-12), 1.0))
    lr = np.log10(r)
    if lr >= _LOGR[0]:
        d = _DSIG[0]
    elif lr <= _LOGR[-1]:
        slope = (_DSIG[-1] - _DSIG[-2]) / (_LOGR[-1] - _LOGR[-2])
        d = _DSIG[-1] + slope * (lr - _LOGR[-1])
    else:
        d = float(np.interp(lr, _LOGR[::-1], _DSIG[::-1]))
    return d / _HWHM_IN_SIGMA


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


def nearest_neighbour_classes(mz, height, resolving_power, *,
                              window_hwhm: float = NEIGHBOUR_WINDOW_HWHM) -> dict:
    """Every peak's separability from its NEAREST neighbour in the same list,
    in the input order. `mz` / `height` are array-likes of the picked peaks
    (any order; a non-finite height reads as 0); the neighbour search covers
    `window_hwhm` HWHM either side. Returns {'resolvability': list[str],
    'sep_hwhm': ndarray, 'd_crit_hwhm': ndarray, 'neighbour': ndarray of input
    positions (-1 when isolated)}. A FLAG (and a tier / level input), never a
    filter."""
    res = Resolution.coerce(resolving_power)
    m = np.asarray(mz, dtype=float)
    h = np.nan_to_num(np.asarray(height, dtype=float), nan=0.0)
    n = len(m)
    cls = ["isolated"] * n
    sep = np.full(n, np.nan)
    dcr = np.full(n, np.nan)
    nb = np.full(n, -1, dtype=int)
    if n < 2:
        return {"resolvability": cls, "sep_hwhm": sep, "d_crit_hwhm": dcr, "neighbour": nb}
    order = np.argsort(m, kind="mergesort")
    ms = m[order]
    for k in range(n):
        i = order[k]
        hw = res.hwhm(ms[k])
        lo = int(np.searchsorted(ms, ms[k] - window_hwhm * hw))
        hi = int(np.searchsorted(ms, ms[k] + window_hwhm * hw, side="right"))
        best = None
        for kk in range(lo, hi):
            if kk == k:
                continue
            d = abs(ms[kk] - ms[k])
            if best is None or d < best[0]:
                best = (d, kk)
        if best is None:
            continue
        j = order[best[1]]
        c = classify_pair(m[i], h[i], m[j], h[j], res)
        cls[i] = c["resolvability"]
        sep[i] = c["sep_hwhm"]
        dcr[i] = c["d_crit_hwhm"]
        nb[i] = j
    return {"resolvability": cls, "sep_hwhm": sep, "d_crit_hwhm": dcr, "neighbour": nb}
