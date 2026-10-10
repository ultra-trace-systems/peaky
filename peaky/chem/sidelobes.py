"""Same-spectrum side lobes of an Orbitrap peak list: weak picked "peaks" one to
five line widths beside a much brighter line, which are not lines of the profile.

A peak list processed from Orbitrap profiles can carry entries the profile does
not hold: a weak peak beside a line at least ~50x brighter, at an offset that
scales with the PARENT'S WIDTH, not with mass (a negative lobe at -2.35 +/- 0.1
parent FWHM at every m/z of a 50-400 range, a positive one at +2.3 to +3.9),
and narrower than the instrument makes a line (median own width 0.44x the
file's typical, against 1.1x for real lines in the same band). Checked
against the raw profile on one 100-microscan batch, such peaks in the band were
absent or 5-20x too high in ~99 % of cases. Given formulas, they read as exotic
ions with nothing behind them.

`sidelobe_parents` flags them in ONE peak list -- never across files: a peak i
is a lobe of the brightest peak j of the same list with

    height_j >= ratio * height_i  and  band[0] <= |mz_i - mz_j| / FWHM_eff(j) <= band[1]

AND an artifact signature: its own width < `narrow` x the file's typical width,
OR its offset in the negative-lobe window (NEG_LOBE_FWHM), OR a mirror lobe on
the other side of the same parent. The band is in units of the parent's
OBSERVED width, self-calibrated per file (`width_calibration`): the parent's
own area / (1.0645 * height) when it is plausible, else the width model scaled
by the file's median observed/model ratio over its bright peaks -- so neither a
mis-scaled model nor the server's area convention moves the band. With too few
bright peaks to measure that ratio (a measured or mascope model; a declared one
is skipped) the "narrow" signature is not used: it would be read against the
raw model. A line whose offset is an isotope fine-structure spacing from a
parent that is itself an isotopologue line of an even brighter peak is exempt
-- neither flagged nor counted as another candidate's mirror lobe: a bright
H-rich ion's 2H line sits +2.92 mDa above its 13C line, right where a positive
lobe would, and a C17+ single-N ion's 15N line -6.32 mDa below it, in the
negative-lobe window at m/z ~450-500.

A candidate of a line's own width (no 'narrow' signature) that sits at an
isotopologue offset (13C, 15N, 29Si, 18O, 34S, 37Cl, 81Br; `ISOTOPOLOGUE_LINES`)
of ANY brighter line of the same list, within max(1 ppm, 0.3 x that line's
FWHM_eff), at a ratio to it that an ion of that mass can make
(`isotopologue_link`), is exempt too: the 13C line of a neighbouring ion can sit
two widths below a much brighter line (m/z 174.121, the 13C line of 173.117 at
0.10x, beside a line 80-120x brighter at 174.124). The brighter line must not be
in the lobe band of a brighter line itself -- a lobe of an ion's 13C line sits
one 13C spacing above the same lobe of the ion, at the 13C ratio, and stays
flagged.

The ledger step is assignment/sidelobe_guard.py; this module reads only numpy
and the isotope constants (chem.isotopes).
"""
from __future__ import annotations

import numpy as np

from peaky.chem import isotopes as ISO

#: a Gaussian's area per (height x FWHM): FWHM = area / (GAUSS_AREA * height)
GAUSS_AREA = 1.0645
#: the parent must be at least this many times brighter than the lobe
SIDELOBE_RATIO = 50.0
#: the lobe's distance from its parent, in parent FWHM_eff
SIDELOBE_BAND_FWHM = (1.0, 5.0)
#: own width / the file's typical width below which a peak is narrower than a line
SIDELOBE_NARROW = 0.7
#: the negative lobe's window, in parent FWHM_eff (measured at -2.35 +/- 0.1)
NEG_LOBE_FWHM = (-2.6, -2.1)
#: a parent's own observed width is used when its observed/model ratio is within
#: this multiple of the file's median ratio; else model x the median
PARENT_OWN_WIDTH = (0.6, 1.6)
#: the file's median observed/model width ratio must lie here, else the model,
#: the widths or the area convention are not what the rule assumes: skipped
CAL_RATIO_RANGE = (0.5, 2.0)
#: bright peaks (>= this x the file's noise edge, its p1 height) calibrate the widths
BRIGHT_X_EDGE = 100.0
#: ... and at least this many are needed to call the calibration measured
MIN_CAL_PEAKS = 10
#: a declared (scalar) model whose observed/model ratio trends with m/z by more
#: than this log-log slope cannot be calibrated by one scale factor
MAX_DECLARED_SLOPE = 0.25
#: the artifact signatures, all of them when the file's widths calibrate the model
SIGNATURES = ("narrow", "neg-lobe", "mirror")
#: ... and the geometric ones only when they do not (too few bright peaks)
UNCALIBRATED_SIGNATURES = ("neg-lobe", "mirror")
#: fine-structure offsets match within this many parent FWHM_eff ...
FINE_TOL_FWHM = 0.3
#: ... and the parent's own isotopologue link to its brighter peak within this
FINE_LINK_TOL_FWHM = 0.5

#: exact spacings (AME2020): the fine-structure offsets below are mDa-scale
#: DIFFERENCES of them, so 2H / 15N / 17O / 18O are kept here to 1e-10 Da
_D13C = ISO.D_13C_EXACT
_D2H = 1.0062767461
_D15N = 0.9970348941
_D17O = 1.0042171369
_D18O = 2.0042449933
#: (name, the parent's shift from its brighter mono line, the exempt line's shift)
FINE_STRUCTURE = (
    ("2H-13C", _D13C, _D2H),                   # +2.922 mDa above the 13C line
    ("17O-13C", _D13C, _D17O),                 # +0.862 mDa above the 13C line
    ("13C15N-18O", _D18O, _D13C + _D15N),      # -3.855 mDa below the 18O line
    ("18O-13C15N", _D13C + _D15N, _D18O),      # +3.855 mDa above the 13C15N line
    # a single-N ion's 15N line sits -6.32 mDa below its 13C line at
    # 13C/15N = 2.95 nC/nN (>= 50 from C17 on): ~-2.4 FWHM at m/z 460-490 on an
    # Orbitrap, inside the negative-lobe window. Its reverse, for an N-rich ion
    # whose 15N line is the brighter: the 13C line +6.32 mDa above it.
    ("15N-13C", _D13C, _D15N),                 # -6.320 mDa below the 13C line
    ("13C-15N", _D15N, _D13C),                 # +6.320 mDa above the 15N line
)

#: (label, spacing above the brighter line, per-atom ratio, the light atom's
#: mass) of the isotopologue lines that exempt a candidate sitting on one of a
#: brighter line of the list
ISOTOPOLOGUE_LINES = (
    ("13C", _D13C, ISO.R_13C_PER_C, 12.0),
    ("15N", _D15N, ISO.R_15N_PER_N, 14.003074),
    ("29Si", ISO.D_29SI, ISO.R_29SI_PER_SI, 27.976927),
    ("18O", _D18O, ISO.R_18O_PER_O, 15.994915),
    ("34S", ISO.D_34S, ISO.R_34S_PER_S, 31.972071),
    ("37Cl", ISO.D_37CL, ISO.R_37CL_PER_CL, 34.968853),
    ("81Br", ISO.D_81BR, ISO.R_81BR_PER_BR, 78.918338),
)
#: the 13C line's plausible share of its brighter line (~0.4 to ~47 carbons);
#: the other lines' bands are this one scaled by their per-atom ratio / 13C's
ISO_LINK_RATIO_13C = (0.004, 0.5)
#: ... and never above this multiple of the share the most atoms the brighter
#: line's mass holds would make (m / the atom's mass): a 13C line at 0.3x of an
#: m/z 168 line would need 28 carbons
ISO_LINK_MASS_CAP_X = 1.5
#: the offset tolerance: the exact-offset window of the isotope tests
#: (satellites.element_window_ppm, max(1 ppm, 4 sigma)) at its 1 ppm floor, since
#: the guard runs before the file is calibrated. A wider, width-scaled tolerance
#: (0.3 FWHM, ~2 ppm at m/z 170-230) was measured on a positive-mode Orbitrap
#: batch: the 8 peaks it spared over this one were lobes (a -2.7 mDa lobe
#: population of a bright line, a "13C" line at 3.4x its prediction, a "29Si"
#: line of a Si-free ion), with the real 13C line a separate peak inside the
#: window; a negative-mode batch flagged the same peaks under both.
ISO_LINK_PPM = 1.0


def model_fwhm(model, mz) -> np.ndarray:
    """The width model's FWHM (Th) at every m/z (chem.resolution.Resolution)."""
    m = np.asarray(mz, dtype=float)
    return float(model.coef) * np.power(m, float(model.exponent)) + float(model.offset)


def observed_fwhm(height, area) -> np.ndarray:
    """Each peak's own FWHM from its area and height (a Gaussian's), NaN where
    either is missing or not positive."""
    h = np.asarray(height, dtype=float)
    a = np.asarray(area, dtype=float)
    ok = np.isfinite(h) & np.isfinite(a) & (h > 0) & (a > 0)
    out = np.full(h.shape, np.nan)
    out[ok] = a[ok] / (GAUSS_AREA * h[ok])
    return out


def width_calibration(mz, height, area, model, *, bright_x_edge: float = BRIGHT_X_EDGE,
                      min_peaks: int = MIN_CAL_PEAKS) -> dict:
    """The file's observed/model width ratio over its bright peaks.

    Returns {'ratio': median ratio (1.0 when not measured), 'n': bright peaks with a
    width, 'measured': bool, 'slope': log-log trend of the ratio with m/z (NaN
    under 3 peaks), 'edge': the noise edge}."""
    m = np.asarray(mz, dtype=float)
    h = np.asarray(height, dtype=float)
    fin = np.isfinite(h) & (h > 0)
    edge = float(np.percentile(h[fin], 1.0)) if fin.any() else float("nan")
    wr = observed_fwhm(h, area) / model_fwhm(model, m)
    bright = fin & np.isfinite(m) & np.isfinite(wr) & (wr > 0) & (h >= bright_x_edge * edge)
    n = int(bright.sum())
    slope = float("nan")
    if n >= 3 and np.ptp(np.log(m[bright])) > 0:
        slope = float(np.polyfit(np.log(m[bright]), np.log(wr[bright]), 1)[0])
    if n < min_peaks:
        return {"ratio": 1.0, "n": n, "measured": False, "slope": slope, "edge": edge}
    return {"ratio": float(np.median(wr[bright])), "n": n, "measured": True,
            "slope": slope, "edge": edge}


def _calibration_skip(cal: dict, model) -> str | None:
    declared = str(getattr(model, "source", "declared")) == "declared"
    if declared and not cal["measured"]:
        return (f"declared width model, and only {cal['n']} bright peaks with a width "
                f"(< {MIN_CAL_PEAKS}) to calibrate it")
    if declared and np.isfinite(cal["slope"]) and abs(cal["slope"]) > MAX_DECLARED_SLOPE:
        return (f"declared width model: the observed/model width ratio trends with m/z "
                f"(log-log slope {cal['slope']:+.2f}), one scale factor cannot calibrate it")
    lo, hi = CAL_RATIO_RANGE
    if cal["measured"] and not (lo <= cal["ratio"] <= hi):
        return (f"the file's median observed/model width ratio {cal['ratio']:.2f} is outside "
                f"{lo:g}-{hi:g}: widths or area convention not as the rule assumes")
    return None


def _fine_structure(i, j, m, h, fw_j, order, ms) -> str | None:
    """The fine-structure spacing candidate i sits at from parent j, when j is
    itself an isotopologue line of an even brighter peak; else None."""
    off = m[i] - m[j]
    for name, d_par, d_line in FINE_STRUCTURE:
        if abs(off - (d_line - d_par)) > FINE_TOL_FWHM * fw_j:
            continue
        target = m[j] - d_par
        tol = FINE_LINK_TOL_FWHM * fw_j
        lo = np.searchsorted(ms, target - tol, "left")
        hi = np.searchsorted(ms, target + tol, "right")
        if any(h[order[k]] > h[j] for k in range(lo, hi)):
            return name
    return None


def isotopologue_link(i, m, h, order, ms, valid, skip=()) -> tuple[int, str] | None:
    """(k, label) when peak i sits at an isotopologue offset (`ISOTOPOLOGUE_LINES`)
    of a brighter valid peak k of the list -- within ISO_LINK_PPM of the exact
    offset, at h[i] / h[k] in that line's ratio band --
    whose position is not in `skip` (the peaks in some line's lobe band); else None. The
    band is ISO_LINK_RATIO_13C scaled by the line's per-atom ratio / 13C's,
    capped at ISO_LINK_MASS_CAP_X x the share of the most atoms m[k] can hold.
    The brightest such k is named."""
    best = None
    tol_max = ISO_LINK_PPM * 1e-6 * m[i]
    lo13, hi13 = ISO_LINK_RATIO_13C
    for label, d, r_atom, m_atom in ISOTOPOLOGUE_LINES:
        target = m[i] - d
        lo = np.searchsorted(ms, target - tol_max, "left")
        hi = np.searchsorted(ms, target + tol_max, "right")
        scale = r_atom / ISO.R_13C_PER_C
        for kk in range(lo, hi):
            k = int(order[kk])
            if k == i or not valid[k] or k in skip or not h[k] > h[i]:
                continue
            if abs(m[i] - m[k] - d) > tol_max:
                continue
            cap = min(hi13 * scale, ISO_LINK_MASS_CAP_X * r_atom * m[k] / m_atom)
            if not (lo13 * scale <= h[i] / h[k] <= cap):
                continue
            if best is None or h[k] > h[best[0]]:
                best = (k, label)
    return best


def sidelobe_parents(mz, height, area, model, *, ratio: float = SIDELOBE_RATIO,
                     band_fwhm=SIDELOBE_BAND_FWHM, narrow: float = SIDELOBE_NARROW) -> dict:
    """Flag the same-spectrum side lobes of ONE peak list (module docstring).

    `mz` / `height` / `area` are the list's picked peaks (any order); `model` the
    width model (chem.resolution.Resolution). Returns, in input order:
    {'parent': input position of the lobe's parent, -1 when not a lobe;
     'offset_fwhm' / 'offset_mda': the lobe's signed offset from it;
     'parent_ratio': parent height / lobe height; 'signature': '+'-joined
     'narrow' / 'neg-lobe' / 'mirror' ('' when not a lobe);
     'skipped': None or why nothing was tested; 'calibration': width_calibration;
     'n_no_signature': in-band, bright-parent peaks spared for want of a signature;
     'n_exempt': {fine-structure name, or '<label> line' for a line on an
     isotopologue offset of a brighter line (`isotopologue_link`): count}
     spared as isotope lines;
     'n_mirror': lobes whose parent has a lobe on the other side too;
     'signatures': the signatures tested -- all three when the calibration is
     measured, only 'neg-lobe' / 'mirror' when it is not}."""
    m = np.asarray(mz, dtype=float)
    h = np.nan_to_num(np.asarray(height, dtype=float), nan=0.0)
    a = np.asarray(area, dtype=float) if area is not None else np.full(m.shape, np.nan)
    n = len(m)
    out = {"parent": np.full(n, -1, dtype=int), "offset_fwhm": np.full(n, np.nan),
           "offset_mda": np.full(n, np.nan), "parent_ratio": np.full(n, np.nan),
           "signature": [""] * n, "skipped": None, "calibration": None,
           "n_no_signature": 0, "n_exempt": {}, "n_mirror": 0, "signatures": ()}
    if n < 2:
        out["skipped"] = "fewer than two peaks"
        return out
    obs = observed_fwhm(h, a)
    if not np.isfinite(obs).any():
        out["skipped"] = "no peak area (the observed widths cannot be measured)"
        return out
    cal = width_calibration(m, h, a, model)
    out["calibration"] = cal
    why = _calibration_skip(cal, model)
    if why:
        out["skipped"] = why
        return out
    # An uncalibrated model (too few bright peaks for a measured or mascope one;
    # a declared one is skipped above): "narrower than a line" would be read
    # against the raw model, and a model 1.5x too wide makes every line read 0.67
    # -- so only the geometric signatures (negative lobe, mirror) are used.
    signatures = SIGNATURES if cal["measured"] else UNCALIBRATED_SIGNATURES
    out["signatures"] = signatures
    med = cal["ratio"]
    mod = model_fwhm(model, m)
    wr = obs / mod                         # own observed / model width
    wrs = wr / med                         # ... relative to the file's typical
    lo_w, hi_w = PARENT_OWN_WIDTH
    own_ok = np.isfinite(wr) & (wr >= lo_w * med) & (wr <= hi_w * med)
    fw_eff = np.where(own_ok, obs, mod * med)
    valid = np.isfinite(m) & (h > 0) & np.isfinite(fw_eff) & (fw_eff > 0)
    if valid.sum() < 2:
        out["skipped"] = "fewer than two peaks with a height and a width"
        return out

    order = np.argsort(np.where(np.isfinite(m), m, np.inf), kind="mergesort")
    ms = m[order]
    band_lo, band_hi = float(band_fwhm[0]), float(band_fwhm[1])
    span = band_hi * float(np.nanmax(np.where(valid, fw_eff, np.nan)))

    def in_band(i, j):
        if j == i or not valid[j] or h[j] < ratio * h[i]:
            return None
        d = (m[i] - m[j]) / fw_eff[j]
        return d if band_lo <= abs(d) <= band_hi else None

    def scan(spared):
        """Every in-band bright-parent geometry (the mirror test reads them all
        -- a lobe's parent holding another one on the opposite side, whose own
        brightest parent may be a different line) and each peak's brightest
        parent. A line at a fine-structure spacing of its parent, or in
        `spared`, is a real isotope line, not a lobe: it is no mirror evidence."""
        parent_side: dict[int, set] = {}
        cand = []
        for i in range(n):
            if not valid[i]:
                continue
            lo = np.searchsorted(ms, m[i] - span, "left")
            hi = np.searchsorted(ms, m[i] + span, "right")
            best, best_d, best_fs = -1, np.nan, None
            for k in range(lo, hi):
                j = int(order[k])
                d = in_band(i, j)
                if d is None:
                    continue
                fs = _fine_structure(i, j, m, h, fw_eff[j], order, ms)
                if fs is None and i not in spared:
                    parent_side.setdefault(j, set()).add(float(np.sign(d)))
                if best < 0 or h[j] > h[best]:
                    best, best_d, best_fs = j, d, fs
            if best >= 0:
                cand.append((i, best, best_d, best_fs))
        return parent_side, cand

    def signatures_of(i, j, d, parent_side):
        sig = []
        if "narrow" in signatures and np.isfinite(wrs[i]) and wrs[i] < narrow:
            sig.append("narrow")
        if "neg-lobe" in signatures and NEG_LOBE_FWHM[0] <= d <= NEG_LOBE_FWHM[1]:
            sig.append("neg-lobe")
        mirrored = "mirror" in signatures and -float(np.sign(d)) in parent_side.get(j, ())
        if mirrored:
            sig.append("mirror")
        return sig, mirrored

    # first scan: the lobe candidates (a signature, no fine-structure spacing).
    # A candidate of a line's own width (not 'narrow': an isotope line is as
    # wide as any line, most lobes are narrower) on an isotopologue offset of a
    # brighter line that is itself clear of every lobe geometry (no line ratio x
    # brighter within the band of it) is that line's isotope line: spared, and
    # the scan is re-read without it as mirror evidence. A brighter line that is itself in some line's band
    # spares nothing: lobes repeat one isotope spacing apart at the parents'
    # own isotope ratio (a lobe of an ion's 13C line sits one 13C spacing above
    # the same lobe of the ion).
    parent_side, cand = scan(frozenset())
    in_some_band = {i for i, j, d, fs in cand}
    lobes = [(i, signatures_of(i, j, d, parent_side)[0]) for i, j, d, fs in cand if fs is None]
    iso_spared: dict[int, str] = {}
    for i, sig in lobes:
        if not sig or "narrow" in sig:
            continue
        link = isotopologue_link(i, m, h, order, ms, valid, skip=in_some_band)
        if link is not None:
            iso_spared[i] = f"{link[1]} line"
    if iso_spared:
        parent_side, cand = scan(frozenset(iso_spared))

    n_no_sig = 0
    exempt: dict[str, int] = {}
    mirror_parents = set()
    for i, j, d, fs in cand:
        sig, mirrored = signatures_of(i, j, d, parent_side)
        if not sig:
            n_no_sig += 1
            continue
        if fs is None:
            fs = iso_spared.get(i)
        if fs is not None:
            exempt[fs] = exempt.get(fs, 0) + 1
            continue
        out["parent"][i] = j
        out["offset_fwhm"][i] = d
        out["offset_mda"][i] = (m[i] - m[j]) * 1e3
        out["parent_ratio"][i] = h[j] / h[i]
        out["signature"][i] = "+".join(sig)
        if mirrored:
            mirror_parents.add(j)
    out["n_no_signature"] = n_no_sig
    out["n_exempt"] = exempt
    out["n_mirror"] = int(sum(1 for i in range(n)
                              if out["parent"][i] in mirror_parents))
    return out
