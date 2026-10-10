"""The response of a peak list's minor carbon / oxygen isotope lines (Orbitrap class).

An Orbitrap scan keeps a line only where it clears that scan's detection floor, so a minor isotope line
near the floor is kept in some scans and lost in the others. The whole-file peak list sums the scans:
it then reads the line LOW -- on 1-microscan files by area about 0.3x its natural-abundance share where the line's expected S/N in the
summed list is ~20, 0.9-1.0x once it clears the floor in every scan -- and by height lower still.
Scored against the natural-abundance prediction, a true C10 product whose 13C line reads 17 % low misses
the pattern-score floor by ~3.4 sigma; at 0.3-0.6x a 13C carbon count reads C3-C6 and the carbon clamps
clear it.

The file measures its own response on the 13C lines of its committed CHO / CHON M0 rows (carbon >= 4):
y = observed / expected 13C area ratio, against x = log10(S/N of the M0 x the line's expected abundance),
the expected S/N of the line itself. Per bin (a bin of too few lines joins the next): the median `y` and a
lower quantile `lo`, each made non-decreasing in x (pool-adjacent violators: a brighter line never reads
lower) and capped at 1.0 -- a dim bin that reads HIGH is detection-selected (only its upward fluctuations
are kept), not an excess, and the expected line is never raised. Fewer than MIN_POINTS such lines: no
response (the lines are read by area against the natural abundance).

Two readers: `band_scale(resp, x)` (from `lo`) is the lower end of the band [lower end, natural share] the
local scorer matches a minor C / O line in -- a line reading inside it costs nothing, one outside is
charged from the nearer end; `scale(resp, x)` (the median) is what the 13C carbon counts are divided by and
the missing-13C expectation multiplied by. The response is a plain dict: it is recorded in the run's stats
and scoring snapshot, and a stand-in judged at that snapshot reads its lines against it.
"""
from __future__ import annotations

import math
import re

import numpy as np
import pandas as pd

R13C = 0.010816                 # 13C / 12C per carbon
D13C = 1.0033548378             # 13C - 12C
MIN_POINTS = 10                 # lines a response needs (else none: read against 1)
BIN = 0.25                      # log10 bin width of the expected line S/N
MIN_PER_BIN = 3
FLOOR, CAP = 0.3, 1.0           # clip of the response
MATERIAL = 0.9                  # a response whose lowest point is under this changes a run (assign re-runs)
MATCH_PPM = 2.0                 # M0 and 13C line matching window for the fit

#: isotopologue labels whose abundance the response scales: carbon and oxygen lines only (a 13C, 13C2,
#: 18O, 17O, 13C18O ... line). Halogen / S / Si / N lines are left alone until measured.
_ISO = r"(?:13C\d*|18O\d*|17O\d*)"
_SCALED = re.compile(rf"^{_ISO}(?:\+?{_ISO})*$")      # '13C', '13C2', '18O', '13C+18O' (IsoSpec's join)
R13C_RAW = 0.0107               # the constant the raw (pre-response) 13C counts have always used
RAMP = 0.5                      # decades above the brightest fitted point over which the response returns to 1
Q_LO = 0.2                      # the band's lower end: this quantile of a bin's readings
#: the band's lower end above a file's brightest fitted line when that line is dim (no bright data): single bright
#: lines read 0.83-0.89 of their natural share by area on the flow-tube and microscan reference files, and the
#: bright knots' lower quantile is 0.89-0.98 there, 1.0 on another instrument -- a fallback, not a measurement
BAND_BRIGHT = 0.8
X_BRIGHT = 2.0                  # log10 line S/N above which a knot counts as bright data
_ALLOWED = {"C", "H", "N", "O"}
_TOKEN = re.compile(r"(\^N|Cl|Br|Si|Na|[A-Z][a-z]?)(\d*)")


def is_scaled_label(label) -> bool:
    s = str(label).replace(" ", "")
    return bool(s) and s != "M0" and bool(_SCALED.match(s))


def _carbons(ion_formula) -> int | None:
    """Carbon count of a CHO / CHON(^N) ion formula, else None (other elements are not fitted on)."""
    if not isinstance(ion_formula, str) or not ion_formula:
        return None
    body = ion_formula.strip().rstrip(".").rstrip("+-")
    comp: dict = {}
    pos = 0
    for m in _TOKEN.finditer(body):
        if m.start() != pos:
            return None
        el = "N" if m.group(1) == "^N" else m.group(1)
        comp[el] = comp.get(el, 0) + (int(m.group(2)) if m.group(2) else 1)
        pos = m.end()
    if pos != len(body) or set(comp) - _ALLOWED:
        return None
    return comp.get("C", 0)


def _pava(y, w):
    """Weighted pool-adjacent-violators: the non-decreasing fit of y."""
    blocks = [[float(v), float(wt), 1] for v, wt in zip(y, w)]
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][0] > blocks[i + 1][0]:
            a, b = blocks[i], blocks[i + 1]
            blocks[i] = [(a[0] * a[1] + b[0] * b[1]) / (a[1] + b[1]), a[1] + b[1], a[2] + b[2]]
            del blocks[i + 1]
            i = max(i - 1, 0)
        else:
            i += 1
    return [v for v, _, n in blocks for _ in range(n)]


def fit(peaks: pd.DataFrame, ions, *, col: str = "area", ppm: float = MATCH_PPM) -> dict | None:
    """The file's response from its peaks (`mz`, `col`, `signal_to_noise`) and committed M0 ions
    (an iterable of (mz, ion_formula)). None when fewer than MIN_POINTS lines measure it."""
    if peaks is None or not len(peaks) or col not in peaks or "signal_to_noise" not in peaks:
        return None
    pk = peaks[["mz", col, "signal_to_noise"]].copy()
    pk[col] = pd.to_numeric(pk[col], errors="coerce")
    pk["signal_to_noise"] = pd.to_numeric(pk["signal_to_noise"], errors="coerce")
    pk = pk[pk.mz.notna() & (pk[col] > 0)].sort_values("mz")
    mz, inten, snr = pk.mz.to_numpy(), pk[col].to_numpy(), pk.signal_to_noise.to_numpy()
    if len(mz) < 2:
        return None

    def brightest(target):
        lo, hi = np.searchsorted(mz, target * (1 - ppm * 1e-6)), np.searchsorted(mz, target * (1 + ppm * 1e-6))
        return None if hi <= lo else lo + int(np.argmax(inten[lo:hi]))

    xs, ys = [], []
    for m, f in ions:
        c = _carbons(f)
        if not c or c < 4 or not np.isfinite(m):
            continue
        i = brightest(float(m))
        if i is None or not (np.isfinite(snr[i]) and snr[i] > 0):
            continue
        j = brightest(mz[i] + D13C)
        if j is None or j == i:
            continue
        y = (inten[j] / inten[i]) / (c * R13C)
        if 0.05 < y < 3.0:
            xs.append(math.log10(snr[i] * c * R13C))
            ys.append(y)
    if len(xs) < MIN_POINTS:
        return None
    xs, ys = np.asarray(xs), np.asarray(ys)
    # bins of BIN in x; a bin with fewer than MIN_PER_BIN lines joins the next one (the sparse bright lines
    # form a knot of their own instead of being dropped), what is left at the bright end joins the last knot
    groups, acc = [], np.zeros(len(xs), bool)
    for lo in np.arange(math.floor(xs.min() / BIN) * BIN, xs.max() + BIN, BIN):
        acc |= (xs >= lo) & (xs < lo + BIN)
        if acc.sum() >= MIN_PER_BIN:
            groups.append(acc)
            acc = np.zeros(len(xs), bool)
    if acc.any():
        if not groups:
            return None
        groups[-1] = groups[-1] | acc
    cx = [float(xs[g].mean()) for g in groups]
    cw = [float(g.sum()) for g in groups]
    cy = _pava([float(np.clip(np.median(ys[g]), FLOOR, CAP)) for g in groups], cw)
    cl = _pava([float(np.clip(np.quantile(ys[g], Q_LO), FLOOR, CAP)) for g in groups], cw)   # <= cy: PAVA keeps order
    return {"x": [round(v, 4) for v in cx], "y": [round(v, 4) for v in cy], "lo": [round(v, 4) for v in cl],
            "n": int(len(xs)), "col": col}


def scale(resp: dict | None, x: float | None) -> float:
    """The response at x = log10(expected S/N of the line); 1.0 without one. A line whose S/N is not
    known reads at the response's brightest point (the scorer judges such a line at the instrument width)."""
    if not resp:
        return 1.0
    xs, ys = resp["x"], resp["y"]
    if x is None or not np.isfinite(x):
        return float(ys[-1])
    if x > xs[-1]:
        # brighter than any fitted line: censoring fades with S/N, so the response returns to 1 over RAMP
        # decades instead of holding the brightest bin's value (a flat curve must not reach bright lines)
        return float(ys[-1] + (1.0 - ys[-1]) * min(1.0, (x - xs[-1]) / RAMP))
    return float(np.interp(x, xs, ys))


def band_scale(resp: dict | None, x: float | None) -> float:
    """The lower end of the band at x = log10(expected S/N of the line), as a share of the natural
    abundance: the fitted lower quantile of the file's readings (`lo`; the median `y` for a record without
    one). Above the brightest fitted line it holds that line's value -- or, where that line is dim (below
    X_BRIGHT), it moves to BAND_BRIGHT over RAMP decades. A line whose S/N is not known reads at the
    brightest point. 1.0 without a response."""
    if not resp:
        return 1.0
    xs, lo = resp["x"], resp.get("lo") or resp["y"]
    if x is None or not np.isfinite(x):
        return float(lo[-1])
    if x <= xs[-1]:
        return float(np.interp(x, xs, lo))
    if xs[-1] >= X_BRIGHT or lo[-1] >= BAND_BRIGHT:
        return float(lo[-1])
    return float(lo[-1] + (BAND_BRIGHT - lo[-1]) * min(1.0, (x - xs[-1]) / RAMP))


def band(resp: dict | None, pred: float, snr0) -> float:
    """The lower end of the band a minor C / O line may read in: pred x band_scale at its expected S/N. A
    line near the scans' floor is kept only where it fluctuates up, so the summed list reads it anywhere
    from its censored share up to its natural share `pred`; a bright line reads within the file's own
    spread of its natural share. Without a response: pred (no band)."""
    if not resp:
        return pred
    x = math.log10(snr0 * pred) if (snr0 is not None and np.isfinite(snr0) and snr0 > 0 and pred > 0) else None
    return pred * band_scale(resp, x)


def is_material(resp: dict | None) -> bool:
    return bool(resp) and min(resp["y"]) < MATERIAL


def carbon_from_13c(resp: dict | None, *, h0: float, h_sat: float, a0=None, a_sat=None, snr0=None,
                    n_c: float | None = None) -> float:
    """A 13C carbon count: the area ratio where both areas are known and the response (if any) is by area,
    else the height ratio, over R13C; with a response, divided by its median reading at the line's
    expected S/N. `n_c` (the formula's carbon count, if any) sets the expected S/N; else the count read
    does. (The raw check of `contradicts` reads heights over R13C_RAW, the constant it always used.)"""
    use_area = (resp is None or resp.get("col") == "area") and bool(a0 and a_sat and a0 > 0 and a_sat > 0)
    c = ((a_sat / a0) if use_area else (h_sat / h0)) / R13C
    if not resp:
        return c
    ref = n_c if n_c else c
    x = math.log10(snr0 * ref * R13C) if (snr0 and snr0 > 0 and ref > 0) else None
    return c / scale(resp, x)


def contradicts(resp: dict | None, n_c: float, *, h0: float, h_sat: float, a0=None, a_sat=None, snr0=None,
                tol=None, area_mode: bool = False) -> tuple[bool, float]:
    """Does the 13C line contradict a formula's carbon count n_c? The raw reading (by height against the
    natural abundance, R13C_RAW -- exactly the check before) decides alone unless the run reads its lines
    by area (`area_mode`) or has a response: then the area reading divided by the response's median at the
    line's expected S/N (1 without one) must contradict TOO, and in the same direction. A true line that
    reads low only because it is dim is then not taken as too few carbons; the price is that near the
    scans' floor, where the response is low, an over-claim whose line reads near the response is not
    cleared by its 13C line either (the line cannot tell the two apart there: C9 at its natural share and
    C14 at 0.64 of its share read the same where the response is 0.64).
    Returns (contradicted, the count reported: the corrected one where it was read)."""
    tol = max(2.5, 0.35 * n_c) if tol is None else tol
    d_raw = (h_sat / h0) / R13C_RAW - n_c
    if not (resp or area_mode):
        return abs(d_raw) > tol, d_raw + n_c
    c_corr = carbon_from_13c(resp, h0=h0, h_sat=h_sat, a0=a0, a_sat=a_sat, snr0=snr0, n_c=n_c)
    d_corr = c_corr - n_c
    return (abs(d_raw) > tol and abs(d_corr) > tol and (d_raw > 0) == (d_corr > 0)), c_corr
