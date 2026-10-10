"""The diagnostic heavy-isotope line a composition predicts, judged on the LEDGER.

One test, two callers. Pass 0 (`passes.directors._twin_verdict`) asks it of a
refused single-channel known-species claim and leaves the answer as a
`known_lead`; the tier engine (`tiers.compute_tiers`) asks it of every
committed M0 whose neutral carries Br, Cl or S and reads the verdict into the
tier -- a line the file could have shown and did not refutes the heteroatom
count whatever else corroborates the row; a line the file could not have shown
leaves the count untested, and an untested count with no other corroboration
does not earn Assigned.

Why the ledger and not the scorer: the scorer credits an isotopologue only
when it matched one, and its silence is not a refutation -- measured on a
ten-file uronium batch, a cyclosiloxane urea adduct shows its 29Si and 30Si
lines at the predicted ratios in every file and the scorer credited two.

Why a floor multiple: an absence in the per-file peak list, or a picked ratio,
counts only for a line predicted at TWIN_REFUTE_X_FLOOR (4x) the per-file
noise edge. Measured (per-scan centroid labels of an Orbitrap batch): the
instrument labels a centroid above S/N 1.8 and the per-scan noise is ~1.6x the
per-file edge, so a line reaches the per-file list only when its mean is ~4x
the edge; below that it is labelled in a fraction of the scans, censored low
or missing, and its absence is the threshold, not evidence.

Why the ion's atom count and the neutral's element: the envelope is the ION's
(a bromo-organic on a bromide adduct carries two Br), so the prediction uses
the ion; but the question is the NEUTRAL's heteroatom, so the element is chosen
from the neutral. When the reagent adduct itself brings Br or Cl, its own heavy
line sits in the same M+2 window as the neutral's 34S / 37Cl / 81Br (within
2.2 mDa), and at picked-height precision the neutral's share cannot be read
off -- the tier caller marks such a test masked (untestable), never refuted:
the ion is pinned, the neutral's element is not (the merge vote's private
class reads it so: docs/EVIDENCE_LEVELS.md section 13).

This module imports only chemistry: `tiers` reads it and `passes.postprocess`
reads `tiers`, so the twin test cannot live in the passes package.

The ION's own M+2 line on a TOF (`heavy_line_verdict`) is the second test here,
and it asks a different question: not whether the neutral's heteroatom shows,
but whether the ion's own Br / Cl envelope does -- the reagent adduct's halogen
included. On a bromide TOF an [M+Br]- reading's 81Br line is the one line its
own composition guarantees, and the twin test above never asks for it (the
reagent masks the neutral's window). At R ~10 000 the whole M+2 cluster (81Br,
37Cl, 34S, 30Si, 18O, 13C2) is one line, so the prediction is the cluster's
summed height (isotopes.nominal_cluster). The same primitive judges one file's
peak list (the tier pass, `tiers.apply_tof_m2`) and every spectrum of a batch
(the TOF branch of the batch REQ check, `batch.iso_checks`); see its docstring
for the rule and the measurement behind the numbers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from peaky.assignment import ledger as L
from peaky.chem import isotopes as ISO

#: the diagnostic heavy-isotope lines tested, per element: (shift from M0,
#: abundance per atom, label). Si has two -- on an Orbitrap above ~m/z 300 the
#: picker resolves the 29Si line (+0.9996) from the 13C line (+1.0034), so the
#: picked M+1 is the 29Si line ALONE and a blended (29Si + 13C) prediction
#: over-demands by the 13C share; the 30Si M+2 is what no organic can fake.
TWIN_LINES = {
    "Br": ((ISO.D_81BR, ISO.R_81BR_PER_BR, "81Br"),),
    "Cl": ((ISO.D_37CL, ISO.R_37CL_PER_CL, "37Cl"),),
    "S": ((ISO.D_34S, ISO.R_34S_PER_S, "34S"),),
    "Si": ((ISO.D_29SI, ISO.R_29SI_PER_SI, "29Si"), (ISO.D_30SI, ISO.R_30SI_PER_SI, "30Si")),
}
#: a line refutes a claim -- by its absence, or by a picked height under
#: TWIN_MIN_FRAC of its prediction -- only when it was predicted at this multiple
#: of the resolved height gate (the per-file noise edge); see the module note.
TWIN_REFUTE_X_FLOOR = 4.0
#: a line PRESENT at less than this fraction of its predicted height refutes
#: the claim (the Si M+1 gate's own fraction, postprocess.SI_M1_MIN_FRAC)
TWIN_MIN_FRAC = 0.6
TWIN_PPM = 15.0        # the M+1 / M+2 window the Si M+1 gate uses
#: the elements whose line the TIER reads (Si keeps its own rule in tiers.py:
#: on a TOF the 29Si M+1 is unresolved from 13C, so its ratio is not a test)
TIER_ELEMENTS = ("Br", "Cl", "S")
#: the order pass 0 tests -- the brightest first-order line first
DEFAULT_ELEMENTS = ("Br", "Cl", "S", "Si")


def _cps(x: float) -> str:
    """A predicted line height for a reason string: one decimal under 10 cps (a
    TOF's 34S line of a 5-cps parent is 0.2 cps, not "0 cps"), else whole cps."""
    x = float(x)
    return f"{x:.1f}" if x < 10 else f"{x:.0f}"


def twin_element(counts: dict, elements=DEFAULT_ELEMENTS) -> str | None:
    """The element whose diagnostic heavy-isotope line would corroborate this
    composition, if any -- the first of `elements` the composition carries;
    None for one monoisotopic in every heteroatom tested (P, F, I), which no
    twin can ever test."""
    for el in elements:
        if counts.get(el, 0) > 0:
            return el
    return None


def peak_near(mzs: pd.Series, target: float, ppm: float = TWIN_PPM):
    """Index of the closest ledger peak within `ppm` of `target`, else None
    (the same reading as passes.postprocess._peak_near)."""
    tol = target * ppm * 1e-6
    d = (mzs - target).abs()
    if not len(d) or d.isna().all():
        return None
    i = d.idxmin()
    return i if d.loc[i] <= tol else None


def reagent_masks(element: str, neutral_counts: dict, ion_counts: dict) -> str | None:
    """The reason the `element` test is uninformative on this ion, or None: the
    reagent adduct adds Br or Cl the neutral lacks, and that halogen's heavy
    line falls in the same M+2 window as the line under test."""
    if element not in TIER_ELEMENTS:
        return None
    label = TWIN_LINES[element][0][2]
    for x, line in (("Br", "81Br"), ("Cl", "37Cl")):
        if int(ion_counts.get(x, 0) or 0) > int(neutral_counts.get(x, 0) or 0):
            return (f"the reagent adduct's {line} line sits in the same M+2 window as the "
                    f"neutral's {label} line, so the neutral's share cannot be read off the picked heights")
    return None


def twin_verdict(ledger: pd.DataFrame, pid, counts: dict, floor, *, element: str | None = None,
                 prefix: str = "", masked_by: str | None = None, ppm: float = TWIN_PPM) -> dict:
    """Judge `counts`' diagnostic line(s) of `element` (default: the first of
    DEFAULT_ELEMENTS it carries) on the ledger peak `pid`.

    Returns {'verdict': 'refuted' | 'deferred', 'kind': 'refuted' | 'supported'
    | 'untestable', 'twin': element or None, 'why': the reason with this file's
    numbers (after `prefix`), 'summary': the same without them, 'lines': one
    record per line tested}. `refuted` when a line predicted at >=
    TWIN_REFUTE_X_FLOOR x `floor` is absent within `ppm` (TWIN_PPM; an
    Orbitrap-class run passes the exact-offset window, `twin_ppm`) or sits under
    TWIN_MIN_FRAC of its prediction; `supported` when a line is present and
    consistent (or present under the multiple: support, its ratio censored);
    `untestable` otherwise -- no floor, no parent height, every line predicted
    under the multiple and absent, or `masked_by` given (the caller knows the
    window is uninformative). Silence never votes against a claim; a
    refutation does."""
    el = element or twin_element(counts)
    if el is None:
        return {"verdict": "deferred", "kind": "untestable", "twin": None, "lines": [],
                "why": prefix + "no diagnostic twin to test (monoisotopic)",
                "summary": "no diagnostic twin to test (monoisotopic)"}
    if masked_by:
        return {"verdict": "deferred", "kind": "untestable", "twin": el, "lines": [],
                "why": prefix + masked_by, "summary": "reagent line masks the twin window"}
    if floor is None:
        return {"verdict": "deferred", "kind": "untestable", "twin": el, "lines": [],
                "why": prefix + f"no resolved detection floor to judge the {el} twin",
                "summary": "no resolved detection floor"}
    idx = ledger.index[ledger["peak_id"] == pid]
    if not len(idx) or pd.isna(ledger.at[idx[0], "height"]) or float(ledger.at[idx[0], "height"]) <= 0:
        return {"verdict": "deferred", "kind": "untestable", "twin": el, "lines": [],
                "why": prefix + f"no parent height to predict the {el} twin from",
                "summary": "no parent height"}
    h = float(ledger.at[idx[0], "height"])
    m0 = float(ledger.at[idx[0], "mz"])
    n = int(counts.get(el, 0))
    support, contra, under, lines = [], [], [], []          # (detail, summary) pairs
    for delta, per_atom, label in TWIN_LINES[el]:
        pred_ratio = n * per_atom
        pred_h = pred_ratio * h
        testable = pred_h >= TWIN_REFUTE_X_FLOOR * floor
        j = peak_near(ledger["mz"], m0 + delta, ppm=ppm)
        rec = {"label": label, "pred_ratio": pred_ratio, "pred_h": pred_h, "testable": testable,
               "obs_ratio": None, "status": ""}
        if j is None or pd.isna(ledger.at[j, "height"]):
            if testable:
                contra.append((f"no {label} line at +{delta:.4f} (predicted {pred_ratio:.2f}x the parent, "
                               f"{_cps(pred_h)} cps)", f"no {label} line where one was predicted above the floor"))
                rec["status"] = "absent"
            else:
                under.append((f"{label} predicted at {_cps(pred_h)} cps, under {TWIN_REFUTE_X_FLOOR:g}x "
                              f"the {floor:.0f}-cps floor", label))
                rec["status"] = "under"
            lines.append(rec)
            continue
        obs = float(ledger.at[j, "height"]) / h
        rec["obs_ratio"] = obs
        if not testable:
            # a line the picker holds although it was predicted near its edge: it
            # is there, and its picked height is censored low (the per-file height
            # averages the scans that labelled it with the ones that did not)
            support.append((f"{label} line at {obs:.2f}x the parent (predicted {pred_ratio:.2f}, "
                            f"{_cps(pred_h)} cps, under {TWIN_REFUTE_X_FLOOR:g}x the floor: ratio censored)",
                            label))
            rec["status"] = "present-censored"
        elif obs >= TWIN_MIN_FRAC * pred_ratio:
            support.append((f"{label} line at {obs:.2f}x the parent (predicted {pred_ratio:.2f})", label))
            rec["status"] = "present"
        else:
            contra.append((f"{label} line at {obs:.2f}x the parent, under {TWIN_MIN_FRAC:g}x the "
                           f"predicted {pred_ratio:.2f}",
                           f"{label} line under {TWIN_MIN_FRAC:g}x its prediction"))
            rec["status"] = "low"
        lines.append(rec)

    def _lines(items):
        labs = [s for _, s in items]
        return (" and ".join(labs) + (" lines" if len(labs) > 1 else " line"))

    if contra:
        return {"verdict": "refuted", "kind": "refuted", "twin": el, "lines": lines,
                "why": prefix + "; ".join(d for d, _ in contra),
                "summary": "; ".join(s for _, s in contra)}
    if support:
        return {"verdict": "deferred", "kind": "supported", "twin": el, "lines": lines,
                "why": prefix + "; ".join(d for d, _ in support)
                       + " -- present in the ledger, not credited by the scorer",
                "summary": f"{_lines(support)} present at the predicted ratio, not credited by the scorer"}
    return {"verdict": "deferred", "kind": "untestable", "twin": el, "lines": lines,
            "why": prefix + "; ".join(d for d, _ in under),
            "summary": f"{_lines(under)} predicted under {TWIN_REFUTE_X_FLOOR:g}x the floor"}


# ---------------------------------------------------------------------------
# the ION's own M+2 line on a TOF
# ---------------------------------------------------------------------------
#: the elements that make an ion's M+2 line testable on a TOF -- the ion's,
#: the reagent adduct's own Br / Cl included. 34S, 13C2 and 18O add to the
#: predicted height but never make an ion testable on their own: a few percent
#: of the parent is no line a TOF's neighbours leave alone.
HEAVY_LINE_ELEMENTS = ("Br", "Cl")
#: seen: a line at >= this fraction of the prediction (the twin test's own)
HEAVY_LINE_FRAC = TWIN_MIN_FRAC
#: the search window: the scorer's TOF match window, or HEAVY_LINE_SIGMA_K x the
#: fitted mass sigma where that is wider (a poorly fitted file's lines wander
#: further). Measured on a bromide/nitrate TOF batch (R ~9 650): the true Br1
#: readings' 81Br lines sit within ~5 ppm (median) of their position, and a
#: 12 ppm window made 4 of 84 batch demotions wrong -- their M+2 line present at
#: -14 to +12 ppm, co-varying with its M0 at r 0.96-1.00.
HEAVY_LINE_MIN_PPM = 15.0
HEAVY_LINE_SIGMA_K = 3.0
#: blended (untestable): a line other than the M0 within this many FWHM of the
#: target at >= the prediction (another ion's line holds the position) ...
HEAVY_LINE_BLEND_FWHM = 1.0
#: ... or the lines within +- this many FWHM summing to >= HEAVY_LINE_FRAC x the
#: prediction: an unresolved split, the picker's sub-peaks sharing the line's
#: counts (on the same batch it spared one wrong and 14 ambiguous demotions and
#: no right one)
HEAVY_LINE_SPLIT_FWHM = 0.5
#: the verdicts: the line seen; absent (a refutation); blended (untestable: the
#: position is another line's or split); dim (predicted under the floor);
#: none (the ion carries neither Br nor Cl -- nothing to test)
HL_SEEN, HL_ABSENT, HL_BLENDED, HL_DIM, HL_NONE = "seen", "absent", "blended", "dim", "none"


def heavy_line_prediction(ion_counts: dict) -> tuple[float, float, str]:
    """(ratio, shift, label) of the ION's M+2 line as a TOF shows it: every
    isotopologue at nominal +2 Da summed relative to the all-light line and its
    probability-weighted shift (isotopes.nominal_cluster), and the halogen line
    that drives it ('81Br', '37Cl' or '81Br/37Cl'). (0.0, nan, '') for an ion
    carrying neither Br nor Cl."""
    c = {k: int(v) for k, v in (ion_counts or {}).items() if v}
    labs = [lab for el, lab in (("Br", "81Br"), ("Cl", "37Cl")) if c.get(el, 0) > 0]
    if not labs:
        return 0.0, float("nan"), ""
    ratio, shift = ISO.nominal_cluster(c, 2)
    return float(ratio), float(shift), "/".join(labs)


def heavy_line_window_ppm(sigma_ppm=None, tol_ppm=None) -> float:
    """The M+2 search window (ppm): the scorer's match window `tol_ppm`
    (HEAVY_LINE_MIN_PPM when unknown) or HEAVY_LINE_SIGMA_K x `sigma_ppm`,
    whichever is wider."""
    def _f(x):
        try:
            x = float(x)
        except (TypeError, ValueError):
            return None
        return x if np.isfinite(x) and x > 0 else None
    tol = _f(tol_ppm) or HEAVY_LINE_MIN_PPM
    sig = _f(sigma_ppm)
    return float(max(tol, HEAVY_LINE_SIGMA_K * sig)) if sig is not None else float(tol)


def _range_reduce(mz, h, lo, hi, exclude, how):
    """Per (lo, hi) window over the sorted lines (mz, h): the tallest line ('max')
    or the summed height ('sum'), 0 where empty; a line within 1e-6 of its
    window's `exclude` m/z (the parent's own line) is left out."""
    a = np.searchsorted(mz, lo, "left")
    b = np.searchsorted(mz, hi, "right")
    k = b - a
    out = np.zeros(len(lo))
    for off in range(int(k.max()) if len(k) else 0):
        sel = k > off
        idx = a[sel] + off
        v = np.where(np.abs(mz[idx] - exclude[sel]) < 1e-6, 0.0, h[idx])
        out[sel] = np.maximum(out[sel], v) if how == "max" else out[sel] + v
    return out


def heavy_line_verdict(mz, height, m0_mz, m0_height, ratio, shift, floor, *, win_ppm, fwhm) -> dict:
    """Judge the ION's own M+2 line of each parent line against a peak list.

    `mz` / `height`: the lines one spectrum (or one file's peak list) shows;
    `m0_mz` / `m0_height`: the parent lines, each with its ion's predicted M+2
    `ratio` and `shift` (heavy_line_prediction; ratio 0 = nothing to test);
    `floor`: the height the prediction must reach to be testable (on a TOF the
    tier pass's counting-detector floor, tiers.tof_assign_floor: k_detect x
    the batch's detection edge); `win_ppm`: the search window
    (heavy_line_window_ppm); `fwhm`: the peak width (Da) at each target.
    Parents, ratios, shifts and widths broadcast against each other.

    pred = ratio x the parent's height, at parent + shift. Testable when pred
    >= floor. Seen: a line within win_ppm at >= HEAVY_LINE_FRAC x pred.
    Blended (untestable): not seen, and a line other than the parent within
    HEAVY_LINE_BLEND_FWHM FWHM at >= pred, or the lines within +-
    HEAVY_LINE_SPLIT_FWHM FWHM summing to >= HEAVY_LINE_FRAC x pred. Absent:
    testable, neither seen nor blended -- the one verdict that refutes.

    Returns arrays: status (HL_*), pred (cps), obs (the tallest line in the
    window), wide (the tallest non-parent line within the blend reach) and
    split (the summed lines within the split reach)."""
    mz = np.asarray(mz, dtype=float)
    h = np.asarray(height, dtype=float)
    ok = np.isfinite(mz) & np.isfinite(h)
    mz, h = mz[ok], h[ok]
    if len(mz) > 1 and np.any(np.diff(mz) < 0):
        o = np.argsort(mz, kind="mergesort")
        mz, h = mz[o], h[o]
    m0, h0, r, s, fw = np.broadcast_arrays(*(np.atleast_1d(np.asarray(x, dtype=float))
                                             for x in (m0_mz, m0_height, ratio, shift, fwhm)))
    has = np.isfinite(r) & (r > 0) & np.isfinite(s)
    tgt = np.where(has, m0 + np.where(np.isfinite(s), s, 0.0), m0)
    pred = np.where(has, r * h0, 0.0)
    w = tgt * float(win_ppm) * 1e-6
    fw = np.where(np.isfinite(fw) & (fw > 0), fw, 0.0)
    obs = _range_reduce(mz, h, tgt - w, tgt + w, m0, "max")
    wide = _range_reduce(mz, h, tgt - HEAVY_LINE_BLEND_FWHM * fw, tgt + HEAVY_LINE_BLEND_FWHM * fw, m0, "max")
    split = _range_reduce(mz, h, tgt - HEAVY_LINE_SPLIT_FWHM * fw, tgt + HEAVY_LINE_SPLIT_FWHM * fw, m0, "sum")
    try:
        fl = float(floor)
    except (TypeError, ValueError):
        fl = float("nan")
    test = has & np.isfinite(pred) & (pred >= fl) if np.isfinite(fl) else np.zeros(len(pred), dtype=bool)
    seen = test & (obs >= HEAVY_LINE_FRAC * pred)
    blend = test & ~seen & ((wide >= pred) | (split >= HEAVY_LINE_FRAC * pred))
    status = np.where(~has, HL_NONE, np.where(~test, HL_DIM, np.where(
        seen, HL_SEEN, np.where(blend, HL_BLENDED, HL_ABSENT)))).astype(object)
    return {"status": status, "pred": pred, "obs": obs, "wide": wide, "split": split}


# ---------------------------------------------------------------------------
# the ELEMENT-EVIDENCE predicate: does this file's peak list support the
# heteroatom a formula claims? (exact fine-structure offsets; self-calibrated)
# ---------------------------------------------------------------------------
# The per-peak grid proposes C / H / N / O only; every S, Cl, Br or Si formula
# a run commits came from a widened search (a certificate, an isotope-pair, a
# contaminant family, a series gap-fill), and none of those needs the element's
# own heavy line. On an Orbitrap the line sits at an EXACT offset from its parent
# (81Br +1.9979521, 37Cl +1.99705, 34S +1.995796, 29Si +0.999568, 30Si
# +1.99684) and the fine structure is resolved, so the question is asked at
# that offset within max(EE_MIN_PPM, EE_SIGMA_K x the file's calibration sigma)
# -- a window an unrelated neighbour line several ppm away never enters (the
# twin test's 15 ppm and the iso-pair finder's 8 ppm did).
#
# When an absence counts is NOT a constant: it is fitted per file from the
# file's own 13C (and 18O) lines of its committed CHON M0 rows
# (`element_calibration`) -- the detection probability as a function of the
# line's predicted height over the local floor, and the observed / predicted
# ratio quantiles. A 100-microscan file showed every 13C line predicted at >= 3x
# the floor and read it at ~0.9x its prediction; a 1-microscan file of the same
# instrument read the same lines at 0.2-0.35x and missed a third of them, so any
# fixed multiple would have called most real lines there absent. A line is
# 'observable' only where the file's own detection reaches EE_DETECT, 'low'
# under the file's own EE_LOW_Q-th percentile ratio at that height, and a file
# with fewer than EE_MIN_CAL calibration lines at the observable level never
# contradicts anything.
#
# Verdicts: 'confirmed' (a line of the element seen at its exact offset, in the
# ratio band), 'contradicted' (no line seen and one that the file would show is
# absent or too low), 'unobservable' (everything else: dim, blended, uncalibrated,
# or a class with no exact-offset test). Silence never contradicts; an absence
# the file's own calibration says it would not miss does.
#: the elements the predicate tests (P, F and I are monoisotopic: no line)
EE_ELEMENTS = ("Br", "Cl", "S", "Si")
#: the search window: max(EE_MIN_PPM, EE_SIGMA_K x the file's calibration sigma)
#: (measured 13C offsets on a 100-microscan Orbitrap batch: median -0.03 ppm,
#: |offset| p95 0.42, p99 0.62 ppm)
EE_MIN_PPM = 1.0
EE_SIGMA_K = 4.0
#: the local floor: the EE_FLOOR_Q-th percentile height of the file's peaks
#: within +- EE_FLOOR_HALF_DA of the line, +- EE_FLOOR_WIDE_DA where fewer than
#: EE_FLOOR_MIN_N peaks sit in the narrow window
EE_FLOOR_Q = 5.0
EE_FLOOR_HALF_DA = 5.0
EE_FLOOR_WIDE_DA = 20.0
EE_FLOOR_MIN_N = 5
#: observable: the file's own detection probability at the line's predicted
#: height / floor -- a logistic in log(height / floor), fitted on every
#: calibration line, the dim ones fixing its shape -- reaches this, with at
#: least EE_MIN_OBS calibration lines at or above that level ...
EE_DETECT = 0.98
EE_MIN_OBS = 10
#: ... where the lines at or above the fitted level are found at least this often
EE_DETECT_EMP = 0.95
#: ... in a file with at least this many calibration lines in all (fewer: the
#: predicate never contradicts). Measured on a 100-microscan Orbitrap batch: ~380
#: lines a file, ~25 of them predicted at >= 3x the local floor
EE_MIN_CAL = 30
#: low: an observed / predicted ratio under this percentile of the file's own
#: calibration ratios at that height
EE_LOW_Q = 1.0
#: high: a line taller than this multiple of its prediction is another ion's
#: (neither confirms nor contradicts)
EE_RATIO_HIGH = 2.5
#: the low edge of the confirmation band where the file is not calibrated
EE_LOW_DEFAULT = 0.5
#: the blend guard's smallest real line, as a fraction of the prediction
EE_BLEND_MIN_FRAC = 0.25
#: the width the blend guard assumes without a width model: an Orbitrap at a
#: conservative R(200) (a wider reach reads more positions as possibly blended)
EE_FALLBACK_R200 = 60_000.0
#: the calibration lines: (shift, per-atom ratio, element, label)
EE_CAL_LINES = ((ISO.D_13C_EXACT, ISO.R_13C_PER_C, "C", "13C"), (ISO.D_18O, ISO.R_18O_PER_O, "O", "18O"))
#: a committed line farther than this from its ion's all-light m/z sits on a
#: heavy isotopologue (a Br2 ion on its 79Br81Br line): its lines are not where
#: the predicate would look
EE_MONO_DA = 0.5
#: the verdicts
EE_CONFIRMED, EE_CONTRADICTED, EE_UNOBSERVABLE = "confirmed", "contradicted", "unobservable"


def element_window_ppm(cal_sigma=None) -> float:
    """The exact-offset search window (ppm): max(EE_MIN_PPM, EE_SIGMA_K x the
    file's calibration sigma); EE_MIN_PPM when the sigma is unknown."""
    try:
        s = float(cal_sigma)
    except (TypeError, ValueError):
        return EE_MIN_PPM
    return float(max(EE_MIN_PPM, EE_SIGMA_K * s)) if np.isfinite(s) and s > 0 else EE_MIN_PPM


def calibrated_sigma(cfg=None) -> float | None:
    """The file's fitted calibration sigma (cfg.cal_sigma, set by
    passes.calibrate) when it is a positive number, else None: the file is not
    calibrated yet (pass 0 runs before `calibrate`) or its backbone was too
    small to calibrate it."""
    try:
        s = float(getattr(cfg, "cal_sigma", None))
    except (TypeError, ValueError):
        return None
    return s if np.isfinite(s) and s > 0 else None


def exact_offset_ppm(cfg=None, wide: float = TWIN_PPM) -> float:
    """A REFUTING search window for this run: on an Orbitrap-class run
    (cfg.instrument_class, not trace-first's synthetic sample) whose file is
    calibrated, the exact-offset window (`element_window_ppm` of cfg.cal_sigma);
    else `wide`, the window the test had before. An uncalibrated file keeps the
    wide window: its lines can sit more than EE_MIN_PPM off their exact offset
    (a 29Si line 1.01 ppm off read 'no 29Si line' at pass 0), and an absence
    read in a window narrower than the file's own scatter refutes a real line."""
    if cfg is not None and getattr(cfg, "instrument_class", None) == "orbitrap" \
            and not getattr(cfg, "trace_sample", False):
        s = calibrated_sigma(cfg)
        if s is not None:
            return element_window_ppm(s)
    return float(wide)


def twin_ppm(cfg=None) -> float:
    """The twin test's search window for this run (`exact_offset_ppm`): the
    exact-offset window on a calibrated Orbitrap-class run -- at TWIN_PPM a
    neighbour line 6.7 ppm from a predicted 81Br line was read as it -- else
    TWIN_PPM."""
    return exact_offset_ppm(cfg, TWIN_PPM)


def _peak_lines(ledger: pd.DataFrame, *, artifacts: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """(mz, height) of every real picked peak of the ledger, sorted by m/z
    (synthetic composite sub-peaks are no line). `artifacts=False` also leaves
    out every row a per-file stage marked 'artifact' -- the side lobes of the
    Orbitrap guard (sidelobe_guard) and cleanup's ringing: peaks the profile
    does not hold, so they confirm no line and set no floor. It is the rule
    REQ's Orbitrap branch applies to the stamped series, where the two kinds
    carry the same role and nothing else tells them apart."""
    if ledger is None or not len(ledger) or "mz" not in ledger.columns or "height" not in ledger.columns:
        return np.zeros(0), np.zeros(0)
    mz = pd.to_numeric(ledger["mz"], errors="coerce").to_numpy(dtype=float)
    h = pd.to_numeric(ledger["height"], errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(mz) & np.isfinite(h) & (h > 0)
    if "synthetic" in ledger.columns:
        ok &= ~ledger["synthetic"].map(L._truthy).to_numpy(dtype=bool)
    if not artifacts and "role" in ledger.columns:
        ok &= ledger["role"].astype(str).to_numpy() != L.ROLE_ARTIFACT
    mz, h = mz[ok], h[ok]
    o = np.argsort(mz, kind="mergesort")
    return mz[o], h[o]


def local_floor(mz: np.ndarray, h: np.ndarray, target: float) -> float:
    """The EE_FLOOR_Q-th percentile height of the peaks within +-
    EE_FLOOR_HALF_DA of `target` (+- EE_FLOOR_WIDE_DA where fewer than
    EE_FLOOR_MIN_N sit there); nan without peaks."""
    for half in (EE_FLOOR_HALF_DA, EE_FLOOR_WIDE_DA):
        a, b = np.searchsorted(mz, [target - half, target + half])
        if b - a >= EE_FLOOR_MIN_N or half == EE_FLOOR_WIDE_DA:
            return float(np.percentile(h[a:b], EE_FLOOR_Q)) if b > a else float("nan")
    return float("nan")


def _nearest(mz: np.ndarray, target: float, tol: float):
    """Index of the closest line within `tol` (Da) of `target`, else None."""
    a, b = np.searchsorted(mz, [target - tol, target + tol])
    if b <= a:
        return None
    return int(a + np.argmin(np.abs(mz[a:b] - target)))


def _blended(mz, h, target, pred, tol, reach, min_frac=EE_BLEND_MIN_FRAC) -> bool:
    """Could a line predicted at `pred` at `target` hide inside a neighbour
    peak? A real line of at least pmin = `min_frac` x pred (the file's own low
    ratio: a weaker line would itself contradict) merged with another line
    within `reach` (one FWHM) is picked at their height-weighted centroid: a
    neighbour of height H at distance d can hold it when d <= tol + (1 - pmin /
    H) x reach. A neighbour as tall as the prediction alone cannot sit far off
    the position (its centroid would be pulled onto it), so a separate line
    several ppm away is no blend."""
    if not np.isfinite(reach) or reach <= 0:
        return False
    pmin = max(float(min_frac), 0.0) * pred
    a, b = np.searchsorted(mz, [target - reach - tol, target + reach + tol])
    for j in range(a, b):
        H = h[j]
        if H < pmin:
            continue
        if abs(mz[j] - target) <= tol + (1.0 - pmin / H) * reach:
            return True
    return False


def _fwhm_fn(resolution):
    """mz -> FWHM (Da): the width model's, else an Orbitrap at EE_FALLBACK_R200."""
    if resolution is not None and hasattr(resolution, "fwhm"):
        def f(m):
            try:
                v = float(resolution.fwhm(float(m)))
            except Exception:  # noqa: BLE001
                v = float("nan")
            return v if np.isfinite(v) and v > 0 else float(m) / (EE_FALLBACK_R200 * (200.0 / float(m)) ** 0.5)
        return f
    return lambda m: float(m) / (EE_FALLBACK_R200 * (200.0 / float(m)) ** 0.5)


class ElementCalibration:
    """One file's detection model for exact-offset isotope lines, fitted on its
    own 13C / 18O lines of committed CHON M0 rows (`element_calibration`).

    `x` (predicted height / local floor), `found` (a line within the window)
    and `ratio` (observed / predicted, NaN where absent) per calibration line
    (blended positions left out); `fit` the logistic detection curve in log x;
    `x_obs` the height / floor where it reaches EE_DETECT, kept when >=
    EE_MIN_OBS lines lie at or above it and >= EE_DETECT_EMP of them were found
    (None: uncalibrated, the predicate never contradicts)."""

    def __init__(self, x, found, ratio, window_ppm):
        self.x = np.asarray(x, dtype=float)
        self.found = np.asarray(found, dtype=bool)
        self.ratio = np.asarray(ratio, dtype=float)
        self.window_ppm = float(window_ppm)
        self.x_obs = None
        self.fit = None
        if len(self.x) >= EE_MIN_CAL:
            self.fit = _logistic_fit(np.log(np.clip(self.x, 1e-3, None)), self.found)
        if self.fit is not None:
            a, b = self.fit
            x98 = float(np.exp((np.log(EE_DETECT / (1.0 - EE_DETECT)) - a) / b))
            above = self.x >= x98
            # the fit must stand on lines: enough of them at or above the level,
            # and found there (a curve that never reaches the level in the data,
            # or a plateau under it, leaves the file uncalibrated)
            if (np.isfinite(x98) and above.sum() >= EE_MIN_OBS
                    and self.found[above].mean() >= EE_DETECT_EMP):
                self.x_obs = x98

    @property
    def n(self) -> int:
        return int(len(self.x))

    @property
    def calibrated(self) -> bool:
        return self.x_obs is not None

    def observable(self, x: float) -> bool:
        return self.x_obs is not None and np.isfinite(x) and x >= self.x_obs

    def low_ratio(self, x: float) -> float:
        """The file's EE_LOW_Q-th percentile observed / predicted ratio of the
        calibration lines found near this height (x/2..2x, widened to x/4..4x
        and x/8..8x until EE_MIN_CAL lines are in, then every line found above
        the floor, then every line found) -- EE_LOW_DEFAULT when uncalibrated."""
        if not self.calibrated:
            return EE_LOW_DEFAULT
        r_ok = self.found & np.isfinite(self.ratio)
        sels = [r_ok & (self.x >= x / f) & (self.x <= x * f) for f in (2.0, 4.0, 8.0)]
        sels += [r_ok & (self.x >= 1.0), r_ok]
        for sel in sels:
            if sel.sum() >= EE_MIN_CAL or sel is sels[-1]:
                if not sel.any():
                    return EE_LOW_DEFAULT
                return float(min(1.0, np.percentile(self.ratio[sel], EE_LOW_Q)))
        return EE_LOW_DEFAULT

    def describe(self) -> str:
        if not self.n:
            return "no calibration lines"
        if not self.calibrated:
            return (f"{self.n} calibration lines" + (f" (under {EE_MIN_CAL})" if self.n < EE_MIN_CAL else
                    f", detection never reaches {EE_DETECT:g} over >= {EE_MIN_OBS} of them")
                    + ": no contradiction")
        return (f"{self.n} calibration lines, observable from {self.x_obs:.1f}x the local floor "
                f"(window {self.window_ppm:.2f} ppm)")


def _logistic_fit(t: np.ndarray, y: np.ndarray, *, iters: int = 50, ridge: float = 1e-6):
    """(a, b) of P(found) = 1 / (1 + exp(-(a + b t))) by maximum likelihood
    (Newton / IRLS); None when it does not converge to a rising curve or the
    lines are all found / all missing."""
    y = np.asarray(y, dtype=float)
    t = np.asarray(t, dtype=float)
    if len(y) < 2 or y.min() == y.max():
        return None
    X = np.column_stack([np.ones_like(t), t])
    w = np.zeros(2)
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(X @ w, -30, 30)))
        g = X.T @ (y - p) - ridge * w
        Hm = (X * (p * (1 - p))[:, None]).T @ X + ridge * np.eye(2)
        try:
            step = np.linalg.solve(Hm, g)
        except np.linalg.LinAlgError:
            return None
        w = w + step
        if np.max(np.abs(step)) < 1e-8:
            break
    if not np.all(np.isfinite(w)) or w[1] <= 0:
        return None
    return float(w[0]), float(w[1])


#: the prefix the re-arbitration stage (passes.rearbitrate_offcal_degenerate)
#: puts before the method of a row it re-reads
REARB_PREFIX = "rearb<-"


def base_method(method) -> str:
    """The proposer's own method of a ledger row: `method` with every leading
    REARB_PREFIX dropped (a re-arbitrated 'rearb<-known:...' is still the
    curated row, 'rearb<-certified:...' still pass 7's)."""
    m = "" if method is None or (isinstance(method, float) and np.isnan(method)) else str(method)
    while m.startswith(REARB_PREFIX):
        m = m[len(REARB_PREFIX):]
    return m


def _ion_body(ion) -> dict:
    """Element counts of an ion formula string (its charge sign dropped)."""
    if not isinstance(ion, str) or not ion.strip():
        return {}
    from peaky.chem import chemistry as C
    return C.parse_formula(ion.strip().rstrip("+-").rstrip("."))


def element_calibration(ledger: pd.DataFrame, *, window_ppm: float, resolution=None,
                        lines=None, blend_lines=None) -> ElementCalibration:
    """The file's ElementCalibration: every committed M0 whose ION is C/H/N/O
    only (not Low / Suspect / tied, not an ion-only row) contributes its 13C
    line (and its 18O line), predicted at first order from the parent's height;
    each line's predicted height over the local floor, whether a peak sits
    within `window_ppm` of its exact offset, and the observed / predicted ratio.
    A position a neighbour could blend (`_blended`) is left out, as the
    predicate leaves it out. `lines` are the lines read (default: every picked
    peak but the artifact rows), `blend_lines` the peaks the blend guard reads
    (default: every picked peak -- an artifact at a position makes it
    untestable rather than empty)."""
    mz, h = lines if lines is not None else _peak_lines(ledger, artifacts=False)
    bmz, bh = blend_lines if blend_lines is not None else _peak_lines(ledger)
    fwhm = _fwhm_fn(resolution)
    recs = _calibration_lines(ledger, mz, h, window_ppm)
    # pass 1: blends judged at EE_BLEND_MIN_FRAC; pass 2 at the file's own low ratio
    cal = None
    for _ in range(2):
        xs, found, ratio = [], [], []
        for pred, tgt, fl, j in recs:
            x = pred / fl
            if j is None:
                frac = EE_BLEND_MIN_FRAC if cal is None else cal.low_ratio(x)
                if _blended(bmz, bh, tgt, pred, tgt * window_ppm * 1e-6, fwhm(tgt), min_frac=frac):
                    continue
            xs.append(x)
            found.append(j is not None)
            ratio.append(h[j] / pred if j is not None else np.nan)
        cal = ElementCalibration(xs, found, ratio, window_ppm)
    return cal


def _calibration_lines(ledger, mz, h, window_ppm) -> list:
    """(pred, target, floor, nearest index or None) of every calibration line
    (`element_calibration`)."""
    out = []
    if ledger is not None and len(ledger) and "role" in ledger.columns and len(mz):
        m0 = ledger[ledger["role"].astype(str) == "M0"]
        for _, r in m0.iterrows():
            if base_method(r.get("method")).startswith("ion_only:"):
                continue
            conf = r.get("confidence")
            if isinstance(conf, str) and conf.startswith(("Low", "Suspect")):
                continue
            t = r.get("tied")
            if (isinstance(t, (bool, np.bool_)) and bool(t)) or (isinstance(t, str) and t.strip().lower() == "true"):
                continue
            ion = _ion_body(r.get("ion_formula"))
            if not ion or set(ion) - {"C", "H", "N", "O"} or ion.get("C", 0) < 1:
                continue
            try:
                m00, h0 = float(r["mz"]), float(r["height"])
            except (TypeError, ValueError):
                continue
            if not (np.isfinite(m00) and np.isfinite(h0) and h0 > 0):
                continue
            for shift, per, el, _lab in EE_CAL_LINES:
                nel = int(ion.get(el, 0))
                if nel <= 0:
                    continue
                pred = h0 * per * nel
                tgt = m00 + shift
                fl = local_floor(mz, h, tgt)
                if not np.isfinite(fl) or fl <= 0:
                    continue
                out.append((pred, tgt, fl, _nearest(mz, tgt, tgt * window_ppm * 1e-6)))
    return out


class EvidenceContext:
    """What the predicate reads of one file: its class, its peak lines, the
    window, the calibration, the width model; on a TOF the ion-M+2 test's floor,
    window and peak width (`heavy_line_verdict`; `tof_fwhm` None: the run has no
    width model, and the TOF test does not run -- tiers.tof_m2_verdicts)."""

    def __init__(self, klass, mz, h, window_ppm, cal, fwhm, tof_floor=None, tof_win_ppm=None,
                 tof_fwhm=None, blend_mz=None, blend_h=None):
        self.klass = klass
        self.mz, self.h = mz, h
        # the blend guard's peaks: every picked peak, the artifact rows included
        self.blend_mz = mz if blend_mz is None else blend_mz
        self.blend_h = h if blend_h is None else blend_h
        self.window_ppm = window_ppm
        self.cal = cal
        self.fwhm = fwhm
        self.tof_floor = tof_floor
        self.tof_win_ppm = tof_win_ppm
        self.tof_fwhm = tof_fwhm

    def describe(self) -> str:
        if self.klass == "orbitrap":
            return f"Orbitrap-class, exact offsets within {self.window_ppm:.2f} ppm; {self.cal.describe()}"
        if self.klass == "tof":
            off = ("no detection floor" if self.tof_floor is None
                   else "no width model" if self.tof_fwhm is None else None)
            return ("TOF-class: Br / Cl by the ion's own M+2 line" + (f" ({off}: off)" if off else "")
                    + "; no S / Si verdicts")
        return "instrument class unknown: no isotope verdicts"


def evidence_context(ledger: pd.DataFrame, *, klass: str | None, cal_sigma=None, resolution=None,
                     tof_floor=None, tof_win_ppm=None, tof_fwhm=None) -> EvidenceContext:
    """The EvidenceContext of one file's ledger (the predicate's per-file
    inputs, built once): on an Orbitrap-class run its calibration too. On a TOF
    `tof_fwhm` (mz -> FWHM, the run's width model: plausibility passes the
    run's Resolution through tiers._fwhm_at, as tof_m2_verdicts does) sizes the
    M+2 test's blend guards; without it a TOF verdict is 'unobservable' (an
    Orbitrap-width fallback would read a TOF's blended M+2 line as absent).
    On an Orbitrap-class run an artifact row (a side lobe the per-file guard
    marked, cleanup's ringing) is no line: it confirms nothing and sets no
    local floor, but the blend guard still reads it (an artifact on a
    predicted position leaves it untestable, never empty) -- REQ's Orbitrap
    rule. The TOF M+2 test reads every peak, as REQ's TOF branch does: its
    blend guards read the same list, so leaving a ringing row out there would
    turn a blend into an absence."""
    mz, h = _peak_lines(ledger, artifacts=klass != "orbitrap")
    bmz, bh = _peak_lines(ledger)
    win = element_window_ppm(cal_sigma)
    fwhm = _fwhm_fn(resolution)
    cal = (element_calibration(ledger, window_ppm=win, resolution=resolution, lines=(mz, h),
                               blend_lines=(bmz, bh))
           if klass == "orbitrap" else ElementCalibration([], [], [], win))
    return EvidenceContext(klass, mz, h, win, cal, fwhm, tof_floor, tof_win_ppm, tof_fwhm,
                           blend_mz=bmz, blend_h=bh)


def _log_fit(obs: float, pred: float) -> float:
    return abs(np.log(max(obs, 1e-12) / max(pred, 1e-12)))


def element_evidence(ctx: EvidenceContext, mz0: float, h0: float, neutral_counts: dict | None,
                     ion_counts: dict, element: str) -> dict:
    """Does the peak list support `element` in this reading? The parent line
    (`mz0`, `h0`: the ion's all-light line) predicts each diagnostic line of
    the element at first order: h0 x per-atom ratio x the ION's atom count, at
    the exact offset. `neutral_counts` names what the neutral carries (None: the
    question is the ion's own line, the reagent's halogen included).

    Orbitrap class: a line is seen within ctx.window_ppm at an observed /
    predicted ratio in [the file's low ratio at that height, EE_RATIO_HIGH];
    'confirmed' when any line of the element is seen; 'contradicted' when none
    is and one the file's calibration calls observable is absent (no peak in
    the window, none that could hold it as a blend) or low; else
    'unobservable'. A halogen the reagent adduct also brings (the ion counts
    more than the neutral) is confirmed only when the observed ratio fits the
    full ion count better than the reagent's alone.
    TOF class: Br / Cl by the ion's own M+2 line (`heavy_line_verdict`: seen ->
    confirmed, absent -> contradicted; blended or dim -> unobservable), and only
    with a detection floor and the run's width model (ctx.tof_fwhm); no S / Si
    verdicts. Unknown class: unobservable.

    Returns {'verdict', 'element', 'lines': [per-line records], 'why'}."""
    el = element
    n_ion = int((ion_counts or {}).get(el, 0) or 0)
    n_neu = n_ion if neutral_counts is None else int(neutral_counts.get(el, 0) or 0)
    out = {"verdict": EE_UNOBSERVABLE, "element": el, "lines": [], "why": ""}
    if el not in TWIN_LINES or n_ion <= 0 or n_neu <= 0:
        out["why"] = f"no {el} to test"
        return out
    try:
        mz0, h0 = float(mz0), float(h0)
    except (TypeError, ValueError):
        out["why"] = "no parent line"
        return out
    if not (np.isfinite(mz0) and np.isfinite(h0) and h0 > 0):
        out["why"] = "no parent line"
        return out
    n_reag = n_ion - n_neu
    if ctx is None or ctx.klass not in ("orbitrap", "tof"):
        out["why"] = "instrument class unknown: no isotope verdict"
        return out
    if ctx.klass == "tof":
        if el not in HEAVY_LINE_ELEMENTS or ctx.tof_floor is None or ctx.tof_fwhm is None:
            out["why"] = (f"TOF-class: no exact-offset {el} line" if el not in HEAVY_LINE_ELEMENTS
                          else "TOF-class: no detection floor" if ctx.tof_floor is None
                          else "TOF-class: no width model (the blend guards need the peak width)")
            return out
        ratio, shift, label = heavy_line_prediction(ion_counts)
        if ratio <= 0:
            out["why"] = "no M+2 prediction"
            return out
        win = ctx.tof_win_ppm if ctx.tof_win_ppm is not None else HEAVY_LINE_MIN_PPM
        v = heavy_line_verdict(ctx.mz, ctx.h, mz0, h0, ratio, shift, ctx.tof_floor, win_ppm=win,
                               fwhm=float(ctx.tof_fwhm(mz0 + shift)))
        st = str(v["status"][0])
        rec = {"label": label, "pred": float(v["pred"][0]), "obs": float(v["obs"][0]), "status": st}
        out["lines"].append(rec)
        if st == HL_SEEN and n_reag > 0:
            rest = dict(ion_counts)
            rest[el] = n_reag
            r_r = heavy_line_prediction(rest)[0]
            obs = float(v["obs"][0])
            if r_r > 0 and _log_fit(obs, r_r * h0) <= _log_fit(obs, ratio * h0):
                out["why"] = (f"the ion's M+2 line at {obs / h0:.2f}x fits the reagent's {el} alone "
                              f"({r_r:.2f}x) as well as the full ion ({ratio:.2f}x)")
                return out
        if st == HL_SEEN:
            out["verdict"] = EE_CONFIRMED
            out["why"] = f"the ion's M+2 ({label}) line seen at {float(v['obs'][0]) / h0:.2f}x (TOF)"
        elif st == HL_ABSENT:
            out["verdict"] = EE_CONTRADICTED
            obs = rec["obs"]
            # a picked line inside the window, under the fraction a sighting needs,
            # is LOW, not absent: say so with its share of the prediction
            found = (f"low ({obs / rec['pred']:.2f} of prediction) within {win:g} ppm, under the "
                     f"{HEAVY_LINE_FRAC:g} a sighting needs" if obs > 0 and rec["pred"] > 0
                     else f"absent within {win:g} ppm")
            out["why"] = (f"the ion's M+2 ({label}) line, predicted at {ratio:.2f}x ({_cps(rec['pred'])} cps), "
                          f"{found} (TOF)")
        else:
            out["why"] = f"the ion's M+2 ({label}) line {st} (TOF)"
        return out
    # Orbitrap class: the exact offsets
    cal = ctx.cal
    seen, contra, notes = [], [], []
    for delta, per, label in TWIN_LINES[el]:
        pred = h0 * per * n_ion
        tgt = mz0 + delta
        fl = local_floor(ctx.mz, ctx.h, tgt)
        x = pred / fl if np.isfinite(fl) and fl > 0 else float("nan")
        low = cal.low_ratio(x) if np.isfinite(x) else EE_LOW_DEFAULT
        obs_ok = bool(np.isfinite(x) and cal.observable(x))
        tol = tgt * ctx.window_ppm * 1e-6
        j = _nearest(ctx.mz, tgt, tol)
        rec = {"label": label, "pred": pred, "floor": fl, "x": x, "observable": obs_ok, "low": low,
               "obs_ratio": None, "status": ""}
        if j is not None:
            r = float(ctx.h[j]) / pred
            rec["obs_ratio"] = r
            if r > EE_RATIO_HIGH:
                rec["status"] = "high"
            elif r < low:
                rec["status"] = "low"
            else:
                rec["status"] = "seen"
                if n_reag > 0:
                    pr = h0 * per * n_reag
                    if _log_fit(float(ctx.h[j]), pr) <= _log_fit(float(ctx.h[j]), pred):
                        rec["status"] = "reagent"
        else:
            rec["status"] = ("blended" if _blended(ctx.blend_mz, ctx.blend_h, tgt, pred, tol, ctx.fwhm(tgt),
                                                   min_frac=low)
                             else "absent")
        out["lines"].append(rec)
        desc = (f"{label} predicted {_cps(pred)} cps ({x:.1f}x the {fl:.1f}-cps floor)" if np.isfinite(x)
                else f"{label} predicted {_cps(pred)} cps")
        if rec["status"] == "seen":
            seen.append(f"{label} line at {rec['obs_ratio']:.2f}x its prediction")
        elif rec["status"] in ("absent", "low") and obs_ok:
            contra.append(desc + (f", no line within {ctx.window_ppm:.2f} ppm" if rec["status"] == "absent"
                                  else f", line low ({rec['obs_ratio']:.2f} of prediction, under the "
                                       f"file's {low:.2f}) within {ctx.window_ppm:.2f} ppm"))
        else:
            why = {"reagent": "fits the reagent's halogen alone", "high": "a taller line holds the position",
                   "blended": "a neighbour could hold it", "absent": "under the file's observable level",
                   "low": "under the file's observable level"}[rec["status"]]
            notes.append(f"{desc}: {why}")
    if seen:
        out["verdict"] = EE_CONFIRMED
        out["why"] = "; ".join(seen)
    elif contra:
        out["verdict"] = EE_CONTRADICTED
        out["why"] = "; ".join(contra)
    else:
        out["why"] = "; ".join(notes) + ("" if cal.calibrated else f" ({cal.describe()})")
    return out
