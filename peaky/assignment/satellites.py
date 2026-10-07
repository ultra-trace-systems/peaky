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
                 prefix: str = "", masked_by: str | None = None) -> dict:
    """Judge `counts`' diagnostic line(s) of `element` (default: the first of
    DEFAULT_ELEMENTS it carries) on the ledger peak `pid`.

    Returns {'verdict': 'refuted' | 'deferred', 'kind': 'refuted' | 'supported'
    | 'untestable', 'twin': element or None, 'why': the reason with this file's
    numbers (after `prefix`), 'summary': the same without them, 'lines': one
    record per line tested}. `refuted` when a line predicted at >=
    TWIN_REFUTE_X_FLOOR x `floor` is absent within TWIN_PPM or sits under
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
        j = peak_near(ledger["mz"], m0 + delta, ppm=TWIN_PPM)
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
