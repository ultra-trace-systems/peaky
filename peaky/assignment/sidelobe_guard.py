"""The Orbitrap same-spectrum side-lobe guard: a per-file step BEFORE pass 0.

A weak entry of an Orbitrap peak list one to five line widths beside a line at
least 50x brighter, narrower than a line or at the lobe's offset, is not a line
of the profile (chem/sidelobes.py has the rule and what it was measured on).
Left in the ledger it is an assignable peak, and the passes give it a formula:
an exotic ion with nothing behind it. This step marks such rows 'artifact' (with
a reason naming the parent, the ratio and the offset) and LOCKS them, so no
pass commits them and no isotope sweep attaches them; it runs before the C42
re-run snapshot, so a re-run from pass 0 keeps it.

Relation to the other side-lobe rules: `cleanup.flag_ringing_artifacts` runs
after pass 6 on what is still unexplained and needs a saturating (>= 50 000 cps)
parent, a TOF-sized gate a 100-microscan Orbitrap list never reaches; the
batch-level `timeseries.flag_sidelobe_channels` needs >= 20 paired spectra.
This one is relative (ratio and widths of the same spectrum), so it reaches the
dim parents those miss -- and is gated to the Orbitrap class for that reason:
the widths and the lobe offsets it reads are an Orbitrap peak list's.

It never runs on a TOF (cfg.instrument_class must be 'orbitrap'; None = off),
on trace-first's synthetic sample (batch-mean heights are not one spectrum's),
without a width model or a peak area, or when the file's own widths cannot
calibrate the model (chem/sidelobes.py `width_calibration`). A measured model
with too few bright peaks to calibrate runs on the geometric signatures alone.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from peaky.assignment import ledger as L
from peaky.chem import sidelobes as SL

__version__ = "1.2.0"   # + lobe_mask: the element-evidence predicate reads no line off a marked side lobe


#: the start of a flagged row's commentary: what `lobe_mask` reads back
LOBE_MARK = "Orbitrap side lobe"


def lobe_mask(ledger: pd.DataFrame) -> np.ndarray:
    """True on the rows this guard marked (role 'artifact', commentary starting
    LOBE_MARK): peaks the profile does not hold. The element-evidence
    predicate (assignment/satellites.py) reads no line off them."""
    if ledger is None or not len(ledger) or "role" not in ledger.columns or "commentary" not in ledger.columns:
        return np.zeros(0 if ledger is None else len(ledger), dtype=bool)
    art = ledger["role"].astype(str).to_numpy() == L.ROLE_ARTIFACT
    com = ledger["commentary"].map(lambda v: isinstance(v, str) and v.startswith(LOBE_MARK)).to_numpy(dtype=bool)
    return art & com


def _skip(reason: str, log) -> dict:
    log(f"[sidelobe] guard skipped: {reason}")
    return {"flagged": 0, "skipped": reason}


def flag_orbitrap_sidelobes(ledger: pd.DataFrame, width_model, cfg, *, log=print) -> dict:
    """Mark and lock this file's same-spectrum side lobes (module docstring).

    Every row of the ledger is a possible parent; only an unexplained, unlocked
    row is ever marked. Returns {'flagged': n, 'skipped': None or the reason,
    'by_signature': {...}, 'n_mirror', 'n_no_signature', 'n_exempt' (fine-structure
    names and '<isotope> line' for an isotope line of a brighter line),
    'signatures': the signatures tested (no 'narrow' when the calibration is not
    measured), 'calibration': {...}} -- the run's per-file stats carry it as
    'sidelobe_guard'."""
    if not getattr(cfg, "sidelobe_guard", True):
        return _skip("off (sidelobe_guard=False)", log)
    if getattr(cfg, "trace_sample", False):
        return _skip("trace sample (batch-mean heights, not one spectrum)", log)
    klass = getattr(cfg, "instrument_class", None)
    if klass != "orbitrap":
        return _skip(f"not an Orbitrap (instrument class {klass or 'unknown'})", log)
    if width_model is None:
        return _skip("no width model", log)
    if "area" not in ledger.columns or not pd.to_numeric(ledger["area"], errors="coerce").notna().any():
        return _skip("no area column (the observed widths cannot be measured)", log)
    res = SL.sidelobe_parents(
        ledger["mz"].to_numpy(dtype=float),
        pd.to_numeric(ledger["height"], errors="coerce").to_numpy(dtype=float),
        pd.to_numeric(ledger["area"], errors="coerce").to_numpy(dtype=float),
        width_model, ratio=float(cfg.sidelobe_ratio),
        band_fwhm=tuple(cfg.sidelobe_band_fwhm), narrow=float(cfg.sidelobe_narrow))
    cal = res.get("calibration")
    cal_out = None if cal is None else {
        "ratio": round(float(cal["ratio"]), 4), "n": int(cal["n"]),
        "measured": bool(cal["measured"]),
        "slope": None if not np.isfinite(cal["slope"]) else round(float(cal["slope"]), 4)}
    if res["skipped"]:
        out = _skip(res["skipped"], log)
        out["calibration"] = cal_out
        return out
    pids, mzs, hts = ledger["peak_id"].tolist(), ledger["mz"].tolist(), ledger["height"].tolist()
    role = ledger["role"].astype(str).to_numpy()
    locked = ledger["locked"].astype(bool).to_numpy()
    flagged, by_sig = [], {}
    for i in np.flatnonzero(res["parent"] >= 0):
        if role[i] != L.ROLE_UNEXPLAINED or locked[i]:
            continue
        j = int(res["parent"][i])
        sig = res["signature"][i]
        L.mark_artifact(
            ledger, pids[i],
            f"{LOBE_MARK} of m/z {mzs[j]:.4f} ({hts[j] / hts[i]:.0f}x brighter, "
            f"{res['offset_fwhm'][i]:+.2f} FWHM = {res['offset_mda'][i]:+.1f} mDa; "
            f"{sig.replace('+', ', ')}): not a line of the profile, excluded from assignment")
        flagged.append(pids[i])
        for s in sig.split("+"):
            by_sig[s] = by_sig.get(s, 0) + 1
    L.lock_peaks(ledger, flagged)
    n_exempt = {k: int(v) for k, v in res["n_exempt"].items()}
    signatures = list(res["signatures"])
    cal_note = (f"width/model {cal_out['ratio']:.2f} over {cal_out['n']} bright peaks"
                if cal_out["measured"] else
                f"UNCALIBRATED: {cal_out['n']} bright peaks (< {SL.MIN_CAL_PEAKS}), "
                f"the model's widths trusted and the 'narrow' signature off")
    log(f"[sidelobe] {len(flagged)} Orbitrap side lobes marked artifact and locked "
        f"(ratio >= {cfg.sidelobe_ratio:g}, {cfg.sidelobe_band_fwhm[0]:g}-"
        f"{cfg.sidelobe_band_fwhm[1]:g} parent FWHM; {cal_note}; tested "
        f"{'/'.join(signatures)}); signatures {by_sig}; spared "
        f"{res['n_no_signature']} with no artifact signature"
        + (f", {sum(n_exempt.values())} isotope lines (fine structure, or on an isotopologue offset "
           f"of a brighter line) {n_exempt}" if n_exempt else ""))
    return {"flagged": len(flagged), "skipped": None, "by_signature": by_sig,
            "n_mirror": int(res["n_mirror"]), "n_no_signature": int(res["n_no_signature"]),
            "n_exempt": n_exempt, "signatures": signatures, "calibration": cal_out}
