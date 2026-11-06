"""The isotopologue hold and the satellite note of a low-file-count promotion (a merged-row gate).

A merged reading that reaches Assigned from one or two files has only those
files' own evidence behind it. Two by-products of a brighter line can look
like such a reading:

(b) an ISOTOPOLOGUE -- HELD at Candidate: a brighter Assigned M0 row of the
    same file whose ion carries C, N, O, S or Cl puts its 13C / 15N / 18O / 34S
    / 37Cl line on the row (within ISO_PPM ppm plus the file's calibrated
    sigma), and the row reads at most ISO_HEIGHT_X times that line's predicted
    height. Held when this holds in EVERY file the reading is Assigned in.
(a) a SATELLITE -- a NOTE only (SATELLITE_MARK; the tier is untouched): on an
    Orbitrap-class batch a line at least SAT_BRIGHT_X times brighter within
    +-SAT_DMDA mDa that the row TRACKS: over the batch's spectra holding both
    lines, the row's height stays within TRACK_X of its median ratio to the
    partner in at least TRACK_SHARE of them (`track_verdict`); with fewer than
    TRACK_MIN_N such spectra the files' own ratios must agree within TRACK_X,
    and a single file with no series is judged on the geometry alone. Noted
    when every file shows such a satellite partner or an isotopologue parent
    (and the reading is not held). Whether the +3.9 mDa partners of uronium
    batches are artefacts is open (their offset is constant over m/z and equals
    a formula spacing), so the clause says so and decides nothing.

A reading with its own 13C line in each of its files -- a line at +1.00336 Da
at OWN_13C_BAND times the share its own carbon count predicts -- is an M0 of its
own: neither held nor noted. A held row keeps its reading, goes to Candidate
and says why (HOLD_MARK); a noted row keeps reading and tier (SATELLITE_MARK).
`table` lists every held or noted file with its partner (`action`).

Pure: reads the merged ledger, the per-file ledgers as their runs wrote them
and the batch time series (sample_item_id, mz, height); mutates only the
merged rows it holds or notes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from peaky.chem import chemistry as C
from peaky.chem import isotopes as ISO

__version__ = "0.2.0"   # the satellite clause is a note only; the hold is the isotopologue clause

#: the most files a reading may reach Assigned from and still be tested
MAX_FILES = 2
#: (a) the satellite window (mDa) and how much brighter its partner must be
SAT_DMDA = 4.0
SAT_BRIGHT_X = 10.0
#: (a) tracking: the factor the row/partner height ratio stays within, the share of
#: spectra that must hold it, and the fewest spectra holding both lines the series test needs
TRACK_X = 2.0
TRACK_SHARE = 0.8
TRACK_MIN_N = 3
#: the window (ppm) a line is read in on the batch's series
TRACK_PPM = 3.0
#: (b) the isotopologue window (ppm, plus the file's calibrated sigma) and the height bar
ISO_PPM = 1.5
ISO_HEIGHT_X = 1.5
#: the own-13C exemption: the line's height over the share the row's carbon count predicts
OWN_13C_BAND = (0.5, 2.0)
#: the single-substitution lines (b) reads: label -> (spacing Da, element, per-atom ratio)
ISO_LINES = {
    "13C": (ISO.D_13C, "C", ISO.R_13C_PER_C),
    "15N": (ISO.D_15N, "N", ISO.R_15N_PER_N),
    "18O": (ISO.D_18O, "O", ISO.R_18O_PER_O),
    "34S": (ISO.D_34S, "S", ISO.R_34S_PER_S),
    "37Cl": (ISO.D_37CL, "Cl", ISO.R_37CL_PER_CL),
}
HOLD_MARK = "isotopologue risk"
SATELLITE_MARK = "satellite risk"
TABLE_COLUMNS = ["mz", "neutral_formula", "adduct", "n_files_assigned", "action", "sample_item_id", "peak_id",
                 "row_mz", "row_height", "rule", "partner_peak_id", "partner_mz", "partner_height",
                 "partner_reading", "d_mDa", "ratio", "track_n", "track_share"]


def empty_table() -> pd.DataFrame:
    """The hold table with no row (its columns)."""
    return pd.DataFrame(columns=TABLE_COLUMNS)


def _num(v) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return x


def _s(v) -> str:
    if v is None:
        return ""
    try:
        if pd.isna(v):
            return ""
    except (TypeError, ValueError):
        pass
    return str(v)


def ion_counts(neutral, adduct) -> dict | None:
    """The ION's element counts of a (neutral, adduct) reading (tiers._ion_counts), or None."""
    from peaky.assignment import tiers as T
    try:
        return T._ion_counts(_s(neutral), _s(adduct))
    except Exception:          # noqa: BLE001 -- an unparseable reading has no ion
        return None


def ion_key(neutral, adduct) -> str:
    """The ion a reading names (assign_batch._ion_key: two labels of one ion share it)."""
    from peaky.batch.assign_batch import _ion_key
    return _ion_key(_s(neutral), _s(adduct))


def _reading(row) -> str:
    nf, ad = _s(row.get("neutral_formula")), _s(row.get("adduct"))
    role = _s(row.get("role"))
    if nf and ad:
        return f"{nf} {ad}" + (f" ({role})" if role and role != "M0" else "")
    return role or "unexplained"


def track_verdict(ts: pd.DataFrame | None, mz_row: float, mz_partner: float, *,
                  ppm: float = TRACK_PPM) -> tuple[int, float]:
    """(n, share): over the series' spectra holding both lines (each read as the
    tallest peak within `ppm` of its m/z), how many there are and the share whose
    row / partner height ratio lies within TRACK_X of the median ratio. (0, nan)
    without a series."""
    if ts is None or not len(ts) or not {"sample_item_id", "mz", "height"} <= set(ts.columns):
        return 0, float("nan")
    if not ts["mz"].is_monotonic_increasing:
        ts = ts.sort_values("mz", kind="mergesort")
    mz = ts["mz"].to_numpy(dtype=float)

    def line(m):
        w = m * ppm * 1e-6
        lo, hi = np.searchsorted(mz, m - w, "left"), np.searchsorted(mz, m + w, "right")
        sub = ts.iloc[lo:hi][["sample_item_id", "height"]]
        if not len(sub):
            return pd.Series(dtype=float)
        return pd.to_numeric(sub["height"], errors="coerce").groupby(sub["sample_item_id"]).max()

    a, b = line(mz_row), line(mz_partner)
    j = pd.concat([a.rename("r"), b.rename("p")], axis=1, join="inner").dropna()
    j = j[(j["r"] > 0) & (j["p"] > 0)]
    if not len(j):
        return 0, float("nan")
    lr = np.log(j["r"].to_numpy() / j["p"].to_numpy())
    ok = np.abs(lr - np.median(lr)) <= np.log(TRACK_X)
    return int(len(j)), float(ok.mean())


def _own_13c(led: pd.DataFrame, row, ion: dict | None, tol_ppm: float) -> bool:
    """The row has its own 13C line: a line at +D_13C within tol_ppm whose height
    over the row's is OWN_13C_BAND x the share its carbon count predicts."""
    nc = int((ion or {}).get("C", 0) or 0)
    h0 = _num(row.get("height"))
    if nc < 1 or not h0 > 0:
        return False
    t = _num(row.get("mz")) + ISO.D_13C
    mz = pd.to_numeric(led["mz"], errors="coerce").to_numpy(dtype=float)
    w = t * tol_ppm * 1e-6
    sel = (mz >= t - w) & (mz <= t + w)
    if not sel.any():
        return False
    h13 = float(pd.to_numeric(led.loc[sel, "height"], errors="coerce").max())
    want = h0 * ISO.R_13C_PER_C * nc
    lo, hi = OWN_13C_BAND
    return bool(want > 0 and lo * want <= h13 <= hi * want)


def _bright_near(led, row) -> pd.DataFrame:
    """The file's lines >= SAT_BRIGHT_X x the row's height within +-SAT_DMDA mDa of it."""
    m, h = _num(row.get("mz")), _num(row.get("height"))
    if not (np.isfinite(m) and h > 0):
        return led.iloc[0:0]
    mz = pd.to_numeric(led["mz"], errors="coerce")
    hh = pd.to_numeric(led["height"], errors="coerce")
    return led[((mz - m).abs() <= SAT_DMDA * 1e-3) & (hh >= SAT_BRIGHT_X * h)
               & (led["peak_id"].astype(str) != _s(row.get("peak_id")))]


def _satellite(led, row, ts, files_rows) -> dict | None:
    """Rule (a) in one file: the brightest line >= SAT_BRIGHT_X brighter within
    +-SAT_DMDA mDa that the row tracks; the record, or None."""
    m, h = _num(row.get("mz")), _num(row.get("height"))
    near = _bright_near(led, row)
    if not len(near):
        return None
    p = near.loc[pd.to_numeric(near["height"], errors="coerce").idxmax()]
    pm, ph = _num(p["mz"]), _num(p["height"])
    n, share = track_verdict(ts, m, pm)
    if n >= TRACK_MIN_N:
        if not share >= TRACK_SHARE:
            return None
    elif len(files_rows) > 1:
        # too few spectra hold both lines: the files' own ratios must agree
        rs = [_num(r["height"]) / _num(r["_partner_h"]) for r in files_rows
              if _num(r.get("_partner_h")) > 0]
        if len(rs) == len(files_rows) and max(rs) / min(rs) > TRACK_X:
            return None
    return dict(rule="satellite", partner_peak_id=_s(p.get("peak_id")), partner_mz=pm, partner_height=ph,
                partner_reading=_reading(p), d_mDa=round((m - pm) * 1e3, 3), ratio=round(ph / h, 2),
                track_n=n, track_share=(round(share, 3) if np.isfinite(share) else np.nan))


def _isotopologue(led, row, tol_ppm) -> dict | None:
    """Rule (b) in one file: a brighter Assigned M0 row whose single-substitution
    line falls on the row at <= ISO_HEIGHT_X x its predicted height; the record, or None."""
    m, h = _num(row.get("mz")), _num(row.get("height"))
    if not (np.isfinite(m) and h > 0):
        return None
    m0 = led[(led["role"].astype(str) == "M0") & (led["tier"].astype(str) == "Assigned")
             & (led["peak_id"].astype(str) != _s(row.get("peak_id")))]
    if not len(m0):
        return None
    pmz = pd.to_numeric(m0["mz"], errors="coerce").to_numpy(dtype=float)
    phh = pd.to_numeric(m0["height"], errors="coerce").to_numpy(dtype=float)
    w = m * tol_ppm * 1e-6
    best = None
    for lab, (d, el, per) in ISO_LINES.items():
        hit = np.where((np.abs(pmz + d - m) <= w) & (phh > h))[0]
        for k in hit:
            p = m0.iloc[k]
            ion = ion_counts(p.get("neutral_formula"), p.get("adduct"))
            n_el = int((ion or {}).get(el, 0) or 0)
            if n_el < 1:
                continue
            pred = phh[k] * per * n_el
            if h <= ISO_HEIGHT_X * pred and (best is None or phh[k] > best["partner_height"]):
                best = dict(rule=f"isotopologue {lab}", partner_peak_id=_s(p.get("peak_id")),
                            partner_mz=float(pmz[k]), partner_height=float(phh[k]), partner_reading=_reading(p),
                            d_mDa=round((m - pmz[k] - d) * 1e3, 3), ratio=round(h / pred, 3),
                            track_n=0, track_share=np.nan)
    return best


def hold_low_file_promotions(merged: pd.DataFrame, per_file: dict, ts: pd.DataFrame | None = None, *,
                             tol_ppm: float = 6.0, sigma_ppm: dict | None = None, orbitrap: bool = True,
                             max_files: int = MAX_FILES, log=print) -> tuple[pd.DataFrame, dict]:
    """Hold at Candidate every Assigned merged M0 reading that reaches Assigned
    from <= `max_files` files when in EACH of those files it sits on the
    isotopologue line of a brighter Assigned row (b); note the readings that
    instead (or also) sit beside a tracking satellite partner in each file (a;
    `orbitrap` batches only), their tier untouched; skip a reading with its own
    13C line in every such file (module doc).

    merged: the merged ledger (mz, neutral_formula, adduct, tier, srcs,
    tier_reason; mutated). per_file: {sample id: its ledger as its run wrote it,
    every peak}. ts: the batch time series (sample_item_id, mz, height) or None.
    tol_ppm: the merge window a file's row is matched to the merged row in.
    sigma_ppm: {sample id: the file's calibrated sigma, ppm} (added to ISO_PPM).
    Returns (table, summary): one row per held or noted file (TABLE_COLUMNS) and
    {tested, held, satellite_noted, own_13c_kept}."""
    out = {"tested": 0, "held": 0, "satellite_noted": 0, "own_13c_kept": 0}
    rows = []
    need = {"mz", "neutral_formula", "adduct", "tier"}
    if merged is None or not len(merged) or not need <= set(merged.columns):
        return empty_table(), out
    if "tier_reason" not in merged.columns:
        merged["tier_reason"] = pd.NA
    sigma_ppm = sigma_ppm or {}
    if ts is not None and len(ts) and {"sample_item_id", "mz", "height"} <= set(ts.columns):
        ts = ts[["sample_item_id", "mz", "height"]].dropna(subset=["mz"]).sort_values("mz", kind="mergesort")
    frames = {}
    for sid, led in (per_file or {}).items():
        if led is None or not len(led) or not {"mz", "height", "role", "tier", "peak_id"} <= set(led.columns):
            continue
        f = led.copy()
        f["_ion"] = [ion_key(n, a) if _s(n) and _s(a) else "" for n, a in zip(f["neutral_formula"], f["adduct"])]
        f["_mz"] = pd.to_numeric(f["mz"], errors="coerce")
        frames[str(sid)] = f
    role_ok = merged["role"].astype(str).eq("M0") if "role" in merged.columns else pd.Series(True, index=merged.index)
    for i in merged.index[role_ok & merged["tier"].astype(str).eq("Assigned")]:
        nf, ad = _s(merged.at[i, "neutral_formula"]), _s(merged.at[i, "adduct"])
        if not nf or not ad:
            continue
        key, m = ion_key(nf, ad), _num(merged.at[i, "mz"])
        srcs = [x for x in _s(merged.at[i, "srcs"]).split(",") if x] if "srcs" in merged.columns else list(frames)
        hits = []
        for sid in srcs:
            f = frames.get(sid)
            if f is None:
                continue
            sel = f[(f["role"].astype(str) == "M0") & (f["_ion"] == key)
                    & ((f["_mz"] - m).abs() <= m * tol_ppm * 1e-6)]
            if len(sel):
                r = sel.loc[pd.to_numeric(sel["height"], errors="coerce").idxmax()]
                if str(r["tier"]) == "Assigned":
                    hits.append((sid, r))
        if not hits or len(hits) > max_files:
            continue
        out["tested"] += 1
        ion = ion_counts(nf, ad)
        tols = {sid: ISO_PPM + (max(0.0, _num(sigma_ppm.get(sid))) if np.isfinite(_num(sigma_ppm.get(sid))) else 0.0)
                for sid, _r in hits}
        if all(_own_13c(frames[sid], r, ion, tols[sid]) for sid, r in hits):
            out["own_13c_kept"] += 1           # an M0 of its own in every file
            continue
        # the files' own partner heights, for the tracking fallback of the satellite clause
        files_rows = []
        for sid, r in hits:
            c = _bright_near(frames[sid], r)
            files_rows.append(dict(height=r["height"],
                                   _partner_h=(float(pd.to_numeric(c["height"], errors="coerce").max())
                                               if len(c) else float("nan"))))
        per = []                               # (sid, row, isotopologue record, satellite record) per file
        for sid, r in hits:
            f = frames[sid]
            per.append((sid, r, _isotopologue(f, r, tols[sid]),
                        _satellite(f, r, ts, files_rows) if orbitrap else None))
        if all(iso is not None for _sid, _r, iso, _sat in per):
            action, recs = "held", [(sid, r, iso) for sid, r, iso, _sat in per]
        elif all(iso is not None or sat is not None for _sid, _r, iso, sat in per):
            action, recs = "noted", [(sid, r, iso if iso is not None else sat) for sid, r, iso, sat in per]
        else:
            continue                           # one file without such a partner: the reading stands
        recs = [dict(sample_item_id=sid, peak_id=_s(r["peak_id"]), row_mz=_num(r["mz"]),
                     row_height=_num(r["height"]), **rec) for sid, r, rec in recs]
        why = "; ".join(f"{x['sample_item_id']}: {x['rule']} of {x['partner_mz']:.4f} {x['partner_reading']} "
                        f"({x['d_mDa']:+.2f} mDa, x{x['ratio']:g})" for x in recs)
        if action == "held":
            merged.at[i, "tier"] = "Candidate"
            msg = (f"{HOLD_MARK}: Assigned in {len(hits)} file(s) only, each on the isotopologue line of a "
                   f"brighter row ({why}); held at Candidate")
            out["held"] += 1
        else:
            msg = (f"{SATELLITE_MARK}: Assigned in {len(hits)} file(s) only, each beside a brighter line it "
                   f"tracks or an isotopologue parent ({why}); a note, the tier is untouched")
            out["satellite_noted"] += 1
        prev = _s(merged.at[i, "tier_reason"])
        merged.at[i, "tier_reason"] = f"{prev} | {msg}" if prev else msg
        for x in recs:
            rows.append(dict(mz=m, neutral_formula=nf, adduct=ad, n_files_assigned=len(hits), action=action, **x))
    if out["held"] or out["satellite_noted"]:
        log(f"[sideband] of {out['tested']} readings Assigned in <= {max_files} files: {out['held']} held at "
            f"Candidate on a brighter row's isotopologue line, {out['satellite_noted']} noted beside a satellite "
            f"partner (tier kept); {out['own_13c_kept']} skipped on their own 13C line")
    return pd.DataFrame(rows, columns=TABLE_COLUMNS), out
