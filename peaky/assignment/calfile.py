"""The tier calibration as a FILE: a core pooled over finished runs, read by a batch.

A batch whose files cannot calibrate the tier engine on their own cores
(`tiers.CAL_MIN_N`) and whose own pooled core (assignment/calpool.py) falls
short as well -- a small scan window whose 13C lines sit under the noise edge --
levels nothing. The runs of the same mode over a campaign hold together what no
batch holds alone. Pooling them inside a batch would make the batch depend on
other batches; a batch stays reproducible from its own inputs when the pooled
calibration is an explicit, hashed INPUT instead:

  * ``peaky calibration pool <run dirs...> --out cal.json`` (`pool_runs`):
    the per-file ledgers of finished run dirs of one reagent and polarity, the
    batch pool's core rules (`calpool.core_rows` / `calpool.pool_core`: distinct
    ions isotope-backed in >= POOL_MIN_FILES files, the tallest row per ion,
    >= POOL_MIN_IONS ions spanning >= POOL_MIN_SPAN_FRAC of the committed range)
    over all the runs' files, plus the campaign core's inorganic ions with their
    own isotope line (HSO4- / SO4- on 34S, chlorine species on 37Cl, bromine on
    81Br: `calpool.INORGANIC_ISOTOPES`), and never a reagent identity, a line on
    the reagent library, a row the alias-series hold flagged or a 13C-shadow line
    of the run's label-twin table. Written as JSON (schema SCHEMA, `RECORD_KEYS`).
  * ``peaky batch --tier-calibration cal.json`` (PassConfig.tier_calibration,
    assign_batch.run): a file that calibrates alone keeps its own calibration
    and a file the batch pool calibrates keeps the batch pool's -- byte for byte
    as without the flag. Only a file left uncalibrated by both is lent the
    file's window (calpool.apply_pool: degeneracy, tiers, the degeneracy-keyed
    flags and its own level re-applied in the window, centred on the file's own
    Assigned rows as the batch pool lends it). Its source label (`choose`):
    ``campaign`` when the json's reagent, polarity and scan range match the
    batch; ``lent`` when reagent and polarity match and the scan range differs,
    or -- only with ``--tier-calibration-lent`` -- when the reagent differs on
    the same polarity and instrument class over the same days. Lending never
    happens implicitly across reagents. A lent window passes a mass-accuracy
    transfer check (`transfer`): the receiving batch's own unfloored core
    scatter (its campaign-rule core rows about each file's own centre, >=
    RECV_MIN_ROWS rows of >= RECV_MIN_IONS distinct ions in >= RECV_MIN_FILES
    files) must be at most RECV_X times
    the lending files' median unfloored scatter (the json stores it,
    `width.per_file_raw_sigma`); otherwise the file stays ``none`` and the
    reason is logged. The check asks whether the lender's mass accuracy
    transfers; it is no control of over-grading, which does not move with the
    window's width.
  * ``peaky relevel <run dir> --tier-calibration cal.json --out <dir>``
    (`relevel_run`): the same choice on a finished run, the evidence levels
    only (no re-assignment, no re-tier: the tier stage needs the tier-time
    ledger a finished run does not keep).

The width a json lends (`width.sigma`) is the RUN sigma -- the median of the
pooled runs' per-file sigmas -- when any of their files calibrated alone (a
file-centred window wants one file's width; the pooled sigma also carries the
spread between files), else the pooled core's sigma (`width.basis` 'run' /
'pool'), as calpool lends inside a batch.

The scan range is read off the data (no run records its method's range): the
median over files of a file's lowest peak m/z, and the highest peak m/z of any
file (`series_facts`). The low edge is the method's (a reagent or background ion
sits within a few Da of it); the high edge is where the spectra stop holding
peaks. Two ranges match (`scan_match`) when the low edges agree within
SCAN_LO_DA or SCAN_LO_FRAC and the high edges within SCAN_HI_X.
"""
from __future__ import annotations

import copy
import datetime as _dt
import glob
import hashlib
import json
import os

import numpy as np
import pandas as pd

from peaky.assignment import calpool as CP
from peaky.assignment import ledger as L
from peaky.assignment import masscal as MC
from peaky.assignment import tiers as T

__version__ = "0.2.1"  # the relevel record states tier_stage "not re-run"
#                        0.2.0 the mass-accuracy transfer check replaces the width-spread gate; mixed scan ranges refused

#: the json schema this module writes and reads
SCHEMA = 1
#: the source a pooled json names itself (a batch reads it as campaign or lent)
SOURCE_CAMPAIGN = T.CAL_SOURCE_CAMPAIGN
#: the mass-accuracy transfer check of a lent window (`transfer`): the receiving batch's own unfloored
#: core scatter must be at most this factor of the lending files' median unfloored scatter ...
RECV_X = 1.5
#: ... measured over at least this many core rows ...
RECV_MIN_ROWS = 40
#: ... of at least this many distinct ions (rows of one bright ion repeated over files are not independent) ...
RECV_MIN_IONS = 8
#: ... in at least this many files
RECV_MIN_FILES = 2
#: two scan ranges' low edges match within this many Da ...
SCAN_LO_DA = 3.0
#: ... or this fraction of the low edge
SCAN_LO_FRAC = 0.05
#: ... and their high edges within this ratio band (the batch's over the json's)
SCAN_HI_X = (0.5, 1.5)
#: lending across reagents needs the acquisition days to overlap within this many days
DAY_SLACK = 1
#: the top-level keys of a calibration json
RECORD_KEYS = ("schema", "source", "reagent", "polarity", "scan_range", "scan_mixed", "instrument_class", "days", "runs",
               "core", "width", "usable", "reason", "fitted_at", "peaky_version")
#: the per-file ledger columns the pool reads
_LEDGER_COLS = ["peak_id", "mz", "height", "role", "neutral_formula", "adduct", "ion_formula", "confidence",
                "ppm_error", "parent_peak_id", "iso_label", "isotopologues", "tier", "tier_reason"]


class CalibrationError(ValueError):
    """A calibration json that cannot be built or read as asked."""


# ---------------------------------------------------------------------------
# facts of a run / a batch series
# ---------------------------------------------------------------------------
def series_facts(ts: pd.DataFrame | None) -> dict:
    """{scan_range: [lo, hi], days: [first, last]} of a batch's peak series
    (sample_item_id, mz[, datetime_utc]): lo = the median over samples of a
    sample's lowest peak m/z, hi = the highest peak m/z of any sample; the UTC
    dates of the first and last spectrum (None without a datetime column). Both
    None on an empty or column-less table."""
    out = {"scan_range": None, "days": None}
    if ts is None or not len(ts) or not {"sample_item_id", "mz"} <= set(ts.columns):
        return out
    mz = pd.to_numeric(ts["mz"], errors="coerce")
    g = mz.groupby(ts["sample_item_id"].astype(str)).agg(["min", "max"]).dropna()
    if len(g):
        out["scan_range"] = [round(float(g["min"].median()), 3), round(float(g["max"].max()), 3)]
    if "datetime_utc" in ts.columns:
        t = pd.to_datetime(ts["datetime_utc"], errors="coerce", utc=True).dropna()
        if len(t):
            out["days"] = [t.min().strftime("%Y-%m-%d"), t.max().strftime("%Y-%m-%d")]
    return out


def _ts_path(run_dir: str) -> str | None:
    """The run's batch series parquet: the manifest's input, else the worker copy, else data/*_ts.parquet."""
    try:
        inp = (json.load(open(os.path.join(run_dir, "run_manifest.json"))) or {}).get("input") or {}
        p = inp.get("ts_parquet")
        if p:
            p = p if os.path.isabs(p) else os.path.join(run_dir, p)
            if os.path.isfile(p):
                return p
    except (OSError, ValueError):
        pass
    for p in [os.path.join(run_dir, "per_file", "_batch_ts.parquet"),
              *sorted(glob.glob(os.path.join(run_dir, "data", "*_ts.parquet")))]:
        if os.path.isfile(p):
            return p
    return None


def _read_series(run_dir: str) -> pd.DataFrame | None:
    p = _ts_path(run_dir)
    if p is None:
        return None
    try:
        return pd.read_parquet(p, columns=["sample_item_id", "mz", "datetime_utc"])
    except Exception:  # noqa: BLE001 -- an older series without datetime_utc
        try:
            return pd.read_parquet(p, columns=["sample_item_id", "mz"])
        except Exception:  # noqa: BLE001
            return None


def _polarity_of(reagent: str | None) -> str | None:
    """'-' / '+' of a reagent profile name (None when unknown)."""
    if not reagent:
        return None
    from peaky.chem import profiles as PR
    try:
        return PR.resolve(str(reagent)).polarity
    except Exception:  # noqa: BLE001 -- a profile this process does not know
        return None


def _klass_of(summary: dict) -> str | None:
    from peaky.assignment import evidence as EV
    return EV.instrument(summary.get("resolution"))[0]


def run_facts(run_dir: str) -> dict:
    """What a finished run dir says about itself: reagent, polarity, instrument
    class, scan range and days (`series_facts` of its series), commit, the files
    (their stats), the batch axis model's edge carry, and the per-file
    calibrations of the files that calibrated alone."""
    run_dir = os.path.expanduser(str(run_dir).rstrip("/"))
    sp = os.path.join(run_dir, "batch_summary.json")
    if not os.path.isfile(sp):
        raise CalibrationError(f"{run_dir}: no batch_summary.json (not a finished batch run dir)")
    summ = json.load(open(sp))
    commit = None
    try:
        commit = ((json.load(open(os.path.join(run_dir, "run_manifest.json"))).get("code") or {})
                  .get("git") or {}).get("commit")
    except (OSError, ValueError):
        pass
    sf = series_facts(_read_series(run_dir))
    per_file = list(summ.get("per_file") or [])
    own = [p["degeneracy_cal"] for p in per_file
           if p.get("degeneracy_cal") and p.get("tier_calibration", T.CAL_SOURCE_PER_FILE) == T.CAL_SOURCE_PER_FILE]
    return {"dir": os.path.basename(run_dir), "path": run_dir, "reagent": summ.get("reagent"),
            "polarity": _polarity_of(summ.get("reagent")), "instrument_class": _klass_of(summ),
            "scan_range": sf["scan_range"], "days": sf["days"], "commit": commit, "per_file": per_file,
            "n_files": len(per_file), "edge_carry": (summ.get("mass_axis") or {}).get("edge_carry"),
            "own_cals": [(float(c["mu"]), float(c["sigma"])) for c in own],
            "own_sids": {str(p.get("sample_id")) for p in per_file if p.get("degeneracy_cal")
                         and p.get("tier_calibration", T.CAL_SOURCE_PER_FILE) == T.CAL_SOURCE_PER_FILE}}


def raw_scatter(ledger: pd.DataFrame, *, carry=None) -> float | None:
    """The unfloored scatter of a file's own calibration core: the scaled MAD of
    its `tiers.core_point` ppm errors (on the batch axis model's carried scale)
    about their median, over the errors within 2 ppm of it (all of them when
    fewer than tiers.CAL_MIN_N are), as `tiers.robust_centre` reads them but
    BEFORE the CAL_SIGMA_FLOOR. None when the file holds fewer than
    tiers.CAL_MIN_N core rows (it cannot calibrate alone)."""
    if ledger is None or not len(ledger) or "role" not in ledger.columns:
        return None
    m0 = ledger[ledger["role"] == L.ROLE_M0]
    kids = ledger.loc[ledger["role"] == L.ROLE_ISO, "parent_peak_id"].value_counts()
    pts = [p for p in (T.core_point(r, kids) for _, r in m0.iterrows()) if p is not None]
    if len(pts) < T.CAL_MIN_N:
        return None
    ppm = np.asarray([p[0] for p in pts], dtype=float)
    if carry:
        ppm = ppm - MC.edge_carry_at(carry, np.asarray([p[1] for p in pts], dtype=float))
    srs = pd.Series(ppm)
    core = srs[(srs - srs.median()).abs() <= 2.0]
    if len(core) < T.CAL_MIN_N:
        core = srs
    return float(1.4826 * (core - core.median()).abs().median())


def receiving_scatter(ledgers: dict, *, reagent=None, polarity: str = "negative", carry=None,
                      exclude_mz=()) -> dict:
    """The receiving batch's own unfloored core scatter for the transfer check:
    the campaign-rule core rows (`calpool.core_rows(inorganic=True)`, the reagent
    library and identity, the alias-series hold, `exclude_mz` shadow lines) of
    every file in `ledgers` ({sid: ledger}) that holds >= CENTRE_MIN_ROWS
    Assigned M0 rows, each about its file's own centre (the median ppm error of
    those Assigned rows, as `calpool.lent_calibration` centres the lent window),
    on the carried scale; {sigma (the scaled MAD of the deviations about their
    median), n_rows, n_ions (distinct ions among the rows), n_files}; sigma None
    without rows. `exclude_mz`: the run's 13C-shadow lines (`_shadow_lines`), as
    the lender side (`pool_runs`) excludes them; a batch decides its lending
    before its label-twin table exists, so it has none to pass."""
    masses = _reagent_masses(reagent)
    devs, nf, ions = [], 0, set()
    for sid in sorted(ledgers, key=str):
        led = ledgers[sid]
        if led is None or not len(led) or not {"role", "tier", "ppm_error", "mz"} <= set(led.columns):
            continue
        rows = CP.core_rows(led, polarity=polarity, reagent_masses=masses, sid=str(sid), inorganic=True,
                            exclude_peaks=_alias_held(led), exclude_mz=exclude_mz)
        a = led[(led["role"] == L.ROLE_M0) & (led["tier"].astype(str) == T.TIER_ASSIGNED)]
        ppm = pd.to_numeric(a["ppm_error"], errors="coerce")
        rp = rows["ppm"].astype(float).to_numpy() if len(rows) else np.empty(0)
        if carry:
            ppm = ppm - MC.edge_carry_at(carry, pd.to_numeric(a["mz"], errors="coerce").to_numpy(dtype=float))
            if len(rows):
                rp = rp - MC.edge_carry_at(carry, rows["mz"].to_numpy(dtype=float))
        if len(ppm.dropna()) < CP.CENTRE_MIN_ROWS or not len(rp):
            continue
        devs += list(rp - float(ppm.median()))
        ions |= set(rows["ion"].astype(str))
        nf += 1
    dv = np.asarray(devs, dtype=float)
    sigma = float(1.4826 * np.median(np.abs(dv - np.median(dv)))) if len(dv) else None
    return {"sigma": None if sigma is None else round(sigma, 4), "n_rows": int(len(dv)), "n_ions": int(len(ions)),
            "n_files": int(nf)}


def _shadow_lines(run_dir: str) -> list:
    """The m/z of the lines the run's label-twin table calls a 13C shadow."""
    p = os.path.join(run_dir, "tables", "label_twins.csv")
    if not os.path.isfile(p):
        return []
    try:
        t = pd.read_csv(p, low_memory=False, usecols=lambda c: c in ("mz", "line_verdict"))
    except (OSError, ValueError):
        return []
    if not {"mz", "line_verdict"} <= set(t.columns):
        return []
    return [float(m) for m in pd.to_numeric(t.loc[t["line_verdict"].astype(str) == "shadow", "mz"],
                                            errors="coerce").dropna()]


def _alias_held(led: pd.DataFrame) -> frozenset:
    """Peak ids of the M0 rows whose tier reason carries the alias-series hold's mark."""
    from peaky.assignment.fluorine import SERIES_ALIAS_MARK
    if "tier_reason" not in led.columns:
        return frozenset()
    hit = led["tier_reason"].astype(str).str.contains(SERIES_ALIAS_MARK, regex=False, na=False)
    return frozenset(led.loc[hit & (led["role"] == L.ROLE_M0), "peak_id"])


def _reagent_masses(reagent: str | None) -> list:
    """The ion masses of the reagent's cluster library (none for a reagent without one)."""
    from peaky.chem import profiles as PR
    from peaky.chem import reagents as RG
    try:
        key = RG.reagent_for_adducts(list(PR.resolve(str(reagent)).adducts or []))
    except Exception:  # noqa: BLE001
        return []
    return [m for (_l, m, _f) in RG.build_library(key)] if key else []


def _spread(sigmas) -> dict:
    s = np.asarray([float(x) for x in sigmas], dtype=float)
    if not len(s):
        return {"n": 0, "min": None, "median": None, "max": None}
    return {"n": int(len(s)), "min": round(float(s.min()), 4), "median": round(float(np.median(s)), 4),
            "max": round(float(s.max()), 4)}


# ---------------------------------------------------------------------------
# the pool
# ---------------------------------------------------------------------------
def pool_runs(run_dirs, *, allow_mixed: bool = False, scan_range=None, log=print,
              min_ions: int = CP.POOL_MIN_IONS, min_files: int = CP.POOL_MIN_FILES,
              min_span_frac: float = CP.POOL_MIN_SPAN_FRAC) -> dict:
    """The calibration record (RECORD_KEYS) pooled over finished run dirs
    (module note). Refuses (CalibrationError) runs of different reagents,
    polarities or scan ranges (`scan_match`) unless `allow_mixed` -- mixed scan
    ranges are then recorded as their union (`scan_mixed` True) -- and a run dir
    without a batch summary. The width records the lending files' unfloored core
    scatter (`raw_scatter`, `width.per_file_raw_sigma`), the reference of a lent
    window's transfer check (`transfer`).
    `scan_range` ([lo, hi]) overrides the range read off the runs' series. A
    file present in two runs is read once (the first). Never raises on thin
    data: a core that cannot calibrate gives `usable` False and its `reason`."""
    dirs = [os.path.expanduser(str(d).rstrip("/")) for d in run_dirs]
    if not dirs:
        raise CalibrationError("no run dirs given")
    facts = [run_facts(d) for d in dirs]
    reagents = sorted({str(f["reagent"]) for f in facts})
    pols = sorted({str(f["polarity"]) for f in facts})
    if (len(reagents) > 1 or len(pols) > 1) and not allow_mixed:
        raise CalibrationError(f"the runs mix reagents {reagents} / polarities {pols}; pass --allow-mixed "
                               "to pool them anyway")
    ranges = [f["scan_range"] for f in facts if f["scan_range"]]
    mixed_scan = any(not scan_match(ranges[0], r) or not scan_match(r, ranges[0]) for r in ranges[1:])
    if mixed_scan and not allow_mixed:
        raise CalibrationError("the runs mix scan ranges " + ", ".join(
            f"{f['dir']} {f['scan_range']}" for f in facts) + "; pass --allow-mixed to pool them anyway")
    klasses = sorted({str(f["instrument_class"]) for f in facts})
    parts, seen, ledgers_m0, raw = [], set(), [], {}
    runs = []
    for f in facts:
        masses = _reagent_masses(f["reagent"])
        polarity = "positive" if f["polarity"] == "+" else "negative"
        shadows = _shadow_lines(f["path"])
        files = sorted(glob.glob(os.path.join(f["path"], "per_file", "*_ledger.csv")))
        n_read, n_rows = 0, 0
        for p in files:
            sid = os.path.basename(p)[: -len("_ledger.csv")]
            if sid in seen:
                log(f"[calibration] {sid} is in more than one run; read once")
                continue
            seen.add(sid)
            led = pd.read_csv(p, low_memory=False, usecols=lambda c: c in _LEDGER_COLS)
            n_read += 1
            ledgers_m0.append(led.loc[led["role"] == L.ROLE_M0, ["role", "mz"]])
            if sid in f["own_sids"]:
                # the lending file's unfloored scatter (the transfer check's reference)
                r_ = raw_scatter(led, carry=f["edge_carry"])
                if r_ is not None:
                    raw[sid] = round(r_, 4)
            rows = CP.core_rows(led, polarity=polarity, reagent_masses=masses, sid=sid, inorganic=True,
                                exclude_peaks=_alias_held(led), exclude_mz=shadows)
            if len(rows) and f["edge_carry"]:
                rows["ppm"] = rows["ppm"].astype(float) - MC.edge_carry_at(
                    f["edge_carry"], rows["mz"].to_numpy(dtype=float))
            if len(rows):
                parts.append(rows)
                n_rows += len(rows)
            del led
        runs.append({"dir": f["dir"], "commit": f["commit"], "n_files": n_read, "n_core_rows": n_rows})
        log(f"[calibration] {f['dir']}: {n_read} per-file ledgers, {n_rows} core rows")
    rows = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=CP.CORE_COLUMNS)
    klass = klasses[0] if len(klasses) == 1 else None
    pool = CP.pool_core(rows, CP.committed_mz_range(ledgers_m0), n_files_read=len(seen),
                        min_ions=min_ions, min_files=min_files, min_span_frac=min_span_frac,
                        klass=klass, source=SOURCE_CAMPAIGN)
    own = [c for f in facts for c in f["own_cals"]]
    run_w = ((float(np.median([c[0] for c in own])), float(np.median([c[1] for c in own])), len(own))
             if own else None)
    lo = [f["scan_range"][0] for f in facts if f["scan_range"]]
    hi = [f["scan_range"][1] for f in facts if f["scan_range"]]
    # the scan range: the override, else (one method) the median low edge and the top, else (runs of
    # different ranges pooled with --allow-mixed) the union of the runs' ranges
    sr = ([float(scan_range[0]), float(scan_range[1])] if scan_range is not None
          else None if not lo
          else [round(float(min(lo)), 3), round(float(max(hi)), 3)] if mixed_scan
          else [round(float(np.median(lo)), 3), round(float(max(hi)), 3)])
    days = [d for f in facts if f["days"] for d in f["days"]]
    core = pool.core
    width = None
    if pool.cal is not None:
        mu, sigma, basis = ((run_w[0], run_w[1], "run") if run_w is not None
                            else (float(pool.cal[0]), float(pool.cal[1]), "pool"))
        width = {"mu": round(mu, 4), "sigma": round(sigma, 4), "basis": basis, "model": "constant",
                 "pooled": {"mu": round(float(pool.cal[0]), 4), "sigma": round(float(pool.cal[1]), 4)},
                 "run": ({"mu": round(run_w[0], 4), "sigma": round(run_w[1], 4), "n_files": int(run_w[2])}
                         if run_w is not None else None),
                 "per_file_sigma": _spread([c[1] for c in own]),
                 # the lending files' unfloored core scatter (median + per file): the transfer check's
                 # reference (`transfer`); the floored per-file sigmas above sit on CAL_SIGMA_FLOOR
                 "per_file_raw_sigma": dict(_spread(list(raw.values())), files=dict(sorted(raw.items()))),
                 "abs_floor_mda": float(MC.ABS_FLOOR_MDA)}
    import peaky
    record = {
        "schema": SCHEMA, "source": SOURCE_CAMPAIGN,
        "reagent": reagents[0] if len(reagents) == 1 else reagents,
        "polarity": pols[0] if len(pols) == 1 else pols,
        "scan_range": sr, "scan_mixed": bool(mixed_scan),
        "instrument_class": klass if klass is not None else klasses,
        "days": [min(days), max(days)] if days else None,
        "runs": runs,
        "core": {"n_ions": int(pool.n_ions), "n_rows": int(len(rows)), "n_files": int(pool.n_files),
                 "n_files_read": int(pool.n_files_read), "n_candidate_ions": int(pool.n_candidate_ions),
                 "n_inorganic_ions": int(sum(1 for nf in core["neutral_formula"]
                                             if "C" not in _elements(nf))) if len(core) else 0,
                 "mz_span": [_r(pool.mz_span[0]), _r(pool.mz_span[1])],
                 "mz_range": [_r(pool.mz_range[0]), _r(pool.mz_range[1])],
                 "min_ions": int(min_ions), "min_files": int(min_files), "min_span_frac": float(min_span_frac),
                 "ions": [{"ion": str(r.ion), "neutral_formula": str(r.neutral_formula), "adduct": str(r.adduct),
                           "mz": round(float(r.mz), 5), "ppm": round(float(r.ppm), 4), "sid": str(r.sid)}
                          for r in core.sort_values("mz", kind="mergesort").itertuples(index=False)]},
        "width": width,
        "usable": pool.cal is not None,
        "reason": pool.reason,
        "fitted_at": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "peaky_version": peaky.__version__,
    }
    return record


def _r(x, nd=4):
    return None if x is None or not np.isfinite(x) else round(float(x), nd)


def _elements(nf) -> dict:
    from peaky.chem import chemistry as C
    return {k: v for k, v in C.parse_formula(T._txt(nf)).items() if v}


# ---------------------------------------------------------------------------
# the file
# ---------------------------------------------------------------------------
def dumps(record: dict) -> str:
    """The json text of a record (keys in RECORD_KEYS order, then any others; stable)."""
    keys = [k for k in RECORD_KEYS if k in record] + sorted(k for k in record if k not in RECORD_KEYS
                                                             and not k.startswith("_"))
    return json.dumps({k: record[k] for k in keys}, indent=1, default=str) + "\n"


def write(record: dict, path: str) -> str:
    """Write the record; returns the file's sha256."""
    text = dumps(record).encode("utf-8")
    path = os.path.expanduser(path)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(text)
    return hashlib.sha256(text).hexdigest()


def sha256_file(path: str) -> str:
    with open(os.path.expanduser(path), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def load(path: str) -> dict:
    """A calibration json, checked (schema, keys), with `_sha256` (of the file's
    bytes) and `_path` added. CalibrationError on anything else."""
    path = os.path.expanduser(str(path))
    try:
        raw = open(path, "rb").read()
        rec = json.loads(raw.decode("utf-8"))
    except (OSError, ValueError) as e:
        raise CalibrationError(f"{path}: not a readable calibration json ({e})") from e
    if not isinstance(rec, dict) or rec.get("schema") != SCHEMA:
        raise CalibrationError(f"{path}: schema {rec.get('schema') if isinstance(rec, dict) else None!r}, "
                               f"this peaky reads schema {SCHEMA}")
    missing = [k for k in ("source", "reagent", "polarity", "scan_range", "instrument_class", "width", "usable")
               if k not in rec]
    if missing:
        raise CalibrationError(f"{path}: missing keys {missing}")
    if rec["usable"] and not (isinstance(rec["width"], dict) and rec["width"].get("sigma") is not None):
        raise CalibrationError(f"{path}: usable but no width")
    rec["_sha256"] = hashlib.sha256(raw).hexdigest()
    rec["_path"] = os.path.abspath(path)
    return rec


def descriptor(record: dict | None) -> dict | None:
    """The compact record a manifest / batch_summary keeps of a json (no core ion list)."""
    if not record:
        return None
    w = record.get("width") or {}
    return {"sha256": record.get("_sha256"), "path": record.get("_path"), "source": record.get("source"),
            "reagent": record.get("reagent"), "polarity": record.get("polarity"),
            "scan_range": record.get("scan_range"), "scan_mixed": bool(record.get("scan_mixed")),
            "instrument_class": record.get("instrument_class"),
            "days": record.get("days"), "usable": record.get("usable"),
            "runs": [dict(r) for r in record.get("runs") or []],
            "n_ions": (record.get("core") or {}).get("n_ions"),
            "width": ({k: w.get(k) for k in ("mu", "sigma", "basis", "per_file_sigma")}
                      | {"per_file_raw_sigma": {k: v for k, v in (w.get("per_file_raw_sigma") or {}).items()
                                                if k != "files"} or None} if w else None)}


def show_text(record: dict) -> str:
    """The `peaky calibration show` text."""
    c, w = record.get("core") or {}, record.get("width") or {}
    out = [f"calibration json (schema {record.get('schema')}, source {record.get('source')})"
           + (f", sha256 {record['_sha256'][:12]}" if record.get("_sha256") else ""),
           f"  reagent {record.get('reagent')}  polarity {record.get('polarity')}  class "
           f"{record.get('instrument_class')}  scan range {record.get('scan_range')}  days {record.get('days')}",
           f"  runs: {len(record.get('runs') or [])}"]
    for r in record.get("runs") or []:
        out.append(f"    {r.get('dir')}  ({r.get('n_files')} files, commit {str(r.get('commit'))[:9]})")
    out.append(f"  core: {c.get('n_ions')} distinct ions ({c.get('n_inorganic_ions', 0)} inorganic) from "
               f"{c.get('n_rows')} rows in {c.get('n_files')} of {c.get('n_files_read')} files; m/z span "
               f"{c.get('mz_span')} of the committed {c.get('mz_range')}")
    if record.get("usable"):
        pf = w.get("per_file_sigma") or {}
        rw = w.get("per_file_raw_sigma") or {}
        out.append(f"  width: {w.get('mu'):+.3f} ± {w.get('sigma'):.3f} ppm (basis {w.get('basis')}; pooled "
                   f"{(w.get('pooled') or {}).get('mu'):+.3f} ± {(w.get('pooled') or {}).get('sigma'):.3f}; "
                   f"per-file sigma n {pf.get('n')} {pf.get('min')}..{pf.get('max')}; unfloored per-file "
                   f"scatter n {rw.get('n')} median {rw.get('median')} ({rw.get('min')}..{rw.get('max')}))")
    else:
        out.append(f"  NOT USABLE: {record.get('reason')}")
    for i in c.get("ions") or []:
        out.append(f"    {i['mz']:>11.5f}  {i['ppm']:+7.3f} ppm  {i['ion']:<18s} {i['neutral_formula']} {i['adduct']}")
    return "\n".join(out)


def compare(a: dict, b: dict) -> dict:
    """Two jsons side by side: the width ratio (b over a, lent and pooled), the
    centre difference, and the core overlap (shared ions; Jaccard)."""
    wa, wb = a.get("width") or {}, b.get("width") or {}
    ia = {i["ion"] for i in (a.get("core") or {}).get("ions") or []}
    ib = {i["ion"] for i in (b.get("core") or {}).get("ions") or []}

    def ratio(x, y):
        return None if x in (None, 0) or y is None else round(float(y) / float(x), 4)
    pa, pb = (wa.get("pooled") or {}), (wb.get("pooled") or {})
    return {"usable": [bool(a.get("usable")), bool(b.get("usable"))],
            "sigma": [wa.get("sigma"), wb.get("sigma")], "width_ratio": ratio(wa.get("sigma"), wb.get("sigma")),
            "pooled_sigma": [pa.get("sigma"), pb.get("sigma")],
            "pooled_ratio": ratio(pa.get("sigma"), pb.get("sigma")),
            "mu_diff": (None if wa.get("mu") is None or wb.get("mu") is None
                        else round(float(wb["mu"]) - float(wa["mu"]), 4)),
            "n_ions": [len(ia), len(ib)], "shared_ions": len(ia & ib),
            "jaccard": round(len(ia & ib) / len(ia | ib), 4) if (ia | ib) else None,
            "same_reagent": a.get("reagent") == b.get("reagent"), "same_polarity": a.get("polarity") == b.get("polarity"),
            "scan_ranges": [a.get("scan_range"), b.get("scan_range")]}


def compare_text(a: dict, b: dict) -> str:
    c = compare(a, b)
    return "\n".join([
        f"width (lent sigma): {c['sigma'][0]} -> {c['sigma'][1]} ppm, ratio {c['width_ratio']}",
        f"pooled sigma: {c['pooled_sigma'][0]} -> {c['pooled_sigma'][1]} ppm, ratio {c['pooled_ratio']}",
        f"centre difference: {c['mu_diff']} ppm",
        f"core: {c['n_ions'][0]} / {c['n_ions'][1]} ions, {c['shared_ions']} shared (Jaccard {c['jaccard']})",
        f"reagent same: {c['same_reagent']}, polarity same: {c['same_polarity']}, scan ranges {c['scan_ranges']}",
        f"usable: {c['usable']}"])


# ---------------------------------------------------------------------------
# the choice a batch makes
# ---------------------------------------------------------------------------
def scan_match(a, b) -> bool:
    """Whether two [lo, hi] scan ranges read off the data are one method's (module note)."""
    if not a or not b:
        return False
    lo_a, hi_a = float(a[0]), float(a[1])
    lo_b, hi_b = float(b[0]), float(b[1])
    if abs(lo_a - lo_b) > max(SCAN_LO_DA, SCAN_LO_FRAC * max(lo_a, lo_b)):
        return False
    return hi_a > 0 and SCAN_HI_X[0] <= hi_b / hi_a <= SCAN_HI_X[1]


def _days_overlap(a, b) -> bool:
    if not a or not b:
        return False
    d = _dt.timedelta(days=DAY_SLACK)
    a0, a1 = (_dt.date.fromisoformat(str(x)) for x in a)
    b0, b1 = (_dt.date.fromisoformat(str(x)) for x in b)
    return a0 - d <= b1 and b0 - d <= a1


def transfer(record: dict, recv: dict | None) -> tuple[str | None, dict]:
    """The mass-accuracy transfer check of a lent window (module note): (why not
    | None, {lender_raw_median, recv_sigma, n_rows, n_files, ratio, max_ratio}).
    The receiving scatter `recv` (`receiving_scatter`) must be measured on >=
    RECV_MIN_ROWS rows of >= RECV_MIN_IONS distinct ions in >= RECV_MIN_FILES
    files and be at most RECV_X times
    the lending files' median unfloored scatter (`width.per_file_raw_sigma`).
    A more precise receiver passes (it is lent a conservatively wide window)."""
    w = record.get("width") or {}
    lend = (w.get("per_file_raw_sigma") or {}).get("median")
    recv = recv or {}
    det = {"lender_raw_median": lend, "recv_sigma": recv.get("sigma"), "n_rows": int(recv.get("n_rows") or 0),
           "n_ions": int(recv.get("n_ions") or 0), "n_files": int(recv.get("n_files") or 0), "ratio": None,
           "max_ratio": RECV_X}
    if not lend:
        return ("the calibration json records no unfloored scatter of lending files (no file of its runs "
                "calibrated alone), so the transfer of its mass accuracy cannot be checked"), det
    if (det["recv_sigma"] is None or det["n_rows"] < RECV_MIN_ROWS or det["n_ions"] < RECV_MIN_IONS
            or det["n_files"] < RECV_MIN_FILES):
        return (f"the batch's own core scatter is not measurable ({det['n_rows']} core rows of {det['n_ions']} ions "
                f"in {det['n_files']} files; the transfer check needs >= {RECV_MIN_ROWS} rows of >= {RECV_MIN_IONS} "
                f"ions in >= {RECV_MIN_FILES} files)"), det
    det["ratio"] = round(float(det["recv_sigma"]) / float(lend), 3)
    if det["ratio"] > RECV_X:
        return (f"the batch's own core scatter {det['recv_sigma']:.3f} ppm ({det['n_rows']} rows, {det['n_files']} "
                f"files) is {det['ratio']:.2f}x the lending files' median {float(lend):.3f} ppm (at most "
                f"{RECV_X:g}x): the lender's mass accuracy does not transfer"), det
    return None, det


def choose(record: dict | None, *, reagent, polarity, instrument_class, scan_range, days,
           lent_ok: bool = False, recv: dict | None = None,
           detail: dict | None = None) -> tuple[str | None, str | None]:
    """(source, reason) a batch reads a calibration json as: (CAL_SOURCE_CAMPAIGN
    | CAL_SOURCE_LENT, None) or (None, why not) -- module note. `recv`: the
    batch's own core scatter (`receiving_scatter`), read only by a lent window's
    transfer check; `detail` (a dict) receives that check's record."""
    if not record:
        return None, None
    tag = f"the calibration json {str(record.get('_sha256') or '')[:12]}".rstrip()
    if not record.get("usable"):
        return None, f"{tag} is not usable: {record.get('reason')}"
    if record.get("polarity") != polarity:
        return None, f"{tag} is of polarity {record.get('polarity')!r}, the batch {polarity!r}"
    if instrument_class is None or record.get("instrument_class") != instrument_class:
        return None, (f"{tag} is of instrument class {record.get('instrument_class')!r}, the batch "
                      f"{instrument_class!r}")

    def _lent(what):
        why, det = transfer(record, recv)
        if detail is not None:
            detail.update(det)
        if why:
            return None, f"{tag} ({what}) is not lent: {why}"
        return T.CAL_SOURCE_LENT, None
    if record.get("reagent") == reagent:
        # a json pooled over several scan ranges (--allow-mixed) is no one method's: at most lent, transfer-checked
        if scan_match(record.get("scan_range"), scan_range) and not record.get("scan_mixed"):
            return T.CAL_SOURCE_CAMPAIGN, None
        if record.get("scan_mixed"):
            return _lent(f"pooled over mixed scan ranges, union {record.get('scan_range')}")
        return _lent(f"scan range {record.get('scan_range')}, the batch {scan_range}")
    if not lent_ok:
        return None, (f"{tag} is of reagent {record.get('reagent')!r}, the batch {reagent!r}: lending across "
                      "reagents needs --tier-calibration-lent")
    if not _days_overlap(record.get("days"), days):
        return None, (f"{tag} (reagent {record.get('reagent')!r}) covers days {record.get('days')}, the batch "
                      f"{days}: lending across reagents needs the same days")
    return _lent(f"reagent {record.get('reagent')!r}")


def as_pool(record: dict, source: str, *, abs_floor_mda: float = MC.ABS_FLOOR_MDA) -> CP.Pool:
    """The calpool.Pool a json lends as `source` (its width about the json's centre;
    calpool.lent_calibration centres it on each file's own Assigned rows)."""
    w = record["width"]
    c = record.get("core") or {}
    return CP.Pool(cal=T._Cal(float(w["mu"]), float(w["sigma"]), abs_floor_mda=abs_floor_mda, source=source),
                   n_ions=int(c.get("n_ions") or 0), n_files=int(c.get("n_files") or 0),
                   n_files_read=int(c.get("n_files_read") or 0), source=source)


def levels_record(per_stats: list, record: dict | None, *, used_as: str | None, reason: str | None,
                  lent_flag: bool) -> dict:
    """batch_summary `levels_calibration`: the batch's weakest file source in one
    word (`source`), the files per source, and the calibration json's descriptor
    with how the batch read it (`as`: campaign / lent / None, `reason`)."""
    words = (T.CAL_SOURCE_PER_FILE, T.CAL_SOURCE_POOLED, T.CAL_SOURCE_CAMPAIGN, T.CAL_SOURCE_LENT,
             T.CAL_SOURCE_NONE)
    files = {k: sum(1 for x in per_stats if x.get("tier_calibration") == k) for k in words}
    seen = {x.get("tier_calibration") for x in per_stats} - {None}
    src = next((w for w in reversed(words) if w in seen), None)
    file = None
    if record is not None:
        file = dict(descriptor(record), used=bool(used_as) and bool(files.get(used_as)), read_as=used_as,
                    reason=reason, lent_flag=bool(lent_flag))
    return {"source": src, "files": files, "file": file}


# ---------------------------------------------------------------------------
# the offline re-level
# ---------------------------------------------------------------------------
def lend_to_summary(summary: dict, per_file: dict, record: dict | None, *, lent_ok: bool,
                    facts: dict) -> tuple[dict, str | None, str | None, dict]:
    """A finished run's batch summary with the json's window lent to every file
    that calibrated neither alone nor on the batch's pool (stats `degeneracy_cal`
    null): `degeneracy_cal` the lent width about the file's own centre (the
    median ppm error of its Assigned M0 rows, on the batch axis model's carried
    scale, when it has CENTRE_MIN_ROWS of them), `tier_calibration` the source.
    `facts`: reagent, polarity, instrument_class, scan_range, days of the run.
    Returns (summary copy, source | None, reason | None, {sid: centre word})."""
    out = copy.deepcopy(summary)
    uncal = {st.get("sample_id"): per_file[st.get("sample_id")] for st in out.get("per_file") or []
             if "degeneracy_cal" in st and st["degeneracy_cal"] is None and st.get("sample_id") in per_file}
    recv = receiving_scatter(uncal, reagent=facts.get("reagent"),
                             polarity="positive" if facts.get("polarity") == "+" else "negative",
                             carry=(summary.get("mass_axis") or {}).get("edge_carry"),
                             exclude_mz=_shadow_lines(facts["path"]) if facts.get("path") else ()) if record else None
    det: dict = {}
    src, why = choose(record, lent_ok=lent_ok, recv=recv, detail=det,
                      **{k: facts.get(k) for k in ("reagent", "polarity", "instrument_class", "scan_range", "days")})
    facts["transfer"] = det or None
    centres = {}
    if src is None:
        return out, None, why, centres
    pool = as_pool(record, src)
    carry = (summary.get("mass_axis") or {}).get("edge_carry")
    for st in out.get("per_file") or []:
        sid = st.get("sample_id")
        if "degeneracy_cal" not in st or st["degeneracy_cal"] is not None or sid not in per_file:
            continue
        led = per_file[sid]
        m0 = led[(led["role"] == L.ROLE_M0) & (led["tier"].astype(str) == T.TIER_ASSIGNED)]
        ppm = pd.to_numeric(m0["ppm_error"], errors="coerce")
        if carry:
            ppm = ppm - MC.edge_carry_at(carry, pd.to_numeric(m0["mz"], errors="coerce").to_numpy(dtype=float))
        cal, centre = CP.lent_calibration(pool, ppm)
        st["degeneracy_cal"] = {"mu": float(cal[0]), "sigma": float(cal[1])}
        st["tier_calibration"] = src
        centres[sid] = centre
    return out, src, why, centres


#: what a re-level does to the tier stage (`relevel.tier_stage`): nothing -- the tiers,
#: and the per-file isotope-guard records, of a re-levelled dir are the source run's
TIER_STAGE_NOT_RERUN = "not re-run"


def relevel_run(run_dir: str, record: dict | None, *, out_dir: str, lent_ok: bool = False, log=print) -> dict:
    """Re-level a finished batch run dir with a calibration json (or none: the
    run's own windows) WITHOUT re-assigning (module note). Writes `out_dir`
    (never the run dir): every top-level file of the run, merged_ledger.csv
    re-stamped, tables/evidence_levels.csv, batch_summary.json with the lent
    per-file windows, `levels_calibration`, the evidence-level counts and
    `levels_not_assessed_reason` recomputed, and `relevel` (what was done).
    The per-file ledgers are not rewritten. Returns the `relevel` record."""
    import shutil

    from peaky.assignment import evidence as EV
    run_dir = os.path.expanduser(str(run_dir).rstrip("/"))
    out_dir = os.path.expanduser(str(out_dir).rstrip("/"))
    if os.path.abspath(out_dir) == os.path.abspath(run_dir):
        raise CalibrationError("the re-level writes a new dir, never the run dir itself")
    facts = run_facts(run_dir)
    src = EV.source_from_run_dir(run_dir)
    summ0 = src.inputs.summary
    summ, used, why, centres = lend_to_summary(summ0, src.per_file, record, lent_ok=lent_ok, facts=facts)
    src.inputs.summary = summ
    log(f"[relevel] {facts['dir']}: " + (f"{len(centres)} file(s) lent the window as {used} "
                                         f"(centres: {sorted(set(centres.values()))})" if used
                                         else f"no window lent ({why or 'no calibration json'})"))
    levels = EV.level_source(src)
    # the merged ledger re-stamped on an exact (round-trip) read of the run's own: the
    # level columns are the new ones, every other column keeps the run's bytes (the
    # level inputs above read the CSVs as the batch reads them)
    mp = os.path.join(run_dir, "merged_ledger.csv")
    merged = (EV.stamp_merged(pd.read_csv(mp, low_memory=False, float_precision="round_trip"), levels)
              if src.inputs.merged is not None and os.path.isfile(mp) else None)
    os.makedirs(os.path.join(out_dir, "tables"), exist_ok=True)
    for f in os.listdir(run_dir):
        p = os.path.join(run_dir, f)
        if os.path.isfile(p) and f not in ("merged_ledger.csv", "batch_summary.json"):
            shutil.copy2(p, os.path.join(out_dir, f))
    EV.for_output(levels).to_csv(os.path.join(out_dir, "tables", "evidence_levels.csv"), index=False)
    if merged is not None:
        merged.to_csv(os.path.join(out_dir, "merged_ledger.csv"), index=False)
    na = (EV.levels_not_assessed_reason(levels["evidence"], levels["evidence_level"])
          if len(levels) and {"evidence", "evidence_level"} <= set(levels.columns) else None)
    if na and why:
        na += f" A calibration file was given and not used: {why}."
    merged_counts = {}
    if merged is not None:
        from peaky.batch.assign_batch import _claims_summary
        claims = _claims_summary(merged, levels)
        merged_counts = {"merged_M0": int(len(merged)),
                         "merged_tiers": merged["tier"].value_counts().to_dict() if len(merged) else {},
                         "merged_claims": claims["merged"]}
    rec = {"run_dir": run_dir, "calibration": descriptor(record), "lent_flag": bool(lent_ok), "read_as": used,
           "reason": why, "transfer": facts.get("transfer"), "centres": centres, "n_pairs": int(len(levels)),
           "merged_counts": merged_counts,
           # a finished run keeps no tier-time ledger: no z-tail, no degeneracy cap and no
           # isotope-contradiction guard is re-run, so the tiers are the source run's
           "tier_stage": TIER_STAGE_NOT_RERUN,
           "peaky_version": __import__("peaky").__version__}
    summ = dict(summ)
    summ["levels_calibration"] = levels_record(summ.get("per_file") or [], record, used_as=used, reason=why,
                                               lent_flag=lent_ok)
    if summ["levels_calibration"]["file"] is not None:
        summ["levels_calibration"]["file"]["transfer"] = facts.get("transfer")
    summ["levels_not_assessed_reason"] = na
    ev = dict(summ.get("evidence_levels") or {})
    ev["pooled"] = EV.summarize(levels["evidence_level"]) if len(levels) else {}
    if merged is not None:
        ev["merged"] = EV.summarize(merged["evidence_level"]) if len(merged) else {}
    ev["n_pairs"] = int(len(levels))
    summ["evidence_levels"] = ev
    if merged is not None:
        summ["claims"] = claims
    summ["relevel"] = rec
    with open(os.path.join(out_dir, "batch_summary.json"), "w") as fh:
        json.dump(summ, fh, indent=2, default=str)
    _relevel_manifest(run_dir, out_dir, record, rec)
    return rec


def _relevel_manifest(run_dir: str, out_dir: str, record: dict | None, rec: dict) -> None:
    """The re-levelled dir's run_manifest.json: the run's own (code, input, config as the
    run had them), with `input.tier_calibration` (the json's path, sha256, lent flag) when
    a json was given, `output.merged_ledger_sha1` refreshed to the re-stamped merged
    ledger (its level columns are new; every other byte is the run's), and `relevel`:
    the source run dir, the sha1 of its manifest and merged ledger, whether the ledger's
    bytes changed, the json's sha256, how it was read, `tier_stage` ("not re-run") and the
    code that re-levelled."""
    from peaky.reporting import provenance as PV
    src_m = os.path.join(run_dir, "run_manifest.json")
    try:
        man = json.load(open(src_m)) if os.path.isfile(src_m) else {}
    except (OSError, ValueError):
        man = {}
    src_led = PV.sha1_file(os.path.join(run_dir, "merged_ledger.csv"))
    new_led = PV.sha1_file(os.path.join(out_dir, "merged_ledger.csv"))
    man.setdefault("output", {})["merged_ledger_sha1"] = new_led
    # the counts read off the merged ledger, re-counted on the re-stamped one (the claims move with the levels;
    # the tiers do not, and are re-counted all the same)
    counts = man["output"].setdefault("counts", {})
    for k, v in (rec.get("merged_counts") or {}).items():
        counts[k] = v
    if record is not None:
        man.setdefault("input", {})["tier_calibration"] = {"path": record.get("_path"), "sha256": record.get("_sha256"),
                                                           "lent_flag": bool(rec.get("lent_flag"))}
    man["relevel"] = {"source_run_dir": run_dir, "source_manifest_sha1": PV.sha1_file(src_m) if os.path.isfile(src_m)
                      else None, "source_merged_ledger_sha1": src_led, "merged_ledger_sha1": new_led,
                      "merged_ledger_changed": src_led != new_led,
                      "cal_sha256": record.get("_sha256") if record is not None else None,
                      "read_as": rec.get("read_as"), "reason": rec.get("reason"), "lent_flag": rec.get("lent_flag"),
                      "tier_stage": rec.get("tier_stage"), "peaky_version": rec.get("peaky_version"),
                      "git": PV.git_info(os.path.dirname(os.path.dirname(__file__)))}
    with open(os.path.join(out_dir, "run_manifest.json"), "w") as fh:
        json.dump(man, fh, indent=2, default=str)
