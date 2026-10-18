"""Per-scan MS2 from a Mascope file's `/metadata`: every scan's trailer and
centroids -> scan events -> precursors -> energy-resolved spectra in the
`sources` schemas.

The server's MS2 endpoints (`sources`) serve averaged spectra only, and only
for the scans whose scan filter its raw reader can parse. Its per-file
`/metadata` (`GET sample/files/{id}/metadata`) holds more: every scan's
centroid list (m/z, intensity, resolution, noise) and, for each scan the
server reads as MS order 'Ms', the instrument's TRAILER. This module rebuilds
the MS2 from that payload. It is PURE -- no client, and no file I/O but the
inclusion-list reader -- and runs in four steps:

  * `scan_table`: one row per scan, read off the TRAILER, never off the
    filter string: MS order, scan event, node, NCE, volts, isolation width,
    scan range, injection time; the filter as written and whether it agrees
    with the trailer (`filter_ok`; `filter_verdict` sums the file up);
  * `map_precursors`: scan events -> experiments (one NCE each) -> list rows
    -> precursor m/z, the headerless scans placed by walking the scan cycle
    (`infer_headerless`); or `assign_from_times` for a file the server groups
    itself (its time-series times name every spectrum's scans);
  * `average_scans` / `build_spectra`: each (precursor, NCE) group's scans
    averaged into one centroid spectrum in `sources.SPECTRA_COLUMNS` /
    `PARENTS_COLUMNS`, keyed as the server keys it, plus every kept peak's
    intensity in every scan of its group (`SCAN_PEAK_COLUMNS`);
  * `ms1_average` / `window_inventory`: the MS1 scans averaged for the
    isolation-window inventory (`sources.window_table`).

What the payload holds -- read on our own files (Orbitrap Exploris 120,
Mascope >= 1.8) and on targeted-MS2 (inclusion-list) files whose scan filters
the server's reader garbles:

  * `stats_per_scan` has a row ONLY for a scan the server reads as MS order
    'Ms'. On an ordinary MS2 file that is the MS1 scans only (own 2026-09-11
    monoterpene standard A: 140 scans, 61 rows, every one 'FTMS + p NSI Full
    ms [40.0000-600.0000]'; its 78 MS2 scans have centroids and no row), and
    the first scan may be missing too. On a garbled file the MS2 scans get
    rows by accident (their garbled order reads 'Ms') and a fraction of the
    scans have none; their centroids are there.
  * A garbled filter is noise: random analyzers, ionizations and polarities
    ('ITMS - c EI', 'TQMS - c EI', 'SQMS + c FABI', 'Sector - c'), and every
    MS2 scan read as 'Full ms [40.0000-<HighMass>]' with no 'ms2 <m/z>@hcd<E>'
    -- a few rows read 'FTMS + p NSI' by chance. The trailer is right on
    every row: `HCD Energy:` ('' on MS1), `Scan Node:` ('Full Scan' / 'tMS2'),
    `Scan Event:`, `MS2 Isolation Width:`, LowMass / HighMass. Polarity and
    MS order are therefore never read off a filter that disagrees with it.
    `Monoisotopic M/Z:` is 0 on every row of these files: no precursor source.
  * Event numbering of an inclusion-list method: the MS2 experiments' events
    run from 1 in experiment order, one per list row -- one shared list of n
    rows over K experiments (one NCE each) gives events 1..n (experiment 1),
    n+1..2n (experiment 2), ...; the Full Scan is the last event (K*n + 1).
    Every event's trailer NCE is then its experiment's.
  * The scan cycle: inside one experiment's time block the instrument walks
    its events in order, cyclically (the last is followed by the first); at a
    block switch the next block starts at its FIRST event (`infer_headerless`).
  * The scan range follows the precursor ("define first mass" mode): HighMass
    = 1.02 x (precursor + isolation width / 2) + 20, to a few hundredths of a
    mDa (the width term is the method's documented one, not a fit). The scan range is set per
    experiment, so the rule is judged per experiment (`rule_verdicts`): an
    experiment whose events all read one HighMass has a fixed range, and
    nothing is derived for it.
  * LowMass may move per experiment (an automatic first mass in one
    block while the other blocks keep the method's): the range is kept per scan,
    and a parent whose spectra differ in it is cut to the range they share
    (`build_spectra`).
  * Every scan's centroids carry a resolution of their own -- a measured
    value, not the instrument's setting: on our own files mostly a multiple
    of 100 on weak peaks (S/N < 10), about one in ten on stronger ones.
  * The MS2 m/z axis is the instrument's own calibration, which the server
    does not correct here: a file can sit tens of ppm off its method's
    precursors. The offset is measured per file against the keys typed
    finely enough to measure it (`sources.axis_offset`; a key typed with two
    decimals measures its own typing), and the precursor search is centred
    on it; where none is measured the search keeps its side-lobe guard.

NCE and volts are both kept and never converted into each other (see
`sources`).
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from itertools import pairwise

import numpy as np
import pandas as pd

from peaky.ms2 import sources as SRC

__version__ = "0.1.0"

# ---------------------------------------------------------------------------
# constants -- each says what it was measured on
# ---------------------------------------------------------------------------
#: Averaging, step 1: the pooled centroids of a group's scans, sorted by m/z,
#: open a new seed bin wherever the gap to the previous one exceeds this (ppm).
#: Step 2 (`MERGE_FWHM`) joins seed bins that are one peak split by the
#: scan-to-scan m/z jitter. Together they reproduce the server's S/N-3
#: by_activation spectra of our own 2026-09-11 monoterpene standards A and B
#: and the 2026-09-10 exploratory file (47 spectra, 893 peaks): 892 of 893
#: matched (A 230/230, B 265/266, exploratory 397/397), m/z identical (891
#: within 0.01 ppm), 2 extra peaks, both at S/N 3.00-3.01. A flat gap rule
#: does worse: 3 ppm joins side lobes to their line (B: 252/266 matched, 13
#: intensities off by > 15 %), 1 ppm alone splits jittered peaks (B 258/266,
#: exploratory 388/397, 16 extra).
SEED_PPM = 1.0
#: Averaging, step 2: adjacent seed bins merge when their gap is below this
#: fraction of the narrower one's FWHM (m/z / resolution, the bin's
#: intensity-weighted resolution). 0.4 / 0.6 / 0.75 matched 893 - 3 / 4 / 11
#: of the same 893 peaks.
MERGE_FWHM = 0.5
#: The per-scan mean of a cluster runs this much above the server's averaged
#: intensity on the median peak of those 47 spectra (5-95 %: 1.000-1.07):
#: the server re-reads a height off its averaged PROFILE, which `/metadata`
#: does not carry. Reported (manifest), never corrected for.
SERVER_HEIGHT_RATIO = 1.020
#: A scan's time in `centroids_meta.time` and the server's time-series time of
#: the same scan agree to < 1 ms on every MS2 scan of our own files (standard
#: A 78/78, B 67/67, the exploratory file 213/213).
TIME_MATCH_S = 1e-3
#: Two roles, one tolerance (Da; float32 limits): an inferred scan is checked
#: against its event's scan range -- no centroid below LowMass or above
#: HighMass by more than this -- and a parent whose spectra recorded scan
#: ranges that differ by more than this is cut to the range they share
#: (`build_spectra`).
RANGE_CHECK_DA = 0.01
#: The scan-range rule: HighMass = RULE_SLOPE x (precursor + width / 2) +
#: RULE_OFFSET_DA. A list is checked against it within `RULE_TOL_MDA` of
#: HighMass -- some 20 x the scatter of HighMass about the rule; a list in the
#: wrong order or the wrong list misses by the spacing of its rows (>= 1 mDa
#: x 1.02, and mostly Da).
RULE_SLOPE = 1.02
RULE_OFFSET_DA = 20.0
RULE_TOL_MDA = 0.5
#: HighMass values are compared rounded to this (Da), per experiment
#: (`rule_verdicts`): two or more seen events of one experiment reading one
#: value are a fixed range, values that differ follow the precursor, and one
#: seen event is unknown; an event whose value another of its experiment
#: shares derives no precursor (`_shared_high_mass`). File-wide (`rule_applies`),
#: fewer than two distinct values say nothing is derived anywhere. Our own
#: standard's precursors 155.1429 / 155.1559 read HighMass 13.3 mDa apart.
RULE_DISTINCT_DA = 1e-3
#: A filter's scan range agrees with the trailer's LowMass / HighMass within
#: this (Da; the filter writes 4 decimals, the trailer float32), and its
#: energy with the trailer's NCE within `FILTER_NCE_TOL`.
FILTER_RANGE_TOL_DA = 1e-3
FILTER_NCE_TOL = 0.005
#: Two roles, one tolerance (mDa; a method types its isolation centres with 4
#: decimals): a list row and the readable filters of its event must agree
#: within this (`map_precursors`); and an event's readable filters name ONE
#: precursor when they agree within it -- a scan of the event whose filter is
#: garbled then takes their median -- else one per scan (a data-dependent
#: event: such a scan has none, `_event_table`). The scan-range rule is
#: checked with `RULE_TOL_MDA` on HighMass, not with this.
SOURCE_MATCH_MDA = 1.0
#: The derived (rule) precursor is rounded to the decimals a method types.
RULE_DECIMALS = 4

PRECURSOR_FILTER = "filter"
PRECURSOR_LIST = "inclusion list"
PRECURSOR_RULE = "scan-range rule"
PRECURSOR_SERVER = "server"

FILTERS_YES, FILTERS_NO, FILTERS_PARTLY, FILTERS_NONE = "yes", "no", "partly", "no header rows"

#: One row per scan of the file (`scan_table`, then `map_precursors` /
#: `assign_from_times`). `order` is 1 / 2 from the trailer, NaN when the trailer
#: does not say (or there is none); `event`, `experiment` and `slot` (the list
#: row, 1-based) are NaN when unknown.
SCAN_COLUMNS: dict[str, str] = {
    "scan": "int64", "t_s": "float64", "has_header": "bool", "order": "float64",
    "event": "float64", "node": "str", "nce": "float64", "hcd_v": "float64",
    "iso_width": "float64", "low_mass": "float64", "high_mass": "float64",
    "inj_time_ms": "float64", "max_inj_time_ms": "float64", "microscans": "float64",
    "agc_fill": "float64", "tic": "float64", "ft_resolution": "float64",
    "filter": "str", "filter_ok": "bool", "filter_parent_mz": "float64",
    "n_centroids": "int64", "min_centroid_mz": "float64", "max_centroid_mz": "float64",
    "inferred": "bool", "parent_mz": "float64", "precursor_source": "str",
    "experiment": "float64", "slot": "float64", "spectrum": "str", "note": "str",
}
#: Every kept peak of every spectrum, in every scan of its group: `intensity`
#: is the scan's summed centroids in the peak's cluster, 0 where the scan has
#: none (so the mean over a peak's rows is the spectrum's intensity);
#: `n_centroids` how many of the scan's centroids it sums.
SCAN_PEAK_COLUMNS: dict[str, str] = {
    "spectrum": "str", "peak_mz": "float64", "scan": "int64", "t_s": "float64",
    "intensity": "float64", "n_centroids": "int64",
}


class ScanMapError(ValueError):
    """The scans cannot be mapped to precursors as asked: the message names the
    events and says why (a list that does not fit the method, an experiment
    with two NCEs, no precursor source). `needs_list` is True when the
    method's inclusion list is what is missing (events with no precursor
    source); any other refusal no list can lift."""

    def __init__(self, message: str, *, needs_list: bool = False):
        super().__init__(message)
        self.needs_list = bool(needs_list)


def _frame(cols: dict[str, list], spec: dict[str, str]) -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(cols.get(c, []), dtype=t) for c, t in spec.items()})


def _num(v) -> float:
    if isinstance(v, str):
        v = v.strip()
        if not v:
            return math.nan
    try:
        return float(v)
    except (TypeError, ValueError):
        return math.nan


def _events_text(events, limit: int = 12) -> str:
    ev = sorted(int(e) for e in events)
    head = ", ".join(str(e) for e in ev[:limit])
    return head + (f" (+{len(ev) - limit} more)" if len(ev) > limit else "")


# ---------------------------------------------------------------------------
# the payload and the scan table
# ---------------------------------------------------------------------------
def metadata_payload(meta: dict) -> dict:
    """The `/metadata` payload itself: unwraps the API's {'data': ...} envelope
    and checks it holds per-scan centroids."""
    if not isinstance(meta, dict):
        raise ValueError("the file metadata must be a JSON object")
    if "centroids_meta" not in meta and isinstance(meta.get("data"), dict):
        meta = meta["data"]
    cm = meta.get("centroids_meta")
    if not isinstance(cm, dict) or "time" not in cm or "data" not in cm:
        raise ValueError("the file metadata holds no per-scan centroids "
                         "('centroids_meta': {'time', 'data'})")
    if len(cm["time"]) != len(cm["data"]):
        raise ValueError(f"the file metadata holds {len(cm['time'])} scan times for "
                         f"{len(cm['data'])} centroid lists")
    return meta


_NUMBER = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")
_MS2 = re.compile(r"\bms2\s+(\d+(?:\.\d*)?)@([A-Za-z]+)(\d+(?:\.\d*)?)")
_MS1 = re.compile(r"\b(?:Full|SIM)\s+ms\b")
_RANGE = re.compile(r"\[([^\]]*)\]\s*$")


def _energy(v) -> tuple[bool, float, str]:
    """('HCD Energy:' present as a number, the NCE, a stepped list as text).
    '' / None -> (False, nan, ''); '9.00' -> (True, 9.0, ''); a stepped
    '20.00,40.00' -> (True, nan, '20.00,40.00')."""
    if v is None:
        return False, math.nan, ""
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return (math.isfinite(float(v)), float(v), "")
    nums = _NUMBER.findall(str(v))
    if not nums:
        return False, math.nan, ""
    if len(nums) == 1:
        return True, float(nums[0]), ""
    return True, math.nan, str(v).strip()


def parse_filter(text) -> dict:
    """A Thermo scan filter read token by token: analyzer (the first token),
    polarity (the first '+' / '-' token), MS order (2 for 'ms2 <m/z>@<act><E>',
    1 for 'Full ms' / 'SIM ms'), the precursor, activation and energy of an
    ms2, and the bracketed scan range. Unparsed fields are None / NaN."""
    s = str(text or "")
    toks = s.split()
    out = {"analyzer": toks[0] if toks else None, "polarity": None, "order": None,
           "parent_mz": math.nan, "activation": None, "energy": math.nan,
           "low": math.nan, "high": math.nan}
    for t in toks:
        if t in ("+", "-"):
            out["polarity"] = t
            break
    m = _MS2.search(s)
    if m:
        out.update(order=2, parent_mz=float(m.group(1)), activation=m.group(2).lower(),
                   energy=float(m.group(3)))
    elif _MS1.search(s):
        out["order"] = 1
    r = _RANGE.search(s)
    if r:
        parts = [p for p in re.split(r"[,\s]+", r.group(1)) if p]
        lo_hi = [x for p in parts for x in p.split("-") if x]
        vals = [_num(x) for x in lo_hi]
        if len(vals) >= 2:
            out["low"], out["high"] = vals[0], vals[-1]
    return out


def _filter_ok(f: dict, order: float, nce: float, low: float, high: float,
               ft_res: float) -> bool:
    """The filter agrees with the trailer: an Orbitrap analyzer on an Orbitrap
    scan (`FT Resolution:` > 0), one polarity, the trailer's MS order (an MS2
    filter with an 'hcd' energy equal to the trailer's NCE), its scan range."""
    if not math.isfinite(order) or f["polarity"] is None or f["order"] is None:
        return False
    if math.isfinite(ft_res) and ft_res > 0 and f["analyzer"] != "FTMS":
        return False
    if int(order) != f["order"]:
        return False
    if f["order"] == 2 and (f["activation"] != "hcd" or not math.isfinite(nce)
                           or abs(f["energy"] - nce) > FILTER_NCE_TOL):
        return False
    for a, b in ((f["low"], low), (f["high"], high)):
        if math.isfinite(b) and not (math.isfinite(a) and abs(a - b) <= FILTER_RANGE_TOL_DA):
            return False
    return True


def scan_table(meta: dict) -> pd.DataFrame:
    """One row per scan of the payload (`SCAN_COLUMNS`), from the TRAILER.

    `order`: 2 when `HCD Energy:` is a number (a stepped list too: NCE NaN,
    said in `note`), 1 when it is '' and the node is 'Full Scan' or '', NaN
    otherwise (e.g. a 'tSIM' node: no Full Scan and no MS2 scan; and on a
    scan with no row: `has_header` False). `t_s` is the
    scan's start in `centroids_meta.time` (s). The filter is kept as written;
    `filter_ok` says whether it agrees with the trailer (`_filter_ok`; a row
    with no HCD energy whose order is unread is checked as an MS1 scan), and
    `filter_parent_mz` is its precursor when it does. The mapping columns
    (`inferred` .. `spectrum`) are blank until `map_precursors` or
    `assign_from_times` fill them."""
    meta = metadata_payload(meta)
    cm = meta["centroids_meta"]
    rows = meta.get("stats_per_scan") or {}
    n = len(cm["time"])
    cols: dict[str, list] = {c: [] for c in SCAN_COLUMNS}
    for i in range(n):
        scan = i + 1
        r = rows.get(str(scan)) or rows.get(scan)
        c = cm["data"][i] or {}
        mzs = np.asarray(c.get("mzs") or [], dtype=np.float64)
        rec = {c_: (math.nan if t == "float64" else "" if t == "str" else False)
               for c_, t in SCAN_COLUMNS.items()}
        rec.update(scan=scan, t_s=_num(cm["time"][i]), n_centroids=len(mzs),
                   min_centroid_mz=float(mzs.min()) if len(mzs) else math.nan,
                   max_centroid_mz=float(mzs.max()) if len(mzs) else math.nan)
        notes = []
        if isinstance(r, dict):
            node = str(r.get("Scan Node:") or "").strip()
            has_e, nce, steps = _energy(r.get("HCD Energy:"))
            if has_e:
                order = 2.0
                if steps:
                    notes.append(f"stepped collision energy ({steps}): no single NCE")
            elif "HCD Energy:" in r and node in ("Full Scan", ""):
                order = 1.0
            else:
                order = math.nan
                notes.append("the trailer gives no MS order (no HCD energy on a "
                             f"'{node}' node)" if "HCD Energy:" in r
                             else "the trailer has no 'HCD Energy:' key")
            ev = _num(r.get("Scan Event:"))
            low, high = _num(r.get("LowMass")), _num(r.get("HighMass"))
            ft_res = _num(r.get("FT Resolution:"))
            filt = str(r.get("ScanType") or "")
            f = parse_filter(filt)
            # a trailer with no HCD energy names no fragmentation: on a node
            # other than the Full Scan (e.g. a 'tSIM' experiment) the MS order
            # stays unread -- such a scan is no Full Scan and in no MS2 block --
            # but its filter is checked as an MS1 one ('SIM ms')
            f_order = order if math.isfinite(order) else (
                1.0 if "HCD Energy:" in r and not has_e else math.nan)
            ok = _filter_ok(f, f_order, nce, low, high, ft_res)
            rec.update(has_header=True, order=order, event=ev, node=node, nce=nce,
                       hcd_v=_energy(r.get("HCD Energy V:"))[1],
                       iso_width=_num(r.get("MS2 Isolation Width:")) if order == 2 else math.nan,
                       low_mass=low, high_mass=high,
                       inj_time_ms=_num(r.get("Ion Injection Time (ms):")),
                       max_inj_time_ms=_num(r.get("Max. Ion Time (ms):")),
                       microscans=_num(r.get("Micro Scan Count:")),
                       agc_fill=_num(r.get("AGC Fill:")), tic=_num(r.get("TIC")),
                       ft_resolution=ft_res, filter=filt, filter_ok=ok,
                       filter_parent_mz=f["parent_mz"] if ok and order == 2 else math.nan)
        rec["note"] = SRC.NOTE_SEP.join(notes)
        for c_ in SCAN_COLUMNS:
            cols[c_].append(rec[c_])
    return _frame(cols, SCAN_COLUMNS)


def filter_verdict(scans: pd.DataFrame) -> dict:
    """Are the file's scan filters readable? 'yes' (every header row's filter
    agrees with its trailer), 'no' (none does, or no MS2 row's does: no
    filter can name a precursor), 'partly', or 'no header rows'; with the
    counts and the polarity signs the agreeing filters carry."""
    h = scans[scans["has_header"]]
    ms2 = h[h["order"] == 2]
    ok, ms2_ok = int(h["filter_ok"].sum()), int(ms2["filter_ok"].sum())
    if not len(h):
        verdict = FILTERS_NONE
    elif ok == len(h):
        verdict = FILTERS_YES
    elif ok == 0 or (len(ms2) and ms2_ok == 0):
        verdict = FILTERS_NO
    else:
        verdict = FILTERS_PARTLY
    signs = sorted({parse_filter(t)["polarity"] for t in h.loc[h["filter_ok"], "filter"]} - {None})
    return {"filters_readable": verdict, "header_rows": len(h), "filters_ok": ok,
            "ms2_rows": len(ms2), "ms2_filters_ok": ms2_ok,
            "polarities_in_filters": "".join(signs)}


def filter_polarity(scans: pd.DataFrame) -> str | None:
    """The polarity the filters give -- only when every filter agrees with its
    trailer and they carry one sign; None otherwise (a garbled file's signs
    are random: ask the user)."""
    v = filter_verdict(scans)
    if v["filters_readable"] == FILTERS_YES and len(v["polarities_in_filters"]) == 1:
        return v["polarities_in_filters"]
    return None


# ---------------------------------------------------------------------------
# inclusion lists
# ---------------------------------------------------------------------------
#: Column names a list's precursor m/z is read from, in this order (matched
#: case- and space-insensitively): the Thermo mass-list export, then the
#: usual spellings.
LIST_MZ_COLUMNS = ("precursor (m/z)", "mass [m/z]", "m/z", "mz", "precursor m/z", "mass")
LIST_NAME_COLUMNS = ("compound", "name", "formula")
LIST_COLUMNS: dict[str, str] = {"row": "int64", "mz": "float64", "compound": "str"}


def _norm(c) -> str:
    return re.sub(r"\s+", " ", str(c).strip().lower())


def inclusion_list_frame(df: pd.DataFrame, source: str = "list") -> pd.DataFrame:
    """A list table (`LIST_COLUMNS`: row 1-based, mz, compound) from any frame
    holding a precursor m/z column (`LIST_MZ_COLUMNS`). Rows without an m/z
    are an error (the row numbering IS the event mapping)."""
    cols = {_norm(c): c for c in df.columns}
    mzc = next((cols[c] for c in LIST_MZ_COLUMNS if c in cols), None)
    if mzc is None:
        raise ValueError(f"{source}: no precursor m/z column (looked for "
                         + ", ".join(repr(c) for c in LIST_MZ_COLUMNS) + ")")
    namec = next((cols[c] for c in LIST_NAME_COLUMNS if c in cols), None)
    mz = pd.to_numeric(df[mzc], errors="coerce")
    bad = [i + 1 for i, v in enumerate(mz) if not (math.isfinite(v) and v > 0)]
    if bad:
        raise ValueError(f"{source}: row(s) {_events_text(bad)} hold no positive m/z in "
                         f"{mzc!r}")
    names = (df[namec].fillna("").astype(str) if namec is not None
             else pd.Series([""] * len(df)))
    return _frame({"row": list(range(1, len(df) + 1)), "mz": mz.astype(float).tolist(),
                   "compound": names.tolist()}, LIST_COLUMNS)


def read_inclusion_list(path: str) -> pd.DataFrame:
    """An inclusion list from a CSV (the Thermo mass-list export: `Compound,
    Formula, Adduct, Precursor (m/z), Precursor Charge (z)[, ...]`; also
    `Mass [m/z]`, `m/z`, `mz`) or an .xlsx (its first sheet). The row order
    is the method's: row k of a shared list is event k of every experiment."""
    import os
    p = os.path.expanduser(str(path))
    if p.lower().endswith((".xlsx", ".xlsm")):
        df = pd.read_excel(p)
    else:
        df = pd.read_csv(p, encoding="utf-8-sig")
        if df.shape[1] == 1:                  # a ';'- or tab-separated export
            head = str(df.columns[0])
            sep = ";" if ";" in head else "\t" if "\t" in head else None
            if sep:
                df = pd.read_csv(p, sep=sep, encoding="utf-8-sig")
    df = df.dropna(how="all")
    return inclusion_list_frame(df, source=os.path.basename(p))


# ---------------------------------------------------------------------------
# the method layout: experiments (event blocks) and the Full Scan
# ---------------------------------------------------------------------------
@dataclass
class Block:
    """One experiment's events (first..last, its NCE, its list rows), or the
    Full Scan (`experiment` None). `exact`: its first and last events are
    the method's (the Full Scan always; an experiment when the layout is
    `closed`), so a walk may wrap from its last event to its first."""
    first: int
    last: int
    nce: float = math.nan
    experiment: int | None = None
    rows: np.ndarray | None = None              # the list's m/z by slot, when given
    exact: bool = True

    @property
    def size(self) -> int:
        return self.last - self.first + 1

    def holds(self, e: int) -> bool:
        return self.first <= e <= self.last

    def next(self, e: int) -> int:
        return e + 1 if e < self.last else self.first

    def prev(self, e: int) -> int:
        return e - 1 if e > self.first else self.last


@dataclass
class Layout:
    """The MS2 experiments (event blocks) and the Full Scan. `closed`: every
    experiment's event range is the method's -- a list sized by the Full Scan
    event (one shared list of n rows, Full Scan = K*n + 1; or lists summing to
    the Full Scan event - 1). Otherwise (no list, no Full Scan event) an
    experiment never seen, two experiments at one NCE, or a block's edge
    event can be missing from it (`open_why` says why), and
    `infer_headerless` walks only straight runs inside one block."""
    blocks: list                                # MS2 experiments, in event order
    full_scan: Block | None
    source: str                                 # 'inclusion list' | 'events'
    notes: list = field(default_factory=list)
    closed: bool = False
    open_why: str = ""

    def all_blocks(self) -> list:
        return self.blocks + ([self.full_scan] if self.full_scan is not None else [])

    def block_of(self, e) -> Block | None:
        if e is None or not math.isfinite(float(e)):
            return None
        e = int(e)
        for b in self.all_blocks():
            if b.holds(e):
                return b
        return None


def _event_nces(scans: pd.DataFrame) -> dict[int, set]:
    h = scans[scans["has_header"] & (scans["order"] == 2) & scans["event"].notna()]
    out: dict[int, set] = {}
    for e, nce in zip(h["event"].astype(int), h["nce"]):
        out.setdefault(int(e), set()).add(round(float(nce), 4) if math.isfinite(nce) else math.nan)
    return out


def _full_scan_event(scans: pd.DataFrame) -> int | None:
    h = scans[scans["has_header"] & (scans["order"] == 1) & scans["event"].notna()]
    ev = sorted({int(e) for e in h["event"]})
    if len(ev) > 1:
        raise ScanMapError(f"the MS1 scans carry {len(ev)} scan events ({_events_text(ev)}): "
                               "a method with one Full Scan event is expected")
    return ev[0] if ev else None


def _nce_runs(nces: dict[int, set]) -> list[list]:
    """[nce, first, last] runs of one NCE over the MS2 events seen, in event order."""
    runs: list[list] = []
    for e in sorted(nces):
        v = next(iter(nces[e]))
        if runs and runs[-1][0] == v:
            runs[-1][2] = e
        else:
            runs.append([v, e, e])
    return runs


def _even_split(runs: list[list], fs: int | None) -> int | None:
    """The rows per experiment when the Full Scan event splits the NCE runs
    evenly (K runs, Full Scan = K*n + 1, run k inside events (k-1)n+1..kn),
    else None."""
    k = len(runs)
    if fs is None or not k or (fs - 1) % k:
        return None
    n = (fs - 1) // k
    ok = all(i * n < r[1] and r[2] <= (i + 1) * n for i, r in enumerate(runs))
    return n if ok else None


def _interleaved(scans: pd.DataFrame, a: Block, b: Block) -> tuple[int, Block] | None:
    """(the first known scan that returns to a block after the other one's
    known scans began, that block) when the known scans of blocks `a` and
    `b` overlap in time -- the two alternate -- else None. Two blocks that
    run one after the other, in either order, do not alternate: time order
    is free (the Full Scan, the last event, runs first)."""
    h = scans[scans["has_header"] & (scans["order"] == 2) & scans["event"].notna()]
    ev = h["event"].astype(int)
    sa = h.loc[ev.map(a.holds), "scan"]
    sb = h.loc[ev.map(b.holds), "scan"]
    if not len(sa) or not len(sb) or sa.max() < sb.min() or sb.max() < sa.min():
        return None
    first, other, blk = (sa, sb, a) if sa.min() < sb.min() else (sb, sa, b)
    return int(first[first > other.min()].min()), blk


def _rule_checks_rows(scans: pd.DataFrame, blocks: list, fs: int | None, pair) -> dict | None:
    """The scan-range rule's fit of the list (`rule_fit`, `RULE_TOL_MDA`) over
    the events of both experiments of `pair` seen with a trailer, when the
    rule follows the precursor in both (`rule_verdicts`) and the fit has
    events: the rule has then checked their rows -- it fits (`events_off`
    empty) or names the events the list puts off it. None when the rule
    cannot check them (a fixed or unjudged range, no width known)."""
    tmp = Layout(blocks=blocks, full_scan=Block(fs, fs) if fs is not None else None,
                 source=PRECURSOR_LIST)
    events = _event_table(scans, tmp)
    verdicts = rule_verdicts(events)
    exps = {b.experiment for b in pair}
    if any(verdicts.get(e) != RULE_FOLLOWS for e in exps):
        return None
    fit = rule_fit(events[events["experiment"].isin(exps) & (events["n_header"] > 0)])
    return fit if fit["n_events"] else None


def method_layout(scans: pd.DataFrame, lists: list | None = None, *,
                  rule_check: bool = True) -> Layout:
    """The experiments the MS2 events belong to.

    With lists: one list shared by every experiment (n rows: experiment k =
    events (k-1)n+1..kn) or one list per experiment, in event order. The
    Full Scan, when the method has one, must be the event right after the
    last experiment's last row -- else the list does not fit the method
    (rows missing or extra) and that is an error. Without a list the blocks
    are read off the events: runs of one NCE in event order, sized equally
    when the Full Scan event says so (K runs, Full Scan = K*n + 1), else
    spanning their observed events (noted). Each experiment must carry one
    NCE: an experiment whose events read two is an error naming them.

    The layout is `closed` only when a list and the Full Scan event pin every
    experiment's range. Without a list it never is: two adjacent experiments
    at one NCE are one NCE run, an experiment never seen with a trailer is
    none, and the Full Scan event splits what is left 'evenly' all the same
    (one run of events 1-4 and a Full Scan event 9 read as one 8-row
    experiment where the method ran two 4-row ones). A shared
    list is also refused when two of the experiments it implies carry one
    NCE and their scans alternate in time: one experiment of the method then
    spans them, and the list's row count only divides the method's (a wrong
    list, or rows missing) -- on a method whose scan range does not follow
    the precursor nothing else would catch it. Where the range follows the
    precursor in both and the list fits the scan-range rule on every event
    of theirs seen (`_rule_checks_rows`; `rule_check`), the rule has checked
    the rows: the two are taken for experiments that run in parallel (noted);
    where it follows the precursor in both and the list misses it, the
    refusal names the events off the rule (the rows are wrong). On a fixed
    range the refusal says such a method takes the list once per
    experiment."""
    nces = _event_nces(scans)
    fs = _full_scan_event(scans)
    stepped = sorted(e for e, v in nces.items() if any(not math.isfinite(x) for x in v))
    if stepped:
        raise ScanMapError(f"event(s) {_events_text(stepped)} carry a stepped or unreadable "
                           "collision energy: one NCE per experiment is required")
    notes: list[str] = []
    blocks: list[Block] = []
    runs = _nce_runs(nces)
    closed, open_why = False, ""
    if lists and not nces:
        # an ordinary MS2 file (its MS2 scans get no trailer row): no MS2
        # event to fit a list to, nor to anchor a walk -- the list says nothing
        named = (scans["has_header"] & (scans["order"] == 2)).any()
        notes.append(("no MS2 trailer row names a scan event" if named else
                      "no MS2 scan has a trailer row")
                     + ": the inclusion list is not used (no event to fit it to)")
        lists = None
    if lists:
        frames = [inclusion_list_frame(x) if not set(LIST_COLUMNS) <= set(x.columns) else x
                  for x in lists]
        sizes = [len(f) for f in frames]
        if any(s == 0 for s in sizes):
            raise ScanMapError("an inclusion list has no rows")
        top, bottom = max(nces), min(nces)
        if fs is not None and fs <= top:
            # a list's event -> row mapping is the numbering of an
            # inclusion-list method, the Full Scan last: nothing to count
            # before a Full Scan that is not last -- one shared list or one
            # per experiment, the lists are not what is wrong
            raise ScanMapError(
                f"the Full Scan is event {fs}, not after the MS2 events ({bottom}-{top}): "
                "a list maps an inclusion-list method's numbering (the MS2 experiments' "
                "events from 1, the Full Scan last), and this method numbers its events "
                "otherwise")
        if len(frames) == 1:
            n = sizes[0]
            if fs is not None:
                m = fs - 1
                if m % n:
                    even = _even_split(runs, fs)
                    hint = (f"; its events read {len(runs)} runs of one NCE (NCE "
                            + "/".join(f"{r[0]:g}" for r in runs) + f"): a shared list of {even} "
                            f"rows, or {len(runs)} lists summing to {m} rows, would fit"
                            if even else "")
                    raise ScanMapError(
                        f"the method has {m} MS2 event(s) before its Full Scan (event {fs}), not "
                        f"a whole number of {n}-row experiments: this {n}-row list does not fit "
                        f"the method (wrong list, or rows missing / extra){hint}")
                k = m // n
                closed = True
            else:
                k = max(1, -(-top // n))
                open_why = ("no Full Scan event was seen, so the list cannot say how many "
                            "experiments the method has")
            if top > k * n:
                raise ScanMapError(f"MS2 event {top} lies past the {k} experiment(s) of {n} "
                                   "rows the list gives: the list does not fit the method")
            arr = frames[0]["mz"].to_numpy(dtype=float)
            blocks = [Block((i * n) + 1, (i + 1) * n, experiment=i + 1, rows=arr)
                      for i in range(k)]
        else:
            off = 0
            for i, f in enumerate(frames):
                blocks.append(Block(off + 1, off + len(f), experiment=i + 1,
                                    rows=f["mz"].to_numpy(dtype=float)))
                off += len(f)
            if fs is not None and fs != off + 1:
                raise ScanMapError(
                    f"the Full Scan is event {fs}, but the {len(frames)} lists end at event {off}: "
                    "the lists do not fit the method (one list per experiment, in event order)")
            if top > off:
                raise ScanMapError(f"MS2 event {top} lies past the {off} rows of the "
                                   f"{len(frames)} lists: the lists do not fit the method")
            closed = fs is not None
            if not closed:
                open_why = ("no Full Scan event was seen, so nothing says the lists are every "
                            "experiment of the method")
        source = PRECURSOR_LIST
    else:
        # without a list nothing pins the experiments: two adjacent experiments
        # at one NCE read as one NCE run, and an experiment never seen with a
        # trailer as none -- the Full Scan event then splits the runs 'evenly'
        # into experiments the method does not have. The even split only sizes
        # the blocks (a switch into a block whose first event was never seen
        # is then no break of the cycle); the layout stays open
        n = _even_split(runs, fs)
        if n is not None:
            blocks = [Block(i * n + 1, (i + 1) * n, experiment=i + 1) for i in range(len(runs))]
            open_why = ("no list: the experiments are the NCE runs of the events seen, sized by "
                        "the Full Scan event -- two experiments at one NCE would read as one, "
                        "and an experiment never seen as none")
        else:
            blocks = [Block(r[1], r[2], experiment=i + 1) for i, r in enumerate(runs)]
            if runs:
                open_why = ("experiment boundaries read off the observed events only (no list, "
                            "and " + ("no Full Scan event was seen" if fs is None else
                                      "the Full Scan event does not split them evenly")
                            + "): an experiment never seen, or an event never seen at a "
                            "block's edge, belongs to no experiment")
        if open_why:
            notes.append(open_why)
        source = "events"
    for b in blocks:
        b.exact = closed
    # one NCE per experiment
    bad = []
    for b in blocks:
        vals = {v for e, s in nces.items() if b.holds(e) for v in s}
        if len(vals) > 1:
            evs = sorted(e for e in nces if b.holds(e))
            bad.append(f"experiment {b.experiment} (events {b.first}-{b.last}) reads NCE "
                       + "/".join(f"{v:g}" for v in sorted(vals)) + f" over events {_events_text(evs)}")
        elif vals:
            b.nce = float(next(iter(vals)))
        else:
            notes.append(f"experiment {b.experiment} (events {b.first}-{b.last}) has no scan with a "
                         "trailer: its NCE is unknown and its scans stay unassigned")
    if bad:
        raise ScanMapError("every experiment must carry one NCE: " + "; ".join(bad)
                           + (" -- the list does not fit the method (wrong list, row count "
                              "or order)" if lists else ""))
    if lists and len(lists) == 1:
        # a shared list whose row count only divides the method's: the
        # experiments it implies split one of the method's, so two of them
        # carry its one NCE and their scans alternate in time. Two experiments
        # of one NCE that run in parallel alternate too: where the scan range
        # follows the precursor in both and fits the list row by row, the
        # rule has checked the rows (a divisor list misses it by the spacing
        # of its rows), so the alternation says nothing against the list
        for a, b in pairwise(blocks):
            if not (math.isfinite(a.nce) and a.nce == b.nce):
                continue
            alt = _interleaved(scans, a, b)
            if alt is not None:
                s, back = alt
                fit = _rule_checks_rows(scans, blocks, fs, (a, b)) if rule_check else None
                head = (f"experiments {a.experiment} (events {a.first}-{a.last}) and "
                        f"{b.experiment} (events {b.first}-{b.last}) of this {a.size}-row list "
                        f"both read NCE {a.nce:g} and their scans alternate in time (scan {s} "
                        f"returns to experiment {back.experiment})")
                if fit is not None and not fit["events_off"]:
                    notes.append(
                        f"experiments {a.experiment} and {b.experiment} of the list both read NCE "
                        f"{a.nce:g} and their scans alternate in time (two experiments run in "
                        "parallel): the scan-range rule checks every row of both against the "
                        "list")
                    continue
                if fit is not None:
                    # the range follows the precursor in both: the rule has
                    # checked the rows, and the list misses it -- two parallel
                    # experiments run with this list would have fitted it
                    raise ScanMapError(
                        head + ", and the list's rows do not fit the scan-range rule (event(s) "
                        f"{_events_text(fit['events_off'])} of {fit['n_events']}, max residual "
                        f"{fit['max_residual_mda']:.3g} mDa > {RULE_TOL_MDA:g}): one experiment "
                        "of the method spans them, so the list's row count only divides the "
                        "method's -- wrong list, or rows missing")
                raise ScanMapError(
                    head + ": one experiment of the method spans them, so "
                    "the list's row count only divides the method's -- wrong list, or rows "
                    "missing. Unless the method runs two experiments at this NCE in parallel "
                    "(each cycle through one, then the other): then give the list once per "
                    "experiment (the same list twice); the scan range does not follow the "
                    "precursor here" + ("" if rule_check else " (or its check is off)")
                    + ", so nothing in the file tells the two apart")
            notes.append(f"experiments {a.experiment} and {b.experiment} of the list both read "
                         f"NCE {a.nce:g} (one after the other in time)")
    if not closed and open_why and open_why not in notes:
        notes.append(open_why)
    full = Block(fs, fs) if fs is not None else None
    return Layout(blocks=blocks, full_scan=full, source=source, notes=notes, closed=closed,
                  open_why=open_why)


# ---------------------------------------------------------------------------
# events: what each one is (NCE, width, range, precursor)
# ---------------------------------------------------------------------------
def _median(values) -> float:
    v = [float(x) for x in values if math.isfinite(_num(x))]
    return float(np.median(v)) if v else math.nan


def rule_precursor(high_mass: float, width: float) -> float:
    """The precursor the scan-range rule implies: (HighMass - offset) / slope
    - width / 2."""
    return (high_mass - RULE_OFFSET_DA) / RULE_SLOPE - width / 2.0


def rule_high_mass(precursor: float, width: float) -> float:
    return RULE_SLOPE * (precursor + width / 2.0) + RULE_OFFSET_DA


EVENT_COLUMNS: dict[str, str] = {
    "event": "int64", "experiment": "int64", "slot": "int64", "nce": "float64",
    "n_header": "int64", "hcd_v": "float64", "iso_width": "float64", "width_from": "str",
    "low_mass": "float64", "high_mass": "float64", "filter_mz": "float64",
    "filter_varies": "bool", "list_mz": "float64",
}


def _event_table(scans: pd.DataFrame, layout: Layout) -> pd.DataFrame:
    """One row per MS2 event of the layout: experiment, slot, NCE, width, the
    scan range its header rows read, the filter's and the list's precursor."""
    h = scans[scans["has_header"] & (scans["order"] == 2) & scans["event"].notna()]
    by = {int(e): g for e, g in h.groupby(h["event"].astype(int))}
    widths = h["iso_width"][np.isfinite(h["iso_width"])]
    file_width = float(widths.median()) if len(widths) else math.nan
    rows = []
    for b in layout.blocks:
        bw = [x for e in range(b.first, b.last + 1) if e in by
              for x in by[e]["iso_width"] if math.isfinite(x)]
        block_width = float(np.median(bw)) if bw else file_width
        for e in range(b.first, b.last + 1):
            g = by.get(e)
            slot = e - b.first + 1
            fp = g["filter_parent_mz"].dropna() if g is not None else pd.Series(dtype=float)
            # one precursor per event (an inclusion list), or one per scan (a
            # data-dependent event): the scans keep their own either way. But
            # an event is one scan range: a data-dependent event whose range
            # follows each precursor reads several, and one HighMass for the
            # event would name none of them (nor check a walk onto it) --
            # refused, on a fixed range only it maps scan by scan
            one_fp = bool(len(fp)) and (fp.max() - fp.min()) * 1e3 <= SOURCE_MATCH_MDA
            hm_all = [float(x) for x in (g["high_mass"] if g is not None else [])
                      if math.isfinite(x)]
            hm = sorted({round(x / RULE_DISTINCT_DA) * RULE_DISTINCT_DA for x in hm_all})
            if len(hm) > 1:
                raise ScanMapError(f"event {e}: its scans read {len(hm)} scan ranges (HighMass "
                                   + ", ".join(f"{x:g}" for x in hm) + "): one event, one "
                                   "scan range is expected (a data-dependent event maps scan "
                                   "by scan on a fixed range only)")
            hm = [float(np.median(hm_all))] if hm_all else []
            width = _median(g["iso_width"]) if g is not None else math.nan
            rows.append({
                "event": e, "experiment": b.experiment, "slot": slot, "nce": b.nce,
                "n_header": 0 if g is None else len(g),
                "hcd_v": _median(g["hcd_v"]) if g is not None else math.nan,
                "iso_width": width if math.isfinite(width) else block_width,
                "width_from": "trailer" if math.isfinite(width) else (
                    "experiment" if bw else "file" if math.isfinite(file_width) else ""),
                "low_mass": float(g["low_mass"].min()) if g is not None else math.nan,
                "high_mass": hm[0] if hm else math.nan,
                "filter_mz": float(fp.median()) if one_fp else math.nan,
                "filter_varies": bool(len(fp)) and not one_fp,
                "list_mz": (float(b.rows[slot - 1]) if b.rows is not None and slot <= len(b.rows)
                            else math.nan),
            })
    return _frame({c: [r[c] for r in rows] for c in EVENT_COLUMNS}, EVENT_COLUMNS)


def rule_applies(events: pd.DataFrame) -> bool:
    """At least two distinct HighMass values across the MS2 events of the
    file: a method whose every event reads one HighMass (one fixed range)
    gives one. The file-wide half of `rule_verdicts`."""
    hm = events["high_mass"][np.isfinite(events["high_mass"])]
    return len({round(float(x) / RULE_DISTINCT_DA) for x in hm}) >= 2


RULE_FOLLOWS, RULE_FIXED, RULE_UNKNOWN = "follows the precursor", "fixed range", "unknown"


def rule_verdicts(events: pd.DataFrame) -> dict[int, str]:
    """Per experiment, whether its scan range follows the precursor -- the scan
    range is a setting of each experiment, so one fixed-range experiment
    beside rule-following ones is a valid method: `RULE_FOLLOWS` when two or
    more of its events were seen with a trailer and their HighMass values
    differ, `RULE_FIXED` when two or more were seen and every one reads one
    HighMass (derived precursors would all coincide), `RULE_UNKNOWN` with one
    event seen or none (one HighMass says nothing). Values are compared at
    `RULE_DISTINCT_DA`."""
    out: dict[int, str] = {}
    for exp, g in events.groupby("experiment", sort=True):
        hm = [round(float(x) / RULE_DISTINCT_DA) for x in g["high_mass"] if math.isfinite(x)]
        out[int(exp)] = (RULE_UNKNOWN if len(hm) < 2 else
                         RULE_FIXED if len(set(hm)) == 1 else RULE_FOLLOWS)
    return out


def rule_holds(verdicts: dict[int, str], file_distinct: bool) -> set:
    """The experiments the scan-range rule is taken to hold in: every
    `RULE_FOLLOWS` one, and an `RULE_UNKNOWN` one (one event seen) when the
    file's HighMass values differ and no experiment reads a fixed range."""
    any_fixed = RULE_FIXED in verdicts.values()
    return {e for e, v in verdicts.items()
            if v == RULE_FOLLOWS or (v == RULE_UNKNOWN and file_distinct and not any_fixed)}


def rule_text(rule: dict) -> str:
    """A mapping report's `rule` in words (`ms2 files --check`, the console):
    where the scan range follows the precursor, and its fit."""
    exps = rule.get("experiments") or {}
    fixed = sorted(int(e) for e, v in exps.items() if v == RULE_FIXED)
    fit = (f" (max residual {rule['max_residual_mda']:.3g} mDa)"
           if rule.get("max_residual_mda") is not None else "")
    if not rule.get("applies"):
        return ("does not apply (" + (f"a fixed range in experiment(s) {_events_text(fixed)}"
                                      if fixed else "one scan range for every event") + ")")
    if fixed:
        return (f"follows the precursor{fit}, except in experiment(s) {_events_text(fixed)} "
                "(a fixed range)")
    return "follows the precursor" + fit


def rule_summary(scans: pd.DataFrame) -> str:
    """`rule_text` of a file whose mapping was refused: the verdicts its MS2
    header rows give (no list), else -- when even the method layout is
    refused -- whether their HighMass values differ at all."""
    try:
        events = _event_table(scans, method_layout(scans))
    except ScanMapError:
        h = scans[scans["has_header"] & (scans["order"] == 2)]
        hm = {round(float(x) / RULE_DISTINCT_DA) for x in h["high_mass"] if math.isfinite(x)}
        return ("HighMass differs between the events (not judged: the mapping was refused)"
                if len(hm) >= 2 else "does not apply (one scan range for every event)")
    verdicts = rule_verdicts(events)
    return rule_text({"applies": bool(rule_holds(verdicts, rule_applies(events))),
                      "experiments": verdicts})


def _shared_high_mass(events: pd.DataFrame) -> set:
    """The seen events whose HighMass another seen event of their experiment
    shares: a precursor derived from it would coincide with that one's."""
    out = set()
    e = events[np.isfinite(events["high_mass"])]
    keys = list(zip(e["experiment"].astype(int),
                    (e["high_mass"] / RULE_DISTINCT_DA).round().astype(np.int64)))
    counts: dict = {}
    for k in keys:
        counts[k] = counts.get(k, 0) + 1
    for ev, k in zip(e["event"].astype(int), keys):
        if counts[k] > 1:
            out.add(int(ev))
    return out


def rule_fit(events: pd.DataFrame, mz_col: str = "list_mz") -> dict:
    """HighMass = slope x (precursor + width / 2) + offset by least squares over
    the events that have both; the residuals of the FIXED rule (1.02, 20) in
    mDa: their max and the events beyond `RULE_TOL_MDA`."""
    e = events[np.isfinite(events["high_mass"]) & np.isfinite(events[mz_col])
               & np.isfinite(events["iso_width"])]
    out = {"n_events": len(e), "slope": None, "offset_da": None,
           "max_residual_mda": None, "events_off": []}
    if not len(e):
        return out
    x = e[mz_col].to_numpy(dtype=float) + e["iso_width"].to_numpy(dtype=float) / 2.0
    y = e["high_mass"].to_numpy(dtype=float)
    res = (y - (RULE_SLOPE * x + RULE_OFFSET_DA)) * 1e3
    out["max_residual_mda"] = float(np.max(np.abs(res)))
    out["events_off"] = [int(v) for v in e["event"].to_numpy()[np.abs(res) > RULE_TOL_MDA]]
    if len(e) >= 2 and np.ptp(x) > 0:
        a, b = np.polyfit(x, y, 1)
        out["slope"], out["offset_da"] = float(a), float(b)
    return out


# ---------------------------------------------------------------------------
# headerless scans: the cycle walk
# ---------------------------------------------------------------------------
def _walk(block: Block, start: int, steps: int) -> list[int]:
    out, e = [], start
    for _ in range(steps):
        e = block.next(e)
        out.append(e)
    return out


def _gap_walks(a: int, ba: Block, b: int, bb: Block, g: int) -> list[list[int]]:
    """Every event sequence of the g scans between a known scan of event `a`
    and the next known scan of event `b` that the cycle allows: the walk on
    in a's block, or -- when b is in another block -- the walk on in a's block
    up to a switch that starts b's block at its first event."""
    out = []
    if ba is bb:
        seq = _walk(ba, a, g + 1)
        if seq[-1] == b:
            out.append(seq[:-1])
        return out
    for p in range(1, g + 2):                 # position (1-based) of the switch
        head = _walk(ba, a, p - 1)
        tail = [bb.first] + _walk(bb, bb.first, g + 1 - p)
        if tail[-1] == b:
            out.append(head + tail[:-1])
    return out


def _range_ok(scans: pd.DataFrame, i: int, low: float, high: float) -> bool | None:
    """The scan's centroids lie in [low, high] (`RANGE_CHECK_DA`); None when
    there is nothing to check (no centroid, or the range is unknown)."""
    lo, hi = scans.at[i, "min_centroid_mz"], scans.at[i, "max_centroid_mz"]
    if not math.isfinite(hi) or not math.isfinite(high):
        return None
    if hi > high + RANGE_CHECK_DA:
        return False
    if math.isfinite(low) and math.isfinite(lo) and lo < low - RANGE_CHECK_DA:
        return False
    return True


def _unseen_ranges(events: pd.DataFrame | None, rule_check: bool = True) -> dict[int, tuple]:
    """Event -> (the HighMass a walk onto it is checked against, the HighMass
    a scan placed on it keeps) for each event never seen with a trailer: in an
    experiment whose events all read one fixed range, that range (it is the
    experiment's, recorded); where the scan-range rule holds and is checked
    (`rule_holds`, `rule_check`), the HighMass the rule predicts from the
    event's list row -- a check only, never a range the scan recorded (NaN);
    else none (the walk cannot be checked: `runs_unverifiable`)."""
    if events is None or not len(events):
        return {}
    verdicts = rule_verdicts(events)
    hold = rule_holds(verdicts, rule_applies(events)) if rule_check else set()
    fixed = {}
    for exp, g in events.groupby("experiment"):
        hm = g["high_mass"][np.isfinite(g["high_mass"])]
        if verdicts.get(int(exp)) == RULE_FIXED and len(hm):
            fixed[int(exp)] = float(hm.median())
    out: dict[int, tuple] = {}
    for r in events.itertuples():
        if math.isfinite(r.high_mass):
            continue
        exp = int(r.experiment)
        if exp in fixed:
            out[int(r.event)] = (fixed[exp], fixed[exp])
        elif exp in hold and math.isfinite(r.list_mz) and math.isfinite(r.iso_width):
            out[int(r.event)] = (rule_high_mass(r.list_mz, r.iso_width), math.nan)
        else:
            out[int(r.event)] = (math.nan, math.nan)
    return out


def infer_headerless(scans: pd.DataFrame, layout: Layout, events: pd.DataFrame | None = None,
                     *, rule_check: bool = True) -> tuple[pd.DataFrame, dict]:
    """Place every scan with no trailer row by walking the scan cycle.

    The model (`Layout`): inside a block the events follow each other in
    order, the last by the first; a block switch starts the next block at its
    first event. It is checked first on every pair of consecutive known scans:
    one transition it does not allow and nothing is inferred (noted). A run of
    headerless scans between two known scans takes the ONE event sequence the
    cycle allows (`_gap_walks`); none or several -> left unassigned. A run
    that crosses a block switch, and a leading / trailing run (walked back /
    on in the first / last block), are placed only when the layout is
    `closed`, every block has a known scan, and no trailer row names a scan
    of no block: such a row shows an experiment outside the layout (e.g. a
    tSIM one) running in the file, and a block of it whose scans lost their
    rows could hide in such a run (`runs_beside_outside_scans`, said). An
    experiment outside the layout none of whose scans keeps a row is
    invisible: nothing in the file tells its scans from the next events of
    the block before them, unless their centroids contradict those events'
    scan ranges. On a layout
    that is not closed a block's edges may not be the method's: only a run
    inside one block, straight from one known scan to the next (a -> a+1 ->
    ... -> b, no wrap), is placed. Each inferred scan is then checked
    against its event's scan range: LowMass..HighMass of its header rows; for
    an event never seen with one, `_unseen_ranges` (its experiment's fixed
    range, or -- where the rule holds and `rule_check` -- the HighMass the
    rule predicts from the list, which the scan does not keep). A centroid
    outside it, or no range to check against, leaves the whole run
    unassigned. A scan WITH a trailer row that is not known -- its trailer
    gives no MS order (e.g. a 'tSIM' experiment: no HCD energy on a node that
    is no Full Scan) or no event -- belongs to no block: the walk never
    crosses it, and a gap that holds one leaves its headerless scans
    unassigned (`runs_blocked_by_header_rows`); only scans with no trailer row
    are ever written. Every headerless scan is counted: inferred or not, and
    why."""
    scans = scans.copy()
    rep = {"headerless": int((~scans["has_header"]).sum()), "inferred": 0, "unassigned": 0,
           "unassigned_scans": [], "runs": 0, "runs_ambiguous": 0, "runs_inconsistent": 0,
           "runs_failed_range": 0, "runs_unverifiable": 0, "runs_blocked_by_header_rows": 0,
           "runs_beside_outside_scans": 0, "model_breaks": 0,
           "layout_closed": bool(layout.closed), "notes": []}
    if not rep["headerless"]:
        return scans, rep
    known = scans.index[scans["has_header"] & scans["event"].notna() & scans["order"].notna()]
    ev_info = {} if events is None else {int(r.event): r for r in events.itertuples()}
    unseen = _unseen_ranges(events, rule_check)
    fs = layout.full_scan
    fs_rows = scans[scans["has_header"] & (scans["order"] == 1)]
    fs_low = float(fs_rows["low_mass"].min()) if len(fs_rows) else math.nan
    fs_high = float(fs_rows["high_mass"].max()) if len(fs_rows) else math.nan

    def block(i):
        return layout.block_of(scans.at[i, "event"])

    unassigned = list(scans.index[~scans["has_header"]])

    def give_up(note: str):
        rep["notes"].append(note)
        rep["unassigned"], rep["unassigned_scans"] = len(unassigned), [
            int(scans.at[i, "scan"]) for i in unassigned]
        return scans, rep

    if not len(known):
        return give_up("no scan with a trailer to walk the cycle from: headerless scans left "
                       "unassigned")
    if not layout.blocks:
        # an ordinary file: its MS2 scans have no trailer row at all, and a walk
        # over the MS1 rows alone would call every one of them a Full Scan
        return give_up("no MS2 event has a trailer row: the scan cycle is unknown, headerless "
                       "scans left unassigned")
    # the model must hold on every pair of consecutive known scans
    pos = {i: k for k, i in enumerate(scans.index)}
    breaks = []
    for i, j in pairwise(known):
        if pos[j] != pos[i] + 1:
            continue
        ba, bb = block(i), block(j)
        a, b = int(scans.at[i, "event"]), int(scans.at[j, "event"])
        if ba is None or bb is None or not (b == ba.next(a) if ba is bb else b == bb.first):
            breaks.append(int(scans.at[j, "scan"]))
    if breaks:
        rep["model_breaks"] = len(breaks)
        return give_up(f"{len(breaks)} pair(s) of consecutive scans break the scan cycle (at "
                       f"scan(s) {_events_text(breaks)}): headerless scans left unassigned")
    # each block runs once, in one time block: a block the known scans return
    # to could hide in any run of headerless scans
    seq = []
    for i in known:
        b = block(i)
        if not seq or seq[-1] is not b:
            seq.append(b)
    if len({id(b) for b in seq}) < len(seq):
        rep["model_breaks"] = 1
        return give_up("the known scans return to a block they left (blocks interleave in "
                       "time): headerless scans left unassigned")
    seen = {id(block(i)) for i in known}
    # every block of the method is known and seen, and the file shows no scan
    # of an experiment outside the layout: no block can hide in a run. A
    # trailer row with no MS order or no event (e.g. a tSIM experiment's) is
    # such a scan -- the list and the Full Scan event pin the list's
    # experiments only, and a block of that experiment that lost its rows
    # would be walked over as the next events of the block before it
    all_seen = all(id(b) in seen for b in layout.all_blocks())
    outside = int((scans["has_header"] & (scans["event"].isna() | scans["order"].isna())).sum())
    complete = layout.closed and all_seen and not outside
    if not layout.closed:
        rep["notes"].append(
            "the experiments' event ranges are not pinned (only the method's list, sized by the "
            "Full Scan event, pins them): only runs of headerless scans inside one block, "
            "straight from one of its known scans to the next, are placed")
    else:
        if not all_seen:
            rep["notes"].append("a block has no scan with a trailer: runs of headerless scans "
                                "that could hide it are left unassigned")
        if outside:
            rep["notes"].append(
                f"{outside} scan(s) with a trailer row belong to no block (no MS order or no "
                "event, e.g. a SIM experiment): an experiment outside the layout runs in this "
                "file, and a block of it whose scans lost their rows could hide in a run that "
                "crosses a block switch, or is leading or trailing -- such runs are left "
                "unassigned; runs inside one block are still placed")
    # why a run that needs every block known is not walked
    not_walked = ("runs_ambiguous" if not (layout.closed and all_seen)
                  else "runs_beside_outside_scans")
    idx = list(scans.index)
    header = scans["has_header"].to_numpy()
    plans: list[tuple[list, list]] = []       # (scan indices, events)
    # runs between known scans, and the leading / trailing runs
    bounds = [(None, known[0]), *pairwise(known), (known[-1], None)]
    for i, j in bounds:
        lo = 0 if i is None else pos[i] + 1
        hi = len(idx) if j is None else pos[j]
        run = idx[lo:hi]
        if not run:
            continue
        if header[lo:hi].any():
            # a scan of no block lies in the gap (its trailer names no MS
            # order or no event): where the cycle stood around it is unknown
            if not header[lo:hi].all():
                rep["runs"] += 1
                rep["runs_blocked_by_header_rows"] += 1
            continue
        rep["runs"] += 1
        g = len(run)
        if i is not None and j is not None:
            ba, bb = block(i), block(j)
            a, b = int(scans.at[i, "event"]), int(scans.at[j, "event"])
            if ba is not bb and not complete:
                rep[not_walked] += 1
                continue
            if ba is bb and not ba.exact:
                # the block's edges may not be the method's: no wrap, only the
                # straight walk a -> a+1 -> ... -> b (unique when it fits)
                if b - a == g + 1:
                    plans.append((run, list(range(a + 1, b))))
                else:
                    rep["runs_ambiguous"] += 1
                continue
            walks = _gap_walks(a, ba, b, bb, g)
        elif j is not None:                   # leading: walk back in the first block
            if not complete:
                rep[not_walked] += 1
                continue
            bb, e, seq = block(j), int(scans.at[j, "event"]), []
            for _ in range(g):
                e = bb.prev(e)
                seq.append(e)
            walks = [seq[::-1]]
        else:                                 # trailing: walk on in the last block
            if not complete:
                rep[not_walked] += 1
                continue
            walks = [_walk(block(i), int(scans.at[i, "event"]), g)]
        if len(walks) != 1:
            rep["runs_ambiguous" if walks else "runs_inconsistent"] += 1
            continue
        plans.append((run, walks[0]))
    # check each planned run against the scan ranges
    for run, seq in plans:
        verdict = True
        for i, e in zip(run, seq):
            if fs is not None and e == fs.first:
                ok = _range_ok(scans, i, fs_low, fs_high)
            else:
                r = ev_info.get(e)
                if r is None:
                    ok = None
                else:
                    high = unseen.get(e, (r.high_mass,))[0]
                    ok = _range_ok(scans, i, r.low_mass, high)
                    if ok is None and scans.at[i, "n_centroids"] == 0 and math.isfinite(high):
                        ok = True             # an empty scan: nothing contradicts the walk
            if ok is not True:
                verdict = ok
                break
        if verdict is not True:
            rep["runs_failed_range" if verdict is False else "runs_unverifiable"] += 1
            continue
        for i, e in zip(run, seq):
            if fs is not None and e == fs.first:
                scans.loc[i, ["order", "event"]] = [1.0, float(e)]
                scans.at[i, "node"] = "Full Scan"
                scans.at[i, "low_mass"], scans.at[i, "high_mass"] = fs_low, fs_high
            else:
                r = ev_info.get(e)
                b = layout.block_of(e)
                scans.loc[i, ["order", "event", "nce"]] = [2.0, float(e), b.nce]
                if r is not None:
                    scans.at[i, "iso_width"] = r.iso_width
                    scans.at[i, "hcd_v"] = r.hcd_v
                    scans.at[i, "low_mass"] = r.low_mass
                    scans.at[i, "high_mass"] = unseen.get(e, (None, r.high_mass))[1]
            scans.at[i, "inferred"] = True
            rep["inferred"] += 1
    left = scans.index[~scans["has_header"] & ~scans["inferred"]]
    rep["unassigned"] = len(left)
    rep["unassigned_scans"] = [int(scans.at[i, "scan"]) for i in left]
    if rep["unassigned"]:
        why = [f"{rep[k]} {t}" for k, t in (
            ("runs_ambiguous", "run(s) the cycle allows several ways"),
            ("runs_inconsistent", "run(s) no walk fits"),
            ("runs_failed_range", "run(s) with a centroid outside the walked event's scan range"),
            ("runs_unverifiable", ("run(s) through an event with no scan range to check "
                                   "against (never seen with a trailer, in an experiment with "
                                   "neither one fixed range nor a checked scan-range rule and "
                                   "list to predict it)")),
            ("runs_blocked_by_header_rows", ("run(s) holding a scan whose trailer gives no MS "
                                             "order or no event (e.g. a SIM scan): it is in no "
                                             "block, so the walk cannot cross it")),
            ("runs_beside_outside_scans", ("run(s) across a block switch, leading or trailing, "
                                           "where a block of the experiment outside the layout "
                                           "could hide"))) if rep[k]]
        rep["notes"].append(f"{rep['unassigned']} of {rep['headerless']} headerless scan(s) left "
                            f"unassigned (scan(s) {_events_text(rep['unassigned_scans'])}"
                            + (": " + ", ".join(why) if why else "") + "): they are in no spectrum")
    return scans, rep


# ---------------------------------------------------------------------------
# precursor mapping
# ---------------------------------------------------------------------------
def _fmt_event_mz(rows: pd.DataFrame, a: str, b: str, limit: int = 8) -> str:
    parts = [f"event {int(r.event)}: {getattr(r, a):.4f} vs {getattr(r, b):.4f}"
             for r in rows.head(limit).itertuples()]
    return "; ".join(parts) + (f" (+{len(rows) - limit} more)" if len(rows) > limit else "")


def map_precursors(scans: pd.DataFrame, lists: list | None = None, *, rule_check: bool = True
                   ) -> tuple[pd.DataFrame, dict]:
    """Each MS2 scan's precursor, NCE, experiment, list row and spectrum key.

    `lists`: None, or one or more inclusion lists (`read_inclusion_list`
    frames; one shared by every experiment, or one per experiment in event
    order). The precursor of an event comes from, in this order: its
    filters (where they agree with the trailer), the list (event -> row),
    the scan-range rule (no list; only in the experiments it holds in,
    `rule_verdicts` / `rule_holds`, and never for an event whose HighMass
    another event of its experiment shares). Wherever two sources speak for
    an event they must agree (`SOURCE_MATCH_MDA`), and a list is ALWAYS
    checked against the rule where the rule holds (`rule_fit`;
    `rule_check=False` skips that for a method known not to set its range by
    the precursor); where it cannot be judged, a misfit is noted. Any
    disagreement is a `ScanMapError` naming the events.
    A scan whose own filter is readable keeps the precursor it names (a
    data-dependent event names a new one each cycle -- mapped on a fixed
    range; where its scan range follows each precursor the event reads
    several ranges, and `_event_table` refuses it). Headerless scans are
    placed by `infer_headerless`, and precursors within `PARENT_TOL_DA` are
    one parent (`cluster_parents`). The spectrum key is the server's:
    `sources.spectrum_key(parent, 'hcd<NCE:.2f>')`. -> (scans, report)."""
    layout = method_layout(scans, lists, rule_check=rule_check)
    events = _event_table(scans, layout)
    filters = filter_verdict(scans)
    verdicts = rule_verdicts(events)
    file_distinct = rule_applies(events)
    hold = rule_holds(verdicts, file_distinct)
    shared = _shared_high_mass(events)
    applies = bool(hold)
    report = {"layout": layout.source, "layout_closed": bool(layout.closed),
              "experiments": [{"experiment": b.experiment, "first_event": b.first,
                               "last_event": b.last, "rows": b.size,
                               "nce": None if math.isnan(b.nce) else b.nce}
                              for b in layout.blocks],
              "full_scan_event": None if layout.full_scan is None else layout.full_scan.first,
              "filters": filters,
              "rule": {"applies": applies, "checked": False,
                       "experiments": {int(e): v for e, v in verdicts.items()}},
              "precursor_sources": {}, "notes": list(layout.notes)}
    # scans whose trailer names no MS order (e.g. a SIM experiment): kept as read
    other = scans[scans["has_header"] & scans["order"].isna()]
    report["no_order_scans"] = len(other)
    if len(other):
        nodes = sorted(set(other["node"]) - {""})
        report["notes"].append(
            f"{len(other)} scan(s) whose trailer gives no MS order"
            + (f" ({', '.join(nodes)})" if nodes else "") + ": no Full Scan and no MS2 scan -- "
            "in no spectrum, not in the MS1 average, and the scan-cycle walk never crosses them")
    seen = events[events["n_header"] > 0]
    # every readable filter against the list
    fr = scans[scans["has_header"] & np.isfinite(scans["filter_parent_mz"])
               & scans["event"].notna()][["scan", "event", "filter_parent_mz"]]
    fr = fr.assign(event=fr["event"].astype(int)).merge(events[["event", "list_mz"]], on="event")
    off = fr[np.isfinite(fr["list_mz"])
             & (np.abs(fr["filter_parent_mz"] - fr["list_mz"]) * 1e3 > SOURCE_MATCH_MDA)]
    if len(off):
        off = off.drop_duplicates("event")
        raise ScanMapError("the inclusion list disagrees with the scan filters ("
                           + _fmt_event_mz(off, "filter_parent_mz", "list_mz")
                           + "): wrong list, or rows in the wrong order")
    # the list (else the filter) against the scan-range rule, in the
    # experiments it holds in
    col = "list_mz" if lists else "filter_mz"
    if applies:
        fit = rule_fit(seen[seen["experiment"].isin(hold)], col)
        report["rule"].update(fit, checked=bool(fit["n_events"]) and rule_check,
                              against=PRECURSOR_LIST if lists else PRECURSOR_FILTER)
        if rule_check and fit["events_off"]:
            bad = seen[seen["event"].isin(fit["events_off"])]
            worst = ", ".join(
                f"event {int(r.event)} (row {int(r.slot)}: {getattr(r, col):.4f}; HighMass "
                f"{r.high_mass:.4f} implies {rule_precursor(r.high_mass, r.iso_width):.4f})"
                for r in bad.head(8).itertuples())
            raise ScanMapError(
                f"the scan ranges of {len(bad)} of {fit['n_events']} events do not fit the "
                f"{'inclusion list' if lists else 'scan filters'} (HighMass = {RULE_SLOPE:g} x "
                f"(precursor + width/2) + {RULE_OFFSET_DA:g}, max residual "
                f"{fit['max_residual_mda']:.3g} mDa > {RULE_TOL_MDA:g}): {worst}"
                + (" -- wrong list, rows in the wrong order, or the wrong number of rows"
                   if lists else ""))
    fixed = sorted(e for e, v in verdicts.items() if v == RULE_FIXED)
    unsure = sorted(e for e, v in verdicts.items() if v == RULE_UNKNOWN and e not in hold
                    and (seen["experiment"] == e).any())
    if lists and (fixed or unsure):
        parts = []
        if fixed:
            parts.append(f"does not follow the precursor in experiment(s) {_events_text(fixed)} "
                         "(each reads one HighMass for all its events: a fixed range)")
        if unsure:
            parts.append(f"cannot be judged in experiment(s) {_events_text(unsure)} (one event "
                         "seen, and " + ("the file reads one HighMass" if not file_distinct else
                                         "another experiment reads a fixed range") + ")")
        note = ("the scan range " + "; it ".join(parts) + ": the list rows there are checked "
                "against " + ("the Full Scan event, " if layout.full_scan is not None else "")
                + "the experiments' NCEs and any readable filters, not against the scan range")
        u = rule_fit(seen[seen["experiment"].isin(unsure)], col) if unsure else None
        if u and u["events_off"]:
            note += (f"; event(s) {_events_text(u['events_off'])} do not fit the scan-range rule "
                     f"(max residual {u['max_residual_mda']:.3g} mDa): a fixed range there, or "
                     "the wrong rows -- check them")
            report["rule"]["unjudged_events_off"] = u["events_off"]
        report["notes"].append(note)
    # each event's precursor
    pmz, psrc = {}, {}
    no_source: dict[int, str] = {}
    for r in events.itertuples():
        e = int(r.event)
        if r.filter_varies:                   # every scan names its own
            pmz[e], psrc[e] = math.nan, PRECURSOR_FILTER
        elif math.isfinite(r.filter_mz):
            pmz[e], psrc[e] = float(r.filter_mz), PRECURSOR_FILTER
        elif math.isfinite(r.list_mz):
            pmz[e], psrc[e] = float(r.list_mz), PRECURSOR_LIST
        elif (int(r.experiment) in hold and e not in shared and math.isfinite(r.high_mass)
              and math.isfinite(r.iso_width)):
            pmz[e] = round(rule_precursor(r.high_mass, r.iso_width), RULE_DECIMALS)
            psrc[e] = PRECURSOR_RULE
        elif r.n_header:
            v = verdicts.get(int(r.experiment))
            no_source[e] = (
                "a fixed range" if v == RULE_FIXED else
                "another event of its experiment reads the same HighMass" if e in shared else
                ("one event seen in its experiment, and "
                 + ("the file reads one HighMass" if not file_distinct else
                    "another experiment reads a fixed range")) if v == RULE_UNKNOWN else
                "no scan range or isolation width to derive it from")
    if no_source:
        why = ("the scan filters are unreadable" if filters["filters_readable"] != FILTERS_YES
               else "their filters name no precursor")
        by: dict[str, list] = {}
        for e, t in no_source.items():
            by.setdefault(t, []).append(e)
        rule_why = (next(iter(by)) if len(by) == 1 else
                    "; ".join(f"event(s) {_events_text(ev)}: {t}" for t, ev in by.items()))
        raise ScanMapError(
            f"no precursor for event(s) {_events_text(no_source)}: {why}, no inclusion list was "
            f"given, and the scan range does not name them ({rule_why}): pass the method's "
            "inclusion list", needs_list=True)
    # place the headerless scans, then stamp every MS2 scan
    scans, inf = infer_headerless(scans, layout, events, rule_check=rule_check)
    report["inference"] = inf
    report["notes"].extend(inf["notes"])
    ev_rows = events.set_index("event")
    counts: dict[str, int] = {}
    unmapped = []
    for i in scans.index[scans["order"] == 2]:
        ev = scans.at[i, "event"]
        e = int(ev) if math.isfinite(ev) else None
        b = layout.block_of(e) if e is not None else None
        own = scans.at[i, "filter_parent_mz"]
        if e is None or b is None or b is layout.full_scan or e not in pmz or not math.isfinite(
                b.nce) or not (math.isfinite(pmz[e]) or math.isfinite(own)):
            unmapped.append(int(scans.at[i, "scan"]))
            continue
        r = ev_rows.loc[e]
        scans.at[i, "parent_mz"] = own if math.isfinite(own) else pmz[e]
        scans.at[i, "precursor_source"] = PRECURSOR_FILTER if math.isfinite(own) else psrc[e]
        scans.at[i, "experiment"] = float(r["experiment"])
        scans.at[i, "slot"] = float(r["slot"])
        scans.at[i, "nce"] = b.nce
        if not math.isfinite(scans.at[i, "iso_width"]):
            scans.at[i, "iso_width"] = float(r["iso_width"])
        src = scans.at[i, "precursor_source"]
        counts[src] = counts.get(src, 0) + 1
    report["precursor_sources"] = counts
    report["rule"]["derived_events"] = sum(1 for v in psrc.values() if v == PRECURSOR_RULE)
    # parents within PARENT_TOL_DA are one parent, as the server keys them
    ms2 = scans.index[(scans["order"] == 2) & np.isfinite(scans["parent_mz"])]
    canon = cluster_parents(scans.loc[ms2, "parent_mz"])
    merged = {}
    for i in ms2:
        raw = float(scans.at[i, "parent_mz"])
        p = canon[raw]
        if round(raw, RULE_DECIMALS) != p:
            merged.setdefault(p, set()).add(raw)
        scans.at[i, "parent_mz"] = p
        scans.at[i, "spectrum"] = SRC.spectrum_key(p, f"hcd{scans.at[i, 'nce']:.2f}")
    if merged:
        report["notes"].append(
            f"precursors within {PARENT_TOL_DA * 1e3:g} mDa are one parent, keyed by their "
            f"median as the server keys it: {len(merged)} parent(s) gather "
            f"{sum(len(v) for v in merged.values())} distinct precursor value(s)")
    if unmapped:
        report["notes"].append(f"{len(unmapped)} MS2 scan(s) with no precursor (scan(s) "
                               f"{_events_text(unmapped)}): their event is in no experiment or "
                               "has no precursor source; they are in no spectrum")
    report["unmapped_ms2_scans"] = unmapped
    report["events"] = len(events)
    report["events_seen"] = int((events["n_header"] > 0).sum())
    _coverage(scans, layout, events, pmz, canon, report)
    return scans, report


def _coverage(scans: pd.DataFrame, layout: Layout, events: pd.DataFrame, pmz: dict,
              canon: dict, report: dict) -> None:
    """Which of the layout's events the file scanned (a trailer row, or a scan
    the cycle placed) -- into `report`: each experiment's `rows_scanned`,
    `events_never_scanned`, the parents with no spectrum at an experiment's
    NCE (`missing_energies`: repr(parent key) -> NCEs) and, noted, an
    acquisition that ends inside an experiment (its tail never scanned) or
    other events never scanned; and the events seen only through placed
    scans, which take their experiment's isolation width
    (`events_width_from_experiment`) -- noted only when the trailers read
    more than one width (else the width they take is the file's own)."""
    ms2 = scans[(scans["order"] == 2) & scans["event"].notna()]
    scanned = {int(e) for e in ms2["event"]}
    never = [int(e) for e in events["event"] if int(e) not in scanned]
    for x, b in zip(report["experiments"], layout.blocks):
        x["rows_scanned"] = sum(1 for e in range(b.first, b.last + 1) if e in scanned)
    report["events_never_scanned"] = never
    # a parent whose event at an NCE was never scanned, and no other event
    # gave it a spectrum at that NCE (two parallel experiments of one NCE)
    have = {(round(float(p), 6), round(float(n), 4))
            for p, n in zip(ms2["parent_mz"], ms2["nce"]) if math.isfinite(p)}
    keys = sorted(set(canon.values()))
    missing: dict[float, set] = {}
    for e in never:
        b = layout.block_of(e)
        v = pmz.get(e, math.nan)
        if b is None or not math.isfinite(b.nce) or not math.isfinite(v):
            continue
        near = [k for k in keys if abs(k - v) <= PARENT_TOL_DA]
        key = near[0] if near else round(v, RULE_DECIMALS)
        if (round(key, 6), round(b.nce, 4)) not in have:
            missing.setdefault(key, set()).add(float(b.nce))
    report["missing_energies"] = {repr(float(k)): sorted(v) for k, v in sorted(missing.items())}
    tail: set = set()
    if never and len(ms2):
        last = ms2.loc[ms2["scan"].idxmax()]
        b = layout.block_of(last["event"])
        e = int(last["event"])
        if b is not None and b is not layout.full_scan and e < b.last and all(
                x in never for x in range(e + 1, b.last + 1)):
            tail = set(range(e + 1, b.last + 1))
            parents = {k for k, v in missing.items() if b.nce in v}
            report["notes"].append(
                f"experiment {b.experiment} (NCE {b.nce:g}, events {b.first}-{b.last}): rows "
                f"{e + 2 - b.first}-{b.size} never scanned -- the acquisition ends inside it "
                f"(last MS2 scan {int(last['scan'])} = event {e}, row {e + 1 - b.first}); "
                f"{len(parents)} parent(s) have no NCE-{b.nce:g} spectrum")
    rest = [e for e in never if e not in tail]
    if rest:
        report["notes"].append(f"event(s) {_events_text(rest)} never scanned (no scan of them was "
                               "read or placed by the scan cycle): their parents have no spectrum "
                               "at their experiment's NCE")
    # events seen only through scans the cycle placed take their experiment's
    # (else the file's) isolation width
    inferred = {int(e) for e in scans.loc[scans["inferred"] & scans["event"].notna(), "event"]}
    w = events[events["width_from"].isin(["experiment", "file"]) & (events["n_header"] == 0)
               & events["event"].isin(inferred)]
    report["events_width_from_experiment"] = [int(x) for x in w["event"]]
    widths = scans.loc[scans["has_header"] & (scans["order"] == 2), "iso_width"]
    if len(w) and len({round(float(x), 4) for x in widths if math.isfinite(x)}) > 1:
        report["notes"].append(
            f"{len(w)} event(s) seen only through scans placed by the scan cycle "
            f"({_events_text(w['event'])}) take their experiment's isolation width (the "
            "trailers read more than one)")


#: Precursor m/z values within this (Da, chained in m/z order) are one
#: parent, keyed by their median rounded to 4 decimals: the server's own
#: grouping of MS2 scans by parent (its `parent_peak_tolerance`, 0.001 Da),
#: so a precursor the filters measure scan by scan keys one spectrum.
PARENT_TOL_DA = 0.001


def cluster_parents(values) -> dict[float, float]:
    """Each distinct precursor value -> its parent key: the sorted distinct
    values chained while the gap is <= `PARENT_TOL_DA`, each chain keyed by
    round(median, 4)."""
    vals = sorted({float(v) for v in values if math.isfinite(float(v))})
    out: dict[float, float] = {}
    chain: list[float] = []

    def close():
        key = round(float(np.median(chain)), 4)
        out.update({v: key for v in chain})

    for v in vals:
        if chain and v - chain[-1] > PARENT_TOL_DA:
            close()
            chain = []
        chain.append(v)
    if chain:
        close()
    return out


# ---------------------------------------------------------------------------
# mapping by the server's own groups (a file it reads)
# ---------------------------------------------------------------------------
def _seconds(t) -> float:
    """A time-series time -> seconds: a number as it is, an ISO string from the
    epoch ('1970-01-01T00:00:55.028905008') through its nanoseconds."""
    if isinstance(t, (int, float)) and not isinstance(t, bool):
        return float(t)
    ts = pd.Timestamp(str(t))
    if ts.tzinfo is not None:
        ts = ts.tz_convert(None)
    return (ts - pd.Timestamp(0)).value / 1e9


def assign_from_times(scans: pd.DataFrame, times: dict) -> tuple[pd.DataFrame, dict]:
    """Map each MS2 scan to the server's spectrum by its time-series times:
    `times` = {spectrum key: [times]} (`get_timeseries(parent, activation=...)`
    `time` lists). A time matches the scan whose `t_s` lies within
    `TIME_MATCH_S`; a time with no such scan, or a scan two keys claim, is
    counted and noted. -> (scans, report): per key its matched / asked counts,
    and the headerless scans in no group (e.g. a hidden tail)."""
    scans = scans.copy()
    t = scans["t_s"].to_numpy(dtype=float)
    order = np.argsort(t, kind="stable")
    ts = t[order]
    rep = {"spectra": {}, "unmatched_times": 0, "claimed_twice": [], "on_ms1_rows": [],
           "notes": []}
    owner: dict[int, str] = {}
    for key, values in (times or {}).items():
        parent, act = SRC._parent_of(str(key), {})
        _, nce = SRC.parse_activation(act)
        vals = list((values or {}).get("time", []) if isinstance(values, dict) else values or [])
        got = 0
        for v in vals:
            x = _seconds(v)
            k = int(np.searchsorted(ts, x))
            cand = [c for c in (k - 1, k) if 0 <= c < len(ts)]
            best = min(cand, key=lambda c: abs(ts[c] - x)) if cand else None
            if best is None or abs(ts[best] - x) > TIME_MATCH_S:
                rep["unmatched_times"] += 1
                continue
            i = scans.index[order[best]]
            scan = int(scans.at[i, "scan"])
            if scan in owner and owner[scan] != key:
                rep["claimed_twice"].append(scan)
                continue
            if scans.at[i, "order"] == 1:
                rep["on_ms1_rows"].append(scan)
            owner[scan] = key
            scans.at[i, "order"] = 2.0
            scans.at[i, "parent_mz"] = parent
            scans.at[i, "nce"] = nce
            scans.at[i, "spectrum"] = str(key)
            scans.at[i, "precursor_source"] = PRECURSOR_SERVER
            got += 1
        rep["spectra"][str(key)] = {"times": len(vals), "matched": got}
    for k, what in (("claimed_twice", "scan(s) two spectra's times both name"),
                    ("on_ms1_rows", "matched scan(s) whose trailer reads MS1")):
        if rep[k]:
            rep["notes"].append(f"{len(rep[k])} {what} ({_events_text(rep[k])})")
    if rep["unmatched_times"]:
        rep["notes"].append(f"{rep['unmatched_times']} time-series time(s) match no scan within "
                            f"{TIME_MATCH_S * 1e3:g} ms")
    free = scans[~scans["has_header"] & (scans["spectrum"] == "")]
    rep["headerless_in_no_group"] = len(free)
    rep["headerless_in_no_group_scans"] = _runs_text(free["scan"].tolist())
    if len(free):
        rep["notes"].append(f"{len(free)} scan(s) with no trailer row are in no server group "
                            f"(scan(s) {rep['headerless_in_no_group_scans']}): in no spectrum")
    return scans, rep


def _runs_text(values) -> str:
    """[1, 154, 155, 156] -> '1, 154-156'."""
    v = sorted(int(x) for x in values)
    out, k = [], 0
    while k < len(v):
        j = k
        while j + 1 < len(v) and v[j + 1] == v[j] + 1:
            j += 1
        out.append(str(v[k]) if j == k else f"{v[k]}-{v[j]}")
        k = j + 1
    return ", ".join(out)


# ---------------------------------------------------------------------------
# averaging
# ---------------------------------------------------------------------------
def _pool(data: list, scan_numbers) -> tuple[np.ndarray, ...]:
    """The centroids of `scan_numbers` (1-based) pooled and sorted by m/z:
    (mz, intensity, noise, resolution, scan)."""
    parts = []
    for s in scan_numbers:
        c = data[int(s) - 1] or {}
        mz = np.asarray(c.get("mzs") or [], dtype=np.float64)
        n = len(mz)

        def arr(name, _n=n, _c=c):
            v = _c.get(name)
            a = np.full(_n, np.nan) if v is None or len(v) != _n else np.asarray(v, dtype=np.float64)
            return a
        parts.append((mz, arr("intensities"), arr("noises"), arr("resolutions"),
                      np.full(n, int(s), dtype=np.int64)))
    if not parts:
        e = np.zeros(0)
        return e, e, e, e, np.zeros(0, dtype=np.int64)
    mz, inten, noise, res, scan = (np.concatenate(x) for x in zip(*parts))
    keep = np.isfinite(mz) & np.isfinite(inten) & (inten > 0)
    mz, inten, noise, res, scan = mz[keep], inten[keep], noise[keep], res[keep], scan[keep]
    o = np.argsort(mz, kind="stable")
    return mz[o], inten[o], noise[o], res[o], scan[o]


def _wmean(values: np.ndarray, weights: np.ndarray, labels: np.ndarray, n: int) -> np.ndarray:
    """Per label, the weighted mean of the finite values (NaN where none)."""
    ok = np.isfinite(values)
    w = np.where(ok, weights, 0.0)
    num = np.bincount(labels, weights=np.where(ok, values, 0.0) * w, minlength=n)
    den = np.bincount(labels, weights=w, minlength=n)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def cluster_centroids(mz: np.ndarray, intensity: np.ndarray, resolution: np.ndarray, *,
                      seed_ppm: float = SEED_PPM, merge_fwhm: float = MERGE_FWHM,
                      nominal_resolution=None) -> np.ndarray:
    """Cluster labels (0..k-1, in m/z order) for centroids already sorted by
    m/z: seed bins at gaps > `seed_ppm`, then adjacent seed bins joined where
    their gap is < `merge_fwhm` x the narrower FWHM (m/z / the bin's
    intensity-weighted resolution). A centroid with no resolution takes
    `nominal_resolution(m/z)` when given; seed bins with none stay apart."""
    n = len(mz)
    if not n:
        return np.zeros(0, dtype=np.int64)
    res = np.asarray(resolution, dtype=np.float64)
    if nominal_resolution is not None:
        miss = ~(np.isfinite(res) & (res > 0))
        if miss.any():
            res = res.copy()
            res[miss] = nominal_resolution(mz[miss])
    gap = np.diff(mz) / mz[:-1] * 1e6
    seed = np.concatenate(([0], np.cumsum(gap > seed_ppm))).astype(np.int64)
    k = int(seed[-1]) + 1
    bmz = _wmean(mz, intensity, seed, k)
    bres = _wmean(np.where(res > 0, res, np.nan), intensity, seed, k)
    if k == 1:
        return seed
    with np.errstate(invalid="ignore", divide="ignore"):
        fwhm = 1e6 / bres
        bgap = np.diff(bmz) / bmz[:-1] * 1e6
        # np.minimum, not fmin: a bin with no resolution has no width, and
        # is joined to no neighbour through the neighbour's
        thr = merge_fwhm * np.minimum(fwhm[:-1], fwhm[1:])
    join = np.isfinite(thr) & (bgap < thr)
    blab = np.concatenate(([0], np.cumsum(~join))).astype(np.int64)
    return blab[seed]


def nominal_resolution(r200: float):
    """R(m/z) = R200 x sqrt(200 / m/z): an Orbitrap's resolving power from the
    trailer's `FT Resolution:` (given at m/z 200). Measured against the
    resolution the per-scan centroids report, on the 47 spectra of
    `SEED_PPM`: strong peaks (S/N > 100) read 1.033 x nominal on the median
    (5-95 %: 0.99-1.11), weak ones (S/N 3-10) 0.73 x, and side lobes up to
    15 x -- so the centroids' own values are used, and this only where a
    payload carries none."""
    r200 = float(r200)
    return lambda m: r200 * np.sqrt(200.0 / np.asarray(m, dtype=np.float64))


@dataclass
class Averaged:
    """One group's averaged spectrum (every cluster; `keep` = S/N >= the
    threshold) and its per-scan matrix (None when not asked for)."""
    mz: np.ndarray
    intensity: np.ndarray
    resolution: np.ndarray
    signal_to_noise: np.ndarray
    n_scans_present: np.ndarray
    keep: np.ndarray
    scans: np.ndarray                         # the group's scan numbers, in order
    per_scan: np.ndarray | None               # clusters x scans: summed intensity (0 = absent)
    per_scan_n: np.ndarray | None             # clusters x scans: centroids summed


def average_scans(meta: dict, scan_numbers, *, noise_threshold: float = SRC.DEFAULT_NOISE_THRESHOLD,
                  nominal_r200: float | None = None, per_scan: bool = True) -> Averaged:
    """Average the centroids of `scan_numbers` (1-based) into one spectrum.

    Clusters by `cluster_centroids`. Per cluster, with N the group's scans:
      * a scan's intensity is the SUM of its centroids in the cluster (two of
        one scan in one cluster are one reading of that scan), its noise their
        intensity-weighted noise, its S/N the two's ratio;
      * mz: intensity-weighted over the centroids; resolution: likewise
        (both identical to the server's on 891 of the 892 peaks matched,
        `SEED_PPM`);
      * intensity: the mean over the N scans, a scan without the peak
        counting 0;
      * S/N: the intensity-weighted mean per-scan S/N x n / sqrt(N), n the
        scans holding the peak -- the averaged spectrum's noise is sqrt(N)
        lower, its height n/N of a scan's. Against the server's S/N on the
        same peaks: median 1.005 (5-95 %: 1.001-1.010). The plainer mean
        intensity / (mean noise / sqrt(N)) reads 0.98-0.99 on the median
        (5 %: 0.91-0.94) and misses 6 of standard B's 266 peaks at the
        threshold (this rule: 1).
    `keep` marks S/N >= `noise_threshold` (NaN S/N -- a payload without
    noises -- keeps the peak). `nominal_r200` fills a missing resolution
    (`nominal_resolution`).

    Every sum runs over the (cluster, scan) pairs that hold a centroid, so
    memory grows with the centroids, never with clusters x scans: an MS1
    average keeps every noise cluster (~55 new ones per scan on our own
    files), and as a dense clusters x scans matrix it grew ~N^1.7 with the
    scans (over a GB for a few hundred MS1 scans). The matrix (`per_scan`,
    `per_scan_n`) is built only when asked for (a group's per-scan
    readings)."""
    meta = metadata_payload(meta)
    data = meta["centroids_meta"]["data"]
    scans = np.asarray([int(s) for s in scan_numbers], dtype=np.int64)
    N = len(scans)
    mz, inten, noise, res, scan = _pool(data, scans)
    nom = nominal_resolution(nominal_r200) if nominal_r200 else None
    lab = cluster_centroids(mz, inten, res, nominal_resolution=nom)
    k = int(lab[-1]) + 1 if len(lab) else 0
    col = {int(s): j for j, s in enumerate(scans)}
    sj = np.asarray([col[int(s)] for s in scan], dtype=np.int64)
    # one reading per (cluster, scan) pair: its centroids summed
    stride = max(N, 1)
    pair, inv = np.unique(lab * stride + sj, return_inverse=True)
    inv = inv.ravel()
    p_lab, p_scan = pair // stride, pair % stride
    n_pairs = len(pair)
    fin = np.isfinite(noise)
    p_int = np.bincount(inv, weights=inten, minlength=n_pairs)
    p_n = np.bincount(inv, minlength=n_pairs)
    nz = np.bincount(inv, weights=np.where(fin, noise, 0.0) * inten, minlength=n_pairs)
    nzw = np.bincount(inv, weights=np.where(fin, inten, 0.0), minlength=n_pairs)
    with np.errstate(invalid="ignore", divide="ignore"):
        p_noise = np.where(nzw > 0, nz / nzw, np.nan)
        p_sn = np.where((p_int > 0) & (p_noise > 0), p_int / p_noise, np.nan)
        ok = np.isfinite(p_sn)
        w = np.where(ok, p_int, 0.0)
        num = np.bincount(p_lab, weights=np.where(ok, p_sn, 0.0) * w, minlength=k)
        den = np.bincount(p_lab, weights=w, minlength=k)
        sn_w = np.where(den > 0, num / den, np.nan)
    present = np.bincount(p_lab, minlength=k)
    sn = sn_w * present / math.sqrt(N) if N else np.full(k, np.nan)
    out_mz = _wmean(mz, inten, lab, k)
    out_res = _wmean(np.where(res > 0, res, np.nan), inten, lab, k)
    if nom is not None:
        miss = ~np.isfinite(out_res)
        out_res[miss] = nom(out_mz[miss])
    keep = ~(sn < float(noise_threshold))     # NaN S/N keeps
    per = per_n = None
    if per_scan:
        per = np.zeros((k, N))
        per_n = np.zeros((k, N), dtype=np.int64)
        per[p_lab, p_scan] = p_int
        per_n[p_lab, p_scan] = p_n
    total = np.bincount(p_lab, weights=p_int, minlength=k)
    return Averaged(mz=out_mz, intensity=total / N if N else np.zeros(k),
                    resolution=out_res, signal_to_noise=sn, n_scans_present=present,
                    keep=keep, scans=scans, per_scan=per, per_scan_n=per_n)


def _file_r200(scans: pd.DataFrame) -> float | None:
    """The one `FT Resolution:` the trailer rows of `scans` read, None when
    they read none or several. Asked of the scans being averaged -- a group's,
    or the MS1 scans -- never of the whole file: a Full Scan and its tMS2
    scans are often set to different resolving powers (e.g. 120000 / 60000)."""
    r = scans["ft_resolution"][np.isfinite(scans["ft_resolution"]) & (scans["ft_resolution"] > 0)]
    vals = sorted({float(x) for x in r})
    return vals[0] if len(vals) == 1 else None


def _resolution_coverage(meta: dict, scan_numbers=None) -> tuple[int, int]:
    """(scans whose centroids carry a resolution each, scans with centroids)
    among `scan_numbers` (1-based; every scan of the payload when None) --
    counted per average: a group's scans, or the MS1 scans."""
    data = metadata_payload(meta)["centroids_meta"]["data"]
    rows = (data if scan_numbers is None else
            [data[int(s) - 1] for s in scan_numbers if 0 < int(s) <= len(data)])
    with_res = of = 0
    for c in rows:
        n = len((c or {}).get("mzs") or [])
        if n:
            of += 1
            r = c.get("resolutions")
            with_res += r is not None and len(r) == n
    return with_res, of


def _fill_text(nominal) -> str:
    return ("FT Resolution " + "/".join(f"{v:g}" for v in sorted(nominal)) + " at m/z 200"
            if nominal else "")


def _resolution_source(with_res: int, of: int, nominal) -> str:
    """How an average's resolutions were read: 'per-scan centroids' (every
    scan carries them), 'nominal: FT Resolution ...' (none does, filled),
    'per-scan centroids; nominal (...) where a scan carries none' (some),
    'none' (none does, nothing filled) -- `nominal` the R200 values used."""
    fill = _fill_text(nominal)
    if with_res == of:
        return "per-scan centroids"
    if with_res:
        return "per-scan centroids; " + (f"nominal ({fill})" if fill else "none") + \
            " where a scan carries none"
    return f"nominal: {fill}" if fill else "none"


def _range_cut(g: pd.DataFrame) -> tuple[float, float]:
    """The m/z range every scan of `g` recorded: (highest LowMass, lowest
    HighMass), NaN where the scans do not say."""
    lo, hi = g["low_mass"].to_numpy(dtype=float), g["high_mass"].to_numpy(dtype=float)
    return (float(np.nanmax(lo)) if np.isfinite(lo).any() else math.nan,
            float(np.nanmin(hi)) if np.isfinite(hi).any() else math.nan)


def build_spectra(sample_id: str, meta: dict, scans: pd.DataFrame, *,
                  noise_threshold: float = SRC.DEFAULT_NOISE_THRESHOLD,
                  ms1=None
                  ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Average every spectrum key of `scans` (its MS2 scans with a key) and
    build the tables `ms2 analyze` reads: (spectra in `sources.SPECTRA_COLUMNS`,
    parents in `sources.PARENTS_COLUMNS`, scan peaks in `SCAN_PEAK_COLUMNS`,
    report). The spectra go through `sources.spectra_table` /
    `parents_table` as a server answer would (one by_activation entry per
    key, one summary group with its scan count, times and mean volts), so
    the precursor is located the same way -- centred on the file's m/z axis
    offset (`sources.axis_offset`, from these spectra, else from `ms1`, the
    MS1 average -- or a callable giving it, called only when the spectra
    read too few offsets). `isolation_width` is the trailer's, per parent
    (`isolation_widths`).

    The scan range is per scan, and an experiment's first mass may differ
    from the others' (an automatic first mass): a product between the two
    would read as absent at that energy only, and move the parent's
    breakdown. So when a parent's spectra differ in the range their scans
    recorded (by more than `RANGE_CHECK_DA`), every spectrum of that parent
    is cut to the range all of them recorded -- noted on its rows, with the
    peaks removed (report `scan_ranges`: every spectrum's [LowMass, HighMass]
    as recorded)."""
    meta = metadata_payload(meta)
    ms2 = scans[(scans["order"] == 2) & (scans["spectrum"] != "")]
    # counted over the scans these spectra average (never the MS1 scans: the
    # MS1 average counts its own, `ms1_average`)
    with_res, of = _resolution_coverage(meta, ms2["scan"])
    # a scan whose centroids carry no resolution: its group's own resolving
    # power (its trailer rows'), else the MS2 scans' when they read one,
    # fills it -- a cluster of such scans alone would otherwise read NaN, its
    # side-lobe rules off. A group none of whose scans lacks them needs none
    ms2_r200 = _file_r200(scans[scans["order"] == 2])
    nominal: set = set()
    centroids, groups = {}, []
    peak_cols: dict[str, list] = {c: [] for c in SCAN_PEAK_COLUMNS}
    rep = {"spectra": 0, "scans": len(ms2), "noise_threshold": float(noise_threshold),
           "clustering": {"seed_ppm": SEED_PPM, "merge_fwhm": MERGE_FWHM},
           "resolution_source": "per-scan centroids",
           "server_height_ratio": SERVER_HEIGHT_RATIO, "scan_ranges": {}, "range_cuts": {}}
    tmap = dict(zip(scans["scan"].astype(int), scans["t_s"].astype(float)))
    # the range every scan of each spectrum recorded, and of each parent
    ranges = {str(k): _range_cut(g) for k, g in ms2.groupby("spectrum", sort=False)}
    rep["scan_ranges"] = {k: [None if math.isnan(lo) else lo, None if math.isnan(hi) else hi]
                          for k, (lo, hi) in ranges.items()}
    cuts: dict[float, tuple[float, float, str]] = {}
    for p, g in ms2.groupby(ms2["parent_mz"].round(6), sort=False):
        rs = [(ranges[str(k)], float(gg["nce"].iloc[0]))
              for k, gg in g.groupby("spectrum", sort=False)]
        los = [r[0][0] for r in rs if math.isfinite(r[0][0])]
        his = [r[0][1] for r in rs if math.isfinite(r[0][1])]
        if (los and max(los) - min(los) > RANGE_CHECK_DA) or (
                his and max(his) - min(his) > RANGE_CHECK_DA):
            by_lo: dict[float, list] = {}
            for (lo, _), nce in rs:
                by_lo.setdefault(round(lo, 2), []).append(nce)
            text = ", ".join(f"{lo:.2f} at NCE " + "/".join(f"{x:g}" for x in sorted(n))
                             for lo, n in sorted(by_lo.items()))
            cuts[float(p)] = (max(los) if los else -math.inf, min(his) if his else math.inf,
                              text)
    for key, g in ms2.groupby("spectrum", sort=False):
        g = g.sort_values("scan")
        parent = float(g["parent_mz"].iloc[0])
        nce = float(g["nce"].iloc[0])
        act = f"hcd{nce:.2f}"
        g_res, g_of = _resolution_coverage(meta, g["scan"])
        r200 = None if g_res == g_of else (_file_r200(g) or ms2_r200)
        if r200:
            nominal.add(r200)
        a = average_scans(meta, g["scan"], noise_threshold=noise_threshold, nominal_r200=r200)
        kp = a.keep.copy()
        cut = cuts.get(round(parent, 6))
        if cut is not None:
            inside = (a.mz >= cut[0]) & (a.mz <= cut[1])
            rep["range_cuts"].setdefault(repr(round(parent, 6)), 0)
            rep["range_cuts"][repr(round(parent, 6))] += int((kp & ~inside).sum())
            kp &= inside
        centroids[key] = {"parent_peak_mz": parent, "activation": act,
                          "mz": a.mz[kp].tolist(), "intensity": a.intensity[kp].tolist(),
                          "resolution": a.resolution[kp].tolist(),
                          "signal_to_noise": a.signal_to_noise[kp].tolist()}
        volts = [float(v) for v in g["hcd_v"] if math.isfinite(v)]
        groups.append({"parent_peak_mz": parent, "activation": act,
                       "hcd_energy": [float(np.mean(volts))] if volts else [],
                       "scan_count": len(g), "t_min": float(g["t_s"].min()),
                       "t_max": float(g["t_s"].max())})
        for ci in np.flatnonzero(kp):
            for j, s in enumerate(a.scans):
                peak_cols["spectrum"].append(str(key))
                peak_cols["peak_mz"].append(float(a.mz[ci]))
                peak_cols["scan"].append(int(s))
                peak_cols["t_s"].append(tmap.get(int(s), math.nan))
                peak_cols["intensity"].append(float(a.per_scan[ci, j]))
                peak_cols["n_centroids"].append(int(a.per_scan_n[ci, j]))
    widths = isolation_widths(scans)
    vals = sorted(set(widths.values()))
    summary = {"groups": groups, "parent_peaks": sorted({g_["parent_peak_mz"] for g_ in groups}),
               "isolation_width": vals[0] if len(vals) == 1 else math.nan,
               "ms2_scan_count": len(ms2),
               "ms1_scan_count": int((scans["order"] == 1).sum())}
    # the key-offset note names where each parent's key came from
    src = ms2.groupby(ms2["parent_mz"].round(6))["precursor_source"].agg(
        lambda v: " / ".join(sorted(set(v) - {""})))
    origin = {float(p): (SRC.KEY_ORIGIN_SERVER if v == PRECURSOR_SERVER else
                         f"the parent key (the {v or 'unknown source'}'s precursor)")
              for p, v in src.items()}
    axis = SRC.axis_offset(centroids, summary, widths=widths, ms1=ms1)
    spectra = SRC.spectra_table(sample_id, centroids, summary)
    # the side-lobe guard only where the axis offset is unknown: centred on a
    # measured offset, the precursor lies in its search window
    parents = SRC.parents_table(sample_id, centroids, summary, key_origin=origin,
                                axis_offset=axis, lobe_guard=axis["ppm"] is None)
    if len(vals) > 1:
        parents["isolation_width"] = [widths.get(round(float(p), 6), math.nan)
                                      for p in parents["parent_mz"]]
    for p, (lo, hi, text) in cuts.items():
        sel = np.isclose(parents["parent_mz"], p, rtol=0, atol=1e-6)
        n = rep["range_cuts"].get(repr(round(p, 6)), 0)
        note = (f"scan range differs between this parent's energies (first mass {text}): every "
                f"spectrum of it is cut to m/z {lo:.2f}-{hi:.2f}, the range all of them recorded "
                f"({n} peak(s) outside it removed)")
        parents.loc[sel, "note"] = [SRC.NOTE_SEP.join(x for x in (note, old) if x)
                                    for old in parents.loc[sel, "note"]]
    rep["resolution_source"] = _resolution_source(with_res, of, nominal)
    if with_res and with_res != of:
        rep["scans_without_resolutions"] = of - with_res
        rep["nominal_r200"] = sorted(nominal)
    rep["spectra"] = len(centroids)
    rep["peaks"] = len(spectra)
    rep["axis_offset"] = axis
    return spectra, parents, _frame(peak_cols, SCAN_PEAK_COLUMNS), rep


def isolation_widths(scans: pd.DataFrame) -> dict[float, float]:
    """Parent m/z (rounded to 1e-6) -> the trailer's isolation width over its
    MS2 scans: the largest when they differ (the window inventory then
    covers every width used)."""
    ms2 = scans[(scans["order"] == 2) & np.isfinite(scans["parent_mz"])
                & np.isfinite(scans["iso_width"])]
    out: dict[float, float] = {}
    for p, g in ms2.groupby(ms2["parent_mz"].round(6)):
        out[float(p)] = float(g["iso_width"].round(6).max())
    return out


def ms1_average(meta: dict, scans: pd.DataFrame) -> dict:
    """The MS1 scans (`order` 1, inferred ones included) averaged by the same
    rule, every cluster kept (as the server's MS1 centroids: no S/N floor;
    the window inventory keeps S/N as a column) -> {mz, intensity, resolution,
    signal_to_noise} for `sources.window_table`. Empty lists when the file
    has no MS1 scan."""
    s = scans.loc[scans["order"] == 1, "scan"].astype(int).tolist()
    if not s:
        return {"mz": [], "intensity": [], "resolution": [], "signal_to_noise": [],
                "scans_without_resolutions": 0, "nominal_r200": None,
                "resolution_source": "none"}
    # counted over the MS1 scans: an MS1 scan without resolutions takes the
    # MS1 rows' resolving power, whatever the MS2 scans carry
    with_res, of = _resolution_coverage(meta, s)
    r200 = None if with_res == of else _file_r200(scans[scans["order"] == 1])
    a = average_scans(meta, s, noise_threshold=-math.inf, nominal_r200=r200, per_scan=False)
    return {"mz": a.mz.tolist(), "intensity": a.intensity.tolist(),
            "resolution": a.resolution.tolist(), "signal_to_noise": a.signal_to_noise.tolist(),
            "scans_without_resolutions": of - with_res, "nominal_r200": r200,
            "resolution_source": _resolution_source(with_res, of, {r200} if r200 else set())}


def window_inventory(sample_id: str, ms1: dict, widths: dict[float, float], *,
                     iso_width: float | None = None, axis_offset_ppm: float = 0.0,
                     lobe_guard: bool = False, anchors: dict | None = None,
                     measured: bool = False) -> pd.DataFrame:
    """`sources.window_table` per isolation width: every parent's MS1 window
    at its own width (`isolation_widths`), or at `iso_width` for all when
    given, centred on the key x (1 + `axis_offset_ppm` 1e-6), `is_parent`
    the MS1 centroid at the precursor the MS2 spectra located (`anchors`),
    else searched for -- with the side-lobe guard when `lobe_guard` (the axis
    offset unknown), the track tolerance first when the offset is `measured`.
    Ordered by parent m/z, then m/z."""
    kw = {"axis_offset_ppm": axis_offset_ppm, "lobe_guard": lobe_guard, "anchors": anchors,
          "measured": measured}
    if iso_width is not None:
        return SRC.window_table(sample_id, ms1, list(widths), float(iso_width), **kw)
    by_w: dict[float, list] = {}
    for p, w in widths.items():
        by_w.setdefault(float(w), []).append(p)
    parts = [SRC.window_table(sample_id, ms1, ps, w, **kw) for w, ps in sorted(by_w.items())]
    if not parts:
        return SRC._frame({}, SRC.WINDOW_COLUMNS)
    out = pd.concat(parts, ignore_index=True)
    return out.sort_values(["parent_mz", "mz"], kind="stable").reset_index(drop=True)


# ---------------------------------------------------------------------------
# the server's own peaks, per scan
# ---------------------------------------------------------------------------
def scan_peak_intensities(meta: dict, scans: pd.DataFrame, spectra: pd.DataFrame, *,
                          mz_factor: float = 1.0) -> pd.DataFrame:
    """Every peak of `spectra` (the server's averaged spectra, `sources.
    SPECTRA_COLUMNS`) read in every scan of its group (`scans.spectrum`, e.g.
    from `assign_from_times`) -> `SCAN_PEAK_COLUMNS`. `mz_factor`: the file's
    m/z calibration factor, by which the server multiplied the m/z it serves
    (`sources.calibration_factor`) -- the per-scan centroids are the raw axis,
    so each peak is matched at its m/z / factor and keeps the server's m/z in
    `peak_mz` (unmatched, a factor of 1 + 3 ppm left 45 of our own standard A's
    230 server peaks with no centroid in any scan, 5 ppm 163). The group's
    centroids are
    clustered by the averaging rule and each server peak takes the cluster
    whose m/z equals its own (the server's m/z is that cluster's, within
    0.01 ppm on 891 of 893 peaks) -- when no other server peak lies within
    half its width of that cluster; a server peak no cluster matches, and
    every server peak of a cluster two of them sit in (two lines our
    averaging merged), takes the scan's centroids nearest to it within half
    its FWHM, each centroid going to its nearest server peak. A peak's rows
    hold every scan of the group (0 where it has nothing there)."""
    meta = metadata_payload(meta)
    data = meta["centroids_meta"]["data"]
    tmap = dict(zip(scans["scan"].astype(int), scans["t_s"].astype(float)))
    cols: dict[str, list] = {c: [] for c in SCAN_PEAK_COLUMNS}
    for key, sp in spectra.groupby("spectrum", sort=False):
        g = scans[scans["spectrum"] == key].sort_values("scan")
        if not len(g) or not len(sp):
            continue
        a = average_scans(meta, g["scan"], noise_threshold=-math.inf)
        sp = sp.sort_values("mz", kind="stable")
        served = sp["mz"].to_numpy(dtype=float)
        pk = served / float(mz_factor)                 # on the centroids' raw axis
        pres = sp["resolution"].to_numpy(dtype=float)
        mz_s, i_s, _, _, sc_s = _pool(data, a.scans)
        col = {int(s): j for j, s in enumerate(a.scans)}
        # the cluster nearest each server peak, and how many server peaks sit
        # within half their width of it: a cluster two claim is two lines
        width = np.where(np.isfinite(pres) & (pres > 0), pk / np.where(pres > 0, pres, 1.0),
                         pk / 1e5)
        near_c = (np.array([int(np.argmin(np.abs(a.mz - m))) for m in pk]) if len(a.mz)
                  else np.full(len(pk), -1))
        claims: dict[int, int] = {}
        for m, w, ci in zip(pk, width, near_c):
            if ci >= 0 and abs(a.mz[ci] - m) <= max(w / 2.0, m * 1e-8):
                claims[int(ci)] = claims.get(int(ci), 0) + 1
        for m, m_served, r, ci in zip(pk, served, pres, near_c):
            if ci >= 0 and abs(a.mz[ci] - m) <= m * 1e-8 and claims.get(int(ci), 0) == 1:
                vals, ns = a.per_scan[ci], a.per_scan_n[ci]
            else:
                fwhm = m / r if math.isfinite(r) and r > 0 else m / 1e5
                # nearest server peak of each centroid, within half that peak's FWHM
                near = np.abs(mz_s - m) <= fwhm / 2.0
                if near.any():
                    j = np.searchsorted(pk, mz_s[near])
                    lo = np.clip(j - 1, 0, len(pk) - 1)
                    hi = np.clip(j, 0, len(pk) - 1)
                    nearest = np.where(np.abs(pk[lo] - mz_s[near]) <= np.abs(pk[hi] - mz_s[near]),
                                       pk[lo], pk[hi])
                    mine = near.copy()
                    mine[near] = nearest == m
                else:
                    mine = near
                vals, ns = np.zeros(len(a.scans)), np.zeros(len(a.scans), dtype=np.int64)
                for s_, i_ in zip(sc_s[mine], i_s[mine]):
                    vals[col[int(s_)]] += i_
                    ns[col[int(s_)]] += 1
            for j, s in enumerate(a.scans):
                cols["spectrum"].append(str(key))
                cols["peak_mz"].append(float(m_served))
                cols["scan"].append(int(s))
                cols["t_s"].append(tmap.get(int(s), math.nan))
                cols["intensity"].append(float(vals[j]))
                cols["n_centroids"].append(int(ns[j]))
    return _frame(cols, SCAN_PEAK_COLUMNS)


# ---------------------------------------------------------------------------
# one file, end to end (still pure: the payload in, tables out)
# ---------------------------------------------------------------------------
@dataclass
class ScanMS2:
    """`extract_payload`'s result: the tables, the MS1 average (None when
    nothing read it) and the report (filters, layout, rule fit, inference
    counts, tolerances)."""
    sample_id: str
    scans: pd.DataFrame
    spectra: pd.DataFrame
    parents: pd.DataFrame
    window: pd.DataFrame
    scan_peaks: pd.DataFrame
    ms1: dict | None
    widths: dict
    report: dict
    notes: list = field(default_factory=list)


def tolerances() -> dict:
    """The constants a run used, for its manifest: every tolerance the mapping
    and the averaging compare with, and the decimals a derived key carries."""
    return {"seed_ppm": SEED_PPM, "merge_fwhm": MERGE_FWHM, "time_match_s": TIME_MATCH_S,
            "range_check_da": RANGE_CHECK_DA,
            "rule": {"slope": RULE_SLOPE, "offset_da": RULE_OFFSET_DA, "tol_mda": RULE_TOL_MDA,
                     "distinct_da": RULE_DISTINCT_DA, "decimals": RULE_DECIMALS},
            "filter_range_tol_da": FILTER_RANGE_TOL_DA, "filter_nce_tol": FILTER_NCE_TOL,
            "source_match_mda": SOURCE_MATCH_MDA, "parent_tol_da": PARENT_TOL_DA,
            "server_height_ratio": SERVER_HEIGHT_RATIO}


def _located(parents: pd.DataFrame) -> dict[float, float]:
    """Parent m/z (rounded to 1e-6) -> the precursor its spectra located, at
    the lowest energy where it was (the parents table's `precursor_mz`)."""
    out: dict[float, float] = {}
    for r in parents.sort_values("nce", kind="stable").itertuples():
        p6 = round(float(r.parent_mz), 6)
        if p6 not in out and math.isfinite(r.precursor_mz):
            out[p6] = float(r.precursor_mz)
    return out


def _spans(scans: pd.DataFrame) -> dict:
    """The MS1 scans' recorded m/z range ([lowest LowMass, highest HighMass]),
    and the MS1 and the mapped MS2 scans' time spans (s): what the window
    inventory saw, and when, beside when the MS2 ran. None where none."""
    ms1 = scans[scans["order"] == 1]
    ms2 = scans[(scans["order"] == 2) & (scans["spectrum"] != "")]

    def rng(a) -> list | None:
        a = np.asarray(a, dtype=float)
        a = a[np.isfinite(a)]
        return [float(a.min()), float(a.max())] if a.size else None
    lo, hi = rng(ms1["low_mass"]), rng(ms1["high_mass"])
    return {"ms1_scan_range": [lo[0], hi[1]] if lo and hi else None,
            "ms1_time_s": rng(ms1["t_s"]), "ms2_time_s": rng(ms2["t_s"])}


def _window_notes(parents: pd.DataFrame, window: pd.DataFrame, widths: dict, span: dict, *,
                  iso_width: float | None, axis_offset_ppm: float, n_ms1: int) -> list[str]:
    """What the window inventory could not see, on the parents rows it
    concerns (in place) -- a window past the MS1 scans' recorded m/z range is
    not measured; a window with no centroid at its precursor (`is_parent`)
    holds an ion the MS1 scans did not see -- and, for the extraction's notes,
    when the MS1 scans all precede the MS2 scans (a snapshot of the start)."""
    out = []
    t1, t2 = span.get("ms1_time_s"), span.get("ms2_time_s")
    if t1 and t2 and t1[1] < t2[0]:
        out.append(f"the MS1 scans ({n_ms1}, t {t1[0]:.1f}-{t1[1]:.1f} s) all precede the MS2 "
                   f"scans (t {t2[0]:.1f}-{t2[1]:.1f} s): the window inventory is the composition "
                   "before the MS2 ran")
    rng = span.get("ms1_scan_range")
    marked = {round(float(p), 6) for p in window.loc[window["is_parent"], "parent_mz"]}
    scale = 1.0 + axis_offset_ppm * 1e-6
    for p, w in widths.items():
        width = float(iso_width) if iso_width is not None else float(w)
        half = width / 2.0 + SRC.WINDOW_PAD_DA
        lo, hi = p * scale - half, p * scale + half
        note = ""
        if rng and (hi < rng[0] or lo > rng[1]):
            note = (f"outside the MS1 scans' m/z range ({rng[0]:.2f}-{rng[1]:.2f}): its MS1 "
                    "window is not measured")
        elif round(float(p), 6) not in marked:
            note = ("the MS1 window holds no centroid at this parent's precursor"
                    + (f" (the MS1 scans: t {t1[0]:.1f}-{t1[1]:.1f} s)" if t1 else "")
                    + ": the MS1 scans did not see it -- its MS1 share there is unseen, not "
                    "measured absent")
        if note:
            sel = np.isclose(parents["parent_mz"], p, rtol=0, atol=1e-6)
            parents.loc[sel, "note"] = [SRC.NOTE_SEP.join(x for x in (old, note) if x)
                                        for old in parents.loc[sel, "note"]]
    return out


def extract_payload(sample_id: str, meta: dict, *, lists: list | None = None,
                    noise_threshold: float = SRC.DEFAULT_NOISE_THRESHOLD,
                    iso_width: float | None = None, with_window: bool = True,
                    rule_check: bool = True) -> ScanMS2:
    """A file's `/metadata` payload -> the `ms2 extract` tables, offline:
    `scan_table` -> `map_precursors` -> `build_spectra` (its m/z axis offset
    read off the spectra, else the MS1 average) -> (`with_window`)
    `window_inventory` at the trailer's widths (`iso_width` replaces them),
    centred on that offset. The MS1 scans are averaged (`ms1_average`) only
    for those two, and only when there is an MS2 parent (`ScanMS2.ms1` None
    when they were not). Raises `ScanMapError` when the precursors cannot
    be mapped as asked."""
    if iso_width is not None and not (math.isfinite(_num(iso_width)) and _num(iso_width) > 0):
        raise ValueError(f"iso_width must be a positive width in Da, got {iso_width!r}")
    meta = metadata_payload(meta)
    scans = scan_table(meta)
    scans, mapping = map_precursors(scans, lists, rule_check=rule_check)
    # the MS1 scans are averaged only where they are read -- the window, and
    # the axis offset when the MS2 precursors read too few: an hour of MS1
    # scans is a large average, and an ordinary file needs none
    ms1_cache: dict = {}

    def ms1() -> dict:
        if "ms1" not in ms1_cache:
            ms1_cache["ms1"] = ms1_average(meta, scans)
        return ms1_cache["ms1"]
    spectra, parents, peaks, avg = build_spectra(sample_id, meta, scans,
                                                 noise_threshold=noise_threshold, ms1=ms1)
    widths = isolation_widths(scans)
    notes = list(mapping.get("notes", []))
    for k, nces in (mapping.get("missing_energies") or {}).items():
        # a parent the method ran at an NCE whose event the file never scanned
        sel = np.isclose(parents["parent_mz"], float(k), rtol=0, atol=1e-6)
        note = ("no spectrum at NCE " + "/".join(f"{x:g}" for x in nces) + ": its event was "
                "never scanned (see the extraction's notes)")
        parents.loc[sel, "note"] = [SRC.NOTE_SEP.join(x for x in (old, note) if x)
                                    for old in parents.loc[sel, "note"]]
    axis = avg["axis_offset"]
    for note in (SRC.axis_note(axis), SRC.axis_unmeasured_note(axis)):
        if note:
            notes.append(note)
    if avg["resolution_source"] == "none" and avg["spectra"]:
        notes.append("no resolution: the centroids carry none and the MS2 trailer rows read no "
                     "single FT Resolution -- every peak's resolution is NaN, so jittered "
                     "centroids of one line are not merged and the side-lobe rules are off")
    if avg.get("scans_without_resolutions"):
        r200 = avg.get("nominal_r200") or []
        notes.append(f"{avg['scans_without_resolutions']} MS2 scan(s) carry no per-centroid "
                     "resolution (the others do): "
                     + ("their centroids take the nominal resolving power (FT Resolution "
                        + "/".join(f"{v:g}" for v in r200) + " at m/z 200)" if r200 else
                        "no single FT Resolution fills it -- a peak of theirs alone reads a NaN "
                        "resolution, not merged by width, its side-lobe rules off"))
    for p, n in avg["range_cuts"].items():
        notes.append(f"parent {p}: its energies' scan ranges differ (an automatic first mass?): "
                     f"its spectra are cut to the range all of them recorded ({n} peak(s) "
                     "removed; see ms2_parents.csv)")
    window = SRC._frame({}, SRC.WINDOW_COLUMNS)
    span = _spans(scans)
    if not with_window:
        notes.append("window inventory not asked for (--no-window)")
    elif not widths:
        notes.append("window inventory skipped: no MS2 parent")
    else:
        if not ms1()["mz"]:
            notes.append("window inventory skipped: the file has no MS1 scan")
        else:
            window = window_inventory(sample_id, ms1(), widths, iso_width=iso_width,
                                      axis_offset_ppm=axis["ppm"] or 0.0,
                                      lobe_guard=axis["ppm"] is None,
                                      anchors=_located(parents),
                                      measured=axis["ppm"] is not None)
            if not len(window):
                notes.append("window inventory empty: no MS1 centroid inside any window")
            notes.extend(_window_notes(parents, window, widths, span, iso_width=iso_width,
                                       axis_offset_ppm=axis["ppm"] or 0.0,
                                       n_ms1=int((scans["order"] == 1).sum())))
    if "ms1" in ms1_cache:
        # the MS1 average's own resolutions (`ms1_average`), said as the MS1 scans'
        m1 = ms1_cache["ms1"]
        avg["ms1_resolution_source"] = m1["resolution_source"]
        avg["ms1_scans_without_resolutions"] = m1["scans_without_resolutions"]
        n = m1["scans_without_resolutions"]
        if n and m1["nominal_r200"]:
            notes.append(f"{n} MS1 scan(s) carry no per-centroid resolution: the MS1 average takes "
                         f"the nominal resolving power ({_fill_text([m1['nominal_r200']])})")
        elif n:
            notes.append(f"{n} MS1 scan(s) carry no per-centroid resolution and the MS1 rows read "
                         "no single FT Resolution: a window peak of theirs alone reads a NaN "
                         "resolution, its side-lobe rules off")
    if iso_width is not None:
        parents["isolation_width"] = float(iso_width)
    report = {"mapping": mapping, "averaging": avg, "tolerances": tolerances(), **span,
              "scans": len(scans), "header_rows": int(scans["has_header"].sum()),
              "ms1_scans": int((scans["order"] == 1).sum()),
              "ms2_scans": int((scans["order"] == 2).sum()),
              "ms2_scans_in_spectra": int(((scans["order"] == 2) & (scans["spectrum"] != "")).sum())}
    return ScanMS2(sample_id=str(sample_id), scans=scans, spectra=spectra, parents=parents,
                   window=window, scan_peaks=peaks, ms1=ms1_cache.get("ms1"), widths=widths,
                   report=report,
                   notes=notes)
