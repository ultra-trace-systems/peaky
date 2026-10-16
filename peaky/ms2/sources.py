"""MS2 from Mascope: the server's MS2 endpoints -> tidy, energy-resolved tables.

Mascope averages a sample's MS2 scans into centroid spectra -- one per (parent
peak, activation) group from Mascope 1.8 on (`by_activation`), one per parent
peak with every collision energy blended before it. This module reads those
payloads into long tables that `assign_fragments` consumes directly, and is
split the way io_mascope is:

  * PURE builders on the payloads exactly as the SDK returns them --
    `parse_activation`, `spectra_table`, `parents_table`, `window_table`,
    `inventory_row`, `hidden_tail_s` -- unit-tested offline on captured
    fixtures (tests/fixtures/ms2);
  * two thin orchestrators, `list_sample` and `extract_sample`, that take a
    client and call only io_mascope's fetchers (io_mascope stays the one module
    that talks to Mascope).

What the server serves (read on the lab's own server, Mascope >= 1.8), and what
each rule below answers:

  * get_summary: parents, one group per (parent, activation) with its scan
    count, time range and HCD energy in VOLTS, the isolation width and the
    MS1 / MS2 scan counts. The activation string carries the NCE
    ('hcd40.00' = NCE 40). An MS1-only sample answers an EMPTY SCAFFOLD with
    every key present and `ms2_scan_count` 0, so "no MS2" is read off the
    COUNT, never off the response's shape. With no summary at all, `ms2
    extract` reads it off the centroid call's HTTP 400 "No scans found".
  * On some files get_summary answers HTTP 400 "Multiple isolation widths
    found for MS2 scans" while the centroid endpoint still serves every
    spectrum (a monoterpene standard of 2026-09-11: six activations of one
    parent). The spectra are then built from the centroid keys and what only
    the summary knows -- scan counts, times, volts, isolation width -- stays
    NaN, with the reason recorded.
  * A server older than 1.8 ignores `by_activation` and answers per parent,
    with no 'activation' / 'parent_peak_mz' fields: the spectra are
    energy-averaged, and that is said, never silently passed off as one energy.
    Its summary has no groups, and its `hcd_energy_map` holds the MEAN volt
    over every scan of a parent (5.16 V for a parent measured at 4.05-7.41 V):
    reported as `hcd_v` with a note, never as the volts applied.
  * The summary counts every MS2 scan but groups (and serves a spectrum for)
    only those whose scan event the server can read: a count with no group is
    'MS2 unreadable', never 'ok', and a shortfall is noted.
  * A TOF sample has no MS2 the server can read (it answers HTTP 500): it is
    skipped on its listing row's instrument type, nothing asked.
  * The parent m/z is the server's own key (the method's isolation centre),
    not the measured precursor: on that 2026-09-11 standard the key 137.13
    sits 2.3 mDa below the C10H17+ precursor (137.1323), and the server's
    time-series endpoint finds nothing for 137.1321 there. Tables keep the
    key and locate the precursor separately (`precursor_mz`).
  * An ingest that predates the server's MS2-window fix can hide MS2 scans
    recorded after the MS1 block: the sample row's `length` then runs past
    `t1` (210.5 s on that standard). `hidden_tail_s` reports it.

NCE and volts are both reported and never converted into each other: the
volts the instrument applied for one NCE depend on the precursor m/z (NCE 1 is
4.05 V at m/z 136 and 4.28 V at m/z 315 in the 2026-09-10 exploratory file).
"""
from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

__version__ = "0.1.0"

#: The S/N floor `ms2 extract` asks the server for (`noise_threshold` of
#: get_averaged_centroids), below the SDK's default of 10. A breakdown curve
#: and a branching ratio need the weak fragments too, and the server's S/N
#: floor removes them: on the 2026-09-11 monoterpene standard A, the [M+H]+
#: window's spectra hold 19-21 peaks at S/N 10 and 25-30 at S/N 3, and ions the
#: raw-file breakdown of that window tracked at 300+ counts per scan (C4H9+ at
#: NCE 5, C6H7+ at NCE 20) are in the S/N 3 spectra and missing from the S/N 10
#: ones. At 0 the lists fill with noise centroids (hundreds per spectrum).
DEFAULT_NOISE_THRESHOLD = 3.0

#: The MS1 inventory reaches this far past the isolation window's nominal edge
#: (Da). A quadrupole's transmission edge is a slope, not a step, so a centroid
#: just outside the nominal half-width is listed (its `share` says whether it
#: matters) rather than silently dropped. Not measured here: a margin, not a fit.
WINDOW_PAD_DA = 0.005

#: Where the precursor is searched for around the server's parent key:
#: max(ppm, mDa). The key is the method's isolation centre as typed: a centre
#: entered with two decimals (137.13) may sit up to 5 mDa from the ion, and on
#: the 2026-09-11 monoterpene standard B it sat 2.3 mDa below C10H17+. The keys
#: of the other 41 spectra in these files sat 0.02-0.35 mDa from their precursors.
PRECURSOR_SEARCH_PPM = 10.0
PRECURSOR_SEARCH_MDA = 5.0

#: A peak in the search window is a precursor candidate only at >= this share
#: of the window's strongest. Nearest-to-the-key alone fails where the key is
#: rounded: on standard B the side peaks at -1.3 / -2.2 mDa (< 1 % of the
#: spectrum, 0.00 of the MS1 window) are NEARER to 137.13 than the precursor at
#: +2.3 mDa (72 % of the low-energy spectrum, 95 % of the MS1 window).
#: Strongest-alone fails too: in the exploratory file's 137.0572 window at
#: NCE 5, the co-isolate 2.2 mDa above the precursor is the stronger peak
#: (19 % vs 13 % of the spectrum). Among the candidates the one NEAREST the
#: key wins (a tie: the stronger).
PRECURSOR_MIN_SHARE = 0.10

#: The precursor is located ONCE per parent -- in its lowest-energy spectrum
#: that holds a candidate -- and followed across the other energies within
#: max(ppm, mDa) of that m/z, so a breakdown curve follows one ion and never
#: jumps to a neighbour once the precursor is spent. Measured: a precursor
#: wanders <= 0.12 mDa across a parent's energies (179.1069: -0.13 -> -0.24
#: mDa from its key), and the nearest other peak in these files sits 1.9 mDa
#: from a precursor (a 2 % satellite in the 161.096 window).
PRECURSOR_TRACK_PPM = 5.0
PRECURSOR_TRACK_MDA = 0.5

#: Notes a parent key this far (mDa) from the located precursor.
KEY_OFFSET_NOTE_MDA = 1.0

#: length - t1 above this (s) is reported as a possibly hidden tail.
HIDDEN_TAIL_MIN_S = 0.5

ENERGY_AVERAGED_SERVER = ("server returns energy-averaged MS2 only (Mascope < 1.8): per-energy "
                          "spectra and breakdown curves need a newer server")
#: A server older than 1.8 sends, per parent, the MEAN volt over all its scans
#: (one value per step of a stepped energy): e.g. 5.16 V for a parent measured
#: at 4.05-7.41 V. It is reported as `hcd_v` with this note, and the inventory's
#: hcd_v_min / hcd_v_max stay NaN, since a range of means is not the volts applied.
OLD_SERVER_MEAN_VOLTS = ("hcd_v is the server's mean over every scan of the parent "
                         "(Mascope < 1.8), not a voltage the instrument applied")
OLD_SERVER_VOLTS_NOTE = ("the server gives one mean volt per parent (Mascope < 1.8), not the "
                         "volts applied: hcd_v_min / hcd_v_max unknown")
HIDDEN_TAIL_NOTE = ("the file runs {s:.1f} s past its last ingested scan (length - t1): MS2 scans "
                    "after the MS1 block may be hidden by an ingest that predates the server's "
                    "MS2-window fix (reprocess the sample on the server to read them)")
#: The two sides of a summary / centroid-key mismatch. The server answers every
#: group it summarises (a group with no peak above the noise threshold comes
#: back with empty lists), so a missing key means the keys disagree.
SUMMARY_ONLY_NOTE = ("the summary lists this group but no centroid spectrum is keyed to it "
                     "(the parent m/z and activation match no key): no peaks")
NO_GROUP_NOTE = ("no summary group matches this spectrum's parent m/z and activation: "
                 "scan_count, t_min, t_max and hcd_v unknown")
#: The server counts MS2 scans but groups none of them: it groups a scan by
#: parent and activation only when it can read the scan's event ('ms2
#: <m/z>@<activation>'), and serves spectra only for the groups, so no spectrum
#: can be asked for and no noise threshold brings one back.
MS2_UNREADABLE = "MS2 unreadable"
MS2_UNREADABLE_NOTE = ("the server counts {n:g} MS2 scans but groups none of them by parent and "
                       "activation (it could not read their scan events): no spectrum can be "
                       "asked for, at any noise threshold")
UNGROUPED_SCANS_NOTE = ("{n:g} of the {total:g} MS2 scans the server counts are in no group "
                        "(it could not read their scan events): they are in no spectrum")
NO_SPECTRUM_NOTE = ("the server's centroid answer holds no spectrum for the summary's {k} "
                    "group(s) (it answered nothing)")
NOT_ORBITRAP = "not an Orbitrap sample"
NOT_ORBITRAP_NOTE = ("the server reads MS2 from Orbitrap raw files only (this sample's "
                     "instrument type is 'tof'): nothing asked")
#: `ms2 extract` with no summary to count MS2 scans on: the centroid call is
#: made, and for a sample with no MS2 scan in its ingested window the server
#: answers HTTP 400 "Failed to get ms2 averaged centroids. No scans found
#: matching the specified filters: ..." rather than an empty answer.
NO_MS2_SCANS_NOTE = ("the server finds no MS2 scan in this sample's ingested window: its "
                     "centroid call answers 'No scans found'")


def _no_ms2_scans(exc: Exception) -> bool:
    """That HTTP 400 'No scans found matching the specified filters'."""
    msg = str(getattr(exc, "message", None) or exc).lower()
    return getattr(exc, "status_code", None) == 400 and "no scans found" in msg


#: The MS1 centroids feed the window inventory only. When the server refuses
#: them for this one sample (`io_mascope.is_sample_error`) -- its HTTP 400 "No
#: scans found matching the specified filters" for a sample window that holds
#: only the MS2 block, its own 504 "MS1 centroid extraction timed out after
#: 120s" on a long file -- the MS2 tables already read are kept and the window
#: is skipped with the server's reason (a server or connection failure still
#: propagates).
WINDOW_FAILED_NOTE = ("window inventory skipped: the server could not serve this sample's MS1 "
                      "centroids ({error}). The MS2 tables are kept; --no-window skips that "
                      "request")
#: Joins a row's notes into its `note` cell; no single note contains it, so
#: `note.split(NOTE_SEP)` gives the notes back.
NOTE_SEP = "; "

#: Long spectra table, one row per centroid of every spectrum.
SPECTRA_COLUMNS: dict[str, str] = {
    "sample_id": "str", "spectrum": "str", "parent_mz": "float64", "activation": "str",
    "kind": "str", "nce": "float64", "hcd_v": "float64", "hcd_v_values": "str",
    "scan_count": "float64", "t_min": "float64", "t_max": "float64",
    "energy_resolved": "bool", "mz": "float64", "intensity": "float64",
    "resolution": "float64", "signal_to_noise": "float64",
}
#: One row per spectrum (and per summary group the centroid answer has no spectrum for).
PARENTS_COLUMNS: dict[str, str] = {
    "sample_id": "str", "spectrum": "str", "parent_mz": "float64", "activation": "str",
    "kind": "str", "nce": "float64", "hcd_v": "float64", "hcd_v_values": "str",
    "scan_count": "float64", "t_min": "float64", "t_max": "float64",
    "isolation_width": "float64", "n_peaks": "int64", "tic": "float64",
    "precursor_mz": "float64", "precursor_intensity": "float64",
    "energy_resolved": "bool", "note": "str",
}
#: The MS1 centroids inside each parent's isolation window.
WINDOW_COLUMNS: dict[str, str] = {
    "sample_id": "str", "parent_mz": "float64", "mz": "float64", "offset_da": "float64",
    "intensity": "float64", "share": "float64", "resolution": "float64",
    "signal_to_noise": "float64", "is_parent": "bool",
}
#: One row per sample (`peaky ms2 list`). List-valued fields are ';'-joined.
INVENTORY_COLUMNS: dict[str, str] = {
    "sample_id": "str", "name": "str", "datetime_utc": "str", "polarity": "str",
    "ms1_scans": "float64", "ms2_scans": "float64", "n_parents": "int64", "parents": "str",
    "n_spectra": "int64", "nce": "str", "hcd_v_min": "float64", "hcd_v_max": "float64",
    "isolation_width": "float64", "energy_resolved": "bool", "hidden_tail_s": "float64",
    "status": "str", "note": "str",
}

_ARRAYS = ("mz", "intensity", "resolution", "signal_to_noise")
_ACTIVATION = re.compile(r"^([A-Za-z]+)(\d+(?:\.\d*)?|\.\d+)?$")


# ---------------------------------------------------------------------------
# small readers
# ---------------------------------------------------------------------------
def parse_activation(activation) -> tuple[str | None, float]:
    """'hcd40.00' -> ('hcd', 40.0); '' / None -> (None, nan).

    Any activation kind reads the same way (letters, then the number it
    carries: 'cid35' -> ('cid', 35.0); a bare 'etd' -> ('etd', nan)). The
    number is the instrument's own energy setting -- the NCE for Thermo HCD --
    and never volts. An unrecognised spelling is kept whole as the kind with
    no number rather than guessed at."""
    if activation is None:
        return None, math.nan
    text = str(activation).strip()
    if not text:
        return None, math.nan
    m = _ACTIVATION.match(text)
    if not m:
        return text, math.nan
    return m.group(1).lower(), float(m.group(2)) if m.group(2) else math.nan


def spectrum_key(parent_mz: float, activation: str = "") -> str:
    """The server's key for a spectrum: '137.1321@hcd1.00', or '137.1321' for
    a parent's energy-averaged spectrum (shortest float repr, as the server
    writes it)."""
    head = repr(float(parent_mz))
    return f"{head}@{activation}" if activation else head


def is_energy_averaged(centroids: dict | None) -> bool:
    """True for the answer of a server older than Mascope 1.8: it ignores
    `by_activation`, and its spectra carry no 'activation' field."""
    return bool(centroids) and any(
        not isinstance(v, dict) or "activation" not in v for v in centroids.values())


def _field(row, *names):
    """The first present, non-null value of `names` in a dict / Series row."""
    if row is None:
        return None
    for n in names:
        try:
            v = row[n]
        except (KeyError, IndexError, TypeError):
            continue
        if v is None or (np.ndim(v) == 0 and pd.isna(v)):
            continue
        return v
    return None


def _num(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return math.nan


def hidden_tail_s(sample_row) -> float:
    """Seconds the file runs past its last ingested scan (`length - t1`) when
    that exceeds `HIDDEN_TAIL_MIN_S`, 0.0 when it does not, NaN when the row
    does not say. An ingest that predates the server's MS2-window fix stops
    at the MS1 block and hides the MS2 scans recorded after it: on the
    2026-09-11 monoterpene standard B the tail is 210.5 s."""
    t1, length = _num(_field(sample_row, "t1")), _num(_field(sample_row, "length"))
    if not (math.isfinite(t1) and math.isfinite(length)):
        return math.nan
    tail = length - t1
    return float(tail) if tail > HIDDEN_TAIL_MIN_S else 0.0


def _fmt(v: float) -> str:
    return f"{v:g}"


def _frame(cols: dict[str, list], spec: dict[str, str]) -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(cols.get(c, []), dtype=t) for c, t in spec.items()})


# ---------------------------------------------------------------------------
# the per-spectrum record both tables are built from
# ---------------------------------------------------------------------------
@dataclass
class _Spectrum:
    spectrum: str
    parent_mz: float
    activation: str
    kind: str
    nce: float
    hcd_v: float = math.nan
    hcd_v_values: str = ""
    scan_count: float = math.nan
    t_min: float = math.nan
    t_max: float = math.nan
    arrays: dict = field(default_factory=dict)
    in_centroids: bool = True
    notes: list = field(default_factory=list)

    @property
    def energy_resolved(self) -> bool:
        return bool(self.activation)

    def order(self):
        return (self.parent_mz, math.isnan(self.nce), 0.0 if math.isnan(self.nce) else self.nce,
                self.activation)


def _volts(values) -> tuple[float, str]:
    """(mean, ';'-joined values when there is more than one)."""
    vals = [float(v) for v in (values or []) if v is not None and math.isfinite(_num(v))]
    if not vals:
        return math.nan, ""
    return float(np.mean(vals)), (";".join(_fmt(v) for v in vals) if len(vals) > 1 else "")


def _arrays(key: str, entry: dict) -> dict:
    mz = np.asarray(entry.get("mz") or [], dtype=np.float64)
    out = {"mz": mz}
    for name in _ARRAYS[1:]:
        vals = entry.get(name)
        arr = (np.full(len(mz), np.nan) if vals is None
               else np.asarray(vals, dtype=np.float64))
        if len(arr) != len(mz):
            raise ValueError(f"centroid list {key!r}: {name} has {len(arr)} values for "
                             f"{len(mz)} m/z")
        out[name] = arr
    return out


def _parent_of(key: str, entry: dict) -> tuple[float, str]:
    """(parent m/z, activation) from the entry's own fields, else its key: a
    server older than 1.8 sends neither field and keys a spectrum by its parent."""
    head, _, tail = str(key).partition("@")
    parent = _num(entry.get("parent_peak_mz")) if "parent_peak_mz" in entry else math.nan
    if not math.isfinite(parent):
        parent = _num(head)
    if not math.isfinite(parent):
        raise ValueError(f"MS2 spectrum key {key!r} names no parent m/z")
    activation = entry["activation"] if "activation" in entry else tail
    return float(parent), str(activation or "")


def _summary_dict(summary) -> dict:
    return summary if isinstance(summary, dict) else {}


def _group_index(summary) -> dict:
    """(parent m/z rounded to 1e-6, activation) -> summary group."""
    out = {}
    for g in _summary_dict(summary).get("groups") or []:
        if not isinstance(g, dict):
            continue
        p = _num(g.get("parent_peak_mz"))
        if math.isfinite(p):
            out[(round(p, 6), str(g.get("activation") or ""))] = g
    return out


def _energy_map(summary) -> dict:
    out = {}
    for k, v in (_summary_dict(summary).get("hcd_energy_map") or {}).items():
        p = _num(k)
        if math.isfinite(p):
            out[round(p, 6)] = list(v or [])
    return out


def _spectrum_records(centroids, summary) -> list[_Spectrum]:
    """One record per spectrum the centroid answer holds, plus one per summary
    group it holds none for; ordered by parent, NCE (none last), activation."""
    centroids = centroids or {}
    summ = _summary_dict(summary)
    has_groups = "groups" in summ
    groups = _group_index(summary)
    emap = _energy_map(summary)
    averaged_server = is_energy_averaged(centroids)
    recs: list[_Spectrum] = []
    used: set = set()
    averaged_parents: set = set()
    for key, entry in centroids.items():
        entry = entry if isinstance(entry, dict) else {}
        parent, act = _parent_of(key, entry)
        kind, nce = parse_activation(act)
        rec = _Spectrum(spectrum=str(key), parent_mz=parent, activation=act,
                        kind=kind or "", nce=nce, arrays=_arrays(str(key), entry))
        p6 = round(parent, 6)
        if act:
            g = groups.get((p6, act))
            if g is not None:
                used.add((p6, act))
                rec.hcd_v, rec.hcd_v_values = _volts(g.get("hcd_energy"))
                rec.scan_count = _num(g.get("scan_count"))
                rec.t_min, rec.t_max = _num(g.get("t_min")), _num(g.get("t_max"))
            elif has_groups:
                rec.notes.append(NO_GROUP_NOTE)
        else:
            # one spectrum over every energy of the parent: it has no single
            # energy of its own. A >= 1.8 summary's groups of that parent say
            # which volts it spans (hcd_v stays NaN); an older summary has no
            # groups, and its energy map holds the server's MEAN volt over all
            # of the parent's scans (one per step of a stepped energy) -- a
            # number the instrument never applied, reported as hcd_v and said so
            averaged_parents.add(p6)
            mine = [g for (gp, _), g in groups.items() if gp == p6]
            if mine:
                rec.scan_count = float(sum(_num(g.get("scan_count")) for g in mine))
                rec.t_min = float(min(_num(g.get("t_min")) for g in mine))
                rec.t_max = float(max(_num(g.get("t_max")) for g in mine))
                vals = [float(v) for g in mine for v in (g.get("hcd_energy") or [])
                        if math.isfinite(_num(v))]
                rec.hcd_v_values = ";".join(_fmt(v) for v in vals)
            elif not has_groups and emap.get(p6):
                rec.hcd_v, rec.hcd_v_values = _volts(emap[p6])
                rec.notes.append(OLD_SERVER_MEAN_VOLTS)
            rec.notes.append(ENERGY_AVERAGED_SERVER if averaged_server else
                             "energy-averaged: one spectrum over every activation of the parent")
        recs.append(rec)
    for (p6, act), g in groups.items():
        # a parent the centroid answer averaged over its activations is one
        # spectrum already, never a missing spectrum per group
        if (p6, act) in used or not act or p6 in averaged_parents:
            continue
        kind, nce = parse_activation(act)
        rec = _Spectrum(spectrum=spectrum_key(_num(g.get("parent_peak_mz")), act),
                        parent_mz=float(_num(g.get("parent_peak_mz"))), activation=act,
                        kind=kind or "", nce=nce, in_centroids=False,
                        arrays={n: np.zeros(0) for n in _ARRAYS})
        rec.hcd_v, rec.hcd_v_values = _volts(g.get("hcd_energy"))
        rec.scan_count = _num(g.get("scan_count"))
        rec.t_min, rec.t_max = _num(g.get("t_min")), _num(g.get("t_max"))
        rec.notes.append(SUMMARY_ONLY_NOTE)
        recs.append(rec)
    recs.sort(key=_Spectrum.order)
    return recs


# ---------------------------------------------------------------------------
# precursor location
# ---------------------------------------------------------------------------
def _search_tol(mz: float) -> float:
    return max(PRECURSOR_SEARCH_PPM * mz * 1e-6, PRECURSOR_SEARCH_MDA * 1e-3)


def _track_tol(mz: float) -> float:
    return max(PRECURSOR_TRACK_PPM * mz * 1e-6, PRECURSOR_TRACK_MDA * 1e-3)


def locate_precursor(mz, intensity, parent_mz: float) -> int | None:
    """Index of the precursor peak around the server's parent key, or None.

    Candidates: peaks within max(`PRECURSOR_SEARCH_PPM`, `PRECURSOR_SEARCH_MDA`)
    of the key holding >= `PRECURSOR_MIN_SHARE` of the strongest candidate;
    the one nearest the key wins, the stronger on a tie. Used on an MS2
    spectrum (`parents_table`) and on the MS1 window (`window_table`)."""
    mz = np.asarray(mz, dtype=np.float64)
    inten = np.nan_to_num(np.asarray(intensity, dtype=np.float64), nan=0.0)
    idx = np.flatnonzero(np.abs(mz - parent_mz) <= _search_tol(parent_mz))
    if not idx.size:
        return None
    idx = idx[inten[idx] >= PRECURSOR_MIN_SHARE * inten[idx].max()]
    order = np.lexsort((-inten[idx], np.abs(mz[idx] - parent_mz)))
    return int(idx[order[0]])


def _anchors(recs: list[_Spectrum]) -> dict[float, float]:
    """parent m/z -> the precursor m/z, located in the parent's lowest-energy
    spectrum that holds a candidate (records are already in NCE order)."""
    out: dict[float, float] = {}
    for r in recs:
        p6 = round(r.parent_mz, 6)
        if p6 in out or not len(r.arrays["mz"]):
            continue
        i = locate_precursor(r.arrays["mz"], r.arrays["intensity"], r.parent_mz)
        if i is not None:
            out[p6] = float(r.arrays["mz"][i])
    return out


# ---------------------------------------------------------------------------
# the tables
# ---------------------------------------------------------------------------
def spectra_table(sample_id: str, centroids: dict | None, summary: dict | None = None
                  ) -> pd.DataFrame:
    """The long spectra table: one row per centroid of every spectrum the
    centroid answer (`get_averaged_centroids`) holds, its spectrum's metadata
    filled from the summary's groups when there is a summary. Columns and
    dtypes are `SPECTRA_COLUMNS`; rows are ordered by parent m/z, NCE (none
    last), activation, then m/z. `spectra[spectra.spectrum == key]` is a peak
    list `assign_fragments` reads as it is (`mz`, `intensity`)."""
    cols: dict[str, list] = {c: [] for c in SPECTRA_COLUMNS}
    for r in _spectrum_records(centroids, summary):
        a = r.arrays
        order = np.argsort(a["mz"], kind="stable")
        n = len(order)
        for c, v in (("sample_id", str(sample_id)), ("spectrum", r.spectrum),
                     ("parent_mz", r.parent_mz), ("activation", r.activation),
                     ("kind", r.kind), ("nce", r.nce), ("hcd_v", r.hcd_v),
                     ("hcd_v_values", r.hcd_v_values), ("scan_count", r.scan_count),
                     ("t_min", r.t_min), ("t_max", r.t_max),
                     ("energy_resolved", r.energy_resolved)):
            cols[c].extend([v] * n)
        for name in _ARRAYS:
            cols[name].extend(a[name][order].tolist())
    return _frame(cols, SPECTRA_COLUMNS)


def parents_table(sample_id: str, centroids: dict | None, summary: dict | None = None, *,
                  refusal: str | None = None) -> pd.DataFrame:
    """One row per spectrum (and per summary group the centroid answer has no
    spectrum for): its metadata, `n_peaks`, `tic` (summed intensity), and the
    precursor peak -- located once per parent (`locate_precursor`, in its
    lowest-energy spectrum) and followed across its energies within
    max(`PRECURSOR_TRACK_PPM`, `PRECURSOR_TRACK_MDA`) of that m/z; NaN where a
    spectrum has no peak there (spent, or under the noise threshold).
    `isolation_width` is the summary's (NaN when it was refused). `refusal`,
    the server's reason for refusing the summary, goes into every row's note.
    No derived survival / breakdown numbers."""
    recs = _spectrum_records(centroids, summary)
    anchors = _anchors(recs)
    iso = _num(_summary_dict(summary).get("isolation_width"))
    cols: dict[str, list] = {c: [] for c in PARENTS_COLUMNS}
    for r in recs:
        mz, inten = r.arrays["mz"], r.arrays["intensity"]
        notes = []
        if refusal:
            notes.append(f"summary refused ({refusal}): scan_count, t_min, t_max, hcd_v and "
                         "isolation width unknown")
        notes.extend(r.notes)
        pmz = pint = math.nan
        anchor = anchors.get(round(r.parent_mz, 6))
        if anchor is not None:
            if abs(anchor - r.parent_mz) * 1e3 > KEY_OFFSET_NOTE_MDA:
                notes.append(f"precursor measured at {anchor:.4f}, "
                             f"{(anchor - r.parent_mz) * 1e3:+.1f} mDa from the server's parent "
                             "key (the method's isolation centre, by which the server is queried)")
            if len(mz):
                d = np.abs(mz - anchor)
                i = int(np.argmin(d))
                if d[i] <= _track_tol(anchor):
                    pmz, pint = float(mz[i]), float(inten[i])
            if r.in_centroids and not math.isfinite(pmz):
                notes.append("no peak at the precursor m/z (spent, or under the noise threshold)")
        elif r.in_centroids:
            notes.append("precursor not found near the parent key in any spectrum of this parent")
        row = {"sample_id": str(sample_id), "spectrum": r.spectrum, "parent_mz": r.parent_mz,
               "activation": r.activation, "kind": r.kind, "nce": r.nce, "hcd_v": r.hcd_v,
               "hcd_v_values": r.hcd_v_values, "scan_count": r.scan_count, "t_min": r.t_min,
               "t_max": r.t_max, "isolation_width": iso, "n_peaks": int(len(mz)),
               "tic": float(np.nansum(inten)) if len(inten) else 0.0,
               "precursor_mz": pmz, "precursor_intensity": pint,
               "energy_resolved": r.energy_resolved, "note": NOTE_SEP.join(notes)}
        for c in PARENTS_COLUMNS:
            cols[c].append(row[c])
    return _frame(cols, PARENTS_COLUMNS)


def window_table(sample_id: str, ms1: dict | None, parent_mzs, isolation_width: float, *,
                 pad_da: float = WINDOW_PAD_DA) -> pd.DataFrame:
    """The isolation-window inventory from the MS1 centroids
    (`get_ms1_centroids`): for every parent, each centroid within
    +-(isolation_width/2 + pad_da) of the parent key, with its offset from the
    key, its `share` of the window's summed intensity and `is_parent` on the
    one `locate_precursor` picks. Every centroid is kept, whatever its S/N
    (filter on `signal_to_noise`); overlapping windows list a centroid under
    each parent. Ordered by parent m/z, then m/z."""
    width = _num(isolation_width)
    if not (math.isfinite(width) and width > 0):
        raise ValueError(f"isolation_width must be a positive number, got {isolation_width!r}")
    ms1 = ms1 or {}
    a = _arrays("MS1", ms1) if ms1.get("mz") is not None else {n: np.zeros(0) for n in _ARRAYS}
    half = width / 2.0 + float(pad_da)
    cols: dict[str, list] = {c: [] for c in WINDOW_COLUMNS}
    for p in sorted({float(x) for x in parent_mzs}):
        sel = np.flatnonzero(np.abs(a["mz"] - p) <= half)
        sel = sel[np.argsort(a["mz"][sel], kind="stable")]
        inten = a["intensity"][sel]
        total = float(np.nansum(inten))
        share = inten / total if total > 0 else np.full(len(sel), np.nan)
        i_par = locate_precursor(a["mz"][sel], inten, p) if len(sel) else None
        n = len(sel)
        cols["sample_id"].extend([str(sample_id)] * n)
        cols["parent_mz"].extend([p] * n)
        cols["mz"].extend(a["mz"][sel].tolist())
        cols["offset_da"].extend((a["mz"][sel] - p).tolist())
        cols["intensity"].extend(inten.tolist())
        cols["share"].extend(share.tolist())
        cols["resolution"].extend(a["resolution"][sel].tolist())
        cols["signal_to_noise"].extend(a["signal_to_noise"][sel].tolist())
        cols["is_parent"].extend([k == i_par for k in range(n)])
    return _frame(cols, WINDOW_COLUMNS)


def inventory_row(sample_id: str, summary: dict | None, *, sample_row=None,
                  centroids: dict | None = None, refusal: str | None = None) -> dict:
    """One `peaky ms2 list` row (`INVENTORY_COLUMNS`) from the summary, the
    sample's listing row and, when given, the centroid answer (`ms2 extract`
    has it; `ms2 list` fetches it only for a sample whose summary was refused).

    `status`: 'summary refused: <reason>' | 'no MS2' (the summary's MS2 count
    is 0; `extract_sample` also sets it when there is no summary and the
    centroid call answers 'No scans found') | 'MS2 may be hidden' (the count is 0 but the file runs past its
    last ingested scan) | 'no summary' (the server answered none) | 'MS2
    unreadable' (the count is not 0 but no scan is grouped by parent, so no
    spectrum exists) | 'energy-averaged server' (Mascope < 1.8) | 'ok';
    `skipped_row` and `error_row` add 'not an Orbitrap sample' and 'error:
    <type>'. `note` says what it means and what to do; a summary whose groups
    hold fewer scans than it counts is noted too."""
    summ = _summary_dict(summary)
    ms1 = _num(summ.get("ms1_scan_count")) if summ else math.nan
    ms2 = _num(summ.get("ms2_scan_count")) if summ else math.nan
    has_groups = "groups" in summ
    recs = _spectrum_records(centroids if centroids is not None else {}, summary)
    averaged_server = is_energy_averaged(centroids) or (bool(summ) and not has_groups)
    # with a centroid answer the row describes the spectra it holds, never a
    # summary group no spectrum is keyed to; without one (`ms2 list`), the groups
    counted = [r for r in recs if r.in_centroids] if centroids is not None else recs
    if centroids is not None or has_groups:
        n_spectra = len(counted)
    else:                         # an older server: one averaged spectrum per parent
        n_spectra = len(summ.get("parent_peaks") or [])
    parents = sorted({float(p) for p in (summ.get("parent_peaks") or [])
                      if math.isfinite(_num(p))} | {r.parent_mz for r in recs})
    nces = sorted({r.nce for r in counted if math.isfinite(r.nce)})
    # the volts applied are the groups'; an older summary's energy map holds
    # one MEAN per parent, never a range of applied volts (OLD_SERVER_MEAN_VOLTS)
    groups = [g for g in (summ.get("groups") or []) if isinstance(g, dict)]
    volts = [_num(v) for g in groups for v in (g.get("hcd_energy") or [])]
    volts = [v for v in volts if math.isfinite(v)]
    tail = hidden_tail_s(sample_row)
    counts_ms2 = math.isfinite(ms2) and ms2 > 0
    # scans the server counts but could not group by parent: no spectrum holds them
    unreadable = bool(summ) and counts_ms2 and not parents and not any(
        r.in_centroids for r in recs)
    grouped = sum(n for n in (_num(g.get("scan_count")) for g in groups) if math.isfinite(n))
    notes: list[str] = []
    if refusal:
        status = f"summary refused: {refusal}"
        notes.append("spectra are built from the centroid keys, and scan counts, times, volts "
                     "and the isolation width stay unknown (pass --iso-width to `ms2 extract` "
                     "for the window inventory)")
    elif summ and ms2 == 0:
        status = "MS2 may be hidden" if tail > 0 else "no MS2"
    elif not summ:
        status = "no summary"
        notes.append("the server answered no MS2 summary")
    elif unreadable:
        status = MS2_UNREADABLE
        notes.append(MS2_UNREADABLE_NOTE.format(n=ms2))
    elif averaged_server:
        status = "energy-averaged server"
    else:
        status = "ok"
    if averaged_server and status != "no MS2":
        notes.append(ENERGY_AVERAGED_SERVER)
        if summ and not has_groups and summ.get("hcd_energy_map"):
            notes.append(OLD_SERVER_VOLTS_NOTE)
    if groups and counts_ms2 and ms2 - grouped >= 1:
        notes.append(UNGROUPED_SCANS_NOTE.format(n=ms2 - grouped, total=ms2))
    if centroids is not None and not n_spectra and groups and counts_ms2:
        notes.append(NO_SPECTRUM_NOTE.format(k=len(groups)))
    elif (not averaged_server and counted
          and not any(r.energy_resolved for r in counted)):
        notes.append("the centroid answer is energy-averaged (asked without by_activation): "
                     "one spectrum per parent, every activation blended")
    if tail > 0:
        notes.append(HIDDEN_TAIL_NOTE.format(s=tail))
    energy_resolved = (not averaged_server) and any(r.energy_resolved for r in counted)
    return {
        "sample_id": str(sample_id),
        "name": str(_field(sample_row, "sample_item_name", "name", "filename") or ""),
        "datetime_utc": str(_field(sample_row, "datetime_utc", "sample_item_datetime_utc") or ""),
        "polarity": str(_field(sample_row, "polarity") or ""),
        "ms1_scans": ms1, "ms2_scans": ms2,
        "n_parents": len(parents), "parents": ";".join(repr(p) for p in parents),
        "n_spectra": int(n_spectra), "nce": ";".join(_fmt(v) for v in nces),
        "hcd_v_min": min(volts) if volts else math.nan,
        "hcd_v_max": max(volts) if volts else math.nan,
        "isolation_width": _num(summ.get("isolation_width")) if summ else math.nan,
        "energy_resolved": bool(energy_resolved), "hidden_tail_s": tail,
        "status": status, "note": NOTE_SEP.join(notes),
    }


def skipped_row(sample_id: str, sample_row=None) -> dict:
    """The inventory row of a sample the server cannot read MS2 from, asked
    nothing: a TOF file (`is_tof`). Mascope reads MS2 from Orbitrap raw files
    only and answers anything else with HTTP 500 "Unexpected error"."""
    row = inventory_row(sample_id, None, sample_row=sample_row)
    row.update(status=NOT_ORBITRAP, note=NOT_ORBITRAP_NOTE)
    return row


def error_row(sample_id: str, exc: Exception, sample_row=None) -> dict:
    """The inventory row of a sample whose MS2 could not be read: status
    'error: <exception type>', the server's message (without its URL) as the
    note, every count unknown."""
    from peaky.io import io_mascope as IO
    row = inventory_row(sample_id, None, sample_row=sample_row)
    row.update(status=f"error: {type(exc).__name__}", note=IO.server_error_text(exc))
    return row


def is_tof(sample_row) -> bool:
    """True when the sample's listing row / record says it is a TOF file (its
    `instrument_type`, else the instrument or file name, as Mascope resolves
    it). An unknown type is asked for: only a known TOF is skipped."""
    from peaky.io import io_mascope as IO
    if sample_row is None:
        return False
    rec = sample_row.to_dict() if isinstance(sample_row, pd.Series) else dict(sample_row)
    return IO.instrument_type_for(rec) == "tof"


def inventory_frame(rows: list[dict]) -> pd.DataFrame:
    """`inventory_row` dicts -> a DataFrame with `INVENTORY_COLUMNS` dtypes."""
    return _frame({c: [r[c] for r in rows] for c in INVENTORY_COLUMNS}, INVENTORY_COLUMNS)


# ---------------------------------------------------------------------------
# orchestration (the only part that needs a client; io_mascope's fetchers only)
# ---------------------------------------------------------------------------
@dataclass
class SampleMS2:
    """`extract_sample`'s result: the three tables, the sample's inventory row
    (`row`: status, counts, notes) and what was asked and used (`params`)."""
    sample_id: str
    spectra: pd.DataFrame
    parents: pd.DataFrame
    window: pd.DataFrame
    row: dict
    params: dict
    refusal: str | None = None
    notes: list = field(default_factory=list)


def _summary_or_refusal(client, sample_id: str):
    from peaky.io import io_mascope as IO
    try:
        return IO.fetch_ms2_summary(client, sample_id), None
    except IO.MS2SummaryRefused as exc:
        return None, exc.reason


def list_sample(client, sample_id: str, sample_row=None) -> dict:
    """One inventory row for `peaky ms2 list`: the summary (one request), plus
    the centroid answer only when the summary is refused, so the row still
    names the parents and activations. `sample_row` is the batch listing's row;
    without one the sample's own record is fetched first (for `hidden_tail_s`
    and the instrument type). A TOF sample is asked nothing more
    (`skipped_row`). Server errors propagate (the CLI records them per sample)."""
    from peaky.io import io_mascope as IO
    if sample_row is None:
        sample_row = IO.fetch_sample_record(client, sample_id)
    if is_tof(sample_row):
        return skipped_row(sample_id, sample_row)
    summary, refusal = _summary_or_refusal(client, sample_id)
    centroids = IO.fetch_ms2_spectra(client, sample_id) if refusal else None
    return inventory_row(sample_id, summary, sample_row=sample_row, centroids=centroids,
                         refusal=refusal)


def extract_sample(client, sample_id: str, sample_row=None, *,
                   noise_threshold: float = DEFAULT_NOISE_THRESHOLD,
                   ms1_ppm: int = 1, iso_width: float | None = None,
                   with_window: bool = True) -> SampleMS2:
    """Fetch one sample's MS2 and build its tables.

    Calls, in order: the sample record (unless `sample_row` is given), the
    summary, then -- unless the summary counts no MS2 scan -- the per-activation
    centroids (`noise_threshold` is the server's S/N floor, default
    `DEFAULT_NOISE_THRESHOLD` = 3, not the SDK's 10), then the MS1
    centroids (`ms1_ppm` binning) for the window inventory
    (not asked with `with_window=False`).
    A TOF sample is asked nothing after its record (`skipped_row`, empty
    tables). A refused summary (io_mascope.MS2SummaryRefused) is recorded, not
    raised; the window then needs `iso_width` (else it is skipped and the note
    says why). With no summary, a centroid answer of HTTP 400 'No scans found'
    reads 'no MS2' ('MS2 may be hidden' when the file runs past `t1`); after a
    refused summary, or with a summary that counts MS2 scans, it is raised. An explicit `iso_width` replaces the server's width. A failed
    MS1 call that is this sample's (`io_mascope.is_sample_error`) skips the
    window only (`WINDOW_FAILED_NOTE`); any other error propagates."""
    from peaky.io import io_mascope as IO
    if iso_width is not None and not (math.isfinite(_num(iso_width)) and _num(iso_width) > 0):
        raise ValueError(f"iso_width must be a positive width in Da, got {iso_width!r}")
    if sample_row is None:
        sample_row = IO.fetch_sample_record(client, sample_id)
    params = {"noise_threshold": float(noise_threshold), "ms1_ppm": int(ms1_ppm),
              "by_activation": True, "iso_width": iso_width,
              "isolation_width_used": None, "isolation_width_source": None,
              "window_pad_da": WINDOW_PAD_DA,
              "precursor_search": {"ppm": PRECURSOR_SEARCH_PPM, "mda": PRECURSOR_SEARCH_MDA,
                                   "min_share": PRECURSOR_MIN_SHARE},
              "precursor_track": {"ppm": PRECURSOR_TRACK_PPM, "mda": PRECURSOR_TRACK_MDA}}
    if is_tof(sample_row):
        return SampleMS2(sample_id=str(sample_id), spectra=_frame({}, SPECTRA_COLUMNS),
                         parents=_frame({}, PARENTS_COLUMNS), window=_frame({}, WINDOW_COLUMNS),
                         row=skipped_row(sample_id, sample_row), params=params)
    summary, refusal = _summary_or_refusal(client, sample_id)
    summ = _summary_dict(summary)
    no_ms2 = bool(summ) and _num(summ.get("ms2_scan_count")) == 0
    centroids, scans_absent = {}, False
    if not no_ms2:
        try:
            centroids = IO.fetch_ms2_spectra(client, sample_id, by_activation=True,
                                             noise_threshold=noise_threshold)
        except Exception as exc:                 # noqa: BLE001 -- only the no-scan 400 is ours
            # with no summary to read the MS2 count off, an MS1-only sample
            # reaches the centroid call, which answers HTTP 400 "No scans found
            # matching the specified filters" rather than an empty answer
            if not (refusal is None and not summ and _no_ms2_scans(exc)):
                raise
            scans_absent = True
    spectra = spectra_table(sample_id, centroids, summary)
    parents = parents_table(sample_id, centroids, summary, refusal=refusal)
    row = inventory_row(sample_id, summary, sample_row=sample_row, centroids=centroids,
                        refusal=refusal)
    if scans_absent:                          # the hidden-tail rule of inventory_row holds
        row["status"] = "MS2 may be hidden" if row["hidden_tail_s"] > 0 else "no MS2"
        row["note"] = NOTE_SEP.join(n for n in (row["note"], NO_MS2_SCANS_NOTE) if n)
    notes: list[str] = []
    server_width = _num(summ.get("isolation_width")) if summ else math.nan
    if iso_width is not None:
        width, source = float(iso_width), "explicit"
    elif math.isfinite(server_width) and server_width > 0:
        width, source = server_width, "server"
    else:
        width, source = math.nan, None
    parent_mzs = sorted({float(p) for p in parents["parent_mz"]})
    window = _frame({}, WINDOW_COLUMNS)
    if not with_window:
        notes.append("window inventory not asked for (--no-window): no MS1 request")
    elif not parent_mzs:
        notes.append("window inventory skipped: no MS2 parent")
    elif source is None:
        notes.append("window inventory skipped: the isolation width is unknown "
                     + ("(the summary was refused)" if refusal else "(the summary gives none)")
                     + ", pass --iso-width")
    else:
        try:
            ms1 = IO.fetch_ms2_ms1(client, sample_id, ppm=ms1_ppm)
        except Exception as exc:                 # noqa: BLE001 -- sorted by is_sample_error
            if not IO.is_sample_error(exc):
                raise                            # the server's or the connection's
            notes.append(WINDOW_FAILED_NOTE.format(
                error=f"{type(exc).__name__}: {IO.server_error_text(exc)}"))
        else:
            window = window_table(sample_id, ms1, parent_mzs, width)
            if not len(window):
                notes.append("window inventory empty: no MS1 centroid inside any window")
    params.update(isolation_width_used=None if source is None else width,
                  isolation_width_source=source)
    return SampleMS2(sample_id=str(sample_id), spectra=spectra, parents=parents, window=window,
                     row=row, params=params, refusal=refusal, notes=notes)


def manifest(res: SampleMS2, *, files: dict | None = None, created_utc: str | None = None
             ) -> dict:
    """The `ms2_manifest.json` record of one extraction: the sample, its status
    and notes, the server's refusal, the parameters, the counts, the written
    files' sha1 and the code identity (reporting.provenance). No credentials
    and no server URL: the refusal reason is the server's message without it."""
    import sys

    import peaky
    from peaky.io import io_mascope as IO
    from peaky.reporting import provenance as PV

    row = res.row
    return {
        "sample_id": res.sample_id, "name": row["name"], "datetime_utc": row["datetime_utc"],
        "polarity": row["polarity"], "status": row["status"],
        "energy_resolved": bool(row["energy_resolved"]),
        "notes": (row["note"].split(NOTE_SEP) if row["note"] else []) + list(res.notes),
        "server": {"summary_refused": res.refusal,
                   "ms1_scans": None if math.isnan(row["ms1_scans"]) else row["ms1_scans"],
                   "ms2_scans": None if math.isnan(row["ms2_scans"]) else row["ms2_scans"],
                   "isolation_width": (None if math.isnan(row["isolation_width"])
                                       else row["isolation_width"]),
                   "hidden_tail_s": (None if math.isnan(row["hidden_tail_s"])
                                     else round(row["hidden_tail_s"], 3))},
        "parameters": res.params,
        "counts": {"parents": int(res.parents["parent_mz"].nunique()),
                   "spectra": int(row["n_spectra"]), "parents_rows": int(len(res.parents)),
                   "peaks": int(len(res.spectra)), "window_rows": int(len(res.window))},
        "files": dict(files or {}),
        "code": {"package_version": peaky.__version__, "sources_version": __version__,
                 "io_mascope_version": IO.__version__,
                 "git": PV.git_info(os.path.dirname(os.path.abspath(__file__))),
                 "python": sys.version.split()[0], "deps": PV.dep_versions()},
        "created_utc": created_utc,
    }
