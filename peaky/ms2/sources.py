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
  * thin orchestrators, `list_sample`, `extract_sample` and `extract_file`,
    that take a client and call only io_mascope's fetchers (io_mascope stays
    the one module that talks to Mascope);
  * the per-scan source (`peaky.ms2.scans` does the work): with `per_scan`,
    `extract_sample` also reads the scans behind each server spectrum (the raw
    file's `/metadata` and each spectrum's time-series times); `extract_file`
    rebuilds a raw file's MS2 from its `/metadata` alone -- for a targeted-MS2
    file whose scan filters the server's reader garbles, which never becomes a
    sample item -- and `needs_check` / `check_file` read an instrument's file
    listing (`peaky ms2 files`).

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
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

__version__ = "0.2.0"  # + the per-scan source: extract_sample(per_scan=True), extract_file,
#                          the raw-file listing (needs_check / check_file). 0.1.0: the MS2 endpoints

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

#: The side-lobe guard, for spectra whose m/z axis offset is unknown (a raw
#: file's own centroids, `extract_file`, when `axis_offset` reads none): a
#: search-window candidate within this many line widths of a peak at least
#: `PRECURSOR_LOBE_RATIO` times brighter that lies OUTSIDE the window is that
#: line's side lobe, not the precursor: the precursor is then ambiguous at
#: that energy (the next energy is tried). An Orbitrap peak list carries weak
#: centroids a few line widths beside a much brighter line (the rule of
#: `chem.sidelobes` and of `attribute.SATELLITE_*`: 5 widths, 50x, the width
#: being the brighter line's m/z / its resolution). On a raw file whose MS2
#: axis sits tens of ppm off, the heavier parents' precursors fall outside the
#: window, and a weak upper lobe a few widths above can be the only peak left
#: inside -- to be taken for the precursor.
#: Never on the server's spectra (calibrated, keyed by the isolation centre)
#: nor on a file whose offset is measured (the search is centred on it): there
#: a weak precursor on its key beside a 50x brighter co-isolate 5 mDa - 5
#: widths away is the precursor, and the guard would refuse it.
PRECURSOR_LOBE_RATIO = 50.0
PRECURSOR_LOBE_FWHM = 5.0

#: A raw file's own MS2 axis (`extract_file`: its centroids as the instrument
#: wrote them) can sit tens of ppm off the method's precursors -- the server
#: calibrates the spectra it serves, nothing calibrates these. `axis_offset`
#: reads the offset off the file against the method's TYPED precursors (the
#: keys): per parent, at its lowest energy, the strongest centroid of its
#: isolation window, when it lies within this (ppm) of the key, is its
#: precursor; the offset is the median over the parents whose keys are typed
#: finely enough (`AXIS_KEY_PPM`). Where measured on off-axis files, the
#: parents' offsets agreed to a fraction of a ppm around their median and the
#: MS1 scans' to 1-2 ppm -- where a co-isolate does not outshine a parent's
#: precursor, which the median out-votes (and, with too few parents, the
#: cross-check of `axis_offset`); on our own standard A (keys 137.1321,
#: 155.1429, 155.1559) the offset reads +0.5 ppm. Centred there, the precursors
#: and the E50s are those of the same file with every m/z divided by
#: (1 + offset).
AXIS_SEARCH_PPM = 50.0
#: A parent reads the offset only when the decimals its key carries bound the
#: key's own rounding within this (ppm: half a unit of its last decimal over
#: its m/z). A key carries the precision it was typed with -- a list row, or a
#: precursor the scan-range rule derives from a HighMass that row set -- and
#: a key typed with two decimals sits up to 5 mDa from its precursor by that
#: alone: an on-axis file whose list read 117.09, 155.14, 187.13, 199.17 and
#: 217.14 gave +8.5 / +19.2 / +15.3 / -3.1 / +15.6 ppm, a 'median offset' of
#: +15.3 ppm that is the typing, not the axis (our own standard B's key 137.13:
#: +16.6 ppm). 4 decimals qualify from m/z 25, 3 from m/z 250; a 4-decimal key
#: whose last digit is 0 reads as 3 (117.0910 -> 117.091) and is left out
#: below m/z 250 too. With every key left out no offset is recorded (said).
AXIS_KEY_PPM = 2.0
#: The MS2 spectra name the offset when at least this many parents read one;
#: else the MS1 window's parents do; else the few that read one.
AXIS_MIN_PARENTS = 3
#: An offset at least this large (ppm) is noted on every parents row and in
#: the extraction's notes: a key's own rounding counts at most `AXIS_KEY_PPM`.
AXIS_NOTE_PPM = 3.0
#: The isolation width `axis_offset` assumes for a parent with none (Da).
AXIS_DEFAULT_WIDTH = 1.0

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


def _locate(mz, intensity, centre: float, resolution=None, *, track_first: bool = False
            ) -> tuple[int | None, str]:
    """(index of the precursor around `centre`, or None; why it is ambiguous,
    '' otherwise). `locate_precursor`'s rule, plus the side-lobe guard when
    `resolution` is given (`PRECURSOR_LOBE_RATIO`, `PRECURSOR_LOBE_FWHM`):
    the callers pass it only where the m/z axis offset is unknown. With
    `track_first` (a measured axis offset and a key typed finely enough: the
    precursor sits at `centre` within a ppm or so) the same rule runs first
    over the peaks within `_track_tol` of `centre` -- a weak precursor there
    is the precursor, whatever a brighter ion farther out in the wide window
    (a neighbouring list row's) holds; the wide search only when none is."""
    mz = np.asarray(mz, dtype=np.float64)
    inten = np.nan_to_num(np.asarray(intensity, dtype=np.float64), nan=0.0)
    tol = _search_tol(centre)
    inside = np.abs(mz - centre) <= tol
    idx = np.flatnonzero(inside)
    if not idx.size:
        return None, ""
    if track_first:
        near = idx[np.abs(mz[idx] - centre) <= _track_tol(centre)]
        if near.size and inten[near].max() > 0:
            idx = near
    idx = idx[inten[idx] >= PRECURSOR_MIN_SHARE * inten[idx].max()]
    order = np.lexsort((-inten[idx], np.abs(mz[idx] - centre)))
    best = int(idx[order[0]])
    if resolution is not None and inten[best] > 0:
        res = np.asarray(resolution, dtype=np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            width = np.where(np.isfinite(res) & (res > 0), mz / res, np.nan)
            d = np.abs(mz - mz[best])
            lobe = (~inside & (inten >= PRECURSOR_LOBE_RATIO * inten[best])
                    & (d <= PRECURSOR_LOBE_FWHM * width))
        if lobe.any():
            j = int(np.flatnonzero(lobe)[np.argmax(inten[lobe])])
            return None, (f"the candidate near the key, {mz[best]:.4f}, sits "
                          f"{d[j] / width[j]:.1f} line widths from {mz[j]:.4f}, a peak "
                          f"{inten[j] / inten[best]:.3g}x brighter just outside the search window "
                          f"({(mz[j] - centre) * 1e3:+.1f} mDa from its centre): a side lobe of "
                          "that line, not the precursor (the m/z axis may sit off the key)")
    return best, ""


def locate_precursor(mz, intensity, parent_mz: float, *, resolution=None,
                     track_first: bool = False) -> int | None:
    """Index of the precursor peak around the server's parent key, or None.

    Candidates: peaks within max(`PRECURSOR_SEARCH_PPM`, `PRECURSOR_SEARCH_MDA`)
    of the key holding >= `PRECURSOR_MIN_SHARE` of the strongest candidate;
    the one nearest the key wins, the stronger on a tie. With `resolution`
    (the peaks' resolving power) a winner that is a side lobe of a much
    brighter line just outside the window is refused (`_locate`): None --
    the side-lobe guard, which `parents_table` / `window_table` turn on only
    where the m/z axis offset is unknown (`lobe_guard`). `track_first`: the
    track tolerance around the key first (`_locate`; a measured offset, a
    finely typed key). Used on an MS2 spectrum and on the MS1 window,
    centred on the key x (1 + the file's axis offset)."""
    return _locate(mz, intensity, parent_mz, resolution, track_first=track_first)[0]


def _track_first(parent_mz: float, measured: bool) -> bool:
    """Look within the track tolerance of the corrected key first: the file's
    axis offset is measured and the key typed finely enough to sit on its ion
    (`key_rounding_ppm` <= `AXIS_KEY_PPM`; a key typed 137.13 can sit 5 mDa
    from it, and keeps the wide search)."""
    return bool(measured) and key_rounding_ppm(parent_mz) <= AXIS_KEY_PPM


def _anchors(recs: list[_Spectrum], axis_offset_ppm: float = 0.0, lobe_guard: bool = False,
             measured: bool = False) -> tuple[dict[float, float], dict[float, str]]:
    """(parent m/z -> the precursor m/z, located in the parent's lowest-energy
    spectrum that holds a candidate (records are already in NCE order);
    parent m/z -> why its precursor is ambiguous). The search is centred on
    the key x (1 + `axis_offset_ppm` 1e-6). With `lobe_guard` a candidate that
    is a side lobe (`_locate`) is refused at that energy and the next energy
    is tried -- a lobe keeps its share of its line at every energy, a weak
    precursor beside a co-isolate that fragments faster does not; a parent
    refused at every energy is ambiguous (the first energy's reason). With
    the offset `measured`, a finely typed key looks within the track
    tolerance first (`_track_first`)."""
    out: dict[float, float] = {}
    ambiguous: dict[float, str] = {}
    for r in recs:
        p6 = round(r.parent_mz, 6)
        if p6 in out or not len(r.arrays["mz"]):
            continue
        i, why = _locate(r.arrays["mz"], r.arrays["intensity"],
                         r.parent_mz * (1.0 + axis_offset_ppm * 1e-6),
                         r.arrays["resolution"] if lobe_guard else None,
                         track_first=_track_first(r.parent_mz, measured))
        if why:
            ambiguous.setdefault(p6, why)
        elif i is not None:
            out[p6] = float(r.arrays["mz"][i])
            ambiguous.pop(p6, None)
    return out, ambiguous


def key_rounding_ppm(key: float) -> float:
    """Half a unit of the last decimal `key` carries, over its m/z (ppm): how
    far its typing alone can put it from the ion (137.13 -> 36.5 ppm,
    137.1321 -> 0.36 ppm). A key with more than 6 decimals counts 0."""
    k = float(key)
    if not (math.isfinite(k) and k > 0):
        return math.inf
    for d in range(7):
        if abs(k - round(k, d)) <= 1e-9 * max(1.0, abs(k)):
            return 0.5 * 10.0 ** -d / k * 1e6
    return 0.0


def axis_offset(centroids: dict | None, summary: dict | None = None, *,
                widths: dict | None = None, ms1: dict | None = None) -> dict:
    """The file's m/z axis offset from its method's typed precursors (ppm,
    negative when it reads low), for spectra averaged from a raw file's own
    centroids.

    Per parent, in its lowest-energy spectrum with peaks: the strongest
    centroid within its isolation window (`widths`, parent m/z rounded to
    1e-6 -> width; else the summary's; else `AXIS_DEFAULT_WIDTH`) is its
    precursor when it lies within `AXIS_SEARCH_PPM` of the key -- at the
    lowest energy the precursor mostly dominates its window, and the median
    over the parents out-votes one where a co-isolate or a lobe does. Only a
    key typed finely enough reads one (`key_rounding_ppm` <= `AXIS_KEY_PPM`):
    the others are counted (`coarse_keys`), never read -- a key typed 137.13
    measures its own typing. A reading is contested, and dropped
    (`contested_keys`; `contested`: key -> the reading), when that strongest
    centroid lies off the key while another precursor candidate
    (`PRECURSOR_MIN_SHARE` of it) sits ON the key (within `_track_tol`): a
    peak on the key is evidence against an axis shift -- the strongest is a
    co-isolate (a one-parent file read -29 ppm off its co-isolate, and `ms2
    analyze` then took the co-isolate for its target). On a file whose axis
    does sit off, the reading was right and the ion on the key is the
    co-isolate: one parent cannot tell, so the reading is kept in the record
    and said (`axis_unmeasured_note`, `parents_table`). The offset is the
    median over the parents when at least `AXIS_MIN_PARENTS` read one; else
    the same rule on the MS1 window (`ms1`, the MS1 centroids, or a callable
    giving them: averaged only then); else the few there are (said) -- and
    with two or more of those few, nothing out-votes a co-isolate off the
    key, so a reading is kept only where every other reading parent's window
    bears it out (a precursor candidate within `_track_tol` of its key x (1 +
    that reading)); the others are dropped (`unconfirmed`: key -> reading),
    and with none borne out no offset is read. -> {'ppm' (None when nothing
    reads one), 'source', 'parents', 'mad_ppm', 'per_parent': {repr(key):
    ppm}, 'coarse_keys', 'contested_keys', 'contested', 'unconfirmed'}."""
    recs = _spectrum_records(centroids, summary)
    coarse = {round(r.parent_mz, 6) for r in recs
              if key_rounding_ppm(r.parent_mz) > AXIS_KEY_PPM}
    iso = _num(_summary_dict(summary).get("isolation_width"))
    widths = widths or {}

    def width_of(p: float) -> float:
        w = _num(widths.get(round(p, 6)))
        return w if math.isfinite(w) and w > 0 else (
            iso if math.isfinite(iso) and iso > 0 else AXIS_DEFAULT_WIDTH)

    def window(mz, inten, p) -> tuple[np.ndarray, np.ndarray]:
        mz = np.asarray(mz, dtype=np.float64)
        inten = np.nan_to_num(np.asarray(inten, dtype=np.float64), nan=0.0)
        sel = np.abs(mz - p) <= width_of(p) / 2.0
        return mz[sel], inten[sel]

    def read(mz, inten, p) -> tuple[float, bool] | None:
        """(ppm, contested) of the parent's reading, None when it reads none.
        Contested: the strongest centroid lies off the key while another
        precursor candidate (`PRECURSOR_MIN_SHARE` of it) sits ON the key,
        within `_track_tol` -- a co-isolate outshining the precursor, not
        the axis."""
        mz, inten = window(mz, inten, p)
        if not mz.size or inten.max() <= 0:
            return None
        j = int(np.argmax(inten))
        ppm = (mz[j] - p) / p * 1e6
        if abs(ppm) > AXIS_SEARCH_PPM:
            return None
        tol = _track_tol(p)
        on_key = (np.abs(mz - p) <= tol) & (inten >= PRECURSOR_MIN_SHARE * inten[j])
        return float(ppm), bool(abs(mz[j] - p) > tol and on_key.any())

    def bears_out(arrays, p, ppm) -> bool:
        """`p`'s window (the arrays its reading came from) holds a precursor
        candidate within `_track_tol` of its key x (1 + `ppm` 1e-6)."""
        mz, inten = window(*arrays, p)
        if not mz.size or inten.max() <= 0:
            return False
        centre = p * (1.0 + ppm * 1e-6)
        return bool(((np.abs(mz - centre) <= _track_tol(centre))
                     & (inten >= PRECURSOR_MIN_SHARE * inten.max())).any())

    ms2: dict[float, float] = {}
    ms2_arrays: dict[float, tuple] = {}
    contested: dict[float, float] = {}
    done: set = set()
    for r in recs:
        p6 = round(r.parent_mz, 6)
        if p6 in done or p6 in coarse or not len(r.arrays["mz"]):
            continue
        done.add(p6)
        v = read(r.arrays["mz"], r.arrays["intensity"], r.parent_mz)
        if v is not None:
            if v[1]:
                contested[r.parent_mz] = v[0]
            else:
                ms2[r.parent_mz] = v[0]
                ms2_arrays[r.parent_mz] = (r.arrays["mz"], r.arrays["intensity"])
    ms1_vals: dict[float, float] = {}
    ms1_arrays: dict[float, tuple] = {}
    fine = sorted({r.parent_mz for r in recs if round(r.parent_mz, 6) not in coarse})
    if len(ms2) < AXIS_MIN_PARENTS and fine and ms1:
        ms1 = ms1() if callable(ms1) else ms1         # averaged only when it is read
        for p in fine if len(ms1.get("mz") or []) else ():
            v = read(ms1["mz"], ms1["intensity"], p)
            if v is not None:
                if v[1]:
                    contested.setdefault(p, v[0])
                else:
                    ms1_vals[p] = v[0]
                    ms1_arrays[p] = (ms1["mz"], ms1["intensity"])
    unconfirmed: dict[float, float] = {}
    if len(ms2) >= AXIS_MIN_PARENTS:
        vals, source = ms2, "MS2 precursors"
    elif len(ms1_vals) >= AXIS_MIN_PARENTS:
        vals, source = ms1_vals, "MS1 window"
    else:
        vals, what, arrays = ((ms2, "MS2 precursors", ms2_arrays) if ms2 else
                              (ms1_vals, "MS1 window", ms1_arrays))
        if len(vals) >= 2:
            # too few to out-vote a co-isolate: each reading must be borne out
            # by every other reading parent's window
            kept = {p: x for p, x in vals.items()
                    if all(bears_out(arrays[q], q, x) for q in vals if q != p)}
            unconfirmed = {p: x for p, x in vals.items() if p not in kept}
            vals = kept
        source = (f"{what} (only {len(vals)} parent(s) read one"
                  + (f"; {len(unconfirmed)} reading(s) no other parent's window bears out, "
                     "dropped" if unconfirmed else "") + ")")
    read_keys = {round(float(p), 6) for p in vals}
    rec = {"coarse_keys": len(coarse),
           "contested_keys": len({round(p, 6) for p in contested} - read_keys),
           "contested": {repr(float(p)): round(x, 3) for p, x in sorted(contested.items())
                         if round(p, 6) not in read_keys},
           "unconfirmed": {repr(float(p)): round(x, 3) for p, x in sorted(unconfirmed.items())}}
    if not vals:
        return {"ppm": None, "source": None, "parents": 0, "mad_ppm": None, "per_parent": {},
                **rec}
    v = np.asarray(list(vals.values()), dtype=float)
    med = float(np.median(v))
    return {"ppm": round(med, 3), "source": source, "parents": int(len(v)),
            "mad_ppm": round(float(np.median(np.abs(v - med))), 3),
            "per_parent": {repr(float(p)): round(x, 3) for p, x in sorted(vals.items())}, **rec}


def axis_unmeasured_note(offset: dict | None) -> str:
    """The extraction's note when no offset could be read because the keys
    that might have read one are typed too coarsely, their readings are
    contested by a candidate on the key, or the few readings disagree with
    each other's windows; '' otherwise. A contested or unconfirmed reading's
    ppm is quoted: if the axis sits there, it is the offset to pass."""
    offset = offset or {}
    coarse, contested = offset.get("coarse_keys"), offset.get("contested_keys")
    unconfirmed = offset.get("unconfirmed") or {}
    if offset.get("ppm") is not None or not (coarse or contested or unconfirmed):
        return ""
    why = []
    if coarse:
        why.append(f"{coarse} parent key(s) carry too few decimals to read it (a key's own "
                   f"rounding over {AXIS_KEY_PPM:g} ppm: typed 137.13, it sits up to 5 mDa from "
                   "its precursor by its typing alone)")
    if contested:
        reads = "/".join(f"{x:+.1f}" for x in (offset.get("contested") or {}).values())
        why.append(f"{contested} parent(s) hold a precursor candidate on their key beside the "
                   "stronger centroid off it (taken for a co-isolate, not the axis"
                   + (f"; that centroid reads {reads} ppm: if the axis sits there, the ion on "
                      "the key is the co-isolate" if reads else "") + ")")
    if unconfirmed:
        why.append(f"the readings of {len(unconfirmed)} parent(s) (" + "/".join(
            f"{x:+.1f}" for x in unconfirmed.values()) + " ppm) disagree, none borne out by "
                   "the other parents' windows")
    return ("the m/z axis offset is not measured: " + "; ".join(why) + ", and no other key read "
            "one -- the precursor search is centred on the keys, its side-lobe guard on; if the "
            "axis sits off, pass `ms2 analyze --mass-offset-ppm`")


def axis_note(offset: dict | None) -> str:
    """The note on an axis offset of at least `AXIS_NOTE_PPM`, '' otherwise."""
    ppm = (offset or {}).get("ppm")
    if ppm is None or abs(ppm) < AXIS_NOTE_PPM:
        return ""
    return (f"the m/z axis sits {ppm:+.1f} ppm off the method's typed precursors (median over "
            f"{offset['parents']} parent(s), {offset['source']}): the precursor search and the "
            "MS1 window's is_parent are centred on each key x (1 + that offset), and `ms2 "
            "analyze` matches its named ions through it")


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


#: What a parent key is, in the key-offset note: the server's (the endpoints'
#: spectra), or -- for spectra averaged from a file's scans -- the precursor
#: source that named it (`parents_table(key_origin=...)`).
KEY_ORIGIN_SERVER = "the server's parent key (the method's isolation centre, by which the server is queried)"


def parents_table(sample_id: str, centroids: dict | None, summary: dict | None = None, *,
                  refusal: str | None = None, key_origin=KEY_ORIGIN_SERVER,
                  axis_offset: dict | None = None, lobe_guard: bool = False) -> pd.DataFrame:
    """One row per spectrum (and per summary group the centroid answer has no
    spectrum for): its metadata, `n_peaks`, `tic` (summed intensity), and the
    precursor peak -- located once per parent (`locate_precursor`, in its
    lowest-energy spectrum that holds a candidate) and followed across its
    energies within max(`PRECURSOR_TRACK_PPM`, `PRECURSOR_TRACK_MDA`) of that
    m/z; NaN where a spectrum has no peak there (spent, or under the noise
    threshold), and for every spectrum of a parent whose precursor is
    ambiguous (noted). `lobe_guard` turns the side-lobe guard on
    (`PRECURSOR_LOBE_*`, `_anchors`): for a raw file's spectra whose m/z
    axis offset is unknown, never for the server's. `isolation_width` is the
    summary's (NaN when it was refused). `refusal`, the server's reason for
    refusing the summary, goes
    into every row's note. `key_origin` names what a key is in the note on a
    key far from its precursor: one text, or {parent m/z rounded to 1e-6:
    text}. `axis_offset` (`axis_offset`'s answer, for spectra averaged from a
    raw file's own centroids): the search is centred on the key x (1 + its
    ppm), and an offset of `AXIS_NOTE_PPM` or more is noted on every row.
    No derived survival / breakdown numbers."""
    recs = _spectrum_records(centroids, summary)
    off_ppm = (axis_offset or {}).get("ppm")
    measured = off_ppm is not None and math.isfinite(_num(off_ppm))
    off_ppm = float(off_ppm) if measured else 0.0
    anchors, ambiguous = _anchors(recs, off_ppm, lobe_guard, measured)
    # two parents whose windows overlap can locate one ion: said on both
    shared: dict[float, list] = {}
    for p6, a in anchors.items():
        for q6, b in anchors.items():
            if q6 != p6 and abs(a - b) <= _track_tol(a):
                shared.setdefault(p6, []).append(q6)
    axis_text = axis_note(axis_offset)
    # a reading a candidate on the key contested, where no offset is measured:
    # one parent cannot tell an axis shift from a co-isolate -- said on its rows
    contested = {} if measured else dict((axis_offset or {}).get("contested") or {})
    iso = _num(_summary_dict(summary).get("isolation_width"))
    cols: dict[str, list] = {c: [] for c in PARENTS_COLUMNS}
    for r in recs:
        mz, inten = r.arrays["mz"], r.arrays["intensity"]
        notes = []
        if refusal:
            notes.append(f"summary refused ({refusal}): scan_count, t_min, t_max, hcd_v and "
                         "isolation width unknown")
        notes.extend(r.notes)
        if axis_text:
            notes.append(axis_text)
        x = contested.get(repr(float(r.parent_mz)))
        if x is not None:
            notes.append(f"axis-offset reading contested: at its lowest energy this window's "
                         f"strongest centroid sits {x:+.1f} ppm off its key while a precursor "
                         "candidate sits on the key, so no offset is read and the precursor is "
                         f"located on the key; if the m/z axis does sit {x:+.1f} ppm off, the "
                         "ion on the key is a co-isolate and this parent's precursor is that "
                         "stronger centroid -- check the window before reading its breakdown")
        pmz = pint = math.nan
        anchor = anchors.get(round(r.parent_mz, 6))
        if anchor is not None:
            # read against the key x (1 + the measured offset): the offset
            # itself is noted once (`axis_note`), not as every key's distance
            centre = r.parent_mz * (1.0 + off_ppm * 1e-6)
            if abs(anchor - centre) * 1e3 > KEY_OFFSET_NOTE_MDA:
                origin = (key_origin.get(round(r.parent_mz, 6), KEY_ORIGIN_SERVER)
                          if isinstance(key_origin, dict) else key_origin)
                notes.append(f"precursor measured at {anchor:.4f}, "
                             f"{(anchor - centre) * 1e3:+.1f} mDa from {origin}"
                             + (f", corrected for the m/z axis offset ({off_ppm:+.1f} ppm)"
                                if off_ppm else ""))
            for q6 in shared.get(round(r.parent_mz, 6), []):
                notes.append(f"located precursor coincides with parent {q6:.4f}'s ({anchor:.4f}): "
                             "their windows overlap and one ion is read for both -- the other "
                             "parent's own ion is weak or absent here; read one breakdown, not two")
            if len(mz):
                d = np.abs(mz - anchor)
                i = int(np.argmin(d))
                if d[i] <= _track_tol(anchor):
                    pmz, pint = float(mz[i]), float(inten[i])
            if r.in_centroids and not math.isfinite(pmz):
                notes.append("no peak at the precursor m/z (spent, or under the noise threshold)")
        elif round(r.parent_mz, 6) in ambiguous:
            notes.append("precursor ambiguous, not located: "
                         + ambiguous[round(r.parent_mz, 6)])
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
                 pad_da: float = WINDOW_PAD_DA, axis_offset_ppm: float = 0.0,
                 lobe_guard: bool = False, anchors: dict | None = None,
                 measured: bool = False) -> pd.DataFrame:
    """The isolation-window inventory from the MS1 centroids
    (`get_ms1_centroids`): for every parent, each centroid within
    +-(isolation_width/2 + pad_da) of the parent key, with its offset from the
    key, its `share` of the window's summed intensity and `is_parent` on the
    one `locate_precursor` picks (its side-lobe guard on with `lobe_guard`,
    as for `parents_table`). Every centroid is
    kept, whatever its S/N (filter on `signal_to_noise`); overlapping windows
    list a centroid under each parent. `axis_offset_ppm`: the m/z axis's
    offset (`axis_offset`) -- the window and the search are centred on the key
    x (1 + it); `offset_da` stays from the key. `anchors` (parent m/z rounded
    to 1e-6 -> the precursor the MS2 spectra located; a raw file's spectra):
    `is_parent` is the centroid nearest that ion within `_track_tol` of it,
    none when the MS1 scans hold nothing there -- never a weaker neighbour
    the wide search would reach; a parent with no anchor is searched for as
    above (`measured`: the offset is, so a finely typed key looks within the
    track tolerance first). Ordered by parent m/z, then m/z."""
    width = _num(isolation_width)
    if not (math.isfinite(width) and width > 0):
        raise ValueError(f"isolation_width must be a positive number, got {isolation_width!r}")
    ms1 = ms1 or {}
    a = _arrays("MS1", ms1) if ms1.get("mz") is not None else {n: np.zeros(0) for n in _ARRAYS}
    half = width / 2.0 + float(pad_da)
    scale = 1.0 + float(axis_offset_ppm) * 1e-6
    cols: dict[str, list] = {c: [] for c in WINDOW_COLUMNS}
    for p in sorted({float(x) for x in parent_mzs}):
        sel = np.flatnonzero(np.abs(a["mz"] - p * scale) <= half)
        sel = sel[np.argsort(a["mz"][sel], kind="stable")]
        inten = a["intensity"][sel]
        total = float(np.nansum(inten))
        share = inten / total if total > 0 else np.full(len(sel), np.nan)
        anchor = (anchors or {}).get(round(p, 6))
        if not len(sel):
            i_par = None
        elif anchor is not None and math.isfinite(_num(anchor)):
            d = np.abs(a["mz"][sel] - float(anchor))
            k = int(np.argmin(d))
            i_par = k if d[k] <= _track_tol(float(anchor)) else None
        else:
            i_par = locate_precursor(a["mz"][sel], inten, p * scale,
                                     resolution=a["resolution"][sel] if lobe_guard else None,
                                     track_first=_track_first(p, measured))
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
#: Where an extraction's spectra come from (`SampleMS2.source`, the manifest's
#: `source`): the server's MS2 endpoints (`extract_sample`; with `per_scan`,
#: the scans behind each spectrum are read too), or a raw file's `/metadata`
#: alone, its spectra averaged here (`extract_file`).
SOURCE_SERVER = "server MS2 endpoints"
SOURCE_SERVER_PER_SCAN = "server MS2 endpoints + the file's per-scan metadata"
SOURCE_FILE = "file metadata, per scan"


@dataclass
class SampleMS2:
    """`extract_sample`'s / `extract_file`'s result: the three tables, the
    inventory row (`row`: status, counts, notes) and what was asked and used
    (`params`); with the per-scan source also the scan table
    (`scans.SCAN_COLUMNS`), every kept peak in every scan of its spectrum
    (`scans.SCAN_PEAK_COLUMNS`) and the mapping report (`per_scan`)."""
    sample_id: str
    spectra: pd.DataFrame
    parents: pd.DataFrame
    window: pd.DataFrame
    row: dict
    params: dict
    refusal: str | None = None
    notes: list = field(default_factory=list)
    scans: pd.DataFrame | None = None
    scan_peaks: pd.DataFrame | None = None
    per_scan: dict | None = None
    source: str = SOURCE_SERVER


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
                   with_window: bool = True, per_scan: bool = False,
                   metadata: dict | None = None, on_metadata=None) -> SampleMS2:
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
    window only (`WINDOW_FAILED_NOTE`); any other error propagates.

    `per_scan`: also read the scans behind every spectrum (`_sample_per_scan`:
    the raw file's `/metadata` -- or `metadata`, already fetched; a fetched
    one is handed to `on_metadata` -- and each spectrum's time-series times).
    The spectra stay the server's; the result gains `scans` and `scan_peaks`
    (the server's peaks read in every scan)."""
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
              "precursor_track": {"ppm": PRECURSOR_TRACK_PPM, "mda": PRECURSOR_TRACK_MDA},
              # the side-lobe guard is for a raw file's unknown axis only
              "precursor_lobe": {"ratio": PRECURSOR_LOBE_RATIO, "fwhm": PRECURSOR_LOBE_FWHM,
                                 "guard_on": False}}
    if per_scan:
        params["per_scan"] = True
    if is_tof(sample_row):
        res = SampleMS2(sample_id=str(sample_id), spectra=_frame({}, SPECTRA_COLUMNS),
                        parents=_frame({}, PARENTS_COLUMNS), window=_frame({}, WINDOW_COLUMNS),
                        row=skipped_row(sample_id, sample_row), params=params)
        if per_scan:
            _attach_per_scan(res, *_no_per_scan("not an Orbitrap sample: no MS2 to map"))
        return res
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
    res = SampleMS2(sample_id=str(sample_id), spectra=spectra, parents=parents, window=window,
                    row=row, params=params, refusal=refusal, notes=notes)
    if per_scan:
        _attach_per_scan(res, *_sample_per_scan(client, sample_id, sample_row, spectra, parents,
                                                centroids, noise_threshold=noise_threshold,
                                                metadata=metadata, on_metadata=on_metadata))
    return res


# ---------------------------------------------------------------------------
# the per-scan source (peaky/ms2/scans.py): a sample's scans, a raw file's MS2
# ---------------------------------------------------------------------------
#: `extract_sample(per_scan=True)` asks one time series per spectrum, one after
#: the other with this pause between them (s): a sample of 29 spectra is 29
#: requests, and a long back-to-back burst is what trips the server's WAF.
TIMESERIES_PAUSE_S = 1.0
#: The per-scan tables are an addition to a sample's MS2 tables: when the
#: server refuses one of their requests for this sample
#: (`io_mascope.is_sample_error`), the MS2 tables are kept and written, the
#: per-scan tables are empty (with their headers) and the note
#: 'per-scan tables skipped: ' + this gives the server's reason.
PER_SCAN_FAILED_NOTE = ("the server could not serve this sample's per-scan data ({error}); "
                        "the MS2 tables are kept")
#: The scans behind a server spectrum: its time-series times (the server's
#: own grouping, read back scan by scan).
PER_SCAN_BY_TIMES = "the server's groups, by their time-series times"


def calibration_factor(record) -> tuple[float, str]:
    """(the m/z calibration factor the server applies to the spectra it
    serves for this sample's file, a note) from the sample record's
    `mz_calibration` (`par.calibration_factor`). The server multiplies the
    averaged MS2 m/z (and their time series') by it; a file's `/metadata`
    centroids are the raw axis. No calibration: 1.0. A calibration that
    carries no factor (the server's failed-calibration marker has no `par`):
    1.0, and the note says the peaks are matched as served."""
    cal = _field(record, "mz_calibration")
    if cal is None:
        return 1.0, ""
    par = cal.get("par") if isinstance(cal, dict) else None
    f = _num(par.get("calibration_factor")) if isinstance(par, dict) else math.nan
    if math.isfinite(f) and f > 0:
        return f, ""
    status = cal.get("status") if isinstance(cal, dict) else None
    return 1.0, ("the sample record's m/z calibration holds no calibration factor"
                 + (f" (status '{status}')" if status else "")
                 + ": the server's peaks are matched to the scans' centroids as served")


def _no_per_scan(reason: str):
    """Empty per-scan tables (with their dtypes), the report and the note of a
    sample whose scans are not read, for `reason`."""
    from peaky.ms2 import scans as SC
    return (SC._frame({}, SC.SCAN_COLUMNS), SC._frame({}, SC.SCAN_PEAK_COLUMNS),
            {"mode": PER_SCAN_BY_TIMES, "skipped": reason}, [f"per-scan tables skipped: {reason}"])


def _attach_per_scan(res: SampleMS2, scans, peaks, info: dict, notes: list) -> None:
    res.scans, res.scan_peaks, res.per_scan = scans, peaks, info
    res.notes.extend(notes)
    res.source = SOURCE_SERVER_PER_SCAN


def _sample_per_scan(client, sample_id: str, sample_row, spectra: pd.DataFrame,
                     parents: pd.DataFrame, centroids: dict, *, noise_threshold: float,
                     metadata: dict | None = None, on_metadata=None):
    """The scans behind a sample's server spectra -> (scans, scan peaks,
    report, notes).

    Asks the raw file's `/metadata` (the record's `sample_file_id`; skipped
    when `metadata` is given) and, per spectrum, its time series's `time` list
    (`TIMESERIES_PAUSE_S` apart). Each time names one scan: on our own files
    every MS2 scan of every group, < 1 ms apart (`scans.TIME_MATCH_S`), none an
    MS1 row (`scans.assign_from_times`). Each server peak is then read in every
    scan of its spectrum (`scans.scan_peak_intensities`) at its m/z over the
    file's calibration factor (`calibration_factor`: the server serves a
    calibrated axis, the scans' centroids are the raw one); the spectra
    themselves are not touched. Counted and noted: times that match no scan,
    a group whose server count its times do not reach, scans with no
    trailer row that no group holds -- the bad first scan the server drops,
    and the MS2 scans of a hidden tail (those after `t1`) -- and server peaks
    that read 0 in every scan of their spectrum (an axis the factor does not
    explain)."""
    from peaky.io import io_mascope as IO
    from peaky.ms2 import scans as SC
    if not centroids:
        return _no_per_scan("no MS2 spectrum to map scans to")
    fid = str(_field(sample_row, "sample_file_id") or "")
    if metadata is None and not fid:
        return _no_per_scan("the sample record names no raw file (sample_file_id)")
    times: dict[str, list] = {}
    try:
        meta = metadata
        if meta is None:
            meta = IO.fetch_file_metadata(client, fid)
            if on_metadata is not None:
                on_metadata(meta)
        # an answer that holds no per-scan centroids (`data: null` reads {})
        # is this sample's: its MS2 tables are kept, no time series asked
        meta = SC.metadata_payload(meta)
        for k, (key, entry) in enumerate(centroids.items()):
            if k:
                time.sleep(TIMESERIES_PAUSE_S)
            parent, act = _parent_of(str(key), entry if isinstance(entry, dict) else {})
            ts = IO.fetch_ms2_timeseries(client, sample_id, parent, activation=act or None,
                                         noise_threshold=noise_threshold)
            times[str(key)] = list(ts.get("time") or [])
    except Exception as exc:                     # noqa: BLE001 -- sorted by is_sample_error
        if not IO.is_sample_error(exc):
            raise
        return _no_per_scan(PER_SCAN_FAILED_NOTE.format(
            error=f"{type(exc).__name__}: {IO.server_error_text(exc)}"))
    scans, rep = SC.assign_from_times(SC.scan_table(meta), times)
    factor, cal_note = calibration_factor(sample_row)
    peaks = SC.scan_peak_intensities(meta, scans, spectra, mz_factor=factor)
    notes = [f"per-scan: {n}" for n in rep["notes"]]
    if cal_note:
        notes.append(f"per-scan: {cal_note}")
    tot = peaks.groupby(["spectrum", "peak_mz"], sort=False)["intensity"].sum()
    zero = int((tot <= 0).sum())
    if zero:
        notes.append(f"per-scan: {zero} of the {len(tot)} server peaks read 0 in every scan of "
                     "their spectrum: no centroid of its scans lies within half a line width of "
                     f"them at m/z / {factor:.9g} (the file's calibration factor) -- their rows "
                     "are zeros, not readings (an m/z axis the record's calibration does not "
                     "explain?)")
    counted = {r.spectrum: r.scan_count for r in parents.itertuples() if math.isfinite(r.scan_count)}
    short = {k: (v["matched"], counted[k]) for k, v in rep["spectra"].items()
             if k in counted and v["matched"] != counted[k]}
    if short:
        notes.append("per-scan: the time series name another number of scans than the server "
                     "counts for " + ", ".join(f"{k} ({m} named, {n:g} counted)"
                                               for k, (m, n) in short.items()))
    free = scans[~scans["has_header"] & (scans["spectrum"] == "")]
    t1 = _num(_field(sample_row, "t1"))
    tail = free[free["t_s"] > t1] if math.isfinite(t1) else free.iloc[0:0]
    if len(tail):
        notes.append(f"per-scan: {len(tail)} of the {len(free)} scan(s) in no group lie after the "
                     f"last ingested scan (t1 = {t1:.1f} s; scan(s) "
                     f"{SC._runs_text(tail['scan'])}): the hidden tail's MS2 scans, with no "
                     "trailer row and no group to name their precursor or energy")
    info = {"mode": PER_SCAN_BY_TIMES, "sample_file_id": fid or None,
            "scans": len(scans), "header_rows": int(scans["has_header"].sum()),
            "ms2_scans_mapped": int((scans["spectrum"] != "").sum()),
            "spectra": rep["spectra"], "unmatched_times": rep["unmatched_times"],
            "claimed_twice": rep["claimed_twice"], "on_ms1_rows": rep["on_ms1_rows"],
            "headerless_in_no_group": rep["headerless_in_no_group"],
            "headerless_in_no_group_scans": rep["headerless_in_no_group_scans"],
            "hidden_tail_scans": int(len(tail)),
            "mz_calibration_factor": factor, "peaks_all_zero": zero,
            "filters": SC.filter_verdict(scans), "tolerances": SC.tolerances(),
            "timeseries_pause_s": TIMESERIES_PAUSE_S}
    return scans, peaks, info, notes


class PolarityUnknown(ValueError):
    """The polarity of a raw file cannot be read and was not given: its scan
    filters are garbled (their signs are random) and no file record says one
    sign. The message says why and asks for `--polarity`."""


class DualPolarityError(ValueError):
    """A raw file whose readable scan filters carry both signs and whose MS2
    scans are read by trailer: a dual-polarity method. The per-scan MS2 is
    built from every MS2 scan and the window from every MS1 scan whatever
    their sign, so a folder labelled with one sign would hold the other's
    scans -- refused, whatever `--polarity` says (`dual_polarity`)."""


def dual_polarity(scans: pd.DataFrame) -> str:
    """Why `scans` (`scans.scan_table`) cannot be extracted under one sign
    (`DualPolarityError`): every filter agrees with its trailer and they carry
    both signs, and the file has MS2 scans by trailer; '' otherwise. A garbled
    file's signs are random and its filters never readable (never this); a
    file whose MS2 scans have no trailer row (an ordinary dual-polarity file)
    reads 'MS2 without trailer' and is not this either."""
    from peaky.ms2 import scans as SC
    v = SC.filter_verdict(scans)
    if (v["filters_readable"] != SC.FILTERS_YES or len(v["polarities_in_filters"]) < 2
            or not (scans["order"] == 2).any()):
        return ""
    return (f"its scan filters are readable and carry both signs ('{v['polarities_in_filters']}'): "
            "a dual-polarity method. The spectra would average every MS2 scan and the window "
            "every MS1 scan whatever their sign, so a folder labelled with one sign would hold "
            "the other's scans -- not extracted (no --polarity splits it)")


def resolve_polarity(explicit: str | None, file_row, scans: pd.DataFrame) -> tuple[str, str]:
    """(polarity, where it came from) for a raw file: `explicit` (checked
    against the others), else the file record's when it is one sign, else the
    scan filters' when every filter agrees with its trailer and they carry one
    sign (`scans.filter_polarity`). A garbled file's filters carry random
    signs, and its listing reads '+-' -- the server's reader saw both -- so
    neither names it: `PolarityUnknown`. An explicit sign that contradicts a
    one-sign record or readable filters is a ValueError."""
    from peaky.ms2 import scans as SC
    rec = str(_field(file_row, "polarity") or "").strip()
    rec_one = rec if rec in ("+", "-") else None
    filt = SC.filter_polarity(scans)
    if explicit is not None:
        if explicit not in ("+", "-"):
            raise ValueError(f"polarity must be '+' or '-', got {explicit!r}")
        for what, v in (("the file record", rec_one), ("the scan filters", filt)):
            if v is not None and v != explicit:
                raise ValueError(f"--polarity {explicit} contradicts {what}, which read '{v}'")
        return explicit, "explicit"
    if rec_one:
        return rec_one, "file record"
    if filt:
        return filt, "scan filters"
    v = SC.filter_verdict(scans)
    readable = {SC.FILTERS_YES: "readable", SC.FILTERS_NO: "not readable",
                SC.FILTERS_PARTLY: "only partly readable"}.get(v["filters_readable"],
                                                               "absent (no trailer row)")
    why = [f"its scan filters are {readable} ({v['filters_ok']} of {v['header_rows']} agree "
           "with their trailers"
           + (f", signs '{v['polarities_in_filters']}'" if v["polarities_in_filters"] else "")
           + ")"]
    if rec:
        why.append(f"the file record reads '{rec}'")
    raise PolarityUnknown("the polarity cannot be read off this file: " + "; ".join(why)
                          + ". Pass --polarity + or -")


def _list_frames(inclusion_lists) -> tuple[list, list]:
    """Inclusion lists (paths, or `scans.read_inclusion_list` frames) -> (frames,
    [{'name', 'rows', 'sha1', 'bytes'}]) for the manifest: a path's base name
    only, and the file's sha1 and size -- the list decides the precursor
    mapping, so a run names the list it read by its content, as it names
    its payload. A frame carries them in its `attrs` (the CLI sets them from
    the file it read); a frame without them is hashed by its rows (m/z and
    name; `bytes` None: no file)."""
    import hashlib

    from peaky.ms2 import scans as SC
    from peaky.reporting import provenance as PV
    frames, info = [], []
    for i, x in enumerate(inclusion_lists or []):
        if isinstance(x, pd.DataFrame):
            f, name = (x if set(SC.LIST_COLUMNS) <= set(x.columns)
                       else SC.inclusion_list_frame(x)), str(x.attrs.get("name") or f"list {i + 1}")
            sha1, size = x.attrs.get("sha1"), x.attrs.get("bytes")
            if not sha1:
                text = f[list(SC.LIST_COLUMNS)].to_csv(index=False, float_format="%.10g")
                sha1, size = hashlib.sha1(text.encode("utf-8")).hexdigest(), None
        else:
            p = os.path.expanduser(str(x))
            f, name = SC.read_inclusion_list(p), os.path.basename(p)
            sha1, size = PV.sha1_file(p), int(os.path.getsize(p))
        frames.append(f)
        info.append({"name": name, "rows": int(len(f)), "sha1": sha1,
                     "bytes": None if size is None else int(size)})
    return frames, info


#: `extract_file`'s status when the file has scans with no trailer row and no
#: MS2 scan by trailer: an ordinary MS2 file, whose MS2 scans the server's
#: `/metadata` gives no row (it reads them as MS order 'Ms' only on a garbled
#: file). Their precursors and energies are the server's groups:
#: `extract_sample(per_scan=True)` maps them.
MS2_WITHOUT_TRAILER = "MS2 without trailer"
MS2_WITHOUT_TRAILER_NOTE = ("{n:g} scan(s) carry no trailer row and none reads MS2 off its trailer: "
                            "an ordinary MS2 file, whose MS2 scans the server reads and groups "
                            "itself -- extract its sample with `peaky ms2 extract --sample-id ID "
                            "--per-scan`")


def extract_file(client, sample_file_id: str, *, metadata: dict | None = None, file_row=None,
                 inclusion_lists=None, polarity: str | None = None,
                 noise_threshold: float = DEFAULT_NOISE_THRESHOLD,
                 iso_width: float | None = None, with_window: bool = True,
                 rule_check: bool = True) -> SampleMS2:
    """One RAW FILE's MS2 from its `/metadata` alone (`scans.extract_payload`):
    for a file the server's MS2 endpoints cannot read -- a targeted-MS2
    (inclusion-list) file whose scan filters its reader garbles never becomes
    a sample item -- or to rebuild a file's spectra scan by scan.

    `metadata`: the payload (`io_mascope.fetch_file_metadata`), else asked of
    `client`; with it no server is needed. `inclusion_lists`: the method's
    lists (paths or frames; one shared list, or one per experiment in event
    order) -- without one the precursors come from readable filters or the
    scan-range rule (`scans.map_precursors`; a `scans.ScanMapError` names the
    events that do not fit). `polarity`: `resolve_polarity` (a garbled file
    needs it). `noise_threshold`: OUR S/N floor on the averaged spectra (the
    server's rule, reproduced: `scans.average_scans`). `iso_width` replaces the
    trailer's isolation width for the window inventory; the MS1 scans of the
    file are averaged for it (`with_window`). `rule_check=False` skips the
    list's check against the scan-range rule (a method that does not set its
    scan range by the precursor). The result's `row` is an inventory row
    (`INVENTORY_COLUMNS`) keyed by the file id, `scans` / `scan_peaks` the
    per-scan tables and `per_scan` the mapping report."""
    from peaky.io import io_mascope as IO
    from peaky.ms2 import scans as SC
    if iso_width is not None and not (math.isfinite(_num(iso_width)) and _num(iso_width) > 0):
        raise ValueError(f"iso_width must be a positive width in Da, got {iso_width!r}")
    if not (math.isfinite(_num(noise_threshold)) and _num(noise_threshold) >= 0):
        raise ValueError(f"noise_threshold must be >= 0, got {noise_threshold!r}")
    frames, lists_info = _list_frames(inclusion_lists)
    if metadata is None:
        if client is None:
            raise ValueError("extract_file needs the file's metadata or a client to ask for it")
        metadata = IO.fetch_file_metadata(client, sample_file_id)
    meta = SC.metadata_payload(metadata)
    table = SC.scan_table(meta)
    dual = dual_polarity(table)
    if dual:
        raise DualPolarityError(dual)
    pol, pol_src = resolve_polarity(polarity, file_row, table)
    res = SC.extract_payload(str(sample_file_id), meta, lists=frames or None,
                             noise_threshold=float(noise_threshold), iso_width=iso_width,
                             with_window=with_window, rule_check=rule_check)
    par = res.parents
    widths = sorted({round(w, 6) for w in res.widths.values()})
    if iso_width is not None:
        width, wsrc = float(iso_width), "explicit"
    elif len(widths) == 1:
        width, wsrc = widths[0], "trailer"
    else:
        # per parent (the parents table holds each one's): no single width
        width, wsrc = None, ("trailer, per parent" if widths else None)
    summary = {"parent_peaks": sorted({float(p) for p in par["parent_mz"]}),
               "groups": [{"parent_peak_mz": float(r.parent_mz), "activation": r.activation,
                           "hcd_energy": [float(r.hcd_v)] if math.isfinite(r.hcd_v) else [],
                           "scan_count": r.scan_count, "t_min": r.t_min, "t_max": r.t_max}
                          for r in par.itertuples()],
               "isolation_width": widths[0] if len(widths) == 1 else math.nan,
               "ms1_scan_count": res.report["ms1_scans"],
               "ms2_scan_count": res.report["ms2_scans"]}
    keys = {r.spectrum: {"parent_peak_mz": float(r.parent_mz), "activation": r.activation,
                         "mz": []} for r in par.itertuples()}
    row = inventory_row(str(sample_file_id), summary, sample_row=file_row or {}, centroids=keys)
    row["polarity"] = pol
    headerless = int((~res.scans["has_header"]).sum())
    if res.report["ms2_scans"] == 0 and not len(par):
        if headerless:
            row["status"] = MS2_WITHOUT_TRAILER
            row["note"] = MS2_WITHOUT_TRAILER_NOTE.format(n=headerless)
        else:
            row["status"], row["note"] = "no MS2", "no scan reads MS2 off its trailer"
    elif not len(par):
        row["status"] = "MS2 unmapped"
        row["note"] = "no MS2 scan could be mapped to a precursor and an energy"
    else:
        row["status"], row["note"] = "ok", ""
    mapping = res.report["mapping"]
    params = {"source": SOURCE_FILE, "noise_threshold": float(noise_threshold),
              "noise_threshold_applied_by": "peaky (the server's averaging rule, per scan)",
              "iso_width": iso_width, "isolation_width_used": width,
              "isolation_width_source": wsrc,
              "isolation_widths": {repr(float(p)): float(w) for p, w in sorted(res.widths.items())},
              "window_pad_da": WINDOW_PAD_DA,
              "precursor_search": {"ppm": PRECURSOR_SEARCH_PPM, "mda": PRECURSOR_SEARCH_MDA,
                                   "min_share": PRECURSOR_MIN_SHARE},
              "precursor_track": {"ppm": PRECURSOR_TRACK_PPM, "mda": PRECURSOR_TRACK_MDA},
              "polarity": pol, "polarity_source": pol_src,
              "inclusion_lists": lists_info, "rule_check": bool(rule_check),
              # the m/z axis offset the precursor search and the window were
              # centred on; `ms2 analyze` matches its named ions through it
              "ms2_axis_offset": res.report["averaging"]["axis_offset"],
              "precursor_lobe": {"ratio": PRECURSOR_LOBE_RATIO, "fwhm": PRECURSOR_LOBE_FWHM,
                                 "guard_on": res.report["averaging"]["axis_offset"]["ppm"] is None},
              "axis_search_ppm": AXIS_SEARCH_PPM, "axis_key_ppm": AXIS_KEY_PPM}
    info = {"mode": "file", "sample_file_id": str(sample_file_id),
            "scans": res.report["scans"], "header_rows": res.report["header_rows"],
            "ms1_scans": res.report["ms1_scans"], "ms2_scans": res.report["ms2_scans"],
            "ms2_scans_in_spectra": res.report["ms2_scans_in_spectra"],
            "ms1_scan_range": res.report["ms1_scan_range"],
            "ms1_time_s": res.report["ms1_time_s"], "ms2_time_s": res.report["ms2_time_s"],
            "mapping": mapping, "averaging": res.report["averaging"],
            "tolerances": res.report["tolerances"]}
    return SampleMS2(sample_id=str(sample_file_id), spectra=res.spectra, parents=par,
                     window=res.window, row=row, params=params, notes=list(res.notes),
                     scans=res.scans, scan_peaks=res.scan_peaks, per_scan=info,
                     source=SOURCE_FILE)


# ---------------------------------------------------------------------------
# an instrument's raw files (`peaky ms2 files`)
# ---------------------------------------------------------------------------
#: One row per raw file of the listing; the check columns are filled for the
#: files `--check` read the `/metadata` of (`check_file`), NaN / '' otherwise.
FILES_COLUMNS: dict[str, str] = {
    "sample_file_id": "str", "filename": "str", "datetime_utc": "str", "length_s": "float64",
    "polarity": "str", "instrument_type": "str", "processing_status": "str",
    "processing_detail": "str", "listed": "bool", "needs_check": "str", "checked": "bool",
    "scans": "float64",
    "header_rows": "float64", "ms2_by_trailer": "float64", "headerless": "float64",
    "filters_readable": "str", "filters_ok": "float64", "events": "float64",
    "experiments": "float64", "nce": "str", "isolation_widths": "str", "scan_range_rule": "str",
    "spectra": "float64", "precursors": "str", "inferred": "float64", "unassigned": "float64",
    "command": "str", "check_note": "str",
}
#: A listing's processing status for a file the server built its sample from.
#: Any other set status (e.g. 'needs_chemistry', the server's word for a file
#: whose filters it garbled into both polarities) marks the file for
#: `--check`; a file with no status (our own server lists none) is not marked.
STATUS_DONE = "done"
#: `needs_check` of a file named with `--check` that the listed range does
#: not hold: no listing row says its polarity or status.
NOT_LISTED = "named with --check (not in the listed range)"


def needs_check(file_row) -> str:
    """Why a listed file is worth reading per scan ('' when it is not): a
    polarity that is not one sign (the listing's '+-' -- the server's reader saw
    both, as on a file whose scan filters it garbles; also a dual-polarity file
    it processed), or a processing status set and other than 'done' (a file
    with no status is not flagged). A routine MS1 file reads '': `ms2 files
    --check` without ids never fetches it, but a file it names (`--check ID`)
    is read whatever this says."""
    pol = str(_field(file_row, "polarity") or "").strip()
    st = str(_field(file_row, "processing_status") or "").strip()
    why = []
    if pol not in ("+", "-"):
        why.append(f"polarity '{pol}'")
    if st and st != STATUS_DONE:
        why.append(f"status '{st}'")
    return ", ".join(why)


def file_row(rec, *, listed: bool = True) -> dict:
    """A `FILES_COLUMNS` row from a listing record (`io_mascope.fetch_sample_files`),
    its check columns blank. `listed=False`: a file named with `--check` that
    the listed range does not hold -- no row says its polarity or status
    (`NOT_LISTED`)."""
    out = {c: (math.nan if t == "float64" else False if t == "bool" else "")
           for c, t in FILES_COLUMNS.items()}
    out.update(sample_file_id=str(_field(rec, "sample_file_id") or ""),
               filename=str(_field(rec, "filename") or ""),
               datetime_utc=str(_field(rec, "datetime_utc") or ""),
               length_s=_num(_field(rec, "length")),
               polarity=str(_field(rec, "polarity") or ""),
               instrument_type=str(_field(rec, "instrument_type") or ""),
               processing_status=str(_field(rec, "processing_status") or ""),
               processing_detail=str(_field(rec, "processing_detail") or ""),
               listed=bool(listed), needs_check=needs_check(rec) if listed else NOT_LISTED)
    return out


def files_frame(rows: list[dict]) -> pd.DataFrame:
    return _frame({c: [r[c] for r in rows] for c in FILES_COLUMNS}, FILES_COLUMNS)


def _fmt_set(values) -> str:
    return "/".join(_fmt(v) for v in sorted({round(float(v), 4) for v in values
                                             if math.isfinite(_num(v))}))


def check_file(meta: dict, row: dict) -> dict:
    """The check columns of one file (`FILES_COLUMNS`) from its `/metadata`:
    scans, trailer rows, MS2 scans by trailer, headerless scans, whether its
    filters are readable, its MS2 events, experiments, NCEs and isolation
    widths, whether its scan range follows the precursor, what a mapping
    without a list gives (spectra, precursor sources, inferred / unassigned
    headerless scans) and the `ms2 extract` command line for it. Nothing is
    averaged."""
    from peaky.ms2 import scans as SC
    scans = SC.scan_table(meta)
    v = SC.filter_verdict(scans)
    ms2 = scans[scans["order"] == 2]
    headerless = int((~scans["has_header"]).sum())
    cols = list(FILES_COLUMNS)
    out = {c: (math.nan if FILES_COLUMNS[c] == "float64" else "")
           for c in cols[cols.index("checked") + 1:]}
    out.update({"checked": True, "scans": float(len(scans)), "header_rows": float(v["header_rows"]),
           "ms2_by_trailer": float(len(ms2)), "headerless": float(headerless),
           "filters_readable": v["filters_readable"], "filters_ok": float(v["filters_ok"]),
           "events": float(ms2["event"].nunique()), "nce": _fmt_set(ms2["nce"]),
           "isolation_widths": _fmt_set(ms2["iso_width"])})
    fid = row.get("sample_file_id", "")
    pol = row.get("polarity", "")
    notes = []
    if not len(ms2):
        if headerless:
            out["check_note"] = MS2_WITHOUT_TRAILER_NOTE.format(n=headerless)
        else:
            out["check_note"] = "no MS2 scan"
        return out
    cmd = f"peaky ms2 extract --file {fid}"
    dual = dual_polarity(scans)
    if dual:
        notes.append(f"cannot be extracted as it is: {dual}")
    elif SC.filter_polarity(scans) is None:
        if pol in ("+", "-"):
            cmd += f" --polarity {pol}"
        else:
            cmd += " --polarity <+|->"
            notes.append(("the file is not in the listed range: no listing names its sign"
                          if row.get("listed") is False else
                          f"the listing reads polarity '{pol}'")
                         + " and the filters name none: pass the method's sign")
    try:
        mapped, rep = SC.map_precursors(scans)
    except SC.ScanMapError as exc:
        out["experiments"] = math.nan
        out["scan_range_rule"] = SC.rule_summary(scans)
        if exc.needs_list and not dual:
            # no precursor source for some events: the list is what is missing
            out["command"] = cmd + " --inclusion-list <the method's list.csv>"
            out["check_note"] = "; ".join([*notes, f"without a list: {exc}"])
        else:
            # refused for a reason no list lifts (or a dual-polarity file): no
            # command would extract it
            out["check_note"] = "; ".join([*notes, ("without a list: " if exc.needs_list else
                                                    "cannot be extracted as it is: ") + str(exc)])
        return out
    inf = rep["inference"]
    out.update(experiments=float(len(rep["experiments"])),
               scan_range_rule=SC.rule_text(rep["rule"]),
               spectra=float(mapped.loc[mapped["spectrum"] != "", "spectrum"].nunique()),
               precursors="; ".join(f"{k} {n}" for k, n in rep["precursor_sources"].items()),
               inferred=float(inf["inferred"]), unassigned=float(inf["unassigned"]))
    if not rep["layout_closed"] and inf["headerless"]:
        cmd += " --inclusion-list <the method's list.csv>"
        # why the layout is open is the cycle walk's note (when it walked):
        # said once, the advice here
        said = any("event ranges are not pinned" in n for n in rep["notes"])
        notes.append(("" if said else "without a list the experiments' event ranges are not "
                      "pinned (only the method's list, sized by the Full Scan event, pins them): ")
                     + "pass the method's list: without it, headerless scans at block edges and "
                     "on a wrap stay unassigned")
    elif SC.PRECURSOR_RULE in rep["precursor_sources"]:
        cmd += " [--inclusion-list <the method's list.csv>]"
        notes.append("precursors from the scan-range rule: the method's list checks them")
    out["command"] = "" if dual else cmd
    out["check_note"] = "; ".join(notes + [n for n in rep["notes"]])
    return out


def manifest(res: SampleMS2, *, files: dict | None = None, created_utc: str | None = None
             ) -> dict:
    """The `ms2_manifest.json` record of one extraction: the sample (or raw
    file), its status, source and notes, the server's refusal, the parameters,
    the counts, the written files' sha1, the code identity
    (reporting.provenance) and, from the per-scan source, its report
    (`per_scan`: the mapping, the rule fit, the inference counts, the
    tolerances). No credentials and no server URL: the refusal reason is the
    server's message without it."""
    import sys

    import peaky
    from peaky.io import io_mascope as IO
    from peaky.ms2 import scans as SC
    from peaky.reporting import provenance as PV

    row = res.row
    counts = {"parents": int(res.parents["parent_mz"].nunique()),
              "spectra": int(row["n_spectra"]), "parents_rows": int(len(res.parents)),
              "peaks": int(len(res.spectra)), "window_rows": int(len(res.window))}
    if res.scans is not None:
        counts.update(scans=int(len(res.scans)),
                      scan_peak_rows=int(len(res.scan_peaks)) if res.scan_peaks is not None else 0)
    out = {
        "sample_id": res.sample_id, "name": row["name"], "datetime_utc": row["datetime_utc"],
        "polarity": row["polarity"], "status": row["status"],
        "energy_resolved": bool(row["energy_resolved"]), "source": res.source,
        "notes": (row["note"].split(NOTE_SEP) if row["note"] else []) + list(res.notes),
        "server": {"summary_refused": res.refusal,
                   "ms1_scans": None if math.isnan(row["ms1_scans"]) else row["ms1_scans"],
                   "ms2_scans": None if math.isnan(row["ms2_scans"]) else row["ms2_scans"],
                   "isolation_width": (None if math.isnan(row["isolation_width"])
                                       else row["isolation_width"]),
                   "hidden_tail_s": (None if math.isnan(row["hidden_tail_s"])
                                     else round(row["hidden_tail_s"], 3))},
        "parameters": res.params,
        "counts": counts,
        "files": dict(files or {}),
        "code": {"package_version": peaky.__version__, "sources_version": __version__,
                 "scans_version": SC.__version__, "io_mascope_version": IO.__version__,
                 "git": PV.git_info(os.path.dirname(os.path.abspath(__file__))),
                 "python": sys.version.split()[0], "deps": PV.dep_versions()},
        "created_utc": created_utc,
    }
    if res.source == SOURCE_FILE:
        # nothing here is the server's: the counts are the trailers' (per_scan)
        out["server"] = None
    if res.per_scan is not None:
        out["per_scan"] = res.per_scan
    return out
