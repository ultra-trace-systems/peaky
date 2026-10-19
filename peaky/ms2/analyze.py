"""`peaky ms2 analyze`: one `ms2 extract` sample folder -> fragment readings
across energies, breakdown curves and E50s, fragment onsets and attributions.

Local computation only: it reads the tables `ms2 extract` wrote
(`ms2_spectra.parquet`, `ms2_parents.csv`, `ms2_window.csv`,
`ms2_manifest.json`) and never asks Mascope anything. The arithmetic lives in
the pure modules -- `readings` (window reading + the ladder gate), `attribute`
(composition, intact test, weights, branching) and `breakdown` (survival, E50,
onsets, ion budget); this module picks each window's ions, recalibrates it,
calls them, and writes the tables, the figures and the manifest.

Window ions are never guessed. The user names the ions (`--ion MZ=FORMULA`);
a parent window's ions are the named ions whose exact m/z lies inside its
isolation window (half the width + `sources.WINDOW_PAD_DA` around the server's
key), and its target is the named ion nearest the precursor `ms2 extract`
located -- within `TARGET_MATCH_PPM`, else the window is skipped with a note.
On a folder whose m/z axis sits off (a raw file's own centroids, `ms2 extract
--file`: its manifest's `ms2_axis_offset`; or `--mass-offset-ppm`), the
located precursor and the MS1 window are read through that offset first
(`folder_axis_offset`); the per-window recalibration then follows as on any
folder.
Two named ions within the reading tolerance of each other are never both
window ions (`drop_aliases`), and one breakdown holds one activation kind.

A folder with the per-scan tables (`ms2_scan_peaks.parquet`, `ms2_scans.csv`:
`ms2 extract --file` or `--per-scan`) also gets replicate statistics: the
averaged analysis's roles and weights re-read scan by scan (`replicates`),
the scatter of F, F_lo and S, every item's presence and CV
(`ms2_replicates.csv`), and a bootstrap interval of every E50. Without them
every table is what it was, the new columns blank, and the run says so. A
peak that is an Orbitrap harmonic ghost of a brighter line is read as no
formula and kept out of the TIC (`readings.harmonic_ghosts`; counted).

Given each named ion's adduct (`--ions` CSV 'adduct' column, `--adduct`,
`--reagent`: `channels.assign_adducts`), every analysed window's products
are read as channels of the ion's adduct -- the reagent ion, its loss, a
monomer shed, a skeletal fragment -- per ion and energy
(`ms2_channels.csv`), and each ion gets a heuristic binding class (adduct,
cluster, covalent, mixed, unknown: `ms2_classes.csv`), both from
`channels`. With no adduct given neither is written, and every other table
is what it was.

Given RIVAL formulas of one ion (`--ions` CSV 'group' column -- or rows
sharing an 'mz' cell --, `--rival MZ=F1,F2`), the window whose target they
are is read once per rival and on decoys, and `arbitrate` says which one
its products support (`ms2_arbitration.csv`); the other tables read one
rival per group -- the one named first, or the supported one with
`--pick-supported` -- never two. With no rival given nothing of this runs.
"""
from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from peaky.chem import chemistry as C
from peaky.ms2 import arbitrate as AR
from peaky.ms2 import attribute as AT
from peaky.ms2 import breakdown as BD
from peaky.ms2 import channels as CH
from peaky.ms2 import readings as RD
from peaky.ms2 import replicates as RP
from peaky.ms2 import scans as SC
from peaky.ms2 import sources as SRC
from peaky.ms2 import subformula as SF

__version__ = "0.4.0"  # + arbitration between rival formulas of one precursor
#                          (ms2_arbitration.csv). 0.3.0: product channels against
#                          each ion's adduct, a binding class per ion
#                          (ms2_channels.csv, ms2_classes.csv). 0.2.0: replicate
#                          statistics and the E50 bootstrap from the per-scan
#                          tables; harmonic ghosts read as no formula, out of the TIC.
#                          0.1.1: per-parent isolation widths from the parents table (an
#                          `ms2 extract --file` folder); the folder's m/z axis offset
#                          (its manifest, or --mass-offset-ppm) in the target match
#                          and the MS1 shares. 0.1.0: PR #80

#: A named ion is a window's target when its exact m/z is within this of the
#: precursor `ms2 extract` located (uncalibrated axis; the tol_mda floor
#: applies below m/z 30). Measured on our own files: the located precursors sat
#: within 1.8 ppm of their formulas (2026-09-10: -0.1 to +0.7 ppm over seven
#: parents; 2026-09-11: -1.1, -1.7, -1.7 and -1.5 ppm), and the nearest other
#: named ion of a window 16 ppm away (C8H9O2+, 2.2 mDa above C4H10^NO4+).
TARGET_MATCH_PPM = 10.0
#: The typed m/z of `--ion MZ=FORMULA` must lie within the precursor search
#: window of `ms2 extract` (max(10 ppm, 5 mDa)) of the formula's exact m/z: a
#: key typed with two decimals is fine, a formula from another window is a typo.
ION_CHECK_PPM = SRC.PRECURSOR_SEARCH_PPM
ION_CHECK_MDA = SRC.PRECURSOR_SEARCH_MDA
#: How many of the target's fragments panel B of the figure draws (by their
#: largest TIC fraction).
FIGURE_TOP_FRAGMENTS = 6

#: What S rests on. Orbitrap intensities are reported per unit injection time
#: (AGC fills the trap to a target charge, so the injection time follows the
#: ion flux and intensity x injection time is what stays constant): spectra of
#: different injection times are on one scale. S compares spectra recorded at
#: different times, so it is a breakdown curve while the precursor's supply --
#: the source, the transmission, the isolation -- holds steady across the
#: energies; F and F_lo divide within one spectrum and do not need that.
S_STEADY = "valid only with a steady precursor supply across the energies"
S_NOTE = ("s = I(E) / I(E_min) compares spectra recorded at different times (Orbitrap "
          "intensities are per unit injection time, so the injection time itself does not "
          "matter): it is a breakdown curve only with a steady precursor supply (source, "
          "transmission, isolation) across the energies. f and f_lo are ratios "
          "within one spectrum and need no such assumption, but they see only the products "
          "the spectrum holds: a product outside the observed m/z range makes them read high, "
          "where s does not")
GATE_NOTES = {
    RD.GATE_LADDER: ("ladder-gated k_max 4: a reading whose loss needs a fourth piece is kept "
                     "only when the same window ion's previous step (this loss minus one named "
                     "closed-shell library molecule, printed among its pieces or not; an unnamed "
                     "remainder or a radical is no step) is read at rank 1 with <= 3 pieces at "
                     "some energy of the window (ladder_support)"),
    RD.GATE_STRICT: "strict k_max 3: no loss of more than three pieces is read",
    RD.GATE_UNGATED: ("ungated k_max {k}: every loss of up to {k} pieces is read, with no "
                      "ladder test (more chance compositions at every peak than the default)"),
}

#: Why a window has no breakdown (`WindowResult.reason`, in the manifest).
#: 'skipped': nothing read; 'readings only': read, but no energy axis.
SKIP_NO_PEAKS = "no spectrum with peaks"
SKIP_NOT_LOCATED = "no precursor located"
SKIP_NO_TARGET = "no named target"
NOT_RESOLVED = "no energy-resolved spectrum"


def gate_note(mode: str, k: int) -> str:
    """The manifest's and the console's words for a gate mode read at `k`:
    an explicit k_max below 3 is stricter than the default, not looser."""
    if mode == RD.GATE_UNGATED and k < 3:
        return (f"k_max {k}: no loss of more than {k} piece{'s' if k > 1 else ''} is read "
                "-- stricter than the default 3 (no ladder test)")
    return GATE_NOTES[mode].format(k=k)

BREAKDOWN_COLUMNS = {
    "parent_mz": "float64", "spectrum": "str", "nce": "float64", "hcd_v": "float64",
    "ion": "str", "role": "str", "ion_mz": "float64", "intensity": "float64",
    "n_precursor_peaks": "int64", "frac": "float64", "frag_weighted": "float64",
    "frag_all": "float64", "f": "float64", "f_lo": "float64", "s": "float64",
    "branching_sum": "float64", "branching_flag": "bool", "tic": "float64",
    "budget_explained": "float64", "budget_unexplained": "float64", "n_peaks": "int64",
    "n_unexplained": "int64",
    # Orbitrap harmonic ghosts of the spectrum: out of the TIC and n_peaks
    "n_harmonic": "int64", "harmonic_intensity": "float64",
    # replicates (per-scan tables; else 0 / NaN): the spectrum's scans, the
    # precursor's CV over them, and the scan-to-scan spread of F, F_lo, S
    "n_scans": "int64", "intensity_cv": "float64", "f_sd": "float64", "f_sem": "float64",
    "f_lo_sd": "float64", "f_lo_sem": "float64", "s_sd": "float64", "s_sem": "float64",
}
E50_COLUMNS = {
    "parent_mz": "float64", "ion": "str", "role": "str", "ion_mz": "float64", "curve": "str",
    "e50_nce": "float64", "e50_v": "float64", "e50_ecm": "float64", "flag": "str",
    "monotonic": "bool", "n_energies": "int64", "nce_min": "float64", "nce_max": "float64",
    "value_at_min": "float64", "value_at_max": "float64", "note": "str",
    # the bootstrap interval (per-scan tables and --bootstrap > 0; else NaN / 0
    # / ''): its bounds, where a NaN bound lies ('below' / 'above' the
    # energies, 'gap': across an unobserved one), the resamples, the share
    # that cross, and how many of the energies the band rests on add none to
    # the curve (the two about the averaged crossing; for S its reference,
    # the lowest energy, too); and, with the per-scan tables whatever
    # --bootstrap (else 0), the window's energies that add no spread (fewer
    # than two scans, or every precursor / fragment peak held)
    "e50_nce_lo": "float64", "e50_nce_hi": "float64", "e50_v_lo": "float64",
    "e50_v_hi": "float64", "e50_ecm_lo": "float64", "e50_ecm_hi": "float64",
    "e50_lo_censor": "str", "e50_hi_censor": "str",
    "boot_n": "int64", "boot_cross": "float64", "n_one_scan": "int64",
    "one_scan_bracket": "int64",
}
FRAGMENT_COLUMNS = {
    "parent_mz": "float64", "fragment": "str", "mz_theo": "float64", "candidates": "str",
    "n_candidates": "int64", "losses": "str", "call": "str", "basis": "str", "weights": "str",
    "weight_source": "str", "excluded": "str", "test_note": "str", "onset_nce": "float64",
    "onset_v": "float64", "in_source": "bool", "max_frac": "float64",
    "max_frac_nce": "float64", "n_spectra": "int64",
    # replicates at its largest TIC fraction's spectrum (per-scan tables; else NaN)
    "presence_at_max": "float64", "cv_at_max": "float64",
}
BRANCHING_COLUMNS = {
    "parent_mz": "float64", "ion": "str", "fragment": "str", "spectrum": "str",
    "nce": "float64", "hcd_v": "float64", "weight": "float64", "frag_gain": "float64",
    "ion_loss": "float64", "branching": "float64",
}
#: The curves an E50 is read off (breakdown column -> curve name).
CURVES = (("f", "F"), ("f_lo", "F_lo"), ("s", "S"))


def _frame(rows, spec: dict[str, str]) -> pd.DataFrame:
    df = pd.DataFrame(rows) if not isinstance(rows, pd.DataFrame) else rows
    return pd.DataFrame({c: pd.Series(df[c].to_numpy() if c in df else [], dtype=t)
                         for c, t in spec.items()})


# ---------------------------------------------------------------------------
# the named ions
# ---------------------------------------------------------------------------
def ion_mz(formula: str) -> float:
    counts, charge = SF.parse_ion(formula)
    return float(C.ion_mass(counts, charge))


def parse_ion_spec(spec: str) -> tuple[str, float, float]:
    """'137.1323=C10H17+' -> ('C10H17+', 137.1323, exact m/z). The formula is
    normalised (`format_ion`); the typed m/z must lie within max(10 ppm,
    5 mDa) of the formula's exact m/z (a typo otherwise)."""
    text = str(spec).strip()
    mz_txt, sep, formula = text.partition("=")
    if not sep:
        raise ValueError(f"--ion {spec!r}: expected MZ=FORMULA, e.g. 137.1323=C10H17+")
    try:
        typed = float(mz_txt)
    except ValueError:
        raise ValueError(f"--ion {spec!r}: {mz_txt!r} is not an m/z") from None
    return _checked(formula.strip(), typed, f"--ion {spec!r}")


def _checked(formula: str, typed: float, what: str) -> tuple[str, float, float]:
    counts, charge = SF.parse_ion(formula)
    norm = SF.format_ion(counts, charge)
    exact = float(C.ion_mass(counts, charge))
    if math.isfinite(typed):
        tol = max(ION_CHECK_PPM * exact * 1e-6, ION_CHECK_MDA * 1e-3)
        if abs(typed - exact) > tol:
            raise ValueError(f"{what}: the m/z given ({typed:g}) is {(typed - exact) * 1e3:+.1f} "
                             f"mDa from {norm} (m/z {exact:.5f}); a typo?")
    return norm, typed, exact


def read_ions_csv(path: str) -> list[tuple[str, float, float]]:
    """A CSV with a `formula` column and optionally `mz` (checked as for --ion;
    a missing value -- empty, R's NA, '#N/A', 'nan', `arbitrate.MISSING_CELLS`
    -- or a non-finite number checks nothing; a cell that is no number is
    refused, its row named) and `adduct` (read by `read_ions_adducts`)."""
    df = pd.read_csv(os.path.expanduser(path), dtype=str, keep_default_na=False)
    if "formula" not in df.columns:
        raise ValueError(f"{path}: needs a 'formula' column (and optionally 'mz')")
    out = []
    for i, r in df.iterrows():
        if not str(r["formula"]).strip():
            continue
        cell = str(r["mz"]).strip() if "mz" in df.columns else ""
        # a missing value (R's NA, a spreadsheet's #N/A, 'nan':
        # `arbitrate.MISSING_CELLS`) or a number that is not finite: no m/z
        if AR._no_mz(cell):
            typed = math.nan
        else:
            try:
                typed = float(cell)
            except ValueError:
                raise ValueError(f"{path} row {i + 2}: the mz cell {cell!r} is no m/z "
                                 "(a number, or empty / NA for none)") from None
        out.append(_checked(str(r["formula"]).strip(), typed, f"{path} row {i + 2}"))
    return out


def read_rival_groups_csv(path: str) -> list[AR.RivalGroup]:
    """The `--ions` CSV's rival groups (`arbitrate.read_rival_groups`): rows
    sharing a `group` cell, or with no group cell sharing one `mz` cell, are
    rival formulas of one ion. [] when none."""
    df = pd.read_csv(os.path.expanduser(path), dtype=str, keep_default_na=False)
    return AR.read_rival_groups(df, str(path))


def read_ions_adducts(path: str) -> dict[str, str]:
    """The `--ions` CSV's optional 'adduct' column: ion formula (normalised as
    `read_ions_csv` writes it) -> its ion-form label ({} without the column;
    a blank cell, or a missing value -- R's NA, `arbitrate.MISSING_CELLS` --,
    is left out: no adduct). A label with no group formula
    (`chemistry.ADDUCT_GROUPS`), or one the row's ion cannot be (it does not
    hold the group, or the charge differs: `channels.ion_adduct`, the test
    `channels.assign_adducts` makes again for an API caller), is refused,
    its row named."""
    df = pd.read_csv(os.path.expanduser(path), dtype=str, keep_default_na=False)
    if "adduct" not in df.columns or "formula" not in df.columns:
        return {}
    out = {}
    for i, r in df.iterrows():
        f, lab = str(r["formula"]).strip(), str(r["adduct"]).strip()
        # a missing value (R's NA for a missing string, 'nan', '#N/A':
        # `arbitrate.MISSING_CELLS`) is a blank cell: no adduct
        if not f or AR._missing(lab):
            continue
        try:
            C.adduct_group(lab)
            counts, charge = SF.parse_ion(f)
            CH.ion_adduct(SF.format_ion(counts, charge), lab)
        except ValueError as exc:
            raise ValueError(f"{path} row {i + 2}: {exc}") from None
        out[SF.format_ion(counts, charge)] = lab
    return out


def collect_ions(items: Iterable[tuple[str, float, float]]) -> dict[str, float]:
    """formula -> exact m/z, in the order given; a formula named twice raises."""
    out: dict[str, float] = {}
    for formula, _, exact in items:
        if formula in out:
            raise ValueError(f"ion {formula} is given twice")
        out[formula] = exact
    if not out:
        raise ValueError("no ions given: name the window ions with --ion MZ=FORMULA or --ions CSV")
    pols = {f[-1] for f in out}
    if len(pols) > 1:
        raise ValueError("ions of both polarities given")
    return out


# ---------------------------------------------------------------------------
# the run folder
# ---------------------------------------------------------------------------
RUN_FILES = ("ms2_spectra.parquet", "ms2_parents.csv", "ms2_window.csv", "ms2_manifest.json")
#: The per-scan tables `ms2 extract --file | --per-scan` adds (read when present).
PER_SCAN_FILES = ("ms2_scans.csv", "ms2_scan_peaks.parquet")
#: The run note of a run that reads product channels and binding classes.
CHANNELS_NOTE = ("product channels (ms2_channels.csv) are read against each ion's adduct; the "
                 "binding class (ms2_classes.csv) is a heuristic read off the lowest energies "
                 "and the energy the ion falls to half by (docs/MS2.md §11.13): it is only as "
                 "good as the named window ions -- an unnamed co-isolate's products are read "
                 "as theirs")
#: The run note of a folder without them.
NO_PER_SCAN_NOTE = ("no per-scan tables (ms2_scan_peaks.parquet; `ms2 extract --per-scan` or "
                    "`--file` writes them): no replicate statistics, no E50 interval")


@dataclass
class RunTables:
    spectra: pd.DataFrame
    parents: pd.DataFrame
    window: pd.DataFrame
    manifest: dict
    inputs: dict = field(default_factory=dict)       # file -> sha1
    notes: list = field(default_factory=list)
    #: the per-scan tables (`scans.SCAN_COLUMNS`, `scans.SCAN_PEAK_COLUMNS`);
    #: None when the folder has none
    scans: pd.DataFrame | None = None
    scan_peaks: pd.DataFrame | None = None


def _typed(df: pd.DataFrame, spec: dict[str, str]) -> pd.DataFrame:
    """A table read back from CSV / parquet in its contract's dtypes (an empty
    CSV cell is '' in a text column; a missing column is its type's blank)."""
    blank = {"str": "", "bool": False, "int64": 0, "float64": np.nan}
    out = {}
    for c, t in spec.items():
        if c not in df:
            out[c] = pd.Series([blank[t]] * len(df), dtype=t)
            continue
        s = df[c]
        if t == "str":
            s = s.fillna("").astype(str)
        elif t == "bool":
            s = s.map(lambda v: str(v).strip().lower() in ("true", "1")
                      if not isinstance(v, (bool, np.bool_)) else bool(v))
        out[c] = s.astype(t)
    return pd.DataFrame(out)


def load_run(run_dir: str) -> RunTables:
    """Read one `ms2 extract` sample folder. The spectra (parquet, else the CSV)
    and the parents table are required; the window table and the manifest are
    read when present (a missing one is noted)."""
    from peaky.reporting import provenance as PV
    d = Path(os.path.expanduser(run_dir))
    if not d.is_dir():
        raise FileNotFoundError(f"--run-dir {run_dir}: not a folder")
    notes, inputs = [], {}
    sp = d / "ms2_spectra.parquet"
    if sp.exists():
        spectra = pd.read_parquet(sp)
    elif (d / "ms2_spectra.csv").exists():
        sp = d / "ms2_spectra.csv"
        spectra = pd.read_csv(sp, keep_default_na=False, na_values=[""])
    else:
        raise FileNotFoundError(f"{d.name}: no ms2_spectra.parquet / .csv -- is it an "
                                "`ms2 extract` sample folder?")
    inputs[sp.name] = PV.sha1_file(str(sp))
    pp = d / "ms2_parents.csv"
    if not pp.exists():
        raise FileNotFoundError(f"{d.name}: no ms2_parents.csv")
    parents = pd.read_csv(pp, keep_default_na=False, na_values=[""])
    inputs[pp.name] = PV.sha1_file(str(pp))
    wp = d / "ms2_window.csv"
    if wp.exists():
        window = pd.read_csv(wp, keep_default_na=False, na_values=[""])
        inputs[wp.name] = PV.sha1_file(str(wp))
    else:
        window = pd.DataFrame()
        notes.append("no ms2_window.csv: shared fragments fall back from MS2 survivors to the "
                     "precursor where first seen, else an equal split, never to MS1 shares")
    mp = d / "ms2_manifest.json"
    manifest = {}
    if mp.exists():
        manifest = json.loads(mp.read_text(encoding="utf-8"))
        inputs[mp.name] = PV.sha1_file(str(mp))
    else:
        notes.append("no ms2_manifest.json: the isolation width comes from the parents table "
                     "or --iso-width")
    scans = scan_peaks = None
    kp = d / "ms2_scan_peaks.parquet"
    if kp.exists():
        scan_peaks = _typed(pd.read_parquet(kp), SC.SCAN_PEAK_COLUMNS)
        inputs[kp.name] = PV.sha1_file(str(kp))
        ks = d / "ms2_scans.csv"
        if ks.exists():
            scans = _typed(pd.read_csv(ks, keep_default_na=False, na_values=[""]),
                           SC.SCAN_COLUMNS)
            inputs[ks.name] = PV.sha1_file(str(ks))
        if not len(scan_peaks):
            notes.append("ms2_scan_peaks.parquet holds no row (the per-scan tables were skipped "
                         "at extract): no replicate statistics, no E50 interval")
            scan_peaks = None
    return RunTables(spectra=_typed(spectra, SRC.SPECTRA_COLUMNS),
                     parents=_typed(parents, SRC.PARENTS_COLUMNS),
                     window=_typed(window, SRC.WINDOW_COLUMNS), manifest=manifest,
                     inputs=inputs, notes=notes, scans=scans, scan_peaks=scan_peaks)


def isolation_width(tables: RunTables, explicit: float | None) -> tuple[float, str | None]:
    """(width, source): an explicit --iso-width; else the width `ms2 extract`
    built the window inventory with (its manifest); else the server's (the
    parents table); else (NaN, None)."""
    if explicit is not None:
        return float(explicit), "explicit"
    used = (tables.manifest.get("parameters") or {}).get("isolation_width_used")
    if used is not None and math.isfinite(float(used)) and float(used) > 0:
        return float(used), "extract manifest"
    w = tables.parents["isolation_width"].dropna()
    w = w[w > 0]
    if len(w):
        return float(w.iloc[0]), "server"
    return math.nan, None


#: `isolation_width`'s source when the parents table holds a different width
#: per parent and nothing replaces them: an `ms2 extract --file` folder whose
#: method isolated its precursors at different widths (the trailer's, per
#: parent). Each window is then read at its own parent's width.
PER_PARENT_WIDTH = "parents table, per parent"


def parent_widths(tables: RunTables) -> dict[float, float]:
    """Parent m/z (rounded to 1e-6) -> its own positive isolation width in the
    parents table (the largest when its rows differ)."""
    par = tables.parents[tables.parents["isolation_width"] > 0]
    return {round(float(p), 6): float(g["isolation_width"].max())
            for p, g in par.groupby("parent_mz")}


# ---------------------------------------------------------------------------
# one window
# ---------------------------------------------------------------------------
@dataclass
class WindowResult:
    parent_mz: float
    status: str                          # 'analysed' | 'readings only' | 'skipped'
    reason: str = ""                     # SKIP_* / NOT_RESOLVED when not analysed
    target: str = ""
    co_isolates: list = field(default_factory=list)
    left_out: list = field(default_factory=list)     # named ions too close to a window ion
    activation_kind: str = ""            # the one energy scale of the breakdown
    precursor_mz: float = math.nan
    offset_ppm: float = math.nan
    readings: pd.DataFrame | None = None
    roles: pd.DataFrame | None = None    # the breakdown's peaks, fragments merged across energies
    breakdown: pd.DataFrame | None = None
    e50: pd.DataFrame | None = None
    fragments: pd.DataFrame | None = None
    branching: pd.DataFrame | None = None
    replicates: pd.DataFrame | None = None   # `replicates.REPLICATE_COLUMNS` (per-scan tables)
    notes: list = field(default_factory=list)
    #: skipped for want of a named target, with the nearest named ion within
    #: the axis reach (`offset_hint`): the --mass-offset-ppm that matches it
    offset_hint_ppm: float = math.nan
    #: peaks read as harmonic ghosts, by order ({'2f': n, '3f': n}), over every
    #: spectrum of the window
    harmonics: dict = field(default_factory=dict)
    #: the breakdown's scans per energy (lowest NCE first; [] without
    #: per-scan tables), and the energies that add no spread: fewer than two
    #: scans (one, or no per-scan row), or every precursor / fragment peak held
    scans_per_energy: list = field(default_factory=list)
    one_scan_energies: int = 0
    #: `channels.CHANNEL_COLUMNS` / `channels.CLASS_COLUMNS` rows (None: no
    #: adduct given for any ion of the run)
    channels: pd.DataFrame | None = None
    classes: pd.DataFrame | None = None
    #: the rival group whose ion is this window's target ('' : none), and its
    #: `arbitrate.ARBITRATION_COLUMNS` rows (None: no rival group here)
    rival_group: str = ""
    arbitration: pd.DataFrame | None = None
    #: a rival group's window the tables skip, another rival matching its
    #: located precursor: what reads it, keyed on the verdict ('' : the
    #: axis offset `offset_hint_ppm` -- the arbitration supports the rival
    #: the tables read --, or no such window; `_rival_skip`)
    rival_hint: str = ""

    def manifest(self) -> dict:
        nce = ([] if self.breakdown is None or not len(self.breakdown) else
               sorted({float(x) for x in self.breakdown["nce"]}))
        volts = (self.breakdown is not None and len(self.breakdown)
                 and bool(np.isfinite(self.breakdown["hcd_v"]).all()))
        out = {"parent_mz": self.parent_mz, "status": self.status, "reason": self.reason,
               "target": self.target, "co_isolates": list(self.co_isolates),
               "left_out": list(self.left_out), "activation_kind": self.activation_kind,
               "precursor_mz": None if math.isnan(self.precursor_mz) else self.precursor_mz,
               "offset_ppm": None if math.isnan(self.offset_ppm) else round(self.offset_ppm, 4),
               "nce": nce, "volts_known": bool(volts), "notes": list(self.notes),
               "offset_hint_ppm": (None if math.isnan(self.offset_hint_ppm)
                                   else round(self.offset_hint_ppm, 2)),
               "harmonic_peaks": dict(self.harmonics),
               "scans_per_energy": list(self.scans_per_energy),
               "one_scan_energies": int(self.one_scan_energies)}
        if self.arbitration is not None:
            # a rival group's window (`arbitrate`): its verdict, the rival the
            # tables read, each rival's verdict
            a = self.arbitration
            out["rival_group"] = self.rival_group
            out["arbitration"] = {
                "verdict": str(a["window_verdict"].iloc[0]) if len(a) else "",
                "winner": str(a["winner"].iloc[0]) if len(a) else "",
                "analysed_with": next((str(r) for r, x in zip(a["rival"], a["analysed"]) if x),
                                      ""),
                "rivals": {str(r): str(v) for r, v in zip(a["rival"], a["verdict"])},
                "hint": self.rival_hint}
        return out


def folder_axis_offset(tables: RunTables, explicit: float | None = None
                       ) -> tuple[float, str | None]:
    """(ppm, source): the m/z axis offset the folder's located precursors and
    MS1 window are read through -- an explicit `--mass-offset-ppm`; else the
    one `ms2 extract --file` measured and recorded (its manifest's
    `parameters.ms2_axis_offset.ppm`); else 0 (a server folder: the server
    calibrates the spectra it serves)."""
    if explicit is not None:
        return float(explicit), "explicit"
    off = ((tables.manifest.get("parameters") or {}).get("ms2_axis_offset") or {}).get("ppm")
    if off is not None and math.isfinite(float(off)):
        return float(off), "extract manifest"
    return 0.0, None


def offset_hint(precursor_mz: float, ions: Mapping[str, float], axis_offset_ppm: float = 0.0
                ) -> float:
    """The --mass-offset-ppm that would match the named ion nearest the
    located precursor (read through the folder's axis offset, as
    `window_ions` reads it), when it is within the reach the file's own
    offset is read over (`sources.AXIS_SEARCH_PPM`); NaN beyond it (a named
    ion Da away is another ion, not this one off-axis) or with no named ion."""
    if not ions:
        return math.nan
    seen = precursor_mz / (1.0 + axis_offset_ppm * 1e-6)
    m = min(ions.values(), key=lambda v: abs(v - seen))
    need = (precursor_mz - m) / m * 1e6
    return float(need) if abs(need) <= SRC.AXIS_SEARCH_PPM else math.nan


def window_ions(parent_mz: float, precursor_mz: float, ions: Mapping[str, float],
                width: float, tol_mda: float, *, axis_offset_ppm: float = 0.0,
                target: str | None = None) -> tuple[str | None, list[str], str]:
    """(target, co-isolates by m/z, note). The target is the named ion nearest
    the located precursor -- read through the folder's m/z axis offset,
    precursor / (1 + `axis_offset_ppm` 1e-6) -- within max(`TARGET_MATCH_PPM`,
    tol_mda); the co-isolates are the other named ions inside +-(width/2 +
    `WINDOW_PAD_DA`) of the key -- none when the width is unknown (said). A
    window no named ion matches says how far the nearest one is and, when
    that is within `sources.AXIS_SEARCH_PPM` (the reach `ms2 extract` reads an
    axis offset over), the offset that would match it; a named ion farther
    away is taken for another ion, and the note just says to name this one.
    `target` (a named ion): the window's target is that one, matched by the
    caller (a rival of a group, `arbitrate`: each rival is read as the
    target in turn)."""
    seen = precursor_mz / (1.0 + axis_offset_ppm * 1e-6)
    tol = max(TARGET_MATCH_PPM * seen * 1e-6, tol_mda * 1e-3)
    if target is not None:
        near = [(abs(ions[target] - seen), target)]
    else:
        near = sorted(((abs(m - seen), f) for f, m in ions.items() if abs(m - seen) <= tol))
    if not near:
        f, m = min(ions.items(), key=lambda kv: abs(kv[1] - seen))
        # an axis offset is offered only within the reach the file's own offset
        # is read over (`offset_hint`): a named ion Da away is another ion, not
        # this one off-axis
        need = offset_hint(precursor_mz, ions, axis_offset_ppm)
        hint = ("; name it with --ion, or -- when the whole folder's m/z axis sits off its "
                f"formulas -- pass --mass-offset-ppm {need:+.1f}"
                if math.isfinite(need) else "; name it with --ion")
        return None, [], (f"skipped: no named ion matches the precursor located at "
                          f"{precursor_mz:.5f}"
                          + (f" ({seen:.5f} through the folder's axis offset of "
                             f"{axis_offset_ppm:+.1f} ppm)" if axis_offset_ppm else "")
                          + f" (nearest {f} at {m:.5f}, {(m - seen) * 1e3:+.1f} mDa = "
                          f"{(m - seen) / m * 1e6:+.1f} ppm)" + hint)
    target = near[0][1]
    if not (math.isfinite(width) and width > 0):
        return target, [], ("isolation width unknown (the server refused the summary and no "
                            "--iso-width was given): only the target is read in this window; "
                            "pass --iso-width W to include co-isolates")
    half = width / 2.0 + SRC.WINDOW_PAD_DA
    co = sorted((m, f) for f, m in ions.items() if f != target and abs(m - parent_mz) <= half)
    return target, [f for _, f in co], ""


def drop_aliases(target: str, co: Sequence[str], ions: Mapping[str, float], *,
                 tol_ppm: float, tol_mda: float) -> tuple[list[str], list[str], list[str]]:
    """(co-isolates kept, named ions left out, notes). Two window ions whose
    exact m/z lie within the reading tolerance of each other (the CHO / N^N
    or CO3 / ^NO3 aliases of a tied ledger row) read the SAME precursor peak
    and every fragment both can make: the window cannot tell them apart, and
    keeping both would give the peak to one and leave the other a phantom
    with I = 0 (its F 0) that still takes a share of every fragment. The
    target -- the named ion nearest the located precursor -- is kept; between
    two co-isolates, the one named first. The other is left out, said."""
    named = list(ions)
    kept, out, notes = [target], [], []
    for f in sorted(co, key=named.index):
        clash = [k for k in kept if abs(ions[f] - ions[k])
                 <= SF._tolerance_da(max(ions[f], ions[k]), tol_ppm, tol_mda)]
        if not clash:
            kept.append(f)
            continue
        k = clash[0]
        out.append(f)
        notes.append(f"{f} left out: {(ions[f] - ions[k]) * 1e3:+.2f} mDa from {k}, inside the "
                     "reading tolerance, so one peak reads as both and the window cannot tell "
                     f"them apart; kept {k} ("
                     + ("the target, nearest the located precursor" if k == target
                        else "named first") + "). Name only the one you mean")
    return [f for f in co if f in kept], out, notes


def inventory_half_width(tables: RunTables, parent_mz: float, *, bound: bool = True) -> float:
    """How far either side of `parent_mz` `ms2 extract` inventoried the MS1
    window (`ms2_window.csv`): half the width it built the inventory with plus
    its pad (its manifest); else half the server's width (the parents table)
    plus `WINDOW_PAD_DA`; else (`bound`) the farthest centroid the table holds
    for this parent, a lower bound; NaN when none of these is known."""
    p = tables.manifest.get("parameters") or {}
    used = p.get("isolation_width_used")
    pad = p.get("window_pad_da")
    pad = float(pad) if pad is not None and math.isfinite(float(pad)) else SRC.WINDOW_PAD_DA
    if used is not None and math.isfinite(float(used)) and float(used) > 0:
        return float(used) / 2.0 + pad
    par = tables.parents[np.isclose(tables.parents["parent_mz"], parent_mz, rtol=0, atol=1e-6)]
    w = par["isolation_width"].dropna()
    w = w[w > 0]
    if len(w):
        return float(w.iloc[0]) / 2.0 + SRC.WINDOW_PAD_DA
    if not bound:
        return math.nan
    win = tables.window
    win = win[np.isclose(win["parent_mz"], parent_mz, rtol=0, atol=1e-6)] if len(win) else win
    if not len(win):
        return math.nan
    return float(np.abs(win["mz"].to_numpy(dtype=float) - parent_mz).max())


def _ms1_shares(window: pd.DataFrame, parent_mz: float, ions: Mapping[str, float],
                tol_mda: float, half_width: float = math.nan, *,
                axis_offset_ppm: float = 0.0) -> dict[str, float]:
    """Each window ion's share of the MS1 isolation window: the centroid
    nearest it within max(`TARGET_MATCH_PPM`, tol_mda), one centroid to one
    ion only (the nearest pairs first: two ions never share one peak's
    share); the centroids' m/z are read through the folder's axis offset
    (m/z / (1 + `axis_offset_ppm` 1e-6)). A window ion no centroid matches has
    share 0 -- the MS1 window was measured and does not hold it -- when it
    lies inside the span `ms2 extract` inventoried (`half_width` either side
    of the key, `inventory_half_width`); beyond it, NaN: not measured, not 0
    (an analyze width wider than the extract's). {} when the window table has
    no row for this parent (unknown, not 0)."""
    w = window[np.isclose(window["parent_mz"], parent_mz, rtol=0, atol=1e-6)]
    if not len(w):
        return {}
    mz = w["mz"].to_numpy(dtype=float) / (1.0 + axis_offset_ppm * 1e-6)
    share = w["share"].to_numpy(dtype=float)
    pairs = sorted((abs(mz[i] - m), f, i) for f, m in ions.items() for i in range(len(mz))
                   if abs(mz[i] - m) <= max(TARGET_MATCH_PPM * m * 1e-6, tol_mda * 1e-3))
    out = {f: (math.nan if math.isfinite(half_width) and abs(m - parent_mz) > half_width
               else 0.0) for f, m in ions.items()}
    used_f, used_i = set(), set()
    for _, f, i in pairs:
        if f in used_f or i in used_i:
            continue
        out[f] = float(share[i])
        used_f.add(f)
        used_i.add(i)
    return out


def located_precursor(tables: RunTables, parent_mz: float) -> float:
    """The precursor `ms2 extract` located for a parent key, as
    `analyze_window` takes it: at the lowest energy of a spectrum with peaks
    where it was located (energy-resolved before energy-averaged); NaN when
    none was."""
    par = _window_parents(tables, parent_mz)[0]
    located = par["precursor_mz"].dropna()
    return float(located.iloc[0]) if len(located) else math.nan


def _window_parents(tables: RunTables, parent_mz: float) -> tuple[pd.DataFrame, int]:
    """(the parent's spectra with peaks, lowest NCE first and energy-averaged
    last; how many spectra without peaks were left out)."""
    par = tables.parents[np.isclose(tables.parents["parent_mz"], parent_mz, rtol=0, atol=1e-6)]
    n_empty = int((par["n_peaks"] <= 0).sum())
    par = par[par["n_peaks"] > 0]
    par = par.assign(_nan=par["nce"].isna()).sort_values(["_nan", "nce", "activation"],
                                                         kind="stable")
    return par, n_empty


def _spectrum_rows(tables: RunTables, par: pd.DataFrame, parent_mz: float
                   ) -> tuple[list, dict[tuple[str, int], float]]:
    """`readings.read_window`'s input for a parent's spectra `par`: (meta,
    peaks) per spectrum -- a peak's index its place among the spectrum's
    rows -- and (spectrum, peak) -> resolving power."""
    spec_rows = []
    resolution: dict[tuple[str, int], float] = {}       # (spectrum, peak) -> resolving power
    for _, r in par.iterrows():
        pk = tables.spectra[tables.spectra["spectrum"] == r["spectrum"]]
        meta = {"spectrum": r["spectrum"], "parent_mz": float(parent_mz), "nce": r["nce"],
                "hcd_v": r["hcd_v"]}
        # the S/N goes with the peaks: a harmonic band's reading near the
        # floor clears no place (`readings.harmonic_ions`)
        cols = [c for c in ("mz", "intensity", "signal_to_noise") if c in pk]
        spec_rows.append((meta, pk[cols].reset_index(drop=True)))
        if "resolution" in pk:
            resolution.update(((str(r["spectrum"]), i), float(v))
                              for i, v in enumerate(pk["resolution"].to_numpy(dtype=float)))
    return spec_rows, resolution


def analyze_window(tables: RunTables, parent_mz: float, ions: Mapping[str, float], *,
                   width: float, k_max: int | None = None, ladder: bool = True,
                   tol_ppm: float = 5.0, tol_mda: float = 0.3,
                   gas_mass: float = BD.N2_MASS, axis_offset_ppm: float = 0.0,
                   bootstrap: int = RP.BOOTSTRAP_B, harmonics: bool = True,
                   adducts: Mapping[str, "CH.IonAdduct | None"] | None = None,
                   target: str | None = None) -> WindowResult:
    """Everything for one parent key: its ions, its offset, the readings across
    its spectra and -- when it has energy-resolved spectra -- the breakdown,
    the E50s, the fragments and the branching. `axis_offset_ppm`: the
    folder's m/z axis offset (`folder_axis_offset`), through which the target
    is matched and the MS1 window read; the window's own offset (precursor
    against target) is then measured as on any folder. With the per-scan
    tables (`tables.scan_peaks`): the replicate statistics and, with
    `bootstrap` > 0 resamples, the E50 intervals (`_replicates`).
    `harmonics`: read harmonic ghosts as no formula (`readings.read_window`).
    `adducts` (named ion -> `channels.IonAdduct` or None): the window's
    product channels and binding classes (`_channels`); None reads none.
    `target` (a named ion): read the window with it as the target, matched
    by the caller (`window_ions`; a rival group's window, `arbitrate`)."""
    res = WindowResult(parent_mz=float(parent_mz), status="skipped")
    # an empty spectrum (ms2 extract writes a parent row with n_peaks 0 for a
    # spectrum the server returned without peaks) measured nothing: not a
    # point of any curve, so it is left out -- said
    par, n_empty = _window_parents(tables, parent_mz)
    if not len(par):
        res.reason = SKIP_NO_PEAKS
        res.notes.append("skipped: no spectrum with peaks")
        return res
    if n_empty:
        res.notes.append(f"{n_empty} spectrum(s) with no peaks left out")
    located = par["precursor_mz"].dropna()
    if not len(located):
        res.reason = SKIP_NOT_LOCATED
        res.notes.append("skipped: ms2 extract located no precursor near the key in any "
                         "spectrum of this parent")
        return res
    pmz = float(located.iloc[0])
    res.precursor_mz = pmz
    target, co, note = window_ions(parent_mz, pmz, ions, width, tol_mda,
                                   axis_offset_ppm=axis_offset_ppm, target=target)
    if note:
        res.notes.append(note)
    if target is None:
        res.reason = SKIP_NO_TARGET
        res.offset_hint_ppm = offset_hint(pmz, ions, axis_offset_ppm)
        return res
    co, res.left_out, alias_notes = drop_aliases(target, co, ions, tol_ppm=tol_ppm,
                                                 tol_mda=tol_mda)
    res.notes.extend(alias_notes)
    res.target, res.co_isolates = target, co
    order = [target] + co
    labels = {f: f for f in order}
    off = SF.mass_offset([(pmz, target)])
    res.offset_ppm = off.offset_ppm
    spec_rows, resolution = _spectrum_rows(tables, par, parent_mz)
    res.readings = RD.read_window(spec_rows, labels, offset_ppm=off.offset_ppm,
                                  tol_ppm=tol_ppm, tol_mda=tol_mda, k_max=k_max, ladder=ladder,
                                  harmonics=harmonics)
    ghosts = res.readings[res.readings["harmonic"].fillna("") != ""]
    if len(ghosts):
        by = ghosts["harmonic"].str.extract(r"^(\d+f)")[0].value_counts()
        res.harmonics = {k: int(by[k]) for k in sorted(by.index)}
        res.notes.append(f"{len(ghosts)} peak(s) are Orbitrap harmonic ghosts of a brighter line "
                         "of their spectrum ("
                         + ", ".join(f"{k}: {v}" for k, v in res.harmonics.items())
                         + "): read as no formula and kept out of the TIC (ms2_readings.csv "
                         "`harmonic`)")
    res.status = "readings only"
    resolved = par[par["nce"].notna()]
    if not len(resolved):
        res.reason = NOT_RESOLVED
        res.notes.append("readings only: no energy-resolved spectrum (an energy-averaged "
                         "spectrum has no energy axis)")
        return res
    if len(resolved) < len(par):
        res.notes.append(f"{len(par) - len(resolved)} energy-averaged spectrum(s) read but left "
                         "out of the breakdown")
    # one energy scale per breakdown: an NCE of HCD is not an NCE of CID
    kinds = resolved["kind"].fillna("").astype(str)
    n_kind = kinds.value_counts()
    if len(n_kind) > 1:
        kind = sorted(n_kind.index, key=lambda k: (-int(n_kind[k]), k != "hcd", k))[0]
        others = ", ".join(f"{int(n_kind[k])} {k or '(no kind)'}"
                           for k in sorted(n_kind.index) if k != kind)
        res.notes.append(f"spectra of {len(n_kind)} activation kinds: one breakdown per energy "
                         f"scale, so it uses the {int(n_kind[kind])} {kind} spectra; the "
                         f"other spectra ({others}) are read (ms2_readings.csv) but left out "
                         "of the breakdown")
        resolved = resolved[kinds == kind]
    res.activation_kind = str(resolved["kind"].fillna("").iloc[0])
    if resolved["nce"].duplicated().any():
        res.notes.append("several spectra share one NCE: the breakdown takes them in activation "
                         "order")
    specs = list(resolved["spectrum"])
    rd = res.readings[res.readings["spectrum"].isin(specs)]
    # one name per physical fragment across the energies (a peak whose rank-1
    # formula changes between energies is still one fragment)
    roles = AT.merge_fragment_peaks(AT.peak_roles(rd, order), tol_ppm=tol_ppm, tol_mda=tol_mda)
    # a fragment that is a side lobe of a brighter line in every spectrum it
    # is in is not an ion: its peaks stay in the TIC, unexplained -- said
    lobes = AT.satellite_fragments(roles, resolution)
    if lobes:
        roles = AT.set_aside_fragments(roles, lobes)
        res.notes.extend(f"side lobe: {t}" for t in lobes.values())
    res.roles = roles
    shared = roles[(roles["role"] == "precursor") & roles["candidates"].str.contains(";")]
    if len(shared):
        res.notes.append(f"{len(shared)} peak(s) read as the precursor of several window ions ("
                         + "; ".join(sorted(set(shared["candidates"])))
                         + "): each is given to the first of them in window order")
    ion_int, frag_int, ion_n = AT.intensity_matrices(roles, specs, order)
    budget = BD.ion_budget(roles).reindex(specs)
    tic = budget["tic"]
    # composition candidates of every fragment, in window order: the ions
    # that read it as a formula it can be at its signal-weighted m/z
    half = inventory_half_width(tables, parent_mz)
    shares = _ms1_shares(tables.window, parent_mz, {f: ions[f] for f in order}, tol_mda, half,
                         axis_offset_ppm=axis_offset_ppm)
    # the named ions this sample lacks (their precursor in no MS2 spectrum of
    # the window, an MS1 share of 0): they decide nothing
    absent = AT.absent_ions(order, ion_int.max().to_dict() if len(ion_int) else {}, shares)
    comps = AT.fragment_compositions(roles, tol_ppm=tol_ppm, tol_mda=tol_mda)
    cand, cand_notes = AT.candidates_across(roles, order, tol_ppm=tol_ppm, tol_mda=tol_mda,
                                            absent=absent)
    # where every candidate the sample holds reads each fragment: an absent
    # ion is not waited for (it is set aside before the intact test)
    att = AT.attribute_fragments(ion_int, frag_int, tic, cand, shares,
                                 shared=AT.shared_presence(roles, cand, specs, absent=absent))
    for f in order:
        # a window ion the lowest-energy spectrum does not hold: what the
        # split gave it, as given (`attribute.share_weights`), never a guess
        if ion_int[f].sum() > 0:
            if ion_int[f].iloc[0] <= 0:
                at = ion_int.index[int(np.argmax((ion_int[f] > 0).to_numpy()))]
                res.notes.append(f"{f}'s precursor is not in the lowest-energy MS2 spectrum, "
                                 f"first seen at {at}: {_given(att, f)}")
            continue
        if f not in co:
            continue
        if shares:
            if shares[f] == 0:
                res.notes.append(f"{f} is named inside the window, but no MS2 spectrum holds its "
                                 "precursor and the MS1 window holds no centroid at it: absent "
                                 "from this sample, it weighs 0 in every fragment it shares and "
                                 "decides none (set aside before the intact test; it takes no "
                                 "fragment by the polyoxy rule)")
            elif not math.isfinite(shares[f]):
                res.notes.append(f"{f} is named inside the window, but no MS2 spectrum holds its "
                                 f"precursor and it lies {ions[f] - parent_mz:+.3f} Da from the "
                                 "key, beyond the MS1 window ms2 extract inventoried (+-"
                                 f"{half:.3f} Da): its MS1 share is unknown, not 0; "
                                 f"{_given(att, f)}")
            else:
                res.notes.append(f"{f} is named inside the window, but no MS2 spectrum holds its "
                                 f"precursor; its MS1 window share is {shares[f]:.4g}: "
                                 f"{_given(att, f)}")
        else:
            res.notes.append(f"{f} is named inside the window, but no MS2 spectrum holds its "
                             f"precursor and there is no MS1 window table: {_given(att, f)}")
    weights = {r.key: r.weights for r in att.itertuples()}
    # a fragment only absent ions can make is no ion's: not in any curve
    # (F_lo counts every composition candidate, so it is taken off there too)
    nobody = set(att.loc[att["basis"] == AT.BASIS_ABSENT, "key"])
    # ... and an absent ion is in no curve at all: not even F_lo of a fragment
    # it shares with an ion the sample holds
    cand_c = {k: [c for c in v if c not in absent] for k, v in cand.items()}
    if nobody:
        # its peaks are unexplained, as without the name (the TIC is unchanged)
        budget = BD.ion_budget(AT.set_aside_fragments(roles, nobody)).reindex(specs)
    curves = BD.survival(ion_int, frag_int, weights, cand_c)
    br = AT.branching(ion_int, frag_int, tic, weights, cand_c)
    meta = resolved.set_index("spectrum")
    by = br.groupby(["spectrum", "ion"]) if len(br) else None
    bsum = by["branching"].sum(min_count=1) if by is not None else pd.Series(dtype=float)
    gsum = by["frag_gain"].sum(min_count=1) if by is not None else pd.Series(dtype=float)
    lost = by["ion_loss"].first() if by is not None else pd.Series(dtype=float)
    rows = []
    for spec in specs:
        for f in order:
            b = float(bsum.get((spec, f), math.nan)) if len(bsum) else math.nan
            g = float(gsum.get((spec, f), math.nan)) if len(gsum) else math.nan
            ls = float(lost.get((spec, f), math.nan)) if len(lost) else math.nan
            rows.append({
                "parent_mz": float(parent_mz), "spectrum": spec, "nce": meta.loc[spec, "nce"],
                "hcd_v": meta.loc[spec, "hcd_v"], "ion": f,
                "role": "target" if f == target else "co-isolate", "ion_mz": ions[f],
                "intensity": ion_int.loc[spec, f], "n_precursor_peaks": ion_n.loc[spec, f],
                "frac": (ion_int.loc[spec, f] / tic[spec]) if tic[spec] > 0 else math.nan,
                "frag_weighted": curves["frag_w"].loc[spec, f],
                "frag_all": curves["frag_all"].loc[spec, f],
                "f": curves["f"].loc[spec, f], "f_lo": curves["f_lo"].loc[spec, f],
                "s": curves["s"].loc[spec, f], "branching_sum": b,
                "branching_flag": AT.branching_breach(g, ls),
                "tic": tic[spec], "budget_explained": budget.loc[spec, "budget_explained"],
                "budget_unexplained": budget.loc[spec, "budget_unexplained"],
                "n_peaks": budget.loc[spec, "n_peaks"],
                "n_unexplained": budget.loc[spec, "n_unexplained"],
                "n_harmonic": budget.loc[spec, "n_harmonic"],
                "harmonic_intensity": budget.loc[spec, "harmonic_intensity"],
                "n_scans": 0, "intensity_cv": math.nan, "f_sd": math.nan, "f_sem": math.nan,
                "f_lo_sd": math.nan, "f_lo_sem": math.nan, "s_sd": math.nan,
                "s_sem": math.nan})
    rep = None
    if tables.scan_peaks is not None:
        rep = _replicates(tables, float(parent_mz), specs, meta, roles, order,
                          list(frag_int.columns), weights, cand_c, ion_int,
                          {m["spectrum"]: pk for m, pk in spec_rows}, bootstrap=bootstrap)
        res.scans_per_energy = [int(rep["n_scans"][sp]) for sp in specs]
        # the energies that add no spread: fewer than two scans, or every
        # precursor / fragment peak held (no scan's reading used)
        res.one_scan_energies = sum(n <= 1 or sp in rep["held"]
                                    for n, sp in zip(res.scans_per_energy, specs))
        res.notes.extend(rep["notes"])
        res.replicates = rep["table"]
        for r in rows:
            r.update(rep["breakdown"].get((r["spectrum"], r["ion"]), {}))
    res.breakdown = _frame(rows, BREAKDOWN_COLUMNS)
    flagged = res.breakdown[res.breakdown["branching_flag"]]
    for ion, g in flagged.groupby("ion", sort=False):
        res.notes.append(_breach_note(ion, g, gsum, lost, budget))
    res.e50 = _e50_rows(res.breakdown, order, target, ions, gas_mass,
                        boot=None if rep is None else rep["boot"],
                        n_one=res.one_scan_energies, boot_n=0 if rep is None else rep["boot_n"],
                        per_scan=rep is not None)
    res.fragments = _fragment_rows(parent_mz, att, roles, frag_int, tic, meta, cand_notes,
                                   {k: v[1] for k, v in comps.items()},
                                   None if rep is None else rep["items"])
    br = br.assign(parent_mz=float(parent_mz), fragment=br["key"],
                   nce=[meta.loc[s, "nce"] for s in br["spectrum"]],
                   hcd_v=[meta.loc[s, "hcd_v"] for s in br["spectrum"]])
    res.branching = _frame(br, BRANCHING_COLUMNS)
    if adducts is not None:
        _channels(res, adducts, tables, specs, meta, order, ions, roles, comps, ion_int,
                  frag_int, weights, shares, rep, {m["spectrum"]: pk for m, pk in spec_rows},
                  gas_mass=gas_mass, tol_mda=tol_mda, axis_offset_ppm=axis_offset_ppm,
                  absent=absent, sources=dict(zip(att["key"], att["weight_source"])))
    res.status = "analysed"
    return res


def _channels(res: WindowResult, adducts: Mapping[str, "CH.IonAdduct | None"],
              tables: RunTables, specs: Sequence[str], meta: pd.DataFrame,
              order: Sequence[str], ions: Mapping[str, float], roles: pd.DataFrame,
              comps: Mapping, ion_int: pd.DataFrame, frag_int: pd.DataFrame,
              weights: Mapping[str, Mapping[str, float]], shares: Mapping[str, float],
              rep: dict | None, peaks: Mapping[str, pd.DataFrame], *, gas_mass: float,
              tol_mda: float, axis_offset_ppm: float, absent: Sequence[str] = (),
              sources: Mapping[str, str] | None = None) -> None:
    """One analysed window's product channels and binding classes
    (`channels`): `res.channels`, `res.classes`. The class reads the
    window's F and S E50s (`res.e50`); the m/z below which a reagent ion is
    unobserved -- the scans' recorded first mass (`ms2_scans.csv` `low_mass`
    of the window's spectra), else `channels.FIRST_MASS_MIN`, or a lower
    peak the spectra hold; whether the ion was observed (absent from the
    sample, `attribute.absent_ions`; or never seen as a precursor and given
    no product mostly its own, `channels.OWN_WEIGHT` -- a weight of an equal
    split, `sources` (fragment key -> `weight_source`) 'equal', is a prior and
    counts for none --, or whose products mostly its own hold at most
    `OWN_WEIGHT` of what it is given in its class window (its lowest energy):
    per spectrum, the weighted intensity of the products mostly its own --
    none of an equal split, and none it reads as 'another ion's product'
    (another ion's whatever its weight) --, which `classify` reads there);
    and, per ion, whether
    an MS1 centroid no named ion matches holds more of the isolation window
    than the ion itself (its products are then read as the named ions':
    said) -- or the ion has no MS1 share to weigh (not seen there, or no MS1
    window rows: said)."""
    forms = CH.fragment_formulas(roles, specs, {k: v[1] for k, v in comps.items()})
    res.channels = CH.window_channels(
        res.parent_mz, specs, meta, order, ions, res.target, ion_int, frag_int, weights,
        forms, {f: adducts.get(f) for f in order}, gas_mass=gas_mass,
        scans=(None if rep is None else
               {s: (rep["mats"][s][1], rep["n_scans"][s], rep["mats"][s][5]) for s in specs}))
    floor, what = _first_mass(tables, specs, peaks)
    unnamed = _unnamed_ms1(tables.window, res.parent_mz, {f: ions[f] for f in order}, tol_mda,
                           axis_offset_ppm=axis_offset_ppm)
    rows = []
    for f in order:
        presence = ""
        mostly = None
        if f in absent:
            presence = ("absent from this sample: no MS2 spectrum holds its precursor and the MS1 "
                        "window no centroid at it -- it decides nothing (no class)")
        elif float(ion_int[f].sum()) <= 0:
            given = [(float((weights.get(k) or {}).get(f, 0.0)), (sources or {}).get(k, ""))
                     for k in frag_int.columns if float(frag_int[k].sum()) > 0]
            given = [(w, src) for w, src in given if w > 0]
            # an equal split is a prior -- nothing is known of the ion's
            # share --, no observation that a product is its own
            measured = [w for w, src in given if src != "equal"]
            own = max(measured, default=0.0)
            # per spectrum, the weighted intensity of the products mostly its
            # own: the class reads them in its window (`channels.classify`).
            # A product it reads as another ion's is no product of its own,
            # whatever the weight it is given
            ch = res.channels
            theirs = {k for t in ch.loc[(ch["ion"] == f) & (ch["channel"] == CH.OTHER_ION),
                                        "products"]
                      for k in str(t).split("; ") if k}
            mine = [k for k in frag_int.columns
                    if float((weights.get(k) or {}).get(f, 0.0)) > CH.OWN_WEIGHT
                    and (sources or {}).get(k, "") != "equal" and k not in theirs]
            mostly = {s: sum(float(weights[k][f]) * float(frag_int.loc[s, k]) for k in mine)
                      for s in specs}
            if not given:
                presence = ("never seen as a precursor in the window, and given no product: it "
                            "decides nothing (no class)")
            elif not measured:
                presence = ("never seen as a precursor in the window, and every product it is "
                            "given is an equal split (weight_source 'equal': nothing is known of "
                            "its share -- a prior, no observation): its shares are another "
                            "ion's fragments read through its formula (no class)")
            elif own <= CH.OWN_WEIGHT:
                presence = (f"never seen as a precursor in the window, and no product it is "
                            f"given is mostly its own (its largest weight {own:.3g}, at most "
                            f"{CH.OWN_WEIGHT:g}): its shares are slivers of another ion's "
                            "fragments read through its formula (no class)")
        r = CH.classify(res.channels[res.channels["ion"] == f], res.e50[res.e50["ion"] == f],
                        adducts.get(f), floor_mz=floor, floor_what=what, ion_mz=ions[f],
                        presence=presence, curves=res.breakdown[res.breakdown["ion"] == f],
                        own_products=mostly)
        own = float(shares.get(f, math.nan)) if shares else math.nan
        said = ""
        if not shares:
            said = ("the MS1 window table holds no row for this key: whether an unnamed "
                    "co-isolate shares the isolation window is unknown")
        elif own == 0:
            said = (f"{f} is not seen in the MS1 window (no centroid at it): its share cannot "
                    "be weighed against an unnamed co-isolate's")
        elif unnamed is not None and math.isfinite(own) and unnamed[1] > own:
            said = (f"an MS1 centroid no named ion matches, at m/z {unnamed[0]:.4f}, holds "
                    f"{unnamed[1]:.0%} of the isolation window, more than {f} ({own:.0%}): its "
                    "products may be read as the named ions' -- name it (--ion): a fragment "
                    "both can make is then split by their survivor weights, and one only it "
                    "explains reads as another ion's product")
        if said:
            r["note"] = "; ".join(t for t in (r["note"], said) if t)
        rows.append(r)
    res.classes = CH._frame(rows, CH.CLASS_COLUMNS)


def _first_mass(tables: RunTables, specs: Sequence[str],
                peaks: Mapping[str, pd.DataFrame]) -> tuple[float, str]:
    """(m/z, what) below which a reagent ion of the window is unobserved: the
    lowest first mass the per-scan table records for the window's spectra
    (`low_mass`), else `channels.FIRST_MASS_MIN` (no MS2 scan of ours starts
    lower) -- or a lower peak the spectra hold."""
    low = math.nan
    sc = tables.scans
    if sc is not None and len(sc) and "low_mass" in sc:
        v = sc.loc[sc["spectrum"].isin(list(specs)), "low_mass"].to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        if len(v):
            low = float(v.min())
    if math.isfinite(low):
        floor, what = low, "the scans' first mass"
    else:
        floor, what = CH.FIRST_MASS_MIN, ("the lowest first mass an MS2 scan is set to "
                                          "(FIRST_MASS_MIN)")
    peak = min((float(peaks[s]["mz"].min()) for s in specs if len(peaks[s])), default=math.nan)
    if math.isfinite(peak) and peak < floor:
        floor, what = peak, "the lowest peak the spectra hold"
    return floor, what


def _unnamed_ms1(window: pd.DataFrame, parent_mz: float, ions: Mapping[str, float],
                 tol_mda: float, *, axis_offset_ppm: float = 0.0,
                 min_sn: float = -math.inf) -> tuple[float, float] | None:
    """(m/z, share) of the largest MS1 centroid of the window (`ms2_window.csv`)
    that no named window ion matches -- within max(`TARGET_MATCH_PPM`,
    `tol_mda`) of none of them, read through the folder's axis offset, as
    `_ms1_shares` reads them, and its m/z given so (the one to name with
    --ion) -- of S/N `min_sn` or more (a NaN S/N passes; none by default);
    None with no window table or no such centroid."""
    if not len(window):
        return None
    w = window[np.isclose(window["parent_mz"], parent_mz, rtol=0, atol=1e-6)]
    if not len(w):
        return None
    mz = w["mz"].to_numpy(dtype=float) / (1.0 + axis_offset_ppm * 1e-6)
    share = w["share"].to_numpy(dtype=float)
    named = np.zeros(len(mz), dtype=bool)
    for m in ions.values():
        named |= np.abs(mz - m) <= max(TARGET_MATCH_PPM * m * 1e-6, tol_mda * 1e-3)
    if math.isfinite(min_sn) and "signal_to_noise" in w:
        sn = w["signal_to_noise"].to_numpy(dtype=float)
        named |= np.isfinite(sn) & (sn < min_sn)          # noise: no ion to name
    if named.all():
        return None
    i = int(np.flatnonzero(~named)[np.argmax(share[~named])])
    return float(mz[i]), float(share[i])


def _replicates(tables: RunTables, parent_mz: float, specs: Sequence[str], meta: pd.DataFrame,
                roles: pd.DataFrame, order: Sequence[str], keys: Sequence[str],
                weights: Mapping[str, Mapping[str, float]],
                cand: Mapping[str, Sequence[str]], ion_int: pd.DataFrame,
                peaks: Mapping[str, pd.DataFrame], *, bootstrap: int) -> dict:
    """The replicate statistics of one window's breakdown (`replicates`):
    the averaged analysis's roles (`roles`, side lobes and harmonics set
    aside), weights and composition candidates re-read in every scan of
    every spectrum (`specs`, lowest energy first; `peaks` each spectrum's
    peak list as read). A held peak (`replicates.spectrum_scans`: read in no
    scan, or on another scale) reads its averaged height in every scan, so a
    curve only held peaks enter scatters by nothing anyone measured: S has
    no spread where its precursor is held, F (F_lo) none where something
    entering it is held and nothing entering it -- the precursor, a
    weighted (a composition-candidate) fragment -- is read. A spectrum of
    two or more scans whose every precursor / fragment peak is held (`held`)
    adds no spread, as one of one scan: the bootstrap draws its averaged
    spectrum. -> {'n_scans': {spectrum: n}, 'held': set of spectra,
    'breakdown': {(spectrum, ion): columns}, 'table': `REPLICATE_COLUMNS`
    rows, 'items': {(spectrum, fragment): (presence, cv)}, 'boot': {(ion,
    curve): `replicates.interval`} or None, 'boot_n', 'notes'}."""
    sp = tables.scan_peaks
    sel = sp[sp["spectrum"].isin(list(specs))]
    by = {str(k): g for k, g in sel.groupby("spectrum", sort=False)}
    notes = []
    sc = {spec: RP.spectrum_scans(spec, peaks[spec]["mz"], peaks[spec]["intensity"],
                                  by.get(spec, sel.iloc[0:0])) for spec in specs}
    n_scans = {spec: len(sc[spec].scans) for spec in specs}
    nce = meta.loc[list(specs), "nce"].to_numpy(dtype=float)
    volts = meta.loc[list(specs), "hcd_v"].to_numpy(dtype=float)
    none = [f"{x:g}" for spec, x in zip(specs, nce) if not n_scans[spec]]
    if none:
        notes.append(f"replicates: no per-scan row for the spectra at NCE {', '.join(none)}: held "
                     "at their averaged spectrum (no spread)")
    if tables.scans is not None and len(tables.scans):
        named = tables.scans["spectrum"].value_counts()
        off = [f"{x:g} ({n_scans[spec]} read, {int(named.get(spec, 0))} in ms2_scans.csv)"
               for spec, x in zip(specs, nce) if n_scans[spec]
               and int(named.get(spec, 0)) not in (0, n_scans[spec])]
        if off:
            notes.append("replicates: the per-scan peaks and the scan table name different scans "
                         "for the spectra at NCE " + ", ".join(off))
    used = roles["role"].isin(["precursor", "fragment"])
    unread, rescaled, held = 0, [], set()
    for spec in specs:
        if not n_scans[spec]:
            continue
        p = roles.loc[used & (roles["spectrum"] == spec), "peak"].to_numpy(dtype=int)
        out_p = ~sc[spec].matched[p]                    # held: no scan's reading used
        fac = sc[spec].factor[p]
        unread += int((out_p & ~np.isfinite(fac)).sum())
        rescaled.extend(fac[out_p & np.isfinite(fac)].tolist())
        if len(p) and out_p.all() and n_scans[spec] >= 2:
            held.add(spec)
    if unread:
        notes.append(f"replicates: {unread} precursor / fragment peak(s) are read in no scan of "
                     "their spectrum: held at their averaged height in every scan (their spread "
                     "is not in the statistics; presence and CV NaN)")
    if rescaled:
        lo_f, hi_f = RP.SCALE_BAND
        span = (f"{rescaled[0]:.3g}" if len(rescaled) == 1
                else f"{min(rescaled):.3g} to {max(rescaled):.3g}")
        notes.append(f"replicates: {len(rescaled)} precursor / fragment peak(s) are read in the "
                     f"scans on another scale than their spectrum's (averaged height / per-scan "
                     f"mean {span}, outside {lo_f:g}-{hi_f:g}: a stray centroid in a scan or "
                     "two, or per-scan rows off another m/z axis than the spectra's?): held at "
                     "their averaged height in every scan (their spread is not in the "
                     "statistics; presence and CV NaN)")
    # (with --bootstrap 0 there is no E50 interval to add nothing to)
    tail = ", nothing to the E50 interval" if bootstrap > 0 else ""
    one = [f"{x:g}" for spec, x in zip(specs, nce) if n_scans[spec] == 1]
    if one:
        notes.append(f"replicates: {len(one)} of {len(specs)} energies hold one scan (NCE "
                     f"{', '.join(one)}): no spread there{tail}")
    gone = [f"{x:g}" for spec, x in zip(specs, nce) if spec in held]
    if gone:
        notes.append("replicates: every precursor / fragment peak of the spectra at NCE "
                     f"{', '.join(gone)} is held: no spread there{tail} "
                     "(counted with the energies of one scan)")
    mats = RP.window_scans(roles, sc, specs, order, keys)
    w, c = RP._weights_matrix(weights, cand, order, keys)
    ref = ion_int.iloc[0].reindex(list(order)).to_numpy(dtype=float)
    bd, items, rows = {}, {}, []
    for spec, x, v in zip(specs, nce, volts):
        im, fm, ip, fp, ok_i, ok_f = mats[spec]
        n = n_scans[spec]
        cv_ = RP.spectrum_curves(im, fm, w, c, ref)
        sds = {k: RP.spread(a)[:2] for k, a in cv_.items()}
        isd = RP.spread(im)[0]
        imean = im.mean(axis=1)
        fsd = RP.spread(fm)[0] if len(fm) else np.zeros(0)
        fmean = fm.mean(axis=1) if len(fm) else np.zeros(0)
        with np.errstate(invalid="ignore", divide="ignore"):
            icv = np.where(imean > 0, isd / imean, np.nan)
            fcv = np.where(fmean > 0, fsd / fmean, np.nan)
        # the items in this spectrum that a scan reads, and that are held
        f_read, f_held = (fmean > 0) & ok_f, (fmean > 0) & ~ok_f
        for i, f in enumerate(order):
            real = n >= 1
            i_read, i_held = imean[i] > 0 and ok_i[i], imean[i] > 0 and not ok_i[i]
            # a curve's spread where something entering it is read, or
            # nothing entering it is held (absent items read 0 in every scan)
            has = {"s": not i_held}
            for k, enter in (("f", w[:, i] > 0), ("f_lo", c[:, i] > 0)):
                has[k] = (i_read or bool((f_read & enter).any())
                          or not (i_held or bool((f_held & enter).any())))
            bd[(spec, f)] = {
                "n_scans": n,
                "intensity_cv": float(icv[i]) if real and ok_i[i] else math.nan,
                **{f"{k}_{m}": (float(sds[k][j][i]) if real and has[k] else math.nan)
                   for k in ("f", "f_lo", "s") for j, m in enumerate(("sd", "sem"))}}
            pres = float(ip[i].mean()) if real and ok_i[i] else math.nan
            # unmatched (a peak of it held): n_present 0 as presence is NaN --
            # never the scans of its other, read peak beside a NaN presence
            rows.append({"parent_mz": parent_mz, "spectrum": spec, "nce": x, "hcd_v": v,
                         "item": "precursor", "name": f, "n_scans": n,
                         "n_present": int(ip[i].sum()) if real and ok_i[i] else 0,
                         "presence": pres,
                         "mean": float(imean[i]),
                         "sd": float(isd[i]) if real and ok_i[i] else math.nan,
                         "cv": float(icv[i]) if real and ok_i[i] else math.nan,
                         "matched": bool(real and ok_i[i])})
        for k, key in enumerate(keys):
            if not fmean[k] > 0:
                continue
            real = n >= 1 and bool(ok_f[k])
            pres = float(fp[k].mean()) if real else math.nan
            items[(spec, key)] = (pres, float(fcv[k]) if real else math.nan)
            rows.append({"parent_mz": parent_mz, "spectrum": spec, "nce": x, "hcd_v": v,
                         "item": "fragment", "name": key, "n_scans": n,
                         "n_present": int(fp[k].sum()) if real else 0, "presence": pres,
                         "mean": float(fmean[k]), "sd": float(fsd[k]) if real else math.nan,
                         "cv": float(fcv[k]) if real else math.nan, "matched": real})
    boot, b_used = None, 0
    if bootstrap > 0 and any(n_scans[s] >= 2 and s not in held for s in specs):
        # a held spectrum's scans are all its averaged spectrum: drawn as one
        # (as a spectrum with no per-scan row), exactly that
        per = [(mats[s][0][:, :1], mats[s][1][:, :1]) if s in held else mats[s][:2]
               for s in specs]
        out = RP.bootstrap_e50(nce, volts, per, w, c, b=int(bootstrap),
                               seed=RP.window_seed(parent_mz))
        boot = {(f, name): RP.interval(out[col][0][i], out[col][1][i], out[col][2][i],
                                       bracket=tuple(a[i] for a in out[col][3]))
                for col, name in CURVES for i, f in enumerate(order)}
        b_used = int(bootstrap)
    elif bootstrap > 0:
        notes.append("replicates: no energy holds two scans" + (" read" if held else "")
                     + ": no E50 interval")
    return {"n_scans": n_scans, "held": held, "breakdown": bd, "items": items, "boot": boot,
            "boot_n": b_used, "table": _frame(rows, RP.REPLICATE_COLUMNS), "notes": notes,
            "mats": mats}


def _given(att: pd.DataFrame, ion: str) -> str:
    """What the attribution gave `ion` of each fragment it can make
    (`AT.attribute_fragments` rows): its weight (3 significant figures;
    `ms2_fragments.csv` holds 6) and the `weight_source`: 'it is given C7H11+
    0.5 (equal), ...', or 'it reads no fragment'."""
    got = [f"{r.key} {float(r.weights.get(ion, 0.0)):.3g} ({r.weight_source})"
           for r in att.itertuples() if ion in str(r.candidates).split(";")]
    return f"it is given {', '.join(got)}" if got else "it reads no fragment"


def _breach_note(ion: str, g: pd.DataFrame, gsum: pd.Series, lost: pd.Series,
                 budget: pd.DataFrame) -> str:
    """The window note for an ion's flagged rows (`AT.branching_breach`):
    the sums where it lost something, and -- where it lost nothing, and the
    ratio is NaN -- its products' gain against its own change. Then, for
    each of those steps, in the order named and each labelled with its NCE:
    how far the unexplained share of the TIC moved from the lowest energy,
    and the TIC against the lowest energy's. Every check here reads TIC
    fractions of OBSERVED peaks: a product outside the observed m/z range
    (below the scan's low-mass limit, as the small ion an adduct releases)
    leaves the TIC, which falls, and inflates every fraction that stays --
    with no trace in the unexplained share."""
    fin = [(s, b, e) for s, b, e in zip(g["spectrum"], g["branching_sum"], g["nce"])
           if math.isfinite(b)]
    nan = [(s, e) for s, b, e in zip(g["spectrum"], g["branching_sum"], g["nce"])
           if not math.isfinite(b)]
    parts = []
    if fin:
        parts.append(f"branching of {ion} sums to "
                     + ", ".join(f"{b:.3f} at NCE {e:g}" for _, b, e in fin)
                     + f" (> {AT.BRANCHING_FLAG})")
    if nan:
        # (+ 0.0: a loss of exactly 0 prints as +0.0000, not -0.0000)
        at = ", ".join(f"NCE {e:g} (its products {gsum[(s, ion)]:+.4f} of the TIC, its own "
                       f"fraction {-lost[(s, ion)] + 0.0:+.4f})" for s, e in nan)
        parts.append((f"branching of {ion} has no ratio" if not fin else "no ratio")
                     + f" at {at}: it lost nothing there, yet its products grew")
    steps = [(s, e) for s, _, e in fin] + nan            # the order they are named in
    u0 = float(budget["budget_unexplained"].iloc[0])
    t0 = float(budget["tic"].iloc[0])
    moved = ", ".join(f"{float(budget.loc[s, 'budget_unexplained']) - u0 + 0.0:+.4f} at NCE {e:g}"
                      for s, e in steps)
    tic = ", ".join(f"{float(budget.loc[s, 'tic']) / t0:.3f} at NCE {e:g}" if t0 > 0 else
                    f"n/a at NCE {e:g}" for s, e in steps)
    return ("; ".join(parts)
            + ". Its products' TIC fractions grew by more than its own fell. Three causes: a "
            "fragment is mis-attributed or a share weight is off; ions not named here (the "
            "precursors' own side-lobe centroids among them) lost signal; or products left the "
            "observed m/z range (below the scan's low-mass limit, as the small ion an adduct "
            "releases) -- the last two take signal out of what is named, and TIC normalisation "
            "inflates every fraction that stays. The unexplained share of the TIC moved by "
            + moved + " from the lowest energy; the TIC was " + tic + " of the lowest "
            "energy's (a TIC that falls while the unexplained share holds says products left "
            "the observed range, or the precursor supply fell)")


def _volts_note(hcd_v: pd.Series, e: BD.E50) -> str:
    """'' when every spectrum's volts are known; 'volts unknown' when none
    is (a refused summary); else how many are -- and, when the crossing is
    interpolated but a spectrum bracketing it has no volts, that the volts
    E50 is the one missing (an E50 on volts needs only its two brackets)."""
    known = int(np.isfinite(hcd_v.to_numpy(dtype=float)).sum())
    n = len(hcd_v)
    if known == n:
        return ""
    if known == 0:
        return "volts unknown"
    if e.flag == BD.FLAG_INTERPOLATED and not math.isfinite(e.volts):
        return f"volts unknown at the crossing (known for {known} of {n} spectra)"
    return f"volts known for {known} of {n} spectra"


#: Where a NaN bound of the E50 interval lies (`replicates.interval`).
_BOUND_WHERE = {"below": "lies below the lowest energy", "above": "lies above the highest energy",
                "gap": "lies across an unobserved energy"}


def _boot_note(b: dict, n: int, e: BD.E50) -> str:
    """The `note` of an E50 row's bootstrap interval, when the resamples
    disagree with the averaged curve -- an E50 some resamples do not cross,
    or a curve with no E50 some of them do (or cross across an unobserved
    energy the averaged curve does not): how the resamples read -- the
    shares add up to 100 % -- and on which side a NaN bound lies. '' when
    they all agree."""
    crossed = e.flag == BD.FLAG_INTERPOLATED
    moved = b["cross"] + (b["gap"] if e.flag != BD.FLAG_GAP else 0.0)
    if (crossed and b["cross"] >= 1.0) or (not crossed and moved <= 0.0):
        return ""
    shares = [f"below the lowest energy in {b['below']:.1%}",
              f"above the highest in {b['above']:.1%}"]
    if b["gap"] > 0:
        shares.append(f"across an unobserved energy in {b['gap']:.1%}")
    if b["none"] > 0:
        shares.append(f"with one energy observed or none in {b['none']:.1%}")
    parts = [f"bootstrap ({n}): crosses in {b['cross']:.1%} of resamples ({', '.join(shares)})"]
    for side, word in (("lo", "lower"), ("hi", "upper")):
        if b[f"{side}_where"]:
            parts.append(f"the interval's {word} bound {_BOUND_WHERE[b[f'{side}_where']]}")
    return "; ".join(parts)


def _has_spread(g: pd.DataFrame, col: str) -> np.ndarray:
    """Per breakdown row: does the curve `col` have a spread the bootstrap
    draws there? F and F_lo (ratios within one spectrum): their `<col>_sd`
    finite -- none with fewer than two scans, where only held peaks enter
    it, or where fewer than two scans observe it (one scan alone holding
    the ion reads the same ratio however often it is drawn). S (the
    precursor over the lowest energy's): also where the precursor itself
    has a CV (`intensity_cv` finite: two scans or more, matched, read in
    one of them at least) -- a resample's S is its mean precursor, empty
    scans 0, so it moves with the draws even where one scan alone reads the
    precursor and the per-scan S has one finite value."""
    def col_of(name):
        return (g[name].to_numpy(dtype=float) if name in g else np.full(len(g), np.nan))
    has = np.isfinite(col_of(f"{col}_sd"))
    if col == "s":
        has = has | np.isfinite(col_of("intensity_cv"))
    return has


def _crossing_scans(g: pd.DataFrame, col: str) -> list[tuple[float, int, bool]]:
    """The two energies an averaged curve's first downward crossing lies
    between (an interpolated one, or one across an unobserved energy), with
    each one's scans (`n_scans`) and whether the curve has a spread there
    (`_has_spread`); [] where the curve does not cross."""
    _, _, code, (xa, xb, _, _) = BD.e50_many(g["nce"], g[col].to_numpy(dtype=float)[:, None],
                                            bracket=True)
    if BD.FLAGS[int(code[0])] not in (BD.FLAG_INTERPOLATED, BD.FLAG_GAP):
        return []
    nce = g["nce"].to_numpy(dtype=float)
    has = _has_spread(g, col)
    out = []
    for x in (xa[0], xb[0]):
        j = int(np.flatnonzero(np.isclose(nce, x))[0])
        out.append((float(x), int(g["n_scans"].iloc[j]), bool(has[j])))
    return out


def _reference_scans(g: pd.DataFrame) -> tuple[float, int, bool]:
    """S's reference: the breakdown's lowest energy (its first row; every
    resample's S is read against its own mean precursor there,
    `bootstrap_e50`), its scans and whether that mean has a spread there
    (`_has_spread`: the precursor's `intensity_cv` finite -- none with
    fewer than two scans, a held precursor, or no per-scan row; one read in
    one of its scans alone has one)."""
    return float(g["nce"].iloc[0]), int(g["n_scans"].iloc[0]), bool(_has_spread(g, "s")[0])


def _bracket_note(thin: Sequence[tuple], zero_width: bool) -> str:
    """The `note` of an E50 band next to an energy that adds no spread
    (`thin`: (NCE, scans, is S's reference) of those of the two about the
    averaged crossing -- and, for S, of the lowest energy it is read against
    -- with fewer than two scans, or more with no spread of the curve there
    (`_has_spread`): 'no spread'), or of zero width: the band holds no scatter of theirs -- it
    is too narrow, and of zero width it is no measurement at all."""
    def name(x, k):
        return f"{x:g} ({k} scan{'' if k == 1 else 's'}{', no spread' if k >= 2 else ''})"
    parts = []
    near = [(x, k) for x, k, ref in thin if not ref]
    if near:
        names = " and ".join(name(x, k) for x, k in near)
        parts.append(f"E50 band: NCE {names} about the crossing "
                     f"{'adds' if len(near) == 1 else 'add'} no scatter to it (too narrow)")
    parts.extend(f"E50 band: NCE {name(x, k)}, the lowest energy S is read against, adds no "
                 "scatter to it (too narrow)" for x, k, ref in thin if ref)
    if zero_width:
        parts.append("the band has zero width: every resample reads the same E50 (no "
                     "scatter at the crossing to draw), not a measured exactness")
    return "; ".join(parts)


def _e50_rows(bd: pd.DataFrame, order: Sequence[str], target: str, ions: Mapping[str, float],
              gas_mass: float, *, boot: Mapping | None = None, n_one: int = 0,
              boot_n: int = 0, per_scan: bool = False) -> pd.DataFrame:
    """`ms2_e50.csv`'s rows of one window; with `boot` (`_replicates`) the
    bootstrap interval of every curve's E50, `boot_n` resamples, `n_one`
    energies that add no spread (the window's, with or without a band).
    `per_scan` (the breakdown has `n_scans`) and a band: how many of the
    energies the band rests on add none to the curve -- the two about the
    averaged crossing and, for S, the lowest energy every resample's S is
    read against (counted once where it is one of the two): fewer than two
    scans, or no spread of the curve there (`_has_spread`: only held peaks
    enter it, fewer than two scans observe a ratio; for S its precursor has
    no CV) -- (`one_scan_bracket`; 0 with no band): the band holds no scatter of
    theirs, said in the note, as is a band of zero width
    (`replicates.zero_width`)."""
    rows = []
    for f in order:
        g = bd[bd["ion"] == f]
        for col, name in CURVES:
            e = BD.e50(g["nce"], g[col], g["hcd_v"])
            ok = g[np.isfinite(g[col])]
            notes = []
            if name == "S":
                notes.append(S_STEADY)
            if not e.monotonic:
                notes.append(f"non-monotonic (rises by > {BD.MONOTONIC_TOL}): the first crossing")
            v_note = _volts_note(g["hcd_v"], e)
            if v_note:
                notes.append(v_note)
            b = (boot or {}).get((f, name))
            if b is not None:
                b_note = _boot_note(b, boot_n, e)
                if b_note:
                    notes.append(b_note)
            # the energies the band rests on that add no spread to it: none
            # without a band (--bootstrap 0, no energy of two scans)
            thin = []
            if per_scan and b is not None:
                about = _crossing_scans(g, col)
                thin = [(x, k, False) for x, k, has in about if k < 2 or not has]
                if name == "S" and about and len(g):
                    x0, k0, has0 = _reference_scans(g)
                    if (k0 < 2 or not has0) and not any(np.isclose(x0, x) for x, _, _ in about):
                        thin.append((x0, k0, True))
                k_note = _bracket_note(thin, RP.zero_width(b["nce_lo"], b["nce_hi"]))
                if k_note:
                    notes.append(k_note)

            def ecm(v, _m=ions[f]):
                return BD.e_cm(v, _m, gas_mass) if math.isfinite(v) else math.nan
            rows.append({
                "parent_mz": float(g["parent_mz"].iloc[0]) if len(g) else math.nan,
                "ion": f, "role": "target" if f == target else "co-isolate", "ion_mz": ions[f],
                "curve": name, "e50_nce": e.nce, "e50_v": e.volts,
                "e50_ecm": (BD.e_cm(e.volts, ions[f], gas_mass) if math.isfinite(e.volts)
                            else math.nan),
                "flag": e.flag, "monotonic": e.monotonic, "n_energies": e.n_energies,
                "nce_min": e.nce_min, "nce_max": e.nce_max,
                "value_at_min": float(ok[col].iloc[0]) if len(ok) else math.nan,
                "value_at_max": float(ok[col].iloc[-1]) if len(ok) else math.nan,
                "note": "; ".join(notes),
                "e50_nce_lo": b["nce_lo"] if b else math.nan,
                "e50_nce_hi": b["nce_hi"] if b else math.nan,
                "e50_v_lo": b["v_lo"] if b else math.nan,
                "e50_v_hi": b["v_hi"] if b else math.nan,
                "e50_ecm_lo": ecm(b["v_lo"]) if b else math.nan,
                "e50_ecm_hi": ecm(b["v_hi"]) if b else math.nan,
                "e50_lo_censor": b["lo_where"] if b else "",
                "e50_hi_censor": b["hi_where"] if b else "",
                "boot_n": boot_n if b else 0, "boot_cross": b["cross"] if b else math.nan,
                "n_one_scan": int(n_one), "one_scan_bracket": len(thin)})
    return _frame(rows, E50_COLUMNS)


def _fragment_rows(parent_mz: float, att: pd.DataFrame, roles: pd.DataFrame,
                   frag_int: pd.DataFrame, tic: pd.Series, meta: pd.DataFrame,
                   cand_notes: Mapping[str, str] | None = None,
                   compositions: Mapping[str, Sequence[str]] | None = None,
                   replicates: Mapping[tuple[str, str], tuple[float, float]] | None = None
                   ) -> pd.DataFrame:
    """`ms2_fragments.csv`'s rows of one window. `test_note` holds the
    intact test's note and, after it, `candidates_across`'s. `compositions`
    (`AT.fragment_compositions`): the formulas each fragment can be where it
    sits; a candidate's loss is the one of its reading as one of them.
    `replicates` ((spectrum, fragment) -> (presence, CV), `_replicates`):
    read at the spectrum of its largest TIC fraction."""
    frac = frag_int.div(tic.replace(0, np.nan), axis=0)
    fr = roles[roles["role"] == "fragment"]
    # each candidate's loss, from the lowest energy at which it reads the
    # fragment as one of its compositions (else at which it reads it at all):
    # one peak row need not hold every candidate, and a candidate may read
    # one physical fragment as two formulas at two energies
    loss_of: dict[str, dict[str, str]] = {}
    any_of: dict[str, dict[str, str]] = {}
    for key, forms, text in zip(fr["key"], fr["formulas"], fr["losses"]):
        inside = set((compositions or {}).get(key, ()))
        formula = dict(AT._pairs(forms))
        for ion, name in AT._pairs(text):
            any_of.setdefault(key, {}).setdefault(ion, name)
            if formula.get(ion) in inside:
                loss_of.setdefault(key, {}).setdefault(ion, name)
    rows = []
    for a in att.itertuples():
        present = frag_int[a.key] > 0
        nce = meta.loc[frag_int.index, "nce"].to_numpy(dtype=float)
        volts = meta.loc[frag_int.index, "hcd_v"].to_numpy(dtype=float)
        on = BD.onset(nce, present.to_numpy())
        i_on = int(np.argmax(present.to_numpy())) if present.any() else None
        imax = int(np.nanargmax(frac[a.key].to_numpy())) if present.any() else None
        formulas = a.key.split(" | ")
        rep_at = ((replicates or {}).get((str(frag_int.index[imax]), a.key), (math.nan, math.nan))
                  if imax is not None else (math.nan, math.nan))
        mine = {**any_of.get(a.key, {}), **loss_of.get(a.key, {})}
        losses = "; ".join(f"{c}: {mine[c]}" for c in a.candidates.split(";") if c in mine)
        rows.append({
            "parent_mz": float(parent_mz), "fragment": a.key,
            "mz_theo": ion_mz(formulas[0]) if len(formulas) == 1 else math.nan,
            "candidates": a.candidates, "n_candidates": a.n_candidates,
            "losses": losses, "call": a.call, "basis": a.basis,
            "weights": AT.format_weights(a.weights), "weight_source": a.weight_source,
            "excluded": a.excluded,
            "test_note": "; ".join(t for t in (a.test_note, (cand_notes or {}).get(a.key, ""))
                                   if t),
            "onset_nce": on,
            "onset_v": volts[i_on] if i_on is not None else math.nan,
            "in_source": bool(present.iloc[0]),
            "max_frac": float(frac[a.key].iloc[imax]) if imax is not None else math.nan,
            "max_frac_nce": nce[imax] if imax is not None else math.nan,
            "n_spectra": int(present.sum()),
            "presence_at_max": rep_at[0], "cv_at_max": rep_at[1]})
    return _frame(rows, FRAGMENT_COLUMNS)


# ---------------------------------------------------------------------------
# rival formulas of one precursor (`arbitrate`)
# ---------------------------------------------------------------------------
#: The run note of a run that arbitrates rival formulas.
ARBITRATION_NOTE = ("rival formulas are arbitrated on the MS2 products (ms2_arbitration.csv, "
                    "docs/MS2.md §11.14): each rival read as the window's target, the other "
                    "named ions fixed, on every rival's calibration (scored on the one it is "
                    "credited with the most on: where the precursor sits between the rivals "
                    "decides nothing), and on decoys whose products are shifted by a few mDa; a "
                    "product counts for one rival against another only where the other cannot "
                    "read it on any rival's calibration; a verdict says which of the rivals "
                    "named the products support, never that an unnamed formula could not do "
                    "better. The other tables read one rival per group (the one named first, "
                    "unless --pick-supported), as the run with it alone named does")


def _group_window(tables: RunTables, parent_mz: float, ions: Mapping[str, float],
                  groups: Sequence[AR.RivalGroup], *, tol_mda: float,
                  axis_offset_ppm: float = 0.0) -> AR.RivalGroup | None:
    """The rival group whose ion is this window's target: the group holding
    the named ion -- of every named ion, every rival included -- nearest the
    located precursor (read through the folder's axis offset), within the
    target match (max(`TARGET_MATCH_PPM`, `tol_mda`)); None otherwise."""
    pmz = located_precursor(tables, parent_mz)
    if not (math.isfinite(pmz) and ions):
        return None
    seen = pmz / (1.0 + axis_offset_ppm * 1e-6)
    tol = max(TARGET_MATCH_PPM * seen * 1e-6, tol_mda * 1e-3)
    f, m = min(ions.items(), key=lambda kv: (abs(kv[1] - seen), kv[0]))
    if abs(m - seen) > tol:
        return None
    return next((g for g in groups if f in g.rivals), None)


def _base_ions(ions: Mapping[str, float], groups: Sequence[AR.RivalGroup],
               pick: Mapping[str, str]) -> dict[str, float]:
    """The named ions the run's tables read: each group's picked rival, none
    of its others."""
    out_ = {r for g in groups for r in g.rivals if r != pick[g.name]}
    return {f: m for f, m in ions.items() if f not in out_}


def _window_tables(tables: RunTables, parent_mz: float) -> RunTables:
    """The folder restricted to one parent key (its spectra, parent rows and
    MS1 window rows; no per-scan tables): what a rival's read needs."""
    par = tables.parents[np.isclose(tables.parents["parent_mz"], parent_mz, rtol=0, atol=1e-6)]
    sp = tables.spectra[tables.spectra["spectrum"].isin(set(par["spectrum"]))]
    win = tables.window
    if len(win):
        win = win[np.isclose(win["parent_mz"], parent_mz, rtol=0, atol=1e-6)]
    return RunTables(spectra=sp.reset_index(drop=True), parents=par.reset_index(drop=True),
                     window=win.reset_index(drop=True), manifest=tables.manifest)


def _shifted(small: RunTables, specs: Sequence[str], keep: set, shift_da: float, *,
             tol_ppm: float = 0.0, tol_mda: float = 0.0,
             min_shift_da: float | None = None) -> RunTables:
    """`small` with every peak of the energy-resolved spectra `specs` moved
    by `shift_da`, except the (spectrum, peak) pairs in `keep` (the window
    ions' precursors, the harmonic ghosts, the non-products): one decoy. A
    peak where the smallest shift of the set (`min_shift_da`, default
    |`shift_da`|) is under `arbitrate.DECOY_TOL_FACTOR` times its reading
    tolerance (max(`tol_ppm`, `tol_mda`) at its m/z) is moved by `shift_da`
    scaled up by the factor that brings that smallest shift there -- every
    shift of the set by one factor, so they stay distinct. A peak's index is
    its place among its spectrum's rows, as `analyze_window` numbers it."""
    sp = small.spectra.copy()
    mz = sp["mz"].to_numpy(dtype=float).copy()
    least = abs(shift_da) if min_shift_da is None else abs(float(min_shift_da))
    for s in specs:
        idx = np.flatnonzero((sp["spectrum"] == s).to_numpy())
        for j, i in enumerate(idx):
            if (s, j) not in keep:
                need = AR.DECOY_TOL_FACTOR * SF._tolerance_da(mz[i], tol_ppm, tol_mda)
                mz[i] += shift_da * (max(1.0, need / least) if least > 0 else 1.0)
    sp["mz"] = mz
    return RunTables(spectra=sp, parents=small.parents, window=small.window,
                     manifest=small.manifest)


def _on_calibration(t: RunTables, specs: Sequence[str], keep: set, factor: float) -> RunTables:
    """`t` with every peak of the energy-resolved spectra `specs` but the
    (spectrum, peak) pairs in `keep` (the window ions' precursors, the
    harmonic ghosts) multiplied by `factor`: read with one rival's
    recalibration, its products sit where another rival's puts them
    (`_cross_reads`). A peak's index is its place among its spectrum's rows,
    as `analyze_window` numbers it."""
    sp = t.spectra.copy()
    mz = sp["mz"].to_numpy(dtype=float).copy()
    for s in specs:
        idx = np.flatnonzero((sp["spectrum"] == s).to_numpy())
        moved = np.array([(s, j) not in keep for j in range(len(idx))], dtype=bool)
        mz[idx[moved]] *= factor
    sp["mz"] = mz
    return RunTables(spectra=sp, parents=t.parents, window=t.window, manifest=t.manifest)


def _rival_read(w: WindowResult, rival: str) -> AR.RivalRead:
    return AR.RivalRead(rival=rival, roles=w.roles, readings=w.readings, fragments=w.fragments,
                        offset_ppm=w.offset_ppm, co_isolates=tuple(w.co_isolates))


def _kept(reads: Mapping[str, AR.RivalRead]) -> set[tuple[str, int]]:
    """The (spectrum, peak) pairs any read takes for a window ion's precursor
    or a harmonic ghost."""
    out = set()
    for rr in reads.values():
        if rr.roles is not None:
            m = rr.roles["role"].isin(["precursor", "harmonic"]).to_numpy()
            out |= {(s, int(p)) for s, p in zip(rr.roles.loc[m, "spectrum"],
                                                 rr.roles.loc[m, "peak"])}
    return out


def _cross_reads(t: RunTables, reads: Mapping[str, AR.RivalRead],
                 read: Callable[[RunTables, str], WindowResult], *, specs: Sequence[str],
                 keep: set, offsets: Mapping[str, float]) -> dict[str, AR.RivalRead]:
    """Every rival read again on each other rival's calibration
    (`arbitrate.RivalRead.cross`): the window's peaks but `keep` (the
    precursors, the harmonic ghosts) moved by the two calibrations' ratio,
    (1 + its offset) / (1 + the other's) (`offsets`: each rival's
    recalibration on its own precursor, the window's), and read as any read
    (`read`: recalibrated on its own precursor, the window's other named ions
    fixed, the same gate, harmonic rule and attribution) -- its products
    read, and given by the attribution, where the other rival reads them."""
    out = {}
    for R, rr in reads.items():
        cross = {}
        if rr.roles is not None and math.isfinite(offsets.get(R, math.nan)):
            for A in reads:
                if A == R or reads[A].roles is None or not math.isfinite(
                        offsets.get(A, math.nan)):
                    continue
                f = (1.0 + offsets[R] * 1e-6) / (1.0 + offsets[A] * 1e-6)
                cross[A] = _rival_read(read(_on_calibration(t, specs, keep, f), R), R)
        out[R] = AR.RivalRead(rival=rr.rival, roles=rr.roles, readings=rr.readings,
                              fragments=rr.fragments, offset_ppm=rr.offset_ppm,
                              co_isolates=rr.co_isolates, cross=cross)
    return out


def _arbitrate_window(tables: RunTables, parent_mz: float, group: AR.RivalGroup,
                      base: Mapping[str, float], main: WindowResult, chosen: str, *,
                      width: float, shifts: Sequence[float], k_max: int | None, ladder: bool,
                      tol_ppm: float, tol_mda: float, gas_mass: float, axis_offset_ppm: float,
                      harmonics: bool) -> pd.DataFrame:
    """One rival group's window: every rival read as its target -- the
    window's other named ions fixed, recalibrated on its own precursor --
    and again on each other rival's calibration (`_cross_reads`), the same
    on each decoy (`_shifted`: the product peaks moved; the precursors,
    harmonic ghosts and non-products kept, `arbitrate.non_products`), and
    `arbitrate.score_window`. `main`: the window as the run's other tables
    read it, as the run with `chosen` alone named does (its target matched,
    never forced); its read is the chosen rival's when that is its target,
    else the chosen rival is read on its own, and no rival is `analysed`. A
    window read with no energy axis, or not read at all, is undecided
    (said)."""
    others = {f: m for f, m in base.items() if f != chosen}
    small = _window_tables(tables, parent_mz)
    kw = dict(width=width, k_max=k_max, ladder=ladder, tol_ppm=tol_ppm, tol_mda=tol_mda,
              gas_mass=gas_mass, axis_offset_ppm=axis_offset_ppm, bootstrap=0,
              harmonics=harmonics, adducts=None)

    def read(t, R):
        return analyze_window(t, parent_mz, {**others, R: ion_mz(R)}, target=R, **kw)
    pmz = main.precursor_mz
    seen = pmz / (1.0 + axis_offset_ppm * 1e-6) if math.isfinite(pmz) else math.nan
    prec = {R: ((seen - ion_mz(R)) / ion_mz(R) * 1e6 if math.isfinite(seen) else math.nan)
            for R in group.rivals}
    ms1 = _ms1_ppm(tables.window, parent_mz, group.rivals, tol_mda, precursor_mz=pmz,
                   axis_offset_ppm=axis_offset_ppm)
    analysed = main.target if main.target in group.rivals else ""
    first = main if analysed == chosen else read(small, chosen)
    note = _arbitration_notes(tables, parent_mz, first, group, tol_mda, axis_offset_ppm,
                              width=width)
    notes = {R: note for R in group.rivals} if note else {}
    if first.roles is None:
        why = {SKIP_NO_PEAKS: "no spectrum with peaks",
               SKIP_NOT_LOCATED: "no precursor located",
               NOT_RESOLVED: "no energy-resolved spectrum (the attribution reads the energy "
                             "axis)"}.get(first.reason, first.reason or first.status)
        return AR.frame(AR.undecided_rows(parent_mz, group, analysed=analysed, why=why,
                                          precursor_ppm=prec, ms1_ppm=ms1, notes=notes))
    specs = list(dict.fromkeys(first.roles["spectrum"]))
    own = {R: _rival_read(first if R == chosen else read(small, R), R) for R in group.rivals}
    offsets = {R: rr.offset_ppm for R, rr in own.items()}
    kept = _kept(own)
    reads = _cross_reads(small, own, read, specs=specs, keep=kept, offsets=offsets)
    # no singly charged product lies above the precursor less an H atom
    ceiling = (pmz - AR.H_ATOM + SF._tolerance_da(pmz, tol_ppm, tol_mda)
               if math.isfinite(pmz) else math.nan)
    exclude = AR.non_products(reads, group.rivals, ceiling_mz=ceiling)
    keep = kept | set(exclude)
    least = min((abs(float(d)) for d in shifts), default=0.0) * 1e-3
    decoys = []
    for d in shifts:
        t2 = _shifted(small, specs, keep, float(d) * 1e-3, tol_ppm=tol_ppm, tol_mda=tol_mda,
                      min_shift_da=least)
        decoys.append(_cross_reads(t2, {R: _rival_read(read(t2, R), R) for R in group.rivals},
                                   read, specs=specs, keep=keep, offsets=offsets))
    rows = AR.score_window(parent_mz, group, reads, decoys, analysed=analysed,
                           precursor_ppm=prec, ms1_ppm=ms1, notes=notes, exclude=exclude)
    return AR.frame(rows)


def _ms1_ppm(window: pd.DataFrame, parent_mz: float, rivals: Sequence[str], tol_mda: float,
             *, precursor_mz: float = math.nan, axis_offset_ppm: float = 0.0
             ) -> dict[str, float]:
    """Each rival's error (ppm) at the MS1 window centroid AT THE LOCATED
    PRECURSOR -- the centroid nearest `precursor_mz` within
    max(`TARGET_MATCH_PPM`, `tol_mda`), both read through the folder's axis
    offset: one centroid for every rival, so a rival MS1 resolves from the
    precursor reads its error there, never at a weaker neighbour of its own;
    NaN for every rival where there is none (no MS1 row, no located
    precursor, no centroid within reach)."""
    out = {R: math.nan for R in rivals}
    if not len(window) or not math.isfinite(precursor_mz):
        return out
    w = window[np.isclose(window["parent_mz"], parent_mz, rtol=0, atol=1e-6)]
    if not len(w):
        return out
    mz = w["mz"].to_numpy(dtype=float) / (1.0 + axis_offset_ppm * 1e-6)
    seen = precursor_mz / (1.0 + axis_offset_ppm * 1e-6)
    i = int(np.argmin(np.abs(mz - seen)))
    if abs(mz[i] - seen) > max(TARGET_MATCH_PPM * seen * 1e-6, tol_mda * 1e-3):
        return out
    return {R: float((mz[i] - ion_mz(R)) / ion_mz(R) * 1e6) for R in rivals}


#: The S/N an MS1 centroid needs for the arbitration's `note` to name it as
#: an ion the products may come from: `ms2 extract`'s own floor
#: (`sources.DEFAULT_NOISE_THRESHOLD`). The window inventory keeps every
#: centroid, whatever its S/N: one below the floor is noise, no ion. A
#: centroid with no S/N (NaN) is kept.
ARBITRATION_NOTE_SN = SRC.DEFAULT_NOISE_THRESHOLD


def _arbitration_notes(tables: RunTables, parent_mz: float, read: WindowResult,
                       group: AR.RivalGroup, tol_mda: float, axis_offset_ppm: float, *,
                       width: float = math.nan) -> str:
    """The caveat every rival row of the window carries ('' : none). Named:
    every rival of the group and the window's co-isolates (`read`: the
    chosen rival's read). From the MS1 window table: a centroid no named ion
    matches, at S/N `ARBITRATION_NOTE_SN` or more -- or with no S/N known
    (NaN) --, holding more of the isolation window than the precursor (the
    largest share at a rival) -- or the precursor not seen there, with such
    a centroid or without (MS1 does not see it). With no MS1 row for the
    window: the same from the lowest-energy MS2 spectrum
    (`_ms2_window_note`)."""
    named = {f: ion_mz(f) for f in dict.fromkeys([*group.rivals, *read.co_isolates]) if f}
    shares = (_ms1_shares(tables.window, parent_mz, named, tol_mda,
                          axis_offset_ppm=axis_offset_ppm) if len(tables.window) else {})
    if not shares:
        return _ms2_window_note(read, parent_mz, width, group)
    at = [shares[R] for R in group.rivals if math.isfinite(shares.get(R, math.nan))]
    own = max(at) if at else math.nan
    unnamed = _unnamed_ms1(tables.window, parent_mz, named, tol_mda,
                           axis_offset_ppm=axis_offset_ppm, min_sn=ARBITRATION_NOTE_SN)
    if own == 0 and unnamed is None:
        return ("the MS1 window holds no centroid at the precursor, nor one no named ion "
                f"matches above S/N {ARBITRATION_NOTE_SN:g}: MS1 does not see the precursor")
    if own == 0:
        return (f"the MS1 window holds no centroid at the precursor, and one no named ion "
                f"matches, at m/z {unnamed[0]:.4f}, holds {unnamed[1]:.0%} of it: the products "
                "may be another ion's, read through each rival's formula")
    if unnamed is not None and math.isfinite(own) and unnamed[1] > own:
        return (f"an MS1 centroid no named ion matches, at m/z {unnamed[0]:.4f}, holds "
                f"{unnamed[1]:.0%} of the isolation window, more than the precursor ({own:.0%}): "
                "the products may be another ion's, read through each rival's formula -- name "
                "it (--ion)")
    return ""


def _ms2_window_note(read: WindowResult, parent_mz: float, width: float,
                     group: AR.RivalGroup) -> str:
    """With no MS1 window row: in the lowest-energy MS2 spectrum of `read`,
    the peaks inside the isolation window (`width`/2 + `sources.WINDOW_PAD_DA`
    of the key) that no read takes for a named ion's precursor (nor a
    harmonic ghost), against the precursor's peaks there -- said when they
    hold more ('' otherwise, or with no width or no energy axis)."""
    r = read.roles
    if r is None or not len(r) or not (math.isfinite(width) and width > 0):
        return ""
    nce = r["nce"].to_numpy(dtype=float)
    if not np.isfinite(nce).any():
        return ""
    low = float(np.nanmin(nce))
    spec = str(r.loc[nce == low, "spectrum"].iloc[0])
    s0 = r[r["spectrum"] == spec]
    s0 = s0[np.abs(s0["mz"].to_numpy(dtype=float) - parent_mz) <= width / 2 + SRC.WINDOW_PAD_DA]
    total = float(s0["intensity"].sum())
    prec = float(s0.loc[(s0["role"] == "precursor") & s0["ion"].isin(group.rivals),
                        "intensity"].sum())
    other = s0[~s0["role"].isin(["precursor", "harmonic"])]
    if total <= 0 or not len(other) or float(other["intensity"].sum()) <= prec:
        return ""
    top = other.loc[other["intensity"].idxmax()]
    return (f"the MS1 window table holds no row for this window; in the lowest-energy MS2 "
            f"spectrum (NCE {low:g}) peaks no named ion matches hold "
            f"{float(other['intensity'].sum()) / total:.0%} of the isolation window, more than "
            f"the precursor ({prec / total:.0%}), the largest at m/z {float(top['mz']):.4f}: the "
            "products may be another ion's, read through each rival's formula -- name it "
            "(--ion)")


def _group_winners(windows: Sequence[WindowResult], group: AR.RivalGroup
                   ) -> list[tuple[float, str]]:
    """(parent key, supported rival) of every window of `group` whose
    arbitration reaches a verdict, in window order."""
    return [(w.parent_mz, str(w.arbitration["winner"].iloc[0])) for w in windows
            if w.rival_group == group.name and w.arbitration is not None
            and len(w.arbitration) and w.arbitration["winner"].iloc[0]]


def _rival_skip(w: WindowResult, group: AR.RivalGroup, pick: str, ions: Mapping[str, float],
                *, axis_offset_ppm: float, winners: Sequence[tuple[float, str]] = ()) -> None:
    """A rival group's window the run's tables skip: the rival they read
    (`pick`) beyond the target match of the located precursor, which another
    rival matches (`_group_window`: the nearest named ion is a rival). The
    skip's own note -- no named ion matches, name it or pass
    --mass-offset-ppm -- is replaced: a rival does match, and the offset
    would read the window with `pick` whatever the arbitration says. Keyed on
    the verdict: where it supports `pick`, the folder's axis is the likely
    cause and the offset hint stands; where it supports the matching rival,
    --pick-supported (or naming it first) reads the tables with it, and the
    hint is cleared (the offset would read the outscored rival on an axis
    that many ppm wrong) -- unless the group's windows (`winners`,
    `_group_winners`) support different rivals: then --pick-supported keeps
    the first rival (`_picks`), whether given or not, and naming this one
    first reads the others with a rival they do not support where their
    precursors match it (and skips them where they do not), so that is said
    instead; where it supports another rival, or none, the remedies are said
    and the hint cleared -- naming a rival first said with the windows that
    support another (`winners`). -> `w.rival_hint` ('' where the offset hint
    stands: the exit sums it up with the axis-offset windows)."""
    pmz = w.precursor_mz
    if not math.isfinite(pmz) or w.arbitration is None or not len(w.arbitration):
        return
    seen = pmz / (1.0 + axis_offset_ppm * 1e-6)
    m = min(ions.items(), key=lambda kv: (abs(kv[1] - seen), kv[0]))[0]
    if m not in group.rivals or m == pick:
        return

    def off(R):
        return (seen - ions[R]) / ions[R] * 1e6
    win = str(w.arbitration["winner"].iloc[0])
    split = ("" if len({x for _, x in winners}) <= 1 else
             "the group's windows support different rivals ("
             + "; ".join(f"{x} at parent {SRC.spectrum_key(p)}" for p, x in winners) + ")")
    need = offset_hint(pmz, {pick: ions[pick]}, axis_offset_ppm)
    head = (f"skipped: the located precursor at {pmz:.5f} matches {m} ({off(m):+.1f} ppm), a "
            f"rival of group {group.name!r}, not {pick}, the one the tables read "
            f"({off(pick):+.1f} ppm, beyond the target match); ")
    axis = (f"-- if the folder's m/z axis sits off its formulas -- pass --mass-offset-ppm "
            f"{need:+.1f}" if math.isfinite(need) else "")
    if win == pick and axis:
        w.offset_hint_ppm = need
        advice, w.rival_hint = f"the arbitration supports {pick}: {axis}", ""
    else:
        w.offset_hint_ppm = math.nan
        if win == m and split:
            advice = (f"the arbitration supports {m} here, but {split}: no one rival reads "
                      f"every window's tables -- --pick-supported keeps {pick}, named first, and "
                      f"naming {m} first reads the other windows with it where their located "
                      "precursors match it (and skips them where they do not)")
        elif win == m:
            advice = (f"the arbitration supports {m}: --pick-supported reads the tables with it "
                      "(or name it first)")
        elif win == pick:
            advice = (f"the arbitration supports {pick}, beyond the reach an axis offset is read "
                      "over (see ms2_arbitration.csv)")
        elif win:
            nw = offset_hint(pmz, {win: ions[win]}, axis_offset_ppm)
            advice = (f"the arbitration supports {win}, which the located precursor does not "
                      f"match either ({off(win):+.1f} ppm): name it first"
                      + (f", and -- if the folder's m/z axis sits off its formulas -- pass "
                         f"--mass-offset-ppm {nw:+.1f}" if math.isfinite(nw) else "")
                      + (f" (but {split})" if split else ""))
        else:
            # naming m first reads every window of the group with m: said
            # where another window supports another rival
            other = [(p, x) for p, x in winners if x != m]
            advice = (f"the arbitration supports none of the rivals: name {m} first to read the "
                      "tables with the rival the precursor matches"
                      + (f", or {axis} to read {pick}" if axis else "")
                      + (" (but " + "; ".join(f"parent {SRC.spectrum_key(p)} supports {x}"
                                              for p, x in other)
                         + f": naming {m} first reads those windows with a rival they do not "
                         "support, where their located precursors match it)" if other else ""))
        w.rival_hint = f"parent {SRC.spectrum_key(w.parent_mz)}: {advice}"
    w.notes = [n for n in w.notes if not n.startswith("skipped: no named ion matches")]
    w.notes.append(head + advice)


def _elsewhere(w: WindowResult, rival: str) -> tuple[str, float, float] | None:
    """(the rival whose calibration `rival`'s scores are read on, its share
    there, its share on its own calibration) in `w`'s arbitration, where
    that is another rival's (`arbitrate.product_table`); else None."""
    a = w.arbitration
    if a is None or not len(a) or rival not in set(a["rival"]):
        return None
    r = a[a["rival"] == rival].iloc[0]
    on = str(r["read_on"])
    if on in ("", rival):
        return None
    return on, float(r["explained"]), float(r["explained_own"])


def _calibration_note(w: WindowResult) -> str:
    """Where the rival the tables read (`analysed`) is scored on another
    rival's calibration (`read_on`): the tables -- the run with it alone
    named, recalibrated on its own precursor -- may not show the products
    the arbitration credits it with (a rival credited with every product on
    the other's calibration, 5 ppm away, was credited with 13 % on its own --
    one product, misread: tested). Said, with the share it is credited with
    on its own ('' where it is scored on its own, or the tables read no
    rival)."""
    a = w.arbitration
    if a is None or not len(a) or not a["analysed"].any():
        return ""
    R = str(a.loc[a["analysed"].to_numpy(dtype=bool), "rival"].iloc[0])
    got = _elsewhere(w, R)
    if got is None:
        return ""
    on, there, mine = got
    return (f"the tables read {R} recalibrated on its own precursor, where it is credited with "
            f"{mine:.0%} of the products; the arbitration credits it with {there:.0%} read on "
            f"{on}'s calibration (read_on): the tables may not show the products it is credited "
            "with there")


def _picks(windows: Sequence[WindowResult], groups: Sequence[AR.RivalGroup]
           ) -> tuple[dict[str, str], list[str]]:
    """--pick-supported: per group, the rival its windows support (every
    window that reaches a verdict supports the same one); else its first
    rival. -> ({group: rival}, notes: a pick other than the first rival
    said, and where its support is read on another rival's calibration --
    the tables read it on its own, `_calibration_note`)."""
    picks, notes = {}, []
    for g in groups:
        ws = list(dict.fromkeys(x for _, x in _group_winners(windows, g)))
        if len(ws) == 1:
            picks[g.name] = ws[0]
            if ws[0] != g.rivals[0]:
                off = [(w.parent_mz, *x) for w in windows if w.rival_group == g.name
                       for x in [_elsewhere(w, ws[0])] if x is not None]
                notes.append(f"rival group {g.name!r}: --pick-supported reads every table with "
                             f"{ws[0]}, which the arbitration supports (named first: "
                             f"{g.rivals[0]})"
                             + ("; the tables read it recalibrated on its own precursor, while "
                                "its support is read on another rival's calibration at "
                                + "; ".join(f"parent {SRC.spectrum_key(p)} ({on}'s: {there:.0%} "
                                            f"of the products, {mine:.0%} on its own)"
                                            for p, on, there, mine in off)
                                + " -- the tables may not show the products it is credited "
                                "with (the windows' notes)" if off else ""))
        else:
            picks[g.name] = g.rivals[0]
            if len(ws) > 1:
                notes.append(f"rival group {g.name!r}: its windows support different rivals ("
                             + ", ".join(ws) + f"): --pick-supported keeps {g.rivals[0]}, "
                             "named first")
    return picks, notes


# ---------------------------------------------------------------------------
# the whole folder
# ---------------------------------------------------------------------------
@dataclass
class AnalyzeResult:
    windows: list
    readings: pd.DataFrame
    breakdown: pd.DataFrame
    e50: pd.DataFrame
    fragments: pd.DataFrame
    branching: pd.DataFrame
    params: dict
    notes: list = field(default_factory=list)
    #: `replicates.REPLICATE_COLUMNS` (None: no per-scan tables)
    replicates: pd.DataFrame | None = None
    #: `channels.CHANNEL_COLUMNS` / `channels.CLASS_COLUMNS` (None: no adduct
    #: given for any ion)
    channels: pd.DataFrame | None = None
    classes: pd.DataFrame | None = None
    #: `arbitrate.ARBITRATION_COLUMNS` (None: no rival group given)
    arbitration: pd.DataFrame | None = None


def analyze_run(tables: RunTables, ions: Mapping[str, float], *, k_max: int | None = None,
                ladder: bool = True, tol_ppm: float = 5.0, tol_mda: float = 0.3,
                gas_mass: float = BD.N2_MASS, iso_width: float | None = None,
                mass_offset_ppm: float | None = None, bootstrap: int = RP.BOOTSTRAP_B,
                harmonics: bool = True,
                adducts: Mapping[str, "CH.IonAdduct | None"] | None = None,
                adduct_notes: Sequence[str] = (),
                rivals: Sequence[AR.RivalGroup] = (), rival_ppm: float | None = None,
                pick_supported: bool = False,
                decoy_shifts_mda: Sequence[float] = AR.DECOY_SHIFTS_MDA,
                rival_notes: Sequence[str] = ()) -> AnalyzeResult:
    """Every parent window of one `ms2 extract` folder (`load_run`).
    `mass_offset_ppm`: the folder's m/z axis offset, replacing the one its
    manifest records (`folder_axis_offset`). `bootstrap`: resamples of each
    E50's interval (0: none; else at least `replicates.BOOTSTRAP_MIN`; needs
    the per-scan tables). `harmonics` False reads harmonic ghosts as any peak
    (a debugging switch). `adducts` (named ion -> `channels.IonAdduct` or
    None, `channels.assign_adducts`; its `adduct_notes` go to the run's
    notes): every analysed window's product channels and binding classes
    (`channels`); None (the default) reads none, and every other table is
    what it was.

    `rivals` (`arbitrate.RivalGroup`, every rival a named ion of `ions`;
    checked by `arbitrate.check_groups` -- no two of one name, no formula in
    two -- and `arbitrate.check_group` with `rival_ppm`): the window whose
    located precursor a group's rival is the nearest named ion of
    (`_group_window`) is read once per rival and once per rival on each
    decoy (`decoy_shifts_mda`), and arbitrated (`arbitrate.score_window`,
    `ms2_arbitration.csv`). Every other table reads one rival per group: the
    one named first -- or, with `pick_supported`, the rival the arbitration
    supports (in every window of the group that reaches a verdict; said) --
    as the run with it alone named does (its target matched, never forced),
    and the group's other rivals are window ions nowhere. `rival_notes` go to
    the run's notes. With no group the run is what it was."""
    SF._check_tol(tol_ppm, tol_mda)
    if (isinstance(bootstrap, bool) or int(bootstrap) != bootstrap or bootstrap < 0
            or 0 < bootstrap < RP.BOOTSTRAP_MIN):
        raise ValueError(f"bootstrap must be 0 (none) or a whole number of resamples >= "
                         f"{RP.BOOTSTRAP_MIN}, got {bootstrap!r}")
    if mass_offset_ppm is not None and not math.isfinite(float(mass_offset_ppm)):
        raise ValueError(f"mass_offset_ppm must be finite, got {mass_offset_ppm!r}")
    axis, axis_src = folder_axis_offset(tables, mass_offset_ppm)
    BD.e_cm(1.0, 100.0, gas_mass)                   # checks the gas mass
    mode, k_read = RD.gate_mode(k_max, ladder=ladder)
    width, wsrc = isolation_width(tables, iso_width)
    # the parents table's widths, when they differ and nothing replaces them:
    # each window at its own parent's (an `ms2 extract --file` folder)
    own = parent_widths(tables) if wsrc == "server" else {}
    per_parent = len(set(own.values())) > 1
    if per_parent:
        width, wsrc = math.nan, PER_PARENT_WIDTH
    groups = list(rivals or ())
    AR.check_groups(groups)
    for g in groups:
        AR.check_group(g, tol_ppm=tol_ppm, tol_mda=tol_mda, rival_ppm=rival_ppm)
        missing = [f for f in g.rivals if f not in ions]
        if missing:
            raise ValueError(f"rival group {g.name!r}: {', '.join(missing)} is not a named ion")
    chosen = {g.name: g.rivals[0] for g in groups}
    keys = sorted({float(x) for x in tables.parents["parent_mz"]})
    kw = dict(k_max=k_max, ladder=ladder, tol_ppm=tol_ppm, tol_mda=tol_mda, gas_mass=gas_mass,
              axis_offset_ppm=axis, harmonics=harmonics)

    def w_of(p):
        return own.get(round(p, 6), math.nan) if per_parent else width
    where = {p: _group_window(tables, p, ions, groups, tol_mda=tol_mda, axis_offset_ppm=axis)
             for p in keys} if groups else {}

    def run_windows(pick: Mapping[str, str], arbitrate: bool) -> list[WindowResult]:
        base = _base_ions(ions, groups, pick)
        out = []
        for p in keys:
            g = where.get(p)
            # the window as the run with the picked rival alone named reads
            # it: its target matched, never forced
            w = analyze_window(tables, p, base, width=w_of(p), bootstrap=int(bootstrap),
                               adducts=adducts, **kw)
            if g is not None:
                w.rival_group = g.name
                if w.target != pick[g.name]:
                    w.notes.append(
                        f"rival group {g.name!r}: the other tables do not read this window "
                        f"with {pick[g.name]} -- as in the run with it alone named, "
                        + (f"its located precursor matches {w.target} first"
                           if w.target else "no named ion matches its located precursor")
                        + " -- while the arbitration reads every rival of the group")
                if arbitrate:
                    w.arbitration = _arbitrate_window(tables, p, g, base, w, pick[g.name],
                                                      width=w_of(p), shifts=decoy_shifts_mda,
                                                      **kw)
            out.append(w)
        return out
    windows = run_windows(chosen, True)
    arb_notes = []
    if groups and pick_supported:
        picks, arb_notes = _picks(windows, groups)
        changed = {k: v for k, v in picks.items() if v != chosen[k]}
        if changed:
            first = {w.parent_mz: w.arbitration for w in windows}
            chosen.update(changed)
            windows = run_windows(chosen, False)
            for w in windows:
                a = first.get(w.parent_mz)
                if a is not None:
                    w.arbitration = a.assign(analysed=a["rival"] == (w.target or ""))
    # the rival the tables read scored on another rival's calibration: said
    for w in windows:
        note = _calibration_note(w)
        if note:
            w.notes.append(f"rival group {w.rival_group!r}: {note}")
            a = w.arbitration
            m = a["analysed"].to_numpy(dtype=bool)
            a.loc[m, "note"] = [f"{n}; {note}" if n else note for n in a.loc[m, "note"]]

    for w in windows:
        if (w.arbitration is not None and w.status == "skipped"
                and w.reason == SKIP_NO_TARGET):
            g = next(x for x in groups if x.name == w.rival_group)
            _rival_skip(w, g, chosen[g.name], ions, axis_offset_ppm=axis,
                        winners=_group_winners(windows, g))

    def cat(attr, spec):
        parts = [getattr(w, attr) for w in windows if getattr(w, attr) is not None]
        return _frame(pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(), spec)
    notes = list(tables.notes)
    per_scan = tables.scan_peaks is not None
    if not per_scan and not any("ms2_scan_peaks.parquet" in n for n in notes):
        notes.append(NO_PER_SCAN_NOTE)
    if any(w.status == "analysed" for w in windows):
        notes.append(S_NOTE)
    if math.isfinite(width) and len(tables.window):
        reach = [inventory_half_width(tables, p, bound=False)
                 for p in sorted(set(tables.window["parent_mz"]))]
        reach = [h for h in reach if math.isfinite(h)]
        if reach and width / 2.0 + SRC.WINDOW_PAD_DA > min(reach) + 1e-9:
            notes.append(f"the isolation width used here ({width:g} Da, {wsrc}) reaches past the "
                         f"MS1 window ms2 extract inventoried (+-{min(reach):.3f} Da around a "
                         "key): a co-isolate beyond it has no MS1 share (unknown, not 0)")
    if adducts is not None:
        notes.extend(adduct_notes)
        notes.append(CHANNELS_NOTE)
    if groups:
        notes.append(ARBITRATION_NOTE)
        notes.extend(rival_notes)
        notes.extend(arb_notes)
        for g in groups:
            if not any(w.rival_group == g.name for w in windows):
                notes.append(f"rival group {g.name!r} ({', '.join(g.rivals)}): no window's "
                             "located precursor is nearest one of its rivals -- not arbitrated")
    params = {"k_max": k_max, "ladder": bool(ladder), "gate": mode, "k_max_read": k_read,
              "gate_note": gate_note(mode, k_read), "tol_ppm": float(tol_ppm),
              "tol_mda": float(tol_mda), "gas_mass": float(gas_mass), "iso_width": iso_width,
              "isolation_width_used": None if math.isnan(width) else width,
              "isolation_width_source": wsrc, "window_pad_da": SRC.WINDOW_PAD_DA,
              "isolation_widths": ({repr(p): w for p, w in sorted(own.items())}
                                   if per_parent else None),
              "target_match_ppm": TARGET_MATCH_PPM,
              "mass_offset_ppm": axis, "mass_offset_source": axis_src,
              "intact_fraction": AT.INTACT_FRACTION,
              "branching_flag": AT.BRANCHING_FLAG, "e50_threshold": BD.E50_THRESHOLD,
              "monotonic_tol": BD.MONOTONIC_TOL,
              # every other constant a call turns on, so a run can be re-read
              "onset_stray_share": AT.ONSET_STRAY_SHARE,
              "satellite_ratio": AT.SATELLITE_RATIO, "satellite_fwhm": AT.SATELLITE_FWHM,
              "polyoxy_max_c": AT.POLYOXY_MAX_C, "polyoxy_min_o": AT.POLYOXY_MIN_O,
              "harmonics": {"on": bool(harmonics), "orders": list(RD.HARMONIC_ORDERS),
                            "ppm": RD.HARMONIC_PPM,
                            "cap": {f"{n}f": c for n, c in RD.HARMONIC_CAP.items()},
                            "clear_sn": RD.HARMONIC_CLEAR_SN},
              "replicates": {"per_scan": per_scan, "bootstrap": int(bootstrap),
                             "bootstrap_min": RP.BOOTSTRAP_MIN, "seed": RP.BOOTSTRAP_SEED,
                             "percentiles": list(RP.BOOTSTRAP_PERCENTILES),
                             "match_da": RP.MATCH_DA, "scale_band": list(RP.SCALE_BAND)},
              "channels": {"on": adducts is not None, "monomer_min_c": CH.MONOMER_MIN_C,
                           "frag_min": CH.FRAG_MIN, "dominant": CH.DOMINANT,
                           "cluster_pop": CH.CLUSTER_POP,
                           "unexplained_max": CH.UNEXPLAINED_MAX,
                           "covalent_ecm": CH.COVALENT_ECM, "skeletal_ecm": CH.SKELETAL_ECM,
                           "no_monomer": CH.NO_MONOMER, "first_mass_min": CH.FIRST_MASS_MIN,
                           "own_weight": CH.OWN_WEIGHT},
              "arbitration": {"on": bool(groups), "rival_ppm": rival_ppm,
                              "pick_supported": bool(pick_supported),
                              "decoy_shifts_mda": [float(x) for x in decoy_shifts_mda],
                              "decoy_tol_factor": AR.DECOY_TOL_FACTOR,
                              "share_margin": AR.SHARE_MARGIN,
                              "diagnostic_min": AR.DIAGNOSTIC_MIN,
                              "diagnostic_floor": AR.DIAGNOSTIC_FLOOR,
                              "excess_min": AR.EXCESS_MIN, "chance_said": AR.CHANCE_SAID,
                              "signature_min": AR.SIGNATURE_MIN,
                              "signature_expect": AR.SIGNATURE_EXPECT,
                              "reagent_neutrals": {("+" if q > 0 else "-"): list(v) for q, v
                                                   in AR.REAGENT_NEUTRALS.items()},
                              "reagent_neutrals_max": AR.REAGENT_NEUTRALS_MAX,
                              "merged_neutrals": list(AR.MERGED_NEUTRALS),
                              "merged_max": AR.MERGED_MAX,
                              "bare_neutrals": {e: list(v) for e, v in AR.BARE_NEUTRALS.items()},
                              "skeletal_elements": list(AR.SKELETAL_ELEMENTS),
                              "skeletal_o_over_c": AR.SKELETAL_O_OVER_C,
                              "note_min_sn": ARBITRATION_NOTE_SN,
                              "groups": [{"name": g.name, "rivals": list(g.rivals),
                                          "source": g.source, "analysed_with": chosen[g.name]}
                                         for g in groups]},
              "ions": [{"formula": f, "mz": m,
                        **({"adduct": (adducts.get(f).label if adducts.get(f) else ""),
                            "adduct_source": (adducts.get(f).source if adducts.get(f) else "")}
                           if adducts is not None else {}),
                        **({"rival_group": next((g.name for g in groups if f in g.rivals), "")}
                           if groups else {})} for f, m in ions.items()]}
    return AnalyzeResult(windows=windows,
                         readings=cat("readings", RD.READINGS_COLUMNS),
                         breakdown=cat("breakdown", BREAKDOWN_COLUMNS),
                         e50=cat("e50", E50_COLUMNS), fragments=cat("fragments", FRAGMENT_COLUMNS),
                         branching=cat("branching", BRANCHING_COLUMNS), params=params,
                         notes=notes,
                         replicates=(cat("replicates", RP.REPLICATE_COLUMNS) if per_scan
                                     else None),
                         channels=(cat("channels", CH.CHANNEL_COLUMNS) if adducts is not None
                                   else None),
                         classes=(cat("classes", CH.CLASS_COLUMNS) if adducts is not None
                                  else None),
                         arbitration=(cat("arbitration", AR.ARBITRATION_COLUMNS) if groups
                                      else None))


# ---------------------------------------------------------------------------
# outputs
# ---------------------------------------------------------------------------
#: The reference categorical palette (light), in its fixed order: slot 1 is the
#: target's, the next ones the co-isolates' (window order, panel A), then the
#: target's fragments (panel B) -- one colour, one ion, across both panels; it
#: never cycles.
_SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7",
           "#e34948")
_INK, _INK2, _GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _volts_axis(ax, nce: np.ndarray, volts: np.ndarray) -> None:
    """Volts as a secondary x axis (same data, a second scale) when every
    spectrum's volts are known and rise with NCE; linear beyond the ends."""
    ok = np.isfinite(nce) & np.isfinite(volts)
    if ok.sum() < 2 or not ok.all():
        return
    x, v = nce[np.argsort(nce)], volts[np.argsort(nce)]
    if not ((np.diff(x) > 0).all() and (np.diff(v) > 0).all()):
        return

    def fwd(a):
        a = np.asarray(a, dtype=float)
        lo = v[0] + (a - x[0]) * (v[1] - v[0]) / (x[1] - x[0])
        hi = v[-1] + (a - x[-1]) * (v[-1] - v[-2]) / (x[-1] - x[-2])
        return np.where(a < x[0], lo, np.where(a > x[-1], hi, np.interp(a, x, v)))

    def inv(b):
        b = np.asarray(b, dtype=float)
        lo = x[0] + (b - v[0]) * (x[1] - x[0]) / (v[1] - v[0])
        hi = x[-1] + (b - v[-1]) * (x[-1] - x[-2]) / (v[-1] - v[-2])
        return np.where(b < v[0], lo, np.where(b > v[-1], hi, np.interp(b, v, x)))
    sec = ax.secondary_xaxis("top", functions=(fwd, inv))
    sec.set_xlabel("HCD (V)", fontsize=8, color=_INK2)
    sec.tick_params(labelsize=7.5, colors=_INK2)


def render_window(w: WindowResult, path: str, *, dpi: int = 150) -> str | None:
    """breakdown_<parent>.png: (A) F of every window ion vs NCE with its F_lo..F
    band, the target emphasised and its E50 in the legend; (B) the target's
    top fragments' TIC fraction vs NCE, with the target's own. Volts on a
    secondary top axis when known. Legends below the panels, never over data."""
    if w.breakdown is None or not len(w.breakdown):
        return None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bd = w.breakdown
    order = [w.target] + list(w.co_isolates)
    tgt = bd[bd["ion"] == w.target]
    nce = tgt["nce"].to_numpy(dtype=float)
    volts = tgt["hcd_v"].to_numpy(dtype=float)
    top_axis = bool(np.isfinite(volts).all())          # room for the volts axis
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(11.5, 5.6 if top_axis else 5.1))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.76 if top_axis else 0.84, bottom=0.27,
                        wspace=0.22)
    e50 = w.e50.set_index(["ion", "curve"]) if w.e50 is not None else None
    for j, f in enumerate(order):
        g = bd[bd["ion"] == f]
        col = _SERIES[j % len(_SERIES)] if j < len(_SERIES) else _INK2
        lw, ms, z = (2.4, 6.5, 4) if f == w.target else (1.3, 4.5, 3)
        x = g["nce"].to_numpy(dtype=float)
        axa.fill_between(x, g["f_lo"].to_numpy(dtype=float), g["f"].to_numpy(dtype=float),
                         color=col, alpha=0.16, lw=0, zorder=1)
        lab = f + (" (target)" if f == w.target else "")
        if e50 is not None and (f, "F") in e50.index:
            r = e50.loc[(f, "F")]
            lab += (f": E50 NCE {r['e50_nce']:.1f}" if r["flag"] == BD.FLAG_INTERPOLATED
                    else f": E50 {r['flag'].replace('_', ' ')}")
        axa.plot(x, g["f"], color=col, lw=lw, marker="o", ms=ms, zorder=z, label=lab,
                 markeredgecolor="white", markeredgewidth=0.8)
        axa.plot(x, g["f_lo"], color=col, lw=0.8, ls=":", zorder=z - 1)
    axa.axhline(BD.E50_THRESHOLD, color=_INK2, lw=0.8, ls="--", zorder=0)
    axa.set_ylim(-0.02, 1.04)
    axa.set_xlabel("NCE", fontsize=9)
    axa.set_ylabel("survival F = I / (I + weighted fragments)", fontsize=9)
    axa.set_title("A  window ions: F (line), F_lo..F band (dotted: F_lo)", loc="left",
                  fontsize=9.5, color=_INK, pad=26 if top_axis else 6)
    # panel B: the target's fragments (weight > 0 for the target), largest first
    fr = w.fragments if w.fragments is not None else pd.DataFrame()
    mine = []
    for r in fr.itertuples():
        wt = dict(kv.split("=") for kv in r.weights.split(";")) if r.weights else {}
        if float(wt.get(w.target, 0)) > 0:
            mine.append((r.max_frac, r.fragment, r.call, float(wt[w.target])))
    mine = sorted(mine, key=lambda t: (-t[0], t[1]))[:FIGURE_TOP_FRAGMENTS]
    tic = tgt["tic"].to_numpy(dtype=float)
    # where the target is unobserved -- no precursor and no fragment it can
    # make, F_lo's 0/0 -- its line breaks; gone (fragments there) it reads 0
    own = np.where(np.isfinite(tgt["f_lo"].to_numpy(dtype=float)), tgt["frac"], np.nan)
    axb.plot(nce, own, color=_SERIES[0], lw=2.0, ls="--", marker="o", ms=5,
             markeredgecolor="white", markeredgewidth=0.8, label=f"{w.target} (target, intact)",
             zorder=4)
    specs = list(tgt["spectrum"])
    roles = w.roles                      # the fragments' names are the merged ones
    # a colour names one ion across both panels: the fragments take the
    # palette's slots after the window ions' (never a co-isolate's colour from
    # panel A), then grey with distinct dashes -- the palette never cycles
    free = list(_SERIES[len(order):])
    for j, (_, key, call, wt) in enumerate(mine):
        y = [roles.loc[(roles["spectrum"] == s) & (roles["key"] == key), "intensity"].sum()
             for s in specs] / np.where(tic > 0, tic, np.nan)
        lab = key + ("" if call == w.target else
                     (f" (shared, w {wt:.2f})" if call == AT.BASIS_AMBIGUOUS else f" ({call})"))
        col, ls = ((free[j], "-") if j < len(free) else
                   (_INK2, ((0, (4, 1.5)), (0, (1, 1.2)), (0, (5, 1.5, 1, 1.5)))[
                       (j - len(free)) % 3]))
        axb.plot(nce, y, color=col, ls=ls, lw=1.4, marker="o", ms=4.5,
                 markeredgecolor="white", markeredgewidth=0.8, label=lab, zorder=3)
    axb.set_ylim(bottom=0)
    axb.set_xlabel("NCE", fontsize=9)
    axb.set_ylabel("fraction of the spectrum's TIC", fontsize=9)
    axb.set_title(f"B  {w.target}: top fragments (TIC fraction)", loc="left", fontsize=9.5,
                  color=_INK, pad=26 if top_axis else 6)
    for ax in (axa, axb):
        ax.grid(color=_GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=8, colors=_INK2)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        _volts_axis(ax, nce, volts)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False,
                  fontsize=7.6, handlelength=2.2, columnspacing=1.4)
    fig.suptitle(f"MS2 breakdown, parent {SRC.spectrum_key(w.parent_mz)} -- target "
                 f"{w.target}, axis offset {w.offset_ppm:+.2f} ppm", x=0.07, y=0.985,
                 ha="left", fontsize=11, weight="bold", color=_INK)
    # no Software key: matplotlib's default carries its project URL, and no
    # output of `ms2 analyze` holds a URL
    fig.savefig(path, dpi=dpi, bbox_inches="tight", metadata={"Software": None})
    plt.close(fig)
    return path


def _by_order(windows) -> dict:
    """Harmonic ghosts over the windows, by order ({'2f': n, ...})."""
    out: dict[str, int] = {}
    for w in windows:
        for k, n in w.harmonics.items():
            out[k] = out.get(k, 0) + int(n)
    return dict(sorted(out.items()))


def _arbitration_counts(a: pd.DataFrame) -> dict:
    """The manifest's arbitration counts: windows arbitrated, by window
    verdict, and rivals by verdict."""
    win = a.drop_duplicates("parent_mz")
    return {"windows": int(len(win)),
            "supported": int((win["window_verdict"] == AR.SUPPORTED).sum()),
            "undecided": int((win["window_verdict"] == AR.UNDECIDED).sum()),
            "rivals": {v: int((a["verdict"] == v).sum()) for v in AR.VERDICTS}}


def manifest(res: AnalyzeResult, tables: RunTables, *, files: dict, created_utc: str | None
             ) -> dict:
    """`ms2_analyze_manifest.json`: the sample, the parameters (gate mode,
    tolerances, gas mass, isolation width and its source, the named ions), each
    window's ions, offset and notes, the counts, the inputs' and outputs' sha1
    and the code identity. No credentials, no server URL."""
    import peaky
    from peaky.reporting import provenance as PV
    em = tables.manifest
    return {
        "sample_id": em.get("sample_id", ""), "name": em.get("name", ""),
        "extract_status": em.get("status", ""),
        "energy_resolved": em.get("energy_resolved"),
        "parameters": res.params,
        "windows": [w.manifest() for w in res.windows],
        "notes": list(res.notes),
        "counts": {"windows": len(res.windows),
                   "analysed": sum(w.status == "analysed" for w in res.windows),
                   "readings_only": sum(w.status == "readings only" for w in res.windows),
                   "skipped": sum(w.status == "skipped" for w in res.windows),
                   "readings_rows": int(len(res.readings)),
                   "breakdown_rows": int(len(res.breakdown)),
                   "fragments": int(len(res.fragments)),
                   "four_piece_readings": int((res.readings["loss_pieces"] == 4).sum()),
                   "harmonic_peaks": int((res.readings["harmonic"].fillna("") != "").sum()),
                   "harmonic_by_order": _by_order(res.windows),
                   "replicate_rows": (None if res.replicates is None
                                      else int(len(res.replicates))),
                   "one_scan_energies": sum(w.one_scan_energies for w in res.windows),
                   "e50_intervals": int((res.e50["boot_n"] > 0).sum()),
                   "channel_rows": None if res.channels is None else int(len(res.channels)),
                   "classes": (None if res.classes is None else
                               {c: int((res.classes["class"] == c).sum())
                                for c in CH.CLASSES}),
                   "arbitration": (None if res.arbitration is None else
                                   _arbitration_counts(res.arbitration))},
        "inputs": dict(tables.inputs), "files": dict(files),
        "code": {"package_version": peaky.__version__, "analyze_version": __version__,
                 "git": PV.git_info(os.path.dirname(os.path.abspath(__file__))),
                 "python": sys.version.split()[0], "deps": PV.dep_versions()},
        "created_utc": created_utc,
    }


def write_outputs(res: AnalyzeResult, tables: RunTables, out_dir: str, *, figures: bool = True,
                  created_utc: str | None = None) -> dict:
    """Write the four tables (+ ms2_branching.csv; + ms2_replicates.csv when
    the folder has per-scan tables; + ms2_channels.csv and ms2_classes.csv
    when the run read adducts; + ms2_arbitration.csv when it arbitrated
    rival formulas), the per-window figures and the manifest into
    `out_dir`; -> {file name: sha1}."""
    from peaky.reporting import provenance as PV
    out = Path(os.path.expanduser(out_dir))
    out.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, df in (("ms2_readings.csv", res.readings), ("ms2_breakdown.csv", res.breakdown),
                     ("ms2_e50.csv", res.e50), ("ms2_fragments.csv", res.fragments),
                     ("ms2_branching.csv", res.branching),
                     ("ms2_replicates.csv", res.replicates),
                     ("ms2_channels.csv", res.channels), ("ms2_classes.csv", res.classes),
                     ("ms2_arbitration.csv", res.arbitration)):
        if df is None:
            # a stale table of an earlier run with per-scan tables (or with
            # adducts) is not this run's: removed, so the folder holds what
            # the manifest lists
            (out / name).unlink(missing_ok=True)
            continue
        df.to_csv(out / name, index=False)
        files[name] = PV.sha1_file(str(out / name))
    if figures:
        for w in res.windows:
            name = f"breakdown_{SRC.spectrum_key(w.parent_mz)}.png"
            if render_window(w, str(out / name)):
                files[name] = PV.sha1_file(str(out / name))
    man = manifest(res, tables, files=files, created_utc=created_utc)
    with open(out / "ms2_analyze_manifest.json", "w", encoding="utf-8") as fh:
        json.dump(man, fh, indent=2, default=str)
    return files
