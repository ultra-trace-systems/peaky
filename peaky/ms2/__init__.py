"""MS2: fragment assignment for isolated precursor ions.

The pure core (no Mascope, no I/O): `subformula` reads each MS2 peak as a
sub-formula of every ion in the isolation window and tests the neutral loss it
implies; `readings` reads one window's spectra across energies (and gates a
fourth loss piece on a ladder); `attribute` decides which window ion made each
fragment; `breakdown` turns the window into survival curves, E50s and an ion
budget. `sources` reads Mascope's MS2 endpoints into tables (`peaky ms2 list |
extract`); `analyze` runs the core on one extracted folder (`peaky ms2
analyze`). See docs/MS2.md.
"""
from peaky.ms2.attribute import (
    BRANCHING_FLAG,
    INTACT_FRACTION,
    attribute_fragments,
    branching,
    branching_breach,
    held_at_lowest,
    intact_test,
    merge_fragment_peaks,
    peak_roles,
    share_weights,
)
from peaky.ms2.breakdown import N2_MASS, E50, e50, e_cm, ion_budget, onset, survival
from peaky.ms2.readings import (
    GATE_LADDER,
    GATE_STRICT,
    GATE_UNGATED,
    READINGS_COLUMNS,
    gate_mode,
    ladder_gate,
    read_window,
)
from peaky.ms2.subformula import (
    CLOSED_SHELL_LOSSES,
    K_MAX_LIMIT,
    OUTPUT_COLUMNS,
    RADICAL_LOSSES,
    AnchorResidual,
    FragmentCandidate,
    LossDecomposition,
    MassOffset,
    apply_offset,
    assign_fragments,
    best_readings,
    decompose_loss,
    format_ion,
    fragment_candidates,
    fragment_ok,
    ion_parity,
    mass_offset,
    parse_ion,
)

__all__ = [
    "CLOSED_SHELL_LOSSES", "RADICAL_LOSSES", "K_MAX_LIMIT", "OUTPUT_COLUMNS",
    "AnchorResidual", "FragmentCandidate", "LossDecomposition", "MassOffset",
    "apply_offset", "assign_fragments", "best_readings", "decompose_loss",
    "format_ion", "fragment_candidates", "fragment_ok", "ion_parity",
    "mass_offset", "parse_ion",
    # readings across energies + the ladder gate
    "GATE_LADDER", "GATE_STRICT", "GATE_UNGATED", "READINGS_COLUMNS", "gate_mode",
    "ladder_gate", "read_window",
    # attribution
    "INTACT_FRACTION", "BRANCHING_FLAG", "attribute_fragments", "branching", "branching_breach",
    "held_at_lowest", "intact_test", "merge_fragment_peaks", "peak_roles", "share_weights",
    # breakdown
    "N2_MASS", "E50", "e50", "e_cm", "ion_budget", "onset", "survival",
]
