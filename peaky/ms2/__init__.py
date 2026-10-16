"""MS2: fragment assignment for isolated precursor ions.

This package is the pure core (no Mascope, no I/O): `subformula` reads each
MS2 peak as a sub-formula of every ion in the isolation window and tests the
neutral loss it implies. See docs/MS2.md.
"""
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
]
