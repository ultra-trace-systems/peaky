"""peaky — multi-pass chemical-formula assignment for high-resolution
mass-spec peaks stored in Mascope.

SDK-native, test-driven. Mascope's scoring maths is the scorer (run in-process by
default via ``mascope_tools``; the network ``match_compounds`` is the opt-in
fallback); this package owns candidate generation, chemistry plausibility, series
logic and arbitration. State lives in one mutable ledger DataFrame (one row per
peak); passes read it and annotate columns in place.

Modules are grouped into sub-packages by responsibility::

    peaky/
      io/         Mascope I/O + the in-process scoring backend
      chem/       masses, isotopes, reagent profiles, contexts
      assignment/ the ledger, passes, arbitration, tiers, cleanup, ...
      batch/      sample selection, merge, time-series, clustering
      reporting/  Van Krevelen, GKA, Excel/markdown, PDF, provenance
      cli.py · paths.py · pipeline.py   (entry points + the orchestration spine)

Public API (import from the package root; internals may move):

    import peaky as ma
    ma.run(sample_id, context)        # single-sample assignment -> dict
    ma.run_batch(batch=..., ...)      # FULL batch pipeline
    ma.PassConfig(...)                # assignment knobs
    ma.get_context("ambient-air") · ma.resolve_reagent("auto", peaks) · ma.build_report(...)

`import peaky` stays cheap: both the API helpers AND the sub-package modules are
resolved lazily via PEP 562 ``__getattr__``, so ``from peaky import chemistry``
still works without eagerly importing matplotlib or the Mascope SDK.
"""
from __future__ import annotations

def _package_version() -> str:
    """The installed distribution's version, which is the one that ships.

    A literal here goes stale the moment pyproject is bumped, and it is not a
    private detail: a run published into Mascope stamps it as the engine version,
    so a stale one makes two different peaky releases indistinguishable in the
    store. Falls back to "unknown" for a source tree that was never installed.
    """
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("mascope-peaky")
    except PackageNotFoundError:  # a bare checkout, not an install
        return "unknown"


__version__ = _package_version()

# public API name -> (dotted submodule path, attribute)
_LAZY = {
    "run": ("assignment.assign", "run"),
    "run_batch": ("pipeline", "run_batch"),
    "run_assign_batch": ("batch.assign_batch", "run"),
    "run_pipeline": ("pipeline", "run_batch"),  # back-compat alias
    "PassConfig": ("assignment.passes", "PassConfig"),
    "get_context": ("chem.contexts", "get_context"),
    "resolve_reagent": ("chem.profiles", "resolve"),
    "ReagentProfile": ("chem.profiles", "ReagentProfile"),
    "build_report": ("reporting.pdf_report", "build"),
}

# short module name -> dotted path under peaky, so `from peaky import <module>`
# keeps resolving after the sub-package move (these are NOT submodules of the root).
_MODULES = {
    "io_mascope": "io.io_mascope", "local_scoring": "io.local_scoring",
    "curate": "io.curate",
    "chemistry": "chem.chemistry", "isotopes": "chem.isotopes",
    "reagents": "chem.reagents", "profiles": "chem.profiles", "contexts": "chem.contexts",
    "ledger": "assignment.ledger", "passes": "assignment.passes",
    "series_gka": "assignment.series_gka", "series_detect": "assignment.series_detect",
    "ladders": "assignment.ladders", "residual": "assignment.residual",
    "labeled": "assignment.labeled",
    "siloxane": "assignment.siloxane", "cleanup": "assignment.cleanup",
    "degeneracy": "assignment.degeneracy", "tiers": "assignment.tiers",
    "plausibility": "assignment.plausibility", "reflists": "assignment.reflists",
    "assign": "assignment.assign",
    "sampling": "batch.sampling", "assign_batch": "batch.assign_batch",
    "timeseries": "batch.timeseries", "clustering": "batch.clustering",
    "cluster": "batch.cluster", "composition": "batch.composition",
    "analyte_viz": "reporting.analyte_viz", "gka_figure": "reporting.gka_figure",
    "gka_widget": "reporting.gka_widget", "qc_figure": "reporting.qc_figure",
    "pdf_report": "reporting.pdf_report", "report": "reporting.report",
    "provenance": "reporting.provenance",
}

__all__ = ["__version__", *sorted(_LAZY), *sorted(_MODULES)]


def __getattr__(name: str):
    import importlib

    if name in _LAZY:
        mod, attr = _LAZY[name]
        return getattr(importlib.import_module(f".{mod}", __name__), attr)
    if name in _MODULES:
        return importlib.import_module(f".{_MODULES[name]}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
