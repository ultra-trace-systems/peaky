"""Importers: the only code that knows a source's URL and file format.

An importer runs OFFLINE from any peaky run (`peaky lists import <source>`):
it fetches one published file, verifies the publisher's checksum, asserts the
exact header it was written for (a changed column fails loudly here, never
inside a run), converts the rows to a schema-v2 snapshot and pins it in the
destination directory's lock. Each importer is registered here by source id.
"""
from __future__ import annotations

from peaky.lists.importers import epacons

IMPORTERS = {
    epacons.SOURCE_ID: epacons,
}


class ImporterError(RuntimeError):
    """A source did not look like what the importer was written for."""


__all__ = ["IMPORTERS", "ImporterError"]
