"""Snapshot schema v2 and its content hash.

A snapshot is one JSON file in the reference-list catalog (data/peaklists/, or a
user directory). Schema v1 (see data/peaklists/README.md) is what peaky shipped
with: an id, a `species` list of neutral formulas and free-text references. v2
adds what a time-resilient snapshot of an EXTERNAL source needs: where it came
from (`source`: DOI, record, file, bytes, the publisher's md5, when it was
retrieved, licence), what made it (`importer`), what it may be used for
(`roles`), how the evidence scale may read a hit (`family_scope`), and a
content hash (`sha256`) over the canonical species payload, so a changed line
is detected no matter how the JSON is re-indented or re-ordered.

Pure functions: no I/O except `read_snapshot` / `write_snapshot`.
"""
from __future__ import annotations

import hashlib
import json

__version__ = "0.1.0"

SCHEMA_V2 = 2

#: what a snapshot may be used for. Only `name` and `class` are consumed today:
#: a `name` list's entries are NAMED context-list entries (level 3c,
#: levels/lists.py::ContextLists); a `class` list's are class entries (a tag).
#: `propose` (a pass-0 proposer family) and `isomer_count` (the number of known
#: structures per formula, for evidence.load_isomer_space) are RESERVED: a
#: snapshot may declare them, nothing reads them yet.
ROLES = ("name", "class", "propose", "isomer_count")

#: how the evidence scale reads a named hit: a `compound` list names the
#: neutral (3c reachable); a `class` list names a family (a tag only)
FAMILY_SCOPES = ("compound", "class")

POLARITIES = ("positive", "negative", "both")

#: the `source` block every v2 snapshot carries (all required; None allowed for
#: a field the publisher does not give, e.g. `concept_doi`)
SOURCE_FIELDS = ("kind", "doi", "concept_doi", "record_id", "version", "file", "bytes",
                 "md5", "retrieved_utc", "licence", "url")

#: the species keys the canonical payload keeps (anything else is dropped from
#: the hash, so a display-only key can be added without re-pinning)
SPECIES_KEYS = ("formula", "name", "inchikey", "inchikey_block", "cas", "ids", "radical")


def canonical_species(species) -> list:
    """The species list reduced to `SPECIES_KEYS`, sorted by (formula, inchikey,
    name) -- the payload the hash is taken over."""
    out = []
    for s in species or ():
        row = {k: s.get(k) for k in SPECIES_KEYS if s.get(k) is not None}
        if "ids" in row and isinstance(row["ids"], dict):
            row["ids"] = {k: v for k, v in sorted(row["ids"].items()) if v is not None}
        out.append(row)
    out.sort(key=lambda r: (str(r.get("formula", "")), str(r.get("inchikey", "")), str(r.get("name", ""))))
    return out


def canonical_species_payload(species) -> bytes:
    """`canonical_species` as compact, key-sorted UTF-8 JSON."""
    return json.dumps(canonical_species(species), sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def species_sha256(species) -> str:
    return hashlib.sha256(canonical_species_payload(species)).hexdigest()


def schema_version(d: dict) -> int:
    try:
        return int(d.get("schema_version", 1) or 1)
    except (TypeError, ValueError):
        return 1


def validate_snapshot(d: dict) -> list[str]:
    """Every reason `d` is not a valid snapshot (empty when it is). A v1 file
    (no `schema_version`, or 1) is accepted as it always was: the loader's own
    checks apply. A v2 file must carry the full provenance block and a content
    hash that matches its species."""
    errs: list[str] = []
    if not d.get("id"):
        errs.append("missing id")
    if schema_version(d) < SCHEMA_V2:
        return errs
    roles = d.get("roles")
    if not isinstance(roles, (list, tuple)) or not roles:
        errs.append("roles: a non-empty list is required")
    else:
        bad = [r for r in roles if r not in ROLES]
        if bad:
            errs.append(f"roles: unknown {bad} (allowed {list(ROLES)})")
    src = d.get("source")
    if not isinstance(src, dict):
        errs.append("source: a dict is required")
    else:
        for k in SOURCE_FIELDS:
            if k not in src:
                errs.append(f"source.{k} missing")
    imp = d.get("importer")
    if not isinstance(imp, dict) or not imp.get("name") or not imp.get("version"):
        errs.append("importer: {name, version} required")
    if d.get("family_scope") not in FAMILY_SCOPES:
        errs.append(f"family_scope: one of {list(FAMILY_SCOPES)} required")
    pol = d.get("polarity")
    if pol not in POLARITIES:
        errs.append(f"polarity: one of {list(POLARITIES)} required")
    if not isinstance(d.get("species"), list):
        errs.append("species: a list is required")
    else:
        for i, s in enumerate(d["species"]):
            if not isinstance(s, dict) or not s.get("formula"):
                errs.append(f"species[{i}]: formula required")
                break
    want = d.get("sha256")
    if not want:
        errs.append("sha256 missing")
    elif isinstance(d.get("species"), list):
        have = species_sha256(d["species"])
        if have != want:
            errs.append(f"sha256 mismatch: file says {want[:12]}..., species hash {have[:12]}...")
    return errs


def read_snapshot(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def write_snapshot(path: str, d: dict) -> str:
    """Write `d` as a snapshot (indent 1, UTF-8, trailing newline). The hash is
    over the canonical payload, so the layout written here never matters."""
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(d, fh, indent=1, ensure_ascii=False)
        fh.write("\n")
    return path
