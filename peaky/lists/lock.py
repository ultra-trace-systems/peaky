"""The catalog lockfile: which snapshot, at which content hash, a catalog
directory pins.

`catalog.lock` sits beside the snapshots it pins (data/peaklists/catalog.lock
for the bundled catalog, <user dir>/catalog.lock for a user's). It is written
by `peaky lists lock` / `peaky lists import` and read by the runtime loader,
which recomputes each snapshot's `species_sha256` and refuses a mismatch
(`SnapshotIntegrityError`). Updating a list = a new snapshot + a lock bump in
one commit; the run manifest records the (id, version, sha256) it resolved, so
an old run names exactly the content it read.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone

from peaky.lists.snapshot import schema_version, species_sha256

__version__ = "0.1.0"

LOCK_NAME = "catalog.lock"
LOCK_SCHEMA = 1

#: where non-bundled snapshots live: $PEAKY_LISTS_DIR, else ~/.cache/peaky/lists
LISTS_DIR_ENV = "PEAKY_LISTS_DIR"
DEFAULT_USER_DIR = os.path.join("~", ".cache", "peaky", "lists")


class SnapshotIntegrityError(RuntimeError):
    """A snapshot's content does not match its own `sha256` or the lock's."""


def user_lists_dir() -> str:
    return os.path.expanduser(os.environ.get(LISTS_DIR_ENV) or DEFAULT_USER_DIR)


def lock_path(directory: str) -> str:
    return os.path.join(directory, LOCK_NAME)


def read_lock(directory: str) -> dict:
    """The lock of `directory`; an empty lock when there is none."""
    p = lock_path(directory)
    if not os.path.exists(p):
        return {"schema": LOCK_SCHEMA, "generated_utc": None, "lists": {}}
    with open(p, encoding="utf-8") as fh:
        d = json.load(fh)
    d.setdefault("lists", {})
    return d


def write_lock(directory: str, lock: dict) -> str:
    lock = dict(lock)
    lock["schema"] = LOCK_SCHEMA
    lock["generated_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    lock["lists"] = {k: lock["lists"][k] for k in sorted(lock.get("lists", {}))}
    p = lock_path(directory)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(lock, fh, indent=1, sort_keys=False)
        fh.write("\n")
    return p


def snapshot_version(d: dict) -> str:
    """A v2 snapshot's version is its source version; a v1 list's its `data_version`."""
    if schema_version(d) >= 2 and isinstance(d.get("source"), dict) and d["source"].get("version"):
        return str(d["source"]["version"])
    return str(d.get("data_version", ""))


def lock_entry(d: dict, filename: str) -> dict:
    src = d.get("source") if isinstance(d.get("source"), dict) else {}
    return {
        "version": snapshot_version(d),
        "sha256": species_sha256(d.get("species", [])),
        "file": os.path.basename(filename),
        "source_doi": src.get("doi"),
        "roles": list(d.get("roles") or ()),
        "schema_version": schema_version(d),
    }


def update_lock(lock: dict, d: dict, filename: str) -> dict:
    lock.setdefault("lists", {})[d["id"]] = lock_entry(d, filename)
    return lock


def relock_directory(directory: str) -> dict:
    """A fresh lock over every *.json snapshot in `directory` (nothing written)."""
    lock = {"schema": LOCK_SCHEMA, "generated_utc": None, "lists": {}}
    for name in sorted(os.listdir(directory)):
        if not name.endswith(".json"):
            continue
        with open(os.path.join(directory, name), encoding="utf-8") as fh:
            d = json.load(fh)
        if d.get("id"):
            update_lock(lock, d, name)
    return lock


def verify(d: dict, entry: dict | None, *, name: str) -> str:
    """Check snapshot `d` (from file `name`) against its own `sha256` (v2) and
    the lock `entry` (None when unpinned). Returns the content hash. Raises
    `SnapshotIntegrityError` naming the list on any mismatch."""
    have = species_sha256(d.get("species", []))
    own = d.get("sha256")
    if own and own != have:
        raise SnapshotIntegrityError(
            f"list {d.get('id')!r} ({name}): its species hash to {have[:12]}... but the file "
            f"declares sha256 {str(own)[:12]}... -- the snapshot was edited after it was written")
    if entry is not None and entry.get("sha256") != have:
        raise SnapshotIntegrityError(
            f"list {d.get('id')!r} ({name}): species hash {have[:12]}... but {LOCK_NAME} pins "
            f"{str(entry.get('sha256'))[:12]}... (lock version {entry.get('version')!r}, "
            f"file version {snapshot_version(d)!r}) -- re-import the list or run `peaky lists lock`")
    return have
