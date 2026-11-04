"""Suspect-list snapshots: pinned, checksummed copies of external chemical lists
(NORMAN-SLE, PubChemLite, ...) that peaky reads OFFLINE.

The rule: a run never touches the network. An IMPORTER (`peaky lists import`)
downloads a source once, verifies the publisher's checksum, converts it to a
snapshot (`snapshot.py`, schema v2 of data/peaklists/*.json) and pins it in a
LOCKFILE (`lock.py`, data/peaklists/catalog.lock). The runtime loader
(`peaky.assignment.reflists.load_catalog`) verifies every snapshot against the
lock and records id / version / sha256 in the run manifest, so a run re-executed
years later reads exactly the list it read the first time -- or refuses, naming
the list.
"""
