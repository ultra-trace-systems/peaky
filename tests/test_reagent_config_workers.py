"""A --reagent-config profile reaches the batch's spawned assign workers.

`peaky batch --reagent X --reagent-config f.json --jobs N` registered X in the
parent only: the pool's workers are fresh interpreters (spawn) whose registry
holds the built-ins, so the first per-file stage that resolved the run's
reagent by name (the evidence space) raised `KeyError: unknown reagent 'X'`
in every worker; --jobs 1 ran. The parent now hands its added profiles to
every worker (`profiles.registry_extras` -> `assign_batch._worker_init`).

Pins: what `registry_extras` carries; a REAL spawned worker resolves the
config profile once handed them (and does not without -- the control that
shows the worker's registry is its own); and `assign_batch.run --jobs 2`
ships them through the pool's initargs to workers that start from the
built-ins only.

Run: pytest tests/test_reagent_config_workers.py -q
"""

from __future__ import annotations

import concurrent.futures as CF
import dataclasses
import json
import multiprocessing as mp
import pickle

import pandas as pd
import pytest

from peaky.assignment import assign as A
from peaky.assignment import ledger as L
from peaky.assignment import tiers as T
from peaky.batch import assign_batch as AB
from peaky.chem import chemistry as C
from peaky.chem import profiles as P
from peaky.io import io_mascope as IO

NAME = "BrCustom"


@pytest.fixture(autouse=True)
def _registry():
    """Each test leaves the process registry as it found it."""
    saved = (dict(P.PROFILES), dict(P._BY_ALIAS))
    yield
    P.PROFILES.clear(); P.PROFILES.update(saved[0])
    P._BY_ALIAS.clear(); P._BY_ALIAS.update(saved[1])


def _builtins_only():
    """The registry a freshly imported module holds (a spawned worker's)."""
    P.PROFILES.clear(); P.PROFILES.update(P._BUILTIN_PROFILES)
    P._BY_ALIAS.clear(); P._BY_ALIAS.update(P._BUILTIN_ALIAS)


def _config(tmp_path, name=NAME, base=P.BR, **over):
    """A one-profile JSON config: `base`'s fields under a new name."""
    entry = {k: v for k, v in dataclasses.asdict(base).items() if k in P._CONFIG_FIELDS}
    entry.update({"name": name, "aliases": [], **over})
    path = tmp_path / "reagents.json"
    path.write_text(json.dumps([entry]))
    return str(path)


# --------------------------------------------------------------------------- the extras
def test_the_extras_carry_what_a_config_added_and_no_builtin(tmp_path):
    assert P.registry_extras() == ({}, {})
    P.load_config(_config(tmp_path, aliases=["brc"]))
    profs, aliases = P.registry_extras()
    assert list(profs) == [NAME]
    assert set(aliases) == {NAME.lower(), "brc"}
    assert all(p is profs[NAME] for p in aliases.values())


def test_the_extras_carry_a_replaced_builtin(tmp_path):
    P.load_config(_config(tmp_path, name="Br", purity=0.5))
    profs, aliases = P.registry_extras()
    assert list(profs) == ["Br"] and profs["Br"].purity == 0.5
    assert aliases["br"] is profs["Br"]


def test_registering_the_extras_rebuilds_the_lookup_in_a_fresh_registry(tmp_path):
    prof = P.resolve(NAME, config=_config(tmp_path))
    extras = pickle.loads(pickle.dumps(P.registry_extras()))   # as spawn ships them
    _builtins_only()
    with pytest.raises(KeyError, match="unknown reagent"):
        P.resolve(NAME)
    P.register_extras(extras)
    got = P.resolve(NAME)
    assert got == prof and P.PROFILES[NAME] is got
    assert P.resolve(f"{NAME}+NO3").adducts == P.compose([prof, P.NO3]).adducts
    P.register_extras(None)    # no-op
    assert P.resolve(NAME) is got


# --------------------------------------------------------------------------- a real spawned worker
def _spawned_resolve(initargs, name):
    with CF.ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn"),
                                initializer=AB._worker_init, initargs=initargs) as ex:
        return ex.submit(P.resolve, name).result()


def test_a_spawned_worker_resolves_the_config_profile(tmp_path):
    prof = P.resolve(NAME, config=_config(tmp_path, base=P.NO3_15N))
    initargs = ("ambient-air", None, {"reagent_profile": prof.name}, None)
    # the control: a spawned worker's registry is its own -- the parent's
    # registration does not reach it (the reported failure)
    with pytest.raises(KeyError, match="unknown reagent"):
        _spawned_resolve(initargs, NAME)
    got = _spawned_resolve(initargs + (P.registry_extras(),), NAME)
    assert got == prof


# --------------------------------------------------------------------------- assign_batch.run --jobs 2
class _FreshRegistryPool:
    """A ProcessPoolExecutor stand-in that runs the REAL worker entry points
    in this process, but as a spawned worker sees the registry: built-ins
    only, plus what the pickled initargs bring. The parent's registry is put
    back on exit."""

    def __init__(self, max_workers=None, mp_context=None, initializer=None, initargs=()):
        self._parent = (dict(P.PROFILES), dict(P._BY_ALIAS))
        _builtins_only()
        initializer(*pickle.loads(pickle.dumps(initargs)))

    def __enter__(self):
        return self

    def __exit__(self, *a):
        P.PROFILES.clear(); P.PROFILES.update(self._parent[0])
        P._BY_ALIAS.clear(); P._BY_ALIAS.update(self._parent[1])
        return False

    def submit(self, fn, *args):
        f = CF.Future()
        try:
            f.set_result(fn(*args))
        except Exception as exc:  # noqa: BLE001 -- surfaced by fut.result(), as a real pool does
            f.set_exception(exc)
        return f


def _ledger() -> pd.DataFrame:
    """A finished single-sample ledger: one Br- adduct reading and the reagent ion."""
    peaks = pd.DataFrame({"peak_id": ["A", "E"],
                          "mz": [C.ion_mz("C10H16O4", "[M+Br]-"), 78.9189],
                          "height": [1.0e5, 1.0e4]})
    led = L.new_ledger(peaks)
    L.commit_assignment(led, "A", neutral_formula="C10H16O4", adduct="[M+Br]-",
                        ion_formula="C10H16O4Br-", ion_score=0.95, compound_score=0.95,
                        ppm_error=-0.3, pass_no=1, method="cheminfo", confidence="High",
                        commentary="Pass 1: C10H16O4 [M+Br]-")
    L.mark_reagent(led, "E", "reagent ion: [Br]-")
    T.apply_tiers(led)
    return led


def _batch_table():
    t0 = pd.Timestamp("2025-10-01 21:00:00", tz="UTC")
    bg = list(range(100, 120))
    rows = []
    for i in range(4):
        for j, sid in enumerate((f"a{i}", f"b{i}")):
            t = t0 + pd.Timedelta(minutes=10 * (2 * i + j))
            rows += [dict(sample_item_id=sid, sample_item_name=f"n_{sid}", datetime_utc=t,
                          mz=float(mz), height=500.0)
                     for mz in bg + list(range(200 + 20 * i, 220 + 20 * i))]
    return pd.DataFrame(rows)


def test_the_batch_pool_hands_its_workers_the_config_profile(tmp_path, monkeypatch):
    prof = P.resolve(NAME, config=_config(tmp_path))       # what pipeline.run_batch does
    seen = []

    def fake_assign(sid, context="ambient-air", **kw):
        # the lookup the per-file evidence space makes (levels.space.Space)
        seen.append(P.resolve(kw["reagent_profile"]))
        return {"ledger": _ledger(), "stats": {"noise_edge_cps": 4.0, "height_gate_cps": 10.0,
                                               "degeneracy_cal": {"mu": 0.0, "sigma": 0.5}},
                "plausibility_audit": [], "summaries": {}, "problems": []}

    monkeypatch.setattr(IO, "connect", lambda *a, **k: "CLIENT")
    monkeypatch.setattr(IO, "fetch_peaks", lambda client, sid, use_cache=True: pd.DataFrame(
        {"peak_id": ["A"], "mz": [200.1], "height": [1.0e5]}))
    monkeypatch.setattr(IO, "estimate_offset", lambda raw: 0.0)
    monkeypatch.setattr(A, "run", fake_assign)
    monkeypatch.setattr(CF, "ProcessPoolExecutor", _FreshRegistryPool)
    monkeypatch.delenv("PEAKY_MATCH_WORKERS", raising=False)
    lines = []
    pk = _batch_table()
    AB.run(peaks=pk, ts_peaks=pk, reagent=NAME, batch="test batch", out_dir=str(tmp_path / "run"),
           k_min=2, k_max=3, min_gain=0.0, n_jobs=2, resolving_power=120_000.0,
           log=lines.append)
    assert any("parallel: 2 worker processes" in ln for ln in lines)
    assert seen and all(p == prof for p in seen)
    assert P.resolve(NAME) is prof                           # the parent's registry is untouched
    summ = json.loads((tmp_path / "run" / "batch_summary.json").read_text())
    assert summ["reagent"] == NAME
