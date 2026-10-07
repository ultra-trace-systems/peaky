"""The reagent halogen of the declared channels (C43) on the evidence scale.

`evidence.channel_halogen` names the reagent halogen from a run's DECLARED
channels; the count of committed cluster adducts (`detect_reagent_halogen`) is
the fallback for a caller that knows none. On the evidence scale of peaky
0.10.0 the halogen feeds two readers, and both must get the halogen the run's
own stage used:

- the merge vote's private class, computed per file in the PARENT
  (`assign_batch._apply` -> `evidence.vote_classes`): the halogen the file's
  own run read rides back in its stats (`reagent_halogen`);
- the pair facts (`levels.source.pair_facts`): per file the run's declared
  channels (`evidence.file_run_inputs`), pooled the batch profile's channels
  (recorded as `batch_summary["reagent_halogen"]`), post-hoc that record, else
  the declared channels of the run's profile, else the count.
"""
from __future__ import annotations

import json

import pandas as pd
import pytest

from peaky.assignment import assign as A
from peaky.assignment import evidence as EV
from peaky.assignment import ledger as L
from peaky.assignment import tiers as T
from peaky.assignment.levels import source as SRC
from peaky.batch import assign_batch as AB
from peaky.chem import chemistry as C
from peaky.chem import profiles as PR
from peaky.io import io_mascope as IO


# --------------------------------------------------------------------------- the rule
@pytest.mark.parametrize("summary,halogen", [
    ({"reagent_halogen": "Br", "reagent": "NO3"}, "Br"),       # the record wins
    ({"reagent_halogen": None, "reagent": "Br"}, None),        # ... a recorded None too
    ({"reagent": "Br+NO3"}, "Br"),                             # a run before the record: its profile
    ({"reagent": "NO3+NO3_15N"}, None),
    ({"reagent": "Ur"}, None),
    ({"reagent": "no such profile"}, EV.DETECT_HALOGEN),       # no declared channels: count
    ({}, EV.DETECT_HALOGEN),
])
def test_a_source_reads_the_recorded_halogen_else_its_profile_s_channels(summary, halogen):
    assert SRC.reagent_halogen(summary) == halogen


def test_a_profile_s_halogen_is_its_declared_channels():
    for name in ("Br", "Br+NO3", "NO3", "NO3+NO3_15N", "Ur", "I"):
        prof = PR.resolve(name)
        assert SRC.reagent_halogen({"reagent": prof.name}) == EV.channel_halogen(prof.adducts)


def test_a_file_levelled_alone_carries_the_run_s_halogen():
    ri = EV.file_run_inputs(sample_id="s", reagent="NO3", context=None, reagent_halogen="Br")
    assert SRC.reagent_halogen(ri.summary) == "Br"
    assert EV.file_run_inputs(sample_id="s", reagent="NO3", context=None).summary["reagent_halogen"] \
        == EV.DETECT_HALOGEN


def _spy_level_pairs(monkeypatch):
    seen = []
    real = EV._level_pairs

    def spy(frames, **kw):
        seen.append((bool(kw.get("with_files")), kw.get("halogen", "<default>")))
        return real(frames, **kw)

    monkeypatch.setattr(EV, "_level_pairs", spy)
    return seen


def test_the_pair_facts_read_the_source_s_halogen(monkeypatch):
    led = _ledger()
    seen = _spy_level_pairs(monkeypatch)
    one = EV.source_from_frames({"f1": led}, run_inputs=EV.RunInputs(summary={"reagent": "Br+NO3"}),
                                mode="adapted")
    SRC.pair_facts(one)
    pooled = EV.source_from_frames({"f1": led, "f2": led},
                                   run_inputs=EV.RunInputs(summary={"reagent": "NO3", "reagent_halogen": "Br"}))
    SRC.pair_facts(pooled)
    assert seen == [(False, "Br"), (True, "Br")]


# --------------------------------------------------------------------------- the batch
_F = "C10H16O5"


def _ledger():
    led = L.new_ledger(pd.DataFrame([("p1", C.ion_mz(_F, "[M-H]-"), 1.0e5)],
                                    columns=["peak_id", "mz", "height"]))
    L.commit_assignment(led, "p1", neutral_formula=_F, adduct="[M-H]-",
                        ion_formula="C10H15O5-", ion_score=0.9, compound_score=0.9,
                        ppm_error=0.1, pass_no=1, method="cheminfo+grid",
                        confidence="High", commentary="stub")
    T.apply_tiers(led)
    return led


#: what each stand-in file's own run read: one declared a bromide channel set,
#: one a halogen-free one, one recorded nothing (a run before the record)
_FILE_HALOGEN = {"a0": "Br", "a1": None}


def _fake_assign(sid, context="ambient-air", **kw):
    stats = {"noise_edge_cps": 4.0, "height_gate_cps": 10.0}
    if sid in _FILE_HALOGEN:
        stats["reagent_halogen"] = _FILE_HALOGEN[sid]
    return {"ledger": _ledger(), "stats": stats, "plausibility_audit": [], "summaries": {}, "problems": []}


def test_the_batch_feeds_each_reader_the_halogen_its_stage_used(monkeypatch, tmp_path):
    t0 = pd.Timestamp("2025-10-01 21:00:00", tz="UTC")
    peaks = pd.DataFrame([dict(sample_item_id=sid, sample_item_name=f"n_{sid}",
                               datetime_utc=t0 + pd.Timedelta(minutes=10 * i), mz=float(mz), height=500.0)
                          for i, sid in enumerate(("a0", "a1", "a2")) for mz in range(100, 120)])
    monkeypatch.setattr(IO, "connect", lambda *a, **k: "CLIENT")
    monkeypatch.setattr(IO, "fetch_peaks", lambda client, sid, use_cache=True: pd.DataFrame(
        {"peak_id": ["p1"], "mz": [C.ion_mz(_F, "[M-H]-")], "height": [1.0e5]}))
    monkeypatch.setattr(IO, "estimate_offset", lambda raw: 0.0)
    monkeypatch.setattr(A, "run", _fake_assign)
    votes = []
    real_vote = EV.vote_classes

    def vote_spy(frame, **kw):
        votes.append(kw.get("halogen", "<default>"))
        return real_vote(frame, **kw)

    monkeypatch.setattr(EV, "vote_classes", vote_spy)
    seen = _spy_level_pairs(monkeypatch)
    AB.run(peaks=peaks, ts_peaks=peaks, reagent="Br", batch="test batch", out_dir=str(tmp_path),
           k_min=3, k_max=3, min_gain=0.0, n_jobs=1, log=lambda *a: None)
    run_dir = next(d for d in tmp_path.iterdir() if (d / "batch_summary.json").is_file()) \
        if not (tmp_path / "batch_summary.json").is_file() else tmp_path
    summ = json.load(open(run_dir / "batch_summary.json"))
    # the vote: each file at the halogen its own run read (a file that recorded
    # none falls back to the count, as before the record)
    order = summ["sample_ids"]
    assert votes == [_FILE_HALOGEN.get(s, EV.DETECT_HALOGEN) for s in order]
    assert [pf.get("reagent_halogen", "<none>") for pf in summ["per_file"]] \
        == [_FILE_HALOGEN.get(s, "<none>") for s in order]
    # the pooled pair facts: the batch profile's declared channels, recorded
    # (the stand-in ledgers hold no cluster adduct: a count would say none)
    assert summ["reagent_halogen"] == EV.channel_halogen(PR.resolve("Br").adducts) == "Br"
    pooled = [h for wf, h in seen if wf]
    assert pooled and set(pooled) == {"Br"}
    # post-hoc: the run dir re-levelled reads the same record
    assert SRC.reagent_halogen(SRC.source_from_run_dir(str(run_dir)).summary) == "Br"
