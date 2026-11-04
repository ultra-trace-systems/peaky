"""The run manifest's `lists` block: the reference-list snapshots a run read,
pinned beside the code and the data."""
import json
import os

from peaky.assignment import reflists as RL
from peaky.assignment.passes import PassConfig
from peaky.lists.importers import epacons as E
from peaky.reporting import provenance as PV


def test_build_manifest_carries_the_lists_block(tmp_path):
    lists, _tags, rec = RL.activate("ambient-air", "uronium", record=True, include=E.LIST_ID)
    record = RL.lists_record(lists, rec)
    m = PV.build_manifest(run_dir=str(tmp_path), batch_name="b", dataset="d", sample_ids=["s"],
                          reagent="Ur", cfg=PassConfig(), lists=record)
    p = PV.write_manifest(str(tmp_path), m)
    back = json.load(open(p))
    ids = {x["id"]: x for x in back["lists"]}
    assert set(ids) == {"contaminants_keller2008", E.LIST_ID}
    me = ids[E.LIST_ID]
    assert me["pinned"] is True and me["version"] == E.VERSION and len(me["sha256"]) == 64
    assert me["roles"] == ["name"] and me["how_activated"] == "explicit (--lists)" and me["schema_version"] == 2
    assert ids["contaminants_keller2008"]["how_activated"] == "always active"
    # record_run passes it through (best-effort wrapper)
    m2 = PV.record_run(run_dir=str(tmp_path), base_out=str(tmp_path), batch_name="b", dataset="d",
                       sample_ids=["s"], reagent="Ur", cfg=PassConfig(), lists=record, log=lambda *_: None)
    assert [x["id"] for x in m2["lists"]] == [x["id"] for x in record]
    assert os.path.exists(os.path.join(tmp_path, "index.jsonl"))


def test_no_lists_means_an_empty_block(tmp_path):
    m = PV.build_manifest(run_dir=str(tmp_path), batch_name="b", dataset=None, sample_ids=[],
                          reagent="Br", cfg=PassConfig())
    assert m["lists"] == []
