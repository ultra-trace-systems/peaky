"""Snapshot schema v2, the content hash, the catalog lock and the loader's
verification (peaky.lists + reflists.load_catalog).

The rule under test: a list is read exactly as pinned or not at all. The hash is
over the canonical species payload, so re-indenting or re-ordering a file
changes nothing, while one changed byte of one species is refused by name.
"""
import json
import os
import random
import warnings

import pytest

from peaky.assignment import reflists as RL
from peaky.lists import lock as LK
from peaky.lists import snapshot as SN
from peaky.lists.lock import SnapshotIntegrityError

MALATHION = {"formula": "C10H19O6PS2", "name": "Malathion", "inchikey": "JXSJBGJIGXNWCI-UHFFFAOYSA-N",
             "inchikey_block": "JXSJBGJIGXNWCI", "cas": "121-75-5", "ids": {"dtxsid": "DTXSID4020791", "cid": 4004},
             "radical": False}
TEXANOL = {"formula": "C12H24O3", "name": "Texanol", "inchikey": "AHHWIHXENZJRFG-UHFFFAOYSA-N",
           "inchikey_block": "AHHWIHXENZJRFG", "cas": "25265-77-4", "ids": {"dtxsid": "DTXSID7027504"},
           "radical": False}
XYLENE = {"formula": "C8H10", "name": "o-Xylene", "inchikey": "CTQNGGLPUBDAKN-UHFFFAOYSA-N",
          "inchikey_block": "CTQNGGLPUBDAKN", "cas": "95-47-6", "ids": {"dtxsid": "DTXSID3021807"}, "radical": False}
ETHYLBENZENE = dict(XYLENE, name="Ethylbenzene", inchikey="YNQLUTRBYVCPMQ-UHFFFAOYSA-N",
                    inchikey_block="YNQLUTRBYVCPMQ", cas="100-41-4", ids={"dtxsid": "DTXSID3020596"})


def v2(species, **over) -> dict:
    d = {
        "schema_version": 2, "id": "test_list_v2", "system": "test", "label": "a test snapshot",
        "data_version": "ignored on v2", "polarity": "both", "native_detection": "",
        "always_active": False, "applies_to_contexts": ["test_ctx"],
        "roles": ["name"], "family_scope": "compound",
        "source": {"kind": "norman-sle", "doi": "10.5281/zenodo.1", "concept_doi": "10.5281/zenodo.0",
                   "record_id": 1, "version": "S99.0.0.1", "file": "x.csv", "bytes": 3, "md5": "0" * 32,
                   "retrieved_utc": "2026-11-04T00:00:00Z", "licence": "CC-BY-4.0", "url": "https://example.org/x.csv"},
        "importer": {"name": "test", "version": "0.0.1"},
        "references": [{"authors": "T. Est", "title": "a test", "year": 2026}],
        "n_species": len(species), "species": list(species),
    }
    d.update(over)
    d["sha256"] = SN.species_sha256(d["species"])
    return d


def write_dir(tmp, d, *, lock=True, filename=None):
    p = os.path.join(tmp, filename or f"{d['id']}.json")
    SN.write_snapshot(p, d)
    if lock:
        LK.write_lock(tmp, LK.update_lock(LK.read_lock(tmp), d, p))
    return p


# ---------------------------------------------------------------- hash
def test_the_hash_ignores_order_layout_and_extra_keys_but_not_content():
    base = [MALATHION, TEXANOL, XYLENE]
    h = SN.species_sha256(base)
    shuffled = list(base); random.Random(3).shuffle(shuffled)
    assert SN.species_sha256(shuffled) == h
    assert SN.species_sha256([dict(s, note="display only") for s in base]) == h
    assert SN.species_sha256(json.loads(json.dumps(base, indent=4))) == h
    assert SN.species_sha256([dict(MALATHION, name="Malathio"), TEXANOL, XYLENE]) != h
    assert SN.species_sha256([MALATHION, TEXANOL]) != h


def test_validate_accepts_v1_and_names_every_v2_defect():
    assert SN.validate_snapshot({"id": "v1", "species": [{"formula": "C2H4O2"}]}) == []
    good = v2([MALATHION])
    assert SN.validate_snapshot(good) == []
    bad = dict(good); bad.pop("source"); bad["roles"] = ["name", "nonsense"]; bad["family_scope"] = "x"
    errs = SN.validate_snapshot(bad)
    assert any(e.startswith("source") for e in errs)
    assert any("nonsense" in e for e in errs)
    assert any(e.startswith("family_scope") for e in errs)
    tampered = dict(good, species=[dict(MALATHION, name="X")])
    assert any(e.startswith("sha256 mismatch") for e in SN.validate_snapshot(tampered))


# ---------------------------------------------------------------- lock
def test_lock_round_trip_and_verify(tmp_path):
    d = v2([MALATHION, TEXANOL])
    p = write_dir(str(tmp_path), d)
    lock = LK.read_lock(str(tmp_path))
    e = lock["lists"]["test_list_v2"]
    assert e["sha256"] == d["sha256"] and e["version"] == "S99.0.0.1" and e["file"] == os.path.basename(p)
    assert e["roles"] == ["name"] and e["source_doi"] == "10.5281/zenodo.1" and e["schema_version"] == 2
    assert LK.verify(d, e, name="x.json") == d["sha256"]
    with pytest.raises(SnapshotIntegrityError, match="test_list_v2"):
        LK.verify(dict(d, species=[MALATHION]), e, name="x.json")
    # relock_directory reproduces the written lock's entries
    assert LK.relock_directory(str(tmp_path))["lists"] == lock["lists"]


def test_user_lists_dir_honours_the_env(monkeypatch, tmp_path):
    monkeypatch.setenv(LK.LISTS_DIR_ENV, str(tmp_path / "mine"))
    assert LK.user_lists_dir() == str(tmp_path / "mine")
    monkeypatch.delenv(LK.LISTS_DIR_ENV)
    assert LK.user_lists_dir().endswith(os.path.join(".cache", "peaky", "lists"))


# ---------------------------------------------------------------- loader
def test_a_pinned_v2_snapshot_loads_with_its_provenance(tmp_path):
    d = v2([MALATHION, TEXANOL, XYLENE, ETHYLBENZENE])
    write_dir(str(tmp_path), d)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cat = RL.load_catalog(str(tmp_path))
    L = cat["test_list_v2"]
    assert L.pinned is True and L.sha256 == d["sha256"] and L.lock_version == "S99.0.0.1"
    assert L.data_version == "S99.0.0.1" and L.schema_version == 2 and L.roles == ("name",)
    assert L.effective_roles == ("name",) and L.family_scope == "compound"
    assert L.source["doi"] == "10.5281/zenodo.1" and L.importer == {"name": "test", "version": "0.0.1"}
    assert L.formulas == {"C10H19O6PS2", "C12H24O3", "C8H10"}
    m = L.meta_of["C10H19O6PS2"]
    assert m["name"] == "Malathion" and m["inchikey_block"] == "JXSJBGJIGXNWCI" and m["n_compounds"] == 1
    # two compounds sharing a formula: one display name, both kept
    x = L.meta_of["C8H10"]
    assert x["name"] == "o-Xylene; Ethylbenzene" and x["n_compounds"] == 2 and x["names"] == ["o-Xylene", "Ethylbenzene"]


def test_one_changed_byte_is_refused_by_name(tmp_path):
    d = v2([MALATHION, TEXANOL])
    p = write_dir(str(tmp_path), d)
    txt = open(p, encoding="utf-8").read().replace('"Malathion"', '"Malathiom"')
    open(p, "w", encoding="utf-8").write(txt)
    RL.clear_catalog_cache()
    with pytest.raises(SnapshotIntegrityError, match="test_list_v2"):
        RL.load_catalog(str(tmp_path))
    # the file's own hash is repaired but the lock still says otherwise: still refused
    d2 = json.loads(txt); d2["sha256"] = SN.species_sha256(d2["species"])
    SN.write_snapshot(p, d2)
    RL.clear_catalog_cache()
    with pytest.raises(SnapshotIntegrityError, match="catalog.lock"):
        RL.load_catalog(str(tmp_path))
    # verify=False (tooling) reads it anyway, unpinned state visible
    cat = RL.load_catalog(str(tmp_path), verify=False)
    assert cat["test_list_v2"].sha256 == d2["sha256"]


def test_an_unpinned_v2_snapshot_warns_and_a_v1_dir_without_lock_stays_silent(tmp_path):
    write_dir(str(tmp_path), v2([MALATHION]), lock=False)
    with pytest.warns(UserWarning, match="UNPINNED"):
        cat = RL.load_catalog(str(tmp_path))
    assert cat["test_list_v2"].pinned is False
    v1dir = tmp_path / "v1"; v1dir.mkdir()
    (v1dir / "plain.json").write_text(json.dumps({"id": "plain_v1", "data_version": "1.0",
                                                  "species": [{"formula": "C2H4O2", "name": "acetic acid"}]}))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cat = RL.load_catalog(str(v1dir))
    L = cat["plain_v1"]
    assert L.pinned is False and L.schema_version == 1 and L.effective_roles == ("name",) and L.sha256


def test_a_malformed_v2_snapshot_raises(tmp_path):
    d = v2([MALATHION]); d.pop("importer")
    (tmp_path / "bad.json").write_text(json.dumps(d))
    with pytest.raises(ValueError, match="importer"):
        RL.load_catalog(str(tmp_path))


def test_the_user_directory_is_read_beside_the_bundled_catalog(monkeypatch, tmp_path):
    monkeypatch.setenv(LK.LISTS_DIR_ENV, str(tmp_path))
    write_dir(str(tmp_path), v2([MALATHION], id="my_user_list"))
    RL.clear_catalog_cache()
    cat = RL.load_catalog()
    assert "my_user_list" in cat and "contaminants_keller2008" in cat
    assert cat["my_user_list"].directory == str(tmp_path)
    # the cache key follows the files: adding one is seen without clearing
    write_dir(str(tmp_path), v2([TEXANOL], id="my_second_list"))
    assert "my_second_list" in RL.load_catalog()


def test_explicit_ids_activate_a_list_no_keyword_names(monkeypatch, tmp_path):
    monkeypatch.setenv(LK.LISTS_DIR_ENV, str(tmp_path))
    write_dir(str(tmp_path), v2([MALATHION], id="my_user_list"))
    RL.clear_catalog_cache()
    lists, tags = RL.activate("ambient-air", "uronium")
    assert "my_user_list" not in {L.id for L in lists}
    lists, tags, rec = RL.activate("ambient-air", "uronium", record=True, include=["my_user_list"])
    assert "my_user_list" in {L.id for L in lists} and rec["explicit"] == ["my_user_list"]
    monkeypatch.setenv(RL.LISTS_ENV, "my_user_list")
    lists, _ = RL.activate("ambient-air", "uronium")
    assert "my_user_list" in {L.id for L in lists}
    with pytest.raises(KeyError, match="no_such_list"):
        RL.activate("ambient-air", "uronium", include=["no_such_list"])
    recs = RL.lists_record(lists, rec)
    me = next(r for r in recs if r["id"] == "my_user_list")
    assert me["pinned"] is True and me["sha256"] and me["how_activated"] == "explicit (--lists)"
    assert me["roles"] == ["name"] and me["version"] == "S99.0.0.1" and me["schema_version"] == 2
    keller = next(r for r in recs if r["id"] == "contaminants_keller2008")
    assert keller["how_activated"] == "always active"
