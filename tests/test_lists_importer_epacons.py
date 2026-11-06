"""The NORMAN-SLE S22 EPACONS importer, offline: parsing, exclusion, the exact
header contract, the built snapshot, and the bundled copy's pin.

The fixture is four real rows of the source file (header + malathion, a sodium
salt the mass table cannot hold, DEET, and a mixture with no formula), written
in the source's own cp1252 encoding.
"""
import csv
import io
import json
import os
import warnings

import pytest

from peaky.assignment import reflists as RL
from peaky.lists import lock as LK
from peaky.lists import snapshot as SN
from peaky.lists.importers import IMPORTERS, epacons as E
from peaky.paths import pkg_data

FIX = os.path.join(os.path.dirname(__file__), "fixtures", "lists", "epacons_head.csv")
BUNDLED = os.path.join(pkg_data("peaklists"), f"{E.LIST_ID}.json")


def test_the_importer_is_registered():
    assert IMPORTERS[E.SOURCE_ID] is E


def test_parse_keeps_neutrals_excludes_salts_and_mixtures_with_a_reason():
    species, excluded, warn = E.parse_csv(FIX)
    by = {s["name"]: s for s in species}
    m = by["Malathion"]
    assert m["formula"] == "C10H19O6PS2" and m["inchikey_block"] == "JXSJBGJIGXNWCI"
    assert m["cas"] == "121-75-5" and m["ids"] == {"dtxsid": "DTXSID4020791"} and m["radical"] is False
    assert by["DEET"]["formula"] == "C12H17NO"
    reasons = {e["name"]: e["reason"] for e in excluded}
    assert "Na" in reasons["Propoxycarbazone-sodium"]           # not a neutral over the mass table
    assert reasons["Polysorbate 80"].startswith("no formula")   # the dashboard's '-'
    assert warn == []                                            # source masses agree with peaky's


def test_a_changed_header_fails_loudly():
    text = open(FIX, encoding=E.ENCODING).read().replace("MOL FORMULA", "MOLECULAR FORMULA", 1)
    with pytest.raises(E.HeaderMismatch, match="MOL FORMULA"):
        E.parse_text(text)
    # a reordered header fails too: the parser is positional by name, never by guess
    rows = list(csv.reader(io.StringIO(open(FIX, encoding=E.ENCODING).read())))
    rows[0] = rows[0][1:] + rows[0][:1]
    with pytest.raises(E.HeaderMismatch):
        E.parse_rows(rows[1:], header=rows[0])


def test_a_name_cannot_break_the_known_label_regex():
    assert E._clean_name("weird, ppm thing") == "weird ppm thing"
    assert E._clean_name("2,4-D") == "2,4-D"


def test_build_snapshot_validates_and_pins(tmp_path):
    species, excluded, warn = E.parse_csv(FIX)
    d = E.build_snapshot(species, excluded, warn, retrieved_utc="2026-11-04T00:00:00Z")
    assert SN.validate_snapshot(d) == []
    assert d["id"] == E.LIST_ID and d["roles"] == ["name"] and d["family_scope"] == "compound"
    assert d["source"]["md5"] == E.MD5 and d["source"]["licence"] == "CC-BY-4.0" and d["source"]["encoding"] == "cp1252"
    assert d["provenance"]["n_excluded"] == 2 and "CC-BY-4.0" in d["provenance"]["attribution"]
    p = SN.write_snapshot(str(tmp_path / f"{E.LIST_ID}.json"), d)
    LK.write_lock(str(tmp_path), LK.update_lock(LK.read_lock(str(tmp_path)), d, p))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cat = RL.load_catalog(str(tmp_path))
    L = cat[E.LIST_ID]
    assert L.pinned is True and L.data_version == E.VERSION and L.roles == ("name",)
    assert L.meta_of["C10H19O6PS2"]["name"] == "Malathion"


def test_run_from_file_refuses_a_file_that_is_not_the_published_one(tmp_path):
    with pytest.raises(E.ChecksumMismatch):
        E.run(dest_dir=str(tmp_path), from_file=FIX)


# ---------------------------------------------------------------- the bundled copy
def test_the_bundled_snapshot_is_the_published_list_pinned():
    d = json.load(open(BUNDLED, encoding="utf-8"))
    assert SN.validate_snapshot(d) == []
    assert d["source"]["md5"] == E.MD5 and d["source"]["bytes"] == E.BYTES and d["source"]["version"] == E.VERSION
    assert d["importer"] == {"name": "peaky.lists.importers.epacons", "version": E.__version__}
    lock = LK.read_lock(pkg_data("peaklists"))
    assert lock["lists"][E.LIST_ID]["sha256"] == d["sha256"] == SN.species_sha256(d["species"])
    assert any(s["formula"] == "C10H19O6PS2" and s["name"] == "Malathion" for s in d["species"])
    assert d["n_species"] == len(d["species"]) >= 1600


def test_every_bundled_list_is_pinned_and_loads_without_warnings():
    lock = LK.read_lock(pkg_data("peaklists"))
    files = sorted(n for n in os.listdir(pkg_data("peaklists")) if n.endswith(".json"))
    assert sorted(e["file"] for e in lock["lists"].values()) == files
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cat = RL.load_catalog(pkg_data("peaklists"), user_dir=None)
    assert all(L.pinned for L in cat.values()) and set(cat) == set(lock["lists"])


def test_the_bundled_list_is_not_active_unless_asked(monkeypatch, tmp_path):
    monkeypatch.setenv(LK.LISTS_DIR_ENV, str(tmp_path / "none"))
    RL.clear_catalog_cache()
    lists, _ = RL.activate("Site A Ur 122-600", "Site A campaign", "uronium")
    assert E.LIST_ID not in {L.id for L in lists}
    lists, _, rec = RL.activate("Site A Ur 122-600", "Site A campaign", "uronium", record=True, include=E.LIST_ID)
    assert E.LIST_ID in {L.id for L in lists}
    me = next(r for r in RL.lists_record(lists, rec) if r["id"] == E.LIST_ID)
    assert me["pinned"] is True and me["how_activated"] == "explicit (--lists)" and me["version"] == E.VERSION
