"""`peaky lists lock | verify | show | import --from-file`: the catalog tooling,
offline, on a copy of the bundled catalog."""
import json
import os
import shutil

import pytest

from peaky import cli
from peaky.assignment import reflists as RL
from peaky.lists import lock as LK
from peaky.lists.importers import epacons as E
from peaky.paths import pkg_data


@pytest.fixture
def catalog_copy(tmp_path):
    d = tmp_path / "cat"
    shutil.copytree(pkg_data("peaklists"), d)
    os.remove(d / LK.LOCK_NAME)
    RL.clear_catalog_cache()
    yield str(d)
    RL.clear_catalog_cache()


def test_lock_then_verify_then_tamper(catalog_copy, capsys):
    assert cli.main(["lists", "lock", "--dir", catalog_copy]) == 0
    lock = LK.read_lock(catalog_copy)
    assert set(lock["lists"]) >= {"contaminants_keller2008", E.LIST_ID}
    assert cli.main(["lists", "verify", "--dir", catalog_copy]) == 0
    out = capsys.readouterr().out
    assert "[lists] OK" in out and f"{len(lock['lists'])} pinned" in out
    # one species name changed -> verify fails naming the list
    p = os.path.join(catalog_copy, f"{E.LIST_ID}.json")
    txt = open(p, encoding="utf-8").read().replace('"Malathion"', '"Malathiom"', 1)
    open(p, "w", encoding="utf-8").write(txt)
    RL.clear_catalog_cache()
    with pytest.raises(SystemExit) as ex:
        cli.main(["lists", "verify", "--dir", catalog_copy])
    assert E.LIST_ID in str(ex.value)


def test_show_prints_provenance(capsys):
    assert cli.main(["lists", "show", E.LIST_ID]) == 0
    info = json.loads(capsys.readouterr().out)
    assert info["id"] == E.LIST_ID and info["pinned"] is True and info["schema_version"] == 2
    assert info["source"]["md5"] == E.MD5 and info["roles"] == ["name"]
    assert info["n_formulas"] >= 700          # 1663 compounds share 766 formulas (isomers)
    with pytest.raises(SystemExit):
        cli.main(["lists", "show", "no_such_list"])


def test_import_from_file_needs_the_published_file(tmp_path, capsys):
    fix = os.path.join(os.path.dirname(__file__), "fixtures", "lists", "epacons_head.csv")
    # the CLI's guard turns the ChecksumMismatch into a non-zero exit with the reason printed
    assert cli.main(["lists", "import", E.SOURCE_ID, "--from-file", fix, "--dest", str(tmp_path)]) != 0
    captured = capsys.readouterr()
    assert "ChecksumMismatch" in captured.err + captured.out
    assert not os.path.exists(tmp_path / f"{E.LIST_ID}.json")
    with pytest.raises(SystemExit):
        cli.main(["lists", "import", "no_such_source", "--dest", str(tmp_path)])


def test_assign_and_batch_parsers_take_lists():
    p = cli.build_parser()
    a = p.parse_args(["assign", "--sample-id", "x", "--lists", "a,b"])
    b = p.parse_args(["batch", "--batch", "B", "--lists", E.LIST_ID])
    assert a.lists == "a,b" and b.lists == E.LIST_ID
    assert RL.explicit_list_ids("a, b,,a") == ["a", "b"]
