"""A schema-v2 snapshot's entries in the evidence scale's context lists: named
hits, the `class` family scope, the `both` polarity, and the list-specificity tag."""
import os

import pytest

from peaky.assignment import reflists as RL
from peaky.assignment.levels import lists as LS
from peaky.lists import lock as LK
from peaky.lists import snapshot as SN
from peaky.lists.importers import epacons as E

from tests.test_lists_snapshot import MALATHION, TEXANOL, XYLENE, v2, write_dir


def test_a_named_v2_entry_is_a_named_context_entry_unless_its_scope_is_class(tmp_path):
    homolog = dict(MALATHION, formula="C11H21O6PS2", name="Malathion homolog", inchikey=None, inchikey_block=None)
    d = v2([MALATHION, TEXANOL, XYLENE, homolog])
    write_dir(str(tmp_path), d)
    cat = RL.load_catalog(str(tmp_path))
    cl = LS.ContextLists("positive", "uronium", [(d["id"], "S99.0.0.1")], catalog=cat)
    hits = [h for h in cl.hits("C10H19O6PS2") if h["id"].startswith("reflist:")]
    assert len(hits) == 1 and hits[0]["named"] and hits[0]["name"] == "Malathion"
    assert cl.context(hits) == "reflist:test_list_v2 = Malathion"
    # `both` polarity contradicts neither run
    assert cl.mode_flag(hits[0], "C10H19O6PS2", "positive", "[M+H]+") == ""
    assert cl.mode_flag(hits[0], "C10H19O6PS2", "negative", "[M-H]-") == ""
    # specificity: the +CH2 homologue is on the list, nothing else
    sp = cl.specificity(hits[0], "C10H19O6PS2")
    assert sp["n"] == 1 and sp["neighbours"] == ["+CH2 C11H21O6PS2"] and sp["n_listed"] == 4
    assert cl.specificity(hits[0], "C12H24O3")["n"] == 0
    # a v1 list has no specificity reading
    v1 = LS.ContextLists("positive", "uronium", [("contaminants_keller2008", "2008.2")], catalog=RL.load_catalog())
    v1_hits = [h for f in v1._cat["contaminants_keller2008"].formulas for h in v1.hits(f)
               if h["id"] == "reflist:contaminants_keller2008"]
    assert v1_hits and all(v1.specificity(h, "C8H10O") is None for h in v1_hits[:3])
    # class scope: present, never named
    d2 = v2([MALATHION], id="class_list", family_scope="class", roles=["class"])
    write_dir(str(tmp_path), d2)
    RL.clear_catalog_cache()
    cat = RL.load_catalog(str(tmp_path))
    cl2 = LS.ContextLists("positive", "uronium", [("class_list", "S99.0.0.1")], catalog=cat)
    h2 = [h for h in cl2.hits("C10H19O6PS2") if h["id"] == "reflist:class_list"]
    assert len(h2) == 1 and not h2[0]["named"] and h2[0]["kind"] == "class"


def test_the_bundled_epacons_names_malathion_and_counts_its_neighbours():
    cl = LS.ContextLists("positive", "uronium", [(E.LIST_ID, E.VERSION)], catalog=RL.load_catalog())
    hits = [h for h in cl.hits("C10H19O6PS2") if h["id"] == f"reflist:{E.LIST_ID}"]
    assert len(hits) == 1 and hits[0]["name"] == "Malathion"
    sp = cl.specificity(hits[0], "C10H19O6PS2")
    assert sp is not None and sp["n_listed"] >= 700 and sp["n"] >= 0
    assert cl.context_source(hits).startswith(f"reflist:{E.LIST_ID}: ")
