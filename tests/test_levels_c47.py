"""The three rule changes of the evidence scale accepted on 2026-10-05 (C47).

1. `lowconf` alone is a 5a ceiling, not a rejection ("not established", where
   5b says "refuted"); with any other rejection the pair is still 5b, and the
   pair still anchors nothing.
2. A contaminant family opens the run's space only from >= 2 files
   (`context.family_union`, FAMILY_MIN_FILES), as the other 2-file minima.
3. A reference list that rescued a dim reading (a tentative lead, setter
   `reflist_dim`) may not also certify it at 3c: the pair takes the level its
   other facts give, with the tag `3c withheld (list rescued the lead)`.

Offline; the decision helpers of tests/test_levels_decide.py.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_levels_decide as TD  # noqa: E402

from peaky.assignment.levels import context as CX  # noqa: E402
from peaky.assignment.levels import decide as DC  # noqa: E402
from peaky.assignment.levels import lists as LS  # noqa: E402
from peaky.assignment.levels import routes as RT  # noqa: E402


# --------------------------------------------------------------------------- 1. the lowconf ceiling
class TestLowconfCeiling:
    def test_lowconf_alone_reads_5a_not_5b(self):
        r = TD._one(dict(TD.ACID, lowconf=True))
        assert r.evidence_level == "5a" and r.claim != "refuted"
        assert r.would_lift == DC.LOWCONF_CEILING_WHY
        assert r.evidence.startswith("5a · " + DC.LOWCONF_CEILING_WHY + " · ion: ")
        assert "lowconf 5a ceiling" in r.tag_kinds.split("|")
        # its step-1 / step-2 facts stand in the record
        assert r.split_pinned

    def test_lowconf_with_a_veto_is_still_refuted(self):
        r = TD._one(dict(TD.ACID, lowconf=True, iso_veto=True))
        assert r.evidence_level == "5b" and r.would_lift == "refuted: iso_veto; lowconf"
        r = TD._one(dict(TD.ACID, lowconf=True, label_veto=True))
        assert r.evidence_level == "5b"

    def test_untestable_or_competitors_decide_before_the_ceiling(self):
        r = TD._one(dict(TD.ACID, lowconf=True, no_comp_info=True))
        assert r.evidence_level == "5b" and r.would_lift == "nothing could be enumerated or tested"
        comps = [("C10H16O4", "[M-H]-", "C9H12N2O4", "[M-H]-", "left")]
        r = TD._one(dict(TD.ACID, lowconf=True), comps=comps)
        assert r.evidence_level == "5a" and r.would_lift.startswith("competitors left: C9H12N2O4")
        assert "lowconf 5a ceiling" in r.tag_kinds.split("|")

    def test_the_ceiling_caps_a_would_be_3c_too(self):
        lists = LS.ContextLists("negative", "ambient-air", [])
        assert TD._one(TD.NITROPHENOL, lists=lists).evidence_level == "3c"
        r = TD._one(dict(TD.NITROPHENOL, lowconf=True), lists=lists)
        assert r.evidence_level == "5a" and r.would_lift == DC.LOWCONF_CEILING_WHY

    def test_a_lowconf_pair_anchors_nothing(self):
        """The pair is no member of the series exclusion: a competitor it would
        have excluded for another pair stays."""
        S = TD._prepared([dict(TD.ACID, lowconf=True)])
        res = DC.inpass(S)
        assert bool(res["lowconf_only"][0]) and bool(res["rej"][0])
        assert res["level"][0] == "5a" and res["why"][0] == DC.LOWCONF_CEILING_WHY

    def test_lowconf_reason_text_and_the_lead_helper(self):
        assert DC.LOWCONF_REASON == "lowconf"
        assert DC.lead_rescued("reflist_dim") and DC.lead_rescued("off_budget,reflist_dim")
        assert DC.lead_rescued("spec_n3|reflist_dim") and not DC.lead_rescued("off_budget")
        assert not DC.lead_rescued("") and not DC.lead_rescued(None) and not DC.lead_rescued(float("nan"))


# --------------------------------------------------------------------------- 3. 3c withheld
class TestThreeCWithheld:
    LEAD = {("C6H5NO3", "[M-H]-"): "reflist_dim"}

    def test_the_list_that_rescued_the_reading_cannot_certify_it(self):
        lists = LS.ContextLists("negative", "ambient-air", [])
        r = TD._one(TD.NITROPHENOL, lists=lists, lead_by=self.LEAD)
        assert r.evidence_level == "4a"                      # its other facts: pinned + an own 13C line
        assert "3c withheld (list rescued the lead)" in r.tag_kinds.split("|")
        assert r.would_lift == DC.LIFT_3C_WITHHELD
        assert r.named_list == "registry:nitroaromatic = nitrophenol"   # the entry is still printed

    def test_without_a_positive_fact_it_reads_4b(self):
        lists = LS.ContextLists("negative", "ambient-air", [])
        r = TD._one(dict(TD.NITROPHENOL, committed_matched="", committed_matched_lines=""), lists=lists,
                    lead_by=self.LEAD)
        assert r.evidence_level == "4b"
        assert "3c withheld (list rescued the lead)" in r.tag_kinds.split("|")

    def test_another_lead_setter_leaves_3c_alone(self):
        lists = LS.ContextLists("negative", "ambient-air", [])
        r = TD._one(TD.NITROPHENOL, lists=lists, lead_by={("C6H5NO3", "[M-H]-"): "off_budget"})
        assert r.evidence_level == "3c"
        assert "3c withheld (list rescued the lead)" not in r.tag_kinds.split("|")


# --------------------------------------------------------------------------- 2. the family union
class TestFamilyUnion:
    def test_a_family_opens_from_two_files_not_one(self):
        win = {"f1": {"fams": ("fluorinated", "siloxane")}, "f2": {"fams": ("siloxane",)}, "f3": {"fams": ()}}
        kept, dropped = CX.family_union(win)
        assert kept == ("siloxane",) and dropped == ("fluorinated",)

    def test_a_one_file_source_keeps_every_family(self):
        kept, dropped = CX.family_union({"f1": {"fams": ("fluorinated", "siloxane")}})
        assert kept == ("fluorinated", "siloxane") and dropped == ()
        assert CX.family_union({}) == ((), ())

    def test_first_seen_order_and_no_duplicates(self):
        win = {"f1": {"fams": ("b", "a", "b")}, "f2": {"fams": ("a", "b")}}
        assert CX.family_union(win) == (("b", "a"), ())

    def test_the_minimum_is_the_scales_two_file_minimum(self):
        assert CX.FAMILY_MIN_FILES == 2 == RT.ROUTE_COFILES == RT.LADDER_MIN_FILES
