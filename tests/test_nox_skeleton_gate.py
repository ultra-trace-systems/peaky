"""The NOx-skeleton tier cap (tiers.compute_tiers with the run's profile).

The organonitrate-skeleton reading widens the context and plausibility windows on
the premise that the sample holds organonitrates, which MS1 cannot confirm for one
formula. An Assigned row whose outcome the reading changed
(plausibility.skeleton_reliance) is Candidate; it keeps its formula. "Changed":
admitted through a skeleton by a proposer that runs the context filter, or spared
the carbon-cluster / mass-degenerate oxygen-monster demote, or accepted by
re-arbitration only on the skeleton. Curated readings keep their tier and are
noted, as are skeleton-reliant rows already Candidate.

All ledgers are synthetic: invented readings, no spectra.

Run: pytest tests/test_nox_skeleton_gate.py -q
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from peaky.assignment import cleanup as CL
from peaky.assignment import ledger as L
from peaky.assignment import plausibility as P
from peaky.assignment import tiers as T
from peaky.chem import chemistry as C
from peaky.chem import contexts as X

AIR = X.get_context("ambient-air")
RUN = X.run_profile(AIR, reagent="NO3", instrument_class="orbitrap")      # skeleton reading on
RAW = dataclasses.replace(RUN, nox_skeleton=False)
G = "cheminfo+grid"                     # pass 1: runs the context filter
LADDER = "ladder:gapfill"               # pass 6: never runs it
SKEL_N3 = "C6H11N3O4"          # skeleton-only N3 (the deuterium-alias shape)
SKEL_N2 = "C4H6N2O8"           # skeleton-only N2: no deuterium note
SKEL_N1 = "C3H7NO5"            # skeleton-only N1
OMON = "C4H7NO6"               # raw O/C 1.5: the O-monster demote's ratio, cleared by the skeleton
CLUSTER = "C3H3NO8"            # a carbon cluster on its raw reading only (DBE/C 1.0 -> 0.67)
HETERO = "C3H8N2O4"            # implausible raw (heteroatom flag; O/C 1.33): no demote on a unique window
REARB = "C8H11N3O8"            # implausible raw (heteroatom flag) only; admitted by the context filter
RAW_OK = "C6H11NO5"            # an ordinary mononitrate: no skeleton involved
CAP_LEAD = "; the reading (-ONO2 / -NO2 groups read off) presumes organonitrate chemistry that MS1 cannot confirm"


def _ledger(rows, *, height=1e4):
    """rows: [(neutral, confidence, method)] -> a ledger of committed M0 rows, one per peak."""
    led = L.new_ledger(pd.DataFrame([(f"p{i}", 100.0 + i, height) for i in range(len(rows))],
                                    columns=["peak_id", "mz", "height"]))
    for i, (n, conf, method) in enumerate(rows):
        L.commit_assignment(led, f"p{i}", neutral_formula=n, adduct="[M+NO3]-", ion_formula="X",
                            ion_score=0.9, compound_score=0.9, ppm_error=0.1, pass_no=1,
                            method=method, confidence=conf, commentary="stub")
    return led


def _with_child(formula, method, *, density=None, adduct="[M+NO3]-"):
    """One M0 row with a 13C child (corroborated), at `density` (the degeneracy stamp)."""
    led = L.new_ledger(pd.DataFrame([("p0", 100.0, 1e4), ("k", 101.0, 1e3)], columns=["peak_id", "mz", "height"]))
    L.commit_assignment(led, "p0", neutral_formula=formula, adduct=adduct, ion_formula="X", ion_score=0.9,
                        compound_score=0.9, ppm_error=0.1, pass_no=1, method=method, confidence="High",
                        commentary="stub")
    L.attach_isotopologue(led, "k", "p0", iso_label="13C")
    if density is not None:
        led["degeneracy_density"] = np.where(led["peak_id"] == "p0", density, np.nan)
    return led


def _tier(led, formula, profile=RUN):
    t = T.compute_tiers(led, profile=profile).set_index("peak_id")
    pid = led.loc[led["neutral_formula"] == formula, "peak_id"].iloc[0]
    return t.at[pid, "tier"], str(t.at[pid, "tier_reason"])


# --------------------------------------------------------------------------- the predicate
def test_the_context_branch_needs_a_filtering_proposer():
    for f in (SKEL_N3, SKEL_N2, SKEL_N1):
        assert X.skeleton_only(f, RUN) and P.rests_on_skeleton(f, RUN), f
        assert not P.rests_on_skeleton(f, RUN, filtered=False), f      # the skeleton played no part
    assert P.skeleton_reliance(SKEL_N3, RUN) == "passes the context windows only through its organonitrate-skeleton reading"
    assert not P.rests_on_skeleton(RAW_OK, RUN)
    for f in (SKEL_N3, OMON, CLUSTER, REARB):
        assert not P.rests_on_skeleton(f, AIR) and not P.rests_on_skeleton(f, None) and not P.rests_on_skeleton(f, RAW)


def test_the_proposer_map():
    # a pass-3 family runs the filter unless a significant GKA series opened it (':gka')
    for m in (G, "gka-series", "residual:series", "cleanup:iso-recovery", "labeled:15N", "contaminant:nitrate",
              "contaminant:organosulfate", "contaminant:siloxane", "contaminant:fluorinated", "rearb<-cheminfo+grid"):
        assert T.consults_context_filter(m), m
    for m in (LADDER, "completion:known-neutral", "certified:multi-channel", "residual:iso-pair", "known:x",
              "contaminant:siloxane:gka", "contaminant:organosulfate:gka", "contaminant:fluorinated:gka",
              "siloxane:ladder", "rearb<-ladder:gapfill", "", None):
        assert not T.consults_context_filter(m), m


def test_the_carbon_cluster_branch_holds_for_any_proposer():
    """The carbon-cluster demote runs on every M0 row: a raw-reading cluster that only the
    skeleton spares is skeleton-reliant whoever proposed it, the filter rejecting it or not."""
    cnt = C.parse_formula(CLUSTER)
    assert P.is_carbon_cluster(cnt, RAW) and not P.is_carbon_cluster(cnt, RUN)
    assert not X.filter_by_profile(CLUSTER, RUN)[0]
    for filtered in (True, False):
        assert P.skeleton_reliance(CLUSTER, RUN, filtered=filtered) == (
            "is spared the carbon-cluster demote only by its organonitrate-skeleton reading")


def test_the_oxygen_monster_branch_needs_a_mass_degenerate_row():
    """The O-monster demote fires on an Assigned row only when it is mass-degenerate."""
    cnt = C.parse_formula(OMON)
    assert X.filter_by_profile(OMON, RUN)[0] and not X.skeleton_only(OMON, RUN)
    assert P.is_oxygen_monster(cnt, RAW) and not P.is_oxygen_monster(cnt, RUN)
    assert not P.rests_on_skeleton(OMON, RUN)
    assert P.rests_on_skeleton(OMON, RUN, mass_degenerate=True, filtered=False)


def test_the_rearbitration_branch():
    """Re-arbitration accepts an alternative only when `implausible` reads it clean under
    the run's profile; a heteroatom flag alone demotes no Assigned row elsewhere."""
    assert X.filter_by_profile(REARB, RUN)[0] and not X.skeleton_only(REARB, RUN)
    assert P.implausible(REARB, profile=RAW) is not None and P.implausible(REARB, profile=RUN) is None
    assert not P.rests_on_skeleton(REARB, RUN, mass_degenerate=True)
    assert P.rests_on_skeleton(REARB, RUN, rearbitrated=True)
    assert not P.rests_on_skeleton(HETERO, RUN)                        # a unique window: no demote acts on it
    assert P.rests_on_skeleton(HETERO, RUN, rearbitrated=True)


@pytest.mark.parametrize("formula", ["C2H3NO5",       # Ceff < 3: no skeleton reading
                                     "C4H2O4"])       # only the small-acid band clears it, not the skeleton
def test_formulas_the_skeleton_does_not_rescue(formula):
    assert not P.rests_on_skeleton(formula, RUN, mass_degenerate=True, rearbitrated=True), formula


def test_heavy_isotope_labels_fold_first():
    """The labelled rescue filters the folded formula; so does the predicate."""
    assert X.skeleton_only("C5H9N3O6", RUN)
    assert P.rests_on_skeleton("C5H9N^N2O6", RUN)


# --------------------------------------------------------------------------- the cap
@pytest.mark.parametrize("formula,d_note", [(SKEL_N3, True), (SKEL_N2, False), (SKEL_N1, False)])
def test_a_skeleton_only_grid_reading_is_candidate(formula, d_note):
    led = _ledger([(RAW_OK, "High", G), (formula, "High", G)])
    tier, why = _tier(led, formula)
    assert tier == T.TIER_CANDIDATE, why
    assert why.startswith(f"{T.NOX_SKELETON_CAP_MARK}: {formula} {P.SKELETON_ONLY_REASON}{CAP_LEAD}"), why
    assert ("C2 + D + O" in why) is d_note, why
    assert "(otherwise Assigned:" in why
    assert _tier(led, RAW_OK) == (T.TIER_ASSIGNED, _tier(led, RAW_OK, profile=None)[1])


def test_a_ladder_row_the_skeleton_did_not_admit_is_left_alone():
    """Pass 6 never consults the windows: a skeleton-only formula there owes nothing to the
    skeleton, so it is neither capped nor noted (as without the skeleton reading)."""
    led = _with_child("C6H5NO3", LADDER)
    assert X.skeleton_only("C6H5NO3", RUN)
    assert _tier(led, "C6H5NO3") == _tier(led, "C6H5NO3", profile=None)
    assert _tier(led, "C6H5NO3")[0] == T.TIER_ASSIGNED and T.NOX_SKELETON_NOTE not in _tier(led, "C6H5NO3")[1]


def test_a_ladder_row_the_skeleton_spares_the_carbon_cluster_demote_is_capped():
    led = _with_child(CLUSTER, LADDER)
    tier, why = _tier(led, CLUSTER)
    assert tier == T.TIER_CANDIDATE and why.startswith(
        f"{T.NOX_SKELETON_CAP_MARK}: {CLUSTER} is spared the carbon-cluster demote only by its "
        "organonitrate-skeleton reading"), why
    assert _tier(led, CLUSTER, profile=None)[0] == T.TIER_ASSIGNED     # the tier stage itself; the demote runs later


def test_the_deuterium_note_counts_14n_only():
    """C2 + D + O is 14N3 within 0.2 mDa; a 15N-bearing N3 reading is no such alias."""
    led = _ledger([("C5H9N^N2O6", "High", "labeled:15N")])
    tier, why = _tier(led, "C5H9N^N2O6")
    assert tier == T.TIER_CANDIDATE and why.startswith(T.NOX_SKELETON_CAP_MARK) and "C2 + D + O" not in why


@pytest.mark.parametrize("method", [G, LADDER])
def test_an_omonster_reading_is_capped_only_when_mass_degenerate(method):
    """A corroborated (13C child) O/C 1.5 row: on a unique window it is Assigned either way;
    on a degenerate window the skeleton is what saves it from the O-monster demote."""
    assert _tier(_with_child(OMON, method, density=1), OMON)[0] == T.TIER_ASSIGNED
    tier, why = _tier(_with_child(OMON, method, density=5), OMON)
    assert tier == T.TIER_CANDIDATE and "oxygen-monster demote" in why, why


def test_a_rearbitrated_reading_is_capped():
    led = _ledger([(REARB, "High", "rearb<-cheminfo+grid"), (RAW_OK, "High", G)])
    tier, why = _tier(led, REARB)
    assert tier == T.TIER_CANDIDATE and "re-arbitration" in why, why
    assert _tier(_ledger([(REARB, "High", G)]), REARB)[0] == T.TIER_ASSIGNED


def test_curated_readings_are_exempt_and_noted():
    for method in ("known:atmospheric", "reflist-rescue:x", "rearb<-known:atmospheric"):
        tier, why = _tier(_with_child(CLUSTER, method), CLUSTER)
        assert tier == T.TIER_ASSIGNED and why.endswith(T.NOX_SKELETON_NOTE), (method, why)
    # a curated skeleton-only formula never went through the filter: nothing to say
    tier, why = _tier(_ledger([(SKEL_N3, "High", "known:atmospheric")]), SKEL_N3)
    assert tier == T.TIER_ASSIGNED and T.NOX_SKELETON_NOTE not in why


def test_a_candidate_for_another_reason_is_noted_not_capped():
    led = _ledger([(SKEL_N3, "Low", G)])
    tier, why = _tier(led, SKEL_N3)
    assert tier == T.TIER_CANDIDATE and why.startswith("Low confidence") and why.endswith(T.NOX_SKELETON_NOTE), why


def test_without_the_skeleton_reading_nothing_changes():
    led = _ledger([(SKEL_N3, "High", G), (OMON, "High", G), (RAW_OK, "Low", G), (CLUSTER, "High", LADDER)])
    t0 = T.compute_tiers(led)
    for profile in (None, AIR, RAW):
        pd.testing.assert_frame_equal(T.compute_tiers(led, profile=profile), t0)


def test_the_summary_counts_capped_and_noted_rows():
    led = _ledger([(SKEL_N3, "High", G), (SKEL_N1, "Low", G), (RAW_OK, "High", G)])
    T.apply_tiers(led, profile=RUN)
    assert T.nox_skeleton_summary(led, RUN) == {"capped": 1, "noted": 1}
    assert T.nox_skeleton_summary(led, AIR) is None and T.nox_skeleton_summary(led, None) is None


# --------------------------------------------------------------------------- merged rows
def _files(*specs):
    out = {}
    for k, (formula, method) in enumerate(specs):
        led = _ledger([(formula, "High", method)])
        T.apply_tiers(led, profile=RUN)
        out[f"f{k}"] = led
    return out


def _merged(rows, tier="Candidate"):
    return pd.DataFrame({"neutral_formula": [r[0] for r in rows], "adduct": [r[1] for r in rows],
                         "tier": tier, "tier_reason": [r[2] for r in rows]})


def test_the_merged_row_says_how_many_files_capped_it():
    files = _files((SKEL_N3, G), (SKEL_N3, G), (RAW_OK, G))
    files["f9"] = _ledger([(SKEL_N3, "High", "known:x")])               # holds the reading, uncapped
    T.apply_tiers(files["f9"], profile=RUN)
    merged = _merged([(SKEL_N3, "[M+NO3]-", "vote note"), (RAW_OK, "[M+NO3]-", np.nan),
                      (SKEL_N3, "[M-H]-", np.nan)])                      # same neutral, another reading
    out, s = T.merged_skeleton_notes(merged, files)
    assert s == {"merged_capped": 1, "merged_noted": 0}
    assert out.at[0, "tier_reason"] == (f"vote note; {T.NOX_SKELETON_CAP_MARK}: Candidate in 2 of the 3 file(s) "
                                        "that commit it, the organonitrate-skeleton reading having changed its "
                                        "outcome there")
    assert pd.isna(out.at[1, "tier_reason"]) and pd.isna(out.at[2, "tier_reason"])
    # merged Assigned (a file holds it otherwise): said, and counted as noted, not capped
    out2, s2 = T.merged_skeleton_notes(_merged([(SKEL_N3, "[M+NO3]-", np.nan)], tier="Assigned"), files)
    assert s2 == {"merged_capped": 0, "merged_noted": 1}
    assert out2.at[0, "tier_reason"] == (f"{T.NOX_SKELETON_NOTE}: capped in 2 of the 3 file(s) that commit it, "
                                         "Assigned by a file that holds it otherwise")


def test_a_merged_note_on_an_empty_reason_and_the_noted_count():
    files = _files((SKEL_N1, "known:x"), (SKEL_N1, G))
    files["f0"] = _ledger([(SKEL_N1, "Low", G)])                         # noted (Candidate for another reason)
    T.apply_tiers(files["f0"], profile=RUN)
    files["f1"] = _ledger([(SKEL_N1, "High", "known:x")])                # neither: no note for a curated, unfiltered row
    T.apply_tiers(files["f1"], profile=RUN)
    out, s = T.merged_skeleton_notes(_merged([(SKEL_N1, "[M+NO3]-", np.nan)]), files)
    assert s == {"merged_capped": 0, "merged_noted": 1}
    assert out.at[0, "tier_reason"] == f"{T.NOX_SKELETON_NOTE} in 1 of the 2 file(s) that commit it"


def test_a_file_holding_the_reading_twice_counts_once():
    led = _ledger([(SKEL_N3, "High", G), (SKEL_N3, "High", G)])
    T.apply_tiers(led, profile=RUN)
    out, _ = T.merged_skeleton_notes(_merged([(SKEL_N3, "[M+NO3]-", np.nan)]), {"a": led})
    assert "Candidate in 1 of the 1 file(s)" in out.at[0, "tier_reason"]


def test_a_merged_ledger_without_a_reason_column():
    out, s = T.merged_skeleton_notes(_merged([(SKEL_N3, "[M+NO3]-", np.nan)]).drop(columns="tier_reason"),
                                     _files((SKEL_N3, G)))
    assert s["merged_capped"] == 1 and out.at[0, "tier_reason"].startswith(T.NOX_SKELETON_CAP_MARK)


# --------------------------------------------------------------------------- the 15N cluster re-read
def _relabel_case(y, method=G, *, density=None, flag=None, x_method=G):
    """A 15N-nitrate run: the covalent organonitrate Y [M-H]- (with a 13C child), whose parent
    X = Y - HNO3 is seen on its own as [X-H]- (committed by `x_method`)."""
    led = L.new_ledger(pd.DataFrame([("y", 200.0, 1e4), ("x", 150.0, 1e4), ("k", 201.0, 1e3)],
                                    columns=["peak_id", "mz", "height"]))
    x = C.parse_formula(y)
    x = C.format_formula({**x, "H": x["H"] - 1, "N": x["N"] - 1, "O": x["O"] - 3})
    for pid, n, m in (("y", y, method), ("x", x, x_method)):
        L.commit_assignment(led, pid, neutral_formula=n, adduct="[M-H]-", ion_formula="X", ion_score=0.9,
                            compound_score=0.9, ppm_error=0.1, pass_no=1, method=m, confidence="High",
                            commentary="stub")
    L.attach_isotopologue(led, "k", "y", iso_label="13C")
    if density is not None:
        led["degeneracy_density"] = np.where(led["peak_id"] == "y", density, np.nan)
    T.apply_tiers(led, profile=RUN)
    if flag:
        led.loc[led.peak_id == "y", flag] = True
    return led, x


def _y(led):
    return led.loc[led.peak_id == "y"].iloc[0]


def _reread(led, profile=RUN):
    CL.relabel_nitrate_clusters(led, log=lambda *a: None, profile=profile)
    return _y(led)


def test_the_15n_cluster_reread_promotes_a_capped_row_whose_parent_is_plain():
    led, x = _relabel_case("C5H8N2O8")
    assert _y(led)["tier"] == T.TIER_CANDIDATE
    r = _reread(led)
    assert (r["neutral_formula"], r["adduct"]) == (x, "[M+NO3]-") and r["tier"] == T.TIER_ASSIGNED
    assert "does not rest on the organonitrate-skeleton reading" in str(r["tier_reason"])


@pytest.mark.parametrize("case,why", [
    (dict(y="C4H6N2O10"), "the cluster parent C4H5NO7 is seen only on rows the cap holds Candidate"),  # X capped
    (dict(y="C4H8N2O9", density=5), "oxygen-monster demote"),           # X C4H7NO6 needs the skeleton when degenerate
    (dict(y="C5H8N2O8", method="rearb<-cheminfo+grid"), "re-arbitration chose the covalent reading"),
    (dict(y="C5H8N2O8", flag="below_assignability"), "flagged the row below assignability"),
])
def test_the_15n_cluster_reread_keeps_the_cap_when_something_else_holds(case, why):
    led, x = _relabel_case(case.pop("y"), **case)
    assert _y(led)["tier"] == T.TIER_CANDIDATE
    r = _reread(led)
    assert r["tier"] == T.TIER_CANDIDATE and str(r["tier_reason"]).startswith(T.NOX_SKELETON_CAP_MARK), r["tier_reason"]
    assert why in str(r["tier_reason"]), r["tier_reason"]


def test_the_15n_cluster_reread_reads_the_rows_degeneracy():
    led, _ = _relabel_case("C4H8N2O9", density=1)
    assert _y(led)["tier"] == T.TIER_CANDIDATE              # capped: Y C4H8N2O9 is skeleton-only
    assert _reread(led)["tier"] == T.TIER_ASSIGNED          # X C4H7NO6 on a unique window needs no skeleton


def test_the_15n_cluster_reread_without_the_profile_keeps_the_cap():
    led, _ = _relabel_case("C5H8N2O8")
    r = _reread(led, profile=None)
    assert r["tier"] == T.TIER_CANDIDATE and "skeleton profile is not known" in str(r["tier_reason"])


def test_the_15n_cluster_reread_only_rejudges_capped_rows():
    led, _ = _relabel_case("C5H8N2O8")
    led.loc[led.peak_id == "y", "tier_reason"] = "Low confidence: something else"
    assert _reread(led)["tier"] == T.TIER_CANDIDATE


# --------------------------------------------------------------------------- one hop
def _anchored(anchor_formula, anchor_method, fill_formula, fill_method=LADDER, *, more=()):
    """An anchor row 'a' and a row 'f' standing on it (anchor_peak_id); `more` = further
    (peak, formula, anchor) rows, each Assigned-eligible."""
    rows = [("a", anchor_formula, anchor_method, None), ("f", fill_formula, fill_method, "a")]
    rows += [(pid, n, LADDER, anc) for pid, n, anc in more]
    led = L.new_ledger(pd.DataFrame([(pid, 100.0 + 14 * i, 1e4) for i, (pid, *_r) in enumerate(rows)],
                                    columns=["peak_id", "mz", "height"]))
    for pid, n, m, _anc in rows:
        L.commit_assignment(led, pid, neutral_formula=n, adduct="[M-H]-", ion_formula="X", ion_score=0.9,
                            compound_score=0.9, ppm_error=0.1, pass_no=6, method=m, confidence="High",
                            commentary="stub")
    led["anchor_peak_id"] = led["anchor_peak_id"].astype(object) if "anchor_peak_id" in led.columns else None
    for pid, _n, _m, anc in rows:
        if anc:
            led.loc[led.peak_id == pid, "anchor_peak_id"] = anc
    return led


def test_a_row_standing_on_a_skeleton_anchor_is_capped_too():
    """C6H7NO4 passes the raw windows, but the ladder proposed it only because its anchor
    C5H5NO4 entered through a skeleton reading: one hop."""
    assert X.filter_by_profile("C6H7NO4", RAW)[0] and X.skeleton_only("C5H5NO4", RUN)
    led = _anchored("C5H5NO4", G, "C6H7NO4")
    t = T.compute_tiers(led, profile=RUN).set_index("peak_id")
    assert t.at["a", "tier"] == T.TIER_CANDIDATE
    assert t.at["f", "tier"] == T.TIER_CANDIDATE and "stands on a ladder anchor" in str(t.at["f", "tier_reason"])
    # ... transitively: a fill on that fill
    led2 = _anchored("C5H5NO4", G, "C6H7NO4", more=[("g", "C7H9NO4", "f")])
    t2 = T.compute_tiers(led2, profile=RUN).set_index("peak_id")
    assert t2.at["g", "tier"] == T.TIER_CANDIDATE and "stands on a ladder anchor" in str(t2.at["g", "tier_reason"])
    # only a ladder fill hops: a pass-2 series row on the same anchor is judged on its own
    led3 = _anchored("C5H5NO4", G, "C6H7NO4", fill_method="gka-series")
    assert _tier(led3, "C6H7NO4") == _tier(led3, "C6H7NO4", profile=None)


def test_a_curated_or_plain_anchor_hands_nothing_on():
    for anchor, method in (("C5H5NO4", "known:x"), (RAW_OK, G)):
        led = _anchored(anchor, method, "C6H7NO4")
        assert _tier(led, "C6H7NO4") == _tier(led, "C6H7NO4", profile=None), (anchor, method)


def test_a_completion_row_of_a_skeleton_admitted_neutral_is_capped():
    """Pass 5 proposes neutrals it finds committed: SKEL_N3 [M+NO3]- from completion takes
    its neutral from the grid's skeleton-only SKEL_N3 [M-H]-."""
    def led_with(siblings):
        """siblings: [(adduct, method, pass_no, confidence)] of SKEL_N3, plus the completion row."""
        rows = siblings + [("[M+NO3]-", "completion:known-neutral", 5, "High")]
        led = L.new_ledger(pd.DataFrame([(f"r{i}", 100.0 + 20 * i, 1e4) for i in range(len(rows))],
                                        columns=["peak_id", "mz", "height"]))
        for i, (adduct, m, pno, conf) in enumerate(rows):
            L.commit_assignment(led, f"r{i}", neutral_formula=SKEL_N3, adduct=adduct, ion_formula="X",
                                ion_score=0.9, compound_score=0.9, ppm_error=0.1, pass_no=pno, method=m,
                                confidence=conf, commentary="stub")
        return led, f"r{len(rows) - 1}"

    def tier_of(led, pid):
        t = T.compute_tiers(led, profile=RUN).set_index("peak_id")
        return t.at[pid, "tier"], str(t.at[pid, "tier_reason"])

    led, c = led_with([("[M-H]-", G, 1, "High")])
    tier, why = tier_of(led, c)
    assert tier == T.TIER_CANDIDATE and "takes its neutral from a sibling" in why, why
    # no hop when pass 5 could not have read the sibling: committed later, or not High / Good
    for sib in (("[M-H]-", "cleanup:iso-recovery", 6, "High"), ("[M-H]-", G, 1, "Low")):
        led, c = led_with([sib])
        assert tier_of(led, c)[0] == T.TIER_ASSIGNED, sib
    # ... nor when a row holds the neutral without the skeleton (pass 5 had it anyway)
    led, c = led_with([("[M-H]-", G, 1, "High"), ("[M+^NO3]-", "known:x", 0, "High")])
    assert tier_of(led, c)[0] == T.TIER_ASSIGNED


def test_the_15n_cluster_reread_judges_a_window_free_rows_parent_by_the_demotes():
    """A ladder row capped on the carbon-cluster leg: its cluster parent is judged by the
    demote legs alone (the ladder never read the windows)."""
    led, x = _relabel_case("C4H4N2O7", LADDER, x_method="known:x")   # Y: raw DBE/C 1.0 cluster; X C4H3NO4 also
    assert _y(led)["tier"] == T.TIER_CANDIDATE
    r = _reread(led)
    assert str(r["tier_reason"]).startswith(T.NOX_SKELETON_CAP_MARK)
    assert "carbon-cluster demote" in str(r["tier_reason"]) and "fails the context windows" not in str(r["tier_reason"])


def test_the_15n_cluster_reread_lifts_a_window_free_cap_only_on_plain_evidence():
    """A ladder row capped on the oxygen-monster leg (Y C5H9NO7, raw O/C 1.4, on a degenerate
    window) whose cluster parent X C5H8O4 (O/C 0.8) needs no skeleton and is shown by an
    uncapped row: the cap was its one demotion, so the re-read lifts it."""
    led, x = _relabel_case("C5H9NO7", LADDER, density=5, x_method="known:x")
    assert x == "C5H8O4" and _y(led)["tier"] == T.TIER_CANDIDATE
    assert "oxygen-monster demote" in str(_y(led)["tier_reason"])
    assert _reread(led)["tier"] == T.TIER_ASSIGNED


def test_the_15n_cluster_reread_keeps_a_one_hop_cap():
    led, x = _relabel_case("C5H8N2O8", LADDER)
    led.loc[led.peak_id == "y", "tier"] = T.TIER_CANDIDATE
    led.loc[led.peak_id == "y", "tier_reason"] = f"{T.NOX_SKELETON_CAP_MARK}: Y {T._HOP_ANCHOR}"
    r = _reread(led)
    assert r["tier"] == T.TIER_CANDIDATE and "standing on a skeleton-reliant row" in str(r["tier_reason"])


def test_a_hop_row_already_candidate_gets_the_hop_note():
    led = _anchored("C5H5NO4", G, "C6H7NO4")
    led.loc[led.peak_id == "f", "confidence"] = "Low"
    tier, why = _tier(led, "C6H7NO4")
    assert tier == T.TIER_CANDIDATE and why.endswith("stands on a row " + T.NOX_SKELETON_NOTE), why


@pytest.mark.parametrize("evidence,suffix", [(False, ""), (True, ":gka")])
def test_pass3_marks_a_family_a_gka_series_opened(monkeypatch, evidence, suffix):
    """The siloxane family (an ambient-air default) commits as 'contaminant:siloxane' when it ran
    through the context filter (no significant series) and with ':gka' when a significant
    C2H6OSi series opened it unfiltered."""
    from peaky.assignment import series_detect as SD
    from peaky.assignment.passes import directors as D
    from peaky.assignment.passes.config import PassConfig
    from peaky.io import io_mascope as IO
    led = L.new_ledger(pd.DataFrame([("p0", 181.9768, 1e5)], columns=["peak_id", "mz", "height"]))
    ev = pd.DataFrame([{"unit": "C2H6OSi", "mass": 74.0188, "n_links": 9, "enrichment": 5.0,
                        "significant": True, "action": "siloxane"}]) if evidence else \
        pd.DataFrame(columns=["unit", "mass", "n_links", "enrichment", "significant", "action"])
    methods = []
    monkeypatch.setattr(SD, "detect_series", lambda *a, **k: ev)
    monkeypatch.setattr(SD, "unit_members", lambda *a, **k: {"p0"})
    # the evidence path grids the heads of the detected chains: one chain, one head formula
    monkeypatch.setattr(SD, "unit_chains", lambda *a, **k: [[("p0", 181.9768), ("p1", 256.0)]])
    monkeypatch.setattr(D.C, "candidates_for_peaks", lambda *a, **k: {"C2H8O2Si"})
    monkeypatch.setattr(D, "_target_peaks", lambda ledger, cfg: ledger[ledger["role"] == L.ROLE_UNEXPLAINED])
    monkeypatch.setattr(D, "_enumerate", lambda *a, **k: {"C2H8O2Si", "C3H5NO6S"})
    monkeypatch.setattr(D, "_mech_ids_for", lambda client, adducts: [])
    monkeypatch.setattr(IO, "score_candidates", lambda *a, **k: pd.DataFrame())
    from collections import defaultdict
    monkeypatch.setattr(D, "commit_winners", lambda ledger, arb, **kw: methods.append(kw["method"]) or defaultdict(int))
    D.run_pass3(None, "s", led, RUN, None, PassConfig(height_cutoff_cps=1.0), ["[M-H]-", "[M+NO3]-"],
                phase="all", reagent="NO3", log=lambda *a: None)
    assert f"contaminant:siloxane{suffix}" in methods, methods
    assert "contaminant:nitrate" in methods, methods            # never opened by a series: filtered
