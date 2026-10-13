"""The tier answers "is the ion right?"; a doubt about the neutral / adduct split
goes to the claim.

Three rules used to demote the tier for doubt about the split while the ion was
pinned: the reagent-N isobar (tiers._reagent_n_isobar), the amine gate
(cleanup.prefer_amine_over_ammonium) and the hydrocarbon-on-N-cluster re-read
(cleanup.relabel_reagent_n_adducts). Pins:

  * the isobar no longer caps the tier, and the ion keeps its own corroboration
    (its isotope lines); the row names the open split (tiers.REAGENT_N_OPEN_MARK)
    only where the split-aware verdict differed from the ion's, and a resolved
    split keeps its wording;
  * an isobar ion that is nitrogen-rich in EVERY stored reading and has nothing
    but its mass is Candidate on its own terms (tiers.REAGENT_N_RICH_MARK);
  * the merge vote's label stage still decides a same-ion pair by the files that
    fixed the nitrogen count (assign_batch.VOTE_TIER), the best-row pick is the
    one it was, and the merged tier is the winning ion's best per-file tier (any
    of its labels);
  * the two re-reads change the label only: never the tier (and the hydrocarbon
    re-read writes no tentative lead);
  * a re-read merged row takes the ion's level from the pooled pair it replaced
    (4b with the split open, or 5a), never a 5b or a bucket.

All fixtures are synthetic.

Run: pytest tests/test_tier_ion.py -q
"""
from __future__ import annotations

import pandas as pd

from peaky.assignment import cleanup as CU
from peaky.assignment import evidence as EV
from peaky.assignment import ledger as L
from peaky.assignment import tiers as T
from peaky.batch import assign_batch as AB
from peaky.chem import chemistry as C


# --------------------------------------------------------------------------- tiers
def _isobar_ledger(extra_alts=(), confidence="High", child=False, sibling=False):
    """One donor-side reagent-N reading: C8H14O3 [M+NH4]+ is the same ion as the
    protonated N-heavier C8H17NO3 [M+H]+. `child` gives it a 13C child; `sibling`
    commits the same neutral on [M+H]+ (an N-free channel)."""
    mz = C.ion_mz("C8H14O3", "[M+NH4]+")
    ids = ["X", "X13", "S"]
    led = L.new_ledger(pd.DataFrame({"peak_id": ids,
                                     "mz": [mz, mz + 1.0033548, C.ion_mz("C8H14O3", "[M+H]+")],
                                     "height": [5e4, 4.5e3, 2e4]}))
    alts = [{"formula": "C8H17NO3", "adduct": "[M+H]+", "ion_score": 0.95, "raw_score": 0.95,
             "eff_score": 0.9, "ppm": 0.2}, *extra_alts]
    L.commit_assignment(led, "X", neutral_formula="C8H14O3", adduct="[M+NH4]+",
                        ion_formula="C8H18NO3+", ion_score=0.95, compound_score=0.95,
                        eff_score=0.9, eff_margin=0.3, tied=False, ppm_error=0.2, pass_no=1,
                        method="cheminfo+grid", confidence=confidence, commentary="Pass 1",
                        alternatives=alts,
                        isotopologues=([{"label": "13C", "score": 0.9, "peak_id": "X13"}] if child else None))
    if child:
        L.attach_isotopologue(led, "X13", "X", iso_label="13C", iso_match_score=0.9)
    if sibling:
        L.commit_assignment(led, "S", neutral_formula="C8H14O3", adduct="[M+H]+",
                            ion_formula="C8H15O3+", ion_score=0.9, compound_score=0.9,
                            eff_score=0.85, eff_margin=0.3, tied=False, ppm_error=0.1, pass_no=1,
                            method="cheminfo+grid", confidence="Good", commentary="Pass 1")
    return led


def _tier(led, pid="X"):
    t = T.compute_tiers(led).set_index("peak_id")
    return t.at[pid, "tier"], t.at[pid, "tier_reason"]


def test_the_isobar_no_longer_caps_and_the_ions_isotope_line_counts():
    tier, why = _tier(_isobar_ledger(child=True))
    assert tier == "Assigned"
    assert "isotopologue-confirmed" in why and T.split_open(why)
    assert why.startswith("unique formula in the calibrated window") or why.startswith("no rival formula")
    assert "C8H17NO3 [M+H]+" in why


def test_a_row_another_rule_holds_keeps_its_reason_word_for_word():
    # Low confidence came first before and still does: the isobar adds nothing to say
    tier, why = _tier(_isobar_ledger(confidence="Low"))
    assert tier == "Candidate"
    assert why == "Low confidence: score/mass evidence below the identification bar"
    assert not T.split_open(why)


def test_an_ion_level_cap_the_isobar_used_to_mask_now_speaks_with_the_open_split():
    # a real rival ion within the close margin and nothing to corroborate the ion:
    # the ion's own density rule holds it, and the row also says the split is open
    rival = {"formula": "C7H14N2O3", "adduct": "[M+H]+", "ion_score": 0.93, "raw_score": 0.93,
             "eff_score": 0.82, "ppm": 0.4}
    tier, why = _tier(_isobar_ledger(extra_alts=[rival]))
    assert tier == "Candidate"
    assert why.startswith("1 alternative(s) within 0.1 effective score") and T.split_open(why)
    assert "reagent-N isobar unresolved" not in why


def test_a_fixed_nitrogen_count_keeps_its_wording():
    tier, why = _tier(_isobar_ledger(sibling=True))
    assert (tier, why) == ("Assigned", "reagent-N isobar: nitrogen count fixed by an N-free sibling channel")


def test_a_row_without_the_isobar_is_untouched_by_the_split_verdict():
    led = _isobar_ledger()
    led.loc[led.peak_id == "X", "alternatives"] = "[]"
    tier, why = _tier(led)
    assert tier == "Assigned" and not T.split_open(why) and "reagent-N" not in why


def _reading_ledger(formula, adduct, alias, *, child=False):
    """One positive reading (`formula` `adduct`) whose stored alternatives hold the
    same-ion readings `alias` [(formula, adduct), ...]; `child` gives it a 13C child."""
    mz = C.ion_mz(formula, adduct)
    led = L.new_ledger(pd.DataFrame({"peak_id": ["R", "R13"], "mz": [mz, mz + 1.0033548],
                                     "height": [5e4, 4.5e3]}))
    alts = [{"formula": f, "adduct": a, "ion_score": 0.95, "raw_score": 0.95, "eff_score": 0.85,
             "ppm": 0.2} for f, a in alias]
    L.commit_assignment(led, "R", neutral_formula=formula, adduct=adduct, ion_formula="",
                        ion_score=0.95, compound_score=0.95, eff_score=0.9, eff_margin=0.3, tied=False,
                        ppm_error=0.2, pass_no=1, method="cheminfo+grid", confidence="High",
                        commentary="Pass 1", alternatives=alts,
                        isotopologues=([{"label": "13C", "score": 0.9, "peak_id": "R13"}] if child else None))
    if child:
        L.attach_isotopologue(led, "R13", "R", iso_label="13C", iso_match_score=0.9)
    return led


# C6H11N3O on urea is the ion C7H16N5O2+: read as C7H12N4O2 [M+NH4]+ or C7H15N5O2
# [M+H]+ too -- every reading carries >= 3 analyte nitrogens
_N_RICH = ("C6H11N3O", "[M+(CH4N2O)H]+", [("C7H12N4O2", "[M+NH4]+"), ("C7H15N5O2", "[M+H]+")])


def test_a_nitrogen_rich_ion_on_mass_alone_stays_candidate_on_its_own_terms():
    tier, why = _tier(_reading_ledger(*_N_RICH), "R")
    assert tier == "Candidate"
    assert why.startswith(T.REAGENT_N_RICH_MARK + ": every stored reading of this ion carries >= 3 "
                          "analyte nitrogens")
    # the split is still named, after the ion's own reason
    assert T.split_open(why) and "C6H11N3O [M+(CH4N2O)H]+ is the same ion as C7H12N4O2 [M+NH4]+" in why
    assert "reagent-N isobar unresolved" not in why


def test_a_nitrogen_rich_ion_with_its_own_isotope_line_is_assigned():
    tier, why = _tier(_reading_ledger(*_N_RICH, child=True), "R")
    assert tier == "Assigned" and "isotopologue-confirmed" in why and T.split_open(why)
    assert T.REAGENT_N_RICH_MARK not in why


def test_the_n_poorest_reading_decides_whether_the_ion_is_nitrogen_rich():
    # the amine-side N3 reading of an ion that also reads as an N1 neutral on urea
    # (C6H14N3O2+): not N-rich in every reading, so the ion is Assigned on its mass
    led = _reading_ledger("C6H13N3O2", "[M+H]+", [("C5H9NO", "[M+(CH4N2O)H]+")])
    assert T._poorest_n(led.iloc[0], [{"formula": "C5H9NO", "adduct": "[M+(CH4N2O)H]+"}]) == 1
    tier, why = _tier(led, "R")
    assert tier == "Assigned" and T.split_open(why) and T.REAGENT_N_RICH_MARK not in why


def test_the_nitrogen_bar_is_the_isobar_rows_only():
    # the same N3 neutral with no same-ion alias stored: no isobar, no bar -- the
    # row is what it was (the card changes nothing outside the reagent-N pairs)
    tier, why = _tier(_reading_ledger("C6H11N3O", "[M+(CH4N2O)H]+", []), "R")
    assert tier == "Assigned" and "reagent-N" not in why and T.REAGENT_N_RICH_MARK not in why


# --------------------------------------------------------------------------- the vote
def _pf(rows):
    """One file's M0 rows as `_m0` hands them to the vote (tier + tier_reason)."""
    led = pd.DataFrame(rows, columns=["mz", "neutral_formula", "adduct", "tier", "ion_score", "tier_reason"])
    led["role"] = "M0"
    return AB._m0(led)


_OPEN = ("unique formula in the calibrated window; " + T.REAGENT_N_OPEN_MARK
         + ": C13H17NO4 [M+H]+ is the same ion as C13H14O4 [M+NH4]+ (...)")


def test_m0_carries_the_votes_tier():
    f = _pf([(252.123, "C13H17NO4", "[M+H]+", "Assigned", 0.98, _OPEN),
             (252.123, "C13H14O4", "[M+NH4]+", "Assigned", 0.97, "reagent-N isobar: nitrogen count fixed by x"),
             (300.0, "C10H12O2", "[M+H]+", "Candidate", 0.90, _OPEN)])
    assert list(f[AB.VOTE_TIER]) == ["Candidate", "Assigned", "Candidate"]
    assert list(f["tier"]) == ["Assigned", "Assigned", "Candidate"]
    assert AB.VOTE_TIER not in AB._M0_COLS


def test_the_label_stage_still_counts_the_files_that_fixed_the_nitrogen():
    # 6 files read the ion as the amine with the split open (Assigned on the ion),
    # 5 files read the NH4 adduct with the nitrogen fixed: the 5 still win the label
    files = {f"o{i}": _pf([(252.1230 + 1e-4 * i, "C13H17NO4", "[M+H]+", "Assigned", 0.98, _OPEN)])
             for i in range(6)}
    files.update({f"f{i}": _pf([(252.1230 + 1e-4 * i, "C13H14O4", "[M+NH4]+", "Assigned", 0.97,
                                 "reagent-N isobar: nitrogen count fixed by an N-free sibling channel")])
                  for i in range(5)})
    m, jit = AB.align(files, tol_ppm=6.0)
    r = m.iloc[0]
    assert (r["neutral_formula"], r["adduct"], r["tier"]) == ("C13H14O4", "[M+NH4]+", "Assigned")
    assert r["tier_reason"] == ("same ion C13H18NO4+ read two ways: kept C13H14O4 [M+NH4]+ (Assigned, "
                                "nitrogen count not left open, in 5 of its 5 files) over the 6-file "
                                "C13H17NO4 [M+H]+ (Assigned, nitrogen count not left open, in 0)")
    # the losers' description prints their TIER
    assert r["alternatives"] == "C13H17NO4 [M+H]+ x6 Assigned 0.98"
    assert set(jit["tier"]) == {"Assigned"}


def test_an_ion_no_file_could_split_is_assigned_at_the_merge():
    files = {f"o{i}": _pf([(252.1230 + 1e-4 * i, "C13H17NO4", "[M+H]+", "Assigned", 0.98, _OPEN)])
             for i in range(3)}
    r = AB.align(files, tol_ppm=6.0)[0].iloc[0]
    assert (r["neutral_formula"], r["tier"]) == ("C13H17NO4", "Assigned") and pd.isna(r["tier_reason"])


def test_the_merged_tier_is_the_winning_ions_whichever_label_a_file_read():
    # 4 files read the ion as the amine at Candidate, 2 as the NH4 adduct Assigned
    # on the ion's evidence with the split open: the label goes to the 4 (no file
    # fixed the nitrogen count), and the ion's tier is the 2 files' Assigned
    files = {f"c{i}": _pf([(252.1230 + 1e-4 * i, "C13H17NO4", "[M+H]+", "Candidate", 0.98, "x")])
             for i in range(4)}
    files.update({f"a{i}": _pf([(252.1230 + 1e-4 * i, "C13H14O4", "[M+NH4]+", "Assigned", 0.97,
                                 "isotopologue-confirmed; " + T.REAGENT_N_OPEN_MARK + ": ...")])
                  for i in range(2)})
    r = AB.align(files, tol_ppm=6.0)[0].iloc[0]
    assert (r["neutral_formula"], r["adduct"], r["tier"]) == ("C13H17NO4", "[M+H]+", "Assigned")
    assert r["alternatives"] == "C13H14O4 [M+NH4]+ x2 Assigned 0.97"
    # with the vote's tier gone (a pure align() frame), the label stage counts the
    # Assigned files and the merged row is the one it always was
    plain = {k: v.drop(columns=[AB.VOTE_TIER]) for k, v in files.items()}
    r = AB.align(plain, tol_ppm=6.0)[0].iloc[0]
    assert (r["neutral_formula"], r["tier"]) == ("C13H14O4", "Assigned")


def test_the_best_row_pick_is_the_one_it_was():
    # the split-open file scores higher, but the fixed file's row donates as before
    files = {"a": _pf([(252.1230, "C13H14O4", "[M+NH4]+", "Assigned", 0.90, "fixed")]),
             "b": _pf([(252.1231, "C13H14O4", "[M+NH4]+", "Assigned", 0.99,
                        "x; " + T.REAGENT_N_OPEN_MARK + ": ...")])}
    r = AB.align(files, tol_ppm=6.0)[0].iloc[0]
    assert r["tier"] == "Assigned" and abs(float(r["ion_score"]) - 0.90) < 1e-12


def test_frames_without_the_votes_tier_vote_as_before():
    plain = {k: v.drop(columns=[AB.VOTE_TIER]) for k, v in {
        "a": _pf([(252.1230, "C13H17NO4", "[M+H]+", "Candidate", 0.98, "")]),
        "b": _pf([(252.1231, "C13H14O4", "[M+NH4]+", "Assigned", 0.97, "")])}.items()}
    r = AB.align(plain, tol_ppm=6.0)[0].iloc[0]
    assert r["neutral_formula"] == "C13H14O4" and r["tier"] == "Assigned"


# --------------------------------------------------------------------------- the re-reads
def _merged():
    return pd.DataFrame([
        dict(mz=C.ion_mz("C9H16O", "[M+NH4]+"), neutral_formula="C9H16O", adduct="[M+NH4]+", tier="Assigned",
             ion_score=0.95, n_files=6, tier_reason=pd.NA),
        dict(mz=C.ion_mz("C9H16O", "[M+NH4]+") + 30, neutral_formula="C10H14O2", adduct="[M+NH4]+",
             tier="Candidate", ion_score=0.80, n_files=2, tier_reason=pd.NA),
        dict(mz=C.ion_mz("C8H8", "[M+NH4]+"), neutral_formula="C8H8", adduct="[M+NH4]+", tier="Assigned",
             ion_score=0.97, n_files=9, tier_reason=pd.NA),
        dict(mz=C.ion_mz("C6H14O4", "[M+NH4]+"), neutral_formula="C6H14O4", adduct="[M+NH4]+", tier="Assigned",
             ion_score=0.92, n_files=4, tier_reason=pd.NA),
    ])


def test_the_two_re_reads_move_the_label_and_never_the_tier():
    m = _merged()
    before = m["tier"].tolist()
    assert CU.relabel_reagent_n_adducts(m, log=lambda *a: None) == {"reagent_n_relabeled": 1}
    out = CU.prefer_amine_over_ammonium(m, log=lambda *a, **k: None)
    assert out["relabeled"] == 2 and out["forced_nh4"] == 1
    assert m["tier"].tolist() == before
    assert list(zip(m.neutral_formula, m.adduct)) == [("C9H19NO", "[M+H]+"), ("C10H17NO2", "[M+H]+"),
                                                       ("C8H11N", "[M+H]+"), ("C6H14O4", "[M+NH4]+")]
    assert [CU.reread_source(t) for t in m["tier_reason"]] == [
        ("C9H16O", "[M+NH4]+"), ("C10H14O2", "[M+NH4]+"), ("C8H8", "[M+NH4]+"), None]
    assert not m["tier_reason"].astype(str).str.contains("Candidate").any()


def _pairs():
    """Pooled pairs of the replaced readings, one per carried / refused level."""
    def pair(nf, ad, lvl, ev, wl="", comps="", tags=""):
        return dict(neutral_formula=nf, adduct=ad, evidence_level=lvl, evidence=ev, would_lift=wl,
                    competitors_left=comps, tags=tags, context="reflist:x = some name",
                    context_source="registry: pass-0 registry (positive)", claim=EV.claim_class(lvl))
    return pd.DataFrame([
        pair("C9H16O", "[M+NH4]+", "4b",
             "4b · ion: unique in the calibrated window · split: open -- amine default (NH4 reading unconfirmed: "
             "parent channels absent; the engine's gate reads this ion as C9H19NO [M+H]+) · positive fact: none · "
             "named list: none · context source: registry: pass-0 registry (positive) · tags: amine default",
             wl="split not pinned: amine default (...)", tags="amine default"),
        pair("C10H14O2", "[M+NH4]+", "5a",
             "5a · competitors left (1) · ion: 1 competitor(s); 1 left · split: amine default (...) · "
             "context source: registry: pass-0 registry (positive) · tags: none",
             wl="competitors left: C9H18FNOSi [M+H]+", comps="C9H18FNOSi [M+H]+"),
        pair("C8H8", "[M+NH4]+", "5b",
             "5b · rejected: below: implausible ionization · ion: unique in the calibrated window · split: x",
             wl="refuted: below: implausible ionization"),
    ])


def test_a_re_read_row_takes_the_ions_level_from_the_pair_it_replaced():
    m = _merged()
    CU.relabel_reagent_n_adducts(m, log=lambda *a: None)
    CU.prefer_amine_over_ammonium(m, log=lambda *a, **k: None)
    out = EV.stamp_merged(m, _pairs()).set_index("neutral_formula")
    a = out.loc["C9H19NO"]
    assert (a["evidence_level"], a["claim"]) == ("4b", "ion")
    assert a["evidence"].startswith("4b · ion: unique in the calibrated window · split: open -- batch-level "
                                    "re-read of the pooled pair C9H16O [M+NH4]+, the same ion (that pair's "
                                    "split: open -- amine default")
    assert a["would_lift"].startswith("split not pinned: batch-level re-read of the pooled pair C9H16O")
    assert pd.isna(a["context"]) and a["context_source"] == "registry: pass-0 registry (positive)"
    b = out.loc["C10H17NO2"]
    assert (b["evidence_level"], b["claim"], b["competitors_left"]) == ("5a", "tentative", "C9H18FNOSi [M+H]+")
    assert " · split: batch-level re-read of the pooled pair C10H14O2 [M+NH4]+, the same ion" in b["evidence"]
    assert b["would_lift"] == "competitors left: C9H18FNOSi [M+H]+"
    # a rejected pair's reasons are the replaced reading's: no level carried
    c = out.loc["C8H11N"]
    assert pd.isna(c["evidence_level"]) and c["evidence"] == EV.NO_POOLED_PAIR_TEXT and c["claim"] == "tentative"
    # the kept amine-impossible NH4 reading is its own pair: not in this table, no level
    assert out.loc["C6H14O4", "evidence"] == EV.NO_POOLED_PAIR_TEXT


def test_a_re_read_pair_pinned_in_the_pool_reads_open_on_the_merged_row():
    m = _merged().iloc[[0]].copy()
    CU.prefer_amine_over_ammonium(m, log=lambda *a, **k: None)
    pinned = _pairs().iloc[[0]].assign(
        evidence_level="4a", claim="neutral",
        evidence="4a (split pinned + positive fact) · ion: unique in the calibrated window · split: PINNED -- "
                 "NH4 adduct tracks its parent (r 0.70) · positive fact: NH4 adduct tracks its parent · named "
                 "list: none · context source: c · tags: none")
    r = EV.stamp_merged(m, pinned).iloc[0]
    assert (r["evidence_level"], r["claim"]) == ("4b", "ion")
    assert "split: open -- batch-level re-read of the pooled pair C9H16O [M+NH4]+, the same ion (that pair's " \
           "split: PINNED -- NH4 adduct tracks its parent (r 0.70))" in r["evidence"]
    assert "positive fact: none" in r["evidence"]


def test_a_row_with_no_re_read_note_stays_unlevelled():
    m = pd.DataFrame([dict(mz=150.0, neutral_formula="C9H19NO", adduct="[M+H]+", tier="Assigned",
                           tier_reason="something else")])
    r = EV.stamp_merged(m, _pairs()).iloc[0]
    assert r["evidence"] == EV.NO_POOLED_PAIR_TEXT and pd.isna(r["evidence_level"])
