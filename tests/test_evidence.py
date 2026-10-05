"""The private fact layer of peaky.assignment.evidence and the merge vote's class.

Before peaky 0.10.0 the evidence levels were a decision over per-pair FACTS
(the isotope axis, the second channel, the anchor, the acid branch, the
reagent-only satellite, tied / below / Low ...). The evidence scale of peaky
0.10.0 replaced that decision as the user-facing level
(tests/test_levels_*.py). The facts and the old decision stay, PRIVATE
(build decision D1): the facts are the scale's step-0 inputs and the fact
columns of tables/evidence_levels.csv; the old decision lives on only as the
merge vote's evidence CLASS (`vote_classes`: 2 neutral backed, 1 formula
confirmed, 0 unconfirmed) and as the --corroborate cross set
(`_source_neutrals`).

So this file pins:
- every FACT the old level tests showed (asserted directly on the fact layer,
  `_level_pairs` / `_series_pooled`);
- where an old test pinned a per-file level, the merge vote's class that level
  gives (the only thing it still decides); where it pinned a pooled level,
  the cross-set membership it gives (the only thing that still reads it);
- the old golden vectors, recast as FACT vectors (the counts of each fact)
  and per-file VOTE vectors on the same fixture sets.

A B-series level letter is asserted nowhere: it reaches no output.

The builders here (`m0`, `child`, `ledger`, `_pooled`, ...) are shared by the
other level-fact tests.

Run: pytest tests/test_evidence.py -q
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from peaky.assignment import evidence as EV  # noqa: E402
from peaky.batch import iso_checks as IC  # noqa: E402
from peaky.chem import isotopes as ISO  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "levels"
REPO = Path(__file__).resolve().parents[1]
#: the private decision's level order (the vote class reads it; never shown)
ORDER = ["2b", "3a", "3b", "4a", "4b", "4c", "4d", "5a", "5b"]
#: the fact layer's per-pair booleans, in the order a fact vector counts them
FACT_ORDER = ["iso", "multiline", "carbon_ev", "chan2", "anchor", "branch", "reagent_only_iso", "ion_only", "tied",
              "below", "lead", "lead_lift", "lowconf", "saturated", "corroborated", "upair", "label_untie",
              "label_veto", "iso_veto"]
# The golden FACT vectors (pairs, FACT_ORDER counts) on the four fixture sets, each
# source corroborated by the neutrals the other holds at the private 4b or better
# by its own evidence (C8). Before peaky 0.10.0 these sets were pinned by their
# level vectors (tv 21/15/107/15/145/15/9/37/1009, tof 6/15/182/14/267/85/84/135/2576,
# orbi 0/10/217/9/205/138/0/35/1093, ur 4/4/0/330/375/293/0/82/73 over
# 2b/3a/3b/4a/4b/4c/4d/5a/5b); the facts under them did not move with the scale.
GOLDEN = {
    "tv": (1373, "386/21/306/238/171/230/43/0/287/236/0/0/394/799/124/0/0/0/0"),
    "tof": (3364, "958/4/649/495/300/455/414/0/285/1263/0/0/887/2218/120/0/0/0/0"),
    # with its lock table (rule H): three leads lifted (lead_lift), their flagged rows' anchors gone
    "orbi": (1707, "238/39/218/436/176/426/0/0/411/890/0/3/14/706/91/0/0/0/0"),
    # with its neutral-pair table (rule U, upair) and its lock table (one lead lifted)
    "ur": (1161, "251/18/250/687/7/0/0/0/6/37/0/1/17/30/0/334/0/0/0"),
}
#: the uronium set without either table, and with the pair table alone
UR_WITHOUT_PAIR = "251/18/250/688/7/0/0/0/6/37/1/0/17/30/0/0/0/0/0"
UR_NO_LOCK = "251/18/250/688/7/0/0/0/6/37/1/0/17/30/0/334/0/0/0"
#: the Orbitrap set without its lock table -- the base a leak guard compares with
ORBI_NO_LOCK = (1707, "238/39/218/436/179/426/0/0/411/890/3/0/14/706/91/0/0/0/0")
#: the merge vote's class over every committed M0 row of every file (rows, class 0/1/2),
#: each file levelled alone with its set's cross set -- what decides a merged winner
VOTE_GOLDEN = {"tv_nitrate": "587 431/47/109", "tv_bromide": "786 569/109/108", "tof": "8313 6789/829/695",
               "orbi": "9773 4965/2669/2139", "ur": "8076 684/7267/125"}

LEDGER_COLUMNS = [
    "role", "peak_id", "parent_peak_id", "iso_label", "neutral_formula", "adduct",
    "ion_formula", "mz", "height", "tier", "method", "confidence", "tied",
    "below_assignability", "degeneracy_density", "degeneracy_note", "resolvability",
    "series_unit", "anchor_peak_id", "isotopologues", "ppm_error_cal", "occurrence",
]


# --------------------------------------------------------------------------- builders
def ion_mz_of(neutral, adduct, ion=None, default=200.0) -> float:
    """The exact m/z of the ion a (neutral, adduct[, ion]) reading makes; `default` when it does not parse."""
    if not neutral or not adduct:
        return default
    counts = {e: v for e, v in EV.ion_composition(neutral, adduct, ion).items() if v}
    mz = ISO.mono_mz(counts, ISO.ion_sign(ion, adduct))
    return mz if mz == mz else default


def label_shift(label) -> float:
    """The exact shift a child label names from its parent (the parent-relative reading of the whole
    label; the tests' legacy '+<digit>' note is no part: '13C+1' is a 13C line)."""
    tot = 0.0
    for part in ISO.split_label(label):
        kind, v = ISO.parse_label_part(part)
        if kind == "set":
            tot += ISO.heavy_shift(v)
        elif kind == "alt":
            tot += ISO.heavy_shift(v[0])
        elif kind == "gen":
            tot += v * ISO.ISOTOPE_SPACING["13C"]
    return tot


def place(rows) -> list:
    """C11+c: a child line counts only at its label's exact spacing from its parent, so every synthetic
    child built without an m/z is put there (its parent's m/z + `label_shift`)."""
    mz = {r["peak_id"]: r["mz"] for r in rows if r.get("role") == "M0"}
    out = []
    for r in rows:
        if r.get("role") == "iso_child" and r.get("mz") is None and r.get("parent_peak_id") in mz:
            r = dict(r, mz=mz[r["parent_peak_id"]] + label_shift(r.get("iso_label")))
        out.append(r)
    return out


def m0(peak_id, neutral, adduct="[M-H]-", ion=None, mz=None, height=1000.0,
       tier="Assigned", method="pass2", confidence="High", tied=False, below=False,
       degeneracy=0.5, note="", resolvability="resolved", series_unit=None,
       anchor=None, isotopologues=""):
    """One committed neutral at its ion's exact m/z unless told otherwise; the defaults are deliberately
    uncorroborated."""
    if mz is None:
        mz = ion_mz_of(neutral, adduct, ion)
    return {
        "role": "M0", "peak_id": peak_id, "parent_peak_id": None, "iso_label": None,
        "neutral_formula": neutral, "adduct": adduct, "ion_formula": ion or neutral,
        "mz": mz, "height": height, "tier": tier, "method": method,
        "confidence": confidence, "tied": tied, "below_assignability": below,
        "degeneracy_density": degeneracy, "degeneracy_note": note,
        "resolvability": resolvability, "series_unit": series_unit,
        "anchor_peak_id": anchor, "isotopologues": isotopologues,
        "ppm_error_cal": 0.4, "occurrence": 0.9,
    }


def child(peak_id, parent, label, height, mz=None):
    """One isotope satellite hanging off an M0 row: at `mz`, else (`ledger`) at its label's exact spacing
    from its parent."""
    row = m0(peak_id, None, adduct=None, height=height, mz=mz)
    row["mz"] = mz
    row.update(role="iso_child", parent_peak_id=parent, iso_label=label,
               tier="Assigned", degeneracy_density=None)
    return row


def reagent(peak_id, formula, mz):
    row = m0(peak_id, formula, adduct=None, mz=mz)
    row.update(role="reagent", tier=None, method=None, confidence=None)
    return row


def ledger(rows) -> pd.DataFrame:
    return pd.DataFrame(place(rows), columns=LEDGER_COLUMNS)


def facts_of(rows, **kw) -> dict:
    """{(neutral, adduct): fact row} of the fact layer over ONE synthetic ledger (levelled alone, as
    the vote reads a file); `kw` = cross / resolution / upair / label / iso."""
    out = EV._level_pairs({"": ledger(rows)}, per_file=True, **kw)
    return {(r.neutral_formula, r.adduct): r for r in out.itertuples(index=False)}


def pooled(per_file: dict, **kw) -> dict:
    """{(neutral, adduct): fact row} of the fact layer over a pooled source ({label: ledger})."""
    out = EV._series_pooled(per_file, **kw)
    return {(r.neutral_formula, r.adduct): r for r in out.itertuples(index=False)}


def vote_of(rows, **kw) -> dict:
    """{(neutral, adduct): the merge vote's class} of the M0 rows of one synthetic file."""
    led = ledger(rows)
    cls = EV.vote_classes(led, **kw)
    m = led.loc[cls.index]
    return dict(zip(zip(m.neutral_formula, m.adduct), cls))


def crosses(frame: pd.DataFrame) -> set:
    """The pairs of a pooled private decision a --corroborate source offers: the private level 4b or
    better, not ion-only (`_source_neutrals`'s rule)."""
    rank = {lv: i for i, lv in enumerate(EV._SERIES_ORDER)}
    ok = frame["evidence_level"].map(lambda v: rank.get(v, 99) <= rank["4b"]) & ~frame["ion_only"].astype(bool)
    return set(zip(frame.loc[ok, "neutral_formula"], frame.loc[ok, "adduct"]))


def _vector(frame: pd.DataFrame) -> str:
    """The fact vector of a fact-layer frame (FACT_ORDER counts)."""
    return "/".join(str(int(frame[c].astype(bool).sum())) for c in FACT_ORDER)


def _vote_vector(per_file: dict, cross=None) -> str:
    c = {0: 0, 1: 0, 2: 0}
    n = 0
    for f in per_file.values():
        v = EV.vote_classes(EV.trim(f), cross=cross)
        n += len(v)
        for k, cnt in v.value_counts().items():
            c[int(k)] += int(cnt)
    return f"{n} {c[0]}/{c[1]}/{c[2]}"


def series_reference():
    """scripts/level_ledger.py as it stood before peaky 0.10.0 (the private decision's
    independent implementation), or skip: the script is being rewritten as the scale's
    decision reference, after which the private decision has no second implementation."""
    spec = importlib.util.spec_from_file_location("level_ledger", REPO / "scripts" / "level_ledger.py")
    LL = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(LL)
    return LL


# --------------------------------------------------------------------------- the vote's class, one case per old level
# (rows, pair, facts that must hold, the vote class) for the passing fixture and its mutant
CASES = {
    # tied on every row is a hard input: class 0; untied, the anchor alone confirms the formula: class 1
    "tied": (([m0("p", "C6H8O4", tied=True, anchor="a")], ("C6H8O4", "[M-H]-"), dict(tied=True, anchor=True), 0),
             ([m0("p", "C6H8O4", tied=False, anchor="a")], ("C6H8O4", "[M-H]-"), dict(tied=False, anchor=True), 1)),
    # a curated compound-scope identity on one structure, and on three: both neutral backed
    "known": (([m0("p", "HNO3", method="known:atmospheric")], ("HNO3", "[M-H]-"), dict(known_fam="atmospheric"), 2),
              ([m0("p", "C6H5NO3", method="known:nitroaromatic")], ("C6H5NO3", "[M-H]-"),
               dict(known_fam="nitroaromatic"), 2)),
    # a curated class; not curated, unique and resolved with no axis: the formula only
    "class": (([m0("p", "C8HF15O2", method="known:perfluoroacid")], ("C8HF15O2", "[M-H]-"),
               dict(known_fam="perfluoroacid"), 2),
              ([m0("p", "C8HF15O2", method="pass2")], ("C8HF15O2", "[M-H]-"), dict(known_fam=""), 1)),
    # the acid branch (bare + cluster) backs the neutral; two cluster channels are a second channel only
    "branch": (([m0("p1", "C10H16O5"), m0("p2", "C10H16O5", adduct="[M+NO3]-")], ("C10H16O5", "[M+NO3]-"),
                dict(chan2=True, branch=True), 2),
               ([m0("p1", "C10H16O5", adduct="[M+NO3]-"), m0("p2", "C10H16O5", adduct="[M+^NO3]-")],
                ("C10H16O5", "[M+NO3]-"), dict(chan2=True, branch=False), 1)),
    # unopposed on a separable peak: the formula confirmed; blended: not
    "separable": (([m0("p", "C7H12O3", degeneracy=0.5, resolvability="resolved")], ("C7H12O3", "[M-H]-"),
                   dict(res_ok=True), 1),
                  ([m0("p", "C7H12O3", degeneracy=0.5, resolvability="blended")], ("C7H12O3", "[M-H]-"),
                   dict(res_ok=False), 0)),
    # exact mass in a dense window: unconfirmed; one axis (the anchor) confirms it
    "dense": (([m0("p", "C7H12O4", degeneracy=2.0)], ("C7H12O4", "[M-H]-"), dict(anchor=False), 0),
              ([m0("p", "C7H12O4", degeneracy=2.0, series_unit="CH2")], ("C7H12O4", "[M-H]-"), dict(anchor=True), 1)),
    # one axis; two axes inside the chemistry (anchor + a second cluster channel) are still not the neutral
    "one_axis": (([m0("p", "C9H14O4", series_unit="CH2")], ("C9H14O4", "[M-H]-"), dict(anchor=True), 1),
                 ([m0("p", "C9H14O4", adduct="[M+NO3]-", series_unit="CH2"), m0("q", "C9H14O4", adduct="[M+^NO3]-")],
                  ("C9H14O4", "[M+NO3]-"), dict(anchor=True, chan2=True, branch=False), 1)),
    # the reagent halogen's own satellite pins the ion, not the neutral; carbon pins the neutral (both class 1)
    "reagent_sat": (([m0("p", "C8H14O2", adduct="[M+Br]-", ion="C8H14O2Br"), child("c", "p", "81Br+1", 950.0),
                      m0("q", "C9H16O2", adduct="[M+Br]-", ion="C9H16O2Br")], ("C8H14O2", "[M+Br]-"),
                     dict(iso=True, reagent_only_iso=True, carbon_ev=False), 1),
                    ([m0("p", "C8H14O2", adduct="[M+Br]-", ion="C8H14O2Br"), child("c", "p", "81Br+1", 950.0),
                      child("c2", "p", "13C+1", 86.0), m0("q", "C9H16O2", adduct="[M+Br]-", ion="C9H16O2Br")],
                     ("C8H14O2", "[M+Br]-"), dict(iso=True, reagent_only_iso=False, carbon_ev=True), 1)),
}


@pytest.mark.parametrize("name", list(CASES))
def test_each_case_and_its_mutant_read_their_facts_and_give_the_votes_class(name):
    """The facts each old per-level case turned on, and the merge vote's class
    the private decision gives them (the B-series level they read is never
    output; the class is what it still decides)."""
    for rows, key, want, cls in CASES[name]:
        f = facts_of(rows)[key]
        assert {k: getattr(f, k) for k in want} == want, (name, key)
        assert vote_of(rows)[key] == cls, (name, key)


def test_hard_inputs_each_leave_the_reading_unconfirmed():
    """Tied, Low confidence, below assignability and a mass-saturated window are
    hard inputs of the private decision: each reads class 0 whatever the axes.
    (Under the scale, Low and below are step-0 rejections, tests/test_levels_decide.py.)"""
    k = ("C6H8O4", "[M-H]-")
    for rows, fact in (([m0("p", "C6H8O4", confidence="Low", anchor="a")], "lowconf"),
                       ([m0("p", "C6H8O4", below=True, anchor="a")], "below"),
                       ([m0("p", "C6H8O4", degeneracy=5.0)], None)):
        if fact:
            assert getattr(facts_of(rows)[k], fact)
        assert vote_of(rows)[k] == 0


def test_a_compound_scope_identity_absent_from_the_isomer_space_is_still_a_curated_family():
    for rows, key, fam in (([m0("p", "C2HF3O2", method="known:perfluoroacid")], ("C2HF3O2", "[M-H]-"), "perfluoroacid"),
                           ([m0("p", "C99H99O99", method="known:atmospheric")], ("C99H99O99", "[M-H]-"), "atmospheric")):
        assert facts_of(rows)[key].known_fam == fam and vote_of(rows)[key] == 2


def test_the_corroborating_source_backs_the_neutral_with_or_without_an_isotope_axis():
    """A 13C line is the isotope axis; the --corroborate source's agreement is
    the axis that backs the neutral (class 2) -- with the line or without it."""
    k = ("C10H16O4", "[M-H]-")
    ok = [m0("p", "C10H16O4", ion="C10H15O4", height=1000.0), child("c", "p", "13C+1", 107.0)]
    bare = [m0("p", "C10H16O4", ion="C10H15O4", height=1000.0)]
    assert facts_of(ok)[k].iso and facts_of(ok)[k].carbon_ev and not facts_of(bare)[k].iso
    assert facts_of(ok, cross={"C10H16O4"})[k].corroborated
    assert vote_of(ok, cross={"C10H16O4"})[k] == vote_of(bare, cross={"C10H16O4"})[k] == 2
    assert vote_of(ok)[k] == 1                                # the line alone: the formula, not the neutral
    # a two-line envelope of the neutral's own elements is itself an outside axis
    ml = [m0("p", "C10H16O4S", ion="C10H15O4S", height=1000.0, series_unit="CH2"),
          child("c", "p", "13C+1", 107.0), child("s", "p", "34S+2", 44.0)]
    f = facts_of(ml)[("C10H16O4S", "[M-H]-")]
    assert f.multiline and f.multiline_elements == "C|S" and vote_of(ml)[("C10H16O4S", "[M-H]-")] == 2


def test_a_brominated_neutrals_own_81br_line_on_a_nitrate_channel_is_its_own_isotope_line():
    """No reagent halogen on a nitrate channel: the 81Br line is ordinary isotope
    evidence (the formula confirmed), and a Br-free ion makes no 81Br line at all (C11+c)."""
    nitrate = [m0("p", "C8H13BrO2", adduct="[M+NO3]-", ion="C8H13BrNO5-", mz=281.9983),
               child("c", "p", "81Br", 950.0, mz=281.9983 + 1.9979535),
               m0("q", "C9H16O2", adduct="[M+NO3]-")]
    f = facts_of(nitrate)[("C8H13BrO2", "[M+NO3]-")]
    assert f.iso and not f.reagent_only_iso and vote_of(nitrate)[("C8H13BrO2", "[M+NO3]-")] == 1
    brfree = [m0("p", "C8H14O2", adduct="[M+NO3]-", ion="C8H14NO5-", mz=204.0877),
              child("c", "p", "81Br", 950.0, mz=204.0877 + 1.9979535),
              m0("q", "C9H16O2", adduct="[M+NO3]-")]
    f = facts_of(brfree)[("C8H14O2", "[M+NO3]-")]
    assert not f.iso and not f.reagent_only_iso and vote_of(brfree)[("C8H14O2", "[M+NO3]-")] == 1


def _bromide_channel(*rows):
    """`rows` on a bromide channel: two [M+Br]- commits make Br the reagent halogen."""
    return [*rows, m0("q", "C9H16O2", adduct="[M+Br]-", ion="C9H16O2Br-"),
            m0("r", "C9H18O2", adduct="[M+Br]-", ion="C9H18O2Br-")]


def test_the_reagent_satellite_needs_the_ion_to_carry_the_reagent_halogen():
    """C11+a: `reagent_only_iso` clears only where the ion's reagent halogen is
    all the neutral's own -- on a brominated neutral's [M-H]- the 81Br line is
    the neutral's. The reagent's line (the ion carries more of the halogen than
    the neutral) keeps the flag. An ion carrying NONE of it: a 1:1 +2 Da line
    on a Br-free ion is no line of that ion (C11+c: its count-aware expectation
    is 0, never in band, so the pair has no isotope axis) -- and C11+c released
    the 2026-09-27 hold: the flag means only "the reagent put the halogen on the
    ion". (The old level each case read -- 4c / 4b / 4d -- is the vote's class 1
    in every case; the facts are what moved.)"""
    def facts(rows):
        out = pooled({"f": ledger(rows)})
        return {k: (bool(r.iso), bool(r.reagent_only_iso)) for k, r in out.items()}
    br = 1.9979535
    brfree = _bromide_channel(m0("p", "C8H14O4", ion="C8H13O4-", mz=173.0819),
                              child("c", "p", "81Br", 950.0, mz=173.0819 + br))
    assert facts(brfree)[("C8H14O4", "[M-H]-")] == (False, False)
    own = _bromide_channel(m0("p", "C7H11BrO4", ion="C7H10BrO4-", mz=236.9768),
                           child("c", "p", "81Br", 950.0, mz=236.9768 + br))
    assert facts(own)[("C7H11BrO4", "[M-H]-")] == (True, False)
    # the reagent's own line: a bromide adduct of a Br-free neutral, and of a brominated one (Br2 > Br)
    adduct = _bromide_channel(m0("p", "C8H14O2", adduct="[M+Br]-", ion="C8H14O2Br-", mz=221.0183),
                              child("c", "p", "81Br", 950.0, mz=221.0183 + br))
    assert facts(adduct)[("C8H14O2", "[M+Br]-")] == (True, True)
    # the Br2 ion is committed on its 79Br81Br line (the scorer's most abundant): its 81Br2 line sits
    # 1.998 Da above it at 0.486x (C11+c; against the mono line the same child at 0.95x of a 1.9456
    # expectation falls out of band)
    more = _bromide_channel(m0("p", "C7H11BrO4", adduct="[M+Br]-", ion="C7H11Br2O4-", mz=316.9030 + br),
                            child("c", "p", "81Br2", 486.0, mz=316.9030 + 2 * br))
    assert facts(more)[("C7H11BrO4", "[M+Br]-")] == (True, True)
    mono = _bromide_channel(m0("p", "C7H11BrO4", adduct="[M+Br]-", ion="C7H11Br2O4-", mz=316.9030),
                            child("c", "p", "81Br", 950.0, mz=316.9030 + br))
    assert facts(mono)[("C7H11BrO4", "[M+Br]-")] == (False, True)
    # a ledger row that stored the NEUTRAL as its ion formula: the adduct carries the reagent
    stored = _bromide_channel(m0("p", "C8H14O2", adduct="[M+Br]-", ion="C8H14O2", mz=221.0183),
                              child("c", "p", "81Br", 950.0, mz=221.0183 + br))
    assert facts(stored)[("C8H14O2", "[M+Br]-")] == (True, True)
    for rows in (brfree, own, adduct, more, mono, stored):
        assert set(vote_of(rows).values()) == {1}
    assert EV.carries_reagent("C8H14O2", "[M+HBr+Br]-", "C8H14O2", "Br")
    assert not EV.carries_reagent("C7H11BrO4", "[M+NO3]-", "C7H11BrNO7-", "Br")
    assert not EV.carries_reagent("C8H14O2", "[M+Br]-", "C8H14O2Br-", None)
    # the hold is released: an ion carrying none of the halogen does not carry the reagent's
    assert not hasattr(EV, "not_the_neutrals_line")
    assert not EV.carries_reagent("C8H14O4", "[M-H]-", "C8H13O4-", "Br")
    assert not EV.carries_reagent("C6H11NO6S", "[M+NO3]-", "C6H11N2O9S-", "Br")
    assert not EV.carries_reagent("C7H11BrO4", "[M-H]-", "C7H10BrO4-", "Br")
    assert not EV.carries_reagent("HBrO", "[M+NO3]-", "HBrNO4-", "Br")
    assert EV.carries_reagent("C8H14O2", "[M+Br]-", "C8H14O2Br-", "Br")
    assert EV.ion_composition("C8H14O2", "[M+HBr+Br]-", "nan") == {"C": 8, "H": 15, "O": 2, "Br": 2}


def test_the_reference_script_reads_the_reagent_satellite_like_the_fact_layer():
    LL = series_reference()
    br = 1.9979535
    rows = ledger(_bromide_channel(
        m0("p", "C8H14O4", ion="C8H13O4-", mz=173.0819), child("c", "p", "81Br", 950.0, mz=173.0819 + br),
        m0("s", "C7H11BrO4", ion="C7H10BrO4-", mz=236.9768), child("d", "s", "81Br", 950.0, mz=236.9768 + br),
        m0("t", "C8H14O2", adduct="[M+Br]-", ion="C8H14O2", mz=221.0183),
        child("e", "t", "81Br", 950.0, mz=221.0183 + br),
        m0("u", "C7H11BrO4", adduct="[M+Br]-", ion="C7H11Br2O4-", mz=316.9030 + br),
        child("g", "u", "81Br2", 486.0, mz=316.9030 + 2 * br)))
    core = EV._series_pooled({"s1": rows})
    ref = LL.assign_levels(LL.measure_source("s1", rows.assign(__file="s1"), "Br"), set())
    m = core.merge(ref, left_on=["neutral_formula", "adduct"], right_on=["neutral", "adduct"])
    assert len(m) == len(core) == 6
    assert (m["reagent_only_iso_x"] == m["reagent_only_iso_y"]).all() and (m["evidence_level"] == m["level"]).all()
    assert set(m.loc[m["reagent_only_iso_x"], "neutral_formula"]) == {"C8H14O2", "C7H11BrO4"}
    assert set(m.loc[m["reagent_only_iso_x"], "adduct"]) == {"[M+Br]-"}
    assert LL.ion_composition("C8H14O2", "[M+HBr+Br]-", float("nan")) == {"C": 8, "H": 15, "O": 2, "Br": 2}
    assert not LL.carries_reagent("C7H11BrO4", "[M-H]-", "C7H10BrO4-", "Br")
    assert not LL.carries_reagent("C8H14O4", "[M-H]-", "C8H13O4-", "Br")
    assert not hasattr(LL, "not_the_neutrals_line")


# --------------------------------------------------------------------------- contract
def test_non_m0_rows_carry_no_level_and_the_scale_replaced_the_old_columns():
    """The per-file stage stamps the scale's columns on M0 rows only; a reagent
    ion and an isotope child carry none. A ledger with no width model is
    class-less: its M0 rows read NA (not assessed)."""
    rows = [m0("p", "C6H8O4", ion="C6H7O4", height=1000.0), child("c", "p", "13C+1", 66.0),
            reagent("r", "Br", 78.918)]
    led = ledger(rows)
    EV.apply_levels(led)
    by = dict(zip(led.peak_id, led.evidence_level))
    assert by["p"] == "NA" and pd.isna(by["c"]) and pd.isna(by["r"])
    assert dict(zip(led.peak_id, led.claim))["p"] == "not assessed"
    for col in EV.COLUMNS:
        assert col in led.columns
    for col in ("evidence_axes", "level_reason", "n_plausible_structures"):
        assert col not in led.columns


def test_the_private_axes_string_keeps_the_order_the_vote_reads():
    """The vote reads `corroborated` from the private axes string (`_vote_class_of`), and a merged
    ledger written before peaky 0.10.0 is read through it (`_stored_own_good`)."""
    rows = [m0("p", "C10H16O4", ion="C10H15O4", height=1000.0, series_unit="CH2"),
            child("c", "p", "13C+1", 107.0)]
    out = EV._series_levels(ledger(rows), cross={"C10H16O4"})
    row = out.iloc[0]
    assert row.evidence_axes.split("|")[:3] == ["iso", "anchor", "corroborated"]
    assert EV._vote_class_of(row.evidence_level, row.evidence_axes) == 2
    assert EV._vote_class_of("5b", "files:3|corroborated") == 2      # the source's agreement backs any reading
    assert EV._vote_class_of("4c", "") == 1 and EV._vote_class_of("5a", "") == 0
    assert EV._vote_class_of(None, None) == 0 and EV._vote_class_of(float("nan"), float("nan")) == 0


def test_missing_columns_and_nulls_do_not_raise():
    led = ledger([m0("p", "C6H8O4")]).drop(columns=["resolvability", "degeneracy_note", "isotopologues"])
    led["tied"] = pd.NA
    led["confidence"] = None
    out = EV._series_levels(led)
    assert out.evidence_level.iloc[0] in ORDER
    assert list(EV.vote_classes(led)) in ([0], [1], [2])
    EV.apply_levels(led)
    assert led.loc[led.role == "M0", "evidence_level"].iloc[0] == "NA"


def test_isomer_space_rows_carry_a_rationale_and_cover_the_one_structure_formulas():
    """The isomer space the private decision reads (one structure: a curated
    identity backs the neutral alone)."""
    iso = EV.load_isomer_space()
    assert {"formula", "n_plausible_structures", "family", "name", "rationale"} <= set(iso.columns)
    assert iso.rationale.astype(str).str.strip().str.len().gt(0).all()
    assert iso.formula.is_unique
    exp = pd.read_csv(FIXTURES / "expected_levels.csv")
    for f in exp.loc[exp.level == "2b", "neutral"].unique():
        assert f in set(iso.formula), f
        assert int(iso.loc[iso.formula == f, "n_plausible_structures"].iloc[0]) == 1
    for fam in ("atmospheric", "reactive_iodine", "nitroaromatic", "perfluoroacid", "chlorinated_paraffin"):
        assert fam in EV.KNOWN_FAMILY_SCOPE


# --------------------------------------------------------------------------- goldens
def _read(name: str) -> pd.DataFrame:
    return pd.read_csv(FIXTURES / f"{name}_ledger.csv.gz", low_memory=False)


def _pooled(prefix: str) -> dict:
    files = sorted(FIXTURES.glob(f"{prefix}_*_ledger.csv.gz"))
    return {p.name[:16]: pd.read_csv(p, low_memory=False) for p in files}


def _iso(prefix: str) -> dict:
    """A golden set's rule H lock table as the fact layer reads it (C11+b)."""
    return IC.facts(pd.read_csv(FIXTURES / f"{prefix}_iso_checks.csv"))


def _ur_pairs() -> set:
    t = pd.read_csv(FIXTURES / "ur_neutral_pairs.csv")
    return set(t.loc[t["upair"].astype(bool), "neutral_formula"])


@pytest.fixture(scope="module")
def expected():
    return pd.read_csv(FIXTURES / "expected_levels.csv")


def test_golden_tv_two_channels_corroborate_each_other():
    no3, br = _read("tv_nitrate"), _read("tv_bromide")
    cross_for_no3 = EV._source_neutrals({"tv_bromide": br})
    cross_for_br = EV._source_neutrals({"tv_nitrate": no3})
    a = EV._series_pooled({"tv_nitrate": no3}, cross=cross_for_no3)
    b = EV._series_pooled({"tv_bromide": br}, cross=cross_for_br)
    both = pd.concat([a, b])
    assert (len(both), _vector(both)) == GOLDEN["tv"]
    assert _vote_vector({"tv_nitrate": no3}, cross_for_no3) == VOTE_GOLDEN["tv_nitrate"]
    assert _vote_vector({"tv_bromide": br}, cross_for_br) == VOTE_GOLDEN["tv_bromide"]


def test_golden_same_air_pair():
    tof, orbi = _pooled("tof"), _pooled("orbi")
    n_tof, n_orbi = EV._source_neutrals(tof), EV._source_neutrals(orbi)
    t = EV._series_pooled(tof, cross=n_orbi)
    o = EV._series_pooled(orbi, cross=n_tof, iso=_iso("orbi"))
    assert (len(t), _vector(t)) == GOLDEN["tof"]
    assert (len(o), _vector(o)) == GOLDEN["orbi"]
    no_lock = EV._series_pooled(orbi, cross=n_tof)
    assert (len(no_lock), _vector(no_lock)) == ORBI_NO_LOCK
    assert _vote_vector(tof, n_orbi) == VOTE_GOLDEN["tof"]
    assert _vote_vector(orbi, n_tof) == VOTE_GOLDEN["orbi"]


def test_golden_uronium_neutral_pair():
    """Rule U on the uronium set: the pair table marks 334 pairs `upair`; the
    lock table (C11+b) lifts one lead. Neither moves any other fact."""
    ur = _pooled("ur")
    with_pair = EV._series_pooled(ur, upair=_ur_pairs(), iso=_iso("ur"))
    assert (len(with_pair), _vector(with_pair)) == GOLDEN["ur"]
    assert _vector(EV._series_pooled(ur, upair=_ur_pairs())) == UR_NO_LOCK
    assert _vector(EV._series_pooled(ur)) == UR_WITHOUT_PAIR
    assert _vote_vector(ur) == VOTE_GOLDEN["ur"]


#: the reference script's per-row facts in expected_levels.csv (tests/fixtures/levels/README.md)
FIXTURE_FACTS = ["iso", "chan2", "anchor", "corroborated", "branch", "reagent_only_iso", "known_fam", "tied",
                 "below", "lowconf", "degeneracy", "saturated", "res_ok", "n_files", "n_axes"]


def _cell(v) -> str:
    """One fact cell as text, alike from the CSV and from the engine (bools, integral floats, NaN)."""
    if v is None or (isinstance(v, float) and v != v):
        return ""
    if isinstance(v, bool) or type(v).__name__ == "bool_":
        return str(bool(v))
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)


def _engine_on_the_fixtures() -> pd.DataFrame:
    tof, orbi = _pooled("tof"), _pooled("orbi")
    no3, br = _read("tv_nitrate"), _read("tv_bromide")
    return pd.concat([
        EV._series_pooled(tof, cross=EV._source_neutrals(orbi)).assign(source="tof"),
        EV._series_pooled(orbi, cross=EV._source_neutrals(tof), iso=_iso("orbi")).assign(source="orbi"),
        EV._series_pooled(_pooled("ur"), upair=_ur_pairs(), iso=_iso("ur")).assign(source="ur"),
        EV._series_pooled({"tv_nitrate": no3}, cross=EV._source_neutrals({"tv_bromide": br})).assign(source="tv_nitrate"),
        EV._series_pooled({"tv_bromide": br}, cross=EV._source_neutrals({"tv_nitrate": no3})).assign(source="tv_bromide")])


def test_every_fixture_row_matches_the_reference_script_fact_for_fact(expected):
    """Every fact the reference script recorded for a fixture row (iso,
    reagent_only_iso, chan2, ...) is the engine's, on all five sources. Its
    level column is the pre-0.10.0 decision: what that level still decides --
    whether the pair is one a --corroborate source offers (the private 4b or
    better, not ion-only) -- is the engine's too."""
    got = _engine_on_the_fixtures()
    m = expected.merge(got, left_on=["source", "neutral", "adduct"],
                       right_on=["source", "neutral_formula", "adduct"], how="left", suffixes=("", "_engine"))
    assert len(m) == len(expected) and m.evidence_level.notna().all(), "every reference row must be read"
    for c in FIXTURE_FACTS:
        a, b = m[c].map(_cell), m[c + "_engine"].map(_cell)
        bad = m.loc[a != b, ["source", "neutral", "adduct", c, c + "_engine"]]
        assert bad.empty, (c, bad.head(10).to_dict("records"))
    rank = {lv: i for i, lv in enumerate(EV._SERIES_ORDER)}
    ion_only = m["ion_only"].astype(bool)            # the engine's (the reference file records no such column)
    offered_ref = m["level"].map(lambda v: rank.get(v, 99) <= rank["4b"]) & ~ion_only
    offered_core = m["evidence_level"].map(lambda v: rank.get(v, 99) <= rank["4b"]) & ~ion_only
    bad = m.loc[offered_ref != offered_core, ["source", "neutral", "adduct"]]
    assert bad.empty, bad.head(10).to_dict("records")


def test_pooled_facts_follow_the_reference_pooling():
    """A pooled source is the reference pooling: all rows of a pair across
    files decide tied / lowconf, any row decides below, the channels are read
    over every file (bare in f1 and clustered in f2: the acid branch)."""
    a = [m0("p", "C9H14O4", tied=True)]
    b = [m0("q", "C9H14O4", tied=False, adduct="[M+NO3]-")]
    out = pooled({"f1": ledger(a), "f2": ledger(b)})
    assert out[("C9H14O4", "[M-H]-")].tied                         # that pair's only row is tied
    assert not out[("C9H14O4", "[M+NO3]-")].tied
    assert out[("C9H14O4", "[M+NO3]-")].branch and out[("C9H14O4", "[M+NO3]-")].chan2
    assert out[("C9H14O4", "[M+NO3]-")].n_files == 1
    both = pooled({"f1": ledger([m0("p", "C9H14O4", tied=True)]), "f2": ledger([m0("p", "C9H14O4", tied=False)])})
    assert not both[("C9H14O4", "[M-H]-")].tied and both[("C9H14O4", "[M-H]-")].n_files == 2
    low = pooled({"f1": ledger([m0("p", "C9H14O4", confidence="Low")]), "f2": ledger([m0("p", "C9H14O4")])})
    assert not low[("C9H14O4", "[M-H]-")].lowconf
    below = pooled({"f1": ledger([m0("p", "C9H14O4", below=True)]), "f2": ledger([m0("p", "C9H14O4")])})
    assert below[("C9H14O4", "[M-H]-")].below


def test_a_satellite_hangs_off_its_parent_in_the_same_file():
    """Two files may reuse a peak id: a child joins the M0 with that id in ITS
    file only (spec section 2), so file 1's neutral gains no isotope axis from
    file 2's satellite."""
    f1 = ledger([m0("p", "C9H14O4", ion="C9H13O4", height=1000.0)])
    f2 = ledger([m0("p", "C10H16O4", ion="C10H15O4", height=1000.0), child("c", "p", "13C+1", 107.0)])
    out = pooled({"f1": f1, "f2": f2})
    assert out[("C10H16O4", "[M-H]-")].iso and out[("C10H16O4", "[M-H]-")].carbon_ev
    assert not out[("C9H14O4", "[M-H]-")].iso and not out[("C9H14O4", "[M-H]-")].carbon_ev


def test_pooled_pairs_are_m0_only_a_reagent_row_forms_no_pair():
    """The pooled fact table (tables/evidence_levels.csv's fact columns) has one
    row per committed (neutral, adduct): a reagent ion is excluded by role."""
    out = EV._series_pooled({"f": ledger([m0("p", "C6H8O4"), reagent("r", "Br", 78.918)])})
    assert list(out.neutral_formula) == ["C6H8O4"]


# --------------------------------------------------------------------------- the cross set (C8)
def test_a_source_corroborates_only_what_it_holds_at_4b_or_better_by_its_own_evidence():
    """The vote's `corroborated` axis is a second SIGHTING of the neutral: the
    source must pin it by an axis of its own (the private 4b) or better. A
    unique reading with no axis (4c), a tied one (5b) or an ion-only pair is the
    source's grid enumerating the formula, not a sighting of it."""
    src = ledger([
        m0("a", "C10H16O4", ion="C10H15O4", height=1000.0), child("a1", "a", "13C+1", 107.0),   # iso
        m0("b", "C8HF15O2", method="known:perfluoroacid"),                                     # a curated class
        m0("c", "C7H12O4"),                                                                    # no axis
        m0("d", "C6H8O4", tied=True, anchor="a"),                                              # tied
        m0("e", "C9H14O4", adduct="[M]-.", method="ion_only:electron_attachment"),  # ion-only
    ])
    assert EV._source_neutrals({"s": src}) == {"C10H16O4", "C8HF15O2"}
    assert EV._source_neutrals({"s": src}, max_level="4c") == {"C10H16O4", "C8HF15O2", "C7H12O4"}
    assert EV.vote_cross_neutrals([src]) == {"C10H16O4", "C8HF15O2"}


def test_two_sources_that_only_agree_cannot_lift_each_other():
    """Two instruments whose grids both fit a mass-degenerate formula with no
    axis used to hand each other the `corroborated` axis; each is now
    corroborated only by what the other pins on its own -- so neither lifts the
    other's reading in the merge vote."""
    x = ledger([m0("p", "C6H10O4", degeneracy=5.0)])
    y = ledger([m0("q", "C6H10O4", degeneracy=5.0, adduct="[M+NO3]-")])
    for me, other in ((x, y), (y, x)):
        cross = EV._source_neutrals({"other": other})
        assert cross == set()
        assert set(EV.vote_classes(me, cross=cross)) == {0}
        # the mutant: any-level membership lifts the reading on the agreement alone
        assert set(EV.vote_classes(me, cross={"C6H10O4"})) == {2}


def test_a_per_file_source_is_relevelled_not_read():
    """A ledger with predicate columns is levelled afresh with no cross set:
    a stored level and axes the source owed to ITS OWN --corroborate do not count."""
    src = ledger([m0("p", "C9H14O4", degeneracy=None)])          # no axis, degeneracy unmeasured
    src["evidence_level"], src["evidence_axes"] = "4a", "iso|corroborated"
    assert EV._source_neutrals({"s": src}) == set()


def test_a_merged_ledger_source_reads_its_stored_level_without_its_own_corroboration():
    """A merged ledger written before peaky 0.10.0 carries the old level + axes:
    it is read without its own `corroborated` axis. A merged ledger of the scale
    carries no such class and is refused (name the run dir)."""
    merged = pd.DataFrame([
        dict(neutral_formula="A1", adduct="[M-H]-", evidence_level="4b", evidence_axes="iso|files:3"),
        dict(neutral_formula="B1", adduct="[M-H]-", evidence_level="4b", evidence_axes="corroborated|files:3"),
        dict(neutral_formula="C1", adduct="[M-H]-", evidence_level="4a", evidence_axes="iso|corroborated|carbon|files:2"),
        dict(neutral_formula="D1", adduct="[M+Br]-", evidence_level="4a",
             evidence_axes="iso|corroborated|reagent_only_iso|files:2"),
        dict(neutral_formula="E1", adduct="[M-H]-", evidence_level="5b", evidence_axes="files:9"),
        dict(neutral_formula="F1", adduct="[M-H]-", evidence_level="3a",
             evidence_axes="corroborated|known:perfluoroacid|files:4"),
        dict(neutral_formula="G1", adduct="[M-H]-", evidence_level="3b", evidence_axes="chan2|branch|files:5"),
        dict(neutral_formula="H1", adduct="[M]-.", evidence_level="4d", evidence_axes="iso|ion_only|files:5",
             ion_only_of="C9H14O4 [M-H]-"),
    ])
    assert EV._source_neutrals({"m": merged}) == {"A1", "C1", "G1"}
    with pytest.raises(ValueError, match="evidence_level"):
        EV._source_neutrals({"m": merged.drop(columns=["evidence_level", "evidence_axes"])})
    scale = merged.drop(columns=["evidence_axes"]).assign(evidence_level="4b")
    with pytest.raises(ValueError, match="name the run dir"):
        EV._source_neutrals({"m": scale})


def test_the_cross_set_equals_the_reference_scripts_on_the_golden_sets():
    """The in-core helper and the pre-0.10.0 reference script pick the same neutrals."""
    LL = series_reference()
    for prefix in ("tof", "orbi"):
        files = _pooled(prefix)
        frame = pd.concat([f.assign(__file=k) for k, f in files.items()], ignore_index=True)
        halogen = LL.detect_reagent_halogen(frame[frame.role == "M0"])
        measured = LL.measure_source(prefix, frame, halogen)
        measured["reagent_halogen"] = halogen or ""
        assert LL.own_good_neutrals(measured) == EV._source_neutrals(files), prefix
