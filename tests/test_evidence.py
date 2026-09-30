"""Evidence levels -- the contract of docs/EVIDENCE_LEVELS.md, pinned before
the code existed (the module failed to collect with ImportError until
`peaky.assignment.evidence` shipped).

`scripts/level_ledger.py` is the executable reference; the fixtures under
tests/fixtures/levels/ and their expected_levels.csv were written by it.
Verified by mutation: every predicate reverted, the 4b/4c boundary swapped,
a reagent row let through, a satellite joined across files -- each fails a
test here or in tests/test_evidence_outputs.py.

Run: pytest tests/test_evidence.py -q
"""

from __future__ import annotations

import gzip
import io
from pathlib import Path

import pandas as pd
import pytest

from peaky.assignment import evidence as EV  # noqa: E402
from peaky.batch import iso_checks as IC  # noqa: E402
from peaky.chem import isotopes as ISO  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "levels"
ORDER = ["2b", "3a", "3b", "4a", "4b", "4c", "4d", "5a", "5b"]
# the three golden vectors -- each source corroborated by the neutrals the other
# holds at 4b or better by its OWN evidence (C8, 2026-09-25; before it, by every
# M0 neutral of the other at any level: 21/15/107/38/162/9/9/33/979,
# 6/16/182/38/260/79/91/135/2557 and 0/12/217/44/215/119/0/30/1070). C17
# (2026-09-27, `multiline` counts in-band elements the neutral supplies) moved
# the TOF set only: seven bromide-adduct 4a rows whose second line was the
# reagent's 81Br go to 4b; it was 6/15/182/22/247/82/99/138/2573. C11+a (2026-09-27,
# `reagent_only_iso` only where the ION carries more of the reagent halogen than
# the neutral) moved it again: six bromide-channel [M-H]- rows 4d -> 4b (four
# brominated neutrals whose 81Br line is their own, two Br-free ions); it was
# 6/15/182/15/254/82/99/138/2573.
# C11+b (rule H, decision D8, 2026-09-28): the Orbitrap and the uronium sets are
# levelled with their lock tables (orbi_iso_checks.csv / ur_iso_checks.csv: the
# rule H lock rows of the live runs), and the four live-locked lead pairs carry
# their flag in tentative_lead -- the Orbitrap set's HBr [M+^NO3]-, C6H9ClO3 and
# C6H10Cl2O4 [M-H]- and the uronium set's C7H11ClO2 [M+(CH4N2O)H]+ lift 5b -> 4b.
# Without the tables the vectors are as before (ORBI_NO_LOCK, UR_NO_LOCK).
# C11+c (2026-09-30: an isotope child counts only at its label's exact spacing
# from the COMMITTED parent line and in band under its count-aware expectation;
# the isotopologues list answers the same question) moved every vector. Before it
# tv 21/15/107/16/143/15/10/37/1009, tof 6/15/182/15/258/82/95/138/2573, orbi
# 0/11/217/9/206/139/0/35/1090 (no lock 0/11/217/9/203/139/0/35/1093), ur
# 4/4/0/331/377/291/0/82/72 (pair table alone 4/4/0/331/376/291/0/82/73, neither
# 4/4/0/25/682/291/0/82/73); tests/fixtures/levels/README.md lists the rows.
GOLDEN = {
    "tv": (1373, "21/15/107/15/145/15/9/37/1009"),
    "tof": (3364, "6/15/182/14/267/85/84/135/2576"),
    "orbi": (1707, "0/10/217/9/205/138/0/35/1093"),
    # the uronium set (C17 + U, 2026-09-27): one source, no corroboration, levelled
    # with its neutral-pair table (rule U, row 9') and its lock table (C11+b);
    # without either it reads 4/4/0/17/687/293/0/82/74
    "ur": (1161, "4/4/0/330/375/293/0/82/73"),
}
UR_WITHOUT_PAIR = "4/4/0/17/687/293/0/82/74"
#: the Orbitrap set without its lock table, the uronium set with its neutral-pair
#: table alone -- the goldens before C11+b, and the base a leak guard levels at
ORBI_NO_LOCK = (1707, "0/10/217/9/202/138/0/35/1096")
UR_NO_LOCK = "4/4/0/330/374/293/0/82/74"

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


def level_of(rows, **kw) -> dict:
    """{(neutral, adduct): level} for the M0 rows of a synthetic ledger."""
    out = EV.compute_levels(ledger(rows), **kw)
    led = ledger(rows)
    m = led[led.role == "M0"].merge(out, on="peak_id")
    return dict(zip(zip(m.neutral_formula, m.adduct), m.evidence_level))


# one row per level: the passing fixture and the mutant that must move it
CASES = {
    "5b": (
        [m0("p", "C6H8O4", tied=True, anchor="a")],
        [m0("p", "C6H8O4", tied=False, anchor="a")],            # no longer hard -> 4b
    ),
    "2b": (
        [m0("p", "HNO3", method="known:atmospheric")],
        [m0("p", "C6H5NO3", method="known:nitroaromatic")],     # 3 structures -> 3a
    ),
    "3a": (
        [m0("p", "C8HF15O2", method="known:perfluoroacid")],
        [m0("p", "C8HF15O2", method="pass2")],                  # not curated -> 4c (unique, resolved, no axis)
    ),
    "3b": (
        [m0("p1", "C10H16O5"), m0("p2", "C10H16O5", adduct="[M+NO3]-")],
        [m0("p1", "C10H16O5"), m0("p2", "C10H16O5", adduct="[M+^NO3]-")],
        # ^ still deprotonated + clustered: the mutant is below, in test_level_3b
    ),
    "4c": (
        [m0("p", "C7H12O3", degeneracy=0.5, resolvability="resolved")],
        [m0("p", "C7H12O3", degeneracy=0.5, resolvability="blended")],   # -> 5a
    ),
    "5a": (
        [m0("p", "C7H12O4", degeneracy=2.0)],
        [m0("p", "C7H12O4", degeneracy=2.0, series_unit="CH2")],        # one axis -> 4b
    ),
    "4b": (
        [m0("p", "C9H14O4", series_unit="CH2")],
        [m0("p", "C9H14O4", adduct="[M+NO3]-", series_unit="CH2"),
         m0("q", "C9H14O4", adduct="[M+^NO3]-")],
        # ^ two axes (anchor + second cluster channel) but neither outside the chemistry -> still 4b
    ),
    "4a": (
        [m0("p", "C10H16O4", ion="C10H15O4", height=1000.0), child("c", "p", "13C+1", 107.0)],
        [m0("p", "C10H16O4", ion="C10H15O4", height=1000.0)],           # no iso axis -> 4b
    ),
    "4d": (
        [m0("p", "C8H14O2", adduct="[M+Br]-", ion="C8H14O2Br"),
         child("c", "p", "81Br+1", 950.0),
         m0("q", "C9H16O2", adduct="[M+Br]-", ion="C9H16O2Br")],
        [m0("p", "C8H14O2", adduct="[M+Br]-", ion="C8H14O2Br"),
         child("c", "p", "81Br+1", 950.0), child("c2", "p", "13C+1", 86.0),
         m0("q", "C9H16O2", adduct="[M+Br]-", ion="C9H16O2Br")],  # carbon pins the neutral -> 4b
    ),
}


# --------------------------------------------------------------------------- per level
def test_level_5b_hard_outranks_everything():
    ok, mut = CASES["5b"]
    assert level_of(ok)[("C6H8O4", "[M-H]-")] == "5b"
    assert level_of(mut)[("C6H8O4", "[M-H]-")] == "4b"
    assert level_of([m0("p", "C6H8O4", confidence="Low", anchor="a")])[("C6H8O4", "[M-H]-")] == "5b"
    assert level_of([m0("p", "C6H8O4", below=True, anchor="a")])[("C6H8O4", "[M-H]-")] == "5b"
    assert level_of([m0("p", "C6H8O4", degeneracy=5.0)])[("C6H8O4", "[M-H]-")] == "5b"


def test_level_2b_identity_on_one_structure():
    ok, mut = CASES["2b"]
    assert level_of(ok)[("HNO3", "[M-H]-")] == "2b"
    assert level_of(mut)[("C6H5NO3", "[M-H]-")] == "3a"          # compound scope, 3 isomers
    # class scope never reaches 2b even at one structure (TFA)
    assert level_of([m0("p", "C2HF3O2", method="known:perfluoroacid")])[("C2HF3O2", "[M-H]-")] == "3a"
    # a compound-scope formula absent from the isomer space is not 2b either
    assert level_of([m0("p", "C99H99O99", method="known:atmospheric")])[("C99H99O99", "[M-H]-")] == "3a"


def test_level_3a_named_class():
    ok, mut = CASES["3a"]
    assert level_of(ok)[("C8HF15O2", "[M-H]-")] == "3a"
    assert level_of(mut)[("C8HF15O2", "[M-H]-")] == "4c"


def test_level_3b_acid_branch():
    ok, _ = CASES["3b"]
    lv = level_of(ok)
    assert lv[("C10H16O5", "[M-H]-")] == "3b" and lv[("C10H16O5", "[M+NO3]-")] == "3b"
    # mutant: two CLUSTER channels are not a branch -> chan2 only -> 4b
    mut = [m0("p1", "C10H16O5", adduct="[M+NO3]-"),
           m0("p2", "C10H16O5", adduct="[M+^NO3]-")]
    assert level_of(mut)[("C10H16O5", "[M+NO3]-")] == "4b"


def test_level_4c_unopposed_on_a_separable_peak():
    ok, mut = CASES["4c"]
    assert level_of(ok)[("C7H12O3", "[M-H]-")] == "4c"
    assert level_of(mut)[("C7H12O3", "[M-H]-")] == "5a"
    # resolvability binds only where measured: no value at all keeps 4c
    assert level_of([m0("p", "C7H12O3", degeneracy=1.0, resolvability=None)])[("C7H12O3", "[M-H]-")] == "4c"


def test_level_5a_exact_mass_only():
    ok, mut = CASES["5a"]
    assert level_of(ok)[("C7H12O4", "[M-H]-")] == "5a"
    assert level_of(mut)[("C7H12O4", "[M-H]-")] == "4b"


def test_level_4b_one_corroboration():
    ok, two_inside = CASES["4b"]
    assert level_of(ok)[("C9H14O4", "[M-H]-")] == "4b"
    lv = level_of(two_inside)
    assert lv[("C9H14O4", "[M+NO3]-")] == "4b"                    # two axes, none outside the chemistry


def test_level_4a_two_axes_one_outside():
    ok, mut = CASES["4a"]
    other = {"C10H16O4"}
    assert level_of(ok, cross=other)[("C10H16O4", "[M-H]-")] == "4a"
    assert level_of(mut, cross=other)[("C10H16O4", "[M-H]-")] == "4b"      # one axis only
    assert level_of(ok)[("C10H16O4", "[M-H]-")] == "4b"                   # iso alone, nothing outside
    # multiline isotope evidence is itself an outside axis
    ml = [m0("p", "C10H16O4S", ion="C10H15O4S", height=1000.0, series_unit="CH2"),
          child("c", "p", "13C+1", 107.0), child("s", "p", "34S+2", 44.0)]
    assert level_of(ml)[("C10H16O4S", "[M-H]-")] == "4a"          # iso + anchor, and the two-line envelope is the outside axis


def test_level_4d_reagent_halogen_pins_the_ion_not_the_neutral():
    ok, mut = CASES["4d"]
    assert level_of(ok)[("C8H14O2", "[M+Br]-")] == "4d"
    assert level_of(mut)[("C8H14O2", "[M+Br]-")] == "4b"
    # a brominated neutral's own 81Br line on a nitrate channel (no reagent halogen) is ordinary isotope
    # evidence (C11+c: a Br-free ion makes no 81Br line at all -- its expectation is 0)
    nitrate = [m0("p", "C8H13BrO2", adduct="[M+NO3]-", ion="C8H13BrNO5-", mz=281.9983),
               child("c", "p", "81Br", 950.0, mz=281.9983 + 1.9979535),
               m0("q", "C9H16O2", adduct="[M+NO3]-")]
    assert level_of(nitrate)[("C8H13BrO2", "[M+NO3]-")] == "4b"
    brfree = [m0("p", "C8H14O2", adduct="[M+NO3]-", ion="C8H14NO5-", mz=204.0877),
              child("c", "p", "81Br", 950.0, mz=204.0877 + 1.9979535),
              m0("q", "C9H16O2", adduct="[M+NO3]-")]
    assert level_of(brfree)[("C8H14O2", "[M+NO3]-")] == "4c"


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
    is 0, never in band, so the pair has no isotope axis and no 4d) -- and C11+c
    released the 2026-09-27 hold: the flag means only "the reagent put the
    halogen on the ion"."""
    def facts(rows):
        out = EV.level_pooled({"f": ledger(rows)})
        return {(n, a): (lv, roi) for n, a, lv, roi in
                zip(out.neutral_formula, out.adduct, out.evidence_level, out.reagent_only_iso)}
    br = 1.9979535
    brfree = _bromide_channel(m0("p", "C8H14O4", ion="C8H13O4-", mz=173.0819),
                              child("c", "p", "81Br", 950.0, mz=173.0819 + br))
    assert facts(brfree)[("C8H14O4", "[M-H]-")] == ("4c", False)
    own = _bromide_channel(m0("p", "C7H11BrO4", ion="C7H10BrO4-", mz=236.9768),
                           child("c", "p", "81Br", 950.0, mz=236.9768 + br))
    assert facts(own)[("C7H11BrO4", "[M-H]-")] == ("4b", False)
    # the reagent's own line: a bromide adduct of a Br-free neutral, and of a brominated one (Br2 > Br)
    adduct = _bromide_channel(m0("p", "C8H14O2", adduct="[M+Br]-", ion="C8H14O2Br-", mz=221.0183),
                              child("c", "p", "81Br", 950.0, mz=221.0183 + br))
    assert facts(adduct)[("C8H14O2", "[M+Br]-")] == ("4d", True)
    # the Br2 ion is committed on its 79Br81Br line (the scorer's most abundant): its 81Br2 line sits
    # 1.998 Da above it at 0.486x (C11+c; against the mono line the same child at 0.95x of a 1.9456
    # expectation falls out of band)
    more = _bromide_channel(m0("p", "C7H11BrO4", adduct="[M+Br]-", ion="C7H11Br2O4-", mz=316.9030 + br),
                            child("c", "p", "81Br2", 486.0, mz=316.9030 + 2 * br))
    assert facts(more)[("C7H11BrO4", "[M+Br]-")] == ("4d", True)
    mono = _bromide_channel(m0("p", "C7H11BrO4", adduct="[M+Br]-", ion="C7H11Br2O4-", mz=316.9030),
                            child("c", "p", "81Br", 950.0, mz=316.9030 + br))
    assert facts(mono)[("C7H11BrO4", "[M+Br]-")] == ("4c", True)
    # a ledger row that stored the NEUTRAL as its ion formula: the adduct carries the reagent
    stored = _bromide_channel(m0("p", "C8H14O2", adduct="[M+Br]-", ion="C8H14O2", mz=221.0183),
                              child("c", "p", "81Br", 950.0, mz=221.0183 + br))
    assert facts(stored)[("C8H14O2", "[M+Br]-")] == ("4d", True)
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


def test_the_reference_script_reads_the_reagent_satellite_like_the_engine():
    import importlib.util
    spec = importlib.util.spec_from_file_location("level_ledger", Path(__file__).resolve().parents[1]
                                                  / "scripts" / "level_ledger.py")
    LL = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(LL)
    br = 1.9979535
    rows = ledger(_bromide_channel(
        m0("p", "C8H14O4", ion="C8H13O4-", mz=173.0819), child("c", "p", "81Br", 950.0, mz=173.0819 + br),
        m0("s", "C7H11BrO4", ion="C7H10BrO4-", mz=236.9768), child("d", "s", "81Br", 950.0, mz=236.9768 + br),
        m0("t", "C8H14O2", adduct="[M+Br]-", ion="C8H14O2", mz=221.0183),
        child("e", "t", "81Br", 950.0, mz=221.0183 + br),
        m0("u", "C7H11BrO4", adduct="[M+Br]-", ion="C7H11Br2O4-", mz=316.9030 + br),
        child("g", "u", "81Br2", 486.0, mz=316.9030 + 2 * br)))
    core = EV.level_pooled({"s1": rows})
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
def test_non_m0_rows_carry_no_level_and_reagent_ions_never_do():
    rows = [m0("p", "C6H8O4", ion="C6H7O4", height=1000.0), child("c", "p", "13C+1", 66.0),
            reagent("r", "Br", 78.918)]
    led = ledger(rows)
    EV.apply_levels(led)
    by = dict(zip(led.peak_id, led.evidence_level))
    assert by["p"] in ORDER and pd.isna(by["c"]) and pd.isna(by["r"])
    for col in ("evidence_level", "evidence_axes", "level_reason", "n_plausible_structures"):
        assert col in led.columns


def test_axes_string_and_reason_are_stable():
    rows = [m0("p", "C10H16O4", ion="C10H15O4", height=1000.0, series_unit="CH2"),
            child("c", "p", "13C+1", 107.0)]
    out = EV.compute_levels(ledger(rows), cross={"C10H16O4"})
    row = out[out.peak_id == "p"].iloc[0]
    assert row.evidence_axes.split("|")[:3] == ["iso", "anchor", "corroborated"]
    assert row.evidence_level == "4a" and row.level_reason.startswith("4a")


def test_missing_columns_and_nulls_do_not_raise():
    led = ledger([m0("p", "C6H8O4")]).drop(columns=["resolvability", "degeneracy_note", "isotopologues"])
    led["tied"] = pd.NA
    led["confidence"] = None
    out = EV.compute_levels(led)
    assert out.evidence_level.iloc[0] in ORDER


def test_isomer_space_rows_carry_a_rationale_and_cover_the_2b_formulas():
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


def _vector(levels: pd.Series) -> str:
    c = levels.value_counts()
    return "/".join(str(int(c.get(k, 0))) for k in ORDER)


def _iso(prefix: str) -> dict:
    """A golden set's rule H lock table as level_pooled reads it (C11+b)."""
    return IC.facts(pd.read_csv(FIXTURES / f"{prefix}_iso_checks.csv"))


@pytest.fixture(scope="module")
def expected():
    return pd.read_csv(FIXTURES / "expected_levels.csv")


def test_golden_tv_two_channels_corroborate_each_other():
    no3, br = _read("tv_nitrate"), _read("tv_bromide")
    cross_for_no3 = EV.source_neutrals({"tv_bromide": br})
    cross_for_br = EV.source_neutrals({"tv_nitrate": no3})
    a = EV.level_pooled({"tv_nitrate": no3}, cross=cross_for_no3)
    b = EV.level_pooled({"tv_bromide": br}, cross=cross_for_br)
    both = pd.concat([a, b])
    assert (len(both), _vector(both.evidence_level)) == GOLDEN["tv"]


def test_golden_same_air_pair():
    tof, orbi = _pooled("tof"), _pooled("orbi")
    n_tof, n_orbi = EV.source_neutrals(tof), EV.source_neutrals(orbi)
    t = EV.level_pooled(tof, cross=n_orbi)
    o = EV.level_pooled(orbi, cross=n_tof, iso=_iso("orbi"))
    assert (len(t), _vector(t.evidence_level)) == GOLDEN["tof"]
    assert (len(o), _vector(o.evidence_level)) == GOLDEN["orbi"]
    no_lock = EV.level_pooled(orbi, cross=n_tof)
    assert (len(no_lock), _vector(no_lock.evidence_level)) == ORBI_NO_LOCK


def _ur_pairs() -> set:
    t = pd.read_csv(FIXTURES / "ur_neutral_pairs.csv")
    return set(t.loc[t["upair"].astype(bool), "neutral_formula"])


def test_golden_uronium_neutral_pair():
    """Rule U on the uronium set: the pair table lifts 306 ion pairs to 4a;
    without it the vector is C17's alone. The lock table (C11+b) lifts one lead."""
    ur = _pooled("ur")
    with_pair = EV.level_pooled(ur, upair=_ur_pairs(), iso=_iso("ur"))
    assert (len(with_pair), _vector(with_pair.evidence_level)) == GOLDEN["ur"]
    assert _vector(EV.level_pooled(ur, upair=_ur_pairs()).evidence_level) == UR_NO_LOCK
    assert _vector(EV.level_pooled(ur).evidence_level) == UR_WITHOUT_PAIR


def test_uronium_rows_match_the_reference_script(expected):
    got = EV.level_pooled(_pooled("ur"), upair=_ur_pairs(), iso=_iso("ur")).assign(source="ur")
    exp = expected[expected.source == "ur"]
    assert len(exp) == GOLDEN["ur"][0]
    m = exp.merge(got, left_on=["source", "neutral", "adduct"],
                  right_on=["source", "neutral_formula", "adduct"], how="left")
    bad = m[m.level != m.evidence_level]
    assert bad.empty, bad[["neutral", "adduct", "level", "evidence_level"]].head(20)


def test_rows_match_the_reference_script_row_for_row(expected):
    tof, orbi = _pooled("tof"), _pooled("orbi")
    n_tof, n_orbi = EV.source_neutrals(tof), EV.source_neutrals(orbi)
    got = pd.concat([EV.level_pooled(tof, cross=n_orbi).assign(source="tof"),
                     EV.level_pooled(orbi, cross=n_tof, iso=_iso("orbi")).assign(source="orbi")])
    exp = expected[expected.source.isin(["tof", "orbi"])]
    m = exp.merge(got, left_on=["source", "neutral", "adduct"],
                  right_on=["source", "neutral_formula", "adduct"], how="left")
    assert m.evidence_level.notna().all(), "every reference row must be levelled"
    bad = m[m.level != m.evidence_level]
    assert bad.empty, bad[["source", "neutral", "adduct", "level", "evidence_level"]].head(20)


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


def test_every_fixture_row_matches_the_reference_script_fact_for_fact(expected):
    """Not only the level: every fact the reference script recorded for a fixture row (iso, reagent_only_iso,
    chan2, ... -- C11+c moved several at unchanged levels) is the engine's, on all five sources."""
    tof, orbi = _pooled("tof"), _pooled("orbi")
    no3, br = _read("tv_nitrate"), _read("tv_bromide")
    got = pd.concat([
        EV.level_pooled(tof, cross=EV.source_neutrals(orbi)).assign(source="tof"),
        EV.level_pooled(orbi, cross=EV.source_neutrals(tof), iso=_iso("orbi")).assign(source="orbi"),
        EV.level_pooled(_pooled("ur"), upair=_ur_pairs(), iso=_iso("ur")).assign(source="ur"),
        EV.level_pooled({"tv_nitrate": no3}, cross=EV.source_neutrals({"tv_bromide": br})).assign(source="tv_nitrate"),
        EV.level_pooled({"tv_bromide": br}, cross=EV.source_neutrals({"tv_nitrate": no3})).assign(source="tv_bromide")])
    m = expected.merge(got, left_on=["source", "neutral", "adduct"],
                       right_on=["source", "neutral_formula", "adduct"], how="left", suffixes=("", "_engine"))
    assert len(m) == len(expected) and m.evidence_level.notna().all(), "every reference row must be levelled"
    assert (m.level == m.evidence_level).all()
    for c in FIXTURE_FACTS:
        a, b = m[c].map(_cell), m[c + "_engine"].map(_cell)
        bad = m.loc[a != b, ["source", "neutral", "adduct", c, c + "_engine"]]
        assert bad.empty, (c, bad.head(10).to_dict("records"))


def test_pooled_equals_the_script_on_the_same_files():
    """level_pooled over N files is the reference pooling: all rows of a pair across
    files decide tied/lowconf, any row decides below, chan2 sees every file."""
    a = [m0("p", "C9H14O4", tied=True)]
    b = [m0("q", "C9H14O4", tied=False, adduct="[M+NO3]-")]
    out = EV.level_pooled({"f1": ledger(a), "f2": ledger(b)})
    lv = dict(zip(zip(out.neutral_formula, out.adduct), out.evidence_level))
    assert lv[("C9H14O4", "[M-H]-")] == "5b"       # that pair's only row is tied: hard
    assert lv[("C9H14O4", "[M+NO3]-")] == "3b"     # the neutral branches ACROSS files: bare in f1, clustered in f2


def test_a_satellite_hangs_off_its_parent_in_the_same_file():
    """Two files may reuse a peak id: a child joins the M0 with that id in ITS
    file only (spec section 2), so file 1's neutral gains no isotope axis from
    file 2's satellite."""
    f1 = ledger([m0("p", "C9H14O4", ion="C9H13O4", height=1000.0)])
    f2 = ledger([m0("p", "C10H16O4", ion="C10H15O4", height=1000.0), child("c", "p", "13C+1", 107.0)])
    out = EV.level_pooled({"f1": f1, "f2": f2})
    lv = dict(zip(zip(out.neutral_formula, out.adduct), out.evidence_level))
    axes = dict(zip(zip(out.neutral_formula, out.adduct), out.evidence_axes))
    assert lv[("C10H16O4", "[M-H]-")] == "4b" and axes[("C10H16O4", "[M-H]-")].startswith("iso")
    assert lv[("C9H14O4", "[M-H]-")] == "4c" and "iso" not in axes[("C9H14O4", "[M-H]-")]


def test_pooled_pairs_are_m0_only_a_reagent_row_forms_no_pair():
    """The pooled batch table (tables/evidence_levels.csv) has one row per
    committed (neutral, adduct): a reagent ion is excluded by role there too,
    not only on the stamp."""
    out = EV.level_pooled({"f": ledger([m0("p", "C6H8O4"), reagent("r", "Br", 78.918)])})
    assert list(out.neutral_formula) == ["C6H8O4"]


# --------------------------------------------------------------------------- the cross set (C8)
def test_a_source_corroborates_only_what_it_holds_at_4b_or_better_by_its_own_evidence():
    """The `corroborated` axis is a second SIGHTING of the neutral: the source
    must pin it by an axis of its own (4b) or better. A 4c (unopposed, no axis),
    a 5b (tied) or an ion-only pair is the source's grid enumerating the formula,
    not a sighting of it."""
    src = ledger([
        m0("a", "C10H16O4", ion="C10H15O4", height=1000.0), child("a1", "a", "13C+1", 107.0),   # iso -> 4b
        m0("b", "C8HF15O2", method="known:perfluoroacid"),                                     # 3a
        m0("c", "C7H12O4"),                                                                    # 4c: no axis
        m0("d", "C6H8O4", tied=True, anchor="a"),                                              # 5b: tied
        m0("e", "C9H14O4", adduct="[M]-.", method="ion_only:electron_attachment"),  # ion-only
    ])
    assert EV.source_neutrals({"s": src}) == {"C10H16O4", "C8HF15O2"}
    assert EV.source_neutrals({"s": src}, max_level="4c") == {"C10H16O4", "C8HF15O2", "C7H12O4"}


def test_two_sources_that_only_agree_cannot_lift_each_other():
    """Two instruments whose grids both fit a mass-degenerate formula with no axis
    (5b on each) used to hand each other the `corroborated` axis and climb to 4b
    together; each is now corroborated only by what the other pins on its own."""
    x = ledger([m0("p", "C6H10O4", degeneracy=5.0)])
    y = ledger([m0("q", "C6H10O4", degeneracy=5.0, adduct="[M+NO3]-")])
    for me, other in ((x, y), (y, x)):
        cross = EV.source_neutrals({"other": other})
        assert cross == set()
        assert set(EV.level_pooled({"me": me}, cross=cross).evidence_level) == {"5b"}
        # the mutant: any-level membership lifts the pair to 4b on the agreement alone
        assert set(EV.level_pooled({"me": me}, cross={"C6H10O4"}).evidence_level) == {"4b"}


def test_a_per_file_source_is_relevelled_not_read():
    """A ledger with predicate columns is levelled afresh with no cross set: a
    stored 4a the source owed to ITS OWN --corroborate does not count."""
    src = ledger([m0("p", "C9H14O4", degeneracy=None)])          # no axis, degeneracy unmeasured -> 5a
    src["evidence_level"], src["evidence_axes"] = "4a", "iso|corroborated"
    assert EV.source_neutrals({"s": src}) == set()


def test_a_merged_ledger_source_reads_its_stored_level_without_its_own_corroboration():
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
    assert EV.source_neutrals({"m": merged}) == {"A1", "C1", "G1"}
    with pytest.raises(ValueError, match="evidence_level"):
        EV.source_neutrals({"m": merged.drop(columns=["evidence_level", "evidence_axes"])})


def test_the_cross_set_equals_the_reference_scripts_on_the_golden_sets():
    """The in-core helper and scripts/level_ledger.py pick the same neutrals."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "level_ledger", Path(__file__).resolve().parents[1] / "scripts" / "level_ledger.py")
    LL = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(LL)
    for prefix in ("tof", "orbi"):
        files = _pooled(prefix)
        frame = pd.concat([f.assign(__file=k) for k, f in files.items()], ignore_index=True)
        halogen = LL.detect_reagent_halogen(frame[frame.role == "M0"])
        measured = LL.measure_source(prefix, frame, halogen)
        measured["reagent_halogen"] = halogen or ""
        assert LL.own_good_neutrals(measured) == EV.source_neutrals(files), prefix
