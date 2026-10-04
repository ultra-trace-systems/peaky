"""C14: the ionization-plausibility demote covers N-only neutrals (definition D4).

`cleanup.demote_implausible_ionization` demotes an M0 whose neutral cannot ionize on an anion channel that needs an
acidic proton or an H-bond site: a pure hydrocarbon (unchanged), and now an N-only neutral -- C >= 1, N >= 1, no
S / P / halogen / Si, and O = 0, or O = 1 with DBE <= 1, or O = 2 with DBE <= 0 (`cleanup.anion_implausible`).
"""
import pandas as pd
import pytest

from peaky.assignment import cleanup as CU
from peaky.assignment import evidence as EV


def _row(formula, adduct="[M-H]-", tier="Assigned", role="M0", **kw):
    return dict(role=role, peak_id=f"p{formula}{adduct}", neutral_formula=formula, adduct=adduct, tier=tier,
                commentary="", below_assignability=False, **kw)


def _demote(rows):
    led = pd.DataFrame(rows)
    logs = []
    out = CU.demote_implausible_ionization(led, log=logs.append)
    return led, out, logs


# --- the rule as a formula predicate -------------------------------------------------------------------------------
@pytest.mark.parametrize("formula, why", [
    ("C7H10", "hydrocarbon"),                 # unchanged leg
    ("C21H46N2", "N-only"),                   # H-saturated N2, the shift decoy's survivor (DBE 0)
    ("C31H44N2", "N-only"),                   # O = 0 at any DBE (DBE 11)
    ("C13H12N2", "N-only"),                   # aryl N-H family: the census cost, hit on purpose
    ("C22H46N2O", "N-only"),                  # O = 1, DBE 1 (the leg that admits amides)
    ("C21H46N2O", "N-only"),                  # O = 1, DBE 0
    ("C25H54N2O2", "N-only"),                 # O = 2, DBE 0
    ("C5H13NO2", "N-only"),                   # an amino-diol, DBE 0
    ("C4H9^N", "N-only"),                     # a labelled N is N, not a hydrocarbon
    ("C3H7^NO", "N-only"),                    # O = 1, DBE 1 only once ^N counts as N
    ("CH5N", "N-only"),                       # one carbon is enough (the C >= 1 gate, from above)
    ("CH4", "hydrocarbon"),                   # ... on the hydrocarbon leg too
])
def test_anion_implausible_hits(formula, why):
    assert CU.anion_implausible(formula) == why


@pytest.mark.parametrize("formula", [
    "C8H16N2O",        # O = 1, DBE 2: a C=O plus a ring or a second C=O fits -> not judged
    "C3H7NO2",         # O = 2, DBE 1: an amino acid fits (a carboxylic acid)
    "C4H11NO3",        # O = 3: out of the rule
    "C6H12O6",         # no N, has O
    "C4H10O",          # O = 1, DBE 0 but no N: an alcohol is not an N-only neutral (the N >= 1 gate)
    "C2H6O2",          # O = 2, DBE 0, no N: a diol
    "C3H6O",           # O = 1, DBE 1, no N: a ketone
    "C5H12O2",         # O = 2, DBE 0, no N (a row the labelled-nitrate run commits)
    "C2H5Br",          # a halocarbon is not a hydrocarbon: the exemption comes before either leg
    "CH2Br2",
    "C2Cl4",
    "C6F14",
    "C10H18Cl4",       # (a row the labelled-nitrate run commits)
    "C2H6S",           # a thio-hydrocarbon
    "C11H6Cl2N2",      # halogen: exempt (fenpiclonil's formula)
    "C8H20N2Si",       # Si: exempt
    "C2H7NS",          # S: exempt
    "C3H10NP",         # P: exempt
    "C5H3F8NO",        # F: exempt
    "C5H9BrN2",        # Br: exempt (a bromo-amine reading in a bromide run)
    "ICN",             # I: exempt -- the curated reactive-iodine compound, one C, one N, no O
    "HNO3",            # no carbon
    "",                # nothing to judge
    None,
    float("nan"),
])
def test_anion_implausible_spares(formula):
    assert CU.anion_implausible(formula) is None


# --- the demote --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("adduct", ["[M-H]-", "[M+NO3]-", "[M+^NO3]-", "[M+Br]-", "[M+HBr+Br]-", "[M+CO3]-"])
def test_n_only_on_every_anion_channel_is_demoted(adduct):
    led, out, _ = _demote([_row("C21H46N2", adduct)])
    assert led.loc[0, "tier"] == "Candidate"
    assert bool(led.loc[0, "below_assignability"]) is True
    assert out == {"ionization_demoted": 1, "ionization_demoted_n_only": 1}


@pytest.mark.parametrize("adduct", ["[M]-.", "[M]-", "[M+O2]-", "[M+H]+", "[M+NH4]+", "[M+(CH4N2O)H]+"])
def test_electron_attachment_and_positive_channels_are_left_alone(adduct):
    led, out, _ = _demote([_row("C21H46N2", adduct)])
    assert led.loc[0, "tier"] == "Assigned"
    assert bool(led.loc[0, "below_assignability"]) is False
    assert led.loc[0, "commentary"] == ""
    assert out == {"ionization_demoted": 0, "ionization_demoted_n_only": 0}


def test_boundaries_keep_their_tier():
    led, out, _ = _demote([_row("C8H16N2O"), _row("C3H7NO2", "[M+NO3]-"), _row("C11H6Cl2N2"),
                           _row("C8H20N2Si", "[M+Br]-")])
    assert (led["tier"] == "Assigned").all()
    assert not led["below_assignability"].astype(bool).any()
    assert out["ionization_demoted"] == 0


def test_candidate_stays_candidate_and_gains_the_flag():
    led, out, _ = _demote([_row("C22H46N2O", "[M+^NO3]-", tier="Candidate")])
    assert led.loc[0, "tier"] == "Candidate"
    assert bool(led.loc[0, "below_assignability"]) is True
    assert out["ionization_demoted_n_only"] == 1


def test_only_m0_rows_are_judged():
    rows = [_row("C21H46N2", role="iso_child"), _row("C21H46N2", role="reagent"), _row("C21H46N2")]
    led, out, _ = _demote(rows)
    assert list(led["tier"]) == ["Assigned", "Assigned", "Candidate"]
    assert out["ionization_demoted"] == 1


def test_the_note_names_the_n_only_reason_and_the_hydrocarbon_note_is_unchanged():
    led, out, logs = _demote([_row("C25H54N2O2", "[M+NO3]-"), _row("C7H10"), _row("C4H9^N")])
    n_only = led.loc[0, "commentary"]
    assert n_only.startswith("N-only neutral via [M+NO3]- (O 2, DBE 0)")
    assert "no carboxylic acid or phenol fits the formula" in n_only
    assert led.loc[1, "commentary"] == ("pure hydrocarbon via [M-H]-: no acidic proton / H-bond site to "
                                        "ionize -- implausible (mass coincidence)")
    # the labelled amine is judged as an N-compound, not as a hydrocarbon
    assert led.loc[2, "commentary"].startswith("N-only neutral via [M-H]- (O 0, DBE 1)")
    assert out == {"ionization_demoted": 3, "ionization_demoted_n_only": 2}
    assert logs == ["[cleanup] demoted 3 implausible-ionization M0 via an anion channel (1 heteroatom-free, 2 N-only)"]


def test_a_prior_note_is_kept():
    led = pd.DataFrame([_row("C21H46N2")])
    led.loc[0, "commentary"] = "series gap-fill"
    CU.demote_implausible_ionization(led, log=lambda *a: None)
    assert led.loc[0, "commentary"].startswith("series gap-fill; N-only neutral via [M-H]-")


def test_a_ledger_without_the_optional_columns_does_not_raise():
    led = pd.DataFrame([dict(role="M0", neutral_formula="C21H46N2", adduct="[M-H]-", tier="Assigned")])
    out = CU.demote_implausible_ionization(led, log=lambda *a: None)
    assert led.loc[0, "tier"] == "Candidate"
    assert out["ionization_demoted"] == 1


def test_a_demoted_n_only_pair_is_below_assignability():
    """The levels read the flag as a hard input: the pair argues with itself,
    whatever its axes -- the merge vote's class drops from 2 (both channels: the
    acid branch) to 0, and the evidence scale rejects it at step 0 by its setter
    ("implausible ionization", levels.decide.BELOW_SETTERS)."""
    from peaky.assignment.levels import decide as DC
    rows = [dict(_row("C21H46N2", "[M-H]-"), mz=323.3432, height=5e4, confidence="High"),
            dict(_row("C21H46N2", "[M+^NO3]-"), mz=387.3357, height=3e4, confidence="High")]
    led = pd.DataFrame(rows)
    before = EV._level_pairs({"": led}, per_file=True)
    assert before["branch"].all() and not before["below"].any() and set(EV.vote_classes(led)) == {2}
    CU.demote_implausible_ionization(led, log=lambda *a: None)
    after = EV._level_pairs({"": led}, per_file=True)
    assert after["below"].all() and set(EV.vote_classes(led)) == {0}
    below = DC.below_classes({"f": led})
    assert {k: v["setters"] for k, v in below.items()} == {
        ("C21H46N2", "[M-H]-"): ["implausible ionization"], ("C21H46N2", "[M+^NO3]-"): ["implausible ionization"]}
