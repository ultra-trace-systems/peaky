"""The degeneracy audit (degeneracy.py) counts the ions THIS run could have
committed inside the calibrated window: its channels, its element space (the
context's budget, a zero cap raised by a family the file opened, the curated
formulas), no box. A commit outside that space gives a lower bound -- enough to
decide "degenerate", never "unique" -- and below three ions the density is not
measured (NaN). Offline: no network, no grid build."""
import dataclasses
import random
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from peaky.assignment import assign as A
from peaky.assignment import degeneracy as D
from peaky.assignment import evidence as EV
from peaky.assignment import ledger as L
from peaky.assignment import passes
from peaky.assignment import plausibility as PL
from peaky.assignment import tiers as T
from peaky.chem import chemistry as C
from peaky.chem import contexts as X

URONIUM = ("[M+H]+", "[M+(CH4N2O)H]+")
NITRATE = ("[M+NO3]-", "[M-H]-", "[M+^NO3]-")


def _ledger(rows):
    """rows: (peak_id, neutral, adduct); each peak sits at its reading's exact m/z."""
    mzs = [C.neutral_mass(f) + C.ADDUCT_SHIFTS[a] for _, f, a in rows]
    led = L.new_ledger(pd.DataFrame({"peak_id": [p for p, _, _ in rows], "mz": mzs,
                                     "height": [1e4] * len(rows)}))
    for p, f, a in rows:
        L.commit_assignment(led, p, neutral_formula=f, adduct=a, ion_formula=f"{f}{a[-1]}",
                            ion_score=0.9, compound_score=0.9, ppm_error=0.0, pass_no=1,
                            method="cheminfo+grid", confidence="Good", commentary="x")
    return led


def _measure(rows, *, sigma=0.2, **kw):
    return D.measure_degeneracy(_ledger(rows), cal=(0.0, sigma), **kw)


def _competitor_adducts(res):
    return {a.split(" (")[0].split(" ", 1)[1] for a in res["alts"]}


# --------------------------------------------------------------------------- the solver
def test_solver_proposes_exactly_the_grid_enumerators_formulas():
    """The analytic per-window solve is the grid enumerator's formula set (<= 3 heteroatom types),
    window by window: 1.9-Da windows tiling 20-400 Da, so every formula of the space is compared --
    P is in the space so Senior's cap is not implied by H >= 0 -- plus exact-hit windows."""
    prof = dataclasses.replace(X.get_context("ambient-air"), max_N=1, max_S=1, max_P=1, max_Si=0, max_Cl=1,
                               max_Br=0, max_F=0, grid_c_max=12, grid_o_max=6)
    cb = D._combos(D._caps(prof))
    grid = [(m, f) for m, f in C.enumerate_grid({"C": (0, 12), "H": (0, 60), "N": (0, 1), "O": (0, 6),
                                                 "S": (0, 1), "P": (0, 1), "Cl": (0, 1)}, 20.0, 400.0)
            if D._het_types(C.parse_formula(f)) <= D.MAX_HET_TYPES]
    rng = random.Random(7)
    # 1.9-Da tiles: just under the solver's 2 x m(H) limit, wide enough that Senior's cap -- which the
    # carbon pruning enforces on its own inside a narrower window -- has to decide some formulas
    windows = [(20.0 + 1.9 * k, 21.9 + 1.9 * k) for k in range(200)]
    windows += [(m - 1e-7, m + 1e-7) for m, _ in rng.sample(grid, 20)]       # exact hits
    n = 0
    for lo, hi in windows:
        want = {f for m, f in grid if lo <= m <= hi}
        rep, cv, hv = D._solve(cb, 12, lo, hi)
        got = {D._formula(cb, i, c, h) for i, c, h in zip(rep, cv, hv)}
        assert got == want, (lo, hi, sorted(got ^ want))
        n += len(want)
    assert n > 10_000


def test_the_window_is_applied_exactly():
    """Inclusive at the edge (to float noise), and not a nanodalton wider -- the DBE solve alone
    carries a tolerance of that size, so the mass test has to be applied on its own."""
    cb = D._combos(D._caps(X.get_context("ambient-air")))
    m = C.neutral_mass("C10H16O5")
    def got(lo, hi):
        rep, cv, hv = D._solve(cb, 40, lo, hi)
        return {D._formula(cb, i, c, h) for i, c, h in zip(rep, cv, hv)}
    assert "C10H16O5" in got(m - 1e-3, m + 1e-12) and "C10H16O5" in got(m - 1e-12, m + 1e-3)
    assert "C10H16O5" not in got(m + 1e-9, m + 1e-3)
    assert "C10H16O5" not in got(m - 1e-3, m - 1e-9)


@pytest.mark.parametrize("name", sorted({id(p): n for n, p in X.CONTEXTS.items()}.values()))
def test_every_context_enumerates_inside_the_ceiling(name):
    """A context that leaves a cap at its 99 default must not open a 99-atom loop."""
    prof = X.get_context(name)
    for p in D.space_profiles(prof, tuple(X.CONTAMINANT_FAMILIES)):
        caps = D._caps(p)
        assert all(caps[e] <= D.ELEMENT_CEILING[e] for e in D.ELEMENT_CEILING)
        assert len(D._combos(caps)["A"]) < 1_000_000


# --------------------------------------------------------------------------- channels
def test_a_positive_channel_is_audited_on_its_own_adducts():
    """Uronium: every competitor is a cation reading (the old fixed Br-CIMS set made
    the channel enumerate only anions, and 82 % of its rows read "unique" at 0). The
    urea channel adds no ion of its own there: X [M+(CH4N2O)H]+ is X+CH4N2O [M+H]+."""
    res = _measure([("P", "C10H16O5", "[M+H]+")], sigma=3.0, context="uronium", adducts=URONIUM)["P"]
    assert res["measured"] and not res["lower_bound"]
    assert res["density"] >= 2
    assert all(a.endswith("+") for a in _competitor_adducts(res))


def test_every_channel_of_the_run_counts():
    """A labelled-nitrate row meets its competitors on the run's other channels too."""
    row = [("P", "C10H16O7", "[M+^NO3]-")]
    full = _measure(row, sigma=3.0, context="ambient-air", adducts=NITRATE)["P"]
    own = _measure(row, sigma=3.0, context="ambient-air", adducts=("[M+^NO3]-",))["P"]
    assert "[M-H]-" in _competitor_adducts(full)
    assert "[M-H]-" not in _competitor_adducts(own)
    assert own["density"] < full["density"]


def test_a_row_on_another_adduct_adds_its_own():
    """A commit on an adduct outside the run's channels is still found on its own."""
    res = _measure([("P", "C10H16O5", "[M-H]-")], context="ambient-air", adducts=("[M+NO3]-",))["P"]
    assert res["measured"] and not res["lower_bound"] and res["density"] >= 1


def test_same_ion_readings_collapse():
    assert D._canonical_ion("C2H3BrO2", "[M+Br]-") == D._canonical_ion("C2H2O2", "[M+HBr+Br]-")
    assert D._canonical_ion("CH2O2", "[M-H+I2]-") == D._canonical_ion("CHIO2", "[M+I]-")
    assert D._canonical_ion("C8H18O5", "[M+NH4]+") == D._canonical_ion("C8H21NO5", "[M+H]+")


# --------------------------------------------------------------------------- the element space
def test_space_profiles_raise_only_a_zero_cap():
    amb = X.get_context("ambient-air")
    profs = D.space_profiles(amb, ("fluorinated", "siloxane", "organosulfate"))
    assert len(profs) == 2 and profs[1].max_F == 17 and profs[1].max_Si == amb.max_Si
    assert D.space_profiles(amb, ("siloxane",)) == [amb]                 # Si 1 binds on ambient
    ur = D.space_profiles("uronium", ("fluorinated",))
    assert len(ur) == 2 and ur[1].max_F == 17


def test_opened_families_reads_declared_reagent_evidence_and_commits():
    ev = pd.DataFrame({"action": ["fluorinated", "glycol_peg", None], "significant": [True, False, True]})
    led = _ledger([("P", "C10H16O5", "[M-H]-")])
    led.loc[led.peak_id == "P", "method"] = "contaminant:halogen_dbp"
    fams = D.opened_families(X.get_context("ambient-air"), "Br", ev, led)
    assert fams == ("organosulfate", "nitrate", "siloxane", "amine", "bromo_organic", "fluorinated",
                    "halogen_dbp")
    assert D.opened_families(None, None, None, None) == ()


def test_fluorine_counts_only_where_the_file_opened_it():
    pfoa = [("P", "C8HF15O2", "[M-H]-")]
    closed = _measure(pfoa, context="ambient-air", adducts=NITRATE)["P"]
    assert closed["lower_bound"]                                   # F is off the ambient budget
    opened = _measure(pfoa, context="ambient-air", adducts=NITRATE, families=("fluorinated",))["P"]
    assert opened["measured"] and not opened["lower_bound"] and opened["density"] >= 1
    curated = _measure(pfoa, context="ambient-air", adducts=NITRATE, curated={"C8HF15O2"})["P"]
    assert curated["measured"] and not curated["lower_bound"]      # a curated list names it


def test_every_competitor_passes_the_context_filter():
    """The space is the context's own: its Van Krevelen and minimum-carbon rules bind competitors."""
    res = _measure([("P", "C10H16O5", "[M-H]-")], sigma=15.0, context="ambient-air", adducts=NITRATE,
                   max_alts=500)["P"]
    assert res["density"] > 10
    for alt in res["alts"]:
        assert X.filter_by_profile(alt.split(" ")[0], X.get_context("ambient-air"))[0], alt


def test_the_commit_outside_the_context_filter_is_outside_the_space():
    """In the budget but outside the context's filter ((H+X)/C 0.5 < 0.7): a bound, not a count."""
    res = _measure([("P", "C8H4O4", "[M-H]-")], sigma=0.05, context="ambient-air", adducts=NITRATE)["P"]
    assert res["lower_bound"] and np.isnan(res["density"]) and "context filter" in res["note"]


def test_a_curated_commit_off_the_window_is_in_the_space():
    """Off the window is not off the space when a curated list names the formula."""
    led = _ledger([("P", "C8HF15O2", "[M-H]-")])
    led.loc[led.peak_id == "P", "mz"] = float(led.loc[led.peak_id == "P", "mz"].iloc[0]) * (1 + 5e-6)
    res = D.measure_degeneracy(led, cal=(0.0, 0.2), context="ambient-air", adducts=NITRATE,
                               curated={"C8HF15O2"})["P"]
    assert not res["lower_bound"] and res["measured"]


def test_uronium_never_counts_a_halogen_competitor():
    res = _measure([("P", "C12H22O6", "[M+H]+")], sigma=4.0, context="uronium", adducts=URONIUM)["P"]
    assert res["density"] >= 2
    for alt in res["alts"]:
        cnt = C.parse_formula(alt.split(" ")[0])
        assert not any(cnt.get(e, 0) for e in ("F", "Cl", "Br", "I")), alt


# --------------------------------------------------------------------------- no box
def test_a_heavy_commit_is_inside_its_own_count():
    """C22H42O6 (outside the old C<=20 box) is measured -- the old audit left it out of
    its own count and density 0 read as 'unique'."""
    res = _measure([("P", "C22H42O6", "[M+(CH4N2O)H]+")], context="uronium", adducts=URONIUM)["P"]
    assert res["measured"] and not res["lower_bound"] and res["density"] >= 1
    hom = _measure([("P", "C10H16O13", "[M+NO3]-")], context="ambient-air", adducts=NITRATE)["P"]
    assert hom["measured"] and not hom["lower_bound"]              # O13 > the old O<=12


# --------------------------------------------------------------------------- outside the space
def test_an_off_space_commit_is_a_lower_bound_or_not_measured():
    row = [("P", "C10H15O4P", "[M-H]-")]                           # P: off the ambient budget
    tight = _measure(row, sigma=0.05, context="ambient-air", adducts=NITRATE)["P"]
    assert not tight["measured"] and np.isnan(tight["density"])
    assert tight["note"].startswith("not measured")
    # the note must not read as saturated / degenerate anywhere downstream
    assert not PL._is_saturated(tight["note"])
    assert T._degeneracy({"degeneracy_density": pd.NA, "degeneracy_note": tight["note"]}) == (None, False)
    wide = _measure(row, sigma=12.0, context="ambient-air", adducts=NITRATE)["P"]
    assert wide["measured"] and wide["lower_bound"] and wide["density"] >= 3
    assert "at least" in wide["note"] and "lower bound" in wide["note"]
    assert T._degeneracy({"degeneracy_density": wide["density"], "degeneracy_note": wide["note"]})[1]


def test_the_lower_bound_decides_at_three():
    """Off-space commit with one other ion: not measured; with two: 'at least 3', degenerate."""
    row = [("P", "C10H15O4P", "[M-H]-")]
    one = _measure(row, sigma=0.2, context="ambient-air", adducts=NITRATE)["P"]
    assert np.isnan(one["density"]) and "; 1 other plausible ion(s)" in one["note"]
    two = _measure(row, sigma=0.3, context="ambient-air", adducts=NITRATE)["P"]
    assert two["density"] == 3 and two["lower_bound"] and "at least 3" in two["note"]
    assert T._degeneracy({"degeneracy_density": two["density"], "degeneracy_note": two["note"]})[1]


def test_a_curated_formula_competes_off_budget():
    """A curated formula counts wherever its ion lands in the window, off-budget or not."""
    row = [("P", "C10H16O5", "[M-H]-")]
    cur = _measure(row, sigma=15.0, context="ambient-air", adducts=NITRATE, curated={"C10H17O3P"},
                   max_alts=500)["P"]
    plain = _measure(row, sigma=15.0, context="ambient-air", adducts=NITRATE, max_alts=500)["P"]
    assert any(a.startswith("C10H17O3P ") for a in cur["alts"])
    assert not any(a.startswith("C10H17O3P ") for a in plain["alts"])
    assert cur["density"] == plain["density"] + 1


def test_outside_the_window_is_not_outside_the_space():
    """An in-space commit off the window counts only what the window holds."""
    led = _ledger([("P", "C10H16O5", "[M-H]-")])
    led.loc[led.peak_id == "P", "mz"] = float(led.loc[led.peak_id == "P", "mz"].iloc[0]) * (1 + 5e-6)
    res = D.measure_degeneracy(led, cal=(0.0, 0.2), context="ambient-air", adducts=NITRATE)["P"]
    assert res["measured"] and not res["lower_bound"]


def test_apply_stamps_na_and_the_level_reads_not_measured():
    led = _ledger([("P", "C10H15O4P", "[M-H]-"), ("Q", "C10H16O5", "[M-H]-")])
    D.apply_degeneracy(led, cal=(0.0, 0.05), context="ambient-air", adducts=NITRATE)
    p = led.loc[led.peak_id == "P"].iloc[0]
    assert pd.isna(p["degeneracy_density"]) and str(p["degeneracy_note"]).startswith("not measured")
    q = led.loc[led.peak_id == "Q"].iloc[0]
    assert float(q["degeneracy_density"]) >= 1
    facts = EV._measure(led.assign(**{"__file": "f"}), halogen=None)
    assert np.isnan(float(facts.loc[facts.neutral_formula == "C10H15O4P", "degeneracy"].iloc[0]))


def test_uncalibrated_is_skipped():
    assert D.measure_degeneracy(_ledger([("P", "C10H16O5", "[M-H]-")]), cal=None) == {}


# --------------------------------------------------------------------------- the stage
def test_the_stage_audits_the_runs_channels_and_space(monkeypatch):
    seen = {}

    def fake(led, **kw):
        seen.update(kw)
        return led

    monkeypatch.setattr(D, "apply_degeneracy", fake)
    ev = pd.DataFrame({"action": ["fluorinated"], "significant": [True]})
    st = SimpleNamespace(profile=X.get_context("uronium"), cfg=SimpleNamespace(reflist_formulas=frozenset({"C9H9Q"})),
                         series_carry={"evidence": ev}, do_pass3=True, reagent=None,
                         led=_ledger([("P", "C10H16O5", "[M+H]+")]), adducts=list(URONIUM),
                         log=lambda *a: None)
    out = A._stage_degeneracy(st)
    assert seen["adducts"] == list(URONIUM) and seen["context"] is st.profile
    assert set(X.get_context("uronium").pass3_families) <= set(seen["families"])
    assert "fluorinated" in seen["families"]
    assert passes.known_formulas("positive", "uronium") <= seen["curated"] and "C9H9Q" in seen["curated"]
    assert out["channels"] == list(URONIUM) and "fluorinated" in out["families"]
