"""Offline tests for the arbitration between rival formulas of one precursor
(peaky/ms2/arbitrate.py) and its wiring into `ms2 analyze` (`--rival
MZ=F1,F2`, the `--ions` CSV's `group` column or a repeated `mz` cell,
`--rival-ppm`, `--pick-supported`, `ms2_arbitration.csv`).

Known answers, our own data (the committed fixtures):

  * the 2026-09-10 exploratory 15N-ammonium file's 315.106 window holds the
    ^NH4 adduct of a cyclic siloxane, C8H28^NO4Si4+, whose one product is
    its ^NH3 loss C8H25O4Si4+ (297.082). An O-rich rival 2.6 ppm below it,
    C10H20^NO10+, reads that peak as C10H17O10+ (the same ^NH3 loss: Si4 =
    C2O6H-8 twice over, 0.82 mDa): the two read the same product and stay
    undecided -- MS2 cannot split them there. C14H19O8+ (+4.5 ppm, no ^N)
    reads nothing; the window's lead over it rests on that one product:
    undecided (DIAGNOSTIC_MIN).
  * the same file's 136.021 window: C7H6NS+ and C6H6NOSi+ (CS swapped for
    OSi, 0.23 mDa, 1.7 ppm below) read the same three products: undecided.
  * monoterpene standard A's 155.156 window: C10H20^N+ (the 15N-ammonium
    adduct) against a protonated diamine C9H19N2+ 11.5 ppm below it: on
    C10H20^N+'s calibration the diamine reads its skeletal products too
    (both its N in an unnamed remainder: contradicted), and the rest the
    co-isolate C10H19O+ can make, so C10H20^N+ has no product of its own the
    rival cannot read: undecided.
  * every other table is the run with the first rival alone named.
  * no decoy window of these, scored as the window against the other
    decoys, is supported; of the synthetic windows' 72, 9 are (a shift on
    an alias spacing), pinned with the count the docs give.

Synthetic: a siloxane adduct whose odd-Si products only it reads, against an
O-rich rival that reads its ^NH3 loss as an alias (supported; the order
named does not matter); the same rival pair on O-poor organic products
(the siloxane contradicted: Si shown in none of what it reads); aliases
only (the same products); products no rival reads (no better than chance);
a polyoxy remainder credited to no rival; a carbon-poor product (a
cluster's reagent ion) out of the signature, and a signature judged on too
little; an adduct's reagent N leaving whole with a skeletal piece; the
verdict's margins one by one on constructed scores (the decoys' lead -- a
negative one lowers nothing --, the diagnostic count and floor, the excess
over chance, a tie, the signature); the verdict wherever the precursor sits
between two rivals; an unnamed survivor and a named co-isolate's products,
no products; a harmonic ghost; the decoy shift keeping the precursors; the
note (every rival named, no noise centroid, the MS2 window without MS1);
the MS1 / MS2 precursor errors; a first-named rival beyond the target match;
the rival input (the CSV group, a repeated m/z cell, --rival, merged with
--ions, missing-value cells) and its refusals; --pick-supported; a window
with no energy axis; a reagent's neutral merged with common ones (an O-rich
adduct, a nitrate cluster: credited), a co-isolate's product a rival reads
only as an oxygen chain, a thiazole's S lost as CS, three rivals, the
chance as the decoys' mean, a diluted window (too little, not chance),
decoy shifts beyond a wide tolerance, a skipped group window's advice; every
rival read on every rival's calibration (a composition tie undecided
wherever the precursor sits, 5 and 11.5 ppm apart), CO2 + 2 H2O credited (an
O-rich acid and a nitrate cluster against their Si aliases) and the merge
bounds, O lost in a skeletal piece and a reagent lost with common neutrals
shown, the documented rules pinned (an element another rival lacks, a C1
product out of the signature, the H-atom ceiling, the decoy shifts through
the wiring, every read the run's reading, a co-isolate's product a rival
claims on another calibration, the diagnostic products named, an
m/z cell's text, the MS2 note's isolation window, a co-isolate's and a
no-S/N centroid in the note), a group whose windows disagree, a skipped
window supporting a third rival, a --rival group named as typed, NA m/z
cells; the rival the tables read scored on another rival's calibration
(said), one odd-Si product deciding nothing, the skeletal piece's bounds (one
reagent neutral, no oxygen chain), a true S + Si ion contradicted on its
own elements (the pooled signature's limit), a skipped window's advice
where another window supports another rival, the refusal of two groups of
one name, an NA adduct cell; and the CLI end to end."""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from peaky import cli
from peaky.ms2 import analyze as AN
from peaky.ms2 import arbitrate as AR
from peaky.ms2 import sources as S
from tests.test_ms2_breakdown import _synthetic, _tables, _write_run, _write_tables

NAN = float("nan")
NH4_IONS = ("C7H6NS+", "C4H10NS2+", "C8H10NO+",
            "C4H10^NO4+", "C8H9O2+", "C5H14^NO3+", "C6H18^NO2+", "C9H14^N+",
            "C11H13O+", "C11H15O2+", "C11H16^NO+", "C11H18^NO2+", "C12H21O2+",
            "C12H22^NO2+", "C8H28^NO4Si4+")
MT_IONS = ("C10H17+", "C10H20^N+", "C10H19O+")
#: the siloxane adduct and its O-rich rival 2.6 ppm below
SIL, ORG = "C8H28^NO4Si4+", "C10H20^NO10+"
NCE = (1, 10, 20, 30)


def _mz(ion):
    return AN.ion_mz(ion)


def _ions(*formulas):
    return AN.collect_ions((f, NAN, _mz(f)) for f in formulas)


def _group(*rivals, name="g"):
    return AR.RivalGroup(name, tuple(rivals), "test")


def _steps(prec, products, nce=NCE):
    """(nce, peaks): the precursor falls by a quarter per step, each product
    grows from 0 at the lowest energy."""
    out = []
    for k, e in enumerate(nce):
        pk = {prec: 1000.0 * (1 - 0.25 * k)}
        for f, v in products.items():
            if k:
                pk[f] = v * k
        out.append((e, pk))
    return out


#: the siloxane's products: its ^NH3 loss (an alias the O-rich rival reads
#: as C10H17O10+), and four odd-Si ions no Si-free formula can be
SIL_PRODUCTS = {"C8H25O4Si4+": 200.0, "C3H9Si+": 60.0, "C7H21O3Si3+": 50.0, "C2H7OSi+": 40.0,
                "C6H19O3Si3+": 30.0}
#: organic products of at most two O: the siloxane reads them only as Si-free
#: fragments with all four Si in an unnamed remainder (C6H9O2+ as a loss of
#: H2O + H2O + C2H15^NSi4)
ORG_PRODUCTS = {"C6H9O2+": 120.0, "C3H7O2+": 60.0, "C7H11O+": 50.0, "C5H9O+": 40.0,
                "C8H13O2+": 30.0}


def _arbitrate(steps, rivals, *, parent=None, ppm=None, **kw):
    t = _synthetic(parent if parent is not None else _mz(rivals[0]), steps)
    res = AN.analyze_run(t, _ions(*rivals), rivals=[_group(*rivals)], rival_ppm=ppm, **kw)
    return res, res.arbitration.set_index("rival")


# --------------------------------------------------------- synthetic ----
def test_a_siloxane_whose_odd_si_products_only_it_reads_is_supported():
    res, a = _arbitrate(_steps(SIL, SIL_PRODUCTS), [SIL, ORG])
    assert a.loc[SIL, "verdict"] == AR.SUPPORTED and a.loc[ORG, "verdict"] == AR.OUTSCORED
    assert set(a["winner"]) == {SIL} and set(a["window_verdict"]) == {AR.SUPPORTED}
    # every product is the siloxane's; the rival reads the ^NH3 loss alone
    assert a.loc[SIL, "explained"] == pytest.approx(1.0)
    assert a.loc[ORG, "explained"] == pytest.approx(200 / 380)
    assert (a.loc[SIL, "n_products"], a.loc[SIL, "n_diagnostic"]) == (5, 4)
    assert a.loc[SIL, "diagnostic_products"].startswith("C3H9Si+ (-C5H19^NO4Si3) 15.8%; "
                                                        "C7H21O3Si3+ (-CH7^NOSi) 13.2%")
    assert a.loc[SIL, "signature"] == "Si 100%" and a.loc[SIL, "elements"] == "Si"
    assert a.loc[ORG, "elements"] == "" and a.loc[ORG, "signature"] == ""
    # the decoys read little: +-19 mDa lands two products on compositions
    # of both rivals; +11 / +15 mDa the ^NH3 loss on a rival's H2O loss (13.0
    # mDa off it), read on the other rival's calibration
    assert a.loc[SIL, "n_decoys"] == len(AR.DECOY_SHIFTS_MDA)
    assert (a.loc[SIL, "chance"], a.loc[SIL, "chance_max"]) == (pytest.approx(0.118421, abs=1e-6),
                                                                pytest.approx(0.526316, abs=1e-6))
    assert (a.loc[SIL, "lead_chance"], a.loc[SIL, "diagnostic_decoy"]) == (0.0, 0.125)
    assert set(a["read_on"]) == {SIL, ORG} and a.loc[SIL, "read_on"] == SIL
    assert (a.loc[SIL, "named_share"], a.loc[ORG, "named_share"]) == (pytest.approx(200 / 380), 1.0)
    assert a.loc[SIL, "mean_abs_ppm"] == pytest.approx(0.0, abs=1e-6)
    assert a.loc[ORG, "mean_abs_ppm"] == pytest.approx(0.157621, abs=1e-6)
    assert a.loc[SIL, "evidence"].startswith("explains 100% of the products' intensity")
    assert "supported: it leads C10H20^NO10+ (53%) beyond the decoys" in a.loc[SIL, "evidence"]
    assert "outscored: C8H28^NO4Si4+ leads it by 47% of the products" in a.loc[ORG, "evidence"]
    # the run's other tables read the first rival, the window's target
    w = next(x for x in res.windows if x.rival_group == "g")
    assert w.target == SIL and a.loc[SIL, "analysed"] and not a.loc[ORG, "analysed"]
    # the precursor's error under each rival (the synthetic peak sits on SIL)
    assert a.loc[SIL, "precursor_ppm"] == pytest.approx(0.0, abs=1e-6)
    assert a.loc[ORG, "precursor_ppm"] == pytest.approx((_mz(SIL) - _mz(ORG)) / _mz(ORG) * 1e6)


def test_the_supported_rival_does_not_depend_on_the_order_named():
    res, a = _arbitrate(_steps(SIL, SIL_PRODUCTS), [ORG, SIL])
    assert a.loc[SIL, "verdict"] == AR.SUPPORTED and a.loc[ORG, "verdict"] == AR.OUTSCORED
    assert list(a["order"]) == [1, 2] and list(a.index) == [ORG, SIL]
    # the other tables read the first named (said in the console and manifest)
    assert a.loc[ORG, "analysed"] and not a.loc[SIL, "analysed"]
    assert next(x for x in res.windows if x.rival_group == "g").target == ORG


def test_pick_supported_reads_the_tables_with_the_supported_rival():
    plain, _ = _arbitrate(_steps(SIL, SIL_PRODUCTS), [ORG, SIL])
    res, a = _arbitrate(_steps(SIL, SIL_PRODUCTS), [ORG, SIL], pick_supported=True)
    assert a.loc[SIL, "analysed"] and not a.loc[ORG, "analysed"]
    w = next(x for x in res.windows if x.rival_group == "g")
    assert w.target == SIL and set(res.breakdown["ion"]) == {SIL}
    assert set(plain.breakdown["ion"]) == {ORG}
    # the verdicts are the first pass's
    assert a["verdict"].to_dict() == plain.arbitration.set_index("rival")["verdict"].to_dict()
    assert any("--pick-supported reads every table with C8H28^NO4Si4+" in n for n in res.notes)
    g = res.params["arbitration"]["groups"][0]
    assert (g["analysed_with"], g["rivals"]) == (SIL, [ORG, SIL])
    assert w.manifest()["arbitration"]["analysed_with"] == SIL


def test_a_rival_whose_element_never_shows_is_contradicted():
    """O-poor organic products: the siloxane can read every one of them, but
    only as an Si-free fragment of a loss that holds all four Si in an
    unnamed remainder -- its Si signature is absent. The organic rival reads
    the same products, so it is not supported by elimination: undecided."""
    _, a = _arbitrate(_steps(SIL, ORG_PRODUCTS), [ORG, SIL])
    assert a.loc[SIL, "verdict"] == AR.CONTRADICTED
    assert a.loc[SIL, "signature"] == "Si 0%" and a.loc[SIL, "explained"] == pytest.approx(1.0)
    assert ("contradicted: its Si signature is absent: it shows Si in 0% of the 100% of the "
            "products judged (it reads 100% of them)" in a.loc[SIL, "evidence"])
    assert a.loc[ORG, "verdict"] == AR.UNDECIDED and set(a["winner"]) == {""}
    assert ("against C8H28^NO4Si4+ (contradicted): reads the same products as C10H20^NO10+"
            in a.loc[ORG, "evidence"])


def test_aliases_read_the_same_products_and_stay_undecided():
    _, a = _arbitrate(_steps(SIL, {"C8H25O4Si4+": 200.0, "C7H21O4Si4+": 60.0}), [SIL, ORG])
    assert set(a["verdict"]) == {AR.UNDECIDED} and set(a["window_verdict"]) == {AR.UNDECIDED}
    assert a.loc[ORG, "evidence"].endswith(f"undecided: reads the same products as {SIL}")
    # Si is shown in what the siloxane reads: no contradiction
    assert a.loc[SIL, "signature"] == "Si 100%"


def test_products_no_rival_reads_are_no_better_than_chance():
    _, a = _arbitrate(_steps(SIL, {101.5: 200.0, 77.3: 60.0}), [SIL, ORG])
    assert set(a["verdict"]) == {AR.UNDECIDED}
    assert (a["explained"] == 0).all() and (a["n_products"] == 0).all()
    assert "no better than chance" in a.loc[SIL, "evidence"]
    assert a.loc[SIL, "signature"] == "Si n/a"


def test_a_polyoxy_remainder_is_credited_to_no_rival():
    """The O-rich rival reads organic products of few O only through a
    carbon-poor polyoxy remainder -- C9H13O2+ as a loss of H2O + CH5^NO7,
    C9H11O+ as one of H2O + H2O + CH5^NO7 --: an oxygen chain, no molecule.
    The attribution gives it both (no other window ion reads them), the
    arbitration credits neither."""
    assert AR.implausible_remainder("CH5^NO7", 1, charge=1)     # ^NH3 + CO2 + H2O + two O2
    assert AR.implausible_remainder("H2O + H2O6", 1)
    assert not AR.implausible_remainder("C4H8O7", 1)       # four C: no chain by this rule
    assert not AR.implausible_remainder("H2O + H2O + H2O2", 0)   # named pieces
    assert not AR.implausible_remainder("CH2O3", 1)        # three O: carbonic acid
    _, a = _arbitrate(_steps(ORG, {"C9H13O2+": 100.0, "C9H11O+": 100.0}), [ORG, SIL])
    r = a.loc[ORG]
    assert (r["explained"], r["n_products"]) == (0.0, 0)
    assert r["uncredited_share"] == pytest.approx(1.0)
    assert set(a["verdict"]) == {AR.UNDECIDED}


def test_shows_reads_the_product_and_the_named_pieces():
    assert AR.shows("Si", "C3H9Si+", "C5H19^NO4Si3", 1)
    assert AR.shows("^N", "C10H17+", "H3^N", 0)               # the reagent loss, named
    assert not AR.shows("Si", "C6H9O2+", "H2O + H2O + C2H15^NSi4", 1)
    assert AR.shows("Si", "C6H9O2+", "H2O + C2H6OSi", 0)      # Si left in a named neutral


def _table(rows):
    """A product table from (rival, peak, intensity, formula, shown) rows: each
    peak its own fragment, credited whole, the rival's own."""
    return pd.DataFrame({
        "rival": [r[0] for r in rows], "spectrum": ["s"] * len(rows),
        "peak": [r[1] for r in rows], "intensity": [r[2] for r in rows],
        "key": [r[3] for r in rows], "weight": [1.0] * len(rows),
        "credited": [r[2] for r in rows], "own": [True] * len(rows),
        "credible": [True] * len(rows), "readers": [r[0] for r in rows],
        "formula": [r[3] for r in rows], "loss": [""] * len(rows), "unnamed": [0] * len(rows),
        "radical": [False] * len(rows), "ppm": [0.0] * len(rows),
        "shown": [r[4] for r in rows], "uncredited": [0.0] * len(rows)})


def test_a_carbon_poor_product_says_nothing_of_the_signature():
    """A cluster that sheds its skeleton whole as one neutral leaves the
    reagent ion (no carbon): it shows none of the skeleton's elements and
    says nothing of them -- out of the signature both ways. An organic
    product read without the element counts against it."""
    A, B = "C2HF3^NO5-", "C3H5^NO6-"
    rows = [(A, 0, 60.0, "O3^N-", "^N"), (A, 1, 10.0, "CF3-", "F"),
            (A, 2, 30.0, "C2F3O2-", "F;^N")]
    sc, total = AR.rival_scores(_table(rows), [A, B])
    assert total == 100.0 and sc[A].explained == 1.0
    assert sc[A].signature == {"F": 1.0}                   # 40 of the 40 judged
    rows.append((A, 3, 40.0, "C2H3O2-", "^N"))             # an organic product, no F
    sc, _ = AR.rival_scores(_table(rows), [A, B])
    assert sc[A].signature["F"] == pytest.approx(40 / 80)


def test_the_decoy_moves_the_products_and_keeps_the_precursors():
    t = _synthetic(_mz(SIL), _steps(SIL, SIL_PRODUCTS))
    small = AN._window_tables(t, float(t.parents["parent_mz"].iloc[0]))
    spec = S.spectrum_key(_mz(SIL), "hcd10.00")
    assert spec in set(small.spectra["spectrum"])
    rows = small.spectra[small.spectra["spectrum"] == spec].reset_index(drop=True)
    keep = {(spec, 0)}
    out = AN._shifted(small, [spec], keep, 0.011)
    moved = out.spectra[out.spectra["spectrum"] == spec].reset_index(drop=True)
    assert moved["mz"].iloc[0] == rows["mz"].iloc[0]
    assert np.allclose(moved["mz"].iloc[1:], rows["mz"].iloc[1:] + 0.011)
    other = out.spectra[out.spectra["spectrum"] != spec]
    assert other["mz"].equals(small.spectra.loc[small.spectra["spectrum"] != spec, "mz"])


def test_a_window_without_an_energy_axis_is_undecided():
    t = _synthetic(_mz(SIL), _steps(SIL, SIL_PRODUCTS)[:1])
    t.parents["nce"] = NAN
    t.spectra["nce"] = NAN
    res = AN.analyze_run(t, _ions(SIL, ORG), rivals=[_group(SIL, ORG)])
    a = res.arbitration
    assert set(a["verdict"]) == {AR.UNDECIDED}
    assert set(a["evidence"]) == {"undecided: no energy-resolved spectrum (the attribution "
                                  "reads the energy axis)"}


# ------------------------------------------------------- the verdict ----
def _sc(explained, keys=(), signature=None):
    """A constructed score: keys as (name, share, own, readers)."""
    return AR._Score(explained=explained, n_products=len(keys),
                     keys=[(k, s, own, frozenset(r), k, "") for k, s, own, r in keys],
                     signature=dict(signature or {}))


def _decide(real, decoys=()):
    rivals = list(real)
    return AR.decide(real, list(decoys), rivals)


def test_the_margins_one_by_one():
    own = [(f"P{i}", 0.2, True, {"A"}) for i in range(3)]
    real = {"A": _sc(0.9, own), "B": _sc(0.3)}
    win, v, _ = _decide(real, [{"A": _sc(0.05), "B": _sc(0.05)}])
    assert (win, v) == ("A", {"A": AR.SUPPORTED, "B": AR.OUTSCORED})
    # a lead the decoys reach as well (A itself clear of its own chance)
    lead = 0.6 - AR.SHARE_MARGIN + 0.01
    win, v, why = _decide(real, [{"A": _sc(lead + 0.02), "B": _sc(0.02)}])
    assert 0.9 - (lead + 0.02) >= AR.EXCESS_MIN
    assert win == "" and v["B"] == AR.UNDECIDED and "beyond the decoys'" in why["B"]
    # too few products of its own that B cannot read (shared ones do not count)
    few = [(f"P{i}", 0.2, i < AR.DIAGNOSTIC_MIN - 1, {"A"}) for i in range(3)]
    win, v, why = _decide({"A": _sc(0.9, few), "B": _sc(0.3)})
    assert win == "" and "that B cannot read (needs" in why["B"]
    assert "more it shares with a named co-isolate" in why["B"]
    # ... or ones B reads too, or below the floor
    read = [(f"P{i}", 0.2, True, {"A", "B"}) for i in range(3)]
    assert _decide({"A": _sc(0.9, read), "B": _sc(0.3)})[0] == ""
    thin = [(f"P{i}", AR.DIAGNOSTIC_FLOOR / 2, True, {"A"}) for i in range(3)]
    assert _decide({"A": _sc(0.9, thin), "B": _sc(0.3)})[0] == ""
    # a leader no better than its own chance
    win, v, why = _decide(real, [{"A": _sc(0.9 - AR.EXCESS_MIN / 2), "B": _sc(0.0)}])
    assert win == "" and "no better than chance" in why["A"] and v["B"] == AR.UNDECIDED
    # a tie
    win, v, why = _decide({"A": _sc(0.5, own), "B": _sc(0.5, own)})
    assert win == "" and "explains as much as A" in why["B"]


def test_the_signature_rule():
    keys = [(f"P{i}", 0.2, True, {"A"}) for i in range(3)]
    low = {"A": _sc(0.9, keys, {"Si": AR.SIGNATURE_MIN}), "B": _sc(0.4)}
    win, v, why = _decide(low)
    assert v["A"] == AR.CONTRADICTED and "its Si signature is absent" in why["A"]
    # B leads now, but explains less than the contradicted A: undecided
    assert win == "" and v["B"] == AR.UNDECIDED
    assert why["B"] == "against A (contradicted): explains more than B (90%)"
    # shown in more than SIGNATURE_MIN: no contradiction
    ok = {"A": _sc(0.9, keys, {"Si": AR.SIGNATURE_MIN + 0.01}), "B": _sc(0.4)}
    assert _decide(ok)[0] == "A"
    # reading too little to judge: no contradiction either
    tiny = {"A": _sc(AR.SIGNATURE_EXPECT / 2, keys, {"Si": 0.0}), "B": _sc(0.0)}
    assert _decide(tiny)[1]["A"] != AR.CONTRADICTED


# ------------------------------------------------------ own fixtures ----
@pytest.fixture(scope="module")
def nh4():
    return _tables("ms2_nh4_exploratory")


def test_exploratory_siloxane_and_its_alias_read_the_same_product(nh4):
    res = AN.analyze_run(nh4, _ions(*NH4_IONS, ORG), rivals=[_group(SIL, ORG)])
    a = res.arbitration.set_index("rival")
    assert set(a["verdict"]) == {AR.UNDECIDED} and set(a["parent_mz"]) == {315.106}
    assert a.loc[ORG, "evidence"].endswith(f"reads the same products as {SIL}")
    assert a.loc[SIL, "explained"] == pytest.approx(a.loc[ORG, "explained"])
    assert a.loc[SIL, "explained"] == pytest.approx(0.833173, abs=1e-6)
    assert (a.loc[SIL, "n_products"], a.loc[SIL, "signature"]) == (1, "Si 100%")
    # the precursor reads 0.03 ppm as the siloxane, 2.63 ppm as the O-rich rival
    assert a.loc[SIL, "precursor_ppm"] == pytest.approx(0.0305, abs=1e-4)
    assert a.loc[ORG, "precursor_ppm"] == pytest.approx(2.6286, abs=1e-4)
    assert a.loc[SIL, "ms1_ppm"] == pytest.approx(-0.3223, abs=1e-4)
    assert a.loc[ORG, "ms1_ppm"] == pytest.approx(2.2757, abs=1e-4)
    # one decoy of eight (+11, +15 mDa) lands the ^NH3 loss on a rival's H2O
    # loss on the other rival's calibration
    assert a["chance"].tolist() == pytest.approx([0.833173 / 8] * 2, abs=1e-6)
    assert a["chance_max"].tolist() == pytest.approx([0.833173] * 2, abs=1e-6)
    assert (a["diagnostic_decoy"] == 0.125).all() and (a["named_share"] == 1).all()
    assert a.loc[SIL, "mean_abs_ppm"] == pytest.approx(0.025218, abs=1e-6)
    assert a.loc[ORG, "mean_abs_ppm"] == pytest.approx(0.182838, abs=1e-6)
    assert set(a["note"]) == {""}


def test_exploratory_siloxane_against_a_rival_that_reads_nothing(nh4):
    """C14H19O8+ (+4.5 ppm, no ^N) reads no product: the siloxane leads it by
    the whole ^NH3 loss, but on one product -- one product is a coin a decoy
    can toss (DIAGNOSTIC_MIN): undecided, said."""
    res = AN.analyze_run(nh4, _ions(*NH4_IONS, "C14H19O8+"), rivals=[_group(SIL, "C14H19O8+")])
    a = res.arbitration.set_index("rival")
    assert set(a["verdict"]) == {AR.UNDECIDED}
    assert a.loc[SIL, "lead"] == pytest.approx(0.833173, abs=1e-6)
    assert a.loc[SIL, "n_diagnostic"] == 1
    assert a.loc[SIL, "diagnostic_products"] == "C8H25O4Si4+ (-H3^N) 83.3%"
    assert ("on 1 product(s) of its own (>= 1% each) that C14H19O8+ cannot read (needs 2)"
            in a.loc[SIL, "evidence"])
    assert a.loc["C14H19O8+", "precursor_ppm"] == pytest.approx(-4.4700, abs=1e-4)


def _captured(monkeypatch):
    """Every window's (group, reads, decoy reads) as `score_window` gets them."""
    got = []
    orig = AR.score_window

    def keep(parent_mz, group, reads, decoys, **kw):
        got.append((group, reads, decoys))
        return orig(parent_mz, group, reads, decoys, **kw)
    monkeypatch.setattr(AR, "score_window", keep)
    return got


def _decoys_as_real(group, decoys):
    """Each decoy scored as the window, against the other decoys: any
    support there is chance."""
    rivals = list(group.rivals)
    sc = [AR.rival_scores(AR.product_table(d, rivals), rivals)[0] for d in decoys]
    return [AR.decide(sc[i], sc[:i] + sc[i + 1:], rivals)[0] for i in range(len(sc))]


#: the decoys read as real that are supported, per window (the shift's
#: index in DECOY_SHIFTS_MDA: the rival) -- the synthetic windows the tests
#: pin; none of the committed fixtures'
DECOYS_SUPPORTED = {
    "siloxane": {},
    "organic products": {7: SIL},                    # +19 mDa: the contradicted rival
    "aliases": {5: ORG, 6: SIL},                     # +11, +15 mDa
    "adduct": {2: "C9H19N2+"},                       # -11 mDa: the outscored rival
    "near tie": {},
    "between": {1: "C12H22N4O3+", 5: "C12H22N4O3+"},  # -15, +11 mDa
    "o-rich adduct": {0: SIL, 7: SIL},               # +-19 mDa: the true ion is ORG
    "nitrate cluster": {0: "C12H14F3O5-"},           # -19 mDa: the contradicted alias
    "c20 siloxane, 10 ppm": {}}


def _decoy_windows(nh4):
    """(name, run) of every window the decoy count is pinned on: the
    committed fixtures' four, then the synthetic ones."""
    B = "C12H22N4O3+"
    between = _moved(_steps(B, {"C8H13O2+": 300.0, "C7H11O+": 150.0, "C6H9O+": 100.0,
                                "C9H15O2+": 80.0}), 1.0, keep={B})
    nitrate = {"^NO3-": 150.0, "C10H15O6-": 150.0, "C10H13O5-": 250.0, "C9H15O4-": 300.0,
               "C9H13O3-": 150.0}
    return [
        ("own siloxane", lambda: AN.analyze_run(nh4, _ions(*NH4_IONS, ORG),
                                                rivals=[_group(SIL, ORG)])),
        ("own siloxane, C14", lambda: AN.analyze_run(nh4, _ions(*NH4_IONS, "C14H19O8+"),
                                                     rivals=[_group(SIL, "C14H19O8+")])),
        ("own thiazole", lambda: AN.analyze_run(nh4, _ions(*NH4_IONS, "C6H6NOSi+"),
                                                rivals=[_group("C7H6NS+", "C6H6NOSi+")])),
        ("own standard A", lambda: AN.analyze_run(
            _tables("ms2_monoterpene_a"), _ions(*MT_IONS, "C9H19N2+"), rival_ppm=12,
            rivals=[_group("C10H20^N+", "C9H19N2+")])),
        ("siloxane", lambda: _arbitrate(_steps(SIL, SIL_PRODUCTS), [SIL, ORG])),
        ("organic products", lambda: _arbitrate(_steps(SIL, ORG_PRODUCTS), [ORG, SIL])),
        ("aliases", lambda: _arbitrate(_steps(SIL, {"C8H25O4Si4+": 200.0, "C7H21O4Si4+": 60.0}),
                                       [SIL, ORG])),
        ("adduct", lambda: _arbitrate(_steps("C10H20^N+", ADDUCT_PRODUCTS),
                                      ["C10H20^N+", "C9H19N2+"], ppm=12)),
        ("near tie", lambda: _arbitrate(_steps(SIL, NEAR_TIE), [SIL, "C14H19O8+"])),
        ("between", lambda: AN.analyze_run(_synthetic(round(_mz(B), 4), between),
                                           _ions("C14H24NO4+", B),
                                           rivals=[_group("C14H24NO4+", B)])),
        ("o-rich adduct", lambda: _arbitrate(_steps(ORG, ORG_OWN), [ORG, SIL])),
        ("nitrate cluster", lambda: _arbitrate(_steps("C10H16^NO9-", nitrate),
                                               ["C10H16^NO9-", "C12H14F3O5-"])),
        ("c20 siloxane, 10 ppm", lambda: _arbitrate(_steps(S10, S10_PRODUCTS), [S10, S10_RIVAL],
                                                    tol_ppm=10.0))]


def test_decoys_read_as_real_are_supported_now_and_then(monkeypatch, nh4):
    """Each decoy scored as the window, against the other seven: any support
    there is false. None of the committed fixtures' 32; of the synthetic
    windows' 72, 9 -- a shift on an alias spacing (-11 mDa: CO for N2,
    11.2; +-19: about Si for CO, 18.0; +11 / +15: the ^NH3 loss for the H2O
    loss, 13.0, read on another rival's calibration) lands a window's few
    products on a rival's compositions, at times the rival the window itself
    does not support. Pinned, and the count the docs and the constants'
    comment give."""
    got = _captured(monkeypatch)
    own, syn = [], []
    for name, run in _decoy_windows(nh4):
        got.clear()
        run()
        assert len(got) == 1, name
        group, _, decoys = got[0]
        assert len(decoys) == len(AR.DECOY_SHIFTS_MDA)
        res = _decoys_as_real(group, decoys)
        (own if name.startswith("own") else syn).append(res)
        if not name.startswith("own"):
            assert {i: r for i, r in enumerate(res) if r} == DECOYS_SUPPORTED[name], name
        else:
            assert res == [""] * len(decoys), name
    n_own = sum(len(r) for r in own)
    n, k = sum(len(r) for r in syn), sum(1 for r in syn for x in r if x)
    assert (n_own, n, k) == (32, 72, 9)
    root = Path(__file__).resolve().parents[1]
    for path in (root / "docs" / "MS2.md", root / "peaky" / "ms2" / "arbitrate.py"):
        text = " ".join(path.read_text().split())
        assert f"{k} of the {n} decoys of the synthetic windows" in text.replace("#: ", ""), path
        assert f"none of the {n_own} of the committed fixtures'" in text.replace("#: ", ""), path
    # and this module's own summary of the known answers
    assert f"of the synthetic windows' {n}, {k} are" in " ".join(__doc__.split())


def test_exploratory_thiazole_and_its_si_rival_read_the_same_products(nh4):
    res = AN.analyze_run(nh4, _ions(*NH4_IONS, "C6H6NOSi+"),
                         rivals=[_group("C7H6NS+", "C6H6NOSi+")])
    a = res.arbitration.set_index("rival")
    assert set(a["verdict"]) == {AR.UNDECIDED}
    assert a.loc["C6H6NOSi+", "evidence"].endswith("reads the same products as C7H6NS+")
    assert (a["n_products"] == 3).all()
    # the co-isolated dithiocarbamate's own products are no product here
    assert a.loc["C7H6NS+", "explained"] == pytest.approx(0.751297, abs=1e-6)
    assert a.loc["C6H6NOSi+", "signature"] == "O 94%; Si 94%"


def test_standard_a_adduct_shares_its_products_with_the_co_isolate():
    t = _tables("ms2_monoterpene_a")
    res = AN.analyze_run(t, _ions(*MT_IONS, "C9H19N2+"), rival_ppm=12,
                         rivals=[_group("C10H20^N+", "C9H19N2+")])
    a = res.arbitration.set_index("rival")
    assert set(a["window_verdict"]) == {AR.UNDECIDED} and set(a["winner"]) == {""}
    assert (a.loc["C10H20^N+", "verdict"], a.loc["C9H19N2+", "verdict"]) == (AR.UNDECIDED,
                                                                            AR.CONTRADICTED)
    # the in-window survivors of unnamed ions are no products
    assert a.loc["C10H20^N+", "explained"] == pytest.approx(0.711448, abs=1e-6)
    # on its own calibration (+9.8 ppm) the diamine reads none; on
    # C10H20^N+'s it reads the skeletal ones -- with both its N in an
    # unnamed remainder, contradicted --: none is C10H20^N+'s own
    assert a.loc["C9H19N2+", "explained"] == pytest.approx(0.185711, abs=1e-6)
    assert (a.loc["C9H19N2+", "read_on"], a.loc["C9H19N2+", "signature"]) == ("C10H20^N+",
                                                                               "N 0%")
    assert ("but on 0 product(s) of its own (>= 1% each) that C9H19N2+ cannot read (needs 2); "
            "1 more it shares with a named co-isolate") in a.loc["C10H20^N+", "evidence"]
    assert a.loc["C10H20^N+", "signature"] == "^N 100%"     # ^NH3 leaves whole everywhere
    assert a.loc["C9H19N2+", "chance"] == pytest.approx(0.081291, abs=1e-6)
    assert a.loc["C9H19N2+", "chance_max"] == pytest.approx(0.650324, abs=1e-6)
    assert a.loc["C9H19N2+", "diagnostic_decoy"] == 0.125
    assert a.loc["C10H20^N+", "named_share"] == pytest.approx(0.643057, abs=1e-6)
    assert a.loc["C10H20^N+", "mean_abs_ppm"] == pytest.approx(0.461082, abs=1e-6)
    assert a.loc["C10H20^N+", "ms1_ppm"] == pytest.approx(-1.581010, abs=1e-6)
    assert a.loc["C9H19N2+", "ms1_ppm"] == pytest.approx(9.929579, abs=1e-6)
    assert a.loc["C10H20^N+", "precursor_ppm"] == pytest.approx(-1.7184, abs=1e-4)
    assert a.loc["C9H19N2+", "precursor_ppm"] == pytest.approx(9.7922, abs=1e-4)
    # the docs' known answer quotes the diamine's precursor error, not the
    # formulas' gap (11.5 ppm)
    docs = " ".join((Path(__file__).resolve().parents[1] / "docs" / "MS2.md").read_text().split())
    assert f"its precursor {a.loc['C9H19N2+', 'precursor_ppm']:.1f} ppm off" in docs
    # the rival is no window ion anywhere: 11.5 ppm off, it would be read as
    # a co-isolate of its own window, not left out as an alias
    plain = AN.analyze_run(t, _ions(*MT_IONS))
    assert "C9H19N2+" not in set(res.breakdown["ion"])
    for name in ("readings", "breakdown", "e50", "fragments", "branching"):
        assert getattr(plain, name).equals(getattr(res, name)), name


def test_the_other_tables_are_the_first_rivals_run(nh4):
    plain = AN.analyze_run(nh4, _ions(*NH4_IONS))
    rival = AN.analyze_run(nh4, _ions(*NH4_IONS, ORG), rivals=[_group(SIL, ORG)])
    for name in ("readings", "breakdown", "e50", "fragments", "branching"):
        assert getattr(plain, name).equals(getattr(rival, name)), name
    assert plain.arbitration is None and not plain.params["arbitration"]["on"]
    assert rival.params["arbitration"]["on"]
    assert [w.status for w in plain.windows] == [w.status for w in rival.windows]
    assert "rival_group" not in plain.windows[0].manifest()
    assert "rival_group" not in plain.params["ions"][0]
    got = AR.frame([]).dtypes
    assert rival.arbitration.dtypes.equals(got) and list(got.index) == list(AR.ARBITRATION_COLUMNS)


# ------------------------------------------------------ the input ----
def test_rival_groups_from_the_ions_csv():
    df = pd.DataFrame({"formula": [SIL, ORG, "C10H17+", "C10H20^N+", "C9H19N2+", "C10H19O+"],
                       "mz": ["315.106", "315.106", "137.13", "155.156", "155.156", ""],
                       "group": ["si", "si", "", "", "", ""]}, dtype=str)
    gs = AR.read_rival_groups(df, "ions.csv")
    assert [(g.name, g.rivals, g.source) for g in gs] == [
        ("si", (SIL, ORG), "ions csv group"),
        ("155.156", ("C10H20^N+", "C9H19N2+"), "ions csv m/z")]
    with pytest.raises(ValueError, match="row 2: group 'x' names one formula only"):
        AR.read_rival_groups(pd.DataFrame({"formula": [SIL], "group": ["x"]}), "ions.csv")
    two = pd.DataFrame({"formula": [SIL, ORG, SIL, "C14H19O8+"], "group": ["a", "a", "b", "b"]})
    with pytest.raises(ValueError, match="row 4: C8H28\\^NO4Si4\\+ is a rival in two groups"):
        AR.read_rival_groups(two, "ions.csv")


def test_the_rival_spec_and_the_group_checks():
    assert AR.parse_rival_spec("315.106=C8H28^NO4Si4+, C10H20^NO10+", check_ppm=10,
                               check_mda=5) == (315.106, [SIL, ORG])
    for spec, msg in (("315.106", "expected MZ=FORMULA"), ("x=C8H28^NO4Si4+,C10H20^NO10+",
                                                          "is not an m/z"),
                      ("315.106=C8H28^NO4Si4+", "two formulas or more"),
                      ("315.2=C8H28^NO4Si4+,C10H20^NO10+", "a typo"),
                      ("315.106=C8H28^NO4Si4+,C8H28^NO4Si4+", "named twice")):
        with pytest.raises(ValueError, match=msg):
            AR.parse_rival_spec(spec, check_ppm=10, check_mda=5)
    AR.check_group(_group(SIL, ORG), tol_ppm=5, tol_mda=0.3)
    with pytest.raises(ValueError, match="7.1 ppm\\) apart, beyond the reading tolerance"):
        AR.check_group(_group(ORG, "C14H19O8+"), tol_ppm=5, tol_mda=0.3)
    AR.check_group(_group(ORG, "C14H19O8+"), tol_ppm=5, tol_mda=0.3, rival_ppm=8)
    with pytest.raises(ValueError, match="--rival-ppm must be in"):
        AR.check_group(_group(SIL, ORG), tol_ppm=5, tol_mda=0.3, rival_ppm=60)
    with pytest.raises(ValueError, match="both polarities"):
        AR.check_group(_group("C10H17+", "C10H17-"), tol_ppm=5, tol_mda=0.3)
    t = _synthetic(_mz(SIL), _steps(SIL, SIL_PRODUCTS))
    with pytest.raises(ValueError, match="C10H20\\^NO10\\+ is not a named ion"):
        AN.analyze_run(t, _ions(SIL), rivals=[_group(SIL, ORG)])


def test_a_group_no_window_is_the_target_of_is_said():
    t = _synthetic(_mz(SIL), _steps(SIL, SIL_PRODUCTS))
    res = AN.analyze_run(t, _ions(SIL, "C7H6NS+", "C6H6NOSi+"),
                         rivals=[_group("C7H6NS+", "C6H6NOSi+", name="far")])
    assert res.arbitration is not None and not len(res.arbitration)
    assert any("rival group 'far'" in n and "not arbitrated" in n for n in res.notes)


# ------------------------------------------------------- the CLI ----
@pytest.fixture
def no_server(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("ms2 analyze asked for Mascope")
    from peaky.io import io_mascope as IO
    monkeypatch.setattr(cli, "_require_creds", boom)
    monkeypatch.setattr(IO, "connect", boom)


def _synthetic_run(d, products=SIL_PRODUCTS, first=SIL):
    return _write_tables(d, _synthetic(_mz(first), _steps(first, products)))


def test_cli_rival_writes_the_arbitration(tmp_path, no_server, capsys):
    run = _synthetic_run(tmp_path / "run")
    out = tmp_path / "out"
    assert cli.main(["ms2", "analyze", "--run-dir", str(run), "--rival",
                     f"315.106={ORG},{SIL}", "--no-figures", "--out-dir", str(out)]) == 0
    printed = capsys.readouterr().out
    a = pd.read_csv(out / "ms2_arbitration.csv", keep_default_na=False, na_values=[""])
    assert list(a.columns) == list(AR.ARBITRATION_COLUMNS)
    assert dict(zip(a["rival"], a["verdict"])) == {ORG: AR.OUTSCORED, SIL: AR.SUPPORTED}
    assert ("  rivals [315.106]: C10H20^NO10+ outscored (53% of the products, decoys 12%); "
            "C8H28^NO4Si4+ supported (100% of the products, decoys 12%, 4 diagnostic)") in printed
    assert "the tables read C10H20^NO10+ (named first)" in printed
    man = json.loads((out / "ms2_analyze_manifest.json").read_text())
    arb = man["parameters"]["arbitration"]
    assert arb["on"] and arb["groups"] == [{"name": "315.106", "rivals": [ORG, SIL],
                                            "source": "--rival", "analysed_with": ORG}]
    assert arb["decoy_shifts_mda"] == list(AR.DECOY_SHIFTS_MDA)
    assert arb["note_min_sn"] == AN.ARBITRATION_NOTE_SN == 3.0
    assert man["counts"]["arbitration"] == {"windows": 1, "supported": 1, "undecided": 0,
                                            "rivals": {"supported": 1, "outscored": 1,
                                                       "contradicted": 0, "undecided": 0}}
    win = man["windows"][0]
    assert win["rival_group"] == "315.106" and win["arbitration"]["winner"] == SIL
    assert {i["formula"]: i["rival_group"] for i in man["parameters"]["ions"]} == \
        {ORG: "315.106", SIL: "315.106"}
    assert AN.ARBITRATION_NOTE in man["notes"]
    from peaky.reporting import provenance as PV
    assert man["files"]["ms2_arbitration.csv"] == PV.sha1_file(str(out / "ms2_arbitration.csv"))
    # with --pick-supported the tables read the supported one
    assert cli.main(["ms2", "analyze", "--run-dir", str(run), "--rival",
                     f"315.106={ORG},{SIL}", "--pick-supported", "--no-figures",
                     "--out-dir", str(out)]) == 0
    assert "the tables read C8H28^NO4Si4+ (--pick-supported)" in capsys.readouterr().out
    bd = pd.read_csv(out / "ms2_breakdown.csv")
    assert set(bd["ion"]) == {SIL}
    # again with no rival: the table is not this run's -- removed
    assert cli.main(["ms2", "analyze", "--run-dir", str(run), "--ion", f"315.106={SIL}",
                     "--no-figures", "--out-dir", str(out)]) == 0
    assert not (out / "ms2_arbitration.csv").exists()
    man = json.loads((out / "ms2_analyze_manifest.json").read_text())
    assert man["counts"]["arbitration"] is None and not man["parameters"]["arbitration"]["on"]
    assert "rivals [" not in capsys.readouterr().out


def test_cli_ions_csv_group_column(tmp_path, no_server):
    run = _synthetic_run(tmp_path / "run")
    ions = tmp_path / "ions.csv"
    ions.write_text(f"formula,mz,group\n{SIL},315.106,si\n{ORG},315.105,si\n")
    assert cli.main(["ms2", "analyze", "--run-dir", str(run), "--ions", str(ions),
                     "--no-figures"]) == 0
    a = pd.read_csv(run / "ms2_arbitration.csv", keep_default_na=False, na_values=[""])
    assert set(a["group"]) == {"si"} and dict(zip(a["rival"], a["order"])) == {SIL: 1, ORG: 2}


@pytest.mark.parametrize("argv, message", [
    (["--rival", "315.106=C10H20^NO10+,C14H19O8+"], "beyond the reading tolerance"),
    (["--rival", "315.106=C8H28^NO4Si4+"], "two formulas or more"),
    (["--rival", "315.106=C8H28^NO4Si4+,C10H20^NO10+", "--rival",
      "315.107=C8H28^NO4Si4+,C14H19O8+", "--rival-ppm", "8"], "is a rival in two groups"),
    (["--rival", "315.106=C8H28^NO4Si4+,C10H20^NO10+", "--ion", "315.106=C10H20^NO10+",
      "--ion", "315.106=C10H20^NO10+"], "given twice"),
    (["--ion", "315.106=C8H28^NO4Si4+", "--rival-ppm", "8"], "need rival formulas"),
    (["--ion", "315.106=C8H28^NO4Si4+", "--pick-supported"], "need rival formulas"),
    (["--rival", "315.106=C8H28^NO4Si4+,C10H20^NO10+", "--rival-ppm", "80"],
     "--rival-ppm must be in"),
    (["--rival", "315.106=C8H28^NO4Si4+,C10H20^NO10+", "--rival",
      "315.106=C14H19O8+,C18H19O3S+", "--rival-ppm", "8"], "two rival groups are named"),
])
def test_cli_refuses_bad_rivals(tmp_path, no_server, argv, message):
    run = _synthetic_run(tmp_path / "run")
    with pytest.raises(SystemExit, match=message):
        cli.main(["ms2", "analyze", "--run-dir", str(run), "--no-figures"] + argv)


def test_cli_help_and_the_constant():
    assert cli.AR_RIVAL_PPM_MAX == AR.RIVAL_PPM_MAX
    p = cli.build_parser()
    sub = next(a for a in p._actions if a.__class__.__name__ == "_SubParsersAction").choices["ms2"]
    an = next(a for a in sub._actions if a.__class__.__name__ == "_SubParsersAction"
              ).choices["analyze"]
    text = an.format_help()
    assert "--rival MZ=F1,F2" in text and "--rival-ppm PPM" in text and "--pick-supported" in text
    a = p.parse_args(["ms2", "analyze", "--run-dir", "D", "--ion", "137.13=C10H17+"])
    assert (a.rival, a.rival_ppm, a.pick_supported) == ([], None, False)


# ------------------------------------------- the reads and the scores ----
def _moved(steps, ppm, keep=()):
    """The same steps with every peak but those in `keep` moved by `ppm` (a
    formula key becomes its exact m/z first)."""
    return [(e, {((_mz(k) if isinstance(k, str) else float(k))
                  * (1.0 if k in keep else 1.0 + ppm * 1e-6)): v for k, v in pk.items()})
            for e, pk in steps]


def _with_ms1(t, rows):
    """`t` with an MS1 window table for its one parent: (m/z, share, S/N) rows."""
    p = float(t.parents["parent_mz"].iloc[0])
    w = pd.DataFrame({"sample_id": "S", "parent_mz": p, "mz": [r[0] for r in rows],
                      "offset_da": [r[0] - p for r in rows],
                      "intensity": [r[1] * 1e5 for r in rows], "share": [r[1] for r in rows],
                      "resolution": NAN, "signal_to_noise": [r[2] for r in rows],
                      "is_parent": False})
    return AN.RunTables(spectra=t.spectra, parents=t.parents,
                        window=S._frame(w, S.WINDOW_COLUMNS), manifest=t.manifest)


def test_a_reagent_n_leaving_with_a_skeletal_piece_is_shown():
    """An ammonium adduct sheds its reagent as ^NH3, and at higher energies
    with a skeletal neutral: the decomposer prints the fewest pieces, so
    ^NH3 + C4H8 reads as ONE unnamed piece C4H11^N. The ^N still leaves
    whole in a library neutral (H3^N) next to a skeletal piece: it is shown.
    A carbon-poor rest is no skeleton, and a skeleton's own heteroatoms are
    read as printed: S + Si + H4 is an alias of O4, and an O-rich product
    read through H4O2SSi is no SO2 lost next to SiH4."""
    assert AR.shows("^N", "C6H9+", "C4H11^N", 1)
    assert AR.shows("^N", "C10H17+", "C10H23^NO4", 1)
    assert AR.shows("N", "C6H9+", "C4H11N", 1)                  # H3N + C4H8
    assert AR.shows("^N", "C2H7OSi+", "C6H21^NO3Si3", 1)        # H3^N + C6H18O3Si3
    assert not AR.shows("^N", "C7H21O3Si3+", "CH7^NOSi", 1)     # the rest CH4OSi: one C
    assert not AR.shows("^N", "C6H9+", "C2H3^NO", 1)            # H3^N leaves C2O: no piece
    assert not AR.shows("Si", "C6H9O2+", "H2O + H2O + C2H15^NSi4", 1)
    assert not AR.shows("S", "C8H13O+", "H2O + H4O2SSi", 1)
    assert not AR.shows("Si", "C5H9O+", "C5H14OSi", 1)          # Si: as printed
    # an S + Si alias of the O-rich rival reading its organic products only
    # through such remainders stays contradicted
    alias = "C10H24^NO6SSi+"
    _, a = _arbitrate(_steps(ORG, ORG_PRODUCTS), [ORG, alias])
    assert a.loc[alias, "verdict"] == AR.CONTRADICTED and a.loc[alias, "signature"] == "S 0%; Si 0%"
    # the ^NH4+ adduct of a monoterpene whose products are all skeletal
    # (C6H9+ read as a loss of C4H11^N, ...): its ^N leaves whole in every
    # one of them -- not contradicted (the diamine reads them too)
    A, B = "C10H20^N+", "C9H19N2+"
    prods = {"C6H9+": 300.0, "C7H11+": 120.0, "C8H13+": 80.0, "C5H7+": 60.0}
    _, a = _arbitrate(_steps(A, prods), [A, B], ppm=12)
    assert a.loc[A, "verdict"] == AR.UNDECIDED and a.loc[A, "signature"] == "^N 100%"
    # the siloxane adduct against a rival with no ^N: its ^NH3 loss weak, its
    # Si products read with the ^N in an unnamed remainder -- shown there too,
    # but where the rest is carbon-poor (C7H21O3Si3+, 50 of 230)
    _, a = _arbitrate(_steps(SIL, {**SIL_PRODUCTS, "C8H25O4Si4+": 50.0}), [SIL, "C14H19O8+"])
    assert a.loc[SIL, "verdict"] == AR.SUPPORTED
    assert a.loc[SIL, "signature"] == "^N 78%; Si 100%"


def test_the_verdict_does_not_follow_the_precursor_between_two_rivals():
    """Two rivals 5.0 ppm apart (within the reading tolerance) and products
    that are sub-formulas of both. Each rival's read is recalibrated on its
    own precursor, so where the precursor sits moves every product by the
    rivals' gap under one of them -- a product at the tolerance's edge falls
    out for that one only. Read on one calibration both read it: no product
    is one the other cannot read, wherever the precursor sits."""
    A, B = "C14H24NO4+", "C12H22N4O3+"
    prods = {"C8H13O2+": 300.0, "C7H11O+": 150.0, "C6H9O+": 100.0, "C9H15O2+": 80.0}
    for err in (-1.0, 1.0):
        for on in (A, B):
            steps = _moved(_steps(on, prods), err, keep={on})
            res = AN.analyze_run(_synthetic(round(_mz(on), 4), steps), _ions(A, B),
                                 rivals=[_group(A, B)])
            a = res.arbitration.set_index("rival")
            assert set(a["verdict"]) == {AR.UNDECIDED}, (err, on)
            assert (a["n_diagnostic"] == 0).all(), (err, on)


def test_a_negative_decoy_lead_does_not_lower_the_margin():
    """The decoys raise the bar, never lower it: a rival whose decoys favour
    the other one (its shifted products land on the other's aliases) still
    needs a real lead of SHARE_MARGIN."""
    own = [(f"P{i}", 0.17, True, {"A"}) for i in range(3)]
    real = {"A": _sc(0.52, own), "B": _sc(0.50)}
    decoys = ([{"A": _sc(0.0), "B": _sc(0.43)}] * 2 + [{"A": _sc(0.0), "B": _sc(0.0)}] * 6)
    win, v, why = _decide(real, decoys)
    assert win == "" and v["B"] == AR.UNDECIDED
    assert "A leads it by 2% of the products, 2% beyond the decoys' 0%" in why["B"]
    # synthetic: a near tie between the siloxane's ^NH3 loss and three
    # organic products of the O-rich rival
    _, a = _arbitrate(_steps(SIL, NEAR_TIE), [SIL, "C14H19O8+"])
    assert set(a["verdict"]) == {AR.UNDECIDED}


#: the siloxane's ^NH3 loss against three organic products of C14H19O8+
NEAR_TIE = {"C8H25O4Si4+": 200.0, "C13H19O6+": 68.0, "C13H17O5+": 68.0, "C12H17O4+": 68.0}


def _far_first():
    """The siloxane window read 6 ppm low: its precursor matches the
    siloxane (6 ppm) and not C14H19O8+ (10.5 ppm, beyond the target match)."""
    return _synthetic(_mz(SIL), _moved(_steps(SIL, SIL_PRODUCTS), -6.0))


def test_a_first_named_rival_beyond_the_target_match_leaves_the_tables_as_the_plain_run():
    """The other tables are the run with the first rival alone named -- also
    when that rival lies beyond the target match: the plain run skips the
    window, and so do they. The arbitration still reads every rival."""
    t, far = _far_first(), "C14H19O8+"
    plain = AN.analyze_run(t, _ions(far))
    res = AN.analyze_run(t, _ions(far, SIL), rivals=[_group(far, SIL)])
    assert [w.status for w in plain.windows] == ["skipped"]
    assert [w.status for w in res.windows] == ["skipped"]
    for name in ("readings", "breakdown", "e50", "fragments", "branching"):
        assert getattr(plain, name).equals(getattr(res, name)), name
    a = res.arbitration.set_index("rival")
    assert not a["analysed"].any() and a.loc[SIL, "verdict"] == AR.SUPPORTED
    assert a.loc[far, "precursor_ppm"] == pytest.approx(-10.5, abs=0.05)
    assert any("the other tables do not read this window with C14H19O8+" in n
               for n in res.windows[0].notes)
    # the same past --rival-ppm 10: standard A's adduct named second
    P, R1 = "C10H20^N+", "C9H19N2+"
    t = _synthetic(_mz(P), _steps(P, ADDUCT_PRODUCTS))
    plain = AN.analyze_run(t, _ions(R1))
    res = AN.analyze_run(t, _ions(R1, P), rivals=[_group(R1, P)], rival_ppm=12)
    for name in ("readings", "breakdown", "e50", "fragments", "branching"):
        assert getattr(plain, name).equals(getattr(res, name)), name
    assert res.windows[0].status == "skipped"


def test_an_unnamed_survivor_in_the_window_is_no_product_and_is_said():
    """An unnamed ion co-isolated in the window that does not fragment: no
    singly charged product of any rival lies above its precursor less an H
    atom, so it is no product -- the verdict is the one without it. With no
    MS1 window table, the lowest-energy MS2 spectrum says it outweighs the
    precursor there."""
    def steps(surv):
        return [(e, {**pk, 315.0: surv}) for e, pk in _steps(SIL, SIL_PRODUCTS)]
    _, a = _arbitrate(steps(3000.0), [SIL, ORG])
    assert a.loc[SIL, "verdict"] == AR.SUPPORTED
    assert a.loc[SIL, "explained"] == pytest.approx(1.0)
    note = a.loc[SIL, "note"]
    assert "the lowest-energy MS2 spectrum" in note and "315.0000" in note
    assert "the products may be another ion's" in note
    _, a = _arbitrate(steps(100.0), [SIL, ORG])
    assert a.loc[SIL, "note"] == ""       # the precursor outweighs it


def test_the_signature_is_judged_on_enough_of_the_products():
    """A cluster's reagent ion (carbon-poor, the skeleton shed whole) holds
    nearly every product, and one small organic peak is read without Si:
    the Si signature rests on a thousandth of the products -- not judged."""
    _, a = _arbitrate(_steps(SIL, {"H4^N+": 500.0, "C5H9O+": 0.5}), [SIL, "C14H19O8+"],
                      ppm=8)
    assert a.loc[SIL, "verdict"] != AR.CONTRADICTED
    assert a.loc[SIL, "signature"] == "^N 100%; Si n/a"
    # judged on enough of them, it is (organic products of at most two O)
    _, a = _arbitrate(_steps(SIL, ORG_PRODUCTS), [ORG, SIL])
    assert "it shows Si in 0% of the 100% of the products judged" in a.loc[SIL, "evidence"]


CO = "C17H15O6+"
#: a named co-isolate's products, none a sub-formula of either rival
CO_PRODUCTS = {"C16H13O5+": 200.0, "C15H11O4+": 200.0, "C14H11O3+": 200.0, "C13H9O3+": 200.0}


def test_a_named_co_isolates_products_are_no_rivals_and_no_decoys():
    """A named co-isolate whose products no rival reads: they are no rival's
    in the window, so they are no product of the arbitration -- nor moved in
    the decoys, where the co-isolate could not take them back and a rival
    would read them at full weight by chance."""
    steps = [(e, {**pk, CO: pk[SIL], **{f: v * k for f, v in CO_PRODUCTS.items() if k}})
             for k, (e, pk) in enumerate(_steps(SIL, SIL_PRODUCTS))]
    t = _synthetic(_mz(SIL), steps)
    res = AN.analyze_run(t, _ions(SIL, ORG, CO), rivals=[_group(SIL, ORG)])
    assert res.windows[0].co_isolates == [CO]
    a = res.arbitration.set_index("rival")
    plain = _arbitrate(_steps(SIL, SIL_PRODUCTS), [SIL, ORG])[1]
    for col in ("verdict", "explained", "chance", "chance_max", "lead", "lead_chance",
                "n_diagnostic", "product_intensity"):
        assert a[col].equals(plain[col]), col


def _ms1_rivals():
    """C11H13O+ at the precursor, C7H14^NO3+ 13.9 ppm below it named first."""
    A, B = "C11H13O+", "C7H14^NO3+"
    t = _synthetic(_mz(A), _steps(A, {"C9H9+": 200.0, "C7H7+": 100.0, "C9H11+": 50.0}))
    return A, B, _with_ms1(t, [(_mz(A), 0.75, 50.0), (_mz(B), 0.10, 20.0),
                               (161.05, 0.15, 20.0)])


def test_the_note_counts_every_rival_as_named():
    A, B, t = _ms1_rivals()
    for order in ([B, A], [A, B]):
        res = AN.analyze_run(t, _ions(*order), rivals=[_group(*order)], rival_ppm=15)
        a = res.arbitration.set_index("rival")
        assert "no named ion matches" not in a.loc[A, "note"], order
        assert "no named ion matches" not in a.loc[B, "note"], order


def test_ms1_ppm_is_read_at_the_precursors_centroid():
    A, B, t = _ms1_rivals()
    res = AN.analyze_run(t, _ions(B, A), rivals=[_group(B, A)], rival_ppm=15)
    a = res.arbitration.set_index("rival")
    assert a.loc[A, "ms1_ppm"] == pytest.approx(0.0, abs=1e-6)
    assert a.loc[B, "ms1_ppm"] == pytest.approx((_mz(A) - _mz(B)) / _mz(B) * 1e6, abs=1e-6)
    assert a.loc[B, "ms1_ppm"] == pytest.approx(a.loc[B, "precursor_ppm"], abs=1e-6)


def test_the_note_cites_no_noise_centroid():
    """An MS1 centroid at S/N 0.15 is no ion the products could be: the
    note says MS1 does not see the precursor, and names no centroid. A real
    one (S/N 20) holding more than the precursor is named."""
    t = _synthetic(_mz(SIL), _steps(SIL, SIL_PRODUCTS))
    res = AN.analyze_run(_with_ms1(t, [(315.0, 1.0, 0.15)]), _ions(SIL, ORG),
                         rivals=[_group(SIL, ORG)])
    note = res.arbitration["note"].iloc[0]
    assert "MS1 does not see the precursor" in note and "315.0000" not in note
    res = AN.analyze_run(_with_ms1(t, [(_mz(SIL), 0.3, 40.0), (315.0, 0.6, 20.0),
                                       (315.05, 0.1, 0.5)]),
                         _ions(SIL, ORG), rivals=[_group(SIL, ORG)])
    note = res.arbitration["note"].iloc[0]
    assert note.startswith("an MS1 centroid no named ion matches, at m/z 315.0000, holds 60% "
                           "of the isolation window, more than the precursor (30%)")


def test_one_formula_in_two_groups_or_two_groups_of_one_name_are_refused():
    t = _synthetic(_mz(SIL), _steps(SIL, SIL_PRODUCTS))
    ions = _ions(SIL, ORG, "C14H19O8+", "C7H6NS+", "C6H6NOSi+")
    with pytest.raises(ValueError, match="C8H28\\^NO4Si4\\+ is a rival in two groups"):
        AN.analyze_run(t, ions, rivals=[_group(SIL, ORG, name="a"),
                                        _group("C14H19O8+", SIL, name="b")])
    with pytest.raises(ValueError, match="two rival groups are named 'a'") as e:
        AN.analyze_run(t, ions, rivals=[_group(SIL, ORG, name="a"),
                                        _group("C7H6NS+", "C6H6NOSi+", name="a")])
    # most likely one ion's rivals split in two: said first
    msg = str(e.value)
    assert ("the rivals of ONE ion go in one group -- one --rival MZ=F1,F2,... or one CSV "
            "group --; only if they are two ions, type each --rival its own m/z") in msg
    assert "a CSV group's shared mz cell" in msg


def test_the_rival_spec_checks_the_mz_against_the_rivals_span():
    """Rivals 22 ppm apart (a CO / N2 swap, 11.2 mDa) under --rival-ppm 25:
    the m/z typed between them is no typo."""
    spec = "501.2750=C25H41O10+,C24H41N2O9+"
    typed, forms = AR.parse_rival_spec(spec, check_ppm=10, check_mda=5)
    assert forms == ["C25H41O10+", "C24H41N2O9+"]
    AR.check_group(_group(*forms), tol_ppm=5, tol_mda=0.3, rival_ppm=25)
    with pytest.raises(ValueError, match="a typo"):
        AR.parse_rival_spec("501.30=C25H41O10+,C24H41N2O9+", check_ppm=10, check_mda=5)


def test_a_window_no_named_ion_matches_is_not_arbitrated():
    """Its precursor 25 ppm from every named ion (another, unnamed ion): no
    group's window, skipped as in the plain run, said."""
    p = _mz(SIL) * (1 + 25e-6)
    t = _synthetic(p, _steps(p, SIL_PRODUCTS))
    res = AN.analyze_run(t, _ions(SIL, ORG), rivals=[_group(SIL, ORG)])
    assert [(w.status, w.reason) for w in res.windows] == [("skipped", AN.SKIP_NO_TARGET)]
    assert not len(res.arbitration) and not len(res.breakdown)
    assert any("not arbitrated" in n for n in res.notes)


#: the synthetic C10H20^N+ adduct's products (its ^NH3 loss, skeletal ones, a
#: ^N-bearing one)
ADDUCT_PRODUCTS = {"C10H17+": 300.0, "C6H9+": 100.0, "C7H11+": 80.0, "C5H10^N+": 50.0,
                   "C8H13+": 40.0}


def test_the_rivals_reads_and_columns_on_a_synthetic_adduct():
    """C10H20^N+ against C9H19N2+ 11.5 ppm below it: the other rival's reads
    hold the window's other named ions, never the chosen rival (no
    co-isolate of itself); every column pinned."""
    P, R1 = "C10H20^N+", "C9H19N2+"
    t = _synthetic(_mz(P), _steps(P, ADDUCT_PRODUCTS))
    res = AN.analyze_run(t, _ions(P, R1), rivals=[_group(P, R1)], rival_ppm=12)
    a = res.arbitration.set_index("rival")
    assert a.loc[P, "explained"] == pytest.approx(1.0)
    # the diamine reads nothing on its own calibration; on C10H20^N+'s it
    # reads the skeletal products, both its N in an unnamed remainder:
    # contradicted. The ^NH3 loss and the ^N-bearing product are C10H20^N+'s own
    assert a.loc[R1, "explained"] == pytest.approx(220 / 570)
    assert (a.loc[R1, "read_on"], a.loc[R1, "signature"]) == (P, "N 0%")
    assert (a.loc[P, "verdict"], a.loc[R1, "verdict"]) == (AR.SUPPORTED, AR.CONTRADICTED)
    assert (a.loc[P, "n_products"], a.loc[P, "n_diagnostic"]) == (5, 2)
    assert a.loc[P, "diagnostic_products"] == "C10H17+ (-H3^N) 52.6%; C5H10^N+ (-C5H10) 8.8%"
    assert (a.loc[P, "chance"], a.loc[P, "chance_max"]) == (0.0, 0.0)
    assert a.loc[R1, "chance"] == pytest.approx(0.085526, abs=1e-6)
    assert a.loc[R1, "chance_max"] == pytest.approx(0.596491, abs=1e-6)
    assert a.loc[R1, "diagnostic_decoy"] == 0.375
    assert a.loc[P, "named_share"] == pytest.approx(0.526316, abs=1e-6)
    assert a.loc[P, "precursor_ppm"] == pytest.approx(0.0, abs=1e-6)
    assert a.loc[R1, "precursor_ppm"] == pytest.approx(11.510607, abs=1e-6)
    assert a.loc[P, "signature"] == "^N 100%"
    # named the other way round (the diamine's window skipped in the tables):
    # the adduct's read holds no diamine either -- the same scores
    res = AN.analyze_run(t, _ions(R1, P), rivals=[_group(R1, P)], rival_ppm=12)
    b = res.arbitration.set_index("rival")
    for col in ("verdict", "explained", "chance", "chance_max", "n_diagnostic",
                "diagnostic_products", "named_share", "signature"):
        assert b[col].equals(a.loc[b.index, col]), col


def test_precursor_and_ms1_errors_read_through_the_axis_offset():
    t = _with_ms1(_synthetic(_mz(SIL), _steps(SIL, SIL_PRODUCTS)), [(_mz(SIL), 1.0, 50.0)])
    res = AN.analyze_run(t, _ions(SIL, ORG), rivals=[_group(SIL, ORG)], mass_offset_ppm=3.0)
    a = res.arbitration.set_index("rival")
    gap = (_mz(SIL) - _mz(ORG)) / _mz(ORG) * 1e6
    for col in ("precursor_ppm", "ms1_ppm"):
        assert a.loc[SIL, col] == pytest.approx(-3.0, abs=1e-3), col
        assert a.loc[ORG, col] == pytest.approx(gap - 3.0, abs=1e-3), col


def test_the_polyoxy_rule_holds_only_c_h_n_o():
    assert AR.implausible_remainder("H2O6", 1)               # H2O2 + two O2: a chain
    assert not AR.implausible_remainder("CH2O4", 1)          # CO2 + H2O2, undone
    assert not AR.implausible_remainder("H2O4", 1)           # H2O2 + one O2
    assert not AR.implausible_remainder("H4O4Si", 1)         # Si or S: a molecule's
    assert not AR.implausible_remainder("H2O5S", 1)


def test_picks_keep_the_first_rival_when_windows_disagree():
    g = _group(SIL, ORG)

    def window(p, winner):
        w = AN.WindowResult(parent_mz=p, status="analysed")
        w.rival_group = "g"
        w.arbitration = AR.frame([{"parent_mz": p, "group": "g", "rival": SIL,
                                   "winner": winner, "window_verdict": AR.SUPPORTED}])
        return w
    picks, notes = AN._picks([window(1.0, ORG), window(2.0, SIL)], [g])
    assert picks == {"g": SIL}
    assert notes == [f"rival group 'g': its windows support different rivals ({ORG}, {SIL}): "
                     f"--pick-supported keeps {SIL}, named first"]
    picks, notes = AN._picks([window(1.0, ORG), window(2.0, "")], [g])
    assert picks == {"g": ORG} and "reads every table with C10H20^NO10+" in notes[0]


def test_a_harmonic_ghost_is_no_product():
    """The precursor's 3f ghost (m/z / 9, 2 % of it) is read as no formula:
    no product of the arbitration, kept in place in the decoys."""
    steps = [(e, {**pk, _mz(SIL) / 9.0: 0.02 * pk[SIL]}) for e, pk in _steps(SIL, SIL_PRODUCTS)]
    res, a = _arbitrate(steps, [SIL, ORG])
    assert res.windows[0].harmonics
    plain = _arbitrate(_steps(SIL, SIL_PRODUCTS), [SIL, ORG])[1]
    for col in ("verdict", "explained", "chance", "product_intensity", "n_diagnostic"):
        assert a.loc[SIL, col] == plain.loc[SIL, col], col


def test_the_docstrings_measured_mz():
    assert f"({_mz('C8H25O4Si4+'):.3f})" in __doc__


def test_blank_mz_cells_name_no_rivals():
    df = pd.DataFrame({"formula": ["C10H17+", "C10H19O+", "C6H9+"], "mz": ["", "", ""]},
                      dtype=str)
    assert AR.read_rival_groups(df, "ions.csv") == []


def test_cli_rival_merges_a_formula_the_ions_csv_names(tmp_path, no_server):
    """--rival on a formula the --ions CSV already names (with its adduct
    cell): one ion, a rival of the group -- not 'given twice'."""
    run = _synthetic_run(tmp_path / "run")
    ions = tmp_path / "ions.csv"
    ions.write_text(f"formula,mz,adduct\n{SIL},315.106,[M+^NH4]+\n")
    out = tmp_path / "out"
    assert cli.main(["ms2", "analyze", "--run-dir", str(run), "--ions", str(ions), "--rival",
                     f"315.106={SIL},{ORG}", "--no-figures", "--out-dir", str(out)]) == 0
    a = pd.read_csv(out / "ms2_arbitration.csv", keep_default_na=False, na_values=[""])
    assert set(a["rival"]) == {SIL, ORG}
    man = json.loads((out / "ms2_analyze_manifest.json").read_text())
    assert {i["formula"]: i["adduct"] for i in man["parameters"]["ions"]}[SIL] == "[M+^NH4]+"
    assert any("named by --ion / --ions and by --rival" in n for n in man["notes"])
    # two --rival groups sharing a formula: refused, said
    with pytest.raises(SystemExit, match="is a rival in two groups"):
        cli.main(["ms2", "analyze", "--run-dir", str(run), "--no-figures", "--rival",
                  f"315.106={SIL},{ORG}", "--rival", f"315.107={SIL},C14H19O8+"])


def test_cli_prints_an_undecided_window(tmp_path, no_server, capsys):
    run = _write_tables(tmp_path / "run", _synthetic(
        _mz(SIL), _steps(SIL, {"C8H25O4Si4+": 200.0, "C7H21O4Si4+": 60.0})))
    out = tmp_path / "out"
    assert cli.main(["ms2", "analyze", "--run-dir", str(run), "--rival", f"315.106={SIL},{ORG}",
                     "--no-figures", "--out-dir", str(out)]) == 0
    assert "  rivals [315.106]: undecided -- C8H28^NO4Si4+ undecided" in capsys.readouterr().out
    man = json.loads((out / "ms2_analyze_manifest.json").read_text())
    assert man["counts"]["arbitration"] == {"windows": 1, "supported": 0, "undecided": 1,
                                            "rivals": {"supported": 0, "outscored": 0,
                                                       "contradicted": 0, "undecided": 2}}


def test_cli_never_folds_a_rival_groups_window(tmp_path, no_server, capsys, monkeypatch):
    """A rival group's window the tables skip (its first rival beyond the
    target match) is printed with its arbitration line, never summed up
    with the windows no named ion matches; the run reads nothing: exit 1."""
    monkeypatch.setattr(cli, "MS2_ANALYZE_UNNAMED_SHOWN", 0)
    run = _write_tables(tmp_path / "run", _far_first())
    with pytest.raises(SystemExit):
        cli.main(["ms2", "analyze", "--run-dir", str(run), "--rival",
                  f"315.106=C14H19O8+,{SIL}", "--no-figures", "--out-dir", str(tmp_path / "o")])
    out = capsys.readouterr().out
    assert "[ms2] parent 315.106" in out and ": skipped\n  rivals [315.106]" in out
    assert (f"  rivals [315.106]: C14H19O8+ outscored" in out
            and "the tables read none of them (see the notes)" in out)
    assert "parents skipped: no named ion" not in out


# ------------------------------------------- merged losses, credited ----
#: the O-rich ^NH4 adduct's own products: its ^NH3 loss, and the same with
#: CO2 + H2O2, CO2 + O2 and CO2 + H2O + O2 -- printed as ONE unnamed piece
#: each (CH5^NO4, CH3^NO4, CH5^NO5), the fewest pieces
ORG_OWN = {"C10H17O10+": 160.0, "C9H15O6+": 120.0, "C9H17O6+": 80.0, "C9H15O5+": 60.0}


def test_a_reagent_neutral_merged_with_common_neutrals_is_credible():
    """The decomposer prints the fewest pieces: the reagent's neutral lost
    with CO2, H2O, H2O2 or an O2 prints as one unnamed piece (^NH3 + CO2 +
    H2O2 as CH5^NO4, H^NO3 + CO2 as CH^NO5). Undone -- the neutral of the
    ion's own reagent out, then common neutrals -- it is no oxygen chain.
    Two O2 or more stay one, and the reagent follows the ion's polarity."""
    for rem, charge in (("CH5^NO4", 1), ("CH3^NO4", 1), ("CH5^NO5", 1), ("H3^NO4", -1),
                        ("CH^NO5", -1), ("CH3^NO6", -1), ("H3NO4", -1), ("H2N^NO6", -1),
                        ("CH2O4", 0), ("CH5NO4", 1)):
        assert not AR.implausible_remainder(rem, 1, charge=charge), rem
    for rem, charge in (("CH5^NO7", 1), ("H3^NO4", 1), ("CH5^NO7", 0), ("H2O6", 0),
                        ("H4O10", -1), ("H4O9", 0), ("H3NO9", 1)):
        assert AR.implausible_remainder(rem, 1, charge=charge), rem


def test_the_true_o_rich_adduct_is_credited_with_its_merged_losses():
    """The O-rich adduct is the true ion; the siloxane reads every one of its
    products as an Si2 / CO3H-4 alias (0.41 mDa) through Si-bearing
    remainders. Credited with its own merged losses, the O-rich rival reads
    the same products: undecided -- never the siloxane supported, nor read
    by --pick-supported."""
    for order in ([ORG, SIL], [SIL, ORG]):
        res, a = _arbitrate(_steps(ORG, ORG_OWN), order, pick_supported=True)
        assert set(a["verdict"]) == {AR.UNDECIDED}, order
        assert a.loc[ORG, "explained"] == pytest.approx(1.0)
        assert a.loc[ORG, "uncredited_share"] == 0.0 and a.loc[SIL, "n_diagnostic"] == 0
        assert "reads the same products as" in a.loc[order[1], "evidence"]
        assert next(w for w in res.windows if w.rival_group == "g").target == order[0]
    # without an O2 loss: ^NH3, then with CO2 + H2O2, + H2O, + CO
    own = {"C10H17O10+": 100.0, "C9H15O6+": 80.0, "C9H13O5+": 60.0, "C8H13O4+": 60.0}
    _, a = _arbitrate(_steps(ORG, own), [ORG, SIL])
    assert set(a["verdict"]) == {AR.UNDECIDED} and a.loc[ORG, "uncredited_share"] == 0.0


def test_a_nitrate_clusters_merged_losses_are_credited():
    """A ^NO3- cluster of a keto-acid sheds H^NO3, and with it H2O and CO2
    ([M-H-H2O]-, [M-H-CO2]-: H3^NO4, CH^NO5 as one piece): credited. Its
    C2F3 / H2^NO4 alias reads two of them as Si-free, F-free fragments --
    contradicted; the cluster supported."""
    A, B = "C10H16^NO9-", "C12H14F3O5-"
    prods = {"^NO3-": 150.0, "C10H15O6-": 150.0, "C10H13O5-": 250.0, "C9H15O4-": 300.0,
             "C9H13O3-": 150.0}
    _, a = _arbitrate(_steps(A, prods), [A, B])
    assert (a.loc[A, "verdict"], a.loc[B, "verdict"]) == (AR.SUPPORTED, AR.CONTRADICTED)
    assert a.loc[A, "explained"] == pytest.approx(1.0) and a.loc[A, "uncredited_share"] == 0.0


def _coiso_steps(products, r, co_products=True):
    """Mass-balanced: the siloxane adduct and the named co-isolate CO, `r`
    times as abundant, each losing a quarter of its precursor per step into
    its own products (`products`: CO's, as shares of its loss)."""
    sil = {f: v / sum(SIL_PRODUCTS.values()) for f, v in SIL_PRODUCTS.items()}
    out = []
    for k, e in enumerate(NCE):
        lost = 250.0 * k
        pk = {SIL: 1000.0 - lost, CO: r * (1000.0 - lost)}
        if k:
            pk.update({f: lost * v for f, v in sil.items()})
            if co_products:
                for f, v in products.items():
                    pk[f] = pk.get(f, 0.0) + r * lost * v
        out.append((e, pk))
    return out


def test_a_co_isolates_product_a_rival_reads_only_as_an_oxygen_chain_is_no_product():
    """The O-rich rival reads the named co-isolate's C9 products only through
    a carbon-poor polyoxy remainder (C9H13O2+ as H2O + CH5^NO7): it cannot
    make them, and the attribution's share it has there is none -- they are
    the co-isolate's, no products, kept in place in the decoys. The
    siloxane, twice outweighed by the co-isolate, is supported as without
    them."""
    c9 = {"C9H13O2+": 0.5, "C9H11O+": 0.5}
    plain = None
    for co in (False, True):
        t = _synthetic(_mz(SIL), _coiso_steps(c9, 2.0, co))
        res = AN.analyze_run(t, _ions(SIL, ORG, CO), rivals=[_group(SIL, ORG)])
        assert res.windows[0].co_isolates == [CO]
        a = res.arbitration.set_index("rival")
        assert (a.loc[SIL, "verdict"], a.loc[ORG, "verdict"]) == (AR.SUPPORTED, AR.OUTSCORED)
        if plain is None:
            plain = a
            continue
        for col in ("explained", "chance", "lead", "n_diagnostic", "product_intensity"):
            assert a[col].equals(plain[col]), col


# ------------------------------------------------- a ring S as CS ----
def test_a_thiazoles_s_leaving_as_cs_is_shown():
    """Protonated benzothiazole loses HCN (C6H5S+), then CS at higher
    energies (C5H5+). CS holds no H, so it is no unnamed piece of its own:
    HCN + CS prints as ONE unnamed C2HNS, its S in it. All of it leaves as
    CS next to a library neutral: shown -- the true formula is not
    contradicted on its own S when C5H5+ dominates. An alias read through a
    mixed remainder one library neutral does not complete stays as printed."""
    assert AR.shows("S", "C5H5+", "C2HNS", 1)
    assert not AR.shows("S", "C8H13O+", "H2O + H4O2SSi", 1)
    assert not AR.shows("S", "C5H5+", "C2H2S", 1)              # CS + C... no neutral
    T_, R_ = "C7H6NS+", "C6H6NOSi+"
    steps = [(e, {T_: 1000.0 * (1 - 0.22 * k), "C6H5S+": s, "C5H5+": c})
             for k, (e, s, c) in enumerate(zip((10, 20, 35, 50), (60, 70, 50, 25),
                                                (5, 60, 160, 260)))]
    _, a = _arbitrate(steps, [T_, R_])
    assert a.loc[T_, "verdict"] != AR.CONTRADICTED and a.loc[T_, "signature"] == "S 100%"
    assert a.loc[R_, "verdict"] == AR.CONTRADICTED


# ------------------------------------- a skipped group window's advice ----
def _far_mirror():
    """C14H19O8+ the true ion, the folder's axis 10.5 ppm low: the located
    precursor matches the siloxane (6 ppm), not C14H19O8+ (10.5 ppm)."""
    prods = {"C14H17O7+": 200.0, "C13H19O6+": 150.0, "C12H17O5+": 80.0, "C11H13O4+": 60.0,
             "C9H11O3+": 50.0}
    return _synthetic(_mz("C14H19O8+"), _moved(_steps("C14H19O8+", prods), -10.5))


def test_cli_a_skipped_group_window_points_to_the_rival_that_reads_it(tmp_path, no_server):
    """The first rival beyond the target match, the second matching the
    located precursor: an axis offset is the remedy only where the
    arbitration supports the first. Where it supports the matching one,
    --pick-supported is (the offset would read the outscored rival)."""
    spec = f"315.106=C14H19O8+,{SIL}"
    run = _write_tables(tmp_path / "run", _far_first())
    with pytest.raises(SystemExit) as e:
        cli.main(["ms2", "analyze", "--run-dir", str(run), "--rival", spec, "--no-figures",
                  "--out-dir", str(tmp_path / "o")])
    msg = str(e.value.code)
    assert "--pick-supported" in msg and "--mass-offset-ppm" not in msg, msg
    assert f"supports {SIL}" in msg
    win = json.loads((tmp_path / "o" / "ms2_analyze_manifest.json").read_text())["windows"][0]
    assert win["offset_hint_ppm"] is None
    assert not any("no named ion matches the precursor" in n for n in win["notes"])
    assert cli.main(["ms2", "analyze", "--run-dir", str(run), "--rival", spec, "--no-figures",
                     "--pick-supported", "--out-dir", str(tmp_path / "p")]) == 0
    # the mirror: the first rival supported -- the axis offset stands, and works
    run = _write_tables(tmp_path / "mirror", _far_mirror())
    with pytest.raises(SystemExit) as e:
        cli.main(["ms2", "analyze", "--run-dir", str(run), "--rival", spec, "--no-figures",
                  "--out-dir", str(tmp_path / "m")])
    assert "pass --mass-offset-ppm -10.5" in str(e.value.code)
    assert cli.main(["ms2", "analyze", "--run-dir", str(run), "--rival", spec, "--no-figures",
                     "--mass-offset-ppm", "-10.5", "--out-dir", str(tmp_path / "n")]) == 0


def test_a_skipped_undecided_group_window_says_both_remedies():
    """Aliases only, read 6 ppm low, C14H19O8+ named first: the precursor
    matches the O-rich rival, the arbitration supports none -- naming the
    matching rival first, or the axis offset that reads the first, is said;
    no offset hint of its own."""
    t = _synthetic(_mz(SIL), _moved(_steps(SIL, {"C8H25O4Si4+": 200.0, "C7H21O4Si4+": 60.0}),
                                    -6.0))
    res = AN.analyze_run(t, _ions("C14H19O8+", SIL, ORG), rivals=[_group("C14H19O8+", SIL, ORG)],
                         rival_ppm=8)
    w = res.windows[0]
    assert (w.status, w.reason) == ("skipped", AN.SKIP_NO_TARGET)
    assert w.arbitration["winner"].iloc[0] == ""
    assert math.isnan(w.offset_hint_ppm)
    assert w.rival_hint.endswith(f"supports none of the rivals: name {ORG} first to read the "
                                 "tables with the rival the precursor matches, or -- if the "
                                 "folder's m/z axis sits off its formulas -- pass "
                                 "--mass-offset-ppm -10.5 to read C14H19O8+")
    assert w.manifest()["arbitration"]["hint"] == w.rival_hint


# ------------------------------------------------ the diluted window ----
def test_a_leader_far_above_its_decoys_on_too_little_is_said_so():
    """Product peaks no rival reads (an unnamed ion's products, noise) dilute
    every share: the siloxane explains 35 % of the products against its
    decoys' 2 % -- short of EXCESS_MIN beyond them, but no chance agreement:
    said as too little of the products, not as chance."""
    noise = {101.5: 300.0, 77.3: 200.0, 143.45: 200.0}
    _, a = _arbitrate(_steps(SIL, {**SIL_PRODUCTS, **noise}), [SIL, ORG])
    assert set(a["verdict"]) == {AR.UNDECIDED}
    ev = a.loc[SIL, "evidence"]
    assert "no better than chance" not in ev and "too little of the products" in ev, ev
    assert a.loc[SIL, "explained"] == pytest.approx(0.352, abs=1e-3)
    _, a = _arbitrate(_steps(SIL, {**SIL_PRODUCTS, **{k: v / 2 for k, v in noise.items()}}),
                      [SIL, ORG])
    assert a.loc[SIL, "verdict"] == AR.SUPPORTED


# ------------------------------------------ decoys beyond the tolerance ----
#: the ^NH4 adduct of a C20 cyclic siloxane, its ^NH3 loss at m/z 741, and
#: an Si-free rival 1.9 ppm off
S10, S10_RIVAL = "C20H64^NO10Si10+", "C43H36^NO12+"
S10_PRODUCTS = {"C20H61O10Si10+": 600.0, "C3H9Si+": 40.0, "C7H21O3Si3+": 30.0,
                "C2H7OSi+": 20.0}


def test_a_decoy_shift_never_stays_inside_the_reading_tolerance():
    """At --tol-ppm 10 a 7 mDa shift of the m/z 741 product (9.4 ppm) stays
    inside its tolerance: that decoy read the real product as itself, a
    chance that is none. Each product's shifts are scaled so the smallest is
    twice its tolerance: the siloxane supported, as at 5 ppm."""
    t = _synthetic(_mz(S10), _steps(S10, S10_PRODUCTS))
    small = AN._window_tables(t, float(t.parents["parent_mz"].iloc[0]))
    spec = S.spectrum_key(_mz(S10), "hcd10.00")
    rows = small.spectra[small.spectra["spectrum"] == spec].reset_index(drop=True)
    out = AN._shifted(small, [spec], set(), 0.007, tol_ppm=10.0, tol_mda=0.3)
    moved = out.spectra[out.spectra["spectrum"] == spec].reset_index(drop=True)
    d = moved["mz"].to_numpy() - rows["mz"].to_numpy()
    tol = np.maximum(10e-6 * rows["mz"].to_numpy(), 0.3e-3)
    assert (d >= AR.DECOY_TOL_FACTOR * tol - 1e-12).all() and (d >= 0.007 - 1e-12).all()
    big = AR.DECOY_TOL_FACTOR * 10e-6 * rows["mz"].to_numpy() > 0.007     # above m/z 350
    assert big.any() and np.allclose(d[~big], 0.007)
    for tol_ppm in (5.0, 10.0):
        _, a = _arbitrate(_steps(S10, S10_PRODUCTS), [S10, S10_RIVAL], tol_ppm=tol_ppm)
        assert a.loc[S10, "verdict"] == AR.SUPPORTED, tol_ppm


# ------------------------------------------------ missing-value cells ----
def test_missing_value_cells_name_no_rivals():
    """An m/z cell 'nan' is no m/z (as the ion reader takes it), and R's NA in
    the group column no group: neither names rivals."""
    df = pd.DataFrame({"formula": ["C10H17+", "C10H20^N+", "C10H19O+"],
                       "mz": ["nan", "nan", "NaN"]}, dtype=str)
    assert AR.read_rival_groups(df, "ions.csv") == []
    df = pd.DataFrame({"formula": ["C10H17+", "C10H20^N+", "C9H19N2+", "C10H19O+"],
                       "mz": ["137.13", "155.156", "155.156", "155.15"],
                       "group": ["NA", "a", "a", "NA"]}, dtype=str)
    assert [(g.name, g.rivals) for g in AR.read_rival_groups(df, "ions.csv")] == [
        ("a", ("C10H20^N+", "C9H19N2+"))]


def test_the_rival_ppm_hint_only_where_it_can_help():
    with pytest.raises(ValueError) as e:
        AR.check_group(_group("C10H17+", "C10H19O+"), tol_ppm=5, tol_mda=0.3, rival_ppm=12)
    assert "pass --rival-ppm" not in str(e.value)               # 116000 ppm: past the cap
    with pytest.raises(ValueError, match="pass --rival-ppm 7.1 or more"):
        AR.check_group(_group(ORG, "C14H19O8+"), tol_ppm=5, tol_mda=0.3)


def test_cli_an_ions_csv_with_nan_mz_cells_runs_as_before(tmp_path, no_server):
    run = _write_tables(tmp_path / "run", _tables("ms2_monoterpene_a"))
    ions = tmp_path / "ions.csv"
    ions.write_text("formula,mz\nC10H17+,nan\nC10H20^N+,nan\nC10H19O+,nan\n")
    assert cli.main(["ms2", "analyze", "--run-dir", str(run), "--ions", str(ions),
                     "--no-figures", "--out-dir", str(tmp_path / "o")]) == 0
    assert not (tmp_path / "o" / "ms2_arbitration.csv").exists()


# ------------------------------------------- every documented rule ----
ALIAS2 = "C9H24^NO7Si2+"          # the siloxane with one Si2 swapped for CO3H-4


def test_three_rivals_the_leader_beats_one_and_not_the_other():
    """Three rivals: the siloxane, the O-rich rival (reads its ^NH3 loss
    only) and the siloxane's Si2 / CO3H-4 alias (reads every product). The
    leader beats the O-rich rival -- outscored -- and not its alias: the
    same products, undecided; its lead is over the best other rival."""
    _, a = _arbitrate(_steps(SIL, SIL_PRODUCTS), [SIL, ORG, ALIAS2])
    assert a["verdict"].to_dict() == {SIL: AR.UNDECIDED, ORG: AR.OUTSCORED,
                                      ALIAS2: AR.UNDECIDED}
    assert set(a["window_verdict"]) == {AR.UNDECIDED} and set(a["winner"]) == {""}
    assert a.loc[SIL, "evidence"].endswith(f"undecided: against {ALIAS2}: reads the same "
                                           f"products as {SIL}")
    assert (a.loc[SIL, "lead"], a.loc[ALIAS2, "lead"]) == (0.0, 0.0)
    assert a.loc[ORG, "lead"] == pytest.approx(200 / 380 - 1.0)
    # constructed: A beats B, not C (which reads A's products)
    own = [(f"P{i}", 0.3, True, {"A", "C"}) for i in range(3)]
    real = {"A": _sc(0.9, own), "B": _sc(0.2), "C": _sc(0.85, own)}
    win, v, why = _decide(real)
    assert (win, v) == ("", {"A": AR.UNDECIDED, "B": AR.OUTSCORED, "C": AR.UNDECIDED})
    assert why["A"].startswith("against C: A leads it by 5%")


def test_the_chance_is_the_decoys_mean():
    """One decoy explaining 0.7 among seven at none: a chance of 0.0875 --
    the mean over eight shifts, never their largest."""
    own = [(f"P{i}", 0.2, True, {"A"}) for i in range(3)]
    real = {"A": _sc(0.9, own), "B": _sc(0.2)}
    decoys = [{"A": _sc(0.7), "B": _sc(0.0)}] + [{"A": _sc(0.0), "B": _sc(0.0)}] * 7
    assert _decide(real, decoys)[:2] == ("A", {"A": AR.SUPPORTED, "B": AR.OUTSCORED})


def test_an_ms1_centroid_beyond_the_target_match_is_no_precursor():
    """The MS1 window holds one centroid, 0.106 Da off the precursor (S/N
    20): no ms1_ppm for any rival, and the note says MS1 sees no precursor
    there, the unnamed centroid holding all of the window."""
    t = _synthetic(_mz(SIL), _steps(SIL, SIL_PRODUCTS))
    res = AN.analyze_run(_with_ms1(t, [(315.0, 1.0, 20.0)]), _ions(SIL, ORG),
                         rivals=[_group(SIL, ORG)])
    a = res.arbitration.set_index("rival")
    assert a["ms1_ppm"].isna().all()
    assert set(a["note"]) == {"the MS1 window holds no centroid at the precursor, and one no "
                              "named ion matches, at m/z 315.0000, holds 100% of it: the products "
                              "may be another ion's, read through each rival's formula"}


def test_a_product_a_co_isolate_shares_with_one_rival_stays_a_product():
    """A named Si3 co-isolate (315.111) shares C3H9Si+ with the siloxane: the
    O-rich rival's read gives it wholly to the co-isolate, the siloxane's
    does not -- a product (no rival's read gives it away wholly: every
    rival's must)."""
    X = "C9H27O6Si3+"
    steps = [(e, {**pk, X: pk[SIL], **({"C3H9Si+": pk["C3H9Si+"] * 2} if "C3H9Si+" in pk
                                        else {})})
             for e, pk in _steps(SIL, SIL_PRODUCTS)]
    res = AN.analyze_run(_synthetic(_mz(SIL), steps), _ions(SIL, ORG, X),
                         rivals=[_group(SIL, ORG)])
    assert res.windows[0].co_isolates == [X]
    a = res.arbitration.set_index("rival")
    assert a.loc[SIL, "product_intensity"] == 2640.0 and a.loc[SIL, "n_products"] == 5
    assert a.loc[SIL, "explained"] == pytest.approx(1920 / 2640)


def test_a_window_whose_precursor_matches_a_named_ion_of_no_group_is_not_arbitrated():
    """The located precursor sits on the siloxane, a named ion of no group;
    the group's rivals -- one of them within the target match -- are not the
    nearest named ion: no arbitration row, said."""
    t = _synthetic(_mz(SIL), _steps(SIL, SIL_PRODUCTS))
    res = AN.analyze_run(t, _ions(SIL, ORG, "C14H19O8+"), rivals=[_group(ORG, "C14H19O8+")],
                         rival_ppm=8)
    assert res.windows[0].target == SIL and not len(res.arbitration)
    assert any("rival group 'g'" in n and "not arbitrated" in n for n in res.notes)


def test_the_ms2_window_note_weighs_the_rivals_precursor_only():
    """No MS1 row: in the lowest-energy MS2 spectrum an unnamed peak twice
    the siloxane's precursor, a named co-isolate's precursor three times it.
    The co-isolate's precursor is no rival's: the unnamed peak outweighs the
    precursor, said."""
    steps = [(e, {**pk, CO: 3 * pk[SIL], 315.0: 2 * pk[SIL]})
             for e, pk in _steps(SIL, SIL_PRODUCTS)]
    res = AN.analyze_run(_synthetic(_mz(SIL), steps), _ions(SIL, ORG, CO),
                         rivals=[_group(SIL, ORG)])
    assert res.windows[0].co_isolates == [CO]
    note = res.arbitration["note"].iloc[0]
    assert "hold 33% of the isolation window, more than the precursor (17%)" in note


def test_the_small_rules():
    # a reagent N leaving whole: the rest must hold none of it
    assert not AR.shows("N", "C6H9+", "C5H14N2", 1)
    # --rival-ppm widens the reading tolerance, never narrows it
    AR.check_group(_group(SIL, ORG), tol_ppm=5, tol_mda=0.3, rival_ppm=1)
    # the m/z typed is checked against the rivals' span widened by the larger
    # of check_ppm and check_mda
    assert AR.parse_rival_spec(f"315.110={SIL},{ORG}", check_ppm=10, check_mda=5)[1] == [SIL, ORG]
    # a radical loss is no named closed-shell loss
    assert AR._named("H2O", 0, False) and not AR._named("CH3", 0, True)


# ------------------------------------- every rival on every calibration ----
#: a composition tie, 5.0 ppm apart: four products both formulas hold, two
#: only the first can make (O4), two only the second (N2)
TIE_A, TIE_B = "C14H24NO4+", "C12H22N4O3+"
TIE = {"C8H13O2+": 300.0, "C7H11O+": 150.0, "C6H9O+": 100.0, "C9H15O2+": 80.0,
       "C4H7O4+": 25.0, "C3H5O4+": 20.0, "C5H9N2O+": 25.0, "C4H7N2+": 20.0}
#: the same, 11.5 ppm apart (--rival-ppm): shared hydrocarbon products, two
#: ^N products only the adduct can make, two N2 products only the diamine
WIDE_A, WIDE_B = "C10H20^N+", "C9H19N2+"
WIDE = {"C7H11+": 300.0, "C6H9+": 150.0, "C5H9+": 100.0, "C7H13+": 80.0,
        "C5H10^N+": 30.0, "C4H8^N+": 25.0, "C4H9N2+": 30.0, "C3H7N2+": 25.0}


def test_a_composition_tie_is_undecided_wherever_the_precursor_sits():
    """Every rival is read on every rival's calibration and scored on the one
    it explains the most on: where the precursor sits between the rivals --
    an MS1 matter -- decides nothing. Two products of its own on either side,
    the rest shared: undecided with the precursor on either rival and every
    product 1 ppm low, on it or 1 ppm high (on its own calibration alone the
    rival the precursor sits off read its own products 6 ppm off, lost them,
    and the other was supported 'on products they cannot read'). 11.5 ppm
    apart, the far rival reads nothing on its own calibration: the same."""
    for err in (-1.0, 0.0, 1.0):
        for on in (TIE_A, TIE_B):
            steps = _moved(_steps(on, TIE), err, keep={on})
            res = AN.analyze_run(_synthetic(round(_mz(on), 4), steps), _ions(TIE_A, TIE_B),
                                 rivals=[_group(TIE_A, TIE_B)])
            a = res.arbitration.set_index("rival")
            assert set(a["verdict"]) == {AR.UNDECIDED}, (err, on)
            assert (a["n_diagnostic"] == 2).all(), (err, on)
            assert a.loc[TIE_A, "explained"] == pytest.approx(a.loc[TIE_B, "explained"])
    for on in (WIDE_A, WIDE_B):
        res = AN.analyze_run(_synthetic(round(_mz(on), 4), _steps(on, WIDE)),
                             _ions(WIDE_A, WIDE_B), rivals=[_group(WIDE_A, WIDE_B)],
                             rival_ppm=12)
        a = res.arbitration.set_index("rival")
        assert set(a["verdict"]) == {AR.UNDECIDED}, on
        far = WIDE_B if on == WIDE_A else WIDE_A
        assert a.loc[far, "read_on"] == on and a.loc[on, "read_on"] == on
        assert (a["n_diagnostic"] == 2).all(), on
        assert f"read on {on}'s calibration" in a.loc[far, "evidence"]


# ------------------------------------------- merged common neutrals ----
#: a protonated O-rich acid and its siloxane alias (Si2 for CO3H-4 twice,
#: 2.8 ppm): its water and CO2 losses, CO2 + 2 H2O (printed as ONE unnamed
#: CH4O4) and that with one more H2O
ACID, ACID_SIL = "C10H17O10+", "C8H25O4Si4+"
ACID_PRODUCTS = {"C10H15O9+": 200.0, "C9H17O8+": 150.0, "C9H15O7+": 120.0, "C10H13O8+": 80.0,
                 "C9H13O6+": 150.0, "C9H11O5+": 120.0}


def test_co2_lost_with_two_waters_is_no_oxygen_chain():
    """CO2 + 2 H2O is three common neutrals: the decomposer prints it as ONE
    unnamed CH4O4. Refused, a true O-rich acid lost those products to its
    siloxane alias (read through Si-H remainders), which was supported --
    and read by --pick-supported. Credited, the two read the same products.
    The same for a nitrate cluster's [M-H-CO2-2H2O]- (H^NO3 + CO2 + 2 H2O,
    one CH5^NO7) against its Si2 alias."""
    for rem, charge in (("CH4O4", 1), ("CH4O4", -1), ("H2O + CH4O4", 1), ("CH4O5", 1),
                        ("CH5^NO7", -1), ("H6O4", 0), ("CH4O5", 0)):
        assert not AR.implausible_remainder(rem, 1, charge=charge), (rem, charge)
    for order in ([ACID, ACID_SIL], [ACID_SIL, ACID]):
        res, a = _arbitrate(_steps(ACID, ACID_PRODUCTS), order, pick_supported=True)
        assert set(a["verdict"]) == {AR.UNDECIDED}, order
        assert a.loc[ACID_SIL, "n_diagnostic"] == 0 and a.loc[ACID, "uncredited_share"] == 0.0
        assert next(w for w in res.windows if w.rival_group == "g").target == order[0]
    N, NS = "C10H16^NO11-", "C9H20^NO8Si2-"
    prods = {"^NO3-": 150.0, "C10H15O8-": 200.0, "C10H13O7-": 120.0, "C9H15O6-": 150.0,
             "C9H13O5-": 100.0, "C9H11O4-": 200.0, "C9H9O3-": 150.0}
    _, a = _arbitrate(_steps(N, prods), [N, NS])
    assert a.loc[NS, "verdict"] != AR.SUPPORTED and a.loc[NS, "n_diagnostic"] == 0
    assert a.loc[N, "uncredited_share"] == 0.0


def test_the_merge_bounds():
    """At most REAGENT_NEUTRALS_MAX (2) reagent neutrals and MERGED_MAX (3)
    common ones, next to one O2 -- on remainders the decomposer prints (an
    integer DBE >= 0, `subformula._remainder_ok`): on an anion two H^NO3 are
    credited, three refused (HNO3 adds no DBE: three print); CO2 + 2 H2O,
    three common neutrals, credited; CO2 + H2O2 with one O2 credited, CO + 2
    H2O2 with two O2 refused. With POLYOXY_MAX_C 1 the DBE rule leaves room
    for at most two of H2O / H2O2 / NH3 and one CO / CO2: four common
    neutrals never print, whatever reagent neutrals or O2 come with them --
    MERGED_MAX 3 is every printable merge, no filter."""
    from collections import Counter
    from itertools import combinations_with_replacement as cwr

    from peaky.chem import chemistry as C
    from peaky.ms2 import attribute as AT
    assert (AR.REAGENT_NEUTRALS_MAX, AR.MERGED_MAX, AT.POLYOXY_MAX_C) == (2, 3, 1)
    for rem, charge, refused in (("H2^N2O6", -1, False),      # 2 H^NO3
                                 ("H3^N3O9", -1, True),       # 3 H^NO3
                                 ("CH4O4", 1, False),         # CO2 + 2 H2O
                                 ("CH2O6", 0, False),         # CO2 + H2O2 + O2
                                 ("CH4O9", 0, True)):         # CO + 2 H2O2 + 2 O2
        assert AR._valid(C.parse_formula(rem)), rem           # printed
        assert AR.implausible_remainder(rem, 1, charge=charge) == refused, (rem, charge)
    common = [Counter(C.parse_formula(n)) for n in AR.MERGED_NEUTRALS]
    for sign, names in AR.REAGENT_NEUTRALS.items():
        reag = [Counter(C.parse_formula(n)) for n in names]
        for k in range(AR.REAGENT_NEUTRALS_MAX + 1):
            for rs in cwr(reag, k):
                for cs in cwr(common, AR.MERGED_MAX + 1):
                    for o2 in (Counter(), Counter({"O": 2})):
                        got = sum((*rs, *cs, o2), Counter())
                        if got["C"] <= AT.POLYOXY_MAX_C:
                            assert not AR._valid(got), (sign, got)


# ------------------------- the signature: O and merged reagent losses ----
#: the ammonium adduct of an O4 acid and its S + Si + H4 alias (0.64 mDa, no O)
O4, O4_ALIAS = "C9H18NO4+", "C9H22NSSi+"


def _o4_steps(true=O4, early=None):
    """Low energies: -NH3 and -NH3-H2O (O shown); high energies, dominating:
    hydrocarbon ions whose losses print the O inside an unnamed skeletal
    piece (C7H9+: H2O + C2H7NO3) -- or, C8H11+'s, in H2O + CH5NO3, the
    reagent's NH3 with CO2 and H2O, a merge `_undone` makes up. `true`: the
    precursor (`early`: its low-energy products)."""
    early = {"C9H15O4+": 100.0, "C9H13O3+": 80.0} if early is None else early
    late = {"C7H9+": 225.0, "C6H9+": 112.5, "C5H7+": 90.0, "C8H11+": 60.0}
    out = []
    for k, e in enumerate((1, 10, 20, 30, 40)):
        pk = {true: 1000.0 * (1 - 0.2 * k) + 1.0}
        if k:
            pk.update({f: v * (1.0 if k <= 2 else 0.6) for f, v in early.items()})
        if k >= 2:
            pk.update({f: v * (k - 1) for f, v in late.items()})
        out.append((e, pk))
    return out


def test_o_lost_in_a_skeletal_piece_is_shown():
    """A skeleton's O leaves in C-O neutrals the decomposer merges with the
    reagent's and an alkene: C7H9+ from the O4 adduct prints its loss as
    H2O + C2H7NO3 (NH3 + C2H4O3) -- all its O in one valid skeletal piece of
    more than one C, its N in one reagent neutral: shown; C8H11+'s O, in H2O
    + CH5NO3, is shown as the reagent's NH3 merged with CO2 and H2O. The
    true O4 adduct is not contradicted on its own O when its high-energy
    hydrocarbon ions dominate (read as the O absent, those pieces left it
    contradicted, O 27 %); its S + Si + H4 alias, which reads the same
    products, is contradicted on S and Si -- supporting nobody: the window
    is undecided. A piece with more N than one reagent neutral of the ion's
    polarity, or more O than its C and one, is no skeletal loss. The
    signature is pooled over the energies: the S + Si + H4 ion built so --
    its -NH3 product at low energies, the same hydrocarbons -- is
    contradicted on its own S and Si (a limit, docs)."""
    assert AR.shows("O", "C7H9+", "H2O + C2H7NO3", 1, charge=1)
    assert AR.shows("O", "C5H7+", "C4H11NO4", 1, charge=1)
    assert AR.shows("O", "C8H11+", "H2O + CH5NO3", 1, charge=1)   # NH3 + CO2 + H2O
    assert not AR.shows("O", "C8H11+", "H2O + CH5NO", 1, charge=1)  # one C, no merge
    assert not AR.shows("O", "C7H9+", "H2O + C2H7NO3", 1)           # no polarity: no NH3
    assert not AR.shows("O", "C6H5-", "C2H2N2O7", 1, charge=-1)     # N2: one HNO3 and more
    assert AR.shows("O", "C6H5-", "C2H5NO3", 1, charge=-1)          # HNO3 + C2H4
    assert not AR.shows("O", "C7H9+", "C2H2O9", 1, charge=1)        # an oxygen chain
    assert not AR.shows("S", "C8H13O+", "H2O + C2H8OS", 1, charge=1)  # S: as printed
    _, a = _arbitrate(_o4_steps(), [O4, O4_ALIAS])
    assert a.loc[O4, "verdict"] != AR.CONTRADICTED, a.loc[O4, "evidence"]
    assert a.loc[O4, "signature"] == "O 100%"
    assert a.loc[O4_ALIAS, "verdict"] == AR.CONTRADICTED
    assert set(a["winner"]) == {""} and set(a["window_verdict"]) == {AR.UNDECIDED}
    _, a = _arbitrate(_o4_steps(O4_ALIAS, {"C9H19SSi+": 100.0}), [O4_ALIAS, O4])
    assert a.loc[O4_ALIAS, "verdict"] == AR.CONTRADICTED and set(a["winner"]) == {""}
    assert a.loc[O4_ALIAS, "signature"] == "S 10%; Si 10%"


def test_a_reagent_lost_with_common_neutrals_is_shown():
    """A credited merge of the reagent's neutral with common ones (H^NO3 +
    CO2 + H2O as ONE unnamed CH3^NO6) shows the reagent's ^N: credited as the
    reagent leaving whole, it is not counted as the reagent absent. The true
    nitrate cluster whose [M-H-CO2-H2O]- dominates is not contradicted on its
    own ^N, in either order; its C2F3 alias is. The ion's polarity holds."""
    assert AR.shows("^N", "C9H13O3-", "CH3^NO6", 1, charge=-1)
    assert AR.shows("^N", "C9H15O4+", "CH5^NO5", 1, charge=1)
    assert not AR.shows("^N", "C9H13O3+", "CH3^NO6", 1, charge=1)
    A, B = "C10H16^NO9-", "C12H14F3O5-"
    prods = {"^NO3-": 20.0, "C10H15O6-": 40.0, "C10H13O5-": 40.0, "C9H15O4-": 60.0,
             "C9H13O3-": 400.0}
    for order in ([A, B], [B, A]):
        _, a = _arbitrate(_steps(A, prods), order)
        assert a.loc[A, "verdict"] != AR.CONTRADICTED, order
        assert a.loc[A, "signature"] == "^N 100%", order
        assert a.loc[B, "verdict"] == AR.CONTRADICTED, order


# ------------------------------------------- the documented rules pinned ----
def test_an_element_another_rival_lacks_is_judged():
    """Three rivals on O-poor organic products: the siloxane and its Si2 alias
    both hold Si, the O-rich rival lacks it -- each is judged on Si (an
    element ANOTHER rival lacks, not one only it holds) and contradicted."""
    _, a = _arbitrate(_steps(SIL, ORG_PRODUCTS), [ORG, SIL, ALIAS2])
    for R in (SIL, ALIAS2):
        assert (a.loc[R, "verdict"], a.loc[R, "elements"], a.loc[R, "signature"]) == (
            AR.CONTRADICTED, "Si", "Si 0%"), R
    assert a.loc[ORG, "elements"] == ""


def test_a_c1_product_free_of_the_element_is_not_judged():
    """A carbon-poor product is at most POLYOXY_MAX_C (one) C: a C1 product
    that does not show F is out of the signature, as a C0 one is."""
    A, B = "C2HF3^NO5-", "C3H5^NO6-"
    rows = [(A, 0, 60.0, "O3^N-", "^N"), (A, 1, 30.0, "C2F3O2-", "F;^N"),
            (A, 2, 80.0, "CHO3-", "")]
    sc, _ = AR.rival_scores(_table(rows), [A, B])
    assert sc[A].signature == {"F": 1.0} and AR._contradicted(sc[A]) == []


def test_a_survivor_above_the_precursor_less_an_h_atom_is_no_product():
    """The ceiling is the precursor less an H atom (the lightest loss): an
    unnamed survivor 0.1 or 0.7 Da below the precursor is no product (an H2
    loss would be 2.016 Da); one 1.5 Da below lies under the ceiling -- a
    product no rival reads, which dilutes every share."""
    def steps(dm):
        return [(e, {**pk, _mz(SIL) - dm: 3000.0}) for e, pk in _steps(SIL, SIL_PRODUCTS)]
    for dm in (0.1, 0.7):
        _, a = _arbitrate(steps(dm), [SIL, ORG])
        assert a.loc[SIL, "verdict"] == AR.SUPPORTED, dm
        assert a.loc[SIL, "product_intensity"] == 2280.0, dm
    _, a = _arbitrate(steps(1.5), [SIL, ORG])
    assert a.loc[SIL, "product_intensity"] == 2280.0 + 4 * 3000.0
    assert a.loc[SIL, "explained"] == pytest.approx(2280.0 / 14280.0)


def test_the_decoys_shift_every_product_by_distinct_amounts(monkeypatch):
    """Through the wiring: at --tol-ppm 10 the m/z 741 product's eight decoy
    shifts are the eight shifts scaled by ONE factor -- distinct, the
    smallest twice its tolerance."""
    got = []
    orig = AN._shifted

    def spy(small, specs, keep, shift_da, **kw):
        out = orig(small, specs, keep, shift_da, **kw)
        got.append((small, out))
        return out
    monkeypatch.setattr(AN, "_shifted", spy)
    _arbitrate(_steps(S10, S10_PRODUCTS), [S10, S10_RIVAL], tol_ppm=10.0)
    spec = S.spectrum_key(_mz(S10), "hcd10.00")
    d = []
    for small, out in got:
        a = small.spectra[small.spectra["spectrum"] == spec].reset_index(drop=True)
        b = out.spectra[out.spectra["spectrum"] == spec].reset_index(drop=True)
        i = int(np.argmin(np.abs(a["mz"].to_numpy() - _mz("C20H61O10Si10+"))))
        d.append(round(float(b["mz"].iloc[i] - a["mz"].iloc[i]) * 1e3, 2))
    assert len(got) == len(AR.DECOY_SHIFTS_MDA)
    assert sorted(d) == [-40.24, -31.77, -23.29, -14.82, 14.82, 23.29, 31.77, 40.24]


@pytest.mark.parametrize("run", [{}, {"ladder": False, "tol_ppm": 4.0, "harmonics": False},
                                 {"k_max": 4}])
def test_every_read_is_the_runs_reading(monkeypatch, run):
    """Every read the arbitration makes -- on every calibration, of the window
    and of each decoy -- has the run's gate, k_max, tolerances and harmonic
    rule, whichever they are (the defaults, each switched off, an explicit
    k_max: a read path that fixed any of them to one value would match one
    setting only), and every rival is read on each other rival's
    calibration, the decoys too."""
    from peaky.ms2 import readings as RD
    calls = []
    orig = RD.read_window

    def spy(*a, **kw):
        calls.append(kw)
        return orig(*a, **kw)
    monkeypatch.setattr(RD, "read_window", spy)
    got = _captured(monkeypatch)
    _arbitrate(_steps(SIL, SIL_PRODUCTS), [SIL, ORG], **run)
    want = (run.get("ladder", True), run.get("k_max"), run.get("tol_ppm", 5.0),
            run.get("harmonics", True))
    # the run's window read (the first rival's own), the other rival's own,
    # each rival's on the other's calibration; on each decoy all four
    assert len(calls) >= 4 + 4 * len(AR.DECOY_SHIFTS_MDA)
    assert all((kw["ladder"], kw["k_max"], kw["tol_ppm"], kw["harmonics"]) == want
               for kw in calls), want
    _, reads, decoys = got[0]
    for rs in (reads, *decoys):
        assert {R: set(rr.cross) for R, rr in rs.items()} == {SIL: {ORG}, ORG: {SIL}}


def test_the_strongest_diagnostic_products_named():
    _, a = _arbitrate(_steps(SIL, SIL_PRODUCTS), [SIL, ORG])
    assert AR.DIAGNOSTIC_SHOWN == 3 and a.loc[SIL, "n_diagnostic"] == 4
    assert a.loc[SIL, "diagnostic_products"].count("; ") == AR.DIAGNOSTIC_SHOWN - 1


def test_an_mz_cell_groups_by_its_text():
    """Rows share an m/z cell by its text: 155.156 and 155.1560 are two cells
    (a typed precision is the user's)."""
    df = pd.DataFrame({"formula": ["C10H20^N+", "C9H19N2+"], "mz": ["155.156", "155.1560"]},
                      dtype=str)
    assert AR.read_rival_groups(df, "ions.csv") == []


def test_the_ms2_window_note_reads_the_isolation_window_only():
    """With no MS1 row, the note weighs the peaks INSIDE the isolation window
    at the lowest energy: products already there (in-source), far below it,
    are no ion co-isolated with the precursor -- no note."""
    steps = [(e, {SIL: 1000.0 * (1 - 0.25 * k), **{f: 5 * v * (k + 1)
                                                    for f, v in SIL_PRODUCTS.items()}})
             for k, e in enumerate(NCE)]
    _, a = _arbitrate(steps, [SIL, ORG])
    assert set(a["note"]) == {""}


def test_the_note_names_a_co_isolate_and_cites_a_centroid_with_no_sn():
    """The note's named ions are the rivals AND the window's co-isolates: a
    co-isolate's MS1 centroid outweighing the precursor is no unnamed ion.
    An unnamed centroid with no S/N (NaN) is no noise: cited."""
    t = _synthetic(_mz(SIL), [(e, {**pk, CO: pk[SIL]}) for e, pk in _steps(SIL, SIL_PRODUCTS)])
    res = AN.analyze_run(_with_ms1(t, [(_mz(SIL), 0.3, 40.0), (_mz(CO), 0.6, 20.0)]),
                         _ions(SIL, ORG, CO), rivals=[_group(SIL, ORG)])
    assert res.windows[0].co_isolates == [CO]
    assert set(res.arbitration["note"]) == {""}
    res = AN.analyze_run(_with_ms1(t, [(_mz(SIL), 0.3, 40.0), (315.0, 0.6, NAN)]),
                         _ions(SIL, ORG), rivals=[_group(SIL, ORG)])
    assert ("an MS1 centroid no named ion matches, at m/z 315.0000"
            in res.arbitration["note"].iloc[0])


def _read(rival, roles, readings=(), weights=None, cross=None):
    """A constructed RivalRead of one spectrum 's': `roles` (peak, role, key),
    the rival's rank-1 `readings` (peak, loss, unnamed pieces) and the
    attribution's `weights` (key -> 'ion=w;...')."""
    w = dict(weights or {})
    return AR.RivalRead(
        rival=rival,
        roles=pd.DataFrame({"spectrum": ["s"] * len(roles), "peak": [r[0] for r in roles],
                            "role": [r[1] for r in roles], "key": [r[2] for r in roles],
                            "mz": [100.0] * len(roles), "intensity": [10.0] * len(roles)}),
        readings=pd.DataFrame({"spectrum": ["s"] * len(readings),
                               "peak": [x[0] for x in readings], "parent": [rival] * len(readings),
                               "rank": [1] * len(readings), "is_precursor": [False] * len(readings),
                               "loss_name": [x[1] for x in readings],
                               "unnamed_pieces": [x[2] for x in readings]}),
        fragments=pd.DataFrame({"fragment": list(w), "weights": list(w.values()),
                                "call": [""] * len(w)}),
        cross=dict(cross or {}))


def test_a_co_isolates_product_a_rival_claims_on_another_calibration_is_a_product():
    """A named co-isolate's product is no product only where every rival's
    reads give it wholly away -- on EVERY calibration: one rival that reads
    it, and is given a share of it, on the other rival's calibration claims
    it (a product); given none there, it is the co-isolate's."""
    A, B, X = "C10H17+", "C9H13O+", "C17H15O6+"
    frag = [(0, "fragment", "P")]
    theirs = {"P": f"{X}=1"}
    for share, out in (({"P": f"{A}=0.5;{X}=0.5"}, set()), (theirs, {("s", 0)})):
        reads = {A: _read(A, frag, weights=theirs,
                          cross={B: _read(A, frag, [(0, "H2O", 0)], share)}),
                 B: _read(B, frag, weights=theirs, cross={A: _read(B, frag, weights=theirs)})}
        assert AR.non_products(reads, [A, B]) == out, share


# ------------------------------------ a group whose windows disagree ----
def _two_windows():
    """W1: the siloxane window read 6 ppm low (its precursor matches the
    siloxane, not C14H19O8+; the arbitration supports the siloxane). W2: a
    C14H19O8+ window on its formula (supported there)."""
    F = "C14H19O8+"
    prods = {"C14H17O7+": 200.0, "C13H19O6+": 150.0, "C12H17O5+": 80.0, "C11H13O4+": 60.0,
             "C9H11O3+": 50.0}
    w1, w2 = _far_first(), _synthetic(_mz(F), _steps(F, prods))
    return AN.RunTables(spectra=pd.concat([w1.spectra, w2.spectra], ignore_index=True),
                        parents=pd.concat([w1.parents, w2.parents], ignore_index=True),
                        window=w1.window, manifest={})


def test_a_skipped_window_of_a_group_whose_windows_disagree_says_so():
    """The skipped window supports the rival its precursor matches, the
    group's other window another: --pick-supported keeps the first rival, so
    advising it (or naming the matching rival first, which reads the other
    window with the rival it outscores) is wrong -- with or without the
    flag, the note says the windows disagree."""
    F = "C14H19O8+"
    for pick in (False, True):
        res = AN.analyze_run(_two_windows(), _ions(F, SIL), rivals=[_group(F, SIL)],
                             pick_supported=pick)
        w1 = next(w for w in res.windows if w.status == "skipped")
        assert w1.arbitration["winner"].iloc[0] == SIL
        note = w1.notes[-1]
        assert "--pick-supported reads the tables with it" not in note, note
        assert "the group's windows support different rivals" in note, note
        assert F in note and SIL in note and "--pick-supported keeps C14H19O8+" in note
        # naming the siloxane first reads the other window with it only where
        # that window's precursor matches it
        assert note.endswith(f"naming {SIL} first reads the other windows with it where their "
                             "located precursors match it (and skips them where they do not)")
        assert "the group's windows support different rivals" in w1.rival_hint
        assert w1.manifest()["arbitration"]["hint"] == w1.rival_hint


def _skipped(winner, rivals):
    """A rival group's window the tables skip, its precursor 1 ppm above the
    siloxane, its arbitration's `winner` ('' : none)."""
    w = AN.WindowResult(parent_mz=315.106, status="skipped")
    w.precursor_mz = _mz(SIL) * (1 + 1e-6)
    w.reason = AN.SKIP_NO_TARGET
    w.arbitration = AR.frame([{"parent_mz": 315.106, "group": "g", "rival": R, "winner": winner,
                               "window_verdict": AR.SUPPORTED if winner else AR.UNDECIDED}
                              for R in rivals])
    return w


def test_a_skipped_window_whose_arbitration_supports_a_third_rival():
    """Three rivals: the tables read the first, beyond the target match; the
    precursor matches the second; the arbitration supports the third: name
    the third first (and the axis offset that matches it) -- but where
    another window of the group supports another rival, said."""
    F, X = "C14H19O8+", "C10H20^NO10+"
    w = _skipped(X, (F, SIL, X))
    AN._rival_skip(w, _group(F, SIL, X), F, _ions(F, SIL, X), axis_offset_ppm=0.0,
                   winners=[(315.106, X)])
    assert w.notes[-1].endswith(f"the arbitration supports {X}, which the located precursor "
                                "does not match either (+3.6 ppm): name it first, and -- if "
                                "the folder's m/z axis sits off its formulas -- pass "
                                "--mass-offset-ppm +3.6")
    assert w.rival_hint.startswith("parent 315.106: the arbitration supports") and \
        math.isnan(w.offset_hint_ppm)
    w = _skipped(X, (F, SIL, X))
    AN._rival_skip(w, _group(F, SIL, X), F, _ions(F, SIL, X), axis_offset_ppm=0.0,
                   winners=[(315.106, X), (315.107, F)])
    assert w.notes[-1].endswith(f"--mass-offset-ppm +3.6 (but the group's windows support "
                                f"different rivals ({X} at parent 315.106; {F} at parent "
                                "315.107))")


def test_a_skipped_undecided_window_says_which_windows_support_another_rival():
    """The arbitration supports none of the rivals: naming the rival the
    precursor matches first is advised -- and where another window of the
    group supports another rival, that naming it first reads that window
    with a rival it does not support is said too."""
    F = "C14H19O8+"
    w = _skipped("", (F, SIL))
    AN._rival_skip(w, _group(F, SIL), F, _ions(F, SIL), axis_offset_ppm=0.0)
    assert f"the arbitration supports none of the rivals: name {SIL} first" in w.notes[-1]
    assert "(but" not in w.notes[-1]
    w = _skipped("", (F, SIL))
    AN._rival_skip(w, _group(F, SIL), F, _ions(F, SIL), axis_offset_ppm=0.0,
                   winners=[(315.1074, F)])
    assert w.notes[-1].endswith(f"(but parent 315.1074 supports {F}: naming {SIL} first reads "
                                "those windows with a rival they do not support, where their "
                                "located precursors match it)")
    assert f"(but parent 315.1074 supports {F}" in w.rival_hint


# ------------------------------------------- the --rival group's name ----
def test_cli_a_rival_group_is_named_by_the_mz_typed(tmp_path, no_server, capsys):
    """Above m/z 1000 two groups 8 mDa apart were both named '1000.01' (six
    significant digits) and refused; the group is named by the m/z as typed."""
    run = _synthetic_run(tmp_path / "run")
    with pytest.raises(SystemExit) as e:
        cli.main(["ms2", "analyze", "--run-dir", str(run), "--no-figures",
                  "--rival", "1000.006=C47H20O24S+,C45H24O22Si3+",
                  "--rival", "1000.014=C37H40O16S2Si7+,C42H22NO27Si+"])
    assert "two rival groups are named" not in str(e.value.code)
    assert "no window was analysed" in str(e.value.code)
    out = tmp_path / "out"
    assert cli.main(["ms2", "analyze", "--run-dir", str(run), "--rival",
                     f"315.1060={SIL},{ORG}", "--no-figures", "--out-dir", str(out)]) == 0
    a = pd.read_csv(out / "ms2_arbitration.csv", dtype={"group": str})
    assert set(a["group"]) == {"315.1060"}
    assert "rivals [315.1060]" in capsys.readouterr().out


# ------------------------------------------------- missing m/z cells ----
def test_cli_an_ions_csv_with_na_mz_cells_runs(tmp_path, no_server):
    """R's write.csv writes NA for a missing number, a spreadsheet #N/A: no
    m/z, as 'nan' -- the run goes on; and NA for a missing string: an
    `adduct` cell of NA is no adduct. A cell that is no number is refused,
    its row named."""
    run = _write_tables(tmp_path / "run", _tables("ms2_monoterpene_a"))
    ions = tmp_path / "ions.csv"
    for cell in ("NA", "#N/A", "null", "<NA>"):
        ions.write_text(f'"formula","mz","adduct"\n"C10H17+",{cell},NA\n"C10H20^N+",{cell},NA\n'
                        f'"C10H19O+",{cell},{cell}\n')
        assert cli.main(["ms2", "analyze", "--run-dir", str(run), "--ions", str(ions),
                         "--no-figures", "--out-dir", str(tmp_path / "o")]) == 0, cell
        assert not (tmp_path / "o" / "ms2_arbitration.csv").exists()
    ions.write_text("formula,mz\nC10H17+,137.13\nC10H20^N+,abc\n")
    with pytest.raises(SystemExit, match="row 3: the mz cell 'abc' is no m/z"):
        cli.main(["ms2", "analyze", "--run-dir", str(run), "--ions", str(ions), "--no-figures",
                  "--out-dir", str(tmp_path / "o")])


# ------------------- the tables' rival scored on another calibration ----
#: the second tie rival's products (N2) and two both formulas hold
ELSEWHERE = {"C5H9N2O+": 250.0, "C4H7N2+": 200.0, "C6H11N2O+": 150.0, "C8H13O2+": 100.0,
             "C7H11O+": 80.0}


def test_the_tables_rival_scored_on_another_calibration_is_said():
    """The precursor sits on TIE_A, every product 1 ppm low, the products
    TIE_B's (5.0 ppm below). TIE_B is supported on TIE_A's calibration
    (`read_on`), credited with all of them; recalibrated on its own
    precursor -- as the tables read it, picked or named first -- it is
    credited with 13 % (one shared product misread), and the tables hold
    none of its own products. Said: in the window's notes, its `note`
    (`explained_own`) and --pick-supported's run note; with TIE_A read,
    nothing is."""
    t = _synthetic(round(_mz(TIE_A), 4), _moved(_steps(TIE_A, ELSEWHERE), -1.0, keep={TIE_A}))
    said = (f"the tables read {TIE_B} recalibrated on its own precursor, where it is credited "
            f"with 13% of the products; the arbitration credits it with 100% read on {TIE_A}'s "
            "calibration (read_on)")
    for order, pick in (([TIE_A, TIE_B], True), ([TIE_B, TIE_A], False)):
        res = AN.analyze_run(t, _ions(TIE_A, TIE_B), rivals=[_group(*order)],
                             pick_supported=pick)
        a = res.arbitration.set_index("rival")
        assert (a.loc[TIE_B, "verdict"], a.loc[TIE_B, "read_on"]) == (AR.SUPPORTED, TIE_A)
        assert a.loc[TIE_B, "analysed"] and a.loc[TIE_B, "explained"] == pytest.approx(1.0)
        assert a.loc[TIE_B, "explained_own"] == pytest.approx(100 / 780)
        assert a.loc[TIE_A, "explained_own"] == a.loc[TIE_A, "explained"]
        assert said in a.loc[TIE_B, "note"] and a.loc[TIE_A, "note"] == "", order
        w = res.windows[0]
        assert w.target == TIE_B and any(said in n for n in w.notes), order
        # the tables hold none of the products it is credited with: one
        # misreading of the shared C8H13O2+, 3.5 ppm off
        assert set(res.fragments["fragment"]) == {"C6H11N3O+"}, order
        runs = [n for n in res.notes if "--pick-supported reads every table" in n]
        assert bool(runs) == pick
        if pick:
            assert (f"while its support is read on another rival's calibration at parent "
                    f"{S.spectrum_key(round(_mz(TIE_A), 4))} ({TIE_A}'s: 100% of the products, "
                    "13% on its own)") in runs[0]
    res = AN.analyze_run(t, _ions(TIE_A, TIE_B), rivals=[_group(TIE_A, TIE_B)])
    assert not any("recalibrated on its own precursor" in n for n in res.windows[0].notes)
    assert set(res.arbitration["note"]) == {""}


# --------------------------------------- one product the swap cannot trade ----
def test_one_product_the_swap_cannot_trade_decides_nothing():
    """The siloxane's ^NH3 loss, which its Si-free twin reads as its alias,
    and ONE odd-Si product only the siloxane can read: one product of its
    own is short of DIAGNOSTIC_MIN -- undecided (four such: supported,
    above)."""
    _, a = _arbitrate(_steps(SIL, {"C8H25O4Si4+": 200.0, "C3H9Si+": 60.0}), [SIL, ORG])
    assert set(a["verdict"]) == {AR.UNDECIDED} and set(a["winner"]) == {""}
    assert a.loc[SIL, "n_diagnostic"] == 1
    assert (f"against {ORG}: {SIL} leads it by 23%, but on 1 product(s) of its own"
            in a.loc[SIL, "evidence"])


# ------------------------------------------------ an NA adduct cell ----
def test_an_na_adduct_cell_is_no_adduct(tmp_path):
    """R's write.csv writes NA for a missing string too: an `adduct` cell of
    NA (or '#N/A', 'nan') is a blank one, no adduct -- not a label refused."""
    p = tmp_path / "ions.csv"
    p.write_text('"","formula","mz","adduct"\n"1","C10H20^N+",155.156,"[M+^NH4]+"\n'
                 '"2","C10H17+",137.1325,NA\n"3","C10H19O+",NA,#N/A\n"4","C10H16+",NA,nan\n')
    assert AN.read_ions_adducts(str(p)) == {"C10H20^N+": "[M+^NH4]+"}
    p.write_text("formula,adduct\nC10H17+,[M+Na]+\n")
    with pytest.raises(ValueError, match="row 2: adduct"):
        AN.read_ions_adducts(str(p))


# ------------------------------------------------- the docs say it ----
def test_the_docs_say_which_element_is_judged_and_what_aliases_decide():
    """The docs, the CHANGELOG and the module say what the rules do:
    `contradicted` judges an element ANOTHER rival lacks (`_distinct`; two Si
    rivals against an Si-free one are both judged on Si); aliases read as
    each other's only the products the swap can trade -- products it cannot
    (an odd number of Si) can decide the pair, DIAGNOSTIC_MIN of them (four:
    the siloxane supported over its Si-free twin; one: undecided, tested
    above); a contradicted alias supports nobody by itself, and leaves the
    window undecided only where both read the same products; a reading
    credited through a merge is not the element absent -- not every
    credited one."""
    root = Path(__file__).resolve().parents[1]
    docs = " ".join((root / "docs" / "MS2.md").read_text().split())
    log = " ".join((root / "CHANGELOG.md").read_text().split())
    mod = " ".join(AR.__doc__.split())
    for text in (docs, log, mod):
        assert "only it holds" not in text
    assert "the elements it holds that another rival lacks and their `signature`" in docs
    for text in (docs, mod):
        assert "read every product as each other's" not in text
        assert "numbers the swap can trade" in text
        assert "cannot trade -- an odd number of Si -- decides the pair" not in text
        assert "the window still undecided" not in text.replace("`", "")
        assert "support never by elimination" in text or "support is never by" in text
    assert "a credited reading is never also the element absent" not in docs
    assert "a reading credited through such a merge is never also counted as" in docs
    assert "with one such product next to its ^NH3 loss, `undecided`" in docs
    assert "one such product decides nothing" in mod
