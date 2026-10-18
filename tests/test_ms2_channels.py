"""Offline tests for the product channels and the binding class
(peaky/ms2/channels.py), the adduct group formulas (chemistry.ADDUCT_GROUPS)
and their wiring into `ms2 analyze` (analyze.py, `peaky ms2 analyze --ions
CSV with an 'adduct' column | --adduct | --reagent`).

Known answers, our own data (the committed fixtures):

  * monoterpene standard A (2026-09-11): C10H17+ ([M+H]+ of C10H16) is
    covalent -- an [M+H]+ ion adds no reagent group --, its skeletal products
    C6H9+ (-C4H8) and others; C10H20^N+ ([M+^NH4]+ of limonene) is an adduct:
    at NCE 1 its products are 89.6 % ^NH3 loss (C10H17+) and 10.4 % ^NH3 +
    C4H8 (C6H9+), and it falls to half by E_cm 0.855 eV.
  * the 2026-09-10 exploratory 15N-ammonium file: its eight ^NH4 adducts --
    four targets and four co-isolates -- read as adducts (C12H22^NO2+ the
    closest call: 58 % reagent loss + small losses against 42 % of its
    in-source [M+^NH4-H2O]+ and [M+^NH4-2H2O]+), its [M+H]+ ions as covalent.

Synthetic: every channel of one product (the reagent ion and a reagent
cluster with nothing of M's carbon, the reagent loss, + small losses, a
monomer shed with both halves named, a skeletal fragment, an unexplained
radical or partial-reagent product), the halves rule (carbons, oxygen), and
one window per class: an adduct, a cluster, a cluster beside a covalent dimer
('mixed'), a skeletal ion, an ion that survives, one whose reagent loss needs
more than an adduct's energy, one that falls apart into an unobserved
reagent ion, one mostly unexplained, an [M+H]+ ion and one with no adduct;
the share's per-scan spread; the note on an unnamed co-isolate; the adduct
precedence (CSV, --adduct, --reagent); every table unchanged with no adduct
given; and the CLI end to end. And the review's cases: a skeletal ion that
falls apart like an adduct (no covalent call, SKELETAL_ECM from both sides);
the energy test read off the curves (one energy, a gap, an ion gone at the
top energies) and COVALENT_ECM from both sides; DOMINANT over the explained
products; the cluster population at any energy the ion is seen at; the
unseen reagent ion against the scans' first mass, and not with skeletal
products; the window cut where the precursor is last seen; an absent ion and
a sliver; another named ion's product; no share spread where nothing is
read; radical reagent neutrals (superoxide, carbonate); an in-source form; a
14N cluster whose shed half holds N, a protonated half, a small loss; the
halves' DBE rule; a strict part of the group; group forms first, closed-shell
M, by-elimination notes; wrong-polarity flags refused; the E50 note and band;
the co-isolate notes; and the wiring of every input. Also: the group table
checked against the built-in forms whatever is registered at run time; no
covalent call for a skeletal ion not known to hold to SKELETAL_ECM (one
energy, a hole, a ladder that starts high; S falling first); no volts at
either end of the ladder (E_cm off the line through the others, F and S
compared on NCE); an equal split gives a never-seen ion no class; S's part
in the energy test; split halves, no channel, every product another ion's;
the adduct evidence's monomer clauses; monomers shed after the precursor is
gone; the unseen-reagent rule's S and COVALENT_ECM; the scan range of the
window's own spectra; a co-isolate with no adduct; the wording for a form
with no group; M itself as the reagent's loss by electron transfer, a
product M's own atoms make; the reagent ion with its own neutral (Ur2H+,
formate and acetate dimers); a C6 loss; the tie note; the CSV row named; and
the lowest own [M+H]+ crossing quoted as measured; and a 14N cluster
of two halves that both hold the reagent neutral's atoms (two organonitrates,
two amides) never reads as one, a 15N reagent does; a tie in an adduct's
window said as one, generically; the energy test on the ion's own products
(a co-isolate's fragment cannot turn a surviving ion into an adduct; the
re-read E50, its E_cm and no band); another ion's product is no product of
a never-seen ion's own, nor an equal split of a pool of one; a bounded E_cm
said for an ion with no adduct; and 'barely fragments' with volts at one
energy only."""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from peaky import cli
from peaky.chem import chemistry as C
from peaky.chem import profiles as PR
from peaky.ms2 import analyze as AN
from peaky.ms2 import channels as CH
from peaky.ms2 import scans as SC
from peaky.ms2 import sources as S
from tests.test_ms2_breakdown import _synthetic, _tables, _write_run
from tests.test_ms2_replicates import _replicated

NAN = float("nan")
MT_IONS = ("C10H17+", "C10H20^N+", "C10H19O+")
NH4_IONS = ("C7H6NS+", "C4H10NS2+", "C8H10NO+",
            "C4H10^NO4+", "C8H9O2+", "C5H14^NO3+", "C6H18^NO2+", "C9H14^N+",
            "C11H13O+", "C11H15O2+", "C11H16^NO+", "C11H18^NO2+", "C12H21O2+",
            "C12H22^NO2+", "C8H28^NO4Si4+")
#: the 2026-09-10 file's 15N-ammonium adducts (targets and co-isolates)
NH4_ADDUCTS = ("C4H10^NO4+", "C5H14^NO3+", "C6H18^NO2+", "C9H14^N+", "C11H16^NO+",
               "C11H18^NO2+", "C12H22^NO2+", "C8H28^NO4Si4+")


def _mz(ion):
    return AN.ion_mz(ion)


def _ions(*formulas):
    return AN.collect_ions((f, NAN, _mz(f)) for f in formulas)


def _adducts(ions, label):
    return {f: CH.ion_adduct(f, label) for f in ions}


def _cls(res, ion, parent=None):
    c = res.classes
    r = c[(c["ion"] == ion) & (True if parent is None else np.isclose(c["parent_mz"], parent))]
    assert len(r) == 1, (ion, parent)
    return r.iloc[0]


def _chan(res, ion, nce, channel, parent=None):
    c = res.channels
    r = c[(c["ion"] == ion) & (c["nce"] == nce) & (c["channel"] == channel)
          & (True if parent is None else np.isclose(c["parent_mz"], parent))]
    assert len(r) == 1, (ion, nce, channel)
    return r.iloc[0]


# ------------------------------------------------------- group formulas ----
def test_every_adduct_label_has_a_group_formula_that_gives_its_shift():
    """ion_mz(M, label) = neutral_mass(M) + neutral_mass(group) - charge x
    M_E, for every built-in label (BUILTIN_ADDUCTS) but sodium and potassium
    (no element of the formula layer)."""
    assert set(C.ADDUCT_GROUPS) | set(C.ADDUCT_NO_GROUP) == C.BUILTIN_ADDUCTS
    assert C.BUILTIN_ADDUCTS <= set(C.ADDUCT_SHIFTS)
    assert not set(C.ADDUCT_GROUPS) & set(C.ADDUCT_NO_GROUP)
    for label, group in C.ADDUCT_GROUPS.items():
        mass = sum(C.M[e] * n for e, n in group.items()) - C.adduct_charge(label) * C.M_E
        assert mass == pytest.approx(C.ADDUCT_SHIFTS[label], abs=1e-8), label
        # and an ion built from it has the m/z ion_mz gives
        m = "C10H16O4"
        ion = {e: C.parse_formula(m).get(e, 0) + group.get(e, 0)
               for e in set(C.parse_formula(m)) | set(group)}
        assert C.ion_mass({e: n for e, n in ion.items() if n}, C.adduct_charge(label)) == \
            pytest.approx(C.ion_mz(m, label), abs=1e-8), label
    assert C.adduct_group("[M+NH4]+") == {"H": 4, "N": 1}
    g = C.adduct_group("[M+^NO3]-")
    g["O"] = 99                                         # a copy, not the table
    assert C.ADDUCT_GROUPS["[M+^NO3]-"]["O"] == 3
    for bad in ("[M+Na]+", "[M+K]+", "[M+Xy]+"):
        with pytest.raises(ValueError, match="no group formula"):
            C.adduct_group(bad)
    assert [C.adduct_charge(x) for x in ("[M+H]+", "[M]+.", "[M-H]-", "[M]-.")] == [1, 1, -1, -1]
    with pytest.raises(ValueError, match="charge sign"):
        C.adduct_charge("[M+H]")


def test_the_group_table_is_checked_against_the_built_in_forms(monkeypatch):
    """A form registered at run time (a solvent cluster's rung, which a
    module of the suite registers when it is imported) has a shift and no
    group formula: the built-in table is what every group is checked
    against, whatever else the live table holds."""
    monkeypatch.setattr(C, "ADDUCT_SHIFTS", dict(C.ADDUCT_SHIFTS))
    label = "[M+(C2H6O)H]+"
    C.register_adduct(label, C.ADDUCT_SHIFTS.get(
        label, C.ADDUCT_SHIFTS["[M+H]+"] + C.neutral_mass("C2H6O")))
    assert label in C.ADDUCT_SHIFTS and label not in C.BUILTIN_ADDUCTS
    assert set(C.ADDUCT_GROUPS) | set(C.ADDUCT_NO_GROUP) == C.BUILTIN_ADDUCTS
    with pytest.raises(ValueError, match="no group formula"):
        C.adduct_group(label)


def test_an_ion_read_as_its_adduct():
    a = CH.ion_adduct("C10H20^N+", "[M+^NH4]+", "x")
    assert (a.ion, a.label, a.source, a.charge) == ("C10H20^N+", "[M+^NH4]+", "x", 1)
    assert a.body == {"C": 10, "H": 16} and a.has_group
    assert (a.reagent_neutral, a.reagent_ion) == ("H3^N", "H4^N+")
    # what each form leaves as on proton transfer, and its reagent ion
    for ion, label, neutral, rion in (
            ("C12H13N2O5+", "[M+(CH4N2O)H]+", "CH4N2O", "CH5N2O+"),
            ("C11H10^NO5-", "[M+^NO3]-", "H^NO3", "^NO3-"),
            ("C2H3O2Br-", "[M+Br]-", "HBr", "Br-"),
            ("C4H5O4I2-", "[M-H+I2]-", "I2", ""),           # A- . I2 -> A- + I2
            ("C2H4O5-", "[M+CO3]-", "CHO3", "CO3-"),         # HCO3: a radical
            ("C2H4Br2O2-", "[M+HBr+Br]-", "H2Br2", "HBr2-")):
        a = CH.ion_adduct(ion, label)
        assert (a.reagent_neutral, a.reagent_ion, a.has_group) == (neutral, rion, True), label
    for ion, label in (("C10H17+", "[M+H]+"), ("C5H7O4-", "[M-H]-"), ("C10H8+.", "[M]+.")):
        a = CH.ion_adduct(ion, label)
        assert not a.has_group and (a.reagent_neutral, a.reagent_ion) == ("", ""), label
    with pytest.raises(ValueError, match="does not hold the group"):
        CH.ion_adduct("C10H17+", "[M+NH4]+")
    with pytest.raises(ValueError, match="anion form"):
        CH.ion_adduct("C10H17+", "[M-H]-")
    with pytest.raises(ValueError, match="no group formula"):
        CH.ion_adduct("C10H17+", "[M+Na]+")
    assert CH.fits("C10H20^N+", "[M+^NH4]+") and not CH.fits("C10H20^N+", "[M+NH4]+")


# ------------------------------------------------------------- channels ----
DIMER = CH.ion_adduct("C24H24NO4+", "[M+NH4]+")          # (C12H10O2)2 . NH4+


@pytest.mark.parametrize("product, channel, halves", [
    ("H4N+", CH.REAGENT_ION, ""),                         # NH4+ itself
    ("H6NO+", CH.REAGENT_ION, ""),                        # NH4+ . H2O: none of M's carbon
    ("C24H21O4+", CH.REAGENT_LOSS, ""),                   # [M+H]+
    ("C24H19O3+", CH.REAGENT_LOSS_SMALL, ""),             # [M+H-H2O]+
    ("C12H14NO2+", CH.MONOMER, "C12H10O2 + C12H10O2"),    # monomer . NH4+
    ("C10H12NO2+", CH.MONOMER, "C10H8O2 + C14H12O2"),
    ("C24H22NO3+", CH.SKELETAL, ""),                      # [M+NH4-H2O]+
    ("C12H14N+", CH.SKELETAL, ""),                        # a half with no O, M with O
    ("C23H22NO2+", CH.SKELETAL, ""),                      # a C1 half (CH2O2): no monomer
    ("C24H20O4+.", CH.UNEXPLAINED, ""),                   # M+. : a radical loss (NH4)
    ("C24H23O4+", CH.UNEXPLAINED, ""),                    # part of NH4 kept (no N, H23)
])
def test_each_channel_of_a_product(product, channel, halves):
    assert CH.product_channel(DIMER, product) == (channel, halves)


def test_channels_by_the_form():
    # no adduct: 'product'; no reagent group: every product skeletal
    assert CH.product_channel(None, "C6H9+") == (CH.PRODUCT, "")
    h = CH.ion_adduct("C24H21O4+", "[M+H]+")
    assert CH.product_channel(h, "C12H11O2+") == (CH.SKELETAL, "")
    # a monomer needs MONOMER_MIN_C carbons on both sides: two C5 halves of
    # a C10 hydrocarbon adduct are a skeletal split, two C6 halves of C12 are
    # monomers (no O in M, no O asked)
    c10 = CH.ion_adduct("C10H20^N+", "[M+^NH4]+")
    assert CH.MONOMER_MIN_C == 6
    assert CH.product_channel(c10, "C5H12^N+") == (CH.SKELETAL, "")
    assert CH.product_channel(c10, "C10H17+") == (CH.REAGENT_LOSS, "")
    assert CH.product_channel(c10, "C6H9+") == (CH.REAGENT_LOSS_SMALL, "")
    c12 = CH.ion_adduct("C12H24^N+", "[M+^NH4]+")
    assert CH.product_channel(c12, "C6H14^N+") == (CH.MONOMER, "C6H10 + C6H10")
    # urea: the uronium ion, urea lost, part of urea kept ([M+UrH-NH3]+)
    ur = CH.ion_adduct("C12H13N2O5+", "[M+(CH4N2O)H]+")
    assert CH.product_channel(ur, "CH5N2O+")[0] == CH.REAGENT_ION
    assert CH.product_channel(ur, "C11H9O4+")[0] == CH.REAGENT_LOSS
    assert CH.product_channel(ur, "C12H10NO5+")[0] == CH.UNEXPLAINED
    # 15N-nitrate: the nitrate ion; a polyoxy H/^N/O composition with none of
    # M's carbon (the C2F3 = H2^NO4 alias of CF3COO-) is no skeletal product
    no3 = CH.ion_adduct("C11H10^NO5-", "[M+^NO3]-")
    assert CH.product_channel(no3, "^NO3-")[0] == CH.REAGENT_ION
    assert CH.product_channel(no3, "H2^NO6-")[0] == CH.REAGENT_ION
    assert CH.product_channel(no3, "C11H9O2-")[0] == CH.REAGENT_LOSS
    assert CH.product_channel(no3, "C9H8^NO5-")[0] == CH.SKELETAL


# ------------------------------------------------------------- classes ----
NCE = (1, 10, 20, 30)


def _run(parent, steps, ions, label, **kw):
    """One synthetic window (volts 4 + 0.2 NCE), every ion read as `label`."""
    t = _synthetic(parent, steps, **kw)
    res = AN.analyze_run(t, _ions(*ions), adducts=_adducts(ions, label))
    return t, res


def test_an_adduct():
    """[M+NH4]+ of C11H8O3: NH3 lost first (then H2O), a little [M+NH4-H2O]+."""
    x = "C11H12NO3+"
    steps = [(e, {x: i, "C11H9O3+": a, "C11H7O2+": b, "C11H10NO2+": c})
             for e, i, a, b, c in zip(NCE, (1000.0, 600.0, 300.0, 100.0),
                                      (300.0, 600.0, 700.0, 600.0), (60.0, 150.0, 300.0, 400.0),
                                      (20.0, 30.0, 40.0, 50.0))]
    _, res = _run(round(_mz(x), 4), steps, (x,), "[M+NH4]+")
    r = _cls(res, x)
    assert r["class"] == CH.CLASS_ADDUCT
    # the window is NCE 1 alone: its products are 380 / 1380 of ion + products
    assert (r["nce_lo"], r["nce_hi"], r["n_energies"]) == (1.0, 1.0, 1)
    assert r["frag"] == pytest.approx(380 / 1380)
    assert (r["share_reagent_loss"], r["share_reagent_loss_small"], r["share_skeletal"]) == \
        pytest.approx((300 / 380, 60 / 380, 20 / 380))
    assert r["share_monomer"] == 0 and r["monomer_pop"] == 0
    assert (r["reagent_neutral"], r["reagent_ion"]) == ("H3N", "H4N+")
    assert r["e50_curve"] == "F" and r["e50_ecm"] < CH.COVALENT_ECM
    assert r["evidence"].startswith("loses the reagent first: in the window (NCE 1, products "
                                    "28 % of ion + products): reagent ion 0 %, reagent loss 79 %")
    # the channel rows: an 'intact' row and the eight product channels per energy
    ch = res.channels[res.channels["nce"] == 10]
    assert list(ch["channel"]) == [CH.INTACT, *CH.PRODUCT_CHANNELS]
    assert _chan(res, x, 10, CH.INTACT)["intensity"] == 600 and \
        math.isnan(_chan(res, x, 10, CH.INTACT)["share"])
    rl = _chan(res, x, 10, CH.REAGENT_LOSS)
    assert (rl["intensity"], rl["share"], rl["products"]) == (600.0, pytest.approx(600 / 780),
                                                              "C11H9O3+")
    # the shares add up to 1 at every energy, and E_cm is the ion's
    prod = res.channels[res.channels["channel"] != CH.INTACT]
    assert np.allclose(prod.groupby("nce")["share"].sum(), 1.0)
    assert np.allclose(res.channels["e_cm"],
                       (4 + 0.2 * res.channels["nce"]) * 28.0061 / (28.0061 + _mz(x)))
    # the products' channel intensity is the breakdown's weighted product sum
    bd = res.breakdown.set_index("nce")
    assert np.allclose(prod.groupby("nce")["intensity"].sum(), bd["frag_weighted"])


def test_a_cluster_names_both_halves():
    """(C12H10O2)2 . NH4+: 94 % of its products at NCE 1 are monomer . NH4+,
    and they hold 83 % of ion + products -- a cluster."""
    x, mono, mh = "C24H24NO4+", "C12H14NO2+", "C12H11O2+"
    steps = [(1, {x: 200.0, mono: 1500.0, mh: 100.0}),
             (10, {x: 50.0, mono: 1300.0, mh: 400.0}),
             (20, {x: 10.0, mono: 800.0, mh: 900.0})]
    _, res = _run(round(_mz(x), 4), steps, (x,), "[M+NH4]+")
    r = _cls(res, x)
    assert r["class"] == CH.CLASS_CLUSTER
    assert r["share_monomer"] == pytest.approx(1500 / 1600)
    assert r["monomer_pop"] == pytest.approx(1500 / 1800)
    assert r["halves"] == "C12H10O2 + C12H10O2"
    assert "sheds whole monomers (C12H10O2 + C12H10O2 the largest split)" in r["evidence"]
    m = _chan(res, x, 1, CH.MONOMER)
    assert (m["products"], m["halves"], m["n_products"]) == (mono, "C12H10O2 + C12H10O2", 1)
    # at higher energies the monomer . NH4+ loses NH3: [Y+H]+, read as
    # 'reagent loss + split' (a cluster's or a covalent dimer's halves) --
    # why the class is read at the lowest energies
    sp = _chan(res, x, 20, CH.SPLIT)
    assert sp["share"] > 0.5 and sp["halves"] == "C12H10O2 + C12H10O2"
    assert _chan(res, x, 20, CH.REAGENT_LOSS)["share"] == 0
    # as the module's docstring says
    assert "reads there as reagent loss + split" in " ".join(CH.__doc__.split())


def test_a_cluster_beside_a_covalent_dimer_is_mixed():
    """Monomer shedding is all of its products at NCE 1, but it holds at most
    16 % of ion + products at any energy (< CLUSTER_POP): the rest holds to
    higher energies, where it loses NH3 whole -- 'mixed'; and a covalent dimer
    that loses only NH3 is an adduct that sheds no monomer."""
    x = "C22H28NO7+"
    mono, dh = "C11H16NO4+", "C22H25O7+"
    steps = [(1, {x: 1000.0, mono: 150.0}), (10, {x: 900.0, mono: 160.0, dh: 50.0}),
             (20, {x: 400.0, mono: 150.0, dh: 400.0}), (30, {x: 50.0, mono: 50.0, dh: 600.0})]
    _, res = _run(round(_mz(x), 4), steps, (x,), "[M+NH4]+")
    r = _cls(res, x)
    assert r["class"] == CH.CLASS_MIXED and r["share_monomer"] == 1.0
    assert r["monomer_pop"] == pytest.approx(150 / 950)
    assert r["halves"] == "C11H12O4 + C11H12O3"
    steps = [(1, {x: 1000.0, dh: 20.0}), (10, {x: 900.0, dh: 150.0}),
             (20, {x: 400.0, dh: 400.0}), (30, {x: 50.0, dh: 600.0})]
    _, res = _run(round(_mz(x), 4), steps, (x,), "[M+NH4]+")
    r = _cls(res, x)
    assert r["class"] == CH.CLASS_ADDUCT and (r["nce_lo"], r["nce_hi"]) == (1.0, 10.0)
    assert r["share_reagent_loss"] == 1.0
    assert r["evidence"].endswith("it sheds no monomer (its monomer . reagent ions at most 0 % of "
                                  "ion + products): M is no cluster of two neutrals")


def test_a_skeletal_ion_is_covalent():
    """Its products keep the reagent, and it holds half its signal to E_cm
    0.97 eV (F) -- past SKELETAL_ECM: a covalent skeleton."""
    x = "C11H12^NO3+"
    steps = [(e, {x: i, "C11H10^NO2+": s, "C10H12^NO+": t, "C11H9O3+": a})
             for e, i, s, t, a in zip(NCE, (1000.0, 900.0, 700.0, 300.0),
                                      (150.0, 250.0, 400.0, 500.0), (50.0, 100.0, 200.0, 300.0),
                                      (20.0, 30.0, 40.0, 50.0))]
    _, res = _run(round(_mz(x), 4), steps, (x,), "[M+^NH4]+")
    r = _cls(res, x)
    assert r["class"] == CH.CLASS_COVALENT
    assert r["share_skeletal"] == pytest.approx(200 / 220)
    assert r["e50_curve"] == "F" and CH.SKELETAL_ECM < r["e50_ecm"] < CH.COVALENT_ECM
    assert r["evidence"].startswith("keeps the reagent and fragments its skeleton")


def test_an_ion_that_survives_is_covalent():
    """A protonated amide read as an ammonium adduct (its formula twin): its
    NH3 loss never holds FRAG_MIN of ion + products, and it holds half its
    signal to the highest energy, E_cm 2.2 eV."""
    x = "C11H12NO+"
    steps = [(e, {x: 1000.0, "C11H9O+": f}) for e, f in ((1, 2.0), (20, 5.0), (40, 10.0),
                                                           (60, 20.0))]
    _, res = _run(round(_mz(x), 4), steps, (x,), "[M+NH4]+")
    r = _cls(res, x)
    assert r["class"] == CH.CLASS_COVALENT and r["e50_curve"] == ""
    assert r["n_energies"] == 4 and r["frag"] == pytest.approx(20 / 1020)
    assert r["share_reagent_loss"] == 1.0
    top = (4 + 0.2 * 60) * 28.0061 / (28.0061 + _mz(x))
    assert top > CH.COVALENT_ECM
    assert r["evidence"].startswith("survives: its products never reach 0.1 of ion + products "
                                    f"(at most 2 %), and it holds half its signal to E_cm "
                                    f"{top:.2f} eV, the highest energy")
    # the docstring quotes it
    doc = " ".join(test_an_ion_that_survives_is_covalent.__doc__.split())
    assert f"E_cm {top:.1f} eV" in doc


def test_a_reagent_loss_that_needs_more_than_an_adducts_energy_is_covalent():
    x = "C8H8NO2+"
    steps = [(e, {x: i, "C8H5O2+": f}) for e, i, f in ((1, 1000.0, 250.0), (20, 800.0, 300.0),
                                                       (40, 600.0, 450.0), (60, 300.0, 700.0))]
    _, res = _run(round(_mz(x), 4), steps, (x,), "[M+NH4]+")
    r = _cls(res, x)
    assert r["share_reagent_loss"] == 1.0 and r["e50_ecm"] > CH.COVALENT_ECM
    assert r["class"] == CH.CLASS_COVALENT
    assert "its products are the reagent's loss" in r["evidence"]
    assert "a protonated amide losing NH3" in r["evidence"]


def test_a_crossing_below_the_lowest_energy_reads_no_energy_test():
    """Already below half at its lowest energy, E_cm 1.39 eV there: the E50
    lies at or below it -- an upper bound, which rules nothing out -- so a
    reagent-loss ion stays an adduct."""
    x = "C11H12NO+"
    steps = [(30, {x: 300.0, "C11H9O+": 700.0}), (40, {x: 100.0, "C11H9O+": 900.0})]
    _, res = _run(round(_mz(x), 4), steps, (x,), "[M+NH4]+")
    r = _cls(res, x)
    assert (r["e50_flag"], r["e50_curve"]) == ("below_range", "F")
    assert r["e50_ecm"] > CH.COVALENT_ECM
    assert r["class"] == CH.CLASS_ADDUCT
    assert "(F, at or below the lowest energy)" in r["evidence"]


def test_the_formula_an_ion_reads_a_fragment_as():
    """The lowest energy at which it reads it as one of the formulas the
    fragment can be where it sits; else the lowest energy it reads it at
    all."""
    roles = pd.DataFrame({
        "spectrum": ["s1", "s2", "s2"], "role": ["fragment"] * 3,
        "key": ["A+ | B+", "A+ | B+", "C+"],
        "formulas": ["X: A+", "X: B+; Y: B+", "X: C+"]})
    got = CH.fragment_formulas(roles, ["s1", "s2"], {"A+ | B+": ["B+"], "C+": []})
    assert got == {("A+ | B+", "X"): "B+", ("A+ | B+", "Y"): "B+", ("C+", "X"): "C+"}
    got = CH.fragment_formulas(roles, ["s1", "s2"], {"A+ | B+": ["A+", "B+"]})
    assert got[("A+ | B+", "X")] == "A+"


def test_an_ion_that_leaves_as_its_reagent_ion():
    """No product at all, while S falls to half by an adduct's energy and the
    reagent ion NH4+ (m/z 18.03) lies below the lowest first mass an MS2 scan
    is set to (FIRST_MASS_MIN) and below every peak the spectra hold: the ion
    left as NH4+, M whole -- an adduct; with a lowest peak below NH4+'s m/z
    it is 'unknown'."""
    x = "C22H28NO9+"
    steps = [(e, {x: i}) for e, i in zip(NCE, (1000.0, 700.0, 400.0, 100.0))]
    t, res = _run(round(_mz(x), 4), steps, (x,), "[M+NH4]+")
    r = _cls(res, x)
    assert r["class"] == CH.CLASS_ADDUCT and r["e50_curve"] == "S"
    assert r["e50_ecm"] < CH.COVALENT_ECM and np.isnan(r["share_reagent_loss"])
    assert ("its reagent ion H4N+ (m/z 18.034) lies below the lowest first mass an MS2 scan is "
            "set to (FIRST_MASS_MIN) (m/z 40.000)") in r["evidence"]
    assert "S, which needs a steady precursor supply" in r["note"]
    steps = [(e, {x: i, 17.0: 5.0}) for e, i in zip(NCE, (1000.0, 700.0, 400.0, 100.0))]
    _, res = _run(round(_mz(x), 4), steps, (x,), "[M+NH4]+")
    assert _cls(res, x)["class"] == CH.CLASS_UNKNOWN
    assert _cls(res, x)["evidence"].startswith("barely fragments")


def test_mostly_unexplained_products_give_no_class():
    """A uronium adduct whose products keep part of urea ([M+UrH-NH3]+)."""
    x = "C12H13N2O5+"
    steps = [(e, {x: i, "C12H10NO5+": u, "CH5N2O+": a})
             for e, i, u, a in zip(NCE, (1000.0, 500.0, 200.0, 50.0), (400.0, 600.0, 700.0, 700.0),
                                   (100.0, 200.0, 300.0, 300.0))]
    _, res = _run(round(_mz(x), 4), steps, (x,), "[M+(CH4N2O)H]+")
    r = _cls(res, x)
    assert r["class"] == CH.CLASS_UNKNOWN and r["share_unexplained"] == pytest.approx(0.8)
    assert r["evidence"].startswith("most of its products in the window (NCE 1) are unexplained")


def test_no_adduct_and_no_reagent_group():
    x, frag = "C10H17+", "C6H9+"
    steps = [(e, {x: i, frag: 1000.0 - i}) for e, i in zip(NCE, (900.0, 700.0, 400.0, 100.0))]
    t = _synthetic(137.133, steps)
    res = AN.analyze_run(t, _ions(x), adducts={x: None})
    r = _cls(res, x)
    assert r["class"] == CH.CLASS_UNKNOWN and r["adduct"] == ""
    assert r["evidence"].startswith("no adduct given")
    assert list(res.channels["channel"].unique()) == [CH.INTACT, CH.PRODUCT]
    assert np.allclose(res.channels.loc[res.channels["channel"] == CH.PRODUCT, "share"], 1.0)
    res = AN.analyze_run(t, _ions(x), adducts=_adducts((x,), "[M+H]+"))
    r = _cls(res, x)
    assert r["class"] == CH.CLASS_COVALENT and r["share_skeletal"] == 1.0
    assert r["evidence"].startswith("an [M+H]+ ion adds no reagent group")


def test_two_window_ions_share_a_fragment_by_their_weights():
    """A shared fragment enters each ion's channels times its attribution
    weight; the products' sum is each ion's frag_weighted."""
    x, y = "C10H20^N+", "C10H19O+"
    steps = [(e, {x: a, y: b, "C10H17+": c, "C6H9+": d})
             for e, a, b, c, d in zip(NCE, (1000.0, 600.0, 300.0, 100.0),
                                      (500.0, 450.0, 300.0, 100.0), (300.0, 600.0, 700.0, 600.0),
                                      (50.0, 200.0, 400.0, 500.0))]
    t = _synthetic(155.1559, steps, width=0.4)
    res = AN.analyze_run(t, _ions(x, y), adducts={x: CH.ion_adduct(x, "[M+^NH4]+"),
                                                  y: CH.ion_adduct(y, "[M+H]+")})
    bd = res.breakdown
    for ion in (x, y):
        prod = res.channels[(res.channels["ion"] == ion) & (res.channels["channel"] != CH.INTACT)]
        got = prod.groupby("nce")["intensity"].sum()
        want = bd[bd["ion"] == ion].set_index("nce")["frag_weighted"]
        assert np.allclose(got, want.reindex(got.index)), ion
    w = res.fragments.set_index("fragment").loc["C10H17+", "weights"]
    wx = dict(kv.split("=") for kv in w.split(";"))
    r = _chan(res, x, 1, CH.REAGENT_LOSS)
    assert r["intensity"] == pytest.approx(float(wx[x]) * 300.0)


def test_the_share_spread_over_the_scans():
    x = "C11H12NO3+"
    rl, sk = "C11H9O3+", "C11H10NO2+"
    steps = [(1, [{x: 1000.0, rl: 300.0, sk: 100.0}, {x: 1100.0, rl: 360.0, sk: 40.0},
                  {x: 900.0, rl: 240.0, sk: 60.0}]),
             (10, [{x: 600.0, rl: 600.0, sk: 30.0}, {x: 640.0, rl: 580.0, sk: 50.0}]),
             (20, [{x: 300.0, rl: 700.0, sk: 40.0}])]
    t = _replicated(round(_mz(x), 4), steps, ions=(x,))
    res = AN.analyze_run(t, _ions(x), adducts=_adducts((x,), "[M+NH4]+"))
    r = _chan(res, x, 1, CH.REAGENT_LOSS)
    per = np.array([300 / 400, 360 / 400, 240 / 300])
    assert r["n_scans"] == 3 and r["share_sd"] == pytest.approx(per.std(ddof=1))
    assert r["share"] == pytest.approx(300 / (300 + 200 / 3))   # the averaged spectrum's
    assert _chan(res, x, 10, CH.SKELETAL)["share_sd"] == pytest.approx(
        np.std([30 / 630, 50 / 630], ddof=1))
    one = _chan(res, x, 20, CH.REAGENT_LOSS)
    assert one["n_scans"] == 1 and math.isnan(one["share_sd"])
    # no per-scan tables: 0 / NaN
    _, plain = _run(round(_mz(x), 4), [(1, {x: 1000.0, rl: 300.0}), (10, {x: 600.0, rl: 600.0})],
                    (x,), "[M+NH4]+")
    assert (plain.channels["n_scans"] == 0).all() and plain.channels["share_sd"].isna().all()


def test_an_unnamed_co_isolate_is_said():
    x = "C11H12NO3+"
    p = round(_mz(x), 4)
    steps = [(e, {x: i, "C11H9O3+": 1000.0 - i})
             for e, i in zip(NCE, (800.0, 600.0, 300.0, 100.0))]
    t = _synthetic(p, steps)
    for share, said in ((0.15, True), (0.6, False)):
        t.window = S._frame({"sample_id": ["S", "S"], "parent_mz": [p, p],
                             "mz": [_mz(x), p + 0.056], "offset_da": [0.0, 0.056],
                             "intensity": [share, 1.0 - share], "share": [share, 1.0 - share],
                             "resolution": [NAN, NAN], "signal_to_noise": [NAN, NAN],
                             "is_parent": [True, False]}, S.WINDOW_COLUMNS)
        res = AN.analyze_run(t, _ions(x), adducts=_adducts((x,), "[M+NH4]+"))
        note = _cls(res, x)["note"]
        assert (f"an MS1 centroid no named ion matches, at m/z {p + 0.056:.4f}, holds "
                f"{1 - share:.0%} of the isolation window, more than {x} ({share:.0%})"
                in note) is said, share


# ------------------------------------------------------ adduct precedence ----
def test_adduct_precedence():
    ions = ["C10H20^N+", "C10H17+", "C12H13N2O5+"]
    nh4 = PR.resolve("NH4_15N")
    out, notes = CH.assign_adducts(ions, reagent=nh4)
    assert [(out[f].label, out[f].source) for f in ions[:2]] == \
        [("[M+^NH4]+", "--reagent NH4_15N"), ("[M+H]+", "--reagent NH4_15N")]
    assert out["C12H13N2O5+"].label == "[M+H]+"
    # read as [M+H]+ by elimination: said (a urea adduct reads covalent so)
    assert notes == ["[M+H]+ by elimination for C10H17+, C12H13N2O5+: none of the reagent "
                     "NH4_15N's forms that add a group fits, and the form adds none (class "
                     "'covalent'); an adduct of another reagent ion (NH4+ in urea data, 14N in "
                     "15N data) reads so -- give its adduct in the ions CSV"]
    # the uronium profile's first channel is [M+H]+: its own adduct is tried first
    assert CH.reagent_adducts(PR.resolve("Ur"))[:2] == ["[M+(CH4N2O)H]+", "[M+H]+"]
    out, _ = CH.assign_adducts(ions, reagent=PR.resolve("Ur"))
    assert out["C12H13N2O5+"].label == "[M+(CH4N2O)H]+" and out["C10H17+"].label == "[M+H]+"
    # the CSV's label wins, then --adduct, then the reagent's; one that does
    # not fit an ion is left to the reagent, else none (said)
    out, notes = CH.assign_adducts(ions, {"C10H17+": "[M+H]+"}, "[M+^NH4]+")
    assert [(out[f].label if out[f] else None) for f in ions] == ["[M+^NH4]+", "[M+H]+", None]
    assert [out[f].source for f in ions[:2]] == ["--adduct", "ions csv"]
    assert notes == ["--adduct [M+^NH4]+ does not fit: C12H13N2O5+ cannot be [M+^NH4]+: it does "
                     "not hold the group H4^N (^N -1 in M)",
                     "no adduct for C12H13N2O5+: analysed, its products read as 'product' only "
                     "and its class 'unknown'"]
    out, notes = CH.assign_adducts(ions, {}, "[M+^NH4]+", reagent=PR.resolve("Ur"))
    assert out["C12H13N2O5+"].source == "--reagent Ur" and "were tried instead" in notes[0]
    with pytest.raises(ValueError, match="does not hold the group"):
        CH.assign_adducts(ions, {"C10H17+": "[M+NH4]+"})
    with pytest.raises(ValueError, match="no group formula"):
        CH.assign_adducts(ions, {}, "[M+Q]+")


def test_the_ions_csv_adduct_column(tmp_path):
    p = tmp_path / "ions.csv"
    p.write_text("formula,mz,adduct\nC10H20^N+,155.156,[M+^NH4]+\nC10H17+,,\n")
    assert AN.read_ions_adducts(str(p)) == {"C10H20^N+": "[M+^NH4]+"}
    assert [f for f, _, _ in AN.read_ions_csv(str(p))] == ["C10H20^N+", "C10H17+"]
    p.write_text("formula,adduct\nC10H17+,[M+Na]+\n")
    with pytest.raises(ValueError, match="row 2: adduct '\\[M\\+Na\\]\\+' has no group formula"):
        AN.read_ions_adducts(str(p))
    p.write_text("formula,mz\nC10H17+,137.13\n")
    assert AN.read_ions_adducts(str(p)) == {}


# ------------------------------------------------- own data, known answers ----
@pytest.fixture(scope="module")
def mta():
    ions = _ions(*MT_IONS)
    ads, _ = CH.assign_adducts(list(ions), reagent=PR.resolve("NH4_15N"))
    return AN.analyze_run(_tables("ms2_monoterpene_a"), ions), \
        AN.analyze_run(_tables("ms2_monoterpene_a"), ions, adducts=ads)


@pytest.fixture(scope="module")
def nh4():
    ions = _ions(*NH4_IONS)
    ads, _ = CH.assign_adducts(list(ions), reagent=PR.resolve("NH4_15N"))
    return AN.analyze_run(_tables("ms2_nh4_exploratory"), ions), \
        AN.analyze_run(_tables("ms2_nh4_exploratory"), ions, adducts=ads)


def test_standard_a_known_answers(mta):
    _, res = mta
    r = _cls(res, "C10H17+")
    assert (r["class"], r["adduct"], r["share_skeletal"]) == (CH.CLASS_COVALENT, "[M+H]+", 1.0)
    sk = _chan(res, "C10H17+", 20, CH.SKELETAL)["products"].split("; ")
    assert "C6H9+" in sk and "C5H7+" in sk
    for parent in (155.1559, 155.1429):
        r = _cls(res, "C10H20^N+", parent)
        assert (r["class"], r["adduct"], r["reagent_neutral"]) == \
            (CH.CLASS_ADDUCT, "[M+^NH4]+", "H3^N")
        assert (r["nce_lo"], r["nce_hi"]) == (1.0, 1.0)
        assert r["share_monomer"] == r["share_skeletal"] == 0
        assert _cls(res, "C10H19O+", parent)["class"] == CH.CLASS_COVALENT
    r = _cls(res, "C10H20^N+", 155.1559)
    assert r["share_reagent_loss"] == pytest.approx(0.89583, abs=1e-5)
    assert r["share_reagent_loss_small"] == pytest.approx(0.10417, abs=1e-5)
    assert r["frag"] == pytest.approx(0.28686, abs=1e-5)
    assert (r["e50_curve"], r["e50_ecm"]) == ("F", pytest.approx(0.85491, abs=1e-5))
    assert _chan(res, "C10H20^N+", 1, CH.REAGENT_LOSS, 155.1559)["products"] == "C10H17+"
    assert _chan(res, "C10H20^N+", 1, CH.REAGENT_LOSS_SMALL, 155.1559)["products"] == "C6H9+"


def test_exploratory_file_known_answers(nh4):
    _, res = nh4
    c = res.classes
    assert set(c.loc[c["class"] == CH.CLASS_ADDUCT, "ion"]) == set(NH4_ADDUCTS)
    assert (c.loc[c["adduct"] == "[M+^NH4]+", "class"] == CH.CLASS_ADDUCT).all()
    assert (c.loc[c["adduct"] == "[M+H]+", "class"] == CH.CLASS_COVALENT).all()
    assert len(c) == len(NH4_IONS)
    r = _cls(res, "C12H22^NO2+")
    assert r["share_reagent_loss_small"] == pytest.approx(0.57569, abs=1e-5)
    assert r["share_skeletal"] == pytest.approx(0.42431, abs=1e-5)
    assert _chan(res, "C12H22^NO2+", 1, CH.SKELETAL)["products"] == "C12H20^NO+; C12H18^N+"
    assert r["evidence"].endswith("it sheds no monomer (its monomer . reagent ions at most 0 % of "
                                  "ion + products): M is no cluster of two neutrals")
    r = _cls(res, "C8H28^NO4Si4+")
    assert (r["share_reagent_loss"], r["e50_ecm"]) == (1.0, pytest.approx(0.37503, abs=1e-5))
    # the 197.130 -> 179.107 -> 161.096 cascade: reagent loss, then + H2O
    assert _chan(res, "C11H18^NO2+", 1, CH.REAGENT_LOSS)["products"] == "C11H15O2+"
    assert _chan(res, "C11H18^NO2+", 1, CH.REAGENT_LOSS_SMALL)["products"] == "C11H13O+"
    # every adduct falls to half below the energy test, the latest at 0.80 eV
    # (C4H10^NO4+, on S); C9H14^N+ is read at one energy, below half there:
    # it falls to half by that energy's E_cm (an upper bound)
    e = c[c["class"] == CH.CLASS_ADDUCT].set_index("ion")
    assert e["e50_ecm"].max() == pytest.approx(0.7976, abs=1e-4)
    assert e["e50_ecm"].idxmax() == "C4H10^NO4+"
    assert (e.loc["C9H14^N+", "e50_flag"], e.loc["C9H14^N+", "e50_ecm"]) == \
        ("single_energy", pytest.approx(0.68694, abs=1e-5))


def test_the_other_tables_do_not_change_with_adducts(mta, nh4):
    for plain, ad in (mta, nh4):
        for name in ("readings", "breakdown", "e50", "fragments", "branching"):
            assert getattr(plain, name).equals(getattr(ad, name)), name
        assert plain.channels is None and plain.classes is None
        assert list(ad.channels.columns) == list(CH.CHANNEL_COLUMNS)
        assert list(ad.classes.columns) == list(CH.CLASS_COLUMNS)
        assert ad.params["channels"]["on"] and not plain.params["channels"]["on"]
        assert [w.status for w in plain.windows] == [w.status for w in ad.windows]
        assert CH.CLASS_COLUMNS["class"] == "str" and CH.CHANNEL_COLUMNS["share"] == "float64"
        for got, spec in ((ad.channels, CH.CHANNEL_COLUMNS), (ad.classes, CH.CLASS_COLUMNS)):
            assert got.dtypes.equals(CH._frame([], spec).dtypes)


# ------------------------------------------------------ the class, edges ----
def _ms1(p, rows):
    """An MS1 window table for key `p` from (m/z, share[, S/N]) rows."""
    rows = [tuple(r) + (NAN,) * (3 - len(r)) for r in rows]
    return S._frame({"sample_id": ["S"] * len(rows), "parent_mz": [p] * len(rows),
                     "mz": [r[0] for r in rows], "offset_da": [r[0] - p for r in rows],
                     "intensity": [r[1] for r in rows], "share": [r[1] for r in rows],
                     "resolution": [NAN] * len(rows), "signal_to_noise": [r[2] for r in rows],
                     "is_parent": [abs(r[0] - p) < 1e-3 for r in rows]}, S.WINDOW_COLUMNS)


def test_a_skeletal_ion_that_falls_apart_like_an_adduct_is_not_covalent():
    """Its products keep the reagent (skeletal), but it falls to half by E_cm
    0.31 eV -- below SKELETAL_ECM, where no covalent skeleton of ours breaks
    (a co-isolate's or background products read as its own?): 'unknown',
    said; the same products crossing above it stay 'covalent'."""
    x = "C19H36NO7+"
    steps = [(e, {x: i, "C19H34NO6+": a, "C18H36NO5+": b})
             for e, i, a, b in zip(NCE, (600.0, 200.0, 80.0, 30.0), (300.0, 600.0, 600.0, 500.0),
                                   (100.0, 200.0, 300.0, 400.0))]
    _, res = _run(round(_mz(x), 4), steps, (x,), "[M+NH4]+")
    r = _cls(res, x)
    assert r["share_skeletal"] == 1.0 and r["e50_ecm"] < 0.32
    assert r["class"] == CH.CLASS_UNKNOWN
    assert r["evidence"].startswith("keeps the reagent and fragments its skeleton")
    assert f"below {CH.SKELETAL_ECM:g} eV" in r["evidence"]
    # the boundary, from both sides: a C11 skeletal ion crossing on F at
    # 0.70 eV is no covalent call, at 0.78 eV it is
    y = "C11H12^NO3+"
    for f10, want in ((0.47, CH.CLASS_UNKNOWN), (0.62, CH.CLASS_COVALENT)):
        steps = [(e, {y: 1000.0 * f, "C11H10^NO2+": 1000.0 * (1 - f)})
                 for e, f in ((1, 0.8), (10, f10), (20, 0.2), (30, 0.1))]
        _, res = _run(round(_mz(y), 4), steps, (y,), "[M+^NH4]+")
        r = _cls(res, y)
        assert r["class"] == want, (f10, r["e50_ecm"])
        assert abs(r["e50_ecm"] - CH.SKELETAL_ECM) < 0.1


def test_the_energy_test_reads_the_curves_not_a_missing_crossing():
    """'Holds half its signal past COVALENT_ECM' is read off the curves:
    one spectrum where F is 0.2 (single_energy), a crossing across an
    unobserved energy (gap), and an ion seen only at the lowest energies
    (above_range below COVALENT_ECM) hold nothing past it."""
    x, rl = "C11H12NO3+", "C11H9O3+"
    p = round(_mz(x), 4)
    _, res = _run(p, [(40, {x: 200.0, rl: 800.0})], (x,), "[M+NH4]+")
    r = _cls(res, x)
    assert r["class"] == CH.CLASS_ADDUCT and "holds half" not in r["evidence"]
    _, res = _run(p, [(1, {x: 900.0, rl: 100.0}), (10, {77.7: 50.0}), (20, {77.7: 50.0}),
                      (60, {x: 100.0, rl: 900.0})], (x,), "[M+NH4]+")
    r = _cls(res, x)
    assert r["class"] == CH.CLASS_ADDUCT and "never falls to half" not in r["evidence"]
    _, res = _run(p, [(1, {x: 1000.0}), (5, {x: 700.0}), (40, {77.7: 50.0}),
                      (60, {77.7: 50.0})], (x,), "[M+NH4]+")
    r = _cls(res, x)
    assert r["class"] == CH.CLASS_UNKNOWN
    assert "E_cm 1.91" not in r["evidence"]
    # the same ion holding F >= 0.5 at its one energy past COVALENT_ECM survives
    _, res = _run(p, [(40, {x: 990.0, rl: 10.0})], (x,), "[M+NH4]+")
    r = _cls(res, x)
    assert r["class"] == CH.CLASS_COVALENT and "its only energy" in r["evidence"]


def test_the_covalent_energy_test_at_its_boundary():
    """A reagent-loss ion crossing on F at 1.19 eV holds past COVALENT_ECM
    (covalent); one crossing at 1.01 eV does not (adduct)."""
    x = "C8H8NO2+"
    for i20, f20, want in ((500.0, 600.0, CH.CLASS_COVALENT), (250.0, 800.0, CH.CLASS_ADDUCT)):
        steps = [(1, {x: 1000.0, "C8H5O2+": 150.0}), (20, {x: i20, "C8H5O2+": f20}),
                 (40, {x: 100.0, "C8H5O2+": 900.0})]
        _, res = _run(round(_mz(x), 4), steps, (x,), "[M+NH4]+")
        r = _cls(res, x)
        assert r["class"] == want and r["e50_curve"] == "F"
        assert abs(r["e50_ecm"] - CH.COVALENT_ECM) < 0.11


def test_dominant_is_a_share_of_the_explained_products():
    """40 % unexplained (part of urea kept), 35 % reagent loss, 25 % skeletal:
    the reagent channels hold 0.35 / 0.6 = 58 % of the explained products --
    an adduct (of all products they would be 35 %: 'mixed')."""
    x = "C12H13N2O5+"
    steps = [(e, {x: i, "C12H10NO5+": u, "C11H9O4+": a, "C12H11N2O4+": s})
             for e, i, u, a, s in zip(NCE, (1000.0, 500.0, 200.0, 50.0),
                                      (160.0, 300.0, 300.0, 300.0),
                                      (140.0, 300.0, 400.0, 400.0), (100.0, 200.0, 300.0, 300.0))]
    _, res = _run(round(_mz(x), 4), steps, (x,), "[M+(CH4N2O)H]+")
    r = _cls(res, x)
    assert (r["share_unexplained"], r["share_reagent_loss"], r["share_skeletal"]) == \
        pytest.approx((0.4, 0.35, 0.25))
    assert r["class"] == CH.CLASS_ADDUCT


def test_a_cluster_is_read_on_its_population_at_any_energy_it_is_seen():
    """monomer_pop is the largest share monomer . reagent ions hold of ion +
    products at an energy the precursor is seen at: one more low energy (its
    products 15 % there) does not turn a cluster 'mixed'; and a two-energy
    window (NCE 1-10: 4 % at NCE 1, 60 % at NCE 10) reads it past the
    window, wherever the precursor is seen -- 95 % at NCE 20."""
    x, mono, mh = "C24H24NO4+", "C12H14NO2+", "C12H11O2+"
    steps = [(1, {x: 850.0, mono: 150.0}), (10, {x: 300.0, mono: 650.0, mh: 50.0}),
             (20, {x: 50.0, mono: 600.0, mh: 350.0}), (30, {x: 10.0, mono: 300.0, mh: 600.0})]
    for st in (steps, steps[1:]):
        _, res = _run(round(_mz(x), 4), st, (x,), "[M+NH4]+")
        r = _cls(res, x)
        assert r["class"] == CH.CLASS_CLUSTER, st[0][0]
        assert r["monomer_pop"] == pytest.approx(0.65)
    steps = [(1, {x: 960.0, mono: 40.0}), (10, {x: 400.0, mono: 600.0}),
             (20, {x: 50.0, mono: 950.0})]
    _, res = _run(round(_mz(x), 4), steps, (x,), "[M+NH4]+")
    r = _cls(res, x)
    assert (r["nce_lo"], r["nce_hi"]) == (1.0, 10.0)
    assert r["ecm_hi"] == pytest.approx(_ecm(10, x))              # the window's last energy
    # read where it is seen: NCE 20 (95 %) too
    assert r["class"] == CH.CLASS_CLUSTER and r["monomer_pop"] == pytest.approx(0.95)
    # the docstring says so
    doc = " ".join(test_a_cluster_is_read_on_its_population_at_any_energy_it_is_seen
                   .__doc__.split())
    assert "95 % at NCE 20" in doc and "reads it at its last energy" not in doc


def test_an_unseen_reagent_ion_needs_one_below_the_scan_range():
    """Only the precursor falls, nothing shows: an ammonium adduct's NH4+ (m/z
    18) is below every scan's first mass (FIRST_MASS_MIN), so it left as
    NH4+ (the test above); the uronium, nitrate and bromide ions (m/z 61-79)
    lie inside a scan range -- unseen, they argue against it: 'unknown'. With
    the scans' recorded first mass (ms2_scans.csv `low_mass`) above the
    reagent ion, it is below the range again."""
    steps = [(e, {"X": i}) for e, i in zip(NCE, (1000.0, 700.0, 400.0, 100.0))]
    for x, label in (("C12H13N2O5+", "[M+(CH4N2O)H]+"), ("C11H18^NO7-", "[M+^NO3]-"),
                     ("C11H18BrO4-", "[M+Br]-")):
        st = [(e, {x: d["X"]}) for e, d in steps]
        t, res = _run(round(_mz(x), 4), st, (x,), label)
        r = _cls(res, x)
        assert r["class"] == CH.CLASS_UNKNOWN, label
        assert r["e50_curve"] == "S" and r["e50_ecm"] < CH.COVALENT_ECM
    # the recorded first mass: 70 (above ^NO3-, 62.99), then 50 (below it)
    x = "C11H18^NO7-"
    t, _ = _run(round(_mz(x), 4), [(e, {x: d["X"]}) for e, d in steps], (x,), "[M+^NO3]-")
    sp = list(t.parents["spectrum"])
    for low, want in ((70.0, CH.CLASS_ADDUCT), (50.0, CH.CLASS_UNKNOWN)):
        t.scans = SC._frame({"scan": list(range(1, len(sp) + 1)), "order": [2.0] * len(sp),
                             "spectrum": sp, "low_mass": [low] * len(sp)}, SC.SCAN_COLUMNS)
        r = _cls(AN.analyze_run(t, _ions(x), adducts=_adducts((x,), "[M+^NO3]-")), x)
        assert r["class"] == want, low
        if want == CH.CLASS_ADDUCT:
            assert "lies below the scans' first mass (m/z 70.000)" in r["evidence"]


def test_an_unseen_reagent_ion_with_skeletal_products_is_no_adduct():
    """The products stay below FRAG_MIN, but every one keeps the reagent
    ([M+NH4-H2O]+): no sign it left as its reagent ion -- 'unknown', where
    the same products read as reagent loss leave it an adduct."""
    x = "C22H28NO9+"
    for prod, want in (("C22H26NO8+", CH.CLASS_UNKNOWN), ("C22H25O9+", CH.CLASS_ADDUCT)):
        steps = [(e, {x: i, prod: f}) for e, i, f in zip(NCE, (1000.0, 700.0, 400.0, 100.0),
                                                          (20.0, 30.0, 25.0, 6.0))]
        _, res = _run(round(_mz(x), 4), steps, (x,), "[M+NH4]+")
        assert _cls(res, x)["class"] == want, prod


def test_the_class_window_ends_where_the_precursor_is_last_seen():
    """A weak urea adduct seen at NCE 1-10 (falling), gone from NCE 20, whose
    only product -- the uronium ion -- shows after it is gone: no product
    while it is seen, so no channel decides ('unknown', said)."""
    x = "C21H39N2O6+"
    steps = [(1, {x: 120.0}), (5, {x: 80.0}), (10, {x: 40.0}), (20, {"CH5N2O+": 40.0}),
             (40, {"CH5N2O+": 40.0})]
    _, res = _run(round(_mz(x), 4), steps, (x,), "[M+(CH4N2O)H]+")
    r = _cls(res, x)
    assert r["class"] == CH.CLASS_UNKNOWN and r["nce_hi"] == 10.0
    assert "only after the precursor is gone (from NCE 20)" in r["evidence"]


def test_an_absent_named_ion_and_a_sliver_get_no_class(mta):
    """A named co-isolate the sample lacks (no precursor, no MS1 centroid)
    decides nothing: 'unknown', said; so is one never seen as a precursor
    whose every product is mostly another ion's (standard A's 137.13
    window: C9H14^N+ given 0.2 % of C10H17+'s fragments)."""
    x, z = "C11H12NO3+", "C12H16NO2+"
    p = round(_mz(x), 4)
    t = _synthetic(p, [(e, {x: i, "C11H9O3+": 1000.0 - i})
                       for e, i in zip(NCE, (900.0, 700.0, 400.0, 100.0))])
    t.window = _ms1(p, [(_mz(x), 1.0)])
    res = AN.analyze_run(t, _ions(x, z), adducts=_adducts((x, z), "[M+NH4]+"))
    r = _cls(res, z)
    assert r["class"] == CH.CLASS_UNKNOWN and "absent from this sample" in r["evidence"]
    assert _cls(res, x)["class"] == CH.CLASS_ADDUCT
    ions = _ions(*MT_IONS, "C9H14^N+")
    ads, _ = CH.assign_adducts(list(ions), reagent=PR.resolve("NH4_15N"))
    res = AN.analyze_run(_tables("ms2_monoterpene_a"), ions, adducts=ads)
    r = _cls(res, "C9H14^N+", 137.1321)
    assert r["class"] == CH.CLASS_UNKNOWN
    assert "never seen as a precursor" in r["evidence"]
    assert _cls(res, "C10H17+", 137.1321)["class"] == CH.CLASS_COVALENT


def test_a_named_ions_unexplained_products_that_another_ion_explains():
    """Two urea adducts in one window, each making its own [M+H]+: a shared
    fragment is split by survivor weights, and the share one ion reads as
    unexplained is the other's reagent loss -- 'another ion's product', kept
    out of UNEXPLAINED_MAX and of the explained products."""
    a, b = "C11H11N2O8+", "C12H15N2O7+"
    ra, rb = "C10H7O7+", "C11H11O6+"
    steps = [(e, {a: i, b: j, ra: 1000.0 - i, rb: (1000.0 - j) / 3})
             for e, i, j in zip(NCE, (700.0, 400.0, 200.0, 50.0), (250.0, 150.0, 80.0, 20.0))]
    t = _synthetic(round(_mz(a), 4), steps, width=0.4)
    res = AN.analyze_run(t, _ions(a, b), adducts=_adducts((a, b), "[M+(CH4N2O)H]+"))
    for ion in (a, b):
        r = _cls(res, ion)
        assert r["class"] == CH.CLASS_ADDUCT, (ion, r["evidence"])
        assert r["share_unexplained"] == 0 and r["share_other"] > 0
    o = _chan(res, b, 1, CH.OTHER_ION)
    assert o["products"] == ra
    # the products still add up to the breakdown's weighted sum
    for ion in (a, b):
        prod = res.channels[(res.channels["ion"] == ion) & (res.channels["channel"] != CH.INTACT)]
        want = res.breakdown[res.breakdown["ion"] == ion].set_index("nce")["frag_weighted"]
        assert np.allclose(prod.groupby("nce")["intensity"].sum(), want)


def test_the_share_spread_is_none_where_nothing_entering_it_is_read():
    """A spectrum whose every per-scan reading is 0 (each peak held at its
    averaged height in every scan) has no share spread, as F and S have
    none there (#82); nor one whose products are all held."""
    x = "C11H12NO3+"
    rl, sk = "C11H9O3+", "C11H10NO2+"
    steps = [(1, [{x: 1000.0, rl: 300.0, sk: 100.0}, {x: 1100.0, rl: 360.0, sk: 40.0},
                  {x: 900.0, rl: 240.0, sk: 60.0}]),
             (10, [{x: 600.0, rl: 600.0, sk: 30.0}, {x: 640.0, rl: 580.0, sk: 50.0},
                   {x: 620.0, rl: 560.0, sk: 70.0}]),
             (20, [{x: 300.0, rl: 700.0, sk: 40.0}, {x: 280.0, rl: 720.0, sk: 30.0}])]
    for held in (None, (x, rl, sk), (rl, sk)):
        t = _replicated(round(_mz(x), 4), steps, ions=(x,))
        if held:
            key = t.parents.loc[t.parents["nce"] == 10, "spectrum"].iloc[0]
            m = t.scan_peaks["spectrum"] == key
            m &= np.isin(np.round(t.scan_peaks["peak_mz"], 4), [round(_mz(f), 4) for f in held])
            t.scan_peaks.loc[m, "intensity"] = 0.0
        res = AN.analyze_run(t, _ions(x), adducts=_adducts((x,), "[M+NH4]+"), bootstrap=0)
        for c in (CH.REAGENT_LOSS, CH.SKELETAL):
            sd = _chan(res, x, 10, c)["share_sd"]
            assert (math.isnan(sd) if held else sd > 0.01), (held, c, sd)
        assert _chan(res, x, 1, CH.REAGENT_LOSS)["share_sd"] > 0.01


def test_a_radical_reagent_neutral_is_the_reagent_loss():
    """Superoxide and carbonate deprotonate an acid: [M-H]- with HO2 / HCO3
    lost is the reagent loss (a radical neutral), never 'skeletal'; HO2- and
    HCO3- themselves are the reagent ion."""
    o2 = CH.ion_adduct("C2H4O4-", "[M+O2]-")
    assert o2.reagent_neutral == "HO2"
    assert CH.product_channel(o2, "C2H3O2-") == (CH.REAGENT_LOSS, "")
    assert CH.product_channel(o2, "HO2-")[0] == CH.REAGENT_ION
    co3 = CH.ion_adduct("C6H8O7-", "[M+CO3]-")
    assert co3.reagent_neutral == "CHO3"
    assert CH.product_channel(co3, "C5H7O4-") == (CH.REAGENT_LOSS, "")
    assert CH.product_channel(co3, "C5H5O3-") == (CH.REAGENT_LOSS_SMALL, "")
    assert CH.product_channel(co3, "CHO3-")[0] == CH.REAGENT_ION
    x = "C6H8O7-"
    steps = [(e, {x: i, "C5H7O4-": f}) for e, i, f in zip(NCE, (700.0, 400.0, 200.0, 50.0),
                                                              (300.0, 600.0, 700.0, 600.0))]
    _, res = _run(round(_mz(x), 4), steps, (x,), "[M+CO3]-")
    r = _cls(res, x)
    assert r["class"] == CH.CLASS_ADDUCT and r["share_reagent_loss"] == 1.0


def test_an_in_source_form_reads_its_reagent_group():
    """[M+^NH4-H2O]+ is the ^NH4 adduct of M - H2O: its ^NH3 loss is the
    reagent loss, as for the same ion read as [M+^NH4]+."""
    a = CH.ion_adduct("C12H20^NO+", "[M+^NH4-H2O]+")
    assert (a.reagent_neutral, a.reagent_ion) == ("H3^N", "H4^N+")
    assert a.body == {"C": 12, "H": 16, "O": 1}
    assert CH.product_channel(a, "C12H17O+") == (CH.REAGENT_LOSS, "")
    assert CH.product_channel(a, "C12H15+") == (CH.REAGENT_LOSS_SMALL, "")
    b = CH.ion_adduct("C12H20^NO+", "[M+^NH4]+")
    for p in ("C12H17O+", "C12H15+", "C9H11+", "C10H16^NO+"):
        assert CH.product_channel(a, p)[0] == CH.product_channel(b, p)[0], p


def test_an_e50_read_off_the_nce_axis_is_said_only_when_it_is():
    """A crossing below the lowest energy, whose volts are known, is no E_cm
    read off the NCE axis (no note); an interpolated one between spectra
    with no volts is (said)."""
    x = "C11H12NO+"
    steps = [(30, {x: 300.0, "C11H9O+": 700.0}), (40, {x: 100.0, "C11H9O+": 900.0})]
    _, res = _run(round(_mz(x), 4), steps, (x,), "[M+NH4]+")
    r = _cls(res, x)
    assert r["e50_flag"] == "below_range" and "read off the NCE axis" not in r["note"]
    t, _ = _run(round(_mz(x), 4), [(1, {x: 900.0, "C11H9O+": 100.0}),
                                   (20, {x: 300.0, "C11H9O+": 700.0}),
                                   (40, {x: 100.0, "C11H9O+": 900.0})], (x,), "[M+NH4]+")
    t.parents.loc[t.parents["nce"] == 20, "hcd_v"] = NAN
    res = AN.analyze_run(t, _ions(x), adducts=_adducts((x,), "[M+NH4]+"))
    r = _cls(res, x)
    assert r["e50_flag"] == "interpolated" and "read off the NCE axis" in r["note"]


def test_the_e50_band_is_the_deciding_curves():
    """S decides (the precursor falls faster than its products rise: the
    reagent ion leaves the range): the class row carries S's band, not NaN."""
    x, rl = "C11H12NO3+", "C11H9O3+"
    steps = [(e, [{x: i * k, rl: f * k} for k in (0.9, 1.0, 1.1)])
             for e, i, f in ((1, 1000.0, 120.0), (8, 800.0, 150.0), (16, 400.0, 200.0),
                             (25, 150.0, 250.0))]
    t = _replicated(round(_mz(x), 4), steps, ions=(x,))
    res = AN.analyze_run(t, _ions(x), adducts=_adducts((x,), "[M+NH4]+"), bootstrap=200)
    r = _cls(res, x)
    e = res.e50[(res.e50["ion"] == x) & (res.e50["curve"] == "S")].iloc[0]
    assert r["e50_curve"] == "S" and np.isfinite(e["e50_ecm_lo"])
    assert (r["e50_ecm_lo"], r["e50_ecm_hi"]) == (e["e50_ecm_lo"], e["e50_ecm_hi"])


def test_the_unnamed_centroids_mz_is_read_through_the_axis_offset():
    """The m/z the note names is the one to name: through the folder's axis
    offset, so --ion takes it."""
    x, z = "C11H12NO3+", "C12H16NO2+"
    off = -28e-6
    p = round(_mz(x) * (1 + off), 4)
    steps = [(e, {_mz(x) * (1 + off): i, _mz("C11H9O3+") * (1 + off): 1000.0 - i})
             for e, i in zip(NCE, (800.0, 600.0, 300.0, 100.0))]
    t = _synthetic(p, steps)
    t.window = _ms1(p, [(_mz(x) * (1 + off), 0.15), (_mz(z) * (1 + off), 0.85)])
    res = AN.analyze_run(t, _ions(x), adducts=_adducts((x,), "[M+NH4]+"), mass_offset_ppm=-28)
    assert f"at m/z {_mz(z):.4f}, holds 85%" in _cls(res, x)["note"]
    AN.parse_ion_spec(f"{_mz(z):.4f}={z}")


def test_the_co_isolate_note_with_no_ms1_share_or_no_ms1_rows():
    """An ion the MS1 window does not see makes no claim against its 0; a
    window with no MS1 rows says a co-isolate is unknown."""
    x, z = "C11H12NO3+", "C12H16NO2+"
    p = round(_mz(x), 4)
    t = _synthetic(p, [(e, {x: i, "C11H9O3+": 1000.0 - i})
                       for e, i in zip(NCE, (800.0, 600.0, 300.0, 100.0))])
    t.window = _ms1(p, [(_mz(z), 1.0)])
    note = _cls(AN.analyze_run(t, _ions(x), adducts=_adducts((x,), "[M+NH4]+")), x)["note"]
    assert "more than" not in note and f"{x} is not seen in the MS1 window" in note
    t.window = S._frame({}, S.WINDOW_COLUMNS)
    note = _cls(AN.analyze_run(t, _ions(x), adducts=_adducts((x,), "[M+NH4]+")), x)["note"]
    assert "the MS1 window table holds no row for this key" in note


def test_the_co_isolate_note_reads_the_tolerance_and_the_ions_share():
    x = "C11H12NO3+"
    p = round(_mz(x), 4)
    t = _synthetic(p, [(e, {x: i, "C11H9O3+": 1000.0 - i})
                       for e, i in zip(NCE, (800.0, 600.0, 300.0, 100.0))])
    # a centroid 2.8 mDa off (12.8 ppm) is the ion's at --tol-mda 3
    t.window = _ms1(p, [(_mz(x), 0.3), (_mz(x) + 0.0028, 0.7)])
    res = AN.analyze_run(t, _ions(x), adducts=_adducts((x,), "[M+NH4]+"), tol_mda=3.0)
    assert "MS1 centroid no named ion matches" not in _cls(res, x)["note"]
    # an unnamed 40 % beside the ion's 30 %: said (more than the ion, less
    # than half)
    t.window = _ms1(p, [(_mz(x), 0.3), (p + 0.05, 0.4), (p - 0.05, 0.3)])
    res = AN.analyze_run(t, _ions(x), adducts=_adducts((x,), "[M+NH4]+"))
    assert "holds 40% of the isolation window, more than" in _cls(res, x)["note"]


# --------------------------------------------------- the channels, edges ----
def test_a_shed_half_holding_n_is_a_monomer_and_a_small_loss_is_small():
    """A 14N cluster whose shed half holds N (an organonitrate, an amide):
    the half that keeps the N with the reagent ion (two N) is monomer
    shedding, whatever the order -- M holds one N, so the other half has
    none to give up the reagent neutral with; the other half's Y . G is as well the
    deprotonated / protonated N-holding half -- composition cannot tell:
    'reagent loss + split', both readings named. A 'small' loss
    holds fewer than MONOMER_MIN_C carbons (C11H16O3 is no small loss)."""
    no3 = CH.ion_adduct("C22H33N2O12-", "[M+NO3]-")
    assert CH.product_channel(no3, "C11H16NO6-") == (
        CH.SPLIT, "C11H17NO6 + C11H16O3 or C11H16O3 + C11H17NO6")
    assert CH.product_channel(no3, "C11H17N2O9-") == (CH.MONOMER, "C11H17NO6 + C11H16O3")
    nh4 = CH.ion_adduct("C22H29N2O7+", "[M+NH4]+")
    assert CH.product_channel(nh4, "C13H20NO4+") == (
        CH.SPLIT, "C13H19NO4 + C9H6O3 or C13H16O4 + C9H9NO3")
    assert CH.product_channel(nh4, "C9H13N2O3+") == (CH.MONOMER, "C9H9NO3 + C13H16O4")
    # a protonated half (the reagent's proton kept): no small loss, and no
    # monomer . reagent ion -- 'reagent loss + split', both halves named
    assert CH.product_channel(DIMER, "C12H11O2+") == (CH.SPLIT, "C12H10O2 + C12H10O2")
    # a deep skeletal fragment of [M+H]+ is no small loss either
    big = CH.ion_adduct("C18H32NO4+", "[M+NH4]+")
    assert CH.product_channel(big, "C2H5O+") == (CH.UNEXPLAINED, "")
    # the class: a window of the N-holding half . NO3- alone is a cluster;
    # one of the other half's (the tie) is not
    x = "C22H33N2O12-"
    for prod, want in (("C11H17N2O9-", CH.CLASS_CLUSTER), ("C11H16NO6-", CH.CLASS_UNKNOWN)):
        steps = [(e, {x: i, prod: 1000.0 - i}) for e, i in zip(NCE, (300.0, 100.0, 50.0, 10.0))]
        _, res = _run(round(_mz(x), 4), steps, (x,), "[M+NO3]-")
        assert _cls(res, x)["class"] == want, prod


@pytest.mark.parametrize("ion, product", [
    ("C24H48NO4+", "C12H14NO2+"),         # its shed half C12H34O2: DBE -4
    ("C24H24NO4+", "C6H20NO+"),           # its kept half C6H16O: DBE -1
    ("C24H24NO4+", "C12H13NO2+."),        # two odd-electron halves
])
def test_a_shed_half_must_be_a_plausible_neutral(ion, product):
    assert CH.product_channel(CH.ion_adduct(ion, "[M+NH4]+"), product) == (CH.SKELETAL, "")


def test_a_strict_part_of_the_group_is_the_reagent_ion():
    ur = CH.ion_adduct("C12H13N2O5+", "[M+(CH4N2O)H]+")
    for p in ("CH2NO+", "H4N+"):            # protonated HNCO, NH4+: of urea alone
        assert CH.product_channel(ur, p) == (CH.REAGENT_ION, ""), p
    assert CH.product_channel(CH.ion_adduct("C11H10^NO5-", "[M+^NO3]-"), "^NO2-")[0] == \
        CH.REAGENT_ION


# ------------------------------------------ the adduct assignment, edges ----
def test_a_reagent_tries_its_group_forms_first():
    """A '+'-composed profile tries each component's own adduct, and every
    form that adds a reagent group before [M+H]+ / [M-H]-; of the group forms
    that fit, one whose M is closed-shell and holds none of the reagent's
    own element first ([M-H+I2]- over [M+I]- for A- . I2; [M+H]+ over
    [M]+. for an even-electron cation); an ion read as a group-less form by
    elimination is said."""
    assert CH.reagent_adducts(PR.resolve("NO3+NO3_15N")) == ["[M+NO3]-", "[M+^NO3]-", "[M-H]-"]
    assert CH.reagent_adducts(PR.resolve("Ur+NH4_15N"))[:2] == ["[M+(CH4N2O)H]+", "[M+^NH4]+"]
    assert CH.reagent_adducts(PR.resolve("Br")) == ["[M+Br]-", "[M+HBr+Br]-", "[M-H]-"]
    for prof, ion, want in (("NO3+NO3_15N", "C11H10^NO5-", "[M+^NO3]-"),
                            ("NO3+Br", "C5H8BrO4-", "[M+Br]-"),
                            ("Ur+NH4_15N", "C10H20^N+", "[M+^NH4]+"),
                            ("Ur+NH4_15N", "C12H13N2O5+", "[M+(CH4N2O)H]+"),
                            ("I", "C5H7I2O4-", "[M-H+I2]-"),
                            ("EasyIC", "C11H16+.", "[M]+."),
                            ("NO3", "C10H15NO11-", "[M+NO3]-")):           # a radical M kept
        out, _ = CH.assign_adducts([ion], reagent=PR.resolve(prof))
        assert out[ion].label == want, (prof, ion)
    # an even-electron cation is not read as a charge transfer with a radical M
    out, _ = CH.assign_adducts(["C10H17+"], reagent=PR.resolve("EasyIC"))
    assert out["C10H17+"].label != "[M]+." and CH._closed_shell(out["C10H17+"].body)
    out, notes = CH.assign_adducts(["C11H12NO3+", "C12H13N2O5+"], reagent=PR.resolve("Ur"))
    assert out["C11H12NO3+"].label == "[M+H]+"
    assert any("[M+H]+ by elimination for C11H12NO3+:" in n for n in notes)


def test_a_wrong_polarity_adduct_or_reagent_is_refused():
    with pytest.raises(ValueError, match="--adduct \\[M\\+NO3\\]- is an anion form; the named "
                                         "ions are cations"):
        CH.assign_adducts(["C11H12NO3+"], {}, "[M+NO3]-")
    with pytest.raises(ValueError, match="--reagent NO3 has no form of the named ions' charge"):
        CH.assign_adducts(["C11H12NO3+", "C10H17+"], reagent=PR.resolve("NO3"))
    # a miss of the right polarity carries ion_adduct's own reason
    _, notes = CH.assign_adducts(["C12H13N2O5+"], {}, "[M+^NH4]+")
    assert "it does not hold the group H4^N" in notes[0]


def test_the_evidence_words_the_case():
    """A reagent-ion product that holds past COVALENT_ECM is no 'protonated
    amide losing NH3' (that twin is an ammonium form's); 'sheds no monomer'
    is read on ion + products, not on the products."""
    x = "C7H4^NO5-"
    steps = [(e, {x: i, "^NO3-": f}) for e, i, f in ((1, 900.0, 100.0), (20, 800.0, 200.0),
                                                      (40, 700.0, 300.0), (60, 300.0, 700.0))]
    _, res = _run(round(_mz(x), 4), steps, (x,), "[M+^NO3]-")
    r = _cls(res, x)
    assert r["class"] == CH.CLASS_COVALENT and "amide" not in r["evidence"]
    assert "releases its reagent" in r["evidence"]
    # a C22 adduct whose monomer . NH4+ ions are at most 2 % of ion +
    # products (but 15 % of its products): it sheds no monomer
    x, mono, mh = "C22H28NO12+", "C11H16NO6+", "C22H25O12+"
    steps = [(e, {x: i, mono: m, mh: f}) for e, i, m, f in
             ((1, 1000.0, 8.0, 40.0), (10, 800.0, 20.0, 120.0), (20, 400.0, 15.0, 600.0))]
    _, res = _run(round(_mz(x), 4), steps, (x,), "[M+NH4]+")
    r = _cls(res, x)
    assert r["class"] == CH.CLASS_ADDUCT
    assert r["share_monomer"] > CH.NO_MONOMER and r["monomer_pop"] <= CH.NO_MONOMER
    assert "it sheds no monomer" in r["evidence"]


# ----------------------------------------------------- the wiring, edges ----
def test_the_wiring_of_inputs_notes_and_energies(tmp_path):
    p = tmp_path / "ions.csv"
    p.write_text("formula,adduct\nC10H20^N1+,[M+^NH4]+\nH19C10O+,[M+H]+\n")
    assert AN.read_ions_adducts(str(p)) == {"C10H20^N+": "[M+^NH4]+", "C10H19O+": "[M+H]+"}
    x = "C11H12NO3+"
    steps = [(e, {x: i, "C11H9O3+": 1000.0 - i})
             for e, i in zip(NCE, (800.0, 600.0, 300.0, 100.0))]
    t = _synthetic(round(_mz(x), 4), steps)
    res = AN.analyze_run(t, _ions(x), adducts=_adducts((x,), "[M+NH4]+"), gas_mass=39.9624,
                         adduct_notes=["a note of the adducts"])
    assert "a note of the adducts" in res.notes
    assert np.allclose(res.channels["e_cm"],
                       (4 + 0.2 * res.channels["nce"]) * 39.9624 / (39.9624 + _mz(x)))
    t = _synthetic(round(_mz(x), 4), steps, volts=False)
    res = AN.analyze_run(t, _ions(x), adducts=_adducts((x,), "[M+NH4]+"))
    assert "no volts known" in _cls(res, x)["note"]
    # the lowest peak of ANY spectrum: a peak below NH4+ in one spectrum only
    y = "C22H28NO9+"
    steps = [(e, {y: i, **({17.0: 5.0} if e == 1 else {})})
             for e, i in zip(NCE, (1000.0, 700.0, 400.0, 100.0))]
    _, res = _run(round(_mz(y), 4), steps, (y,), "[M+NH4]+")
    assert _cls(res, y)["class"] == CH.CLASS_UNKNOWN
    # a cluster with two splits: the largest product's halves
    d, m1, m2 = "C24H24NO4+", "C12H14NO2+", "C10H12NO2+"
    steps = [(1, {d: 200.0, m1: 1500.0, m2: 100.0}), (10, {d: 50.0, m2: 1200.0})]
    _, res = _run(round(_mz(d), 4), steps, (d,), "[M+NH4]+")
    assert _cls(res, d)["halves"] == "C12H10O2 + C12H10O2"


# ------------------------------------------------- the class, more edges ----
def _ecm(nce, ion):
    """The synthetic windows' E_cm: volts 4 + 0.2 NCE on N2."""
    return (4 + 0.2 * nce) * 28.0061 / (28.0061 + _mz(ion))


def test_a_skeletal_ion_not_known_to_reach_skeletal_ecm_is_no_covalent_call():
    """'Covalent' needs the ion KNOWN to hold half its signal to
    SKELETAL_ECM: a one-energy F below half (at or below that energy), a
    crossing across an unobserved energy, and a crossing at or below the
    lowest energy of a ladder that starts high are no such knowledge --
    'unknown', the bound said; one held to past 0.75 eV across a hole, or
    to the top of a ladder past it, is."""
    y, sk = "C11H12^NO3+", "C11H10^NO2+"
    p = round(_mz(y), 4)

    def cls(steps):
        return _cls(_run(p, steps, (y,), "[M+^NH4]+")[1], y)
    r = cls([(1, {y: 200.0, sk: 800.0})])
    assert (r["class"], r["share_skeletal"], r["e50_flag"]) == \
        (CH.CLASS_UNKNOWN, 1.0, "single_energy")
    assert (r["e50_nce"], r["e50_ecm"]) == (1.0, pytest.approx(_ecm(1, y)))
    assert "falls to half by E_cm 0.50 eV (F, at or below its only energy), below 0.75 eV" \
        in r["evidence"]
    r = cls([(1, {y: 870.0, sk: 130.0}), (3, {77.7: 50.0}), (5, {y: 200.0, sk: 800.0})])
    assert (r["class"], r["e50_flag"], r["e50_nce"]) == (CH.CLASS_UNKNOWN, "gap", 5.0)
    assert ("falls to half by E_cm 0.60 eV (F, across an unobserved energy, after E_cm 0.50 eV)"
            in r["evidence"])
    full = [(e, {y: 1000.0 * f, sk: 1000.0 * (1 - f)})
            for e, f in ((1, 0.62), (20, 0.4), (30, 0.2), (40, 0.1))]
    r = cls(full)
    assert (r["class"], r["e50_flag"]) == (CH.CLASS_UNKNOWN, "interpolated")
    assert r["e50_ecm"] < CH.SKELETAL_ECM
    r = cls(full[1:])                     # the same ladder from NCE 20: below half there
    assert (r["class"], r["e50_flag"]) == (CH.CLASS_UNKNOWN, "below_range")
    assert r["e50_ecm"] == pytest.approx(_ecm(20, y)) and r["e50_ecm"] > CH.SKELETAL_ECM
    assert ("not known to hold half its signal to 0.75 eV (it falls to half by E_cm 0.95 eV (F, "
            "at or below the lowest energy))") in r["evidence"]
    # known: half held to NCE 20 (0.95 eV) before a hole, or to the top
    r = cls([(20, {y: 900.0, sk: 100.0}), (30, {77.7: 50.0}), (40, {y: 200.0, sk: 800.0})])
    assert (r["class"], r["e50_flag"]) == (CH.CLASS_COVALENT, "gap")
    assert "holds half its signal to E_cm 0.95 eV, the last energy F is seen at half or more" \
        in r["evidence"]
    r = cls([(20, {y: 850.0, sk: 150.0}), (40, {y: 600.0, sk: 400.0})])
    assert r["class"] == CH.CLASS_COVALENT and r["e50_curve"] == ""
    # S falls to half below SKELETAL_ECM while F crosses above it: the
    # earlier of the two decides -- no covalent call
    r = cls([(1, {y: 1000.0, sk: 150.0}), (10, {y: 400.0, sk: 100.0}),
             (20, {y: 200.0, sk: 250.0}), (30, {y: 100.0, sk: 300.0})])
    assert (r["class"], r["e50_curve"]) == (CH.CLASS_UNKNOWN, "S")
    assert r["e50_ecm"] < CH.SKELETAL_ECM


def test_a_spectrum_with_no_volts_at_either_end_of_the_ladder():
    """HCD volts are linear in NCE at one m/z: a spectrum with no volts at
    the bottom or the top of the ladder (a one-scan spectrum of a scan with
    no trailer row) has its E_cm read off the line through the others
    (said), and F and S are compared on NCE, where both lie -- the curve
    that falls first decides, whatever volts are known."""
    x, rl, sm = "C11H12NO3+", "C11H9O3+", "C11H7O2+"
    p = round(_mz(x), 4)
    steps = [(1, {x: 1000.0, rl: 500.0, sm: 100.0}), (10, {x: 950.0, rl: 1100.0, sm: 200.0}),
             (20, {x: 850.0, rl: 1500.0, sm: 300.0}), (30, {x: 600.0, rl: 1800.0, sm: 400.0}),
             (40, {x: 300.0, rl: 1900.0, sm: 500.0})]
    t, res = _run(p, steps, (x,), "[M+NH4]+")
    known = _cls(res, x)
    assert (known["class"], known["e50_curve"]) == (CH.CLASS_ADDUCT, "F")
    t.parents.loc[t.parents["nce"] == 1, "hcd_v"] = NAN
    r = _cls(AN.analyze_run(t, _ions(x), adducts=_adducts((x,), "[M+NH4]+")), x)
    assert (r["class"], r["e50_curve"], r["e50_nce"]) == \
        (CH.CLASS_ADDUCT, "F", pytest.approx(known["e50_nce"]))
    assert r["e50_ecm"] == pytest.approx(known["e50_ecm"])
    assert "read off the NCE axis" in r["note"] and "first on S" not in r["note"]
    assert r["ecm_hi"] == pytest.approx(_ecm(1, x))
    # below half at a lowest spectrum with no volts: F decides, at that energy
    t, _ = _run(p, [(1, {x: 300.0, rl: 700.0}), (10, {x: 250.0, rl: 900.0}),
                    (20, {x: 100.0, rl: 950.0})], (x,), "[M+NH4]+")
    t.parents.loc[t.parents["nce"] == 1, "hcd_v"] = NAN
    r = _cls(AN.analyze_run(t, _ions(x), adducts=_adducts((x,), "[M+NH4]+")), x)
    assert (r["e50_curve"], r["e50_flag"], r["e50_nce"]) == ("F", "below_range", 1.0)
    assert r["e50_ecm"] == pytest.approx(_ecm(1, x))
    # the top spectrum with no volts: an ion that survives still holds to it
    y = "C11H12NO+"
    st = [(e, {y: 1000.0, "C11H9O+": f}) for e, f in ((1, 2.0), (20, 5.0), (40, 10.0), (60, 20.0))]
    t, _ = _run(round(_mz(y), 4), st, (y,), "[M+NH4]+")
    t.parents.loc[t.parents["nce"] == 60, "hcd_v"] = NAN
    r = _cls(AN.analyze_run(t, _ions(y), adducts=_adducts((y,), "[M+NH4]+")), y)
    assert r["class"] == CH.CLASS_COVALENT
    assert (f"holds half its signal to E_cm {_ecm(60, y):.2f} eV, read off the NCE axis, the "
            "highest energy it is seen at") in r["evidence"]
    # no volts at all: F and S are still compared on NCE -- S falls first
    t = _synthetic(p, [(1, {x: 1000.0}), (10, {x: 300.0, rl: 100.0}), (20, {x: 200.0, rl: 400.0}),
                       (30, {x: 100.0, rl: 800.0})], volts=False)
    r = _cls(AN.analyze_run(t, _ions(x), adducts=_adducts((x,), "[M+NH4]+")), x)
    assert (r["e50_curve"], r["e50_nce"]) == ("S", pytest.approx(1 + 9 * 0.5 / 0.7))
    assert math.isnan(r["e50_ecm"]) and "no volts known" in r["note"]


def test_a_never_seen_ion_given_only_an_equal_split_gets_no_class(mta):
    """An equal split (weight_source 'equal': no MS1 window to weigh a
    never-seen co-isolate by) is a prior, no observation: the co-isolate's
    0.5 of the target's fragment is no product of its own -- 'unknown', said
    (with three candidates it would be 1/3: the class must not hang on how
    many ions are named)."""
    x, z = "C11H12NO3+", "C12H16NO2+"
    rl, sm = "C11H9O3+", "C11H7O2+"
    steps = [(e, {x: i, rl: a, sm: b})
             for e, i, a, b in zip(NCE, (1000.0, 600.0, 300.0, 100.0), (300.0, 600.0, 700.0, 600.0),
                                   (60.0, 150.0, 300.0, 400.0))]
    _, res = _run(round(_mz(x), 4), steps, (x, z), "[M+NH4]+")
    fr = res.fragments.set_index("fragment").loc[sm]
    assert fr["weight_source"] == "equal" and dict(
        kv.split("=") for kv in fr["weights"].split(";")) == {x: "0.5", z: "0.5"}
    r = _cls(res, z)
    assert r["class"] == CH.CLASS_UNKNOWN
    assert "every product it is given is an equal split" in r["evidence"]
    assert _cls(res, x)["class"] == CH.CLASS_ADDUCT
    # standard A with no MS1 window table: C9H14^N+ (named in the 137.13
    # window, never seen there) is given an equal half of C10H17+'s fragments
    ions = _ions(*MT_IONS, "C9H14^N+")
    ads, _ = CH.assign_adducts(list(ions), reagent=PR.resolve("NH4_15N"))
    t = _tables("ms2_monoterpene_a")
    t.window = S._frame({}, S.WINDOW_COLUMNS)
    r = _cls(AN.analyze_run(t, ions, adducts=ads), "C9H14^N+", 137.1321)
    assert r["class"] == CH.CLASS_UNKNOWN and "equal split" in r["evidence"]
    # a measured half (their MS1 shares equal) is no product mostly its own
    t = _synthetic(round(_mz(x), 4), steps)
    t.window = _ms1(round(_mz(x), 4), [(_mz(x), 0.5), (_mz(z), 0.5)])
    res = AN.analyze_run(t, _ions(x, z), adducts=_adducts((x, z), "[M+NH4]+"))
    assert res.fragments.set_index("fragment").loc[sm, "weight_source"] == \
        "ms2 survivors + ms1 window"
    r = _cls(res, z)
    assert r["class"] == CH.CLASS_UNKNOWN
    assert "its largest weight 0.5, at most 0.5" in r["evidence"]


def test_an_ammonium_adduct_that_leaves_as_nh4_is_read_on_s():
    """No product at all, its precursor falling (S) while F stays 1 to a top
    energy past COVALENT_ECM: S's crossing (0.32 eV) is the most it holds
    half its signal to -- an adduct, not an ion that 'survives'."""
    x = "C22H28NO9+"
    steps = [(e, {x: i}) for e, i in zip((1, 10, 30, 60, 90), (1000.0, 300.0, 200.0, 150.0, 120.0))]
    _, res = _run(round(_mz(x), 4), steps, (x,), "[M+NH4]+")
    r = _cls(res, x)
    assert (r["class"], r["e50_curve"]) == (CH.CLASS_ADDUCT, "S")
    assert _ecm(90, x) > CH.COVALENT_ECM and "holds half" not in r["evidence"]
    e50 = res.e50[res.e50["ion"] == x]
    axis = res.channels[res.channels["ion"] == x].drop_duplicates("spectrum")[["nce", "e_cm"]]
    v, how = CH._holds(e50, axis)
    assert v == pytest.approx(r["e50_ecm"]) and how.endswith("where S falls to half")


def test_the_class_branches_with_no_dominant_channel():
    """Of the explained products: [Y+H]+ halves alone (a cluster's or a
    covalent dimer's, after the reagent's loss) -- 'unknown'; no channel at
    0.5 -- 'mixed'; every product another named ion's -- 'unknown', and the
    run goes on."""
    x, rl, sk, mono, sp = "C24H24NO4+", "C24H21O4+", "C24H22NO3+", "C12H14NO2+", "C12H11O2+"
    p = round(_mz(x), 4)
    _, res = _run(p, [(1, {x: 800.0, sp: 200.0}), (10, {x: 400.0, sp: 600.0})], (x,), "[M+NH4]+")
    r = _cls(res, x)
    assert (r["class"], r["share_split"]) == (CH.CLASS_UNKNOWN, 1.0)
    assert r["evidence"].startswith("its products are halves of M split after the reagent's loss")
    _, res = _run(p, [(1, {x: 600.0, rl: 160.0, sk: 120.0, mono: 120.0}),
                      (10, {x: 300.0, rl: 320.0, sk: 240.0, mono: 240.0})], (x,), "[M+NH4]+")
    r = _cls(res, x)
    assert r["class"] == CH.CLASS_MIXED
    assert (r["share_reagent_loss"], r["share_skeletal"], r["share_monomer"]) == \
        pytest.approx((0.4, 0.3, 0.3))
    assert r["evidence"].startswith("no channel holds 0.5 of the explained products")
    a, b, ra = "C11H11N2O8+", "C12H15N2O7+", "C10H7O7+"
    steps = [(e, {a: i, b: j, ra: 1000.0 - i})
             for e, i, j in zip(NCE, (700.0, 400.0, 200.0, 50.0), (250.0, 150.0, 80.0, 20.0))]
    t = _synthetic(round(_mz(a), 4), steps, width=0.4)
    res = AN.analyze_run(t, _ions(a, b), adducts=_adducts((a, b), "[M+(CH4N2O)H]+"))
    r = _cls(res, b)
    assert (r["class"], r["share_other"]) == (CH.CLASS_UNKNOWN, 1.0)
    # none of its own: another ion's products open no class window
    assert r["evidence"].startswith("barely fragments: its own products (another named ion's "
                                    "left out) never reach 0.1")
    assert _cls(res, a)["class"] == CH.CLASS_ADDUCT


def test_the_adduct_evidence_says_what_m_sheds():
    """An adduct of an M that could split: its monomer . reagent ions above
    NO_MONOMER of ion + products are said ('sheds monomers too'); none, but
    [Y+H]+ halves above it, are said as halves split after the reagent's
    loss."""
    x, rl, mono, sp = "C24H24NO4+", "C24H21O4+", "C12H14NO2+", "C12H11O2+"
    p = round(_mz(x), 4)
    for extra, said in ((mono, "it sheds monomers too (up to 10 % of ion + products, NCE 10)"),
                        (sp, "but up to 7 % of ion + products are halves split after the "
                             "reagent's loss")):
        _, res = _run(p, [(1, {x: 1000.0, rl: 300.0, extra: 100.0}),
                          (10, {x: 500.0, rl: 800.0, extra: 150.0})], (x,), "[M+NH4]+")
        r = _cls(res, x)
        assert r["class"] == CH.CLASS_ADDUCT and said in r["evidence"], extra


def test_a_cluster_shed_after_the_precursor_is_gone_is_no_cluster():
    """monomer_pop is read only where the precursor is seen: monomer .
    reagent ions alone once it is gone tell what became of its products,
    not of the ion -- it stays 'mixed'."""
    x, mono = "C22H28NO7+", "C11H16NO4+"
    steps = [(1, {x: 1000.0, mono: 150.0}), (10, {x: 700.0, mono: 160.0}),
             (20, {mono: 600.0}), (30, {mono: 400.0})]
    _, res = _run(round(_mz(x), 4), steps, (x,), "[M+NH4]+")
    r = _cls(res, x)
    assert r["class"] == CH.CLASS_MIXED
    assert r["monomer_pop"] == pytest.approx(160 / 860)


def test_an_unseen_reagent_ion_needs_s_to_fall_first_and_by_covalent_ecm():
    """The unseen-reagent rule reads S: products that show only after the
    precursor is gone make F fall with S (a tie, F's) -- 'unknown', said;
    and S falling to half above COVALENT_ECM is no adduct's fall."""
    x, rl = "C22H28NO9+", "C22H25O9+"
    _, res = _run(round(_mz(x), 4), [(1, {x: 1000.0}), (10, {rl: 500.0}), (20, {rl: 300.0})],
                  (x,), "[M+NH4]+")
    r = _cls(res, x)
    assert (r["class"], r["e50_curve"]) == (CH.CLASS_UNKNOWN, "F")
    assert "only after the precursor is gone (from NCE 10)" in r["evidence"]
    # S falls to half at NCE 85 (past COVALENT_ECM), F never in what it is
    # seen at (to NCE 60): no adduct
    steps = [(e, {x: i}) for e, i in zip((1, 30, 60, 90), (1000.0, 900.0, 800.0, 300.0))]
    _, res = _run(round(_mz(x), 4), steps, (x,), "[M+NH4]+")
    ch = res.channels[res.channels["ion"] == x]
    e50 = res.e50[res.e50["ion"] == x].copy()
    e50.loc[e50["curve"] == "F", ["flag", "nce_max"]] = ["above_range", 60.0]
    e50.loc[e50["curve"] == "S", ["flag", "e50_nce", "e50_ecm"]] = \
        ["interpolated", 85.0, _ecm(85, x)]
    assert _ecm(85, x) > CH.COVALENT_ECM > _ecm(60, x)
    a = CH.ion_adduct(x, "[M+NH4]+")
    r = CH.classify(ch, e50, a, floor_mz=CH.FIRST_MASS_MIN, ion_mz=_mz(x))
    assert (r["class"], r["e50_curve"]) == (CH.CLASS_UNKNOWN, "S")
    e50.loc[e50["curve"] == "S", ["e50_nce", "e50_ecm"]] = [55.0, _ecm(55, x)]
    e50.loc[e50["curve"] == "F", "nce_max"] = 30.0
    assert CH.classify(ch, e50, a, floor_mz=CH.FIRST_MASS_MIN, ion_mz=_mz(x))["class"] == \
        CH.CLASS_ADDUCT
    # F and S with no data: observed in no spectrum, whatever the form
    e50["flag"] = "no_data"
    r = CH.classify(ch, e50, CH.ion_adduct(x, "[M+H]+"), ion_mz=_mz(x))
    assert r["class"] == CH.CLASS_UNKNOWN and "observed in no spectrum" in r["evidence"]


def test_the_scan_range_is_read_per_spectrum_of_the_window():
    """The first mass a reagent ion must lie below is the lowest one of the
    window's own spectra (`ms2_scans.csv` `low_mass`): one spectrum set to
    50 puts ^NO3- (62.99) inside the range; a scan of another key's
    spectrum set to 50 does not."""
    x = "C11H18^NO7-"
    steps = [(e, {x: i}) for e, i in zip(NCE, (1000.0, 700.0, 400.0, 100.0))]
    t, _ = _run(round(_mz(x), 4), steps, (x,), "[M+^NO3]-")
    sp = list(t.parents["spectrum"])
    for low, extra, want in (([50.0, 70.0, 70.0, 70.0], (), CH.CLASS_UNKNOWN),
                             ([70.0] * 4, (50.0,), CH.CLASS_ADDUCT)):
        spec = sp + ["S|999.0000|hcd1.00"] * len(extra)
        t.scans = SC._frame({"scan": list(range(1, len(spec) + 1)), "order": [2.0] * len(spec),
                             "spectrum": spec, "low_mass": list(low) + list(extra)},
                            SC.SCAN_COLUMNS)
        r = _cls(AN.analyze_run(t, _ions(x), adducts=_adducts((x,), "[M+^NO3]-")), x)
        assert r["class"] == want, (low, extra)


def test_a_co_isolate_with_no_adduct_explains_no_product():
    """A product one ion reads as unexplained is 'another ion's product'
    only where the other ion reads it in a channel of its own -- an ion with
    no adduct reads every product as 'product', which explains nothing."""
    a, b = "C11H11N2O8+", "C12H15N2O7+"
    ra, rb = "C10H7O7+", "C11H11O6+"
    steps = [(e, {a: i, b: j, ra: 1000.0 - i, rb: (1000.0 - j) / 3})
             for e, i, j in zip(NCE, (700.0, 400.0, 200.0, 50.0), (250.0, 150.0, 80.0, 20.0))]
    t = _synthetic(round(_mz(a), 4), steps, width=0.4)
    res = AN.analyze_run(t, _ions(a, b), adducts={a: CH.ion_adduct(a, "[M+(CH4N2O)H]+"),
                                                  b: None})
    r = _cls(res, a)
    assert r["share_other"] == 0 and r["share_unexplained"] > 0
    assert _chan(res, a, 1, CH.UNEXPLAINED)["products"] == rb


def test_the_evidence_of_a_form_with_no_reagent_group_names_no_proton():
    """An [M-H]- ion is no protonated dimer: the evidence says a dimer ion."""
    x = "C11H9O2-"
    steps = [(e, {x: i, "C11H7O-": 1000.0 - i}) for e, i in zip(NCE, (900.0, 700.0, 400.0, 100.0))]
    _, res = _run(round(_mz(x), 4), steps, (x,), "[M-H]-")
    r = _cls(res, x)
    assert r["class"] == CH.CLASS_COVALENT
    assert "a dimer ion shedding a monomer" in r["evidence"] and "protonated" not in r["evidence"]


def test_the_lowest_mh_crossing_is_quoted_as_measured(mta):
    """SKELETAL_ECM lies below the lowest F crossing of an [M+H]+ ion of our
    own files: standard A's C10H19O+, 0.799 eV in the 155.1559 window and
    0.800 in the 155.1429 one -- quoted so in the source and the docs; and no
    first mass is quoted for our own files but the one they hold (40)."""
    _, res = mta
    e = res.e50[(res.e50["curve"] == "F") & (res.e50["flag"] == "interpolated")]
    e = e[e["ion"].isin(["C10H17+", "C10H19O+"])]            # its [M+H]+ ions
    low = e.loc[e["e50_ecm"].idxmin()]
    assert (low["ion"], low["parent_mz"]) == ("C10H19O+", pytest.approx(155.1559))
    assert low["e50_ecm"] == pytest.approx(0.79884, abs=1e-5)
    assert low["e50_ecm"] > CH.SKELETAL_ECM
    other = e[(e["ion"] == "C10H19O+") & ~np.isclose(e["parent_mz"], 155.1559)]
    assert other["e50_ecm"].iloc[0] == pytest.approx(0.80007, abs=1e-5)
    root = Path(__file__).resolve().parents[1]
    src = " ".join(Path(CH.__file__).read_text(encoding="utf-8").split())
    doc = (root / "docs" / "MS2.md").read_text(encoding="utf-8")
    sec = " ".join(doc[doc.index("### 11.13 "):doc.index("## 12. ")].split())
    for text in (src, sec):
        assert "0.799" in text and "51.30" not in text
    assert "C10H19O+'s, 0.800 eV" not in src and "C10H19O⁺ 0.800 —" not in sec


# ---------------------------------------------- the channels, more edges ----
def test_m_itself_is_the_group_left_by_electron_transfer():
    """O2- . M -> M- + O2 (M's electron affinity the larger): M itself is
    the reagent's loss for a form whose group is a neutral molecule of its
    own (O2, Br2); a product M's own atoms make whole holds a group of M's
    elements by count only -- unexplained, no skeletal fragment; one with
    more O than M kept the reagent."""
    q = CH.ion_adduct("C10H6O4-", "[M+O2]-")                  # a quinone . O2-
    assert CH.product_channel(q, "C10H6O2-") == (CH.REAGENT_LOSS, "")
    assert CH.product_channel(q, "C10H6O3-") == (CH.SKELETAL, "")      # O3 > M's O2
    assert CH.product_channel(CH.ion_adduct("C10H6Br2O2-", "[M+Br2]-"), "C10H6O2-") == \
        (CH.REAGENT_LOSS, "")
    o = CH.ion_adduct("C11H8O6-", "[M+O2]-")                   # M C11H8O4
    assert CH.product_channel(o, "C10H6O3-") == (CH.UNEXPLAINED, "")    # a part of M
    # a group that is no neutral of its own: M- is a radical loss, as before
    no3 = CH.ion_adduct("C11H10^NO5-", "[M+^NO3]-")
    assert CH.product_channel(no3, "C11H10O2-")[0] == CH.UNEXPLAINED
    x = "C10H6O4-"
    steps = [(e, {x: i, "C10H6O2-": f}) for e, i, f in zip(NCE, (950.0, 700.0, 350.0, 100.0),
                                                            (50.0, 300.0, 650.0, 900.0))]
    _, res = _run(round(_mz(x), 4), steps, (x,), "[M+O2]-")
    r = _cls(res, x)
    assert (r["class"], r["share_reagent_loss"]) == (CH.CLASS_ADDUCT, 1.0)


def test_a_reagent_ion_with_its_own_neutral_is_the_reagent_ion():
    """An analyte clustered with the urea dimer ion can only be read as the
    urea adduct of M + urea: its Ur2H+ (m/z 121) holds the group and a
    whole urea -- the reagent's carbon, none of M's: 'reagent ion', as
    HCOO- . HCOOH of formate and CH3COO- . CH3COOH of acetate; a product
    with M's carbon besides is not."""
    x = "C13H17N4O5+"                                      # C11H8O3 . Ur2H+
    out, notes = CH.assign_adducts([x], reagent=PR.resolve("Ur"))
    a = out[x]
    assert (a.label, notes) == ("[M+(CH4N2O)H]+", [])
    assert CH.product_channel(a, "C2H9N4O2+") == (CH.REAGENT_ION, "")
    assert CH.product_channel(a, "C12H13N2O4+") == (CH.REAGENT_LOSS, "")   # C11H8O3 . UrH+
    assert CH.product_channel(a, "C13H15N4O4+") == (CH.SKELETAL, "")
    assert CH.product_channel(CH.ion_adduct("C12H11O4-", "[M+CHO2]-"), "C2H3O4-") == \
        (CH.REAGENT_ION, "")
    assert CH.product_channel(CH.ion_adduct("C12H9O4-", "[M+C2H3O2]-"), "C4H7O4-") == \
        (CH.REAGENT_ION, "")
    steps = [(e, {x: i, "C2H9N4O2+": 0.85 * (1000.0 - i), "C12H13N2O4+": 0.15 * (1000.0 - i)})
             for e, i in zip((1, 5, 10, 20, 40), (950.0, 850.0, 700.0, 450.0, 200.0))]
    _, res = _run(round(_mz(x), 4), steps, (x,), "[M+(CH4N2O)H]+")
    r = _cls(res, x)
    assert (r["class"], r["share_reagent_ion"]) == (CH.CLASS_ADDUCT, pytest.approx(0.85))


def test_a_loss_of_six_carbons_is_no_small_loss():
    """A small loss off [M+H]+ holds fewer than MONOMER_MIN_C carbons: [Y+H]+
    with a C6 neutral gone is a split, both halves named."""
    assert CH.product_channel(DIMER, "C18H15O3+") == (CH.SPLIT, "C18H14O3 + C6H6O")
    assert CH.product_channel(DIMER, "C19H17O3+")[0] == CH.REAGENT_LOSS_SMALL   # a C5 loss


def test_two_group_forms_that_fit_equally_are_said():
    """An ion two group forms fit equally well -- closed-shell M, none of
    the reagent's own element, as few labelled atoms in M -- is read as the
    first, said (a Br-bearing M with NO3- against an N-bearing M with Br-);
    one whose other reading puts the reagent's own element in M, or a 15N
    label, is no tie."""
    out, notes = CH.assign_adducts(["C11H9N^NO5-"], reagent=PR.resolve("NO3+NO3_15N"))
    assert out["C11H9N^NO5-"].label == "[M+^NO3]-"
    assert not any("equally" in n for n in notes)
    out, notes = CH.assign_adducts(["C6H5BrNO5-"], reagent=PR.resolve("NO3+Br"))
    assert out["C6H5BrNO5-"].label == "[M+NO3]-"
    assert any("several fit equally well: C6H5BrNO5- ([M+NO3]-; also [M+Br]-)" in n
               for n in notes)
    out, notes = CH.assign_adducts(["C11H11Br2O2-"], reagent=PR.resolve("Br"))
    assert out["C11H11Br2O2-"].label == "[M+HBr+Br]-"
    assert not any("equally" in n for n in notes)


def test_an_ions_csv_adduct_the_ion_cannot_be_names_its_row(tmp_path):
    p = tmp_path / "ions.csv"
    p.write_text("formula,adduct\nC10H20^N+,[M+^NH4]+\nC10H19O+,[M+H]+\nC10H17+,[M-H]-\n")
    with pytest.raises(ValueError, match="ions.csv row 4: C10H17\\+ is a cation; \\[M-H\\]- is an "
                                         "anion form"):
        AN.read_ions_adducts(str(p))
    p.write_text("formula,adduct\nH17C10+,[M+NH4]+\n")
    with pytest.raises(ValueError, match="ions.csv row 2: C10H17\\+ cannot be \\[M\\+NH4\\]\\+"):
        AN.read_ions_adducts(str(p))


# ------------------------------------------------------------------- CLI ----
@pytest.fixture
def no_server(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("ms2 analyze asked for Mascope")
    from peaky.io import io_mascope as IO
    monkeypatch.setattr(cli, "_require_creds", boom)
    monkeypatch.setattr(IO, "connect", boom)


def test_cli_reads_the_adducts_and_writes_channels_and_classes(tmp_path, no_server, capsys):
    run = _write_run(tmp_path / "run", "ms2_monoterpene_a")
    ions = tmp_path / "ions.csv"
    ions.write_text("formula,mz,adduct\nC10H17+,137.1323,\nC10H20^N+,155.1558,[M+^NH4]+\n"
                    "C10H19O+,155.1428,\n")
    out = tmp_path / "out"
    assert cli.main(["ms2", "analyze", "--run-dir", str(run), "--ions", str(ions),
                     "--adduct", "[M+H]+", "--no-figures", "--out-dir", str(out)]) == 0
    printed = capsys.readouterr().out
    names = {p.name for p in out.iterdir()}
    assert {"ms2_channels.csv", "ms2_classes.csv"} <= names
    for name, spec in (("ms2_channels.csv", CH.CHANNEL_COLUMNS),
                       ("ms2_classes.csv", CH.CLASS_COLUMNS)):
        assert list(pd.read_csv(out / name, nrows=0).columns) == list(spec), name
    cls = pd.read_csv(out / "ms2_classes.csv", keep_default_na=False, na_values=[""])
    got = {(r.parent_mz, r.ion): (r.adduct, r.adduct_source, r._9) for r in cls.itertuples()}
    assert got[(155.1559, "C10H20^N+")] == ("[M+^NH4]+", "ions csv", "adduct")
    assert got[(137.1321, "C10H17+")] == ("[M+H]+", "--adduct", "covalent")
    assert "  class: C10H20^N+ [M+^NH4]+ adduct; C10H19O+ [M+H]+ covalent" in printed
    man = json.loads((out / "ms2_analyze_manifest.json").read_text())
    ch = man["parameters"]["channels"]
    assert ch == {"on": True, "monomer_min_c": 6, "frag_min": 0.1, "dominant": 0.5,
                  "cluster_pop": 0.4, "unexplained_max": 0.5, "covalent_ecm": 1.1,
                  "skeletal_ecm": 0.75, "no_monomer": 0.05, "first_mass_min": 40.0,
                  "own_weight": 0.5}
    assert {i["formula"]: (i["adduct"], i["adduct_source"]) for i in man["parameters"]["ions"]} \
        == {"C10H17+": ("[M+H]+", "--adduct"), "C10H20^N+": ("[M+^NH4]+", "ions csv"),
            "C10H19O+": ("[M+H]+", "--adduct")}
    assert man["counts"]["classes"] == {"adduct": 2, "cluster": 0, "covalent": 3, "mixed": 0,
                                        "unknown": 0}
    assert man["counts"]["channel_rows"] == len(pd.read_csv(out / "ms2_channels.csv"))
    assert AN.CHANNELS_NOTE in man["notes"]
    from peaky.reporting import provenance as PV
    assert man["files"]["ms2_classes.csv"] == PV.sha1_file(str(out / "ms2_classes.csv"))
    # again with no adduct: the two tables are not this run's -- removed
    assert cli.main(["ms2", "analyze", "--run-dir", str(run), "--ion", "137.1323=C10H17+",
                     "--no-figures", "--out-dir", str(out)]) == 0
    names = {p.name for p in out.iterdir()}
    assert not {"ms2_channels.csv", "ms2_classes.csv"} & names
    man = json.loads((out / "ms2_analyze_manifest.json").read_text())
    assert man["counts"]["classes"] is None and not man["parameters"]["channels"]["on"]
    assert "adduct" not in man["parameters"]["ions"][0]
    assert "class:" not in capsys.readouterr().out


def test_cli_reagent_profile(tmp_path, no_server, capsys):
    run = _write_run(tmp_path / "run", "ms2_monoterpene_a")
    assert cli.main(["ms2", "analyze", "--run-dir", str(run), "--ion", "137.1323=C10H17+",
                     "--ion", "155.1558=C10H20^N+", "--ion", "155.1428=C10H19O+",
                     "--reagent", "NH4_15N", "--no-figures"]) == 0
    cls = pd.read_csv(run / "ms2_classes.csv", keep_default_na=False, na_values=[""])
    assert set(cls["adduct_source"]) == {"--reagent NH4_15N"}
    assert dict(zip(cls["ion"], cls["adduct"])) == {"C10H17+": "[M+H]+", "C10H20^N+": "[M+^NH4]+",
                                                     "C10H19O+": "[M+H]+"}


@pytest.mark.parametrize("argv, message", [
    (["--adduct", "[M+Q]+"], "no group formula"),
    (["--reagent", "auto"], "--reagent auto reads a peak table"),
    (["--reagent", "nonesuch"], "unknown reagent"),
    (["--reagent", "NO3"], "--reagent NO3 has no form of the named ions' charge"),
    (["--adduct", "[M+NO3]-"], "--adduct \\[M\\+NO3\\]- is an anion form"),
])
def test_cli_refuses_a_bad_adduct(tmp_path, no_server, argv, message):
    run = _write_run(tmp_path / "run", "ms2_monoterpene_a")
    with pytest.raises(SystemExit, match=message):
        cli.main(["ms2", "analyze", "--run-dir", str(run), "--ion", "137.1323=C10H17+",
                  "--no-figures"] + argv)


def test_cli_refuses_an_ions_csv_adduct_the_ion_cannot_be(tmp_path, no_server):
    run = _write_run(tmp_path / "run", "ms2_monoterpene_a")
    ions = tmp_path / "ions.csv"
    ions.write_text("formula,adduct\nC10H17+,[M+NH4]+\n")
    with pytest.raises(SystemExit, match="row 2: C10H17\\+ cannot be \\[M\\+NH4\\]\\+"):
        cli.main(["ms2", "analyze", "--run-dir", str(run), "--ions", str(ions), "--no-figures"])


def test_cli_help_names_the_adduct_inputs():
    p = cli.build_parser()
    sub = next(a for a in p._actions if a.__class__.__name__ == "_SubParsersAction").choices["ms2"]
    an = next(a for a in sub._actions if a.__class__.__name__ == "_SubParsersAction"
              ).choices["analyze"]
    text = an.format_help()
    assert "--adduct LABEL" in text and "--reagent PROFILE" in text and "'adduct'" in text
    a = p.parse_args(["ms2", "analyze", "--run-dir", "D", "--ion", "137.13=C10H17+"])
    assert (a.adduct, a.reagent) == (None, None)


# -------------------------------------- ties, splits and unseen products ----
def test_a_product_that_is_a_monomer_and_a_protonated_half_is_a_split():
    """In 14N data a half that holds the reagent neutral's atoms (an amide, an
    organonitrate) makes a product that reads both as Y . G and as [Y'+H]+ /
    [Y'-H]- (Y' = Y + the reagent neutral): composition cannot tell a
    cluster's monomer . reagent ion from a protonated or deprotonated half.
    It is 'reagent loss + split', both pairs of halves named (the split's
    first), and decides no cluster: a window of [B+H]+ alone reads as one of
    [A+H]+ does, said. The half that holds N with the reagent ion (two N) is
    a monomer alone here, where the shed half holds no N to give up the
    reagent neutral with (M holds one N; two N-holding halves never read as
    a cluster); a 15N label breaks the tie; and a monomer is still
    read before the small losses."""
    nh4 = CH.ion_adduct("C23H31N2O7+", "[M+NH4]+")           # C11H12O5 + C12H15NO2 (an amide)
    assert CH.product_channel(nh4, "C11H13O5+") == (CH.SPLIT, "C11H12O5 + C12H15NO2")
    assert CH.product_channel(nh4, "C12H16NO2+") == (
        CH.SPLIT, "C12H15NO2 + C11H12O5 or C12H12O2 + C11H15NO5")
    assert CH.product_channel(nh4, "C12H19N2O2+") == (CH.MONOMER, "C12H15NO2 + C11H12O5")
    no3 = CH.ion_adduct("C22H31N2O16-", "[M+NO3]-")          # C11H14O6 + C11H17NO7 (a nitrate)
    assert CH.product_channel(no3, "C11H13O6-") == (CH.SPLIT, "C11H14O6 + C11H17NO7")
    assert CH.product_channel(no3, "C11H16NO7-") == (
        CH.SPLIT, "C11H17NO7 + C11H14O6 or C11H16O4 + C11H15NO9")
    assert CH.product_channel(no3, "C11H17N2O10-") == (CH.MONOMER, "C11H17NO7 + C11H14O6")
    n15 = CH.ion_adduct("C23H31N^NO7+", "[M+^NH4]+")
    assert CH.product_channel(n15, "C12H16NO2+") == (CH.SPLIT, "C12H15NO2 + C11H12O5")
    assert CH.product_channel(n15, "C12H16^NO2+") == (CH.MONOMER, "C12H12O2 + C11H15NO5")
    # a urea cluster's Y . UrH+ whose shed half holds urea's atoms is [M+H]+
    # - C5H4O too: read before the small losses, a monomer
    ur = CH.ion_adduct("C18H25N4O7+", "[M+(CH4N2O)H]+")       # C11H12O4 + C6H8N2O2
    assert CH.product_channel(ur, "C12H17N2O5+") == (CH.MONOMER, "C11H12O4 + C6H8N2O2")
    x = "C23H31N2O7+"
    for prod, tie in (("C11H13O5+", False), ("C12H16NO2+", True)):
        steps = [(e, {x: i, prod: 1000.0 - i}) for e, i in zip(NCE, (700.0, 400.0, 200.0, 50.0))]
        _, res = _run(round(_mz(x), 4), steps, (x,), "[M+NH4]+")
        r = _cls(res, x)
        assert (r["class"], r["share_split"], r["share_monomer"], r["monomer_pop"]) == \
            (CH.CLASS_UNKNOWN, 1.0, 0.0, 0.0), prod
        assert r["evidence"].startswith("its products are halves of M split after the reagent's "
                                        "loss"), prod
        assert ("also read as a monomer . reagent ion" in r["note"]) is tie, prod
    # a covalent N-holding dimer that loses NH3 first and cleaves into its
    # protonated halves later sheds no monomer, whichever half holds the N
    for half in ("C11H13O5+", "C12H16NO2+"):
        steps = [(1, {x: 1000.0, "C23H28NO7+": 300.0}), (10, {x: 500.0, "C23H28NO7+": 500.0}),
                 (20, {x: 200.0, "C23H28NO7+": 300.0, half: 500.0})]
        _, res = _run(round(_mz(x), 4), steps, (x,), "[M+NH4]+")
        r = _cls(res, x)
        assert r["class"] == CH.CLASS_ADDUCT and "it sheds no monomer" in r["evidence"], half


def test_an_isotope_label_is_the_reagents_not_ms():
    """Only the reagent carries a 15N label: of the group forms that fit
    alike, the one whose M holds the fewer labelled atoms is read -- a 14N
    organonitrate or nitro compound with the 15N reagent ion, whatever the
    order of the names in a composed profile, with no tie said (an M holding
    ^N needs a covalent 15N product AND the minor 14N reagent ion). So its
    15N reagent ion is its reagent ion and its H^NO3 loss the reagent loss:
    an adduct, as under NO3_15N alone."""
    ions = ["C11H17N^NO10-", "C11H9N^NO5-", "C11H17N^N2O10-"]
    for name in ("NO3+NO3_15N", "NO3_15N+NO3", "NO3_15N"):
        out, notes = CH.assign_adducts(ions, reagent=PR.resolve(name))
        assert (out["C11H17N^NO10-"].label, out["C11H17N^NO10-"].body) == \
            ("[M+^NO3]-", {"C": 11, "H": 17, "N": 1, "O": 7}), name
        assert out["C11H9N^NO5-"].label == "[M+^NO3]-", name
        # two ^N: both readings put ^N in M -- the one with fewer
        assert out["C11H17N^N2O10-"].body.get("^N") == 1, name
        assert not any("equally" in n for n in notes), (name, notes)
    x = "C11H17N^NO10-"
    ads, _ = CH.assign_adducts([x], reagent=PR.resolve("NO3+NO3_15N"))
    assert CH.product_channel(ads[x], "^NO3-")[0] == CH.REAGENT_ION
    assert CH.product_channel(ads[x], "C11H16NO7-")[0] == CH.REAGENT_LOSS
    steps = [(e, {x: i, "^NO3-": 0.25 * (1000.0 - i), "C11H16NO7-": 0.75 * (1000.0 - i)})
             for e, i in zip(NCE, (700.0, 400.0, 200.0, 50.0))]
    res = AN.analyze_run(_synthetic(round(_mz(x), 4), steps), _ions(x), adducts=ads)
    r = _cls(res, x)
    assert (r["class"], r["share_reagent_ion"], r["share_reagent_loss"]) == \
        (CH.CLASS_ADDUCT, pytest.approx(0.25), pytest.approx(0.75))


def _two_urea_adducts(b_int, own):
    """Two urea adducts in one window, a strong (a) and a weak (b): a's
    [M+H]+ C10H7O7+ is a sub-formula of b, split by survivor weights, and b
    reads it as unexplained -- 'another ion's product' for b. `own`: b's own
    products per energy ({formula: [one value per NCE]})."""
    a, b, ra = "C11H11N2O8+", "C12H15N2O7+", "C10H7O7+"
    steps = []
    for j, (e, i) in enumerate(zip(NCE, (700.0, 400.0, 200.0, 50.0))):
        d = {a: i, b: b_int[j], ra: 1000.0 - i}
        d.update({f: v[j] for f, v in own.items() if v[j]})
        steps.append((e, d))
    return a, b, steps


def test_the_class_reads_the_ions_own_products():
    """'Another ion's product' is out of the class's reading altogether --
    out of the class window (FRAG_MIN: what the ION turns into first), of
    UNEXPLAINED_MAX's denominator and of the explained products (and of the
    energy test: `test_the_energy_test_reads_the_ions_own_products`): (1) b's own
    products, 6 counts of its [M+H]+ and 40 of [M+UrH-NH3]+ (part of urea
    kept: unexplained, 96 % of them), are 'unknown', named beside a or not --
    never 'adduct' on the sliver; (2) a sliver of its own reagent loss
    (0.26 of 3-6 counts, a fragment a can make too) beside 79-250 of a's
    shared one opens no window: 'unknown', it barely fragments; (3) no
    product of its own at NCE 1, its reagent loss from NCE 10: the window
    runs on to it, an adduct."""
    rb, ub = "C11H11O6+", "C12H12NO7+"
    a, b, steps = _two_urea_adducts((250.0, 200.0, 120.0, 40.0),
                                    {rb: [6.0] * 4, ub: [40.0] * 4})
    t = _synthetic(round(_mz(a), 4), steps, width=0.4)
    res = AN.analyze_run(t, _ions(a, b), adducts=_adducts((a, b), "[M+(CH4N2O)H]+"))
    r = _cls(res, b)
    assert r["share_other"] > 0.5 and r["share_unexplained"] < CH.UNEXPLAINED_MAX
    assert r["class"] == CH.CLASS_UNKNOWN
    assert r["evidence"].startswith("most of its own products in the window (NCE 1) are "
                                    "unexplained (96 % of them; of all it is given: ")
    assert "another named ion's products are left out" in r["note"]
    t = _synthetic(round(_mz(b), 4), steps, width=0.4)
    alone = _cls(AN.analyze_run(t, _ions(b), adducts=_adducts((b,), "[M+(CH4N2O)H]+")), b)
    assert alone["class"] == CH.CLASS_UNKNOWN
    a, b, steps = _two_urea_adducts((250.0, 240.0, 220.0, 200.0), {rb: [3.0, 4.0, 5.0, 6.0]})
    t = _synthetic(round(_mz(a), 4), steps, width=0.4)
    res = AN.analyze_run(t, _ions(a, b), adducts=_adducts((a, b), "[M+(CH4N2O)H]+"))
    r = _cls(res, b)
    assert r["class"] == CH.CLASS_UNKNOWN and r["evidence"].startswith("barely fragments")
    assert r["frag"] < 0.01 and r["share_other"] > 0.99
    assert _cls(res, a)["class"] == CH.CLASS_ADDUCT
    a, b, steps = _two_urea_adducts((250.0, 230.0, 120.0, 40.0), {rb: [0.0, 60.0, 180.0, 260.0]})
    t = _synthetic(round(_mz(a), 4), steps, width=0.4)
    res = AN.analyze_run(t, _ions(a, b), adducts=_adducts((a, b), "[M+(CH4N2O)H]+"))
    r = _cls(res, b)
    assert (r["class"], r["nce_lo"]) == (CH.CLASS_ADDUCT, 1.0) and r["nce_hi"] >= 10.0
    assert r["share_reagent_loss"] > 0 and r["evidence"].startswith("loses the reagent first")
    # (4) a cluster's population is read on ion + its OWN products: another
    # ion's, however strong, dilute neither it nor the window
    x, mono = "C24H24NO4+", "C12H14NO2+"
    _, res = _run(round(_mz(x), 4), [(1, {x: 200.0, mono: 1500.0}),
                                     (10, {x: 50.0, mono: 1300.0})], (x,), "[M+NH4]+")
    ch = res.channels[res.channels["ion"] == x].copy()
    e50, ad = res.e50[res.e50["ion"] == x], CH.ion_adduct(x, "[M+NH4]+")
    base = CH.classify(ch, e50, ad, ion_mz=_mz(x))
    ch.loc[ch["channel"] == CH.OTHER_ION, "intensity"] = 5000.0
    r = CH.classify(ch, e50, ad, ion_mz=_mz(x))
    assert (r["class"], base["class"]) == (CH.CLASS_CLUSTER, CH.CLASS_CLUSTER)
    assert r["monomer_pop"] == base["monomer_pop"] == pytest.approx(1300 / 1350)
    assert r["frag"] == base["frag"] and r["share_other"] > 0.5


def test_a_never_seen_ion_is_classed_on_products_mostly_its_own_in_the_window():
    """An ion never seen as a precursor is read at its lowest energy (the
    class window); a product mostly its own anywhere else is no evidence of
    what it turns into first. z, named beside x (MS1 shares 0.6 / 0.4), is
    given 0.4 of x's C11H7O2+: with a product only it makes at NCE 10 or
    above -- 2 counts of its reagent loss, or 500 of a skeletal one -- its
    class window holds only that sliver: 'unknown', said. With its own
    product in the window too, it is read on it."""
    x, z = "C11H12NO3+", "C12H16NO2+"
    rl, sm = "C11H9O3+", "C11H7O2+"
    p = round(_mz(x), 4)

    def cls(own=None, amount=0.0, at=()):
        steps = []
        for e, i, va, vb in zip(NCE, (1000.0, 600.0, 300.0, 100.0), (300.0, 600.0, 700.0, 600.0),
                                (60.0, 150.0, 300.0, 400.0)):
            d = {x: i, rl: va, sm: vb}
            if own and e in at:
                d[own] = amount
            steps.append((e, d))
        t = _synthetic(p, steps)
        t.window = _ms1(p, [(_mz(x), 0.6), (_mz(z), 0.4)])
        res = AN.analyze_run(t, _ions(x, z), adducts=_adducts((x, z), "[M+NH4]+"))
        assert _cls(res, x)["class"] == CH.CLASS_ADDUCT
        return _cls(res, z)
    assert "its largest weight 0.4, at most 0.5" in cls()["evidence"]
    for own, amount, at in (("C12H13O2+", 2.0, (10,)), ("C12H14NO+", 500.0, (10, 20, 30)),
                            ("C12H14NO+", 2.0, (1,))):
        r = cls(own, amount, at)
        assert r["class"] == CH.CLASS_UNKNOWN, (own, amount, at)
        assert ("never seen as a precursor in the window, and the products mostly its own hold "
                in r["evidence"]), r["evidence"]
        assert "its class window (NCE 1)" in r["evidence"]
    r = cls("C12H14NO+", 500.0, (1, 10, 20, 30))
    assert r["share_skeletal"] > 0.9 and r["evidence"].startswith(
        "keeps the reagent and fragments its skeleton")


def test_volts_at_one_energy_only_bound_the_e_cm_and_say_so():
    """With fewer than two energies with volts, an energy without has no E_cm
    of its own -- only a bound off the one with (E_cm rises with NCE),
    worded as one: a skeletal ion that never falls to half, volts at NCE 1
    only, holds half its signal to AT LEAST NCE 1's E_cm (never 'the highest
    energy it is seen at''s); a reagent-loss ion crossing at NCE 45, volts
    at NCE 20 only, holds to at least NCE 20's (past COVALENT_ECM: covalent),
    and with volts at NCE 60 only falls to half by NCE 60's at the latest
    and is not known to hold anything (an adduct, the energy test said
    unread)."""
    y, sk = "C11H12^NO3+", "C11H10^NO2+"
    steps = [(e, {y: 1000.0 * f, sk: 1000.0 * (1 - f)})
             for e, f in ((1, 0.85), (20, 0.75), (40, 0.65), (60, 0.6))]
    t, res = _run(round(_mz(y), 4), steps, (y,), "[M+^NH4]+")
    assert _cls(res, y)["class"] == CH.CLASS_COVALENT
    t.parents.loc[t.parents["nce"] > 1, "hcd_v"] = NAN
    r = _cls(AN.analyze_run(t, _ions(y), adducts=_adducts((y,), "[M+^NH4]+")), y)
    assert r["class"] == CH.CLASS_UNKNOWN
    assert (f"it holds half its signal to at least E_cm {_ecm(1, y):.2f} eV (a bound: NCE 1's, "
            "the nearest energy with volts below NCE 60") in r["evidence"]
    assert "volts at one energy only (NCE 1)" in r["note"]
    x = "C8H8NO2+"
    steps = [(e, {x: i, "C8H5O2+": f}) for e, i, f in ((1, 1000.0, 250.0), (20, 800.0, 300.0),
                                                       (40, 600.0, 450.0), (60, 300.0, 700.0))]
    for keep, want in ((20, CH.CLASS_COVALENT), (60, CH.CLASS_ADDUCT)):
        t, _ = _run(round(_mz(x), 4), steps, (x,), "[M+NH4]+")
        t.parents.loc[t.parents["nce"] != keep, "hcd_v"] = NAN
        r = _cls(AN.analyze_run(t, _ions(x), adducts=_adducts((x,), "[M+NH4]+")), x)
        assert r["class"] == want, keep
        assert f"volts at one energy only (NCE {keep})" in r["note"], keep
        if keep == 20:
            assert (f"holds half its signal to at least E_cm {_ecm(20, x):.2f} eV (a bound: "
                    "NCE 20's") in r["evidence"]
        else:
            assert r["e50_ecm"] == pytest.approx(_ecm(60, x))
            assert (f"falls to half by E_cm {_ecm(60, x):.2f} eV (F; a bound: NCE 60's, the "
                    "nearest energy with volts above NCE 45") in r["evidence"]
            assert "the E50's E_cm is a bound: NCE 60's" in r["note"]
            assert "read off the NCE axis" not in r["note"]
    # an ion with no adduct: no class, but its E50's E_cm is the same bound,
    # said as one
    r = _cls(AN.analyze_run(t, _ions(x), adducts={x: None}), x)
    assert (r["class"], r["e50_ecm"]) == (CH.CLASS_UNKNOWN, pytest.approx(_ecm(60, x)))
    assert "the E50's E_cm is a bound: NCE 60's" in r["note"]
    assert "volts at one energy only (NCE 60)" in r["note"]
    assert "energy test" not in r["note"]
    # one that barely fragments, volts at its lowest energy only: the E_cm
    # its products are read up to is unknown for that reason, not 'no volts'
    steps = [(e, {x: 1000.0, "C8H5O2+": f}) for e, f in ((1, 20.0), (20, 25.0), (40, 30.0))]
    t, _ = _run(round(_mz(x), 4), steps, (x,), "[M+NH4]+")
    t.parents.loc[t.parents["nce"] > 1, "hcd_v"] = NAN
    r = _cls(AN.analyze_run(t, _ions(x), adducts=_adducts((x,), "[M+NH4]+")), x)
    assert r["class"] == CH.CLASS_UNKNOWN
    assert "up to E_cm unknown (volts at one energy only)" in r["evidence"]
    assert "no volts" not in r["evidence"]


def test_the_rows_name_each_ions_role_mz_and_largest_split():
    """The urea trimer ion of an analyte . Ur3H+ is its reagent ion (two
    whole ureas); each row's role is its ion's (the target, a co-isolate);
    a class row's ion_mz is the ion's own, not the window's; and a cluster's
    halves are its largest split's, wherever on the ladder it lies."""
    a = CH.ion_adduct("C14H21N6O6+", "[M+(CH4N2O)H]+")          # C11H8O3 . Ur3H+
    assert CH.product_channel(a, "C3H13N6O3+") == (CH.REAGENT_ION, "")
    x, y = "C10H20^N+", "C10H19O+"
    steps = [(e, {x: i, y: j, "C10H17+": c}) for e, i, j, c in
             zip(NCE, (1000.0, 600.0, 300.0, 100.0), (500.0, 450.0, 300.0, 100.0),
                 (300.0, 600.0, 700.0, 600.0))]
    t = _synthetic(155.1559, steps, width=0.4)
    res = AN.analyze_run(t, _ions(x, y), adducts={x: CH.ion_adduct(x, "[M+^NH4]+"),
                                                  y: CH.ion_adduct(y, "[M+H]+")})
    for ion, role in ((x, "target"), (y, "co-isolate")):
        assert set(res.channels.loc[res.channels["ion"] == ion, "role"]) == {role}, ion
        assert _cls(res, ion)["role"] == role
        assert _cls(res, ion)["ion_mz"] == pytest.approx(AN.ion_mz(ion), abs=1e-9)
    d = "C24H24NO4+"
    _, res = _run(round(_mz(d), 4), [(1, {d: 1000.0, "C10H12NO2+": 100.0}),
                                     (10, {d: 300.0, "C12H14NO2+": 1500.0})], (d,), "[M+NH4]+")
    assert _cls(res, d)["halves"] == "C12H10O2 + C12H10O2"


def test_the_quick_references_name_the_binding_class():
    """SKILL.md (the command reference an agent follows) and README.md name
    how `ms2 analyze` reads the binding class and the two tables it writes."""
    root = Path(__file__).resolve().parents[1]
    for name in ("SKILL.md", "README.md"):
        text = (root / name).read_text(encoding="utf-8")
        assert any("ms2 analyze" in line and "--reagent" in line and "--adduct" in line
                   for line in text.splitlines()), name
        for word in ("ms2_channels.csv", "ms2_classes.csv", "§11.13"):
            assert word in text, (name, word)


# ------------------------------------------ 14N clusters and co-isolates ----
def test_two_halves_that_both_hold_the_reagent_neutral_never_read_as_a_cluster():
    """Y . G is monomer shedding alone only where the shed half M - Y cannot
    have given up the reagent neutral (M - Y - R is no plausible half): two
    N in a 14N product are a sure sign only where M holds one N. A 14N
    cluster of two organonitrates with NO3- -- or of two amides with NH4+ --
    sheds products each of which reads as well as the other half's
    deprotonated (protonated) dinitrate (diamide) beside an N-free half:
    'reagent loss + split' whichever half it sheds, class 'unknown'. A 15N
    reagent tells (its ^N is no part of M): a cluster. The same holds for a
    half that holds the reagent's own element: an organobromide half . Br-
    is a monomer (the other half holds no Br), the other half . Br- a tie."""
    no3 = CH.ion_adduct("C22H36N3O18-", "[M+NO3]-")         # C11H17NO8 + C11H19NO7
    assert CH.product_channel(no3, "C11H17N2O11-") == (
        CH.SPLIT, "C11H18N2O11 + C11H18O4 or C11H17NO8 + C11H19NO7")
    assert CH.product_channel(no3, "C11H19N2O10-") == (
        CH.SPLIT, "C11H20N2O10 + C11H16O5 or C11H19NO7 + C11H17NO8")
    nh4 = CH.ion_adduct("C23H34N3O5+", "[M+NH4]+")          # C12H15NO2 + C11H15NO3
    for prod in ("C12H19N2O2+", "C11H19N2O3+"):
        assert CH.product_channel(nh4, prod)[0] == CH.SPLIT, prod
    n15 = CH.ion_adduct("C22H36N2^NO18-", "[M+^NO3]-")
    assert CH.product_channel(n15, "C11H17N^NO11-") == (CH.MONOMER, "C11H17NO8 + C11H19NO7")
    br = CH.ion_adduct("C17H29Br2O5-", "[M+Br]-")            # C11H18O3 + C6H11BrO2
    assert CH.product_channel(br, "C6H11Br2O2-") == (CH.MONOMER, "C6H11BrO2 + C11H18O3")
    assert CH.product_channel(br, "C11H18BrO3-") == (
        CH.SPLIT, "C11H19BrO3 + C6H10O2 or C11H18O3 + C6H11BrO2")
    for x, label, a, b, want in (
            ("C22H36N3O18-", "[M+NO3]-", "C11H17N2O11-", "C11H19N2O10-", CH.CLASS_UNKNOWN),
            ("C22H36N2^NO18-", "[M+^NO3]-", "C11H17N^NO11-", "C11H19N^NO10-",
             CH.CLASS_CLUSTER)):
        steps = [(e, {x: i, a: (1000.0 - i) / 2, b: (1000.0 - i) / 2})
                 for e, i in zip(NCE, (700.0, 400.0, 200.0, 50.0))]
        _, res = _run(round(_mz(x), 4), steps, (x,), label)
        r = _cls(res, x)
        assert r["class"] == want, label
        if want == CH.CLASS_UNKNOWN:
            assert (r["share_split"], r["share_monomer"], r["monomer_pop"]) == (1.0, 0.0, 0.0)
            assert "also read as a monomer . reagent ion" in r["note"]
        else:
            assert r["share_monomer"] == 1.0 and r["monomer_pop"] == pytest.approx(0.95)


def test_a_tie_in_an_adducts_window_is_said_as_one():
    """An adduct whose window holds a product that is a protonated half and
    a monomer . reagent ion alike (the tie) never says it sheds no monomer
    . reagent ion: 'no product reads only as one', the tie's share said as
    both. The class note names the tie generically -- any half that holds
    the reagent neutral's atoms (a 14N amide or organonitrate, an
    organobromide), not a 14N N-holding half alone. Without the tie (the
    N-free half protonated), the plain clause."""
    x = "C23H31N2O7+"                                        # C11H12O5 + C12H15NO2 (an amide)
    p = round(_mz(x), 4)
    for half, tie in (("C12H16NO2+", True), ("C11H13O5+", False)):
        steps = [(e, {x: i, "C23H28NO7+": 0.75 * (1000.0 - i), half: 0.25 * (1000.0 - i)})
                 for e, i in zip(NCE, (700.0, 400.0, 200.0, 50.0))]
        _, res = _run(p, steps, (x,), "[M+NH4]+")
        r = _cls(res, x)
        assert (r["class"], r["monomer_pop"]) == (CH.CLASS_ADDUCT, 0.0), half
        if tie:
            assert "sheds no monomer" not in r["evidence"]
            assert r["evidence"].endswith(
                "; no product reads only as a monomer . reagent ion, but up to 8 % of ion + "
                "products are halves split after the reagent's loss or, as well, monomer . "
                "reagent ions (composition cannot tell)")
            assert ("a half of M holds the reagent neutral's atoms (a 14N amide or "
                    "organonitrate, an organobromide, ...)") in r["note"]
            assert "N-holding" not in r["note"]
        else:
            assert r["evidence"].endswith(
                "; it sheds no monomer . reagent ion, but up to 8 % of ion + products are halves "
                "split after the reagent's loss (a cluster's or a covalent dimer's: composition "
                "cannot tell)")
            assert "also read as a monomer . reagent ion" not in r["note"]
    # a tie above the window only (the window, NCE 1, holds the N-free
    # half's [Y+H]+): the window's splits are said plainly, the tie noted
    steps = [(e, {x: i, "C23H28NO7+": 0.75 * (1000.0 - i), "C11H13O5+": 0.25 * (1000.0 - i),
                  **({"C12H16NO2+": 100.0} if e >= 20 else {})})
             for e, i in zip(NCE, (700.0, 400.0, 200.0, 50.0))]
    _, res = _run(p, steps, (x,), "[M+NH4]+")
    r = _cls(res, x)
    assert (r["class"], r["nce_hi"]) == (CH.CLASS_ADDUCT, 1.0)
    assert "; it sheds no monomer . reagent ion, but up to 8 % of ion + products" in r["evidence"]
    assert "also read as a monomer . reagent ion" in r["note"]


def _shared_fragment(x_steps, shared=True, replicated=False):
    """x = C8H8NO2+ and y = C9H12NO+, both [M+NH4]+, 36 mDa apart: x loses
    NH3 (C8H5O2+); y loses NH3, then CH2O, to C8H7+ -- a sub-formula of x,
    given to x in part by the survivor split, which x cannot explain:
    'another ion's product' for x. `x_steps`: (x, C8H5O2+) per energy (0:
    no peak)."""
    x, y = "C8H8NO2+", "C9H12NO+"
    steps = []
    for e, (i, rl), j, s in zip(NCE, x_steps, (500.0, 400.0, 300.0, 200.0),
                                (500.0, 1000.0, 2000.0, 3000.0)):
        d = {k: v for k, v in {x: i, "C8H5O2+": rl, y: j}.items() if v}
        if shared:
            d["C8H7+"] = s
        steps.append((e, [{k: v * f for k, v in d.items()} for f in (0.9, 1.0, 1.1)]
                      if replicated else d))
    p = round(_mz(x), 4)
    t = (_replicated(p, steps, ions=(x, y)) if replicated else _synthetic(p, steps, width=0.4))
    res = AN.analyze_run(t, _ions(x, y), adducts=_adducts((x, y), "[M+NH4]+"),
                         **({"bootstrap": 200} if replicated else {}))
    return x, res


def test_the_energy_test_reads_the_ions_own_products():
    """'Another ion's product' is out of the energy test too. x holds 94 %
    of its signal to NCE 30 (E_cm 1.57 eV), losing the reagent a little:
    the breakdown's F, which counts y's C8H7+, crosses at NCE 13 (E_cm
    1.04 eV) -- read so, x would be an adduct. On its own products F never
    falls to half: covalent, as without y's product. Where x does fall to
    half on its own products, the class row's E50 is that crossing (NCE
    13.75), its E_cm read off the spectra's (volts linear in NCE), and with
    the per-scan tables no band (the bootstrap's is of the breakdown's F);
    ms2_e50.csv's F is the breakdown's whatever the class reads. And x seen
    only at NCE 10 and 30: the breakdown's F, 0 at NCE 1 where x is given
    only y's product, falls at or below the lowest energy; on its own
    products x is unobserved at NCE 1 and 20, and falls to half across the
    hole (a 'gap' bracketed on the re-read F: by NCE 30, after NCE 10)."""
    held = [(1000.0, 150.0), (980.0, 150.0), (960.0, 150.0), (940.0, 150.0)]
    x, res = _shared_fragment(held)
    r = _cls(res, x)
    f = res.e50[(res.e50["ion"] == x) & (res.e50["curve"] == "F")].iloc[0]
    assert f["flag"] == "interpolated" and f["e50_ecm"] < CH.COVALENT_ECM
    assert r["share_other"] > 0.5
    assert (r["class"], r["e50_flag"], r["e50_curve"]) == (CH.CLASS_COVALENT, "above_range", "")
    assert (f"it holds half its signal to E_cm {_ecm(30, x):.2f} eV, the highest energy it is "
            "seen at") in r["evidence"]
    assert "the energy test reads F on its own products" in r["note"]
    x, res = _shared_fragment(held, shared=False)
    r = _cls(res, x)
    assert r["class"] == CH.CLASS_COVALENT and "energy test" not in r["note"]
    falls = [(1000.0, 100.0), (700.0, 400.0), (300.0, 800.0), (100.0, 900.0)]
    x, res = _shared_fragment(falls)
    r = _cls(res, x)
    f = res.e50[(res.e50["ion"] == x) & (res.e50["curve"] == "F")].iloc[0]
    assert f["e50_nce"] < 13.0
    assert (r["e50_curve"], r["e50_flag"]) == ("F", "interpolated")
    assert r["e50_nce"] == pytest.approx(10 + 10 * (700 / 1100 - 0.5) / (700 / 1100 - 300 / 1100))
    assert r["e50_ecm"] == pytest.approx(_ecm(r["e50_nce"], x))
    assert "read off the NCE axis" not in r["note"]          # its spectra's own E_cm
    x, res = _shared_fragment(falls, replicated=True)
    r = _cls(res, x)
    f = res.e50[(res.e50["ion"] == x) & (res.e50["curve"] == "F")].iloc[0]
    assert np.isfinite(f["e50_ecm_lo"]) and r["e50_curve"] == "F"
    assert math.isnan(r["e50_ecm_lo"]) and math.isnan(r["e50_ecm_hi"])
    assert "no band" in r["note"]
    x, res = _shared_fragment([(0.0, 0.0), (900.0, 150.0), (0.0, 0.0), (100.0, 900.0)])
    r = _cls(res, x)
    f = res.e50[(res.e50["ion"] == x) & (res.e50["curve"] == "F")].iloc[0]
    assert f["flag"] == "below_range"
    assert (r["e50_curve"], r["e50_flag"], r["e50_nce"]) == ("F", "gap", 30.0)
    assert r["e50_ecm"] == pytest.approx(_ecm(30, x))
    assert (f"falls to half by E_cm {_ecm(30, x):.2f} eV (F, across an unobserved energy, "
            f"after E_cm {_ecm(10, x):.2f} eV)") in r["evidence"]


def test_another_ions_product_never_makes_a_never_seen_ions_own():
    """A never-seen ion b whose MS1 share (0.6) outweighs a's is given 0.6 of
    a's [M+H]+ -- a product it cannot explain, another ion's -- and 2 counts
    of its own reagent loss at NCE 1: the weight makes the first no product
    of its own (`OWN_WEIGHT`), so its class window holds 1 % of products
    its own: 'unknown', never 'adduct' on 1.2 counts. With an MS1 share of
    0.4 no product it is given is mostly its own. Given only a's [M+H]+, it
    is still given products (F and S of the breakdown hold data), none its
    own."""
    for sb, rb, said in (
            (0.6, 2.0, "the products mostly its own hold 1 % of what it is given"),
            (0.4, 2.0, "no product it is given is mostly its own (its largest weight 0.4"),
            (0.6, 0.0, "the products mostly its own hold 0 % of what it is given")):
        a, b, steps = _two_urea_adducts((0.0, 0.0, 0.0, 0.0), {"C11H11O6+": [rb, 0.0, 0.0, 0.0]})
        p = round(_mz(a), 4)
        t = _synthetic(p, steps, width=0.4)
        t.window = _ms1(p, [(_mz(a), 1 - sb), (_mz(b), sb)])
        res = AN.analyze_run(t, _ions(a, b), adducts=_adducts((a, b), "[M+(CH4N2O)H]+"))
        assert _chan(res, b, 1, CH.OTHER_ION)["products"] == "C10H7O7+", (sb, rb)
        r = _cls(res, b)
        assert r["class"] == CH.CLASS_UNKNOWN and said in r["evidence"], (sb, rb)


def test_an_equal_split_of_one_candidate_is_no_product_of_its_own():
    """An equal split is 1 / its live candidates -- up to 1, not at most 1/2:
    x is seen only from NCE 10 with no MS1 share, and z, never seen, lies
    beyond the inventoried MS1 span (its share unknown), so z alone is left
    to split C11H7O2+ at NCE 1 equally with ('equal', weight 1;
    `attribute.share_weights` reads candidates at the lowest energy). A
    prior, no observation: z's products mostly its own hold 0 % of its
    class window, no class (counting the equal split would class it on
    x's fragment)."""
    x, z = "C11H12NO3+", "C12H16NO2+"
    rl, sm, zown = "C11H9O3+", "C11H7O2+", "C12H14NO+"
    p = round(_mz(x), 4)
    steps = [(1, {sm: 200.0}),
             (10, {x: 600.0, rl: 600.0, sm: 150.0, zown: 300.0}),
             (20, {x: 300.0, rl: 700.0, sm: 300.0, zown: 300.0}),
             (30, {x: 100.0, rl: 600.0, sm: 400.0, zown: 300.0})]
    t = _synthetic(p, steps)
    t.manifest = {"parameters": {"isolation_width_used": 0.04, "window_pad_da": 0.0}}
    t.window = _ms1(p, [(p - 0.011, 1.0)])
    res = AN.analyze_run(t, _ions(x, z), iso_width=0.4, adducts=_adducts((x, z), "[M+NH4]+"))
    fr = res.fragments.set_index("fragment").loc[sm]
    assert fr["weight_source"] == "equal" and f"{z}=1" in fr["weights"]
    r = _cls(res, z)
    assert r["class"] == CH.CLASS_UNKNOWN
    assert "the products mostly its own hold 0 % of what it is given" in r["evidence"]
