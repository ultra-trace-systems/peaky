"""How a window ion falls apart: each MS2 product's CHANNEL relative to the
ion's adduct, and a binding CLASS per ion (adduct / cluster / covalent).

Pure: no Mascope, no I/O. It reads what `analyze` already decided for one
window -- each fragment's formula as each window ion reads it, its share
weights (`attribute`) and the intensity matrices -- against the ion's ADDUCT:
the ion form it was made by (`[M+NH4]+`, `[M+^NO3]-`, ...), whose group
formula `chemistry.ADDUCT_GROUPS` holds. Ion X = M + group G (an in-source
form is read as its reagent's adduct of M less the in-source loss:
`IN_SOURCE`); the reagent neutral R it leaves as on proton transfer is
X - [M+H]+ = G - H (a cation: NH3 of NH4+, urea of the uronium adduct) or
X - [M-H]- = G + H (an anion: HNO3 of NO3-, HBr of Br-, the radicals HO2 of
superoxide and HCO3 of carbonate).

Channels of one product P of X (`product_channel`, the first that applies):

  * 'reagent ion': P is made of the group's atoms alone (P <= G), or holds
    the group and none of M's carbon (the reagent ion with small inorganic
    pieces), or the group with whole reagent neutrals and none of M's carbon
    besides theirs (the reagent ion's cluster with its own neutral: the
    urea dimer ion of the uronium adduct, HCOO- . HCOOH of formate): the
    charge stayed on the reagent, nothing of M's skeleton.
  * 'reagent loss': P = [M+H]+ / [M-H]-: the reagent neutral left by proton
    transfer; or, for a form whose group is a neutral molecule of its own
    (an integer DBE: O2 of superoxide, Br2, I2), P = M: the group left whole
    by electron transfer (O2- . M -> M- + O2, M's electron affinity the
    larger).
  * 'monomer shedding': P holds the whole group, P = Y + G, and BOTH Y and
    the loss M - Y are plausible closed-shell neutrals -- integer DBE >= 0,
    Senior's second rule, an H (`subformula`'s remainder rule), at least
    `MONOMER_MIN_C` carbons each and, when M holds O, at least one O each:
    a cluster of two neutrals that split, the reagent staying on one half.
    Both halves are named ('kept + shed'). Read before the small losses.
    Where M holds the reagent neutral's atoms (a 14N amide or
    organonitrate half, an organobromide half), Y . G may as well be
    [Y'+H]+ / [Y'-H]-, Y' = Y + the reagent neutral R, of two plausible
    halves: composition cannot tell -- it is read as 'reagent loss +
    split', below, both readings named. Y . G is monomer shedding alone
    only where the shed half M - Y cannot have given up R (M - Y - R is
    no plausible half: it lacks R's atoms, or is too small); two N in a
    14N product (its half's own beside the reagent's) are a sure sign
    only where M holds one N. A 14N cluster whose two halves both hold
    R's atoms (two organonitrates with NO3-, two amides with NH4+) reads
    'reagent loss + split' whichever half it sheds -- class 'unknown'; a
    15N label breaks the tie (the reagent's ^N is no part of M).
  * 'reagent loss + small losses': P = [M+H]+ / [M-H]- - L, L a SMALL
    closed-shell loss -- fewer than `MONOMER_MIN_C` carbons in all, which
    `subformula` splits into molecules (library pieces, at most one unnamed
    remainder): the same, then H2O, CO, ...
  * 'reagent loss + split': P = [Y+H]+ / [Y-H]-, Y and M - Y plausible
    halves as above: the reagent's neutral gone and M split in two -- a
    cluster's monomer shed with the reagent's proton, or a covalent dimer's
    [M+H]+ cleaved after it lost the reagent; composition cannot tell
    (neither the reagent channels nor monomer shedding: the class counts it
    in neither). Both halves named (the one holding the charge first); for
    a product that also reads as a monomer . reagent ion (the tie above),
    'split or monomer': 'Y' + M - Y' or Y + M - Y'.
  * 'skeletal': P holds the whole group, holds atoms M alone cannot supply
    (P is no sub-formula of M), and is none of these: a covalent fragment
    that kept the reagent ([M+NH4-H2O]+, -CO, ...). A product M's own atoms
    could make whole holds a group made of M's elements by count only (O2,
    CHO2 in an oxygenated M; NH4 in an N-compound): no evidence it kept the
    reagent.
  * 'unexplained': none of these (part of the group kept, a radical loss,
    a deep fragment of [M+H]+, a product M's own atoms make);
    'another ion's product' when another named
    window ion that is given a share of the same fragment reads it in a
    channel of its own (`window_channels`): most likely that ion's, given
    to this one by the survivor split (`attribute`).

An ion form that adds no reagent group -- [M+H]+, [M-H]-, a charge transfer
-- has nothing to lose but its skeleton: every product is 'skeletal' (a
dimer ion that sheds a monomer cannot be told from a skeletal split by
composition: an ester's [M+H]+ cleaved at its ester bond reads as two
halves of six carbons or more just as well). An
ion with no adduct given is read as 'intact' / 'product' only, class
'unknown'.

A channel's intensity at one energy is the sum of its products'
intensities, each times the ion's attribution weight (`attribute`: 1 for a
fragment only it can make, its share of a shared one); its share is that
over every product the ion is given there (`frag_weighted` of the
breakdown). The 'intact' row carries the precursor itself, share NaN (its F
is in the breakdown). With the per-scan tables each share is re-read scan by
scan (`share_sd`; none where nothing entering the shares is read).

The CLASS (`classify`) is a HEURISTIC read off the lowest energies -- what
the ion turns into FIRST -- and the energy it falls to half by, on the ion's
OWN products: 'another ion's product' is left out of all of it (the energy
test reads F on the ion's own products, not the breakdown's F, which counts
every product it is given). Why those:
a product goes on to fragment at higher energies, and its channel moves
with it (a cluster's monomer . reagent ion loses the reagent neutral and
reads there as reagent loss + split); and the channels are read off OBSERVED
products only -- an ammonium adduct's NH4+ lies below every scan's first
mass, and is seen only as the precursor's fall on S. The chemistry the
class rests on (generic): reagent adducts lose the reagent neutral (then
H2O, ...) and fall to half by about 1 eV E_cm or below (our own ^NH4+
adducts: 0.38-0.86 eV); loosely bound clusters
of two neutrals with the reagent shed a whole neutral, most of their
population already at the lowest energies, and fall to half at lower
energies still; a covalent dimer of the same formula keeps its neutral
whole and loses the reagent; covalently bound ions -- protonated amides,
the formula twins of ammonium adducts, among them -- hold beyond 1 eV.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from peaky.chem import chemistry as C
from peaky.ms2 import subformula as SF

__all__ = [
    "INTACT", "PRODUCT", "REAGENT_ION", "REAGENT_LOSS", "REAGENT_LOSS_SMALL", "MONOMER", "SPLIT",
    "SKELETAL", "OTHER_ION", "UNEXPLAINED", "PRODUCT_CHANNELS", "REAGENT_CHANNELS",
    "MONOMER_MIN_C", "IN_SOURCE",
    "CHANNEL_COLUMNS", "IonAdduct", "ion_adduct", "fits", "reagent_adducts", "assign_adducts",
    "SOURCE_CSV", "SOURCE_DEFAULT", "SOURCE_REAGENT", "product_channel",
    "fragment_formulas", "window_channels",
    "CLASS_ADDUCT", "CLASS_CLUSTER", "CLASS_COVALENT", "CLASS_MIXED", "CLASS_UNKNOWN", "CLASSES",
    "FRAG_MIN", "DOMINANT", "CLUSTER_POP", "UNEXPLAINED_MAX", "COVALENT_ECM", "SKELETAL_ECM",
    "NO_MONOMER", "FIRST_MASS_MIN", "OWN_WEIGHT", "CLASS_COLUMNS",
    "classify",
]

INTACT = "intact"
#: the one product channel of an ion with no adduct given
PRODUCT = "product"
REAGENT_ION = "reagent ion"
REAGENT_LOSS = "reagent loss"
REAGENT_LOSS_SMALL = "reagent loss + small losses"
MONOMER = "monomer shedding"
SPLIT = "reagent loss + split"
SKELETAL = "skeletal"
OTHER_ION = "another ion's product"
UNEXPLAINED = "unexplained"
#: The product channels of an ion with an adduct, in this order.
PRODUCT_CHANNELS = (REAGENT_ION, REAGENT_LOSS, REAGENT_LOSS_SMALL, MONOMER, SPLIT, SKELETAL,
                    OTHER_ION, UNEXPLAINED)
#: The channels in which the reagent and M part: the reagent-loss group the
#: class reads.
REAGENT_CHANNELS = (REAGENT_ION, REAGENT_LOSS, REAGENT_LOSS_SMALL)

#: Both halves of a monomer split hold at least this many carbons; a 'small'
#: loss off [M+H]+ / [M-H]- holds fewer (more could be a whole neutral). Below
#: it a composition cannot tell a shed neutral from a skeletal loss: a C10
#: monoterpene's [M+H]+ C10H17+ makes C5H7+ (67) with C5H10, two closed-shell
#: C5 halves -- a skeletal fragment. At 6 no ion of fewer than 12 C can read
#: as a cluster.
MONOMER_MIN_C = 6

#: In-source forms: an ion form that is a reagent adduct of M less an
#: in-source loss is read as that reagent's adduct of M - loss, so its
#: reagent leaves as it does off the plain adduct ([M+^NH4-H2O]+, the
#: ledger's dehydrated ammonium adduct, loses ^NH3 as [M+^NH4]+ does).
#: Form -> the loss.
IN_SOURCE: dict[str, str] = {"[M+^NH4-H2O]+": "H2O"}

CHANNEL_COLUMNS: dict[str, str] = {
    "parent_mz": "float64", "spectrum": "str", "nce": "float64", "hcd_v": "float64",
    "e_cm": "float64", "ion": "str", "role": "str", "adduct": "str", "channel": "str",
    "intensity": "float64", "share": "float64", "n_products": "int64", "products": "str",
    "halves": "str",
    # per-scan tables (else 0 / NaN): the spectrum's scans, the share's
    # scan-to-scan sd (ddof 1; NaN where none of the ion's products there is
    # read in a scan)
    "n_scans": "int64", "share_sd": "float64",
}


# ---------------------------------------------------------------------------
# the ion's adduct
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class IonAdduct:
    """One window ion read as an ion form: `body` = M (the ion's counts minus
    the group; M less the in-source loss for an `IN_SOURCE` form, whose
    `group` is then its reagent's), `reagent_neutral` what the ion leaves as
    on proton transfer, X - [M+H]+ / X - [M-H]- ('' when the form adds no
    reagent group; a radical for superoxide or carbonate: HO2, CHO3),
    `reagent_ion` the group as an ion ('' when it is no ion of its own),
    `has_group` whether the form adds a reagent group at all ([M+H]+ and
    [M-H]- do not), `source` where the label came from (the ions CSV,
    --adduct, --reagent)."""
    ion: str
    label: str
    source: str = ""
    charge: int = 1
    group: dict = field(default_factory=dict)
    body: dict = field(default_factory=dict)
    reagent_neutral: str = ""
    reagent_ion: str = ""
    has_group: bool = False


def _sub(a: Mapping[str, int], b: Mapping[str, int]) -> dict[str, int]:
    """a - b, element-wise (any sign), zeros dropped."""
    out = dict(a)
    for e, n in b.items():
        out[e] = out.get(e, 0) - n
    return {e: n for e, n in out.items() if n}


def _le(a: Mapping[str, int], b: Mapping[str, int]) -> bool:
    """a <= b element-wise (a is a sub-formula of b)."""
    return all(n <= b.get(e, 0) for e, n in a.items() if n > 0)


def _formula(counts: Mapping[str, int]) -> str:
    return C.format_formula({e: n for e, n in counts.items() if n})


def _closed_loss(counts: Mapping[str, int]) -> bool:
    """A non-empty, all-positive closed-shell loss `subformula` can split into
    molecules (library pieces plus at most one unnamed remainder, up to
    `K_MAX_LIMIT` pieces)."""
    if not counts or any(n < 0 for n in counts.values()):
        return False
    return SF.decompose_loss(dict(counts), k_max=SF.K_MAX_LIMIT,
                             allow_radical=False) is not None


def _small_loss(counts: Mapping[str, int]) -> bool:
    """A closed-shell loss (`_closed_loss`) of fewer than `MONOMER_MIN_C`
    carbons in all: one that cannot be a whole neutral of a cluster."""
    return counts.get("C", 0) < MONOMER_MIN_C and _closed_loss(counts)


def _neutral_ok(counts: Mapping[str, int]) -> bool:
    """A plausible closed-shell neutral of its own: integer DBE >= 0, Senior's
    second rule, an H or halogen (`subformula`'s rule for an unnamed loss
    piece)."""
    if not counts or any(n < 0 for n in counts.values()):
        return False
    v = SF._vec(counts)
    return bool(SF._remainder_ok(v[None, :], np.array([False]))[0])


def _closed_shell(counts: Mapping[str, int]) -> bool:
    """An integer DBE >= 0: an even-electron neutral (no radical)."""
    if not counts or any(n < 0 for n in counts.values()):
        return False
    t2 = 2 + int(SF._vec(counts) @ SF._VM2)
    return t2 >= 0 and t2 % 2 == 0


def _read_group(label: str) -> dict[str, int]:
    """The group a form is read by: its `ADDUCT_GROUPS` group, plus the
    in-source loss of an `IN_SOURCE` form (its reagent's group)."""
    group = C.adduct_group(label)
    loss = IN_SOURCE.get(label)
    return _sub(group, {e: -n for e, n in C.parse_formula(loss).items()}) if loss else group


def _adds_group(label: str) -> bool:
    """Does the form add a reagent group -- anything but H to M?"""
    return bool({e for e, n in _read_group(label).items() if n > 0} - {"H"})


def ion_adduct(ion: str, label: str, source: str = "") -> IonAdduct:
    """`ion` (an ion formula) read as ion form `label`. Raises ValueError when
    the label has no group formula (`chemistry.adduct_group`), its charge is
    not the ion's, or the ion cannot hold the group (M would have a negative
    count, or nothing)."""
    counts, charge = SF.parse_ion(ion)
    norm = SF.format_ion(counts, charge)
    group = _read_group(label)
    if C.adduct_charge(label) != charge:
        raise ValueError(f"{norm} is {'a cation' if charge > 0 else 'an anion'}; {label} is "
                         f"{'an anion' if charge > 0 else 'a cation'} form")
    body = _sub(counts, group)
    neg = {e: n for e, n in body.items() if n < 0}
    if neg or not body:
        raise ValueError(f"{norm} cannot be {label}: it does not hold the group "
                         f"{_formula({e: n for e, n in group.items() if n > 0}) or '(none)'}"
                         + (f" ({', '.join(f'{e} {n:+d}' for e, n in neg.items())} in M)"
                            if neg else ""))
    has_group = bool({e for e, n in group.items() if n > 0} - {"H"})
    neutral = ""
    if has_group:
        # X - [M+H]+ (a cation) / X - [M-H]- (an anion): closed-shell or not
        # (HO2 of superoxide, CHO3 of carbonate are radicals)
        r = _sub(group, {"H": charge})
        if r and all(n > 0 for n in r.values()):
            neutral = _formula(r)
    rion = ""
    if has_group and all(n > 0 for n in group.values()):
        rion = SF.format_ion(group, charge)
    return IonAdduct(ion=norm, label=label, source=source, charge=charge, group=group, body=body,
                     reagent_neutral=neutral, reagent_ion=rion, has_group=has_group)


def fits(ion: str, label: str) -> bool:
    """Can `ion` be read as ion form `label` (`ion_adduct` does not raise)?"""
    try:
        ion_adduct(ion, label)
    except ValueError:
        return False
    return True


#: Where an ion's adduct came from (`IonAdduct.source`).
SOURCE_CSV = "ions csv"
SOURCE_DEFAULT = "--adduct"
SOURCE_REAGENT = "--reagent"


def reagent_adducts(profile) -> list[str]:
    """A reagent profile's ion forms in the order `assign_adducts` tries
    them: its diagnostic adduct (`detect_adduct`, the reagent's own channel;
    each component's, in name order, for a '+'-composed profile, which has
    none of its own) first, then its analyte channels in their order -- and
    of these every form that adds a reagent group before the ones that add
    none ([M+H]+, [M-H]-, a charge transfer: they fit nearly any ion of their
    charge). So a uronium profile, whose first channel is [M+H]+, still
    reads an ion holding urea as the urea adduct, and NO3+Br reads a bromide
    adduct as [M+Br]-, not [M-H]-."""
    first = [getattr(profile, "detect_adduct", None) or ""]
    name = str(getattr(profile, "name", "") or "")
    if not first[0] and "+" in name:
        from peaky.chem import profiles as PR
        first = [getattr(PR.PROFILES.get(part.strip()), "detect_adduct", None) or ""
                 for part in name.split("+")]
    out = first + list(getattr(profile, "adducts", ()) or ())
    labels = [a for a in dict.fromkeys(out) if a and a in C.ADDUCT_GROUPS]
    return sorted(labels, key=lambda a: not _adds_group(a))


def _reagent_element(a: IonAdduct) -> bool:
    """Does M hold an element of the group other than C, H, N, O (the
    reagent's own: I of iodide, Br of bromide)? An ion that holds two
    iodines read as [M+I]- puts one in M, as [M-H+I2]- none."""
    own = {e for e, n in a.group.items() if n > 0} - {"C", "H", "N", "^N", "O"}
    return any(a.body.get(e, 0) > 0 for e in own)


def _labelled(a: IonAdduct) -> int:
    """How many isotope-labelled atoms (^N) M holds. Only a labelled reagent
    carries the label: an M that holds one is a covalent product of the
    labelled reagent clustered with the minor unlabelled reagent ion, where
    the other reading is an unlabelled analyte with the main reagent ion (a
    14N organonitrate or nitro compound . ^NO3-) -- so of two readings that
    fit alike the one with the fewer is read, whatever the order of the
    names in a composed profile (NO3+NO3_15N)."""
    return sum(n for e, n in a.body.items() if e.startswith("^") and n > 0)


def assign_adducts(ions: Sequence[str], per_ion: Mapping[str, str] | None = None,
                   default: str | None = None, reagent=None
                   ) -> tuple[dict[str, IonAdduct | None], list[str]]:
    """Each named ion's adduct, -> ({ion: IonAdduct or None}, notes):

      * its own label (`per_ion`: the --ions CSV's 'adduct' column) -- one
        the ion cannot be is an error (a typo, like a wrong m/z);
      * else `default` (--adduct LABEL) when the ion can be it (else said,
        with `ion_adduct`'s reason);
      * else the reagent profile's (`reagent`, --reagent): of the forms of
        `reagent_adducts` the ion can be, a form that adds a reagent group
        first, then one whose M is closed-shell (an integer DBE: [M+H]+ of
        C10H16, not [M]+. of a radical C10H17), then one whose M holds none of
        the reagent's own element (`_reagent_element`), then one whose M
        holds the fewer labelled atoms (`_labelled`: the ^N is the
        reagent's), then the profile's order. An ion read as a form that
        adds no group while the reagent
        has forms that do is said ('by elimination': an adduct of another
        reagent ion -- NH4+ in urea data -- reads covalent so), as is one
        two group forms fit equally well;
      * else none: the ion is still analysed, its products read as
        'product' only and its class 'unknown' -- said.

    A label must be one of `chemistry.ADDUCT_GROUPS` (raises otherwise), and
    a `default` label or a `reagent` with no form of the named ions' charge
    is refused (ValueError): it can fit none of them."""
    per_ion = dict(per_ion or {})
    if default:
        C.adduct_group(default)
    for lab in per_ion.values():
        if lab:
            C.adduct_group(lab)
    signs = {SF.parse_ion(f)[1] for f in ions}
    sign = next(iter(signs)) if len(signs) == 1 else 0
    kind = {1: "cations", -1: "anions"}.get(sign, "")
    if default and sign and C.adduct_charge(default) != sign:
        raise ValueError(f"--adduct {default} is "
                         f"{'a cation' if C.adduct_charge(default) > 0 else 'an anion'} form; "
                         f"the named ions are {kind}")
    tries = reagent_adducts(reagent) if reagent is not None else []
    rname = getattr(reagent, "name", "") if reagent is not None else ""
    if sign:
        own = [a for a in tries if C.adduct_charge(a) == sign]
        if tries and not own:
            raise ValueError(f"--reagent {rname} has no form of the named ions' charge (they are "
                             f"{kind}; its forms: {', '.join(tries)})")
        tries = own
    group_forms = any(_adds_group(a) for a in tries)
    out: dict[str, IonAdduct | None] = {}
    none, missed, elim, ties = [], [], {}, []
    for f in ions:
        lab = per_ion.get(f, "")
        if lab:
            out[f] = ion_adduct(f, lab, SOURCE_CSV)
            continue
        out[f] = None
        if default:
            try:
                out[f] = ion_adduct(f, default, SOURCE_DEFAULT)
                continue
            except ValueError as exc:
                missed.append(str(exc))
        cands = []
        for i, lab in enumerate(tries):
            try:
                a = ion_adduct(f, lab, f"{SOURCE_REAGENT} {rname}".strip())
            except ValueError:
                continue
            cands.append(((not a.has_group, not _closed_shell(a.body), _reagent_element(a),
                           _labelled(a), i), a))
        if cands:
            cands.sort(key=lambda t: t[0])
            key, a = cands[0]
            out[f] = a
            if group_forms and not a.has_group:
                elim.setdefault(a.label, []).append(f)
            same = [b.label for k, b in cands[1:] if b.has_group and k[:4] == key[:4]]
            if a.has_group and same:
                ties.append(f"{f} ({a.label}; also {', '.join(same)})")
        if out[f] is None:
            none.append(f)
    notes = []
    if missed:
        notes.append(f"--adduct {default} does not fit: " + "; ".join(missed)
                     + (f" -- the reagent {rname}'s forms were tried instead" if tries else ""))
    for lab, fs in elim.items():
        notes.append(f"{lab} by elimination for {', '.join(fs)}: none of the reagent {rname}'s "
                     "forms that add a group fits, and the form adds none (class 'covalent'); "
                     "an adduct of another reagent ion (NH4+ in urea data, 14N in 15N data) "
                     "reads so -- give its adduct in the ions CSV")
    if ties:
        notes.append(f"of the reagent {rname}'s forms, several fit equally well: "
                     + "; ".join(ties) + " -- read as the first; give the adduct in the ions CSV "
                     "if it is another")
    if none:
        notes.append(f"no adduct for {', '.join(none)}: analysed, its products read as 'product' "
                     "only and its class 'unknown'")
    return out, notes


def _halves_ok(y: Mapping[str, int], d: Mapping[str, int], body: Mapping[str, int]) -> bool:
    """Both halves of a split M = Y + D plausible closed-shell neutrals of at
    least `MONOMER_MIN_C` carbons, each with an O when M holds O."""
    for h in (y, d):
        if not _neutral_ok(h) or h.get("C", 0) < MONOMER_MIN_C:
            return False
        if body.get("O", 0) > 0 and h.get("O", 0) < 1:
            return False
    return True


def _neutral_group(adduct: IonAdduct) -> bool:
    """Is the form's group a neutral molecule of its own -- every count
    positive, an integer DBE (O2 of superoxide, Br2, I2): one that can leave
    whole by electron transfer, M keeping the charge?"""
    g = adduct.group
    return bool(g) and all(n > 0 for n in g.values()) and _closed_shell(g)


def _reagent_cluster(y: Mapping[str, int], adduct: IonAdduct) -> bool:
    """y (a product less the group) made of whole reagent neutrals and small
    inorganic pieces -- none of M's carbon besides the reagent's own: the
    reagent ion clustered with its own neutral (Ur2H+ = UrH+ . urea of the
    uronium adduct; HCOO- . HCOOH of formate, CH3COO- . CH3COOH of
    acetate). Read only for a reagent neutral that holds carbon (a
    carbon-free one never adds any: the plain rule reads it)."""
    r = _sub(adduct.group, {"H": adduct.charge})
    if not r or any(n <= 0 for n in r.values()) or r.get("C", 0) <= 0:
        return False
    rest = dict(y)
    while _le(r, rest):
        rest = _sub(rest, r)
    return rest.get("C", 0) <= 0 and all(n >= 0 for n in rest.values())


def product_channel(adduct: IonAdduct | None, product: str) -> tuple[str, str]:
    """(channel, halves) of one product formula of the ion `adduct` reads
    (module docstring; 'another ion's product' is `window_channels`' call).
    `halves` names the split of a monomer shedding or a 'reagent loss +
    split' as 'kept + shed' (the half that keeps the charge first) -- for a
    split that also reads as a monomer . reagent ion, both: 'Y' + D' or Y +
    D' --; '' otherwise. No adduct: 'product'."""
    if adduct is None or not adduct.label:
        return PRODUCT, ""
    if not adduct.has_group:
        return SKELETAL, ""
    p, _ = SF.parse_ion(product)
    body, charge = adduct.body, adduct.charge
    gpos = {e: n for e, n in adduct.group.items() if n > 0}
    if _le(p, gpos):
        return REAGENT_ION, ""
    holds = _le(gpos, p)
    y = _sub(p, adduct.group) if holds else {}
    if holds and body.get("C", 0) > 0 and y.get("C", 0) <= 0:
        # the group with H / O / N but none of M's carbon: the reagent ion
        # clustered with small inorganic pieces, nothing of M's skeleton
        # (and the C2F3 = H2^NO4 alias of CF3COO- in 15N data; HO2- of
        # superoxide, HCO3- of carbonate)
        return REAGENT_ION, ""
    mh = _sub(body, {"H": -charge})                                 # [M+H]+ / [M-H]-
    if p == mh:
        return REAGENT_LOSS, ""
    if p == body and _neutral_group(adduct):
        # M itself: the group left whole by electron transfer (O2- . M ->
        # M- + O2), M intact -- never a skeletal product, though M holds
        # the group's atoms
        return REAGENT_LOSS, ""
    if holds and body.get("C", 0) > 0 and _reagent_cluster(y, adduct):
        # the reagent ion with whole reagent neutrals: their carbon is the
        # reagent's, not M's (an analyte clustered with the urea dimer ion
        # can only be read as the urea adduct of M + urea)
        return REAGENT_ION, ""
    mono = ""
    if holds and y and all(n > 0 for n in y.values()):
        d = _sub(body, y)
        if _halves_ok(y, d, body):
            mono = f"{_formula(y)} + {_formula(d)}"
    if _le(p, mh):
        if not mono and _small_loss(_sub(mh, p)):
            return REAGENT_LOSS_SMALL, ""
        yh = _sub(p, {"H": charge})                                 # Y of [Y+H]+ / [Y-H]-
        if yh and all(n > 0 for n in yh.values()):
            d = _sub(body, yh)
            if all(n > 0 for n in d.values()) and _halves_ok(yh, d, body):
                # Y . G is also [Y'+H]+ / [Y'-H]- (Y' = Y + the reagent
                # neutral) where M - Y could have given the neutral up (a
                # 14N amide or organonitrate half, an organobromide half):
                # composition cannot tell -- read as the split, which
                # decides no cluster, both readings named
                split = f"{_formula(yh)} + {_formula(d)}"
                return SPLIT, f"{split} or {mono}" if mono else split
    if mono:
        return MONOMER, mono
    if holds and not _le(p, body):
        # it holds atoms M alone cannot supply: it kept the reagent. A
        # product M's own atoms make whole holds a group of M's elements
        # (O2, CHO2 in an oxygenated M) by count only: no evidence
        return SKELETAL, ""
    return UNEXPLAINED, ""


# ---------------------------------------------------------------------------
# one window
# ---------------------------------------------------------------------------
def _pairs(text: str) -> list[tuple[str, str]]:
    return [(a, b) for a, _, b in (item.partition(": ") for item in str(text).split("; ")
                                   if item)]


def fragment_formulas(roles: pd.DataFrame, specs: Sequence[str],
                      compositions: Mapping[str, Sequence[str]]) -> dict[tuple[str, str], str]:
    """(fragment key, window ion) -> the formula that ion reads the fragment
    as: at the lowest energy at which it reads it as one of the formulas the
    fragment can be where it sits (`attribute.fragment_compositions`), else
    at the lowest energy it reads it at all -- the rule
    `ms2_fragments.csv`'s losses follow."""
    fr = roles[roles["role"] == "fragment"]
    pos = {s: i for i, s in enumerate(specs)}
    fr = fr.assign(_o=fr["spectrum"].map(pos).fillna(len(pos))).sort_values("_o", kind="stable")
    inside: dict[tuple[str, str], str] = {}
    first: dict[tuple[str, str], str] = {}
    for key, text in zip(fr["key"], fr["formulas"]):
        ok = set(compositions.get(key, ()))
        for ion, f in _pairs(text):
            first.setdefault((key, ion), f)
            if f in ok:
                inside.setdefault((key, ion), f)
    return {**first, **inside}


def window_channels(parent_mz: float, specs: Sequence[str], meta: pd.DataFrame,
                    order: Sequence[str], ions_mz: Mapping[str, float],
                    target: str, ion_int: pd.DataFrame, frag_int: pd.DataFrame,
                    weights: Mapping[str, Mapping[str, float]],
                    formulas: Mapping[tuple[str, str], str],
                    adducts: Mapping[str, IonAdduct | None], *, gas_mass: float,
                    scans: Mapping[str, tuple] | None = None) -> pd.DataFrame:
    """`ms2_channels.csv`'s rows of one window: per (spectrum, window ion) an
    'intact' row (the precursor, share NaN) and one row per product channel
    (`PRODUCT_CHANNELS`, or 'product' alone for an ion with no adduct), its
    weighted intensity and its share of the ion's attributed products there;
    `products` the fragments in it (largest first), `halves` each monomer
    split's (or 'reagent loss + split''s) two neutrals in the same order. A
    fragment an ion reads as 'unexplained' while another window ion given a
    share of it reads it in a channel of its own is 'another ion's product'
    for the first. `scans` (spectrum -> (fragment matrix, scans[, read]):
    `replicates.window_scans`' fragments x scans, rows in `frag_int`'s column
    order, the spectrum's per-scan rows and, per fragment, whether a scan
    reads it -- False for a peak held at its averaged height, `analyze`):
    each share's scan-to-scan sd, where two scans or more are read and at
    least one of the ion's products there is read in them (a held peak
    enters every scan at its averaged height, as in F's spread)."""
    from peaky.ms2 import breakdown as BD
    keys = list(frag_int.columns)
    chan: dict[tuple[str, str], tuple[str, str]] = {}
    for f in order:
        for k in keys:
            if float((weights.get(k) or {}).get(f, 0.0)) > 0:
                form = formulas.get((k, f), "")
                chan[(k, f)] = (product_channel(adducts.get(f), form) if form
                                else (UNEXPLAINED, ""))
    # a fragment one ion cannot explain while another ion given a share of
    # it reads it in a channel of its own: most likely that ion's
    read = dict(chan)
    for (k, f), (c, _) in read.items():
        if c == UNEXPLAINED and any(read[(k, g)][0] not in (UNEXPLAINED, PRODUCT)
                                    for g in order if g != f and (k, g) in read):
            chan[(k, f)] = (OTHER_ION, "")
    rows = []
    for spec in specs:
        nce, volts = meta.loc[spec, "nce"], meta.loc[spec, "hcd_v"]
        sc = (scans or {}).get(spec)
        n_sc = int(sc[1]) if sc is not None else 0
        for i, f in enumerate(order):
            ad = adducts.get(f)
            label = ad.label if ad is not None else ""
            names = PRODUCT_CHANNELS if label else (PRODUCT,)
            ecm = (BD.e_cm(float(volts), ions_mz[f], gas_mass)
                   if math.isfinite(float(volts)) else math.nan)
            base = {"parent_mz": float(parent_mz), "spectrum": spec, "nce": nce,
                    "hcd_v": volts, "e_cm": ecm, "ion": f,
                    "role": "target" if f == target else "co-isolate", "adduct": label}
            per: dict[str, list] = {c: [] for c in names}
            wvec: dict[str, np.ndarray] = {c: np.zeros(len(keys)) for c in names}
            for j, k in enumerate(keys):
                if (k, f) not in chan:
                    continue
                c, half = chan[(k, f)]
                w = float(weights[k][f])
                wvec[c][j] = w
                v = w * float(frag_int.loc[spec, k])
                if v > 0:
                    per[c].append((v, k, half))
            tot = sum(v for c in names for v, _, _ in per[c])
            rows.append({**base, "channel": INTACT, "intensity": float(ion_int.loc[spec, f]),
                         "share": math.nan, "n_products": 0, "products": "", "halves": "",
                         "n_scans": n_sc, "share_sd": math.nan})
            # per-scan shares: each scan's channel over its products -- none
            # where no product of the ion there is read in a scan (every one
            # held at its averaged height: a spread nobody measured)
            sd = {}
            if sc is not None and n_sc >= 2 and len(keys):
                fm = np.asarray(sc[0], dtype=float)
                allw = sum(wvec.values())
                ok = (np.asarray(sc[2], dtype=bool) if len(sc) > 2
                      else np.ones(len(keys), dtype=bool))
                there = (allw > 0) & (frag_int.loc[spec].to_numpy(dtype=float) > 0)
                if (there & ok).any():
                    totj = allw @ fm
                    for c in names:
                        with np.errstate(invalid="ignore", divide="ignore"):
                            sh = np.where(totj > 0, (wvec[c] @ fm) / totj, np.nan)
                        sh = sh[np.isfinite(sh)]
                        sd[c] = float(np.std(sh, ddof=1)) if len(sh) >= 2 else math.nan
            for c in names:
                items = sorted(per[c], key=lambda t: (-t[0], t[1]))
                val = sum(v for v, _, _ in items)
                rows.append({**base, "channel": c, "intensity": val,
                             "share": val / tot if tot > 0 else math.nan,
                             "n_products": len(items),
                             "products": "; ".join(k for _, k, _ in items),
                             "halves": "; ".join(h for _, _, h in items if h),
                             "n_scans": n_sc, "share_sd": sd.get(c, math.nan)})
    return _frame(rows, CHANNEL_COLUMNS)


# ---------------------------------------------------------------------------
# a class per ion
# ---------------------------------------------------------------------------
CLASS_ADDUCT = "adduct"
CLASS_CLUSTER = "cluster"
CLASS_COVALENT = "covalent"
CLASS_MIXED = "mixed"
CLASS_UNKNOWN = "unknown"
CLASSES = (CLASS_ADDUCT, CLASS_CLUSTER, CLASS_COVALENT, CLASS_MIXED, CLASS_UNKNOWN)

#: The class window: the energies from the lowest up to the first at which the
#: ion's OWN products (all but 'another ion's product') hold at least this
#: share of ion + own products (1 - F, less another ion's, >= FRAG_MIN) --
#: what the ion turns into FIRST, before its first products go on to
#: fragment; another ion's products open no window -- and never past the last
#: energy its precursor is seen at
#: (a product that shows only after the precursor is gone shows what became
#: of the ion's products, not of the ion). On our own files every ^NH4 adduct
#: is past it at its lowest energy (the 2026-09-11 limonene adduct C10H20^N+
#: 0.29 at NCE 1; the 2026-09-10 file's 0.20 or more at NCE 1). A protonated
#: amide read as an ammonium adduct (its formula twin) stays below it to the
#: highest energies (a few % of products) and is read by the energy it
#: survives to.
FRAG_MIN = 0.1
#: A channel group dominates with at least this share of the explained
#: products in the window (all but 'unexplained' and 'another ion's
#: product'): a majority. On our own files the closest call is the
#: 2026-09-10 file's C12H22^NO2+, whose lowest energy holds 58 % reagent loss
#: (-^NH3, then -H2O) against 42 % of its in-source [M+^NH4-H2O]+ and
#: [M+^NH4-2H2O]+ (skeletal).
DOMINANT = 0.5
#: A cluster: monomer shedding dominates the window's explained products AND
#: they hold at least this share of ion + own products (another ion's left
#: out) at some energy the precursor is seen at (the most they hold: read at
#: the window's last energy alone, a ladder that starts lower would cap it
#: at that energy's 1 - F) -- the population falls apart into its two
#: neutrals before anything else happens to it. A loosely bound cluster of
#: two neutrals with the reagent
#: has more than half its population shed as monomer . reagent ions at the
#: lowest energies; where a covalent dimer of the same formula shares the
#: isolated population, a third or less, the rest holding to energies where
#: it loses the reagent ('mixed'). 0.4 lies between.
CLUSTER_POP = 0.4
#: Most of the ion's own products in the window unexplained (part of the
#: reagent kept, a radical loss, an unnamed co-isolate's products): no class.
#: A product another named ion explains ('another ion's product') is no part
#: of it, nor of its denominator -- so a class never rests on a sliver beside
#: another ion's products: the explained products it reads are at least half
#: the ion's own, which hold FRAG_MIN of ion + own products at the window's
#: last energy.
UNEXPLAINED_MAX = 0.5
#: An ion still holding half its signal above this centre-of-mass energy
#: (eV; N2 by default) is no adduct, whatever its products: reagent adducts
#: fall to half by about 1 eV, covalent ions may hold beyond it. Own data:
#: every ^NH4 adduct falls to half by E_cm 0.38-0.86 eV (the 2026-09-11
#: limonene adduct 0.855, the 2026-09-10 file's 0.38-0.80; its C9H14^N+ is
#: read at one energy, below half there: by 0.69). The [M+H]+ ions of the
#: same files whose F crosses between two energies do so at 0.80-1.66
#: (C10H19O+ 0.80 -- 0.799 and 0.800 in its two windows: a protonated
#: alcohol loses water easily --, C10H17+ 1.073, C4H10NS2+ 1.126,
#: C11H15O2+ 1.183, C11H13O+ 1.661); of the others C8H9O2+ and C7H6NS+ never
#: fall to half on F, C8H10NO+ crosses across an unobserved energy, and
#: C12H21O2+ is below half at its lowest energy (0.51 eV) only because a
#: 50-fold stronger co-isolated adduct's fragments are shared with it. So the
#: test reads one way only: holding past it rules an adduct out; breaking
#: below it makes none. 'Holding' is read off the curves (F, and S beyond its
#: reference energy): the crossing; the highest energy a curve that never
#: falls to half is seen at; the one energy of a one-energy F that is at
#: least half; the last energy a curve is seen at or above half before it
#: falls across an unobserved one -- never the mere absence of a crossing
#: (`_holds`).
COVALENT_ECM = 1.1
#: Skeletal products the window's majority, but the ion falls to half below
#: this E_cm (eV): no covalent call -- 'unknown'. A covalent skeleton breaks
#: no more easily than a protonated alcohol loses water: the lowest crossing
#: on F of any [M+H]+ ion of our own files is C10H19O+'s, 0.80 eV (0.799 in
#: the 155.1559 window, 0.800 in the 155.1429 one; C12H21O2+'s lower one is
#: the attribution artefact `COVALENT_ECM` names), and 0.75 lies below it.
#: An ion that falls apart as easily as an adduct while its products keep
#: the reagent is an adduct whose products are another ion's (an unnamed
#: co-isolate's monomer . reagent ions, background ions) as likely as not.
#: Read one way, as `COVALENT_ECM` is the other: 'covalent' only where the
#: ion is KNOWN to hold half its signal to this energy (`_holds`); an E50
#: below it, or one not known to reach it -- bounded from above only (a
#: crossing at or below the lowest energy, a one-energy F below half), across
#: an unobserved energy, no volts, or a lower bound below it (volts at one
#: energy only) -- is no covalent call.
SKELETAL_ECM = 0.75
#: An adduct of an M that could split into two monomers (>= 2 x
#: `MONOMER_MIN_C` carbons) whose monomer . reagent ions hold at most this
#: share of ion + own products (at every energy it is seen at) sheds no
#: monomer: M is no cluster of two neutrals (a covalent dimer keeps its
#: neutral whole and loses the reagent). A product that is as well a
#: protonated / deprotonated half (the tie of a half that holds the reagent
#: neutral's atoms: 'reagent loss + split') is no monomer . reagent ion
#: here. Read on ion + own products (another ion's left out), as
#: `CLUSTER_POP`: a share
#: of the products alone rises with every other product gone (a dimer that
#: barely fragments can shed a few % of its population as monomers and still
#: have them make a large share of its few products).
NO_MONOMER = 0.05
#: The lowest first mass an MS2 scan of ours is set to (m/z): every reagent
#: ion below it is unobservable whatever the scan (NH4+, m/z 18). Our own
#: three files record a first mass of 40.00 on every per-scan row they have
#: (their MS1 scans), and the lowest MS2 centroid of each lies within 0.01 of
#: it. Where the per-scan tables record the scans' first mass
#: (`ms2_scans.csv` `low_mass`), that is read instead.
FIRST_MASS_MIN = 40.0
#: An ion never seen as a precursor in the window is classed only where a
#: product is mostly its own (weight above this, and read in a channel of
#: its own: 'another ion's product' is no product of its own, whatever its
#: weight), and where such products hold more than this share of what it is
#: given in its class window (its lowest energy): its shares are otherwise
#: slivers of another ion's fragments read through its formula, and a
#: product of its own elsewhere on the ladder tells nothing of what it turns
#: into first. An equal split (`attribute.share_weights` 'equal': nothing is
#: known of its share) is a prior, no observation, and never counts -- it is
#: 1 / the fragment's live candidates, at most 1/2 (a split needs two
#: candidates the sample holds: absence is read at any energy).
OWN_WEIGHT = 0.5

CLASS_COLUMNS: dict[str, str] = {
    "parent_mz": "float64", "ion": "str", "role": "str", "ion_mz": "float64", "adduct": "str",
    "adduct_source": "str", "reagent_neutral": "str", "reagent_ion": "str", "class": "str",
    # the class window: its lowest and last energy (NCE, the last's E_cm), how
    # many energies, and at its last one the ion's own products' share of ion
    # + own products (1 - F, less 'another ion's product')
    "nce_lo": "float64", "nce_hi": "float64", "ecm_hi": "float64", "n_energies": "int64",
    "frag": "float64",
    # the channels' shares of the ion's attributed products, pooled over the
    # window (NaN: no product there)
    "share_reagent_ion": "float64", "share_reagent_loss": "float64",
    "share_reagent_loss_small": "float64", "share_monomer": "float64",
    "share_split": "float64", "share_skeletal": "float64", "share_other": "float64",
    "share_unexplained": "float64",
    # monomer shedding's largest share of ion + own products at an energy
    # the precursor is seen at, and its largest split ('kept + shed')
    "monomer_pop": "float64", "halves": "str",
    # the energy the ion falls to half by -- the earlier of F's and S's
    # crossing, compared on NCE (the one axis both lie on); F on the ion's
    # own products where another ion's product is given to it -- as
    # `e50_nce` and its E_cm; for a crossing not bracketed by two observed
    # energies an upper bound ('below_range': at or below the lowest energy;
    # 'single_energy': a one-energy F below half, at or below that energy;
    # 'gap': by the first energy observed after an unobserved one), and with
    # volts at one energy only, whatever the flag, the E_cm of the nearest
    # spectrum with volts at or above the crossing (an upper bound, said in
    # the note where it is not the crossing's own) -- and that curve's 16-84 % band (per-scan tables; else NaN; NaN
    # for F read on the ion's own products: the bootstrap's is of the
    # breakdown's F)
    "e50_curve": "str", "e50_nce": "float64", "e50_ecm": "float64", "e50_flag": "str",
    "e50_ecm_lo": "float64", "e50_ecm_hi": "float64",
    "evidence": "str", "note": "str",
}


def _ecm_at(nce: float, axis: pd.DataFrame) -> tuple[float, bool]:
    """(the ion's E_cm at `nce`, whether it is a spectrum's own): read
    linearly off the window's spectra whose volts are known (`axis`: nce,
    e_cm) -- between two of them, or, outside them, on the straight line
    through them (HCD volts are linear in NCE at one m/z: a spectrum at the
    bottom or the top of the ladder with no volts, a one-scan spectrum of a
    scan with no trailer row, is read so) when two energies or more have
    volts; (NaN, False) where fewer do."""
    a = axis[np.isfinite(axis["e_cm"].to_numpy(dtype=float))]
    if not len(a) or not math.isfinite(nce):
        return math.nan, False
    x, y = a["nce"].to_numpy(dtype=float), a["e_cm"].to_numpy(dtype=float)
    o = np.argsort(x)
    x, y = x[o], y[o]
    if x[0] - 1e-9 <= nce <= x[-1] + 1e-9:
        return float(np.interp(nce, x, y)), bool(np.isclose(x, nce, rtol=0, atol=1e-9).any())
    if len(np.unique(x)) < 2:
        return math.nan, False
    k, b = np.polyfit(x, y, 1)
    return float(k * nce + b), False


def _ecm_bound(nce: float, axis: pd.DataFrame, side: str) -> tuple[float, bool, float]:
    """(E_cm, a spectrum's own, the NCE of the spectrum a bound is taken
    from): `_ecm_at` (NaN NCE: no bound), else -- fewer than two energies
    with volts and `nce` outside them -- a bound off the one there is (E_cm
    rises with NCE): side 'lo', the E_cm of the highest spectrum with volts
    at or below `nce` (a lower bound); 'hi', of the lowest at or above it
    (an upper bound); (NaN, False, NaN) with none."""
    v, own = _ecm_at(nce, axis)
    if math.isfinite(v) or not math.isfinite(nce):
        return v, own, math.nan
    a = axis[np.isfinite(axis["e_cm"].to_numpy(dtype=float))]
    x, y = a["nce"].to_numpy(dtype=float), a["e_cm"].to_numpy(dtype=float)
    m = np.flatnonzero(x <= nce + 1e-9 if side == "lo" else x >= nce - 1e-9)
    if not len(m):
        return math.nan, False, math.nan
    j = m[np.argmax(y[m])] if side == "lo" else m[np.argmin(y[m])]
    return float(y[j]), False, float(x[j])


def _bound_txt(nb: float, nce: float, side: str) -> str:
    """How an E_cm taken as a bound off one spectrum (`_ecm_bound`) is said."""
    return (f"a bound: NCE {nb:.4g}'s, the nearest energy with volts "
            f"{'below' if side == 'lo' else 'above'} NCE {nce:.4g}, whose own is unread -- too "
            "few energies have volts")


def _own_f(ch: pd.DataFrame) -> pd.DataFrame:
    """F on the ion's OWN products, per spectrum (spectrum, nce, e_cm, F;
    lowest energy first): precursor / (precursor + every product but
    'another ion's product'), NaN where neither is there (the ion is
    unobserved on its own products)."""
    piv = ch.pivot_table(index=["nce", "spectrum"], columns="channel", values="intensity",
                         aggfunc="sum", sort=True).fillna(0.0)
    i = (piv[INTACT] if INTACT in piv else pd.Series(0.0, index=piv.index)).to_numpy(dtype=float)
    names = [c for c in piv.columns if c not in (INTACT, OTHER_ION)]
    w = i + (piv[names].sum(axis=1).to_numpy(dtype=float) if names else 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        f = np.where(w > 0, i / w, np.nan)
    ecm = ch.drop_duplicates("spectrum").set_index("spectrum")["e_cm"]
    specs = [k[1] for k in piv.index]
    return pd.DataFrame({"spectrum": specs, "nce": [k[0] for k in piv.index],
                         "e_cm": ecm.reindex(specs).to_numpy(dtype=float), "F": f})


def _own_e50(e50: pd.DataFrame, own: pd.DataFrame) -> pd.DataFrame:
    """`e50` (`ms2_e50.csv` rows) with its F row read on F of the ion's own
    products (`own`, `_own_f`) by the breakdown's rule (`breakdown.e50`):
    its E_cm interpolated off the spectra's E_cm with the crossing's
    fraction, as the breakdown's is off their volts (E_cm is linear in volts
    at one m/z: the same value), NaN where a bracketing spectrum has none;
    its band NaN (the bootstrap's is of the breakdown's F)."""
    from peaky.ms2 import breakdown as BD
    out = e50.copy()
    m = (out["curve"] == "F").to_numpy()
    if not m.any():
        return out
    e = BD.e50(own["nce"], own["F"], own["e_cm"])
    ok = own[np.isfinite(own["F"].to_numpy(dtype=float))]
    upd = {"e50_nce": e.nce, "e50_v": math.nan, "e50_ecm": e.volts, "flag": e.flag,
           "monotonic": e.monotonic, "n_energies": e.n_energies, "nce_min": e.nce_min,
           "nce_max": e.nce_max,
           "value_at_min": float(ok["F"].iloc[0]) if len(ok) else math.nan,
           "value_at_max": float(ok["F"].iloc[-1]) if len(ok) else math.nan}
    upd.update({c: math.nan for c in ("e50_nce_lo", "e50_nce_hi", "e50_v_lo", "e50_v_hi",
                                      "e50_ecm_lo", "e50_ecm_hi", "boot_cross")})
    for c, v in upd.items():
        if c in out:
            out.loc[m, c] = v
    return out


def _curve_values(ch: pd.DataFrame, curves: pd.DataFrame | None,
                  own: pd.DataFrame | None = None) -> pd.DataFrame:
    """The ion's F and S per energy (nce, F, S; NaN where unobserved): the
    breakdown's rows (`curves`: nce, f, s), else read off its channel rows --
    F = precursor / (precursor + products); S = precursor / the lowest
    energy's, where the ion is observed. `own` (`_own_f`): F on the ion's
    own products instead, spectrum by spectrum."""
    if curves is not None and len(curves):
        c = curves.sort_values("nce", kind="stable")
        out = pd.DataFrame({"nce": c["nce"].to_numpy(dtype=float),
                            "F": c["f"].to_numpy(dtype=float),
                            "S": c["s"].to_numpy(dtype=float)})
        at = (c["spectrum"].to_numpy() if "spectrum" in c else None)
    else:
        piv = ch.pivot_table(index=["nce", "spectrum"], columns="channel", values="intensity",
                             aggfunc="sum", sort=True).fillna(0.0)
        i = (piv[INTACT] if INTACT in piv else pd.Series(0.0, index=piv.index)
             ).to_numpy(dtype=float)
        names = [c for c in piv.columns if c != INTACT]
        w = i + (piv[names].sum(axis=1).to_numpy(dtype=float) if names else 0.0)
        ref = float(i[0]) if len(i) and i[0] > 0 else math.nan
        with np.errstate(invalid="ignore", divide="ignore"):
            f = np.where(w > 0, i / w, np.nan)
            s = np.where(w > 0, i / ref, np.nan)
        out = pd.DataFrame({"nce": [k[0] for k in piv.index], "F": f, "S": s})
        at = np.array([k[1] for k in piv.index])
    if own is not None:
        if at is not None:
            out["F"] = own.set_index("spectrum")["F"].reindex(at).to_numpy(dtype=float)
        else:
            out["F"] = own.groupby("nce")["F"].first().reindex(out["nce"]).to_numpy(dtype=float)
    return out


@dataclass(frozen=True)
class _Reading:
    """One curve's E50 as a bracket on NCE: it lies in [lo, hi] (-inf: at or
    below the lowest energy; inf: above every energy the curve is seen at),
    `ecm` the breakdown's E_cm of an interpolated crossing (NaN: not read)."""
    curve: str
    flag: str
    lo: float
    hi: float
    ecm: float = math.nan


def _readings(e50: pd.DataFrame, curves: pd.DataFrame | None = None) -> list[_Reading]:
    """F's and S's E50 brackets (`ms2_e50.csv` rows): an interpolated
    crossing [x, x]; 'below_range' [-inf, its lowest energy]; a one-energy F
    [-inf, it] below half there, [it, inf] at half or more (a one-energy S
    is its own reference: none); 'above_range' [its highest energy, inf];
    'gap' [the last energy at or above half, the first below it after the
    unobserved ones] -- read off the curve's values (`curves`: nce, F, S;
    none without them); no data: none."""
    from peaky.ms2 import breakdown as BD
    out = []
    for curve in ("F", "S"):
        r = e50[e50["curve"] == curve]
        if not len(r):
            continue
        r = r.iloc[0]
        flag = str(r["flag"])
        lo_e, hi_e = float(r["nce_min"]), float(r["nce_max"])
        if flag == "interpolated":
            x = float(r["e50_nce"])
            out.append(_Reading(curve, flag, x, x, float(r["e50_ecm"])))
        elif flag == "below_range":
            out.append(_Reading(curve, flag, -math.inf, lo_e))
        elif flag == "single_energy" and curve == "F":
            below = float(r["value_at_min"]) < 0.5
            out.append(_Reading(curve, flag, -math.inf if below else lo_e,
                                lo_e if below else math.inf))
        elif flag == "above_range":
            out.append(_Reading(curve, flag, hi_e, math.inf))
        elif flag == "gap" and curves is not None and len(curves):
            _, _, _, (xa, xb, _, _) = BD.e50_many(curves["nce"], curves[curve].to_numpy()[:, None],
                                                  bracket=True)
            if math.isfinite(xa[0]) and math.isfinite(xb[0]):
                out.append(_Reading(curve, flag, float(xa[0]), float(xb[0])))
    return out


def _crossing(e50: pd.DataFrame, axis: pd.DataFrame, curves: pd.DataFrame | None = None
              ) -> tuple[str, float, float, str, bool, float]:
    """(curve, NCE, E_cm, flag, read off the NCE axis, bound NCE) of the
    energy the ion falls to half by: the earlier of F's and S's crossing,
    compared on NCE -- both lie on the window's one NCE axis, along which
    the ion's E_cm rises (a crossing whose volts are unknown is no later
    than one whose are) -- its upper bound where it is not bracketed
    (`_readings`: a curve
    below half at the lowest energy, or at its only one, counts as that
    energy; a crossing across an unobserved energy as the first observed
    after it); its E_cm the breakdown's, else read off the NCE axis
    (`_ecm_bound`). F holds only the products the spectra hold, and reads
    high when one leaves the observed m/z range (an ammonium adduct's NH4+);
    S sees that loss, but needs a steady precursor supply. 'Read off the NCE
    axis': no spectrum's own E_cm (a bracketing spectrum has no volts, or
    the bottom or top one), on the line through two energies with volts or
    more; with fewer, the E_cm of the nearest spectrum with volts above it
    (an upper bound: 'bound NCE' that spectrum's NCE, else NaN). ('', NaN,
    NaN, F's flag, False, NaN) when neither crosses."""
    flags = {str(c): str(f) for c, f in zip(e50["curve"], e50["flag"])}
    cross = [r for r in _readings(e50, curves) if math.isfinite(r.hi)]
    if not cross:
        return "", math.nan, math.nan, flags.get("F", ""), False, math.nan
    best = min(cross, key=lambda r: (r.hi, r.curve != "F"))
    if math.isfinite(best.ecm):
        return best.curve, best.hi, best.ecm, best.flag, False, math.nan
    ecm, own, nb = _ecm_bound(best.hi, axis, "hi")
    return (best.curve, best.hi, ecm, best.flag,
            bool(math.isfinite(ecm) and not own and not math.isfinite(nb)), nb)


def _holds(e50: pd.DataFrame, axis: pd.DataFrame, curves: pd.DataFrame | None = None
           ) -> tuple[float, str]:
    """(E_cm, how) the ion is KNOWN to hold half its signal up to, read off
    its curves -- F, and S beyond its reference energy (where it is 1 by
    construction) -- as the lowest of their E50 brackets' lower ends
    (`_readings`; a curve that falls rules the others out), compared on
    NCE: an interpolated crossing; the highest energy a curve that never
    falls to half is seen at ('above_range': its `nce_max`, never an energy
    it is not seen at); a one-energy F's energy when it is half or more
    there; the last energy a curve is seen at or above half before it falls
    across an unobserved energy ('gap'); -inf where one falls at or below
    the lowest energy (`below_range`, a one-energy F below half: it held
    nothing past it). Its E_cm the breakdown's, else read off the NCE axis
    (`_ecm_bound`); with fewer than two energies with volts, the E_cm of the
    nearest spectrum with volts below it -- a lower bound, said so ('at
    least E_cm x eV (a bound: NCE n's ...)'). (NaN, '') when no curve reads,
    or its E_cm cannot be read."""
    rd = _readings(e50, curves)
    if not rd:
        return math.nan, ""
    low = min(rd, key=lambda r: (r.lo, r.curve != "F"))
    if low.lo == -math.inf:
        return -math.inf, ""
    nb = math.nan
    if math.isfinite(low.ecm):
        v, own = low.ecm, True
    else:
        v, own, nb = _ecm_bound(low.lo, axis, "lo")
    if not math.isfinite(v):
        return math.nan, ""
    val = (f"at least E_cm {v:.2f} eV ({_bound_txt(nb, low.lo, 'lo')})" if math.isfinite(nb)
           else f"E_cm {v:.2f} eV" + ("" if own else ", read off the NCE axis"))
    how = {"interpolated": f"{val}, where {low.curve} falls to half",
           "above_range": f"{val}, the highest energy it is seen at (it never falls to half)",
           "single_energy": f"{val}, its only energy",
           "gap": f"{val}, the last energy {low.curve} is seen at half or more before it falls "
                  "across an unobserved one"}[low.flag]
    return v, how


def _pct(v: float) -> str:
    return f"{v * 100:.0f} %" if math.isfinite(v) else "n/a"


def classify(ch: pd.DataFrame, e50: pd.DataFrame, adduct: IonAdduct | None, *,
             floor_mz: float = math.nan, floor_what: str = "the lowest peak they hold",
             ion_mz: float = math.nan, presence: str = "",
             curves: pd.DataFrame | None = None,
             own_products: Mapping[str, float] | None = None) -> dict:
    """One `CLASS_COLUMNS` row: the binding class of one window ion, from its
    channel rows (`window_channels`: every spectrum of one window), its
    `ms2_e50.csv` rows (F, S) and its breakdown rows (`curves`: nce, f, s --
    where a crossing across an unobserved energy lies; else read off the
    channel rows). The class reads the ion's OWN products -- every channel
    but 'another ion's product' (most likely that ion's: no part of what
    this one turns into) -- for the class window, `UNEXPLAINED_MAX` and the
    explained products, ion + own products for `CLUSTER_POP` /
    `NO_MONOMER`, and for the energy test F on its own products (`_own_f`:
    with another ion's product given to it, its F row is re-read so,
    `_own_e50`, and the E50 the row reports is that one's); the shares it
    reports are of all its products, as in `ms2_channels.csv`. HEURISTIC,
    read in this order:

      * no adduct: 'unknown';
      * the ion not observed (`presence`, `analyze`'s: absent from this
        sample, or never seen as a precursor and given only slivers of
        another ion's fragments; or F and S without data): 'unknown';
      * never seen as a precursor, and the products mostly its own
        (`own_products`: per spectrum, the weighted intensity of the
        products whose weight is above `OWN_WEIGHT` -- no equal split, and
        none it reads as 'another ion's product' --, `analyze`'s) at most
        `OWN_WEIGHT` of what it is given in the class window (its lowest
        energy): 'unknown' -- a product mostly its own elsewhere on the
        ladder tells nothing of what it turns into first;
      * a form that adds no reagent group ([M+H]+, [M-H]-): 'covalent';
      * the class window (`FRAG_MIN`, up to the last energy the precursor is
        seen at) never reached -- the ion barely fragments while it is seen:
        'covalent' when it holds half its signal past `COVALENT_ECM` (it
        survives, `_holds`); 'adduct' when S falls to half by `COVALENT_ECM`
        while its reagent ion lies below `floor_mz` (the scans' first mass,
        `FIRST_MASS_MIN`, or a lower peak the spectra hold: `floor_what`; the
        reagent ion left the observed range, M intact) and its products, if
        any, are no non-reagent majority; else 'unknown';
      * most of its own products in the window unexplained
        (`UNEXPLAINED_MAX`): 'unknown';
      * of the explained products (all but 'unexplained' and 'another ion's
        product'), monomer shedding >= `DOMINANT`: 'cluster' when its
        products hold `CLUSTER_POP` of ion + own products, else 'mixed'; the
        reagent channels (reagent ion, reagent loss, + small losses) >=
        `DOMINANT`: 'adduct' -- unless the ion holds half its signal past
        `COVALENT_ECM`: 'covalent' (a covalent ion that releases its
        reagent, as a protonated amide loses NH3, or an adduct bound
        unusually strongly); skeletal >= `DOMINANT`: 'covalent' where it is
        known to hold half its signal to `SKELETAL_ECM` (`_holds`), else
        'unknown' (it falls to half below it, or is not known to reach it);
        'reagent loss + split' >= `DOMINANT`: 'unknown' (a cluster's or a
        covalent dimer's halves); none: 'mixed'.

    The energy the ion falls to half by is the earlier of F's and S's
    crossing, compared on NCE (`_crossing`). `evidence` says what decided,
    with the numbers; `note` the caveats."""
    ch = ch.sort_values("nce", kind="stable")
    out = {c: (math.nan if t == "float64" else (0 if t == "int64" else ""))
           for c, t in CLASS_COLUMNS.items()}
    if len(ch):
        first = ch.iloc[0]
        out.update(parent_mz=float(first["parent_mz"]), ion=str(first["ion"]),
                   role=str(first["role"]), adduct=str(first["adduct"]))
    out["ion_mz"] = float(ion_mz)
    if adduct is not None:
        out.update(adduct=adduct.label, adduct_source=adduct.source,
                   reagent_neutral=adduct.reagent_neutral, reagent_ion=adduct.reagent_ion)
    axis = ch.drop_duplicates("spectrum")[["nce", "e_cm"]]
    # whether the ion's F and S hold any data: the breakdown's, every product
    # it is given
    fl = {str(c): str(f) for c, f in zip(e50["curve"], e50["flag"])}
    # another named ion's products (most likely that ion's) are in the
    # breakdown's F: the energy test reads F on the ion's own products
    oth = ch[ch["channel"] == OTHER_ION]
    has_other = bool(len(oth) and (oth["intensity"].to_numpy(dtype=float) > 0).any())
    own_f = _own_f(ch) if has_other else None
    e50r = _own_e50(e50, own_f) if has_other else e50
    cv = _curve_values(ch, curves, own_f) if len(ch) else None
    curve, x50, ecm50, flag, read, bound = _crossing(e50r, axis, cv)
    out.update(e50_curve=curve, e50_nce=x50, e50_ecm=ecm50, e50_flag=flag)
    band = e50r[e50r["curve"] == curve] if curve else e50r.iloc[0:0]
    if len(band):
        out.update(e50_ecm_lo=float(band.iloc[0]["e50_ecm_lo"]),
                   e50_ecm_hi=float(band.iloc[0]["e50_ecm_hi"]))
    # the breakdown's band of F, given up where F is re-read
    f_band = e50[e50["curve"] == "F"]
    no_band = bool(has_other and curve == "F" and len(f_band)
                   and np.isfinite(f_band[["e50_ecm_lo", "e50_ecm_hi"]].to_numpy(dtype=float)
                                   ).any())
    notes = []
    if read:
        notes.append("the E50's E_cm is read off the NCE axis (the volts at the crossing are "
                     "unknown)")
    if curve == "S":
        notes.append("the ion falls to half first on S, which needs a steady precursor supply "
                     "across the energies")
    # the volts: none, or at one energy only -- said for an ion with no
    # adduct too (its E50's E_cm is then a bound all the same)
    tests = adduct is not None and bool(adduct.label)
    va = axis[np.isfinite(axis["e_cm"].to_numpy(dtype=float))]
    one_volts = va["nce"].nunique() == 1 and len(va) < len(axis)
    if not len(va):
        notes.append("no volts known: " + (f"the energy test ({COVALENT_ECM:g} eV E_cm) cannot "
                                           "be read" if tests else "no E_cm is read"))
    elif one_volts:
        notes.append(f"volts at one energy only (NCE {float(va['nce'].iloc[0]):g}): the others' "
                     "E_cm is not read, only bounded by it (E_cm rises with NCE)"
                     + (f" -- the energy tests ({SKELETAL_ECM:g} and {COVALENT_ECM:g} eV E_cm) "
                        "decide only where that bound settles them" if tests else ""))
    if math.isfinite(bound):
        notes.append(f"the E50's E_cm is a bound: NCE {bound:.4g}'s, the nearest energy with "
                     f"volts above the crossing (NCE {x50:.4g}) -- it falls to half by it at the "
                     "latest")
    if not tests:
        out.update({"class": CLASS_UNKNOWN, "note": "; ".join(notes),
                    "evidence": "no adduct given (--ions 'adduct' column, --adduct, --reagent): "
                                "its products are read as 'product' only"})
        return out
    # per spectrum (lowest energy first): the precursor and the products
    piv = ch.pivot_table(index=["nce", "spectrum"], columns="channel", values="intensity",
                         aggfunc="sum", sort=True).fillna(0.0)
    names = [c for c in PRODUCT_CHANNELS if c in piv.columns]
    prod = piv[names].sum(axis=1) if names else pd.Series(0.0, index=piv.index)
    intact = (piv[INTACT] if INTACT in piv else pd.Series(0.0, index=piv.index)
              ).to_numpy(dtype=float)
    # its own products: all but another named ion's (most likely that ion's)
    other = (piv[OTHER_ION].to_numpy(dtype=float) if OTHER_ION in piv
             else np.zeros(len(piv)))
    own = prod.to_numpy(dtype=float) - other
    whole = intact + own
    with np.errstate(invalid="ignore", divide="ignore"):
        frag = np.where(whole > 0, own / whole, np.nan)
    nce = np.array([k[0] for k in piv.index], dtype=float)
    ecm = np.array([_ecm_at(v, axis)[0] for v in nce], dtype=float)
    # the window: never past the last energy the precursor is seen at (the
    # lowest, for one never seen)
    seen = np.flatnonzero(intact > 0)
    last_seen = int(seen[-1]) if len(seen) else 0
    reach_all = np.flatnonzero(np.nan_to_num(frag, nan=0.0) >= FRAG_MIN)
    reach = reach_all[reach_all <= last_seen]
    late = reach_all[reach_all > last_seen] if not len(reach) else reach_all[:0]
    last = int(reach[0]) if len(reach) else last_seen
    win = piv.iloc[:last + 1]
    wsum = {c: (float(win[c].sum()) if c in win else 0.0) for c in PRODUCT_CHANNELS}
    tot = float(prod.iloc[:last + 1].sum())
    sh = {c: (wsum[c] / tot if tot > 0 else math.nan) for c in PRODUCT_CHANNELS}
    # what the class reads: its own products in the window, and of these the
    # explained ones
    tot_own = max(tot - wsum[OTHER_ION], 0.0)
    explained = max(tot_own - wsum[UNEXPLAINED], 0.0)
    unexpl = wsum[UNEXPLAINED] / tot_own if tot_own > 0 else math.nan
    # a never-seen ion: what of its products in the window is mostly its own
    mine = math.nan
    if not len(seen) and own_products is not None and tot > 0:
        mine = sum(float(own_products.get(k[1], 0.0)) for k in win.index) / tot

    def most(col, upto):
        # the largest share of ion + products a channel holds at an energy
        # the precursor is seen at, up to `upto`, and that energy
        v = (piv[col].to_numpy(dtype=float) if col in piv else np.zeros(len(piv)))
        at_ = [(v[i] / whole[i], nce[i]) for i in (seen if len(seen) else [0])
               if i <= upto and whole[i] > 0]
        return max(at_) if at_ else (math.nan, math.nan)
    # a monomer . reagent ion comes only of a split of the ion itself: read
    # at any energy it is seen at; a [Y+H]+ half may come of [M+H]+ too (a
    # covalent dimer's, cleaved after the reagent left): read in the window
    pop, pop_nce = most(MONOMER, len(piv))
    split_pop, _ = most(SPLIT, last)
    halves = ""
    rows_m = ch[(ch["channel"] == MONOMER) & (ch["halves"] != "")]
    if len(rows_m):
        halves = str(rows_m.sort_values("intensity", ascending=False).iloc[0]["halves"]
                     ).split("; ")[0]
    out.update(nce_lo=float(nce[0]), nce_hi=float(nce[last]), ecm_hi=float(ecm[last]),
               n_energies=last + 1, frag=float(frag[last]),
               share_reagent_ion=sh[REAGENT_ION], share_reagent_loss=sh[REAGENT_LOSS],
               share_reagent_loss_small=sh[REAGENT_LOSS_SMALL], share_monomer=sh[MONOMER],
               share_split=sh[SPLIT], share_skeletal=sh[SKELETAL], share_other=sh[OTHER_ION],
               share_unexplained=sh[UNEXPLAINED], monomer_pop=pop, halves=halves)
    # how far up it holds half its signal (`_holds`): past COVALENT_ECM rules
    # an adduct out; a crossing at or below the lowest energy bounds the E50
    # from above only
    holds_to, held = _holds(e50r, axis, cv)
    holds = math.isfinite(holds_to) and holds_to > COVALENT_ECM
    fin = ecm[np.isfinite(ecm)]
    if has_other:
        notes.append("another named ion's products are left out of the class window, of "
                     "UNEXPLAINED_MAX, of monomer_pop and of the energy test ('another ion's "
                     "product': most likely that ion's): the energy test reads F on its own "
                     "products, not ms2_e50.csv's F, which counts them"
                     + ("; so no band (ms2_e50.csv's is of that F)" if no_band else ""))
    split_rows = ch[(ch["channel"] == SPLIT) & (ch["intensity"] > 0)]
    ties = split_rows["halves"].str.contains(" or ", regex=False)
    if ties.any():
        notes.append("a 'reagent loss + split' product is also read as a monomer . reagent ion "
                     "-- a half of M holds the reagent neutral's atoms (a 14N amide or "
                     "organonitrate, an organobromide, ...): composition cannot tell; it decides "
                     "no cluster, and its halves name both readings")
    # a tie in the class window: a split there may be a monomer . reagent ion
    tie_win = bool((ties & (split_rows["nce"].to_numpy(dtype=float) <= nce[last] + 1e-9)).any())
    top = float(ecm[last_seen]) if math.isfinite(ecm[last_seen]) else math.nan
    its = "its own products (another named ion's left out)" if has_other else "its products"
    pw = "own products" if has_other else "products"
    after = ""
    if flag == "gap":
        gap = [r for r in _readings(e50r, cv) if r.curve == curve]
        lo_e = _ecm_bound(gap[0].lo, axis, "lo")[0] if gap else math.nan
        after = (f", across an unobserved energy, after E_cm {lo_e:.2f} eV"
                 if math.isfinite(lo_e) else ", across an unobserved energy")
    at = (f"E_cm {ecm50:.2f} eV ({curve}"
          + {"below_range": ", at or below the lowest energy",
             "single_energy": ", at or below its only energy", "gap": after}.get(flag, "")
          + (f"; {_bound_txt(bound, x50, 'hi')}" if math.isfinite(bound) else "")
          + ")" if curve and math.isfinite(ecm50) else "")
    win_txt = (f"NCE {nce[0]:g}" if last == 0 else f"NCE {nce[0]:g}-{nce[last]:g}")
    shares = (f"reagent ion {_pct(sh[REAGENT_ION])}, reagent loss {_pct(sh[REAGENT_LOSS])}, "
              f"+ small losses {_pct(sh[REAGENT_LOSS_SMALL])}, monomer shedding "
              f"{_pct(sh[MONOMER])}, + split {_pct(sh[SPLIT])}, skeletal "
              f"{_pct(sh[SKELETAL])}, another ion's {_pct(sh[OTHER_ION])}, unexplained "
              f"{_pct(sh[UNEXPLAINED])}")
    # the channel groups' shares of its own explained products in the window
    rl, ms, sk, sp = ((sum(wsum[c] for c in grp) / explained if explained > 0 else math.nan)
                      for grp in (REAGENT_CHANNELS, (MONOMER,), (SKELETAL,), (SPLIT,)))
    cls, ev = CLASS_UNKNOWN, ""
    if presence:
        ev = presence
    elif fl.get("F", "no_data") == "no_data" and fl.get("S", "no_data") == "no_data":
        ev = ("it is observed in no spectrum of the window (no precursor and no product it is "
              "given: F and S without data): no class")
    elif math.isfinite(mine) and mine <= OWN_WEIGHT:
        ev = (f"never seen as a precursor in the window, and the products mostly its own hold "
              f"{_pct(mine)} of what it is given in its class window ({win_txt}), the lowest "
              f"energy -- at most {OWN_WEIGHT:g} (a product is mostly its own with a weight "
              f"above {OWN_WEIGHT:g}): its shares there are slivers of another ion's fragments "
              "read through its formula, and one of its own elsewhere tells nothing of what it "
              "turns into first (no class)")
    elif not adduct.has_group:
        cls = CLASS_COVALENT
        ev = (f"an {adduct.label} ion adds no reagent group: it has nothing to lose but its "
              "skeleton (every product skeletal; a dimer ion shedding a monomer cannot be told "
              "from a skeletal split by composition)")
    elif not len(reach) or tot_own <= 0:
        mx = (float(np.nanmax(frag[:last + 1])) if np.isfinite(frag[:last + 1]).any() else 0.0)
        rion = SF.parse_ion(adduct.reagent_ion) if adduct.reagent_ion else None
        rmz = float(C.ion_mass(*rion)) if rion else math.nan
        if holds:
            cls = CLASS_COVALENT
            ev = (f"survives: {its} never reach {FRAG_MIN:g} of ion + {pw} (at most "
                  f"{_pct(mx)}), and it holds half its signal to {held}, past "
                  f"{COVALENT_ECM:g} eV, where adducts are gone")
        elif (curve == "S" and math.isfinite(ecm50) and ecm50 <= COVALENT_ECM
              and math.isfinite(rmz) and math.isfinite(floor_mz) and rmz < floor_mz
              and (tot_own <= 0 or (math.isfinite(rl) and rl >= DOMINANT))):
            cls = CLASS_ADDUCT
            ev = (f"falls to half by {at} with {pw} of at most {_pct(mx)} of ion + "
                  f"{pw} in its spectra"
                  + (f" ({shares})" if tot > 0 else "")
                  + f": its reagent ion {adduct.reagent_ion} (m/z {rmz:.3f}) lies below "
                  f"{floor_what} (m/z {floor_mz:.3f}), so it left as the reagent ion, M intact")
        elif len(late):
            ev = (("its precursor is in no spectrum" if not len(seen) else
                   f"no product of its own while its precursor is seen ({win_txt}; at most "
                   f"{_pct(mx)} of ion + {pw})" if has_other else
                   f"no product while its precursor is seen ({win_txt}; at most {_pct(mx)} of "
                   "ion + products)")
                  + f": its products show only after the precursor is gone (from NCE "
                  f"{nce[late[0]]:g}) -- what became of its products, not of the ion: too "
                  "little to tell")
        else:
            ev = (f"barely fragments: {its} never reach {FRAG_MIN:g} of ion + {pw} "
                  f"(at most {_pct(mx)}) up to E_cm "
                  + (f"{top:.2f} eV" if math.isfinite(top) else
                     "unknown (volts at one energy only)" if one_volts else "unknown (no volts)")
                  + (f", and it falls to half by {at}" if at else "")
                  + (f" ({shares})" if tot > 0 else "")
                  + ": too little to tell")
    elif unexpl >= UNEXPLAINED_MAX:
        ev = ((f"most of its own products in the window ({win_txt}) are unexplained "
               f"({_pct(unexpl)} of them; of all it is given: {shares})" if has_other else
               f"most of its products in the window ({win_txt}) are unexplained ({shares})")
              + ": part of the reagent kept, a radical loss, or a co-isolate's products not "
              "named")
    else:
        head = (f"in the window ({win_txt}, {pw} {_pct(out['frag'])} of ion + {pw}): "
                + shares)
        if ms >= DOMINANT:
            if pop >= CLUSTER_POP:
                cls = CLASS_CLUSTER
                ev = (f"sheds whole monomers ({halves} the largest split): {head}; the monomer . "
                      f"reagent ions hold up to {_pct(pop)} of ion + {pw} (NCE {pop_nce:g})")
            else:
                cls = CLASS_MIXED
                ev = (f"sheds whole monomers ({halves} the largest split), but they hold at most "
                      f"{_pct(pop)} of ion + {pw} (< {CLUSTER_POP:g}): {head}; the rest "
                      "holds to higher energies -- a cluster beside a covalent ion of the same "
                      "formula?")
        elif rl >= DOMINANT:
            if holds:
                cls = CLASS_COVALENT
                twin = adduct.label in ("[M+NH4]+", "[M+^NH4]+")
                ev = (f"its products are the reagent's loss ({head}), but it holds half its "
                      f"signal to {held}, past {COVALENT_ECM:g} eV where adducts are gone: a "
                      "covalent ion that releases its reagent at that energy"
                      + (" (a protonated amide losing NH3 -- an ammonium adduct's formula "
                         "twin)" if twin else "")
                      + ", or an adduct bound unusually strongly")
            else:
                cls = CLASS_ADDUCT
                ev = (f"loses the reagent first: {head}"
                      + (f"; falls to half by {at}" if at else ""))
                if adduct.body.get("C", 0) >= 2 * MONOMER_MIN_C:
                    if pop > NO_MONOMER:
                        ev += (f"; it sheds monomers too (up to {_pct(pop)} of ion + {pw}, "
                               f"NCE {pop_nce:g})")
                    elif math.isfinite(split_pop) and split_pop > NO_MONOMER and tie_win:
                        # a split in the window that is as well a monomer .
                        # reagent ion (a half holds the reagent neutral's
                        # atoms): no 'sheds no monomer'
                        ev += (f"; no product reads only as a monomer . reagent ion, but up to "
                               f"{_pct(split_pop)} of ion + {pw} are halves split after the "
                               "reagent's loss or, as well, monomer . reagent ions (composition "
                               "cannot tell)")
                    elif math.isfinite(split_pop) and split_pop > NO_MONOMER:
                        ev += (f"; it sheds no monomer . reagent ion, but up to "
                               f"{_pct(split_pop)} of ion + {pw} are halves split after "
                               "the reagent's loss (a cluster's or a covalent dimer's: "
                               "composition cannot tell)")
                    else:
                        ev += (f"; it sheds no monomer (its monomer . reagent ions at most "
                               f"{_pct(pop if math.isfinite(pop) else 0.0)} of ion + "
                               f"{pw}): M is no cluster of two neutrals")
        elif sk >= DOMINANT:
            # covalent only where it is KNOWN to hold half its signal to
            # SKELETAL_ECM: an E50 bounded from above, read across a hole or
            # with no volts may lie below it
            if math.isfinite(holds_to) and holds_to >= SKELETAL_ECM:
                cls = CLASS_COVALENT
                ev = (f"keeps the reagent and fragments its skeleton: {head}"
                      + (f"; falls to half by {at}" if at else "")
                      + f"; it holds half its signal to {held}")
            elif curve and math.isfinite(ecm50) and ecm50 < SKELETAL_ECM:
                ev = (f"keeps the reagent and fragments its skeleton: {head}; but it falls to "
                      f"half by {at}, below {SKELETAL_ECM:g} eV, where no covalent skeleton of "
                      "ours breaks: an adduct whose products are a co-isolate's or background "
                      "ions? too little to tell")
            else:
                known = "; ".join(t for t in (
                    f"it holds half its signal to {held}" if held else "",
                    f"it falls to half by {at}" if at else "") if t) or (
                    "no energy it falls to half by is read" + ("" if len(fin) else
                                                               " (no volts)"))
                ev = (f"keeps the reagent and fragments its skeleton: {head}; but it is not "
                      f"known to hold half its signal to {SKELETAL_ECM:g} eV ({known}), "
                      "below which no covalent skeleton of ours breaks: too little to tell")
        elif sp >= DOMINANT:
            ev = (f"its products are halves of M split after the reagent's loss ({head}): a "
                  "cluster's monomer shed with the reagent's proton, or a covalent dimer's "
                  "[M+H]+ cleaved -- composition cannot tell")
        else:
            cls = CLASS_MIXED
            ev = f"no channel holds {DOMINANT:g} of the explained products: {head}"
    out.update({"class": cls, "evidence": ev, "note": "; ".join(notes)})
    return out


def _frame(rows, spec: dict[str, str]) -> pd.DataFrame:
    df = pd.DataFrame(rows) if not isinstance(rows, pd.DataFrame) else rows
    return pd.DataFrame({c: pd.Series(df[c].to_numpy() if c in df else [], dtype=t)
                         for c, t in spec.items()})
