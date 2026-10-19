"""Arbitration between RIVAL formulas of one precursor ion: which of two or
more formulas -- all within MS1's tolerance of one m/z, so MS1 cannot tell
them apart -- the window's MS2 products support.

Pure: no Mascope, no I/O. `analyze` reads the window once per rival -- the
rival as its target, the window's other named ions fixed, the window
recalibrated on that rival's own precursor, the same readings, ladder gate
and attribution as any window --, reads every rival again on each other
rival's CALIBRATION (`RivalRead.cross`: a whole read, its products where the
other rival's recalibration puts them), and does the same on each DECOY of
the window (`DECOY_SHIFTS_MDA`); this module turns those reads into scores
and a verdict (`score_window`).

The window's PRODUCTS (`product_table`) are its peaks in the energy-resolved
spectra that no read takes for a window ion's precursor or a harmonic ghost,
less the peaks that are no rival's product (`non_products`): those above the
located precursor less an H atom (no singly charged product of any rival
lies there: an unnamed ion co-isolated in the window, an isotopologue of the
precursor) and those every rival's read gives wholly to a named co-isolate
(its products: no rival can claim them -- a rival that reads one only
through an oxygen chain, below, claims none). One set for every rival:
their precursors are one peak.

What a rival is credited with. Read on the calibration -- its own
precursor's or another rival's -- on which it is credited with the most
(`product_table`: the rivals' ppm gap moves every product between their
calibrations, and which one holds the precursor is an MS1 matter, never
weighed: read on its own alone, the rival the precursor sits off lost its
own products at the tolerance's edge, and a composition tie was
'supported' for the other -- tested), a product's intensity times its
attribution weight (`attribute`: 1 for a fragment only it can make; its
share of one it shares with a named co-isolate) -- where it reads the
product with a
CREDIBLE loss: one whose unnamed remainder, if any, is no carbon-poor
polyoxy composition (at most `attribute.POLYOXY_MAX_C` C and at least
`attribute.POLYOXY_MIN_O` O, nothing but C, H, N, ^N and O) that no merge
of real neutrals makes up (`implausible_remainder`). The decomposer prints
the fewest pieces, so the reagent's neutral lost with common ones prints as
one unnamed piece -- a nitrate cluster's [M-H-CO2]- as a loss of CH^NO5
(H^NO3 + CO2), an ammonium adduct's ^NH3 + CO2 + H2O2 as CH5^NO4 --: undone
(the neutral of the ion's own polarity's reagent, `REAGENT_NEUTRALS`, then
at most `MERGED_MAX` of `MERGED_NEUTRALS` and one O2; CO2 + 2 H2O, three
common neutrals, as one CH4O4 too) it is credible (tested: refused, a true
O-rich adduct or acid lost those products to the siloxane alias that reads
them as Si2 / CO3H-4 twins, and the siloxane was supported; credited, the
two read the same products). What no such merge makes up -- O beyond what
the merged neutrals and one O2 hold: H2O6, CH5^NO7 on an ammonium adduct
(nitric acid is no neutral it sheds) -- the core accepts (an integer DBE,
Senior's rule, an H: `subformula._remainder_ok`), but it is an oxygen
chain, no molecule: an O-rich rival reads the organic products of a
co-isolated ion through it and would be credited with products it cannot
make (tested, synthetic).

Scores per rival, over the window's energies (`rival_scores`): the explained
share of the products' intensity; the products it is credited with; the
DIAGNOSTIC products -- called its own by the attribution (no co-isolate
shares it) and read credibly by no other rival, on no rival's calibration
(a product both formulas hold is read by both on one of them) -- their
count, share and the strongest named with their losses; the mean |ppm| and
the share read with fully named closed-shell losses (the readings' rank
quality); and, for every element it holds that another rival lacks, its
SIGNATURE: the share of what it is credited with that it reads with the
element shown -- in the product, or leaving whole in named library
neutrals, or in a remainder a merge of real neutrals makes up (`shows`:
what `implausible_remainder` credits -- a nitrate cluster's H^NO3 + CO2 +
H2O as one CH3^NO6 -- shows its reagent's ^N); for a reagent's N or ^N,
`REAGENT_ELEMENTS`, also a library neutral that takes all of it out of the
unnamed remainder next to a skeletal piece: ^NH3 + C4H8 prints as ONE
unnamed piece C4H11^N, the fewest pieces, and its ^N still leaves whole;
for O, `SKELETAL_ELEMENTS`, also all of it in one valid skeletal piece of
more than one C -- a skeleton's C-O neutrals, its N only in one reagent
neutral of the ion's polarity, at most `SKELETAL_O_OVER_C` O beyond its C;
for S also as CS, `BARE_NEUTRALS`, next to one library neutral). A carbon-poor product (at most
`attribute.POLYOXY_MAX_C` C: the reagent ion of a cluster that shed its
skeleton whole, as one neutral) that does not show the element says nothing
of it and is not judged: a nitrate cluster of a perfluorinated acid gives
^NO3- with all its F in the acid shed. A signature judged on less than
`SIGNATURE_EXPECT` of the products is none (NaN).

The chance control (`DECOY_SHIFTS_MDA`): every product peak of the window
moved by a few mDa -- by at least `DECOY_TOL_FACTOR` times its reading
tolerance -- (the precursors, the harmonic ghosts and the non-products
kept, so the recalibration, the ghost rule and a co-isolate's own products
are the window's own) and read again by every rival: what a rival 'explains'
there it explains by chance. A rival's `chance` is its mean explained share
over the decoys, a lead's `lead_chance` the decoys' mean of the same
difference; the verdict asks a lead to beat its chance by `SHARE_MARGIN` --
the decoys raise that bar, never lower it (a decoy lead in the other rival's
favour is no chance the real lead has).

The verdict (`decide`): a rival is CONTRADICTED when it holds an element
another rival lacks, reads at least `SIGNATURE_EXPECT` of the products, and
shows that element in at most `SIGNATURE_MIN` of what it reads, judged on at
least `SIGNATURE_EXPECT` of them (a siloxane's products keep Si or lose it
in a named neutral; a rival that reads the products only as Si-free
fragments of Si-bearing remainders is no siloxane). The leader -- the rival
not contradicted that explains the most -- is SUPPORTED when, against every
other rival, its lead less the decoys' mean lead (none when that is
negative) is at least `SHARE_MARGIN`, it has at least `DIAGNOSTIC_MIN`
products of `DIAGNOSTIC_FLOOR` or more that the other cannot read, and its
explained share beats its own chance by `EXCESS_MIN`; the rivals it beats are
OUTSCORED (or contradicted). Else the window is UNDECIDED and every rival
not beaten says why (short of `EXCESS_MIN`: no better than chance, or --
its decoys' mean under `CHANCE_SAID` of its share -- too little of the
products, though above chance). Rivals whose formulas swap atoms within the
reading tolerance (Si2 for CO3H-4, 0.41 mDa; S + Si + H4 for O4, 0.64 mDa)
read a product that holds the swapped atoms in numbers the swap can trade
(an even number of Si, against an Si-free twin) as each other's aliases: on
such products alone, undecided, said (`same products`). Products the swap
cannot trade -- an odd number of Si -- are what can decide the pair: at
least `DIAGNOSTIC_MIN` of them, `DIAGNOSTIC_FLOOR` or more each, with the
share margins (the siloxane whose four odd-Si products only it reads
supported over its Si-free twin; one such product decides nothing: both
tested). An alias that reads the products only through remainders holding
its swapped atoms is contradicted, which supports no rival by itself: the
other must still beat it on products it cannot read -- support is never by
elimination --, and where both read the same products the window stays
undecided (tested).
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from itertools import combinations_with_replacement
from typing import Collection, Mapping, Sequence

import numpy as np
import pandas as pd

from peaky.chem import chemistry as C
from peaky.ms2 import attribute as AT
from peaky.ms2 import subformula as SF

__all__ = [
    "SUPPORTED", "OUTSCORED", "CONTRADICTED", "UNDECIDED", "VERDICTS", "DECOY_SHIFTS_MDA",
    "DECOY_TOL_FACTOR", "SHARE_MARGIN", "DIAGNOSTIC_MIN", "DIAGNOSTIC_FLOOR", "EXCESS_MIN",
    "CHANCE_SAID", "SIGNATURE_MIN", "SIGNATURE_EXPECT", "RIVAL_PPM_MAX", "DIAGNOSTIC_SHOWN",
    "H_ATOM", "REAGENT_ELEMENTS", "BARE_NEUTRALS", "SKELETAL_ELEMENTS", "SKELETAL_O_OVER_C",
    "REAGENT_NEUTRALS", "REAGENT_NEUTRALS_MAX",
    "MERGED_NEUTRALS", "MERGED_MAX", "MISSING_CELLS", "ARBITRATION_COLUMNS",
    "RivalGroup", "RivalRead", "parse_rival_spec", "read_rival_groups", "check_group",
    "check_groups", "implausible_remainder", "shows", "non_products", "product_table",
    "rival_scores", "decide", "score_window", "undecided_rows",
]

SUPPORTED = "supported"
OUTSCORED = "outscored"
CONTRADICTED = "contradicted"
UNDECIDED = "undecided"
VERDICTS = (SUPPORTED, OUTSCORED, CONTRADICTED, UNDECIDED)

#: The decoys: every product peak moved by these (mDa), the precursors,
#: harmonic ghosts and non-products kept -- the mass-shift decoy the ladder
#: gate was checked with (PR #80: +-7..19 mDa), short of a nominal mass and
#: beyond the reading tolerance of a product below m/z 700 at the default 5
#: ppm (`DECOY_TOL_FACTOR` scales them up above it). Shifts on an alias
#: spacing (Si for CO, 18.0 mDa; CO for N2, 11.2) land a real fragment on
#: its alias, and a rival rich in those atoms reads it there: the chance is
#: a mean over eight shifts, not one.
DECOY_SHIFTS_MDA = (-19.0, -15.0, -11.0, -7.0, 7.0, 11.0, 15.0, 19.0)
#: A decoy moves a product by at least this many times its reading tolerance
#: (max(tol_ppm x m/z, tol_mda)): a real product read up to a tolerance off
#: lands, moved, a tolerance beyond any reading of itself. Where the
#: smallest shift falls short of it -- a product above m/z 700 at 5 ppm, or
#: any wider --tol-ppm --, every shift of that product is scaled up by one
#: factor (`analyze._shifted`): at --tol-ppm 10 a 7 mDa shift of an m/z 741
#: product (9.4 ppm) stayed inside its tolerance, and that decoy read the
#: real product as itself -- a chance that is none (tested).
DECOY_TOL_FACTOR = 2.0
#: The supported rival's lead over every other rival -- its explained share
#: less the other's -- must beat the decoys' mean of that lead (or 0, when
#: the decoys favour the other rival: they raise the bar, never lower it) by
#: this share of the products' intensity, ...
SHARE_MARGIN = 0.10
#: ... on at least this many products of its own (no named co-isolate shares
#: them) that the other rival cannot read credibly on any rival's
#: calibration, ...
DIAGNOSTIC_MIN = 2
#: ... each holding at least this share of the window's product intensity;
DIAGNOSTIC_FLOOR = 0.01
#: and its explained share must beat its own decoys' mean by this. The four
#: are set on decoy windows read as REAL -- each decoy scored against the
#: other seven, where any support is false -- and on known answers (ions
#: read against every formula within the reading tolerance of them, mostly
#: alias swaps such as Si2 / CO3H-4, CS / OSi, C2F3 / H2^NO4). One diagnostic
#: product is a coin a decoy can toss: with one, or a smaller excess over
#: chance, markedly more decoy windows are called supported, and a decoy
#: window whose products one reagent ion dominates (a shift lands it on
#: another composition) is the hardest case; three products, or a wider
#: margin, lose known answers' supports. Read as real, a decoy window is
#: still supported now and then: none of the 32 of the committed fixtures'
#: four windows, but 9 of the 72 decoys of the synthetic windows the tests
#: pin -- a shift on an alias spacing (-11 mDa: CO for N2, 11.2; +-19: about
#: Si for CO, 18.0; +11 / +15: the ^NH3 loss for the H2O loss, 13.0, on
#: another rival's calibration) lands a window's few products on a rival's
#: compositions, at times on the rival the window itself does not support
#: (an O-rich adduct's products on its siloxane alias; tested). EXCESS_MIN was 0.2
#: while the products held peaks no rival can have made (an unnamed ion
#: co-isolated in the window, a named co-isolate's own products): out of
#: the products (`non_products`), every share -- and every decoy's -- reads
#: larger, and 0.35 keeps decoy windows read as real about as rarely
#: supported as before, with more known answers supported.
EXCESS_MIN = 0.35
#: A leader short of `EXCESS_MIN` is said to be no better than chance where
#: its decoys' mean reaches this share of what it explains; else it explains
#: too little of the products, though above chance (product peaks no rival
#: reads -- an unnamed ion's products, noise -- dilute every share; tested).
CHANCE_SAID = 0.5
#: A rival is contradicted when it shows an element another rival lacks in
#: at most this share of what it is credited with ...
SIGNATURE_MIN = 0.3
#: ... where it is credited with at least this share of the products, and
#: the signature is judged on at least this share of them (below it, too
#: little is read to expect anything). A true ion's products keep its
#: skeleton's elements or lose them whole in library neutrals (the reagent
#: N of an ammonium adduct as ^NH3, alone or with a skeletal piece), or in
#: an H-free neutral the library lacks (`BARE_NEUTRALS`: a thiazole's ring S
#: as CS): in the windows tested its signature sits above SIGNATURE_MIN --
#: an element lost in a neutral `shows` does not know reads as not shown.
#: A siloxane formula read against a window of organic products shows Si
#: only through an Si2 / CO3H-4 alias of an O-rich product, if at all: in
#: the synthetic window of O-poor organic products, in none of them
#: (tested).
SIGNATURE_EXPECT = 0.05
#: The widest window `--rival-ppm` may set (ppm between two rivals): the reach
#: `ms2 extract` reads a folder's axis offset over (`sources.AXIS_SEARCH_PPM`).
#: Two formulas farther apart are two ions an MS1 axis can tell apart.
RIVAL_PPM_MAX = 50.0
#: How many diagnostic products a row names (the strongest first).
DIAGNOSTIC_SHOWN = 3
#: The H atom's mass (Da): the lightest loss there is, so no singly charged
#: product of an ion lies above its precursor less this (`non_products`).
H_ATOM = 1.00782503207

ARBITRATION_COLUMNS: dict[str, str] = {
    "parent_mz": "float64", "group": "str", "rival": "str", "rival_mz": "float64",
    # 1 = the rival named first; whether the run's other tables read the
    # window with it as its target (`analyze`: the first rival, or the
    # supported one with --pick-supported -- where the window's located
    # precursor matches it, as in the run with it alone named)
    "order": "int64", "analysed": "bool",
    "window_verdict": "str", "winner": "str", "verdict": "str",
    # the share of the window's product intensity it is credited with, its
    # decoys' mean and largest, and the lead over the best other rival with
    # the decoys' mean of that lead
    "explained": "float64", "chance": "float64", "chance_max": "float64",
    "lead": "float64", "lead_chance": "float64",
    "n_products": "int64", "n_diagnostic": "int64", "diagnostic_share": "float64",
    "diagnostic_products": "str", "diagnostic_decoy": "float64",
    # the readings' quality: credited intensity read with fully named
    # closed-shell losses, mean |ppm|; given it by the attribution but read
    # only through a polyoxy remainder (not credited)
    "named_share": "float64", "mean_abs_ppm": "float64", "uncredited_share": "float64",
    # the located precursor's error under this rival on the folder's m/z
    # axis (MS2), and that of the MS1 window centroid at the located
    # precursor (NaN: none within the target match); the rival on whose
    # precursor's calibration its scores are read (its own, or another
    # rival's where it is credited with more there: `product_table`), and
    # the share it is credited with on its own -- as the run's other tables,
    # recalibrated on its precursor, read it (`explained` where `read_on`
    # is itself)
    "precursor_ppm": "float64", "ms1_ppm": "float64", "read_on": "str",
    "explained_own": "float64",
    # the elements it holds that another rival lacks, and their signature
    "elements": "str", "signature": "str",
    "product_intensity": "float64", "n_spectra": "int64", "n_decoys": "int64",
    "evidence": "str", "note": "str",
}


# ---------------------------------------------------------------------------
# the rivals
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RivalGroup:
    """Rival readings of ONE ion: `rivals` (normalised ion formulas, in the
    order named: the first is the one the run's other tables read unless
    --pick-supported picks the supported one), `name` (the CSV's group label,
    else the m/z cell or --rival m/z as typed), `source` (where it was
    named)."""
    name: str
    rivals: tuple[str, ...]
    source: str = ""


def _norm(formula: str) -> str:
    counts, charge = SF.parse_ion(formula)
    return SF.format_ion(counts, charge)


def _mz(formula: str) -> float:
    counts, charge = SF.parse_ion(formula)
    return float(C.ion_mass(counts, charge))


def parse_rival_spec(spec: str, *, check_ppm: float, check_mda: float
                     ) -> tuple[float, list[str]]:
    """'315.106=C8H28^NO4Si4+,C10H20^NO10+' -> (315.106, [formulas,
    normalised]). The typed m/z must lie within the rivals' span -- from the
    lightest formula's exact m/z to the heaviest's -- widened by
    max(`check_ppm`, `check_mda`) (as `--ion` checks one formula: a typo
    otherwise); how far apart the rivals may lie is `check_group`'s (with
    `--rival-ppm`), so the m/z typed between two rivals 20 ppm apart is no
    typo. At least two distinct formulas."""
    text = str(spec).strip()
    mz_txt, sep, rest = text.partition("=")
    if not sep:
        raise ValueError(f"--rival {spec!r}: expected MZ=FORMULA,FORMULA[,...], e.g. "
                         "315.106=C8H28^NO4Si4+,C10H20^NO10+")
    try:
        typed = float(mz_txt)
    except ValueError:
        raise ValueError(f"--rival {spec!r}: {mz_txt!r} is not an m/z") from None
    out: list[str] = []
    for f in (x.strip() for x in rest.split(",")):
        if not f:
            continue
        norm = _norm(f)
        if norm in out:
            raise ValueError(f"--rival {spec!r}: {norm} is named twice")
        out.append(norm)
    if len(out) < 2:
        raise ValueError(f"--rival {spec!r}: a rival group needs two formulas or more")
    exact = {f: _mz(f) for f in out}
    tol = max(check_ppm * typed * 1e-6, check_mda * 1e-3)
    if not min(exact.values()) - tol <= typed <= max(exact.values()) + tol:
        near = min(out, key=lambda f: abs(typed - exact[f]))
        raise ValueError(f"--rival {spec!r}: the m/z given ({typed:g}) is "
                         f"{(typed - exact[near]) * 1e3:+.1f} mDa from {near} (m/z "
                         f"{exact[near]:.5f}), the nearest rival; a typo?")
    return typed, out


#: The cells a CSV writer leaves for a missing value (R's NA, pandas' and
#: numpy's NaN, a spreadsheet's #N/A; case ignored): no group, no m/z.
MISSING_CELLS = frozenset({"", "na", "n/a", "#n/a", "nan", "-nan", "null", "none", "<na>"})


def _missing(cell) -> bool:
    return str(cell).strip().lower() in MISSING_CELLS


def _no_mz(cell) -> bool:
    """An m/z cell that names no m/z: missing, or a number that is not finite
    ('inf'), as `analyze.read_ions_csv` reads it."""
    if _missing(cell):
        return True
    try:
        return not math.isfinite(float(str(cell).strip()))
    except ValueError:
        return False


def read_rival_groups(df: pd.DataFrame, where: str) -> list[RivalGroup]:
    """The rival groups of an `--ions` CSV (read as text, `formula` column):
    rows sharing a `group` cell, and rows with no group cell that share one
    `mz` cell (the same text), are rival readings of one ion -- a missing
    value (`MISSING_CELLS`: R's NA, 'nan') is no cell, and an m/z cell that
    is no finite number names no m/z, as the ion reader takes it (tested). A
    group cell naming one formula only is refused, its row named; the same
    formula in two groups too. -> groups in the order first named."""
    if "formula" not in df.columns:
        return []
    has_group = "group" in df.columns
    has_mz = "mz" in df.columns
    by: dict[tuple[str, str], list[tuple[int, str]]] = {}
    for i, r in df.iterrows():
        f = str(r["formula"]).strip()
        if not f:
            continue
        g = "" if not has_group or _missing(r["group"]) else str(r["group"]).strip()
        if g:
            by.setdefault(("group", g), []).append((i + 2, _norm(f)))
        elif has_mz and not _no_mz(r["mz"]):
            by.setdefault(("mz", str(r["mz"]).strip()), []).append((i + 2, _norm(f)))
    out, seen = [], {}
    for (kind, label), rows in by.items():
        forms = list(dict.fromkeys(f for _, f in rows))
        if len(forms) < 2:
            if kind == "group":
                raise ValueError(f"{where} row {rows[0][0]}: group {label!r} names one formula "
                                 "only (a rival group needs two or more)")
            continue
        for row, f in rows:
            if f in seen and seen[f] != label:
                raise ValueError(f"{where} row {row}: {f} is a rival in two groups "
                                 f"({seen[f]!r} and {label!r})")
            seen[f] = label
        src = "ions csv group" if kind == "group" else "ions csv m/z"
        out.append(RivalGroup(name=label, rivals=tuple(forms), source=src))
    return out


def check_group(group: RivalGroup, *, tol_ppm: float, tol_mda: float,
                rival_ppm: float | None = None) -> None:
    """Refuse a group whose rivals MS1 could tell apart: every two of them
    must lie within the reading tolerance of each other (max(`tol_ppm`,
    `tol_mda`) at their m/z) -- or within `rival_ppm` (`--rival-ppm`, at most
    `RIVAL_PPM_MAX`), a wider window stated for an axis known to read
    worse. One polarity."""
    if rival_ppm is not None and not (math.isfinite(rival_ppm) and 0 < rival_ppm
                                      <= RIVAL_PPM_MAX):
        raise ValueError(f"--rival-ppm must be in (0, {RIVAL_PPM_MAX:g}] ppm, got {rival_ppm!r}")
    if len(group.rivals) < 2:
        raise ValueError(f"rival group {group.name!r}: two formulas or more are needed")
    if len({f[-1] for f in group.rivals}) > 1:
        raise ValueError(f"rival group {group.name!r}: rivals of both polarities")
    mzs = {f: _mz(f) for f in group.rivals}
    for a, b in ((a, b) for i, a in enumerate(group.rivals) for b in group.rivals[i + 1:]):
        m = max(mzs[a], mzs[b])
        tol = SF._tolerance_da(m, tol_ppm, tol_mda)
        wide = (rival_ppm or 0.0) * m * 1e-6
        d = abs(mzs[a] - mzs[b])
        if d > max(tol, wide):
            # --rival-ppm offered only where it can help: up to its cap
            need = math.ceil(d / m * 1e7) / 10.0
            raise ValueError(
                f"rival group {group.name!r}: {a} and {b} lie {d * 1e3:.2f} mDa "
                f"({d / m * 1e6:.1f} ppm) apart, beyond the reading tolerance ({tol * 1e3:.2f} "
                "mDa)" + (f" and --rival-ppm {rival_ppm:g}" if rival_ppm else "")
                + ": MS1 tells them apart, so they are two ions, not rivals of one "
                + ("(they share an m/z cell of the --ions CSV, which names rivals: give each "
                   "its own m/z if they are two ions" if group.source == "ions csv m/z" else
                   "(name them as ions")
                + (f", or pass --rival-ppm {need:g} or more for an axis known to read worse)"
                   if need <= RIVAL_PPM_MAX else ")"))


def check_groups(groups: Sequence[RivalGroup]) -> None:
    """Refuse two rival groups of one name (a CSV group label, a CSV group's
    shared m/z cell or a --rival m/z, as typed: two --rival typed alike, or
    a label typed as an m/z) and one formula in two groups (a window reads
    one rival per group: a formula in two would be read and left out at
    once). Two groups typed alike are most likely the rivals of ONE ion
    split in two -- which, typed apart, a window would read as one group,
    the other never arbitrated: the refusal says to name them in one group
    first (tested)."""
    names: set[str] = set()
    seen: dict[str, str] = {}
    for g in groups:
        if g.name in names:
            raise ValueError(f"two rival groups are named {g.name!r} (a --rival m/z, an --ions "
                             "CSV group label or a CSV group's shared mz cell, as typed): the "
                             "rivals of ONE ion go in one group -- one --rival MZ=F1,F2,... or "
                             "one CSV group --; only if they are two ions, type each --rival its "
                             "own m/z or give the CSV groups distinct labels")
        names.add(g.name)
        for f in g.rivals:
            if f in seen:
                raise ValueError(f"{f} is a rival in two groups ({seen[f]!r} and {g.name!r})")
            seen[f] = g.name


# ---------------------------------------------------------------------------
# one reading
# ---------------------------------------------------------------------------
def _remainder(loss_name: str, unnamed: int) -> dict[str, int]:
    """The unnamed remainder of a reading's loss (printed last, `subformula`),
    as counts; {} when every piece is named."""
    if not unnamed or not loss_name:
        return {}
    return {e: n for e, n in C.parse_formula(str(loss_name).split(" + ")[-1]).items() if n}


_POLYOXY = frozenset({"C", "H", "N", "^N", "O"})


#: The neutral each polarity's reagent leaves as -- ammonium and urea for a
#: positive ion, nitric acid for a negative one, 15N or not --, the ion's own
#: polarity's only (`implausible_remainder`: on an ammonium adduct H^NO3 +
#: CO2 + 2 H2O would read CH5^NO7, an oxygen chain, as a molecule) ...
REAGENT_NEUTRALS: dict[int, tuple[str, ...]] = {1: ("H3^N", "H3N", "CH4N2O"),
                                                -1: ("H^NO3", "HNO3")}
#: ... at most this many of them (a dimer cluster sheds two), ...
REAGENT_NEUTRALS_MAX = 2
#: ... and the common closed-shell neutrals an O-rich ion loses with them,
#: at most `MERGED_MAX`, next to at most one O2 (a peroxy group's). Three:
#: an O-rich acid's CO2 + 2 H2O prints as ONE unnamed CH4O4 (a nitrate
#: cluster's H^NO3 + CO2 + 2 H2O as CH5^NO7); refused at two, the true acid
#: lost those products to its siloxane alias -- which reads them through Si-H
#: remainders this rule never polices -- and the alias was supported
#: (tested, both polarities).
MERGED_NEUTRALS = ("H2O", "CO2", "H2O2", "CO")
MERGED_MAX = 3


def _sign(ion: str) -> int:
    q = SF.parse_ion(ion)[1]
    return (q > 0) - (q < 0)


def implausible_remainder(loss_name: str, unnamed: int, *, charge: int = 0) -> bool:
    """Is the reading's unnamed remainder a carbon-poor polyoxy composition
    (at most `attribute.POLYOXY_MAX_C` C, at least `POLYOXY_MIN_O` O,
    nothing but C, H, N, ^N, O) that no merge of real neutrals makes up? The
    decomposer prints the fewest pieces, so a reagent's neutral lost with
    common ones prints as ONE unnamed piece -- ^NH3 + CO2 + H2O2 as CH5^NO4,
    H^NO3 + CO2 as CH^NO5, H^NO3 + H2O as H3^NO4 (a nitrate cluster's
    [M-H-H2O]-), and common ones alone too (CO2 + 2 H2O as CH4O4) --:
    undone (`_undone`: the reagent neutrals of the ion's polarity,
    `charge`'s sign -- none for 0 --, then common neutrals and one O2) it is
    credible. What no such merge makes up is an oxygen chain, no molecule --
    O beyond what the merged neutrals and one O2 hold (H2O6: H2O2 and two
    O2; CH5^NO7 on an ammonium adduct, where nitric acid is no neutral it
    sheds): such a reading is credited to no rival."""
    rem = _remainder(loss_name, unnamed)
    if not (rem and set(rem) <= _POLYOXY and rem.get("C", 0) <= AT.POLYOXY_MAX_C
            and rem.get("O", 0) >= AT.POLYOXY_MIN_O):
        return False
    return not _undone(tuple(sorted(rem.items())), (charge > 0) - (charge < 0))


def _pieces(names: Sequence[str]) -> tuple[Counter, ...]:
    return tuple(Counter(C.parse_formula(n)) for n in names)


@lru_cache(maxsize=1 << 12)
def _undone(remainder: tuple[tuple[str, int], ...], sign: int) -> bool:
    """Do at most `REAGENT_NEUTRALS_MAX` of polarity `sign`'s
    `REAGENT_NEUTRALS`, at most `MERGED_MAX` `MERGED_NEUTRALS` and at most
    one O2 make up `remainder` whole?"""
    rem = Counter(dict(remainder))
    reag = _pieces(REAGENT_NEUTRALS.get(sign, ()))
    common = _pieces(MERGED_NEUTRALS)
    for nr in range(REAGENT_NEUTRALS_MAX + 1):
        for rs in combinations_with_replacement(reag, nr):
            for nc in range(MERGED_MAX + 1):
                for cs in combinations_with_replacement(common, nc):
                    got = sum((*rs, *cs), Counter())
                    if got == rem or got + Counter({"O": 2}) == rem:
                        return True
    return False


#: The elements a CIMS reagent brings into an ion (ammonium, urea, nitrate,
#: and their 15N forms): they leave as a library neutral (NH3, urea, HNO3)
#: that the decomposer, printing the fewest pieces, merges with a skeletal
#: loss -- ^NH3 + C4H8 prints as ONE unnamed C4H11^N. For them a reading
#: also shows the element when one library neutral takes all of it out of
#: the remainder next to a skeletal piece (`_leaves_whole`). A skeleton's
#: own heteroatoms (Si, S, P, a halogen) are read as printed -- but for an
#: H-free neutral the library lacks (`BARE_NEUTRALS`): an alias formula
#: reads O-rich products through mixed remainders (S + Si + H4 for O4, 0.64
#: mDa) that one library neutral would split as well (tested).
REAGENT_ELEMENTS = ("N", "^N")
#: A skeleton's heteroatom neutrals that hold no H and are no library piece:
#: the remainder rule refuses them as an unnamed piece of their own (no H),
#: so the decomposer prints them merged with the neutral lost next to them.
#: CS leaves a thiophene's or a thiazole's ring at high energy: protonated
#: benzothiazole loses HCN (C6H5S+), then CS (C5H5+), and HCN + CS prints
#: as ONE unnamed C2HNS. The element shows where one of them takes all of it
#: out of the remainder and the rest is one closed-shell library neutral
#: (`_leaves_bare`; tested: the true thiazole is not contradicted on its own
#: S when C5H5+ dominates its products).
BARE_NEUTRALS: dict[str, tuple[str, ...]] = {"S": ("CS",)}
#: A skeleton's elements that leave in its own C-bearing neutrals: an organic
#: ion's O goes as H2O, CO, CO2 and in skeletal pieces (an alkene with the
#: acid's CO2), and the decomposer, printing the fewest pieces, merges them
#: with the reagent's neutral -- the ammonium adduct of an O4 acid that ends
#: in C7H9+ at high energy prints its loss as H2O + C2H7NO3. The element
#: shows where ONE valid skeletal piece (`subformula._remainder_ok`, of more
#: than `attribute.POLYOXY_MAX_C` C) holds all of it, a skeleton's C-O
#: neutrals with at most one reagent neutral (`_leaves_skeletal`): its N or
#: ^N all in ONE of the ion's own polarity's `REAGENT_NEUTRALS` (an NH3, an
#: HNO3), the rest C, H and O only, at most `SKELETAL_O_OVER_C` O beyond its
#: C (tested: the true O4 adduct, whose hydrocarbon ions dominate its high
#: energies, is not contradicted on its own O -- contradicted, its O at 27 %,
#: while every such piece read as the O absent --, and its S + Si + H4 alias,
#: read through the same pieces, still is on S and Si: a skeleton's other
#: heteroatoms are read as printed). A piece holding more N than one reagent
#: neutral (a whole N2O5 group) or an oxygen chain (C2H2O9) is no skeletal
#: loss (tested on such pieces): no skeleton loses its O so, and an alias
#: formula that reads products through them would be shown its O.
SKELETAL_ELEMENTS = ("O",)
#: ... the O a skeletal piece may hold beyond its C, its reagent neutral out:
#: a skeleton's C-O neutrals hold no more (an alkene with CO2 and H2O, C(n+1)
#: H(2n+2)O3; the O4 adduct's NH3 + C2H4O3, C4H8O4).
SKELETAL_O_OVER_C = 1


def shows(element: str, formula: str, loss_name: str, unnamed: int, *,
          charge: int = 0) -> bool:
    """Does a reading show `element`: in the product, or leaving whole in
    named library neutrals -- none of it in the unnamed remainder; or the
    remainder a merge of real neutrals (`_undone`: the reagent neutrals of
    `charge`'s polarity, common ones and one O2 -- what
    `implausible_remainder` credits: a nitrate cluster's H^NO3 + CO2 + H2O
    as one CH3^NO6 shows its ^N); for a reagent's element
    (`REAGENT_ELEMENTS`) also all of it in ONE closed-shell library neutral
    the remainder holds next to a skeletal piece (`_leaves_whole`: an
    ammonium adduct's ^NH3 lost with a skeletal C4H8, printed as one unnamed
    C4H11^N); for a `SKELETAL_ELEMENTS` one, all of it in the remainder,
    one valid skeletal piece (`_leaves_skeletal`: C-O neutrals and at most
    one reagent neutral of `charge`'s polarity); for an element of a
    `BARE_NEUTRALS` piece, all of it in that piece next to one closed-shell
    library neutral (`_leaves_bare`: a thiazole's HCN + CS as C2HNS)?"""
    if SF.parse_ion(formula)[0].get(element, 0) > 0:
        return True
    rem = _remainder(loss_name, unnamed)
    if rem.get(element, 0) == 0:
        return True
    items = tuple(sorted(rem.items()))
    sign = (charge > 0) - (charge < 0)
    if _undone(items, sign):
        return True
    if element in REAGENT_ELEMENTS:
        return _leaves_whole(element, items)
    if element in SKELETAL_ELEMENTS:
        return _leaves_skeletal(items, sign)
    return _leaves_bare(element, items)


#: The closed-shell library neutrals, as counts (`_leaves_whole`).
_LIBRARY = tuple((name, C.parse_formula(name)) for name in SF.CLOSED_SHELL_LOSSES)


@lru_cache(maxsize=1 << 12)
def _leaves_bare(element: str, remainder: tuple[tuple[str, int], ...]) -> bool:
    """Can ONE of `element`'s `BARE_NEUTRALS` take all of it out of the
    unnamed `remainder`, the rest nothing or one closed-shell library
    neutral?"""
    rem = dict(remainder)
    for name in BARE_NEUTRALS.get(element, ()):
        piece = C.parse_formula(name)
        if any(rem.get(e, 0) < n for e, n in piece.items()):
            continue
        rest = {e: rem.get(e, 0) - piece.get(e, 0) for e in {*rem, *piece}}
        rest = {e: n for e, n in rest.items() if n}
        if rest.get(element, 0):
            continue
        if not rest or any(rest == {e: n for e, n in lib.items() if n} for _, lib in _LIBRARY):
            return True
    return False


@lru_cache(maxsize=1 << 14)
def _leaves_whole(element: str, remainder: tuple[tuple[str, int], ...]) -> bool:
    """Can ONE closed-shell library neutral holding `element` take all of it
    out of the unnamed `remainder`, the rest nothing, a library neutral, or
    a skeletal piece: a valid unnamed one (`subformula._remainder_ok`, its
    own parity) of more than `attribute.POLYOXY_MAX_C` C?"""
    rem = dict(remainder)
    for _, piece in _LIBRARY:
        if piece.get(element, 0) <= 0 or any(rem.get(e, 0) < n for e, n in piece.items()):
            continue
        rest = {e: rem.get(e, 0) - piece.get(e, 0) for e in {*rem, *piece}}
        rest = {e: n for e, n in rest.items() if n}
        if not rest:
            return True
        if rest.get(element, 0):
            continue
        vec = SF._vec(rest)
        if tuple(int(n) for n in vec) in SF._LIB_INDEX:
            return True
        if rest.get("C", 0) > AT.POLYOXY_MAX_C and _valid(rest):
            return True
    return False


def _valid(piece: Mapping[str, int]) -> bool:
    """A valid unnamed piece on its own parity (`subformula._remainder_ok`:
    an integer DBE, Senior's rule, an H)."""
    vec = SF._vec(dict(piece))
    t2 = 2 + int(vec.astype(np.int64) @ SF._VM2)
    return bool(SF._remainder_ok(vec[None, :], np.array([bool(t2 % 2)]))[0])


@lru_cache(maxsize=1 << 12)
def _leaves_skeletal(remainder: tuple[tuple[str, int], ...], sign: int = 0) -> bool:
    """Is the unnamed `remainder` one skeletal piece: valid (`_valid`), more
    than `attribute.POLYOXY_MAX_C` C, nothing but C, H, N, ^N and O -- its N
    and ^N all in ONE reagent neutral of polarity `sign` (`REAGENT_NEUTRALS`;
    none for 0), the rest C, H and O with at most `SKELETAL_O_OVER_C` O
    beyond its C?"""
    rem = dict(remainder)
    if not (set(rem) <= _POLYOXY and rem.get("C", 0) > AT.POLYOXY_MAX_C and _valid(rem)):
        return False
    rests = []
    if not rem.get("N", 0) and not rem.get("^N", 0):
        rests.append(rem)
    else:
        for name in REAGENT_NEUTRALS.get(sign, ()):
            piece = C.parse_formula(name)
            if all(rem.get(e, 0) >= n for e, n in piece.items()):
                rests.append({e: rem.get(e, 0) - piece.get(e, 0) for e in {*rem, *piece}})
    return any(not r.get("N", 0) and not r.get("^N", 0)
               and r.get("O", 0) <= r.get("C", 0) + SKELETAL_O_OVER_C for r in rests)


def _carbon_poor(formula: str) -> bool:
    """A product of at most `attribute.POLYOXY_MAX_C` C (a reagent ion or its
    cluster: NO3-, HSO4-, NH4+, Br-, the uronium ion)."""
    return bool(formula) and SF.parse_ion(formula)[0].get("C", 0) <= AT.POLYOXY_MAX_C


def _named(loss_name: str, unnamed: int, radical: bool) -> bool:
    return bool(loss_name) and not unnamed and not radical


# ---------------------------------------------------------------------------
# the window
# ---------------------------------------------------------------------------
@dataclass
class RivalRead:
    """One read of the window with `rival` as its target: `analyze`'s roles
    (energy-resolved spectra, side lobes set aside), readings and fragment
    rows (`analyze.FRAGMENT_COLUMNS`: candidates, call, weights); None for a
    window read without an energy axis. `offset_ppm`: the window's
    recalibration on this rival's precursor. `cross`: this rival read again,
    a whole read, on each other rival's calibration (that rival -> the read):
    what it reads, and is given by the attribution, where the other reads."""
    rival: str
    roles: pd.DataFrame | None = None
    readings: pd.DataFrame | None = None
    fragments: pd.DataFrame | None = None
    offset_ppm: float = math.nan
    co_isolates: tuple = ()
    cross: Mapping[str, "RivalRead"] = field(default_factory=dict)


def _weights(fragments: pd.DataFrame | None) -> tuple[dict, dict]:
    """fragment key -> {ion: weight}, key -> call."""
    if fragments is None or not len(fragments):
        return {}, {}
    w = {k: {a: float(b) for a, _, b in (kv.partition("=") for kv in str(t).split(";") if kv)}
         for k, t in zip(fragments["fragment"], fragments["weights"])}
    return w, dict(zip(fragments["fragment"], fragments["call"]))


#: `product_table`'s columns (`calibration`: the rival on whose precursor's
#: calibration the row's rival is read).
PRODUCT_COLUMNS = ("rival", "calibration", "spectrum", "peak", "intensity", "key", "weight",
                   "credited", "own", "credible", "readers", "formula", "loss", "unnamed",
                   "radical", "ppm", "shown", "uncredited")


def _have(reads: Mapping[str, RivalRead], rivals: Sequence[str]) -> list[str]:
    return [R for R in rivals if reads.get(R) is not None and reads[R].roles is not None]


def _reads_of(rr: RivalRead, rival: str, rivals: Sequence[str]) -> list[tuple[str, RivalRead]]:
    """(calibration, read) of one rival: its own read first, then its reads
    on each other rival's calibration, in the order named."""
    out = [(rival, rr)]
    for A in rivals:
        x = rr.cross.get(A) if A != rival else None
        if x is not None and x.roles is not None:
            out.append((A, x))
    return out


def non_products(reads: Mapping[str, RivalRead], rivals: Sequence[str], *,
                 ceiling_mz: float = math.nan) -> set[tuple[str, int]]:
    """The (spectrum, peak) pairs of the window's energy-resolved spectra
    that are no rival's product: a peak above `ceiling_mz` (the located
    precursor less an H atom, `H_ATOM`, plus the reading tolerance: no singly
    charged product of any rival lies there -- an unnamed ion co-isolated in
    the window, an isotopologue of the precursor), and a fragment every
    rival's reads give wholly to named co-isolates (more than 0 for a
    co-isolate on some calibration, and on none of the rival's own
    calibration and every other rival's (`RivalRead.cross`) a weight of the
    rival's own where it reads the peak credibly -- not through a reading it
    cannot be credited with, `implausible_remainder`: an O-rich rival reads a
    co-isolate's organic products through an oxygen chain, and the
    attribution shares them with it (tested) --: their products, which no
    rival can claim and a decoy must not move). Taken from the window's own
    reads; the decoys keep these peaks in place and leave them out too."""
    have = _have(reads, rivals)
    if not have:
        return set()
    out: set[tuple[str, int]] = set()
    ref = reads[have[0]].roles
    if math.isfinite(ceiling_mz) and "mz" in ref:
        hi = ref["mz"].to_numpy(dtype=float) > ceiling_mz
        out |= {(s, int(p)) for s, p in zip(ref.loc[hi, "spectrum"], ref.loc[hi, "peak"])}
    specs = set(ref["spectrum"])
    theirs = None
    for R in have:
        holds, others = set(), set()
        for _, x in _reads_of(reads[R], R, rivals):
            w, _ = _weights(x.fragments)
            can = _credible_keys(x.readings, R, specs)
            r = x.roles
            for s, p, role, key in zip(r["spectrum"], r["peak"], r["role"], r["key"]):
                if role != "fragment":
                    continue
                ws = w.get(key, {})
                k = (s, int(p))
                if k in can and ws.get(R, 0.0) > 0:
                    holds.add(k)
                if any(v > 0 for a, v in ws.items() if a != R):
                    others.add(k)
        mine = others - holds
        theirs = mine if theirs is None else theirs & mine
    return out | (theirs or set())


def _credible(rd: pd.DataFrame | None, rival: str, specs: set) -> pd.DataFrame | None:
    """A read's readings of `rival` that can credit a product: ranked, no
    precursor reading, in the energy-resolved spectra `specs`."""
    if rd is None:
        return None
    return rd[(rd["parent"] == rival) & (rd["rank"] >= 1) & ~rd["is_precursor"]
              & rd["spectrum"].isin(specs)]


def _credible_keys(rd: pd.DataFrame | None, rival: str, specs: set) -> set[tuple[str, int]]:
    """The (spectrum, peak) pairs one read's readings of `rival` read with a
    credible loss (no `implausible_remainder`, its own polarity's reagent)."""
    x = _credible(rd, rival, specs)
    if x is None:
        return set()
    q = _sign(rival)
    return {(s, int(p)) for s, p, ln, u in zip(x["spectrum"], x["peak"], x["loss_name"],
                                                x["unnamed_pieces"])
            if not implausible_remainder(ln, int(u), charge=q)}


def _readable(rr: RivalRead, rival: str, specs: set,
              rivals: Sequence[str] = ()) -> set[tuple[str, int]]:
    """The (spectrum, peak) pairs `rival` reads with a credible loss on its
    own calibration or any other rival's (`rr.cross`)."""
    out: set[tuple[str, int]] = set()
    for _, x in _reads_of(rr, rival, list(rivals) or list(rr.cross)):
        out |= _credible_keys(x.readings, rival, specs)
    return out


def _read_rows(x: RivalRead, R: str, specs: set, skip: set) -> tuple[list[tuple], float]:
    """One read's product rows of rival `R` -- (spectrum, peak, intensity,
    key, weight, credited, own, credible, formula, loss, unnamed, radical,
    ppm, shown, uncredited) per peak not in `skip` -- and what it credits."""
    rd = _credible(x.readings, R, specs)
    elems = list(SF.parse_ion(R)[0])
    q = _sign(R)
    best, shown, seen = {}, {}, set()
    for s, p, f, ln, u, rad, ppm in zip(rd["spectrum"], rd["peak"], rd["formula"],
                                        rd["loss_name"], rd["unnamed_pieces"],
                                        rd["radical_loss"], rd["ppm"]):
        key = (s, int(p))
        seen.add(key)
        if implausible_remainder(ln, int(u), charge=q):
            continue
        if key not in best:
            best[key] = (f, ln, int(u), bool(rad), float(ppm))
        shown.setdefault(key, set()).update(e for e in elems
                                            if shows(e, f, ln, int(u), charge=q))
    w, call = _weights(x.fragments)
    r = x.roles
    rows, got = [], 0.0
    for s, p, role, key, inten in zip(r["spectrum"], r["peak"], r["role"], r["key"],
                                      r["intensity"]):
        k = (s, int(p))
        if k in skip:
            continue
        inten = float(inten) if math.isfinite(float(inten)) else 0.0
        wt = float(w.get(key, {}).get(R, 0.0)) if role == "fragment" else 0.0
        cred = k in best
        f, ln, u, rad, ppm = best.get(k, ("", "", 0, False, math.nan))
        credited = inten * wt if cred else 0.0
        got += credited
        rows.append((s, int(p), inten, key if role == "fragment" else "", wt, credited,
                     role == "fragment" and call.get(key, "") == R, cred, f, ln, u, rad, ppm,
                     ";".join(sorted(shown.get(k, ()))),
                     inten * wt if (not cred and k in seen) else 0.0))
    return rows, got


def product_table(reads: Mapping[str, RivalRead], rivals: Sequence[str], *,
                  exclude: Collection[tuple[str, int]] = (), own_only: bool = False
                  ) -> pd.DataFrame:
    """One row per (rival, product peak) of the window's energy-resolved
    spectra, each rival read on the CALIBRATION -- its own precursor's or
    another rival's (`RivalRead.cross`) -- on which it is credited with the
    most (its own on a tie; `calibration`): the rivals' ppm gap moves every
    product between their calibrations, and which calibration holds the
    precursor is an MS1 matter, never weighed. Per row: the peak's
    intensity, the rival's fragment key and attribution weight there, what
    it is credited with (intensity x weight where it reads the peak with a
    credible loss: `implausible_remainder`), whether the attribution calls
    the fragment its own, which rivals read the peak credibly (`readers`: on
    any rival's calibration), its best credible reading (formula, loss,
    unnamed pieces, radical, ppm), the elements that reading shows
    (`shows`, any credible reading of the peak) and what the attribution
    gave it through a polyoxy remainder only (`uncredited`). Products: the
    peaks no read of the window takes for a window ion's precursor or a
    harmonic ghost, less `exclude` (`non_products`) -- one set for every
    rival and calibration. `own_only`: every rival read on its own
    calibration (`readers` still on any), as the run's other tables read
    the rival they read. Empty when no read has an energy axis."""
    cols = {c: [] for c in PRODUCT_COLUMNS}
    have = _have(reads, rivals)
    if not have:
        return pd.DataFrame(cols)
    skip = {(s, int(p)) for s, p in exclude}
    for R in have:
        r = reads[R].roles
        m = r["role"].isin(["precursor", "harmonic"]).to_numpy()
        skip |= {(s, int(p)) for s, p in zip(r.loc[m, "spectrum"], r.loc[m, "peak"])}
    specs = set(reads[have[0]].roles["spectrum"])
    readable = {R: _readable(reads[R], R, specs, rivals) for R in have}
    names = ("spectrum", "peak", "intensity", "key", "weight", "credited", "own", "credible",
             "formula", "loss", "unnamed", "radical", "ppm", "shown", "uncredited")
    for R in have:
        pick = None
        for cal, x in _reads_of(reads[R], R, rivals)[:1 if own_only else None]:
            rows, got = _read_rows(x, R, specs, skip)
            if pick is None or got > pick[0] * (1 + 1e-12) + 1e-12:
                pick = (got, cal, rows)
        _, cal, rows = pick
        for row in rows:
            cols["rival"].append(R)
            cols["calibration"].append(cal)
            for c, v in zip(names, row):
                cols[c].append(v)
            cols["readers"].append(";".join(R2 for R2 in have
                                            if (row[0], row[1]) in readable[R2]))
    return pd.DataFrame(cols)[list(PRODUCT_COLUMNS)]


def _distinct(rival: str, rivals: Sequence[str]) -> list[str]:
    """The elements `rival` holds that another rival lacks (in the formula's
    own element order)."""
    mine = SF.parse_ion(rival)[0]
    others = [set(SF.parse_ion(B)[0]) for B in rivals if B != rival]
    return [e for e in mine if any(e not in o for o in others)]


@dataclass
class _Score:
    explained: float = 0.0
    n_products: int = 0
    keys: list = field(default_factory=list)   # (key, share, own, readers, formula, loss)
    named: float = math.nan
    ppm: float = math.nan
    uncredited: float = 0.0
    signature: dict = field(default_factory=dict)
    judged: dict = field(default_factory=dict)     # element -> share of the products judged
    read_on: str = ""                              # the calibration it is read on


def rival_scores(table: pd.DataFrame, rivals: Sequence[str]) -> tuple[dict, float]:
    """Per rival: its explained share of the window's product intensity, its
    products (fragment keys it is credited with: share, own, readers, the
    reading at its strongest peak), the named-loss share and mean |ppm| of
    what it is credited with, what it was given through a polyoxy remainder
    only, and the signature of each element only it (or some) holds -- NaN
    where it is judged on less than `SIGNATURE_EXPECT` of the products
    (`judged`: that share). -> ({rival: _Score}, the product intensity)."""
    if not len(table):
        return {R: _Score() for R in rivals}, 0.0
    one = table.drop_duplicates(["spectrum", "peak"])
    total = float(one["intensity"].sum())
    out = {}
    for R in rivals:
        g = table[table["rival"] == R]
        sc = _Score()
        if not len(g) or total <= 0:
            out[R] = sc
            continue
        sc.read_on = str(g["calibration"].iloc[0]) if "calibration" in g else ""
        cred = g[g["credited"] > 0]
        got = float(cred["credited"].sum())
        sc.explained = got / total
        sc.uncredited = float(g["uncredited"].sum()) / total
        if got > 0:
            sc.named = float(cred.loc[[_named(a, b, c) for a, b, c in zip(
                cred["loss"], cred["unnamed"], cred["radical"])], "credited"].sum()) / got
            ok = np.isfinite(cred["ppm"].to_numpy(dtype=float))
            if ok.any():
                sc.ppm = float(np.average(np.abs(cred["ppm"].to_numpy(dtype=float)[ok]),
                                          weights=cred["credited"].to_numpy(dtype=float)[ok]))
        for key, k in cred.groupby("key", sort=False):
            readers = set()
            for x in k["readers"]:
                readers |= set(filter(None, str(x).split(";")))
            top = k.loc[k["credited"].idxmax()]
            sc.keys.append((key, float(k["credited"].sum()) / total, bool(k["own"].all()),
                            frozenset(readers), str(top["formula"]), str(top["loss"])))
        sc.n_products = len(sc.keys)
        # a carbon-poor product (the reagent ion, its cluster) that does not
        # show the element says nothing of it: the rival's skeleton left
        # whole, as one neutral -- not judged (out of the numerator and the
        # denominator); one that shows it is judged. Judged on too little
        # of the products, the signature is none
        poor = np.array([_carbon_poor(f) for f in cred["formula"]], dtype=bool)
        for e in _distinct(R, rivals):
            m = np.array([e in str(x).split(";") for x in cred["shown"]], dtype=bool)
            judged = m | ~poor
            base = float(cred.loc[judged, "credited"].sum())
            sc.judged[e] = base / total
            sc.signature[e] = (float(cred.loc[m, "credited"].sum()) / base
                               if base > 0 and base >= SIGNATURE_EXPECT * total else math.nan)
        out[R] = sc
    return out, total


def _pairwise(sc: _Score, other: str, floor: float) -> list:
    """The products a rival is credited with as its own that `other` cannot
    read credibly, each holding at least `floor` of the product intensity."""
    return [k for k in sc.keys if k[2] and other not in k[3] and k[1] >= floor]


def _diagnostic(sc: _Score, rival: str, floor: float) -> list:
    """Its own products no other rival reads credibly, each >= `floor`."""
    return [k for k in sc.keys if k[2] and k[3] <= {rival} and k[1] >= floor]


def _contradicted(sc: _Score) -> list[str]:
    """The distinct elements a rival shows in at most `SIGNATURE_MIN` of
    what it reads, where it reads at least `SIGNATURE_EXPECT` (and the
    signature is judged on at least that: `rival_scores`)."""
    if not sc.explained >= SIGNATURE_EXPECT:
        return []
    return [e for e, v in sc.signature.items() if math.isfinite(v) and v <= SIGNATURE_MIN]


def _judged_text(sc: _Score, e: str) -> str:
    """'Si in 0% of the 100% of the products judged' (the judged share
    unknown: 'Si in 0% of what it reads')."""
    j = sc.judged.get(e, math.nan)
    return (f"{e} in {sc.signature[e]:.0%} of the {j:.0%} of the products judged"
            if math.isfinite(j) else f"{e} in {sc.signature[e]:.0%} of what it reads")


def decide(real: Mapping[str, _Score], decoys: Sequence[Mapping[str, _Score]],
           rivals: Sequence[str]) -> tuple[str, dict[str, str], dict[str, str]]:
    """(winner or '', {rival: verdict}, {rival: why}) for one window, from
    the rivals' scores on the window (`real`) and on its decoys. See the
    module docstring."""
    verdict: dict[str, str] = {}
    why: dict[str, str] = {}
    E = {R: real[R].explained for R in rivals}
    for R in rivals:
        els = _contradicted(real[R])
        if els:
            verdict[R] = CONTRADICTED
            why[R] = ("its " + ", ".join(els) + " signature is absent: it shows "
                      + ", ".join(_judged_text(real[R], e) for e in els)
                      + f" (it reads {E[R]:.0%} of them), the rest read as products free of "
                      "it, all of it in an unnamed remainder")
    live = [R for R in rivals if R not in verdict]
    if not live:
        return "", verdict, why

    def chance(R):
        return float(np.mean([d[R].explained for d in decoys])) if decoys else 0.0

    def lead_chance(A, B):
        return (float(np.mean([d[A].explained - d[B].explained for d in decoys]))
                if decoys else 0.0)
    A = max(live, key=lambda R: (E[R], -list(rivals).index(R)))
    fails = {}
    for B in rivals:
        if B == A:
            continue
        # the decoys raise the bar, never lower it: a decoy lead in B's
        # favour is no chance the real lead has
        lead, lc = E[A] - E[B], max(lead_chance(A, B), 0.0)
        pw = _pairwise(real[A], B, DIAGNOSTIC_FLOOR)
        shared = [k for k in real[A].keys if not k[2] and B not in k[3]
                  and k[1] >= DIAGNOSTIC_FLOOR]
        if E[B] >= E[A]:
            same = not _pairwise(real[A], B, 0.0) and not _pairwise(real[B], A, 0.0)
            fails[B] = (f"reads the same products as {A}" if same else
                        f"explains {'more than' if E[B] > E[A] else 'as much as'} {A} "
                        f"({E[B]:.0%})")
        elif lead - lc < SHARE_MARGIN:
            fails[B] = (f"{A} leads it by {lead:.0%} of the products, {lead - lc:.0%} beyond "
                        f"the decoys' {lc:.0%} (needs {SHARE_MARGIN:.0%})")
        elif len(pw) < DIAGNOSTIC_MIN:
            fails[B] = (f"{A} leads it by {lead:.0%}, but on {len(pw)} product(s) of its own "
                        f"(>= {DIAGNOSTIC_FLOOR:.0%} each) that {B} cannot read (needs "
                        f"{DIAGNOSTIC_MIN})"
                        + (f"; {len(shared)} more it shares with a named co-isolate"
                           if shared else ""))
        else:
            fails[B] = ""
    ex = E[A] - chance(A)
    if ex < EXCESS_MIN:
        # short of the excess: chance where the decoys' mean reaches half of
        # what it explains; else the products it reads are too few a share
        # (peaks no rival reads dilute it -- an unnamed ion's products, noise)
        low = (f"{A} explains {E[A]:.0%} of the products, {ex:.0%} beyond its decoys' "
               f"{chance(A):.0%} (needs {EXCESS_MIN:.0%}): "
               + ("no better than chance" if chance(A) >= CHANCE_SAID * E[A] else
                  "too little of the products, though above chance"))
        for R in rivals:
            if R not in verdict:
                verdict[R], why[R] = UNDECIDED, low
        return "", verdict, why
    beaten = [B for B, f in fails.items() if not f]
    for B in beaten:
        if B not in verdict:
            n = len(_pairwise(real[A], B, DIAGNOSTIC_FLOOR))
            verdict[B] = OUTSCORED
            why[B] = (f"{A} leads it by {E[A] - E[B]:.0%} of the products ("
                      f"{E[A] - E[B] - max(lead_chance(A, B), 0.0):.0%} beyond the decoys), "
                      f"with {n} product(s) of its own it cannot read")
    if len(beaten) == len(rivals) - 1:
        verdict[A], why[A] = SUPPORTED, ""
        return A, verdict, why
    open_ = [B for B in rivals if B != A and fails[B]]
    verdict[A] = UNDECIDED
    why[A] = "; ".join(f"against {B}" + (" (contradicted)" if verdict.get(B) == CONTRADICTED
                                          else "") + f": {fails[B]}" for B in open_)
    for B in open_:
        if B not in verdict:
            verdict[B], why[B] = UNDECIDED, fails[B]
    return "", verdict, why


def _name_products(keys: Sequence, n: int = DIAGNOSTIC_SHOWN) -> str:
    """'C3H9Si+ (-C5H19^NO4Si3) 15.8%; ...': the strongest first."""
    top = sorted(keys, key=lambda k: (-k[1], k[0]))[:n]
    return "; ".join(f"{k[0]} (-{k[5]}) {k[1]:.1%}" if k[5] else f"{k[0]} {k[1]:.1%}"
                     for k in top)


def _signature_text(sc: _Score) -> str:
    return "; ".join(f"{e} {v:.0%}" if math.isfinite(v) else f"{e} n/a"
                     for e, v in sc.signature.items())


def score_window(parent_mz: float, group: RivalGroup, reads: Mapping[str, RivalRead],
                 decoy_reads: Sequence[Mapping[str, RivalRead]], *, analysed: str,
                 precursor_ppm: Mapping[str, float], ms1_ppm: Mapping[str, float],
                 notes: Mapping[str, str] | None = None,
                 exclude: Collection[tuple[str, int]] = ()) -> list[dict]:
    """`ARBITRATION_COLUMNS` rows of one window: every rival scored on the
    window and on its decoys -- the same products in both, less `exclude`
    (`non_products`) --, and the verdict (`decide`). `analysed`: the rival
    the run's other tables read the window with ('' : none of them)."""
    rivals = list(group.rivals)
    table = product_table(reads, rivals, exclude=exclude)
    real, total = rival_scores(table, rivals)
    # read on its own calibration, as the run's other tables read it
    own = {R: real[R].explained for R in rivals}
    if any(real[R].read_on not in ("", R) for R in rivals):
        mine = rival_scores(product_table(reads, rivals, exclude=exclude, own_only=True),
                            rivals)[0]
        own = {R: mine[R].explained for R in rivals}
    decs = []
    for dr in decoy_reads:
        t = product_table(dr, rivals, exclude=exclude)
        if len(t):
            decs.append(rival_scores(t, rivals)[0])
    n_spec = int(table["spectrum"].nunique()) if len(table) else 0
    if total <= 0:
        why = ("no energy-resolved spectrum (the attribution reads the energy axis)"
               if not len(table) else "no product peak: nothing to arbitrate on")
        return undecided_rows(parent_mz, group, analysed=analysed, why=why,
                              precursor_ppm=precursor_ppm, ms1_ppm=ms1_ppm, notes=notes)
    winner, verdict, why = decide(real, decs, rivals)
    rows = []
    for i, R in enumerate(rivals):
        sc = real[R]
        others = [B for B in rivals if B != R]
        best = max(others, key=lambda B: real[B].explained)
        lead = sc.explained - real[best].explained
        ch = [d[R].explained for d in decs]
        lc = [d[R].explained - d[best].explained for d in decs]
        diag = _diagnostic(sc, R, DIAGNOSTIC_FLOOR)
        d_decoy = [len(_diagnostic(d[R], R, DIAGNOSTIC_FLOOR)) for d in decs]
        ev = _evidence(R, verdict[R], why.get(R, ""), sc, ch, diag, rivals, real)
        rows.append({
            "parent_mz": float(parent_mz), "group": group.name, "rival": R, "rival_mz": _mz(R),
            "order": i + 1, "analysed": R == analysed,
            "window_verdict": SUPPORTED if winner else UNDECIDED, "winner": winner,
            "verdict": verdict[R], "explained": sc.explained,
            "chance": float(np.mean(ch)) if ch else math.nan,
            "chance_max": float(np.max(ch)) if ch else math.nan,
            "lead": lead, "lead_chance": float(np.mean(lc)) if lc else math.nan,
            "n_products": sc.n_products, "n_diagnostic": len(diag),
            "diagnostic_share": float(sum(k[1] for k in diag)),
            "diagnostic_products": _name_products(diag),
            "diagnostic_decoy": float(np.mean(d_decoy)) if d_decoy else math.nan,
            "named_share": sc.named, "mean_abs_ppm": sc.ppm,
            "uncredited_share": sc.uncredited,
            "precursor_ppm": float(precursor_ppm.get(R, math.nan)),
            "ms1_ppm": float(ms1_ppm.get(R, math.nan)), "read_on": sc.read_on,
            "explained_own": own[R],
            "elements": ";".join(_distinct(R, rivals)), "signature": _signature_text(sc),
            "product_intensity": total, "n_spectra": n_spec, "n_decoys": len(decs),
            "evidence": ev, "note": (notes or {}).get(R, "")})
    return rows


def _evidence(R: str, verdict: str, why: str, sc: _Score, chance: Sequence[float],
              diag: Sequence, rivals: Sequence[str], real: Mapping[str, _Score]) -> str:
    """One sentence: what it explains against chance, its diagnostic
    products, and why the verdict."""
    ch = float(np.mean(chance)) if len(chance) else math.nan
    head = (f"explains {sc.explained:.0%} of the products' intensity"
            + (f" (decoys {ch:.0%})" if math.isfinite(ch) else "")
            + f" over {sc.n_products} product(s)"
            + (f", read on {sc.read_on}'s calibration" if sc.read_on and sc.read_on != R
               else ""))
    if diag:
        head += (f"; {len(diag)} diagnostic (its own, no other rival reads them): "
                 + _name_products(diag))
    if verdict == SUPPORTED:
        others = [B for B in rivals if B != R]
        return (head + "; supported: it leads "
                + ", ".join(f"{B} ({real[B].explained:.0%})" for B in others)
                + " beyond the decoys, on products they cannot read")
    if verdict in (OUTSCORED, CONTRADICTED):
        return head + f"; {verdict}: {why}"
    return head + (f"; undecided: {why}" if why else "; undecided")


def undecided_rows(parent_mz: float, group: RivalGroup, *, analysed: str, why: str,
                   precursor_ppm: Mapping[str, float] | None = None,
                   ms1_ppm: Mapping[str, float] | None = None,
                   notes: Mapping[str, str] | None = None) -> list[dict]:
    """Every rival undecided, said why (a window with nothing to score)."""
    return [{"parent_mz": float(parent_mz), "group": group.name, "rival": R, "rival_mz": _mz(R),
             "order": i + 1, "analysed": R == analysed, "window_verdict": UNDECIDED,
             "winner": "", "verdict": UNDECIDED, "explained": math.nan, "chance": math.nan,
             "chance_max": math.nan, "lead": math.nan, "lead_chance": math.nan,
             "n_products": 0, "n_diagnostic": 0, "diagnostic_share": math.nan,
             "diagnostic_products": "", "diagnostic_decoy": math.nan, "named_share": math.nan,
             "mean_abs_ppm": math.nan, "uncredited_share": math.nan,
             "precursor_ppm": float((precursor_ppm or {}).get(R, math.nan)),
             "ms1_ppm": float((ms1_ppm or {}).get(R, math.nan)), "read_on": "",
             "explained_own": math.nan,
             "elements": ";".join(_distinct(R, group.rivals)), "signature": "",
             "product_intensity": math.nan, "n_spectra": 0, "n_decoys": 0,
             "evidence": f"undecided: {why}", "note": (notes or {}).get(R, "")}
            for i, R in enumerate(group.rivals)]


def frame(rows) -> pd.DataFrame:
    """Rows -> a table in `ARBITRATION_COLUMNS` dtypes."""
    df = pd.DataFrame(rows) if not isinstance(rows, pd.DataFrame) else rows
    return pd.DataFrame({c: pd.Series(df[c].to_numpy() if c in df else [], dtype=t)
                         for c, t in ARBITRATION_COLUMNS.items()})

