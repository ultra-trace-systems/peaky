"""Composition accounting for the assignment report: signal-weighting and the
ammonium/amine degeneracy.

The composition page used to report distinct-neutral COUNTS by backbone only. For
a positive urea-CIMS batch that is misleading on two fronts (1, 2), and on any
batch whose reagent ion reads as an analyte on a third (3); all addressed here:

  1. SIGNAL vs COUNT. A compound is counted once regardless of abundance, so a
     swarm of dim species can dominate the count while a few bright ones carry the
     chemistry. `signal_by_backbone` weights each distinct neutral by its summed
     M0 intensity, so the report can show e.g. "CHON is 47% by count but 6% by
     signal".

  2. THE AMMONIUM/AMINE SHADOW. `cleanup.prefer_amine_over_ammonium` re-reads an
     [M+NH4]+ adduct of a CHO neutral X as [M+H]+ of the amine X+NH3 (the two ions
     are mass- and isotope-identical). That conversion inflates the CHON count, and
     frequently the original CHO partner X is ALSO assigned (from its own [M+H]+),
     so the merged ledger carries BOTH X (CHO) and X+NH3 (CHON) as distinct
     neutrals for what may be a single molecule. `amine_shadow_stats` /
     `collapsed_composition` quantify and optionally collapse that degeneracy, so
     the page can report the count "two ways" (as-assigned vs ammonium-as-CHO).

  3. INORGANIC IONS. A reagent ion read as an analyte (HNO3 as NO3- and
     HNO3.NO3- on a nitrate inlet), an inorganic acid, a peroxide, or carbon held
     only as a carbon oxide or a pseudo-halide (CO2, ICN, INCO, HNCO on an iodide
     inlet) is no organic chemistry: `composition_class` puts every such neutral
     in its own `INORGANIC` class, and `assigned_composition` weights the
     Assigned readings only, the inorganic ones apart from CHO / CHON / CHOS.

All pure (formula arithmetic only); no I/O, no plotting. `neutral_signal` is a
{neutral_formula -> summed cps} map the caller builds from the per-file M0 rows;
`reading_signal` the same per (neutral_formula, adduct) reading.
"""
from __future__ import annotations

from collections import Counter

from peaky.chem import chemistry as C


def backbone(formula: str) -> str:
    """CHOS if S present, else CHON if N present, else CHO. Si/F/Cl/Br are
    additions to the backbone, not a separate class (matches analyte_viz)."""
    c = C.parse_formula(str(formula))
    if c.get("S", 0):
        return "CHOS"
    if c.get("N", 0):
        return "CHON"
    return "CHO"


#: the class of an inorganic neutral -- the reagent ion's own readings (HNO3 as
#: NO3- or HNO3.NO3-), inorganic acids, peroxides, and carbon held only as a
#: carbon oxide / sulfide or a pseudo-halide (`is_inorganic_carbon`) -- reported
#: apart from the organic backbones (CHO / CHON / CHOS), never folded into them
INORGANIC = "inorganic"
#: the organic backbone classes, in print order
ORGANIC_CLASSES = ("CHO", "CHON", "CHOS")

_HALOGENS = ("F", "Cl", "Br", "I")
_PSEUDO_HALIDE_ELEMENTS = frozenset({"C", "H", "N", "O", "S", *_HALOGENS})


def _count(c: dict, el: str) -> int:
    """Atoms of `el` in a parsed formula, a labelled isotope (^N, ^C) included."""
    return int(c.get(el, 0)) + int(c.get("^" + el, 0))


def is_inorganic_carbon(formula: str) -> bool:
    """One carbon held the way inorganic chemistry holds it: a carbon oxide or
    sulfide (CO, CO2, CO3, OCS, CS2: no H, nothing but O / S beside the carbon),
    or a pseudo-halide -- a cyanide, cyanate, fulminate or thiocyanate with at
    most one H and at most one halogen (HCN, HNCO, HSCN, ICN, INCO, ClCN,
    NCNO2). Formic acid (CH2O2), fluoroform (CHF3) and a polyhalogenated
    one-carbon species (chloropicrin CCl3NO2) stay organic."""
    c = C.parse_formula(str(formula))
    if not c or _count(c, "C") != 1 or not {k.lstrip("^") for k in c} <= _PSEUDO_HALIDE_ELEMENTS:
        return False
    n_h, n_x = _count(c, "H"), sum(_count(c, x) for x in _HALOGENS)
    if not _count(c, "N"):
        return n_h == 0 and n_x == 0
    return n_h <= 1 and n_x <= 1


def is_inorganic(formula: str) -> bool:
    """An inorganic neutral formula: carbon-free (a non-empty formula: a missing
    one is no class), or one carbon held as inorganic carbon (`is_inorganic_carbon`)."""
    c = C.parse_formula(str(formula))
    return bool(c) and (not _count(c, "C") or is_inorganic_carbon(formula))


def composition_class(formula: str) -> str:
    """`INORGANIC` for an inorganic neutral (`is_inorganic`), else its organic `backbone`."""
    return INORGANIC if is_inorganic(formula) else backbone(formula)


def assigned_readings(merged) -> set:
    """The (neutral_formula, adduct) readings a merged row holds at tier Assigned
    (an ion-only row, `ion_only_of` set, is never one)."""
    if merged is None or not len(merged) or "tier" not in merged.columns:
        return set()
    m = merged[merged["tier"].astype(str) == "Assigned"]
    if "ion_only_of" in m.columns:
        m = m[m["ion_only_of"].isna()]
    return {(str(n), str(a)) for n, a in zip(m["neutral_formula"], m["adduct"])
            if str(n) not in ("", "nan")}


def assigned_composition(merged, reading_signal: dict) -> dict:
    """The composition of the run's ASSIGNED readings, inorganic ones apart.

    `reading_signal` = {(neutral, adduct): summed per-file M0 height}. Only the
    readings a merged row holds at tier Assigned count (`assigned_readings`).
    Returns {signal: {class: cps} over ORGANIC_CLASSES + INORGANIC, total,
    organic_frac: {class: share of the organic signal}, inorganic_frac: share
    of the total, count: {class: distinct Assigned neutrals}, n_readings,
    by_neutral: {organic neutral: (class, cps over its Assigned readings)}}."""
    keep = assigned_readings(merged)
    sig: dict = {}
    neutrals: dict = {}
    by_neutral: dict = {}
    for (n, a) in keep:
        kl = composition_class(n)
        v = float(reading_signal.get((n, a), 0.0) or 0.0)
        sig[kl] = sig.get(kl, 0.0) + v
        neutrals.setdefault(kl, set()).add(n)
        if kl != INORGANIC:
            by_neutral[n] = (kl, by_neutral.get(n, (kl, 0.0))[1] + v)
    total = sum(sig.values())
    organic = sum(v for k, v in sig.items() if k != INORGANIC)
    return {"signal": sig, "total": total,
            "organic_frac": {k: v / organic for k, v in sig.items() if k != INORGANIC and organic > 0},
            "inorganic_frac": (sig.get(INORGANIC, 0.0) / total) if total > 0 else 0.0,
            "count": {k: len(v) for k, v in neutrals.items()},
            "n_readings": len(keep), "by_neutral": by_neutral}


def top_share(ac: dict, klass: str = "CHO", n: int = 5) -> float:
    """The share of the Assigned ORGANIC signal the `n` brightest neutrals of
    `klass` carry (`assigned_composition`'s `by_neutral`); 0 without organic signal."""
    bn = ac.get("by_neutral") or {}
    organic = sum(v for _k, v in bn.values())
    if not organic > 0:
        return 0.0
    top = sorted((v for k, v in bn.values() if k == klass), reverse=True)[:n]
    return float(sum(top)) / organic


def minus_nh3(formula: str) -> str | None:
    """The CHO 'shadow' of an amine: X+NH3 -> X (remove one N and three H). The
    [M+H]+ of the returned neutral is the SAME ion as the [M+NH4]+ of `formula`'s
    de-aminated parent, i.e. the mass-degenerate partner. None if no NH3 to remove."""
    c = dict(C.parse_formula(str(formula)))
    if c.get("N", 0) < 1 or c.get("H", 0) < 3:
        return None
    c["N"] -= 1
    c["H"] -= 3
    if c["N"] == 0:
        del c["N"]
    return C.format_formula(c)


def _neutrals(merged) -> list[str]:
    return [str(f) for f in merged["neutral_formula"].dropna().unique() if str(f) != "nan"]


def count_by_backbone(merged) -> dict:
    """Distinct-neutral count per backbone class (the as-assigned composition)."""
    return dict(Counter(backbone(f) for f in _neutrals(merged)))


def signal_by_backbone(merged, neutral_signal: dict) -> tuple[dict, dict]:
    """Backbone composition weighted by summed M0 signal. Returns
    (fractions, absolute) where fractions sum to ~1 over classes with signal.
    A neutral missing from `neutral_signal` contributes 0 (it had no M0 height)."""
    absolute: dict = {}
    for f in _neutrals(merged):
        kl = backbone(f)
        absolute[kl] = absolute.get(kl, 0.0) + float(neutral_signal.get(f, 0.0) or 0.0)
    tot = sum(absolute.values()) or 1.0
    fractions = {k: v / tot for k, v in absolute.items()}
    return fractions, absolute


def amine_shadow_stats(merged) -> dict:
    """Quantify the ammonium/amine degeneracy in the distinct-neutral count.

    A 'shadowed' amine is an N-bearing neutral whose exact X-NH3 CHO twin is ALSO
    present in the ledger — the [M+H]+(amine) / [M+NH4]+(CHO) pair the re-read
    cannot distinguish, counted twice. Returns counts + a few examples.
    `collapsed_neutrals` is the distinct count after removing each shadowed amine
    (its CHO twin remains)."""
    neu = set(_neutrals(merged))
    amines = [f for f in neu if C.parse_formula(f).get("N", 0) >= 1]
    shadowed = []
    for f in amines:
        twin = minus_nh3(f)
        if twin is not None and twin in neu:
            shadowed.append((f, twin))
    return {
        "n_neutrals": len(neu),
        "n_amine": len(amines),
        "n_shadowed": len(shadowed),
        "collapsed_neutrals": len(neu) - len(shadowed),
        "examples": [f"{a}={t}+NH3" for a, t in sorted(shadowed)[:6]],
    }


def collapsed_composition(merged) -> tuple[dict, dict, int]:
    """Two-way backbone counts: (as_assigned, ammonium_as_cho, n_collapsed).

    Classes are `composition_class` (an inorganic neutral is `INORGANIC`).
    `ammonium_as_cho` re-reads every shadowed amine (one with a present X-NH3 twin)
    back into its CHO twin's class — i.e. the composition if the parsimony NH4->amine
    re-read had NOT been applied to the cases where the bare CHO is independently
    seen. The twin already exists, so collapsing just removes the duplicate amine."""
    neu = set(_neutrals(merged))
    as_assigned = dict(Counter(composition_class(f) for f in neu))
    dropped: Counter = Counter()
    for f in neu:
        if C.parse_formula(f).get("N", 0) >= 1:
            twin = minus_nh3(f)
            if twin is not None and twin in neu:
                dropped[composition_class(f)] += 1
    collapsed = {k: as_assigned.get(k, 0) - dropped.get(k, 0) for k in as_assigned}
    return as_assigned, collapsed, int(sum(dropped.values()))


def top_species_by_signal(merged, neutral_signal: dict, *, n: int = 8,
                          inorganic: bool | None = None) -> list[dict]:
    """Top-n distinct neutrals by summed M0 signal, with class (`composition_class`)
    + signal fraction (of ALL of `neutral_signal`). `inorganic` = False keeps the
    organic neutrals only, True the inorganic ones only (`is_inorganic`), None both. Useful for a
    findings page: the chemistry lives in a handful of bright peaks."""
    tot = sum(float(v or 0.0) for v in neutral_signal.values()) or 1.0
    rows = []
    seen = set()
    for f in _neutrals(merged):
        if inorganic is not None and is_inorganic(f) != inorganic:
            continue
        if f in seen:
            continue
        seen.add(f)
        rows.append({"neutral_formula": f, "signal": float(neutral_signal.get(f, 0.0) or 0.0),
                     "frac": float(neutral_signal.get(f, 0.0) or 0.0) / tot,
                     "klass": composition_class(f)})
    rows.sort(key=lambda r: r["signal"], reverse=True)
    return rows[:n]


def oligomer_flag(merged, *, c_min: int = 18, c_max: int = 40, o_min: int = 7,
                  tiers=("Assigned",)) -> list[str]:
    """Distinct neutrals that look like accretion / oligomer products (high carbon
    AND high oxygen) — the HOM dimers that are often the most event-specific signal.
    `c_max` excludes the absurdly large fits (C>40 in a monoterpene system is almost
    always a high-heteroatom mass coincidence, not a real oligomer). Only neutrals
    a merged row holds at one of `tiers` count (default Assigned; never an ion-only
    row): a Candidate reading of that size is a mass fit the run did not confirm.
    `tiers=None`, or a ledger without a tier column, keeps every neutral. Returned
    sorted by carbon then oxygen; the caller may re-sort by signal."""
    if tiers is not None and merged is not None and "tier" in getattr(merged, "columns", []):
        m = merged[merged["tier"].astype(str).isin(set(tiers))]
        if "ion_only_of" in m.columns:
            m = m[m["ion_only_of"].isna()]
        merged = m
    out = []
    for f in _neutrals(merged):
        c = C.parse_formula(f)
        nc = c.get("C", 0)
        if c_min <= nc <= c_max and c.get("O", 0) >= o_min and not c.get("Si", 0):
            out.append((nc, c.get("O", 0), f))
    out.sort(key=lambda t: (-t[0], -t[1]))
    return [f for _, _, f in out]
