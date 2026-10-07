"""The enumeration space of a run (step 1 of the evidence scale) and the ion /
adduct composition helpers the scale's steps share.

`Space` is the run's element space exactly as the degeneracy audit builds it
(the context profile + the opened contaminant families, plus the curated
formulas: the pass-0 registry of the polarity and the active reference lists'
closed-shell formulas), over the REAGENT PROFILE's adducts. It enumerates through
`degeneracy.enumerate_window` with no truncation, answers "is this neutral a
plausible decomposition" (`plausible_neutral`) and widens itself to admit a
committed neutral the filter drops (`relaxed`, the D5 widening): same-class
competitors are then enumerated instead of the pair reading "unique" vacuously.

Pure: no I/O beyond the bundled reference-list catalog.
"""
from __future__ import annotations

import dataclasses

from peaky.assignment import degeneracy as DG
from peaky.assignment import passes
from peaky.assignment import reflists as RL
from peaky.assignment import tiers as T
from peaky.chem import chemistry as C
from peaky.chem import contexts as X
from peaky.chem import profiles as PR
from peaky.chem import reagents as RG

# adducts that read the ION only (no neutral + reagent split): never a decomposition channel
ION_ONLY = frozenset({"[M]-."})
# the probe formula adduct_delta reads an adduct's composition off (any formula with every element an adduct can
# remove, in excess)
_DELTA_PROBE = "C20H40N4O20S2"


def ion_counts_of(neutral, adduct, ion=None) -> dict:
    """The ion's element counts (`evidence.ion_composition`: the stored ion
    formula when it carries a charge sign, else neutral + adduct, a labelled
    reagent atom as '^N'), zero counts dropped; {} when unreadable."""
    from peaky.assignment import evidence as EV
    try:
        c = EV.ion_composition(neutral, adduct, ion)
    except Exception:  # noqa: BLE001 -- an unreadable formula / adduct has no composition
        return {}
    return {k: int(v) for k, v in (c or {}).items() if v}


def same_formula(f1, f2) -> bool:
    """Two formulas with the same element counts ('C2H8O2Si1' == 'C2H8O2Si');
    the strings compared when either cannot be parsed."""
    try:
        a = {k: v for k, v in C.parse_formula(str(f1)).items() if v}
        b = {k: v for k, v in C.parse_formula(str(f2)).items() if v}
        return a == b
    except Exception:  # noqa: BLE001
        return str(f1) == str(f2)


_DELTA: dict = {}


def adduct_delta(adduct) -> dict | None:
    """What ``adduct`` adds to (+) or removes from (-) the neutral, per element
    ('^N' kept apart from N), e.g. ``'[M+^NO3]-'`` -> {^N: 1, O: 3}; None for an
    adduct with no composition (``tiers._ion_counts`` cannot read it)."""
    v = _DELTA.get(adduct)
    if v is None:
        base = C.parse_formula(_DELTA_PROBE)
        ic = T._ion_counts(_DELTA_PROBE, adduct, labelled=True)
        v = None if ic is None else {k: ic.get(k, 0) - base.get(k, 0) for k in set(ic) | set(base)}
        _DELTA[adduct] = v
    return v


def reagent_supply(adducts, el: str) -> int:
    """The most atoms of ``el`` any of ``adducts`` adds to the neutral."""
    m = 0
    for ad in adducts:
        dl = adduct_delta(ad)
        if dl:
            m = max(m, dl.get(el, 0))
    return m


def decompositions(space: "Space", counts: dict, neutral, adduct, adducts) -> list[dict]:
    """The ion ``counts`` read as neutral + adduct over ``adducts`` and the
    pair's own ``adduct`` (ion-only channels and adducts without a composition
    skipped): one dict per adduct, ``neutral`` (None when a count goes
    negative), ``ok`` + ``why`` (``space.plausible_neutral``), ``counts``."""
    out = []
    for ad in dict.fromkeys(list(adducts) + [adduct]):
        if ad in ION_ONLY:
            continue
        dl = adduct_delta(ad)
        if dl is None:
            continue
        neu = {k: counts.get(k, 0) - dl.get(k, 0) for k in set(counts) | set(dl)}
        ok, why = space.plausible_neutral(neu)
        f = C.format_formula({k: v for k, v in neu.items() if v > 0}) if all(v >= 0 for v in neu.values()) else None
        out.append(dict(neutral=f, adduct=ad, ok=ok, why=why, counts=neu))
    return out


class Space(DG.EnumerationSpace):
    """The run's enumeration space: ``reagent`` (a profile name, its adducts
    are the channels), ``context`` (a context name), the run's active reference
    lists (``[(id, version), ...]``) and the opened contaminant ``families``."""

    def __init__(self, reagent: str, context: str, reflists_active, families, *, catalog=None):
        self.prof = PR.resolve(reagent)
        self.ctx = X.get_context(context)
        self.adducts = list(self.prof.adducts)
        self.reagent = RG.reagent_for_adducts(self.adducts)
        self.polarity = "positive" if any(str(a).rstrip().endswith("+") for a in self.adducts) else "negative"
        cat = RL.load_catalog() if catalog is None else catalog
        act = [cat[x[0]] for x in (reflists_active or []) if x[0] in cat]
        curated = frozenset(passes.known_formulas(self.polarity, getattr(self.ctx, "label", None))) \
            | frozenset(RL.prior_formulas(act))
        self.families = tuple(families)
        super().__init__(DG.space_profiles(self.ctx, self.families), curated)
        self._reason: dict = {}
        self.relaxed_note = ""
        self._relaxed: dict = {}

    def canon(self, neutral: str, adduct: str):
        """The canonical ion key; None for any adduct the key cannot read (a
        decoy's wrong adduct outside ``chemistry.ADDUCT_SHIFTS`` still has a
        composition; the enumeration itself only walks ADDUCT_SHIFTS)."""
        k = (neutral, adduct)
        v = self._ion.get(k, 0)
        if v == 0:
            try:
                v = DG._canonical_ion(neutral, adduct)
            except Exception:  # noqa: BLE001
                v = None
            self._ion[k] = v
        return v

    def space_reason(self, neutral: str):
        """None when the space proposes ``neutral``; else why not
        (``degeneracy._space_reason``: the FIRST profile's reason)."""
        r = self._reason.get(neutral, 0)
        if r == 0:
            r = self._reason[neutral] = DG._space_reason(neutral, self.profiles, self.curated)
        return r

    def plausible_neutral(self, counts: dict) -> tuple[bool, str]:
        """A decomposition's neutral is plausible: counts >= 0, not empty, no
        labelled reagent atom, an integer DBE >= 0 under Senior's cap
        (``chemistry.dbe_ok``; a curated formula is not exempt from it), then
        inside the run's space (``space_reason``: heteroatom types, the O cap,
        the context's caps and filter; a curated formula passes)."""
        if any(v < 0 for v in counts.values()):
            return False, "negative count"
        cnt = {k: v for k, v in counts.items() if v}
        if not cnt:
            return False, "empty"
        if "^N" in cnt:
            return False, "carries the labelled reagent atom"
        ok, why = C.dbe_ok(cnt)
        if not ok:
            return False, why
        f = C.format_formula(cnt)
        r = self.space_reason(f)
        if r is not None:
            return False, r
        return True, ""

    def enumerate(self, mz: float, lo_ppm: float, hi_ppm: float, channels) -> dict:
        """``{ion_key: (neutral, adduct, ppm)}`` of every plausible ion on
        ``channels`` within [lo_ppm, hi_ppm] at ``mz`` (the reading nearest the
        window's midpoint represents its ion)."""
        return DG.enumerate_window(mz, lo_ppm, hi_ppm, channels, self)

    def relaxed(self, neutral: str):
        """D5: this space widened just enough to admit the committed neutral's
        CLASS -- every profile's element caps, C/O box and Van Krevelen windows
        raised to include it, its halogen / Si minimum-carbon rule lowered to
        its own C, and a context-filter failure of the same kind as its own
        admitted (``accept_kinds``). None when no widening admits it. Memoised."""
        hit = self._relaxed.get(neutral, 0)
        if hit != 0:
            return hit
        cnt = {k: v for k, v in C.parse_formula(str(neutral)).items() if v}
        out = None
        if cnt:
            nC, nSi = cnt.get("C", 0), cnt.get("Si", 0)
            Ceff = nC + nSi
            Heff = cnt.get("H", 0) + sum(cnt.get(x, 0) for x in ("F", "Cl", "Br", "I"))
            vals = {}
            if Ceff >= 3:
                vals = {"h_to_c": Heff / Ceff, "o_to_c": cnt.get("O", 0) / Ceff, "n_to_c": cnt.get("N", 0) / Ceff,
                        "dbe_to_c": C.dbe(cnt) / Ceff}
            profs = []
            for prof in self.profiles:
                kw = {}
                for el in DG.ELEMENT_CEILING:
                    n = cnt.get(el, 0)
                    if n > (getattr(prof, f"max_{el}", 0) or 0):
                        kw[f"max_{el}"] = n
                if nC > int(getattr(prof, "grid_c_max", 40)):
                    kw["grid_c_max"] = nC
                if cnt.get("O", 0) > int(getattr(prof, "grid_o_max", 30)):
                    kw["grid_o_max"] = cnt.get("O", 0)
                for name, v in vals.items():
                    lo, hi = getattr(prof, name)
                    if not (lo <= v <= hi):
                        kw[name] = (min(lo, v), max(hi, v))
                mcf = dict(getattr(prof, "min_C_for", {}) or {})
                ch = {el: nC for el, m in mcf.items() if cnt.get(el, 0) >= 1 and nC < m}
                if ch:
                    mcf.update(ch)
                    kw["min_C_for"] = mcf
                profs.append(dataclasses.replace(prof, **kw) if kw else prof)
            sp = object.__new__(type(self))
            sp.__dict__.update(self.__dict__)
            sp.profiles = profs
            sp._ok, sp._ion, sp._reason, sp._relaxed = {}, self._ion, {}, {}
            sp._build_spaces()
            kinds = set()
            why0 = None
            for prof in profs:
                keep, w = X.filter_by_profile(str(neutral), prof)
                if keep:
                    why0 = None
                    break
                why0 = why0 or w
                if w:
                    kinds.add(DG.filter_kind(w))
            sp.accept_kinds = frozenset(kinds) if why0 else frozenset()
            r = sp.space_reason(str(neutral))
            if r is None or (r.startswith("context filter: ")
                             and DG.filter_kind(r[len("context filter: "):]) in sp.accept_kinds):
                sp.relaxed_note = "space widened to admit the committed neutral (" + (
                    self.space_reason(str(neutral)) or "") + ")"
                out = sp
        self._relaxed[neutral] = out
        return out
