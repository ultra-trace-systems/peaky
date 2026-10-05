"""Honest mass-degeneracy measurement (single-file / sum-spectrum only).

The pipeline commits one winner per peak and its stored ``candidate_density`` is
relative to whichever NARROW element box that peak's pass enumerated (CHO/CHON,
or a single opened contaminant family). That undercounts cross-family
degeneracy: at heavy m/z a fluorinated, a CHNOS and a Si formula can all sit
within the instrument's mass accuracy and the per-pass density never compares
them. This module re-measures, for every committed M0 peak, how many distinct,
chemically-plausible IONS fall inside the *calibrated* mass window -- the honest
"how many things could this peak be" count.

What is counted: every ion THIS run could have committed there.

* its channels -- the adducts the run scored (``adducts``: assign.run's detected
  reagent channels plus the opportunistic ones the server resolved; a row on any
  other adduct adds its own). Until 2026-09 every run was audited with one fixed
  Br-CIMS set, so a uronium (+) channel enumerated only negative adducts (82 % of
  its M0 rows read "unique" at density 0) and a nitrate channel never tried
  [M+NO3]-;
* its element space -- the context's element budget (``contexts.element_budget``
  caps; its Van Krevelen and minimum-carbon rules through ``filter_by_profile``),
  a cap the context sets to zero raised to a contaminant family's ceiling where
  the file opened that family (``opened_families``: the context's declared pass-3
  families, the reagent's organohalogen family, the families pass 3 opened on GKA
  evidence), plus every curated formula (the pass-0 registry, the active reference
  lists). Never a formula no pass of the run could keep: the fixed relaxed box
  this replaced counted F+Si+N-type mixtures (70-80 % of all competitor slots on
  three channels) and, on a uronium channel, F/Cl/Br formulas the context rules out;
* no box -- C up to the context's ``grid_c_max``, O up to ``grid_o_max``, H from
  an integer DBE under Senior's cap and the structural oxygen cap (the grid
  enumerator's own rules), at most MAX_HET_TYPES heteroatom types. The old box
  (C <= 20, O <= 12, H <= 36) left the committed formula itself out of its own
  count on 6-45 % of the rows, where density 0 then read as "unique".

A commit outside that space (off-budget and uncurated -- the plausibility stage
demotes those anyway -- or outside the context filter) makes the count a LOWER
BOUND: the others plus itself. At >= 3 that still decides "degenerate"; below 3
the density is not measured (NaN): never "unique" and never "degenerate" for
the merge vote's private class (docs/EVIDENCE_LEVELS.md section 13), no tier cap.

The enumeration is analytic and per peak (no grid): for every heteroatom
combination the neutral mass is linear in the carbon count and the DBE, so the
formulas inside a sub-mDa window are solved for directly.

Policy (user decision 2026-06-13, revised same day): this module is MEASUREMENT
ONLY -- it never sets a tier itself; it stamps ``degeneracy_density`` +
``degeneracy_note`` (the competing tie set) so a reader can see exactly how
identifiable each mass really is. The VERDICT lives in tiers.py, which reads
these columns: an uncorroborated commit whose honest density is high (or
MASS-SATURATED) is capped at Candidate there. (The initial decision to stamp
without demoting was revised once the contradiction surfaced -- a row reading
``tier=Assigned, "unique formula"`` while also carrying ``degeneracy_density=27,
MASS-SATURATED`` is self-contradictory. Corroborated commits -- committed
isotopologue child / second channel / series anchor -- are still spared, because
that corroboration is exactly the extra-spectral evidence that breaks the tie.)
Separation of concerns is preserved: degeneracy MEASURES, tiers JUDGES, and
degeneracy.apply_degeneracy must run before tiers.apply_tiers.

Same-ion decomposition readings (covalent ``Y(Br)[M-H]-`` vs cluster
``Y'.HBr.Br-``, or ``X [M+NH4]+`` vs ``X+NH3 [M+H]+``) are the SAME ion and are
deduplicated, so they never inflate the count -- exactly as tiers.py treats them.
"""
from __future__ import annotations

import dataclasses
import itertools
import re
from bisect import bisect_left, bisect_right

import numpy as np
import pandas as pd

from peaky.chem import chemistry as C
from peaky.chem import contexts as X
from peaky.assignment import ledger as L
from peaky.assignment import tiers as T

__version__ = "0.2.0"

K_SIGMA = 3.0        # calibrated half-window, in sigma
MAX_ALTS = 6         # competitors listed in the note
# Realistic-competitor gate: real atmospheric/contaminant species carry at most
# a few heteroatom TYPES (CHO, CHON, CHOS, a single-halogen organic, a siloxane
# ...). A competitor must use at most this many distinct heteroatom types.
_HET = ("N", "S", "P", "F", "Cl", "Br", "Si", "I")
MAX_HET_TYPES = 3
SATURATION_DENSITY = 8   # beyond this the mass is not identifiable by mass alone
# The heteroatoms the enumeration iterates (C and H are solved for), and the most
# of each any context cap or contaminant family names -- a context that leaves a
# cap at its 99 default must not open a 99-atom loop.
_ELS = ("N", "O", "S", "P", "Si", "F", "Cl", "Br", "I")
ELEMENT_CEILING = {"N": 8, "S": 2, "P": 1, "Si": 12, "F": 17, "Cl": 4, "Br": 2, "I": 2}
# a halide reagent's own covalent-halogen family, which pass 3 opens on every file
REAGENT_FAMILY = {"Br": "bromo_organic", "Cl": "chloro_organic"}
_MH = C.M["H"]
_CH2 = 12.0 + 2 * _MH      # one more carbon at the same DBE
_EPS = 1e-9


def _het_types(counts: dict) -> int:
    return sum(1 for el in _HET if counts.get(el, 0) > 0)


def _canonical_ion(neutral: str, adduct: str) -> str | None:
    """A canonical ion-composition key, so covalent-vs-cluster aliases collapse
    to one (same physical ion = one candidate)."""
    cnt = T._ion_counts(neutral, adduct)
    if not cnt:
        return None
    return "".join(f"{el}{cnt[el]}" for el in sorted(cnt))


# ---------------------------------------------------------------------------
# the run's element space
# ---------------------------------------------------------------------------
def opened_families(profile, reagent=None, evidence=None, ledger=None) -> tuple[str, ...]:
    """The contaminant families a file could commit from: the context's declared
    pass-3 families, the reagent's organohalogen family, the families pass 3 opened
    on GKA evidence (``evidence``: series_detect's table, carried by the run), and
    any family a committed row names (``contaminant:<family>``)."""
    fams = list(getattr(profile, "pass3_families", ()) or ())
    if reagent in REAGENT_FAMILY:
        fams.append(REAGENT_FAMILY[reagent])
    if evidence is not None and len(evidence):
        from peaky.assignment import series_detect as SD
        fams.extend(SD.families_from_evidence(evidence))
    if ledger is not None and "method" in ledger.columns and "role" in ledger.columns:
        meth = ledger.loc[ledger["role"] == L.ROLE_M0, "method"].dropna().astype(str)
        fams.extend(m.split(":")[1] for m in meth if m.startswith("contaminant:") and m.count(":") >= 1)
    return tuple(dict.fromkeys(f for f in fams if f in X.CONTAMINANT_FAMILIES))


def space_profiles(context, families=()) -> list:
    """The element spaces a run commits from: the context profile, plus one per
    opened family that raises a cap the context sets to ZERO to that family's
    ceiling (fluorine on ambient air). A family never lifts a non-zero context cap
    -- the pass-3 box takes min(family, context) there (directors.build_ranges) and
    the plausibility stage demotes anything over the budget."""
    base = X.get_context(context) if isinstance(context, str) else context
    out = [base]
    for fam in families:
        add = (X.CONTAMINANT_FAMILIES.get(fam) or {}).get("add", {})
        kw = {f"max_{el}": int(hi) for el, (_lo, hi) in add.items()
              if hasattr(base, f"max_{el}") and getattr(base, f"max_{el}") == 0}
        if kw:
            prof = dataclasses.replace(base, **kw)
            if all(prof != p for p in out):
                out.append(prof)
    return out


def _caps(profile) -> dict[str, int]:
    caps = {"O": int(getattr(profile, "grid_o_max", 30))}
    for el, ceiling in ELEMENT_CEILING.items():
        caps[el] = max(0, min(int(getattr(profile, f"max_{el}", 0) or 0), ceiling))
    return caps


_COMBOS: dict = {}


def _combos(caps: dict, require: frozenset = frozenset()) -> dict:
    """Every heteroatom assignment of the space (<= MAX_HET_TYPES types, O free;
    with ``require``, only those carrying one of those elements), with the
    per-assignment mass terms the solver needs. Cached per space."""
    key = (tuple(caps[e] for e in _ELS), tuple(sorted(require)))
    cb = _COMBOS.get(key)
    if cb is not None:
        return cb
    het = [e for e in _ELS if e != "O" and caps[e] > 0]
    rows = []
    for k in range(0, MAX_HET_TYPES + 1):
        for types in itertools.combinations(het, k):
            if require and not (set(types) & require):
                continue
            for counts in itertools.product(*[range(1, caps[e] + 1) for e in types]):
                rows.append([dict(zip(types, counts)).get(e, 0) for e in _ELS])
    het_arr = np.array(rows, dtype=np.int64).reshape(-1, len(_ELS))
    o = np.arange(0, caps["O"] + 1, dtype=np.int64)
    arr = np.repeat(het_arr, len(o), axis=0)
    arr[:, _ELS.index("O")] = np.tile(o, len(het_arr))
    col = {e: arr[:, i] for i, e in enumerate(_ELS)}
    base = arr @ np.array([C.M[e] for e in _ELS])
    halo = col["F"] + col["Cl"] + col["Br"] + col["I"]
    hconst = 2 * (1 + col["Si"]) + col["N"] + col["P"] - halo
    a = base + _MH * hconst                                   # mass at C = 0, DBE = 0
    a_low = a - 2 * _MH * (col["Si"] + col["N"] / 2.0 + 1.0)  # its floor at the Senior cap
    cb = {"arr": arr, "base": base, "A": a, "A_low": a_low, "hconst": hconst, **col}
    if len(_COMBOS) > 16:
        _COMBOS.clear()
    _COMBOS[key] = cb
    return cb


def _solve(cb: dict, cmax: int, m_lo: float, m_hi: float):
    """(combo index, C, H) of every closed-shell formula of the space whose neutral
    mass lies in [m_lo, m_hi]: integer DBE in 0..int(C + Si + N/2 + 1), H >= 0, and
    O <= 2(C + N + S + P) + 4 -- the grid enumerator's rules (chemistry.enumerate_grid)."""
    a = cb["A"]
    # mass(C, d) = A + C*CH2 - 2*d*mH with 0 <= d <= cap(C): the carbons that can reach
    # the window lie between the d = 0 ceiling and the Senior-cap floor (+12 per C)
    c_lo = np.maximum(np.ceil((m_lo - a) / _CH2 - _EPS), 0)
    c_hi = np.minimum(np.floor((m_hi - cb["A_low"]) / 12.0 + _EPS), cmax)
    n = (c_hi - c_lo + 1).astype(np.int64)
    idx = np.flatnonzero(n > 0)
    if not len(idx):
        return np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0, np.int64)
    cnt = n[idx]
    rep = np.repeat(idx, cnt)
    cv = c_lo[rep].astype(np.int64) + (np.arange(int(cnt.sum())) - np.repeat(np.cumsum(cnt) - cnt, cnt))
    am = a[rep] + cv * _CH2
    d_lo = np.ceil((am - m_hi) / (2 * _MH) - _EPS)
    d_hi = np.floor((am - m_lo) / (2 * _MH) + _EPS)
    d = d_lo.astype(np.int64)
    nn, si = cb["N"][rep], cb["Si"][rep]
    cap = np.floor(cv + si + nn / 2.0 + 1.0).astype(np.int64)
    h = cb["hconst"][rep] + 2 * cv - 2 * d
    mass = cb["base"][rep] + 12.0 * cv + h * _MH
    keep = ((d_lo <= d_hi) & (d >= 0) & (d <= cap) & (h >= 0)
            & (cb["O"][rep] <= 2 * (cv + nn + cb["S"][rep] + cb["P"][rep]) + 4)
            & (mass >= m_lo) & (mass <= m_hi))      # the window itself, inclusive, no tolerance
    return rep[keep], cv[keep], h[keep]


def _formula(cb: dict, i: int, c: int, h: int) -> str:
    cnt = {"C": int(c), "H": int(h)}
    for e, v in zip(_ELS, cb["arr"][i]):
        if v:
            cnt[e] = int(v)
    return C.format_formula(cnt)


def _space_reason(neutral: str, profiles: list, curated) -> str | None:
    """None when the enumeration of these spaces proposes ``neutral`` (at some
    mass); else why not, in words."""
    if neutral in curated:
        return None
    cnt = C.parse_formula(str(neutral))
    if not cnt:
        return "unparseable formula"
    extra = sorted(e for e in cnt if e not in ("C", "H", *_ELS))
    if extra:
        return f"element {extra[0]} is outside every enumerated space"
    if _het_types(cnt) > MAX_HET_TYPES:
        return f"{_het_types(cnt)} heteroatom types > {MAX_HET_TYPES}"
    ok, why = C.oxygen_ok(cnt)
    if not ok:
        return why
    d = C.dbe(cnt)
    c, si, nn = cnt.get("C", 0), cnt.get("Si", 0), cnt.get("N", 0)
    if d < 0 or abs(d - round(d)) > 1e-9 or d > int(c + si + nn / 2.0 + 1.0):
        return f"DBE {d:g} is not a closed-shell value"
    first = None
    for prof in profiles:
        caps = _caps(prof)
        why = None
        if c > int(getattr(prof, "grid_c_max", 40)):
            why = f"C{c} > the context's C{int(getattr(prof, 'grid_c_max', 40))}"
        elif cnt.get("O", 0) > caps["O"]:
            why = f"O{cnt.get('O', 0)} > the context's O{caps['O']}"
        else:
            for el in ELEMENT_CEILING:
                if cnt.get(el, 0) > caps[el]:
                    why = f"{el}{cnt[el]} over the {prof.label} element budget ({el}<={caps[el]})"
                    break
        if why is None:
            keep, w = X.filter_by_profile(str(neutral), prof)
            if keep:
                return None
            why = f"context filter: {w}"
        first = first or why
    return first


def _curated_masses(curated) -> tuple[list[float], list[str]]:
    rows = []
    for f in curated or ():
        try:
            rows.append((C.neutral_mass(str(f)), str(f)))
        except Exception:
            continue
    rows.sort()
    return [m for m, _ in rows], [f for _, f in rows]


# ---------------------------------------------------------------------------
# the measurement
# ---------------------------------------------------------------------------
def _ledger_adducts(m0: pd.DataFrame) -> list[str]:
    """The channels a ledger was committed on (a caller without the run's adducts)."""
    ad = m0["adduct"].dropna().astype(str)
    meth = m0["method"].fillna("").astype(str) if "method" in m0.columns else pd.Series("", index=m0.index)
    ad = ad[~meth.loc[ad.index].str.startswith("ion_only:")]
    return list(dict.fromkeys(a for a in ad if a in C.ADDUCT_SHIFTS))


# ---------------------------------------------------------------------------
# the enumeration (public: the evidence levels enumerate competitors with it)
# ---------------------------------------------------------------------------
def filter_kind(reason) -> str:
    """A context-filter failure reason with its numbers blanked: the RULE that
    failed, not the value (``'O=7 implausible for C=2'`` -> ``'O=# implausible
    for C=#'``). A widened space admits failures of the kinds it names."""
    return re.sub(r"[-+]?\d+(?:\.\d+)?", "#", str(reason))


class EnumerationSpace:
    """One run's element space, ready to enumerate: the context profile (+ one
    profile per opened family, ``space_profiles``) with each profile's
    heteroatom combinations, and the curated formulas by mass. ``accept_kinds``
    admits context-filter failures of those kinds (``filter_kind``; empty = the
    filter decides). Memoises its filter verdicts and canonical ion keys."""

    def __init__(self, profiles: list, curated=frozenset(), *, accept_kinds=frozenset()):
        self.profiles = list(profiles)
        self.curated = frozenset(str(f) for f in (curated or ()))
        self.accept_kinds = frozenset(accept_kinds)
        self._build_spaces()
        self.cur_m, self.cur_f = _curated_masses(self.curated)
        self._ok: dict = {}
        self._ion: dict = {}

    def _build_spaces(self) -> None:
        base_caps = _caps(self.profiles[0])
        self.spaces = []
        for i, prof in enumerate(self.profiles):
            caps = _caps(prof)
            require = frozenset(e for e in ELEMENT_CEILING if caps[e] > base_caps[e]) if i else frozenset()
            self.spaces.append((prof, _combos(caps, require), int(getattr(prof, "grid_c_max", 40))))

    def admits(self, si: int, neutral: str) -> bool:
        """Profile ``si``'s context filter keeps ``neutral`` (or fails it on an
        accepted kind)."""
        ok = self._ok.get((si, neutral))
        if ok is None:
            keep, why = X.filter_by_profile(neutral, self.spaces[si][0])
            ok = bool(keep) or bool(self.accept_kinds and why and filter_kind(why) in self.accept_kinds)
            self._ok[(si, neutral)] = ok
        return ok

    def canon(self, neutral: str, adduct: str) -> str | None:
        """The canonical ion key (``_canonical_ion``), memoised."""
        k = (neutral, adduct)
        v = self._ion.get(k, 0)
        if v == 0:
            v = self._ion[k] = _canonical_ion(neutral, adduct)
        return v


def enumeration_space(context, families=(), curated=frozenset(), *, accept_kinds=frozenset()) -> EnumerationSpace:
    """The space ``measure_degeneracy`` counts in: ``context`` (a ContextProfile
    or its name) with the opened ``families``, plus the ``curated`` formulas."""
    return EnumerationSpace(space_profiles(context, families), curated, accept_kinds=accept_kinds)


def enumerate_window(mz: float, lo_ppm: float, hi_ppm: float, channels, space: EnumerationSpace, *,
                     centre: float | None = None) -> dict[str, tuple[str, str, float]]:
    """``{ion_key: (neutral, adduct, ppm)}`` of EVERY plausible ion of ``space``
    on ``channels`` whose ppm error at ``mz`` lies in [lo_ppm, hi_ppm]
    (inclusive; no truncation). Within one ion key the reading closest to
    ``centre`` (default: the window's midpoint) wins. A channel without an
    adduct shift is skipped. A non-finite window is refused (ValueError): it
    would admit every ion of the space."""
    if not (np.isfinite(lo_ppm) and np.isfinite(hi_ppm)):
        raise ValueError(f"enumerate_window: a non-finite ppm window [{lo_ppm}, {hi_ppm}] at m/z {mz} "
                         "(an uncalibrated source has no window to enumerate in)")
    ion_lo = mz / (1 + hi_ppm * 1e-6)
    ion_hi = mz / (1 + lo_ppm * 1e-6)
    mid = 0.5 * (lo_ppm + hi_ppm) if centre is None else centre
    found: dict[str, tuple[str, str, float]] = {}

    def _take(neutral: str, adduct: str, theo_ion: float):
        ion = space.canon(neutral, adduct)
        if ion is None:
            return
        ppm = (mz - theo_ion) / theo_ion * 1e6
        prev = found.get(ion)
        if prev is None or abs(ppm - mid) < abs(prev[2] - mid):
            found[ion] = (neutral, adduct, ppm)

    for adduct in channels:
        if adduct not in C.ADDUCT_SHIFTS:
            continue
        shift = C.ADDUCT_SHIFTS[adduct]
        m_lo, m_hi = ion_lo - shift, ion_hi - shift
        if m_hi < 1:
            continue
        for si, (_prof, cb, cmax) in enumerate(space.spaces):
            rep, cv, hv = _solve(cb, cmax, m_lo, m_hi)
            for i, c, h in zip(rep, cv, hv):
                neutral = _formula(cb, i, c, h)
                if space.admits(si, neutral):
                    _take(neutral, adduct, float(cb["base"][i]) + 12.0 * int(c) + int(h) * _MH + shift)
        for j in range(bisect_left(space.cur_m, m_lo), bisect_right(space.cur_m, m_hi)):
            _take(space.cur_f[j], adduct, space.cur_m[j] + shift)
    return found


def measure_degeneracy(ledger: pd.DataFrame, *, cal: tuple[float, float] | None,
                       context: str = "ambient-air", adducts=None, families=(),
                       curated=frozenset(), k_sigma: float = K_SIGMA,
                       max_alts: int = MAX_ALTS,
                       log=lambda *_: None) -> dict[str, dict]:
    """Return {peak_id: {density, note, alts, measured, lower_bound}} for every
    committed M0 peak. Pure; does not mutate the ledger. ``cal`` is (mu, sigma)
    ppm from tiers._calibrate (None -> uncalibrated, returns {}). ``adducts`` are
    the run's channels (None: the ledger's own committed channels); ``families``
    the contaminant families the file opened (``opened_families``); ``curated``
    the formulas a curated list names. ``density`` is NaN where not measured."""
    if cal is None:
        log("[degeneracy] uncalibrated; skipped")
        return {}
    mu, sigma = cal[0], cal[1]
    lo_ppm, hi_ppm = mu - k_sigma * sigma, mu + k_sigma * sigma
    m0 = ledger[ledger["role"] == L.ROLE_M0]
    channels = [a for a in (list(adducts) if adducts else _ledger_adducts(m0)) if a in C.ADDUCT_SHIFTS]
    channels = list(dict.fromkeys(channels))
    space = enumeration_space(context, families, curated)
    profiles, cur = space.profiles, space.curated
    base_caps = _caps(profiles[0])
    raised = sorted({f"{e}<={_caps(p)[e]}" for p in profiles[1:] for e in ELEMENT_CEILING
                     if _caps(p)[e] > base_caps[e]})
    log(f"[degeneracy] channels {channels}; space: the {profiles[0].label} budget"
        + (f" + {', '.join(raised)} (opened families)" if raised else "")
        + f" + {len(cur)} curated; window [{lo_ppm:+.2f},{hi_ppm:+.2f}] ppm")

    out: dict[str, dict] = {}
    for _, r in m0.iterrows():
        mz = r.get("mz")
        if mz is None or pd.isna(mz):
            continue
        mz = float(mz)
        nf, ad = str(r.get("neutral_formula") or ""), str(r.get("adduct") or "")
        chans = channels + ([ad] if ad in C.ADDUCT_SHIFTS and ad not in channels else [])
        found = enumerate_window(mz, lo_ppm, hi_ppm, chans, space, centre=mu)

        assigned_ion = _canonical_ion(nf, ad) if ad in C.ADDUCT_SHIFTS else None
        comp = sorted((v for k, v in found.items() if k != assigned_ion), key=lambda t: abs(t[2] - mu))
        alts = [f"{n} {a} ({p:+.2f} ppm)" for n, a, p in comp[:max_alts]]
        density: float = float(len(found))
        lower = False
        outside = None
        if assigned_ion is None or assigned_ion not in found:
            outside = ("its adduct has no ion composition" if assigned_ion is None
                       else _space_reason(nf, profiles, cur))
        if outside is not None:
            # the commit is not in the space the count enumerates: the others plus
            # itself is a lower bound -- enough to decide "degenerate", never "unique"
            density = float(len(comp) + 1)
            lower = True
        more = "" if len(comp) <= max_alts else f" (+{len(comp) - max_alts} more)"
        tail = (f" — the committed formula lies outside this run's enumerated space "
                f"({outside}), so the count is a lower bound") if lower else ""
        if lower and density < DEGEN_LOWER_DECIDES:
            note = (f"not measured: the committed formula lies outside this run's "
                    f"enumerated space ({outside}); {len(comp)} other plausible "
                    f"ion(s) within ±{k_sigma:.0f}σ calibrated window"
                    + (f": {'; '.join(alts)}{more}" if alts else ""))
            out[str(r["peak_id"])] = {"density": float("nan"), "note": note, "alts": alts,
                                      "measured": False, "lower_bound": True}
            continue
        at = "at least " if lower else ""
        if density <= 1:
            note = f"unique within ±{k_sigma:.0f}σ calibrated mass window"
        elif density > SATURATION_DENSITY:
            note = (f"MASS-SATURATED: {at}{int(density)} plausible formulas (≤{MAX_HET_TYPES} "
                    f"heteroatom types) within ±{k_sigma:.0f}σ calibrated window — "
                    "not identifiable from accurate mass alone; needs isotope "
                    f"envelope / time-series corroboration{tail}")
        else:
            note = (f"MASS-DEGENERATE: {at}{int(density)} plausible ions within ±"
                    f"{k_sigma:.0f}σ calibrated window — competitors: {'; '.join(alts)}{more}{tail}")
        out[str(r["peak_id"])] = {"density": density, "note": note, "alts": alts,
                                  "measured": True, "lower_bound": lower}
    return out


# a lower bound decides "degenerate" at the tier engine's threshold (> 2 ions)
DEGEN_LOWER_DECIDES = T.DEGEN_DEMOTE_DENSITY + 1


def apply_degeneracy(ledger: pd.DataFrame, *, cal=None, context="ambient-air",
                     adducts=None, families=(), curated=frozenset(),
                     log=lambda *_: None, **kw) -> pd.DataFrame:
    """Stamp degeneracy_density / degeneracy_note onto the M0 rows (in place)."""
    for col in ("degeneracy_density", "degeneracy_note"):
        if col not in ledger.columns:
            ledger[col] = pd.Series(pd.NA, index=ledger.index, dtype="object")
        elif ledger[col].dtype != object:
            ledger[col] = ledger[col].astype("object")
    if cal is None:
        cal = T._calibrate(
            ledger[ledger["role"] == L.ROLE_M0],
            ledger.loc[ledger["role"] == L.ROLE_ISO, "parent_peak_id"].value_counts())
    res = measure_degeneracy(ledger, cal=cal, context=context, adducts=adducts,
                             families=families, curated=curated, log=log, **kw)
    if not res:
        return ledger
    for i in ledger.index[ledger["role"] == L.ROLE_M0]:
        pid = str(ledger.at[i, "peak_id"])
        if pid in res:
            d = res[pid]["density"]
            ledger.at[i, "degeneracy_density"] = d if d == d else pd.NA
            ledger.at[i, "degeneracy_note"] = res[pid]["note"]
    return ledger
