"""MS2 fragment assignment, the core: which sub-formulas of a precursor ION fit
a fragment m/z, and what the precursor must have lost to make each of them.

Pure arithmetic over `peaky.chem.chemistry`, like the layer under it: no
Mascope, no I/O, no plotting. A fragment of a singly-charged precursor is an
element-wise sub-formula of the precursor ION (the charge stays on the
fragment), and the difference is the neutral loss. What this module adds over
a plain sub-formula search is the test on that loss:

  * DBE is affine in the element counts, so DBE(loss) = DBE(parent) -
    DBE(fragment) + 1. A loss between two ions of the same electron parity has
    an integer DBE (closed-shell); between ions of opposite parity a
    half-integer one (a radical).
  * A closed-shell loss must split into <= k_max closed-shell neutrals, not be
    ONE molecule. MS2 fragments are often second-generation: on the 2026-09-10
    exploratory 15N-ammonium file the C12H22^NO2+ adduct lost ^NH3 and then
    two waters (159.117 = C12H15+, loss H7^NO2, DBE -2 -- no single molecule
    has it). k molecules each of DBE >= 0 sum to DBE(loss) + k - 1, so
    k >= 1 - DBE(loss) is a necessary bound and rejects the absurd ones (H8
    needs four H2) before any search.
  * The pieces come from a small ordered library of common neutrals plus at
    most ONE unnamed remainder, and the remainder must be a plausible molecule
    on its own: integer DBE >= 0, Senior's second rule, and at least one H or
    halogen. The last clause is what keeps a bare-carbon "loss" out: on the
    same file 88.0215 (C3H6NS+) fits C7H6NS+ minus C4 at -0.5 ppm, but the
    0.4 Da isolation window also held C4H10NS2+ (13 % of its signal), and
    C4H10NS2+ minus CH4S is the reading.

Loss names are formulas joined by ' + ' -- never trivial names, never a
structure: 'C3H6O' is a composition, whatever the reader thinks left.

Ranking is per (peak, parent) and never across parents: deciding which window
ion a fragment came from needs intensities across collision energies, which
this module does not see. `assign_fragments` reports, for every peak, every
parent that can explain it (`n_parents` > 1 = composition-ambiguous).
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, Mapping, NamedTuple, Sequence

import numpy as np
import pandas as pd

from peaky.chem import chemistry as C

__all__ = [
    "CLOSED_SHELL_LOSSES", "RADICAL_LOSSES", "K_MAX_LIMIT",
    "parse_ion", "format_ion", "ion_parity", "fragment_ok",
    "LossDecomposition", "decompose_loss",
    "FragmentCandidate", "fragment_candidates",
    "AnchorResidual", "MassOffset", "mass_offset", "apply_offset",
    "assign_fragments", "best_readings", "OUTPUT_COLUMNS",
]

# ---------------------------------------------------------------------------
# Element axis, valences
# ---------------------------------------------------------------------------
# One fixed axis over every element chemistry.M knows ('^N' = 15N is its own
# element here, for mass AND counts: a labelled-ammonium adduct loses ^NH3,
# 18.0236, which is not NH3, 17.0265).
_EL: tuple[str, ...] = tuple(C.M)
_IDX = {e: i for i, e in enumerate(_EL)}
# Valences of the DBE / Senior accounting, the same as chemistry.dbe(): C/Si 4,
# N/^N/P 3, O/S 2 (S is never hypervalent here -- SO2 and SO3 are LIBRARY
# pieces, so they need no valence rule), H and the halogens 1.
_VALENCE_OF = {"C": 4, "Si": 4, "N": 3, "^N": 3, "P": 3, "O": 2, "S": 2,
               "H": 1, "F": 1, "Cl": 1, "Br": 1, "I": 1}
_VAL = np.array([_VALENCE_OF[e] for e in _EL], dtype=np.int64)
# 2*DBE = 2 + sum((v - 2) * n): integer arithmetic for the parity tests, and
# exactly chemistry.dbe() (C +1, N/P +1/2, O/S 0, H/halogen -1/2 per atom).
_VM2 = _VAL - 2
_H_OR_HALOGEN = np.array([e in ("H", "F", "Cl", "Br", "I") for e in _EL])
# 2 * chemistry.seniors_cap = 2 + sum(_CAP2 * n), read off seniors_cap itself so
# the fragment gate has one source (C, Si 2; N, ^N 1; nothing else -- P adds to
# the DBE but not to the cap, as in chemistry). seniors_cap is affine; the
# tests pin that on random formulas.
_CAP2 = np.array([round(2 * (C.seniors_cap({e: 1}) - C.seniors_cap({}))) for e in _EL],
                 dtype=np.int64)
assert C.seniors_cap({}) == 1.0

# ---------------------------------------------------------------------------
# Loss libraries (Hill formulas, in PRIORITY order)
# ---------------------------------------------------------------------------
# The library only NAMES pieces and orders ties: a loss that is not a library
# formula is still accepted as one unnamed piece when it passes the remainder
# rules. Priority decides between decompositions of equal size -- H4O2 is
# 'H2O + H2O', not 'H2 + H2O2' -- and the order the pieces are written in.
# The first 37 are the common small neutrals of MS2 interpretation; the next
# three were added for neutrals peaky's own channels lose often enough that a
# reading should not count them as unnamed: ethanol (ethyl esters / ethers),
# H2SO4 (organosulfates) and C2H6OSi (dimethylsilanone, the ring-contraction
# loss of every cyclic methylsiloxane). The last three carry no H or halogen,
# so the remainder rule would refuse them as unnamed pieces, and they are real
# losses: CS2 and COS leave dithiocarbamates and thiocarbamates (the
# 2026-09-10 file's 136.025 co-isolate is a dithiocarbamate ester), N2O leaves
# nitroso / N-nitro species.
CLOSED_SHELL_LOSSES: tuple[str, ...] = (
    "H2O", "H3N", "H3^N", "CO", "CO2", "H2", "CH4", "C2H4", "CH2O", "CH4O",
    "CH2O2", "C2H2O", "C2H4O2", "CHN", "CHNO", "CH4N2O", "H2O2", "HNO2", "H^NO2",
    "HNO3", "H^NO3", "H2S", "CH4S", "O2S", "O3S", "HF", "HCl", "HBr", "HI",
    "N2", "O2", "C2H2", "C3H6", "C4H8", "C3H6O", "C2H4O", "C5H8",
    "C2H6O", "H2O4S", "C2H6OSi", "CS2", "COS", "N2O",
)
# Radical pieces: a radical loss carries exactly ONE of these (or one unnamed
# radical remainder) next to closed-shell pieces.
RADICAL_LOSSES: tuple[str, ...] = (
    "H", "CH3", "HO", "HO2", "NO", "^NO", "NO2", "^NO2", "NO3", "^NO3",
    "CHO", "CH3O", "C2H5", "Cl", "Br", "I",
)

# k_max above 6 is refused. The search (`_decompose`) only ever holds the piece
# multisets that fit inside the loss, prunes any whose rest needs more pieces
# than are left, and stops at the first size that reads, so depth costs nothing
# for a real loss. The cap bounds the pathological one: with no pruning a loss
# could hold C(44, 5) ~ 1.1M five-piece multisets (3M with a radical); random
# large halogenated losses at k_max 6 take a few ms, about 0.1 s and < 20 MB at
# worst (once -- results are cached per loss); without the pruning the deep
# ones take 10-50x longer. 6 still reads the longest chain an
# ammonium adduct of a hexose walks: ^NH3 + 5 H2O = H13^NO5, DBE -5, k >= 6.
K_MAX_LIMIT = 6
# Sub-formula grids above this many rows are refused rather than built (about
# 30 MB of int16 counts before the validity filter): a C60H100O30N3S2 parent
# is 2.3M rows; anything beyond 5M is not a CIMS precursor.
_GRID_MAX_ROWS = 5_000_000


def _vec(counts: Mapping[str, int]) -> np.ndarray:
    v = np.zeros(len(_EL), dtype=np.int64)
    for el, n in counts.items():
        if n:
            if el not in _IDX:
                raise ValueError(f"unknown element {el!r}")
            v[_IDX[el]] = n
    return v


def _formula_of(vec: Sequence[int]) -> str:
    return C.format_formula({e: int(n) for e, n in zip(_EL, vec) if n})


_N_CLOSED = len(CLOSED_SHELL_LOSSES)
_LIB_NAMES: tuple[str, ...] = CLOSED_SHELL_LOSSES + RADICAL_LOSSES
_LIB_VEC = np.vstack([_vec(C.parse_formula(f)) for f in _LIB_NAMES]).astype(np.int16)


# ---------------------------------------------------------------------------
# Ion formulas
# ---------------------------------------------------------------------------
# body, ONE sign, optional radical mark ('.', the bullet U+2022, the middle dot U+00B7)
_ION_RE = re.compile("^((?:\\^?[A-Z][a-z]?\\d*)+)([+-])([.•·]?)$")
_BODY_RE = re.compile(r"(?:\^?[A-Z][a-z]?\d*)+")
_EL_TOKEN = re.compile(r"(\^?[A-Z][a-z]?)(\d*)")


def _parse_body(body: str, what: str) -> dict[str, int]:
    """chemistry.parse_formula, but strict: the whole string must be element
    tokens, and an unknown symbol raises. parse_formula skips what it cannot
    read ('C2H4o2' -> C2H4, 'H2O-H2' -> H4O), and a typo must not silently
    shrink a precursor or a loss."""
    if not _BODY_RE.fullmatch(body):
        raise ValueError(f"not a formula: {what} (expected element tokens such as "
                         f"'C2H4O2' or 'H3^N')")
    for el, _ in _EL_TOKEN.findall(body):
        if el not in C.M:
            raise ValueError(f"unknown element {el!r} in {what}")
    return {el: n for el, n in C.parse_formula(body).items() if n}


def parse_ion(ion: str) -> tuple[dict[str, int], int]:
    """'C10H17O+' -> ({'C': 10, 'H': 17, 'O': 1}, +1).

    Exactly one sign is required -- an ion formula without one is a neutral
    passed by mistake, and '2+' / '++' would be a doubly charged ion, which
    nothing here models. A trailing radical mark ('C7H8+.' or 'C7H8+' with a
    bullet) is accepted and checked: on an even-electron formula it is a typo
    and raises. Unknown element symbols raise (chemistry.parse_formula would
    silently drop them)."""
    if not isinstance(ion, str):
        raise ValueError(f"ion formula must be a string, got {ion!r}")
    m = _ION_RE.match(ion.strip())
    if not m:
        raise ValueError(
            f"not a singly-charged ion formula: {ion!r} (expected e.g. 'C10H17O+', "
            f"'C10H17O5-', 'C4H10^NO4+', 'C7H8+.')")
    body, sign, mark = m.groups()
    counts = _parse_body(body, repr(ion))
    if not counts:
        raise ValueError(f"empty ion formula {ion!r}")
    charge = 1 if sign == "+" else -1
    if mark and ion_parity(counts) == "EE":
        raise ValueError(f"{ion!r} carries a radical mark but is even-electron "
                         f"(DBE {C.dbe(counts):g})")
    return counts, charge


def format_ion(counts: Mapping[str, int], charge: int) -> str:
    """({'C': 6, 'H': 9}, +1) -> 'C6H9+'. Hill order as chemistry.format_formula;
    no radical mark (the parity is a column of its own)."""
    if charge not in (1, -1) or isinstance(charge, bool):
        raise ValueError(f"charge must be +1 or -1, got {charge!r}")
    body = C.format_formula(dict(counts))
    if not body:
        raise ValueError("empty ion formula")
    return body + ("+" if charge > 0 else "-")


def _as_counts(ion: str | Mapping[str, int]) -> dict[str, int]:
    if not isinstance(ion, Mapping):
        return parse_ion(ion)[0]
    if any(n < 0 for n in ion.values()):
        raise ValueError(f"negative count in {dict(ion)!r}")
    return dict(ion)


def ion_parity(ion: str | Mapping[str, int]) -> str:
    """'EE' (even-electron) or 'OE' (odd-electron) for a singly-charged ION.

    An ion is even-electron iff its formula's DBE is HALF-integer: C10H17O+ 2.5
    EE, C10H17O5- 2.5 EE, C7H8+. 4.0 OE. Note chemistry.odd_electron() is the
    NEUTRAL test (half-integer = radical); on an ion formula the same
    half-integer DBE means the opposite, so it is not reused here."""
    t2 = 2 + int(_vec(_as_counts(ion)) @ _VM2)
    return "EE" if t2 % 2 else "OE"


def _gate(sub: np.ndarray, col: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    """THE fragment gate, vectorised -- `fragment_ok` and every parent's
    sub-formula grid both call it. `sub` holds ion counts, one row per formula,
    whose columns sit at the `_EL` positions `col`. -> (ok, 2*DBE) per row.
    Column by column, so a 5M-row grid never needs an (n, E) int64 copy."""
    n = len(sub)
    t2 = np.full(n, 2, dtype=np.int64)
    cap2 = np.full(n, 2, dtype=np.int64)
    non_h = np.zeros(n, dtype=np.int64)
    for c, g in enumerate(col):
        x = sub[:, c].astype(np.int64)
        t2 += _VM2[g] * x
        cap2 += _CAP2[g] * x
        if g != _IDX["H"]:
            non_h += x
    ok = (non_h > 0) & (t2 >= np.where(t2 % 2 == 1, -1, 0)) & (t2 <= cap2 + 1)
    return ok, t2


def fragment_ok(ion: str | Mapping[str, int]) -> bool:
    """The structural gate on a FRAGMENT ion formula: at least one non-H atom,
    DBE >= -0.5 when even-electron (H3O+, NH4+ and every protonated saturated
    molecule sit at -0.5) or >= 0 when odd-electron, and DBE <= Senior cap +
    0.5 (the ion's half step above `chemistry.seniors_cap`).

    Fragments only: a PARENT is an input and is never gated. An ammonium
    cluster of a saturated molecule, C6H18^NO2+ (ion DBE -1.5), fails this
    test and is still a real window ion and its own precursor reading -- do
    not pre-screen window ions with it."""
    v = _vec(_as_counts(ion))
    return bool(_gate(v[None, :], range(len(_EL)))[0][0])


# ---------------------------------------------------------------------------
# Loss decomposition
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LossDecomposition:
    """How a neutral loss splits into molecules: `pieces` in library priority
    order with the unnamed remainder (if any) last; `radical` when the loss has
    a half-integer DBE (exactly one piece is then a radical)."""
    formula: str
    pieces: tuple[str, ...]
    unnamed: int
    radical: bool

    @property
    def k(self) -> int:
        return len(self.pieces)

    @property
    def name(self) -> str:
        return " + ".join(self.pieces)


def _check_k_max(k_max: int) -> int:
    if isinstance(k_max, bool) or not isinstance(k_max, (int, np.integer)) \
            or not 1 <= k_max <= K_MAX_LIMIT:
        raise ValueError(f"k_max must be an integer in 1..{K_MAX_LIMIT}, got {k_max!r}")
    return int(k_max)


def _remainder_ok(R: np.ndarray, radical: np.ndarray) -> np.ndarray:
    """Is each row of R (counts, (m, E)) acceptable as the ONE unnamed piece?

    Closed-shell: integer DBE >= 0, Senior's second rule (sum of valences >=
    2 x the largest valence present), and at least one H or halogen. Radical
    (`radical` True for that row): half-integer DBE >= 1/2, sum of valences >=
    2 x largest - 1 (CH3: 7 >= 7, HO: 3 >= 3), and H or halogen. DBE >= 0 is
    Senior's third rule for these valences; the second rule rejects pieces
    like NH (4 < 6); the H/halogen clause rejects the bare-skeleton pieces the
    other two let through: C4 (DBE 5) is how C7H6NS+ would "lose" its way to
    the co-isolate's C3H6NS+ fragment (see the module docstring), and a lone C
    or O atom is no closed-shell molecule. CO, CO2, N2, O2, SO2, CS2, COS and
    N2O are named library pieces and never need this test."""
    R = R.astype(np.int64)
    r2 = 2 + R @ _VM2
    vsum = R @ _VAL
    vmax = np.where(R > 0, _VAL, 0).max(axis=1)
    hx = (R[:, _H_OR_HALOGEN] > 0).any(axis=1)
    closed_ok = (r2 % 2 == 0) & (r2 >= 0) & (vsum >= 2 * vmax)
    radical_ok = (r2 % 2 == 1) & (r2 >= 1) & (vsum >= 2 * vmax - 1)
    return hx & np.where(radical, radical_ok, closed_ok)


_LIB_INDEX = {tuple(int(n) for n in v): i for i, v in enumerate(_LIB_VEC)}
_ROWS_PER_BLOCK = 1 << 12     # frontier rows per (rows, library, E) int16 broadcast


def _k_min2(t2: np.ndarray, radical: np.ndarray) -> np.ndarray:
    """The necessary piece count of a formula from its 2*DBE: k molecules of
    DBE >= 0 (closed) / one of DBE >= 1/2 (radical) sum to DBE + k - 1, so
    k >= 1 - DBE, resp. k >= 3/2 - DBE, and k >= 1."""
    return np.maximum(1, (3 + radical.astype(np.int64) - t2) // 2)


@lru_cache(maxsize=1 << 16)
def _decompose(key: tuple[int, ...], k_max: int) -> LossDecomposition | None:
    """`key` = loss counts on the `_EL` axis (all >= 0, not all 0).

    A search by total size k = 1, 2, ... k_max. The frontier holds every
    multiset of k - 1 library pieces that fits inside the loss (at most one
    radical piece, only for a radical loss), as ascending library-index tuples
    in lexicographic order. Size k reads either as one more library piece that
    makes the loss exactly (all named) or as the frontier multiset plus ONE
    unnamed remainder; all-named is tried first. The first hit is the answer,
    which is the documented choice: fewest pieces, then fewer unnamed, then
    library priority. Frontier multisets whose rest needs more pieces than are
    left (`_k_min2`) are dropped, so only plausible splits are ever held."""
    L = np.array(key, dtype=np.int16)
    t2 = 2 + int(L.astype(np.int64) @ _VM2)
    radical = bool(t2 % 2)
    k_min = int(_k_min2(np.array([t2]), np.array([radical]))[0])
    if k_min > k_max:
        return None
    formula = _formula_of(key)
    if k_min == 1:
        # size 1, the common case: a library formula, else one valid unnamed
        # piece (a library hit always has the loss's own parity -- tested)
        if key in _LIB_INDEX:
            return LossDecomposition(formula, (_LIB_NAMES[_LIB_INDEX[key]],), 0, radical)
        if _remainder_ok(L[None, :], np.array([radical]))[0]:
            return LossDecomposition(formula, (formula,), 1, radical)
    n_lib = len(_LIB_NAMES) if radical else _N_CLOSED     # a closed loss has no radical piece
    lib = _LIB_VEC[:n_lib]
    lib_rad = np.arange(n_lib) >= _N_CLOSED
    lib_ids = np.arange(n_lib)
    E = len(_EL)
    # the frontier, size 0: the empty multiset
    idx = np.zeros((1, 0), dtype=np.int16)        # (m, s) library indices, ascending
    acc = np.zeros((1, E), dtype=np.int16)        # (m, E) their element sums
    nrad = np.zeros(1, dtype=bool)                # (m,) holds the radical piece
    for k in range(1, k_max + 1):
        # every allowed one-piece extension that still fits, in (row, piece) =
        # lexicographic order: pieces at or after the row's last (no
        # permutations), and no second radical
        last = idx[:, -1].astype(np.int64) if idx.shape[1] else np.zeros(len(idx), np.int64)
        rows, js, exact = [], [], []
        for b in range(0, len(idx), _ROWS_PER_BLOCK):
            s = acc[b:b + _ROWS_PER_BLOCK, None, :] + lib[None, :, :]
            ok = ((s <= L).all(axis=2) & (lib_ids[None, :] >= last[b:b + _ROWS_PER_BLOCK, None])
                  & ~(lib_rad[None, :] & nrad[b:b + _ROWS_PER_BLOCK, None]))
            r, j = np.nonzero(ok)
            rows.append(r + b)
            js.append(j)
            exact.append((s[r, j] == L).all(axis=1))
        rows, js, exact = np.concatenate(rows), np.concatenate(js), np.concatenate(exact)
        if k >= max(k_min, 2):                    # size 1 was the fast path above
            if exact.any():                       # all named
                i = int(np.flatnonzero(exact)[0])
                named = tuple(int(n) for n in idx[rows[i]]) + (int(js[i]),)
                return LossDecomposition(formula, tuple(_LIB_NAMES[n] for n in named),
                                         0, radical)
            # named frontier multiset + one unnamed remainder, which is the
            # radical piece when the named part has none
            rem = L[None, :] - acc
            hit = np.flatnonzero(rem.any(axis=1) & _remainder_ok(rem, radical & ~nrad))
            if len(hit):
                h = int(hit[0])
                named = tuple(_LIB_NAMES[int(n)] for n in idx[h])
                return LossDecomposition(formula, named + (_formula_of(rem[h]),), 1, radical)
        if k == k_max:
            return None
        # the size-k frontier; an exact row is complete and cannot grow
        rows, js = rows[~exact], js[~exact]
        idx = np.concatenate([idx[rows], js[:, None].astype(np.int16)], axis=1)
        acc = acc[rows] + lib[js]
        nrad = nrad[rows] | lib_rad[js]
        rest = (L[None, :] - acc).astype(np.int64)
        keep = _k_min2(2 + rest @ _VM2, radical & ~nrad) <= k_max - k
        idx, acc, nrad = idx[keep], acc[keep], nrad[keep]
        if not len(idx):
            return None
    return None


def decompose_loss(loss: str | Mapping[str, int], *, k_max: int = 3,
                   allow_radical: bool = True) -> LossDecomposition | None:
    """Split a neutral loss into <= k_max molecules, or None when it cannot be.

    Closed-shell loss (integer DBE): closed-shell pieces only. Radical loss
    (half-integer DBE; None unless `allow_radical`): exactly one radical piece
    plus closed-shell ones. Pieces are library formulas plus at most ONE
    unnamed remainder, which must itself pass: integer DBE >= 0 (radical:
    half-integer >= 1/2), Senior's second rule (sum of valences >= 2 x the
    largest; radical: >= 2 x largest - 1), and at least one H or halogen --
    the clause that rejects bare-carbon / bare-oxygen pieces (C4, C, O; CO,
    CO2, N2, O2, SO2, CS2, COS, N2O stay possible because they are named). The choice is
    deterministic: fewest pieces, then fewer unnamed, then library priority.
    A loss string is parsed strictly ('C2H4o2' or 'H2O-H2' raise). k_max is
    1..K_MAX_LIMIT. Memoised per (loss, k_max)."""
    k_max = _check_k_max(k_max)
    if isinstance(loss, str):
        counts = _parse_body(loss.strip(), f"loss {loss!r}")
    else:
        counts = dict(loss)
    if any(n < 0 for n in counts.values()):
        raise ValueError(f"negative count in loss {loss!r}")
    key = tuple(int(n) for n in _vec(counts))
    if not any(key):
        raise ValueError(f"empty loss {loss!r}")
    dec = _decompose(key, k_max)
    if dec is not None and dec.radical and not allow_radical:
        return None
    return dec


# ---------------------------------------------------------------------------
# Sub-formula grid of a parent ion
# ---------------------------------------------------------------------------
class _Grid(NamedTuple):
    els: tuple[str, ...]   # the parent's elements (its own axis)
    col: list[int]         # their positions on the global `_EL` axis
    parent: np.ndarray     # (E,) parent counts on the global axis
    sub: np.ndarray        # (n, len(els)) valid fragment counts, mass-sorted
    mass: np.ndarray       # (n,) exact ion m/z, ascending
    t2: np.ndarray         # (n,) 2*DBE


def _check_grid_size(key: tuple[tuple[str, int], ...], charge: int) -> None:
    n_rows = math.prod(n + 1 for _, n in key)
    if n_rows > _GRID_MAX_ROWS:
        raise ValueError(f"parent {format_ion(dict(key), charge)} has {n_rows:,} "
                         f"sub-formulas (limit {_GRID_MAX_ROWS:,})")


@lru_cache(maxsize=32)
def _parent_grid(key: tuple[tuple[str, int], ...], charge: int) -> _Grid:
    """Every element-wise sub-formula of the parent ion that passes the
    fragment gate (`_gate`, the one `fragment_ok` uses; the parent itself
    always stays), with its ion m/z, sorted by mass. Built once per parent
    composition AND charge -- the masses carry `- charge * M_E`, so the cache
    key must too (numpy, one meshgrid): a C30H50O15 parent has 25k
    sub-formulas before the gate, the 2026-09-10 siloxane adduct
    C8H28^NO4Si4+ 13k."""
    _check_grid_size(key, charge)
    els = tuple(e for e, _ in key)
    cnt = [n for _, n in key]
    axes = np.meshgrid(*[np.arange(n + 1, dtype=np.int16) for n in cnt], indexing="ij")
    sub = np.stack([a.ravel() for a in axes], axis=1)
    col = [_IDX[e] for e in els]
    ok, t2 = _gate(sub, col)
    # the parent itself (the last meshgrid row) is an INPUT, not a reading, so
    # the gate never removes it: an ammonium cluster of a saturated molecule
    # sits at DBE -1.5 (C6H14O2 + ^NH4+ = C6H18^NO2+, a co-isolate of the
    # 2026-09-10 file's 137.057 window) and is still its own precursor peak
    ok[-1] = True
    sub, t2 = sub[ok], t2[ok]
    mass = sub.astype(np.float64) @ np.array([C.M[e] for e in els]) - charge * C.M_E
    order = np.argsort(mass, kind="stable")
    sub, t2, mass = sub[order], t2[order], mass[order]
    parent = np.zeros(len(_EL), dtype=np.int64)
    parent[col] = cnt
    return _Grid(els, col, parent, sub, mass, t2)


def _ion_key(counts: Mapping[str, int]) -> tuple[tuple[str, int], ...]:
    return tuple((e, int(counts[e])) for e in _EL if counts.get(e, 0))


def _resolve_parent(parent: str) -> tuple[tuple[tuple[str, int], ...], int]:
    """A parent is an ion formula STRING, checked by `parse_ion` (sign, known
    elements, radical mark); anything else raises. format_ion(counts, charge)
    turns parsed counts back into one."""
    counts, charge = parse_ion(parent)
    return _ion_key(counts), charge


# ---------------------------------------------------------------------------
# Candidates for one peak and one parent
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FragmentCandidate:
    """One reading of a fragment m/z as a sub-formula of one parent ion."""
    formula: str            # fragment ion, e.g. 'C6H9+'
    parity: str             # 'EE' / 'OE'
    dbe: float
    mz_theo: float          # exact m/z of `formula`
    ppm: float              # (measured - theoretical) / theoretical * 1e6
    loss: str               # parent - fragment, Hill; '' for the precursor itself
    loss_pieces: int
    loss_name: str          # pieces joined by ' + '; '' for the precursor
    unnamed_pieces: int
    radical_loss: bool      # half-integer-DBE loss (fragment parity != parent's)
    is_precursor: bool
    pieces: tuple[str, ...] = ()

    def sort_key(self) -> tuple:
        """Within one (peak, parent): closed-shell losses before radical ones,
        then fewer loss pieces, fewer unnamed pieces, smaller |ppm|, formula."""
        return (self.radical_loss, self.loss_pieces, self.unnamed_pieces,
                abs(self.ppm), self.formula)


def _tolerance_da(mz: float, tol_ppm: float, tol_mda: float) -> float:
    # the mDa floor: an Orbitrap's low-mass error is an ABSOLUTE offset (the
    # 2026-09-10 file's sub-80 ions sat at -0.12 mDa, assignment/masscal.py),
    # and its C2H3O+ fragment read -3.8 / -4.1 ppm (-0.16 / -0.18 mDa) after
    # the precursor recalibration. 5 ppm at m/z 43 is 0.21 mDa -- no margin.
    return max(tol_ppm * mz * 1e-6, tol_mda * 1e-3)


def _check_tol(tol_ppm: float, tol_mda: float) -> None:
    for name, v in (("tol_ppm", tol_ppm), ("tol_mda", tol_mda)):
        if not (isinstance(v, (int, float, np.floating, np.integer))
                and math.isfinite(v) and v >= 0):
            raise ValueError(f"{name} must be a finite number >= 0, got {v!r}")


def _candidates(pkey: tuple[tuple[str, int], ...], charge: int, mz: float,
                tol_ppm: float, tol_mda: float, k_max: int,
                allow_radical: bool) -> list[FragmentCandidate]:
    grid = _parent_grid(pkey, charge)
    tol = _tolerance_da(mz, tol_ppm, tol_mda)
    lo = int(np.searchsorted(grid.mass, mz - tol, side="left"))
    hi = int(np.searchsorted(grid.mass, mz + tol, side="right"))
    parent_odd = (2 + sum(_VM2[_IDX[e]] * n for e, n in pkey)) % 2 == 0   # OE parent
    out: list[FragmentCandidate] = []
    for i in range(lo, hi):
        theo = float(grid.mass[i])
        t2 = int(grid.t2[i])
        frag = {e: int(n) for e, n in zip(grid.els, grid.sub[i]) if n}
        formula = format_ion(frag, charge)
        parity = "EE" if t2 % 2 else "OE"
        ppm = (mz - theo) / theo * 1e6
        loss = grid.parent.copy()
        loss[grid.col] -= grid.sub[i]
        key = tuple(int(n) for n in loss)
        if not any(key):
            out.append(FragmentCandidate(formula, parity, t2 / 2, theo, ppm, "", 0, "",
                                         0, False, True))
            continue
        dec = _decompose(key, k_max)
        if dec is None or (dec.radical and not allow_radical):
            continue
        # parity bookkeeping: a radical loss flips the electron parity
        assert dec.radical == ((parity == "OE") != parent_odd)
        out.append(FragmentCandidate(formula, parity, t2 / 2, theo, ppm, dec.formula,
                                     dec.k, dec.name, dec.unnamed, dec.radical, False,
                                     dec.pieces))
    out.sort(key=FragmentCandidate.sort_key)
    return out


def fragment_candidates(parent_ion: str, mz: float, tol_ppm: float = 5.0,
                        tol_mda: float = 0.3, k_max: int = 3,
                        allow_radical: bool = True) -> list[FragmentCandidate]:
    """Every sub-formula of `parent_ion` (an ion formula string, see
    `parse_ion`) within tolerance of `mz` whose loss decomposes (see
    `decompose_loss`), best first.

    The window is max(tol_ppm * mz, tol_mda) -- ppm at high mass, an absolute
    floor at low mass. `mz` should already be on the recalibrated axis
    (`apply_offset`). The parent itself is a candidate (is_precursor, loss '').
    Ranking: closed-shell losses before radical ones, then fewer loss pieces,
    fewer unnamed pieces, smaller |ppm|, formula. Equal readings are all
    returned -- nothing is hidden behind the first."""
    _check_tol(tol_ppm, tol_mda)
    k_max = _check_k_max(k_max)
    mz = float(mz)
    if not (math.isfinite(mz) and mz > 0):
        raise ValueError(f"mz must be a finite positive number, got {mz!r}")
    pkey, charge = _resolve_parent(parent_ion)
    return _candidates(pkey, charge, mz, tol_ppm, tol_mda, k_max, bool(allow_radical))


# ---------------------------------------------------------------------------
# Recalibration (constant-ppm model)
# ---------------------------------------------------------------------------
class AnchorResidual(NamedTuple):
    ion: str
    measured_mz: float
    theo_mz: float
    error_ppm: float       # before the offset
    residual_ppm: float    # after it


@dataclass(frozen=True)
class MassOffset:
    offset_ppm: float
    anchors: tuple[AnchorResidual, ...]

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(list(self.anchors), columns=list(AnchorResidual._fields))


def apply_offset(mz, offset_ppm: float):
    """The recalibrated m/z: mz / (1 + offset_ppm * 1e-6). Scalars and arrays."""
    return mz / (1.0 + float(offset_ppm) * 1e-6)


def mass_offset(anchors: Iterable[tuple[float, str]]) -> MassOffset:
    """Constant-ppm offset of an MS2 spectrum from anchors of known composition:
    (measured_mz, ion_formula) pairs -- the surviving precursor, optionally the
    reagent ion. offset_ppm = median anchor error (signed: negative when the
    axis reads low; the mean of two anchors); each anchor's theoretical m/z is
    `chemistry.ion_mass` of its own formula AND charge (an anion sits two
    electron masses above the cation), and its residual is read after the
    offset is applied. Constant-ppm only: one or two anchors
    cannot fit the low-mass absolute term, which is what `tol_mda` absorbs."""
    rows = []
    for item in anchors:
        measured, ion = item
        measured = float(measured)
        if not (math.isfinite(measured) and measured > 0):
            raise ValueError(f"anchor m/z must be finite and positive, got {item!r}")
        counts, charge = parse_ion(ion)
        theo = C.ion_mass(counts, charge)
        rows.append((format_ion(counts, charge), measured, theo,
                     (measured - theo) / theo * 1e6))
    if not rows:
        raise ValueError("mass_offset needs at least one anchor")
    offset = float(np.median([r[3] for r in rows]))
    res = tuple(AnchorResidual(ion, m, theo, err,
                               (apply_offset(m, offset) - theo) / theo * 1e6)
                for ion, m, theo, err in rows)
    return MassOffset(offset, res)


# ---------------------------------------------------------------------------
# Whole spectrum x window ions -> long table
# ---------------------------------------------------------------------------
OUTPUT_COLUMNS: dict[str, str] = {
    "peak": "int64", "mz": "float64", "mz_cal": "float64", "intensity": "float64",
    "parent": "str", "parent_ion": "str", "formula": "str", "parity": "str",
    "dbe": "float64", "ppm": "float64", "loss": "str", "loss_pieces": "int64",
    "loss_name": "str", "unnamed_pieces": "int64", "radical_loss": "bool",
    "is_precursor": "bool", "rank": "int64", "n_candidates": "int64",
    "n_parents": "int64",
}


def _normalise_peaks(peaks) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(peaks, pd.DataFrame):
        if "mz" not in peaks.columns:
            raise ValueError("peaks DataFrame needs an 'mz' column")
        mz = pd.to_numeric(peaks["mz"], errors="raise").to_numpy(dtype=np.float64)
        inten = (pd.to_numeric(peaks["intensity"], errors="coerce").to_numpy(np.float64)
                 if "intensity" in peaks.columns else np.full(len(mz), np.nan))
    else:
        arr = np.asarray(peaks, dtype=np.float64)
        if arr.ndim > 1:
            # (mz, intensity) pairs or an (n, 2) centroid array: flattening
            # would read every intensity as one more m/z
            raise ValueError(f"peaks must be a 1-D array-like of m/z, got shape {arr.shape}; "
                             f"pass a DataFrame with 'mz' and 'intensity' columns instead")
        mz = arr.reshape(-1)
        inten = np.full(len(mz), np.nan)
    if not np.isfinite(mz).all() or (mz <= 0).any():
        raise ValueError("every peak m/z must be finite and positive")
    return mz, inten


def _normalise_parents(parents) -> list[tuple[str, str, tuple, int]]:
    """-> [(label, ion formula, key, charge)] in the order given."""
    if isinstance(parents, str):
        items = [(parents, parents)]
    elif isinstance(parents, Mapping):
        items = [(str(k), v) for k, v in parents.items()]
    elif isinstance(parents, Sequence):
        items = [(str(p), p) for p in parents]
    else:
        raise ValueError(f"parents must be an ion formula or a mapping label -> ion "
                         f"formula, got {type(parents).__name__}")
    if not items:
        raise ValueError("no parent ions given")
    labels = [lab for lab, _ in items]
    if len(set(labels)) != len(labels):
        raise ValueError(f"duplicate parent labels: {labels}")
    out = []
    for lab, ion in items:
        pkey, charge = _resolve_parent(ion)
        _check_grid_size(pkey, charge)            # refused up front, peaks or not
        out.append((lab, format_ion(dict(pkey), charge), pkey, charge))
    if len({c for *_, c in out}) > 1:
        raise ValueError("parents of both polarities in one window")
    return out


def assign_fragments(peaks, parents, *, offset_ppm: float = 0.0, tol_ppm: float = 5.0,
                     tol_mda: float = 0.3, k_max: int = 3,
                     allow_radical: bool = True) -> pd.DataFrame:
    """Read every MS2 peak against every ion of the isolation window.

    `peaks`: a DataFrame with `mz` (and optionally `intensity`), or a 1-D
    array-like of m/z (2-D input such as (mz, intensity) pairs raises).
    `parents`: one ion formula string, an ordered mapping label -> ion formula
    string, or a sequence of them -- the target plus the co-isolated window
    ions. Every parent's sub-formula grid is size-checked before any peak.
    This is the COMPOSITION test only: a peak readable from two parents keeps
    both readings (`n_parents` 2); picking one needs intensities across
    collision energies and is not done here.

    Long format, one row per (peak, parent, candidate), in input peak order,
    then parent order as given, then rank. `peak` is the 0-based row position
    in the input. `rank` 1 = best within (peak, parent); `n_candidates` counts
    them; `n_parents` counts the parents with >= 1 candidate for the peak. A
    peak no parent explains still gets one row, so every input peak is in the
    output: its own peak / mz / mz_cal / intensity, '' in every other text
    column, 0 in the piece counts, rank, n_candidates and n_parents, False in
    both flags, NaN dbe and ppm. Without intensities (an m/z array, or a
    DataFrame with no `intensity` column) `intensity` is NaN. `mz_cal` =
    `apply_offset(mz, offset_ppm)` is the axis every candidate is judged on.
    Column dtypes are fixed (`OUTPUT_COLUMNS`), whatever the content."""
    _check_tol(tol_ppm, tol_mda)
    k_max = _check_k_max(k_max)
    if not math.isfinite(float(offset_ppm)):
        raise ValueError(f"offset_ppm must be finite, got {offset_ppm!r}")
    mz, inten = _normalise_peaks(peaks)
    plist = _normalise_parents(parents)
    mz_cal = apply_offset(mz, offset_ppm)
    cols: dict[str, list] = {c: [] for c in OUTPUT_COLUMNS}

    def emit(i, parent, parent_ion, cand, rank, n_cand, n_par):
        cols["peak"].append(i)
        cols["mz"].append(mz[i])
        cols["mz_cal"].append(mz_cal[i])
        cols["intensity"].append(inten[i])
        cols["parent"].append(parent)
        cols["parent_ion"].append(parent_ion)
        cols["formula"].append(cand.formula if cand else "")
        cols["parity"].append(cand.parity if cand else "")
        cols["dbe"].append(cand.dbe if cand else np.nan)
        cols["ppm"].append(cand.ppm if cand else np.nan)
        cols["loss"].append(cand.loss if cand else "")
        cols["loss_pieces"].append(cand.loss_pieces if cand else 0)
        cols["loss_name"].append(cand.loss_name if cand else "")
        cols["unnamed_pieces"].append(cand.unnamed_pieces if cand else 0)
        cols["radical_loss"].append(cand.radical_loss if cand else False)
        cols["is_precursor"].append(cand.is_precursor if cand else False)
        cols["rank"].append(rank)
        cols["n_candidates"].append(n_cand)
        cols["n_parents"].append(n_par)

    for i in range(len(mz)):
        per_parent = [(lab, ion, _candidates(pkey, charge, float(mz_cal[i]), tol_ppm,
                                             tol_mda, k_max, bool(allow_radical)))
                      for lab, ion, pkey, charge in plist]
        n_par = sum(1 for *_, c in per_parent if c)
        if n_par == 0:
            emit(i, "", "", None, 0, 0, 0)
            continue
        for lab, ion, cands in per_parent:
            for r, cand in enumerate(cands, start=1):
                emit(i, lab, ion, cand, r, len(cands), n_par)
    return pd.DataFrame({c: pd.Series(v, dtype=OUTPUT_COLUMNS[c])
                         for c, v in cols.items()})


def best_readings(df: pd.DataFrame) -> pd.DataFrame:
    """The rank-1 reading of every (peak, parent) plus the unexplained rows --
    one row per peak and explaining parent, in the input order."""
    keep = (df["rank"] == 1) | (df["n_candidates"] == 0)
    return df.loc[keep].reset_index(drop=True)
