"""MS2 readings across collision energies, and the ladder gate on a fourth loss piece.

Pure, on top of `subformula.assign_fragments`: no Mascope, no I/O. One parent
window's spectra (one per energy) are read against the same window ions with one
recalibration, and stacked into one long table (`READINGS_COLUMNS`).

The ladder gate. `assign_fragments` defaults to `k_max` 3: every extra loss piece
admits more losses, and so more chance compositions, at every peak. An ammonium
adduct of a polyol or an oxygenated product walks a longer chain -- [M+NH4]+
loses NH3 and then water step by step, so its fourth step (NH3 + 3 H2O, DBE -3)
needs four molecules and no library can do it in fewer. The gate admits a
four-piece reading only when the SAME window ion shows, at any energy of the
window, the step before it: a rank-1 reading with <= 3 pieces whose loss is this
loss minus ONE named closed-shell library molecule (NH3 + 2 H2O, or 3 H2O),
whether the printed decomposition holds that piece or not; an unnamed
remainder or a radical (OH*) is no step. A chain whose previous
step is observed is a ladder; a lone four-piece composition is not read. On our
own 2026-09-10 and 2026-09-11 files no peak needs a fourth piece at all (the
gate reads them exactly as k_max 3 does -- tested); the gate is for the adduct
chains those files do not hold.
"""
from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from peaky.chem import chemistry as C
from peaky.ms2 import subformula as SF

__all__ = [
    "GATE_LADDER", "GATE_STRICT", "GATE_UNGATED", "LADDER_K", "READINGS_COLUMNS",
    "gate_mode", "ladder_gate", "rerank", "read_window",
]

#: The gate modes (`ms2_analyze_manifest.json` records which one ran).
GATE_LADDER = "ladder-k4"        # read at k_max 4, keep a 4-piece reading only on a ladder
GATE_STRICT = "strict-k3"        # k_max 3, nothing gated (PR #78's default)
GATE_UNGATED = "ungated"         # an explicit k_max, every reading kept
#: The piece count the ladder gate admits: one more than the default k_max.
LADDER_K = 4

#: The long readings table: the spectrum's identity and energy, then every
#: `assign_fragments` column, then the ladder support of a gated reading.
READINGS_COLUMNS: dict[str, str] = {
    "spectrum": "str", "parent_mz": "float64", "nce": "float64", "hcd_v": "float64",
    **SF.OUTPUT_COLUMNS, "ladder_support": "str",
}
_SPEC_COLS = ("spectrum", "parent_mz", "nce", "hcd_v")


def gate_mode(k_max: int | None = None, *, ladder: bool = True) -> tuple[str, int]:
    """(mode, the k_max the window is READ at). An explicit `k_max` is ungated
    (k_max 3 is the strict default); without one the ladder gate reads at 4
    unless `ladder` is off, which is the strict 3."""
    if k_max is not None:
        k = SF._check_k_max(k_max)
        return (GATE_STRICT if k == 3 else GATE_UNGATED), k
    return (GATE_LADDER, LADDER_K) if ladder else (GATE_STRICT, 3)


def _vec(formula: str) -> np.ndarray:
    """A formula's counts on subformula's fixed element axis (N and ^N are two
    elements there)."""
    return SF._vec({e: n for e, n in C.parse_formula(formula).items() if n})


def _minus_vec(loss: str, piece: str) -> np.ndarray | None:
    """loss - piece as a count vector, None when the piece is not inside it."""
    d = _vec(loss) - _vec(piece)
    return None if (d < 0).any() else d


def _minus(loss: str, piece: str) -> str | None:
    """loss - piece as a Hill formula written as `assign_fragments` writes a
    loss (`subformula._formula_of`, one fixed element order), None when the
    piece is not inside it. The ladder compares count vectors, never these
    strings: `format_formula` sorts N and ^N by the same key and keeps their
    input order, so a formula built from a set would follow the hash seed."""
    d = _minus_vec(loss, piece)
    return None if d is None else SF._formula_of(d)


def _unexplained_like(row: pd.Series) -> dict:
    """The one row a peak keeps when no parent reads it (subformula §3 step 8),
    with the peak's own spectrum, m/z and intensity."""
    out = {c: row[c] for c in _SPEC_COLS + ("peak", "mz", "mz_cal", "intensity")}
    for c, t in SF.OUTPUT_COLUMNS.items():
        if c in out:
            continue
        out[c] = "" if t == "str" else (False if t == "bool" else
                                        (0 if t == "int64" else math.nan))
    out["ladder_support"] = ""
    return out


def _frame(rows: list[dict] | pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame(rows) if not isinstance(rows, pd.DataFrame) else rows
    return pd.DataFrame({c: pd.Series(df[c].to_numpy() if c in df else [], dtype=t)
                         for c, t in READINGS_COLUMNS.items()})


def rerank(df: pd.DataFrame, keep) -> pd.DataFrame:
    """Keep the rows `keep` (a boolean mask) and re-rank: `rank` and
    `n_candidates` within (spectrum, peak, parent), `n_parents` within
    (spectrum, peak). A peak whose every reading was dropped becomes ONE
    unexplained row (rank 0, '' text, NaN dbe / ppm, ...), at the place its
    first row held; the order of everything else is kept."""
    keep = np.asarray(keep, dtype=bool)
    if keep.all() or not len(df):
        return _frame(df.reset_index(drop=True))
    df = df.reset_index(drop=True)
    out: list[dict] = []
    for _, g in df.groupby(["spectrum", "peak"], sort=False):
        kept = g[keep[g.index]]
        if not len(kept):
            first = g.iloc[0]
            out.append(_unexplained_like(first) if first["rank"] else first.to_dict())
            continue
        kept = kept.copy()
        if (kept["rank"] > 0).any():
            pos = kept.groupby("parent", sort=False).cumcount() + 1
            kept["rank"] = pos.to_numpy()
            kept["n_candidates"] = kept.groupby("parent", sort=False)["rank"].transform(
                "size").to_numpy()
            kept["n_parents"] = kept["parent"].nunique()
        out.extend(kept.to_dict("records"))
    return _frame(out)


def ladder_gate(df: pd.DataFrame, *, k: int = LADDER_K) -> pd.DataFrame:
    """Keep a `k`-piece reading only when the same window ion's step before it
    is read: at any spectrum of the table (one window, every energy), a reading
    of the same `parent` that is rank 1 among its <= k-1-piece candidates
    (what k_max k-1 would rank first) and whose loss is this loss minus ONE
    named closed-shell library molecule, printed among this reading's pieces
    or not (`_ladder_step`; an unnamed remainder or a radical is no step).
    `ladder_support` records that loss (the first printed piece, in the
    decomposition's order, that is one and finds one; else the first such
    step). Readings above k pieces are dropped; every
    other reading is kept as it is; then `rerank`.

    The table must hold ONE window, read at k_max >= k (`read_window`)."""
    if not len(df):
        return _frame(df)
    df = df.reset_index(drop=True).copy()
    if "ladder_support" not in df:
        df["ladder_support"] = ""
    read = df["rank"] > 0
    # rank 1 among the <= k-1-piece candidates of each (spectrum, peak, parent):
    # the candidates are already in rank order, so it is the first such row
    short = read & (df["loss_pieces"] <= k - 1) & ~df["is_precursor"]
    first = short & ~df[short].duplicated(["spectrum", "peak", "parent"]).reindex(
        df.index, fill_value=True)
    # each window ion's read steps as count vectors (never strings: see _minus)
    steps: dict[str, set] = {}
    for parent, loss in zip(df.loc[first, "parent"], df.loc[first, "loss"]):
        steps.setdefault(parent, set()).add(tuple(_vec(loss).tolist()))
    keep = ~read | (df["loss_pieces"] < k)
    support = df["ladder_support"].to_numpy(dtype=object)
    for i in np.flatnonzero(read & (df["loss_pieces"] == k)):
        row = df.iloc[i]
        mine = steps.get(row["parent"], set())
        prev = _ladder_step(row["loss"], row["loss_name"], mine)
        if prev is not None:
            keep[i] = True
            support[i] = SF._formula_of(prev)
    df["ladder_support"] = support
    return rerank(df, keep.to_numpy())


def _one_molecule(vec: np.ndarray) -> bool:
    """Is this count vector ONE named closed-shell library molecule (what
    `decompose_loss` reads in a single piece, with no unnamed remainder and no
    radical)?"""
    if (vec < 0).any() or not vec.any():
        return False
    dec = SF.decompose_loss({e: int(n) for e, n in zip(SF._EL, vec) if n}, k_max=1,
                            allow_radical=False)
    return dec is not None and dec.unnamed == 0


def _ladder_step(loss: str, loss_name: str, steps: set) -> np.ndarray | None:
    """The read step (a count vector of `steps`) a reading of `loss` stands
    on, or None: a read step that is this loss minus ONE named closed-shell
    library molecule (`_one_molecule`). First the pieces its decomposition
    prints that are one (in that order: `ladder_support` names the first
    that finds a step); then any read step one such molecule back (the steps
    in a fixed order). The printed decomposition is one of several of the
    same loss -- fewest pieces, then fewest unnamed, then library order --
    and need not hold the piece the ladder took: CH11NO5 prints as H3N + CH4O
    + H2O2 + H2O2, with no H2O, while the step one H2O back (CH9NO4, H2O +
    H2O + CH5NO2) is read. Nor is every printed piece a step: an unnamed
    remainder or a radical (OH*) is no molecule the chain lost, and taking
    it as one admitted a reading by what the printer happened to list
    (H8NO3 prints its OH*, so the step H7NO2 admitted it, while H8NO4 prints
    HO2* and the step one OH* back did not)."""
    for piece in loss_name.split(" + "):
        prev = _minus_vec(loss, piece)
        if (prev is not None and prev.any() and tuple(prev.tolist()) in steps
                and _one_molecule(_vec(piece))):
            return prev
    whole = _vec(loss)
    for st in sorted(steps):
        prev = np.asarray(st, dtype=whole.dtype)
        if prev.any() and _one_molecule(whole - prev):
            return prev
    return None


def read_window(spectra: Sequence[tuple[Mapping, pd.DataFrame]], ions: Mapping[str, str], *,
                offset_ppm: float = 0.0, tol_ppm: float = 5.0, tol_mda: float = 0.3,
                k_max: int | None = None, ladder: bool = True,
                allow_radical: bool = True) -> pd.DataFrame:
    """Read every spectrum of ONE parent window against its window ions.

    `spectra`: (meta, peaks) pairs, one per energy, where `meta` holds
    `spectrum`, `parent_mz`, `nce`, `hcd_v` and `peaks` is a DataFrame with
    `mz` and `intensity`. `ions`: label -> ion formula (the target first, then
    the co-isolates), passed to `assign_fragments` as they are. One
    `offset_ppm` for the whole window (`mass_offset` on its target's located
    precursor). The gate (`gate_mode`): by default the window is read at
    k_max 4 and `ladder_gate` keeps a four-piece reading only on a ladder;
    `ladder=False` is the strict k_max 3; an explicit `k_max` is read
    ungated. Columns `READINGS_COLUMNS`, in spectrum order, then the
    `assign_fragments` order."""
    mode, k_read = gate_mode(k_max, ladder=ladder)
    parts = []
    for meta, peaks in spectra:
        df = SF.assign_fragments(peaks, dict(ions), offset_ppm=offset_ppm, tol_ppm=tol_ppm,
                                 tol_mda=tol_mda, k_max=k_read, allow_radical=allow_radical)
        for c in _SPEC_COLS:
            df.insert(_SPEC_COLS.index(c), c, meta[c])
        df["ladder_support"] = ""
        parts.append(df)
    out = _frame(pd.concat(parts, ignore_index=True) if parts else pd.DataFrame())
    if mode == GATE_LADDER:
        out = ladder_gate(out, k=LADDER_K)
    return out
