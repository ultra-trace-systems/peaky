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

Harmonic ghosts (`harmonic_ghosts`). An Orbitrap reads m/z off an ion's
axial frequency f (m/z ~ 1/f^2), and a strong line's transient carries weak
harmonics at 2f and 3f: peaks at about m/z / 4 and m/z / 9 that no ion made.
Read as sub-formulas they would be fragments; `read_window` reads them as
no formula (`harmonic` names the line) and `breakdown.ion_budget` keeps them
out of the TIC. A place the window reads as an ion at some energy -- where no
line's band holds it, or above the cap of every line whose band does, clear
of the S/N floor -- is an ion's (`harmonic_ions`): no ghost there at any
energy, so a product is never read at some energies and erased at the others.
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
    "HARMONIC_ORDERS", "HARMONIC_PPM", "HARMONIC_CAP", "HARMONIC_CLEAR_SN", "gate_mode",
    "ladder_gate", "rerank", "harmonic_ions", "harmonic_ghosts", "read_window",
]

#: The gate modes (`ms2_analyze_manifest.json` records which one ran).
GATE_LADDER = "ladder-k4"        # read at k_max 4, keep a 4-piece reading only on a ladder
GATE_STRICT = "strict-k3"        # k_max 3, nothing gated (PR #78's default)
GATE_UNGATED = "ungated"         # an explicit k_max, every reading kept
#: The piece count the ladder gate admits: one more than the default k_max.
LADDER_K = 4

#: The long readings table: the spectrum's identity and energy, then every
#: `assign_fragments` column, then the ladder support of a gated reading and,
#: for a peak read as a harmonic ghost of a brighter line, what it is
#: (`harmonic_ghosts`; '' otherwise).
READINGS_COLUMNS: dict[str, str] = {
    "spectrum": "str", "parent_mz": "float64", "nce": "float64", "hcd_v": "float64",
    **SF.OUTPUT_COLUMNS, "ladder_support": "str", "harmonic": "str",
}

#: The harmonics read (n: the line's n-th harmonic, nf, seen at m/z / n^2).
#: 2f and 3f: on our own 2026-09-10 exploratory file the 179.107 window's
#: precursor carries its 2f ghost at m/z 44.777 in the server's S/N-3 spectra
#: at NCE 1, 5, 10 and 15 (0.08-0.10 % of the line, constant while the
#: precursor falls); a 3f ghost holds a few per cent of its line and is seen
#: wherever m/z / 9 lies above the scan's first mass. 4f (m/z / 16) was not
#: seen at all.
HARMONIC_ORDERS = (2, 3)
#: How far (ppm of m/z / n^2) a ghost may sit from its nominal place. m/z is
#: not exactly proportional to 1 / f^2 (the calibration has higher terms), so
#: the ghost sits off m/z / n^2 by an amount that depends on the calibration
#: and the m/z: -2 ppm on our own file, tens of ppm on others; 40 ppm holds
#: that with margin (2 mDa at m/z 50).
HARMONIC_PPM = 40.0
#: The most a ghost holds of its line in the SAME spectrum (by order): above
#: it the peak is an ion. 2f: 0.5 %, five times our own ghost's 0.08-0.10 %
#: (the real ions seen at m/z / 4 of a line hold a few % to tens of % of it).
#: 3f: 8 % -- a 3f ghost holds a few per cent of its line, and up to about
#: twice that where it is read near the S/N floor (a weak ghost is seen only
#: where the noise lifts it).
HARMONIC_CAP = {2: 0.005, 3: 0.08}
#: The S/N a peak in a harmonic band, above the cap of every line whose band
#: holds it, needs for that reading to make its place an ion's at every
#: energy (`harmonic_ions`). Near the S/N floor a peak is seen only where the
#: noise lifts it -- read up to about two noise sd high, so its share of its
#: line up to s / (s - 2) of the true one at S/N s: twice it at S/N 4, 1.25
#: times at 10. A ghost's share reads above the cap only there (at the
#: highest energies, where its line has fallen and the ghost is read only on
#: the noise's upswings); at S/N 10 a 3f ghost of a few per cent stays under
#: the 8 % cap. A spectrum without S/N (NaN) clears its place as before.
HARMONIC_CLEAR_SN = 10.0
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
    out["harmonic"] = ""
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


def _as_harmonic(df: pd.DataFrame, ghosts: Mapping[int, str]) -> pd.DataFrame:
    """One spectrum's `assign_fragments` rows with each ghost peak's readings
    replaced by ONE unexplained-like row (as `rerank` writes a peak whose
    every reading was dropped: rank 0, '' text, NaN numbers), at the place
    its first row held, its `harmonic` naming the line. Every other row is
    kept as it is (whole peaks only are dropped: nothing to re-rank)."""
    ghost = df["peak"].isin(list(ghosts)).to_numpy()
    first = ghost & ~df["peak"].duplicated().to_numpy()
    out = df[~ghost | first].copy()
    sel = out["peak"].isin(list(ghosts)).to_numpy()
    for c, t in SF.OUTPUT_COLUMNS.items():
        if c in ("peak", "mz", "mz_cal", "intensity"):
            continue
        out.loc[sel, c] = "" if t == "str" else (False if t == "bool" else
                                                 (0 if t == "int64" else math.nan))
    out.loc[sel, "ladder_support"] = ""
    out.loc[sel, "harmonic"] = out.loc[sel, "peak"].map(ghosts).astype(str)
    return out.reset_index(drop=True)


def harmonic_ions(spectra: Sequence[tuple], *, orders: Sequence[int] = HARMONIC_ORDERS,
                  ppm: float = HARMONIC_PPM, cap: Mapping[int, float] | None = None,
                  clear_sn: float = HARMONIC_CLEAR_SN) -> np.ndarray:
    """The places a window reads as ions, so that `harmonic_ghosts`
    (`ions_at`) reads a peak there as an ion at every energy. `spectra`: the
    window's spectra, each (m/z, intensity) or (m/z, intensity, S/N). A
    peak's m/z is such a place when, in its own spectrum, no line's cap
    holds it:

      * no line L of its spectrum has it within `ppm` of m/z(L) / n^2: a
        ghost exists only beside its line, so it is no ghost there -- a
        product of a fragile precursor that is gone at that energy, one that
        a ppm of jitter puts just out of the band, a spectrum of that peak
        alone;
      * or it holds MORE than `cap[n]` of every line whose band holds it,
        read at an S/N of at least `clear_sn` (or with none known). A ghost
        holds a constant share of its line (0.098 -> 0.081 % on our own file
        while the line falls by a fifth), so such a place is an ion's; but a
        ghost read near the S/N floor is seen only where the noise lifts it,
        its share read high (`HARMONIC_CLEAR_SN`): that reading clears
        nothing, else one noisy energy would switch the rule off for the
        ghost at every energy.

    A product that grows from under the cap at its first energies to above
    it later (or whose line is gone at the later ones) is so read at all of
    them, its onset and in-source flag those of the reading without the
    rule. What it costs: a product above the cap only where it is near the
    floor is read only there. Sorted."""
    cap = HARMONIC_CAP if cap is None else cap
    out: list[float] = []
    for spec in spectra:
        m = np.asarray(spec[0], dtype=float)
        i_ = np.asarray(spec[1], dtype=float)
        sn = (np.asarray(spec[2], dtype=float) if len(spec) > 2 and spec[2] is not None
              else np.full(len(m), np.nan))
        ok = np.isfinite(m) & np.isfinite(i_) & (i_ > 0)
        if not ok.any():
            continue
        o = np.argsort(m[ok], kind="stable")
        m, i_, sn = m[ok][o], i_[ok][o], sn[ok][o]
        in_band = np.zeros(len(m), dtype=bool)
        under = np.zeros(len(m), dtype=bool)
        for n in orders:
            t = m / float(n) ** 2                          # each line's place
            lo = np.searchsorted(m, t * (1.0 - ppm * 1e-6), side="left")
            hi = np.searchsorted(m, t * (1.0 + ppm * 1e-6), side="right")
            for j in np.flatnonzero(hi > lo):              # the lines with a peak in band
                k = np.arange(lo[j], hi[j])
                k = k[k != j]
                in_band[k] = True
                under[k] |= i_[k] <= cap[n] * i_[j]
        with np.errstate(invalid="ignore"):
            near_floor = in_band & (sn < clear_sn)        # NaN: no S/N known, not near
        out.extend(m[~under & ~near_floor].tolist())
    return np.sort(np.asarray(out, dtype=float))


def _near(m: np.ndarray, places: np.ndarray, tol_ppm: float, tol_mda: float) -> np.ndarray:
    """Which of `m` lie within the reading tolerance (`subformula`'s
    max(tol_ppm, tol_mda)) of one of the sorted `places`."""
    if not len(places):
        return np.zeros(len(m), dtype=bool)
    tol = np.maximum(tol_ppm * np.abs(m) * 1e-6, tol_mda * 1e-3)
    k = np.searchsorted(places, m)
    lo = places[np.clip(k - 1, 0, len(places) - 1)]
    hi = places[np.clip(k, 0, len(places) - 1)]
    with np.errstate(invalid="ignore"):
        return (np.abs(m - lo) <= tol) | (np.abs(m - hi) <= tol)


def harmonic_ghosts(mz, intensity, *, orders: Sequence[int] = HARMONIC_ORDERS,
                    ppm: float = HARMONIC_PPM, cap: Mapping[int, float] | None = None,
                    ions_at: Sequence[float] = (), tol_ppm: float = 5.0,
                    tol_mda: float = 0.3) -> dict[int, str]:
    """The peaks of ONE spectrum that are harmonic ghosts of a brighter line
    of the same spectrum -> {peak index: what it is}.

    Every line L is tested (a co-isolate's or a strong fragment's harmonic is
    as much a ghost as the precursor's): for each order n of `orders`, the
    peaks within `ppm` of m/z(L) / n^2 that hold at most `cap[n]` of L's
    intensity (`HARMONIC_CAP`); of several, the brightest -- one line has one
    ghost per order, and a second, weaker peak in the band is something else.
    A peak above the cap is an ion, whatever its place, and so is a peak
    within the reading tolerance (`tol_ppm` / `tol_mda`) of a place the
    window reads as an ion at another energy (`ions_at`, `harmonic_ions`: in
    no line's band there, or above the cap clear of the S/N floor): it is
    no candidate, and of the others in its band the
    brightest is the ghost -- a product brighter than the line's true ghost
    at one energy does not take its place. m/z ratios within one spectrum:
    the axis offset cancels. Where a peak is the ghost of several lines it
    is named after the brightest. On our own three files this flags the
    four 2f ghosts of the 2026-09-10 file's 179.107 precursor and nothing
    else (none of them a reading)."""
    cap = HARMONIC_CAP if cap is None else cap
    m = np.asarray(mz, dtype=float)
    i_ = np.asarray(intensity, dtype=float)
    out: dict[int, tuple[float, str]] = {}
    ok = np.isfinite(m) & np.isfinite(i_) & (i_ > 0)
    if ok.sum() < 2:
        return {}
    cand = ok & ~_near(m, np.sort(np.asarray(ions_at, dtype=float)), tol_ppm, tol_mda)
    for n in orders:
        target = m / float(n) ** 2
        for j in np.flatnonzero(ok):
            t = target[j]
            band = cand & (np.abs(m - t) <= ppm * 1e-6 * t) & (i_ <= cap[n] * i_[j])
            band[j] = False
            if not band.any():
                continue
            k = int(np.flatnonzero(band)[np.argmax(i_[band])])
            if k in out and out[k][0] >= i_[j]:
                continue
            out[k] = (float(i_[j]), (f"{n}f ghost of the line at {m[j]:.4f} (m/z / {n * n}): "
                                     f"{(m[k] - t) / t * 1e6:+.1f} ppm, "
                                     f"{i_[k] / i_[j] * 100:.3g} % of it"))
    return {k: v[1] for k, v in sorted(out.items())}


def read_window(spectra: Sequence[tuple[Mapping, pd.DataFrame]], ions: Mapping[str, str], *,
                offset_ppm: float = 0.0, tol_ppm: float = 5.0, tol_mda: float = 0.3,
                k_max: int | None = None, ladder: bool = True,
                allow_radical: bool = True, harmonics: bool = True) -> pd.DataFrame:
    """Read every spectrum of ONE parent window against its window ions.

    `spectra`: (meta, peaks) pairs, one per energy, where `meta` holds
    `spectrum`, `parent_mz`, `nce`, `hcd_v` and `peaks` is a DataFrame with
    `mz` and `intensity`. `ions`: label -> ion formula (the target first, then
    the co-isolates), passed to `assign_fragments` as they are. One
    `offset_ppm` for the whole window (`mass_offset` on its target's located
    precursor). The gate (`gate_mode`): by default the window is read at
    k_max 4 and `ladder_gate` keeps a four-piece reading only on a ladder;
    `ladder=False` is the strict k_max 3; an explicit `k_max` is read
    ungated. `harmonics` (default): a peak that is a harmonic ghost of a
    brighter line of its spectrum (`harmonic_ghosts`) is read as no formula
    -- one unexplained-like row whose `harmonic` names the line -- before the
    gate, so it supports no ladder either; a place some spectrum of the
    window reads as an ion (`harmonic_ions`, with the peaks'
    `signal_to_noise` where they carry it) is an ion's in all of them.
    Columns `READINGS_COLUMNS`, in spectrum order, then the
    `assign_fragments` order."""
    mode, k_read = gate_mode(k_max, ladder=ladder)
    spectra = list(spectra)
    ions_at = (harmonic_ions([(pk["mz"].to_numpy(dtype=float),
                               pk["intensity"].to_numpy(dtype=float),
                               (pk["signal_to_noise"].to_numpy(dtype=float)
                                if "signal_to_noise" in pk else None))
                              for _, pk in spectra])
               if harmonics else np.zeros(0))
    parts = []
    for meta, peaks in spectra:
        df = SF.assign_fragments(peaks, dict(ions), offset_ppm=offset_ppm, tol_ppm=tol_ppm,
                                 tol_mda=tol_mda, k_max=k_read, allow_radical=allow_radical)
        for c in _SPEC_COLS:
            df.insert(_SPEC_COLS.index(c), c, meta[c])
        df["ladder_support"] = ""
        df["harmonic"] = ""
        ghosts = (harmonic_ghosts(peaks["mz"].to_numpy(dtype=float),
                                  peaks["intensity"].to_numpy(dtype=float), ions_at=ions_at,
                                  tol_ppm=tol_ppm, tol_mda=tol_mda)
                  if harmonics and len(peaks) else {})
        if ghosts:
            df = _as_harmonic(df, ghosts)
        parts.append(df)
    out = _frame(pd.concat(parts, ignore_index=True) if parts else pd.DataFrame())
    if mode == GATE_LADDER:
        out = ladder_gate(out, k=LADDER_K)
    return out
