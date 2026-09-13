"""Time-resolved disposition layer.

A sum spectrum cannot tell a bright stable INLET/instrument contaminant from a
real ambient ANALYTE, nor a degenerate reagent-background cluster from chemistry.
A *time series* of the same sample over a day can: in a halide-CIMS the physically
meaningful quantity is the analyte normalised to the reagent ion (removes
instrument-sensitivity + reagent-flow common-mode drift), and then

  * a FLAT normalised trace (low coefficient of variation, no diel) == inlet /
    instrument background or a constant reagent cluster -- NOT ambient chemistry;
  * a VARIABLE trace that co-varies with a known chemical family == real ambient
    analyte.

This module ingests a batch's per-sample peak table, builds the reagent-normalised
intensity matrix, measures each peak's variability (`cv_norm`) and (optionally)
its correlation to reference family traces, and stamps a `ts_*` disposition onto
the ledger. It then applies CONSERVATIVE auto-actions: demote a flat di-bromide /
background-channel commit (TS-confirmed background) and flag inlet contaminants.
It never changes a formula -- only the tier/role annotation, with commentary.

All pure pandas/numpy; no network. Reference (2026-06-16 time-series unlock).
"""
from __future__ import annotations

import bisect
import functools
import os
import re

import numpy as np
import pandas as pd

from peaky.assignment import ledger as L

__version__ = "0.3.0"  # predicted diagnostic satellites in the batch stamp (stamp_source,
                       # track coherence); 0.2.1: bin_ids, the row-aligned bin rule

DEFAULT_TOL_PPM = 5.0
FLAT_CV = 0.25          # cv_norm below this == flat / background
COVARY_R = 0.70         # correlation above this == co-varies with the family


def auto_bin_minutes(ts: pd.DataFrame, *, target_bins: int = 50,
                     time_col: str = "datetime_utc") -> int:
    """Time-bin width (minutes) for the correlation / cluster / Van Krevelen layer.

    Bins at the NATIVE sample cadence (median inter-sample spacing) so the traces
    are NOT downsampled. A coarse bin (the old span/target_bins ~= 29 min on a 24 h
    batch) smears sharp features -- zero-air periods, fast trends -- that drive the
    real co-variation, pushing genuinely-changing channels into the flat bucket
    (validated on the June-3 uronium batch: native 5-min recovered 1018 changing
    channels / 95 families vs 752 / 54 at 29-min). Floored at 1 min; falls back to
    span/target_bins only when per-sample times are unavailable (<3 samples).
    Shared by clustering + VK so they bin identically.

    The bin is rounded UP (ceil) to the cadence, never down: a bin narrower than the
    real inter-sample spacing aliases — the fixed-width grid periodically catches
    ZERO samples (a beat between the grid pitch and the slightly-irregular sample
    times), leaving empty bins that render as a spurious regular comb of drop-to-floor
    teeth. ceil guarantees the bin is >= the sample spacing, so every bin holds >= 1
    sample. (e.g. a 73 s cadence -> 2 min, not the aliasing 1 min.)"""
    if "sample_item_id" in ts.columns:
        t = pd.to_datetime(ts.drop_duplicates("sample_item_id")[time_col], utc=True)
    else:
        t = pd.to_datetime(ts[time_col], utc=True).drop_duplicates()
    t = t.dropna().sort_values()
    if len(t) >= 3:
        cadence_min = t.diff().dropna().dt.total_seconds().median() / 60.0
        if cadence_min > 0:
            return max(1, int(np.ceil(cadence_min)))
    span_min = (t.max() - t.min()).total_seconds() / 60.0 if len(t) >= 2 else 30.0
    return max(1, int(round(span_min / target_bins)))


# ---------------------------------------------------------------------------
# match-flattened -> one row per physical peak
# ---------------------------------------------------------------------------
# Mascope's peak loaders return the MATCH-FLATTENED table: "when a peak matches
# multiple isotopes it is expanded into one row per match" (SDK `load_peaks` /
# `samples.get_peaks`, both `matches=True` by default). The batch time series is a
# table of PHYSICAL peaks -- one row per (sample, peak) -- so a peak that two
# targets both claim comes back twice with byte-identical mz/area/height and only
# the advisory `target_*` columns differing.
#
# Two targets collide exactly when they imply the SAME ion: a neutral read as
# [M+NO3]- and a neutral one HNO3 heavier read as [M-H]- are the same ion formula,
# so both score ~1.0 on the same peak. The pair is then separated by exactly
# 0.00 ppm and survives every mass-based filter downstream.
#
# Left in, each such peak is counted twice: `build_matrix` sums heights per
# (sample, bin) so that bin's intensity doubles, and a per-trace peak count reads
# 2.0 peaks/sample for one ion -- which looks like two merged ions (measured:
# 0.29% and 0.42% of rows on two field batches, always pairs).
#
# Ranking columns, most decisive first; the target ids are the final tie-break so
# the winner is fixed by content, never by the order the server returned rows in.
_MATCH_SCORE_COLS = ("match_score_compound", "match_score_ion", "match_score_isotope")
_MATCH_ID_COLS = ("target_isotope_id", "target_ion_id", "target_compound_id")


def collapse_peak_matches(peaks: pd.DataFrame, *, log=None) -> pd.DataFrame:
    """Collapse Mascope's match-expanded rows to ONE row per physical peak.

    Keyed on (sample_item_id, peak_id) -- or (sample_item_id, mz) for a frame
    trimmed to the TS columns, which carries no peak_id. The surviving row is the
    one whose match scores are highest, so the (advisory) Mascope identity in the
    `target_*` / `ionization_mechanism` columns stays the best one on offer;
    peaky's own `neutral_formula` / `adduct` / `tier`, stamped later by
    `annotate_peaks` from the merged ledger, are the authoritative assignment.

    Row ORDER is preserved and a frame that is already one-row-per-peak comes back
    unchanged (not even copied), so this is free on clean input and idempotent on
    its own output -- safe to call at every point a time series enters."""
    if peaks is None or not hasattr(peaks, "columns") or not len(peaks):
        return peaks
    cols = peaks.columns
    if "peak_id" in cols and not peaks["peak_id"].isna().any():
        peak_key = "peak_id"
    elif "mz" in cols:
        peak_key = "mz"
    else:
        return peaks            # nothing identifies a peak; never key on the sample
    key = (["sample_item_id", peak_key] if "sample_item_id" in cols else [peak_key])
    d = peaks.reset_index(drop=True)
    if not d.duplicated(subset=key, keep=False).any():
        return peaks
    rank, asc = [], []
    for c in _MATCH_SCORE_COLS:
        if c in cols:
            rank.append(c)
            asc.append(False)                      # best score first
    for c in _MATCH_ID_COLS:
        if c in cols:
            rank.append(c)
            asc.append(True)                       # deterministic final tie-break
    if rank:
        # mergesort == stable, so unmatched peaks (all-NaN scores) keep their order
        order = d.sort_values(rank, ascending=asc, kind="mergesort",
                              na_position="last").index.to_numpy()
    else:
        order = np.arange(len(d))
    lost = d.iloc[order].duplicated(subset=key, keep="first").to_numpy()
    out = d.iloc[np.sort(order[~lost])].reset_index(drop=True)
    if log:
        log(f"[ts] collapsed {int(lost.sum())} match-expanded row(s) -> "
            f"{len(out)} physical peaks (a peak claimed by >1 target came back "
            f"once per target, at identical m/z and height)")
    return out


# ---------------------------------------------------------------------------
# matrix construction
# ---------------------------------------------------------------------------
def bin_ids(peaks: pd.DataFrame, *, tol_ppm: float = DEFAULT_TOL_PPM,
            mz_col="mz", height_col="height", sample_col="sample_item_id") -> np.ndarray:
    """The m/z bin of every ROW of `peaks`, by the gap rule `build_matrix` pivots
    on: sort the peaks by m/z and start a new bin wherever two consecutive m/z
    are more than `tol_ppm` apart. Bins are numbered 0.. in ascending m/z -- the
    same numbers as that matrix's columns for the same table and tolerance
    (`build_matrix` calls this, so the two cannot drift). A row `build_matrix`
    drops (a missing sample id, m/z or height) gets -1 and never bridges a gap,
    so it cannot change where a bin ends for the rows that are kept.

    This is what a batch-level operation uses to ask "which bin is this peak
    in?" without pivoting the whole batch: the residual stage of
    `assign_batch.run` (sampling.residual_universe) reads the per-peak stamp
    and the per-sample heights off the long table, bin by bin, at a fraction
    of the dense matrix's memory."""
    n = len(peaks)
    out = np.full(n, -1, dtype=np.int64)
    if n == 0:
        return out
    d = peaks[[sample_col, mz_col, height_col]]
    idx = np.flatnonzero(d.notna().all(axis=1).to_numpy())
    if len(idx) == 0:
        return out
    mz = d[mz_col].to_numpy()[idx]           # native dtype: the same arithmetic
    order = np.argsort(mz, kind="mergesort")  # as the matrix's sorted column
    ms = mz[order]
    b = np.zeros(len(ms), dtype=np.int64)
    if len(ms) > 1:
        gaps = np.diff(ms) / ms[:-1] * 1e6
        b[1:] = np.cumsum(gaps > tol_ppm)
    out[idx[order]] = b
    return out


def build_matrix(peaks: pd.DataFrame, *, tol_ppm: float = DEFAULT_TOL_PPM,
                 mz_col="mz", height_col="height", sample_col="sample_item_id"
                 ) -> tuple[pd.DataFrame, pd.Series]:
    """Gap-cluster peaks into m/z bins (ppm tolerance; the rule is `bin_ids`)
    and pivot to a samples x bin intensity matrix. Returns (matrix, bin_mz)."""
    d = peaks[[sample_col, mz_col, height_col]].dropna().sort_values(mz_col).reset_index(drop=True)
    if len(d) == 0:
        return pd.DataFrame(), pd.Series(dtype=float)
    d["_bin"] = bin_ids(d, tol_ppm=tol_ppm, mz_col=mz_col, height_col=height_col,
                        sample_col=sample_col)
    wsum = (d[mz_col] * d[height_col]).groupby(d["_bin"]).sum()
    hsum = d[height_col].groupby(d["_bin"]).sum()
    bin_mz = (wsum / hsum).rename("mz")
    mat = d.pivot_table(index=sample_col, columns="_bin", values=height_col, aggfunc="sum")
    return mat, bin_mz


ANCHOR_MARGIN = 2.0     # how much brighter an off-ledger track must be to displace it

# --- sidelobe-contaminated ion CHANNELS --------------------------------------
# A saturating peak rings: FT/Gibbs sidelobes sit a few mDa either side of it at a
# ~fixed fraction of its height. When an assigned ion's m/z lands on one, the
# FORMULA can still be right (the neutral is corroborated on another channel) while
# the measured HEIGHT on that channel is the neighbour's sidelobe, not the analyte.
# Quantifying from it tracks the wrong compound.
#
# Static features CANNOT separate this from a real ion that merely sits near a
# bright peak. Measured over 25498 raw tracks / 30 runs of the 2026 field campaign:
#   satellite fraction  artifact 0.69% vs independent 0.23%  (artifact is BIGGER)
#   |Δm/z| to parent    artifact 11.5 mDa vs independent 10.1 mDa
# Only the TIME SERIES separates them — a sidelobe holds a ~constant ratio to its
# parent, an independent ion varies on its own:
#   ratio-cv            artifact 0.033-0.051   |gap|   independent 0.21-1.09
# Hence this runs at MERGE level (where the batch TS exists), not in per-file
# cleanup. SIDELOBE_CV sits in the empty gap, biased to under-flag.
SIDELOBE_DMZ = 0.012        # Da; how close the saturating parent must be
SIDELOBE_FACTOR = 100.0     # parent must be >= this x brighter (satellite <1%)
SIDELOBE_MIN_PARENT = 50000.0   # cps; below this a peak is too weak to ring
SIDELOBE_CV = 0.08          # ratio-to-parent cv below this == locked to the parent
SIDELOBE_MIN_PAIRS = 20     # samples needed before a cv is trustworthy
SIDELOBE_MIN_FRAC = 0.20    # a track must carry this share of the ion's samples


def flag_sidelobe_channels(merged: pd.DataFrame, ts_peaks: pd.DataFrame, *,
                           dmz: float = SIDELOBE_DMZ, factor: float = SIDELOBE_FACTOR,
                           min_parent: float = SIDELOBE_MIN_PARENT,
                           cv_max: float = SIDELOBE_CV,
                           min_pairs: int = SIDELOBE_MIN_PAIRS,
                           min_frac: float = SIDELOBE_MIN_FRAC,
                           track_gap: float = 8e-4,
                           demote_uncorroborated: bool = True, log=print) -> dict:
    """Stamp ``intensity_suspect`` / ``sidelobe_parent_mz`` onto a merged ledger.

    ``intensity_suspect=True`` means: TRUST THE FORMULA, DO NOT QUANTIFY THIS
    CHANNEL. The assignment is left completely untouched — no tier change, no
    retraction — because the neutral is usually real and corroborated elsewhere
    (C18H30O6 is clean on ``[M+H]+`` at 343.211 while its urea adduct at 403.244
    rides a 132x-brighter neighbour's sidelobe at a locked 0.71%, cv 0.033).

    A flagged channel whose neutral has no OTHER ion channel is additionally demoted
    Assigned -> Candidate (``demote_uncorroborated``): nothing but the sidelobe
    supports it. Returns counts. A no-op (columns still added, all False) without
    a TS."""
    if "intensity_suspect" not in merged.columns:
        merged["intensity_suspect"] = False
        merged["sidelobe_parent_mz"] = np.nan
    if ts_peaks is None or not len(merged) or not len(ts_peaks):
        return {"suspect": 0, "checked": 0, "demoted": 0}
    need = {"mz", "height", "sample_item_id"}
    if not need <= set(ts_peaks.columns):
        return {"suspect": 0, "checked": 0, "demoted": 0}

    ts = ts_peaks[["sample_item_id", "mz", "height"]].dropna().sort_values("mz")
    tmz = ts["mz"].to_numpy(dtype=float)
    # saturating parents: m/z bins whose MEDIAN height clears min_parent
    med = ts.assign(_r=ts["mz"].round(3)).groupby("_r")["height"].median()
    parents = med[med >= min_parent]
    if not len(parents):
        return {"suspect": 0, "checked": 0, "demoted": 0}
    pmz = parents.index.to_numpy(dtype=float)

    n_sus = n_chk = 0
    for i in merged.index:
        mz = merged.at[i, "mz"]
        if pd.isna(mz) or pd.isna(merged.at[i, "neutral_formula"]):
            continue
        i0, i1 = np.searchsorted(tmz, [mz - 2.5e-3, mz + 2.5e-3])
        if i1 - i0 < min_pairs:
            continue
        w = ts.iloc[i0:i1].sort_values("mz")
        off = w["mz"].to_numpy() - mz
        lab = np.r_[0, np.cumsum(np.diff(off) > track_gap)]
        w = w.assign(_t=lab)
        # Evaluate EVERY substantial raw track, not just the most-sampled one: the
        # contaminated track is often not the biggest. C18H30O6's dominant track
        # (n=475, cv 0.106) hides a second one (n=288, cv 0.033) that is plainly
        # locked to the neighbour -- and the exported trace is a mix of both, so
        # the channel is unreliable if ANY track carrying a real share of the
        # samples is locked.
        tracks = [g.drop_duplicates("sample_item_id") for _, g in w.groupby("_t")]
        n_ion = max((len(g) for g in tracks), default=0)
        if n_ion < min_pairs:
            continue
        cand_par = pmz[(np.abs(pmz - mz) < dmz) & (np.abs(pmz - mz) > 1.5e-3)]
        if not len(cand_par):
            continue
        checked = False
        for g in tracks:
            if len(g) < max(min_pairs, min_frac * n_ion):
                continue
            h_med = float(g["height"].median())
            if h_med <= 0:
                continue
            par, par_h = None, 0.0
            for pm in cand_par:
                ph = float(parents.loc[pm])
                if ph >= factor * h_med and ph > par_h:
                    par, par_h = pm, ph
            if par is None:
                continue
            checked = True
            j0, j1 = np.searchsorted(tmz, [par - 1.2e-3, par + 1.2e-3])
            if j1 <= j0:
                continue
            ptr = ts.iloc[j0:j1].groupby("sample_item_id")["height"].max()
            gg = g.set_index("sample_item_id")["height"]
            k = gg.index.intersection(ptr.index)
            if len(k) < min_pairs:
                continue
            ratio = (gg[k] / ptr[k]).replace([np.inf, -np.inf], np.nan).dropna()
            if len(ratio) < min_pairs or ratio.mean() <= 0:
                continue
            cv = float(ratio.std() / ratio.mean())
            if cv < cv_max:
                merged.at[i, "intensity_suspect"] = True
                merged.at[i, "sidelobe_parent_mz"] = float(par)
                n_sus += 1
                break
        n_chk += int(checked)

    # A flagged channel whose neutral has NO other ion channel rests entirely on a
    # peak now shown to be the neighbour's sidelobe -- there is no evidence left for
    # the compound, so the ASSIGNMENT (not just its intensity) is unsupported.
    # Demoted, never deleted, per the ledger's no-drop rule. A corroborated neutral
    # (C18H30O6 is clean on [M+H]+ at 343.211) keeps its tier; only this channel's
    # intensity is in doubt.
    n_dem = 0
    if demote_uncorroborated and "tier" in merged.columns and n_sus:
        by_neutral = merged.dropna(subset=["neutral_formula"]).groupby("neutral_formula")
        n_ch = by_neutral["adduct"].nunique()
        for i in merged.index[merged["intensity_suspect"].fillna(False)]:
            nf = merged.at[i, "neutral_formula"]
            if int(n_ch.get(nf, 1)) > 1:
                continue                       # corroborated elsewhere -> formula stands
            if str(merged.at[i, "tier"]) == "Assigned":
                merged.at[i, "tier"] = "Candidate"
                n_dem += 1
        if n_dem:
            log(f"[timeseries] {n_dem} of them had no other ion channel -> the "
                f"assignment rests only on the sidelobe; demoted to Candidate")
    return {"suspect": n_sus, "checked": n_chk, "demoted": n_dem}
    if n_sus:
        log(f"[timeseries] {n_sus} ion channel(s) of {n_chk} checked are "
            f"sidelobe-contaminated (locked to a >={factor:.0f}x neighbour, "
            f"ratio-cv < {cv_max}) -- formula kept, intensity_suspect=True")
    return {"suspect": n_sus, "checked": n_chk}


def _consensus_offsets(n_ions: int, ion: np.ndarray, signed: np.ndarray,
                       height: np.ndarray, halfwin: np.ndarray, *,
                       anchor_margin: float = ANCHOR_MARGIN) -> np.ndarray:
    """Per-ion systematic raw-vs-ledger m/z offset, returned as an array indexed by
    ledger-ion position (0.0 for ions with no candidate).

    An ion's candidates are sometimes split into two distinct raw TRACKS either side
    of the ledger mass (measured on the Wind-zone-2 uronium batch: C18H30O6
    [M+(CH4N2O)H]+ has a track at -1.2 mDa in 475 samples and another at +0.45 mDa
    in 288, with an empty 1.0 mDa gap between them). A mean or median lands in that
    empty gap and belongs to neither, so the tracks are separated explicitly --
    single-linkage on the sorted offsets, split wherever the gap exceeds the ion's
    ``halfwin`` -- and one track is chosen.

    Two rules pick it, both learned from real failures:

    * **Score a track by its BRIGHTEST member, not its summed height.** Summing
      conflates brightness with prevalence, and FT ringing sidelobes of a bright
      neighbour are ubiquitous-but-dim: they recur beside the parent in every
      sample. On Wind-zone-2 that let the sidelobe track of C12H19NO6 [M+H]+
      (1576 cps in 559 samples, flagged `role=artifact` by the assignment's own
      cleanup as the sidelobe of a 601146-cps peak 5 mDa away) outvote the real
      track (2390 cps in 70). A track's claim rests on how bright it gets.
    * **Anchor to the ledger mass.** Offset 0 is not an arbitrary point -- it is
      where the ASSIGNMENT committed the formula, oracle-scored across the assigned
      files. The track holding it is displaced only by one at least
      ``anchor_margin`` x brighter. Where no candidate sits within ``halfwin`` of
      the ledger mass there is nothing to anchor to and the brightest track wins
      outright (the C18H30O6 case this whole stage exists for).
    """
    out = np.zeros(n_ions, dtype=float)
    if not len(ion):
        return out
    order = np.lexsort((signed, ion))
    i_s, o_s, h_s = ion[order], signed[order], height[order]
    h_s = np.clip(np.nan_to_num(h_s, nan=0.0), 0.0, None)
    starts = np.flatnonzero(np.r_[True, i_s[1:] != i_s[:-1]])
    ends = np.r_[starts[1:], len(i_s)]
    for s, e in zip(starts, ends):
        o, h = o_s[s:e], h_s[s:e]
        w = halfwin[i_s[s]]
        if len(o) == 1:
            out[i_s[s]] = o[0]
            continue
        # split into tracks: a gap wider than halfwin starts a new one
        tstart = np.r_[0, np.flatnonzero(np.diff(o) > w) + 1]
        bright = np.maximum.reduceat(h, tstart)           # per-track peak height
        wsum = np.add.reduceat(h, tstart)
        wmean = np.divide(np.add.reduceat(o * h, tstart), wsum,
                          out=np.zeros(len(tstart)), where=wsum > 0)
        # tracks with no weight at all fall back to their plain mean offset
        tend = np.r_[tstart[1:], len(o)]
        for t in np.flatnonzero(wsum <= 0):
            wmean[t] = float(np.mean(o[tstart[t]:tend[t]]))
        b = int(np.argmax(bright))
        # the track holding the ledger mass (offset 0), if any candidate is near it
        j = int(np.argmin(np.abs(o)))
        if abs(o[j]) <= w:
            c0 = int(np.searchsorted(tstart, j, side="right") - 1)
            if b != c0 and bright[b] <= anchor_margin * bright[c0]:
                b = c0
        out[i_s[s]] = float(wmean[b])
    return out


def _resolve_one_to_one(peaks: pd.DataFrame, ok: np.ndarray, near: np.ndarray,
                        signed: np.ndarray, n_ions: int, halfwin: np.ndarray,
                        sample_col: str, height_col: str, *, consensus: bool = True
                        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reduce the many-to-one mass match to ONE peak per (sample, ledger ion).

    Returns ``(winner, loser)`` boolean masks over ``peaks`` rows. The winner is the
    candidate closest to the ion's CONSENSUS m/z (the ledger mass plus that ion's
    measured systematic offset, see ``_consensus_offsets``); ties break to the
    BRIGHTER peak, then to the lowest row index (deterministic -> byte-reproducible).

    Ranking against the consensus rather than the bare ledger mass is what makes the
    choice consistent ACROSS the parquet. The raw ts peaks of an ion sit at a small
    systematic offset from its calibrated ledger mass, and when a second (shoulder /
    split) track exists on the other side, "nearest the ledger mass" flips between
    the two tracks sample by sample -- whichever happens to be present -- splicing
    two physically different peaks into one time series. Measured on the Wind-zone-2
    uronium batch before this change: C18H30O6 [M+(CH4N2O)H]+ flipped 232 times and
    C19H34O6Si [M+NH4]+ 378 times between their two tracks.

    With no ``sample_col`` the whole table is treated as ONE spectrum (one winner
    per ion overall)."""
    pos = np.flatnonzero(ok)
    if sample_col in peaks.columns:
        samp = pd.factorize(peaks[sample_col].to_numpy(), use_na_sentinel=False)[0][pos]
    else:
        samp = np.zeros(len(pos), dtype=np.int64)
    ion = near[pos]
    off = signed[pos]
    if height_col in peaks.columns:
        h = pd.to_numeric(peaks[height_col], errors="coerce").to_numpy(dtype=float)[pos]
        h = np.nan_to_num(h, nan=-np.inf)
    else:
        h = np.zeros(len(pos), dtype=float)
    cons = (_consensus_offsets(n_ions, ion, off, h, halfwin) if consensus
            else np.zeros(n_ions, dtype=float))
    d = np.abs(off - cons[ion])          # distance from the ion's consensus m/z
    # sort within each (sample, ion) group: nearest first, then brightest, then
    # first-seen. np.lexsort applies the LAST key as primary.
    order = np.lexsort((pos, -h, d, ion, samp))
    s_s, i_s = samp[order], ion[order]
    first = np.empty(len(order), dtype=bool)
    first[0] = True
    first[1:] = (s_s[1:] != s_s[:-1]) | (i_s[1:] != i_s[:-1])
    winner = np.zeros(len(peaks), dtype=bool)
    loser = np.zeros(len(peaks), dtype=bool)
    winner[pos[order[first]]] = True
    loser[pos[order[~first]]] = True
    return winner, loser, cons


# --- trace-level reconciliation of the merged ledger ---------------------------
# The merged ledger's m/z is an ANCHOR minted from the few assigned samples. On a
# TOF the assignment snaps that anchor to theory (a formula is only committed
# where a sample's draw lands near it): real anchors sit 1.8 ppm (sd) from the
# theoretical mass while a same-size anchor drawn at random from the ion's own
# trace sits 5.7 ppm away -- the trace genuinely lives ~5.8 ppm off theory, and a
# window centred on the anchor misses most of it (22.6 % of one TOF batch's
# anchors were outside their own trace's +-6 ppm window; C10H16O9 [M+NO3]- kept
# 40 % of its spectra from the anchor and 81 % from the trace centre). Two winner
# samples also mint two competing rows for ONE ion when their draws differ by
# more than the merge tolerance (26.8 % of well-populated TOF rows shared a trace).
# Re-centring every anchor on its own trace lifted mean coverage 61 -> 74 % over
# 867 TOF ions, gains > 5 pp on 41 %, losses on 3.5 %; the same pass moved an
# Orbitrap ledger by 0.11 ppm and changed nothing -- the no-op that validates it.
# So, between the merge and the stamp: re-centre (batch.traces.PeakIndex.mean_shift
# from each anchor), collapse the rows that converge on one trace, and stamp from
# the trace centres with a window sized to the batch's own per-ion scatter.
RECENTRE_MAX_DRIFT_PPM = 10.0   # an anchor may move at most this far (measured: the
                                # median move is 2.7 ppm; +-10 ppm delivers +11.6 of the
                                # +12.7 pp total and the uncorroborated landings live beyond)
RECENTRE_GUARD_COV = 0.10       # an anchor covering < this share of spectra ...
RECENTRE_GUARD_PPM = 6.0        # ... may not move further than this uncorroborated
                                # (one such row re-centred to 84 % coverage on peaks
                                # that tracked nothing, r 0.19)
STAMP_TOL_SIGMA = 2.5           # stamping half-window = this many per-ion sigmas ...
STAMP_TOL_MAX_X = 2.0           # ... never wider than this x the merge tolerance
TRACE_WINNER, TRACE_COLLAPSED, TRACE_SINGLE = "winner", "collapsed", "single"


def _trace_index(ts_peaks, index, tol_ppm):
    from peaky.batch import traces as TR
    if index is not None:
        return index
    if ts_peaks is None or not len(ts_peaks):
        return None
    return TR.PeakIndex(ts_peaks, tol_ppm=tol_ppm)


def recentre_ledger(merged: pd.DataFrame, ts_peaks: pd.DataFrame | None = None, *,
                    index=None, tol_ppm: float = DEFAULT_TOL_PPM,
                    max_drift_ppm: float = RECENTRE_MAX_DRIFT_PPM,
                    guard_cov: float = RECENTRE_GUARD_COV,
                    guard_ppm: float = RECENTRE_GUARD_PPM, log=print) -> dict:
    """Re-centre every merged row on its own TRACE (in place). Adds

      mz_anchor         the merge's m/z (what the assignment committed)
      mz_trace          the trace centre the stamp should use (= mz_anchor when
                        the row did not move)
      trace_offset_ppm  mz_trace vs mz_anchor
      trace_cov_anchor  share of spectra with a peak within tol of the anchor
      trace_cov         ... of the trace centre
      trace_moved       the row moved (the trace centre covers strictly more)
      trace_guarded     the move was refused by the low-evidence guard

    A row moves only where the re-centred window covers MORE spectra than the
    anchor's (never fewer), by at most `max_drift_ppm`. Guard: an anchor covering
    < `guard_cov` of the spectra that wants to move > `guard_ppm` must be
    corroborated (seen in >= 2 assigned files, or Assigned) -- an almost-empty
    anchor plus a large jump is how a label lands on a neighbour's trace. Without
    a time series this is a no-op that still adds the columns."""
    idx = _trace_index(ts_peaks, index, tol_ppm)
    n = len(merged)
    mz = pd.to_numeric(merged["mz"], errors="coerce").to_numpy(dtype=float) if "mz" in merged.columns \
        else np.full(n, np.nan)
    merged["mz_anchor"] = mz
    merged["mz_trace"] = mz
    merged["trace_offset_ppm"] = 0.0
    merged["trace_cov_anchor"] = np.nan
    merged["trace_cov"] = np.nan
    merged["trace_moved"] = False
    merged["trace_guarded"] = False
    out = {"n_rows": int(n), "n_recentred": 0, "n_guarded": 0,
           "median_abs_move_ppm": 0.0, "mean_cov_anchor": None, "mean_cov_trace": None,
           "max_drift_ppm": float(max_drift_ppm), "tol_ppm": float(tol_ppm)}
    if idx is None or n == 0 or len(idx) == 0:
        return out
    nf = pd.to_numeric(merged.get("n_files", pd.Series(1, index=merged.index)),
                       errors="coerce").fillna(1).to_numpy()
    tier = merged["tier"].astype(str).to_numpy() if "tier" in merged.columns \
        else np.full(n, "", dtype=object)
    cov_a = np.full(n, np.nan); cov_t = np.full(n, np.nan)
    mzt = mz.copy(); moved = np.zeros(n, dtype=bool); guarded = np.zeros(n, dtype=bool)
    for i in range(n):
        a = mz[i]
        if not np.isfinite(a):
            continue
        cov_a[i] = idx.coverage_at(a, tol_ppm)
        c = idx.mean_shift(a, tol_ppm=tol_ppm, max_drift_ppm=max_drift_ppm)
        cov_c = idx.coverage_at(c, tol_ppm)
        move = abs(c - a) / a * 1e6
        if cov_c > cov_a[i] and move > 0:
            corroborated = (nf[i] >= 2) or (tier[i] == "Assigned")
            if cov_a[i] < guard_cov and move > guard_ppm and not corroborated:
                guarded[i] = True
                cov_t[i] = cov_a[i]
                continue
            mzt[i] = c; cov_t[i] = cov_c; moved[i] = True
        else:
            cov_t[i] = cov_a[i]
    merged["mz_trace"] = mzt
    merged["trace_offset_ppm"] = np.where(np.isfinite(mz) & (mz > 0), (mzt - mz) / mz * 1e6, 0.0)
    merged["trace_cov_anchor"] = cov_a
    merged["trace_cov"] = cov_t
    merged["trace_moved"] = moved
    merged["trace_guarded"] = guarded
    ok = np.isfinite(cov_a)
    out.update(n_recentred=int(moved.sum()), n_guarded=int(guarded.sum()),
               median_abs_move_ppm=round(float(np.median(np.abs(merged.loc[moved, "trace_offset_ppm"])))
                                         if moved.any() else 0.0, 3),
               mean_cov_anchor=round(float(np.nanmean(cov_a)), 4) if ok.any() else None,
               mean_cov_trace=round(float(np.nanmean(cov_t)), 4) if ok.any() else None)
    if log:
        log(f"[traces] re-centred {out['n_recentred']} of {n} ledger rows on their own "
            f"trace (median move {out['median_abs_move_ppm']} ppm, cap {max_drift_ppm:g}); "
            f"{out['n_guarded']} low-evidence moves refused; mean spectra coverage "
            f"{out['mean_cov_anchor']} -> {out['mean_cov_trace']}")
    return out


_TIER_RANK_TRACE = {"Assigned": 2, "Candidate": 1}


def collapse_trace_labels(merged: pd.DataFrame, *, tol_ppm: float = DEFAULT_TOL_PPM,
                          log=print) -> dict:
    """Rows whose trace centres fall within `tol_ppm` of each other are competing
    labels for ONE ion (in place). Adds `trace_id` (gap-cluster of `mz_trace`) and
    `trace_role`: 'single' (alone on its trace), 'winner' (the label kept), or
    'collapsed' (a competing label the stamp must not use). The winner is chosen
    by the MERGE's own ordering -- most assigned files, then tier, then ion_score
    -- with proximity to the trace centre only as a deterministic tie-break
    before the m/z itself. Never ion_score FIRST: per-file, it is blind to
    reproducibility, and on a TOF batch it handed a 1-file C14H17NO4S [M-H]-
    (0.878) the trace of the Orbitrap-confirmed C10H16O6 [M+NO3]- seen in 6 files
    (0.832). And never proximity BEFORE the score: on a TOF the anchors were
    snapped to theory by the assignment, so which label's anchor sits nearer the
    trace centre says nothing about which label is right -- on a live batch it
    handed the trace of the known monomer C10H16O9 to a 1-file C15H21NO3 [M+Br]-
    whose anchor happened to land 0.25 ppm from the centre (score 0.846 vs 0.960);
    the score ordering picked the reference-list label in 4 of the 4 same-file,
    same-tier ties where exactly one label was on the list. Nothing is dropped:
    collapsed rows stay in the ledger, flagged, exactly like `dup_candidate`
    peaks."""
    n = len(merged)
    merged["trace_id"] = -1
    merged["trace_role"] = TRACE_SINGLE
    out = {"n_traces": 0, "n_collapsed": 0, "n_multi_label_traces": 0}
    if n == 0 or "mz_trace" not in merged.columns:
        return out
    mzt = pd.to_numeric(merged["mz_trace"], errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(mzt)
    if not ok.any():
        return out
    pos = np.flatnonzero(ok)
    order = pos[np.argsort(mzt[pos], kind="mergesort")]
    srt = mzt[order]
    tid = np.zeros(len(order), dtype=np.int64)
    if len(order) > 1:
        gaps = np.diff(srt) / srt[:-1] * 1e6
        tid[1:] = np.cumsum(gaps > tol_ppm)
    trace_id = np.full(n, -1, dtype=np.int64)
    trace_id[order] = tid
    merged["trace_id"] = trace_id
    role = np.full(n, TRACE_SINGLE, dtype=object)
    nf = pd.to_numeric(merged.get("n_files", pd.Series(1, index=merged.index)),
                       errors="coerce").fillna(0).to_numpy()
    tr = merged["tier"].astype(str).map(_TIER_RANK_TRACE).fillna(0).to_numpy() \
        if "tier" in merged.columns else np.zeros(n)
    off = np.abs(pd.to_numeric(merged.get("trace_offset_ppm", pd.Series(0.0, index=merged.index)),
                               errors="coerce").fillna(0).to_numpy())
    sc = pd.to_numeric(merged.get("ion_score", pd.Series(0.0, index=merged.index)),
                       errors="coerce").fillna(0).to_numpy()
    multi = 0
    for t in np.unique(tid):
        rows = order[tid == t]
        if len(rows) < 2:
            continue
        multi += 1
        # lexsort: last key is primary -- n_files, tier, ion_score, then proximity
        # to the trace centre, then the row's m/z (deterministic).
        k = np.lexsort((mzt[rows], off[rows], -sc[rows], -tr[rows], -nf[rows]))
        win = rows[k[0]]
        role[rows] = TRACE_COLLAPSED
        role[win] = TRACE_WINNER
    merged["trace_role"] = role
    out.update(n_traces=int(len(np.unique(tid))), n_collapsed=int((role == TRACE_COLLAPSED).sum()),
               n_multi_label_traces=int(multi))
    if log and out["n_collapsed"]:
        log(f"[traces] {out['n_collapsed']} competing label(s) collapsed: {n} ledger rows "
            f"sit on {out['n_traces']} traces ({multi} traces carried more than one label)")
    return out


def stamp_tolerance(index, centres, *, tol_ppm: float = DEFAULT_TOL_PPM,
                    k_sigma: float = STAMP_TOL_SIGMA, max_x: float = STAMP_TOL_MAX_X) -> tuple[float, float]:
    """The stamping half-window for this batch, from its own per-ion mass scatter:
    max(tol_ppm, min(max_x * tol_ppm, k_sigma * sigma)), where sigma is the third
    quartile of the robust per-trace scatter at `centres` (batch.traces.
    batch_scatter_ppm -- the dimmer traces, which a window sized from the bright
    ones loses). Returns (stamp_tol_ppm, sigma_ppm); sigma NaN (and the window =
    tol_ppm) when no trace is populated enough to measure. An Orbitrap (third
    quartile 0.24-0.34 ppm) keeps the merge tolerance; a TOF (3.8-4.2 ppm) gets a
    ~10 ppm window, near the +-12 ppm membership cap the trace-batching bake-off
    settled on -- the per-ion cloud is wider than the merge tolerance there, and
    a window that cuts through it loses real spectra to the one-to-one contest
    (measured on a 230-spectrum TOF batch: a 12 ppm window doubled the share of
    ions gaining > 5 pp of coverage over a 6 ppm one, 21 -> 42 %, with the share
    losing unchanged at 3.7 %)."""
    from peaky.batch import traces as TR
    sigma = TR.batch_scatter_ppm(index, centres, tol_ppm=tol_ppm) if index is not None else float("nan")
    if not np.isfinite(sigma):
        return float(tol_ppm), float("nan")
    return float(max(tol_ppm, min(max_x * tol_ppm, k_sigma * sigma))), round(float(sigma), 3)


def identified_rows(ledger: pd.DataFrame) -> pd.DataFrame:
    """Per-file summary of every IDENTIFIED ion in a full ledger -- analyte or
    not -- for the parquet ion-formula stamp (`stamping_frame`).

    Returns columns (mz, role, ion_formula, iso_label, neutral_formula, adduct):

      * M0        -- assigned analytes; carries neutral/adduct AND ion_formula.
      * reagent   -- reagent-cluster ions; ion_formula from the ledger (the
                     labeler records it: known formula = assigned, whatever the
                     class). iso_label = the isotopologue tag parsed from the
                     label commentary ('79Br+81Br', '127I+127I', ...) so the
                     heavy lines of one reagent formula stay distinct rows.
      * iso_child -- isotope satellites; ion_formula = the PARENT ion's formula
                     (joined via parent_peak_id), iso_label its own (13C, 81Br).
                     neutral_formula stays empty ON PURPOSE: quantification
                     sums per neutral must not silently double-count satellites.
      * artifact  -- FT ringing sidelobes: no ion (they are ghosts of a bright
                     neighbour), role only.
    """
    cols = ["mz", "role", "ion_formula", "iso_label", "neutral_formula", "adduct"]
    if ledger is None or not len(ledger) or "role" not in ledger.columns:
        return pd.DataFrame(columns=cols)
    led = ledger
    out = []
    m0 = led[led["role"] == "M0"]
    for _, r in m0.iterrows():
        out.append((r["mz"], "M0", r.get("ion_formula"), None,
                    r.get("neutral_formula"), r.get("adduct")))
    ionf_of = dict(zip(led["peak_id"], led.get("ion_formula", pd.Series(dtype=object)))) \
        if "peak_id" in led.columns else {}

    def _iso_tag(commentary) -> str | None:
        # 'reagent ion: [I2]-. (127I+127I) (-0.3 ppm)' -> '127I+127I'
        for g in re.findall(r"\(([^)]+)\)", str(commentary or "")):
            if "ppm" not in g:
                return g
        return None

    for _, r in led[led["role"] == "reagent"].iterrows():
        f = r.get("ion_formula")
        if pd.notna(f) and f:
            out.append((r["mz"], "reagent", f, _iso_tag(r.get("commentary")),
                        None, None))
    for _, r in led[led["role"] == "iso_child"].iterrows():
        pf = ionf_of.get(r.get("parent_peak_id"))
        if pd.notna(pf) and pf:
            out.append((r["mz"], "iso_child", pf, r.get("iso_label"), None, None))
    for _, r in led[led["role"] == "artifact"].iterrows():
        out.append((r["mz"], "artifact", None, None, None, None))
    return pd.DataFrame(out, columns=cols)


# --- predicted isotope satellites in the batch stamp ---------------------------
# A per-file ledger claims a satellite only where that file's picker picked it,
# and `identified_rows` carries only what was claimed. The faint diagnostic lines
# -- 15N (0.36 % per N), 18O (0.20 % per O), a single 34S / 29Si / 30Si -- sit
# below the picker's edge (~150-220 cps) in most files, so a parent Assigned in
# every assigned file still leaves its 15N / 18O tracks unexplained wherever a
# plume lifts them into view. Measured on the Texas Ur 122-600 batch (6154
# spectra, 15 assigned): C12H27O4P [M+(CH4N2O)H]+ and C12H14O [M+NH4]+ are
# Assigned in every per-file ledger with only their 13C satellite claimed; their
# 15N line (m/z 328.2013, 109 spectra, up to 1.1 kcps) and 18O line (194.1425,
# 4 spectra, up to 1.0 kcps) were picked in none of the 15 files, so the stamp
# could not explain them -- while a targeted single-sample assign of a plume
# file claimed both (the per-file logic is right; only the batch stamp was
# blind). So the stamp PREDICTS every M0's diagnostic lines from its ion
# formula, at the parent's stamped m/z (its trace centre, so the instrument
# offset carries over) plus the line's exact shift, and `annotate_peaks` stamps
# them under the same intensity-consistency gate the per-file envelope passes
# use: a TS peak takes a predicted label only where the parent's track has a
# peak in the SAME sample and height_child / (height_parent * rel) sits in the
# window. The per-file ledgers are never touched -- this is the batch stamp only.
PRED_SAT_LABELS = ("13C", "81Br", "37Cl", "15N", "34S", "29Si", "30Si", "18O")
    # cleanup.reclaim_satellites' diagnostic set (its DELT table)
PRED_SAT_MIN_REL = 0.001        # postprocess.complete_isotope_envelopes' diag_min_rel
PRED_SAT_MAX_SHIFT = 2.5        # every line in the set is an M+1 / M+2
PRED_SAT_MERGE_DA = 0.0005      # keep each substitution its own line (_predicted_lines)
PRED_RATIO_MIN, PRED_RATIO_MAX = 0.3, 3.5
    # the height_child / (height_parent * rel) window under which
    # postprocess.complete_isotope_envelopes attaches an unexplained satellite
PRED_TRACK_MIN_N, PRED_TRACK_MIN_SHARE = 10, 0.5
    # track coherence: once a predicted line has been judged in >= MIN_N samples
    # (parent present, a candidate on the line) it keeps its stamps only if
    # >= MIN_SHARE of them passed the window (_predicted_track_coherence)
STAMP_M0, STAMP_OBSERVED, STAMP_PREDICTED = "M0", "observed", "predicted"


@functools.lru_cache(maxsize=8192)
def _predicted_lines(ion_formula: str) -> tuple:
    """((delta_mass, rel, label), ...) of an ION's diagnostic satellite lines --
    PRED_SAT_LABELS only, one line per label, each at its EXACT shift.

    The envelope is chem.isotopes.isotope_pattern's, but merged at 0.5 mDa
    instead of its 6 mDa default: the default folds the 18O line (+2.0042) into
    13C2 (+2.0067) on any carbon-rich ion and labels the centroid 13C2 -- outside
    the diagnostic set, and 2.5 mDa (13 ppm at m/z 194) from the 18O peak an
    Orbitrap resolves, so the C12H18NO+ 18O track this stamp exists to explain
    would be missed. Each diagnostic label is then read off the pattern line
    nearest its own shift (within 0.6 mDa): the m/z is the exact substitution,
    the rel is the pattern's (atom-count aware, cross terms included). Two
    labels landing on one unresolvable line (37Cl / 30Si, 0.2 mDa apart) keep
    the one with the larger per-atom expectation. Cached: a batch stamps
    thousands of parents, many alike."""
    from peaky.chem import chemistry as C
    from peaky.chem import isotopes as ISO
    counts = C.parse_formula(ion_formula)
    if not counts:
        return ()
    try:
        pat = ISO.isotope_pattern(ion_formula, min_rel=PRED_SAT_MIN_REL,
                                  max_shift=PRED_SAT_MAX_SHIFT,
                                  merge_da=PRED_SAT_MERGE_DA,
                                  diag_min_rel=PRED_SAT_MIN_REL)
    except Exception:
        return ()
    if not pat:
        return ()
    table = [
        (ISO.D_13C,  "13C",  "C",  ISO.R_13C_PER_C),
        (ISO.D_81BR, "81Br", "Br", ISO.R_81BR_PER_BR),
        (ISO.D_37CL, "37Cl", "Cl", ISO.R_37CL_PER_CL),
        (ISO.D_15N,  "15N",  "N",  ISO.R_15N_PER_N),
        (ISO.D_34S,  "34S",  "S",  ISO.R_34S_PER_S),
        (ISO.D_29SI, "29Si", "Si", ISO.R_29SI_PER_SI),
        (ISO.D_30SI, "30Si", "Si", ISO.R_30SI_PER_SI),
        (ISO.D_18O,  "18O",  "O",  ISO.R_18O_PER_O),
    ]
    best: dict[int, tuple] = {}          # pattern line -> (weight, delta, rel, label)
    for d, label, el, per in table:
        nel = counts.get(el, 0)
        if nel < 1:
            continue
        j = min(range(len(pat)), key=lambda k: abs(pat[k][0] - d))
        if abs(pat[j][0] - d) > 0.0006:
            continue
        w = nel * per
        if j not in best or w > best[j][0]:
            best[j] = (w, float(d), float(pat[j][1]), label)
    return tuple(sorted((d, rel, label) for _w, d, rel, label in best.values()))


def predicted_satellite_rows(stamp: pd.DataFrame, *, tol_ppm: float = DEFAULT_TOL_PPM,
                             mz_floor_da: float = 1.5e-3) -> tuple[pd.DataFrame, dict]:
    """PREDICTED diagnostic-satellite rows for every M0 row of a stamping frame
    that carries an ion_formula: one row per (parent, label) at the parent's
    stamped m/z + the line's shift, role 'iso_child', the parent's ion_formula,
    iso_label, iso_rel (the predicted height relative to the parent) and
    stamp_source='predicted', linked to its parent by parent_stamp_id.

    Precedence -- a predicted line never sits on a track something known already
    explains:

      * an OBSERVED satellite of the same (ion_formula, iso_label) supersedes it
        (a per-file ledger claimed that line; its measured m/z stamps);
      * any row already in the frame within the stamping window of the predicted
        m/z supersedes it -- an M0 (an assigned analyte at a satellite offset IS
        an analyte), a reagent line, another parent's observed satellite, an
        artifact -- so a predicted line can never displace an assigned analyte.
        `annotate_peaks` enforces the same order at stamping time.

    Returns (rows, info); info counts parents, lines, both kinds of supersession
    and the rows kept. The frame must already carry `stamp_id` (stamping_frame
    assigns it before calling this); without it nothing is predicted."""
    cols = ["mz", "role", "ion_formula", "iso_label", "iso_rel", "stamp_source",
            "stamp_id", "parent_stamp_id"]
    info = {"n_parents": 0, "n_lines": 0, "n_superseded_observed": 0,
            "n_superseded_track": 0, "n_predicted": 0}
    if stamp is None or not len(stamp) or "ion_formula" not in stamp.columns \
            or "stamp_id" not in stamp.columns:
        return pd.DataFrame(columns=cols), info
    role = (stamp["role"].astype(str) if "role" in stamp.columns
            else pd.Series("M0", index=stamp.index))
    smz = pd.to_numeric(stamp["mz"], errors="coerce")
    m0 = stamp[(role == "M0") & stamp["ion_formula"].notna() & smz.notna()]
    observed: set = set()
    if "iso_label" in stamp.columns:
        o = stamp[(role == "iso_child") & stamp["iso_label"].notna()
                  & stamp["ion_formula"].notna()]
        observed = set(zip(o["ion_formula"].astype(str), o["iso_label"].astype(str)))
    known = np.sort(smz.dropna().to_numpy(dtype=float))
    rows = []
    for pid, pmz, f in zip(m0["stamp_id"], smz.loc[m0.index], m0["ion_formula"]):
        f = str(f).strip()
        if not f:
            continue
        lines = _predicted_lines(f)
        if not lines:
            continue
        info["n_parents"] += 1
        for d, rel, label in lines:
            info["n_lines"] += 1
            if (f, label) in observed:
                info["n_superseded_observed"] += 1
                continue
            rows.append((float(pmz) + d, f, label, rel, int(pid)))
    if not rows:
        return pd.DataFrame(columns=cols), info
    pred = pd.DataFrame(rows, columns=["mz", "ion_formula", "iso_label", "iso_rel",
                                       "parent_stamp_id"])
    if len(known):
        pm = pred["mz"].to_numpy(dtype=float)
        j = np.searchsorted(known, pm)
        jl = np.clip(j - 1, 0, len(known) - 1)
        jr = np.clip(j, 0, len(known) - 1)
        near = np.minimum(np.abs(known[jl] - pm), np.abs(known[jr] - pm))
        taken = near <= np.maximum(pm * tol_ppm * 1e-6, mz_floor_da)
        info["n_superseded_track"] = int(taken.sum())
        pred = pred[~taken].reset_index(drop=True)
    info["n_predicted"] = int(len(pred))
    pred["role"] = "iso_child"
    pred["stamp_source"] = STAMP_PREDICTED
    pred["stamp_id"] = np.arange(len(pred)) + int(stamp["stamp_id"].max()) + 1
    return pred[cols], info


def stamping_frame(merged: pd.DataFrame,
                   identified: pd.DataFrame | None, *,
                   tol_ppm: float = DEFAULT_TOL_PPM, mz_floor_da: float = 1.5e-3,
                   predict_satellites: bool = True) -> pd.DataFrame:
    """Union frame for `annotate_peaks`: the merged ANALYTE ledger plus one row
    per identified NON-analyte ion, so the parquet stamp distinguishes
    'identified non-analyte' (reagent ladder, isotope satellites, artifacts)
    from 'unknown'. `identified` is the concat of `identified_rows()` over the
    per-file ledgers (None/empty -> analytes only, with role/ion_formula/
    iso_label stamped on them).

    Analyte rows keep every merged column and gain role='M0' + the modal
    per-file ion_formula for their (neutral_formula, adduct) key. Non-analyte
    rows are aggregated across files: reagent / iso_child by (ion_formula,
    iso_label) at the median m/z; artifacts (no formula key) by m/z gap
    clustering (>3 mDa starts a new track).

    Every row carries `stamp_source`: 'M0' (a merged analyte), 'observed' (an
    ion a per-file ledger identified) or 'predicted' -- the diagnostic isotope
    satellites (`PRED_SAT_LABELS`) of every M0 with a known ion_formula that no
    per-file ledger claimed, added by `predicted_satellite_rows` (which see for
    the precedence: observed beats predicted, and a predicted line is dropped
    from any track an M0 / reagent / observed satellite / artifact already
    holds, within `tol_ppm` -- pass the STAMPING window, the one `annotate_peaks`
    will use). `stamp_id` / `parent_stamp_id` link a predicted row to its parent
    for the intensity gate; `iso_rel` is its predicted height relative to the
    parent. `predict_satellites=False` restores the observed-only stamp. The
    frame's `.attrs['predicted_satellites']` carries the prediction counts."""
    stamp = merged.copy()
    # trace-level reconciliation (recentre_ledger / collapse_trace_labels): stamp
    # from each ion's TRACE CENTRE, not the merge anchor, and never from a
    # collapsed competing label -- its trace already has a winner.
    if "trace_role" in stamp.columns:
        stamp = stamp[stamp["trace_role"].astype(str) != TRACE_COLLAPSED].copy()
    if "mz_trace" in stamp.columns:
        mzt = pd.to_numeric(stamp["mz_trace"], errors="coerce")
        stamp["mz"] = mzt.where(mzt.notna(), stamp["mz"])
    stamp["role"] = "M0"
    if "ion_formula" not in stamp.columns:
        stamp["ion_formula"] = None
    stamp["iso_label"] = None
    stamp["stamp_source"] = STAMP_M0
    if identified is not None and len(identified):
        idf = identified
        # modal per-file ion_formula onto the merged analyte rows
        m0 = idf[(idf["role"] == "M0") & idf["ion_formula"].notna()]
        if len(m0):
            mode = (m0.groupby(["neutral_formula", "adduct"])["ion_formula"]
                      .agg(lambda s: s.mode().iloc[0]))
            key = list(zip(stamp["neutral_formula"], stamp["adduct"]))
            stamp["ion_formula"] = [
                mode.get(k) if pd.isna(v) else v
                for k, v in zip(key, stamp["ion_formula"])
            ]
        aux = []
        for (f, tag), grp in idf[idf["role"].isin(("reagent", "iso_child"))].groupby(
                ["ion_formula", "iso_label"], dropna=False):
            role = grp["role"].iloc[0]
            aux.append({"mz": float(grp["mz"].median()), "role": role,
                        "ion_formula": f,
                        "iso_label": None if pd.isna(tag) else tag,
                        "stamp_source": STAMP_OBSERVED})
        art = idf.loc[idf["role"] == "artifact", "mz"].dropna().sort_values()
        if len(art):
            start = 0
            vals = art.to_numpy()
            for i in range(1, len(vals) + 1):
                if i == len(vals) or vals[i] - vals[i - 1] > 3e-3:
                    aux.append({"mz": float(np.median(vals[start:i])),
                                "role": "artifact", "ion_formula": None,
                                "iso_label": None, "stamp_source": STAMP_OBSERVED})
                    start = i
        if aux:
            stamp = pd.concat([stamp, pd.DataFrame(aux)], ignore_index=True)
    stamp["stamp_id"] = np.arange(len(stamp))
    stamp["parent_stamp_id"] = -1
    stamp["iso_rel"] = np.nan
    info: dict = {}
    if predict_satellites:
        pred, info = predicted_satellite_rows(stamp, tol_ppm=tol_ppm,
                                              mz_floor_da=mz_floor_da)
        if len(pred):
            stamp = pd.concat([stamp, pred], ignore_index=True)
    stamp.attrs["predicted_satellites"] = info
    return stamp


def _nearest_ion(lmz: np.ndarray, pmz: np.ndarray, tol_ppm: float, mz_floor_da: float):
    """Nearest ledger m/z for every peak (searchsorted): (near, signed, ok) --
    ledger position, peak-minus-ledger offset, and 'within the window'."""
    j = np.searchsorted(lmz, pmz)
    jl = np.clip(j - 1, 0, len(lmz) - 1)
    jr = np.clip(j, 0, len(lmz) - 1)
    near = np.where(np.abs(lmz[jl] - pmz) <= np.abs(lmz[jr] - pmz), jl, jr)
    tol = np.maximum(pmz * tol_ppm * 1e-6, mz_floor_da)
    signed = pmz - lmz[near]                 # + == peak above ledger mass
    ok = np.isfinite(pmz) & (np.abs(signed) <= tol)
    return near, signed, ok


def _predicted_ratio_gate(peaks: pd.DataFrame, led: pd.DataFrame, near_known: np.ndarray,
                          ok_known: np.ndarray, near_pred: np.ndarray, cand: np.ndarray,
                          sample_col: str, height_col: str,
                          window: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    """The per-sample intensity-consistency gate for PREDICTED satellite
    candidates. Returns two masks over `peaks`, (passed, evaluable): `evaluable`
    is True for a candidate (mask `cand`, matched to the predicted row at ledger
    position `near_pred`) whose parent -- the row `parent_stamp_id` names -- was
    stamped on a peak in the SAME sample (a tier-1 winner: `ok_known` /
    `near_known`) with a finite ratio height_child / (height_parent * iso_rel),
    i.e. the gate could be judged; `passed` is the subset whose ratio lies
    inside `window`. Everything else is False on both: no parent peak in that
    sample, no height column, a zero / missing height. This is the same
    discriminator the per-file envelope passes rely on -- a real satellite sits
    at the predicted height, an independent compound at a satellite offset does
    not."""
    n = len(peaks)
    passed = np.zeros(n, dtype=bool)
    evaluable = np.zeros(n, dtype=bool)
    need = {"stamp_id", "parent_stamp_id", "iso_rel"}
    if not cand.any() or height_col not in peaks.columns or not need <= set(led.columns):
        return passed, evaluable
    h = pd.to_numeric(peaks[height_col], errors="coerce").to_numpy(dtype=float)
    if sample_col in peaks.columns:
        samp = pd.factorize(peaks[sample_col].to_numpy(), use_na_sentinel=False)[0]
    else:
        samp = np.zeros(n, dtype=np.int64)
    n_led = len(led)
    # the parents' stamped heights, keyed by (sample, ledger position); with
    # one_to_one off several peaks may carry one ion in a sample -> the brightest
    win = ok_known & (near_known >= 0) & np.isfinite(h)
    if not win.any():
        return passed, evaluable
    kw = samp[win].astype(np.int64) * n_led + near_known[win]
    order = np.argsort(kw, kind="stable")
    ks, hs = kw[order], h[win][order]
    uniq, start = np.unique(ks, return_index=True)
    hmax = np.maximum.reduceat(hs, start)
    # each candidate's parent: stamp_id -> ledger position (a hand-built frame
    # may repeat an id or leave it blank: first occurrence wins, blanks are no parent)
    sid = pd.to_numeric(led["stamp_id"], errors="coerce").to_numpy(dtype=float)
    keep = np.isfinite(sid)
    pos_of = pd.Series(np.arange(n_led)[keep], index=sid[keep])
    pos_of = pos_of[~pos_of.index.duplicated(keep="first")]
    c = np.flatnonzero(cand)
    par_id = pd.to_numeric(led["parent_stamp_id"], errors="coerce").to_numpy(dtype=float)[near_pred[c]]
    ppos = pos_of.reindex(par_id).to_numpy(dtype=float)
    has_parent = np.isfinite(ppos)
    ppos_i = np.where(has_parent, ppos, 0).astype(np.int64)
    key = samp[c].astype(np.int64) * n_led + ppos_i
    j = np.searchsorted(uniq, key)
    jj = np.minimum(j, len(uniq) - 1)
    found = has_parent & (j < len(uniq)) & (uniq[jj] == key)
    hp = np.where(found, hmax[jj], np.nan)
    rel = pd.to_numeric(led["iso_rel"], errors="coerce").to_numpy(dtype=float)[near_pred[c]]
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = h[c] / (hp * rel)
    lo, hi = window
    ev = found & np.isfinite(ratio)
    good = ev & (ratio >= lo) & (ratio <= hi)
    evaluable[c[ev]] = True
    passed[c[good]] = True
    return passed, evaluable


def _predicted_track_coherence(peaks: pd.DataFrame, near_pred: np.ndarray, passed: np.ndarray,
                               evaluable: np.ndarray, n_led: int, sample_col: str,
                               min_n: int, min_share: float
                               ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The TRACK rule behind the per-sample gate, tallied per predicted ledger
    row over SAMPLES (a shoulder beside the real peak is not a second vote):
    n_eval = samples in which the parent was stamped and a candidate sat on the
    line (the gate could be judged), n_pass = those in which a candidate passed
    it. A track is JUDGED once n_eval >= min_n and KEPT when
    n_pass >= min_share * n_eval; an unjudged track is kept on the per-sample
    gate alone. min_n <= 0 or min_share <= 0 switches the rule off.

    Why: a true satellite's height ratio is a constant of nature, so it passes
    the per-sample window in (nearly) every sample where both peaks are seen;
    an independent compound sitting on the line fails it in most samples and
    passes in the few where its own height happens to dip into the window --
    and those few would be mislabelled. The per-sample window cannot tell the
    two apart; the batch can (measured on two live runs: 7 and 8 such tracks,
    passing 1-26 % of their judged samples, every one with a median ratio
    outside 0.5-2 -- 24 and 268 stamps that a per-sample gate alone hands out).

    Returns (n_eval, n_pass, judged, kept), each indexed by ledger position."""
    n = len(peaks)
    if sample_col in peaks.columns:
        samp = pd.factorize(peaks[sample_col].to_numpy(), use_na_sentinel=False)[0]
    else:
        samp = np.zeros(n, dtype=np.int64)

    def _samples_per_row(mask: np.ndarray) -> np.ndarray:
        if not mask.any():
            return np.zeros(n_led, dtype=np.int64)
        key = np.unique(samp[mask].astype(np.int64) * n_led + near_pred[mask])
        return np.bincount(key % n_led, minlength=n_led)

    n_eval = _samples_per_row(evaluable)
    n_pass = _samples_per_row(passed)
    if min_n > 0 and min_share > 0:
        judged = n_eval >= int(min_n)
    else:
        judged = np.zeros(n_led, dtype=bool)
    kept = ~judged | (n_pass >= float(min_share) * n_eval)
    return n_eval, n_pass, judged, kept


PRED_TRACK_COLS = ["stamp_id", "ion_formula", "iso_label", "ion_mz", "iso_rel", "n_candidates",
                   "n_eval", "n_pass", "pass_share", "judged", "kept", "n_stamped"]


def _predicted_track_table(led: pd.DataFrame, pos_pred: np.ndarray, near_pred: np.ndarray,
                           cand: np.ndarray, stamped: np.ndarray, n_eval: np.ndarray,
                           n_pass: np.ndarray, judged: np.ndarray, kept: np.ndarray) -> pd.DataFrame:
    """One row per predicted line that had at least one candidate peak: the
    audit behind `annotate_peaks(..., stats=)` (PRED_TRACK_COLS; `n_candidates`
    counts peaks in the window in any sample, `n_eval` / `n_pass` count SAMPLES,
    `n_stamped` the peaks that carry the label after one-to-one)."""
    n_led = len(led)
    n_cand = np.bincount(near_pred[cand], minlength=n_led)
    n_st = np.bincount(near_pred[stamped], minlength=n_led)
    rows = pos_pred[n_cand[pos_pred] > 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        share = np.where(n_eval[rows] > 0, n_pass[rows] / np.maximum(n_eval[rows], 1), np.nan)

    def _col(name):
        return (led[name].to_numpy()[rows] if name in led.columns
                else np.full(len(rows), None, dtype=object))
    return pd.DataFrame({
        "stamp_id": _col("stamp_id"), "ion_formula": _col("ion_formula"),
        "iso_label": _col("iso_label"), "ion_mz": led["mz"].to_numpy(dtype=float)[rows],
        "iso_rel": _col("iso_rel"), "n_candidates": n_cand[rows], "n_eval": n_eval[rows],
        "n_pass": n_pass[rows], "pass_share": share, "judged": judged[rows], "kept": kept[rows],
        "n_stamped": n_st[rows],
    })[PRED_TRACK_COLS]


def annotate_peaks(peaks: pd.DataFrame, ledger: pd.DataFrame, *,
                   tol_ppm: float = DEFAULT_TOL_PPM, mz_floor_da: float = 1.5e-3,
                   mz_col: str = "mz", sample_col: str = "sample_item_id",
                   height_col: str = "height",
                   one_to_one: bool = True, consensus: bool = True,
                   pred_ratio: tuple[float, float] = (PRED_RATIO_MIN, PRED_RATIO_MAX),
                   pred_track_min_n: int = PRED_TRACK_MIN_N,
                   pred_track_min_share: float = PRED_TRACK_MIN_SHARE,
                   stats: dict | None = None) -> pd.DataFrame:
    """Stamp every time-series peak with the assigned formula/channel of the nearest
    ledger ion within tolerance. Returns a COPY of ``peaks`` with these added columns:

      * ``neutral_formula`` -- the assigned neutral formula (or <NA> if unmatched)
      * ``adduct``          -- the ionisation channel (e.g. ``[M+H]+`` / ``[M+NH4]+``)
      * ``tier``            -- the ledger tier (Assigned / Candidate / ...)
      * ``ion_mz``          -- the matched ledger ion m/z (NaN if unmatched); when
                               the ledger was trace-reconciled this is the ion's
                               TRACE CENTRE (`mz_trace`), and its merge anchor is
                               the ledger's `mz_anchor`
      * ``dup_candidate``   -- True for a peak that fell inside an ion's window but
                               LOST the one-to-one contest (its four columns above
                               stay <NA>); an audit trail, never a second stamp
      * ``intensity_suspect`` -- carried from the ledger's own column
                               (`flag_sidelobe_channels`): the formula is trusted
                               but this channel's HEIGHT is a bright neighbour's
                               ringing sidelobe — exclude it from quantification
      * ``role`` / ``ion_formula`` / ``iso_label`` -- identity for EVERY known
                               ion, analyte or not, when the ledger frame carries
                               them (see `stamping_frame`): the reagent ladder
                               ([I3]-, [Br2]-.), isotope satellites (parent ion +
                               13C/81Br/...) and ringing artifacts stop looking
                               like unknowns. `ion_formula.notna()` = identified;
                               `neutral_formula.notna()` = analyte with a
                               molecular reading. Ledgers without the columns
                               (plain merged analytes) emit them all-<NA>.
      * ``stamp_source``    -- where the stamped identity came from: 'M0' (a
                               merged analyte), 'observed' (an ion a per-file
                               ledger identified) or 'predicted' (a diagnostic
                               isotope satellite predicted from its parent's ion
                               formula, see below). <NA> on unstamped rows and on
                               ledgers without the column.

    A raw ts peak is matched to the *nearest* assigned ion whose m/z is within
    ``max(mz*tol_ppm*1e-6, mz_floor_da)`` -- the mDa floor absorbs the small
    raw-vs-calibrated offset at low m/z. Fully vectorised (searchsorted); safe on
    multi-million-row batch time-series. A peak matching no assigned ion keeps
    <NA>/NaN (unassigned).

    ONE-TO-ONE (``one_to_one=True``, the default). The mass match alone is
    many-to-one: neighbouring raw peaks each independently grab their nearest ion,
    so a split/shoulder peak inside the window gets stamped with the SAME
    formula+adduct as the real peak -- downstream ``groupby(formula, adduct)`` then
    sees two traces for one ion in one sample (measured: 0.14% of (sample, ion)
    pairs, 2.9% of ions, on a 2.4M-row uronium batch). The ASSIGNMENT never does
    this -- one formula owns exactly one peak (verified: 9784 per-file M0 keys, zero
    on >1 peak) and leaves the shoulder `unexplained` -- so the duplication is an
    artifact of re-matching by mass instead of carrying peak identity. This keeps
    the single best peak per (sample, ion) and flags the rest via ``dup_candidate``.
    Pass ``one_to_one=False`` for the raw many-to-one behaviour.

    CONSENSUS (``consensus=True``, the default; requires ``one_to_one``). The winner
    is the candidate nearest the ion's CONSENSUS m/z -- its ledger mass plus the
    height-weighted modal offset of its own candidates across the whole batch --
    not the bare ledger mass. This is what keeps the SAME physical peak stamped in
    every sample: where an ion has two raw tracks, "nearest the ledger mass" picks
    whichever is present, alternating between them (measured: 232 and 378 flips for
    two ions on the Wind-zone-2 batch). Pass ``consensus=False`` to rank on the bare
    ledger mass instead.

    PREDICTED SATELLITES (rows with ``stamp_source == 'predicted'``, built by
    `stamping_frame`). Three rules keep a predicted line from ever displacing a
    known ion or mislabelling an analyte:

      * **Known rows first.** The match runs in two tiers: every M0 / reagent /
        observed-satellite / artifact row first, and a peak that fell inside ANY
        known row's window -- winner or one-to-one loser -- is never offered to a
        predicted line. Predicted rows are matched only among the peaks no known
        row claimed. The consensus ordering within a tier is unchanged.
      * **The per-sample intensity gate** (`pred_ratio`, the per-file passes'
        0.3-3.5 window; `_predicted_ratio_gate`): a peak takes a predicted label
        only when its parent was stamped on a peak in the SAME sample and
        height / (height_parent * iso_rel) lies inside the window. A real
        analyte sitting at a satellite offset fails it -- its height has nothing
        to do with the parent's -- while a true satellite passes in every sample
        where both are visible; with no parent peak in the sample, no label.
      * **Track coherence** (`pred_track_min_n` / `pred_track_min_share`, the
        module's PRED_TRACK_*; `_predicted_track_coherence`): a true satellite's
        ratio is a constant of nature, so it passes the window in nearly every
        judged sample, whereas an independent compound on the line fails in most
        and passes in the few where its height happens to fit -- the mislabels
        the per-sample gate alone would hand out. Once a line has been judged in
        at least `pred_track_min_n` samples (parent present, a candidate on the
        line) it keeps its stamps only if at least `pred_track_min_share` of
        them passed; otherwise the WHOLE track is left unexplained (no stamp, no
        dup flag). Lines judged in fewer samples stand on the per-sample gate.
        Either value <= 0 switches the rule off.

    The one-to-one contest for a predicted line then runs among the surviving
    candidates only, so a shoulder that fails the gate cannot beat the real
    satellite to the label. Pass a dict as ``stats`` to receive, under
    ``'predicted_tracks'``, one audit row per predicted line that had a candidate
    (PRED_TRACK_COLS: samples judged / passed, pass share, judged, kept, peaks
    stamped) -- the batch writes it as tables/predicted_satellites.csv.
    """
    out = peaks.copy()
    n = len(out)
    nf = np.full(n, None, dtype=object)
    ad = np.full(n, None, dtype=object)
    ti = np.full(n, None, dtype=object)
    im = np.full(n, np.nan, dtype=float)
    dup = np.zeros(n, dtype=bool)
    sus = np.zeros(n, dtype=bool)
    ro = np.full(n, None, dtype=object)
    io = np.full(n, None, dtype=object)
    il = np.full(n, None, dtype=object)
    ss = np.full(n, None, dtype=object)
    if stats is not None:
        stats["predicted_tracks"] = pd.DataFrame(columns=PRED_TRACK_COLS)
    cols = getattr(ledger, "columns", None)
    if n and cols is not None and "mz" in cols and mz_col in out.columns:
        led = ledger.dropna(subset=["mz"]).sort_values("mz").reset_index(drop=True)
        if len(led):
            lmz_all = led["mz"].to_numpy(dtype=float)
            is_pred = ((led["stamp_source"].astype(str) == STAMP_PREDICTED).to_numpy()
                       if "stamp_source" in led.columns else np.zeros(len(led), dtype=bool))
            pmz = pd.to_numeric(out[mz_col], errors="coerce").to_numpy(dtype=float)
            ok = np.zeros(n, dtype=bool)           # stamped
            near = np.full(n, -1, dtype=np.int64)  # ledger position (into led)
            in_win_known = np.zeros(n, dtype=bool)
            # tier 1: every KNOWN row -- analytes, reagent ladder, observed
            # satellites, artifacts
            pos1 = np.flatnonzero(~is_pred)
            if len(pos1):
                lmz1 = lmz_all[pos1]
                nr1, sg1, ok1 = _nearest_ion(lmz1, pmz, tol_ppm, mz_floor_da)
                in_win_known = ok1.copy()
                if one_to_one and ok1.any():
                    # consensus half-window: a third of each ion's own tolerance,
                    # so two tracks separated by more than that stay resolved as
                    # separate modes
                    halfwin = np.maximum(lmz1 * tol_ppm * 1e-6, mz_floor_da) / 3.0
                    ok1, dup1, _cons = _resolve_one_to_one(
                        out, ok1, nr1, sg1, len(lmz1), halfwin,
                        sample_col, height_col, consensus=consensus)
                    dup[dup1] = True
                ok[ok1] = True
                near[ok1] = pos1[nr1[ok1]]
            # tier 2: PREDICTED satellites, only for peaks no known row claimed,
            # only where the parent's same-sample height licenses the label, and
            # only on lines that pass coherently across the batch
            pos2 = np.flatnonzero(is_pred)
            if len(pos2) and not in_win_known.all():
                lmz2 = lmz_all[pos2]
                nr2, sg2, cand = _nearest_ion(lmz2, pmz, tol_ppm, mz_floor_da)
                cand &= ~in_win_known
                near2 = pos2[nr2]
                if cand.any():
                    passed, evaluable = _predicted_ratio_gate(
                        out, led, near, ok, near2, cand, sample_col, height_col, pred_ratio)
                    n_eval, n_pass, judged, kept = _predicted_track_coherence(
                        out, near2, passed, evaluable, len(led), sample_col,
                        pred_track_min_n, pred_track_min_share)
                    ok2 = passed & kept[near2]
                    if one_to_one and ok2.any():
                        halfwin = np.maximum(lmz2 * tol_ppm * 1e-6, mz_floor_da) / 3.0
                        ok2, dup2, _cons = _resolve_one_to_one(
                            out, ok2, nr2, sg2, len(lmz2), halfwin,
                            sample_col, height_col, consensus=consensus)
                        dup[dup2] = True
                    ok[ok2] = True
                    near[ok2] = near2[ok2]
                    if stats is not None:
                        stats["predicted_tracks"] = _predicted_track_table(
                            led, pos2, near2, cand, ok2, n_eval, n_pass, judged, kept)
            lnf = led["neutral_formula"].to_numpy() if "neutral_formula" in led.columns \
                else np.full(len(led), None, dtype=object)
            lad = (led["adduct"].to_numpy() if "adduct" in led.columns
                   else np.full(len(led), None, dtype=object))
            lti = (led["tier"].to_numpy() if "tier" in led.columns
                   else np.full(len(led), None, dtype=object))
            lsus = (led["intensity_suspect"].fillna(False).to_numpy(dtype=bool)
                    if "intensity_suspect" in led.columns
                    else np.zeros(len(led), dtype=bool))
            nf[ok] = lnf[near[ok]]
            ad[ok] = lad[near[ok]]
            ti[ok] = lti[near[ok]]
            im[ok] = lmz_all[near[ok]]
            sus[ok] = lsus[near[ok]]
            for arr, col in ((ro, "role"), (io, "ion_formula"), (il, "iso_label"),
                             (ss, "stamp_source")):
                if col in led.columns:
                    arr[ok] = led[col].to_numpy()[near[ok]]
    out["neutral_formula"] = nf
    out["adduct"] = ad
    out["tier"] = ti
    out["ion_mz"] = im
    out["dup_candidate"] = dup
    out["intensity_suspect"] = sus
    out["role"] = ro
    out["ion_formula"] = io
    out["iso_label"] = il
    out["stamp_source"] = ss
    return out


def reagent_total(mat: pd.DataFrame, bin_mz: pd.Series, reagent_mzs, *, tol_ppm=8.0):
    """Per-sample sum of the reagent bins (the normaliser). reagent_mzs is a list
    of reagent ion m/z (e.g. the Br3- isotopologues)."""
    cols = []
    bm = bin_mz.sort_values()
    arr = bm.to_numpy(); idx = bm.index.to_numpy()
    for r in reagent_mzs:
        i = bisect.bisect_left(arr, r)
        for j in (i - 1, i):
            if 0 <= j < len(arr) and abs(arr[j] - r) / r * 1e6 <= tol_ppm:
                cols.append(idx[j])
    cols = [c for c in set(cols) if c in mat.columns]
    if not cols:
        return None
    return mat[cols].sum(axis=1)


def normalize(mat: pd.DataFrame, reagent_series) -> pd.DataFrame:
    """Divide every bin by the per-sample reagent total (concentration proxy)."""
    if reagent_series is None:
        return mat
    return mat.div(reagent_series.replace(0, np.nan), axis=0)


def bin_metrics(norm: pd.DataFrame, bin_mz: pd.Series) -> pd.DataFrame:
    """Per-bin presence + cv_norm on the (reagent-normalised) matrix."""
    n = len(norm)
    presence = norm.notna().sum() / n if n else norm.notna().sum()
    mean = norm.mean(); std = norm.std()
    cv = (std / mean).replace([np.inf, -np.inf], np.nan)
    out = pd.DataFrame({"mz": bin_mz.reindex(norm.columns), "presence": presence,
                        "median": norm.median(), "cv_norm": cv})
    out.index.name = "_bin"
    return out


def family_trace(norm: pd.DataFrame, bin_ids):
    """z-scored mean log-trace of a set of bins (a reference family trace)."""
    bb = [b for b in bin_ids if b in norm.columns]
    if not bb:
        return None
    lg = np.log10(norm[bb].clip(lower=norm[norm > 0].min().min() or 1e-9))
    z = (lg - lg.mean()) / lg.std()
    return z.mean(axis=1)


def correlate(norm: pd.DataFrame, trace) -> pd.Series:
    if trace is None:
        return pd.Series(np.nan, index=norm.columns)
    lg = np.log10(norm.clip(lower=norm[norm > 0].min().min() or 1e-9))
    with np.errstate(invalid="ignore", divide="ignore"):  # flat bins -> NaN r (fine)
        return lg.apply(lambda c: c.corr(trace))


# ---------------------------------------------------------------------------
# disposition + ledger application
# ---------------------------------------------------------------------------
def _nearest(bin_mz_sorted_vals, bin_mz_sorted_idx, mz, tol_ppm):
    i = bisect.bisect_left(bin_mz_sorted_vals, mz)
    best = None
    for j in (i - 1, i):
        if 0 <= j < len(bin_mz_sorted_vals):
            ppm = abs(bin_mz_sorted_vals[j] - mz) / mz * 1e6
            if ppm <= tol_ppm and (best is None or ppm < best[1]):
                best = (bin_mz_sorted_idx[j], ppm)
    return best[0] if best else None


def _disposition(row, cv, r_mono, r_formic):
    """Classify one M0 row from its formula + time-series behavior."""
    ion = str(row.get("ion_formula", "")); adduct = str(row.get("adduct", ""))
    neutral = str(row.get("neutral_formula", ""))
    if "CO3" in adduct:
        return "background:CO3-channel (TS-flat)" if (pd.notna(cv) and cv < FLAT_CV) else "CO3-channel"
    if re.search(r"Br[23]", ion):
        return "background:di-bromide cluster (TS-flat)" if (pd.notna(cv) and cv < FLAT_CV) else "di-bromide"
    if pd.notna(cv) and cv < FLAT_CV:
        if "Si" in neutral or "F" in ion:
            return "background:inlet/instrument contaminant (TS-flat)"
        return "background:flat (TS-flat)"
    if pd.notna(r_mono) and r_mono >= COVARY_R:
        return "ambient:biogenic-SOA (co-varies)"
    if pd.notna(r_formic) and r_formic >= 0.9:
        return "ambient:acid/oxygenate pool (co-varies)"
    if pd.notna(cv) and cv >= 0.45:
        return "ambient:variable"
    return "intermediate"


def apply_timeseries(ledger: pd.DataFrame, peaks: pd.DataFrame, *,
                     reagent_mzs=None, mono_anchor_mzs=None, formic_mz=None,
                     tol_ppm: float = DEFAULT_TOL_PPM, demote=True, log=print) -> dict:
    """Annotate `ledger` (in place) with ts_cv_norm / ts_r_mono / ts_r_formic /
    ts_disposition from the time-series `peaks` table, and (if demote) cap a flat
    di-bromide / CO3-channel Assigned commit at Candidate (TS-confirmed
    background). Returns a summary dict. Reagent normaliser + anchors are taken
    from the ledger when not supplied.
    """
    summary = {"annotated": 0, "demoted": 0, "ambient": 0, "background": 0}
    for col in ("ts_cv_norm", "ts_r_mono", "ts_r_formic", "ts_disposition"):
        if col not in ledger.columns:
            ledger[col] = np.nan if col != "ts_disposition" else ""

    mat, bin_mz = build_matrix(peaks, tol_ppm=tol_ppm)
    if mat.empty:
        log("[timeseries] no peaks; skipped"); return summary

    # reagent normaliser: explicit, else the ledger's reagent Br_n rows
    if reagent_mzs is None:
        rr = ledger[(ledger["role"] == L.ROLE_REAGENT)
                    & ledger["ion_formula"].astype(str).str.match(r"Br\d-")]
        reagent_mzs = rr["mz"].dropna().tolist()
    rt = reagent_total(mat, bin_mz, reagent_mzs) if reagent_mzs else None
    norm = normalize(mat, rt)
    met = bin_metrics(norm, bin_mz)

    # reference family traces (optional)
    bmz_s = bin_mz.sort_values(); bvals = bmz_s.to_numpy(); bidx = bmz_s.index.to_numpy()
    def bins_for(mzs):
        out = []
        for m in (mzs or []):
            b = _nearest(bvals, bidx, m, tol_ppm)
            if b is not None:
                out.append(b)
        return out
    if mono_anchor_mzs is None:
        mono_anchor_mzs = ledger.loc[
            ledger["neutral_formula"].astype(str).isin(
                {"C10H16O3", "C10H16O4", "C10H16O5", "C10H16O6"}), "mz"].dropna().tolist()
    mono_tr = family_trace(norm, bins_for(mono_anchor_mzs))
    formic_b = _nearest(bvals, bidx, formic_mz, tol_ppm) if formic_mz else \
        _nearest(bvals, bidx, 124.9243, tol_ppm)
    formic_tr = norm[formic_b].pipe(lambda c: np.log10(c.clip(lower=1e-9))) if formic_b in norm.columns else None
    r_mono = correlate(norm, mono_tr)
    r_formic = correlate(norm, formic_tr)

    # stamp the ledger (M0 rows)
    for i in ledger.index[ledger["role"] == L.ROLE_M0]:
        mz = ledger.at[i, "mz"]
        if pd.isna(mz):
            continue
        b = _nearest(bvals, bidx, float(mz), tol_ppm)
        cv = float(met.at[b, "cv_norm"]) if (b is not None and b in met.index and pd.notna(met.at[b, "cv_norm"])) else np.nan
        rm = float(r_mono.get(b, np.nan)) if b is not None else np.nan
        rf = float(r_formic.get(b, np.nan)) if b is not None else np.nan
        disp = _disposition(ledger.loc[i], cv, rm, rf)
        ledger.at[i, "ts_cv_norm"] = cv
        ledger.at[i, "ts_r_mono"] = rm
        ledger.at[i, "ts_r_formic"] = rf
        ledger.at[i, "ts_disposition"] = disp
        summary["annotated"] += 1
        if disp.startswith("ambient"):
            summary["ambient"] += 1
        elif disp.startswith("background"):
            summary["background"] += 1
            # conservative auto-demote: a flat di-bromide / CO3 background commit
            # must not stay Assigned once the time series shows it is background
            if demote and str(ledger.at[i, "tier"]) == "Assigned" and (
                    "di-bromide" in disp or "CO3-channel" in disp):
                ledger.at[i, "tier"] = "Candidate"
                ledger.at[i, "tier_reason"] = (str(ledger.at[i, "tier_reason"] or "")
                    + " | time-series: flat background (reagent/inlet), demoted").strip(" |")
                summary["demoted"] += 1
    log(f"[timeseries] {summary}")
    return summary


# ---------------------------------------------------------------------------
# Reproducible single-compound time-series query
# ---------------------------------------------------------------------------
def find_ts_parquet(run_dir: str) -> str:
    """The cached batch time series in a run dir (`*_ts.parquet`)."""
    import glob
    hits = sorted(glob.glob(os.path.join(os.path.expanduser(run_dir), "*_ts.parquet")))
    if not hits:
        raise FileNotFoundError(f"no *_ts.parquet in {run_dir}")
    return hits[0]


def trace(run_dir: str, query, *, tol_ppm: float = DEFAULT_TOL_PPM,
          value: str = "height", ts: "pd.DataFrame | None" = None,
          ledger: "pd.DataFrame | None" = None) -> pd.DataFrame:
    """Pull the temporal trace of ONE compound from a finished run -- assigned OR
    unassigned. Reproducible: reads the run's own ``*_ts.parquet`` (full per-sample
    peak table) and ``merged_ledger.csv`` (the pipeline's assignments), so the
    answer is fixed by the run, not by re-deriving anything.

    query : a NEUTRAL FORMULA (str; resolved to its m/z via merged_ledger, taking
            the highest-ion-score adduct) or a float M/Z (use any peak, assigned
            or not). tol_ppm sets the m/z window summed per time point.

    Returns a tidy DataFrame [datetime_utc, <value>] (one row per sample time,
    summed over the window, time-sorted). ``df.attrs`` carries: mz, assignment
    ('<formula> <adduct> (<tier>)' or 'unassigned'), n_peak_ids, tol_ppm.
    """
    import os as _os
    ts = ts if ts is not None else pd.read_parquet(find_ts_parquet(run_dir))
    if ledger is None:
        mlp = _os.path.join(_os.path.expanduser(run_dir), "merged_ledger.csv")
        ledger = pd.read_csv(mlp) if _os.path.exists(mlp) else pd.DataFrame()

    assignment = "unassigned"
    if isinstance(query, str):
        hit = ledger[ledger.get("neutral_formula").astype(str) == query] \
            if "neutral_formula" in ledger.columns else ledger.iloc[0:0]
        if not len(hit):
            raise KeyError(f"{query!r} is not an assigned neutral in {run_dir} "
                           "(pass a float m/z to trace an unassigned peak)")
        if "ion_score" in hit.columns:
            hit = hit.sort_values("ion_score", ascending=False, na_position="last")
        row = hit.iloc[0]
        mz = float(row["mz"])
        assignment = (f"{query} {row.get('adduct', '')}".strip()
                      + f" ({row.get('tier', '?')})")
    else:
        mz = float(query)
        if len(ledger) and "mz" in ledger.columns:
            d = (ledger["mz"].astype(float) - mz).abs() / mz * 1e6
            j = d.idxmin() if len(d) else None
            if j is not None and d.loc[j] <= tol_ppm:
                r = ledger.loc[j]
                assignment = (f"{r.get('neutral_formula')} {r.get('adduct', '')}".strip()
                              + f" ({r.get('tier', '?')})")

    win = (ts["mz"].astype(float) - mz).abs() / mz * 1e6 <= tol_ppm
    sub = ts[win]
    if not len(sub):
        out = pd.DataFrame({"datetime_utc": [], value: []})
    else:
        out = (sub.groupby("datetime_utc", as_index=False)[value].sum()
               .sort_values("datetime_utc").reset_index(drop=True))
    out.attrs.update({"mz": mz, "assignment": assignment,
                      "n_peak_ids": int(sub["peak_id"].nunique()) if len(sub) else 0,
                      "tol_ppm": tol_ppm, "value": value})
    return out
