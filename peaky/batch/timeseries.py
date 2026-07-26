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
import os
import re

import numpy as np
import pandas as pd

from peaky.assignment import ledger as L

__version__ = "0.1.0"

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
# matrix construction
# ---------------------------------------------------------------------------
def build_matrix(peaks: pd.DataFrame, *, tol_ppm: float = DEFAULT_TOL_PPM,
                 mz_col="mz", height_col="height", sample_col="sample_item_id"
                 ) -> tuple[pd.DataFrame, pd.Series]:
    """Gap-cluster peaks into m/z bins (ppm tolerance) and pivot to a
    samples x bin intensity matrix. Returns (matrix, bin_mz)."""
    d = peaks[[sample_col, mz_col, height_col]].dropna().sort_values(mz_col).reset_index(drop=True)
    mz = d[mz_col].to_numpy()
    if len(mz) == 0:
        return pd.DataFrame(), pd.Series(dtype=float)
    gaps = np.diff(mz) / mz[:-1] * 1e6
    binid = np.zeros(len(mz), dtype=np.int64)
    binid[1:] = np.cumsum(gaps > tol_ppm)
    d["_bin"] = binid
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
# bright peak. Measured over 25498 raw tracks / 30 runs of the TC2026 campaign:
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


def stamping_frame(merged: pd.DataFrame,
                   identified: pd.DataFrame | None) -> pd.DataFrame:
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
    clustering (>3 mDa starts a new track)."""
    stamp = merged.copy()
    stamp["role"] = "M0"
    if "ion_formula" not in stamp.columns:
        stamp["ion_formula"] = None
    stamp["iso_label"] = None
    if identified is None or not len(identified):
        return stamp
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
                    "iso_label": None if pd.isna(tag) else tag})
    art = idf.loc[idf["role"] == "artifact", "mz"].dropna().sort_values()
    if len(art):
        start = 0
        vals = art.to_numpy()
        for i in range(1, len(vals) + 1):
            if i == len(vals) or vals[i] - vals[i - 1] > 3e-3:
                aux.append({"mz": float(np.median(vals[start:i])),
                            "role": "artifact", "ion_formula": None,
                            "iso_label": None})
                start = i
    if aux:
        stamp = pd.concat([stamp, pd.DataFrame(aux)], ignore_index=True)
    return stamp


def annotate_peaks(peaks: pd.DataFrame, ledger: pd.DataFrame, *,
                   tol_ppm: float = DEFAULT_TOL_PPM, mz_floor_da: float = 1.5e-3,
                   mz_col: str = "mz", sample_col: str = "sample_item_id",
                   height_col: str = "height",
                   one_to_one: bool = True, consensus: bool = True) -> pd.DataFrame:
    """Stamp every time-series peak with the assigned formula/channel of the nearest
    ledger ion within tolerance. Returns a COPY of ``peaks`` with five added columns:

      * ``neutral_formula`` -- the assigned neutral formula (or <NA> if unmatched)
      * ``adduct``          -- the ionisation channel (e.g. ``[M+H]+`` / ``[M+NH4]+``)
      * ``tier``            -- the ledger tier (Assigned / Candidate / ...)
      * ``ion_mz``          -- the matched ledger ion m/z (NaN if unmatched)
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
    ledger mass instead."""
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
    cols = getattr(ledger, "columns", None)
    if n and cols is not None and "mz" in cols and mz_col in out.columns:
        led = ledger.dropna(subset=["mz"]).sort_values("mz").reset_index(drop=True)
        if len(led):
            lmz = led["mz"].to_numpy(dtype=float)
            pmz = pd.to_numeric(out[mz_col], errors="coerce").to_numpy(dtype=float)
            j = np.searchsorted(lmz, pmz)
            jl = np.clip(j - 1, 0, len(lmz) - 1)
            jr = np.clip(j, 0, len(lmz) - 1)
            near = np.where(np.abs(lmz[jl] - pmz) <= np.abs(lmz[jr] - pmz), jl, jr)
            tol = np.maximum(pmz * tol_ppm * 1e-6, mz_floor_da)
            signed = pmz - lmz[near]                 # + == peak above ledger mass
            ok = np.isfinite(pmz) & (np.abs(signed) <= tol)
            if one_to_one and ok.any():
                # consensus half-window: a third of each ion's own tolerance, so two
                # tracks separated by more than that stay resolved as separate modes
                halfwin = np.maximum(lmz * tol_ppm * 1e-6, mz_floor_da) / 3.0
                ok, dup, _cons = _resolve_one_to_one(
                    out, ok, near, signed, len(lmz), halfwin,
                    sample_col, height_col, consensus=consensus)
            lnf = led["neutral_formula"].to_numpy()
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
            im[ok] = lmz[near[ok]]
            sus[ok] = lsus[near[ok]]
            for arr, col in ((ro, "role"), (io, "ion_formula"), (il, "iso_label")):
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
