"""Correlation-cluster figures for a batch — the in-package home of what used to
be the out-of-repo scratch driver `run_clusters.py`.

`cluster_batch(out_dir, ts, profile)` produces, from a representative-merge
`merged_ledger.csv` (+ the per-file ledgers it left) and the full-batch peak time
series, three figure sets + their CSVs:

  * CHANGING  — ONE unified peak space: assigned ion channels + gated unassigned
                bins, clustered on DE-GLUED residual correlation (log10 -> per-channel
                diel anomaly -> shared common-mode removed; corr_space='raw' restores
                the legacy raw-correlation behaviour). Families are labeled by their
                assigned members ("co-varies with X"); anchor-free families are NOVEL.
                Near-identical shapes merged, flat-mean clusters demoted, cohesion-
                flagged, with big standalone changers on their own small-multiples page;
  * BACKGROUND — the uncorrelated remainder + Si, split three ways: common-mode /
                boundary-layer diel carriers (named panel — they ARE the shared wave),
                low-amplitude diel-structured, and genuinely flat (bunched);
  * UNASSIGNED — leftover TS bins that did not enter the union (below the presence
                bar / isotope satellites), gated on cv/structure then clustered;

plus a per-cluster workbook and a channel-agreement QC table. Everything is
parameterised by the ReagentProfile (adducts / normaliser / reagent-ion regex /
label) and the output dir — nothing reagent- or batch-specific is hardcoded.

Pure local compute (no network); reads/writes only under `out_dir`.
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np
import pandas as pd

from peaky.reporting import analyte_viz as V
from peaky.batch import cluster as CL
from peaky import paths as PT
from peaky.batch import timeseries as TS

__version__ = "0.2.0"  # + unified assigned+unassigned residual clustering
FLOOR_DEFAULT = 200.0   # min median cps for a channel to enter clustering (legacy gate)
UNION_PRESENCE = 0.30   # an UNASSIGNED bin joins the unified clustering only if it is
                        # detected in at least this fraction of samples (precision over
                        # recall: episodic sub-30% unknowns stay in the unassigned set)


def _longest_detected_run(tr) -> int:
    """Longest run of consecutive DETECTED (nonzero) bins in a raw trace.
    NaN / 0 = not detected. Distinguishes a real multi-bin episode (rise+decay)
    from a sporadic single-bin spike at the detection limit. Intensity-agnostic:
    unlike a median-cps floor it does not penalise sharp transients whose signal
    is zero most of the run (e.g. prompt accretion dimers)."""
    y = np.nan_to_num(np.asarray(tr, dtype=float), nan=0.0)
    best = cur = 0
    for v in y:
        cur = cur + 1 if v > 0 else 0
        if cur > best:
            best = cur
    return best


def cluster_batch(out_dir, ts, profile, *, merged=None, tag=None, label=None,
                  floor: float = FLOOR_DEFAULT, bin_minutes: int | None = None,
                  gate: str = "median", min_run: int = 3,
                  corr_space: str = "residual",
                  log=print) -> dict:
    """Build the cluster/flat/unassigned figure sets for one batch.

    out_dir : run folder; holds `merged_ledger.csv` + `per_file/*_ledger.csv`
              (written by assign_batch) and receives the figures + CSVs.
    ts      : the full-batch per-sample peak time series (DataFrame or parquet path).
    profile : ReagentProfile (adducts / normaliser / reagent_ion_re / label).
    merged  : merged ledger (DataFrame or path); default `out_dir/merged_ledger.csv`.
    tag     : filename token (default profile.name); label: title text (profile.label).
    corr_space : 'residual' (default) correlates diel-anomaly + shared-mode-removed
              residuals — raw log traces of an ambient batch share a boundary-layer
              diel wave that glues unrelated chemistry into giant families (measured
              a validation dataset: largest raw family n=104 at within-r 0.76 collapsed to
              0.46 once the wave was removed; ~92% of supra-threshold pair correlation
              was the wave). 'raw' restores the legacy behaviour. Rendering is always
              on raw traces; only the similarity input changes. In residual space the
              gated UNASSIGNED bins join the SAME clustering (unified peak space) —
              families are labeled by their assigned members; anchor-free ones 'novel'.
    """
    OUT = os.path.expanduser(out_dir)
    P = PT.run_paths(OUT).ensure()
    FIG, TAB = P.figures, P.tables       # .png -> figures/, .csv/.xlsx -> tables/
    tag = tag or profile.name
    label = label or profile.label
    ADDUCTS, NORM = profile.adducts, profile.normaliser
    FLOOR = floor

    # clear stale cluster pages so a shorter run (fewer pages) can't leave orphans
    # that the report would still glob in
    for _old in glob.glob(f"{FIG}/clusters_*_{tag}_p*.png"):
        os.remove(_old)

    if merged is None:
        merged = os.path.join(OUT, "merged_ledger.csv")
    merged = pd.read_csv(merged) if isinstance(merged, str) else merged.copy()
    merged["role"] = "M0"                                  # analyte_table expects a role col

    if isinstance(ts, str):
        ts = pd.read_parquet(os.path.expanduser(ts))
    ts = ts.copy()
    ts["datetime_utc"] = pd.to_datetime(ts["datetime_utc"], utc=True)
    # Native per-sample time resolution by default (BIN_MIN=None): the samples are
    # the common time axis, so traces + correlation use one point per sample at its
    # real time — no re-gridding onto a uniform lattice (which aliased into spurious
    # empty bins). Pass bin_minutes=int to time-bin instead.
    BIN_MIN = bin_minutes
    span_min = (ts["datetime_utc"].max() - ts["datetime_utc"].min()).total_seconds() / 60.0
    if corr_space == "residual" and span_min / 60.0 < CL.DIURNAL_SPAN_H:
        # the removed common mode is a DIEL (boundary-layer) phenomenon; a short
        # batch (chamber run, single event) has no diel wave to remove — its shared
        # structure IS the chemistry, and the panel median would erase the main
        # family. Fall back to the legacy raw correlation.
        log(f"[corr-space] batch spans {span_min / 60.0:.1f} h < {CL.DIURNAL_SPAN_H:.0f} h "
            "(~2 diel cycles) — no diel common mode to remove; using RAW correlation")
        corr_space = "raw"
    log(f"=== {tag} ({label}) : merged M0={len(merged)}, TS samples={ts['sample_item_id'].nunique()}, "
        f"span={span_min:.0f}min -> res={'native/sample' if BIN_MIN is None else str(BIN_MIN)+'min'} ===")

    def cv_of(traces, cols):
        out = {}
        for f in cols:
            tr = traces[f].dropna().values if f in traces else np.array([])
            out[f] = float(np.std(tr) / np.mean(tr)) if len(tr) and np.mean(tr) > 0 else np.nan
        return out

    def reagent_mzs():
        # the merged ledger is M0-only (no ion_formula); read the reagent ions from a
        # per-file ledger (keeps role/ion_formula) so Br gets its physical /Br3- norm.
        if not profile.reagent_ion_re:
            return None
        for f in sorted(glob.glob(f"{OUT}/per_file/*_ledger.csv")):
            d = pd.read_csv(f)
            if "ion_formula" not in d.columns:
                continue
            m = d[d["ion_formula"].astype(str).str.match(profile.reagent_ion_re)]["mz"].dropna()
            if len(m):
                return sorted(set(round(float(x), 4) for x in m))
        return None

    # ---- assigned analytes, clustered PER ION CHANNEL (formula+adduct) ----------
    # Cluster each ion channel SEPARATELY (not the per-neutral SUM): channels of one
    # neutral often diverge in time, so summing blends divergent shapes.
    chan = merged[merged["role"] == "M0"].dropna(subset=["neutral_formula"]).copy()
    chan = chan[chan["neutral_formula"].astype(str) != ""]
    chan["key"] = chan["neutral_formula"].astype(str) + "|" + chan["adduct"].astype(str)
    chan = chan.drop_duplicates("key")
    ion_mz = dict(zip(chan["key"], chan["mz"]))
    is_si = dict(zip(chan["key"], chan["neutral_formula"].map(V._is_contaminant)))

    # ---- unassigned TS bins (detected up front: in residual mode the gated ones
    # join the SAME clustering as the assigned channels — one unified peak space).
    # 'Explained' = ALL roles from the per-file ledgers (M0 + iso_child + reagent +
    # artifact), NOT just merged M0 (which would flag every satellite).
    _expl = []
    for _pf in glob.glob(f"{OUT}/per_file/*_ledger.csv"):
        _d = pd.read_csv(_pf)
        _expl += _d.loc[_d["role"].astype(str) != "unexplained", "mz"].dropna().tolist()
    mat, bin_mz = TS.build_matrix(ts)
    tstamp = ts.groupby("sample_item_id")["datetime_utc"].first()
    assigned_mz = np.sort(np.array(_expl)) if _expl else np.sort(merged["mz"].dropna().to_numpy())

    def is_assigned(mz, tol=8.0):
        i = np.searchsorted(assigned_mz, mz)
        return any(0 <= j < len(assigned_mz)
                   and abs(assigned_mz[j] - mz) / mz * 1e6 <= tol for j in (i - 1, i))

    median_h = mat.median()
    un_bins = [b for b in mat.columns if not is_assigned(float(bin_mz[b]))
               and median_h[b] >= 50.0 and mat[b].notna().sum() >= CL.MIN_POINTS]
    log(f"UNASSIGNED bins: {len(un_bins)} (of {mat.shape[1]} TS bins)")
    # union-entry gates for an unassigned bin (precision over recall):
    #   presence >= UNION_PRESENCE of samples (an episodic sub-30% unknown stays in
    #   the unassigned section), and NOT an isotope satellite — a bin exactly one
    #   13C/15N/34S/37Cl/81Br/18O step above a BRIGHTER known peak co-varies with it
    #   trivially and must not be folded as a "novel co-varying compound".
    _known = sorted((float(bin_mz[b]), float(median_h[b]))
                    for b in mat.columns if is_assigned(float(bin_mz[b])))
    _kmz = np.array([m for m, _ in _known])
    _kmed = np.array([h for _, h in _known])
    presence = {b: float((mat[b] > 0).sum()) / max(1, len(mat)) for b in un_bins}
    iso_parent = {b: CL.isotope_satellite_of(float(bin_mz[b]), _kmz, _kmed,
                                             float(median_h[b])) for b in un_bins}
    un_enter = ([b for b in un_bins if presence[b] >= UNION_PRESENCE
                 and iso_parent[b] is None]
                if corr_space == "residual" else [])
    un_key = {b: f"?{float(bin_mz[b]):.4f}" for b in un_enter}
    key_bin = {v: k for k, v in un_key.items()}
    union_map = {**ion_mz, **{un_key[b]: float(bin_mz[b]) for b in un_enter}}
    if un_enter:
        log(f"UNION: {len(un_enter)} unassigned bins join the clustering "
            f"(presence>={UNION_PRESENCE:.0%}, isotope-satellites rejected: "
            f"{sum(1 for b in un_bins if iso_parent[b] is not None)})")

    grid, traces_raw = V.ion_traces(ts, union_map, mode="raw", bin_minutes=BIN_MIN)
    med = {k: (float(np.nanmedian(traces_raw[k])) if k in traces_raw and traces_raw[k].notna().any() else 0.0)
           for k in union_map}
    cv = cv_of(traces_raw, list(union_map))

    def _enter(k):
        """Entry gate to clustering.
        gate='median'  : legacy median-cps floor (>=FLOOR) — blind to transients.
        gate='episode' : detected (nonzero) in >=min_run consecutive bins — rescues
                         sharp low-abundance episodes (e.g. accretion dimers) the
                         median floor drops, while still rejecting sporadic spikes.
        Unassigned union members passed their own (stricter) gates already."""
        if k not in traces_raw:
            return False
        if k in key_bin:
            return True
        if gate == "episode":
            return _longest_detected_run(traces_raw[k].values) >= min_run
        return med.get(k, 0) >= FLOOR

    # per-ion legend (formula+adduct + match-score) + per-cluster workbook metadata;
    # unassigned union members are labeled by their m/z ("? m/z 203.0093")
    keylab = {r.key: f"{V.ion_label(r.neutral_formula, r.adduct)} ({float(r.ion_score):.2f})"
              for r in chan.itertuples()}
    keylab.update({k: f"? m/z {union_map[k]:.4f}" for k in key_bin})
    ion_lbl = lambda k: keylab.get(k, k)
    meta = {r.key: {"neutral_formula": r.neutral_formula,
                    "channel": V.ADDUCT_SUFFIX.get(str(r.adduct), str(r.adduct)),
                    "adduct": r.adduct, "m_z": round(float(r.mz), 4),
                    "match_score": round(float(r.ion_score), 3), "tier": r.tier,
                    "median_cps": round(med.get(r.key, 0), 0),
                    "cv": round(cv.get(r.key, float("nan")), 3),
                    "member_type": "assigned"}
            for r in chan.itertuples()}
    meta.update({k: {"neutral_formula": "", "channel": "?", "adduct": "",
                     "m_z": round(union_map[k], 4), "match_score": "",
                     "tier": "unassigned", "median_cps": round(med.get(k, 0), 0),
                     "cv": round(cv.get(k, float("nan")), 3),
                     "member_type": "unassigned"} for k in key_bin})
    MCOLS = ["member_type", "neutral_formula", "channel", "m_z", "match_score",
             "tier", "median_cps", "cv"]

    rmz = reagent_mzs()

    def goodk(keys):
        return [k for k in dict.fromkeys(keys)
                if k in traces_raw and np.isfinite(traces_raw[k].values.astype(float)).sum() >= CL.MIN_POINTS]

    # Cluster ALL bright organic ion channels (+ the gated unassigned union members)
    # on SHAPE — NO per-trace cv gate (it can't see coherence). Then MERGE
    # near-identical-shape clusters. The non-clustering remainder + Si contamination
    # is the background bucket.
    #
    # corr_space='residual' (default): correlate in DE-GLUED space — log10 -> each
    # channel's own hour-of-day cycle subtracted (diel anomaly: only synchronized
    # DAY-TO-DAY behaviour correlates) -> remaining shared mode (panel median over
    # the ASSIGNED members only, so unknown bins can't bias it) OLS-regressed out.
    # Raw space left ~92% of supra-threshold pair correlation to the shared
    # boundary-layer wave; residual space keeps only real co-emission.
    clust_assigned = goodk([k for k in ion_mz if _enter(k) and not is_si[k]])
    clust_unknown = goodk(list(key_bin))
    clust_cols = clust_assigned + clust_unknown
    log(f"CLUSTERING {len(clust_assigned)} organic ion-channels + {len(clust_unknown)} "
        f"unassigned bins on {corr_space.upper()} shape (entry gate='{gate}'"
        f"{f', min_run={min_run}' if gate=='episode' else f', floor={FLOOR:.0f}cps'}; no cv gate)")
    if corr_space == "residual":
        corr_frame, r2_cm, cm_trace = CL.decompose(traces_raw, clust_cols, grid,
                                                   ref_cols=clust_assigned)
        Lg, cm = CL.correlate(corr_frame, clust_cols, log=False)
    else:
        r2_cm, cm_trace = pd.Series(dtype=float), None
        Lg, cm = CL.correlate(traces_raw, clust_cols)
    lab, big = CL.cluster(cm) if len(clust_cols) >= CL.MIN_MEMBERS else (pd.Series(dtype=int), [])
    log(f"  {corr_space}: {len(big)} clusters>=3")
    lab, big = CL.merge_similar(Lg, lab, big, merge_r=CL.MERGE_R)   # collapse near-duplicate shapes
    rows, _ = CL.cluster_rows(clust_cols, lab, big, cm, traces_raw, grid, median_cps=med)
    # DEMOTE flat clusters: members co-vary but the family MEAN doesn't move.
    # hours=grid makes the split STRUCTURE-AWARE: a low-amplitude family whose mean
    # carries a coherent diel wave is KEPT, and a structureless settling family
    # (startup fall, eta2~0) is demoted even past the strict starts-high guard.
    rows, flat_rows = CL.split_flat_clusters(rows, traces_raw, hours=grid)
    dyn_ids = {r[0] for r in rows}
    flat_cluster_members = [m for r in flat_rows for m in r[1]]
    remainder = [k for k in clust_cols if int(lab.get(k, -1)) not in dyn_ids]   # not in a DYNAMIC family
    # label each family by its assigned members (chemical CONTEXT for the unknowns:
    # "co-varies with X", never "is X"); a family with no assigned anchor is NOVEL.
    clabels = {}
    for cid, mem, *_ in rows:
        anchors = sorted((m for m in mem if m not in key_bin),
                         key=lambda m: -med.get(m, 0))
        clabels[int(cid)] = (f"co-varies with {anchors[0].split('|')[0]}" if anchors
                             else "novel (no assigned anchor)")
    # cohesion backstop: mean within-cluster r in the SAME space that built the
    # clusters; a family below COHESION_MIN is flagged (reported, never re-cut).
    cohesion = CL.validate_cohesion(rows, cm)
    # per-member GAIN vs the family's mean RAW log trace: ~1 = fully modulated
    # with the family; <<1 = a large flat baseline with a small co-varying
    # component riding on it (correlation is scale-invariant, so a bright
    # regional-background acid can sit at r~0.9 while drawing visually flat —
    # the gain column is what the eye is missing on the log panels)
    gain = {}
    for cid, mem, *_ in rows:
        cols = [m for m in mem if m in traces_raw.columns]
        if len(cols) < 2:
            continue
        _Lgr = CL.log_traces(traces_raw, cols)
        _fam = _Lgr.mean(axis=1)
        for m in cols:
            y = _Lgr[m]
            ok = y.notna() & _fam.notna()
            if int(ok.sum()) >= CL.MIN_POINTS and float(_fam[ok].var()) > 0:
                x = _fam[ok]
                gain[m] = float(((x - x.mean()) * (y[ok] - y[ok].mean())).sum()
                                / ((x - x.mean()) ** 2).sum())
    for m, g in gain.items():
        meta[m]["gain_vs_family"] = round(g, 2)
    MCOLS = MCOLS + ["gain_vs_family"]
    _n_flagged = sum(1 for _, f in cohesion.values() if f)
    if _n_flagged:
        log(f"COHESION: {_n_flagged} families below mean within-r {CL.COHESION_MIN:g} (flagged)")
    # BIG STANDALONE CHANGERS: single channels that change a lot on their own.
    changers = CL.big_changers(traces_raw, remainder, grid, fold_min=CL.BIG_CHANGE_FOLD,
                               median_cps=med)
    changer_set = {c for c, _, _ in changers}
    remainder = [k for k in remainder if k not in changer_set]
    CL.render_changers(changers, traces_raw, grid, f"{FIG}/clusters_changers_{tag}", ion_lbl,
                       title=f"{label} · Large standalone changes (≥{CL.BIG_CHANGE_FOLD:g}× fold, "
                             f"or ≥{CL.BIG_CHANGE_FOLD_BRIGHT:g}× if bright) — {len(changers)} channels")
    pd.DataFrame({"ion": [ion_lbl(c) for c, _, _ in changers],
                  "neutral_formula": [c.split("|")[0] for c, _, _ in changers],
                  "channel": [meta[c]["channel"] for c, _, _ in changers],
                  "fold": [round(f, 1) for _, f, _ in changers],
                  "peak_hour": [round(ph, 2) for _, _, ph in changers],
                  "median_cps": [round(med[c], 0) for c, _, _ in changers]}).to_csv(f"{TAB}/clusters_changers_{tag}.csv", index=False)
    _n_unk_in_fam = sum(1 for r in rows for m in r[1] if m in key_bin)
    _n_novel = sum(1 for v in clabels.values() if v.startswith("novel"))
    log(f"CHANGING: {len(rows)} dynamic families covering {sum(len(r[1]) for r in rows)} "
        f"({_n_unk_in_fam} unassigned members, {_n_novel} novel families); "
        f"{len(flat_rows)} flat clusters ({len(flat_cluster_members)} ch) demoted; "
        f"{len(changers)} big standalone changers; "
        f"{len(remainder)} background")
    for cid, mem, rbar, sh, ph in rows:
        _coh, _flag = cohesion.get(int(cid), (float("nan"), False))
        log(f"  c{cid}: n={len(mem)} r̄={rbar:.2f} {sh} h{ph:.1f}"
            f"{' [LOW COHESION]' if _flag else ''} | {clabels[int(cid)]} | "
            f"{', '.join(ion_lbl(m) for m in mem[:5])}")
    posc = traces_raw[clust_cols].values if clust_cols else np.array([])
    posc = posc[np.isfinite(posc) & (posc > 0)]
    # top = true max (+20% log headroom), NOT a 99.5 pct cap, so the brightest
    # traces are never clipped at the high end; bottom stays a 1-pct/50-cps floor.
    ylimc = (max(50, np.percentile(posc, 1)), float(np.nanmax(posc)) * 1.2) if len(posc) else None
    Zc = (Lg - Lg.mean()) / Lg.std() if len(clust_cols) else Lg
    _space_note = ("clustered on DE-GLUED residuals (diel anomaly + shared-mode removed); "
                   if corr_space == "residual" else "")
    CL.render_clusters(rows, grid, Zc, traces_raw, ion_lbl, f"{FIG}/clusters_changing_{tag}",
                       remaining=None, mode="raw", ylim=ylimc,
                       title=f"{label} · Co-varying ion channels — {sum(len(r[1]) for r in rows)} channels in {len(rows)} families",
                       subtitle=f"legend:  formula+adduct  (match-score); '? m/z' = unassigned member — {_space_note}"
                                "near-identical merged, flat families demoted")
    # per-cluster workbook: one tab per (dynamic) cluster
    CL.write_cluster_workbook(list(rows), f"{TAB}/clusters_changing_{tag}.xlsx",
                              meta=meta, item_label=ion_lbl, member_cols=MCOLS)
    clustered = [c for c in clust_cols if int(lab.get(c, -1)) in dyn_ids]   # CSV = dynamic families only
    pd.DataFrame({"ion": [ion_lbl(c) for c in clustered],
                  "member_type": [meta[c]["member_type"] for c in clustered],
                  # unknowns keep their "?<mz>" key here (same as the changers CSV):
                  # an empty string round-trips through read_csv as float NaN and
                  # crashes the report's families() string-join
                  "neutral_formula": [c.split("|")[0] for c in clustered],
                  "channel": [meta[c]["channel"] for c in clustered],
                  "m_z": [meta[c]["m_z"] for c in clustered],
                  "match_score": [meta[c]["match_score"] for c in clustered],
                  "tier": [meta[c]["tier"] for c in clustered],
                  "cluster": [int(lab[c]) for c in clustered],
                  "cluster_label": [clabels.get(int(lab[c]), "") for c in clustered],
                  "cluster_cohesion": [round(cohesion.get(int(lab[c]), (float("nan"), False))[0], 3)
                                       for c in clustered],
                  "gain_vs_family": [round(gain.get(c, float("nan")), 2) for c in clustered],
                  "r2_common_mode": [round(float(r2_cm.get(c, float("nan"))), 3) for c in clustered],
                  "cv": [round(cv[c], 3) for c in clustered],
                  "median_cps": [round(med[c], 0) for c in clustered]}).to_csv(f"{TAB}/clusters_changing_{tag}.csv", index=False)

    # BACKGROUND = the uncorrelated remainder + Si contamination, split THREE ways:
    #   1. COMMON-MODE carriers (residual mode): channels whose anomaly variance was
    #      >= COMMON_R2 explained by the shared boundary-layer wave — they ARE the
    #      wave; named as one labeled panel so the removed structure stays visible
    #      (removed from clustering coverage intentionally, not silently dropped);
    #   2. STRUCTURED: low-amplitude but independently diel-structured (eta2 gate) —
    #      real weak ambient signal that didn't correlate into a family;
    #   3. FLAT: no family dynamics, no time-of-day structure.
    flat_cols = goodk(list(dict.fromkeys(remainder + [k for k in ion_mz if is_si[k] and _enter(k)])))
    cm_carriers = [c for c in flat_cols
                   if float(r2_cm.get(c, 0.0)) >= CL.COMMON_R2] if corr_space == "residual" else []
    # With the diel-anomaly transform each channel's OWN diel cycle is already
    # removed, so few channels ride a shared RESIDUAL mode — don't spend a whole
    # report page on a handful; fold a near-empty carrier set back into flat.
    if len(cm_carriers) < CL.MIN_MEMBERS:
        cm_carriers = []
    _cmset = set(cm_carriers)
    eta2 = {c: CL.trace_diurnal_eta2(traces_raw, c, grid) for c in flat_cols}
    structured = [c for c in flat_cols if c not in _cmset
                  and eta2[c] >= CL.DIURNAL_ETA2
                  and CL.trace_dynamic_range(traces_raw, c) >= CL.DIURNAL_MIN_RANGE]
    truly_flat = [c for c in flat_cols if c not in _cmset and c not in set(structured)]
    pos = traces_raw[flat_cols].values if flat_cols else np.array([])
    pos = pos[np.isfinite(pos) & (pos > 0)]
    ylim = (max(50, np.percentile(pos, 1)), float(np.nanmax(pos)) * 1.2) if len(pos) else None
    log(f"BACKGROUND: {len(flat_cols)} channels — {len(cm_carriers)} common-mode carriers / "
        f"{len(structured)} low-amplitude diel-structured / {len(truly_flat)} flat (bunched)")
    _pages = []
    if cm_carriers:
        _pages.append(CL.render_flat_panel(
            cm_carriers, traces_raw, grid, f"{FIG}/clusters_flat_{tag}_p1.png", ion_lbl,
            label=f"{label} · common-mode / boundary-layer diel", ylim=ylim,
            title=f"{label} · Common-mode / boundary-layer diel — {len(cm_carriers)} channels",
            subtitle=(f"n={len(cm_carriers)} — channels whose day-to-day anomaly is >= "
                      f"{CL.COMMON_R2:.0%} explained by the SHARED wave (panel-median common "
                      "mode): they ride the boundary-layer/temperature breathing together. "
                      "Removed from family clustering ON PURPOSE — raw-space correlation "
                      "among them reflects the wave, not chemistry; the median line below "
                      "IS the common mode")))
    cls_of = {c: CL.formula_class(c.split("|")[0]) for c in flat_cols}
    if structured:
        # subdivide by chemical class — one panel per class with the member
        # formulas listed beneath, so the structured set is READABLE instead of
        # one anonymous 160-trace blob
        _sgroups = {}
        for c in structured:
            _sgroups.setdefault(cls_of[c], []).append(c)
        _sgroups = dict(sorted(_sgroups.items(), key=lambda kv: -len(kv[1])))
        _pages += CL.render_grouped_flat(
            _sgroups, traces_raw, grid, f"{FIG}/clusters_flat_{tag}", ion_lbl,
            ylim=ylim, start_page=len(_pages) + 1,
            title=f"{label} · Structured background — {len(structured)} low-amplitude diel channels",
            subtitle=(f"channels below the amplitude gates whose trace is time-of-day STRUCTURED "
                      f"(diurnal eta2 >= {CL.DIURNAL_ETA2:g}): a coherent low-amplitude daily wave = "
                      "real ambient signal, not background; kept out of clustering only because "
                      "pairwise correlation is noise-diluted at this amplitude. Grouped by backbone "
                      "class; legend = formula+adduct (match-score)"),
            group_note=lambda name, cols: "median eta2 "
                f"{np.median([eta2[c] for c in cols]):.2f}")
    CL.render_flat_panel(truly_flat, traces_raw, grid,
                         f"{FIG}/clusters_flat_{tag}_p{len(_pages)+1}.png", ion_lbl,
                         label=f"{label} · low-amplitude / common-mode background", ylim=ylim,
                         title=f"{label} · Low-amplitude / common-mode background — "
                               f"{len(truly_flat)} ion channels (bunched)",
                         subtitle=(f"n={len(truly_flat)} — no real family dynamics AND individually "
                                   f"sub-threshold (per-channel diurnal eta2 < {CL.DIURNAL_ETA2:g}): "
                                   "uncorrelated, or in a co-varying cluster whose mean is flat, + Si. "
                                   "The median's gentle daily wave is the SHARED boundary-layer breathing "
                                   "(common-mode), not a hidden analyte — real analytes surface on the "
                                   "structured-background / family pages. Bunched so they don't bloat the "
                                   "cluster count"))
    pd.DataFrame({"ion": [ion_lbl(c) for c in flat_cols],
                  "member_type": [meta[c]["member_type"] for c in flat_cols],
                  "neutral_formula": [c.split("|")[0] for c in flat_cols],   # "?<mz>" for unknowns, never ""
                  "class": [cls_of[c] for c in flat_cols],
                  "channel": [meta[c]["channel"] for c in flat_cols], "cluster": 0,
                  "set": ["common-mode" if c in _cmset else
                          ("structured" if c in set(structured) else "flat") for c in flat_cols],
                  "diurnal_eta2": [round(eta2[c], 3) for c in flat_cols],
                  "r2_common_mode": [round(float(r2_cm.get(c, float("nan"))), 3) for c in flat_cols],
                  "cv": [round(cv[c], 3) for c in flat_cols],
                  "median_cps": [round(med[c], 0) for c in flat_cols]}).to_csv(f"{TAB}/clusters_flat_{tag}.csv", index=False)

    # CHANNEL-AGREEMENT QC: do a neutral's ion channels actually track in time?
    ca = V.channel_agreement(ts, merged[["neutral_formula", "adduct", "mz"]], bin_minutes=BIN_MIN)
    ca.to_csv(f"{TAB}/channel_agreement_{tag}.csv", index=False)
    if len(ca):
        log(f"CHANNEL AGREEMENT: {len(ca)} multi-channel neutrals — "
            f"{ca['verdict'].value_counts().to_dict()}")

    # ---- UNASSIGNED leftover: gated bins that did NOT enter the union ----------
    # (below the presence bar, isotope satellites, or raw mode) — they keep the
    # legacy varying/flat treatment so nothing is silently dropped. Bins that DID
    # enter the union live entirely in the unified figures above (family / changer
    # / common-mode / structured / flat page).
    un_left = [b for b in un_bins if b not in set(un_enter)]
    # isotope satellites of known peaks get their OWN bunch: they track their
    # parent's time series (so a bright diel parent puts a visibly-diel trace in
    # the leftover set — NOT an unexplained compound), and they are excluded from
    # the union for the same reason. Assignment-side satellite claiming should
    # absorb them at the source; this panel is the honest holding pen until then.
    un_sat = [b for b in un_left if iso_parent[b] is not None]
    un_left = [b for b in un_left if iso_parent[b] is None]

    def to_grid(m, bin_minutes=BIN_MIN):
        m = m.reindex(tstamp.sort_values().index)
        hr = (tstamp.reindex(m.index) - tstamp.min()).dt.total_seconds().values / 3600.0
        if bin_minutes is None:                            # NATIVE — one row per sample
            return hr, m.reset_index(drop=True)
        g = np.arange(0, np.nanmax(hr) + bin_minutes / 60.0, bin_minutes / 60.0)
        return g, m.groupby(np.digitize(hr, g)).median().reindex(range(1, len(g) + 1))

    rt = mat.sum(axis=1) if NORM == "tic" else TS.reagent_total(mat, bin_mz, rmz or [])
    _left_all = un_left + un_sat
    ggrid, un_raw = to_grid(mat[_left_all]) if _left_all else (grid, pd.DataFrame())
    _, un_norm = (to_grid(mat[un_left].div((rt if rt is not None else 1).replace(0, np.nan), axis=0)
                          if rt is not None else mat[un_left]) if un_left else (grid, pd.DataFrame()))
    # GATE: the unassigned set has no inherent cv gate; split flat bins out and
    # cluster only the varying ones, bunch the flat tail. hours=ggrid also promotes
    # low-amplitude but diel-STRUCTURED bins (the amplitude gates can't see them).
    un_vary, un_flat = CL.split_varying(un_raw, un_left, hours=ggrid)
    log(f"UNASSIGNED leftover ({len(un_left)} bins outside the union, "
        f"+{len(un_sat)} isotope satellites bunched separately): "
        f"{len(un_vary)} varying / {len(un_flat)} flat (bunched)")
    mzlab = lambda b: f"{bin_mz[b]:.4f}"
    posu = un_raw[un_left].values if un_left else np.array([])
    posu = posu[np.isfinite(posu) & (posu > 0)]
    ylimu = (max(50, np.percentile(posu, 1)), float(np.nanmax(posu)) * 1.2) if len(posu) else None
    Lgu, cmu = CL.correlate(un_norm, un_vary)
    labu, bigu = CL.cluster(cmu)
    rowsu, Zu = CL.cluster_rows(un_vary, labu, bigu, cmu, un_raw, ggrid,
                                median_cps={b: float(median_h[b]) for b in un_vary})
    log(f"UNASSIGNED: {len(bigu)} clusters>=3 covering {sum(len(r[1]) for r in rowsu)}")
    for cid, mem, rbar, sh, ph in rowsu:
        log(f"  c{cid}: n={len(mem)} r̄={rbar:.2f} {sh} h{ph:.1f} m/z {bin_mz[mem].min():.1f}-{bin_mz[mem].max():.1f}")
    rem_u = CL.remaining_row(un_vary, labu, bigu, un_raw, ggrid)[1] if len(un_vary) else None
    paths_u = CL.render_clusters(rowsu, ggrid, Zu, un_raw, mzlab, f"{FIG}/clusters_unassigned_{tag}",
                                 remaining=rem_u, mode="raw", ylim=ylimu,
                                 title=f"{label} · Unexplained peaks — {len(un_vary)} varying bins in {len(bigu)} clusters")
    _npages_u = len(paths_u)
    if un_flat:
        CL.render_flat_panel(un_flat, un_raw, ggrid, f"{FIG}/clusters_unassigned_{tag}_p{_npages_u+1}.png",
                             mzlab, label=f"{label} · flat / non-varying unexplained", ylim=ylimu,
                             title=f"{label} · Unexplained — {len(un_flat)} flat / non-varying bins (bunched)")
        _npages_u += 1
    if un_sat:
        satlab = lambda b: f"{bin_mz[b]:.4f} (sat of {iso_parent[b]:.4f})"
        CL.render_flat_panel(un_sat, un_raw, ggrid, f"{FIG}/clusters_unassigned_{tag}_p{_npages_u+1}.png",
                             satlab, label=f"{label} · isotope satellites", ylim=ylimu,
                             title=f"{label} · Unclaimed isotope satellites — {len(un_sat)} bins",
                             subtitle=(f"n={len(un_sat)} — bins one 13C/15N/34S/37Cl/81Br/18O step above a "
                                       "BRIGHTER known peak: each is that parent's heavy-isotope line, not an "
                                       "independent compound, and TRACKS the parent's time series (a bright "
                                       "diel parent puts a visibly-diel trace here). Excluded from family "
                                       "clustering on purpose; parent m/z per member in the set CSV — these "
                                       "should be absorbed once assignment-side satellite claiming covers them"))
    labu_idx = set(labu.index)
    _unset = set(un_enter)

    def _un_cluster(b):
        if b in _unset:
            return int(lab.get(un_key[b], 0))            # union cluster id (0 = none)
        return int(labu[b]) if b in labu_idx else 0

    pd.DataFrame({"mz": [round(float(bin_mz[b]), 4) for b in un_bins],
                  "in_union": [b in _unset for b in un_bins],
                  "cluster": [_un_cluster(b) for b in un_bins],
                  "cluster_label": [clabels.get(_un_cluster(b), "") for b in un_bins],
                  "presence": [round(presence[b], 2) for b in un_bins],
                  "isotope_parent_mz": [round(iso_parent[b], 4) if iso_parent[b] else ""
                                        for b in un_bins],
                  "median_cps": [round(float(median_h[b]), 0) for b in un_bins]}).to_csv(f"{TAB}/clusters_unassigned_{tag}.csv", index=False)

    # GATE / FUNNEL summary -> the single source of truth the PDF report reads to
    # DOCUMENT the thresholds these figures apply and the unexplained funnel counts
    # (so a reader knows why only N of the unexplained peaks are drawn). Thresholds
    # come from the real constants in cluster.py / this module, not magic numbers.
    n_unassigned_any = int(sum(1 for b in mat.columns
                               if not is_assigned(float(bin_mz[b]))))
    summary = {
        "tag": tag,
        "corr_space": corr_space,                       # 'residual' (de-glued) | 'raw'
        "gates": {
            "match_tol_ppm": 8.0,                       # is_assigned() tolerance
            "unassigned_median_cps_floor": 50.0,        # un_bins brightness floor
            "assigned_clustering_floor_cps": FLOOR,     # assigned-channel floor (median gate)
            "entry_gate": gate,                         # 'median' | 'episode'
            "min_consecutive_bins": min_run,            # episode-gate run length
            "min_trace_points": CL.MIN_POINTS,          # persistence gate
            "varying_cv_min": CL.FLAT_CV,               # sustained-change half
            "varying_burst_range": CL.PEAK_RANGE,       # transient-burst half
            "diurnal_eta2_min": CL.DIURNAL_ETA2,        # structure half (amplitude-blind)
            "diurnal_min_range": CL.DIURNAL_MIN_RANGE,  # min movement for a structured trace
            "cluster_corr_r": round(1 - CL.DIST_T, 2),  # complete-linkage cut
            "merge_corr_r": CL.MERGE_R,                 # near-duplicate merge
            "min_cluster_members": CL.MIN_MEMBERS,
            "big_change_fold": CL.BIG_CHANGE_FOLD,
            "union_presence_min": UNION_PRESENCE,       # unassigned union-entry presence
            "common_mode_r2": CL.COMMON_R2,             # wave-carrier threshold
            "cohesion_min": CL.COHESION_MIN,            # within-family flag bar
            "diel_bins": CL.DIEL_BINS,                  # hour-of-day anomaly profile bins
        },
        "unassigned": {
            "n_ts_bins": int(mat.shape[1]),
            "n_unassigned_any": n_unassigned_any,
            "n_after_brightness_persistence": len(un_bins),
            "n_entered_union": len(un_enter),
            "n_isotope_rejected": int(sum(1 for b in un_bins if iso_parent[b] is not None)),
            "n_below_presence": int(sum(1 for b in un_bins if presence[b] < UNION_PRESENCE
                                        and iso_parent[b] is None)),
            "n_in_families": int(_n_unk_in_fam),
            "n_varying_plotted": len(un_vary),
            "n_flat_bunched": len(un_flat),
            "n_isotope_satellites_bunched": len(un_sat),
            "n_clusters": len(bigu),
        },
        "assigned": {
            "n_dynamic_families": len(rows),
            "n_channels_in_families": int(sum(len(r[1]) for r in rows)),
            "n_novel_families": int(_n_novel),
            "n_low_cohesion_flagged": int(_n_flagged),
            "n_flat_clusters_demoted": len(flat_rows),
            "n_big_changers": len(changers),
            "n_common_mode_carriers": len(cm_carriers),
            "n_flat_background": len(truly_flat),
            "n_structured_background": len(structured),
        },
    }
    with open(f"{OUT}/clusters_summary.json", "w") as _fh:
        json.dump(summary, _fh, indent=2)

    log(f"DONE — figures in {OUT}")
    return {"changing": rows, "flat_clusters": flat_rows, "changers": changers,
            "unassigned": rowsu, "channel_agreement": ca, "bin_minutes": BIN_MIN,
            "labels": clabels, "cohesion": cohesion,
            "summary": summary, "out_dir": OUT, "tag": tag}
