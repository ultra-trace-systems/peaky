"""The 15N-twin facts of a labelled-nitrate batch (rule K, C18; docs/EVIDENCE_LEVELS.md §3
`label_untie` / `label_veto`).

A 15N-labelled nitrate reagent (profile label `^N`, adduct `[M+^NO3]-`) makes a
neutral X's cluster at TWO masses 0.99703 Da apart: the labelled [X+^NO3]- and,
from the reagent's 14N impurity and ambient 14N nitrate, the light [X+NO3]-. The
two lines are one cluster seen through the reagent's isotopologues, so each
arbitrates the other's reading, in its own direction:

  track (14N line -> its 15N sibling) : a committed [X+NO3]- reading ties by
        construction with the organonitrate [X'-H]- (X' = X + HNO3, the SAME ion).
        The line TRACKS X's cluster when its 14N/15N ratio q(t) follows the batch's
        cluster ratio k_cl(t) -- the per-spectrum median of q over the bright CHO
        acid clusters committed on [M+^NO3]- (15N line >= REF_MIN_H15 counts, both
        lines co-detected in >= CODETECT_MIN spectra), the pair under test left out:
        median(q / k_cl) in RATIO_LO..RATIO_HI, sd(log q / k_cl) <= SD_MAX and
        r(log q, log k_cl) >= R_MIN over >= STATS_MIN spectra. Each committed line
        gets a verdict (LINE_VERDICTS): `tracks`; `consistent` (tested, not above
        the cluster ratio -- an organonitrate can only ADD 14N intensity, so a noisy
        or a reagent-dominated cluster such as trifluoroacetic acid's, at the
        reagent's own impurity, is still X's cluster); `excess` (tested, above
        RATIO_HI x the cluster ratio); `absent` (the 15N partner in <=
        PARTNER_ABSENT_SHARE of the >= PARTNER_MIN_SPECTRA spectra of the 14N line);
        `untestable`. A tracking line breaks the arbiter's tie in the cluster's
        favour where every file's tie is with the same-ion alias alone
        (`alias_only_ties`); an excess / absent / untestable line gives X nothing
        (`alien`: out of X's per-neutral pools, both ways), and an excess or absent
        one is refuted (cluster or organonitrate undecided: hard 5b). The two
        clusters of one neutral also count as ONE channel (evidence.LABEL_FOLD).
  veto  (15N line -> its 14N twin)     : a real reagent cluster [Y+^NO3]- carries
        the reagent's 14N impurity at f = (1 - purity) / purity of its height
        (ReagentProfile.purity; 0.98 -> 0.0204, the labelled nitrate dimer of the
        reagent-ion scan reads 0.0186 and trifluoroacetic acid's cluster 0.0206 in
        the same batch). Over the spectra where that twin would be detected with
        probability >= PMIN (the batch's own detection curve: the 13C lines of every
        committed M0 of >= PDET_MIN_C carbons; an empty bin takes the populated bin
        below it, 0 below the first), and where the twin's m/z is not below the
        spectrum's lowest peak (the scan start), E = sum of those probabilities;
        the twin is refuted when E >=
        E_MIN and it is seen in <= REFUTE_SHARE x E of them (passes >= PASS_SHARE x
        E; untestable when E < E_MIN): the reagent's two isotopologues refute the
        cluster reading, a hard input (5b).

Measured on the labelled-nitrate regression batch (2026-09-26 output audit rule K,
re-measured on the trunk 2026-09-27): 14 of the
42 co-detected [X+^NO3]- pairs track (the audit's merged population: 14 of 40),
8 tied [X+NO3]- rows untie (5b -> 3b,
0.179 % of the batch signal), mismatched X14/Y15 pairs pass 42/600 and
organonitrate lines 3/150; of the 291 committed [X+NO3]- lines 14 track, 18 are
consistent, 13 excess, 220 have no 15N partner (1.35 % of the signal) and 26 are
untestable, and the C11-C12 [M-H]- acids whose only cluster was such a line leave
the acid branch (identified 48.3 -> 44.6 % with the rest of the rule); the twin veto
refutes 3 pooled [Y+^NO3]- readings (0 of 42 / 42 / 53 expected detections; two
move a level, the third is the first one's line under another formula, already
5b), the same set for f 0.0150-0.0236, never a positive control (a 13C-backed
cluster of an identified acid) and every testable N-free [M-H]- control.
`untie` in the table marks a line that WOULD clear a tie; the level's
`label_untie` records where it did (the pooled pair was tied).
The table is batch-only (it needs the stamped time series), never an axis, never
in `cross`, never per file.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from peaky.chem import chemistry as C

NO3, NO3L = "[M+NO3]-", "[M+^NO3]-"
DELTA_15N = C.M["^N"] - C.M["N"]          # 0.99703 Da
D13C = 1.0033548378                        # the 13C line (as batch/neutral_pairs.py)
TRACE_TOL_PPM = 2.0
# the untie (cluster-k)
CODETECT_MIN = 100
REF_MIN_H15 = 1000.0
STATS_MIN = 50
RATIO_LO, RATIO_HI = 0.5, 2.0
SD_MAX = 0.25
R_MIN = 0.8
REF_ELEMENTS = frozenset({"C", "H", "O"})
#: the 15N partner is ABSENT when it is found in <= PARTNER_ABSENT_SHARE of the
#: >= PARTNER_MIN_SPECTRA spectra that hold the 14N line (the cluster puts the 15N
#: line at >= 2.6x the 14N one in every spectrum of the regression batch)
PARTNER_ABSENT_SHARE = 0.2
PARTNER_MIN_SPECTRA = 10
#: what a 14N line's verdict does to its reading: `tracks` -- X's cluster (the
#: tie may be broken); `consistent` -- tested, not above the cluster ratio (a
#: noisy or a reagent-dominated cluster: an organonitrate can only ADD 14N
#: intensity), counts as X's cluster; `excess` -- above RATIO_HI x the cluster
#: ratio and `absent` -- no 15N partner: cluster or organonitrate undecided,
#: gives X nothing and reads 5b; `untestable` -- gives X nothing
LINE_VERDICTS = ("tracks", "consistent", "excess", "absent", "untestable")
FOREIGN = ("excess", "absent", "untestable")
REFUTED = ("excess", "absent")
# the veto (14N twin)
C13_PER_C = 0.0107
PDET_MIN_C = 4
PDET_EDGES = np.array([0, 30, 50, 70, 90, 110, 130, 160, 200, 250, 300, 400, 600, 1000, 1e12])
PMIN = 0.5
E_MIN = 3.0
REFUTE_SHARE = 0.2
PASS_SHARE = 0.5
TABLE_COLUMNS = (
    "neutral_formula", "adduct", "committed", "mz", "mz_partner", "n14", "n_codetected", "partner_share",
    "h15_median", "reference", "k_ratio", "k_sd", "k_r", "k_n", "cluster_k", "line_verdict",
    "alias_only_tie", "untie", "alien", "f", "E", "obs", "twin_verdict", "veto", "note",
)


def in_scope(prof) -> bool:
    """A 15N-labelled nitrate channel: the profile's label is `^N` and it clusters
    on `[M+^NO3]-` (NO3_15N alone, or composed as NO3+NO3_15N)."""
    return (getattr(prof, "label_isotope", None) == "^N"
            and NO3L in set(getattr(prof, "adducts", None) or ()))


def twin_fraction(prof) -> float:
    """The 14N twin / 15N M0 height ratio a pure reagent cluster carries:
    (1 - purity) / purity of the profile's labelled reagent."""
    from peaky.chem import isotopes
    p = getattr(prof, "purity", None)
    p = float(p) if p else isotopes.LABEL_PURITY_15N
    return (1.0 - p) / p


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=list(TABLE_COLUMNS))


def alias_only_ties(ledger: pd.DataFrame) -> pd.DataFrame:
    """Per file: the tied committed `[M+NO3]-` M0 rows and whether the tie is
    with same-ion decomposition aliases alone (the tier engine's own test: the
    tie disappears once `tiers._drop_decomposition_aliases` removes them)."""
    from peaky.assignment import tiers as T
    cols = ["neutral_formula", "adduct", "alias_only"]
    if ledger is None or not len(ledger) or "role" not in ledger.columns:
        return pd.DataFrame(columns=cols)
    m0 = ledger[(ledger["role"].astype(str) == "M0") & (ledger["adduct"].astype(str) == NO3)]
    if "tied" in m0.columns:
        m0 = m0[m0["tied"].map(lambda v: bool(T._truthy(v)))]
    else:
        m0 = m0.iloc[0:0]
    out = []
    for _, r in m0.iterrows():
        alts_all = T._alts(r.get("alternatives"))
        alts, n_aliased = T._drop_decomposition_aliases(r, alts_all)
        tied_free = T._margin_density_tie(r, alts, n_aliased, len(alts_all))[3]
        out.append((str(r.get("neutral_formula") or ""), NO3, bool(n_aliased > 0 and not tied_free)))
    return pd.DataFrame(out, columns=cols)


class _Traces:
    """The stamped batch time series as one m/z-sorted array: per target m/z, the
    TALLEST peak within +-TRACE_TOL_PPM in each spectrum (NaN where none)."""

    def __init__(self, ts: pd.DataFrame):
        t = ts[["sample_item_id", "mz", "height"]].copy()
        when = pd.to_datetime(ts.groupby("sample_item_id")["datetime_utc"].first(), utc=True).sort_values()
        self.order = list(when.index)
        code = {sid: i for i, sid in enumerate(self.order)}
        t["mz"] = pd.to_numeric(t["mz"], errors="coerce")
        t = t.dropna(subset=["mz"]).sort_values("mz", kind="mergesort")
        self.mz = t["mz"].to_numpy(float)
        self.h = pd.to_numeric(t["height"], errors="coerce").to_numpy(float)
        self.s = t["sample_item_id"].map(code).to_numpy(int)
        # each spectrum's scan start: its lowest peak, capped at the batch's median
        # lowest peak (a sparse spectrum with nothing near the start keeps the
        # batch's edge); a line below it was never measured there
        low = np.full(len(self.order), np.inf)
        np.fmin.at(low, self.s, self.mz)
        if np.isfinite(low).any():
            low = np.minimum(low, np.median(low[np.isfinite(low)]))
        self.low = low

    def heights(self, targets) -> np.ndarray:
        targets = np.asarray(targets, float)
        H = np.full((len(self.order), len(targets)), np.nan)
        tol = TRACE_TOL_PPM * 1e-6
        lo = np.searchsorted(self.mz, targets * (1 - tol))
        hi = np.searchsorted(self.mz, targets * (1 + tol))
        for j, (a, b) in enumerate(zip(lo, hi)):
            if b > a:
                np.fmax.at(H[:, j], self.s[a:b], self.h[a:b])
        return H


def _committed(frames: dict) -> pd.DataFrame:
    """(neutral_formula, adduct, ion_formula) of every regular committed M0 pair
    of the pooled per-file ledgers."""
    from peaky.assignment import evidence as EV
    parts = []
    for f in frames.values():
        if f is None or not len(f) or "role" not in f.columns:
            continue
        m0 = f[f["role"].astype(str) == "M0"]
        m0 = m0[~EV.is_ion_only(m0).to_numpy()]
        cols = [c for c in ("neutral_formula", "adduct", "ion_formula") if c in m0.columns]
        parts.append(m0[cols])
    if not parts:
        return pd.DataFrame(columns=["neutral_formula", "adduct", "ion_formula"])
    m = pd.concat(parts, ignore_index=True, sort=False)
    m = m[m["neutral_formula"].notna() & m["adduct"].notna()]
    m["neutral_formula"] = m["neutral_formula"].astype(str)
    m["adduct"] = m["adduct"].astype(str)
    return m.drop_duplicates(["neutral_formula", "adduct"]).reset_index(drop=True)


def _cluster_k(tr: _Traces, neutrals: list, ref_pool=None) -> pd.DataFrame:
    """The cluster-k test on the 14N line of every neutral in `neutrals`; the
    references are drawn from `ref_pool` (the neutrals committed on the labelled
    cluster; all of `neutrals` when None). Module docstring."""
    mz14 = np.array([C.ion_mz(n, NO3) for n in neutrals])
    mz15 = np.array([C.ion_mz(n, NO3L) for n in neutrals])
    H14, H15 = tr.heights(mz14), tr.heights(mz15)
    both = np.isfinite(H14) & np.isfinite(H15)
    nco = both.sum(axis=0)
    n14 = np.isfinite(H14).sum(axis=0)
    rec = pd.DataFrame({"neutral_formula": neutrals, "mz": mz14, "mz_partner": mz15, "n14": n14,
                        "n_codetected": nco, "partner_share": nco / np.maximum(n14, 1)})
    rec["h15_median"] = [float(np.median(H15[both[:, j], j])) if nco[j] else np.nan for j in range(len(neutrals))]
    tested = nco >= CODETECT_MIN
    cho = np.array([set(k for k, v in C.parse_formula(n).items() if v) <= REF_ELEMENTS for n in neutrals])
    pool = np.ones(len(neutrals), bool) if ref_pool is None else np.isin(neutrals, list(ref_pool))
    ref = tested & cho & pool & (np.nan_to_num(rec["h15_median"].to_numpy(float)) >= REF_MIN_H15)
    rec["reference"] = ref
    Q = H14 / H15
    Qref = Q[:, ref]
    ref_idx = {j: i for i, j in enumerate(np.flatnonzero(ref))}
    stats = []
    for j in range(len(neutrals)):
        if not tested[j] or not ref.any():
            stats.append((np.nan, np.nan, np.nan, 0))
            continue
        keep = [i for jj, i in ref_idx.items() if jj != j]
        if not keep:
            stats.append((np.nan, np.nan, np.nan, 0))
            continue
        with np.errstate(all="ignore"):
            k = np.nanmedian(Qref[:, keep], axis=1)
        q = Q[:, j]
        ok = np.isfinite(q) & np.isfinite(k) & (q > 0) & (k > 0)
        n_ok = int(ok.sum())
        if n_ok < STATS_MIN:
            stats.append((np.nan, np.nan, np.nan, n_ok))
            continue
        lg = np.log(q[ok] / k[ok])
        r = float(np.corrcoef(np.log(q[ok]), np.log(k[ok]))[0, 1])
        stats.append((float(np.exp(np.median(lg))), float(np.std(lg, ddof=1)), r, n_ok))
    rec["k_ratio"], rec["k_sd"], rec["k_r"], rec["k_n"] = zip(*stats) if stats else ([], [], [], [])
    rec["cluster_k"] = (rec["k_ratio"].between(RATIO_LO, RATIO_HI) & (rec["k_sd"] <= SD_MAX)
                        & (rec["k_r"] >= R_MIN)).fillna(False).astype(bool)
    stats_ok = rec["k_ratio"].notna()
    absent = (rec["n14"] >= PARTNER_MIN_SPECTRA) & (rec["partner_share"] <= PARTNER_ABSENT_SHARE)
    # an absent partner outranks the statistics: on a long batch a line can be
    # co-detected in >= CODETECT_MIN spectra and still lack its 15N partner in most
    rec["line_verdict"] = np.select(
        [absent, rec["cluster_k"], stats_ok & (rec["k_ratio"] > RATIO_HI), stats_ok],
        ["absent", "tracks", "excess", "consistent"], default="untestable")
    rec["adduct"] = NO3
    return rec


def _pdet_curve(tr: _Traces, committed: pd.DataFrame) -> np.ndarray:
    """P(a 13C line is detected | its expected height), per PDET_EDGES bin, from
    the 13C lines of every committed M0 of >= PDET_MIN_C carbons. A height below
    every measured line reads 0 (never testable); a gap between populated bins
    takes the bin below it."""
    ions = committed.copy()
    ions["nC"] = [C.parse_formula(str(i) if isinstance(i, str) and i else n).get("C", 0)
                  for i, n in zip(ions.get("ion_formula", pd.Series([""] * len(ions))), ions["neutral_formula"])]
    ions = ions[ions["nC"] >= PDET_MIN_C]
    mz = []
    for n, a in zip(ions["neutral_formula"], ions["adduct"]):
        try:
            mz.append(C.ion_mz(n, a))
        except Exception:
            mz.append(np.nan)
    ions = ions.assign(mz=mz).dropna(subset=["mz"])
    if ions.empty:
        return np.zeros(len(PDET_EDGES) - 1)
    H0 = tr.heights(ions["mz"].to_numpy(float))
    H13 = tr.heights(ions["mz"].to_numpy(float) + D13C)
    seen = np.isfinite(H0)
    e = (C13_PER_C * ions["nC"].to_numpy(float))[None, :] * np.nan_to_num(H0)
    e, d = e[seen], np.isfinite(H13)[seen]
    b = np.clip(np.searchsorted(PDET_EDGES, e, side="right") - 1, 0, len(PDET_EDGES) - 2)
    num = np.bincount(b, weights=d.astype(float), minlength=len(PDET_EDGES) - 1)
    den = np.bincount(b, minlength=len(PDET_EDGES) - 1)
    # a bin the batch holds no line in takes the populated bin below it, 0 below
    # the first: a height under every measured 13C line is never testable
    out, last = np.zeros(len(den)), 0.0
    for i, (x, n) in enumerate(zip(num, den)):
        last = x / n if n else last
        out[i] = last
    return out


def _twin_veto(tr: _Traces, neutrals: list, f: float, curve: np.ndarray) -> pd.DataFrame:
    """The 14N-twin test on every committed [Y+^NO3]- (module docstring)."""
    mz = np.array([C.ion_mz(n, NO3L) for n in neutrals])
    H0, TW = tr.heights(mz), tr.heights(mz - DELTA_15N)
    # a spectrum whose lowest peak lies above the twin never measured it
    # (a twin that was found was measured, even as the spectrum's lowest peak)
    seen = np.isfinite(H0) & (((mz - DELTA_15N)[None, :] >= tr.low[:, None]) | np.isfinite(TW))
    e = np.where(seen, np.nan_to_num(H0) * f, 0.0)
    b = np.clip(np.searchsorted(PDET_EDGES, e, side="right") - 1, 0, len(curve) - 1)
    p = np.where(seen, curve[b], 0.0)
    use = p >= PMIN
    E = (p * use).sum(axis=0)
    obs = (np.isfinite(TW) & use).sum(axis=0)
    verdict = np.where(E < E_MIN, "untestable",
                       np.where(obs >= PASS_SHARE * E, "passes",
                                np.where(obs <= REFUTE_SHARE * E, "refuted", "unclear")))
    out = pd.DataFrame({"neutral_formula": neutrals, "adduct": NO3L, "mz": mz, "mz_partner": mz - DELTA_15N,
                        "f": f, "E": E, "obs": obs, "twin_verdict": verdict})
    out["veto"] = out["twin_verdict"].eq("refuted")
    out["note"] = [f"no 14N twin: seen in {int(o)} of {x:.0f} expected detections" if v else ""
                   for o, x, v in zip(out["obs"], out["E"], out["veto"])]
    return out


def measure(ts: pd.DataFrame | None, frames: dict, prof, *, alias_ties: dict | None = None,
            log=print) -> pd.DataFrame:
    """The label-twin table: one row per committed [X+^NO3]- neutral's 14N line
    (the untie) and per committed [Y+^NO3]- (the veto). Empty with a header when
    the profile is not a labelled-nitrate channel, the batch has no time series,
    or nothing is committed on either nitrate adduct. `alias_ties`: {label:
    alias_only_ties(ledger)}."""
    if not in_scope(prof) or ts is None or not len(ts):
        return _empty()
    com = _committed(frames)
    n15 = sorted(set(com.loc[com["adduct"] == NO3L, "neutral_formula"]) - {""})
    n14 = set(com.loc[com["adduct"] == NO3, "neutral_formula"]) - {""}
    if not n15 and not n14:
        return _empty()
    tr = _Traces(ts)
    ck = _cluster_k(tr, sorted(set(n15) | n14), ref_pool=n15)
    # the untie guard: every file's tie on the [X+NO3]- reading is with same-ion aliases only
    parts = [t for t in (alias_ties or {}).values() if t is not None and len(t)]
    if parts:
        at = pd.concat(parts, ignore_index=True)
        alias_ok = at.groupby("neutral_formula")["alias_only"].all()
    else:
        alias_ok = pd.Series(dtype=bool)
    ck["alias_only_tie"] = ck["neutral_formula"].map(alias_ok).fillna(False).astype(bool)
    ck["committed"] = ck["neutral_formula"].isin(n14)
    ck["untie"] = ck["line_verdict"].eq("tracks") & ck["alias_only_tie"] & ck["committed"]
    ck["alien"] = ck["committed"] & ck["line_verdict"].isin(FOREIGN)
    ck["veto"] = ck["committed"] & ck["line_verdict"].isin(REFUTED)
    ck["note"] = [_line_note(r) if v else "" for r, v in zip(ck.itertuples(index=False), ck["veto"])]
    f = twin_fraction(prof)
    tw = _twin_veto(tr, n15, f, _pdet_curve(tr, com)) if n15 else pd.DataFrame(columns=list(TABLE_COLUMNS))
    tw["committed"] = True
    out = pd.concat([ck, tw], ignore_index=True, sort=False)
    for c in TABLE_COLUMNS:
        if c not in out.columns:
            out[c] = np.nan
    for c in ("committed", "reference", "cluster_k", "alias_only_tie", "untie", "alien", "veto"):
        out[c] = out[c].fillna(False).astype(bool)
    out["note"] = out["note"].fillna("")
    out = out[list(TABLE_COLUMNS)].sort_values(["neutral_formula", "adduct"]).reset_index(drop=True)
    c14 = ck[ck["committed"]]
    log(f"[label_twins] {len(c14)} committed [M+NO3]- lines: "
        + ", ".join(f"{int((c14['line_verdict'] == v).sum())} {v}" for v in LINE_VERDICTS)
        + f" ({int(out['untie'].sum())} untie); {len(n15)} [M+^NO3]- readings: {int(tw['veto'].sum())} refuted by "
        f"their 14N twin (f {f:.4f}, {int((tw['twin_verdict'] == 'untestable').sum())} untestable)")
    return out


def _line_note(r) -> str:
    if r.line_verdict == "absent":
        return (f"no 15N partner: seen in {int(r.n_codetected)} of the {int(r.n14)} spectra of the 14N line; "
                f"cluster or organonitrate undecided")
    return (f"the 14N line runs {r.k_ratio:.1f}x its cluster share of the 15N line; "
            f"cluster or organonitrate undecided")


def facts(table: pd.DataFrame | None) -> dict | None:
    """What evidence.level_pooled reads (its `label=`): {'untie': {(n, a)},
    'veto': {(n, a): note}, 'alien': {(n, a)}} -- None for an empty table (out of
    scope, or no time series), where nothing of rule K applies, the one-channel
    fold included."""
    if table is None or not len(table):
        return None
    alien = set()
    if "alien" in table.columns:
        t = table[table["alien"].map(_truth)]
        alien = set(zip(t["neutral_formula"].astype(str), t["adduct"].astype(str)))
    return {"untie": untie(table), "veto": veto(table), "alien": alien}


def untie(table: pd.DataFrame | None) -> set:
    """{(neutral, '[M+NO3]-')} whose tie the 15N sibling breaks."""
    if table is None or not len(table) or "untie" not in table.columns:
        return set()
    t = table[table["untie"].map(_truth)]
    return set(zip(t["neutral_formula"].astype(str), t["adduct"].astype(str)))


def veto(table: pd.DataFrame | None) -> dict:
    """{(neutral, adduct): note} rule K refutes: an [M+^NO3]- reading whose 14N
    twin is absent, an [M+NO3]- reading whose 14N line exceeds its cluster share
    or has no 15N partner."""
    if table is None or not len(table) or "veto" not in table.columns:
        return {}
    t = table[table["veto"].map(_truth)]
    notes = t["note"] if "note" in t.columns else pd.Series([""] * len(t), index=t.index)
    return {(str(n), str(a)): (str(x) if isinstance(x, str) else "")
            for n, a, x in zip(t["neutral_formula"], t["adduct"], notes)}


def _truth(v) -> bool:
    if isinstance(v, str):
        return v.strip().lower() in ("true", "1", "yes")
    return bool(v) if v is not None and not (isinstance(v, float) and np.isnan(v)) else False


def summary(table: pd.DataFrame | None, prof) -> dict:
    """The funnel, for batch_summary.json."""
    if table is None or not len(table):
        return {"in_scope": bool(in_scope(prof)), "lines": 0, "untie": 0, "readings": 0, "refuted": 0}
    t14 = table[table["adduct"] == NO3]
    t15 = table[table["adduct"] == NO3L]
    c14 = t14[t14["committed"].map(_truth)] if "committed" in t14.columns else t14
    verdicts = c14["line_verdict"].astype(str) if "line_verdict" in c14.columns else pd.Series(dtype=str)
    return {"in_scope": True, "f": float(t15["f"].iloc[0]) if len(t15) else None,
            "lines": int(len(t14)), "codetected": int((pd.to_numeric(t14["n_codetected"]) >= CODETECT_MIN).sum()),
            "references": int(t14["reference"].map(_truth).sum()), "cluster_k": int(t14["cluster_k"].map(_truth).sum()),
            "committed_lines": int(len(c14)), **{f"lines_{v}": int((verdicts == v).sum()) for v in LINE_VERDICTS},
            "alien": int(c14["alien"].map(_truth).sum()) if "alien" in c14.columns else 0,
            "untie": int(t14["untie"].map(_truth).sum()), "readings": int(len(t15)),
            "testable": int((t15["twin_verdict"] != "untestable").sum()),
            "passes": int((t15["twin_verdict"] == "passes").sum()),
            "refuted": int(t15["veto"].map(_truth).sum()),
            "lines_refuted": int(c14["veto"].map(_truth).sum())}
