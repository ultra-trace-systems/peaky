"""The reagent-ion water ladder of a batch (C15).

A soft-interface CIMS carries its reagent ions hydrated: Br-.(H2O)n, NO3-.(H2O)n,
HNO3.NO3-.(H2O)n ... The per-file reagent library declares only the first rung of the
halide cores (`reagents.build_library`, k = 1) and nothing for the nitrate cores, whose
bare ions are the HNO3 / HNO2 analyte readings. How far a ladder reaches is not a
constant of the chemistry: it moves with the source and the humidity. On a five-day
bromide/nitrate TOF batch the ladders end at n <= 8 (Br-) and n <= 5 (NO3-) before an
instrument restart and run past n = 15 after it, where the higher rungs were committed
as C1-C37 organics (C13H12O8 [M-H]- at Br-.(H2O)12, C13H22N2O4 [M+NO3]- at NO3-.(H2O)15).

So the ladder is MEASURED on the batch's own time series, per acquisition segment:

  * segments  -- the spectra in time order, cut where two consecutive spectra are more
                 than max(GAP_MIN_MINUTES, GAP_X_MEDIAN x the median spacing) apart; a
                 segment of fewer than MIN_SEGMENT_SPECTRA spectra joins its neighbour.
                 One segment when the frame carries no time.
  * a rung    -- core.(H2O)n, n = 1 .. N_MAX, for every core the profile declares
                 (`ReagentProfile.water_cores`; halogen isotopologues enumerated). In
                 a segment it PASSES when the core and every rung 1..n are present (a
                 peak within the window) in >= MIN_PRESENCE of the segment's spectra,
                 and the rung itself is present >= DECOY_X x the highest presence of
                 its decoy offsets (DECOY_OFFSETS_DA; floor DECOY_FLOOR, which never
                 binds while MIN_PRESENCE > DECOY_X x DECOY_FLOOR). A rung that
                 passes in any segment is a reagent ion of the batch -- stamped and
                 stripped over every segment (a known limit: a peak that is the rung
                 after a restart and something else before it goes with the rung).
  * the window -- the batch's stamping half-window (`MassScale.stamp_ppm`).

A passing rung becomes a reagent row of the batch stamp (`stamp_rows`), and a merged
ANALYTE row whose m/z sits within the window of a passing rung is taken out of the
merged ledger (`strip_rung_rows`) and listed in tables/reagent_water.csv: its reading
is the water cluster. The per-file ledgers are untouched. Nothing here tiers or levels
a row; a profile without water cores, or a batch where no core is present, changes
nothing.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd

from peaky.chem import chemistry as C

WATER = C.neutral_mass("H2O")
#: decoy offsets (Da) a rung's presence is compared with, in the same segment
DECOY_OFFSETS_DA = (-0.05, -0.035, -0.02, 0.02, 0.035, 0.05)
MIN_PRESENCE = 0.5
DECOY_X = 3.0
DECOY_FLOOR = 0.02
N_MAX = 45
GAP_MIN_MINUTES = 60.0
GAP_X_MEDIAN = 5.0
MIN_SEGMENT_SPECTRA = 10

#: halogen isotopologues a core is enumerated over (mass, tag) -- the reagent
#: library's own tags, so a rung row joins the per-file `[Br1+1xH2O]- (79Br)` rows
_ISOTOPES = {"Br": ((78.9183371, "79Br"), (80.9162906, "81Br")),
             "Cl": ((34.96885268, "35Cl"), (36.96590259, "37Cl"))}
TABLE_COLUMNS = ("core", "iso_tag", "n", "ion_formula", "mz", "mz_obs", "segments",
                 "presence", "decoy_presence", "displaced")


@dataclass(frozen=True)
class Core:
    """One reagent-side core ion: `composition` is its neutral atom count (the anion
    adds an electron), `tag` the halogen isotopologue ('' for none)."""
    composition: str
    tag: str
    mz: float

    @property
    def label(self) -> str:
        return f"{self.composition}-" + (f" ({self.tag})" if self.tag else "")


def cores(water_cores, *, polarity: str = "-") -> list[Core]:
    """The core ions of the declared compositions, halogen isotopologues enumerated
    (a composition with k Br atoms gives k + 1 cores: 79Br..81Br combinations)."""
    out: list[Core] = []
    sign = -1.0 if polarity == "+" else 1.0
    for comp in water_cores or ():
        cnt = C.parse_formula(str(comp))
        hal = [el for el in _ISOTOPES if cnt.get(el, 0)]
        rest = {el: v for el, v in cnt.items() if el not in hal}
        base = C.neutral_mass(C.format_formula(rest)) if rest else 0.0
        if not hal:
            out.append(Core(str(comp), "", base + sign * C.M_E))
            continue
        el = hal[0]
        for combo in itertools.combinations_with_replacement(_ISOTOPES[el], cnt[el]):
            mass = base + sum(m for m, _ in combo) + sign * C.M_E
            out.append(Core(str(comp), "+".join(t for _, t in combo), mass))
    return out


def rung_formula(core: Core, n: int, charge: str = "-") -> str:
    """The ion formula of core.(H2O)n, formatted like the reagent library's."""
    cnt = dict(C.parse_formula(core.composition))
    cnt["H"] = cnt.get("H", 0) + 2 * n
    cnt["O"] = cnt.get("O", 0) + n
    return C.format_formula(cnt) + charge


def segments(ts: pd.DataFrame, *, gap_min: float = GAP_MIN_MINUTES, gap_x: float = GAP_X_MEDIAN,
             min_spectra: int = MIN_SEGMENT_SPECTRA) -> pd.Series:
    """{spectrum id: segment index} over the frame's spectra (`sample_item_id`),
    cut at acquisition gaps; one segment when the frame carries no time."""
    ids = pd.unique(ts["sample_item_id"])
    if "datetime_utc" not in ts.columns or len(ids) < 2:
        return pd.Series(0, index=pd.Index(ids, name="sample_item_id"))
    t = pd.to_datetime(ts.groupby("sample_item_id")["datetime_utc"].first(), utc=True, errors="coerce")
    if t.isna().all():
        return pd.Series(0, index=t.index)
    t = t.sort_values()
    gaps = t.diff().dt.total_seconds().div(60.0)
    thr = max(float(gap_min), float(gap_x) * float(gaps.median()))
    seg = (gaps > thr).cumsum().fillna(0).astype(int)
    # a segment too short to measure a presence on joins its neighbour
    sizes = seg.value_counts().sort_index()
    order = list(sizes.index)
    label = {}
    for i, s in enumerate(order):
        if sizes[s] >= min_spectra or len(order) == 1:
            label[s] = s
        elif i > 0:
            label[s] = label[order[i - 1]]
        else:
            label[s] = None                      # the first one: joins the next kept segment
    nxt = next((label[s] for s in order if label[s] is not None), 0)
    label = {s: (v if v is not None else nxt) for s, v in label.items()}
    # renumber 0.. in time order
    uniq = {v: i for i, v in enumerate(dict.fromkeys(label[s] for s in order))}
    return seg.map(lambda s: uniq[label[s]])


def _presence(sorted_mz: dict, sids, targets: np.ndarray, tol_ppm: float) -> tuple[np.ndarray, list]:
    """Share of `sids` holding a peak within tol of each target, and the matched m/z."""
    hits = np.zeros(len(targets), dtype=float)
    mzs: list[list[float]] = [[] for _ in targets]
    for sid in sids:
        mz = sorted_mz.get(sid)
        if mz is None or not len(mz):
            continue
        j = np.clip(np.searchsorted(mz, targets), 1, len(mz) - 1)
        left, right = mz[j - 1], mz[j]
        best = np.where(np.abs(right - targets) < np.abs(left - targets), right, left)
        ok = np.abs(best - targets) <= targets * tol_ppm * 1e-6
        hits += ok
        for k in np.nonzero(ok)[0]:
            mzs[k].append(float(best[k]))
    n = max(len(sids), 1)
    return hits / n, mzs


def detect(ts: pd.DataFrame, water_cores, *, tol_ppm: float, polarity: str = "-",
           n_max: int = N_MAX, min_presence: float = MIN_PRESENCE, decoy_x: float = DECOY_X,
           decoy_floor: float = DECOY_FLOOR, decoy_offsets=DECOY_OFFSETS_DA) -> pd.DataFrame:
    """One row per PASSING rung (columns TABLE_COLUMNS minus `displaced`): the core,
    n, its ion formula, exact and observed m/z (median of the matched peaks in the
    segments it passed in), those segments, the presence and decoy presence there."""
    empty = pd.DataFrame(columns=[c for c in TABLE_COLUMNS if c != "displaced"])
    cs = cores(water_cores, polarity=polarity)
    if not cs or ts is None or not len(ts):
        return empty
    seg = segments(ts)
    frame = ts[["sample_item_id", "mz"]].dropna()
    sorted_mz = {sid: np.sort(g["mz"].to_numpy(dtype=float)) for sid, g in frame.groupby("sample_item_id")}
    by_seg = {s: list(seg.index[seg == s]) for s in sorted(seg.unique())}
    charge = "+" if polarity == "+" else "-"
    rows = []
    for core in cs:
        ns = np.arange(1, n_max + 1)
        rung_mz = core.mz + ns * WATER
        passed: dict[int, list] = {}
        for s, sids in by_seg.items():
            core_p, _ = _presence(sorted_mz, sids, np.array([core.mz]), tol_ppm)
            if core_p[0] < min_presence:
                continue
            pres, mzs = _presence(sorted_mz, sids, rung_mz, tol_ppm)
            dec = np.zeros(len(ns))
            for off in decoy_offsets:
                d, _ = _presence(sorted_mz, sids, rung_mz + off, tol_ppm)
                dec = np.maximum(dec, d)
            contiguous = np.cumprod(pres >= min_presence).astype(bool)
            ok = contiguous & (pres >= decoy_x * np.maximum(dec, decoy_floor))
            for k in np.nonzero(ok)[0]:
                passed.setdefault(int(ns[k]), []).append((s, float(pres[k]), float(dec[k]), mzs[k]))
        for n, hits in sorted(passed.items()):
            obs = [m for _, _, _, ms in hits for m in ms]
            rows.append(dict(core=core.composition, iso_tag=core.tag, n=n,
                             ion_formula=rung_formula(core, n, charge),
                             mz=float(core.mz + n * WATER), mz_obs=float(np.median(obs)) if obs else np.nan,
                             segments="|".join(str(s) for s, _, _, _ in hits),
                             presence="|".join(f"{p:.2f}" for _, p, _, _ in hits),
                             decoy_presence="|".join(f"{d:.2f}" for _, _, d, _ in hits)))
    return pd.DataFrame(rows, columns=empty.columns) if rows else empty


def strip_rung_rows(merged: pd.DataFrame, rungs: pd.DataFrame, *, tol_ppm: float, log=print):
    """(kept, stripped): the merged analyte rows whose m/z sits within tol of a passing
    rung's observed m/z are the water cluster, not the reading; they leave the merged
    ledger. `stripped` carries the rung each one sat on (`rung`)."""
    if merged is None or not len(merged) or rungs is None or not len(rungs):
        return merged, (merged.iloc[0:0].copy() if merged is not None else pd.DataFrame())
    ref = pd.to_numeric(rungs["mz_obs"], errors="coerce").fillna(pd.to_numeric(rungs["mz"])).to_numpy()
    order = np.argsort(ref)
    ref = ref[order]
    labels = [f"{rungs['core'].iloc[i]}{('(' + rungs['iso_tag'].iloc[i] + ')') if rungs['iso_tag'].iloc[i] else ''}"
              f".(H2O){int(rungs['n'].iloc[i])}" for i in order]
    mz = pd.to_numeric(merged["mz"], errors="coerce").to_numpy()
    j = np.clip(np.searchsorted(ref, mz), 1, max(len(ref) - 1, 1))
    cand = np.stack([j - 1, np.minimum(j, len(ref) - 1)], axis=1) if len(ref) > 1 else np.zeros((len(mz), 2), int)
    rung_of = [None] * len(mz)
    for i, m in enumerate(mz):
        if not np.isfinite(m):
            continue
        for k in cand[i]:
            if abs(ref[k] - m) <= m * tol_ppm * 1e-6:
                rung_of[i] = labels[k]
                break
    hit = np.array([r is not None for r in rung_of])
    stripped = merged.loc[hit].copy()
    stripped["rung"] = [r for r in rung_of if r is not None]
    kept = merged.loc[~hit].reset_index(drop=True)
    if len(stripped):
        log(f"[reagent-water] {len(stripped)} merged row(s) sit on a passing water rung and leave the "
            f"merged ledger: " + ", ".join(f"{a} {b} = {r}" for a, b, r in
                                           zip(stripped["neutral_formula"].astype(str).head(4),
                                               stripped["adduct"].astype(str).head(4),
                                               stripped["rung"].head(4))) + (" ..." if len(stripped) > 4 else ""))
    return kept, stripped


def stamp_rows(rungs: pd.DataFrame) -> pd.DataFrame:
    """The passing rungs as reagent rows for the batch stamp (`timeseries.stamping_frame`
    aux rows): m/z = the observed median, the isotopologue tag as `iso_label` so the
    79Br and 81Br rungs of one formula stay two tracks."""
    cols = ["mz", "role", "ion_formula", "iso_label", "neutral_formula", "adduct"]
    if rungs is None or not len(rungs):
        return pd.DataFrame(columns=cols)
    mz = pd.to_numeric(rungs["mz_obs"], errors="coerce").fillna(pd.to_numeric(rungs["mz"]))
    return pd.DataFrame({"mz": mz.to_numpy(), "role": "reagent", "ion_formula": rungs["ion_formula"].to_numpy(),
                         "iso_label": [t if t else None for t in rungs["iso_tag"]],
                         "neutral_formula": None, "adduct": None}, columns=cols)


def table(rungs: pd.DataFrame, stripped: pd.DataFrame) -> pd.DataFrame:
    """tables/reagent_water.csv: the passing rungs, each with the merged readings it
    displaced (`neutral adduct`, '; '-joined)."""
    if rungs is None or not len(rungs):
        return pd.DataFrame(columns=list(TABLE_COLUMNS))
    t = rungs.copy()
    lab = [f"{c}{('(' + g + ')') if g else ''}.(H2O){int(n)}" for c, g, n in zip(t["core"], t["iso_tag"], t["n"])]
    disp: dict[str, list] = {}
    if stripped is not None and len(stripped):
        for r, nf, ad in zip(stripped["rung"], stripped["neutral_formula"].astype(str), stripped["adduct"].astype(str)):
            disp.setdefault(r, []).append(f"{nf} {ad}")
    t["displaced"] = ["; ".join(disp.get(x, [])) for x in lab]
    return t[list(TABLE_COLUMNS)]


def measure(ts: pd.DataFrame | None, profile, *, tol_ppm: float, log=print) -> dict:
    """The batch's ladder for a reagent profile: {rungs, n_cores, segment_sizes,
    tol_ppm}. Empty rungs when the profile declares no water cores or there is no
    time series."""
    wc = tuple(getattr(profile, "water_cores", None) or ())
    pol = getattr(profile, "polarity", "-") or "-"
    out = {"rungs": detect(None, (), tol_ppm=tol_ppm), "n_cores": len(cores(wc, polarity=pol)), "tol_ppm": float(tol_ppm),
           "segment_sizes": {}}
    if not wc or ts is None or not len(ts):
        return out
    seg = segments(ts)
    out["segment_sizes"] = {str(k): int(v) for k, v in seg.value_counts().sort_index().items()}
    out["rungs"] = detect(ts, wc, tol_ppm=tol_ppm, polarity=pol)
    r = out["rungs"]
    log(f"[reagent-water] {len(r)} passing rung(s) over {out['n_cores']} core(s) in "
        f"{len(out['segment_sizes'])} segment(s) {out['segment_sizes']} at +-{tol_ppm:.2f} ppm"
        + (": " + ", ".join(f"{c}{('(' + g + ')') if g else ''} n<={int(g_['n'].max())}"
                            for (c, g), g_ in r.groupby(["core", "iso_tag"], sort=False)) if len(r) else ""))
    return out


def summary(rungs: pd.DataFrame, stripped: pd.DataFrame, *, n_cores: int, tol_ppm: float,
            segment_sizes: dict) -> dict:
    """batch_summary['merge_gates']['reagent_water']."""
    by_core: dict[str, list] = {}
    if rungs is not None and len(rungs):
        for c, g, n in zip(rungs["core"], rungs["iso_tag"], rungs["n"]):
            by_core.setdefault(f"{c}-" + (f" ({g})" if g else ""), []).append(int(n))
    return {"n_cores": int(n_cores), "tol_ppm": float(tol_ppm), "segments": segment_sizes,
            "n_rungs": int(len(rungs)) if rungs is not None else 0,
            "rungs_by_core": by_core,
            "n_stripped": int(len(stripped)) if stripped is not None else 0,
            "stripped": ([f"{a} {b} ({r})" for a, b, r in zip(stripped["neutral_formula"].astype(str),
                                                            stripped["adduct"].astype(str), stripped["rung"])]
                         if stripped is not None and len(stripped) else [])}
