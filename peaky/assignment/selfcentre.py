"""The file's own scoring centre when nothing else measured one (the self-centre).

A sample's candidates are scored against its mass offset (io_mascope.scoring_for_sample):
the median of its Mascope anchors when at least MIN_OFFSET_ANCHORS matched; zero on peaks
the batch's axis correction moved; else an ASSUMED zero. An anchorless Orbitrap file the
batch's axis model did not reach is then judged at 0 ppm wherever it sits. At the class
width (0.58 ppm) a correct ion 0.6 ppm off already misses Good and one 1 ppm off misses
the pass-1 floor: on an anchorless NO+ file at -0.5 ppm two thirds of the rows read
Low, and bright CHO lines stayed unexplained in every such file while the anchored
files of the same batch assigned them. The pass-1 calibration cannot see it either:
its backbone is the rows pass 1 committed, which the zero-centred score had already cut
to |error| <~ 0.5 ppm.

`measure` estimates the file's centre BEFORE pass 0, from a backbone chosen without the
mass term:

  * the bright peaks: unexplained (reagent ions and side lobes are already marked), at
    least MIN_X_EDGE x the file's noise edge and under BRIGHT_X x its median height (the
    brightest lines -- reagent and lock-mass-like ions -- sit off the analyte axis), the
    MAX_PEAKS brightest;
  * enumerated as pass 1 enumerates (the CHON grid of the run's context, its context
    filter, its channels, +-search_ppm around the raw m/z) and scored by the local scorer,
    reading the PATTERN-ONLY score (`ion_score_massfree`), which ignores the mass error;
  * a peak is a backbone point when exactly ONE ion formula's pattern holds there
    (pattern-only score >= tau_suspect; every other reading is struck by its own isotope
    lines), and that reading is CHO / CHON with C >= 1, pattern-Good (>= tau_good) and
    13C-confirmed: its 13C line matched in ratio, within C13_PPM of the M0's own error.

The centre is the backbone's median raw error. It is accepted with at least MIN_N points
whose scaled MAD is at most MAX_SPREAD_PPM (a curved axis no constant follows is not
centred) and whose bulk (centre +- 2 spreads) lies inside the enumeration window (a
centre near the window's edge is truncated); it is applied when it sits at least
MIN_SHIFT_PPM from zero. The file is then scored once at it: assign.run registers it
(io_mascope.set_self_centre) before pass 0, every line of every pass is judged against
it, and the sample's scoring snapshot says `mu_source` "self" with the measurement under
`self_centre`. The pass-0 known-species gate keeps its own centre (`prior_offset`).

Only an Orbitrap-class file scored locally whose snapshot reads `assumed_zero` is
measured: a file with its own anchors, one the batch's axis reached, a stand-in judged at
an inherited snapshot (a decoy arm), the trace-first sample and a TOF keep their centre.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__version__ = "0.1.0"

#: a backbone peak is at least this many times the file's noise edge ...
MIN_X_EDGE = 10.0
#: ... and under this many times the file's median peak height (massqc.BRIGHT_X)
BRIGHT_X = 100.0
#: the brightest peaks tried
MAX_PEAKS = 150
#: backbone points needed before a centre is stated
MIN_N = 8
#: the backbone's scaled MAD (ppm) above which no constant centre is stated
MAX_SPREAD_PPM = 0.6
#: a centre this close to zero (ppm) is not applied: the file stays at zero
MIN_SHIFT_PPM = 0.1
#: a 13C line confirms the reading when its own error is within this of the M0's (ppm)
C13_PPM = 1.5

_BACKBONE = {"C", "H", "N", "O"}


def eligibility(snapshot: dict | None, cfg, *, inherited: bool, local: bool) -> str | None:
    """None when this file may measure its own centre, else why not (recorded):
    switched off, the network scorer (it judges the server's own offset), the
    trace-first sample, not an Orbitrap-class file, an inherited scoring trend,
    or a centre measured otherwise (`mu_source` other than "assumed_zero")."""
    if not getattr(cfg, "self_centre", True):
        return "off (PassConfig.self_centre=False)"
    if not local:
        return "the network scorer judges the server's own offset"
    if getattr(cfg, "trace_sample", False):
        return "the trace-first sample"
    snap = snapshot or {}
    if getattr(cfg, "instrument_class", None) != "orbitrap" or str(snap.get("instrument_type") or "") != "orbi":
        return "not an Orbitrap-class file"
    if inherited:
        return "scored at an inherited mass trend"
    if snap.get("mu_source") != "assumed_zero":
        return f"centre measured otherwise (mu_source {snap.get('mu_source')})"
    return None


def _cho_chon(formula: str) -> bool:
    from peaky.chem import chemistry as C
    cnt = {k: v for k, v in C.parse_formula(str(formula)).items() if v}
    return bool(cnt) and set(cnt) <= _BACKBONE and cnt.get("C", 0) >= 1


def _ion_key(ion) -> tuple:
    """An ion formula's composition, order-free (two readings of one ion are one ion)."""
    from peaky.chem import chemistry as C
    s = str(ion).rstrip("+-.")
    return tuple(sorted((k, v) for k, v in C.parse_formula(s).items() if v))


def _bright(led: pd.DataFrame, cfg) -> pd.DataFrame:
    from peaky.assignment import ledger as L
    un = L.unassigned_peaks(led)
    if not len(un) or "height" not in un.columns:
        return un.iloc[0:0]
    h = pd.to_numeric(un["height"], errors="coerce")
    keep = h.notna() & (h > 0)
    edge = getattr(cfg, "noise_edge_cps", None)
    if edge is not None and np.isfinite(edge) and edge > 0:
        keep &= h >= MIN_X_EDGE * float(edge)
    allh = pd.to_numeric(led["height"], errors="coerce")
    allh = allh[allh > 0]
    if len(allh):
        keep &= h <= BRIGHT_X * float(allh.median())
    return un[keep].assign(_h=h[keep]).sort_values(["_h", "mz"], ascending=[False, True]).head(MAX_PEAKS)


def backbone(scored: pd.DataFrame, peak_ids, cfg) -> pd.DataFrame:
    """The backbone points of `scored` (the local scorer's flat table) on the peaks
    `peak_ids`: one row per peak (`peak_id`, `ion_formula`, `neutral_formula`,
    `ppm`, `ion_score_massfree`) where exactly one ion formula's pattern holds, it
    is a CHO / CHON reading with C >= 1, pattern-Good and its 13C line confirms it.
    No mass term is read anywhere."""
    cols = ["peak_id", "ion_formula", "neutral_formula", "ppm", "ion_score_massfree"]
    if scored is None or not len(scored) or "ion_score_massfree" not in scored.columns:
        return pd.DataFrame(columns=cols)
    ids = set(peak_ids)
    base = scored[scored["is_base"].astype(bool) & scored["sample_peak_id"].isin(ids)].copy()
    base["_mf"] = pd.to_numeric(base["ion_score_massfree"], errors="coerce").fillna(0.0)
    c13 = scored[(scored["iso_label"] == "13C") & scored["sample_peak_id"].notna()]
    c13_ppm = {(r.compound_formula, r.mechanism_id): float(r.ppm_error)
               for r in c13.itertuples() if r.ppm_error is not None and pd.notna(r.ppm_error)}
    base["_ion"] = base["ion_formula"].map(_ion_key)
    rows = []
    for pid, g in base.groupby("sample_peak_id", sort=True):
        live = g[g["_mf"] >= cfg.tau_suspect]
        if live["_ion"].nunique() != 1:
            continue
        live = live.sort_values(["_mf", "compound_formula", "mechanism_id"], ascending=[False, True, True])
        best = live.iloc[0]
        if best["_mf"] < cfg.tau_good or best["ppm_error"] is None or pd.isna(best["ppm_error"]):
            continue
        ppm = float(best["ppm_error"])
        confirmed = [r for r in live.itertuples()
                     if _cho_chon(r.compound_formula)
                     and abs(c13_ppm.get((r.compound_formula, r.mechanism_id), np.inf) - ppm) <= C13_PPM]
        if not confirmed:
            continue
        rows.append({"peak_id": pid, "ion_formula": best["ion_formula"],
                     "neutral_formula": confirmed[0].compound_formula, "ppm": ppm,
                     "ion_score_massfree": float(best["_mf"])})
    return pd.DataFrame(rows, columns=cols)


def centre_of(points: pd.DataFrame, mzs, cfg) -> dict:
    """The centre the backbone `points` states (see the module note): a record with
    `applied`, `mu_ppm` (None when no centre is stated), `n`, `spread_ppm`, the
    backbone's m/z span and `why` when nothing is applied."""
    n = int(len(points))
    rec = {"applied": False, "mu_ppm": None, "n": n, "spread_ppm": None,
           "mz_lo": None, "mz_hi": None, "why": None}
    if n:
        mz = np.asarray(mzs, dtype=float)
        rec["mz_lo"], rec["mz_hi"] = round(float(mz.min()), 4), round(float(mz.max()), 4)
    if n < MIN_N:
        rec["why"] = f"a backbone of {n} unique 13C-confirmed CHO/CHON ions (needs {MIN_N})"
        return rec
    e = points["ppm"].to_numpy(dtype=float)
    mu = float(np.median(e))
    spread = float(1.4826 * np.median(np.abs(e - mu)))
    rec["spread_ppm"] = round(spread, 3)
    if spread > MAX_SPREAD_PPM:
        rec["why"] = (f"the backbone scatters {spread:.2f} ppm (scaled MAD), over {MAX_SPREAD_PPM:g}: "
                      "no constant centre")
        return rec
    rec["mu_ppm"] = round(mu, 3)
    window = float(getattr(cfg, "search_ppm", 3.0))
    if abs(mu) + 2 * spread > window:
        rec["why"] = (f"the centre {mu:+.2f} ppm sits within 2 spreads of the {window:g} ppm search "
                      "window's edge: truncated, not stated")
        rec["mu_ppm"] = None
        return rec
    if abs(mu) < MIN_SHIFT_PPM:
        rec["why"] = f"within {MIN_SHIFT_PPM:g} ppm of zero: scored at zero"
        return rec
    rec["applied"] = True
    return rec


def measure(client, sample_id: str, led: pd.DataFrame, profile, pre, cfg, adducts, *, log=print) -> dict:
    """The file's self-centre record (`centre_of`, plus `n_peaks` tried); reads the
    ledger and the sample's peaks, changes neither."""
    from peaky.assignment.passes import core as PC
    from peaky.assignment.passes import directors as D
    from peaky.chem import chemistry as C
    from peaky.io import io_mascope as IO

    tgt = _bright(led, cfg)
    gadducts = [a for a in adducts if a in C.ADDUCT_SHIFTS and a not in PC.ABSTRACTION_ADDUCTS]
    scored = None
    if len(tgt) and gadducts:
        ranges = D.build_ranges(profile, pre, include_N=True)
        forms = C.candidates_for_peaks(tgt["mz"].tolist(), ranges, gadducts, ppm_tolerance=cfg.search_ppm)
        forms = D._context_filter(forms, profile)
        if forms:
            scored = IO.score_candidates(client, sample_id, sorted(forms), mechanism_ids=cfg.mechanism_ids)
    pts = backbone(scored, tgt["peak_id"].tolist() if len(tgt) else [], cfg)
    mzs = tgt.set_index("peak_id").loc[pts["peak_id"], "mz"].to_numpy(dtype=float) if len(pts) else []
    rec = centre_of(pts, mzs, cfg)
    rec["n_peaks"] = int(len(tgt))
    if rec["applied"]:
        log(f"[self-centre] {rec['n']} unique 13C-confirmed CHO/CHON ions of {rec['n_peaks']} bright peaks "
            f"(m/z {rec['mz_lo']:.0f}-{rec['mz_hi']:.0f}) sit at {rec['mu_ppm']:+.3f} ppm "
            f"(scaled MAD {rec['spread_ppm']:.3f}): the file is scored at it from pass 0")
    else:
        log(f"[self-centre] not applied ({rec['n_peaks']} bright peaks tried): {rec['why']}")
    return rec
