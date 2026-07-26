"""Pass 4 -- the residual explainer.

Runs after Passes 1-3 + reagent labeling and attacks what is left, using
PATTERN evidence instead of chemical-prior filters:

  Stage A  Isotope-pair resolution. The residual of a halide-CIMS spectrum is
           dominated by ~1.998-Da doublets (Br/Cl isotope envelopes). Each
           doublet's light member becomes an M0 hypothesis whose ion MUST
           contain the implied halogen count; candidates come from the local
           grid; Mascope match_compounds validates the full isotope pattern.
           Explaining the light member automatically claims the heavy partner
           as the 81Br/37Cl child.

  Stage B  Deep series propagation. Bright residual peaks that sit 1-2 exact
           repeat/cluster units (CH2, O, H2O, CO, CO2, C2H2O, C2H4O2, HX) from
           an assigned anchor are proposed as series/cluster extensions and
           validated by Mascope.

Acceptance policy (user directive 2026-06-10):
  * |ppm| <= ppm_strict (1.0): accept on score alone.
  * ppm_strict < |ppm| <= ppm_pattern (4.0): accept ONLY with pattern evidence
    (Mascope-confirmed isotopologue partner, or >=2 supporting series anchors),
    capped at Low.
  * Plausibility filter is DBE-ONLY (integer DBE >= 0 + Senior's rule, enforced
    structurally by the grid). No H/C / O/C ratio gates in this pass --
    pattern evidence replaces the chemical priors.
  * Confidence is capped at Good (never High) -- pattern-driven assignments
    stay below the locked Pass-1 backbone.

All stages take an injectable `score_fn` (defaults to io_mascope.score_candidates)
so the acceptance logic is unit-testable offline.
"""
from __future__ import annotations

import bisect

import numpy as np
import pandas as pd

from peaky.chem import chemistry as C
from peaky.chem import contexts as X
from peaky.io import io_mascope as IO
from peaky.assignment import ledger as L
from peaky.assignment import series_gka as G
from peaky.assignment.passes import (PassConfig, arbitrate, confidence_label, z_of, _f,
                     _prefer_adduct_reading)

__version__ = "0.2.0"

# isotope spacings
D_PAIR_BR = 1.997795
D_PAIR_CL = 1.997050
D_13C = 1.0033548
R_13C = 0.0107   # 13C abundance per carbon


def carbon_count_from_13c(ledger: pd.DataFrame, peak_id, *, ppm: float = 6.0):
    """Measure a peak's carbon count from its 13C satellite, if present in the
    spectrum. Returns (c_lo, c_hi) bracketing the estimate (+/-1 carbon at high
    m/z where the satellite straddles two integers), or None when no usable
    satellite exists. This is the density-killer: a measured carbon count
    collapses the candidate grid ~5x before any scoring (v17 audit)."""
    try:
        i = ledger.index[ledger["peak_id"] == peak_id][0]
    except IndexError:
        return None
    mz0 = float(ledger.at[i, "mz"])
    h0 = float(ledger.at[i, "height"])
    if not np.isfinite(h0) or h0 <= 0:
        return None
    target = mz0 + D_13C
    tol = target * ppm * 1e-6
    d = (ledger["mz"] - target).abs()
    j = d.idxmin()
    if d.loc[j] > tol:
        return None
    hsat = float(ledger.at[j, "height"])
    if not np.isfinite(hsat) or hsat <= 0 or hsat >= h0:
        return None
    c_est = (hsat / h0) / R_13C
    # +/-1 carbon tolerance, widened a touch for the Poisson noise on a small
    # satellite; never let the floor drop below 1
    lo = max(1, int(np.floor(c_est)) - 1)
    hi = int(np.ceil(c_est)) + 1
    return (lo, hi)


# ---------------------------------------------------------------------------
# Stage A helpers
# ---------------------------------------------------------------------------
def find_iso_pairs(ledger: pd.DataFrame, *, ppm_tol: float = 8.0,
                   min_height: float = 0.0) -> pd.DataFrame:
    """Find ~1.998-Da doublets within the UNEXPLAINED residual.

    Returns DataFrame[light_pid, heavy_pid, light_mz, ratio, element, n_halogen]
    where element is 'Br' (ratio ~1 per Br) or 'Cl' (ratio ~0.32 per Cl) and
    n_halogen is the implied halogen count in the ION (1 or 2)."""
    un = ledger[(ledger["role"] == L.ROLE_UNEXPLAINED)
                & (ledger["height"].fillna(0) >= min_height)]
    rows = []
    mzs = un["mz"].to_numpy()
    order = np.argsort(mzs)
    ms = mzs[order]
    hs = un["height"].to_numpy()[order]
    pids = un["peak_id"].to_numpy()[order]
    claimed_heavy: set = set()

    def _best_near(i, delta):
        target = ms[i] + delta
        tol = target * ppm_tol * 1e-6
        j = bisect.bisect_left(ms, target - tol)
        best = None
        while j < len(ms) and ms[j] <= target + tol:
            if pids[j] not in claimed_heavy and j != i:
                if best is None or hs[j] > best[1]:
                    best = (j, hs[j])
            j += 1
        return best

    for i in range(len(ms)):
        if hs[i] <= 0:
            continue
        # mixed BrCl FIRST: its M+2 ratio (~1.29 = 0.973 + 0.320) falls inside
        # the single-Br acceptance band, so without this check a BrCl compound
        # is read as Br1 and the constrained enumeration fits NOTHING (the v22
        # CH2O-ladder dead end). Classification requires the diagnostic M+4
        # satellite (81Br37Cl, ~0.31 of M0) -- a plain Br1 has no M+4.
        b2 = _best_near(i, D_PAIR_BR)
        if b2 is not None:
            r2 = b2[1] / hs[i]
            b4 = _best_near(i, D_PAIR_BR + D_PAIR_CL)
            # the M+4 satellite IS the diagnostic (a plain Br1 has none);
            # M+2 gets a wide band because composite overlap drags it off the
            # textbook 1.29 (v23: real members at r2=0.94/1.17 were missed)
            if b4 is not None and 0.8 <= r2 <= 1.6 \
                    and 0.15 <= b4[1] / hs[i] <= 0.50:
                rows.append({"light_pid": pids[i], "heavy_pid": pids[b2[0]],
                             "light_mz": float(ms[i]), "ratio": float(r2),
                             "element": "BrCl", "n_halogen": 1,
                             "m4_pid": pids[b4[0]]})
                claimed_heavy.add(pids[b2[0]])
                claimed_heavy.add(pids[b4[0]])
                continue
        for delta, element in ((D_PAIR_BR, "Br"), (D_PAIR_CL, "Cl")):
            best = _best_near(i, delta)
            if best is None:
                continue
            ratio = best[1] / hs[i]
            n_hal = 0
            if element == "Br":
                if 0.6 <= ratio <= 1.4:
                    n_hal = 1
                elif 1.5 <= ratio <= 2.6:
                    n_hal = 2
            else:
                if 0.22 <= ratio <= 0.45:
                    n_hal = 1
                elif 0.5 <= ratio <= 0.85:
                    n_hal = 2
            if n_hal:
                rows.append({"light_pid": pids[i], "heavy_pid": pids[best[0]],
                             "light_mz": float(ms[i]), "ratio": float(ratio),
                             "element": element, "n_halogen": n_hal,
                             "m4_pid": None})
                claimed_heavy.add(pids[best[0]])
                break   # one element interpretation per light peak
    return pd.DataFrame(rows)


def _halogen_count_in_ion(neutral: str, adduct: str, element: str) -> int:
    n = C.parse_formula(neutral).get(element, 0)
    if element == "Br" and "Br" in adduct:
        n += 1
    if element == "Cl" and "Cl" in adduct:
        n += 1
    return n


def candidates_for_pair(light_mz: float, element: str, n_halogen: int,
                        adducts: list[str], *, ranges: dict,
                        ppm: float) -> set[str]:
    """Grid candidates for a pair's light member: the ion must contain exactly
    n_halogen atoms of `element` ('BrCl' = exactly one Br AND one Cl in the
    ion). DBE-only filtering (grid-structural)."""
    out: set[str] = set()
    for adduct in adducts:
        if adduct not in C.ADDUCT_SHIFTS:
            continue
        r = dict(ranges)
        if element == "BrCl":
            need_br = 1 - (1 if "Br" in adduct else 0)
            r["Br"] = (need_br, need_br)
            r["Cl"] = (1, 1)
            for f in C.candidates_for_peaks([light_mz], r, [adduct],
                                            ppm_tolerance=ppm):
                cnt = C.parse_formula(f)
                if cnt.get("Br", 0) == need_br and cnt.get("Cl", 0) == 1 \
                        and not (cnt.get("F", 0) >= 1 and cnt.get("O", 0) > 6):
                    out.add(f)
            continue
        # halogen needed in the NEUTRAL given this adduct
        need = n_halogen - (1 if element in adduct else 0)
        if need < 0:
            continue
        r[element] = (need, need)
        for f in C.candidates_for_peaks([light_mz], r, [adduct], ppm_tolerance=ppm):
            cnt = C.parse_formula(f)
            if cnt.get(element, 0) != need:
                continue
            # fluorochemical oxygen cap (same rule as the pass-3 family): a
            # fluorinated candidate with O>6 is mass-fit junk -- v19 showed the
            # F-enabled pair grid producing C9H13ClF2O16-class monsters that
            # DBE-only filtering cannot stop.
            if cnt.get("F", 0) >= 1 and cnt.get("O", 0) > 6:
                continue
            out.add(f)
    return out


def _accept(score: float, ppm: float | None, has_pattern: bool,
            cfg: PassConfig) -> tuple[bool, str]:
    """The Pass-4 acceptance rule. Returns (accept, reason).

    When the run is self-calibrated (cfg.cal_mu/cal_sigma fitted on the pass-1
    backbone) the strict/pattern bands are calibrated z-scores; otherwise the
    static residual_ppm_* windows apply."""
    if score is None or score < cfg.tau_suspect:
        return False, "score below floor"
    if ppm is None or pd.isna(ppm):
        return False, "no mass error reported"
    z = z_of(ppm, cfg)
    if z is not None:
        if z <= cfg.cal_z_accept:
            return True, f"z={z:.1f} within calibrated accuracy"
        if z <= cfg.cal_z_pattern and has_pattern:
            return True, f"z={z:.1f} > accept band but pattern evidence corroborates"
        return False, f"z={z:.1f} without pattern evidence"
    a = abs(ppm)
    if a <= cfg.residual_ppm_strict:
        return True, f"|ppm|={a:.2f} within strict tolerance"
    if a <= cfg.residual_ppm_pattern and has_pattern:
        return True, f"|ppm|={a:.2f} > strict but pattern evidence corroborates"
    return False, f"|ppm|={a:.2f} without pattern evidence"


def _cap_conf(conf: str) -> str:
    return conf.replace("High", "Good")


# ---------------------------------------------------------------------------
# Stage A -- isotope-pair resolution
# ---------------------------------------------------------------------------
def stage_a_iso_pairs(client, sample_id: str, ledger: pd.DataFrame, profile,
                      pre, cfg: PassConfig, adducts: list[str], *,
                      score_fn=None, log=print) -> dict:
    score_fn = score_fn or IO.score_candidates
    out = {"committed": 0, "locked": 0, "iso_attached": 0}
    pairs = find_iso_pairs(ledger, min_height=cfg.height_cutoff)
    if len(pairs) == 0:
        log("[pass4.A] no isotope pairs in residual")
        return out
    log(f"[pass4.A] {len(pairs)} isotope pairs in residual "
        f"({(pairs.element == 'Br').sum()} Br-like, {(pairs.element == 'Cl').sum()} Cl-like)")
    cmax = max(12, (pre.estimated_max_C + 4) if getattr(pre, "estimated_max_C", 0) else 40)
    cmax = min(cmax, 40)
    base_ranges = {"C": (1, cmax), "H": (0, 2 * cmax + 4), "O": (0, 30),
                   "N": (0, profile.max_N), "S": (0, min(1, profile.max_S))}
    # F is earned by SAMPLE evidence, never default: when the residual carries
    # significant CF2/C2F4 chains (decoy-validated), pair enumeration may use
    # F. The Br/Cl doublet pins the halogen count and the z-gate + arbitration
    # margins bound the density risk. Without this, a chain HEAD with no tail
    # peak is invisible to both pass 3 (not a chain member) and pass 4 (no F
    # in the grid) -- which is how TFA.Br- at 192.9116 (-0.8 ppm, textbook
    # 0.978 doublet) sat unexplained from v13 through v18.
    f_evidence = False
    try:
        from peaky.assignment import series_detect as SD
        ev = SD.detect_series(ledger, ppm=5.0)
        f_evidence = bool(len(ev) and ((ev["significant"])
                                       & (ev["action"] == "fluorinated")).any())
    except Exception:
        pass
    if f_evidence:
        # F is enabled ONLY for carbon-CLAMPED pairs (below). Adding F(0,17) to the
        # WIDE-carbon grid is a combinatorial blow-up: each pair's ranges differ
        # (per-pair halogen count + clamp) so the grid cache misses and rebuilds a
        # ~20M-formula grid per pair -- on a sample with many unclamped pairs this
        # spins for many minutes (a dense Br- batch: 68 pairs). The legitimate fluorinated
        # finds (e.g. TFA.Br- @192.9116) carry a 13C satellite -> they ARE clamped,
        # so F where C is pinned (tiny grid) keeps them; F on the wide grid only
        # produces the C9H13ClF2O16-class mass-fit monsters the O>6 cap exists for.
        log("[pass4.A] fluorinated chain evidence -> F enabled for carbon-clamped "
            "pairs only (F x wide-carbon grid is a combinatorial blow-up)")
    # enumerate per pair, pooled scoring. CARBON CLAMP: when the light peak has
    # a measured 13C satellite, restrict C to the measured count -- this both
    # shrinks the grid ~5x and turns "closest fit" into "consistent with the
    # observed carbon number" (v17 audit: the clamp is what makes the residual
    # defensibly assignable instead of mass-fit guessing).
    cand_by_formula: dict[str, list] = {}
    n_clamped = 0
    for _, p in pairs.iterrows():
        ranges = dict(base_ranges)
        clamp = carbon_count_from_13c(ledger, p["light_pid"])
        if clamp is not None:
            ranges["C"] = clamp
            n_clamped += 1
            if f_evidence:
                ranges["F"] = (0, 17)         # F only where carbon is pinned
        cands = candidates_for_pair(p["light_mz"], p["element"], p["n_halogen"],
                                    adducts, ranges=ranges,
                                    ppm=cfg.residual_ppm_pattern)
        for f in cands:
            cand_by_formula.setdefault(f, []).append(p["light_pid"])
    if not cand_by_formula:
        log("[pass4.A] no grid candidates for any pair")
        return out
    log(f"[pass4.A] {len(cand_by_formula)} candidate formulas "
        f"(DBE-only filter; {n_clamped}/{len(pairs)} pairs carbon-clamped)")
    scored = score_fn(client, sample_id, sorted(cand_by_formula), mechanism_ids=cfg.mechanism_ids)
    if scored is None or len(scored) == 0:
        # empty scoring is a SERVER failure (500/timeout swallowed by the
        # resilient IO layer), NOT "nothing to assign". Make it loud -- a
        # silent zero-commit pass looks identical to a clean residual (v17).
        log(f"[pass4.A] WARNING scoring returned EMPTY for "
            f"{len(cand_by_formula)} candidates -- server likely degraded; "
            f"residual left UNASSIGNED (not a clean result)")
        out["scoring_empty"] = True
        return out
    arb = arbitrate(scored, cfg)
    win = arb.get("winners", pd.DataFrame())
    kids = arb.get("iso_children", pd.DataFrame())
    pair_by_light = {p["light_pid"]: p for _, p in pairs.iterrows()}
    for _, w in win.iterrows():
        pid = w["peak_id"]
        if pid not in pair_by_light:
            continue   # only commit onto pair light members in this stage
        p = pair_by_light[pid]
        # pattern evidence = Mascope-confirmed satellite OR the OBSERVED doublet
        # that anchored this peak (the heavy partner sits in the residual,
        # unmatched, so Mascope's n_iso is 0 -- but we measured the doublet
        # ourselves; ignoring it forced the strict gate and killed every real
        # Br-adduct peak, v17). A carbon-clamped fit is also pattern-supported.
        has_pattern = (w["n_iso"] >= 1 or p["n_halogen"] >= 1
                       or carbon_count_from_13c(ledger, pid) is not None)
        ok, why = _accept(w["raw_score"], w["ppm_error"], has_pattern, cfg)
        if not ok:
            continue
        try:
            if L.role_of(ledger, pid) != L.ROLE_UNEXPLAINED:
                continue
            conf = _cap_conf(confidence_label(
                w["raw_score"], w["ppm_error"], w["n_iso"], w["tied"], cfg,
                suffix="iso-pair"))
            if conf == "Reject":
                continue
            # reagent-halogen reading: a covalent Y(Br) [M+Br]- iso-pair winner
            # is the SAME ion as the Br-free SOA core Y' [M+HBr+Br]- (di-bromide
            # reagent cluster). Prefer the cluster reading so the reported
            # neutral is bromine-free (e.g. 409.0015 C15H23BrO3 -> C15H22O3
            # [M+HBr+Br]-), consistent with passes 1/3/5.
            w = _prefer_adduct_reading(w, cfg)
            note = w["_relabel_note"] if "_relabel_note" in w else ""
            L.commit_assignment(
                ledger, pid, neutral_formula=w["neutral"], adduct=w["adduct"],
                ion_formula=w["ion_formula"], ion_score=w["ion_score"],
                compound_score=w["compound_score"], ppm_error=w["ppm_error"],
                eff_score=w.get("eff_score"), eff_margin=w.get("eff_margin"),
                tied=w.get("tied"),
                pass_no=4, method="residual:iso-pair", confidence=conf,
                commentary=(f"Pass 4 (iso-pair): {p['element']} doublet "
                            f"(ratio {p['ratio']:.2f}, n_{p['element']}="
                            f"{p['n_halogen']} in ion) anchors this peak; "
                            f"{w['neutral']} {w['adduct']} scored "
                            f"{w['raw_score']:.2f} by Mascope. Accepted: {why}. "
                            f"DBE-only plausibility (no ratio filters).{note}"),
                alternatives=w["alternatives"])
            out["committed"] += 1
            # attach the heavy partner: prefer Mascope's own attribution,
            # fall back to the observed-pair labeling
            attached = False
            if len(kids):
                mine = kids[(kids["parent_peak_id"] == pid)]
                for _, k in mine.iterrows():
                    try:
                        L.attach_isotopologue(ledger, k["peak_id"], pid,
                                              iso_label=k["iso_label"],
                                              iso_match_score=k["iso_score"])
                        out["iso_attached"] += 1
                        if k["peak_id"] == p["heavy_pid"]:
                            attached = True
                    except L.LedgerError:
                        continue
            if not attached:
                try:
                    lab = {"Br": "81Br", "Cl": "37Cl",
                           "BrCl": "81Br/37Cl"}[p["element"]]
                    L.attach_isotopologue(ledger, p["heavy_pid"], pid,
                                          iso_label=lab + "(pair)")
                    out["iso_attached"] += 1
                except L.LedgerError:
                    pass
            # mixed-halogen pairs also own their M+4 satellite (81Br37Cl)
            if p.get("m4_pid") is not None:
                try:
                    L.attach_isotopologue(ledger, p["m4_pid"], pid,
                                          iso_label="81Br37Cl(pair)")
                    out["iso_attached"] += 1
                except L.LedgerError:
                    pass
        except L.LedgerError:
            continue
    log(f"[pass4.A] {out}")
    return out


# ---------------------------------------------------------------------------
# Stage B -- deep series / cluster propagation
# ---------------------------------------------------------------------------
def stage_b_series(client, sample_id: str, ledger: pd.DataFrame, profile,
                   cfg: PassConfig, adducts: list[str], *, reagent: str | None,
                   score_fn=None, log=print) -> dict:
    score_fn = score_fn or IO.score_candidates
    out = {"committed": 0, "locked": 0, "iso_attached": 0}
    anchors = set(ledger.loc[ledger["role"] == L.ROLE_M0, "neutral_formula"].dropna())
    if not anchors:
        return out
    units = tuple(G.ORGANIC_UNITS) + ("C2H4O2",)
    if reagent in ("Br", "Cl"):
        units = units + ("H" + reagent,)
    un = ledger[(ledger["role"] == L.ROLE_UNEXPLAINED)
                & (ledger["height"].fillna(0) >= cfg.height_cutoff)]
    proposals: dict[str, dict] = {}
    for _, prow in un.iterrows():
        for prop in G.propose_for_peak(prow["mz"], anchors, adducts, units=units,
                                       ppm=cfg.residual_ppm_pattern,
                                       max_steps=cfg.residual_max_steps):
            f = prop.neutral_formula
            # This stage applied only the STRUCTURAL gates (DBE + oxygen cap), never
            # the context's chemical plausibility, so it could walk a heteroatom the
            # context forbids into a neutral. Pass-0 known species are legitimate
            # ANCHORS but must not be springboards: on the first iodide batch it
            # extrapolated +CO / +C2H2O / +O off HOI / HIO2 / INO3 into CHIO2, CHIO3,
            # C2H3IO2, C2H3IO3 and INO4 -- each really the I2 cluster of an acid
            # already in the ledger (CHIO2 [M+I]- == [HCOOH-H+I2]-) and
            # un-confirmable, 127I being monoisotopic. ambient-air's max_I=0 rejects
            # them here now, exactly as max_F/max_P=0 handle F and P.
            keep, _why = X.filter_by_context(f, profile.label)
            if not keep:
                continue
            ok, _why = C.dbe_ok(f)        # structural gates: DBE + oxygen cap
            if ok:
                ok, _why = C.oxygen_ok(f)
            if not ok:
                continue
            cur = proposals.get(f)
            if cur is None or prop.n_supporting_anchors > cur["support"]:
                proposals[f] = {"support": prop.n_supporting_anchors,
                                "anchor": prop.anchor_formula,
                                "unit": prop.unit, "steps": prop.n_steps}
    proposals = {f: v for f, v in proposals.items() if f not in anchors}
    if not proposals:
        log("[pass4.B] no series proposals")
        return out
    log(f"[pass4.B] {len(proposals)} deep-series proposals (<= {cfg.residual_max_steps} steps)")
    scored = score_fn(client, sample_id, sorted(proposals), mechanism_ids=cfg.mechanism_ids)
    if scored is None or len(scored) == 0:
        log(f"[pass4.B] WARNING scoring returned EMPTY for {len(proposals)} "
            f"proposals -- server likely degraded; residual left UNASSIGNED")
        out["scoring_empty"] = True
        return out
    arb = arbitrate(scored, cfg)
    for _, w in arb.get("winners", pd.DataFrame()).iterrows():
        f = w["neutral"]
        meta = proposals.get(f)
        if meta is None:
            continue
        has_pattern = (w["n_iso"] >= 1) or (meta["support"] >= 2)
        ok, why = _accept(w["raw_score"], w["ppm_error"], has_pattern, cfg)
        if not ok:
            continue
        pid = w["peak_id"]
        try:
            if L.role_of(ledger, pid) != L.ROLE_UNEXPLAINED:
                continue
            conf = _cap_conf(confidence_label(
                w["raw_score"], w["ppm_error"], w["n_iso"], w["tied"], cfg,
                suffix="deep-series"))
            if conf == "Reject":
                continue
            L.commit_assignment(
                ledger, pid, neutral_formula=f, adduct=w["adduct"],
                ion_formula=w["ion_formula"], ion_score=w["ion_score"],
                compound_score=w["compound_score"], ppm_error=w["ppm_error"],
                eff_score=w.get("eff_score"), eff_margin=w.get("eff_margin"),
                tied=w.get("tied"),
                pass_no=4, method="residual:series", confidence=conf,
                commentary=(f"Pass 4 (deep series): {meta['steps']:+d}x"
                            f"{meta['unit']} from anchor {meta['anchor']} "
                            f"({meta['support']} supporting anchors); scored "
                            f"{w['raw_score']:.2f}. Accepted: {why}."),
                alternatives=w["alternatives"], series_unit=meta["unit"])
            out["committed"] += 1
        except L.LedgerError:
            continue
    log(f"[pass4.B] {out}")
    return out


def explain_residual(client, sample_id: str, ledger: pd.DataFrame, profile,
                     pre, cfg: PassConfig, adducts: list[str], *,
                     reagent: str | None = None, score_fn=None,
                     log=print) -> dict:
    a = stage_a_iso_pairs(client, sample_id, ledger, profile, pre, cfg, adducts,
                          score_fn=score_fn, log=log)
    b = stage_b_series(client, sample_id, ledger, profile, cfg, adducts,
                       reagent=reagent, score_fn=score_fn, log=log)
    merged = {}
    for k in set(a) | set(b):
        merged[k] = (a.get(k, 0) or 0) + (b.get(k, 0) or 0)
    return merged


def characterize_residual(ledger: pd.DataFrame, *, min_height: float = 0.0) -> pd.DataFrame:
    """Describe every UNEXPLAINED peak by what the isotope pattern DOES tell us,
    even when no formula is assignable. This is the honest answer to 'why is
    this peak unexplained': it records the measured carbon count (13C), the
    halogen signature (Br/Cl doublet), whether the peak is itself a heavy
    isotope twin of a lighter residual peak, and a coarse tier. Tiers:
      - 'iso-partner'    : a 13C/81Br/37Cl satellite of a lighter residual peak
                           (explained as a satellite; no independent formula)
      - 'has-constraints': carbon and/or halogen count measured -> a constrained
                           solve is possible (feeds the Pass-4 clamp)
      - 'isolated'       : bright, no measurable isotope structure (genuine
                           single peak; needs orthogonal evidence e.g. time-series)
    """
    un = ledger[(ledger["role"] == L.ROLE_UNEXPLAINED)
                & (ledger["height"].fillna(0) >= min_height)].copy()
    if "synthetic" in un.columns:   # composite co-component sub-peaks aren't raw residual
        un = un[~un["synthetic"].fillna(False).astype(bool)]
    mzs = ledger["mz"].to_numpy()
    hs = ledger["height"].to_numpy()

    def near(target, ppm=6.0):
        tol = target * ppm * 1e-6
        i = int(np.abs(mzs - target).argmin())
        return i if abs(mzs[i] - target) <= tol else None

    rows = []
    for _, r in un.iterrows():
        mz, h = float(r["mz"]), float(r["height"])
        clamp = carbon_count_from_13c(ledger, r["peak_id"])
        # heavy-twin of a lighter residual peak?
        twin_of = None
        for d, lab in ((D_13C, "13C"), (D_PAIR_BR, "81Br"), (D_PAIR_CL, "37Cl")):
            j = near(mz - d)
            if j is not None and hs[j] > 0 and h < hs[j] * 1.45:
                twin_of = lab
                break
        # halogen doublet to the RIGHT (this peak is the light member)?
        nbr = ncl = 0
        jb = near(mz + D_PAIR_BR)
        if jb is not None and h > 0:
            rt = hs[jb] / h
            nbr = 1 if 0.55 <= rt <= 1.45 else (2 if 1.45 < rt <= 2.6 else 0)
        jc = near(mz + D_PAIR_CL)
        if jc is not None and h > 0 and nbr == 0:
            rt = hs[jc] / h
            ncl = 1 if 0.18 <= rt <= 0.5 else 0
        if twin_of:
            tier = "iso-partner"
        elif clamp or nbr or ncl:
            tier = "has-constraints"
        else:
            tier = "isolated"
        rows.append({
            "peak_id": r["peak_id"], "mz": round(mz, 4), "height": int(h),
            "tier": tier, "twin_of": twin_of,
            "c_count": None if not clamp else f"{clamp[0]}-{clamp[1]}",
            "n_Br": nbr, "n_Cl": ncl})
    cols = ["peak_id", "mz", "height", "tier", "twin_of", "c_count", "n_Br", "n_Cl"]
    return (pd.DataFrame(rows, columns=cols)
            .sort_values("height", ascending=False))
