"""Post-assignment residual cleanup (single-file, runs after pass 6).

Three honest reclassifications of the leftover 'unexplained' residual, none of
which invent chemistry the spectrum doesn't support:

  1. flag_ringing_artifacts -- a weak peak sitting within a few mDa of a MUCH
     brighter peak is a peak-shape / ringing / shoulder artifact of that bright
     ion, not a real ion. Role -> 'artifact'.

  2. label_bromide_clusters -- a strongly negative mass-defect peak (defect <
     -0.16 => >=2 reagent Br) that carries a Br isotope partner and has no sane
     covalent reading is a bromide reagent-cluster ion. Role -> 'reagent'.

  3. recover_isotope_gated -- the genuine molecules the score gate dropped:
     enumerate ONLY low-complexity CHO/CHON/CHOS (+<=1 covalent Br/Cl), and
     commit a winner ONLY when the measured halogen isotope pattern CONFIRMS the
     candidate's halogen count (an independent corroboration that justifies the
     relaxed score floor). The halogen-isotope requirement + complexity clamp is
     what keeps the N2S2 / O-monster mass-fit flood out.
"""
from __future__ import annotations

import bisect
import json

import numpy as np
import pandas as pd

from peaky.chem import chemistry as C
from peaky.chem import contexts as X
from peaky.assignment import degeneracy as D
from peaky.io import io_mascope as IO
from peaky.assignment import ledger as L

__version__ = "0.5.0"   # + reclaim_satellites now covers 15N/34S/29Si/30Si/18O (not just 13C/81Br/37Cl)
# (history) v43-review fixes: ringing brightness floor (H4),
                        # CHO-only isotope-confirmed recovery (H1/H2), covalent
                        # cluster commentary (M4)

BR = 1.99795          # 81Br - 79Br
C13 = 1.003355

# ---- 1. ringing / shoulder artifacts -------------------------------------
# FT ringing / detector-shoulder satellites only appear next to a SATURATING
# ion, and their amplitude scales UP with the parent intensity. So a real
# artifact needs (a) the parent to be genuinely bright (>= MIN_PARENT cps -- a
# dim 6k-cps peak cannot ring), (b) a large brightness factor (>= the floor, so
# the satellite is <~1% of the parent), and (c) sub-resolution proximity. These
# three gates are what separate a true sidelobe from a resolved independent ion
# that merely happens to sit nearby (the 273.05 false flag, parent only 6385 cps
# at 43x and 2.3% -- physics runs backwards, v43 review H4).
MIN_RING_PARENT = 50000.0   # cps; below this a peak is too weak to ring
RING_FACTOR = 100.0         # parent must be >= this x brighter (=> satellite <1%)


def flag_ringing_artifacts(ledger: pd.DataFrame, *, factor: float = RING_FACTOR,
                           dmz: float = 0.012, min_parent: float = MIN_RING_PARENT,
                           log=print) -> dict:
    """Mark an unexplained peak as 'artifact' when a SATURATING peak (>= min_parent
    cps) that is >=`factor`x brighter sits within `dmz`. The tight dmz + the
    brightness floor + the high factor together ensure only sub-resolution
    sidelobes of a saturating ion are flagged, never a resolved independent
    neighbour.

    ⚠ UNEXPLAINED PEAKS ONLY -- AND DELIBERATELY SO. A pass that already committed
    an M0 onto a sidelobe makes it invisible here (this runs post-pass-6), which
    looks like an obvious bug to fix by also displacing committed M0s. DO NOT: it
    was measured and it is catastrophic. The static evidence available at this
    point -- Δm/z, satellite fraction, brightness ratio -- CANNOT distinguish a
    sidelobe from a real ion that merely sits beside a bright peak. Over the
    TC2026 campaign the rule "displace an unlocked, uncorroborated M0 within dmz of
    a >=factor x parent" selects 74 commits across 30 runs, and scoring every one
    of them against the time-series oracle (a sidelobe holds a ~constant ratio to
    its parent; ratio-cv < 0.06) gives **0/53 correct, 51 false positives** --
    C12H16O6, C13H29NO9, C4H8N2P2S, the siloxanes, all real.

    The contaminated cases are caught instead by
    `timeseries.flag_sidelobe_channels`, which runs at MERGE level where the batch
    TS exists and can measure that ratio (6/6 caught, 0 false positives of 72). It
    keeps the formula and flags `intensity_suspect` when the neutral is corroborated
    on another channel, and demotes Assigned->Candidate when it is not."""
    pk = ledger.dropna(subset=["mz"]).sort_values("mz")
    mzs = pk["mz"].tolist(); hts = pk["height"].tolist()
    n = 0
    for pid, mz, h in zip(ledger["peak_id"], ledger["mz"], ledger["height"]):
        if ledger.loc[ledger.peak_id == pid, "role"].iloc[0] != L.ROLE_UNEXPLAINED:
            continue
        if pd.isna(mz) or pd.isna(h) or h <= 0:
            continue
        lo = bisect.bisect_left(mzs, mz - dmz)
        hit = None
        for i in range(lo, len(mzs)):
            if mzs[i] > mz + dmz:
                break
            if (abs(mzs[i] - mz) > 1e-4 and hts[i] >= min_parent
                    and hts[i] >= factor * h):
                if hit is None or hts[i] > hit[1]:
                    hit = (mzs[i], hts[i])
        if hit is not None:
            L.mark_artifact(ledger, pid,
                            f"FT ringing/sidelobe of {hit[0]:.4f} "
                            f"({hit[1]:.0f} cps, {hit[1] / h:.0f}x brighter, "
                            f"Δ{(hit[0] - mz) * 1000:+.0f} mDa, satellite "
                            f"{100 * h / hit[1]:.1f}% of parent)")
            n += 1
    log(f"[cleanup] flagged {n} ringing/sidelobe artifacts")
    return {"flagged": n}


# ---- 2. bromide reagent clusters -----------------------------------------
# A covalent di-/tri-bromo organic is formula-DEGENERATE with the bromide-
# cluster ion (same ion formula), so the reagent-adduct reading stays the
# correct CALL -- but the commentary must not assert "no covalent reading" when
# the oracle gives that covalent neutral a high score. So we score the
# mass-feasible covalent bromo-organics before wording the note.
CLUSTER_COVALENT_BOX = "C0-12 H0-22 N0-1 O0-8 S0-1 Cl0-1 Br1-3"
CLUSTER_COVALENT_TAU = 0.70


def label_bromide_clusters(ledger: pd.DataFrame, client=None, sample_id=None, *,
                           score_fn=None, defect_max: float = -0.16,
                           covalent_tau: float = CLUSTER_COVALENT_TAU,
                           log=print) -> dict:
    """Label strongly negative-defect unexplained peaks that carry a Br isotope
    partner (1.998 Da, ratio 0.4-3.0) as reagent bromide-cluster ions.

    When the oracle is available, each peak's commentary first tests whether a
    covalent di-/tri-bromo organic fits (the formula-degenerate reading): a
    high-scoring tie gets the "reagent-adduct reading preferred over degenerate
    di-bromo organic" wording (the pipeline's existing precedent), NOT the false
    "no covalent reading". The role stays reagent either way."""
    pk = ledger.dropna(subset=["mz"]).sort_values("mz")
    mzs = pk["mz"].tolist(); hts = pk["height"].tolist()
    def near(t, tol=0.006):
        i = bisect.bisect_left(mzs, t - tol)
        best = None
        while i < len(mzs) and mzs[i] <= t + tol:
            if best is None or abs(mzs[i] - t) < abs(best[0] - t):
                best = (mzs[i], hts[i])
            i += 1
        return best

    # phase 1: collect the cluster peaks (defect + Br-twin gate), before marking
    targets = []   # (peak_id, mz)
    for pid, mz, h in zip(list(ledger["peak_id"]), list(ledger["mz"]), list(ledger["height"])):
        if ledger.loc[ledger.peak_id == pid, "role"].iloc[0] != L.ROLE_UNEXPLAINED:
            continue
        if pd.isna(mz) or pd.isna(h):
            continue
        defect = float(mz) - round(float(mz))
        if defect >= defect_max:
            continue
        up, dn = near(mz + BR), near(mz - BR)
        twin = (up is not None and 0.4 < up[1] / h < 3.0) or \
               (dn is not None and h > 0 and 0.4 < (h / dn[1] if dn[1] else 9) < 3.0)
        if twin:
            targets.append((pid, float(mz)))

    # phase 2: covalent-fit oracle check (only if a scorer is available)
    best_cov = {}   # peak_id -> (ion_formula, score)
    if targets and (score_fn is not None or client is not None):
        sf = score_fn or IO.score_candidates
        box = C.parse_ranges(CLUSTER_COVALENT_BOX)
        per, allf = {}, set()
        for pid, mz in targets:
            cands = {f for f in C.candidates_for_peaks(
                        [mz], box, ["[M-H]-", "[M+Br]-"], ppm_tolerance=2.0,
                        mass_min=mz - 90, mass_max=mz + 2)
                     if C.parse_formula(f).get("Br", 0) >= 1}
            per[pid] = cands; allf |= cands
        if allf:
            try:
                fr = sf(client, sample_id, sorted(allf), allow_partial=True)
                fr = fr[fr["sample_peak_id"].notna() & (fr["sample_peak_intensity"] > 0)]
                for pid, mz in targets:
                    sub = fr[(fr["sample_peak_mz"] - mz).abs() < 0.006]
                    if not sub.empty:
                        r = sub.loc[sub["ion_score"].idxmax()]
                        if float(r["ion_score"]) >= covalent_tau:
                            best_cov[pid] = (str(r["ion_formula"]), float(r["ion_score"]))
            except Exception as e:   # the oracle must never crash the label step
                log(f"[cleanup] cluster covalent-check skipped: {e}")

    # phase 3: mark, with honest wording
    for pid, mz in targets:
        defect = mz - round(mz)
        if pid in best_cov:
            ion, s = best_cov[pid]
            note = (f"bromide cluster ion (defect {defect:+.3f}); reagent-adduct "
                    f"reading preferred over degenerate di-bromo organic "
                    f"(ion {ion} also fits, score {s:.2f})")
        elif best_cov or (score_fn is not None or client is not None):
            # the oracle ran and found no covalent tie for THIS peak
            note = (f"bromide cluster ion (defect {defect:+.3f}, multi-Br reagent "
                    "region; no covalent reading scores above threshold)")
        else:
            # offline: no oracle was consulted -> make no claim about covalency
            note = (f"bromide cluster ion (defect {defect:+.3f}, multi-Br reagent "
                    "region)")
        # known formula -> assigned: record the ion formula when we have one
        L.mark_reagent(ledger, pid, note,
                       ion_formula=(best_cov[pid][0] if pid in best_cov else None))
    log(f"[cleanup] labelled {len(targets)} bromide reagent-cluster ions "
        f"({len(best_cov)} with a degenerate covalent tie)")
    return {"labelled": len(targets), "covalent_ties": len(best_cov)}


# ---- 3. isotope-confirmed low-complexity recovery ------------------------
# CHO neutrals + (optionally) a covalent halogen the isotope pattern can confirm.
# We deliberately do NOT enumerate N or S in the recovery: the only corroboration
# the recovery has is the halogen isotope envelope, and that confirms the adduct
# Br -- it says nothing about N or S. A borderline-score CHON/CHOS fit is exactly
# the degenerate class the review flagged (C18H14N2O3S: N2 + S, 34S absent, beaten
# by other formulas). So an uncorroborated heteroatom is not recoverable here;
# the confident CHON/CHOS are committed in the main passes, not in cleanup.
RECOVERY_BOX = "C0-20 H0-36 O0-12 Cl0-1 Br0-2"
RECOVERY_ADDUCTS = ("[M-H]-", "[M+Br]-")


def _measure_pattern(near, mz, h):
    """(r2, r4) = measured M+2/M0 and M+4/M0 intensity ratios."""
    p2, p4 = near(mz + BR), near(mz + 2 * BR)
    r2 = (p2[1] / h) if (p2 is not None and h > 0) else 0.0
    r4 = (p4[1] / h) if (p4 is not None and h > 0) else 0.0
    return r2, r4


def _pattern_ok(nbr_ion, ncl_ion, r2, r4):
    """Does the measured (r2, r4) confirm the candidate's halogen count?
    Recovery REQUIRES a halogen handle, so a (0 Br, 0 Cl) ion is rejected here
    (it has no independent corroboration for the relaxed score floor)."""
    if nbr_ion == 1 and ncl_ion == 0:
        return 0.78 <= r2 <= 1.20            # 1 Br: M+2/M0 ~ 0.97
    if nbr_ion == 2 and ncl_ion == 0:
        return 1.55 <= r2 <= 2.35 and 0.55 <= r4 <= 1.35   # 2 Br: 1.95 / 0.95
    if nbr_ion == 0 and ncl_ion == 1:
        return 0.20 <= r2 <= 0.48            # 1 Cl: M+2/M0 ~ 0.32
    if nbr_ion == 1 and ncl_ion == 1:
        return 1.10 <= r2 <= 1.55            # BrCl: ~1.29
    return False


def recover_isotope_gated(client, sample_id, ledger, profile, cfg, *,
                          score_fn=None, score_floor: float = 0.65,
                          z_max: float = 2.5, log=print) -> dict:
    """Recover genuine molecules the score gate dropped, gated on a CONFIRMED
    halogen isotope pattern + a low-complexity clamp."""
    score_fn = score_fn or IO.score_candidates
    mu = getattr(cfg, "cal_mu", None)
    sigma = getattr(cfg, "cal_sigma", None) or 0.5
    pk = ledger.dropna(subset=["mz"]).sort_values("mz")
    mzs = pk["mz"].tolist(); hts = pk["height"].tolist()
    def near(t, tol=0.006):
        i = bisect.bisect_left(mzs, t - tol); best = None
        while i < len(mzs) and mzs[i] <= t + tol:
            if best is None or abs(mzs[i] - t) < abs(best[0] - t):
                best = (mzs[i], hts[i])
            i += 1
        return best

    un = ledger[ledger["role"] == L.ROLE_UNEXPLAINED]
    ranges = C.parse_ranges(RECOVERY_BOX)
    # enumerate per unexplained peak, union for ONE batched score
    per = {}
    allf = set()
    for _, p in un.iterrows():
        mz = float(p["mz"])
        cands = {f for f in C.candidates_for_peaks(
                    [mz], ranges, list(RECOVERY_ADDUCTS), ppm_tolerance=2.0,
                    mass_min=mz - 82, mass_max=mz + 2)
                 if X.filter_by_profile(f, profile)[0]
                 and D._het_types(C.parse_formula(f)) <= 2}
        if cands:
            per[p["peak_id"]] = (mz, float(p["height"]), cands)
            allf |= cands
    if not allf:
        log("[cleanup] recovery: no low-complexity candidates")
        return {"recovered": 0}
    fr = score_fn(client, sample_id, sorted(allf), allow_partial=True,
                  mechanism_ids=getattr(cfg, "mechanism_ids", None))
    # a no-match response can be a bare empty frame with NO columns (seen when
    # probing negative halogen adducts against a positive run) -- it
    # must not crash run_cleanup, which would skip every later cleanup step
    if fr is None or not len(fr) or "sample_peak_id" not in fr.columns:
        log("[cleanup] recovery: oracle returned no scoreable matches")
        return {"recovered": 0}
    fr = fr[fr["sample_peak_id"].notna() & (fr["sample_peak_intensity"] > 0)]

    recovered = 0
    for pid, (mz, h, _c) in per.items():
        if ledger.loc[ledger.peak_id == pid, "role"].iloc[0] != L.ROLE_UNEXPLAINED:
            continue
        sub = fr[(fr["sample_peak_mz"] - mz).abs() < 0.006].copy()
        if sub.empty:
            continue
        if mu is not None:
            sub = sub[((sub["ppm_error"] - mu) / sigma).abs() <= z_max]
        sub = sub[sub["ion_score"] >= score_floor]
        if sub.empty:
            continue
        r2, r4 = _measure_pattern(near, mz, h)
        sub = sub.sort_values("ion_score", ascending=False)
        chosen = None
        for _, r in sub.iterrows():
            ic = C.parse_formula(r["ion_formula"])
            if _pattern_ok(ic.get("Br", 0), ic.get("Cl", 0), r2, r4):
                chosen = r
                break
        if chosen is None:
            continue
        # parse neutral/adduct back out of the candidate set for this peak,
        # preferring the reagent-adduct reading (fewest halogens in the neutral)
        neutral, adduct = _decompose(chosen["ion_formula"], _c)
        if neutral is None:
            continue
        # H1/H2: the recovery's only corroboration is the halogen isotope (it
        # confirms the adduct Br, not N or S). A CHON/CHOS neutral here is an
        # uncorroborated heteroatom fit -> refuse it (C18H14N2O3S: N2+S, 34S
        # absent). Only CHO + isotope-confirmed covalent halogen is recoverable.
        nc = C.parse_formula(neutral)
        if nc.get("N", 0) > 0 or nc.get("S", 0) > 0 or nc.get("P", 0) > 0:
            continue
        ppm = float(chosen["ppm_error"])
        z = (ppm - mu) / sigma if mu is not None else 0.0
        L.commit_assignment(
            ledger, pid, neutral_formula=neutral, adduct=adduct,
            ion_formula=str(chosen["ion_formula"]),
            ion_score=float(chosen["ion_score"]),
            compound_score=float(chosen["ion_score"]),
            ppm_error=ppm, pass_no=7, method="cleanup:iso-recovery",
            confidence="Good (recovered)",
            commentary=(f"Recovery: halogen isotope pattern confirms the ion "
                        f"(M+2/M0={r2:.2f}); score {chosen['ion_score']:.2f}, "
                        f"z={z:+.1f} within calibrated accuracy."))
        recovered += 1
    log(f"[cleanup] recovered {recovered} isotope-confirmed molecules")
    return {"recovered": recovered}


def _decompose(ion_formula, candidate_set):
    """Map a scored ion back to (neutral, adduct), PREFERRING the reading with
    the fewest halogens in the neutral. The reagent halogen's ion isotope cannot
    prove it is covalent (C5H10O3.Br- == C5H11BrO3 [M-H]-, same ion), and the
    pipeline policy is the reagent-adduct reading -- so a tie goes to the
    Br-in-the-adduct interpretation, deterministically."""
    target = str(ion_formula).rstrip("-")
    best = None   # (neutral_halogen_count, neutral, adduct)
    for f in sorted(candidate_set):
        for a in RECOVERY_ADDUCTS:
            if _ion_str(f, a) == target:
                cnt = C.parse_formula(f)
                nh = cnt.get("Br", 0) + cnt.get("Cl", 0)
                if best is None or nh < best[0]:
                    best = (nh, f, a)
    return (best[1], best[2]) if best else (None, None)


def _ion_str(neutral, adduct):
    cnt = dict(C.parse_formula(neutral))
    for sign, tok in __import__("re").findall(r"([+-])\^?([A-Za-z0-9]+)",
                                              adduct.split("]")[0][2:]):
        for el, k in C.parse_formula(tok).items():
            cnt[el] = cnt.get(el, 0) + (k if sign == "+" else -k)
    cnt = {k: v for k, v in cnt.items() if v}
    return C.format_formula(cnt)


def reclaim_satellites(ledger: pd.DataFrame, *, ppm: float = 6.0, log=print) -> dict:
    """Final sweep: attach an UNEXPLAINED peak that is a clean satellite of an
    assigned M0 (parent carries that element AND the intensity ratio is physically
    consistent) as an iso_child. Covers the full diagnostic set 13C / 15N / 81Br /
    37Cl / 34S / 29Si / 30Si / 18O -- so a faint single-heteroatom satellite (a
    CHON's 15N line, a mono-S 34S) that the envelope passes and the server both
    leak is claimed here instead of floating free as a base peak a mass-coincidence
    phantom can grab. Touches only unexplained rows -- it can never demote a real
    M0. Every gate is atom-count-aware (ratio ~ n_atoms * per-atom abundance), so
    it cannot mis-grab an unrelated neighbour at the satellite offset."""
    from peaky.chem import isotopes as ISO
    m0 = ledger[ledger["role"] == L.ROLE_M0].dropna(subset=["mz"]).sort_values("mz")
    if not len(m0):
        return {"reclaimed": 0}
    mz = m0["mz"].to_numpy(); pid = m0["peak_id"].to_numpy()
    ionf = m0["ion_formula"].astype(str).to_numpy(); ph = m0["height"].to_numpy()
    # (delta-m, label, element, per-atom abundance). 13C first (most common), then
    # the heavy-halogen M+2 doublets, then the faint mono-heteroatom diagnostics.
    DELT = [
        (ISO.D_13C,  "13C",  "C",  ISO.R_13C_PER_C),
        (ISO.D_81BR, "81Br", "Br", ISO.R_81BR_PER_BR),
        (ISO.D_37CL, "37Cl", "Cl", ISO.R_37CL_PER_CL),
        (ISO.D_15N,  "15N",  "N",  ISO.R_15N_PER_N),
        (ISO.D_34S,  "34S",  "S",  ISO.R_34S_PER_S),
        (ISO.D_29SI, "29Si", "Si", ISO.R_29SI_PER_SI),
        (ISO.D_30SI, "30Si", "Si", ISO.R_30SI_PER_SI),
        (ISO.D_18O,  "18O",  "O",  ISO.R_18O_PER_O),
    ]
    n = 0
    for i in ledger.index[ledger["role"] == L.ROLE_UNEXPLAINED]:
        cmz = ledger.at[i, "mz"]; chh = ledger.at[i, "height"]
        if pd.isna(cmz):
            continue
        cmz = float(cmz); chh = float(chh) if pd.notna(chh) else 0.0
        for d, label, el, per in DELT:
            t = cmz - d; tol = cmz * ppm * 1e-6
            j = bisect.bisect_left(mz, t - tol); best = None
            while j < len(mz) and mz[j] <= t + tol:
                if best is None or abs(mz[j] - t) < abs(mz[best] - t):
                    best = j
                j += 1
            if best is None:
                continue
            cnt = C.parse_formula(ionf[best]); nel = cnt.get(el, 0)
            if nel < 1 or ph[best] <= 0:
                continue
            ratio = chh / ph[best]
            if label == "81Br":                # ~0.97 per Br adduct/atom
                ok = 0.55 <= ratio <= 1.4 * nel
            elif label == "37Cl":              # 37Cl ~0.32 per Cl
                ok = 0.18 <= ratio <= 0.5 * nel
            else:                              # 13C/15N/34S/29Si/30Si/18O: count-aware
                exp = nel * per
                ok = 0.3 * exp <= ratio <= 2.5 * exp and ratio < 1.0
            if not ok:
                continue
            try:
                L.attach_isotopologue(ledger, ledger.at[i, "peak_id"], pid[best],
                                      iso_label=label)
                ledger.at[i, "commentary"] = (
                    f"reclaimed {label} satellite of {ionf[best]} "
                    f"(ratio {ratio:.2f}); envelope-pass leak")
                n += 1
            except L.LedgerError:
                pass
            break
    log(f"[cleanup] reclaimed {n} leaked isotopologue satellites")
    return {"reclaimed": n}


def reclaim_envelope_tails(ledger: pd.DataFrame, *, ppm: float = 6.0, log=print) -> dict:
    """Attach the DEEP multi-halogen isotope envelope (k>=2 of ³⁷Cl / ⁸¹Br) of a
    poly-halogen M0 to leaked unexplained peaks. reclaim_satellites only does the
    single k=1 step, so a chlorinated paraffin's M+4/M+6/... ³⁷Cl tail -- which for
    many Cl is BRIGHTER than M0 -- leaks into 'unexplained' (the dominant satellite
    leak measured on a nitrate-CIMS batch). Walk k=2..nX from each M0 and accept an
    unexplained peak whose intensity ratio matches the binomial C(nX,k)(p/q)^k.
    Touches only unexplained rows; the ratio gate (and exact +k·Δ mass) prevent
    grabbing an unrelated neighbour. p/q: ³⁷Cl 0.3199, ⁸¹Br 0.9728.

    KNOWN LIMITATION (do not trust this to clear real tails): on real batches this
    has been observed to be a no-op — the deep-tail leak it targets is, in
    practice, already absorbed upstream (reclaim_satellites k=1 + the isotope-locked
    known-species/CP recovery), so by the time this runs there are typically no
    leaked unexplained tails left for the binomial gate to match. The unit test
    exercises only the synthetic poly-³⁷Cl case. Kept (harmless, additive — touches
    only unexplained rows) pending a live-data re-evaluation of the gate; see
    docs/ROADMAP.md."""
    from math import comb
    from peaky.chem import isotopes as ISO
    m0 = ledger[ledger["role"] == L.ROLE_M0].dropna(subset=["mz"]).sort_values("mz")
    if not len(m0):
        return {"tails": 0}
    mz = m0["mz"].to_numpy(); pid = m0["peak_id"].to_numpy()
    ionf = m0["ion_formula"].astype(str).to_numpy(); ph = m0["height"].to_numpy()
    STEPS = [(ISO.D_37CL, "37Cl", "Cl", 0.3199), (ISO.D_81BR, "81Br", "Br", 0.9728)]
    n = 0
    for i in ledger.index[ledger["role"] == L.ROLE_UNEXPLAINED]:
        cmz = ledger.at[i, "mz"]
        if pd.isna(cmz):
            continue
        cmz = float(cmz); chh = float(ledger.at[i, "height"]) if pd.notna(ledger.at[i, "height"]) else 0.0
        done = False
        for D, label, el, pq in STEPS:
            for k in range(2, 11):
                t = cmz - k * D; tol = cmz * ppm * 1e-6
                j = bisect.bisect_left(mz, t - tol); best = None
                while j < len(mz) and mz[j] <= t + tol:
                    if best is None or abs(mz[j] - t) < abs(mz[best] - t):
                        best = j
                    j += 1
                if best is None:
                    continue
                nX = C.parse_formula(ionf[best]).get(el, 0)
                if k > nX or ph[best] <= 0:
                    continue
                exp = comb(nX, k) * (pq ** k)            # binomial intensity vs M0
                ratio = chh / ph[best]
                if 0.35 * exp <= ratio <= 2.8 * exp:
                    try:
                        L.attach_isotopologue(ledger, ledger.at[i, "peak_id"], pid[best],
                                              iso_label=f"{k}x{label}")
                        ledger.at[i, "commentary"] = (
                            f"reclaimed {k}x{label} envelope tail of {ionf[best]} "
                            f"(ratio {ratio:.2f} vs binomial {exp:.2f})")
                        n += 1; done = True
                    except L.LedgerError:
                        pass
                    break
            if done:
                break
    log(f"[cleanup] reclaimed {n} deep halogen-envelope tails (k>=2)")
    return {"tails": n}


F_DEMOTE_MIN = 4   # F count above which an unanchored, non-PFCA fit is an F-monster


def _confirmed_iso_labels(s) -> set:
    """Set of server-confirmed isotope labels from a row's isotopologues JSON
    (composite labels split: '13C+81Br' -> {'13C','81Br'})."""
    out: set = set()
    if isinstance(s, str) and s.strip().startswith("["):
        try:
            for d in json.loads(s):
                for p in str(d.get("label", "")).split("+"):
                    if p:
                        out.add(p)
        except Exception:
            pass
    return out


def _iso_count(s) -> int:
    if isinstance(s, str) and s.strip().startswith("["):
        try:
            return len(json.loads(s))
        except Exception:
            return 0
    return 0


def _is_pfca(cnt: dict) -> bool:
    """Perfluorocarboxylic acid CnHF(2n-1)O2 (TFA, PFPrA, ... PFOA)."""
    nC = cnt.get("C", 0)
    return (nC >= 2 and cnt.get("H", 0) == 1 and cnt.get("O", 0) == 2
            and cnt.get("F", 0) == 2 * nC - 1)


def demote_unconfirmed_fluorine(ledger: pd.DataFrame, *, f_min: int = F_DEMOTE_MIN,
                                log=print) -> dict:
    """Demote M0 assignments resting on UNCONFIRMED fluorine. ¹⁹F is 100%
    monoisotopic, so a high-F formula has NO isotope twin to corroborate it; when it
    is also not a known PFCA and carries no Cl/Br/S anchor (whose ³⁷Cl/⁸¹Br/³⁴S
    pattern WOULD confirm a heavy atom), a high-F fit is almost always a mass
    coincidence ('F-monster', e.g. C11H6F16). Demote Assigned->Candidate and flag
    below_assignability. Real PFCAs and isotope-anchored F species are kept."""
    n = 0
    has_ba = "below_assignability" in ledger.columns
    has_iso = "isotopologues" in ledger.columns
    target = (ledger.index[ledger["role"] == L.ROLE_M0]
              if "role" in ledger.columns else ledger.index)   # merged ledger has no role col
    for i in target:
        cnt = C.parse_formula(str(ledger.at[i, "neutral_formula"] or ""))
        if cnt.get("F", 0) < f_min:
            continue
        if _is_pfca(cnt):
            continue
        # a Cl/Br/S anchor exempts F only when its diagnostic isotope is CONFIRMED
        # (34S/37Cl/81Br) -- NOT merely present in the formula. A reagent Br adduct's
        # 81Br confirms the ADDUCT, not the neutral, so it does not count. On the
        # merged ledger (no isotopologues column) fall back to formula-presence.
        if has_iso:
            labs = _confirmed_iso_labels(ledger.at[i, "isotopologues"])
            ad = str(ledger.at[i, "adduct"]) if "adduct" in ledger.columns else ""
            if ((cnt.get("S", 0) and "34S" in labs)
                    or (cnt.get("Cl", 0) and "37Cl" in labs)
                    or (cnt.get("Br", 0) and "81Br" in labs and "Br" not in ad)):
                continue
        elif cnt.get("Cl", 0) or cnt.get("Br", 0) or cnt.get("S", 0):
            continue
        if str(ledger.at[i, "tier"]) == "Assigned":
            ledger.at[i, "tier"] = "Candidate"
        if has_ba:
            ledger.at[i, "below_assignability"] = True
        if "commentary" in ledger.columns:
            note = (f"unconfirmed fluorine (F{cnt.get('F', 0)}; ¹⁹F monoisotopic, no "
                    "CONFIRMED Cl/Br/S isotope anchor, not a PFCA) -- likely mass coincidence")
            prev = str(ledger.at[i, "commentary"] or "")
            ledger.at[i, "commentary"] = (prev + "; " + note) if prev and prev != "nan" else note
        n += 1
    log(f"[cleanup] demoted {n} unconfirmed-fluorine M0 (F>={f_min}, F-monsters)")
    return {"f_demoted": n}


HC_CARBON_MAX = 0.35   # H/C below this (F-free) -> implausibly carbon-rich skeleton


def demote_implausible_carbon(ledger: pd.DataFrame, *, hc_max: float = HC_CARBON_MAX,
                              log=print) -> dict:
    """Demote M0 assignments resting on an implausibly carbon-rich skeleton:
    (H+F)/C below `hc_max`. F counts as an H-equivalent rather than exempting
    the formula: true perfluoro classes have high (H+F)/C (PFCA ~2, C6F6 =1)
    and are spared, while an F-decorated bare-carbon fit (C16HF3O, (H+F)/C
    0.25) no longer slips through on an F-free clause. F>=4 fluorine-monsters
    are additionally handled by demote_unconfirmed_fluorine. A bare carbon
    cluster such as C27H8 / C36H6O is a high-mass mass-coincidence, not a real
    organic-aerosol molecule (real SOA sits at H/C ~1-2). Demote
    Assigned->Candidate and flag below_assignability. C>=2 only. Same
    arithmetic as the plausibility 'carbon-rich' flag."""
    n = 0
    has_ba = "below_assignability" in ledger.columns
    target = (ledger.index[ledger["role"] == L.ROLE_M0]
              if "role" in ledger.columns else ledger.index)   # merged ledger has no role col
    for i in target:
        cnt = C.parse_formula(str(ledger.at[i, "neutral_formula"] or ""))
        nc = cnt.get("C", 0)
        if nc < 2:
            continue
        hc = (cnt.get("H", 0) + cnt.get("F", 0)) / nc   # F counts as H-equivalent
        if hc >= hc_max:
            continue
        if str(ledger.at[i, "tier"]) == "Assigned":
            ledger.at[i, "tier"] = "Candidate"
        if has_ba:
            ledger.at[i, "below_assignability"] = True
        if "commentary" in ledger.columns:
            note = f"implausibly carbon-rich ((H+F)/C {hc:.2f}) -- likely mass coincidence"
            prev = str(ledger.at[i, "commentary"] or "")
            ledger.at[i, "commentary"] = (prev + "; " + note) if prev and prev != "nan" else note
        n += 1
    log(f"[cleanup] demoted {n} implausibly-carbon-rich M0 ((H+F)/C<{hc_max})")
    return {"c_demoted": n}


# anion channels need a functional group: [M-H]- an acidic proton, the cluster
# adducts ([M+Br]-/[M+CO3]-/[M+NO3]-/...) an H-bond donor / polar site. Electron
# attachment [M]-. is the ONLY negative channel a heteroatom-free species can use.
_EA_ADDUCTS = {"[M]-.", "[M]-", "[M+O2]-"}

# Functional-group cluster anions whose atoms, "absorbed" into a hydrocarbon
# neutral, yield a CLOSED-SHELL oxygenated/N neutral whose radical anion M-. has
# the same ion mass. A pure hydrocarbon cannot bind CO3-/NO3- (no polar site --
# demote_implausible_ionization), but the SAME even-H CxHyOz- ion IS exactly the
# radical anion of that closed-shell neutral: e.g. "C6H6 [M+CO3]-" == "C7H6O3
# [M]-.". The [M-H]- reading is parity-forbidden (it would need a radical neutral
# C7H7O3), so M-. is the only sane oxygenated identity.
_RADICAL_CLUSTER_ATOMS = {
    "[M+CO3]-": {"C": 1, "O": 3},
    "[M+NO3]-": {"N": 1, "O": 3},
    "[M+^NO3]-": {"N": 1, "O": 3},
}
# channels that require a real functional group -> independent proof the neutral
# exists (corroboration for the radical-anion relabel).
_PRIMARY_ANION_CHANNELS = ("[M-H]-", "[M+Br]-", "[M+Cl]-")


def relabel_radical_anions(ledger: pd.DataFrame, *, log=print) -> dict:
    """Relabel pure-hydrocarbon FG-cluster anions (e.g. C6H6 [M+CO3]-) as the
    radical anion M-. of the closed-shell oxygenated neutral that shares the ion
    mass (C7H6O3 [M]-.). The even-H CxHyOz- ion cannot be [M-H]- (parity: that
    needs a radical neutral) but IS exactly M-. of CxHyOz -- a sane oxygenated
    molecule, not an impossible hydrocarbon carbonate adduct.

    Corroborated when that neutral is independently assigned via a functional-group
    channel ([M-H]-/[M+Br]-/[M+Cl]-): it stays a visible Candidate (not below-
    assignability). Uncorroborated -> Candidate + below_assignability: a flagged
    best-guess, still shown (users want the best guess even when it's a guess),
    just lowest-confidence. Runs BEFORE demote_implausible_ionization so the
    relabeled rows (now hetero-bearing neutral + [M]-. adduct) escape the
    hydrocarbon demote. Negative-mode FG-cluster anions only."""
    if "neutral_formula" not in ledger.columns or "adduct" not in ledger.columns:
        return {"radical_relabeled": 0, "radical_corroborated": 0}
    has_ba = "below_assignability" in ledger.columns
    corrob = {str(ledger.at[j, "neutral_formula"] or "")
              for j in ledger.index
              if str(ledger.at[j, "adduct"]) in _PRIMARY_ANION_CHANNELS}
    corrob.discard("");  corrob.discard("nan")
    target = (ledger.index[ledger["role"] == L.ROLE_M0]
              if "role" in ledger.columns else ledger.index)
    n = nc = 0
    for i in target:
        atoms = _RADICAL_CLUSTER_ATOMS.get(str(ledger.at[i, "adduct"]))
        if atoms is None:
            continue
        cnt = C.parse_formula(str(ledger.at[i, "neutral_formula"] or ""))
        if cnt.get("C", 0) < 1:
            continue
        if sum(cnt.get(e, 0) for e in ("O", "N", "S", "P", "F", "Cl", "Br", "I", "Si")):
            continue                                 # only pure-hydrocarbon mislabels
        ion = dict(cnt)
        for e, k in atoms.items():
            ion[e] = ion.get(e, 0) + k               # closed-shell neutral == ion composition
        if not C.dbe_ok(ion)[0] or not C.oxygen_ok(ion)[0]:
            continue                                 # not a valid closed-shell molecule
        new_neutral = C.format_formula(ion)
        is_corrob = new_neutral in corrob
        ledger.at[i, "neutral_formula"] = new_neutral
        ledger.at[i, "adduct"] = "[M]-."
        if "ion_formula" in ledger.columns:
            ledger.at[i, "ion_formula"] = new_neutral + "-"
        if "dbe" in ledger.columns:
            ledger.at[i, "dbe"] = C.dbe(ion)
        if str(ledger.at[i, "tier"]) == "Assigned":
            ledger.at[i, "tier"] = "Candidate"
        if has_ba:
            ledger.at[i, "below_assignability"] = not is_corrob
        if "confidence" in ledger.columns:
            ledger.at[i, "confidence"] = ("Good (radical anion, corroborated)"
                                          if is_corrob else "Low (radical anion)")
        note = (f"radical anion M-. of {new_neutral}: the even-H {new_neutral}- ion "
                "cannot be [M-H]- (would need a radical neutral) -- it is M-. of the "
                "closed-shell neutral, not a hydrocarbon+CO3 adduct; "
                + ("corroborated by an independent [M-H]-/[M+Br]- assignment of the same neutral"
                   if is_corrob else "uncorroborated best-guess (no [M-H]-/[M+Br]- partner)"))
        for col in ("commentary",):
            if col in ledger.columns:
                prev = str(ledger.at[i, col] or "")
                ledger.at[i, col] = (prev + "; " + note) if prev and prev != "nan" else note
        if "tier_reason" in ledger.columns:
            ledger.at[i, "tier_reason"] = note
        n += 1
        nc += int(is_corrob)
    log(f"[cleanup] relabeled {n} hydrocarbon FG-cluster anions as radical anions M-. "
        f"({nc} corroborated by [M-H]-/[M+Br]-)")
    return {"radical_relabeled": n, "radical_corroborated": nc}


def _ts_pair_r(ts_peaks, mz_a: float, mz_b: float, *, tol: float = 0.004,
               min_partner: float = 300.0, min_seen: int = 4):
    """Batch time-correlation of two ion masses: Pearson r on log1p per-sample
    max heights (absent sample = 0 -- co-absence is signal for an event pair).
    Returns (r, n_samples) or (None, n) when the partner is not genuinely
    present (max < min_partner or seen in < min_seen samples) or a series is
    constant."""
    pk = ts_peaks
    if "peak_id" in pk.columns:
        pk = pk.drop_duplicates(["sample_item_id", "peak_id"])
    smp = pk["sample_item_id"].unique()
    if len(smp) < 6:
        return None, len(smp)
    prof = {}
    for key, mz0 in (("a", mz_a), ("b", mz_b)):
        d = pk[(pk["mz"] - mz0).abs() <= tol]
        ts = d.groupby("sample_item_id")["height"].max()
        prof[key] = np.array([float(ts.get(s, 0.0)) for s in smp])
    b = prof["b"]
    if b.max() < min_partner or int((b > 0).sum()) < min_seen:
        return None, len(smp)
    a, b = np.log1p(prof["a"]), np.log1p(b)
    if a.std() == 0 or b.std() == 0:
        return None, len(smp)
    return float(np.corrcoef(a, b)[0, 1]), len(smp)


def annotate_easyic_ambiguity(ledger: pd.DataFrame, *, ts_peaks=None,
                              ts_r_min: float = 0.9, log=print) -> dict:
    """EasyIC⁺ fragmentation ambiguity (easyic context only). A low-pressure
    charge-transfer source FRAGMENTS, so three MS1-irreducible readings recur
    (all three observed on the 2026-08-31 KORBI2 gin run):

      1. A CnH2n "[M+H]+" commit (the alkene reading) is the SAME ion as an
         alcohol's in-source dehydration [CnH2n+2O + H - H2O]+. When that
         alcohol is independently committed on its own hydride channel [M-H]+
         (its ion sits exactly +O away), the dehydration reading is
         corroborated -> RELABEL onto the alcohol (the radical-anion ruling:
         stays a visible Candidate). Uncorroborated -> keep the alkene,
         append the ambiguity note. (Gin: C4H8/C5H10 [M+H]+ were dehydrated
         butanol/amyl alcohol -- the C4H9O+/C5H11O+ partners were present.)
      2. Any heteroatom-free CxHyOz "[M+H]+" is ion-identical to
         [CxH(y+2)Oz - H]+: protonated CARBONYL vs hydride-abstracted ALCOHOL
         (59.049 = acetone AND propanol, both real in the gin). Commentary
         only -- MS1 cannot split them, and both may contribute.
      3. A pure-hydrocarbon cation with DBE >= 2 may be a FRAGMENT of a larger
         analyte (monoterpenes fragment; C7H9+ etc.), not the intact neutral.
         Commentary only.

    Locked rows are skipped (the pass-0 ethanol hydride lock already carries
    its own observation-based commentary).

    With a batch `ts_peaks` table, a SECOND corroboration route opens: a
    single spectrum cannot tell genuine-butene + genuine-MEK from dehydrated
    butanol + its hydride ion, but the BATCH can -- the dehydration pair
    (CnH2n·H+ and CnH2n+1O+, exactly +O apart) rises and falls together.
    Pearson r on log heights across the batch >= `ts_r_min` (with the partner
    genuinely present) promotes the note to the relabel, and the carbonyl
    commit sitting ON the partner mass gets the correlation evidence appended
    to its dual-reading note (the alcohol contributes on that channel too)."""
    if "neutral_formula" not in ledger.columns or "adduct" not in ledger.columns:
        return {"easyic_dehydration": 0, "easyic_dual": 0, "easyic_fragment": 0}
    out = {"easyic_dehydration": 0, "easyic_dual": 0, "easyic_fragment": 0}
    # alcohols committed on the hydride channel corroborate a dehydration ion
    alcohols = {str(ledger.at[j, "neutral_formula"] or "")
                for j in ledger.index
                if str(ledger.at[j, "adduct"]) == "[M-H]+"
                and str(ledger.at[j, "role"]) == L.ROLE_M0}
    target = (ledger.index[ledger["role"] == L.ROLE_M0]
              if "role" in ledger.columns else ledger.index)
    partner_hits = []   # (partner_mz, alkene_mz, alcohol_formula, r, n)
    for i in target:
        if "locked" in ledger.columns and bool(ledger.at[i, "locked"]):
            continue
        add = str(ledger.at[i, "adduct"])
        nf = str(ledger.at[i, "neutral_formula"] or "")
        if not nf or nf == "nan":
            continue
        cnt = C.parse_formula(nf)
        hetero = sum(cnt.get(e, 0)
                     for e in ("N", "S", "P", "F", "Cl", "Br", "I", "Si"))
        nC, nH, nO = cnt.get("C", 0), cnt.get("H", 0), cnt.get("O", 0)
        note = None
        if add == "[M+H]+" and hetero == 0 and nO == 0 and nC >= 3 \
                and nH == 2 * nC:
            alc = C.format_formula({"C": nC, "H": nH + 2, "O": 1})
            basis = None
            if alc in alcohols:
                basis = (f"corroborated by {alc} committed on its own hydride "
                         "channel [M-H]+")
            elif ts_peaks is not None and len(ts_peaks) and "mz" in ledger.columns:
                # the batch can tell the pair apart: the dehydration ion and
                # its +O hydride partner rise and fall TOGETHER; independent
                # butene + MEK would not.
                mz_i = float(ledger.at[i, "mz"])
                r, nsm = _ts_pair_r(ts_peaks, mz_i, mz_i + C.M["O"])
                if r is not None and r >= ts_r_min:
                    basis = ("corroborated by batch time-correlation with its "
                             f"hydride-channel ion at m/z {mz_i + C.M['O']:.4f} "
                             f"(r={r:.2f} over {nsm} samples)")
                    partner_hits.append((mz_i + C.M["O"], mz_i, alc, r, nsm))
            if basis:
                ledger.at[i, "neutral_formula"] = alc
                ledger.at[i, "adduct"] = "[M+H-H2O]+"
                if "dbe" in ledger.columns:
                    ledger.at[i, "dbe"] = C.dbe(alc)
                if str(ledger.at[i, "tier"]) == "Assigned":
                    ledger.at[i, "tier"] = "Candidate"
                if "confidence" in ledger.columns:
                    ledger.at[i, "confidence"] = \
                        "Good (easyic dehydration, corroborated)"
                note = (f"in-source dehydration of {alc}: same ion as the {nf} "
                        f"[M+H]+ alkene reading; {basis} (EasyIC fragments "
                        "alcohols)")
                out["easyic_dehydration"] += 1
            else:
                note = (f"EasyIC ambiguity: this ion is {nf} [M+H]+ (alkene) OR "
                        f"an alcohol's in-source dehydration [{alc}+H-H2O]+ "
                        f"(no corroborating {alc} [M-H]+ in this sample)")
                out["easyic_dual"] += 1
        elif add == "[M+H]+" and hetero == 0 and nO >= 1:
            alc = C.format_formula({"C": nC, "H": nH + 2, "O": nO})
            note = (f"EasyIC ambiguity: same ion is [{nf}+H]+ (carbonyl/ether) "
                    f"and [{alc}-H]+ (alcohol via hydride abstraction) -- MS1 "
                    "cannot distinguish; both may contribute")
            out["easyic_dual"] += 1
        elif add in ("[M+H]+", "[M]+.") and hetero == 0 and nO == 0 \
                and C.dbe(cnt) >= 2:
            note = (f"fragmenting source: {nf} may be a FRAGMENT of a larger "
                    "analyte (monoterpenes/aromatics fragment under EasyIC), "
                    "not the intact neutral")
            out["easyic_fragment"] += 1
        if note and "commentary" in ledger.columns:
            prev = str(ledger.at[i, "commentary"] or "")
            ledger.at[i, "commentary"] = \
                (prev + "; " + note) if prev and prev != "nan" else note
    # symmetric upgrade: the carbonyl commit sitting ON a time-corroborated
    # partner mass carries an alcohol contribution too -- append the evidence
    # to its dual-reading note (reading kept; the batch cannot apportion the
    # carbonyl/alcohol split, only prove the alcohol is in there).
    if partner_hits and "commentary" in ledger.columns and "mz" in ledger.columns:
        for pmz, amz, alc, r, nsm in partner_hits:
            near = ledger.index[(ledger["mz"] - pmz).abs() <= 0.004]
            for j in near:
                if str(ledger.at[j, "role"]) != L.ROLE_M0 \
                        or str(ledger.at[j, "adduct"]) != "[M+H]+":
                    continue
                extra = (f"batch time-correlation with the -H2O dehydration ion "
                         f"at m/z {amz:.4f} (r={r:.2f} over {nsm} samples) -- a "
                         f"significant {alc} [M-H]+ (alcohol) contribution on "
                         "this channel")
                prev = str(ledger.at[j, "commentary"] or "")
                ledger.at[j, "commentary"] = \
                    (prev + "; " + extra) if prev and prev != "nan" else extra
    log(f"[cleanup] easyic ambiguity: {out['easyic_dehydration']} dehydration "
        f"relabels, {out['easyic_dual']} dual-reading notes, "
        f"{out['easyic_fragment']} fragment notes"
        + (f", {len(partner_hits)} TS-corroborated pairs" if partner_hits else ""))
    return out


# Positive-mode N-carrying reagent-cluster cations. A PURE HYDROCARBON has no
# site to bind these (and would show [M+H]+ if it ionized at all), so the
# cluster's N(/O) belongs to the ANALYTE: re-read as [M+H]+ of the N-heterocycle
# obtained by absorbing the cluster into the neutral. {atoms added to neutral M
# to reach the [M+H]+ neutral M'} = (cluster cation composition) - H.
_REAGENT_N_CLUSTERS = {
    "[M+NH4]+": {"N": 1, "H": 3},                      # ion-H = M + NH3
    "[M+(CH4N2O)H]+": {"C": 1, "H": 4, "N": 2, "O": 1},  # ion-H = M + CH4N2O (urea)
}


def relabel_reagent_n_adducts(ledger: pd.DataFrame, *, log=print) -> dict:
    """Positive-mode plausibility arbitration. A pure hydrocarbon assigned via an
    N-carrying reagent-cluster cation ([M+NH4]+ / uronium [M+(CH4N2O)H]+) is
    implausible: a hydrocarbon has no basic/polar site to bind the cluster, and a
    real one would ionize as [M+H]+. The reagent N is better read as ANALYTE N ->
    re-read as [M+H]+ of the N-heterocycle M' = M + (cluster - H) (C5H6
    [M+(CH4N2O)H]+ -> C6H10N2O [M+H]+; C5H6 [M+NH4]+ -> C5H9N [M+H]+).

    SKIPPED when the same hydrocarbon also has its own [M+H]+ row -- then it is a
    genuine hydrocarbon (e.g. a terpene C10H16 that legitimately forms [M+NH4]+),
    so its cluster adducts are left alone. The re-read is tiered Candidate +
    below_assignability: the specific N-heterocycle is rarely cross-channel-
    confirmed and the region is often reagent background, but the protonated-
    heterocycle label is the saner best-guess and stays visible. Positive adducts
    only (negative reagents never hit these)."""
    if "neutral_formula" not in ledger.columns or "adduct" not in ledger.columns:
        return {"reagent_n_relabeled": 0}
    has_ba = "below_assignability" in ledger.columns
    hc_with_mh = {str(ledger.at[j, "neutral_formula"] or "")
                  for j in ledger.index if str(ledger.at[j, "adduct"]) == "[M+H]+"}
    target = (ledger.index[ledger["role"] == L.ROLE_M0]
              if "role" in ledger.columns else ledger.index)
    n = 0
    for i in target:
        add = _REAGENT_N_CLUSTERS.get(str(ledger.at[i, "adduct"]))
        if add is None:
            continue
        f_raw = str(ledger.at[i, "neutral_formula"] or "")
        cnt = C.parse_formula(f_raw)
        if cnt.get("C", 0) < 1:
            continue
        if sum(cnt.get(e, 0) for e in ("O", "N", "S", "P", "F", "Cl", "Br", "I", "Si")):
            continue                                  # only pure-hydrocarbon mislabels
        if f_raw in hc_with_mh:
            continue                                  # genuine hydrocarbon (has its own [M+H]+)
        m2 = {e: cnt.get(e, 0) + add.get(e, 0) for e in set(cnt) | set(add)}
        if not C.dbe_ok(m2)[0] or not C.oxygen_ok(m2)[0]:
            continue                                  # not a valid closed-shell neutral
        new = C.format_formula(m2)
        ledger.at[i, "neutral_formula"] = new
        ledger.at[i, "adduct"] = "[M+H]+"
        if "ion_formula" in ledger.columns:
            ledger.at[i, "ion_formula"] = C.format_formula({**m2, "H": m2.get("H", 0) + 1}) + "+"
        if "dbe" in ledger.columns:
            ledger.at[i, "dbe"] = C.dbe(m2)
        if str(ledger.at[i, "tier"]) == "Assigned":
            ledger.at[i, "tier"] = "Candidate"
        if has_ba:
            ledger.at[i, "below_assignability"] = True
        if "confidence" in ledger.columns:
            ledger.at[i, "confidence"] = "Low (reagent-N re-read)"
        note = (f"re-read as [M+H]+ of {new}: a pure hydrocarbon has no site to bind the "
                "reagent N-cluster and would show [M+H]+ if it ionized -- the cluster N "
                "belongs to the analyte (N-heterocycle), not [M+reagent]+ of a hydrocarbon; "
                "tentative (this region is often reagent background)")
        if "commentary" in ledger.columns:
            prev = str(ledger.at[i, "commentary"] or "")
            ledger.at[i, "commentary"] = (prev + "; " + note) if prev and prev != "nan" else note
        if "tier_reason" in ledger.columns:
            ledger.at[i, "tier_reason"] = note
        n += 1
    log(f"[cleanup] re-read {n} hydrocarbon reagent-N-cluster adducts as [M+H]+ of an N-heterocycle")
    return {"reagent_n_relabeled": n}


def _norm_formula(f) -> str:
    """Canonicalise a neutral formula string (re-parse + re-format) so parent
    lookups compare like-for-like regardless of element ordering."""
    try:
        return C.format_formula(C.parse_formula(str(f)))
    except Exception:
        return ""


def relabel_nitrate_clusters(ledger: pd.DataFrame, *, log=print) -> dict:
    """¹⁵N-nitrate isobar arbitration. In a ¹⁵N-NO₃⁻ CIMS run of a NOx-oxidation
    experiment the chamber holds abundant *unlabelled* ¹⁴NO₃⁻; a highly-oxygenated
    analyte X clusters with it to give [X+¹⁴NO₃]⁻ -- which is the EXACT same ion
    (same composition, same mass) as the deprotonated covalent organonitrate [Y−H]⁻
    with Y = X + HNO₃ (e.g. [C10H14O3+¹⁴NO₃]⁻ == C10H15NO6 [M−H]⁻). The formula
    grid enumerates only the covalent reading (¹⁴NO₃ is deliberately NOT a scoring
    channel for the labelled profile, to avoid an uncontrolled exact-isobar
    competitor), so every ¹⁴NO₃ cluster is mislabeled as a covalent organonitrate.
    Mass alone cannot separate the two.

    Re-read [Y−H]⁻ (Y a covalent ¹⁴N-organonitrate: N≥1, O≥3, no ¹⁵N) as the cluster
    [X+NO₃]⁻ when the parent X = Y − HNO₃ is INDEPENDENTLY present -- assigned
    elsewhere as its own deprotonated [X−H]⁻ and/or as the ¹⁵N reagent cluster
    [X+¹⁵NO₃]⁻. Either channel is accepted (the lenient corroboration bar: [X−H]⁻
    proves X exists, [X+¹⁵NO₃]⁻ proves X clusters with nitrate; in a NOx run with a
    ~98% ¹⁵N reagent the chamber ¹⁴NO₃⁻ cluster then vastly outweighs the ¹⁵N one).
    Peaks whose parent is not independently seen keep the covalent organonitrate
    label -- this IS a NOx run and genuine organonitrates are real products.

    Tier is PRESERVED: the cluster reading rests on the SAME ion, mass and score as
    the covalent one (exact isobar) plus the extra parent corroboration, so whatever
    tier the covalent fit earned is exactly what the cluster reading deserves.

    Gated by the caller on the labelled-nitrate profile (label_isotope '^N'); it is
    only meaningful when ¹⁴NO₃ is off the scoring grid."""
    if "neutral_formula" not in ledger.columns or "adduct" not in ledger.columns:
        return {"nitrate_cluster_relabeled": 0}
    # parents independently present, by channel (normalised neutral strings).
    def _present(adduct):
        rows = ledger.index[ledger["adduct"].astype(str) == adduct]
        out = set()
        for j in rows:
            nf = _norm_formula(ledger.at[j, "neutral_formula"])
            if nf and C.parse_formula(nf).get("C", 0) >= 1:
                out.add(nf)
        return out
    parent_mh = _present("[M-H]-")
    parent_clu15 = _present("[M+^NO3]-")
    corroborated = parent_mh | parent_clu15
    if not corroborated:
        return {"nitrate_cluster_relabeled": 0}

    has_conf = "confidence" in ledger.columns
    target = (ledger.index[ledger["role"] == L.ROLE_M0]
              if "role" in ledger.columns else ledger.index)
    n = 0
    for i in target:
        if str(ledger.at[i, "adduct"]) != "[M-H]-":
            continue
        cnt = C.parse_formula(str(ledger.at[i, "neutral_formula"] or ""))
        if cnt.get("^N", 0):                          # a real ¹⁵N covalent product, not a ¹⁴NO₃ cluster
            continue
        if cnt.get("C", 0) < 1 or cnt.get("N", 0) < 1 or cnt.get("O", 0) < 3:
            continue                                   # not an organonitrate-ester reading
        # parent X = Y − HNO₃ (drop one covalent nitrate ester: -H, -N, -3 O).
        x = {e: cnt.get(e, 0) for e in set(cnt)}
        x["H"] = x.get("H", 0) - 1
        x["N"] = x.get("N", 0) - 1
        x["O"] = x.get("O", 0) - 3
        if any(x.get(e, 0) < 0 for e in ("H", "N", "O")):
            continue
        x = {e: v for e, v in x.items() if v > 0}
        if not C.dbe_ok(x)[0] or not C.oxygen_ok(x)[0]:
            continue                                   # X not a valid closed-shell neutral
        xf = C.format_formula(x)
        if xf not in corroborated:
            continue                                   # parent not independently detected
        via = ("[M-H]- + [M+^NO3]-" if (xf in parent_mh and xf in parent_clu15)
               else "[M-H]-" if xf in parent_mh else "[M+^NO3]-")
        ion = C.format_formula({**x, "N": x.get("N", 0) + 1, "O": x.get("O", 0) + 3}) + "-"
        ledger.at[i, "neutral_formula"] = xf
        ledger.at[i, "adduct"] = "[M+NO3]-"
        if "ion_formula" in ledger.columns:
            ledger.at[i, "ion_formula"] = ion
        if "dbe" in ledger.columns:
            ledger.at[i, "dbe"] = C.dbe(x)
        if has_conf:
            ledger.at[i, "confidence"] = "Good (¹⁴NO₃ cluster re-read)"
        note = (f"re-read as [{xf}+NO₃]⁻ chamber-¹⁴NO₃ cluster: exact isobar of the "
                f"covalent organonitrate [M−H]⁻, and the cluster parent {xf} is "
                f"independently detected (via {via}); in a ¹⁵N-nitrate NOx run the "
                "free chamber ¹⁴NO₃⁻ cluster dominates, so this is the ¹⁴NO₃ adduct, "
                "not a covalent organonitrate (mass cannot distinguish them)")
        if "commentary" in ledger.columns:
            prev = str(ledger.at[i, "commentary"] or "")
            ledger.at[i, "commentary"] = (prev + "; " + note) if prev and prev != "nan" else note
        if "tier_reason" in ledger.columns:
            ledger.at[i, "tier_reason"] = note
        n += 1
    log(f"[cleanup] re-read {n} covalent-organonitrate [M-H]- as chamber-¹⁴NO₃ clusters "
        f"(corroborated parent)")
    return {"nitrate_cluster_relabeled": n}


def demote_implausible_ionization(ledger: pd.DataFrame, *, log=print) -> dict:
    """Demote M0s whose ionization is chemically impossible for the assigned neutral.
    A PURE HYDROCARBON (no O/N/S/P/halogen/Si) has no acidic proton to lose and no
    H-bond donor / polar site to anchor an anion cluster, so it cannot ionize as
    [M-H]- or as a halide/carbonate/nitrate/sulfate/carboxylate cluster — regardless
    of how well the exact mass + isotope pattern fit (the pattern of a C/H ion just
    confirms the C count). Such an assignment is a mass coincidence (e.g. C7H10/C7H12
    [M-H]-, C2H2 [M+CO3]-): Assigned->Candidate + below_assignability. Electron
    attachment ([M]-./[M+O2]-) is exempt (the one route an electron-poor hydrocarbon
    has). Negative-mode anion channels only; positive adducts are left alone."""
    n = 0
    has_ba = "below_assignability" in ledger.columns
    target = (ledger.index[ledger["role"] == L.ROLE_M0]
              if "role" in ledger.columns else ledger.index)
    for i in target:
        cnt = C.parse_formula(str(ledger.at[i, "neutral_formula"] or ""))
        if cnt.get("C", 0) < 1:
            continue
        hetero = sum(cnt.get(e, 0) for e in ("O", "N", "S", "P", "F", "Cl", "Br", "I", "Si"))
        if hetero > 0:
            continue                                 # has a functional-group atom
        ad = str(ledger.at[i, "adduct"])
        if not ad.endswith("-") or ad in _EA_ADDUCTS:
            continue                                 # only FG-requiring anion channels
        if str(ledger.at[i, "tier"]) == "Assigned":
            ledger.at[i, "tier"] = "Candidate"
        if has_ba:
            ledger.at[i, "below_assignability"] = True
        if "commentary" in ledger.columns:
            note = (f"pure hydrocarbon via {ad}: no acidic proton / H-bond site to "
                    "ionize -- implausible (mass coincidence)")
            prev = str(ledger.at[i, "commentary"] or "")
            ledger.at[i, "commentary"] = (prev + "; " + note) if prev and prev != "nan" else note
        n += 1
    log(f"[cleanup] demoted {n} implausible-ionization M0 (heteroatom-free via anion channel)")
    return {"ionization_demoted": n}


def demote_speculative_residual(ledger: pd.DataFrame, cfg=None, *, log=print) -> dict:
    """Demote speculative residual-tail commits that reached Assigned on weak
    evidence (plausibility-audit rule-gaps). Targets ONLY method startswith
    'residual' (the pass-4 residual explainer / series gap-fill) — so pass-0/1/2
    grid analytes and known-species are untouched. Demote Assigned->Candidate +
    below_assignability when any of:
      * off-calibration: |z| > cal_z_accept (committed beyond the calibrated window);
      * uncorroborated multi-N: n_iso==0 AND N>=3 (a Br-doublet confirms the adduct
        Br, not the neutral's C/N backbone);
      * series gap-fill with no anchors ('0 supporting anchors' in the commentary);
      * sole minor-background channel (n_iso==0, adduct in minor_channels, and the
        neutral has no primary-channel partner)."""
    if "method" not in ledger.columns or "tier" not in ledger.columns:
        return {"residual_demoted": 0}
    minor = set(getattr(cfg, "minor_channels", None) or ("[M+CO3]-", "[M+O2]-", "[M]-."))
    mu = getattr(cfg, "cal_mu", None)
    sigma = getattr(cfg, "cal_sigma", None) or 0.5
    zacc = getattr(cfg, "cal_z_accept", 2.0)
    has_ba = "below_assignability" in ledger.columns
    is_m0 = ledger["role"] == L.ROLE_M0 if "role" in ledger.columns else pd.Series(True, index=ledger.index)
    # neutrals with a primary-channel (non-minor) M0 assignment somewhere
    primary = set()
    if "adduct" in ledger.columns:
        prim = ledger[is_m0 & ~ledger["adduct"].astype(str).isin(minor)]
        primary = set(prim["neutral_formula"].astype(str))
    n = 0
    for i in ledger.index[is_m0 & (ledger["tier"] == "Assigned")]:
        if not str(ledger.at[i, "method"] or "").startswith("residual"):
            continue
        cnt = C.parse_formula(str(ledger.at[i, "neutral_formula"] or ""))
        ni = _iso_count(ledger.at[i, "isotopologues"]) if "isotopologues" in ledger.columns else 0
        ad = str(ledger.at[i, "adduct"]) if "adduct" in ledger.columns else ""
        ppm = ledger.at[i, "ppm_error"]
        z = abs((float(ppm) - mu) / sigma) if (mu is not None and pd.notna(ppm)) else 0.0
        comm = str(ledger.at[i, "commentary"] or "")
        reason = None
        if mu is not None and z > zacc:
            reason = f"off-calibration (z={z:.1f} > {zacc})"
        elif ni == 0 and cnt.get("N", 0) >= 3:
            reason = f"N{cnt.get('N', 0)} with no isotope corroboration"
        elif "0 supporting anchors" in comm:
            reason = "series gap-fill with no supporting anchors"
        elif ni == 0 and ad in minor and str(ledger.at[i, "neutral_formula"]) not in primary:
            reason = f"sole minor channel ({ad}), no isotope or primary-channel support"
        if reason is None:
            continue
        ledger.at[i, "tier"] = "Candidate"
        if has_ba:
            ledger.at[i, "below_assignability"] = True
        if "commentary" in ledger.columns:
            note = f"speculative residual fit -- {reason}; not Assigned-grade"
            prev = comm
            ledger.at[i, "commentary"] = (prev + "; " + note) if prev and prev != "nan" else note
        n += 1
    log(f"[cleanup] demoted {n} speculative-residual M0 (weak Assigned residual fits)")
    return {"residual_demoted": n}


# ---- reagent-precursor / brominated-background halocarbons -----------------
# A bromomethane reagent-precursor fragment (CH2Br2 -> CHBr2-, m/z 170.845) is
# MASS-DEGENERATE with an absurd bare-element + reagent-cluster reading (neutral
# "C" via [M+HBr+Br]-): the SAME ion at the SAME mass. Scoring ties exactly and
# the neutral-halogen complexity penalty then hands the win to the bare-element
# cluster, so the report names "neutral C". We catch these on the INVARIANT ION
# COMPOSITION (independent of which neutral the fitter guessed -> robust to
# ion-formula string ordering). The >=2 bromines are isotope-confirmable and the
# match is exact composition, so this is safe. Br-CIMS only.
#   (ion neutral-composition, true neutral, true adduct, label, role)
_REAGENT_HALOCARBONS = [
    ("CHBr2",    "CH2Br2",    "[M-H]-", "dibromomethane (Br-CIMS reagent precursor)", L.ROLE_REAGENT),
    ("CBr3",     "CHBr3",     "[M-H]-", "bromoform (Br-CIMS reagent precursor)",      L.ROLE_REAGENT),
    ("C2HBr2O2", "C2H2Br2O2", "[M-H]-", "dibromoacetic acid (brominated background)", L.ROLE_M0),
]


def _ion_comp(s):
    """Canonical element-count tuple of an ion formula (charge/radical stripped)."""
    try:
        return tuple(sorted(C.parse_formula(str(s).rstrip("+-.")).items()))
    except Exception:
        return None


def relabel_reagent_halocarbons(ledger: pd.DataFrame, *, reagent=None, log=print) -> dict:
    """Reclassify bromomethane reagent-precursor / brominated-background ions that
    the degenerate 'bare element + [M+HBr+Br]-' reading mislabels (see note above).
    Reagent precursors (CH2Br2/CHBr3) -> role=reagent, out of the analyte pool;
    a named brominated background (dibromoacetic acid) keeps M0 but gets its real
    neutral + a note. Matched on the invariant ion composition. Br-CIMS only -- gate
    on the REAGENT ELEMENT (cfg.reagent_element), not the context profile."""
    if "ion_formula" not in ledger.columns or reagent != "Br":
        return {"relabeled": 0}                    # bromide-reagent specific
    targets = {c: t for t in _REAGENT_HALOCARBONS
               for c in (_ion_comp(t[0]),) if c}
    comps = ledger["ion_formula"].map(_ion_comp)
    n = 0
    for i in ledger.index:
        t = targets.get(comps.at[i])
        if t is None or bool(ledger.at[i, "locked"]):
            continue
        _, neutral, adduct, label, role = t
        if role == L.ROLE_REAGENT:
            L.mark_reagent(ledger, ledger.at[i, "peak_id"], f"reagent precursor: {label}")
        else:                                      # named background: fix neutral+adduct so
            ledger.at[i, "neutral_formula"] = neutral   # (neutral, adduct) match the ion
            ledger.at[i, "adduct"] = adduct
            ledger.at[i, "commentary"] = label
        n += 1
    if n:
        log(f"[cleanup] reagent-halocarbon relabel: {n} ion(s) reclassified")
    return {"relabeled": n}


def run_cleanup(client, sample_id, ledger, profile, cfg, *, log=print) -> dict:
    """Orchestrate the cleanup steps (recovery first, so a recovered molecule
    isn't then mislabelled a cluster/artifact; satellite reclaim last)."""
    rec = recover_isotope_gated(client, sample_id, ledger, profile, cfg, log=log)
    # Br-CIMS only: the defect+1.998-twin heuristic reads ANY heavy-halogen
    # cluster region as "bromide" (under iodide it grabbed the I2NO2-/IBr.I-
    # neighborhood with a false bromide note), so gate on the reagent element.
    if getattr(cfg, "reagent_element", None) == "Br":
        clu = label_bromide_clusters(ledger, client, sample_id, log=log)
    else:
        clu = {"labelled": 0, "covalent_ties": 0}
    rhc = relabel_reagent_halocarbons(ledger, reagent=getattr(cfg, "reagent_element", None),
                                      log=log)
    art = flag_ringing_artifacts(ledger, log=log)
    sat = reclaim_satellites(ledger, log=log)
    tails = reclaim_envelope_tails(ledger, log=log)   # deep poly-halogen envelope (k>=2)
    # NB: demote_unconfirmed_fluorine is NOT called here -- it must run AFTER
    # tiers.apply_tiers (which recomputes tier and would re-promote the F-monster).
    # assign.run calls it post-tiering.
    return {"recovered": rec["recovered"], "clusters": clu["labelled"],
            "cluster_covalent_ties": clu.get("covalent_ties", 0),
            "reagent_halocarbons": rhc["relabeled"],
            "artifacts": art["flagged"], "reclaimed_satellites": sat["reclaimed"],
            "envelope_tails": tails["tails"]}


def prefer_amine_over_ammonium(ledger: pd.DataFrame, *, ts_peaks=None,
                               r_min: float = 0.6, r_reject: float = 0.2,
                               min_overlap: int = 12, protected=None,
                               log=print) -> dict:
    """Uronium / positive urea-CIMS: a [M+NH4]+ adduct of a CHO neutral X is mass-
    AND isotope-identical to [M+H]+ of the amine X+NH3 (the SAME ion formula), so
    the accurate mass + isotope pattern CANNOT distinguish an ammonium adduct of X
    from a protonated amine X+NH3. The one discriminator is TIME: a true ammonium
    adduct of X must track X's own [M+H]+/urea parent trace, because it is the same
    molecule ionised two ways.

    POLICY -- ammonium adducts are rare, so the burden of proof is on the ADDUCT
    reading; the protonated CHON wins by default. Per [M+NH4]+-mass neutral X, with a
    batch time series (2 h-binned log-correlation r of the [M+NH4]+ trace vs the best
    of X's [M+H]+ / [M+(CH4N2O)H]+ parent traces):

      * KEEP as [M+NH4]+ adduct (Assigned) ONLY when it TRACKS a shaped parent
        (r >= r_min, overlap >= min_overlap) -- confirmed same molecule.
      * OTHERWISE default to [M+H]+ of the protonated CHON (amine X+NH3), at
        Candidate tier: independent time trace (r <= r_reject), only weak tracking
        (r_reject<r<r_min), a flat unconfirmable parent, or a parent absent from the
        TS. We do NOT assert an ammonium adduct we cannot confirm.

    Overriding exceptions (keep [M+NH4]+):
      * X contains Si: ammonium adducts of siloxanes are real contaminant chemistry;
        the +NH3 re-read would fabricate an "aminosiloxane". (Tier unchanged.)
      * X is PROTECTED -- identity established by curation/cross-channel evidence
        independent of the NH4 channel (reflist-rescue / known-species / certified
        provenance; pass the neutral-formula set as `protected`). Holds e.g. NBBS.
        (Tier unchanged.)
      * the amine X+NH3 is valence-impossible (saturated X -> negative DBE): the
        ammonium reading is the only valid one, kept but capped Candidate.

    Without a time series (single-sample run) nothing can be confirmed, so every
    non-Si/protected [M+NH4]+ defaults to the protonated CHON (Candidate).

    A re-read changes neutral_formula + adduct and caps the tier at Candidate (the
    ion/m-z/score/ppm are identical). Mutates in place; returns a summary."""
    keys = ["relabeled", "kept_covary", "kept_protected", "kept_si", "forced_nh4"]
    if not {"neutral_formula", "adduct"} <= set(ledger.columns):
        return {k: 0 for k in keys}
    protected = {str(p) for p in (protected or ())}
    is_m0 = (ledger["role"] == L.ROLE_M0) if "role" in ledger.columns \
        else pd.Series(True, index=ledger.index)

    verdict = _covariation_verdict(ledger, ts_peaks, r_min, r_reject, min_overlap)
    counts = {k: 0 for k in keys}

    def _cap_tier(i):
        if "tier" in ledger.columns and str(ledger.at[i, "tier"]) == "Assigned":
            ledger.at[i, "tier"] = "Candidate"

    def _reread(i, X, *, why):
        """DEFAULT reading for an unconfirmed [M+NH4]+ mass: read it as [M+H]+ of the
        protonated CHON (the amine X+NH3), at Candidate tier. Ammonium adducts are
        rare, so the CHON reading wins unless the adduct is confirmed by tracking. If
        the amine is valence-impossible the ammonium reading is the only valid one --
        keep it, but still cap to Candidate (it is unconfirmed)."""
        cnt = C.parse_formula(X)
        cnt["N"] = cnt.get("N", 0) + 1
        cnt["H"] = cnt.get("H", 0) + 3
        ok, _ = C.dbe_ok(cnt)
        if not ok:                                        # amine impossible -> keep NH4
            _cap_tier(i)
            _note(ledger, i, f"[M+NH4]+ kept (amine valence-impossible), unconfirmed "
                  f"({why}); capped Candidate")
            counts["forced_nh4"] += 1
            return
        ledger.at[i, "neutral_formula"] = C.format_formula(cnt)
        ledger.at[i, "adduct"] = "[M+H]+"
        _cap_tier(i)
        _note(ledger, i, f"read as protonated CHON (amine {C.format_formula(cnt)}), "
              f"Candidate: [M+NH4]+ of {X} not confirmed ({why})")
        counts["relabeled"] += 1

    _WHY = {"reject": "independent time trace",
            "ambiguous": "only weak tracking",
            "presence-cap": "parent flat, adduct unconfirmable",
            "presence-reread": "parent channels absent from TS",
            "presence-keep": "no time series to confirm"}

    for i in ledger.index[is_m0 & (ledger["adduct"] == "[M+NH4]+")]:
        X = str(ledger.at[i, "neutral_formula"] or "")
        if not X or X == "nan":
            continue
        if C.parse_formula(X).get("Si"):                  # siloxane adduct -> keep
            counts["kept_si"] += 1
            continue
        if X in protected:                                # curated identity -> keep
            counts["kept_protected"] += 1
            continue
        v, r, ov = verdict(X)
        if v == "keep":                                   # tracks a shaped parent -> real adduct
            _note(ledger, i, f"[M+NH4]+ confirmed: tracks parent (r={r:.2f})")
            counts["kept_covary"] += 1
            continue
        why = _WHY.get(v, "unconfirmed")
        if v in ("reject", "ambiguous"):
            why += f" (r={r:.2f})"
        _reread(i, X, why=why)                            # DEFAULT: protonated CHON, Candidate

    mode = "co-variation" if ts_peaks is not None else "presence"
    log(f"[uronium-amine] ({mode}) default->CHON: {counts['relabeled']} read as "
        f"protonated CHON (Candidate); {counts['kept_covary']} kept as [M+NH4]+ adduct "
        f"(tracks parent); {counts['kept_protected']} protected + {counts['kept_si']} Si "
        f"kept; {counts['forced_nh4']} amine-impossible kept-NH4")
    return counts


def _note(ledger, i, msg):
    if "tier_reason" in ledger.columns:
        ledger.at[i, "tier_reason"] = (
            str(ledger.at[i, "tier_reason"] or "") + f" | {msg}").strip(" |")


def _covariation_verdict(ledger, ts_peaks, r_min, r_reject, min_overlap):
    """Return verdict(neutral) -> (tag, r, overlap). The tag drives the gate:

      'keep'            r >= r_min against a SHAPED parent -> real adduct.
      'reject'          r <= r_reject against a shaped parent -> re-read to amine.
      'ambiguous'       r between the thresholds -> keep but cap Candidate.
      'presence-keep'   no shaped parent overlaps testably, but X IS assigned as
                        [M+H]+/urea in the ledger -> (no-TS path) keep uncapped.
      'presence-cap'    same, but the TS path caps it Candidate (a present-but-flat
                        parent cannot confirm the adduct).
      'presence-reread' no shaped parent AND X is absent as [M+H]+/urea -> the
                        parent channels are missing, which refutes the ammonium
                        reading (a real X would protonate / form the dominant urea
                        adduct); re-read to the amine.

    A parent is a valid correlation target only if it is SHAPED (cv >= FLAT_CV): a
    flat parent (steady background) has no time structure to track, and two flat
    traces correlate at noise level, so flat parents route to the presence branch
    instead of producing a spurious reject. Without a time series the gate degrades
    to the binary presence test ('presence-keep' / 'presence-reread')."""
    corrob = set(ledger.loc[ledger["adduct"].isin(["[M+H]+", "[M+(CH4N2O)H]+"]),
                            "neutral_formula"].dropna().astype(str))
    if ts_peaks is None:
        return lambda X: (("presence-keep" if X in corrob else "presence-reread"),
                          float("nan"), 0)

    import numpy as np
    from peaky.batch import cluster as CL
    mat, tbins = _binned_matrix(ts_peaks)
    bm = tbins.sort_values(); arr = bm.to_numpy(); idx = bm.index.to_numpy()

    def _logtrace(neutral, adduct):
        try:
            mz = C.ion_mz(neutral, adduct)
        except Exception:
            return None
        i = np.searchsorted(arr, mz); best = None
        for j in (i - 1, i):
            if 0 <= j < len(arr):
                ppm = abs(arr[j] - mz) / mz * 1e6
                if ppm <= 8 and (best is None or ppm < best[1]):
                    best = (idx[j], ppm)
        if best is None:
            return None
        col = mat[best[0]].to_numpy(float).copy()
        col[~(col > 0)] = np.nan                          # undetected bins -> NaN
        return np.log10(col)

    def _shaped(logtrace):                                # cv over detected bins
        lin = 10.0 ** logtrace[np.isfinite(logtrace)]
        m = float(lin.mean()) if len(lin) else 0.0
        return m > 0 and float(lin.std()) / m >= CL.FLAT_CV

    def verdict(X):
        nh4 = _logtrace(X, "[M+NH4]+")
        best_r, best_ov = float("nan"), 0
        shaped_parent = False           # did any testable parent carry real shape?
        if nh4 is not None:
            for ref in ("[M+H]+", "[M+(CH4N2O)H]+"):
                t = _logtrace(X, ref)
                if t is None:
                    continue
                ok = np.isfinite(nh4) & np.isfinite(t)
                if int(ok.sum()) < min_overlap:
                    continue                              # sparse parent: cannot correlate
                shaped_parent = shaped_parent or _shaped(t[ok])
                rr = float(np.corrcoef(nh4[ok], t[ok])[0, 1])
                if not np.isfinite(best_r) or rr > best_r:
                    best_r, best_ov = rr, int(ok.sum())
        if np.isfinite(best_r):
            if best_r >= r_min:                           # co-varies -> real adduct
                return ("keep", best_r, best_ov)          # (high r itself proves shape)
            if best_r <= r_reject and shaped_parent:      # a shaped parent it fails to track
                return ("reject", best_r, best_ov)        # -> independent -> amine
            if best_r > r_reject:                         # weak-but-positive tracking
                return ("ambiguous", best_r, best_ov)
            # low r but every testable parent is FLAT: correlation is noise, not a
            # verdict -> fall through to presence.
        # no shaped parent gave a verdict -> presence decides. Present (even if
        # flat) keeps the ammonium reading but capped; absent refutes it -> re-read.
        return (("presence-cap" if X in corrob else "presence-reread"), best_r, best_ov)
    return verdict


def _binned_matrix(ts_peaks, *, bin_hours: float = 2.0):
    """Time-bin a batch peak table into a (time-bin x m/z-bin) median-intensity
    matrix. Denoises the per-sample trace and aligns irregular acquisition onto a
    fixed grid so the [M+NH4]+/parent log-correlation is stable. Falls back to the
    raw per-sample matrix when the table has no `datetime_utc`."""
    import numpy as np
    from peaky.batch import timeseries as TS
    mat, bin_mz = TS.build_matrix(ts_peaks)
    if "datetime_utc" not in getattr(ts_peaks, "columns", ()):
        return mat, bin_mz
    tmap = (ts_peaks[["sample_item_id", "datetime_utc"]].drop_duplicates()
            .set_index("sample_item_id")["datetime_utc"])
    dt = pd.to_datetime(tmap.reindex(mat.index))
    if dt.isna().all():
        return mat, bin_mz
    step = np.timedelta64(int(bin_hours * 3600), "s")
    binkey = dt.values.astype("datetime64[s]").astype("int64") // step.astype("int64")
    binned = mat.groupby(binkey).median()
    return binned, bin_mz
