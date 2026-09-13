"""Dedicated PDMS / siloxane-ladder assignment.

The heavy unexplained residual of a positive-mode source is often a
polydimethylsiloxane (silicone / column-bleed) oligomer ladder: peaks spaced by
exactly +C2H6OSi (74.0188 Da, the dimethylsiloxane repeat). These are
MASS-DEGENERATE per peak -- many CHON O-monsters fit the same accurate mass at
the calibrated offset and out-score the true Si formula in per-peak arbitration
(O is free in the complexity prior), so the general passes either miss them or
commit a monster that a later CHON-centric audit then clears. The result is a
bright ladder stuck in 'unexplained'.

This pass assigns them decisively using the two pieces of evidence the general
machinery does not combine:

  1. the +C2H6OSi LADDER (>= MIN_LADDER members spaced by 74.0188 within ppm), and
  2. the Si ISOTOPE ENVELOPE -- a Si_k species has an M+1 dominated by 29Si
     (4.685%/Si) far larger than the 13C of the competing CHON monster, plus a
     30Si M+2; the oracle's isotopologue match confirms the Si count.

It runs LATE (after the audits) and LOCKS its commits, so the CHON-centric
heuristics (carbon clamp reading 29Si as 13C, O-monster mass-fits) cannot undo
it. Tier is Candidate: the ladder identity (siloxane oligomer) is certain, the
exact per-rung Si count is the oracle's best isotope-corroborated estimate.
"""
from __future__ import annotations

import bisect

import pandas as pd

from peaky.chem import chemistry as C
from peaky.assignment import ledger as L
from peaky.assignment.passes import PassConfig, z_of, _mech_to_adduct, _f

__version__ = "0.2.0"  # work set + seed test draw from the admission gate

SILOXANE_UNIT = "C2H6OSi"
UNIT_MASS = C.neutral_mass(SILOXANE_UNIT)        # 74.01879
MIN_LADDER = 3                                    # peaks to call it a ladder
LADDER_PPM = 4.0                                  # spacing tolerance

# Si-count INTENSITY gate (from the AP low-T cross-pipeline audit): the 29Si M+1
# must be ~consistent with the CLAIMED Si count, not merely matched by the oracle.
# Predicted M+1 fraction = nSi*29Si-abund + nC*13C-abund; a peak whose M+1 is far
# below that is over-claiming Si (a high-O HOM has only a small 13C M+1) -- e.g.
# C8H26O5Si4 @393.004 with M+1 ~13% where Si4 needs ~27% was the HOM C10H18O11.
_SI29_ABUND = 0.0468       # 29Si natural abundance (per Si)
_C13_ABUND = 0.0107        # 13C natural abundance (per C)
_D_29SI = 0.999568         # 29Si - 28Si mass
SI_M1_MIN_FRAC = 0.6       # observed (M+1)/(M0) must be >= this * predicted


def _m1_ratio(smz: list, sh: list, m0_mz: float, m0_h: float):
    """Observed (M+1)/(M0) height ratio: nearest peak ~+1 Da (the 29Si/13C M+1).
    Returns 0.0 when no peak sits there, None when the M0 height is unusable."""
    if m0_h <= 0:
        return None
    t = m0_mz + _D_29SI
    j = bisect.bisect_left(smz, t); best = None
    for k in (j - 1, j):
        if 0 <= k < len(smz) and abs(smz[k] - t) < 0.006:
            if best is None or abs(smz[k] - t) < abs(smz[best] - t):
                best = k
    return (sh[best] / m0_h) if best is not None else 0.0


def _find_ladders(mzs: list[float], heights: list[float], *,
                  min_height: float) -> list[list[int]]:
    """Index chains of peaks spaced by UNIT_MASS (+C2H6OSi). Greedy: from each
    seed walk +unit while a peak exists within LADDER_PPM."""
    order = sorted(range(len(mzs)), key=lambda i: mzs[i])
    sm = [mzs[i] for i in order]
    used: set[int] = set()
    chains: list[list[int]] = []
    for s in range(len(order)):
        if order[s] in used or heights[order[s]] < min_height:
            continue
        chain = [order[s]]
        cur = sm[s]
        while True:
            target = cur + UNIT_MASS
            j = bisect.bisect_left(sm, target)
            best = None
            for k in (j - 1, j):
                if 0 <= k < len(sm) and abs(sm[k] - target) / target * 1e6 <= LADDER_PPM:
                    if best is None or abs(sm[k] - target) < abs(sm[best] - target):
                        best = k
            if best is None or order[best] in used:
                break
            chain.append(order[best])
            cur = sm[best]
        if len(chain) >= MIN_LADDER:
            chains.append(chain)
            used.update(chain)
    return chains


def _si_box(mz: float) -> dict:
    """A CHOSi(N) enumeration box wide enough for a long PDMS oligomer at m/z."""
    return C.parse_ranges("C2-28 H6-84 N0-2 O1-16 Si1-13")


def assign_siloxane_ladder(client, sample_id: str, ledger: pd.DataFrame,
                           profile, cfg: PassConfig, *, adducts=None,
                           score_fn=None, log=print) -> dict:
    """Assign the +C2H6OSi PDMS ladder. Commits (locked, Candidate) the best
    on-trend, Si-isotope-corroborated CHOSi formula for each ladder member,
    displacing an UNLOCKED non-Si commit. Returns a summary dict."""
    from peaky.io import io_mascope as IO
    out = {"ladders": 0, "members": 0, "committed": 0, "displaced": 0,
           "iso_attached": 0}
    if getattr(profile, "max_Si", 0) < 3:
        return out                                   # context forbids siloxanes
    score_fn = score_fn or IO.score_candidates
    adducts = adducts or list(profile.reagent_adducts)
    gad = [a for a in adducts if a in C.ADDUCT_SHIFTS]

    from peaky.assignment import admission as ADM
    work = ledger[ADM.admissible(ledger, cfg)]
    mzs = work["mz"].tolist(); hts = work["height"].tolist()
    # `work` IS the admission-gated set (brightness OR persistence), so every
    # member may seed a chain: a persistent sub-gate peak admitted into `work`
    # must not be refused as a seed by a second, brightness-only test (the
    # lowest rung of a weak ladder is exactly such a peak).
    chains = _find_ladders(mzs, hts, min_height=0.0)
    if not chains:
        return out
    member_mz = sorted({mzs[i] for ch in chains for i in ch})
    out["ladders"] = len(chains); out["members"] = len(member_mz)

    # one enumeration + one oracle call over all ladder members
    formulas: set[str] = set()
    for mz in member_mz:
        for f in C.candidates_for_peaks([mz], _si_box(mz), gad,
                                        ppm_tolerance=cfg.search_ppm,
                                        mass_min=mz - 80, mass_max=mz + 2):
            if C.parse_formula(f).get("Si", 0) >= 3 and C.dbe_ok(f)[0]:
                formulas.add(f)
    if not formulas:
        return out
    scored = score_fn(client, sample_id, sorted(formulas),
                      mechanism_ids=cfg.mechanism_ids, allow_partial=True)
    if scored is None or len(scored) == 0:
        log("[siloxane] oracle returned nothing for the ladder candidates")
        return out
    base = scored[scored["is_base"] & scored["sample_peak_id"].notna()
                  & scored["ion_score"].notna()].copy()
    # Si-isotopologue confirmation set: (compound, ion) with a matched 29Si/30Si
    iso = scored[(~scored["is_base"]) & scored["sample_peak_id"].notna()
                 & (pd.to_numeric(scored["iso_score"], errors="coerce").fillna(0) > 0.4)]
    si_confirmed = {(r.compound_formula, r.ion_formula)
                    for r in iso.itertuples()
                    if "Si" in str(getattr(r, "iso_label", ""))}
    kids = iso

    mzs_all = ledger["mz"]
    _pairs = sorted((float(m), float(h) if pd.notna(h) else 0.0)
                    for m, h in zip(ledger["mz"], ledger["height"]))
    _smz = [p[0] for p in _pairs]; _sh = [p[1] for p in _pairs]
    for mz in member_mz:
        sub = base[(base["sample_peak_mz"].astype(float) - mz).abs() < 0.012].copy()
        if sub.empty:
            continue
        # on-trend only (calibrated); Si-isotope envelope must corroborate
        sub["z"] = sub.apply(lambda r: z_of(r["ppm_error"], cfg, r["sample_peak_mz"]), axis=1)
        sub = sub[sub["z"].notna() & (sub["z"] <= cfg.cal_z_pattern)]
        sub["si_ok"] = sub.apply(
            lambda r: (r["compound_formula"], r["ion_formula"]) in si_confirmed, axis=1)
        sub = sub[sub["si_ok"]]
        if sub.empty:
            continue
        sub["pen"] = sub["compound_formula"].map(C.complexity_penalty)
        sub["eff"] = sub["ion_score"].astype(float) - sub["pen"] - sub["z"] * 0.02
        r = sub.sort_values("eff", ascending=False).iloc[0]
        pid = r["sample_peak_id"]
        # locate the ledger row + its current role
        idx = ledger.index[ledger["peak_id"] == pid]
        if not len(idx):
            continue
        i0 = idx[0]
        # Si-count intensity gate: the 29Si M+1 must be ~consistent with the claimed
        # Si count, not merely matched. Skip when the M+1 is far below what nSi
        # predicts (a high-O HOM masquerading via a 13C-only M+1). Checked BEFORE any
        # displacement so a real assignment is never cleared for an over-claimed Si fit.
        ion = C.parse_formula(r["ion_formula"])
        nSi = ion.get("Si", 0); nCi = ion.get("C", 0)
        pred_m1 = nSi * _SI29_ABUND + nCi * _C13_ABUND
        m0h = float(ledger.at[i0, "height"]) if pd.notna(ledger.at[i0, "height"]) else 0.0
        obs_m1 = _m1_ratio(_smz, _sh, float(r["sample_peak_mz"]), m0h)
        if nSi > 0 and pred_m1 > 0 and obs_m1 is not None and obs_m1 < SI_M1_MIN_FRAC * pred_m1:
            log(f"[siloxane] skip {r['compound_formula']} @{mz:.4f}: 29Si M+1 ratio "
                f"{obs_m1:.3f} << {SI_M1_MIN_FRAC:g}x predicted {pred_m1:.3f} "
                f"(Si{nSi} over-claimed)")
            out["si_underclaimed"] = out.get("si_underclaimed", 0) + 1
            continue
        role = ledger.at[i0, "role"]
        if role == L.ROLE_M0 and bool(ledger.at[i0, "locked"]):
            continue                                  # never override a locked ID
        if role == L.ROLE_M0:
            try:
                L.clear_assignment(ledger, pid, reason="displaced by siloxane ladder")
                out["displaced"] += 1
            except L.LedgerError:
                continue
        elif role != L.ROLE_UNEXPLAINED:
            continue                                  # iso_child / reagent / artifact
        nC = C.parse_formula(r["compound_formula"]).get("Si", 0)
        try:
            L.commit_assignment(
                ledger, pid, neutral_formula=r["compound_formula"],
                adduct=_mech_to_adduct(r), ion_formula=r["ion_formula"],
                ion_score=float(r["ion_score"]), compound_score=_f(r.get("compound_score")),
                ppm_error=float(r["ppm_error"]), pass_no=7, method="siloxane:ladder",
                confidence="Low (siloxane-ladder)",
                commentary=(f"PDMS/siloxane oligomer (Si{nC}); member of a "
                            f"+C2H6OSi (+74.019) ladder, 29Si/30Si envelope "
                            f"confirmed; exact per-rung Si count is the isotope-"
                            f"corroborated best estimate (mass-degenerate)."))
            L.lock_peaks(ledger, [pid])
            out["committed"] += 1
            # attach the Si/13C satellites this compound/ion confirmed
            for k in kids[(kids["compound_formula"] == r["compound_formula"])
                          & (kids["ion_formula"] == r["ion_formula"])].itertuples():
                if k.sample_peak_id and k.sample_peak_id != pid:
                    try:
                        L.attach_isotopologue(ledger, k.sample_peak_id, pid,
                                              iso_label=k.iso_label,
                                              iso_match_score=_f(k.iso_score))
                        out["iso_attached"] += 1
                    except L.LedgerError:
                        pass
        except L.LedgerError:
            continue
    log(f"[siloxane] {out}")
    return out
