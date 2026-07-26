"""Labelled-reagent heavy-isotope rescue (single-file, runs after cleanup /
before tiers).

In a labelled-reagent run (e.g. ¹⁵N-nitrate CIMS) the reagent radical can add to
a VOC and leave a covalent heavy atom in the PRODUCT — a ¹⁵N-organonitrate. The
formula grid enumerates only the light isotope, so every such product sits j·Δ
(Δ = m(¹⁵N) − m(¹⁴N) = 0.99703 Da) off any formula the grid can express. The
assigner then either leaves the peak unexplained or lets a flexible composition
(a partially-fluorinated CHONF fit) absorb the mass shift — exactly the F-fits the
plausibility/tier F-gates now demote.

This pass repairs the ROOT cause for the labelled profile only: for each unexplained
peak (and each F-bearing M0 fit), it re-enumerates the CHON grid at mass − j·Δ with
N ≥ j, substitutes j nitrogens with ¹⁵N (`^N`), and scores the heavy-isotope
reading against the real spectrum. A confident, chemically-plausible ¹⁵N reading
replaces the fit (or fills the residual). j ∈ 1..`label_max` (2 = mono/di-nitrate).

Gated entirely on `profile.label_isotope`; a no-op for every unlabelled profile.
"""
from __future__ import annotations

import pandas as pd

from peaky.chem import chemistry as C
from peaky.chem import contexts as X
from peaky.io import io_mascope as IO
from peaky.assignment import ledger as L

__version__ = "0.1.0"

DELTA_15N = C.M["^N"] - C.M["N"]        # 0.99703 Da
# CHON(S) box for the labelled re-enumeration (organonitrate products); wider O
# than a hydrocarbon box because nitrate chemistry is oxidative.
LABELED_BOX = "C0-40 H0-60 N0-4 O0-25 S0-1"


def _n15_variants(neutral_counts: dict, base: str, label: str, j: int) -> dict | None:
    """Move j atoms of `base` to `label` (e.g. 'N' -> '^N'). None if too few."""
    if neutral_counts.get(base, 0) < j:
        return None
    out = dict(neutral_counts)
    out[base] -= j
    out[label] = out.get(label, 0) + j
    if out[base] == 0:
        del out[base]
    return out


def rescue_labeled(client, sample_id, ledger, context_profile, cfg, *,
                   adducts, label_isotope=None, label_max=2,
                   score_fn=None, log=print) -> dict:
    """Fill/repair peaks that are really heavy-isotope products of a labelled
    reagent. `context_profile` is the ContextProfile (element caps) used to keep
    the light-isotope skeleton chemically plausible. Returns
    {'rescued': n_unexplained_filled, 'reread': n_fits_swapped}."""
    label = label_isotope                                  # '^N'
    if not label or "neutral_formula" not in ledger.columns:
        return {"rescued": 0, "reread": 0}
    base = label.lstrip("^")                               # 'N'
    kmax = int(label_max)
    adducts = list(adducts or [])
    score_fn = score_fn or IO.score_candidates
    tau = float(getattr(cfg, "tau_good", 0.8))
    ppm_tol = float(getattr(cfg, "search_ppm", 3.0))
    box = C.parse_ranges(LABELED_BOX)

    # targets: the residual (unexplained) + any F-bearing M0 fit (the mass-shift
    # absorbers). role column present on a single-file ledger.
    role = ledger.get("role")
    is_m0 = role.astype(str).eq(L.ROLE_M0) if role is not None else pd.Series(True, index=ledger.index)
    is_un = role.astype(str).eq(L.ROLE_UNEXPLAINED) if role is not None else pd.Series(False, index=ledger.index)

    def _partial_f(f):
        # a PARTIALLY-fluorinated fit (F>=1, F<2H) is the mass-shift absorber to
        # repair; a genuine (per/poly)fluoro species (F>=2H, e.g. PFCA/TFA) is
        # real chamber PFAS background and MUST be left alone.
        d = C.parse_formula(f)
        nf = d.get("F", 0)
        return nf >= 1 and nf < 2 * d.get("H", 0)
    fbear = ledger["neutral_formula"].astype(str).map(_partial_f)
    targets = ledger.index[is_un | (is_m0 & fbear)]
    if not len(targets):
        return {"rescued": 0, "reread": 0}

    # per-target: enumerate 15N candidate neutrals; union for ONE batched score.
    per: dict = {}          # peak_id -> {'mz','is_un','cur_score','cands':set}
    allf: set = set()
    for i in targets:
        mz = ledger.at[i, "mz"]
        if pd.isna(mz):
            continue
        mz = float(mz)
        cands: set = set()
        for a in adducts:
            if a not in C.ADDUCT_SHIFTS:
                continue
            m_neu = mz - C.ADDUCT_SHIFTS[a]
            for j in range(1, kmax + 1):
                # a j-labelled product's LIGHT-isotope mass is m_neu - j*Δ
                light = m_neu - j * DELTA_15N
                if light < 30:
                    continue
                for f in C.candidates_for_peaks([light + C.ADDUCT_SHIFTS[a]], box, [a],
                                                ppm_tolerance=ppm_tol):
                    d = C.parse_formula(f)
                    v = _n15_variants(d, base, label, j)
                    if v is None:
                        continue
                    if not C.dbe_ok(v)[0] or not C.oxygen_ok(v)[0]:
                        continue
                    if not X.filter_by_profile(C.format_formula(C.fold_isotopes(v)), context_profile)[0]:
                        continue
                    cands.add(C.format_formula(v))
        if cands:
            cur = ledger.at[i, "ion_score"] if "ion_score" in ledger.columns else None
            per[i] = {"mz": mz, "is_un": bool(is_un.get(i, False)),
                      "cur": (float(cur) if pd.notna(cur) else 0.0)}
            per[i]["cands"] = cands
            allf |= cands
    if not allf:
        log(f"[labeled] no {label} candidates for {len(targets)} target peaks")
        return {"rescued": 0, "reread": 0}

    fr = score_fn(client, sample_id, sorted(allf), allow_partial=True,
                  mechanism_ids=getattr(cfg, "mechanism_ids", None))
    if fr is None or not len(fr) or "sample_peak_id" not in fr.columns:
        log(f"[labeled] scorer returned no {label} matches")
        return {"rescued": 0, "reread": 0}
    fr = fr[fr["sample_peak_id"].notna()]

    # per-(compound,ion) isotopologue corroboration: a real 15N product must show
    # a matched NON-M0 sibling (the 2% 14N-impurity peak at M0-Δ is the decisive
    # 15N-specific confirmation; a matched 13C/18O also corroborates the skeleton).
    corrob = (fr[~fr["is_base"] & fr["sample_peak_id"].notna()]
              .groupby(["compound_formula", "mechanism_id"]).size())

    # A real 15N product sits at the instrument's OWN mass accuracy (the calibrated
    # 14N core: cal_mu +- cal_sigma). Accept a 15N reading only INSIDE that window
    # -- exactly the tier engine's z-tail gate (Z_TAIL). A blind +-2 ppm window let
    # in off-calibration coincidences (15N mass fits landing ~2 ppm from an
    # unrelated peak) that the tier engine then demoted anyway. Gate here so the
    # rescue never proposes a fill it cannot defend.
    cal_mu = getattr(cfg, "cal_mu", None)
    cal_sigma = getattr(cfg, "cal_sigma", None) or 0.5
    Z_TAIL = 2.6
    def _on_cal(ppm):
        if ppm is None or pd.isna(ppm):
            return False
        if cal_mu is None:
            return abs(float(ppm)) <= 1.0
        return abs((float(ppm) - cal_mu) / cal_sigma) <= Z_TAIL

    n_res = n_re = 0
    for i, info in per.items():
        mz = info["mz"]
        sub = fr[(fr["sample_peak_mz"] - mz).abs() < mz * 2.0 * 1e-6]
        sub = sub[sub["compound_formula"].isin(info["cands"]) & (sub["ion_score"] >= tau)]
        if sub.empty:
            continue
        # discipline: on-calibration mass + organonitrate plausibility + isotope
        # corroboration + not mass-degenerate. Filter candidates BEFORE the winner.
        keep_idx = []
        for j, cr in sub[sub["is_base"]].iterrows():
            if not _on_cal(cr.get("ppm_error")):        # off instrument accuracy -> coincidence
                continue
            v = C.parse_formula(str(cr["compound_formula"]))
            n15 = v.get(label, 0)
            if v.get("O", 0) < 3 * n15:                 # each covalent nitrate carries >=3 O
                continue
            if C.fold_isotopes(v).get("O", 0) / max(v.get("C", 1), 1) > 1.3:  # O-monster
                continue
            if int(corrob.get((cr["compound_formula"], cr["mechanism_id"]), 0)) < 1:
                continue                                 # no matched isotopologue -> mass coincidence
            keep_idx.append((j, float(cr["ion_score"])))
        if not keep_idx:
            continue
        # mass-degeneracy: too many distinct plausible 15N readings -> not identifiable
        if len({sub.at[j, "compound_formula"] for j, _ in keep_idx}) > 2:
            continue
        j_best = max(keep_idx, key=lambda t: t[1])[0]
        r = sub.loc[j_best]
        s = float(r["ion_score"])
        # An existing fit here is a PARTIALLY-fluorinated mass-shift absorber
        # (target selection excluded genuine PFAS). It is chemically impossible
        # in a fluorine-free chamber, so a discipline-passing 15N reading wins
        # even if the coincidental F mass-fit scored marginally higher -- but
        # guard against throwing away a clearly better fit (>0.1 score gap).
        if not info["is_un"] and s < info["cur"] - 0.10:
            continue
        neutral = str(r["compound_formula"])
        adduct = _adduct_of(r, adducts)
        if ledger.at[i, "locked"]:
            continue
        note = (f"{label}-labelled product re-read: covalent {label} from the "
                f"labelled reagent (organonitrate); light-isotope grid was "
                f"{('unexplained' if info['is_un'] else 'a mass-shift F-fit')} at "
                f"{mz:.4f} m/z. ion score {s:.2f}")
        L.commit_assignment(
            ledger, ledger.at[i, "peak_id"], neutral_formula=neutral, adduct=adduct,
            ion_formula=str(r.get("ion_formula")), ion_score=s, compound_score=s,
            ppm_error=float(r["ppm_error"]) if pd.notna(r.get("ppm_error")) else None,
            pass_no=7, method="labeled:15N",
            confidence="Good (15N-labelled)", commentary=note, overwrite=True)
        if info["is_un"]:
            n_res += 1
        else:
            n_re += 1
    log(f"[labeled] {label}: filled {n_res} unexplained + re-read {n_re} F-fits "
        f"as heavy-isotope products")
    return {"rescued": n_res, "reread": n_re}


def _adduct_of(row, adducts: list[str]) -> str:
    """Best-guess the peaky adduct label from the scored mechanism id."""
    mech = str(row.get("mechanism_id", ""))
    from peaky.io.local_scoring import adduct_to_mech
    for a in adducts:
        try:
            if adduct_to_mech(a) == mech:
                return a
        except ValueError:
            # decomposition-alias channels ([M-H+I2]-, mixed +/- terms) have no
            # mechanism string by design -- they can never be the scored mech
            continue
    return adducts[0] if adducts else "[M-H]-"
