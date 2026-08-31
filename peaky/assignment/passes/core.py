"""passes.core — split from the former passes.py monolith."""

from __future__ import annotations

import re

import pandas as pd

from peaky.chem import chemistry as C
from peaky.assignment import ledger as L
from peaky.assignment import series_gka as G


from .config import PassConfig

__all__ = [
    "confidence_label",
    "_BACKBONE_ELEMENTS",
    "CAL_ARB_WEIGHT",
    "calibrate",
    "z_of",
    "_conf_suffix",
    "relabel_confidence",
    "arbitrate",
    "_f",
    "_DIFF_TO_ADDUCT",
    "_mech_to_adduct",
    "commit_winners",
    "_prefer_adduct_reading",
    "_iso_list",
    "_commentary",
]


def confidence_label(
    score: float,
    ppm: float | None,
    n_iso: int,
    tied: bool,
    cfg: PassConfig,
    suffix: str = "",
) -> str:
    # ppm proximity is judged against the CALIBRATED mass center when known, so a
    # uniform instrument offset (e.g. the -2.4 ppm of the uronium source) does not
    # read as "off-mass" and cap every commit at Low. Pre-calibration commits
    # (pass 1, cal_mu still None) use 0 and are re-graded by relabel_confidence
    # once calibrate() has fitted the center.
    center = cfg.cal_mu if cfg.cal_mu is not None else 0.0
    a = abs(ppm - center) if ppm is not None and pd.notna(ppm) else 99.0
    if score >= cfg.tau_high and a <= cfg.ppm * 1.5 and n_iso >= 1 and not tied:
        lab = "High"
    elif score >= cfg.tau_good and a <= cfg.ppm * 2:
        lab = "Good"
    elif score >= cfg.tau_low:
        lab = "Low"
    elif score >= cfg.tau_suspect:
        lab = "Suspect"
    else:
        lab = "Reject"
    return f"{lab} ({suffix})" if suffix and lab not in ("Reject",) else lab


_BACKBONE_ELEMENTS = {"C", "H", "O", "N"}


CAL_ARB_WEIGHT = 0.04


def calibrate(ledger: pd.DataFrame, cfg: PassConfig, *, log=print) -> tuple | None:
    """Fit the instrument's real mass accuracy (mu, sigma of the ppm error)
    over the committed High/Good CHO-CHON backbone and store it on cfg.

    Robust fit: median + scaled MAD, so a few bad commits can't widen the
    window they would then pass through. Returns (mu, sigma, n) or None when
    the backbone is too small to trust."""
    m0 = ledger[(ledger["role"] == L.ROLE_M0) & ledger["ppm_error"].notna()].copy()
    # Select the backbone by SCORE, not the confidence LABEL: the label embeds a
    # |ppm|<=2 sub-gate centered on 0, so at a large systematic mass offset (e.g.
    # the -2.4 ppm uronium source) it would exclude exactly the genuine backbone
    # and bias mu toward 0. Score is offset-independent; the median+MAD then finds
    # the true center robustly. (For a well-calibrated source the two selections
    # coincide.)
    score = pd.to_numeric(m0.get("ion_score"), errors="coerce").fillna(0.0)
    chon = (
        m0["neutral_formula"]
        .astype(str)
        .map(
            lambda f: (
                bool(C.parse_formula(f))
                and set(C.parse_formula(f)) <= _BACKBONE_ELEMENTS
            )
        )
    )
    ppm = m0.loc[(score >= cfg.tau_good) & chon, "ppm_error"].astype(float)
    if len(ppm) < cfg.cal_min_n:
        log(
            f"[calibrate] backbone too small (n={len(ppm)} < {cfg.cal_min_n}); "
            "mass gate stays off"
        )
        return None
    mu = float(ppm.median())
    sigma = max(float(1.4826 * (ppm - mu).abs().median()), cfg.cal_sigma_floor)
    cfg.cal_mu, cfg.cal_sigma = mu, sigma
    log(
        f"[calibrate] backbone n={len(ppm)}: ppm mu={mu:+.3f} sigma={sigma:.3f} "
        f"-> accept |z|<={cfg.cal_z_accept} ({mu - cfg.cal_z_accept * sigma:+.2f}"
        f"..{mu + cfg.cal_z_accept * sigma:+.2f} ppm), pattern-evidence up to "
        f"|z|<={cfg.cal_z_pattern}"
    )
    return mu, sigma, len(ppm)


def z_of(ppm, cfg: PassConfig) -> float | None:
    """Calibrated z-score of a ppm error; None when uncalibrated or ppm is NaN."""
    if cfg.cal_mu is None or cfg.cal_sigma is None:
        return None
    if ppm is None or pd.isna(ppm):
        return None
    return abs(float(ppm) - cfg.cal_mu) / cfg.cal_sigma


def _conf_suffix(conf) -> str:
    """The parenthetical method tag of a confidence label ('Good (series)' ->
    'series'); '' when there is none."""
    m = re.search(r"\(([^)]+)\)", str(conf or ""))
    return m.group(1) if m else ""


def relabel_confidence(ledger: pd.DataFrame, cfg: PassConfig, *, log=print) -> int:
    """Re-grade committed-M0 confidence labels against the calibrated mass center.

    Pass 1 commits BEFORE calibrate() runs, so at a large systematic mass offset
    its labels -- judged against 0 ppm -- read the whole high-score backbone as
    'Low', which then caps the report tier at Candidate and (because the tier
    engine recalibrates off High/Good rows) starves the mass-error gate. Re-judged
    against cal_mu the backbone recovers its true High/Good grade, while an
    off-trend mass monster that looked Good near 0 is correctly demoted. The
    method suffix (series / siloxane / ...) is preserved. No-op when uncalibrated.
    """
    if cfg.cal_mu is None:
        return 0
    # only the UNLOCKED pass-1 backbone: locked commits (pass-0 known species,
    # the siloxane ladder, any pass-1 High) carry a DELIBERATE grade -- e.g. a
    # known composite contaminant (silanediol n=4, ~35% Mascope score) is locked
    # by the known-species privilege and must not be re-graded to Reject here.
    m0 = ledger[(ledger["role"] == L.ROLE_M0) & ~ledger["locked"].astype(bool)]
    kids = (
        ledger[ledger["role"] == L.ROLE_ISO]
        .groupby("parent_peak_id")["peak_id"]
        .nunique()
    )
    n = 0
    for i, r in m0.iterrows():
        vals = [
            float(v)
            for v in (r.get("ion_score"), r.get("compound_score"))
            if v is not None and pd.notna(v)
        ]
        score = min(vals) if vals else 0.0
        n_iso = int(kids.get(r["peak_id"], 0))
        tied = bool(r.get("tied")) if pd.notna(r.get("tied")) else False
        new = confidence_label(
            score,
            r.get("ppm_error"),
            n_iso,
            tied,
            cfg,
            _conf_suffix(r.get("confidence")),
        )
        if new != str(r.get("confidence")):
            ledger.at[i, "confidence"] = new
            n += 1
    if n:
        log(
            f"[relabel] re-graded {n} confidence labels to the calibrated center "
            f"(mu={cfg.cal_mu:+.2f} ppm, sigma={cfg.cal_sigma:.2f})"
        )
    return n


def arbitrate(scored: pd.DataFrame, cfg: PassConfig) -> dict:
    """Decide a single best M0 owner per peak from the flat per-isotopologue
    scored table, with complexity-penalised effective scores.

    Returns:
      {
        'winners':  DataFrame[peak_id, neutral, adduct, ion_formula, ion_score,
                              compound_score, raw_score, eff_score, ppm_error,
                              n_iso, tied, alternatives(list)],
        'iso_children': DataFrame[peak_id, parent_peak_id, iso_label, iso_score],
      }
    """
    if scored is None or len(scored) == 0:
        return {"winners": pd.DataFrame(), "iso_children": pd.DataFrame()}

    # base ions that matched a real peak with a usable ion score
    base = scored[
        scored["is_base"]
        & scored["sample_peak_id"].notna()
        & scored["ion_score"].notna()
    ].copy()
    if len(base) == 0:
        return {"winners": pd.DataFrame(), "iso_children": pd.DataFrame()}

    base["raw_score"] = base[["ion_score", "compound_score"]].min(axis=1, skipna=True)

    # Mascope-confirmed isotopologues per (compound, ion): non-base rows for the
    # same ion that matched a real peak with score > 0.4
    iso = scored[
        (~scored["is_base"])
        & scored["sample_peak_id"].notna()
        & (pd.to_numeric(scored["iso_score"], errors="coerce").fillna(0) > 0.4)
    ]
    iso_count = (
        iso.groupby(["compound_formula", "ion_formula"])["sample_peak_id"]
        .nunique()
        .to_dict()
    )

    # diagnostic isotope labels confirmed per (compound, ion), e.g. {'13C','81Br','34S'}
    iso_labels: dict[tuple, set] = {}
    for (cf, ifl), g in iso.groupby(["compound_formula", "ion_formula"]):
        labs: set[str] = set()
        for lab in g["iso_label"].astype(str):
            labs.update(lab.split("+"))
        iso_labels[(cf, ifl)] = labs

    # Evidence-adjusted penalty on the NEUTRAL. The complexity prior (skepticism
    # about heteroatoms) applies, BUT a heteroatom whose diagnostic isotope is
    # Mascope-confirmed has its skepticism WAIVED -- direct evidence overrides
    # the prior. Heteroatoms with a usable isotope diagnostic and no confirmation
    # additionally take the gate penalty. Elements without a light-isotope
    # diagnostic (N, P, I, F -- all effectively mono-isotopic) keep the plain
    # complexity prior; those are corroborated by cross-channel/series evidence
    # at tier time, not by an isotope twin. Si DOES have a diagnostic twin
    # (29Si/30Si), so it is gated here -- a diag TUPLE credits either the M+1
    # (29Si) or M+2 (30Si) line (str.startswith accepts a tuple of prefixes).
    _DIAG = {
        "Cl": ("37Cl", cfg.het_iso_penalty_halogen),
        "Br": ("81Br", cfg.het_iso_penalty_halogen),
        "S": ("34S", cfg.het_iso_penalty_S),
        "Si": (("29Si", "30Si"), cfg.het_iso_penalty_Si),
    }

    def _evidence_penalty(row) -> float:
        cnt = C.parse_formula(row["compound_formula"])
        labs = iso_labels.get((row["compound_formula"], row["ion_formula"]), set())
        pen = 0.0
        for el, n in cnt.items():
            if n <= 0 or el not in C._COMPLEXITY_WEIGHT:
                continue
            prior = min(C._COMPLEXITY_WEIGHT[el] * n * 0.01, cfg.complexity_cap)
            if el in _DIAG:
                diag, gate = _DIAG[el]
                confirmed = any(s.startswith(diag) for s in labs)
                if el == cfg.reagent_element:
                    # ion isotope can't prove NEUTRAL ownership of the reagent
                    # halogen: keep the prior; gate only if not even ion-level
                    # confirmation exists.
                    pen += prior + (0.0 if confirmed else gate)
                elif confirmed:
                    continue  # confirmed -> waive skepticism
                else:
                    pen += prior + gate
            else:
                pen += prior  # no diagnostic -> plain prior
        return min(pen, 0.50)

    base["penalty"] = base.apply(_evidence_penalty, axis=1)
    # Channel prior: minor background channels (CO3-/O2-/electron attachment)
    # pay a ranking penalty so a near-tie goes to the primary channel. The
    # commit-side corroboration gate lives in commit_winners.
    base["adduct_label"] = base.apply(_mech_to_adduct, axis=1)

    # Calibration-aware off-trend penalty: once the mass calibration is known, a
    # candidate whose ppm sits far from the calibrated center is unlikely to be
    # the true formula even if its raw (OFFSET-BLIND) oracle score is high (the
    # server scores |ppm| vs theoretical, not vs the instrument's real center).
    # Without this, at a large systematic offset a mass-coincidence nearer 0 ppm
    # out-scores the true on-trend formula, WINS arbitration, then is z-rejected
    # at commit -- leaving the peak unexplained instead of taking the on-trend
    # formula (the mass-degenerate high-Si PDMS failure at the uronium -2.45 ppm
    # offset). Pre-calibration (pass 1) cal_mu is None -> no penalty.
    def _cal_offtrend(ppm):
        z = z_of(ppm, cfg)
        return 0.0 if z is None else max(0.0, z - cfg.cal_z_accept) * CAL_ARB_WEIGHT

    base["cal_penalty"] = base["ppm_error"].map(_cal_offtrend)
    base["eff_score"] = (
        base["raw_score"]
        - base["penalty"]
        - base["adduct_label"].isin(cfg.minor_channels) * cfg.minor_channel_penalty
        - base["cal_penalty"]
    )
    # reference-list selection prior: a neutral on an active reference peaklist gets
    # a small tie-break bonus so a known literature/contaminant formula wins a
    # near-tie over a mass-coincidence monster (a soft prior, not an override).
    if cfg.reflist_formulas and cfg.reflist_prior:
        base["eff_score"] += (
            base["compound_formula"].isin(cfg.reflist_formulas) * cfg.reflist_prior
        )

    winners = []
    for pid, grp in base.groupby("sample_peak_id"):
        grp = grp.sort_values("eff_score", ascending=False)
        top = grp.iloc[0]
        n_iso = int(iso_count.get((top["compound_formula"], top["ion_formula"]), 0))
        runner_eff = float(grp.iloc[1]["eff_score"]) if len(grp) > 1 else None
        tied = runner_eff is not None and (float(top["eff_score"]) - runner_eff) < 0.05
        # keep up to 6 alternatives: candidate DENSITY is the report's
        # confidence currency (tiers.py), and a 3-deep list saturates too early
        alts = []
        for _, r in grp.iloc[1:7].iterrows():
            alts.append(
                {
                    "formula": r["compound_formula"],
                    "adduct": r["adduct_label"],
                    "ion_score": _f(r["ion_score"]),
                    "raw_score": _f(r["raw_score"]),
                    "eff_score": _f(r["eff_score"]),
                    "ppm": _f(r["ppm_error"]),
                }
            )
        winners.append(
            {
                "peak_id": pid,
                "neutral": top["compound_formula"],
                "ion_formula": top["ion_formula"],
                "adduct": top["adduct_label"],
                "ion_score": _f(top["ion_score"]),
                "compound_score": _f(top["compound_score"]),
                "raw_score": _f(top["raw_score"]),
                "eff_score": _f(top["eff_score"]),
                "eff_margin": (
                    None if runner_eff is None else float(top["eff_score"]) - runner_eff
                ),
                "ppm_error": _f(top["ppm_error"]),
                "n_iso": n_iso,
                "tied": bool(tied),
                "alternatives": alts,
            }
        )
    win_df = pd.DataFrame(winners)

    # iso children attributed to the winning (compound, ion) pairs only
    children = []
    win_keys = {(w["neutral"], w["ion_formula"]): w["peak_id"] for w in winners}
    for _, r in iso.iterrows():
        key = (r["compound_formula"], r["ion_formula"])
        if key in win_keys:
            parent = win_keys[key]
            if r["sample_peak_id"] != parent:
                children.append(
                    {
                        "peak_id": r["sample_peak_id"],
                        "parent_peak_id": parent,
                        "iso_label": r["iso_label"],
                        "iso_score": _f(r["iso_score"]),
                    }
                )
    return {"winners": win_df, "iso_children": pd.DataFrame(children)}


def _f(v):
    return None if v is None or pd.isna(v) else float(v)


_DIFF_TO_ADDUCT = {
    (("H", -1),): "[M-H]-",
    (("Br", 1),): "[M+Br]-",
    (("Cl", 1),): "[M+Cl]-",
    (("I", 1),): "[M+I]-",
    (("N", 1), ("O", 3)): "[M+NO3]-",
    (("H", 1), ("O", 4), ("S", 1)): "[M+HSO4]-",
    (("C", 1), ("H", 1), ("O", 2)): "[M+CHO2]-",
    (("C", 2), ("H", 3), ("O", 2)): "[M+C2H3O2]-",
    (("C", 1), ("O", 3)): "[M+CO3]-",
    (("O", 2),): "[M+O2]-",
    (): "[M]-.",
    (("H", 1),): "[M+H]+",
    (("Na", 1),): "[M+Na]+",
    (("H", 4), ("N", 1)): "[M+NH4]+",
    (("K", 1),): "[M+K]+",
    # protonated-urea (uronium) adduct: ion = neutral + CH4N2O + H. Without this
    # entry the urea-channel assignments fall through to the "[M-H]-" default and
    # are mislabeled (positive-mode urea-CIMS: ~1/3 of the backbone is this
    # channel). diff sorts alphabetically C,H,N,O.
    (("C", 1), ("H", 5), ("N", 2), ("O", 1)): "[M+(CH4N2O)H]+",
    # MULTI-ATOM HALIDE CLUSTER CHANNELS. Same trap as the urea entry above: a
    # channel registered in a profile but missing HERE does not error -- it falls
    # through the `.get(diff, "[M-H]-")` default below and is silently written to
    # the ledger as a DEPROTONATION, which also inflates the [M-H]- census. Found
    # on the first iodide batch: 2 ions (CN2, C2H2O5) committed on [M+I2]- were
    # reported as [M-H]-. The Br cluster channels had the same latent gap.
    # Diff tuples sort alphabetically by element: Br < C < H < I < N < O.
    (("I", 2),): "[M+I2]-",
    (("I", 3),): "[M+I3]-",
    # deprotonated-acid + I2 cluster (A⁻·I₂). MIXED-sign diff: the ion is one H
    # SHORT of the neutral plus two iodines. Without this entry a committed
    # [M-H+I2]- row whose label is re-derived from (ion, compound) falls through
    # to "[M-H]-" -- the same silent-mislabel trap as the [M+I2]- gap above.
    (("H", -1), ("I", 2)): "[M-H+I2]-",
    (("Br", 2),): "[M+Br2]-",
    (("Br", 3),): "[M+Br3]-",
    (("Br", 2), ("H", 1)): "[M+HBr+Br]-",
    (("Br", 3), ("H", 1)): "[M+HBr+Br2]-",
    (("Br", 1), ("C", 1), ("H", 1), ("O", 3)): "[M+HBr+CO3]-",
}


def _mech_to_adduct(row) -> str:
    """Adduct label from the exact ion-vs-compound element difference."""
    ion = str(row.get("ion_formula") or "")
    # fold heavy isotopes ('^N' -> 'N') before the element diff so a 15N reagent
    # cluster reads the same (N+1, O+3) diff as 14N; the '^N' marker in the raw
    # ion string then upgrades the label to the heavy adduct just below.
    ci = C.fold_isotopes(C.parse_formula(ion))
    cc = C.fold_isotopes(C.parse_formula(str(row.get("compound_formula") or "")))
    diff = tuple(
        sorted(
            (el, ci.get(el, 0) - cc.get(el, 0))
            for el in set(ci) | set(cc)
            if ci.get(el, 0) != cc.get(el, 0)
        )
    )
    add = _DIFF_TO_ADDUCT.get(diff, "[M-H]-")
    # The element diff cannot see isotopic labelling: a ¹⁵N-nitrate reagent cluster
    # has the SAME (N+1, O+3) diff as a ¹⁴N one, but the server writes the heavy N
    # as '^N' in the ion formula. Without this, the ¹⁵N adduct is labelled [M+NO3]-
    # and the +61.99 (¹⁴N) shift puts ion_mz / jitter ~1 Da off.
    if add == "[M+NO3]-" and "^N" in ion:
        return "[M+^NO3]-"
    # Nor can it see polarity: the pipeline's Br-CIMS roots read every diff
    # negative by default, but a CATION row must flip the EasyIC channels -- an
    # EMPTY diff is charge transfer [M]+. (not electron attachment [M]-., a
    # -0.0011 two-electron mass error on every anchor) and an H-1 diff is
    # HYDRIDE abstraction [M-H]+ (not deprotonation). The charge sign comes off
    # the ion formula itself ('C16H10+'), falling back to the server's bare '+'
    # mechanism stamp.
    positive = ion.rstrip(".").endswith("+") or (
        str(row.get("ionization_mechanism") or "") == "+")
    if positive:
        if add == "[M]-.":
            return "[M]+."
        if diff == (("H", -1),):
            return "[M-H]+"
    return add


def commit_winners(
    ledger: pd.DataFrame,
    arb: dict,
    *,
    pass_no: int,
    method: str,
    context: str,
    cfg: PassConfig,
    lock: bool,
    min_raw_score: float,
    confidence_suffix: str = "",
    claim_unexplained_only: bool = False,
    only_peaks: set | None = None,
) -> dict:
    """Commit arbitration winners + their isotope children into the ledger.
    Skips peaks that are locked or already own a better assignment. Returns a
    summary dict."""
    win = arb.get("winners", pd.DataFrame())
    kids = arb.get("iso_children", pd.DataFrame())
    if len(win):
        # stronger claims commit first, so within-pass conflicts (two winners
        # wanting the same isotope child) resolve toward the better evidence
        win = win.sort_values("eff_score", ascending=False, na_position="last")
    committed, locked_ids = [], []
    rejected = {"nan_ppm": 0, "mass_gate": 0, "minor_channel": 0}
    # series/GKA proposals and known-neutral completions carry structural
    # evidence by construction (chain membership / an already-assigned neutral)
    series_like = ("series" in method) or ("gka" in method) or ("completion" in method)
    for _, w in win.iterrows():
        pid = w["peak_id"]
        if only_peaks is not None and pid not in only_peaks:
            continue  # evidence-scoped pass: only its own member peaks
        if w["raw_score"] is None or w["raw_score"] < min_raw_score:
            continue
        # A match with no mass error at all carries no mass evidence -- the 9
        # score-only bromides of v13 came through here. Never commit.
        if w["ppm_error"] is None or pd.isna(w["ppm_error"]):
            rejected["nan_ppm"] += 1
            continue
        # Calibrated mass gate: judge the ppm error against the instrument's
        # real accuracy, not a fixed window. Pattern evidence (a Mascope-
        # confirmed isotopologue or a series-derived proposal) buys the
        # 2..4 sigma band; nothing buys more.
        z = z_of(w["ppm_error"], cfg)
        if z is not None:
            pattern = (w["n_iso"] or 0) >= 1 or series_like
            if z > cfg.cal_z_pattern or (z > cfg.cal_z_accept and not pattern):
                rejected["mass_gate"] += 1
                continue
        # Minor-channel corroboration gate: a CO3-/O2-/M-. winner must be Good+
        # on raw score, or come from series evidence, or have its neutral
        # independently assigned via a primary channel elsewhere in the ledger.
        if (
            w["adduct"] in cfg.minor_channels
            and w["raw_score"] < cfg.tau_good
            and not series_like
        ):
            others = ledger[
                (ledger["role"] == L.ROLE_M0)
                & (ledger["neutral_formula"] == w["neutral"])
                & ~ledger["adduct"].isin(cfg.minor_channels)
            ]
            if len(others) == 0:
                rejected["minor_channel"] += 1
                continue
        # Reagent-halogen decomposition policy: covalent X(Br)[M-H]- and
        # Y.Br- / Y.HBr.Br- produce the IDENTICAL ion, so no spectral evidence
        # can distinguish them. The reagent halogen is read as part of the
        # adduct/cluster whenever the decomposition Y = X - HX is structurally
        # valid (user policy, 2026-06-11).
        w = _prefer_adduct_reading(w, cfg)
        try:
            if L.is_locked(ledger, pid):
                continue
            cur_role = L.role_of(ledger, pid)
        except L.LedgerError:
            continue  # peak not in ledger (sub-threshold) -> skip
        # later passes claim free peaks only -- a contaminant family must not
        # displace an earlier pass's isotope-confirmed assignment
        if claim_unexplained_only and cur_role != L.ROLE_UNEXPLAINED:
            continue
        # don't overwrite an existing higher-or-equal raw score
        if cur_role == L.ROLE_M0:
            existing = ledger.loc[ledger.peak_id == pid, "ion_score"].iloc[0]
            if pd.notna(existing) and existing >= w["raw_score"]:
                continue
        conf = confidence_label(
            w["raw_score"],
            w["ppm_error"],
            w["n_iso"],
            w["tied"],
            cfg,
            suffix=confidence_suffix,
        )
        if conf == "Reject":
            continue
        commentary = _commentary(w, pass_no, method)
        try:
            L.commit_assignment(
                ledger,
                pid,
                neutral_formula=w["neutral"],
                adduct=w["adduct"],
                ion_formula=w["ion_formula"],
                ion_score=w["ion_score"],
                compound_score=w["compound_score"],
                ppm_error=w["ppm_error"],
                eff_score=w.get("eff_score"),
                eff_margin=w.get("eff_margin"),
                tied=w.get("tied"),
                pass_no=pass_no,
                method=method,
                confidence=conf,
                commentary=commentary,
                alternatives=w["alternatives"],
                isotopologues=_iso_list(kids, pid),
                overwrite=(cur_role == L.ROLE_M0),
            )
            committed.append(pid)
            if lock and conf == "High":
                locked_ids.append(pid)
        except L.LedgerError:
            continue
    # attach isotope children (parents must now own M0)
    n_iso_attached = 0
    n_displaced = 0
    if len(kids):
        for _, k in kids.iterrows():
            try:
                if not (
                    k["parent_peak_id"] in set(committed)
                    or L.role_of(ledger, k["parent_peak_id"]) == L.ROLE_M0
                ):
                    continue
                # M0-vs-iso-child arbitration: when the predicted child peak
                # already owns its own M0, a Mascope-confirmed two-peak
                # explanation (parent + correct-ratio satellite) beats a
                # weaker independent formula on the child -- displace it.
                # (Cases: 428.99 'Br2 amine' was the 81Br twin of 426.99;
                # 297.02 'organosilicon CO3' was the 81Br twin of 295.02.)
                if L.role_of(ledger, k["peak_id"]) == L.ROLE_M0 and not L.is_locked(
                    ledger, k["peak_id"]
                ):
                    crow = ledger.loc[ledger.peak_id == k["peak_id"]].iloc[0]
                    prow = ledger.loc[ledger.peak_id == k["parent_peak_id"]].iloc[0]
                    child_conf = str(crow["confidence"])
                    # only High children are immune: a Mascope-confirmed
                    # two-peak explanation (parent + correct-ratio satellite)
                    # beats an equal-or-weaker single-peak formula (v16 audit:
                    # the 462.99/464.99 'Good vs Good' doublet never displaced)
                    child_immune = child_conf.startswith("High")
                    parent_stronger = pd.isna(crow["ion_score"]) or (
                        pd.notna(prow["ion_score"])
                        and prow["ion_score"] >= crow["ion_score"]
                    )
                    if not child_immune and parent_stronger:
                        L.displace_to_isotopologue(
                            ledger,
                            k["peak_id"],
                            k["parent_peak_id"],
                            iso_label=k["iso_label"],
                            iso_match_score=k["iso_score"],
                        )
                        n_displaced += 1
                        n_iso_attached += 1
                    continue
                L.attach_isotopologue(
                    ledger,
                    k["peak_id"],
                    k["parent_peak_id"],
                    iso_label=k["iso_label"],
                    iso_match_score=k["iso_score"],
                )
                n_iso_attached += 1
            except L.LedgerError:
                continue
    if locked_ids:
        L.lock_peaks(ledger, locked_ids)
    return {
        "committed": len(committed),
        "locked": len(locked_ids),
        "iso_attached": n_iso_attached,
        "iso_displaced": n_displaced,
        "rejected": rejected,
    }


_RI_CACHE: set | None = None


def _reactive_iodine_formulas() -> set:
    """Formulas the pass-0 registry declares REAL covalent-iodine species
    (HOI, INO2, INO3, ICl, CINO, ...). Their I2X- lines were ruled ambient
    analytes on TIME behaviour (the HOI2-/I2NO2- ruling, docs/REAGENTS.md), so
    the acid-cluster relabel must never re-read them -- an O-bearing subset
    (INO2 -> "HNO2", CINO -> "CHNO") would otherwise slip through the generic
    oxy-acid guard. The registry, not a hand copy, is the authority."""
    global _RI_CACHE
    if _RI_CACHE is None:
        from .directors import _known_species  # lazy: core loads before directors
        _RI_CACHE = set(_known_species("negative").get("reactive_iodine", {}))
    return _RI_CACHE


def _prefer_adduct_reading(w, cfg: PassConfig):
    """Relabel a winner whose NEUTRAL carries the reagent halogen so the Br
    sits in the adduct/cluster, not the neutral (user reagent rule). With
    Y = X - HBr (structurally valid):
      X(Br) [M-H]-       -> Y [M+Br]-          (deprotonation == Br adduct)
      X(Br) [M+Br]-      -> Y [M+HBr+Br]-       (covalent == HBr cluster)
      X(Br) [M+<chan>]-  -> Y [M+HBr+<chan>]-   (covalent+air-ion == HBr cluster
                                                 on the same background channel,
                                                 e.g. the 426.976 CO3 case)
    Iodide-specific (acids only; see the inline guard):
      X(I)  [M+I]-       -> M [M-H+I2]-         (M = X with I->H; covalent ==
                                                 conjugate-base . I2 cluster)
    The ion formula, score and ppm are unchanged -- only the decomposition is.
    A relabel onto a cluster adduct only fires when that adduct's exact mass is
    registered, so we never invent an unmodelled channel."""
    el = cfg.reagent_element
    if not el:
        return w
    x = w["neutral"]
    cx = C.parse_formula(x)
    if cx.get(el, 0) < 1:
        return w
    # Iodide: the preferred decomposition of a covalent MONO-iodine winner on
    # [M+I]- is the deprotonated-acid + I2 cluster, NOT the Br-style intact-
    # molecule HX cluster (which would report M-HI, e.g. formic -> "CO2").
    # With M = X with its I swapped for H (same DBE), the ion is identical:
    #   X(I) [M+I]-  ->  M [M-H+I2]-      (covalent == conjugate-base . I2)
    # and M is the acid the run already believes on [M+I]-/[M-H]-. Guards: a
    # real oxy-acid only -- O>=1 AND a C/N/S skeleton atom -- so HOI (-> "H2O"),
    # HIO3 (-> "H2O3") and ICl/IBr/ICN stay covalent: the pass-0 reactive-
    # iodine species must never be re-read (their I2X- lines were ruled ambient
    # analytes on time behaviour, docs/REAGENTS.md).
    if el == "I" and w["adduct"] == "[M+I]-" and cx.get("I", 0) == 1:
        m = dict(cx)
        m["I"] -= 1
        m["H"] = m.get("H", 0) + 1
        m = {k: v for k, v in m.items() if v > 0}
        if (
            m.get("O", 0) >= 1
            and any(m.get(e, 0) for e in ("C", "N", "S"))
            and C.dbe_ok(m)[0]
            and C.oxygen_ok(m)[0]
            and "[M-H+I2]-" in C.ADDUCT_SHIFTS
            and x not in _reactive_iodine_formulas()
        ):
            w = w.copy()
            w["adduct"] = "[M-H+I2]-"
            w["neutral"] = C.format_formula(m)
            w["_relabel_note"] = (
                f" Ion identical to covalent {x} [M+I]-; deprotonated-acid + I2 "
                f"cluster reading preferred (I assigned to the cluster, not the "
                f"neutral)."
            )
            return w
        # not an acid -> fall through to the generic HX-subtraction rules
    hx = "H" + el
    if hx not in G.REPEAT_UNITS:
        return w
    y = G.formula_add(x, hx, -1)
    if (
        not y
        or not C.dbe_ok(y)[0]
        or not C.oxygen_ok(y)[0]
        or C.parse_formula(y).get("C", 0) < 1
    ):
        return w
    adduct = w["adduct"]
    if adduct == "[M-H]-":
        new_adduct = f"[M+{el}]-"
    elif adduct == f"[M+{el}]-":
        new_adduct = f"[M+{hx}+{el}]-"
    elif adduct.startswith("[M+") and adduct.endswith("]-") and hx not in adduct:
        # background channel (CO3/O2/...): insert the HBr cluster unit
        new_adduct = f"[M+{hx}+{adduct[3:-2]}]-"
    else:
        return w
    if new_adduct not in C.ADDUCT_SHIFTS:
        return w  # unmodelled cluster channel -> keep the covalent reading
    w = w.copy()
    w["adduct"] = new_adduct
    w["neutral"] = y
    w["_relabel_note"] = (
        f" Ion identical to covalent {x} {adduct}; reagent-adduct reading "
        f"preferred ({el} assigned to the adduct/cluster, not the neutral)."
    )
    return w


def _iso_list(kids: pd.DataFrame, parent_pid) -> list[dict]:
    if kids is None or len(kids) == 0:
        return []
    sub = kids[kids["parent_peak_id"] == parent_pid]
    return [
        {"label": r["iso_label"], "score": r["iso_score"], "peak_id": r["peak_id"]}
        for _, r in sub.iterrows()
    ]


def _commentary(w, pass_no, method) -> str:
    base = (
        f"Pass {pass_no} ({method}): {w['neutral']} {w['adduct']}, "
        f"ion score {w['ion_score']:.2f}, ppm "
        f"{w['ppm_error']:.2f}"
        if w["ppm_error"] is not None
        else f"Pass {pass_no} ({method}): {w['neutral']} {w['adduct']}, "
        f"ion score {w['ion_score']:.2f}"
    )
    if w["n_iso"]:
        base += f"; {w['n_iso']} isotopologue(s) confirmed by Mascope"
    if w["alternatives"]:
        a = w["alternatives"][0]
        margin = (
            (w["eff_score"] - a["eff_score"])
            if (w["eff_score"] is not None and a.get("eff_score") is not None)
            else None
        )
        if margin is not None:
            base += f". Nearest competitor {a['formula']} trails by {margin:.2f}" + (
                " (TIE)" if w["tied"] else ""
            )
    note = w.get("_relabel_note") if hasattr(w, "get") else None
    if note:
        base += note
    return base
