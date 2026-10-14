"""passes.postprocess — split from the former passes.py monolith."""

from __future__ import annotations

import math
import re

import numpy as np
import pandas as pd

from peaky.chem import chemistry as C
from peaky.chem import isotopes as ISO
from peaky.assignment import ledger as L
from peaky.assignment import iso_response as IR
from peaky.assignment import satellites as SAT


from .config import PassConfig
from .core import cal_sigma_at, z_of

__all__ = [
    "_D13C",
    "_DBR",
    "_R13C",
    "_R81BR",
    "_D29SI",
    "_R29SI",
    "SI_M1_MIN_FRAC",
    "_peak_near",
    "_si_m1_consistent",
    "complete_isotope_envelopes",
    "_M1_RATIO",
    "detect_composites",
    "split_composites",
    "demote_carbon_inconsistent",
    "demote_massgate_monsters",
    "audit_isotopes",
    "audit_mass_gate",
    "REARB_WINNER_DBE_PER_C",
    "REARB_ALT_MIN_SCORE",
    "REARB_MAX_SCORE_DROP",
    "rearbitrate_offcal_degenerate",
]


_D13C = 1.0033548  # 13C - 12C


_DBR = 1.9979535  # 81Br - 79Br


_R13C = 0.0107  # 13C natural abundance per carbon


_R81BR = 0.9728  # 81Br/79Br abundance ratio


_D29SI = 0.999568  # 29Si - 28Si (the Si M+1)


_R29SI = 0.0468  # 29Si natural abundance per Si


SI_M1_MIN_FRAC = 0.6  # observed (M+1)/(M0) must be >= this * predicted Si M+1


def _pass_no(v, default: int = 1) -> int:
    """A ledger pass_no cell as an int. `int(pd.to_numeric(v, ...) or 1)` looks
    equivalent and is not: NaN is TRUTHY, so it survives the `or` and int(nan)
    raises ValueError, and pd.NA raises on the truth-test itself. A row with no
    pass_no reads as `default` (1 = a normal pass), never as the pass-0
    known-species lock."""
    n = pd.to_numeric(v, errors="coerce")
    return default if pd.isna(n) else int(n)


def _peak_near(mzs: "pd.Series", target: float, ppm: float = 5.0):
    """Index of the closest ledger peak within ppm of target, else None."""
    tol = target * ppm * 1e-6
    d = (mzs - target).abs()
    i = d.idxmin()
    return i if d.loc[i] <= tol else None


def _si_m1_consistent(
    ledger: pd.DataFrame, m0_mz: float, m0_h: float, n_si: int, n_c: int
) -> bool:
    """Is the 29Si M+1 intensity consistent with the CLAIMED Si count? A Si_k species'
    M+1 is dominated by 29Si (n_si*4.68%) plus 13C (n_c*1.07%); when the observed
    (M+1)/(M0) ratio is far below that, the Si count is over-claimed -- a high-O
    organic with only a 13C M+1 masquerading as a siloxane (the C10H18O11 vs
    C8H26O5Si4 degeneracy at m/z 393). True = OK to commit; False = skip."""
    if n_si <= 0 or m0_h <= 0:
        return True
    pred = n_si * _R29SI + n_c * _R13C
    if pred <= 0:
        return True
    j = _peak_near(ledger["mz"], m0_mz + _D29SI, ppm=15.0)
    obs = (
        float(ledger.at[j, "height"]) / m0_h
        if j is not None and pd.notna(ledger.at[j, "height"])
        else 0.0
    )
    return obs >= SI_M1_MIN_FRAC * pred


def _ee_cleared(ledger: pd.DataFrame, i) -> bool:
    """A peak the element-evidence stage cleared and locked (plausibility.cleared_by_element_evidence)."""
    from peaky.assignment import plausibility as PL
    return PL.cleared_by_element_evidence(ledger, i)


#: the fine-structure merge (Da) the strict own-line test reads (an Orbitrap at
#: R ~170k resolves lines ~0.5 mDa apart that isotope_pattern's 6-mDa merge joins)
FINE_MERGE_DA = 0.0005
#: the merge window of the envelope pattern (isotopes.isotope_pattern's default)
ENVELOPE_MERGE_DA = 0.006
_CARBON_ONLY = re.compile(r"^13C\d*$")


def _strict_line(dmass: float, label: str) -> bool:
    """A heteroatom M+1 / M+2 line: the strict own-line test applies (not a
    carbon-only 13C / 13C2 line, not the labelled reagent's 14N copy, not an
    M+4+ multi-isotope centroid)."""
    lab = str(label)
    return dmass < 2.5 and not _CARBON_ONLY.match(lab) and lab != "14N"


def _fine_lines(ion_formula: str) -> list:
    """(shift, rel) of the ion's fine-structure lines up to +2.5 Da (FINE_MERGE_DA merge)."""
    try:
        return [(d, r) for d, r, _lab in ISO.isotope_pattern(ion_formula, min_rel=1e-4, max_shift=2.5,
                                                             merge_da=FINE_MERGE_DA)]
    except Exception:  # noqa: BLE001
        return []


def _own_line_ok(obs_mz: float, obs_h: float, pmz: float, ph: float, dmass: float, rel: float,
                 fine: list, *, pos_ppm: float, floor) -> bool:
    """Is the peak at (`obs_mz`, `obs_h`) the parent's own line predicted at
    +`dmass` (`rel` of the parent, the envelope's merged line)? Its position
    within `pos_ppm` of the merged line or of a fine-structure line merged into
    it, and its height at satellites.ISO_RATIO_BAND x the merged prediction or
    the nearest fine line's -- the lower bound 0.3 (the sweep's own) where the
    line is predicted under the twin test's TWIN_REFUTE_X_FLOOR x the gate."""
    comps = [(d, r) for d, r in fine if abs(d - dmass) <= ENVELOPE_MERGE_DA] or [(dmass, rel)]
    tol = (pmz + dmass) * pos_ppm * 1e-6
    near = [(abs(obs_mz - (pmz + d)), d, r) for d, r in [(dmass, rel)] + comps]
    if min(t[0] for t in near) > tol:
        return False
    _dist, _d, r_near = min(near[1:] or near)
    lo, hi = SAT.ISO_RATIO_BAND
    censored = floor is not None and np.isfinite(floor) and floor > 0 \
        and ph * rel < SAT.TWIN_REFUTE_X_FLOOR * floor
    if censored:
        lo = 0.3
    for pred in (ph * rel, ph * r_near):
        if pred > 0 and lo <= obs_h / pred <= hi:
            return True
    return False


def own_line_of(ion_formula, pmz: float, ph: float, obs_mz: float, obs_h: float, dmass: float, cfg, *,
                wide: float, fallback_rel: float | None = None, cache: dict | None = None) -> bool:
    """The envelope sweep's own-line test (`_own_line_ok`) for another stage:
    is the line at (`obs_mz`, `obs_h`) the parent ion's own line near
    +`dmass`? Read against the parent's envelope line (isotope_pattern, the
    sweep's call) nearest `dmass` within ENVELOPE_MERGE_DA -- else the line
    (`dmass`, `fallback_rel`), or True without one -- in the window
    satellites.iso_position_ppm(cfg, the calibration sigma at the parent,
    `wide`), against the resolved height gate. `cache` keeps each ion's
    envelope and fine lines."""
    cache = {} if cache is None else cache
    key = str(ion_formula)
    if key not in cache:
        try:
            env = [(d, r) for d, r, _lab in ISO.isotope_pattern(key, min_rel=0.06, max_shift=12.0,
                                                                diag_min_rel=0.001)]
        except Exception:  # noqa: BLE001 -- an unparseable ion: the fallback line alone
            env = []
        cache[key] = (env, _fine_lines(key))
    env, fine = cache[key]
    near = [(abs(d - dmass), d, r) for d, r in env if abs(d - dmass) <= ENVELOPE_MERGE_DA]
    if near:
        _dist, d, rel = min(near)
    elif fallback_rel is not None:
        d, rel = dmass, float(fallback_rel)
    else:
        return True
    try:
        floor = float(cfg.height_cutoff)
    except Exception:  # noqa: BLE001 -- an unresolved gate: the 0.5x bound everywhere
        floor = None
    pos = SAT.iso_position_ppm(cfg, cal_sigma_at(cfg, pmz), wide)
    return _own_line_ok(float(obs_mz), float(obs_h), float(pmz), float(ph), d, rel, fine, pos_ppm=pos, floor=floor)


def scored_line_is_own(ledger: pd.DataFrame, parent_pid, child_pid, ion_formula, label, cfg, *,
                       cache: dict | None = None) -> bool:
    """A scorer-matched isotopologue line a COMMIT attaches (commit_winners'
    iso children, pass 7's member lines): on a calibrated Orbitrap-class run a
    heteroatom M+1 / M+2 line (`_strict_line`) is the parent's own only by the
    envelope sweep's test (`own_line_of`) -- the scorer's own window took a
    line 2.5-3.2 ppm off a certificate's 34S position as its 34S line, where
    the sweep refuses the same line. True everywhere else: off an Orbitrap,
    before the file is calibrated (passes 0 and 1: a scorer line there keeps
    today's reading), for a carbon-only 13C / labelled-copy 14N / M+4+ line,
    or without the two rows."""
    if not SAT.orbitrap_class(cfg) or SAT.calibrated_sigma(cfg) is None:
        return True
    pi = ledger.index[ledger["peak_id"] == parent_pid]
    ci = ledger.index[ledger["peak_id"] == child_pid]
    if not len(pi) or not len(ci):
        return True
    try:
        pmz, ph = float(ledger.at[pi[0], "mz"]), float(ledger.at[pi[0], "height"])
        cmz, ch = float(ledger.at[ci[0], "mz"]), float(ledger.at[ci[0], "height"])
    except (TypeError, ValueError):
        return True
    if not all(math.isfinite(v) for v in (pmz, ph, cmz, ch)) or ph <= 0 or ch <= 0:
        return True
    dm = cmz - pmz
    if not _strict_line(dm, str(label).split("(")[0]) or not isinstance(ion_formula, str) or not ion_formula:
        return True
    return own_line_of(ion_formula, pmz, ph, cmz, ch, dm, cfg, wide=5.0, cache=cache)


def complete_isotope_envelopes(
    ledger: pd.DataFrame,
    cfg: PassConfig,
    *,
    min_rel: float = 0.06,
    ppm: float = 12.0,
    log=print,
) -> dict:
    """Claim the FULL predicted isotope envelope (M+1/M+2/M+4...) of every
    committed M0, so multi-isotope species (Si-rich silanediols, multi-Br/Cl
    compounds) don't leak satellites into the residual.

    Two actions per predicted satellite line:
      * an UNEXPLAINED peak at the right mass + consistent intensity is attached
        as an iso_child (the envelope was incompletely claimed by the server);
      * a WEAK committed M0 (not High/Assigned, not locked) that is really a
        parent's satellite is DISPLACED into the iso_child role -- this is the
        393/395 silanediol bug, where the Si4+Br M+2 at 395 got mis-assigned a
        Cl-F-S formula because its M+4/M+2 ratio (~0.26) mimicked a Cl doublet.

    Processed parent-before-satellite (ascending m/z): a satellite is always
    heavier than its parent, so the true parent claims it first. The pattern is
    formula-specific (a CHO ion predicts only 13C, plus a 13C2 M+2 above ~28 C;
    halogen/Si M+2 lines need the actual heteroatom), and an intensity-
    consistency gate is the discriminator against coincidental neighbours -- a
    real satellite sits at the predicted height, an independent compound does
    not. The match tolerance is tight for M+1/M+2 (to separate 13C from 29Si,
    3.8 mDa apart) and looser for the multi-isotope M+4+ centroid.

    On an Orbitrap-class run a HETEROATOM M+1 / M+2 line (15N, 18O, 34S, 29Si,
    30Si, 37Cl, 81Br, ...; not a carbon-only 13C line, not the labelled reagent's
    14N copy) is the parent's own only at its exact position and height
    (`_own_line_ok`): within satellites.iso_position_ppm (max(2 ppm, 3 sigma) of
    the file's calibration) of the predicted line or of one of the fine-structure
    lines merged into it, at 0.5-2x the prediction (from 0.3x where the line is
    predicted under the twin test's 4x the gate: its per-file height is censored
    low there). A neighbour ion 2.6-4.8 ppm off a predicted 34S / 18O position or
    at 2-3x the ratio stays its own peak. Off an Orbitrap, and before the file is
    calibrated, the windows are as above."""
    out = {"attached": 0, "displaced": 0}
    mzs = ledger["mz"]
    strict = SAT.orbitrap_class(cfg)
    try:
        floor = float(cfg.height_cutoff) if strict else None
    except Exception:  # noqa: BLE001 -- an unresolved gate: the 0.5x bound everywhere
        floor = None
    order = ledger[ledger["role"] == L.ROLE_M0].sort_values("mz")["peak_id"].tolist()
    for pid in order:
        idx = ledger.index[ledger["peak_id"] == pid]
        if not len(idx):
            continue
        i = idx[0]
        if str(ledger.at[i, "role"]) != L.ROLE_M0:
            continue  # displaced by an earlier parent
        ionf = ledger.at[i, "ion_formula"]
        if ionf is pd.NA or pd.isna(ionf) or not str(ionf).strip():
            continue
        pmz = float(ledger.at[i, "mz"])
        ph = float(ledger.at[i, "height"])
        if not (ph > 0):
            continue
        # an ION-ONLY parent (the `ion_only` stage's [M]-. row beside its acid)
        # claims only UNEXPLAINED satellites: the bucket must not move anything
        # already assigned, and its 13C line sits 0.3 mDa from where a weak
        # cluster reading of another neutral can legitimately be (the 13C of
        # C7H6O4-. vs C2H7NO3 [M+NO3]-) -- a displacement there would turn an
        # existing Candidate into the satellite of a row whose neutral is open.
        io_parent = ("ion_only_of" in ledger.columns
                     and not pd.isna(ledger.at[i, "ion_only_of"]))
        try:
            # max_shift 12: keep the M+7/M+8 envelope of 4+ heavy-halogen ions
            # (a Br4 M+8 is ~0.9x M0) instead of leaking it into the residual.
            # diag_min_rel keeps the faint single-heteroatom diagnostic lines
            # (15N 0.36%, 18O, a single 34S/29Si/30Si) that min_rel=0.06 would
            # prune -- so they are CLAIMED (unexplained) or DISPLACE a weak phantom
            # M0 sitting on them (the 15N-satellite-of-a-CHON -> phantom-P leak),
            # instead of floating free for the opportunistic passes to grab. The
            # intensity-consistency gate below, not the floor, is the discriminator.
            pattern = ISO.isotope_pattern(str(ionf), min_rel=min_rel,
                                          max_shift=12.0, diag_min_rel=0.001)
        except Exception:
            continue
        fine = None              # the fine-structure lines, read once per parent when a strict line needs them
        pos_ppm = SAT.iso_position_ppm(cfg, cal_sigma_at(cfg, pmz), 5.0) if strict else 5.0
        for dmass, rel, label in pattern:
            # tolerance is shift-aware: M+1/M+2 must separate 13C (+1.0034) from
            # 29Si (+0.9996) -- 3.8 mDa apart -- so they use a tight window; the
            # multi-isotope M+4+ centroid is approximate and uses the loose one
            line_ppm = 5.0 if dmass < 2.5 else ppm
            j = _peak_near(mzs, pmz + dmass, ppm=line_ppm)
            if j is None:
                continue
            tpid = ledger.at[j, "peak_id"]
            if tpid == pid:
                continue
            th = float(ledger.at[j, "height"])
            if not (th > 0):
                continue
            if strict and _strict_line(dmass, label):
                if fine is None:
                    fine = _fine_lines(str(ionf))
                if not _own_line_ok(float(ledger.at[j, "mz"]), th, pmz, ph, dmass, rel, fine,
                                    pos_ppm=pos_ppm, floor=floor):
                    out["not_own"] = out.get("not_own", 0) + 1
                    continue
            ratio = th / (ph * rel)
            score = min(ratio, 1.0 / ratio) if ratio > 0 else 0.0
            role_j = str(ledger.at[j, "role"])
            if role_j == L.ROLE_UNEXPLAINED:
                if 0.3 <= ratio <= 3.5:
                    try:
                        # a peak the element-evidence stage cleared is locked against
                        # new readings, not against being a committed M0's line
                        L.attach_isotopologue(
                            ledger, tpid, pid, iso_label=label, iso_match_score=score,
                            overwrite=_ee_cleared(ledger, j)
                        )
                        out["attached"] += 1
                    except L.LedgerError:
                        pass
            elif role_j == L.ROLE_M0:
                if io_parent:
                    continue
                # the LABELLED-reagent impurity line (14N at -0.997 of a ^N adduct):
                # a fit sitting exactly at the predicted 2 % of a >=10x brighter
                # labelled parent is that parent's satellite whatever its own
                # score/lock -- the CHON [M+H]+ mass-fit on the 14N line of a
                # bright [M+^NH4]+ is the phantom this absorbs (pass-1 locks it as
                # High before the parent's envelope is known). Pass-0 known-species
                # locks are kept.
                if label == "14N" and 0.5 <= ratio <= 2.0 and ph >= 10 * th \
                        and _pass_no(ledger.at[j, "pass_no"]) != 0:
                    try:
                        if bool(ledger.at[j, "locked"]):
                            ledger.at[j, "locked"] = False
                        L.displace_to_isotopologue(
                            ledger, tpid, pid, iso_label=label, iso_match_score=score
                        )
                        out["displaced"] += 1
                    except L.LedgerError:
                        pass
                    continue
                if bool(ledger.at[j, "locked"]) or \
                        str(ledger.at[j, "confidence"]).startswith("High"):
                    # locked, or a High-confidence fit that earned it via its OWN
                    # isotope children -- a real compound, never a satellite.
                    continue
                sc_j = ledger.at[j, "ion_score"]
                weak_score = pd.isna(sc_j) or float(sc_j) < cfg.tau_high
                # A WEAK-scoring victim on a loose intensity match is a satellite
                # (the tier column is NA here -- tiers run later -- so gate on the
                # standalone score). A STRONG-scoring victim is normally a real
                # compound, EXCEPT on a FAINT M+1/M+2 diagnostic line (tight 5 ppm
                # mass, small predicted intensity: a 15N/34S/29Si/30Si/18O line, not
                # the bright 13C or halogen M+2) whose measured height is what the
                # parent's envelope predicts to within a tight factor. There, mass
                # identity + intensity consistency outweigh a high accurate-mass
                # score: a good score on a mono-isotopic-heteroatom ghost -- a P/S
                # formula sitting on a CHON's 15N line -- is exactly the phantom we
                # must absorb (the parent is ~100x brighter, so an independent
                # co-eluter would sit far ABOVE the faint predicted line, ratio>>2).
                faint_line = dmass < 2.5 and rel < 0.10
                if (weak_score and 0.45 <= ratio <= 2.2) or \
                        (faint_line and 0.5 <= ratio <= 2.0):
                    try:
                        L.displace_to_isotopologue(
                            ledger, tpid, pid, iso_label=label, iso_match_score=score
                        )
                        out["displaced"] += 1
                    except L.LedgerError:
                        pass
    if out["attached"] or out["displaced"]:
        log(
            f"[iso-envelope] attached {out['attached']} unclaimed satellites, "
            f"displaced {out['displaced']} mis-assigned satellites onto their "
            f"true parents"
        )
    return out


_M1_RATIO = {
    "C": 0.0107,
    "Si": 0.0508,
    "N": 0.003653,
    "S": 0.007896,
    "O": 0.000381,
    "H": 0.000115,
}


def detect_composites(
    ledger: pd.DataFrame,
    cfg: PassConfig,
    *,
    min_m1_rel: float = 0.06,
    excess_frac: float = 0.25,
    min_excess: float = 400.0,
    ppm: float = 8.0,
    log=print,
) -> dict:
    """Flag committed M0 peaks that are UNRESOLVED COMPOSITES -- their M0 (and
    M+2/M+4) intensity exceeds what their own M+1 satellite implies, because a
    coincident co-eluting compound shares the m/z.

    The discriminator is the even/odd isotope split: the M+1 region (13C, 29Si,
    15N -- all halogen-free) scales ONLY with the assigned compound, so it gives
    the assigned compound's true intensity S = M+1_obs / M+1_predicted. If the
    observed M0 markedly exceeds S, the excess is a co-component, and its
    halogen content is read off the EVEN-shift residual (M+2/M0, M+4/M+2 ~ Br /
    BrCl / Br2). This is the silanediol case the isotope-pattern 'mismatch'
    flagged: C8H26O5Si4 (Si4) at 393 sits on a ~45% BrCl compound -- formula and
    prediction are both correct; the peak is mixed. n=2 (clean) is not flagged.

    Flags only (does not demote): the assigned compound IS present; the note
    records the co-component fraction + halogen guess so the report is honest."""
    out = {"flagged": 0}
    mzs = ledger["mz"]
    if "composite_note" not in ledger.columns:
        ledger["composite_note"] = pd.Series(pd.NA, index=ledger.index, dtype="object")

    def _sum_window(lo, hi):
        m = (mzs >= lo) & (mzs <= hi)
        return float(ledger.loc[m, "height"].sum(skipna=True))

    for i, r in ledger[ledger["role"] == L.ROLE_M0].iterrows():
        ionf = r["ion_formula"]
        if ionf is pd.NA or pd.isna(ionf) or not str(ionf).strip():
            continue
        cnt = C.parse_formula(str(ionf))
        m1_rel = sum(_M1_RATIO.get(el, 0.0) * n for el, n in cnt.items())
        if m1_rel < min_m1_rel:
            continue  # too few C/Si to diagnose a composite
        m0 = float(r["mz"])
        h0 = float(r["height"])
        if not (h0 > 0):
            continue
        # observed M+1 region: 13C(+1.0034) + 29Si(+0.9996) + 15N(+0.997)
        h1 = _sum_window(m0 + 0.9940, m0 + 1.0070)
        if h1 <= 0:
            continue
        s_assigned = h1 / m1_rel  # implied true intensity of the M0 owner
        excess = h0 - s_assigned
        if excess < min_excess or excess / h0 < excess_frac:
            continue
        # characterise the co-component via the even-shift residual. SUM (never
        # overwrite) lines that round to the same integer shift, else the big
        # 81Br M+2 (rel ~1.14) is clobbered by the tiny 13C2 line at +2.007.
        pat: dict[int, float] = {}
        try:
            for d, rel, _ in ISO.isotope_pattern(str(ionf), min_rel=0.01):
                pat[round(d)] = pat.get(round(d), 0.0) + rel
        except Exception:
            pat = {}
        h2 = _sum_window(m0 + 1.992, m0 + 2.004)
        h4 = _sum_window(m0 + 3.990, m0 + 4.002)
        x2 = h2 - s_assigned * pat.get(2, 0.0)  # co-component M+2
        x4 = h4 - s_assigned * pat.get(4, 0.0)  # co-component M+4
        hal = "unknown"
        if x2 > min_excess:
            r2 = x2 / excess
            r4 = x4 / x2 if x2 > 0 else 0.0
            if r2 >= 1.6:
                hal = "Br2"
            elif r2 >= 1.15 and r4 >= 0.22:
                hal = "BrCl"
            elif r2 >= 0.7:
                hal = "Br"
            elif 0.22 <= r2 <= 0.45:
                hal = "Cl"
        ledger.at[i, "composite_note"] = (
            f"composite: ~{100 * excess / h0:.0f}% co-eluting {hal} component "
            f"(~{excess:.0f} cps); M+1 implies {str(r['neutral_formula'])} "
            f"= ~{s_assigned:.0f} of {h0:.0f} cps"
        )
        # structured fields for the de-blending step (split_composites)
        ledger.at[i, "assigned_fraction"] = max(0.0, min(1.0, s_assigned / h0))
        ledger.at[i, "co_height"] = excess
        ledger.at[i, "co_halogen"] = hal
        out["flagged"] += 1
    if out["flagged"]:
        log(
            f"[composite] flagged {out['flagged']} unresolved composite peaks "
            f"(M0 inflated beyond the M+1-implied owner intensity)"
        )
    return out


def split_composites(
    ledger: pd.DataFrame, cfg: PassConfig, *, log=print
) -> pd.DataFrame:
    """De-blend the peaks `detect_composites` flagged: the owner keeps its
    `assigned_fraction` of the measured height, and a SYNTHETIC sub-peak
    ('<host>.2', same m/z) is created carrying the co-eluting compound's share
    (co_height) plus its halogen guess. The sub-peak is a characterised residual
    (role unexplained, synthetic=True, host_peak_id->host) -- a target for a
    later constrained match that NAMES the co-component. Signal is conserved:
    effective = height*assigned_fraction, so host + sub-peak sum to the original
    height (the host's measured `height` is never altered).

    Returns the (possibly grown) ledger -- new synthetic rows are appended, so
    the caller must rebind: `led = split_composites(led, cfg)`."""
    if "co_height" not in ledger.columns:
        return ledger
    syn = ledger["synthetic"].fillna(False).astype(bool)
    hosts = ledger[(ledger["role"] == L.ROLE_M0) & ledger["co_height"].notna() & (~syn)]
    existing = set(ledger["peak_id"])
    new_rows = []
    for _, r in hosts.iterrows():
        co_h = float(r["co_height"])
        sub_id = f"{r['peak_id']}.2"
        if co_h < 1.0 or sub_id in existing:
            continue
        hal = str(r["co_halogen"])
        row = {c: pd.NA for c in ledger.columns}
        row.update(
            {
                "peak_id": sub_id,
                "mz": float(r["mz"]),
                "height": co_h,
                "area": float("nan"),
                "role": L.ROLE_UNEXPLAINED,
                "synthetic": True,
                "host_peak_id": r["peak_id"],
                "assigned_fraction": 1.0,
                "locked": False,
                "co_height": float("nan"),
                "commentary": (
                    f"co-eluting {hal} component split from peak "
                    f"{r['mz']:.4f} (~{co_h:.0f} cps); host owner "
                    f"{str(r['neutral_formula'])} keeps "
                    f"{100 * float(r['assigned_fraction']):.0f}%"
                ),
            }
        )
        new_rows.append(row)
        existing.add(sub_id)
    if not new_rows:
        return ledger
    add = pd.DataFrame(new_rows)[list(ledger.columns)]
    log(
        f"[composite] split {len(new_rows)} composite peaks into fractional "
        f"sub-peaks (owner keeps assigned_fraction; co-component -> '<id>.2')"
    )
    return pd.concat([ledger, add], ignore_index=True)


def _num(v):
    """a finite float, else None"""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f == f and f not in (float("inf"), float("-inf")) else None


def demote_carbon_inconsistent(
    ledger: pd.DataFrame, cfg: PassConfig, *, log=print
) -> int:
    """Clear committed M0s whose carbon count is contradicted by their 13C
    satellite -- the 'O15 monster' class. Run BEFORE pass 4 (not just in the
    end-of-run audit) so the freed bright peaks are re-offered the correct
    carbon-clamped interpretation. Without this, pass 1 grabs a lattice peak
    with a low-carbon CHON mass-fit (e.g. C11H10N2O15 on the 4.7k-cps 409.0015,
    whose 13C satellite measures ~C16), pass 4 skips it because it is no longer
    unexplained, and the audit only clears it after every pass has run -- too
    late to re-assign as the di-bromide SOA cluster (C15H22O3 [M+HBr+Br]-)."""
    mzs = ledger["mz"]
    n = 0
    for _, r in ledger[
        (ledger["role"] == L.ROLE_M0) & ~ledger["locked"].astype(bool)
    ].iterrows():
        cnt = C.parse_formula(str(r["ion_formula"]))
        n_c = cnt.get("C", 0)
        if n_c < 8:
            continue
        # the carbon clamp reads the M+1 region as 13C only; a Si-bearing formula
        # has a 29Si M+1 (4.7%/Si) far larger than 13C, so the measured "13C
        # ratio" over-estimates carbon and would wrongly clear a real siloxane.
        # Skip them -- their carbon is corroborated by the Si-isotope envelope.
        if cnt.get("Si", 0) > 0:
            continue
        # measure carbon from a committed 13C child, else an unclaimed satellite
        k = ledger[
            (ledger["role"] == L.ROLE_ISO)
            & (ledger["parent_peak_id"] == r["peak_id"])
            & (ledger["iso_label"].astype(str) == "13C")
        ]
        if len(k):
            h_sat = float(k.iloc[0]["height"])
            a_sat = _num(k.iloc[0].get("area"))
        else:
            j = _peak_near(mzs, float(r["mz"]) + _D13C)
            if j is None or ledger.at[j, "role"] != L.ROLE_UNEXPLAINED:
                continue
            h_sat = float(ledger.at[j, "height"])
            a_sat = _num(ledger.at[j, "area"]) if "area" in ledger else None
        h0 = float(r["height"])
        if not (h0 > 0 and h_sat > 0):
            continue
        # only clamp on a RELIABLY-measured 13C satellite (>= the peak-detection
        # floor). Below it the ratio is noise: a genuine low-intensity M0 whose
        # weak/peak-picker-lost 13C reads as too-few-carbons would be falsely
        # cleared (the real [M+15NO3]- M0s at ~2k cps whose ~150 cps 13C sits
        # near the floor). The over-claim O-monster always has a BRIGHT 13C.
        if h_sat < cfg.height_cutoff:
            continue
        # raw and, where the run reads its lines by area (iso_response.py), by area against the file's
        # response too: both must contradict, in the same direction
        bad, c_est = IR.contradicts(getattr(cfg, "iso_response_fit", None), n_c, h0=h0, h_sat=h_sat,
                                    a0=_num(r.get("area")), a_sat=a_sat, snr0=_num(r.get("signal_to_noise")),
                                    area_mode=bool(getattr(cfg, "iso_area", False)))
        if bad:
            try:
                L.clear_assignment(
                    ledger,
                    r["peak_id"],
                    reason=f"carbon-clamp (pre-pass-4): 13C ratio measures ~C"
                    f"{c_est:.0f}, formula claims C{n_c}",
                )
                n += 1
            except L.LedgerError:
                continue
    if n:
        log(
            f"[pre-pass4] carbon-clamp demoted {n} C-inconsistent M0 monsters "
            f"-> re-offered to the residual passes"
        )
    return n


def demote_massgate_monsters(
    ledger: pd.DataFrame, cfg: PassConfig, *, log=print
) -> int:
    """Clear pre-calibration M0s whose calibrated mass error is egregious
    (z > cal_z_pattern) BEFORE pass 4, the mass-gate twin of
    demote_carbon_inconsistent. Pass 1 grabs bright halogen-doublet peaks with
    high-O CHON mass-fits that the END mass-gate audit clears (e.g. C11H10N2O16
    on the 424.99 di-bromide peak, z=7.3) -- too late for pass 4 to re-claim.
    Only the clear monsters (z > pattern band) are cleared here; the 2..4-sigma
    tier is left for the end-of-run audit to keep this conservative."""
    if cfg.cal_mu is None:
        return 0
    n = 0
    for _, r in ledger[
        (ledger["role"] == L.ROLE_M0) & ~ledger["locked"].astype(bool)
    ].iterrows():
        z = z_of(r["ppm_error"], cfg, r.get("mz"))
        if z is not None and z > cfg.cal_z_pattern:
            try:
                L.clear_assignment(
                    ledger,
                    r["peak_id"],
                    reason=f"mass-gate (pre-pass-4): z={z:.1f} > {cfg.cal_z_pattern}",
                )
                n += 1
            except L.LedgerError:
                continue
    if n:
        log(
            f"[pre-pass4] mass-gate demoted {n} z>{cfg.cal_z_pattern} monsters "
            f"-> re-offered to the residual passes"
        )
    return n


def audit_isotopes(ledger: pd.DataFrame, cfg: PassConfig, *, log=print) -> dict:
    """Post-run isotope-physics audit. Validates committed M0s against what
    the isotope pattern REQUIRES, independent of match scores (v16 audit):

    1. Br-doublet repair: two M0s 1.99795 apart at ~1:1 height are one
       single-Br compound, not two formulas. If the lighter ion carries Br,
       the heavier peak becomes its 81Br child; if neither formula carries
       Br, both are wrong (the doublet proves Br) and both are cleared.
    2. 13C sweeper: attach the obvious unclaimed 13C satellite (right place,
       right magnitude) as evidence instead of leaving it unexplained.
    3. 13C carbon clamp: a committed 13C child measures the carbon count;
       a formula whose C is far outside it is wrong (C19 claimed, ~C11 seen).
    4. 13C completeness: a formula predicting a comfortably-visible 13C
       satellite that has NO peak at +1.0034 is wrong.
    """
    out = {
        "doublet_child": 0,
        "doublet_cleared": 0,
        "c13_attached": 0,
        "c13_clamp": 0,
        "c13_missing": 0,
    }
    mzs = ledger["mz"]

    # --- 1. Br-doublet repair over committed M0s ---
    m0 = ledger[
        (ledger["role"] == L.ROLE_M0) & ~ledger["locked"].astype(bool)
    ].sort_values("mz")
    rows = list(m0[["peak_id", "mz", "height", "ion_formula"]].itertuples(index=False))
    for a in range(len(rows)):
        lt = rows[a]
        for b in range(a + 1, len(rows)):
            hv = rows[b]
            d = hv.mz - lt.mz
            if d > _DBR + 0.005:
                break
            if abs(d - _DBR) > 0.004:
                continue
            hr = hv.height / lt.height
            if not (0.6 <= hr <= 1.45):
                continue
            try:
                if (
                    L.role_of(ledger, lt.peak_id) != L.ROLE_M0
                    or L.role_of(ledger, hv.peak_id) != L.ROLE_M0
                ):
                    continue
                n_br = C.parse_formula(str(lt.ion_formula)).get("Br", 0)
                if n_br >= 1:
                    # the lighter formula genuinely carries Br -> a ~1:1 twin
                    # 1.998 above IS its ⁸¹Br isotopologue (valid regardless of
                    # the reagent system).
                    L.clear_assignment(
                        ledger,
                        hv.peak_id,
                        reason=f"isotope audit: 81Br twin of "
                        f"{lt.mz:.4f} (ratio {hr:.2f})",
                    )
                    L.attach_isotopologue(
                        ledger, hv.peak_id, lt.peak_id, iso_label="81Br"
                    )
                    out["doublet_child"] += 1
                elif cfg.reagent_element == "Br":
                    # clear-both ONLY in Br-CIMS. There a ~1:1 1.998 doublet is
                    # strong evidence of an (unassigned) bromine, so two non-Br
                    # formulas are both wrong. With any OTHER reagent (e.g.
                    # ¹⁵N-nitrate) bromine is not in play: unrelated CHON
                    # compounds routinely sit ~1.998 apart at ~1:1, and the
                    # spacing is NOT halogen evidence -- clearing both destroys
                    # real M0s (54 genuine [M+¹⁵NO₃]⁻ M0s on the ¹⁵NO₃⁻ batch).
                    L.clear_assignment(
                        ledger,
                        lt.peak_id,
                        reason=f"isotope audit: Br doublet with {hv.mz:.4f} "
                        f"(ratio {hr:.2f}) but no Br in formula",
                    )
                    L.clear_assignment(
                        ledger,
                        hv.peak_id,
                        reason=f"isotope audit: Br doublet with {lt.mz:.4f} "
                        f"(ratio {hr:.2f}) but no Br in formula",
                    )
                    out["doublet_cleared"] += 2
            except L.LedgerError:
                continue

    # --- 2-4. 13C physics on every surviving M0 ---
    # the satellite detection floor: the height gate for a spectrum, or the
    # caller's per-spectrum floor for a derived table (see PassConfig.audit_floor_cps)
    floor = cfg.audit_floor_cps if cfg.audit_floor_cps is not None else cfg.height_cutoff
    kids = ledger[ledger["role"] == L.ROLE_ISO]
    resp = getattr(cfg, "iso_response_fit", None)
    for _, r in ledger[ledger["role"] == L.ROLE_M0].iterrows():
        if bool(r["locked"]):
            continue
        n_c = C.parse_formula(str(r["ion_formula"])).get("C", 0)
        if n_c < 1:
            continue
        # the 13C height the summed list is expected to keep: the natural share x the file's median response
        # at the line's expected S/N (a line near the scans' floor is lost in the scans it does not clear).
        # The response is an AREA ratio and heights read lower still: the expectation is an upper bound
        natural = float(r["height"]) * _R13C * n_c
        snr0 = _num(r.get("signal_to_noise"))
        x13 = math.log10(snr0 * _R13C * n_c) if snr0 and snr0 > 0 else None
        expected = natural * (IR.scale(resp, x13) if resp else 1.0)
        k = kids[
            (kids["parent_peak_id"] == r["peak_id"])
            & (kids["iso_label"].astype(str) == "13C")
        ]
        if not len(k):
            j = _peak_near(mzs, r["mz"] + _D13C, ppm=cfg.audit_sat_ppm)
            if (
                j is not None
                and ledger.at[j, "role"] == L.ROLE_UNEXPLAINED
                and expected > 0
                and ledger.at[j, "height"] >= 0.3 * expected
                and ledger.at[j, "height"] <= 2.5 * natural
            ):
                try:
                    L.attach_isotopologue(
                        ledger, ledger.at[j, "peak_id"], r["peak_id"], iso_label="13C"
                    )
                    out["c13_attached"] += 1
                    k = ledger.loc[[j]]
                except L.LedgerError:
                    pass
        if len(k):
            h_sat = float(k.iloc[0]["height"])
            bad, c_est = IR.contradicts(resp, n_c, h0=float(r["height"]),
                                        h_sat=h_sat, a0=_num(r.get("area")), a_sat=_num(k.iloc[0].get("area")),
                                        snr0=snr0, area_mode=bool(getattr(cfg, "iso_area", False)))
            # clamp ONLY on a reliably-measured 13C satellite (>= the detection
            # floor). A sub-floor 13C ratio is noise and under-reads carbon,
            # which would falsely clear genuine low-intensity M0s (the ~2k cps
            # [M+15NO3]- compounds whose ~150 cps 13C sits near the floor). The
            # over-claim O-monster case always carries a BRIGHT 13C, so it fires.
            if (
                n_c >= 8
                and h_sat >= floor
                and bad
            ):
                try:
                    L.clear_assignment(
                        ledger,
                        r["peak_id"],
                        reason=f"isotope audit: 13C ratio measures ~C"
                        f"{c_est:.0f}, formula claims C{n_c}",
                    )
                    out["c13_clamp"] += 1
                except L.LedgerError:
                    pass
        elif (
            expected >= 1.5 * floor
            and _peak_near(mzs, r["mz"] + _D13C, ppm=cfg.audit_sat_ppm) is None
        ):
            # twin-satellite fallback: when the peak has a halogen isotope
            # twin, the twin's OWN 13C satellite (13C+81Br / 13C+37Cl) is
            # equally valid carbon evidence. v20 falsely cleared C3H6O3.Br-
            # (10.3k cps): its plain 13C is peak-picker-lost, but the twin's
            # satellite at +1.998+1.0034 exists and is carbon-consistent.
            twins = ledger[
                (ledger["parent_peak_id"] == r["peak_id"])
                & ledger["iso_label"].astype(str).str.contains("Br|Cl", regex=True)
            ]
            if any(
                _peak_near(mzs, float(t["mz"]) + _D13C, ppm=cfg.audit_sat_ppm) is not None
                for _, t in twins.iterrows()
            ):
                continue
            # cross-channel fallback: the SAME neutral independently assigned
            # High/Good on another peak (other adduct) is positive evidence
            # that outweighs one absent satellite -- an absent 13C can be a
            # peak-picker loss, an agreeing second channel cannot. (v21
            # cleared five sub-ppm [M+Br]- partners of Good [M-H]-
            # assignments, e.g. C10H16O6 at 311.013 / 2.4k cps.)
            others = ledger[
                (ledger["role"] == L.ROLE_M0)
                & (ledger["peak_id"] != r["peak_id"])
                & (ledger["neutral_formula"] == r["neutral_formula"])
                & ledger["confidence"].astype(str).str.startswith(("High", "Good"))
            ]
            if len(others):
                continue
            try:
                L.clear_assignment(
                    ledger,
                    r["peak_id"],
                    reason=f"isotope audit: predicted 13C satellite "
                    f"({expected:.0f} cps) absent from spectrum",
                )
                out["c13_missing"] += 1
            except L.LedgerError:
                continue

    n = sum(out.values())
    if n:
        log(
            f"[audit] isotope physics: {out['doublet_child']} doublet twins "
            f"re-attached, {out['doublet_cleared']} no-Br doublet formulas "
            f"cleared, {out['c13_attached']} 13C satellites attached, "
            f"{out['c13_clamp']} carbon-clamp clears, "
            f"{out['c13_missing']} missing-13C clears"
        )
    return out


def audit_mass_gate(ledger: pd.DataFrame, cfg: PassConfig, *, log=print) -> dict:
    """Post-run sweep: apply the calibrated mass gate to commits that predate
    calibration (pass 1 runs before the backbone exists). Clears, never
    rewrites: a >4-sigma mass error means the formula is wrong no matter what
    the isotope pattern says; a 2..4-sigma Low/Suspect with no pattern
    evidence is just the closest of many candidates."""
    out = {"cleared_z": 0, "cleared_z_noiso": 0, "cleared_nan": 0}
    if cfg.cal_mu is None:
        return out
    m0 = ledger[(ledger["role"] == L.ROLE_M0) & ~ledger["locked"].astype(bool)]
    parents_with_kids = set(
        ledger.loc[ledger["role"] == L.ROLE_ISO, "parent_peak_id"].dropna()
    )
    for _, r in m0.iterrows():
        weak = not str(r["confidence"]).startswith(("High", "Good"))
        has_kids = r["peak_id"] in parents_with_kids
        z = z_of(r["ppm_error"], cfg, r.get("mz"))
        try:
            if z is None:
                if pd.isna(r["ppm_error"]) and weak and not has_kids:
                    L.clear_assignment(
                        ledger, r["peak_id"], reason="mass-gate: no ppm error"
                    )
                    out["cleared_nan"] += 1
            elif z > cfg.cal_z_pattern:
                L.clear_assignment(
                    ledger,
                    r["peak_id"],
                    reason=f"mass-gate: z={z:.1f} > {cfg.cal_z_pattern}",
                )
                out["cleared_z"] += 1
            elif z > cfg.cal_z_accept and weak and not has_kids:
                L.clear_assignment(
                    ledger,
                    r["peak_id"],
                    reason=f"mass-gate: z={z:.1f} without pattern evidence",
                )
                out["cleared_z_noiso"] += 1
        except L.LedgerError:
            continue
    n = sum(out.values())
    if n:
        log(
            f"[audit] mass gate cleared {n} assignments "
            f"(z>{cfg.cal_z_pattern}: {out['cleared_z']}, "
            f"{cfg.cal_z_accept}<z<={cfg.cal_z_pattern} no-evidence: "
            f"{out['cleared_z_noiso']}, no-ppm: {out['cleared_nan']})"
        )
    return out


# ---------------------------------------------------------------------------
# off-calibration degenerate-winner re-arbitration
# ---------------------------------------------------------------------------
# The winner-selection bug: pass 1 commits the highest-eff_score candidate per
# peak BEFORE the mass calibration is fitted (calibrate() runs on the pass-1
# backbone). The local in-process scorer (PEAKY_LOCAL_SCORING) can rank a
# mass-degenerate competitor ABOVE the corroborated one, and at a sub-ppm
# coincidence an off-calibration high-DBE / heteroatom-rich "monster" can out-
# score the real on-trend molecule. The tier engine THEN demotes it to Candidate
# (off-cal, mass-error-distribution test) -- but only after it has displaced the
# better, on-cal alternative from the M0 slot entirely.
#
# This stage applies the SAME calibration-sigma + corroboration gate the tier
# engine computes (tiers._calibrate / Z_TAIL_DEMOTE), but AT WINNER-SELECTION:
# a committed winner that is (a) off-calibration beyond the tier's tail, (b)
# uncorroborated (no isotopologue / cross-channel / series-anchor support -- the
# exact evidence that would break a mass degeneracy), and (c) sitting in the
# aromatic-monster corner (high DBE/C) is very likely a mass-fit artifact. If a
# stored alternative is on-calibration, chemically plausible (plausibility
# oracle), and STRICTLY less unsaturated (lower DBE), it is preferred. The
# lower-DBE + plausible guards are essential: a blunt "swap to the best on-cal
# alternative" reverses many correct calls (it would replace a reasonable off-cal
# CHO formula with an on-cal-but-implausible high-DBE / O-monster competitor).
REARB_WINNER_DBE_PER_C = 0.70   # winner must sit in the aromatic-monster corner
REARB_ALT_MIN_SCORE = 0.60      # don't displace toward a noise-grade alternative
REARB_MAX_SCORE_DROP = 0.35     # ... nor one far weaker than the (mass-fit) winner


def _ion_formula_str(neutral: str, adduct: str) -> str:
    """Ion element composition for a (neutral, adduct) reading, as a formula
    string. Handles parenthesised adduct groups (e.g. the uronium
    '[M+(CH4N2O)H]+') by stripping the brackets before tokenising."""
    cnt = dict(C.parse_formula(str(neutral)))
    inner = str(adduct).split("]")[0]
    inner = inner[2:] if inner.startswith("[M") else inner
    inner = inner.replace("(", "").replace(")", "")
    for sign, tok in re.findall(r"([+-])\^?([A-Za-z0-9]+)", inner):
        for el, k in C.parse_formula(tok).items():
            cnt[el] = cnt.get(el, 0) + (k if sign == "+" else -k)
    return C.format_formula({k: v for k, v in cnt.items() if v})


def rearbitrate_offcal_degenerate(
    ledger: pd.DataFrame, cfg: PassConfig, *, log=print, profile=None
) -> dict:
    """Displace off-calibration, uncorroborated, high-DBE 'monster' M0 winners
    with an on-calibration, plausible, less-unsaturated stored alternative.

    Reuses tiers._calibrate (the isotopologue-backed CHO/CHON core) so the
    off-cal gate is IDENTICAL to the one the report tier engine applies -- the
    point being to apply it at winner-selection, not only at tiering. Returns
    {'swapped': n}. Mutates the ledger in place (overwrite commits). No-op when
    uncalibrated. `profile`: the run's ContextProfile -- an alternative is
    judged plausible on the same readings the run's plausibility demotes use
    (`plausibility.implausible(profile=)`: the NOx skeleton on a nitrate run);
    None = the raw gates."""
    from peaky.assignment import tiers as T
    from peaky.assignment import plausibility as PL
    from .core import confidence_label

    out = {"swapped": 0}
    m0 = ledger[ledger["role"] == L.ROLE_M0]
    if not len(m0):
        return out
    kids_of = ledger.loc[ledger["role"] == L.ROLE_ISO, "parent_peak_id"].value_counts()
    # the batch axis model's edge carry rides on the calibration: a winner just outside the
    # model's scope is judged at the centre plus the model's edge (tiers._cal_z)
    cal = T._calibrate(m0, kids_of, abs_floor_mda=cfg.cal_abs_floor_mda,
                       carry=getattr(cfg, "axis_edge_carry", None))
    if cal is None:
        log("[rearbitrate] uncalibrated; off-cal winner re-arbitration skipped")
        return out
    chan_count = m0.groupby("neutral_formula")["adduct"].nunique()
    reflist = cfg.reflist_formulas or frozenset()
    gate = SAT.LineGate(ledger, cfg)     # the commit contract (Orbitrap-class; inert elsewhere)

    for pid in m0["peak_id"].tolist():
        idx = ledger.index[ledger["peak_id"] == pid]
        if not len(idx):
            continue
        i = idx[0]
        r = ledger.loc[i]
        if bool(r.get("locked")) or str(r.get("method") or "").startswith("known:"):
            continue
        ppm = r.get("ppm_error")
        if ppm is None or pd.isna(ppm):
            continue
        # the tier engine's own z (mass-dependent centre when the trend is fitted)
        z_win = T._cal_z(cal, ppm, r.get("mz"))
        if abs(z_win) <= T.Z_TAIL_DEMOTE:
            continue  # winner on-calibration -> the committed reading stands
        # corroboration (same definition as tiers): the evidence that would break
        # a mass degeneracy. Never displace a corroborated winner.
        iso_ev = (kids_of.get(pid, 0) > 0) or bool(T._alts(r.get("isotopologues")))
        cross = int(chan_count.get(r.get("neutral_formula"), 0)) >= 2
        anchor = pd.notna(r.get("anchor_peak_id")) or pd.notna(r.get("series_unit"))
        if iso_ev or cross or anchor:
            continue
        cnt_w = C.parse_formula(str(r.get("neutral_formula") or ""))
        nc_w = cnt_w.get("C", 0)
        dbe_w = C.dbe(cnt_w)
        if nc_w < 1 or dbe_w / nc_w < REARB_WINNER_DBE_PER_C:
            continue  # not an aromatic-monster winner -> leave it (tier owns it)
        raw_w = float(r["ion_score"]) if pd.notna(r.get("ion_score")) else 0.0

        best = None  # (sort_key, alt, z_alt, raw_alt)
        for a in T._alts(r.get("alternatives")):
            af, ad, pa = a.get("formula"), a.get("adduct"), a.get("ppm")
            if not af or pa is None:
                continue
            z_alt = T._cal_z(cal, pa, r.get("mz"))     # same peak, same m/z
            if abs(z_alt) > cfg.cal_z_accept:          # alternative must be on-cal
                continue
            if C.dbe(C.parse_formula(str(af))) >= dbe_w:  # only toward LESS unsaturation
                continue
            if PL.implausible(str(af), profile=profile) is not None:   # ... and only to a plausible one
                continue
            # ... and only to one whose own S / Cl / Br / Si line is not absent
            # where the file would show it (the commit contract)
            if gate.row_verdict(ledger, pid, str(af), _ion_formula_str(str(af), str(ad)))["verdict"] \
                    == SAT.LG_REFUSED:
                out["own_line_refused"] = out.get("own_line_refused", 0) + 1
                continue
            raw = a.get("raw_score")
            raw = a.get("ion_score") if raw is None else raw
            if raw is None:
                continue
            raw = float(raw)
            if raw < REARB_ALT_MIN_SCORE or (raw_w - raw) > REARB_MAX_SCORE_DROP:
                continue
            key = (str(af) not in reflist, -raw, abs(z_alt))  # reflist, score, on-cal
            if best is None or key < best[0]:
                best = (key, a, z_alt, raw)
        if best is None:
            continue

        _, alt, z_alt, raw = best
        af, ad = str(alt["formula"]), str(alt.get("adduct"))
        conf = confidence_label(raw, alt.get("ppm"), 0, False, cfg)
        if conf == "Reject":
            continue
        # the disqualified off-cal monster is recorded in the commentary, NOT
        # re-listed as a competitor (it failed the calibration gate). Remaining
        # alternatives keep the density / margin honest for the tier engine.
        rest = [x for x in T._alts(r.get("alternatives"))
                if not (str(x.get("formula")) == af and str(x.get("adduct")) == ad)]
        eff = alt.get("eff_score")
        eff = raw if eff is None else float(eff)
        rest_eff = [float(x["eff_score"]) for x in rest if x.get("eff_score") is not None]
        margin = (eff - max(rest_eff)) if rest_eff else None
        note = (f"re-arbitrated: displaced off-cal {r.get('neutral_formula')} "
                f"{r.get('adduct')} (z={z_win:+.1f}σ, uncorroborated mass-fit, "
                f"DBE {dbe_w:.0f}) with on-cal {af} {ad} (z={z_alt:+.1f}σ, plausible, "
                f"DBE {C.dbe(C.parse_formula(af)):.0f})")
        L.commit_assignment(
            ledger, pid, neutral_formula=af, adduct=ad,
            ion_formula=_ion_formula_str(af, ad), ion_score=raw, compound_score=raw,
            eff_score=eff, eff_margin=margin, tied=False, ppm_error=alt.get("ppm"),
            pass_no=int(r["pass_no"]) if pd.notna(r.get("pass_no")) else 0,
            method=f"rearb<-{r.get('method')}", confidence=conf, commentary=note,
            alternatives=rest, isotopologues=[], overwrite=True,
        )
        out["swapped"] += 1

    if out["swapped"]:
        log(f"[rearbitrate] displaced {out['swapped']} off-cal aromatic-monster M0 "
            f"winners (>|{T.Z_TAIL_DEMOTE}|σ off-cal, uncorroborated) with on-cal "
            "plausible lower-DBE alternatives")
    return out
