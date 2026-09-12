"""Pass 6 -- anchored homolog / oxidation-ladder gap-fill.

Diagonal families in the residual (the rotating-GKA view) are homologous /
functionalisation SERIES: a committed analyte and its neighbours related by an
exact repeat unit -- +O oxidation, +/-CH2 / +C2H4 carbon homology, +CO / +CO2 /
+CH2O functionalisation, -H2O dehydration. This pass walks those ladders OUT
from committed anchors and fills the gaps primary matching missed, validated by
Mascope and gated hard against the false positives the adversarial diagonal
analysis surfaced (2026-06-12):

  G1 same adduct along a ladder -- mixed [M-H]- nitrate -> [M+HBr+Br]- di-bromide
     ladders were the dominant false positive at the non-integer GKA rotation;
  G2 valid neighbour chemistry -- integer DBE>=0, structural O cap; +/-CH2 and
     +C2H4 conserve DBE by construction (true homologs);
  G3 the target peak is bright and currently UNEXPLAINED (or a weak Candidate);
  G4 isotope-satellite guard -- 81Br (+1.99795) / 13C (+1.00336) lines of a
     brighter peak masquerade as ladder members; reject them;
  G5 never overwrite a strong competing M0 (ion_score >= 0.90) -- the di-bromide
     reading of a peak already owned by a confident fluorinated/other M0 is kept
     only as a flagged alternative;
  G6 Mascope confirms the proposed neutral under its adduct (score >= tau_good
     in calibrated terms); accept up to ppm_max on the strength of ladder
     membership (this is a completion pass, so it carries pattern evidence the
     bare mass gate lacks);
  G7 Candidate tier only, confidence inherited+capped from the anchor.

[M+HBr+Br]- / [M+HBr+Br2]- have no native server mechanism, so they are scored
through their covalent-equivalent ion (neutral + H2Br2 / H2Br3 as [M-H]-), the
SAME ion -- exactly the alias the reagent-Br policy trades on.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from peaky.chem import chemistry as C
from peaky.io import io_mascope as IO
from peaky.assignment import ledger as L
from peaky.assignment.passes import PassConfig, confidence_label, z_of, _prefer_adduct_reading

__version__ = "0.1.0"

_D_BR = 1.99795
_D_13C = 1.0033548

# repeat units as signed element-count deltas. Oxidation/functionalisation +
# carbon homology -- the inventory that reproduced every confirmed rung.
_UNITS: dict[str, dict[str, int]] = {
    "+O":    {"O": 1},
    "+2O":   {"O": 2},
    "+CO":   {"C": 1, "O": 1},
    "+CO2":  {"C": 1, "O": 2},
    "+CH2O": {"C": 1, "H": 2, "O": 1},
    "+CH2":  {"C": 1, "H": 2},
    "-CH2":  {"C": -1, "H": -2},
    "+C2H4": {"C": 2, "H": 4},
    "-H2O":  {"H": -2, "O": -1},
}
# ladders are only walked under genuine analyte adducts
_ANCHOR_ADDUCTS = ("[M+Br]-", "[M+HBr+Br]-", "[M-H]-")
# ladders are only walked on SOA-like CHON(S) backbones -- fluorinated/chlorinated
# /Si contaminants form their OWN homolog ladders (perfluoro CH2 series, siloxanes)
# that the adversarial diagonal analysis flagged as the dominant false positive,
# so an anchor carrying any of these is never a SOA ladder seed.
_SOA_ELEMENTS = frozenset("CHONS")
# absolute oxygen ceiling for a ladder member: validated SOA tops out at O5;
# O>9 only ever appeared as the lattice-monster mass fits (C21H14O15-type)
_O_CEIL = 9


def _apply(counts: dict[str, int], delta: dict[str, int], k: int) -> dict[str, int] | None:
    out = dict(counts)
    for el, dv in delta.items():
        out[el] = out.get(el, 0) + dv * k
        if out[el] < 0:
            return None
    return {el: n for el, n in out.items() if n > 0}


def _scoreable(neutral: str, adduct: str) -> tuple[str, str]:
    """Map a (neutral, adduct) proposal to a (formula, native_adduct) the server
    can score. [M+HBr+Br]-/[M+HBr+Br2]- -> covalent-equivalent as [M-H]- (same
    ion). Everything else scores natively."""
    if adduct == "[M+HBr+Br]-":
        c = dict(C.parse_formula(neutral)); c["H"] = c.get("H", 0) + 2; c["Br"] = c.get("Br", 0) + 2
        return C.format_formula(c), "[M-H]-"
    if adduct == "[M+HBr+Br2]-":
        c = dict(C.parse_formula(neutral)); c["H"] = c.get("H", 0) + 2; c["Br"] = c.get("Br", 0) + 3
        return C.format_formula(c), "[M-H]-"
    return neutral, adduct


def _is_iso_satellite(mzs, hs, mz: float, h: float) -> bool:
    """True if this peak is a likely 81Br (+1.998) or 13C (+1.003) satellite of a
    brighter peak -- the trap that put fake members on the GKA diagonals."""
    for d, lo, hi in ((_D_BR, 0.5, 1.7), (_D_13C, 0.02, 0.5)):
        t = mz - d
        i = int(np.abs(mzs - t).argmin())
        if abs(mzs[i] - t) <= t * 6e-6:
            hp = hs[i]
            if hp > 0 and hp >= h and lo <= h / hp <= hi:
                return True
    return False


def run_ladder_gapfill(client, sample_id: str, ledger: pd.DataFrame, profile,
                       cfg: PassConfig, adducts: list[str], *,
                       score_fn=None, log=print, max_steps: int = 4,
                       ppm_max: float = 3.0) -> dict:
    score_fn = score_fn or IO.score_candidates
    out = {"committed": 0, "alternatives": 0, "rejected_satellite": 0,
           "rejected_overwrite": 0}
    m0 = ledger[ledger["role"] == L.ROLE_M0]
    anchors = m0[m0["adduct"].astype(str).isin(_ANCHOR_ADDUCTS)
                 & m0["neutral_formula"].notna()
                 & m0["neutral_formula"].map(
                     lambda f: set(C.parse_formula(f)) <= _SOA_ELEMENTS)]
    if not len(anchors):
        return out
    # already-committed (neutral, adduct) ions -- the SAME neutral at a DIFFERENT
    # adduct is a distinct ion/peak and stays proposable (this is the +HBr
    # pairing case: the di-bromide form of a neutral already seen at [M+Br]-)
    existing = set(zip(m0["neutral_formula"], m0["adduct"].astype(str)))
    # bromine-free neutrals already seen at [M+Br]- -- the +HBr pairing evidence
    # that lets a di-bromide [M+HBr+Br]- gap commit at a lower score
    br_neutrals = set(m0.loc[m0["adduct"].astype(str) == "[M+Br]-", "neutral_formula"].dropna())
    from peaky.assignment import admission as ADM
    tgt = ledger[(ledger["role"].isin([L.ROLE_UNEXPLAINED, L.ROLE_M0]))
                 & ADM.admissible(ledger, cfg)]
    tmz = tgt["mz"].to_numpy()
    torder = np.argsort(tmz)
    tmz_s = tmz[torder]
    tpid = tgt["peak_id"].to_numpy()[torder]
    mzs = ledger["mz"].to_numpy(); hs = ledger["height"].to_numpy()

    def _peak_for(theo):
        i = int(np.abs(tmz_s - theo).argmin())
        if abs(tmz_s[i] - theo) <= theo * ppm_max * 1e-6:
            return tpid[i], tmz_s[i]
        return None, None

    # generate proposals: anchor -> neighbour under the SAME adduct, that lands
    # on a bright currently-unexplained (or weak-Candidate) peak
    proposals: dict[object, dict] = {}   # peak_id -> proposal
    for _, a in anchors.iterrows():
        nf = str(a["neutral_formula"]); adduct = str(a["adduct"])
        a_conf = str(a["confidence"])
        cnt = C.parse_formula(nf)
        for uname, delta in _UNITS.items():
            for k in range(1, max_steps + 1):
                nb = _apply(cnt, delta, k)
                if nb is None:
                    continue
                # SOA chemistry gate: O ceiling kills the lattice-monster mass
                # fits the loose Senior cap let through (C21H14O15-type)
                if nb.get("O", 0) > min(nb.get("C", 1), _O_CEIL):
                    continue
                f = C.format_formula(nb)
                if (f, adduct) in existing:
                    continue
                ok, _ = C.dbe_ok(f)
                if ok:
                    ok, _ = C.oxygen_ok(f)
                if not ok:
                    continue
                theo = C.ion_mz(f, adduct)
                pid, pmz = _peak_for(theo)
                if pid is None:
                    continue
                # G3: claim only currently-UNEXPLAINED peaks. Never overwrite an
                # existing M0 -- pass 6 is a completion pass and must not displace
                # a primary identification (the v33 TFA-overwrite regression).
                i0 = ledger.index[ledger["peak_id"] == pid][0]
                if str(ledger.at[i0, "role"]) != L.ROLE_UNEXPLAINED:
                    out["rejected_overwrite"] += 1
                    continue
                h = float(ledger.at[i0, "height"])
                # G4: isotope-satellite guard (81Br/13C lines on the diagonals)
                if _is_iso_satellite(mzs, hs, pmz, h):
                    out["rejected_satellite"] += 1
                    continue
                ppm = (pmz - theo) / theo * 1e6
                cur = proposals.get(pid)
                # prefer fewer steps, then better ppm, then stronger anchor
                key = (k, abs(ppm), 0 if a_conf.startswith(("High", "Good")) else 1)
                if cur is None or key < cur["key"]:
                    proposals[pid] = {"neutral": f, "adduct": adduct, "anchor": nf,
                                      "anchor_pid": a["peak_id"], "unit": uname,
                                      "steps": k, "ppm": ppm, "pmz": pmz,
                                      "a_conf": a_conf, "key": key}
    if not proposals:
        log("[pass6] no ladder gap-fill proposals")
        return out
    log(f"[pass6] {len(proposals)} ladder gap-fill proposals "
        f"(anchored homolog/oxidation steps); scoring...")
    # score (covalent-equiv for the HBr-cluster adducts), one server round
    form_of = {pid: _scoreable(p["neutral"], p["adduct"]) for pid, p in proposals.items()}
    allf = sorted({f for f, _ in form_of.values()})
    scored = score_fn(client, sample_id, allf, mechanism_ids=cfg.mechanism_ids)
    if scored is None or len(scored) == 0:
        log("[pass6] WARNING scoring returned EMPTY -- residual left unfilled")
        out["scoring_empty"] = True
        return out
    base = scored[scored["is_base"] & scored["sample_peak_id"].notna()
                  & scored["ion_score"].notna()].copy()
    for pid, p in proposals.items():
        f, _ = form_of[pid]
        rows = base[base["compound_formula"] == f]
        if not len(rows):
            continue
        rows = rows.assign(d=(rows["sample_peak_mz"] - p["pmz"]).abs()).sort_values("d")
        if rows.iloc[0]["d"] > p["pmz"] * ppm_max * 1e-6:
            continue
        r = rows.iloc[0]
        raw = float(min([v for v in (r["ion_score"], r["compound_score"])
                         if pd.notna(v)] or [0.0]))
        ppm = float(r["ppm_error"]) if pd.notna(r["ppm_error"]) else p["ppm"]
        # G6 acceptance. A bare ladder fill needs a Good-grade score (tau_good).
        # A di-bromide [M+HBr+Br]- gap may commit at a lower score IF its
        # bromine-free neutral is independently seen at [M+Br]- (the +HBr pairing
        # corroboration) -- these score 0.65-0.69 because the 2-Br envelope is
        # weaker, but the pairing is the evidence the score lacks.
        paired = (p["adduct"] == "[M+HBr+Br]-" and p["neutral"] in br_neutrals)
        # paired di-bromide gaps score 0.63-0.69 (weak 2-Br envelope) -- the
        # +HBr pairing is the evidence the score lacks, so they pass at the
        # suspect floor; a bare [M+Br]- ladder fill must reach tau_good.
        floor = cfg.tau_suspect if paired else cfg.tau_good
        if raw < floor:
            out.setdefault("rejected_score", 0)
            out["rejected_score"] += 1
            continue
        # confidence capped by anchor strength; commit only Good/Low (never High)
        cap = "Good" if p["a_conf"].startswith(("High", "Good")) else "Low"
        conf = confidence_label(raw, ppm, 0, False, cfg, suffix="ladder")
        conf = conf.replace("High", "Good")
        if cap == "Low" or paired:
            conf = conf.replace("Good", "Low")
        if conf.startswith("Reject"):
            continue
        try:
            if L.role_of(ledger, pid) != L.ROLE_UNEXPLAINED:
                continue   # became claimed since proposal time
            w = pd.Series({"neutral": p["neutral"], "adduct": p["adduct"],
                           "ion_formula": r["ion_formula"], "raw_score": raw})
            w = _prefer_adduct_reading(w, cfg)
            note = w["_relabel_note"] if "_relabel_note" in w else ""
            L.commit_assignment(
                ledger, pid, neutral_formula=w["neutral"], adduct=w["adduct"],
                ion_formula=r["ion_formula"], ion_score=float(r["ion_score"]),
                compound_score=(float(r["compound_score"]) if pd.notna(r["compound_score"]) else None),
                ppm_error=ppm, pass_no=6, method="ladder:gapfill", confidence=conf,
                commentary=(f"Pass 6 (ladder gap-fill): {p['steps']:+d}x{p['unit']} "
                            f"from anchor {p['anchor']} {p['adduct']} (peak "
                            f"{p['anchor_pid']}); {w['neutral']} {w['adduct']} scored "
                            f"{raw:.2f} by Mascope at {ppm:+.2f} ppm. Homologous-series "
                            f"completion, Candidate.{note}"),
                anchor_peak_id=p["anchor_pid"], series_unit=p["unit"])
            out["committed"] += 1
        except L.LedgerError:
            continue
    log(f"[pass6] {out}")
    return out
