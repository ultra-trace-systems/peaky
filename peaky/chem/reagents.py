"""Reagent-ion library + labeler.

In chemical-ionization MS the reagent anion (Br-, I-, NO3-, ...) forms bright
cluster ions that are NOT sample chemistry: bare R_n clusters and R.(neutral)
clusters (water, HNO3, small acids). These otherwise land in 'unexplained' and
dominate the residual by signal. This module enumerates the cluster m/z (with
halogen isotopologue combinations) and labels matching ledger peaks as reagent.

It is keyed on the reagent detected from the sample's adducts, so a Br-CIMS run
gets the Br_n / Br.(acid) library and an I-CIMS run gets the I_n library.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from peaky.chem import chemistry as C
from peaky.assignment import ledger as L

__version__ = "0.5.0"

# isotope masses for the reagent halogens (light, heavy, heavy abundance)
_HALOGEN_ISO = {
    "Br": [(78.9183371, "79Br"), (80.9162906, "81Br")],
    "Cl": [(34.96885268, "35Cl"), (36.96590259, "37Cl")],
    "I":  [(126.9044719, "127I")],
}
_M_E = C.M_E

# small neutrals that cluster onto the reagent anion. ONLY genuine reagent /
# background species belong here -- water and the hydrogen halide the reagent
# itself sheds (HBr for Br-CIMS, HI for I-CIMS: resolved per reagent in
# build_library, so an iodide library no longer carries phantom [In+HBr]-
# entries). Organic acids (HCOOH/CH3COOH/pyruvic/pinic/...) were REMOVED
# 2026-06-12: a [Br1+acid]- ion IS [acid+Br]- = the primary [M+Br]- ANALYTE
# channel, so the labeler was stealing real ambient-acid analytes (formic
# acid's 232k-cps line among them) and burying them as "reagent". They are now
# left for the assignment passes, exactly like the HNO3/HNO2 ruling.
_CLUSTER_NEUTRALS = {
    "H2O": "H2O",
    # HF clusters onto bromide as [Br+HF]- = BrHF- (m/z 98.925). HF is a
    # background/contaminant volatile (released by the fluorinated instrument
    # background, abundant in this source), not an analyte -- it was the only
    # clean ID among the variable-unassigned residual (time-series v47, -0.14 ppm),
    # so it belongs in the inorganic/halogen background-cluster list, not as an
    # organic. Extra Br_n.HF entries are harmless (labeled only if present).
    "HF": "HF",
}


# Iodine-source poly-iodide background clusters. Learned from the
# 2026-07-21 iodide batch (docs/REAGENTS.md): besides the bare In⁻ ladder and the
# IOₓ⁻ oxides (both generated below), an iodide source throws bright + time-STABLE
# pure-iodine-oxide clusters I₂O⁻ (269.80, ~2M cps) and I₃O⁻ (396.71). These are
# source background, so they are LABELLED (immovable) rather than left in the
# residual or mis-read as exotic organoiodines (I is off the neutral grid).
# HOI₂⁻ and I₂NO₂⁻ are deliberately NOT here: they are the [M+I]- analyte reading
# of HOI / INO₂ (time-VARYING ambient reactive-iodine species, 55x/2.3x over the
# 2026-07-21 batch) -- pass-0 `reactive_iodine` known species, the same ruling as
# the reagent-acid clusters below. ⚠ CAVEAT (mirror of the I₃⁻/ambient-I₂ blind
# spot): I₂O⁻ is composition-identical to [IO+I]-, so a locked I₂O⁻ hides any
# ambient IO radical -- a campaign targeting IO must check I₂O⁻/I₂⁻ ratio drift
# before trusting this label. Iodine-specific -- generated only for
# reagent == "I". Formula = the ION composition (anion: +1 electron added by
# build_library, so the dict here is the neutral atom count).
_IODINE_BACKGROUND: dict[str, dict[str, int]] = {
    "[I2O]-": {"I": 2, "O": 1},
    "[I3O]-": {"I": 3, "O": 1},
}


# Positive-mode reagent ions: a protonated-reagent cluster series [R_n + H]+.
# Keyed like the halogen library but the cluster is a CATION (lose an electron),
# the repeat unit is a molecular neutral (urea CH4N2O), and there is no isotope
# branching (C/H/N/O reagents have no bright heavy-isotope cluster the way Br
# does). Used by urea-CIMS (uronium): [urea_n+H]+ at 61.04 / 121.07 / 181.10 /
# 241.14 ... are the dominant reagent ions and otherwise dominate 'unexplained'.
_POSITIVE_REAGENTS = {
    "urea": "CH4N2O",
}


# EasyIC⁺ source ions -- the Orbitrap's internal-calibration (EASY-IC)
# fluoranthene cation beam used as a LOW-PRESSURE charge-transfer reagent
# (KORBI2 EasyIC+ batches, 2026). Three ion classes, all pure source -- no
# analyte atoms -- so they must be LABELLED, not left red in the residual:
#   * the fluoranthene ladder: C16H10+. itself (202.0776, the reagent ion;
#     server 'test compound'), its H-loss fragment C16H9+, protonated C16H11+
#     and the dimer cation (C32H20+., in-window only for mz40-500);
#   * the air-plasma cations the discharge throws: N3+ (42.0087) and NO2+
#     (45.9924) are the bright in-window pair (both server-matched on the
#     2026-02-26 mz40-500 batch); NO+/O2+. sit below a 40 Da window start but
#     are kept for wider windows;
#   * urea CROSSOVER: the instrument alternates with the uronium source, and
#     [urea_n+H]+ (61.0396 / 121.0720) DOMINATE the mz40-160 EasyIC batches --
#     the urea positive library is merged in so they read as source ions, not
#     analytes. Formula dicts here are the NEUTRAL composition; build adds the
#     cation charge (lose an electron).
_EASYIC_SOURCE_IONS: dict[str, dict[str, int]] = {
    "[C16H10]+. (fluoranthene)": {"C": 16, "H": 10},
    "[C16H10-H]+ (fluoranthene fragment)": {"C": 16, "H": 9},
    "[C16H10+H]+ (fluoranthene)": {"C": 16, "H": 11},
    "[(C16H10)2]+. (fluoranthene dimer)": {"C": 32, "H": 20},
    "[N3]+ (air plasma)": {"N": 3},
    "[NO2]+ (air plasma)": {"N": 1, "O": 2},
    "[NO]+ (air plasma)": {"N": 1, "O": 1},
    "[O2]+. (air plasma)": {"O": 2},
}


def _build_easyic_library() -> list[tuple[str, float, str]]:
    """[(label, ion_mz, ion_formula)] for the EasyIC⁺ source ions + the urea
    crossover series (see _EASYIC_SOURCE_IONS)."""
    out = [(label, C.neutral_mass(d) - _M_E, C.format_formula(d) + "+")
           for label, d in _EASYIC_SOURCE_IONS.items()]
    out.extend(_build_positive_library("urea"))
    return out


def _proton() -> float:
    return C.M["H"] - _M_E   # H+ = proton (one H atom minus its electron)


def _build_positive_library(reagent: str, *, max_n: int = 6
                            ) -> list[tuple[str, float, str]]:
    """[(label, ion_mz, ion_formula)] for the protonated-reagent cluster series,
    n = 1..max_n:

      * [R_n + H]+   -- the bare protonated cluster (61.04 / 121.07 / ...);
      * [R_n + NH4]+ for n >= 2 -- a urea MULTIMER charged by ambient ammonia; an
        ion-source cluster. (The n=1 case [urea+NH4]+ == [NH3+(urea)H]+ at 78.07 is
        NOT here: it is ammonia measured via its single urea adduct -- the same ion,
        but the meaningful neutral is the AMBIENT NH3 analyte, so it is registered
        as a pass-0 known species, not stolen as reagent.)

    The ion has the elemental composition R_n + (H or NH4) and charge +1 (an
    electron removed)."""
    unit = _POSITIVE_REAGENTS.get(reagent)
    if unit is None:
        return []
    unit_cnt = C.parse_formula(unit)
    out: list[tuple[str, float, str]] = []
    for n in range(1, max_n + 1):
        base = {el: c * n for el, c in unit_cnt.items()}
        # [R_n + H]+
        dH = dict(base); dH["H"] = dH.get("H", 0) + 1
        out.append((f"[({unit}){n}+H]+", C.neutral_mass(dH) - _M_E,
                    C.format_formula(dH) + "+"))
        # [R_n + NH4]+ for n>=2 (ammonia on a urea multimer = ion-source cluster);
        # n=1 is the ambient-NH3 analyte (pass-0 known species), left off here.
        if n >= 2:
            dN = dict(base); dN["N"] = dN.get("N", 0) + 1; dN["H"] = dN.get("H", 0) + 4
            out.append((f"[({unit}){n}+NH4]+", C.neutral_mass(dN) - _M_E,
                        C.format_formula(dN) + "+"))
    return out


def build_library(reagent: str = "Br", *, max_n: int = 4, max_neutral: int = 1
                  ) -> list[tuple[str, float, str]]:
    """Return [(label, ion_mz, ion_formula)] for the reagent's cluster ions.

    Negative-mode halide reagents (Br/Cl/I):
      * bare R_n^-  (odd n = closed-shell anion R-, R3-, R5-; even n = radical
        anion R2-., R4-. -- e.g. the di-bromide Br2-. the user registered on
        the server 2026-06-12)
      * R_n^- . (neutral)_k  for the reagent/background neutral list
      * the reagent-oxide anions RO-/RO2-/RO3-
    All halogen isotopologue combinations are enumerated.

    Positive-mode molecular reagents (urea):
      * the protonated cluster series [R_n + H]+ (see _build_positive_library).

    The ion_formula is the elemental composition of the ion -- a reagent cluster
    has a KNOWN formula, so it must be recorded as an assignment, not left blank:
    known formula -> assigned, regardless of whether the species is an analyte or
    an ion-source cluster (it's just a different class)."""
    if reagent == "EasyIC":
        return _build_easyic_library()
    if reagent in _POSITIVE_REAGENTS:
        return _build_positive_library(reagent)
    if reagent not in _HALOGEN_ISO:
        return []
    isos = _HALOGEN_ISO[reagent]
    out: list[tuple[str, float, str]] = []

    # bare R_n clusters (charge -1) for n = 1..max_n. Both parities are real
    # reagent ions in a halide source: odd n are closed-shell (R-, R3-), even n
    # are radical anions (R2-., R4-.). All are pure reagent -- no analyte atoms
    # -- so they must be LABELLED, not left red in the residual.
    core_masses: list[tuple[str, float, int]] = []   # (label, mass, n)
    for n in range(1, max_n + 1):
        f_core = C.format_formula({reagent: n}) + "-"
        for combo in itertools.combinations_with_replacement(range(len(isos)), n):
            mass = sum(isos[i][0] for i in combo) + _M_E   # anion: +1 electron
            tag = "+".join(isos[i][1] for i in combo)
            radical = "." if n % 2 == 0 else ""
            label = f"[{reagent}{n}]-{radical} ({tag})"
            out.append((label, mass, f_core))
            core_masses.append((label, mass, n))

    # R_n^- . (neutral)_k clusters -- neutrals adduct onto each bare core.
    # The shed hydrogen halide is the REAGENT'S OWN (HBr / HCl / HI), not a
    # fixed HBr: [I+HI]- is the iodide analog of the Br-CIMS [Br+HBr]-.
    neutrals = dict(_CLUSTER_NEUTRALS)
    neutrals["H" + reagent] = "H" + reagent
    for label, core_mz, n in core_masses:
        for name, formula in neutrals.items():
            nm = C.neutral_mass(formula)
            for k in range(1, max_neutral + 1):
                d = {reagent: n}
                for el, c in C.parse_formula(formula).items():
                    d[el] = d.get(el, 0) + k * c
                out.append((f"[{reagent}{n}+{k}x{name}]-", core_mz + k * nm,
                            C.format_formula(d) + "-"))

    # reagent-halogen oxide anions RO-/RO2-/RO3- -- BOTH halogen isotopologues
    # (the 81Br twin of BrO- at 96.91 was previously dropped: only isos[0] was
    # used, so [81BrO]- never entered the library and sat in 'unexplained').
    # Br/Cl ONLY: for iodine the IOx- anions are ion-identical to the [M-H]-
    # deprotonation of the iodine OXYACIDS (IO- == [HOI-H]-, IO2- == [HIO2-H]-,
    # IO3- == [HIO3-H]- -- and iodate is the SIGNATURE of iodic acid, THE key
    # iodide-CIMS analyte for new-particle formation). Labelling them reagent
    # would lock the acids' dominant channel away from pass-0 `reactive_iodine`
    # -- the HNO3/NO3- ruling, applied to iodine oxides.
    if reagent != "I":
        for no in (1, 2, 3):
            f_ox = C.format_formula({reagent: 1, "O": no}) + "-"
            for biso_mass, biso_tag in isos:
                out.append((f"[{reagent}O{no if no > 1 else ''}]- ({biso_tag})",
                            biso_mass + no * C.M["O"] + _M_E, f_ox))

    # iodine-source poly-iodide background clusters (I₂O⁻/HOI₂⁻/I₂NO₂⁻/I₃O⁻) --
    # the bright non-ladder iodine ions this source throws. Iodine-specific.
    if reagent == "I":
        for label, d in _IODINE_BACKGROUND.items():
            out.append((label, C.neutral_mass(d) + _M_E, C.format_formula(d) + "-"))
    return out


def label_reagents(ledger: pd.DataFrame, reagent: str = "Br", *, ppm: float = 15.0,
                   only_unexplained: bool = True, lock: bool = True) -> int:
    """Mark ledger peaks matching a reagent-cluster m/z as role='reagent'.

    Reagent clusters are NOT sample chemistry, so once claimed they must be
    immovable: `lock=True` (default) LOCKS each labelled peak so a later pass with
    `claim_unexplained_only=False` (e.g. the pass-1 backbone) cannot overwrite the
    reagent label with an analyte M0 -- the bug that let the bright [urea+H]+
    (61.04) and [urea2+H]+ (121.07) reagent ions be re-read as `CHNO`/`CH4N2O`
    ammonium-adduct analytes and dominate the 'assigned' signal.

    `only_unexplained=True` skips already-committed peaks; use
    `reclaim_reagent_clusters` to DISPLACE an M0 that a pass already put on a
    reagent mass. Returns the number of peaks labelled."""
    lib = build_library(reagent)
    if not lib:
        return 0
    lib_sorted = sorted(lib, key=lambda x: x[1])
    masses = [t[1] for t in lib_sorted]
    import bisect
    n = 0
    for i, row in ledger.iterrows():
        if only_unexplained and row["role"] != L.ROLE_UNEXPLAINED:
            continue
        mz = row["mz"]
        tol = mz * ppm * 1e-6
        lo = bisect.bisect_left(masses, mz - tol)
        hi = bisect.bisect_right(masses, mz + tol)
        if hi > lo:
            # nearest label; record the KNOWN ion formula as the assignment
            best = min(lib_sorted[lo:hi], key=lambda x: abs(x[1] - mz))
            try:
                L.mark_reagent(ledger, row["peak_id"],
                               f"reagent ion: {best[0]} ({(mz-best[1])/best[1]*1e6:+.1f} ppm)",
                               ion_formula=best[2])
                if lock:
                    L.lock_peaks(ledger, [row["peak_id"]])
                n += 1
            except L.LedgerError:
                continue
    return n


def label_reagent_isotopologues(ledger: pd.DataFrame, *, ppm: float = 12.0,
                                min_rel: float = 0.004, gate: float = 3.0,
                                log=print) -> dict:
    """Claim the ISOTOPE SATELLITES of every already-labelled reagent ion.

    A reagent cluster ion is bright, so its ¹³C/¹⁵N (and for halide reagents ³⁷Cl/
    ⁸¹Br) satellites are large in ABSOLUTE terms even though small relative to the
    parent -- the ¹³C/¹⁵N of the 15.9M-cps urea dimer [urea2+H]+ are 332k/224k cps.
    The reagent labeler claimed only the monoisotopic line, and
    `complete_isotope_envelopes` runs on role=M0 analytes only, so these satellites
    landed in 'unexplained' and dominated the residual. This predicts each reagent
    ion's envelope from its KNOWN ion_formula (`isotopes.isotope_pattern`) and marks
    the matching UNEXPLAINED peaks reagent too.

    Guarded by intensity: a satellite peak is claimed only if its height is within
    `gate`x the predicted (rel_intensity x parent height) -- a peak far BRIGHTER than
    the reagent satellite should be has a co-eluting analyte on top, so it is left.
    `merge_da` is tightened to 3 mDa so the ¹³C (+1.00335) and ¹⁵N (+0.99703) lines
    stay resolved (they are separate peaks in the data, 6.3 mDa apart)."""
    from peaky.chem import isotopes as ISO
    if "height" not in ledger.columns:
        return {"iso_reagent": 0}
    reag = ledger[(ledger["role"] == L.ROLE_REAGENT)
                  & ledger["ion_formula"].notna()].copy()
    if not len(reag):
        return {"iso_reagent": 0}
    mzs = ledger["mz"].to_numpy()
    claimed = 0
    for _, pr in reag.iterrows():
        pmz = float(pr["mz"]); ph = float(pr["height"]) if pd.notna(pr["height"]) else 0.0
        ionf = str(pr["ion_formula"]).rstrip("+-")
        try:
            pattern = ISO.isotope_pattern(ionf, min_rel=min_rel, max_shift=6.5,
                                          merge_da=0.003)
        except Exception:
            continue
        for dm, rel, lab in pattern:
            if dm <= 0.01:                       # skip the M0 line itself
                continue
            target = pmz + dm
            tol = target * ppm * 1e-6
            cand = ledger.index[(np.abs(mzs - target) <= tol)
                                & (ledger["role"].to_numpy() == L.ROLE_UNEXPLAINED)]
            if not len(cand):
                continue
            i = min(cand, key=lambda j: abs(float(ledger.at[j, "mz"]) - target))
            h = float(ledger.at[i, "height"]) if pd.notna(ledger.at[i, "height"]) else 0.0
            pred = rel * ph
            if pred > 0 and h > gate * pred:     # analyte sitting on top -> leave it
                continue
            try:
                L.mark_reagent(ledger, ledger.at[i, "peak_id"],
                               f"reagent isotopologue: {lab} of {pr['ion_formula']} "
                               f"(rel {rel*100:.1f}%)",
                               ion_formula=pr["ion_formula"])
                L.lock_peaks(ledger, [ledger.at[i, "peak_id"]])
                claimed += 1
            except L.LedgerError:
                continue
    if claimed:
        log(f"[reagent] claimed {claimed} reagent isotopologue satellite(s) "
            f"(¹³C/¹⁵N/heavy-halogen of the reagent ions)")
    return {"iso_reagent": claimed}


def reclaim_reagent_clusters(ledger: pd.DataFrame, reagent: str = "Br", *,
                             ppm: float = 12.0, log=print) -> dict:
    """Authoritatively claim the reagent-cluster masses: for each library ion, take
    the BRIGHTEST peak within `ppm` and force it to role='reagent' (+lock), even if
    a pass already committed an analyte M0 there (displacing that phantom and its
    isotope children). This is the post-hoc guard for the reagent-vs-analyte
    exact-mass degeneracy (urea `[R_n+H]+`/`[R_n+NH4]+` == `CHNO`/`CH4N2O`
    ammonium/urea analyte reading). Returns counts."""
    lib = build_library(reagent)
    if not lib or "height" not in ledger.columns:
        return {"reagent": 0, "displaced_m0": 0}
    reag = disp = 0
    mzs = ledger["mz"].to_numpy()
    for _label, ion_mz, ion_formula in lib:
        tol = ion_mz * ppm * 1e-6
        cand = ledger.index[(ledger["mz"] - ion_mz).abs() <= tol]
        cand = [i for i in cand if not bool(ledger.at[i, "locked"])]
        if not cand:
            continue
        # brightest peak at this reagent mass is the reagent ion itself
        i = max(cand, key=lambda j: (ledger.at[j, "height"]
                                     if pd.notna(ledger.at[j, "height"]) else 0.0))
        pid = ledger.at[i, "peak_id"]
        role = str(ledger.at[i, "role"])
        if role == L.ROLE_REAGENT:
            continue
        if role == L.ROLE_M0:
            L.clear_assignment(ledger, pid, reason=f"reagent cluster {_label}")
            disp += 1
        try:
            L.mark_reagent(ledger, pid,
                           f"reagent ion: {_label} "
                           f"({(ledger.at[i, 'mz']-ion_mz)/ion_mz*1e6:+.1f} ppm)",
                           ion_formula=ion_formula)
            L.lock_peaks(ledger, [pid])
            reag += 1
        except L.LedgerError:
            continue
    if reag:
        log(f"[reagent] reclaimed {reag} reagent-cluster ion(s) "
            f"({disp} displaced an analyte M0 phantom)")
    return {"reagent": reag, "displaced_m0": disp}


def strip_reagent_cluster_rows(merged: pd.DataFrame, reagent: str = "Br", *,
                               ppm: float = 12.0, log=print):
    """Merge-level guard: drop analyte M0 rows whose ion m/z coincides with a
    reagent-cluster ion. The batch merge keeps only per-file M0 rows, so a reagent
    cluster that a per-file pass mislabelled as an analyte (urea `[R_n+H]+` read as
    `CHNO`/`CH4N2O` on the [M+NH4]+/urea channel) survives into the merged ledger
    and dominates the 'assigned' signal. This removes them by exact-mass match to
    the reagent library. Returns (kept, stripped) DataFrames."""
    lib = build_library(reagent)
    if not lib or not len(merged):
        return merged, merged.iloc[0:0]
    masses = sorted(m for _l, m, _f in lib)
    import bisect
    drop = []
    for i, mz in merged["mz"].items():
        tol = mz * ppm * 1e-6
        lo = bisect.bisect_left(masses, mz - tol)
        hi = bisect.bisect_right(masses, mz + tol)
        if hi > lo:
            drop.append(i)
    stripped = merged.loc[drop]
    kept = merged.drop(index=drop).reset_index(drop=True)
    if len(stripped):
        log(f"[reagent] merge guard: removed {len(stripped)} reagent-cluster ion(s) "
            f"mislabelled as analyte ({', '.join(stripped['neutral_formula'].astype(str).head(4))}...)")
    return kept, stripped


def reagent_for_adducts(adducts: list[str]) -> str | None:
    """Pick the reagent-cluster library key implied by the sample's adducts.

    Returns a halogen symbol ("Br"/"Cl"/"I") for a halide-CIMS source, or a
    positive molecular-reagent key ("urea") for a urea-CIMS / uronium source.
    NB: this is the CLUSTER-LIBRARY key, not the arbitration `reagent_element`
    (a molecular reagent puts no halogen in the neutral, so assign.run sets
    cfg.reagent_element only for the halogen keys)."""
    for a in adducts:
        # positive molecular reagents: the urea adduct [M+(CH4N2O)H]+
        if "CH4N2O" in a:
            return "urea"
        # EasyIC fluoranthene charge transfer: the molecular-cation channel
        if a == "[M]+.":
            return "EasyIC"
        if "Br" in a:
            return "Br"
        if a.endswith("I]-") or "+I" in a:
            return "I"
        if "Cl" in a:
            return "Cl"
    return None
