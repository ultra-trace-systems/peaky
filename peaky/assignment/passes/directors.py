"""passes.directors — split from the former passes.py monolith."""

from __future__ import annotations


import pandas as pd

from peaky.chem import chemistry as C
from peaky.chem import contexts as X
from peaky.io import io_mascope as IO
from peaky.assignment import ledger as L
from peaky.assignment import series_gka as G


from .config import PassConfig
from .core import _f, _mech_to_adduct, arbitrate, commit_winners, confidence_label
from .postprocess import _DBR, _peak_near, _si_m1_consistent

__all__ = [
    "_silanediol_series",
    "_known_species",
    "_D37CL",
    "_RECOVERABLE_KNOWN_FAMS",
    "run_pass0_known",
    "_recover_isotope_locked_known",
    "run_pass5_completion",
    "build_ranges",
    "ranges_to_string",
    "_resolve_hx_clusters",
    "_resolve_acid_i2_clusters",
    "_target_peaks",
    "_family_ok",
    "_context_filter",
    "_enumerate",
    "_mech_ids_for",
    "run_pass1",
    "run_pass2",
    "run_pass3",
    "run_pass_certified",
]


def _silanediol_series(n_max: int = 8) -> list[str]:
    """HO-(Si(CH3)2-O)n-H: PDMS hydrolysis products, the classic inlet/tubing
    contamination. Composition C(2n)H(6n+2)O(n+1)Si(n)."""
    return [f"C{2 * n}H{6 * n + 2}O{n + 1}Si{n}" for n in range(1, n_max + 1)]


def _known_species(polarity: str = "negative") -> dict:
    # The known-species privilege is reagent/polarity-specific. The lists below
    # are NEGATIVE-mode (Br/halide-CIMS): small atmospheric acids/radicals seen
    # as Br- adducts, [M-H]- nitroaromatics, and the silanediol [M+Br]-/[M-H]-
    # contaminant series. In POSITIVE mode (urea-CIMS) none of these apply -- the
    # N-base / oxygenated-VOC analytes are reachable by the organic grid, and the
    # silanediol series would be scored under the wrong (anion) ion form -- so
    # pass 0 is a no-op and the grid + pass-3 families carry the sample.
    if str(polarity) == "positive":
        # Positive (urea-CIMS): the N-base / oxygenated-VOC analytes are reachable
        # by the organic grid, so most of pass 0 is a no-op. EXCEPTION:
        # ORGANOPHOSPHATE esters / phosphine oxides -- ubiquitous lab & indoor
        # contaminants that ionise well as [M+H]+ / [M+(urea)H]+ but are INVISIBLE
        # to the CHNOS grid (P is off by default; opening the P grid floods mass-
        # degeneracy with P2/P3, N-rich monsters). Supply them as explicit known
        # formulas. P is monoisotopic -> no isotope twin to confirm, so the commit
        # is gated on CROSS-CHANNEL corroboration (>=2 ion channels) in pass 0.
        organophosphate = {
            "C6H15O4P": "triethyl phosphate (TEP)",
            "C9H21O4P": "tripropyl phosphate (TPrP)",
            "C12H27O4P": "tri-n-butyl phosphate (TBP / TiBP)",
            "C18H15O4P": "triphenyl phosphate (TPhP)",
            "C18H15OP": "triphenylphosphine oxide (TPPO)",
            "C18H39O7P": "tris(2-butoxyethyl) phosphate (TBEP)",
            "C24H51O4P": "tris(2-ethylhexyl) phosphate (TEHP)",
            "C21H21O4P": "tricresyl phosphate (TMPP / TCrP)",
        }
        # ORGANOTHIOPHOSPHATE / -DITHIOATE insecticides: agricultural
        # organophosphorus pesticides (P + 1-3 S), primary AMBIENT analytes over
        # farmland (not lab contaminants). Like the phosphate esters they are
        # invisible to the CHNOS grid (P off; and the S2/S3 count is above the
        # grid's max_S), and they ionise as [M+H]+ / [M+(urea)H]+. Unlike the
        # esters they DO carry a ³⁴S twin, so a confirmed ³⁴S envelope can stand
        # in for the 2nd channel (see the gate below). Seeded from a field
        # campaign where the malathion family was the brightest unexplained
        # nocturnal cluster (P off the grid) + common OP insecticides. Only
        # those present + on-cal + corroborated commit; listing absent ones is
        # harmless.
        organothiophosphate = {
            "C10H19O6PS2": "malathion",
            "C11H21O6PS2": "malathion homolog (+CH2)",
            "C9H17O6PS2": "malathion homolog (-CH2)",
            "C8H13O5PS2": "malathion transformation product (-C2H6O)",
            "C10H19O7PS": "malaoxon (malathion oxon)",
            "C5H12NO3PS2": "dimethoate",
            "C8H19O2PS3": "disulfoton",
            "C7H17O2PS3": "phorate",
            "C9H21O2PS3": "terbufos",
            "C10H15O3PS2": "fenthion",
            "C12H21N2O3PS": "diazinon",
            "C10H14NO5PS": "parathion",
            "C8H10NO5PS": "methyl parathion",
            "C9H12NO5PS": "fenitrothion",
            "C11H12NO4PS2": "phosmet",
            "C10H12N3O3PS2": "azinphos-methyl",
            "C9H11Cl3NO3PS": "chlorpyrifos",
            "C7H7Cl3NO3PS": "chlorpyrifos-methyl",
            # N-bearing malathion-relative dithioate observed in ambient air
            # with a [M+H]+/urea pair (m/z 444.13); an isobaric CHON-Cl2 fit
            # exists, but the >=2-channel gate + family context favour the
            # P-thioate.
            "C16H30NO7S2P": "organothiophosphate (C16 N-dithioate; Cl2 isobar)",
        }
        # AMBIENT AMMONIA. NH3 is measured in urea-CIMS via its urea adduct
        # [NH3+(CH4N2O)H]+ (m/z 78.066) -- the SAME ion as [urea+NH4]+, but the
        # meaningful neutral is the ambient NH3, not the reagent. Its other channels
        # ([M+H]+ 18.03, [M+NH4]+ 35.06) fall below the measured range, so it is a
        # single-channel known species (the >=2-channel gate is P-only). Supplying
        # it here keeps the ambient-NH3 signal as an analyte instead of discarding it
        # as a reagent cluster, and stops the CHON grid reading 78.066 as the phantom
        # `CH4N2O [M+NH4]+` (urea as its own ammonium adduct).
        ambient_inorganic = {
            "H3N": "ammonia (ambient; via urea adduct)",
        }
        return {
            "organophosphate": organophosphate,
            "organothiophosphate": organothiophosphate,
            "ambient_inorganic": ambient_inorganic,
        }
    atmos = {
        # small atmospheric acids / radicals detected as Br- adducts -- the
        # PRIMARY analytes of a Br-CIMS, all invisible to the organic grid:
        # HO2 is a radical (half-integer DBE); the rest are C0 inorganics.
        # HNO3/HNO2 confirmed AMBIENT analytes by the user (2026-06-12),
        # so they were removed from the reagent-cluster library.
        "HO2": "hydroperoxyl radical",
        "HNO3": "nitric acid",
        "HNO2": "nitrous acid",
        "HNO4": "peroxynitric acid",
    }
    # REACTIVE IODINE species -- the canonical iodide-CIMS analytes, detected as
    # [M+I]- (I2X- ions). Covalent iodine is monoisotopic and OFF the organic grid
    # (like F/P), so these must be supplied as known formulas. Safe on exact mass
    # alone (the PFCA precedent): at defect -0.19..-0.27 around m/z 270-335 no
    # grid-reachable organic exists, so the only degeneracy is other iodine
    # species. ICl/IBr additionally get their 37Cl/81Br envelope attached by the
    # oracle (both confirmed on the 2026-07-21 batch: I2Cl- 288.778 with
    # a 0.31-ratio 37Cl twin, I2Br- 332.728 with a 0.96-ratio 81Br twin). HOI and
    # INO2 are time-VARYING there (photochemical HOI 55x over 7 h) -- ambient
    # analytes, not source background, which is why they are NOT in the reagent
    # cluster library. In a non-iodide run the [M+I]- channel is absent from
    # cfg.mechanism_ids, so this family self-gates to iodide sources.
    # The oxyacids ALSO own their deprotonation lines: IO-/IO2-/IO3- are the
    # [M-H]- ions of HOI/HIO2/HIO3 (iodate IO3- is iodic acid's DOMINANT channel
    # -- the NPF tracer), which is why build_library skips the generic IOx-
    # reagent-oxide entries for iodine. OIO (the IO2 radical, a canonical daytime
    # iodine species) commits via [OIO+I]- = I2O2- (285.799). The IO radical
    # itself is NOT listed: its [IO+I]- is composition-identical to the locked
    # I2O- source cluster (see the reagents.py caveat).
    reactive_iodine = {
        "HOI": "hypoiodous acid",
        "HIO2": "iodous acid",
        "HIO3": "iodic acid",
        "IO2": "iodine dioxide radical (OIO)",
        "INO2": "iodine nitrite (nighttime I + NO2 reservoir)",
        "INO3": "iodine nitrate",
        "CNI": "iodine cyanide (ICN)",
        "CINO": "iodine isocyanate (INCO)",
        "ICl": "iodine monochloride",
        "IBr": "iodine monobromide",
    }
    # Atmospheric nitroaromatics (brown-carbon tracers from NOx + aromatic VOC /
    # biomass burning), detected as [M-H]-. These are H-POOR / high-DBE, so the
    # ambient Van Krevelen floor + DBE/C ceiling block them from the organic grid
    # -- they must be supplied as known tracers (the v45->v46 fix; dinitrophenol
    # was independently confirmed present by an Orbitool assignment). Only those
    # actually present + |ppm|<=2 commit, so listing absent ones is harmless.
    nitroaromatic = {
        "C6H4N2O5": "2,4-dinitrophenol",
        "C6H5NO3": "nitrophenol",
        "C6H5NO4": "nitrocatechol",
        "C7H6N2O5": "dinitrocresol",
    }
    contam = {
        f: "dimethylsilanediol oligomer (PDMS hydrolysis, inlet/tubing)"
        for f in _silanediol_series()
    }
    # Perfluorocarboxylic acids CnHF(2n-1)O2 -- ubiquitous environmental / lab
    # contaminants and a classic CIMS signal (TFA, PFPrA, PFBA, ... PFOA). They are
    # F-rich so the organic grid (max_F=0, or pass-4's clamped-F path) misses the
    # clean low-F ones (TFA was the brightest peak in the ¹⁵NO₃⁻ batch yet went
    # unexplained). The PFCA formula is HIGHLY specific (high negative mass defect),
    # so supply the series as known formulas; only those present + on-cal commit.
    # In a narrow high-m/z window TFA's [M-H]- (112.99) is out of range and it is
    # seen ONLY as the reagent adduct, so do NOT require the deprotonation channel.
    perfluoroacid = {
        f"C{n}HF{2 * n - 1}O2": f"perfluoro-C{n} acid (PFCA)" for n in range(2, 13)
    }
    # Chlorinated paraffins (SCCP/MCCP/LCCP): saturated CnH(2n+2-x)Clx, the
    # persistent-organic-pollutant family the user's screenshot showed (C10H17Cl5 ..
    # C14H22Cl8). Cl 3-15 is FAR above the organic grid's max_Cl (<=2), so they are
    # never enumerated -> they land in 'unexplained'. Supply the (tight, 2-parameter)
    # family as known formulas; they commit ONLY with a confirmed ³⁷Cl envelope, so
    # this is safe despite the wide n/x range. Listing absent ones is harmless.
    chlorinated_paraffin = {}
    for _n in range(10, 31):
        for _x in range(3, 16):
            _h = 2 * _n + 2 - _x
            if _h >= 1:
                chlorinated_paraffin[f"C{_n}H{_h}Cl{_x}"] = (
                    f"chlorinated paraffin C{_n}Cl{_x}"
                )
    return {
        "atmospheric": atmos,
        "reactive_iodine": reactive_iodine,
        "nitroaromatic": nitroaromatic,
        "perfluoroacid": perfluoroacid,
        "chlorinated_paraffin": chlorinated_paraffin,
        "contaminant:silanediol": contam,
    }


_D37CL = 1.9970499


_RECOVERABLE_KNOWN_FAMS = {"chlorinated_paraffin"}


def run_pass0_known(
    client,
    sample_id: str,
    ledger: pd.DataFrame,
    profile,
    cfg: PassConfig,
    adducts: list[str],
    *,
    score_fn=None,
    log=print,
) -> dict:
    """Pass 0 -- assign explicit KNOWN species (contaminant series + small
    atmospheric acids/radicals) before the organic passes run. Mascope scores
    the compositions (its isotope model covers 29Si/30Si + the reagent
    halogen); commits are LOCKED so pass 1 cannot displace them with grid CHO
    fits. Each still passes the mass gate (|ppm|<=2) and the 81Br-twin
    consistency check, so a composite collision is refused, not locked."""
    score_fn = score_fn or IO.score_candidates
    out = {"committed": 0, "locked": 0, "iso_attached": 0}
    registry = _known_species(getattr(profile, "polarity", "negative"))
    label_of = {f: (fam, lbl) for fam, d in registry.items() for f, lbl in d.items()}
    formulas = sorted(label_of)
    if not formulas:  # positive mode: pass 0 is a no-op
        log(
            f"[pass0] no known-species list for polarity="
            f"{getattr(profile, 'polarity', 'negative')!r}; skipping"
        )
        return out
    scored = score_fn(client, sample_id, formulas, mechanism_ids=cfg.mechanism_ids)
    if scored is None or len(scored) == 0:
        log("[pass0] WARNING scoring returned EMPTY for the known-species list")
        out["scoring_empty"] = True
        return out
    base = scored[
        scored["is_base"]
        & scored["sample_peak_id"].notna()
        & scored["ion_score"].notna()
    ]
    # cross-channel corroboration count (on-cal): how many distinct ion channels
    # each known formula matches. Monoisotopic-P organophosphates require >=2.
    if "mechanism_id" in base.columns:
        _onc = base[
            (pd.to_numeric(base["ppm_error"], errors="coerce") - cfg.prior_offset).abs()
            <= 2.0
        ]
        ope_channels = (
            _onc.groupby("compound_formula")["mechanism_id"].nunique().to_dict()
        )
    else:
        ope_channels = {}
    kids = scored[
        (~scored["is_base"])
        & scored["sample_peak_id"].notna()
        & (pd.to_numeric(scored["iso_score"], errors="coerce").fillna(0) > 0.4)
    ]
    # A matched DIAGNOSTIC heavy-isotope satellite (³⁴S / ³⁷Cl / ⁸¹Br) is
    # INDEPENDENT corroboration that a pure phosphate ester (monoisotopic in every
    # atom) lacks: the envelope both confirms the heteroatom count and refutes a
    # single-channel mass coincidence, so it substitutes for the >=2-channel
    # requirement below (e.g. des-ethyl malathion, seen only as [M+H]+, is ³⁴S-
    # confirmed). ¹³C is DELIBERATELY EXCLUDED -- every carbon-bearing formula has
    # a ¹³C line, so it refutes nothing and must never license an off-grid P.
    _DIAG_ISO = ("34S", "37Cl", "81Br")
    iso_confirmed = set(
        kids[kids["iso_label"].astype(str).str.contains("|".join(_DIAG_ISO), na=False)][
            "compound_formula"
        ].unique()
    )
    mzs = ledger["mz"]
    for _, r in base.iterrows():
        ppm = r["ppm_error"]
        # the known-list privilege still requires the MASS, but judged against the
        # rough offset (cfg.prior_offset) so a uniformly-shifted instrument (e.g.
        # -1.9 ppm) doesn't drop on-trend contaminants and hand the peak to an
        # off-trend CHO mass-fit (the silanediol-vs-C5H10O6 collision at -1.9 ppm).
        if ppm is None or pd.isna(ppm) or abs(float(ppm) - cfg.prior_offset) > 2.0:
            continue
        pid = r["sample_peak_id"]
        try:
            if L.role_of(ledger, pid) != L.ROLE_UNEXPLAINED:
                continue
            # self-twin consistency: a [M+Br]- contaminant claim must own a
            # consistent 81Br twin of its OWN. v25 lesson: silanediol n=1
            # (170.9482) collided with lactic acid's 81Br child (170.9485);
            # the 12k cps peak's twin at 172.946 was 427 cps (ratio 0.04) --
            # the peak belongs to the lactic-acid envelope, not the
            # contaminant. A composite minor component cannot be LOCKED.
            if "Br" in str(r["ion_formula"]):
                i0 = ledger.index[ledger["peak_id"] == pid][0]
                m0 = float(ledger.at[i0, "mz"])
                h0 = float(ledger.at[i0, "height"])
                tw = _peak_near(mzs, m0 + _DBR, ppm=8.0)
                rt = (float(ledger.at[tw, "height"]) / h0) if tw is not None else 0.0
                if not (0.5 <= rt <= 1.7):
                    log(
                        f"[pass0] skip {r['compound_formula']} @{m0:.4f}: "
                        f"own-81Br-twin ratio {rt:.2f} inconsistent "
                        f"(composite or wrong claim)"
                    )
                    continue
            fam, lbl = label_of[r["compound_formula"]]
            # organophosphates are monoisotopic in P -> require >=2 ion channels
            # (e.g. [M+H]+ AND [M+(urea)H]+) before locking, since there is no
            # isotope twin to confirm a single-channel mass coincidence. The one
            # accepted substitute for the 2nd channel is a confirmed DIAGNOSTIC
            # heavy-isotope envelope (³⁴S / ³⁷Cl / ⁸¹Br) -- independent evidence a
            # phosphate ester cannot forge. Generic across families (a chlorinated
            # thiophosphate qualifies via ³⁷Cl, not just malathion via ³⁴S); ¹³C is
            # excluded upstream in `iso_confirmed`.
            _iso_ok = r["compound_formula"] in iso_confirmed
            if (
                fam in ("organophosphate", "organothiophosphate")
                and ope_channels.get(r["compound_formula"], 0) < 2
                and not _iso_ok
            ):
                log(
                    f"[pass0] skip {r['compound_formula']} @{float(r['sample_peak_mz']):.4f}: "
                    f"single ion channel, no diagnostic-isotope (³⁴S/³⁷Cl/⁸¹Br) "
                    f"envelope (P needs >=2 channels or an isotope twin to corroborate)"
                )
                continue
            tag = (
                "atmospheric"
                if fam == "atmospheric"
                else "reactive-iodine"
                if fam == "reactive_iodine"
                else "nitroaromatic"
                if fam == "nitroaromatic"
                else "organophosphate"
                if fam == "organophosphate"
                else "organothiophosphate"
                if fam == "organothiophosphate"
                else "perfluoroacid"
                if fam == "perfluoroacid"
                else "chlorinated-paraffin"
                if fam == "chlorinated_paraffin"
                else "ambient"
                if fam == "ambient_inorganic"
                else "contaminant"
            )
            fam_kids = kids[kids["compound_formula"] == r["compound_formula"]]
            n_kids = int((fam_kids["sample_peak_id"] != pid).sum())
            # chlorinated paraffins (Cl is off the organic grid at Cl>2): commit ONLY
            # when the ³⁷Cl envelope is confirmed (>=2 matched ³⁷Cl satellites), so a
            # CnHmClx mass coincidence is rejected. Cl IS isotope-confirmable (unlike
            # monoisotopic F/P), and the server's aggregate score is artificially low
            # for ¹⁵N-labelled poly-Cl (¹⁴N phantoms + wide envelope), so we lock on
            # the isotope evidence instead of the depressed compound_score.
            if fam == "chlorinated_paraffin" and n_kids < 2:
                log(
                    f"[pass0] skip {r['compound_formula']} @{float(r['sample_peak_mz']):.4f}: "
                    f"³⁷Cl envelope not confirmed (n_kids={n_kids})"
                )
                continue
            # silanediol / any Si-rich known species: the 29Si M+1 must MATCH the Si
            # count, not merely exist. A high-O organic is mass-degenerate with a Si_k
            # oligomer; if the M+1 is too small the Si is over-claimed -> skip, leaving
            # the peak for the organic grid (the C10H18O11 vs C8H26O5Si4 case @393).
            _c0 = C.parse_formula(r["compound_formula"])
            if _c0.get("Si", 0) > 0:
                _ix = ledger.index[ledger["peak_id"] == pid]
                _m0h = (
                    float(ledger.at[_ix[0], "height"])
                    if len(_ix) and pd.notna(ledger.at[_ix[0], "height"])
                    else 0.0
                )
                if not _si_m1_consistent(
                    ledger,
                    float(r["sample_peak_mz"]),
                    _m0h,
                    _c0.get("Si", 0),
                    _c0.get("C", 0),
                ):
                    log(
                        f"[pass0] skip {r['compound_formula']} @{float(r['sample_peak_mz']):.4f}: "
                        f"29Si M+1 too small for Si{_c0.get('Si', 0)} (over-claimed; likely "
                        "a high-O organic) -- left for the grid"
                    )
                    out["si_underclaimed"] = out.get("si_underclaimed", 0) + 1
                    continue
            conf = (
                f"Good ({tag})"
                if float(r["ion_score"]) >= 0.7 or n_kids >= 2
                else f"Low ({tag})"
            )
            L.commit_assignment(
                ledger,
                pid,
                neutral_formula=r["compound_formula"],
                adduct=_mech_to_adduct(r),
                ion_formula=r["ion_formula"],
                ion_score=float(r["ion_score"]),
                compound_score=_f(r.get("compound_score")),
                ppm_error=float(ppm),
                pass_no=0,
                method=f"known:{fam}",
                confidence=conf,
                commentary=(
                    f"Pass 0 (known {tag}): {r['compound_formula']} "
                    f"{_mech_to_adduct(r)} = {lbl}, ppm "
                    f"{float(ppm):.2f}, ion score {float(r['ion_score']):.2f}"
                    + (
                        "; excluded from the organic grid (radical / C0 inorganic)"
                        if fam == "atmospheric"
                        else "; H-poor nitroaromatic blocked by the ambient "
                        "VK floor/DBE ceiling -- assigned as a known BrC tracer"
                        if fam == "nitroaromatic"
                        else "; organophosphate contaminant (P off the grid); "
                        f"corroborated across {ope_channels.get(r['compound_formula'], 0)} "
                        "ion channels (monoisotopic P, no isotope twin)"
                        if fam == "organophosphate"
                        else "; organothiophosphate/-dithioate pesticide (P off the "
                        "grid, S above grid max); corroborated by "
                        + (
                            f"{ope_channels.get(r['compound_formula'], 0)} ion channels"
                            if ope_channels.get(r["compound_formula"], 0) >= 2
                            else "a confirmed diagnostic-isotope (³⁴S/³⁷Cl/⁸¹Br) "
                            "envelope (single channel)"
                        )
                        if fam == "organothiophosphate"
                        else "; perfluorocarboxylic acid (F off the grid); "
                        "known PFCA series formula, exact-mass committed"
                        if fam == "perfluoroacid"
                        else "; chlorinated paraffin (Cl off the grid); ³⁷Cl "
                        f"envelope confirmed ({n_kids} satellites), isotope-locked"
                        if fam == "chlorinated_paraffin"
                        else ""
                    )
                ),
            )
            out["committed"] += 1
            L.lock_peaks(ledger, [pid])
            out["locked"] += 1
            for _, k in fam_kids.iterrows():
                if k["sample_peak_id"] == pid:
                    continue
                try:
                    L.attach_isotopologue(
                        ledger,
                        k["sample_peak_id"],
                        pid,
                        iso_label=k["iso_label"],
                        iso_match_score=_f(k["iso_score"]),
                    )
                    out["iso_attached"] += 1
                except L.LedgerError:
                    continue
        except L.LedgerError:
            continue
    # isotope-confirmed RECOVERY of known species the server scored too low to
    # anchor (e.g. ¹⁵N-labelled chlorinated paraffins, whose aggregate score
    # collapses so the base ion comes back UNANCHORED and the main loop above --
    # which iterates only server-anchored bases -- never sees it).
    rec = _recover_isotope_locked_known(ledger, scored, label_of, cfg, log=log)
    for k, v in rec.items():
        out[k] = out.get(k, 0) + v
    log(f"[pass0] {out}")
    return out


def _recover_isotope_locked_known(
    ledger: pd.DataFrame,
    scored: pd.DataFrame,
    label_of: dict,
    cfg: PassConfig,
    *,
    anchor_tol: float = 2.0,
    sat_ppm: float = 7.0,
    min_sats: int = 2,
    height_floor: float = 20.0,
    log=print,
) -> dict:
    """Recover an isotope-confirmable KNOWN species (Cl/Br/S family) whose server
    compound_score was too low to anchor a real peak.

    For ¹⁵N-labelled poly-Cl the server's aggregate match_score collapses (the
    ¹⁴N phantom lines + the wide Cl envelope drag it under possible_match_
    threshold), so ``match_compounds`` returns the base ion UNANCHORED
    (sample_peak_id / ppm NaN). ``run_pass0_known``'s main loop only iterates
    server-anchored bases, so those congeners never reach the ³⁷Cl confirmation
    and are left unexplained -- exactly the "too low score on the server" miss.

    Re-anchor against the LEDGER by exact mass (the server's theoretical M0 mz)
    and commit ONLY when BOTH hold:

      * a real, still-unexplained ledger peak sits within ``anchor_tol`` ppm of
        the theoretical M0 mass (offset-aware, like the main loop), AND
      * >= ``min_sats`` ³⁷Cl satellites (M0 + k·Δ³⁷Cl) are present in the ledger.

    This is the SAME isotope evidence the committed congeners passed; only the
    server's depressed aggregate score is bypassed. It cannot fabricate: no real
    peak, or no ³⁷Cl envelope, means no commit. Returns counter deltas
    (committed / locked / iso_attached / recovered) for the caller to fold in."""
    out = {"committed": 0, "locked": 0, "iso_attached": 0, "recovered": 0}
    if scored is None or len(scored) == 0 or "is_base" not in scored.columns:
        return out
    fam_of = {f: fam for f, (fam, _lbl) in label_of.items()}
    is_base = scored["is_base"].fillna(False)
    recoverable = scored["compound_formula"].map(
        lambda f: fam_of.get(f) in _RECOVERABLE_KNOWN_FAMS
    )
    bases = scored[is_base & recoverable]
    if not len(bases):
        return out
    mzs = ledger["mz"]
    for _, r in bases.iterrows():
        cf = r["compound_formula"]
        theo = r.get("theo_mz")
        if theo is None or pd.isna(theo):
            continue
        theo = float(theo)
        # offset-aware exact-mass anchor onto a still-unexplained ledger peak
        i = _peak_near(mzs, theo * (1 + cfg.prior_offset * 1e-6), ppm=anchor_tol)
        if i is None:
            continue
        pid = ledger.at[i, "peak_id"]
        try:
            if L.role_of(ledger, pid) != L.ROLE_UNEXPLAINED:
                continue
        except L.LedgerError:
            continue
        bh = ledger.at[i, "height"]
        if pd.isna(bh) or float(bh) < height_floor:
            continue
        led_ppm = (float(ledger.at[i, "mz"]) - theo) / theo * 1e6
        # ³⁷Cl envelope confirmation against the LEDGER (NOT the server iso_score,
        # which is itself depressed). Server satellite theo_mz preferred, with the
        # M0 + k·Δ³⁷Cl ladder as a fallback; count DISTINCT, still-unexplained
        # ledger peaks so a server/ladder mz that resolve to one peak count once.
        sib = scored[
            (scored["compound_formula"] == cf)
            & (scored["ion_formula"] == r["ion_formula"])
            & (~scored["is_base"].fillna(False))
        ]
        sat_theos = {
            round(float(tm), 3)
            for tm in sib["theo_mz"].dropna()
            if float(tm) > theo + 0.5
        }
        sat_theos.update(round(theo + k * _D37CL, 3) for k in (1, 2, 3, 4))
        sat_pids: list = []
        for tm in sorted(sat_theos):
            j = _peak_near(mzs, tm, ppm=sat_ppm)
            if j is None:
                continue
            jh = ledger.at[j, "height"]
            if pd.isna(jh) or float(jh) < height_floor * 0.5:
                continue
            jpid = ledger.at[j, "peak_id"]
            if jpid == pid or jpid in sat_pids:
                continue
            try:
                if L.role_of(ledger, jpid) == L.ROLE_UNEXPLAINED:
                    sat_pids.append(jpid)
            except L.LedgerError:
                continue
        if len(sat_pids) < min_sats:
            continue
        fam, lbl = label_of[cf]
        sc = _f(r.get("ion_score"))
        adduct = _mech_to_adduct(r)
        L.commit_assignment(
            ledger,
            pid,
            neutral_formula=cf,
            adduct=adduct,
            ion_formula=r["ion_formula"],
            ion_score=(0.0 if sc is None else float(sc)),
            compound_score=_f(r.get("compound_score")),
            ppm_error=float(led_ppm),
            pass_no=0,
            method=f"known:{fam}",
            confidence="Good (chlorinated-paraffin, recovered)",
            commentary=(
                f"Pass 0 (known chlorinated-paraffin, RECOVERED): {cf} "
                f"{adduct} = {lbl}, ppm {led_ppm:.2f}; the server "
                f"compound_score ({_f(r.get('compound_score'))}) was too low "
                "to anchor a peak (¹⁵N-phantom / wide-envelope depression), "
                "so it was exact-mass anchored to the ledger and "
                f"isotope-locked on its ³⁷Cl envelope ({len(sat_pids)} "
                "satellites)."
            ),
        )
        out["committed"] += 1
        out["recovered"] += 1
        L.lock_peaks(ledger, [pid])
        out["locked"] += 1
        for spid in sat_pids:
            try:
                L.attach_isotopologue(ledger, spid, pid, iso_label="37Cl")
                out["iso_attached"] += 1
            except L.LedgerError:
                continue
    return out


def run_pass5_completion(
    client,
    sample_id: str,
    ledger: pd.DataFrame,
    profile,
    cfg: PassConfig,
    adducts: list[str],
    *,
    score_fn=None,
    log=print,
) -> dict:
    """Pass 5 -- known-neutral completion. Opens NO new formula space: it only
    proposes neutrals the run already believes, onto unexplained peaks that are

      (a) cross-channel partners: another registered adduct of an assigned
          neutral (e.g. the [M+Br]- of a Good [M-H]- compound, TFA's [M-H]-), or
      (b) series-gap members: a CH2-bracketed gap inside an assigned homolog
          ladder (the C2/C5/C6 hydroxy-acid ladder whose missing C3 rung,
          C3H6O3.Br- at 10.3k cps, was the biggest unexplained peak of v20).

    Mascope still scores every proposal (its ppm/isotope attribution is
    authoritative) and the normal commit gates apply; 'completion' in the
    method name grants the pattern-evidence band, since the evidence is an
    independently assigned neutral."""
    score_fn = score_fn or IO.score_candidates
    out = {"committed": 0, "locked": 0, "iso_attached": 0}
    m0 = ledger[ledger["role"] == L.ROLE_M0].dropna(subset=["neutral_formula"])
    anchors = m0[m0["confidence"].astype(str).str.startswith(("High", "Good"))]
    assigned = set(anchors["neutral_formula"])
    if not assigned:
        log("[pass5] no High/Good anchors; skipping")
        return out
    un = ledger[ledger["role"] == L.ROLE_UNEXPLAINED]
    gadducts = [a for a in adducts if a in C.ADDUCT_SHIFTS]

    def un_peak_near(target: float):
        if not len(un):
            return None
        d = (un["mz"] - target).abs()
        i = d.idxmin()
        return (
            un.at[i, "peak_id"] if d.loc[i] <= target * cfg.search_ppm * 1e-6 else None
        )

    targets: dict[str, set] = {}
    n_cross = n_gap = 0
    # (a) cross-channel partners of assigned neutrals
    for nf in assigned:
        for ad in gadducts:
            try:
                pid = un_peak_near(C.ion_mz(nf, ad))
            except Exception:
                continue
            if pid is not None:
                targets.setdefault(nf, set()).add(pid)
                n_cross += 1
    # (b) CH2-bracketed gaps between assigned ladder anchors
    for nf in sorted(assigned):
        for k in (2, 3):
            if G.formula_add(nf, "CH2", k) not in assigned:
                continue
            for j in range(1, k):
                mid = G.formula_add(nf, "CH2", j)
                if not mid or mid in assigned or not C.dbe_ok(mid)[0]:
                    continue
                for ad in gadducts:
                    pid = un_peak_near(C.ion_mz(mid, ad))
                    if pid is not None:
                        targets.setdefault(mid, set()).add(pid)
                        n_gap += 1
    if not targets:
        log("[pass5] no completion targets")
        return out
    only = set().union(*targets.values())
    log(
        f"[pass5] {len(targets)} known neutrals -> {len(only)} target peaks "
        f"({n_cross} cross-channel, {n_gap} series-gap)"
    )
    scored = score_fn(
        client, sample_id, sorted(targets), mechanism_ids=cfg.mechanism_ids
    )
    if scored is None or len(scored) == 0:
        log(
            f"[pass5] WARNING scoring returned EMPTY for {len(targets)} known "
            f"neutrals -- server likely degraded; completion skipped"
        )
        out["scoring_empty"] = True
        return out
    arb = arbitrate(scored, cfg)
    s = commit_winners(
        ledger,
        arb,
        pass_no=5,
        method="completion:known-neutral",
        context=profile.label,
        cfg=cfg,
        lock=False,
        min_raw_score=cfg.tau_suspect,
        confidence_suffix="completion",
        claim_unexplained_only=True,
        only_peaks=only,
    )
    log(f"[pass5] {s}")
    return s


def build_ranges(
    profile: X.ContextProfile,
    pre,
    *,
    include_N: bool,
    extra_elements: dict[str, tuple[int, int]] | None = None,
    o_max: int | None = None,
    c_max: int | None = None,
) -> dict[str, tuple[int, int]]:
    """Build a NEUTRAL-formula grid box.

    Pass 1/2 are CHO(N) only: heteroatoms are NOT auto-added from the (noisy)
    prescan -- they enter the neutral exclusively via `extra_elements` (Pass 3
    contaminant families). This is what prevents the [M+Br]- alias from being
    mis-read as a brominated neutral: in a Br-CIMS sample the Br lives in the
    ADDUCT, not the neutral. The prescan only caps C here.

    The box width defaults to the context's grid_c_max / grid_o_max (40 / 30 for
    the ambient Br-CIMS profiles; wider for a heavier positive source like
    urea-CIMS). An explicit c_max / o_max argument overrides the profile.
    """
    if o_max is None:
        o_max = getattr(profile, "grid_o_max", 30)
    if c_max is None:
        c_max = getattr(profile, "grid_c_max", 40)
    cmax = c_max
    if pre is not None and getattr(pre, "estimated_max_C", 0):
        cmax = min(c_max, max(12, pre.estimated_max_C + 4))
    r = {
        "C": (0, cmax),
        "H": (0, cmax * 2 + 4),
        "O": (0, o_max),
        "N": (0, profile.max_N if include_N else 0),
        "S": (0, 0),
        "P": (0, 0),
        "Si": (0, 0),
        "F": (0, 0),
        "Cl": (0, 0),
        "Br": (0, 0),
        "I": (0, 0),
    }
    if extra_elements:
        for el, (lo, hi) in extra_elements.items():
            cap = getattr(profile, f"max_{el}", hi) or hi
            r[el] = (lo, min(hi, cap))
    return r


def ranges_to_string(r: dict[str, tuple[int, int]]) -> str:
    return " ".join(f"{el}{lo}-{hi}" for el, (lo, hi) in r.items() if hi > 0)


def _resolve_hx_clusters(
    client,
    sample_id: str,
    ledger: pd.DataFrame,
    profile,
    cfg: PassConfig,
    reagent: str,
    hx: str,
    *,
    log=print,
) -> dict:
    """Explain unassigned peaks as anchor.HX clusters (Y.HBr.Br- etc.).

    For each anchor Y, the cluster composition X = Y+HX under [M+X]- is scored
    by Mascope (identical ion to the covalent alias), but committed with
    neutral = Y and adduct = '[M+HX+X]-' so the target list reports the real
    analyte, with commentary naming the cluster interpretation."""
    out = {"committed": 0, "locked": 0, "iso_attached": 0, "claimed_formulas": set()}
    anchors = ledger.loc[ledger["role"] == L.ROLE_M0, ["peak_id", "neutral_formula"]]
    anchor_by_formula = dict(zip(anchors["neutral_formula"], anchors["peak_id"]))
    if not anchor_by_formula:
        return out
    adduct = f"[M+{reagent}]-"
    if adduct not in C.ADDUCT_SHIFTS:
        return out
    # cluster bases: anchors plus their +/-CH2 homologs (GKA-validated bridge --
    # e.g. glutaric sits between anchored succinic and adipic acids).
    ys: dict[str, tuple[object, str]] = {
        y: (apid, "anchor") for y, apid in anchor_by_formula.items()
    }
    for y, apid in anchor_by_formula.items():
        for s in (+1, -1):
            y2 = G.formula_add(y, "CH2", s)
            if y2 and y2 not in ys:
                keep, _ = X.filter_by_context(y2, profile.label)
                if keep:
                    ys[y2] = (apid, f"homolog of anchor {y} ({s:+d}CH2)")
    tgt = _target_peaks(ledger, cfg)
    # propose: peak mz near ANY isotopologue line of Y+HX under [M+X]- is fine;
    # at minimum the base line within series ppm
    proposals: dict[str, tuple[str, object, str]] = {}  # X -> (Y, anchor_pid, note)
    tmz = sorted(tgt["mz"].tolist())
    import bisect as _bs

    for y, (apid, note) in ys.items():
        x = G.formula_add(y, hx, +1)
        if not x:
            continue
        theo = C.ion_mz(x, adduct)
        # accept proposal if a target peak sits at the base line OR the +2
        # heavy-isotope line (Br2 envelopes often have the base line weak)
        for line in (theo, theo + 1.99795):
            tol = line * cfg.series_ppm * 1e-6
            j = _bs.bisect_left(tmz, line - tol)
            if j < len(tmz) and tmz[j] <= line + tol:
                proposals[x] = (y, apid, note)
                break
    if not proposals:
        return out
    scored = IO.score_candidates(
        client, sample_id, sorted(proposals), mechanism_ids=cfg.mechanism_ids
    )
    if len(scored) == 0:
        return out
    n_committed = 0
    n_iso = 0
    want_x = scored["compound_formula"].isin(proposals)
    for (x, ion_f), grp in scored[want_x].groupby(["compound_formula", "ion_formula"]):
        # only the CLUSTER ion form (reagent count = neutral's + 1 from adduct)
        if (
            C.parse_formula(ion_f).get(reagent, 0)
            != C.parse_formula(x).get(reagent, 0) + 1
        ):
            continue
        y, apid, note = proposals[x]
        brow = grp[grp["is_base"]].iloc[0] if grp["is_base"].any() else None
        if brow is None:
            continue
        ion_score = brow["ion_score"]
        if pd.isna(ion_score) or float(ion_score) < cfg.series_min_score:
            continue
        attributed_iso = grp[
            (~grp["is_base"])
            & grp["sample_peak_id"].notna()
            & (pd.to_numeric(grp["iso_score"], errors="coerce").fillna(0) > 0.4)
        ]
        # target peak: base-line attribution preferred; for split Br2 envelopes
        # fall back to the ledger peak at the base line's theoretical m/z,
        # requiring at least one Mascope-attributed heavy isotopologue.
        pid = brow["sample_peak_id"]
        ppm_err = _f(brow["ppm_error"])
        envelope_note = ""
        if pid is None or pd.isna(pid):
            if len(attributed_iso) == 0:
                continue
            theo0 = float(brow["theo_mz"])
            tol = theo0 * cfg.series_ppm * 1e-6
            cand = tgt[(tgt["mz"] - theo0).abs() <= tol]
            if len(cand) == 0:
                continue
            pid = cand.sort_values("height", ascending=False)["peak_id"].iloc[0]
            ppm_err = float((cand["mz"].iloc[0] - theo0) / theo0 * 1e6)
            envelope_note = (
                " Base isotopologue attribution recovered from the "
                "heavy-isotope line (split halogen envelope)."
            )
        try:
            if L.is_locked(ledger, pid) or L.role_of(ledger, pid) != L.ROLE_UNEXPLAINED:
                continue
            score = float(ion_score)
            conf = confidence_label(
                score, ppm_err, len(attributed_iso), False, cfg, suffix=f"{hx}-cluster"
            )
            if conf == "Reject":
                continue
            L.commit_assignment(
                ledger,
                pid,
                neutral_formula=y,
                adduct=f"[M+{hx}+{reagent}]-",
                ion_formula=ion_f,
                ion_score=score,
                compound_score=_f(brow["compound_score"]),
                ppm_error=ppm_err,
                pass_no=3,
                method=f"cluster:{hx}",
                confidence=conf,
                commentary=(
                    f"Pass 3 (cluster): {hx} cluster of {y} ({note}, "
                    f"ref peak {apid}); ion {ion_f} scored {score:.2f} "
                    f"by Mascope. Composition identical to covalent {x};"
                    f" cluster reading preferred.{envelope_note}"
                ),
                anchor_peak_id=apid,
                series_unit=hx,
            )
            out["claimed_formulas"].add(x)
            n_committed += 1
            for _, k in attributed_iso.iterrows():
                kp = k["sample_peak_id"]
                if kp == pid:
                    continue
                try:
                    L.attach_isotopologue(
                        ledger,
                        kp,
                        pid,
                        iso_label=k["iso_label"],
                        iso_match_score=_f(k["iso_score"]),
                    )
                    n_iso += 1
                except L.LedgerError:
                    continue
        except L.LedgerError:
            continue
    out["committed"] = n_committed
    out["iso_attached"] = n_iso
    log(f"[pass3:cluster-{hx}] {{'committed': {n_committed}, 'iso_attached': {n_iso}}}")
    return out


def _resolve_acid_i2_clusters(
    client,
    sample_id: str,
    ledger: pd.DataFrame,
    profile,
    cfg: PassConfig,
    *,
    score_fn=None,
    log=print,
) -> dict:
    """Iodide CIMS: explain unassigned peaks as I2 clusters of DEPROTONATED
    acids the run already believes ([A-H+I2]- = A⁻ · I₂).

    I₂⁻· is the brightest reagent ion in an iodide source, and acids A that are
    in the ledger on [M+I]-/[M-H]- ALSO appear as the conjugate base bound to
    I2, at A - H + 2*126.9045 (e.g. [HCOOH-H+I2]- @298.807, [CH3COOH-H+I2]-
    @312.823, [HNO4-H+I2]- @331.792). On the registered channels that ion is
    only reachable as covalent (A-H+I) [M+I]- -- the off-grid organoiodine
    reading the series passes used to invent (CHIO2 et al.) -- while [M+I2]- of
    A-H is an open-shell radical (integer-DBE blocked), so after the off-grid
    fix these peaks sit in the unexplained residual. The iodide analog of
    `_resolve_hx_clusters`, with the DEPROTONATED stoichiometry: score the
    covalent alias X = A - H + I under [M+I]- (the identical ion), commit
    neutral = A, adduct = '[M-H+I2]-'.

    Anchor eligibility: M0 neutrals (plus their context-plausible +/-CH2
    homologs) with an abstractable H, O>=1 AND a C/N/S skeleton atom (acidity
    proxy -- keeps H2O and HCl out) and NO iodine (the poly-iodide space
    belongs to the reagent labeler and pass-0). Claims UNEXPLAINED peaks only,
    so the pass-0 reactive-iodine species (HOI/INO2/ICl/IBr..., committed +
    LOCKED before any of this runs) always keep their I2X- lines: the
    contested inorganic clusters (HOI2-, I2NO2-) were ruled ambient analytes
    on TIME behaviour (docs/REAGENTS.md), and this resolver must never re-read
    them -- the Br organic-acid lesson, inverted."""
    out = {"committed": 0, "locked": 0, "iso_attached": 0, "claimed_formulas": set()}
    anchors = ledger.loc[ledger["role"] == L.ROLE_M0, ["peak_id", "neutral_formula"]]
    anchor_by_formula = dict(zip(anchors["neutral_formula"], anchors["peak_id"]))
    if not anchor_by_formula:
        return out
    if "[M-H+I2]-" not in C.ADDUCT_SHIFTS or "[M+I]-" not in C.ADDUCT_SHIFTS:
        return out

    def _acid_ok(f: str) -> bool:
        cnt = C.parse_formula(f)
        return (
            cnt.get("H", 0) >= 1
            and cnt.get("O", 0) >= 1
            and any(cnt.get(e, 0) for e in ("C", "N", "S"))
            and cnt.get("I", 0) == 0
        )

    # cluster bases: acid anchors plus their +/-CH2 homologs (GKA-validated
    # bridge, as in the HBr resolver).
    ys: dict[str, tuple[object, str]] = {
        y: (apid, "anchor")
        for y, apid in anchor_by_formula.items()
        if _acid_ok(y)
    }
    for y, (apid, _n) in list(ys.items()):
        for s in (+1, -1):
            y2 = G.formula_add(y, "CH2", s)
            if y2 and y2 not in ys and y2 not in anchor_by_formula and _acid_ok(y2):
                keep, _ = X.filter_by_context(y2, profile.label)
                if keep:
                    ys[y2] = (apid, f"homolog of anchor {y} ({s:+d}CH2)")
    if not ys:
        return out
    tgt = _target_peaks(ledger, cfg)
    tmz = sorted(tgt["mz"].tolist())
    import bisect as _bs

    # propose where a target peak sits at the [A-H+I2]- base line (iodine is
    # monoisotopic: no heavy-twin fallback line, unlike the Br2 envelopes)
    proposals: dict[str, tuple[str, object, str]] = {}  # X -> (A, anchor_pid, note)
    for y, (apid, note) in ys.items():
        cnt = dict(C.parse_formula(y))
        cnt["H"] -= 1
        cnt["I"] = cnt.get("I", 0) + 1
        cnt = {k: v for k, v in cnt.items() if v > 0}
        x = C.format_formula(cnt)
        line = C.ion_mz(y, "[M-H+I2]-")   # == C.ion_mz(x, "[M+I]-")
        tol = line * cfg.series_ppm * 1e-6
        j = _bs.bisect_left(tmz, line - tol)
        if j < len(tmz) and tmz[j] <= line + tol:
            proposals[x] = (y, apid, note)
    if not proposals:
        return out
    score_fn = score_fn or IO.score_candidates
    scored = score_fn(
        client, sample_id, sorted(proposals), mechanism_ids=cfg.mechanism_ids
    )
    if scored is None or len(scored) == 0:
        return out
    n_committed = 0
    n_iso = 0
    want_x = scored["compound_formula"].isin(proposals)
    for (x, ion_f), grp in scored[want_x].groupby(["compound_formula", "ion_formula"]):
        # only the CLUSTER ion form (I count = covalent alias's + 1 from the
        # [M+I]- adduct, i.e. the neutral acid's + 2)
        if (
            C.parse_formula(ion_f).get("I", 0)
            != C.parse_formula(x).get("I", 0) + 1
        ):
            continue
        y, apid, note = proposals[x]
        brow = grp[grp["is_base"]].iloc[0] if grp["is_base"].any() else None
        if brow is None:
            continue
        ion_score = brow["ion_score"]
        if pd.isna(ion_score) or float(ion_score) < cfg.series_min_score:
            continue
        pid = brow["sample_peak_id"]
        if pid is None or pd.isna(pid):
            continue
        ppm_err = _f(brow["ppm_error"])
        attributed_iso = grp[
            (~grp["is_base"])
            & grp["sample_peak_id"].notna()
            & (pd.to_numeric(grp["iso_score"], errors="coerce").fillna(0) > 0.4)
        ]
        try:
            if L.is_locked(ledger, pid) or L.role_of(ledger, pid) != L.ROLE_UNEXPLAINED:
                continue
            score = float(ion_score)
            conf = confidence_label(
                score, ppm_err, len(attributed_iso), False, cfg, suffix="I2-cluster"
            )
            if conf == "Reject":
                continue
            L.commit_assignment(
                ledger,
                pid,
                neutral_formula=y,
                adduct="[M-H+I2]-",
                ion_formula=ion_f,
                ion_score=score,
                compound_score=_f(brow["compound_score"]),
                ppm_error=ppm_err,
                pass_no=3,
                method="cluster:I2",
                confidence=conf,
                commentary=(
                    f"Pass 3 (cluster): I2 cluster of deprotonated {y} ({note}, "
                    f"ref peak {apid}); ion {ion_f} scored {score:.2f} by "
                    f"Mascope via the covalent alias {x} [M+I]- (identical "
                    f"ion); conjugate-base . I2 reading preferred."
                ),
                anchor_peak_id=apid,
            )
            out["claimed_formulas"].add(x)
            n_committed += 1
            for _, k in attributed_iso.iterrows():
                kp = k["sample_peak_id"]
                if kp == pid:
                    continue
                try:
                    L.attach_isotopologue(
                        ledger,
                        kp,
                        pid,
                        iso_label=k["iso_label"],
                        iso_match_score=_f(k["iso_score"]),
                    )
                    n_iso += 1
                except L.LedgerError:
                    continue
        except L.LedgerError:
            continue
    out["committed"] = n_committed
    out["iso_attached"] = n_iso
    log(f"[pass3:cluster-I2] {{'committed': {n_committed}, 'iso_attached': {n_iso}}}")
    return out


def _target_peaks(ledger: pd.DataFrame, cfg: PassConfig) -> pd.DataFrame:
    un = L.unassigned_peaks(ledger)
    return un[un["height"].fillna(0) >= cfg.height_cutoff]


def _family_ok(formula: str, ranges: dict[str, tuple[int, int]]) -> bool:
    """Structural gates (integer DBE + Senior + oxygen cap) plus the family's
    element ceilings. Used for Pass-3 contaminant families, where the family's
    ranges -- not the context caps -- are the elemental authority."""
    ok, _ = C.dbe_ok(formula)
    if not ok:
        return False
    ok, _ = C.oxygen_ok(formula)
    if not ok:
        return False
    cnt = C.parse_formula(formula)
    for el, n in cnt.items():
        lo, hi = ranges.get(el, (0, 0))
        if n > hi:
            return False
    return True


def _context_filter(formulas, context: str) -> list[str]:
    """Context plausibility gate for the generic passes.

    This is also where the OFF-GRID ELEMENT invariant is enforced, via the context's
    own heteroatom caps: `ambient-air` sets max_F/max_P/max_I = 0 because those
    elements are monoisotopic (or, for P, otherwise unconfirmable), so a neutral
    containing them can never be isotope-confirmed. They may enter a neutral ONLY
    through the curated pass-0 known-species list, which does not pass through here.
    Without that, the SERIES passes extrapolate off pass-0 anchors and invent them:
    the first iodide batch produced 8 covalent-organoiodine neutrals (CHIO2,
    C2H3IO2, INO4, ...), each really an [acid-H+I2]- reagent cluster."""
    out = []
    for f in formulas:
        keep, _ = X.filter_by_context(f, context)
        if keep:
            out.append(f)
    return out


def _enumerate(
    client,
    mzs,
    mech_ids,
    ranges: dict,
    cfg: PassConfig,
    adducts: list[str],
    *,
    use_grid: bool = True,
) -> set[str]:
    """Candidate NEUTRAL formulas for these m/z. The local grid is primary
    (complete for CHO/CHON in-range, never fails); cheminfo is best-effort and
    only consulted when cfg.use_cheminfo is set."""
    formulas: set[str] = set()
    if use_grid:
        gadducts = [a for a in adducts if a in C.ADDUCT_SHIFTS]
        formulas.update(
            C.candidates_for_peaks(
                list(mzs), ranges, gadducts, ppm_tolerance=cfg.search_ppm
            )
        )
    if cfg.use_cheminfo and mech_ids:
        rng_str = ranges_to_string(ranges)
        if rng_str:
            bulk = IO.query_candidates_bulk(
                client,
                list(mzs),
                mech_ids,
                formula_ranges=rng_str,
                ppm=cfg.search_ppm,
                limit=cfg.limit_per_peak,
                workers=cfg.workers,
            )
            for cands in bulk.values():
                formulas.update(cands)
    return formulas


def _mech_ids_for(client, adducts: list[str]) -> list[str]:
    names = [IO.ADDUCT_TO_MECH[a] for a in adducts if a in IO.ADDUCT_TO_MECH]
    return list(IO.resolve_mechanism_ids(client, names).values())


def run_pass1(
    client,
    sample_id: str,
    ledger: pd.DataFrame,
    profile,
    pre,
    cfg: PassConfig,
    adducts: list[str],
    *,
    log=print,
) -> dict:
    tgt = _target_peaks(ledger, cfg)
    mzs = tgt["mz"].tolist()
    mech_ids = _mech_ids_for(client, adducts)
    log(f"[pass1] {len(mzs)} target peaks; adducts={adducts}")
    # single CHO+CHON enumeration; arbitration's complexity penalty handles the
    # CHO-before-CHON preference, so no need for two separate sub-passes.
    ranges = build_ranges(profile, pre, include_N=True)
    formulas = _enumerate(client, mzs, mech_ids, ranges, cfg, adducts)
    formulas = set(_context_filter(formulas, profile.label))
    log(f"[pass1] {len(formulas)} context-plausible CHO/CHON candidate formulas")
    scored = IO.score_candidates(
        client, sample_id, sorted(formulas), mechanism_ids=cfg.mechanism_ids
    )
    log(f"[pass1] scored rows={len(scored)}")
    arb = arbitrate(scored, cfg)
    summary = commit_winners(
        ledger,
        arb,
        pass_no=1,
        method="cheminfo+grid",
        context=profile.label,
        cfg=cfg,
        lock=True,
        min_raw_score=cfg.tau_low,
    )
    log(f"[pass1] {summary}")
    return summary


def run_pass2(
    client,
    sample_id: str,
    ledger: pd.DataFrame,
    profile,
    cfg: PassConfig,
    adducts: list[str],
    *,
    log=print,
) -> dict:
    units = tuple(G.ORGANIC_UNITS)
    if {"siloxane", "pdms"} & set(profile.pass3_families):
        units = units + ("C2H6OSi",)  # the PDMS dimethylsiloxane rung (+74.019)
    if "fluorinated" in profile.pass3_families:
        units = units + ("CF2",)
    total = {"committed": 0, "locked": 0, "iso_attached": 0}
    tried: set[str] = set()
    # Iterative GKA: each round, confirmed members (incl. last round's) act as
    # anchors, so homologous series are walked outward step by step.
    for it in range(cfg.series_max_iter):
        anchors = set(
            ledger.loc[ledger["role"] == L.ROLE_M0, "neutral_formula"].dropna()
        )
        if not anchors:
            break
        tgt = _target_peaks(ledger, cfg)
        proposals: set[str] = set()
        for mz in tgt["mz"]:
            for p in G.propose_for_peak(
                mz, anchors, adducts, units=units, ppm=cfg.series_ppm, max_steps=1
            ):
                proposals.add(p.neutral_formula)
        proposals = set(_context_filter(proposals, profile.label)) - anchors - tried
        if not proposals:
            log(f"[pass2.{it}] no new proposals; stopping")
            break
        tried |= proposals
        scored = IO.score_candidates(
            client, sample_id, sorted(proposals), mechanism_ids=cfg.mechanism_ids
        )
        arb = arbitrate(scored, cfg)
        s = commit_winners(
            ledger,
            arb,
            pass_no=2,
            method="gka-series",
            context=profile.label,
            cfg=cfg,
            lock=False,
            min_raw_score=cfg.series_min_score,
            confidence_suffix="series",
            claim_unexplained_only=True,
        )
        for k in total:
            total[k] += s[k]
        log(f"[pass2.{it}] {len(anchors)} anchors -> {len(proposals)} proposals -> {s}")
        if s["committed"] == 0:
            break
    log(f"[pass2] total {total}")
    return total


def run_pass3(
    client,
    sample_id: str,
    ledger: pd.DataFrame,
    profile,
    pre,
    cfg: PassConfig,
    adducts: list[str],
    *,
    log=print,
) -> dict:
    tgt = _target_peaks(ledger, cfg)
    if len(tgt) == 0:
        log("[pass3] nothing unassigned; skipping")
        return {"committed": 0, "locked": 0, "iso_attached": 0}
    mzs = tgt["mz"].tolist()
    total = {"committed": 0, "locked": 0, "iso_attached": 0}
    from peaky.chem import reagents as _RG

    reagent = _RG.reagent_for_adducts(adducts)

    # --- HX-cluster resolution (halide CIMS) -------------------------------
    # A peak at anchor+HX under [M+X]- is the analyte's HX cluster
    # (Y.HX.X-), NOT a new covalent organohalogen. Resolve these against the
    # anchors FIRST so the organohalogen family below never sees them.
    cluster_claimed: set[str] = set()
    if reagent in ("Br", "Cl"):
        hx = "H" + reagent
        s = _resolve_hx_clusters(
            client, sample_id, ledger, profile, cfg, reagent, hx, log=log
        )
        for k in total:
            total[k] += s[k]
        cluster_claimed = s.get("claimed_formulas", set())
    elif reagent == "I":
        # iodide: the cluster stoichiometry is DEPROTONATED (A- . I2), not the
        # Br-style intact-molecule Y.HX.X- -- see _resolve_acid_i2_clusters.
        s = _resolve_acid_i2_clusters(
            client, sample_id, ledger, profile, cfg, log=log
        )
        for k in total:
            total[k] += s[k]
        cluster_claimed = s.get("claimed_formulas", set())

    # halide-CIMS: also try covalent organohalogens. The arbitration keeps the
    # complexity prior on the reagent element (its ion isotope can't prove
    # neutral ownership), so these only win with a real score margin.
    families = list(profile.pass3_families)
    if reagent == "Br" and "bromo_organic" not in families:
        families.append("bromo_organic")
    if reagent == "Cl" and "chloro_organic" not in families:
        families.append("chloro_organic")

    # --- automatic GKA series detection (the machine 'rotating plot') -------
    # Repeat-unit structure in the residual opens the matching contaminant
    # family even when the context has it off (e.g. CF2 links -> fluorinated).
    from peaky.assignment import series_detect as SD

    evidence = SD.detect_series(
        ledger, ppm=cfg.search_ppm, min_height=cfg.height_cutoff
    )
    log(
        "[pass3] series evidence: "
        + ", ".join(
            f"{r.unit}:{r.n_links}x{r.enrichment}" + ("*" if r.significant else "")
            for r in evidence.itertuples()
        )
    )
    fam_members: dict[str, set] = {}
    for r in evidence.itertuples():
        action = r.action
        if r.significant and isinstance(action, str) and action:
            fam_members.setdefault(r.action, set()).update(
                SD.unit_members(
                    ledger, r.mass, ppm=cfg.search_ppm, min_height=cfg.height_cutoff
                )
            )
    for fam in SD.families_from_evidence(evidence):
        if fam not in families:
            families.append(fam)
            log(
                f"[pass3] GKA evidence opened family: {fam} "
                f"({len(fam_members.get(fam, []))} chain-member targets)"
            )
    anchors_now = set(
        ledger.loc[ledger["role"] == L.ROLE_M0, "neutral_formula"].dropna()
    )
    for fam_key in families:
        fam = X.CONTAMINANT_FAMILIES.get(fam_key)
        if not fam:
            continue
        try:
            ranges = build_ranges(
                profile, pre, include_N=True, extra_elements=fam["add"]
            )
            # family-specific adducts unioned with the sample's reagent adducts
            fam_adducts = list(
                dict.fromkeys(
                    [a for a in fam["adducts"] if a in C.ADDUCT_SHIFTS] + adducts
                )
            )
            mech_ids = _mech_ids_for(client, fam_adducts)
            if fam_key in fam_members:
                # CHAIN-BASED generation for evidence-opened families: grid the
                # chain HEADS only, then propagate arithmetically along each
                # detected chain. Cheaper than gridding every member, and it
                # imposes series consistency -- a member's formula must be its
                # neighbour's formula +/- the unit.
                link_ppm = max(2.0, 2 * cfg.ppm)
                formulas = set()
                for r in evidence.itertuples():
                    if (
                        r.action != fam_key
                        or not r.significant
                        or r.unit not in G.REPEAT_UNITS
                    ):
                        continue
                    for chain in SD.unit_chains(
                        ledger,
                        r.mass,
                        ppm=link_ppm,
                        min_height=cfg.height_cutoff,
                        min_len=2,
                    ):
                        heads = C.candidates_for_peaks(
                            [chain[0][1]],
                            ranges,
                            [a for a in fam_adducts if a in C.ADDUCT_SHIFTS],
                            ppm_tolerance=cfg.search_ppm,
                        )
                        for f0 in heads:
                            if not _family_ok(f0, ranges):
                                continue
                            f = f0
                            formulas.add(f)
                            for _k in range(1, len(chain)):
                                f = G.formula_add(f, r.unit, +1)
                                if f and _family_ok(f, ranges):
                                    formulas.add(f)
                                else:
                                    break
            else:
                formulas = _enumerate(client, mzs, mech_ids, ranges, cfg, fam_adducts)
            # Structural-only filtering is EARNED BY EVIDENCE: GKA-opened
            # families bypass the context caps (ambient max_F=0 would veto the
            # very candidates the detected CF2 chains justify), with the chain
            # membership + arbitration priors as the guard. Profile-default
            # families keep the full context filter -- without chain evidence,
            # dropping the ratio priors lets mass-fit junk flood in (v9/v10
            # lesson: 142 'amines').
            if fam_key in fam_members:
                formulas = {f for f in formulas if _family_ok(f, ranges)}
            else:
                formulas = set(_context_filter(formulas, profile.label))
            if fam_key in ("bromo_organic", "chloro_organic"):
                # drop covalent-X aliases of anchor.HX clusters: if stripping
                # one HX from X yields an existing anchor, the cluster reading
                # owns that composition.
                el = "Br" if fam_key == "bromo_organic" else "Cl"
                formulas = {
                    f
                    for f in formulas
                    if G.formula_add(f, "H" + el, -1) not in anchors_now
                }
                formulas -= cluster_claimed
            if not formulas:
                continue
            scored = IO.score_candidates(
                client, sample_id, sorted(formulas), mechanism_ids=cfg.mechanism_ids
            )
            arb = arbitrate(scored, cfg)
            s = commit_winners(
                ledger,
                arb,
                pass_no=3,
                method=f"contaminant:{fam_key}",
                context=profile.label,
                cfg=cfg,
                lock=False,
                min_raw_score=cfg.tau_suspect,
                confidence_suffix=fam_key,
                claim_unexplained_only=True,
                only_peaks=fam_members.get(fam_key),
            )
        except Exception as e:
            log(f"[pass3:{fam_key}] FAILED: {type(e).__name__}: {e}")
            continue
        for k in total:
            total[k] += s[k]
        log(f"[pass3:{fam_key}] {s}")
    log(f"[pass3] total {total}")
    total["series_evidence"] = evidence.to_dict("records")
    return total


# ---------------------------------------------------------------------------
# Certified-neutral discovery (the pass-5 INVERSE): multi-channel / cluster-
# ladder convergence licenses off-grid formula space for UNEXPLAINED peaks.
# Pure algorithm in peaky.assignment.certified_neutral; this director wires it
# to the ledger + oracle. Ground truth: the NBBS urea ladder (214/274/334 ->
# core 213.0823 = C10H15NO2S) and cross-channel malathion (C10H19O6PS2).
# ---------------------------------------------------------------------------
_CERT_MAX_CANDIDATES = 24   # a certificate whose expanded box yields more is
#                             mass-saturated -> skip (degeneracy guard)
_CERT_ENUM_TOL_MDA = 2.0    # candidate-enumeration window around the certified
#                             core (tighter than the 3-mDa convergence gate: the
#                             weighted core is sub-mDa when the channels agree)
_CERT_DIAG_ISO = ("34S", "37Cl", "81Br")   # 13C excluded -- refutes nothing


def run_pass_certified(
    client,
    sample_id: str,
    ledger: pd.DataFrame,
    profile,
    cfg: PassConfig,
    adducts: list[str],
    *,
    reagent: str | None = None,
    ts_peaks=None,
    score_fn=None,
    log=print,
) -> dict:
    """Pass 7 -- certified-neutral discovery over the unexplained residual.

    Find groups of unexplained peaks whose back-calculated neutral masses
    CONVERGE across >=2 distinct ion channels (adducts and/or reagent-cluster
    ladder rungs); for each certified core mass enumerate the EXPANDED element
    box (P / S / Cl open -- the per-peak grid keeps them shut because a lone
    mass cannot earn a monoisotopic heteroatom); score through the oracle;
    commit the winning formula onto EVERY member peak (same neutral, each under
    its own channel label) so the tier engine's cross-channel corroboration
    sees the certificate. S/Cl/Br winners additionally want their diagnostic
    isotope envelope; a matched one earns Good confidence.

    ts_peaks is OPTIONAL (a batch may not include the reagent mass range, and a
    single-sample run has no TS at all): when provided, member-channel time
    co-variation is annotated as extra corroboration; when None the pass is
    fully functional on the single-spectrum mass domain.
    """
    from peaky.assignment import certified_neutral as CN

    score_fn = score_fn or IO.score_candidates
    out = {"certificates": 0, "committed": 0, "peaks_claimed": 0,
           "rungs_committed": 0, "iso_attached": 0, "skipped_saturated": 0,
           "displaced": 0, "corroborated_existing": 0}
    un = L.unassigned_peaks(ledger)
    # ALSO interrogate weak M0s: an uncorroborated Low/Suspect or near-tie
    # commit (e.g. a single-channel [M+Na]+ mass fit on an instrument with no
    # Na source) is weaker evidence than a multi-channel certificate, so it
    # must not shield its peak from re-reading. Locked and Good/High-confidence
    # owners are NOT interrogated.
    m0 = ledger[(ledger["role"] == L.ROLE_M0) & (~ledger["locked"].astype(bool))]
    _conf = m0["confidence"].astype(str)
    weak = m0[_conf.str.startswith("Low") | _conf.str.startswith("Suspect")
              | (m0["tied"].fillna(False).astype(bool))]
    weak_ids = set(weak["peak_id"])
    interro = pd.concat([un, weak], axis=0)
    if len(interro) < 2:
        return out
    offsets = CN.channel_offsets(adducts, reagent)
    if len(offsets) < 2:
        log("[pass7] <2 distinct channels for this profile; skipping")
        return out
    certs = CN.find_certificates(
        interro[["peak_id", "mz", "height"]].reset_index(drop=True), offsets)
    if not certs:
        log("[pass7] no multi-channel certificates in the residual")
        return out

    # enumerate expanded-box candidates per certificate (off-grid P/S/Cl only:
    # plain CHON near the core is pass-1's territory and already lost there)
    cert_formulas: dict[int, list[str]] = {}
    all_formulas: set[str] = set()
    for ci, cert in enumerate(certs):
        forms = CN.enumerate_certified(cert.core_mass, profile,
                                       tol_mda=_CERT_ENUM_TOL_MDA, force=True)
        forms = [f for f in forms
                 if any(C.parse_formula(f).get(el, 0) > 0 for el in ("P", "S", "Cl"))]
        if not forms:
            continue
        if len(forms) > _CERT_MAX_CANDIDATES:
            out["skipped_saturated"] += 1
            continue
        cert_formulas[ci] = forms
        all_formulas |= set(forms)
    if not all_formulas:
        log(f"[pass7] {len(certs)} certificates, none with off-grid candidates")
        return out
    out["certificates"] = len(cert_formulas)
    log(f"[pass7] {len(certs)} certificates -> {len(cert_formulas)} with "
        f"{len(all_formulas)} off-grid candidate formulas")

    scored = score_fn(client, sample_id, sorted(all_formulas),
                      mechanism_ids=cfg.mechanism_ids)
    if scored is None or len(scored) == 0:
        log("[pass7] WARNING scoring returned EMPTY for certified candidates")
        return out
    base = scored[scored["is_base"]
                  & scored["sample_peak_id"].notna()
                  & scored["ion_score"].notna()]
    kids = scored[(~scored["is_base"])
                  & scored["sample_peak_id"].notna()
                  & (pd.to_numeric(scored["iso_score"], errors="coerce").fillna(0) > 0.4)]

    for ci, cert in enumerate(certs):
        forms = cert_formulas.get(ci)
        if not forms:
            continue
        member_ids = {h.peak_id for h in cert.hits}
        # rank candidates: server-anchored member coverage, then complexity-
        # penalised mean score (the arbitrate() spirit, per-certificate)
        ranked = []
        for f in forms:
            rows = base[(base["compound_formula"] == f)
                        & (base["sample_peak_id"].isin(member_ids))]
            ok_rows = rows[
                (pd.to_numeric(rows["ppm_error"], errors="coerce")
                 - cfg.prior_offset).abs() <= 2.0]
            if not len(ok_rows):
                continue
            eff = (float(ok_rows["ion_score"].mean())
                   - C.complexity_penalty(f))
            ranked.append((len(set(ok_rows["sample_peak_id"])), eff, f, ok_rows))
        ranked.sort(key=lambda t: (-t[0], -t[1], t[2]))
        if not ranked or ranked[0][0] < 2:
            continue   # the oracle must anchor >=2 member channels
        n_anchored, eff, winner, win_rows = ranked[0]
        tied = len(ranked) > 1 and ranked[1][0] == n_anchored and (eff - ranked[1][1]) < 0.02
        # diagnostic-isotope gate for isotope-confirmable winners (13C never counts)
        wf = C.parse_formula(winner)
        wants_iso = any(wf.get(el, 0) > 0 for el in ("S", "Cl", "Br"))
        win_kids = kids[kids["compound_formula"] == winner]
        iso_ok = bool(win_kids["iso_label"].astype(str).str.contains(
            "|".join(_CERT_DIAG_ISO), na=False).any())
        # optional TS corroboration (guarded: fully optional)
        ts_note = ""
        if ts_peaks is not None and len(ts_peaks):
            cov = CN.ts_covariation(ts_peaks, [h.mz for h in cert.hits])
            if cov is not None:
                ts_note = (f"; member channels co-vary in time (r_min={cov:.2f})"
                           if cov >= 0.6 else
                           f"; member channels DO NOT co-vary (r_min={cov:.2f})")
                if cov < 0.3:
                    log(f"[pass7] skip {winner} @core {cert.core_mass:.4f}: "
                        f"channels anti-correlated in time (r_min={cov:.2f})")
                    continue
        n_corr = CN.corroboration_count(
            cert.n_channels, iso_confirmed=iso_ok,
            ts_covaries=bool(ts_note and "co-vary in time" in ts_note))
        conf = ("Good (certified)"
                if (iso_ok or (not wants_iso and cert.n_channels >= 3)) and not tied
                else "Low (certified)")
        cert_desc = (f"CERTIFIED core {cert.core_mass:.4f} by {cert.n_channels} "
                     f"channels (spread {cert.spread_mda:.2f} mDa): "
                     + ", ".join(f"{h.mz:.4f}[{h.adduct}"
                                 + (f"+{h.cluster_order}R]" if h.cluster_order else "]")
                                 for h in cert.hits))
        anchored_by_pid = {r["sample_peak_id"]: r for _, r in win_rows.iterrows()}
        committed_any = False
        strong_cert = iso_ok or cert.n_channels >= 3
        displaced_note: dict = {}
        for h in cert.hits:
            try:
                role_now = L.role_of(ledger, h.peak_id)
                if role_now == L.ROLE_M0 and h.peak_id in weak_ids:
                    row_i = ledger.index[ledger["peak_id"] == h.peak_id][0]
                    inc_formula = str(ledger.at[row_i, "neutral_formula"])
                    inc_adduct = str(ledger.at[row_i, "adduct"])
                    if inc_formula == winner:
                        # the certificate CONFIRMS the incumbent -- corroboration,
                        # not displacement (the tier engine sees the channels)
                        out["corroborated_existing"] += 1
                        continue
                    if not strong_cert:
                        continue   # a 2-channel no-iso cert is not stronger evidence
                    L.clear_assignment(
                        ledger, h.peak_id,
                        reason=f"displaced by certified-neutral {winner}: "
                               f"{cert.n_channels}-channel certificate"
                               + ("+isotope envelope" if iso_ok else "")
                               + f" outweighs weak {inc_formula} {inc_adduct}")
                    displaced_note[h.peak_id] = (
                        f"; DISPLACED weak incumbent {inc_formula} {inc_adduct}")
                    out["displaced"] += 1
                elif role_now != L.ROLE_UNEXPLAINED:
                    continue
                r = anchored_by_pid.get(h.peak_id)
                if r is not None:
                    L.commit_assignment(
                        ledger, h.peak_id,
                        neutral_formula=winner,
                        adduct=_mech_to_adduct(r),
                        ion_formula=r["ion_formula"],
                        ion_score=float(r["ion_score"]),
                        compound_score=_f(r.get("compound_score")),
                        ppm_error=_f(r.get("ppm_error")),
                        tied=tied,
                        pass_no=7,
                        method="certified:multi-channel",
                        confidence=conf,
                        commentary=(f"Pass 7 (certified-neutral): {winner} -- {cert_desc}"
                                    f"; {n_corr} corroborations"
                                    + ("; diagnostic isotope envelope confirmed" if iso_ok else "")
                                    + ts_note
                                    + displaced_note.get(h.peak_id, "")),
                    )
                else:
                    # a ladder rung above the oracle's registered channels
                    # ([M+2R+H]+ ...): committed on the certificate evidence,
                    # with the rung's OWN ppm vs the certified core so the
                    # calibrated mass-gate audit can judge it.
                    off = next(o for a, n, o in offsets
                               if a == h.adduct and n == h.cluster_order)
                    theo = cert.core_mass + off
                    rung_ppm = (h.mz - theo) / theo * 1e6
                    rung_adduct = (h.adduct if h.cluster_order == 0 else
                                   f"[M+{h.cluster_order}R{h.adduct[2:]}")
                    L.commit_assignment(
                        ledger, h.peak_id,
                        neutral_formula=winner,
                        adduct=rung_adduct,
                        ion_score=float(win_rows["ion_score"].min()),
                        ppm_error=float(rung_ppm),
                        tied=tied,
                        pass_no=7,
                        method="certified:ladder-rung",
                        confidence=conf,
                        commentary=(f"Pass 7 (certified-neutral, ladder rung "
                                    f"n={h.cluster_order}): {winner} -- {cert_desc}"
                                    + ts_note
                                    + displaced_note.get(h.peak_id, "")),
                    )
                    out["rungs_committed"] += 1
                out["peaks_claimed"] += 1
                committed_any = True
                for _, k in kids[(kids["compound_formula"] == winner)
                                 & (kids["sample_peak_id"] != h.peak_id)].iterrows():
                    try:
                        L.attach_isotopologue(ledger, k["sample_peak_id"], h.peak_id,
                                              iso_label=k["iso_label"],
                                              iso_match_score=_f(k["iso_score"]))
                        out["iso_attached"] += 1
                    except L.LedgerError:
                        continue
            except L.LedgerError:
                continue
        if committed_any:
            out["committed"] += 1
            log(f"[pass7] {winner} committed on {cert.n_channels} channels "
                f"@core {cert.core_mass:.4f} ({conf}"
                + (", iso-confirmed" if iso_ok else "") + ")")
    log(f"[pass7] {out}")
    return out
