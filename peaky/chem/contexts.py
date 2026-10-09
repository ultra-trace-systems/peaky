"""Experimental-context module.

A context bundles everything that depends on *where the sample came from* rather
than on chemistry-in-the-abstract:

  * plausibility bounds (Van Krevelen ratios, heteroatom caps)
  * the reagent / adduct system the instrument uses
  * which contaminant families Pass 3 is allowed to open
  * the class-label vocabulary used in the report

The hard structural gates (integer DBE >= 0, Senior's rule) live in chemistry.py
and are ALWAYS enforced; a context can only add further restrictions on top.

Adding a context = one CONTEXTS dict entry.
"""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field

from peaky.chem import chemistry as C

__version__ = "0.5.0"   # + NOx-skeleton readings and the C3-C4 small-acid band (run-level switches)


@dataclass(frozen=True)
class ContextProfile:
    label: str
    description: str
    # Ion-source polarity. "negative" is the Br/halide-CIMS default the pipeline
    # was built around; "positive" (urea-CIMS / APCI+) reads cation adducts
    # and turns OFF the Br-specific composite test
    # (its halogen-free-M+1 discriminator needs a halogen adduct to read the
    # co-component off the even-shift residual -- see assign.run).
    polarity: str = "negative"
    # Van Krevelen ratio windows (applied only for C >= 3)
    h_to_c: tuple = (0.0, 99.0)
    o_to_c: tuple = (0.0, 99.0)
    n_to_c: tuple = (0.0, 99.0)
    dbe_to_c: tuple = (0.0, 99.0)
    # heteroatom caps
    max_N: int = 99
    max_S: int = 99
    max_P: int = 99
    max_F: int = 99
    max_Cl: int = 99
    max_Br: int = 99
    max_I: int = 99
    max_Si: int = 99
    # NEUTRAL-formula grid box width (build_ranges). The default 40/30 matches the
    # ambient Br-CIMS box; a heavier-mass source (urea-CIMS reaches ~730 Da, so
    # neutrals ~670 Da) widens C so the >500 Da peaks get candidates.
    grid_c_max: int = 40
    grid_o_max: int = 30
    # a halogen / Si in a NEUTRAL needs a minimum carbon scaffold, else it is
    # almost always a reagent-cluster alias. {element: min_C}.
    min_C_for: dict = field(default_factory=dict)
    # default adducts to search (ion forms from chemistry.ADDUCT_SHIFTS)
    reagent_adducts: tuple = ("[M-H]-",)
    # SOURCE SOLVENTS this ion source carries in bulk, as neutral formulas
    # (assignment/solvent_clusters.py). A positive source running on solvent
    # vapour builds proton-/hydride-bound CLUSTERS of it -- [(C2H6O)2+H]+ was
    # 6.5 % of the total signal on a 2026-09-21 EasyIC+ acquisition -- and those
    # ions are unreachable by the neutral grid ON PURPOSE: the cluster's implied
    # neutral has DBE < 0. Listing the solvents here opens the cluster channel;
    # the family then self-gates on each solvent's own monomer ion, so naming one
    # the source does not actually carry costs nothing. Empty = channel off,
    # which is every negative-mode context (a halide reagent's own clusters are
    # the reagent library's job, not this one).
    source_solvents: tuple = ()
    # contaminant families Pass 3 may open (keys into CONTAMINANT_FAMILIES)
    pass3_families: tuple = ()
    # NOx SKELETON readings (`vk_readings`): judge the Van Krevelen windows on the
    # carbon skeleton of an organonitrate / nitroaromatic as well as on the raw
    # neutral. An -ONO2 / -NO2 group adds N, 2 O and one DBE and replaces an H,
    # so a dinitrate's raw O/C and DBE/C sit outside windows written for CHO
    # skeletons. Off on every built-in context: a run switches it on
    # (`run_profile`) only where it has been validated.
    nox_skeleton: bool = False
    # C3-C4 POLYCARBONYL-ACID band (`small_acid_band_applies`): the raw reading of
    # a C3-C4 CHO(S) acid may reach H/C 0.5, O/C 2.0 and DBE/C 1.0 (every carbon
    # a carbonyl / carboxyl carbon: acetylenedicarboxylic C4H2O4, mesoxalic
    # C3H2O5). Off by default, switched on with `nox_skeleton`.
    small_acid_band: bool = False


# ---------------------------------------------------------------------------
# Contaminant families (Pass 3). Each is an additive element budget layered on
# top of a CHO(N) core, plus the adducts they typically appear under. These are
# *opened* per-context, never on by default.
# ---------------------------------------------------------------------------
CONTAMINANT_FAMILIES: dict[str, dict] = {
    "organosulfate": {"add": {"S": (1, 1), "O": (3, 6)}, "adducts": ("[M-H]-",),
                      "note": "R-OSO3H organosulfate / sulfate ester"},
    "sulfate":       {"add": {"S": (1, 1), "O": (3, 4)}, "adducts": ("[M-H]-", "[M+HSO4]-"),
                      "note": "inorganic / small sulfate"},
    "nitrate":       {"add": {"N": (1, 2), "O": (3, 8)},
                      "adducts": ("[M-H]-", "[M+NO3]-", "[M+^NO3]-"),
                      "note": "organonitrate"},
    # siloxane / pdms: `run_adducts` names the run channels the family may ALSO
    # take beside its own `adducts` (a family without the key takes every one):
    # the positive channels siloxanes show -- sodium / urea adducts, the methyl-
    # loss quantifier ion, a charge-transfer source's radical cation and hydride
    # abstraction -- never an anion cluster. Unioned with a nitrate source's
    # [M+NO3]-, the grid fitted Si1 "clusters" (O6-O9, N) onto ordinary
    # nitrate-cluster lines, none with a 29Si / 30Si line.
    "siloxane":      {"add": {"Si": (1, 6), "O": (1, 6), "C": (2, 12), "H": (6, 36)},
                      "adducts": ("[M+H]+", "[M+NH4]+", "[M+^NH4]+", "[M-H]-"),
                      "run_adducts": ("[M+Na]+", "[M+(CH4N2O)H]+", "[M-CH3]+", "[M]+.", "[M-H]+"),
                      "note": "PDMS / siloxane column bleed (D3..D6)"},
    "pdms":          {"add": {"Si": (4, 12), "O": (3, 14), "C": (8, 26),
                              "H": (18, 78), "N": (0, 2)},
                      "adducts": ("[M+H]+", "[M+NH4]+", "[M+^NH4]+", "[M+Na]+",
                                  "[M+(CH4N2O)H]+"),
                      "run_adducts": ("[M-CH3]+", "[M]+.", "[M-H]+"),
                      "note": "long-chain polydimethylsiloxane / silicone bleed "
                              "(Si-O-Si(CH3)2 ladder, +C2H6OSi = +74.019); the "
                              "Si>6 oligomers the short siloxane family can't reach"},
    "fluorinated":   {"add": {"F": (1, 17), "O": (0, 6)}, "adducts": ("[M-H]-",),
                      "note": "PFAS / CF2 series contaminant; O capped at "
                              "fluorochemical levels -- v16 audit: an open O "
                              "range gridded junk like C6H5F3O13 onto Br-"
                              "doublet peaks"},
    "halogen_dbp":   {"add": {"Cl": (1, 4), "Br": (1, 2)}, "adducts": ("[M-H]-",),
                      "note": "halogenated disinfection by-product"},
    "phthalate":     {"add": {"O": (4, 4)}, "adducts": ("[M+H]+", "[M+NH4]+", "[M+^NH4]+"),
                      "note": "phthalate plasticiser (CnH(2n-6)O4)"},
    "glycol_peg":    {"add": {"O": (2, 12)},
                      "adducts": ("[M+H]+", "[M+NH4]+", "[M+^NH4]+", "[M+Na]+"),
                      "note": "PEG / PPG (+C2H4O repeat)"},
    "amine":         {"add": {"N": (1, 3)}, "adducts": ("[M+H]+",),
                      "note": "aliphatic / aromatic amine"},
    # REDUCED organosulfur (thioureas, thioethers, thiazoles, sulfoxides): the
    # positive grid is CHO(N)-only, so S reaches a neutral only here (or via the
    # pass-0 indoor_sulfur list). Committed on score, then held to the tier
    # engine's isotopologue gate -- a neutral S needs its confirmed 34S line.
    # Opened by the 2026-09-10 15N-ammonium file: C5H12N2S / C9H18N2S / C11H22N2S
    # [M+H]+ (the thio-analogues of the C9H18N2O / C11H22N2O ureas) each with a
    # 4 % 34S satellite, unreachable by any other pass.
    "organosulfur":  {"add": {"S": (1, 2)},
                      "adducts": ("[M+H]+", "[M+^NH4]+", "[M+NH4]+", "[M+(CH4N2O)H]+"),
                      "note": "reduced organosulfur (thiourea / thioether / thiazole; 34S-gated)"},
    "bromo_organic": {"add": {"Br": (1, 2)}, "adducts": ("[M-H]-",),
                      "note": "covalent organobromine (iso-gated on 81Br)"},
    "chloro_organic": {"add": {"Cl": (1, 2)}, "adducts": ("[M-H]-",),
                       "note": "covalent organochlorine (iso-gated on 37Cl)"},
}


# ---------------------------------------------------------------------------
# Context profiles
# ---------------------------------------------------------------------------
_AMBIENT = ContextProfile(
    label="ambient-air",
    description=("Outdoor / ambient air. VOC oxidation chemistry (OH/O3/NO3/Cl): "
                 "CHO, organonitrates, organosulfates; routine contaminant load."),
    # H/C ceiling 2.75 (not 2.6): a saturated C3 polyol -- glycerol C3H8O3,
    # propylene glycol C3H8O2 -- has H/C 2.67 and is REAL atmospheric signal
    # (biomass burning / cooking / industrial). DBE>=0 already caps H/C at
    # 2+2/Ceff, so 2.75 only admits the C3 glycols the old 2.6 wrongly clipped
    # (they slipped in here ONLY via the pass-4 iso-pair bypass).
    h_to_c=(0.7, 2.75), o_to_c=(0.0, 1.5), n_to_c=(0.0, 0.4), dbe_to_c=(0.0, 0.75),
    # max_I=0 for the SAME reason as max_F/max_P=0: ¹²⁷I is monoisotopic, so a
    # covalent iodine in a neutral can never be isotope-confirmed. In an iodide-CIMS
    # source every such formula is also exactly degenerate with an [acid−H+I₂]⁻
    # reagent cluster -- the first iodide batch had the series passes extrapolate
    # CHIO2 / C2H3IO2 / CHIO3 / C2H3IO3 / INO4 off the pass-0 iodine anchors, each
    # one really the I₂ cluster of an acid already in the ledger (CHIO2 [M+I]⁻ ==
    # [HCOOH−H+I₂]⁻). Ambient iodine chemistry reaches a neutral through the
    # pass-0 reactive_iodine known-species list, which bypasses this filter --
    # exactly the PFCA/organophosphate precedent for F and P. (The `water` context
    # keeps max_I=2: iodinated disinfection byproducts are the analyte there.)
    max_N=3, max_S=1, max_P=0, max_F=0, max_Si=1, max_Cl=2, max_Br=2, max_I=0,
    min_C_for={"Br": 5, "Cl": 5, "F": 3},
    reagent_adducts=("[M-H]-", "[M+NO3]-"),
    pass3_families=("organosulfate", "nitrate", "siloxane", "amine"),
)

_CHAMBER = ContextProfile(
    label="chamber",
    description=("Smog / environmental chamber. Clean known precursor + controlled "
                 "oxidant; HOMs and accretion dimers, tight unsaturation."),
    h_to_c=(0.9, 2.75), o_to_c=(0.0, 2.2), n_to_c=(0.0, 0.4), dbe_to_c=(0.0, 0.7),
    max_N=2, max_S=1, max_P=0, max_F=0, max_Si=1,
    min_C_for={"Br": 5, "Cl": 5, "F": 3},
    reagent_adducts=("[M-H]-", "[M+NO3]-"),
    pass3_families=("organosulfate", "nitrate"),
)

_INDOOR = ContextProfile(
    label="indoor-air",
    description=("Indoor air. Siloxanes (personal-care / sealants), glycols, amines, "
                 "phthalates are REAL signal here, not just background."),
    # H/C ceiling 2.75: glycols/glycerol are explicitly REAL indoor signal (see
    # description) -- the old 2.5 ceiling contradicted that by clipping them.
    h_to_c=(0.7, 2.75), o_to_c=(0.0, 1.5), n_to_c=(0.0, 0.5), dbe_to_c=(0.0, 0.9),
    max_N=3, max_S=1, max_P=1, max_F=2, max_Si=6, max_Cl=2, max_Br=1,
    min_C_for={"Br": 5, "Cl": 4, "F": 2},
    reagent_adducts=("[M-H]-", "[M+H]+", "[M+NH4]+"),
    pass3_families=("siloxane", "glycol_peg", "phthalate", "amine", "organosulfate"),
)

_HEADSPACE = ContextProfile(
    label="object-headspace",
    description=("Headspace over an object / material. Terpenes, esters, aldehydes, "
                 "alcohols; broad H/C, sample-specific volatiles."),
    h_to_c=(0.8, 2.6), o_to_c=(0.0, 1.3), n_to_c=(0.0, 0.5), dbe_to_c=(0.0, 1.0),
    max_N=3, max_S=2, max_P=0, max_F=0, max_Si=2,
    min_C_for={"Br": 4, "Cl": 4, "F": 3},
    reagent_adducts=("[M+H]+", "[M-H]-", "[M+NH4]+"),
    pass3_families=("siloxane", "glycol_peg", "amine"),
)

_COMBUSTION = ContextProfile(
    label="combustion",
    description="Combustion / soot precursors / biomass burning; PAHs, high DBE.",
    h_to_c=(0.2, 2.2), o_to_c=(0.0, 1.5), n_to_c=(0.0, 0.5), dbe_to_c=(0.0, 1.1),
    max_N=4, max_S=1, max_P=0, max_F=0,
    reagent_adducts=("[M-H]-",),
    pass3_families=("nitrate",),
)

_WATER = ContextProfile(
    label="water",
    description="Drinking water / DBPs / wastewater; halogenated species expected.",
    h_to_c=(0.5, 2.2), o_to_c=(0.0, 1.5), n_to_c=(0.0, 0.5), dbe_to_c=(0.0, 0.9),
    max_N=5, max_S=2, max_P=1, max_Cl=6, max_Br=4, max_I=2, max_F=4,
    reagent_adducts=("[M-H]-",),
    pass3_families=("halogen_dbp", "organosulfate", "nitrate"),
)

_FOOD = ContextProfile(
    label="food",
    description="Food / beverage / fermentation; natural products + plasticisers.",
    h_to_c=(0.5, 2.6), o_to_c=(0.0, 1.3), n_to_c=(0.0, 0.5), dbe_to_c=(0.0, 1.0),
    max_N=8, max_S=2, max_P=1,
    reagent_adducts=("[M+H]+", "[M-H]-"),
    pass3_families=("phthalate", "glycol_peg", "siloxane"),
)

_URONIUM = ContextProfile(
    label="uronium",
    description=("Urea-CIMS POSITIVE mode (protonated-urea / uronium reagent). "
                 "N-heavy chemistry: oxygenated VOC + amine / N-base analytes seen "
                 "as [M+H]+ and [M+urea+H]+; the urea reagent forms [urea_n+H]+ "
                 "cluster ions. Background/inlet-characterisation sample."),
    polarity="positive",
    # N-heavy positive VK windows. H/C spans aromatic N-heterocycles (~0.5) to
    # saturated amines/amino-alcohols (~2.6). O/C allows HOMs (urea-CIMS detects
    # oxygenated VOC up to O/C~1.5). N/C up to 0.6 admits the N-bases the source
    # is selective for (urea reagent is N-rich) without opening the polyamide
    # corner. DBE/C up to 1.1 admits aromatic / heterocyclic N.
    h_to_c=(0.4, 2.6), o_to_c=(0.0, 1.5), n_to_c=(0.0, 0.6), dbe_to_c=(0.0, 1.1),
    # max_Si 12: the heavy unexplained residual is a long PDMS/silicone ladder
    # (Si up to ~10-12, +C2H6OSi rungs) -- the `pdms` Pass-3 family needs the cap
    # raised to reach it. Si only enters the neutral via the siloxane/pdms
    # families (Pass 1/2 are CHO(N)-only), so this does not loosen the backbone.
    max_N=5, max_S=2, max_P=1, max_F=0, max_Si=12, max_Cl=0, max_Br=0, max_I=0,
    # urea-CIMS reaches ~730 Da -> neutrals ~670 Da; widen C past the ambient 40.
    grid_c_max=46, grid_o_max=32,
    # Si only as a siloxane scaffold (PDMS bleed), never a bare-Si mass-fit.
    min_C_for={"Si": 2},
    reagent_adducts=("[M+H]+", "[M+(CH4N2O)H]+", "[M+Na]+", "[M+NH4]+"),
    # SOURCE SOLVENTS: the same proton-bound solvent clustering as the EasyIC
    # source -- it is positive-mode CI physics, not a fluoranthene speciality --
    # so the channel is opened here too. No hydride ladder: this reagent
    # protonates, it does not abstract H- ([M-H]+ is not among the adducts
    # above, which is what `solvent_clusters.carriers_for` reads). In practice
    # it is inert on the measured urea batches: their windows start at m/z
    # 50-130, above ethanol's and methanol's monomer ions, and the family
    # refuses to build a ladder whose monomer it cannot see -- 93.0911 stays
    # unexplained there rather than becoming an unprovable cluster claim.
    source_solvents=("H2O", "CH4O", "C2H6O", "C3H6O"),
    pass3_families=("amine", "siloxane", "pdms", "glycol_peg", "phthalate"),
)

_EASYIC = ContextProfile(
    label="easyic",
    description=("EasyIC⁺ fluoranthene charge transfer (low-pressure, mildly "
                 "fragmenting). Aromatics survive as [M]+. molecular radical "
                 "cations; alcohols/alkanes lose a hydride ([M-H]+, ethanol -> "
                 "C2H5O+); monoterpenes and larger aliphatics FRAGMENT into "
                 "even-electron CxHy(O)+ pieces. Reduced, hydrocarbon-rich "
                 "chemistry -- not an oxidation-product source."),
    polarity="positive",
    # VK windows span the CT-selective space: PAHs down at H/C 0.6 / DBE/C 0.75
    # (fluoranthene-class analytes; benzene 1.0 / 0.67), alkyl fragments up at
    # H/C ~2.3. O/C 1.2 covers oxygenated VOC without opening the HOM corner
    # (charge transfer is not an oxidation-product channel); DBE/C 1.0 admits
    # the full aromatic/PAH ladder.
    h_to_c=(0.3, 2.6), o_to_c=(0.0, 1.2), n_to_c=(0.0, 0.6), dbe_to_c=(0.0, 1.0),
    # same positive-mode inlet reality as uronium: PDMS ladder reachable, no
    # halogens in the neutral (nothing brings them in without a halide reagent).
    max_N=5, max_S=2, max_P=1, max_F=0, max_Si=12, max_Cl=0, max_Br=0, max_I=0,
    # mz40-500 windows -> neutrals <= ~500 Da; the ambient 40/30 box covers it.
    min_C_for={"Si": 2},
    reagent_adducts=("[M]+.", "[M-H]+", "[M-CH3]+", "[M+H]+"),
    # SOURCE SOLVENTS. This source runs on solvent vapour and clusters it: on a
    # 2026-09-21 acquisition the ethanol dimer (C2H5OH)2H+ (93.0911) alone was
    # 6.5 % of the TOTAL signal and ~half of everything left unexplained, with
    # its 13C satellite (94.0944) and the EtOH.H3O+ mixed cluster (65.0597)
    # unexplained beside it -- none of them reachable by the neutral grid (the
    # implied neutral C4H12O2 has DBE -1). Ethanol and acetone are the observed
    # cluster formers; methanol and water are listed because they co-cluster
    # with them (EtOH.H3O+) and cost nothing when absent -- the family needs a
    # solvent's own monomer ion before it will build any ladder from it.
    # Having the [M-H]+ hydride channel is what additionally opens the
    # hydride-bound ladder ([EtOH-H]+.EtOH = 91.0754) here.
    source_solvents=("H2O", "CH4O", "C2H6O", "C3H6O"),
    pass3_families=("amine", "siloxane", "pdms", "glycol_peg", "phthalate"),
)

_NONE = ContextProfile(
    label="none",
    description="Structural gates only (integer DBE>=0, Senior's rule).",
)

# ¹⁵N-labelled ammonium CIMS (profiles.NH4_15N). Same positive-mode inlet
# reality and N-heavy analyte space as uronium (oxygenated VOC, esters, ketones,
# amines, siloxanes), but the reagent cluster is [M+^NH4]+ (+19.0309) and the
# proton-transfer product [M+H]+ is a DECLUSTERING channel that dehydrates
# readily ([M+H-H2O]+ / [M+^NH4-H2O]+, see cleanup.relabel_ammonium_dehydration).
# The ¹⁵N label removes the [M+NH4]+ / amine [M+H]+ degeneracy, so the amine
# time-tracking gate and the reagent-N isobar gate are both moot for the
# labelled channel; ambient amines are still read on [M+H]+ at their ¹⁴N mass.
# Built from the 2026-09-10 exploratory file (m/z 40-600).
_AMMONIUM_15N = ContextProfile(
    label="ammonium-15n",
    description=("¹⁵N-ammonium CIMS POSITIVE mode ([M+^NH4]+ cluster reagent, "
                 "+19.0309). Oxygenated VOC, esters, ketones, acids seen as "
                 "[M+^NH4]+ with strong declustering to [M+H]+ and in-source "
                 "dehydration; amines / N-bases as [M+H]+; siloxanes (D4) on "
                 "both channels. The 15N label breaks the NH4-adduct / amine "
                 "isobar. Background/inlet-characterisation sample."),
    polarity="positive",
    # H/C up to 3.0 (not 2.6): saturated amines CnH(2n+3)N are 3.0 at C3 and
    # 2.75 at C4 -- the 2.6 ceiling of the uronium context silently dropped
    # C3H9N/C4H11N [M+H]+ (29 kcps diethylamine/butylamine on the 2026-09-10
    # file), and an ammonium source is exactly where N-bases are expected.
    h_to_c=(0.4, 3.0), o_to_c=(0.0, 1.5), n_to_c=(0.0, 0.6), dbe_to_c=(0.0, 1.1),
    max_N=5, max_S=2, max_P=1, max_F=0, max_Si=12, max_Cl=0, max_Br=0, max_I=0,
    grid_c_max=46, grid_o_max=32,
    min_C_for={"Si": 2},
    reagent_adducts=("[M+^NH4]+", "[M+H]+"),
    # SOURCE SOLVENTS: as for uronium -- the proton-bound ladder is opened, the
    # hydride ladder is not (an ammonium reagent protonates). See _EASYIC.
    source_solvents=("H2O", "CH4O", "C2H6O", "C3H6O"),
    pass3_families=("amine", "organosulfur", "siloxane", "pdms", "glycol_peg", "phthalate"),
)

CONTEXTS: dict[str, ContextProfile] = {
    "ambient-air": _AMBIENT, "ambient": _AMBIENT, "atmospheric": _AMBIENT,
    "chamber": _CHAMBER, "smog-chamber": _CHAMBER, "flow-tube": _CHAMBER,
    "indoor-air": _INDOOR, "indoor": _INDOOR,
    "object-headspace": _HEADSPACE, "headspace": _HEADSPACE,
    "combustion": _COMBUSTION, "biomass": _COMBUSTION,
    "water": _WATER, "wastewater": _WATER,
    "food": _FOOD, "wine": _FOOD, "beverage": _FOOD,
    "uronium": _URONIUM, "urea-cims": _URONIUM, "urea": _URONIUM,
    "easyic": _EASYIC, "easy-ic": _EASYIC, "charge-transfer": _EASYIC,
    # LABELLED ammonium only. The bare "ammonium" / "nh4" / "nh4-cims" spellings
    # used to resolve here, so an UNLABELLED ammonium user silently got the ¹⁵N
    # channels ([M+^NH4]+, the ¹⁴N impurity satellite, the labelled known-species
    # locks) and no error to say so. An unknown context raises in get_context,
    # which is the honest answer until an unlabelled ammonium context exists.
    "ammonium-15n": _AMMONIUM_15N, "15nh4": _AMMONIUM_15N,
    "15n-ammonium": _AMMONIUM_15N, "nh4-15n": _AMMONIUM_15N,
    "none": _NONE,
}


def get_context(name: str) -> ContextProfile:
    p = CONTEXTS.get((name or "").lower())
    if p is None:
        raise ValueError(f"unknown context {name!r}; known: {sorted(CONTEXTS)}")
    return p


def filter_by_context(formula: str, context="ambient-air") -> tuple[bool, str | None]:
    """Return (keep, reason). Composes the universal structural gates from
    chemistry.dbe_ok with the context-specific bounds. ``context`` is a context
    name or a ContextProfile (a run's own, `run_profile`)."""
    return filter_by_profile(formula, context if isinstance(context, ContextProfile)
                             else get_context(context))


def element_budget(formula: str, profile: "ContextProfile | str") -> tuple[bool, str | None]:
    """Return (keep, reason) for the context's ELEMENT BUDGET alone -- steps 1-3
    of `filter_by_profile`: the universal structural gate (integer DBE >= 0,
    Senior), the carbon-free allowlist, and the heteroatom caps. Not the
    halogen/Si minimum-carbon scaffold (a reagent-alias guard) and not the Van
    Krevelen windows, which shape what the grid ENUMERATES and would reject real
    aromatics (nitrobenzoic acid, DBE/C 0.86).

    The budget is what a commit made outside the per-peak grid must still
    respect unless a curated list names the formula: ambient-air sets max_P /
    max_F / max_I = 0 because those elements are monoisotopic and can never be
    isotope-confirmed, and max_S = 1. `plausibility.demote_off_budget` holds
    every other commit path to it."""
    if isinstance(profile, str):
        profile = get_context(profile)
    cnt = C.parse_formula(formula)

    # 1. universal structural gate (integer DBE>=0, Senior)
    ok, why = C.dbe_ok(cnt)
    if not ok:
        return False, why

    # 2. inorganic / no-carbon: only a tight atmospheric allowlist
    if cnt.get("C", 0) == 0:
        allowed = _inorganic_allowed(cnt)
        if profile.label in ("ambient-air", "chamber", "indoor-air") and allowed:
            return True, None
        return False, "no carbon"

    # 3. heteroatom caps
    for el, cap in (("N", profile.max_N), ("S", profile.max_S),
                    ("P", profile.max_P), ("F", profile.max_F),
                    ("Cl", profile.max_Cl), ("Br", profile.max_Br),
                    ("I", profile.max_I), ("Si", profile.max_Si)):
        n = cnt.get(el, 0)
        if n > cap:
            return False, f"{el}={n} > {cap}"
    return True, None


# ---------------------------------------------------------------------------
# Van Krevelen readings: the raw neutral, and the carbon skeleton of its NOx groups
# ---------------------------------------------------------------------------
#: the ratio windows a reading is judged on, in the order (and with the names)
#: the filter's reason string reports them
VK_WINDOWS = (("h_to_c", "(H+X)/C"), ("o_to_c", "O/C"), ("n_to_c", "N/C"), ("dbe_to_c", "DBE/C"))
#: at most this many -ONO2 / -NO2 groups are discounted, whatever the context's
#: N cap (a trinitrate -- nitroglycerin, a trinitro-aromatic -- is the chemical
#: ceiling the readings credit)
NOX_K_MAX = 3
#: a skeleton at or above this DBE/C is aromatic enough to carry NITRO groups
#: (R-NO2: 2 O per N, the skeleton keeps none of them); below it only the
#: NITRATE stoichiometry (R-O-NO2: 3 O per N) is read
NITRO_SKELETON_DBE_PER_C = 0.5
#: the small-acid band's widened windows (raw reading only)
SMALL_ACID_WINDOWS = {"h_to_c": (0.5, None), "o_to_c": (None, 2.0), "dbe_to_c": (None, 1.0)}
#: the reagent profiles (chem.profiles names) whose runs read NOx skeletons
NOX_SKELETON_REAGENTS = ("NO3", "NO3_15N")


def nox_skeletons(cnt: dict, profile: "ContextProfile") -> list[tuple[int, dict]]:
    """The NOx SKELETONS of a neutral, ``[(k, counts)]`` for k = k_max .. 1, when
    ``profile.nox_skeleton`` is set ([] otherwise): k -NO2 groups replaced by H,
    so H + k, N - k, O - 2k and DBE - k (removing an N is -1/2 DBE, adding an H
    another -1/2). A nitrate R-O-NO2 reads as the alcohol R-OH (the bridging O
    stays in the skeleton), a nitro R-NO2 as R-H.

    k_max = min(N, floor(DBE), Ceff, NOX_K_MAX): each group needs an N, its
    N=O double bond and its own carbon. A skeleton is kept when its
    stoichiometry is a NITRATE one (it keeps >= k O, i.e. O >= 3k overall) or,
    when its DBE/C >= NITRO_SKELETON_DBE_PER_C (aromatic), a NITRO one
    (O >= 2k). A labelled (15N) N is a group's N first: a heavy nitrate group is
    a nitrate group."""
    if not getattr(profile, "nox_skeleton", False):
        return []
    cnt = {k: v for k, v in cnt.items() if v}
    c_eff = cnt.get("C", 0) + cnt.get("Si", 0)
    n_lab, n_n, n_o = cnt.get("^N", 0), cnt.get("N", 0), cnt.get("O", 0)
    if c_eff < 1:
        return []
    d = C.dbe(cnt)
    k_max = min(n_lab + n_n, int(math.floor(d + 1e-9)), c_eff, NOX_K_MAX)
    out = []
    for k in range(k_max, 0, -1):
        o_s = n_o - 2 * k
        if o_s < 0:
            continue
        if not (o_s >= k or (d - k) / c_eff >= NITRO_SKELETON_DBE_PER_C):
            continue
        skel = dict(cnt)
        take_lab = min(k, n_lab)
        skel["^N"] = n_lab - take_lab
        skel["N"] = n_n - (k - take_lab)
        skel["O"] = o_s
        skel["H"] = cnt.get("H", 0) + k
        out.append((k, {e: v for e, v in skel.items() if v}))
    return out


def _vk_ratios(cnt: dict) -> dict:
    c_eff = cnt.get("C", 0) + cnt.get("Si", 0)
    h_eff = cnt.get("H", 0) + sum(cnt.get(x, 0) for x in ("F", "Cl", "Br", "I"))
    return {"h_to_c": h_eff / c_eff, "o_to_c": cnt.get("O", 0) / c_eff,
            "n_to_c": cnt.get("N", 0) / c_eff, "dbe_to_c": C.dbe(cnt) / c_eff}


def vk_readings(cnt: dict, profile: "ContextProfile") -> list[tuple[int, dict]]:
    """The Van Krevelen readings of a neutral: ``[(k, {h_to_c, o_to_c, n_to_c,
    dbe_to_c})]`` on Ceff = C + Si and Heff = H + halogens. ``k = 0`` (the raw
    neutral, exactly the ratios the filter always read) comes first; then, only
    when ``profile.nox_skeleton`` is set, the skeleton readings k = k_max .. 1
    (`nox_skeletons`). The skeleton's N/C counts the N its groups did not take:
    an amine or ring N keeps its window. Empty for Ceff < 3, where the ratios
    mean nothing."""
    cnt = {k: v for k, v in cnt.items() if v}
    if cnt.get("C", 0) + cnt.get("Si", 0) < 3:
        return []
    return [(0, _vk_ratios(cnt))] + [(k, _vk_ratios(sk)) for k, sk in nox_skeletons(cnt, profile)]


def small_acid_band_applies(cnt: dict, profile: "ContextProfile") -> bool:
    """The C3-C4 polycarbonyl-acid band covers this neutral (raw reading only):
    ``profile.small_acid_band`` set, 3 <= Ceff <= 4, no N, at least one H (an
    acid), 2 <= O <= 2 Ceff and at least one C=O (DBE >= 1: a DBE-0 C3 with O/C 2
    is all gem-diols, not an acid)."""
    if not getattr(profile, "small_acid_band", False):
        return False
    c_eff = cnt.get("C", 0) + cnt.get("Si", 0)
    n_o = cnt.get("O", 0)
    return (3 <= c_eff <= 4 and cnt.get("N", 0) == 0 and cnt.get("^N", 0) == 0
            and cnt.get("H", 0) >= 1 and 2 <= n_o <= 2 * c_eff and C.dbe(cnt) >= 1)


def vk_windows(cnt: dict, profile: "ContextProfile", k: int) -> dict:
    """``{window: (lo, hi)}`` reading ``k`` is judged on: the profile's own
    windows, widened by the small-acid band for the raw reading it covers."""
    win = {name: tuple(getattr(profile, name)) for name, _ in VK_WINDOWS}
    if k == 0 and small_acid_band_applies(cnt, profile):
        for name, (lo, hi) in SMALL_ACID_WINDOWS.items():
            w_lo, w_hi = win[name]
            win[name] = (min(w_lo, lo) if lo is not None else w_lo,
                         max(w_hi, hi) if hi is not None else w_hi)
    return win


def vk_distance(ratios: dict, windows: dict) -> float:
    """Total out-of-window distance of a reading (0 = inside every window)."""
    tot = 0.0
    for name, _ in VK_WINDOWS:
        lo, hi = windows[name]
        v = ratios[name]
        tot += (lo - v) if v < lo else ((v - hi) if v > hi else 0.0)
    return tot


def vk_passing_k(cnt: dict, profile: "ContextProfile") -> int | None:
    """The k of the first reading (raw first) inside every window, else None.
    None also for Ceff < 3 (no ratio test there)."""
    for k, r in vk_readings(cnt, profile):
        if vk_distance(r, vk_windows(cnt, profile, k)) == 0.0:
            return k
    return None


def skeleton_only(formula: str, profile: "ContextProfile") -> bool:
    """The context filter admits ``formula`` ONLY through a NOx skeleton reading
    (k >= 1): its raw reading fails the windows, a skeleton one passes."""
    if not getattr(profile, "nox_skeleton", False):
        return False
    cnt = C.parse_formula(str(formula))
    k = vk_passing_k(cnt, profile)
    return bool(k) and filter_by_profile(str(formula), profile)[0]


def run_profile(profile: "ContextProfile", *, reagent: str | None,
                instrument_class: str | None, trace_sample: bool = False) -> "ContextProfile":
    """The context profile a RUN judges its formulas on: ``profile`` with the
    NOx-skeleton readings and the small-acid band switched on for a nitrate-
    reagent run (``NOX_SKELETON_REAGENTS``) on an Orbitrap-class axis, in a
    context that opens the organonitrate family; unchanged otherwise. Off on
    the trace-first sample (its +-12 ppm search window), on TOF-class and
    unknown-class runs and on every other reagent until validated there."""
    on = (reagent in NOX_SKELETON_REAGENTS and instrument_class == "orbitrap"
          and not trace_sample and "nitrate" in tuple(profile.pass3_families or ()))
    if not on or (profile.nox_skeleton and profile.small_acid_band):
        return profile
    return dataclasses.replace(profile, nox_skeleton=True, small_acid_band=True)


#: the ContextProfile switches a run can set beyond its named context
PROFILE_FLAGS = ("nox_skeleton", "small_acid_band")


def profile_flags(profile: "ContextProfile") -> dict:
    """The run-level switches set on ``profile`` ({} when none is)."""
    return {f: True for f in PROFILE_FLAGS if getattr(profile, f, False)}


def as_profile(context, flags: dict | None = None) -> "ContextProfile":
    """A ContextProfile from a profile or a context name, with the run-level
    ``flags`` (``profile_flags``) applied."""
    prof = context if isinstance(context, ContextProfile) else get_context(context)
    kw = {f: bool(v) for f, v in (flags or {}).items() if f in PROFILE_FLAGS}
    return dataclasses.replace(prof, **kw) if kw else prof


def filter_by_profile(formula: str, profile: "ContextProfile") -> tuple[bool, str | None]:
    """Return (keep, reason) for a formula against an explicit profile. Same
    rules as filter_by_context but takes the profile directly -- used by the
    degeneracy audit with a relaxed profile (the contaminant families the
    pipeline can open raise the strict ambient F/Si caps)."""
    # 1-3. the element budget (structural gate, carbon-free allowlist, caps)
    ok, why = element_budget(formula, profile)
    if not ok:
        return False, why
    cnt = C.parse_formula(formula)
    nC = cnt.get("C", 0); nN = cnt.get("N", 0)
    nO = cnt.get("O", 0); nSi = cnt.get("Si", 0)
    if nC == 0:
        return True, None   # an allowlisted carbon-free analyte

    # 4. heteroatom-in-neutral minimum carbon scaffold (reagent-alias guard)
    for el, min_c in profile.min_C_for.items():
        if cnt.get(el, 0) >= 1 and nC < min_c:
            return False, f"{el} in neutral needs C>={min_c} (got C={nC}); likely reagent alias"

    # 5. Van Krevelen ratios (only meaningful for Ceff>=3).
    #    - Si is a tetravalent backbone atom (treated like C in DBE/Senior), so
    #      the carbon-equivalent denominator is Ceff = C + Si.
    #    - Halogens are monovalent H-substituents (treated like H in DBE), so
    #      the hydrogen-equivalent numerator is Heff = H + F + Cl + Br + I.
    #      Without this, halogen-substituted compounds are falsely rejected:
    #      trichloroacetic acid C2HCl3O2 has H/C=0.5 but (H+X)/C=2.0.
    #    - With profile.nox_skeleton the carbon skeleton of up to NOX_K_MAX
    #      -ONO2 / -NO2 groups is read too (`vk_readings`), and with
    #      profile.small_acid_band a C3-C4 polycarbonyl acid's raw reading is
    #      judged on the band's wider windows (`vk_windows`). A formula passes
    #      when ANY reading is inside every window; a failure reports the RAW
    #      reading on the profile's own windows, exactly as before.
    Ceff = nC + nSi
    if Ceff >= 3:
        readings = vk_readings(cnt, profile)
        if any(vk_distance(r, vk_windows(cnt, profile, k)) == 0.0 for k, r in readings):
            return True, None
        raw = readings[0][1]
        for key, name in VK_WINDOWS:
            win, val = getattr(profile, key), raw[key]
            if not (win[0] <= val <= win[1]):
                return False, f"{name}={val:.2f} out of {win}"
        return True, None    # unreachable: the raw reading failed some window
    elif nC in (1, 2):
        if nO > 2 * nC + 2:
            return False, f"O={nO} implausible for C={nC}"
        if nN > nC + 1:
            return False, f"N={nN} implausible for C={nC}"
    return True, None


def _inorganic_allowed(cnt: dict[str, int]) -> bool:
    """Tight allowlist of carbon-free atmospheric analytes (acids + halogen/N/S
    oxides). Explicit element-count checks, no formula-string matching."""
    H = cnt.get("H", 0); O = cnt.get("O", 0); N = cnt.get("N", 0)
    S = cnt.get("S", 0); F = cnt.get("F", 0); Cl = cnt.get("Cl", 0)
    Br = cnt.get("Br", 0); I = cnt.get("I", 0)
    other = sum(v for k, v in cnt.items()
                if k not in ("H", "O", "N", "S", "F", "Cl", "Br", "I"))
    if other:
        return False
    halo = F + Cl + Br + I
    # inorganic acids
    if N == 1 and 2 <= O <= 3 and H == 1 and S == 0 and halo == 0:
        return True   # HNO2 / HNO3
    if S == 1 and 3 <= O <= 4 and H in (1, 2) and N == 0 and halo == 0:
        return True   # H2SO3 / H2SO4
    if S == 1 and O == 0 and H == 2 and N == 0 and halo == 0:
        return True   # H2S
    # oxygen-only O, O2, O3 ; H2O2
    if 1 <= O <= 3 and H == 0 and N == 0 and S == 0 and halo == 0:
        return True
    if H == 2 and O == 2 and N == 0 and S == 0 and halo == 0:
        return True
    # halogen oxides / hydrides (Br2, HBr, HOBr, ClO, HCl, I2, HI, IO3 ...)
    if Br >= 1 and Br <= 2 and H <= 1 and O <= 3 and N == 0 and S == 0 and F == 0 and Cl == 0 and I == 0:
        return True
    if Cl >= 1 and Cl <= 2 and H <= 1 and O <= 2 and N == 0 and S == 0 and F == 0 and Br == 0 and I == 0:
        return True
    if I >= 1 and I <= 2 and H <= 1 and O <= 3 and N == 0 and S == 0 and F == 0 and Br == 0 and Cl == 0:
        return True
    # NO, NO2 ; SO2
    if N == 1 and 1 <= O <= 2 and H == 0 and S == 0 and halo == 0:
        return True
    if S == 1 and O == 2 and H == 0 and N == 0 and halo == 0:
        return True
    return False


# ---------------------------------------------------------------------------
# Reporting: compound class + oxidation level + heteroatom tags
# ---------------------------------------------------------------------------
_CLASS_BANDS = [
    (0, 0, "Inorganic / C-free"),
    (1, 4, "Small molecule (C1-4)"),
    (5, 9, "C5-C9"),
    (10, 10, "C10 monomer"),
    (11, 14, "C11-C14"),
    (15, 15, "C15"),
    (16, 20, "C16-C20 dimer / accretion"),
]


def classify_compound(formula: str) -> tuple[str, str, str]:
    """(class, oxidation_level, heteroatom_tags) for a neutral formula."""
    cnt = C.parse_formula(formula)
    nC = cnt.get("C", 0); nO = cnt.get("O", 0)
    cls = "Heavy (C>20)"
    for lo, hi, label in _CLASS_BANDS:
        if lo <= nC <= hi:
            cls = label
            break
    oc = nO / nC if nC else 0.0
    ox = "low-O (O/C<0.2)" if oc < 0.2 else ("moderate-O" if oc < 0.7 else "high-O (O/C>=0.7)")
    tags = []
    for el, name in (("N", "organic-N"), ("S", "organosulfur"), ("Br", "brominated"),
                     ("Cl", "chlorinated"), ("F", "fluorinated"), ("Si", "silicon-bearing")):
        if cnt.get(el, 0):
            tags.append(name)
    if not tags:
        tags.append("CHO only")
    return cls, ox, ", ".join(tags)
