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

from dataclasses import dataclass, field

from peaky.chem import chemistry as C

__version__ = "0.3.0"


@dataclass(frozen=True)
class ContextProfile:
    label: str
    description: str
    # Ion-source polarity. "negative" is the Br/halide-CIMS default the pipeline
    # was built around; "positive" (urea-CIMS / APCI+) switches the opportunistic
    # channels to cation adducts and turns OFF the Br-specific composite test
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
    # contaminant families Pass 3 may open (keys into CONTAMINANT_FAMILIES)
    pass3_families: tuple = ()


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
    "siloxane":      {"add": {"Si": (1, 6), "O": (1, 6), "C": (2, 12), "H": (6, 36)},
                      "adducts": ("[M+H]+", "[M+NH4]+", "[M-H]-"),
                      "note": "PDMS / siloxane column bleed (D3..D6)"},
    "pdms":          {"add": {"Si": (4, 12), "O": (3, 14), "C": (8, 26),
                              "H": (18, 78), "N": (0, 2)},
                      "adducts": ("[M+H]+", "[M+NH4]+", "[M+Na]+", "[M+(CH4N2O)H]+"),
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
    "phthalate":     {"add": {"O": (4, 4)}, "adducts": ("[M+H]+", "[M+NH4]+"),
                      "note": "phthalate plasticiser (CnH(2n-6)O4)"},
    "glycol_peg":    {"add": {"O": (2, 12)}, "adducts": ("[M+H]+", "[M+NH4]+", "[M+Na]+"),
                      "note": "PEG / PPG (+C2H4O repeat)"},
    "amine":         {"add": {"N": (1, 3)}, "adducts": ("[M+H]+",),
                      "note": "aliphatic / aromatic amine"},
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
    reagent_adducts=("[M]+.", "[M-H]+", "[M+H]+"),
    pass3_families=("amine", "siloxane", "pdms", "glycol_peg", "phthalate"),
)

_NONE = ContextProfile(
    label="none",
    description="Structural gates only (integer DBE>=0, Senior's rule).",
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
    "none": _NONE,
}


def get_context(name: str) -> ContextProfile:
    p = CONTEXTS.get((name or "").lower())
    if p is None:
        raise ValueError(f"unknown context {name!r}; known: {sorted(CONTEXTS)}")
    return p


def filter_by_context(formula: str, context: str = "ambient-air") -> tuple[bool, str | None]:
    """Return (keep, reason). Composes the universal structural gates from
    chemistry.dbe_ok with the context-specific bounds."""
    return filter_by_profile(formula, get_context(context))


def filter_by_profile(formula: str, profile: "ContextProfile") -> tuple[bool, str | None]:
    """Return (keep, reason) for a formula against an explicit profile. Same
    rules as filter_by_context but takes the profile directly -- used by the
    degeneracy audit with a relaxed profile (the contaminant families the
    pipeline can open raise the strict ambient F/Si caps)."""
    cnt = C.parse_formula(formula)
    nC = cnt.get("C", 0); nH = cnt.get("H", 0); nN = cnt.get("N", 0)
    nO = cnt.get("O", 0); nS = cnt.get("S", 0); nP = cnt.get("P", 0)
    nF = cnt.get("F", 0); nCl = cnt.get("Cl", 0); nBr = cnt.get("Br", 0)
    nI = cnt.get("I", 0); nSi = cnt.get("Si", 0)

    # 1. universal structural gate (integer DBE>=0, Senior)
    ok, why = C.dbe_ok(cnt)
    if not ok:
        return False, why

    # 2. inorganic / no-carbon: only a tight atmospheric allowlist
    if nC == 0:
        allowed = _inorganic_allowed(cnt)
        if profile.label in ("ambient-air", "chamber", "indoor-air") and allowed:
            return True, None
        return False, "no carbon"

    # 3. heteroatom caps
    for el, cap, n in (("N", profile.max_N, nN), ("S", profile.max_S, nS),
                       ("P", profile.max_P, nP), ("F", profile.max_F, nF),
                       ("Cl", profile.max_Cl, nCl), ("Br", profile.max_Br, nBr),
                       ("I", profile.max_I, nI), ("Si", profile.max_Si, nSi)):
        if n > cap:
            return False, f"{el}={n} > {cap}"

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
    d = C.dbe(cnt)
    Ceff = nC + nSi
    Heff = nH + nF + nCl + nBr + nI
    if Ceff >= 3:
        for name, val, win in (("(H+X)/C", Heff / Ceff, profile.h_to_c),
                               ("O/C", nO / Ceff, profile.o_to_c),
                               ("N/C", nN / Ceff, profile.n_to_c),
                               ("DBE/C", d / Ceff, profile.dbe_to_c)):
            if not (win[0] <= val <= win[1]):
                return False, f"{name}={val:.2f} out of {win}"
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
