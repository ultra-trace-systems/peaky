"""Reagent profiles — ONE config per reagent system, replacing the adducts /
element-ranges / normaliser / label constants that were copy-pasted inline across
the time-series, clustering and validation scripts.

A profile is everything the pipeline needs to treat a batch's reagent correctly.
New reagent = add a ReagentProfile, not edit code. `resolve()` picks one by name
or auto-detects from a loaded peak table (the server's own adduct mechanisms via
io_mascope.recognised_adducts; it stops rather than guess when none is diagnostic).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ReagentProfile:
    name: str  # short key, e.g. "Br" / "Ur"
    label: str  # display label (figures + console), e.g. "Br- CIMS"
    polarity: str  # "-" or "+"
    adducts: list[str]  # analyte channels (peaky adduct labels)
    normaliser: str  # "reagent" | "tic"  (for the TS/correlation layer)
    reagent_ion_re: str | None  # regex on ion_formula picking the reagent ions
    ranges: str  # grid element ranges for local enumeration
    detect_adduct: str | None  # presence of this adduct => this reagent (auto-detect)
    context: str = "ambient-air"  # default assign.run context (mode + VK priors + caps)
    # isotopic purity of a labelled reagent (0.98 = 98 % 15N). assign.run
    # publishes it via isotopes.set_label_purity; it then sets the height of a
    # '^X' ion's unlabelled impurity line in BOTH consumers -- the local scorer's
    # predict_isotopes call and peaky's own envelope predictor
    # (isotopes.isotope_pattern). None => isotopes.LABEL_PURITY_15N (0.98).
    purity: float | None = None
    # labelled-reagent covalent-product rescue (labeled.py): the caret heavy
    # isotope a product can carry ('^N' = 15N organonitrate) and the max count.
    # None => no rescue (every unlabelled profile).
    label_isotope: str | None = None
    label_max: int = 2
    # The profile's recommended height gate for the height-gated assignment
    # passes, as a MULTIPLE of the sample's own noise edge (the 1st percentile of
    # its picked peak heights). None = use the package policy
    # (passes.config.AUTO_HEIGHT_CUTOFF_X_EDGE): on a batch the multiple is DERIVED
    # from the batch's own peaks -- 1.0 for a picker that stops at the noise edge,
    # higher where the picker demonstrably picks into the noise (see
    # admission.derive_height_cutoff_x_edge) -- and 1.0 for a single sample.
    # What decides this number is the PEAK PICKER, not the reagent chemistry, so
    # every bundled profile leaves it None. Pin a number here -- in a
    # --reagent-config profile written for your own instrument -- only to override
    # the derivation for that instrument.
    height_cutoff_x_edge: float | str | None = None
    # True when `detect_adduct` is a WEAK signature -- one the server also stamps in
    # runs of other reagents, so its presence is not evidence that THIS reagent is in
    # the inlet. The bare molecular cation [M]+. is the case in point: a uronium batch
    # stamps it on charge-transfer peaks while carrying no fluoranthene at all. Auto-
    # detect composes the STRONG signatures it sees and only falls back on the weak
    # ones when nothing strong matched, so a stray stamp can never bolt a phantom
    # reagent onto a real one.
    detect_weak: bool = False
    # ION-ONLY channels (cleanup.commit_ion_only_electron_attachment, pipeline
    # stage `ion_only`): adducts a profile opens for the committed-composition,
    # open-process bucket. "[M]-." = the radical anion an O-rich closed-shell
    # acid forms when a primary source ion (an electron / O2-.) attaches instead
    # of the reagent: the +1.0078 Da line beside the acid's [M-H]-, pinned to
    # the acid's composition by exact mass and its own 13C, anti-correlated with
    # the acid and switching with the source state. NEVER a grid channel (every
    # even-mass peak would gain a CHO radical-anion reading against the
    # organonitrate [M-H]- and 13C lines): the stage anchors on committed
    # [M-H]- rows and commits Candidate rows on this adduct only. Empty = off.
    # Copied onto PassConfig.ion_only_channels by `apply_ion_only_channels`.
    ion_only_channels: tuple = ()
    # REAGENT-WATER CORES (C15, batch.reagent_water): the neutral compositions of
    # the reagent-side core ions whose water clusters core.(H2O)n the batch looks
    # for in its own time series. A rung becomes a reagent row only where the
    # batch shows it -- the core and every lower rung present in half a segment's
    # spectra, the rung itself 3x above its decoy offsets -- so declaring a core
    # never claims a peak by itself. Halogen isotopologues are enumerated by the
    # module. Empty = off (the positive-mode profiles: urea.H+.H2O is 5e-5 of
    # urea.H+ on the uronium Orbitrap).
    water_cores: tuple = ()
    # NEUTRAL PAIR (rule U, docs/EVIDENCE_LEVELS.md section 3.2 `upair`): a
    # (bare, cluster) adduct pair of this chemistry whose two ions, seen together
    # at exact mass and co-varying, were read as establishing the NEUTRAL the way
    # the acid branch does on the anion channels -- the uronium pair ([M+H]+,
    # [M+(CH4N2O)H]+). `batch/neutral_pairs.py` measures the fact from the batch
    # time series; the evidence scale records it and reads it nowhere (the same
    # two routes are a tag there, at their measured base rate). Empty
    # = off (every other bundled profile: an unscoped fact would lift two-channel
    # rows of any chemistry).
    neutral_pair: tuple = ()
    # SIDE CHANNELS: adducts the chemistry also makes beside its analyte channels,
    # scored on top of `adducts` when the server resolves their mechanism
    # (assign.run). DECLARED here, never opened by polarity: an extra channel
    # offered to every peak doubles the alias space, and on the measured batches
    # the polarity-wide set read ions the run already held, or ions with no
    # support (a carbonate cluster of the C(n-1) neutral is the same ion as the
    # Cn radical anion; a di-bromide reading whose own 81Br line is missing; a
    # sodium adduct that tracks no partner). Only the uronium profile declares
    # one: [M+NH4]+, the ammonium adduct the urea source makes, which the batch's
    # amine gate (cleanup.prefer_amine_over_ammonium) then confirms by time
    # behaviour or re-reads as the protonated amine. Empty = closed. A run opens
    # others with `--side-channels` (PassConfig.side_channels), which outranks
    # this tuple; copied onto the cfg by `apply_side_channels`.
    side_channels: tuple = ()
    aliases: tuple = field(default_factory=tuple)


BR = ReagentProfile(
    name="Br",
    label="Br- CIMS",
    polarity="-",
    adducts=["[M+Br]-", "[M-H]-", "[M+HBr+Br]-"],
    normaliser="reagent",
    reagent_ion_re=r"Br\d-$",
    ranges="C0-40 H0-80 N0-3 O0-18 S0-2 Cl0-2 Br0-2",
    detect_adduct="[M+Br]-",
    context="ambient-air",
    # Br- / Br2-. / Br3- and HNO3.Br- (a mixed-inlet core: its water ladder is a
    # reagent-side series like NO3-'s; the core itself stays the HNO3 [M+Br]- reading)
    water_cores=("Br", "Br2", "Br3", "HBrNO3"),
    aliases=("br", "bromide", "br-cims", "br-"),
)

UR = ReagentProfile(
    name="Ur",
    label="Ur+ CIMS",
    polarity="+",
    adducts=["[M+H]+", "[M+(CH4N2O)H]+"],
    normaliser="tic",
    reagent_ion_re=None,
    ranges="C0-40 H0-90 N0-8 O0-15 S0-2",
    detect_adduct="[M+(CH4N2O)H]+",
    context="uronium",
    neutral_pair=("[M+H]+", "[M+(CH4N2O)H]+"),
    # the ammonium adduct of the urea source, kept or re-read by the batch's amine gate
    side_channels=("[M+NH4]+",),
    aliases=("ur", "uronium", "urea", "urea-cims", "ur+"),
)

# NO3⁻ (nitrate) CIMS — PROVISIONAL built-in; validate + refine for your instrument
# (or override via a --reagent-config file). Negative mode; highly oxygenated
# molecules detected as the [M+NO3]⁻ cluster (and [M-H]⁻ when acidic). Reagent ions
# are the NO3⁻ / (HNO3)ₙ·NO3⁻ cluster series.
# IODINE: the reactive-iodine species (HOI, HIO2, HIO3 = iodic acid, OIO, INO2,
# INO3, ICl, IBr ...) are reachable on this profile's channels ([M-H]⁻ = IO⁻/IO2⁻/
# IO3⁻ and the [M+NO3]⁻ cluster) through the pass-0 `reactive_iodine` known-species
# family (directors._known_species, negative polarity). Iodine stays OFF the
# neutral grid on purpose (monoisotopic, no envelope to confirm a C-H-N-O-S-I
# organic; see the iodide profile). Pass 0 commits within `PassConfig.pass0_ppm`
# of the prior offset (2 ppm default); on a TOF where the nitrate clusters sit
# 7-11 ppm high, raise pass0_ppm or the iodine acids never commit.
NO3 = ReagentProfile(
    name="NO3",
    label="NO3- CIMS",
    polarity="-",
    adducts=["[M+NO3]-", "[M-H]-"],
    normaliser="reagent",
    reagent_ion_re=r"(HNO3)*NO3-?$",
    ranges="C0-40 H0-60 N0-3 O0-25 S0-2",
    detect_adduct="[M+NO3]-",
    context="ambient-air",
    ion_only_channels=("[M]-.",),   # the electron-attachment line beside each acid's [M-H]-
    # NO3-, HNO3.NO3-, (HNO3)2.NO3- and NO2-. NO3-, HNO3.NO3- and NO2- stay the HNO3 /
    # HNO2 analyte readings and only their water clusters are reagent-side; the
    # (HNO3)2.NO3- core has no analyte reading and is itself a reagent row
    # (batch.reagent_water.REAGENT_ONLY_CORES), as are its water clusters
    water_cores=("NO3", "HN2O6", "H2N3O9", "NO2"),
    aliases=("no3", "nitrate", "no3-", "nitrate-cims"),
)

# ¹⁵N-labelled nitrate CIMS (server reagent '^NO3-'). Same chemistry as NO3 above,
# but the cluster adduct is the heavy [M+¹⁵NO3]⁻ = [M+^NO3]- (+62.9855, mechanism
# '[M+^NO3]-'); the deprotonation channel [M-H]- is isotope-independent. Reagent
# cluster ions ((H^NO3)ₙ·^NO3⁻) usually sit below a >120 m/z acquisition window, so
# the correlation layer normalises on TIC, not on a reagent ion. detect_adduct is
# [M+^NO3]- so auto-detect distinguishes it from the ¹⁴N NO3 profile above.
NO3_15N = ReagentProfile(
    name="NO3_15N",
    label="[15N]O3- CIMS",
    polarity="-",
    adducts=["[M+^NO3]-", "[M-H]-"],
    normaliser="tic",
    reagent_ion_re=None,
    ranges="C0-40 H0-60 N0-3 O0-25 S0-2",
    detect_adduct="[M+^NO3]-",
    context="ambient-air",
    purity=0.98,  # ~98% 15N reagent
    label_isotope="^N",   # covalent 15N products (organonitrates) rescued by labeled.py
    label_max=2,          # up to di-organonitrate
    ion_only_channels=("[M]-.",),   # as NO3: the M-. line beside each acid's [M-H]-
    water_cores=("^NO3", "H^N2O6", "H2^N3O9"),   # the labelled cores (the monomer and dimer sit below a m/z 130 window)
    aliases=(
        "no3-15n",
        "15no3",
        "15no3-",
        "^no3",
        "^no3-",
        "15n-nitrate",
        "nitrate-15n",
        "nitrate-15n-cims",
    ),
)

# Iodide (I⁻) CIMS — negative mode. A SOFT chemical ionisation: most analytes are
# detected as the [M+I]⁻ adduct cluster, and strong acids (HNO3, HCOOH, ...) ALSO
# appear on the deprotonation channel [M-H]⁻ (both server-confirmed on the
# 2026-07-21 batch — see docs/REAGENTS.md). The reagent-ion ladder is
# I⁻ (127) / I₂⁻· (254, the BRIGHTEST ion in every sample) / I₃⁻ (381), plus the
# pure-iodine-oxide poly-iodide (I₂O⁻/I₃O⁻) source-background clusters (labelled
# by reagents.build_library("I")). The IOₓ⁻ oxide anions are NOT labelled: they
# are the [M-H]⁻ ions of the iodine oxyacids (IO₃⁻ = iodic acid's dominant
# channel). The AMBIENT reactive-iodine species (HOI, HIO2, HIO3, OIO, INO2,
# ICl, IBr, ICN, INCO ...) are pass-0 `reactive_iodine` known species on the
# [M+I]⁻ / [M-H]⁻ channels — covalent iodine is MONOISOTOPIC (only ¹²⁷I), so it
# cannot be isotope-confirmed and is kept OFF the neutral grid (no I in
# `ranges`, like F/P): iodine reaches a neutral only via the adduct or the
# known-species list. [M+I2]⁻ is kept as a secondary analyte channel (server
# mechanism +I2-). [M-H+I2]⁻ (conjugate base · I₂, e.g. [HCOOH-H+I2]⁻ @298.807)
# is a cluster-DECOMPOSITION alias like the Br-CIMS [M+HBr+Br]⁻ -- no server
# mechanism; pass 3 scores the covalent alias (M-H+I) [M+I]⁻ (the same ion) and
# commits the acid reading. The In⁻ ladder is in the measured 40-600 window, so
# the correlation layer normalises on the reagent ion.
IODIDE = ReagentProfile(
    name="I",
    label="I- CIMS",
    polarity="-",
    adducts=["[M+I]-", "[M-H]-", "[M+I2]-", "[M-H+I2]-"],
    normaliser="reagent",
    reagent_ion_re=r"I\d*-$",
    ranges="C0-40 H0-80 N0-3 O0-20 S0-2 Cl0-1",
    detect_adduct="[M+I]-",
    context="ambient-air",
    aliases=("i", "iodide", "iodide-cims", "i-", "i-cims", "iodine"),
)

# EasyIC⁺ -- the Orbitrap's internal-calibration (EASY-IC) fluoranthene cation
# beam used as a LOW-PRESSURE, mildly fragmenting charge-transfer CI source
# (2026 EasyIC+ acquisition batches). Three ionization channels:
#   [M]+.   charge transfer -- aromatics keep the intact skeleton as RADICAL
#           molecular cations (server mechanism '+': toluene 92.0621 and the
#           C16H10+. reagent ion itself are server-matched on the 2026-02-26
#           mz40-500 batch); monoterpenes and larger aliphatics FRAGMENT, so
#           their CxHy+ pieces land on this channel (odd-electron) or on...
#   [M-H]+  ...HYDRIDE abstraction (even-electron): alcohols' primary channel
#           -- ethanol is C2H5O+ @45.0335 ONLY ([M+H]+ 47.049 absent). No
#           server mechanism, so it is a local-scoring channel (the [M-H+I2]-
#           ruling) and stays out of ADDUCT_TO_MECH.
#   [M+H]+  a real secondary channel (protonated acetone 59.049 observed).
# The C16H10+. reagent ion (202.0776) is in-window only for the mz40-500
# batches -- mz40-160 misses it -- so the correlation layer normalises on TIC
# (the NO3_15N ruling). Source ions (fluoranthene ladder, N3+/NO2+ air plasma,
# urea crossover from the alternating uronium source) are labelled by
# reagents.build_library("EasyIC"). Auto-detect: the server's bare '+' stamp
# maps to [M]+.; Ur batches carry +(CH4N2O)H+ and resolve first (dict order).
# FRAGMENTATION AMBIGUITY: because the source fragments, three readings are
# MS1-irreducible (carbonyl [M+H]+ vs alcohol [M-H]+; alkene [M+H]+ vs alcohol
# [M+H-H2O]+ dehydration; hydrocarbon cation vs fragment-of-larger-analyte).
# cleanup.annotate_easyic_ambiguity (easyic context only) relabels
# corroborated dehydrations and stamps the rest into commentary -- the
# 2026-08-31 gin-run lessons (59.049 was acetone AND propanol; C4H8 [M+H]+
# was dehydrated butanol, its C4H9O+ hydride partner present at x50).
EASYIC = ReagentProfile(
    name="EasyIC",
    label="EasyIC+ CT",
    polarity="+",
    # [M-CH3]+ added 2026-09-22 off the certified-mixture audit: it is the
    # quantifier channel of the cyclic methylsiloxanes (D4 281.0511, D5
    # 355.0699 -- the brightest peak of the 210-500 window), and without it the
    # ion is committed under a neutral that cannot exist (C9H26O5Si5).
    adducts=["[M]+.", "[M-H]+", "[M-CH3]+", "[M+H]+"],
    normaliser="tic",
    reagent_ion_re=None,
    # Si stays OUT of the grid deliberately. The siloxanes reach [M-CH3]+
    # through the pass-0 `cyclosiloxane` known list (D3-D7, L2-L5) where the
    # commit is gated on >=2 channels or a confirmed ²⁹Si/³⁰Si envelope; opening
    # Si here would instead let the SERIES passes propose Si formulas against
    # the mass-degenerate CHON O-monsters that siloxane.py exists to beat, and
    # would multiply the cached enumeration grid for every pass.
    ranges="C0-40 H0-80 N0-5 O0-15 S0-2",
    detect_adduct="[M]+.",
    detect_weak=True,   # a bare "+" stamp also appears in other positive-mode runs
    context="easyic",
    aliases=("easyic", "easy-ic", "easyic+", "fluoranthene", "charge-transfer"),
)

# ¹⁵N-labelled AMMONIUM CIMS ('^NH4+' ionisation mode, server mechanism
# '[M+^NH4]+'; first file: the 2026-09-10 exploratory acquisition).
# Positive mode; the analyte cluster is [M+^NH4]+ (+19.0309). Why the label:
# with ¹⁴NH4+ the adduct of a CHO neutral X is mass- AND isotope-identical to
# [M+H]+ of the amine X+NH3 (the uronium/NH4 degeneracy that
# prefer_amine_over_ammonium has to arbitrate by time behaviour). With ¹⁵N the
# adduct sits +0.99703 Da from any unlabelled ion, so every [M+^NH4]+ commit is
# free of that degeneracy and the isobar gate (tiers.N_DONOR_ADDUCTS) does not
# apply. Ambient amines still appear on [M+H]+ at their ¹⁴N mass.
# Measured on the exploratory file:
#   * ¹⁴N satellite / ¹⁵N adduct = 0.018-0.021 on the 20 brightest adducts ->
#     effective purity 0.98 (= the reagent's nominal 98 atom %); ambient ¹⁴NH₃
#     is not visible, so [M+NH4]+ is NOT an analyte channel here (it would only
#     re-claim the satellites; assign.run keeps it closed even when asked for).
#   * the bare reagent ions (^NH4+ 19.031, ^NH4+·H2O 37.041, (^NH3)2H+ 37.054)
#     sit BELOW the m/z 40-600 window and no water/ammonia cluster of them is
#     seen in-window -> the correlation layer normalises on TIC (the NO3_15N /
#     uronium ruling). reagents.build_library("ammonium15N") still labels them
#     for wider windows.
#   * DECLUSTERING is strong: [M+H]+ of the same neutral at 0.3-0.95 x the
#     adduct for esters/ketones/aromatics, and oxygenates go on to lose water
#     ([M+H-H2O]+ at 0.9 x [M+H]+ for C11H14O2; C12H18O2 shows NO surviving
#     [M+H]+ at all, only [M+H-H2O]+ / [M+^NH4-H2O]+). MS2 of the parents
#     reproduces the cascade (197.130 -> 179.107 -> 161.096). Handled by
#     cleanup.relabel_ammonium_dehydration (the ammonium-15n context stage).
#   * a uronium crossover [urea+H]+ (61.040, the source alternates with the
#     urea module) is present but no analyte urea adduct was seen; the urea
#     channel is therefore not in `adducts`.
# label_isotope stays None: an ammonium reagent adds no covalent ¹⁵N to a
# product (no radical chemistry), so the labeled.py rescue must not run.
NH4_15N = ReagentProfile(
    name="NH4_15N",
    label="[15N]H4+ CIMS",
    polarity="+",
    adducts=["[M+^NH4]+", "[M+H]+"],
    normaliser="tic",
    reagent_ion_re=None,
    ranges="C0-40 H0-90 N0-6 O0-15 S0-2",
    detect_adduct="[M+^NH4]+",
    context="ammonium-15n",
    purity=0.98,          # measured 0.98 (14N/15N adduct pairs), = nominal 98 atom %
    label_isotope=None,   # no covalent 15N products from an ammonium reagent
    aliases=(
        "nh4-15n",
        "15nh4",
        "15nh4+",
        "^nh4",
        "^nh4+",
        "nh4_15n",
        "ammonium-15n",
        "15n-ammonium",
        "ammonium-15n-cims",
        "labelled-ammonium",
    ),
)

PROFILES: dict[str, ReagentProfile] = {
    BR.name: BR,
    UR.name: UR,
    NO3.name: NO3,
    NO3_15N.name: NO3_15N,
    IODIDE.name: IODIDE,
    EASYIC.name: EASYIC,
    NH4_15N.name: NH4_15N,
}
_BY_ALIAS = {a: p for p in PROFILES.values() for a in (p.name.lower(), *p.aliases)}
# the registry as the package ships it: what a freshly imported module holds
# (a spawned worker process), against which `registry_extras` reads what a
# caller added
_BUILTIN_PROFILES = dict(PROFILES)
_BUILTIN_ALIAS = dict(_BY_ALIAS)


# --- registry / config-driven reagents ------------------------------------
# New reagent = register a ReagentProfile (in code, or from a JSON/TOML config so
# users add reagents WITHOUT forking the package).
_CONFIG_FIELDS = (
    "name",
    "label",
    "polarity",
    "adducts",
    "normaliser",
    "reagent_ion_re",
    "ranges",
    "detect_adduct",
    "context",
    "height_cutoff_x_edge",
    "aliases",
    # labelled-reagent fields (NO3_15N / NH4_15N style profiles from a config)
    "purity",
    "label_isotope",
    "label_max",
    # ion-only channels (`[M]-.` on the nitrate profiles): a list in the config
    "ion_only_channels",
    # reagent-water cores (batch.reagent_water): a list of neutral compositions
    "water_cores",
    # the (bare, cluster) neutral pair of rule U: a two-item list in the config
    "neutral_pair",
    # declared side channels (scored when the server resolves them): a list in the config
    "side_channels",
)


def register(profile: "ReagentProfile", *, overwrite: bool = True) -> "ReagentProfile":
    """Add (or replace) a reagent profile in the registry + alias map."""
    if not overwrite and profile.name in PROFILES:
        raise ValueError(f"reagent {profile.name!r} already registered")
    PROFILES[profile.name] = profile
    for a in (profile.name.lower(), *profile.aliases):
        _BY_ALIAS[a] = profile
    return profile


def from_dict(entry: dict) -> "ReagentProfile":
    """Build a ReagentProfile from a plain dict (config entry)."""
    kw = {k: entry[k] for k in _CONFIG_FIELDS if k in entry}
    if "aliases" in kw:
        kw["aliases"] = tuple(kw["aliases"])
    if "ion_only_channels" in kw:
        kw["ion_only_channels"] = tuple(kw["ion_only_channels"] or ())
    if "water_cores" in kw:
        kw["water_cores"] = tuple(kw["water_cores"] or ())
    if "side_channels" in kw:
        kw["side_channels"] = _side_tuple(kw["side_channels"])
    if "neutral_pair" in kw:
        kw["neutral_pair"] = tuple(kw["neutral_pair"] or ())
        if kw["neutral_pair"] and len(kw["neutral_pair"]) != 2:
            raise ValueError(f"neutral_pair must be two adducts (bare, cluster), got {kw['neutral_pair']!r}")
    if kw.get("height_cutoff_x_edge") is not None:
        kw["height_cutoff_x_edge"] = _x_edge_value(kw["height_cutoff_x_edge"])  # 'auto' or a number
    return ReagentProfile(**kw)


def load_config(path: str) -> list:
    """Register reagent profiles from a JSON or TOML file (so users add reagents
    without editing the package). Accepts a top-level list of entries, a
    `{"reagents": [...]}` wrapper, or a `{name: {fields...}}` mapping. Each entry
    carries the ReagentProfile fields listed in `_CONFIG_FIELDS`: the required
    name/label/polarity/adducts/normaliser/reagent_ion_re/ranges/detect_adduct,
    plus optional context/aliases, the labelled-reagent trio purity /
    label_isotope / label_max, height_cutoff_x_edge and the declared
    side_channels (a list of adduct labels; empty = closed)."""
    import json
    import os

    p = os.path.expanduser(path)
    raw = open(p, "rb").read()
    if p.endswith(".toml"):
        import tomllib

        data = tomllib.loads(raw.decode())
    else:
        data = json.loads(raw.decode())
    if isinstance(data, dict):
        entries = (
            data["reagents"]
            if isinstance(data.get("reagents"), list)
            else [{"name": k, **v} for k, v in data.items()]
        )
    else:
        entries = data
    return [register(from_dict(e)) for e in entries]


def registry_extras() -> tuple[dict, dict]:
    """What this process's registry holds beyond the built-ins: the profiles
    `register` / `load_config` added or replaced, and the aliases that point at
    them, as a picklable `(profiles, aliases)` pair. A spawned process imports
    this module afresh -- built-ins only -- so a profile registered from a
    --reagent-config file is an unknown reagent there until it is handed these
    (`register_extras`); the batch's worker pool does exactly that."""
    return ({k: p for k, p in PROFILES.items() if _BUILTIN_PROFILES.get(k) is not p},
            {a: p for a, p in _BY_ALIAS.items() if _BUILTIN_ALIAS.get(a) is not p})


def register_extras(extras) -> None:
    """Apply another process's `registry_extras()` to this one's registry, so a
    name or alias resolves here to the profile it resolved to there. None or
    empty is a no-op."""
    profs, aliases = extras or ({}, {})
    PROFILES.update(profs)
    _BY_ALIAS.update(aliases)


# --- the noise-edge height gate's multiple ---------------------------------
# ONE resolution order, used by every entry point that resolves a profile and
# builds a PassConfig (the CLI, assign_batch, the pipeline, the MCP tools):
#     an explicit caller / command-line value
#         > the reagent profile's own height_cutoff_x_edge
#             > passes.config.DEFAULT_HEIGHT_CUTOFF_X_EDGE.
# The default constant lives with the gate it defaults (passes/config.py) and is
# imported lazily here: this module is the chemistry layer and must not import
# the assignment layer at module scope.
def _x_edge_value(v):
    """A multiple as stored: the 'auto' token stays a token, anything else a float."""
    from peaky.assignment.passes.config import AUTO_HEIGHT_CUTOFF_X_EDGE, is_auto_x_edge

    return AUTO_HEIGHT_CUTOFF_X_EDGE if is_auto_x_edge(v) else float(v)


def resolve_height_cutoff_x_edge(explicit: float | str | None = None,
                                 profile: "ReagentProfile | None" = None) -> float | str:
    """The height gate as a multiple of the sample's own noise edge: `explicit`
    if given, else the profile's own value, else the package POLICY -- the
    'auto' token, which the batch path resolves to a number from the batch's own
    peaks (admission.derive_height_cutoff_x_edge) and a single-sample run reads
    as DEFAULT_HEIGHT_CUTOFF_X_EDGE (PassConfig.height_cutoff_x_edge_resolved).
    `None` at either level means UNSET and falls through -- 0.0 is a value, not
    an absence; 'auto' is a value too (an explicit request for the derivation,
    which outranks a profile's number)."""
    if explicit is not None:
        return _x_edge_value(explicit)
    from_profile = getattr(profile, "height_cutoff_x_edge", None)
    if from_profile is not None:
        return _x_edge_value(from_profile)
    from peaky.assignment.passes.config import AUTO_HEIGHT_CUTOFF_X_EDGE

    return AUTO_HEIGHT_CUTOFF_X_EDGE


def height_cutoff_x_edge_source(explicit: float | None = None,
                                profile: "ReagentProfile | None" = None) -> str:
    """Where `resolve_height_cutoff_x_edge` took its value from, for the run log
    and the batch summary -- so a reader can tell a profile-supplied multiple
    from the package default.

    This is the LABEL, not the value, and a run resolves more than once on the
    way down (the pipeline stamps the cfg, assign_batch re-resolves that same
    cfg): a second pass must not relabel the first. So an explicit value that
    merely repeats what it would have got anyway -- the profile's multiple, or
    the package default when no profile carries one -- keeps that credit."""
    from peaky.assignment.passes.config import is_auto_x_edge

    from_profile = getattr(profile, "height_cutoff_x_edge", None)
    if explicit is not None:
        same_as_profile = (from_profile is not None
                           and _x_edge_value(explicit) == _x_edge_value(from_profile))
        # the package policy is 'auto': an explicit 'auto' against a profile with
        # no opinion merely repeats what it would have got anyway
        same_as_default = from_profile is None and is_auto_x_edge(explicit)
        if not (same_as_profile or same_as_default):
            return ("an explicit --height-cutoff-x-edge auto"
                    if is_auto_x_edge(explicit)
                    else "an explicit --height-cutoff-x-edge / cfg value")
    if from_profile is not None:
        return f"the {getattr(profile, 'name', '?')} reagent profile"
    return "the package default (auto)"


def apply_height_cutoff_x_edge(cfg, profile: "ReagentProfile | None" = None, *,
                               explicit: float | None = None,
                               log=None) -> tuple[float, str]:
    """Stamp the resolved multiple onto a PassConfig; return (value, source).
    A cfg that ALREADY carries a multiple was set deliberately by its caller (or
    stamped by an earlier call on the way down), so it counts as explicit and
    outranks the profile -- INCLUDING a multiple that happens to equal the package
    default, which is why `PassConfig.height_cutoff_x_edge` is None when unset
    rather than pre-filled with the default: explicitness is read off the field,
    never guessed by comparing it to 1.0. `log` prints the one-line 'which gate,
    from where' record for the run -- skipped when an absolute `height_cutoff_cps`
    override is in force, since the multiple is then not what gates (assign.run
    reports that override itself)."""
    from peaky.assignment.passes.config import is_auto_x_edge

    if explicit is None:
        explicit = getattr(cfg, "height_cutoff_x_edge", None)
    value = resolve_height_cutoff_x_edge(explicit, profile)
    source = height_cutoff_x_edge_source(explicit, profile)
    cfg.height_cutoff_x_edge = value
    if log is not None and getattr(cfg, "height_cutoff_cps", None) is None:
        if is_auto_x_edge(value):
            log(f"[gate] height cutoff = auto: derived per batch from the persistence "
                f"table, 1x the sample's noise edge without one (from {source})")
        else:
            log(f"[gate] height cutoff = {value:g}x the sample's noise edge "
                f"(from {source})")
    return value, source


def apply_ion_only_channels(cfg, profile: "ReagentProfile | None" = None, *,
                            log=None) -> tuple:
    """Stamp the ion-only channels in force onto a PassConfig; return them.
    The same explicitness rule as `apply_height_cutoff_x_edge`: a cfg that
    ALREADY carries a tuple -- an empty one included -- was set deliberately by
    its caller (an A/B arm switching the bucket off against a profile that
    declares it) and outranks the profile, which is why
    `PassConfig.ion_only_channels` is None when unset rather than pre-filled
    with (). No profile (a context-only entry point, `--reagent auto` on the
    MCP path) resolves to nothing: the stage stays off."""
    cur = getattr(cfg, "ion_only_channels", None)
    if cur is None:
        cur = tuple(getattr(profile, "ion_only_channels", ()) or ()) if profile is not None else ()
        cfg.ion_only_channels = cur
        source = f"profile {profile.name}" if (profile is not None and cur) else "no profile channel"
    else:
        cur = tuple(cur)
        prof_t = tuple(getattr(profile, "ion_only_channels", ()) or ()) if profile is not None else ()
        # a tuple equal to the profile's was stamped from it by an earlier call on
        # the way down (pipeline -> assign_batch); only a DIFFERENT tuple is an override
        source = (f"profile {profile.name}" if (profile is not None and cur == prof_t)
                  else "explicit config")
    if log is not None and cur:
        log(f"[gate] ion-only channels {list(cur)} (from {source})")
    return cur


#: the word that closes every side channel on the command line (`--side-channels none`)
SIDE_CHANNELS_NONE = "none"


def _side_tuple(value) -> tuple:
    """A side-channel list as stored: a tuple of adduct labels, in order, once
    each. A bare string is one adduct (a config's `"side_channels": "[M+NH4]+"`),
    and the word `none` (any case) or an empty value is the empty tuple."""
    if value is None:
        return ()
    items = [value] if isinstance(value, str) else list(value)
    out: list[str] = []
    for a in items:
        a = str(a).strip()
        if not a or a.lower() == SIDE_CHANNELS_NONE:
            continue
        if a not in out:
            out.append(a)
    return tuple(out)


def side_channels_source(channels, profile: "ReagentProfile | None" = None) -> str:
    """Where a run's side channels came from, for the run log and the batch
    summary: the profile's own declaration, an explicit choice (a flag, a cfg
    field, a library caller), or nothing declared. A tuple equal to the
    profile's was stamped from it on the way down (pipeline -> assign_batch), so
    it keeps the profile's credit -- the `apply_ion_only_channels` rule."""
    cur = _side_tuple(channels)
    prof_t = _side_tuple(getattr(profile, "side_channels", ())) if profile is not None else ()
    if profile is not None and cur == prof_t:
        return f"profile {profile.name}" if cur else f"profile {profile.name} (declares none)"
    return "explicit config" if channels is not None else "no profile"


def apply_side_channels(cfg, profile: "ReagentProfile | None" = None, *,
                        explicit=None, log=None) -> tuple:
    """Stamp the side channels a run may open onto a PassConfig; return them.

    The resolution order of `apply_height_cutoff_x_edge`: an `explicit` value
    (the `--side-channels` flag; `()` or `none` closes them all) > a tuple the
    cfg ALREADY carries (set deliberately by its caller, or stamped by an
    earlier call on the way down -- an empty one included, which is why
    `PassConfig.side_channels` is None when unset) > the profile's own
    `side_channels` > nothing. No profile (a forced `--adducts` list, a
    context-only entry point) declares nothing, so every side channel stays
    closed unless asked for. assign.run opens a stamped channel only when the
    server resolves its mechanism (offline: the sample registers it), and only
    on the run's polarity."""
    if explicit is not None:
        cfg.side_channels = _side_tuple(explicit)
    cur = getattr(cfg, "side_channels", None)
    if cur is None:
        cur = _side_tuple(getattr(profile, "side_channels", ())) if profile is not None else ()
        source = side_channels_source(None if profile is None else cur, profile)
    else:
        cur = _side_tuple(cur)
        source = side_channels_source(cur, profile)
    cfg.side_channels = cur
    if log is not None:
        log(f"[gate] side channels {list(cur) if cur else 'closed'} (from {source})")
    return cur


def _merge_ranges(sources: list[str]) -> str:
    """Union of element boxes: widest [lo, hi] per element, first-seen order."""
    from peaky.chem import chemistry as C

    merged: dict[str, tuple[int, int]] = {}
    for s in sources:
        for el, (lo, hi) in C.parse_ranges(s).items():
            if el in merged:
                lo0, hi0 = merged[el]
                merged[el] = (min(lo0, lo), max(hi0, hi))
            else:
                merged[el] = (lo, hi)
    return " ".join(f"{el}{lo}-{hi}" for el, (lo, hi) in merged.items())


def compose(profiles: "list[ReagentProfile]") -> ReagentProfile:
    """Merge several reagent profiles into one, for a module running MORE THAN ONE
    reagent at once (e.g. a mixed nitrate/bromide inlet) or a labelled reagent whose
    unlabelled isotopologue is also present.

    This exists because a single-profile answer SILENTLY DROPS a whole ionization
    channel: the profile's `adducts` list is the only menu the passes ever see, so a
    mixed NO3/Br batch resolved to Br alone can never offer [M+NO3]- as a reading —
    every nitrate-clustered analyte is then missed outright or forced into a bromide
    interpretation, with nothing in the log to say so.

    Merge rules: adducts, declared side channels and reagent-ion regexes are
    unioned; the element box takes
    the widest bound per element; `normaliser` stays "reagent" only if every
    component agrees (a component that must normalise on TIC, because its reagent
    ions sit outside the acquisition window, forces TIC for the whole); a single
    labelled component carries its label/purity through. Composing across polarities
    is refused — that is two acquisitions, not one reagent system.
    """
    ps: list[ReagentProfile] = []
    for p in profiles:                       # dedupe, preserve order
        if p not in ps:
            ps.append(p)
    if not ps:
        raise ValueError("compose() needs at least one profile")
    if len(ps) == 1:
        return ps[0]
    pol = {p.polarity for p in ps}
    if len(pol) > 1:
        raise ValueError(
            "cannot compose reagent profiles of different polarity: "
            + ", ".join(f"{p.name}({p.polarity})" for p in ps))
    labelled = [p for p in ps if p.label_isotope]
    if len({p.label_isotope for p in labelled}) > 1:
        raise ValueError(
            "cannot compose two differently-labelled reagents: "
            + ", ".join(f"{p.name}({p.label_isotope})" for p in labelled))
    lab = labelled[0] if labelled else None
    adducts: list[str] = []
    for p in ps:
        for a in p.adducts:
            if a not in adducts:
                adducts.append(a)
    res = [p.reagent_ion_re for p in ps if p.reagent_ion_re]
    ctx = {p.context for p in ps}
    edge = {p.height_cutoff_x_edge for p in ps}
    # ion-only channels are unioned like the adduct menu: a component that opens
    # the electron-attachment bucket keeps it in the mix (NO3+Br, NO3+NO3_15N)
    ion_only: list[str] = []
    for p in ps:
        for a in (p.ion_only_channels or ()):
            if a not in ion_only:
                ion_only.append(a)
    # reagent-water cores are unioned too: a mixed NO3/Br inlet carries both ladders
    water: list[str] = []
    for p in ps:
        for c in (p.water_cores or ()):
            if c not in water:
                water.append(c)
    # declared side channels are unioned like the adduct menu: a component's own
    # side chemistry stays declared in the mix
    side: list[str] = []
    for p in ps:
        for a in (p.side_channels or ()):
            if a not in side:
                side.append(a)
    # the neutral pair survives only when every component that declares one
    # declares the same pair (two different pairs would be two rules)
    pairs = {tuple(p.neutral_pair) for p in ps if p.neutral_pair}
    return ReagentProfile(
        name="+".join(p.name for p in ps),
        label=" / ".join(p.label for p in ps),
        polarity=ps[0].polarity,
        adducts=adducts,
        normaliser="reagent" if all(p.normaliser == "reagent" for p in ps) else "tic",
        reagent_ion_re="|".join(f"(?:{r})" for r in res) if res else None,
        ranges=_merge_ranges([p.ranges for p in ps]),
        detect_adduct=None,               # composed profiles are never auto-detected
        context=ps[0].context if len(ctx) == 1 else "ambient-air",
        purity=lab.purity if lab else None,
        label_isotope=lab.label_isotope if lab else None,
        label_max=lab.label_max if lab else 2,
        height_cutoff_x_edge=next(iter(edge)) if len(edge) == 1 else None,
        ion_only_channels=tuple(ion_only),
        water_cores=tuple(water),
        neutral_pair=next(iter(pairs)) if len(pairs) == 1 else (),
        side_channels=tuple(side),
        aliases=(),
    )


# Channels almost every reagent system carries, so their presence identifies NO
# reagent: bare (de)protonation is offered by Br, NO3, iodide, uronium and EasyIC
# alike. A profile whose `detect_adduct` is one of these is treated as a WEAK
# signature whatever it declares -- otherwise it would attach itself to every run
# of the matching polarity. (The old first-match resolve() dodged this only by
# registry order, which is luck, not a rule.)
_GENERIC_DETECT = frozenset({"[M-H]-", "[M+H]+"})


def resolve(
    reagent: str = "auto", peaks=None, *, config: str | None = None
) -> ReagentProfile:
    """Return a ReagentProfile. `reagent` may be a name/alias, a '+'-joined
    combination ('NO3+Br'), or 'auto' to detect from a loaded peak table (its server
    adduct mechanisms; it raises when none is diagnostic). `config` (a JSON/TOML
    path) registers extra/override reagents before resolving.

    Auto-detect returns EVERY reagent system the peak table evidences, composed into
    one profile (see `compose`) -- not the first that happens to match. A module
    running a mixed inlet, and a labelled reagent whose unlabelled isotopologue is
    also present, both show two diagnostic adducts, and taking one of them silently
    discards the other channel's chemistry.

    Auto-detect never GUESSES. When none of the table's server mechanisms is a
    reagent's diagnostic adduct (no matches at all, or only generic ones such as
    [M+H]+ / [M-H]-), it raises ValueError naming the mechanisms it saw and the
    polarity they carry, and asks for an explicit reagent. Picking the first
    registered profile of the polarity instead handed a uronium batch the bromide
    profile, and a batch or sample NAME is never read: names carry dates and
    instrument tokens whose hyphens are not charges."""
    if config:
        load_config(config)
    if reagent and reagent.lower() in _BY_ALIAS:
        return _BY_ALIAS[reagent.lower()]
    if reagent and reagent != "auto" and "+" in reagent:
        keys = [k.strip().lower() for k in reagent.split("+") if k.strip()]
        if keys and all(k in _BY_ALIAS for k in keys):
            return compose([_BY_ALIAS[k] for k in keys])
    if reagent != "auto":
        raise KeyError(f"unknown reagent {reagent!r}; known: {sorted(_BY_ALIAS)}")
    if peaks is None:
        raise ValueError("reagent='auto' needs a peaks table to detect from")
    from peaky.io import io_mascope as IO

    # the channels the server's own matches name -- WITHOUT detect_adducts' [M-H]-
    # default, so a table with no matches cannot select a profile that declares it
    seen = set(IO.recognised_adducts(peaks))
    matched = [p for p in PROFILES.values() if p.detect_adduct and p.detect_adduct in seen]
    # Strong signatures name a specific reagent species and compose. Weak ones are
    # only believed when nothing strong matched (see ReagentProfile.detect_weak).
    def _weak(p):
        return p.detect_weak or p.detect_adduct in _GENERIC_DETECT

    hits = [p for p in matched if not _weak(p)] or matched
    if hits:
        return compose(hits)
    mechs = _mechanism_keys(peaks)
    pol = {"+": "positive", "-": "negative"}.get(_detect_polarity(peaks), "unknown")
    shown = ", ".join(mechs[:12]) + (" ..." if len(mechs) > 12 else "") if mechs else "none"
    signatures = ", ".join(f"{p.detect_adduct} ({p.name})"
                           for p in PROFILES.values() if p.detect_adduct)
    raise ValueError(
        "could not auto-detect reagent: no server match names a reagent's diagnostic "
        f"adduct (mechanisms seen: {shown}; polarity {pol}). Auto-detect reads these "
        f"signatures: {signatures}. Name the reagent instead (--reagent NAME; known: "
        f"{', '.join(PROFILES)}).")


def _mechanism_keys(peaks) -> list[str]:
    """The table's distinct server mechanisms in the standard notation
    ('-H+' and '[M-H]-' are one key), first-seen order; [] without the column."""
    if "ionization_mechanism" not in getattr(peaks, "columns", []):
        return []
    from peaky.io import io_mascope as IO

    out: list[str] = []
    for m in peaks["ionization_mechanism"].dropna().unique():
        k = IO._mechanism_key(m)
        if k and k not in out:
            out.append(k)
    return out


def _charge_sign(key: str) -> str | None:
    """The ion's charge sign of a mechanism in the standard notation, read after
    the closing bracket ('[M+Br]-' -> '-', '[M]+.' -> '+'); None for text that
    does not read as a mechanism. The legacy spelling's last character is NOT the
    charge ('-H+' is the negative [M-H]-), hence the notation first."""
    if not key.startswith("[") or "]" not in key:
        return None
    signs = {c for c in key.rsplit("]", 1)[1] if c in "+-"}
    return signs.pop() if len(signs) == 1 else None


_POLARITY_WORDS = {"+": "+", "positive": "+", "pos": "+",
                   "-": "-", "negative": "-", "neg": "-"}


def _detect_polarity(peaks) -> str | None:
    """'+' / '-' from the table's own server mechanisms (the charge each one
    carries), else from a 'polarity' column's words (positive / negative / + / -);
    None when neither says, or they name both polarities. The batch and sample
    names are never read: a dated or instrument-coded name is full of hyphens."""
    signs = {s for s in map(_charge_sign, _mechanism_keys(peaks)) if s}
    if signs:
        return signs.pop() if len(signs) == 1 else None
    if "polarity" in getattr(peaks, "columns", []):
        words = {_POLARITY_WORDS.get(str(v).strip().lower())
                 for v in peaks["polarity"].dropna().unique()}
        words.discard(None)
        if len(words) == 1:
            return words.pop()
    return None
