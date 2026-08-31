"""Reagent profiles — ONE config per reagent system, replacing the adducts /
element-ranges / normaliser / label constants that were copy-pasted inline across
the time-series, clustering and validation scripts.

A profile is everything the pipeline needs to treat a batch's reagent correctly.
New reagent = add a ReagentProfile, not edit code. `resolve()` picks one by name
or auto-detects from a loaded peak table (polarity + the server's own adduct
mechanisms via io_mascope.detect_adducts).
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
    # isotopic purity of a labelled reagent (0.98 = 98% 15N); threaded to local
    # scoring's predict_isotopes for '^X' adducts
    purity: float | None = None
    # labelled-reagent covalent-product rescue (labeled.py): the caret heavy
    # isotope a product can carry ('^N' = 15N organonitrate) and the max count.
    # None => no rescue (every unlabelled profile).
    label_isotope: str | None = None
    label_max: int = 2
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
    aliases=("ur", "uronium", "urea", "urea-cims", "ur+"),
)

# NO3⁻ (nitrate) CIMS — PROVISIONAL built-in; validate + refine for your instrument
# (or override via a --reagent-config file). Negative mode; highly oxygenated
# molecules detected as the [M+NO3]⁻ cluster (and [M-H]⁻ when acidic). Reagent ions
# are the NO3⁻ / (HNO3)ₙ·NO3⁻ cluster series.
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
    aliases=("no3", "nitrate", "no3-", "nitrate-cims"),
)

# ¹⁵N-labelled nitrate CIMS (server reagent '^NO3-'). Same chemistry as NO3 above,
# but the cluster adduct is the heavy [M+¹⁵NO3]⁻ = [M+^NO3]- (+62.9855, mechanism
# '+^NO3-'); the deprotonation channel [M-H]- is isotope-independent. Reagent
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
# (KORBI2 EasyIC+ batches, 2026). Three ionization channels:
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
    adducts=["[M]+.", "[M-H]+", "[M+H]+"],
    normaliser="tic",
    reagent_ion_re=None,
    ranges="C0-40 H0-80 N0-5 O0-15 S0-2",
    detect_adduct="[M]+.",
    context="easyic",
    aliases=("easyic", "easy-ic", "easyic+", "fluoranthene", "charge-transfer"),
)

PROFILES: dict[str, ReagentProfile] = {
    BR.name: BR,
    UR.name: UR,
    NO3.name: NO3,
    NO3_15N.name: NO3_15N,
    IODIDE.name: IODIDE,
    EASYIC.name: EASYIC,
}
_BY_ALIAS = {a: p for p in PROFILES.values() for a in (p.name.lower(), *p.aliases)}


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
    "aliases",
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
    return ReagentProfile(**kw)


def load_config(path: str) -> list:
    """Register reagent profiles from a JSON or TOML file (so users add reagents
    without editing the package). Accepts a top-level list of entries, a
    `{"reagents": [...]}` wrapper, or a `{name: {fields...}}` mapping. Each entry
    carries the ReagentProfile fields (name/label/polarity/adducts/normaliser/
    reagent_ion_re/ranges/detect_adduct, + optional context/aliases)."""
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


def resolve(
    reagent: str = "auto", peaks=None, *, config: str | None = None
) -> ReagentProfile:
    """Return a ReagentProfile. `reagent` may be a name/alias, or 'auto' to detect
    from a loaded peak table (its server adduct mechanisms, then polarity). `config`
    (a JSON/TOML path) registers extra/override reagents before resolving."""
    if config:
        load_config(config)
    if reagent and reagent.lower() in _BY_ALIAS:
        return _BY_ALIAS[reagent.lower()]
    if reagent != "auto":
        raise KeyError(f"unknown reagent {reagent!r}; known: {sorted(_BY_ALIAS)}")
    if peaks is None:
        raise ValueError("reagent='auto' needs a peaks table to detect from")
    from peaky.io import io_mascope as IO

    seen = set(IO.detect_adducts(peaks))
    for p in PROFILES.values():
        if p.detect_adduct in seen:
            return p
    # fall back on polarity if no diagnostic adduct matched
    pol = _detect_polarity(peaks)
    for p in PROFILES.values():
        if p.polarity == pol:
            return p
    raise ValueError(f"could not auto-detect reagent (adducts={seen}, polarity={pol})")


def _detect_polarity(peaks) -> str | None:
    for col in ("polarity", "sample_batch_name", "ionization_mechanism"):
        if col in getattr(peaks, "columns", []):
            s = " ".join(map(str, peaks[col].dropna().unique()[:20]))
            if "+" in s and "-" not in s:
                return "+"
            if "-" in s and "+" not in s:
                return "-"
    return None
