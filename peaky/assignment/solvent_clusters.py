"""Source-solvent CLUSTER ions (positive mode) -- the non-covalent ladders a CI
source builds out of its own solvent vapour.

A low-pressure positive source that carries solvent vapour (ethanol, methanol,
acetone, and the water that is always there) does not only protonate single
molecules: it builds proton-bound and hydride-bound CLUSTERS of them, and those
clusters can be the brightest ions in the spectrum. On a 2026-09-21 EasyIC+
acquisition the ethanol dimer (C2H5OH)2H+ at m/z 93.0911 carried 6.5 % of the
TOTAL signal and ~half of everything the run left in role=unexplained.

peaky could not reach it, and must not be taught to:

  * the implied covalent neutral of [(C2H6O)2+H]+ is C4H12O2, whose DBE is -1.
    The grid's chemistry gate (`chemistry.dbe_ok`) rejects it, and that gate is
    RIGHT -- C4H12O2 is not a molecule. The ion is a CLUSTER: two intact
    ethanols sharing one proton. Relaxing the gate to let the grid "find" it
    would open the whole DBE<0 corner of formula space to every other peak.

So the cluster ions get their own channel, exactly the way the reagent's own
cluster ions (`chem/reagents.py`) and the pass-0 known species
(`passes/directors.py`) do: an explicit, enumerated family, committed on
evidence, locked, and kept OFF the covalent grid.

What is enumerated, per source solvent S (and mixtures of them):

  [S_n + H]+       proton-bound ladder, n = 2..MAX_UNITS
                   (EtOH)2H+ 93.0911, EtOH.H3O+ 65.0597, (EtOH)3H+ 139.1117 ...
  [S_n - H]+       hydride-bound ladder -- the EasyIC [M-H]+ channel SOLVATED:
                   the charge carrier is [S-H]+ (ethanol's only monomer channel
                   in that source, C2H5O+ 45.0335) with n-1 intact solvents
                   hanging off it. [EtOH-H]+.EtOH = 91.0754.
                   Enumerated only where the context actually has a hydride
                   channel ([M-H]+ in its reagent_adducts) -- EasyIC, not the
                   urea / ammonium sources.
  ... - H2O        the in-source CONDENSATION rung: a cluster of two H-bonded
                   solvents loses water. 91.0754 - H2O = 73.0648 (C4H9O+).
                   A child rung: it commits only behind a committed parent.

What licenses a commit (all of them, per rung):

  1. MONOMER EVIDENCE. Every non-water constituent must show its own monomer
     ion ([S+H]+ or [S-H]+) in this spectrum, above the height gate, carrying
     at least MIN_MONOMER_FRAC of the spectrum's total peak height. A source
     solvent is a major ion by definition; a trace one is not a cluster former.
     Water is the exception (`Solvent.anchor=False`): H3O+ sits at m/z 19.018,
     below every acquisition window these sources use, so a water rung is
     licensed by its SPACING off an anchored rung instead.
  2. A CONTIGUOUS LADDER. The rung below -- the same cluster minus one solvent
     (for the condensation rung: the same cluster before the water loss) --
     must be an OBSERVED peak, and the observed rung-to-rung spacing must be
     that solvent's exact neutral mass to within LADDER_PPM. The contiguity is
     what carries the weight: no observed rung below, no commit, so a ladder
     can only ever grow up from a monomer this spectrum actually shows. The
     spacing tolerance on top is a consistency check -- it is what says the two
     peaks really are one solvent apart rather than two separate ions that each
     happened to land inside their own mass window.
  3. MASS. |ppm - cfg.prior_offset| <= MASS_PPM, the pass-0 gate.
  4. INTENSITY SANITY. A rung may not tower over the monomers it would be built
     from (MAX_RUNG_X_MONOMER), and a condensation rung may not tower over its
     own parent cluster (MAX_CONDENSATION_RATIO) -- that second ratio is what
     separates a real water loss from a genuine analyte sitting on the same
     mass, and it is why C4H9O+ commits as the ethanol dimer's condensation
     product on one EasyIC run and stays a covalent C4H8O on another.
  5. ...and for a rung that would DISPLACE a legal covalent reading, two more:
     the rung has to be a major ion of the spectrum (ALIAS_MIN_RUNG_FRAC), and
     it may not be a WATER rung. Overruling a nameable molecule is a stronger
     claim than explaining an unexplained ion, so it takes a stronger fact --
     and a water rung is the family's weakest evidence, not its strongest: it
     is anchored by nothing, so the whole claim is an 18.0106 step, and that is
     the commonest mass difference a spectrum has. `_commit_rung` carries the
     measurement that settled it.

What it reports. The MEANINGFUL neutral is the solvent, not the cluster (the
ammonia-via-urea-adduct and easyic-hydride rulings): [(C2H6O)2+H]+ is committed
as C2H6O on the adduct `[M+(C2H6O)H]+`, ion formula C4H13O2+. Repeated units are
written out -- `[M+(C2H6O)(C2H6O)H]+`, not `(C2H6O)2` -- because that is the
notation `tiers._ion_counts` already parses, so the ion composition of a cluster
row is readable by the same grammar as every other row.

Two evidence classes, and they are tiered differently:

  * no covalent reading EXISTS. The same ion read as [M+H]+ / [M-H]+ of a
    neutral gives DBE < 0 (every saturated-alcohol / water cluster is like
    this). Nothing else can explain the peak -> `known:solvent_cluster`,
    Assigned, like any other pass-0 known species.
  * a covalent reading exists. [(C2H6O)2-H]+ is C4H11O2+, which also reads as
    protonated C4H10O2 (a butanediol); acetone.EtOH.H+ is C5H13O2+, which also
    reads as protonated C5H12O2. MS1 cannot separate a cluster from its covalent
    isomer -- the codebase already rules that the ADDUCT reading wins such a
    split (`tiers._drop_decomposition_aliases`, 2026-06-11) -- so the cluster
    reading is committed, but as `cluster:solvent` at Candidate with the
    covalent reading recorded in `alternatives`. Honest, visible, not locked to
    a claim MS1 cannot make.

Both classes are LOCKED: the point of running before pass 1 is that the grid
never gets to re-read a cluster mass as a covalent molecule.

Rows this family does NOT claim -- a broken ladder, an absent monomer, a
condensation rung 30x its parent, a trace or water rung that is not entitled to
overrule a molecule -- keep their covalent reading and get the cluster-vs-covalent dual
note from `cleanup.annotate_easyic_ambiguity` instead (`covalent_alias_notes`
below builds it from this same library). That is not a fallback for a broken
run: it is how the same C4H9O+ can be the ethanol dimer's condensation product
on a solvent-soaked acquisition and protonated butanone on a headspace sample
that happens to contain butanone.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, replace

import pandas as pd

from peaky.chem import chemistry as C
from peaky.assignment import ledger as L
from peaky.assignment import tiers as T

__version__ = "0.1.0"

# charge-carrier shifts, written the way chemistry.ADDUCT_SHIFTS writes them
PROTON = C.M["H"] - C.M_E          # [M+H]+  : gain a proton
HYDRIDE = -C.M["H"] - C.M_E        # [M-H]+  : lose a hydride (H + one electron)
WATER = C.M["O"] + 2 * C.M["H"]

#: carrier key -> (mass shift, the monomer adduct it corresponds to)
CARRIERS = {"H": (PROTON, "[M+H]+"), "-H": (HYDRIDE, "[M-H]+")}


@dataclass(frozen=True)
class Solvent:
    """A source solvent: the neutral that clusters, not an analyte."""

    formula: str          # Hill neutral formula
    name: str
    #: does it have to prove itself by its OWN monomer ion? Water does not --
    #: H3O+ (19.018) is below every window these sources acquire, so a water
    #: rung is licensed by the exact spacing off an anchored rung instead.
    anchor: bool = True

    @property
    def mass(self) -> float:
        return C.neutral_mass(self.formula)


#: The catalog. Deliberately SMALL: every entry is a solvent an ion source
#: actually carries in bulk, and each one it does not carry is dead weight that
#: only adds in-family mass degeneracy (2x ethanol and acetone+methanol are the
#: same ion composition -- `build_library` collapses such pairs on evidence, but
#: the fewer of them the better). A context opts in by listing formulas in
#: `ContextProfile.source_solvents`; nothing here is on by default.
SOURCE_SOLVENTS: tuple[Solvent, ...] = (
    Solvent("H2O", "water", anchor=False),
    Solvent("CH4O", "methanol"),
    Solvent("C2H6O", "ethanol"),
    Solvent("C3H6O", "acetone"),
)
_BY_FORMULA = {s.formula: s for s in SOURCE_SOLVENTS}

MAX_UNITS = 4          # solvent molecules per cluster (n=1 is the monomer, never ours)
MAX_WATER = 1          # waters per cluster -- water is UNANCHORED (no in-window
                       # monomer of its own), so each water rung rests on nothing
                       # but its spacing off the rung below; one such step is a
                       # hydrate, two is speculation
MASS_PPM = 2.0         # |ppm - prior_offset|, the pass-0 known-species gate
LADDER_PPM = 4.0       # rung-to-rung spacing vs the solvent's exact neutral mass.
                       # Loose ON PURPOSE relative to MASS_PPM: both rungs are
                       # already inside their own 2 ppm windows, so this can only
                       # bite in the corner where they sit at opposite edges --
                       # which is a peak-picking blend, not a wrong ladder. What
                       # rejects a wrong ladder is the rung having to EXIST.
MIN_MONOMER_FRAC = 0.002    # a source solvent carries >=0.2 % of the total signal
#: Backstop, not a law of nature: a stable proton-bound dimer routinely outshines
#: the protonated monomer it came from (it does on the run this family was built
#: for), so the cap has headroom. What it is here to catch is the other thing --
#: a bright ANALYTE sitting on a cluster mass in a spectrum whose solvent is only
#: just over MIN_MONOMER_FRAC. The claim itself rests on the ladder, not on this.
MAX_RUNG_X_MONOMER = 3.0
MAX_CONDENSATION_RATIO = 5.0  # a -H2O rung <= this x its parent cluster
#: Extra floor for the rungs that would DISPLACE a legal covalent reading.
#: Explaining an otherwise-unexplained ion and overruling a nameable molecule are
#: not the same claim, so they do not get the same evidence bar: a rung may take a
#: peak off the covalent grid only when the cluster is itself a MAJOR ion of this
#: spectrum, not a trace peak that merely happens to sit one solvent above
#: another. Measured: on an ethanol-soaked run the displacing rungs are ~1 % of
#: total signal each, while the trace coincidences an earlier draft was claiming
#: in a gin headspace -- C5H10O2 / C6H12O2 esters at 100-150 cps, which are real
#: analytes there -- are all below 0.07 %. Rungs with NO covalent reading keep the
#: ordinary height gate: there is nothing to displace, their alternative is
#: 'unexplained'.
ALIAS_MIN_RUNG_FRAC = 0.001


# ---------------------------------------------------------------------------
# what a context asks for
# ---------------------------------------------------------------------------
def solvents_for(profile) -> tuple[Solvent, ...]:
    """The source solvents this experimental context declares
    (`ContextProfile.source_solvents`), in catalog order. Empty -> the family is
    a no-op, which is every negative-mode and every un-opted-in context."""
    want = set(getattr(profile, "source_solvents", ()) or ())
    return tuple(s for s in SOURCE_SOLVENTS if s.formula in want)


def carriers_for(profile) -> tuple[str, ...]:
    """Which charge carriers the source can build a cluster around. The proton
    always; the HYDRIDE only where the context actually has a hydride-abstraction
    channel ([M-H]+ in its reagent_adducts -- EasyIC's charge-transfer source,
    not the urea / ammonium reagents, which protonate and do not abstract H-)."""
    ad = tuple(getattr(profile, "reagent_adducts", ()) or ())
    return ("H", "-H") if "[M-H]+" in ad else ("H",)


# ---------------------------------------------------------------------------
# step 1 -- which solvents is this spectrum actually carrying?
# ---------------------------------------------------------------------------
def _peaks_near(ledger: pd.DataFrame, mz: float, *, ppm: float, offset: float = 0.0):
    """Row indices within `ppm` of `mz`, judged against the run's rough mass
    offset (the pass-0 convention: a uniformly shifted instrument must not lose
    its known species)."""
    if "mz" not in ledger.columns or not len(ledger):
        return []
    err = (ledger["mz"] - mz) / mz * 1e6 - offset
    return list(ledger.index[err.abs() <= ppm])


def detect_solvents(ledger: pd.DataFrame, solvents, carriers, *,
                    min_frac: float = MIN_MONOMER_FRAC, ppm: float = MASS_PPM,
                    offset: float = 0.0, height_cutoff: float = 0.0,
                    log=print) -> dict:
    """Which anchor solvents this spectrum carries, by their OWN monomer ions.

    Returns ``{formula: {"height", "frac", "channels": [(adduct, mz, height)]}}``
    for each anchor solvent whose monomer channels together clear `min_frac` of
    the spectrum's total peak height. Non-anchor solvents (water) are never in
    the result -- they carry no in-window monomer to detect -- and are admitted
    to a cluster by the ladder-spacing rule instead.

    Presence of the PEAK is the test, not what some pass made of it: on one
    EasyIC file ethanol's hydride ion was read as protonated acetaldehyde (the
    carbonyl/alcohol degeneracy `cleanup.annotate_easyic_ambiguity` documents),
    and the solvent is no less present for that."""
    total = float(pd.to_numeric(ledger.get("height"), errors="coerce").sum()) \
        if "height" in ledger.columns else 0.0
    out: dict = {}
    for s in solvents:
        if not s.anchor:
            continue
        channels = []
        for carrier in carriers:
            shift, adduct = CARRIERS[carrier]
            mz = s.mass + shift
            hits = [i for i in _peaks_near(ledger, mz, ppm=ppm, offset=offset)
                    if _height(ledger, i) >= height_cutoff]
            if not hits:
                continue
            i = max(hits, key=lambda j: _height(ledger, j))
            channels.append((adduct, float(ledger.at[i, "mz"]), _height(ledger, i)))
        if not channels:
            continue
        h = sum(c[2] for c in channels)
        frac = (h / total) if total else 0.0
        if frac < min_frac:
            log(f"[solvent] {s.name} monomer present but weak "
                f"({frac * 100:.3f} % of total signal < {min_frac * 100:.1f} %) "
                "-- not treated as a source solvent")
            continue
        out[s.formula] = {"height": h, "frac": frac, "channels": channels,
                          "name": s.name}
    return out


# ---------------------------------------------------------------------------
# step 2 -- the cluster library
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ClusterIon:
    units: tuple          # constituent solvent formulas, Hill-sorted, len >= 2
    carrier: str          # 'H' | '-H'
    water_loss: int       # 0 | 1  (the in-source condensation rung)
    core: str             # the solvent reported as the neutral
    adduct: str           # '[M+(C2H6O)H]+' -- core is M, the rest is the adduct
    ion_formula: str      # 'C4H13O2+'
    mz: float
    covalent: tuple       # same-ion covalent readings ((neutral, adduct), ...)
    aliases: tuple = ()   # other solvent multisets with this exact composition

    @property
    def n_units(self) -> int:
        return len(self.units)

    def label(self) -> str:
        """Human label: '[(C2H6O)2+H]+', '[(C2H6O)(H2O)+H]+', '[(C2H6O)2-H-H2O]+'."""
        seen: dict[str, int] = {}
        for u in self.units:
            seen[u] = seen.get(u, 0) + 1
        body = "".join(f"({u}){n if n > 1 else ''}" for u, n in sorted(seen.items()))
        tail = "+H" if self.carrier == "H" else "-H"
        return f"[{body}{tail}{'-H2O' * self.water_loss}]+"


def _counts(units, carrier: str, water_loss: int) -> dict:
    """Element counts of the cluster ION (charge implied, electron mass is not
    an element)."""
    cnt: dict[str, int] = {}
    for u in units:
        for el, n in C.parse_formula(u).items():
            cnt[el] = cnt.get(el, 0) + n
    cnt["H"] = cnt.get("H", 0) + (1 if carrier == "H" else -1)
    if water_loss:
        cnt["H"] = cnt.get("H", 0) - 2
        cnt["O"] = cnt.get("O", 0) - 1
    return {k: v for k, v in cnt.items() if v}


def _adduct_label(units, core: str, carrier: str, water_loss: int) -> str:
    """The peaky adduct string for a cluster whose reported neutral is `core`.

    Repeated units are written out -- `[M+(C2H6O)(C2H6O)H]+` -- because
    `tiers._ion_counts` flattens the parentheses and lets `parse_formula` sum the
    repeats; a `(C2H6O)2` shorthand would silently parse as C2H6O2 and lose an
    ethanol from every ion composition derived from the label."""
    rest = list(units)
    rest.remove(core)
    body = "".join(f"({u})" for u in sorted(rest))
    tail = "H]+" if carrier == "H" else "-H]+"
    return f"[M+{body}{tail}" if not water_loss else f"[M+{body}{tail[:-2]}-H2O]+"


def _adduct_delta(adduct: str) -> dict | None:
    """Element counts an ion form ADDS to its neutral ('[M-CH3]+' -> {C:-1, H:-3}),
    read with the same grammar the ledger's adduct labels are read by
    (`tiers._ion_counts`) rather than a second parser of this module's own -- a
    one-carbon stand-in neutral goes in, its own carbon comes back out."""
    cnt = T._ion_counts("C", adduct)
    if cnt is None:
        return None
    cnt = dict(cnt)
    cnt["C"] = cnt.get("C", 0) - 1
    return {k: v for k, v in cnt.items() if v}


def _covalent_readings(cnt: dict, channels) -> tuple:
    """The same ion read as a COVALENT neutral, over every SINGLE-MOLECULE ion
    form the source has: ((neutral, adduct), ...).

    Empty when no such neutral is a legal closed-shell molecule -- which is the
    whole point of this family: `[(C2H6O)2+H]+` would need C4H12O2, DBE -1, and
    there is nothing else the peak can be.

    `channels` is the context's own `reagent_adducts`, not a fixed pair, because
    which readings compete is the SOURCE's property and it grows: the EasyIC
    profile gained `[M-CH3]+` (the methylsiloxanes' quantifier channel), which
    makes C4H9O+ also the methyl loss of pentanol. A reagent-CLUSTER channel in
    that list ([M+NH4]+, the uronium adduct) needs the reagent's own atoms in the
    ion and so drops out on the non-negativity test, without being special-cased;
    `[M]+.` drops out because a neutral equal to the whole ion is a radical."""
    out = []
    for adduct in channels:
        delta = _adduct_delta(adduct)
        if delta is None:
            continue
        neutral = dict(cnt)
        for el, n in delta.items():
            neutral[el] = neutral.get(el, 0) - n
        if any(v < 0 for v in neutral.values()):
            continue
        neutral = {k: v for k, v in neutral.items() if v}
        if not neutral or not C.dbe_ok(neutral)[0] or not C.oxygen_ok(neutral)[0]:
            continue
        out.append((C.format_formula(neutral), adduct))
    return tuple(out)


def _pick_core(units, detected) -> str | None:
    """Which constituent is reported as the NEUTRAL: the heaviest one with its
    own monomer evidence (water never qualifies -- it has none to have). None
    when the multiset has no anchored member at all, which disqualifies it."""
    anchored = [u for u in units if u in detected]
    if not anchored:
        return None
    return max(anchored, key=lambda u: (C.neutral_mass(u), detected[u]["height"], u))


def _evidence_key(units, detected):
    """Ranking of one solvent multiset against another of the SAME ion
    composition (2x ethanol vs acetone+methanol). Best first: everything
    anchored, then the weakest constituent as strong as possible, then the
    fewest distinct solvents (parsimony), then alphabetical for determinism."""
    anchored = all(u in detected or not _BY_FORMULA[u].anchor for u in units)
    weakest = min((detected[u]["height"] for u in units if u in detected),
                  default=0.0)
    return (not anchored, -weakest, len(set(units)), units)


def build_library(detected: dict, solvents, carriers, *, channels=(),
                  max_units: int = MAX_UNITS, condensation: bool = True
                  ) -> list[ClusterIon]:
    """Every cluster rung the detected solvents can build, ordered the way it
    must be committed: smallest ladder rung first, condensation rungs last (a
    rung is licensed by the rung below it, so the parent has to exist already).

    Multisets that share one ion composition are COLLAPSED to the best-evidenced
    one (`_evidence_key`), the rest recorded as `aliases`: they are the same
    peak at the same mass and only one of them can own it. On a spectrum with a
    27 %-of-signal ethanol and no in-window methanol, 93.0911 resolves to the
    ethanol dimer and records acetone+methanol as the alias it beat.

    Every rung's adduct label is registered with `chemistry.register_adduct`, so
    `ion_mz(core, adduct)` reproduces the rung's mass from the committed row.

    `channels` are the source's own single-molecule ion forms
    (`ContextProfile.reagent_adducts`), against which each rung's COVALENT
    alternatives are worked out; it defaults to the cluster carriers alone, which
    is the minimum any source with those carriers has."""
    channels = tuple(channels) or tuple(CARRIERS[c][1] for c in carriers)
    pool = [s.formula for s in solvents if s.formula in detected or not s.anchor]
    monomer_ions = {C.format_formula(_counts((f,), carrier, 0)) + "+"
                    for f in pool for carrier in carriers}
    best: dict[str, tuple] = {}            # ion key -> (evidence key, ClusterIon)
    for n in range(2, max_units + 1):
        for combo in itertools.combinations_with_replacement(sorted(pool), n):
            core = _pick_core(combo, detected)
            if core is None:
                continue                    # water-only cluster: no anchor, not ours
            if sum(1 for u in combo if not _BY_FORMULA[u].anchor) > MAX_WATER:
                continue
            for carrier in carriers:
                # the condensation rung is a two-solvent H-bonded pair losing its
                # bridging water. It stays at n=2: a 3+ cluster would have to lose
                # water from a rung the ladder rule cannot point at.
                for water_loss in ((0, 1) if condensation and n == 2 else (0,)):
                    cnt = _counts(combo, carrier, water_loss)
                    if cnt.get("H", 0) < 1 or min(cnt.values()) < 0:
                        continue
                    ion = C.format_formula(cnt) + "+"
                    # never enumerate an ion that IS a monomer channel. A water
                    # rung that then loses water is the molecule it started
                    # from -- [(C3H6O)(H2O)-H-H2O]+ is exactly [C3H6O-H]+,
                    # 57.0335 -- and claiming it as a cluster would steal the
                    # monomer this family needs as its own anchor.
                    if ion in monomer_ions:
                        continue
                    key = _evidence_key(combo, detected)
                    prev = best.get(ion)
                    if prev is not None and prev[0] <= key:
                        best[ion] = (prev[0], replace(          # the sitting reading wins
                            prev[1], aliases=prev[1].aliases + (combo,)))
                        continue
                    label = _adduct_label(combo, core, carrier, water_loss)
                    shift = (sum(C.neutral_mass(u) for u in combo)
                             - C.neutral_mass(core)
                             + CARRIERS[carrier][0] - water_loss * WATER)
                    C.register_adduct(label, shift)
                    cluster = ClusterIon(
                        units=tuple(combo), carrier=carrier, water_loss=water_loss,
                        core=core, adduct=label, ion_formula=ion,
                        mz=C.neutral_mass(cnt) - C.M_E,
                        covalent=_covalent_readings(cnt, channels),
                        aliases=(() if prev is None
                                 else prev[1].aliases + (prev[1].units,)))
                    best[ion] = (key, cluster)
    lib = [c for _k, c in best.values()]
    lib.sort(key=lambda c: (c.water_loss, c.n_units, c.mz))
    return lib


# ---------------------------------------------------------------------------
# step 3 -- commit the rungs the ladder supports
# ---------------------------------------------------------------------------
def _parent_rung(cluster: ClusterIon, detected: dict, committed: dict):
    """The rung directly below `cluster`, as
    ``(removed_solvent, mz, height, description)``, or None when the ladder is
    broken there.

    For a condensation rung that is the un-dehydrated cluster, which must be one
    THIS run committed. Otherwise it is the cluster minus one solvent: a rung
    already committed here, or -- when that leaves a single molecule -- that
    solvent's own observed monomer ion on the same carrier. Any one removal that
    lands on an observed peak is enough; the spacing test (in `_commit_rung`) is
    what makes it a ladder rather than a coincidence."""
    if cluster.water_loss:
        dry = C.format_formula(_counts(cluster.units, cluster.carrier, 0)) + "+"
        hit = committed.get(dry)
        return ("H2O", hit[1], hit[2], f"the committed {hit[3]}") if hit else None
    for unit in sorted(set(cluster.units)):
        rest = list(cluster.units)
        rest.remove(unit)
        if len(rest) == 1:
            solvent = rest[0]
            info = detected.get(solvent)          # water is never the last one left
            want = CARRIERS[cluster.carrier][1]
            for adduct, mz, h in (info or {}).get("channels", ()):
                if adduct == want:
                    return (unit, mz, h, f"the observed {solvent} {adduct}")
        else:
            sub = C.format_formula(_counts(tuple(rest), cluster.carrier, 0)) + "+"
            hit = committed.get(sub)
            if hit:
                return (unit, hit[1], hit[2], f"the committed {hit[3]}")
    return None


def assign_solvent_clusters(ledger: pd.DataFrame, profile, cfg, *,
                            max_units: int = MAX_UNITS,
                            min_monomer_frac: float = MIN_MONOMER_FRAC,
                            log=print) -> dict:
    """Claim the source-solvent cluster ladders. Offline: the ion masses are
    exact by construction and the evidence is the ladder, so no scorer is asked
    (and none could be -- the neutral it would have to be given does not exist).

    Runs straight after pass 0 and LOCKS what it commits, so pass 1 never gets to
    re-read a cluster mass as a covalent molecule. Claims UNEXPLAINED peaks only.
    """
    out = {"committed": 0, "locked": 0, "assigned_tier": 0, "candidate_tier": 0,
           "solvents": {}}
    solvents = solvents_for(profile)
    if not solvents:
        log("[solvent] no source solvents declared for context "
            f"{getattr(profile, 'label', '?')!r}; skipping")
        return out
    carriers = carriers_for(profile)
    offset = float(getattr(cfg, "prior_offset", 0.0) or 0.0)
    hcut = cfg.height_cutoff      # resolved gate (raises if unresolved: fail closed)
    total = float(pd.to_numeric(ledger.get("height"), errors="coerce").sum()) \
        if "height" in ledger.columns else 0.0
    detected = detect_solvents(ledger, solvents, carriers, min_frac=min_monomer_frac,
                               offset=offset, height_cutoff=hcut, log=log)
    out["solvents"] = {f: round(d["frac"], 5) for f, d in detected.items()}
    if not detected:
        log("[solvent] no source-solvent monomer is present and strong enough in "
            "this spectrum; no cluster ladder can be licensed")
        return out
    log("[solvent] source solvents: "
        + ", ".join(f"{d['name']} {d['frac'] * 100:.2f} % of signal "
                    f"({'/'.join(c[0] for c in d['channels'])})"
                    for d in detected.values()))
    committed: dict[str, tuple] = {}     # ion_formula -> (pid, mz, height, label)
    for cluster in build_library(detected, solvents, carriers, max_units=max_units,
                                 channels=getattr(profile, "reagent_adducts", ())):
        claimed = _commit_rung(ledger, cluster, detected, committed,
                               offset=offset, hcut=hcut, total=total, out=out,
                               log=log)
        if claimed is not None:
            committed[cluster.ion_formula] = claimed
    log(f"[solvent] committed {out['committed']} solvent-cluster ion(s) "
        f"({out['assigned_tier']} with no possible covalent reading, "
        f"{out['candidate_tier']} with a same-ion covalent alternative)")
    return out


def _commit_rung(ledger, cluster: ClusterIon, detected, committed, *,
                 offset, hcut, total, out, log=print):
    """Gate and commit one rung; returns (peak_id, mz, height, label) or None."""
    hits = [i for i in _peaks_near(ledger, cluster.mz, ppm=MASS_PPM, offset=offset)
            if str(ledger.at[i, "role"]) == L.ROLE_UNEXPLAINED
            and not _locked(ledger, i)
            and _height(ledger, i) >= hcut]
    if not hits:
        return None
    i = max(hits, key=lambda j: _height(ledger, j))
    mz, height = float(ledger.at[i, "mz"]), _height(ledger, i)
    ppm = (mz - cluster.mz) / cluster.mz * 1e6

    parent = _parent_rung(cluster, detected, committed)
    if parent is None:
        return None
    unit, parent_mz, parent_h, parent_what = parent
    step = -WATER if cluster.water_loss else C.neutral_mass(unit)
    gap_ppm = ((mz - parent_mz) - step) / cluster.mz * 1e6
    if abs(gap_ppm) > LADDER_PPM:
        log(f"[solvent] skip {cluster.label()} @{mz:.4f}: spacing to {parent_what} "
            f"is {mz - parent_mz:+.5f} Da, {gap_ppm:+.1f} ppm off exact "
            f"{'-H2O' if cluster.water_loss else unit} -- not a ladder rung")
        return None

    monomer_total = sum(detected[u]["height"] for u in set(cluster.units)
                        if u in detected)
    if monomer_total and height > MAX_RUNG_X_MONOMER * monomer_total:
        log(f"[solvent] skip {cluster.label()} @{mz:.4f}: {height:.3g} cps is "
            f"{height / monomer_total:.0f}x every monomer it would be built from "
            f"({monomer_total:.3g} cps) -- an analyte on this mass, not a cluster "
            "of them")
        return None
    # A WATER rung may never displace a covalent reading, however bright. Every
    # other unit is anchored by its own monomer ion AND steps by a rare mass
    # difference: on an orange-peel headspace spectrum 64 of 833 peaks (7.7 %)
    # had a +C2H6O partner within 4 ppm. Water is anchored by nothing -- the
    # whole claim is its 18.0106 step -- and that step is the commonest
    # difference there is: 218 of those same 833 peaks (26 %) had a +H2O
    # partner. Unguarded it cost 11 of 18 files an Assigned C3H6O2 [M+H]+,
    # displaced by an acetone-water cluster whose "monomer anchor" at 57.0335
    # was itself committed as protonated acrolein. A water rung with NO covalent
    # reading is still welcome -- EtOH.H3O+ (65.0597) implies a DBE -1 neutral,
    # so there is no molecule to lose; it is only displacement water cannot buy.
    if cluster.covalent and any(not _BY_FORMULA[u].anchor for u in cluster.units):
        log(f"[solvent] skip {cluster.label()} @{mz:.4f}: a water rung rests on its "
            "18.0106 step alone (water has no monomer of its own), which is the "
            "commonest mass difference in a spectrum -- not enough to overrule "
            + " / ".join(f"{f} {a}" for f, a in cluster.covalent))
        return None
    if cluster.covalent and total and height < ALIAS_MIN_RUNG_FRAC * total:
        log(f"[solvent] skip {cluster.label()} @{mz:.4f}: {height:.3g} cps is "
            f"{height / total * 100:.3f} % of the spectrum, too small to overrule "
            "the covalent reading "
            + " / ".join(f"{f} {a}" for f, a in cluster.covalent))
        return None
    if cluster.water_loss and parent_h and height > MAX_CONDENSATION_RATIO * parent_h:
        log(f"[solvent] skip {cluster.label()} @{mz:.4f}: {height / parent_h:.0f}x its "
            f"parent cluster ({parent_h:.3g} cps) -- too bright to be that parent's "
            "water loss; left for the covalent reading")
        return None

    basis = (f"rung {cluster.n_units} of the {detected[cluster.core]['name']} ladder, "
             f"spacing to {parent_what} at m/z {parent_mz:.4f} is "
             f"{mz - parent_mz:+.5f} Da (exact {unit}, {gap_ppm:+.1f} ppm)")
    if cluster.water_loss:
        basis = (f"in-source condensation (water loss) off {parent_what} at m/z "
                 f"{parent_mz:.4f}, {gap_ppm:+.1f} ppm from exact -H2O")
    evidence = "; ".join(
        f"{detected[u]['name']} monomer {detected[u]['frac'] * 100:.2f} % of signal"
        for u in sorted(set(cluster.units)) if u in detected)
    alias_note = ("; same composition as "
                  + ", ".join("+".join(a) for a in cluster.aliases)
                  + ", resolved here by monomer evidence") if cluster.aliases else ""
    if cluster.covalent:
        method, tier_hint = "cluster:solvent", "candidate_tier"
        conf = "Low (solvent cluster, covalent alias)"
        verdict = ("; the same ion also reads as "
                   + " / ".join(f"{f} {a}" for f, a in cluster.covalent)
                   + " -- MS1 cannot separate a solvent cluster from its covalent "
                     "isomer, so the cluster reading is committed (a same-ion split "
                     "goes to the adduct reading) but held at Candidate with the "
                     "covalent reading recorded as an alternative")
    else:
        method, tier_hint = "known:solvent_cluster", "assigned_tier"
        conf = "Good (solvent cluster)"
        impossible = _implied_neutral(cluster.ion_formula)
        verdict = (f"; NOT a covalent molecule -- the implied neutral {impossible} "
                   f"has DBE {C.dbe(impossible):g}, so the grid's chemistry gate "
                   "correctly cannot reach this ion")
    try:
        L.commit_assignment(
            ledger, ledger.at[i, "peak_id"],
            neutral_formula=cluster.core, adduct=cluster.adduct,
            ion_formula=cluster.ion_formula, ion_score=0.0, ppm_error=ppm,
            pass_no=0, method=method, confidence=conf,
            alternatives=[{"formula": f, "adduct": a,
                           "note": "same-ion covalent reading of a solvent cluster"}
                          for f, a in cluster.covalent],
            commentary=(f"Source-solvent cluster {cluster.label()} = "
                        f"{cluster.ion_formula} at {ppm:+.2f} ppm: {basis}; "
                        f"{evidence}{alias_note}{verdict}."))
    except L.LedgerError:
        return None
    pid = ledger.at[i, "peak_id"]
    L.lock_peaks(ledger, [pid])
    out["committed"] += 1
    out["locked"] += 1
    out[tier_hint] += 1
    return (pid, mz, height, cluster.label())


def _height(ledger, i) -> float:
    h = ledger.at[i, "height"] if "height" in ledger.columns else None
    return float(h) if h is not None and pd.notna(h) else 0.0


def _locked(ledger, i) -> bool:
    return ("locked" in ledger.columns
            and bool(pd.notna(ledger.at[i, "locked"]) and ledger.at[i, "locked"]))


def _implied_neutral(ion_formula: str) -> str:
    """The neutral a [M+H]+ reading of this ion would need -- the formula whose
    DBE is negative, which is the reason the grid cannot (and must not) reach
    the cluster."""
    cnt = C.parse_formula(str(ion_formula).rstrip("+-"))
    cnt["H"] = cnt.get("H", 0) - 1
    return C.format_formula(cnt)


# ---------------------------------------------------------------------------
# step 4 -- the rows this family did NOT claim
# ---------------------------------------------------------------------------
#: monomer-detection window for the ANNOTATION layer. Wider than the commit
#: gate on purpose: nothing is claimed there, and the caller
#: (`cleanup.annotate_easyic_ambiguity`) runs post-tier with no PassConfig to
#: read the run's mass offset from, so the window has to absorb it.
ANNOTATE_PPM = 6.0


def covalent_alias_notes(ledger: pd.DataFrame, profile, *,
                         min_monomer_frac: float = MIN_MONOMER_FRAC,
                         max_units: int = MAX_UNITS, ppm: float = ANNOTATE_PPM,
                         log=lambda *_a: None) -> dict:
    """``{row index: note}`` for committed rows whose ION is a solvent-cluster
    ion of a solvent this spectrum demonstrably carries.

    The commit layer above claims a rung only with the full ladder behind it.
    When the ladder is broken -- the rung below was not picked, a condensation
    ion is far too bright for its parent, the family was never run -- the peak
    keeps its covalent reading, and that reading is then exactly as unfalsifiable
    as the carbonyl-vs-alcohol pair `cleanup.annotate_easyic_ambiguity` already
    notes: same ion, same isotopes, different molecule. This builds that note
    from the same library, so the two layers cannot describe the chemistry
    differently."""
    solvents = solvents_for(profile)
    need = ("neutral_formula", "adduct", "role", "method")
    if not solvents or not set(need) <= set(ledger.columns):
        return {}
    carriers = carriers_for(profile)
    detected = detect_solvents(ledger, solvents, carriers, ppm=ppm,
                               min_frac=min_monomer_frac, log=log)
    if not detected:
        return {}
    lib = {c.ion_formula: c for c in build_library(
        detected, solvents, carriers, max_units=max_units,
        channels=getattr(profile, "reagent_adducts", ()))}
    notes: dict = {}
    for i in ledger.index:
        if str(ledger.at[i, "role"]) != L.ROLE_M0:
            continue
        method = ledger.at[i, "method"]
        method = "" if pd.isna(method) else str(method)
        if method.startswith(("known:solvent_cluster", "cluster:solvent")):
            continue          # already READ as a cluster; it says so itself
        cnt = T._ion_counts(ledger.at[i, "neutral_formula"], ledger.at[i, "adduct"])
        if not cnt:
            continue
        cluster = lib.get(C.format_formula(cnt) + "+")
        if cluster is None:
            continue
        who = " + ".join(_BY_FORMULA[u].name for u in cluster.units)
        notes[i] = (
            f"cluster-vs-covalent ambiguity: this ion is also the source-solvent cluster "
            f"{cluster.label()} ({who}), and "
            + "; ".join(f"{detected[u]['name']} carries "
                        f"{detected[u]['frac'] * 100:.2f} % of this spectrum's signal"
                        for u in sorted(set(cluster.units)) if u in detected)
            + " -- a proton/hydride-bound cluster and its covalent isomer are the "
              "same ion, so MS1 cannot split them. The solvent-cluster channel did "
              "not claim this peak (its ladder evidence was incomplete, or the "
              "channel is off for this run), so the covalent reading stands and the "
              "cluster reading is recorded beside it")
    return notes
