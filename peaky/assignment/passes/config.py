"""passes.config — split from the former passes.py monolith."""

from __future__ import annotations

from dataclasses import dataclass


__all__ = [
    "PassConfig",
]


@dataclass
class PassConfig:
    ppm: float = 1.0  # user m/z trust
    # Grid-enumeration tolerance. The measured instrument accuracy is
    # sigma~0.35 ppm (self-calibration), so 5 ppm was ~14 sigma and enumerated
    # a large candidate cloud the calibrated z-gate then rejected -- pure wasted
    # scoring (and bigger match_compounds requests that time out on a flaky
    # server). 3 ppm is still ~8 sigma: safely past local calibration drift,
    # but ~1.7x fewer candidates. Enumeration only (a formula never gridded
    # can never be scored); match_compounds keeps its 5 ppm window so it still
    # attributes real 29Si/81Br satellites, and the z-gate owns ppm rejection.
    search_ppm: float = 3.0  # grid enumeration tolerance
    height_cutoff: float = 100.0
    limit_per_peak: int = 25
    workers: int = 12
    # confidence thresholds (on the RAW min(ion,compound) score)
    tau_high: float = 0.90
    tau_good: float = 0.80
    tau_low: float = 0.70
    tau_suspect: float = 0.50
    complexity_cap: float = 0.20
    require_iso_for_high: bool = True
    series_ppm: float = 3.0
    series_min_score: float = 0.60
    series_max_iter: int = 3  # iterative GKA: chain confirmed members as anchors
    # Pass 4 (residual explainer) acceptance policy: <=strict ppm on score
    # alone; up to pattern ppm ONLY with pattern evidence (confirmed isotope
    # partner / >=2 series anchors). DBE-only plausibility in pass 4.
    residual_ppm_strict: float = 1.0
    residual_ppm_pattern: float = 4.0
    residual_max_steps: int = 2
    # explicit ionization-mechanism ids for match_compounds. None = server
    # auto-selects the sample's configured channels; set by assign.run to the
    # sample's channels PLUS extras like +CO3- so background air-ion adducts
    # get scored too.
    mechanism_ids: list | None = None
    # enumeration: the local grid is the primary, reliable candidate source.
    # cheminfo is an optional best-effort enrichment (compound names) and is the
    # flaky/slow dependency, so it is OFF by default in the search path.
    use_cheminfo: bool = False
    # isotopologue gating: a heteroatom in the NEUTRAL must be backed by its
    # diagnostic isotope confirmed by Mascope, else the candidate is penalised.
    # Cl/Br satellites are large (always visible if real) -> strong penalty;
    # 34S is small (4.4%) -> softer penalty.
    het_iso_penalty_halogen: float = 0.30
    het_iso_penalty_S: float = 0.12
    # Si has a genuine, diagnostic 29Si(4.7%)+30Si(3.1%) M+1/M+2 pattern (a real
    # multi-Si siloxane confirms easily; a single-Si 29Si sits near the server
    # isotope floor). A bare-Si mass fit with NO confirmed satellite is almost
    # always a mass coincidence (PDMS/silanol formulas are dense in mass space),
    # so an unconfirmed Si takes a SOFT gate on top of the complexity prior --
    # sized like 34S, not the halogens, so a real-but-faint single-Si is not
    # nuked outright (the tiers-level Si demote is the belt-and-suspenders catch).
    het_iso_penalty_Si: float = 0.12
    # The reagent halogen (e.g. Br in Br-CIMS) is special: its heavy isotope in
    # the ION cannot prove the halogen sits in the NEUTRAL (covalent X(Br)[M-H]-
    # and Y.HBr.Br- / Y[M+Br]- aliases share the ion). Confirmation therefore
    # waives only the gate penalty, never the complexity prior, so the
    # adduct/cluster interpretation wins ties. Set by assign.run.
    reagent_element: str | None = None
    # Self-calibration mass gate (ROADMAP 1): mu/sigma of the ppm error fitted
    # on the pass-1 High/Good CHO-CHON backbone (set by assign.run via
    # calibrate()). A candidate is judged by z = |ppm - mu| / sigma:
    # z <= cal_z_accept on score alone; up to cal_z_pattern only WITH pattern
    # evidence (confirmed isotopologue or series membership); beyond that the
    # best fit within tolerance is just the closest of many -- reject. A match
    # with NO ppm at all carries no mass evidence and is never committed.
    cal_mu: float | None = None
    cal_sigma: float | None = None
    # mass-dependent centre ppm = cal_a + cal_b * 1000/mz (masscal.fit_mass_trend;
    # cal_b is the constant absolute offset in mDa). None = constant model. Used
    # by z_of ONLY when the caller passes the peak's m/z; cal_sigma_trend is the
    # residual sigma of that model (floored like cal_sigma).
    cal_a: float | None = None
    cal_b: float | None = None
    cal_sigma_trend: float | None = None
    # absolute floor (mDa) on the trend sigma, active only where it exceeds the
    # ppm sigma (below ~m/z 120 at sigma 0.25 ppm): the Orbitrap's residual at
    # the low-mass edge curves faster than 1/mz (46.065 sat 0.9 ppm = 0.04 mDa off
    # the fitted trend), and 0.03 mDa is the absolute-accuracy floor the backbone
    # itself shows (mDa MAD 0.01-0.05 across the range).
    cal_abs_floor_mda: float = 0.03
    # rough mass offset (ppm) seeded from the sample's own matches BEFORE the
    # pass-1 self-calibration, so the pre-calibration pass-0 known-species gate is
    # not blind to a large systematic instrument offset (set by assign.run).
    prior_offset: float = 0.0
    cal_z_accept: float = 2.0
    cal_z_pattern: float = 4.0
    cal_sigma_floor: float = 0.25  # don't let a lucky tight fit reject everything
    cal_min_n: int = 20  # min backbone size to trust a fit
    # Channel priors: the reagent / deprotonation channels are PRIMARY; the
    # background air-ion channels (carbonate, superoxide, electron attachment)
    # are MINOR -- real but rare, and offering them to every peak doubles the
    # alias space. A minor-channel candidate pays a ranking penalty (so a
    # near-tie goes to the primary channel) and a minor-channel WINNER may only
    # commit with corroboration: a Good+ score, series-evidence method, or the
    # same neutral independently assigned via a primary channel.
    minor_channels: tuple = ("[M+CO3]-", "[M+O2]-", "[M]-.")
    minor_channel_penalty: float = 0.12
    # Reference-list selection prior: a candidate neutral on an ACTIVE reference
    # peaklist (a published product of the sample's chemistry, or a known
    # contaminant) is far more likely real than a mass-coincidence monster of
    # similar score. Add a small TIE-BREAK bonus to its eff_score -- enough to win
    # a near-tie (gap < the 0.05 tie window), never enough to override a clearly
    # better isotope-scored fit. Empty set / 0.0 -> no-op. Set by assign.run from
    # the run's context-active reference lists (reflists.active_lists).
    reflist_formulas: frozenset = frozenset()
    reflist_prior: float = 0.04
