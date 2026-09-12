"""passes.config — split from the former passes.py monolith."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from peaky.assignment import masscal


__all__ = [
    "PassConfig",
    "noise_edge",
]

NOISE_EDGE_Q = 0.01   # the sample's noise edge = this quantile of its picked heights


def noise_edge(heights, q: float = NOISE_EDGE_Q) -> float | None:
    """The sample's peak-picker detection edge: the `q` quantile (1st percentile)
    of its picked peak heights. This is the only height scale that transfers
    between instruments and modes -- measured 0.8 cps on a TOF, 9-11 cps on
    Orbitrap EasyIC modes, 140-180 cps on a urea 122-600 mode and 640-870 cps
    with the reagent ion in range (an ~1000x spread, stable to 2-16 % within one
    mode across weeks). Every height threshold in the passes is a multiple of it.
    None when there are no finite heights."""
    h = np.asarray(heights, dtype=float)
    h = h[np.isfinite(h)]
    if h.size == 0:
        return None
    return float(np.percentile(h, 100.0 * q))


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
    # Height gate for the height-gated passes (ladders, siloxane, residual,
    # reflist rescue, isotope-satellite checks). Expressed as a MULTIPLE of the
    # sample's own noise edge (`noise_edge`: the 1st percentile of its picked
    # heights, stamped onto `noise_edge_cps` by assign.run) -- an absolute cps
    # value was a no-op on modes whose edge sits above it and blinded the passes
    # on modes whose edge sits far below it (EasyIC: 87 % of picked peaks under
    # the old 100 cps; TOF: 97 %). 1.0 = every picked peak but the bottom 1 % is
    # eligible. `height_cutoff_cps` is an explicit ABSOLUTE override (offline
    # callers / tests); when set it wins. Read the resolved value via the
    # `height_cutoff` property.
    height_cutoff_x_edge: float = 1.0
    height_cutoff_cps: float | None = None
    noise_edge_cps: float | None = None   # runtime: set per sample by assign.run
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
    # residual sigma of that model (floored like cal_sigma). cal_mz_lo/hi are
    # the backbone's m/z coverage: outside it the centre is held at the nearest
    # edge (masscal.centre), never a 1/mz extrapolation onto masses the fit
    # never saw (a 150-480 backbone would put +2.25 ppm on m/z 61).
    cal_a: float | None = None
    cal_b: float | None = None
    cal_sigma_trend: float | None = None
    cal_mz_lo: float | None = None
    cal_mz_hi: float | None = None
    # absolute floor (mDa) on the trend sigma, active only where it exceeds the
    # ppm sigma (below ~m/z 120 at sigma 0.25 ppm) -- see masscal.ABS_FLOOR_MDA
    # for the physics. This field is the single runtime owner of the value; the
    # tier engine reads it via apply_tiers(cfg=) / _calibrate(abs_floor_mda=).
    cal_abs_floor_mda: float = masscal.ABS_FLOOR_MDA
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

    # Fields assign.run stamps onto the shared cfg PER SAMPLE at runtime. They are
    # run-derived, not user knobs, so the reproducibility fingerprint
    # (provenance.build_manifest) drops exactly these -- declared here, next to
    # the fields, so a new runtime field is added in one place. A ClassVar, not a
    # dataclass field: asdict()/pickle/deepcopy are unaffected.
    RUNTIME_FIELDS: ClassVar[tuple[str, ...]] = (
        "mechanism_ids", "prior_offset", "reagent_element", "noise_edge_cps",
        # passes.calibrate fits these onto the shared cfg during a run: they are
        # the calibration's OUTPUT, not user knobs, so two identical re-runs must
        # not fingerprint differently because a different sample finished last.
        "cal_a", "cal_b", "cal_sigma_trend", "cal_mz_lo", "cal_mz_hi")

    @property
    def height_cutoff(self) -> float:
        """The resolved height gate in cps: the absolute override if given, else
        `height_cutoff_x_edge` x the sample's noise edge. FAILS CLOSED: before
        assign.run has stamped an edge (and without an override) there is no
        gate to resolve, and returning 0 here would silently un-gate every
        height-gated pass -- so this raises instead."""
        if self.height_cutoff_cps is not None:
            return float(self.height_cutoff_cps)
        if self.noise_edge_cps is not None:
            return float(self.height_cutoff_x_edge) * float(self.noise_edge_cps)
        raise RuntimeError(
            "height gate unresolved: assign.run stamps noise_edge_cps from the "
            "sample's picked heights; offline callers pass "
            "PassConfig(height_cutoff_cps=...)")
