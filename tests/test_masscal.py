"""Offline tests for the mass-dependent calibration centre (peaky/assignment/
masscal.py): the acceptance rule on a flat source, the backbone-range clamp,
and the end-to-end path calibrate() -> PassConfig -> compute_tiers on a real
1/mz trend (every earlier test hand-set the trend on the cfg)."""
import numpy as np
import pandas as pd

from peaky.assignment import ledger as L, masscal as MC, tiers as T
from peaky.assignment.passes import config as PCfg, core as PC
from peaky.chem import chemistry as C


def _flat_acceptances(n_seeds: int, n_rows: int = 30) -> int:
    """How many of `n_seeds` FLAT (trend-free) samples grow a trend. Rows are
    spread uniformly in 1/mz so BOTH halves of the fit range are populated and
    the STATISTICAL rule, not the coverage guard, is what decides."""
    accepted = 0
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        mz = 1000.0 / rng.uniform(1000 / 420, 1000 / 60, n_rows)
        ppm = -0.3 + rng.normal(0, 0.2, n_rows)
        if MC.fit_mass_trend(mz, ppm, min_n=20, sigma_floor=0.15) is not None:
            accepted += 1
    return accepted


def test_flat_source_keeps_the_constant_model():
    # 30 rows, sigma 0.2 ppm, m/z 60-420, no trend. The old |b| > 2*SE rule is a
    # 5 % two-sided test, so a flat source grew a phantom trend in ~6-7 % of
    # samples at ANY n (Monte Carlo, 21-300 rows): 2/50 and ~14/200 here. The
    # 3-SE + variance-ratio rule must stay under 2 %.
    assert _flat_acceptances(50) <= 1, "flat sources grew a trend"       # <= 2 %
    # the 50-seed budget is 1 sample wide, so confirm the RATE on a wider sweep
    wide = _flat_acceptances(200)
    assert wide <= 4, f"{wide}/200 flat sources grew a trend"            # <= 2 %


def test_a_single_corroborated_outlier_is_a_lever_not_a_trend():
    # 20 near-noise-free points at m/z 150-169 + ONE point at m/z 480 that is
    # -1.0 ppm off -- the shape of the tests/test_rearbitrate.py calibration core.
    # The least-squares line passes THROUGH the lever (its residual is ~0, so
    # trimming never removes it and the variance ratio even improves), and the
    # slope is significant because the lever alone carries Sxx: all three
    # statistical conditions pass. It used to fit b = +0.22 mDa, i.e. a centre of
    # +2.25 ppm at m/z 61 out of 21 rows that never went below m/z 150. Only the
    # coverage guard rejects it.
    mz = np.array([150.0 + i for i in range(20)] + [480.2])
    ppm = np.array([[-0.10, 0.0, 0.10, 0.05, -0.05][i % 5] for i in range(20)] + [-1.0])
    assert MC.fit_mass_trend(mz, ppm, min_n=20, sigma_floor=0.15) is None


def test_centre_is_clamped_to_the_backbone_coverage():
    rng = np.random.default_rng(1)
    mz = rng.uniform(150, 480, 60)
    ppm = -0.2 - 0.3 * 1000 / mz + rng.normal(0, 0.1, 60)
    fit = MC.fit_mass_trend(mz, ppm, min_n=20, sigma_floor=0.1)
    assert fit is not None and abs(fit.b + 0.3) < 0.05
    assert 150 <= fit.mz_lo < 160 and 470 < fit.mz_hi <= 480
    # below the backbone the centre is the constant model at the low edge ...
    c61 = MC.centre(fit.a, fit.b, 61, fit.mz_lo, fit.mz_hi)
    assert c61 == MC.centre(fit.a, fit.b, fit.mz_lo, fit.mz_lo, fit.mz_hi)
    assert c61 == MC.centre(fit.a, fit.b, 150, fit.mz_lo, fit.mz_hi)
    # ... not the 1/mz extrapolation (which would be > 2 ppm further out)
    assert abs(MC.centre(fit.a, fit.b, 61) - c61) > 2.0
    # and above it likewise
    assert MC.centre(fit.a, fit.b, 900, fit.mz_lo, fit.mz_hi) == \
        MC.centre(fit.a, fit.b, fit.mz_hi, fit.mz_lo, fit.mz_hi)
    # the same through PassConfig (cal_center / z_of) and the tier _Cal (_cal_z)
    cfg = PCfg.PassConfig()
    cfg.cal_mu, cfg.cal_sigma = -0.8, 0.25
    cfg.cal_a, cfg.cal_b, cfg.cal_sigma_trend = fit.a, fit.b, fit.sigma
    cfg.cal_mz_lo, cfg.cal_mz_hi = fit.mz_lo, fit.mz_hi
    assert PC.cal_center(cfg, 61) == PC.cal_center(cfg, 150)
    cal = T._Cal(-0.8, 0.25, fit.a, fit.b, fit.sigma, fit.mz_lo, fit.mz_hi,
                 abs_floor_mda=cfg.cal_abs_floor_mda)
    assert T._cal_z(cal, c61, 61) == 0.0
    # the two consumers agree exactly (one implementation)
    for m in (61, 150, 300, 480, 900):
        assert abs(abs(T._cal_z(cal, -1.0, m)) - PC.z_of(-1.0, cfg, m)) < 1e-12
    # without a range no clamp: the pre-existing hand-set-trend behaviour
    cfg.cal_mz_lo = cfg.cal_mz_hi = None
    assert abs(PC.cal_center(cfg, 61) - MC.centre(fit.a, fit.b, 61)) < 1e-12


# --- end-to-end: a corroborated CHO backbone on a real -0.12 mDa trend ---------
_BACKBONE = (["C2H2O2", "C3H6O", "C4H8O", "C3H6O2", "C4H6O2", "C5H10O", "C4H8O2",
              "C5H8O2", "C6H12O", "C5H10O2"]                      # the light half
             + [f"C{k}H{2 * k}O2" for k in range(8, 27)]
             + ["C10H16O3", "C12H20O4", "C15H26O3", "C18H30O4", "C20H36O3", "C22H40O4"])


def _trend_ledger(probe_ppm: float, *, a=-0.25, b=-0.12, noise=0.08, seed=3):
    """35 iso-backed High CHO [M+H]+ rows (m/z 59-397) whose ppm errors follow
    ppm = a + b*1000/mz, plus an UNCORROBORATED acetic-acid [M+H]+ probe at
    m/z 61.028 at `probe_ppm`."""
    rng = np.random.default_rng(seed)
    rows, commits = [], []
    for i, nf in enumerate(_BACKBONE):
        mz = C.ion_mz(nf, "[M+H]+")
        pid, cid = f"bb{i}", f"bb{i}c"
        rows += [(pid, mz, 1e5), (cid, mz + 1.003355, 1e4)]
        commits.append((pid, cid, nf, a + b * 1000 / mz + rng.normal(0, noise)))
    probe_mz = C.ion_mz("C2H4O2", "[M+H]+")
    rows.append(("probe", probe_mz, 5e4))
    led = L.new_ledger(pd.DataFrame(rows, columns=["peak_id", "mz", "height"]))
    for pid, cid, nf, ppm in commits:
        L.commit_assignment(led, pid, neutral_formula=nf, adduct="[M+H]+",
                            ion_formula=C.format_formula(
                                {**C.parse_formula(nf), "H": C.parse_formula(nf)["H"] + 1}) + "+",
                            ion_score=0.95, compound_score=0.95, eff_score=0.93,
                            eff_margin=0.3, tied=False, ppm_error=ppm, pass_no=1,
                            method="cheminfo+grid", confidence="High", commentary="backbone",
                            isotopologues=[{"label": "13C", "peak_id": cid}])
        L.attach_isotopologue(led, cid, pid, iso_label="13C", iso_match_score=0.9)
    L.commit_assignment(led, "probe", neutral_formula="C2H4O2", adduct="[M+H]+",
                        ion_formula="C2H5O2+", ion_score=0.85, compound_score=0.85,
                        eff_score=0.85, eff_margin=0.3, tied=False, ppm_error=probe_ppm,
                        pass_no=1, method="cheminfo+grid", confidence="Good",
                        commentary="probe")
    return led, probe_mz


def test_calibrate_fits_the_trend_and_the_tier_gate_judges_at_the_peaks_mz():
    led, probe_mz = _trend_ledger(-2.2)
    cfg = PCfg.PassConfig()
    assert PC.calibrate(led, cfg, log=lambda *a: None) is not None
    assert cfg.cal_b is not None and abs(cfg.cal_b + 0.12) < 0.02, cfg.cal_b
    assert cfg.cal_mz_lo is not None and 59 <= cfg.cal_mz_lo <= 61
    assert cfg.cal_mz_hi is not None and 390 <= cfg.cal_mz_hi <= 400
    # the constant centre sits near the heavy end of the backbone; the trend
    # centre at m/z 61 is ~-2.2 ppm
    assert abs(PC.cal_center(cfg, probe_mz) + 2.2) < 0.3
    assert PC.z_of(-2.2, cfg, probe_mz) < 1.0 < PC.z_of(-2.2, cfg)

    # the tier engine calibrates itself from the same corroborated core ...
    m0 = led[led["role"] == L.ROLE_M0]
    kids = led.loc[led["role"] == L.ROLE_ISO, "parent_peak_id"].value_counts()
    cal = T._calibrate(m0, kids, abs_floor_mda=cfg.cal_abs_floor_mda)
    assert cal is not None and cal.b is not None and abs(cal.b + 0.12) < 0.02
    # ... and the constant-only reading of the same core would demote the probe
    assert abs(T._cal_z(T._Cal(cal[0], cal[1]), -2.2, probe_mz)) > T.Z_TAIL_DEMOTE

    # on-trend at -2.2 ppm: Assigned; off-trend at +1.0 ppm: Candidate
    tiers = T.compute_tiers(led, cfg=cfg).set_index("peak_id")
    assert tiers.at["probe", "tier"] == T.TIER_ASSIGNED, tiers.at["probe", "tier_reason"]
    led2, _ = _trend_ledger(+1.0)
    tiers2 = T.compute_tiers(led2, cfg=cfg).set_index("peak_id")
    assert tiers2.at["probe", "tier"] == T.TIER_CANDIDATE
    assert "mass error" in str(tiers2.at["probe", "tier_reason"])
    # the backbone itself is untouched
    assert (tiers.loc[[f"bb{i}" for i in range(len(_BACKBONE))], "tier"] == T.TIER_ASSIGNED).all()
