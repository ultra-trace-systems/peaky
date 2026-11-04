"""The sideband / isotopologue hold of a reading Assigned in <= 2 files (batch/sideband.py).

Synthetic two-file merges: each file's ledger is built here, merged by the vote
(assign_batch.align) as a batch merges them, and the merged ledger handed to
the hold with the files' ledgers and a synthetic batch series.
"""
import numpy as np
import pandas as pd

from peaky.batch import assign_batch as AB
from peaky.batch import sideband as SB
from peaky.chem import chemistry as C
from peaky.chem import isotopes as ISO

PARENT = ("C9H14O2", "[M+(CH4N2O)H]+")       # a bright urea adduct
SAT = ("C15H18O", "[M+H]+")                   # the reading a +3.85 mDa satellite of it takes
PMZ = C.ion_mz(*PARENT)
SMZ = PMZ + 3.85e-3
FILES = ("s01", "s02")


def _row(pid, mz, h, nf=None, ad=None, tier="Assigned", role="M0", **kw):
    return dict(peak_id=pid, mz=mz, height=h, neutral_formula=nf, adduct=ad, tier=tier, role=role,
                ion_score=0.9, tier_reason="", **kw)


def _ledgers(extra=None, files=FILES, sat_tier="Assigned"):
    """{sid: ledger}: the bright parent (2e6) and its satellite (1e5, 20x fainter) in every
    file, plus `extra(sid)` rows."""
    out = {}
    for k, sid in enumerate(files):
        rows = [_row(f"{sid}p", PMZ, 2e6, *PARENT),
                _row(f"{sid}s", SMZ + k * 1e-5, 1e5, *SAT, tier=sat_tier),
                _row(f"{sid}u", PMZ + 1.5, 3e3, role="unexplained", tier=None)]
        rows += (extra(sid) if extra else [])
        out[sid] = pd.DataFrame(rows)
    return out


def _merged(ledgers):
    merged, _jit = AB.align({sid: AB._m0(led) for sid, led in ledgers.items()}, tol_ppm=6.0)
    merged["role"] = "M0"
    return merged


def _series(ratio_noise=0.05, n=12, seed=1):
    """A batch series of n spectra: the parent ramps over two decades, the satellite
    follows it at 1/20 (log-normal scatter `ratio_noise`), or on its own when
    ratio_noise is None."""
    rng = np.random.default_rng(seed)
    hp = 10 ** np.linspace(4.5, 6.5, n)
    hs = hp / 20 * np.exp(rng.normal(0, ratio_noise, n)) if ratio_noise is not None \
        else 10 ** rng.uniform(3, 5.5, n)
    rows = []
    for k in range(n):
        rows += [dict(sample_item_id=f"t{k:02d}", mz=PMZ, height=hp[k]),
                 dict(sample_item_id=f"t{k:02d}", mz=SMZ, height=hs[k])]
    return pd.DataFrame(rows)


def _tier(merged, nf, ad):
    r = merged[(merged["neutral_formula"] == nf) & (merged["adduct"] == ad)]
    assert len(r) == 1, merged[["mz", "neutral_formula", "adduct", "tier"]]
    return r.iloc[0]


def test_a_satellite_of_a_20x_brighter_parent_that_tracks_it_is_held():
    led = _ledgers()
    merged = _merged(led)
    assert _tier(merged, *SAT)["tier"] == "Assigned"                 # the vote's verdict, two files
    tab, summ = SB.hold_low_file_promotions(merged, led, _series(), log=lambda *a: None)
    r = _tier(merged, *SAT)
    assert r["tier"] == "Candidate" and SB.HOLD_MARK in r["tier_reason"]
    assert summ["held"] == 1 and summ["satellite"] == 1
    assert sorted(tab["sample_item_id"]) == list(FILES) and set(tab["rule"]) == {"satellite"}
    assert set(tab["partner_peak_id"]) == {"s01p", "s02p"}
    assert (tab["d_mDa"].between(3.8, 3.9)).all() and (tab["ratio"] >= 19).all()
    assert _tier(merged, *PARENT)["tier"] == "Assigned"              # the parent itself stands


def test_the_same_row_with_its_own_13c_line_stays_assigned():
    nc = 15                                                          # C15H19O+: the satellite reading's carbons

    def own13c(sid):
        h = 1e5 * ISO.R_13C_PER_C * nc
        return [_row(f"{sid}c13", SMZ + ISO.D_13C, h, role="iso_child", tier=None)]

    led = _ledgers(own13c)
    merged = _merged(led)
    tab, summ = SB.hold_low_file_promotions(merged, led, _series(), log=lambda *a: None)
    assert _tier(merged, *SAT)["tier"] == "Assigned"
    assert summ["held"] == 0 and summ["own_13c_kept"] == 1 and not len(tab)


def test_a_13c_line_of_a_brighter_row_at_1_1x_its_prediction_is_held():
    # the row sits at the 13C spacing of a brighter Assigned M0 and reads 1.1x the
    # line that M0's carbon count predicts: it can be that line
    par = ("C10H16O3", "[M+H]+")
    pmz = C.ion_mz(*par)
    hp = 4e5
    pred = hp * ISO.R_13C_PER_C * 10
    row = ("C6H14N2O3", "[M+H]+")                                    # a reading of the 13C line's mass
    led = {sid: pd.DataFrame([_row(f"{sid}p", pmz, hp, *par),
                              _row(f"{sid}r", pmz + ISO.D_13C, 1.1 * pred, *row)]) for sid in FILES}
    merged = _merged(led)
    tab, summ = SB.hold_low_file_promotions(merged, led, None, sigma_ppm={s: 0.2 for s in FILES},
                                            log=lambda *a: None)
    r = _tier(merged, *row)
    assert r["tier"] == "Candidate" and "isotopologue 13C" in r["tier_reason"]
    assert summ["isotopologue"] == 1 and set(tab["rule"]) == {"isotopologue 13C"}
    # at 2x its prediction the row is brighter than that line can be: it stands
    led2 = {sid: f.assign(height=np.where(f["peak_id"].str.endswith("r"), 2.0 * pred, f["height"]))
            for sid, f in led.items()}
    merged2 = _merged(led2)
    SB.hold_low_file_promotions(merged2, led2, None, log=lambda *a: None)
    assert _tier(merged2, *row)["tier"] == "Assigned"


def test_a_partner_the_row_does_not_track_holds_nothing():
    led = _ledgers()
    merged = _merged(led)
    _tab, summ = SB.hold_low_file_promotions(merged, led, _series(ratio_noise=None), log=lambda *a: None)
    assert _tier(merged, *SAT)["tier"] == "Assigned" and summ["held"] == 0


def test_a_reading_assigned_in_three_files_is_not_tested():
    files = ("s01", "s02", "s03")
    led = _ledgers(files=files)
    merged = _merged(led)
    _tab, summ = SB.hold_low_file_promotions(merged, led, _series(), log=lambda *a: None)
    assert _tier(merged, *SAT)["tier"] == "Assigned"
    assert summ["tested"] == 0 and summ["held"] == 0                 # three files each: neither row is tested


def test_the_satellite_rule_is_an_orbitrap_rule():
    led = _ledgers()
    merged = _merged(led)
    _tab, summ = SB.hold_low_file_promotions(merged, led, _series(), orbitrap=False, log=lambda *a: None)
    assert _tier(merged, *SAT)["tier"] == "Assigned" and summ["held"] == 0
