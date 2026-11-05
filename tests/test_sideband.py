"""The isotopologue hold and the satellite note of a reading Assigned in <= 2 files (batch/sideband.py).

Synthetic merges: each file's ledger is built here and merged by the vote
(assign_batch.align) as a batch merges them -- or, where the test needs a
merged row the vote would not make, written by hand -- and handed to the gate
with the files' ledgers and a synthetic batch series. The last test runs the
gate inside a whole (offline) batch. Every rule parameter is pinned: each test
names the parameter it fails without.
"""
import json

import numpy as np
import pandas as pd

from peaky.assignment import evidence as EV
from peaky.assignment import ledger as L
from peaky.batch import assign_batch as AB
from peaky.batch import sideband as SB
from peaky.chem import chemistry as C
from peaky.chem import isotopes as ISO
from peaky.io import io_mascope as IO

NOLOG = dict(log=lambda *a, **k: None)
FILES = ("s01", "s02")

# --------------------------------------------------------------------------- the satellite (a): a note only
PARENT = ("C9H14O2", "[M+(CH4N2O)H]+")       # a bright urea adduct
SAT = ("C15H18O", "[M+H]+")                   # the reading a +3.85 mDa line beside it takes
PMZ = C.ion_mz(*PARENT)
SMZ = PMZ + 3.85e-3


def _row(pid, mz, h, nf=None, ad=None, tier="Assigned", role="M0", **kw):
    return dict(peak_id=pid, mz=mz, height=h, neutral_formula=nf, adduct=ad, tier=tier, role=role,
                ion_score=0.9, tier_reason="", **kw)


def _sat_ledgers(extra=None, files=FILES, parent_h=(2e6, 2e6), dmz=3.85e-3):
    """{sid: ledger}: the parent and, `dmz` above it, the satellite (1e5) in every file."""
    out = {}
    for k, sid in enumerate(files):
        rows = [_row(f"{sid}p", PMZ, parent_h[min(k, len(parent_h) - 1)], *PARENT),
                _row(f"{sid}s", PMZ + dmz + k * 1e-5, 1e5, *SAT),
                _row(f"{sid}u", PMZ + 1.5, 3e3, role="unexplained", tier=None)]
        rows += (extra(sid) if extra else [])
        out[sid] = pd.DataFrame(rows)
    return out


def _merged(ledgers):
    merged, _jit = AB.align({sid: AB._m0(led) for sid, led in ledgers.items()}, tol_ppm=6.0)
    merged["role"] = "M0"
    return merged


def _series(ratio_noise=0.08, n=12, seed=1, smz=SMZ, ratio=20.0, ramp=False, own13c=None):
    """A batch series of n spectra (plus 10 noise picks each at 50 cps: the spectra's
    noise edge): the parent nearly flat (+-10 %) -- or, `ramp`, ramping over two
    decades -- and the satellite following it at 1/`ratio` (log-normal scatter
    `ratio_noise`), or on its own when ratio_noise is None; `own13c` = the
    satellite's own 13C line at that multiple of its carbon count's share. A flat
    parent is tracked (within TRACK_X) but cannot EXPLAIN the satellite (centred R^2
    of the fit through the origin < SAT_FIT_R2): the satellite clause notes it. A
    ramping one explains it: with no own 13C line it is held as an artefact."""
    rng = np.random.default_rng(seed)
    hp = (10 ** np.linspace(4.5, 6.5, n)) if ramp else 1e6 * (1 + 0.1 * np.sin(np.arange(n)))
    hs = hp / ratio * np.exp(rng.normal(0, ratio_noise, n)) if ratio_noise is not None \
        else 10 ** rng.uniform(3, 5.5, n)
    rows = []
    for k in range(n):
        rows += [dict(sample_item_id=f"t{k:02d}", mz=PMZ, height=hp[k]),
                 dict(sample_item_id=f"t{k:02d}", mz=smz, height=hs[k])]
        if own13c:
            rows.append(dict(sample_item_id=f"t{k:02d}", mz=smz + ISO.D_13C,
                             height=hs[k] * ISO.R_13C_PER_C * 15 * own13c))
        rows += [dict(sample_item_id=f"t{k:02d}", mz=150.0 + 7.1 * j, height=50.0) for j in range(10)]
    return pd.DataFrame(rows)


def _get(merged, nf, ad):
    r = merged[(merged["neutral_formula"] == nf) & (merged["adduct"] == ad)]
    assert len(r) == 1, merged[["mz", "neutral_formula", "adduct", "tier"]]
    return r.iloc[0]


def test_the_satellite_window_is_two_sided():                                 # SAT_DMDA, the lower side
    # the partner 3.85 mDa ABOVE the row (the U-mode side bands sit on both sides of their parent)
    led = _sat_ledgers(dmz=-3.85e-3)
    merged = _merged(led)
    tab, summ = SB.hold_low_file_promotions(merged, led, _series(smz=PMZ - 3.85e-3), **NOLOG)
    assert summ["satellite_noted"] == 1 and _get(merged, *SAT)["tier"] == "Assigned"
    assert (tab["d_mDa"].between(-3.9, -3.8)).all()


def test_a_tracking_satellite_of_a_20x_brighter_line_is_noted_and_its_tier_kept():
    led = _sat_ledgers()
    merged = _merged(led)
    assert _get(merged, *SAT)["tier"] == "Assigned"                  # the vote's verdict, two files
    tab, summ = SB.hold_low_file_promotions(merged, led, _series(), **NOLOG)
    r = _get(merged, *SAT)
    assert r["tier"] == "Assigned" and SB.SATELLITE_MARK in r["tier_reason"]
    assert SB.HOLD_MARK not in r["tier_reason"]
    assert summ["satellite_noted"] == 1 and summ["held"] == 0
    assert sorted(tab["sample_item_id"]) == list(FILES) and set(tab["action"]) == {"noted"}
    assert set(tab["rule"]) == {"satellite"} and set(tab["partner_peak_id"]) == {"s01p", "s02p"}
    assert (tab["d_mDa"].between(3.8, 3.9)).all() and (tab["ratio"] >= 19).all()
    assert _get(merged, *PARENT)["tier"] == "Assigned"


def test_a_satellite_the_parent_explains_across_the_spectra_is_held():    # the per-spectrum test (Part D)
    # the parent ramps over two decades and the satellite follows it at 1/20 (5 % scatter): the fit
    # through the origin explains it (R2 >= SAT_FIT_R2) and it has no 13C line: an artefact -> held
    led = _sat_ledgers()
    merged = _merged(led)
    tab, summ = SB.hold_low_file_promotions(merged, led, _series(ramp=True, ratio_noise=0.05), **NOLOG)
    r = _get(merged, *SAT)
    assert r["tier"] == "Candidate" and SB.SATELLITE_HOLD_MARK in r["tier_reason"]
    assert summ["satellite_held"] == 1 and summ["satellite_noted"] == 0 and summ["held"] == 0
    assert set(tab["action"]) == {"satellite_held"} and (tab["fit_r2"] >= 0.9).all()
    assert (tab["fit_n"] == 12).all() and np.allclose(tab["fit_slope"], 1 / 20, rtol=0.1)
    assert set(tab["own_13c_series"]) == {False}
    assert _get(merged, *PARENT)["tier"] == "Assigned"


def test_an_explained_line_with_its_own_13c_line_in_the_series_is_only_noted():   # "no own 13C line"
    # the same explained satellite, but its 13C line sits at its own carbon count's share in every
    # spectrum of the series (the files' ledgers do not list it): an ion -> the note only
    led = _sat_ledgers()
    merged = _merged(led)
    tab, summ = SB.hold_low_file_promotions(merged, led, _series(ramp=True, ratio_noise=0.05, own13c=1.0),
                                            **NOLOG)
    r = _get(merged, *SAT)
    assert r["tier"] == "Assigned" and SB.SATELLITE_MARK in r["tier_reason"]
    assert SB.SATELLITE_HOLD_MARK not in r["tier_reason"]
    assert summ["satellite_held"] == 0 and summ["satellite_noted"] == 1
    assert set(tab["own_13c_series"]) == {True} and (tab["fit_r2"] >= 0.9).all()
    # ... at 3x its share the line is not its own 13C line: held again (SAT_OWN_13C_BAND)
    led2 = _sat_ledgers()
    merged2 = _merged(led2)
    _t2, summ2 = SB.hold_low_file_promotions(merged2, led2, _series(ramp=True, ratio_noise=0.05, own13c=3.0),
                                             **NOLOG)
    assert summ2["satellite_held"] == 1


def test_a_tracking_line_the_flat_parent_cannot_explain_is_only_noted():   # SAT_FIT_R2 in the merge
    led = _sat_ledgers()
    merged = _merged(led)
    tab, summ = SB.hold_low_file_promotions(merged, led, _series(), **NOLOG)
    assert summ["satellite_noted"] == 1 and summ["satellite_held"] == 0
    assert (tab["fit_r2"] < 0.9).all() and (tab["track_share"] >= 0.8).all()


def test_too_few_spectra_for_the_fit_leave_the_note():                         # SAT_FIT_MIN_N in the merge
    led = _sat_ledgers()
    merged = _merged(led)
    _tab, summ = SB.hold_low_file_promotions(merged, led, _series(ramp=True, ratio_noise=0.05, n=4), **NOLOG)
    assert summ["satellite_noted"] == 1 and summ["satellite_held"] == 0


def test_the_artefact_test_reads_a_partner_at_least_twice_the_row():           # SAT_HOLD_MIN_X
    # a 2.5x partner (no note: under SAT_BRIGHT_X) that explains the row across the series: held
    assert SB.SAT_HOLD_MIN_X == 2.0
    led = _sat_ledgers(parent_h=(2.5e5, 2.5e5))
    merged = _merged(led)
    tab, summ = SB.hold_low_file_promotions(merged, led, _series(ramp=True, ratio_noise=0.05, ratio=2.5), **NOLOG)
    assert summ["satellite_held"] == 1 and set(tab["rule"]) == {"shoulder"}
    assert _get(merged, *SAT)["tier"] == "Candidate"
    # a 1.5x partner is a comparable ion, no shoulder: nothing is read
    led2 = _sat_ledgers(parent_h=(1.5e5, 1.5e5))
    merged2 = _merged(led2)
    tab2, summ2 = SB.hold_low_file_promotions(merged2, led2, _series(ramp=True, ratio_noise=0.05, ratio=1.5),
                                              **NOLOG)
    assert summ2["satellite_held"] == 0 and summ2["satellite_noted"] == 0 and not len(tab2)
    # ... and a 2.5x partner the flat series cannot explain leaves the reading alone (no note either)
    led3 = _sat_ledgers(parent_h=(2.5e5, 2.5e5))
    merged3 = _merged(led3)
    tab3, summ3 = SB.hold_low_file_promotions(merged3, led3, _series(ratio=2.5), **NOLOG)
    assert summ3["satellite_held"] == 0 and summ3["satellite_noted"] == 0
    assert _get(merged3, *SAT)["tier"] == "Assigned"


def test_a_partner_under_ten_times_brighter_is_no_satellite():               # SAT_BRIGHT_X
    led = _sat_ledgers(parent_h=(5e5, 5e5))                                  # 5x the satellite
    merged = _merged(led)
    _tab, summ = SB.hold_low_file_promotions(merged, led, _series(ratio=5.0), **NOLOG)
    assert SB.SATELLITE_MARK not in str(_get(merged, *SAT)["tier_reason"]) and summ["satellite_noted"] == 0


def test_a_partner_6_mda_away_is_no_satellite():                             # SAT_DMDA, the upper side
    led = _sat_ledgers(dmz=6.0e-3)
    merged = _merged(led)
    _tab, summ = SB.hold_low_file_promotions(merged, led, _series(smz=PMZ + 6.0e-3), **NOLOG)
    assert SB.SATELLITE_MARK not in str(_get(merged, *SAT)["tier_reason"]) and summ["satellite_noted"] == 0


def test_a_partner_the_row_does_not_track_notes_nothing():                    # TRACK_X, TRACK_SHARE
    led = _sat_ledgers()
    merged = _merged(led)
    _tab, summ = SB.hold_low_file_promotions(merged, led, _series(ratio_noise=None), **NOLOG)
    assert SB.SATELLITE_MARK not in str(_get(merged, *SAT)["tier_reason"]) and summ["satellite_noted"] == 0


def test_without_a_series_the_files_own_ratios_must_agree():                  # the file-ratio fallback
    led = _sat_ledgers(parent_h=(2e6, 1e7))                                   # 20x in one file, 100x in the other
    merged = _merged(led)
    _tab, summ = SB.hold_low_file_promotions(merged, led, None, **NOLOG)
    assert summ["satellite_noted"] == 0
    led2 = _sat_ledgers(parent_h=(2e6, 3e6))                                  # 20x and 30x: they agree
    merged2 = _merged(led2)
    _tab, summ2 = SB.hold_low_file_promotions(merged2, led2, None, **NOLOG)
    assert summ2["satellite_noted"] == 1


def test_the_satellite_clause_is_an_orbitrap_clause():
    led = _sat_ledgers()
    merged = _merged(led)
    _tab, summ = SB.hold_low_file_promotions(merged, led, _series(), orbitrap=False, **NOLOG)
    assert SB.SATELLITE_MARK not in str(_get(merged, *SAT)["tier_reason"]) and summ["satellite_noted"] == 0


def test_a_satellite_row_with_its_own_13c_line_is_neither_held_nor_noted():
    def own13c(sid):
        return [_row(f"{sid}c13", SMZ + ISO.D_13C, 1e5 * ISO.R_13C_PER_C * 15, role="iso_child", tier=None)]
    led = _sat_ledgers(own13c)
    merged = _merged(led)
    tab, summ = SB.hold_low_file_promotions(merged, led, _series(), **NOLOG)
    r = _get(merged, *SAT)
    assert r["tier"] == "Assigned" and SB.SATELLITE_MARK not in str(r["tier_reason"])
    assert summ["own_13c_kept"] == 1 and not len(tab)


def test_a_reading_assigned_in_three_files_is_not_tested():                  # MAX_FILES
    led = _sat_ledgers(files=("s01", "s02", "s03"))
    merged = _merged(led)
    _tab, summ = SB.hold_low_file_promotions(merged, led, _series(), **NOLOG)
    assert summ["tested"] == 0 and summ["satellite_noted"] == 0


# --------------------------------------------------------------------------- the isotopologue (b): the hold
IPAR = ("C10H16O3", "[M+H]+")
IPMZ = C.ion_mz(*IPAR)
IROW = ("C6H14N2O3", "[M+H]+")                 # a reading of the 13C line's mass
HP = 4e5
PRED13 = HP * ISO.R_13C_PER_C * 10             # the parent's 13C line


def _iso_ledgers(files=FILES, row_h=1.1 * PRED13, d=ISO.D_13C, dppm=0.0, parent=IPAR, parent_h=HP,
                 parent_tier="Assigned", parent_in=None, own13c_x=None, row_tier=None):
    """{sid: ledger}: the parent and, at its `d` spacing (+ `dppm`), the row in every file;
    `parent_in` = the files holding the parent (default every file); `own13c_x` = {sid: the
    row's own 13C line at that multiple of its carbon count's share}; `row_tier` = {sid: tier}."""
    pmz = C.ion_mz(*parent)
    rmz = pmz + d
    rmz += rmz * dppm * 1e-6
    out = {}
    for sid in files:
        rows = [_row(f"{sid}r", rmz, row_h, *IROW, tier=(row_tier or {}).get(sid, "Assigned"))]
        if parent_in is None or sid in parent_in:
            rows.append(_row(f"{sid}p", pmz, parent_h, *parent, tier=parent_tier))
        if own13c_x and sid in own13c_x:
            rows.append(_row(f"{sid}o", rmz + ISO.D_13C, row_h * ISO.R_13C_PER_C * 6 * own13c_x[sid],
                             role="iso_child", tier=None))
        out[sid] = pd.DataFrame(rows)
    return out


def _hold(led, merged=None, **kw):
    merged = _merged(led) if merged is None else merged
    tab, summ = SB.hold_low_file_promotions(merged, led, None, sigma_ppm=kw.pop("sigma", {s: 0.2 for s in led}),
                                            **NOLOG, **kw)
    return merged, tab, summ


def test_a_13c_line_of_a_brighter_row_at_1_1x_its_prediction_is_held():     # ISO_HEIGHT_X
    merged, tab, summ = _hold(_iso_ledgers())
    r = _get(merged, *IROW)
    assert r["tier"] == "Candidate" and SB.HOLD_MARK in r["tier_reason"] and "isotopologue 13C" in r["tier_reason"]
    assert summ["held"] == 1 and set(tab["rule"]) == {"isotopologue 13C"} and set(tab["action"]) == {"held"}
    merged2, _t, _s = _hold(_iso_ledgers(row_h=2.0 * PRED13))                # brighter than the line can be
    assert _get(merged2, *IROW)["tier"] == "Assigned"


def test_an_isotopologue_row_beside_a_satellite_partner_too_is_held():       # the hold before the note
    # each file holds both a brighter Assigned parent whose 13C line the row is and a >= 10x line
    # 3.85 mDa below the row that it tracks: the isotopologue clause decides -- held, not merely noted
    led = _iso_ledgers()
    rmz = IPMZ + ISO.D_13C
    for sid, f in led.items():
        led[sid] = pd.concat([f, pd.DataFrame([_row(f"{sid}b", rmz - 3.85e-3, 12 * 1.1 * PRED13,
                                                   role="unexplained", tier=None)])], ignore_index=True)
    assert all(len(SB._bright_near(f, f[f.peak_id.str.endswith("r")].iloc[0])) for f in led.values())
    merged, tab, summ = _hold(led)
    r = _get(merged, *IROW)
    assert r["tier"] == "Candidate" and SB.HOLD_MARK in r["tier_reason"] and summ["held"] == 1
    assert summ["satellite_noted"] == 0 and set(tab["action"]) == {"held"}
    assert set(tab["rule"]) == {"isotopologue 13C"}


def test_a_row_5_ppm_off_the_spacing_is_not_the_line():                      # ISO_PPM
    merged, _t, summ = _hold(_iso_ledgers(dppm=5.0))
    assert _get(merged, *IROW)["tier"] == "Assigned" and summ["held"] == 0


def test_the_window_widens_by_the_files_sigma():                             # sigma in the window
    merged, _t, summ = _hold(_iso_ledgers(dppm=1.9), sigma={s: 0.5 for s in FILES})
    assert _get(merged, *IROW)["tier"] == "Candidate" and summ["held"] == 1
    merged2, _t2, summ2 = _hold(_iso_ledgers(dppm=1.9), sigma={s: 0.1 for s in FILES})
    assert _get(merged2, *IROW)["tier"] == "Assigned" and summ2["held"] == 0


def test_a_row_held_in_one_file_but_not_the_other_stands():                  # "in every file"
    merged, _t, summ = _hold(_iso_ledgers(parent_in={"s01"}))
    r = _get(merged, *IROW)
    assert r["tier"] == "Assigned" and summ["held"] == 0 and summ["satellite_noted"] == 0
    assert SB.HOLD_MARK not in str(r["tier_reason"])


def test_a_parent_fainter_than_the_row_holds_nothing():                      # the parent is brighter
    # a Cl3 parent's 37Cl line reads 0.96x it: a row at 1.2x the parent sits under 1.5x the line,
    # but a line cannot be brighter than the ion it belongs to
    par = ("C2HCl3O2", "[M-H]-")
    pred = 1e5 * ISO.R_37CL_PER_CL * 3
    merged, _t, summ = _hold(_iso_ledgers(parent=par, parent_h=1e5, d=ISO.D_37CL, row_h=1.2e5))
    assert pred * 1.5 > 1.2e5                                                 # the height bar alone would pass
    assert _get(merged, *IROW)["tier"] == "Assigned" and summ["held"] == 0


def test_a_parent_without_the_element_puts_no_line_there():                  # the element count
    # the 34S spacing of an S-free parent: no 34S line exists
    merged, _t, summ = _hold(_iso_ledgers(d=ISO.D_34S, row_h=0.04 * HP))
    assert _get(merged, *IROW)["tier"] == "Assigned" and summ["held"] == 0


def test_a_candidate_parent_holds_nothing():                                  # the parent is Assigned
    merged, _t, summ = _hold(_iso_ledgers(parent_tier="Candidate"))
    assert _get(merged, *IROW)["tier"] == "Assigned" and summ["held"] == 0


def test_own_13c_skips_only_in_band_and_only_in_every_file():                # OWN_13C_BAND, "each file"
    # (the parent itself is always skipped: the row is its own 13C line, in band)
    merged, _t, summ = _hold(_iso_ledgers(own13c_x={"s01": 1.0, "s02": 1.0}))
    assert _get(merged, *IROW)["tier"] == "Assigned" and summ["own_13c_kept"] == 2
    merged2, _t2, summ2 = _hold(_iso_ledgers(own13c_x={"s01": 3.0, "s02": 3.0}))   # 3x its share: not its own
    assert _get(merged2, *IROW)["tier"] == "Candidate" and summ2["own_13c_kept"] == 1
    merged3, _t3, summ3 = _hold(_iso_ledgers(own13c_x={"s01": 1.0}))                # one file only
    assert _get(merged3, *IROW)["tier"] == "Candidate" and summ3["own_13c_kept"] == 1


def test_only_the_files_that_assign_the_reading_count():                      # the Assigned-file count
    # three files carry the ion, two of them Assigned: the reading reaches Assigned from two
    led = _iso_ledgers(files=("s01", "s02", "s03"), row_tier={"s03": "Candidate"})
    merged, _t, summ = _hold(led)
    assert summ["tested"] >= 1 and _get(merged, *IROW)["tier"] == "Candidate" and summ["held"] == 1


def test_a_file_reading_another_ion_at_the_mass_is_not_one_of_its_files():   # the same-ion match
    led = _iso_ledgers(parent_in={"s01"})
    other = led["s02"].copy()
    other.loc[other["peak_id"] == "s02r", ["neutral_formula", "adduct"]] = ["C7H18N2O2", "[M+H]+"]   # another ion
    other.loc[other["peak_id"] == "s02r", "height"] = 5 * PRED13
    led["s02"] = other
    m = C.ion_mz(*IPAR) + ISO.D_13C
    merged = pd.DataFrame([dict(mz=m, neutral_formula=IROW[0], adduct=IROW[1], tier="Assigned", role="M0",
                                srcs="s01,s02", tier_reason="")])
    merged, tab, summ = _hold(led, merged=merged)
    assert merged.at[0, "tier"] == "Candidate" and summ["held"] == 1 and list(tab["sample_item_id"]) == ["s01"]


# --------------------------------------------------------------------------- the gate inside a batch
def test_the_batch_merge_holds_the_isotopologue_reading(tmp_path, monkeypatch):
    """Two files each commit the parent and, on its 13C line, another reading: the merged
    ledger written by the batch reads that reading Candidate with the hold's note, the
    summary counts it, and tables/sideband_holds.csv names the parent."""
    from peaky.assignment import assign as A
    rmz = IPMZ + ISO.D_13C
    lines = [(IPAR, IPMZ, HP), (IROW, rmz, 1.1 * PRED13)]
    t0 = pd.Timestamp("2025-10-01 00:00:00", tz="UTC")
    ts = pd.DataFrame([dict(sample_item_id=f"u{k:02d}", sample_item_name=f"n{k:02d}",
                            datetime_utc=t0 + pd.Timedelta(hours=2 * k), mz=float(mz), height=float(h))
                       for k in range(6) for (_r, mz, h) in lines]
                      + [dict(sample_item_id=f"u{k:02d}", sample_item_name=f"n{k:02d}",
                              datetime_utc=t0 + pd.Timedelta(hours=2 * k), mz=float(300 + j), height=500.0)
                         for k in range(6) for j in range(20)])

    def ledger():
        peaks = pd.DataFrame([{"peak_id": f"p{j}", "mz": mz, "height": h} for j, (_r, mz, h) in enumerate(lines)])
        led = L.new_ledger(peaks)
        for j, ((n, a), _mz, _h) in enumerate(lines):
            L.commit_assignment(led, f"p{j}", neutral_formula=n, adduct=a,
                                ion_formula=C.format_formula(EV.ion_composition(n, a, None)) + "+",
                                ion_score=0.95, compound_score=0.95, ppm_error=0.1, pass_no=1,
                                method="cheminfo+grid", confidence="High", commentary="Pass 1")
        led["tier"] = led["tier"].astype(object)
        led.loc[led["role"] == L.ROLE_M0, "tier"] = "Assigned"
        led["below_assignability"] = False
        return led

    def fake_assign(sid, context="ambient-air", **kw):
        return {"ledger": ledger(),
                "stats": {"noise_edge_cps": 4.0, "height_gate_cps": 10.0, "side_channels": [],
                          "degeneracy_cal": {"mu": 0.0, "sigma": 0.2}},
                "plausibility_audit": [], "summaries": {}, "problems": []}

    monkeypatch.setattr(IO, "connect", lambda *a, **k: "CLIENT")
    monkeypatch.setattr(IO, "fetch_peaks", lambda client, sid, use_cache=True: pd.DataFrame(
        {"peak_id": ["A"], "mz": [300.1], "height": [1.0e5]}))
    monkeypatch.setattr(IO, "estimate_offset", lambda raw: 0.0)
    monkeypatch.setattr(A, "run", fake_assign)
    AB.run(peaks=ts, ts_peaks=ts, reagent="Ur", batch="test batch", out_dir=str(tmp_path),
           k_min=2, k_max=2, min_gain=0.0, n_jobs=1, residual=False, resolving_power=100_000,
           log=lambda *a: None)
    summ = json.load(open(tmp_path / "batch_summary.json"))
    assert summ["merge_gates"]["sideband"]["held"] == 1
    merged = pd.read_csv(tmp_path / "merged_ledger.csv")
    r = merged[(merged.neutral_formula == IROW[0]) & (merged.adduct == IROW[1])]
    assert len(r) == 1 and r.tier.iloc[0] == "Candidate" and SB.HOLD_MARK in str(r.tier_reason.iloc[0])
    p = merged[(merged.neutral_formula == IPAR[0]) & (merged.adduct == IPAR[1])]
    assert p.tier.iloc[0] == "Assigned"
    tab = pd.read_csv(tmp_path / "tables" / "sideband_holds.csv")
    assert len(tab) == 2 and set(tab["action"]) == {"held"} and set(tab["partner_reading"]) == {f"{IPAR[0]} {IPAR[1]}"}


# --------------------------------------------------------------------------- Part D wiring (repair: behaviour, not pins)
def _two_partner_series(n=12, seed=3):
    """The ramping 20x parent the satellite follows, and a 3x line 2 mDa below the satellite that varies on its
    own (the dimmer partner), noise picks at 50 cps."""
    rng = np.random.default_rng(seed)
    hp = 10 ** np.linspace(4.5, 6.5, n)
    hs = hp / 20 * np.exp(rng.normal(0, 0.05, n))
    hd = 10 ** rng.uniform(4.0, 5.5, n)
    rows = []
    for k in range(n):
        rows += [dict(sample_item_id=f"t{k:02d}", mz=PMZ, height=hp[k]), dict(sample_item_id=f"t{k:02d}", mz=SMZ, height=hs[k]),
                 dict(sample_item_id=f"t{k:02d}", mz=SMZ - 2e-3, height=hd[k])]
        rows += [dict(sample_item_id=f"t{k:02d}", mz=150.0 + 7.1 * j, height=50.0) for j in range(10)]
    return pd.DataFrame(rows)


def test_the_artefact_test_reads_the_brightest_partner():                     # the partner choice
    def dim(sid):
        return [_row(f"{sid}d", SMZ - 2e-3, 3e5, role="unexplained", tier=None)]     # 3x the satellite
    led = _sat_ledgers(dim)
    merged = _merged(led)
    tab, summ = SB.hold_low_file_promotions(merged, led, _two_partner_series(), **NOLOG)
    assert summ["satellite_held"] == 1 and (tab["partner_mz"].round(4) == round(PMZ, 4)).all()


def test_own_13c_is_read_only_where_testable():                                # SAT_13C_TESTABLE_X at the use site
    # the spectra's edge 2000 cps: the satellite's 13C line is testable (>= 3x the edge) in its 4 brightest
    # spectra only, and present there at its share; absent (untestable) in the 8 others -> its own 13C line
    n = 12
    hp = 10 ** np.linspace(4.5, 6.5, n)
    hs = hp / 20
    rows = []
    for k in range(n):
        sid = f"t{k:02d}"
        rows += [dict(sample_item_id=sid, mz=PMZ, height=hp[k]), dict(sample_item_id=sid, mz=SMZ, height=hs[k])]
        if hs[k] * ISO.R_13C_PER_C * 15 >= 3 * 2000.0:
            rows.append(dict(sample_item_id=sid, mz=SMZ + ISO.D_13C, height=hs[k] * ISO.R_13C_PER_C * 15))
        rows += [dict(sample_item_id=sid, mz=150.0 + 7.1 * j, height=2000.0) for j in range(10)]
    ts = pd.DataFrame(rows)
    assert 2 <= sum(1 for x in hs if x * ISO.R_13C_PER_C * 15 >= 6000.0) < n / 2
    led = _sat_ledgers()
    merged = _merged(led)
    tab, summ = SB.hold_low_file_promotions(merged, led, ts, **NOLOG)
    assert summ["satellite_held"] == 0 and set(tab["own_13c_series"]) == {True}


def test_the_artefact_hold_needs_a_partner_in_every_file():                    # every file
    led = _sat_ledgers()
    led["s02"] = led["s02"][led["s02"]["peak_id"] != "s02p"].reset_index(drop=True)   # s02: no partner
    merged = _merged(led)
    _tab, summ = SB.hold_low_file_promotions(merged, led, _series(ramp=True, ratio_noise=0.05), **NOLOG)
    assert summ["satellite_held"] == 0 and summ["satellite_noted"] == 0
    assert _get(merged, *SAT)["tier"] == "Assigned"


def test_the_hold_record_carries_the_log_log_exponent():
    led = _sat_ledgers()
    merged = _merged(led)
    tab, _s = SB.hold_low_file_promotions(merged, led, _series(ramp=True, ratio_noise=0.05), **NOLOG)
    assert np.allclose(tab["fit_loglog_slope"], 1.0, atol=0.05) and (tab["fit_loglog_r2"] > 0.95).all()
