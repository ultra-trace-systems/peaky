"""The episode stage of the sample selection (batch/sampling.py, THE EPISODE STAGE).

The fixture (tests/fixtures/episodes) is a synthetic 40-spectrum uronium-type batch: 3
persistent ions with their 13C lines, a +3.85 mDa satellite at 0.4x, a 4-spectrum plume
(a caffeine-shaped ion + its 13C line, an NMP-type solvent ion, the solvent cluster of a
persistent ion), pair lines and noise. The presence cover takes ONE plume spectrum (ep20,
its flank); the episode stage adds the peak (ep21) and a neighbour. The batch tests run
`assign_batch.run` offline over it (tests/fixtures/episodes/harness.py). Every rule
parameter is pinned by a small table built here: each test names what it fails without.
"""
import filecmp
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "episodes")
sys.path.insert(0, HERE)

import harness as H  # noqa: E402
import make_episodes as MK  # noqa: E402
from peaky.batch import sampling as SS  # noqa: E402
from peaky.reporting import pdf_report as PR  # noqa: E402

PLUME = ("ep20", "ep21", "ep22", "ep23")
PLUME_KEYS = {"E1", "E1_13C", "S1", "S1_13C", "K1"}


@pytest.fixture(scope="module")
def fx():
    return MK.build()


# --------------------------------------------------------------------------- the fixture
def test_stored_fixture_is_the_generator():
    """batch_ts.csv / components.json are what make_episodes.build() writes (no drift)."""
    t, comp = MK.build()
    stored = MK.load()
    pd.testing.assert_frame_equal(stored.reset_index(drop=True), t.reset_index(drop=True),
                                  check_dtype=False)
    with open(os.path.join(HERE, "components.json")) as fh:
        assert json.load(fh) == json.loads(json.dumps(comp))


# --------------------------------------------------------------------------- Part A on the fixture
def test_the_cover_alone_gives_the_plume_one_file(fx):
    """Negative control of the stage: the presence cover takes a single plume spectrum."""
    t, _ = fx
    cover = SS.select_cover_sample_ids(t)
    assert [s for s in cover if s in PLUME] == ["ep20"]


def test_the_stage_selects_the_plume_peak_and_a_neighbour(fx):
    t, comp = fx
    cover = SS.select_cover_sample_ids(t)
    u = SS.episode_universe(t)
    sel = SS.select_episode_samples(t, u, assigned=cover)
    meta = sel.attrs["selection"]
    assert comp["plume_peak"] == "ep21"
    assert list(sel["sample_item_id"]) == ["ep21", "ep22"]       # the peak, then its nearest brighter neighbour
    assert set(sel["role"]) == {SS.ROLE_EPISODE} and list(sel["episode"]) == [1, 1]
    assert list(sel["bins_new"]) == [5, 5]
    files = set(cover) | set(sel["sample_item_id"])
    assert len(files & set(PLUME)) >= 3 and "ep21" in files
    assert meta["n_episodes"] == 1 and meta["n_free"] == 1 and meta["k"] == 2
    assert meta["stop_reason"] == SS.STOP_EXHAUSTED and meta["n_over_budget"] == 0
    assert sel.attrs["flag_files"] == ["ep20", "ep21", "ep22"]


def test_the_episode_is_the_plume(fx):
    t, comp = fx
    u = SS.episode_universe(t)
    eps = u.attrs["episodes"]
    assert len(eps) == 1
    e = eps[0]
    assert (e["start"], e["end"], e["peak_sample"], e["n_spectra"], e["n_bins"]) == \
        ("ep20", "ep23", "ep21", 4, 5)
    plume_mz = sorted(x["mz"] for x in comp["lines"] if x["key"] in PLUME_KEYS)
    assert np.allclose(e["bins_mz"], plume_mz, atol=2e-4)
    assert e["wanted"] == ["ep21", "ep22", "ep20"]
    # persistent ions, their satellite and the pair lines are not episodic
    assert set(u["n_present"]) == {4} and len(u) == 5


# --------------------------------------------------------------------------- the rule's parameters
T0 = pd.Timestamp("2031-01-01", tz="UTC")


def _table(n, lines, *, noise=20, edge=100.0, flat=False, edges=None):
    """n spectra of `noise` unique noise picks at `edge`..2x`edge` cps (the noise edge is
    then ~`edge`; `flat`: every pick at `edge`, so the edge is exactly `edge`; `edges` =
    {spectrum index: its own flat edge}), plus `lines` = [(mz, {spectrum index: height})]."""
    rows = []
    for i in range(n):
        e = (edges or {}).get(i, edge)
        for k in range(noise):
            rows.append((f"s{i:03d}", 50.0 + 11.0 * k + 0.013 * i,
                         e if (flat or (edges and i in edges)) else e * (1 + k / noise)))
    for mz, hs in lines:
        for i, h in hs.items():
            rows.append((f"s{i:03d}", mz, float(h)))
    t = pd.DataFrame(rows, columns=["sample_item_id", "mz", "height"])
    t["datetime_utc"] = T0 + pd.to_timedelta(t["sample_item_id"].str[1:].astype(int) * 10, unit="min")
    t["peak_id"] = [f"p{j}" for j in range(len(t))]
    return t


def _episodic_mz(t, **kw):
    u = SS.episode_universe(t, **kw)
    return [round(float(x), 3) for x in u["bin_mz"]], u


def test_presence_bounds_min_n_and_max_n():                      # EPISODE_MIN_N, EPISODE_MAX_N
    lines = [(300.0, {10: 5e3, 11: 5e3}),                        # 2 samples: too few
             (310.0, {10: 5e3, 11: 5e3, 12: 5e3}),               # 3: in
             (320.0, {i: 5e3 for i in range(5, 24)}),            # 19: in
             (330.0, {i: 5e3 for i in range(5, 25)})]            # 20: out (max(19, 1 % of 40))
    mzs, u = _episodic_mz(_table(40, lines))
    assert mzs == [310.0, 320.0] and u.attrs["episode_meta"]["n_max"] == 19


def test_presence_bound_grows_with_one_percent_of_the_samples():  # EPISODE_MAX_FRAC
    lines = [(320.0, {i: 5e3 for i in range(5, 26)})]            # 21 of 2100 samples: <= 1 %
    mzs, u = _episodic_mz(_table(2100, lines, noise=3))
    assert mzs == [320.0] and u.attrs["episode_meta"]["n_max"] == 21


def test_the_one_percent_bound_rounds_down():                    # floor(EPISODE_MAX_FRAC x samples)
    # 2180 samples: 1 % = 21.8 -> 21 (a ceil, or 1.01 %, would give 22)
    lines = [(320.0, {i: 5e3 for i in range(5, 26)}),            # 21: in
             (330.0, {i: 5e3 for i in range(5, 27)})]            # 22: out
    mzs, u = _episodic_mz(_table(2180, lines, noise=3))
    assert mzs == [320.0] and u.attrs["episode_meta"]["n_max"] == 21


def test_the_lift_and_edge_bounds_are_inclusive():               # >= EPISODE_MIN_LIFT, >= EPISODE_MIN_X_EDGE
    # every pick at 100 cps: the edge and the baseline are exactly 100; the line at exactly 5x
    mzs, u = _episodic_mz(_table(40, [(300.0, {7: 500.0, 8: 500.0, 9: 500.0})], flat=True))
    assert mzs == [300.0] and float(u["lift"].iloc[0]) == 5.0
    # the peak sample's own edge is 200 (the others 100): the peak at exactly 3x it, lift 6
    t = _table(40, [(300.0, {7: 300.0, 8: 600.0, 9: 300.0})], flat=True, edges={8: 200.0})
    mzs, u = _episodic_mz(t)
    assert mzs == [300.0] and float(u["max_x_edge"].iloc[0]) == 3.0 and float(u["lift"].iloc[0]) == 6.0


def test_a_tie_at_the_maximum_goes_to_the_earliest_sample():
    _, u = _episodic_mz(_table(40, [(300.0, {10: 5e3, 11: 4e3, 12: 5e3})]))
    assert list(u["peak_sample"]) == ["s010"]


def test_samples_are_ordered_by_time_not_by_id():                # sample_table's time order
    # ids that do not sort in time order: time position i carries id s{7i mod 40}
    t = _table(40, [(300.0, {10: 2e3, 11: 5e3, 12: 4e3, 13: 2e3})])
    pid = {f"s{i:03d}": f"s{(7 * i) % 40:03d}" for i in range(40)}
    t["sample_item_id"] = t["sample_item_id"].map(pid)
    _, u = _episodic_mz(t)
    (e,) = u.attrs["episodes"]
    at = {i: pid[f"s{i:03d}"] for i in (10, 11, 12, 13)}
    assert (e["start"], e["end"], e["peak_sample"], e["n_spectra"]) == (at[10], at[13], at[11], 4)
    assert e["wanted"] == [at[11], at[12], at[10]]


def test_lift_over_the_baseline():                               # EPISODE_MIN_LIFT, the baseline
    # present in 19 of 30 samples: the baseline is the median of its own heights there
    flat = {i: 1e3 for i in range(19)}
    lines = [(300.0, {**flat, 5: 4.9e3}),                        # lift 4.9: out
             (310.0, {**flat, 5: 5.1e3})]                        # lift 5.1: in
    mzs, u = _episodic_mz(_table(30, lines))
    assert mzs == [310.0]
    assert float(u["baseline_cps"].iloc[0]) == pytest.approx(1e3)
    # absent elsewhere: the baseline is the samples' noise edge
    mzs, u = _episodic_mz(_table(40, [(300.0, {7: 600.0, 8: 600.0, 9: 600.0})]))
    assert mzs == [300.0] and float(u["baseline_cps"].iloc[0]) == pytest.approx(100.0, rel=0.05)


def test_the_peak_against_its_own_samples_edge():                # EPISODE_MIN_X_EDGE
    lines = [(300.0, {7: 1e3, 8: 2.9e3, 9: 1e3})]
    t = _table(40, lines)
    # spectrum 8 has a noise edge of ~1000 cps: the peak is 2.9x it, the lift is still ~29
    t.loc[(t["sample_item_id"] == "s008") & (t["mz"] < 290), "height"] *= 10
    mzs, _ = _episodic_mz(t)
    assert mzs == []
    t.loc[(t["sample_item_id"] == "s008") & (t["mz"] == 300.0), "height"] = 3.1e3
    mzs, _ = _episodic_mz(t)
    assert mzs == [300.0]


def test_a_run_inside_a_longer_one_does_not_cut_the_episode():
    a = (300.0, {i: 5e3 for i in range(10, 21)})                 # run 10-20
    b = (310.0, {12: 5e3, 13: 5e3, 14: 5e3})                    # inside a's run
    c = (320.0, {22: 5e3, 23: 5e3, 24: 5e3})                    # touches a's end (one gap)
    _, u = _episodic_mz(_table(40, [a, b, c]))
    assert [(e["start"], e["end"], e["n_bins"]) for e in u.attrs["episodes"]] == [("s010", "s024", 3)]


def test_runs_allow_one_gap_and_episodes_join_by_overlap():      # EPISODE_GAP
    a = (300.0, {10: 3e3, 11: 5e3, 13: 3e3})                    # one gap (12): one run 10-13
    b = (310.0, {14: 5e3, 15: 3e3, 16: 3e3})                    # touches a's run: same episode
    c = (320.0, {19: 5e3, 20: 3e3, 21: 3e3})                    # two samples after b's end: new episode
    d = (330.0, {25: 5e3, 26: 3e3, 29: 3e3})                    # two missing (27, 28): run 25-26
    _, u = _episodic_mz(_table(40, [a, b, c, d]))
    eps = u.attrs["episodes"]
    assert [(e["start"], e["end"], e["n_bins"]) for e in eps] == \
        [("s010", "s016", 2), ("s019", "s021", 1), ("s025", "s026", 1)]


def test_the_peak_sample_holds_the_most_bins():
    # bin 300 peaks in s010 (200x edge: the largest summed multiple of any sample), but s011
    # and s012 hold all three bins at >= 3x edge; s012 is the brighter of the two
    lines = [(300.0, {10: 2e4, 11: 2e3, 12: 400.0}),
             (310.0, {10: 200.0, 11: 4e3, 12: 5e3}),
             (320.0, {11: 4e3, 12: 5e3, 13: 4e3})]
    _, u = _episodic_mz(_table(40, lines))
    (e,) = u.attrs["episodes"]
    assert e["peak_sample"] == "s012" and e["n_bright"]["s012"] == 3


def test_neighbours_need_half_the_bins_and_come_nearest_first():  # EPISODE_NEIGHBOUR_FRAC, _NEIGHBOURS
    # four bins in s010-s014 at 50x edge; the peak s012 (bin 300 at 200x). At >= 3x edge:
    # s011 holds 1 of 4 (25 %), s013 2 of 4 (50 %), s010 and s014 all 4 (s014 brighter)
    four = [(300.0 + 10 * j, {10: 5e3, 11: 5e3, 12: 5e3, 13: 5e3, 14: 6e3}) for j in range(4)]
    for j, (_m, hs) in enumerate(four):
        if j >= 1:
            hs[11] = 150.0
        if j >= 2:
            hs[13] = 150.0
    four[0][1][12] = 2e4
    t = _table(40, four)
    _, u = _episodic_mz(t)
    (e,) = u.attrs["episodes"]
    assert e["peak_sample"] == "s012"
    # nearest first even with fewer bins (s013 before s014 / s010); never s011 (25 %)
    assert e["neighbours"] == ["s013", "s014"]
    _, u3 = _episodic_mz(t, n_neighbours=3)
    assert u3.attrs["episodes"][0]["neighbours"] == ["s013", "s014", "s010"]
    _, u25 = _episodic_mz(t, neighbour_frac=0.25)
    assert u25.attrs["episodes"][0]["neighbours"] == ["s013", "s011"]
    # the three bins on 300+ peak in s014 (6e3): their own peak spectrum is wanted before the neighbours
    assert e["bin_peaks"] == ["s014"] and e["wanted"] == ["s012", "s014", "s013"]


# --------------------------------------------------------------------------- the budget
def _three_episodes():
    lines = [(300.0, {5: 2e3, 6: 9e3, 7: 2e3}),               # dim
             (310.0, {15: 2e3, 16: 9e4, 17: 2e3}),             # brightest
             (320.0, {25: 2e3, 26: 3e4, 27: 2e3})]             # middle
    t = _table(40, lines)
    return t, SS.episode_universe(t)


def test_the_budget_serves_every_peak_before_a_neighbour_by_brightness():
    t, u = _three_episodes()
    sel = SS.select_episode_samples(t, u, assigned=[], k_max=4)
    assert list(sel["sample_item_id"]) == ["s016", "s026", "s006", "s015"]
    meta = sel.attrs["selection"]
    assert meta["stop_reason"] == SS.STOP_KMAX and meta["n_over_budget"] == 5
    assert SS.episode_budget_warning(meta) is not None


def test_the_budget_flags_only_when_a_wanted_file_is_left_out(fx):  # K_EPISODE_MAX, the flag
    t, _ = fx
    cover = SS.select_cover_sample_ids(t)
    u = SS.episode_universe(t)
    full = SS.select_episode_samples(t, u, assigned=cover, k_max=2)     # exactly what it wants
    assert full.attrs["selection"]["stop_reason"] == SS.STOP_EXHAUSTED
    assert SS.episode_budget_warning(full.attrs["selection"]) is None
    hit = SS.select_episode_samples(t, u, assigned=cover, k_max=1)
    m = hit.attrs["selection"]
    assert list(hit["sample_item_id"]) == ["ep21"]
    assert m["stop_reason"] == SS.STOP_KMAX and m["episodes"][0]["over_budget"] == ["ep22"]
    assert "k_episode_max=1" in SS.episode_budget_warning(m)
    assert (SS.K_EPISODE_MAX, SS.EPISODE_MIN_X_EDGE, SS.EPISODE_BRIGHT_X_EDGE) == (None, 5.0, 3.0)
    none = SS.select_episode_samples(t, u, assigned=cover, k_max=0)
    assert len(none) == 0 and none.attrs["selection"]["stop_reason"] == SS.STOP_KMAX


@pytest.mark.parametrize("cap,picks,stop", [(None, 9, "exhausted"), (9, 9, "exhausted"), (4, 4, "k_max"),
                                             (0, 0, "k_max")])
def test_no_budget_by_default_and_the_cap_is_optional(cap, picks, stop):   # K_EPISODE_MAX = None
    """Three qualifying single-bin episodes, each wanting its peak and two neighbours (9 files):
    with no cap all are taken; a cap is flagged when it binds."""
    t, u = _three_episodes()
    kw = {} if cap is None else {"k_max": cap}
    sel = SS.select_episode_samples(t, u, assigned=[], **kw)
    m = sel.attrs["selection"]
    assert (len(sel), m["stop_reason"], m["k_max"], m["n_over_budget"]) == (picks, stop, cap, 9 - picks)
    assert (SS.episode_budget_warning(m) is None) == (stop == "exhausted")


@pytest.mark.parametrize("peak,min_x,covered", [(980.0, 5.0, False), (1000.0, 5.0, True), (980.0, 4.0, True),
                                                (1000.0, 5.5, False)])
def test_the_coverage_threshold(peak, min_x, covered):           # EPISODE_MIN_X_EDGE, --episode-min-x-edge
    """An episodic bin (lift >= 5: the other spectra's edge is 100) whose peak spectrum's own
    edge is 200: at 4.9x that edge it is not covered at the default 5.0, at 5.0x it is."""
    t = _table(40, [(300.0, {7: 400.0, 8: peak, 9: 400.0})], flat=True, edges={8: 200.0})
    u = SS.episode_universe(t, min_x_edge=min_x)
    assert len(u) == 1 and bool(u["qualifies"].iloc[0]) is covered
    sel = SS.select_episode_samples(t, u, assigned=[])
    m = sel.attrs["selection"]
    assert ("s008" in list(sel["sample_item_id"])) is covered
    assert (m["n_qualifying_bins"], m["min_x_edge"]) == (int(covered), min_x)
    if not covered:
        assert m["stop_reason"] == SS.STOP_EMPTY and m["episodes"][0]["wanted"] == []


def test_a_qualifying_bins_own_peak_spectrum_is_selected():
    """Bins A (peak s010) and B (peak s015) form one episode whose peak sample is s012 and whose
    neighbours are s013 / s011: both bins still get their own peak spectrum."""
    a = (300.0, {9: 3e3, 10: 9e3, 11: 6e3, 12: 3e3, 13: 2e3})
    b = (310.0, {12: 3e3, 13: 3e3, 14: 3e3, 15: 8e3})
    t = _table(40, [a, b])
    u = SS.episode_universe(t)
    (e,) = u.attrs["episodes"]
    assert (e["peak_sample"], e["bin_peaks"], e["neighbours"]) == ("s012", ["s010", "s015"], ["s013", "s011"])
    sel = SS.select_episode_samples(t, u, assigned=[])
    assert list(sel["sample_item_id"]) == ["s012", "s010", "s015", "s013", "s011"]


def test_a_file_the_cover_took_is_free(fx):
    t, _ = fx
    u = SS.episode_universe(t)
    sel = SS.select_episode_samples(t, u, assigned=["ep21", "ep22", "ep20"])
    m = sel.attrs["selection"]
    assert len(sel) == 0 and m["n_free"] == 3 and m["k"] == 0 and m["stop_reason"] == SS.STOP_EXHAUSTED
    assert m["episodes"][0]["files"] == ["ep21", "ep22", "ep20"] and m["episodes"][0]["picked"] == []


# --------------------------------------------------------------------------- the row flag
def test_the_flag_is_only_on_episodic_bins():
    """A line present in 10 spectra at 1.5x edge (in the presence range, lift < 5) beside an
    episodic one in the same file: only the episodic line's row is flagged."""
    t = _table(40, [(300.0, {9: 2e3, 10: 5e3, 11: 2e3}), (310.0, {i: 150.0 for i in range(5, 15)})])
    u = SS.episode_universe(t)
    assert u.attrs["episode_meta"]["n_bins_in_range"] == 2 and len(u) == 1
    led = t[t["sample_item_id"] == "s010"][["peak_id", "mz", "height"]].reset_index(drop=True)
    f = SS.episode_row_flags(led, "s010", u, ["s010"])
    assert list(led.loc[f, "mz"]) == [300.0]


def test_row_flags_by_peak_id_then_by_mz_for_a_row_without_one(fx):
    t, _ = fx
    u = SS.episode_universe(t)
    pk = t[t["sample_item_id"] == "ep21"].reset_index(drop=True)
    led = pk[["peak_id", "mz", "height", "key"]].copy()
    syn = pd.DataFrame([{"peak_id": None, "mz": float(pk.loc[pk["key"] == "E1", "mz"].iloc[0]) + 1e-4,
                         "height": 1.0, "key": "syn"},
                        {"peak_id": None, "mz": 250.0, "height": 1.0, "key": "syn2"},
                        # a row without an id 10 ppm off an episodic bin: outside the tolerance (6 ppm)
                        {"peak_id": None, "mz": float(pk.loc[pk["key"] == "E1", "mz"].iloc[0]) * (1 + 10e-6),
                         "height": 1.0, "key": "syn10ppm"},
                        # a row WITH a peak id the series does not list on an episodic bin: by id, not m/z
                        {"peak_id": "other-1", "mz": float(pk.loc[pk["key"] == "S1", "mz"].iloc[0]) + 1e-4,
                         "height": 1.0, "key": "idrow"}])
    led = pd.concat([led, syn], ignore_index=True)
    f = SS.episode_row_flags(led, "ep21", u, ["ep21"])
    assert set(led.loc[f, "key"]) == PLUME_KEYS | {"syn"}
    assert not SS.episode_row_flags(led, "ep23", u, ["ep21"]).any()       # not a flagged file
    assert not SS.episode_row_flags(led, "ep21", None, ["ep21"]).any()


# --------------------------------------------------------------------------- the batch
@pytest.fixture(scope="module")
def plume_run(tmp_path_factory):
    t, comp = MK.build()
    mp = pytest.MonkeyPatch()
    H.patch(t, comp, mp.setattr)
    d = tmp_path_factory.mktemp("plume")
    summ = H.run_batch(str(d), t)
    mp.undo()
    return d, summ, t


def test_the_batch_assigns_the_episode_files_and_writes_the_record(plume_run):
    d, summ, _t = plume_run
    sel = pd.read_csv(d / "tables" / "selected_samples.csv")
    ep = sel[sel["role"] == SS.ROLE_EPISODE]
    assert list(ep["sample_item_id"]) == ["ep21", "ep22"] and list(ep["pick"]) == [8, 9]
    assert summ["sample_ids"][-2:] == ["ep21", "ep22"] and summ["n_files"] == 9
    tab = pd.read_csv(d / "tables" / "episodes.csv", keep_default_na=False)
    assert len(tab) == 1
    r = tab.iloc[0]
    assert (r["peak_sample"], r["start"], r["end"], int(r["n_bins"])) == ("ep21", "ep20", "ep23", 5)
    assert (r["picked"], r["files"], r["over_budget"]) == ("ep21;ep22", "ep21;ep22;ep20", "")
    e = summ["selection"]["episodes"]
    assert e["k"] == 2 and e["stop_reason"] == SS.STOP_EXHAUSTED and e["sample_ids"] == ["ep21", "ep22"]
    assert e["k_max"] is None and e["min_x_edge"] == 5.0 and e["n_qualifying_bins"] == 5
    assert e["est_extra_wall_s"] >= 0 and e["assign_wall_s"] >= e["est_extra_wall_s"]
    assert e["episodes"][0]["picked"] == ["ep21", "ep22"] and "bins" not in e["episodes"][0]
    # the stage's yield is counted apart from the cover's (summary only; its ions read stage = cover)
    assert summ["n_files_by_stage"] == {"cover": 7, "episode": 2, "residual": 0}


def test_the_batch_flags_the_plume_rows(plume_run):
    d, _summ, t = plume_run
    key = dict(zip(t["peak_id"], t["key"]))
    flagged = {}
    for f in sorted(os.listdir(d / "per_file")):
        if not f.endswith("_ledger.csv"):
            continue
        led = pd.read_csv(d / "per_file" / f)
        assert "episode" in led.columns, f
        sid = f[: -len("_ledger.csv")]
        flagged[sid] = {key.get(p, "?") for p in led.loc[led["episode"].astype(bool), "peak_id"]}
    assert flagged.pop("ep20") == PLUME_KEYS and flagged.pop("ep21") == PLUME_KEYS \
        and flagged.pop("ep22") == PLUME_KEYS
    assert all(not v for v in flagged.values()), flagged


def test_the_batch_flags_the_budget(tmp_path, monkeypatch):
    t, comp = MK.build()
    H.patch(t, comp, monkeypatch.setattr)
    summ = H.run_batch(str(tmp_path), t, k_episode_max=1)
    e = summ["selection"]["episodes"]
    assert e["stop_reason"] == SS.STOP_KMAX and e["n_over_budget"] == 1 and e["sample_ids"] == ["ep21"]
    assert summ["n_files_by_stage"] == {"cover": 7, "episode": 1, "residual": 0}
    assert any("WARNING: the cap k_episode_max=1" in s for s in
               (x[1] for x in PR._selection_lines({"n_files": 8, "batch": summ})))


def test_the_methods_page_names_the_stage(plume_run):
    _d, summ, _t = plume_run
    text = " ".join(s for _k, s in PR._selection_lines({"n_files": 9, "batch": summ}))
    assert "Episode stage: 1 short bright episode(s)" in text and "2 added (no cap)" in text
    assert "peaking at ≥5× the edge (5)" in text and "WARNING: the cap" not in text


# --------------------------------------------------------------------------- GOLDEN
@pytest.mark.parametrize("variant", ["no_episode", "plume_off"])
def test_golden_selection_and_merged_ledger_byte_identical(variant, tmp_path, monkeypatch):
    """A batch with no episode (episode stage on), and the plume batch with the stage off,
    give the selection and the merged ledger the code before the stage gave, byte for
    byte (goldens cut on the parent commit by harness.py)."""
    t, comp = MK.build(plume=(variant != "no_episode"))
    H.patch(t, comp, monkeypatch.setattr)
    kw = {"episodes": False} if variant == "plume_off" else {}
    summ = H.run_batch(str(tmp_path), t, **kw)
    gold = H.GOLDEN_DIR if variant == "no_episode" else H.GOLDEN_OFF_DIR
    for f, g in H.GOLDEN_FILES.items():
        assert filecmp.cmp(tmp_path / f, os.path.join(gold, g), shallow=False), f
    led = pd.read_csv(tmp_path / "per_file" / "ep00_ledger.csv")
    assert "episode" not in led.columns
    if variant == "no_episode":
        # the stage ran and found nothing: an empty record and a header-only table
        e = summ["selection"]["episodes"]
        assert e["n_episodes"] == 0 and e["k"] == 0 and e["stop_reason"] == SS.STOP_EMPTY
        assert len(pd.read_csv(tmp_path / "tables" / "episodes.csv")) == 0
        assert summ["n_files_by_stage"] == {"cover": 6, "residual": 0}    # no episode key without a file
    else:
        assert "episodes" not in summ["selection"]
        assert not (tmp_path / "tables" / "episodes.csv").exists()


# --------------------------------------------------------------------------- the CLI
def test_cli_flags():
    from peaky import cli
    p = cli.build_parser()
    a = p.parse_args(["batch", "--batch", "B"])
    assert a.episodes is SS.EPISODE_DEFAULT and a.k_episode_max is None and a.episode_min_x_edge == 5.0
    a = p.parse_args(["batch", "--batch", "B", "--no-episodes", "--k-episode-max", "3",
                      "--episode-min-x-edge", "4.5"])
    assert a.episodes is False and a.k_episode_max == 3 and a.episode_min_x_edge == 4.5


def test_cmd_batch_forwards_the_episode_knobs(monkeypatch):
    from peaky import cli
    from peaky import pipeline as PL
    got = {}

    class _Stop(Exception):
        pass

    def fake_run_batch(**kw):
        got.update(kw)
        raise _Stop

    monkeypatch.setattr(cli, "_require_creds", lambda: None)
    monkeypatch.setattr(PL, "run_batch", fake_run_batch)
    a = cli.build_parser().parse_args(["batch", "--batch", "B", "--no-episodes", "--k-episode-max", "5",
                                       "--episode-min-x-edge", "4.5", "--no-progress"])
    with pytest.raises(_Stop):
        cli.cmd_batch(a)
    assert (got["episodes"], got["k_episode_max"], got["episode_min_x_edge"]) == (False, 5, 4.5)


def test_the_batch_applies_the_coverage_threshold(tmp_path, monkeypatch):
    """--episode-min-x-edge reaches the stage: above every bin's peak, nothing qualifies and
    the batch is the stage-off batch (the parent's selection, byte for byte)."""
    t, comp = MK.build()
    H.patch(t, comp, monkeypatch.setattr)
    summ = H.run_batch(str(tmp_path), t, episode_min_x_edge=1e6)
    e = summ["selection"]["episodes"]
    assert (e["min_x_edge"], e["n_qualifying_bins"], e["k"], e["stop_reason"]) == (1e6, 0, 0, SS.STOP_EMPTY)
    assert filecmp.cmp(tmp_path / "tables" / "selected_samples.csv",
                       os.path.join(H.GOLDEN_OFF_DIR, "selected_samples.csv"), shallow=False)
