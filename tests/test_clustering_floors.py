"""Brightness floors of the time-series clustering scale with the batch noise edge.

The floors were absolute (200 cps for an assigned channel, 50 cps for an
unassigned bin): right for an Orbitrap batch whose detection edge sits near
60 cps, but a counting TOF (edge ~0.5-1 cps) only clears them with its reagent
ions. They are now multiples of `batch_summary.json['noise_edge_batch_cps']`
when the run records it, and the legacy cps values when it does not; the
traced unassigned set is capped at the `top_n` brightest by median.
"""
import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from peaky.batch import cluster as CL
from peaky.batch import clustering as CLU
from peaky.chem import profiles as P


# ---- unit pieces -----------------------------------------------------------------

def _summary(tmp_path, **kv):
    (tmp_path / "batch_summary.json").write_text(json.dumps(kv))
    return tmp_path


def test_batch_noise_edge_reads_the_run_summary(tmp_path):
    assert CLU.batch_noise_edge(_summary(tmp_path, noise_edge_batch_cps=0.74)) == pytest.approx(0.74)


@pytest.mark.parametrize("payload", [None, {}, {"noise_edge_batch_cps": None},
                                     {"noise_edge_batch_cps": 0.0},
                                     {"noise_edge_batch_cps": -1.0},
                                     {"noise_edge_batch_cps": "n/a"},
                                     {"noise_edge_batch_cps": float("nan")}])
def test_batch_noise_edge_absent_or_unusable_is_none(tmp_path, payload):
    if payload is not None:
        (tmp_path / "batch_summary.json").write_text(json.dumps(payload))
    assert CLU.batch_noise_edge(tmp_path) is None


def test_batch_noise_edge_unreadable_file_is_none(tmp_path):
    (tmp_path / "batch_summary.json").write_text("{not json")
    assert CLU.batch_noise_edge(tmp_path) is None


def test_floors_fall_back_to_the_legacy_cps_values_without_an_edge():
    a, u, src = CLU.resolve_floors(None)
    assert (a, u) == (CLU.FLOOR_DEFAULT, CLU.UNASSIGNED_FLOOR_DEFAULT) == (200.0, 50.0)
    assert src.startswith("default")
    assert CLU.resolve_floors(0.0)[:2] == (200.0, 50.0)


def test_floors_scale_with_the_edge():
    a, u, src = CLU.resolve_floors(0.6)
    assert a == pytest.approx(CLU.FLOOR_X_EDGE * 0.6)
    assert u == pytest.approx(CLU.UNASSIGNED_FLOOR_X_EDGE * 0.6)
    assert src == "batch noise edge"
    # an Orbitrap-like edge of ~60 cps reproduces the legacy floors
    a60, u60, _ = CLU.resolve_floors(60.0)
    assert a60 == pytest.approx(200.0, rel=0.01) and u60 == pytest.approx(50.0, rel=0.01)
    # the multiples are parameters
    a2, u2, _ = CLU.resolve_floors(0.6, floor_x_edge=10.0, unassigned_floor_x_edge=2.0)
    assert (a2, u2) == pytest.approx((6.0, 1.2))


def test_an_explicit_floor_wins_over_the_edge():
    a, u, src = CLU.resolve_floors(0.6, floor=100.0)
    assert a == 100.0 and u == pytest.approx(CLU.UNASSIGNED_FLOOR_X_EDGE * 0.6)
    assert "caller" in src
    assert CLU.resolve_floors(None, unassigned_floor=7.0)[:2] == (200.0, 7.0)


def test_top_by_median_keeps_the_brightest_in_input_order():
    med = {"a": 5.0, "b": 50.0, "c": 1.0, "d": 20.0, "e": float("nan")}
    kept, over = CLU.top_by_median(list("abcde"), med, 2)
    assert kept == ["b", "d"] and over == ["a", "c", "e"]
    assert CLU.top_by_median(list("abcde"), med, None) == (list("abcde"), [])
    assert CLU.top_by_median(list("ab"), med, 5) == (["a", "b"], [])
    kept4, over4 = CLU.top_by_median(list("abcde"), med, 4)
    assert over4 == ["e"]                         # a non-finite median ranks last


# ---- a synthetic week of a TOF-scale batch -----------------------------------------

T0 = datetime(2025, 3, 3, tzinfo=timezone.utc)


def _week(scale, n_unknown=1):
    """A 7-day hourly batch (residual space): a 3-channel Assigned family, ten
    background channels with their own day-to-day patterns, `n_unknown`
    unassigned bins co-varying with the family. Heights are the Orbitrap-like
    values of the existing clustering test times `scale`."""
    n = 169
    hrs = np.arange(n, dtype=float)
    day = (hrs // 24).astype(int)
    daypat = np.array([0.0, 0.35, -0.25, 0.3, -0.35, 0.15, -0.2, 0.25])[day]
    wave = 0.25 * np.sin(2 * np.pi * hrs / 24)
    fam = 10 ** (wave + 0.15 * daypat)
    bg_pats = {j: np.array([np.cos(1.3 * j + 2.7 * dd) for dd in range(8)])[day] for j in range(10)}
    bg_mz = [230.0 + 3.7 * j for j in range(10)]
    unk_mz = [150.0 + 2.37 * k for k in range(n_unknown)]
    rng = np.random.default_rng(7)
    unk_gain = 1500.0 * (1.0 - 0.02 * np.arange(n_unknown))        # distinct medians
    rows = []
    for i in range(n):
        sid, t = f"L{i:03d}", T0 + timedelta(hours=i)
        rows += [(sid, t, 286.9, scale * 4000 * fam[i]), (sid, t, 302.9, scale * 3000 * fam[i]),
                 (sid, t, 272.9, scale * 2500 * fam[i])]
        for j, bmz in enumerate(bg_mz):
            rows.append((sid, t, bmz, scale * 1000 * 10 ** (wave[i] + 0.15 * bg_pats[j][i])))
        for k, umz in enumerate(unk_mz):
            rows.append((sid, t, umz, scale * unk_gain[k] * fam[i] * (1 + 0.03 * rng.standard_normal())))
    ts = pd.DataFrame(rows, columns=["sample_item_id", "datetime_utc", "mz", "height"])
    merged = pd.DataFrame(
        [{"neutral_formula": "C10H16O2", "adduct": "[M+Br]-", "mz": 286.9, "ion_score": .95, "tier": "Assigned"},
         {"neutral_formula": "C10H16O3", "adduct": "[M+Br]-", "mz": 302.9, "ion_score": .92, "tier": "Assigned"},
         {"neutral_formula": "C9H14O2", "adduct": "[M+Br]-", "mz": 272.9, "ion_score": .90, "tier": "Assigned"}]
        + [{"neutral_formula": f"C{8 + j}H{18 + 2 * j}O3", "adduct": "[M+Br]-", "mz": bmz,
            "ion_score": .85, "tier": "Candidate"} for j, bmz in enumerate(bg_mz)])
    return ts, merged


def _run_dir(path, merged, **summary):
    path.mkdir(parents=True, exist_ok=True)
    merged.to_csv(path / "merged_ledger.csv", index=False)
    (path / "per_file").mkdir(exist_ok=True)
    pf = merged.copy()
    pf["role"] = "M0"
    pf["ion_formula"] = ""
    pf.to_csv(path / "per_file" / "s00_ledger.csv", index=False)
    if summary:
        (path / "batch_summary.json").write_text(json.dumps(summary))
    return path


def test_a_tof_scale_batch_clusters_at_its_own_edge(tmp_path):
    ts, merged = _week(scale=1e-3)               # family 2.5-4 cps, unknown 1.5 cps
    old = _run_dir(tmp_path / "old", merged)     # a run dir without the edge key
    s_old = CLU.cluster_batch(str(old), ts, P.resolve("Br"), tag="T", log=lambda *a: None)["summary"]
    assert s_old["gates"]["assigned_clustering_floor_cps"] == 200.0
    assert s_old["gates"]["unassigned_median_cps_floor"] == 50.0
    assert s_old["gates"]["noise_edge_batch_cps"] is None
    assert s_old["assigned"]["n_dynamic_families"] == 0           # nothing clears 200 cps
    assert s_old["unassigned"]["n_after_brightness_persistence"] == 0

    new = _run_dir(tmp_path / "new", merged, noise_edge_batch_cps=0.3)
    s_new = CLU.cluster_batch(str(new), ts, P.resolve("Br"), tag="T", log=lambda *a: None)["summary"]
    g = s_new["gates"]
    assert g["noise_edge_batch_cps"] == pytest.approx(0.3)
    assert g["assigned_clustering_floor_cps"] == pytest.approx(CLU.FLOOR_X_EDGE * 0.3)
    assert g["unassigned_median_cps_floor"] == pytest.approx(CLU.UNASSIGNED_FLOOR_X_EDGE * 0.3)
    assert g["floor_source"] == "batch noise edge"
    assert s_new["assigned"]["n_dynamic_families"] >= 1
    assert s_new["unassigned"]["n_entered_union"] >= 1
    assert s_new["unassigned"]["n_in_families"] >= 1


def test_the_same_batch_at_orbitrap_scale_is_unchanged_without_an_edge(tmp_path):
    """No edge key: the floors are the legacy 200 / 50 cps, the explicit floor
    keeps working, and the summary reports the fallback."""
    ts, merged = _week(scale=1.0)
    d = _run_dir(tmp_path / "orbi", merged)
    s = CLU.cluster_batch(str(d), ts, P.resolve("Br"), tag="O", floor=100.0,
                          log=lambda *a: None)["summary"]
    assert s["gates"]["assigned_clustering_floor_cps"] == 100.0
    assert s["gates"]["unassigned_median_cps_floor"] == 50.0
    assert s["gates"]["floor_source"].startswith("default")
    assert s["assigned"]["n_dynamic_families"] >= 1


def test_an_explicit_edge_argument_overrides_the_summary(tmp_path):
    ts, merged = _week(scale=1e-3)
    d = _run_dir(tmp_path / "arg", merged, noise_edge_batch_cps=1000.0)
    s = CLU.cluster_batch(str(d), ts, P.resolve("Br"), tag="A", noise_edge_batch_cps=0.3,
                          log=lambda *a: None)["summary"]
    assert s["gates"]["noise_edge_batch_cps"] == pytest.approx(0.3)
    assert s["assigned"]["n_dynamic_families"] >= 1


def test_the_panel_axis_floor_follows_the_unassigned_floor(tmp_path, monkeypatch):
    """A fixed 50 cps bottom drew a few-cps family as an empty panel."""
    seen = []
    orig = CL.render_clusters

    def spy(rows, *a, **kw):
        seen.append(kw.get("ylim"))
        return orig(rows, *a, **kw)
    monkeypatch.setattr(CL, "render_clusters", spy)
    ts, merged = _week(scale=1e-3)
    d = _run_dir(tmp_path / "ax", merged, noise_edge_batch_cps=0.3)
    CLU.cluster_batch(str(d), ts, P.resolve("Br"), tag="Y", log=lambda *a: None)
    changing = seen[0]                            # the family pages render first
    assert changing is not None and changing[0] < 5.0 < changing[1]


def test_the_traced_unassigned_set_is_capped_by_median(tmp_path):
    ts, merged = _week(scale=1.0, n_unknown=12)
    d = _run_dir(tmp_path / "cap", merged)
    res = CLU.cluster_batch(str(d), ts, P.resolve("Br"), tag="C", floor=100.0, top_n=3,
                            log=lambda *a: None)
    un = res["summary"]["unassigned"]
    assert res["summary"]["gates"]["unassigned_top_n"] == 3
    assert un["n_entered_union"] == 3 and un["n_union_over_cap"] == 9
    assert un["n_varying_plotted"] <= 3
    assert un["n_varying_over_cap"] == un["n_union_over_cap"] - un["n_varying_plotted"] - un["n_flat_bunched"]
    tab = pd.read_csv(d / "tables" / "clusters_unassigned_C.csv")
    # the union keeps the three brightest unknowns (the lowest m/z carry the largest gain)
    assert sorted(tab.loc[tab.in_union, "mz"].round(2)) == [150.0, 152.37, 154.74]
    assert int(tab["over_cap"].sum()) == un["n_varying_over_cap"]
    # no cap: all twelve join the union
    d2 = _run_dir(tmp_path / "nocap", merged)
    un2 = CLU.cluster_batch(str(d2), ts, P.resolve("Br"), tag="C", floor=100.0, top_n=None,
                            log=lambda *a: None)["summary"]["unassigned"]
    assert un2["n_entered_union"] == 12 and un2["n_union_over_cap"] == 0


def test_top_n_must_be_positive(tmp_path):
    ts, merged = _week(scale=1.0)
    d = _run_dir(tmp_path / "bad", merged)
    with pytest.raises(ValueError):
        CLU.cluster_batch(str(d), ts, P.resolve("Br"), tag="B", top_n=0, log=lambda *a: None)
