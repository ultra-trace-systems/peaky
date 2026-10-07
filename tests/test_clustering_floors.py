"""Brightness floors of the time-series clustering follow the batch noise edge down.

The floors were absolute (200 cps for an assigned channel, 50 cps for an
unassigned bin). On a counting TOF (edge ~0.5-1 cps) few channels clear them
(a bromide/nitrate TOF batch: 10 assigned channels and 16 unassigned bins, 3
families, no unassigned cluster). They are now min(legacy floor, multiple x
`batch_summary.json['noise_edge_batch_cps']`) when the run records the edge, so
the edge only lowers them -- an Orbitrap batch with an edge at or above ~60 cps
keeps 200 / 50 cps -- and the legacy values when it does not; the traced
unassigned set is capped at the `top_n` brightest by median. The panel axes
follow the unassigned floor, and the PDF prints the floors and the cap.
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


@pytest.mark.parametrize("edge", [60.755, 61.209, 152.8, 201.16, 758.06])
def test_an_edge_above_60_cps_keeps_the_legacy_floors(edge):
    """The edge only lowers the floors: Orbitrap batches record edges of 35-758
    cps, and a scaled floor above 200 / 50 cps would thin their figures."""
    a, u, src = CLU.resolve_floors(edge)
    assert (a, u) == (CLU.FLOOR_DEFAULT, CLU.UNASSIGNED_FLOOR_DEFAULT) == (200.0, 50.0)
    assert src == "batch noise edge (capped at the default: assigned, unassigned)"
    # custom multiples are capped the same way
    assert CLU.resolve_floors(edge, floor_x_edge=50.0, unassigned_floor_x_edge=5.0)[:2] == (200.0, 50.0)


def test_an_edge_below_60_cps_lowers_the_floors():
    a, u, src = CLU.resolve_floors(35.07)
    assert a == pytest.approx(3.33 * 35.07) and u == pytest.approx(0.83 * 35.07)
    assert src == "batch noise edge"
    # between the two caps (3.33 x 60.1 > 200, 0.83 x 60.1 < 50) only one is capped
    a, u, src = CLU.resolve_floors(60.1)
    assert a == 200.0 and u == pytest.approx(0.83 * 60.1) and u < 50.0
    assert src.endswith("(capped at the default: assigned)")


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


def test_cps_round_keeps_sub_cps_medians_visible():
    assert CLU.cps_round(0.47162) == 0.472
    assert CLU.cps_round(2.649) == 2.65
    assert CLU.cps_round(1.353e-4) == 1.35e-4
    assert CLU.cps_round(57.26) == 57.3
    assert CLU.cps_round(123456.7) == 123457.0    # whole cps from 100 up, as before
    assert CLU.cps_round(0) == 0.0
    assert np.isnan(CLU.cps_round(float("nan")))


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


def _with_leftovers(ts, merged, scale):
    """Add three flat Candidate channels (they land on the background page) and an
    episodic unknown detected in 40 of the 169 samples (below the union's presence
    bar, so it takes the leftover unassigned path), both at 2000 x `scale` cps."""
    rng = np.random.default_rng(11)
    n = ts["sample_item_id"].nunique()
    flat_mz = [400.0 + 5.1 * j for j in range(3)]
    rows = []
    for i in range(n):
        sid, t = f"L{i:03d}", T0 + timedelta(hours=i)
        rows += [(sid, t, fmz, scale * 2000 * (1 + 0.02 * rng.standard_normal())) for fmz in flat_mz]
        if i % 4 == 0 and i < 160:
            rows.append((sid, t, 199.5, scale * 2000 * (1 + 0.5 * np.sin(i / 7))))
    ts = pd.concat([ts, pd.DataFrame(rows, columns=ts.columns)], ignore_index=True)
    merged = pd.concat([merged, pd.DataFrame(
        [{"neutral_formula": f"C{14 + j}H{20 + 2 * j}O6", "adduct": "[M+Br]-", "mz": fmz,
          "ion_score": .8, "tier": "Candidate"} for j, fmz in enumerate(flat_mz)])],
        ignore_index=True)
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
    # the CSVs keep the few-cps medians the floors rank by (not rounded to whole cps)
    for name in ("clusters_changing_T.csv", "clusters_unassigned_T.csv"):
        med = pd.read_csv(new / "tables" / name)["median_cps"]
        assert len(med) and (med < 10).all() and (med != med.round(0)).any(), (name, list(med))


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


def test_an_orbitrap_batch_with_a_high_edge_keeps_its_clusters(tmp_path):
    """An edge above ~60 cps leaves the floors, and so every count, as without it."""
    ts, merged = _week(scale=1.0, n_unknown=4)
    ts, merged = _with_leftovers(ts, merged, 1.0)
    s0 = CLU.cluster_batch(str(_run_dir(tmp_path / "noedge", merged)), ts, P.resolve("Br"),
                           tag="O", log=lambda *a: None)["summary"]
    s1 = CLU.cluster_batch(str(_run_dir(tmp_path / "edge", merged, noise_edge_batch_cps=758.06)),
                           ts, P.resolve("Br"), tag="O", log=lambda *a: None)["summary"]
    assert s1["gates"]["assigned_clustering_floor_cps"] == 200.0
    assert s1["gates"]["unassigned_median_cps_floor"] == 50.0
    assert s1["gates"]["floor_source"].startswith("batch noise edge (capped")
    assert s1["assigned"] == s0["assigned"] and s1["unassigned"] == s0["unassigned"]
    assert s0["assigned"]["n_dynamic_families"] >= 1


def test_the_panel_axes_follow_the_unassigned_floor(tmp_path, monkeypatch):
    """With the floors at the edge scale, a fixed 50 cps axis bottom would draw a
    few-cps family, background or unexplained trace as an empty panel: the family,
    background and unassigned pages all bottom out at the unassigned floor."""
    seen = []

    def spying(name):
        orig = getattr(CL, name)

        def spy(*a, **kw):
            path = next((x for x in a if isinstance(x, str)), "")
            seen.append((name, str(path).rsplit("/", 1)[-1], kw.get("ylim")))
            return orig(*a, **kw)
        monkeypatch.setattr(CL, name, spy)
    for name in ("render_clusters", "render_flat_panel", "render_grouped_flat"):
        spying(name)
    ts, merged = _with_leftovers(*_week(scale=1e-3), 1e-3)
    d = _run_dir(tmp_path / "ax", merged, noise_edge_batch_cps=0.3)
    s = CLU.cluster_batch(str(d), ts, P.resolve("Br"), tag="Y", log=lambda *a: None)["summary"]
    assert s["assigned"]["n_flat_background"] >= 1
    assert s["unassigned"]["n_below_presence"] >= 1          # the episodic unknown
    pages = {p: y for _, p, y in seen}
    for page in ("clusters_changing_Y", "clusters_flat_Y_p1.png", "clusters_unassigned_Y"):
        assert page in pages, sorted(pages)
        lo, hi = pages[page]
        assert lo < min(5.0, hi), (page, lo, hi)      # a 50 cps bottom: blank or inverted
        assert lo >= s["gates"]["unassigned_median_cps_floor"]


def test_the_pdf_prints_tof_floors_and_the_cap(monkeypatch):
    """The report printed the floors with :.0f ('median >= 0 cps' on a TOF) and its
    unexplained funnel did not know about the top_n cap."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from peaky.reporting import pdf_report as R

    pages = []
    monkeypatch.setattr(R, "_text_lines", lambda fig, lines, **kw: pages.append(lines))
    monkeypatch.setattr(R, "_close", lambda pdf, fig: plt.close(fig))
    gates = {"match_tol_ppm": 8.0, "unassigned_median_cps_floor": 0.47162,
             "assigned_clustering_floor_cps": 1.8922, "unassigned_top_n": 400,
             "entry_gate": "median", "min_trace_points": 8, "varying_cv_min": 0.3,
             "varying_burst_range": 1.7, "cluster_corr_r": 0.6, "merge_corr_r": 0.85,
             "min_cluster_members": 3, "big_change_fold": 5.0, "union_presence_min": 0.3}
    un = {"n_ts_bins": 9000, "n_unassigned_any": 8200, "n_after_brightness_persistence": 7695,
          "n_entered_union": 400, "n_union_over_cap": 1251, "n_isotope_rejected": 127,
          "n_in_families": 90, "n_varying_plotted": 400, "n_varying_over_cap": 5900,
          "n_flat_bunched": 868, "n_isotope_satellites_bunched": 127, "n_clusters": 14}
    ctx = {"tag": "T", "clusters": {"gates": gates, "unassigned": un}}
    R._unexplained_gate_page(ctx, None)
    R.methods(ctx, None)
    funnel = "\n".join(str(t) for _, t in pages[0])
    meth = "\n".join(str(t) for _, t in pages[1])
    assert "median ≥ 0.472 cps" in funnel
    assert "1251 more qualified" in funnel and "400 brightest" in funnel
    assert "5900 more VARYING" in funnel and "over_cap" in funnel
    assert "unexplained ≥0.472 cps median; assigned ≥1.89 cps" in meth
    # an uncapped Orbitrap run prints the legacy floors as before, with no cap lines
    pages.clear()
    gates.update(unassigned_median_cps_floor=50.0, assigned_clustering_floor_cps=200.0)
    un.update(n_union_over_cap=0, n_varying_over_cap=0)
    R._unexplained_gate_page(ctx, None)
    R.methods(ctx, None)
    funnel = "\n".join(str(t) for _, t in pages[0])
    assert "median ≥ 50 cps" in funnel and "over_cap" not in funnel and "more qualified" not in funnel
    assert "unexplained ≥50 cps median; assigned ≥200 cps" in "\n".join(str(t) for _, t in pages[1])


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
