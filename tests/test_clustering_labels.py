"""A co-varying family is named by an Assigned member when it has one.

The 'co-varies with X' label took the brightest member with a formula, whatever
its tier, so a family holding Assigned readings was often named by a Candidate
one. Now the Assigned members rank first (by median among themselves), then
the rest by median; an all-unassigned family stays 'novel'.
"""
import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from peaky.batch import clustering as CLU
from peaky.chem import profiles as P


def test_an_assigned_member_names_the_family_before_a_brighter_candidate():
    tier = {"A|[M+Br]-": "Candidate", "B|[M+Br]-": "Assigned", "C|[M+Br]-": "Assigned"}
    med = {"A|[M+Br]-": 5000.0, "B|[M+Br]-": 300.0, "C|[M+Br]-": 900.0, "?150.0000": 9000.0}
    assert CLU.family_label(list(med), tier, med, {"?150.0000"}) == "co-varies with C"


def test_without_an_assigned_member_the_brightest_formula_names_it():
    tier = {"A|[M+Br]-": "Candidate", "B|[M+Br]-": "Candidate"}
    med = {"A|[M+Br]-": 50.0, "B|[M+Br]-": 300.0, "?150.0000": 9000.0}
    assert CLU.family_label(list(med), tier, med, {"?150.0000"}) == "co-varies with B"


def test_an_all_unassigned_family_is_novel():
    med = {"?150.0000": 1.0, "?151.0000": 2.0}
    assert CLU.family_label(list(med), {}, med, set(med)) == "novel (no assigned anchor)"


def test_a_missing_or_nan_median_ranks_last_within_its_tier():
    tier = {"A|x": "Assigned", "B|x": "Assigned"}
    med = {"A|x": float("nan"), "B|x": 10.0}
    assert CLU.family_label(["A|x", "B|x"], tier, med) == "co-varies with B"
    assert CLU.family_label(["A|x", "B|x"], tier, {"B|x": 10.0}) == "co-varies with B"


def test_cluster_batch_labels_by_the_assigned_member(tmp_path):
    """A one-week batch: a three-channel family whose brightest channel is a
    Candidate reading; the label names the brightest Assigned channel."""
    t0 = datetime(2025, 3, 3, tzinfo=timezone.utc)
    n = 169
    hrs = np.arange(n, dtype=float)
    day = (hrs // 24).astype(int)
    daypat = np.array([0.0, 0.35, -0.25, 0.3, -0.35, 0.15, -0.2, 0.25])[day]
    wave = 0.25 * np.sin(2 * np.pi * hrs / 24)
    fam = 10 ** (wave + 0.15 * daypat)
    bg_pats = {j: np.array([np.cos(1.3 * j + 2.7 * dd) for dd in range(8)])[day] for j in range(10)}
    bg_mz = [230.0 + 3.7 * j for j in range(10)]
    rows = []
    for i in range(n):
        sid, t = f"L{i:03d}", t0 + timedelta(hours=i)
        rows += [(sid, t, 286.9, 4000 * fam[i]), (sid, t, 302.9, 3000 * fam[i]),
                 (sid, t, 272.9, 2500 * fam[i]), (sid, t, 199.1234, 1500 * fam[i])]
        for j, bmz in enumerate(bg_mz):
            rows.append((sid, t, bmz, 1000 * 10 ** (wave[i] + 0.15 * bg_pats[j][i])))
    ts = pd.DataFrame(rows, columns=["sample_item_id", "datetime_utc", "mz", "height"])
    merged = pd.DataFrame(
        [{"neutral_formula": "C10H16O2", "adduct": "[M+Br]-", "mz": 286.9, "ion_score": .95, "tier": "Candidate"},
         {"neutral_formula": "C10H16O3", "adduct": "[M+Br]-", "mz": 302.9, "ion_score": .92, "tier": "Assigned"},
         {"neutral_formula": "C9H14O2", "adduct": "[M+Br]-", "mz": 272.9, "ion_score": .90, "tier": "Assigned"}]
        + [{"neutral_formula": f"C{8 + j}H{18 + 2 * j}O3", "adduct": "[M+Br]-", "mz": bmz,
            "ion_score": .85, "tier": "Candidate"} for j, bmz in enumerate(bg_mz)])
    merged.to_csv(tmp_path / "merged_ledger.csv", index=False)
    (tmp_path / "per_file").mkdir()
    pf = merged.copy()
    pf["role"] = "M0"
    pf["ion_formula"] = ""
    pf.to_csv(tmp_path / "per_file" / "s00_ledger.csv", index=False)
    (tmp_path / "batch_summary.json").write_text(json.dumps({"noise_edge_batch_cps": 30.0}))

    res = CLU.cluster_batch(str(tmp_path), ts, P.resolve("Br"), tag="L", log=lambda *a: None)
    cc = pd.read_csv(tmp_path / "tables" / "clusters_changing_L.csv")
    fam_rows = cc[cc["neutral_formula"].isin(["C10H16O2", "C10H16O3", "C9H14O2"])]
    assert fam_rows["cluster"].nunique() == 1, cc[["neutral_formula", "cluster"]]
    cid = int(fam_rows["cluster"].iloc[0])
    assert res["labels"][cid] == "co-varies with C10H16O3"
    assert set(cc.loc[cc["cluster"] == cid, "cluster_label"]) == {"co-varies with C10H16O3"}
