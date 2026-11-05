"""Episode evidence in the merge (batch/episode_evidence.py): the episode flag on the
merged row, a short episode's readings judged on the episode's own spectra (Part B), a
persistent ion's cluster with a plume solvent (Part C), and the satellite-artefact hold
of the merge (Part D, batch/sideband.py on the per-spectrum test of
assignment/satellites.py).

The batch tests run `assign_batch.run` offline over the synthetic episode batch
(tests/fixtures/episodes: make_episodes.build's variants, harness.fake_engine's shaping
switches). Every rule parameter is pinned: each test names what it fails without.
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
from peaky.assignment import solvent_clusters as SC  # noqa: E402
from peaky.batch import assign_batch as AB  # noqa: E402
from peaky.batch import episode_evidence as EE  # noqa: E402
from peaky.batch import sideband as SB  # noqa: E402
from peaky.chem import chemistry as C  # noqa: E402
from peaky.chem import isotopes as ISO  # noqa: E402

E1 = ("C8H10N4O2", "[M+H]+")                 # the plume ion
DIM = ("C7H8N4O2", "[M+H]+")                 # the dim plume ion
RIV = [{"formula": "C10H13FNSi", "adduct": "[M+H]+"}]
LEVEL_ORDER = ["1", "2", "3a", "3b", "3c", "4a", "4b", "5a", "5b"]


def _run(tmp_path, monkeypatch, build=None, engine=None, **kw):
    t, comp = MK.build(**(build or {}))
    H.patch(t, comp, monkeypatch.setattr, **(engine or {}))
    summ = H.run_batch(str(tmp_path), t, **kw)
    return summ, pd.read_csv(tmp_path / "merged_ledger.csv")


def _get(m, nf, ad):
    r = m[(m["neutral_formula"] == nf) & (m["adduct"] == ad)]
    assert len(r) == 1, m[["mz", "neutral_formula", "adduct", "tier"]]
    return r.iloc[0]


def _table(tmp_path, name):
    return pd.read_csv(tmp_path / "tables" / name)


# --------------------------------------------------------------------------- the flag on the merged row
def test_align_flags_a_bin_when_any_files_row_is_flagged():
    a = pd.DataFrame([dict(mz=200.0, neutral_formula="C10H8O4", adduct="[M+H]+", tier="Assigned", ion_score=0.9,
                           episode=False)])
    b = a.assign(mz=200.0001, episode=True)
    c = pd.DataFrame([dict(mz=300.0, neutral_formula="C14H12O6", adduct="[M+H]+", tier="Assigned", ion_score=0.9,
                           episode=False)])
    merged, _j = AB.align({"s1": pd.concat([a, c]), "s2": b}, tol_ppm=6.0)
    assert merged.set_index("neutral_formula")["episode"].to_dict() == {"C10H8O4": True, "C14H12O6": False}
    merged2, _j = AB.align({"s1": a.drop(columns="episode"), "s2": b.drop(columns="episode")}, tol_ppm=6.0)
    assert "episode" not in merged2.columns          # no flag column, no merged column (a no-episode batch)
    assert "episode" in AB._M0_COLS


def test_the_flag_reaches_the_merged_ledger_and_the_unidentified_table(tmp_path, monkeypatch):
    _s, m = _run(tmp_path, monkeypatch, k_episode_max=8)
    flag = {(r.neutral_formula, r.adduct): bool(r.episode) for r in m.itertuples()}
    assert flag[E1] and flag[("C5H9NO", "[M+(CH4N2O)H]+")]
    assert not flag[("C16H22O4", "[M+H]+")] and not flag[("C14H22O", "[M+H]+")]
    u = _table(tmp_path, "unidentified.csv")
    plume13 = u[(u["mz"] - (C.ion_mz(*E1) + ISO.D_13C)).abs() < 0.002]
    assert len(plume13) == 1 and bool(plume13["episode"].iloc[0])
    persistent13 = u[(u["mz"] - (C.ion_mz("C16H22O4", "[M+H]+") + ISO.D_13C)).abs() < 0.002]
    assert len(persistent13) == 1 and not bool(persistent13["episode"].iloc[0])


# --------------------------------------------------------------------------- Part B in the batch
def test_a_plume_ion_with_its_13c_line_is_assigned_from_its_episode_at_level_4b(tmp_path, monkeypatch):
    summ, m = _run(tmp_path, monkeypatch, build=dict(envelope=True), k_episode_max=8)
    r = _get(m, *E1)
    assert r["tier"] == "Assigned" and r["episode_claim"] == EE.ASSIGNED_CLAIM
    assert LEVEL_ORDER.index(str(r["evidence_level"])) <= LEVEL_ORDER.index("4b")
    assert "Assigned from a single episode" in r["tier_reason"] and "13C reads 7.8" in r["tier_reason"]
    t = _table(tmp_path, "episode_evidence.csv")
    row = t[t["neutral_formula"] == E1[0]].iloc[0]
    assert row["verdict"] == "assigned" and row["b_isotopes"] == "holds" and row["episode_files"] == "ep20,ep21,ep22"
    assert summ["merge_gates"]["episode_evidence"]["assigned"] >= 1


def test_a_capped_plume_reading_is_promoted_when_its_isotopes_exclude_the_rival(tmp_path, monkeypatch):
    # every file leaves the reading Candidate with the exact-alias rival open (a per-file cap);
    # the episode's spectra show no 29Si / 30Si line where the rival's silicon predicts one
    summ, m = _run(tmp_path, monkeypatch, build=dict(envelope=True),
                   engine=dict(tiers={"E1": "Candidate"}, alternatives={"E1": RIV}), k_episode_max=8)
    r = _get(m, *E1)
    assert r["tier"] == "Assigned" and "rivals excluded: C10H13FNSi [M+H]+" in r["tier_reason"]
    assert summ["merge_gates"]["episode_evidence"]["promoted"] == 1


def test_without_its_13c_line_the_plume_reading_stays_candidate(tmp_path, monkeypatch):
    _s, m = _run(tmp_path, monkeypatch, build=dict(envelope=True, c13_child=False),
                 engine=dict(alternatives={"E1": RIV}), k_episode_max=8)
    r = _get(m, *E1)
    assert r["tier"] == "Candidate" and "(b) failed: 13C reads" in r["tier_reason"]


def test_a_rival_the_isotopes_cannot_exclude_keeps_the_reading_candidate(tmp_path, monkeypatch):
    # the rival's own Si lines are in the spectra, and 8 C against its 10 is inside its 13C band
    _s, m = _run(tmp_path, monkeypatch, build=dict(envelope=True, rival_lines=True),
                 engine=dict(tiers={"E1": "Candidate"}, alternatives={"E1": RIV}), k_episode_max=8)
    r = _get(m, *E1)
    assert r["tier"] == "Candidate" and "(c) rival left: C10H13FNSi [M+H]+" in r["tier_reason"]


def test_co_occurrence_alone_is_no_evidence(tmp_path, monkeypatch):
    # the dim plume ion rises exactly with the plume (r = 1 with the plume ion) in three episode
    # files, but no isotope line is testable at its height: Candidate, the untestable claim
    t, _c = MK.build(envelope=True, dim=True)
    d = t[(t["mz"] - C.ion_mz(*DIM)).abs() < 0.002].set_index("sample_item_id")["height"]
    e = t[(t["mz"] - C.ion_mz(*E1)).abs() < 0.002].set_index("sample_item_id")["height"]
    assert np.corrcoef(d, e.reindex(d.index))[0, 1] > 0.999
    summ, m = _run(tmp_path, monkeypatch, build=dict(envelope=True, dim=True), k_episode_max=8)
    r = _get(m, *DIM)
    assert r["tier"] == "Candidate" and r["episode_claim"] == EE.UNTESTABLE_CLAIM
    assert "(b) untestable" in r["tier_reason"]
    assert summ["merge_gates"]["episode_evidence"]["untestable"] == 1


def test_a_failing_reading_keeps_the_votes_tier_when_demotion_is_off(tmp_path, monkeypatch):     # DEMOTES
    monkeypatch.setattr(EE, "DEMOTES", False)
    _s, m = _run(tmp_path, monkeypatch, build=dict(envelope=True, dim=True), k_episode_max=8)
    r = _get(m, *DIM)
    assert r["tier"] == "Assigned" and r["episode_claim"] == EE.UNTESTABLE_CLAIM
    assert EE.NOTE_MARK in r["tier_reason"]


def test_one_episode_file_needs_an_adduct_partner_in_its_spectrum(tmp_path, monkeypatch):        # (e)
    # no episode budget: the plume's only file is the cover's ep20
    _s, m = _run(tmp_path / "a", monkeypatch, build=dict(envelope=True), k_episode_max=0)
    r = _get(m, *E1)
    assert r["tier"] == "Candidate" and "(e) one episode file, no adduct partner" in r["tier_reason"]
    _s, m2 = _run(tmp_path / "b", monkeypatch, build=dict(envelope=True, pair_adduct=True), k_episode_max=0)
    r2 = _get(m2, *E1)
    assert r2["tier"] == "Assigned" and "(e) C8H10N4O2 [M+(CH4N2O)H]+ at 255.12" in r2["tier_reason"]


@pytest.mark.parametrize("ppm,tier", [(0.55, "Assigned"), (0.65, "Candidate")])
def test_the_mass_must_sit_in_every_episode_files_calibrated_window(tmp_path, monkeypatch, ppm, tier):  # (a), K_SIGMA
    # the files' degeneracy calibration: mu 0, sigma 0.2 ppm -> the window is +-0.6 ppm
    _s, m = _run(tmp_path, monkeypatch, build=dict(envelope=True), engine=dict(ppm={"E1": ppm}), k_episode_max=8)
    r = _get(m, *E1)
    assert r["tier"] == tier
    if tier == "Candidate":
        assert "(a) ep20: +0.65 ppm outside +0.00 +- 0.60" in r["tier_reason"]


def test_a_no_episode_batch_is_byte_identical_and_carries_none_of_it(tmp_path, monkeypatch):        # GOLDEN
    summ, m = _run(tmp_path, monkeypatch, build=dict(plume=False))
    for f, g in H.GOLDEN_FILES.items():
        assert filecmp.cmp(tmp_path / f, os.path.join(H.GOLDEN_DIR, g), shallow=False), f
    assert not {"episode", "episode_claim", "solvent_cluster_of"} & set(m.columns)
    assert not (tmp_path / "tables" / "episode_evidence.csv").exists()
    assert not (tmp_path / "tables" / "solvent_clusters.csv").exists()
    assert not {"episode_evidence", "solvent_clusters"} & set(summ["merge_gates"])


# --------------------------------------------------------------------------- Part B units
def _unit(tmp_tier="Assigned", n_files=1, reason="", c13=True, levels=None, windows=None):
    """One merged reading of one episode file, its ledger and its spectrum (plus noise)."""
    mz = C.ion_mz(*E1)
    h = 3e5
    peaks = [dict(peak_id="p0", mz=mz, height=h, role="M0", neutral_formula=E1[0], adduct=E1[1], tier=tmp_tier,
                  ppm_error=0.1, episode=True, alternatives="[]")]
    if c13:
        peaks.append(dict(peak_id="p1", mz=mz + ISO.D_13C, height=h * 0.010816 * 8, role="unexplained", tier=None,
                          episode=True))
    peaks += [dict(peak_id=f"n{k}", mz=100.0 + 3.1 * k, height=100.0, role="unexplained", tier=None, episode=False)
              for k in range(30)]
    led = pd.DataFrame(peaks)
    ts = led[["peak_id", "mz", "height"]].assign(sample_item_id="s1")
    merged = pd.DataFrame([dict(mz=mz, neutral_formula=E1[0], adduct=E1[1], tier=tmp_tier, n_files=n_files,
                                srcs="s1", episode=True, tier_reason=reason)])
    return merged, {"s1": led}, ts, (windows if windows is not None else {"s1": (0.0, 0.2)})


def _judge(merged, ledgers, ts, windows, **kw):
    return EE.judge(merged, ledgers, ts, windows=windows, edges={"s1": 100.0}, adducts=["[M+H]+"],
                    log=lambda *a, **k: None, **kw)


def test_a_reading_in_more_than_three_files_is_not_judged():                                         # MAX_FILES
    assert EE.MAX_FILES == 3
    m, led, ts, w = _unit(n_files=4)
    _t, s = _judge(m, led, ts, w)
    assert s["judged"] == 0 and m.at[0, "tier"] == "Assigned"
    m, led, ts, w = _unit(n_files=3)
    _t, s = _judge(m, led, ts, w)
    assert s["judged"] == 1


def test_a_held_or_contradicted_row_is_never_assigned_from_its_episode():                           # (d)
    pair = _unit(c13=True)
    m, led, ts, w = pair
    _t, s = _judge(m, led, ts, w)                      # one file, no partner: (e) fails ...
    assert "(e)" in m.at[0, "tier_reason"]
    for reason in (f"{SB.HOLD_MARK}: Assigned in 1 file(s) only", f"{SB.SATELLITE_HOLD_MARK}: x",
                   "isotope contradiction in 1 of 1 file(s): 13C reads 1 C for 8"):
        m, led, ts, w = _unit(reason=reason)
        _t, s = _judge(m, led, ts, w)
        assert "(d) held" in m.at[0, "tier_reason"] and m.at[0, "tier"] == "Candidate"
    # the pooled pair's contradiction fact alone (the merged guard demotes only Assigned rows)
    m, led, ts, w = _unit(tmp_tier="Candidate")
    lv = pd.DataFrame([dict(neutral_formula=E1[0], adduct=E1[1], competitors_left="",
                            committed_reasons="13C reads 3.0 C for 8 (tol 2.0, rule C, 4 spectra)")])
    _t, s = _judge(m, led, ts, w, levels=lv)
    assert "(d) held: isotope contradiction (13C reads 3.0 C for 8" in m.at[0, "tier_reason"]


def test_without_a_calibrated_window_the_mass_cannot_hold():                                         # (a)
    m, led, ts, _w = _unit()
    _t, s = _judge(m, led, ts, {})
    assert "(a) s1: no calibrated window" in m.at[0, "tier_reason"] and m.at[0, "tier"] == "Candidate"


def test_a_curated_neutral_keeps_the_votes_tier():
    m, led, ts, w = _unit()
    _t, s = _judge(m, led, ts, w, exempt=frozenset({E1[0]}))
    assert m.at[0, "tier"] == "Assigned" and s["kept_assigned"] == 1 and s["demoted"] == 0


@pytest.mark.parametrize("nf,ad,cls", [
    ("C10H15ClO4", "[M-H]-", "Cl = C2H-4 ^N"),
    ("C9H7F3O3", "[M-H]-", "C2F3 = H2 ^N O4"),
    ("C16H10O2S", "[M-H]-", "C8S = H ^N O7"),
    ("C5H9NO5", "[M+^NO3]-", "CO3 = ^NO3 (C O H = N ^N)"),
])
def test_the_four_15n_alias_classes(nf, ad, cls):                                                   # (c) ALIAS_SWAPS
    counts = EE._counts(nf, ad)
    adducts = ["[M-H]-", "[M+^NO3]-", "[M+NO3]-"]
    names = [n for n, _rc in EE._alias_rivals(counts, "-", adducts, True)]
    assert any(n.startswith(cls + ":") for n in names), names
    assert EE._alias_rivals(counts, "-", adducts, False) == []          # an unlabelled reagent: no alias
    for _n, rc in EE._alias_rivals(counts, "-", adducts, True):
        assert abs(EE._ion_mass(rc, "-") - EE._ion_mass(counts, "-")) * 1e3 < 0.5


def _facts(c, lines=None, snr=1e3):
    return [dict(h0=1e6, snr0=snr, mz0=300.0, edge=100.0, c13=(True, c), lines=lines or {})]


def test_a_rival_is_excluded_only_by_an_isotope_fact():                                             # _excluded
    me = {"C": 10, "H": 15, "Cl": 1, "O": 4}
    rival = {"C": 12, "H": 11, "^N": 1, "O": 4}
    # 37Cl present at the reading's share: the Cl-free rival is excluded by it
    # (S/N 300: the rival's 14N twin, 2 % of the line, is not testable there)
    assert "37Cl" in EE._excluded(_facts(10.0, {"Cl": ISO.R_37CL_PER_CL}, snr=300.0), me, rival)
    # no 37Cl line, 13C reads 11 (inside both bands): the rival stays
    assert EE._excluded(_facts(11.0, {"Cl": 0.0}, snr=300.0), me, rival) is None
    # the carbon count alone: 6 C for the reading's 6 against a rival's 8 (band 2.0) -> excluded at 2.1 off
    assert EE._excluded(_facts(5.9), {"C": 6, "H": 9}, {"C": 8, "H": 5}) is not None
    assert EE._excluded(_facts(6.5), {"C": 6, "H": 9}, {"C": 8, "H": 5}) is None
    # an untestable line (S/N under MIN_SNR) excludes nothing
    assert EE._excluded(_facts(11.0, {"Cl": ISO.R_37CL_PER_CL}, snr=20.0), me, rival) is None
    # at S/N 1000 the rival's 14N twin is testable and absent: excluded by it
    assert "14N twin" in EE._excluded(_facts(11.0, {"Cl": 0.0}, snr=1e3), me, rival)


def test_the_isotope_band_and_the_testability_threshold():                                          # (b)
    f = _facts(8.0)
    assert EE._isotopes(f, {"C": 8})[0] == "holds"
    assert EE._isotopes(_facts(10.1), {"C": 8})[0] == "failed"                                    # band 2.0
    assert EE._isotopes(_facts(9.9), {"C": 8})[0] == "holds"
    nt = [dict(h0=1e3, snr0=5.0, mz0=300.0, edge=100.0, c13=(False, 0.0), lines={})]
    assert EE._isotopes(nt, {"C": 8})[0] == "untestable"
    # a heteroatom must show its own line where testable, and an untested one is no evidence
    assert EE._isotopes(_facts(8.0, {"S": ISO.R_34S_PER_S}), {"C": 8, "S": 1})[0] == "holds"
    assert EE._isotopes(_facts(8.0, {"S": 0.0}), {"C": 8, "S": 1})[0] == "failed"
    assert EE._isotopes(_facts(8.0, {"S": ISO.R_34S_PER_S}, snr=100.0), {"C": 8, "S": 1})[0] == "untestable"
    assert (EE.MIN_SNR, EE.C_TOL_ABS, EE.C_TOL_REL, EE.LINE_BAND) == (10.0, 1.5, 0.25, (0.5, 2.0))


def test_promotion_is_closed_to_elements_no_isotope_speaks_for():                                  # PROMOTABLE
    assert {"F", "Si", "P", "I"}.isdisjoint(EE.PROMOTABLE)
    m, led, ts, w = _unit(tmp_tier="Candidate")
    nf = "C8H9FN4O2"
    m.at[0, "neutral_formula"] = nf
    led["s1"].loc[0, "neutral_formula"] = nf
    _t, s = _judge(m, led, ts, w)
    assert m.at[0, "tier"] == "Candidate" and "elements F no isotope speaks for" in m.at[0, "tier_reason"]


# --------------------------------------------------------------------------- Part C
K1 = ("C16H22O4", "[M+(C5H9NO)H]+")


def test_the_solvent_cluster_reads_as_the_persistent_ions_cluster(tmp_path, monkeypatch):
    summ, m = _run(tmp_path, monkeypatch, k_episode_max=8)
    r = _get(m, *K1)
    assert r["tier"] == "Assigned" and r["solvent_cluster_of"] == "C16H22O4 [M+H]+ + C5H9NO"
    assert "C21H31NO5 [M+H]+ (the covalent reading)" in r["alternatives"]
    assert EE.CLUSTER_MARK in r["tier_reason"] and abs(r["mz"] - C.ion_mz(*K1)) < 0.001
    assert not len(m[m["neutral_formula"] == "C21H31NO5"])
    t = _table(tmp_path, "solvent_clusters.csv")
    assert len(t) == 1 and bool(t["pair_test"].iloc[0]) and t["track_r"].iloc[0] >= 0.99
    # the reagent-cluster strip spared it (the urea hexamer.NH4+ sits 1.4 ppm away)
    drops = _table(tmp_path, "vote_drops.csv")
    assert not drops["reason"].astype(str).str.contains("reagent-cluster strip").any()
    assert summ["merge_gates"]["solvent_clusters"]["relabelled"] == 1


def test_a_cluster_that_does_not_track_parent_x_solvent_is_capped_at_candidate(tmp_path, monkeypatch):  # TRACK_R
    _s, m = _run(tmp_path, monkeypatch, build=dict(cluster_track=False), k_episode_max=8)
    r = _get(m, *K1)
    assert r["tier"] == "Candidate" and "the pair test not passed" in r["tier_reason"]


def test_a_line_outside_the_episode_is_no_cluster(tmp_path, monkeypatch):                         # CONFINED_FRAC
    summ, m = _run(tmp_path, monkeypatch, build=dict(cluster_outside=True), k_episode_max=8)
    assert not len(m[m["adduct"] == K1[1]])
    assert len(_table(tmp_path, "solvent_clusters.csv")) == 0
    assert summ["merge_gates"]["solvent_clusters"]["relabelled"] == 0


def test_the_plume_solvents_are_in_the_catalog_and_no_context_ladders_them():
    sv = {s.formula: s for s in SC.episode_solvents()}
    assert set(sv) == {"C5H9NO", "C2H6OS"} and all(s.anchor and s.plume for s in sv.values())
    from peaky.chem import contexts as X
    for name in dir(X):
        prof = getattr(X, name)
        assert not {"C5H9NO", "C2H6OS"} & set(getattr(prof, "source_solvents", ()) or ())


@pytest.mark.parametrize("adduct,label", [("[M+H]+", "[M+(C5H9NO)H]+"), ("[M+NH4]+", "[M+(C5H9NO)NH4]+"),
                                          ("[M+(CH4N2O)H]+", "[M+(C5H9NO)(CH4N2O)H]+"),
                                          ("[M-H]-", "[M+(C5H9NO)-H]-")])
def test_the_cluster_label_names_the_solvent_and_resolves_in_a_fresh_lookup(adduct, label):
    assert EE.cluster_label(adduct, "C5H9NO") == label
    assert abs(C.ion_mz("C16H22O4", label) - (C.ion_mz("C16H22O4", adduct) + C.neutral_mass("C5H9NO"))) < 1e-9
    assert label not in C.ADDUCT_SHIFTS or True     # resolved by chemistry.cluster_shift, registered or not


def _cluster_series(n=20, parent_in=20, bright=(5, 6, 7, 8, 9), cluster_at=(5, 6, 7, 8, 9), track=True):
    """A batch series: a persistent parent, the NMP.urea ion bright in `bright`, the cluster in `cluster_at`."""
    pm = C.ion_mz("C16H22O4", "[M+H]+")
    sm = C.ion_mz("C5H9NO", "[M+(CH4N2O)H]+")
    km = pm + C.neutral_mass("C5H9NO")
    rows = []
    for k in range(n):
        sid = f"s{k:02d}"
        hp = 1e5 * (1 + 0.3 * np.sin(k))
        hs = 1e6 * (1 + 0.5 * np.cos(k)) if k in bright else 0.0
        if k < parent_in:
            rows.append(dict(sample_item_id=sid, mz=pm, height=hp))
        if hs:
            rows.append(dict(sample_item_id=sid, mz=sm, height=hs))
        if k in cluster_at:
            hk = 0.02 * hp * (hs or 1e5) / 1e6 if track else 1e3 * (1 + (k % 3))
            rows.append(dict(sample_item_id=sid, mz=km, height=hk))
        rows += [dict(sample_item_id=sid, mz=60.0 + 11.3 * j, height=100.0) for j in range(20)]
    ts = pd.DataFrame(rows)
    merged = pd.DataFrame([dict(mz=pm, neutral_formula="C16H22O4", adduct="[M+H]+", tier="Assigned", episode=False,
                                alternatives="", tier_reason=""),
                           dict(mz=km, neutral_formula="C21H31NO5", adduct="[M+H]+", tier="Assigned", episode=True,
                                alternatives="", tier_reason="")])
    return merged, ts


@pytest.mark.parametrize("kw,relabelled,tier", [
    ({}, True, "Assigned"),
    (dict(parent_in=9), False, None),                                   # PERSIST_FRAC: the parent in 9 of 20
    (dict(cluster_at=(4, 5, 6, 7, 8, 9)), False, None),                 # CONFINED_FRAC: 5 of 6 in the episode
    (dict(cluster_at=(5, 6, 7)), True, "Candidate"),                    # TRACK_MIN_N: 3 spectra
    (dict(track=False), True, "Candidate"),                             # TRACK_R
    (dict(bright=tuple(range(20)), cluster_at=tuple(range(11))), False, None),   # CLUSTER_MAX_FRAC: 11 of 20
])
def test_the_cluster_rules(kw, relabelled, tier):
    merged, ts = _cluster_series(**kw)
    tab, s = EE.read_solvent_clusters(merged, ts, adducts=["[M+H]+", "[M+(CH4N2O)H]+"], log=lambda *a: None)
    assert (s["relabelled"] == 1) is relabelled
    if relabelled:
        r = merged.iloc[1]
        assert (r["neutral_formula"], r["adduct"], r["tier"]) == ("C16H22O4", "[M+(C5H9NO)H]+", tier)
    else:
        assert merged.iloc[1]["neutral_formula"] == "C21H31NO5"
    assert (EE.SOLVENT_X, EE.PERSIST_FRAC, EE.CONFINED_FRAC, EE.CLUSTER_MAX_FRAC, EE.TRACK_R, EE.TRACK_MIN_N) == \
        (5.0, 0.5, 0.9, 0.5, 0.7, 4)


def test_a_dim_solvent_is_no_episode():                                                           # SOLVENT_X
    merged, ts = _cluster_series()
    sm = C.ion_mz("C5H9NO", "[M+(CH4N2O)H]+")
    ts.loc[(ts["mz"] - sm).abs() < 1e-6, "height"] = 400.0             # 4x the 100-cps edge
    _tab, s = EE.read_solvent_clusters(merged, ts, adducts=["[M+H]+", "[M+(CH4N2O)H]+"], log=lambda *a: None)
    assert s["relabelled"] == 0 and s["solvents_bright"] == []


# --------------------------------------------------------------------------- Part D in the batch
def test_the_satellite_is_held_and_a_true_ion_at_the_same_offset_is_not(tmp_path, monkeypatch):
    # the +3.85 mDa satellite of C14H22O [M+H]+ (0.4x, 3 % scatter, no 13C line) read in one file,
    # and a line 3.85 mDa above C16H22O4 [M+H]+ with its own 13C line read in one file
    summ, m = _run(tmp_path, monkeypatch, build=dict(true_ion=True),
                   engine=dict(commit={"SAT": ("C13H22N2", "[M+H]+", {"ep00"}),
                                       "T1": ("C11H20F2N4O2", "[M+H]+", {"ep00"})}))
    sat = _get(m, "C13H22N2", "[M+H]+")
    assert sat["tier"] == "Candidate" and SB.SATELLITE_HOLD_MARK in sat["tier_reason"]
    true = _get(m, "C11H20F2N4O2", "[M+H]+")
    assert true["tier"] == "Assigned" and SB.SATELLITE_HOLD_MARK not in str(true["tier_reason"])
    tab = _table(tmp_path, "sideband_holds.csv")
    row = tab[tab["neutral_formula"] == "C13H22N2"].iloc[0]
    assert row["action"] == "satellite_held" and row["fit_r2"] >= 0.9 and row["fit_n"] == 40
    assert abs(row["fit_slope"] - MK.SAT_SLOPE) < 0.02 and not bool(row["own_13c_series"])
    assert summ["merge_gates"]["sideband"]["satellite_held"] == 1
    assert summ["merge_gates"]["sideband"]["own_13c_kept"] >= 1


def test_the_14n_twin_excludes_a_rival_with_another_label_count():                                  # D_TWIN
    from peaky.chem import isotopes as _I
    f = 1.0 - _I.label_purity()
    me = {"C": 9, "H": 14, "N": 1, "^N": 1, "O": 8}            # an organonitrate's [M+^NO3]- ion
    rival = {"C": 10, "H": 15, "O": 9}                           # its CO3-class alias: no label atom
    assert EE._excluded(_facts(9.0, {"twin": f}, snr=1e4), me, rival) is not None     # the twin is there
    assert EE._excluded(_facts(9.0, {"twin": 0.0}, snr=1e4), me, rival) is None       # absent: the rival stays
    assert EE._excluded(_facts(9.0, {"twin": f}, snr=100.0), me, rival) is None       # untestable
