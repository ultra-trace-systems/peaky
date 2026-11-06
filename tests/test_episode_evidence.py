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
                 engine=dict(tiers={"E1": "Candidate"}, alternatives={"E1": RIV}), k_episode_max=8)
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
    summ, m = _run(tmp_path, monkeypatch, build=dict(envelope=True, dim=True),
                   engine=dict(tiers={"E2": "Candidate"}), k_episode_max=8)
    r = _get(m, *DIM)
    assert r["tier"] == "Candidate" and r["episode_claim"] == EE.UNTESTABLE_CLAIM
    assert "(b) untestable" in r["tier_reason"]
    assert summ["merge_gates"]["episode_evidence"]["untestable"] == 1


@pytest.mark.parametrize("mode,dim_tier,single_tier", [
    (None, "Assigned", "Assigned"),             # the default, "promote-only": never demotes
    ("measured", "Assigned", "Assigned"),       # an untestable line kept, (e) never demotes
    ("promote-only", "Assigned", "Assigned"),   # every failing vote-Assigned row kept with the note
    ("strict", "Candidate", "Candidate"),       # any failed clause demotes
])
def test_the_three_episode_evidence_modes(tmp_path, monkeypatch, mode, dim_tier, single_tier):     # MODE
    assert EE.MODE == "promote-only" and EE.MODES == ("measured", "promote-only", "strict")
    kw = {} if mode is None else {"episode_evidence": mode}
    # the dim plume ion, Assigned by the files: its ONLY failure is "(b) untestable" (3 episode files: (e) holds)
    summ, m = _run(tmp_path / "a", monkeypatch, build=dict(envelope=True, dim=True), k_episode_max=8, **kw)
    r = _get(m, *DIM)
    assert r["tier"] == dim_tier and r["episode_claim"] == EE.UNTESTABLE_CLAIM and EE.NOTE_MARK in r["tier_reason"]
    assert summ["merge_gates"]["episode_evidence"]["mode"] == (mode or "promote-only")
    # the mode the run used, on record at the top level (the run manifest's switch reads it)
    assert json.load(open(tmp_path / "a" / "batch_summary.json"))["episode_evidence_mode"] == (mode or "promote-only")
    t = _table(tmp_path / "a", "episode_evidence.csv")
    row = t[t["neutral_formula"] == DIM[0]].iloc[0]
    assert not bool(row["measured_failure"]) and row["mode"] == (mode or "promote-only")
    # the plume ion in ONE episode file, Assigned by it: (e) fails -- no measurement against it
    _s, m2 = _run(tmp_path / "b", monkeypatch, build=dict(envelope=True), k_episode_max=0, **kw)
    r2 = _get(m2, *E1)
    assert r2["tier"] == single_tier and "(e) one episode file, no adduct partner" in r2["tier_reason"]
    t2 = _table(tmp_path / "b", "episode_evidence.csv")
    row2 = t2[t2["neutral_formula"] == E1[0]].iloc[0]
    assert not bool(row2["measured_failure"]) and str(row2["measurement"]) in ("", "nan")


def test_an_unknown_mode_is_refused():
    m, led, ts, w = _unit()
    with pytest.raises(ValueError):
        _judge(m, led, ts, w, mode="lenient")


def test_the_cli_and_the_pipeline_carry_the_mode(monkeypatch):
    from peaky import cli
    from peaky import pipeline as PL
    p = cli.build_parser()
    assert p.parse_args(["batch", "--batch", "B"]).episode_evidence == "promote-only"
    got = {}

    class _Stop(Exception):
        pass

    def fake_run_batch(**kw):
        got.update(kw)
        raise _Stop
    monkeypatch.setattr(cli, "_require_creds", lambda: None)
    monkeypatch.setattr(PL, "run_batch", fake_run_batch)
    with pytest.raises(_Stop):
        cli.cmd_batch(p.parse_args(["batch", "--batch", "B", "--episode-evidence", "strict", "--no-progress"]))
    assert got["episode_evidence"] == "strict"
    with pytest.raises(SystemExit):
        p.parse_args(["batch", "--batch", "B", "--episode-evidence", "lenient"])


def test_one_episode_file_needs_an_adduct_partner_in_its_spectrum(tmp_path, monkeypatch):        # (e)
    # no episode budget: the plume's only file is the cover's ep20
    _s, m = _run(tmp_path / "a", monkeypatch, build=dict(envelope=True), engine=dict(tiers={"E1": "Candidate"}),
                 k_episode_max=0)
    r = _get(m, *E1)
    assert r["tier"] == "Candidate" and "(e) one episode file, no adduct partner" in r["tier_reason"]
    _s, m2 = _run(tmp_path / "b", monkeypatch, build=dict(envelope=True, pair_adduct=True),
                  engine=dict(tiers={"E1": "Candidate"}), k_episode_max=0)
    r2 = _get(m2, *E1)
    assert r2["tier"] == "Assigned" and "(e) C8H10N4O2 [M+(CH4N2O)H]+ at 255.12" in r2["tier_reason"]


@pytest.mark.parametrize("ppm,tier", [(0.55, "Assigned"), (0.65, "Candidate")])
def test_the_mass_must_sit_in_every_episode_files_calibrated_window(tmp_path, monkeypatch, ppm, tier):  # (a), K_SIGMA
    # the files' degeneracy calibration: mu 0, sigma 0.2 ppm -> the window is +-0.6 ppm
    _s, m = _run(tmp_path, monkeypatch, build=dict(envelope=True), engine=dict(tiers={"E1": "Candidate"}, ppm={"E1": ppm}),
                 k_episode_max=8)
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
        m, led, ts, w = _unit(tmp_tier="Candidate", reason=reason)
        _t, s = _judge(m, led, ts, w)
        assert "(d) held" in m.at[0, "tier_reason"] and m.at[0, "tier"] == "Candidate"
    # the pooled pair's contradiction fact alone (the merged guard demotes only Assigned rows)
    m, led, ts, w = _unit(tmp_tier="Candidate")
    lv = pd.DataFrame([dict(neutral_formula=E1[0], adduct=E1[1], competitors_left="",
                            committed_reasons="13C reads 3.0 C for 8 (tol 2.0, rule C, 4 spectra)")])
    _t, s = _judge(m, led, ts, w, levels=lv)
    assert "(d) held: isotope contradiction (13C reads 3.0 C for 8" in m.at[0, "tier_reason"]


def test_without_a_calibrated_window_the_mass_cannot_hold():                                         # (a)
    m, led, ts, _w = _unit(tmp_tier="Candidate")
    _t, s = _judge(m, led, ts, {})
    assert "(a) s1: no calibrated window" in m.at[0, "tier_reason"] and m.at[0, "tier"] == "Candidate"


def test_a_curated_neutral_keeps_the_votes_tier():
    # strict: the unit's file reads the mass 2 ppm outside its window
    m, led, ts, w = _unit()
    led["s1"].loc[0, "ppm_error"] = 2.0
    _t, s = _judge(m, led, ts, w, exempt=frozenset({E1[0]}), mode="strict")
    assert m.at[0, "tier"] == "Assigned" and s["kept_assigned"] == 1 and s["demoted"] == 0
    m, led, ts, w = _unit()
    led["s1"].loc[0, "ppm_error"] = 2.0
    _t, s = _judge(m, led, ts, w, mode="strict")
    assert m.at[0, "tier"] == "Candidate" and s["demoted"] == 1
    # measured: a 13C line read against the formula (a measurement)
    m, leds, ts = _unit2(tier="Assigned", c13=(8.0, 4.0))
    _t, s = _j2(m, leds, ts, exempt=frozenset({E1[0]}), mode="measured")
    assert m.at[0, "tier"] == "Assigned" and s["demoted"] == 0
    m, leds, ts = _unit2(tier="Assigned", c13=(8.0, 4.0))
    _t, s = _j2(m, leds, ts, mode="measured")
    assert m.at[0, "tier"] == "Candidate" and s["demoted"] == 1


def test_the_batch_wires_the_curated_exemption(tmp_path, monkeypatch):                          # one source
    # the solvent ion C5H9NO (a curated reference-list formula) and the plume ion both read 2 ppm outside
    # their windows -- an (a) failure, which strict demotes: the plume ion goes to Candidate, the curated
    # neutral keeps Assigned
    _s, m = _run(tmp_path, monkeypatch, build=dict(envelope=True), engine=dict(ppm={"S1": 2.0, "E1": 2.0}),
                 k_episode_max=8, episode_evidence="strict")
    assert _get(m, "C5H9NO", "[M+(CH4N2O)H]+")["tier"] == "Assigned"
    assert _get(m, *E1)["tier"] == "Candidate"
    from peaky.assignment import reflists as RL
    from peaky.batch import assign_batch as _AB  # noqa: F401
    cat = RL.load_catalog()
    summ = json.load(open(tmp_path / "batch_summary.json"))
    assert "C5H9NO" in EE.curated_exempt_from_summary(summ, cat)
    assert summ["episode_evidence_mode"] == "strict"
    # the replay's set is the batch's: the active lists' priors arrive by their recorded id + version
    L = cat["contaminants_keller2008"]
    fake = {"context": summ.get("context"), "reflists_context": {"active": [[L.id, L.data_version, "always active"]]}}
    got = EE.curated_exempt_from_summary(fake, cat)
    assert got == EE.curated_exempt(summ.get("context"), [L]) and set(L.formulas) <= got
    assert set(L.formulas) - EE.curated_exempt(summ.get("context"))        # the list adds formulas
    with pytest.raises(ValueError):
        EE.curated_exempt_from_summary({"reflists_context": {"active": [[L.id, "0.0", "x"]]}}, cat)


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


# --------------------------------------------------------------------------- Part C: the specific track (repair)
PM = C.ion_mz("C16H22O4", "[M+H]+")
SOLV_UREA = C.ion_mz("C5H9NO", "[M+(CH4N2O)H]+")
SOLV_DIMER = 2 * C.neutral_mass("C5H9NO") + C.M["H"] - C.M_E
KM = PM + C.neutral_mass("C5H9NO")
EP = tuple(range(4, 14))                     # a 10-spectrum episode in 24


def _hp(k, a=0.6):
    return 1e5 * (1 + a * np.sin(2.3 * k + 1))


def _hs(k):
    return 1e6 * (1 + 0.5 * np.cos(0.9 * k))


def _cseries(n=24, ep=EP, cluster_at=None, track=True, a=0.6, parent_in=None, parent_tier="Assigned",
             background=False, form=SOLV_UREA, dppm=0.0, decoys=0, seed=7):
    """A batch series: a persistent parent (amplitude `a` of its own variation), the NMP ion (`form`)
    in an episode `ep` (over a 1e3 background everywhere with `background`), the parent's NMP cluster
    in `cluster_at` (parent x solvent, or a flat pattern with track=False), `decoys` random persistent
    Assigned lines, noise picks at 100 cps (the edge)."""
    cluster_at = ep if cluster_at is None else cluster_at
    parent_in = n if parent_in is None else parent_in
    rng = np.random.default_rng(seed)
    # decoy persistent parents: nearly flat to moderately varying (CV ~0-0.25), as the field's persistent ions are
    dec = [(400.0 + 1.37 * j, rng.uniform(0.02, 0.35), rng.uniform(0.3, 2.5), rng.uniform(0, 6.3)) for j in range(decoys)]
    rows = []
    for k in range(n):
        sid = f"s{k:02d}"
        hs = _hs(k) if k in ep else (1e3 if background else 0.0)
        if k < parent_in:
            rows.append(dict(sample_item_id=sid, mz=PM, height=_hp(k, a)))
        if hs:
            rows.append(dict(sample_item_id=sid, mz=form, height=hs))
        if k in cluster_at:
            hk = 0.02 * _hp(k, a) * (_hs(k) if k in ep else 1e6) / 1e6 if track else 2e3 * (1 + 0.1 * (k % 2))
            rows.append(dict(sample_item_id=sid, mz=KM * (1 + dppm * 1e-6), height=hk))
        for m_, c_, w_, ph in dec:
            rows.append(dict(sample_item_id=sid, mz=m_, height=1e5 * (1 + c_ * np.sin(w_ * k + ph))))
        rows += [dict(sample_item_id=sid, mz=60.0 + 11.3 * j, height=100.0) for j in range(20)]
    ts = pd.DataFrame(rows)
    merged = pd.DataFrame([dict(mz=PM, neutral_formula="C16H22O4", adduct="[M+H]+", tier=parent_tier, episode=False,
                                alternatives="", tier_reason=""),
                           dict(mz=KM * (1 + dppm * 1e-6), neutral_formula="C21H31NO5", adduct="[M+H]+",
                                tier="Assigned", episode=True, alternatives="", tier_reason="")])
    return merged, ts, dec


def _read(merged, ts):
    return EE.read_solvent_clusters(merged, ts, adducts=["[M+H]+", "[M+(CH4N2O)H]+"], log=lambda *a: None)


def test_the_series_construction_is_what_the_cluster_tests_need():
    ep = list(EP)
    p = np.array([_hp(k) for k in ep]); s = np.array([_hs(k) for k in ep]); k_ = 0.02 * p * s / 1e6
    pt = EE.pair_test(k_, p, s)
    assert pt["specific"] and pt["r"] > 0.99 and pt["r"] - pt["r_solvent"] >= 0.2 and pt["cv_parent"] >= 0.3
    pf = np.array([_hp(k, 0.05) for k in ep])
    assert not EE.pair_test(0.02 * pf * s / 1e6, pf, s)["specific"]          # a flat parent


@pytest.mark.parametrize("kw,relabelled,tier", [
    ({}, True, "Assigned"),                                              # the specific track: the parent's tier
    (dict(a=0.05), True, "Candidate"),                                   # a flat parent: co-occurs with the solvent
    (dict(track=False), True, "Candidate"),                              # no track: co-occurs with the solvent
    (dict(cluster_at=EP[:5]), True, "Candidate"),                        # 5 shared spectra (< SPECIFIC_MIN_N 6)
    (dict(cluster_at=EP[:6]), True, "Assigned"),                         # 6
    (dict(cluster_at=EP[:3]), False, None),                              # TRACK_MIN_N: 3 shared spectra
    (dict(cluster_at=EP[:4]), True, "Candidate"),                        # ... 4
    (dict(parent_tier="Candidate"), False, None),                        # a KNOWN (Assigned) persistent ion only
    (dict(parent_in=11), False, None),                                   # PERSIST_FRAC: the parent in 11 of 24
    (dict(cluster_at=(2,) + EP), True, "Assigned"),                      # CONFINED_FRAC: 10 of 11 = 0.909 reads
    (dict(background=True), True, "Assigned"),                           # a solvent background everywhere + an episode
    (dict(background=True, cluster_at=EP + (20,)), True, "Assigned"),    # 10 of 11 spectra inside: confined
    (dict(background=True, cluster_at=EP[:8] + (20, 21)), False, None),  # 8 of 10 inside: not confined
    (dict(form=SOLV_DIMER), True, "Assigned"),                           # the solvent seen only as its dimer
    (dict(dppm=2.5), True, "Assigned"),                                  # inside the 3-ppm cluster window
    (dict(dppm=4.5), False, None),                                       # outside it
])
def test_the_cluster_rules(kw, relabelled, tier):
    merged, ts, _d = _cseries(**kw)
    tab, s = _read(merged, ts)
    assert (s["relabelled"] == 1) is relabelled, (kw, s)
    r = merged.iloc[1]
    if relabelled:
        assert (r["neutral_formula"], r["adduct"], r["tier"]) == ("C16H22O4", "[M+(C5H9NO)H]+", tier)
        if tier == "Candidate":
            assert f"solvent cluster of C16H22O4 [M+H]+: {EE.CO_OCCURS}" in r["tier_reason"]
            assert not bool(tab["pair_test"].iloc[0])
        else:
            assert "specifically" in r["tier_reason"] and bool(tab["pair_test"].iloc[0])
    else:
        assert r["neutral_formula"] == "C21H31NO5"


def test_cluster_confinement_reads_the_lines_presence():                                         # CONFINED_FRAC
    # no background: a spectrum outside the episode has no solvent; 10 of 11 cluster spectra inside reads,
    # 9 of 11 (0.82) does not
    m1, ts1, _d = _cseries(cluster_at=EP + (2,))
    assert _read(m1, ts1)[1]["relabelled"] == 1
    m2, ts2, _d = _cseries(cluster_at=EP + (1, 2))
    assert _read(m2, ts2)[1]["relabelled"] == 0


def test_cluster_max_frac():                                                                       # CLUSTER_MAX_FRAC
    # a 12-spectrum episode in 24 (half) and the cluster in it: reads; the cluster in those 12 + 1 more
    # (12 of 13 = 0.92 inside: confined) is in more than half the batch: no plume cluster
    ep = tuple(range(4, 16))
    m1, ts1, _d = _cseries(ep=ep)
    assert _read(m1, ts1)[1]["relabelled"] == 1
    m2, ts2, _d = _cseries(ep=ep, cluster_at=ep + (20,))
    assert _read(m2, ts2)[1]["relabelled"] == 0


def test_the_track_is_read_on_the_episodes_spectra_only():                                       # the episode's spectra
    # a parent of amplitude 0.44 tracks specifically over the 10 episode spectra (gain 0.29); the cluster is
    # also present once outside the episode, over the solvent's background, at parent x solvent there: read
    # over every spectrum that point would take the gain to 0.19 (not specific)
    merged, ts, _d = _cseries(a=0.44, background=True, cluster_at=EP + (15,))
    sel = (ts["sample_item_id"] == "s15") & ((ts["mz"] - KM).abs() < 1e-6)
    ts.loc[sel, "height"] = 0.02 * _hp(15, 0.44) * 1e3 / 1e6
    tab, s = _read(merged, ts)
    assert s["relabelled"] == 1 and merged.iloc[1]["tier"] == "Assigned" and tab["n_track"].iloc[0] == 10


def test_a_dim_solvent_is_no_episode():                                                           # SOLVENT_X
    merged, ts, _d = _cseries()
    ts.loc[(ts["mz"] - SOLV_UREA).abs() < 1e-6, "height"] = 400.0                # 4x the 100-cps edge
    tab, s = _read(merged, ts)
    assert s["relabelled"] == 0 and s["solvents_bright"] == []


def _decoy_rates(a):
    merged, ts, dec = _cseries(decoys=60, a=a)
    line = EE._series_lines(ts)
    ep = [f"s{k:02d}" for k in EP]
    ks, ss = line(KM).reindex(ep), line(SOLV_UREA).reindex(ep)
    own = EE.pair_test(ks, line(PM).reindex(ep), ss)
    plain = spec = 0
    for m_, *_r in dec:
        pt = EE.pair_test(ks, line(m_).reindex(ep), ss)
        plain += int(pt["r"] >= EE.TRACK_R)
        spec += int(pt["specific"])
    return own, plain / len(dec), spec / len(dec)


def test_random_persistent_parents_fail_the_specific_test():                                      # the decoy test
    # 60 random persistent lines (CV ~0-0.25) beside the episode-confined cluster: the SPECIFIC test is passed
    # by <= 10 % of them, while the cluster's own (varying) parent passes it ...
    own, plain, spec = _decoy_rates(0.6)
    assert own["specific"] and spec <= 0.1, (plain, spec)
    # ... and with a flat true parent (the field's case) the plain track r >= 0.7 is the solvent episode's
    # profile: most random parents pass it, none passes the specific test, the true one neither
    own_f, plain_f, spec_f = _decoy_rates(0.05)
    assert plain_f >= 0.5 and spec_f <= 0.1 and not own_f["specific"] and own_f["r"] >= EE.TRACK_R, (plain_f, spec_f)


def _pt_case(n=10, a=0.6, mix=0.0, noise=0.0, seed=0, start=0):
    """(k, p, s): a parent of amplitude a, the solvent, and k = (1 - mix) x parent x solvent + mix x solvent
    (scaled), with log-normal noise, over spectra start .. start + n - 1."""
    rng = np.random.default_rng(seed)
    kk = np.arange(start, start + n)
    p = _hp(kk, a); s = _hs(kk)
    prod = p * s / (p * s).mean()
    k = ((1 - mix) * prod + mix * s / s.mean()) * np.exp(rng.normal(0, noise, n))
    return k, p, s


def _find(pred, **fixed):
    """A deterministic search over mix / noise / seed for a pair-test case whose statistics satisfy `pred`."""
    for seed in range(40):
        for mix in np.linspace(0.0, 1.0, 21):
            for noise in (0.0, 0.05, 0.1, 0.2, 0.3, 0.5):
                case = _pt_case(mix=mix, noise=noise, seed=seed, **fixed)
                pt = EE.pair_test(*case)
                if pred(pt):
                    return pt
    raise AssertionError("no case found")


def test_the_pair_test_thresholds_at_their_boundaries():                                          # pair_test
    ok = dict(cv=lambda pt: pt["cv_parent"] >= 0.35)
    # r: 0.65 fails, 0.75 passes (gain and CV comfortably met)
    lo = _find(lambda pt: 0.62 <= pt["r"] <= 0.68 and pt["r"] - pt["r_solvent"] >= 0.3 and ok["cv"](pt))
    hi = _find(lambda pt: 0.72 <= pt["r"] <= 0.78 and pt["r"] - pt["r_solvent"] >= 0.3 and ok["cv"](pt))
    assert not lo["specific"] and hi["specific"]
    # the gain over the solvent alone: 0.15 fails, 0.25 passes (r >= 0.8)
    g1 = _find(lambda pt: pt["r"] >= 0.8 and 0.12 <= pt["r"] - pt["r_solvent"] <= 0.18 and ok["cv"](pt))
    g2 = _find(lambda pt: pt["r"] >= 0.8 and 0.22 <= pt["r"] - pt["r_solvent"] <= 0.28 and ok["cv"](pt))
    assert not g1["specific"] and g2["specific"]
    # the parent's CV: 0.25 fails, 0.35 passes
    c1 = EE.pair_test(*_pt_case(a=0.4))
    c2 = EE.pair_test(*_pt_case(a=0.55))
    assert 0.2 <= c1["cv_parent"] < 0.3 <= c2["cv_parent"] <= 0.4, (c1, c2)
    assert min(c1["r"] - c1["r_solvent"], c2["r"] - c2["r_solvent"]) >= 0.2 and not c1["specific"] and c2["specific"]
    # the number of spectra: 5 fails, 6 passes (the same spectra, the gain and the CV met by both)
    start = next(st for st in range(40) if EE.pair_test(*_pt_case(n=6, start=st))["specific"]
                 and (lambda q: q["r"] - q["r_solvent"] >= 0.2 and q["cv_parent"] >= 0.3)(
                     EE.pair_test(*_pt_case(n=5, start=st))))
    n5, n6 = EE.pair_test(*_pt_case(n=5, start=start)), EE.pair_test(*_pt_case(n=6, start=start))
    assert n5["r"] > 0.99 and n6["r"] > 0.99 and not n5["specific"] and n6["specific"]
    assert (EE.TRACK_R, EE.SPECIFIC_GAIN, EE.PARENT_CV_MIN, EE.SPECIFIC_MIN_N) == (0.7, 0.2, 0.3, 6)


def test_the_fixture_cluster_co_occurs_and_is_capped(tmp_path, monkeypatch):
    # the fixture's plume is 4 spectra: the cluster reads as the persistent ion's cluster, capped at Candidate
    summ, m = _run(tmp_path, monkeypatch, k_episode_max=8)
    r = _get(m, *K1)
    assert r["tier"] == "Candidate" and r["solvent_cluster_of"] == "C16H22O4 [M+H]+ + C5H9NO"
    assert f"solvent cluster of C16H22O4 [M+H]+: {EE.CO_OCCURS}" in r["tier_reason"]
    assert "C21H31NO5 [M+H]+ (the covalent reading)" in r["alternatives"]
    assert not len(m[m["neutral_formula"] == "C21H31NO5"])
    t = _table(tmp_path, "solvent_clusters.csv")
    assert len(t) == 1 and not bool(t["pair_test"].iloc[0]) and t["n_track"].iloc[0] == 4
    drops = _table(tmp_path, "vote_drops.csv")
    assert not drops["reason"].astype(str).str.contains("reagent-cluster strip").any()
    assert summ["merge_gates"]["solvent_clusters"]["relabelled"] == 1


def test_another_ion_at_the_cluster_mass_is_re_read_only_on_the_specific_test():
    for a, relabelled in ((0.05, 0), (0.6, 1)):
        merged, ts, _d = _cseries(a=a)
        merged.loc[1, ["neutral_formula", "adduct"]] = ["C16H30FN4O5", "[M+H]+"]      # another ion, 0.5 ppm away
        _t, s = _read(merged, ts)
        assert s["relabelled"] == relabelled
        assert merged.iloc[1]["neutral_formula"] == ("C16H22O4" if relabelled else "C16H30FN4O5")


# --------------------------------------------------------------------------- Part B: every file, use-site bars (repair)
SAT_S = ("C8H10N4O2S", "[M+H]+")


def _unit2(c13=(8.0, 8.0), ppm=(0.1, 0.1), s34=None, flags=(True, True), tier="Candidate", nf=E1[0], ad=E1[1],
           extra=None, ion_only=False, n_files=2, h=3e5, alts="[]", reason=""):
    """One merged reading in TWO episode files s1, s2: per file its 13C line at that carbon count
    (None: no line), its ppm_error, its 34S line at that multiple of its prediction (s34), its row's
    episode flag; `extra(sid)` adds rows."""
    mz = C.ion_mz(nf, ad)
    leds, tss = {}, []
    for j, sid in enumerate(("s1", "s2")):
        rows = [dict(peak_id=f"{sid}p0", mz=mz, height=h, role="M0", neutral_formula=nf, adduct=ad, tier=tier,
                     ppm_error=ppm[j], episode=flags[j], alternatives=alts)]
        if c13[j] is not None:
            rows.append(dict(peak_id=f"{sid}p1", mz=mz + ISO.D_13C, height=h * 0.010816 * c13[j], role="unexplained",
                             tier=None, episode=flags[j]))
        if s34 is not None and s34[j]:
            rows.append(dict(peak_id=f"{sid}p2", mz=mz + ISO.D_34S, height=h * ISO.R_34S_PER_S * s34[j],
                             role="unexplained", tier=None, episode=flags[j]))
        rows += (extra(sid, mz, h) if extra else [])
        rows += [dict(peak_id=f"{sid}n{k}", mz=100.0 + 3.1 * k, height=100.0, role="unexplained", tier=None,
                      episode=False) for k in range(30)]
        led = pd.DataFrame(rows)
        leds[sid] = led
        tss.append(led[["peak_id", "mz", "height"]].assign(sample_item_id=sid))
    merged = pd.DataFrame([dict(mz=mz, neutral_formula=nf, adduct=ad, tier=tier, n_files=n_files, srcs="s1,s2",
                                episode=True, tier_reason=reason, ion_only_of=("s1p9" if ion_only else pd.NA))])
    return merged, leds, pd.concat(tss, ignore_index=True)


def _j2(merged, leds, ts, adducts=("[M+H]+",), **kw):
    return EE.judge(merged, leds, ts, windows={"s1": (0.0, 0.2), "s2": (0.0, 0.2)}, edges={"s1": 100.0, "s2": 100.0},
                    adducts=list(adducts), log=lambda *a, **k: None, **kw)


def test_a_solvent_cluster_row_is_not_judged_by_part_b():                                       # Part C rows
    # a row Part C read as a plume-solvent cluster carries its own tier rule (capped unless specific): the
    # episode evidence never judges it, in any mode
    for mode in EE.MODES:
        m, leds, ts = _unit2(tier="Assigned", c13=(8.0, 4.0))
        m["solvent_cluster_of"] = "C9H16O [M+NH4]+ + C5H9NO"
        tab, s = _j2(m, leds, ts, mode=mode)
        assert len(tab) == 0 and s["judged"] == 0 and m.at[0, "tier"] == "Assigned"
        assert EE.NOTE_MARK not in str(m.at[0, "tier_reason"])


def test_the_two_file_unit_is_assigned_when_both_files_agree():
    m, leds, ts = _unit2()
    _t, s = _j2(m, leds, ts)
    assert m.at[0, "tier"] == "Assigned" and s["promoted"] == 1


def test_the_mass_window_binds_in_every_episode_file():                                           # (a) every file
    m, leds, ts = _unit2(ppm=(0.1, 2.0))
    _j2(m, leds, ts)
    assert m.at[0, "tier"] == "Candidate" and "(a) s2: +2.00 ppm outside" in m.at[0, "tier_reason"]


def test_one_contradicting_13c_file_fails_the_isotopes():                                         # (b) every file
    m, leds, ts = _unit2(c13=(8.0, 4.0))
    _j2(m, leds, ts)
    assert m.at[0, "tier"] == "Candidate" and "(b) failed" in m.at[0, "tier_reason"]


def test_one_file_without_its_heteroatom_line_fails_the_isotopes():                               # (b) S line, every file
    m, leds, ts = _unit2(nf=SAT_S[0], s34=(1.0, 0.0))
    _j2(m, leds, ts)
    assert m.at[0, "tier"] == "Candidate" and "(b) failed" in m.at[0, "tier_reason"]
    m2, leds2, ts2 = _unit2(nf=SAT_S[0], s34=(1.0, 1.0))
    _j2(m2, leds2, ts2)
    assert m2.at[0, "tier"] == "Assigned"


def test_a_rival_must_be_excluded_in_every_testable_file():                                       # (c) every file
    me, rival = {"C": 8, "H": 11, "N": 4, "O": 2}, {"C": 10, "H": 14, "N": 1, "O": 2}
    one = _facts(6.5)                                         # 6.5 C: the reading's band (2) holds, the rival's (2.5) not
    two = one + [dict(h0=1e6, snr0=1e3, mz0=300.0, edge=100.0, c13=(True, 8.5), lines={})]   # 8.5: both bands hold
    assert EE._excluded(one, me, rival) is not None
    assert EE._excluded(two, me, rival) is None


def _spec(h0, mz0=195.0877, extra=()):
    rows = [dict(peak_id="r", mz=mz0, height=h0)] + [dict(peak_id=f"x{i}", mz=mz0 + d, height=hh)
                                                     for i, (d, hh) in enumerate(extra)]
    return EE._Spectrum(pd.DataFrame(rows), 100.0)


def test_the_13c_line_is_testable_from_a_predicted_snr_of_10():                                   # (b) the S/N bar
    share = 8 * 0.010816
    for snr, testable in ((15.0, True), (7.0, False)):
        h0 = snr * 100.0 / share
        f = EE._file_facts(_spec(h0), {"peak_id": "r", "mz": 195.0877}, {"C": 8, "H": 11, "N": 4, "O": 2}, "+")
        assert f["c13"][0] is testable, (snr, f)


def test_a_heteroatom_line_is_testable_from_a_predicted_snr_of_10():                             # (b) heteroatom bar
    for snr, tested in ((15.0, 1), (7.0, 0)):
        h0 = snr * 100.0 / ISO.R_34S_PER_S
        f = [dict(h0=h0, snr0=float("nan"), mz0=300.0, edge=100.0, c13=(True, 8.0), lines={"S": ISO.R_34S_PER_S})]
        assert EE._line_state(f, "S", 1)[0] == tested


def test_the_adduct_partner_must_stand_at_3x_the_edge():                                         # (e) partner height
    pmz = C.ion_mz(E1[0], "[M+(CH4N2O)H]+")

    def part(x):
        return lambda sid, mz, h: [dict(peak_id=f"{sid}q", mz=pmz, height=x * 100.0, role="unexplained", tier=None,
                                        episode=True)] if sid == "s1" else []
    for x, ok in ((2.0, False), (4.0, True)):
        m, leds, ts = _unit2(extra=part(x), n_files=1)
        m.at[0, "srcs"] = "s1"
        _j2(m, leds, ts, adducts=("[M+H]+", "[M+(CH4N2O)H]+"))
        assert (m.at[0, "tier"] == "Assigned") is ok, (x, m.at[0, "tier_reason"])


def test_an_exotic_rival_counts_within_half_a_mda():                                             # ALIAS_MDA at the use site
    rv = [{"alternatives": json.dumps([{"formula": "C10H13FNSi", "adduct": "[M+H]+"}])}]
    m0 = C.ion_mz("C10H13FNSi", "[M+H]+")
    assert len(EE._named_rivals(rv, None, m0 + 0.0004)) == 1
    assert len(EE._named_rivals(rv, None, m0 + 0.0006)) == 0


def _iso_parent(sid, mz, h):
    """A brighter Assigned row whose 13C line falls on the reading (row <= 1.5 x its prediction)."""
    return [dict(peak_id=f"{sid}P", mz=mz - ISO.D_13C, height=3e6, role="M0", neutral_formula="C10H11NO3",
                 adduct="[M+H]+", tier="Assigned", ppm_error=0.1, episode=False, alternatives="[]")]


def test_the_low_file_hold_is_read_on_the_episode_rows():                                        # (d) the re-read
    # a Candidate reading (the hold never tested it) on the 13C line of a brighter Assigned row in both
    # episode files, no tier_reason mark: (d) holds it
    m, leds, ts = _unit2(c13=(None, None), extra=_iso_parent)
    _j2(m, leds, ts)
    assert "(d) held: isotopologue risk" in m.at[0, "tier_reason"]


def test_the_own_13c_skip_needs_every_episode_file():                                            # (d) every file
    # its own 13C line in s1 only, the isotopologue parent in both: still held (the hold's own rule)
    m, leds, ts = _unit2(c13=(8.0, None), extra=_iso_parent)
    _j2(m, leds, ts)
    assert "(d) held: isotopologue risk" in m.at[0, "tier_reason"]
    m2, leds2, ts2 = _unit2(c13=(8.0, 8.0), extra=_iso_parent)          # in both: an M0 of its own
    _j2(m2, leds2, ts2)
    assert "(d) held" not in m2.at[0, "tier_reason"]


@pytest.mark.parametrize("ad,ion_only", [("[M]+.", False), (E1[1], True)])
def test_radical_and_ion_only_readings_are_never_promoted(ad, ion_only):                         # promotion closed
    nf = "C8H11N4O2" if ad == "[M]+." else E1[0]
    m, leds, ts = _unit2(nf=nf, ad=ad, ion_only=ion_only)
    _j2(m, leds, ts, adducts=("[M+H]+", "[M]+."))
    assert m.at[0, "tier"] == "Candidate" and "never promoted" in m.at[0, "tier_reason"]


def test_the_file_rows_own_flag_is_required():                                                   # the episode files
    m, leds, ts = _unit2(flags=(False, False))
    _t, s = _j2(m, leds, ts)
    assert s["judged"] == 0 and m.at[0, "tier"] == "Candidate"
    m2, leds2, ts2 = _unit2(flags=(True, False))                         # one episode file: (e) fails, no partner
    _t, s2 = _j2(m2, leds2, ts2)
    assert s2["judged"] == 1 and "(1 file(s): s1)" in m2.at[0, "tier_reason"]


# --------------------------------------------------------------------------- measured: each clause (second repair)
RIVAL_ALT = json.dumps([{"formula": "C10H13FNSi", "adduct": "[M+H]+"}])


def _si_lines(fac):
    return lambda sid, mz, h: [dict(peak_id=f"{sid}si{k}", mz=mz + d, height=h * r * fac, role="unexplained", tier=None,
                                    episode=True) for k, (d, r) in enumerate(((ISO.D_29SI, ISO.R_29SI_PER_SI),
                                                                               (ISO.D_30SI, ISO.R_30SI_PER_SI)))]


@pytest.mark.parametrize("kw,demoted,clause", [
    (dict(ppm=(0.1, 2.0)), False, None),                       # (a) outside an existing window: never demotes
    (dict(c13=(8.0, 4.0)), True, "(b) 13C reads"),                                    # (b) a contradicting 13C line
    (dict(alts=RIVAL_ALT, extra=_si_lines(1.0)), True, "(c) favoured rival: C10H13FNSi [M+H]+ (its 30Si line"),
    (dict(reason="isotopologue risk: Assigned in 2 file(s) only"), True, "(d) held: isotopologue risk"),
    (dict(alts=RIVAL_ALT, extra=_si_lines(1.0), h=1500.0), False, None),           # (c) rival left, isotopes untestable
    (dict(n_files=1, flags=(True, False)), False, None),                             # (e) one file: never demotes
    (dict(nf="C8H9FN4O2"), False, None),                                              # elements F only: never demotes
])
def test_measured_demotes_only_on_a_contradicting_measurement(kw, demoted, clause):          # measured, each clause
    m, leds, ts = _unit2(tier="Assigned", **kw)
    if kw.get("n_files") == 1:
        m.at[0, "srcs"] = "s1"
    tab, s = _j2(m, leds, ts, mode="measured")
    row = tab.iloc[0]
    assert (m.at[0, "tier"] == "Candidate") is demoted, (kw, m.at[0, "tier_reason"])
    assert bool(row["measured_failure"]) is demoted
    if demoted:
        assert clause in row["measurement"], row["measurement"]
    else:
        assert row["measurement"] == "" and s["kept_assigned"] == 1
    assert row["prior_reason"] == kw.get("reason", "")


def test_measured_keeps_a_reading_with_no_calibrated_window():                                 # (a) untestable
    # s2 has no calibration: (a) cannot be read there -- no measurement, so measured keeps the vote's tier
    m, leds, ts = _unit2(tier="Assigned")
    tab, _s = EE.judge(m, leds, ts, windows={"s1": (0.0, 0.2)}, edges={"s1": 100.0, "s2": 100.0},
                       adducts=["[M+H]+"], mode="measured", log=lambda *a, **k: None)
    row = tab.iloc[0]
    assert "s2: no calibrated window" in row["a_mass"] and m.at[0, "tier"] == "Assigned"
    assert not bool(row["measured_failure"]) and row["measurement"] == ""
    m, leds, ts = _unit2(tier="Assigned")
    EE.judge(m, leds, ts, windows={"s1": (0.0, 0.2)}, edges={"s1": 100.0, "s2": 100.0}, adducts=["[M+H]+"],
             mode="strict", log=lambda *a, **k: None)
    assert m.at[0, "tier"] == "Candidate"


def test_strict_demotes_the_untestable_and_scope_failures_too():
    for kw in (dict(alts=RIVAL_ALT, extra=_si_lines(1.0), h=1500.0), dict(nf="C8H9FN4O2")):
        m, leds, ts = _unit2(tier="Assigned", **kw)
        _j2(m, leds, ts, mode="strict")
        assert m.at[0, "tier"] == "Candidate"


def test_judge_defaults_to_promote_only():                                                       # judge(mode=None)
    # NEGATIVE CONTROL of the default: a 13C contradiction that measured demotes stays Assigned by default
    m, leds, ts = _unit2(tier="Assigned", c13=(8.0, 4.0))
    tab, s = _j2(m, leds, ts)
    assert s["mode"] == "promote-only" and m.at[0, "tier"] == "Assigned" and tab.iloc[0]["mode"] == "promote-only"
    assert bool(tab.iloc[0]["measured_failure"]) and "(b) 13C reads" in tab.iloc[0]["measurement"]
    m, leds, ts = _unit2(tier="Assigned", c13=(8.0, 4.0))
    _j2(m, leds, ts, mode="measured")
    assert m.at[0, "tier"] == "Candidate"


def test_measured_never_demotes_on_a_window_miss():                                             # (a) never demotes
    # NEGATIVE CONTROL: both files read the mass far outside their windows, every other clause holds:
    # measured keeps the vote's Assigned (no measurement), strict demotes it
    m, leds, ts = _unit2(tier="Assigned", ppm=(2.0, 2.0))
    tab, s = _j2(m, leds, ts, mode="measured")
    row = tab.iloc[0]
    assert m.at[0, "tier"] == "Assigned" and s["demoted"] == 0 and "outside" in row["a_mass"]
    assert not bool(row["measured_failure"]) and row["measurement"] == ""
    m, leds, ts = _unit2(tier="Assigned", ppm=(2.0, 2.0))
    _j2(m, leds, ts, mode="strict")
    assert m.at[0, "tier"] == "Candidate"


def test_a_favoured_rival_needs_its_fact_in_every_testable_file():                            # _favours, every file
    me, rival = {"C": 8, "H": 11, "N": 4, "O": 2}, {"C": 10, "H": 14, "N": 1, "O": 2}
    near = _facts(9.6)                                      # 9.6 C: nearer 10 than 8, inside the rival's band
    assert EE._favours(near, me, rival) is not None
    assert EE._favours(near + _facts(8.2), me, rival) is None
    assert EE._favours(_facts(8.9), me, rival) is None      # nearer the reading
    # nearer a C30 rival than the C8 reading, but outside the rival's own band (7.5): no measured fact for it
    far = {"C": 30, "H": 40, "N": 1, "O": 2}
    assert EE._favours(_facts(20.0), me, far) is None
    assert EE._favours(_facts(27.0), me, far) is not None


def _find_pt(pred, n=10):
    for seed in range(60):
        for mix in np.linspace(0.0, 1.0, 41):
            for noise in (0.0, 0.03, 0.05, 0.1, 0.2):
                for a in (0.45, 0.55, 0.65, 0.8):
                    for st in range(0, 12):
                        case = _pt_case(n=n, a=a, mix=mix, noise=noise, seed=seed, start=st)
                        pt = EE.pair_test(*case)
                        if pred(pt, case):
                            return pt, case
    raise AssertionError("no case")


def test_the_solvent_alone_tracking_better_is_no_specific_track():                           # the sign of the gain
    # r(line, solvent alone) exceeds r(line, parent x solvent) by >= 0.22: the parent does not carry the line
    pt, _c = _find_pt(lambda q, c: q["r"] >= 0.7 and q["r_solvent"] - q["r"] >= 0.22 and q["cv_parent"] >= 0.3)
    assert not pt["specific"]


def test_the_parent_cv_is_the_population_one():                                                # CV, ddof 0
    # at n = 6 a parent with population CV 0.28 (sample CV >= 0.3) is not varying enough
    def ok(q, c):
        p = c[1]
        return (q["r"] >= 0.9 and q["r"] - q["r_solvent"] >= 0.25 and 0.27 <= p.std() / p.mean() < 0.3
                and p.std(ddof=1) / p.mean() >= 0.3)
    pt, case = _find_pt(ok, n=6)
    assert abs(pt["cv_parent"] - case[1].std() / case[1].mean()) < 1e-12 and not pt["specific"]


def test_the_merged_isotope_guard_reads_the_votes_tiers_before_the_episode_evidence(tmp_path, monkeypatch):
    """Step 6e (the merged isotope guard) runs before step 6g (the episode evidence): the guard tests the merged
    rows Assigned by the vote, so a reading the episode evidence then promotes is not among its tested rows (the
    guard after the promotion would test it too)."""
    summ, m = _run(tmp_path, monkeypatch, build=dict(envelope=True),
                   engine=dict(tiers={"E1": "Candidate"}, alternatives={"E1": RIV}), k_episode_max=8)
    ee, g = summ["merge_gates"]["episode_evidence"], summ["merge_gates"]["isotope_guard"]
    assert ee["promoted"] == 1 and _get(m, *E1)["tier"] == "Assigned"
    n_assigned = int((m["tier"] == "Assigned").sum())
    assert g["tested"] == n_assigned - ee["promoted"] + ee["demoted"], (g["tested"], n_assigned, ee)
    assert summ["isotope_guard_mode"] == "note" and summ["episode_evidence_mode"] == EE.MODE
