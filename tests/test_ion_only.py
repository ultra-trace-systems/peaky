"""The ION-ONLY bucket (C7): the +1.0078 Da electron-attachment line beside a
committed [M-H]- acid, committed as a Candidate "[M]-." whose composition is
pinned while the ionization process and the neutral stay open.

Pins: the stage's placement and gate; the profile knob and its copy rule; the
commit itself (row shape, link, commentary, nothing existing moves); the two
separability guards (the calibrated gate vs the 13C/+H gap, the picker's
resolved-pair evidence); the evidence levels (own 13C -> 4d, else 5a; never a
second channel or a corroboration in either direction; the reference script
agrees); the batch path (the merged row carries the link, batch_summary counts
the bucket); publish (no mechanism for the adduct -> null); the scorecard
(counted on its own, kept out of the Candidate tile).

Run: pytest tests/test_ion_only.py -q
"""

from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from peaky.assignment import assign as A
from peaky.assignment import cleanup as CL
from peaky.assignment import evidence as EV
from peaky.assignment import ledger as L
from peaky.assignment.passes import config as PCfg
from peaky.assignment.passes import postprocess as PP
from peaky.chem import chemistry as C
from peaky.chem import profiles as P
from peaky.io import io_mascope as IO
from peaky.io import publish as PUB

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import level_ledger as LL  # noqa: E402

D13C = 1.0033548
X = "C10H16O5"                       # the acid: C10H15O5- at 215.0925
MZ_PARENT = C.ion_mz(X, "[M-H]-")
MZ_EA = C.ion_mz(X, "[M]-.")         # the radical anion, +1.00783 above the parent
#: dim unexplained doublets 4.0 mDa apart: the picker resolves the 13C/+H spacing here
RESOLVED_PAIRS = [(180.1000, 180.1040), (190.2000, 190.2040), (200.3000, 200.3040), (230.1000, 230.1040)]


def _cfg(**kw) -> PCfg.PassConfig:
    base = dict(height_cutoff_cps=1.0, cal_mu=0.0, cal_sigma=0.3, ion_only_channels=("[M]-.",))
    base.update(kw)
    return PCfg.PassConfig(**base)


def _ledger(*, with_13c=True, with_ea=True, ea_13c=True, resolved=True, ea_ppm=0.0,
            extra=()) -> pd.DataFrame:
    """A finished ledger: the acid X on [M-H]- (Assigned, its 13C child), the
    +H line as an unexplained peak (with its own 13C), a second unrelated
    acid, and the resolved-pair filler that shows the picker resolves 4 mDa."""
    rows = [("A", MZ_PARENT, 40_000.0)]
    if with_13c:
        rows.append(("A13", MZ_PARENT + D13C, 4_400.0))
    if with_ea:
        rows.append(("H", MZ_EA * (1 + ea_ppm * 1e-6), 5_200.0))
        if ea_13c:
            rows.append(("H13", MZ_EA + D13C, 560.0))
    rows.append(("B", C.ion_mz("C8H12O4", "[M-H]-"), 9_000.0))
    if resolved:
        for k, (a, b) in enumerate(RESOLVED_PAIRS):
            rows += [(f"r{k}a", a, 30.0), (f"r{k}b", b, 25.0)]
    rows += list(extra)
    peaks = pd.DataFrame([{"peak_id": p, "mz": m, "height": h} for p, m, h in rows])
    led = L.new_ledger(peaks)
    L.commit_assignment(led, "A", neutral_formula=X, adduct="[M-H]-", ion_formula="C10H15O5-",
                        ion_score=0.96, compound_score=0.95, ppm_error=0.1, pass_no=1,
                        method="cheminfo+grid", confidence="High", commentary="Pass 1")
    if with_13c:
        L.attach_isotopologue(led, "A13", "A", iso_label="13C", iso_match_score=0.9)
    L.commit_assignment(led, "B", neutral_formula="C8H12O4", adduct="[M-H]-", ion_formula="C8H11O4-",
                        ion_score=0.9, compound_score=0.9, ppm_error=-0.2, pass_no=1,
                        method="cheminfo+grid", confidence="Good", commentary="Pass 1")
    led["tier"] = led["tier"].astype(object)
    led.loc[led["role"] == L.ROLE_M0, "tier"] = "Assigned"
    led["below_assignability"] = False
    return led


def _row(led, pid):
    return led.loc[led["peak_id"] == pid].iloc[0]


# --------------------------------------------------------------------------- placement
def test_stage_sits_after_reflist_rescue_before_the_final_sweep_and_is_gated_on_the_channel():
    names = [s.name for s in A._STAGES]
    i = names.index("ion_only")
    assert names[i - 1] == "reflist_rescue" and names[i + 1] == "iso_env_final"
    assert names.index("evidence") > i and names.index("timeseries") > i
    stage = A._STAGES[i]
    assert stage.safe is False
    on = types.SimpleNamespace(cfg=_cfg())
    off = types.SimpleNamespace(cfg=_cfg(ion_only_channels=()))
    unset = types.SimpleNamespace(cfg=PCfg.PassConfig())
    assert stage.when(on) and not stage.when(off) and not stage.when(unset)


# --------------------------------------------------------------------------- the knob
def test_nitrate_profiles_declare_the_channel_and_compose_unions_it():
    assert P.NO3.ion_only_channels == ("[M]-.",) and P.NO3_15N.ion_only_channels == ("[M]-.",)
    assert P.BR.ion_only_channels == () and P.UR.ion_only_channels == () and P.IODIDE.ion_only_channels == ()
    assert P.compose([P.NO3, P.BR]).ion_only_channels == ("[M]-.",)
    assert P.compose([P.BR, P.NO3]).ion_only_channels == ("[M]-.",)
    assert P.compose([P.NO3, P.NO3_15N]).ion_only_channels == ("[M]-.",)     # once, not twice
    assert "ion_only_channels" in P._CONFIG_FIELDS
    prof = P.from_dict({"name": "T", "label": "t", "polarity": "-", "adducts": ["[M-H]-"],
                        "normaliser": "tic", "reagent_ion_re": None, "ranges": "C0-10 H0-20 O0-5",
                        "detect_adduct": None, "ion_only_channels": ["[M]-."]})
    assert prof.ion_only_channels == ("[M]-.",)
    assert P.from_dict({"name": "U", "label": "u", "polarity": "-", "adducts": ["[M-H]-"],
                        "normaliser": "tic", "reagent_ion_re": None, "ranges": "C0-10",
                        "detect_adduct": None}).ion_only_channels == ()


def test_apply_ion_only_channels_copies_the_profile_unless_the_cfg_already_chose():
    cfg = PCfg.PassConfig()
    assert cfg.ion_only_channels is None                      # unset, not ()
    assert P.apply_ion_only_channels(cfg, P.NO3) == ("[M]-.",) and cfg.ion_only_channels == ("[M]-.",)
    off = PCfg.PassConfig(ion_only_channels=())               # an explicit "off" outranks the profile
    assert P.apply_ion_only_channels(off, P.NO3) == () and off.ion_only_channels == ()
    none = PCfg.PassConfig()
    assert P.apply_ion_only_channels(none, None) == () and none.ion_only_channels == ()
    assert "ion_only_channels" not in PCfg.PassConfig.RUNTIME_FIELDS   # a knob, fingerprinted
    lines = []
    P.apply_ion_only_channels(PCfg.PassConfig(), P.NO3_15N, log=lines.append)
    assert lines and "[M]-." in lines[0] and "NO3_15N" in lines[0]


def test_new_ledgers_carry_the_link_column_as_na():
    led = L.new_ledger(pd.DataFrame({"peak_id": ["p"], "mz": [100.0], "height": [1.0]}))
    assert "ion_only_of" in led.columns and pd.isna(led["ion_only_of"].iloc[0])


# --------------------------------------------------------------------------- the commit
def test_commits_the_radical_anion_beside_its_acid_and_nothing_else_moves():
    led = _ledger()
    before = led.copy()
    lines = []
    out = CL.commit_ion_only_electron_attachment(led, _cfg(), log=lines.append)
    assert out["ion_only_committed"] == 1 and out["ion_only_parents"] == 2 and out["ion_only_candidates"] == 1
    assert out["ion_only_skipped_gate"] == 0 and out["ion_only_skipped_unresolved"] == 0
    h = _row(led, "H")
    assert h["role"] == L.ROLE_M0 and h["neutral_formula"] == X and h["adduct"] == "[M]-."
    assert h["ion_formula"] == "C10H16O5-" and h["tier"] == "Candidate"
    assert h["method"] == CL.ION_ONLY_METHOD and h["confidence"] == "Good (ion only)"
    assert h["ion_only_of"] == "A" and int(h["pass_no"]) == 9
    assert pd.isna(h["anchor_peak_id"]) and pd.isna(h["series_unit"]) and not bool(h["locked"])
    assert h["below_assignability"] is False or h["below_assignability"] == False  # noqa: E712
    assert "mDa" in h["commentary"] and "0.13x the parent" in h["commentary"] and "peak A" in h["commentary"]
    assert abs(float(h["ppm_error"])) < 1e-6 and abs(float(h["mz"]) - MZ_EA) < 1e-9
    assert "ion-only" in h["tier_reason"]
    # the parent and everything else are untouched
    for pid in ("A", "A13", "B", "r0a"):
        pd.testing.assert_series_equal(_row(before, pid), _row(led, pid), check_names=False)
    assert L.validate(led) == []
    assert lines and "1 [M]-. row(s) committed" in lines[-1]
    # a second pass finds nothing left to claim
    out2 = CL.commit_ion_only_electron_attachment(led, _cfg(), log=lambda *a: None)
    assert out2["ion_only_committed"] == 0 and out2["ion_only_candidates"] == 0


def test_the_final_sweep_claims_the_new_rows_own_13c_and_the_levels_follow():
    led = _ledger()
    parent_before = EV.compute_levels(led).set_index("peak_id").loc["A", "evidence_level"]
    CL.commit_ion_only_electron_attachment(led, _cfg(), log=lambda *a: None)
    PP.complete_isotope_envelopes(led, _cfg(), log=lambda *a: None)
    assert _row(led, "H13")["role"] == L.ROLE_ISO and _row(led, "H13")["parent_peak_id"] == "H"
    EV.apply_levels(led)
    by = led.set_index("peak_id")
    assert by.loc["A", "evidence_level"] == parent_before == "4b"          # unchanged
    assert by.loc["H", "evidence_level"] == "4d"
    assert by.loc["H", "level_reason"].startswith("4d: ion-only channel")
    assert "ion_only" in by.loc["H", "evidence_axes"] and "iso" in by.loc["H", "evidence_axes"]
    assert pd.isna(by.loc["H13", "evidence_level"])
    # without its own satellite the row is exact mass only
    led2 = _ledger(ea_13c=False)
    CL.commit_ion_only_electron_attachment(led2, _cfg(), log=lambda *a: None)
    EV.apply_levels(led2)
    assert led2.set_index("peak_id").loc["H", "evidence_level"] == "5a"
    assert led2.set_index("peak_id").loc["H", "level_reason"].startswith("5a: ion-only channel")


# --------------------------------------------------------------------------- the guards
def test_a_gate_wider_than_half_the_gap_skips_the_parent_and_says_so():
    led = _ledger()
    lines = []
    # 2.6 x 6 ppm at m/z 216 = 3.4 mDa > 2.24 mDa: the gate cannot tell +H from 13C
    out = CL.commit_ion_only_electron_attachment(led, _cfg(cal_sigma=6.0), log=lines.append)
    assert out["ion_only_committed"] == 0 and out["ion_only_skipped_gate"] == 2
    assert _row(led, "H")["role"] == L.ROLE_UNEXPLAINED
    assert lines and "cannot separate" in lines[-1] and "no ion-only row committed" in lines[-1]
    # the uncalibrated window (3 ppm) is narrow enough at this mass
    out = CL.commit_ion_only_electron_attachment(led, _cfg(cal_mu=None, cal_sigma=None), log=lambda *a: None)
    assert out["ion_only_committed"] == 1


def test_a_picker_that_does_not_resolve_the_spacing_skips_every_parent_and_says_so():
    led = _ledger(resolved=False)             # a TOF-like file: no picked pair under ~5.6 mDa
    lines = []
    out = CL.commit_ion_only_electron_attachment(led, _cfg(), log=lines.append)
    assert out["ion_only_committed"] == 0 and out["ion_only_skipped_unresolved"] == 2
    assert _row(led, "H")["role"] == L.ROLE_UNEXPLAINED
    assert lines and "does not resolve" in lines[-1] and "no ion-only row committed" in lines[-1]
    # the resolved pairs must sit near the parent: pairs 200 Da away do not count
    far = [(f"f{k}a", 480.0 + k, 30.0) for k in range(4)] + [(f"f{k}b", 480.004 + k, 25.0) for k in range(4)]
    led = _ledger(resolved=False, extra=far)
    assert CL.commit_ion_only_electron_attachment(led, _cfg(), log=lambda *a: None)["ion_only_committed"] == 0
    centres = CL._resolved_pair_centres(np.array([480.0, 480.004, 481.0, 481.004, 482.0, 482.004]))
    assert CL._resolves_at(centres, 470.0) and not CL._resolves_at(centres, 216.0)


def test_uncalibrated_runs_use_a_three_ppm_window_and_calibrated_ones_the_z_gate():
    led = _ledger(ea_ppm=2.5)
    assert CL.commit_ion_only_electron_attachment(led, _cfg(cal_mu=None, cal_sigma=None),
                                                  log=lambda *a: None)["ion_only_committed"] == 1
    led = _ledger(ea_ppm=4.0)
    assert CL.commit_ion_only_electron_attachment(led, _cfg(cal_mu=None, cal_sigma=None),
                                                  log=lambda *a: None)["ion_only_committed"] == 0
    led = _ledger(ea_ppm=1.0)                 # z = 1.0 / 0.3 = 3.3 > 2.6 on the calibrated gate
    assert CL.commit_ion_only_electron_attachment(led, _cfg(), log=lambda *a: None)["ion_only_committed"] == 0
    led = _ledger(ea_ppm=0.6)                 # z = 2.0
    assert CL.commit_ion_only_electron_attachment(led, _cfg(), log=lambda *a: None)["ion_only_committed"] == 1


def test_occupied_positions_and_excluded_parents_are_left_alone():
    # the +H position holds an M0 already: untouched
    led = _ledger()
    L.commit_assignment(led, "H", neutral_formula="C9H13NO5", adduct="[M-H]-", ion_formula="C9H12NO5-",
                        ion_score=0.8, pass_no=2, method="grid", confidence="Good", commentary="x")
    led.loc[led["peak_id"] == "H", "tier"] = "Candidate"
    out = CL.commit_ion_only_electron_attachment(led, _cfg(), log=lambda *a: None)
    assert out["ion_only_committed"] == 0 and _row(led, "H")["neutral_formula"] == "C9H13NO5"
    # ... or a satellite, a reagent ion, an artifact
    for role_setter in (lambda l: L.attach_isotopologue(l, "H", "B", iso_label="18O"),
                        lambda l: L.mark_reagent(l, "H", "reagent"),
                        lambda l: l.__setitem__("role", l["role"].where(l["peak_id"] != "H", L.ROLE_ARTIFACT))):
        led = _ledger()
        role_setter(led)
        role = _row(led, "H")["role"]
        assert CL.commit_ion_only_electron_attachment(led, _cfg(), log=lambda *a: None)["ion_only_committed"] == 0
        assert _row(led, "H")["role"] == role
    # a parent below assignability is not a parent
    led = _ledger()
    led.loc[led["peak_id"] == "A", "below_assignability"] = True
    out = CL.commit_ion_only_electron_attachment(led, _cfg(), log=lambda *a: None)
    assert out["ion_only_committed"] == 0 and out["ion_only_parents"] == 1
    # a carbon-free parent (nitric acid) is not a parent
    hno3 = C.ion_mz("HNO3", "[M-H]-")
    led = _ledger(extra=[("N", hno3, 5e5), ("NH", C.ion_mz("HNO3", "[M]-."), 100.0)])
    L.commit_assignment(led, "N", neutral_formula="HNO3", adduct="[M-H]-", ion_formula="NO3-",
                        ion_score=0.99, pass_no=0, method="known:atmospheric", confidence="High", commentary="k")
    led.loc[led["peak_id"] == "N", "tier"] = "Assigned"
    out = CL.commit_ion_only_electron_attachment(led, _cfg(), log=lambda *a: None)
    assert out["ion_only_committed"] == 1 and _row(led, "NH")["role"] == L.ROLE_UNEXPLAINED
    # no parent at all, or no channel: nothing
    led = _ledger()
    led.loc[led["role"] == L.ROLE_M0, "adduct"] = "[M+NO3]-"
    assert CL.commit_ion_only_electron_attachment(led, _cfg(), log=lambda *a: None)["ion_only_parents"] == 0
    led = _ledger()
    assert CL.commit_ion_only_electron_attachment(led, _cfg(ion_only_channels=()), log=lambda *a: None) == {
        "ion_only_committed": 0, "ion_only_parents": 0, "ion_only_candidates": 0,
        "ion_only_skipped_gate": 0, "ion_only_skipped_unresolved": 0, "ion_only_channels": []}
    assert _row(led, "H")["role"] == L.ROLE_UNEXPLAINED


# --------------------------------------------------------------------------- the levels
def test_an_ion_only_row_is_never_a_second_channel_nor_a_corroboration_either_way():
    led = _ledger(with_13c=False)             # the parent alone: no axis -> 5a (degeneracy unmeasured)
    parent_alone = EV.compute_levels(led).set_index("peak_id").loc["A", "evidence_level"]
    CL.commit_ion_only_electron_attachment(led, _cfg(), log=lambda *a: None)
    PP.complete_isotope_envelopes(led, _cfg(), log=lambda *a: None)
    lv = EV.compute_levels(led).set_index("peak_id")
    assert lv.loc["A", "evidence_level"] == parent_alone == "5a"        # no chan2 from the ion-only row
    assert "chan2" not in lv.loc["A", "evidence_axes"]
    assert lv.loc["H", "evidence_level"] == "4d"
    # a corroborating source that names X corroborates the ACID, never the ion-only row
    lv = EV.compute_levels(led, cross={X}).set_index("peak_id")
    assert "corroborated" in lv.loc["A", "evidence_axes"] and lv.loc["A", "evidence_level"] == "4b"
    assert "corroborated" not in lv.loc["H", "evidence_axes"] and lv.loc["H", "evidence_level"] == "4d"
    # and the ion-only row never contributes its neutral to a cross set
    only_ion_only = led[led["peak_id"].isin(["H", "H13", "B"])]
    assert EV.corroborating_neutrals([only_ion_only]) == {"C8H12O4"}
    assert EV.corroborating_neutrals([led]) == {X, "C8H12O4"}
    # the pair table records the flag
    pairs = EV.level_pooled({"f": EV.trim(led)})
    assert pairs.set_index(["neutral_formula", "adduct"]).loc[(X, "[M]-."), "ion_only"] == True  # noqa: E712
    assert pairs.set_index(["neutral_formula", "adduct"]).loc[(X, "[M-H]-"), "ion_only"] == False  # noqa: E712
    # a merged ledger (no method column) is recognised by its link
    merged = pd.DataFrame({"neutral_formula": [X, X], "adduct": ["[M-H]-", "[M]-."], "ion_only_of": [pd.NA, "A"]})
    assert EV.is_ion_only(merged).tolist() == [False, True]
    assert EV.corroborating_neutrals([merged.assign(role="M0")]) == {X}


def test_the_reference_script_levels_ion_only_rows_exactly_as_the_core_does():
    led = _ledger()
    CL.commit_ion_only_electron_attachment(led, _cfg(), log=lambda *a: None)
    PP.complete_isotope_envelopes(led, _cfg(), log=lambda *a: None)
    core = EV.compute_levels(led, cross={X}).merge(led[["peak_id", "neutral_formula", "adduct"]], on="peak_id")
    frame = led.assign(__file="f")
    script = LL.assign_levels(LL.measure_source("f", frame, None), {X})
    got = {(r.neutral, r.adduct): r.level for r in script.itertuples()}
    for r in core.itertuples():
        assert got[(r.neutral_formula, r.adduct)] == r.evidence_level, (r.neutral_formula, r.adduct)
    assert got[(X, "[M]-.")] == "4d" and got[(X, "[M-H]-")] == "4a"      # iso + corroborated
    # and the script's cross sets skip ion-only rows too
    frame2 = led[led["peak_id"].isin(["H", "H13", "B"])].assign(__file="g")
    assert set(LL.measure_source("g", frame2, None).query("~ion_only")["neutral"]) == {"C8H12O4"}


# --------------------------------------------------------------------------- batch + publish + board
def _batch_table(spec, height=500.0):
    t0 = pd.Timestamp("2025-10-01 21:00:00", tz="UTC")
    rows = []
    for i, (sid, mzs) in enumerate(spec.items()):
        t = t0 + pd.Timedelta(minutes=10 * i)
        rows += [dict(sample_item_id=sid, sample_item_name=f"n_{sid}", datetime_utc=t,
                      mz=float(mz), height=height) for mz in mzs]
    return pd.DataFrame(rows)


def test_batch_merged_row_carries_the_link_and_the_summary_counts_the_bucket(tmp_path, monkeypatch):
    from peaky.batch import assign_batch as AB

    bg = list(range(100, 120))
    spec = {f"a{i}": bg + list(range(200 + 20 * i, 220 + 20 * i)) for i in range(4)}
    pk = _batch_table(spec)
    seen = []

    def fake_assign(sid, context="ambient-air", **kw):
        seen.append(kw)
        led = _ledger()
        CL.commit_ion_only_electron_attachment(led, _cfg(), log=lambda *a: None)
        PP.complete_isotope_envelopes(led, _cfg(), log=lambda *a: None)
        EV.apply_levels(led)
        return {"ledger": led, "stats": {"noise_edge_cps": 4.0, "height_gate_cps": 10.0},
                "plausibility_audit": [], "summaries": {}, "problems": []}

    monkeypatch.setattr(IO, "connect", lambda *a, **k: "CLIENT")
    monkeypatch.setattr(IO, "fetch_peaks", lambda client, sid, use_cache=True: pd.DataFrame(
        {"peak_id": ["A"], "mz": [MZ_PARENT], "height": [1.0e5]}))
    monkeypatch.setattr(IO, "estimate_offset", lambda raw: 0.0)
    monkeypatch.setattr(A, "run", fake_assign)
    AB.run(peaks=pk, ts_peaks=pk, reagent="NO3", batch="test batch", out_dir=str(tmp_path),
           k_min=2, k_max=3, min_gain=0.0, n_jobs=1, log=lambda *a: None)
    # the profile's channel reached the per-file cfg
    assert seen and all(kw["cfg"].ion_only_channels == ("[M]-.",) for kw in seen)
    merged = pd.read_csv(tmp_path / "merged_ledger.csv")
    row = merged[(merged.neutral_formula == X) & (merged.adduct == "[M]-.")]
    assert len(row) == 1 and row.ion_only_of.iloc[0] == "A" and row.tier.iloc[0] == "Candidate"
    assert row.evidence_level.iloc[0] == "4d"
    acid = merged[(merged.neutral_formula == X) & (merged.adduct == "[M-H]-")]
    assert len(acid) == 1 and pd.isna(acid.ion_only_of.iloc[0]) and acid.evidence_level.iloc[0] == "4b"
    summ = json.load(open(tmp_path / "batch_summary.json"))
    io = summ["ion_only"]
    assert io["channels"] == ["[M]-."] and io["merged"] == 1 and io["n_files_with"] == len(seen)
    assert io["per_file_rows"] == len(seen) and io["merged_levels"] == {"4d": 1}
    # a Br run opens no channel and reports an empty bucket
    seen.clear()
    AB.run(peaks=pk, ts_peaks=pk, reagent="Br", batch="test batch", out_dir=str(tmp_path / "br"),
           k_min=2, k_max=3, min_gain=0.0, n_jobs=1, log=lambda *a: None)
    assert all(kw["cfg"].ion_only_channels == () for kw in seen)
    assert json.load(open(tmp_path / "br" / "batch_summary.json"))["ion_only"]["channels"] == []


def test_publish_sends_a_null_mechanism_for_the_ion_only_adduct():
    assert "[M]-." not in IO.ADDUCT_TO_MECH
    led = _ledger()
    CL.commit_ion_only_electron_attachment(led, _cfg(), log=lambda *a: None)
    rows, summary = PUB.build_rows(led, intensity_column="height", bands=PUB.DEFAULT_TIER_BANDS,
                                   mechanism_ids={"[M-H]-": "mech-1"})
    by = {r["sample_peak_id"]: r for r in rows}
    assert by["A"]["ionization_mechanism_id"] == "mech-1"
    assert by["H"]["ionization_mechanism_id"] is None and by["H"]["assigned_formula"] == X
    assert summary["resolved_mechanisms"] == 3       # A, its 13C child (inherits) and B; the ion-only row carries none


def test_scorecard_counts_the_bucket_on_its_own_and_keeps_it_out_of_the_candidate_tile():
    import scorecard as SC

    led = _ledger()
    CL.commit_ion_only_electron_attachment(led, _cfg(), log=lambda *a: None)
    EV.apply_levels(led)
    led.loc[led["peak_id"] == "B", "tier"] = "Candidate"
    run = types.SimpleNamespace(ledger=led, ts=None, name="r", channel="c|x", reagent="NO3", path_kind="cover",
                                code="abc", summary={"n_files": 1}, n_spectra=0, per_file=pd.DataFrame())
    h = SC.headline(run, pd.DataFrame(columns=["ion_mz"]))
    assert h["assigned"] == 1 and h["candidate"] == 1 and h["ion_only"] == 1
    assert h["ion_only_levels"] == {"5a": 1}          # no sweep ran here: exact mass only
    assert ("ion_only", "ion-only rows", 0) in SC.KEY_METRICS
    counts = SC._ledger_counts(led, "f")
    assert counts["candidate"] == 1 and counts["assigned"] == 1


# --------------------------------------------------------------------------- the merge vote
def test_an_ion_only_reading_never_outvotes_a_regular_reading_at_the_merge():
    """The radical anion of a C_n acid sits 0.44 mDa from the labelled-nitrate
    cluster of the C_{n-1} organonitrate: files that left the peak unexplained
    and read it ion-only must not outvote the files that read the cluster."""
    from peaky.batch import assign_batch as AB

    mz = 294.0958
    def m0(nf, ad, tier, link):
        return pd.DataFrame([{"mz": mz, "neutral_formula": nf, "adduct": ad, "tier": tier,
                              "ion_score": 0.0 if pd.notna(link) else 0.9, "admitted_by": "height",
                              "occurrence": 0.9, "ion_only_of": link}])
    per_file = {f"io{k}": m0("C11H18O9", "[M]-.", "Candidate", "P") for k in range(8)}
    per_file.update({f"reg{k}": m0("C10H17NO5", "[M+^NO3]-", "Candidate", pd.NA) for k in range(4)})
    merged, _ = AB.align(per_file, tol_ppm=6.0)
    assert len(merged) == 1
    r = merged.iloc[0]
    assert (r.neutral_formula, r.adduct) == ("C10H17NO5", "[M+^NO3]-")   # 4 files beat 8 ion-only files
    assert pd.isna(r.ion_only_of) and r.n_files == 12 and r.n_files_ion == 4
    assert "C11H18O9 [M]-. x8" in r.alternatives
    assert "ion-only reading" in str(r.tier_reason) and "vote 4 of 12" in str(r.tier_reason)
    # alone, the ion-only reading wins its cluster as any reading would
    merged, _ = AB.align({k: v for k, v in per_file.items() if k.startswith("io")}, tol_ppm=6.0)
    assert (merged.iloc[0].neutral_formula, merged.iloc[0].adduct) == ("C11H18O9", "[M]-.")
    assert merged.iloc[0].ion_only_of == "P" and pd.isna(merged.iloc[0].tier_reason)
    # and a regular reading in ONE file beats it, with the note
    per_file2 = {k: v for k, v in per_file.items() if k.startswith("io")}
    per_file2["reg0"] = per_file["reg0"]
    r = AB.align(per_file2, tol_ppm=6.0)[0].iloc[0]
    assert r.adduct == "[M+^NO3]-" and "vote 1 of 9" in str(r.tier_reason)


def test_the_channel_log_names_the_profile_when_an_earlier_call_already_applied_it():
    cfg = PCfg.PassConfig()
    P.apply_ion_only_channels(cfg, P.NO3)                    # pipeline: silent
    lines = []
    P.apply_ion_only_channels(cfg, P.NO3, log=lines.append)  # assign_batch: logs
    assert lines and "from profile NO3" in lines[0]
    lines = []
    P.apply_ion_only_channels(PCfg.PassConfig(ion_only_channels=("[M+O2]-",)), P.NO3, log=lines.append)
    assert lines and "explicit config" in lines[0]


def test_the_final_sweep_never_displaces_an_existing_row_on_behalf_of_an_ion_only_parent():
    """The 13C of the new [M]-. row sits 0.3 mDa from where a weak cluster
    reading of another neutral can be; the sweep attaches only UNEXPLAINED
    peaks to an ion-only parent."""
    led = _ledger()
    CL.commit_ion_only_electron_attachment(led, _cfg(), log=lambda *a: None)
    # the +H row's 13C position holds a weak Candidate M0 of another neutral
    L.commit_assignment(led, "H13", neutral_formula="C2H7NO3", adduct="[M+NO3]-", ion_formula="C2H7N2O6-",
                        ion_score=0.55, pass_no=2, method="grid", confidence="Good", commentary="weak")
    led.loc[led["peak_id"] == "H13", "tier"] = "Candidate"
    PP.complete_isotope_envelopes(led, _cfg(), log=lambda *a: None)
    assert _row(led, "H13")["role"] == L.ROLE_M0 and _row(led, "H13")["neutral_formula"] == "C2H7NO3"
    EV.apply_levels(led)
    assert led.set_index("peak_id").loc["H", "evidence_level"] == "5a"      # no satellite of its own
    # the same weak row under a REGULAR parent is still displaced (the sweep's own rule):
    # re-commit the acid's 13C peak as a weak M0 of another neutral (A is not locked)
    led2 = _ledger()
    L.commit_assignment(led2, "A13", neutral_formula="C2H7NO3", adduct="[M+NO3]-", ion_formula="C2H7N2O6-",
                        ion_score=0.55, pass_no=2, method="grid", confidence="Good", commentary="weak")
    assert _row(led2, "A13")["role"] == L.ROLE_M0
    PP.complete_isotope_envelopes(led2, _cfg(), log=lambda *a: None)
    assert _row(led2, "A13")["role"] == L.ROLE_ISO and _row(led2, "A13")["parent_peak_id"] == "A"


def test_scorecard_brightest_flags_ion_only_rows_and_leaves_them_out_of_not_assigned():
    import scorecard as SC

    led = _ledger()
    CL.commit_ion_only_electron_attachment(led, _cfg(), log=lambda *a: None)
    led.loc[led["peak_id"] == "B", "tier"] = "Candidate"
    ions = pd.DataFrame({
        "ion_mz": [MZ_PARENT, MZ_EA, C.ion_mz("C8H12O4", "[M-H]-")], "n": [10, 10, 10],
        "med_h": [40_000.0, 5_200.0, 9_000.0], "sum_h": [1.0, 1.0, 1.0], "role": ["M0", "M0", "M0"],
        "ion_formula": ["C10H15O5-", "C10H16O5-", "C8H11O4-"], "iso_label": [None, None, None],
        "neutral": [X, X, "C8H12O4"], "adduct": ["[M-H]-", "[M]-.", "[M-H]-"],
        "tier": ["Assigned", "Candidate", "Candidate"], "suspect": [False, False, False], "stamp_source": ["", "", ""],
        "presence": [1.0, 1.0, 1.0]})
    run = types.SimpleNamespace(ledger=led, ts=None, name="r", channel="c|x", reagent="NO3", path_kind="cover",
                                code="abc", summary={"n_files": 1}, n_spectra=10, per_file=pd.DataFrame())
    b = SC.brightest(run, ions, pd.DataFrame(), pd.DataFrame(columns=["neutral", "adduct", "level", "n_axes", "axes"]))
    by = {(r["neutral"], r["adduct"]): r for r in b["rows"]}
    assert by[(X, "[M]-.")]["ion_only"] and not by[(X, "[M-H]-")]["ion_only"] and not by[("C8H12O4", "[M-H]-")]["ion_only"]
    assert b["ion_only_in_top"] == 1 and b["m0_not_assigned"] == 1     # the regular Candidate, not the bucket
