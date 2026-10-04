"""The evidence stage's WIRING (the scale of peaky 0.10.0) -- docs/EVIDENCE_LEVELS.md.

The decision is pinned by tests/test_levels_*.py; this file pins where the
stage runs and what each entry point writes:

- the per-file stage (`evidence.apply_levels`, assign's `evidence` stage): the
  file levelled ALONE in "adapted" mode, the columns on every committed M0 row
  and on nothing else, NA on a TOF-class or class-less file before any fact
  work, the run's reagent profile / window / gate threaded from the run;
- the degeneracy stage's calibration persisted per file (`degeneracy_cal`);
- the pooled stage (`assign_batch.run`): the per-file ledgers re-read from disk
  in sorted order and levelled as one source, the merged ledger stamped by ion,
  tables/evidence_levels.csv (the new columns + the facts, no pre-0.10.0 level
  columns), batch_summary's evidence_levels / claims / reflists_context;
- `--corroborate`: the merge vote's cross set (`vote_cross_neutrals`) and the
  other-source partners of an Orbitrap-class run dir; never a level.

The report / PDF / publish surfaces of the scale are pinned by their own tests.

Run: pytest tests/test_evidence_outputs.py -q
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pandas as pd
import pytest

from peaky import cli
from peaky.assignment import assign as A
from peaky.assignment import evidence as EV
from peaky.assignment import ledger as L
from peaky.assignment import tiers as T
from peaky.assignment.levels import competitors as CP
from peaky.assignment.levels import decide as DC
from peaky.assignment.levels import source as SRC
from peaky.chem import chemistry as C
from peaky.chem import contexts as X
from peaky.chem.resolution import Resolution

ROOT = Path(__file__).resolve().parents[1]
ORBI = 120_000.0      # a constant resolving power: R(200) = 120 000 -> Orbitrap class
TOF = 9_000.0         # R(200) = 9 000 -> TOF class
CAL = (0.0, 0.5)      # the degeneracy stage's (mu, sigma) ppm
OLD_COLUMNS = ("evidence_axes", "level_reason", "n_plausible_structures")


def _ledger() -> pd.DataFrame:
    """A finished single-sample ledger in the real schema at exact masses: one
    M0 with its 13C child (A, B), a second M0 (C) and a reagent ion (E)."""
    mz_a = C.ion_mz("C10H16O4", "[M-H]-")
    peaks = pd.DataFrame({"peak_id": ["A", "B", "C", "E"],
                          "mz": [mz_a, mz_a + 1.0033548, C.ion_mz("C7H12O4", "[M-H]-"), 78.9189],
                          "height": [1.0e5, 1.07e4, 8.0e4, 1.0e4]})
    led = L.new_ledger(peaks)
    L.commit_assignment(led, "A", neutral_formula="C10H16O4", adduct="[M-H]-",
                        ion_formula="C10H15O4-", ion_score=0.97, compound_score=0.96,
                        ppm_error=-0.3, pass_no=1, method="cheminfo", confidence="High",
                        commentary="Pass 1: C10H16O4 [M-H]-",
                        isotopologues=[{"label": "13C", "score": 0.93, "peak_id": "B"}])
    L.attach_isotopologue(led, "B", "A", iso_label="13C", iso_match_score=0.93)
    L.commit_assignment(led, "C", neutral_formula="C7H12O4", adduct="[M-H]-",
                        ion_formula="C7H11O4-", ion_score=0.83, compound_score=0.83,
                        ppm_error=0.5, pass_no=2, method="gka-series", confidence="Good (series)",
                        commentary="Pass 2 series from C8H14O4 -CH2")
    L.mark_reagent(led, "E", "reagent ion: [Br]-")
    T.apply_tiers(led)
    return led


def _inputs(resolution=ORBI, cal=CAL, reagent="NO3"):
    return EV.file_run_inputs(sample_id="s1", reagent=reagent, context="ambient-air", resolution=resolution,
                              height_gate_cps=100.0, degeneracy_cal=cal)


# --------------------------------------------------------------------------- the per-file stage
def test_the_stage_sits_after_every_tier_stage_and_before_timeseries():
    names = [s.name for s in A._STAGES]
    i = names.index("evidence")
    for earlier in ("degeneracy", "tiers", "plausibility", "reflist_rescue", "iso_env_final"):
        assert names.index(earlier) < i, (earlier, names)
    assert names.index("timeseries") == i + 1
    stage = A._STAGES[i]
    assert stage.safe is False and stage.store is True


def test_the_per_file_stage_writes_the_columns_on_m0_rows_only():
    led = _ledger()
    s = EV.apply_levels(led, run_inputs=_inputs())
    by = led.set_index("peak_id")
    for c in EV.COLUMNS:
        assert c in led.columns, c
    for pid in ("A", "C"):
        lv = by.loc[pid, "evidence_level"]
        assert lv in EV.LEVELS + EV.BUCKETS, (pid, lv)
        assert str(by.loc[pid, "evidence"]).startswith(lv), by.loc[pid, "evidence"]
        assert by.loc[pid, "claim"] == EV.claim_class(lv)
        assert " · context source: " in by.loc[pid, "evidence"]
    for pid in ("B", "E"):              # an isotope child and a reagent ion carry nothing
        assert all(pd.isna(by.loc[pid, c]) for c in EV.COLUMNS), pid
    assert not set(OLD_COLUMNS) & set(led.columns)
    assert set(s) == {"levels", "claims", "n_levelled", "n_pairs", "instrument"}
    assert list(s["claims"]) == list(EV.CLAIM_KEYS) and sum(s["claims"].values()) == 2
    assert s["n_levelled"] == 2 and s["n_pairs"] == 2
    assert s["instrument"] == {"class": "orbitrap", "r200": ORBI}


@pytest.mark.parametrize("resolution, tail", [
    (TOF, "(width model R(200) = 9 000 < 50 000)"),
    (None, "(no width model: the instrument class is unknown)"),
])
def test_a_tof_or_class_less_file_reads_na_before_any_fact_work(monkeypatch, resolution, tail):
    monkeypatch.setattr(SRC, "pair_facts", lambda src: pytest.fail("fact work on a source the scale does not assess"))
    led = _ledger()
    s = EV.apply_levels(led, run_inputs=_inputs(resolution=resolution))
    m0 = led[led["role"] == "M0"]
    assert (m0["evidence_level"] == "NA").all() and (m0["claim"] == "not assessed").all()
    assert m0["evidence"].map(lambda t: t.startswith("NA · not assessed on this instrument class") and t.endswith(tail)).all()
    for c in ("would_lift", "competitors_left", "tags", "context", "context_source"):
        assert (m0[c].fillna("") == "").all(), c
    off = led[led["role"] != "M0"]
    assert off[list(EV.COLUMNS)].isna().all().all()
    assert s["levels"] == {"NA": 2} and s["claims"]["not assessed"] == 2


def test_the_stages_resolution_argument_sets_the_class():
    led = _ledger()
    EV.apply_levels(led, run_inputs=_inputs(resolution=None), resolution=TOF)
    assert (led.loc[led["role"] == "M0", "evidence_level"] == "NA").all()
    led = _ledger()
    EV.apply_levels(led, run_inputs=_inputs(resolution=None), resolution=ORBI)
    assert "NA" not in set(led.loc[led["role"] == "M0", "evidence_level"])


def test_an_uncalibrated_file_alone_has_no_window_and_says_so():
    led = _ledger()
    s = EV.apply_levels(led, run_inputs=_inputs(cal=None))
    m0 = led[led["role"] == "M0"]
    assert m0["evidence_level"].isna().all() and (m0["claim"] == "tentative").all()
    assert (m0["evidence"] == EV.NO_WINDOW_TEXT).all()
    assert s["levels"] == {} and s["n_levelled"] == 0


def test_an_uncalibrated_file_without_the_calibrated_ppm_column_says_so_instead_of_crashing():
    """A decoy shift arm that Assigned nothing: tiers.stamp_calibrated_ppm wrote no
    `ppm_error_cal` and the degeneracy stage had no calibration. The stage is
    safe=False (a crash kills the run): the missing column reads as no stamp
    centre, so the file has no window and says so -- persisted or refitted."""
    for cal in (None, "absent"):
        led = _ledger().drop(columns=["ppm_error_cal"], errors="ignore")
        s = EV.apply_levels(led, run_inputs=_inputs(cal=cal))
        m0 = led[led["role"] == "M0"]
        assert (m0["evidence"] == EV.NO_WINDOW_TEXT).all() and (m0["claim"] == "tentative").all(), cal
        assert s["n_levelled"] == 0


def _blank_ledger() -> pd.DataFrame:
    """A file the engine committed nothing on (an empty decoy arm, a blank)."""
    return L.new_ledger(pd.DataFrame({"peak_id": ["A", "B"], "mz": [199.1, 250.2], "height": [1.0e5, 2.0e4]}))


def test_a_source_with_no_committed_pair_levels_to_an_empty_frame_with_the_columns():
    """Zero committed pairs: no row, the columns a levelled source writes (the
    scale's columns first, the step facts, pass B's facts) -- through
    level_source, the per-file stage and the pooled stage alike."""
    full = EV.level_source(EV.source_from_frames({"s1": _ledger()}, run_inputs=_inputs(), mode="adapted"))
    empty = EV.level_source(EV.source_from_frames({"s1": _blank_ledger()}, run_inputs=_inputs(), mode="adapted"))
    assert empty.empty and list(empty.columns[:10]) == ["neutral_formula", "adduct", *EV.COLUMNS]
    assert set(empty.columns) <= set(full.columns)
    head = len(DC.RECORD_COLUMNS) + 3              # the pair, the record's columns, claim
    assert list(empty.columns[:head]) == list(full.columns[:head])
    assert set(CP.PASS_B_COLUMNS) <= set(empty.columns)
    # the per-file stage
    led = _blank_ledger()
    s = EV.apply_levels(led, run_inputs=_inputs())
    assert s["n_pairs"] == 0 and s["levels"] == {} and set(EV.COLUMNS) <= set(led.columns)
    # the pooled stage's entry point, and the stamp of an empty merged ledger
    pooled = EV.level_batch({"s1": _blank_ledger(), "s2": _blank_ledger()},
                            run_inputs=EV.RunInputs(summary=_inputs().summary))
    assert pooled.empty and list(pooled.columns) == list(empty.columns)
    assert len(EV.stamp_merged(pd.DataFrame(columns=["neutral_formula", "adduct", "mz"]), pooled)) == 0


def test_a_batch_that_committed_nothing_writes_an_empty_level_table(tmp_path, monkeypatch):
    monkeypatch.setattr(sys.modules[__name__], "_ledger", _blank_ledger)
    _run_batch(tmp_path, monkeypatch)
    pairs = pd.read_csv(tmp_path / "tables" / "evidence_levels.csv", keep_default_na=False)
    assert pairs.empty and list(pairs.columns[:10]) == ["neutral_formula", "adduct", *EV.COLUMNS]
    ev = json.load(open(tmp_path / "batch_summary.json"))["evidence_levels"]
    assert ev["n_pairs"] == 0 and ev["pooled"] == {}


def test_an_unknown_reagent_profile_is_not_levelled_and_says_so():
    led = _ledger()
    EV.apply_levels(led, run_inputs=_inputs(reagent=None))
    m0 = led[led["role"] == "M0"]
    assert m0["evidence_level"].isna().all() and (m0["evidence"] == EV.NO_REAGENT_TEXT).all()


def test_the_stage_function_threads_the_run_and_logs_the_claims():
    led = _ledger()
    lines = []
    st = types.SimpleNamespace(
        led=led, cfg=types.SimpleNamespace(height_cutoff=100.0, noise_edge_cps=4.0), corroborate=set(),
        resolving_power=Resolution.coerce(ORBI), sample_id="s1", reagent_profile="NO3", adducts=["[M-H]-"],
        profile=X.get_context("ambient-air"), reflists_active=None, degeneracy_cal=CAL, reflists_context=None,
        log=lines.append)
    s = A._stage_evidence(st)
    line = next(ln for ln in lines if "evidence levels" in ln)
    c = s["claims"]
    assert "levelled alone, instrument class orbitrap" in line
    assert line.endswith("claims " + " | ".join(f"{k} {c[k]}" for k in EV.CLAIM_KEYS))
    assert "NA" not in set(led.loc[led["role"] == "M0", "evidence_level"])
    # no width model on the run: NA, and the log says the class is unknown
    led2, lines2 = _ledger(), []
    A._stage_evidence(types.SimpleNamespace(**{**vars(st), "led": led2, "resolving_power": None,
                                               "log": lines2.append}))
    assert (led2.loc[led2["role"] == "M0", "evidence_level"] == "NA").all()
    assert any("instrument class unknown" in ln for ln in lines2)


def test_the_reagent_profile_is_found_from_the_runs_adducts():
    from peaky.chem import profiles as PR
    p = PR.resolve("Br")
    assert A._profile_name_for(p.adducts) in {q.name for q in PR._BY_ALIAS.values() if set(q.adducts) == set(p.adducts)}
    assert A._profile_name_for(["[M+Xx]-"]) is None and A._profile_name_for([]) is None


def test_the_degeneracy_calibration_is_persisted_in_the_stats(monkeypatch):
    """assign.run writes the degeneracy stage's own (mu, sigma) into the run's
    stats as `degeneracy_cal` (null when uncalibrated; no key when the stage
    did not run) -- the per-file and the pooled level's step-1 window."""
    from peaky.io import io_mascope as IO

    peaks = pd.DataFrame({"peak_id": ["p1", "p2"], "mz": [199.0976, 255.0], "height": [1e4, 2e3]})

    def run_with(stage):
        monkeypatch.setattr(A, "_STAGES", [stage] if stage is not None else [])
        try:
            return A.run("deg-cal-test", adducts=["[M-H]-"], peaks=peaks, log=lambda *a: None)
        finally:
            IO.unregister_offline_sample("deg-cal-test")

    def setter(cal):
        return A._Stage("degeneracy", lambda st: (setattr(st, "degeneracy_cal", cal), {})[1])

    assert run_with(setter((0.12, 0.45)))["stats"]["degeneracy_cal"] == {"mu": 0.12, "sigma": 0.45}
    assert run_with(setter(None))["stats"]["degeneracy_cal"] is None
    assert "degeneracy_cal" not in run_with(None)["stats"]


# --------------------------------------------------------------------------- the vote class
def test_vote_classes_are_the_private_decisions_class_per_m0_row():
    led = _ledger()
    vc = EV.vote_classes(EV.trim(led))
    m0 = led.index[led["role"] == "M0"]
    assert list(vc.index) == list(m0) and set(vc.unique()) <= {0, 1, 2}
    assert vc.loc[led.index[led["peak_id"] == "A"][0]] == 1          # its own 13C line: the formula confirmed
    # the --corroborate source holds the neutral: neutral backed
    vc2 = EV.vote_classes(EV.trim(led), cross={"C10H16O4"})
    assert vc2.loc[led.index[led["peak_id"] == "A"][0]] == 2
    # the full ledger and its trim give the same classes
    assert EV.vote_classes(led).equals(vc)
    assert EV.VOTE_CLASS_TEXT == {2: "neutral backed", 1: "formula confirmed", 0: "unconfirmed"}


# --------------------------------------------------------------------------- the pooled stage
def _batch_table(spec, height=500.0):
    t0 = pd.Timestamp("2025-10-01 21:00:00", tz="UTC")
    rows = []
    for i, (sid, mzs) in enumerate(spec.items()):
        t = t0 + pd.Timedelta(minutes=10 * i)
        rows += [dict(sample_item_id=sid, sample_item_name=f"n_{sid}", datetime_utc=t,
                      mz=float(mz), height=height) for mz in mzs]
    return pd.DataFrame(rows)


def _run_batch(tmp_path, monkeypatch, *, resolving_power=ORBI, corroborate=None, seen=None, lines=None):
    from peaky.batch import assign_batch as AB
    from peaky.io import io_mascope as IO

    bg = list(range(100, 120))
    spec = {}
    for i in range(4):
        spec[f"a{i}"] = bg + list(range(200 + 20 * i, 220 + 20 * i))
        spec[f"b{i}"] = bg + list(range(200 + 20 * i, 220 + 20 * i))
    pk = _batch_table(spec)

    def fake_assign(sid, context="ambient-air", **kw):
        if seen is not None:
            seen.append(kw)
        return {"ledger": _ledger(), "stats": {"noise_edge_cps": 4.0, "height_gate_cps": 10.0,
                                               "degeneracy_cal": {"mu": CAL[0], "sigma": CAL[1]}},
                "plausibility_audit": [], "summaries": {}, "problems": []}

    monkeypatch.setattr(IO, "connect", lambda *a, **k: "CLIENT")
    monkeypatch.setattr(IO, "fetch_peaks", lambda client, sid, use_cache=True: pd.DataFrame(
        {"peak_id": ["A"], "mz": [200.1], "height": [1.0e5]}))
    monkeypatch.setattr(IO, "estimate_offset", lambda raw: 0.0)
    monkeypatch.setattr(A, "run", fake_assign)
    return AB.run(peaks=pk, ts_peaks=pk, reagent="Br", batch="test batch", out_dir=str(tmp_path),
                  k_min=2, k_max=3, min_gain=0.0, n_jobs=1, corroborate=corroborate,
                  resolving_power=resolving_power, log=(lines.append if lines is not None else (lambda *a: None)))


def test_the_pooled_stage_stamps_merged_and_writes_the_table_and_the_summaries(tmp_path, monkeypatch):
    seen = []
    res = _run_batch(tmp_path, monkeypatch, seen=seen)
    # the per-file runs get the profile and the activation record, never the cross set
    assert seen and all(kw.get("reagent_profile") == "Br" and "corroborate" not in kw for kw in seen)
    assert all(set(kw["reflists_context"]) == {"tags", "matched"} for kw in seen)
    merged = pd.read_csv(tmp_path / "merged_ledger.csv", keep_default_na=False)
    for c in EV.COLUMNS:
        assert c in merged.columns, c
    assert not set(OLD_COLUMNS) & set(merged.columns) and "vote_class" not in merged.columns
    row = merged[(merged.neutral_formula == "C10H16O4") & (merged.adduct == "[M-H]-")]
    assert len(row) == 1 and row.evidence_level.iloc[0] in EV.LEVELS + EV.BUCKETS
    assert (merged["claim"] == merged["evidence_level"].map(EV.claim_class)).all()
    pairs = pd.read_csv(tmp_path / "tables" / "evidence_levels.csv", keep_default_na=False)
    assert list(pairs.columns[:10]) == ["neutral_formula", "adduct", *EV.COLUMNS]
    assert not set(OLD_COLUMNS) & set(pairs.columns)
    assert {"iso_veto", "label_veto", "lowconf", "below", "ion_only", "n_competitors", "split_pinned",
            "tag_kinds", "anchor_kind", "anchor_why"} <= set(pairs.columns)
    assert not set(EV.INTERNAL_COLUMNS) & set(pairs.columns)       # the in-pass tokens stay in memory (D21)
    assert set(pairs["anchor_kind"]) <= {"two routes", "ladder", "listed", "none"}
    assert len(pairs) == 2 and res["evidence"] is not None and len(res["evidence"]) == 2
    summ = json.load(open(tmp_path / "batch_summary.json"))
    ev = summ["evidence_levels"]
    assert ev["scale"] == f"peaky {EV.SCALE_RELEASE}" == "peaky 0.10.0"
    assert ev["instrument"] == {"class": "orbitrap", "r200": ORBI}
    assert sum(ev["pooled"].values()) == ev["n_pairs"] == 2 and sum(ev["merged"].values()) == len(merged)
    assert set(ev["pooled"]) <= set(EV.LEVELS + EV.BUCKETS)
    assert ev["n_unstamped"] == 0 and ev["side_channels_locked"] is True and ev["unlocked"] == []
    assert ev["partners"] == {} and ev["amine_r_min"] == 0.6 and summ["amine_r_min"] == 0.6
    assert ev["n_corroborate"] == 0 and ev["cross_source"] == []
    keys = list(summ)
    assert keys.index("claims") == keys.index("evidence_levels") + 1
    assert list(summ["claims"]["merged"]) == list(EV.CLAIM_KEYS)
    rc = summ["reflists_context"]
    assert set(rc) == {"tags", "matched", "active"}
    assert all(len(x) == 3 and x[2] for x in rc["active"])
    assert {"always active"} <= {x[2] for x in rc["active"]}
    assert all(p["degeneracy_cal"] == {"mu": CAL[0], "sigma": CAL[1]} for p in summ["per_file"])
    assert set(summ["ion_only"]["merged_levels"]) <= set(EV.LEVELS + EV.BUCKETS)
    # jitter.csv carries the vote class, never a level
    jit = pd.read_csv(tmp_path / "tables" / "jitter.csv")
    assert "vote_class" in jit.columns and "evidence_level" not in jit.columns
    assert set(jit["vote_class"].dropna().astype(int)) <= {0, 1, 2}


@pytest.mark.parametrize("resolving_power", [TOF, None])
def test_the_pooled_stage_on_a_tof_batch_is_not_assessed(tmp_path, monkeypatch, resolving_power):
    """A TOF-class or class-less batch: every pair NA, and tables/evidence_levels.csv
    still carries the pooled pair facts (D17) -- never the scale's own facts
    (no enumeration, no gate, no pass A on such a run)."""
    from peaky.assignment.levels import competitors as _CP
    monkeypatch.setattr(_CP, "q1_pass", lambda *a, **k: pytest.fail("pass A on a run the scale does not assess"))
    _run_batch(tmp_path, monkeypatch, resolving_power=resolving_power)
    merged = pd.read_csv(tmp_path / "merged_ledger.csv", keep_default_na=False)
    assert (merged["evidence_level"] == "NA").all() and (merged["claim"] == "not assessed").all()
    pairs = pd.read_csv(tmp_path / "tables" / "evidence_levels.csv", keep_default_na=False)
    assert list(pairs.columns[:10]) == ["neutral_formula", "adduct", *EV.COLUMNS] and len(pairs) == 2
    assert {"iso_veto", "label_veto", "lowconf", "below", "ion_only", "tied", "lead", "upair", "iso", "chan2",
            "n_files"} <= set(pairs.columns)
    assert not {"n_competitors", "split_pinned", "tag_kinds"} & set(pairs.columns)
    assert not set(OLD_COLUMNS) & set(pairs.columns)
    if resolving_power is None:
        return
    summ = json.load(open(tmp_path / "batch_summary.json"))
    assert summ["evidence_levels"]["instrument"]["class"] == "tof"
    assert summ["evidence_levels"]["pooled"] == {"NA": 2}
    assert summ["claims"]["merged"]["not assessed"] == len(merged)


def test_stamp_merged_joins_by_ion_and_a_re_read_gets_no_level():
    pairs = pd.DataFrame({"neutral_formula": ["C10H16O4", "C7H12O4"], "adduct": ["[M-H]-", "[M-H]-"],
                          "evidence_level": ["4b", "NA"], "evidence": ["4b · x", "NA · y"], "would_lift": ["w", ""],
                          "competitors_left": ["", ""], "tags": ["", ""], "context": ["", ""],
                          "context_source": ["c", ""], "claim": ["ion", "not assessed"]})
    merged = pd.DataFrame([
        dict(mz=199.09, neutral_formula="C10H16O4", adduct="[M-H]-", tier="Assigned"),
        dict(mz=159.07, neutral_formula="C7H12O4", adduct="[M-H]-", tier="Assigned"),
        dict(mz=300.0, neutral_formula="C9H99O9", adduct="[M-H]-", tier="Candidate"),   # a re-read no file holds
    ])
    out = EV.stamp_merged(merged, pairs)
    assert out.index.equals(merged.index) and len(out) == 3
    assert list(out.evidence_level[:2]) == ["4b", "NA"] and pd.isna(out.evidence_level.iloc[2])
    assert list(out.claim) == ["ion", "not assessed", "tentative"]
    assert out.evidence.iloc[2] == EV.NO_POOLED_PAIR_TEXT == "no pooled pair: a batch-level re-read"
    assert out.would_lift.iloc[0] == "w" and pd.isna(out.would_lift.iloc[2])
    out0 = EV.stamp_merged(merged, pairs.iloc[0:0])
    assert out0.evidence_level.isna().all() and set(EV.COLUMNS) <= set(out0.columns)
    assert (out0.claim == "tentative").all() and (out0.evidence == EV.NO_POOLED_PAIR_TEXT).all()
    assert len(EV.stamp_merged(merged.iloc[0:0], pairs)) == 0


# --------------------------------------------------------------------------- --corroborate
def test_vote_cross_neutrals_resolves_run_dirs_and_csvs(tmp_path):
    # the ledger holds C10H16O4 by its own 13C line (the private 4b) and C7H12O4
    # on a series commit alone: only the first is a sighting
    run = tmp_path / "run"; (run / "per_file").mkdir(parents=True)
    _ledger().to_csv(run / "per_file" / "s1_ledger.csv", index=False)
    assert EV.vote_cross_neutrals([str(run)]) == {"C10H16O4"}
    assert EV.vote_cross_neutrals([str(tmp_path)]) == {"C10H16O4"}   # the out-dir holding one run
    assert EV.vote_cross_neutrals([str(run / "per_file" / "s1_ledger.csv")]) == {"C10H16O4"}
    # a pre-0.10.0 merged ledger: its stored private level, without its own `corroborated`
    merged = tmp_path / "m.csv"
    pd.DataFrame({"neutral_formula": ["HNO3", "C5H8O3"], "adduct": ["[M-H]-", "[M-H]-"],
                  "evidence_level": ["2b", "4b"], "evidence_axes": ["known:atmospheric|files:3", "corroborated|files:2"]}
                 ).to_csv(merged, index=False)
    assert EV.vote_cross_neutrals([str(merged)]) == {"HNO3"}
    # a merged ledger of the evidence scale carries no vote class: say so, never guess
    new = tmp_path / "new.csv"
    pd.DataFrame({"neutral_formula": ["HNO3"], "adduct": ["[M-H]-"], "evidence_level": ["4b"],
                  "claim": ["ion"]}).to_csv(new, index=False)
    with pytest.raises(ValueError, match="evidence scale"):
        EV.vote_cross_neutrals([str(new)])
    bare = tmp_path / "bare.csv"
    pd.DataFrame({"neutral_formula": ["HNO3"], "adduct": ["[M-H]-"]}).to_csv(bare, index=False)
    with pytest.raises(ValueError, match="evidence_level"):
        EV.vote_cross_neutrals([str(bare)])
    assert EV.vote_cross_neutrals([]) == set()
    with pytest.raises(FileNotFoundError):
        EV.vote_cross_neutrals([str(tmp_path / "nowhere")])


def test_corroborate_feeds_the_vote_and_an_orbitrap_run_dir_gives_partners(tmp_path, monkeypatch):
    src_dir = tmp_path / "srcrun"
    _run_batch(src_dir, monkeypatch)                               # an Orbitrap-class run dir
    tof_dir = tmp_path / "tofrun"
    _run_batch(tof_dir, monkeypatch, resolving_power=TOF)          # a TOF-class run dir
    lines = []
    _run_batch(tmp_path / "main", monkeypatch, corroborate=[str(src_dir), str(tof_dir)], lines=lines)
    ev = json.load(open(tmp_path / "main" / "batch_summary.json"))["evidence_levels"]
    assert ev["cross_source"] == [str(src_dir), str(tof_dir)] and ev["n_corroborate"] >= 1
    assert list(ev["partners"]) == ["srcrun"]                      # the TOF run dir gives none ...
    assert any("--corroborate tofrun: instrument class tof" in ln for ln in lines)   # ... and says so
    # a ledger CSV is not a run dir: the vote reads it, no partners
    csv = tmp_path / "one.csv"
    _ledger().to_csv(csv, index=False)
    lines2 = []
    _run_batch(tmp_path / "main2", monkeypatch, corroborate=[str(csv)], lines=lines2)
    ev2 = json.load(open(tmp_path / "main2" / "batch_summary.json"))["evidence_levels"]
    assert ev2["partners"] == {} and ev2["n_corroborate"] == 1
    assert any("one.csv: not a batch run dir" in ln for ln in lines2)


def test_cli_corroborate_is_repeatable_on_assign_and_batch_and_says_what_it_does():
    p = cli.build_parser()
    a = p.parse_args(["assign", "--sample-id", "x", "--corroborate", "r1", "--corroborate", "r2"])
    assert a.corroborate == ["r1", "r2"]
    assert p.parse_args(["assign", "--sample-id", "x"]).corroborate == []
    b = p.parse_args(["batch", "--batch", "b", "--corroborate", "r1"])
    assert b.corroborate == ["r1"]
    sub = next(a for a in p._actions if a.dest == "command" or getattr(a, "choices", None))
    helps = [act.help for act in sub.choices["batch"]._actions if act.dest == "corroborate"]
    assert helps and "feeds the merge vote's evidence class and (Orbitrap run dirs) the other-source " \
                     "partner tag; it never moves an evidence level" in " ".join(helps[0].split())


def test_reflists_activate_records_which_keyword_unlocked_each_tag():
    from peaky.assignment import reflists as RL
    from peaky.assignment.levels import lists as LS
    texts = ("alpha-pinene ozonolysis", "a dataset", "Nitrate CIMS")
    lists, tags = RL.activate(*texts)
    lists2, tags2, rec = RL.activate(*texts, record=True)
    assert [x.id for x in lists] == [x.id for x in lists2] and tags == tags2       # the same lists activate
    assert rec["tags"] == sorted(tags) and set(rec["matched"]) == set(tags)
    for tag, pairs in rec["matched"].items():
        for field, kw in pairs:
            assert field in RL.ACTIVATION_FIELDS and kw.lower() in texts[RL.ACTIVATION_FIELDS.index(field)].lower()
    assert rec == LS.activation_record(*texts)
    ctx = LS.reflists_context(lists2, rec)
    assert [a[0] for a in ctx["active"]] == [x.id for x in lists2]
    for lid, _ver, how in ctx["active"]:
        assert how == "always active" or how.startswith("keyword '"), how
    # two texts of a single sample, named by the caller
    _l, _t, rec1 = RL.activate("ambient-air", "Nitrate CIMS", record=True, fields=("context", "reagent label"))
    assert all(f in ("context", "reagent label") for pairs in rec1["matched"].values() for f, _k in pairs)
