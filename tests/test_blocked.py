"""Every exit writes why: the blocked reading and its gate on the unexplained rows.

`blocked.py` keeps what each gate between enumeration and commit stopped -- the
context filter, the commit gates, the pass-0 ppm gate, the labelled rescue's
gates, the grid box, a clear -- and `assign.run` stamps `blocked_reading` /
`blocked_by` on every unexplained row. Pinned here: the texts, the choice of
the best reading, the exits themselves on injected scorers, the ledger API that
keeps the two columns on unexplained rows only, the batch table, the
`would_lift` of a pair with no level -- and, on an offline synthetic run, that
the record changes no existing column: the ledger with the recorder switched
off is byte-identical to the one with it on, minus the two new columns.

Offline: no network.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from peaky.assignment import assign as A  # noqa: E402
from peaky.assignment import blocked as BL  # noqa: E402
from peaky.assignment import evidence as EV  # noqa: E402
from peaky.assignment import labeled as LB  # noqa: E402
from peaky.assignment import ledger as L  # noqa: E402
from peaky.assignment.passes import config as PCfg  # noqa: E402
from peaky.assignment.passes import core as PC  # noqa: E402
from peaky.assignment.passes import directors as PD  # noqa: E402
from peaky.chem import chemistry as C  # noqa: E402
from peaky.chem import contexts as X  # noqa: E402
from peaky.chem import profiles as P  # noqa: E402
from tests.test_c42_centre import _ACIDS, ORBI, _peaks  # noqa: E402


@pytest.fixture(autouse=True)
def _no_recorder_leaks():
    BL.stop()
    yield
    BL.stop()


# --------------------------------------------------------------------------- texts
def test_the_context_reason_reads_in_a_few_characters():
    prof = X.get_context("ambient-air")
    assert BL.compact_reason(X.filter_by_context("C5H10N2O8", prof)[1]) == "O/C 1.60>1.50"
    assert BL.compact_reason(X.filter_by_context("C4H2O4", prof)[1]) == "(H+X)/C 0.50<0.70"
    assert BL.compact_reason(X.filter_by_context("C6H6N2O2", prof)[1]) == "DBE/C 0.83>0.75"
    assert BL.compact_reason("N=4 > 3") == "N 4>3"
    assert BL.compact_reason("Cl in neutral needs C>=3 (got C=1); likely reagent alias") == "Cl on C 1<3"
    assert BL.compact_reason("no carbon") == "no carbon"


@pytest.mark.parametrize("text, klass", [
    ("context:O/C 1.60>1.50", "context:O/C"),
    ("context:(H+X)/C 0.50<0.70", "context:(H+X)/C"),
    ("score<min_raw 0.62<0.70 (pass 1)", "score<min_raw"),
    ("mass-gate z=2.6>2.0 without pattern (pass 1)", "mass-gate z without pattern"),
    ("mass-gate z=2.6>3.0 (pass 2)", "mass-gate z"),
    ("pass0:ppm 3.0>2.0", "pass0:ppm"),
    ("pass0:deferred: single channel; no diagnostic twin", "pass0:deferred"),
    ("cleared:mass-gate: z=2.2 without pattern evidence", "cleared:mass-gate"),
    ("cleared:isotope audit: predicted 13C satellite (741 cps) absent", "cleared:isotope audit"),
    ("labeled:sibling", "labeled:sibling"),
    ("labeled:O<3x15N", "labeled:O<3x15N"),
    ("no candidate in box (C0-16 H0-36 O0-30 N0-2, integer DBE)", "no candidate in box"),
    (BL.NOT_ADMITTED, "not admitted"),
    ("released: a line of the cleared reading C4H6O4 [M-H]-", "released"),
    ("cleared:displaced by certified-neutral C19H37ClN4O3", "cleared:displaced by certified-neutral"),
    ("pass4:z=2.6 without pattern evidence (deep series)", "pass4:z without pattern evidence"),
    ("pass4:|ppm|=1.20 without pattern evidence (iso-pair)", "pass4:|ppm| without pattern evidence"),
    ("pass4:score 0.18<0.50 (deep series)", "pass4:score"),
    ("labeled:off-cal z=3.1>2.6", "labeled:off-cal z"),
    ("labeled:degenerate (3 readings)", "labeled:degenerate"),
    ("context:N 4>3", "context:N"),
    ("context:Cl on C 1<3", "context:Cl on C"),
    ("confidence Reject 0.45<0.50 (pass 3)", "confidence Reject"),
    (None, "(none)"),
])
def test_a_gate_text_counts_under_its_class(text, klass):
    assert BL.gate_class(text) == klass


def test_a_reading_is_written_with_its_ion_score_and_ppm():
    b = BL._Block(gate="g", neutral="C5H10N2O8", adduct="[M+^NO3]-", score=0.912, ppm=0.214)
    t = BL.reading_text(b)
    assert t == "C5H10N2O8 [M+^NO3]- (C5H10N2^NO11-), score 0.91, ppm +0.21"
    assert BL.reading_key(t) == "C5H10N2O8 [M+^NO3]- (C5H10N2^NO11-)" and BL.reading_score(t) == pytest.approx(0.91)
    assert BL.reading_text(BL._Block(gate="g", neutral="C4H6O4", adduct="[M-H]-")) == \
        "C4H6O4 [M-H]- (C4H5O4-), score n/a"
    assert BL.reading_text(BL._Block(gate="released")) is None


def test_the_best_reading_is_the_highest_score_then_an_event_then_an_unscored_reading():
    rec = BL.Recorder()
    rec.add("p", "score<min_raw 0.30<0.70 (pass 1)", neutral="C1", adduct="[M-H]-", score=0.30, ppm=0.1)
    rec.add("p", "context:O/C 1.60>1.50", neutral="C2", adduct="[M-H]-", ppm=0.4)
    rec.add("p", "context:O/C 1.70>1.50", neutral="C3", adduct="[M-H]-", ppm=-0.1)
    assert BL.best(rec.blocks["p"]).neutral == "C3"            # unscored, nearest the centre
    rec.add("p", "released: a line of the cleared reading X [M-H]-")
    assert BL.best(rec.blocks["p"]).gate.startswith("released")
    rec.add("p", "score<min_raw 0.62<0.70 (pass 1)", neutral="C4", adduct="[M-H]-", score=0.62)
    assert BL.best(rec.blocks["p"]).neutral == "C4"            # a reading at or above the floor wins
    # one reading met again keeps its FIRST exit; the later one only fills what it lacked
    rec.add("p", "pass4:score 0.62<0.50 (deep series)", neutral="C4", adduct="[M-H]-", score=0.62, ppm=0.3)
    b = BL.best(rec.blocks["p"])
    assert b.gate == "score<min_raw 0.62<0.70 (pass 1)" and b.ppm == 0.3
    # ... unless it got through and was cleared: the clear replaces it
    rec.add("p", "cleared:mass-gate: z=2.2", neutral="C4", adduct="[M-H]-", score=0.75)
    assert BL.best(rec.blocks["p"]).gate.startswith("cleared:")


def test_a_formula_reaches_the_targets_inside_the_window_only():
    tg = pd.DataFrame({"peak_id": ["a", "b"], "mz": [C.ion_mz("C5H8O4", "[M-H]-") * (1 + 1e-6), 300.0]})
    hits = BL.attribute(tg, {"C5H8O4", "C9H14O4"}, ["[M-H]-", "[M+NO3]-"], 3.0)
    assert [(h[0], h[1], h[2]) for h in hits] == [("a", "C5H8O4", "[M-H]-")]
    assert hits[0][3] == pytest.approx(1.0, abs=1e-6)


def test_a_note_that_raises_is_counted_and_never_reaches_the_pass():
    rec = BL.start()
    BL.note_at("p", "g", 100.0, neutral="C4H6O4", adduct="[M+nothing]-")       # an adduct with no mass
    assert rec.errors == 1 and rec.first_error.startswith("note_at: KeyError") and "p" not in rec.blocks
    led = L.new_ledger(pd.DataFrame({"peak_id": ["p"], "mz": [100.0], "height": [1.0]}))
    logs = []
    BL.stamp(led, rec, log=logs.append)
    assert any("1 note(s) failed" in m for m in logs)


def test_every_note_is_a_no_op_outside_a_run():
    assert not BL.active()
    BL.note("p", "g", neutral="C1", adduct="[M-H]-", score=0.5)
    BL.note_context(pd.DataFrame({"peak_id": ["p"], "mz": [100.0]}), {"C5H8O4": "x"}, ["[M-H]-"], 3.0)
    assert not BL.active()


def test_a_run_on_another_thread_never_notes_into_or_stops_this_ones():
    import threading
    mine = BL.start()
    seen = {}

    def other():
        seen["active_before"] = BL.active()
        theirs = BL.start()
        BL.note("q", "g", neutral="C1", adduct="[M-H]-")
        BL.stop()
        seen["theirs"] = set(theirs.blocks)

    t = threading.Thread(target=other)
    t.start()
    t.join()
    assert seen == {"active_before": False, "theirs": {"q"}}
    assert BL.active() and not mine.blocks
    BL.note("p", "g", neutral="C2", adduct="[M-H]-")
    assert set(mine.blocks) == {"p"}


# --------------------------------------------------------------------------- the ledger API
def _ledger():
    led = L.new_ledger(pd.DataFrame({"peak_id": ["a", "b", "c"], "mz": [131.035, 132.038, 150.0],
                                     "height": [1e5, 6e3, 1e3]}))
    L.commit_assignment(led, "a", neutral_formula="C5H8O4", adduct="[M-H]-", ion_formula="C5H7O4-",
                        ion_score=0.81, compound_score=0.79, ppm_error=0.4, pass_no=1,
                        method="cheminfo+grid", confidence="Good", commentary="c")
    L.attach_isotopologue(led, "b", "a", iso_label="13C")
    return led


def test_a_clear_during_a_run_is_stamped_as_the_rows_blocked_reading():
    rec = BL.start()
    led = _ledger()
    L.clear_assignment(led, "a", reason="mass-gate: z=2.2 without pattern evidence")
    BL.stamp(led, rec, log=lambda *a: None)
    a, b = led.set_index("peak_id").loc["a"], led.set_index("peak_id").loc["b"]
    assert a[BL.GATE] == "cleared:mass-gate: z=2.2 without pattern evidence"
    assert a[BL.READING] == "C5H8O4 [M-H]- (C5H7O4-), score 0.79, ppm +0.40"
    assert b[BL.GATE] == "released: a line of the cleared reading C5H8O4 [M-H]-" and pd.isna(b[BL.READING])
    assert list(led.columns[-2:]) == list(BL.COLUMNS)


def test_after_the_run_the_ledger_api_keeps_the_columns_on_unexplained_rows_only():
    led = _ledger()
    BL.stamp(led, BL.Recorder(), log=lambda *a: None)          # the columns, as a written ledger carries them
    assert led[BL.GATE].isna().tolist() == [True, True, False]
    # the batch's per-file rewrite clears a reading: written on the row directly
    L.clear_assignment(led, "a", reason="batch isotopologue gate: the 13C line of X")
    a = led.set_index("peak_id").loc["a"]
    assert a[BL.GATE] == "cleared:batch isotopologue gate: the 13C line of X"
    assert a[BL.READING].startswith("C5H8O4 [M-H]- (C5H7O4-), score 0.79")
    # a commit, an attach, a reagent / artifact mark empty them again
    L.commit_assignment(led, "a", neutral_formula="C5H8O4", adduct="[M-H]-", ion_score=0.9, pass_no=1,
                        method="m", confidence="Good", commentary="c")
    L.attach_isotopologue(led, "b", "a", iso_label="13C")
    L.mark_artifact(led, "c", "ringing")
    assert led[list(BL.COLUMNS)].isna().all().all()


# --------------------------------------------------------------------------- the exits
def test_commit_winners_notes_the_winner_each_gate_refuses():
    led = L.new_ledger(pd.DataFrame({"peak_id": ["a", "b", "c"], "mz": [131.0, 145.0, 159.0],
                                     "height": [1e5, 1e5, 1e5]}))
    cfg = PCfg.PassConfig(height_cutoff_cps=10)
    cfg.cal_mu, cfg.cal_sigma = 0.0, 0.3
    win = pd.DataFrame([
        dict(peak_id="a", neutral="C5H8O4", ion_formula="C5H7O4-", adduct="[M-H]-", ion_score=0.62,
             compound_score=0.62, raw_score=0.62, eff_score=0.62, ppm_error=0.1, n_iso=0, tied=False,
             alternatives=[]),
        dict(peak_id="b", neutral="C6H10O4", ion_formula="C6H9O4-", adduct="[M-H]-", ion_score=0.9,
             compound_score=0.9, raw_score=0.9, eff_score=0.9, ppm_error=1.5, n_iso=0, tied=False,
             alternatives=[]),
        dict(peak_id="c", neutral="C7H12O4", ion_formula="C7H11O4-", adduct="[M-H]-", ion_score=0.9,
             compound_score=0.9, raw_score=0.9, eff_score=0.9, ppm_error=0.1, n_iso=0, tied=False,
             alternatives=[]),
    ])
    rec = BL.start()
    s = PC.commit_winners(led, {"winners": win}, pass_no=1, method="cheminfo+grid", context="ambient-air",
                          cfg=cfg, lock=False, min_raw_score=0.70)
    assert s["committed"] == 1 and s["rejected"]["mass_gate"] == 1
    assert set(rec.blocks) == {"a", "b"}
    assert BL.best(rec.blocks["a"]).gate == "score<min_raw 0.62<0.70 (pass 1)"
    assert BL.best(rec.blocks["b"]).gate == "mass-gate z=5.0>2.0 without pattern (pass 1)"


def _scored_row(formula, adduct, ion, pid, mz, ppm, score, is_base=True, label="M0"):
    return dict(compound_formula=formula, ion_formula=ion, mechanism_id=adduct, is_base=is_base,
                ion_score=score, compound_score=score, sample_peak_id=pid, sample_peak_mz=mz,
                ppm_error=ppm, iso_score=score, iso_label=label)


def test_the_pass0_ppm_gate_notes_the_known_species_it_refuses():
    mz = C.ion_mz("HNO2", "[M-H]-") * (1 - 3e-6)
    led = L.new_ledger(pd.DataFrame({"peak_id": ["n"], "mz": [mz], "height": [8e3]}))
    cfg = PCfg.PassConfig(height_cutoff_cps=10)

    def score_fn(client, sid, formulas, mechanism_ids=None):
        return pd.DataFrame([_scored_row("HNO2", "[M-H]-", "NO2-", "n", mz, -3.0, 0.12)])

    rec = BL.start()
    PD.run_pass0_known(None, "s", led, X.get_context("ambient-air"), cfg, ["[M-H]-"], score_fn=score_fn,
                       log=lambda *a: None)
    b = BL.best(rec.blocks["n"])
    assert b.gate == "pass0:ppm 3.0>2.0" and (b.neutral, b.adduct, b.score) == ("HNO2", "[M-H]-", 0.12)
    assert L.role_of(led, "n") == L.ROLE_UNEXPLAINED


def test_the_labelled_rescue_notes_the_sibling_gate():
    prof = P.resolve("NO3_15N")
    neu = "C9H15^NO5"
    mz = C.ion_mz(neu, "[M-H]-")
    led = L.new_ledger(pd.DataFrame({"peak_id": ["p1"], "mz": [mz], "height": [5e4]}))

    def score_fn(client, sid, formulas, allow_partial=True, mechanism_ids=None):
        # a confident on-cal reading with NO matched sibling line
        return pd.DataFrame([_scored_row(neu, "[M-H]-", "C9H14^NO5-", "p1", mz, 0.2, 0.95)])

    rec = BL.start()
    out = LB.rescue_labeled(None, "s", led, X.get_context(prof.context), PCfg.PassConfig(height_cutoff_cps=10),
                            adducts=["[M-H]-"], label_isotope="^N", label_max=1, score_fn=score_fn,
                            log=lambda *a: None)
    assert out["rescued"] == 0
    b = BL.best(rec.blocks["p1"])
    assert b.gate == "labeled:sibling" and b.neutral == neu and b.score == pytest.approx(0.95)


@pytest.mark.parametrize("context, family, adduct, neutral, gated", [
    ("ambient-air", "organosulfate", "[M-H]-", "C5H10N2O8", False),     # O/C 1.60: a profile-default family
    ("uronium", "organochlorine", "[M+H]+", "C5H9ClO8", True),          # the 37Cl-gated family, Cl lifted
])
def test_a_pass3_familys_context_rejection_is_the_peaks_blocked_reading(monkeypatch, context, family, adduct,
                                                                        neutral, gated):
    """Pass 3's context filter is an exit for every family: the reading it
    rejects for an unexplained peak is the peak's blocked reading. An
    isotope-gated family filters under the gate's profile (its element's cap
    lifted), so its reading is refused for its O/C, not for its chlorine."""
    import dataclasses
    from peaky.chem import isotopes as ISO
    prof = X.get_context(context)
    mz = C.ion_mz(neutral, adduct)
    rows = [("p", mz, 5e4)] + ([("p37", mz + ISO.D_37CL, ISO.R_37CL_PER_CL * 5e4)] if gated else [])
    led = L.new_ledger(pd.DataFrame(rows, columns=["peak_id", "mz", "height"]))
    assert X.family_iso_gate(family) == ("Cl" if gated else None)
    assert not X.filter_by_profile(neutral, prof)[0]
    monkeypatch.setattr(PD, "_enumerate", lambda client, mzs, mech_ids, ranges, cfg, adducts, **kw: {neutral})
    monkeypatch.setattr(PD, "_mech_ids_for", lambda client, adducts: [])
    monkeypatch.setattr(PD.IO, "score_candidates", lambda *a, **k: pd.DataFrame())
    cfg = PCfg.PassConfig(height_cutoff_cps=10)
    cfg.instrument_class = "orbitrap"
    rec = BL.start()
    PD.run_pass3(None, "s", led, dataclasses.replace(prof, pass3_families=(family,)), None, cfg, [adduct],
                 reagent=None, log=lambda *a: None)
    b = BL.best(rec.blocks.get("p"))
    assert b is not None and b.neutral == neutral and b.adduct == adduct, b
    BL.stamp(led, log=lambda *a: None)
    by = led.set_index("peak_id")
    assert L.role_of(led, "p") == L.ROLE_UNEXPLAINED
    assert by.at["p", "blocked_by"] == "context:O/C 1.60>1.50", by.at["p", "blocked_by"]
    assert by.at["p", "blocked_reading"].startswith(f"{neutral} {adduct}")


# --------------------------------------------------------------------------- an offline run
def _spectrum() -> pd.DataFrame:
    """CHO acids on the trend (commit, calibrate), an organonitrate the ambient
    VK window refuses (O/C 1.60), an acid 3 ppm off, a mass no CHON formula
    reaches and a peak under the height gate."""
    pk = _peaks(_ACIDS)
    nit = _peaks(["C5H10N2O8"], start=100, ppm=lambda mz: 0.2)
    off = _peaks(["C6H10O5"], start=200, ppm=lambda mz: 3.0)
    odd = pd.DataFrame([{"peak_id": "odd", "mz": 211.5432, "height": 5e4, "area": 5e4, "signal_to_noise": 1e3},
                        {"peak_id": "tiny", "mz": 233.1111, "height": 50, "area": 50, "signal_to_noise": 2}])
    return pd.concat([pk, nit, off, odd], ignore_index=True).sort_values("mz").reset_index(drop=True)


def _run(sid):
    cfg = PCfg.PassConfig(height_cutoff_cps=100)
    return A.run(sid, context="ambient-air", cfg=cfg, peaks=_spectrum(), adducts=["[M-H]-"], use_cache=False,
                 log=lambda *a: None, scoring=ORBI)


@pytest.fixture(scope="module")
def runs():
    on = _run("blocked-on")
    with pytest.MonkeyPatch.context() as mp:
        # the recorder off: no exit notes anything and nothing is stamped
        mp.setattr(BL, "start", lambda: BL.Recorder())
        mp.setattr(BL, "stamp", lambda *a, **k: {})
        off = _run("blocked-off")
    return on, off


def test_the_record_changes_no_existing_column(runs):
    on, off = runs
    lon, loff = on["ledger"], off["ledger"]
    assert list(lon.columns) == list(loff.columns) + list(BL.COLUMNS)
    assert lon[list(loff.columns)].to_csv(index=False) == loff.to_csv(index=False)
    assert on["stats"] == off["stats"]
    assert on["problems"] == off["problems"] == []


def test_every_unexplained_row_says_why_and_no_other_row_does(runs):
    led = runs[0]["ledger"].set_index("peak_id")
    un = led["role"] == L.ROLE_UNEXPLAINED
    assert un.sum() >= 5
    assert led.loc[un, BL.GATE].notna().all()
    assert led.loc[~un, list(BL.COLUMNS)].isna().all().all()
    nit = led.loc["p100_0"]
    assert nit[BL.GATE] == "context:O/C 1.60>1.50"
    assert nit[BL.READING].startswith("C5H10N2O8 [M-H]- (C5H9N2O8-), score ")
    assert BL.reading_score(nit[BL.READING]) > 0.9          # scored once, at the stamp
    assert led.loc["p200_0", BL.READING].startswith("C6H10O5 [M-H]- (C6H9O5-)")
    assert led.loc["p200_0", BL.READING].endswith("ppm +3.00")
    assert led.loc["p200_0", BL.GATE] == "score<min_raw 0.00<0.70 (pass 1)"     # its first exit
    assert led.loc["odd", BL.GATE].startswith("no candidate in box (C0-") and pd.isna(led.loc["odd", BL.READING])
    assert led.loc["tiny", BL.GATE] == BL.NOT_ADMITTED


# --------------------------------------------------------------------------- the batch table
def _file(rows):
    led = L.new_ledger(pd.DataFrame([{"peak_id": r[0], "mz": r[1], "height": r[2]} for r in rows]))
    led[BL.READING] = pd.Series([r[3] for r in rows], dtype=object)
    led[BL.GATE] = pd.Series([r[4] for r in rows], dtype=object)
    return led


def test_the_batch_table_pools_the_unexplained_rows_by_mz():
    r1 = "C5H10N2O8 [M-H]- (C5H9N2O8-), score 0.95, ppm +0.20"
    r2 = "C5H10N2O8 [M-H]- (C5H9N2O8-), score 0.91, ppm +0.31"
    f1 = _file([("a", 225.03648, 2e4, r1, "context:O/C 1.60>1.50"), ("b", 100.0, 50.0, None, BL.NOT_ADMITTED)])
    f2 = _file([("a", 225.03652, 3e4, r2, "context:O/C 1.60>1.50")])
    f3 = _file([("x", 225.03650, 1e4, "C9H5N5 [M-H]- (C9H4N5-), score 0.20, ppm -0.4",
                 "score<min_raw 0.20<0.70 (pass 1)")])
    L.commit_assignment(f3, "x", neutral_formula="C9H5N5", adduct="[M-H]-", ion_score=0.9, pass_no=1,
                        method="m", confidence="Good", commentary="c")    # an M0 row is not pooled
    f3 = pd.concat([f3, _file([("y", 225.03651, 1e4, "C9H5N5 [M-H]- (C9H4N5-), score 0.20, ppm -0.4",
                                "score<min_raw 0.20<0.70 (pass 1)")])], ignore_index=True)
    t = BL.batch_table({"f1": f1, "f2": f2, "f3": f3}, tol_ppm=3.0)
    assert list(t.columns) == list(BL.TABLE_COLUMNS) and len(t) == 2
    top = t.iloc[0]
    assert top["n_files"] == 3 and top["n_rows"] == 3 and top["median_height"] == pytest.approx(2e4)
    assert top["reading"] == "C5H10N2O8 [M-H]- (C5H9N2O8-)" and top["n_files_reading"] == 2
    assert top["reading_score_median"] == pytest.approx(0.93)
    assert top["gate"] == "context:O/C" and top["gates"] == "context:O/C 2; score<min_raw 1"
    assert t.iloc[1]["gate"] == "not admitted" and t.iloc[1]["reading"] == ""
    # an offset moves a file's rows before the clustering, as the merge aligns
    split = BL.batch_table({"f1": f1, "f2": f2}, tol_ppm=3.0, offsets={"f2": 10.0})
    assert len(split) == 3
    assert BL.batch_table({}, tol_ppm=3.0).columns.tolist() == list(BL.TABLE_COLUMNS)


def test_the_batch_writes_the_table(tmp_path, monkeypatch):
    from tests.test_evidence_outputs import _run_batch
    _run_batch(tmp_path, monkeypatch)
    t = pd.read_csv(tmp_path / "tables" / "unexplained_blocked.csv", keep_default_na=False)
    assert list(t.columns) == list(BL.TABLE_COLUMNS)


# --------------------------------------------------------------------------- would_lift on no level
def test_a_pair_with_no_level_says_what_would_lift_it():
    assert EV.NO_RUN_WINDOW_LIFT == ("no file of this source calibrated (needs ≥ 20 isotope-backed core rows "
                                     "in a file, or a batch-pooled core of ≥ 15 distinct carbon-bearing ions "
                                     "isotope-backed in ≥ 2 files)")
    assert EV.NO_POOLED_PAIR_LIFT == "batch-level re-read: the re-read pair is not levelled"
    merged = pd.DataFrame([dict(mz=300.0, neutral_formula="C9H99O9", adduct="[M-H]-", tier="Candidate")])
    out = EV.stamp_merged(merged, None)
    assert out.at[0, "would_lift"] == EV.NO_POOLED_PAIR_LIFT and out.at[0, "evidence"] == EV.NO_POOLED_PAIR_TEXT


def test_an_uncalibrated_source_says_a_calibrated_file_would_lift_it():
    from tests.test_evidence_outputs import _inputs, _ledger as _ev_ledger
    from peaky.assignment.levels import source as SRC
    led = _ev_ledger()
    EV.apply_levels(led, run_inputs=_inputs(cal=None))
    m0 = led[led["role"] == "M0"]
    assert (m0["evidence"] == EV.NO_WINDOW_TEXT).all() and (m0["would_lift"] == EV.NO_WINDOW_LIFT).all()
    assert led.loc[led["role"] != "M0", "would_lift"].isna().all()
    # pooled: a source none of whose files is calibrated has no run window
    src = EV.source_from_frames({"f1": _ev_ledger()}, run_inputs=_inputs(cal=None), mode="run")
    assert SRC.no_run_window(src)
    frame = EV.level_source(src)
    assert len(frame) == 2 and (frame["evidence"] == EV.NO_RUN_WINDOW_TEXT).all()
    assert (frame["would_lift"] == EV.NO_RUN_WINDOW_LIFT).all() and (frame["claim"] == "tentative").all()
    # ... and the merged rows it is stamped on say the same; a re-read says why it has none
    merged = pd.DataFrame([dict(mz=199.09, neutral_formula="C10H16O4", adduct="[M-H]-", tier="Assigned"),
                           dict(mz=300.0, neutral_formula="C9H99O9", adduct="[M-H]-", tier="Candidate")])
    out = EV.stamp_merged(merged, frame)
    assert out["would_lift"].tolist() == [EV.NO_RUN_WINDOW_LIFT, EV.NO_POOLED_PAIR_LIFT]
