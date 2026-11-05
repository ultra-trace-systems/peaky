"""Real but unidentified lines reach the outputs (peaky/batch/unidentified.py):
tables/unidentified.csv (one row per bright unidentified m/z bin, with why) and
the merge's vote-drop recorder (tables/vote_drops.csv)."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from peaky.assignment import blocked as BL
from peaky.assignment import evidence as EV
from peaky.assignment import ledger as L
from peaky.batch import assign_batch as AB
from peaky.batch import unidentified as UN
from peaky.chem import chemistry as C
from peaky.chem import isotopes as ISO
from peaky.io import io_mascope as IO

EDGE = 4.0
PAR = ("C10H16O4", "[M-H]-")
PMZ = C.ion_mz(*PAR)
UNK = 287.12345            # a line no formula fits
BOX = "no candidate in box (C0-40 H0-84 O0-30 N0-3, integer DBE)"


def _ledger(sid, *, unk_h=None, parent_tier="Assigned", child=False, extra_m0=None):
    """A per-file ledger: the parent M0 (``parent_tier``), optionally its 13C child, an
    unexplained line at UNK (``unk_h``) stamped as the exit recorder stamps it, faint
    background rows, and ``extra_m0`` = (peak_id, mz, neutral, adduct, tier)."""
    rows = [dict(peak_id=f"{sid}p", mz=PMZ, height=2.0e5, area=2.0e5)]
    if child:
        rows.append(dict(peak_id=f"{sid}c", mz=PMZ + ISO.D_13C_EXACT, height=2.0e4, area=2.0e4))
    if unk_h is not None:
        rows.append(dict(peak_id=f"{sid}u", mz=UNK, height=unk_h, area=unk_h))
    if extra_m0 is not None:
        rows.append(dict(peak_id=extra_m0[0], mz=extra_m0[1], height=3.0e4, area=3.0e4))
    rows += [dict(peak_id=f"{sid}n{k}", mz=150.0 + 7.3 * k, height=6.0, area=6.0) for k in range(5)]
    led = L.new_ledger(pd.DataFrame(rows))
    L.commit_assignment(led, f"{sid}p", neutral_formula=PAR[0], adduct=PAR[1], ion_formula="C10H15O4-",
                        ion_score=0.97, compound_score=0.96, ppm_error=0.1, pass_no=1, method="cheminfo",
                        confidence="High", commentary="Pass 1")
    if child:
        L.attach_isotopologue(led, f"{sid}c", f"{sid}p", iso_label="13C", iso_match_score=0.9)
    if extra_m0 is not None:
        pid, _mz, n, a, _t = extra_m0
        L.commit_assignment(led, pid, neutral_formula=n, adduct=a,
                            ion_formula=C.format_formula(EV.ion_composition(n, a, None)) + "-", ion_score=0.9,
                            compound_score=0.9, ppm_error=0.2, pass_no=1, method="cheminfo", confidence="High",
                            commentary="Pass 1")
    led["tier"] = led["tier"].astype(object)
    led.loc[led["peak_id"] == f"{sid}p", "tier"] = parent_tier
    if extra_m0 is not None:
        led.loc[led["peak_id"] == extra_m0[0], "tier"] = extra_m0[4]
    led[BL.READING] = pd.Series(pd.NA, index=led.index, dtype=object)
    led[BL.GATE] = pd.Series(pd.NA, index=led.index, dtype=object)
    u = led["role"] == L.ROLE_UNEXPLAINED
    led.loc[u, BL.GATE] = BOX
    return led


def _merged(*rows):
    return pd.DataFrame([dict(mz=m, neutral_formula=n, adduct=a, tier=t, tier_reason=pd.NA) for m, n, a, t in rows])


STATS = {s: {"sample_id": s, "noise_edge_cps": EDGE} for s in ("f1", "f2", "f3")}


# --------------------------------------------------------------------------- unexplained, no formula
def test_a_bright_unexplained_line_in_two_files_is_one_row_no_formula_with_its_readings():
    leds = {"f1": _ledger("f1", unk_h=4.0e3), "f2": _ledger("f2", unk_h=6.0e3), "f3": _ledger("f3")}
    leds["f2"].loc[leds["f2"]["peak_id"] == "f2u", [BL.READING, BL.GATE]] = [
        "C12H24O6S2 [M-H]- (C12H23O6S2-), score 0.31, ppm +2.10", "score<min_raw 0.31<0.70 (pass 1)"]
    t = UN.build(leds, STATS, _merged((PMZ, *PAR, "Assigned")), tol_ppm=6.0)
    assert list(t.columns) == list(UN.TABLE_COLUMNS) and len(t) == 1
    r = t.iloc[0]
    assert r["why"] == "no_formula" and r["roles"] == "unexplained:2"
    assert r["n_files_present"] == 2 and r["n_files_selected"] == 3
    assert r["h_max"] == 6.0e3 and r["x_edge_max"] == pytest.approx(6.0e3 / EDGE)
    assert r["mz"] == pytest.approx(UNK, abs=1e-5) and pd.isna(r["mz_corrected"])
    tried = json.loads(r["readings_tried"])
    assert {d["blocked_by"] for d in tried} == {BOX, "score<min_raw 0.31<0.70 (pass 1)"}
    assert any(d["neutral"] == "C12H24O6S2" and d["ion"] == "C12H23O6S2-" for d in tried)
    # no 13C line where a one-carbon line would show: the line carries no carbon (a measured 0)
    assert r["c13_count"] == pytest.approx(0.0) and not bool(r["episode"])
    # a one-atom 34S / 37Cl line would clear 3x the edge (no server S/N on these rows): absent, not untestable
    assert r["s34_line"] == "absent" and r["cl37_line"] == "absent"


def test_a_faint_unexplained_line_is_not_listed_and_a_merged_row_explains_a_line():
    leds = {"f1": _ledger("f1", unk_h=15.0), "f2": _ledger("f2", unk_h=18.0)}          # under 5 x the edge
    assert len(UN.build(leds, STATS, _merged((PMZ, *PAR, "Assigned")), tol_ppm=6.0)) == 0
    leds = {"f1": _ledger("f1", unk_h=4.0e3), "f2": _ledger("f2", unk_h=4.0e3)}
    # another file's reading at the line reached the merged ledger (any tier): not unidentified
    m = _merged((PMZ, *PAR, "Assigned"), (UNK + 0.0005, "C15H20O5", "[M-H]-", "Candidate"))
    assert len(UN.build(leds, STATS, m, tol_ppm=6.0)) == 0
    # ... an episode flag makes a faint line bright (the hook an episode-aware selection sets)
    leds = {"f1": _ledger("f1", unk_h=15.0)}
    leds["f1"]["episode"] = leds["f1"]["peak_id"] == "f1u"
    t = UN.build(leds, STATS, _merged((PMZ, *PAR, "Assigned")), tol_ppm=6.0)
    assert len(t) == 1 and bool(t.iloc[0]["episode"]) is True


def test_a_reading_tried_and_refused_by_a_gate_is_all_blocked():
    leds = {"f1": _ledger("f1", unk_h=4.0e3)}
    leds["f1"].loc[leds["f1"]["peak_id"] == "f1u", [BL.READING, BL.GATE]] = [
        "C10H17NO8 [M-H]- (C10H16NO8-), score 0.93, ppm +0.20", "context:O/C 0.80>0.75"]
    t = UN.build(leds, STATS, _merged((PMZ, *PAR, "Assigned")), tol_ppm=6.0)
    assert t.iloc[0]["why"] == "all_blocked"


# --------------------------------------------------------------------------- orphaned children
def test_an_iso_child_of_a_candidate_parent_is_an_orphan_with_its_parent_as_partner():
    leds = {"f1": _ledger("f1", child=True, parent_tier="Candidate"),
            "f2": _ledger("f2", child=True, parent_tier="Candidate")}
    t = UN.build(leds, STATS, _merged((PMZ, *PAR, "Candidate")), tol_ppm=6.0)
    assert len(t) == 1
    r = t.iloc[0]
    assert r["why"] == "orphan_iso_child" and r["roles"] == "iso_child:2"
    assert r["partner_mz"] == pytest.approx(PMZ, abs=1e-5) and r["partner_role"] == f"M0 Candidate {PAR[0]} {PAR[1]}"
    assert r["mz"] == pytest.approx(PMZ + ISO.D_13C_EXACT, abs=1e-5)
    tried = json.loads(r["readings_tried"])
    assert len(tried) == 1 and "parent not Assigned in the merged ledger" in tried[0]["blocked_by"]
    # the same child under an Assigned parent is explained
    leds = {"f1": _ledger("f1", child=True), "f2": _ledger("f2", child=True)}
    assert len(UN.build(leds, STATS, _merged((PMZ, *PAR, "Assigned")), tol_ppm=6.0)) == 0


def test_an_artifact_whose_ringing_parent_is_unexplained_is_an_orphan_artifact():
    led = _ledger("f1", unk_h=4.0e5)
    i = led.index[led["peak_id"] == "f1n0"][0]
    led.at[i, "mz"] = UNK + 0.011
    led.at[i, "height"] = 2.0e3
    L.mark_artifact(led, "f1n0", f"FT ringing/sidelobe of {UNK:.4f} (400000 cps, 200x brighter)")
    t = UN.build({"f1": led}, STATS, _merged((PMZ, *PAR, "Assigned")), tol_ppm=6.0)
    whys = dict(zip(t["mz"].round(3), t["why"]))
    assert whys[round(UNK + 0.011, 3)] == "orphan_artifact" and whys[round(UNK, 3)] == "no_formula"
    art = t[t["why"] == "orphan_artifact"].iloc[0]
    assert art["partner_mz"] == pytest.approx(UNK, abs=1e-4) and art["partner_role"] == "unexplained"


# --------------------------------------------------------------------------- the vote's drops
def test_a_reading_assigned_in_one_file_and_removed_after_the_vote_is_vote_dropped():
    xmz = 245.0301
    leds = {"f1": _ledger("f1", extra_m0=("f1x", xmz, "C9H10O7", "[M-H]-", "Assigned")), "f2": _ledger("f2")}
    before = _merged((PMZ, *PAR, "Assigned"), (xmz, "C9H10O7", "[M-H]-", "Assigned"))
    after = before.iloc[:1].copy()
    log = UN.DropLog()
    assert log.note(before, after, "isotopologue strip") == 1
    align_input = {s: AB._m0(led) for s, led in leds.items()}
    d = UN.vote_drops(leds, align_input, after, log, tol_ppm=6.0)
    assert list(d.columns) == list(UN.DROP_COLUMNS) and len(d) == 1
    r = d.iloc[0]
    assert (r["file"], r["peak_id"], r["tier_in"]) == ("f1", "f1x", "Assigned")
    assert r["reason"] == "isotopologue strip (C9H10O7 [M-H]-)"
    t = UN.build(leds, STATS, after, drops=d, tol_ppm=6.0)
    assert len(t) == 1 and t.iloc[0]["why"] == "vote_dropped" and t.iloc[0]["roles"] == "M0:1"
    tried = json.loads(t.iloc[0]["readings_tried"])
    assert tried == [{"adduct": "[M-H]-", "blocked_by": "isotopologue strip (C9H10O7 [M-H]-)", "ion": "C9H9O7-",
                      "neutral": "C9H10O7"}]
    # a row the vote kept (a merged row at its m/z) is no drop, and a row kept out of the vote
    # (the trace-first scope) says so
    assert len(UN.vote_drops(leds, align_input, before, log, tol_ppm=6.0)) == 0
    scoped = {"f1": align_input["f1"].drop(index=leds["f1"].index[leds["f1"]["peak_id"] == "f1x"]),
              "f2": align_input["f2"]}
    assert UN.vote_drops(leds, scoped, after, UN.DropLog(), tol_ppm=6.0)["reason"].tolist() == [UN.REASON_SCOPE]


# --------------------------------------------------------------------------- inside a batch
def _run(tmp_path, monkeypatch, ledgers):
    from peaky.assignment import assign as A
    t0 = pd.Timestamp("2025-10-01 00:00:00", tz="UTC")
    sids = sorted(ledgers)
    ts = pd.DataFrame([dict(sample_item_id=s, sample_item_name=f"n{k}", datetime_utc=t0 + pd.Timedelta(hours=k),
                            mz=float(m), height=float(h))
                       for k, s in enumerate(sids)
                       for m, h in zip(ledgers[s]["mz"], ledgers[s]["height"])]
                      + [dict(sample_item_id=s, sample_item_name=f"n{k}", datetime_utc=t0 + pd.Timedelta(hours=k),
                              mz=float(400 + 3 * j + k), height=500.0) for k, s in enumerate(sids) for j in range(15)])

    def fake_assign(sid, context="ambient-air", **kw):
        return {"ledger": ledgers[sid].copy(),
                "stats": {"noise_edge_cps": EDGE, "height_gate_cps": 10.0, "side_channels": [],
                          "degeneracy_cal": {"mu": 0.0, "sigma": 0.3}},
                "plausibility_audit": [], "summaries": {}, "problems": []}

    monkeypatch.setattr(IO, "connect", lambda *a, **k: "CLIENT")
    monkeypatch.setattr(IO, "fetch_peaks", lambda client, sid, use_cache=True: pd.DataFrame(
        {"peak_id": ["A"], "mz": [300.1], "height": [1.0e5]}))
    monkeypatch.setattr(IO, "estimate_offset", lambda raw: 0.0)
    monkeypatch.setattr(A, "run", fake_assign)
    AB.run(peaks=ts, ts_peaks=ts, reagent="NO3", batch="test batch", out_dir=str(tmp_path), k_min=len(sids),
           k_max=len(sids), min_gain=0.0, n_jobs=1, residual=False, resolving_power=120_000, log=lambda *a: None)
    return json.load(open(tmp_path / "batch_summary.json"))


def test_the_batch_writes_the_table_the_drop_list_and_the_count(tmp_path, monkeypatch):
    leds = {"u00": _ledger("u00", unk_h=4.0e3), "u01": _ledger("u01", unk_h=6.0e3), "u02": _ledger("u02")}
    summ = _run(tmp_path, monkeypatch, leds)
    t = pd.read_csv(tmp_path / "tables" / "unidentified.csv")
    assert list(t.columns) == list(UN.TABLE_COLUMNS)
    r = t[(t["mz"] - UNK).abs() < 1e-3]
    assert len(r) == 1 and r.iloc[0]["why"] == "no_formula" and r.iloc[0]["roles"] == "unexplained:2"
    assert r.iloc[0]["first_seen"].startswith("2025-10-01T00:00") and r.iloc[0]["last_seen"].startswith("2025-10-01T01:00")
    assert {d["blocked_by"] for d in json.loads(r.iloc[0]["readings_tried"])} == {BOX}
    assert summ["unidentified_bright"] == len(t) and summ["unidentified"]["by_why"]["no_formula"] >= 1
    d = pd.read_csv(tmp_path / "tables" / "vote_drops.csv")
    assert list(d.columns) == list(UN.DROP_COLUMNS) and summ["unidentified"]["vote_drops"] == len(d)


GOLDEN = Path(__file__).parent / "fixtures" / "unidentified"


def test_a_batch_with_no_bright_unidentified_line_has_an_empty_table_and_its_merged_ledger(tmp_path, monkeypatch):
    """Byte identity: the evidence-output fixture batch (every line explained) writes the
    header-only table and the merged ledger the parent commit wrote (golden)."""
    from tests.test_evidence_outputs import _run_batch
    _run_batch(tmp_path, monkeypatch)
    t = pd.read_csv(tmp_path / "tables" / "unidentified.csv")
    assert len(t) == 0 and list(t.columns) == list(UN.TABLE_COLUMNS)
    assert (tmp_path / "merged_ledger.csv").read_bytes() == (GOLDEN / "merged_ledger_evidence_batch.csv").read_bytes()
    assert json.load(open(tmp_path / "batch_summary.json"))["unidentified_bright"] == 0


# --------------------------------------------------------------------------- report, PDF, workbook
def test_the_pdf_section_and_the_workbook_sheet_show_the_rows(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.backends.backend_pdf import PdfPages
    from peaky.reporting import pdf_report as PR
    from peaky.reporting import report as RP
    leds = {"f1": _ledger("f1", unk_h=4.0e3), "f2": _ledger("f2", unk_h=6.0e3, child=True, parent_tier="Candidate")}
    t = UN.build(leds, STATS, _merged((PMZ, *PAR, "Candidate")), tol_ppm=6.0)
    assert sorted(t["why"]) == ["no_formula", "orphan_iso_child"]
    (tmp_path / "tables").mkdir()
    t.to_csv(tmp_path / "tables" / "unidentified.csv", index=False)
    assert PR.SECTIONS.index(PR.unidentified) == PR.SECTIONS.index(PR.reference_lists) + 1
    with PdfPages(tmp_path / "r.pdf") as pdf:
        PR.unidentified({"out_dir": str(tmp_path)}, pdf)
        assert pdf.get_pagecount() == 1
    assert PR._tried_summary(t.iloc[0]["readings_tried"]).startswith("no reading: ") or \
        PR._tried_summary(t.iloc[0]["readings_tried"]).startswith("C10H16O4")
    assert PR._tried_summary("[]") == "-"
    # a run without the table renders no page
    with PdfPages(tmp_path / "r2.pdf") as pdf:
        PR.unidentified({"out_dir": str(tmp_path / "none")}, pdf)
        assert pdf.get_pagecount() == 0
    # the single-sample workbook: the sheet follows "Below assignability" and holds the rows
    led = leds["f2"]
    single = UN.build_single(led, EDGE, sample_id="f2")
    assert sorted(single["why"]) == ["no_formula", "orphan_iso_child"]
    sheets = RP.build_sheets(led, unidentified=single)
    names = list(sheets)
    assert names.index(RP.UNIDENTIFIED_SHEET) == names.index("Below assignability") + 1
    assert len(sheets[RP.UNIDENTIFIED_SHEET]) == 2
    assert RP.UNIDENTIFIED_SHEET not in RP.build_sheets(led)
    RP.write_excel(led, tmp_path / "w.xlsx", sample_id="f2", unidentified=single)
    assert (tmp_path / "w.xlsx").stat().st_size > 0


def test_a_large_table_is_written_gzip(tmp_path, monkeypatch):
    t = pd.DataFrame({c: ["x" * 50] * 200 for c in UN.TABLE_COLUMNS})
    monkeypatch.setattr(UN, "GZIP_OVER_BYTES", 1000)
    p = UN.write(t, str(tmp_path))
    assert p.endswith("unidentified.csv.gz") and not (tmp_path / "unidentified.csv").exists()
    assert len(pd.read_csv(p)) == 200
    monkeypatch.setattr(UN, "GZIP_OVER_BYTES", 10 ** 9)
    assert UN.write(t, str(tmp_path)).endswith("unidentified.csv")


def test_a_dropped_row_the_per_file_rewrite_released_is_still_vote_dropped():
    xmz = 245.0301
    leds = {"f1": _ledger("f1", extra_m0=("f1x", xmz, "C9H10O7", "[M-H]-", "Assigned"))}
    drops = pd.DataFrame([dict(file="f1", peak_id="f1x", mz=xmz, neutral_formula="C9H10O7", adduct="[M-H]-",
                               tier_in="Assigned", reason="element-signature removal (C9H10O7 [M-H]-)")])
    L.clear_assignment(leds["f1"], "f1x", reason="batch element-signature removal")
    assert leds["f1"].loc[leds["f1"]["peak_id"] == "f1x", "role"].iloc[0] == L.ROLE_UNEXPLAINED
    t = UN.build(leds, STATS, _merged((PMZ, *PAR, "Assigned")), drops=drops, tol_ppm=6.0)
    assert len(t) == 1 and t.iloc[0]["why"] == "vote_dropped"


def test_with_the_votes_long_form_a_row_is_dropped_only_when_its_own_cluster_left():
    """A cluster's mean can sit more than the window from a member at the end of a chain: the
    long form (align's jitter) decides by the row's own cluster, not by distance."""
    x1, x2, x3 = 300.0000, 300.0017, 300.0034          # a 5.7 ppm chain: one cluster at 6 ppm
    leds = {"f1": _ledger("f1", extra_m0=("f1x", x1, "C14H28O5", "[M-H]-", "Assigned")),
            "f2": _ledger("f2", extra_m0=("f2x", x2, "C14H28O5", "[M-H]-", "Assigned")),
            "f3": _ledger("f3", extra_m0=("f3x", x3, "C14H28O5", "[M-H]-", "Assigned"))}
    for s, x in zip(("f1", "f2", "f3"), (x1, x2, x3)):
        leds[s].loc[leds[s]["peak_id"] == f"{s}x", "mz"] = x
    merged, jitter = AB.align({s: AB._m0(led) for s, led in leds.items()}, tol_ppm=6.0)
    row = merged[merged["neutral_formula"] == "C14H28O5"]
    assert len(row) == 1
    far = merged.copy()
    far.loc[row.index, "mz"] = x1 - 0.003           # the survivor's m/z > 6 ppm from every member
    d = UN.vote_drops(leds, None, far, UN.DropLog(), tol_ppm=6.0, jitter=jitter)
    assert len(d) == 0                              # its cluster's row survived
    assert len(UN.vote_drops(leds, None, far, UN.DropLog(), tol_ppm=6.0)) == 3   # the window rule alone
    log = UN.DropLog()
    after = merged.drop(index=row.index)
    log.note(merged, after, "reagent-cluster strip")
    d = UN.vote_drops(leds, None, after, log, tol_ppm=6.0, jitter=jitter)
    assert sorted(d["peak_id"]) == ["f1x", "f2x", "f3x"] and set(d["reason"]) == {"reagent-cluster strip (C14H28O5 [M-H]-)"}


# =========================================================================== repair after refute (2026-11-05)
def _rows_ledger(sid, rows):
    """A per-file ledger from (peak_id, mz, height, role, neutral, adduct, tier[, extra cols]) rows: M0 rows
    committed, the rest set by role (an iso_child names its parent by the `parent` extra), the exit
    recorder's columns stamped on unexplained rows; plus faint background rows."""
    base = [dict(peak_id=r[0], mz=r[1], height=r[2], area=r[2], **(r[7] if len(r) > 7 else {})) for r in rows]
    base += [dict(peak_id=f"{sid}n{k}", mz=150.0 + 7.3 * k, height=6.0, area=6.0) for k in range(5)]
    led = L.new_ledger(pd.DataFrame(base))
    for r in rows:
        pid, mz, h, role, n, a, t = r[:7]
        extra = r[7] if len(r) > 7 else {}
        if role == "M0":
            L.commit_assignment(led, pid, neutral_formula=n, adduct=a,
                                ion_formula=C.format_formula(EV.ion_composition(n, a, None)) + ("-" if a.endswith("-") else "+"),
                                ion_score=0.9, compound_score=0.9, ppm_error=0.1, pass_no=1, method="cheminfo",
                                confidence="High", commentary="Pass 1")
        elif role == "reagent":
            L.mark_reagent(led, pid, "reagent ion")
    led["tier"] = led["tier"].astype(object)
    for r in rows:
        if r[3] == "M0":
            led.loc[led["peak_id"] == r[0], "tier"] = r[6]
    for r in rows:
        extra = r[7] if len(r) > 7 else {}
        if r[3] == "iso_child":
            i = led.index[led["peak_id"] == r[0]][0]
            led.at[i, "role"] = "iso_child"
            led.at[i, "parent_peak_id"] = extra.get("parent")
            led.at[i, "iso_label"] = "13C"
    for c in ("parent_peak_id",):
        if c in led.columns:
            led[c] = led[c].astype(object)
    led[BL.READING] = pd.Series(pd.NA, index=led.index, dtype=object)
    led[BL.GATE] = pd.Series(pd.NA, index=led.index, dtype=object)
    led.loc[led["role"] == L.ROLE_UNEXPLAINED, BL.GATE] = BOX
    return led


def _ppm(x, p):
    return x * (1 + p * 1e-6)


# --------------------------------------------------------------------------- the bright threshold (5x the edge)
@pytest.mark.parametrize("x, listed", [(4.99, False), (5.0, True), (9.0, True)])
def test_the_bright_threshold_is_five_times_the_files_noise_edge(x, listed):
    leds = {"f1": _ledger("f1", unk_h=x * EDGE)}
    t = UN.build(leds, STATS, _merged((PMZ, *PAR, "Assigned")), tol_ppm=6.0)
    assert (len(t) == 1) is listed and UN.BRIGHT_X == 5.0


# --------------------------------------------------------------------------- the table's S/N bars
def test_the_tables_13c_count_and_line_states_need_their_snr():
    def one(snr, extra=()):
        rows = [("u", UNK, 4.0e4, "unexplained", None, None, None, {"signal_to_noise": snr})] + list(extra)
        return UN.build({"f1": _rows_ledger("f1", rows)}, STATS, _merged((PMZ, *PAR, "Assigned")), tol_ppm=6.0).iloc[0]
    # predicted one-carbon 13C line at S/N 100 x R13C = 1.1: untestable; at S/N 2000: 21.6 -> a measured 0
    assert pd.isna(one(100.0)["c13_count"]) and one(2000.0)["c13_count"] == pytest.approx(0.0)
    # a one-atom 34S line at S/N 100 x 0.0443 = 4.4 cannot be read absent; at S/N 1000 (44) it can
    assert one(100.0)["s34_line"] == "untestable" and one(1000.0)["s34_line"] == "absent"
    # a peak one 34S spacing above is 'present', whatever the S/N
    r = one(100.0, [("s", UNK + ISO.D_34S, 2.0e3, "unexplained", None, None, None)])
    # the 37Cl position is 1.25 mDa (4.3 ppm) above it, outside the 3-ppm probe: absent there (one 37Cl atom's
    # line at S/N 100 x 0.32 = 32 would show)
    assert r["s34_line"] == "present" and r["cl37_line"] == "absent"


# --------------------------------------------------------------------------- the orphan conditions
def test_each_orphan_condition():
    PM = ("C10H16O4", "[M-H]-")
    c_mz = PMZ + ISO.D_13C_EXACT
    merged = _merged((PMZ, *PM, "Assigned"))

    def table(rows):
        return UN.build({"f1": _rows_ledger("f1", rows)}, STATS, merged, tol_ppm=6.0)
    # a reagent parent explains its child
    t = table([("r", PMZ, 2e5, "reagent", None, None, None), ("c", c_mz, 2e4, "iso_child", None, None, None, {"parent": "r"})])
    assert len(t) == 0
    # no parent row in the file: an orphan
    t = table([("c", c_mz, 2e4, "iso_child", None, None, None, {"parent": "gone"})])
    assert list(t["why"]) == ["orphan_iso_child"] and "no parent row in the file" in t.iloc[0]["readings_tried"]
    # a parent that is itself an isotope child: an orphan
    t = table([("p", PMZ, 2e5, "M0", *PM, "Assigned"),
               ("c", c_mz, 2e4, "iso_child", None, None, None, {"parent": "p"}),
               ("cc", c_mz + ISO.D_13C_EXACT, 2e3, "iso_child", None, None, None, {"parent": "c"})])
    assert list(t["why"]) == ["orphan_iso_child"] and t.iloc[0]["mz"] == pytest.approx(c_mz + ISO.D_13C_EXACT, abs=1e-5)
    # the parent's merged row may sit anywhere in the merge window (here 4 ppm off): explained
    m4 = _merged((_ppm(PMZ, 4.0), *PM, "Assigned"))
    led = _rows_ledger("f1", [("p", PMZ, 2e5, "M0", *PM, "Assigned"),
                              ("c", c_mz, 2e4, "iso_child", None, None, None, {"parent": "p"})])
    assert len(UN.build({"f1": led}, STATS, m4, tol_ppm=6.0)) == 0


# --------------------------------------------------------------------------- the table's two windows
def test_the_hold_and_the_binning_windows_are_the_merge_window():
    # a line 4.5 ppm from a merged row of another file is held by it (one ion by the merge window)
    leds = {"f1": _ledger("f1", unk_h=4.0e3)}
    assert len(UN.build(leds, STATS, _merged((PMZ, *PAR, "Assigned"), (_ppm(UNK, 4.5), "C14H24O6", "[M-H]-", "Assigned")),
                        tol_ppm=6.0)) == 0
    # two lines 9 ppm apart in two files: two bins
    f2 = _ledger("f2", unk_h=4.0e3)
    f2.loc[f2["peak_id"] == "f2u", "mz"] = _ppm(UNK, 9.0)
    t = UN.build({"f1": _ledger("f1", unk_h=4.0e3), "f2": f2}, STATS, _merged((PMZ, *PAR, "Assigned")), tol_ppm=6.0)
    assert len(t) == 2


# --------------------------------------------------------------------------- F-M1-1: lines the merge absorbed
X_ = ("C10H16O4", "[M-H]-")
Y_ = ("C9H12O5", "[M-H]-")                    # another ion; placed 5 ppm above X_'s merged row


def test_a_line_resolved_from_a_merged_row_in_its_own_file_is_absorbed_by_merge():
    xm, ym = PMZ, _ppm(PMZ, 5.0)
    merged = _merged((xm, *X_, "Assigned"))
    # f1 holds both peaks (the merged ion's and a Candidate of another ion 5 ppm above); f2 the second
    # one only: the vote's cluster took both as X_, the line gets its own row with the merged row as partner
    leds = {"f1": _rows_ledger("f1", [("x", xm, 2e5, "M0", *X_, "Assigned"), ("y", ym, 3e3, "M0", *Y_, "Candidate")]),
            "f2": _rows_ledger("f2", [("y", ym, 3e3, "M0", *Y_, "Candidate")])}
    t = UN.build(leds, STATS, merged, tol_ppm=6.0)
    assert len(t) == 1
    r = t.iloc[0]
    assert r["why"] == "absorbed_by_merge" and r["roles"] == "M0:2" and r["mz"] == pytest.approx(ym, abs=1e-5)
    assert r["partner_mz"] == pytest.approx(xm, abs=1e-5) and r["partner_role"] == f"merged Assigned {X_[0]} {X_[1]}"
    tried = json.loads(r["readings_tried"])
    assert tried[0]["neutral"] == Y_[0] and tried[0]["blocked_by"].startswith(
        f"absorbed by the merge: the vote kept Assigned {X_[0]} {X_[1]} +5.0 ppm")
    # an unexplained line beside the merged ion's peak in the same file: the same
    leds = {"f1": _rows_ledger("f1", [("x", xm, 2e5, "M0", *X_, "Assigned"), ("u", ym, 3e3, "unexplained", None, None, None)])}
    t = UN.build(leds, STATS, merged, tol_ppm=6.0)
    assert list(t["why"]) == ["absorbed_by_merge"] and t.iloc[0]["roles"] == "unexplained:1"
    # no file holds both peaks: the line is the merged ion by the merge window -- held, no row
    leds = {"f1": _rows_ledger("f1", [("x", xm, 2e5, "M0", *X_, "Assigned")]),
            "f2": _rows_ledger("f2", [("u", ym, 3e3, "unexplained", None, None, None)])}
    assert len(UN.build(leds, STATS, merged, tol_ppm=6.0)) == 0
    # a vote loser on the merged ion's own peak (another reading, one pick): not absorbed
    leds = {"f1": _rows_ledger("f1", [("x", xm, 2e5, "M0", *X_, "Assigned")]),
            "f2": _rows_ledger("f2", [("y", xm, 3e3, "M0", *Y_, "Candidate")])}
    assert len(UN.build(leds, STATS, merged, tol_ppm=6.0)) == 0
    # the same ion read another way beside it is the merged ion: not absorbed
    leds = {"f1": _rows_ledger("f1", [("x", xm, 2e5, "M0", *X_, "Assigned"), ("y", ym, 3e3, "M0", *X_, "Candidate")])}
    assert len(UN.build(leds, STATS, merged, tol_ppm=6.0)) == 0


# --------------------------------------------------------------------------- inside a batch: the wiring
def test_the_batch_lists_the_vote_drops_of_both_merge_gates(tmp_path, monkeypatch):
    """The isotopologue strip and the element-signature removal of the two-gate batch
    (tests/test_isotopologue_rows.py): both readings are in vote_drops.csv with their gate and
    reading, and their released per-file rows are unidentified rows why = vote_dropped."""
    from tests.test_isotopologue_rows import R, test_batch_parent_removed_by_the_element_signature_gate
    test_batch_parent_removed_by_the_element_signature_gate(tmp_path, monkeypatch)
    d = tmp_path / "a"
    drops = pd.read_csv(d / "tables" / "vote_drops.csv")
    reasons = set(drops["reason"])
    assert "element-signature removal (C8H13BrO4 [M-H]-)" in reasons
    assert f"isotopologue strip ({R[0]} {R[1]})" in reasons
    summ = json.load(open(d / "batch_summary.json"))
    assert summ["unidentified"]["vote_drops"] == len(drops) >= 4
    t = pd.read_csv(d / "tables" / "unidentified.csv")
    pb = t[(t["mz"] - C.ion_mz("C8H13BrO4", "[M-H]-")).abs() < 1e-3]
    assert len(pb) == 1 and pb.iloc[0]["why"] == "vote_dropped"
    assert "element-signature removal" in pb.iloc[0]["readings_tried"]


def test_the_batch_decides_drops_by_the_rows_own_cluster(tmp_path, monkeypatch):
    """A 5-file single-linkage chain (5.5 ppm steps: one cluster, its mean 11 ppm from the ends):
    nothing is a drop, though the window rule alone would list the ends."""
    ym = 300.0
    leds = {}
    for k in range(5):
        leds[f"u{k:02d}"] = _ledger(f"u{k:02d}", extra_m0=(f"u{k:02d}y", _ppm(ym, 5.5 * k), "C14H28O5", "[M-H]-", "Assigned"))
        leds[f"u{k:02d}"].loc[leds[f"u{k:02d}"]["peak_id"] == f"u{k:02d}y", "mz"] = _ppm(ym, 5.5 * k)
    summ = _run(tmp_path, monkeypatch, leds)
    tol = summ["mass_scale"]["merge_ppm"]
    jit = pd.read_csv(tmp_path / "tables" / "jitter.csv")
    jy = jit[jit["neutral_formula"] == "C14H28O5"]
    assert jy["cluster"].nunique() == 1 and len(jy) == 5 and tol < 11.0
    drops = pd.read_csv(tmp_path / "tables" / "vote_drops.csv")
    assert not (drops["neutral_formula"] == "C14H28O5").any()


def test_the_batch_writes_the_per_file_guards_note_on_the_merged_row(tmp_path, monkeypatch):
    fact = "13C reads 6.0 C for 10 (tol 2.5, per-file 13C, 1 files)"
    leds = {}
    for s in ("u00", "u01", "u02"):
        led = _ledger(s)
        led.loc[led["peak_id"] == f"{s}p", ["tier", "tier_reason"]] = ["Candidate", f"isotope contradiction: {fact}"]
        leds[s] = led
    summ = _run(tmp_path, monkeypatch, leds)
    merged = pd.read_csv(tmp_path / "merged_ledger.csv")
    r = merged[merged["neutral_formula"] == PAR[0]].iloc[0]
    assert r["tier"] == "Candidate" and f"isotope contradiction in 3 of 3 file(s): {fact}" in str(r["tier_reason"])
    assert summ["merge_gates"]["isotope_guard"]["per_file_noted"] == 1
