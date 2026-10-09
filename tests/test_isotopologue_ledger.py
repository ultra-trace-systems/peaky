"""The isotopologue gate's decision, recorded (`iso_checks.reconcile_per_file`,
`iso_checks.parent_lines`).

A merged row the gate reads as another merged ion's isotopologue leaves the merged
ledger. The line is still a valid assignment -- the parent's isotopologue -- so the
per-file ledgers that committed the stripped reading as an M0 record it: the row
becomes the parent's iso_child where the file commits the parent, and is released
to unexplained (with a note naming the parent and the line) where it does not or
where the parent itself left the merged ledger afterwards. The parent's merged row
lists the lines it was given (`isotopologue_lines`).

All ledgers and series here are synthetic: invented readings, exact isotope ratios.

Run: pytest tests/test_isotopologue_ledger.py -q
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import pytest

from peaky.assignment import ledger as L
from peaky.batch import iso_checks as IC
from peaky.chem import chemistry as C

from test_isotopologue_rows import ORBI, P, R, SCALE, X, _case, _gate, _keys, _line, quiet

MARK = IC.SAT_LEDGER_MARK


def _ionf(n, a):
    from peaky.assignment import tiers as T
    return C.format_formula(T._ion_counts(n, a)) + a[-1]


def _ledger(rows, tag="f"):
    """rows: [(peak suffix, mz, reading or None)] -> a ledger, each reading committed
    as an Assigned M0 (unexplained where None)."""
    led = L.new_ledger(pd.DataFrame([(f"{tag}_{k}", float(mz), 1e4) for k, mz, _r in rows],
                                    columns=["peak_id", "mz", "height"]))
    for k, _mz, rd in rows:
        if rd is not None:
            L.commit_assignment(led, f"{tag}_{k}", neutral_formula=rd[0], adduct=rd[1], ion_formula=_ionf(*rd),
                                ion_score=0.9, compound_score=0.9, ppm_error=0.1, pass_no=1,
                                method="cheminfo+grid", confidence="High", commentary="stub")
    led["tier"] = np.where(led["role"] == L.ROLE_M0, "Assigned", None)
    led["evidence_level"] = np.where(led["role"] == L.ROLE_M0, "4a", None)
    return led


def _stripped_case():
    merged, ts, lmz, e = _case(np.ones(9))
    kept, tab, summ = _gate(merged, ts)
    assert summ["n_stripped"] == 1
    return kept, tab, lmz


def _row(led, mz):
    return led.loc[np.isclose(led["mz"], mz)].iloc[0]


# --------------------------------------------------------------------------- (a) committed parent
def test_committed_reading_becomes_the_parents_iso_child():
    kept, tab, lmz = _stripped_case()
    pmz = C.ion_mz(*P)
    led = {"f1": _ledger([("p", pmz, P), ("r", lmz, R), ("x", C.ion_mz(*X), X)], "f1"),
           "f2": _ledger([("p", pmz, P), ("r", lmz, R)], "f2")}
    # f3 already reads the line as P's 13C child: left as it is
    f3 = _ledger([("p", pmz, P), ("r", lmz, None)], "f3")
    L.attach_isotopologue(f3, "f3_r", "f3_p", iso_label="13C")
    led["f3"] = f3
    before3 = f3.copy()
    changed, summ = IC.reconcile_per_file(led, tab, mass_scale=SCALE, log=quiet)
    assert changed == {"f1", "f2"} and summ["iso_child"] == 2 and summ["released"] == 0, summ
    assert summ["files"] == {"f1": {"iso_child": 1, "reagent": 0, "released": 0},
                             "f2": {"iso_child": 1, "reagent": 0, "released": 0}}
    for sid in ("f1", "f2"):
        r = _row(led[sid], lmz)
        assert r["role"] == L.ROLE_ISO and r["parent_peak_id"] == f"{sid}_p" and r["iso_label"] == "13C"
        assert (r["parent_neutral_formula"], r["parent_adduct"]) == P
        assert pd.isna(r["neutral_formula"]) and pd.isna(r["tier"]) and pd.isna(r["evidence_level"])
        c = str(r["commentary"])
        assert c.startswith(f"{MARK}: the 13C line of C8H12NO10- at m/z ") and f"(was {R[0]} {R[1]}, Assigned;" in c
        assert c.endswith("Was: stub"), c
        # the child's isotope fit is the gate's own agreement with the parent's prediction
        rho = float(tab["rho_area"].iloc[0])
        assert r["iso_match_score"] == pytest.approx(min(rho, 1.0 / rho))
        assert not L.validate(led[sid])
        assert _row(led[sid], pmz)["role"] == L.ROLE_M0
    pd.testing.assert_frame_equal(led["f3"], before3)
    # the role counts a recount reads off the rewritten ledger
    st = L.stats(led["f1"])
    assert st["by_role"][L.ROLE_M0] == 2 and st["by_role"][L.ROLE_ISO] == 1
    # idempotent: a second pass finds no M0 on the line
    assert IC.reconcile_per_file(led, tab, mass_scale=SCALE, log=quiet)[0] == set()


def test_the_rows_own_children_follow_it_and_a_lock_is_kept():
    kept, tab, lmz = _stripped_case()
    pmz = C.ion_mz(*P)
    led = _ledger([("p", pmz, P), ("r", lmz, R), ("r13", lmz + 1.00335, None)], "f1")
    L.attach_isotopologue(led, "f1_r13", "f1_r", iso_label="13C")
    L.lock_peaks(led, ["f1_r"])
    IC.reconcile_per_file({"f1": led}, tab, mass_scale=SCALE, log=quiet)
    r, g = _row(led, lmz), _row(led, lmz + 1.00335)
    assert r["role"] == L.ROLE_ISO and bool(r["locked"])
    assert g["role"] == L.ROLE_ISO and g["parent_peak_id"] == "f1_p" and g["iso_label"] == "13C"
    assert not L.validate(led)


def test_parent_lines_column_on_the_parent_row():
    kept, tab, lmz = _stripped_case()
    out = IC.parent_lines(kept, tab)
    assert list(out.columns) == list(kept.columns) + [IC.SAT_LINES_COLUMN] and len(out) == len(kept)
    p = out.loc[out["neutral_formula"] == P[0], IC.SAT_LINES_COLUMN].iloc[0]
    rho = float(tab["rho_area"].iloc[0])
    assert p == f"13C {lmz:.4f} (area x{rho:.2f} of predicted, 9 spectra; was {R[0]} {R[1]})", p
    assert (out.loc[out["neutral_formula"] != P[0], IC.SAT_LINES_COLUMN] == "").all()


def test_two_lines_of_one_parent_are_joined_in_mz_order():
    l1, e1 = _line(P, "13C")
    l2, e2 = _line(P, "18O")
    pmz = C.ion_mz(*P)
    from test_isotopologue_rows import _merged, _series, _wave
    r2 = ("C13H10O6", "[M-H]-")
    merged = _merged([(pmz, P, "Assigned"), (l1, R, "Assigned"), (l2, r2, "Candidate")])
    ap = _wave()
    ts = _series([(pmz, ap), (l1, e1 * ap), (l2, e2 * ap)])
    kept, tab, summ = _gate(merged, ts)
    assert summ["n_stripped"] == 2
    # '; '-joined entries, each closing on ')': split after the parenthesis
    lines = IC.parent_lines(kept, tab)[IC.SAT_LINES_COLUMN].iloc[0].split("); ")
    assert len(lines) == 2 and lines[1].endswith(f"was {r2[0]} {r2[1]})"), lines
    assert [x.split()[:2] for x in lines] == [["13C", f"{l1:.4f}"], ["18O", f"{l2:.4f}"]], lines


# --------------------------------------------------------------------------- (b) parent not committed
def test_parent_not_committed_in_a_file_releases_the_row():
    kept, tab, lmz = _stripped_case()
    pmz = C.ion_mz(*P)
    led = {"f1": _ledger([("p", pmz, None), ("r", lmz, R), ("r13", lmz + 1.00335, None)], "f1")}
    L.attach_isotopologue(led["f1"], "f1_r13", "f1_r", iso_label="13C")
    changed, summ = IC.reconcile_per_file(led, tab, mass_scale=SCALE, log=quiet)
    # the row's own line goes with it, and says so
    g = _row(led["f1"], lmz + 1.00335)
    assert g["role"] == L.ROLE_UNEXPLAINED
    assert str(g["commentary"]) == f"CLEARED ({MARK}: a line of {R[0]} {R[1]}, released with it).", g["commentary"]
    assert changed == {"f1"} and summ["released"] == 1 and summ["iso_child"] == 0
    r = _row(led["f1"], lmz)
    assert r["role"] == L.ROLE_UNEXPLAINED and pd.isna(r["neutral_formula"]) and pd.isna(r["evidence_level"])
    c = str(r["commentary"])
    assert c.startswith(f"CLEARED ({MARK}: the 13C line of C8H12NO10- at m/z {pmz:.4f}, which this file does "
                        "not commit;"), c
    assert f"was {R[0]} {R[1]}, Assigned" in c and c.endswith("Was: stub")


def test_reagent_parent_marks_a_reagent_isotopologue():
    from test_isotopologue_rows import _merged, _series
    from peaky.chem.resolution import Resolution
    rg_mz = C.ion_mz("HNO3", "[M-H]-")
    lines = IC._parent_lines({"N": 1, "O": 3}, Resolution.from_dict(ORBI).fwhm(rg_mz + 1))
    sh, e, tags = next(x for x in lines if IC._sat_label(x[2]) == "18O")
    ap = 1e7 * (1.5 + np.sin(np.arange(9) / 2.0))
    on = ("CH3O3", "[M-H]-")
    merged = _merged([(rg_mz + sh, on, "Candidate"), (C.ion_mz(*X), X, "Assigned")])
    ts = _series([(rg_mz, ap), (rg_mz + sh, e * ap), (C.ion_mz(*X), 3e4)])
    reag = pd.DataFrame({"mz": [rg_mz], "role": "reagent", "ion_formula": ["NO3-"], "iso_label": [None]})
    kept, tab, summ = _gate(merged, ts, reagents=reag)
    assert summ["n_stripped"] == 1
    led = _ledger([("g", rg_mz, None), ("r", rg_mz + sh, on)], "f1")
    L.mark_reagent(led, "f1_g", "reagent ion: NO3- (+0.1 ppm)", ion_formula="NO3-")
    no_reagent = _ledger([("r", rg_mz + sh, on)], "f2")
    changed, s2 = IC.reconcile_per_file({"f1": led, "f2": no_reagent}, tab, mass_scale=SCALE, log=quiet)
    assert changed == {"f1", "f2"} and s2["reagent"] == 1 and s2["released"] == 1, s2
    r = _row(led, rg_mz + sh)
    assert r["role"] == L.ROLE_REAGENT and r["ion_formula"] == "NO3-"
    assert str(r["commentary"]).startswith(f"reagent isotopologue: 18O of NO3- (18O); {MARK}")
    from peaky.batch import timeseries as TSI
    ids = TSI.identified_rows(led)
    assert ids.loc[np.isclose(ids["mz"], rg_mz + sh), "iso_label"].iloc[0] == "18O"
    assert _row(no_reagent, rg_mz + sh)["role"] == L.ROLE_UNEXPLAINED
    # a reagent parent has no merged row: no column entry anywhere
    assert (IC.parent_lines(kept, tab)[IC.SAT_LINES_COLUMN] == "").all()


# --------------------------------------------------------------------------- (c) parent removed
def test_parent_removed_releases_the_row():
    kept, tab, lmz = _stripped_case()
    out, n = IC.parent_removed(tab, [{"neutral_formula": P[0], "adduct": P[1]}], log=quiet)
    assert n == 1
    pmz = C.ion_mz(*P)
    led = {"f1": _ledger([("p", pmz, P), ("r", lmz, R)], "f1")}
    changed, summ = IC.reconcile_per_file(led, out, mass_scale=SCALE, log=quiet)
    assert summ["released"] == 1 and summ["released_parent_removed"] == 1 and summ["iso_child"] == 0
    r = _row(led["f1"], lmz)
    assert r["role"] == L.ROLE_UNEXPLAINED
    assert f"whose reading {P[0]} {P[1]} then left the merged ledger" in str(r["commentary"])
    # the removed parent's merged row is gone, and a 'parent removed' row names no line
    assert (IC.parent_lines(kept, out)[IC.SAT_LINES_COLUMN] == "").all()


# --------------------------------------------------------------------------- (d) mixed / exempt
@pytest.mark.parametrize("kind", ["mixed", "exempt"])
def test_mixed_or_exempt_lines_are_untouched(kind):
    if kind == "mixed":
        merged, ts, lmz, e = _case(np.full(9, 2.5))
        kept, tab, summ = _gate(merged, ts)
    else:
        merged, ts, lmz, e = _case(np.ones(9))
        kept, tab, summ = _gate(merged, ts, exempt=frozenset({R[0]}))
    assert tab["verdict"].tolist() == [kind] and R in _keys(kept)
    pmz = C.ion_mz(*P)
    led = {"f1": _ledger([("p", pmz, P), ("r", lmz, R)], "f1")}
    before = led["f1"].copy()
    changed, s = IC.reconcile_per_file(led, tab, mass_scale=SCALE, log=quiet)
    assert changed == set() and s["iso_child"] == s["released"] == 0
    pd.testing.assert_frame_equal(led["f1"], before)
    assert (IC.parent_lines(kept, tab)[IC.SAT_LINES_COLUMN] == "").all()


# --------------------------------------------------------------------------- (e) TOF / skipped
@pytest.mark.parametrize("kw", [{"klass": "tof"}, {"resolution": None}])
def test_skipped_gate_rewrites_nothing(kw):
    merged, ts, lmz, e = _case(np.ones(9))
    kept, tab, summ = _gate(merged, ts, **kw)
    assert not summ["ran"] and not len(tab)
    pmz = C.ion_mz(*P)
    led = {"f1": _ledger([("p", pmz, P), ("r", lmz, R)], "f1")}
    before = led["f1"].copy()
    assert IC.reconcile_per_file(led, tab, mass_scale=SCALE, log=quiet)[0] == set()
    pd.testing.assert_frame_equal(led["f1"], before)
    out = IC.parent_lines(kept, tab)
    assert IC.SAT_LINES_COLUMN in out.columns and (out[IC.SAT_LINES_COLUMN] == "").all()


# --------------------------------------------------------------------------- through the batch
def _batch(tmp_path, monkeypatch, plan, *, rows=14, parent=P, **kw):
    """assign_batch.run over three files s01 / s05 / s09; plan: {sid: (P committed,
    R committed)}; a file not committing R leaves its line unexplained, one not
    committing P leaves the parent's peak unexplained."""
    from peaky.assignment import assign as A_
    from peaky.assignment import tiers as T
    from peaky.batch import assign_batch as AB
    from peaky.chem.resolution import Resolution
    from peaky.io import io_mascope as IO

    pmz = C.ion_mz(*parent)
    lmz, e = _line(parent, "13C")
    xmz = C.ion_mz(*X)
    xl, ex = _line(X, "13C")
    t0 = pd.Timestamp("2021-02-18 00:00", tz="UTC")
    recs = []
    for i in range(rows):
        w = 1e5 * (1.5 + np.sin(i / 2.0))
        lines = [(pmz, w), (lmz, e * w * (1.0 + 0.05 * np.cos(i))), (xmz, 4e4), (xl, ex * 4e4)] + \
                [(60.0 + 3.7 * j, 50.0) for j in range(80)]
        recs += [dict(sample_item_id=f"s{i:02d}", sample_item_name=f"n{i:02d}",
                      datetime_utc=t0 + pd.Timedelta(minutes=10 * i), peak_id=f"s{i:02d}_{k}", mz=float(m),
                      height=0.75 * float(h), area=float(h)) for k, (m, h) in enumerate(lines)]
    pk = pd.DataFrame(recs)

    def fake_assign(sid, context="ambient-air", **kw_):
        has_p, has_r = plan[sid]
        led = L.new_ledger(pd.DataFrame([("p1", pmz, 1e5), ("p2", lmz, 1e4), ("p3", xmz, 3e4)],
                                        columns=["peak_id", "mz", "height"]))
        for pid, rd, on in (("p1", parent, has_p), ("p2", R, has_r), ("p3", X, True)):
            if on:
                L.commit_assignment(led, pid, neutral_formula=rd[0], adduct=rd[1], ion_formula=_ionf(*rd),
                                    ion_score=0.9, compound_score=0.9, ppm_error=0.1, pass_no=1,
                                    method="cheminfo+grid", confidence="High", commentary="stub")
        if has_p and not has_r:
            L.attach_isotopologue(led, "p2", "p1", iso_label="13C")
        T.apply_tiers(led)
        led.loc[led["role"] == L.ROLE_M0, "tier"] = T.TIER_ASSIGNED
        return {"ledger": led, "stats": {"noise_edge_cps": 50.0, "height_gate_cps": 50.0, **L.stats(led)},
                "plausibility_audit": [], "summaries": {}, "problems": []}

    monkeypatch.setattr(IO, "connect", lambda *a, **k: "CLIENT")
    monkeypatch.setattr(IO, "fetch_peaks", lambda client, sid, use_cache=True: pk[pk["sample_item_id"] == sid]
                        [["peak_id", "mz", "height"]].reset_index(drop=True))
    monkeypatch.setattr(IO, "estimate_offset", lambda raw: 0.0)
    monkeypatch.setattr(A_, "run", fake_assign)
    d = tmp_path / "run"
    AB.run(peaks=pk, ts_peaks=pk, reagent="NO3", batch="test batch", out_dir=str(d), sample_ids=sorted(plan),
           n_jobs=1, resolving_power=Resolution.from_dict(ORBI), mass_axis="off", residual=False, log=quiet, **kw)
    summ = json.load(open(os.path.join(d, "batch_summary.json")))
    merged = pd.read_csv(os.path.join(d, "merged_ledger.csv"))
    per = {sid: pd.read_csv(os.path.join(d, "per_file", f"{sid}_ledger.csv")) for sid in plan}
    lev = pd.read_csv(os.path.join(d, "tables", "evidence_levels.csv"))
    return summ, merged, per, lev, lmz, pmz


def _roles(led):
    return led["role"].value_counts().to_dict()


def test_batch_records_the_line_in_the_per_file_ledgers(tmp_path, monkeypatch):
    """(a) + (b) through assign_batch.run: R committed in s01 and s05 (P too), s09
    reads the line as P's child already, s13 commits R but not P. R leaves the merged
    ledger; s01 / s05 now read it as P's 13C child, s13 releases it, s09 is
    untouched; each file's role counts in batch_summary match its rewritten ledger;
    the merged P row lists the line; R's pair is no longer levelled."""
    plan = {"s01": (True, True), "s05": (True, True), "s09": (True, False), "s13": (False, True)}
    summ, merged, per, lev, lmz, pmz = _batch(tmp_path, monkeypatch, plan)
    g = summ["merge_gates"]["isotopologue"]
    assert g["n_stripped"] == 1 and R not in _keys(merged), g
    assert g["per_file"]["iso_child"] == 2 and g["per_file"]["released"] == 1, g["per_file"]
    for sid in ("s01", "s05"):
        r = per[sid].loc[np.isclose(per[sid]["mz"], lmz)].iloc[0]
        assert r["role"] == "iso_child" and r["parent_peak_id"] == "p1" and r["iso_label"] == "13C"
        assert r["parent_neutral_formula"] == P[0] and str(r["commentary"]).startswith(MARK)
    r = per["s13"].loc[np.isclose(per["s13"]["mz"], lmz)].iloc[0]
    assert r["role"] == "unexplained" and "which this file does not commit" in str(r["commentary"])
    r = per["s09"].loc[np.isclose(per["s09"]["mz"], lmz)].iloc[0]
    assert r["role"] == "iso_child" and str(r["commentary"]) in ("nan", "")
    # no per-file ledger holds R's reading any more
    assert not any(((p["neutral_formula"] == R[0]) & (p["role"] == "M0")).any() for p in per.values())
    # the per-file role counts are the rewritten ledgers'
    for st in summ["per_file"]:
        led = per[st["sample_id"]]
        assert {k: v for k, v in st["by_role"].items() if v} == _roles(led), st["sample_id"]
        assert st["n_M0"] == int((led["role"] == "M0").sum())
    assert summ["per_file"][0]["isotopologue_gate"] == {"iso_child": 1, "reagent": 0, "released": 0}
    # the merged view
    lines = merged.set_index("neutral_formula")[IC.SAT_LINES_COLUMN]
    assert str(lines[P[0]]).startswith(f"13C {lmz:.4f} (area x") and f"was {R[0]} {R[1]})" in str(lines[P[0]])
    assert lines.drop(P[0]).isna().all()
    # R's pair is no longer levelled (the SAT veto row reads no pooled pair)
    assert not ((lev["neutral_formula"] == R[0]) & (lev["adduct"] == R[1])).any()


def test_batch_without_the_gate_rewrites_nothing(tmp_path, monkeypatch):
    """(e) --no-isotopologue-gate: R stays in the merged ledger and in every per-file
    ledger, the column exists and is empty."""
    plan = {"s01": (True, True), "s05": (True, True), "s09": (True, False)}
    summ, merged, per, lev, lmz, pmz = _batch(tmp_path, monkeypatch, plan, isotopologue_rows=False)
    assert R in _keys(merged) and IC.SAT_LINES_COLUMN in merged.columns
    assert merged[IC.SAT_LINES_COLUMN].isna().all()
    for sid in ("s01", "s05"):
        assert per[sid].loc[np.isclose(per[sid]["mz"], lmz), "role"].iloc[0] == "M0"
    assert "per_file" not in summ["merge_gates"]["isotopologue"] or \
        summ["merge_gates"]["isotopologue"]["per_file"]["iso_child"] == 0
    assert all("isotopologue_gate" not in st for st in summ["per_file"])


def test_a_rewritten_files_resolvability_counts_are_recounted():
    """The batch recounts a rewritten file's stats from its ledger: the M0 rows'
    resolvability classes too, so a stripped reading's class leaves the counts."""
    from peaky.batch import assign_batch as AB
    led = pd.DataFrame(dict(peak_id=["a", "b", "c"], mz=[100.0, 101.0, 102.0], height=[1.0, 1.0, 1.0],
                            role=[L.ROLE_M0, L.ROLE_M0, L.ROLE_ISO],
                            resolvability=["isolated", "blended", pd.NA]))
    st = {"resolvability": {"isolated": 1, "blended": 1, "resolved": 1}, "n_M0": 3}
    AB._recount_roles(st, led)
    assert st["resolvability"] == {"isolated": 1, "blended": 1} and st["n_M0"] == 2
    # a file whose run stamped no resolvability keeps none
    st2 = {"n_M0": 3}
    AB._recount_roles(st2, led)
    assert "resolvability" not in st2


# --------------------------------------------------------------------------- (f) element-signature removal
BR = ("C6H9BrO3", "[M-H]-")                          # an invented bromine reading the batch refutes
BR_PAIR = {"neutral_formula": BR[0], "adduct": BR[1], "mz": C.ion_mz(*BR), "tier": "Assigned",
           "note": "the M+2 (81Br) line absent in 9 of 9 detectable spectra"}


def test_signature_removal_releases_the_reading_in_every_file():
    """Every M0 row committing a removed reading is released, whatever its m/z (the batch
    refuted the reading, not one file's line): its own children with it, a lock kept, its
    per-reading columns emptied, the refuted line named. Other readings, and the removed
    reading's line where a file reads it otherwise, are untouched; a second pass finds nothing."""
    bmz, xmz = C.ion_mz(*BR), C.ion_mz(*X)
    f1 = _ledger([("b", bmz, BR), ("b13", bmz + 1.00335, None), ("x", xmz, X)], "f1")
    L.attach_isotopologue(f1, "f1_b13", "f1_b", iso_label="13C")
    L.lock_peaks(f1, ["f1_b"])
    f2 = _ledger([("b", bmz * (1 + 4e-6), BR), ("x", xmz, X)], "f2")       # its own m/z, 4 ppm off
    f3 = _ledger([("b", bmz, None), ("x", xmz, X)], "f3")                   # the line unexplained
    led = {"f1": f1, "f2": f2, "f3": f3}
    before3 = f3.copy()
    changed, summ = IC.release_signature_removed(led, [BR_PAIR], log=quiet)
    assert changed == {"f1", "f2"} and summ == {"released": 2, "children": 1,
                                                 "files": {"f1": {"released": 1, "children": 1},
                                                           "f2": {"released": 1, "children": 0}}}, summ
    for sid in ("f1", "f2"):
        r = led[sid].loc[led[sid]["peak_id"] == f"{sid}_b"].iloc[0]
        assert r["role"] == L.ROLE_UNEXPLAINED and pd.isna(r["neutral_formula"]) and pd.isna(r["evidence_level"])
        c = str(r["commentary"])
        assert c.startswith(f"CLEARED ({IC.SIG_LEDGER_MARK}: the batch's REQ check refutes the reading on its "
                            "element-signature line (the M+2 (81Br) line absent"), c
        assert f"(was {BR[0]} {BR[1]}, Assigned))" in c and c.endswith("Was: stub"), c
        assert _row(led[sid], xmz)["role"] == L.ROLE_M0
        assert not L.validate(led[sid])
    assert bool(_row(f1, bmz)["locked"])
    g = _row(f1, bmz + 1.00335)
    assert g["role"] == L.ROLE_UNEXPLAINED and pd.isna(g["parent_peak_id"])
    assert str(g["commentary"]) == (f"CLEARED ({IC.SIG_LEDGER_MARK}: a line of the refuted reading {BR[0]} {BR[1]}, "
                                    "released with it)."), g["commentary"]
    pd.testing.assert_frame_equal(f3, before3)
    assert IC.release_signature_removed(led, [BR_PAIR], log=quiet)[0] == set()


@pytest.mark.parametrize("pairs", [None, []])
def test_no_removal_releases_nothing(pairs):
    led = {"f1": _ledger([("b", C.ion_mz(*BR), BR)], "f1")}
    before = led["f1"].copy()
    changed, summ = IC.release_signature_removed(led, pairs, log=quiet)
    assert changed == set() and summ == {"released": 0, "children": 0, "files": {}}
    pd.testing.assert_frame_equal(led["f1"], before)


def test_a_released_files_stats_record_the_release():
    from peaky.batch import assign_batch as AB
    led = _ledger([("b", C.ion_mz(*BR), BR), ("x", C.ion_mz(*X), X)], "f1")
    IC.release_signature_removed({"f1": led}, [BR_PAIR], log=quiet)
    st = {"n_M0": 2}
    AB._recount_roles(st, led, signature={"released": 1, "children": 0})
    assert st["n_M0"] == 1 and st["element_signature_gate"] == {"released": 1, "children": 0}
    assert "isotopologue_gate" not in st


def test_the_same_neutral_on_another_adduct_is_kept():
    """The batch refuted the (neutral, adduct) pair, not the neutral."""
    other = (BR[0], "[M+NO3]-")
    led = {"f1": _ledger([("b", C.ion_mz(*BR), BR), ("o", C.ion_mz(*other), other)], "f1")}
    changed, summ = IC.release_signature_removed(led, [BR_PAIR], log=quiet)
    assert summ["released"] == 1
    assert _row(led["f1"], C.ion_mz(*other))["role"] == L.ROLE_M0


def test_a_reagent_row_carrying_the_pair_is_left_alone():
    """mark_reagent keeps a row's neutral / adduct (cleanup's reagent-precursor relabel marks an
    M0 reagent without clearing it), so a reagent row can carry the removed pair: it stays
    reagent, and the release does not raise."""
    bmz = C.ion_mz(*BR)
    f1 = _ledger([("b", bmz, BR), ("r", bmz + 0.5, BR)], "f1")
    L.mark_reagent(f1, "f1_r", "reagent precursor: invented")
    changed, summ = IC.release_signature_removed({"f1": f1}, [BR_PAIR], log=quiet)
    assert summ["released"] == 1
    assert f1.loc[f1["peak_id"] == "f1_r", "role"].iloc[0] == L.ROLE_REAGENT


def test_every_per_reading_column_is_emptied():
    bmz = C.ion_mz(*BR)
    f1 = _ledger([("b", bmz, BR)], "f1")
    for c in IC._READING_COLUMNS:
        if c not in f1.columns:
            f1[c] = pd.NA
        f1[c] = f1[c].astype(object)
    f1.loc[f1["peak_id"] == "f1_b", list(IC._READING_COLUMNS)] = "x"
    IC.release_signature_removed({"f1": f1}, [BR_PAIR], log=quiet)
    r = f1.loc[f1["peak_id"] == "f1_b"].iloc[0]
    left = [c for c in IC._READING_COLUMNS if pd.notna(r[c])]
    assert not left, left


def test_a_ledger_problem_after_the_release_is_reported():
    """The summary's `problems` is the only signal of a broken ledger: a stale parent stamp on
    another reading's child is reported for the file the release touched."""
    bmz, xmz = C.ion_mz(*BR), C.ion_mz(*X)
    f1 = _ledger([("b", bmz, BR), ("x", xmz, X), ("k", xmz + 1.00335, None)], "f1")
    L.attach_isotopologue(f1, "f1_k", "f1_x", iso_label="13C")
    f1.loc[f1["peak_id"] == "f1_k", "parent_neutral_formula"] = "C1H4"     # a stale stamp
    changed, summ = IC.release_signature_removed({"f1": f1}, [BR_PAIR], log=quiet)
    assert changed == {"f1"} and "problems" in summ and "f1" in summ["problems"], summ


def test_a_reading_another_merged_row_still_holds_is_not_gone():
    """A reading can hold two merged rows; the one a known-species decision marks stays. The
    other row leaves, but the reading has not left the merged ledger: the pair reads
    reading_left False, and neither the per-file release nor parent_removed acts on it."""
    t = pd.DataFrame([dict(neutral_formula=BR[0], adduct=BR[1], check="REQ", instrument="orbitrap", veto=True,
                           verdict="absent", line="M+2 (81Br)", note=BR_PAIR["note"])])
    for c in IC.TABLE_COLUMNS:
        if c not in t.columns:
            t[c] = np.nan
    bmz = C.ion_mz(*BR)
    merged = pd.DataFrame({"neutral_formula": [BR[0], BR[0], X[0]], "adduct": [BR[1], BR[1], X[1]],
                           "mz": [bmz, bmz * (1 + 9e-6), C.ion_mz(*X)], "tier": "Assigned",
                           "tier_reason": [f"{IC.KNOWN_LOCK_MARK} by 2 files", "", ""]})
    out, s = IC.remove_signature_vetoed(merged, t[list(IC.TABLE_COLUMNS)], klass="orbitrap", log=quiet)
    assert s["removed"] == 1 and len(out) == 2 and s["pairs"][0]["reading_left"] is False, s
    assert IC.gone_pairs(s["pairs"]) == []
    led = {"f1": _ledger([("b", bmz, BR)], "f1")}
    before = led["f1"].copy()
    assert IC.release_signature_removed(led, s["pairs"], log=quiet)[0] == set()
    pd.testing.assert_frame_equal(led["f1"], before)
    kept, tab, lmz = _stripped_case()
    assert IC.parent_removed(tab, [{"neutral_formula": P[0], "adduct": P[1], "reading_left": False}],
                             log=quiet)[1] == 0
    # ... and both drop when the only row goes
    out, s = IC.remove_signature_vetoed(merged.iloc[1:], t[list(IC.TABLE_COLUMNS)], klass="orbitrap", log=quiet)
    assert s["pairs"][0]["reading_left"] is True and len(IC.gone_pairs(s["pairs"])) == 1


def test_record_per_file_gives_the_line_to_its_parent_before_releasing_the_reading():
    """R sits on P's 13C line (the isotopologue gate strips it as P's satellite, P kept), and the
    element-signature removal also took R out of the merged ledger elsewhere. The row on the
    line is P's line: it becomes P's iso_child with the gate's mark; releasing R first would
    have left it unexplained."""
    kept, tab, lmz = _stripped_case()
    pmz = C.ion_mz(*P)
    led = {"f1": _ledger([("p", pmz, P), ("r", lmz, R)], "f1")}
    gone = [{"neutral_formula": R[0], "adduct": R[1], "note": "34S absent", "reading_left": True}]
    changed, rw, sg = IC.record_per_file(led, tab, gone, mass_scale=SCALE, log=quiet)
    assert changed == {"f1"} and rw["iso_child"] == 1 and sg["released"] == 0, (rw, sg)
    r = _row(led["f1"], lmz)
    assert r["role"] == L.ROLE_ISO and r["parent_peak_id"] == "f1_p" and str(r["commentary"]).startswith(MARK)
    # a file whose R is off the line releases it there
    led2 = {"f2": _ledger([("p", pmz, P), ("r", C.ion_mz(*R) + 0.3, R)], "f2")}
    changed, rw, sg = IC.record_per_file(led2, tab, gone, mass_scale=SCALE, log=quiet)
    assert rw["iso_child"] == 0 and sg["released"] == 1


def test_batch_restamp_names_no_released_reading_on_its_lines(tmp_path, monkeypatch):
    """Through assign_batch.run: PB (bromine) is refuted on its 81Br line and removed; s05 / s09
    read its 13C line as PB's child, s01 commits R there. Every per-file ledger releases PB and
    its line, and the re-stamped series reads the per-file rows as the gates left them: no
    spectrum names PB's ion on that line."""
    PB = ("C8H13BrO4", "[M-H]-")
    plan = {"s01": (True, True), "s05": (True, False), "s09": (True, False)}
    summ, merged, per, lev, lmz, pmz = _batch(tmp_path, monkeypatch, plan, parent=PB)
    g = summ["merge_gates"]
    assert g["element_signature"]["removed"] == 1 and g["element_signature"]["restamped"], g["element_signature"]
    assert g["element_signature"]["per_file"]["released"] == 3 and g["element_signature"]["per_file"]["children"] == 2
    assert g["isotopologue"]["restamped"] and g["isotopologue"]["per_file"]["released_parent_removed"] == 1
    for sid in ("s05", "s09"):
        r = per[sid].loc[np.isclose(per[sid]["mz"], lmz)].iloc[0]
        assert r["role"] == "unexplained" and IC.SIG_LEDGER_MARK in str(r["commentary"])
    ts = pd.read_parquet(os.path.join(tmp_path, "run", "per_file", "_batch_ts.parquet"))
    on_line = ts[np.isclose(ts["mz"].astype(float), lmz, rtol=3e-6, atol=0)]
    assert len(on_line) and not (on_line["ion_formula"].astype(str) == _ionf(*PB)).any(), \
        on_line[["mz", "ion_formula"]].drop_duplicates().to_dict("records")
