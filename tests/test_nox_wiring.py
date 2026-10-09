"""The NOx-skeleton readings end to end: `assign.run` and `assign_batch.run` wire
the run's context profile into every consumer.

tests/test_nox_skeleton.py proves each piece on directly called units; this file
proves the runs connect them -- a nitrate-reagent run on an Orbitrap-class axis
switches the readings on and its grid admits a dinitrate, its twin tie-break, its
plausibility demotes, its rearbitration, its evidence level and its residual
series all see the same profile; a TOF run and a trace-first sample keep the
named context; and a batch records the switches its files ran with, in the
level summary, batch_summary.json and the PDF's scrutiny scan. Offline:
synthetic peak tables served from memory, the local scorer, no network.
"""
from __future__ import annotations

import inspect
import json
import os

import numpy as np
import pandas as pd
import pytest
from mascope_tools.composition.heuristic_filter import (
    anchor_on_monoisotopic,
    predict_isotopes,
)

from peaky.assignment import assign as A
from peaky.assignment import evidence as EV
from peaky.assignment import ledger as L
from peaky.assignment import passes
from peaky.assignment import plausibility as PL
from peaky.assignment import residual as R
from peaky.assignment.passes import directors as D
from peaky.assignment.passes.config import PassConfig
from peaky.batch import assign_batch as AB
from peaky.chem import chemistry as C
from peaky.chem import contexts as X
from peaky.io import io_mascope as IO

SID = "offline-nox-wiring"
ON = {"nox_skeleton": True, "small_acid_band": True}
ADDUCTS = ["[M+NO3]-", "[M-H]-"]

#: (tag, ion formula, height): isoprene dihydroxy dinitrate C5H10N2O8 [M+NO3]- (raw O/C 1.6:
#: admissible only through its k=2 skeleton C5H12O4), a plain CHO acid C6H8O4 [M+NO3]-, and
#: C3H6NO5-, which is propionic acid C3H6O2 [M+NO3]- or its skeleton-only twin, glyceryl
#: nitrate C3H7NO5 [M-H]- (the same ion: the nitrate run reports the cluster reading)
LINES = (("D", "C5H10N3O11", 2.0e5), ("A", "C6H8NO7", 2.0e5), ("T", "C3H6NO5", 1.0e5))


def _table(scale: float = 1.0) -> pd.DataFrame:
    rows = []
    for tag, ion, h in LINES:
        mzs, ints, _ = anchor_on_monoisotopic(*predict_isotopes(ion, -1))
        for i, (m, r) in enumerate(zip(mzs, ints / ints[0])):
            if r >= 0.005:
                rows.append({"peak_id": f"{tag}{i}", "mz": float(m), "height": h * scale * float(r)})
    t = pd.DataFrame(rows)
    t["signal_to_noise"] = t["height"] / 20.0
    return t


def _m0(res, pid):
    led = res["ledger"]
    r = led[(led["peak_id"] == pid) & (led["role"] == "M0")]
    return None if r.empty else r.iloc[0]


@pytest.fixture(autouse=True)
def _clean():
    IO.unregister_offline_sample(SID)
    yield
    IO.unregister_offline_sample(SID)


@pytest.fixture
def spies(monkeypatch):
    """What the run handed its grid / residual arbitration, its plausibility
    demotes, its rearbitration and its evidence level: the run's own context
    profile (or its switches)."""
    seen: dict = {}
    real_pl, real_rearb, real_ri = PL.demote_implausible, passes.rearbitrate_offcal_degenerate, EV.file_run_inputs

    def pl(led, **kw):
        seen["plausibility"] = kw.get("context")
        return real_pl(led, **kw)

    def rearb(led, cfg, **kw):
        seen["rearbitrate"] = kw.get("profile")
        return real_rearb(led, cfg, **kw)

    def ri(**kw):
        seen["evidence"] = kw.get("context_flags")
        return real_ri(**kw)

    def arb_spy(real_arb):
        def arb(scored, cfg, profile=None):
            seen.setdefault("arbitrate", []).append((inspect.stack()[1].function, profile))
            return real_arb(scored, cfg, profile)
        return arb

    monkeypatch.setattr(D, "arbitrate", arb_spy(D.arbitrate))
    monkeypatch.setattr(R, "arbitrate", arb_spy(R.arbitrate))
    monkeypatch.setattr(PL, "demote_implausible", pl)
    monkeypatch.setattr(passes, "rearbitrate_offcal_degenerate", rearb)
    monkeypatch.setattr(EV, "file_run_inputs", ri)
    return seen


def _run(scoring="orbi", trace_sample=False):
    cfg = PassConfig()
    cfg.trace_sample = trace_sample
    return A.run(SID, context="ambient-air", cfg=cfg, peaks=_table(), use_cache=False, scoring=scoring,
                 adducts=ADDUCTS, reagent_profile="NO3", reagent_n_relabel=False, log=lambda *a: None)


def test_a_nitrate_orbitrap_run_reads_the_skeleton_end_to_end(spies):
    res = _run()
    # the run derived the switches and recorded them, on the result and in its stats
    assert res["context_flags"] == ON and res["stats"]["context_flags"] == ON
    # the pass-1 grid admitted and committed the dinitrate (raw O/C 1.6: out of ambient-air's 1.5)
    d = _m0(res, "D0")
    assert d is not None and (d["neutral_formula"], d["adduct"]) == ("C5H10N2O8", "[M+NO3]-")
    assert int(d["pass_no"]) == 1
    # the twin tie-break in pass-1 arbitration: the cluster reading, not the skeleton-only twin
    t = _m0(res, "T0")
    assert t is not None and (t["neutral_formula"], t["adduct"]) == ("C3H6O2", "[M+NO3]-")
    # every downstream consumer was handed the run's profile, not the context's name
    for k in ("plausibility", "rearbitrate"):
        assert isinstance(spies[k], X.ContextProfile) and X.profile_flags(spies[k]) == ON, k
    assert spies["evidence"] == ON
    # the grid passes' arbitration (the twin tie-break) read the run's profile, every call
    grid = [(f, p) for f, p in spies["arbitrate"] if f in ("run_pass1", "run_pass2", "run_pass3",
                                                           "stage_a_iso_pairs", "stage_b_series")]
    assert "run_pass1" in {f for f, _ in grid}
    assert all(p is not None and X.profile_flags(p) == ON for _, p in grid), grid


@pytest.mark.parametrize("scoring,trace_sample", [("tof", False), ("orbi", True)])
def test_a_tof_run_and_a_trace_first_sample_keep_the_named_context(spies, scoring, trace_sample):
    res = _run(scoring, trace_sample)
    assert res["context_flags"] == {} and res["stats"]["context_flags"] == {}
    d = _m0(res, "D0")
    assert d is None or d["neutral_formula"] != "C5H10N2O8"
    assert "C5H10N2O8" not in set(res["ledger"]["neutral_formula"].dropna())
    for k in ("plausibility", "rearbitrate"):
        assert X.profile_flags(X.as_profile(spies[k])) == {}, k
    assert not spies["evidence"]
    assert all(X.profile_flags(p) == {} for _, p in spies["arbitrate"] if p is not None)


def test_the_residual_series_stage_filters_on_the_run_profile():
    """Pass 4.B proposes deep-series homologs of the committed anchors on the unexplained peaks;
    C4H8N2O8 (-CH2 from the dinitrate anchor, raw O/C 2.0) passes its context filter only
    through the NOx skeleton, so it is proposed on the nitrate run and not on the named context."""
    target = C.ion_mz("C4H8N2O8", "[M+NO3]-")
    peaks = pd.DataFrame([("a", C.ion_mz("C5H10N2O8", "[M+NO3]-"), 1e6), ("u", target, 1e6)],
                         columns=["peak_id", "mz", "height"])
    run_prof = A.run_context_profile("ambient-air", reagent_profile="NO3", adducts=ADDUCTS,
                                     instrument_class="orbitrap", trace_sample=False)
    got = {}
    for name, prof in (("run", run_prof), ("named", X.get_context("ambient-air"))):
        led = L.new_ledger(peaks.copy())
        L.commit_assignment(led, "a", neutral_formula="C5H10N2O8", adduct="[M+NO3]-",
                            ion_formula="C5H10N3O11-", ion_score=0.95, compound_score=0.95,
                            ppm_error=0.1, pass_no=1, method="cheminfo+grid", confidence="High",
                            commentary="anchor")
        asked = []

        def score(client, sid, formulas, **kw):
            asked.extend(formulas)
            return pd.DataFrame()

        R.stage_b_series(None, SID, led, prof, PassConfig(height_cutoff_cps=1.0), ADDUCTS, reagent="NO3",
                         score_fn=score, log=lambda *a: None)
        got[name] = set(asked)
    assert "C4H8N2O8" in got["run"]
    assert "C4H8N2O8" not in got["named"]


# --- the batch: its record is the switches its files ran with -------------------------------------------------
_T0 = pd.Timestamp("2025-10-01 21:00:00", tz="UTC")
_SIDS = [f"s{i}" for i in range(4)]


def _batch(monkeypatch, tmp_path, scoring):
    """An offline 4-file batch through the real per-file assigner: every file serves the same
    three lines (heights scaled per file)."""
    def file_table(sid):
        return _table(1.0 + 0.1 * _SIDS.index(sid))

    ts = pd.concat([file_table(s).assign(sample_item_id=s, sample_item_name=f"n_{s}",
                                         datetime_utc=_T0 + pd.Timedelta(minutes=10 * i))
                    for i, s in enumerate(_SIDS)], ignore_index=True)
    if scoring == "tof":
        ts["instrument_type"] = "tof"          # the roster's class wins over the width model
    real = A.run
    files: dict = {}

    def per_file(sid, context="ambient-air", **kw):
        kw.pop("peaks", None)
        res = real(sid, context=context, peaks=file_table(sid), scoring=scoring, use_cache=False,
                   **{**kw, "log": lambda *a: None})
        files[sid] = res["context_flags"]
        return res

    levelled: dict = {}
    real_lb = EV.level_batch

    def level_batch(ledgers, run_inputs=None, **kw):
        levelled["summary"] = dict(run_inputs.summary)
        return real_lb(ledgers, run_inputs=run_inputs, **kw)

    monkeypatch.setattr(A, "run", per_file)
    monkeypatch.setattr(EV, "level_batch", level_batch)
    monkeypatch.setattr(IO, "connect", lambda *a, **k: None)
    monkeypatch.setattr(IO, "fetch_peaks", lambda client, sid, use_cache=True: file_table(sid))
    monkeypatch.setattr(IO, "estimate_offset", lambda raw: 0.0)
    out = str(tmp_path)
    AB.run(peaks=ts, ts_peaks=ts, reagent="NO3", batch="b", out_dir=out, residual=False, n_jobs=1,
           resolving_power=250000.0, log=lambda *a: None)
    for s in _SIDS:
        IO.unregister_offline_sample(s)
    summ = json.load(open(os.path.join(out, "batch_summary.json")))
    merged = pd.read_csv(os.path.join(out, "merged_ledger.csv"))
    # the PDF's scrutiny scan reads the run's profile back from batch_summary.json
    from peaky.reporting import pdf_report as PR
    scanned: dict = {}
    real_scan = PL.scan

    def scan(m, **kw):
        scanned["profile"] = kw.get("profile")
        return real_scan(m, **kw)

    monkeypatch.setattr(PL, "scan", scan)
    PR.load_context(out, tag="X", label="X")
    return summ, merged, files, levelled["summary"], scanned.get("profile")


@pytest.mark.parametrize("scoring,want", [("orbi", ON), ("tof", {})])
def test_a_batch_records_the_switches_its_files_ran_with(monkeypatch, tmp_path, scoring, want):
    summ, merged, files, level_summary, pdf_prof = _batch(monkeypatch, tmp_path, scoring)
    # every file's own run derived them; the batch's record, its level summary and the
    # per-file stats it keeps agree with the files
    assert sorted(files) == _SIDS and all(f == want for f in files.values())
    assert summ["context_flags"] == want
    assert level_summary["context_flags"] == want
    assert [pf["context_flags"] for pf in summ["per_file"]] == [want] * len(_SIDS)
    # the batch's record is what assign.run derives from the same reagent, class and trace flag
    klass = "tof" if scoring == "tof" else "orbitrap"
    assert X.profile_flags(A.run_context_profile(
        "ambient-air", reagent_profile="NO3", adducts=ADDUCTS, instrument_class=klass,
        trace_sample=False)) == want
    # the PDF judges the merged ledger on the same profile
    assert X.profile_flags(pdf_prof) == want
    assert ("C5H10N2O8" in set(merged["neutral_formula"].dropna())) == bool(want)


def test_the_shared_derivation_follows_reagent_class_and_trace_flag():
    """assign.run_context_profile is the one derivation: the named context's profile with the
    switches on for a nitrate reagent (by name, alias or channels) on an Orbitrap-class
    axis, off for TOF, unknown class, the trace-first sample and another reagent."""
    def flags(**kw):
        base = dict(reagent_profile="NO3", adducts=ADDUCTS, instrument_class="orbitrap", trace_sample=False)
        return X.profile_flags(A.run_context_profile("ambient-air", **{**base, **kw}))

    assert flags() == ON
    assert flags(reagent_profile=None) == ON                  # from the channels
    assert flags(reagent_profile="NO3_15N", adducts=["[M+^NO3]-", "[M-H]-"]) == ON
    for kw in (dict(instrument_class="tof"), dict(instrument_class=None), dict(trace_sample=True),
               dict(reagent_profile="Br", adducts=["[M+Br]-", "[M-H]-"])):
        assert flags(**kw) == {}, kw
    assert np.all([isinstance(A.run_context_profile(c, reagent_profile="NO3", adducts=ADDUCTS,
                                                    instrument_class="orbitrap", trace_sample=False),
                              X.ContextProfile) for c in ("ambient-air", X.get_context("chamber"))])
