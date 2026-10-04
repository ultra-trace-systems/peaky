"""Rule K pins from the second mutation hunt (peaky/batch/label_twins.py,
peaky/assignment/evidence.py, scripts/level_ledger.py; docs/EVIDENCE_LEVELS.md
§3 `label_untie` / `label_alien` / `label_veto`, §4.2, §6.3).

Each test pins one rule a surviving mutant broke, on small synthetic inputs with
exact expected values; the helpers are those of tests/test_label_twins.py and
its core / veto / levels companions.

Run: pytest tests/test_label_twins_hunt.py -q
"""
from __future__ import annotations

from types import SimpleNamespace as NS

import numpy as np
import pandas as pd
import pytest

from peaky.assignment import evidence as EV
from peaky.batch import label_twins as LT
from peaky.chem import chemistry as C
from tests import test_label_twins as TL
from tests.test_evidence import child, ledger, m0
from tests.test_label_twins import (LABEL, NO3, NO3L, REFS, K, _alias_ledger, _frames, _j1_rows, _levels, _ll, _measure,
                                    _nitrate_rows, _refs_spec, _row, _series)
from tests.test_label_twins_core import _mostly_absent
from tests.test_label_twins_levels import J, X, _engine, _run_dir, _script
from tests.test_label_twins_veto import F0, MZ15, TWIN, Y, _ones, _ts, _veto

N = TL.N_SPECTRA
T0 = pd.Timestamp("2026-08-11", tz="UTC")
QUIET = lambda *a: None   # noqa: E731


def _h(ts: pd.DataFrame, neutral: str, adduct: str) -> np.ndarray:
    """The heights of one line of a _series, in spectrum (time) order."""
    return ts[(ts.neutral_formula == neutral) & (ts.adduct == adduct)]["height"].to_numpy(float)


# =========================================================================== the traces
def test_the_spectra_are_ordered_by_time_not_by_id():
    """_Traces: spectra (scan starts and height rows) run in time order, not sample-id order."""
    ts = pd.DataFrame(dict(sample_item_id=["a", "b", "b"],
                           datetime_utc=[T0 + pd.Timedelta(minutes=1), T0, T0],
                           mz=[20.0, 30.0, 40.0], height=[1.0, 2.0, 3.0]))
    tr = LT._Traces(ts)
    assert tr.order == ["b", "a"]
    assert list(tr.low) == [25.0, 20.0]                   # b 30 capped at the median 25; a 20
    assert np.array_equal(tr.heights([20.0, 40.0]), [[np.nan, 3.0], [1.0, np.nan]], equal_nan=True)


def test_the_scan_start_follows_its_own_spectrum_whatever_the_ids():
    """_Traces.low[i] is the scan start of order[i] when ids run against time (E = 50, refuted)."""
    i = np.arange(N)
    ts = _ts([(MZ15, np.where(i < 60, 10000.0, np.nan)), (MZ15 + 5.0, np.where(i >= 60, 1.0, np.nan))],
             start=np.where(i < 50, TWIN - 1.0, np.nan))
    ts["sample_item_id"] = ts["sample_item_id"].map(lambda s: f"z{N - int(s[1:]):03d}")   # ids run against time
    tr = LT._Traces(ts)
    low = dict(zip(tr.order, tr.low))
    assert [low[f"z{N - k:03d}"] for k in range(N)] == pytest.approx(
        [TWIN - 1.0] * 50 + [MZ15] * 10 + [MZ15 + 2.5] * 60)
    r = LT._twin_veto(tr, [Y], F0, _ones()).iloc[0]
    assert (r.E, r.obs, r.twin_verdict) == (50.0, 0, "refuted")


def test_a_peak_without_a_height_still_marks_the_scan_start():
    """The scan start is the lowest recorded m/z whatever its height cell reads (E = 120, refuted)."""
    ts = _ts([(MZ15, np.full(N, 10000.0))], start=TWIN - 0.5)
    ts.loc[ts["mz"] == TWIN - 0.5, "height"] = np.nan
    tr = LT._Traces(ts)
    assert list(tr.low) == [TWIN - 0.5] * N
    r = LT._twin_veto(tr, [Y], F0, _ones()).iloc[0]
    assert (r.E, r.obs, r.twin_verdict) == (120.0, 0, "refuted")


# =========================================================================== the committed pairs
def test_role_less_frames_are_skipped():
    """_committed and alias_only_ties skip a non-empty frame without a `role` column (no KeyError)."""
    x = "C10H16O4"
    bare = pd.DataFrame({"neutral_formula": ["C5H8O2"], "adduct": [NO3L]})
    got = LT._committed({"f1": _frames([x], no3=[x])["f1"], "f2": bare})
    assert sorted(map(tuple, got[["neutral_formula", "adduct"]].values)) == [(x, NO3), (x, NO3L)]
    a = LT.alias_only_ties(pd.DataFrame({"neutral_formula": [x], "adduct": [NO3], "tied": [True]}))
    assert a.empty and list(a.columns) == ["neutral_formula", "adduct", "alias_only"]


def test_a_neutral_committed_off_nitrate_gets_no_14n_line():
    """measure: only [M+NO3]- commits make 14N lines; an [M-H]- commit is no line and not alien."""
    frames = dict(_frames(REFS), f2=EV.trim(ledger([m0("h", "C5H8O4", adduct="[M-H]-",
                                                       mz=C.ion_mz("C5H8O4", "[M-H]-"))])))
    t = LT.measure(_series(_refs_spec()), frames, LABEL, log=QUIET)
    assert "C5H8O4" not in set(t.neutral_formula) and len(t) == 2 * len(REFS)
    assert LT.facts(t) == {"untie": set(), "veto": {}, "alien": set()}
    assert LT.summary(t, LABEL)["committed_lines"] == 0


# =========================================================================== cluster-k
def test_h15_median_is_read_on_the_co_detected_spectra_only():
    """h15_median is the median 15N height over the co-detected spectra only (101 of 120)."""
    x = "C9H16O4"
    spec = _refs_spec(**{x: {"phase": 1.8, "share": 100.5 / N}})
    ts = _series(spec)
    r = _row(LT.measure(ts, _frames(list(spec)), LABEL, log=QUIET), x)
    h15 = _h(ts, x, NO3L)
    assert (r.n14, r.n_codetected) == (101, 101)
    assert r.h15_median == float(np.median(h15[:101]))
    assert r.h15_median != float(np.median(h15))


def test_an_organonitrate_cluster_is_never_a_reference():
    """cluster-k references are C/H/O only: a bright co-detected CHON cluster never calibrates k_cl."""
    n = "C10H15NO7"
    t = _measure(_refs_spec(**{n: {"phase": 1.1}}))
    r = _row(t, n)
    assert r.n_codetected == 120 and r.h15_median >= LT.REF_MIN_H15 and not r.reference
    assert LT.summary(t, LABEL)["references"] == len(REFS)


def test_leave_one_out_drops_the_tested_neutral_not_a_position():
    """Leave-one-out drops the tested neutral itself, not the reference at its index position."""
    dim = "C10H10O4"                                      # sorts first, not a reference (h15 300)
    t = _measure({dim: {"phase": 1.0, "h15": 300.0}, REFS[0]: {"phase": 0.0, "q": 5.0}, REFS[1]: {"phase": 0.6}})
    lines = t[t.adduct == NO3]
    assert list(lines.neutral_formula) == [dim, REFS[0], REFS[1]] and list(lines.reference) == [False, True, True]
    r = _row(t, REFS[0])
    assert 4.5 < r.k_ratio < 5.5 and r.line_verdict == "excess"
    r = _row(t, REFS[1])
    assert 0.18 < r.k_ratio < 0.22 and r.line_verdict == "consistent"


def test_k_cl_skips_a_reference_missing_from_a_spectrum():
    """k_cl is a NaN-skipping median: a reference missing from 9 spectra costs the others none."""
    spec = _refs_spec()
    spec[REFS[0]] = {"phase": 0.0, "share": 110.5 / N}
    t = _measure(spec)
    assert _row(t, REFS[0]).reference and _row(t, REFS[0]).n_codetected == 111
    assert [int(_row(t, n).k_n) for n in REFS[1:]] == [120] * 5


def test_a_zero_k_cl_spectrum_is_skipped():
    """A spectrum with k_cl = 0 carries no ratio (k_n 90, finite sd, tracks)."""
    spec = {REFS[0]: {"phase": 0.0}, REFS[1]: {"phase": 0.6}}
    ts = _series(spec)
    sel = ts.index[(ts.neutral_formula == REFS[1]) & (ts.adduct == NO3)][:30]
    ts.loc[sel, "height"] = 0.0
    r = _row(LT.measure(ts, _frames(list(spec)), LABEL, log=QUIET), REFS[0])
    assert r.k_n == 90 and np.isfinite(r.k_sd) and r.line_verdict == "tracks"


def test_n14_counts_every_spectrum_holding_the_14n_line():
    """n14 counts every spectrum holding the 14N line, zero heights included (partner_share 1.0)."""
    x = "C9H16O4"
    spec = _refs_spec(**{x: {"phase": 1.8}})
    ts = _series(spec)
    sel = ts.index[(ts.neutral_formula == x) & (ts.adduct == NO3)][-70:]
    ts.loc[sel, "height"] = 0.0
    r = _row(LT.measure(ts, _frames(list(spec)), LABEL, log=QUIET), x)
    assert (r.n14, r.n_codetected, r.partner_share) == (120, 120, 1.0)


def test_partner_share_of_a_single_spectrum_line():
    """partner_share = n_codetected / n14: one spectrum with both lines reads 1.0."""
    x = "C9H16O4"
    r = _row(_measure(_refs_spec(**{x: {"phase": 1.8, "share": 0.5 / N}})), x)
    assert (r.n14, r.n_codetected, r.partner_share) == (1, 1, 1.0)


def test_a_single_reference_calibrates_the_other_lines():
    """One bright CHO reference is enough to test the other lines (the reference itself is untestable)."""
    s = "C10H16O4S"
    t = _measure({REFS[1]: {"phase": 0.6}, s: {"phase": 1.0}})
    assert _row(t, REFS[1]).reference and not _row(t, s).reference
    r = _row(t, s)
    assert r.k_n == 120 and r.line_verdict == "tracks"
    assert _row(t, REFS[1]).line_verdict == "untestable"


def test_cluster_k_without_a_pool_draws_references_from_every_neutral():
    """_cluster_k with ref_pool None (its documented default) draws references from every neutral."""
    spec = _refs_spec(**{"C10H16O4S": {"phase": 1.0}})
    ck = LT._cluster_k(LT._Traces(_series(spec)), sorted(spec))
    assert sorted(ck.loc[ck.reference, "neutral_formula"]) == sorted(REFS)
    assert ck.set_index("neutral_formula").loc["C10H16O4S", "line_verdict"] == "tracks"


# =========================================================================== line verdicts and the untie
def test_a_line_above_its_cluster_share_whose_partner_is_mostly_absent_is_absent(monkeypatch):
    """absent outranks excess: a 3x line with its partner in 108 of 600 spectra reads absent."""
    r = _mostly_absent(monkeypatch, q=3.0)
    assert r.k_ratio > LT.RATIO_HI
    assert (r.line_verdict, bool(r.alien), bool(r.veto), bool(r.untie)) == ("absent", True, True, False)
    assert r.note == ("no 15N partner: seen in 108 of the 600 spectra of the 14N line; "
                      "cluster or organonitrate undecided")


def test_a_consistent_line_never_unties():
    """untie needs a tracking line: an alias-only tied, committed consistent line keeps its tie."""
    x = "C10H18O3"
    spec = _refs_spec(**{x: {"phase": 2.0, "noise": 0.6}})
    t = _measure(spec, _frames(list(spec), no3=[x], tied=[x]), {"f1": LT.alias_only_ties(_alias_ledger(x))})
    r = _row(t, x)
    assert (r.line_verdict, bool(r.alias_only_tie), bool(r.committed)) == ("consistent", True, True)
    assert not r.untie and not r.alien and not r.veto and LT.facts(t)["untie"] == set()


def test_the_log_counts_the_untied_lines():
    """The run log's '(N untie)' counts untied lines, not lines passing cluster-k."""
    x = "C10H18O3"
    spec = _refs_spec(**{x: {"phase": 2.0}})
    logs = []
    LT.measure(_series(spec), _frames(list(spec), no3=[x], tied=[x]), LABEL,
               alias_ties={"f1": LT.alias_only_ties(_alias_ledger(x))}, log=logs.append)
    assert logs == ["[label_twins] 1 committed [M+NO3]- lines: 1 tracks, 0 consistent, 0 excess, 0 absent, "
                    "0 untestable (1 untie); 7 [M+^NO3]- readings: 0 refuted by their 14N twin "
                    "(f 0.0204, 0 untestable)"]


# =========================================================================== the veto
def test_a_twin_found_as_the_lowest_peak_just_above_its_mz_was_measured():
    """A twin found (+1 ppm) as each spectrum's lowest peak was measured (E = 120, passes)."""
    r = _veto(np.full(N, 10000.0), np.ones(N), _ones(), start=None, twin_mz=TWIN * (1 + 1e-6))
    assert (r.E, r.obs, r.twin_verdict) == (120.0, 120, "passes")


def test_a_twin_exactly_at_the_scan_start_was_measured():
    """A scan start exactly at the twin m/z is not above it: the twin was measurable (E = 59, refuted)."""
    i = np.arange(N)
    ts = _ts([(TWIN, np.where(i < 61, 200.0, np.nan)), (MZ15, np.where(i >= 61, 10000.0, np.nan))], start=None)
    tr = LT._Traces(ts)
    assert list(tr.low) == [TWIN] * N
    r = LT._twin_veto(tr, [Y], F0, _ones()).iloc[0]
    assert (r.E, r.obs, r.twin_verdict) == (59.0, 0, "refuted")


# =========================================================================== the table readers and the funnel
def test_summary_counts_a_legacy_table_on_its_14n_lines():
    """summary(): no `committed` column = every 14N line committed; untie counts 14N lines only."""
    t = pd.DataFrame([dict(neutral_formula="A", adduct=NO3, n_codetected=120, reference=False, cluster_k=False,
                           line_verdict="consistent", untie=False, alien=False, veto=False, note=""),
                      dict(neutral_formula="A", adduct=NO3L, n_codetected=np.nan, reference=False, cluster_k=False,
                           untie=True, alien=False, f=F0, E=120.0, obs=120, twin_verdict="passes", veto=False,
                           note="")])
    s = LT.summary(t, LABEL)
    assert (s["committed_lines"], s["lines_consistent"], s["untie"], s["readings"], s["passes"]) == (1, 1, 0, 1, 1)


# =========================================================================== the level facts (engine)
def test_an_alien_line_keeps_its_own_multiline():
    """An alien 14N line keeps its own multiline (C|S) -- with iso and the anchor, the
    neutral backed (the merge vote's class 2), engine and script alike."""
    s = "C8H6O2S"
    rows = ledger([m0("n14", s, adduct=NO3, ion="C8H6NO5S", mz=C.ion_mz(s, NO3), anchor="a"),
                   child("c1", "n14", "13C+1", 1000.0 * EV.C13_PER_CARBON * 8),
                   child("c2", "n14", "34S", 1000.0 * 0.0443)])
    lab = K(alien={(s, NO3)})
    r = _engine(rows, lab).loc[(s, NO3)]
    assert (bool(r.multiline), r.multiline_elements, bool(r.iso), bool(r.anchor)) == (True, "C|S", True, True)
    assert EV._vote_class_of(r.evidence_level, r.evidence_axes) == 2
    assert _engine(rows).loc[(s, NO3), "evidence_level"] == r.evidence_level
    assert _script(rows, lab).loc[(s, NO3), "level"] == r.evidence_level


def test_an_alien_line_keeps_its_own_reagent_satellite_fact():
    """An alien line keeps its own reagent_only_iso (81Br only), engine and script alike."""
    y, br = "C6H10O5", "[M+Br]-"
    rows = ledger([m0("b", y, adduct=br, ion=y + "Br", mz=C.ion_mz(y, br)),
                   child("b81", "b", "81Br", 1000.0 * 0.9728)])
    lab = K(alien={(y, br)})
    r = _engine(rows, lab).loc[(y, br)]
    assert bool(r.reagent_only_iso) and bool(r.iso) and not bool(r.neutral_backed)
    assert bool(_engine(rows).loc[(y, br), "reagent_only_iso"])
    LL = _ll()
    frame = LL.measure_source("s1", rows.assign(__file="s1"), "Br")
    got = LL.assign_levels(frame, set(), None, lab).set_index(["neutral", "adduct"]).loc[(y, br)]
    assert bool(got["reagent_only_iso"]) and got["level"] == r.evidence_level


def test_an_untie_on_a_below_row_clears_only_the_tie():
    """The untie is guarded by the tie alone: a tied below row takes label_untie and stays hard on below."""
    rows = _nitrate_rows(X)
    rows.loc[rows.peak_id == "n14", "below_assignability"] = True
    lv = _levels(rows, label=K(untie={(X, NO3)}))
    assert lv[NO3] == (("below",), "chan2|branch|label_untie|files:1")
    ref = _script(rows, K(untie={(X, NO3)}))
    assert bool(ref.loc[(X, NO3), "label_untie"]) and not bool(ref.loc[(X, NO3), "tied"])
    assert bool(ref.loc[(X, NO3), "below"])


def test_decide_without_a_label_veto_field_is_not_vetoed():
    """evidence._decide (the merge vote's private decision): a record without the label_veto field is
    not vetoed (the absent fact is False) -- the formula confirmed, class 1; with the veto, unconfirmed."""
    r = NS(degeneracy=1.0, tied=False, below=False, lowconf=False, ion_only=False, iso=True, saturated=False,
           n_axes=1, known_fam="", branch=False, neutral_backed=True, reagent_only_iso=False, cross=False,
           res_ok=True, upair=False, n_plausible_structures=pd.NA, resolvability="resolved", corroborated=False,
           chan2=False, anchor=False)
    assert EV._vote_class_of(EV._decide(r)[0], "iso") == 1
    vetoed = EV._decide(NS(**vars(r), label_veto=True, label_note="n"))
    assert EV._vote_class_of(vetoed[0], "iso") == 0 and vetoed[1].endswith("(n)")


# =========================================================================== the reference script
def test_level_of_without_a_label_veto_column_is_not_vetoed():
    """level_ledger.level_of: a row without the label_veto column is not vetoed."""
    LL = _ll()
    frame = pd.DataFrame([dict(degeneracy=1.0, tied=False, below=False, lowconf=False, ion_only=False, iso=True,
                               saturated=False, n_axes=1, known_fam="", branch=False, neutral_backed=True,
                               reagent_only_iso=False, cross=False, res_ok=True, upair=False, neutral="C10H16O5")])
    rec = frame.iloc[0].to_dict()
    assert frame.apply(LL.level_of, axis=1).tolist() == [EV._decide(NS(**rec, chan2=False, anchor=False,
                                                                      corroborated=False,
                                                                      n_plausible_structures=pd.NA))[0]]
    vetoed = frame.assign(label_veto=True).apply(LL.level_of, axis=1).tolist()
    assert vetoed != frame.apply(LL.level_of, axis=1).tolist()
    assert vetoed == [EV._decide(NS(**rec, label_veto=True, chan2=False, anchor=False, corroborated=False,
                                    n_plausible_structures=pd.NA))[0]]


def test_auto_reads_a_runs_own_table_whatever_its_adducts(tmp_path):
    """--label-twins auto applies the run's own table with no [M+^NO3]- scope check (§6.3), as the engine."""
    spec = _refs_spec(**{J: {"phase": 0.9, "share15": 0.0}})
    table = LT.measure(_series(spec), {"f1": EV.trim(_j1_rows(J))}, LABEL, log=QUIET)
    run = _run_dir(tmp_path / "RUN_1", _j1_rows(J), table)
    LL = _ll()
    got = LL.run([str(run)], [], None, "auto").set_index(["neutral", "adduct"])
    core = _engine(_j1_rows(J), LT.facts(table))
    assert bool(got.loc[(J, NO3), "label_veto"]) and not bool(got.loc[(J, "[M-H]-"), "branch"])
    assert bool(core.loc[(J, NO3), "label_veto"]) and not bool(core.loc[(J, "[M-H]-"), "branch"])
    assert got["level"].to_dict() == core["evidence_level"].to_dict()


def test_the_reference_script_reads_a_hand_edited_table_strictly(tmp_path):
    """label_twin_facts reads 'yes'/'no' cells as facts() does: 'no' is False."""
    LL = _ll()
    y, x = "C12H14O2", "C10H18O4"
    p = tmp_path / "label_twins.csv"
    pd.DataFrame({"neutral_formula": [y, x], "adduct": [NO3L, NO3L], "untie": ["no", "no"],
                  "veto": ["yes", "no"], "alien": ["no", "no"], "note": ["n", ""]}).to_csv(p, index=False)
    got = LL.label_twin_facts(str(p))
    assert got == {"untie": set(), "veto": {(y, NO3L)}, "alien": set()}
    assert set(LT.facts(pd.read_csv(p))["veto"]) == got["veto"]


def test_a_named_table_fires_on_its_own_batch_even_without_a_labelled_commit(tmp_path):
    """--label-twins <csv> applies a table to the source that holds every pair it names -- also a
    labelled batch that committed nothing on [M+^NO3]- (the engine judges its 14N lines) -- and
    never to another batch."""
    from tests.test_label_twins import _j1_rows, _ll, _refs_spec, _series, LABEL, K
    j = "C11H18O6"
    rows = _j1_rows(j)
    table = LT.measure(_series(_refs_spec(**{j: {"phase": 0.9, "share15": 0.0}})), {"f1": EV.trim(rows)}, LABEL,
                       log=lambda *a: None)
    csv = tmp_path / "t.csv"
    table.to_csv(csv, index=False)
    run = tmp_path / "run" / "per_file"
    run.mkdir(parents=True)
    rows.to_csv(run / "s1_ledger.csv", index=False)
    LL = _ll()
    got = LL.run([str(tmp_path / "run")], [], None, str(csv)).set_index("adduct")
    core = EV._series_pooled({"s1": rows}, label=LT.facts(table)).set_index("adduct")
    assert got["level"].to_dict() == core["evidence_level"].to_dict()
    assert bool(got.at["[M+NO3]-", "label_veto"]) and bool(core.at["[M+NO3]-", "label_veto"])
    other = tmp_path / "other" / "per_file"
    other.mkdir(parents=True)
    _j1_rows("C11H18O5").to_csv(other / "s1_ledger.csv", index=False)
    untouched = LL.run([str(tmp_path / "other")], [], None, str(csv))
    assert not untouched["label_veto"].any() and untouched["branch"].all()

