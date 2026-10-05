"""What an offline sample is scored at (card C35).

`scoring_for_sample` reads a sample's instrument class from its server record
and its width and offset from its own server matches. An offline sample
(`register_offline_sample`: a decoy arm of a scored run, the trace-first
synthetic sample) has neither, so on its own it falls back to the more
forgiving class -- a TOF's -- at zero offset. A table that stands in for a
measured sample is registered with what that sample was judged at instead.

Offline: the clients are stubs; nothing here touches a network.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest
from mascope_tools.composition import (
    PatternScoring,
    resolve_fallback_sigma_ppm,
    resolve_match_tolerance_ppm,
    scoring_sigma_ppm,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import scorecard as SC  # noqa: E402
from peaky.io import io_mascope as IO  # noqa: E402
from peaky.io import local_scoring as LS  # noqa: E402

SID = "offline-c35"
# a measured Orbitrap sample's snapshot, as a 0.9.0 batch summary records it
SNAP = {"score_version": IO.SCORE_VERSION, "sigma_ppm": 0.7123, "mu_ppm": -1.25, "sigma_source": "fitted",
        "mu_source": "fitted", "fitted_anchors": 23, "mz_tolerance_ppm": 5.0, "abundance_floor": 0.02,
        "instrument_type": "orbi", "has_signal_to_noise": True}


def _table(snr: bool = False) -> pd.DataFrame:
    t = pd.DataFrame({"peak_id": ["a", "b"], "mz": [100.0, 200.0], "height": [1000.0, 500.0]})
    return t.assign(signal_to_noise=[30.0, 15.0]) if snr else t


class _Samples:
    def __init__(self, record):
        self.record, self.asked = record, []

    def get(self, sid):
        self.asked.append(sid)
        if isinstance(self.record, Exception):
            raise self.record
        return self.record


class _Client:
    def __init__(self, record):
        self.samples = _Samples(record)


@pytest.fixture(autouse=True)
def _clean():
    yield
    IO.unregister_offline_sample(SID)


def _class(kind):
    return (scoring_sigma_ppm(None, resolve_fallback_sigma_ppm(kind)), resolve_match_tolerance_ppm(kind))


def test_an_offline_sample_on_its_own_falls_back_to_the_forgiving_class_at_zero_offset():
    IO.register_offline_sample(SID, _table())
    sc = IO.scoring_for_sample(None, SID)
    snap = IO.scoring_snapshot(None, SID)
    assert (sc.sigma_ppm, sc.mz_tolerance_ppm) == pytest.approx(_class("tof")) and sc.mu_ppm == 0.0
    assert snap["sigma_source"] == "instrument_class" and snap["mu_source"] == "assumed_zero"


def test_a_snapshot_is_inherited_as_measured_and_says_so():
    IO.register_offline_sample(SID, _table(snr=True), scoring=SNAP)
    sc = IO.scoring_for_sample(None, SID)
    assert (sc.sigma_ppm, sc.mu_ppm, sc.mz_tolerance_ppm, sc.abundance_floor) == (0.7123, -1.25, 5.0, 0.02)
    snap = IO.scoring_snapshot(None, SID)
    assert snap["sigma_source"] == snap["mu_source"] == "inherited"
    assert snap["inherited"] == {"sigma_source": "fitted", "mu_source": "fitted", "fitted_anchors": 23,
                                 "has_signal_to_noise": True, "snr_source": None}   # C46: the measured sample's verdict
    assert snap["fitted_anchors"] == 0 and snap["instrument_type"] == "orbi" and snap["has_signal_to_noise"]


def test_signal_to_noise_is_the_tables_own_not_the_measured_samples():
    IO.register_offline_sample(SID, _table(snr=False), scoring=SNAP)
    assert IO.scoring_snapshot(None, SID)["has_signal_to_noise"] is False


def test_an_unset_width_or_offset_is_the_librarys_default_and_none_measured():
    from mascope_tools.composition.heuristic_filter import FALLBACK_SIGMA_PPM
    IO.register_offline_sample(SID, _table(), scoring=PatternScoring(mz_tolerance_ppm=8.0, abundance_floor=0.03))
    sc = IO.scoring_for_sample(None, SID)
    # the given window and floor are kept through the rebuild
    assert (sc.sigma_ppm, sc.mu_ppm, sc.mz_tolerance_ppm, sc.abundance_floor) == (FALLBACK_SIGMA_PPM, 0.0, 8.0, 0.03)
    IO.register_offline_sample(SID, _table(), scoring=PatternScoring(sigma_ppm=0.6, mu_ppm=None, mz_tolerance_ppm=5.0))
    sc = IO.scoring_for_sample(None, SID)
    assert (sc.sigma_ppm, sc.mu_ppm) == (0.6, 0.0)
    # a given offset is kept when only the width is unset
    IO.register_offline_sample(SID, _table(), scoring=PatternScoring(sigma_ppm=None, mu_ppm=-1.2, mz_tolerance_ppm=5.0))
    sc = IO.scoring_for_sample(None, SID)
    assert (sc.sigma_ppm, sc.mu_ppm) == (FALLBACK_SIGMA_PPM, -1.2)


@pytest.mark.parametrize("bad", ["Orbitrap", "qtof", {"sigma_ppm": 0.7}, dict(SNAP, mu_ppm=None),
                                 dict(SNAP, sigma_ppm="abc"), dict(SNAP, sigma_ppm=0.0), dict(SNAP, mu_ppm=float("nan")),
                                 dict(SNAP, mz_tolerance_ppm=-5.0), PatternScoring(sigma_ppm=float("nan")), 5, ["orbi"],
                                 PatternScoring(mz_tolerance_ppm=0), PatternScoring(mz_tolerance_ppm=None),
                                 PatternScoring(sigma_ppm="0.7"), dict(SNAP, abundance_floor=float("nan")),
                                 dict(SNAP, abundance_floor=-1), dict(SNAP, abundance_floor="x"), dict(SNAP, sigma_ppm=True),
                                 PatternScoring(mu_ppm=float("nan")), PatternScoring(mz_tolerance_ppm=5.0, abundance_floor=1.0),
                                 PatternScoring(mz_tolerance_ppm=5.0, abundance_floor=-0.1),
                                 PatternScoring(mz_tolerance_ppm=5.0, abundance_floor="0.01"),
                                 dict(SNAP, abundance_floor=1.0), dict(SNAP, sigma_ppm=None), dict(SNAP, sigma_ppm="0.7"),
                                 dict(SNAP, sigma_ppm=10 ** 400)])
def test_a_scoring_that_names_no_class_or_misses_a_number_is_refused(bad):
    with pytest.raises((ValueError, TypeError)):
        IO.register_offline_sample(SID, _table(), scoring=bad)
    assert SID not in IO._OFFLINE


def test_scoring_is_refused_for_a_sample_a_server_serves():
    from peaky.assignment import assign as A
    with pytest.raises(ValueError, match="OFFLINE"):
        A.run("server-sample", scoring=SNAP, log=lambda *a: None)


def test_a_class_is_read_whatever_its_case():
    IO.register_offline_sample(SID, _table(), scoring=" ORBI ")
    assert IO.scoring_for_sample(None, SID).mz_tolerance_ppm == 5.0


def test_a_decoy_arms_table_keeps_its_files_signal_to_noise():
    """The arm is rebuilt from the per-file ledger, which carries each peak's signal_to_noise: kept, so the arm is
    judged in the same SNR mode as its file; a run without the column leaves it empty (the no-SNR mode)."""
    pf = pd.DataFrame({"__file": ["f1", "f1"], "peak_id": ["a", "b"], "mz": [100.0, 200.0], "height": [10.0, 5.0],
                       "signal_to_noise": [40.0, 20.0]})
    run = _run({})
    run.per_file = pf
    assert SC.raw_peaks_of(run, "f1")["signal_to_noise"].tolist() == [40.0, 20.0]
    run.per_file = pf.drop(columns="signal_to_noise")
    assert SC.raw_peaks_of(run, "f1")["signal_to_noise"].isna().all()


def test_a_pattern_scoring_is_inherited_as_it_is():
    given = PatternScoring(sigma_ppm=2.5, mu_ppm=0.4, mz_tolerance_ppm=15.0)
    IO.register_offline_sample(SID, _table(), scoring=given)
    assert IO.scoring_for_sample(None, SID) is given
    snap = IO.scoring_snapshot(None, SID)
    assert snap["sigma_source"] == "inherited" and snap["inherited"]["sigma_source"] is None


@pytest.mark.parametrize("kind", ["orbi", "tof"])
def test_an_instrument_class_is_judged_at_that_class(kind):
    IO.register_offline_sample(SID, _table(), scoring=kind)
    sc = IO.scoring_for_sample(None, SID)
    assert (sc.sigma_ppm, sc.mz_tolerance_ppm) == pytest.approx(_class(kind)) and sc.mu_ppm == 0.0
    assert IO.scoring_snapshot(None, SID)["instrument_type"] == kind


def test_registering_again_replaces_the_scoring_and_forgets_the_old_one():
    IO.register_offline_sample(SID, _table(), scoring=SNAP)
    assert IO.scoring_for_sample(None, SID).sigma_ppm == 0.7123
    IO.register_offline_sample(SID, _table(), scoring="tof")
    assert IO.scoring_for_sample(None, SID).sigma_ppm == pytest.approx(_class("tof")[0])
    IO.unregister_offline_sample(SID)
    assert SID not in IO._SCORING_CACHE and SID not in IO._OFFLINE_SCORING


def test_a_measured_sample_is_untouched_by_an_offline_registration_of_another():
    """The inherited scoring is keyed on the offline sample only: a sample served by a server (here a TOF file
    with no anchors) is read from its own record as before."""
    IO.register_offline_sample(SID, _table(), scoring=SNAP)
    cl = _Client({"filename": "KLTOF2_x_2026.03.30-16h59m12s", "instrument": "KLTOF2"})
    try:
        sc = IO.scoring_for_sample(cl, "measured-c35", peaks=_table())
        assert (sc.sigma_ppm, sc.mz_tolerance_ppm) == pytest.approx(_class("tof")) and sc.mu_ppm == 0.0
        assert IO.scoring_snapshot(cl, "measured-c35")["sigma_source"] == "instrument_class"
    finally:
        IO._SCORING_CACHE.pop("measured-c35", None)


def test_the_offline_run_hands_the_scorer_what_it_was_registered_with(monkeypatch):
    """`assign.run(peaks=, scoring=)`: the scorer judges every candidate at the given measurement (both
    registrations of the run carry it)."""
    from peaky.assignment import assign as A
    seen = []
    real = LS.score_candidates_local

    def spy(*a, scoring=None, **k):
        seen.append(scoring)
        return real(*a, scoring=scoring, **k)
    monkeypatch.setattr(LS, "score_candidates_local", spy)
    peaks = pd.DataFrame({"peak_id": ["p1", "p2", "p3"], "mz": [61.98837, 124.98402, 160.99819],
                          "height": [5e4, 2e4, 1e4], "area": [5e4, 2e4, 1e4]})
    A.run(SID, peaks=peaks, scoring=SNAP, adducts=["[M-H]-", "[M+NO3]-"], use_cache=False, log=lambda *a: None)
    assert seen and all(s is not None and (s.sigma_ppm, s.mu_ppm) == (0.7123, -1.25) for s in seen)


def _run(summary):
    return SC.Run(path="/x/run", ledger=pd.DataFrame(), summary=summary, manifest={}, ts=None,
                  per_file=pd.DataFrame())


def test_a_decoy_arm_is_judged_at_its_files_measurement():
    run = _run({"pattern_scoring": {"f1": SNAP, "f2": None, "f4": dict(SNAP, mu_ppm=None),
                                    "f5": dict(SNAP, sigma_ppm="abc"), "f6": {k: v for k, v in SNAP.items()
                                                                          if k != "mz_tolerance_ppm"},
                                    "f7": dict(SNAP, sigma_ppm=float("nan")), "f8": dict(SNAP, sigma_ppm=0.0),
                                    "f9": dict(SNAP, mz_tolerance_ppm=-5.0), "f10": dict(SNAP, abundance_floor=float("nan")),
                                    "f11": dict(SNAP, abundance_floor=1.0), "f12": dict(SNAP, sigma_ppm=None),
                                    "f13": dict(SNAP, sigma_ppm=10 ** 400)}})
    assert SC.decoy_scoring(run, "f1") is SNAP
    assert SC.decoy_scoring(run, "f2") is None and SC.decoy_scoring(run, "f3") is None
    # a snapshot that could not be inherited is no snapshot: its arms take the class fallback, labelled so
    assert all(SC.decoy_scoring(run, f) is None for f in ("f4", "f5", "f6", "f7", "f8", "f9", "f10", "f11", "f12", "f13"))
    assert SC.decoy_scoring(_run({}), "f1") is None          # a run from before 0.9.0 records none


def test_the_decoy_chain_hands_the_engine_its_files_measurement(monkeypatch):
    """decoy -> run_engine_offline -> assign.run(peaks=, scoring=) -> the scorer, nothing stubbed but the spy: every
    candidate of every arm of a file whose run recorded a snapshot is scored at that snapshot."""
    seen = []
    real = LS.score_candidates_local

    def spy(*a, scoring=None, **k):
        seen.append(scoring)
        return real(*a, scoring=scoring, **k)
    monkeypatch.setattr(LS, "score_candidates_local", spy)
    run = _run({"pattern_scoring": {"f1": SNAP}})
    run.ledger = pd.DataFrame({"adduct": ["[M-H]-", "[M+NO3]-"]})
    run.per_file = pd.DataFrame({"__file": ["f1"] * 3, "peak_id": ["p1", "p2", "p3"],
                                 "mz": [61.98837, 124.98402, 160.99819], "height": [5e4, 2e4, 1e4],
                                 "area": [5e4, 2e4, 1e4], "signal_to_noise": [500.0, 200.0, 100.0]})
    dc = SC.decoy(run, "shift", 0.35, 1)
    assert dc["scoring"] == {"f1": "inherited"} and seen
    assert all(s is not None and (s.sigma_ppm, s.mu_ppm, s.abundance_floor) == (0.7123, -1.25, 0.02) for s in seen)


def test_the_card_and_the_row_say_what_the_arms_were_judged_at():
    assert SC.decoy_scoring_summary({"scoring": {"a": "inherited", "b": "inherited"}}) == "inherited"
    assert SC.decoy_scoring_summary({"scoring": {"a": "inherited", "b": "class-fallback"}}) == "mixed"
    assert SC.decoy_scoring_summary({"scoring": {"a": "unrecorded"}}) == "unrecorded"
    assert SC.decoy_scoring_summary({}) is None
    assert "`a` inherited" in SC._scoring_note({"scoring": {"a": "inherited"}})


def test_the_decoy_card_says_which_arms_inherited(monkeypatch):
    seen = {}

    def fake_engine(run, peaks, sample_id, adducts, log=lambda *a: None, scoring=None):
        seen[sample_id] = scoring
        return pd.DataFrame()

    def no_counts(led, sid):           # only the arms' scoring is under test: an arm that cannot be counted is
        raise RuntimeError("stub")      # an error the card records, and the card is still returned
    monkeypatch.setattr(SC, "run_engine_offline", fake_engine)
    monkeypatch.setattr(SC, "raw_peaks_of", lambda run, f: _table())
    monkeypatch.setattr(SC, "_ledger_counts", no_counts)
    # f9: a malformed snapshot of a file that is not a decoy file breaks nothing
    run = _run({"pattern_scoring": {"f1": SNAP, "f3": dict(SNAP, sigma_ppm=float("nan")),
                                    "f9": dict(SNAP, sigma_ppm="abc", sigma_source="fitted")}})
    run.per_file = pd.DataFrame({"__file": ["f1", "f2", "f3"], "height": [3.0, 2.0, 1.0]})
    out = SC.decoy(run, "control", 0.0, 3)
    assert seen == {"f1-control": SNAP, "f2-control": None, "f3-control": None}
    # a snapshot that cannot be inherited is labelled as what its arms got: the class fallback
    assert out["scoring"] == {"f1": "inherited", "f2": "class-fallback", "f3": "class-fallback"}
    assert out["scoring_detail"] == {"f1": {"sigma_ppm": 0.7123, "mu_ppm": -1.25, "mz_tolerance_ppm": 5.0,
                                            "abundance_floor": 0.02, "fitted_anchors": 23, "sigma_source": "fitted"}}
    # a kept manifest from before the field existed re-counts as unrecorded, per file
    kept = SC.decoy(run, "control", 0.0, 2, ledgers_dir="/nonexistent-kept")
    assert kept["scoring"] == {"f1": "unrecorded", "f2": "unrecorded"}


# --------------------------------------------------------------------------- refute round 5: every field, every rule
TOF = {"score_version": IO.SCORE_VERSION, "sigma_ppm": 3.2, "mu_ppm": -0.65, "sigma_source": "fitted",
       "mu_source": "fitted", "fitted_anchors": 29, "mz_tolerance_ppm": 15.0, "abundance_floor": 0.02,
       "instrument_type": "tof", "has_signal_to_noise": True}
PS = dict(sigma_ppm=0.7, mu_ppm=-1.25, mz_tolerance_ppm=15.0, abundance_floor=0.02)


def _t(snr=None):
    t = pd.DataFrame({"peak_id": ["a", "b"], "mz": [100.0, 200.0], "height": [1000.0, 500.0]})
    return t if snr is None else t.assign(signal_to_noise=snr)


def test_a_tof_snapshots_window_and_floor_are_inherited():
    IO.register_offline_sample(SID, _t(), scoring=TOF)
    assert IO.scoring_for_sample(None, SID).mz_tolerance_ppm == 15.0
    snap = IO.scoring_snapshot(None, SID)
    assert snap["mz_tolerance_ppm"] == 15.0 and snap["abundance_floor"] == 0.02


def test_a_snapshot_without_a_floor_is_the_librarys_default():
    IO.register_offline_sample(SID, _t(), scoring={k: v for k, v in TOF.items() if k != "abundance_floor"})
    assert IO.scoring_for_sample(None, SID).abundance_floor == PatternScoring().abundance_floor


@pytest.mark.parametrize("field", ["sigma_ppm", "mu_ppm", "mz_tolerance_ppm", "abundance_floor"])
@pytest.mark.parametrize("bad", [True, False, "0.5", float("nan"), float("inf"), 10 ** 400],
                         ids=["True", "False", "text", "nan", "inf", "overflow"])
@pytest.mark.parametrize("kind", ["snapshot", "PatternScoring"])
def test_every_field_refuses_what_is_not_a_finite_number(kind, field, bad):
    scoring = dict(TOF, **{field: bad}) if kind == "snapshot" else PatternScoring(**dict(PS, **{field: bad}))
    with pytest.raises((ValueError, TypeError)):
        IO.register_offline_sample(SID, _t(), scoring=scoring)
    if kind == "snapshot":                         # and the card's label agrees: no snapshot to inherit
        assert SC.decoy_scoring(_run({"pattern_scoring": {"f": scoring}}), "f") is None


def test_an_unset_floor_of_a_pattern_scoring_is_refused():
    with pytest.raises((ValueError, TypeError)):
        IO.register_offline_sample(SID, _t(), scoring=PatternScoring(**dict(PS, abundance_floor=None)))


@pytest.mark.parametrize("kind", ["snapshot", "PatternScoring"])
def test_a_negative_width_is_refused_and_a_zero_floor_is_not(kind):
    mk = (lambda **k: dict(TOF, **k)) if kind == "snapshot" else (lambda **k: PatternScoring(**dict(PS, **k)))
    with pytest.raises(ValueError):
        IO.register_offline_sample(SID, _t(), scoring=mk(sigma_ppm=-0.5))
    IO.register_offline_sample(SID, _t(), scoring=mk(abundance_floor=0.0))
    assert IO.scoring_for_sample(None, SID).abundance_floor == 0.0


def test_a_class_is_recorded_normalised():
    IO.register_offline_sample(SID, _t(), scoring=" ORBI ")
    assert IO.scoring_snapshot(None, SID)["instrument_type"] == "orbi"


def test_registering_again_without_a_scoring_forgets_the_old_one():
    IO.register_offline_sample(SID, _t(), scoring=TOF)
    IO.register_offline_sample(SID, _t())
    assert IO.scoring_snapshot(None, SID)["sigma_source"] == "instrument_class"


def test_an_all_null_snr_column_is_no_signal_to_noise():
    """The server sends the column with nulls for a file that stores none."""
    IO.register_offline_sample(SID, _t(snr=[float("nan"), None]), scoring=TOF)
    assert IO.scoring_snapshot(None, SID)["has_signal_to_noise"] is False


@pytest.mark.parametrize("ps", [{"f": "orbi"}, [TOF], "garbled", 5])
def test_a_pattern_scoring_that_is_not_a_dict_of_dicts_is_no_snapshot(ps):
    assert SC.decoy_scoring(_run({"pattern_scoring": ps}), "f") is None


def test_the_note_names_every_item_an_arm_does_not_inherit_and_a_non_fitted_width():
    det = dict({k: TOF[k] for k in ("sigma_ppm", "mu_ppm", "mz_tolerance_ppm", "abundance_floor", "fitted_anchors")},
               sigma_source="instrument_class")
    note = SC._scoring_note({"scoring": {"f": "inherited"}, "scoring_detail": {"f": det}})
    for item in ("opportunistic channels", "height cutoff", "prior offset", "occurrence table", "time series",
                 "reference lists", "corroboration", "abundance floor"):
        assert item in note, item
    assert "the run's own instrument_class width" in note
    small = SC._scoring_note({"scoring": {"f": "inherited"}, "scoring_detail": {"f": dict(det, abundance_floor=0.004)}})
    assert "floor 0.004" in small                  # three significant figures, not 0.00


def test_an_omitted_floor_is_recorded_as_the_one_the_arm_got(monkeypatch):
    monkeypatch.setattr(SC, "run_engine_offline", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(SC, "raw_peaks_of", lambda run, f: _t())
    monkeypatch.setattr(SC, "_ledger_counts", lambda led, sid: (_ for _ in ()).throw(RuntimeError("stub")))
    run = _run({"pattern_scoring": {"f1": {k: v for k, v in TOF.items() if k != "abundance_floor"}}})
    run.per_file = pd.DataFrame({"__file": ["f1"], "height": [1.0]})
    out = SC.decoy(run, "control", 0.0, 1)
    assert out["scoring_detail"]["f1"]["abundance_floor"] == PatternScoring().abundance_floor
