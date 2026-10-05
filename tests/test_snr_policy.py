"""The signal-to-noise the score reads is judged before it is believed (C46).

On one TOF the server's `signal_to_noise` column is not a signal-to-noise: over
a file's peaks it does not track height (a 478-count peak carries 1.1, a 2-count
peak 12), where every Orbitrap file gives Spearman 0.999. Read as an SNR it
excuses every missing line of a bright ion and charges dim ions for lines they
could never show. A table whose column fails the test is scored at the
counting-statistics SNR h / sqrt(h + edge^2) instead.

Offline: a stub client, no network.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mascope_tools.composition import PatternScoring  # noqa: E402
from mascope_tools.composition.heuristic_filter import (  # noqa: E402
    anchor_on_monoisotopic,
    predict_isotopes,
)

from peaky.io import io_mascope  # noqa: E402
from peaky.io import local_scoring as LS  # noqa: E402


def _table(n=200, *, tracking: bool, seed=0, height_col="height") -> pd.DataFrame:
    """`n` picked peaks. `tracking`: SNR = height / 20 (an Orbitrap's flat ~20 cps
    noise); otherwise an SNR that has nothing to do with height (the TOF's)."""
    rng = np.random.default_rng(seed)
    h = np.exp(rng.uniform(np.log(1.0), np.log(5000.0), n))
    snr = h / 20.0 if tracking else rng.uniform(0.5, 2.0, n)
    return pd.DataFrame({"peak_id": [f"p{i}" for i in range(n)],
                         "mz": np.sort(rng.uniform(60, 600, n)), height_col: h,
                         "signal_to_noise": snr})


class TestAssessSnr:
    def test_a_column_that_tracks_height_is_the_servers_own(self):
        a = LS.assess_snr(_table(tracking=True))
        assert a["source"] == LS.SNR_SOURCE_SERVER
        assert a["spearman"] > 0.99 and a["n"] == 200
        assert a["edge"] == pytest.approx(np.percentile(_table(tracking=True)["height"], 1))

    def test_a_column_unrelated_to_height_is_not_a_signal_to_noise(self):
        a = LS.assess_snr(_table(tracking=False))
        assert a["source"] == LS.SNR_SOURCE_POISSON
        assert abs(a["spearman"]) < LS.SNR_MIN_SPEARMAN
        assert a["edge"] is not None and a["edge"] > 0

    def test_too_few_peaks_to_judge_keep_the_column(self):
        a = LS.assess_snr(_table(n=LS.SNR_ASSESS_MIN_PEAKS - 1, tracking=False))
        assert a["source"] == LS.SNR_SOURCE_SERVER and a["spearman"] is None
        assert a["n"] == LS.SNR_ASSESS_MIN_PEAKS - 1

    def test_no_column_or_no_value_is_the_no_snr_mode(self):
        t = _table(tracking=True)
        assert LS.assess_snr(t.drop(columns=["signal_to_noise"]))["source"] == LS.SNR_SOURCE_NONE
        t["signal_to_noise"] = np.nan
        assert LS.assess_snr(t)["source"] == LS.SNR_SOURCE_NONE
        assert LS.assess_snr(_table(tracking=True, height_col="intensity"))["source"] == LS.SNR_SOURCE_NONE

    def test_one_peak_per_peak_id(self):
        """The raw server table has one row per match; a peak counted five
        times must not weigh five times in the correlation."""
        t = _table(n=40, tracking=False)
        dup = pd.concat([t, t.iloc[:5]] * 4, ignore_index=True)
        assert LS.assess_snr(dup)["n"] == 40


class TestPoissonSnr:
    def test_the_counting_statistics_formula(self):
        out = LS.poisson_snr([478.0, 4.0, 0.0, np.nan, -1.0], 0.8)
        assert out[0] == pytest.approx(478.0 / np.sqrt(478.0 + 0.64))
        assert out[1] == pytest.approx(4.0 / np.sqrt(4.0 + 0.64))
        assert np.isnan(out[2]) and np.isnan(out[3]) and np.isnan(out[4])

    def test_no_edge_is_sqrt_h(self):
        assert LS.poisson_snr([100.0], None)[0] == pytest.approx(10.0)
        assert LS.poisson_snr([100.0], 0.0)[0] == pytest.approx(10.0)

    def test_with_poisson_snr_replaces_only_that_column(self):
        t = _table(n=50, tracking=False)
        out = LS.with_poisson_snr(t, 0.7)
        assert list(out.columns) == list(t.columns)
        assert (out["height"] == t["height"]).all()
        np.testing.assert_allclose(out["signal_to_noise"], LS.poisson_snr(t["height"], 0.7))
        assert not (out["signal_to_noise"] == t["signal_to_noise"]).any()


def _bromide_spectrum(*, base_height: float, snr_value: float, drop: set, filler: int = 40):
    """The [M+Br]- ion of C10H16O6 with the lines in `drop` left out, every peak
    carrying the server's `snr_value`, plus `filler` dim far-away peaks so the
    table can be judged. The 81Br line (0.97x) is the one a bromide cluster
    must show."""
    mzs, ints, labels = anchor_on_monoisotopic(*predict_isotopes("C10H16O6Br", -1))
    rel = ints / ints[0]
    rows = []
    for i, (m, r, lab) in enumerate(zip(mzs, rel, labels)):
        if lab in drop or r < 0.01:
            continue
        rows.append({"peak_id": f"L{i}", "mz": float(m), "height": base_height * float(r),
                     "signal_to_noise": snr_value})
    rng = np.random.default_rng(1)
    for j in range(filler):
        rows.append({"peak_id": f"f{j}", "mz": 60.0 + j * 3.1, "height": float(rng.uniform(0.8, 4.0)),
                     "signal_to_noise": snr_value})
    return pd.DataFrame(rows), labels


TOF = PatternScoring(sigma_ppm=4.0, mu_ppm=0.0, mz_tolerance_ppm=15.0)


def _score(peaks):
    out = LS.score_candidates_local(peaks, ["C10H16O6"], ["[M+Br]-"], scoring=TOF)
    base = out[out["is_base"]]
    assert len(base) == 1
    return float(base["ion_score"].iloc[0])


class TestWhatTheFallbackChanges:
    def test_a_bright_bromide_cluster_with_no_81br_line_is_excused_by_the_bogus_column(self):
        peaks, labels = _bromide_spectrum(base_height=500.0, snr_value=1.1, drop={"81Br"})
        assert "81Br" in set(labels)
        with_server = _score(peaks)
        a = LS.assess_snr(peaks)
        assert a["source"] == LS.SNR_SOURCE_POISSON  # a flat column over 40+ peaks
        with_fallback = _score(LS.with_poisson_snr(peaks, a["edge"]))
        # at "SNR" 1.1 the 0.97x line is under the detectability gate: nothing is
        # charged and the reading scores as if the line were there
        assert with_server >= LS.PROBABLE_THRESHOLD
        # at the counting-statistics SNR (~22 at 500 counts) the missing line is
        # charged at its weight: a bromide cluster without its 81Br line is not Good
        assert with_fallback < LS.PROBABLE_THRESHOLD
        assert with_fallback < with_server

    def test_a_dim_ion_is_no_longer_charged_for_a_line_it_could_not_show(self):
        """The other face of the column: dim peaks carry HIGH values, so their
        missing lines are charged although they were never within reach."""
        peaks, _ = _bromide_spectrum(base_height=2.0, snr_value=35.0, drop={"81Br"})
        with_server = _score(peaks)
        with_fallback = _score(LS.with_poisson_snr(peaks, 0.7))
        assert with_server < LS.PROBABLE_THRESHOLD
        assert with_fallback > with_server
        assert with_fallback >= LS.PROBABLE_THRESHOLD  # the tier pass's floor decides such a row

    def test_a_complete_envelope_scores_the_same_either_way(self):
        peaks, _ = _bromide_spectrum(base_height=500.0, snr_value=1.1, drop=set())
        a = LS.assess_snr(peaks)
        assert abs(_score(peaks) - _score(LS.with_poisson_snr(peaks, a["edge"]))) < 0.02


class _Samples:
    def get(self, sample_id):
        return {"filename": "KLTOF2_x_2026.03.30-16h59m12s", "instrument": "KLTOF2"}


class _Client:
    def __init__(self):
        self.samples = _Samples()


@pytest.fixture(autouse=True)
def _no_cache():
    io_mascope._SCORING_CACHE.clear()
    yield
    io_mascope._SCORING_CACHE.clear()


class TestTheSnapshotAndTheTableTheScorerReads:
    def test_the_snapshot_records_the_fallback_and_peaks_for_scoring_applies_it(self):
        peaks = _table(tracking=False)
        io_mascope.scoring_for_sample(_Client(), "tof1", peaks=peaks)
        snap = io_mascope.scoring_snapshot(_Client(), "tof1", peaks=peaks)
        assert snap["snr_source"] == io_mascope.SNR_SOURCE_POISSON
        assert snap["snr_n"] == 200 and abs(snap["snr_spearman"]) < 0.5
        assert snap["snr_edge"] == pytest.approx(np.percentile(peaks["height"], 1))
        out = io_mascope.peaks_for_scoring("tof1", peaks)
        np.testing.assert_allclose(out["signal_to_noise"], LS.poisson_snr(peaks["height"], snap["snr_edge"]))

    def test_a_tracking_column_is_left_alone(self):
        peaks = _table(tracking=True)
        io_mascope.scoring_for_sample(_Client(), "orbi1", peaks=peaks)
        snap = io_mascope.scoring_snapshot(_Client(), "orbi1", peaks=peaks)
        assert snap["snr_source"] == io_mascope.SNR_SOURCE_SERVER
        assert io_mascope.peaks_for_scoring("orbi1", peaks) is peaks

    def test_an_unjudged_sample_reads_its_table_as_it_is(self):
        peaks = _table(tracking=False)
        assert io_mascope.peaks_for_scoring("never-scored", peaks) is peaks

    def test_a_stand_in_judges_its_own_table(self):
        """An offline sample inherits another sample's width and offset; whether
        its own peaks carry a signal-to-noise -- and whether that is one -- is
        this table's own question."""
        given = {"sigma_ppm": 4.0, "mu_ppm": 0.0, "mz_tolerance_ppm": 15.0, "instrument_type": "tof",
                 "sigma_source": "fitted", "mu_source": "fitted", "fitted_anchors": 40}
        _, snap = io_mascope._inherited_scoring(given, _table(tracking=False))
        assert snap["snr_source"] == io_mascope.SNR_SOURCE_POISSON
        _, snap2 = io_mascope._inherited_scoring(given, _table(tracking=True))
        assert snap2["snr_source"] == io_mascope.SNR_SOURCE_SERVER
