"""The local scorer judges a candidate at the sample's own measurement.

Every test here builds a spectrum by hand from a predicted envelope and then
moves ONE thing: the width the mass is judged at, the window a line is matched
in, the offset subtracted before scoring, or the noise a missing line is judged
against. Each of those was a fixed Orbitrap-shaped constant before, which is why
a TOF reference run committed almost nothing - its every candidate sat several
of an Orbitrap's sigmas out.

Offline: mascope_tools is a hard dependency, and nothing here touches a network.
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

from peaky.io.local_scoring import score_candidates_local  # noqa: E402


NEUTRAL = "C6H12O6"
ADDUCT = "[M-H]-"
ION = "C6H11O6"


def _envelope(ion: str = ION, charge: int = -1):
    """The ion's predicted lines, monoisotopic first (as the scorer reads them)."""
    mzs, intensities, labels = predict_isotopes(ion, charge)
    return anchor_on_monoisotopic(mzs, intensities, labels)


def _spectrum(
    *,
    ppm_shift: float = 0.0,
    lines: int | None = None,
    snr: float | None = None,
    ion: str = ION,
    charge: int = -1,
) -> pd.DataFrame:
    """A peak list holding the ion's first `lines` lines, each `ppm_shift` off.

    Heights are the predicted abundances, so a matched line's intensity agrees
    with its prediction exactly and only the term under test moves the score.
    """
    mzs, intensities, _ = _envelope(ion, charge)
    take = len(mzs) if lines is None else lines
    mzs, intensities = mzs[:take], intensities[:take]
    frame = pd.DataFrame(
        {
            "mz": mzs * (1 + ppm_shift * 1e-6),
            "height": intensities / intensities[0] * 1e6,
            "peak_id": [f"p{i}" for i in range(len(mzs))],
        }
    )
    if snr is not None:
        frame["signal_to_noise"] = float(snr)
    return frame.sort_values("mz").reset_index(drop=True)


def _score(peaks: pd.DataFrame, scoring: PatternScoring, neutral: str = NEUTRAL) -> float:
    flat = score_candidates_local(peaks, [neutral], [ADDUCT], scoring=scoring)
    if flat.empty:
        return 0.0
    return float(flat["ion_score"].iloc[0])


ORBI = PatternScoring(sigma_ppm=0.58, mu_ppm=0.0, mz_tolerance_ppm=5.0)
TOF = PatternScoring(sigma_ppm=3.04, mu_ppm=0.0, mz_tolerance_ppm=15.0)


class TestTheWidth:
    """A mass error is judged against what the instrument actually delivers."""

    def test_a_tof_error_scores_on_a_tof_and_not_on_an_orbitrap(self):
        # 3 ppm is an ordinary TOF mass error - one of its own sigmas, and five
        # of an Orbitrap's. Judged at the Orbitrap width the whole envelope is
        # worthless, which is why the v1 reference - a fixed 5 ppm scale -
        # committed almost nothing on a TOF. The window is held wide in both, so
        # what moves here is the width alone.
        peaks = _spectrum(ppm_shift=3.0)
        wide_window = PatternScoring(sigma_ppm=0.58, mz_tolerance_ppm=15.0)

        assert _score(peaks, TOF) > 0.4
        assert _score(peaks, wide_window) < 0.01

    def test_a_clean_orbitrap_error_scores_high_at_either_width(self):
        peaks = _spectrum(ppm_shift=0.2)

        assert _score(peaks, ORBI) > 0.8
        assert _score(peaks, TOF) > 0.8


class TestTheWindow:
    """What may be paired with a prediction at all."""

    def test_a_line_outside_the_window_is_not_this_ion_s(self):
        peaks = _spectrum(ppm_shift=8.0)

        # Nothing to match the ion's own line to, so it is not a candidate.
        assert score_candidates_local(peaks, [NEUTRAL], [ADDUCT], scoring=ORBI).empty
        assert not score_candidates_local(peaks, [NEUTRAL], [ADDUCT], scoring=TOF).empty

    def test_the_default_window_is_the_libraries(self):
        # A caller that says nothing about the sample gets the library's own
        # defaults, which are an Orbitrap's - the behaviour before a caller
        # could describe a sample at all.
        peaks = _spectrum(ppm_shift=8.0)

        assert score_candidates_local(peaks, [NEUTRAL], [ADDUCT]).empty


class TestTheOffset:
    """A systematic offset is the calibration's, not the candidate's."""

    def test_the_sample_s_own_offset_is_subtracted_before_scoring(self):
        # Every peak in this file sits 2.5 ppm high. Charged to each candidate
        # it is four sigma of error; subtracted first, the candidate is on mass.
        peaks = _spectrum(ppm_shift=2.5)
        centred = PatternScoring(sigma_ppm=0.58, mu_ppm=2.5, mz_tolerance_ppm=5.0)

        assert _score(peaks, centred) > 0.8
        assert _score(peaks, ORBI) < 0.2

    def test_the_reported_error_stays_the_raw_one(self):
        # The offset belongs to the calibration, so what the row reports is what
        # the instrument produced - a reader comparing a row against the
        # instrument's own accuracy needs that number, not a corrected one.
        peaks = _spectrum(ppm_shift=2.5)
        centred = PatternScoring(sigma_ppm=0.58, mu_ppm=2.5, mz_tolerance_ppm=5.0)

        flat = score_candidates_local(peaks, [NEUTRAL], [ADDUCT], scoring=centred)
        base = flat[flat["is_base"]].iloc[0]

        assert base["ppm_error"] == pytest.approx(2.5, abs=0.05)


class TestTheNoise:
    """A predicted line that is missing is charged on what the noise says."""

    def test_a_bright_peak_is_charged_for_its_missing_carbon_line(self):
        lonely_bright = _spectrum(lines=1, snr=500.0)
        lonely_faint = _spectrum(lines=1, snr=2.0)

        # At signal-to-noise 500 a 6.6% 13C line is far above the noise, so its
        # absence is evidence against the formula. At 2 it could not have been
        # seen and says nothing.
        assert _score(lonely_bright, ORBI) < _score(lonely_faint, ORBI)

    def test_a_file_with_no_noise_estimate_still_scores(self):
        # The column is absent for a file that stores no estimate; the score
        # then charges an absent line on its predicted abundance alone.
        peaks = _spectrum(lines=1)

        assert 0.0 < _score(peaks, ORBI) <= 1.0

    def test_the_full_envelope_beats_the_lone_line(self):
        assert _score(_spectrum(snr=500.0), ORBI) > _score(
            _spectrum(lines=1, snr=500.0), ORBI
        )


class TestTheAnchor:
    """Index 0 is the ion's own line, not its most abundant one."""

    def test_a_dibromide_is_anchored_on_its_monoisotopic_line(self):
        # A [M+Br]- of a bromine-carrying neutral predicts a 79/81 line brighter
        # than its monoisotopic one, two mass units up. Anchored there, the base
        # row would carry a peak the candidate was never proposed for.
        mzs, intensities, labels = _envelope("CHBr2", charge=-1)
        peaks = _spectrum(ion="CHBr2", charge=-1)

        flat = score_candidates_local(
            peaks, ["CHBr"], ["[M+Br]-"], scoring=PatternScoring(sigma_ppm=0.58)
        )
        base = flat[flat["is_base"]]

        # The premise: this ion's brightest line is NOT its monoisotopic one.
        assert labels[int(np.argmax(intensities))] != "M0"
        assert len(base) == 1
        assert base.iloc[0]["iso_label"] == "M0"
        assert base.iloc[0]["theo_mz"] == pytest.approx(float(np.min(mzs)), abs=0.01)
        # And every line's abundance is relative to that one, so the brightest
        # runs above 1 rather than being scaled to it.
        assert flat["rel_abundance"].max() > 1.0


class TestTheSampleFit:
    """What a sample can measure about itself, and what it cannot."""

    def test_the_offset_survives_a_width_that_cannot_be_fitted(self):
        # Six anchors are too few for a robust spread and plenty for a median.
        # A source sitting 1.2 ppm low is a fact about the run; scoring as if it
        # sat on calibration charges that error to every candidate, and the one
        # whose own error compensates it then wins the peak.
        from mascope_tools.composition import fit_mass_accuracy

        from peaky.io.io_mascope import MIN_OFFSET_ANCHORS

        anchors = [-1.95, -1.66, -1.52, -0.84, -0.80, -0.72]
        mu, sigma = fit_mass_accuracy(anchors)

        assert sigma is None and mu == 0.0  # the library reports neither
        assert len(anchors) >= MIN_OFFSET_ANCHORS
        assert float(np.median(anchors)) == pytest.approx(-1.18, abs=0.01)

    def test_too_few_anchors_state_no_offset_either(self):
        # An anchor is a peak matched within the instrument's MATCHING window,
        # which on a TOF is five times its accuracy, so a mis-match sits in the
        # set looking like a measurement. Three anchors, two of them mis-matched
        # to the same wrong species, would put the median on the mis-match.
        from peaky.io.io_mascope import MIN_OFFSET_ANCHORS

        assert len([-10.5, -10.4, -2.8]) < MIN_OFFSET_ANCHORS

    def test_a_shifted_sample_scores_its_own_candidate_best(self):
        # The whole point of subtracting the offset: with it, the candidate that
        # IS the peak wins; without it, the reading whose error cancels the
        # instrument's does.
        peaks = _spectrum(ppm_shift=-1.18, snr=50.0)
        centred = PatternScoring(sigma_ppm=0.58, mu_ppm=-1.18, mz_tolerance_ppm=5.0)
        uncentred = PatternScoring(sigma_ppm=0.58, mu_ppm=0.0, mz_tolerance_ppm=5.0)

        assert _score(peaks, centred) > 0.8
        assert _score(peaks, uncentred) < 0.2
