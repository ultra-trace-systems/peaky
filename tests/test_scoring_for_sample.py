"""What the reference scores at when the sample measured no offset.

`scoring_for_sample` reads the sample's anchors - the peaks Mascope itself
attributed to a known species - and asks the shared library to fit them. The
library answers `None` for an offset it did not measure, which is a different
fact from an offset of zero, and this module is where that None has to stop:
a PatternScoring carries a number.

Offline: the client is a stub and nothing here touches a network.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from peaky.io import io_mascope  # noqa: E402


class _Samples:
    def get(self, sample_id):
        # A TOF file, so the fallback width and the matching window are the
        # TOF class's rather than an Orbitrap's.
        return {"filename": "KLTOF2_x_2026.03.30-16h59m12s", "instrument": "KLTOF2"}


class _Client:
    def __init__(self, peaks):
        self._peaks = peaks
        self.samples = _Samples()


def _peaks(n_anchors: int) -> pd.DataFrame:
    """A peak frame carrying `n_anchors` server matches peaky can read."""
    rows = []
    for i in range(n_anchors):
        rows.append(
            {
                "peak_id": f"p{i}",
                "mz": 78.91834 + i,
                "intensity": 1000.0,
                "signal_to_noise": 50.0,
                "target_compound_formula": "HBr",
                "target_isotope_formula": "HBr",
                "ionization_mechanism": "-H+",
            }
        )
    # One unmatched peak, so the frame is never empty even with no anchors.
    rows.append({"peak_id": "px", "mz": 100.0, "intensity": 10.0,
                 "signal_to_noise": 3.0, "target_compound_formula": None,
                 "target_isotope_formula": None, "ionization_mechanism": None})
    return pd.DataFrame(rows)


@pytest.fixture(autouse=True)
def _no_cache():
    """The fit is cached per sample; these cases are all the same sample."""
    io_mascope._SCORING_CACHE.clear()
    yield
    io_mascope._SCORING_CACHE.clear()


class TestAnOffsetTheSampleNeverMeasured:
    def test_a_sample_with_no_anchors_still_yields_a_scoring(self):
        """A file whose matches were just cleared has no anchors at all.

        This is not hypothetical: applying a calibration removes the sample's
        matches, and the reference run that follows it reads a frame with none.
        Before, the library's None reached `float()` and the run died there.
        """
        scoring = io_mascope.scoring_for_sample(
            _Client(_peaks(0)), "s1", peaks=_peaks(0)
        )
        assert scoring.mu_ppm == 0.0
        assert scoring.sigma_ppm > 0

    def test_it_says_the_offset_was_assumed_rather_than_measured(self):
        io_mascope.scoring_for_sample(_Client(_peaks(0)), "s2", peaks=_peaks(0))
        snapshot = io_mascope.scoring_snapshot(_Client(_peaks(0)), "s2")
        assert snapshot["mu_source"] == "assumed_zero"
        assert snapshot["fitted_anchors"] == 0

    def test_enough_anchors_still_measure_the_offset(self):
        """The rule above must not swallow an offset the sample does show."""
        from peaky.chem import chemistry as C

        # Six real species, every one sitting a clear 2 ppm high: too few
        # anchors for the library to fit a width, enough for peaky's own
        # offset rule. Built from the theoretical masses rather than from
        # round numbers, because an anchor further out than the instrument's
        # matching window is correctly not an anchor at all.
        formulas = ["HBr", "HNO3", "H2SO4", "CH2O2", "C2H4O2", "C3H6O3"]
        rows = [
            {
                "peak_id": f"a{i}",
                "mz": C.ion_mz(formula, "[M-H]-") * (1 + 2e-6),
                "intensity": 1000.0,
                "signal_to_noise": 50.0,
                "target_compound_formula": formula,
                "target_isotope_formula": formula,
                "ionization_mechanism": "-H+",
            }
            for i, formula in enumerate(formulas)
        ]
        peaks = pd.DataFrame(rows)
        scoring = io_mascope.scoring_for_sample(_Client(peaks), "s3", peaks=peaks)
        assert scoring.mu_ppm == pytest.approx(2.0, abs=0.1)
