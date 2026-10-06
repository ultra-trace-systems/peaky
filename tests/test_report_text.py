"""The PDF report's data-driven sentences say only what the run's data support.

Pure helpers on synthetic numbers (no run folder, no network)."""
import numpy as np

from peaky.batch import cluster as CL
from peaky.reporting import pdf_report as R


# --------------------------------------------------------------------------- the event sentence
def _hours(n):
    return np.linspace(0.0, 48.0, n)


def test_a_bump_below_the_transient_bar_is_no_transient_and_gives_the_band():
    tt = np.full(120, 100.0)
    tt[30] = 131.0                                    # 1.31x: under the 1.7x bar
    tt[60:70] = 70.0                                  # a dip the band must show
    s = R._event_sentence(_hours(120), tt)
    assert s.startswith("No transient event")
    assert "1.31x" in s and f"{CL.PEAK_RANGE:.1f}x transient bar" in s
    assert "transient event)" not in s and "decays" not in s
    assert "within 0.70-1.00x" in s


def test_a_peak_over_the_bar_that_decays_is_a_transient_event():
    tt = np.full(120, 100.0)
    tt[30:36] = [150.0, 250.0, 300.0, 240.0, 180.0, 130.0]
    s = R._event_sentence(_hours(120), tt)
    assert "a transient event" in s and "3.0x the late-run baseline" in s
    assert s.startswith(f"Total signal peaks at hour {_hours(120)[32]:.1f}")


def test_a_maximum_in_the_last_tenth_of_the_samples_is_a_rise_the_run_ends_on():
    tt = np.full(120, 100.0)
    tt[-3:] = [150.0, 200.0, 260.0]                   # over the bar, but at the very end
    s = R._event_sentence(_hours(120), tt)
    assert "transient event" not in s and "rise the run ends on" in s
    # the same late maximum under the bar is simply no transient
    tt[-3:] = [105.0, 108.0, 110.0]
    assert R._event_sentence(_hours(120), tt).startswith("No transient event")


def test_the_bar_is_the_reports_own_and_too_few_samples_say_nothing():
    tt = np.full(60, 100.0)
    tt[10] = 160.0                                    # 1.6x: over a 1.5x bar, under 1.7x
    assert "a transient event" in R._event_sentence(_hours(60), tt, bar=1.5)
    assert R._event_sentence(_hours(60), tt).startswith("No transient event")
    assert R._event_sentence(_hours(2), np.array([1.0, 2.0])) is None
    assert R._event_sentence(_hours(5), np.zeros(5)) is None
