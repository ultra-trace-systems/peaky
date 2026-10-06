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


# --------------------------------------------------------------------------- the reference-list chance level
class _List:
    """The two attributes reflists.match_by_mass reads off a list."""
    id = "toy"

    def __init__(self, formulas):
        self._f = set(formulas)

    def pool(self, include_radicals=False):
        return set(self._f)


def test_the_chance_null_reruns_the_rescue_match_on_shifted_masses():
    from peaky.chem import chemistry as C
    toy = _List({"C10H16O4", "C8H12O5", "C12H20O6"})
    obs = [C.ion_mz(f, "[M-H]-") for f in ("C10H16O4", "C8H12O5")]
    null = R.reflist_chance(obs, [toy], ["[M-H]-"])
    assert set(null) == {-40.0, -25.0, -15.0, 15.0, 25.0, 40.0}
    assert all(v == 0 for v in null.values())          # a sparse list: nothing by chance
    assert R.reflist_chance(obs, [toy], ["[M-H]-"], shifts=(0.0,)) == {0.0: 2}
    s = R._reflist_chance_text(2, null)
    assert "±15-40 ppm give 0-0 matches within 1 ppm" in s and "exceed that chance level" in s


def test_the_chance_sentence_says_at_or_below_chance_when_the_null_matches_more():
    tof = {-40.0: 238, -25.0: 260, -15.0: 296, 15.0: 273, 25.0: 272, 40.0: 256}
    s = R._reflist_chance_text(110, tof)
    assert "give 238-296 matches" in s and "110 observed" in s
    assert "below that chance level" in s and "no evidence by itself" in s
    assert "within that chance level" in R._reflist_chance_text(250, tof)
    orbi = {-40.0: 50, -25.0: 44, -15.0: 103, 15.0: 88, 25.0: 49, 40.0: 74}
    s = R._reflist_chance_text(261, orbi)
    assert "exceed that chance level (4.2x its median)" in s
    assert R._reflist_chance_text(5, {}) == ""
