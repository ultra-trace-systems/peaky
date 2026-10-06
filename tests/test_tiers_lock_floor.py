"""The score floor on locked readings (tiers.LOCKED_SCORE_METHODS).

A pass-0 known species and a pass-7 certified neutral earn their tier from a
lock (the list identity; the channels' convergent neutral mass), and the tier
engine used to take the lock's word whatever the ion's own match score: a
known species was Assigned by its own branch before the Low/Suspect one was
asked, and a certified commit carries 'Good (certified)' with no score floor.
On a spectrum shifted a few ppm off its true formulas both paths Assigned wrong
readings at ion scores down to 0.001. Under the engine's own Suspect band edge
(PassConfig.tau_suspect) such a commit is Candidate.

Offline.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from peaky.assignment import ledger as L  # noqa: E402
from peaky.assignment import tiers as T  # noqa: E402
from peaky.assignment.passes.config import PassConfig  # noqa: E402
from peaky.batch.assign_batch import align  # noqa: E402

PEAKS = pd.DataFrame({
    "peak_id": ["K1", "K2", "K3", "K4", "C1", "C2", "R1", "S1"],
    "mz": [276.9714, 312.9731, 212.9792, 162.9825, 183.1005, 243.1325, 271.0789, 93.0910],
    "height": [130.0, 72.0, 9.0e4, 3.0e3, 900.0, 800.0, 400.0, 5.0e4]})


def _ledger(scores: dict | None = None) -> pd.DataFrame:
    s = {"K1": 0.034, "K2": 0.52, "K3": 0.996, "K4": 0.50, "C1": 0.41, "C2": 0.86, "R1": 0.14}
    s.update(scores or {})
    led = L.new_ledger(PEAKS)
    # pass-0 known species: a Low lock far under the floor, a Low lock just over
    # it, a Good lock, and one exactly at the edge
    for pid, nf, ad, ion, conf in (
            ("K1", "C4HF7O2", "[M+^NO3]-", "C4HF7O2.^NO3-", "Low (perfluoroacid)"),
            ("K2", "C6HF11O2", "[M-H]-", "C6F11O2-", "Low (perfluoroacid)"),
            ("K3", "C4HF7O2", "[M-H]-", "C4F7O2-", "Good (perfluoroacid)"),
            ("K4", "C3HF5O2", "[M-H]-", "C3F5O2-", "Low (perfluoroacid)")):
        L.commit_assignment(led, pid, neutral_formula=nf, adduct=ad, ion_formula=ion,
                            ion_score=s[pid], compound_score=s[pid], ppm_error=0.1, pass_no=0,
                            method="known:perfluoroacid", confidence=conf,
                            commentary=f"Pass 0 (known perfluoroacid): {nf} {ad}")
    L.lock_peaks(led, ["K1", "K2", "K3", "K4"])
    # a pass-7 certificate on two channels of one P neutral: one member under the
    # floor, one well over it (the floor is the member's own score)
    for pid, ad, ion in (("C1", "[M+H]+", "C4H16N4O2P+"),
                         ("C2", "[M+(CH4N2O)H]+", "C5H20N6O3P+")):
        L.commit_assignment(led, pid, neutral_formula="C4H15N4O2P", adduct=ad, ion_formula=ion,
                            ion_score=s[pid], compound_score=s[pid], ppm_error=0.3, pass_no=7,
                            method="certified:multi-channel", confidence="Good (certified)",
                            commentary="Pass 7 (certified-neutral)")
    # the isotope-confirmed recovery path of a chlorinated paraffin: 'Good' on its
    # 37Cl envelope, whatever the server scored
    L.commit_assignment(led, "R1", neutral_formula="C12H23Cl3", adduct="[M-H]-",
                        ion_formula="C12H22Cl3-", ion_score=s["R1"], ppm_error=0.4, pass_no=0,
                        method="known:chlorinated_paraffin",
                        confidence="Good (chlorinated-paraffin, recovered)",
                        commentary="Pass 0 (known chlorinated-paraffin, RECOVERED)")
    # a source-solvent cluster: never scored (ion_score 0 by construction), its
    # gate is the ladder
    L.commit_assignment(led, "S1", neutral_formula="C2H6O", adduct="[M+C2H6O+H]+",
                        ion_formula="C4H13O2+", ion_score=0.0, ppm_error=0.2, pass_no=0,
                        method="known:solvent_cluster", confidence="Good (solvent cluster)",
                        commentary="Source-solvent cluster")
    return led


def _tiers(led, cfg=None) -> dict:
    out = T.compute_tiers(led, cfg=cfg)
    return {r.peak_id: (r.tier, r.tier_reason) for r in out.itertuples()}


class TestTheFloor:
    def test_the_floor_is_the_engines_suspect_band_edge(self):
        assert T.lock_score_floor(None) == PassConfig().tau_suspect == pytest.approx(0.50)
        assert T.lock_score_floor(PassConfig(tau_suspect=0.3)) == pytest.approx(0.3)

    def test_a_known_species_lock_under_the_floor_is_candidate(self):
        t = _tiers(_ledger())
        assert t["K1"][0] == T.TIER_CANDIDATE
        assert t["K1"][1].startswith("known species (pass-0 locked list) with ion score 0.03")
        assert "Suspect band edge (0.50)" in t["K1"][1]
        # the recovered chlorinated paraffin is a known: lock like any other
        assert t["R1"][0] == T.TIER_CANDIDATE and "ion score 0.14" in t["R1"][1]

    def test_a_known_species_lock_at_or_over_the_floor_keeps_its_branch(self):
        t = _tiers(_ledger())
        for pid in ("K2", "K3", "K4"):        # 0.52 (Low), 0.996 (Good), 0.50 exactly
            assert t[pid][0] == T.TIER_ASSIGNED, (pid, t[pid])
            assert t[pid][1].startswith("known species (pass-0 locked list, mass + own-twin"), t[pid]

    def test_a_certified_member_under_the_floor_is_candidate_its_sibling_is_not(self):
        t = _tiers(_ledger())
        assert t["C1"][0] == T.TIER_CANDIDATE
        assert t["C1"][1].startswith("certified neutral (pass-7 multi-channel certificate) with ion score 0.41")
        # the other channel of the same certificate scores 0.86: still Assigned on the
        # second channel, as before
        assert t["C2"][0] == T.TIER_ASSIGNED and "second ionization channel" in t["C2"][1]

    def test_a_certified_ladder_rung_is_floored_too(self):
        led = _ledger()
        led.loc[led["peak_id"] == "C2", "method"] = "certified:ladder-rung"
        led.loc[led["peak_id"] == "C2", "ion_score"] = 0.2
        t = _tiers(led)
        assert t["C2"][0] == T.TIER_CANDIDATE and "ion score 0.20" in t["C2"][1]

    def test_a_solvent_cluster_is_exempt(self):
        t = _tiers(_ledger())
        assert t["S1"][0] == T.TIER_ASSIGNED
        assert t["S1"][1].startswith("source-solvent cluster ion with no possible covalent reading")

    def test_the_floor_follows_the_cfg(self):
        t = _tiers(_ledger(), cfg=PassConfig(tau_suspect=0.3))
        assert t["C1"][0] == T.TIER_ASSIGNED          # 0.41 clears a 0.3 edge
        assert t["K1"][0] == T.TIER_CANDIDATE and "Suspect band edge (0.30)" in t["K1"][1]

    def test_a_lock_with_no_recorded_score_is_not_floored(self):
        """No number, no judgment: a ledger written without the commit's score (an
        old CSV, a hand-built frame) keeps the lock's own branch."""
        led = _ledger()
        led.loc[led["peak_id"] == "K1", "ion_score"] = np.nan
        assert _tiers(led)["K1"][0] == T.TIER_ASSIGNED

    def test_rows_off_the_locked_paths_are_untouched(self):
        """A grid commit at the same low score is judged by its own branches, as before."""
        led = _ledger()
        i = led.index[led["peak_id"] == "K1"][0]
        led.at[i, "method"] = "cheminfo+grid"
        led.at[i, "confidence"] = "Good"
        t = _tiers(led)
        assert "ion score" not in str(t["K1"][1])

    def test_a_csv_round_trip_gives_the_same_verdicts(self, tmp_path):
        led = _ledger()
        path = tmp_path / "ledger.csv"
        led.to_csv(path, index=False)
        assert _tiers(pd.read_csv(path, low_memory=False)) == _tiers(led)

    def test_apply_tiers_stamps_the_floor(self):
        led = T.apply_tiers(_ledger())
        by = led.set_index("peak_id")
        assert by.at["K1", "tier"] == T.TIER_CANDIDATE and by.at["C1", "tier"] == T.TIER_CANDIDATE
        assert by.at["K3", "tier"] == T.TIER_ASSIGNED and by.at["S1", "tier"] == T.TIER_ASSIGNED


def test_a_reading_locked_under_the_floor_in_every_file_is_candidate_at_the_merge():
    """The batch's merged tier is the winning reading's best per-file tier: a
    known-species reading whose every file holds it under the floor loses
    Assigned, one that some file holds over the floor keeps it."""
    per_file = {}
    for src, (s_low, s_mix) in {"f1": (0.03, 0.80), "f2": (0.45, 0.45)}.items():
        led = T.apply_tiers(_ledger({"K1": s_low, "K2": s_mix}))
        per_file[src] = led[led["role"] == "M0"]
    merged, _jit = align(per_file)
    by = merged.set_index(merged["neutral_formula"] + " " + merged["adduct"])
    assert by.at["C4HF7O2 [M+^NO3]-", "tier"] == T.TIER_CANDIDATE
    assert by.at["C6HF11O2 [M-H]-", "tier"] == T.TIER_ASSIGNED
