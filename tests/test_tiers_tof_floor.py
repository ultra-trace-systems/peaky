"""The counting-detector floor of the tier pass (C46).

On a TOF an M0 under k_detect x the batch's typical detection edge cannot be
tier Assigned whatever hangs under it: a handful-of-ions centroid has no
testable mass and no testable isotope line, and the kid or series step that
"corroborates" it is itself sub-edge. The floor is the BATCH's because a file's
own edge follows its total ion count -- the two files that produced both
silicon false readings of the finish line had edges 5x under the batch's.

Offline.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from peaky.assignment import ledger as L  # noqa: E402
from peaky.assignment import tiers as T  # noqa: E402
from peaky.assignment.passes.config import PassConfig  # noqa: E402
from peaky.batch.assign_batch import batch_noise_edge  # noqa: E402


def _ledger():
    peaks = pd.DataFrame({
        "peak_id": ["A", "B", "C", "D"],
        "mz": [279.0896, 280.0930, 329.0250, 330.0284],
        "height": [1.5, 0.2, 5.0, 0.6]})
    led = L.new_ledger(peaks)
    for pid, kid, formula, ion in (("A", "B", "C10H24N2Si", "C10H24BrN2Si-"),
                                   ("C", "D", "C10H18O7", "C10H18BrO7-")):
        L.commit_assignment(led, pid, neutral_formula=formula, adduct="[M+Br]-",
                            ion_formula=ion, ion_score=0.97, compound_score=0.97,
                            eff_score=0.95, eff_margin=0.3, tied=False, ppm_error=0.4,
                            pass_no=1, method="cheminfo+grid", confidence="High",
                            commentary="Pass 1",
                            isotopologues=[{"label": "13C", "score": 0.9, "peak_id": kid}])
        L.attach_isotopologue(led, kid, pid, iso_label="13C", iso_match_score=0.9)
    return led


def _tiers(cfg):
    out = T.compute_tiers(_ledger(), cfg=cfg)
    return {r.peak_id: (r.tier, r.tier_reason) for r in out.itertuples()}


class TestTheFloor:
    def test_a_sub_edge_tof_centroid_is_candidate_whatever_corroborates_it(self):
        cfg = PassConfig(instrument_type="tof", noise_edge_batch_cps=0.74, noise_edge_cps=0.126)
        assert T.tof_assign_floor(cfg) == pytest.approx(3 * 0.74)
        t = _tiers(cfg)
        assert t["A"][0] == T.TIER_CANDIDATE
        assert "sub-edge centroid" in t["A"][1] and "batch's detection edge" in t["A"][1]
        assert "1.5 counts" in t["A"][1] and "2.22" in t["A"][1]
        # the same evidence above the floor is untouched
        assert t["C"][0] == T.TIER_ASSIGNED and "sub-edge" not in t["C"][1]

    def test_an_orbitrap_never_sees_the_floor(self):
        cfg = PassConfig(instrument_type="orbi", noise_edge_batch_cps=0.74, noise_edge_cps=0.126)
        assert T.tof_assign_floor(cfg) is None
        t = _tiers(cfg)
        assert t["A"][0] == T.TIER_ASSIGNED and t["C"][0] == T.TIER_ASSIGNED

    def test_without_a_batch_the_files_own_edge_sizes_it(self):
        """A single-sample run of the low-count file: its own edge is all there
        is, and 1.5 counts stand 12x above it -- the floor cannot see what the
        batch would have seen (said so in PassConfig)."""
        cfg = PassConfig(instrument_type="tof", noise_edge_cps=0.126)
        assert T.tof_assign_floor(cfg) == pytest.approx(3 * 0.126)
        t = _tiers(cfg)
        assert t["A"][0] == T.TIER_ASSIGNED
        cfg2 = PassConfig(instrument_type="tof", noise_edge_cps=0.74)
        assert _tiers(cfg2)["A"][0] == T.TIER_CANDIDATE
        assert "file's detection edge" in _tiers(cfg2)["A"][1]

    def test_no_cfg_no_class_or_no_edge_means_no_floor(self):
        assert T.tof_assign_floor(None) is None
        assert T.tof_assign_floor(PassConfig()) is None
        assert T.tof_assign_floor(PassConfig(instrument_type="tof")) is None
        assert T.tof_assign_floor(PassConfig(instrument_type="tof", noise_edge_cps=0.0)) is None
        assert T.tof_assign_floor(PassConfig(instrument_type="tof", noise_edge_cps=float("nan"))) is None

    def test_the_multiple_is_the_scorers_detectability_threshold(self):
        from mascope_tools.composition.heuristic_filter import DETECT_SNR_K
        assert T.TOF_ASSIGN_FLOOR_X_EDGE == DETECT_SNR_K


class TestMedianOfFileEdges:
    def test_the_median_of_the_files_own_edges_ignoring_what_no_file_measured(self):
        assert batch_noise_edge(None, [], edges=[0.58, None, 0.78, float("nan"), 0.126, 0.0]) == pytest.approx(0.58)
        assert batch_noise_edge(None, [], edges=[None, float("nan")]) is None
        assert batch_noise_edge(None, [], edges=[]) is None
