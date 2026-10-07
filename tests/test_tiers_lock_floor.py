"""The score floor on certified readings (tiers.LOCKED_SCORE_METHODS).

A pass-7 certified neutral earns its tier from the channels' convergent neutral
mass and carried 'Good (certified)' with no score floor; on a spectrum shifted a
few ppm off its true formulas that path Assigned wrong readings at low ion
scores. Under the engine's own Suspect band edge (PassConfig.tau_suspect) a
certified commit is Candidate. A pass-0 known species is not floored: real
known species (NO2-, HSO4-, ethanol, cyclosiloxanes) score in the same range as
the decoy's known-species locks, so its lock keeps its own branch.

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
from peaky.batch import assign_batch as AB  # noqa: E402
from peaky.batch.assign_batch import align  # noqa: E402

PEAKS = pd.DataFrame({
    "peak_id": ["K1", "K2", "K3", "K4", "K5", "C1", "C2", "R1", "S1"],
    "mz": [276.9714, 312.9731, 262.9766, 162.9825, 212.9792, 183.1005, 243.1325, 271.0789, 93.0910],
    "height": [130.0, 72.0, 9.0e4, 3.0e3, 4.0e4, 900.0, 800.0, 400.0, 5.0e4]})


def _ledger(scores: dict | None = None, *, k1_second_channel: bool = False) -> pd.DataFrame:
    """`k1_second_channel` commits K5 as the [M-H]- channel of K1's neutral."""
    s = {"K1": 0.034, "K2": 0.52, "K3": 0.996, "K4": 0.50, "K5": 0.95, "C1": 0.41, "C2": 0.86,
         "R1": 0.14}
    s.update(scores or {})
    led = L.new_ledger(PEAKS)
    # pass-0 known species, each on one ion channel: a Low lock far under the
    # floor, a Low lock just over it, a Good lock, and one exactly at the edge
    known = [("K1", "C4HF7O2", "[M+^NO3]-", "C4HF7O2.^NO3-", "Low (perfluoroacid)"),
             ("K2", "C6HF11O2", "[M-H]-", "C6F11O2-", "Low (perfluoroacid)"),
             ("K3", "C5HF9O2", "[M-H]-", "C5F9O2-", "Good (perfluoroacid)"),
             ("K4", "C3HF5O2", "[M-H]-", "C3F5O2-", "Low (perfluoroacid)")]
    if k1_second_channel:
        known.append(("K5", "C4HF7O2", "[M-H]-", "C4F7O2-", "Good (perfluoroacid)"))
    for pid, nf, ad, ion, conf in known:
        L.commit_assignment(led, pid, neutral_formula=nf, adduct=ad, ion_formula=ion,
                            ion_score=s[pid], compound_score=s[pid], ppm_error=0.1, pass_no=0,
                            method="known:perfluoroacid", confidence=conf,
                            commentary=f"Pass 0 (known perfluoroacid): {nf} {ad}")
    L.lock_peaks(led, [k[0] for k in known])
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

    def test_a_known_species_lock_under_the_edge_is_not_floored(self):
        t = _tiers(_ledger())
        assert t["K1"][0] == T.TIER_ASSIGNED, t["K1"]
        assert t["K1"][1].startswith("known species (pass-0 locked list, mass + own-twin"), t["K1"]
        assert "Suspect band edge" not in t["K1"][1]
        # the recovered chlorinated paraffin is a known: lock like any other
        assert t["R1"][0] == T.TIER_ASSIGNED and "Suspect band edge" not in t["R1"][1]
        assert not T.LOCKED_SCORE_METHODS[0].startswith("known:") and len(T.LOCKED_SCORE_METHODS) == 1

    def test_a_known_species_lock_at_or_over_the_floor_keeps_its_branch(self):
        t = _tiers(_ledger())
        for pid in ("K2", "K3", "K4"):        # 0.52 (Low), 0.996 (Good), 0.50 exactly
            assert t[pid][0] == T.TIER_ASSIGNED, (pid, t[pid])
            assert t[pid][1].startswith("known species (pass-0 locked list, mass + own-twin"), t[pid]

    def test_a_certified_member_under_the_floor_is_candidate_its_sibling_is_not(self):
        """A certificate converges two or more channels by construction, so the
        second channel that spares a known species does not spare a member."""
        t = _tiers(_ledger())
        assert t["C1"][0] == T.TIER_CANDIDATE
        assert t["C1"][1].startswith("certified neutral (pass-7 multi-channel certificate) with ion score 0.41")
        # the other channel of the same certificate scores 0.86: still Assigned on the
        # second channel, as before
        assert t["C2"][0] == T.TIER_ASSIGNED and "second ionization channel" in t["C2"][1]

    def test_a_known_species_on_two_channels_keeps_its_route_text(self):
        """A second ion channel of the listed neutral is still reported as the
        known species' corroboration route; no floor text appears."""
        t = _tiers(_ledger({"K5": 0.20}, k1_second_channel=True))
        assert t["K1"][0] == T.TIER_ASSIGNED and t["K5"][0] == T.TIER_ASSIGNED
        assert "corroborated by 2 ion channels" in t["K1"][1]
        assert "Suspect band edge" not in t["K1"][1] and "Suspect band edge" not in t["K5"][1]

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
        t = _tiers(_ledger({"C1": 0.25}), cfg=PassConfig(tau_suspect=0.3))
        assert t["C1"][0] == T.TIER_CANDIDATE and "Suspect band edge (0.30)" in t["C1"][1]

    def test_a_lock_with_no_recorded_score_is_not_floored(self):
        """No number, no judgment: a ledger written without the commit's score (an
        old CSV, a hand-built frame) keeps the lock's own branch."""
        led = _ledger()
        led.loc[led["peak_id"] == "C1", "ion_score"] = np.nan
        assert _tiers(led)["C1"][0] == T.TIER_ASSIGNED

    def test_rows_off_the_locked_paths_are_untouched(self):
        """A grid commit at the same low score is judged by its own branches, as before."""
        led = _ledger()
        i = led.index[led["peak_id"] == "C1"][0]
        led.at[i, "method"] = "cheminfo+grid"
        led.at[i, "confidence"] = "Good"
        t = _tiers(led)
        assert "Suspect band edge" not in str(t["C1"][1])

    def test_a_csv_round_trip_gives_the_same_verdicts(self, tmp_path):
        led = _ledger()
        path = tmp_path / "ledger.csv"
        led.to_csv(path, index=False)
        assert _tiers(pd.read_csv(path, low_memory=False)) == _tiers(led)

    def test_apply_tiers_stamps_the_floor(self):
        led = T.apply_tiers(_ledger())
        by = led.set_index("peak_id")
        assert by.at["C1", "tier"] == T.TIER_CANDIDATE and by.at["K1", "tier"] == T.TIER_ASSIGNED
        assert by.at["K3", "tier"] == T.TIER_ASSIGNED and by.at["S1", "tier"] == T.TIER_ASSIGNED


def test_a_reading_certified_under_the_floor_in_every_file_is_candidate_at_the_merge():
    """The batch's merged tier is the winning reading's best per-file tier: a
    certified reading whose every file holds it under the floor loses Assigned,
    one that some file holds over the floor keeps it; a known species under the
    edge keeps Assigned."""
    per_file = {}
    for src, (s_low, s_mix) in {"f1": (0.30, 0.80), "f2": (0.45, 0.45)}.items():
        led = T.apply_tiers(_ledger({"C1": s_low, "C2": s_mix, "K1": 0.03}))
        per_file[src] = led[led["role"] == "M0"]
    merged, _jit = align(per_file)
    by = merged.set_index(merged["neutral_formula"] + " " + merged["adduct"])
    assert by.at["C4H15N4O2P [M+H]+", "tier"] == T.TIER_CANDIDATE
    assert by.at["C4H15N4O2P [M+(CH4N2O)H]+", "tier"] == T.TIER_ASSIGNED
    assert by.at["C4HF7O2 [M+^NO3]-", "tier"] == T.TIER_ASSIGNED


def _pool(tier: str, *, srcs=("f1", "f2")) -> list[dict]:
    """known_evidence records of K1's reading: one channel, two satellite lines."""
    return [dict(neutral="C4HF7O2", adduct="[M+^NO3]-", family="perfluoroacid", label="PFBA", src=src,
                 mz=276.9714, verdict="confirmed", why="two satellites", summary="two satellites",
                 n_channels=1, n_satellites=2, ion_score=0.034, tier=tier, admitted_by="height",
                 occurrence=0.5) for src in srcs]


def _merged_two_files():
    """K1's reading merged over two files, held at Candidate there (as a per-file
    rule such as the counting-detector floor would hold it)."""
    per_file = {}
    for src in ("f1", "f2"):
        led = T.apply_tiers(_ledger())
        led.loc[led["peak_id"] == "K1", "tier"] = T.TIER_CANDIDATE
        per_file[src] = led[led["role"] == "M0"]
    merged, _jit = align(per_file)
    return merged, merged.index[(merged["neutral_formula"] == "C4HF7O2")
                                & (merged["adduct"] == "[M+^NO3]-")][0]


@pytest.mark.parametrize("vote_winner", ["species", "other"])
def test_the_merged_row_says_why_a_batch_known_species_is_candidate(vote_winner):
    """lock_known_species: when every confirming file holds the reading at
    Candidate (here a per-file rule) the merged row says so, whether the vote
    already read the species (kept) or the batch locks it over another reading."""
    merged, i = _merged_two_files()
    if vote_winner == "other":
        merged.at[i, "neutral_formula"], merged.at[i, "adduct"] = "C9H6O8", "[M-H]-"
        merged.at[i, "tier"] = T.TIER_ASSIGNED
    AB.lock_known_species(merged, _pool(T.TIER_CANDIDATE), tol_ppm=6.0, log=lambda *a, **k: None)
    assert merged.at[i, "neutral_formula"] == "C4HF7O2" and merged.at[i, "tier"] == T.TIER_CANDIDATE
    assert str(merged.at[i, "tier_reason"]).endswith(
        "; Candidate: no confirming file holds it at Assigned (see the per-file tier reasons)")


def test_no_note_when_a_confirming_file_holds_it_at_assigned():
    merged, i = _merged_two_files()
    merged.at[i, "tier"] = T.TIER_ASSIGNED
    pool = _pool(T.TIER_CANDIDATE, srcs=("f1",)) + _pool(T.TIER_ASSIGNED, srcs=("f2",))
    AB.lock_known_species(merged, pool, tol_ppm=6.0, log=lambda *a, **k: None)
    assert merged.at[i, "tier"] == T.TIER_ASSIGNED
    assert "no confirming file holds it" not in str(merged.at[i, "tier_reason"])
