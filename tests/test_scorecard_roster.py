"""scripts/scorecard.py -- the roster block measures what it claims.

M2's presence test looks for a roster line inside the run's own mass accuracy
(4 x its measured sigma, 1 ppm at least, its tolerance at most), in a share
of the spectra, never on an isotope or reagent line, and never calls another
split of the same ion composition a misread. The roster claims carry the
scale they read, identified beside neutral-or-better, and a delta across a
change of scale, presence test or decoy calibration is marked, never taken.

Run: pytest tests/test_scorecard_roster.py -q
"""

import dataclasses
import html
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import scorecard as SC  # noqa: E402
from peaky.chem import chemistry as C  # noqa: E402
from test_scorecard import write_run  # noqa: E402


def with_sigma(run, sigma):
    return dataclasses.replace(run, summary=dict(run.summary, mass_scale={"sigma_ppm": sigma}))


@pytest.fixture
def run(tmp_path):
    return SC.load_run(str(write_run(tmp_path / "out")))


def test_the_presence_window_follows_the_runs_measured_mass_sigma(run):
    assert run.tol_ppm == 6.0 and SC.roster_window_ppm(run) == 6.0            # no sigma measured: the tolerance
    assert SC.roster_window_ppm(with_sigma(run, 0.2)) == 1.0                    # 0.8 ppm, never below 1 ppm
    assert SC.roster_window_ppm(with_sigma(run, 0.5)) == pytest.approx(2.0)
    assert SC.roster_window_ppm(with_sigma(run, 3.7)) == 6.0                    # never above the tolerance
    assert SC.roster_window_ppm(with_sigma(run, "n/a")) == 6.0


def line(neutral, adduct, mz, role="M0", presence=1.0, tier="Assigned", ion=None):
    """One stamped ion of the ion table."""
    return dict(ion_mz=mz, n=int(round(presence * 4)), med_h=100.0, sum_h=400.0, role=role,
                ion_formula=ion or (C.format_formula(SC.EV.ion_composition(neutral, adduct, None)) + adduct[-1] if neutral else "X-"),
                iso_label=None, neutral=neutral, adduct=adduct, tier=tier, suspect=False, stamp_source="", presence=presence)


ROSTER = pd.DataFrame({
    "name": ["near line", "rare line", "isotope line", "organonitrate", "other reading", "itself"],
    "formula": ["C10H16O5", "C9H14O5", "C8H12O5", "C10H15NO7", "C7H10O6", "C10H16O3"],
    "class": ["x"] * 6, "reference": ["r"] * 6, "note": [""] * 6, "roster": ["t"] * 6})


def m2_rows(run, ions):
    tracks = pd.DataFrame(columns=["mz", "n", "n_peaks", "med_h", "sum_h", "presence"])
    m2 = SC.missed_m2(run, ions.sort_values("ion_mz").reset_index(drop=True), tracks, ROSTER)
    return m2, {r["neutral"]: r for r in m2["rows"] if r["source"] == "roster:t"}


def test_the_presence_test_counts_only_a_real_sighting_of_the_formula(run):
    t = {f: C.ion_mz(f, "[M-H]-") for f in ROSTER["formula"]}
    ions = pd.DataFrame([
        line("C10H16O5", "[M-H]-", t["C10H16O5"] * (1 + 3e-6)),                 # 3 ppm off: a neighbour at 1 ppm
        line("C9H14O5", "[M-H]-", t["C9H14O5"], presence=0.1),                   # in 10 % of the spectra only
        line(None, None, t["C8H12O5"], role="iso_child", ion="C7(13C)H11O5-"),    # another ion's satellite
        line("C10H14O4", "[M+NO3]-", C.ion_mz("C10H14O4", "[M+NO3]-")),          # the organonitrate's own ion, split
        line("C6H6O7", "[M-H]-", t["C7H10O6"] * (1 + 0.5e-6)),                   # another ion read on the line
        line("C10H16O3", "[M-H]-", t["C10H16O3"]),
    ])
    assert C.ion_mz("C10H14O4", "[M+NO3]-") == pytest.approx(t["C10H15NO7"])
    # the run's flat tolerance (no sigma measured): the 3-ppm neighbour and the rare line still count
    m2, st = m2_rows(run, ions)
    assert m2["window_ppm"] == 6.0 and m2["test"] == SC.ROSTER_TEST and m2["scale"] == "peaky 0.10.0"
    assert st["C10H16O5"]["status"] == "assigned" and st["C9H14O5"]["status"] == "absent"
    assert st["C8H12O5"]["status"] == "isotope/reagent line" and st["C8H12O5"]["read"] == "C7(13C)H11O5- (iso_child)"
    assert st["C10H15NO7"]["status"] == "same ion" and st["C10H15NO7"]["read"].endswith("= C10H14O4 [M+NO3]-")
    assert st["C7H10O6"]["status"] == "read as" and st["C10H16O3"]["status"] == "assigned"
    r = m2["roster"]["t"]
    assert (r["present"], r["assigned"], r["same_ion"], r["read_as"], r["iso_reagent"], r["absent"]) == (4, 2, 1, 1, 1, 1)
    # at an Orbitrap's sigma the window is 1 ppm: the neighbour 3 ppm away is no sighting
    m2, st = m2_rows(with_sigma(run, 0.2), ions)
    assert m2["window_ppm"] == 1.0 and st["C10H16O5"]["status"] == "absent"
    assert st["C7H10O6"]["status"] == "read as"                                  # 0.5 ppm: still on the line
    assert m2["roster"]["t"]["present"] == 3


def test_the_roster_claims_sit_side_by_side_and_name_their_scale(run):
    t = {f: C.ion_mz(f, "[M-H]-") for f in ROSTER["formula"]}
    ions = pd.DataFrame([line("C10H16O3", "[M-H]-", t["C10H16O3"]), line("C10H16O5", "[M-H]-", t["C10H16O5"]),
                         line("C9H14O5", "[M-H]-", t["C9H14O5"], tier="Candidate")])
    lv = pd.DataFrame(dict(neutral=["C10H16O3", "C10H16O5", "C9H14O5"], adduct=["[M-H]-"] * 3, level=["3c", "4a", "4b"]))
    tracks = pd.DataFrame(columns=["mz", "n", "n_peaks", "med_h", "sum_h", "presence"])
    m2 = SC.missed_m2(run, ions.sort_values("ion_mz").reset_index(drop=True), tracks, ROSTER, lv)
    rc = m2["roster_claim"]["t"]
    assert (rc["identified"], rc["neutral"], rc["ion"], rc["neutral_or_better"]) == (1, 1, 1, 2)
    card = SC.build_card(run, rosters=ROSTER, board=[], log=lambda *a: None)
    card["m2"] = m2
    row = SC.board_row(card)
    assert (row["roster_identified"], row["roster_neutral_or_better"]) == (1, 2)
    assert row["roster_test"] == SC.ROSTER_TEST and row["scale"] == SC.EV.SCALE_RELEASE and row["roster_window_ppm"] == 6.0
    md = SC.render_md(card)
    assert "read as itself and identified (3c) 1, neutral or better (3c + 4a) 2" in md
    assert "(presence test 2). Claims on the level scale peaky 0.10.0." in md


def test_a_delta_across_a_change_of_measure_is_marked_not_taken():
    new = {"roster_identified": 2, "roster_neutral_or_better": 16, "roster_present": 28, "assigned": 500,
           "decoy_shift_rate": 3.0, "claims_schema": 2, "scale": "0.10.0", "roster_test": 2,
           "decoy_calibration": "control", "decoy_scoring": "inherited"}
    old = {"roster_identified": 20, "roster_present": 33, "assigned": 514, "decoy_shift_rate": 0.0,
           "claims_schema": 1, "decoy_scoring": "inherited"}
    d = {x["metric"]: x for x in SC.delta(new, old)}
    ri = d["roster identified"]
    assert ri["prev"] is None and ri["delta"] is None
    assert ri["note"] == "not comparable: level scale pre-0.10.0 -> 0.10.0; roster presence test 1 -> 2 (previous 20)"
    assert d["roster present"]["note"] == "not comparable: roster presence test 1 -> 2 (previous 33)"
    assert d["decoy (shift) Assigned %"]["note"].startswith("not comparable: decoy calibration own -> control")
    assert d["Assigned rows"]["delta"] == -14 and "note" not in d["Assigned rows"]
    assert "note" not in d["roster neutral or better"]                 # nothing to compare: no note either
    # measured alike: a plain delta, no note
    same = {x["metric"]: x for x in SC.delta(new, dict(new, roster_present=30, decoy_shift_rate=2.0))}
    assert same["roster present"]["delta"] == -2 and same["decoy (shift) Assigned %"]["delta"] == pytest.approx(1.0)
    assert not any(x.get("note") for x in same.values())
    # a schema-2 row written before the `scale` key reads its claims_schema's scale
    assert SC.row_scale({"claims_schema": 2}) == "0.10.0" and SC.row_scale({}) == "before the claim"
    assert SC.comparable(new, {"claims_schema": 2, "roster_test": 2}, "roster_identified") == (True, "")


def test_the_card_board_and_page_say_not_comparable(run, tmp_path):
    out = tmp_path / "board"
    first = SC.build_card(run, rosters=SC.load_rosters(), board=[], log=lambda *a: None)
    # the same channel's previous row, written on the pre-0.10.0 scale by the older presence test
    prev = {k: v for k, v in first["row"].items() if k not in ("scale", "roster_test", "decoy_calibration")}
    prev.update(claims_schema=1, roster_identified=20, roster_present=7, written_utc="2026-01-01T00:00:00Z")
    SC.write_outputs([dict(first, row=prev)], str(out), log=lambda *a: None)
    board = SC.read_board(str(out / "scoreboard.jsonl"))
    card = SC.build_card(run, rosters=SC.load_rosters(), board=board, log=lambda *a: None)
    SC.write_outputs([card], str(out), log=lambda *a: None)
    md = (out / run.name / "SCORECARD.md").read_text()
    s6 = md[md.index("## 6. Delta"):]
    assert "| metric | previous | now | delta | note |" in s6
    assert "not comparable: level scale pre-0.10.0 -> 0.10.0; roster presence test 1 -> 2 (previous 20)" in s6
    claims_table = (out / "SCOREBOARD.md").read_text().split("## All metrics")[0]
    assert "roster identified / neutral or better / present / n" in claims_table
    assert "| not comparable: level scale pre-0.10.0 -> 0.10.0; roster presence test 1 -> 2 |" in claims_table
    page = html.unescape((out / "scoreboard.html").read_text())
    assert "vs previous row" in page
    assert "not comparable: level scale pre-0.10.0 -> 0.10.0; roster presence test 1 -> 2" in page    # claims table
    assert "(previous 20)" in page                                                                     # run-panel delta
