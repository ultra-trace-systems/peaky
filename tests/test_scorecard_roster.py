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


def test_the_best_read_line_in_the_window_stands_for_the_formula(run):
    """Every stamped line in the window that passes the share test is read, not only the nearest: an isotope
    satellite of another ion nearest the target never hides a reading of the formula, another split of its ion
    or another reading a little further off; among lines of one status the nearest stands."""
    t = {f: C.ion_mz(f, "[M-H]-") for f in ROSTER["formula"]}
    run = with_sigma(run, 0.5)                                                  # a 2 ppm window
    ions = pd.DataFrame([
        # near line: an isotope satellite at +0.1 ppm, the formula's own reading at -1.5 ppm
        line(None, None, t["C10H16O5"] * (1 + 0.1e-6), role="iso_child", ion="C9(13C)H15O5-"),
        line("C10H16O5", "[M-H]-", t["C10H16O5"] * (1 - 1.5e-6)),
        # rare line: an isotope satellite at +0.1 ppm, another reading at +1.5 ppm, a rarer one nearer
        line(None, None, t["C9H14O5"] * (1 + 0.1e-6), role="iso_child", ion="C8(13C)H13O5-"),
        line("C8H10O6", "[M-H]-", t["C9H14O5"] * (1 + 1.5e-6)),
        line("C7H6O7", "[M-H]-", t["C9H14O5"] * (1 + 1.2e-6), presence=0.1),
        # other reading: two other readings, the nearer one stands
        line("C6H6O7", "[M-H]-", t["C7H10O6"] * (1 + 1.0e-6)),
        line("C11H14O2", "[M-H]-", t["C7H10O6"] * (1 - 0.3e-6)),
        # isotope line: only satellites in the window -- no sighting
        line(None, None, t["C8H12O5"] * (1 + 0.2e-6), role="iso_child", ion="C7(13C)H11O5-"),
        line(None, None, t["C8H12O5"] * (1 - 1.0e-6), role="iso_child", ion="C6(13C)H9O6-"),
    ])
    m2, st = m2_rows(run, ions)
    assert m2["window_ppm"] == pytest.approx(2.0)
    assert st["C10H16O5"]["status"] == "assigned" and st["C10H16O5"]["read"] == ""
    assert st["C9H14O5"]["status"] == "read as" and st["C9H14O5"]["read"].endswith("= C8H10O6 [M-H]-")
    assert st["C7H10O6"]["status"] == "read as" and st["C7H10O6"]["read"].endswith("= C11H14O2 [M-H]-")
    assert st["C8H12O5"]["status"] == "isotope/reagent line"


def test_a_reagent_line_of_the_expected_ions_own_composition_is_a_sighting(run):
    """The reagent's reference ions (its ion, its water cluster) are read as themselves on their reagent line; a
    roster formula whose ion is a reagent line is the same ion read as the reagent; a reagent line of another
    composition is no sighting."""
    roster = pd.DataFrame({"name": ["water", "other"], "formula": ["H2O", "C3H6O3"], "class": ["x"] * 2,
                           "reference": ["r"] * 2, "note": [""] * 2, "roster": ["t"] * 2})
    water = C.ion_mz("H2O", "[M+NO3]-")
    nitrate = C.ion_mz("HNO3", "[M-H]-")
    ions = pd.DataFrame([
        line(None, None, nitrate, role="reagent", ion="NO3-"),
        line(None, None, water, role="reagent", ion="H2NO4-"),
        # a reagent line of another composition at the roster formula's [M-H]- mass
        line(None, None, C.ion_mz("C3H6O3", "[M-H]-"), role="reagent", ion="C2HO4-"),
    ]).sort_values("ion_mz").reset_index(drop=True)
    tracks = pd.DataFrame(columns=["mz", "n", "n_peaks", "med_h", "sum_h", "presence"])
    m2 = SC.missed_m2(run, ions, tracks, roster)
    rows = {(r["source"], r["neutral"]): r for r in m2["rows"]}
    ref = rows[("reference_ion", "H2O")]
    assert (ref["status"], ref["adduct"], ref["read"]) == ("assigned", "[M+NO3]-", "H2NO4- (reagent)")
    assert rows[("reference_ion", "HNO3")]["status"] == "assigned"            # the reagent ion itself, on NO3-
    assert rows[("roster:t", "H2O")]["status"] == "same ion"
    assert rows[("roster:t", "C3H6O3")]["status"] == "isotope/reagent line"
    src = m2["sources"]["reference_ion"]
    assert src["assigned"] == 2 and src["present"] == 2
    # the card's sources line counts every status, so its counts add up to n
    card = SC.build_card(run, rosters=roster, board=[], log=lambda *a: None)
    card["m2"] = m2
    md = SC.render_md(card)
    n = src["n"]
    assert (f"- `reference_ion` ({n}): Assigned 2, Candidate 0, the same ion read as another split 0, read as other "
            f"0, unstamped 0, on an isotope / reagent line only {src['iso_reagent']}, absent {src['absent']}") in md
    assert sum(src[k] for k in ("assigned", "candidate", "same_ion", "read_as", "unstamped", "iso_reagent",
                                "absent")) == n
    page = html.unescape(SC.render_html([card], [card["row"]]))
    assert "0 same ion, 0 read as other, 0 unstamped" in page and "isotope / reagent line, " in page


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


def test_a_decoy_delta_across_a_change_of_headline_arm_or_ppm_shifts_is_not_taken():
    """The headline below m/z 350 quotes the ppm arms, or the Da arm (blind there on an Orbitrap): a switch of arm,
    or of the ppm shifts, is a change of measure, never an improvement or a regression."""
    base = {"decoy_calibration": "control", "decoy_scoring": "inherited", "claims_schema": 2, "scale": "0.10.0"}
    ppm = dict(base, decoy_headline_arm="ppm", decoy_headline_lt_350_rate=1.5, decoy_ppm_k=[9.0, -9.0],
               decoy_ppm_rate=1.9, decoy_ppm_identified_rate=0.0, decoy_shift_rate=0.0)
    da = dict(base, decoy_headline_arm="shift", decoy_headline_lt_350_rate=0.0, decoy_ppm_k=None,
              decoy_ppm_rate=None, decoy_shift_rate=0.0)
    assert SC.comparable(da, ppm, "decoy_headline_lt_350_rate") == (False, "decoy headline arm ppm -> shift")
    d = {x["metric"]: x for x in SC.delta(da, ppm)}
    hl = d["decoy (headline shift arm) Assigned below 350 %"]
    assert hl["prev"] is None and hl["delta"] is None
    assert hl["note"] == "not comparable: decoy headline arm ppm -> shift (previous 1.5)"
    assert d["decoy (shift) Assigned %"]["delta"] == 0                  # the Da arm itself: measured alike
    # the same arm at other shifts: the ppm rates and the headline are not comparable, the Da arm still is
    wide = dict(ppm, decoy_ppm_k=[6.0, -6.0, 12.0, -12.0], decoy_ppm_rate=2.7, decoy_headline_lt_350_rate=2.2)
    d = {x["metric"]: x for x in SC.delta(wide, ppm)}
    assert d["decoy (ppm shift) Assigned %"]["note"] == (
        "not comparable: decoy ppm shifts +9/-9 -> +12/+6/-6/-12 (previous 1.9)")
    assert d["decoy (headline shift arm) Assigned below 350 %"]["delta"] is None
    assert SC.comparable(wide, ppm, "decoy_shift_rate") == (True, "")
    # the shifts in another order, or a row before the key (no ppm arm) against one without: measured alike
    assert SC.comparable(dict(ppm, decoy_ppm_k=[-9, 9]), ppm, "decoy_ppm_rate") == (True, "")
    assert SC.comparable(da, dict(da, decoy_ppm_k=[]), "decoy_ppm_rate") == (True, "")
    assert SC.comparable(da, {k: v for k, v in da.items() if k != "decoy_headline_arm"},
                         "decoy_headline_lt_350_rate") == (False, "decoy headline arm none -> shift")
    # the board's two tables and the page take no delta across either change
    board = [dict(ppm, channel="B|NO3|cover", run="r1", written_utc="t1", code="a"),
             dict(wide, channel="B|NO3|cover", run="r2", written_utc="t2", code="b")]
    md = SC.render_board_md(board)
    assert "(+0.7)" not in md and "(+0.8)" not in md
    claims = SC.claim_board_rows(board)[0]
    assert claims["basis"] == "not comparable: decoy ppm shifts +9/-9 -> +12/+6/-6/-12"


def test_the_card_board_and_page_say_not_comparable(run, tmp_path):
    out = tmp_path / "board"
    first = SC.build_card(run, rosters=SC.load_rosters(), board=[], log=lambda *a: None)
    # the same channel's previous row, written on the pre-0.10.0 scale by the older presence test
    prev = {k: v for k, v in first["row"].items() if k not in ("scale", "roster_test", "decoy_calibration")}
    # ... and its decoy arms calibrated on their own (a row before the field)
    prev.update(claims_schema=1, roster_identified=20, roster_present=7, written_utc="2026-01-01T00:00:00Z",
                decoy_shift_rate=1.0, decoy_scoring="inherited")
    SC.write_outputs([dict(first, row=prev)], str(out), log=lambda *a: None)
    board = SC.read_board(str(out / "scoreboard.jsonl"))
    card = SC.build_card(run, rosters=SC.load_rosters(), board=board, log=lambda *a: None)
    # this card's arms ran at the control's calibration: the Da arm's rate is no delta of the previous one
    card["row"].update(decoy_shift_rate=3.0, decoy_calibration="control", decoy_scoring="inherited")
    card["delta"] = SC.delta(card["row"], SC.previous_row(board, card["row"]["channel"]))
    SC.write_outputs([card], str(out), log=lambda *a: None)
    md = (out / run.name / "SCORECARD.md").read_text()
    s6 = md[md.index("## 6. Delta"):]
    assert "| metric | previous | now | delta | note |" in s6
    assert "not comparable: level scale pre-0.10.0 -> 0.10.0; roster presence test 1 -> 2 (previous 20)" in s6
    claims_table = (out / "SCOREBOARD.md").read_text().split("## All metrics")[0]
    assert "roster identified / neutral or better / present / n" in claims_table
    assert "| not comparable: level scale pre-0.10.0 -> 0.10.0; roster presence test 1 -> 2 |" in claims_table
    # the All-metrics table: the Da arm's rate without a delta, the tier counts measured alike with theirs
    assert "not comparable: decoy calibration own -> control (previous 1.0)" in s6
    all_metrics = (out / "SCOREBOARD.md").read_text().split("## All metrics")[1]
    assert "| 3.0 |" in all_metrics and "(+2.0)" not in all_metrics
    page = html.unescape((out / "scoreboard.html").read_text())
    assert "vs previous row" in page
    assert "not comparable: level scale pre-0.10.0 -> 0.10.0; roster presence test 1 -> 2" in page    # claims table
    assert "(previous 20)" in page                                                                     # run-panel delta
    # the run panel's tiles: no delta chip across the change of measure (roster present 7 -> 4, Da arm 1.0 -> 3.0)
    tiles = page[page.index("decoy Da shift Assigned %"):]
    assert tiles.startswith("decoy Da shift Assigned %</span><b>3.0</b></div>")
    present = page[page.index("<span>roster present</span>"):]
    assert present.startswith("<span>roster present</span><b>4</b></div>")
    assert "vs previous" not in page.split("<span>roster present</span>")[1].split("</div>")[0]
