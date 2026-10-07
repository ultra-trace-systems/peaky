"""The evidence scale (docs/EVIDENCE_LEVELS.md) and its claims in the
user-facing outputs: the assignment workbook and markdown (report.py), the PDF
report (pdf_report.py) and the publish provenance (publish.py).

evidence.claim_class and the stage's stamping are pinned by the evidence tests;
this file pins that:

- the workbook opens on a By claim sheet over the four claims and the two
  buckets (reagent, not assessed), `claim` sits directly before
  `evidence_level` and the level's text columns follow it wherever the level is
  shown, the B-series columns (evidence_axes, level_reason,
  n_plausible_structures) are gone, the Assigned sheet keeps `tier_reason`
  under its own name and the Unassigned sheet's residual class is
  `residual_evidence`; By evidence level runs over the levels + buckets with a
  tag-kind histogram; a ledger without levels renders unchanged;
- the literal level token NA survives a CSV round trip (read off the claim, or
  re-read as written), and a run not assessed says so (Summary, markdown, the
  PDF cover and levels page);
- a ledger levelled on an older scale renders: EVERY letter reads as no level
  (the letters both scales share too: an old 4a is not a 4a here), its claims
  re-read tentative, and the outputs say so;
- the PDF leads with a Claims page whose signal shares carry an explicit
  `unmatched` bucket, the levels page draws every level and bucket (never a
  KeyError), and the cover / findings / methods / appendix carry the claim;
- publish carries the scale's columns (evidence.COLUMNS) into the provenance.

The tier is shown beside the claim everywhere and never read off it.

Run: pytest tests/test_claims_outputs.py -q
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from peaky.assignment import evidence as EV
from peaky.assignment import ledger as L
from peaky.assignment import tiers as T
from peaky.chem import chemistry as C
from peaky.io import publish as P
from peaky.reporting import pdf_report as PR
from peaky.reporting import report as R

OLD_COLUMNS = ("evidence_axes", "level_reason", "n_plausible_structures")
NA_TEXT = "NA · not assessed on this instrument class (width model R(200) = 9 652 < 50 000)"

# the level each committed peak is given after the stage runs: one row per claim
# and both directions of a tier / claim disagreement
LEVELS = {"A": "4a", "C": "5a", "D": "3c", "F": "4b"}
TAGS = {"A": "side channels locked | lead: spec_n3 (a tentative lead: not answered by an own isotope line)",
        "C": "", "D": "class list: reflist:x -- tag (formula-only / class entry)", "F": "side channels locked"}


def _ledger(levels: bool = True) -> pd.DataFrame:
    """A finished single-sample ledger in the real schema: A (iso-confirmed, with
    its 13C child B), C (series), D (a near-tie: Candidate, with one alternative),
    F, a reagent ion E and an unexplained peak G. With `levels`, the evidence
    stage has run and the levels are set to LEVELS (claim re-read off each, the
    text columns filled)."""
    peaks = pd.DataFrame({"peak_id": ["A", "B", "C", "D", "F", "E", "G"],
                          "mz": [200.1, 201.1035, 191.0, 145.05, 173.08, 78.9189, 999.0],
                          "height": [1.0e5, 1.07e4, 8.0e4, 5.0e4, 3.0e4, 1.0e4, 1.0e3]})
    led = L.new_ledger(peaks)
    L.commit_assignment(led, "A", neutral_formula="C10H16O4", adduct="[M-H]-",
                        ion_formula="C10H15O4-", ion_score=0.97, compound_score=0.96,
                        ppm_error=-0.3, pass_no=1, method="cheminfo", confidence="High",
                        commentary="Pass 1: C10H16O4 [M-H]-",
                        isotopologues=[{"label": "13C", "score": 0.93, "peak_id": "B"}])
    L.attach_isotopologue(led, "B", "A", iso_label="13C", iso_match_score=0.93)
    L.commit_assignment(led, "C", neutral_formula="C7H12O4", adduct="[M-H]-",
                        ion_formula="C7H11O4-", ion_score=0.83, compound_score=0.83,
                        ppm_error=0.5, pass_no=2, method="gka-series", confidence="Good (series)",
                        commentary="Pass 2 series from C8H14O4 -CH2")
    L.commit_assignment(led, "D", neutral_formula="C6H10O4", adduct="[M-H]-",
                        ion_formula="C6H9O4-", ion_score=0.85, compound_score=0.85,
                        eff_score=0.84, eff_margin=0.02, tied=True, ppm_error=0.3, pass_no=1,
                        method="cheminfo+grid", confidence="Good", commentary="Pass 1 near-tie",
                        alternatives=[{"formula": "C2H6N2O6", "ion_score": 0.84, "raw_score": 0.84,
                                       "eff_score": 0.82, "ppm": 0.4}])
    L.commit_assignment(led, "F", neutral_formula="C7H10O5", adduct="[M-H]-",
                        ion_formula="C7H9O5-", ion_score=0.9, compound_score=0.9,
                        ppm_error=0.2, pass_no=1, method="cheminfo", confidence="Good",
                        commentary="Pass 1: C7H10O5 [M-H]-")
    L.mark_reagent(led, "E", "reagent ion: [Br]-")
    T.apply_tiers(led)
    if levels:
        EV.apply_levels(led)
        for pid, lv in LEVELS.items():
            i = led.index[led["peak_id"] == pid][0]
            led.at[i, "evidence_level"] = lv
            led.at[i, "claim"] = EV.claim_class(lv)
            led.at[i, "evidence"] = f"{lv} · ion: unique in the calibrated window · context source: none · tags: none"
            led.at[i, "would_lift"] = "level 2 (MS2 / standards) is not automatic" if lv == "3c" else "x"
            led.at[i, "competitors_left"] = "C9H12O5 [M-H]-" if lv == "5a" else ""
            led.at[i, "tags"] = TAGS[pid]
    return led


def _real_ledger() -> pd.DataFrame:
    """A small ledger at exact masses, levelled by the per-file stage itself on
    an Orbitrap-class width model (the stage's own strings, not hand-set ones)."""
    mz_a = C.ion_mz("C10H16O4", "[M-H]-")
    peaks = pd.DataFrame({"peak_id": ["A", "B", "C", "E"],
                          "mz": [mz_a, mz_a + 1.0033548, C.ion_mz("C7H12O4", "[M-H]-"), 78.9189],
                          "height": [1.0e5, 1.07e4, 8.0e4, 1.0e4]})
    led = L.new_ledger(peaks)
    L.commit_assignment(led, "A", neutral_formula="C10H16O4", adduct="[M-H]-",
                        ion_formula="C10H15O4-", ion_score=0.97, compound_score=0.96,
                        ppm_error=-0.3, pass_no=1, method="cheminfo", confidence="High",
                        commentary="Pass 1: C10H16O4 [M-H]-",
                        isotopologues=[{"label": "13C", "score": 0.93, "peak_id": "B"}])
    L.attach_isotopologue(led, "B", "A", iso_label="13C", iso_match_score=0.93)
    L.commit_assignment(led, "C", neutral_formula="C7H12O4", adduct="[M-H]-",
                        ion_formula="C7H11O4-", ion_score=0.83, compound_score=0.83,
                        ppm_error=0.5, pass_no=2, method="gka-series", confidence="Good (series)",
                        commentary="Pass 2 series from C8H14O4 -CH2")
    L.mark_reagent(led, "E", "reagent ion: [Br]-")
    T.apply_tiers(led)
    return led


def _inputs(resolution=120_000.0):
    return EV.file_run_inputs(sample_id="s1", reagent="NO3", context="ambient-air", resolution=resolution,
                              height_gate_cps=100.0, degeneracy_cal=(0.0, 0.5))


def _round_trip(led: pd.DataFrame, tmp_path: Path, name: str = "led.csv") -> pd.DataFrame:
    """The ledger as a reader gets it back: written and read with the default parser."""
    path = tmp_path / name
    led.to_csv(path, index=False)
    return pd.read_csv(path)


def _by(led: pd.DataFrame, col: str) -> dict:
    return dict(zip(led["peak_id"], led[col]))


# --------------------------------------------------------------------------- the workbook
def test_the_workbook_opens_on_the_by_claim_sheet_with_the_counts_and_the_signal():
    led = _ledger()
    tier = _by(led, "tier")
    assert tier["A"] == T.TIER_ASSIGNED and tier["C"] == T.TIER_ASSIGNED and tier["D"] == T.TIER_CANDIDATE
    sheets = R.build_sheets(led, "ambient-air", sample_id="T")
    assert list(sheets)[0] == "By claim"
    keys = list(sheets)
    assert keys.index("By evidence level") == keys.index("Candidates") + 1   # unchanged
    bc = sheets["By claim"]
    summ = bc[bc.section == "summary"].set_index("claim")
    assert list(summ.index) == list(EV.CLAIM_KEYS)                          # 4 claims + 2 buckets
    assert [summ.loc[c, "n"] for c in EV.CLAIM_KEYS] == [1, 1, 1, 1, 0, 0]
    assert summ.loc["neutral", "share"] == pytest.approx(0.25)
    h = 1.0e5 + 8.0e4 + 5.0e4 + 3.0e4
    assert summ.loc["identified", "signal"] == pytest.approx(5.0e4)          # D
    assert summ.loc["neutral", "signal_share"] == pytest.approx(1.0e5 / h)    # A
    assert summ.loc["tentative", "signal"] == pytest.approx(8.0e4)
    assert summ.loc["identified", "n_assigned"] == 0 and summ.loc["identified", "n_candidate"] == 1
    assert summ.loc["tentative", "n_assigned"] == 1 and summ.loc["identified", "n_ion_only"] == 0
    assert summ.loc["identified", "levels"] == "1 2 3c"
    assert summ.loc["neutral", "levels"] == "4a" and summ.loc["ion", "levels"] == "4b"
    assert summ.loc["tentative", "levels"] == "5a 5b, no level"
    assert summ.loc["reagent", "levels"] == "reagent" and summ.loc["not assessed", "levels"] == "NA"
    assert summ.loc["identified", "level_hist"] == "3c: 1"
    for c in EV.CLAIM_KEYS:
        assert summ.loc[c, "meaning"] == EV.CLAIM_MEANING[c]
        assert (bc.section == f"brightest {c}").sum() == summ.loc[c, "n"]
    assert not set(OLD_COLUMNS) & set(bc.columns)


def test_the_by_claim_sheet_lists_where_tier_and_claim_part_brightest_first():
    bc = R.build_sheets(_ledger())["By claim"]
    dis = bc[bc.section == "tier disagrees"]
    # Assigned but tentative (C, 8e4) before Candidate but identified (D, 5e4); A and F agree
    assert list(dis.peak_id) == ["C", "D"]
    assert list(zip(dis.tier, dis.claim, dis.evidence_level)) == [
        (T.TIER_ASSIGNED, "tentative", "5a"), (T.TIER_CANDIDATE, "identified", "3c")]
    assert dis[["mz", "height", "neutral_formula", "adduct", "evidence", "would_lift"]].notna().all().all()


def test_claim_sits_directly_before_evidence_level_and_the_text_columns_follow():
    sheets = R.build_sheets(_ledger())
    carrying = [k for k, v in sheets.items() if "evidence_level" in v.columns]
    for name in ("Assigned", "Candidates", "Peak ownership", "Target list"):
        assert name in carrying, name
        cols = list(sheets[name].columns)
        i = cols.index("evidence_level")
        assert cols[i - 1] == "claim", (name, cols)
        assert cols[i + 1:i + 7] == ["evidence", "would_lift", "competitors_left", "tags", "context",
                                     "context_source"], (name, cols)
    for name, df in sheets.items():
        assert not set(OLD_COLUMNS) & set(df.columns), name
        if "evidence_level" in df.columns:
            cols = list(df.columns)
            assert cols[cols.index("evidence_level") - 1] == "claim", (name, cols)


def test_the_assigned_sheet_keeps_tier_reason_and_the_residual_class_is_renamed():
    sheets = R.build_sheets(_ledger())
    a = sheets["Assigned"]
    assert "tier_reason" in a.columns and list(a.columns).count("evidence") == 1
    assert a.set_index("neutral_formula").loc["C10H16O4", "evidence"].startswith("4a · ")
    un = sheets["Unassigned"]
    assert "residual_evidence" in un.columns and "evidence" not in un.columns
    assert un.set_index("peak_id").loc["G", "residual_evidence"] in ("iso-partner", "has-constraints", "isolated")
    # unlevelled ledgers keep the names too
    plain = R.build_sheets(_ledger(levels=False))
    assert "tier_reason" in plain["Assigned"].columns and "evidence" not in plain["Assigned"].columns
    assert "residual_evidence" in plain["Unassigned"].columns


def test_the_claim_sits_on_committed_rows_only():
    sheets = R.build_sheets(_ledger())
    own = sheets["Peak ownership"].set_index("peak_id")
    for pid in ("B", "E", "G"):                     # isotope child, reagent ion, unexplained
        assert pd.isna(own.loc[pid, "claim"]), pid
        assert pd.isna(own.loc[pid, "evidence_level"]), pid
    assert own.loc["A", "claim"] == "neutral" and own.loc["F", "claim"] == "ion"
    cand = sheets["Candidates"]
    d = cand[cand.peak_id == "D"].set_index("rank")
    assert d.loc[1, "claim"] == "identified" and d.loc[1, "evidence"].startswith("3c · ")
    assert d.loc[2, "claim"] == "" and d.loc[2, "evidence"] == ""            # the alternative makes none
    assert set(sheets["Assigned"].claim) == {"neutral", "tentative", "ion"}
    uniq = sheets["Unique formulas"].set_index("neutral_formula")
    cols = list(sheets["Unique formulas"].columns)
    assert cols[cols.index("best_tier") + 1] == "best_claim"
    assert uniq.loc["C6H10O4", "best_claim"] == "identified"
    assert uniq.loc["C10H16O4", "best_claim"] == "neutral"


def test_an_older_ledger_without_the_column_reads_the_claim_off_its_level():
    led = _ledger()
    now = R.build_sheets(led)
    old = R.build_sheets(led.drop(columns=["claim"]))
    pd.testing.assert_frame_equal(now["By claim"], old["By claim"])
    own = old["Peak ownership"].set_index("peak_id")
    assert own.loc["C", "claim"] == "tentative" and pd.isna(own.loc["B", "claim"])
    # a committed row with no level reads tentative
    led2 = led.drop(columns=["claim"])
    led2.loc[led2.peak_id == "F", "evidence_level"] = pd.NA
    assert R.build_sheets(led2)["Peak ownership"].set_index("peak_id").loc["F", "claim"] == "tentative"


def test_a_ledger_without_levels_renders_unchanged():
    sheets = R.build_sheets(_ledger(levels=False), "ambient-air", sample_id="T")
    assert list(sheets)[0] == "Summary"
    assert "By claim" not in sheets and "By evidence level" not in sheets
    for name, df in sheets.items():
        assert "claim" not in df.columns and "best_claim" not in df.columns, name
        assert not set(EV.COLUMNS) & set(df.columns), name
    assert not sheets["Summary"].section.eq("Claims").any()
    assert not sheets["Summary"].section.eq("Evidence levels").any()
    assert not sheets["Read me"].section.eq("Claims").any()


def test_the_summary_sheet_puts_the_claims_between_coverage_and_tiers():
    ss = R.summary_stats(_ledger(), context="ambient-air")
    secs = list(dict.fromkeys(ss.section))
    assert secs.index("Coverage") < secs.index("Claims") == secs.index("Coverage") + 1
    assert secs.index("Claims") < secs.index("Tiers") < secs.index("Evidence levels")
    cl = ss[ss.section == "Claims"].set_index("metric")["value"]
    assert list(cl.index) == list(EV.CLAIM_KEYS)
    assert cl["identified"].startswith("1  (25% of assignments, 19% of assigned signal) -- ")
    assert cl["identified"].endswith(EV.CLAIM_MEANING["identified"])
    assert cl["not assessed"].startswith("0  (0% of assignments")
    # the Tiers rows keep their denominators and their form
    assert ss[(ss.section == "Tiers") & (ss.metric == T.TIER_ASSIGNED)].value.iloc[0].startswith("3  (75% of")
    ev = ss[ss.section == "Evidence levels"].set_index("metric")["value"]
    assert ev["scale"] == R.SCALE_NAME and R.SCALE_NAME == f"the evidence scale of peaky {EV.SCALE_RELEASE}"
    assert list(ev.index) == ["scale", "3c", "4a", "4b", "5a"]
    assert ev["4a"].startswith("1  (25% of assignments) -- ")


def test_the_read_me_opens_on_the_claims_and_names_every_level_and_column():
    rm = R.build_sheets(_ledger())["Read me"]
    assert rm.section.iloc[0] == "Claims" and rm.topic.iloc[0] == "claim"
    block = rm[rm.section == "Claims"]
    assert set(EV.CLAIM_KEYS) <= set(block.topic)
    assert block.explanation.str.contains("n_identified|tier-Assigned").any()
    lv = rm[rm.section == "Evidence levels"]
    assert {*EV.LEVEL_ORDER, *EV.BUCKETS} <= set(lv.topic)
    assert set(EV.COLUMNS) - {"claim"} <= set(lv.topic)
    text = " ".join(rm.explanation.astype(str))
    for gone in (*OLD_COLUMNS, "2b", "3a", "4c", "4d", "axes"):
        assert gone not in text, gone
    assert "'tier_reason' column" in text and "'residual_evidence'" in text
    assert R.legend_sheet().section.iloc[0] == "Tiers"                         # the default is unchanged


def test_the_by_evidence_level_sheet_runs_over_the_levels_and_buckets_with_tag_kinds():
    lv = R.build_sheets(_ledger())["By evidence level"]
    summ = lv[lv.section == "summary"].set_index("level")
    assert list(summ.index) == ["3c", "4a", "4b", "5a"]
    assert summ.loc["4a", "tag_kinds"] == "lead: 1; side channels locked: 1"
    assert summ.loc["5a", "tag_kinds"] == "(none): 1"
    assert summ.loc["3c", "tag_kinds"] == "class list: 1"
    assert summ.loc["4a", "share"] == pytest.approx(0.25)
    bright = lv[lv.section == "brightest 5a"].iloc[0]
    assert bright["competitors_left"] == "C9H12O5 [M-H]-" and bright["evidence"].startswith("5a · ")
    assert not set(OLD_COLUMNS) & set(lv.columns) and "axes" not in lv.columns
    # a bucket and a committed row with no level get their own rows
    led = _ledger()
    led.loc[led.peak_id == "F", "evidence_level"] = "reagent"
    led.loc[led.peak_id == "C", ["evidence_level", "claim"]] = [pd.NA, "tentative"]
    summ = R.evidence_level_sheet(led[led.role == L.ROLE_M0]).query("section == 'summary'").set_index("level")
    assert list(summ.index) == ["3c", "4a", "reagent", "no level"]


def test_tag_kind_reads_the_words_before_the_detail():
    assert R.tag_kind("two routes: [M-H]- + [M+NO3]- (r 0.9) -- tag; x") == "two routes"
    assert R.tag_kind("named list (not used: split open): reflist:x") == "named list"
    assert R.tag_kind("side channels locked") == "side channels locked"
    assert R.tag_kind("series exclusion x2") == "series exclusion"
    assert R.tag_kinds("ladder: CH2 x3 -- tag | lead (a tentative lead) | ladder: CF2") == ["ladder", "lead"]
    assert R.tag_kinds(pd.NA) == [] and R.tag_kinds("") == []


def test_write_excel_round_trip_opens_on_by_claim_with_claim_and_level_chips(tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    path = tmp_path / "t.xlsx"
    R.write_excel(_ledger(), path, "ambient-air", sample_id="T")
    assert pd.ExcelFile(path).sheet_names[0] == "By claim"
    wb = openpyxl.load_workbook(path)
    ws = wb["By claim"]
    head = [c.value for c in ws[1]]
    j = head.index("claim") + 1
    fills = {ws.cell(row=i, column=j).value: ws.cell(row=i, column=j).fill.start_color.rgb[-6:]
             for i in range(2, 2 + len(EV.CLAIM_KEYS))}
    assert fills == {"identified": R._FILL["good"][0], "neutral": R._FILL["okay"][0],
                     "ion": R._FILL["info"][0], "tentative": R._FILL["warn"][0],
                     "reagent": R._FILL["neutral"][0], "not assessed": R._FILL["neutral"][0]}
    ws = wb["Assigned"]
    head = [c.value for c in ws[1]]
    j = head.index("evidence_level") + 1
    chips = {ws.cell(row=i, column=j).value: ws.cell(row=i, column=j).fill.start_color.rgb[-6:]
             for i in range(2, ws.max_row + 1)}
    assert chips == {"4a": R._FILL["okay"][0], "5a": R._FILL["warn"][0], "4b": R._FILL["info"][0]}
    # the long strings wrap at their widths; the level's evidence never gets a verdict chip
    k = head.index("evidence") + 1
    assert ws.column_dimensions[openpyxl.utils.get_column_letter(k)].width == R._WRAP_COLS["evidence"]
    assert ws.cell(row=2, column=k).alignment.wrap_text
    assert ws.cell(row=2, column=k).fill.start_color.rgb in (None, "00000000")
    assert "tier_reason" in head and head.count("evidence") == 1


def test_every_level_and_bucket_has_a_chip_and_every_claim_key_a_colour():
    for lv in [*EV.LEVELS, *EV.BUCKETS]:
        assert lv in R._LEVEL_CHIP, lv
    for c in EV.CLAIM_KEYS:
        assert c in R._CLAIM_CHIP and c in PR._CLAIM_COLOUR and c in PR._CLAIM_RANK, c
    for lv in [*EV.LEVELS, *EV.BUCKETS]:
        assert lv in PR._LEVEL_COLOUR, lv


def test_markdown_carries_the_claims_line_and_the_row_tags(tmp_path):
    led = _ledger()
    result = {"ledger": led, "stats": L.stats(led), "sample_id": "T", "context": "ambient-air",
              "prescan": {}, "problems": []}
    result["stats"]["by_tier"] = led[led.role == L.ROLE_M0].tier.value_counts().to_dict()
    md = R.write_markdown(result, tmp_path / "t.md").read_text()
    lines = md.splitlines()
    ci = next(i for i, ln in enumerate(lines) if ln.startswith("- Claims:"))
    assert lines[ci] == "- Claims: identified 1 | neutral 1 | ion 1 | tentative 1"
    assert lines[ci + 1].startswith("- Tiers:")
    assert f"[{T.TIER_ASSIGNED} · tentative]" in md and f"[{T.TIER_CANDIDATE} · identified]" in md
    assert f"[{T.TIER_ASSIGNED} · neutral]" in md
    plain = _ledger(levels=False)
    res0 = dict(result, ledger=plain, stats=L.stats(plain))
    md0 = R.write_markdown(res0, tmp_path / "t0.md").read_text()
    assert "- Claims:" not in md0 and " · " not in md0


# --------------------------------------------------------------------------- NA and older scales
def test_a_not_assessed_ledger_keeps_na_through_a_csv_round_trip(tmp_path):
    led = _real_ledger()
    EV.apply_levels(led, run_inputs=_inputs(resolution=9_000.0))       # a TOF-class file
    back = _round_trip(led, tmp_path)
    assert back.loc[back.role == "M0", "evidence_level"].isna().all()   # the default parser ate NA
    view, info = R.scale_view(back)
    assert (view.loc[view.role == "M0", "evidence_level"] == "NA").all()
    assert info["n_na"] == 2 and info["na_reason"].startswith("not assessed on this instrument class")
    assert view.loc[view.role != "M0", "evidence_level"].isna().all()
    sheets = R.build_sheets(back)
    summ = sheets["By claim"].query("section == 'summary'").set_index("claim")
    assert summ.loc["not assessed", "n"] == 2 and summ.loc["tentative", "n"] == 0
    assert summ.loc["not assessed", "level_hist"] == "NA: 2"
    ev = sheets["Summary"].query("section == 'Evidence levels'").set_index("metric")["value"]
    assert "width model R(200) = 9 000 < 50 000" in ev["not assessed"]
    assert ev["NA"].startswith("2  (100% of assignments)")
    assert sheets["Assigned"]["evidence_level"].eq("NA").all()
    result = {"ledger": back, "stats": L.stats(back), "sample_id": "T", "context": "ambient-air",
              "prescan": {}, "problems": []}
    md = R.write_markdown(result, tmp_path / "na.md").read_text()
    assert "- Claims: identified 0 | neutral 0 | ion 0 | tentative 0 | not assessed 2" in md
    assert "- Evidence levels: 2 of 2 not assessed -- not assessed on this instrument class" in md


def _b_series(led: pd.DataFrame) -> pd.DataFrame:
    """The ledger as a peaky before the evidence scale wrote it: B-series letters,
    their axes / reason / isomer columns and the claims read on that scale."""
    old = led.copy()
    for c in ("evidence", "would_lift", "competitors_left", "tags", "context", "context_source"):
        old = old.drop(columns=c)
    letters = {"A": "2b", "C": "5b", "D": "4c", "F": "4a"}
    for pid, lv in letters.items():
        i = old.index[old["peak_id"] == pid][0]
        old.at[i, "evidence_level"] = lv
        old.at[i, "claim"] = {"2b": "identified", "4a": "identified", "4c": "ion"}.get(lv, "tentative")
    old["evidence_axes"] = old["evidence_level"].map(
        lambda v: "iso|chan2" if isinstance(v, str) and v in ("2b", "4a") else pd.NA)
    old["level_reason"] = old["evidence_level"].map(lambda v: f"{v}: test" if isinstance(v, str) else pd.NA)
    old["n_plausible_structures"] = pd.NA
    return old


def test_an_older_scale_ledger_renders_every_letter_as_no_level_and_says_so(tmp_path):
    old = _round_trip(_b_series(_ledger()), tmp_path)
    view, info = R.scale_view(old)
    assert info["old_scale"] and info["n_unknown"] == 4 and info["unknown"] == ["2b", "4a", "4c", "5b"]
    assert "claim" not in view.columns                                    # read on the old scale: dropped
    assert R.OLD_SCALE_NO_LEVEL == f"no level (pre-{EV.SCALE_RELEASE} scale)"
    sheets = R.build_sheets(old, "ambient-air", sample_id="T")
    own = sheets["Peak ownership"].set_index("peak_id")
    assert pd.isna(own.loc["A", "evidence_level"]) and own.loc["A", "claim"] == "tentative"   # 2b: unknown
    assert pd.isna(own.loc["D", "evidence_level"]) and own.loc["D", "claim"] == "tentative"   # 4c: unknown
    # a letter both scales share is no level either: an old 4a is not a 4a of this scale
    assert pd.isna(own.loc["F", "evidence_level"]) and own.loc["F", "claim"] == "tentative"
    assert pd.isna(own.loc["C", "evidence_level"]) and own.loc["C", "claim"] == "tentative"   # 5b
    for name, df in sheets.items():
        assert not set(OLD_COLUMNS) & set(df.columns), name
    ev = sheets["Summary"].query("section == 'Evidence levels'").set_index("metric")["value"]
    assert "older scale" in ev.index and "its 4 levelled row(s)" in ev["older scale"]
    assert "(2b, 4a, 4c, 5b;" in ev["older scale"] and R.OLD_SCALE_NO_LEVEL in ev["older scale"]
    assert ev[R.OLD_SCALE_NO_LEVEL].startswith("4  (100% of assignments)") and "no level" not in ev.index
    assert not set(EV.LEVELS) & set(ev.index)
    summ = sheets["By evidence level"].query("section == 'summary'").set_index("level")
    assert list(summ.index) == [R.OLD_SCALE_NO_LEVEL]
    hist = sheets["By claim"].query("section == 'summary'").set_index("claim")["level_hist"]
    assert hist["tentative"] == f"{R.OLD_SCALE_NO_LEVEL}: 4"
    R.write_excel(old, tmp_path / "old.xlsx", "ambient-air", sample_id="T")
    result = {"ledger": old, "stats": L.stats(old), "sample_id": "T", "context": "ambient-air",
              "prescan": {}, "problems": []}
    md = R.write_markdown(result, tmp_path / "old.md").read_text()
    assert "- Claims: identified 0 | neutral 0 | ion 0 | tentative 4" in md
    assert "levelled on a scale older than the evidence scale of peaky" in md
    # an older ledger whose letters all exist here is still flagged by its columns, and every letter goes
    led = _ledger().drop(columns=["claim"]).assign(evidence_axes="iso")
    view, info = R.scale_view(led)
    n_lettered = int(led["evidence_level"].notna().sum())
    assert info["old_scale"] and n_lettered and info["n_unknown"] == n_lettered
    assert view["evidence_level"].isna().all()


def test_the_stage_output_renders_and_the_workbook_writes(tmp_path):
    led = _real_ledger()
    before = R.build_sheets(led, "ambient-air", sample_id="T")
    assert "By evidence level" not in before and "By claim" not in before
    for sheet in ("Assigned", "Candidates", "Target list", "Peak ownership"):
        assert "evidence_level" not in before[sheet].columns, sheet
    EV.apply_levels(led, run_inputs=_inputs())
    after = R.build_sheets(led, "ambient-air", sample_id="T")
    keys = list(after)
    assert keys[0] == "By claim" and keys.index("By evidence level") == keys.index("Candidates") + 1
    for sheet in ("Assigned", "Candidates", "Target list", "Peak ownership"):
        assert set(EV.COLUMNS) <= set(after[sheet].columns), sheet
    a = after["Assigned"].set_index("neutral_formula")
    for nf in a.index:
        lv = a.loc[nf, "evidence_level"]
        assert lv in EV.LEVELS and a.loc[nf, "evidence"].startswith(lv + " · ")
        assert " · context source: " in a.loc[nf, "evidence"]
    summ = after["By evidence level"].query("section == 'summary'")
    assert set(summ.level) <= set(EV.LEVELS) | set(EV.BUCKETS) and summ.n.sum() == 2
    assert after["Summary"].section.eq("Evidence levels").any()
    path = tmp_path / "t.xlsx"
    R.write_excel(led, path, "ambient-air", sample_id="T")
    names = pd.ExcelFile(path).sheet_names
    assert names[0] == "By claim" and "By evidence level" in names


# --------------------------------------------------------------------------- the PDF
MERGED = [  # neutral, adduct, tier, level, ion_only_of
    ("C10H16O4", "[M-H]-", "Assigned", "4a", None),
    ("C8H12O5", "[M-H]-", "Candidate", "3c", None),        # Candidate but identified
    ("C5H8O3", "[M-H]-", "Assigned", "4b", None),
    ("C9H14O4", "[M]-.", "Candidate", "4b", "p9"),         # an ion-only row
    ("C7H10O4", "[M-H]-", "Assigned", "5b", None),         # Assigned but tentative
    ("C10H19NO4", "[M+NO3]-", "Candidate", "5a", None),
    ("C6H6O3", "[M-H]-", "Candidate", None, None),         # a batch-level re-read: no level
]
HEIGHTS = {("C10H16O4", "[M-H]-"): 1000, ("C8H12O5", "[M-H]-"): 500, ("C5H8O3", "[M-H]-"): 300,
           ("C9H14O4", "[M]-."): 200, ("C7H10O4", "[M-H]-"): 100, ("C10H19NO4", "[M+NO3]-"): 100,
           ("C3H4O4", "[M-H]-"): 800}                        # the last is in no merged row
#: enough confirmed isotopes on the first channel to fill the appendix's isotopes field
ISO = ["13C", "13C2", "18O", "13C18O", "34S", "81Br", "37Cl", "15N", "29Si", "30Si", "2H"]
#: the B-series letters an older run wrote, row for row
B_SERIES = ["2b", "3a", "4b", "4d", "5b", "5a", None]


def _run_dir(d: Path, *, claim_col: bool = True, levels: bool = True, merged: list = MERGED,
             old: bool = False) -> None:
    """A batch run dir: merged_ledger.csv (`merged` rows) + two per-file ledgers whose
    M0 heights total 3000 (800 of them in a reading no merged row carries). With
    `old`, the merged ledger is levelled on the B-series scale."""
    (d / "per_file").mkdir(parents=True, exist_ok=True)
    (d / "tables").mkdir(exist_ok=True)
    rows = []
    for k, (nf, ad, tier, lv, io) in enumerate(merged):
        r = dict(mz=150.0 + 10 * k, neutral_formula=nf, adduct=ad, tier=tier, ion_score=0.9 - 0.05 * k,
                 n_files=2, formula_agree=True, ion_only_of=io)
        if levels and old:
            b = B_SERIES[k]
            r.update(evidence_level=b, evidence_axes="iso" if b else None, level_reason=f"{b}: test",
                     n_plausible_structures=None,
                     claim={"2b": "identified", "3a": "identified", "4b": "ion", "4d": "ion"}.get(b, "tentative"))
        elif levels:
            r.update(evidence_level=lv,
                     evidence=(NA_TEXT if lv == "NA" else f"{lv} · test" if lv else EV.NO_POOLED_PAIR_TEXT),
                     would_lift="", competitors_left="", tags="", context="", context_source="")
            if claim_col:
                r["claim"] = EV.claim_class(lv)
        rows.append(r)
    pd.DataFrame(rows).to_csv(d / "merged_ledger.csv", index=False)
    pf = [dict(peak_id=f"p{k}", mz=150.0 + 10 * k, role="M0", neutral_formula=nf, adduct=ad, height=h,
               tier="Assigned", isotopologues=json.dumps([{"label": lab} for lab in ISO]) if k == 0 else "")
          for k, ((nf, ad), h) in enumerate(HEIGHTS.items())]
    extra = [dict(peak_id="i1", mz=151.0, role="iso_child", height=5000),
             dict(peak_id="u1", mz=411.0, role="unexplained", height=7000)]
    pd.DataFrame(pf[:4] + extra[:1]).to_csv(d / "per_file" / "s1_ledger.csv", index=False)
    pd.DataFrame(pf[4:] + extra[1:]).to_csv(d / "per_file" / "s2_ledger.csv", index=False)


class _Pdf:
    def __init__(self):
        self.pages = 0
        self.figs = []

    def savefig(self, fig):
        self.pages += 1
        self.figs.append(fig)


def _lines(section, ctx, monkeypatch) -> list:
    """The (style, text) lines a section hands to _text_lines, in order."""
    got = []
    orig = PR._text_lines
    monkeypatch.setattr(PR, "_text_lines", lambda fig, lines, **kw: (got.extend(lines), orig(fig, lines, **kw))[1])
    section(ctx, _Pdf())
    monkeypatch.setattr(PR, "_text_lines", orig)
    return [(s, t) for s, t in got if s != "gap"]


def _zeros(**kw) -> dict:
    return {**{c: 0 for c in EV.CLAIM_KEYS}, **kw}


def test_load_context_counts_the_claims_by_tier_and_by_signal(tmp_path):
    _run_dir(tmp_path)
    ctx = PR.load_context(str(tmp_path), tag="X", label="X")
    assert ctx["claims"] == _zeros(identified=1, neutral=1, ion=2, tentative=3)
    assert ctx["evidence_levels"] == {"3c": 1, "4a": 1, "4b": 2, "5a": 1, "5b": 1}
    assert ctx["claim_by_tier"] == {
        "Assigned": _zeros(neutral=1, ion=1, tentative=1),
        "Candidate": _zeros(identified=1, tentative=2),
        "ion-only": _zeros(ion=1)}                                       # kept out of Candidate
    assert ctx["claim_of_pair"][("C8H12O5", "[M-H]-")] == "identified"
    assert ctx["n_unlevelled"] == 1
    sig = ctx["claim_signal"]
    assert set(sig) == {*EV.CLAIM_KEYS, "unmatched"} and sum(sig.values()) == pytest.approx(1.0)
    assert sig["identified"] == pytest.approx(500 / 3000) and sig["neutral"] == pytest.approx(1000 / 3000)
    assert sig["ion"] == pytest.approx(500 / 3000) and sig["tentative"] == pytest.approx(200 / 3000)
    assert sig["unmatched"] == pytest.approx(800 / 3000)
    assert ctx["claim_signal_assigned"] == pytest.approx(
        _zeros(neutral=1000 / 3000, ion=300 / 3000, tentative=100 / 3000))
    assert ctx["claim_signal_by_tier"]["ion-only"]["ion"] == pytest.approx(200 / 3000)


def test_the_assigned_signal_is_the_crosstab_row_and_keeps_ion_only_rows_apart(tmp_path):
    # an ion-only row the merge left Assigned stays in its own row, out of the Assigned signal
    rows = [m if m[4] is None else (m[0], m[1], "Assigned", m[3], m[4]) for m in MERGED]
    _run_dir(tmp_path, merged=rows)
    ctx = PR.load_context(str(tmp_path), tag="X", label="X")
    assert ctx["claim_signal_assigned"] == ctx["claim_signal_by_tier"]["Assigned"]
    assert ctx["claim_signal_assigned"]["ion"] == pytest.approx(300 / 3000)
    assert ctx["claim_signal_by_tier"]["ion-only"]["ion"] == pytest.approx(200 / 3000)


def test_an_older_merged_ledger_without_the_column_gets_the_claims_from_the_level(tmp_path):
    _run_dir(tmp_path / "new")
    _run_dir(tmp_path / "old", claim_col=False)
    new = PR.load_context(str(tmp_path / "new"), tag="X", label="X")
    old = PR.load_context(str(tmp_path / "old"), tag="X", label="X")
    for k in ("claims", "claim_by_tier", "claim_of_pair", "claim_signal"):
        assert old[k] == new[k], k
    assert list(old["merged"]["claim"]) == [EV.claim_class(m[3]) for m in MERGED]


def test_the_claims_page_leads_and_renders_one_page_or_none(tmp_path):
    assert PR.SECTIONS.index(PR.claims) == 1
    assert PR.SECTIONS.index(PR.evidence_levels) == PR.SECTIONS.index(PR.coverage) + 1
    _run_dir(tmp_path / "with")
    ctx = PR.load_context(str(tmp_path / "with"), tag="X", label="X")
    pdf = _Pdf(); PR.claims(ctx, pdf); assert pdf.pages == 1
    ax = pdf.figs[0].axes[0]
    assert ax.get_title(loc="left") == "Merged rows and committed signal by claim"
    assert ax.get_xlabel() == "% of merged rows / of all per-file M0 height"
    labels = [t.get_text() for t in pdf.figs[0].legends[0].get_texts()]
    assert labels == [*EV.CLAIMS, "no merged row"]                       # no bucket the run lacks
    no_sig = {k: v for k, v in ctx.items() if not k.startswith("claim_signal")}
    pdf = _Pdf(); PR.claims(no_sig, pdf); assert pdf.pages == 1          # no per-file data: still a page
    ax = pdf.figs[0].axes[0]
    assert ax.get_title(loc="left") == "Merged rows by claim" and ax.get_xlabel() == "% of merged rows"
    bare = dict(ctx, merged=ctx["merged"].drop(columns=["claim"]))
    pdf = _Pdf(); PR.claims(bare, pdf); assert pdf.pages == 1            # the page reads the level itself
    _run_dir(tmp_path / "without", levels=False)
    ctx0 = PR.load_context(str(tmp_path / "without"), tag="X", label="X")
    assert "claims" not in ctx0 and "claim_signal" not in ctx0
    pdf = _Pdf(); PR.claims(ctx0, pdf); assert pdf.pages == 0


def test_the_claims_page_lists_both_disagreements_and_the_unmatched_bucket(tmp_path, monkeypatch):
    _run_dir(tmp_path)
    ctx = PR.load_context(str(tmp_path), tag="X", label="X")
    text = [t for _s, t in _lines(PR.claims, ctx, monkeypatch)]
    assert any(t.startswith("no merged row") and "26.7%" in t for t in text)
    assert any("1 Assigned but tentative · 1 Candidate but identified" in t for t in text)
    rows = [t for t in text if "C7H10O4" in t or "C8H12O5" in t]
    assert [r.split()[1] for r in rows] == ["C7H10O4", "C8H12O5"]         # in the count line's order
    assert any(t.startswith("ion-only") for t in text)
    mono = [t for s, t in _lines(PR.claims, ctx, monkeypatch) if s == "m"]
    table = [t for t in mono if t.split()[:1] and t.split()[0] in EV.CLAIMS]
    assert [t.split()[0] for t in table[:4]] == list(EV.CLAIMS)
    assert table[0].split()[-1] == "3c" and table[1].split()[-1] == "4a"
    joined = " ".join(text)
    for gone in ("1-4a", "4b-4d", "2b", "4c", "4d"):
        assert gone not in joined, gone


def _parting(n_at: int, n_ci: int, *, heights: bool = True) -> dict:
    """A claims-page context with `n_at` Assigned-but-tentative rows, every one
    brighter (and higher-scoring) than the `n_ci` Candidate-but-identified rows."""
    rows, mx = [], {}
    for k in range(n_at):
        rows.append(dict(mz=200.0 + k, neutral_formula=f"C{10 + k}H20O2", adduct="[M-H]-", tier="Assigned",
                         ion_score=0.90 + k / 1000, evidence_level="5b"))
        mx[(f"C{10 + k}H20O2", "[M-H]-")] = 1000.0 + k
    for k in range(n_ci):
        rows.append(dict(mz=300.0 + k, neutral_formula=f"C{10 + k}H18O5", adduct="[M-H]-", tier="Candidate",
                         ion_score=0.50 + k / 1000, evidence_level="3c"))
        mx[(f"C{10 + k}H18O5", "[M-H]-")] = 10.0 + k
    merged = pd.DataFrame(rows)
    merged["claim"] = merged["evidence_level"].map(EV.claim_class)
    ctx = {"merged": merged, "claims": EV.summarize_claims(merged["claim"])}
    if heights:
        ctx["max_h_by_channel"] = mx
    return ctx


def _parted(ctx, monkeypatch) -> tuple:
    """(count line, table header, [(neutral, tier) of each listed row]) off the claims page."""
    lines = _lines(PR.claims, ctx, monkeypatch)
    k = [t for _s, t in lines].index("Where tier and claim part")
    head = next(t for s, t in lines[k:] if s == "m")
    rows = [t.split() for s, t in lines[k:] if s == "m" and t is not head and t.split()[0][0].isdigit()]
    return lines[k + 1][1], head, [(r[1], r[3]) for r in rows]


def test_neither_direction_of_the_parting_pushes_the_other_off_the_list(monkeypatch):
    count, head, rows = _parted(_parting(14, 20), monkeypatch)
    assert count == "14 Assigned but tentative · 20 Candidate but identified -- the brightest 6 of each below"
    assert [t for _n, t in rows] == ["Assigned"] * 6 + ["Candidate"] * 6
    assert [n for n, _t in rows[:6]] == [f"C{10 + k}H20O2" for k in range(13, 7, -1)]   # brightest first
    assert [n for n, _t in rows[6:]] == [f"C{10 + k}H18O5" for k in range(19, 13, -1)]
    # a direction with fewer than 6 hands its unused slots to the other
    count, _head, rows = _parted(_parting(3, 20), monkeypatch)
    assert count.endswith(" -- the brightest 3 and 9 below")
    assert [t for _n, t in rows] == ["Assigned"] * 3 + ["Candidate"] * 9
    count, _head, rows = _parted(_parting(20, 0), monkeypatch)
    assert count.endswith(" -- the brightest 12 below") and len(rows) == 12


def test_without_heights_the_parting_is_ranked_and_labelled_by_ion_score(monkeypatch):
    count, head, rows = _parted(_parting(14, 20, heights=False), monkeypatch)
    assert count.endswith(" -- the 6 of each with the highest ion score below")
    assert "brightest" not in count and "max cps" not in head and head.split()[-1] == "score"
    assert len(rows) == 12
    lines = [t for _s, t in _lines(PR.claims, _parting(14, 20, heights=False), monkeypatch)]
    assert any(t.split()[:2] == ["319.0000", "C29H18O5"] and t.split()[-1] == "0.52" for t in lines)


def test_the_cover_leads_with_the_claims_line(tmp_path, monkeypatch):
    _run_dir(tmp_path)
    ctx = PR.load_context(str(tmp_path), tag="X", label="X")
    lines = _lines(PR.cover, ctx, monkeypatch)
    k = [t for _s, t in lines].index("Summary")
    assert lines[k + 1] == ("b", "Claims: 1 identified · 1 neutral · 2 ion · 3 tentative   (of 7 merged rows)")
    style, dim = lines[k + 2]
    assert style == "dim" and dim == ("   identified carries 17% of the committed-peak signal, neutral 33%, "
                                      "ion 17%, tentative 7% -- see the Claims page")
    for t in (dim, lines[k + 3][1]):
        assert "more" not in t and "[cover]" not in t and "[residual]" not in t
    assert lines[k + 3] == ("dim", "   the other 27% is per-file signal with no merged row")
    assert lines[k + 4][1].startswith("Unique analytes assigned (M0):")     # the tier line stays
    lv = [t for _s, t in lines if t.startswith("Evidence levels")]
    assert lv == ["Evidence levels (3c best .. 5b):  3c 1 · 4a 1 · 4b 2 · 5a 1 · 5b 1"]
    assert any(R.SCALE_NAME in t for _s, t in lines)
    _run_dir(tmp_path / "plain", levels=False)
    plain = _lines(PR.cover, PR.load_context(str(tmp_path / "plain"), tag="X", label="X"), monkeypatch)
    k0 = [t for _s, t in plain].index("Summary")
    assert plain[k0 + 1][1].startswith("Unique analytes assigned (M0):")


def test_findings_methods_and_the_appendix_carry_the_claim_only_when_levelled(tmp_path, monkeypatch):
    _run_dir(tmp_path)
    ctx = PR.load_context(str(tmp_path), tag="X", label="X")
    f = [t for s, t in _lines(PR.findings, ctx, monkeypatch) if s == "m"]
    assert f[0].split() == ["share", "class", "claim", "neutral"]
    assert any(t.split()[2] == "neutral" and t.split()[3] == "C10H16O4" for t in f[1:])
    m = " ".join(t for _s, t in _lines(PR.methods, ctx, monkeypatch))
    assert "read from the evidence level alone" in m and "Claims page" in m and R.SCALE_NAME in m
    app = _lines(PR.assignments_table, ctx, monkeypatch)
    head = app[0][1]
    assert head.split()[3:5] == ["tier", "claim"]
    _run_dir(tmp_path / "plain", levels=False)
    ctx0 = PR.load_context(str(tmp_path / "plain"), tag="X", label="X")
    f0 = [t for s, t in _lines(PR.findings, ctx0, monkeypatch) if s == "m"]
    assert f0[0] == "   share   class   neutral"
    app0 = _lines(PR.assignments_table, ctx0, monkeypatch)
    assert "claim" not in app0[0][1]
    full = [t for _s, t in app if t.startswith("C10H16O4")]
    full0 = [t for _s, t in app0 if t.startswith("C10H16O4")]
    # the claim column ('not assessed' fits) widens the row; the isotopes field keeps its 36 characters
    assert full[0].endswith("…") and len(full[0]) == len(full0[0]) + 13 and full[0][-36:] == full0[0][-36:]
    labs = {"13C", "18O", "34S", "13C2", "37Cl", "13C18O"}               # 33 characters joined
    ctx["iso_by_channel"][("C8H12O5", "[M-H]-")] = labs
    app = _lines(PR.assignments_table, ctx, monkeypatch)
    assert any(t.startswith("C8H12O5") and t.endswith("  13C, 18O, 34S, 13C2, 37Cl, 13C18O") for _s, t in app)
    assert "Claims page" not in " ".join(t for _s, t in _lines(PR.methods, ctx0, monkeypatch))


def test_the_levels_page_renders_every_level_and_bucket_and_is_skipped_without(tmp_path, monkeypatch):
    every = [("C10H16O4", "[M-H]-", "Assigned", lv, None) for lv in [*EV.LEVELS, *EV.BUCKETS]]
    every = [(f"C{10 + k}H16O4", a, t, lv, io) for k, (_n, a, t, lv, io) in enumerate(every)]
    _run_dir(tmp_path / "all", merged=every)
    ctx = PR.load_context(str(tmp_path / "all"), tag="X", label="X")
    assert ctx["evidence_levels"] == {k: 1 for k in [*EV.LEVELS, *EV.BUCKETS]}
    assert ctx["claims"] == _zeros(identified=1, neutral=1, ion=1, tentative=2, reagent=1, **{"not assessed": 1})
    pdf = _Pdf(); PR.evidence_levels(ctx, pdf); assert pdf.pages == 1
    ax = pdf.figs[0].axes[0]
    assert [t.get_text() for t in ax.get_xticklabels()] == [*EV.LEVELS, *EV.BUCKETS]
    text = [t for _s, t in _lines(PR.evidence_levels, ctx, monkeypatch)]
    assert [t.split()[0] for t in text if t.split()[:1] and t.split()[0] in [*EV.LEVELS, *EV.BUCKETS]] \
        == [*EV.LEVELS, *EV.BUCKETS]
    joined = " ".join(text)
    for gone in ("four axes", "corroborating source", "4d", "2a"):
        assert gone not in joined, gone
    # what the page says the tags do, and how the merged rows are stamped, as the code does it: a route / ladder /
    # partner never unlocks a level but can anchor the series exclusion; the join is by (neutral, adduct)
    assert "never move the level" not in joined and "stamped by ion" not in joined
    assert "facts that never unlock a level" in joined and "can anchor the series exclusion" in joined
    assert "stamped by (neutral, adduct) reading" in joined
    tags_doc = dict(R._LEVEL_COLUMN_LEGEND)["tags"]
    assert tags_doc.startswith("Facts that never unlock a level") and "series exclusion" in tags_doc
    cl = [t for _s, t in _lines(PR.claims, ctx, monkeypatch)]
    assert any(t.startswith("• 1 merged row(s) are not assessed on this instrument class (width model R(200)")
               and t.count("not assessed on this instrument class") == 1 for t in cl)
    assert any(t.startswith("• 1 merged row(s) are reagent ions") for t in cl)
    out = PR.build(str(tmp_path / "all"), tag="X", label="X", out_pdf=str(tmp_path / "all.pdf"),
                   sections=[PR.cover, PR.claims, PR.evidence_levels, PR.findings, PR.methods,
                             PR.assignments_table])
    assert Path(out).stat().st_size > 0
    _run_dir(tmp_path / "without", levels=False)
    ctx0 = PR.load_context(str(tmp_path / "without"), tag="X", label="X")
    assert "evidence_levels" not in ctx0
    pdf = _Pdf(); PR.evidence_levels(ctx0, pdf); assert pdf.pages == 0


def test_a_run_not_assessed_says_so_on_the_cover_and_the_levels_page(tmp_path, monkeypatch):
    na_rows = [(nf, ad, tier, "NA", io) for nf, ad, tier, _lv, io in MERGED]
    _run_dir(tmp_path / "keep", merged=na_rows)
    _run_dir(tmp_path / "noclaim", merged=na_rows, claim_col=False)      # NA re-read as written
    for d in ("keep", "noclaim"):
        ctx = PR.load_context(str(tmp_path / d), tag="X", label="X")
        assert ctx["evidence_levels"] == {"NA": 7}, d
        assert ctx["claims"] == _zeros(**{"not assessed": 7}) and ctx["n_unlevelled"] == 0, d
        assert ctx["levels_info"]["na_reason"].endswith("(width model R(200) = 9 652 < 50 000)"), d
    lines = _lines(PR.cover, ctx, monkeypatch)
    k = [t for _s, t in lines].index("Summary")
    assert lines[k + 1] == ("b", "Claims: not assessed on this instrument class   (of 7 merged rows)")
    assert "R(200) = 9 652 < 50 000" in lines[k + 2][1]
    text = [t for _s, t in _lines(PR.evidence_levels, ctx, monkeypatch)]
    assert any(t.startswith("Not assessed on this instrument class (width model R(200) = 9 652 < 50 000). ")
               for t in text)
    assert lines[k + 2][1] == ("   width model R(200) = 9 652 < 50 000: the evidence scale needs an "
                               "Orbitrap-class width model; the tier below is the run's verdict")
    pdf = _Pdf(); PR.claims(ctx, pdf); assert pdf.pages == 1


def test_an_older_scale_run_renders_and_says_so(tmp_path, monkeypatch):
    _run_dir(tmp_path, old=True)
    ctx = PR.load_context(str(tmp_path), tag="X", label="X")
    info = ctx["levels_info"]
    assert info["old_scale"] and info["unknown"] == ["2b", "3a", "4b", "4d", "5a", "5b"] and info["n_unknown"] == 6
    # every old letter is no level here, the shared ones (4b, 5a, 5b) included
    assert ctx["evidence_levels"] == {}
    # the stored claims (read on the old scale) are dropped and re-read here: all tentative
    assert ctx["claims"] == _zeros(tentative=7)
    assert ctx["n_unlevelled"] == 7
    cover = " ".join(t for _s, t in _lines(PR.cover, ctx, monkeypatch))
    assert "this run was levelled on a scale older than the evidence scale of peaky" in cover
    assert "(2b, 3a, 4b, 4d, 5a, 5b;" in cover
    page = " ".join(t for _s, t in _lines(PR.evidence_levels, ctx, monkeypatch))
    assert f"7 merged row(s) carry {R.OLD_SCALE_NO_LEVEL}" in page
    out = PR.build(str(tmp_path), tag="X", label="X", out_pdf=str(tmp_path / "old.pdf"),
                   sections=[PR.cover, PR.claims, PR.evidence_levels, PR.findings, PR.assignments_table])
    assert Path(out).stat().st_size > 0


# --------------------------------------------------------------------------- publish
def test_publish_provenance_carries_the_scale_columns_and_drops_empty_ones():
    for c in EV.COLUMNS:
        assert c in P._ENGINE_PROVENANCE_COLUMNS, c
    for c in OLD_COLUMNS:
        assert c not in P._ENGINE_PROVENANCE_COLUMNS, c
    row = pd.Series({"tier": "Assigned", "evidence_level": "4b",
                     "evidence": "4b · ion: unique in the calibrated window", "would_lift": "4a needs ...",
                     "competitors_left": "", "tags": pd.NA, "context": "", "context_source": "x: always active",
                     "claim": "ion", "confidence": "High"})
    out = P._engine_provenance(row, "assigned")
    assert out["evidence_level"] == "4b" and out["claim"] == "ion"
    assert out["evidence"].startswith("4b · ") and out["context_source"] == "x: always active"
    for c in ("competitors_left", "tags", "context"):
        assert c not in out, c


def _publish_ledger(levels: list, claims: list | None, **extra) -> pd.DataFrame:
    rows = []
    for k, lv in enumerate(levels):
        r = {"peak_id": f"P{k:018d}", "mz": 150.0 + k, "role": "M0", "neutral_formula": "C6H6",
             "ion_score": 0.9, "height": 10.0 + k, "tier": "Assigned", "evidence_level": lv}
        if claims is not None:
            r["claim"] = claims[k]
        r.update(extra)
        rows.append(r)
    return pd.DataFrame(rows)


def _published(led: pd.DataFrame, tmp_path: Path) -> tuple[list, dict]:
    back = _round_trip(led, tmp_path, "pub.csv")
    rows, summary = P.build_rows(back, intensity_column="height", bands={"assigned": 0.75, "candidate": 0.45})
    return [r["provenance"]["engine_provenance"] for r in rows], summary


def test_publish_keeps_na_through_the_csv_and_derives_claims_on_the_scale(tmp_path):
    prov, summary = _published(_publish_ledger(["NA", "4a"], ["not assessed", "neutral"]), tmp_path)
    assert [p.get("evidence_level") for p in prov] == ["NA", "4a"]
    assert [p.get("claim") for p in prov] == ["not assessed", "neutral"]
    assert summary["levels_before_scale"] == 0
    prov, _ = _published(_publish_ledger(["3c", "4a", "4b", "reagent"], None), tmp_path)
    assert [p.get("claim") for p in prov] == ["identified", "neutral", "ion", "reagent"]


def test_publish_reads_an_older_scale_ledger_on_this_scale(tmp_path):
    led = _publish_ledger(["2b", "4a", "4d"], ["identified", "identified", "ion"], evidence_axes="iso")
    prov, summary = _published(led, tmp_path)
    assert summary["levels_before_scale"] == 3
    # every letter publishes as no level -- the shared 4a too: an old 4a is not a 4a of this scale
    assert [p.get("evidence_level") for p in prov] == [None, None, None]
    assert [p.get("claim") for p in prov] == ["tentative", "tentative", "tentative"]
    assert all("evidence_axes" not in p for p in prov)
