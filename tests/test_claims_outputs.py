"""The claim classes (C13) in the user-facing outputs: the assignment workbook
(report.py) and the PDF report (pdf_report.py).

evidence.claim_class and the stage's stamping are pinned by the evidence tests;
this file pins that the workbook opens on a By claim sheet, that `claim` sits
directly before `evidence_level` wherever the level is shown, that a ledger
without levels renders unchanged, that an older ledger without the column gets
the claim read off its level, and that the PDF leads with a Claims page whose
signal shares carry an explicit `unmatched` bucket. The tier is shown beside
the claim everywhere and never read off it.

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
from peaky.reporting import pdf_report as PR
from peaky.reporting import report as R

# the level each committed peak is given after the stage runs: one row per claim
# and both directions of a tier / claim disagreement
LEVELS = {"A": "4a", "C": "5a", "D": "3a", "F": "4b"}


def _ledger(levels: bool = True) -> pd.DataFrame:
    """A finished single-sample ledger in the real schema: A (iso-confirmed, with
    its 13C child B), C (series), D (a near-tie: Candidate, with one alternative),
    F, a reagent ion E and an unexplained peak G. With `levels`, the evidence
    stage has run and the levels are set to LEVELS (claim re-read off each)."""
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
    return led


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
    assert list(summ.index) == list(EV.CLAIMS)
    assert summ.loc["identified", "n"] == 2 and summ.loc["ion", "n"] == 1 and summ.loc["tentative", "n"] == 1
    assert summ.loc["identified", "share"] == pytest.approx(0.5)
    h = 1.0e5 + 8.0e4 + 5.0e4 + 3.0e4
    assert summ.loc["identified", "signal"] == pytest.approx(1.5e5)          # A + D
    assert summ.loc["identified", "signal_share"] == pytest.approx(1.5e5 / h)
    assert summ.loc["tentative", "signal"] == pytest.approx(8.0e4)
    assert summ.loc["identified", "n_assigned"] == 1 and summ.loc["identified", "n_candidate"] == 1
    assert summ.loc["tentative", "n_assigned"] == 1 and summ.loc["identified", "n_ion_only"] == 0
    assert summ.loc["identified", "levels"] == "1 2a 2b 3a 3b 4a"
    assert summ.loc["ion", "levels"] == "4b 4c 4d"
    assert summ.loc["tentative", "levels"] == "5a 5b, no level"
    assert summ.loc["identified", "level_hist"] in ("4a: 1; 3a: 1", "3a: 1; 4a: 1")
    assert summ.loc["identified", "meaning"] == EV.CLAIM_MEANING["identified"]
    for c in EV.CLAIMS:
        assert (bc.section == f"brightest {c}").sum() == summ.loc[c, "n"]


def test_the_by_claim_sheet_lists_where_tier_and_claim_part_brightest_first():
    bc = R.build_sheets(_ledger())["By claim"]
    dis = bc[bc.section == "tier disagrees"]
    # Assigned but tentative (C, 8e4) before Candidate but identified (D, 5e4); A and F agree
    assert list(dis.peak_id) == ["C", "D"]
    assert list(zip(dis.tier, dis.claim, dis.evidence_level)) == [
        (T.TIER_ASSIGNED, "tentative", "5a"), (T.TIER_CANDIDATE, "identified", "3a")]
    assert dis[["mz", "height", "neutral_formula", "adduct", "evidence_axes", "level_reason"]].notna().all().all()


def test_claim_sits_directly_before_evidence_level_on_every_sheet():
    sheets = R.build_sheets(_ledger())
    carrying = [k for k, v in sheets.items() if "evidence_level" in v.columns]
    for name in ("Assigned", "Candidates", "Peak ownership", "Target list"):
        assert name in carrying, name
    for name in carrying:
        cols = list(sheets[name].columns)
        assert cols[cols.index("evidence_level") - 1] == "claim", (name, cols)


def test_the_claim_sits_on_committed_rows_only():
    sheets = R.build_sheets(_ledger())
    own = sheets["Peak ownership"].set_index("peak_id")
    for pid in ("B", "E", "G"):                     # isotope child, reagent ion, unexplained
        assert pd.isna(own.loc[pid, "claim"]), pid
    assert own.loc["A", "claim"] == "identified" and own.loc["F", "claim"] == "ion"
    cand = sheets["Candidates"]
    d = cand[cand.peak_id == "D"].set_index("rank")
    assert d.loc[1, "claim"] == "identified" and d.loc[2, "claim"] == ""      # the alternative makes none
    assert set(sheets["Assigned"].claim) == {"identified", "tentative", "ion"}
    uniq = sheets["Unique formulas"].set_index("neutral_formula")
    cols = list(sheets["Unique formulas"].columns)
    assert cols[cols.index("best_tier") + 1] == "best_claim"
    assert uniq.loc["C6H10O4", "best_claim"] == "identified"


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
    assert not sheets["Summary"].section.eq("Claims").any()
    assert not sheets["Read me"].section.eq("Claims").any()


def test_the_summary_sheet_puts_the_claims_between_coverage_and_tiers():
    ss = R.summary_stats(_ledger(), context="ambient-air")
    secs = list(dict.fromkeys(ss.section))
    assert secs.index("Coverage") < secs.index("Claims") == secs.index("Coverage") + 1
    assert secs.index("Claims") < secs.index("Tiers") < secs.index("Evidence levels")
    cl = ss[ss.section == "Claims"].set_index("metric")["value"]
    assert list(cl.index) == list(EV.CLAIMS)
    assert cl["identified"].startswith("2  (50% of assignments, 58% of assigned signal) -- ")
    assert cl["identified"].endswith(EV.CLAIM_MEANING["identified"])
    # the Tiers rows keep their denominators and their form
    assert ss[(ss.section == "Tiers") & (ss.metric == T.TIER_ASSIGNED)].value.iloc[0].startswith("3  (75% of")


def test_the_read_me_opens_on_the_claims():
    rm = R.build_sheets(_ledger())["Read me"]
    assert rm.section.iloc[0] == "Claims" and rm.topic.iloc[0] == "claim"
    block = rm[rm.section == "Claims"]
    assert set(EV.CLAIMS) <= set(block.topic)
    assert block.explanation.str.contains("n_identified|tier-Assigned").any()
    assert R.legend_sheet().section.iloc[0] == "Tiers"                         # the default is unchanged


def test_write_excel_round_trip_opens_on_by_claim_with_claim_chips(tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    path = tmp_path / "t.xlsx"
    R.write_excel(_ledger(), path, "ambient-air", sample_id="T")
    assert pd.ExcelFile(path).sheet_names[0] == "By claim"
    ws = openpyxl.load_workbook(path)["By claim"]
    head = [c.value for c in ws[1]]
    j = head.index("claim") + 1
    fills = {ws.cell(row=i, column=j).value: ws.cell(row=i, column=j).fill.start_color.rgb[-6:]
             for i in range(2, 5)}
    assert fills == {"identified": R._FILL["good"][0], "ion": R._FILL["info"][0],
                     "tentative": R._FILL["warn"][0]}


def test_markdown_carries_the_claims_line_and_the_row_tags(tmp_path):
    led = _ledger()
    result = {"ledger": led, "stats": L.stats(led), "sample_id": "T", "context": "ambient-air",
              "prescan": {}, "problems": []}
    result["stats"]["by_tier"] = led[led.role == L.ROLE_M0].tier.value_counts().to_dict()
    md = R.write_markdown(result, tmp_path / "t.md").read_text()
    lines = md.splitlines()
    ci = next(i for i, ln in enumerate(lines) if ln.startswith("- Claims:"))
    assert lines[ci] == "- Claims: identified 2 | ion 1 | tentative 1"
    assert lines[ci + 1].startswith("- Tiers:")
    assert f"[{T.TIER_ASSIGNED} · tentative]" in md and f"[{T.TIER_CANDIDATE} · identified]" in md
    plain = _ledger(levels=False)
    res0 = dict(result, ledger=plain, stats=L.stats(plain))
    md0 = R.write_markdown(res0, tmp_path / "t0.md").read_text()
    assert "- Claims:" not in md0 and " · " not in md0


# --------------------------------------------------------------------------- the PDF
MERGED = [  # neutral, adduct, tier, level, ion_only_of
    ("C10H16O4", "[M-H]-", "Assigned", "4a", None),
    ("C8H12O5", "[M-H]-", "Candidate", "3a", None),        # Candidate but identified
    ("C5H8O3", "[M-H]-", "Assigned", "4b", None),
    ("C9H14O4", "[M]-.", "Candidate", "4d", "p9"),         # an ion-only row
    ("C7H10O4", "[M-H]-", "Assigned", "5b", None),         # Assigned but tentative
    ("C10H19NO4", "[M+NO3]-", "Candidate", "5b", None),
    ("C6H6O3", "[M-H]-", "Candidate", None, None),         # a batch-level re-read: no level
]
HEIGHTS = {("C10H16O4", "[M-H]-"): 1000, ("C8H12O5", "[M-H]-"): 500, ("C5H8O3", "[M-H]-"): 300,
           ("C9H14O4", "[M]-."): 200, ("C7H10O4", "[M-H]-"): 100, ("C10H19NO4", "[M+NO3]-"): 100,
           ("C3H4O4", "[M-H]-"): 800}                        # the last is in no merged row
#: enough confirmed isotopes on the first channel to fill the appendix's isotopes field
ISO = ["13C", "13C2", "18O", "13C18O", "34S", "81Br", "37Cl", "15N", "29Si", "30Si", "2H"]


def _run_dir(d: Path, *, claim_col: bool = True, levels: bool = True, merged: list = MERGED) -> None:
    """A batch run dir: merged_ledger.csv (`merged` rows) + two per-file ledgers whose
    M0 heights total 3000 (800 of them in a reading no merged row carries)."""
    (d / "per_file").mkdir(parents=True, exist_ok=True)
    (d / "tables").mkdir(exist_ok=True)
    rows = []
    for k, (nf, ad, tier, lv, io) in enumerate(merged):
        r = dict(mz=150.0 + 10 * k, neutral_formula=nf, adduct=ad, tier=tier, ion_score=0.9 - 0.05 * k,
                 n_files=2, formula_agree=True, ion_only_of=io)
        if levels:
            r.update(evidence_level=lv, evidence_axes="", level_reason=f"{lv}: test",
                     n_plausible_structures=None)
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


def test_load_context_counts_the_claims_by_tier_and_by_signal(tmp_path):
    _run_dir(tmp_path)
    ctx = PR.load_context(str(tmp_path), tag="X", label="X")
    assert ctx["claims"] == {"identified": 2, "ion": 2, "tentative": 3}
    assert ctx["claim_by_tier"] == {
        "Assigned": {"identified": 1, "ion": 1, "tentative": 1},
        "Candidate": {"identified": 1, "ion": 0, "tentative": 2},
        "ion-only": {"identified": 0, "ion": 1, "tentative": 0}}       # kept out of Candidate
    assert ctx["claim_of_pair"][("C8H12O5", "[M-H]-")] == "identified"
    assert ctx["n_unlevelled"] == 1
    sig = ctx["claim_signal"]
    assert set(sig) == {*EV.CLAIMS, "unmatched"} and sum(sig.values()) == pytest.approx(1.0)
    assert sig["identified"] == pytest.approx(1500 / 3000) and sig["ion"] == pytest.approx(500 / 3000)
    assert sig["tentative"] == pytest.approx(200 / 3000) and sig["unmatched"] == pytest.approx(800 / 3000)
    assert ctx["claim_signal_assigned"] == pytest.approx(
        {"identified": 1000 / 3000, "ion": 300 / 3000, "tentative": 100 / 3000})
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
                         ion_score=0.50 + k / 1000, evidence_level="3a"))
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
    assert lines[k + 1] == ("b", "Claims: 2 identified · 2 ion · 3 tentative   (of 7 merged rows)")
    style, dim = lines[k + 2]
    assert style == "dim" and "identified carries 50% of the committed-peak signal" in dim
    assert "Claims page" in dim
    for t in (dim, lines[k + 3][1]):
        assert "more" not in t and "[cover]" not in t and "[residual]" not in t
    assert lines[k + 3] == ("dim", "   the other 27% is per-file signal with no merged row")
    assert lines[k + 4][1].startswith("Unique analytes assigned (M0):")     # the tier line stays
    assert any(t.startswith("Evidence levels (2b best .. 5b):") for _s, t in lines)
    _run_dir(tmp_path / "plain", levels=False)
    plain = _lines(PR.cover, PR.load_context(str(tmp_path / "plain"), tag="X", label="X"), monkeypatch)
    k0 = [t for _s, t in plain].index("Summary")
    assert plain[k0 + 1][1].startswith("Unique analytes assigned (M0):")


def test_findings_methods_and_the_appendix_carry_the_claim_only_when_levelled(tmp_path, monkeypatch):
    _run_dir(tmp_path)
    ctx = PR.load_context(str(tmp_path), tag="X", label="X")
    f = [t for s, t in _lines(PR.findings, ctx, monkeypatch) if s == "m"]
    assert f[0].split() == ["share", "class", "claim", "neutral"]
    assert any(t.split()[2] == "identified" and t.split()[3] == "C10H16O4" for t in f[1:])
    m = " ".join(t for _s, t in _lines(PR.methods, ctx, monkeypatch))
    assert "read from the evidence level alone" in m and "Claims page" in m
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
    # the claim column widens the row; the isotopes field keeps its 36 characters
    assert full[0].endswith("…") and len(full[0]) == len(full0[0]) + 11 and full[0][-36:] == full0[0][-36:]
    labs = {"13C", "18O", "34S", "13C2", "37Cl", "13C18O"}               # 33 characters joined
    ctx["iso_by_channel"][("C8H12O5", "[M-H]-")] = labs
    app = _lines(PR.assignments_table, ctx, monkeypatch)
    assert any(t.startswith("C8H12O5") and t.endswith("  13C, 18O, 34S, 13C2, 37Cl, 13C18O") for _s, t in app)
    assert "Claims page" not in " ".join(t for _s, t in _lines(PR.methods, ctx0, monkeypatch))
