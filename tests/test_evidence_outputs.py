"""The evidence stage's WIRING and OUTPUTS (B2) -- docs/EVIDENCE_LEVELS.md §6-§7.

The predicates and the goldens are pinned by tests/test_evidence.py; this file
pins where the stage runs, that a batch recomputes the level on the pooled
per-file ledgers and stamps the merged ledger, that every output carries the
four columns when the ledger does and renders unchanged when it does not, and
that `--corroborate` reaches both entry points.

Run: pytest tests/test_evidence_outputs.py -q
"""

from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path

import pandas as pd
import pytest

from peaky import cli
from peaky.assignment import assign as A
from peaky.assignment import evidence as EV
from peaky.assignment import ledger as L
from peaky.assignment import tiers as T
from peaky.io import publish as P
from peaky.reporting import pdf_report as PR
from peaky.reporting import report as R

ROOT = Path(__file__).resolve().parents[1]


def _ledger() -> pd.DataFrame:
    """A finished single-sample ledger in the real schema: one iso-confirmed M0
    (A, with its 13C child B), one series M0 (C) and a reagent ion (E)."""
    peaks = pd.DataFrame({"peak_id": ["A", "B", "C", "E"],
                          "mz": [200.1, 201.1035, 191.0, 78.9189],
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


# --------------------------------------------------------------------------- the stage
def test_the_stage_sits_after_every_tier_stage_and_before_timeseries():
    names = [s.name for s in A._STAGES]
    i = names.index("evidence")
    for earlier in ("tiers", "plausibility", "reflist_rescue", "iso_env_final"):
        assert names.index(earlier) < i, (earlier, names)
    assert names.index("timeseries") == i + 1
    stage = A._STAGES[i]
    assert stage.safe is False and stage.store is True


def test_apply_levels_on_a_real_ledger_writes_the_four_columns():
    led = _ledger()
    s = EV.apply_levels(led)
    by = led.set_index("peak_id")
    assert by.loc["A", "evidence_level"] == "4b"          # iso alone, nothing outside
    assert by.loc["C", "evidence_level"] in EV.LEVELS
    assert pd.isna(by.loc["B", "evidence_level"]) and pd.isna(by.loc["E", "evidence_level"])
    assert by.loc["A", "evidence_axes"].startswith("iso") and "carbon" in by.loc["A", "evidence_axes"]
    assert by.loc["A", "level_reason"].startswith("4b")
    assert pd.isna(by.loc["A", "n_plausible_structures"])   # not in the isomer space
    assert s["levels"].get("4b") == 1 and s["n_levelled"] == 2 and s["n_pairs"] == 2
    # the corroborated axis lifts it to 4a: two axes, one from outside the channel
    led2 = _ledger()
    EV.apply_levels(led2, cross={"C10H16O4"})
    assert led2.set_index("peak_id").loc["A", "evidence_level"] == "4a"


def test_the_stage_function_logs_and_returns_the_summary():
    led = _ledger()
    lines = []
    st = types.SimpleNamespace(led=led, cfg=None, corroborate=set(), log=lines.append)
    s = A._stage_evidence(st)
    assert s["levels"] and any("evidence levels" in ln for ln in lines)
    assert "evidence_level" in led.columns


# --------------------------------------------------------------------------- the batch
def test_level_pooled_pools_files_and_stamp_merged_joins_by_ion():
    a = EV.trim(_ledger())
    b = _ledger()
    L.commit_assignment(b, "C", neutral_formula="C7H12O4", adduct="[M+NO3]-",
                        ion_formula="C7H12NO7-", ion_score=0.8, compound_score=0.8,
                        ppm_error=0.2, pass_no=1, method="cheminfo", confidence="Good",
                        commentary="clustered", overwrite=True)
    pairs = EV.level_pooled({"f1": a, "f2": EV.trim(b)})
    lv = dict(zip(zip(pairs.neutral_formula, pairs.adduct), pairs.evidence_level))
    assert lv[("C10H16O4", "[M-H]-")] == "4b"
    assert lv[("C7H12O4", "[M-H]-")] == "3b" and lv[("C7H12O4", "[M+NO3]-")] == "3b"   # branch ACROSS files
    files = dict(zip(zip(pairs.neutral_formula, pairs.adduct), pairs.evidence_axes))
    assert files[("C10H16O4", "[M-H]-")].endswith("files:2")
    merged = pd.DataFrame([
        dict(mz=199.09, neutral_formula="C10H16O4", adduct="[M-H]-", tier="Assigned"),
        dict(mz=191.0, neutral_formula="C7H12O4", adduct="[M-H]-", tier="Assigned"),
        dict(mz=300.0, neutral_formula="C9H99O9", adduct="[M-H]-", tier="Candidate"),   # a re-read no file holds
    ])
    out = EV.stamp_merged(merged, pairs)
    assert list(out.evidence_level) [:2] == ["4b", "3b"] and pd.isna(out.evidence_level.iloc[2])
    assert out.index.equals(merged.index) and len(out) == 3
    assert str(out["n_plausible_structures"].dtype) == "Int64"
    # an empty pair table still yields the four (all-NA) columns
    out0 = EV.stamp_merged(merged, pairs.iloc[0:0])
    assert out0.evidence_level.isna().all() and set(EV.COLUMNS) <= set(out0.columns)


def _batch_table(spec, height=500.0):
    t0 = pd.Timestamp("2025-10-01 21:00:00", tz="UTC")
    rows = []
    for i, (sid, mzs) in enumerate(spec.items()):
        t = t0 + pd.Timedelta(minutes=10 * i)
        rows += [dict(sample_item_id=sid, sample_item_name=f"n_{sid}", datetime_utc=t,
                      mz=float(mz), height=height) for mz in mzs]
    return pd.DataFrame(rows)


@pytest.mark.parametrize("corroborate, expected", [(None, "4b"), ({"C10H16O4"}, "4a")])
def test_batch_run_recomputes_the_level_on_the_pooled_files(tmp_path, monkeypatch, corroborate, expected):
    """assign_batch.run: per-file ledgers pooled -> tables/evidence_levels.csv,
    merged_ledger.csv stamped by ion, batch_summary['evidence_levels'] filled,
    and the corroborating set reaches the per-file assign call."""
    from peaky.batch import assign_batch as AB
    from peaky.io import io_mascope as IO

    bg = list(range(100, 120))
    spec = {}
    for i in range(4):
        spec[f"a{i}"] = bg + list(range(200 + 20 * i, 220 + 20 * i))
        spec[f"b{i}"] = bg + list(range(200 + 20 * i, 220 + 20 * i))
    pk = _batch_table(spec)
    seen = []

    def fake_assign(sid, context="ambient-air", **kw):
        seen.append(kw)
        led = _ledger()
        return {"ledger": led, "stats": {"noise_edge_cps": 4.0, "height_gate_cps": 10.0},
                "plausibility_audit": [], "summaries": {}, "problems": []}

    monkeypatch.setattr(IO, "connect", lambda *a, **k: "CLIENT")
    monkeypatch.setattr(IO, "fetch_peaks", lambda client, sid, use_cache=True: pd.DataFrame(
        {"peak_id": ["A"], "mz": [200.1], "height": [1.0e5]}))
    monkeypatch.setattr(IO, "estimate_offset", lambda raw: 0.0)
    monkeypatch.setattr(A, "run", fake_assign)
    res = AB.run(peaks=pk, ts_peaks=pk, reagent="Br", batch="test batch", out_dir=str(tmp_path),
                 k_min=2, k_max=3, min_gain=0.0, n_jobs=1, corroborate=corroborate,
                 log=lambda *a: None)
    assert seen and all(kw.get("corroborate") == (corroborate or set()) for kw in seen)
    merged = pd.read_csv(tmp_path / "merged_ledger.csv")
    assert "evidence_level" in merged.columns
    row = merged[(merged.neutral_formula == "C10H16O4") & (merged.adduct == "[M-H]-")]
    assert len(row) == 1 and row.evidence_level.iloc[0] == expected
    assert row.evidence_axes.iloc[0].endswith(f"files:{len(seen)}")
    summ = json.load(open(tmp_path / "batch_summary.json"))
    ev = summ["evidence_levels"]
    assert ev["pooled"].get(expected) == 1 and ev["merged"].get(expected) == 1
    assert ev["n_pairs"] == 2 and ev["n_unstamped"] == 0
    assert ev["n_corroborate"] == (1 if corroborate else 0) and ev["cross_source"] == []
    pairs = pd.read_csv(tmp_path / "tables" / "evidence_levels.csv")
    assert set(EV.COLUMNS) <= set(pairs.columns) and len(pairs) == 2
    assert res["evidence"] is not None and len(res["evidence"]) == 2


def test_corroborating_neutrals_resolves_run_dirs_and_csvs(tmp_path):
    # the ledger holds C10H16O4 at 4b (its own 13C line) and C7H12O4 at 5a (the
    # series commit carries no axis the levels read): only the first is a sighting
    run = tmp_path / "run"; (run / "per_file").mkdir(parents=True)
    _ledger().to_csv(run / "per_file" / "s1_ledger.csv", index=False)
    assert EV.corroborating_neutrals([str(run)]) == {"C10H16O4"}
    assert EV.corroborating_neutrals([str(tmp_path)]) == {"C10H16O4"}   # the out-dir holding one run
    assert EV.corroborating_neutrals([str(run / "per_file" / "s1_ledger.csv")]) == {"C10H16O4"}   # a ledger CSV
    # a merged ledger has no predicate columns: its stored level, without its own `corroborated`
    merged = tmp_path / "m.csv"
    pd.DataFrame({"neutral_formula": ["HNO3", "C5H8O3"], "adduct": ["[M-H]-", "[M-H]-"],
                  "evidence_level": ["2b", "4b"], "evidence_axes": ["known:atmospheric|files:3", "corroborated|files:2"]}
                 ).to_csv(merged, index=False)
    assert EV.corroborating_neutrals([str(merged)]) == {"HNO3"}
    bare = tmp_path / "bare.csv"
    pd.DataFrame({"neutral_formula": ["HNO3"], "adduct": ["[M-H]-"]}).to_csv(bare, index=False)
    with pytest.raises(ValueError, match="evidence_level"):
        EV.corroborating_neutrals([str(bare)])      # nothing to judge the sighting by: say so, never guess
    assert EV.corroborating_neutrals([]) == set()
    with pytest.raises(FileNotFoundError):
        EV.corroborating_neutrals([str(tmp_path / "nowhere")])


# --------------------------------------------------------------------------- outputs
def test_build_sheets_without_levels_is_unchanged_and_with_levels_adds_them(tmp_path):
    led = _ledger()
    before = R.build_sheets(led, "ambient-air", sample_id="T")
    assert "By evidence level" not in before
    assert "evidence_level" not in before["Assigned"].columns
    assert "evidence_level" not in before["Peak ownership"].columns
    assert "evidence_level" not in before["Candidates"].columns
    assert "evidence_level" not in before["Target list"].columns
    assert not before["Summary"].section.eq("Evidence levels").any()

    EV.apply_levels(led)
    after = R.build_sheets(led, "ambient-air", sample_id="T")
    keys = list(after)
    assert keys.index("By evidence level") == keys.index("Candidates") + 1
    for sheet in ("Assigned", "Candidates", "Target list", "Peak ownership"):
        assert "evidence_level" in after[sheet].columns, sheet
    assert {"level_reason", "n_plausible_structures"} <= set(after["Assigned"].columns)
    assert set(after["Assigned"].evidence_level) == {"4b"} or "4b" in set(after["Assigned"].evidence_level)
    lv = after["By evidence level"]
    summ = lv[lv.section == "summary"]
    assert "4b" in set(summ.level) and int(summ.loc[summ.level == "4b", "n"].iloc[0]) == 1
    assert (lv.section.str.startswith("brightest")).any()
    assert after["Summary"].section.eq("Evidence levels").any()
    legend = after["Read me"]
    assert set(EV.LEVELS) <= set(legend.iloc[:, 1].astype(str))
    # the workbook writes and the new sheet is in the file
    path = tmp_path / "t.xlsx"
    R.write_excel(led, path, "ambient-air", sample_id="T")
    assert "By evidence level" in pd.ExcelFile(path).sheet_names


def _run_dir(d: Path, with_levels: bool) -> None:
    (d / "per_file").mkdir(parents=True, exist_ok=True)
    (d / "tables").mkdir(exist_ok=True)
    rows = [dict(mz=169.1223, neutral_formula="C10H16O2", adduct="[M+H]+", tier="Assigned",
                 ion_score=0.9, n_files=6, formula_agree=True),
            dict(mz=200.0, neutral_formula="C10H19NO2", adduct="[M+NH4]+", tier="Candidate",
                 ion_score=0.7, n_files=3, formula_agree=False)]
    if with_levels:
        for r, lv in zip(rows, ("4a", "5b")):
            r.update(evidence_level=lv, evidence_axes="iso|anchor|files:6" if lv == "4a" else "",
                     level_reason=f"{lv}: test", n_plausible_structures=None)
    pd.DataFrame(rows).to_csv(d / "merged_ledger.csv", index=False)
    pd.DataFrame([dict(peak_id=1, mz=169.1223, role="M0", neutral_formula="C10H16O2", adduct="[M+H]+",
                       height=10000, tier="Assigned", ppm_error=0.3, parent_peak_id=None)]
                 ).to_csv(d / "per_file" / "s1_ledger.csv", index=False)


class _Pdf:
    def __init__(self):
        self.pages = 0

    def savefig(self, fig):
        self.pages += 1


def test_pdf_page_renders_with_levels_and_is_skipped_without(tmp_path):
    assert PR.SECTIONS.index(PR.evidence_levels) == PR.SECTIONS.index(PR.coverage) + 1
    d = tmp_path / "with"; _run_dir(d, True)
    ctx = PR.load_context(str(d), tag="Ur", label="Ur")
    assert ctx["evidence_levels"] == {"4a": 1, "5b": 1}
    pdf = _Pdf(); PR.evidence_levels(ctx, pdf); assert pdf.pages == 1
    pdf = _Pdf(); PR.cover(ctx, pdf); assert pdf.pages == 1     # the cover line renders too
    out = PR.build(str(d), tag="Ur", label="Ur", out_pdf=str(d / "r.pdf"),
                   sections=[PR.cover, PR.evidence_levels])
    assert os.path.getsize(out) > 0
    d0 = tmp_path / "without"; _run_dir(d0, False)
    ctx0 = PR.load_context(str(d0), tag="Ur", label="Ur")
    assert "evidence_levels" not in ctx0
    pdf = _Pdf(); PR.evidence_levels(ctx0, pdf); assert pdf.pages == 0


def test_publish_provenance_carries_the_level_and_drops_na():
    row = pd.Series({"tier": "Assigned", "evidence_level": "4b", "evidence_axes": "iso",
                     "level_reason": "4b: one corroboration (iso)", "n_plausible_structures": pd.NA,
                     "confidence": "High"})
    out = P._engine_provenance(row, "Assigned")
    assert out["evidence_level"] == "4b" and out["evidence_axes"] == "iso"
    assert out["level_reason"].startswith("4b") and "n_plausible_structures" not in out
    for c in EV.COLUMNS:
        assert c in P._ENGINE_PROVENANCE_COLUMNS


def test_cli_corroborate_is_repeatable_on_assign_and_batch():
    p = cli.build_parser()
    a = p.parse_args(["assign", "--sample-id", "x", "--corroborate", "r1", "--corroborate", "r2"])
    assert a.corroborate == ["r1", "r2"]
    assert p.parse_args(["assign", "--sample-id", "x"]).corroborate == []
    b = p.parse_args(["batch", "--batch", "b", "--corroborate", "r1"])
    assert b.corroborate == ["r1"]


def test_scorecard_counts_only_the_four_axes_of_the_in_core_string():
    sys.path.insert(0, str(ROOT / "scripts"))
    import scorecard as SC  # noqa: E402
    led = pd.DataFrame({"neutral_formula": ["C10H16O4", "C7H12O4"], "adduct": ["[M-H]-", "[M-H]-"],
                        "evidence_level": ["4a", "5a"],
                        "evidence_axes": ["iso|anchor|corroborated|carbon|files:3", ""]})
    run = types.SimpleNamespace(ledger=led, per_file=pd.DataFrame(), name="r", path="")
    out = SC.levels_for(run, None, [])
    assert list(out.n_axes) == [3, 0] and list(out.level) == ["4a", "5a"] and set(out.source) == {"in-core"}
