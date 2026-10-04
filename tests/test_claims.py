"""The CLAIM a level supports -- identified / neutral / ion / tentative, and
the two buckets beside them (reagent, not assessed).

`evidence.claim_class` reads a level of the evidence scale of peaky 0.10.0 as
what a reader may say about a committed formula: identified (level <= 3: 3c),
neutral (4a), ion (4b), tentative (5a, 5b, no level); the reagent bucket reads
"reagent" and NA "not assessed". Pins: the mapping and the tally; which rows carry a claim (every
committed M0 row per file, every merged row; never an isotope child or a
reagent ion); the batch wiring (merged ledger, batch_summary['claims'],
tables/evidence_levels.csv, the log line); the entry points that pass the
tallies on (the stage log, the CLI, the provenance counts, publish); and that
the claim is ADDITIVE -- nothing upstream reads it: not the tiers, not the
merge vote, not the level.

Run: pytest tests/test_claims.py -q
"""

from __future__ import annotations

import json
import types
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from peaky import cli
from peaky.assignment import assign as A
from peaky.assignment import evidence as EV
from peaky.assignment import ledger as L
from peaky.assignment import tiers as T
from peaky.batch import assign_batch as AB
from peaky.chem import chemistry as C
from peaky.io import io_mascope as IO
from peaky.io import publish as PUB


def _ledger() -> pd.DataFrame:
    """A finished single-sample ledger (the test_evidence_outputs fixture): one
    iso-confirmed M0 (A, with its 13C child B), one series M0 (C) and a reagent
    ion (E)."""
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


def _batch_table(spec, height=500.0):
    t0 = pd.Timestamp("2025-10-01 21:00:00", tz="UTC")
    rows = []
    for i, (sid, mzs) in enumerate(spec.items()):
        t = t0 + pd.Timedelta(minutes=10 * i)
        rows += [dict(sample_item_id=sid, sample_item_name=f"n_{sid}", datetime_utc=t,
                      mz=float(mz), height=height) for mz in mzs]
    return pd.DataFrame(rows)


ORBI = 120_000.0     # R(200) = 120 000: the scale assesses the run


def _orbi_inputs():
    return EV.file_run_inputs(sample_id="s1", reagent="NO3", context="ambient-air", resolution=ORBI,
                              height_gate_cps=100.0, degeneracy_cal=(0.0, 0.5))


def _run_batch(tmp_path, monkeypatch, *, corroborate=None, lines=None, resolving_power=ORBI):
    """assign_batch.run over eight synthetic files, every per-file assign
    returning the fixture ledger (no network)."""
    bg = list(range(100, 120))
    spec = {}
    for i in range(4):
        spec[f"a{i}"] = bg + list(range(200 + 20 * i, 220 + 20 * i))
        spec[f"b{i}"] = bg + list(range(200 + 20 * i, 220 + 20 * i))
    pk = _batch_table(spec)

    def fake_assign(sid, context="ambient-air", **kw):
        return {"ledger": _ledger(), "stats": {"noise_edge_cps": 4.0, "height_gate_cps": 10.0,
                                               "degeneracy_cal": {"mu": 0.0, "sigma": 0.5}},
                "plausibility_audit": [], "summaries": {}, "problems": []}

    monkeypatch.setattr(IO, "connect", lambda *a, **k: "CLIENT")
    monkeypatch.setattr(IO, "fetch_peaks", lambda client, sid, use_cache=True: pd.DataFrame(
        {"peak_id": ["A"], "mz": [200.1], "height": [1.0e5]}))
    monkeypatch.setattr(IO, "estimate_offset", lambda raw: 0.0)
    monkeypatch.setattr(A, "run", fake_assign)
    log = lines.append if lines is not None else (lambda *a: None)
    return AB.run(peaks=pk, ts_peaks=pk, reagent="Br", batch="test batch", out_dir=str(tmp_path),
                  k_min=2, k_max=3, min_gain=0.0, n_jobs=1, corroborate=corroborate,
                  resolving_power=resolving_power, log=log)


# --------------------------------------------------------------------------- the mapping
@pytest.mark.parametrize("level, claim", [
    ("1", "identified"), ("2", "identified"), ("3c", "identified"),
    ("4a", "neutral"), ("4b", "ion"),
    ("5a", "tentative"), ("5b", "tentative"),
    ("reagent", "reagent"), ("NA", "not assessed"),
    (None, "tentative"), (np.nan, "tentative"), (pd.NA, "tentative"), ("", "tentative"),
    ("nan", "tentative"), ("6", "tentative"), (" 4a ", "neutral"),
    # a level of the pre-0.10.0 decision is no level of the scale
    ("2b", "tentative"), ("3a", "tentative"), ("4c", "tentative"), ("4d", "tentative"),
])
def test_claim_class_reads_every_level(level, claim):
    assert EV.claim_class(level) == claim


def test_the_claim_sets_cover_the_scale_once():
    assert EV.CLAIMS == ("identified", "neutral", "ion", "tentative")
    assert EV.CLAIM_KEYS == EV.CLAIMS + ("reagent", "not assessed")
    sets = (EV.CLAIM_IDENTIFIED, EV.CLAIM_NEUTRAL, EV.CLAIM_ION)
    assert not any(a & b for i, a in enumerate(sets) for b in sets[i + 1:])
    assert EV.CLAIM_IDENTIFIED | EV.CLAIM_NEUTRAL | EV.CLAIM_ION == set(EV.LEVEL_ORDER) - {"5a", "5b"}
    assert set(EV.CLAIM_MEANING) == set(EV.CLAIM_KEYS)
    assert EV.COLUMNS[-1] == "claim" and EV.COLUMNS == (
        "evidence_level", "evidence", "would_lift", "competitors_left", "tags", "context", "context_source", "claim")
    assert EV.LEVELS == ["3c", "4a", "4b", "5a", "5b"] and EV.BUCKETS == ["reagent", "NA"]


def test_summarize_claims_keeps_zeros_in_order_and_skips_na():
    s = EV.summarize_claims(["ion", "ion", None, pd.NA, "tentative", "not assessed"])
    assert s == {"identified": 0, "neutral": 0, "ion": 2, "tentative": 1, "reagent": 0, "not assessed": 1}
    assert list(s) == list(EV.CLAIM_KEYS)
    assert EV.summarize_claims([]) == dict.fromkeys(EV.CLAIM_KEYS, 0)
    assert EV.summarize_claims(pd.Series(["identified"] * 3))["identified"] == 3


def test_summarize_counts_the_levels_and_the_buckets():
    assert EV.summarize(["4b", "NA", "reagent", "3c", None, "4b"]) == {"3c": 1, "4b": 2, "reagent": 1, "NA": 1}
    assert list(EV.summarize(["NA", "reagent", "5b", "3c"])) == ["3c", "5b", "reagent", "NA"]


# --------------------------------------------------------------------------- per file
def test_apply_levels_stamps_the_claim_on_m0_rows_only():
    led = _ledger()
    s = EV.apply_levels(led, run_inputs=_orbi_inputs())
    by = led.set_index("peak_id")
    for pid in ("A", "C"):
        assert by.loc[pid, "claim"] == EV.claim_class(by.loc[pid, "evidence_level"])
    # an isotope child and a reagent ion make no claim
    assert pd.isna(by.loc["B", "claim"]) and pd.isna(by.loc["E", "claim"])
    m0 = led[led["role"] == "M0"]
    assert s["claims"] == EV.summarize_claims(m0["claim"])
    assert sum(s["claims"].values()) == len(m0) == 2
    # a class-less file: every committed row claims "not assessed"
    led2 = _ledger()
    s2 = EV.apply_levels(led2)
    assert (led2.loc[led2["role"] == "M0", "claim"] == "not assessed").all() and s2["claims"]["not assessed"] == 2


def test_the_stage_log_line_carries_the_claims():
    led = _ledger()
    lines = []
    st = types.SimpleNamespace(led=led, cfg=None, corroborate=set(), resolving_power=None, log=lines.append)
    s = A._stage_evidence(st)
    line = next(ln for ln in lines if "evidence levels" in ln)
    c = s["claims"]
    assert line.endswith(f"; claims identified {c['identified']} | neutral {c['neutral']} | ion {c['ion']} | "
                         f"tentative {c['tentative']} | reagent {c['reagent']} | not assessed {c['not assessed']}")


# --------------------------------------------------------------------------- merged
def test_stamp_merged_gives_every_row_a_claim():
    pairs = pd.DataFrame({"neutral_formula": ["C10H16O4"], "adduct": ["[M-H]-"], "evidence_level": ["4b"],
                          "evidence": ["4b · x"], "would_lift": [""], "competitors_left": [""], "tags": [""],
                          "context": [""], "context_source": [""], "claim": ["ion"]})
    merged = pd.DataFrame([
        dict(mz=199.09, neutral_formula="C10H16O4", adduct="[M-H]-", tier="Assigned"),
        dict(mz=300.0, neutral_formula="C9H99O9", adduct="[M-H]-", tier="Candidate"),   # no per-file reading
    ])
    out = EV.stamp_merged(merged, pairs)
    assert list(out["claim"]) == ["ion", "tentative"] and pd.isna(out["evidence_level"].iloc[1])
    # an empty pair table: every row still carries a claim, tentative
    out0 = EV.stamp_merged(merged, pairs.iloc[0:0])
    assert list(out0["claim"]) == ["tentative", "tentative"]
    out_none = EV.stamp_merged(merged, None)
    assert list(out_none["claim"]) == ["tentative", "tentative"]
    # the claim is re-read off the joined level, whatever the pairs frame says
    out1 = EV.stamp_merged(merged, pairs.assign(claim="identified"))
    assert list(out1["claim"]) == ["ion", "tentative"]
    # an empty merged ledger keeps the column
    empty = EV.stamp_merged(merged.iloc[0:0], pairs)
    assert "claim" in empty.columns and len(empty) == 0


def test_claims_summary_counts_ion_only_rows_under_their_own_key():
    merged = pd.DataFrame({
        "neutral_formula": ["C10H16O4", "C7H12O4", "C8H12O4", "C10H16O4", "C5H8O3"],
        "adduct": ["[M-H]-", "[M-H]-", "[M-H]-", "[M]-.", "[M-H]-"],
        "tier": ["Assigned", "Candidate", "Assigned", "Candidate", "Assigned"],
        "evidence_level": ["3c", "4a", "5b", "4b", pd.NA],
        "ion_only_of": [pd.NA, pd.NA, pd.NA, "A", pd.NA],
        "stage": ["cover", "cover", "residual", "cover", "residual"],
    })
    merged["claim"] = merged["evidence_level"].map(EV.claim_class)
    levels = pd.DataFrame({"neutral_formula": ["C10H16O4"], "adduct": ["[M-H]-"],
                           "evidence_level": ["3c"], "claim": ["identified"]})
    s = AB._claims_summary(merged, levels)
    z = dict.fromkeys(EV.CLAIM_KEYS, 0)
    assert set(s) == {"merged", "pooled", "per_stage", "by_tier", "n_unlevelled"}
    assert s["merged"] == {**z, "identified": 1, "neutral": 1, "ion": 1, "tentative": 2}
    assert s["pooled"] == {**z, "identified": 1}
    assert s["per_stage"] == {"cover": {**z, "identified": 1, "neutral": 1, "ion": 1},
                              "residual": {**z, "tentative": 2}}
    # the ion-only row leaves its Candidate tier and counts on its own
    assert list(s["by_tier"]) == ["Assigned", "Candidate", "ion-only"]
    assert s["by_tier"]["Assigned"] == {**z, "identified": 1, "tentative": 2}
    assert s["by_tier"]["Candidate"] == {**z, "neutral": 1}
    assert s["by_tier"]["ion-only"] == {**z, "ion": 1}
    assert s["n_unlevelled"] == 1
    # tier and claim are not nested: a Candidate neutral and an Assigned tentative both stand
    assert s["by_tier"]["Candidate"]["neutral"] == 1 and s["by_tier"]["Assigned"]["tentative"] == 2
    # an empty merged ledger: zeros, no tiers, no stages
    e = AB._claims_summary(merged.iloc[0:0], levels.iloc[0:0])
    assert e == {"merged": z, "pooled": z, "per_stage": {}, "by_tier": {}, "n_unlevelled": 0}
    # no ion-only rows (or no link column): no 'ion-only' key
    assert "ion-only" not in AB._claims_summary(merged.drop(columns=["ion_only_of"]), levels)["by_tier"]


def test_batch_run_stamps_the_claim_and_tallies_it(tmp_path, monkeypatch):
    lines = []
    _run_batch(tmp_path, monkeypatch, lines=lines)
    merged = pd.read_csv(tmp_path / "merged_ledger.csv", keep_default_na=False)
    assert "claim" in merged.columns and (merged["claim"] != "").all()
    assert (merged["claim"] == merged["evidence_level"].map(EV.claim_class)).all()
    summ = json.load(open(tmp_path / "batch_summary.json"))
    # the block sits right after evidence_levels
    keys = list(summ)
    assert keys.index("claims") == keys.index("evidence_levels") + 1
    c = summ["claims"]
    assert set(c) == {"merged", "pooled", "per_stage", "by_tier", "n_unlevelled"}
    assert c["merged"] == EV.summarize_claims(merged["claim"]) and list(c["merged"]) == list(EV.CLAIM_KEYS)
    assert sum(c["merged"].values()) == len(merged)
    assert sum(c["pooled"].values()) == summ["evidence_levels"]["n_pairs"]
    assert sum(sum(v.values()) for v in c["by_tier"].values()) == len(merged)
    assert c["n_unlevelled"] == summ["evidence_levels"]["n_unstamped"] == 0
    pairs = pd.read_csv(tmp_path / "tables" / "evidence_levels.csv", keep_default_na=False)
    assert "claim" in pairs.columns
    assert (pairs["claim"] == pairs["evidence_level"].map(EV.claim_class)).all()
    # one log line of its own, never on the DONE line the progress panel parses
    m = c["merged"]
    assert ("[assign_batch] claims (merged): " + " | ".join(f"{k} {m[k]}" for k in EV.CLAIM_KEYS)) in lines
    assert not any("claims" in ln for ln in lines if ln.startswith("[assign_batch] DONE"))


def test_corroborate_moves_no_level_and_no_claim(tmp_path, monkeypatch):
    """--corroborate feeds the merge vote's class (and partners from an
    Orbitrap run dir); a ready cross set changes no level and no claim."""
    _run_batch(tmp_path / "bare", monkeypatch)
    _run_batch(tmp_path / "cross", monkeypatch, corroborate={"C10H16O4", "C7H12O4"})
    for rel in ("merged_ledger.csv", "tables/evidence_levels.csv"):
        a = pd.read_csv(tmp_path / "bare" / rel, keep_default_na=False)
        b = pd.read_csv(tmp_path / "cross" / rel, keep_default_na=False)
        pd.testing.assert_frame_equal(a, b)


def test_the_claim_changes_no_ion_tier_or_level(tmp_path, monkeypatch):
    """Nothing upstream reads the claim: with claim_class replaced by a constant
    the batch writes the same merged ledger and pairs table, claim aside."""
    _run_batch(tmp_path / "real", monkeypatch)
    monkeypatch.setattr(EV, "claim_class", lambda level: "tentative")
    _run_batch(tmp_path / "flat", monkeypatch)
    for rel in ("merged_ledger.csv", "tables/evidence_levels.csv", "tables/jitter.csv"):
        real = pd.read_csv(tmp_path / "real" / rel, keep_default_na=False)
        flat = pd.read_csv(tmp_path / "flat" / rel, keep_default_na=False)
        pd.testing.assert_frame_equal(real.drop(columns=["claim"], errors="ignore"),
                                      flat.drop(columns=["claim"], errors="ignore"))
    assert (pd.read_csv(tmp_path / "flat" / "merged_ledger.csv")["claim"] == "tentative").all()


# --------------------------------------------------------------------------- the vote
def test_the_vote_class_is_not_the_claim():
    """The vote's class is the private pre-0.10.0 decision's (neutral backed /
    formula confirmed / unconfirmed): a corroborated 5b reading votes as class 2
    while no level of the scale is read for it, and the scale's claim sets are
    not the vote's."""
    assert AB._evidence_class("5b", "corroborated") == 2 and AB._evidence_class("4c", "") == 1
    assert EV._VOTE_GOOD != EV.CLAIM_IDENTIFIED and EV._VOTE_MID != EV.CLAIM_ION
    assert not hasattr(AB, "EVIDENCE_CLASS_GOOD") and not hasattr(AB, "_level_text")


def test_the_claim_is_never_a_predicate_nor_a_vote_column():
    assert "claim" not in EV.PREDICATE_COLUMNS
    assert "claim" not in AB._M0_COLS
    # nor is any level column: the vote reads its own class
    assert not {"evidence_level", "evidence_axes"} & set(AB._M0_COLS) and "vote_class" in AB._M0_COLS


def _m0e(rows, claims=None):
    df = pd.DataFrame(rows, columns=["mz", "neutral_formula", "adduct", "tier", "ion_score",
                                     "evidence_level", "evidence_axes"])
    df["vote_class"] = [AB._evidence_class(a, b) for a, b in zip(df["evidence_level"], df["evidence_axes"])]
    df = df.drop(columns=["evidence_level", "evidence_axes"])
    if claims is not None:
        df["claim"] = claims
    return df


def test_a_contradicting_claim_column_does_not_move_the_vote():
    """A class-2 reading in 2 files beats a class-0 reading in 9 on its vote
    class; a claim column that says the opposite (the class-2 rows 'tentative',
    the class-0 rows 'identified') changes neither the winner nor the tier nor
    the note."""
    def files(claimed):
        out = {}
        for i in range(9):
            out[f"e{i:02d}"] = _m0e([(282.0853 + 1e-4 * i, "C14H21N", "[M+Br]-", "Candidate", 0.93, "5a", "")],
                                    ["identified"] if claimed else None)
        for i in range(2):
            out[f"f{i:02d}"] = _m0e([(282.0831 + 1e-4 * i, "C9H16O6", "[M+NO3]-", "Assigned", 0.95, "4a",
                                      "iso|chan2|corroborated|branch")],
                                    ["tentative"] if claimed else None)
        return out

    plain, jp = AB.align(files(False), tol_ppm=12.0)
    claimed, jc = AB.align(files(True), tol_ppm=12.0)
    assert plain.iloc[0]["neutral_formula"] == claimed.iloc[0]["neutral_formula"] == "C9H16O6"
    assert plain.iloc[0]["tier"] == claimed.iloc[0]["tier"] == "Assigned"
    assert plain.equals(claimed) and jp.equals(jc)
    assert "claim" not in claimed.columns


# --------------------------------------------------------------------------- entry points
def test_publish_carries_the_claims_beside_merged_tiers():
    assert "claims" in PUB.BATCH_CONFIG_KEYS
    assert PUB.BATCH_CONFIG_KEYS.index("claims") > PUB.BATCH_CONFIG_KEYS.index("merged_tiers")
    claims = {"merged": {"identified": 3, "ion": 2, "tentative": 1}}
    cfg = PUB.batch_config({"merged_tiers": {"Assigned": 5}, "claims": claims}, log=lambda *a: None)
    assert cfg["claims"] == claims and cfg["merged_tiers"] == {"Assigned": 5}


def test_publish_reads_the_claim_off_the_level_of_a_pre_claim_ledger():
    """A per-file ledger written without the claim column still carries its
    levels: its M0 rows publish the claim claim_class derives (the one a
    current run stamps), every other role none."""
    led = _ledger()
    EV.apply_levels(led, run_inputs=_orbi_inputs())
    old = led.drop(columns=["claim"])

    def claims(frame):
        rows, _ = PUB.build_rows(frame, intensity_column="height", bands=PUB.DEFAULT_TIER_BANDS)
        return {r["sample_peak_id"]: r["provenance"]["engine_provenance"].get("claim") for r in rows}

    got = claims(old)
    assert got == claims(led)
    assert got["A"] == EV.claim_class(led.loc[led["peak_id"] == "A", "evidence_level"].iloc[0])
    assert got["A"] in EV.CLAIM_KEYS and got["C"] in EV.CLAIM_KEYS
    assert got["B"] is None and got["E"] is None          # the 13C child and the reagent ion
    assert "claim" not in old.columns                      # the caller's frame is untouched
    # a ledger with neither column publishes no claim: nothing is invented
    assert set(claims(led.drop(columns=list(EV.COLUMNS))).values()) == {None}


def test_publish_batch_reads_the_claims_off_an_older_merged_ledger(tmp_path, capsys):
    """A batch_summary.json written before the 'claims' key: the published
    config's merged tally is read off the merged ledger's levels."""
    quiet = {"log": lambda *a: None}
    merged = pd.DataFrame({"evidence_level": ["3c", "4b", "5a", None]})
    derived = {"merged": {**dict.fromkeys(EV.CLAIM_KEYS, 0), "identified": 1, "ion": 1, "tentative": 2}}
    assert PUB.batch_config({"merged_tiers": {"Assigned": 4}}, merged=merged, **quiet)["claims"] == derived
    stamped = merged.assign(claim=["identified", "ion", "ion", "tentative"])
    assert PUB.batch_config({}, merged=stamped, **quiet)["claims"] == {
        "merged": {**dict.fromkeys(EV.CLAIM_KEYS, 0), "identified": 1, "ion": 2, "tentative": 1}}
    # a recorded tally wins; no ledger, or one with neither column, adds none
    recorded = {"merged": {"identified": 9, "ion": 0, "tentative": 0}}
    assert PUB.batch_config({"claims": recorded}, merged=merged, **quiet)["claims"] == recorded
    assert "claims" not in PUB.batch_config({"merged_tiers": {}}, **quiet)
    assert "claims" not in PUB.batch_config({}, merged=pd.DataFrame({"mz": [1.0]}), **quiet)

    # the CLI hands the merged ledger over (a dry run writes the payload)
    run = tmp_path / "run"
    run.mkdir()
    merged.assign(mz=[100.0, 200.0, 300.0, 400.0], neutral_formula=["C5H8O2"] * 4,
                  adduct=["[M-H]-"] * 4).to_csv(run / "merged_ledger.csv", index=False)
    (run / "batch_summary.json").write_text(json.dumps({"merged_tiers": {"Assigned": 4}}))
    out = tmp_path / "payload.json"
    cli.cmd_publish_batch(cli.build_parser().parse_args(
        ["publish-batch", str(run), "--dry-run", "--no-resolve-mechanisms", "--out", str(out)]))
    capsys.readouterr()
    assert json.loads(out.read_text())["config"]["claims"] == derived


def test_cli_prints_the_single_sample_claims(monkeypatch, tmp_path, capsys):
    from peaky.reporting import gka_widget as GW
    from peaky.reporting import report as R

    monkeypatch.setattr(R, "write_excel", lambda *a, **k: None)
    monkeypatch.setattr(R, "write_markdown", lambda *a, **k: None)
    monkeypatch.setattr(GW, "build_points", lambda *a, **k: [])
    monkeypatch.setattr(GW, "render_html", lambda *a, **k: "<html></html>")
    led = pd.DataFrame({"mz": [200.1], "height": [1e5], "role": ["M0"]})
    stats = {"by_role": {"M0": 1, "iso_child": 0, "reagent": 0, "unexplained": 0},
             "signal_by_role": {"M0": 1.0, "iso_child": 0.0, "reagent": 0.0},
             "count_frac_by_role": {"unexplained": 0.0}}
    args = SimpleNamespace(sample_id="S1", ppm=1.0)
    out = {"ledger": led, "stats": stats, "problems": [], "context": "ambient-air",
           "summaries": {"evidence": {"claims": {"identified": 1, "neutral": 0, "ion": 0, "tentative": 2,
                                                 "reagent": 0, "not assessed": 0}}}}
    cli._write_assign_outputs(args, out, str(tmp_path / "S1"))
    text = capsys.readouterr().out
    assert "claims: identified 1 | neutral 0 | ion 0 | tentative 2 | reagent 0 | not assessed 0" in text
    assert text.index("claims:") > text.index("unexplained 0")
    # a run with no evidence summary prints no claims line
    cli._write_assign_outputs(args, {k: v for k, v in out.items() if k != "summaries"}, str(tmp_path / "S2"))
    assert "claims:" not in capsys.readouterr().out


def test_cli_batch_prints_the_merged_claims(monkeypatch, capsys):
    from peaky import pipeline as PL

    merged = {"identified": 4, "neutral": 1, "ion": 2, "tentative": 7, "reagent": 0, "not assessed": 0}
    res = {"ctx": SimpleNamespace(out_dir="/tmp/peaky-test-run", run_id="rid"),
           "assign": {"summary": {"claims": {"merged": merged}}}}
    monkeypatch.setattr(cli, "_require_creds", lambda *a, **k: None)
    monkeypatch.setattr(PL, "run_batch", lambda **kw: res)
    cli.cmd_batch(cli.build_parser().parse_args(["batch", "--batch", "B"]))
    text = capsys.readouterr().out
    assert "claims (merged): identified 4 | neutral 1 | ion 2 | tentative 7 | reagent 0 | not assessed 0" in text
    assert text.index("claims (merged)") > text.index("[batch] done")
    # a summary without claims (an older assign) prints none
    monkeypatch.setattr(PL, "run_batch", lambda **kw: {"ctx": res["ctx"], "assign": {"summary": {}}})
    cli.cmd_batch(cli.build_parser().parse_args(["batch", "--batch", "B"]))
    assert "claims (merged)" not in capsys.readouterr().out


def test_cli_pool_prints_the_merged_claims(monkeypatch, capsys):
    from peaky import pipeline as PL

    merged = {"identified": 3, "ion": 5, "tentative": 1}
    res = {"ctx": SimpleNamespace(out_dir="/tmp/peaky-test-pool", run_id="rid"),
           "assign": {"summary": {"claims": {"merged": merged}}}}
    monkeypatch.setattr(cli, "_require_creds", lambda *a, **k: None)
    monkeypatch.setattr(PL, "run_pooled_batches", lambda **kw: res)
    cli.cmd_pool(cli.build_parser().parse_args(["pool", "--batches", "B.*"]))
    text = capsys.readouterr().out
    assert "claims (merged): identified 3 | ion 5 | tentative 1" in text
    assert text.index("claims (merged)") > text.index("[pool] unified ledger")
    monkeypatch.setattr(PL, "run_pooled_batches", lambda **kw: {"ctx": res["ctx"], "assign": {"summary": {}}})
    cli.cmd_pool(cli.build_parser().parse_args(["pool", "--batches", "B.*"]))
    assert "claims (merged)" not in capsys.readouterr().out


def test_the_provenance_counts_carry_the_merged_claims(monkeypatch, tmp_path):
    from peaky import pipeline as PL
    from peaky.reporting import provenance as PV

    class _Batches:
        def list(self, dataset=None):   # noqa: A001
            return pd.DataFrame({"sample_batch_id": ["B-id"], "sample_batch_name": ["B"]})

    merged = {"identified": 1, "ion": 2, "tentative": 3}
    got = {}
    ts = pd.DataFrame({"sample_item_id": ["s1", "s1", "s2", "s2"],
                       "mz": [100.0, 200.0, 100.0, 300.0], "height": [5.0] * 4,
                       "sample_batch_name": ["b1", "b1", "b2", "b2"]})
    monkeypatch.setattr(IO, "connect", lambda *a, **k: SimpleNamespace(batches=_Batches()))
    monkeypatch.setattr(AB, "run", lambda **kw: {"summary": {"claims": {"merged": merged}}, "sample_ids": []})
    monkeypatch.setattr(PL, "generate_report", lambda ctx, ts, **kw: {})
    monkeypatch.setattr(PV, "record_run", lambda **kw: got.__setitem__("rec", kw))
    PL.run_batch(batch="B", dataset="D", reagent="Br", base_out=str(tmp_path), ts=ts,
                 do_report=False, log=lambda *a: None)
    assert got["rec"]["counts"]["merged_claims"] == merged
    # a summary without the block records None, never raises
    monkeypatch.setattr(AB, "run", lambda **kw: {"summary": {}, "sample_ids": []})
    PL.run_batch(batch="B", dataset="D", reagent="Br", base_out=str(tmp_path / "old"), ts=ts,
                 do_report=False, log=lambda *a: None)
    assert got["rec"]["counts"]["merged_claims"] is None


def _wait(manager, jid, timeout=30.0):
    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if manager.get(jid).status in ("done", "error"):
            return manager.get(jid)
        time.sleep(0.02)
    raise AssertionError(f"job {jid} did not finish")


def test_the_mcp_tools_return_the_claims(monkeypatch, tmp_path):
    from peaky import mcp_server as M
    from peaky import pipeline as PL

    claims = {"identified": 1, "ion": 1, "tentative": 0}
    led = pd.DataFrame({"peak_id": ["p0", "p1", "p2"], "mz": [100.0, 200.0, 201.0],
                        "height": [9.0, 20.0, 2.0], "role": ["M0", "M0", "iso_child"],
                        "neutral_formula": ["C5H8O", "C10H16O4", pd.NA],
                        "adduct": ["[M-H]-", "[M-H]-", pd.NA],
                        "claim": ["identified", "ion", pd.NA]})
    monkeypatch.setattr(M, "JOBS", M.JobManager())
    monkeypatch.setattr(A, "run", lambda sample_id, context="ambient-air", **kw: {
        "ledger": led, "stats": {}, "summaries": {"evidence": {"claims": claims}}})
    job = _wait(M.JOBS, M.assign_sample("sid1", output_dir=str(tmp_path))["job_id"])
    assert job.status == "done", job.view()
    assert job.result["claims"] == claims
    assert [r["claim"] for r in job.result["top_species"]] == ["ion", "identified"]   # brightest first
    # a ledger without the column (and a run without the summary): no claim key, claims None
    monkeypatch.setattr(A, "run", lambda sample_id, context="ambient-air", **kw: {
        "ledger": led.drop(columns=["claim"]), "stats": {}})
    job = _wait(M.JOBS, M.assign_sample("sid1", output_dir=str(tmp_path))["job_id"])
    assert job.status == "done" and job.result["claims"] is None
    assert all("claim" not in r for r in job.result["top_species"])

    batch_claims = {"merged": {"identified": 2, "ion": 0, "tentative": 5}}
    monkeypatch.setattr(PL, "run_batch", lambda **kw: {
        "ctx": SimpleNamespace(out_dir=str(tmp_path), run_id="rid"),
        "assign": {"summary": {"claims": batch_claims}}})
    job = _wait(M.JOBS, M.run_batch("some batch", dataset="D")["job_id"])
    assert job.status == "done" and job.result["claims"] == batch_claims
    monkeypatch.setattr(PL, "run_batch", lambda **kw: {"ctx": None, "assign": {"merged_M0": 3}})
    job = _wait(M.JOBS, M.run_batch("some batch", dataset="D")["job_id"])
    assert job.status == "done" and job.result["claims"] is None
